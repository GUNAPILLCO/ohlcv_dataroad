"""Pruebas SINTETICAS de la enmienda E01 de TDA-17 (configs/tda17_enmienda_01.yaml).

1. Mecanismo de integridad (src/ohlcv_dataroad/ingest/tda17_enmiendas.py): protocolo base intacto, huella
   autorizada, aprobacion, lista blanca, configuracion efectiva trazable, rechazo sin vuelta al comportamiento
   anterior y verificacion posterior contra un commit Git.
2. E01-A, ADX con DX no definido: comparacion de las opciones A (literal, solo como referencia aqui) y B (motor),
   con un prototipo B independiente; contratos que empiezan con precios constantes.
3. E01-B, marca de sesion de VWAP: sin huecos, HI corto, HI largo, cambio de jornada y roll.
Sin datos de MNQ ni holdout. El registro del repositorio queda PENDIENTE_AUDITORIA: las pruebas que necesitan
E01 aprobada usan una copia temporal (``approved_amendment_root``).
"""
from __future__ import annotations

import inspect
import json
import math
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ohlcv_dataroad.ingest import tda17_enmiendas as enm
from ohlcv_dataroad.ingest import tda17_indicators as ind
from ohlcv_dataroad.ingest import tda17_primitives as pr
from ohlcv_dataroad.ingest import tda17_protocol_freeze as frz
from tda17_synthetic import (approve_registry, approved_amendment_root, clean_bars, commit_tree, git, make_frame,
                             rebuild_tda04)

ROOT = Path(__file__).resolve().parents[1]
E01_YAML = ROOT / "configs" / "tda17_enmienda_01.yaml"
BASE = ind.load_protocol()


@pytest.fixture()
def root(tmp_path):
    return approved_amendment_root(tmp_path / "repo")


def _registry(r):
    return json.loads((r / enm.REGISTRY_JSON).read_text(encoding="utf-8"))


def _write_registry(r, reg):
    (r / enm.REGISTRY_JSON).write_text(json.dumps(reg, indent=2), encoding="utf-8")


def _e01_doc():
    return yaml.safe_load(E01_YAML.read_text(encoding="utf-8"))


# =========================================================================== 1. mecanismo de integridad
def test_repository_registry_is_pending_or_verifiably_approved_and_never_falls_back():
    """El registro del repositorio esta PENDIENTE (produccion rechaza) o APROBADO con un commit auditado
    verificable (produccion activa exactamente E01). En ningun caso se calcula con el comportamiento anterior."""
    reg = _registry(ROOT)
    assert [e["id"] for e in reg["enmiendas"]] == ["E01"]
    entry = reg["enmiendas"][0]
    frame = make_frame(contracts=(("MNQ_A", 1),), bars_per_day=50).drop(columns=[ind.SYNTHETIC_MARKER])
    if entry["estado"] == enm.ESTADO_PENDIENTE:
        assert entry["aprobacion"] is None
        with pytest.raises(ind.ProtocolNotAuthorizedError, match="no aprobada"):
            enm.load_effective_protocol()
        with pytest.raises(ind.ProtocolNotAuthorizedError):
            ind.compute_catalog(frame)                                      # nunca calcula con la opcion A
    else:
        assert entry["estado"] == enm.ESTADO_APROBADA
        assert enm.verify_approval_against_git(ROOT, entry) == []           # commit auditado A verificable
        tr = enm.load_effective_protocol()["trazabilidad_protocolo"]
        assert tr["modo"] == "PRODUCCION" and tr["activable"]
        assert [(e["id"], e["sha256"]) for e in tr["enmiendas"]] == [("E01", enm.AUTHORIZED_AMENDMENTS["E01"]["sha256"])]
    with pytest.raises(ind.ProtocolNotAuthorizedError):
        ind._compute(frame, BASE)                                           # el protocolo base solo no basta


def test_authorized_fingerprint_matches_file_and_registry():
    digest = frz.sha256_file(E01_YAML)
    assert enm.AUTHORIZED_AMENDMENTS["E01"]["sha256"] == digest
    assert _registry(ROOT)["enmiendas"][0]["sha256"] == digest
    assert enm.FROZEN_MANIFEST_SHA256 == frz.sha256_file(ROOT / frz.MANIFEST_JSON)
    doc = _e01_doc()
    assert doc["enmienda"]["protocolo_base"] == _registry(ROOT)["protocolo_base"] == {
        "id": "TDA17_PROTOCOLO_v1", "commit": ind.FROZEN_COMMIT, "sha256_yaml": ind.FROZEN_YAML_SHA256,
        "sha256_manifiesto": enm.FROZEN_MANIFEST_SHA256}


def test_no_circular_fingerprints():
    reg_text = (ROOT / enm.REGISTRY_JSON).read_text(encoding="utf-8")
    reg_hash = frz.sha256_file(ROOT / enm.REGISTRY_JSON)
    mod_path = ROOT / "src" / "ohlcv_dataroad" / "ingest" / "tda17_enmiendas.py"
    mod_hash = frz.sha256_file(mod_path)
    e01_text = E01_YAML.read_text(encoding="utf-8")
    assert reg_hash not in e01_text and mod_hash not in e01_text            # E01 solo cita el protocolo base
    assert reg_hash not in mod_path.read_text(encoding="utf-8")             # el modulo no cita el registro
    assert frz.sha256_file(E01_YAML) not in e01_text
    reg = json.loads(reg_text)
    approvals = [e.pop("aprobacion") for e in reg["enmiendas"]]
    assert mod_hash not in json.dumps(reg)                                  # fuera de la aprobacion, nada cita el modulo
    for a in approvals:                                                     # la aprobacion cita lo auditado (sentido unico)
        assert a is None or a["sha256_auditados"][enm.IMPLEMENTATION_FILES[0]] == mod_hash
    assert "estado" not in _e01_doc()["enmienda"]                           # la aprobacion no cambia la huella


def test_approved_amendment_gives_traceable_effective_configuration(root):
    eff = enm.load_effective_protocol(root)
    tr = eff["trazabilidad_protocolo"]
    assert tr["base"]["commit"] == ind.FROZEN_COMMIT and tr["base"]["ficheros_protegidos_verificados"] == 9
    assert tr["base"]["sha256_yaml"] == ind.FROZEN_YAML_SHA256
    assert tr["enmiendas"][0]["id"] == "E01" and tr["enmiendas"][0]["estado"] == enm.ESTADO_APROBADA
    assert tr["enmiendas"][0]["sha256"] == enm.AUTHORIZED_AMENDMENTS["E01"]["sha256"]
    assert tr["sha256_registro"] == frz.sha256_file(root / enm.REGISTRY_JSON) and tr["activable"]
    body = {k: v for k, v in eff.items() if k != "trazabilidad_protocolo"}
    assert tr["sha256_efectiva"] == enm.canonical_sha256(body)
    assert enm.diff_paths(BASE, body) == {r for _, r in enm.WHITELIST["E01"]}
    assert frz.sha256_file(root / frz.PROTOCOL_YAML) == ind.FROZEN_YAML_SHA256   # el YAML original no cambia
    for el_b, el_e in zip(BASE["catalogo"], body["catalogo"]):                 # parametros, K y U intactos
        assert el_b["configuraciones"] == el_e["configuraciones"] and el_b.get("formula") == el_e.get("formula")
    for key in ("analisis", "g2", "decisiones", "recuentos", "identidades", "referencias_tda04"):
        assert body[key] == BASE[key]
    assert body["sensibilidades"]["S1"] == BASE["sensibilidades"]["S1"]
    assert body["sensibilidades"]["S2"] == BASE["sensibilidades"]["S2"]


def test_draft_document_is_traceable_but_not_activable():
    doc = enm.effective_document(ROOT, borrador=True)
    assert doc["trazabilidad_protocolo"]["activable"] is False and doc["trazabilidad_protocolo"]["modo"] == "BORRADOR"
    estado = _registry(ROOT)["enmiendas"][0]["estado"]
    assert doc["trazabilidad_protocolo"]["enmiendas"][0]["estado"] == estado
    if estado == enm.ESTADO_PENDIENTE:
        with pytest.raises(ind.ProtocolNotAuthorizedError):
            enm.effective_document(ROOT)


def _corrupt(root, case):
    reg = _registry(root)
    e = reg["enmiendas"][0]
    yaml_path = root / enm.AUTHORIZED_AMENDMENTS["E01"]["fichero"]
    if case == "registro_ausente":
        (root / enm.REGISTRY_JSON).unlink()
    elif case == "enmienda_ausente":
        yaml_path.unlink()
    elif case == "entrada_E01_ausente":
        reg["enmiendas"] = []
    elif case == "pendiente":
        e["estado"] = enm.ESTADO_PENDIENTE
    elif case == "aprobada_sin_responsable":
        e["aprobacion"]["aprobada_por"] = ""
    elif case == "estado_desconocido":
        e["estado"] = "ACTIVA"
    elif case == "enmienda_alterada":
        yaml_path.write_text(yaml_path.read_text(encoding="utf-8").replace("n + k(1/n) DX finitas", "n DX finitas", 1),
                             encoding="utf-8")
    elif case == "enmienda_y_registro_alterados":                     # el registro se edita de forma coherente
        yaml_path.write_text(yaml_path.read_text(encoding="utf-8") + "# cambio\n", encoding="utf-8")
        e["sha256"] = frz.sha256_file(yaml_path)
    elif case == "registro_con_otra_huella":
        e["sha256"] = "0" * 64
    elif case == "registro_otro_fichero":
        e["fichero"] = "configs/otra.yaml"
    elif case == "enmienda_no_autorizada":
        reg["enmiendas"].append({**e, "id": "E02"})
    elif case == "registro_otra_base":
        reg["protocolo_base"]["commit"] = "6245b5498e9380110df2cd6c51217d54d4f2cdd0"
    elif case == "fichero_protegido_alterado":
        p = root / "tests" / "test_tda17_supertrend.py"
        p.write_text(p.read_text(encoding="utf-8") + "\n# cambio\n", encoding="utf-8")
    elif case == "manifiesto_alterado":
        p = root / frz.MANIFEST_JSON
        p.write_text(p.read_text(encoding="utf-8").replace('"fecha_preparacion": "2026-10-16"', '"fecha_preparacion": "2026-10-17"'),
                     encoding="utf-8")
    elif case == "yaml_base_alterado":
        p = root / frz.PROTOCOL_YAML
        p.write_text(p.read_text(encoding="utf-8").replace("umbral_hueco_largo_min: 90", "umbral_hueco_largo_min: 120"),
                     encoding="utf-8")
    elif case == "fichero_protegido_ausente":
        (root / "tests" / "test_tda17_protocol.py").unlink()
    if case not in ("registro_ausente",):
        if (root / enm.REGISTRY_JSON).is_file():
            _write_registry(root, reg)


REJECTIONS = ["registro_ausente", "enmienda_ausente", "entrada_E01_ausente", "pendiente", "aprobada_sin_responsable",
              "estado_desconocido", "enmienda_alterada", "enmienda_y_registro_alterados", "registro_con_otra_huella",
              "registro_otro_fichero", "enmienda_no_autorizada", "registro_otra_base", "fichero_protegido_alterado",
              "manifiesto_alterado", "yaml_base_alterado", "fichero_protegido_ausente"]


@pytest.mark.parametrize("case", REJECTIONS)
def test_absent_unapproved_altered_or_incompatible_amendments_are_rejected(root, case):
    enm.load_effective_protocol(root)                                       # control: la copia aprobada carga
    _corrupt(root, case)
    with pytest.raises(ind.ProtocolNotAuthorizedError):
        enm.load_effective_protocol(root)


@pytest.mark.parametrize("approved", [False, True])
def test_authorized_fingerprint_barrier_rejects_altered_amendment_by_itself(tmp_path, approved):
    """La huella autorizada actua sola: con el registro PENDIENTE no interviene la aprobacion, y con el registro
    APROBADO rechaza antes que ella (motivo exacto)."""
    r = approved_amendment_root(tmp_path / "repo", approve=approved)
    _corrupt(r, "enmienda_alterada")
    with pytest.raises(ind.ProtocolNotAuthorizedError, match="no es la version autorizada"):
        enm.effective_document(r, borrador=True)


def test_draft_mode_still_rejects_altered_amendments(root):
    _corrupt(root, "enmienda_alterada")
    with pytest.raises(ind.ProtocolNotAuthorizedError):
        enm.effective_document(root, borrador=True)


WHITELIST_CASES = {                                 # cada caso debe rechazarse por SU barrera (motivo exacto)
    "parametro_A8": "fuera de la lista blanca", "umbral": "fuera de la lista blanca",
    "operacion_extra": "fuera de la lista blanca", "falta_operacion": "faltan o se repiten",
    "operacion_repetida": "faltan o se repiten", "anadir_sobre_existente": "fuera de la lista blanca",
    "sustituir_sin_cambio": "mismo valor que la base", "ruta_inexistente": "fuera de la lista blanca",
    "op_desconocida": "fuera de la lista blanca", "campo_extra": "mal formado"}


@pytest.mark.parametrize("case", list(WHITELIST_CASES))
def test_whitelist_rejects_any_other_change(case):
    doc = _e01_doc()
    ops = doc["cambios"]
    if case == "parametro_A8":
        ops.append({"op": "sustituir", "ruta": "catalogo.A8.configuraciones", "valor": []})
    elif case == "umbral":
        ops.append({"op": "sustituir", "ruta": "reglas_frontera.umbral_hueco_largo_min", "valor": 120})
    elif case == "operacion_extra":
        ops.append({"op": "anadir", "ruta": "catalogo.A9.nota", "valor": "x"})
    elif case == "falta_operacion":
        ops.pop()
    elif case == "operacion_repetida":
        ops.append(dict(ops[0]))
    elif case == "anadir_sobre_existente":
        ops[0] = {"op": "anadir", "ruta": "calentamiento.tres_niveles", "valor": "x"}
    elif case == "sustituir_sin_cambio":
        ops[2]["valor"] = BASE["calentamiento"]["tres_niveles"]
    elif case == "ruta_inexistente":
        ops[0]["ruta"] = "catalogo.A99.dx_no_definido"
    elif case == "op_desconocida":
        ops[0]["op"] = "eliminar"
    elif case == "campo_extra":
        ops[0]["nota"] = "x"
    with pytest.raises(ind.ProtocolNotAuthorizedError, match=WHITELIST_CASES[case]):
        enm.apply_amendment(BASE, doc, "E01")


@pytest.mark.parametrize("case", ["ruta_de_mas", "ruta_sin_cambio"])
def test_changed_paths_barrier_is_independent_of_the_operation_list(case):
    eff = enm.apply_amendment(BASE, _e01_doc(), "E01")
    allowed = {r for _, r in enm.WHITELIST["E01"]}
    enm.check_changed_paths(BASE, eff, allowed, "E01")                       # control
    if case == "ruta_de_mas":
        next(e for e in eff["catalogo"] if e["id"] == "A8")["configuraciones"][0]["params"]["n"] = 15
        reason = re.escape("de mas: ['catalogo.A8.configuraciones.n14.params.n']")
    else:
        eff["calentamiento"]["registro"] = BASE["calentamiento"]["registro"]
        reason = re.escape("sin cambio: ['calentamiento.registro']")
    with pytest.raises(ind.ProtocolNotAuthorizedError, match=reason):
        enm.check_changed_paths(BASE, eff, allowed, "E01")


@pytest.mark.parametrize("part", ["E01-A", "E01-B"])
def test_engine_requires_each_part_of_e01(part):
    eff = enm.apply_amendment(BASE, _e01_doc(), "E01")
    if part == "E01-A":
        del next(e for e in eff["catalogo"] if e["id"] == "A8")["dx_no_definido"]
    else:
        del eff["reglas_frontera"]["ventanas_marcas"]
    with pytest.raises(ind.ProtocolNotAuthorizedError, match=part):
        ind.require_e01(eff)


def test_whitelist_application_is_exact_and_base_is_not_mutated():
    before = json.dumps(BASE, sort_keys=True, default=str)
    eff = enm.apply_amendment(BASE, _e01_doc(), "E01")
    assert json.dumps(BASE, sort_keys=True, default=str) == before
    a8 = next(e for e in eff["catalogo"] if e["id"] == "A8")
    assert a8["dx_no_definido"]["regla"] == "OMITIR_EN_ETAPA_2"
    ind.require_e01(eff)
    with pytest.raises(ind.ProtocolNotAuthorizedError):
        ind.require_e01(BASE)


def test_engine_rejects_unknown_window_expressions():
    eff = enm.apply_amendment(BASE, _e01_doc(), "E01")
    eff["reglas_frontera"]["ventanas_marcas"]["asignacion"]["A1"] = {"niveles": "2n"}
    with pytest.raises(ind.ProtocolNotAuthorizedError):
        ind.mark_windows(eff)
    eff["reglas_frontera"]["ventanas_marcas"]["asignacion"]["A1"] = {"semanas": "n"}
    with pytest.raises(ind.ProtocolNotAuthorizedError):
        ind.mark_windows(eff)
    del eff["reglas_frontera"]["ventanas_marcas"]["asignacion"]["A1"]
    with pytest.raises(ind.ProtocolNotAuthorizedError):
        ind.mark_windows(eff)


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


GIT = pytest.mark.skipif(shutil.which("git") is None, reason="git no disponible")
CALC_LINE = "            self.tr = np.where(self.cons, conv_tr, self.h - self.l)             # TR segun P1"


def _git_repo_with_audit_and_publication(tmp_path):
    """A = commit auditado (registro PENDIENTE); P = commit publicado (registro APROBADO que cita A)."""
    r = approved_amendment_root(tmp_path / "repo", approve=False)
    audited = commit_tree(r, "A: version auditada")
    approve_registry(r, commit_auditado=audited)
    return r, audited, commit_tree(r, "P: aprobacion")


def test_f1_git_reference_covers_every_file_that_implements_or_verifies_e01():
    required = {"src/ohlcv_dataroad/ingest/tda17_indicators.py", "src/ohlcv_dataroad/ingest/tda17_primitives.py",
                "tests/test_tda17_indicators.py", "tests/test_tda17_enmienda_01.py", "tests/tda17_synthetic.py",
                "src/ohlcv_dataroad/ingest/tda17_enmiendas.py", "configs/tda17_enmienda_01.yaml", enm.REGISTRY_JSON}
    assert set(enm.AMENDMENT_TRACKED_FILES) == required
    assert set(enm.AUDITED_FILES) == required - {enm.REGISTRY_JSON}               # el registro no se cita a si mismo
    assert not set(enm.AMENDMENT_TRACKED_FILES) & set(frz.PROTOCOL_FILES)          # la base se verifica aparte


@GIT
def test_f1_reference_commit_accepts_the_published_tree(tmp_path):
    r, audited, published = _git_repo_with_audit_and_publication(tmp_path)
    assert enm.verify_against_git(r, published) == []
    entry = _registry(r)["enmiendas"][0]
    assert enm.verify_approval_against_git(r, entry) == []
    assert enm.load_effective_protocol(r)["trazabilidad_protocolo"]["activable"]
    errors = enm.verify_against_git(r, audited)                              # A: el registro aun era PENDIENTE
    assert errors == [f"[git {audited}] difiere del commit de referencia: {enm.REGISTRY_JSON}"]


@GIT
def test_f1_changing_a_calculation_line_in_the_engine_breaks_integrity(tmp_path):
    r, _, published = _git_repo_with_audit_and_publication(tmp_path)
    p = r / "src" / "ohlcv_dataroad" / "ingest" / "tda17_indicators.py"
    text = p.read_text(encoding="utf-8")
    assert text.count(CALC_LINE) == 1
    p.write_text(text.replace(CALC_LINE, CALC_LINE.replace("self.h - self.l)", "conv_tr)")), encoding="utf-8")
    assert enm.verify_against_git(r, published) == [
        f"[git {published}] difiere del commit de referencia: src/ohlcv_dataroad/ingest/tda17_indicators.py"]
    with pytest.raises(ind.ProtocolNotAuthorizedError, match="distintos de la version auditada"):
        enm.load_effective_protocol(r)                                        # tambien sin Git: huella aprobada


@GIT
@pytest.mark.parametrize("rel", ["src/ohlcv_dataroad/ingest/tda17_primitives.py", "tests/test_tda17_indicators.py",
                                 "tests/test_tda17_enmienda_01.py", "tests/tda17_synthetic.py",
                                 "src/ohlcv_dataroad/ingest/tda17_enmiendas.py", "configs/tda17_enmienda_01.yaml",
                                 "reports/mnq/TDA17_enmiendas.json"])
def test_f1_any_tracked_file_changed_after_the_reference_commit_is_detected(tmp_path, rel):
    r, _, published = _git_repo_with_audit_and_publication(tmp_path)
    p = r / rel
    p.write_text(p.read_text(encoding="utf-8") + ("\n" if rel.endswith(".json") else "\n# cambio posterior\n"),
                 encoding="utf-8")
    assert f"[git {published}] difiere del commit de referencia: {rel}" in enm.verify_against_git(r, published)


@GIT
def test_f1_unknown_reference_commit_is_never_conforming(tmp_path):
    r, _, _ = _git_repo_with_audit_and_publication(tmp_path)
    assert enm.verify_against_git(r, "0" * 40)


@GIT
def test_f2_approval_must_match_the_audited_commit(tmp_path):
    """El motor rechaza DIRECTAMENTE (via de produccion) una aprobacion cuyas huellas cuadran con el arbol pero
    cuyo commit auditado no existe o contiene otra version."""
    r, audited, _ = _git_repo_with_audit_and_publication(tmp_path)
    p = r / "tests" / "tda17_synthetic.py"
    p.write_text(p.read_text(encoding="utf-8") + "\n# cambio despues de la auditoria\n", encoding="utf-8")
    later = commit_tree(r, "cambio no auditado")
    approve_registry(r, commit_auditado=audited)                              # cita A; huellas del arbol actual
    with pytest.raises(ind.ProtocolNotAuthorizedError, match=re.escape(
            "[aprobacion E01] tests/tda17_synthetic.py del commit auditado difiere de la huella aprobada")):
        enm.load_effective_protocol(r)
    approve_registry(r, commit_auditado="0" * 40)                            # formato valido, commit inexistente
    with pytest.raises(ind.ProtocolNotAuthorizedError, match=f"el commit auditado {'0' * 40} no existe"):
        enm.load_effective_protocol(r)
    approve_registry(r, commit_auditado=later)                               # control: el commit que si contiene el arbol
    assert enm.load_effective_protocol(r)["trazabilidad_protocolo"]["activable"]


def _rmtree(path):
    def onerror(func, p, _exc):                                              # objetos de .git de solo lectura (Windows)
        os.chmod(p, stat.S_IWRITE)
        func(p)
    shutil.rmtree(path, onerror=onerror)


def _fake_git(fail_on, exc):
    real = subprocess.run

    def run(cmd, *a, **k):
        if cmd[0] == "git" and (fail_on is None or fail_on in cmd):
            raise exc
        return real(cmd, *a, **k)
    return run


def _git_case(tmp_path, monkeypatch, case):
    """Raiz con una aprobacion cuyas huellas cuadran con el arbol pero con un problema en el commit auditado."""
    if case == "fichero_ausente_en_A":
        r = approved_amendment_root(tmp_path / "repo", approve=False)
        aside = tmp_path / "aparte.py"
        shutil.move(r / "tests" / "tda17_synthetic.py", aside)
        audited = commit_tree(r)                                             # A sin el fichero
        shutil.move(aside, r / "tests" / "tda17_synthetic.py")
        approve_registry(r, commit_auditado=audited)
        return r
    if case == "repositorio_padre":                                          # root dentro de otro repositorio
        r = approved_amendment_root(tmp_path / "padre" / "repo", approve=False)
        approve_registry(r, commit_auditado=commit_tree(tmp_path / "padre"))
        return r
    r = approved_amendment_root(tmp_path / "repo")
    enm.load_effective_protocol(r)                                           # control: la aprobacion completa carga
    if case == "commit_inexistente":
        approve_registry(r, commit_auditado="1" * 40)
    elif case == "huella_distinta_en_A":
        p = r / "src" / "ohlcv_dataroad" / "ingest" / "tda17_primitives.py"
        audited = _registry(r)["enmiendas"][0]["aprobacion"]["referencia_auditoria"]["commit_auditado"]
        p.write_text(p.read_text(encoding="utf-8") + "\n# no auditado\n", encoding="utf-8")
        approve_registry(r, commit_auditado=audited)                         # huellas del arbol nuevo, commit A
    elif case == "commit_no_es_un_commit":
        approve_registry(r, commit_auditado=git(r, "rev-parse", "HEAD^{tree}").stdout.strip())
    elif case == "sin_repositorio_git":
        _rmtree(r / ".git")
    elif case == "git_no_ejecutable":
        monkeypatch.setattr(subprocess, "run", _fake_git(None, FileNotFoundError(2, "git no encontrado")))
    elif case == "git_agota_el_tiempo":
        monkeypatch.setattr(subprocess, "run", _fake_git("cat-file", subprocess.TimeoutExpired("git", 60)))
    elif case == "git_falla_al_leer":
        monkeypatch.setattr(subprocess, "run", _fake_git("show", PermissionError(13, "acceso denegado")))
    return r


GIT_REJECTIONS = {                                  # cada caso, por su motivo exacto
    "commit_inexistente": f"el commit auditado {'1' * 40} no existe",
    "commit_no_es_un_commit": "no existe en el repositorio",
    "fichero_ausente_en_A": "el commit auditado no contiene tests/tda17_synthetic.py",
    "huella_distinta_en_A": "tda17_primitives.py del commit auditado difiere de la huella aprobada",
    "sin_repositorio_git": "sin acceso al repositorio Git",
    "repositorio_padre": "no es la raiz del repositorio Git",
    "git_no_ejecutable": "sin acceso al repositorio Git .*FileNotFoundError",
    "git_agota_el_tiempo": "error de acceso a Git: TimeoutExpired",
    "git_falla_al_leer": "error de acceso a Git al leer .*PermissionError",
}


@GIT
@pytest.mark.parametrize("case", list(GIT_REJECTIONS))
def test_f2_production_requires_a_verifiable_audited_commit(tmp_path, monkeypatch, case):
    r = _git_case(tmp_path, monkeypatch, case)
    with pytest.raises(ind.ProtocolNotAuthorizedError, match="no corresponde a un commit auditado verificable") as exc:
        enm.load_effective_protocol(r)
    assert re.search(GIT_REJECTIONS[case], str(exc.value))
    with pytest.raises(ind.ProtocolNotAuthorizedError, match=GIT_REJECTIONS[case]):
        enm.effective_document(r, borrador=True)                            # tampoco se exporta como aprobada
    errors = enm.verify_approval_against_git(r, _registry(r)["enmiendas"][0])   # la misma barrera que ``verificar``
    assert errors and all(e.startswith("[aprobacion E01] ") for e in errors)
    assert any(re.search(GIT_REJECTIONS[case], e) for e in errors)


@GIT
def test_draft_of_an_approved_registry_cannot_activate_production(root):
    doc = enm.effective_document(root, borrador=True)
    assert doc["trazabilidad_protocolo"]["modo"] == "BORRADOR" and doc["trazabilidad_protocolo"]["activable"] is False
    prod = enm.load_effective_protocol(root)
    assert prod["trazabilidad_protocolo"]["modo"] == "PRODUCCION" and prod["trazabilidad_protocolo"]["activable"]
    assert doc["trazabilidad_protocolo"]["sha256_efectiva"] == prod["trazabilidad_protocolo"]["sha256_efectiva"]
    assert set(inspect.signature(ind.compute_catalog).parameters) == {"df"}     # el motor no acepta configuraciones
    assert set(inspect.signature(ind.build_panel_p).parameters) == {"series", "df"}


def _approval_case(r, case):
    reg = _registry(r)
    e = reg["enmiendas"][0]
    a = e["aprobacion"]
    ref = a["referencia_auditoria"]
    if case == "fecha_formato":
        a["fecha"] = "17/10/2026"
    elif case == "fecha_inexistente":
        a["fecha"] = "2026-02-30"
    elif case == "fecha_anterior_a_la_preparacion":
        a["fecha"] = "2026-10-16"
    elif case == "fecha_futura":
        a["fecha"] = "2099-01-01"
    elif case == "fecha_no_texto":
        a["fecha"] = 20261017
    elif case == "responsable_en_blanco":
        a["aprobada_por"] = "   "
    elif case == "referencia_texto_libre":
        a["referencia_auditoria"] = "auditoria ok"
    elif case == "documento_vacio":
        ref["documento"] = ""
    elif case == "commit_abreviado":
        ref["commit_auditado"] = "9cdea2f"
    elif case == "commit_no_hexadecimal":
        ref["commit_auditado"] = "z" * 40
    elif case == "sin_huellas_auditadas":
        del a["sha256_auditados"]
    elif case == "huellas_incompletas":
        del a["sha256_auditados"]["tests/tda17_synthetic.py"]
    elif case == "huella_distinta":
        a["sha256_auditados"]["src/ohlcv_dataroad/ingest/tda17_primitives.py"] = "0" * 64
    elif case == "campo_extra":
        a["firma"] = "x"
    elif case == "pendiente_con_aprobacion":
        e["estado"] = enm.ESTADO_PENDIENTE
    elif case == "fichero_auditado_modificado":
        q = r / "tests" / "test_tda17_indicators.py"
        q.write_text(q.read_text(encoding="utf-8") + "\n# cambio\n", encoding="utf-8")
    _write_registry(r, reg)


APPROVAL_CASES = ["fecha_formato", "fecha_inexistente", "fecha_anterior_a_la_preparacion", "fecha_futura",
                  "fecha_no_texto", "responsable_en_blanco", "referencia_texto_libre", "documento_vacio",
                  "commit_abreviado", "commit_no_hexadecimal", "sin_huellas_auditadas", "huellas_incompletas",
                  "huella_distinta", "campo_extra", "pendiente_con_aprobacion", "fichero_auditado_modificado"]


@pytest.mark.parametrize("case", APPROVAL_CASES)
def test_f2_incomplete_or_arbitrary_approval_is_rejected(root, case):
    enm.load_effective_protocol(root)                                       # control: aprobacion simulada completa
    _approval_case(root, case)
    with pytest.raises(ind.ProtocolNotAuthorizedError):
        enm.load_effective_protocol(root)
    with pytest.raises(ind.ProtocolNotAuthorizedError):
        enm.effective_document(root, borrador=True)                         # tampoco en borrador


def test_f2_registry_declares_it_is_not_a_cryptographic_signature():
    reg = _registry(ROOT)
    assert "no una firma criptografica" in reg["descripcion"]
    assert "no una firma criptografica" in enm.__doc__
    entry = reg["enmiendas"][0]
    if entry["estado"] == enm.ESTADO_PENDIENTE:
        assert entry["aprobacion"] is None
    else:                                                                   # declaracion completa, sin campo de firma
        assert entry["estado"] == enm.ESTADO_APROBADA and set(entry["aprobacion"]) == set(enm.APPROVAL_FIELDS)


def test_s2_cannot_run_without_a_scale_assignment(root):
    with pytest.raises(ind.ProtocolNotAuthorizedError, match="S2 no ejecutable"):
        enm.require_s2_scale_assignment(enm.load_effective_protocol(root))
    assert "sensibilidades.escalas_S2" not in {r for _, r in enm.WHITELIST["E01"]}   # E01 no la define


# =========================================================================== 2. E01-A: ADX (opciones A y B)
CFG_E = enm.apply_amendment(BASE, _e01_doc(), "E01")


def K_of(n):
    return next(c for e in CFG_E["catalogo"] if e["id"] == "A8" for c in e["configuraciones"] if c["params"]["n"] == n)["K"]


def stage1(b, n):
    """Terna y DX por actualizacion CONS con las primitivas del motor (+DI, -DI y DX no cambian con E01)."""
    alpha = pr.wilder_alpha(n)
    idx = np.flatnonzero(b.cons)
    up, dn = b.h[idx] - b.h_prev[idx], b.l_prev[idx] - b.l[idx]
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    w_tr, w_p, w_m = (pr.seeded_smoother(x, n, alpha) for x in (b.tr[idx], pdm, mdm))
    pdi, mdi = 100 * pr.safe_div(w_p, w_tr), 100 * pr.safe_div(w_m, w_tr)
    dx = 100 * pr.safe_div(np.abs(pdi - mdi), pdi + mdi)
    return idx, pdi, mdi, dx


def _at_bars(b, n, upd):
    U = 2 * (n + pr.k_weight(1 / n)) - 1
    elig = pr.cons_counter(b.cons) >= U                                     # elegibilidad estructural sin cambios
    return np.where(elig, pr.values_at_bars(upd, b.cons), np.nan), elig


def adx_option_a(b, n):
    """Opcion A (lectura literal, ya NO es la del motor): una DX no definida en la semilla o despues propaga NaN."""
    u1 = n + pr.k_weight(1 / n)
    idx, pdi, mdi, dx = stage1(b, n)
    upd = np.full(idx.size, np.nan)
    if idx.size >= u1 + n - 1:
        x = dx[u1 - 1:]
        state = float(np.mean(x[:n]))
        upd[u1 - 1 + n - 1] = state
        for i in range(n, x.size):
            state = state + (x[i] - state) / n
            upd[u1 - 1 + i] = state
    return {"di_pos": _at_bars(b, n, pdi)[0], "di_neg": _at_bars(b, n, mdi)[0], "adx": _at_bars(b, n, upd)[0]}


def adx_option_b(b, n):
    """Prototipo INDEPENDIENTE (bucle) de la opcion B aprobada (E01-A)."""
    alpha = pr.wilder_alpha(n)
    u1 = n + pr.k_weight(alpha)
    idx, pdi, mdi, dx = stage1(b, n)
    upd = np.full(idx.size, np.nan)
    seed, state, cnt = [], None, 0
    for j in range(u1 - 1, idx.size):
        x = dx[j]
        if np.isfinite(x):
            cnt += 1
            if state is None:
                seed.append(x)
                if len(seed) == n:
                    state = sum(seed) / n
            else:
                state = state + alpha * (x - state)
        if state is not None and cnt >= n + pr.k_weight(alpha):
            upd[j] = state                                              # activado; se retiene si DX no es finito
    adx, elig = _at_bars(b, n, upd)
    return {"di_pos": _at_bars(b, n, pdi)[0], "di_neg": _at_bars(b, n, mdi)[0], "adx": adx, "elig": elig}


def bars(flat=0, n_bars=1200, seed=3, gaps=()):
    o, h, lo, c, v, cons, nd = clean_bars(n_bars, seed=seed)
    c[:flat], o[:flat], h[:flat], lo[:flat] = 15000.0, 15000.0, 15000.5, 14999.5
    cons = cons.copy()
    cons[list(gaps)] = False
    return ind.Bars(o=o, h=h, l=lo, c=c, v=v, cons=cons, new_day=nd)


def first(x):
    f = np.flatnonzero(np.isfinite(x))
    return int(f[0]) + 1 if f.size else None


@pytest.mark.parametrize("n", [14, 60])
@pytest.mark.parametrize("seed", range(6))
def test_b_equals_a_when_every_dx_is_finite(n, seed):
    b = bars(0, K_of(n) + 600, seed=seed, gaps=(30, 31, 200, 400))
    eng, a, proto = ind.a8_adx(b, {"n": n}), adx_option_a(b, n), adx_option_b(b, n)
    for k in ("di_pos", "di_neg"):
        np.testing.assert_array_equal(eng[k], a[k])                         # E01 no cambia +DI ni -DI
        np.testing.assert_array_equal(eng[k], proto[k])
    np.testing.assert_allclose(eng["adx"], a["adx"], rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(eng["adx"], proto["adx"], rtol=1e-12, equal_nan=True)
    E = first(eng["adx"])
    assert E == first(a["adx"]) == K_of(n) + sum(g + 1 <= E for g in (30, 31, 200, 400))   # E = K + huecos en 2..E


@pytest.mark.parametrize("n", [14, 60])
def test_dx_formula_is_unchanged(n):
    b = bars(n + 40, K_of(n) + 300)
    _, pdi, mdi, dx = stage1(b, n)
    ok = np.isfinite(pdi) & np.isfinite(mdi)
    expected = np.where(ok & (pdi + mdi != 0), 100 * np.abs(pdi - mdi) / np.where(pdi + mdi != 0, pdi + mdi, 1), np.nan)
    np.testing.assert_array_equal(np.isnan(dx), np.isnan(expected))
    np.testing.assert_allclose(dx[np.isfinite(dx)], expected[np.isfinite(dx)], rtol=1e-15)
    assert np.isnan(dx).any() and np.isfinite(dx).any()                     # hay DX no definidos (0/0), nunca 0
    eng = ind.a8_adx(b, {"n": n})
    np.testing.assert_array_equal(eng["di_pos"], _at_bars(b, n, pdi)[0])


FLATS = [0, 20, 80, 110, 300, 700]


@pytest.mark.parametrize("n", [14, 60])
@pytest.mark.parametrize("flat", FLATS)
def test_contracts_starting_with_constant_prices(n, flat):
    K = K_of(n)
    b = bars(flat, K + flat + 800)
    eng, a, proto = ind.a8_adx(b, {"n": n}), adx_option_a(b, n), adx_option_b(b, n)
    np.testing.assert_array_equal(eng["di_pos"], a["di_pos"])
    np.testing.assert_allclose(eng["adx"], proto["adx"], rtol=1e-12, equal_nan=True)   # motor = prototipo B
    assert int(np.flatnonzero(proto["elig"])[0]) + 1 == K                   # elegibilidad estructural: siempre K
    fa, fb = first(a["adx"]), first(eng["adx"])
    assert fb is not None and fb >= K                                       # con suficientes DX finitas, B publica
    idx, _, _, dx = stage1(b, n)
    u1 = n + pr.k_weight(1 / n)
    fin = np.flatnonzero(np.isfinite(dx[u1 - 1:]))
    assert fb == max(K, idx[(u1 - 1) + fin[n + pr.k_weight(1 / n) - 1]] + 1)   # activacion: n + k DX finitas
    if np.isnan(dx[u1 - 1:u1 - 1 + n]).any():
        assert fa is None                                                   # A: ADX NaN durante todo el contrato
    else:
        assert fa == fb


def test_b_retains_state_when_dx_undefined_after_activation():
    n, alpha = 14, 1 / 14
    dx = np.array([10.0] * (n + pr.k_weight(alpha)) + [np.nan, np.nan, 30.0])
    y = ind.wilder_finite_inputs(dx, n, alpha)
    assert y[-3] == y[-4] == y[-2] == 10.0                                  # B retiene sin decaimiento
    assert math.isclose(y[-1], 10.0 + alpha * 20.0)
    a = np.full(dx.size, np.nan)
    state = dx[:n].mean()
    for i in range(n, dx.size):
        state = state + alpha * (dx[i] - state)
        a[i] = state
    assert np.isnan(a[-1]) and np.isnan(a[-2])                              # A propaga el NaN


def test_roll_resets_both_options():
    n = 14
    K = K_of(n)
    b2 = bars(0, K + 300, seed=5)
    for f in (lambda b: ind.a8_adx(b, {"n": n}), lambda b: adx_option_a(b, n)):
        r2 = f(b2)
        assert first(r2["adx"]) == K and np.isnan(r2["adx"][:K - 1]).all()


# =========================================================================== 3. E01-B: marca de sesion de VWAP
def _skeleton(kinds, long_at=()):
    tipo = np.array(kinds, dtype=object)
    long_hi = np.zeros(tipo.size, bool)
    long_hi[list(long_at)] = True
    return tipo, long_hi


def test_vwap_session_mark_without_gaps():
    tipo, long_hi = _skeleton(["REINICIO"] + ["CONS"] * 30)
    m = ind.session_marks(tipo, long_hi)
    assert all((m[k] == 0).all() for k in ind.MARK_KEYS)                    # definida desde la barra 1 (K = 1)


@pytest.mark.parametrize("long_gap", [False, True])
def test_vwap_session_mark_with_internal_gap_is_cumulative(long_gap):
    kinds = ["REINICIO"] + ["CONS"] * 9 + ["HI"] + ["CONS"] * 9 + ["HI"] + ["CONS"] * 5
    tipo, long_hi = _skeleton(kinds, long_at=(10,) if long_gap else ())
    m = ind.session_marks(tipo, long_hi)
    exp_hi = np.r_[np.zeros(10), np.ones(10), np.full(6, 2)]
    np.testing.assert_array_equal(m["n_disc_HI"], exp_hi)                  # acumula desde la apertura hasta t
    np.testing.assert_array_equal(m["cruza_hueco_largo"], (np.arange(26) >= 10) & long_gap)
    assert (m["n_disc_FJ"] == 0).all() and (m["n_disc_FJL"] == 0).all()


@pytest.mark.parametrize("opening", ["FJ", "FJL", "REINICIO"])
def test_vwap_session_mark_resets_with_vwap_and_ignores_the_opening_pair(opening):
    kinds = ["REINICIO"] + ["CONS"] * 4 + ["HI"] + ["CONS"] * 4 + [opening] + ["CONS"] * 3 + ["HI"] + ["CONS"] * 3
    tipo, long_hi = _skeleton(kinds, long_at=(5,))
    m = ind.session_marks(tipo, long_hi)
    np.testing.assert_array_equal(m["n_disc_HI"], np.r_[np.zeros(5), np.ones(5), np.zeros(4), np.ones(4)])
    np.testing.assert_array_equal(m["cruza_hueco_largo"], np.r_[np.zeros(5), np.ones(5), np.zeros(8)])   # no hereda
    assert m["n_disc_FJ"][10] == 0 and m["n_disc_FJL"][10] == 0             # la apertura no es contaminacion


@pytest.fixture(scope="module")
def approved(tmp_path_factory):
    r = approved_amendment_root(tmp_path_factory.mktemp("e01"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(enm, "ROOT", r)
        yield r


def _session_reference(df, tipo):
    """Bucle independiente: HI desde la primera barra de (contrato, trading_date) hasta t; largo si delta > 90."""
    out = {k: np.zeros(len(df)) for k in ind.MARK_KEYS}
    key = list(zip(df["contract"], df["trading_date"]))
    delta = df["delta_minutes"].to_numpy(float)
    hi = n_long = 0
    for t in range(len(df)):
        if t == 0 or key[t] != key[t - 1]:
            hi = n_long = 0
        elif tipo[t] == "HI":
            hi += 1
            n_long += delta[t] > 90
        out["n_disc_HI"][t] = hi
        out["cruza_hueco_largo"][t] = float(n_long > 0)
    return out


def test_vwap_session_marks_on_full_frame_with_short_and_long_gaps_day_changes_and_roll(approved):
    df = make_frame(contracts=(("MNQ_A", 3), ("MNQ_B", 3)), bars_per_day=400, hi_every=37, long_hi_every=151, fjl_every=2)
    s, m = ind.compute_catalog(df.drop(columns=[ind.SYNTHETIC_MARKER]))
    tipo = m["tipo_par"].to_numpy()
    assert {"HI", "FJ", "FJL", "REINICIO"} <= set(tipo) and (df["delta_minutes"][tipo == "HI"] > 90).any()
    ref = _session_reference(df, tipo)
    for k in ind.MARK_KEYS:
        np.testing.assert_array_equal(m[ind.mark_column("sesion", None, k)].to_numpy(), ref[k])
    spec = ind.mark_windows(enm.load_effective_protocol())["D8_sesion__dist"]
    series = ind.series_discontinuity(m, spec)
    np.testing.assert_array_equal(series["n_disc_HI"], ref["n_disc_HI"])     # marca de la serie VWAP para S1/S2
    opens = np.isin(tipo, ["FJ", "FJL", "REINICIO"])
    assert (series["n_disc_HI"][opens] == 0).all() and (series["cruza_hueco_largo"][opens] == 0).all()
    assert np.isfinite(s["D8_sesion__dist"].to_numpy()[opens]).all()        # VWAP reinicia: ln(C/TP) en la apertura


def test_series_without_window_have_zero_marks_and_rvol_is_out_of_scope(approved):
    eff = enm.load_effective_protocol()
    spec = ind.mark_windows(eff)
    assert not any(k.startswith("D7_") for k in spec)                      # RVOL: solo S7 (rvol_cruza_roll)
    m = pd.DataFrame(index=range(5))
    for sid in ("E1_barra__valor", "E5_barra__valor"):
        assert all((v == 0).all() for v in ind.series_discontinuity(m, spec[sid]).values())
    rules = eff["sensibilidades"]["marcas_por_serie"]
    assert "escalas S y M" in rules["S2"] and "cualquiera de las series" in rules["varias_series"]
    assert "MEMORIA EFECTIVA" in eff["reglas_frontera"]["ventanas_marcas"]["memoria_efectiva"]


def test_vwap_s2_mark_is_not_contaminated_by_previous_session_gap():
    """Contraejemplo de la regla v2 (VWAP sin ventana): un HI largo al final de la jornada d no debe marcar la
    jornada d+1, y si debe marcar el resto de la jornada d (VWAP acumula a traves del HI)."""
    df = make_frame(contracts=(("MNQ_A", 2),), bars_per_day=300, hi_every=0, long_hi_every=0, dense_gaps=None, fjl_every=0)
    ts = df["timestamp"].copy()
    ts.iloc[250:] = ts.iloc[250:] + pd.Timedelta(minutes=120)               # HI largo en el par (249, 250), jornada 1
    df = rebuild_tda04(df.assign(timestamp=ts)[["timestamp", "source_file", "contract", "trading_date", "segment_id",
                                               "open", "high", "low", "close", "volume"]])
    tipo = ind.pair_types(df)
    long_hi = df["delta_minutes"].to_numpy(float) > ind.LONG_GAP_MIN
    m = ind.session_marks(tipo, long_hi)
    assert tipo[250] == "HI" and long_hi[250]
    day2 = int(np.flatnonzero(df["trading_date"].to_numpy() != df["trading_date"].iloc[0])[0])
    assert (m["cruza_hueco_largo"][250:day2] == 1).all() and (m["cruza_hueco_largo"][day2:] == 0).all()
