"""Gobernanza de E03 (``ohlcv_dataroad.tda17_e03.enmienda``): registro, aprobacion A3, publicacion P3 y ruta oficial.

Las garantias son las de E02 aplicadas a E03. Todas SINTETICAS. Las aprobaciones son SIMULADAS ("PRUEBA_SINTETICA") y
viven solo en un REPOSITORIO GIT TEMPORAL: clon del proyecto en la publicacion REAL P2 de E02 (E01 y E02 tal como se
publicaron), ficheros de E03 del arbol de trabajo, commit A3 (registro PENDIENTE), aprobacion que cita A3, commit P3 y
un remoto 'origin' TEMPORAL (repositorio desnudo) con P3 publicado; ``m2.AUTHORIZED_REMOTE`` se fija a ese remoto SOLO
en la prueba (como en las pruebas de E02). El remoto del clon se elimina antes de nada: ninguna prueba puede publicar
en el repositorio del proyecto. Las entradas analizadas son SINTETICAS (``data/sintetico``); la verificacion estricta
del protocolo de E02 comprueba por SHA-256, como en las pruebas de E02, los ficheros de datos del manifiesto
(enlazados, nunca escritos ni cargados). Sin holdout ni ``data/raw``.

Registro del PROYECTO (R4, como E02 v11): admite exactamente PENDIENTE_AUDITORIA y APROBADA (``check_project_state``); el
mismo control se ejerce en el repositorio temporal en ambos estados y frente a aprobaciones falsificadas.

Negativas: tipo EXACTO y mensaje COMPLETO (``assert_rejects`` de las pruebas de E02). Los controles que solo se
alcanzan tras otra barrera (E03_DOC, E03_P3_APROBADA) se prueban fijando con ``monkeypatch`` la constante o la funcion
de la barrera anterior.
"""
from __future__ import annotations

import copy
import inspect
import json
import os
import shutil
import sys
import threading
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))             # tests/ (ayudantes de E01/E02)

from ohlcv_dataroad.ingest import tda17_analysis as an  # noqa: E402
from ohlcv_dataroad.ingest import tda17_enmiendas as e01  # noqa: E402
from ohlcv_dataroad.ingest import tda17_enmiendas_e02 as m2  # noqa: E402
from ohlcv_dataroad.ingest import tda17_entradas as ent  # noqa: E402
from ohlcv_dataroad.ingest import tda17_salidas as sal  # noqa: E402
from ohlcv_dataroad.tda17_e03 import enmienda as m3  # noqa: E402
from ohlcv_dataroad.tda17_e03 import motor as mo  # noqa: E402
from test_tda17_enmienda_02 import assert_rejects, data_files, git, write_json  # noqa: E402
from test_tda17_procedencia import SYN_VAL, SYN_VAR, _small_frame, _split, _trust_synthetic  # noqa: E402

SRC = Path(m2.ROOT)
APPROVAL_DATE = "2026-10-25"
PENDING_MSG = "[E03_PENDIENTE] E03 no aprobada (estado PENDIENTE_AUDITORIA): la ruta oficial del motor E03 no se activa"


# =========================================================================== registro del PROYECTO (PENDIENTE o APROBADA)
NO_P3_MSG = f"[E03_P3_EXISTE] la publicacion {'0' * 40} no existe en el repositorio"
P3_A3_MSG = "[E03_P3_A3] el commit auditado A3 debe ser antecesor estricto de la publicacion P3"


def pending_registry(reg: dict) -> dict:
    """Forma PENDIENTE (la del commit auditado) de un registro de E03: la aprobacion solo cambia ``estado`` y
    ``aprobacion``."""
    reg = copy.deepcopy(reg)
    (entry,) = reg["enmiendas"]
    entry.update(estado=e01.ESTADO_PENDIENTE, aprobacion=None)
    return reg


def check_project_state(root: Path, tmp_path: Path) -> str:
    """Estado de E03 en el registro de ``root``: PENDIENTE_AUDITORIA (hasta el commit auditado; produccion y ruta
    oficial rechazan con E03_PENDIENTE) o APROBADA (aprobacion PREPARATORIA verificable que cita el commit auditado, con
    acta existente; nunca oficial sin una P3 valida). Cualquier otro estado es un fallo (como E02 v11). Devuelve el
    estado."""
    reg = json.loads((root / m3.REGISTRY_E03).read_text(encoding="utf-8"))
    assert list(reg) == list(m3.REGISTRY_FIELDS) and reg["requiere"] == m3._requiere()
    (entry,) = reg["enmiendas"]
    assert entry["estado"] in (e01.ESTADO_PENDIENTE, e01.ESTADO_APROBADA), entry["estado"]
    assert entry["sha256"] == m3.E03_SHA256 == e01.sha256_text_file(root / m3.E03_YAML)
    doc = yaml.safe_load((root / m3.E03_YAML).read_text(encoding="utf-8"))["enmienda"]
    assert doc["entorno_numerico"]["hilos_blas"] == 1 and doc["puntos_de_control"]["reanudacion_oficial"] == "DESHABILITADA"
    assert doc["base"]["publicacion_E02"] == m3.E02_P2 and doc["base"]["commit_auditado_E02"] == m3.E02_A2
    assert doc["paralelismo"]["memoria_insuficiente"] == "RECHAZO_EXPLICITO" and doc["paralelismo"]["reserva_memoria_mb"] == 4096
    assert doc["publicacion_oficial"]["comprobante_E03"] == m3.COMPROBANTE
    r = m3.official_resources(doc)
    assert r.memoria_estricta is True and r.reserva_mb == 4096.0 and r.hilos_blas is None
    if entry["estado"] == e01.ESTADO_PENDIENTE:
        assert entry["aprobacion"] is None
        errors, info = m3.verify(root)
        assert errors == [] and info["ruta_oficial_e03"] == f"BLOQUEADA: {PENDING_MSG}" and info["estado"] == e01.ESTADO_PENDIENTE
        assert_rejects(PENDING_MSG, m3.official_e03, root, "0" * 40)
        assert_rejects(PENDING_MSG, m3.load_e03_aprobada, root)
        assert_rejects(PENDING_MSG, m3.official_publish_th35_e03, "0" * 40, root, ejecucion=tmp_path / "ej")
        assert not (tmp_path / "ej").exists()
        return e01.ESTADO_PENDIENTE
    a = entry["aprobacion"]
    assert isinstance(a, dict) and set(a) == set(e01.APPROVAL_FIELDS), a
    ref = a["referencia_auditoria"]
    assert isinstance(ref, dict) and set(ref) == set(e01.AUDIT_REFERENCE_FIELDS), ref
    commit, documento = ref["commit_auditado"], ref["documento"]
    assert isinstance(commit, str) and len(commit) == 40 and set(commit) <= set("0123456789abcdef"), commit
    assert a["sha256_auditados"] == {r: e01.sha256_text_file(root / r) for r in m3.AUDITED_E03}    # arbol = aprobado
    assert a["sha256_auditados"] == {r: m2._hash_at(root, commit, r) for r in m3.AUDITED_E03}      # = commit auditado
    assert pending_registry(reg) == json.loads(git(root, "show", f"{commit}:{m3.REGISTRY_E03}"))  # commit: PENDIENTE
    assert commit != m3.E02_P2 and m2._ancestor(root, m3.E02_P2, commit) and m2._ancestor(root, commit, "HEAD")
    acta = root / documento
    assert isinstance(documento, str) and not Path(documento).is_absolute() and ".." not in Path(documento).parts
    assert acta.is_file() and not acta.is_symlink(), documento                                    # acta existente
    prep = m3.load_e03_aprobada(root)
    assert (prep["estado"], prep["modo"], prep["ejecucion_oficial"], prep["publicacion_E03"], prep["aprobacion"]) == \
        (e01.ESTADO_APROBADA, "PRODUCCION", False, None, a)
    errors, info = m3.verify(root)                                         # sin --referencia-git: nunca oficial
    assert errors == [] and info["estado"] == e01.ESTADO_APROBADA and info["ruta_oficial_e03"] == f"BLOQUEADA: {NO_P3_MSG}"
    assert_rejects(NO_P3_MSG, m3.official_e03, root, "0" * 40)              # sin P3 no hay ejecucion oficial
    assert_rejects(P3_A3_MSG, m3.official_e03, root, commit)                # el commit auditado no es una P3
    assert_rejects(NO_P3_MSG, m3.official_publish_th35_e03, "0" * 40, root, ejecucion=tmp_path / "ej")
    assert not (tmp_path / "ej").exists()
    return e01.ESTADO_APROBADA


def test_project_registry_is_pending_or_approved_and_the_official_e03_route_needs_a_valid_p3(tmp_path):
    """R4: el registro del proyecto admite exactamente PENDIENTE_AUDITORIA y APROBADA (antes solo PENDIENTE)."""
    assert check_project_state(SRC, tmp_path) in (e01.ESTADO_PENDIENTE, e01.ESTADO_APROBADA)


def test_the_e02_publication_constants_are_the_published_ones():
    assert m2._git(SRC, "rev-parse", f"{m3.E02_P2}^{{commit}}").stdout.decode().strip() == m3.E02_P2
    assert m2._ancestor(SRC, m3.E02_A2, m3.E02_P2) and m2._ancestor(SRC, m2.E01_P1, m3.E02_A2)
    shown = json.loads(m2._git(SRC, "show", f"{m3.E02_P2}:{m2.REGISTRY_E02}").stdout.decode("utf-8"))
    (entry,) = shown["enmiendas"]
    assert e01.canonical_sha256(entry) == m3.E02_ENTRY_SHA256 and entry["estado"] == e01.ESTADO_APROBADA
    assert entry["aprobacion"]["referencia_auditoria"]["commit_auditado"] == m3.E02_A2


# =========================================================================== repositorio temporal P2 -> A3 -> P3
def approve(root: Path, a3: str, **override) -> dict:
    reg = json.loads((root / m3.REGISTRY_E03).read_text(encoding="utf-8"))
    entry = reg["enmiendas"][0]
    entry["estado"] = e01.ESTADO_APROBADA
    entry["aprobacion"] = {"fecha": APPROVAL_DATE, "aprobada_por": "PRUEBA_SINTETICA",
                           "referencia_auditoria": {"documento": "PRUEBA_SINTETICA", "commit_auditado": a3},
                           "sha256_auditados": {r: e01.sha256_text_file(root / r) for r in m3.AUDITED_E03}}
    entry["aprobacion"].update(override)
    write_json(root / m3.REGISTRY_E03, reg)
    return reg


def pending(root: Path) -> None:
    reg = json.loads((root / m3.REGISTRY_E03).read_text(encoding="utf-8"))
    reg["enmiendas"][0].update(estado=e01.ESTADO_PENDIENTE, aprobacion=None)
    write_json(root / m3.REGISTRY_E03, reg)


def build_e03_repo(dest: Path) -> dict:
    """Clon en la P2 REAL + ficheros de E03 del arbol de trabajo -> A3 (registro PENDIENTE) -> aprobacion simulada ->
    P3, publicado en un remoto TEMPORAL."""
    git(dest.parent, "clone", "-q", str(SRC), dest.name)
    git(dest, "remote", "remove", "origin")                                 # nunca el repositorio del proyecto
    git(dest, "checkout", "-q", "-B", "main", m3.E02_P2)
    for rel in m3.TRACKED_E03:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SRC / rel, dest / rel)
    pending(dest)
    for rel in data_files():
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        if not (dest / rel).exists():
            os.link(SRC / rel, dest / rel)                                  # solo lectura en las pruebas
    git(dest, "add", "--", *m3.TRACKED_E03)
    git(dest, "commit", "-q", "-m", "A3: E03 auditada (sintetica)")
    a3 = git(dest, "rev-parse", "HEAD")
    approve(dest, a3)
    git(dest, "add", "--", m3.REGISTRY_E03)
    git(dest, "commit", "-q", "-m", "P3: aprobacion de E03 (sintetica)")
    p3 = git(dest, "rev-parse", "HEAD")
    bare = dest.parent / f"{dest.name}_remoto.git"
    git(dest.parent, "init", "-q", "--bare", bare.name)
    git(dest, "remote", "add", "origin", str(bare))
    git(dest, "push", "-q", "origin", "main")
    return {"root": dest, "A3": a3, "P3": p3, "bare": bare}


@pytest.fixture(scope="module")
def built3(tmp_path_factory):
    return build_e03_repo(tmp_path_factory.mktemp("m3") / "r")


@pytest.fixture
def repo3(built3, monkeypatch):
    """El repositorio en P3 limpio (rama main, remoto en P3) antes de cada prueba."""
    root, p3 = built3["root"], built3["P3"]
    git(root, "checkout", "-q", "-f", "main")
    git(root, "reset", "-q", "--hard", p3)
    git(root, "clean", "-fdq", "-e", "data/")
    for b in git(root, "branch", "--format=%(refname:short)").split():
        if b != "main":
            git(root, "branch", "-q", "-D", b)
    git(root, "config", "--unset-all", "url.https://example.invalid/.insteadof", check=False)
    git(root, "remote", "remove", "origin", check=False)
    git(root, "remote", "add", "origin", str(built3["bare"]))
    git(root, "push", "-q", "-f", "origin", f"{p3}:refs/heads/main")      # remoto TEMPORAL real en P3
    monkeypatch.setattr(m2, "AUTHORIZED_REMOTE", str(built3["bare"]))      # identidad autorizada SOLO en la prueba
    for rel in data_files():
        if not (root / rel).exists():
            os.link(SRC / rel, root / rel)
    return built3


def edit_registry(root: Path, fn) -> None:
    reg = json.loads((root / m3.REGISTRY_E03).read_text(encoding="utf-8"))
    fn(reg)
    write_json(root / m3.REGISTRY_E03, reg)


def test_the_synthetic_approval_chain_is_accepted(repo3):
    root, a3, p3 = repo3["root"], repo3["A3"], repo3["P3"]
    prep = m3.load_e03_aprobada(root)
    assert prep["estado"] == e01.ESTADO_APROBADA and prep["ejecucion_oficial"] is False and prep["publicacion_E03"] is None
    ctx = m3.official_e03(root, p3)
    info = ctx["e03"]
    assert info["ejecucion_oficial"] is True and info["publicacion_E03"] == p3 and info["commit_auditado_E03"] == a3
    assert info["remoto_E03"] == {"repositorio": str(repo3["bare"]), "ref": "refs/heads/main", "sha": p3}
    assert ctx["cfg"] == an.official_config(m3.E02_P2, root)                 # la configuracion de E02, sin cambios
    errors, v = m3.verify(root, p3)
    assert errors == [] and v["ruta_oficial_e03"] == "ACTIVABLE"


# --------------------------------------------------------------------------- R4: control del proyecto en ambos estados
ACTA = "reports/mnq/TDA17_E03_acta_sintetica.md"


def publish_with_minutes(repo3) -> str:
    """P3' = P3 + acta SINTETICA + aprobacion simulada que la cita, publicada en el remoto TEMPORAL. Devuelve P3'."""
    root, a3 = repo3["root"], repo3["A3"]
    (root / ACTA).parent.mkdir(parents=True, exist_ok=True)
    (root / ACTA).write_text("# Acta SINTETICA de E03 (PRUEBA_SINTETICA)\n", encoding="utf-8")
    approve(root, a3, referencia_auditoria={"documento": ACTA, "commit_auditado": a3})
    git(root, "add", "--", ACTA, m3.REGISTRY_E03)
    git(root, "commit", "-q", "-m", "P3 con acta (sintetica)")
    git(root, "push", "-q", "origin", "main")
    return git(root, "rev-parse", "HEAD")


def test_the_project_state_check_in_a_temporary_repository_in_both_states(repo3, tmp_path):
    """El control del registro del proyecto (``check_project_state``) sobre el repositorio temporal: la P3 publicada
    con acta (APROBADA) habilita la via oficial, y solo con esa P3; el commit auditado A3 (PENDIENTE) no."""
    root, a3 = repo3["root"], repo3["A3"]
    p3 = publish_with_minutes(repo3)
    assert check_project_state(root, tmp_path / "aprobada") == e01.ESTADO_APROBADA
    info = m3.official_e03(root, p3)["e03"]
    assert (info["ejecucion_oficial"], info["publicacion_E03"], info["commit_auditado_E03"]) == (True, p3, a3)
    assert info["remoto_E03"] == {"repositorio": str(repo3["bare"]), "ref": "refs/heads/main", "sha": p3}
    errors, v = m3.verify(root, p3)
    assert errors == [] and v["ruta_oficial_e03"] == "ACTIVABLE"
    git(root, "checkout", "-q", a3)                                         # el commit auditado: registro PENDIENTE
    assert check_project_state(root, tmp_path / "pendiente") == e01.ESTADO_PENDIENTE


def _append(path: Path) -> None:
    with open(path, "ab") as fh:
        fh.write(b"\n# cambio no auditado\n")


def _forged_states(root: Path, a3: str):
    """Aprobaciones del proyecto falsificadas o incoherentes (sobre la P3 publicada con acta)."""
    def entry(fn):
        return lambda: edit_registry(root, lambda r: fn(r["enmiendas"][0]))

    def ref(**kw):
        return entry(lambda e: e["aprobacion"]["referencia_auditoria"].update(**kw))
    yield "otro_commit", ref(commit_auditado=m3.E02_P2)
    yield "commit_inexistente", ref(commit_auditado="f" * 40)
    yield "commit_abreviado", ref(commit_auditado=a3[:12])
    yield "huella", entry(lambda e: e["aprobacion"]["sha256_auditados"].update({m3.AUDITED_E03[2]: "0" * 64}))
    yield "huella_falta", entry(lambda e: e["aprobacion"]["sha256_auditados"].pop(m3.AUDITED_E03[-1]))
    yield "fichero_alterado", lambda: _append(root / m3.AUDITED_E03[2])
    yield "yaml_alterado", lambda: _append(root / m3.E03_YAML)
    yield "sin_acta", lambda: (root / ACTA).unlink()
    yield "acta_inexistente", ref(documento="reports/mnq/NO_EXISTE.md")
    yield "acta_directorio", ref(documento="reports/mnq")
    yield "acta_absoluta", ref(documento=str((root / ACTA).resolve()))
    yield "acta_fuera", ref(documento=f"../{root.name}/{ACTA}")
    yield "campos", entry(lambda e: e["aprobacion"].update(extra=1))
    yield "fecha_futura", entry(lambda e: e["aprobacion"].update(fecha="2999-01-01"))
    yield "responsable", entry(lambda e: e["aprobacion"].update(aprobada_por=" "))
    yield "estado_desconocido", entry(lambda e: e.update(estado="APROBADA_PROVISIONAL"))
    yield "aprobada_sin_datos", entry(lambda e: e.update(aprobacion=None))
    yield "pendiente_con_datos", entry(lambda e: e.update(estado=e01.ESTADO_PENDIENTE))
    yield "registro_cambiado", lambda: edit_registry(root, lambda r: r.update(descripcion="cambiada"))


FORGED = ["otro_commit", "commit_inexistente", "commit_abreviado", "huella", "huella_falta", "fichero_alterado",
          "yaml_alterado", "sin_acta", "acta_inexistente", "acta_directorio", "acta_absoluta", "acta_fuera", "campos",
          "fecha_futura", "responsable", "estado_desconocido", "aprobada_sin_datos", "pendiente_con_datos",
          "registro_cambiado"]


@pytest.mark.parametrize("case", FORGED)
def test_a_forged_or_inconsistent_project_approval_is_always_rejected(repo3, tmp_path, case):
    root, a3 = repo3["root"], repo3["A3"]
    publish_with_minutes(repo3)
    assert check_project_state(root, tmp_path / "control") == e01.ESTADO_APROBADA     # control sin falsificar
    assert [c for c, _ in _forged_states(root, a3)] == FORGED
    (fn,) = [f for c, f in _forged_states(root, a3) if c == case]
    fn()
    with pytest.raises((AssertionError, m3.ProtocolNotAuthorizedError)):
        check_project_state(root, tmp_path / "falsificada")
    assert not (tmp_path / "falsificada" / "ej").exists()


# --------------------------------------------------------------------------- registro y documento
def _registry_cases(root: Path, a3: str):
    yield "falta", lambda: (root / m3.REGISTRY_E03).unlink(), "[E03_REGISTRO] falta el registro de E03"
    yield "json", lambda: (root / m3.REGISTRY_E03).write_text("{", encoding="utf-8"), \
        "[E03_REGISTRO] el registro de E03 no es JSON valido"
    yield "estructura", lambda: edit_registry(root, lambda r: r.update(extra=1)), \
        f"[E03_REG_ESTRUCTURA] el registro de E03 debe tener exactamente {list(m3.REGISTRY_FIELDS)}"
    yield "base", lambda: edit_registry(root, lambda r: r["protocolo_base"].update(commit="0" * 40)), \
        "[E03_BASE] el registro de E03 no declara el protocolo base congelado"
    yield "requiere", lambda: edit_registry(root, lambda r: r["requiere"].update(publicacion_E02="1" * 40)), \
        "[E03_REQUIERE] el registro de E03 no declara la E02 publicada (entrada, A2, P2)"
    yield "entrada", lambda: edit_registry(root, lambda r: r["enmiendas"][0].update(alcance="ESTADISTICA")), \
        "[E03_ENTRADA] el registro de E03 debe tener una unica entrada E03 bien formada"
    yield "dos_entradas", lambda: edit_registry(root, lambda r: r["enmiendas"].append(r["enmiendas"][0])), \
        "[E03_ENTRADA] el registro de E03 debe tener una unica entrada E03 bien formada"
    yield "estado", lambda: edit_registry(root, lambda r: r["enmiendas"][0].update(estado="OTRO")), \
        "[E03_ESTADO] estado desconocido 'OTRO'"
    yield "pendiente", lambda: pending(root), PENDING_MSG
    yield "pendiente_con_datos", lambda: edit_registry(root, lambda r: r["enmiendas"][0].update(estado=e01.ESTADO_PENDIENTE)), \
        "[E03_PENDIENTE] E03 no aprobada (estado PENDIENTE_AUDITORIA): la ruta oficial del motor E03 no se activa"


@pytest.mark.parametrize("case", ["falta", "json", "estructura", "base", "requiere", "entrada", "dos_entradas",
                                  "estado", "pendiente", "pendiente_con_datos"])
def test_registry_rejections(repo3, case):
    root, a3, p3 = repo3["root"], repo3["A3"], repo3["P3"]
    (fn, msg), = [(f, m) for c, f, m in _registry_cases(root, a3) if c == case]
    fn()
    assert_rejects(msg, m3.official_e03, root, p3)


def test_pending_with_approval_data_is_rejected_in_draft_mode(repo3):
    root = repo3["root"]
    edit_registry(root, lambda r: r["enmiendas"][0].update(estado=e01.ESTADO_PENDIENTE))
    assert_rejects("[E03_PENDIENTE_CON_DATOS] E03 PENDIENTE con datos de aprobacion", m3.effective_e03, root, borrador=True)


def test_yaml_fingerprint_and_content(repo3, monkeypatch):
    root, p3 = repo3["root"], repo3["P3"]
    path = root / m3.E03_YAML
    original = path.read_bytes()
    path.write_bytes(original + b"\n# cambio\n")
    digest = e01.sha256_text_file(path)
    assert_rejects(f"[E03_HUELLA] {m3.E03_YAML} no es la version autorizada de E03 (SHA-256 {digest})",
                   m3.official_e03, root, p3)
    msg = ("[E03_DOC] el YAML de E03 no es la enmienda implementada (alcance, base, criterio, entorno, reanudacion, "
           "memoria o comprobante)")
    for ruta, valor in ((("entorno_numerico", "hilos_blas"), 2), (("entorno_numerico", "hilos_blas"), True),
                        (("paralelismo", "memoria_insuficiente"), "SECUENCIAL"),
                        (("paralelismo", "reserva_memoria_mb"), 0), (("paralelismo", "reserva_memoria_mb"), 4096.5),
                        (("publicacion_oficial", "comprobante_E03"), "otro.json"),
                        (("puntos_de_control", "reanudacion_oficial"), "PERMITIDA_CON_VERIFICACION"),
                        (("base", "publicacion_E02"), "1" * 40), (("criterio_equivalencia", "tolerancias"), "1e-12"),
                        (("decisiones_estadisticas",), "UNA")):
        doc = yaml.safe_load(original.decode("utf-8"))
        node = doc["enmienda"]
        for k in ruta[:-1]:
            node = node[k]
        node[ruta[-1]] = valor
        path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
        h = e01.sha256_text_file(path)
        monkeypatch.setattr(m3, "E03_SHA256", h)                            # barrera anterior (huella) fijada
        edit_registry(root, lambda r, h=h: r["enmiendas"][0].update(sha256=h))
        assert_rejects(msg, m3.official_e03, root, p3)


# --------------------------------------------------------------------------- aprobacion (A3)
@pytest.mark.parametrize("case", ["campos", "fecha_futura", "fecha_anterior", "fecha_formato", "responsable", "referencia",
                                  "commit_formato", "cobertura", "arbol", "commit_inexistente", "commit_sin_ficheros"])
def test_approval_rejections(repo3, case):
    root, a3, p3 = repo3["root"], repo3["A3"], repo3["P3"]
    msgs = {
        "campos": f"[E03_CAMPOS] la aprobacion debe tener exactamente {list(e01.APPROVAL_FIELDS)}",
        "fecha_futura": "[E03_FECHA] fecha '2999-01-01' no ISO valida o fuera de [fecha_preparacion, hoy]",
        "fecha_anterior": "[E03_FECHA] fecha '2026-10-24' no ISO valida o fuera de [fecha_preparacion, hoy]",
        "fecha_formato": "[E03_FECHA] fecha '25/10/2026' no ISO valida o fuera de [fecha_preparacion, hoy]",
        "responsable": "[E03_RESPONSABLE] aprobada_por vacio (responsable declarado obligatorio)",
        "referencia": "[E03_REFERENCIA] referencia_auditoria debe tener documento no vacio y commit_auditado",
        "commit_formato": "[E03_COMMIT_FORMATO] commit_auditado no es un SHA-1 Git completo (40 hexadecimales)",
        "cobertura": "[E03_COBERTURA] sha256_auditados no cubre exactamente los ficheros auditados de E03",
        "arbol": f"[E03_ARBOL] ficheros distintos de la version auditada de E03: {[m3.AUDITED_E03[2]]}",
        "commit_inexistente": f"[E03_COMMIT] el commit auditado {'f' * 40} no existe o no contiene la version aprobada "
                              f"de {m3.AUDITED_E03}",
        "commit_sin_ficheros": f"[E03_COMMIT] el commit auditado {m3.E02_P2} no existe o no contiene la version "
                               f"aprobada de {m3.AUDITED_E03}",
    }
    if case == "campos":
        approve(root, a3, extra=1)
    elif case.startswith("fecha"):
        approve(root, a3, fecha={"fecha_futura": "2999-01-01", "fecha_anterior": "2026-10-24",
                                 "fecha_formato": "25/10/2026"}[case])
    elif case == "responsable":
        approve(root, a3, aprobada_por="  ")
    elif case == "referencia":
        approve(root, a3, referencia_auditoria={"documento": "", "commit_auditado": a3})
    elif case == "commit_formato":
        approve(root, a3, referencia_auditoria={"documento": "x", "commit_auditado": a3[:12]})
    elif case == "cobertura":
        reg = approve(root, a3)
        reg["enmiendas"][0]["aprobacion"]["sha256_auditados"].pop(m3.AUDITED_E03[-1])
        write_json(root / m3.REGISTRY_E03, reg)
    elif case == "arbol":
        with open(root / m3.AUDITED_E03[2], "ab") as fh:
            fh.write(b"\n# cambio no auditado\n")
    elif case == "commit_inexistente":
        approve(root, a3, referencia_auditoria={"documento": "x", "commit_auditado": "f" * 40})
    else:
        approve(root, a3, referencia_auditoria={"documento": "x", "commit_auditado": m3.E02_P2})
    assert_rejects(msgs[case], m3.official_e03, root, p3)


def test_an_audited_commit_outside_the_e02_publication_history_is_rejected(repo3):
    """A3 con los mismos ficheros pero sin la P2 de E02 en su historia (commit huerfano)."""
    root, p3 = repo3["root"], repo3["P3"]
    git(root, "checkout", "-q", "--orphan", "huerfana")
    git(root, "add", "--", *m3.TRACKED_E03)
    git(root, "commit", "-q", "-m", "A3 huerfano")
    orphan = git(root, "rev-parse", "HEAD")
    git(root, "checkout", "-q", "-f", "main")
    approve(root, orphan)
    assert_rejects("[E03_ASCENDENCIA] el commit auditado de E03 no desciende de la publicacion P2 de E02",
                   m3.official_e03, root, p3)


def test_head_must_descend_from_the_audited_commit(repo3):
    root, a3, p3 = repo3["root"], repo3["A3"], repo3["P3"]
    git(root, "checkout", "-q", "-b", "otra", m3.E02_P2)
    for rel in m3.TRACKED_E03:
        git(root, "checkout", p3, "--", rel)
    git(root, "commit", "-q", "-m", "mismos ficheros, otra historia")
    assert_rejects("[E03_A3_HEAD] HEAD no desciende del commit auditado A3 de E03", m3.official_e03, root, p3)


# --------------------------------------------------------------------------- publicacion (P3) y remoto
def test_publication_rejections(repo3, monkeypatch):
    root, a3, p3 = repo3["root"], repo3["A3"], repo3["P3"]
    assert_rejects("[E03_P3_FORMATO] la ejecucion oficial de E03 exige el SHA-1 completo de la publicacion P3",
                   m3.official_e03, root, p3[:12])
    assert_rejects("[E03_P3_FORMATO] la ejecucion oficial de E03 exige el SHA-1 completo de la publicacion P3",
                   m3.official_e03, root, None)
    assert_rejects(f"[E03_P3_EXISTE] la publicacion {'e' * 40} no existe en el repositorio", m3.official_e03, root, "e" * 40)
    assert_rejects("[E03_P3_A3] el commit auditado A3 debe ser antecesor estricto de la publicacion P3",
                   m3.official_e03, root, a3)
    assert_rejects("[E03_P3_A3] el commit auditado A3 debe ser antecesor estricto de la publicacion P3",
                   m3.official_e03, root, m3.E02_P2)
    edit_registry(root, lambda r: r.update(descripcion="cambiada"))
    assert_rejects(f"[E03_P3_ARBOL] el arbol no coincide con la publicacion P3: "
                   f"{[f'[git {p3}] difiere del commit de referencia: {m3.REGISTRY_E03}']}", m3.official_e03, root, p3)
    git(root, "checkout", "-q", "-f", "main")
    # P3 no antecesora de HEAD: HEAD en A3 con el registro aprobado en el arbol
    git(root, "checkout", "-q", a3)
    git(root, "checkout", p3, "--", m3.REGISTRY_E03)
    assert_rejects("[E03_P3_HEAD] la publicacion P3 no es HEAD ni antecesora de HEAD", m3.official_e03, root, p3)
    git(root, "checkout", "-q", "-f", "main")
    # una publicacion cuyo registro no contiene la aprobacion (barrera del arbol fijada)
    pending(root)
    git(root, "commit", "-q", "-am", "registro pendiente")
    px = git(root, "rev-parse", "HEAD")
    git(root, "checkout", p3, "--", m3.REGISTRY_E03)
    monkeypatch.setattr(m3, "verify_against_git_e03", lambda r, c: [])
    assert_rejects("[E03_P3_APROBADA] la publicacion P3 no contiene la aprobacion de E03 que cita A3",
                   m3.official_e03, root, px)


def test_a_local_approval_that_is_not_published_never_enables_production(repo3):
    """Registro APROBADO y commits A3/P3 verificables SOLO localmente: el remoto autorizado sigue en P2."""
    root, p3 = repo3["root"], repo3["P3"]
    git(root, "push", "-q", "-f", "origin", f"{m3.E02_P2}:refs/heads/main")
    assert m3.load_e03_aprobada(root)["estado"] == e01.ESTADO_APROBADA          # la aprobacion local es coherente
    assert_rejects(f"[E03_REMOTO] la publicacion P3 {p3} no esta en la rama publicada del remoto ({m3.E02_P2})",
                   m3.official_e03, root, p3)


def test_a_forged_publication_on_a_local_branch_is_rejected(repo3):
    """Nueva P3' local (descendiente de A3, aprobacion coherente) que nunca se publico."""
    root, p3 = repo3["root"], repo3["P3"]
    edit_registry(root, lambda r: r.update(descripcion="P3 falsificada"))
    git(root, "commit", "-q", "-am", "P3 falsificada")
    forged = git(root, "rev-parse", "HEAD")
    assert_rejects(f"[E03_REMOTO] la publicacion P3 {forged} no esta en la rama publicada del remoto ({p3})",
                   m3.official_e03, root, forged)


def test_remote_identity_rejections(repo3, monkeypatch, tmp_path):
    root, p3, bare = repo3["root"], repo3["P3"], repo3["bare"]
    git(root, "remote", "set-url", "origin", str(tmp_path / "otro.git"))
    assert_rejects(f"[E03_REMOTO_NO_AUTORIZADO] origin ({tmp_path / 'otro.git'}) no es el repositorio autorizado {bare}",
                   m3.official_e03, root, p3)
    git(root, "remote", "remove", "origin")
    assert_rejects(f"[E03_REMOTO_NO_AUTORIZADO] origin (inexistente) no es el repositorio autorizado {bare}",
                   m3.official_e03, root, p3)
    git(root, "remote", "add", "origin", str(bare))
    git(root, "config", "url.https://example.invalid/.insteadOf", "file:///nada")
    assert_rejects("[E03_REMOTO_NO_AUTORIZADO] la configuracion de Git redirige URL (url.*.insteadOf): no se puede "
                   "garantizar la identidad del repositorio autorizado", m3.official_e03, root, p3)
    git(root, "config", "--unset-all", "url.https://example.invalid/.insteadof")
    missing = tmp_path / "inaccesible.git"
    monkeypatch.setattr(m2, "AUTHORIZED_REMOTE", str(missing))
    git(root, "remote", "set-url", "origin", str(missing))
    assert_rejects("[E03_REMOTO] no se pudo consultar la rama publicada refs/heads/main del repositorio autorizado",
                   m3.official_e03, root, p3)


def test_a_modified_e02_file_blocks_the_official_e03_route_with_the_e02_control(repo3):
    root, p3 = repo3["root"], repo3["P3"]
    rel = "src/ohlcv_dataroad/ingest/tda17_analysis.py"
    with open(root / rel, "ab") as fh:
        fh.write(b"\n# cambio\n")
    assert_rejects(f"[E02_ARBOL] ficheros distintos de la version auditada de E02: {[rel]}", m3.official_e03, root, p3)


# =========================================================================== ruta oficial con el motor E03
@pytest.fixture
def official3(repo3, monkeypatch):
    """Entradas SINTETICAS en el repositorio temporal; motor y E02 con 3 replicas (solo mecanica) y 2 workers."""
    root = repo3["root"]
    variables, validity = _split(_small_frame())
    (root / "data" / "sintetico").mkdir(parents=True, exist_ok=True)
    variables.to_parquet(root / SYN_VAR, index=False)
    validity.to_parquet(root / SYN_VAL, index=False)
    monkeypatch.setattr(ent, "VARIABLES", SYN_VAR)
    monkeypatch.setattr(ent, "VALIDITY", SYN_VAL)
    monkeypatch.setattr(ent, "INPUTS", (SYN_VAR, SYN_VAL))
    assert not any(r.startswith("data/interim") for r in ent.INPUTS)        # guarda: nunca la entrada real
    from ohlcv_dataroad.ingest import tda17_protocol_freeze as frz
    ctx = {**repo3, "hashes": {SYN_VAR: frz.sha256_file(root / SYN_VAR), SYN_VAL: frz.sha256_file(root / SYN_VAL)}}
    _trust_synthetic(monkeypatch, ctx)
    real_e03, real_e02, real_res = mo.th35_pipeline_e03, an.th35_pipeline, m3.official_resources

    def small(c):
        c = copy.deepcopy(c)
        c["analisis"]["bootstrap"]["replicas"] = 3
        return c
    monkeypatch.setattr(mo, "th35_pipeline_e03", lambda df, c, motor: real_e03(df, small(c), motor))
    monkeypatch.setattr(an, "th35_pipeline", lambda df, c: real_e02(df, small(c)))
    def two_workers(doc):
        r = real_res(doc)
        r.procesos = 2
        return r
    monkeypatch.setattr(m3, "official_resources", two_workers)
    out = ctx["out"] = root / "reports" / "mnq"
    shutil.rmtree(out / sal.STAGING, ignore_errors=True)        # restos ignorados por git de una prueba anterior
    for n in m3._salidas_e03():
        if os.path.lexists(out / n):
            (out / n).unlink()
    return ctx


class _Muerte(BaseException):
    """Muerte simulada del proceso (no es Exception: nada la captura salvo el deshacer, aqui anulado)."""


def _receipt_cases(root, p3, a3, out):
    """Alteraciones del comprobante de E03 (sin recalcular): cada una se rechaza con el campo exacto; se restaura."""
    path = out / m3.COMPROBANTE
    good = path.read_bytes()
    comp = json.loads(good)

    def with_receipt(change, message):
        c = copy.deepcopy(comp)
        change(c)
        path.write_bytes((json.dumps(c, indent=2) + "\n").encode("utf-8"))
        assert_rejects(message, m3.check_official_publication_e03, p3, root)
        path.write_bytes(good)
    vinculo = f"[E03_COMPROBANTE_VINCULO] {m3.COMPROBANTE} no corresponde a esta publicacion E03: "
    with_receipt(lambda c: c.update(publicacion_E03=a3), vinculo + "['publicacion_E03']")
    with_receipt(lambda c: c.update(commit_auditado_E03=p3), vinculo + "['commit_auditado_E03']")
    with_receipt(lambda c: c["remoto_E03"].update(sha=a3), vinculo + "['remoto_E03']")       # no contiene P3
    with_receipt(lambda c: c.update(sha256_yaml_E03="0" * 64), vinculo + "['sha256_yaml_E03']")
    with_receipt(lambda c: c.update(sha256_registro_E02="0" * 64), vinculo + "['sha256_registro_E02']")
    with_receipt(lambda c: c["tablas"].update({sal.CAUSAL: "0" * 64}), vinculo + "['tablas']")
    with_receipt(lambda c: c.update(reanudacion="PERMITIDA"), vinculo + "['reanudacion']")
    with_receipt(lambda c: c["motor"]["hilos_blas"].update({k: 24 for k in c["motor"]["hilos_blas"]}),
                 vinculo + "['motor']")
    with_receipt(lambda c: c["motor"]["recursos"].update(memoria_estricta=False), vinculo + "['motor']")
    with_receipt(lambda c: c.pop("motor"),
                 f"[E03_COMPROBANTE] {m3.COMPROBANTE} no es JSON con exactamente los campos {m3.COMPROBANTE_FIELDS}")
    path.write_bytes(good[:-3])
    assert_rejects(f"[E03_COMPROBANTE] {m3.COMPROBANTE} no es JSON con exactamente los campos "
                   f"{m3.COMPROBANTE_FIELDS}", m3.check_official_publication_e03, p3, root)
    path.unlink()
    assert_rejects(f"[E03_COMPROBANTE] falta el comprobante {m3.COMPROBANTE}", m3.check_official_publication_e03, p3, root)
    path.write_bytes(good)
    assert m3._check_comprobante(root, out, sal.check_official_publication(m3.E02_P2, root),
                                 m3.official_e03(root, p3)["e03"]) == comp


def test_official_e03_route_publishes_the_same_bytes_as_the_e02_route(official3, tmp_path):
    ctx = official3
    root, p3, a3, out = ctx["root"], ctx["P3"], ctx["A3"], ctx["out"]
    antes = mo.hilos_blas_efectivos()
    rec = m3.official_publish_th35_e03(p3, root, ejecucion=tmp_path / "ej")
    assert mo.hilos_blas_efectivos() == antes                                   # 1 hilo solo durante la ejecucion
    assert sal.check_official_publication(m3.E02_P2, root) == rec               # verificador de E02 sin cambios
    assert m3.check_official_publication_e03(p3, root) == rec
    published = {n: (out / n).read_bytes() for n in [*rec["tablas"], sal.PROVENANCE]}
    assert not (out / sal.STAGING).exists() and not (tmp_path / "ej" / "publicacion.json").exists()
    comp = json.loads((out / m3.COMPROBANTE).read_text(encoding="utf-8"))
    assert list(comp) == m3.COMPROBANTE_FIELDS and comp["publicacion_E03"] == p3 and comp["commit_auditado_E03"] == a3
    assert comp["remoto_E03"] == {"repositorio": str(ctx["bare"]), "ref": "refs/heads/main", "sha": p3}
    assert comp["sha256_registro_E02"] == sal._sha(published[sal.PROVENANCE]) and comp["tablas"] == rec["tablas"]
    assert comp["sha256_yaml_E03"] == m3.E03_SHA256 and comp["hilos_blas"] == 1
    assert set(comp["motor"]["hilos_blas"].values()) == {1} and comp["motor"]["recursos"]["memoria_estricta"] is True
    assert comp["motor"]["recursos"]["reserva_mb"] == 4096.0 and comp["motor"]["lanzadas_sin_memoria_suficiente"] == 0
    done = [json.loads(x) for x in (tmp_path / "ej" / "progreso.jsonl").read_text(encoding="utf-8").splitlines()]
    done = [e for e in done if e["evento"] == "tarea_completa"]
    assert done and all(set(e["hilos_blas"].values()) == {1} for e in done)
    assert (tmp_path / "ej" / "COMPLETA.json").is_file()
    _receipt_cases(root, p3, a3, out)
    assert m3.reproduce_official_publication_e03(p3, root, ejecucion=tmp_path / "ej_rep") == rec
    assert m3.reproduce_with_e02_engine_e03(p3, root) == rec                   # el motor E02 reproduce lo de E03
    for n in [*published, m3.COMPROBANTE]:
        (out / n).unlink()
    with mo.hilos_blas_fijados(1):
        rec2 = sal.official_publish_th35(m3.E02_P2, root)                       # ruta E02 en el mismo entorno
    assert rec2 == rec and {n: (out / n).read_bytes() for n in published} == published
    assert_rejects(f"[E03_COMPROBANTE] falta el comprobante {m3.COMPROBANTE}", m3.check_official_publication_e03, p3, root)
    # salidas existentes: rechazo ANTES del calculo (el directorio de la ejecucion ni siquiera se crea)
    assert_rejects(f"[SALIDA_EXISTENTE] ya existen salidas oficiales (no se sobrescriben): "
                   f"{[n for n in sal.OFFICIAL if n in published]}",
                   m3.official_publish_th35_e03, p3, root, ejecucion=tmp_path / "ej2")
    assert not (tmp_path / "ej2").exists()


def test_an_e03_publication_interrupted_between_the_receipt_and_the_e02_record_is_never_valid(official3, tmp_path,
                                                                                             monkeypatch):
    """Muerte del proceso (sin deshacer) despues de promover tablas y comprobante y antes del registro de E02: la
    publicacion no es valida para E02 ni para E03; la recuperacion retira lo demostrablemente suyo."""
    root, p3, out = official3["root"], official3["P3"], official3["out"]
    real = sal._promote

    def die_before_record(src, dst):
        if dst.name == sal.PROVENANCE:
            raise _Muerte("muerte simulada")
        real(src, dst)
    with pytest.MonkeyPatch.context() as mp:                                     # solo la muerte (no el fixture)
        mp.setattr(sal, "_promote", die_before_record)
        mp.setattr(sal, "_rollback", lambda out_dir, hashes: None)               # el proceso muere: nada se deshace
        with pytest.raises(_Muerte):
            m3.official_publish_th35_e03(p3, root, ejecucion=tmp_path / "ej")
    assert (out / m3.COMPROBANTE).exists() and (out / sal.STAGING).exists() and not (out / sal.PROVENANCE).exists()
    assert_rejects("[SALIDA_INCOMPLETA] publicacion oficial interrumpida o en curso (sin registro de procedencia "
                   "confirmado)", m3.check_official_publication_e03, p3, root)
    with sal._exclusive(out):
        m3._destino_libre(out)                                                   # recuperacion bajo el cerrojo
    assert not any(os.path.lexists(out / n) for n in m3._salidas_e03()) and not (out / sal.STAGING).exists()


# =========================================================================== transaccion de publicacion (sin calculo)
def _files(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    blobs = {sal.CAUSAL: b"a,b\n1,2\n", sal.DIAGNOSTIC: b"c\n3\n"}
    return out, blobs, {"registro": "E02"}, {"comprobante": "E03"}


def _published(out):
    return sorted(n for n in os.listdir(out) if n != sal.LOCK)


def test_the_e03_receipt_is_promoted_after_the_tables_and_before_the_e02_record(tmp_path, monkeypatch):
    out, blobs, record, comp = _files(tmp_path)
    order, real = [], sal._promote
    monkeypatch.setattr(sal, "_promote", lambda s, d: (order.append(d.name), real(s, d)))
    m3._publish_files_e03(out, blobs, record, comp)
    assert order == [sal.CAUSAL, sal.DIAGNOSTIC, m3.COMPROBANTE, sal.PROVENANCE]
    assert _published(out) == sorted(order)
    assert json.loads((out / m3.COMPROBANTE).read_text(encoding="utf-8")) == comp
    assert (out / sal.PROVENANCE).read_bytes() == (json.dumps(record, indent=2) + "\n").encode("utf-8")
    assert_rejects(f"[SALIDA_EXISTENTE] ya existen salidas oficiales (no se sobrescriben): "
                   f"{[n for n in m3._salidas_e03() if n in order]}", m3._publish_files_e03, out, blobs, record, comp)


@pytest.mark.parametrize("fallo", [sal.DIAGNOSTIC, m3.COMPROBANTE, sal.PROVENANCE])
def test_a_failure_during_the_e03_publication_leaves_nothing_published(tmp_path, monkeypatch, fallo):
    out, blobs, record, comp = _files(tmp_path)
    real = sal._promote

    def fail(src, dst):
        if dst.name == fallo:
            raise OSError("fallo simulado")
        real(src, dst)
    monkeypatch.setattr(sal, "_promote", fail)
    with pytest.raises(OSError, match="fallo simulado"):
        m3._publish_files_e03(out, blobs, record, comp)
    assert _published(out) == []                                                # se deshace todo lo suyo
    monkeypatch.undo()
    m3._publish_files_e03(out, blobs, record, comp)
    assert len(_published(out)) == 4


@pytest.mark.parametrize("muerte", [m3.COMPROBANTE, sal.PROVENANCE])
def test_a_publication_killed_before_the_e02_record_is_recovered_including_the_receipt(tmp_path, monkeypatch, muerte):
    out, blobs, record, comp = _files(tmp_path)
    real = sal._promote

    def die(src, dst):
        if dst.name == muerte:
            raise _Muerte("muerte simulada")
        real(src, dst)
    monkeypatch.setattr(sal, "_promote", die)
    monkeypatch.setattr(sal, "_rollback", lambda out_dir, hashes: None)
    with pytest.raises(_Muerte):
        m3._publish_files_e03(out, blobs, record, comp)
    monkeypatch.undo()
    assert (out / sal.STAGING).exists() and not (out / sal.PROVENANCE).exists()
    assert (out / m3.COMPROBANTE).exists() == (muerte == sal.PROVENANCE)
    comp2 = {"comprobante": "E03", "segunda": True}
    m3._publish_files_e03(out, blobs, record, comp2)                             # recupera y publica
    assert _published(out) == sorted([*blobs, m3.COMPROBANTE, sal.PROVENANCE])
    assert json.loads((out / m3.COMPROBANTE).read_text(encoding="utf-8")) == comp2


def test_a_foreign_receipt_blocks_the_recovery_and_nothing_is_removed(tmp_path, monkeypatch):
    out, blobs, record, comp = _files(tmp_path)
    real = sal._promote

    def die(src, dst):
        if dst.name == sal.PROVENANCE:
            raise _Muerte("muerte simulada")
        real(src, dst)
    monkeypatch.setattr(sal, "_promote", die)
    monkeypatch.setattr(sal, "_rollback", lambda out_dir, hashes: None)
    with pytest.raises(_Muerte):
        m3._publish_files_e03(out, blobs, record, comp)
    monkeypatch.undo()
    data = (out / m3.COMPROBANTE).read_bytes()
    (out / m3.COMPROBANTE).unlink()
    (out / m3.COMPROBANTE).write_bytes(data)                                     # mismos bytes, OTRO fichero
    before = {n: (out / n).read_bytes() for n in _published(out) if (out / n).is_file()}
    assert_rejects(f"[E03_SALIDA_RECUPERACION] no puede demostrarse que {m3.COMPROBANTE} pertenezca a la publicacion "
                   f"interrumpida: se conserva y la recuperacion queda bloqueada",
                   m3._publish_files_e03, out, blobs, record, comp)
    assert {n: (out / n).read_bytes() for n in _published(out) if (out / n).is_file()} == before
    assert (out / sal.STAGING).exists()


def test_an_orphan_receipt_is_never_overwritten_or_removed(tmp_path):
    out, blobs, record, comp = _files(tmp_path)
    (out / m3.COMPROBANTE).write_bytes(b"{}\n")
    assert_rejects(f"[SALIDA_EXISTENTE] ya existen salidas oficiales (no se sobrescriben): {[m3.COMPROBANTE]}",
                   m3._publish_files_e03, out, blobs, record, comp)
    assert _published(out) == [m3.COMPROBANTE] and (out / m3.COMPROBANTE).read_bytes() == b"{}\n"


def test_existing_outputs_are_rejected_before_any_computation(official3, tmp_path, monkeypatch):
    """Rechazo previo (barato) tras la cadena oficial y antes de cargar entradas o crear la ejecucion; la
    comprobacion bajo el cerrojo al publicar se mantiene (``_publish_files_e03``)."""
    root, p3, out = official3["root"], official3["P3"], official3["out"]
    out.mkdir(parents=True, exist_ok=True)
    (out / m3.COMPROBANTE).write_bytes(b"{}\n")
    monkeypatch.setattr(ent, "load_official_inputs", lambda *a, **k: pytest.fail("no debe cargar entradas"))
    assert_rejects(f"[SALIDA_EXISTENTE] ya existen salidas oficiales (no se sobrescriben): {[m3.COMPROBANTE]}",
                   m3.official_publish_th35_e03, p3, root, ejecucion=tmp_path / "ej")
    assert not (tmp_path / "ej").exists() and (out / m3.COMPROBANTE).read_bytes() == b"{}\n"
    src = inspect.getsource(m3.official_publish_th35_e03)
    assert src.index("_official_run_e03(") < src.index("_publish_files_e03(")
    assert "_destino_libre(" in inspect.getsource(m3._publish_files_e03)


def test_official_e03_route_has_no_resumption_and_needs_a_new_directory_outside_the_repository(official3, tmp_path,
                                                                                                monkeypatch):
    root, p3 = official3["root"], official3["P3"]
    assert "reanudar" not in inspect.signature(m3.official_publish_th35_e03).parameters
    assert "reanudar" not in inspect.signature(m3._official_run_e03).parameters
    (tmp_path / "ej").mkdir()
    assert_rejects(f"[E03_EJECUCION_EXISTENTE] el directorio de una ejecucion oficial no puede existir: "
                   f"{(tmp_path / 'ej').resolve()}", m3.official_publish_th35_e03, p3, root, ejecucion=tmp_path / "ej")
    inside = root / "ejecucion_e03"
    assert_rejects(f"[E03_EJECUCION_EN_REPOSITORIO] el directorio de la ejecucion debe estar fuera del repositorio: "
                   f"{inside.resolve()}", m3.official_publish_th35_e03, p3, root, ejecucion=inside)
    monkeypatch.setattr(mo, "psutil", None)
    assert_rejects("[E03_ENTORNO] sin medida de la memoria privada de los procesos (psutil): el control de RAM no es "
                   "verificable", m3.official_publish_th35_e03, p3, root, ejecucion=tmp_path / "ej3")
    assert not any((official3["out"] / n).exists() for n in sal.OFFICIAL) and not inside.exists()


def test_a_dead_worker_during_the_official_e03_run_publishes_nothing(official3, tmp_path, monkeypatch):
    """R4: un worker que muere durante la ejecucion oficial (aqui, justo antes de recibir su tarea) termina la
    ejecucion con E03_TRABAJADOR: ningun worker vivo y ninguna salida oficial, comprobante ni area de preparacion."""
    root, p3, out = official3["root"], official3["P3"], official3["out"]
    real = mo._Motor._enviar

    def kill_then_send(self, w):
        w.proc.terminate()
        w.proc.join(30)
        return real(self, w)
    monkeypatch.setattr(mo._Motor, "_enviar", kill_then_send)
    res = {}

    def run():
        try:
            m3.official_publish_th35_e03(p3, root, ejecucion=tmp_path / "ej")
        except BaseException as exc:                                          # noqa: BLE001
            res["exc"] = exc
    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(300)
    assert not th.is_alive(), "la ruta oficial sigue esperando a un worker muerto (bloqueo indefinido)"
    exc = res.get("exc")
    assert type(exc) is mo.MotorE03Error and str(exc).startswith("[E03_TRABAJADOR] el worker "), repr(exc)
    assert "durante el envio" in str(exc), str(exc)
    assert not any(os.path.lexists(out / n) for n in m3._salidas_e03()) and not (out / sal.STAGING).exists()
    assert not (tmp_path / "ej" / "COMPLETA.json").exists()
    eventos = [json.loads(x)["evento"] for x in (tmp_path / "ej" / "progreso.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "worker_fallo" in eventos
    vivos = [c for c in mo.psutil.Process().children(recursive=True)
             if c.is_running() and "multiprocessing" in " ".join(c.cmdline())]
    assert vivos == [], vivos


def test_official_e03_route_rejects_an_unverifiable_blas(official3, tmp_path, monkeypatch):
    root, p3 = official3["root"], official3["P3"]
    monkeypatch.setattr(mo, "_openblas", lambda: [])
    with pytest.raises(mo.MotorE03Error) as info:
        m3.official_publish_th35_e03(p3, root, ejecucion=tmp_path / "ej")
    assert str(info.value) == "[E03_BLAS] no se pueden fijar ni verificar los hilos de OpenBLAS de este proceso: []"
    assert not any((official3["out"] / n).exists() for n in sal.OFFICIAL) and not (tmp_path / "ej").exists()
