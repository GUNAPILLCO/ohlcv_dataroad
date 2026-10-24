"""Pruebas del cargador escalonado de E02 (migracion M2; src/ohlcv_dataroad/ingest/tda17_enmiendas_e02.py).

SPEC-M2 seccion 11 y checklist de garantias de la propuesta de E02 (seccion 7). Todas SINTETICAS: sin datos de MNQ
ni holdout. Las aprobaciones son SIMULADAS ("PRUEBA_SINTETICA") y viven solo en REPOSITORIOS GIT TEMPORALES:
clon del repositorio (historia real con P1 y A1), ficheros de E02 del arbol de trabajo, commit A2 (registro
PENDIENTE), aprobacion que cita A2, commit P2 y un remoto 'origin' TEMPORAL (repositorio desnudo) con P2 publicado.
El remoto del clon se elimina antes de nada: ninguna prueba puede publicar en el repositorio del proyecto. Las
entradas de datos del manifiesto (no versionadas) se enlazan (hardlink) y NUNCA se escriben.

Negativas: tipo EXACTO ``ProtocolNotAuthorizedError`` y mensaje COMPLETO (``assert_rejects``; el comparador se
autoprueba en T-CMP). Controles solo alcanzables tras otra barrera (E01_REGISTRO, E01_ENTRADA, E01_A1, E02_DOC,
E02_LISTA_BLANCA, E02_ANADIR, E02_P2_APROBADA, PROTOCOLO_GIT) se prueban fijando con ``monkeypatch`` la constante o
la funcion de la barrera anterior: documentan la redundancia sin rebajar la prueba.
"""
from __future__ import annotations

import copy
import inspect
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml

from ohlcv_dataroad.ingest import tda17_analysis as an
from ohlcv_dataroad.ingest import tda17_enmiendas as e01
from ohlcv_dataroad.ingest import tda17_enmiendas_e02 as m2
from ohlcv_dataroad.ingest import tda17_g2 as g2
from ohlcv_dataroad.ingest import tda17_protocol_freeze as frz
from ohlcv_dataroad.ingest import tda17_salidas as sal

Err = m2.ProtocolNotAuthorizedError
SRC = Path(m2.ROOT)
DATE = "2026-10-18T12:00:00+00:00"
APPROVAL_DATE = "2026-10-18"
PENDING_MSG = "[E02_PENDIENTE] E02 no aprobada (estado PENDIENTE_AUDITORIA): la configuracion no es activable"
NO_OFICIAL_MSG = "[E02_NO_OFICIAL] la configuracion no procede de la ejecucion oficial vinculada a P2"


# =========================================================================== estado del registro del PROYECTO (v11)
def project_e02_approved() -> bool:
    """Estado de E02 en el registro del repositorio del PROYECTO: False = PENDIENTE_AUDITORIA (hasta A2), True =
    APROBADA (aprobacion preparatoria desde P2). Cualquier otro estado es un fallo. Las pruebas sobre el proyecto
    admiten ambos estados (como E01); las de los repositorios sinteticos no dependen de el (``build_repo``)."""
    (entry,) = json.loads((SRC / m2.REGISTRY_E02).read_text(encoding="utf-8"))["enmiendas"]
    assert entry["estado"] in (e01.ESTADO_PENDIENTE, e01.ESTADO_APROBADA), entry["estado"]
    return entry["estado"] == e01.ESTADO_APROBADA


def project_rejection(p2: str) -> str:
    """Motivo EXACTO con que el repositorio del proyecto rechaza una ejecucion oficial con una publicacion ``p2``
    inexistente: PENDIENTE -> E02_PENDIENTE; APROBADA -> la aprobacion es activable, pero P2 no existe."""
    if project_e02_approved():
        return f"[E02_P2_EXISTE] la publicacion {p2} no existe en el repositorio"
    return PENDING_MSG


def pending_registry(reg: dict) -> dict:
    """Forma PENDIENTE (la de A2) de un registro de E02: la aprobacion solo cambia ``estado`` y ``aprobacion``."""
    reg = copy.deepcopy(reg)
    (entry,) = reg["enmiendas"]
    assert entry["estado"] in (e01.ESTADO_PENDIENTE, e01.ESTADO_APROBADA), entry["estado"]
    entry.update(estado=e01.ESTADO_PENDIENTE, aprobacion=None)
    return reg


# =========================================================================== T-CMP: comparador exacto
def assert_rejects(message: str, fn, *args, **kwargs) -> None:
    """Negativa valida solo si lanza EXACTAMENTE ProtocolNotAuthorizedError con el mensaje COMPLETO esperado."""
    try:
        fn(*args, **kwargs)
    except Exception as exc:                                               # noqa: BLE001 (se comprueba el tipo exacto)
        assert type(exc) is Err, f"tipo {type(exc).__name__}: {exc}"
        assert str(exc) == message, f"motivo distinto:\n  obtenido: {exc}\n  esperado: {message}"
        return
    raise AssertionError(f"sin rechazo (esperado: {message})")


def _fails(fn) -> bool:
    try:
        fn()
    except AssertionError:
        return True
    return False


def test_comparator_accepts_only_the_exact_type_and_reason():
    def boom(exc):
        def f():
            raise exc
        return f
    msg = "[E02_FECHA] fecha '2999-01-01' no ISO valida o fuera de [fecha_preparacion, hoy]"
    assert not _fails(lambda: assert_rejects(msg, boom(Err(msg))))
    assert _fails(lambda: assert_rejects(msg, boom(Err("[E02_FECHA] fecha '18/10/2026' no ISO valida o fuera de [fecha_preparacion, hoy]"))))
    assert _fails(lambda: assert_rejects(msg, boom(Err("[E02_FECHA] otro motivo"))))
    assert _fails(lambda: assert_rejects(msg, boom(Err(msg + " (texto adicional)"))))
    assert _fails(lambda: assert_rejects(msg, boom(ValueError(msg))))
    assert _fails(lambda: assert_rejects(msg, boom(an.PendingDefinitionError(msg))))          # subclase: otro tipo
    assert _fails(lambda: assert_rejects(msg, lambda: None))


# =========================================================================== repositorio temporal A2 -> P2
def git(root, *args, check=True):
    env = {**os.environ, "GIT_AUTHOR_DATE": DATE, "GIT_COMMITTER_DATE": DATE}
    res = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.autocrlf=false",
                          "-c", "commit.gpgsign=false", *args], cwd=root, capture_output=True, env=env)
    if check and res.returncode != 0:
        raise RuntimeError(f"git {args}: {res.stderr.decode('utf-8', 'replace')}")
    return res.stdout.decode("utf-8", "replace").strip()


def data_files() -> list[str]:
    m = frz.load_manifest(SRC)
    return [r for g in ("sha256_entradas", "sha256_artefactos_previos") for r in m[g] if r.startswith("data/")]


def write_json(path: Path, obj) -> None:
    path.write_bytes((json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def approve(root: Path, a2: str, **override) -> dict:
    reg = json.loads((root / m2.REGISTRY_E02).read_text(encoding="utf-8"))
    entry = reg["enmiendas"][0]
    entry["estado"] = e01.ESTADO_APROBADA
    entry["aprobacion"] = {"fecha": APPROVAL_DATE, "aprobada_por": "PRUEBA_SINTETICA",
                           "referencia_auditoria": {"documento": "PRUEBA_SINTETICA", "commit_auditado": a2},
                           "sha256_auditados": {r: e01.sha256_text_file(root / r) for r in m2.AUDITED_E02}}
    entry["aprobacion"].update(override)
    write_json(root / m2.REGISTRY_E02, reg)
    return reg


def build_repo(dest: Path) -> dict:
    """Clon (historia real hasta P1) + ficheros de E02 del arbol de trabajo -> A2 -> aprobacion simulada -> P2,
    publicado en un remoto TEMPORAL. A2 lleva SIEMPRE el registro PENDIENTE, aunque el proyecto ya este aprobado:
    la aprobacion simulada no depende del estado del repositorio real (v11)."""
    git(dest.parent, "clone", "-q", str(SRC), dest.name)
    git(dest, "remote", "remove", "origin")                                 # nunca el repositorio del proyecto
    git(dest, "checkout", "-q", "-B", "main", m2.E01_P1)
    for rel in m2.TRACKED_E02:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SRC / rel, dest / rel)
    write_json(dest / m2.REGISTRY_E02, pending_registry(json.loads((dest / m2.REGISTRY_E02).read_text(encoding="utf-8"))))
    for rel in data_files():
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        if not (dest / rel).exists():
            os.link(SRC / rel, dest / rel)                                  # solo lectura en las pruebas
    git(dest, "add", "-A")
    git(dest, "commit", "-q", "-m", "A2: E02 auditada (sintetica)")
    a2 = git(dest, "rev-parse", "HEAD")
    approve(dest, a2)
    git(dest, "add", "-A")
    git(dest, "commit", "-q", "-m", "P2: aprobacion de E02 (sintetica)")
    p2 = git(dest, "rev-parse", "HEAD")
    bare = dest.parent / f"{dest.name}_remoto.git"
    git(dest.parent, "init", "-q", "--bare", bare.name)
    git(dest, "remote", "add", "origin", str(bare))
    git(dest, "push", "-q", "origin", "main")
    git(dest, "fetch", "-q", "origin")
    return {"root": dest, "A2": a2, "P2": p2, "bare": bare}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return build_repo(tmp_path_factory.mktemp("m2") / "r")


@pytest.fixture
def repo(built, monkeypatch):
    """El repositorio en P2 limpio (sin cambios, rama main, remoto en P2) antes de cada prueba."""
    root, p2 = built["root"], built["P2"]
    git(root, "checkout", "-q", "-f", "main")
    git(root, "reset", "-q", "--hard", p2)
    git(root, "clean", "-fdq")
    for b in git(root, "branch", "--format=%(refname:short)").split():
        if b != "main":
            git(root, "branch", "-q", "-D", b)
    git(root, "remote", "remove", "origin", check=False)                  # una prueba puede haberlo eliminado
    git(root, "remote", "add", "origin", str(built["bare"]))
    git(root, "push", "-q", "-f", "origin", f"{p2}:refs/heads/main")      # remoto TEMPORAL real en P2
    monkeypatch.setattr(m2, "AUTHORIZED_REMOTE", str(built["bare"]))       # identidad autorizada SOLO en la prueba
    git(root, "update-ref", "refs/remotes/origin/main", p2)
    for rel in data_files():
        if not (root / rel).exists():
            os.link(SRC / rel, root / rel)
    return built


def edit_json(root: Path, rel: str, fn) -> None:
    obj = json.loads((root / rel).read_text(encoding="utf-8"))
    fn(obj)
    write_json(root / rel, obj)


def append(root: Path, rel: str, text: str = "\n# cambio de prueba\n") -> None:
    with open(root / rel, "ab") as fh:
        fh.write(text.encode("utf-8"))


# =========================================================================== estado del repositorio real (sin aprobar)
def test_project_repository_keeps_e02_pending_and_e01_untouched():
    """El registro del proyecto esta PENDIENTE (hasta A2: produccion y ejecucion oficial rechazan) o APROBADO por
    una aprobacion PREPARATORIA verificable (desde P2: activable, nunca oficial sin una P2 publicada). E01 queda
    intacta en ambos estados (v11: antes solo admitia PENDIENTE)."""
    reg = json.loads((SRC / m2.REGISTRY_E02).read_text(encoding="utf-8"))
    assert list(reg) == list(m2.REGISTRY_FIELDS)
    (entry,) = reg["enmiendas"]
    assert entry["sha256"] == m2.E02_SHA256
    assert e01.sha256_text_file(SRC / m2.E02_YAML) == m2.E02_SHA256
    for rel, h in m2.E01_FILES_SHA256.items():
        assert e01.sha256_text_file(SRC / rel) == h, rel
    e01_reg = json.loads((SRC / m2.E01_REGISTRY).read_text(encoding="utf-8"))
    assert [e["id"] for e in e01_reg["enmiendas"]] == ["E01"]
    assert e01.canonical_sha256(e01_reg["enmiendas"][0]) == m2.E01_ENTRY_SHA256
    if not project_e02_approved():
        assert (entry["estado"], entry["aprobacion"]) == (e01.ESTADO_PENDIENTE, None)
        assert_rejects(PENDING_MSG, m2.load_effective_protocol_e02)
        assert_rejects(PENDING_MSG, m2.official_config_e02, None, "a" * 40)
    else:
        assert entry["estado"] == e01.ESTADO_APROBADA and set(entry["aprobacion"]) == set(e01.APPROVAL_FIELDS)
        a2 = entry["aprobacion"]["referencia_auditoria"]["commit_auditado"]
        assert re.fullmatch(r"[0-9a-f]{40}", a2), a2
        assert git(SRC, "rev-parse", f"{a2}^@") == m2.E01_P1                # A2: unico padre P1
        git(SRC, "merge-base", "--is-ancestor", a2, "HEAD")                 # HEAD = A2 o desciende de A2
        assert entry["aprobacion"]["sha256_auditados"] == {r: e01.sha256_text_file(SRC / r) for r in m2.AUDITED_E02}
        assert pending_registry(reg) == json.loads(git(SRC, "show", f"{a2}:{m2.REGISTRY_E02}"))  # A2: PENDIENTE
        prep = m2.load_effective_protocol_e02()
        tr = prep["trazabilidad_protocolo"]
        assert (tr["modo"], tr["activable"], tr["ejecucion_oficial"], tr["publicacion_E02"]) == \
            ("PRODUCCION", True, False, None)
        assert_rejects(NO_OFICIAL_MSG, m2.require_e02_oficial, prep)       # preparatoria: nunca oficial
        assert_rejects(project_rejection("a" * 40), m2.official_config_e02, None, "a" * 40)
    tr = m2.effective_document_e02(borrador=True)["trazabilidad_protocolo"]
    assert (tr["modo"], tr["activable"], tr["ejecucion_oficial"]) == ("BORRADOR", False, False)


def test_cli_verificar_on_the_project_repository(capsys):
    assert m2.main(["verificar"]) == 0
    out = capsys.readouterr().out
    estado, modo = ("APROBADA", "modo PRODUCCION; activable: SI") if project_e02_approved() else \
        ("PENDIENTE_AUDITORIA", "modo BORRADOR; activable: NO")
    assert out.startswith(f"VERIFICACION DE E02: CONFORME (protocolo 9cdea2f estricto; E01 24f9f1d; E02 {estado} ")
    assert f"{modo}; ejecucion oficial: NO" in out                         # sin --referencia-git: nunca oficial


# =========================================================================== T-WL: lista blanca
def test_whitelist_has_no_forbidden_route_and_matches_the_yaml():
    assert m2.check_whitelist_static() == []
    for bad in ("analisis.bootstrap.semilla", "analisis.bootstrap.replicas", "g2.G2_2.semillas", "g2.G2_3b.replicas",
                "identidades.verificacion_numerica.nulos_esperados", "identidades.lineales", "decisiones",
                "catalogo.A1.configuraciones", "analisis.r2.hueco_espectral_minimo", "analisis.clasificacion.umbral_r2",
                "sensibilidades.S6.fechas", "analisis", "identidades.verificacion_numerica.tau_relativa.x"):
        assert m2._forbidden(bad), bad
    for ok in (r for _, r in m2.WHITELIST_E02):
        assert not m2._forbidden(ok), ok
    doc = yaml.safe_load((SRC / m2.E02_YAML).read_text(encoding="utf-8"))
    assert sorted((o["op"], o["ruta"]) for o in doc["cambios"]) == sorted(m2.WHITELIST_E02)
    assert all(o["op"] == "anadir" for o in doc["cambios"])
    assert doc["enmienda"]["id"] == "E02" and doc["enmienda"]["requiere"] == "E01"
    assert doc["enmienda"]["protocolo_base"] == e01._base_reference()


def test_whitelist_static_check_detects_forbidden_routes(monkeypatch):
    monkeypatch.setattr(m2, "WHITELIST_E02", m2.WHITELIST_E02 + (("sustituir", "analisis.bootstrap.semilla"),))
    assert m2.check_whitelist_static() == ["ruta prohibida en la lista blanca: analisis.bootstrap.semilla",
                                           "operacion no admitida en E02: sustituir analisis.bootstrap.semilla"]


# =========================================================================== T-POS y T-OFI positivas
def test_production_load_with_e02_approved(repo):
    root, a2 = repo["root"], repo["A2"]
    cfg = m2.load_effective_protocol_e02(root)
    tr = cfg["trazabilidad_protocolo"]
    assert (tr["modo"], tr["activable"], tr["ejecucion_oficial"], tr["publicacion_E02"]) == ("PRODUCCION", True, False, None)
    e1, e2 = tr["enmiendas"]
    assert (e1["id"], e1["estado"], e1["aprobacion"]["referencia_auditoria"]["commit_auditado"]) == ("E01", "APROBADA", m2.E01_A1)
    assert (e2["id"], e2["estado"], e2["aprobacion"]["referencia_auditoria"]["commit_auditado"]) == ("E02", "APROBADA", a2)
    assert tr["sha256_registro_E02"] == e01.sha256_text_file(root / m2.REGISTRY_E02)
    base = e01.load_effective_protocol(root)
    strip = lambda c: {k: v for k, v in c.items() if k != "trazabilidad_protocolo"}   # noqa: E731
    assert e01.diff_paths(strip(base), strip(cfg)) == {r for _, r in m2.WHITELIST_E02}
    assert tr["sha256_efectiva"] == e01.canonical_sha256(strip(cfg))
    m2.require_e02(cfg)
    assert_rejects("[E02_NO_OFICIAL] la configuracion no procede de la ejecucion oficial vinculada a P2",
                   m2.require_e02_oficial, cfg, root)
    assert an.production_config(root)["trazabilidad_protocolo"]["activable"] is True


def test_official_execution_at_p2_and_after_unrelated_commits(repo, tmp_path):
    root, p2 = repo["root"], repo["P2"]
    cfg = m2.official_config_e02(root, p2)
    tr = cfg["trazabilidad_protocolo"]
    assert (tr["ejecucion_oficial"], tr["publicacion_E02"], tr["remoto_E02"]) == (True, p2, {"repositorio": str(repo["bare"]), "ref": "refs/heads/main", "sha": p2})
    m2.require_e02_oficial(cfg, root)
    assert an.official_config(p2, root)["trazabilidad_protocolo"]["ejecucion_oficial"] is True
    # v5: no existe ninguna primitiva que reciba una tabla; las tablas fuera del esquema minimo siguen bloqueadas y
    # las del esquema solo se publican por la ruta unica (que no se ejecuta aqui: leeria las entradas reales)
    assert not hasattr(sal, "write_official_table")
    with pytest.raises(an.PendingDefinitionError) as info:
        sal.other_official_table("TDA17_catalogo.csv")
    assert type(info.value) is an.PendingDefinitionError and str(info.value) == (
        f"SALIDAS_OTRAS_TABLAS: definicion pendiente -- {an.PENDIENTES['SALIDAS_OTRAS_TABLAS']}")
    assert_rejects("[SALIDA_RUTA_UNICA] TDA17_clasificacion_final.csv solo se publica con official_publish_th35",
                   sal.other_official_table, "TDA17_clasificacion_final.csv")
    assert not (root / "reports" / "mnq" / "TDA17_clasificacion_final.csv").exists()
    (root / "notas_posteriores.txt").write_bytes(b"cambio ajeno a E01 y E02\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "posterior")
    assert m2.official_config_e02(root, p2)["trazabilidad_protocolo"]["publicacion_E02"] == p2
    with pytest.raises(an.PendingDefinitionError, match="^K_PA_MAX_T_V3: "):                # pasa la ruta oficial y
        g2.official_g2_controls(p2, root)                                                     # se detiene en lo pendiente
    errors, info = m2.verify(root, p2)
    assert errors == [] and info["ejecucion_oficial"] is True


def test_preparatory_approval_at_a2_is_never_official(repo):
    root, a2, p2 = repo["root"], repo["A2"], repo["P2"]
    git(root, "checkout", "-q", "--detach", a2)
    assert_rejects(PENDING_MSG, m2.load_effective_protocol_e02, root)                     # registro de A2: pendiente
    tr = m2.effective_document_e02(root, borrador=True)["trazabilidad_protocolo"]
    assert (tr["modo"], tr["activable"], tr["ejecucion_oficial"]) == ("BORRADOR", False, False)
    approve(root, a2)                                                                     # aprobacion aun sin publicar
    tr = m2.load_effective_protocol_e02(root)["trazabilidad_protocolo"]
    assert (tr["activable"], tr["ejecucion_oficial"], tr["publicacion_E02"]) == (True, False, None)
    assert_rejects("[E02_P2_HEAD] la publicacion P2 no es HEAD ni antecesora de HEAD", m2.official_config_e02, root, p2)


def test_reproducible_a2_and_p2(repo, tmp_path):
    other = build_repo(tmp_path / "r")
    assert (other["A2"], other["P2"]) == (repo["A2"], repo["P2"])


def test_synthetic_a2_registry_never_depends_on_the_project_state(repo):
    """v11: la A2 sintetica lleva SIEMPRE el registro PENDIENTE y la aprobacion simulada cita la A2 sintetica, este
    el proyecto PENDIENTE o APROBADO (su registro solo difiere en ``estado`` y ``aprobacion``)."""
    root, a2, p2 = repo["root"], repo["A2"], repo["P2"]
    reg_a2 = json.loads(git(root, "show", f"{a2}:{m2.REGISTRY_E02}"))
    reg_p2 = json.loads(git(root, "show", f"{p2}:{m2.REGISTRY_E02}"))
    (entry,) = reg_a2["enmiendas"]
    assert (entry["estado"], entry["aprobacion"]) == (e01.ESTADO_PENDIENTE, None)
    assert pending_registry(json.loads((SRC / m2.REGISTRY_E02).read_text(encoding="utf-8"))) == reg_a2
    assert pending_registry(reg_p2) == reg_a2
    aprobacion = reg_p2["enmiendas"][0]["aprobacion"]
    assert (aprobacion["aprobada_por"], aprobacion["referencia_auditoria"]["commit_auditado"]) == ("PRUEBA_SINTETICA", a2)
    otro = copy.deepcopy(reg_p2)
    otro["enmiendas"][0]["estado"] = "OTRO"
    assert _fails(lambda: pending_registry(otro))                          # ningun otro estado se normaliza


# =========================================================================== C1: configuracion oficial no falsificable
def test_forged_official_configs_are_rejected(repo):
    """Regresion C1: indicadores de trazabilidad fabricados (APROBADA, activable, ejecucion_oficial, P2 valido) no
    pasan ``require_e02_oficial``: la configuracion se deriva de nuevo con M2 y debe ser identica. Desde v5 la
    publicacion ni siquiera recibe una configuracion (ruta unica con P2; tda17_salidas)."""
    root, p2, a2 = repo["root"], repo["P2"], repo["A2"]
    distinta = "[E02_OFICIAL_DISTINTA] la configuracion recibida no es identica a la ejecucion oficial derivada de M2"
    ofi = m2.official_config_e02(root, p2)
    m2.require_e02_oficial(ofi, root)
    altered = copy.deepcopy(ofi)                                            # contenido alterado, indicadores intactos
    altered["analisis"]["clasificacion"]["umbral_r2"] = 0.5
    assert_rejects(distinta, m2.require_e02_oficial, altered, root)
    prep = m2.load_effective_protocol_e02(root)                             # preparatoria con indicadores fabricados
    prep["trazabilidad_protocolo"].update(ejecucion_oficial=True, publicacion_E02=p2)
    assert_rejects(distinta, m2.require_e02_oficial, prep, root)
    forged = copy.deepcopy(ofi)                                             # indicadores coherentes, P2 = A2
    forged["trazabilidad_protocolo"]["publicacion_E02"] = a2
    assert_rejects("[E02_OFICIAL_NO_VERIFICABLE] la configuracion oficial no se puede derivar de M2: "
                   "[E02_P2_A2] el commit auditado A2 debe ser antecesor estricto de la publicacion P2",
                   m2.require_e02_oficial, forged, root)
    _pending(root)                                                          # configuracion obsoleta: E02 ya no aprobada
    assert_rejects(f"[E02_OFICIAL_NO_VERIFICABLE] la configuracion oficial no se puede derivar de M2: {PENDING_MSG}",
                   m2.require_e02_oficial, ofi, root)
    assert_rejects(PENDING_MSG, sal.official_publish_th35, p2, root)
    assert list(inspect.signature(sal.official_publish_th35).parameters) == ["publicacion", "root"]


# =========================================================================== T-E01: preservacion de E01
def test_e01_is_preserved_in_the_m2_repository(repo):
    root, p2 = repo["root"], repo["P2"]
    assert subprocess.run(["git", "diff", "--quiet", m2.E01_P1, p2, "--", *m2.E01_FILES_SHA256], cwd=root).returncode == 0
    assert e01.verify_against_git(root, p2) == []
    errors, info = e01.verify(root, p2)
    assert errors == [] and info["activable"] is True
    eff = e01.load_effective_protocol(root)
    assert eff["trazabilidad_protocolo"]["enmiendas"][0]["aprobacion"]["referencia_auditoria"]["commit_auditado"] == m2.E01_A1
    assert m2.E01_P1 in git(root, "rev-list", p2) and repo["A2"] in git(root, "rev-list", p2)


# =========================================================================== T-GIT-P2
def test_verify_against_git_e02(repo):
    root, p2 = repo["root"], repo["P2"]
    assert m2.verify_against_git_e02(root, p2) == []
    append(root, "tests/test_tda17_causalidad.py")
    assert m2.verify_against_git_e02(root, p2) == [f"[git {p2}] difiere del commit de referencia: tests/test_tda17_causalidad.py"]
    ghost = "0" * 40
    errs = m2.verify_against_git_e02(root, ghost)
    assert errs == [f"[git {ghost}] no contiene {r}" for r in list(m2.E01_FILES_SHA256) + m2.TRACKED_E02]


# =========================================================================== T-NEG: una negativa por control
def _monkey_e01_registry(monkeypatch, root, fn):
    """Altera el registro de E01 en el arbol y fija su nueva huella en el cargador (supera E01_FICHEROS)."""
    edit_json(root, m2.E01_REGISTRY, fn)
    monkeypatch.setitem(m2.E01_FILES_SHA256, m2.E01_REGISTRY, e01.sha256_text_file(root / m2.E01_REGISTRY))


def _monkey_yaml(monkeypatch, root, fn):
    """Altera el YAML de E02 y fija su huella en el cargador y en el registro (supera E02_HUELLA)."""
    doc = yaml.safe_load((root / m2.E02_YAML).read_text(encoding="utf-8"))
    fn(doc)
    (root / m2.E02_YAML).write_bytes(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False).encode("utf-8"))
    sha = e01.sha256_text_file(root / m2.E02_YAML)
    monkeypatch.setattr(m2, "E02_SHA256", sha)
    edit_json(root, m2.REGISTRY_E02, lambda r: r["enmiendas"][0].update(sha256=sha))
    return sha


def _pending(root):
    edit_json(root, m2.REGISTRY_E02, lambda r: r["enmiendas"][0].update(estado=e01.ESTADO_PENDIENTE, aprobacion=None))


def _approval_field(**kv):
    return lambda r: r["enmiendas"][0]["aprobacion"].update(**kv)


def _ref(**kv):
    return lambda r: r["enmiendas"][0]["aprobacion"]["referencia_auditoria"].update(**kv)


def _neg_e01_ficheros(ctx, mp):
    append(ctx["root"], "tests/tda17_synthetic.py")
    return "[E01_FICHEROS] fichero auditado de E01 alterado en el arbol: tests/tda17_synthetic.py"


def _neg_e01_bytes(ctx, mp):
    root = ctx["root"]
    rel = "tests/tda17_synthetic.py"
    orig = (root / rel).read_bytes()
    append(root, rel)
    git(root, "commit", "-q", "-am", "cambia E01")
    (root / rel).write_bytes(orig)                                          # arbol restaurado, cambio confirmado
    return "[E01_BYTES] HEAD cambia ficheros de E01 respecto de su publicacion P1"


def _neg_e01_historia(ctx, mp):
    root = ctx["root"]
    orphan = git(root, "commit-tree", f"{ctx['P2']}^{{tree}}", "-m", "sin historia")
    git(root, "checkout", "-q", "--detach", orphan)
    return "[E01_HISTORIA] la publicacion P1 de E01 no es antecesora de HEAD"


def _neg_e01_registro(ctx, mp):
    def add(r):
        r["enmiendas"].append(copy.deepcopy(r["enmiendas"][0]) | {"id": "E02"})
    _monkey_e01_registry(mp, ctx["root"], add)
    return "[E01_REGISTRO] el registro de E01 debe contener solo E01: ['E01', 'E02']"


def _neg_e01_entrada(ctx, mp):
    _monkey_e01_registry(mp, ctx["root"], lambda r: r["enmiendas"][0].update(alcance="alterado"))
    return "[E01_ENTRADA] la entrada E01 del registro no es la publicada"


def _neg_e01_a1(ctx, mp):
    def cite(r):
        r["enmiendas"][0]["aprobacion"]["referencia_auditoria"]["commit_auditado"] = m2.E01_P1
    _monkey_e01_registry(mp, ctx["root"], cite)
    entry = json.loads((ctx["root"] / m2.E01_REGISTRY).read_text(encoding="utf-8"))["enmiendas"][0]
    mp.setattr(m2, "E01_ENTRY_SHA256", e01.canonical_sha256(entry))
    return "[E01_A1] la aprobacion de E01 no cita su commit auditado A1"


def _neg_e02_registro(ctx, mp):
    (ctx["root"] / m2.REGISTRY_E02).unlink()
    return "[E02_REGISTRO] falta el registro de E02"


def _neg_e02_registro_json(ctx, mp):
    (ctx["root"] / m2.REGISTRY_E02).write_bytes(b"{no es json")
    return "[E02_REGISTRO] el registro de E02 no es JSON valido"


def _neg_e02_reg_estructura(ctx, mp):
    edit_json(ctx["root"], m2.REGISTRY_E02, lambda r: r.update(extra=1))
    return "[E02_REG_ESTRUCTURA] el registro de E02 debe tener exactamente ['descripcion', 'protocolo_base', 'requiere', 'enmiendas']"


def _neg_e02_base(ctx, mp):
    edit_json(ctx["root"], m2.REGISTRY_E02, lambda r: r["protocolo_base"].update(commit="0" * 40))
    return "[E02_BASE] el registro de E02 no declara el protocolo base congelado"


def _neg_e02_requiere(ctx, mp):
    edit_json(ctx["root"], m2.REGISTRY_E02, lambda r: r["requiere"].update(publicacion_E01=m2.E01_A1))
    return "[E02_REQUIERE] el registro de E02 no declara la E01 publicada (entrada, A1, P1)"


def _neg_e02_entrada(ctx, mp):
    edit_json(ctx["root"], m2.REGISTRY_E02, lambda r: r["enmiendas"].append(copy.deepcopy(r["enmiendas"][0])))
    return "[E02_ENTRADA] el registro de E02 debe tener una unica entrada E02 bien formada"


def _neg_e02_entrada_id(ctx, mp):
    edit_json(ctx["root"], m2.REGISTRY_E02, lambda r: r["enmiendas"][0].update(id="E03"))
    return "[E02_ENTRADA] el registro de E02 debe tener una unica entrada E02 bien formada"


def _neg_e02_huella(ctx, mp):
    append(ctx["root"], m2.E02_YAML)
    return f"[E02_HUELLA] {m2.E02_YAML} no es la version autorizada de E02 (SHA-256 {e01.sha256_text_file(ctx['root'] / m2.E02_YAML)})"


def _neg_e02_huella_registro(ctx, mp):
    edit_json(ctx["root"], m2.REGISTRY_E02, lambda r: r["enmiendas"][0].update(sha256="0" * 64))
    return f"[E02_HUELLA] {m2.E02_YAML} no es la version autorizada de E02 (SHA-256 {m2.E02_SHA256})"


def _neg_e02_doc(ctx, mp):
    _monkey_yaml(mp, ctx["root"], lambda d: d["enmienda"].update(requiere="E00"))
    return "[E02_DOC] el YAML de E02 no declara id E02, el protocolo base y requiere E01"


def _neg_e02_estado(ctx, mp):
    edit_json(ctx["root"], m2.REGISTRY_E02, lambda r: r["enmiendas"][0].update(estado="APROBADA_PARCIAL"))
    return "[E02_ESTADO] estado desconocido 'APROBADA_PARCIAL'"


def _neg_e02_pendiente(ctx, mp):
    _pending(ctx["root"])
    return PENDING_MSG


def _neg_e02_campos_falta(ctx, mp):
    edit_json(ctx["root"], m2.REGISTRY_E02, lambda r: r["enmiendas"][0]["aprobacion"].pop("fecha"))
    return "[E02_CAMPOS] la aprobacion debe tener exactamente ['fecha', 'aprobada_por', 'referencia_auditoria', 'sha256_auditados']"


def _neg_e02_campos_sobra(ctx, mp):
    edit_json(ctx["root"], m2.REGISTRY_E02, _approval_field(firma="x"))
    return "[E02_CAMPOS] la aprobacion debe tener exactamente ['fecha', 'aprobada_por', 'referencia_auditoria', 'sha256_auditados']"


def _fecha(value):
    def build(ctx, mp):
        edit_json(ctx["root"], m2.REGISTRY_E02, _approval_field(fecha=value))
        return f"[E02_FECHA] fecha {value!r} no ISO valida o fuera de [fecha_preparacion, hoy]"
    return build


def _neg_e02_responsable(ctx, mp):
    edit_json(ctx["root"], m2.REGISTRY_E02, _approval_field(aprobada_por="   "))
    return "[E02_RESPONSABLE] aprobada_por vacio (responsable declarado obligatorio)"


def _neg_e02_referencia(ctx, mp):
    edit_json(ctx["root"], m2.REGISTRY_E02, _ref(documento=""))
    return "[E02_REFERENCIA] referencia_auditoria debe tener documento no vacio y commit_auditado"


def _commit_fmt(value_fn):
    def build(ctx, mp):
        edit_json(ctx["root"], m2.REGISTRY_E02, _ref(commit_auditado=value_fn(ctx)))
        return "[E02_COMMIT_FORMATO] commit_auditado no es un SHA-1 Git completo (40 hexadecimales)"
    return build


def _neg_e02_cobertura_falta(ctx, mp):
    edit_json(ctx["root"], m2.REGISTRY_E02, lambda r: r["enmiendas"][0]["aprobacion"]["sha256_auditados"].pop("src/ohlcv_dataroad/ingest/tda17_g2.py"))
    return "[E02_COBERTURA] sha256_auditados no cubre exactamente los ficheros auditados de E02"


def _neg_e02_cobertura_sobra(ctx, mp):
    edit_json(ctx["root"], m2.REGISTRY_E02, lambda r: r["enmiendas"][0]["aprobacion"]["sha256_auditados"].update({"x.py": "0" * 64}))
    return "[E02_COBERTURA] sha256_auditados no cubre exactamente los ficheros auditados de E02"


def _neg_e02_arbol(ctx, mp):
    append(ctx["root"], "src/ohlcv_dataroad/ingest/tda17_g2.py")
    return "[E02_ARBOL] ficheros distintos de la version auditada de E02: ['src/ohlcv_dataroad/ingest/tda17_g2.py']"


def _neg_e02_commit_inexistente(ctx, mp):
    edit_json(ctx["root"], m2.REGISTRY_E02, _ref(commit_auditado="0" * 40))
    return f"[E02_COMMIT] el commit auditado {'0' * 40} no existe o no contiene la version aprobada de {m2.AUDITED_E02}"


def _neg_e02_commit_sin_ficheros(ctx, mp):
    edit_json(ctx["root"], m2.REGISTRY_E02, _ref(commit_auditado=m2.E01_P1))
    return f"[E02_COMMIT] el commit auditado {m2.E01_P1} no existe o no contiene la version aprobada de {m2.AUDITED_E02}"


def _neg_e02_ascendencia(ctx, mp):
    root = ctx["root"]
    orphan = git(root, "commit-tree", f"{ctx['A2']}^{{tree}}", "-m", "A2 sin historia")
    edit_json(root, m2.REGISTRY_E02, _ref(commit_auditado=orphan))
    return "[E02_ASCENDENCIA] el commit auditado de E02 no desciende de la publicacion P1 de E01"


def _neg_e02_a2_head(ctx, mp):
    root = ctx["root"]
    git(root, "checkout", "-q", "-b", "paralela", m2.E01_P1)
    git(root, "checkout", "-q", ctx["P2"], "--", ".")
    git(root, "commit", "-q", "-m", "arbol de P2 sin descender de A2")
    return "[E02_A2_HEAD] HEAD no desciende del commit auditado A2 de E02"


def _neg_e02_lista_blanca(ctx, mp):
    _pending(ctx["root"])
    _monkey_yaml(mp, ctx["root"], lambda d: d["cambios"].pop())
    return "[E02_LISTA_BLANCA] operaciones de E02 distintas de su lista blanca"


def _neg_e02_lista_blanca_extra(ctx, mp):
    _pending(ctx["root"])
    _monkey_yaml(mp, ctx["root"], lambda d: d["cambios"].append({"op": "anadir", "ruta": "analisis.extra", "valor": 1}))
    return "[E02_LISTA_BLANCA] operaciones de E02 distintas de su lista blanca"


def _neg_e02_anadir_existente(ctx, mp):
    _pending(ctx["root"])

    def swap(d):
        d["cambios"][-1]["ruta"] = "analisis.bootstrap.semilla"
    _monkey_yaml(mp, ctx["root"], swap)
    mp.setattr(m2, "WHITELIST_E02", m2.WHITELIST_E02[:-1] + (("anadir", "analisis.bootstrap.semilla"),))
    return "[E02_ANADIR] operacion no admitida o 'anadir' sobre clave existente: analisis.bootstrap.semilla"


def _neg_e02_anadir_sustituir(ctx, mp):
    _pending(ctx["root"])

    def swap(d):
        d["cambios"][-1]["op"] = "sustituir"
    _monkey_yaml(mp, ctx["root"], swap)
    last = m2.WHITELIST_E02[-1][1]                                          # ruta de la ultima operacion del YAML
    mp.setattr(m2, "WHITELIST_E02", m2.WHITELIST_E02[:-1] + (("sustituir", last),))
    return f"[E02_ANADIR] operacion no admitida o 'anadir' sobre clave existente: {last}"


def _neg_e02_anadir_ruta(ctx, mp):
    _pending(ctx["root"])

    def swap(d):
        d["cambios"][-1]["ruta"] = "noexiste.clave"
    _monkey_yaml(mp, ctx["root"], swap)
    mp.setattr(m2, "WHITELIST_E02", m2.WHITELIST_E02[:-1] + (("anadir", "noexiste.clave"),))
    return "[E02_ANADIR] ruta no aplicable: noexiste.clave (ruta inexistente en el protocolo base: noexiste.clave)"


def _neg_git_no_ejecutable(ctx, mp):
    def boom(*a, **k):
        raise FileNotFoundError("git no instalado")
    mp.setattr(e01.subprocess, "run", boom)
    return "[GIT] sin acceso a Git: FileNotFoundError: git no instalado"


def _neg_git_tiempo(ctx, mp):
    def slow(*a, **k):
        raise subprocess.TimeoutExpired("git", 60)
    mp.setattr(e01.subprocess, "run", slow)
    return "[GIT] sin acceso a Git: TimeoutExpired: Command 'git' timed out after 60 seconds"


def _neg_git_lectura(ctx, mp):
    real = e01.subprocess.run

    def flaky(cmd, *a, **k):
        if cmd[:2] == ["git", "diff"]:
            return subprocess.CompletedProcess(cmd, 128, b"", b"fatal: error de lectura")
        return real(cmd, *a, **k)
    mp.setattr(e01.subprocess, "run", flaky)
    return "[GIT] error de Git al comparar los ficheros de E01 con su publicacion P1"


def _neg_git_ruta_larga(ctx, mp):
    real = e01.subprocess.run

    def long_path(cmd, *a, **k):
        if cmd[:2] == ["git", "show"] and "tda17_enmienda_02.yaml" in cmd[2]:
            return subprocess.CompletedProcess(cmd, 128, b"", b"error: unable to create file: Filename too long")
        return real(cmd, *a, **k)
    mp.setattr(e01.subprocess, "run", long_path)
    return f"[E02_COMMIT] el commit auditado {ctx['A2']} no existe o no contiene la version aprobada de ['{m2.E02_YAML}']"


NEGATIVES = {
    "E01_FICHEROS": _neg_e01_ficheros, "E01_BYTES": _neg_e01_bytes, "E01_HISTORIA": _neg_e01_historia,
    "E01_REGISTRO": _neg_e01_registro, "E01_ENTRADA": _neg_e01_entrada, "E01_A1": _neg_e01_a1,
    "E02_REGISTRO-falta": _neg_e02_registro, "E02_REGISTRO-json": _neg_e02_registro_json,
    "E02_REG_ESTRUCTURA": _neg_e02_reg_estructura, "E02_BASE": _neg_e02_base, "E02_REQUIERE": _neg_e02_requiere,
    "E02_ENTRADA-dos": _neg_e02_entrada, "E02_ENTRADA-id": _neg_e02_entrada_id,
    "E02_HUELLA-yaml": _neg_e02_huella, "E02_HUELLA-registro": _neg_e02_huella_registro,
    "E02_DOC": _neg_e02_doc, "E02_ESTADO": _neg_e02_estado, "E02_PENDIENTE": _neg_e02_pendiente,
    "E02_CAMPOS-falta": _neg_e02_campos_falta, "E02_CAMPOS-sobra": _neg_e02_campos_sobra,
    "E02_FECHA-futura": _fecha("2999-01-01"), "E02_FECHA-anterior": _fecha("2026-10-17"),
    "E02_FECHA-no_iso": _fecha("18/10/2026"), "E02_FECHA-inexistente": _fecha("2026-02-30"),
    "E02_RESPONSABLE": _neg_e02_responsable, "E02_REFERENCIA": _neg_e02_referencia,
    "E02_COMMIT_FORMATO-abreviado": _commit_fmt(lambda c: c["A2"][:12]),
    "E02_COMMIT_FORMATO-no_hex": _commit_fmt(lambda c: "g" * 40),
    "E02_COBERTURA-falta": _neg_e02_cobertura_falta, "E02_COBERTURA-sobra": _neg_e02_cobertura_sobra,
    "E02_ARBOL": _neg_e02_arbol, "E02_COMMIT-inexistente": _neg_e02_commit_inexistente,
    "E02_COMMIT-sin_ficheros": _neg_e02_commit_sin_ficheros, "E02_ASCENDENCIA": _neg_e02_ascendencia,
    "E02_A2_HEAD": _neg_e02_a2_head,
    "GIT-no_ejecutable": _neg_git_no_ejecutable, "GIT-tiempo": _neg_git_tiempo, "GIT-lectura": _neg_git_lectura,
    "GIT-ruta_larga": _neg_git_ruta_larga,
}
BORRADOR_NEGATIVES = {
    "E02_LISTA_BLANCA-falta": _neg_e02_lista_blanca, "E02_LISTA_BLANCA-sobra": _neg_e02_lista_blanca_extra,
    "E02_ANADIR-existente": _neg_e02_anadir_existente, "E02_ANADIR-sustituir": _neg_e02_anadir_sustituir,
    "E02_ANADIR-ruta": _neg_e02_anadir_ruta,
}


@pytest.mark.parametrize("case", list(NEGATIVES), ids=list(NEGATIVES))
def test_negative_production_load(repo, monkeypatch, case):
    message = NEGATIVES[case](repo, monkeypatch)
    assert_rejects(message, m2.load_effective_protocol_e02, repo["root"])


@pytest.mark.parametrize("case", list(BORRADOR_NEGATIVES), ids=list(BORRADOR_NEGATIVES))
def test_negative_whitelist_in_borrador(repo, monkeypatch, case):
    message = BORRADOR_NEGATIVES[case](repo, monkeypatch)
    assert_rejects(message, m2.effective_document_e02, repo["root"], borrador=True)


def test_negative_pending_with_approval_data_in_borrador(repo):
    edit_json(repo["root"], m2.REGISTRY_E02, lambda r: r["enmiendas"][0].update(estado=e01.ESTADO_PENDIENTE))
    assert_rejects("[E02_PENDIENTE_CON_DATOS] E02 PENDIENTE con datos de aprobacion",
                   m2.effective_document_e02, repo["root"], borrador=True)


def test_negative_paths_barrier_is_independent(repo):
    base = {"analisis": {"bootstrap": {"replicas": 300}}}
    for _, ruta in m2.WHITELIST_E02:                                       # contenedores existentes en la base
        node = base
        for seg in ruta.split(".")[:-1]:
            node = node.setdefault(seg, {})
    new = copy.deepcopy(base)
    for _, ruta in m2.WHITELIST_E02:
        node = new
        for seg in ruta.split(".")[:-1]:
            node = node[seg]
        node[ruta.split(".")[-1]] = 1
    m2.check_paths(base, new)
    new["analisis"]["bootstrap"]["replicas"] = 301
    assert_rejects("[E02_RUTAS] la configuracion efectiva no cambia exactamente las rutas autorizadas: "
                   "['analisis.bootstrap.replicas']", m2.check_paths, base, new)


def test_negative_git_without_repository_and_parent_repository(repo, tmp_path):
    bare_dir = tmp_path / "sin_repo"
    bare_dir.mkdir()
    assert_rejects(f"[GIT] {bare_dir} no es la raiz de un repositorio Git", m2.load_effective_protocol_e02, bare_dir)
    sub = repo["root"] / "configs"
    assert_rejects(f"[GIT] {sub} no es la raiz de un repositorio Git", m2.load_effective_protocol_e02, sub)


# =========================================================================== T-OFI negativas
def test_negative_official_p2_format_and_existence(repo):
    root, p2, a2 = repo["root"], repo["P2"], repo["A2"]
    for bad in (None, p2[:12], p2.upper(), "g" * 40):
        assert_rejects("[E02_P2_FORMATO] la ejecucion oficial exige el SHA-1 completo de la publicacion P2",
                       m2.official_config_e02, root, bad)
    ghost = "0" * 40
    assert_rejects(f"[E02_P2_EXISTE] la publicacion {ghost} no existe en el repositorio", m2.official_config_e02, root, ghost)
    for not_after_a2 in (a2, m2.E01_P1):
        assert_rejects("[E02_P2_A2] el commit auditado A2 debe ser antecesor estricto de la publicacion P2",
                       m2.official_config_e02, root, not_after_a2)


def test_negative_official_branch_from_a2_with_the_p2_tree(repo):
    root, p2, a2 = repo["root"], repo["P2"], repo["A2"]
    git(root, "checkout", "-q", "-b", "desde_a2", a2)
    git(root, "checkout", "-q", p2, "--", ".")
    git(root, "commit", "-q", "-m", "arbol de P2 sin descender de P2")
    assert m2.load_effective_protocol_e02(root)["trazabilidad_protocolo"]["activable"] is True
    assert_rejects("[E02_P2_HEAD] la publicacion P2 no es HEAD ni antecesora de HEAD", m2.official_config_e02, root, p2)


def test_negative_official_tree_differs_from_p2(repo):
    root, p2 = repo["root"], repo["P2"]
    edit_json(root, m2.REGISTRY_E02, lambda r: r.update(descripcion=r["descripcion"] + " (posterior)"))
    git(root, "commit", "-q", "-am", "registro cambiado tras P2")
    assert m2.load_effective_protocol_e02(root)["trazabilidad_protocolo"]["activable"] is True
    assert_rejects(f"[E02_P2_ARBOL] el arbol no coincide con la publicacion P2: "
                   f"['[git {p2}] difiere del commit de referencia: {m2.REGISTRY_E02}']", m2.official_config_e02, root, p2)


def test_negative_official_publication_without_the_approval(repo, monkeypatch):
    root, a2 = repo["root"], repo["A2"]
    approved = (root / m2.REGISTRY_E02).read_bytes()
    git(root, "checkout", "-q", "-b", "p2_sin_aprobacion", a2)
    (root / "acta.md").write_bytes(b"acta sin aprobacion\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "P2 sin aprobacion")
    p2bad = git(root, "rev-parse", "HEAD")
    (root / m2.REGISTRY_E02).write_bytes(approved)
    git(root, "commit", "-q", "-am", "aprobacion despues de P2")
    monkeypatch.setattr(m2, "verify_against_git_e02", lambda r, c: [])     # aisla la barrera (redundante con P2_ARBOL)
    assert_rejects("[E02_P2_APROBADA] la publicacion P2 no contiene la aprobacion de E02 que cita A2",
                   m2.official_config_e02, root, p2bad)


def test_negative_official_strict_protocol_verification(repo):
    root, p2 = repo["root"], repo["P2"]
    rel = "data/interim/mnq/tda04_variables_1m.parquet"
    (root / rel).unlink()                                                   # quita el enlace, no el original
    try:
        assert_rejects(f"[PROTOCOLO_ESTRICTO] verificacion estricta del protocolo NO CONFORME: ['[sha256_entradas] FALTA: {rel}']",
                       m2.official_config_e02, root, p2)
    finally:
        os.link(SRC / rel, root / rel)
    art = "reports/mnq/TDA07_momentos_cuantiles_drift_colas.csv"
    append(root, art, "x\n")
    assert_rejects(f"[PROTOCOLO_ESTRICTO] verificacion estricta del protocolo NO CONFORME: ['[sha256_entradas] HASH DISTINTO: {art}']",
                   m2.official_config_e02, root, p2)


def test_negative_official_protocol_against_the_freeze_commit(repo, monkeypatch):
    monkeypatch.setattr(m2.frz, "verify_against_git", lambda r, c: [f"[git {c}] difiere del commit de referencia: configs/tda17.yaml"])
    assert_rejects(f"[PROTOCOLO_GIT] el protocolo no coincide con el commit de congelamiento: "
                   f"['[git {m2.ind.FROZEN_COMMIT}] difiere del commit de referencia: configs/tda17.yaml']",
                   m2.official_config_e02, repo["root"], repo["P2"])


def test_negative_official_remote_local_ref_says_published_but_real_remote_does_not(repo):
    """C1 (v3): la referencia local refs/remotes/origin/main indica P2 publicada, pero el remoto REAL esta en A2."""
    root, p2, a2 = repo["root"], repo["P2"], repo["A2"]
    git(root, "push", "-q", "-f", "origin", f"{a2}:refs/heads/main")       # remoto temporal real: sin P2
    git(root, "update-ref", "refs/remotes/origin/main", p2)                # referencia local desactualizada
    assert git(root, "rev-parse", "refs/remotes/origin/main") == p2
    assert_rejects(f"[E02_REMOTO] la publicacion P2 {p2} no esta en la rama publicada del remoto ({a2})",
                   m2.official_config_e02, root, p2)


def test_negative_official_remote_unavailable_or_without_branch(repo, monkeypatch):
    root, p2, a2 = repo["root"], repo["P2"], repo["A2"]
    msg = "[E02_REMOTO] no se pudo consultar la rama publicada refs/heads/main del repositorio autorizado"
    git(root, "push", "-q", "origin", ":refs/heads/main")                  # rama publicada borrada (remoto temporal)
    assert_rejects(msg, m2.official_config_e02, root, p2)
    git(root, "push", "-q", "-f", "origin", f"{p2}:refs/heads/main")
    gone = str(root.parent / "no_existe.git")                              # repositorio autorizado inaccesible
    monkeypatch.setattr(m2, "AUTHORIZED_REMOTE", gone)
    git(root, "remote", "set-url", "origin", gone)
    assert_rejects(msg, m2.official_config_e02, root, p2)
    assert_rejects(msg, m2.verify_remote_publication, root, p2, a2)


def test_negative_official_remote_is_not_the_authorized_repository(repo, tmp_path):
    """Remoto DISTINTO del autorizado con el mismo contenido (P2 publicado): se rechaza por identidad."""
    root, p2, a2 = repo["root"], repo["P2"], repo["A2"]
    mirror = tmp_path / "espejo.git"
    git(tmp_path, "clone", "-q", "--bare", str(repo["bare"]), mirror.name)
    assert git(mirror, "rev-parse", "refs/heads/main") == p2
    git(root, "remote", "set-url", "origin", str(mirror))
    assert_rejects(f"[E02_REMOTO_NO_AUTORIZADO] origin ({mirror}) no es el repositorio autorizado {m2.AUTHORIZED_REMOTE}",
                   m2.official_config_e02, root, p2)
    git(root, "remote", "remove", "origin")
    assert_rejects(f"[E02_REMOTO_NO_AUTORIZADO] origin (inexistente) no es el repositorio autorizado {m2.AUTHORIZED_REMOTE}",
                   m2.verify_remote_publication, root, p2, a2)


def test_negative_official_remote_url_rewriting(repo, tmp_path):
    """Una regla url.<espejo>.insteadOf=<autorizado> redirigiria la consulta: se rechaza antes de consultar."""
    root, p2 = repo["root"], repo["P2"]
    mirror = tmp_path / "espejo.git"
    git(tmp_path, "clone", "-q", "--bare", str(repo["bare"]), mirror.name)
    git(root, "config", f"url.{mirror}.insteadOf", m2.AUTHORIZED_REMOTE)
    try:
        assert_rejects("[E02_REMOTO_NO_AUTORIZADO] la configuracion de Git redirige URL (url.*.insteadOf): no se puede "
                       "garantizar la identidad del repositorio autorizado", m2.official_config_e02, root, p2)
    finally:
        git(root, "config", "--unset", f"url.{mirror}.insteadOf")


def test_authorized_remote_is_the_project_github_repository():
    assert m2.AUTHORIZED_REMOTE == "https://github.com/GUNAPILLCO/ohlcv_dataroad.git"
    url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=SRC, capture_output=True, text=True).stdout.strip()
    assert m2.normalize_remote(url) == m2.normalize_remote(m2.AUTHORIZED_REMOTE)    # sin red: solo la configuracion
    for same in ("git@github.com:GUNAPILLCO/ohlcv_dataroad.git", "https://github.com/gunapillco/ohlcv_dataroad/"):
        assert m2.normalize_remote(same) == "https://github.com/gunapillco/ohlcv_dataroad"
    assert m2.normalize_remote("https://github.com/otro/ohlcv_dataroad.git") != m2.normalize_remote(m2.AUTHORIZED_REMOTE)
    src = (SRC / "src" / "ohlcv_dataroad" / "ingest" / "tda17_enmiendas_e02.py").read_text(encoding="utf-8")
    assert '"ls-remote", "--exit-code", AUTHORIZED_REMOTE' in src and '"fetch", "--quiet", AUTHORIZED_REMOTE' in src
    assert '"ls-remote", "--exit-code", "origin"' not in src


def test_official_remote_commit_missing_locally_is_downloaded_or_rejected(repo, tmp_path, monkeypatch):
    root, p2, a2 = repo["root"], repo["P2"], repo["A2"]
    other = tmp_path / "o"
    git(tmp_path, "clone", "-q", "-b", "main", str(repo["bare"]), "o")      # otro clon publica un commit posterior
    (other / "posterior.txt").write_bytes(b"publicado por otro clon\n")
    git(other, "add", "-A")
    git(other, "commit", "-q", "-m", "posterior publicado")
    newer = git(other, "rev-parse", "HEAD")
    git(other, "push", "-q", "origin", "HEAD:refs/heads/main")
    assert subprocess.run(["git", "cat-file", "-e", f"{newer}^{{commit}}"], cwd=root, capture_output=True).returncode != 0
    real = m2.e01.subprocess.run

    def no_fetch(cmd, *a, **k):
        if cmd[:2] == ["git", "fetch"]:
            return subprocess.CompletedProcess(cmd, 1, b"", b"fatal: sin red")
        return real(cmd, *a, **k)
    monkeypatch.setattr(m2.e01.subprocess, "run", no_fetch)
    assert_rejects(f"[E02_REMOTO] no se pudo descargar el commit remoto {newer}", m2.verify_remote_publication, root, p2, a2)
    monkeypatch.setattr(m2.e01.subprocess, "run", real)
    assert m2.verify_remote_publication(root, p2, a2) == newer              # descargado: historia P1 -> A2 -> P2 -> newer
    tr = m2.official_config_e02(root, p2)["trazabilidad_protocolo"]
    assert tr["remoto_E02"] == {"repositorio": str(repo["bare"]), "ref": "refs/heads/main", "sha": newer}


def test_negative_official_requires_the_preparatory_controls_first(repo):
    root, p2 = repo["root"], repo["P2"]
    _pending(root)
    assert_rejects(PENDING_MSG, m2.official_config_e02, root, p2)
    assert_rejects(PENDING_MSG, an.official_config, p2, root)
    assert_rejects(PENDING_MSG, an.production_config, root)


# =========================================================================== T-INT y garantias estaticas
def test_static_production_guards():
    src = (SRC / "src" / "ohlcv_dataroad" / "ingest" / "tda17_enmiendas_e02.py").read_text(encoding="utf-8")
    assert "_DISABLED" not in src and "environ" not in src and "getenv" not in src
    ids = set(re.findall(r'_fail\("([A-Z0-9_]+)"', src))
    assert ids == {"GIT", "E01_FICHEROS", "E01_BYTES", "E01_HISTORIA", "E01_REGISTRO", "E01_ENTRADA", "E01_A1",
                   "E02_REGISTRO", "E02_REG_ESTRUCTURA", "E02_BASE", "E02_REQUIERE", "E02_ENTRADA", "E02_CAMPOS",
                   "E02_FECHA", "E02_RESPONSABLE", "E02_REFERENCIA", "E02_COMMIT_FORMATO", "E02_COBERTURA", "E02_ARBOL",
                   "E02_COMMIT", "E02_ASCENDENCIA", "E02_A2_HEAD", "E02_RUTAS", "E02_HUELLA", "E02_DOC", "E02_ESTADO",
                   "E02_PENDIENTE", "E02_PENDIENTE_CON_DATOS", "E02_LISTA_BLANCA", "E02_ANADIR", "PROTOCOLO_ESTRICTO",
                   "PROTOCOLO_GIT", "E02_REMOTO", "E02_REMOTO_HISTORIA", "E02_P2_FORMATO", "E02_P2_EXISTE", "E02_P2_A2",
                   "E02_P2_HEAD", "E02_P2_ARBOL", "E02_P2_APROBADA", "E02_REQUERIDA", "E02_NO_OFICIAL",
                   "E02_OFICIAL_NO_VERIFICABLE", "E02_OFICIAL_DISTINTA", "E02_REMOTO_NO_AUTORIZADO"}
    official = re.search(r"def official_config_e02.*?(?=\ndef )", src, re.S).group(0)
    order = [official.index(t) for t in ("E02_P2_FORMATO", "load_effective_protocol_e02(root)", "E02_P2_EXISTE",
                                         "E02_P2_A2", "E02_P2_HEAD", "E02_P2_ARBOL", "E02_P2_APROBADA",
                                         "verify_frozen_protocol(root)", "verify_remote_publication(")]
    assert order == sorted(order)
    assert set(m2.AUDITED_E02) >= {"src/ohlcv_dataroad/ingest/tda17_analysis.py", "src/ohlcv_dataroad/ingest/tda17_g2.py",
                                   "src/ohlcv_dataroad/ingest/tda17_causalidad.py", "src/ohlcv_dataroad/ingest/tda17_salidas.py",
                                   "src/ohlcv_dataroad/ingest/tda17_entradas.py", "tests/test_tda17_procedencia.py",
                                   "tests/test_tda17_salidas.py"}
    assert m2.REGISTRY_E02 not in m2.AUDITED_E02 and m2.TRACKED_E02 == [m2.REGISTRY_E02] + m2.AUDITED_E02
    phase2 = sorted(p.name for p in (SRC / "src" / "ohlcv_dataroad" / "ingest").glob("tda17_*.py"))
    covered = {Path(r).name for r in m2.AUDITED_E02} | {Path(r).name for r in m2.E01_FILES_SHA256} | {"tda17_protocol_freeze.py"}
    assert set(phase2) <= covered, set(phase2) - covered                    # todo modulo de TDA-17 esta auditado
    tests = sorted(p.name for p in (SRC / "tests").glob("test_tda17_*.py"))
    covered_t = {Path(r).name for r in m2.AUDITED_E02 + list(m2.E01_FILES_SHA256) + frz.PROTOCOL_FILES}
    assert set(tests) <= covered_t, set(tests) - covered_t
