"""TDA-17 -- cargador escalonado de la enmienda E02 (migracion M2).

Especificacion: ``reports/mnq/TDA17_especificacion_migracion_M2.md`` (SPEC-M2) y checklist de garantias de
``reports/mnq/TDA17_E02_propuesta_decisiones_normativas.md`` (seccion 7). El cargador de E01
(``tda17_enmiendas.py``) NO cambia: este modulo comprueba que E01 sigue exactamente como se publico (8 ficheros,
entrada del registro, commit auditado A1, publicacion P1), ejecuta el cargador de E01 sin modificar y verifica
despues E02 con su registro, su aprobacion, su commit auditado A2 y su publicacion P2 propios.

Cada control rechaza con ``ProtocolNotAuthorizedError`` y un identificador ``[ID]`` al principio del mensaje.
Ningun control tiene excepcion, parametro, variable de entorno ni modo que lo omita: la omision de un control solo
puede demostrarse con mutaciones del codigo fuente sobre clones Git (SPEC-M2 seccion 11, T-MUT).

Usos:
- BORRADOR: ``effective_document_e02(root, borrador=True)`` inspecciona una E02 pendiente (``activable`` false,
  ``ejecucion_oficial`` false). Nunca se usa en produccion.
- APROBACION PREPARATORIA: ``load_effective_protocol_e02(root)`` exige E02 APROBADA, su commit A2 y que HEAD
  descienda de A2 (``ejecucion_oficial`` false, ``publicacion_E02`` null). No produce resultados del informe.
- EJECUCION OFICIAL: ``official_config_e02(root, publicacion=P2)`` aplica lo anterior y ademas exige P2 completa,
  existente, descendiente estricta de A2 y antecesora de HEAD; arbol identico a P2 en los ficheros de E01 y de E02;
  la aprobacion publicada en P2; la verificacion estricta del protocolo congelado (9 ficheros, 5 entradas y 188
  artefactos) y su comparacion con el commit de congelamiento; y que P2 este en la rama remota publicada con A2 y
  P1 en su historia. Unica via para producir resultados oficiales (``ejecucion_oficial`` true).

Herramienta: ``python -m ohlcv_dataroad.ingest.tda17_enmiendas_e02 verificar [--referencia-git P2]``.

DECISION IRIS: NINGUNA.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

import yaml

from ohlcv_dataroad.ingest import tda17_enmiendas as e01
from ohlcv_dataroad.ingest import tda17_indicators as ind
from ohlcv_dataroad.ingest import tda17_protocol_freeze as frz

ProtocolNotAuthorizedError = e01.ProtocolNotAuthorizedError
ROOT = e01.ROOT

# --------------------------------------------------------------------------- E01 publicada (SPEC-M2 3.1)
E01_A1 = "386815bde75833ec64763f65b70495d0f535c44d"
E01_P1 = "24f9f1d2bb831a25e4dec6a651c0a5129f0bbca1"
E01_REGISTRY = e01.REGISTRY_JSON
E01_ENTRY_SHA256 = "3150ea3a5e6fda45280d6ab3936f61136f837ae6d106251b3f474cfedfdd5db9"
E01_FILES_SHA256 = {
    "configs/tda17_enmienda_01.yaml": "62b91747d5102aef51cc2c51f014e20d3ba1e635e98cddee7c57ecebefa5d5cb",
    "reports/mnq/TDA17_enmiendas.json": "f286b9cf6c3ed2865ad006b7fdd5e1ee3df518feaaab7e96e9a363ab37877785",
    "src/ohlcv_dataroad/ingest/tda17_enmiendas.py": "6c5d96d0513b5e0962f476c7175529b061c2c4ea6c81340c86b1918a9575b569",
    "src/ohlcv_dataroad/ingest/tda17_indicators.py": "9d197a52576f18a19f6ebf51a6fb5f3bf966253515a1ec3439e3a44b4042dd1a",
    "src/ohlcv_dataroad/ingest/tda17_primitives.py": "11f90df582baf823a8c523eff0ad268ea720207fa8eb9f8bc8227d49f1a9fb32",
    "tests/test_tda17_indicators.py": "c3dc92f122beec2b4621e76c9012f178906814cf0c24130291f738f0ee83c417",
    "tests/test_tda17_enmienda_01.py": "25a86dbe393501c6262d628dffdde7fd1df3aa8e6e54f0834b1908fc6d611790",
    "tests/tda17_synthetic.py": "23e4f30a13069c8b717526d63bad029b50ad73493f7eca001a9b96f54d6ec64e",
}

# --------------------------------------------------------------------------- E02 (SPEC-M2 3.3, 5, 7)
REGISTRY_E02 = "reports/mnq/TDA17_enmiendas_E02.json"
E02_YAML = "configs/tda17_enmienda_02.yaml"
E02_SHA256 = "fd7478d557e9a798c52b2af0806e89a6945c90cf5d8f35ea328afb581ad0d317"
# Operaciones autorizadas (decisiones 1-10 y 12 del registro de decisiones; la 11 esta aplazada y la 13 no es YAML;
# autorizacion complementaria del 2026-10-19: decisiones 14, 15, 16, 17+18 y 19).
WHITELIST_E02 = (
    ("anadir", "identidades.verificacion_numerica.convencion"),
    ("anadir", "identidades.verificacion_numerica.ambito"),
    ("anadir", "identidades.verificacion_numerica.procedimiento_bloqueo"),
    ("anadir", "analisis.bootstrap.intervalo"),
    ("anadir", "analisis.bootstrap.ambito"),
    ("anadir", "analisis.bootstrap.reproducibilidad"),
    ("anadir", "analisis.bootstrap.estadistico_replica"),
    ("anadir", "analisis.spearman.validacion_detalle"),
    ("anadir", "analisis.clasificacion.celdas_no_publicables"),
    ("anadir", "analisis.clasificacion.subniveles_r2"),
    ("anadir", "analisis.auditoria_causalidad"),
    ("anadir", "analisis.spearman.validacion_replica_invalida"),
    ("anadir", "analisis.clasificacion.subniveles_r2_global_no_publicable"),
    ("anadir", "identidades.verificacion_numerica.relaciones_N"),
    ("anadir", "salidas.esquema_minimo"),
    ("anadir", "salidas.diagnostico_bloqueo"),
)
# Rutas que la lista blanca nunca puede contener ni tocar (SPEC-M2 seccion 7). '*' = un segmento cualquiera.
FORBIDDEN_PATHS = (
    "catalogo.*.configuraciones", "catalogo.*.formula",
    "analisis.umbrales", "analisis.clasificacion.umbral_r2", "analisis.clasificacion.umbral_parcial",
    "analisis.clasificacion.anios_min", "analisis.clasificacion.segmentos_min",
    "analisis.r2.tau_relativa", "analisis.r2.tolerancia_cruzada", "analisis.r2.hueco_espectral_minimo",
    "identidades.lineales", "identidades.rango",
    "identidades.verificacion_numerica.tau_relativa", "identidades.verificacion_numerica.nulos_esperados",
    "analisis.exactitud.residuo_max", "analisis.exactitud.casi_exacta_max",
    "analisis.bootstrap.replicas", "analisis.bootstrap.longitud_bloque_jornadas", "analisis.bootstrap.semilla",
    "g2.*.semillas", "g2.*.semilla", "g2.*.replicas", "g2.*.df", "g2.*.tolerancias",
    "sensibilidades.S6.fechas", "decisiones", "protocolo", "entradas",
)
AUDITED_E02 = [
    E02_YAML,
    "src/ohlcv_dataroad/ingest/tda17_enmiendas_e02.py",
    "tests/test_tda17_enmienda_02.py",
    "src/ohlcv_dataroad/ingest/tda17_analysis.py",
    "src/ohlcv_dataroad/ingest/tda17_g2.py",
    "src/ohlcv_dataroad/ingest/tda17_causalidad.py",
    "src/ohlcv_dataroad/ingest/tda17_salidas.py",
    "src/ohlcv_dataroad/ingest/tda17_entradas.py",
    "tests/test_tda17_analysis.py",
    "tests/test_tda17_g2.py",
    "tests/test_tda17_rango_diagnostico.py",
    "tests/test_tda17_causalidad.py",
    "tests/test_tda17_e02_motor.py",
    "tests/test_tda17_procedencia.py",
    "tests/test_tda17_salidas.py",
]
TRACKED_E02 = [REGISTRY_E02] + AUDITED_E02
REGISTRY_FIELDS = ("descripcion", "protocolo_base", "requiere", "enmiendas")
ENTRY_FIELDS = ("id", "fichero", "sha256", "alcance", "estado", "fecha_preparacion", "aprobacion")
REMOTE_BRANCH = "refs/heads/main"
# Repositorio GitHub AUTORIZADO para la publicacion de P2 (identidad fijada en este fichero auditado; no es un
# parametro ni una variable de entorno). La consulta remota se hace contra esta URL, no contra el nombre "origin".
AUTHORIZED_REMOTE = "https://github.com/GUNAPILLCO/ohlcv_dataroad.git"
_HEX40 = re.compile(r"[0-9a-f]{40}")
_ISO = re.compile(r"\d{4}-\d{2}-\d{2}")


def _fail(cid: str, msg: str):
    raise ProtocolNotAuthorizedError(f"[{cid}] {msg}")


# --------------------------------------------------------------------------- acceso a Git (fallo = rechazo)
def _git(root: Path, *args: str):
    res, err = e01._git(root, *args)
    if err:
        _fail("GIT", f"sin acceso a Git: {err}")
    return res


def _check_repository_root(root: Path) -> None:
    """``root`` debe ser la raiz de un repositorio Git (no una carpeta sin repositorio ni una subcarpeta de otro)."""
    res = _git(root, "rev-parse", "--show-toplevel")
    top = res.stdout.decode("utf-8", "replace").strip()
    if res.returncode != 0 or not top or Path(top).resolve() != Path(root).resolve():
        _fail("GIT", f"{root} no es la raiz de un repositorio Git")


def _ancestor(root: Path, a: str, b: str) -> bool:
    res = _git(root, "merge-base", "--is-ancestor", a, b)
    if res.returncode not in (0, 1):
        _fail("GIT", f"error de Git al comprobar la ascendencia {a[:12]} -> {b[:12]}")
    return res.returncode == 0


def _hash_at(root: Path, commit: str, rel: str) -> str | None:
    """Huella LF de ``rel`` en ``commit``; None si el commit o el fichero no existen o no pueden leerse (el
    llamante rechaza: nunca se trata como coincidencia)."""
    res = _git(root, "show", f"{commit}:{rel}")
    return None if res.returncode else frz.sha256_bytes(res.stdout, Path(rel).suffix)


def _commit_exists(root: Path, commit: str) -> bool:
    return _git(root, "cat-file", "-e", f"{commit}^{{commit}}").returncode == 0


# --------------------------------------------------------------------------- 1. E01 publicada e intacta
def verify_e01(root: Path) -> dict:
    """Pasos 1-7 de SPEC-M2 seccion 6: E01 identica a su publicacion y cargador de E01 sin cambios."""
    for rel, h in E01_FILES_SHA256.items():
        if not (root / rel).is_file() or e01.sha256_text_file(root / rel) != h:
            _fail("E01_FICHEROS", f"fichero auditado de E01 alterado en el arbol: {rel}")
    res = _git(root, "diff", "--quiet", E01_P1, "HEAD", "--", *E01_FILES_SHA256)
    if res.returncode not in (0, 1):
        _fail("GIT", "error de Git al comparar los ficheros de E01 con su publicacion P1")
    if res.returncode == 1:
        _fail("E01_BYTES", "HEAD cambia ficheros de E01 respecto de su publicacion P1")
    if not _ancestor(root, E01_P1, "HEAD"):
        _fail("E01_HISTORIA", "la publicacion P1 de E01 no es antecesora de HEAD")
    reg = json.loads((root / E01_REGISTRY).read_text(encoding="utf-8"))
    ents = reg.get("enmiendas") if isinstance(reg, dict) and isinstance(reg.get("enmiendas"), list) else []
    ids = [e.get("id") for e in ents if isinstance(e, dict)]
    if ids != ["E01"] or len(ents) != 1:
        _fail("E01_REGISTRO", f"el registro de E01 debe contener solo E01: {ids}")
    entry = ents[0]
    if e01.canonical_sha256(entry) != E01_ENTRY_SHA256:
        _fail("E01_ENTRADA", "la entrada E01 del registro no es la publicada")
    if ((entry.get("aprobacion") or {}).get("referencia_auditoria") or {}).get("commit_auditado") != E01_A1:
        _fail("E01_A1", "la aprobacion de E01 no cita su commit auditado A1")
    return e01.load_effective_protocol(root)              # cargador de E01 sin cambios: E01 contra A1


# --------------------------------------------------------------------------- 2. registro y aprobacion de E02
def _registry_e02(root: Path) -> dict:
    path = root / REGISTRY_E02
    if not path.is_file():
        _fail("E02_REGISTRO", "falta el registro de E02")
    try:
        reg = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError):
        _fail("E02_REGISTRO", "el registro de E02 no es JSON valido")
    if not isinstance(reg, dict) or set(reg) != set(REGISTRY_FIELDS):
        _fail("E02_REG_ESTRUCTURA", f"el registro de E02 debe tener exactamente {list(REGISTRY_FIELDS)}")
    if reg["protocolo_base"] != e01._base_reference():
        _fail("E02_BASE", "el registro de E02 no declara el protocolo base congelado")
    if reg["requiere"] != {"E01": E01_ENTRY_SHA256, "commit_auditado_E01": E01_A1, "publicacion_E01": E01_P1}:
        _fail("E02_REQUIERE", "el registro de E02 no declara la E01 publicada (entrada, A1, P1)")
    ents = reg["enmiendas"] if isinstance(reg["enmiendas"], list) else []
    entry = ents[0] if len(ents) == 1 and isinstance(ents[0], dict) else {}
    if set(entry) != set(ENTRY_FIELDS) or entry.get("id") != "E02" or entry.get("fichero") != E02_YAML:
        _fail("E02_ENTRADA", "el registro de E02 debe tener una unica entrada E02 bien formada")
    return entry


def _approval(root: Path, entry: dict) -> dict:
    """Pasos 12-15b de SPEC-M2 seccion 6 (estructura, contenido y commit auditado A2 de la aprobacion)."""
    a = entry.get("aprobacion")
    if not isinstance(a, dict) or set(a) != set(e01.APPROVAL_FIELDS):
        _fail("E02_CAMPOS", f"la aprobacion debe tener exactamente {list(e01.APPROVAL_FIELDS)}")
    fecha = a["fecha"]
    try:
        ok = bool(isinstance(fecha, str) and _ISO.fullmatch(fecha) and
                  dt.date.fromisoformat(str(entry.get("fecha_preparacion"))) <= dt.date.fromisoformat(fecha) <= dt.date.today())
    except ValueError:
        ok = False
    if not ok:
        _fail("E02_FECHA", f"fecha {fecha!r} no ISO valida o fuera de [fecha_preparacion, hoy]")
    if not isinstance(a["aprobada_por"], str) or not a["aprobada_por"].strip():
        _fail("E02_RESPONSABLE", "aprobada_por vacio (responsable declarado obligatorio)")
    ref = a["referencia_auditoria"]
    if not isinstance(ref, dict) or set(ref) != set(e01.AUDIT_REFERENCE_FIELDS) or \
            not isinstance(ref.get("documento"), str) or not ref["documento"].strip():
        _fail("E02_REFERENCIA", "referencia_auditoria debe tener documento no vacio y commit_auditado")
    commit = ref["commit_auditado"]
    if not isinstance(commit, str) or not _HEX40.fullmatch(commit):
        _fail("E02_COMMIT_FORMATO", "commit_auditado no es un SHA-1 Git completo (40 hexadecimales)")
    hashes = a["sha256_auditados"]
    if not isinstance(hashes, dict) or sorted(hashes) != sorted(AUDITED_E02):
        _fail("E02_COBERTURA", "sha256_auditados no cubre exactamente los ficheros auditados de E02")
    changed = [r for r in AUDITED_E02 if not (root / r).is_file() or e01.sha256_text_file(root / r) != hashes[r]]
    if changed:
        _fail("E02_ARBOL", f"ficheros distintos de la version auditada de E02: {changed}")
    bad = [r for r in AUDITED_E02 if _hash_at(root, commit, r) != hashes[r]]
    if bad:
        _fail("E02_COMMIT", f"el commit auditado {commit} no existe o no contiene la version aprobada de {bad}")
    if not _ancestor(root, E01_P1, commit):
        _fail("E02_ASCENDENCIA", "el commit auditado de E02 no desciende de la publicacion P1 de E01")
    if not _ancestor(root, commit, "HEAD"):
        _fail("E02_A2_HEAD", "HEAD no desciende del commit auditado A2 de E02")
    return a


# --------------------------------------------------------------------------- 3. lista blanca y configuracion efectiva
def _forbidden(path: str) -> bool:
    """La ruta coincide con una ruta prohibida, la contiene o esta contenida en ella."""
    parts = path.split(".")
    for pat in FORBIDDEN_PATHS:
        p = pat.split(".")
        n = min(len(p), len(parts))
        if all(x == "*" or x == y for x, y in zip(p[:n], parts[:n])):
            return True
    return False


def check_whitelist_static() -> list[str]:
    """T-WL: ninguna operacion de la lista blanca toca una ruta prohibida y todas son 'anadir' sin repetir."""
    errors = [f"ruta prohibida en la lista blanca: {r}" for _, r in WHITELIST_E02 if _forbidden(r)]
    errors += [f"operacion no admitida en E02: {op} {r}" for op, r in WHITELIST_E02 if op != "anadir"]
    if len(set(WHITELIST_E02)) != len(WHITELIST_E02):
        errors.append("la lista blanca repite operaciones")
    return errors


def check_paths(base: dict, new: dict) -> None:
    """Barrera independiente de la lista de operaciones: la efectiva difiere de E01 exactamente en las rutas."""
    changed, allowed = e01.diff_paths(base, new), {r for _, r in WHITELIST_E02}
    if changed != allowed:
        _fail("E02_RUTAS", f"la configuracion efectiva no cambia exactamente las rutas autorizadas: {sorted(changed ^ allowed)}")


def _effective(root: Path, produccion: bool) -> dict:
    _check_repository_root(root)
    eff1 = verify_e01(root)
    entry = _registry_e02(root)
    path = root / E02_YAML
    digest = e01.sha256_text_file(path) if path.is_file() else None
    if digest != E02_SHA256 or entry.get("sha256") != E02_SHA256:
        _fail("E02_HUELLA", f"{E02_YAML} no es la version autorizada de E02 (SHA-256 {digest})")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    head = doc.get("enmienda") if isinstance(doc, dict) and isinstance(doc.get("enmienda"), dict) else {}
    if head.get("id") != "E02" or head.get("protocolo_base") != e01._base_reference() or head.get("requiere") != "E01":
        _fail("E02_DOC", "el YAML de E02 no declara id E02, el protocolo base y requiere E01")
    estado = entry["estado"]
    if estado not in (e01.ESTADO_PENDIENTE, e01.ESTADO_APROBADA):
        _fail("E02_ESTADO", f"estado desconocido {estado!r}")
    if produccion and estado != e01.ESTADO_APROBADA:
        _fail("E02_PENDIENTE", f"E02 no aprobada (estado {estado}): la configuracion no es activable")
    approval = None
    if estado == e01.ESTADO_APROBADA:
        approval = _approval(root, entry)
    elif entry["aprobacion"] is not None:
        _fail("E02_PENDIENTE_CON_DATOS", "E02 PENDIENTE con datos de aprobacion")
    ops = doc.get("cambios")
    if not isinstance(ops, list) or not all(isinstance(o, dict) and set(o) == {"op", "ruta", "valor"} for o in ops) or \
            sorted((o["op"], o["ruta"]) for o in ops) != sorted(WHITELIST_E02):
        _fail("E02_LISTA_BLANCA", "operaciones de E02 distintas de su lista blanca")
    base = {k: v for k, v in eff1.items() if k != "trazabilidad_protocolo"}
    new = copy.deepcopy(base)
    for o in ops:
        try:
            parent, key = e01._resolve_parent(new, o["ruta"])
        except ProtocolNotAuthorizedError as exc:
            _fail("E02_ANADIR", f"ruta no aplicable: {o['ruta']} ({exc})")
        if o["op"] != "anadir" or key in parent:
            _fail("E02_ANADIR", f"operacion no admitida o 'anadir' sobre clave existente: {o['ruta']}")
        parent[key] = copy.deepcopy(o["valor"])
    check_paths(base, new)
    ind._validate_cfg_structure(new)
    tr = copy.deepcopy(eff1["trazabilidad_protocolo"])
    tr["enmiendas"].append({"id": "E02", "fichero": E02_YAML, "sha256": E02_SHA256, "estado": estado,
                            "aprobacion": approval})
    tr["sha256_registro_E02"] = e01.sha256_text_file(root / REGISTRY_E02)
    tr["sha256_efectiva"] = e01.canonical_sha256(new)
    tr["modo"] = "PRODUCCION" if produccion else "BORRADOR"
    tr["activable"] = bool(produccion)
    tr["ejecucion_oficial"] = False                        # solo official_config_e02 la vincula a P2
    tr["publicacion_E02"] = None
    new["trazabilidad_protocolo"] = tr
    return new


def load_effective_protocol_e02(root: Path | None = None) -> dict:
    """APROBACION PREPARATORIA: E01 publicada e intacta + E02 APROBADA, verificada contra A2 y antecesora de HEAD.
    No basta para producir resultados del informe (``require_e02_oficial``)."""
    return _effective(Path(ROOT if root is None else root), produccion=True)


def effective_document_e02(root: Path | None = None, *, borrador: bool = False) -> dict:
    """Configuracion efectiva E01 + E02. Con ``borrador`` admite E02 PENDIENTE (``activable`` false); nunca se usa
    en produccion."""
    return _effective(Path(ROOT if root is None else root), produccion=not borrador)


# --------------------------------------------------------------------------- 4. ejecucion oficial (SPEC-M2 6.1, checklist)
def verify_against_git_e02(root, commit: str) -> list[str]:
    """Verificacion final contra el commit publicado P2: los 8 ficheros de E01 y los seguidos de E02."""
    root = Path(root)
    errors = []
    for rel in list(E01_FILES_SHA256) + TRACKED_E02:
        h = _hash_at(root, commit, rel)
        if h is None:
            errors.append(f"[git {commit}] no contiene {rel}")
        elif not (root / rel).is_file() or e01.sha256_text_file(root / rel) != h:
            errors.append(f"[git {commit}] difiere del commit de referencia: {rel}")
    return errors


def verify_frozen_protocol(root: Path) -> None:
    """Checklist 1: verificacion estricta oficial del protocolo congelado (9 ficheros, 5 entradas, 188 artefactos)
    y comparacion de los ficheros del protocolo y el manifiesto con el commit de congelamiento."""
    try:
        errors = frz.verify_strict(root)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError, frz.ProtocolError) as exc:
        errors = [f"{type(exc).__name__}: {exc}"]
    if errors:
        _fail("PROTOCOLO_ESTRICTO", f"verificacion estricta del protocolo NO CONFORME: {errors}")
    try:
        errors = frz.verify_against_git(root, ind.FROZEN_COMMIT)
    except (OSError, ValueError) as exc:
        _fail("GIT", f"sin acceso a Git: {type(exc).__name__}: {exc}")
    if errors:
        _fail("PROTOCOLO_GIT", f"el protocolo no coincide con el commit de congelamiento: {errors}")


def normalize_remote(url: str) -> str:
    """Identidad comparable de un remoto: ``git@host:ruta`` -> ``https://host/ruta``; minusculas; sin ``/`` ni
    ``.git`` finales. Una ruta local se resuelve (solo aparece en los repositorios temporales de las pruebas)."""
    u = url.strip()
    if u.startswith("git@") and ":" in u:
        host, path = u[4:].split(":", 1)
        u = f"https://{host}/{path}"
    elif "://" not in u:
        u = Path(u).resolve().as_posix()
    u = u.lower().rstrip("/")
    return u[:-4] if u.endswith(".git") else u


def _check_authorized_remote(root: Path) -> None:
    """``origin`` debe ser el repositorio autorizado y ninguna regla ``url.*.insteadOf`` / ``pushInsteadOf`` de la
    configuracion de Git (en cualquier ambito) puede redirigir las URL."""
    rules = _git(root, "config", "--get-regexp", r"^url\..*\.(insteadof|pushinsteadof)$")
    if rules.returncode not in (0, 1):
        _fail("GIT", "error de Git al leer las reglas url.*.insteadOf")
    if rules.returncode == 0 and rules.stdout.strip():
        _fail("E02_REMOTO_NO_AUTORIZADO", "la configuracion de Git redirige URL (url.*.insteadOf): no se puede "
              "garantizar la identidad del repositorio autorizado")
    res = _git(root, "remote", "get-url", "origin")
    url = res.stdout.decode("utf-8", "replace").strip() if res.returncode == 0 else ""
    if not url or normalize_remote(url) != normalize_remote(AUTHORIZED_REMOTE):
        _fail("E02_REMOTO_NO_AUTORIZADO", f"origin ({url or 'inexistente'}) no es el repositorio autorizado {AUTHORIZED_REMOTE}")


def verify_remote_publication(root: Path, publicacion: str, a2: str) -> str:
    """Checklist 2: P2 publicada en el REMOTO REAL del repositorio AUTORIZADO (``AUTHORIZED_REMOTE``). Exige que
    ``origin`` sea ese repositorio y que no haya reglas de redireccion de URL (``E02_REMOTO_NO_AUTORIZADO``); despues
    consulta la rama publicada directamente en la URL autorizada (``git ls-remote <URL> refs/heads/main``), nunca la
    referencia local ``refs/remotes/origin/main``, que puede estar desactualizada. Si el commit remoto no esta en el
    repositorio local, se descarga de la misma URL para comprobar la historia. Cualquier fallo (remoto inaccesible, rama inexistente, respuesta
    mal formada, descarga fallida) es un rechazo. Devuelve el SHA remoto real.

    Historia: P2 alcanzable desde el remoto (``E02_REMOTO``), A2 antecesor estricto de P2 (``E02_P2_A2``) y P1
    antecesor de A2 (``E02_ASCENDENCIA``) implican, por transitividad de la ascendencia, que A2 y P1 estan en la
    historia remota; ``E02_REMOTO_HISTORIA`` se conserva como defensa en profundidad."""
    _check_authorized_remote(root)
    res = _git(root, "ls-remote", "--exit-code", AUTHORIZED_REMOTE, REMOTE_BRANCH)
    lines = res.stdout.decode("utf-8", "replace").split("\n") if res.returncode == 0 else []
    refs = [ln.split("\t") for ln in lines if ln.strip()]
    if res.returncode != 0 or len(refs) != 1 or len(refs[0]) != 2 or refs[0][1] != REMOTE_BRANCH or \
            not _HEX40.fullmatch(refs[0][0]):
        _fail("E02_REMOTO", f"no se pudo consultar la rama publicada {REMOTE_BRANCH} del repositorio autorizado")
    remote = refs[0][0]
    if not _commit_exists(root, remote):
        fetched = _git(root, "fetch", "--quiet", AUTHORIZED_REMOTE, REMOTE_BRANCH)
        if fetched.returncode != 0 or not _commit_exists(root, remote):
            _fail("E02_REMOTO", f"no se pudo descargar el commit remoto {remote}")
    if not _ancestor(root, publicacion, remote):
        _fail("E02_REMOTO", f"la publicacion P2 {publicacion} no esta en la rama publicada del remoto ({remote})")
    if not (_ancestor(root, E01_P1, remote) and _ancestor(root, a2, remote)):
        _fail("E02_REMOTO_HISTORIA", "la historia remota no contiene A2 y P1")
    return remote


def official_config_e02(root: Path | None = None, publicacion: str | None = None) -> dict:
    """EJECUCION OFICIAL: configuracion de produccion vinculada a la publicacion P2 de E02 (SPEC-M2 6.1 y checklist
    1-2). No hay ningun parametro, variable ni modo que omita controles."""
    root = Path(ROOT if root is None else root)
    if not isinstance(publicacion, str) or not _HEX40.fullmatch(publicacion):
        _fail("E02_P2_FORMATO", "la ejecucion oficial exige el SHA-1 completo de la publicacion P2")
    eff = load_effective_protocol_e02(root)
    a2 = eff["trazabilidad_protocolo"]["enmiendas"][-1]["aprobacion"]["referencia_auditoria"]["commit_auditado"]
    if not _commit_exists(root, publicacion):
        _fail("E02_P2_EXISTE", f"la publicacion {publicacion} no existe en el repositorio")
    if publicacion == a2 or not _ancestor(root, a2, publicacion):
        _fail("E02_P2_A2", "el commit auditado A2 debe ser antecesor estricto de la publicacion P2")
    if not _ancestor(root, publicacion, "HEAD"):
        _fail("E02_P2_HEAD", "la publicacion P2 no es HEAD ni antecesora de HEAD")
    errors = verify_against_git_e02(root, publicacion)
    if errors:
        _fail("E02_P2_ARBOL", f"el arbol no coincide con la publicacion P2: {errors}")
    shown = _git(root, "show", f"{publicacion}:{REGISTRY_E02}")
    try:
        published = json.loads(shown.stdout.decode("utf-8")) if shown.returncode == 0 else {}
    except (ValueError, UnicodeDecodeError):
        published = {}
    ents = published.get("enmiendas") if isinstance(published, dict) else None
    pe = ents[0] if isinstance(ents, list) and len(ents) == 1 and isinstance(ents[0], dict) else {}
    cited = ((pe.get("aprobacion") or {}).get("referencia_auditoria") or {}).get("commit_auditado")
    if pe.get("estado") != e01.ESTADO_APROBADA or cited != a2:
        _fail("E02_P2_APROBADA", "la publicacion P2 no contiene la aprobacion de E02 que cita A2")
    verify_frozen_protocol(root)
    remote = verify_remote_publication(root, publicacion, a2)
    tr = eff["trazabilidad_protocolo"]
    tr["ejecucion_oficial"] = True
    tr["publicacion_E02"] = publicacion
    tr["remoto_E02"] = {"repositorio": AUTHORIZED_REMOTE, "ref": REMOTE_BRANCH, "sha": remote}
    return eff


# --------------------------------------------------------------------------- 5. exigencias para las rutas de produccion
def require_e02(cfg: dict) -> None:
    """Rechaza una configuracion sin E01 y E02 APROBADAS en su trazabilidad o no activable (SPEC-M2 seccion 9)."""
    tr = cfg.get("trazabilidad_protocolo") if isinstance(cfg, dict) else None
    tr = tr if isinstance(tr, dict) else {}
    ents = {e.get("id"): e for e in tr.get("enmiendas", []) if isinstance(e, dict)}
    if set(ents) != {"E01", "E02"} or any(e.get("estado") != e01.ESTADO_APROBADA for e in ents.values()) or \
            tr.get("modo") != "PRODUCCION" or tr.get("activable") is not True:
        _fail("E02_REQUERIDA", "la configuracion no contiene E01 y E02 APROBADAS y activables")


def require_e02_oficial(cfg: dict, root: Path | None = None) -> None:
    """Rutas que escriben o publican resultados oficiales. Los indicadores de la trazabilidad (``ejecucion_oficial``,
    ``publicacion_E02``) los puede fabricar el llamante: NO bastan. Tras comprobarlos, la configuracion oficial se
    DERIVA DE NUEVO con ``official_config_e02(root, P2)`` (Git, huellas, aprobacion, publicacion, protocolo estricto,
    remoto) y la recibida debe ser identica a la derivada (huella canonica de todo el diccionario)."""
    require_e02(cfg)
    tr = cfg["trazabilidad_protocolo"]
    if tr.get("ejecucion_oficial") is not True or not isinstance(tr.get("publicacion_E02"), str) or \
            not _HEX40.fullmatch(tr["publicacion_E02"]):
        _fail("E02_NO_OFICIAL", "la configuracion no procede de la ejecucion oficial vinculada a P2")
    try:
        derived = official_config_e02(root, tr["publicacion_E02"])
    except ProtocolNotAuthorizedError as exc:
        _fail("E02_OFICIAL_NO_VERIFICABLE", f"la configuracion oficial no se puede derivar de M2: {exc}")
    if e01.canonical_sha256(cfg) != e01.canonical_sha256(derived):
        _fail("E02_OFICIAL_DISTINTA", "la configuracion recibida no es identica a la ejecucion oficial derivada de M2")


# --------------------------------------------------------------------------- 6. herramienta de verificacion
def verify(root: Path, referencia_git: str | None = None) -> tuple[list[str], dict]:
    """Verificacion estricta del protocolo y de E01, cargador de E02 (borrador si esta pendiente, produccion si esta
    aprobada) y, con ``referencia_git``, comparacion con ese commit; con aprobacion y referencia, la ruta oficial y,
    si existe alguna salida oficial o una publicacion interrumpida, su comprobacion (``tda17_salidas``)."""
    errors: list[str] = []
    try:
        verify_frozen_protocol(root)
    except ProtocolNotAuthorizedError as exc:
        errors.append(str(exc))
    e01_errors, _ = e01.verify(root, referencia_git)
    errors += [f"[E01] {e}" for e in e01_errors]
    info: dict = {}
    try:
        doc = effective_document_e02(root, borrador=True)
        info = doc["trazabilidad_protocolo"]
        if info["enmiendas"][-1]["estado"] == e01.ESTADO_APROBADA:
            info = (official_config_e02(root, referencia_git) if referencia_git else
                    load_effective_protocol_e02(root))["trazabilidad_protocolo"]
            if referencia_git:
                from ohlcv_dataroad.ingest import tda17_salidas as sal
                cfg = official_config_e02(root, referencia_git)
                out_dir = Path(root) / cfg["salidas"]["esquema_minimo"]["directorio"]
                if any(os.path.lexists(out_dir / f) for f in sal.OFFICIAL + [sal.STAGING]):  # interrumpida o enlace
                    info = {**info, "salidas_oficiales": sal.check_official_publication(referencia_git, root)["estado"]}
    except ProtocolNotAuthorizedError as exc:
        errors.append(str(exc))
    if referencia_git:
        errors += verify_against_git_e02(root, referencia_git)
    return errors, info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("accion", choices=["verificar"])
    ap.add_argument("--referencia-git", metavar="P2", help="compara E01 y E02 con la publicacion P2 y ejecuta la ruta oficial")
    a = ap.parse_args(argv)
    errors, info = verify(ROOT, a.referencia_git)
    if errors:
        print("VERIFICACION DE E02: NO CONFORME")
        print("\n".join(errors))
        return 1
    e02 = info["enmiendas"][-1]
    salidas = f"; salidas oficiales {info['salidas_oficiales']} CONFORMES" if "salidas_oficiales" in info else ""
    print(f"VERIFICACION DE E02: CONFORME (protocolo {ind.FROZEN_COMMIT[:7]} estricto; E01 {E01_P1[:7]}; "
          f"E02 {e02['estado']} {e02['sha256'][:12]}; modo {info['modo']}; activable: {'SI' if info['activable'] else 'NO'}; "
          f"ejecucion oficial: {'SI' if info['ejecucion_oficial'] else 'NO'}; efectiva {info['sha256_efectiva'][:12]}{salidas})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
