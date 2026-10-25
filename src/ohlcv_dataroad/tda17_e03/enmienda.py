"""TDA-17 -- enmienda E03 (motor paralelo, SOLO implementacion): registro, aprobacion y ruta oficial con el motor E03.

E03 no cambia el protocolo, E01, E02 ni ningun fichero auditado: la configuracion efectiva sigue siendo la de E02
(``an.official_config`` con la publicacion P2 de E02), las entradas las de ``tda17_entradas`` y las tablas, el registro
de procedencia y la publicacion transaccional los de ``tda17_salidas``. Lo unico que cambia es QUIEN calcula el
resultado: el motor paralelo ``tda17_e03.motor`` en lugar de ``an.th35_pipeline``, con el criterio de aceptacion de
tablas identicas byte a byte en el mismo entorno numerico (1 hilo de OpenBLAS por proceso).

Garantias (las de E02 aplicadas a E03; cada control rechaza con ``ProtocolNotAuthorizedError`` y ``[ID]``):
- E02 PUBLICADA e intacta: la ruta oficial deriva la configuracion con ``an.official_config(E02_P2)`` (E01, E02, A2,
  P2, protocolo congelado estricto y P2 en el remoto autorizado), y el registro de E03 declara la entrada de E02
  publicada, su commit auditado A2 y su publicacion P2.
- Registro de E03 bien formado, YAML con la huella fijada en este modulo y contenido de la enmienda exacto.
- APROBACION PREPARATORIA (``load_e03_aprobada``): estado APROBADA con fecha, responsable, documento, commit auditado
  A3 (SHA-1 completo) cuyo arbol contiene la version aprobada de cada fichero auditado, A3 descendiente de P2 y HEAD
  descendiente de A3. No basta para producir resultados.
- EJECUCION OFICIAL (``official_e03(root, P3)``): ademas, P3 existente, descendiente ESTRICTA de A3 y antecesora de
  HEAD; arbol identico a P3 en los ficheros de E03; la aprobacion publicada en P3 citando A3; la configuracion oficial
  de E02; y P3 en la rama publicada del repositorio AUTORIZADO de E02 (consulta directa con ``git ls-remote`` a la URL
  autorizada, sin reglas ``insteadOf``), con A3 y P2 en su historia. Una aprobacion local (registro editado, commits
  locales no publicados) nunca habilita produccion.
- Entorno numerico: la ruta oficial y su reproduccion fijan y verifican 1 hilo efectivo de OpenBLAS en el proceso
  principal (``motor.hilos_blas_fijados``) y los workers se verifican en cada tarea (``E03_BLAS``).
- Reanudacion oficial DESHABILITADA: la ruta oficial no tiene parametro de reanudacion y exige un directorio de
  ejecucion nuevo fuera del repositorio (``motor.Ejecucion(oficial=True)``).

Ningun control tiene excepcion, parametro, variable de entorno ni modo que lo omita.

Herramienta: ``python -m ohlcv_dataroad.tda17_e03.enmienda verificar [--referencia-git P3]``.

DECISION ESTADISTICA: NINGUNA.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

import yaml

from ohlcv_dataroad.ingest import tda17_enmiendas as e01
from ohlcv_dataroad.ingest import tda17_enmiendas_e02 as m2

ProtocolNotAuthorizedError = m2.ProtocolNotAuthorizedError
ROOT = m2.ROOT

# --------------------------------------------------------------------------- E02 publicada (base de E03)
E02_A2 = "2b6ad7abdf6609e6970f1fe6ee1e2569edc698b4"
E02_P2 = "ff224dc7e30f7631dc46eea17db5a60dd44b07cc"
E02_ENTRY_SHA256 = "27b8e5b775310212e7b660af7b9a0afad3917af25600f197b0882692bded3b49"   # entrada E02 publicada en P2

# --------------------------------------------------------------------------- E03
E03_YAML = "configs/tda17_enmienda_03.yaml"
E03_SHA256 = "85cbf62b6c1ac9b5e0fca7e869d6502a0f5784bcde5ed26be809d3cc8042c747"
REGISTRY_E03 = "reports/mnq/TDA17_enmiendas_E03.json"
NAMESPACE = "src/ohlcv_dataroad/tda17_e03"
TESTS = "tests/tda17_e03"
AUDITED_E03 = [
    E03_YAML,
    f"{NAMESPACE}/__init__.py",
    f"{NAMESPACE}/motor.py",
    f"{NAMESPACE}/enmienda.py",
    f"{TESTS}/e03_equivalencia.py",
    f"{TESTS}/test_e03_motor.py",
    f"{TESTS}/test_e03_gobernanza.py",
    f"{TESTS}/test_e03_cobertura.py",
]
TRACKED_E03 = [REGISTRY_E03] + AUDITED_E03
REGISTRY_FIELDS = ("descripcion", "protocolo_base", "requiere", "enmiendas")
ENTRY_FIELDS = ("id", "fichero", "sha256", "alcance", "estado", "fecha_preparacion", "aprobacion")
HILOS_BLAS_OFICIALES = 1
REANUDACION_OFICIAL = "DESHABILITADA"
_HEX40 = re.compile(r"[0-9a-f]{40}")
_ISO = re.compile(r"\d{4}-\d{2}-\d{2}")


def _fail(cid: str, msg: str):
    raise ProtocolNotAuthorizedError(f"[{cid}] {msg}")


def _requiere() -> dict:
    return {"E02": E02_ENTRY_SHA256, "commit_auditado_E02": E02_A2, "publicacion_E02": E02_P2}


# --------------------------------------------------------------------------- 1. registro y documento de E03
def _registry_e03(root: Path) -> dict:
    path = root / REGISTRY_E03
    if not path.is_file():
        _fail("E03_REGISTRO", "falta el registro de E03")
    try:
        reg = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError):
        _fail("E03_REGISTRO", "el registro de E03 no es JSON valido")
    if not isinstance(reg, dict) or set(reg) != set(REGISTRY_FIELDS):
        _fail("E03_REG_ESTRUCTURA", f"el registro de E03 debe tener exactamente {list(REGISTRY_FIELDS)}")
    if reg["protocolo_base"] != e01._base_reference():
        _fail("E03_BASE", "el registro de E03 no declara el protocolo base congelado")
    if reg["requiere"] != _requiere():
        _fail("E03_REQUIERE", "el registro de E03 no declara la E02 publicada (entrada, A2, P2)")
    ents = reg["enmiendas"] if isinstance(reg["enmiendas"], list) else []
    entry = ents[0] if len(ents) == 1 and isinstance(ents[0], dict) else {}
    if set(entry) != set(ENTRY_FIELDS) or entry.get("id") != "E03" or entry.get("fichero") != E03_YAML or \
            entry.get("alcance") != "IMPLEMENTACION":
        _fail("E03_ENTRADA", "el registro de E03 debe tener una unica entrada E03 bien formada")
    return entry


def _check_doc(doc) -> dict:
    """El YAML de E03 es exactamente la enmienda implementada (alcance, base, criterio, entorno, reanudacion)."""
    d = doc.get("enmienda") if isinstance(doc, dict) and isinstance(doc.get("enmienda"), dict) else {}
    g = lambda *ks: _get(d, ks)                                             # noqa: E731
    ok = (d.get("id") == "E03" and d.get("alcance") == "IMPLEMENTACION" and d.get("decisiones_estadisticas") == "NINGUNA"
          and g("base", "protocolo") == e01._base_reference()["commit"] and g("base", "publicacion_E01") == m2.E01_P1
          and g("base", "commit_auditado_E02") == E02_A2 and g("base", "publicacion_E02") == E02_P2
          and g("criterio_equivalencia", "tipo") == "BYTES_IDENTICOS" and g("criterio_equivalencia", "tolerancias") == "NINGUNA"
          and g("entorno_numerico", "hilos_blas") == HILOS_BLAS_OFICIALES
          and type(g("entorno_numerico", "hilos_blas")) is int
          and g("entorno_numerico", "regla_workers") == "IGUAL_AL_PROCESO_PRINCIPAL"
          and str(g("paralelismo", "sorteos")).split(" ")[0] == "PROCESO_PRINCIPAL"
          and g("paralelismo", "orden_de_agregacion") == "E02"
          and g("puntos_de_control", "ejecucion_incompleta") == "NUNCA_VALIDA"
          and g("puntos_de_control", "reanudacion_oficial") == REANUDACION_OFICIAL
          and g("paralelismo", "memoria_insuficiente") == "RECHAZO_EXPLICITO"
          and type(g("paralelismo", "reserva_memoria_mb")) is int and g("paralelismo", "reserva_memoria_mb") > 0
          and g("publicacion_oficial", "comprobante_E03") == COMPROBANTE)
    if not ok:
        _fail("E03_DOC", "el YAML de E03 no es la enmienda implementada (alcance, base, criterio, entorno, reanudacion, "
                        "memoria o comprobante)")
    return d


def _get(d: dict, ks: tuple):
    for k in ks:
        d = d.get(k) if isinstance(d, dict) else None
    return d


# --------------------------------------------------------------------------- 2. aprobacion (commit auditado A3)
def _approval(root: Path, entry: dict) -> dict:
    a = entry.get("aprobacion")
    if not isinstance(a, dict) or set(a) != set(e01.APPROVAL_FIELDS):
        _fail("E03_CAMPOS", f"la aprobacion debe tener exactamente {list(e01.APPROVAL_FIELDS)}")
    fecha = a["fecha"]
    try:
        ok = bool(isinstance(fecha, str) and _ISO.fullmatch(fecha) and
                  dt.date.fromisoformat(str(entry.get("fecha_preparacion"))) <= dt.date.fromisoformat(fecha) <= dt.date.today())
    except ValueError:
        ok = False
    if not ok:
        _fail("E03_FECHA", f"fecha {fecha!r} no ISO valida o fuera de [fecha_preparacion, hoy]")
    if not isinstance(a["aprobada_por"], str) or not a["aprobada_por"].strip():
        _fail("E03_RESPONSABLE", "aprobada_por vacio (responsable declarado obligatorio)")
    ref = a["referencia_auditoria"]
    if not isinstance(ref, dict) or set(ref) != set(e01.AUDIT_REFERENCE_FIELDS) or \
            not isinstance(ref.get("documento"), str) or not ref["documento"].strip():
        _fail("E03_REFERENCIA", "referencia_auditoria debe tener documento no vacio y commit_auditado")
    commit = ref["commit_auditado"]
    if not isinstance(commit, str) or not _HEX40.fullmatch(commit):
        _fail("E03_COMMIT_FORMATO", "commit_auditado no es un SHA-1 Git completo (40 hexadecimales)")
    hashes = a["sha256_auditados"]
    if not isinstance(hashes, dict) or sorted(hashes) != sorted(AUDITED_E03):
        _fail("E03_COBERTURA", "sha256_auditados no cubre exactamente los ficheros auditados de E03")
    changed = [r for r in AUDITED_E03 if not (root / r).is_file() or e01.sha256_text_file(root / r) != hashes[r]]
    if changed:
        _fail("E03_ARBOL", f"ficheros distintos de la version auditada de E03: {changed}")
    bad = [r for r in AUDITED_E03 if m2._hash_at(root, commit, r) != hashes[r]]
    if bad:
        _fail("E03_COMMIT", f"el commit auditado {commit} no existe o no contiene la version aprobada de {bad}")
    if not m2._ancestor(root, E02_P2, commit):
        _fail("E03_ASCENDENCIA", "el commit auditado de E03 no desciende de la publicacion P2 de E02")
    if not m2._ancestor(root, commit, "HEAD"):
        _fail("E03_A3_HEAD", "HEAD no desciende del commit auditado A3 de E03")
    return a


def _effective(root: Path, produccion: bool) -> dict:
    m2._check_repository_root(root)
    entry = _registry_e03(root)
    path = root / E03_YAML
    digest = e01.sha256_text_file(path) if path.is_file() else None
    if digest != E03_SHA256 or entry.get("sha256") != E03_SHA256:
        _fail("E03_HUELLA", f"{E03_YAML} no es la version autorizada de E03 (SHA-256 {digest})")
    doc = _check_doc(yaml.safe_load(path.read_text(encoding="utf-8")))
    estado = entry["estado"]
    if estado not in (e01.ESTADO_PENDIENTE, e01.ESTADO_APROBADA):
        _fail("E03_ESTADO", f"estado desconocido {estado!r}")
    if produccion and estado != e01.ESTADO_APROBADA:
        _fail("E03_PENDIENTE", f"E03 no aprobada (estado {estado}): la ruta oficial del motor E03 no se activa")
    approval = None
    if estado == e01.ESTADO_APROBADA:
        approval = _approval(root, entry)
    elif entry["aprobacion"] is not None:
        _fail("E03_PENDIENTE_CON_DATOS", "E03 PENDIENTE con datos de aprobacion")
    return {"documento": doc, "estado": estado, "aprobacion": approval, "sha256_yaml": E03_SHA256,
            "sha256_registro_E03": e01.sha256_text_file(root / REGISTRY_E03),
            "modo": "PRODUCCION" if produccion else "BORRADOR", "ejecucion_oficial": False, "publicacion_E03": None}


def load_e03_aprobada(root: Path | None = None) -> dict:
    """APROBACION PREPARATORIA: E03 APROBADA, verificada contra A3, A3 descendiente de P2 y antecesor de HEAD. No
    basta para producir resultados (``official_e03``)."""
    return _effective(Path(ROOT if root is None else root), produccion=True)


def effective_e03(root: Path | None = None, *, borrador: bool = False) -> dict:
    """Estado de E03; con ``borrador`` admite E03 PENDIENTE (nunca en produccion)."""
    return _effective(Path(ROOT if root is None else root), produccion=not borrador)


# --------------------------------------------------------------------------- 3. ejecucion oficial (P3 publicada)
def verify_against_git_e03(root, commit: str) -> list[str]:
    """Ficheros seguidos de E03 identicos a los del commit publicado P3."""
    root = Path(root)
    errors = []
    for rel in TRACKED_E03:
        h = m2._hash_at(root, commit, rel)
        if h is None:
            errors.append(f"[git {commit}] no contiene {rel}")
        elif not (root / rel).is_file() or e01.sha256_text_file(root / rel) != h:
            errors.append(f"[git {commit}] difiere del commit de referencia: {rel}")
    return errors


def _check_authorized_remote(root: Path) -> None:
    """``origin`` es el repositorio AUTORIZADO de E02 (``m2.AUTHORIZED_REMOTE``) y ninguna regla ``insteadOf`` /
    ``pushInsteadOf`` redirige las URL."""
    rules = m2._git(root, "config", "--get-regexp", r"^url\..*\.(insteadof|pushinsteadof)$")
    if rules.returncode not in (0, 1):
        _fail("GIT", "error de Git al leer las reglas url.*.insteadOf")
    if rules.returncode == 0 and rules.stdout.strip():
        _fail("E03_REMOTO_NO_AUTORIZADO", "la configuracion de Git redirige URL (url.*.insteadOf): no se puede "
              "garantizar la identidad del repositorio autorizado")
    res = m2._git(root, "remote", "get-url", "origin")
    url = res.stdout.decode("utf-8", "replace").strip() if res.returncode == 0 else ""
    if not url or m2.normalize_remote(url) != m2.normalize_remote(m2.AUTHORIZED_REMOTE):
        _fail("E03_REMOTO_NO_AUTORIZADO", f"origin ({url or 'inexistente'}) no es el repositorio autorizado {m2.AUTHORIZED_REMOTE}")


def verify_remote_publication_e03(root: Path, publicacion: str, a3: str) -> str:
    """P3 publicada en la rama ``refs/heads/main`` del REMOTO REAL del repositorio autorizado, consultada directamente
    en su URL (nunca ``refs/remotes/origin/main``); si el commit remoto no esta en el repositorio local se descarga de
    la misma URL. Cualquier fallo es un rechazo. Exige A3 y P2 en la historia remota. Devuelve el SHA remoto."""
    _check_authorized_remote(root)
    url, branch = m2.AUTHORIZED_REMOTE, m2.REMOTE_BRANCH
    res = m2._git(root, "ls-remote", "--exit-code", url, branch)
    lines = res.stdout.decode("utf-8", "replace").split("\n") if res.returncode == 0 else []
    refs = [ln.split("\t") for ln in lines if ln.strip()]
    if res.returncode != 0 or len(refs) != 1 or len(refs[0]) != 2 or refs[0][1] != branch or not _HEX40.fullmatch(refs[0][0]):
        _fail("E03_REMOTO", f"no se pudo consultar la rama publicada {branch} del repositorio autorizado")
    remote = refs[0][0]
    if not m2._commit_exists(root, remote):
        fetched = m2._git(root, "fetch", "--quiet", url, branch)
        if fetched.returncode != 0 or not m2._commit_exists(root, remote):
            _fail("E03_REMOTO", f"no se pudo descargar el commit remoto {remote}")
    if not m2._ancestor(root, publicacion, remote):
        _fail("E03_REMOTO", f"la publicacion P3 {publicacion} no esta en la rama publicada del remoto ({remote})")
    if not (m2._ancestor(root, E02_P2, remote) and m2._ancestor(root, a3, remote)):
        _fail("E03_REMOTO_HISTORIA", "la historia remota no contiene A3 y P2")
    return remote


def official_e03(root: Path | None = None, publicacion: str | None = None) -> dict:
    """EJECUCION OFICIAL con el motor E03, vinculada a la publicacion P3. Devuelve la configuracion oficial de E02
    (``cfg``, sin cambios) y la trazabilidad de E03. No hay ningun parametro, variable ni modo que omita controles."""
    from ohlcv_dataroad.ingest import tda17_analysis as an
    root = Path(ROOT if root is None else root)
    if not isinstance(publicacion, str) or not _HEX40.fullmatch(publicacion):
        _fail("E03_P3_FORMATO", "la ejecucion oficial de E03 exige el SHA-1 completo de la publicacion P3")
    info = load_e03_aprobada(root)
    a3 = info["aprobacion"]["referencia_auditoria"]["commit_auditado"]
    if not m2._commit_exists(root, publicacion):
        _fail("E03_P3_EXISTE", f"la publicacion {publicacion} no existe en el repositorio")
    if publicacion == a3 or not m2._ancestor(root, a3, publicacion):
        _fail("E03_P3_A3", "el commit auditado A3 debe ser antecesor estricto de la publicacion P3")
    if not m2._ancestor(root, publicacion, "HEAD"):
        _fail("E03_P3_HEAD", "la publicacion P3 no es HEAD ni antecesora de HEAD")
    errors = verify_against_git_e03(root, publicacion)
    if errors:
        _fail("E03_P3_ARBOL", f"el arbol no coincide con la publicacion P3: {errors}")
    shown = m2._git(root, "show", f"{publicacion}:{REGISTRY_E03}")
    try:
        published = json.loads(shown.stdout.decode("utf-8")) if shown.returncode == 0 else {}
    except (ValueError, UnicodeDecodeError):
        published = {}
    ents = published.get("enmiendas") if isinstance(published, dict) else None
    pe = ents[0] if isinstance(ents, list) and len(ents) == 1 and isinstance(ents[0], dict) else {}
    cited = ((pe.get("aprobacion") or {}).get("referencia_auditoria") or {}).get("commit_auditado")
    if pe.get("estado") != e01.ESTADO_APROBADA or cited != a3:
        _fail("E03_P3_APROBADA", "la publicacion P3 no contiene la aprobacion de E03 que cita A3")
    remote = verify_remote_publication_e03(root, publicacion, a3)
    cfg = an.official_config(E02_P2, root)                 # E01, E02, A2, P2, protocolo estricto y remoto de P2
    info.update(ejecucion_oficial=True, publicacion_E03=publicacion, commit_auditado_E03=a3,
                remoto_E03={"repositorio": m2.AUTHORIZED_REMOTE, "ref": m2.REMOTE_BRANCH, "sha": remote})
    return {"cfg": cfg, "e03": info}


# --------------------------------------------------------------------------- 4. ruta oficial con el motor E03
COMPROBANTE = "TDA17_procedencia_E03.json"
COMPROBANTE_FIELDS = ["enmienda", "publicacion_E03", "commit_auditado_E03", "remoto_E03", "sha256_yaml_E03",
                      "publicacion_E02", "registro_E02", "sha256_registro_E02", "tablas", "hilos_blas",
                      "reanudacion", "motor"]


def official_resources(doc: dict):
    """Recursos de la ejecucion oficial: los del YAML y control de memoria ESTRICTO (ninguna tarea arranca si no cabe
    con la reserva; ``E03_MEMORIA_INSUFICIENTE``)."""
    from ohlcv_dataroad.tda17_e03 import motor as mo
    p = doc["paralelismo"]
    return mo.Recursos(procesos=int(p["procesos_max"]), reserva_mb=float(p["reserva_memoria_mb"]),
                       hilos_blas=None, cortes_por_bloque=int(p["cortes_por_bloque"]),
                       replicas_por_bloque=int(p["replicas_por_bloque"]),
                       minimo_medidas_eta=int(doc["progreso"]["minimo_medidas_eta"]),
                       intervalo_estado_s=float(doc["progreso"]["intervalo_s"]), max_sin_medir=int(p["max_sin_medir"]),
                       margen_pico=float(p["margen_pico"]), memoria_estricta=True,
                       espera_memoria_s=float(p["espera_memoria_s"]))


def _check_run_directory(root: Path, ejecucion: Path) -> Path:
    """Directorio NUEVO (no existe) y fuera del repositorio: ningun punto de control previo puede leerse."""
    ej = Path(ejecucion).resolve()
    if ej == root.resolve() or root.resolve() in ej.parents:
        _fail("E03_EJECUCION_EN_REPOSITORIO", f"el directorio de la ejecucion debe estar fuera del repositorio: {ej}")
    if os.path.lexists(ej):
        _fail("E03_EJECUCION_EXISTENTE", f"el directorio de una ejecucion oficial no puede existir: {ej}")
    return ej


def _official_run_e03(publicacion: str, root: Path, ejecucion: Path, antes_del_calculo=None):
    """Equivalente de ``sal._official_run`` con el motor E03, con 1 hilo de OpenBLAS fijado y verificado. La
    reanudacion no existe en esta ruta. ``antes_del_calculo``: comprobacion barata (de cfg) tras verificar la cadena
    oficial y antes de cargar las entradas (la publicacion rechaza ahi salidas ya existentes)."""
    from ohlcv_dataroad.ingest import tda17_entradas as ent
    from ohlcv_dataroad.ingest import tda17_salidas as sal
    from ohlcv_dataroad.tda17_e03 import motor as mo
    ejecucion = _check_run_directory(root, ejecucion)
    if mo.privada_mb() is None:
        _fail("E03_ENTORNO", "sin medida de la memoria privada de los procesos (psutil): el control de RAM no es verificable")
    ctx = official_e03(root, publicacion)
    cfg, info = ctx["cfg"], ctx["e03"]
    if antes_del_calculo is not None:
        antes_del_calculo(cfg)
    with mo.hilos_blas_fijados(HILOS_BLAS_OFICIALES) as efectivos:
        df, procedencia = ent.load_official_inputs(cfg, root)
        result, resumen = mo.run_th35_e03(df, cfg, ejecucion, official_resources(info["documento"]), oficial=True,
                                          identidad_extra={"publicacion_E02": E02_P2, "publicacion_E03": publicacion})
    if resumen["hilos_blas"] != efectivos or set(efectivos.values()) != {HILOS_BLAS_OFICIALES}:
        _fail("E03_ENTORNO", f"hilos BLAS de la ejecucion {resumen['hilos_blas']} distintos de los oficiales "
                             f"({HILOS_BLAS_OFICIALES})")
    result["procedencia"] = procedencia
    prov = result.get("procedencia") or {}
    if prov.get("publicacion_E02") != E02_P2 or \
            prov.get("sha256_efectiva") != cfg["trazabilidad_protocolo"]["sha256_efectiva"]:
        sal._reject("SALIDA_PROCEDENCIA", "el resultado no procede de la configuracion oficial de esta publicacion")
    blobs = {n: sal.csv_bytes(t) for n, t in sal.official_tables(result, cfg).items()}
    return cfg, info, result, blobs, resumen


def _comprobante(info: dict, registro_bytes: bytes, tablas: dict[str, str], resumen: dict) -> dict:
    from ohlcv_dataroad.ingest import tda17_salidas as sal
    return {"enmienda": "E03", "publicacion_E03": info["publicacion_E03"],
            "commit_auditado_E03": info["commit_auditado_E03"], "remoto_E03": info["remoto_E03"],
            "sha256_yaml_E03": E03_SHA256, "publicacion_E02": E02_P2, "registro_E02": sal.PROVENANCE,
            "sha256_registro_E02": sal._sha(registro_bytes), "tablas": tablas, "hilos_blas": HILOS_BLAS_OFICIALES,
            "reanudacion": REANUDACION_OFICIAL, "motor": json.loads(json.dumps(resumen, default=str))}


def _salidas_e03() -> list[str]:
    """Salidas de una publicacion E03: las oficiales de E02 y el comprobante."""
    from ohlcv_dataroad.ingest import tda17_salidas as sal
    return [*sal.OFFICIAL, COMPROBANTE]


def _recover_e03(out_dir: Path) -> None:
    """Bajo el cerrojo. Una publicacion E03 interrumpida antes de confirmar (preparacion presente y registro de E02
    ausente) puede haber promovido el comprobante, que ``sal._recover`` no conoce: se retira SOLO si es
    demostrablemente suyo (``sal._owned`` con el diario de la preparacion, comprobado ANTES de tocar nada) y DESPUES
    de que ``sal._recover`` haya retirado las salidas de E02 (si las rechaza, no se toca nada). Un comprobante ajeno
    bloquea la recuperacion sin tocar nada."""
    from ohlcv_dataroad.ingest import tda17_salidas as sal
    staging, comp = out_dir / sal.STAGING, out_dir / COMPROBANTE
    nuestro = False
    if staging.exists() and not os.path.lexists(out_dir / sal.PROVENANCE) and os.path.lexists(comp):
        try:
            journal = json.loads((staging / sal.JOURNAL).read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            journal = {}
        digest = journal.get(COMPROBANTE) if isinstance(journal, dict) else None
        if not sal._owned(staging / COMPROBANTE, comp, digest):
            _fail("E03_SALIDA_RECUPERACION", f"no puede demostrarse que {COMPROBANTE} pertenezca a la publicacion "
                                             f"interrumpida: se conserva y la recuperacion queda bloqueada")
        nuestro = True
    sal._recover(out_dir)
    if nuestro:
        comp.unlink()


def _destino_libre(out_dir: Path) -> None:
    """Bajo el cerrojo: recupera una publicacion interrumpida y rechaza si queda alguna salida (no se sobrescribe)."""
    from ohlcv_dataroad.ingest import tda17_salidas as sal
    _recover_e03(out_dir)
    existing = [n for n in _salidas_e03() if os.path.lexists(out_dir / n)]
    if existing:
        sal._reject("SALIDA_EXISTENTE", f"ya existen salidas oficiales (no se sobrescriben): {existing}")


def _publish_files_e03(out_dir: Path, blobs: dict[str, bytes], record: dict, comprobante: dict) -> None:
    """Transaccion de E02 (``sal._exclusive``, ``sal._stage_and_promote``: preparacion con fsync, diario de huellas,
    promocion sin sobrescribir y deshacer de lo propio) con el comprobante de E03 promovido DESPUES de las tablas y
    ANTES del registro de E02, que sigue siendo el ultimo y confirma la publicacion."""
    from ohlcv_dataroad.ingest import tda17_salidas as sal
    files = {**blobs, COMPROBANTE: (json.dumps(comprobante, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
             sal.PROVENANCE: (json.dumps(record, indent=2, ensure_ascii=False) + "\n").encode("utf-8")}
    with sal._exclusive(out_dir):
        _destino_libre(out_dir)
        sal._stage_and_promote(out_dir, files)


def official_publish_th35_e03(publicacion: str, root: Path | None = None, *, ejecucion: Path | str) -> dict:
    """Publica las salidas oficiales de TH35 calculadas con el motor E03 (``publicacion`` = P3): las MISMAS tablas y el
    MISMO registro de procedencia (vinculado a P2) que ``sal.official_publish_th35``, que no cambia, y el comprobante
    de E03 (``COMPROBANTE``) en la misma transaccion, antes del registro que la confirma. Las salidas existentes se
    rechazan antes del calculo y de nuevo bajo el cerrojo al publicar. ``ejecucion``: directorio NUEVO fuera del
    repositorio (progreso y puntos de control del motor)."""
    from ohlcv_dataroad.ingest import tda17_salidas as sal
    root = Path(ROOT if root is None else root)

    def destino_libre(cfg):
        out = root / sal.schema(cfg)["directorio"]
        out.mkdir(parents=True, exist_ok=True)
        with sal._exclusive(out):
            _destino_libre(out)

    cfg, info, result, blobs, resumen = _official_run_e03(publicacion, root, Path(ejecucion), destino_libre)
    record = sal._provenance(result, cfg, E02_P2, {n: sal._sha(b) for n, b in blobs.items()})
    registro_bytes = (json.dumps(record, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    comprobante = _comprobante(info, registro_bytes, record["tablas"], resumen)
    _publish_files_e03(root / sal.schema(cfg)["directorio"], blobs, record, comprobante)
    return record


def _check_comprobante(root: Path, out_dir: Path, record: dict, info: dict) -> dict:
    """El comprobante de E03 existe, es un fichero regular, tiene exactamente sus campos y vincula ESTA publicacion
    E03 (P3, A3, remoto, YAML) con los bytes exactos del registro de E02 y con las huellas de sus tablas. El SHA remoto
    registrado es la rama publicada al publicar: debe ser un commit que contiene P3 (la rama puede haber avanzado)."""
    from ohlcv_dataroad.ingest import tda17_salidas as sal
    path = out_dir / COMPROBANTE
    raw = sal._regular_bytes(path)
    if raw is None:
        _fail("E03_COMPROBANTE", f"falta el comprobante {COMPROBANTE}" if not os.path.lexists(path) else
              f"{COMPROBANTE} no es un fichero regular (enlace o punto de reanalisis)")
    try:
        comp = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        comp = None
    if not isinstance(comp, dict) or list(comp) != COMPROBANTE_FIELDS:
        _fail("E03_COMPROBANTE", f"{COMPROBANTE} no es JSON con exactamente los campos {COMPROBANTE_FIELDS}")
    registro = sal._regular_bytes(out_dir / sal.PROVENANCE)
    motor = comp["motor"] if isinstance(comp["motor"], dict) else {}
    esperado = {"enmienda": "E03", "publicacion_E03": info["publicacion_E03"],
                "commit_auditado_E03": info["commit_auditado_E03"],
                "sha256_yaml_E03": E03_SHA256, "publicacion_E02": E02_P2, "registro_E02": sal.PROVENANCE,
                "sha256_registro_E02": None if registro is None else sal._sha(registro), "tablas": record["tablas"],
                "hilos_blas": HILOS_BLAS_OFICIALES, "reanudacion": REANUDACION_OFICIAL}
    distintos = [k for k, v in esperado.items() if comp[k] != v]
    remoto = comp["remoto_E03"] if isinstance(comp["remoto_E03"], dict) else {}
    sha = remoto.get("sha")
    identidad = {k: info["remoto_E03"][k] for k in ("repositorio", "ref")}
    if list(remoto) != ["repositorio", "ref", "sha"] or {k: remoto[k] for k in identidad} != identidad \
            or not isinstance(sha, str) or not _HEX40.fullmatch(sha) or not m2._commit_exists(root, sha) \
            or not m2._ancestor(root, info["publicacion_E03"], sha):
        distintos.append("remoto_E03")
    hilos = motor.get("hilos_blas")
    if not isinstance(hilos, dict) or not hilos or set(hilos.values()) != {HILOS_BLAS_OFICIALES} or \
            (motor.get("recursos") or {}).get("memoria_estricta") is not True:
        distintos.append("motor")
    distintos = [k for k in COMPROBANTE_FIELDS if k in distintos]
    if distintos:
        _fail("E03_COMPROBANTE_VINCULO", f"{COMPROBANTE} no corresponde a esta publicacion E03: {distintos}")
    return comp


def check_official_publication_e03(publicacion: str, root: Path | None = None) -> dict:
    """Comprueba una publicacion E03 ya escrita sin recalcular: cadena oficial de E03 (P3, A3, remoto, E02), la
    publicacion de E02 completa y confirmada (``sal.check_official_publication``, sin cambios) y el comprobante de
    E03 vinculado a ella. Devuelve el registro de E02."""
    from ohlcv_dataroad.ingest import tda17_salidas as sal
    root = Path(ROOT if root is None else root)
    ctx = official_e03(root, publicacion)
    record = sal.check_official_publication(E02_P2, root)
    _check_comprobante(root, root / sal.schema(ctx["cfg"])["directorio"], record, ctx["e03"])
    return record


def reproduce_official_publication_e03(publicacion: str, root: Path | None = None, *, ejecucion: Path | str) -> dict:
    """``check_official_publication_e03`` y recalculo COMPLETO con el motor E03 en una ejecucion nueva (1 hilo de
    OpenBLAS, sin puntos de control previos): cada tabla publicada debe ser identica byte a byte."""
    from ohlcv_dataroad.ingest import tda17_salidas as sal
    root = Path(ROOT if root is None else root)
    record = check_official_publication_e03(publicacion, root)
    cfg, _info, result, blobs, _ = _official_run_e03(publicacion, root, Path(ejecucion))
    out_dir = root / sal.schema(cfg)["directorio"]
    if sorted(blobs) != sorted(record["tablas"]) or result["estado"] != record["estado"]:
        sal._reject("SALIDA_NO_REPRODUCIBLE", "el recalculo no produce el mismo conjunto de tablas ni el mismo estado")
    differ = [n for n, b in blobs.items() if sal._regular_bytes(out_dir / n) != b]
    if differ:
        sal._reject("SALIDA_NO_REPRODUCIBLE", f"tablas publicadas distintas del recalculo oficial: {differ}")
    return record


def reproduce_with_e02_engine_e03(publicacion: str, root: Path | None = None) -> dict:
    """Reproduccion con el MOTOR E02 (``sal.reproduce_official_publication``, sin cambios) en el mismo entorno
    numerico que la ejecucion oficial de E03, tras ``check_official_publication_e03`` (cadena oficial de E03 y
    comprobante) y con 1 hilo de OpenBLAS fijado."""
    from ohlcv_dataroad.ingest import tda17_salidas as sal
    from ohlcv_dataroad.tda17_e03 import motor as mo
    root = Path(ROOT if root is None else root)
    check_official_publication_e03(publicacion, root)
    with mo.hilos_blas_fijados(HILOS_BLAS_OFICIALES):
        return sal.reproduce_official_publication(E02_P2, root)


# --------------------------------------------------------------------------- 5. herramienta de verificacion
def verify(root: Path | None = None, referencia_git: str | None = None) -> tuple[list[str], dict]:
    """Estado de E03 sin ejecutar nada: registro, huella y contenido del YAML; si esta aprobada, la aprobacion
    preparatoria y, con ``referencia_git`` (P3), la cadena oficial completa."""
    root = Path(ROOT if root is None else root)
    errors, info = [], {}
    try:
        info = effective_e03(root, borrador=True)
        if info["estado"] == e01.ESTADO_APROBADA:
            info = official_e03(root, referencia_git)["e03"] if referencia_git else load_e03_aprobada(root)
        try:
            official_e03(root, referencia_git or "0" * 40)
            info["ruta_oficial_e03"] = "ACTIVABLE"
        except ProtocolNotAuthorizedError as exc:
            info["ruta_oficial_e03"] = f"BLOQUEADA: {exc}"
    except ProtocolNotAuthorizedError as exc:
        errors.append(str(exc))
    if referencia_git:
        errors += verify_against_git_e03(root, referencia_git)
    return errors, info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ohlcv_dataroad.tda17_e03.enmienda")
    ap.add_argument("accion", choices=["verificar"])
    ap.add_argument("--referencia-git", metavar="P3", help="compara E03 con la publicacion P3 y verifica la ruta oficial")
    a = ap.parse_args(argv)
    errors, info = verify(ROOT, a.referencia_git)
    if errors:
        print("VERIFICACION DE E03: NO CONFORME")
        print("\n".join(errors))
        return 1
    print(f"VERIFICACION DE E03: CONFORME (estado {info['estado']}; YAML {info['sha256_yaml'][:12]}; hilos BLAS "
          f"{HILOS_BLAS_OFICIALES}; reanudacion oficial {REANUDACION_OFICIAL}; ruta oficial E03 {info['ruta_oficial_e03']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
