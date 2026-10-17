"""Enmiendas del protocolo TDA-17 sin modificar el protocolo congelado (configs/tda17.yaml, commit 9cdea2f).

Uso (desde la raiz del repositorio):

    python -m ohlcv_dataroad.ingest.tda17_enmiendas verificar [--referencia-git COMMIT]
    python -m ohlcv_dataroad.ingest.tda17_enmiendas efectiva --salida RUTA [--borrador]

Mecanismo (en este orden; cualquier fallo lanza ``ProtocolNotAuthorizedError`` y NUNCA se vuelve en
silencio al comportamiento anterior):

1. Protocolo base: el manifiesto congelado coincide con la huella fijada aqui (``FROZEN_MANIFEST_SHA256``)
   y declara CONGELADO; los 9 ficheros protegidos coinciden con sus huellas del manifiesto y con las
   auditadas del registro de congelamiento; el YAML base es el congelado (``tda17_indicators.load_protocol``).
2. Huella autorizada: cada enmienda requerida (``REQUIRED_AMENDMENTS``) existe, su SHA-256 coincide con el
   fijado en ``AUTHORIZED_AMENDMENTS`` (en este modulo, fuera del registro) y con el del registro, y declara
   el mismo protocolo base.
3. Aprobacion: el registro (``reports/mnq/TDA17_enmiendas.json``) declara la enmienda APROBADA con fecha ISO
   valida (entre la preparacion y hoy), responsable declarado, referencia de auditoria identificable (documento
   y commit Git auditado, 40 hexadecimales) y las huellas de TODOS los ficheros auditados (``AUDITED_FILES``),
   que deben coincidir con los del arbol de trabajo. Ademas, OBLIGATORIAMENTE al cargar, el commit auditado
   existe en el repositorio Git de ``root`` y contiene cada fichero auditado con esa huella; sin Git, sin
   repositorio, con un commit inexistente, un fichero ausente o una huella distinta, se rechaza.
   Una enmienda no autorizada en el registro se rechaza.
   El registro es una DECLARACION de aprobacion, no una firma criptografica: no autentica a quien aprueba.
   Lo que si se comprueba es que lo aprobado es exactamente lo auditado (huellas del arbol y del commit
   auditado al cargar) y, antes de ejecutar el pipeline real, ``verificar --referencia-git P`` (el commit
   publicado contiene exactamente el arbol verificado).
4. Lista blanca: solo se aplican las operaciones (op, ruta) de ``WHITELIST``, todas y exactamente ellas;
   despues se comprueba que la configuracion efectiva solo difiere de la base en esas rutas.
5. Trazabilidad: la configuracion efectiva lleva ``trazabilidad_protocolo`` con las huellas del protocolo base,
   del manifiesto, de cada enmienda, del registro y de la propia configuracion efectiva.

Sin referencias circulares: el YAML de la enmienda cita las huellas del protocolo base; el registro y este
modulo citan la huella del YAML de la enmienda; nada cita la huella del registro ni la de este modulo (la del
registro y la de la configuracion efectiva solo se calculan al cargar). El commit que contiene la enmienda no
puede citarse dentro de ella: se verifica despues con ``verificar --referencia-git COMMIT``.

Aprobacion (fuera de este modulo, tras la auditoria independiente):
  A. commit A con la version auditada y el registro aun PENDIENTE_AUDITORIA;
  B. en el registro: ``estado`` APROBADA y ``aprobacion`` = {fecha, aprobada_por,
     referencia_auditoria: {documento, commit_auditado: A}, sha256_auditados: huellas de AUDITED_FILES en A};
  C. commit P (publicado) y ``verificar --referencia-git P`` (obligatorio antes del pipeline real).
El registro cita A (anterior a el) y las huellas de ficheros que no citan el registro: no hay ciclo. Ni el YAML
de la enmienda ni su huella cambian al aprobarse.

No calcula indicadores ni lee datos de MNQ; ``verificar`` reutiliza la verificacion estricta oficial
(que solo calcula SHA-256 y nunca abre data/raw ni el holdout).
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

from ohlcv_dataroad.ingest import tda17_indicators as ind
from ohlcv_dataroad.ingest import tda17_protocol_freeze as frz

ROOT = Path(__file__).resolve().parents[3]
REGISTRY_JSON = "reports/mnq/TDA17_enmiendas.json"
MANIFEST_JSON = frz.MANIFEST_JSON
PROTOCOL_YAML = frz.PROTOCOL_YAML
# Huella (LF) del manifiesto congelado en ind.FROZEN_COMMIT: ancla los 9 ficheros protegidos.
FROZEN_MANIFEST_SHA256 = "1b14b65a07dd207625071c3f657863574a0b4ab801a92bd917caace24f8db4d0"

ESTADO_PENDIENTE = "PENDIENTE_AUDITORIA"
ESTADO_APROBADA = "APROBADA"
APPROVAL_FIELDS = ("fecha", "aprobada_por", "referencia_auditoria", "sha256_auditados")
AUDIT_REFERENCE_FIELDS = ("documento", "commit_auditado")

# Huellas AUTORIZADAS (SHA-256 tras normalizar CRLF a LF). Cambiar el YAML de una enmienda exige cambiar esta
# constante (y queda a la vista en el diff y en la verificacion contra Git).
AUTHORIZED_AMENDMENTS = {
    "E01": {"fichero": "configs/tda17_enmienda_01.yaml",
            "sha256": "62b91747d5102aef51cc2c51f014e20d3ba1e635e98cddee7c57ecebefa5d5cb"},
}
REQUIRED_AMENDMENTS = ("E01",)

# Lista blanca: las UNICAS operaciones que una enmienda puede aplicar (op, ruta).
WHITELIST = {
    "E01": (
        ("anadir", "catalogo.A8.dx_no_definido"),
        ("anadir", "calentamiento.clases.CONTADOR_CONS.A8_dx_no_definido"),
        ("sustituir", "calentamiento.tres_niveles"),
        ("sustituir", "calentamiento.registro"),
        ("anadir", "reglas_frontera.ventanas_marcas"),
        ("anadir", "sensibilidades.marcas_por_serie"),
    ),
}

# Ficheros que implementan o verifican las enmiendas: la aprobacion cubre sus huellas (AUDITED_FILES) y la
# verificacion contra Git los compara todos, ademas del registro (AMENDMENT_TRACKED_FILES).
IMPLEMENTATION_FILES = [
    "src/ohlcv_dataroad/ingest/tda17_enmiendas.py",
    "src/ohlcv_dataroad/ingest/tda17_indicators.py",
    "src/ohlcv_dataroad/ingest/tda17_primitives.py",
    "tests/test_tda17_indicators.py",
    "tests/test_tda17_enmienda_01.py",
    "tests/tda17_synthetic.py",
]
AUDITED_FILES = [a["fichero"] for a in AUTHORIZED_AMENDMENTS.values()] + IMPLEMENTATION_FILES
AMENDMENT_TRACKED_FILES = [REGISTRY_JSON] + AUDITED_FILES

ProtocolNotAuthorizedError = ind.ProtocolNotAuthorizedError


# --------------------------------------------------------------------------- huellas
def sha256_text_file(path: Path) -> str:
    return frz.sha256_file(Path(path))


def canonical_sha256(obj) -> str:
    data = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- 1. protocolo base
def verify_base(root: Path) -> dict:
    """Protocolo base intacto: manifiesto congelado (huella fija), 9 ficheros protegidos y YAML congelado."""
    root = Path(root)
    manifest_path = root / MANIFEST_JSON
    if not manifest_path.is_file():
        raise ProtocolNotAuthorizedError(f"falta el manifiesto congelado {MANIFEST_JSON}")
    digest = sha256_text_file(manifest_path)
    if digest != FROZEN_MANIFEST_SHA256:
        raise ProtocolNotAuthorizedError(f"{MANIFEST_JSON} no es el congelado en {ind.FROZEN_COMMIT[:7]} (SHA-256 {digest})")
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    if m.get("estado") != frz.ESTADO_CONGELADO or list(m.get("sha256_protocolo", {})) != frz.PROTOCOL_FILES:
        raise ProtocolNotAuthorizedError("el manifiesto no acredita un protocolo CONGELADO con los 9 ficheros protegidos")
    errors = []
    for rel, expected in m["sha256_protocolo"].items():
        path = root / rel
        if not path.is_file():
            errors.append(f"FALTA {rel}")
        elif sha256_text_file(path) != expected:
            errors.append(f"HASH DISTINTO {rel}")
    errors += frz.verify_freeze_record(root, m)
    if errors:
        raise ProtocolNotAuthorizedError("ficheros protegidos del protocolo base alterados: " + "; ".join(errors))
    cfg = ind.load_protocol(root / PROTOCOL_YAML, manifest_path)
    return cfg


def _base_reference() -> dict:
    return {"id": "TDA17_PROTOCOLO_v1", "commit": ind.FROZEN_COMMIT, "sha256_yaml": ind.FROZEN_YAML_SHA256,
            "sha256_manifiesto": FROZEN_MANIFEST_SHA256}


# --------------------------------------------------------------------------- 2-3. huella y aprobacion
def load_registry(root: Path) -> dict:
    path = Path(root) / REGISTRY_JSON
    if not path.is_file():
        raise ProtocolNotAuthorizedError(f"falta el registro de enmiendas {REGISTRY_JSON}")
    reg = json.loads(path.read_text(encoding="utf-8"))
    if reg.get("protocolo_base") != _base_reference():
        raise ProtocolNotAuthorizedError("el registro de enmiendas no declara el protocolo base congelado")
    entries = reg.get("enmiendas")
    if not isinstance(entries, list) or len({e.get("id") for e in entries}) != len(entries):
        raise ProtocolNotAuthorizedError("registro de enmiendas mal formado")
    unknown = [e.get("id") for e in entries if e.get("id") not in AUTHORIZED_AMENDMENTS]
    if unknown:
        raise ProtocolNotAuthorizedError(f"enmiendas no autorizadas en el registro: {unknown}")
    return reg


def verify_amendment(root: Path, reg: dict, amendment_id: str, *, require_approved: bool = True) -> tuple[dict, dict]:
    """Devuelve (documento de la enmienda, entrada del registro) si la huella autorizada, la del registro y la
    del fichero coinciden, el protocolo base es el congelado y (si se exige) la enmienda esta APROBADA."""
    auth = AUTHORIZED_AMENDMENTS[amendment_id]
    entry = next((e for e in reg["enmiendas"] if e.get("id") == amendment_id), None)
    if entry is None:
        raise ProtocolNotAuthorizedError(f"{amendment_id}: ausente del registro de enmiendas")
    if entry.get("fichero") != auth["fichero"]:
        raise ProtocolNotAuthorizedError(f"{amendment_id}: el registro apunta a otro fichero ({entry.get('fichero')})")
    path = Path(root) / auth["fichero"]
    if not path.is_file():
        raise ProtocolNotAuthorizedError(f"{amendment_id}: falta {auth['fichero']}")
    digest = sha256_text_file(path)
    if digest != auth["sha256"]:
        raise ProtocolNotAuthorizedError(f"{amendment_id}: {auth['fichero']} no es la version autorizada (SHA-256 {digest})")
    if entry.get("sha256") != auth["sha256"]:
        raise ProtocolNotAuthorizedError(f"{amendment_id}: la huella del registro no es la autorizada")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    head = doc.get("enmienda", {}) if isinstance(doc, dict) else {}
    if head.get("id") != amendment_id or head.get("protocolo_base") != _base_reference():
        raise ProtocolNotAuthorizedError(f"{amendment_id}: identificador o protocolo base incompatibles")
    if entry.get("estado") not in (ESTADO_PENDIENTE, ESTADO_APROBADA):
        raise ProtocolNotAuthorizedError(f"{amendment_id}: estado desconocido {entry.get('estado')}")
    if require_approved and entry.get("estado") != ESTADO_APROBADA:
        raise ProtocolNotAuthorizedError(f"{amendment_id}: no aprobada (estado {entry.get('estado')}); "
                                         "no se aplica ni se recurre al comportamiento anterior")
    if entry.get("estado") == ESTADO_APROBADA:
        check_approval(root, entry, amendment_id)
        require_audited_commit(root, entry, amendment_id)
    elif entry.get("aprobacion") is not None:
        raise ProtocolNotAuthorizedError(f"{amendment_id}: PENDIENTE con datos de aprobacion")
    return doc, entry


_HEX40 = re.compile(r"[0-9a-f]{40}")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def check_approval(root: Path, entry: dict, amendment_id: str, today: dt.date | None = None) -> None:
    """Declaracion de aprobacion completa y coherente con lo auditado (no es una firma criptografica).

    fecha: ISO AAAA-MM-DD valida, no anterior a la preparacion ni posterior a hoy; aprobada_por: texto no vacio
    (identidad declarada, no autenticada); referencia_auditoria: documento no vacio y commit_auditado de 40
    hexadecimales; sha256_auditados: exactamente AUDITED_FILES, cada huella igual a la del arbol de trabajo."""
    def fail(msg):
        raise ProtocolNotAuthorizedError(f"{amendment_id}: aprobacion invalida: {msg}")
    a = entry.get("aprobacion")
    if not isinstance(a, dict) or set(a) != set(APPROVAL_FIELDS):
        fail(f"campos requeridos {list(APPROVAL_FIELDS)}")
    try:
        if not isinstance(a["fecha"], str) or not _ISO_DATE.fullmatch(a["fecha"]):
            raise ValueError
        fecha = dt.date.fromisoformat(a["fecha"])
        prep = dt.date.fromisoformat(str(entry.get("fecha_preparacion")))
    except ValueError:
        fail(f"fecha no ISO valida ({a.get('fecha')!r}) o fecha_preparacion ausente")
    if not prep <= fecha <= (today or dt.date.today()):
        fail(f"fecha {fecha} fuera de [{prep}, hoy]")
    if not isinstance(a["aprobada_por"], str) or not a["aprobada_por"].strip():
        fail("aprobada_por vacio")
    ref = a["referencia_auditoria"]
    if not isinstance(ref, dict) or set(ref) != set(AUDIT_REFERENCE_FIELDS):
        fail(f"referencia_auditoria debe tener {list(AUDIT_REFERENCE_FIELDS)}")
    if not isinstance(ref["documento"], str) or not ref["documento"].strip():
        fail("referencia_auditoria.documento vacio")
    if not isinstance(ref["commit_auditado"], str) or not _HEX40.fullmatch(ref["commit_auditado"]):
        fail("referencia_auditoria.commit_auditado no es un SHA-1 Git completo (40 hexadecimales)")
    hashes = a["sha256_auditados"]
    if not isinstance(hashes, dict) or sorted(hashes) != sorted(AUDITED_FILES):
        fail("sha256_auditados no cubre exactamente los ficheros auditados")
    changed = [rel for rel in AUDITED_FILES
               if not (Path(root) / rel).is_file() or sha256_text_file(Path(root) / rel) != hashes[rel]]
    if changed:
        fail(f"ficheros distintos de la version auditada: {changed}")


GIT_TIMEOUT_S = 60


def _git(root: Path, *args: str):
    """Ejecuta git en ``root``. Devuelve (resultado, None) o (None, motivo) si git no puede ejecutarse."""
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, timeout=GIT_TIMEOUT_S), None
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def verify_approval_against_git(root: Path, entry: dict) -> list[str]:
    """La version aprobada es la del commit auditado: ``root`` es la raiz de un repositorio Git, el commit
    ``commit_auditado`` existe y cada fichero auditado suyo tiene la huella registrada en ``sha256_auditados``.
    Devuelve la lista de errores; cualquier fallo de acceso a Git es un error."""
    a = entry["aprobacion"]
    commit = a["referencia_auditoria"]["commit_auditado"]
    tag = f"[aprobacion {entry['id']}]"
    top, err = _git(root, "rev-parse", "--show-toplevel")
    if err or top.returncode != 0:
        return [f"{tag} sin acceso al repositorio Git de {root}: {err or top.stderr.decode(errors='replace').strip()}"]
    try:
        same = os.path.samefile(top.stdout.decode(errors="replace").strip(), root)
    except OSError:
        same = False
    if not same:
        return [f"{tag} {root} no es la raiz del repositorio Git ({top.stdout.decode(errors='replace').strip()})"]
    found, err = _git(root, "cat-file", "-e", f"{commit}^{{commit}}")
    if err:
        return [f"{tag} error de acceso a Git: {err}"]
    if found.returncode != 0:
        return [f"{tag} el commit auditado {commit} no existe en el repositorio"]
    errors = []
    for rel in AUDITED_FILES:
        shown, err = _git(root, "show", f"{commit}:{rel}")
        if err:
            errors.append(f"{tag} error de acceso a Git al leer {rel}: {err}")
        elif shown.returncode != 0:
            errors.append(f"{tag} el commit auditado no contiene {rel}")
        elif frz.sha256_bytes(shown.stdout, Path(rel).suffix) != a["sha256_auditados"][rel]:
            errors.append(f"{tag} {rel} del commit auditado difiere de la huella aprobada")
    return errors


def require_audited_commit(root: Path, entry: dict, amendment_id: str) -> None:
    """Barrera obligatoria de la via de produccion: sin commit auditado verificable no hay aprobacion."""
    errors = verify_approval_against_git(Path(root), entry)
    if errors:
        raise ProtocolNotAuthorizedError(f"{amendment_id}: la aprobacion no corresponde a un commit auditado "
                                         "verificable: " + "; ".join(errors))


# --------------------------------------------------------------------------- 4. lista blanca
def _resolve_parent(cfg: dict, ruta: str):
    """Devuelve (contenedor, clave) de ``ruta``; en listas de elementos con ``id`` el segmento es el id."""
    parts = ruta.split(".")
    node = cfg
    for seg in parts[:-1]:
        if isinstance(node, list):
            hits = [x for x in node if isinstance(x, dict) and x.get("id") == seg]
            if len(hits) != 1:
                raise ProtocolNotAuthorizedError(f"ruta inexistente en el protocolo base: {ruta}")
            node = hits[0]
        elif isinstance(node, dict) and seg in node:
            node = node[seg]
        else:
            raise ProtocolNotAuthorizedError(f"ruta inexistente en el protocolo base: {ruta}")
    if not isinstance(node, dict):
        raise ProtocolNotAuthorizedError(f"ruta no modificable: {ruta}")
    return node, parts[-1]


def apply_amendment(base: dict, doc: dict, amendment_id: str) -> dict:
    """Configuracion efectiva = base + operaciones de la enmienda, SOLO si son exactamente las de la lista blanca."""
    ops = doc.get("cambios")
    if not isinstance(ops, list) or not all(isinstance(o, dict) and set(o) == {"op", "ruta", "valor"} for o in ops):
        raise ProtocolNotAuthorizedError(f"{amendment_id}: 'cambios' mal formado")
    declared = [(o["op"], o["ruta"]) for o in ops]
    allowed = WHITELIST[amendment_id]
    outside = [d for d in declared if d not in allowed]
    if outside:
        raise ProtocolNotAuthorizedError(f"{amendment_id}: operaciones fuera de la lista blanca: {outside}")
    if sorted(declared) != sorted(allowed):
        raise ProtocolNotAuthorizedError(f"{amendment_id}: faltan o se repiten operaciones de la lista blanca")
    eff = copy.deepcopy(base)
    for o in ops:
        parent, key = _resolve_parent(eff, o["ruta"])
        if o["op"] == "anadir" and key in parent:
            raise ProtocolNotAuthorizedError(f"{amendment_id}: 'anadir' sobre una clave existente: {o['ruta']}")
        if o["op"] == "sustituir" and (key not in parent or parent[key] == o["valor"]):
            raise ProtocolNotAuthorizedError(f"{amendment_id}: 'sustituir' sin clave previa o con el mismo valor que la base: {o['ruta']}")
        parent[key] = copy.deepcopy(o["valor"])
    check_changed_paths(base, eff, {r for _, r in allowed}, amendment_id)
    ind._validate_cfg_structure(eff)
    return eff


def check_changed_paths(base: dict, eff: dict, allowed_paths: set[str], amendment_id: str) -> None:
    """Segunda barrera, independiente de la lista de operaciones: la configuracion efectiva difiere de la base
    exactamente en las rutas autorizadas (ni mas, ni menos)."""
    changed = diff_paths(base, eff)
    if changed != allowed_paths:
        raise ProtocolNotAuthorizedError(f"{amendment_id}: la configuracion efectiva no cambia exactamente las rutas "
                                         f"autorizadas (de mas: {sorted(changed - allowed_paths)}; "
                                         f"sin cambio: {sorted(allowed_paths - changed)})")


def diff_paths(a, b, prefix: str = "") -> set[str]:
    """Rutas (con ids de elementos en las listas del catalogo) en las que ``a`` y ``b`` difieren."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = set()
        for k in set(a) | set(b):
            p = f"{prefix}.{k}" if prefix else str(k)
            if k not in a or k not in b:
                out.add(p)
            else:
                out |= diff_paths(a[k], b[k], p)
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and \
            all(isinstance(x, dict) and "id" in x for x in a + b) and [x["id"] for x in a] == [x["id"] for x in b]:
        out = set()
        for x, y in zip(a, b):
            out |= diff_paths(x, y, f"{prefix}.{x['id']}")
        return out
    return set() if a == b else {prefix}


# --------------------------------------------------------------------------- 5. configuracion efectiva
def load_effective_protocol(root: Path | None = None) -> dict:
    """UNICA via de produccion: protocolo congelado + enmiendas requeridas APROBADAS, con trazabilidad."""
    return _effective(Path(ROOT if root is None else root), require_approved=True)


def _effective(root: Path, *, require_approved: bool) -> dict:
    base = verify_base(root)
    reg = load_registry(root)
    eff, trace = base, []
    for aid in REQUIRED_AMENDMENTS:
        doc, entry = verify_amendment(root, reg, aid, require_approved=require_approved)
        eff = apply_amendment(eff, doc, aid)
        trace.append({"id": aid, "fichero": entry["fichero"], "sha256": entry["sha256"], "estado": entry["estado"],
                      "aprobacion": entry.get("aprobacion")})
    eff["trazabilidad_protocolo"] = {
        "base": {**_base_reference(), "ficheros_protegidos_verificados": len(frz.PROTOCOL_FILES)},
        "enmiendas": trace,
        "sha256_registro": sha256_text_file(root / REGISTRY_JSON),
        "sha256_efectiva": canonical_sha256({k: v for k, v in eff.items() if k != "trazabilidad_protocolo"}),
        "modo": "PRODUCCION" if require_approved else "BORRADOR",
        "activable": require_approved,       # solo la via de produccion (todas APROBADAS y verificadas) activa
    }
    return eff


def effective_document(root: Path | None = None, *, borrador: bool = False) -> dict:
    """Configuracion efectiva exportable. ``borrador`` admite enmiendas PENDIENTES solo para auditarlas: el
    documento lo declara (``modo: BORRADOR``, ``activable: false``, aunque el registro este APROBADO) y el motor
    nunca lo usa (solo ``load_effective_protocol``)."""
    return _effective(Path(ROOT if root is None else root), require_approved=not borrador)


def require_s2_scale_assignment(cfg: dict) -> dict:
    """S2 se limita a las escalas S y M (sensibilidades.S2), pero ni el protocolo congelado ni E01 asignan cada
    configuracion a S, M o L. Hasta que una enmienda aprobada lo haga, S2 NO puede ejecutarse.
    Requisito ya decidido para la fase de analisis: la exclusion por union de marcas en calculos multiserie
    debe acompanarse de controles de poblacion efectiva y evaluabilidad, sin modificar umbrales congelados."""
    scales = (cfg.get("sensibilidades") or {}).get("escalas_S2")
    if not scales:
        raise ProtocolNotAuthorizedError("S2 no ejecutable: la asignacion de configuraciones a las escalas S/M/L "
                                         "esta pendiente de decision y no figura en ninguna enmienda aprobada")
    return scales


# --------------------------------------------------------------------------- verificacion posterior
def verify_against_git(root: Path, commit: str) -> list[str]:
    """Referencia externa: protocolo base (9 ficheros y manifiesto), enmiendas, registro, modulo de autorizacion,
    motor, primitivas y pruebas de fase 1 y de E01 (AMENDMENT_TRACKED_FILES) coinciden con los de ``commit``."""
    errors = frz.verify_against_git(root, commit)
    for rel in AMENDMENT_TRACKED_FILES:
        shown = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=root, capture_output=True)
        if shown.returncode != 0:
            errors.append(f"[git {commit}] no contiene {rel}")
        elif not (root / rel).is_file():
            errors.append(f"[git {commit}] FALTA en el arbol de trabajo: {rel}")
        elif frz.sha256_bytes(shown.stdout, Path(rel).suffix) != sha256_text_file(root / rel):
            errors.append(f"[git {commit}] difiere del commit de referencia: {rel}")
    return errors


def verify(root: Path, referencia_git: str | None = None) -> tuple[list[str], dict]:
    """Verificacion estricta oficial del protocolo base (188 artefactos incluidos), contra su commit de
    congelamiento, y de las enmiendas (huella, registro, lista blanca; si estan APROBADAS, aprobacion y commit
    auditado), sin exigir aprobacion. ``activable`` solo si la via de produccion carga."""
    errors = frz.verify_strict(root) + frz.verify_against_git(root, ind.FROZEN_COMMIT)
    info = {}
    try:
        info = _effective(root, require_approved=False)["trazabilidad_protocolo"]
        if all(t["estado"] == ESTADO_APROBADA for t in info["enmiendas"]):
            info = load_effective_protocol(root)["trazabilidad_protocolo"]
    except ProtocolNotAuthorizedError as exc:
        errors.append(f"enmiendas: {exc}")
    if referencia_git:
        errors += verify_against_git(root, referencia_git)
    return errors, info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("accion", choices=["verificar", "efectiva"])
    ap.add_argument("--referencia-git", metavar="COMMIT",
                    help="verificar: compara ademas protocolo base, enmiendas, registro y este modulo con los de COMMIT")
    ap.add_argument("--salida", metavar="RUTA", help="efectiva: fichero JSON de la configuracion efectiva")
    ap.add_argument("--borrador", action="store_true", help="efectiva: admite enmiendas PENDIENTES (activable: false)")
    a = ap.parse_args(argv)
    try:
        if a.accion == "verificar":
            errors, info = verify(ROOT, a.referencia_git)
            if errors:
                print("VERIFICACION DE ENMIENDAS: NO CONFORME")
                print("\n".join(errors))
                return 1
            estados = ", ".join(f"{t['id']} {t['estado']} {t['sha256'][:12]}" for t in info["enmiendas"])
            print(f"VERIFICACION DE ENMIENDAS: CONFORME (base {ind.FROZEN_COMMIT[:7]}; {estados}; "
                  f"activable: {'SI' if info['activable'] else 'NO'}; efectiva {info['sha256_efectiva'][:12]})")
        else:
            if not a.salida:
                ap.error("efectiva exige --salida")
            doc = effective_document(ROOT, borrador=a.borrador)
            Path(a.salida).write_text(json.dumps(doc, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
            t = doc["trazabilidad_protocolo"]
            print(f"configuracion efectiva escrita en {a.salida} (sha256_efectiva {t['sha256_efectiva']}; "
                  f"activable: {'SI' if t['activable'] else 'NO'})")
    except ProtocolNotAuthorizedError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
