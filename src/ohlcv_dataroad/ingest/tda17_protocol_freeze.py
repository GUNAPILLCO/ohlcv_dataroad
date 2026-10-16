"""Verificacion estricta y transicion atomica del protocolo TDA-17 (PENDIENTE_AUDITORIA_FINAL -> CONGELADO).

Uso (desde la raiz del repositorio):

    python -m ohlcv_dataroad.ingest.tda17_protocol_freeze verificar [--referencia-git COMMIT]
    python -m ohlcv_dataroad.ingest.tda17_protocol_freeze manifiesto            # solo en PENDIENTE_AUDITORIA_FINAL
    python -m ohlcv_dataroad.ingest.tda17_protocol_freeze congelar --confirmar

- ``verificar`` es la verificacion OFICIAL y siempre ESTRICTA: falla si falta o difiere cualquier
  fichero del protocolo, cualquiera de las entradas obligatorias o cualquiera de los artefactos
  historicos del manifiesto. No admite omisiones (el modo portable existe solo en los tests).
- ``manifiesto`` recalcula UNICAMENTE los SHA-256 de los ficheros del protocolo. Los hashes de las
  entradas y de los artefactos historicos no se recalculan nunca: se verifican.
- ``congelar`` aplica las transiciones de estado declaradas en ``TRANSICIONES`` a YAML e informe,
  reescribe el manifiesto con el registro de congelamiento y vuelve a verificar en modo estricto.
  Todo se calcula en memoria antes de escribir; si algo falla durante o despues de la escritura,
  se restauran los bytes originales de los tres ficheros. El registro guarda las huellas auditadas
  de TODOS los ficheros del protocolo.
- Referencia externa: el registro vive dentro del manifiesto, asi que una reescritura completa y
  coherente del manifiesto solo se detecta fuera de el: ``verificar --referencia-git <commit del
  congelamiento>`` y los SHA-256 auditados publicados en la revision.

No calcula indicadores ni analiza datos de MNQ: solo lee bytes para calcular SHA-256. Nunca abre
data/raw/ ni el holdout: todas las rutas del manifiesto se validan (texto y resolucion de ruta, sin
leer contenido) ANTES de calcular cualquier hash, y una sola ruta prohibida detiene la verificacion.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

import yaml

ROOT = Path(__file__).resolve().parents[3]

PROTOCOL_YAML = "configs/tda17.yaml"
REPORT_MD = "reports/mnq/TDA17_protocolo_congelamiento.md"
MANIFEST_JSON = "reports/mnq/TDA17_manifiesto_protocolo.json"
SNAPSHOT_YAML = "configs/mnq_snapshot.yaml"

# Ficheros cuyo hash forma parte del protocolo (orden fijo).
PROTOCOL_FILES = [
    PROTOCOL_YAML,
    REPORT_MD,
    "tests/test_tda17_protocol.py",
    "tests/test_tda17_calentamiento_huecos.py",
    "tests/test_tda17_diseno_v4a_sintetico.py",
    "tests/test_tda17_diseno_v4b_g23b_viabilidad.py",
    "tests/test_tda17_g2_2_generador.py",
    "tests/test_tda17_supertrend.py",
    "src/ohlcv_dataroad/ingest/tda17_protocol_freeze.py",
]

ESTADO_PENDIENTE = "PENDIENTE_AUDITORIA_FINAL"
ESTADO_CONGELADO = "CONGELADO"

# Unicas lineas que cambian al congelar: (fichero, linea en PENDIENTE, linea en CONGELADO), sin fin de linea.
TRANSICIONES = [
    (PROTOCOL_YAML, f"  estado: {ESTADO_PENDIENTE}", f"  estado: {ESTADO_CONGELADO}"),
    (REPORT_MD, f"**Estado:** `{ESTADO_PENDIENTE}`.", f"**Estado:** `{ESTADO_CONGELADO}`."),
]

N_ENTRADAS = 5
N_ARTEFACTOS_PREVIOS = 188

TEXT_SUFFIXES = {".md", ".csv", ".yaml", ".yml", ".json", ".py", ".txt"}
NORMALIZACION = ("Ficheros de texto (.md .csv .yaml .yml .json .py .txt): SHA-256 tras normalizar CRLF a LF; "
                 "binarios: SHA-256 de los bytes")


class ProtocolError(RuntimeError):
    pass


# --------------------------------------------------------------------------- hashes
def sha256_bytes(data: bytes, suffix: str) -> str:
    if suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """SHA-256 independiente del fin de linea en ficheros de texto (el repositorio usa `* text=auto`)."""
    if path.suffix.lower() in TEXT_SUFFIXES:
        return sha256_bytes(path.read_bytes(), path.suffix)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- transiciones de linea
def replace_line(data: bytes, old: str, new: str) -> bytes:
    """Sustituye la UNICA linea completa igual a ``old`` (ignorando CR final) por ``new``, conservando su fin de linea."""
    lines = data.split(b"\n")
    hits = [i for i, ln in enumerate(lines) if ln.rstrip(b"\r") == old.encode("utf-8")]
    if len(hits) != 1:
        raise ProtocolError(f"se esperaba exactamente una linea {old!r}; encontradas {len(hits)}")
    i = hits[0]
    cr = b"\r" if lines[i].endswith(b"\r") else b""
    lines[i] = new.encode("utf-8") + cr
    return b"\n".join(lines)


def count_line(data: bytes, line: str) -> int:
    return sum(ln.rstrip(b"\r") == line.encode("utf-8") for ln in data.split(b"\n"))


def report_state(data: bytes) -> str | None:
    for estado in (ESTADO_PENDIENTE, ESTADO_CONGELADO):
        if count_line(data, f"**Estado:** `{estado}`.") == 1:
            return estado
    return None


# --------------------------------------------------------------------------- verificacion estricta
def load_manifest(root: Path) -> dict:
    return json.loads((root / MANIFEST_JSON).read_text(encoding="utf-8"))


HASH_GROUPS = ("sha256_protocolo", "sha256_entradas", "sha256_artefactos_previos")
FORBIDDEN_PREFIXES = ("data/raw",)


def _forbidden_reason(rel: str, holdout_files: list[str]) -> str | None:
    """Motivo por el que una ruta relativa normalizada esta prohibida, o None (aritmetica de texto)."""
    low = rel.lower()
    if any(low == p or low.startswith(p + "/") for p in FORBIDDEN_PREFIXES):
        return "data/raw"
    if "holdout" in low or "hold-out" in low:
        return "holdout"
    parts = low.split("/")
    for f in holdout_files:
        if f.lower() in parts or f.lower() in low:
            return f"fichero del holdout {f}"
    return None


def validate_manifest_paths(root: Path, m: dict, holdout_files: list[str]) -> list[str]:
    """Valida TODAS las rutas del manifiesto antes de abrir ningun fichero.

    Rechaza rutas que no sean texto relativo POSIX normalizado, absolutas, con '..', con barra invertida
    o unidad, que referencien data/raw o el holdout, que salgan del repositorio, o que RESUELVAN (enlaces
    simbolicos incluidos) a una ubicacion prohibida o externa. Solo consulta metadatos de ruta (resolve);
    nunca lee contenido."""
    errors = []
    root_res = root.resolve()
    for group in HASH_GROUPS:
        entries = m.get(group, {})
        if not isinstance(entries, dict):
            errors.append(f"[{group}] RUTA PROHIBIDA: el grupo no es un diccionario")
            continue
        for rel in entries:
            if not isinstance(rel, str) or not rel:
                errors.append(f"[{group}] RUTA PROHIBIDA: ruta vacia o no textual: {rel!r}")
                continue
            pure = PurePosixPath(rel)
            if ("\\" in rel or ":" in rel or pure.is_absolute() or PureWindowsPath(rel).is_absolute()
                    or ".." in pure.parts or "." in pure.parts or pure.as_posix() != rel):
                errors.append(f"[{group}] RUTA PROHIBIDA (no relativa normalizada): {rel}")
                continue
            reason = _forbidden_reason(rel, holdout_files)
            if reason:
                errors.append(f"[{group}] RUTA PROHIBIDA ({reason}): {rel}")
                continue
            resolved = (root / rel).resolve(strict=False)
            try:
                rel_res = resolved.relative_to(root_res).as_posix()
            except ValueError:
                errors.append(f"[{group}] RUTA PROHIBIDA (resuelve fuera del repositorio): {rel}")
                continue
            reason = _forbidden_reason(rel_res, holdout_files)
            if reason:
                errors.append(f"[{group}] RUTA PROHIBIDA (resuelve a {reason}: {rel_res}): {rel}")
    return errors


def verify_strict(root: Path = ROOT, manifest: dict | None = None) -> list[str]:
    """Verificacion oficial. Devuelve la lista de errores (vacia = conforme). Nunca omite ficheros ausentes.

    ``manifest`` permite verificar un manifiesto candidato EN MEMORIA antes de escribirlo.
    Orden: (1) validar todas las rutas del manifiesto sin abrir ningun fichero de datos; si alguna esta
    prohibida, devolver los errores SIN calcular ningun hash; (2) solo despues, comprobaciones y hashes."""
    m = manifest if manifest is not None else load_manifest(root)
    snap = yaml.safe_load((root / SNAPSHOT_YAML).read_text(encoding="utf-8"))
    path_errors = validate_manifest_paths(root, m, list(snap["holdout"]["holdout_files"]))
    if path_errors:
        return path_errors
    errors: list[str] = []
    cfg = yaml.safe_load((root / PROTOCOL_YAML).read_text(encoding="utf-8"))
    estado = cfg["protocolo"]["estado"]
    rep = report_state((root / REPORT_MD).read_bytes())

    if estado not in (ESTADO_PENDIENTE, ESTADO_CONGELADO):
        errors.append(f"estado desconocido en YAML: {estado}")
    if m.get("estado") != estado:
        errors.append(f"estado del manifiesto ({m.get('estado')}) != YAML ({estado})")
    if rep != estado:
        errors.append(f"estado del informe ({rep}) != YAML ({estado})")
    if m.get("protocolo_id") != cfg["protocolo"]["id"] or m.get("base_git") != cfg["protocolo"]["base_git"]:
        errors.append("protocolo_id o base_git del manifiesto no coinciden con el YAML")
    if m.get("recuentos") != cfg["recuentos"]:
        errors.append("recuentos del manifiesto != YAML")

    if list(m.get("sha256_protocolo", {})) != PROTOCOL_FILES:
        errors.append("la lista de ficheros del protocolo del manifiesto no es la declarada")
    if len(m.get("sha256_entradas", {})) != N_ENTRADAS or m.get("n_entradas") != N_ENTRADAS:
        errors.append(f"el manifiesto debe declarar {N_ENTRADAS} entradas")
    if len(m.get("sha256_artefactos_previos", {})) != N_ARTEFACTOS_PREVIOS or m.get("n_artefactos_previos") != N_ARTEFACTOS_PREVIOS:
        errors.append(f"el manifiesto debe declarar {N_ARTEFACTOS_PREVIOS} artefactos previos")

    for group in HASH_GROUPS:                     # rutas ya validadas: solo ahora se calculan hashes
        for rel, digest in m.get(group, {}).items():
            path = root / rel
            if not path.is_file():
                errors.append(f"[{group}] FALTA: {rel}")
            elif sha256_file(path) != digest:
                errors.append(f"[{group}] HASH DISTINTO: {rel}")

    if estado == ESTADO_PENDIENTE and "congelamiento" in m:
        errors.append("un manifiesto PENDIENTE no puede tener registro de congelamiento")
    if estado == ESTADO_CONGELADO:
        errors.extend(verify_freeze_record(root, m))
    return errors


def verify_freeze_record(root: Path, m: dict) -> list[str]:
    """En CONGELADO, contra las huellas auditadas en PENDIENTE de TODOS los ficheros del protocolo (PROTOCOL_FILES):

    - los dos ficheros con linea de estado: deshacer la transicion reproduce exactamente su hash auditado;
    - los demas (tests y herramienta): su hash actual es exactamente el auditado.
    Recalcular el hash ordinario de sha256_protocolo no basta para ocultar una modificacion. Una reescritura
    completa del manifiesto (incluido este registro) solo se detecta contra una referencia externa:
    el commit Git del congelamiento (``verify_against_git``) y los hashes publicados en la auditoria."""
    rec = m.get("congelamiento")
    if not rec:
        return ["CONGELADO sin registro de congelamiento en el manifiesto"]
    errors = []
    auditado = rec.get("sha256_auditado", {})
    if list(auditado) != PROTOCOL_FILES:
        errors.append("el registro debe contener las huellas auditadas de todos los ficheros del protocolo")
    if [list(t) for t in rec.get("transiciones", [])] != [list(t) for t in TRANSICIONES]:
        errors.append("las transiciones registradas no son las declaradas")
    state_files = {rel: (old, new) for rel, old, new in TRANSICIONES}
    for rel in PROTOCOL_FILES:
        path = root / rel
        if not path.is_file():
            errors.append(f"{rel}: FALTA")
            continue
        data = path.read_bytes()
        if rel in state_files:
            old, new = state_files[rel]
            try:
                data = replace_line(data, new, old)
            except ProtocolError as exc:
                errors.append(f"{rel}: {exc}")
                continue
            if sha256_bytes(data, path.suffix) != auditado.get(rel):
                errors.append(f"{rel}: fuera de la linea de estado difiere de la version auditada")
        else:
            if sha256_bytes(data, path.suffix) != auditado.get(rel):
                errors.append(f"{rel}: difiere de la version auditada")
            if m.get("sha256_protocolo", {}).get(rel) != auditado.get(rel):
                errors.append(f"{rel}: el hash del manifiesto no es el auditado")
    if rec.get("verificacion_estricta_previa") != "CONFORME":
        errors.append("el registro no acredita la verificacion estricta previa")
    return errors


def verify_against_git(root: Path, commit: str) -> list[str]:
    """Referencia externa: todos los ficheros del protocolo y el manifiesto coinciden con los del ``commit``."""
    errors = []
    for rel in PROTOCOL_FILES + [MANIFEST_JSON]:
        shown = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=root, capture_output=True)
        if shown.returncode != 0:
            errors.append(f"[git {commit}] no contiene {rel}")
        elif not (root / rel).is_file():
            errors.append(f"[git {commit}] FALTA en el arbol de trabajo: {rel}")
        elif sha256_bytes(shown.stdout, Path(rel).suffix) != sha256_file(root / rel):
            errors.append(f"[git {commit}] difiere del commit de referencia: {rel}")
    return errors


# --------------------------------------------------------------------------- escritura
def manifest_bytes(m: dict) -> bytes:
    return (json.dumps(m, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _atomic_write_all(root: Path, new: dict[str, bytes]) -> None:
    """Escribe todos los ficheros o ninguno: copia de seguridad en memoria, os.replace y restauracion ante fallo."""
    backup = {rel: (root / rel).read_bytes() for rel in new}
    written: list[str] = []
    try:
        for rel, data in new.items():
            tmp = root / (rel + ".tda17tmp")
            tmp.write_bytes(data)
            os.replace(tmp, root / rel)
            written.append(rel)
    except BaseException:
        for rel in written:
            (root / rel).write_bytes(backup[rel])
        for rel in new:
            (root / (rel + ".tda17tmp")).unlink(missing_ok=True)
        raise


def refresh_protocol_hashes(root: Path = ROOT, *, _hook_tras_escritura=None) -> dict:
    """PENDIENTE: recalcula solo sha256_protocolo; entradas y artefactos se verifican, nunca se recalculan.

    El manifiesto candidato se verifica EN MEMORIA antes de escribir; si algo falla antes, durante o
    despues de la escritura, el manifiesto queda identico byte a byte a su version anterior."""
    original = (root / MANIFEST_JSON).read_bytes()
    m = json.loads(original.decode("utf-8"))
    snap = yaml.safe_load((root / SNAPSHOT_YAML).read_text(encoding="utf-8"))
    path_errors = validate_manifest_paths(root, m, list(snap["holdout"]["holdout_files"]))
    if path_errors:                               # antes de calcular ningun hash
        raise ProtocolError("rutas prohibidas en el manifiesto; no se lee ni se escribe nada:\n" + "\n".join(path_errors))
    cfg = yaml.safe_load((root / PROTOCOL_YAML).read_text(encoding="utf-8"))
    if cfg["protocolo"]["estado"] != ESTADO_PENDIENTE or m.get("estado") != ESTADO_PENDIENTE:
        raise ProtocolError("el manifiesto solo se regenera en PENDIENTE_AUDITORIA_FINAL")
    m["recuentos"] = cfg["recuentos"]
    m["sha256_protocolo"] = {rel: sha256_file(root / rel) for rel in PROTOCOL_FILES}
    m["n_entradas"] = N_ENTRADAS
    m["n_artefactos_previos"] = N_ARTEFACTOS_PREVIOS
    m["normalizacion_hash"] = NORMALIZACION
    errors = verify_strict(root, manifest=m)
    if errors:
        raise ProtocolError("verificacion estricta del manifiesto candidato fallida; no se escribe nada:\n" + "\n".join(errors))
    _atomic_write_all(root, {MANIFEST_JSON: manifest_bytes(m)})
    try:
        if _hook_tras_escritura is not None:
            _hook_tras_escritura()
        post = verify_strict(root)
        if post:
            raise ProtocolError("verificacion estricta posterior fallida:\n" + "\n".join(post))
    except BaseException:
        (root / MANIFEST_JSON).write_bytes(original)
        raise
    return m


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


def check_git(root: Path, base_git: str) -> list[str]:
    errors = []
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    if _git(root, "merge-base", "--is-ancestor", base_git, "HEAD").returncode != 0:
        errors.append(f"HEAD {head} no desciende de base_git {base_git}")
    dirty = _git(root, "status", "--porcelain", "--untracked-files=no").stdout.strip()
    if dirty:
        errors.append("hay ficheros versionados modificados:\n" + dirty)
    return errors


def freeze(root: Path = ROOT, *, confirmar: bool, comprobar_git: bool = True,
           fecha_utc: str | None = None, _hook_tras_escritura=None) -> dict:
    """Transicion atomica PENDIENTE_AUDITORIA_FINAL -> CONGELADO."""
    if not confirmar:
        raise ProtocolError("el congelamiento exige --confirmar")
    cfg = yaml.safe_load((root / PROTOCOL_YAML).read_text(encoding="utf-8"))
    m = load_manifest(root)
    # 1. Precondiciones: los tres ficheros en PENDIENTE y verificacion estricta conforme.
    if cfg["protocolo"]["estado"] != ESTADO_PENDIENTE or m.get("estado") != ESTADO_PENDIENTE \
            or report_state((root / REPORT_MD).read_bytes()) != ESTADO_PENDIENTE:
        raise ProtocolError("YAML, informe y manifiesto deben estar en PENDIENTE_AUDITORIA_FINAL")
    errors = verify_strict(root)
    if comprobar_git:
        errors += check_git(root, cfg["protocolo"]["base_git"])
    if errors:
        raise ProtocolError("precondiciones no cumplidas; no se modifica nada:\n" + "\n".join(errors))

    # 2. Nuevo contenido, calculado y validado en memoria.
    new: dict[str, bytes] = {}
    auditado = {rel: sha256_file(root / rel) for rel in PROTOCOL_FILES}      # huellas auditadas de todos
    for rel, old, nuevo in TRANSICIONES:
        new[rel] = replace_line((root / rel).read_bytes(), old, nuevo)
    if yaml.safe_load(new[PROTOCOL_YAML].decode("utf-8"))["protocolo"]["estado"] != ESTADO_CONGELADO:
        raise ProtocolError("el YAML transformado no queda en CONGELADO")
    if report_state(new[REPORT_MD]) != ESTADO_CONGELADO:
        raise ProtocolError("el informe transformado no queda en CONGELADO")
    head = _git(root, "rev-parse", "HEAD").stdout.strip() if comprobar_git else None
    m2 = dict(m)
    m2["estado"] = ESTADO_CONGELADO
    m2["sha256_protocolo"] = {rel: (sha256_bytes(new[rel], Path(rel).suffix) if rel in new else m["sha256_protocolo"][rel])
                              for rel in PROTOCOL_FILES}
    m2["congelamiento"] = {
        "fecha_utc": fecha_utc or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": head,
        "sha256_auditado": auditado,
        "transiciones": [list(t) for t in TRANSICIONES],
        "verificacion_estricta_previa": "CONFORME",
        "python": platform.python_version(),
    }
    new[MANIFEST_JSON] = manifest_bytes(m2)

    # 3. Escritura de los tres ficheros y verificacion estricta posterior; ante cualquier fallo, restauracion.
    backup = {rel: (root / rel).read_bytes() for rel in new}
    _atomic_write_all(root, new)
    try:
        if _hook_tras_escritura is not None:
            _hook_tras_escritura()
        post = verify_strict(root)
        if post:
            raise ProtocolError("verificacion estricta posterior fallida:\n" + "\n".join(post))
    except BaseException:
        for rel, data in backup.items():
            (root / rel).write_bytes(data)
        raise
    return m2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("accion", choices=["verificar", "manifiesto", "congelar"])
    ap.add_argument("--confirmar", action="store_true")
    ap.add_argument("--referencia-git", metavar="COMMIT",
                    help="verificar: compara ademas los ficheros del protocolo y el manifiesto con los de COMMIT")
    a = ap.parse_args(argv)
    try:
        if a.accion == "verificar":
            errors = verify_strict(ROOT)
            if a.referencia_git:
                errors += verify_against_git(ROOT, a.referencia_git)
            if errors:
                print("VERIFICACION ESTRICTA: NO CONFORME")
                print("\n".join(errors))
                return 1
            m = load_manifest(ROOT)
            print(f"VERIFICACION ESTRICTA: CONFORME (estado {m['estado']}; {len(m['sha256_protocolo'])} ficheros del protocolo, "
                  f"{len(m['sha256_entradas'])} entradas, {len(m['sha256_artefactos_previos'])} artefactos previos)")
        elif a.accion == "manifiesto":
            refresh_protocol_hashes(ROOT)
            print("sha256_protocolo regenerado; verificacion estricta CONFORME")
        else:
            m = freeze(ROOT, confirmar=a.confirmar)
            print(f"CONGELADO {m['congelamiento']['fecha_utc']} (HEAD {m['congelamiento']['git_head']}); verificacion estricta CONFORME")
    except ProtocolError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
