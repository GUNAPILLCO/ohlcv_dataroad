"""TDA-17 -- fase 2: UNICA via de publicacion de las salidas oficiales de TH35 (E02, decisiones 17-19).

Ruta unica ``official_publish_th35(publicacion, root)``: solo recibe la publicacion P2. Deriva la configuracion
oficial con M2 (``tda17_analysis.official_config``), ejecuta ``official_th35`` EN EL MISMO PROCESO (entradas leidas y
verificadas contra el manifiesto, TH37 obligatoria) y escribe exactamente las tablas del esquema minimo de E02
(``salidas.esquema_minimo``) construidas a partir del resultado que acaba de calcular, con el registro de procedencia
``TDA17_procedencia_oficial.json``. No existe ninguna funcion que reciba una tabla, un resultado o una configuracion
para escribirlos: ``official_tables`` y ``diagnostic_table`` son funciones puras (no escriben) y la escritura solo
ocurre dentro de ``official_publish_th35``.

- Sin bloqueo: auditoria causal, estabilidad por celda, R2 de reconstruccion, Spearman y clasificacion final.
- Con bloqueo (BLOQUEADO_HASTA_EXPLICAR): SOLO la auditoria causal y ``TDA17_diagnostico_bloqueo.csv`` (seccion 4.3
  de la propuesta aprobada); nunca la clasificacion, los subniveles ni la estabilidad por celda.
- Las demas tablas de ``salidas.tablas`` siguen BLOQUEADAS (``PENDIENTES['SALIDAS_OTRAS_TABLAS']``).
- Ninguna salida oficial previa se sobrescribe (``SALIDA_EXISTENTE``).

Publicacion transaccional (correccion tras la auditoria v5):
- Las tablas y el registro se calculan y serializan ANTES de tocar el disco.
- Un cerrojo del sistema operativo sobre ``.TDA17_publicacion.lock`` excluye una segunda publicacion simultanea, de
  este u otro proceso (``SALIDA_CONCURRENTE``). El sistema lo libera si el proceso muere: nunca queda huerfano.
- Todo se prepara en ``.TDA17_publicacion_en_curso`` (fsync de cada fichero y un diario con sus huellas) y se
  promueve SIN SOBRESCRIBIR con un enlace duro (``_promote`` = ``os.link``, en Windows y en POSIX: atomico y falla si
  el destino existe): primero las tablas y AL FINAL el registro de procedencia, que es el punto de confirmacion. Sin
  registro no hay publicacion valida (``SALIDA_INCOMPLETA`` / ``SALIDA_REGISTRO``).
- Propiedad (correccion tras la auditoria v7): el fichero preparado NO se borra al promover; sigue enlazado en la
  preparacion hasta la confirmacion y ancla la identidad del fichero (dispositivo e inodo / File ID, que no puede
  reutilizarse mientras exista ese enlace). Un fichero oficial es de esta publicacion solo si tiene la MISMA
  identidad que su preparado Y los bytes de su diario; un fichero ajeno, aunque tenga bytes identicos, tiene otra
  identidad. La identidad se lee SIN seguir enlaces (``os.lstat``; correccion tras la auditoria v8): un enlace
  simbolico (o cualquier punto de reanalisis de Windows) nunca es propio, aunque apunte al preparado.
- Un fallo en el proceso deshace lo que es demostrablemente suyo (aunque el fallo ocurra despues de enlazar). Si el
  proceso muere, el siguiente intento, bajo el cerrojo, retira lo demostrablemente suyo; si alguna salida oficial
  presente no puede demostrarse suya, NO borra nada, la conserva y bloquea la recuperacion (``SALIDA_RECUPERACION``).
- Si otro proceso (sin el cerrojo) crea un destino despues de ``SALIDA_EXISTENTE``, la promocion de ese fichero
  falla, el fichero ajeno queda intacto, se deshace lo propio y se rechaza con ``SALIDA_EXISTENTE``.

Verificacion posterior: ``check_official_publication`` (registro, P2, configuracion efectiva, entradas del
manifiesto, conjunto de tablas segun el estado, esquema y huella de los bytes exactos de cada tabla; registro y tablas
como ficheros regulares, sin enlaces simbolicos ni puntos de reanalisis, y enlaces rotos como salidas presentes) y
``reproduce_official_publication`` (recalcula ``official_th35`` y exige tablas identicas byte a byte, sin normalizar
finales de linea).

Cada control rechaza con ``ProtocolNotAuthorizedError`` y un identificador ``[ID]``. No lee data/raw ni el holdout.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import stat
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ohlcv_dataroad.ingest import tda17_analysis as an
from ohlcv_dataroad.ingest import tda17_enmiendas_e02 as m2

ProtocolNotAuthorizedError = m2.ProtocolNotAuthorizedError
CAUSAL = "TDA17_auditoria_causalidad.csv"
STABILITY = "TDA17_estabilidad_celdas.csv"
R2 = "TDA17_r2_reconstruccion.csv"
SPEARMAN = "TDA17_correlacion_spearman.csv"
CLASSIFICATION = "TDA17_clasificacion_final.csv"
DIAGNOSTIC = "TDA17_diagnostico_bloqueo.csv"
PROVENANCE = "TDA17_procedencia_oficial.json"
UNBLOCKED = [CAUSAL, STABILITY, R2, SPEARMAN, CLASSIFICATION]
BLOCKED = [CAUSAL, DIAGNOSTIC]
STATES = ("SIN_BLOQUEO", "BLOQUEADO_HASTA_EXPLICAR")
PROVENANCE_FIELDS = ["publicacion_E02", "remoto_E02", "sha256_efectiva", "entradas", "huella_marco", "filas", "estado",
                     "bloqueos", "tablas"]
DIAGNOSTIC_SECTIONS = ["AMBITO", "ESPECTRO", "DIRECCION_NULA", "R2", "EXACTITUD", "RESIDUO_IDENTIDAD", "RELACION_N"]
OFFICIAL = UNBLOCKED + [DIAGNOSTIC, PROVENANCE]
LOCK = ".TDA17_publicacion.lock"
STAGING = ".TDA17_publicacion_en_curso"
JOURNAL = "diario.json"


def _reject(cid: str, msg: str):
    raise ProtocolNotAuthorizedError(f"[{cid}] {msg}")


# --------------------------------------------------------------------------- esquema (E02, salidas)
def schema(cfg: dict) -> dict:
    """Esquema de E02: {tabla: columnas} de las cinco tablas y del diagnostico, directorio y conjuntos por estado.
    Cualquier diferencia con lo implementado es un rechazo (el codigo no publica otro esquema en silencio)."""
    em = an.e02_value(cfg, "salidas.esquema_minimo")
    dg = an.e02_value(cfg, "salidas.diagnostico_bloqueo")
    tabs = em.get("tablas") if isinstance(em.get("tablas"), dict) else {}
    ok = (em.get("ruta_publicacion") == "official_publish_th35" and em.get("recibe_tablas") is False
          and em.get("calculo") == "OFFICIAL_TH35_EN_EL_MISMO_PROCESO" and em.get("sin_bloqueo") == UNBLOCKED
          and em.get("con_bloqueo") == BLOCKED and list(tabs) == UNBLOCKED
          and (em.get("procedencia") or {}).get("fichero") == PROVENANCE
          and (em.get("procedencia") or {}).get("campos") == PROVENANCE_FIELDS
          and em.get("formato") == {"separador": ",", "fin_de_linea": "LF", "codificacion": "utf-8", "indice": False}
          and tabs.get(CAUSAL, {}).get("columnas") == "TABLA_CAMPOS_DE_ANALISIS_AUDITORIA_CAUSALIDAD"
          and dg.get("fichero") == DIAGNOSTIC and dg.get("solo_con_estado") == "BLOQUEADO_HASTA_EXPLICAR"
          and dg.get("secciones") == DIAGNOSTIC_SECTIONS and dg.get("marca") == "DIAGNOSTICO_BLOQUEADO"
          and dg.get("levanta_el_bloqueo") is False
          and all(n in cfg["salidas"]["tablas"] for n in UNBLOCKED))
    if not ok:
        _reject("SALIDA_ESQUEMA", "el esquema de salidas de E02 no es el implementado")
    cols = {n: list(tabs[n]["columnas"]) for n in UNBLOCKED if n != CAUSAL}
    cols[CAUSAL] = list(an.e02_value(cfg, "analisis.auditoria_causalidad")["tabla"]["campos"])
    cols[DIAGNOSTIC] = list(dg["columnas"])
    return {"directorio": em["directorio"], "columnas": cols}


def _exact(frame: pd.DataFrame, columns: list[str], nombre: str) -> pd.DataFrame:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        _reject("SALIDA_COLUMNAS", f"{nombre}: el resultado oficial no contiene las columnas {missing}")
    return frame[columns].reset_index(drop=True)


# --------------------------------------------------------------------------- tablas (funciones puras)
def _r2_table(result: dict) -> pd.DataFrame:
    parts = []
    for cell, a in result["analisis_celdas"].items():
        if a.get("estado_matriz") != "DEFINIDA":
            continue
        lab = a["exactitud"].set_index("columna")["etiqueta"]
        t = a["r2_todas"].copy()
        t.insert(0, "celda", cell)
        t["etiqueta_exactitud"] = t["columna"].map(lab)
        parts.append(t)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _spearman_table(result: dict) -> pd.DataFrame:
    parts = [t.assign(celda=cell) for cell, t in result["spearman"].items()]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _classification_table(result: dict) -> pd.DataFrame:
    sub = result["subniveles"].rename(columns={"motivo": "motivo_subnivel"})
    return result["clasificacion"].merge(sub, on="columna", how="left", validate="one_to_one")


def _rows(base: dict, seccion: str, elemento, columna, metodo, valor, etiqueta) -> dict:
    v = np.nan if valor is None else float(valor)
    return {**base, "seccion": seccion, "elemento": "" if elemento is None else str(elemento),
            "columna": columna or "", "metodo": metodo or "", "valor": v, "etiqueta": etiqueta or ""}


def diagnostic_table(result: dict, cfg: dict) -> pd.DataFrame:
    """``TDA17_diagnostico_bloqueo.csv`` (seccion 4.3): formato largo, una fila por valor, con la marca
    DIAGNOSTICO_BLOQUEADO. AMBITO recoge cada bloqueo y, en las celdas con diagnostico, los recuentos observados y
    registrados; las demas secciones vienen de ``diagnostico_bloqueo`` (``blocking_dossier``) de cada celda."""
    marca = an.e02_value(cfg, "salidas.diagnostico_bloqueo")["marca"]
    rows = []
    for b in result["bloqueos"]:
        cell, _, text = b.partition(": ")
        a = result["analisis_celdas"].get(cell, {})
        rows.append(_rows({"celda": cell, "alcance": a.get("alcance_bloqueo") or "", "matriz": ""},
                          "AMBITO", "bloqueo", None, None, None, text))
    for cell, a in result["analisis_celdas"].items():
        dz = a.get("diagnostico_bloqueo")
        if dz is None:
            continue
        amb = dz["ambito_afectado"]
        base = {"celda": cell, "alcance": amb["alcance"], "matriz": ""}
        for what, (obs, exp) in amb["recuentos"].items():
            m = {**base, "matriz": what}
            rows.append(_rows(m, "AMBITO", "nulos_observados", None, "AUTOVALOR_A1a", obs, None))
            rows.append(_rows(m, "AMBITO", "nulos_esperados", None, "REGISTRO", exp, None))
        for what, sp in dz["espectros"].items():
            m = {**base, "matriz": what}
            rows += [_rows(m, "ESPECTRO", i, None, "AUTOVALOR_A1a", v, None) for i, v in enumerate(sp["autovalores"])]
            rows += [_rows(m, "ESPECTRO", i, None, "SINGULAR", v, None) for i, v in enumerate(sp["singulares"])]
        for what, d in dz["direcciones_nulas"].items():
            m = {**base, "matriz": what}
            for k, rec in enumerate(d.to_dict("records")):
                rows.append(_rows(m, "DIRECCION_NULA", k, None, "AUTOVALOR_A1a", rec.pop("autovalor"), None))
                rows += [_rows(m, "DIRECCION_NULA", k, c, "CARGA", v, None) for c, v in rec.items()]
        for key, t in dz["r2_diagnostico"].items():
            for rec in t.to_dict("records"):
                for metodo, campo in (("base", "r2"), ("espectral", "r2_espectral"), ("minima_norma", "r2_minima_norma")):
                    rows.append(_rows(base, "R2", None, rec["columna"], f"{key}_{metodo}", rec[campo],
                                      rec["etiqueta_publicacion"]))
        for rec in dz["exactitud_diagnostico"].to_dict("records"):
            rows.append(_rows(base, "EXACTITUD", rec["identidades"], rec["columna"], "residuo_max", rec["residuo_max"],
                              rec["etiqueta"]))
        for lid, rec in dz["residuos_identidades"].items():
            rows.append(_rows(base, "RESIDUO_IDENTIDAD", lid, None, "residuo_relativo", rec["residuo_relativo"], None))
        for r in dz["relaciones_N"]:
            if r["tipo"] in ("LA-P", "NL-P"):
                for what, d in r["cargas"].items():
                    m = {**base, "matriz": what}
                    if d.empty:
                        rows += [_rows(m, "RELACION_N", r["id"], c, r["tipo"], None, "SIN_DIRECCIONES_NULAS")
                                 for c in r["columnas_P"]]
                    for k, rec in enumerate(d.to_dict("records")):
                        rows += [_rows(m, "RELACION_N", r["id"], c, r["tipo"], v, f"DIRECCION_{k}")
                                 for c, v in rec.items() if c != "autovalor"]
            elif r["tipo"] in ("LA-F", "NL-F"):
                rows += [_rows(base, "RELACION_N", r["id"], c, r["tipo"], None, "SIN_DIRECCION_EN_P")
                         for c in r["columnas_P"]]
            elif r["tipo"] == "DES":
                rows.append(_rows(base, "RELACION_N", r["id"], None, r["tipo"], r["violaciones"], "VIOLACIONES"))
            else:
                rows.append(_rows(base, "RELACION_N", r["id"], None, r["tipo"], None, "NO_IDENTIDAD"))
            if r["estado"] == "PENDIENTE_EXPEDIENTE":
                rows.append(_rows(base, "RELACION_N", r["id"], None, r["tipo"], None, "PENDIENTE_EXPEDIENTE"))
    out = pd.DataFrame(rows, columns=["celda", "alcance", "matriz", "seccion", "elemento", "columna", "metodo", "valor",
                                      "etiqueta"])
    out["marca"] = marca
    return out


def official_tables(result: dict, cfg: dict) -> dict[str, pd.DataFrame]:
    """Tablas que corresponden al ``result`` de ``official_th35`` segun su estado, con las columnas EXACTAS del
    esquema y en su orden. Funcion pura: no escribe nada."""
    sch = schema(cfg)["columnas"]
    estado = result.get("estado")
    if estado not in STATES:
        _reject("SALIDA_ESTADO", f"estado del resultado no publicable: {estado!r}")
    if (estado == "SIN_BLOQUEO") != (not result["bloqueos"]):
        _reject("SALIDA_ESTADO", "el estado del resultado no corresponde a sus bloqueos")
    out = {CAUSAL: _exact(result["auditoria_causalidad"], sch[CAUSAL], CAUSAL)}
    if estado == "BLOQUEADO_HASTA_EXPLICAR":
        if "clasificacion" in result or "subniveles" in result:
            _reject("SALIDA_ESTADO", "un resultado bloqueado no puede contener clasificacion ni subniveles")
        out[DIAGNOSTIC] = _exact(diagnostic_table(result, cfg), sch[DIAGNOSTIC], DIAGNOSTIC)
        return out
    out[STABILITY] = _exact(result["per_cell"], sch[STABILITY], STABILITY)
    out[R2] = _exact(_r2_table(result), sch[R2], R2)
    out[SPEARMAN] = _exact(_spearman_table(result), sch[SPEARMAN], SPEARMAN)
    out[CLASSIFICATION] = _exact(_classification_table(result), sch[CLASSIFICATION], CLASSIFICATION)
    return out


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serializacion determinista del esquema (separador ',', LF, UTF-8, sin indice)."""
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _provenance(result: dict, cfg: dict, publicacion: str, hashes: dict[str, str]) -> dict:
    prov, tr = result["procedencia"], cfg["trazabilidad_protocolo"]
    return {"publicacion_E02": publicacion, "remoto_E02": tr["remoto_E02"], "sha256_efectiva": tr["sha256_efectiva"],
            "entradas": prov["entradas"], "huella_marco": prov["huella_marco"], "filas": prov["filas"],
            "estado": result["estado"], "bloqueos": list(result["bloqueos"]), "tablas": hashes}


def _official_run(publicacion: str, root: Path) -> tuple[dict, dict, dict[str, bytes]]:
    """Configuracion oficial + ``official_th35`` en este proceso + tablas serializadas, con la comprobacion de que el
    resultado procede de esa misma configuracion y publicacion."""
    cfg = an.official_config(publicacion, root)
    result = an.official_th35(publicacion, root)
    prov = result.get("procedencia") or {}
    if prov.get("publicacion_E02") != publicacion or \
            prov.get("sha256_efectiva") != cfg["trazabilidad_protocolo"]["sha256_efectiva"]:
        _reject("SALIDA_PROCEDENCIA", "el resultado de official_th35 no procede de la configuracion oficial de esta publicacion")
    blobs = {n: csv_bytes(t) for n, t in official_tables(result, cfg).items()}
    return cfg, result, blobs


def _sha(data: bytes) -> str:
    """Huella de los bytes EXACTOS (sin normalizar finales de linea)."""
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- publicacion transaccional
@contextlib.contextmanager
def _exclusive(out_dir: Path):
    """Cerrojo del sistema operativo sobre ``LOCK`` (no bloqueante): excluye otra publicacion simultanea de este u
    otro proceso. El sistema lo libera al cerrar o si el proceso muere."""
    fd = os.open(out_dir / LOCK, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            _reject("SALIDA_CONCURRENTE", "otra publicacion oficial esta en curso")
        try:
            yield
        finally:
            if sys.platform == "win32":
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    finally:
        os.close(fd)


def _write_durable(path: Path, data: bytes) -> None:
    """Fichero nuevo (nunca sobrescribe) con los bytes exactos y fsync."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)


def _promote(src: Path, dst: Path) -> None:
    """Enlaza ``src`` en ``dst`` de forma atomica y SIN SOBRESCRIBIR: si ``dst`` existe (aunque haya aparecido despues
    de la comprobacion inicial) lanza ``FileExistsError`` y no toca ninguno de los dos. ``src`` NO se borra: ancla la
    identidad del fichero publicado hasta la confirmacion (``_owned``)."""
    os.link(src, dst)                                      # CreateHardLinkW / link(2): falla si existe


def _plain(st: os.stat_result) -> bool:
    """Fichero regular de verdad: ni enlace simbolico ni otro punto de reanalisis (Windows)."""
    return stat.S_ISREG(st.st_mode) and not (getattr(st, "st_file_attributes", 0) or 0) & \
        stat.FILE_ATTRIBUTE_REPARSE_POINT


def _owned(staged: Path, path: Path, digest: str | None) -> bool:
    """``path`` es de la publicacion solo si es el MISMO fichero que su preparado (dispositivo e inodo / File ID; el
    preparado sigue enlazado, asi que la identidad no puede reutilizarse) y conserva los bytes del diario. Un fichero
    ajeno con bytes identicos tiene otra identidad: nunca es propio. La identidad se lee SIN seguir enlaces
    (``os.lstat``): un enlace simbolico u otro sustituto que apunte al preparado no es el preparado y nunca es propio;
    la identidad se vuelve a comprobar tras leer los bytes."""
    if digest is None:
        return False
    try:
        a, b = os.lstat(staged), os.lstat(path)
        if not (_plain(a) and _plain(b)) or a.st_ino == 0 or (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
            return False
        data = path.read_bytes()
        c = os.lstat(path)
    except OSError:
        return False
    return _plain(c) and (c.st_dev, c.st_ino) == (a.st_dev, a.st_ino) and _sha(data) == digest


def _regular_bytes(path: Path) -> bytes | None:
    """Bytes de ``path`` solo si es un fichero regular de verdad (``_plain``, leido SIN seguir enlaces) y su identidad
    no cambia durante la lectura; si falta, es un enlace (tambien roto) u otro punto de reanalisis, None."""
    try:
        a = os.lstat(path)
        if not _plain(a):
            return None
        data = path.read_bytes()
        b = os.lstat(path)
    except OSError:
        return None
    return data if _plain(b) and (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino) else None


def _recover(out_dir: Path) -> None:
    """Restos de una publicacion cuyo proceso murio sin deshacerla (solo se llama bajo el cerrojo).
    - Con registro de procedencia presente, la publicacion se confirmo (o hay otra): solo se retira la preparacion y
      ninguna salida oficial se toca (despues, SALIDA_EXISTENTE).
    - Sin registro: se retiran las salidas oficiales demostrablemente suyas (``_owned`` con su diario). Si alguna
      salida oficial presente NO puede demostrarse suya, no se borra nada, se conserva la preparacion como evidencia
      y la recuperacion queda bloqueada (SALIDA_RECUPERACION)."""
    staging = out_dir / STAGING
    if not staging.exists():
        return
    if (out_dir / PROVENANCE).exists():
        shutil.rmtree(staging)
        return
    try:
        journal = json.loads((staging / JOURNAL).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        journal = {}                                       # el diario se escribe antes de promover: nada promovido
    if not isinstance(journal, dict):
        journal = {}
    present = [n for n in OFFICIAL if os.path.lexists(out_dir / n)]       # tambien enlaces rotos
    foreign = [n for n in present if not _owned(staging / n, out_dir / n, journal.get(n))]
    if foreign:
        _reject("SALIDA_RECUPERACION", f"no puede demostrarse que {foreign} pertenezcan a la publicacion interrumpida: "
                                       f"se conservan y la recuperacion queda bloqueada")
    for n in present:
        (out_dir / n).unlink()
    shutil.rmtree(staging)


def _rollback(out_dir: Path, hashes: dict[str, str]) -> None:
    """Fallo dentro de este proceso antes de confirmar: retira TODO lo demostrablemente suyo (``_owned``), aunque el
    fallo haya ocurrido despues de enlazar y antes de anotarlo; nunca un fichero ajeno. La preparacion se retira solo
    si no queda nada suyo publicado (si no, la conserva para ``_recover``)."""
    staging, left = out_dir / STAGING, False
    for n in reversed(list(hashes)):
        path = out_dir / n
        if _owned(staging / n, path, hashes[n]):
            try:
                path.unlink()
            except OSError:
                left = True
    if not left:
        shutil.rmtree(staging, ignore_errors=True)


def _stage_and_promote(out_dir: Path, files: dict[str, bytes]) -> None:
    """Prepara TODO en ``STAGING`` (cada fichero con fsync y despues el diario de huellas) y promueve: primero las
    tablas y al final el registro de procedencia, que confirma la publicacion."""
    assert list(files)[-1] == PROVENANCE
    staging = out_dir / STAGING
    staging.mkdir()
    hashes = {n: _sha(b) for n, b in files.items()}
    try:
        for n, b in files.items():
            _write_durable(staging / n, b)
        _write_durable(staging / JOURNAL, (json.dumps(hashes, indent=2) + "\n").encode("utf-8"))
        for n in files:
            try:
                _promote(staging / n, out_dir / n)
            except FileExistsError:
                _reject("SALIDA_EXISTENTE", f"{n} aparecio durante la publicacion (no se sobrescribe)")
    except BaseException:
        _rollback(out_dir, hashes)
        raise
    shutil.rmtree(staging, ignore_errors=True)             # ya confirmada; un resto lo retira _recover


# --------------------------------------------------------------------------- ruta unica de publicacion
def official_publish_th35(publicacion: str, root: Path | None = None) -> dict:
    """PRODUCCION: publica las salidas oficiales de TH35 de la ejecucion oficial vinculada a P2. Solo recibe P2: la
    configuracion, las entradas y el calculo se derivan aqui (``official_th35``, en el mismo proceso). Publica de forma
    transaccional, bajo cerrojo, las tablas que tocan segun el estado y el registro de procedencia; devuelve el
    registro."""
    root = Path(m2.ROOT if root is None else root)
    cfg, result, blobs = _official_run(publicacion, root)
    record = _provenance(result, cfg, publicacion, {n: _sha(b) for n, b in blobs.items()})
    files = {**blobs, PROVENANCE: (json.dumps(record, indent=2, ensure_ascii=False) + "\n").encode("utf-8")}
    out_dir = root / schema(cfg)["directorio"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with _exclusive(out_dir):
        _recover(out_dir)
        existing = [n for n in OFFICIAL if os.path.lexists(out_dir / n)]
        if existing:
            _reject("SALIDA_EXISTENTE", f"ya existen salidas oficiales (no se sobrescriben): {existing}")
        _stage_and_promote(out_dir, files)
    return record


# --------------------------------------------------------------------------- verificacion posterior
def check_official_publication(publicacion: str, root: Path | None = None) -> dict:
    """Comprueba una publicacion ya escrita sin recalcular: registro bien formado que cita ``publicacion``, la
    configuracion efectiva oficial derivada de nuevo y las entradas del manifiesto; tablas = las del estado
    registrado (ninguna otra salida oficial presente), cabecera = esquema y huella de los bytes exactos = la
    registrada. Una publicacion interrumpida (preparacion presente sin registro confirmado) no es valida. El registro
    y las tablas deben ser ficheros regulares (sin enlaces simbolicos ni puntos de reanalisis; auditoria v9) y la
    presencia se mira sin seguir enlaces: un enlace roto es una salida presente."""
    root = Path(m2.ROOT if root is None else root)
    cfg = an.official_config(publicacion, root)
    sch = schema(cfg)
    out_dir = root / sch["directorio"]
    if os.path.lexists(out_dir / STAGING) and not os.path.lexists(out_dir / PROVENANCE):
        _reject("SALIDA_INCOMPLETA", "publicacion oficial interrumpida o en curso (sin registro de procedencia confirmado)")
    raw = _regular_bytes(out_dir / PROVENANCE)
    if raw is None and os.path.lexists(out_dir / PROVENANCE):
        _reject("SALIDA_REGISTRO", f"el registro {PROVENANCE} no es un fichero regular (enlace o punto de reanalisis)")
    try:
        record = json.loads(raw.decode("utf-8"))
    except (AttributeError, ValueError, UnicodeDecodeError):
        _reject("SALIDA_REGISTRO", f"falta o no es JSON valido el registro {PROVENANCE}")
    if not isinstance(record, dict) or list(record) != PROVENANCE_FIELDS or not isinstance(record.get("tablas"), dict):
        _reject("SALIDA_REGISTRO", f"el registro {PROVENANCE} no tiene exactamente los campos {PROVENANCE_FIELDS}")
    if record["publicacion_E02"] != publicacion:
        _reject("SALIDA_P2", f"el registro cita la publicacion {record['publicacion_E02']}, no {publicacion}")
    tr = cfg["trazabilidad_protocolo"]
    if record["sha256_efectiva"] != tr["sha256_efectiva"] or record["remoto_E02"] != tr["remoto_E02"]:
        _reject("SALIDA_EFECTIVA", "el registro no corresponde a la configuracion oficial vigente de esta publicacion")
    from ohlcv_dataroad.ingest import tda17_entradas as ent
    if record["entradas"] != ent.expected_input_hashes(root):
        _reject("SALIDA_ENTRADAS", "el registro no cita las entradas del manifiesto")
    expected = {"SIN_BLOQUEO": UNBLOCKED, "BLOQUEADO_HASTA_EXPLICAR": BLOCKED}.get(record["estado"])
    if expected is None or sorted(record["tablas"]) != sorted(expected) or \
            (record["estado"] == "SIN_BLOQUEO") != (not record["bloqueos"]):
        _reject("SALIDA_CONJUNTO", f"tablas registradas no corresponden al estado {record['estado']!r}")
    extra = [n for n in UNBLOCKED + [DIAGNOSTIC] if n not in expected and os.path.lexists(out_dir / n)]
    if extra:
        _reject("SALIDA_CONJUNTO", f"salidas oficiales no registradas para el estado {record['estado']}: {extra}")
    for n in expected:
        path = out_dir / n
        data = _regular_bytes(path)
        if data is None and os.path.lexists(path):
            _reject("SALIDA_HUELLA", f"{n} no es un fichero regular (enlace o punto de reanalisis)")
        if data is None or _sha(data) != record["tablas"][n]:
            _reject("SALIDA_HUELLA", f"{n} falta o no tiene la huella registrada")
        header = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").split("\n", 1)[0].split(",")
        if header != sch["columnas"][n]:
            _reject("SALIDA_ESQUEMA", f"{n} no tiene las columnas del esquema")
    return record


def reproduce_official_publication(publicacion: str, root: Path | None = None) -> dict:
    """Comprobacion completa: ``check_official_publication`` y, ademas, recalcula ``official_th35`` en este proceso y
    exige que cada tabla publicada sea identica byte a byte a la recalculada, sin normalizar finales de linea
    (vinculacion con el calculo verificado)."""
    root = Path(m2.ROOT if root is None else root)
    record = check_official_publication(publicacion, root)
    cfg, result, blobs = _official_run(publicacion, root)
    out_dir = root / schema(cfg)["directorio"]
    if sorted(blobs) != sorted(record["tablas"]) or result["estado"] != record["estado"]:
        _reject("SALIDA_NO_REPRODUCIBLE", "el recalculo no produce el mismo conjunto de tablas ni el mismo estado")
    differ = [n for n, b in blobs.items() if _regular_bytes(out_dir / n) != b]
    if differ:
        _reject("SALIDA_NO_REPRODUCIBLE", f"tablas publicadas distintas del recalculo oficial: {differ}")
    return record


def other_official_table(nombre: str):
    """Tablas de ``salidas.tablas`` fuera del esquema minimo: sin esquema ni vinculacion, siguen bloqueadas."""
    if nombre in UNBLOCKED + [DIAGNOSTIC]:
        _reject("SALIDA_RUTA_UNICA", f"{nombre} solo se publica con official_publish_th35")
    an.pending("SALIDAS_OTRAS_TABLAS")
