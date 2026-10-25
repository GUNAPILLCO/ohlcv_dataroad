"""TDA-17 -- E03 (``ohlcv_dataroad.tda17_e03.motor``): motor de ejecucion PARALELO de TH37 -> TH35 (solo
implementacion; DECISION ESTADISTICA: NINGUNA).

Objetivo: producir EXACTAMENTE el mismo resultado que ``tda17_analysis.th35_pipeline`` (motor E02, secuencial)
repartiendo entre procesos las tareas independientes. Criterio de aceptacion: tablas oficiales identicas BYTE A BYTE
(``tda17_salidas.official_tables`` + ``csv_bytes``). No cambia ningun fichero auditado de E02, del protocolo ni de E01:
este modulo LLAMA a las funciones auditadas y solo reorganiza su ejecucion.

Descomposicion (orden canonico de agregacion = el de E02; el orden de finalizacion de las tareas nunca importa):
- TH37 componente 1: ``ca.audit_future(df, cfg, [subplan], full)`` por (contrato, bloque de cortes consecutivos) --
  la funcion admite un subconjunto del plan; las filas NO_APLICABLE del contrato van solo en su primer bloque.
- TH37 componente 2: ``ca.audit_parameter_origin`` sin cambios (una tarea).
- TH37 componente 3: cuerpo LITERAL del bucle de ``ca.audit_roll`` para un contrato (``_t_th37_roll``), con las
  comprobaciones previas en el proceso principal. Unica reimplementacion de TH37 (probada contra ``audit_roll``).
- Barrera TH37: ``ca.audit_table`` + ``ca.verify_causal_audit`` sin cambios, en el proceso principal, ANTES de TH35.
- TH35: catalogo, panel, casos completos y celdas en el proceso principal (sin cambios). Por celda con flujo,
  ``an.analyze_cell`` + momentos diarios + ``an.bootstrap_spearman_fixed`` en un worker; las replicas de R2_TODAS
  por bloques (cuerpo LITERAL del bucle de ``an.bootstrap_r2``) y la agregacion LITERAL en el principal.
- Validacion de Spearman: cuerpo LITERAL de ``an._validation_cell`` por replica; rondas y agregacion LITERALES de
  ``an.spearman_validation`` en el principal. Las primeras replicas de la ronda ampliada son las de la ronda de 30
  (prefijo del mismo flujo): se reutilizan solo si los sorteos son identicos.
- Sorteos: SIEMPRE en el proceso principal con ``an.cell_draws`` (flujos PCG64, semillas y orden de E02); los workers
  solo reciben indices de jornadas.

Hilos BLAS: los resultados de E02 DEPENDEN del numero de hilos de OpenBLAS (la particion del trabajo de BLAS/LAPACK
cambia los ultimos bits; comprobado: con 1 hilo por worker difieren los autovalores de GLOBAL). Por eso cada worker
arranca con los hilos EFECTIVOS del proceso principal (no con las variables de entorno que pudiera haber en ese
momento) y se verifica en cada tarea que son los mismos (``E03_BLAS``); forzar otro valor (``Recursos.hilos_blas``)
solo se usa en el benchmark y rompe la equivalencia. ``hilos_blas_fijados(n)`` fija los hilos EFECTIVOS del proceso
principal con la API de OpenBLAS (``scipy_openblas_set_num_threads64_``), los verifica y los restaura al salir: la
ruta oficial de E03 lo usa con 1 hilo.

Datos de TH37 COMPARTIDOS de solo lectura: el marco de entrada y el catalogo completo (``series``, ``marcas``) se
escriben una vez por bloques de pandas (``compartir_marcos``): cada bloque numerico (float/int/uint/bool) en un
``.npy`` que los workers proyectan en memoria (``mmap``, modo ``"r"``: paginas del fichero compartidas entre procesos,
no memoria privada) y el resto (cadenas, objetos, fechas) dentro del manifiesto. El worker reconstruye cada marco con
la MISMA estructura de bloques, tipos, columnas e indice (``pandas.api.internals.create_dataframe_from_blocks``) sin
copiar los bloques proyectados (arrays de solo lectura: una escritura falla, nunca se ignora). Cada fichero se
verifica por SHA-256 al abrirlo y el principal comprueba antes de lanzar ninguna tarea que la reconstruccion es
identica bit a bit al original (``E03_COMPARTIDO``).

Recursos: procesos ``spawn`` (Windows), numero de workers DINAMICO segun la memoria disponible (reserva fija) y el pico
de memoria PRIVADA medido por tipo de tarea (las paginas compartidas se cuentan una sola vez por etapa), y limitado por
el rendimiento medido (``_Motor``). Progreso verificable (``estado.json``, ``progreso.jsonl``), tiempos por etapa y
estimacion del tiempo restante SOLO con tareas medidas. Puntos de control por tarea con SHA-256 e identidad de la
ejecucion; una ejecucion incompleta nunca produce resultado (``COMPLETA.json`` solo al terminar). La REANUDACION solo
existe fuera de la ruta oficial (``Ejecucion(oficial=True)`` la rechaza: el SHA-256 del registro de progreso protege
la integridad frente a errores, no la autenticidad frente a quien puede escribir el directorio).
"""
from __future__ import annotations

import collections
import contextlib
import copy
import ctypes
import gc
import hashlib
import json
import math
import multiprocessing as mp
import multiprocessing.connection as mpc
import os
import pickle
import platform
import random
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ohlcv_dataroad.ingest import tda17_analysis as an
from ohlcv_dataroad.ingest import tda17_causalidad as ca
from ohlcv_dataroad.ingest import tda17_indicators as ind

try:                                                     # opcional (entorno aislado); si falta, ctypes / sin RSS
    import psutil
except ImportError:                                      # pragma: no cover - depende del entorno
    psutil = None

MB = 1024 * 1024
BLAS_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


class MotorE03Error(RuntimeError):
    """Fallo del motor (trabajador muerto, punto de control invalido, memoria): la ejecucion entera falla."""


def _fail(cid: str, msg: str):
    raise MotorE03Error(f"[{cid}] {msg}")


# =========================================================================== recursos
@dataclass
class Recursos:
    """Parametros de ejecucion (no estadisticos). ``procesos`` = 0: todo en el proceso principal (referencia)."""
    procesos: int = 20                       # maximo de workers (project_hardware.md: ~20 de 24 nucleos)
    reserva_mb: float = 4096.0               # memoria disponible que nunca se compromete
    hilos_blas: int | None = None            # None: los del proceso principal (OBLIGATORIO para bytes identicos)
    cortes_por_bloque: int = 12              # TH37 componente 1
    replicas_por_bloque: int = 10            # TH35 R2_TODAS
    minimo_medidas_eta: int = 5              # tareas medidas por tipo antes de estimar el tiempo restante
    intervalo_estado_s: float = 20.0
    max_sin_medir: int = 2                   # tareas simultaneas de un tipo hasta medir su primer pico
    margen_pico: float = 1.15
    memoria_estricta: bool = False           # ruta OFICIAL: ninguna tarea arranca si no cabe (E03_MEMORIA_INSUFICIENTE)
    espera_memoria_s: float = 10.0           # con memoria_estricta: espera a que el sistema libere antes de rechazar
    barajar: int | None = None               # SOLO pruebas: orden de despacho aleatorio (semilla)
    fallo_tras: int | None = None            # SOLO pruebas: interrupcion simulada tras n tareas nuevas

    def resumen(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# =========================================================================== memoria (psutil o ctypes)
class _MEMSTATUS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def memoria_disponible_mb() -> float:
    if psutil is not None:
        return psutil.virtual_memory().available / MB
    if sys.platform == "win32":
        st = _MEMSTATUS()
        st.dwLength = ctypes.sizeof(_MEMSTATUS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.ullAvailPhys / MB
    with open("/proc/meminfo", encoding="ascii") as fh:                      # pragma: no cover
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    raise MotorE03Error("[E03_MEMORIA] no se puede medir la memoria disponible")


def rss_mb(pid: int | None = None) -> float | None:
    if psutil is None:
        return None
    try:
        return psutil.Process(pid).memory_info().rss / MB
    except psutil.Error:
        return None


def privada_mb(pid: int | None = None) -> float | None:
    """Memoria PRIVADA del proceso (Windows: ``private`` = PrivateUsage, la memoria comprometida que no puede
    compartirse; otros sistemas: rss - shared). Las paginas de los ficheros proyectados de solo lectura no cuentan:
    son comunes a todos los workers y se contabilizan una vez por etapa (``_Motor._cabe``). Sin psutil: None."""
    if psutil is None:
        return None
    try:
        mi = psutil.Process(pid).memory_info()
    except psutil.Error:
        return None
    if hasattr(mi, "private"):
        return mi.private / MB
    return (mi.rss - getattr(mi, "shared", 0)) / MB                        # pragma: no cover - no Windows


def _openblas() -> list[tuple[str, object]]:
    """Bibliotecas OpenBLAS de numpy y scipy (``<pkg>.libs/*openblas*``), en orden."""
    import glob
    out = []
    for pkg in ("numpy", "scipy"):
        with contextlib.suppress(Exception):
            mod = __import__(pkg)
            for f in sorted(glob.glob(os.path.join(os.path.dirname(mod.__file__), "..", f"{pkg}.libs", "*openblas*"))):
                out.append((f"{pkg}:{os.path.basename(f)}", ctypes.CDLL(f)))
    return out


def _fn_openblas(lib, nombre: str):
    return getattr(lib, f"scipy_openblas_{nombre}64_", None) or getattr(lib, f"scipy_openblas_{nombre}", None)


def hilos_blas_efectivos() -> dict[str, int]:
    """Hilos de cada OpenBLAS cargable de numpy y scipy (``scipy_openblas_get_num_threads64_``). Los resultados de E02
    dependen de este valor (particion del trabajo de BLAS/LAPACK): los workers deben tener el mismo que el principal."""
    out = {}
    for nombre, lib in _openblas():
        fn = _fn_openblas(lib, "get_num_threads")
        if fn is not None:
            fn.restype = ctypes.c_int
            out[nombre] = int(fn())
    return out


@contextlib.contextmanager
def hilos_blas_fijados(n: int):
    """Fija los hilos EFECTIVOS de cada OpenBLAS de este proceso (``scipy_openblas_set_num_threads64_``), comprueba
    que todas las bibliotecas los tienen y los restaura al salir. Rechaza (``E03_BLAS``) si no hay ningun OpenBLAS
    verificable o si alguna biblioteca no los adopta. Los workers heredan los efectivos del principal
    (``_Motor._nuevo_worker``) y se verifican en cada tarea."""
    libs = [(nombre, lib, _fn_openblas(lib, "set_num_threads")) for nombre, lib in _openblas()]
    antes = hilos_blas_efectivos()
    if not libs or not antes or any(fn is None for _, _, fn in libs) or sorted(antes) != sorted(x for x, _, _ in libs):
        _fail("E03_BLAS", f"no se pueden fijar ni verificar los hilos de OpenBLAS de este proceso: {sorted(antes)}")
    try:
        for _, _, fn in libs:
            fn.argtypes = [ctypes.c_int]
            fn(int(n))
        efectivos = hilos_blas_efectivos()
        if set(efectivos.values()) != {int(n)} or sorted(efectivos) != sorted(antes):
            _fail("E03_BLAS", f"hilos BLAS efectivos {efectivos} distintos de los fijados ({n})")
        yield efectivos
    finally:
        for nombre, _, fn in libs:
            fn(int(antes[nombre]))


def entorno_numerico() -> dict:
    import scipy
    blas = None
    with contextlib.suppress(Exception):
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"].get("openblas configuration")
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "blas": blas, "hilos_blas": hilos_blas_efectivos(),
            "psutil": getattr(psutil, "__version__", None),
            "plataforma": platform.platform(), "nucleos_logicos": os.cpu_count()}


# =========================================================================== utilidades de ficheros
def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _dump(path: Path, obj) -> str:
    data = pickle.dumps(obj, protocol=5)
    _write_atomic(path, data)
    return _sha_bytes(data)


def _json(obj) -> bytes:
    return (json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n").encode("utf-8")


def _file_id(task_id: str) -> str:
    return hashlib.sha1(task_id.encode("utf-8")).hexdigest()[:20]


# =========================================================================== marcos compartidos de solo lectura
COMPARTIBLES = "fiub"                                    # bloques numpy que se proyectan: float, int, uint, bool


def _estructura(frame: pd.DataFrame) -> list[tuple]:
    """Estructura de bloques de pandas: (tipo, columnas que ocupa, forma, C-contiguo) por bloque, en orden."""
    return [(str(b.dtype), b.mgr_locs.as_array.tolist(), tuple(b.shape),
             bool(getattr(b.values, "flags", None) is not None and b.values.flags.c_contiguous)) for b in frame._mgr.blocks]


def _compartible(v) -> bool:
    return type(v) is np.ndarray and v.dtype.kind in COMPARTIBLES and v.ndim == 2 and v.flags.c_contiguous


def compartir_marcos(ej: "Ejecucion", nombre: str, marcos: tuple[pd.DataFrame, ...]) -> tuple[tuple[str, str], float]:
    """Escribe ``marcos`` para que los workers los abran SIN COPIA (``_abrir_marcos``): un ``.npy`` por bloque numerico
    C-contiguo y el resto de bloques, el indice, las columnas, los atributos y las banderas en el manifiesto. Antes de
    devolver, reconstruye los marcos desde los ficheros y comprueba que son identicos bit a bit a los originales
    (``E03_COMPARTIDO``). Devuelve (referencia del manifiesto, MB proyectados)."""
    manifiesto, total = [], 0
    for i, f in enumerate(marcos):
        bloques = []
        for j, b in enumerate(f._mgr.blocks):
            v, loc = b.values, b.mgr_locs.as_array.copy()
            if _compartible(v):
                bloques.append(("npy", ej.dato(f"{nombre}_{i}_{j:03d}.npy", v), str(v.dtype), v.shape, loc))
                total += v.nbytes
            else:
                bloques.append(("objeto", v, None, None, loc))
        manifiesto.append({"bloques": bloques, "index": f.index, "columns": f.columns, "attrs": copy.deepcopy(f.attrs),
                           "duplicados": f.flags.allows_duplicate_labels})
    ref = ej.dato(f"{nombre}.marcos.pkl", manifiesto)
    _verificar_reconstruccion(marcos, _abrir_marcos(ref))
    return ref, total / MB


def _abrir_marcos(ref: tuple[str, str]) -> tuple[pd.DataFrame, ...]:
    """Reconstruye los marcos de ``compartir_marcos`` con la MISMA estructura de bloques; los bloques numericos son
    vistas ``ndarray`` de solo lectura de la proyeccion del fichero (verificado por SHA-256), sin copia."""
    from pandas.api.internals import create_dataframe_from_blocks
    path, sha = ref
    if _sha_file(Path(path)) != sha:
        _fail("E03_DATOS", f"fichero de datos del worker alterado: {path}")
    out = []
    for m in pickle.loads(Path(path).read_bytes()):
        bloques = []
        for tipo, v, dtype, shape, loc in m["bloques"]:
            if tipo == "npy":
                p, s = v
                if _sha_file(Path(p)) != s:
                    _fail("E03_DATOS", f"fichero de datos del worker alterado: {p}")
                v = np.load(p, mmap_mode="r", allow_pickle=False).view(np.ndarray)
                if str(v.dtype) != dtype or v.shape != shape or v.flags.writeable:
                    _fail("E03_COMPARTIDO", f"bloque proyectado distinto del manifiesto: {p}")
            bloques.append((v, loc))
        f = create_dataframe_from_blocks(bloques, index=m["index"], columns=m["columns"])
        f.attrs.update(copy.deepcopy(m["attrs"]))
        f.flags.allows_duplicate_labels = m["duplicados"]
        out.append(f)
    return tuple(out)


def _verificar_reconstruccion(originales: tuple[pd.DataFrame, ...], reconstruidos: tuple[pd.DataFrame, ...]) -> None:
    """Identidad bit a bit entre cada marco y su reconstruccion: tipo, columnas, indice, tipos, estructura de bloques,
    atributos, banderas y contenido de cada bloque (SHA-256 de su memoria; los bloques no proyectados, por su pickle).
    Los bloques proyectados deben ser de solo lectura y estar respaldados por el fichero (sin copia)."""
    if len(originales) != len(reconstruidos):
        _fail("E03_COMPARTIDO", "numero de marcos reconstruidos distinto")
    for k, (a, b) in enumerate(zip(originales, reconstruidos)):
        same = (type(a) is type(b) and a.columns.equals(b.columns) and type(a.columns) is type(b.columns) and
                a.columns.dtype == b.columns.dtype and a.index.equals(b.index) and type(a.index) is type(b.index) and
                a.index.dtype == b.index.dtype and list(map(str, a.dtypes)) == list(map(str, b.dtypes)) and
                _estructura(a) == _estructura(b) and pickle.dumps(a.attrs) == pickle.dumps(b.attrs) and
                a.flags.allows_duplicate_labels == b.flags.allows_duplicate_labels)
        if not same:
            _fail("E03_COMPARTIDO", f"marco {k}: estructura reconstruida distinta de la original")
        for j, (x, y) in enumerate(zip(a._mgr.blocks, b._mgr.blocks)):
            if _compartible(x.values):
                if y.values.flags.writeable or not _respaldado_por_fichero(y.values):
                    _fail("E03_COMPARTIDO", f"marco {k}, bloque {j}: no es una proyeccion de solo lectura")
                if hashlib.sha256(x.values).digest() != hashlib.sha256(y.values).digest():
                    _fail("E03_COMPARTIDO", f"marco {k}, bloque {j}: contenido distinto del original")
            elif pickle.dumps(x.values, protocol=5) != pickle.dumps(y.values, protocol=5):
                _fail("E03_COMPARTIDO", f"marco {k}, bloque {j}: bloque no proyectado distinto del original")


def _respaldado_por_fichero(v: np.ndarray) -> bool:
    """La cadena de ``base`` del array termina en un ``mmap`` (proyeccion del fichero)."""
    import mmap
    b = v
    while isinstance(b, np.ndarray):
        if isinstance(b, np.memmap) and isinstance(b._mmap, mmap.mmap):
            return True
        b = b.base
    return isinstance(b, mmap.mmap)


# =========================================================================== tareas (se ejecutan en los workers)
_CACHE: dict[str, object] = {}
_CACHE_GRUPO: list[str | None] = [None]


def _cargar(ref: tuple[str, str], grupo: str):
    """Fichero de datos del worker (pickle, .npy o marcos compartidos) verificado por SHA-256; cache por grupo de
    etapa: al cambiar de grupo se libera la memoria del anterior (y las proyecciones de sus ficheros)."""
    path, sha = ref
    if _CACHE_GRUPO[0] != grupo:
        _CACHE.clear()
        gc.collect()
        _CACHE_GRUPO[0] = grupo
    if path not in _CACHE:
        if path.endswith(".marcos.pkl"):
            _CACHE[path] = _abrir_marcos(ref)
        else:
            if _sha_file(Path(path)) != sha:
                _fail("E03_DATOS", f"fichero de datos del worker alterado: {path}")
            _CACHE[path] = np.load(path, mmap_mode="r") if path.endswith(".npy") else pickle.loads(Path(path).read_bytes())
    return _CACHE[path]


def _liberar_cache() -> None:
    _CACHE.clear()
    _CACHE_GRUPO[0] = None
    gc.collect()


def _th37_datos(datos) -> tuple[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame]]:
    """(df, (series, marcas)) de TH37: marcos compartidos de solo lectura."""
    df, series, marks = _cargar(datos, "TH37")
    return df, (series, marks)


def _t_th37_future(datos, subplan: dict, cfg: dict) -> list[dict]:
    df, full = _th37_datos(datos)
    return ca.audit_future(df, cfg, [subplan], full)


def _t_th37_origin(datos, cfg: dict) -> list[dict]:
    df, full = _th37_datos(datos)
    return ca.audit_parameter_origin(df, cfg, full)


def _t_th37_roll(datos, rebanada, a: int, z: int, cfg: dict) -> list[dict]:
    """Cuerpo LITERAL del bucle de ``ca.audit_roll`` para el contrato [a, z). ``rebanada`` = (series_f, marks_f,
    tda_f) ya restringidas a [a, z) (las unicas filas que el bucle compara)."""
    df, _full = _th37_datos(datos)
    series_az, marks_az, tda_az = _cargar(rebanada, "TH37")
    d7s, d7m = ca._d7_columns(cfg)
    rows_idx = np.arange(len(df))
    out = []
    name = df["contract"].iloc[a]
    other = (rows_idx < a) | (rows_idx >= z)
    for tname, fn in ca.ROLL_TRANSFORMS.items():
        alt = df.copy()
        for col in ca.PRICE_COLUMNS:
            v = alt[col].to_numpy(float)
            alt[col] = np.where(other, fn(v, rows_idx), v)
        s2, m2 = ca._catalog(alt)
        motive = f"PRECIO_{tname}"
        out += ca._compare_frame("ROLL_PRECIO", name, None, motive, series_az, s2.iloc[a:z], list(series_az.columns))
        out += ca._compare_frame("ROLL_PRECIO", name, None, motive, marks_az, m2.iloc[a:z], list(marks_az.columns))
        out += ca._compare_frame("ROLL_PRECIO", name, None, motive, tda_az, ca._tda04(alt).iloc[a:z], list(tda_az.columns))
    alt = df.copy()
    v = alt["volume"].to_numpy(float)
    alt["volume"] = np.where(other, v + (rows_idx % 5) + 1, v)
    s2, m2 = ca._catalog(alt)
    non_d7 = [c for c in series_az.columns if c not in d7s]
    out += ca._compare_frame("ROLL_VOLUMEN", name, None, "VOLUMEN_DEMAS_CONTRATOS", series_az, s2.iloc[a:z], non_d7)
    out += ca._compare_frame("ROLL_VOLUMEN", name, None, "VOLUMEN_DEMAS_CONTRATOS", marks_az, m2.iloc[a:z], list(marks_az.columns))
    for col in d7s:
        cid = col.split("__")[0]
        cross = marks_az[f"{cid}__rvol_cruza_roll"].to_numpy(bool)
        fa, fb = series_az[col].to_numpy(), s2[col].iloc[a:z].to_numpy()
        disc, mad = ca.compare_arrays(fa[~cross], fb[~cross])
        out.append(ca._row("ROLL_VOLUMEN", name, None, "VOLUMEN_DEMAS_CONTRATOS (filas sin rvol_cruza_roll)", col,
                           int((~cross).sum()), disc, mad, "CONFORME" if disc == 0 else "DISCREPANCIA"))
    return out


def _t_th35_cell(datos_celda, cfg: dict, celda: str, draws: np.ndarray, momentos_path: str) -> dict:
    """``an.analyze_cell`` y, con matriz definida, lo que ``an.cell_bootstrap`` calcula antes de las replicas de R2:
    momentos diarios (que se guardan para los bloques de R2) y el IC de Spearman con rangos fijos."""
    sub, sub_df = _cargar(datos_celda, "TH35C")
    a = an.analyze_cell(sub, sub_df, cfg, celda=celda)
    out = {"analisis": a}
    if a.get("estado_matriz") != "DEFINIDA":
        return out
    X = sub.to_numpy(np.float64)
    names = list(sub.columns)
    day_codes = sub_df["trading_date"].to_numpy()
    m = an.daily_moments(X, day_codes)
    m_rank = an.daily_moments(an.average_ranks(X), day_codes)
    if m.n.size != draws.shape[1]:
        _fail("E03_SORTEOS", f"{celda}: jornadas de la celda ({m.n.size}) distintas de las de los sorteos ({draws.shape[1]})")
    out["jornadas"] = int(m.n.size)
    out["spearman"] = an.bootstrap_spearman_fixed(m_rank, draws, names, a["spearman"].to_numpy(), cfg)
    out["momentos"] = (momentos_path, _dump(Path(momentos_path), m))
    return out


def _t_th35_r2_block(momentos, draws_block: np.ndarray, p: int, todo: list[int], tau: float):
    """Cuerpo LITERAL del bucle de replicas de ``an.bootstrap_r2`` para un bloque de replicas consecutivas."""
    m = _cargar(momentos, "TH35B")
    R = draws_block.shape[0]
    vals = np.full((R, p), np.nan)
    cause = np.full((R, p), None, dtype=object)
    base = np.zeros((R, p), np.int64)
    for r, days in enumerate(draws_block):
        zero = an.replica_zero_variance(m, days)
        if zero.any():
            cause[r, todo] = "VAR_CERO"                                     # en R2_TODAS intervienen todas las columnas
            continue
        C = an.replica_correlation(m, days)
        for j in todo:
            vals[r, j], cause[r, j], base[r, j] = an.r2_official_checked(C, j, [k for k in range(p) if k != j], tau)
    return vals, cause, base


def _t_th35_val_prep(datos_val, rangos_path: str) -> dict:
    """Lo que ``an._validation_cell`` calcula antes de las replicas: rangos medios y momentos diarios de los rangos."""
    X, codes = _cargar(datos_val[0], "TH35V"), _cargar(datos_val[1], "TH35V")
    X = np.asarray(X, np.float64)
    ranks = an.average_ranks(X)
    m_rank = an.daily_moments(ranks, codes)
    return {"m_rank": (rangos_path, _dump(Path(rangos_path), m_rank)), "jornadas": int(m_rank.n.size)}


def _t_th35_val_replica(datos_val, m_rank_ref, days: np.ndarray) -> tuple:
    """Cuerpo LITERAL de una iteracion del bucle de ``an._validation_cell``."""
    X, codes = _cargar(datos_val[0], "TH35V"), _cargar(datos_val[1], "TH35V")
    X = np.asarray(X, np.float64)
    m_rank = _cargar(m_rank_ref, "TH35V")
    key = ("rbd", datos_val[1][0])
    if key not in _CACHE:
        _CACHE[key] = an.day_rows(codes)
    rbd = _CACHE[key]
    p = X.shape[1]
    iu = np.triu_indices(p, 1)
    zf = an.replica_zero_variance(m_rank, days)
    with np.errstate(invalid="ignore", divide="ignore"):
        cf = np.full((p, p), np.nan) if zf.all() else an.replica_correlation(m_rank, days)
    ce = an.exact_rank_replica_correlation(X, rbd, days)
    return bool(zf.any()), cf[iu], ce[iu]


TASKS = {f.__name__: f for f in (_t_th37_future, _t_th37_origin, _t_th37_roll, _t_th35_cell, _t_th35_r2_block,
                                 _t_th35_val_prep, _t_th35_val_replica)}


def _muestreo_memoria(stop: threading.Event, out: list) -> None:
    while not stop.wait(0.25):
        for k, fn in enumerate((privada_mb, rss_mb)):
            v = fn()
            if v is not None and v > out[k]:
                out[k] = v


def _ejecutar_tarea(fn_name: str, args: tuple):
    """Ejecuta una tarea midiendo duracion y picos de memoria PRIVADA (la que decide el control de RAM) y RSS (incluye
    las paginas compartidas) del proceso durante la tarea."""
    inicio = [privada_mb() or 0.0, rss_mb() or 0.0]
    pico = list(inicio)
    stop = threading.Event()
    th = threading.Thread(target=_muestreo_memoria, args=(stop, pico), daemon=True)
    th.start()
    t0 = time.perf_counter()
    try:
        res = TASKS[fn_name](*args)
    finally:
        stop.set()
        th.join()
    fin = [privada_mb() or 0.0, rss_mb() or 0.0]
    return res, {"duracion_s": time.perf_counter() - t0, "privada_inicio_mb": inicio[0],
                 "privada_pico_mb": max(pico[0], fin[0]), "privada_fin_mb": fin[0], "rss_inicio_mb": inicio[1],
                 "rss_pico_mb": max(pico[1], fin[1]), "rss_fin_mb": fin[1], "hilos_blas": hilos_blas_efectivos()}


def _trabajador(conn) -> None:
    """Bucle de un worker: recibe (id, funcion, argumentos), devuelve ("ok"|"error", id, resultado, medidas)."""
    while True:
        msg = conn.recv()
        if msg is None:
            return
        task_id, fn_name, args = msg
        try:
            res, med = _ejecutar_tarea(fn_name, args)
            conn.send(("ok", task_id, res, med))
        except BaseException as exc:                     # noqa: BLE001 - se reenvia al principal y se relanza alli
            tb = traceback.format_exc()
            try:
                pickle.dumps(exc)
                payload = exc
            except Exception:                            # noqa: BLE001
                payload = MotorE03Error(f"[E03_TAREA] {type(exc).__name__}: {exc}")
            conn.send(("error", task_id, payload, {"traceback": tb}))
            if not isinstance(exc, Exception):
                return


# =========================================================================== planificador
@dataclass
class Tarea:
    id: str
    fn: str
    args: tuple
    tipo: str                                   # tipo de tarea: memoria y tiempos se miden por tipo
    rasgos: tuple = (1.0,)                      # variables del modelo de duracion (estimacion del tiempo restante)
    pico_inicial_mb: float = 1024.0             # estimacion de memoria PRIVADA previa a la primera medida
    deps: tuple = ()
    compartido: tuple | None = None             # (id, MB) de los ficheros proyectados que usa (se cuentan una vez)


@dataclass
class _Worker:
    proc: object
    conn: object
    pid: int
    tarea: Tarea | None = None
    inicio: float = 0.0
    activos_al_inicio: int = 0


@dataclass
class _Etapa:
    nombre: str
    total: int
    hechas: int = 0
    reutilizadas: int = 0
    inicio: float = field(default_factory=time.time)
    fin: float | None = None


@contextlib.contextmanager
def _hilos_blas(n: int):
    """Los workers ``spawn`` heredan estas variables al arrancar y fijan los hilos BLAS al importar numpy. El proceso
    principal no cambia (sus calculos son los del motor E02)."""
    old = {k: os.environ.get(k) for k in BLAS_VARIABLES}
    for k in BLAS_VARIABLES:
        os.environ[k] = str(n)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class Ejecucion:
    """Directorio de una ejecucion: identidad, datos de los workers, puntos de control, progreso y estado.

    Identidad = configuracion, marco de entrada, motor y entorno numerico. Un punto de control solo se reutiliza
    con ``reanudar=True``, identidad identica y SHA-256 del fichero igual al registrado; si no, se rechaza.

    ``oficial=True`` (ruta oficial de E03): la reanudacion esta DESHABILITADA (``E03_REANUDACION_OFICIAL``) y el
    directorio no puede existir; ningun punto de control se lee nunca. El SHA-256 del registro de progreso vive en el
    mismo directorio que el punto de control: detecta errores, pero no impide que quien pueda escribir el directorio
    sustituya ambos (no es una prueba de autenticidad)."""

    def __init__(self, directorio: Path | str, identidad: dict, *, reanudar: bool = False, oficial: bool = False):
        self.dir = Path(directorio)
        self.datos = self.dir / "datos"
        self.tareas = self.dir / "tareas"
        self.identidad = identidad
        self.reanudar = reanudar
        self.oficial = oficial
        ident_path = self.dir / "identidad.json"
        if oficial and reanudar:
            _fail("E03_REANUDACION_OFICIAL", "la reanudacion de una ejecucion oficial esta deshabilitada")
        if oficial and os.path.lexists(self.dir):
            _fail("E03_EJECUCION", f"el directorio de una ejecucion oficial no puede existir: {self.dir}")
        if reanudar:
            if not ident_path.is_file():
                _fail("E03_CHECKPOINT", f"no hay ejecucion que reanudar en {self.dir}")
            prev = json.loads(ident_path.read_text(encoding="utf-8"))
            if prev != json.loads(_json(identidad)):
                _fail("E03_CHECKPOINT", "la identidad de la ejecucion (configuracion, entradas, motor o entorno) no coincide")
            if (self.dir / "COMPLETA.json").exists():
                _fail("E03_CHECKPOINT", "la ejecucion ya esta completa: no se reanuda")
        else:
            if self.dir.exists() and any(self.dir.iterdir()):
                _fail("E03_EJECUCION", f"el directorio de la ejecucion no esta vacio: {self.dir}")
            self.dir.mkdir(parents=True, exist_ok=True)
            _write_atomic(ident_path, _json(identidad))
        self.datos.mkdir(exist_ok=True)
        self.tareas.mkdir(exist_ok=True)
        self.registro = self._leer_registro()
        self.reutilizadas: list[str] = []

    def _leer_registro(self) -> dict[str, dict]:
        out = {}
        path = self.dir / "progreso.jsonl"
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                with contextlib.suppress(ValueError):
                    rec = json.loads(line)
                    if rec.get("evento") == "tarea_completa":
                        out[rec["id"]] = rec
        return out

    def evento(self, **rec) -> None:
        rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec}
        with open(self.dir / "progreso.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def dato(self, nombre: str, obj) -> tuple[str, str]:
        """Escribe un fichero de datos para los workers y devuelve (ruta, SHA-256). Si ya existe con el mismo
        contenido (reanudacion sintetica) no se reescribe: un fichero proyectado no puede sustituirse en Windows."""
        path = self.datos / nombre
        tmp = path.with_name(path.name + ".tmp")
        if nombre.endswith(".npy"):                      # conserva el orden de memoria (C o Fortran) del original
            with open(tmp, "wb") as fh:
                np.save(fh, obj, allow_pickle=False)
                fh.flush()
                os.fsync(fh.fileno())
        else:
            tmp.write_bytes(pickle.dumps(obj, protocol=5))
        sha = _sha_file(tmp)
        if path.is_file() and _sha_file(path) == sha:
            tmp.unlink()
        else:
            os.replace(tmp, path)
        return str(path), sha

    def guardado(self, task_id: str):
        """(True, resultado) si hay punto de control valido de ``task_id`` (solo al reanudar; nunca en la oficial)."""
        if self.oficial or not self.reanudar or task_id not in self.registro:
            return False, None
        rec = self.registro[task_id]
        path = self.tareas / f"{_file_id(task_id)}.pkl"
        if not path.is_file() or _sha_file(path) != rec["sha256"]:
            _fail("E03_CHECKPOINT", f"punto de control alterado o ausente: {task_id}")
        res = pickle.loads(path.read_bytes())
        for ref in _referencias(res):
            if not Path(ref[0]).is_file() or _sha_file(Path(ref[0])) != ref[1]:
                _fail("E03_CHECKPOINT", f"fichero derivado alterado o ausente: {ref[0]} ({task_id})")
        self.reutilizadas.append(task_id)
        return True, res

    def guardar(self, task_id: str, res, med: dict, etapa: str) -> None:
        sha = _dump(self.tareas / f"{_file_id(task_id)}.pkl", res)
        self.evento(evento="tarea_completa", etapa=etapa, id=task_id, sha256=sha, **med)

    def estado(self, doc: dict) -> None:
        _write_atomic(self.dir / "estado.json", _json(doc))


def _referencias(res) -> list[tuple[str, str]]:
    """Ficheros derivados citados por un resultado (momentos guardados por una tarea)."""
    if isinstance(res, dict):
        return [v for k, v in res.items() if k in ("momentos", "m_rank")]
    return []


class _Motor:
    """Ejecuta un conjunto de tareas independientes (con dependencias simples) y devuelve {id: resultado}.

    Despacho: tareas listas en orden de prioridad (coste estimado descendente) o barajadas (pruebas). Una tarea se
    lanza si memoria_disponible - crecimiento_posible_de_las_activas - crecimiento_de_esta >= reserva, donde el
    crecimiento usa el pico medido del tipo (o la estimacion inicial, con a lo sumo ``max_sin_medir`` simultaneas
    hasta la primera medida). Si no cabe y hay activas, espera a que terminen. Si no cabe y NO hay ninguna activa:
    con ``memoria_estricta`` (ruta oficial) cierra los workers libres, espera ``espera_memoria_s`` a que el sistema
    libere memoria y, si sigue sin caber, RECHAZA la ejecucion (``E03_MEMORIA_INSUFICIENTE``) sin lanzar la tarea;
    sin ella (pruebas y benchmarks) la lanza sola (equivale al motor secuencial) y lo registra.
    El numero de workers se limita ademas por el rendimiento medido: si con mas workers activos la productividad
    del tipo cae, se fija el nivel mejor."""

    def __init__(self, recursos: Recursos, ejecucion: Ejecucion):
        self.r = recursos
        self.ej = ejecucion
        self.workers: list[_Worker] = []
        self.pico: dict[str, float] = {}
        self.medidas: dict[str, list] = collections.defaultdict(list)     # (rasgos, duracion, activos)
        self.tope_rendimiento: dict[str, int] = {}
        self.etapas: list[_Etapa] = []
        self.nuevas = 0
        self.ctx = mp.get_context("spawn")
        self.t0 = time.time()
        self.memoria_min = math.inf
        self.max_activos = 0
        self.secuencial_por_memoria = 0
        self.hilos_principal = hilos_blas_efectivos()
        self.compartido_cargado: set[str] = set()      # ficheros proyectados ya residentes (tarea completada)

    # ------------------------------------------------------------------ workers
    def _nuevo_worker(self) -> _Worker:
        a, b = self.ctx.Pipe(duplex=True)
        hilos = self.r.hilos_blas
        if hilos is None and len(set(self.hilos_principal.values())) == 1:
            hilos = next(iter(self.hilos_principal.values()))  # los EFECTIVOS del principal, no el entorno heredado
        with _hilos_blas(hilos) if hilos is not None else contextlib.nullcontext():
            p = self.ctx.Process(target=_trabajador, args=(b,), daemon=True)
            p.start()
        b.close()
        w = _Worker(proc=p, conn=a, pid=p.pid)
        self.workers.append(w)
        self.ej.evento(evento="worker_inicio", pid=p.pid, workers=len(self.workers))
        return w

    def cerrar(self) -> None:
        for w in self.workers:
            with contextlib.suppress(Exception):
                if w.tarea is None:
                    w.conn.send(None)
        for w in self.workers:
            w.proc.join(timeout=5)
            if w.proc.is_alive():
                w.proc.terminate()
                w.proc.join(timeout=5)
        self.workers = []
        _liberar_cache()                                 # proyecciones abiertas en el principal (procesos = 0)

    # ------------------------------------------------------------------ politica de memoria y rendimiento
    def _pico(self, t: Tarea) -> float:
        return self.pico.get(t.tipo, t.pico_inicial_mb)

    def _activos(self) -> list[_Worker]:
        return [w for w in self.workers if w.tarea is not None]

    def _compartido_pendiente(self, tareas: list[Tarea]) -> float:
        """MB de ficheros proyectados que las tareas usan y que aun pueden no estar residentes: se cuentan UNA vez
        (son paginas comunes a todos los workers) hasta que se completa la primera tarea que los usa (el worker los
        lee enteros al verificar su SHA-256)."""
        pend = {t.compartido[0]: t.compartido[1] for t in tareas
                if t.compartido is not None and t.compartido[0] not in self.compartido_cargado}
        return float(sum(pend.values()))

    def _balance_memoria(self, t: Tarea, w: _Worker | None) -> dict:
        """Memoria disponible - compartida pendiente - crecimiento posible de las activas - crecimiento de ``t`` en
        ``w`` (o en un worker nuevo), frente a la reserva."""
        activos = self._activos()
        disp = memoria_disponible_mb()
        crec = sum(max(0.0, self._pico(x.tarea) - (privada_mb(x.pid) or 0.0)) for x in activos)
        base = (privada_mb(w.pid) or 0.0) if w is not None else 0.0
        compartida = self._compartido_pendiente([x.tarea for x in activos] + [t])
        prevista = max(0.0, self._pico(t) - base)
        return {"disponible_mb": disp, "compartida_pendiente_mb": compartida, "crecimiento_activas_mb": crec,
                "prevista_mb": prevista, "reserva_mb": self.r.reserva_mb,
                "cabe": disp - compartida - crec - prevista >= self.r.reserva_mb}

    def _cabe(self, t: Tarea, w: _Worker | None) -> bool:
        """Topes de concurrencia (solo con activas) y balance de memoria (SIEMPRE, tambien sin activas)."""
        activos = self._activos()
        if activos:
            if t.tipo not in self.pico and sum(1 for x in activos if x.tarea.tipo == t.tipo) >= self.r.max_sin_medir:
                return False
            tope = self.tope_rendimiento.get(t.tipo)
            if tope is not None and sum(1 for x in activos if x.tarea.tipo == t.tipo) >= tope:
                return False
        return self._balance_memoria(t, w)["cabe"]

    def _cerrar_libres(self) -> None:
        """Cierra los workers sin tarea (sus caches ocupan memoria privada); solo se llama sin tareas activas."""
        libres = [w for w in self.workers if w.tarea is None]
        for w in libres:
            with contextlib.suppress(Exception):
                w.conn.send(None)
        for w in libres:
            w.proc.join(timeout=10)
            if w.proc.is_alive():
                w.proc.terminate()
                w.proc.join(timeout=10)
            self.ej.evento(evento="worker_cerrado_por_memoria", pid=w.pid)
        self.workers = [w for w in self.workers if w.tarea is not None]

    def _sin_memoria_y_sin_activas(self, t: Tarea, w: _Worker | None) -> _Worker | None:
        """La tarea no cabe y no hay ninguna activa. Estricta: cierra los workers libres, espera a que el sistema
        libere y, si sigue sin caber, rechaza SIN lanzar la tarea (no se elimina la reserva). No estricta: la lanza
        sola y lo registra. Devuelve el worker a usar (None: uno nuevo)."""
        if not self.r.memoria_estricta:
            self.secuencial_por_memoria += 1
            self.ej.evento(evento="memoria_baja_secuencial", id=t.id, disponible_mb=round(memoria_disponible_mb()))
            return w
        self._cerrar_libres()
        limite = time.time() + self.r.espera_memoria_s
        while True:
            b = self._balance_memoria(t, None)
            if b["cabe"]:
                self.ej.evento(evento="memoria_recuperada", id=t.id, disponible_mb=round(b["disponible_mb"]))
                return None
            if time.time() >= limite:
                break
            time.sleep(0.5)
        self.ej.evento(evento="memoria_insuficiente", id=t.id, **{k: round(v) for k, v in b.items() if k != "cabe"})
        _fail("E03_MEMORIA_INSUFICIENTE",
              f"la tarea {t.id} no cabe: disponibles {b['disponible_mb']:.0f} MB - compartidos pendientes "
              f"{b['compartida_pendiente_mb']:.0f} MB - prevista {b['prevista_mb']:.0f} MB < reserva "
              f"{b['reserva_mb']:.0f} MB; no se lanza y la ejecucion se rechaza")

    def _actualizar_rendimiento(self, tipo: str) -> None:
        """Productividad (tareas por segundo y worker * activos) por nivel de concurrencia; si un nivel mayor rinde
        menos que uno menor (con >= 3 medidas en cada uno), el tipo queda limitado al nivel menor."""
        por_nivel = collections.defaultdict(list)
        for rasgos, dur, activos in self.medidas[tipo]:
            por_nivel[activos].append(dur / max(1e-9, float(np.sum(rasgos))))
        niveles = sorted(n for n, v in por_nivel.items() if len(v) >= 3)
        if len(niveles) < 2:
            return
        prod = {n: n / float(np.median(por_nivel[n])) for n in niveles}
        mejor = max(prod, key=prod.get)
        if mejor < niveles[-1] and prod[niveles[-1]] < 0.95 * prod[mejor]:
            self.tope_rendimiento[tipo] = mejor

    # ------------------------------------------------------------------ estimacion del tiempo restante
    def _modelo(self, tipo: str):
        med = self.medidas[tipo]
        k = len(med[0][0]) if med else 1
        if len(med) < max(self.r.minimo_medidas_eta, k + 2):
            return None
        A = np.array([m[0] for m in med], float)
        y = np.array([m[1] for m in med], float)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return np.clip(coef, 0.0, None)

    def _eta(self, pendientes: list[Tarea]) -> dict:
        activos = self._activos()
        todas = pendientes + [w.tarea for w in activos]
        tipos = sorted({t.tipo for t in todas})
        modelos = {t: self._modelo(t) for t in tipos}
        sin = [t for t, m in modelos.items() if m is None]
        con = [t for t in tipos if t not in sin]
        if not con:
            return {"segundos": None, "estado": "NO_ESTIMABLE", "motivo": f"menos de {self.r.minimo_medidas_eta} "
                    f"tareas medidas de: {sin}"}
        trabajo = sum(float(np.dot(modelos[t.tipo], t.rasgos)) for t in pendientes if t.tipo in con)
        ahora = time.time()
        trabajo += sum(max(0.0, float(np.dot(modelos[w.tarea.tipo], w.tarea.rasgos)) - (ahora - w.inicio))
                       for w in activos if w.tarea.tipo in con)
        paralelo = max(1, len(activos))
        out = {"segundos": trabajo / paralelo, "estado": "ESTIMADO",
               "metodo": "duracion ~ rasgos (minimos cuadrados por tipo con las tareas medidas) / workers activos",
               "medidas": {t: len(self.medidas[t]) for t in tipos}}
        if sin:                                          # solo los tipos medidos: el resto no se inventa
            out.update(estado="COTA_INFERIOR", sin_estimar={t: sum(1 for x in todas if x.tipo == t) for t in sin},
                       motivo=f"menos de {self.r.minimo_medidas_eta} tareas medidas de: {sin} (excluidas)")
        return out

    def _escribir_estado(self, etapa: _Etapa, pendientes: list[Tarea], estado: str = "EN_CURSO") -> None:
        disp = memoria_disponible_mb()
        self.memoria_min = min(self.memoria_min, disp)
        self.ej.estado({
            "estado": estado, "etapa": etapa.nombre, "transcurrido_s": round(time.time() - self.t0, 1),
            "etapas": [{"etapa": e.nombre, "hechas": e.hechas, "total": e.total, "reutilizadas": e.reutilizadas,
                        "progreso": f"{e.hechas} de {e.total}",
                        "duracion_s": round((e.fin or time.time()) - e.inicio, 1)} for e in self.etapas],
            "workers_activos": len(self._activos()), "workers_vivos": len(self.workers),
            "memoria_disponible_mb": round(disp), "memoria_disponible_min_mb": round(self.memoria_min),
            "rss_workers_mb": {w.pid: round(rss_mb(w.pid) or 0) for w in self.workers},
            "privada_workers_mb": {w.pid: round(privada_mb(w.pid) or 0) for w in self.workers},
            "pico_privado_por_tipo_mb": {k: round(v) for k, v in self.pico.items()},
            "tope_por_rendimiento": self.tope_rendimiento,
            "tiempo_restante_etapa": self._eta(pendientes) if estado == "EN_CURSO" else None,
            "recursos": self.r.resumen()})

    # ------------------------------------------------------------------ ejecucion
    def ejecutar(self, etapa_nombre: str, tareas: list[Tarea]) -> dict[str, object]:
        ids = [t.id for t in tareas]
        if len(set(ids)) != len(ids):
            raise ValueError(f"tareas repetidas en {etapa_nombre}")
        etapa = _Etapa(etapa_nombre, len(tareas))
        self.etapas.append(etapa)
        self.ej.evento(evento="etapa_inicio", etapa=etapa_nombre, total=len(tareas))
        res: dict[str, object] = {}
        pendientes = []
        for t in tareas:
            ok, val = self.ej.guardado(t.id)
            if ok:
                res[t.id] = val
                etapa.hechas += 1
                etapa.reutilizadas += 1
                self.ej.evento(evento="tarea_reutilizada", etapa=etapa_nombre, id=t.id)
            else:
                pendientes.append(t)
        if self.r.barajar is not None:
            random.Random(self.r.barajar).shuffle(pendientes)
        else:
            pendientes.sort(key=lambda t: -float(np.sum(t.rasgos)))
        if self.r.procesos <= 0:
            self._ejecutar_local(etapa, pendientes, res)
        else:
            self._ejecutar_procesos(etapa, pendientes, res)
        etapa.fin = time.time()
        self.ej.evento(evento="etapa_fin", etapa=etapa_nombre, duracion_s=round(etapa.fin - etapa.inicio, 3),
                       hechas=etapa.hechas, reutilizadas=etapa.reutilizadas)
        self._escribir_estado(etapa, [], estado="ETAPA_COMPLETA")
        return res

    def _completar(self, etapa: _Etapa, t: Tarea, val, med: dict, activos: int, res: dict) -> None:
        if self.r.hilos_blas is None and med.get("hilos_blas") != self.hilos_principal:
            _fail("E03_BLAS", f"hilos BLAS del worker {med.get('hilos_blas')} distintos de los del proceso principal "
                              f"{self.hilos_principal} (los resultados de E02 dependen de ellos): {t.id}")
        res[t.id] = val
        etapa.hechas += 1
        self.nuevas += 1
        self.medidas[t.tipo].append((tuple(t.rasgos), med["duracion_s"], activos))
        pico = med.get("privada_pico_mb") or med.get("rss_pico_mb")
        if pico:
            prev = self.pico.get(t.tipo, 0.0)
            self.pico[t.tipo] = max(prev, pico * self.r.margen_pico)
        if t.compartido is not None:
            self.compartido_cargado.add(t.compartido[0])
        self._actualizar_rendimiento(t.tipo)
        self.ej.guardar(t.id, val, {**med, "activos": activos, "tipo": t.tipo}, etapa.nombre)
        if self.r.fallo_tras is not None and self.nuevas >= self.r.fallo_tras:
            _fail("E03_INTERRUPCION_SIMULADA", f"interrupcion simulada tras {self.nuevas} tareas (solo pruebas)")

    def _ejecutar_local(self, etapa: _Etapa, pendientes: list[Tarea], res: dict) -> None:
        ultimo = 0.0
        for t in self._orden_deps(pendientes, res):
            val, med = _ejecutar_tarea(t.fn, self._resolver(t, res))
            self._completar(etapa, t, val, med, 1, res)
            if time.time() - ultimo >= self.r.intervalo_estado_s:
                self._escribir_estado(etapa, [x for x in pendientes if x.id not in res])
                ultimo = time.time()

    @staticmethod
    def _orden_deps(pendientes: list[Tarea], res: dict) -> list[Tarea]:
        hechas, out, resto = set(res), [], list(pendientes)
        while resto:
            listas = [t for t in resto if all(d in hechas for d in t.deps)]
            if not listas:
                raise ValueError("dependencias circulares o ausentes")
            for t in listas:
                out.append(t)
                hechas.add(t.id)
            resto = [t for t in resto if t.id not in hechas]
        return out

    @staticmethod
    def _resolver(t: Tarea, res: dict) -> tuple:
        """Argumentos con marcadores ``("__dep__", id, clave)`` sustituidos por el resultado de la dependencia."""
        def sub(a):
            if isinstance(a, tuple) and len(a) == 3 and a[0] == "__dep__":
                return res[a[1]][a[2]]
            return a
        return tuple(sub(a) for a in t.args)

    def _ejecutar_procesos(self, etapa: _Etapa, pendientes: list[Tarea], res: dict) -> None:
        cola = list(pendientes)
        ultimo = 0.0
        try:
            while cola or self._activos():
                self._despachar(cola, res)
                activos = self._activos()
                if not activos:
                    if cola:
                        raise ValueError("dependencias ausentes en la etapa")
                    break
                listos = mpc.wait([w.conn for w in activos] + [w.proc.sentinel for w in activos], timeout=0.5)
                for w in activos:
                    if w.conn in listos or w.conn.poll():
                        try:
                            tipo, tid, val, med = w.conn.recv()
                        except (EOFError, OSError):
                            _fail("E03_TRABAJADOR", f"el worker {w.pid} termino durante la tarea {w.tarea.id}")
                        t, n_act = w.tarea, w.activos_al_inicio
                        w.tarea = None
                        if tipo == "error":
                            self.ej.evento(evento="tarea_error", etapa=etapa.nombre, id=tid,
                                           error=f"{type(val).__name__}: {val}", traceback=med.get("traceback"))
                            raise val
                        self._completar(etapa, t, val, med, n_act, res)
                    elif w.proc.sentinel in listos and not w.proc.is_alive():
                        _fail("E03_TRABAJADOR", f"el worker {w.pid} termino (codigo {w.proc.exitcode}) durante la "
                                                f"tarea {w.tarea.id}")
                if time.time() - ultimo >= self.r.intervalo_estado_s:
                    self._escribir_estado(etapa, cola)
                    ultimo = time.time()
        except BaseException:
            self._abortar()
            raise

    def _despachar(self, cola: list[Tarea], res: dict) -> None:
        while cola:
            hechas = set(res)
            t = next((x for x in cola if all(d in hechas for d in x.deps)), None)
            if t is None:
                return
            libres = [w for w in self.workers if w.tarea is None and w.proc.is_alive()]
            w = libres[0] if libres else None
            if w is None and len(self.workers) >= self.r.procesos:
                return
            if not self._cabe(t, w):
                if self._activos():
                    return                               # espera a que terminen las activas
                w = self._sin_memoria_y_sin_activas(t, w)
            if w is None:
                w = self._nuevo_worker()
            cola.remove(t)
            w.tarea, w.inicio = t, time.time()
            w.activos_al_inicio = len(self._activos())
            self.max_activos = max(self.max_activos, w.activos_al_inicio)
            w.conn.send((t.id, t.fn, self._resolver(t, res)))

    def _abortar(self) -> None:
        for w in self.workers:
            with contextlib.suppress(Exception):
                w.proc.terminate()
        for w in self.workers:
            with contextlib.suppress(Exception):
                w.proc.join(timeout=10)
        self.workers = []

    def resumen(self) -> dict:
        return {"etapas": [{"etapa": e.nombre, "tareas": e.total, "reutilizadas": e.reutilizadas,
                            "duracion_s": round((e.fin or time.time()) - e.inicio, 3)} for e in self.etapas],
                "pico_por_tipo_mb": {k: round(v / self.r.margen_pico) for k, v in self.pico.items()},
                "max_workers_activos": self.max_activos, "tope_por_rendimiento": self.tope_rendimiento,
                "memoria_disponible_min_mb": None if self.memoria_min == math.inf else round(self.memoria_min),
                "lanzadas_sin_memoria_suficiente": self.secuencial_por_memoria, "hilos_blas": self.hilos_principal,
                "medida_memoria": "PRIVADA" if privada_mb() is not None else "NO_DISPONIBLE",
                "datos_compartidos": sorted(self.compartido_cargado),
                "reutilizadas": list(self.ej.reutilizadas), "recursos": self.r.resumen()}


# =========================================================================== TH37 (equivale a ca.run_causal_audit)
def _bloques(cortes: list, tam: int) -> list[list]:
    return [cortes[i:i + tam] for i in range(0, len(cortes), tam)] or [[]]


def causal_audit_e03(df: pd.DataFrame, cfg: dict, motor: _Motor) -> pd.DataFrame:
    """Misma tabla que ``ca.run_causal_audit(df, cfg)``: mismas comprobaciones previas, mismo catalogo y plan, mismas
    funciones por tarea y concatenacion en el orden de E02 (componente 1 por plan y corte, 2, 3 por contrato)."""
    ca.audit_config(cfg)
    ca.check_catalog_config(cfg)
    ca.check_unadjusted_input(df)
    df = df.reset_index(drop=True)
    full = ca._catalog(df)
    plan = ca.contract_cut_plan(df, *full, cfg)
    # comprobaciones de audit_roll previas al bucle (en el principal, como en E02)
    ac = ca.audit_config(cfg)
    if ac["roll"]["precio"]["transformaciones"] != list(ca.ROLL_TRANSFORMS) or ac["roll"]["volumen"]["solo_puede_cambiar"] != "D7":
        raise ca.ProtocolNotAuthorizedError("auditoria de roll distinta de la implementada")
    ca.check_unadjusted_input(df)
    tda_f = ca._tda04(df)
    ej = motor.ej
    datos, compartido_mb = compartir_marcos(ej, "th37", (df, *full))      # una copia para todos los workers
    comp = ("th37", compartido_mb)
    n = len(df)
    pico = 4.0 * (df.memory_usage(deep=True).sum() + full[0].memory_usage().sum() + full[1].memory_usage(deep=True).sum()) / MB
    tareas, orden = [], []
    for i, p in enumerate(plan):
        cortes = list(p["cortes"].items())
        for k, bloque in enumerate(_bloques(cortes, motor.r.cortes_por_bloque)):
            sub = {**p, "cortes": dict(bloque), "no_aplicables": p["no_aplicables"] if k == 0 else []}
            tid = f"TH37F/{i:03d}/{k:04d}"
            g = sum(p["a"] + c for c, _ in bloque)
            tareas.append(Tarea(tid, "_t_th37_future", (datos, sub, cfg), "TH37F", (float(len(bloque)), g / 1e6),
                                pico, (), comp))
            orden.append(tid)
    tareas.append(Tarea("TH37O", "_t_th37_origin", (datos, cfg), "TH37O", (1.0,), pico, (), comp))
    orden.append("TH37O")
    roll_ids = []
    for i, (a, z) in enumerate(ind.contract_segments(df)):
        reb = ej.dato(f"th37_roll_{i:03d}.pkl", (full[0].iloc[a:z], full[1].iloc[a:z], tda_f.iloc[a:z]))
        tid = f"TH37R/{i:03d}"
        tareas.append(Tarea(tid, "_t_th37_roll", (datos, reb, int(a), int(z), cfg), "TH37R", (float(n) / 1e6,), pico,
                            (), comp))
        roll_ids.append(tid)
    del full, tda_f                                   # ya estan en los ficheros de los workers: libera RAM del principal
    gc.collect()
    res = motor.ejecutar("TH37", tareas)
    rows: list[dict] = []
    for tid in orden:
        rows += res[tid]
    rows.append(ca._row("ROLL_PRECIO", None, None, "ENTRADA_SIN_AJUSTES", "COLUMNAS_DE_ENTRADA", len(df), 0, 0.0, "CONFORME"))
    for tid in roll_ids:
        rows += res[tid]
    return ca.audit_table(rows)


# =========================================================================== TH35 (equivale a an.th35_pipeline)
def _r2_config(cfg: dict):
    """Comprobaciones y parametros de ``an.bootstrap_r2`` (literal)."""
    er = an.e02_value(cfg, "analisis.bootstrap.estadistico_replica")
    if er.get("r2") != "METODO_OFICIAL_SOBRE_LA_CORRELACION_DE_LA_REPLICA" or \
            er.get("replica_invalida", {}).get("tratamiento") != "IC_NO_CALCULABLE_SI_ALGUNA_INVALIDA" or \
            list(er.get("replica_invalida", {}).get("causas", [])) != list(an.INVALID_CAUSES):
        raise an.ProtocolNotAuthorizedError(f"estadistico de replica distinto del implementado: {er}")
    tau = cfg["analisis"]["r2"]["tau_relativa"]
    q = an.interval_quantiles(cfg)
    reg = an.registered_columns(cfg)
    return tau, q, reg


def _r2_aggregate(vals, cause, base, names, point: pd.DataFrame, q, reg, R: int) -> pd.DataFrame:
    """Agregacion LITERAL de ``an.bootstrap_r2`` sobre las R replicas reunidas en orden."""
    pt = point.set_index("columna")
    rows = []
    for j, nm in enumerate(names):
        rec = {"columna": nm, "r2": float(pt.loc[nm, "r2"]), "replicas": R, "ic_inf": np.nan, "ic_sup": np.nan,
               "replicas_invalidas": 0, "invalidas_por_causa": {}, "indices_invalidas": []}
        if nm in reg:
            rec.update(ic_estado="DEGENERADO_REGISTRO", ic_inf=1.0, ic_sup=1.0)
        else:
            bad = np.flatnonzero(cause[:, j] != None)                          # noqa: E711 (array de objetos)
            rec.update(replicas_invalidas=int(bad.size), indices_invalidas=bad.tolist(),
                       invalidas_por_causa={c: int(np.sum(cause[:, j] == c)) for c in an.INVALID_CAUSES if np.any(cause[:, j] == c)})
            if not bool(pt.loc[nm, "publicable"]):
                rec["ic_estado"] = "R2_NO_PUBLICABLE"
            elif bad.size:
                rec["ic_estado"] = "IC_NO_CALCULABLE"
            else:
                rec["ic_estado"] = "CALCULADO"
                rec["ic_inf"], rec["ic_sup"] = an.percentile_interval(vals[:, j], q)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.attrs.update(tamano_base=base, cuantiles=q)
    return out


def r2_block_tasks(cell: str, momentos: tuple[str, str], draws: np.ndarray, p: int, todo: list[int], tau: float,
                   por_bloque: int, jornadas: int) -> list[Tarea]:
    """Tareas de R2_TODAS de una celda: bloques de replicas CONSECUTIVAS de los sorteos del principal."""
    pico = 3.0 * jornadas * p * p * 8 / MB + 512
    return [Tarea(f"TH35B/{cell}/{b0:05d}", "_t_th35_r2_block", (momentos, draws[b0:b0 + por_bloque], p, todo, tau),
                  "TH35B", (float(len(draws[b0:b0 + por_bloque])),), pico) for b0 in range(0, len(draws), por_bloque)]


def r2_from_blocks(cell: str, rb: dict, R: int, por_bloque: int):
    """(vals, cause, base) de las R replicas reunidas EN ORDEN (bloque 0, 1, ...) como en el bucle de E02."""
    parts = [rb[f"TH35B/{cell}/{b0:05d}"] for b0 in range(0, R, por_bloque)]
    vals = np.concatenate([x[0] for x in parts], axis=0)
    cause = np.concatenate([x[1] for x in parts], axis=0)
    base = np.concatenate([x[2] for x in parts], axis=0)
    if vals.shape[0] != R:
        _fail("E03_REPLICAS", f"{cell}: {vals.shape[0]} replicas reunidas, se esperaban {R}")
    return vals, cause, base


def _cell_draws(cfg: dict, celda: str, n_days: int, replicas: int | None = None) -> np.ndarray:
    """Sorteos de la celda en el PROCESO PRINCIPAL, con la llamada de E02."""
    return an.cell_draws(cfg, celda, n_days, bloque=cfg["analisis"]["bootstrap"]["longitud_bloque_jornadas"],
                         replicas=replicas)


def th35_pipeline_e03(df: pd.DataFrame, cfg: dict, motor: _Motor) -> dict:
    """Mismo resultado que ``an.th35_pipeline(df, cfg)`` (mismas claves, valores y orden)."""
    if ind.SYNTHETIC_MARKER in df.columns:
        raise ValueError("th35_pipeline recibe la entrada sin la columna de marco sintetico")
    df = df.reset_index(drop=True)
    table = causal_audit_e03(df, cfg, motor)
    ca.verify_causal_audit(table, df, cfg)                                  # barrera TH37 antes de cualquier TH35
    series, _marks = ind.compute_catalog(df)
    panel = ind.build_panel_p(series, df)
    ok = an.complete_cases(panel)
    cells = an.cell_definitions(df["timestamp"], cfg)
    names = list(panel.columns)
    ej = motor.ej
    stream = an.bootstrap_stream_cells(cfg)
    R = cfg["analisis"]["bootstrap"]["replicas"]
    p = len(names)
    tareas, draws = [], {}
    for i, cell in enumerate(stream):
        sel = cells[cell]["mascara"] & ok
        sub, sub_df = panel[sel], df[sel]
        n_days = int(np.unique(sub_df["trading_date"].to_numpy()).size)
        draws[cell] = _cell_draws(cfg, cell, n_days) if n_days >= 1 else None
        ref = ej.dato(f"celda_{i:03d}.pkl", (sub, sub_df))
        mom = str(ej.datos / f"momentos_{i:03d}.pkl")
        pico = 8.0 * sub.memory_usage().sum() / MB + 512
        tareas.append(Tarea(f"TH35C/{cell}", "_t_th35_cell", (ref, cfg, cell, draws[cell], mom), "TH35C",
                            (float(len(sub)) / 1e5,), pico))
        del sub, sub_df
    rc = motor.ejecutar("TH35_CELDAS", tareas)
    analyses = {cell: rc[f"TH35C/{cell}"]["analisis"] for cell in stream}
    # R2_TODAS por bloques de replicas (solo celdas con matriz definida, como cell_bootstrap); las comprobaciones de
    # bootstrap_r2 solo se aplican si alguna celda llega al bootstrap, como en E02
    if any(a["estado_matriz"] == "DEFINIDA" for a in analyses.values()):
        tau, q, reg = _r2_config(cfg)
        todo = [j for j, nm in enumerate(names) if nm not in reg]
    bloques = []
    for cell in stream:
        r = rc[f"TH35C/{cell}"]
        if analyses[cell]["estado_matriz"] == "DEFINIDA":
            bloques += r2_block_tasks(cell, r["momentos"], draws[cell], p, todo, tau, motor.r.replicas_por_bloque,
                                      r["jornadas"])
    rb = motor.ejecutar("TH35_BOOTSTRAP_R2", bloques)
    blocks, boots, rows = [], {}, []
    for cell in stream:
        a = analyses[cell]
        blocks += [f"{cell}: {b}" for b in a["bloqueos"]]
        if a["estado_matriz"] == "DEFINIDA":
            r = rc[f"TH35C/{cell}"]
            vals, cause, base = r2_from_blocks(cell, rb, R, motor.r.replicas_por_bloque)
            boots[cell] = {"jornadas": r["jornadas"],
                           "r2_todas": _r2_aggregate(vals, cause, base, names, a["r2_todas"], q, reg, R),
                           "spearman": r["spearman"]}
        if cells[cell]["tipo"] in an.PIPELINE_CELL_TYPES:
            rows += an._cell_rows(cell, a, boots.get(cell), cells[cell], names)
    per_cell = pd.DataFrame(rows)
    vcells = an.e02_value(cfg, "analisis.spearman.validacion_detalle")["celdas"]
    if all(analyses.get(c, {}).get("estado_matriz") == "DEFINIDA" for c in vcells):
        data = {c: (cells[c]["mascara"] & ok) for c in vcells}
        validation = spearman_validation_e03(panel, df, data, cfg, motor)
    else:
        validation = {"resultado": "NO_EJECUTABLE", "publicar_ic_spearman": False,
                      "motivo": f"celdas de validacion sin matriz definida: {vcells}"}
    spearman = {c: an.spearman_publication(b["spearman"], validation) for c, b in boots.items()}
    out = {"auditoria_causalidad": table, "analisis_celdas": analyses, "bootstrap": boots, "per_cell": per_cell,
           "validacion_spearman": validation, "spearman": spearman, "bloqueos": blocks,
           "poblacion": int(ok.sum())}
    if blocks:
        out["estado"] = "BLOQUEADO_HASTA_EXPLICAR"
        return out
    r2_global = analyses["GLOBAL"]["r2_todas"]
    out.update(estado="SIN_BLOQUEO", clasificacion=an.classify_columns(per_cell, cfg, blocks),
               subniveles=an.subniveles(r2_global, cfg, blocks))
    return out


# =========================================================================== validacion de Spearman (equivale a an.spearman_validation)
def spearman_validation_e03(panel: pd.DataFrame, df: pd.DataFrame, masks: dict[str, np.ndarray], cfg: dict,
                            motor: _Motor) -> dict:
    """Comprobaciones, rondas y agregacion LITERALES de ``an.spearman_validation``; las replicas de cada ronda en
    paralelo (``_t_th35_val_replica``). ``masks`` = {celda: mascara & casos completos} en el orden de E02."""
    vd = an.e02_value(cfg, "analisis.spearman.validacion_detalle")
    an.spearman_invalid_rule(cfg)
    val = cfg["analisis"]["spearman"]["validacion"]
    if vd.get("si_falla_la_ampliada") != "IC_SPEARMAN_NO_PUBLICADOS_COEFICIENTES_SI" or \
            cfg["analisis"]["spearman"]["ic_bootstrap"] != "RANGOS_DE_LA_CELDA_FIJOS":
        raise an.ProtocolNotAuthorizedError(f"validacion de Spearman distinta de la implementada: {vd}")
    if sorted(masks) != sorted(vd["celdas"]):
        raise an.ProtocolNotAuthorizedError(f"la validacion exige exactamente las celdas {vd['celdas']}")
    tol = val["tolerancia_mediana_dif_ic"]
    ej = motor.ej
    prep, refs = [], {}
    for i, (c, mask) in enumerate(masks.items()):
        X = panel[mask].to_numpy(np.float64)
        codes = df.loc[mask, "trading_date"].to_numpy()
        refs[c] = (ej.dato(f"val_X_{i:03d}.npy", X), ej.dato(f"val_codes_{i:03d}.pkl", codes))
        pico = 4.0 * X.nbytes / MB + 512
        prep.append(Tarea(f"TH35V/prep/{c}", "_t_th35_val_prep", (refs[c], str(ej.datos / f"val_rangos_{i:03d}.pkl")),
                          "TH35VP", (X.nbytes / 1e9,), pico))
        refs[c] = (*refs[c], pico, X.shape, int(np.unique(codes).size))
        del X, codes
    rp = motor.ejecutar("TH35_VALIDACION_PREPARACION", prep)
    hechas: dict[tuple, tuple] = {}
    draws_prev: dict[str, np.ndarray] = {}
    rounds = []
    for k in (val["replicas_exactas"], val["alternativa_si_falla"]["replicas_exactas"]):
        tareas, draws = [], {}
        for c in masks:
            ref_x, ref_codes, pico, shape, n_days = refs[c]
            if rp[f"TH35V/prep/{c}"]["jornadas"] != n_days:
                _fail("E03_SORTEOS", f"{c}: jornadas de los rangos distintas de las de los sorteos")
            draws[c] = _cell_draws(cfg, c, n_days, replicas=k)
            prev = draws_prev.get(c)
            for r, days in enumerate(draws[c]):
                if prev is not None and r < len(prev) and np.array_equal(prev[r], days) and (c, r) in hechas:
                    continue
                tareas.append(Tarea(f"TH35V/{k}/{c}/{r:05d}", "_t_th35_val_replica",
                                    ((ref_x, ref_codes), rp[f"TH35V/prep/{c}"]["m_rank"], days), "TH35VR",
                                    (shape[0] / 1e6,), 5.0 * shape[0] * shape[1] * 8 / MB + 512))
        rr = motor.ejecutar(f"TH35_VALIDACION_{k}", tareas)
        res = []
        for c in masks:
            for r in range(k):
                tid = f"TH35V/{k}/{c}/{r:05d}"
                if tid in rr:
                    hechas[(c, r)] = rr[tid]
            res.append(_validation_cell_aggregate(c, k, [hechas[(c, r)] for r in range(k)], cfg))
            draws_prev[c] = draws[c]
        ok = all(not r["replicas_invalidas"] and r["mediana_dif_ic"] <= tol for r in res)
        rounds.append({"replicas": k, "celdas": res, "pasa": ok})
        if ok:
            return {"resultado": "VALIDADA", "rondas": rounds, "publicar_ic_spearman": True}
    return {"resultado": "NO_VALIDADA", "rondas": rounds, "publicar_ic_spearman": False,
            "publicacion": vd["si_falla_la_ampliada"]}


def _validation_cell_aggregate(celda: str, k: int, replicas: list[tuple], cfg: dict) -> dict:
    """Parte posterior al bucle de ``an._validation_cell`` (literal), con las k replicas en orden."""
    q = an.interval_quantiles(cfg)
    npairs = replicas[0][1].size
    fixed, exact = np.empty((k, npairs)), np.empty((k, npairs))
    invalid = []
    for r, (zany, f, e) in enumerate(replicas):
        fixed[r], exact[r] = f, e
        if zany or not np.isfinite(fixed[r]).all() or not np.isfinite(exact[r]).all():
            invalid.append(r)
    out = {"celda": celda, "replicas": k, "pares": int(npairs), "replicas_invalidas": invalid}
    if invalid:
        return {**out, "mediana_dif_ic": None, "motivo": "REPLICA_INVALIDA"}
    lo_f, hi_f = np.quantile(fixed, list(q), axis=0, method="linear")
    lo_e, hi_e = np.quantile(exact, list(q), axis=0, method="linear")
    diffs = np.r_[np.abs(lo_f - lo_e), np.abs(hi_f - hi_e)]
    return {**out, "mediana_dif_ic": float(np.median(diffs))}


# =========================================================================== punto de entrada del motor
def identidad_ejecucion(df: pd.DataFrame, cfg: dict, recursos: Recursos, extra: dict | None = None) -> dict:
    """Identidad que liga los puntos de control: configuracion, marco, codigo del motor y entorno numerico. Los
    parametros de reparto (procesos, reserva) no cambian el resultado y no forman parte de ella, salvo los que
    definen las tareas (bloques)."""
    from ohlcv_dataroad.ingest import tda17_entradas as ent
    cfg_sha = _sha_bytes(json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
    return {"motor": "E03", "sha256_motor": _sha_file(Path(__file__)), "sha256_configuracion": cfg_sha,
            "huella_marco": ent.frame_fingerprint(df), "filas": int(len(df)),
            "cortes_por_bloque": recursos.cortes_por_bloque, "replicas_por_bloque": recursos.replicas_por_bloque,
            "hilos_blas_forzados": recursos.hilos_blas, "entorno": {k: v for k, v in entorno_numerico().items()
                                                            if k not in ("psutil", "nucleos_logicos")},
            **(extra or {})}


def run_th35_e03(df: pd.DataFrame, cfg: dict, directorio: Path | str, recursos: Recursos | None = None, *,
                 reanudar: bool = False, identidad_extra: dict | None = None, oficial: bool = False) -> tuple[dict, dict]:
    """Ejecuta TH37 -> TH35 con el motor E03 en ``directorio`` (fuera del repositorio). Devuelve (resultado igual al
    de ``an.th35_pipeline``, resumen del motor). Solo al terminar escribe ``COMPLETA.json``. ``oficial``: sin
    reanudacion y en un directorio nuevo (``Ejecucion``)."""
    recursos = recursos or Recursos()
    if oficial and (not recursos.memoria_estricta or recursos.procesos < 1 or recursos.reserva_mb <= 0):
        _fail("E03_RECURSOS_OFICIALES", "la ejecucion oficial exige control de memoria estricto, al menos 1 proceso "
                                        f"y reserva positiva: {recursos.resumen()}")
    ej = Ejecucion(directorio, identidad_ejecucion(df, cfg, recursos, identidad_extra), reanudar=reanudar,
                   oficial=oficial)
    motor = _Motor(recursos, ej)
    ej.evento(evento="inicio", recursos=recursos.resumen(), entorno=entorno_numerico(), reanudar=reanudar)
    try:
        result = th35_pipeline_e03(df, cfg, motor)
    except BaseException as exc:
        ej.evento(evento="fallo", error=f"{type(exc).__name__}: {exc}")
        ej.estado({"estado": "FALLO", "error": f"{type(exc).__name__}: {exc}", **motor.resumen()})
        raise
    finally:
        motor.cerrar()
    resumen = {**motor.resumen(), "duracion_total_s": round(time.time() - motor.t0, 3),
               "estado_resultado": result["estado"], "reanudada": bool(ej.reutilizadas)}
    ej.evento(evento="fin", **{k: v for k, v in resumen.items() if k != "recursos"})
    _write_atomic(ej.dir / "COMPLETA.json", _json(resumen))
    ej.estado({**resumen, "estado": "COMPLETA"})
    return result, resumen
