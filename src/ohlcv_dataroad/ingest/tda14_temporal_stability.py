"""TDA-14 -- Estabilidad temporal y cambio estructural (TH29, TH30, TH31).

Roadmap: ``docs/methodology/Tsay_OHLCV_analysis_roadmap.md`` § TDA-14.
Backlog: ``TH29`` (estabilidad por propiedad), ``TH30`` (log-precio frente
a retornos) y ``TH31`` (fechas de cambio conjunto).

ALCANCE DE ESTE MODULO (primera etapa de implementacion autorizada):
estimadores, mascaras, ventanas, controles sinteticos, detector, reglas de
clasificacion, calibracion G2, validacion y ``STOP-14``. Todo se ejercita
SOLO con mundos sinteticos. La carga de datos reales
(:func:`load_population`) existe, aplica la guardia de hold-out, pero NO
se ejecuta en esta etapa: no hay ``run_tda14`` y ninguna funcion de aqui
escribe artefactos en ``reports/``.

Es caracterizacion estadistica exploratoria y retrospectiva: no construye
targets, señales, features ni modelos; no interpreta un cambio estadistico
como un regimen economico; no atribuye efectos a eventos institucionales.

---------------------------------------------------------------------
HEREDADO (se reutiliza sin redefinir)
---------------------------------------------------------------------

- ``r_t`` = ``r_1m`` con ``r_1m_valid=True`` (TDA-04): un retorno que cruza
  un roll, una frontera de jornada o un hueco no existe aqui (``NaN``).
- Guardia de hold-out (``holdout_guard``) y carga de TDA-04 (TDA-07).
- Proteccion de memoria y numero de procesos (``memory_safe_workers``,
  ``available_memory_gb`` y ``default_n_workers`` de TDA-13), sin cambios.
- Años completos 2020-2024 (TDA-06); 2019 se excluye (6 jornadas).

---------------------------------------------------------------------
REPRESENTACION: PANEL DE JORNADAS
---------------------------------------------------------------------

Una fila por jornada CME (``trading_date``) y una columna por minuto de
SESION: ``(minuto_NY - 18:00) mod 1440``. La columna 0 es la apertura de
las 18:00 de Nueva York, asi el orden de columnas es el orden temporal
(un reloj 0..1439 partiria la sesion a medianoche). Mascaras por
estimador:

- media, desviacion, cuantiles, fraccion de ceros: retornos validos;
- ``rho1`` y ACF de ``|r|``: PARES de retornos validos dentro del MISMO
  bloque de continuidad (barras presentes consecutivas de una jornada):
  ningun par cruza un hueco, una jornada o un roll;
- perfil de reloj: retornos validos agregados en tramos de 30 minutos de
  sesion; un tramo sin apoyo minimo se excluye.

---------------------------------------------------------------------
DETECTOR (todo fijado antes de ver datos reales)
---------------------------------------------------------------------

Lotes de 7 jornadas; ventanas de 9 lotes (63 jornadas) con paso de 3
lotes (21 jornadas). Para cada estadistico (``DETECTOR_STATISTICS``) se
calcula su valor por ventana y un error tipico aproximado por lotes (la
dispersion de los 9 lotes / 3), resumido en un error COMUN por
estadistico (mediana de los cuadrados en todas las ventanas: el de una
sola ventana es demasiado ruidoso para ponderar). La escala absoluta de
ese error NO se interpreta: ignora la persistencia entre lotes, y los
umbrales salen de mundos negativos con persistencia de volatilidad, que
absorben su sesgo. Consecuencia de fondo: con volatilidad persistente la
escala varia mucho entre ventanas aunque el proceso no cambie, los
umbrales calibrados suben y la potencia para detectar cambios de escala
baja. Es una limitacion del problema, no un ajuste.

- Fechas (TH31): en cada frontera de la rejilla de paso 21 se compara la
  ventana anterior con la posterior. Un estadistico marca la frontera si
  ``|z| > c_fecha`` Y ``|diferencia| >= materialidad``; una familia la
  marca si la marca cualquiera de sus estadisticos. Hay fecha candidata
  cuando ``k*`` o mas familias la marcan A LA VEZ; las fronteras
  contiguas se funden en un intervalo. Los tres desfases de la rejilla de
  21 estan dentro del mismo procedimiento calibrado: no se cuentan como
  confirmaciones independientes.
- Veredictos (TH29): sobre las ventanas SIN solapamiento. Ver
  :func:`classify_series`.
- ``STOP-14``: ver :func:`decide_stop14`.

El volumen (familia ACTIVIDAD) se calcula y se informa, pero queda FUERA
del detector, de las fechas y de ``STOP-14`` (``VOLUME_STATUS``): no hay
control negativo que represente su comportamiento. No recibe veredicto
oficial.

---------------------------------------------------------------------
CALIBRACION G2 Y ACEPTACION
---------------------------------------------------------------------

Falsa alarma GLOBAL en un mundo negativo: alguna familia del detector con
veredicto ``TENDENCIA``, ``CAMBIO_ABRUPTO`` o ``INESTABLE``, o alguna
fecha candidata, o ``STOP-14`` activado con cualquier alcance. Un unico
nivel ``alpha`` fija todos los umbrales (``c_fecha`` y ``c_veredicto`` de
cada estadistico, cuantil ``1 - alpha`` de su distribucion en los mundos
negativos de CALIBRACION), y ``k*`` es el menor entero >= 2 para el que
existe un ``alpha`` cuya falsa alarma global en calibracion es <=
``G2_CALIBRATION_RATE`` (2,5 %). Despues se calibra el margen de tipo en
mundos positivos de calibracion. Los umbrales se congelan con SHA-256
(:class:`FrozenThresholds`) y la validacion (:func:`validate_procedure`)
solo acepta mundos con rol ``VALIDACION`` y umbrales congelados intactos:
no puede cambiar ningun umbral.

Aceptacion: el limite superior de confianza de Clopper-Pearson de la
tasa de falsas alarmas debe ser <= ``G2_TARGET_RATE`` (5 %), al nivel
``G2_CONFIDENCE`` (95 %) repartido por Bonferroni entre todos los
controles negativos evaluados. Calibrar al 2,5 % deja margen para poder
DEMOSTRAR <= 5 % con un numero finito de mundos. El resultado se llama
``COMPATIBLE_CON_CONTROLES_EVALUADOS``: dice que, en los mundos
negativos evaluados, la tasa es como mucho del 5 % con esa confianza; no
es una garantia universal para cualquier mundo real.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd
from scipy.special import bdtr, betaincinv     # solo scipy.special: scipy.stats/optimize/signal estan bloqueados por Smart App Control

from ohlcv_dataroad.config import SnapshotConfig
from ohlcv_dataroad.ingest.holdout_guard import (
    HoldoutIsolationError,
    validate_last_timestamps_before_boundary,
    validate_research_holdout_disjoint,
)
from ohlcv_dataroad.ingest.tda06_intraday_calendar_profile import COMPLETE_YEARS, attach_calendar_fields
from ohlcv_dataroad.ingest.tda07_marginal_distribution import load_tda04_inputs, verify_timestamp_alignment
from ohlcv_dataroad.ingest.tda13_extremes_inventory import (
    available_memory_gb,
    default_n_workers,
    memory_safe_workers,
    wilson_interval,
)

# ---------------------------------------------------------------------------
# Panel y calendario
# ---------------------------------------------------------------------------

PANEL_WIDTH = 1440
SESSION_OPEN_MINUTE_NY = 18 * 60
EXCLUDED_YEARS: tuple[int, ...] = (2019,)        # 6 jornadas: fuera de toda la etapa

# ---------------------------------------------------------------------------
# Familias y estadisticos
# ---------------------------------------------------------------------------

FAM_LEVEL, FAM_SCALE, FAM_SHAPE = "NIVEL", "ESCALA", "FORMA"
FAM_LINEAR, FAM_MAGNITUDE, FAM_CLOCK = "DEPENDENCIA_LINEAL", "DEPENDENCIA_MAGNITUD", "RELOJ"
FAM_DISCRETE, FAM_ACTIVITY = "DISCRECION", "ACTIVIDAD"
DETECTOR_FAMILIES: tuple[str, ...] = (FAM_LEVEL, FAM_SCALE, FAM_SHAPE, FAM_LINEAR, FAM_MAGNITUDE, FAM_CLOCK, FAM_DISCRETE)
assert FAM_ACTIVITY not in DETECTOR_FAMILIES

# estadistico -> (familia, materialidad primaria en sus unidades). Las materialidades son CONVENCIONES
# documentadas (no leyes estadisticas); su sensibilidad se informa con MATERIALITY_FACTORS.
DETECTOR_STATISTICS: dict[str, tuple[str, float]] = {
    "media_en_sigma": (FAM_LEVEL, 0.01),                      # media por barra en desviaciones tipicas
    "log_sigma": (FAM_SCALE, math.log(1.25)),                 # cociente de desviaciones 1,25
    "log_q01_en_sigma": (FAM_SHAPE, math.log(1.10)),          # |q01|/sigma, cociente 1,10 (convencion 0,10 de TDA-11)
    "log_q99_en_sigma": (FAM_SHAPE, math.log(1.10)),
    "rho1": (FAM_LINEAR, 0.03),
    "persistencia_acf_abs": (FAM_MAGNITUDE, 0.05),            # media de la ACF de |r|, retardos 1..60
    "log_amplitud_perfil": (FAM_CLOCK, math.log(1.25)),       # desviacion del log-perfil de tramos de 30 min
    "correlacion_perfil": (FAM_CLOCK, 0.10),                  # correlacion con el perfil de toda la muestra
    "log_fraccion_cero": (FAM_DISCRETE, math.log(1.25)),      # barras sin cambio de precio
}
# Calculados e informados, fuera del detector: volumen (sin control negativo) y curtosis (ver kurtosis_evaluability).
DESCRIPTIVE_STATISTICS: dict[str, str] = {"log_volumen_medio": FAM_ACTIVITY, "curtosis_total": FAM_SHAPE}
MATERIALITY_FACTORS: tuple[float, ...] = (0.5, 1.0, 2.0)
STRICT_MATERIALITY_FACTOR = 2.0                               # "robusto" = no estable tambien con el doble de materialidad
VOLUME_STATUS = "DESCRIPTIVO_NO_CALIBRADO"
VOLUME_VERDICT = "SIN_VEREDICTO_OFICIAL"

# Grupos de consecuencias de STOP-14 (que obligaria a hacer en TDA-15/16 la inestabilidad de cada grupo).
STOP_GROUPS: dict[str, tuple[str, ...]] = {
    "ESCALA": (FAM_SCALE, FAM_DISCRETE),
    "MEDIA": (FAM_LEVEL, FAM_LINEAR),
    "FORMA": (FAM_SHAPE,),
    "RELOJ": (FAM_CLOCK,),
    "DEPENDENCIA_MAGNITUD": (FAM_MAGNITUDE,),
}
STOP_CONSEQUENCES: dict[str, str] = {
    "ESCALA": "no mezclar periodos con retornos crudos; usar la serie estandarizada por escala causal",
    "MEDIA": "tratamiento de la media por subperiodo",
    "FORMA": "pruebas sobre la distribucion del residuo solo por subperiodo",
    "RELOJ": "perfil estacional estimado por subperiodo",
    "DEPENDENCIA_MAGNITUD": "filtro de volatilidad por subperiodo",
}
STOP_NOT_ACTIVE, STOP_SCALE, STOP_SPECIFIC, STOP_BROAD = "NO_ACTIVADO", "ALCANCE_ESCALA", "ALCANCE_ESPECIFICO", "ALCANCE_AMPLIO"
STOP_HUMAN_REVIEW = "NO_VALIDADO_REVISION_HUMANA"

# ---------------------------------------------------------------------------
# Veredictos
# ---------------------------------------------------------------------------

V_STABLE, V_TREND, V_JUMP, V_UNSTABLE = "ESTABLE", "TENDENCIA", "CAMBIO_ABRUPTO", "INESTABLE"
OFFICIAL_VERDICTS: tuple[str, ...] = (V_STABLE, V_TREND, V_JUMP, V_UNSTABLE)
V_NOT_EVALUABLE = "NO_EVALUABLE"                               # estado separado: falta de evidencia
NONSTABLE_VERDICTS: tuple[str, ...] = (V_TREND, V_JUMP, V_UNSTABLE)
A_EPISODE, A_NO_PATTERN, A_TYPE_UNKNOWN = "EPISODIO_TRANSITORIO", "SIN_PATRON", "TIPO_NO_IDENTIFICABLE"
A_NO_CHANGE, A_NOT_MATERIAL = "SIN_EVIDENCIA_DE_CAMBIO", "VARIACION_ESTADISTICA_NO_MATERIAL"
A_FEW_WINDOWS, A_WIDE_CI, A_LOW_POWER = "FALTA_DE_VENTANAS", "INCERTIDUMBRE_SUPERA_MATERIALIDAD", "POTENCIA_INSUFICIENTE"
A_EDGE = "CAMBIO_EN_BORDE_SIN_HISTORIA_SUFICIENTE"   # cambio en las primeras/ultimas ventanas: no se sabe si es permanente
TYPE_STEP, TYPE_TREND, TYPE_EPISODE, TYPE_EDGE = "SALTO", "TENDENCIA", "EPISODIO", "BORDE"
EXPECTED_TYPE: dict[str, tuple[str, str]] = {          # tipo inyectado -> (veredicto, anotacion) correctos
    TYPE_STEP: (V_JUMP, ""),
    TYPE_TREND: (V_TREND, ""),
    TYPE_EPISODE: (V_UNSTABLE, A_EPISODE),
    TYPE_EDGE: (V_UNSTABLE, A_EDGE),
}
# Soporte temporal de una inestabilidad (lo que STOP-14 necesita saber, ademas del veredicto):
SUP_PERMANENT = "PERMANENTE_CON_SOPORTE"          # el nivel nuevo se mantiene >= min_side ventanas hasta el final
SUP_NO_PATTERN = "SIN_PATRON_EN_LA_MUESTRA"       # variacion material que ningun modelo explica (no concentrada en un borde)
SUP_EDGE = "BORDE_SIN_SOPORTE_POSTERIOR"          # cambio en un borde: sin historia suficiente para saber si persiste
SUP_TRANSIENT = "TRANSITORIO"                     # episodio con retorno al nivel previo
SUP_UNPROVEN = "PERMANENCIA_NO_DEMOSTRADA"        # tipo no identificable entre alternativas que incluyen borde o episodio
SUP_NONE = "SIN_INESTABILIDAD"
PERSISTENT_SUPPORTS: tuple[str, ...] = (SUP_PERMANENT, SUP_NO_PATTERN)
SPECIFIC_TYPES: tuple[tuple[str, str], ...] = ((V_JUMP, ""), (V_TREND, ""), (V_UNSTABLE, A_EPISODE))   # tipos que el procedimiento AFIRMA

# ---------------------------------------------------------------------------
# Calibracion y aceptacion G2
# ---------------------------------------------------------------------------

G2_TARGET_RATE = 0.05            # tasa global maxima que hay que DEMOSTRAR (limite superior de confianza)
G2_CALIBRATION_RATE = 0.025      # objetivo en la mitad de calibracion (margen para poder demostrar el 5 %)
G2_CONFIDENCE = 0.95             # confianza conjunta, repartida por Bonferroni entre los controles negativos
TYPE_CONFUSION_LIMIT = 0.10      # confusion de tipos maxima a demostrar
TYPE_CALIBRATION_RATE = 0.05
# Capacidad minima de identificar tipos (aciertos / detectados, sobre el limite INFERIOR de confianza). PROPUESTA
# PROVISIONAL pendiente de aprobacion: sin ella, 0 errores con 100 % de abstenciones validaria los tipos.
TYPE_CAPACITY_MIN = 0.50
POWER_LEVEL_MATERIAL, POWER_LEVEL_DOUBLE = "MATERIALIDAD", "DOBLE"
POWER_REQUIRED: dict[str, float] = {POWER_LEVEL_MATERIAL: 0.50, POWER_LEVEL_DOUBLE: 0.80}   # sobre el limite INFERIOR de confianza
ALPHA_GRID: tuple[float, ...] = (0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.01, 0.0075, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0005)
MARGIN_GRID: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0)
STATUS_COMPATIBLE = "COMPATIBLE_CON_CONTROLES_EVALUADOS"
STATUS_NOT_VALIDATED = "NO_VALIDADO"
ROLE_CALIBRATION, ROLE_VALIDATION, ROLE_TEST, ROLE_BENCHMARK = "CALIBRACION", "VALIDACION", "PRUEBA", "BENCHMARK"
ROLE_CODES: dict[str, int] = {ROLE_CALIBRATION: 1, ROLE_VALIDATION: 2, ROLE_TEST: 3, ROLE_BENCHMARK: 4}
BASE_SEED = 14


class CalibrationRoleError(Exception):
    """Se intento calibrar con mundos de validacion (o validar con mundos de calibracion)."""


class FrozenThresholdsError(Exception):
    """Los umbrales congelados no estan completos o su SHA-256 no coincide: fueron modificados."""


class CalibrationFailure(Exception):
    """Ningun nivel de la rejilla cumple el objetivo de calibracion: el procedimiento no puede validarse."""


# ---------------------------------------------------------------------------
# Especificacion del detector
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DetectorSpec:
    batch_sessions: int = 7
    window_batches: int = 9                  # 63 jornadas
    step_batches: int = 3                    # 21 jornadas
    min_side_windows: int = 4                # ventanas minimas a cada lado de un cambio (y para estimar niveles)
    min_evaluable_windows: int = 16          # de unas 22 ventanas sin solapamiento
    min_session_bar_fraction: float = 0.90   # jornada completa: >= 90 % de las barras validas tipicas
    min_window_complete_fraction: float = 57 / 63
    min_window_bar_fraction: float = 0.90
    min_tail_support: int = 20               # TDA-12
    min_profile_bin_bars: int = 200          # apoyo minimo de un tramo de 30 min en una ventana
    max_lag: int = 60
    profile_bin_minutes: int = 30
    min_batches_for_se: int = 3

    def __post_init__(self):
        if self.window_batches % self.step_batches:
            raise ValueError("window_batches debe ser multiplo de step_batches (ventanas sin solapamiento dentro de la rejilla)")

    @property
    def window_sessions(self) -> int:
        return self.batch_sessions * self.window_batches

    @property
    def step_sessions(self) -> int:
        return self.batch_sessions * self.step_batches

    @property
    def offset(self) -> int:
        """Pasos de la rejilla entre una ventana y la siguiente sin solapamiento."""
        return self.window_batches // self.step_batches


# ---------------------------------------------------------------------------
# Panel de jornadas y mascaras
# ---------------------------------------------------------------------------

@dataclass
class SessionPanel:
    r: np.ndarray                     # (jornadas, minutos de sesion); NaN = sin retorno valido
    present: np.ndarray               # bool: hay barra
    session_date: np.ndarray          # (jornadas,) datetime64[D]
    contract: np.ndarray              # (jornadas,) contrato de la jornada
    volume: np.ndarray | None = None
    close: np.ndarray | None = None

    def __post_init__(self):
        if self.r.shape != self.present.shape:
            raise ValueError("r y present deben tener la misma forma")
        if (np.isfinite(self.r) & ~self.present).any():
            raise ValueError("hay retornos validos en minutos sin barra")
        if len(self.session_date) != self.r.shape[0] or len(self.contract) != self.r.shape[0]:
            raise ValueError("session_date y contract necesitan una entrada por jornada")

    @property
    def n_sessions(self) -> int:
        return self.r.shape[0]


def session_minute(minute_of_day) -> np.ndarray:
    """Minuto de sesion: 0 = 18:00 de Nueva York (apertura CME), en orden temporal dentro de la jornada."""
    return (np.asarray(minute_of_day, dtype=np.int64) - SESSION_OPEN_MINUTE_NY) % PANEL_WIDTH


def continuity_runs(valid: np.ndarray) -> np.ndarray:
    """Identificador de bloque de continuidad por celda (-1 fuera de bloque): RETORNOS VALIDOS consecutivos de una jornada.

    Definicion heredada de TDA-08/TDA-09 (``compute_block_ids_with_contract`` aplicado a la poblacion
    ``r_1m_valid=True``): un bloque se corta en cualquier minuto sin retorno valido, AUNQUE la barra exista (un retorno
    invalido de TDA-04, un dato marcado, la primera barra de la jornada o tras un hueco). La jornada y el contrato quedan
    cortados por construccion: cada fila del panel es una jornada con un solo contrato (``build_session_panel``).
    Correccion de auditoria: la version anterior formaba los bloques con barras PRESENTES y dejaba pasar pares de retardo
    > 1 que atravesaban un retorno invalido rodeado de barras presentes."""
    s, w = valid.shape
    prev = np.concatenate([np.zeros((s, 1), dtype=bool), valid[:, :-1]], axis=1)
    start = valid & ~prev
    rid = np.cumsum(start.ravel()).reshape(s, w)
    return np.where(valid, rid, -1)


def pair_mask(valid: np.ndarray, runs: np.ndarray, lag: int) -> np.ndarray:
    """Pares (t, t+lag) con ambos retornos validos y en el mismo bloque de retornos validos consecutivos: ningun par
    atraviesa un minuto sin retorno valido."""
    return valid[:, :-lag] & valid[:, lag:] & (runs[:, :-lag] == runs[:, lag:]) & (runs[:, :-lag] >= 0)


def assert_before_holdout(timestamps, boundary_utc) -> None:
    """Guardia a nivel de panel: ninguna barra puede alcanzar la frontera del hold-out."""
    ts = pd.to_datetime(pd.Series(timestamps))
    boundary = pd.Timestamp(boundary_utc)
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    if boundary.tzinfo is not None:
        boundary = boundary.tz_convert("UTC").tz_localize(None)
    if len(ts) and ts.max() >= boundary:
        raise HoldoutIsolationError(f"TDA-14: barra con timestamp {ts.max()} >= frontera del hold-out {boundary}; no se construye el panel.")


def build_session_panel(df: pd.DataFrame, holdout_boundary_utc) -> SessionPanel:
    """Panel desde las columnas de TDA-04 + calendario: ``timestamp, trading_date, contract, minute_of_day, r_1m, r_1m_valid`` (``volume`` y ``close`` opcionales)."""
    need = ("timestamp", "trading_date", "contract", "minute_of_day", "r_1m", "r_1m_valid")
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"faltan columnas: {missing}")
    assert_before_holdout(df["timestamp"], holdout_boundary_utc)
    td = pd.to_datetime(df["trading_date"]).dt.normalize()
    dates = np.sort(td.unique())
    row = np.searchsorted(dates, td.to_numpy())
    col = session_minute(df["minute_of_day"].to_numpy())
    flat = row * PANEL_WIDTH + col
    if len(np.unique(flat)) != len(flat):
        raise ValueError("dos barras en la misma jornada y minuto de sesion")
    per_session = pd.Series(df["contract"].to_numpy()).groupby(row).nunique()
    if (per_session > 1).any():
        raise ValueError("una jornada con mas de un contrato: TDA-04 certifica que los rolls caen en cambio de jornada")
    s = len(dates)
    present = np.zeros((s, PANEL_WIDTH), dtype=bool)
    present[row, col] = True
    r = np.full((s, PANEL_WIDTH), np.nan)
    valid = df["r_1m_valid"].to_numpy(dtype=bool) & np.isfinite(df["r_1m"].to_numpy(dtype=float))
    r[row[valid], col[valid]] = df["r_1m"].to_numpy(dtype=float)[valid]
    contract = pd.Series(df["contract"].to_numpy()).groupby(row).first().reindex(range(s)).to_numpy()
    out = {}
    for name in ("volume", "close"):
        if name in df.columns:
            m = np.full((s, PANEL_WIDTH), np.nan)
            m[row, col] = df[name].to_numpy(dtype=float)
            out[name] = m
    return SessionPanel(r=r, present=present, session_date=dates.astype("datetime64[D]"), contract=contract, **out)


def load_population(config: SnapshotConfig) -> SessionPanel:
    """Poblacion real de TDA-14 (NO se ejecuta en esta etapa). Guardias de hold-out ANTES y DESPUES de leer, igual que TDA-07..13."""
    validate_research_holdout_disjoint(config)
    variables, validity = load_tda04_inputs(config)
    validate_last_timestamps_before_boundary(config, variables.groupby("source_file")["timestamp"].max().to_dict())
    verify_timestamp_alignment(variables, validity)
    df = attach_calendar_fields(variables, validity)
    df = df.loc[~pd.to_datetime(df["trading_date"]).dt.year.isin(EXCLUDED_YEARS)]
    return build_session_panel(df, config.holdout_boundary_utc)


# ---------------------------------------------------------------------------
# Estadisticos suficientes por jornada (sumables entre jornadas)
# ---------------------------------------------------------------------------

def session_sufficient_stats(panel: SessionPanel, max_lag: int = 60, bin_minutes: int = 30) -> dict[str, np.ndarray]:
    """Sumas por jornada de las que salen, sumando jornadas, todos los estadisticos salvo los cuantiles."""
    r = panel.r
    valid = np.isfinite(r)
    runs = continuity_runs(valid)
    x = np.where(valid, r, 0.0)
    a = np.abs(x)
    s, w = r.shape
    out: dict[str, np.ndarray] = {
        "n": valid.sum(1).astype(float), "s1": x.sum(1), "s2": (x * x).sum(1), "s3": (x ** 3).sum(1), "s4": (x ** 4).sum(1),
        "n_zero": (valid & (r == 0)).sum(1).astype(float), "a1": a.sum(1), "a2": (a * a).sum(1),
        "present_n": panel.present.sum(1).astype(float),
    }
    pm = pair_mask(valid, runs, 1)
    out["r_np"] = pm.sum(1).astype(float)
    out["r_xy"] = (x[:, :-1] * x[:, 1:] * pm).sum(1)
    out["r_x"] = (x[:, :-1] * pm).sum(1)
    out["r_y"] = (x[:, 1:] * pm).sum(1)
    lag_n, lag_xy, lag_x, lag_y = (np.zeros((s, max_lag)) for _ in range(4))
    for lag in range(1, max_lag + 1):
        pm = pair_mask(valid, runs, lag)
        xa, ya = a[:, :-lag] * pm, a[:, lag:] * pm
        lag_n[:, lag - 1] = pm.sum(1)
        lag_xy[:, lag - 1] = (xa * ya).sum(1)
        lag_x[:, lag - 1] = xa.sum(1)
        lag_y[:, lag - 1] = ya.sum(1)
    out.update(lag_n=lag_n, lag_xy=lag_xy, lag_x=lag_x, lag_y=lag_y)
    nb = w // bin_minutes
    out["bin_a"] = a[:, : nb * bin_minutes].reshape(s, nb, bin_minutes).sum(2)
    out["bin_n"] = valid[:, : nb * bin_minutes].reshape(s, nb, bin_minutes).sum(2).astype(float)
    if panel.volume is not None:
        vf = np.isfinite(panel.volume) & panel.present
        out["vol_sum"] = np.where(vf, panel.volume, 0.0).sum(1)
        out["vol_n"] = vf.sum(1).astype(float)
    return out


def _sum_rows(suff: dict, rows) -> dict:
    return {k: v[rows].sum(0) for k, v in suff.items()}


def reference_log_profile(suff: dict, min_bin_bars: float) -> np.ndarray:
    """Log-perfil centrado de toda la muestra (tramos con apoyo suficiente; NaN en el resto)."""
    tot = _sum_rows(suff, slice(None))
    return _log_profile(tot["bin_a"], tot["bin_n"], min_bin_bars)


def _log_profile(bin_a, bin_n, min_bin_bars) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.log(bin_a / bin_n)
    lp = np.where((bin_n >= min_bin_bars) & np.isfinite(lp), lp, np.nan)
    if np.isfinite(lp).sum() < 2:
        return np.full_like(lp, np.nan, dtype=float)
    return lp - np.nanmean(lp)


def statistics_from_sums(sm: dict, q01: float, q99: float, ref_lp: np.ndarray, min_bin_bars: float) -> dict[str, float]:
    """Todos los estadisticos de un grupo de jornadas a partir de sus sumas (y de sus cuantiles, que no son sumables)."""
    nan = float("nan")
    n = sm["n"]
    out = dict.fromkeys(list(DETECTOR_STATISTICS) + list(DESCRIPTIVE_STATISTICS), nan)
    if n < 2:
        return out
    m = sm["s1"] / n
    var = sm["s2"] / n - m * m
    if not var > 0:
        return out
    sd = math.sqrt(var)
    out["media_en_sigma"] = m / sd
    out["log_sigma"] = math.log(sd)
    out["log_q01_en_sigma"] = math.log(-q01 / sd) if q01 < 0 else nan
    out["log_q99_en_sigma"] = math.log(q99 / sd) if q99 > 0 else nan
    m4 = sm["s4"] / n - 4 * m * sm["s3"] / n + 6 * m * m * sm["s2"] / n - 3 * m ** 4
    out["curtosis_total"] = m4 / var ** 2
    if sm["r_np"] > 0:
        cov = (sm["r_xy"] - m * (sm["r_x"] + sm["r_y"]) + sm["r_np"] * m * m) / sm["r_np"]
        out["rho1"] = cov / var
    am = sm["a1"] / n
    va = sm["a2"] / n - am * am
    ln = sm["lag_n"]
    if va > 0 and (ln > 0).all():
        cov = (sm["lag_xy"] - am * (sm["lag_x"] + sm["lag_y"]) + ln * am * am) / ln
        out["persistencia_acf_abs"] = float(np.mean(cov / va))
    lp = _log_profile(sm["bin_a"], sm["bin_n"], min_bin_bars)
    ok = np.isfinite(lp) & np.isfinite(ref_lp)
    if ok.sum() >= 3:
        sd_lp = float(np.std(lp[ok]))
        out["log_amplitud_perfil"] = math.log(sd_lp) if sd_lp > 0 else nan
        if sd_lp > 0 and np.std(ref_lp[ok]) > 0:
            out["correlacion_perfil"] = float(np.corrcoef(lp[ok], ref_lp[ok])[0, 1])
    out["log_fraccion_cero"] = math.log(sm["n_zero"] / n) if sm["n_zero"] > 0 else nan
    if "vol_sum" in sm and sm["vol_n"] > 0 and sm["vol_sum"] > 0:
        out["log_volumen_medio"] = math.log(sm["vol_sum"] / sm["vol_n"])
    return out


def _quantiles(panel: SessionPanel, rows) -> tuple[float, float, int]:
    v = panel.r[rows]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), 0
    q01, q99 = np.quantile(v, [0.01, 0.99])
    return float(q01), float(q99), int(v.size)


# ---------------------------------------------------------------------------
# Ventanas: valores y error tipico por lotes (sin umbrales)
# ---------------------------------------------------------------------------

@dataclass
class PanelStats:
    """Resultado del detector SIN umbrales: valores y errores por ventana de la rejilla de paso 21."""
    values: dict[str, np.ndarray]
    se: dict[str, np.ndarray]
    window_start: np.ndarray            # primera jornada (indice) de cada ventana
    evaluable: np.ndarray               # cobertura suficiente
    offset: int
    window_sessions: int
    n_sessions_used: int
    n_sessions_dropped: int
    session_date: np.ndarray | None = None
    descriptive: dict[str, np.ndarray] = field(default_factory=dict)
    se_by_window: dict[str, np.ndarray] = field(default_factory=dict)   # error por lotes de cada ventana (solo informativo)


def analyze_panel(panel: SessionPanel, spec: DetectorSpec = DetectorSpec()) -> PanelStats:
    suff = session_sufficient_stats(panel, spec.max_lag, spec.profile_bin_minutes)
    b = panel.n_sessions // spec.batch_sessions
    used = b * spec.batch_sessions
    if b < spec.window_batches:
        raise ValueError("el panel no tiene jornadas suficientes para una ventana")
    batch_bin_min = spec.min_profile_bin_bars / spec.window_batches
    ref_lp = reference_log_profile({k: v[:used] for k, v in suff.items()}, spec.min_profile_bin_bars)
    names = list(DETECTOR_STATISTICS) + list(DESCRIPTIVE_STATISTICS)
    batch_vals = {k: np.full(b, np.nan) for k in names}
    for j in range(b):
        rows = slice(j * spec.batch_sessions, (j + 1) * spec.batch_sessions)
        q01, q99, _ = _quantiles(panel, rows)
        st = statistics_from_sums(_sum_rows(suff, rows), q01, q99, ref_lp, batch_bin_min)
        for k in names:
            batch_vals[k][j] = st[k]
    n_valid = suff["n"][:used]
    typical = float(np.median(n_valid[n_valid > 0])) if (n_valid > 0).any() else 0.0
    complete = n_valid >= spec.min_session_bar_fraction * typical
    starts = np.arange(0, b - spec.window_batches + 1, spec.step_batches)
    values = {k: np.full(len(starts), np.nan) for k in names}
    se = {k: np.full(len(starts), np.nan) for k in names}
    evaluable = np.zeros(len(starts), dtype=bool)
    ws = spec.window_sessions
    for i, wb in enumerate(starts):
        s0 = wb * spec.batch_sessions
        rows = slice(s0, s0 + ws)
        cov_ok = complete[rows].mean() >= spec.min_window_complete_fraction - 1e-12 and n_valid[rows].sum() >= spec.min_window_bar_fraction * ws * typical
        q01, q99, nv = _quantiles(panel, rows)
        tails_ok = nv * 0.01 >= spec.min_tail_support
        evaluable[i] = bool(cov_ok and tails_ok)
        if not evaluable[i]:
            continue
        st = statistics_from_sums(_sum_rows(suff, rows), q01, q99, ref_lp, spec.min_profile_bin_bars)
        for k in names:
            values[k][i] = st[k]
            bv = batch_vals[k][wb: wb + spec.window_batches]
            bv = bv[np.isfinite(bv)]
            if bv.size >= spec.min_batches_for_se:
                se[k][i] = float(np.std(bv, ddof=1) / math.sqrt(bv.size))
    descriptive = {k: values.pop(k) for k in DESCRIPTIVE_STATISTICS}
    for k in DESCRIPTIVE_STATISTICS:
        se.pop(k)
    # Error comun por estadistico: con 9 lotes por ventana el error de CADA ventana es demasiado ruidoso para ponderar
    # (varia un orden de magnitud entre ventanas del mismo mundo). Se usa la mediana de sus cuadrados en todas las
    # ventanas (todas tienen el mismo numero de jornadas); los errores por ventana se conservan para informarlos.
    se_common = {}
    for k, s in se.items():
        pooled = math.sqrt(np.nanmedian(s ** 2)) if np.isfinite(s).any() else float("nan")
        se_common[k] = np.where(np.isfinite(values[k]), pooled, np.nan)
    return PanelStats(values=values, se=se_common, window_start=starts * spec.batch_sessions, evaluable=evaluable, offset=spec.offset,
                      window_sessions=ws, n_sessions_used=used, n_sessions_dropped=panel.n_sessions - used,
                      session_date=panel.session_date, descriptive=descriptive, se_by_window=se)


def volume_descriptive(ps: PanelStats) -> pd.DataFrame:
    """Volumen por ventana: obligatorio pero NO calibrado y sin veredicto oficial de estabilidad."""
    v = ps.descriptive.get("log_volumen_medio")
    if v is None:
        v = np.full(len(ps.window_start), np.nan)
    med = np.nanmedian(v) if np.isfinite(v).any() else np.nan
    return pd.DataFrame({
        "ventana_inicio": ps.window_start, "log_volumen_medio": v, "cociente_frente_a_mediana": np.exp(v - med),
        "estado": VOLUME_STATUS, "veredicto_oficial": VOLUME_VERDICT,
    })


def boundary_series(ps: PanelStats, stat: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diferencia y z entre la ventana anterior y la posterior de cada frontera (jornada de la frontera)."""
    v, se, k = ps.values[stat], ps.se[stat], ps.offset
    if len(v) <= k:
        return np.array([]), np.array([]), np.array([], dtype=int)
    d = v[k:] - v[:-k]
    with np.errstate(invalid="ignore", divide="ignore"):
        z = d / np.hypot(se[k:], se[:-k])
    return d, z, ps.window_start[k:]


def nonoverlapping(ps: PanelStats, stat: str) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(0, len(ps.window_start), ps.offset)
    return ps.values[stat][idx], ps.se[stat][idx]


def constancy_statistic(v: np.ndarray, se: np.ndarray) -> float:
    """``D``: suma de desviaciones cuadraticas ponderadas respecto de un nivel constante, por grado de libertad."""
    ok = np.isfinite(v) & np.isfinite(se) & (se > 0)
    if ok.sum() < 2:
        return float("nan")
    y, w = v[ok], 1.0 / se[ok] ** 2
    mu = np.sum(w * y) / np.sum(w)
    return float(np.sum(w * (y - mu) ** 2) / (ok.sum() - 1))


def max_abs_z(ps: PanelStats, stat: str) -> float:
    _, z, _ = boundary_series(ps, stat)
    z = np.abs(z[np.isfinite(z)])
    return float(z.max()) if z.size else float("nan")


# ---------------------------------------------------------------------------
# Clasificacion de una serie de ventanas (TH29)
# ---------------------------------------------------------------------------

def _wfit_segments(y, w, cuts) -> tuple[float, np.ndarray, list[float]]:
    fitted = np.empty_like(y)
    levels = []
    for a, b in zip((0,) + tuple(cuts), tuple(cuts) + (len(y),)):
        mu = float(np.sum(w[a:b] * y[a:b]) / np.sum(w[a:b]))
        fitted[a:b] = mu
        levels.append(mu)
    return float(np.sum(w * (y - fitted) ** 2)), fitted, levels


def _wfit_trend(y, w) -> tuple[float, np.ndarray, float]:
    t = np.arange(len(y), dtype=float)
    sw = w.sum()
    tm, ym = np.sum(w * t) / sw, np.sum(w * y) / sw
    den = np.sum(w * (t - tm) ** 2)
    slope = float(np.sum(w * (t - tm) * (y - ym)) / den) if den > 0 else 0.0
    fitted = ym + slope * (t - tm)
    return float(np.sum(w * (y - fitted) ** 2)), fitted, slope


def fit_change_models(y: np.ndarray, se: np.ndarray, min_side: int, materiality: float = 0.0) -> list[dict]:
    """Ajustes ponderados: salto permanente, salto en un borde, tendencia y tres niveles (dos cambios).

    Criterio de Schwarz con varianza desconocida, ``n log(RSS/n) + k log n``: no depende de la escala absoluta del error
    (que ignora la persistencia). Un modelo de tres niveles solo es candidato si sus DOS cambios son materiales: un cambio
    menor que la materialidad no se cuenta como cambio."""
    n = len(y)
    w = 1.0 / se ** 2
    ln = math.log(n)
    fits = []
    best_step = best_edge = None
    for cp in range(1, n):
        rss, fitted, lv = _wfit_segments(y, w, (cp,))
        cand = {"modelo": TYPE_STEP if min_side <= cp <= n - min_side else TYPE_EDGE, "rss": rss, "k": 3, "fitted": fitted, "niveles": lv, "cortes": (cp,)}
        if cand["modelo"] == TYPE_STEP and (best_step is None or rss < best_step["rss"]):
            best_step = cand
        if cand["modelo"] == TYPE_EDGE and (best_edge is None or rss < best_edge["rss"]):
            best_edge = cand
    fits += [f for f in (best_step, best_edge) if f is not None]
    rss, fitted, slope = _wfit_trend(y, w)
    fits.append({"modelo": TYPE_TREND, "rss": rss, "k": 2, "fitted": fitted, "pendiente": slope})
    best3 = None
    for c1 in range(min_side, n - min_side):
        for c2 in range(c1 + 1, n - min_side + 1):
            rss, fitted, lv = _wfit_segments(y, w, (c1, c2))
            if abs(lv[1] - lv[0]) < materiality or abs(lv[2] - lv[1]) < materiality:
                continue
            if best3 is None or rss < best3["rss"]:
                best3 = {"modelo": TYPE_EPISODE, "rss": rss, "k": 5, "fitted": fitted, "niveles": lv, "cortes": (c1, c2)}
    if best3 is not None:
        fits.append(best3)
    for f in fits:
        f["criterio"] = n * math.log(max(f["rss"], 1e-300) / n) + f["k"] * ln
    return sorted(fits, key=lambda f: f["criterio"])


def observed_variation(y: np.ndarray, se: np.ndarray) -> float:
    """Variacion observada entre ventanas MAS ALLA DEL RUIDO: ``2 * sqrt(max(var(y) - media(se^2), 0))``.
    El factor 2 hace que, para un salto con la mitad de las ventanas a cada lado, valga exactamente el tamaño del salto;
    para una alternancia entre dos niveles vale la distancia entre ellos. No usa ningun umbral nuevo."""
    if len(y) < 2:
        return float("nan")
    return float(2.0 * math.sqrt(max(np.var(y, ddof=1) - np.mean(se ** 2), 0.0)))


def classify_series(v: np.ndarray, se: np.ndarray, materiality: float, c_verdict: float, type_margin: float,
                    min_side: int = 4, min_evaluable: int = 16) -> dict:
    """Veredicto oficial de UN estadistico sobre sus ventanas sin solapamiento.

    1. Menos de ``min_evaluable`` ventanas evaluables, o un intervalo tipico (1,96 errores) mas ancho que la materialidad:
       ``NO_EVALUABLE`` (falta de evidencia; estado separado de los cuatro veredictos).
    2. ``D <= c_verdict``: ``ESTABLE`` (sin evidencia de cambio).
    3. Se ajustan salto permanente (al menos ``min_side`` ventanas a cada lado, hasta el final), salto en un borde, tendencia y
       tres niveles con dos cambios materiales; gana el menor criterio de Schwarz (:func:`fit_change_models`). Solo si
       TANTO el recorrido del ajuste ganador COMO la variacion observada mas alla del ruido (:func:`observed_variation`)
       son menores que la materialidad: ``ESTABLE`` (variacion estadistica no material).
    4. Si el ganador deja todavia una inestabilidad residual (``RSS/(n-k) > c_verdict``, el mismo umbral calibrado), o si
       la variacion observada es material pero la curva ajustada no lo es: ``INESTABLE`` sin patron.
    5. Tipo: salto -> ``CAMBIO_ABRUPTO``; tendencia -> ``TENDENCIA``; tres niveles con niveles anterior y posterior
       NO distintos materialmente y tramo intermedio distinto de ambos -> ``INESTABLE`` episodio transitorio; tres niveles
       sin retorno al nivel previo -> ``INESTABLE`` tipo no identificable (con soporte permanente); salto en las primeras
       o ultimas ``min_side`` ventanas -> ``INESTABLE`` ``CAMBIO_EN_BORDE_SIN_HISTORIA_SUFICIENTE`` (no se sabe si persiste).
    6. Si algun modelo de OTRO tipo esta a menos de ``type_margin`` del ganador: ``INESTABLE`` tipo no identificable
       (nunca se elige un tipo arbitrariamente); su soporte es permanente solo si todas las alternativas cercanas lo son.
    Cada resultado lleva ``soporte`` (:data:`SUP_PERMANENT`, :data:`SUP_NO_PATTERN`, :data:`SUP_EDGE`,
    :data:`SUP_TRANSIENT`, :data:`SUP_UNPROVEN`), que es lo que usa STOP-14.
    """
    ok = np.isfinite(v) & np.isfinite(se) & (se > 0)
    base = {"veredicto": V_NOT_EVALUABLE, "anotacion": A_FEW_WINDOWS, "D": float("nan"), "modelo": "", "recorrido": float("nan"), "n_ventanas": int(ok.sum())}
    if ok.sum() < min_evaluable:
        return base
    y, s = v[ok], se[ok]
    if np.median(1.96 * s) > materiality:
        return {**base, "anotacion": A_WIDE_CI}
    d = constancy_statistic(y, s)
    base["D"] = d
    if not d > c_verdict:
        return {**base, "veredicto": V_STABLE, "anotacion": A_NO_CHANGE}
    fits = fit_change_models(y, s, min_side, materiality)
    best = fits[0]
    span = float(np.max(best["fitted"]) - np.min(best["fitted"]))
    observed = observed_variation(y, s)
    residual = best["rss"] / max(len(y) - best["k"], 1)
    base.update(modelo=best["modelo"], recorrido=span, variacion_observada=observed, residuo_por_gl=residual,
                cortes=best.get("cortes"), niveles=best.get("niveles"))
    # Correccion de auditoria: la materialidad no se juzga solo por el recorrido de la curva ajustada (una oscilacion
    # grande que ningun modelo capta tiene curva plana). Cuenta tambien la variacion OBSERVADA mas alla del ruido.
    if max(span, observed) < materiality:
        return {**base, "veredicto": V_STABLE, "anotacion": A_NOT_MATERIAL, "soporte": SUP_NONE}
    if residual > c_verdict or span < materiality:
        return {**base, "veredicto": V_UNSTABLE, "anotacion": A_NO_PATTERN, "soporte": SUP_NO_PATTERN}
    if best["modelo"] == TYPE_STEP:
        verdict, ann, sup = V_JUMP, "", SUP_PERMANENT
    elif best["modelo"] == TYPE_TREND:
        verdict, ann, sup = V_TREND, "", SUP_PERMANENT
    elif best["modelo"] == TYPE_EPISODE:
        a1, a2, a3 = best["niveles"]
        is_episode = abs(a3 - a1) < materiality and abs(a2 - a1) >= materiality and abs(a2 - a3) >= materiality
        # tres niveles que NO vuelven al previo: el ultimo nivel difiere materialmente del primero y dura >= min_side ventanas
        verdict, ann, sup = (V_UNSTABLE, A_EPISODE, SUP_TRANSIENT) if is_episode else (V_UNSTABLE, A_TYPE_UNKNOWN, SUP_PERMANENT)
        if is_episode:
            c1, c2 = best["cortes"]
            base["episodio_ventanas"] = c2 - c1
    else:
        # Salto en un borde (correccion de auditoria): anotacion propia, nunca 'tipo no identificable' generico.
        cp = best["cortes"][0]
        verdict, ann, sup = V_UNSTABLE, A_EDGE, SUP_EDGE
        base["borde"] = "INICIO" if cp < min_side else "FIN"
        base["ventanas_tras_el_cambio"] = len(y) - cp
    close = [f["modelo"] for f in fits[1:] if f["modelo"] != best["modelo"] and f["criterio"] - best["criterio"] < type_margin]
    if ann in (A_EPISODE, "") and close:
        # Tipo no identificable. Solo conserva soporte permanente si TODAS las alternativas cercanas son permanentes;
        # si una es un borde o un episodio, la permanencia no esta demostrada.
        verdict, ann = V_UNSTABLE, A_TYPE_UNKNOWN
        sup = SUP_UNPROVEN if ({best["modelo"], *close} & {TYPE_EDGE, TYPE_EPISODE}) else SUP_PERMANENT
        base["tipos_en_duda"] = sorted({best["modelo"], *close})
    return {**base, "veredicto": verdict, "anotacion": ann, "soporte": sup}


def support_of(v: dict) -> str:
    """Soporte temporal de un veredicto. Si falta el campo (veredicto construido fuera de :func:`classify_series`), se
    deduce de forma CONSERVADORA: 'tipo no identificable' sin soporte declarado NO cuenta como permanente."""
    if "soporte" in v:
        return v["soporte"]
    if v["veredicto"] in (V_JUMP, V_TREND):
        return SUP_PERMANENT
    if v["veredicto"] != V_UNSTABLE:
        return SUP_NONE
    return {A_NO_PATTERN: SUP_NO_PATTERN, A_EPISODE: SUP_TRANSIENT, A_EDGE: SUP_EDGE}.get(v["anotacion"], SUP_UNPROVEN)


def combine_family(results: list[dict]) -> dict:
    """Veredicto de una familia: no estable si lo es algun estadistico; si sus estadisticos discrepan en el tipo, tipo no identificable."""
    nonstable = [r for r in results if r["veredicto"] in NONSTABLE_VERDICTS]
    if nonstable:
        kinds = {(r["veredicto"], r["anotacion"]) for r in nonstable}
        if len(kinds) == 1:
            verdict, ann = kinds.pop()
        else:
            verdict, ann = V_UNSTABLE, A_TYPE_UNKNOWN
        # Soporte de la familia: el mas fuerte que DEMUESTRE alguno de sus estadisticos (uno con cambio permanente
        # sostenido basta para que la familia lo tenga); un borde solo cuenta como borde.
        sups = {support_of(r) for r in nonstable}
        sup = next(s for s in (SUP_PERMANENT, SUP_NO_PATTERN, SUP_UNPROVEN, SUP_EDGE, SUP_TRANSIENT) if s in sups)
        return {"veredicto": verdict, "anotacion": ann, "soporte": sup}
    if any(r["veredicto"] == V_NOT_EVALUABLE for r in results):
        return {"veredicto": V_NOT_EVALUABLE, "anotacion": next(r["anotacion"] for r in results if r["veredicto"] == V_NOT_EVALUABLE)}
    return {"veredicto": V_STABLE, "anotacion": results[0]["anotacion"] if results else A_NO_CHANGE}


# ---------------------------------------------------------------------------
# STOP-14
# ---------------------------------------------------------------------------

PS_CONFIRMED = "PERSISTENTE_CONFIRMADA"                    # ventanas + anual explicito + sensibilidad explicita
PS_PENDING = "CANDIDATA_PENDIENTE_DE_COMPROBACIONES"         # ventanas si; falta el analisis anual o el de sensibilidad
PS_CONTRADICTORY = "CONTRADICTORIA"                          # ventanas y analisis anual discrepan
PS_EXPLAINED = "EXPLICADA_POR_ROLLS_O_HUECOS"                # comprobacion explicita: la explican las sensibilidades
PS_NONE = "NO_PERSISTENTE"                                   # estable, no evaluable, episodio o no robusta
PS_EDGE = "CAMBIO_EN_BORDE_SIN_SOPORTE_POSTERIOR"            # cambio al principio/final: nunca persistente automaticamente
PS_UNPROVEN = "PERMANENCIA_NO_DEMOSTRADA"                    # tipo no identificable entre alternativas con borde o episodio
STOP_PENDING = "PENDIENTE_DE_COMPROBACIONES_EXPLICITAS"
STOP_CONTRADICTORY = "RESULTADOS_CONTRADICTORIOS_REVISION_HUMANA"


def persistence_status(primary: dict, strict: dict, annual_confirmed: bool | None = None, explained_by_sensitivity: bool | None = None) -> str:
    """Estado de persistencia de UNA familia para STOP-14.

    Candidata en ventanas: tendencia, salto permanente o inestable sin patron/tipo no identificable, robusta al doble de
    materialidad (un episodio transitorio NO es persistente: marca su periodo). Para ser ``PERSISTENTE_CONFIRMADA`` hacen
    falta las DOS comprobaciones explicitas: ``annual_confirmed is True`` (el analisis anual muestra la inestabilidad) y
    ``explained_by_sensitivity is False`` (las sensibilidades de rolls y huecos NO la explican). ``None`` significa "no
    comprobado" y NUNCA cuenta como comprobacion superada (correccion de auditoria): deja la familia pendiente.
    Contradicciones (no se resuelven solas): ventanas candidatas con anual ``False``, o ventanas ``ESTABLE`` con anual ``True``.

    Soporte temporal (correccion de auditoria): solo un cambio PERMANENTE CON SOPORTE (el nivel nuevo dura al menos
    ``min_side`` ventanas hasta el final) o una inestabilidad material SIN PATRON en la muestra pueden ser persistentes,
    con ambos niveles de materialidad. Un cambio en el borde (``CAMBIO_EN_BORDE_SIN_SOPORTE_POSTERIOR``) o un tipo no
    identificable cuyas alternativas incluyen borde o episodio (``PERMANENCIA_NO_DEMOSTRADA``) nunca activan STOP-14
    automaticamente, aunque el analisis anual y las sensibilidades sean favorables.
    """
    sup, sup_strict = support_of(primary), support_of(strict)
    if primary["veredicto"] in NONSTABLE_VERDICTS and sup == SUP_EDGE:
        return PS_EDGE
    if primary["veredicto"] in NONSTABLE_VERDICTS and sup == SUP_UNPROVEN:
        return PS_UNPROVEN
    windows_candidate = (primary["veredicto"] in NONSTABLE_VERDICTS and sup in PERSISTENT_SUPPORTS
                         and strict["veredicto"] in NONSTABLE_VERDICTS and sup_strict in PERSISTENT_SUPPORTS)
    if not windows_candidate:
        return PS_CONTRADICTORY if (primary["veredicto"] == V_STABLE and annual_confirmed is True) else PS_NONE
    if explained_by_sensitivity is True:
        return PS_EXPLAINED
    if annual_confirmed is False:
        return PS_CONTRADICTORY
    if annual_confirmed is None or explained_by_sensitivity is None:
        return PS_PENDING
    return PS_CONFIRMED


def _scope_of(families: list[str]) -> tuple[str, list[str]]:
    groups = sorted(g for g, fams in STOP_GROUPS.items() if any(f in families for f in fams))
    non_scale = [g for g in groups if g != "ESCALA"]
    if len(non_scale) >= 2:
        return STOP_BROAD, groups
    if len(non_scale) == 1:
        return STOP_SPECIFIC, groups
    return (STOP_SCALE if groups else STOP_NOT_ACTIVE), groups


def decide_stop14(primary: dict[str, dict], strict: dict[str, dict], annual_confirmed: dict | None = None,
                  explained_by_sensitivity: dict | None = None, procedure_validated: bool = True, scope_validated: bool = True) -> dict:
    """Alcance de STOP-14. Una propiedad inestable restringe solo lo que depende de ella; el alcance amplio exige dos o
    mas grupos centrales distintos de la escala. El volumen nunca interviene.

    Activacion DEFINITIVA solo con familias ``PERSISTENTE_CONFIRMADA``. Si las familias pendientes o contradictorias
    cambiarian el alcance, no hay decision definitiva: ``PENDIENTE_DE_COMPROBACIONES_EXPLICITAS`` (falta informacion) o
    ``RESULTADOS_CONTRADICTORIOS_REVISION_HUMANA`` (hay contradicciones), con el alcance provisional que resultaria."""
    annual_confirmed = annual_confirmed or {}
    explained_by_sensitivity = explained_by_sensitivity or {}
    status = {f: persistence_status(primary[f], strict.get(f, primary[f]), annual_confirmed.get(f), explained_by_sensitivity.get(f))
              for f in DETECTOR_FAMILIES if f in primary}
    confirmed = sorted(f for f, s in status.items() if s == PS_CONFIRMED)
    pending = sorted(f for f, s in status.items() if s == PS_PENDING)
    contradictory = sorted(f for f, s in status.items() if s == PS_CONTRADICTORY)
    episodes = sorted(f for f in DETECTOR_FAMILIES if f in primary and primary[f]["veredicto"] == V_UNSTABLE and primary[f]["anotacion"] == A_EPISODE)
    scope, groups = _scope_of(confirmed)
    provisional, prov_groups = _scope_of(confirmed + pending + contradictory)
    definitive = provisional == scope and prov_groups == groups
    if not definitive:
        scope = STOP_CONTRADICTORY if contradictory else STOP_PENDING
    elif scope != STOP_NOT_ACTIVE and (not procedure_validated or (scope in (STOP_SPECIFIC, STOP_BROAD) and not scope_validated)):
        scope = STOP_HUMAN_REVIEW
    return {"alcance": scope, "definitivo": definitive and scope not in (STOP_HUMAN_REVIEW,), "alcance_provisional": provisional,
            "grupos": groups, "grupos_provisionales": prov_groups, "estado_por_familia": status,
            "familias_persistentes": confirmed, "familias_pendientes": pending, "familias_contradictorias": contradictory,
            "cambios_en_borde": sorted(f for f, s in status.items() if s == PS_EDGE),
            "permanencia_no_demostrada": sorted(f for f, s in status.items() if s == PS_UNPROVEN),
            "episodios": episodes, "consecuencias": [STOP_CONSEQUENCES[g] for g in groups],
            "nivel5": "por subperiodo o no se ejecuta" if scope == STOP_BROAD else ("solo las consecuencias listadas" if groups else "sin restriccion definitiva de TDA-14")}


# ---------------------------------------------------------------------------
# Umbrales congelados
# ---------------------------------------------------------------------------

def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=lambda o: repr(o))


@dataclass(frozen=True)
class FrozenThresholds:
    alpha: float
    k_star: int
    c_date: tuple[tuple[str, float], ...]
    c_verdict: tuple[tuple[str, float], ...]
    type_margin: float | None
    materiality: tuple[tuple[str, float], ...]
    detector_spec: tuple[tuple[str, float], ...]
    calibration_info: tuple[tuple[str, float], ...]
    sha256: str = ""

    def payload(self) -> dict:
        d = asdict(self)
        d.pop("sha256")
        return d

    def c_date_of(self, stat: str) -> float:
        return dict(self.c_date)[stat]

    def c_verdict_of(self, stat: str) -> float:
        return dict(self.c_verdict)[stat]

    def materiality_of(self, stat: str) -> float:
        return dict(self.materiality)[stat]


def freeze_thresholds(thr: FrozenThresholds) -> FrozenThresholds:
    unsigned = replace(thr, sha256="")
    return replace(unsigned, sha256=hashlib.sha256(_canonical(unsigned.payload()).encode()).hexdigest())


def verify_frozen(thr: FrozenThresholds, require_complete: bool = True) -> None:
    if hashlib.sha256(_canonical(thr.payload()).encode()).hexdigest() != thr.sha256:
        raise FrozenThresholdsError("el SHA-256 de los umbrales no coincide: se modificaron despues de congelarlos")
    if require_complete and thr.type_margin is None:
        raise FrozenThresholdsError("falta el margen de tipo: la calibracion no esta completa")


def _spec_from(thr: FrozenThresholds) -> DetectorSpec:
    d = dict(thr.detector_spec)
    return DetectorSpec(**{k: (int(v) if isinstance(DetectorSpec.__dataclass_fields__[k].default, int) else float(v)) for k, v in d.items()})


# ---------------------------------------------------------------------------
# Regla completa aplicada a un panel
# ---------------------------------------------------------------------------

def _thresholds_dicts(thr: FrozenThresholds):
    return dict(thr.c_date), dict(thr.c_verdict), dict(thr.materiality)


def apply_rule(ps: PanelStats, thr: FrozenThresholds, materiality_factor: float = 1.0, type_margin: float | None = None,
               power_ok: dict[str, bool] | None = None) -> dict:
    """Decisiones del procedimiento completo: veredictos por familia (y por estadistico), fechas candidatas y STOP-14."""
    spec = _spec_from(thr)
    c_date, c_verd, mat = _thresholds_dicts(thr)
    margin = thr.type_margin if type_margin is None else type_margin
    margin = 0.0 if margin is None else margin

    def verdicts(factor):
        by_stat = {}
        for st in DETECTOR_STATISTICS:
            v, se = nonoverlapping(ps, st)
            by_stat[st] = classify_series(v, se, mat[st] * factor, c_verd[st], margin, spec.min_side_windows, spec.min_evaluable_windows)
        by_fam = {f: combine_family([by_stat[s] for s, (fam, _) in DETECTOR_STATISTICS.items() if fam == f]) for f in DETECTOR_FAMILIES}
        return by_stat, by_fam

    stat_v, fam_v = verdicts(materiality_factor)
    _, fam_strict = verdicts(materiality_factor * STRICT_MATERIALITY_FACTOR)
    if power_ok is not None:
        for f, ok in power_ok.items():
            if not ok and fam_v.get(f, {}).get("veredicto") == V_STABLE:
                fam_v[f] = {"veredicto": V_NOT_EVALUABLE, "anotacion": A_LOW_POWER}

    n_b = max(len(ps.window_start) - ps.offset, 0)
    fam_flags = {f: np.zeros(n_b, dtype=bool) for f in DETECTOR_FAMILIES}
    bsess = ps.window_start[ps.offset:]
    for st, (fam, _) in DETECTOR_STATISTICS.items():
        d, z, _ = boundary_series(ps, st)
        if d.size:
            fam_flags[fam] |= np.isfinite(z) & (np.abs(z) > c_date[st]) & (np.abs(d) >= mat[st] * materiality_factor)
    count = np.sum([fam_flags[f] for f in DETECTOR_FAMILIES], axis=0) if n_b else np.zeros(0, dtype=int)
    hits = np.flatnonzero(count >= thr.k_star)
    dates = []
    for h in hits:
        if dates and h == dates[-1]["ultimo_indice"] + 1:
            dates[-1].update(ultimo_indice=int(h), jornada_fin=int(bsess[h]), familias_max=max(dates[-1]["familias_max"], int(count[h])))
        else:
            dates.append({"primer_indice": int(h), "ultimo_indice": int(h), "jornada_inicio": int(bsess[h]), "jornada_fin": int(bsess[h]), "familias_max": int(count[h])})
    stop = decide_stop14(fam_v, fam_strict)
    fam_alarm = {f: fam_v[f]["veredicto"] in NONSTABLE_VERDICTS for f in DETECTOR_FAMILIES}
    return {
        "veredictos_estadistico": stat_v, "veredictos_familia": fam_v, "veredictos_familia_estricta": fam_strict,
        "recuento_por_frontera": count, "fechas": dates, "stop14": stop,
        "alarma_familia": fam_alarm, "alarma_fechas": bool(dates), "alarma_stop14": stop["alcance"] != STOP_NOT_ACTIVE,
        "alarma_global": any(fam_alarm.values()) or bool(dates) or stop["alcance"] != STOP_NOT_ACTIVE,
        "no_evaluable": {f: fam_v[f]["veredicto"] == V_NOT_EVALUABLE for f in DETECTOR_FAMILIES},
    }


# ---------------------------------------------------------------------------
# Mundos sinteticos (controles negativos y positivos)
# ---------------------------------------------------------------------------

SHAPE_STEP, SHAPE_TREND, SHAPE_EPISODE = "SALTO", "TENDENCIA", "EPISODIO"
INJECTABLE = (FAM_LEVEL, FAM_SCALE, FAM_SHAPE, FAM_LINEAR, FAM_MAGNITUDE, FAM_CLOCK, FAM_DISCRETE)


@dataclass(frozen=True)
class Change:
    """Cambio inyectado en un parametro del mundo. ``start``/``end`` en fraccion de jornadas.
    Multiplicativos (ESCALA, DEPENDENCIA_MAGNITUD, RELOJ, DISCRECION): el parametro se multiplica por ``magnitude``.
    Aditivos: NIVEL (deriva en sigmas), DEPENDENCIA_LINEAL (theta de un MA(1)); FORMA: grados de libertad destino."""
    family: str
    shape: str
    magnitude: float
    start: float = 0.5
    end: float = 1.0


@dataclass(frozen=True)
class WorldSpec:
    """Mundo sintetico. Sin ``changes`` es un control NEGATIVO: ninguna propiedad cambia, pero conserva volatilidad
    diaria persistente, perfil de reloj, colas, dependencia intradia en magnitud, discrecion por tick, rolls y huecos.
    ``daily_phi`` y ``seasonal_amplitude`` son PROVISIONALES: el protocolo los fija estimando una vez la persistencia real
    (aun no autorizado). El volumen generado NO representa el real (por eso queda fuera del detector)."""
    label: str = "NEGATIVO"
    n_sessions: int = 1420
    width: int = PANEL_WIDTH
    active_minutes: int = 1380            # 18:00-17:00 de Nueva York
    base_sigma: float = 3e-4
    daily_phi: float = 0.97
    daily_sd: float = 0.10
    seasonal_amplitude: float = 0.0
    seasonal_period: int = 252
    t_df: float = 4.0
    intra_phi: float = 0.98
    intra_sd: float = 0.05
    profile_amplitude: float = 1.0
    ma_theta: float = 0.0
    drift_sigma: float = 0.0
    base_price: float = 15000.0
    tick: float = 0.25
    roll_every: int = 63
    gap_probability: float = 0.02
    changes: tuple[Change, ...] = ()


def _change_path(ch: Change, u: np.ndarray) -> np.ndarray:
    if ch.shape == SHAPE_STEP:
        return (u >= ch.start).astype(float)
    if ch.shape == SHAPE_TREND:
        return u.copy()
    if ch.shape == SHAPE_EPISODE:
        return ((u >= ch.start) & (u < ch.end)).astype(float)
    raise ValueError(f"forma de cambio desconocida: {ch.shape}")


def _param_paths(spec: WorldSpec) -> dict[str, np.ndarray]:
    s = spec.n_sessions
    u = np.arange(s) / max(s - 1, 1)
    p = {"escala": np.ones(s), "deriva": np.full(s, spec.drift_sigma), "df": np.full(s, spec.t_df), "theta": np.full(s, spec.ma_theta),
         "intra": np.ones(s), "reloj": np.full(s, spec.profile_amplitude), "precio": np.ones(s)}
    for ch in spec.changes:
        g = _change_path(ch, u)
        if ch.family == FAM_SCALE:
            p["escala"] *= ch.magnitude ** g
        elif ch.family == FAM_MAGNITUDE:
            p["intra"] *= ch.magnitude ** g
        elif ch.family == FAM_CLOCK:
            p["reloj"] *= ch.magnitude ** g
        elif ch.family == FAM_DISCRETE:
            p["precio"] *= ch.magnitude ** g
        elif ch.family == FAM_LEVEL:
            p["deriva"] = p["deriva"] + ch.magnitude * g
        elif ch.family == FAM_LINEAR:
            p["theta"] = p["theta"] + ch.magnitude * g
        elif ch.family == FAM_SHAPE:
            p["df"] = p["df"] + (ch.magnitude - p["df"]) * g
        else:
            raise ValueError(f"familia no inyectable: {ch.family}")
    return p


def clock_profile(active_minutes: int, amplitude: float) -> np.ndarray:
    """Perfil intradia (minuto de sesion): picos en la apertura de contado (09:30) y el cierre (16:00) de Nueva York, media 1."""
    c = np.arange(active_minutes, dtype=float)
    scale = active_minutes / 1380.0
    bumps = 1.5 * np.exp(-((c - 930 * scale) / (30 * scale)) ** 2) + 0.8 * np.exp(-((c - 1320 * scale) / (40 * scale)) ** 2) + 0.4 * np.exp(-((c - 840 * scale) / (60 * scale)) ** 2)
    prof = 0.6 + amplitude * bumps
    return prof / prof.mean()


def simulate_world(spec: WorldSpec, rng: np.random.Generator) -> SessionPanel:
    s, m = spec.n_sessions, spec.active_minutes
    p = _param_paths(spec)
    # Volatilidad diaria: AR(1) en logaritmos, arrancando en su distribucion estacionaria.
    e_day = rng.standard_normal(s)
    h = np.empty(s)
    h[0] = rng.normal(0.0, spec.daily_sd / math.sqrt(1 - spec.daily_phi ** 2))
    for t in range(1, s):
        h[t] = spec.daily_phi * h[t - 1] + spec.daily_sd * e_day[t]
    h = h - h.mean()
    seas = spec.seasonal_amplitude * np.sin(2 * np.pi * np.arange(s) / spec.seasonal_period)
    sigma_day = spec.base_sigma * np.exp(h + seas) * p["escala"]
    # Dependencia intradia en magnitud: AR(1) por jornada (vectorizado entre jornadas), arranque estacionario.
    sd_intra = spec.intra_sd * p["intra"]
    e_intra = rng.standard_normal((s, m))
    g = np.empty((s, m))
    g[:, 0] = e_intra[:, 0] * sd_intra / math.sqrt(1 - spec.intra_phi ** 2)
    for c in range(1, m):
        g[:, c] = spec.intra_phi * g[:, c - 1] + sd_intra * e_intra[:, c]
    g -= g.mean()
    prof = np.vstack([clock_profile(m, a) for a in np.unique(p["reloj"])])
    prof_idx = np.searchsorted(np.unique(p["reloj"]), p["reloj"])
    sig = sigma_day[:, None] * prof[prof_idx] * np.exp(g)
    df = p["df"][:, None]
    z = rng.standard_t(np.broadcast_to(df, (s, m))) * np.sqrt((df - 2.0) / df)
    th = p["theta"][:, None]
    zl = np.concatenate([np.zeros((s, 1)), z[:, :-1]], axis=1)
    eps = (z + th * zl) / np.sqrt(1.0 + th * th)
    ret = sig * eps + p["deriva"][:, None] * sig
    # Precio continuo dentro de cada contrato y reiniciado en cada roll (serie sin ajustar, como la real): el nivel de
    # precio, y con el la discrecion, solo deriva dentro de un contrato.
    tot = ret.sum(1)
    cum = np.cumsum(tot)
    first = (np.arange(s) // spec.roll_every) * spec.roll_every
    before_contract = np.where(first > 0, cum[np.maximum(first - 1, 0)], 0.0)
    session_open = cum - tot - before_contract
    logp = (np.log(spec.base_price * p["precio"]) + session_open)[:, None] + np.cumsum(ret, axis=1)
    price = np.maximum(np.round(np.exp(logp) / spec.tick), 1.0) * spec.tick
    lp = np.log(price)
    r_act = np.full((s, m), np.nan)
    r_act[:, 1:] = np.diff(lp, axis=1)
    present_act = np.ones((s, m), dtype=bool)
    gap_rows = np.flatnonzero(rng.random(s) < spec.gap_probability)
    for row in gap_rows:
        length = int(rng.integers(1, max(2, round(31 * m / 1380))))      # hasta 30 min en una jornada de 1380
        start = int(rng.integers(1, max(m - length - 1, 2)))
        present_act[row, start:start + length] = False
        r_act[row, start:start + length + 1] = np.nan
    r_act[~present_act] = np.nan
    r = np.full((s, spec.width), np.nan)
    present = np.zeros((s, spec.width), dtype=bool)
    r[:, :m], present[:, :m] = r_act, present_act
    close = np.full((s, spec.width), np.nan)
    close[:, :m] = np.where(present_act, price, np.nan)
    vol = np.full((s, spec.width), np.nan)
    vol[:, :m] = np.where(present_act, np.round(np.exp(np.log(800.0) + 0.5 * (h[:, None] + np.log(prof[prof_idx])) + 0.3 * rng.standard_normal((s, m)))), np.nan)
    dates = np.busday_offset(np.datetime64("2020-01-02"), np.arange(s), roll="forward")
    contract = np.array([f"C{i // spec.roll_every:03d}" for i in range(s)], dtype=object)
    return SessionPanel(r=r, present=present, session_date=dates, contract=contract, volume=vol, close=close)


def world_rng(role: str, label: str, rep: int) -> np.random.Generator:
    """Semilla propia de cada (rol, tipo de mundo, replica): el resultado no depende del reparto entre procesos, y los
    roles de calibracion y validacion usan corrientes de numeros aleatorios distintas."""
    return np.random.default_rng([BASE_SEED, ROLE_CODES[role], zlib.crc32(label.encode()), rep])


# ---------------------------------------------------------------------------
# Ejecucion de replicas (proteccion de memoria heredada de TDA-13)
# ---------------------------------------------------------------------------

@dataclass
class WorldResult:
    role: str
    label: str
    rep: int
    stats: PanelStats
    seconds: float
    peak_working_set_gb: float | None
    peak_commit_gb: float | None
    thresholds_sha256: str = ""
    world_spec_sha256: str = ""         # huella de la especificacion con la que SE EJECUTO este mundo
    detector_spec_sha256: str = ""      # huella del detector con el que se analizo


def process_peak_memory_gb() -> tuple[float | None, float | None]:
    """Pico de memoria fisica y comprometida de ESTE proceso (Windows); ``(None, None)`` si no se puede medir."""
    if os.name != "nt":
        return None, None
    try:
        class _PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong), ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PMC), ctypes.c_ulong]
        c = _PMC()
        c.cb = ctypes.sizeof(_PMC)
        if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return None, None
        return c.PeakWorkingSetSize / 2 ** 30, c.PeakPagefileUsage / 2 ** 30
    except (AttributeError, OSError):
        return None, None


def run_world_task(task: tuple) -> WorldResult:
    role, label, rep, wspec, dspec, thr_sha = task
    t0 = time.perf_counter()
    panel = simulate_world(wspec, world_rng(role, label, rep))
    ps = analyze_panel(panel, dspec)
    ws, cm = process_peak_memory_gb()
    if label != wspec.label:
        raise ValueError(f"la tarea {label!r} no coincide con la etiqueta de su especificacion {wspec.label!r}")
    return WorldResult(role, label, rep, ps, time.perf_counter() - t0, ws, cm, thr_sha,
                       world_spec_sha256(wspec), detector_spec_sha256(dspec))


def resolve_workers(requested: int, n_tasks: int) -> int:
    """Procesos efectivos: nunca mas que la proteccion de memoria y el tope de CPU de TDA-13 permiten."""
    return max(1, min(memory_safe_workers(requested, available_memory_gb()), default_n_workers(n_tasks)))


def run_world_tasks(tasks: list[tuple], n_workers: int = 1) -> list[WorldResult]:
    workers = resolve_workers(n_workers, len(tasks)) if n_workers > 1 else 1
    if workers <= 1 or len(tasks) <= 1:
        res = [run_world_task(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            res = list(pool.map(run_world_task, tasks, chunksize=1))
    return sorted(res, key=lambda r: (r.role, r.label, r.rep))


def make_tasks(role: str, wspec: WorldSpec, n: int, dspec: DetectorSpec = DetectorSpec(), first_rep: int = 0, thr: FrozenThresholds | None = None) -> list[tuple]:
    """Tareas de mundos. Las de VALIDACION exigen umbrales congelados completos: se crean despues de calibrar."""
    if role == ROLE_VALIDATION:
        if thr is None:
            raise FrozenThresholdsError("los mundos de validacion se crean solo con umbrales congelados")
        verify_frozen(thr)
    return [(role, wspec.label, first_rep + i, wspec, dspec, thr.sha256 if thr else "") for i in range(n)]


# ---------------------------------------------------------------------------
# Calibracion conjunta (solo mundos de CALIBRACION)
# ---------------------------------------------------------------------------

def _require_role(results: list[WorldResult], role: str) -> None:
    bad = sorted({r.role for r in results if r.role != role})
    if bad:
        raise CalibrationRoleError(f"se esperaban solo mundos {role}; llegaron tambien {bad}")


def _build_thresholds(alpha: float, k_star: int, null_z: dict, null_d: dict, dspec: DetectorSpec, info: dict) -> FrozenThresholds:
    q = 1.0 - alpha
    c_date = tuple((s, float(np.quantile(null_z[s][np.isfinite(null_z[s])], q, method="higher"))) for s in DETECTOR_STATISTICS)
    c_verd = tuple((s, float(np.quantile(null_d[s][np.isfinite(null_d[s])], q, method="higher"))) for s in DETECTOR_STATISTICS)
    return FrozenThresholds(alpha=float(alpha), k_star=int(k_star), c_date=c_date, c_verdict=c_verd, type_margin=None,
                            materiality=tuple((s, float(m)) for s, (_, m) in DETECTOR_STATISTICS.items()),
                            detector_spec=tuple(sorted(asdict(dspec).items())), calibration_info=tuple(sorted(info.items())))


def _check_protocol_worlds(results: list[WorldResult], allowed: dict[str, WorldSpec], det_sha: str, where: str) -> None:
    """Cada mundo debe corresponder a una especificacion PREDECLARADA del protocolo (misma etiqueta y misma huella) y
    haberse analizado con el detector del protocolo."""
    for r in results:
        spec = allowed.get(r.label)
        if spec is None:
            raise ValidationManifestError(f"{where}: mundo {r.label!r} sin especificacion predeclarada en el protocolo")
        if r.world_spec_sha256 != world_spec_sha256(spec):
            raise ValidationManifestError(f"{where}: el mundo {r.label}#{r.rep} se ejecuto con una especificacion distinta de la congelada")
        if r.detector_spec_sha256 != det_sha:
            raise ValidationManifestError(f"{where}: el mundo {r.label}#{r.rep} se analizo con un detector distinto del del protocolo")


def calibrate_thresholds(negatives: list[WorldResult], manifest: "ValidationManifest", target: float = G2_CALIBRATION_RATE) -> FrozenThresholds:
    """Umbrales de fechas, veredictos y (por derivacion) STOP-14 calibrados CONJUNTAMENTE: se elige el mayor ``alpha``
    (umbrales mas bajos) cuya falsa alarma GLOBAL en los mundos negativos de calibracion es <= ``target``, con el menor
    ``k*`` >= 2 para el que exista. Devuelve umbrales SIN margen de tipo (ver :func:`calibrate_type_margin`).

    Protocolo congelado (correccion de auditoria): los mundos de calibracion deben ser variantes negativas o de estres
    PREDECLARADAS en ``manifest`` (etiqueta y huella), analizadas con su detector, y debe haber mundos de TODAS las
    variantes negativas. Los umbrales registran la huella del manifiesto: la validacion solo los acepta con ese mismo
    manifiesto."""
    _require_role(negatives, ROLE_CALIBRATION)
    if not negatives:
        raise CalibrationFailure("sin mundos negativos de calibracion")
    dspec = manifest.detector_spec
    allowed = {s.label: s for _, s in manifest.negative_specs} | {s.label: s for _, s in manifest.stress_specs}
    _check_protocol_worlds(negatives, allowed, detector_spec_sha256(dspec), "calibracion")
    absent = sorted(s.label for _, s in manifest.negative_specs if not any(r.label == s.label for r in negatives))
    if absent:
        raise ValidationManifestError(f"calibracion: faltan mundos de las variantes negativas {absent}")
    null_z = {s: np.array([max_abs_z(r.stats, s) for r in negatives]) for s in DETECTOR_STATISTICS}
    null_d = {s: np.array([constancy_statistic(*nonoverlapping(r.stats, s)) for r in negatives]) for s in DETECTOR_STATISTICS}
    for s in DETECTOR_STATISTICS:
        if not np.isfinite(null_z[s]).any() or not np.isfinite(null_d[s]).any():
            raise CalibrationFailure(f"el estadistico {s} no es evaluable en los mundos negativos")
    n = len(negatives)
    for k_star in range(2, len(DETECTOR_FAMILIES) + 1):
        for alpha in ALPHA_GRID:
            if alpha * n < 1:          # cuantil no estimable con tan pocos mundos
                continue
            thr = _build_thresholds(alpha, k_star, null_z, null_d, dspec, {})
            alarms = [apply_rule(r.stats, thr, type_margin=0.0)["alarma_global"] for r in negatives]
            rate = float(np.mean(alarms))
            if rate <= target:
                info = {"n_mundos_calibracion": n, "tasa_global_calibracion": rate, "objetivo_calibracion": target,
                        "especificaciones_calibracion": ",".join(sorted({r.world_spec_sha256 for r in negatives})),
                        "manifiesto_sha256": manifest.sha256()}
                return freeze_thresholds(_build_thresholds(alpha, k_star, null_z, null_d, dspec, info))
    raise CalibrationFailure("ningun (alpha, k*) de la rejilla cumple el objetivo de calibracion")


def type_metrics(outcomes: list[tuple[str, dict | None]]) -> dict:
    """Dos metricas SEPARADAS sobre mundos positivos (correccion de auditoria):

    - ``errores``: detectados con un tipo ESPECIFICO afirmado (salto, tendencia, episodio) distinto del inyectado; para
      un cambio en el borde, afirmar cualquier tipo especifico es error (afirma una permanencia que no puede conocerse).
    - ``correctos``: detectados con EXACTAMENTE el resultado esperado (:data:`EXPECTED_TYPE`; para el borde, la anotacion
      de borde). ``abstenciones``: detectados sin tipo afirmado ni acierto (tipo no identificable, sin patron, borde en
      un mundo que no es de borde).
    Pocas o ninguna equivocacion con muchas abstenciones NO es capacidad de clasificar: la capacidad es
    ``correctos / detectados``. ``outcomes``: (tipo inyectado, veredicto de la familia inyectada o None si no se evaluo)."""
    per = {}
    for typ, v in outcomes:
        d = per.setdefault(typ, {"mundos": 0, "detectados": 0, "correctos": 0, "errores": 0, "abstenciones": 0})
        d["mundos"] += 1
        if v is None or v["veredicto"] not in NONSTABLE_VERDICTS:
            continue
        d["detectados"] += 1
        got = (v["veredicto"], v["anotacion"])
        if got == EXPECTED_TYPE[typ]:
            d["correctos"] += 1
        elif got in SPECIFIC_TYPES:
            d["errores"] += 1
        else:
            d["abstenciones"] += 1
    tot = {k: sum(d[k] for d in per.values()) for k in ("mundos", "detectados", "correctos", "errores", "abstenciones")}
    det = tot["detectados"]
    tot.update(tasa_errores=tot["errores"] / det if det else float("nan"), capacidad=tot["correctos"] / det if det else float("nan"),
               tasa_abstencion=tot["abstenciones"] / det if det else float("nan"), por_tipo=per)
    return tot


def type_errors(results: list[WorldResult], truth: dict[str, tuple[str, str]], thr: FrozenThresholds, type_margin: float) -> dict:
    """:func:`type_metrics` sobre mundos positivos. ``truth``: etiqueta del mundo -> (familia inyectada, tipo inyectado)."""
    outcomes = []
    for r in results:
        fam, typ = truth[r.label]
        outcomes.append((typ, apply_rule(r.stats, thr, type_margin=type_margin)["veredictos_familia"][fam]))
    return type_metrics(outcomes)


def calibrate_type_margin(thr: FrozenThresholds, positives: list[WorldResult], manifest: "ValidationManifest",
                          target: float = TYPE_CALIBRATION_RATE) -> FrozenThresholds:
    """Menor margen de la rejilla cuya confusion de tipos (errores / detectados) en mundos positivos de CALIBRACION es
    <= ``target``. Si ninguno lo cumple, margen infinito: nunca se publican tipos. Devuelve los umbrales congelados completos.
    Los mundos deben ser controles con tipo PREDECLARADOS en el mismo manifiesto con el que se calibraron los umbrales; la
    verdad (tipo y familia) sale del manifiesto."""
    _require_role(positives, ROLE_CALIBRATION)
    verify_frozen(thr, require_complete=False)
    if dict(thr.calibration_info).get("manifiesto_sha256") != manifest.sha256():
        raise ValidationManifestError("los umbrales se calibraron con otro manifiesto de protocolo")
    _check_protocol_worlds(positives, {s.label: s for _, _, s in manifest.type_specs}, detector_spec_sha256(manifest.detector_spec), "calibracion de tipos")
    truth = {s.label: (fam, typ) for typ, fam, s in manifest.type_specs}
    chosen, stats = math.inf, {}
    for margin in MARGIN_GRID:
        te = type_errors(positives, truth, thr, margin)
        if te["detectados"] == 0 or te["errores"] / te["detectados"] <= target:
            chosen, stats = margin, te
            break
    info = dict(thr.calibration_info) | {"n_mundos_tipo": len(positives), "confusion_tipo_calibracion": stats.get("tasa_errores", float("nan")),
                                         "capacidad_tipo_calibracion": stats.get("capacidad", float("nan")),
                                         "abstencion_tipo_calibracion": stats.get("tasa_abstencion", float("nan"))}
    return freeze_thresholds(replace(thr, type_margin=float(chosen), calibration_info=tuple(sorted(info.items()))))


# ---------------------------------------------------------------------------
# Aceptacion (solo mundos de VALIDACION; nunca cambia umbrales)
# ---------------------------------------------------------------------------

def clopper_pearson_upper(k: int, n: int, confidence: float) -> float:
    """Limite superior unilateral exacto de una proporcion binomial."""
    if n <= 0:
        return float("nan")
    return 1.0 if k >= n else float(betaincinv(k + 1, n - k, confidence))


def clopper_pearson_lower(k: int, n: int, confidence: float) -> float:
    if n <= 0:
        return float("nan")
    return 0.0 if k <= 0 else float(betaincinv(k, n - k + 1, 1.0 - confidence))


def max_accepted_false_alarms(n: int, target: float = G2_TARGET_RATE, confidence: float = G2_CONFIDENCE) -> int:
    """Mayor numero de falsas alarmas en ``n`` mundos cuyo limite superior de confianza sigue <= ``target`` (-1 si ninguno)."""
    k = -1
    while k + 1 <= n and clopper_pearson_upper(k + 1, n, confidence) <= target:
        k += 1
    return k


def acceptance_probability(n: int, true_rate: float, target: float = G2_TARGET_RATE, confidence: float = G2_CONFIDENCE) -> float:
    """Probabilidad de que el criterio acepte si la tasa real es ``true_rate`` (se informa antes de validar)."""
    kmax = max_accepted_false_alarms(n, target, confidence)
    return 0.0 if kmax < 0 else float(bdtr(kmax, n, true_rate))


def rate_row(control: str, decision: str, k: int, n: int, confidence: float, limit: float, upper: bool = True) -> dict:
    lo, hi = wilson_interval(k, n)
    bound = clopper_pearson_upper(k, n, confidence) if upper else clopper_pearson_lower(k, n, confidence)
    ok = (bound <= limit) if upper else (bound >= limit)
    return {"control": control, "decision": decision, "eventos": k, "n": n, "tasa": k / n if n else float("nan"),
            "wilson_95_inf": lo, "wilson_95_sup": hi, "confianza": confidence,
            "limite_confianza": bound, "tipo_limite": "superior" if upper else "inferior", "criterio": limit,
            "estado": STATUS_COMPATIBLE if ok else STATUS_NOT_VALIDATED}


class ValidationManifestError(Exception):
    """Faltan controles negativos obligatorios, tienen menos replicas de las exigidas, hay mundos invalidos o repetidos,
    o un mundo ejecutado no coincide con la especificacion congelada: la validacion NO puede aprobarse."""


STATUS_ABSENT = "NO_EVALUADO_CONTROL_AUSENTE"
STATUS_INSUFFICIENT = "NO_EVALUADO_REPLICAS_INSUFICIENTES"


def world_spec_sha256(spec: WorldSpec) -> str:
    """Huella de TODA la especificacion de un mundo: persistencia, estacionalidad, colas, perfil, discrecion, rolls,
    huecos, cambios inyectados (familia, forma, magnitud, inicio y fin) y tamaño."""
    return hashlib.sha256(_canonical(asdict(spec)).encode()).hexdigest()


def detector_spec_sha256(spec: DetectorSpec) -> str:
    return hashlib.sha256(_canonical(asdict(spec)).encode()).hexdigest()


def _first_session(frac: float, n_sessions: int) -> int:
    """Primera jornada en la que actua un cambio que empieza en ``frac`` (misma regla que :func:`_change_path`)."""
    idx = np.flatnonzero(np.arange(n_sessions) / max(n_sessions - 1, 1) >= frac)
    return int(idx[0]) if idx.size else n_sessions


def implied_type(ch: Change, spec: WorldSpec, dspec: DetectorSpec) -> str:
    """Tipo que REALMENTE produce un cambio inyectado, segun su forma y su posicion en las ventanas sin solapamiento del
    detector: salto con al menos ``min_side`` ventanas completas a cada lado -> SALTO; salto con menos a algun lado ->
    BORDE; tendencia -> TENDENCIA; episodio con ``min_side`` ventanas antes y despues y al menos una dentro -> EPISODIO.
    Cualquier otra configuracion es ``AMBIGUO`` y no puede declararse como ningun tipo."""
    ws = dspec.window_sessions
    n_w = (spec.n_sessions // dspec.batch_sessions) * dspec.batch_sessions // ws
    s0 = _first_session(ch.start, spec.n_sessions)
    before, after = s0 // ws, n_w - math.ceil(s0 / ws)
    if ch.shape == SHAPE_TREND:
        return TYPE_TREND
    if ch.shape == SHAPE_STEP:
        if before >= dspec.min_side_windows and after >= dspec.min_side_windows:
            return TYPE_STEP
        return TYPE_EDGE if 0 < s0 < spec.n_sessions else "AMBIGUO"
    if ch.shape == SHAPE_EPISODE:
        s1 = _first_session(ch.end, spec.n_sessions)
        after_end, inside = n_w - math.ceil(s1 / ws), s1 // ws - math.ceil(s0 / ws)
        if before >= dspec.min_side_windows and after_end >= dspec.min_side_windows and inside >= 1:
            return TYPE_EPISODE
    return "AMBIGUO"


def _base(spec: WorldSpec) -> WorldSpec:
    """La especificacion sin etiqueta ni cambios: el 'mundo de fondo' sobre el que se inyecta un cambio."""
    return replace(spec, label="", changes=())


def manifest_coherence_errors(m: "ValidationManifest") -> list[str]:
    """Contradicciones entre lo que el manifiesto DECLARA y lo que sus especificaciones HACEN (independiente de hashes):

    - controles negativos y variantes de estres: sin ningun cambio inyectado;
    - control parcial: exactamente un cambio, de ESCALA, sobre el mismo mundo de fondo que una variante negativa;
    - controles con tipo: exactamente un cambio, de la familia declarada, cuyo tipo real (:func:`implied_type`) es el
      declarado;
    - controles de potencia: exactamente un cambio, de la familia declarada, y un nivel conocido;
    - todo control positivo o parcial se construye sobre el mundo de fondo de una variante negativa o de estres
      PREDECLARADA (el protocolo congelado); ninguna especificacion se aparta de el sin estar declarada."""
    err = []
    bases = {_base(s) for _, s in m.negative_specs} | {_base(s) for _, s in m.stress_specs}
    for kind, specs in (("negativo", m.negative_specs), ("estres", m.stress_specs)):
        for name, s in specs:
            if s.changes:
                err.append(f"control {kind} {name!r} con cambios inyectados {[(c.family, c.shape) for c in s.changes]}")
    if m.partial_spec is not None:
        p = m.partial_spec
        if len(p.changes) != 1 or p.changes[0].family != FAM_SCALE:
            err.append(f"control parcial con cambios {[(c.family, c.shape) for c in p.changes]}: debe tener solo un cambio de ESCALA")
        if _base(p) not in {_base(s) for _, s in m.negative_specs}:
            err.append("control parcial construido sobre un mundo de fondo distinto de las variantes negativas")
    for typ, fam, s in m.type_specs:
        if len(s.changes) != 1:
            err.append(f"control de tipo {s.label!r}: {len(s.changes)} cambios; debe tener exactamente uno")
            continue
        ch = s.changes[0]
        if ch.family != fam:
            err.append(f"control de tipo {s.label!r}: familia declarada {fam}, inyectada {ch.family}")
        real = implied_type(ch, s, m.detector_spec)
        if real != typ:
            err.append(f"control de tipo {s.label!r}: tipo declarado {typ}, el cambio inyectado produce {real}")
    for fam, level, s in m.power_specs:
        if level not in POWER_REQUIRED:
            err.append(f"control de potencia {s.label!r}: nivel desconocido {level}")
        if len(s.changes) != 1 or s.changes[0].family != fam:
            err.append(f"control de potencia {s.label!r}: debe tener un unico cambio de {fam}; tiene {[(c.family, c.shape) for c in s.changes]}")
    positives = ([m.partial_spec] if m.partial_spec is not None else []) + [s for _, _, s in m.type_specs] + [s for _, _, s in m.power_specs]
    for s in positives:
        if _base(s) not in bases:
            err.append(f"{s.label!r} se aparta del protocolo: su mundo de fondo no es ninguna variante negativa ni de estres predeclarada")
    return err


@dataclass(frozen=True)
class ValidationManifest:
    """Controles de la validacion con sus especificaciones de mundo CONGELADAS, fijados ANTES de crearla.

    - ``negative_specs``: (nombre de variante, WorldSpec). Imprescindibles: sin ellos se lanza ``ValidationManifestError``.
    - ``partial_spec``: mundo con cambio SOLO de escala (control negativo parcial).
    - ``type_specs``: (tipo inyectado, familia inyectada, WorldSpec): la verdad de cada mundo sale del manifiesto.
    - ``power_specs``: (familia, nivel de potencia, WorldSpec).
    Parcial, tipos y potencia ausentes o con menos replicas que el minimo: la decision que depende de ellos queda
    bloqueada y la ausencia figura en la tabla. Cada etiqueta de mundo identifica UNA especificacion. Minimos aprobados:
    1000 por control negativo y parcial; 100 por tipo y por nivel de potencia (minimo tecnico provisional)."""
    negative_specs: tuple[tuple[str, WorldSpec], ...] = ()
    partial_spec: WorldSpec | None = None
    type_specs: tuple[tuple[str, str, WorldSpec], ...] = ()
    power_specs: tuple[tuple[str, str, WorldSpec], ...] = ()
    detector_spec: DetectorSpec = DetectorSpec()                       # detector del protocolo (calibracion y validacion)
    stress_specs: tuple[tuple[str, WorldSpec], ...] = ()               # variantes negativas de estres PREDECLARADAS (solo calibracion)
    min_negative_replicates: int = 1000
    min_partial_replicates: int = 1000
    min_type_replicates: int = 100
    min_power_replicates: int = 100

    def __post_init__(self):
        labels = [s.label for _, s in self.negative_specs] + ([self.partial_spec.label] if self.partial_spec else []) \
            + [s.label for _, _, s in self.type_specs] + [s.label for _, _, s in self.power_specs] + [s.label for _, s in self.stress_specs]
        if len(labels) != len(set(labels)):
            raise ValidationManifestError("el manifiesto repite etiquetas de mundo: cada etiqueta debe identificar una sola especificacion")
        if len({n for n, _ in self.negative_specs}) != len(self.negative_specs):
            raise ValidationManifestError("el manifiesto repite variantes negativas")
        bad = [t for t, _, _ in self.type_specs if t not in EXPECTED_TYPE]
        if bad:
            raise ValidationManifestError(f"tipos desconocidos en el manifiesto: {bad}")
        contradictions = manifest_coherence_errors(self)
        if contradictions:
            raise ValidationManifestError("especificacion contradictoria: " + "; ".join(contradictions))

    @property
    def negative_variants(self) -> tuple[str, ...]:
        return tuple(n for n, _ in self.negative_specs)

    @property
    def required_types(self) -> tuple[str, ...]:
        return tuple(sorted({t for t, _, _ in self.type_specs}))

    @property
    def required_power(self) -> tuple[tuple[str, str], ...]:
        return tuple((f, lv) for f, lv, _ in self.power_specs)

    @property
    def n_controls(self) -> int:
        """Controles entre los que se reparte la confianza: cada variante negativa y el control parcial (planificado
        aunque falte: el reparto se fija con el manifiesto, no con lo que llegue)."""
        return len(self.negative_specs) + 1

    def spec_hashes(self) -> dict[str, str]:
        out = {f"NEGATIVO:{n}": world_spec_sha256(s) for n, s in self.negative_specs}
        if self.partial_spec is not None:
            out["PARCIAL"] = world_spec_sha256(self.partial_spec)
        out |= {f"TIPO:{t}:{f}:{s.label}": world_spec_sha256(s) for t, f, s in self.type_specs}
        out |= {f"POTENCIA:{f}:{lv}": world_spec_sha256(s) for f, lv, s in self.power_specs}
        out |= {f"ESTRES:{n}": world_spec_sha256(s) for n, s in self.stress_specs}
        out["DETECTOR"] = detector_spec_sha256(self.detector_spec)
        return out

    def sha256(self) -> str:
        return hashlib.sha256(_canonical(asdict(self)).encode()).hexdigest()


@dataclass
class ValidationResult:
    table: pd.DataFrame
    global_validated: bool
    dates_validated: bool
    families_validated: dict[str, bool]
    stop14_scope_validated: bool
    types_validated: bool
    power_ok: dict[str, bool]
    thresholds_sha256: str
    manifest_sha256: str = ""
    missing_controls: tuple[str, ...] = ()
    specs_table: pd.DataFrame | None = None       # control, etiqueta, huella de la especificacion congelada, mundos verificados
    type_metrics: dict | None = None
    types_validated_by_type: dict[str, bool] = field(default_factory=dict)


def _check_worlds(results: list[WorldResult], thr: FrozenThresholds, where: str) -> None:
    bad_role = sorted({r.role for r in results if r.role != ROLE_VALIDATION})
    if bad_role:
        raise ValidationManifestError(f"{where}: mundos con rol {bad_role}; solo se admiten mundos {ROLE_VALIDATION}")
    if any(r.thresholds_sha256 != thr.sha256 for r in results):
        raise ValidationManifestError(f"{where}: mundos sin la huella de los umbrales congelados o creados con otros umbrales")


def _check_spec(results: list[WorldResult], spec: WorldSpec, det_sha: str, where: str) -> str:
    """Cada mundo ejecutado debe coincidir con la especificacion congelada: misma etiqueta, misma huella de mundo y misma
    huella de detector (la de los umbrales). Cualquier discrepancia se rechaza."""
    sha = world_spec_sha256(spec)
    for r in results:
        if r.label != spec.label:
            raise ValidationManifestError(f"{where}: mundo con etiqueta {r.label!r}; el manifiesto congela {spec.label!r}")
        if r.world_spec_sha256 != sha:
            raise ValidationManifestError(f"{where}: el mundo {r.label}#{r.rep} se ejecuto con una especificacion distinta de la congelada")
        if r.detector_spec_sha256 != det_sha:
            raise ValidationManifestError(f"{where}: el mundo {r.label}#{r.rep} se analizo con un detector distinto del de los umbrales")
    return sha


def _absent_row(control: str, decision: str, status: str, n: int, required: int) -> dict:
    return {"control": control, "decision": decision, "eventos": np.nan, "n": n, "tasa": np.nan, "wilson_95_inf": np.nan,
            "wilson_95_sup": np.nan, "confianza": np.nan, "limite_confianza": np.nan, "tipo_limite": "", "criterio": required,
            "estado": status}


def validate_procedure(thr: FrozenThresholds, manifest: ValidationManifest, negatives: dict[str, list[WorldResult]],
                       partial_negatives: list[WorldResult] | None = None,
                       typed_positives: list[WorldResult] | None = None,
                       power_worlds: dict[tuple[str, str], list[WorldResult]] | None = None,
                       confidence: float = G2_CONFIDENCE) -> ValidationResult:
    """Aceptacion con los umbrales congelados (no los cambia; verifica su SHA-256 antes y despues).

    - MANIFIESTO: las variantes negativas deben ser EXACTAMENTE las del manifiesto, cada una con al menos
      ``min_negative_replicates`` mundos de rol ``VALIDACION``, creados con estos umbrales y sin repetir ningun
      (etiqueta, replica). Si no, ``ValidationManifestError``: sin controles negativos no hay validacion posible.
    - TRAZABILIDAD (correccion de auditoria): cada mundo de cada control debe coincidir con la especificacion congelada
      en el manifiesto (etiqueta, huella completa del mundo y huella del detector); si no, ``ValidationManifestError``.
      Las huellas verificadas se devuelven en ``specs_table``.
    - Parcial, tipos o potencia ausentes o con menos replicas de las exigidas: fila explicita
      (``NO_EVALUADO_CONTROL_AUSENTE`` / ``NO_EVALUADO_REPLICAS_INSUFICIENTES``) y bloqueo de lo que depende de ellos:
      sin parcial no hay STOP-14 especifico/amplio automatico; sin tipos no se publican tipos; sin potencia la familia
      no puede ser ESTABLE.
    - Bonferroni sobre los ``manifest.n_controls`` controles planificados (``1 - (1-confidence)/V``).
    - Global: si alguna variante supera el 5 % (limite superior), TODO el procedimiento queda NO_VALIDADO (solo descriptivo).
    - Por alcance: familias, fechas y STOP-14 tienen ademas sus tasas propias; cada una que falle queda NO_VALIDADA.
    - Parcial (mundos con cambio SOLO de escala): falsa alarma = familia fuera del grupo ESCALA no estable, o STOP-14
      especifico/amplio (alcance provisional). Si falla, STOP-14 especifico/amplio no se activa automaticamente.
    - Tipos (correccion de auditoria): DOS metricas separadas (:func:`type_metrics`). Tipos especificos incorrectos /
      detectados, limite superior <= 10 %. Capacidad = aciertos / detectados, limite INFERIOR >= ``TYPE_CAPACITY_MIN``.
      Se informan tambien las abstenciones. Los tipos solo se validan si se cumplen AMBAS: la ausencia de errores no es
      evidencia de capacidad.
    - Potencia: limite INFERIOR de confianza >= 50 % (materialidad) y >= 80 % (doble).
    """
    verify_frozen(thr)
    sha_before = thr.sha256
    det_sha = detector_spec_sha256(_spec_from(thr))
    if not manifest.negative_specs:
        raise ValidationManifestError("el manifiesto no congela ninguna especificacion de control negativo")
    if not negatives:
        raise ValidationManifestError("sin controles negativos no puede aprobarse ninguna validacion")
    if set(negatives) != set(manifest.negative_variants):
        raise ValidationManifestError(f"variantes negativas {sorted(negatives)} distintas de las del manifiesto {sorted(manifest.negative_variants)}")
    specs_rows = []
    neg_spec = dict(manifest.negative_specs)
    for variant, res in negatives.items():
        if len(res) < manifest.min_negative_replicates:
            raise ValidationManifestError(f"{variant}: {len(res)} mundos; el manifiesto exige al menos {manifest.min_negative_replicates}")
        _check_worlds(res, thr, variant)
        specs_rows.append({"control": f"NEGATIVO:{variant}", "etiqueta": neg_spec[variant].label,
                           "sha256_especificacion": _check_spec(res, neg_spec[variant], det_sha, variant), "mundos_verificados": len(res)})
    all_results = [r for rs in negatives.values() for r in rs] + list(partial_negatives or []) + list(typed_positives or []) + [r for rs in (power_worlds or {}).values() for r in rs]
    _check_worlds(all_results, thr, "validacion")
    keys = [(r.label, r.rep) for r in all_results]
    if len(keys) != len(set(keys)):
        raise ValidationManifestError("hay mundos repetidos (misma etiqueta y replica): ninguno puede contarse dos veces")
    unknown_power = sorted(set(power_worlds or {}) - set(manifest.required_power))
    if unknown_power:
        raise ValidationManifestError(f"controles de potencia no previstos en el manifiesto: {unknown_power}")
    partial = list(partial_negatives or [])
    if partial:
        if manifest.partial_spec is None:
            raise ValidationManifestError("mundos parciales sin especificacion congelada en el manifiesto")
        specs_rows.append({"control": "PARCIAL", "etiqueta": manifest.partial_spec.label,
                           "sha256_especificacion": _check_spec(partial, manifest.partial_spec, det_sha, "PARCIAL"), "mundos_verificados": len(partial)})
    typed = list(typed_positives or [])
    type_by_label = {s.label: (t, f, s) for t, f, s in manifest.type_specs}
    unknown = sorted({r.label for r in typed} - set(type_by_label))
    if unknown:
        raise ValidationManifestError(f"mundos con tipo sin especificacion congelada en el manifiesto: {unknown}")
    for lab in sorted({r.label for r in typed}):
        t, f, s = type_by_label[lab]
        sub = [r for r in typed if r.label == lab]
        specs_rows.append({"control": f"TIPO:{t}:{f}", "etiqueta": lab, "sha256_especificacion": _check_spec(sub, s, det_sha, f"TIPO {lab}"),
                           "mundos_verificados": len(sub)})
    power_spec = {(f, lv): s for f, lv, s in manifest.power_specs}
    for key, res in (power_worlds or {}).items():
        specs_rows.append({"control": f"POTENCIA:{key[0]}:{key[1]}", "etiqueta": power_spec[key].label,
                           "sha256_especificacion": _check_spec(res, power_spec[key], det_sha, f"POTENCIA {key}"), "mundos_verificados": len(res)})
    # Coherencia de fases (correccion de auditoria): umbrales y validacion bajo el MISMO protocolo congelado.
    if dict(thr.calibration_info).get("manifiesto_sha256") != manifest.sha256():
        raise ValidationManifestError("los umbrales se calibraron con otro manifiesto de protocolo: calibracion y validacion no corresponden")
    if detector_spec_sha256(manifest.detector_spec) != det_sha:
        raise ValidationManifestError("el detector del manifiesto no es el de los umbrales congelados")
    conf_each = 1.0 - (1.0 - confidence) / manifest.n_controls
    missing = []
    rows = []
    fam_ok = dict.fromkeys(DETECTOR_FAMILIES, True)
    dates_ok = stop_ok = global_ok = True
    for variant, res in negatives.items():
        dec = [apply_rule(r.stats, thr) for r in res]
        n = len(dec)
        row = rate_row(variant, "GLOBAL", sum(d["alarma_global"] for d in dec), n, conf_each, G2_TARGET_RATE)
        rows.append(row)
        global_ok &= row["estado"] == STATUS_COMPATIBLE
        row = rate_row(variant, "FECHAS", sum(d["alarma_fechas"] for d in dec), n, conf_each, G2_TARGET_RATE)
        rows.append(row)
        dates_ok &= row["estado"] == STATUS_COMPATIBLE
        row = rate_row(variant, "STOP14", sum(d["alarma_stop14"] for d in dec), n, conf_each, G2_TARGET_RATE)
        rows.append(row)
        stop_ok &= row["estado"] == STATUS_COMPATIBLE
        for f in DETECTOR_FAMILIES:
            row = rate_row(variant, f"FAMILIA_{f}", sum(d["alarma_familia"][f] for d in dec), n, conf_each, G2_TARGET_RATE)
            rows.append(row)
            fam_ok[f] &= row["estado"] == STATUS_COMPATIBLE
            rows.append({**rate_row(variant, f"NO_EVALUABLE_{f}", sum(d["no_evaluable"][f] for d in dec), n, conf_each, 1.0), "estado": "INFORMATIVO"})
    scale_group = set(STOP_GROUPS["ESCALA"])
    if manifest.partial_spec is None or not partial:
        rows.append(_absent_row("NEGATIVO_PARCIAL_SOLO_ESCALA", "FAMILIAS_NO_ESCALA_O_STOP14_AMPLIO", STATUS_ABSENT, len(partial), manifest.min_partial_replicates))
        missing.append("NEGATIVO_PARCIAL_SOLO_ESCALA")
        scope_ok = False
    elif len(partial) < manifest.min_partial_replicates:
        rows.append(_absent_row("NEGATIVO_PARCIAL_SOLO_ESCALA", "FAMILIAS_NO_ESCALA_O_STOP14_AMPLIO", STATUS_INSUFFICIENT, len(partial), manifest.min_partial_replicates))
        missing.append("NEGATIVO_PARCIAL_SOLO_ESCALA")
        scope_ok = False
    else:
        dec = [apply_rule(r.stats, thr) for r in partial]
        k = sum(any(d["alarma_familia"][f] for f in DETECTOR_FAMILIES if f not in scale_group) or d["stop14"]["alcance_provisional"] in (STOP_SPECIFIC, STOP_BROAD) for d in dec)
        row = rate_row("NEGATIVO_PARCIAL_SOLO_ESCALA", "FAMILIAS_NO_ESCALA_O_STOP14_AMPLIO", k, len(dec), conf_each, G2_TARGET_RATE)
        rows.append(row)
        scope_ok = row["estado"] == STATUS_COMPATIBLE
        for f in DETECTOR_FAMILIES:
            if f not in scale_group:
                rows.append({**rate_row("NEGATIVO_PARCIAL_SOLO_ESCALA", f"FAMILIA_{f}", sum(d["alarma_familia"][f] for d in dec), len(dec), conf_each, G2_TARGET_RATE), "estado": "INFORMATIVO"})
    per_type = {t: sum(type_by_label[r.label][0] == t for r in typed) for t in manifest.required_types}
    short = {t: n for t, n in per_type.items() if n < manifest.min_type_replicates}
    tm = None
    if not manifest.type_specs or short:
        status = STATUS_ABSENT if not typed else STATUS_INSUFFICIENT
        rows.append(_absent_row("POSITIVOS_CON_TIPO", f"TIPOS {short or 'NO_PLANIFICADOS'}", status, len(typed), manifest.min_type_replicates))
        missing.append("POSITIVOS_CON_TIPO")
        types_ok = False
        types_by_type = dict.fromkeys(manifest.required_types, False)
    else:
        tm = type_metrics([(type_by_label[r.label][0], apply_rule(r.stats, thr)["veredictos_familia"][type_by_label[r.label][1]]) for r in typed])
        types_ok, type_rows = type_metric_rows(tm, confidence, manifest.required_types)
        rows += type_rows
        types_by_type = {r["control"].removeprefix("POSITIVOS_TIPO_"): r["estado"] == STATUS_COMPATIBLE
                         for r in type_rows if r["control"].startswith("POSITIVOS_TIPO_")}
    power_ok = dict.fromkeys(DETECTOR_FAMILIES, True)
    for fam, level in manifest.required_power:
        res = (power_worlds or {}).get((fam, level), [])
        if len(res) < manifest.min_power_replicates:
            rows.append(_absent_row(f"POSITIVO_{fam}", f"POTENCIA_{level}", STATUS_ABSENT if not res else STATUS_INSUFFICIENT, len(res), manifest.min_power_replicates))
            missing.append(f"POTENCIA_{fam}_{level}")
            power_ok[fam] = False
            continue
        det = sum(apply_rule(r.stats, thr)["veredictos_familia"][fam]["veredicto"] in NONSTABLE_VERDICTS for r in res)
        row = rate_row(f"POSITIVO_{fam}", f"POTENCIA_{level}", det, len(res), confidence, POWER_REQUIRED[level], upper=False)
        rows.append(row)
        power_ok[fam] = power_ok[fam] and row["estado"] == STATUS_COMPATIBLE
    for fam in DETECTOR_FAMILIES:                      # una familia sin ningun control de potencia exigido tampoco puede ser ESTABLE
        if not any(f == fam for f, _ in manifest.required_power):
            power_ok[fam] = False
    verify_frozen(thr)
    if thr.sha256 != sha_before:
        raise FrozenThresholdsError("los umbrales cambiaron durante la validacion")
    families = {f: global_ok and fam_ok[f] for f in DETECTOR_FAMILIES}
    return ValidationResult(table=pd.DataFrame(rows), global_validated=global_ok, dates_validated=global_ok and dates_ok,
                            families_validated=families, stop14_scope_validated=global_ok and stop_ok and scope_ok,
                            types_validated=global_ok and types_ok, power_ok=power_ok, thresholds_sha256=thr.sha256,
                            manifest_sha256=manifest.sha256(), missing_controls=tuple(missing),
                            specs_table=pd.DataFrame(specs_rows), type_metrics=tm,
                            types_validated_by_type={t: global_ok and ok for t, ok in types_by_type.items()})


def type_metric_rows(tm: dict, confidence: float = G2_CONFIDENCE, required_types: tuple[str, ...] | None = None) -> tuple[bool, list[dict]]:
    """Filas de validacion de los tipos. Deben cumplirse TODOS estos criterios:

    - tipos especificos incorrectos / detectados: limite superior <= ``TYPE_CONFUSION_LIMIT``;
    - capacidad GLOBAL (aciertos / detectados): limite inferior >= ``TYPE_CAPACITY_MIN``;
    - capacidad DE CADA TIPO (correccion de auditoria): limite inferior >= ``TYPE_CAPACITY_MIN`` en cada tipo exigido,
      con la confianza repartida por Bonferroni entre los tipos (``1 - (1-confidence)/K``). Un tipo que no se identifica
      nunca, o que no tiene detecciones, no se aprueba aunque la capacidad global sea alta.
    Abstenciones y detecciones se informan aparte. Devuelve (tipos validados, filas); cada fila por tipo lleva su estado."""
    conf_row = rate_row("POSITIVOS_CON_TIPO", "TIPOS_ESPECIFICOS_INCORRECTOS", tm["errores"], tm["detectados"], confidence, TYPE_CONFUSION_LIMIT)
    cap_row = rate_row("POSITIVOS_CON_TIPO", "CAPACIDAD_DE_IDENTIFICAR_TIPOS", tm["correctos"], tm["detectados"], confidence, TYPE_CAPACITY_MIN, upper=False)
    rows = [conf_row, cap_row,
            {**rate_row("POSITIVOS_CON_TIPO", "ABSTENCIONES", tm["abstenciones"], tm["detectados"], confidence, 1.0), "estado": "INFORMATIVO"},
            {**rate_row("POSITIVOS_CON_TIPO", "DETECTADOS", tm["detectados"], tm["mundos"], confidence, 1.0), "estado": "INFORMATIVO"}]
    types = tuple(required_types) if required_types is not None else tuple(sorted(tm["por_tipo"]))
    conf_type = 1.0 - (1.0 - confidence) / max(len(types), 1)
    per_type_ok = True
    for t in types:
        d = tm["por_tipo"].get(t, {"correctos": 0, "detectados": 0, "abstenciones": 0, "errores": 0})
        row = rate_row(f"POSITIVOS_TIPO_{t}", "CAPACIDAD_DE_IDENTIFICAR_TIPOS", d["correctos"], d["detectados"], conf_type, TYPE_CAPACITY_MIN, upper=False)
        if d["detectados"] == 0:
            row["estado"] = STATUS_NOT_VALIDATED                           # sin detecciones no hay evidencia de capacidad
        rows.append(row)
        per_type_ok &= row["estado"] == STATUS_COMPATIBLE
    return conf_row["estado"] == STATUS_COMPATIBLE and cap_row["estado"] == STATUS_COMPATIBLE and per_type_ok, rows


def final_decisions(ps: PanelStats, thr: FrozenThresholds, validation: ValidationResult | None,
                    annual_confirmed: dict | None = None, explained_by_sensitivity: dict | None = None) -> dict:
    """Decisiones publicables: aplica las consecuencias de la validacion. Sin validacion, todo es descriptivo.
    STOP-14 solo es definitivo con las comprobaciones anual y de sensibilidad explicitas (ver :func:`decide_stop14`)."""
    verify_frozen(thr)
    trace = {"umbrales_sha256": thr.sha256, "manifiesto_calibracion_sha256": dict(thr.calibration_info).get("manifiesto_sha256", ""),
             "alpha": thr.alpha, "k_star": thr.k_star, "margen_tipo": thr.type_margin,
             "validacion_umbrales_sha256": validation.thresholds_sha256 if validation else "",
             "manifiesto_validacion_sha256": validation.manifest_sha256 if validation else ""}
    if validation is None or validation.thresholds_sha256 != thr.sha256:
        dec = apply_rule(ps, thr)
        dec["estado"] = "DESCRIPTIVO_SIN_VALIDACION"
        dec["fechas"] = []
        dec["stop14"] = {**dec["stop14"], "alcance": STOP_HUMAN_REVIEW if dec["stop14"]["alcance"] != STOP_NOT_ACTIVE else STOP_NOT_ACTIVE}
        dec["trazabilidad"] = trace
        return dec
    if validation.manifest_sha256 != trace["manifiesto_calibracion_sha256"]:
        raise ValidationManifestError("la validacion se hizo con un manifiesto distinto del de la calibracion de estos umbrales")
    dec = apply_rule(ps, thr, power_ok=validation.power_ok)
    dec["trazabilidad"] = trace
    if not validation.global_validated:
        dec["estado"] = "SOLO_DESCRIPTIVO_ERROR_GLOBAL_NO_VALIDADO"
        dec["fechas"] = []
    else:
        dec["estado"] = STATUS_COMPATIBLE
    if not validation.dates_validated:
        dec["fechas"] = []
    for f, ok in validation.families_validated.items():
        dec["veredictos_familia"][f]["validacion"] = STATUS_COMPATIBLE if ok else STATUS_NOT_VALIDATED
    if not validation.types_validated:
        for f, v in dec["veredictos_familia"].items():
            if v["veredicto"] in NONSTABLE_VERDICTS:
                dec["veredictos_familia"][f] = {**v, "veredicto": V_UNSTABLE, "anotacion": A_TYPE_UNKNOWN}
    dec["stop14"] = decide_stop14(dec["veredictos_familia"], dec["veredictos_familia_estricta"], annual_confirmed, explained_by_sensitivity,
                                  procedure_validated=validation.global_validated, scope_validated=validation.stop14_scope_validated)
    return dec


# ---------------------------------------------------------------------------
# Tabla anual con bootstrap por bloques de jornadas; evaluabilidad de la curtosis
# ---------------------------------------------------------------------------

ANNUAL_BLOCK_SESSIONS: tuple[int, ...] = (1, 5, 20)
KURTOSIS_MAX_REL_HALFWIDTH = 0.25
KURTOSIS_MIN_YEARS = 4


def annual_table(panel: SessionPanel, n_boot: int, block_sessions: int, seed: int, spec: DetectorSpec = DetectorSpec(),
                 years: tuple[int, ...] = COMPLETE_YEARS) -> pd.DataFrame:
    """Estadistico x año con IC percentil 95 % por bootstrap de bloques de jornadas (dentro de cada año)."""
    suff = session_sufficient_stats(panel, spec.max_lag, spec.profile_bin_minutes)
    ref_lp = reference_log_profile(suff, spec.min_profile_bin_bars)
    yr = pd.to_datetime(panel.session_date).year.to_numpy()
    rng = np.random.default_rng([BASE_SEED, 99, seed, block_sessions])
    names = list(DETECTOR_STATISTICS) + list(DESCRIPTIVE_STATISTICS)
    rows = []
    for y in years:
        idx = np.flatnonzero(yr == y)
        if idx.size == 0:
            continue
        q01, q99, _ = _quantiles(panel, idx)
        point = statistics_from_sums(_sum_rows(suff, idx), q01, q99, ref_lp, spec.min_profile_bin_bars)
        boot = {k: [] for k in names}
        nb = int(math.ceil(idx.size / block_sessions))
        for _ in range(n_boot):
            starts = rng.integers(0, idx.size, size=nb)
            pick = idx[(starts[:, None] + np.arange(block_sessions)[None, :]).ravel() % idx.size][: idx.size]
            bq01, bq99, _ = _quantiles(panel, pick)
            st = statistics_from_sums(_sum_rows(suff, pick), bq01, bq99, ref_lp, spec.min_profile_bin_bars)
            for k in names:
                boot[k].append(st[k])
        for k in names:
            b = np.asarray(boot[k], dtype=float)
            b = b[np.isfinite(b)]
            lo, hi = (np.quantile(b, [0.025, 0.975]) if b.size else (np.nan, np.nan))
            rows.append({"anio": y, "estadistico": k, "familia": DETECTOR_STATISTICS.get(k, (DESCRIPTIVE_STATISTICS.get(k),))[0],
                         "valor": point[k], "ic95_inf": float(lo), "ic95_sup": float(hi), "bloque_jornadas": block_sessions, "n_boot": n_boot,
                         "n_jornadas": int(idx.size), "en_detector": k in DETECTOR_STATISTICS})
    return pd.DataFrame(rows)


def kurtosis_evaluability(annual: pd.DataFrame) -> dict:
    """La curtosis es evaluable solo si la semiamplitud relativa de su IC es <= 25 % en al menos 4 de los 5 años completos."""
    k = annual.loc[annual["estadistico"] == "curtosis_total"]
    rel = ((k["ic95_sup"] - k["ic95_inf"]) / 2.0 / k["valor"].abs()).to_numpy()
    years_ok = int(np.sum(np.isfinite(rel) & (rel <= KURTOSIS_MAX_REL_HALFWIDTH)))
    return {"evaluable": years_ok >= KURTOSIS_MIN_YEARS, "anios_que_cumplen": years_ok, "semiamplitud_relativa": rel.tolist()}


# ---------------------------------------------------------------------------
# TH30: log-precio frente a retornos, dentro de cada contrato
# ---------------------------------------------------------------------------

TH30_SUB_SESSIONS: tuple[int, ...] = (21, 10)
TH30_R_MORE_STABLE = "MOMENTOS_DE_R_RELATIVAMENTE_MAS_ESTABLES"
TH30_NEITHER = "TAMPOCO_LOS_MOMENTOS_DE_R_SON_ESTABLES"
TH30_INCONCLUSIVE = "NO_EVALUABLE"


def th30_within_contract(panel: SessionPanel, sub_sessions: int) -> pd.DataFrame:
    """Para cada contrato (serie SIN ajuste de rolls) y cada serie (ln C y r): subventanas consecutivas de ``sub_sessions``
    jornadas; desplazamiento de la media entre subventanas en desviaciones internas, y cociente max/min de desviaciones.
    Ambas medidas no tienen unidades, asi que las dos series se comparan en la misma escala."""
    if panel.close is None:
        raise ValueError("TH30 necesita el precio de cierre")
    lnc = np.log(np.where(panel.present, panel.close, np.nan))
    rows = []
    for c in pd.unique(panel.contract):
        sess = np.flatnonzero(panel.contract == c)
        n_sub = sess.size // sub_sessions
        if n_sub < 2:
            continue
        for name, mat in (("LN_C", lnc), ("R", panel.r)):
            means, sds = [], []
            for j in range(n_sub):
                v = mat[sess[j * sub_sessions:(j + 1) * sub_sessions]]
                v = v[np.isfinite(v)]
                means.append(v.mean())
                sds.append(v.std())
            means, sds = np.array(means), np.array(sds)
            rows.append({"contrato": c, "serie": name, "subventanas": n_sub, "jornadas_subventana": sub_sessions,
                         "desplazamiento_media_en_sd": float((means.max() - means.min()) / sds.mean()) if sds.mean() > 0 else np.nan,
                         "cociente_sd_max_min": float(sds.max() / sds.min()) if sds.min() > 0 else np.nan})
    return pd.DataFrame(rows)


def th30_summary(table: pd.DataFrame, r_stable_shift: float = 0.5) -> dict:
    """Conclusion LIMITADA: 'los momentos de r son relativamente mas estables que los de ln C por un factor X' o 'tampoco
    los de r lo son'. Nunca afirma estacionariedad. ``r_stable_shift`` (desplazamiento mediano en sd) es una convencion."""
    if table.empty:
        return {"conclusion": TH30_INCONCLUSIVE, "factor": float("nan")}
    med = table.groupby("serie")["desplazamiento_media_en_sd"].median()
    if "LN_C" not in med or "R" not in med or not med["R"] > 0:
        return {"conclusion": TH30_INCONCLUSIVE, "factor": float("nan")}
    factor = float(med["LN_C"] / med["R"])
    conclusion = TH30_R_MORE_STABLE if (factor > 1 and med["R"] <= r_stable_shift) else TH30_NEITHER
    return {"conclusion": conclusion, "factor": factor, "desplazamiento_mediano_ln_c": float(med["LN_C"]),
            "desplazamiento_mediano_r": float(med["R"]), "nota": "no demuestra estacionariedad; compara estabilidad relativa dentro de cada contrato"}
