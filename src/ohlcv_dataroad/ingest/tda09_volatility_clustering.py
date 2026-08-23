"""TDA-09 -- Dependencia en magnitud (volatility clustering).

Implementa la etapa TDA-09 del roadmap
(``docs/methodology/Tsay_OHLCV_analysis_roadmap.md``, seccion "TDA-09"):
mide si el TAMAÑO de los movimientos de MNQ tiene memoria (agrupamiento
de volatilidad), y compara esa memoria antes y despues de quitar el
patron determinista de reloj (TDA-06, ``s(m)``).

Resuelve TH19, TH20 y TH21 (ver informe,
``reports/mnq/TDA09_volatility_clustering.md``, generado automaticamente
por ``run_tda09.py`` a partir de este modulo -- nunca escrito a mano).

CORRECCION DE AUDITORIA (v1, posterior al cierre v0): una revision
detecto varios problemas de fondo que esta version corrige:

1. **Estabilidad temporal incompleta**: el roadmap pide año, segmento Y
   ventana rodante -- la v0 solo tenia año y segmento. Se añade
   persistencia por VENTANA RODANTE (mes calendario, misma convencion
   que ``rolling_rho1_by_month`` de TDA-08).
2. **Sin incertidumbre por grupo**: la v0 solo reportaba ``rho`` puntual
   por año/segmento. Se añade IC 95% bootstrap (rezago 1, costo acotado)
   para año, segmento Y ventana rodante, con un motor NUEVO
   (``bootstrap_rho_by_group_all_groups``) que amortiza el resampleo en
   UNA sola pasada por replica para TODOS los grupos a la vez (no una
   pasada por grupo -- ver docstring para el porque).
3. **Interpretacion causal indebida de TH21**: la v0 decia implicitamente
   que "el reloj explica X% del clustering". Se corrige: ``clock_attribution``
   es una comparacion DESCRIPTIVA de una metrica de energia antes/despues
   del ajuste -- nunca una descomposicion causal. La conclusion permitida
   es "el clustering sobrevive claramente al ajuste horario", no un
   porcentaje de atribucion causal.
4. **Rolls en log_hl no verificados explicitamente**: la v0 confiaba
   implicitamente en que TDA-04 certifico que todo roll coincide con un
   cambio de ``trading_date`` (cierto para las poblaciones filtradas por
   ``r_1m_valid``, pero ``log_hl_pop``/``log_hl_tilde_pop`` NO estan
   filtradas asi). Se añade ``compute_block_ids_with_contract``, que
   exige ADEMAS el mismo ``contract`` entre dos filas consecutivas --
   nunca depende silenciosamente de un invariante de otra etapa cuando se
   puede verificar aqui sin costo relevante.
5. **Lenguaje de "estabilidad"**: no se declara "la magnitud es estable".
   Se declara "la PRESENCIA del clustering es estable entre años y
   segmentos, aunque su INTENSIDAD varia" -- una afirmacion mas precisa
   y menos fuerte que la original.
6. **TH20 reforzado**: sigue sin afirmar memoria larga; la ambiguedad
   memoria-larga-vs-cambios-de-nivel queda expresamente abierta (TDA-14).
7. **Informe no se generaba automaticamente**: ``run_tda09.py`` ahora
   genera ``TDA09_volatility_clustering.md`` a partir de los resultados
   (``render_report``) -- una sola ejecucion produce TODO TDA-09.
8. **Progreso visible**: ``run_tda09_analysis`` acepta ``verbose=True`` y
   imprime 8 etapas con tiempo por etapa y tiempo acumulado; el runner
   añade una 9na etapa (persistencia de artefactos) con el mismo formato.
9. **Optimizaciones de rendimiento** (documentadas, sin cambiar
   metodologia): el bootstrap por grupo amortiza el resampleo entre
   todos los grupos de una misma agrupacion (año/segmento/mes) en una
   sola pasada por replica, en vez de resamplear una vez por grupo --
   evita un costo multiplicativo (grupos x replicas) sin cambiar ningun
   resultado (es exactamente la misma definicion, solo computada de
   forma mas eficiente).

Que es "magnitud" aqui, en una frase: NO es prediccion de direccion --
``|r_t|``, ``r_t^2`` y ``log(H_t/L_t)`` son proxies de cuan GRANDE fue el
movimiento, sin importar el signo. Si periodos agitados tienden a seguir
agitados (y periodos tranquilos, tranquilos), eso es "memoria en
magnitud" aunque la direccion siga siendo poco predecible (TDA-08 ya
cerro esa pregunta por separado: TH16/17/18).

Decision sobre r_t vs residuo/innovacion (seccion 8 del roadmap): TDA-08
(CLOSED) encontro que la dependencia lineal en la media es diminuta
(``beta_1~0.0059``, ~0.14 ticks) y `NOT SEPARABLE WITH OHLCV LAST`
(STOP-8a/8b). Por parsimonia (G4), este modulo usa ``r_1m`` directamente
como "innovacion" para TDA-09 -- no se ajusta un modelo AR nuevo. Se
ejecuta, como maximo, UNA sensibilidad barata (``mean_removal_sensitivity``):
comparar ACF(|r|) contra ACF(|r - beta_1*r_{t-1}|) en una grilla pequeña
de rezagos, para verificar (no solo asumir) que remover la media no
cambia materialmente la conclusion de magnitud.

TOPOLOGIA TEMPORAL (bloqueante): se reutilizan ``compute_acf``/
``bootstrap_rho``/``acf_by_group``/``g2_permutation_null_by_minute`` de
``tda08_linear_mean_dependence.py`` TAL CUAL -- ningun cambio a ese
modulo CLOSED. La construccion de bloques de continuidad usa una version
PROPIA de esta etapa, ``compute_block_ids_with_contract`` (ver punto 4
arriba) -- mas estricta que ``compute_block_ids`` de TDA-08 porque
tambien exige mismo ``contract``. Ninguna ACF de esta etapa cruza
``trading_date``, gaps o rolls de contrato.

CRUDO vs AJUSTADO POR RELOJ (seccion 7 del roadmap):

- ``|r_1m|``/``r_1m^2`` (crudo) vs ``|r_tilde|``/``r_tilde^2`` (ajustado)
  -- ``r_tilde = r_1m / s(m)`` ya fue construido y persistido por TDA-06
  (RETROSPECTIVO). ``|r_tilde| = |r_1m|/s(m)`` y ``r_tilde^2 =
  r_1m^2/s(m)^2`` son identidades algebraicas directas de esa definicion.
- ``log_hl`` (crudo) vs ``log_hl_tilde = log_hl / s(m)`` (ajustado) --
  transformacion NUEVA de esta etapa, dimensionalmente coherente porque
  ``s(m)`` fue estimado por TDA-06 usando exactamente ``log_hl`` como
  proxy elegido (verificado con ``verify_s_m_is_log_hl_proxy``).

Todo lo etiquetado "ajustado"/``r_tilde``/``log_hl_tilde`` hereda la
etiqueta ``RETROSPECTIVO`` de TDA-06 -- NUNCA se presenta como una
transformacion causal disponible en produccion.

Que NO hace este modulo (deliberadamente, fuera de alcance de TDA-09):

- No diseña ningun modelo predictivo, feature, target ni señal.
- No ajusta ARCH/GARCH/EGARCH/TGARCH como modelo productivo -- el LM de
  Engle se implementa UNICAMENTE como test/diagnostico (TH19).
- No ejecuta EVT, BDS, TAR/STAR, Markov-Switching, fractional
  differencing (no se estima ``d``), quantile regression.
- No abre ``data/raw/`` ni ``config.holdout_files``.
- No modifica ningun artefacto de TDA-00..TDA-08/TDA08-H.
- No inicia TDA-10 ni TDA-11.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ohlcv_dataroad.config import SnapshotConfig
from ohlcv_dataroad.ingest.holdout_guard import (
    validate_last_timestamps_before_boundary,
    validate_research_holdout_disjoint,
)
from ohlcv_dataroad.ingest.tda06_intraday_calendar_profile import attach_calendar_fields
from ohlcv_dataroad.ingest.tda07_marginal_distribution import (
    RTildeInvariantError,
    TimestampAlignmentError,
    assign_segment_label,
    build_r1m_population,
    build_r_tilde_population,
    load_r_tilde,
    load_segmentation_cutoffs,
    load_tda04_inputs,
    verify_r_tilde_invariants,
    verify_timestamp_alignment,
)
from ohlcv_dataroad.ingest.tda08_linear_mean_dependence import (
    COMPLETE_YEARS,
    PARTIAL_YEARS,
    acf_by_group,
    annotate_portmanteau_calibration,
    bartlett_se,
    bootstrap_rho,
    compute_acf,
    compute_portmanteau_q,
    g2_null1_calibration_summary,
    g2_null1_portmanteau_summary,
    g2_permutation_null_by_minute,
)

__all__ = [
    "SMProxyMismatchError",
    "MAX_LAG_MAGNITUDE",
    "BOOTSTRAP_LAGS_MAGNITUDE",
    "PORTMANTEAU_M_MAGNITUDE",
    "G2_LAGS",
    "STABILITY_LAGS",
    "ARCH_LM_ORDERS",
    "TH21_ENERGY_M",
    "STOP9_FRACTION_REMOVED_THRESHOLD",
    "CLOCK_FLATNESS_RATIO_THRESHOLD",
    "TH20_MIN_FRACTION_SURVIVES",
    "TH19_MIN_MATERIAL_RHO",
    "TH21_SURVIVES_GENUINE_THRESHOLD",
    "TH21_SURVIVES_ARTIFACT_THRESHOLD",
    "GROUP_BOOTSTRAP_N_BOOT",
    "GROUP_BOOTSTRAP_LAG",
    "ANALYSIS_STAGES",
    "TOTAL_STAGES",
    "build_log_hl_population",
    "verify_s_m_is_log_hl_proxy",
    "build_log_hl_tilde",
    "compute_block_ids_with_contract",
    "mean_removal_sensitivity",
    "clock_profile_flatness",
    "g2_global_permutation_null",
    "block_relative_position",
    "engle_lm_statistic",
    "calibrate_engle_lm",
    "same_clock_next_trading_day",
    "dependence_energy",
    "clock_attribution",
    "decide_stop9",
    "decide_th19",
    "classify_th21",
    "th20_status_label",
    "bootstrap_rho_by_group_all_groups",
    "year_month_labels",
    "decay_form_diagnostic",
    "TDA09Result",
    "run_tda09_analysis",
]

# ---------------------------------------------------------------------------
# Convenciones PREDECLARADAS (nunca elegidas despues de ver el resultado)
# ---------------------------------------------------------------------------

# "Varios cientos de rezagos" (roadmap TH19): 600 minutos = 10 horas.
MAX_LAG_MAGNITUDE = 600

BOOTSTRAP_LAGS_MAGNITUDE = (1, 2, 5, 10, 20, 30, 60, 120, 240, 360, 480, 600)
PORTMANTEAU_M_MAGNITUDE = (10, 20, 40, 60, 120, 240, 480, 600)
G2_NULL_MAX_LAG = 60
G2_LAGS = tuple(k for k in BOOTSTRAP_LAGS_MAGNITUDE if k <= G2_NULL_MAX_LAG)
# ``g2_null1_portmanteau_summary`` (TDA-08) asume columnas de rezago
# CONTIGUAS (1..N) para poder tomar el prefijo ``[:m]`` -- se calcula el
# null de permutacion sobre el rango CONTIGUO completo, y se subselecciona
# por indice para la tabla de calibracion de ``rho`` en la grilla dispersa
# ``G2_LAGS`` -- exactamente el patron de TDA-08 (``bootstrap_lag_idx``).
G2_FULL_LAGS = tuple(range(1, G2_NULL_MAX_LAG + 1))

# Grilla pequeña y predeclarada para persistencia por año/segmento/ventana
# rodante (mensual): "no hace falta recalcular cientos de lags para cada
# subdivision si una tabla pequeña de lags predeclarados responde
# correctamente la pregunta".
STABILITY_LAGS = (1, 5, 20, 60)

# LM de Engle: cantidad pequeña y predeclarada de ordenes.
ARCH_LM_ORDERS = (1, 5, 20)

# Ventana predeclarada para la metrica de "cantidad total de dependencia"
# (TH21): 240 minutos = 4 horas.
TH21_ENERGY_M = 240

# STOP-9: si la fraccion de dependencia que SOBREVIVE al ajuste cae por
# debajo de este umbral (es decir, se remueve >=90%) en TODAS las
# variables evaluadas, se declara STOP-9 ACTIVADO.
STOP9_FRACTION_REMOVED_THRESHOLD = 0.90

# Umbral para decidir si el perfil de reloj de la serie AJUSTADA quedo
# "efectivamente removido" -- el null POR MINUTO (mas conservador) sigue
# siendo el PRINCIPAL en todos los casos; el null GLOBAL solo se reporta
# como sensibilidad secundaria para la serie ajustada.
CLOCK_FLATNESS_RATIO_THRESHOLD = 0.20

# TH20 solo se habilita si una fraccion no trivial de la dependencia
# sobrevive al ajuste.
TH20_MIN_FRACTION_SURVIVES = 0.05

# TH19: materialidad minima de rho en el rezago 1 de |r| crudo (G5 --
# "no solo estadisticamente distinto de cero"). Predeclarado.
TH19_MIN_MATERIAL_RHO = 0.02

# TH21 (clasificacion DESCRIPTIVA, nunca causal -- ver clock_attribution):
# umbrales predeclarados sobre la fraccion de energia de dependencia que
# SOBREVIVE al ajuste, para clasificar la comparacion en una de tres
# etiquetas descriptivas.
TH21_SURVIVES_GENUINE_THRESHOLD = 0.5
TH21_SURVIVES_ARTIFACT_THRESHOLD = 0.1

DEFAULT_N_BOOT = 300
DEFAULT_BOOTSTRAP_SEED = 0
DEFAULT_N_PERM = 200
PERMUTATION_SEED = 1
GLOBAL_PERMUTATION_SEED = 5
SYNTHETIC_NULL_SEED = 2
N_PERM_ARCH_LM = 60
ARCH_LM_PERMUTATION_SEED = 3
NULL_PERCENTILE = 97.5

MIN_N_FOR_GROUP = 200

# Bootstrap por grupo (año/segmento/ventana rodante mensual): n_boot
# reducido (presupuesto computacional documentado -- ver
# ``bootstrap_rho_by_group_all_groups``) y UN SOLO rezago (el mas
# informativo para la pregunta de estabilidad: ¿el clustering de
# rezago-1 aparece en todos los subperiodos?). Los rezagos 5/20/60 del
# punto estimado (STABILITY_LAGS) no llevan IC bootstrap por costo.
GROUP_BOOTSTRAP_N_BOOT = 50
GROUP_BOOTSTRAP_LAG = 1
GROUP_BOOTSTRAP_SEED = 6
GROUP_BOOTSTRAP_MIN_REPLICAS = 10

ANALYSIS_STAGES = 8
TOTAL_STAGES = 9  # 8 de analisis (run_tda09_analysis) + 1 de persistencia de artefactos (run_tda09.py)


class SMProxyMismatchError(Exception):
    """``tda06_s_m.parquet`` no fue construido con ``log_hl`` como proxy.

    TDA-06 (CLOSED) eligio ``log_hl`` como proxy de ``s(m)`` (informe,
    seccion 11). El ajuste ``log_hl_tilde = log_hl / s(m)`` de esta etapa
    depende de esa eleccion especifica para ser dimensionalmente
    coherente -- si una version futura de TDA-06 cambiara de proxy sin
    que esta etapa se actualizara, aplicar la misma formula seria una
    transformacion arbitraria. Comprobacion bloqueante, nunca asumida.
    """


# ---------------------------------------------------------------------------
# Progreso visible -- 8 etapas de analisis + tiempo acumulado
# ---------------------------------------------------------------------------

class _StageTimer:
    """Imprime ``[TDA09 i/N] mensaje...`` y, al abrir la siguiente etapa (o
    al llamar ``finish()``), el tiempo que tomo la etapa anterior y el
    tiempo total acumulado desde el inicio. Guarda todos los tiempos por
    etapa en ``self.stage_timings`` para incluirlos en el informe.
    """

    def __init__(self, total_stages: int = ANALYSIS_STAGES, verbose: bool = True):
        self.total = total_stages
        self.verbose = verbose
        self.t_start = time.perf_counter()
        self.stage_timings: dict[str, float] = {}
        self._last = self.t_start
        self._i = 0
        self._current_name = ""

    def stage(self, name: str) -> None:
        now = time.perf_counter()
        if self._i > 0:
            elapsed_stage = now - self._last
            self.stage_timings[self._current_name] = elapsed_stage
            if self.verbose:
                total_elapsed = now - self.t_start
                print(f"[TDA09 {self._i}/{self.total}] {self._current_name} -- completado en {elapsed_stage:.1f}s (total: {total_elapsed:.1f}s)")
        self._i += 1
        self._current_name = name
        self._last = now
        if self.verbose:
            print(f"[TDA09 {self._i}/{self.total}] {name}...")

    def finish(self) -> dict[str, float]:
        now = time.perf_counter()
        elapsed_stage = now - self._last
        self.stage_timings[self._current_name] = elapsed_stage
        if self.verbose:
            total_elapsed = now - self.t_start
            print(f"[TDA09 {self._i}/{self.total}] {self._current_name} -- completado en {elapsed_stage:.1f}s (total: {total_elapsed:.1f}s)")
        self.stage_timings["_total_analysis"] = now - self.t_start
        return self.stage_timings


# ---------------------------------------------------------------------------
# Topologia -- bloques de continuidad que ADEMAS exigen mismo contrato
# ---------------------------------------------------------------------------

def compute_block_ids_with_contract(
    timestamp: pd.Series, trading_date: pd.Series, contract: pd.Series, expected_delta_seconds: float,
) -> np.ndarray:
    """Como ``compute_block_ids`` (TDA-08) pero exige ADEMAS el mismo
    ``contract`` entre dos filas consecutivas para pertenecer al mismo
    bloque de continuidad -- defensa EXPLICITA e independientemente
    verificable de que ninguna ACF cruza un roll de contrato.

    Por que hace falta esto ademas de lo que TDA-08 ya hacia: TDA-04
    (informe, seccion 5) certifico que todo roll coincide con un cambio
    de ``trading_date``, asi que para poblaciones filtradas a
    ``r_1m_valid=True`` (que ya excluyen la fila ``ROLL_BOUNDARY``) esa
    proteccion era implicita. Mismo esta comprobacion se aplica aqui de
    forma UNIFORME a las 4 poblaciones de esta etapa -- en particular a
    ``log_hl``/``log_hl_tilde``, que incluyen TODAS las barras admisibles
    (no solo ``r_1m_valid=True``) y por tanto NO heredan esa proteccion
    implicita. Nunca se depende silenciosamente de un invariante de otra
    etapa cuando se puede verificar directamente aqui, sin costo
    adicional relevante (una comparacion de arrays mas).
    """
    ts = pd.to_datetime(timestamp)
    order = np.argsort(ts.to_numpy())
    ts_sorted = ts.to_numpy()[order]
    td_sorted = np.asarray(trading_date)[order]
    contract_sorted = np.asarray(contract)[order]

    n = len(ts_sorted)
    if n == 0:
        return np.array([], dtype=int)

    delta_seconds = np.diff(ts_sorted).astype("timedelta64[ns]").astype(np.int64) / 1e9
    same_date = td_sorted[1:] == td_sorted[:-1]
    same_contract = contract_sorted[1:] == contract_sorted[:-1]
    continues = (np.abs(delta_seconds - expected_delta_seconds) < 1e-6) & same_date & same_contract
    new_block = np.concatenate([[True], ~continues])
    block_ids_sorted = np.cumsum(new_block) - 1

    block_ids = np.empty(n, dtype=int)
    block_ids[order] = block_ids_sorted
    return block_ids


# ---------------------------------------------------------------------------
# Poblacion de log_hl -- TODAS las barras admisibles (log_hl nunca es NaN
# por construccion, TDA-04; no depende de r_1m_valid)
# ---------------------------------------------------------------------------

def build_log_hl_population(variables: pd.DataFrame, validity: pd.DataFrame, cutoffs: list[int]) -> tuple[pd.DataFrame, list[str]]:
    """Poblacion de analisis de ``log_hl`` crudo: TODAS las filas admisibles (no solo ``r_1m_valid``).

    Reutiliza ``attach_calendar_fields`` de TDA-06 (misma logica DST-aware
    de ``minute_of_day``/``year_ny`` que TDA-06/07/08 ya certificaron) SIN
    filtrar por ``r_1m_valid`` -- a diferencia de ``build_r1m_population``
    (TDA-07), porque ``log_hl`` no necesita una barra anterior comparable
    (TDA-04: "nunca es NaN por una regla de no-cruce").
    """
    df = attach_calendar_fields(variables, validity)
    df["segment_label"], labels = assign_segment_label(df["minute_of_day"], cutoffs)
    return df, labels


def verify_s_m_is_log_hl_proxy(s_m_table: pd.DataFrame) -> None:
    """Invariante bloqueante: ``tda06_s_m.parquet`` debe provenir EXCLUSIVAMENTE del proxy ``log_hl``."""
    proxies = s_m_table["proxy"].unique()
    if len(proxies) != 1 or proxies[0] != "log_hl":
        raise SMProxyMismatchError(
            f"tda06_s_m.parquet tiene proxy(es) {list(proxies)!r}, se esperaba exclusivamente 'log_hl' "
            "(TDA-06 CLOSED, informe seccion 11). TDA-09 se detiene sin construir log_hl_tilde -- "
            "aplicar 'log_hl / s(m)' solo es dimensionalmente coherente si s(m) se estimo sobre log_hl."
        )


def build_log_hl_tilde(log_hl_pop: pd.DataFrame, s_m_table: pd.DataFrame) -> pd.DataFrame:
    """``log_hl_tilde = log_hl / s(minute_of_day)`` -- ajuste RETROSPECTIVO, dimensionalmente coherente.

    Division protegida (``s_m=0``/no finito -> ``NaN`` explicito, nunca
    ``inf``), misma mecanica que ``tda06_intraday_calendar_profile.build_r_tilde``.
    Requiere que ``verify_s_m_is_log_hl_proxy`` ya haya pasado.
    """
    s_map = s_m_table.set_index("minute_of_day")["s_m"]
    s_m = log_hl_pop["minute_of_day"].map(s_map).to_numpy(dtype=float)
    log_hl = log_hl_pop["log_hl"].to_numpy(dtype=float)
    valid = np.isfinite(s_m) & (s_m > 0) & np.isfinite(log_hl)
    log_hl_tilde = np.where(valid, log_hl / np.where(s_m == 0, np.nan, s_m), np.nan)

    out = log_hl_pop.copy()
    out["s_m"] = s_m
    out["log_hl_tilde"] = log_hl_tilde
    out["label"] = "RETROSPECTIVO"
    return out


# ---------------------------------------------------------------------------
# Decision sobre r_t vs innovacion -- sensibilidad barata unica, NO un
# nuevo modelo de media
# ---------------------------------------------------------------------------

def mean_removal_sensitivity(r1m_pop: pd.DataFrame, block_ids_raw: np.ndarray, lags: tuple[int, ...] = STABILITY_LAGS) -> pd.DataFrame:
    """Compara ACF(|r|) contra ACF(|r - beta_1*r_{t-1}|) -- ¿remover la media cambia la conclusion de magnitud?

    ``beta_1`` se recalcula aqui mismo via ``compute_acf`` a rezago 1. El
    residuo ``e_t = r_t - beta_1*r_{t-1}`` solo se define donde existe un
    par valido del MISMO bloque de continuidad (nunca se fabrica un
    vecino); los bloques de la subpoblacion de ``e`` se recalculan con
    ``compute_block_ids_with_contract`` sobre sus propios timestamps
    (bloques mas cortos automaticamente donde se elimino una fila).
    """
    values = r1m_pop["r_1m"].to_numpy(dtype=float)
    acf1 = compute_acf(values, block_ids_raw, max_lag=1)
    beta_1 = float(acf1.loc[0, "beta"]) if bool(acf1.loc[0, "estimable"]) else 0.0

    same_block_prev = np.concatenate([[False], block_ids_raw[1:] == block_ids_raw[:-1]])
    prev_values = np.concatenate([[np.nan], values[:-1]])
    e = values - beta_1 * prev_values

    sub = r1m_pop.loc[same_block_prev].copy()
    sub["e"] = e[same_block_prev]
    sub_block_ids = compute_block_ids_with_contract(sub["timestamp"], sub["trading_date"], sub["contract"], 60.0)

    max_lag = max(lags)
    abs_r_acf = compute_acf(np.abs(values), block_ids_raw, max_lag).set_index("lag")
    abs_e_acf = compute_acf(np.abs(sub["e"].to_numpy(dtype=float)), sub_block_ids, max_lag).set_index("lag")

    rows = []
    for k in lags:
        rows.append({
            "lag": k,
            "rho_abs_r": float(abs_r_acf.loc[k, "rho"]) if k in abs_r_acf.index and bool(abs_r_acf.loc[k, "estimable"]) else np.nan,
            "rho_abs_e": float(abs_e_acf.loc[k, "rho"]) if k in abs_e_acf.index and bool(abs_e_acf.loc[k, "estimable"]) else np.nan,
            "beta_1_used": beta_1,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# G2 -- perfil de reloj de la serie AJUSTADA: ¿esta "efectivamente
# removido"?
# ---------------------------------------------------------------------------

def clock_profile_flatness(values: np.ndarray, minute_of_day: np.ndarray) -> float:
    """``std(mediana por minuto) / mean(mediana por minuto)`` -- cuanto varia el nivel tipico entre minutos."""
    df = pd.DataFrame({"m": np.asarray(minute_of_day), "v": np.asarray(values, dtype=float)})
    med = df.groupby("m")["v"].median().dropna()
    mean_level = float(med.mean())
    if not np.isfinite(mean_level) or mean_level == 0:
        return float("nan")
    return float(med.std() / mean_level)


def g2_global_permutation_null(
    values: np.ndarray, block_ids: np.ndarray, lags: tuple[int, ...],
    n_perm: int = DEFAULT_N_PERM, seed: int = GLOBAL_PERMUTATION_SEED,
) -> np.ndarray:
    """Null SECUNDARIO para la serie AJUSTADA: permutacion GLOBAL (sin condicionar por minuto).

    Solo defendible si el perfil de reloj de la serie ajustada ya quedo
    efectivamente removido (``clock_profile_flatness`` bajo) -- por eso
    NUNCA es el null PRINCIPAL (ese rol lo cumple siempre
    ``g2_permutation_null_by_minute``, mas conservador).
    """
    rng = np.random.default_rng(seed)
    max_lag = max(lags)
    out = np.full((n_perm, len(lags)), np.nan)
    for p in range(n_perm):
        permuted = rng.permutation(values)
        acf_tab = compute_acf(permuted, block_ids, max_lag).set_index("lag")
        for li, k in enumerate(lags):
            if k in acf_tab.index:
                out[p, li] = acf_tab.loc[k, "rho"]
    return out


# ---------------------------------------------------------------------------
# LM de Engle -- diagnostico/test, NUNCA modelo ajustado
# ---------------------------------------------------------------------------

def block_relative_position(block_ids: np.ndarray) -> np.ndarray:
    """Posicion (0-index) de cada fila DENTRO de su propio bloque de continuidad."""
    return pd.Series(np.arange(len(block_ids))).groupby(np.asarray(block_ids)).cumcount().to_numpy()


def engle_lm_statistic(x: np.ndarray, block_ids: np.ndarray, order: int) -> dict:
    """LM de Engle para efecto ARCH de orden ``order`` sobre ``x`` (tipicamente un retorno/innovacion).

    Regresion OLS: ``x_t^2`` sobre una constante y ``x_{t-1}^2 .. x_{t-order}^2``,
    restringida a filas con una cadena COMPLETA de ``order`` rezagos
    validos DENTRO del mismo bloque de continuidad.
    """
    x = np.asarray(x, dtype=float)
    pos = block_relative_position(block_ids)
    eligible = pos >= order
    n_eff = int(eligible.sum())
    if n_eff < order + 5:
        return {"order": order, "n_eff": n_eff, "LM": float("nan"), "R2": float("nan"), "estimable": False}

    idx = np.where(eligible)[0]
    y = x[idx] ** 2
    cols = [np.ones(n_eff)]
    for j in range(1, order + 1):
        cols.append(x[idx - j] ** 2)
    X = np.column_stack(cols)

    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta_hat
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    lm = n_eff * r2
    return {"order": order, "n_eff": n_eff, "LM": float(lm), "R2": float(r2), "estimable": True}


def _permute_within_minute(values: np.ndarray, minute_of_day: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permuta ``values`` DENTRO de cada ``minute_of_day``."""
    values = np.asarray(values, dtype=float)
    minute_of_day = np.asarray(minute_of_day)
    order = np.argsort(minute_of_day, kind="stable")
    sorted_minute = minute_of_day[order]
    boundaries = np.searchsorted(sorted_minute, np.arange(1441))
    permuted = values.copy()
    for i in range(1440):
        lo, hi = boundaries[i], boundaries[i + 1]
        if hi - lo > 1:
            g = order[lo:hi]
            permuted[g] = rng.permutation(values[g])
    return permuted


def calibrate_engle_lm(
    x: np.ndarray, block_ids: np.ndarray, minute_of_day: np.ndarray, order: int,
    n_perm: int = N_PERM_ARCH_LM, seed: int = ARCH_LM_PERMUTATION_SEED,
) -> dict:
    """Calibracion EMPIRICA del LM de Engle: percentil del ``LM`` real dentro de ``n_perm`` replicas permutadas por minuto."""
    real = engle_lm_statistic(x, block_ids, order)
    if not real["estimable"]:
        return {**real, "n_perm": 0, "null_p50": float("nan"), "null_p975": float("nan"), "percentile_of_real": float("nan"), "exceeds_calibration_threshold": False}

    rng = np.random.default_rng(seed)
    null_lm = np.full(n_perm, np.nan)
    for p in range(n_perm):
        permuted = _permute_within_minute(x, minute_of_day, rng)
        result = engle_lm_statistic(permuted, block_ids, order)
        null_lm[p] = result["LM"]
    finite = null_lm[~np.isnan(null_lm)]
    pct = float((finite < real["LM"]).mean() * 100.0) if finite.size else float("nan")
    return {
        **real, "n_perm": int(finite.size),
        "null_p50": float(np.median(finite)) if finite.size else float("nan"),
        "null_p975": float(np.percentile(finite, NULL_PERCENTILE)) if finite.size else float("nan"),
        "percentile_of_real": pct,
        "exceeds_calibration_threshold": bool(pct >= NULL_PERCENTILE) if not np.isnan(pct) else False,
    }


# ---------------------------------------------------------------------------
# Diagnostico "same-clock-position" entre jornadas -- SEPARADO y distinto
# de una ACF continua de 1.380 minutos
# ---------------------------------------------------------------------------

def same_clock_next_trading_day(
    pop_df: pd.DataFrame, value_col: str, n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict:
    """Correlacion, agrupada a traves de TODOS los minutos, entre el valor de ``value_col`` en (trading_date=d, minuto=m) y en (trading_date=SIGUIENTE dia de negociacion presente, minuto=m).

    Explicitamente NO es una ACF continua de 1.380/2.760 minutos (ese
    rezago es ``NOT_ESTIMABLE`` bajo el motor continuo). Empareja el
    MISMO minuto del reloj en dos jornadas de negociacion CONSECUTIVAS
    presentes en los datos. El bootstrap remuestrea PARES DE DIAS
    CONSECUTIVOS completos (con reemplazo).
    """
    df = pop_df[["trading_date", "minute_of_day", value_col]].copy()
    dates = sorted(df["trading_date"].unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}
    df["date_idx"] = df["trading_date"].map(date_to_idx)

    pivot = df.pivot_table(index="date_idx", columns="minute_of_day", values=value_col, aggfunc="first")
    pivot = pivot.reindex(range(len(dates)))
    mat = pivot.to_numpy(dtype=float)

    a = mat[:-1]
    b = mat[1:]
    n_day_pairs = a.shape[0]

    def _pooled_rho(aa: np.ndarray, bb: np.ndarray) -> tuple[float, int]:
        mask = np.isfinite(aa) & np.isfinite(bb)
        x = aa[mask]
        y = bb[mask]
        n_pairs = int(x.size)
        if n_pairs < 2:
            return float("nan"), n_pairs
        ex = x - x.mean()
        ey = y - y.mean()
        denom = np.sqrt(np.sum(ex ** 2) * np.sum(ey ** 2))
        rho = float(np.sum(ex * ey) / denom) if denom > 0 else float("nan")
        return rho, n_pairs

    rho, n_pairs = _pooled_rho(a, b)

    rng = np.random.default_rng(seed)
    boot = np.full(n_boot, np.nan)
    for bi in range(n_boot):
        sampled = rng.integers(0, n_day_pairs, size=n_day_pairs)
        r_b, _ = _pooled_rho(a[sampled], b[sampled])
        boot[bi] = r_b
    finite = boot[~np.isnan(boot)]
    ci_lo, ci_hi = (float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))) if finite.size else (float("nan"), float("nan"))

    return {"rho": rho, "n_pairs": n_pairs, "n_day_pairs": int(n_day_pairs), "ci_lo": ci_lo, "ci_hi": ci_hi}


# ---------------------------------------------------------------------------
# TH21 -- comparacion DESCRIPTIVA de energia de dependencia (NUNCA una
# descomposicion causal de "cuanto explica el reloj")
# ---------------------------------------------------------------------------

def dependence_energy(acf_tab: pd.DataFrame, m: int) -> float:
    """``Q(m) = sum_{k=1}^{m} n_pairs_k * rho_k^2`` -- la MISMA metrica de portmanteau de TDA-08, reutilizada como "cantidad total de dependencia" no negativa."""
    q_tab = compute_portmanteau_q(acf_tab, (m,))
    row = q_tab.loc[q_tab["m"] == m].iloc[0]
    return float(row["Q"]) if bool(row["estimable"]) else float("nan")


def clock_attribution(acf_raw: pd.DataFrame, acf_adjusted: pd.DataFrame, m: int = TH21_ENERGY_M) -> dict:
    """Compara la energia de dependencia (``Q(m)``) ANTES y DESPUES del ajuste por ``s(m)`` -- una comparacion DESCRIPTIVA, NO una descomposicion causal.

    IMPORTANTE (correccion de auditoria v1): ``fraction_removed``/
    ``fraction_survives`` son un CAMBIO DESCRIPTIVO de una metrica de
    energia entre dos series (cruda y ajustada) -- NUNCA se interpretan
    como "el reloj explica X% del clustering". Esa lectura implicaria una
    descomposicion causal aditiva (energia_total = energia_reloj +
    energia_dinamica) que esta metrica no garantiza: `Q(m)` no es lineal
    en la transformacion `s(m)`, y ambas series comparten la MISMA
    dinamica subyacente ademas de diferir en escala por minuto -- por eso
    la fraccion puede incluso ser NEGATIVA (la energia sube tras
    ajustar) sin que eso signifique que el reloj "aportaba menos que
    cero". La UNICA lectura valida es: "la energia de dependencia
    cambio de esta manera al estandarizar por el perfil de reloj" -- y,
    cuando la fraccion que sobrevive es alta, "el clustering sobrevive
    claramente al ajuste horario y el reloj no explica la mayor parte de
    la persistencia observada" (ver ``classify_th21``).
    """
    q_raw = dependence_energy(acf_raw, m)
    q_adj = dependence_energy(acf_adjusted, m)
    if not np.isfinite(q_raw) or q_raw <= 0:
        return {"m": m, "Q_raw": q_raw, "Q_adjusted": q_adj, "fraction_removed": float("nan"), "fraction_survives": float("nan")}
    fraction_survives = q_adj / q_raw
    return {
        "m": m, "Q_raw": q_raw, "Q_adjusted": q_adj,
        "fraction_removed": 1.0 - fraction_survives, "fraction_survives": fraction_survives,
    }


def decide_stop9(attributions: list[dict], threshold: float = STOP9_FRACTION_REMOVED_THRESHOLD) -> dict:
    """STOP-9: lectura HOLISTICA (misma filosofia que ``decide_stop6`` de TDA-06) sobre varias variables de magnitud.

    STOP-9 se declara ACTIVADO solo si TODAS las variables evaluadas
    muestran una fraccion removida por encima del umbral (colapso
    practicamente completo en cada una).
    """
    valid = [a for a in attributions if np.isfinite(a.get("fraction_removed", float("nan")))]
    if not valid:
        return {"stop9_activated": False, "reason": "NINGUNA variable produjo una fraccion valida (denominador inestable) -- STOP-9 no se puede evaluar, se reporta NO ACTIVADO por defecto conservador."}
    all_collapse = all(a["fraction_removed"] >= threshold for a in valid)
    return {
        "stop9_activated": bool(all_collapse),
        "threshold": threshold,
        "fractions_removed": {a.get("variable", "?"): a["fraction_removed"] for a in valid},
        "reason": (
            f"Las {len(valid)} variable(s) evaluada(s) superan el umbral de colapso ({threshold})."
            if all_collapse else
            "Al menos una variable conserva una fraccion de dependencia por debajo del umbral de colapso -- STOP-9 NO ACTIVADO."
        ),
    }


def classify_th21(clock_attribution_rows: list[dict], genuine_threshold: float = TH21_SURVIVES_GENUINE_THRESHOLD, artifact_threshold: float = TH21_SURVIVES_ARTIFACT_THRESHOLD) -> dict:
    """Clasificacion DESCRIPTIVA (nunca causal) de TH21 a partir de la fraccion de energia que SOBREVIVE al ajuste.

    ``CLUSTERING_GENUINO``: la fraccion que sobrevive es alta (>= ``genuine_threshold``)
    en TODAS las variables evaluadas -- lectura permitida: "el clustering
    sobrevive claramente al ajuste horario y el reloj no explica la mayor
    parte de la persistencia observada".
    ``ARTEFACTO_DE_ESTACIONALIDAD``: la fraccion que sobrevive es baja
    (<= ``artifact_threshold``) en TODAS.
    ``MIXTO``: ni lo uno ni lo otro, o resultados dispares entre variables.
    Nunca se reporta un porcentaje como "atribuible al reloj" -- ver
    docstring de ``clock_attribution``.
    """
    valid = [a for a in clock_attribution_rows if np.isfinite(a.get("fraction_survives", float("nan")))]
    if not valid:
        return {"verdict": "MIXTO", "reason": "Fraccion no estimable para ninguna variable evaluada -- se reporta MIXTO por defecto conservador."}
    survives = [a["fraction_survives"] for a in valid]
    if min(survives) >= genuine_threshold:
        verdict = "CLUSTERING_GENUINO"
    elif max(survives) <= artifact_threshold:
        verdict = "ARTEFACTO_DE_ESTACIONALIDAD"
    else:
        verdict = "MIXTO"
    return {
        "verdict": verdict,
        "fractions_survives": {a.get("variable", "?"): a["fraction_survives"] for a in valid},
        "genuine_threshold": genuine_threshold, "artifact_threshold": artifact_threshold,
    }


def decide_th19(g2_null1_calibration: pd.DataFrame, materiality_threshold: float = TH19_MIN_MATERIAL_RHO) -> dict:
    """Clasifica TH19 de forma programatica: exige AMBAS cosas (G5) -- que ``|r|`` crudo en el rezago 1 supere la calibracion G2, Y que su magnitud sea material (no solo distinta de cero)."""
    rho_cal = g2_null1_calibration
    if "kind_of_row" in rho_cal.columns:
        rho_cal = rho_cal[rho_cal["kind_of_row"] == "rho_calibration"]
    row = rho_cal[(rho_cal["variable"] == "abs_r") & (rho_cal["raw_adjusted"] == "raw") & (rho_cal["lag"] == 1)]
    if row.empty:
        return {"verdict": "RESULTADO_INDETERMINADO", "reason": "No hay calibracion G2 disponible para |r| crudo en el rezago 1."}
    exceeds = bool(row["exceeds_calibration_threshold"].iloc[0])
    rho_1 = float(row["rho_real_abs"].iloc[0])
    material = rho_1 >= materiality_threshold
    if exceeds and material:
        verdict = "VOLATILITY_CLUSTERING_DETECTABLE"
    elif not material:
        verdict = "NO_VOLATILITY_CLUSTERING_MATERIAL"
    else:
        verdict = "RESULTADO_INDETERMINADO"
    return {"verdict": verdict, "rho_1_abs_r_raw": rho_1, "exceeds_g2_threshold": exceeds, "materiality_threshold": materiality_threshold}


def th20_status_label(th20_enabled: bool, decay_diagnostics: dict[str, dict]) -> str:
    """``RESUELTA``/``NO_HABILITADA``/``INDETERMINADA`` para TH20 -- nunca afirma memoria larga (ver ``decay_form_diagnostic``)."""
    if not th20_enabled:
        return "NO_HABILITADA"
    if not decay_diagnostics or not all(d.get("estimable") for d in decay_diagnostics.values()):
        return "INDETERMINADA"
    return "RESUELTA"


# ---------------------------------------------------------------------------
# TH20 -- forma del decaimiento (log-log vs semi-log), solo si TH19/TH21
# habilitan la pregunta. Diagnostico DESCRIPTIVO -- NUNCA estima memoria
# larga ni un parametro `d`; la ambiguedad memoria-larga-vs-cambios-de-
# nivel queda EXPLICITAMENTE abierta para TDA-14.
# ---------------------------------------------------------------------------

def decay_form_diagnostic(acf_tab: pd.DataFrame) -> dict:
    """Compara un ajuste log-log (``log(rho_k) ~ log(k)``) contra uno semi-log (``log(rho_k) ~ k``) sobre los rezagos con ``rho_k>0`` estimable.

    NO estima ningun parametro de memoria larga (``d``) ni declara
    "memoria larga verdadera" -- solo reporta cual de las dos formas
    describe mejor (mayor R^2) la ACF observada. Un decaimiento lento
    puede deberse a memoria larga genuina O a cambios de nivel/regimen no
    modelados (Tsay, C3) -- esa ambiguedad NO se resuelve aqui, queda
    abierta para TDA-14.
    """
    sub = acf_tab.loc[acf_tab["estimable"] & (acf_tab["rho"] > 0)].copy()
    if len(sub) < 5:
        return {"estimable": False, "n_points": int(len(sub))}

    k = sub["lag"].to_numpy(dtype=float)
    rho = sub["rho"].to_numpy(dtype=float)
    log_rho = np.log(rho)

    def _fit_r2(xvar: np.ndarray, yvar: np.ndarray) -> tuple[float, float, float]:
        X = np.column_stack([np.ones_like(xvar), xvar])
        beta_hat, *_ = np.linalg.lstsq(X, yvar, rcond=None)
        y_hat = X @ beta_hat
        ss_res = float(np.sum((yvar - y_hat) ** 2))
        ss_tot = float(np.sum((yvar - yvar.mean()) ** 2))
        r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        return float(beta_hat[1]), float(beta_hat[0]), r2

    slope_loglog, intercept_loglog, r2_loglog = _fit_r2(np.log(k), log_rho)
    slope_semilog, intercept_semilog, r2_semilog = _fit_r2(k, log_rho)

    if r2_loglog > r2_semilog:
        form_hint = "mas compatible con log-log (posible decaimiento polinomial / persistencia lenta -- NO se afirma memoria larga, ver docstring)"
    elif r2_semilog > r2_loglog:
        form_hint = "mas compatible con semi-log (posible decaimiento exponencial / persistencia corta)"
    else:
        form_hint = "sin diferencia clara entre ambas formas"

    return {
        "estimable": True, "n_points": int(len(sub)),
        "slope_loglog": slope_loglog, "intercept_loglog": intercept_loglog, "r2_loglog": r2_loglog,
        "slope_semilog": slope_semilog, "intercept_semilog": intercept_semilog, "r2_semilog": r2_semilog,
        "form_hint": form_hint,
    }


# ---------------------------------------------------------------------------
# Estabilidad temporal -- año, segmento Y VENTANA RODANTE (mensual), con
# bootstrap de incertidumbre POR GRUPO (correccion de auditoria v1,
# puntos 1 y 2)
# ---------------------------------------------------------------------------

def year_month_labels(trading_date) -> np.ndarray:
    """``"YYYY-MM"`` a partir de ``trading_date`` -- la "ventana rodante" mas simple y transparente (misma convencion que ``rolling_rho1_by_month`` de TDA-08)."""
    td = pd.to_datetime(pd.Series(np.asarray(trading_date)))
    return (td.dt.year.astype(str) + "-" + td.dt.month.astype(str).str.zfill(2)).to_numpy()


def bootstrap_rho_by_group_all_groups(
    values: np.ndarray, block_ids: np.ndarray, trading_date: np.ndarray, group_labels: np.ndarray, lag: int,
    n_boot: int = GROUP_BOOTSTRAP_N_BOOT, seed: int = GROUP_BOOTSTRAP_SEED, min_boot_replicas: int = GROUP_BOOTSTRAP_MIN_REPLICAS,
) -> pd.DataFrame:
    """Bootstrap de bloques por jornada para ``rho`` en UN rezago fijo, con IC 95% calculado SIMULTANEAMENTE para TODOS los valores de ``group_labels``.

    OPTIMIZACION DE RENDIMIENTO (documentada, sin cambiar metodologia):
    una implementacion ingenua resamplearia los dias UNA VEZ POR GRUPO
    (año/segmento/mes), multiplicando el costo por el numero de grupos.
    Esta version resamplea los dias UNA SOLA VEZ POR REPLICA y, dentro de
    esa misma replica, calcula ``rho`` para TODOS los grupos a la vez via
    un ``groupby`` (la misma tecnica de ``acf_by_group`` de TDA-08) --
    exactamente la misma definicion estadistica, computada en una
    fraccion del tiempo.

    Misma definicion "principal" que ``acf_by_group`` (TDA-08): un par
    ``(t, t-lag)`` se cuenta para el grupo de la observacion MAS RECIENTE
    (``t``), sin exigir que ``t-lag`` pertenezca al mismo grupo -- evita
    el error que TDA-08 corrigio en su 1a revision (filtrar FILAS por
    grupo antes de construir los pares fabrica una condicion mas
    estricta que "group_t == group_t", y ademas -- para grupos que
    recurren varias veces por dia, como los segmentos horarios --
    fabricaria vecinos entre dias distintos que nunca fueron
    temporalmente adyacentes).

    Salida: ``DataFrame`` con ``group``, ``n_boot_used``, ``rho_ci_lo``,
    ``rho_ci_hi`` (``NaN`` si ``n_boot_used < min_boot_replicas``).
    """
    values = np.asarray(values, dtype=float)
    block_ids = np.asarray(block_ids)
    group_labels = np.asarray(group_labels)
    trading_date = np.asarray(trading_date)

    unique_dates, inverse = np.unique(trading_date, return_inverse=True)
    n_dates = len(unique_dates)
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    boundaries = np.searchsorted(sorted_inverse, np.arange(n_dates + 1))
    date_to_rows = [order[boundaries[i]:boundaries[i + 1]] for i in range(n_dates)]

    all_groups = pd.unique(group_labels)
    boot_rho = {g: np.full(n_boot, np.nan) for g in all_groups}

    rng = np.random.default_rng(seed)
    for b in range(n_boot):
        sampled_days = rng.integers(0, n_dates, size=n_dates)
        idx_parts, slot_parts = [], []
        for slot, d in enumerate(sampled_days):
            rows = date_to_rows[d]
            idx_parts.append(rows)
            slot_parts.append(np.full(len(rows), slot, dtype=np.int64))
        idx = np.concatenate(idx_parts)
        slots = np.concatenate(slot_parts)

        vals = values[idx]
        grp = group_labels[idx]
        composite_block = block_ids[idx].astype(np.int64) * (n_dates + 1) + slots
        T = vals.size
        if lag >= T:
            continue
        same = composite_block[lag:] == composite_block[:-lag]
        if not same.any():
            continue

        x_t = vals[lag:][same]
        x_tk = vals[:-lag][same]
        g_t = grp[lag:][same]

        pairs = pd.DataFrame({"g": g_t, "x_t": x_t, "x_tk": x_tk})
        grouped = pairs.groupby("g", observed=True)
        mean_t_g = grouped["x_t"].transform("mean").to_numpy()
        mean_tk_g = grouped["x_tk"].transform("mean").to_numpy()
        e_t = pairs["x_t"].to_numpy() - mean_t_g
        e_tk = pairs["x_tk"].to_numpy() - mean_tk_g

        agg = pd.DataFrame({"g": g_t, "prod": e_t * e_tk, "et2": e_t * e_t, "etk2": e_tk * e_tk}).groupby("g", observed=True).agg(
            n=("prod", "size"), sum_prod=("prod", "sum"), sum_et2=("et2", "sum"), sum_etk2=("etk2", "sum"),
        )
        for g, row in agg.iterrows():
            if g not in boot_rho or row["n"] < 2:
                continue
            denom = np.sqrt(row["sum_et2"] * row["sum_etk2"])
            if denom > 0:
                boot_rho[g][b] = row["sum_prod"] / denom

    rows = []
    for g in all_groups:
        arr = boot_rho[g]
        finite = arr[~np.isnan(arr)]
        enough = finite.size >= min_boot_replicas
        rows.append({
            "group": g, "n_boot_used": int(finite.size),
            "rho_ci_lo": float(np.percentile(finite, 2.5)) if enough else float("nan"),
            "rho_ci_hi": float(np.percentile(finite, 97.5)) if enough else float("nan"),
        })
    return pd.DataFrame(rows)


def _persistence_table_with_ci(
    values: np.ndarray, block_ids: np.ndarray, trading_date: np.ndarray, group_labels: np.ndarray,
    variable: str, kind: str, lags: tuple[int, ...] = STABILITY_LAGS,
    ci_lag: int = GROUP_BOOTSTRAP_LAG, n_boot: int = GROUP_BOOTSTRAP_N_BOOT, seed: int = GROUP_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Punto estimado (``acf_by_group``, TDA-08, sin cambios) + IC 95% bootstrap por grupo (SOLO en ``ci_lag``, costo acotado -- ver ``bootstrap_rho_by_group_all_groups``)."""
    point = acf_by_group(values, block_ids, group_labels, lags, MIN_N_FOR_GROUP)
    point.insert(0, "variable", variable)
    point.insert(1, "raw_adjusted", kind)

    ci = bootstrap_rho_by_group_all_groups(values, block_ids, trading_date, group_labels, ci_lag, n_boot, seed)
    merged = point.merge(ci, on="group", how="left")
    at_ci_lag = merged["lag"] == ci_lag
    merged["rho_ci_lo"] = np.where(at_ci_lag, merged["rho_ci_lo"], np.nan)
    merged["rho_ci_hi"] = np.where(at_ci_lag, merged["rho_ci_hi"], np.nan)
    merged["n_boot_used"] = np.where(at_ci_lag, merged["n_boot_used"], np.nan)
    return merged


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------

@dataclass
class TDA09Result:
    r1m_population: pd.DataFrame
    r_tilde_population: pd.DataFrame
    log_hl_population: pd.DataFrame
    log_hl_tilde_population: pd.DataFrame
    segment_labels: list[str]

    acf_table: pd.DataFrame          # largo: variable, raw/adjusted, lag, rho, n_pairs, estimable
    bootstrap_table: pd.DataFrame    # largo: variable, raw/adjusted, kind, lag/label, rho_point, ci_lo, ci_hi, n_pairs/n_day_pairs
    portmanteau_table: pd.DataFrame  # largo: variable, raw/adjusted, m, Q, estimable, calibration_status
    clock_attribution_table: pd.DataFrame
    stop9_decision: dict
    persistence_by_year: pd.DataFrame
    persistence_by_segment: pd.DataFrame
    persistence_rolling_window: pd.DataFrame
    arch_lm_table: pd.DataFrame
    g2_null1_calibration: pd.DataFrame
    g2_secondary_global_calibration: pd.DataFrame
    g2_synthetic_moment_check: dict
    mean_removal_sensitivity_table: pd.DataFrame
    clock_flatness_table: pd.DataFrame
    decay_diagnostics: dict[str, dict]
    th20_enabled: bool
    th19_verdict: dict = field(default_factory=dict)
    th21_verdict: dict = field(default_factory=dict)
    th20_verdict: str = ""
    stage_timings: dict[str, float] = field(default_factory=dict)


def run_tda09_analysis(config: SnapshotConfig, verbose: bool = True) -> TDA09Result:
    """Orquesta TDA-09 completo en 8 etapas con progreso visible: ACF/portmanteau/bootstrap de magnitud (crudo vs ajustado), estabilidad por año/segmento/ventana rodante con IC bootstrap, G2, LM de Engle, comparacion descriptiva de energia de reloj (TH21), STOP-9, forma del decaimiento (TH20 condicional)."""
    timer = _StageTimer(ANALYSIS_STAGES, verbose=verbose)

    # --- Etapa 1: carga de datos y poblaciones ------------------------------
    timer.stage("Carga de datos y construccion de poblaciones (r_1m/r_tilde/log_hl/log_hl_tilde)")
    validate_research_holdout_disjoint(config)

    variables, validity = load_tda04_inputs(config)
    last_timestamps = variables.groupby("source_file")["timestamp"].max().to_dict()
    validate_last_timestamps_before_boundary(config, last_timestamps)
    verify_timestamp_alignment(variables, validity)

    n_boot = config.tda09_n_boot
    n_perm = config.tda09_n_perm
    n_perm_arch_lm = config.tda09_n_perm_arch_lm

    cutoffs = load_segmentation_cutoffs(config)

    r1m_pop, segment_labels = build_r1m_population(variables, validity, cutoffs)
    r1m_pop = r1m_pop.sort_values("timestamp").reset_index(drop=True)

    r_tilde = load_r_tilde(config)
    verify_r_tilde_invariants(r_tilde, variables)
    r_tilde_pop, _ = build_r_tilde_population(r_tilde, cutoffs)
    r_tilde_pop = r_tilde_pop.sort_values("timestamp").reset_index(drop=True)

    log_hl_pop, _ = build_log_hl_population(variables, validity, cutoffs)
    log_hl_pop = log_hl_pop.sort_values("timestamp").reset_index(drop=True)

    s_m_table = pd.read_parquet(config.tda06_s_m_parquet_path)
    verify_s_m_is_log_hl_proxy(s_m_table)
    log_hl_tilde_pop = build_log_hl_tilde(log_hl_pop, s_m_table)
    log_hl_tilde_pop = log_hl_tilde_pop.loc[log_hl_tilde_pop["log_hl_tilde"].notna()].sort_values("timestamp").reset_index(drop=True)

    # Bloques de continuidad -- CONTRACT-AWARE (correccion de auditoria v1,
    # punto 4): exige mismo `contract` ademas de mismo trading_date/delta.
    block_ids_r1m = compute_block_ids_with_contract(r1m_pop["timestamp"], r1m_pop["trading_date"], r1m_pop["contract"], 60.0)
    block_ids_r_tilde = compute_block_ids_with_contract(r_tilde_pop["timestamp"], r_tilde_pop["trading_date"], r_tilde_pop["contract"], 60.0)
    block_ids_log_hl = compute_block_ids_with_contract(log_hl_pop["timestamp"], log_hl_pop["trading_date"], log_hl_pop["contract"], 60.0)
    block_ids_log_hl_tilde = compute_block_ids_with_contract(log_hl_tilde_pop["timestamp"], log_hl_tilde_pop["trading_date"], log_hl_tilde_pop["contract"], 60.0)

    # --- Etapa 2: sensibilidad de media + ACF (8 series) + portmanteau -----
    timer.stage("Sensibilidad de media, ACF de magnitud/direccion (hasta 600 rezagos) y portmanteau")
    mean_removal_table = mean_removal_sensitivity(r1m_pop, block_ids_r1m)

    r_raw = r1m_pop["r_1m"].to_numpy(dtype=float)
    r_adj = r_tilde_pop["r_tilde"].to_numpy(dtype=float)
    abs_r_raw = r1m_pop["abs_r_1m"].to_numpy(dtype=float)
    abs_r_adj = np.abs(r_adj)
    r2_raw = r1m_pop["r2_1m"].to_numpy(dtype=float)
    r2_adj = r_adj ** 2
    log_hl_raw = log_hl_pop["log_hl"].to_numpy(dtype=float)
    log_hl_adj = log_hl_tilde_pop["log_hl_tilde"].to_numpy(dtype=float)

    pop_by_key: dict[tuple[str, str], pd.DataFrame] = {
        ("r", "raw"): r1m_pop, ("r", "adjusted"): r_tilde_pop,
        ("abs_r", "raw"): r1m_pop, ("abs_r", "adjusted"): r_tilde_pop,
        ("r2", "raw"): r1m_pop, ("r2", "adjusted"): r_tilde_pop,
        ("log_hl", "raw"): log_hl_pop, ("log_hl", "adjusted"): log_hl_tilde_pop,
    }
    block_ids_by_key: dict[tuple[str, str], np.ndarray] = {
        ("r", "raw"): block_ids_r1m, ("r", "adjusted"): block_ids_r_tilde,
        ("abs_r", "raw"): block_ids_r1m, ("abs_r", "adjusted"): block_ids_r_tilde,
        ("r2", "raw"): block_ids_r1m, ("r2", "adjusted"): block_ids_r_tilde,
        ("log_hl", "raw"): block_ids_log_hl, ("log_hl", "adjusted"): block_ids_log_hl_tilde,
    }

    series_spec = [
        # (name, raw/adjusted, values, block_ids, minute_of_day, has_g2_null, in_bootstrap, in_stability)
        ("r", "raw", r_raw, block_ids_r1m, r1m_pop["minute_of_day"].to_numpy(), False, True, False),
        ("r", "adjusted", r_adj, block_ids_r_tilde, r_tilde_pop["minute_of_day"].to_numpy(), False, True, False),
        ("abs_r", "raw", abs_r_raw, block_ids_r1m, r1m_pop["minute_of_day"].to_numpy(), True, True, True),
        ("abs_r", "adjusted", abs_r_adj, block_ids_r_tilde, r_tilde_pop["minute_of_day"].to_numpy(), True, True, True),
        ("r2", "raw", r2_raw, block_ids_r1m, r1m_pop["minute_of_day"].to_numpy(), False, False, False),
        ("r2", "adjusted", r2_adj, block_ids_r_tilde, r_tilde_pop["minute_of_day"].to_numpy(), False, False, False),
        ("log_hl", "raw", log_hl_raw, block_ids_log_hl, log_hl_pop["minute_of_day"].to_numpy(), True, True, True),
        ("log_hl", "adjusted", log_hl_adj, block_ids_log_hl_tilde, log_hl_tilde_pop["minute_of_day"].to_numpy(), True, True, True),
    ]

    g2_portmanteau_m = tuple(m for m in PORTMANTEAU_M_MAGNITUDE if m <= G2_NULL_MAX_LAG)

    acf_rows, portmanteau_rows = [], []
    acf_lookup: dict[tuple[str, str], pd.DataFrame] = {}
    port_lookup: dict[tuple[str, str], pd.DataFrame] = {}

    for name, kind, values, block_ids, minute_of_day, has_g2_null, in_bootstrap, in_stability in series_spec:
        T = values.size
        acf_tab = compute_acf(values, block_ids, MAX_LAG_MAGNITUDE)
        acf_lookup[(name, kind)] = acf_tab
        se = bartlett_se(acf_tab["rho"].to_numpy(), T)
        out = acf_tab.copy()
        out.insert(0, "variable", name)
        out.insert(1, "raw_adjusted", kind)
        out["bartlett_se"] = se
        acf_rows.append(out)

        port = annotate_portmanteau_calibration(
            compute_portmanteau_q(acf_tab, PORTMANTEAU_M_MAGNITUDE), has_null1_calibration=has_g2_null,
        )
        port_lookup[(name, kind)] = port
        port_out = port.copy()
        port_out.insert(0, "variable", name)
        port_out.insert(1, "raw_adjusted", kind)
        portmanteau_rows.append(port_out)

    acf_table = pd.concat(acf_rows, ignore_index=True)
    portmanteau_table = pd.concat(portmanteau_rows, ignore_index=True)

    # --- Etapa 3: bootstrap (series completas + same-clock-position) -------
    timer.stage("Bootstrap de bloques por jornada (rezagos clave) y diagnostico same-clock-position")
    bootstrap_rows = []
    for name, kind, values, block_ids, minute_of_day, has_g2_null, in_bootstrap, in_stability in series_spec:
        if not in_bootstrap:
            continue
        trading_date_arr = pop_by_key[(name, kind)]["trading_date"].to_numpy()
        rho_boot, _ = bootstrap_rho(values, block_ids, trading_date_arr, BOOTSTRAP_LAGS_MAGNITUDE, n_boot, DEFAULT_BOOTSTRAP_SEED)
        acf_idx = acf_lookup[(name, kind)].set_index("lag")
        for li, k in enumerate(BOOTSTRAP_LAGS_MAGNITUDE):
            finite = rho_boot[:, li][~np.isnan(rho_boot[:, li])]
            bootstrap_rows.append({
                "variable": name, "raw_adjusted": kind, "kind": "continuous_lag", "label": str(k),
                "rho_point": float(acf_idx.loc[k, "rho"]) if k in acf_idx.index else np.nan,
                "n_pairs": int(acf_idx.loc[k, "n_pairs"]) if k in acf_idx.index else 0,
                "ci_lo": float(np.percentile(finite, 2.5)) if finite.size else np.nan,
                "ci_hi": float(np.percentile(finite, 97.5)) if finite.size else np.nan,
            })

    _SAME_CLOCK_COL = "_same_clock_value"
    same_clock_specs = [
        ("abs_r", "raw", r1m_pop, abs_r_raw),
        ("abs_r", "adjusted", r_tilde_pop, abs_r_adj),
        ("log_hl", "raw", log_hl_pop, log_hl_raw),
        ("log_hl", "adjusted", log_hl_tilde_pop, log_hl_adj),
    ]
    same_clock_rows = []
    for name, kind, pop, values_for_pairing in same_clock_specs:
        pop_with_col = pop[["trading_date", "minute_of_day"]].assign(**{_SAME_CLOCK_COL: values_for_pairing})
        result = same_clock_next_trading_day(pop_with_col, _SAME_CLOCK_COL, n_boot, DEFAULT_BOOTSTRAP_SEED)
        same_clock_rows.append({
            "variable": name, "raw_adjusted": kind, "kind": "same_clock_next_trading_day", "label": "SAME_CLOCK_NEXT_DAY",
            "rho_point": result["rho"], "n_pairs": result["n_pairs"], "ci_lo": result["ci_lo"], "ci_hi": result["ci_hi"],
            "n_day_pairs": result["n_day_pairs"],
        })
    bootstrap_table = pd.concat([pd.DataFrame(bootstrap_rows), pd.DataFrame(same_clock_rows)], ignore_index=True)

    # --- Etapa 4: G2 (principal + secundario + sintetico) + aplanamiento ---
    timer.stage("Calibracion G2 (null principal por minuto, secundario global, sintetico) y diagnostico de aplanamiento de reloj")
    g2_null1_rows, g2_secondary_rows = [], []
    for name, kind, values, block_ids, minute_of_day, has_g2_null, in_bootstrap, in_stability in series_spec:
        if not has_g2_null:
            continue
        acf_tab = acf_lookup[(name, kind)]
        port = port_lookup[(name, kind)]
        perm_null_full = g2_permutation_null_by_minute(values, minute_of_day, block_ids, G2_FULL_LAGS, n_perm, PERMUTATION_SEED)
        sparse_idx = [G2_FULL_LAGS.index(k) for k in G2_LAGS]
        real_rho_by_lag = {k: float(acf_tab.loc[acf_tab["lag"] == k, "rho"].iloc[0]) for k in G2_LAGS}
        cal = g2_null1_calibration_summary(real_rho_by_lag, perm_null_full[:, sparse_idx], G2_LAGS)
        cal.insert(0, "variable", name)
        cal.insert(1, "raw_adjusted", kind)
        cal.insert(2, "kind_of_row", "rho_calibration")
        g2_null1_rows.append(cal)

        real_q_by_m = {m: float(port.loc[port["m"] == m, "Q"].iloc[0]) for m in g2_portmanteau_m if (port["m"] == m).any()}
        port_null = g2_null1_portmanteau_summary(perm_null_full, acf_tab["n_pairs"].to_numpy(), g2_portmanteau_m, real_q=real_q_by_m)
        port_null.insert(0, "variable", name)
        port_null.insert(1, "raw_adjusted", kind)
        g2_null1_rows.append(port_null.rename(columns={"m": "lag"}).assign(kind_of_row="portmanteau_calibration"))

        if kind == "adjusted":
            global_null_full = g2_global_permutation_null(values, block_ids, G2_FULL_LAGS, n_perm, GLOBAL_PERMUTATION_SEED)
            sec = g2_null1_calibration_summary(real_rho_by_lag, global_null_full[:, sparse_idx], G2_LAGS)
            sec.insert(0, "variable", name)
            sec.insert(1, "raw_adjusted", kind)
            sec.insert(2, "null_type", "global_permutation_secondary")
            g2_secondary_rows.append(sec)

    g2_null1_calibration = pd.concat(g2_null1_rows, ignore_index=True) if g2_null1_rows else pd.DataFrame()
    g2_secondary_global_calibration = pd.concat(g2_secondary_rows, ignore_index=True) if g2_secondary_rows else pd.DataFrame()

    flatness_rows = []
    for name in ("abs_r", "log_hl"):
        raw_values = abs_r_raw if name == "abs_r" else log_hl_raw
        adj_values = abs_r_adj if name == "abs_r" else log_hl_adj
        raw_minute = r1m_pop["minute_of_day"].to_numpy() if name == "abs_r" else log_hl_pop["minute_of_day"].to_numpy()
        adj_minute = r_tilde_pop["minute_of_day"].to_numpy() if name == "abs_r" else log_hl_tilde_pop["minute_of_day"].to_numpy()
        flat_raw = clock_profile_flatness(raw_values, raw_minute)
        flat_adj = clock_profile_flatness(adj_values, adj_minute)
        ratio = (flat_adj / flat_raw) if flat_raw not in (0, None) and np.isfinite(flat_raw) and flat_raw != 0 else float("nan")
        flatness_rows.append({
            "variable": name, "flatness_raw": flat_raw, "flatness_adjusted": flat_adj, "ratio": ratio,
            "clock_effectively_removed": bool(ratio <= CLOCK_FLATNESS_RATIO_THRESHOLD) if np.isfinite(ratio) else False,
        })
    clock_flatness_table = pd.DataFrame(flatness_rows)

    # --- Etapa 5: persistencia por año / segmento / ventana rodante ---------
    timer.stage("Persistencia por año, segmento y ventana rodante (mensual), con IC bootstrap por grupo")
    year_rows, segment_rows, rolling_rows = [], [], []
    for name, kind, values, block_ids, minute_of_day, has_g2_null, in_bootstrap, in_stability in series_spec:
        if not in_stability:
            continue
        pop = pop_by_key[(name, kind)]
        trading_date_arr = pop["trading_date"].to_numpy()

        year_rows.append(_persistence_table_with_ci(values, block_ids, trading_date_arr, pop["year_ny"].to_numpy(), name, kind))
        segment_rows.append(_persistence_table_with_ci(values, block_ids, trading_date_arr, pop["segment_label"].to_numpy(), name, kind))
        rolling_rows.append(_persistence_table_with_ci(values, block_ids, trading_date_arr, year_month_labels(trading_date_arr), name, kind))

    persistence_by_year = pd.concat(year_rows, ignore_index=True)
    persistence_by_segment = pd.concat(segment_rows, ignore_index=True)
    persistence_rolling_window = pd.concat(rolling_rows, ignore_index=True).rename(columns={"group": "year_month"})

    # --- Etapa 6: LM de Engle ------------------------------------------------
    timer.stage("LM de Engle (r_1m crudo y r_tilde ajustado)")
    arch_rows = []
    for name, values, block_ids, minute_of_day in [
        ("r_1m", r_raw, block_ids_r1m, r1m_pop["minute_of_day"].to_numpy()),
        ("r_tilde", r_adj, block_ids_r_tilde, r_tilde_pop["minute_of_day"].to_numpy()),
    ]:
        for order in ARCH_LM_ORDERS:
            result = calibrate_engle_lm(values, block_ids, minute_of_day, order, n_perm_arch_lm, ARCH_LM_PERMUTATION_SEED)
            result["series"] = name
            arch_rows.append(result)
    arch_lm_table = pd.DataFrame(arch_rows)

    # --- Etapa 7: TH21 (comparacion descriptiva de energia) + STOP-9 + TH20 -
    timer.stage("Comparacion descriptiva de energia de dependencia (TH21), STOP-9 y diagnostico de decaimiento (TH20)")
    clock_attr_rows = []
    for name in ("abs_r", "log_hl"):
        attr = clock_attribution(acf_lookup[(name, "raw")], acf_lookup[(name, "adjusted")], TH21_ENERGY_M)
        attr["variable"] = name
        clock_attr_rows.append(attr)
    clock_attribution_table = pd.DataFrame(clock_attr_rows)
    stop9_decision = decide_stop9(clock_attr_rows)
    th21_verdict = classify_th21(clock_attr_rows)

    abs_r_survives = clock_attribution_table.loc[clock_attribution_table["variable"] == "abs_r", "fraction_survives"].iloc[0]
    log_hl_survives = clock_attribution_table.loc[clock_attribution_table["variable"] == "log_hl", "fraction_survives"].iloc[0]
    th20_enabled = bool(
        (np.isfinite(abs_r_survives) and abs_r_survives >= TH20_MIN_FRACTION_SURVIVES) or
        (np.isfinite(log_hl_survives) and log_hl_survives >= TH20_MIN_FRACTION_SURVIVES)
    )
    decay_diagnostics: dict[str, dict] = {}
    if th20_enabled:
        decay_diagnostics["abs_r_adjusted"] = decay_form_diagnostic(acf_lookup[("abs_r", "adjusted")])
        decay_diagnostics["log_hl_adjusted"] = decay_form_diagnostic(acf_lookup[("log_hl", "adjusted")])
    th20_verdict = th20_status_label(th20_enabled, decay_diagnostics)

    # --- Etapa 8: null sintetico (diagnostico) + veredicto final TH19 -------
    timer.stage("Diagnostico del null sintetico heredado de TDA-08 y veredicto final TH19")
    from ohlcv_dataroad.ingest.tda08_linear_mean_dependence import (
        draw_synthetic_empirical_sample,
        g2_synthetic_null_moment_check,
    )
    s_m_values_r = r_tilde_pop["s_m"].to_numpy(dtype=float)
    r_tilde_pool = r_adj[~np.isnan(r_adj)]
    synthetic_sample = draw_synthetic_empirical_sample(r_tilde_pool, s_m_values_r, SYNTHETIC_NULL_SEED)
    synthetic_abs = np.abs(synthetic_sample)
    moment_check = g2_synthetic_null_moment_check(abs_r_raw, synthetic_abs, r1m_pop["minute_of_day"].to_numpy(), s_m_values_r)
    moment_check["note"] = (
        "Null sintetico heredado de TDA-08 (resampleo empirico de r_tilde reescalado por s(m)), aplicado a |r| "
        "-- TDA-08 ya demostro que este null NO preserva varianza/curtosis reales (solo el perfil de escala). "
        "Se reevalua aqui sobre |r| y se documenta como SENSIBILIDAD FALLIDA/DIAGNOSTICO, EXCLUIDA de la "
        "inferencia principal de G2 (que usa exclusivamente el null por permutacion, g2_null1_calibration). "
        "No se construyo un null sintetico propio para log_hl (G4): el null por permutacion ya calibra la "
        "inferencia principal de log_hl sin necesitar un segundo sistema de nulls."
    )

    th19_verdict = decide_th19(g2_null1_calibration)

    stage_timings = timer.finish()

    return TDA09Result(
        r1m_population=r1m_pop, r_tilde_population=r_tilde_pop,
        log_hl_population=log_hl_pop, log_hl_tilde_population=log_hl_tilde_pop,
        segment_labels=segment_labels,
        acf_table=acf_table, bootstrap_table=bootstrap_table, portmanteau_table=portmanteau_table,
        clock_attribution_table=clock_attribution_table, stop9_decision=stop9_decision,
        persistence_by_year=persistence_by_year, persistence_by_segment=persistence_by_segment,
        persistence_rolling_window=persistence_rolling_window,
        arch_lm_table=arch_lm_table,
        g2_null1_calibration=g2_null1_calibration, g2_secondary_global_calibration=g2_secondary_global_calibration,
        g2_synthetic_moment_check=moment_check,
        mean_removal_sensitivity_table=mean_removal_table, clock_flatness_table=clock_flatness_table,
        decay_diagnostics=decay_diagnostics, th20_enabled=th20_enabled,
        th19_verdict=th19_verdict, th21_verdict=th21_verdict, th20_verdict=th20_verdict,
        stage_timings=stage_timings,
    )
