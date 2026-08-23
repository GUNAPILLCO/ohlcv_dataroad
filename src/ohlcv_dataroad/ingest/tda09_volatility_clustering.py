"""TDA-09 -- Dependencia en magnitud (volatility clustering).

Implementa la etapa TDA-09 del roadmap
(``docs/methodology/Tsay_OHLCV_analysis_roadmap.md``, seccion "TDA-09"):
mide si el TAMAÑO de los movimientos de MNQ tiene memoria (agrupamiento
de volatilidad), separando explicitamente cuanto de esa memoria es un
patron determinista de reloj (TDA-06, ``s(m)``) y cuanto sobrevive tras
quitarlo.

Resuelve TH19, TH20 y TH21 (ver informe,
``reports/mnq/TDA09_volatility_clustering.md``, para el estado final de
cada una).

Que es "magnitud" aqui, en una frase: NO es prediccion de direccion --
``|r_t|``, ``r_t^2`` y ``log(H_t/L_t)`` son proxies de cuan GRANDE fue el
movimiento, sin importar el signo. Si periodos agitados tienden a seguir
agitados (y periodos tranquilos, tranquilos), eso es "memoria en
magnitud" aunque la direccion siga siendo poco predecible (TDA-08 ya
cerro esa pregunta por separado: TH16/17/18).

Decision sobre r_t vs residuo/innovacion (seccion 8 de la tarea): TDA-08
(CLOSED) encontro que la dependencia lineal en la media es diminuta
(``beta_1~0.0059``, ~0.14 ticks) y `NOT SEPARABLE WITH OHLCV LAST`
(STOP-8a/8b). Por parsimonia (G4), este modulo usa ``r_1m`` directamente
como "innovacion" para TDA-09 -- no se ajusta un modelo AR nuevo. Se
ejecuta, como maximo, UNA sensibilidad barata (``mean_removal_sensitivity``):
comparar ACF(|r|) contra ACF(|r - beta_1*r_{t-1}|) en una grilla pequeña
de rezagos, para verificar (no solo asumir) que remover la media no
cambia materialmente la conclusion de magnitud.

TOPOLOGIA TEMPORAL (bloqueante, heredada de TDA-08 sin modificarla): se
reutilizan ``compute_block_ids``/``compute_acf``/``bootstrap_rho``/
``acf_by_group``/``g2_permutation_null_by_minute`` de
``tda08_linear_mean_dependence.py`` TAL CUAL -- ningun cambio a ese
modulo. Ninguna ACF de esta etapa cruza ``trading_date``, gaps o rolls.

CRUDO vs AJUSTADO POR RELOJ (el resultado central de la etapa, seccion 7
de la tarea):

- ``|r_1m|``/``r_1m^2`` (crudo) vs ``|r_tilde|``/``r_tilde^2`` (ajustado)
  -- ``r_tilde = r_1m / s(m)`` ya fue construido y persistido por TDA-06
  (RETROSPECTIVO). ``|r_tilde| = |r_1m|/s(m)`` y ``r_tilde^2 =
  r_1m^2/s(m)^2`` son identidades algebraicas directas de esa definicion
  -- no se inventa ninguna transformacion nueva para estas dos variables.
- ``log_hl`` (crudo) vs ``log_hl_tilde = log_hl / s(m)`` (ajustado) --
  transformacion NUEVA de esta etapa, pero dimensionalmente identica a la
  de ``r_tilde``: ``s(m)`` fue estimado por TDA-06 usando exactamente
  ``log_hl`` como proxy elegido (informe TDA-06, seccion 11) -- dividir
  ``log_hl`` por el mismo indice estacional que ya lo normaliza es
  coherente por construccion, no una convencion arbitraria. Se verifica
  (``verify_s_m_is_log_hl_proxy``) que ``tda06_s_m.parquet`` en efecto
  proviene de ``log_hl`` antes de usarlo asi -- si TDA-06 cambiara de
  proxy, esta etapa debe fallar explicitamente, no asumirlo en silencio.

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

from dataclasses import dataclass

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
    compute_block_ids,
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
    "build_log_hl_population",
    "verify_s_m_is_log_hl_proxy",
    "build_log_hl_tilde",
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
    "decay_form_diagnostic",
    "TDA09Result",
    "run_tda09_analysis",
]

# ---------------------------------------------------------------------------
# Convenciones PREDECLARADAS (seccion 10/11/12/13/14/17/18 de la tarea --
# nunca elegidas despues de ver el resultado)
# ---------------------------------------------------------------------------

# "Varios cientos de rezagos" (roadmap TH19): 600 minutos = 10 horas --
# cubre corto plazo, decaimiento y persistencia de varias horas, y queda
# comodamente por debajo del techo estructural de continuidad que TDA-08
# ya documento (max rezago continuo estimable = 1.378, ver ese informe
# seccion 1) sin necesitar volver a verificarlo aqui.
MAX_LAG_MAGNITUDE = 600

BOOTSTRAP_LAGS_MAGNITUDE = (1, 2, 5, 10, 20, 30, 60, 120, 240, 360, 480, 600)
PORTMANTEAU_M_MAGNITUDE = (10, 20, 40, 60, 120, 240, 480, 600)
G2_NULL_MAX_LAG = 60
G2_LAGS = tuple(k for k in BOOTSTRAP_LAGS_MAGNITUDE if k <= G2_NULL_MAX_LAG)
# ``g2_null1_portmanteau_summary`` (TDA-08) asume columnas de rezago
# CONTIGUAS (1..N) para poder tomar el prefijo ``[:m]`` -- se calcula el
# null de permutacion sobre el rango CONTIGUO completo (igual que TDA-08:
# ``null_lags = tuple(range(1, G2_NULL_MAX_LAG + 1))``), y se subselecciona
# por indice para la tabla de calibracion de ``rho`` en la grilla dispersa
# ``G2_LAGS`` -- exactamente el patron de TDA-08 (``bootstrap_lag_idx``).
G2_FULL_LAGS = tuple(range(1, G2_NULL_MAX_LAG + 1))

# Grilla pequeña y predeclarada para persistencia por año/segmento
# (seccion 19 de la tarea: "no hace falta recalcular cientos de lags para
# cada subdivision si una tabla pequeña de lags predeclarados responde
# correctamente la pregunta").
STABILITY_LAGS = (1, 5, 20, 60)

# LM de Engle: cantidad pequeña y predeclarada de ordenes (seccion 14 de
# la tarea).
ARCH_LM_ORDERS = (1, 5, 20)

# Ventana predeclarada para la metrica de "cantidad total de dependencia"
# (TH21, seccion 17 de la tarea): 240 minutos = 4 horas -- captura
# dependencia acumulada de varias horas sin exigir el rango completo de
# MAX_LAG_MAGNITUDE.
TH21_ENERGY_M = 240

# STOP-9 (seccion 22): si la fraccion de dependencia que SOBREVIVE al
# ajuste cae por debajo de este umbral (es decir, se remueve >=90%), se
# declara STOP-9 ACTIVADO. Punto de referencia DESCRIPTIVO predeclarado,
# leido de forma holistica junto con |r| Y log_hl (igual que TDA-06 hizo
# con STOP-6) -- no una puerta binaria ciega sobre una sola cifra.
STOP9_FRACTION_REMOVED_THRESHOLD = 0.90

# Umbral para decidir si el perfil de reloj de la serie AJUSTADA quedo
# "efectivamente removido" (seccion 15 de la tarea) -- comparacion de
# rango relativo, no una decision automatica: se documenta siempre, y el
# null POR MINUTO (mas conservador, nunca asume nada) sigue siendo el
# PRINCIPAL en todos los casos; el null GLOBAL solo se reporta como
# sensibilidad secundaria para la serie ajustada.
CLOCK_FLATNESS_RATIO_THRESHOLD = 0.20

# TH20 solo se habilita si una fraccion no trivial de la dependencia
# sobrevive al ajuste (seccion 18 de la tarea: "solo si TH19 demuestra que
# una cantidad MATERIAL de dependencia... sobrevive").
TH20_MIN_FRACTION_SURVIVES = 0.05

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


class SMProxyMismatchError(Exception):
    """``tda06_s_m.parquet`` no fue construido con ``log_hl`` como proxy.

    TDA-06 (CLOSED) eligio ``log_hl`` como proxy de ``s(m)`` (informe,
    seccion 11: "proxy elegido: log_hl"). El ajuste ``log_hl_tilde =
    log_hl / s(m)`` de esta etapa depende de esa eleccion especifica para
    ser dimensionalmente coherente (dividir ``log_hl`` por un indice
    estacional estimado sobre SI MISMO). Si una version futura de TDA-06
    cambiara de proxy sin que esta etapa se actualizara, aplicar la misma
    formula seria una transformacion arbitraria, no la coherente descrita
    en el informe -- por eso esta comprobacion es bloqueante en vez de
    asumida.
    """


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
    (TDA-04: "nunca es NaN por una regla de no-cruce"). Misma convencion
    que us TDA-06 uso para el perfil de ``log_hl``/``volume`` (poblacion
    completa, ~1.918.050 filas sobre el conjunto de investigacion real,
    frente a las ~1.914.530 de ``r_1m_valid``).
    """
    df = attach_calendar_fields(variables, validity)
    df["segment_label"], labels = assign_segment_label(df["minute_of_day"], cutoffs)
    return df, labels


def verify_s_m_is_log_hl_proxy(s_m_table: pd.DataFrame) -> None:
    """Invariante bloqueante: ``tda06_s_m.parquet`` debe provenir EXCLUSIVAMENTE del proxy ``log_hl``.

    Ver ``SMProxyMismatchError`` para el porque. Se ejecuta ANTES de
    construir ``log_hl_tilde`` -- si falla, esta etapa se detiene sin
    producir la rama ajustada de ``log_hl`` (la rama de ``r_tilde``, que
    no depende de esta comprobacion, no se ve afectada).
    """
    proxies = s_m_table["proxy"].unique()
    if len(proxies) != 1 or proxies[0] != "log_hl":
        raise SMProxyMismatchError(
            f"tda06_s_m.parquet tiene proxy(es) {list(proxies)!r}, se esperaba exclusivamente 'log_hl' "
            "(TDA-06 CLOSED, informe seccion 11). TDA-09 se detiene sin construir log_hl_tilde -- "
            "aplicar 'log_hl / s(m)' solo es dimensionalmente coherente si s(m) se estimo sobre log_hl."
        )


def build_log_hl_tilde(log_hl_pop: pd.DataFrame, s_m_table: pd.DataFrame) -> pd.DataFrame:
    """``log_hl_tilde = log_hl / s(minute_of_day)`` -- ajuste RETROSPECTIVO, dimensionalmente coherente.

    Identica mecanica de proteccion que ``tda06_intraday_calendar_profile.build_r_tilde``
    (division protegida: ``NaN`` explicito donde ``s_m`` no esta definido
    o no es positivo, nunca ``inf``) -- no se duplica esa funcion, se
    reimplementa la misma regla de dos lineas para no importar un
    artefacto especifico de ``r_1m`` sobre una poblacion de ``log_hl`` que
    tiene una fila distinta (TODAS las barras, no solo ``r_1m_valid``).

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
# Decision sobre r_t vs innovacion (seccion 8 de la tarea) -- sensibilidad
# barata unica, NO un nuevo modelo de media
# ---------------------------------------------------------------------------

def mean_removal_sensitivity(r1m_pop: pd.DataFrame, block_ids_raw: np.ndarray, lags: tuple[int, ...] = STABILITY_LAGS) -> pd.DataFrame:
    """Compara ACF(|r|) contra ACF(|r - beta_1*r_{t-1}|) -- ¿remover la media cambia la conclusion de magnitud?

    ``beta_1`` se recalcula aqui mismo (nunca se congela un numero externo
    de TDA-08, para no arriesgar que quede obsoleto si el conjunto de
    investigacion cambia) via ``compute_acf`` a rezago 1 sobre la misma
    poblacion ``r1m_pop``/``block_ids_raw``. El residuo ``e_t = r_t -
    beta_1*r_{t-1}`` solo se define donde existe un par valido del MISMO
    bloque de continuidad (nunca se fabrica un vecino) -- las filas sin
    predecesor valido quedan excluidas de la subpoblacion de ``e``, y los
    bloques de continuidad de ESA subpoblacion se recalculan desde sus
    propios timestamps (``compute_block_ids`` genera bloques mas cortos
    automaticamente donde se elimino una fila, sin necesitar logica
    adicional).

    Es, deliberadamente, la UNICA sensibilidad de este tipo en la etapa
    (seccion 8 de la tarea: "puede realizarse como maximo una sensibilidad
    barata"). No es un modelo AR ajustado ni se persiste como artefacto
    separado -- se reporta en el informe (§8).
    """
    values = r1m_pop["r_1m"].to_numpy(dtype=float)
    acf1 = compute_acf(values, block_ids_raw, max_lag=1)
    beta_1 = float(acf1.loc[0, "beta"]) if bool(acf1.loc[0, "estimable"]) else 0.0

    same_block_prev = np.concatenate([[False], block_ids_raw[1:] == block_ids_raw[:-1]])
    prev_values = np.concatenate([[np.nan], values[:-1]])
    e = values - beta_1 * prev_values

    sub = r1m_pop.loc[same_block_prev].copy()
    sub["e"] = e[same_block_prev]
    sub_block_ids = compute_block_ids(sub["timestamp"], sub["trading_date"], 60.0)

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
# removido"? (seccion 15 de la tarea -- se verifica, no se asume)
# ---------------------------------------------------------------------------

def clock_profile_flatness(values: np.ndarray, minute_of_day: np.ndarray) -> float:
    """``std(mediana por minuto) / mean(mediana por minuto)`` -- cuanto varia el nivel tipico entre minutos.

    Cuanto MAS BAJO, mas "plano" es el perfil por minuto de ``values`` --
    un perfil perfectamente plano (sin patron de reloj) da ~0. Se usa la
    MEDIANA por minuto (robusta a colas, misma preferencia de TDA-06) en
    vez de la media.
    """
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
    efectivamente removido (``clock_profile_flatness`` bajo, ver
    ``CLOCK_FLATNESS_RATIO_THRESHOLD``) -- por eso NUNCA es el null
    PRINCIPAL de esta etapa (ese rol lo cumple siempre
    ``g2_permutation_null_by_minute``, mas conservador porque no depende
    de esa verificacion). Se reporta como sensibilidad secundaria,
    explicitamente comparada contra el principal (informe, TH21).
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
# LM de Engle -- diagnostico/test, NUNCA modelo ajustado (seccion 14)
# ---------------------------------------------------------------------------

def block_relative_position(block_ids: np.ndarray) -> np.ndarray:
    """Posicion (0-index) de cada fila DENTRO de su propio bloque de continuidad.

    Una fila con posicion ``p`` garantiza que las ``p`` filas anteriores
    pertenecen al MISMO bloque (los bloques son, por construccion de
    ``compute_block_ids``, tramos de delta=60s exacto y mismo
    ``trading_date`` -- nunca hay un salto oculto dentro de un bloque).
    Se usa para decidir, sin ambiguedad, que filas tienen una cadena
    COMPLETA de ``order`` rezagos validos para el LM de Engle.
    """
    return pd.Series(np.arange(len(block_ids))).groupby(np.asarray(block_ids)).cumcount().to_numpy()


def engle_lm_statistic(x: np.ndarray, block_ids: np.ndarray, order: int) -> dict:
    """LM de Engle para efecto ARCH de orden ``order`` sobre ``x`` (tipicamente un retorno/innovacion).

    Regresion OLS: ``x_t^2`` sobre una constante y ``x_{t-1}^2 .. x_{t-order}^2``,
    restringida a filas con una cadena COMPLETA de ``order`` rezagos
    validos DENTRO del mismo bloque de continuidad
    (``block_relative_position(block_ids) >= order``) -- nunca se
    construye un rezago que cruce ``trading_date``, un hueco o un roll.

    ``LM = n_eff * R^2`` (formulacion estandar de Engle, Tsay C3) -- bajo
    topologia fragmentada y muestras de ``n~10^6``, la calibracion
    asintotica clasica (``chi2(order)``) NO se usa como evidencia
    principal (seccion 14 de la tarea): este numero se reporta junto a su
    calibracion EMPIRICA por permutacion (``calibrate_engle_lm``), nunca
    solo.

    Salida: ``dict`` con ``order``, ``n_eff``, ``LM``, ``R2``, ``estimable``.
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
    """Permuta ``values`` DENTRO de cada ``minute_of_day`` -- misma mecanica que ``g2_permutation_null_by_minute`` (TDA-08), expuesta como paso reutilizable.

    Se reimplementa (en vez de extraerse de TDA-08, para no modificar ese
    modulo CLOSED) porque el LM de Engle necesita el ARRAY PERMUTADO
    completo, no solo el ``rho_k`` que ``g2_permutation_null_by_minute``
    devuelve.
    """
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
    """Calibracion EMPIRICA del LM de Engle: percentil del ``LM`` real dentro de ``n_perm`` replicas permutadas por minuto.

    Misma logica de G2 (Null 1: permutacion dentro de ``minute_of_day``,
    preserva el perfil de reloj y la marginal real, destruye la
    dependencia serial) aplicada al estadistico LM en vez de a ``rho_k``.
    ``n_perm`` reducido respecto del G2 principal (60 vs 200) por
    presupuesto computacional -- cada replica exige una regresion OLS
    completa sobre ~n filas, no solo una correlacion; decision de
    presupuesto documentada explicitamente, no una reduccion silenciosa.
    """
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
# de una ACF continua de 1.380 minutos (seccion 11 de la tarea)
# ---------------------------------------------------------------------------

def same_clock_next_trading_day(
    pop_df: pd.DataFrame, value_col: str, n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict:
    """Correlacion, agrupada a traves de TODOS los minutos, entre el valor de ``value_col`` en (trading_date=d, minuto=m) y en (trading_date=SIGUIENTE dia de negociacion presente, minuto=m).

    Explicitamente NO es una ACF continua de 1.380/2.760 minutos: la
    topologia de no-cruce vigente (TDA-08, seccion 1) hace que ese rezago
    sea ``NOT_ESTIMABLE`` bajo el motor continuo (max rezago realmente
    estimable = 1.378). Este diagnostico no conecta el final de un
    ``trading_date`` con el siguiente minuto a minuto -- en su lugar,
    empareja el MISMO minuto del reloj en dos jornadas de negociacion
    CONSECUTIVAS presentes en los datos (no necesariamente consecutivas
    en el calendario: fines de semana/feriados simplemente hacen que el
    "siguiente dia de negociacion" no sea el dia calendario siguiente).

    El bootstrap remuestrea PARES DE DIAS CONSECUTIVOS completos (con
    reemplazo) -- mismo principio de bloque por jornada que el resto de
    la etapa, aplicado aqui a la unidad "par (dia_i, dia_i+1)".

    Salida: ``dict`` con ``rho``, ``n_pairs`` (pares minuto-valido
    presentes en ambos dias, sumados sobre todos los pares de dias),
    ``n_day_pairs`` (numero de pares de dias consecutivos), ``ci_lo``,
    ``ci_hi``.
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
# TH21 -- fraccion de dependencia removida por el reloj / que sobrevive
# ---------------------------------------------------------------------------

def dependence_energy(acf_tab: pd.DataFrame, m: int) -> float:
    """``Q(m) = sum_{k=1}^{m} n_pairs_k * rho_k^2`` -- la MISMA metrica de portmanteau de TDA-08, reutilizada como "cantidad total de dependencia" no negativa.

    Se elige esta metrica (en vez de inventar una nueva) porque ya es la
    unica cantidad de la etapa que (a) pondera cada rezago por su propia
    precision (``n_pairs_k``, mas pares = mas confianza), (b) es
    monotonamente no negativa por construccion, y (c) ya se calcula de
    todas formas para el portmanteau -- reusarla evita definir una segunda
    convencion de "energia de dependencia" sin necesidad (G4).
    """
    q_tab = compute_portmanteau_q(acf_tab, (m,))
    row = q_tab.loc[q_tab["m"] == m].iloc[0]
    return float(row["Q"]) if bool(row["estimable"]) else float("nan")


def clock_attribution(acf_raw: pd.DataFrame, acf_adjusted: pd.DataFrame, m: int = TH21_ENERGY_M) -> dict:
    """Fraccion de la dependencia (energia ``Q(m)``) removida/que sobrevive al ajuste por ``s(m)``.

    ``fraccion_removida = 1 - Q_adjusted/Q_raw``; ``fraccion_sobrevive =
    Q_adjusted/Q_raw``. Si ``Q_raw`` es 0 o no finito, ambas fracciones se
    reportan como ``NaN`` explicito -- nunca se fuerza un porcentaje sobre
    un denominador inestable (seccion 17 de la tarea).
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

    ``attributions`` es una lista de resultados de ``clock_attribution``
    (una entrada por variable, p.ej. ``|r|`` y ``log_hl``). STOP-9 se
    declara ACTIVADO solo si TODAS las variables evaluadas muestran una
    fraccion removida por encima del umbral (colapso practicamente
    completo en cada una) -- si UNA variable muestra dependencia que
    sobrevive de forma material, no se activa (el roadmap exige "colapsa
    PRACTICAMENTE POR COMPLETO", no "colapsa en promedio").
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


# ---------------------------------------------------------------------------
# TH20 -- forma del decaimiento (log-log vs semi-log), solo si TH19/TH21
# habilitan la pregunta
# ---------------------------------------------------------------------------

def decay_form_diagnostic(acf_tab: pd.DataFrame) -> dict:
    """Compara un ajuste log-log (``log(rho_k) ~ log(k)``) contra uno semi-log (``log(rho_k) ~ k``) sobre los rezagos con ``rho_k>0`` estimable.

    NO estima ningun parametro de memoria larga (``d``) ni declara
    "memoria larga verdadera" -- solo reporta cual de las dos formas
    describe mejor (mayor R^2) la ACF observada, como diagnostico
    descriptivo (seccion 18 de la tarea). Ambigueadad de interpretacion
    (cambios de nivel no modelados vs memoria larga genuina) queda
    EXPLICITAMENTE abierta para TDA-14 -- no se resuelve aqui.
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
        form_hint = "mas compatible con log-log (posible decaimiento polinomial / persistencia lenta)"
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
    arch_lm_table: pd.DataFrame
    g2_null1_calibration: pd.DataFrame
    g2_secondary_global_calibration: pd.DataFrame
    g2_synthetic_moment_check: dict
    mean_removal_sensitivity_table: pd.DataFrame
    clock_flatness_table: pd.DataFrame
    decay_diagnostics: dict[str, dict]
    th20_enabled: bool


def run_tda09_analysis(config: SnapshotConfig) -> TDA09Result:
    """Orquesta TDA-09 completo: ACF/portmanteau/bootstrap de magnitud (crudo vs ajustado), G2, LM de Engle, atribucion de reloj (TH21), STOP-9, forma del decaimiento (TH20 condicional)."""
    validate_research_holdout_disjoint(config)

    variables, validity = load_tda04_inputs(config)
    last_timestamps = variables.groupby("source_file")["timestamp"].max().to_dict()
    validate_last_timestamps_before_boundary(config, last_timestamps)
    verify_timestamp_alignment(variables, validity)

    n_boot = config.tda09_n_boot
    n_perm = config.tda09_n_perm
    n_perm_arch_lm = config.tda09_n_perm_arch_lm

    cutoffs = load_segmentation_cutoffs(config)

    # --- Poblaciones -------------------------------------------------------
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

    # --- Bloques de continuidad por poblacion -------------------------------
    block_ids_r1m = compute_block_ids(r1m_pop["timestamp"], r1m_pop["trading_date"], 60.0)
    block_ids_r_tilde = compute_block_ids(r_tilde_pop["timestamp"], r_tilde_pop["trading_date"], 60.0)
    block_ids_log_hl = compute_block_ids(log_hl_pop["timestamp"], log_hl_pop["trading_date"], 60.0)
    block_ids_log_hl_tilde = compute_block_ids(log_hl_tilde_pop["timestamp"], log_hl_tilde_pop["trading_date"], 60.0)

    # --- Sensibilidad de media (seccion 8) ----------------------------------
    mean_removal_table = mean_removal_sensitivity(r1m_pop, block_ids_r1m)

    # --- Series de magnitud (crudo y ajustado) ------------------------------
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

    acf_rows = []
    portmanteau_rows = []
    bootstrap_rows = []
    g2_null1_rows = []
    g2_secondary_rows = []
    acf_lookup: dict[tuple[str, str], pd.DataFrame] = {}

    for name, kind, values, block_ids, minute_of_day, has_g2_null, in_bootstrap, _ in series_spec:
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
        port.insert(0, "variable", name)
        port.insert(1, "raw_adjusted", kind)
        portmanteau_rows.append(port)

        if in_bootstrap:
            trading_date_arr = pop_by_key[(name, kind)]["trading_date"].to_numpy()
            rho_boot, _ = bootstrap_rho(values, block_ids, trading_date_arr, BOOTSTRAP_LAGS_MAGNITUDE, n_boot, DEFAULT_BOOTSTRAP_SEED)
            acf_idx = acf_tab.set_index("lag")
            for li, k in enumerate(BOOTSTRAP_LAGS_MAGNITUDE):
                finite = rho_boot[:, li][~np.isnan(rho_boot[:, li])]
                bootstrap_rows.append({
                    "variable": name, "raw_adjusted": kind, "kind": "continuous_lag", "label": str(k),
                    "rho_point": float(acf_idx.loc[k, "rho"]) if k in acf_idx.index else np.nan,
                    "n_pairs": int(acf_idx.loc[k, "n_pairs"]) if k in acf_idx.index else 0,
                    "ci_lo": float(np.percentile(finite, 2.5)) if finite.size else np.nan,
                    "ci_hi": float(np.percentile(finite, 97.5)) if finite.size else np.nan,
                })

        if has_g2_null:
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

    acf_table = pd.concat(acf_rows, ignore_index=True)
    portmanteau_table = pd.concat(portmanteau_rows, ignore_index=True)
    bootstrap_table = pd.DataFrame(bootstrap_rows)
    g2_null1_calibration = pd.concat(g2_null1_rows, ignore_index=True) if g2_null1_rows else pd.DataFrame()
    g2_secondary_global_calibration = pd.concat(g2_secondary_rows, ignore_index=True) if g2_secondary_rows else pd.DataFrame()

    # --- Diagnostico de "flatness" de reloj (justifica/documenta el null secundario) ---
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

    # --- Same-clock-position entre jornadas (seccion 11) --------------------
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
    bootstrap_table = pd.concat([bootstrap_table, pd.DataFrame(same_clock_rows)], ignore_index=True)

    # --- TH21: atribucion de reloj + STOP-9 ---------------------------------
    clock_attr_rows = []
    for name in ("abs_r", "log_hl"):
        attr = clock_attribution(acf_lookup[(name, "raw")], acf_lookup[(name, "adjusted")], TH21_ENERGY_M)
        attr["variable"] = name
        clock_attr_rows.append(attr)
    clock_attribution_table = pd.DataFrame(clock_attr_rows)
    stop9_decision = decide_stop9(clock_attr_rows)

    # --- Persistencia por año / segmento (grilla pequeña predeclarada) ------
    year_rows = []
    segment_rows = []
    for name, kind, values, block_ids, _mod, _has_null, _in_boot, in_stability in series_spec:
        if not in_stability:
            continue
        pop = pop_by_key[(name, kind)]
        yt = acf_by_group(values, block_ids, pop["year_ny"].to_numpy(), STABILITY_LAGS, MIN_N_FOR_GROUP)
        yt.insert(0, "variable", name)
        yt.insert(1, "raw_adjusted", kind)
        year_rows.append(yt)
        st = acf_by_group(values, block_ids, pop["segment_label"].to_numpy(), STABILITY_LAGS, MIN_N_FOR_GROUP)
        st.insert(0, "variable", name)
        st.insert(1, "raw_adjusted", kind)
        segment_rows.append(st)
    persistence_by_year = pd.concat(year_rows, ignore_index=True)
    persistence_by_segment = pd.concat(segment_rows, ignore_index=True)

    # --- LM de Engle (r_1m crudo y r_tilde ajustado) ------------------------
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

    # --- G2 synthetic (Null 2, TDA-08 heredado) sobre |r| crudo -------------
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

    # --- TH20: forma del decaimiento, condicional a supervivencia material --
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

    return TDA09Result(
        r1m_population=r1m_pop, r_tilde_population=r_tilde_pop,
        log_hl_population=log_hl_pop, log_hl_tilde_population=log_hl_tilde_pop,
        segment_labels=segment_labels,
        acf_table=acf_table, bootstrap_table=bootstrap_table, portmanteau_table=portmanteau_table,
        clock_attribution_table=clock_attribution_table, stop9_decision=stop9_decision,
        persistence_by_year=persistence_by_year, persistence_by_segment=persistence_by_segment,
        arch_lm_table=arch_lm_table,
        g2_null1_calibration=g2_null1_calibration, g2_secondary_global_calibration=g2_secondary_global_calibration,
        g2_synthetic_moment_check=moment_check,
        mean_removal_sensitivity_table=mean_removal_table, clock_flatness_table=clock_flatness_table,
        decay_diagnostics=decay_diagnostics, th20_enabled=th20_enabled,
    )
