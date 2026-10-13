"""TDA-13 Parte A -- Extremos: inventario, triage y agrupamiento (TH06, TH27).

Roadmap: ``docs/methodology/Tsay_OHLCV_analysis_roadmap.md`` § TDA-13,
Parte A (OBLIGATORIA). Backlog: ``TH06`` (triage de extremos) y ``TH27``
(agrupamiento de extremos). La Parte B (EVT / POT-GPD, ``TH28``) NO se
implementa aqui: este modulo no ajusta ninguna distribucion de cola ni
estima el indice extremal (metodo avanzado que el roadmap deja fuera del
minimo, G4).

Es caracterizacion estadistica: no construye targets, señales, features
ni modelos; no abre ``data/raw/`` ni ``holdout_files``; no modifica
ningun artefacto de TDA-00..TDA-12.

---------------------------------------------------------------------
HEREDADO (se reutiliza sin redefinir)
---------------------------------------------------------------------

- ``r_t`` = ``r_1m`` con ``r_1m_valid=True`` (TDA-04). Los retornos que
  cruzan un roll, una frontera de jornada o un hueco (``invalid_reason``
  de TDA-04) NO forman parte de esa poblacion; aqui se recuperan SOLO
  para el triage, con el retorno contrafactual ``r_naive_1m`` de TDA-07
  (``build_contrafactual_table``), nunca escrito sobre TDA-04.
- Segmentos horarios: cortes estables de TDA-06.
- ``sigma_hat_(t-1)`` y ``z_t = r_t / sigma_hat_(t-1)``: estimador
  PRIMARIO de TDA-10 (EWMA, semivida 60, sobre ``r_1m``), con sus
  funciones sin modificar. Estado de volatilidad: decil CAUSAL de TDA-12
  (``causal_volatility_decile``: fronteras con jornadas anteriores).
- Niveles de cola: los de ``QUANTILE_LEVELS`` de TDA-07 (1 %/99 %,
  0,1 %/99,9 %, 5 %/95 %). Bootstrap: jornadas completas con reemplazo,
  ``n_boot`` de configuracion, semilla 0, IC percentil 95 % (politica de
  TDA-07..TDA-12).
- Evidencia de calidad: mascara ``bad_data`` (TDA-00), categoria de
  barra plana / candidata a forward-fill (TDA-02/05), causa de cada
  hueco (TDA-02), y la firma de "tick aislado" de TDA-03 (revierte mas
  de la mitad en la barra siguiente; volumen bajo el percentil 5 global).

---------------------------------------------------------------------
DECISIONES NUEVAS DE ESTA ETAPA -- EXPLORATORIAS
(fijadas en el codigo ANTES de la primera ejecucion sobre el conjunto de
investigacion; se someten a revision)
---------------------------------------------------------------------

D1. **Definicion de extremo.** Excedencia de un umbral RELATIVO a la
    celda ``segmento horario x decil causal de volatilidad``
    (``SEGMENT_X_VOL``, definicion PRIMARIA, roadmap metodo 1): ``x`` es
    extremo negativo si ``x < q_p(celda)`` y positivo si
    ``x > q_(1-p)(celda)``. Nivel primario ``p = 1 %`` por cola;
    sensibilidades ``0,1 %`` y ``5 %``. Como contraste se calculan
    tambien las definiciones ``SEGMENT`` (solo segmento) y ``GLOBAL``
    (un umbral para toda la muestra: el anti-patron que el roadmap
    advierte, incluido solo para mostrar su efecto). Los umbrales usan
    toda la muestra: son RETROSPECTIVOS (el estado de volatilidad es
    causal; el umbral no).
D2. **Poblacion comun.** Las tres definiciones y las dos series (``r``
    y ``z``) se evaluan sobre las MISMAS barras: retorno valido, ``z``
    finito y decil causal definido. Asi las tasas de excedencia son
    comparables por construccion.
D3. **Triage** (``classify_extremes``): reglas ordenadas, la primera que
    aplica decide. ``EVENTO_REAL`` exige evidencia POSITIVA, no la mera
    ausencia de error.
D4. **Referencia de independencia.** Proceso de Bernoulli independiente
    barra a barra (el analogo en tiempo discreto de un Poisson) con la
    MISMA intensidad que lo observado. Referencia PRIMARIA
    ``RELOJ_Y_ANIO``: probabilidad por minuto del reloj (controla la
    estacionalidad horaria) escalada por el nivel de cada año.
    Secundarias: ``RELOJ`` y ``HOMOGENEA`` (el Poisson literal de
    intensidad unica). Las esperanzas son exactas y respetan los bloques
    de continuidad (nunca se mide un tiempo a traves de un hueco, un
    cierre de sesion o un roll).
D5. **Estadisticos de agrupamiento.** Fraccion de tiempos entre
    excedencias ``<= k`` barras (``k`` en 1, 5, 10, 60), fraccion de
    excedencias en rachas de 2 o mas barras consecutivas, longitud media
    de racha e indice de dispersion del conteo diario; cada uno como
    cociente observado / referencia. Estadistico del veredicto: ``k=10``.
D6. **Regla de veredicto de TH27** (``decide_th27``): ver su docstring.
    No usa ningun umbral numerico nuevo: "distinguible" = IC 95 %
    bootstrap del cociente por encima de 1 Y valor observado por encima
    del percentil 97,5 de la referencia simulada.
D7. **Lectura de TH06** (``decide_th06``): el backlog pide "la mayoria
    clasificable"; se lee literalmente como menos de la mitad de los
    extremos en ``INDETERMINADO``.

Unidades (G5): cada cociente se reporta con IC bootstrap por jornadas y
con su traduccion (fraccion de excedencias, numero de barras). No se
reporta ningun p-valor.

---------------------------------------------------------------------
AUDITORIA CORRECTIVA (posterior a la revision externa)
---------------------------------------------------------------------

Se conservan como base de comparacion el nivel (1 %), la definicion
(`SEGMENT_X_VOL`), la ventana (10 barras), las series y las semividas.
Cambian cuatro cosas, cada una por un defecto reproducido:

A. **Referencia `RELOJ_Y_ANIO`.** La formula original,
   ``p(minuto) * tasa(año) / tasa(global)``, no conserva a la vez los
   totales por minuto y por año (ni, por tanto, la intensidad total)
   cuando la composicion horaria difiere entre años. Se sustituye por un
   ajuste proporcional iterativo (`fit_two_way_rates`): el modelo
   multiplicativo ``p = u(minuto) * v(año)`` cuyos totales esperados por
   minuto y por año coinciden con los observados dada la exposicion (el
   numero de barras de cada combinacion). Es el modelo mas sencillo que
   conserva ambos margenes; no añade ningun parametro de ajuste. La
   formula original se conserva como referencia secundaria
   `RELOJ_X_FACTOR_ANIO_PREAUDITORIA` solo para comparar.
B. **Calibracion G2.** El criterio "la MEDIA de los controles cae dentro
   de la banda" no controla las falsas detecciones, y la banda simulada
   (Bernoulli con probabilidades fijas) ignora que umbrales, estados y
   tasas se estiman de los datos. Ahora la banda de decision se obtiene
   del PROCEDIMIENTO COMPLETO aplicado a series sin dependencia
   (`run_null_replicates`), y la tasa de falsas detecciones de la regla
   se mide en replicas independientes de las que fijan la banda
   (`summarize_calibration`), con su intervalo binomial. El criterio de
   aceptacion esta en `G2_NOMINAL_RATE` y en el docstring de
   `summarize_calibration`; se fijo antes de volver a evaluar los datos
   reales.
C. **Incertidumbre.** Los IC por jornadas de `analyze_variant` son
   CONDICIONALES a los umbrales y a la referencia estimados con toda la
   muestra. Para la variante primaria se añade un bootstrap que
   re-estima umbrales y referencia en cada replica
   (`full_reestimation_bootstrap`). `sigma_hat`, `z` y el decil causal
   se mantienen como atributos de cada barra (dependen de su propio
   pasado) y los cortes de segmento son los de TDA-06: esa parte de la
   incertidumbre sigue sin cubrirse y se declara.
D. **TH06.** `EVENTO_REAL` es una clasificacion OPERATIVA (sin indicios
   de artefacto y con evidencia favorable), no una verificacion
   independiente del evento. Las categorias no cambian; cambia su
   interpretacion y se añade la sensibilidad de la lectura a las reglas
   (`triage_sensitivity`).

TH27 se separa en tres afirmaciones (`decide_th27`): (1) hay
agrupamiento frente a la independencia; (2) persiste al estandarizar con
el EWMA elegido; (3) existe un agrupamiento propio de los extremos, no
explicable por cambios de volatilidad. (2) NO implica (3): un estimador
rezagado deja agrupamiento residual aunque los retornos sean
independientes dada la volatilidad. Los controles `VOLATILIDAD_PURA_*`
(`generate_null_series`) miden cuanto.
"""
from __future__ import annotations

import os
import pickle
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd

from ohlcv_dataroad.config import SnapshotConfig
from ohlcv_dataroad.ingest.holdout_guard import (
    validate_last_timestamps_before_boundary,
    validate_research_holdout_disjoint,
)
from ohlcv_dataroad.ingest.tda06_intraday_calendar_profile import COMPLETE_YEARS, attach_calendar_fields
from ohlcv_dataroad.ingest.tda07_marginal_distribution import (
    QUANTILE_LEVELS,
    REASON_NON_CONSECUTIVE_MINUTE,
    REASON_ROLL_BOUNDARY,
    REASON_TRADING_DATE_BOUNDARY,
    TRIM_FRACTION_EACH_TAIL,
    assign_segment_label,
    build_contrafactual_table,
    load_segmentation_cutoffs,
    load_tda04_inputs,
    verify_timestamp_alignment,
)
from ohlcv_dataroad.ingest.tda09_volatility_clustering import NULL_PERCENTILE, compute_block_ids_with_contract
from ohlcv_dataroad.ingest.tda10_scale_vs_shape import (
    EWMA_HALFLIVES_MINUTES,
    MIN_VALID_SIGMA_HAT,
    PRIMARY_ESTIMATOR,
    VOLATILITY_DECILES,
    build_sigma_hat,
    burn_in_length,
    run_causality_checks,
    standardize_return,
)
from ohlcv_dataroad.ingest.tda12_conditional_quantiles import (
    CI_LEVEL,
    DEFAULT_SEED,
    MIN_TAIL_SUPPORT_LOW,
    MIN_TAIL_SUPPORT_UNRELIABLE,
    causal_volatility_decile,
    support_label,
    verify_causal_decile_no_lookahead,
)

# --- D1: niveles y definiciones (todos tomados de QUANTILE_LEVELS de TDA-07)
PRIMARY_LEVEL = 0.01
SENSITIVITY_LEVELS: tuple[float, ...] = (0.001, 0.05)
THRESHOLD_LEVELS: tuple[float, ...] = (PRIMARY_LEVEL,) + SENSITIVITY_LEVELS
assert all(lvl in QUANTILE_LEVELS and (1.0 - lvl) in QUANTILE_LEVELS for lvl in THRESHOLD_LEVELS)

DEF_GLOBAL, DEF_SEGMENT, DEF_SEGMENT_X_VOL = "GLOBAL", "SEGMENT", "SEGMENT_X_VOL"
DEFINITIONS: tuple[str, ...] = (DEF_GLOBAL, DEF_SEGMENT, DEF_SEGMENT_X_VOL)
PRIMARY_DEFINITION = DEF_SEGMENT_X_VOL

SIDE_NEG, SIDE_POS, SIDE_BOTH = "NEG", "POS", "BOTH"
SIDES: tuple[str, ...] = (SIDE_NEG, SIDE_POS, SIDE_BOTH)

SERIES_R = "r_1m"
SERIES_Z = f"z_{PRIMARY_ESTIMATOR[0]}_{PRIMARY_ESTIMATOR[1]:g}_raw"
# Sensibilidad de la estandarizacion: las OTRAS semividas EWMA ya definidas en TDA-10 (ninguna nueva).
Z_SENSITIVITY_HALFLIVES: tuple[float, ...] = tuple(h for h in EWMA_HALFLIVES_MINUTES if h != PRIMARY_ESTIMATOR[1])

# --- D5: ventanas (en barras de 1 minuto) y rachas
CLUSTER_WINDOWS: tuple[int, ...] = (1, 5, 10, 60)
PRIMARY_WINDOW = 10
GAP_BIN_EDGES: tuple[int, ...] = (1, 2, 5, 10, 30, 60)      # tiempos entre excedencias: 1, 2, 3-5, 6-10, 11-30, 31-60, >60
RUN_LENGTH_MAX = 5                                           # rachas de longitud 1, 2, 3, 4 y ">=5"

# --- D4: referencias
NULL_CLOCK_YEAR, NULL_CLOCK, NULL_HOMOGENEOUS = "RELOJ_Y_ANIO", "RELOJ", "HOMOGENEA"
NULL_CLOCK_YEAR_LEGACY = "RELOJ_X_FACTOR_ANIO_PREAUDITORIA"   # formula original (hallazgo A): solo para comparar
PRIMARY_NULL = NULL_CLOCK_YEAR
NULLS: tuple[str, ...] = (NULL_CLOCK_YEAR, NULL_CLOCK, NULL_HOMOGENEOUS, NULL_CLOCK_YEAR_LEGACY)
NULL_SIM_SEED = 13
CALIBRATION_SEED = 131
BOOTSTRAP_FULL_SEED = 132
IPF_MAX_ITER = 500
IPF_TOL = 1e-9          # error absoluto maximo admitido en un margen (en numero de excedencias)

# --- Calibracion G2 (hallazgo B). Familias SIN dependencia temporal: fijan la banda y miden las falsas detecciones.
NULL_FAMILY_PERMUTATION = "PERMUTACION_GLOBAL"
NULL_FAMILY_CLOCK = "SIMULADO_PERFIL_RELOJ"
# Tercera familia (segunda auditoria, H3): permutar dentro de cada (minuto de Nueva York, año). Conserva la distribucion de cada minuto EN CADA AÑO,
# que es la estructura que la referencia `RELOJ_Y_ANIO` dice controlar; las dos anteriores mezclan los años.
NULL_FAMILY_MINUTE_YEAR = "PERMUTACION_MINUTO_X_ANIO"
PREVIOUS_INDEPENDENCE_FAMILIES: tuple[str, ...] = (NULL_FAMILY_PERMUTATION, NULL_FAMILY_CLOCK)      # las usadas hasta la segunda auditoria (solo para comparar)
INDEPENDENCE_FAMILIES: tuple[str, ...] = PREVIOUS_INDEPENDENCE_FAMILIES + (NULL_FAMILY_MINUTE_YEAR,)
MINUTES_PER_DAY = 24 * 60
SMALL_GROUP_SIZE = 30            # un grupo de permutacion con menos barras que esto apenas tiene permutaciones distintas utiles (se reporta; no se fusiona)
# Controles de "solo volatilidad": retornos independientes DADA una senda de volatilidad tomada de los datos.
# No calibran la banda; sirven para separar la afirmacion 2 de la 3.
VOL_FAMILY_EWMA_FAST = f"VOLATILIDAD_PURA_EWMA_{min(EWMA_HALFLIVES_MINUTES):g}"
VOL_FAMILY_EWMA_PRIMARY = f"VOLATILIDAD_PURA_EWMA_{PRIMARY_ESTIMATOR[1]:g}"
VOL_FAMILY_LOCAL = f"VOLATILIDAD_PURA_LOCAL_{PRIMARY_WINDOW}"
VOLATILITY_FAMILIES: tuple[str, ...] = (VOL_FAMILY_EWMA_PRIMARY, VOL_FAMILY_EWMA_FAST, VOL_FAMILY_LOCAL)
ALL_FAMILIES: tuple[str, ...] = INDEPENDENCE_FAMILIES + VOLATILITY_FAMILIES
# Diagnostico de la banda antigua: excedencias Bernoulli con la tasa conocida frente a la tasa re-estimada.
DIAG_BERNOULLI_KNOWN, DIAG_BERNOULLI_REFIT = "BERNOULLI_TASA_CONOCIDA", "BERNOULLI_TASA_REESTIMADA"
G2_NOMINAL_RATE = (100.0 - NULL_PERCENTILE) / 100.0     # 2,5 %: cola superior nominal de la banda
G2_CI_LEVEL = 0.95
# Resultado de G2 (segunda auditoria, H2): pasar el criterio es NO RECHAZAR la calibracion en los mundos nulos evaluados, no certificar la tasa nominal.
G2_STATUS_COMPATIBLE = "G2_COMPATIBLE_CON_CONTROLES_EVALUADOS"
G2_STATUS_INCOMPATIBLE = "G2_INCOMPATIBLE_CON_CONTROLES_EVALUADOS"
G2_STATUS_NOT_EVALUABLE = "G2_NO_EVALUABLE"
G2_POWER_RATES = (0.025, 0.04, 0.05, 0.06, 0.075, 0.10)   # tasas verdaderas para las que se reporta cuanto distingue el criterio

# --- Triage
CAT_ERROR, CAT_ROLL, CAT_BOUNDARY, CAT_REAL, CAT_UNDETERMINED = (
    "ERROR_DATO", "ARTEFACTO_ROLL", "ARTEFACTO_FRONTERA", "EVENTO_REAL", "INDETERMINADO",
)
CATEGORIES: tuple[str, ...] = (CAT_ERROR, CAT_ROLL, CAT_BOUNDARY, CAT_REAL, CAT_UNDETERMINED)
FF_ACTIVE = "ACTIVA"
GAP_CAUSE_UNKNOWN = "UNKNOWN"
# Firma de "tick aislado" de TDA-03 (`find_extreme_jumps`), reutilizada tal cual:
REVERSAL_FRACTION_TDA03 = 0.5
LOW_VOLUME_QUANTILE_TDA03 = 0.05
# Contexto de reloj (NO es atribucion): unico horario macro citado en el repositorio (TDA-06: 08:30 NY).
MACRO_CLOCK_MINUTE = 8 * 60 + 30
MACRO_CLOCK_WINDOW_MINUTES = 5

POP_ELIGIBLE, POP_VALID_NO_STATE, POP_CROSSING = "VALIDO_CON_ESTADO", "VALIDO_SIN_ESTADO", "CRUCE_FRONTERA"

# --- Veredictos
TH27_NONE = "SIN_AGRUPAMIENTO_DETECTABLE"
TH27_EXPLAINED = "AGRUPAMIENTO_NO_DETECTABLE_TRAS_ESTANDARIZAR"
TH27_PERSISTS = "AGRUPAMIENTO_PERSISTE_CON_EWMA_ORIGEN_NO_IDENTIFICADO"
TH27_INCONCLUSIVE = "INCONCLUSO"
CLAIM_YES, CLAIM_NO, CLAIM_INCONCLUSIVE, CLAIM_NOT_DEMONSTRABLE = "DETECTADO", "NO_DETECTADO", "INCONCLUSO", "NO_IDENTIFICADO_POR_ESTE_PROCEDIMIENTO"
TH06_BASIS = {
    "ERROR_DATO": "DEMOSTRADO (invariante fisico de TDA-00)",
    "ARTEFACTO_ROLL": "ESTRUCTURAL (el retorno cruza un cambio de contrato)",
    "ARTEFACTO_FRONTERA": "ESTRUCTURAL (el retorno cruza una jornada o un hueco)",
    "EVENTO_REAL": "OPERATIVA_NO_VERIFICADA (sin indicios de artefacto y con evidencia favorable; no confirmado con una fuente independiente)",
    "INDETERMINADO": "SIN_EVIDENCIA_SUFICIENTE",
}
REVERSAL_SENSITIVITY_FRACTIONS: tuple[float, ...] = (0.25, 0.75, 1.0)   # sensibilidad descriptiva alrededor del 0,5 de TDA-03
TH06_MOSTLY_CLASSIFIABLE = "MAYORIA_CLASIFICABLE"
TH06_HIGH_UNDETERMINED = "FRACCION_ALTA_INDETERMINADA"
TH06_MAJORITY = 0.5   # lectura literal de "la mayoria" (D7)


class MissingEvidenceError(Exception):
    """Falta, o no cuadra fila a fila, una mascara de calidad necesaria para el triage: no se clasifica con evidencia parcial."""


# ---------------------------------------------------------------------------
# Carga y universo de triage
# ---------------------------------------------------------------------------

def load_universe(config: SnapshotConfig) -> pd.DataFrame:
    """Una fila por barra de la serie canonica (TDA-04), con el retorno contrafactual y TODA la evidencia de contexto que usa el triage.

    Nada se recalcula de etapas anteriores: se leen sus artefactos y se
    exige que cuadren fila a fila (`MissingEvidenceError` si no).
    """
    validate_research_holdout_disjoint(config)
    variables, validity = load_tda04_inputs(config)
    validate_last_timestamps_before_boundary(config, variables.groupby("source_file")["timestamp"].max().to_dict())
    verify_timestamp_alignment(variables, validity)
    contra = build_contrafactual_table(variables, validity)

    df = attach_calendar_fields(variables, validity)
    cutoffs = load_segmentation_cutoffs(config)
    df["segment_label"], _ = assign_segment_label(df["minute_of_day"], cutoffs)
    df["r_naive_1m"] = contra["r_naive_1m"].to_numpy()
    df["invalid_reason"] = validity.sort_values("timestamp")["invalid_reason"].to_numpy()

    ticks = pd.read_parquet(config.tda05_tick_variables_parquet_path, columns=["timestamp", "ff_category"]).sort_values("timestamp")
    if len(ticks) != len(df) or not np.array_equal(ticks["timestamp"].to_numpy(), df["timestamp"].to_numpy()):
        raise MissingEvidenceError("tda05_tick_variables no comparte los timestamps de TDA-04: no se puede adjuntar `ff_category`.")
    df["ff_category"] = ticks["ff_category"].astype(str).to_numpy()

    bad = pd.read_parquet(config.bad_data_mask_path, columns=["source_file", "timestamp", "bad_data"])
    merged = df[["source_file", "timestamp"]].merge(bad, on=["source_file", "timestamp"], how="left", validate="one_to_one")
    if merged["bad_data"].isna().any():
        raise MissingEvidenceError(f"{int(merged['bad_data'].isna().sum())} barras sin fila en la mascara `bad_data` de TDA-00.")
    df["bad_data"] = merged["bad_data"].astype(bool).to_numpy()

    gaps = pd.read_csv(config.tda02_gaps_csv_path, usecols=["before_ts_utc", "after_ts_utc", "cause"])
    unknown = gaps.loc[gaps["cause"] == GAP_CAUSE_UNKNOWN]
    to_naive = lambda s: pd.to_datetime(s, utc=True).dt.tz_localize(None)   # noqa: E731
    ts = df["timestamp"]
    df["last_before_unknown_gap"] = ts.isin(to_naive(unknown["before_ts_utc"])).to_numpy()
    df["first_after_unknown_gap"] = ts.isin(to_naive(unknown["after_ts_utc"])).to_numpy()
    return add_context_flags(df)


def add_context_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Añade la evidencia derivada (barra anterior/siguiente). Entrada ordenada por `timestamp`; no modifica ninguna columna existente."""
    out = df.copy()
    ts = out["timestamp"]
    nxt_ok = (
        (ts.shift(-1) - ts).dt.total_seconds().eq(60)
        & (out["trading_date"].shift(-1) == out["trading_date"])
        & (out["contract"].shift(-1) == out["contract"])
    )
    out["next_bar_contiguous"] = nxt_ok.to_numpy()
    jump = out["close"] - out["close"].shift(1)
    nxt = out["close"].shift(-1) - out["close"]
    # Formula de TDA-03: signo opuesto y mas de la mitad de la magnitud del salto, en la barra siguiente.
    reverts = (np.sign(nxt) != np.sign(jump)) & (nxt.abs() > REVERSAL_FRACTION_TDA03 * jump.abs())
    out["reverts_next_bar"] = (reverts & nxt_ok).to_numpy()
    # Parte del salto que se deshace en la barra siguiente (positivo = retrocede); NaN si no hay barra siguiente contigua o el salto es nulo.
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where((jump.to_numpy() != 0) & nxt_ok.to_numpy(), -nxt.to_numpy() / jump.to_numpy(), np.nan)
    out["reversal_fraction_next_bar"] = frac
    out["low_volume_threshold"] = float(out["volume"].quantile(LOW_VOLUME_QUANTILE_TDA03))
    out["low_volume"] = (out["volume"] < out["low_volume_threshold"]).to_numpy()
    out["bad_data_prev"] = out["bad_data"].shift(1, fill_value=False).to_numpy()
    out["ff_category_prev"] = out["ff_category"].shift(1, fill_value=FF_ACTIVE).to_numpy()
    # Borde de un hueco de causa desconocida SIN cruzarlo: la barra previa es la primera tras el hueco, o esta es la ultima antes.
    out["adjacent_unknown_gap"] = (out["first_after_unknown_gap"].shift(1, fill_value=False) | out["last_before_unknown_gap"]).to_numpy()
    out["clock_0830_window"] = ((out["minute_of_day"] >= MACRO_CLOCK_MINUTE) & (out["minute_of_day"] < MACRO_CLOCK_MINUTE + MACRO_CLOCK_WINDOW_MINUTES)).to_numpy()
    return out


# ---------------------------------------------------------------------------
# Umbrales relativos y excedencias (D1, D2)
# ---------------------------------------------------------------------------

def cell_codes(definition: str, segment: np.ndarray, decile: np.ndarray) -> np.ndarray:
    """Etiqueta de celda de cada barra segun la definicion de extremo."""
    seg = np.asarray(segment).astype(str)
    if definition == DEF_GLOBAL:
        return np.full(seg.size, "ALL", dtype=object)
    if definition == DEF_SEGMENT:
        return seg.astype(object)
    if definition == DEF_SEGMENT_X_VOL:
        dec = pd.Series(np.asarray(decile, dtype=float).astype(int).astype(str))
        return (pd.Series(seg) + "|D" + dec).to_numpy(dtype=object)
    raise ValueError(f"definicion desconocida: {definition}")


def compute_thresholds(x: np.ndarray, cells: np.ndarray, level: float) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Umbrales `q_level` y `q_(1-level)` de `x` por celda (convencion de cuantil de TDA-07) y excedencias ESTRICTAS.

    Salida: tabla por celda (`n`, umbrales, excedencias a cada lado, tasa
    realizada, soporte) y cuatro vectores alineados con `x`:
    `exceed_neg`, `exceed_pos`, `thr_lo`, `thr_hi`. Con valores discretos
    (ticks) puede haber empates en el umbral: la desigualdad estricta
    hace que la tasa realizada sea <= `level`; se reporta, no se corrige.
    """
    x = np.asarray(x, dtype=float)
    codes, uniques = pd.factorize(pd.Series(cells), sort=True)
    thr_lo, thr_hi = np.full(x.size, np.nan), np.full(x.size, np.nan)
    rows = []
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniques) + 1))
    for c, name in enumerate(uniques):
        idx = order[bounds[c]:bounds[c + 1]]
        lo, hi = np.quantile(x[idx], [level, 1.0 - level])
        thr_lo[idx], thr_hi[idx] = lo, hi
        n_neg, n_pos = int((x[idx] < lo).sum()), int((x[idx] > hi).sum())
        rows.append({
            "cell": name, "n": int(idx.size), "level": level, "threshold_lo": float(lo), "threshold_hi": float(hi),
            "n_exceed_neg": n_neg, "n_exceed_pos": n_pos,
            "rate_neg": n_neg / idx.size, "rate_pos": n_pos / idx.size,
            "expected_per_tail": level * idx.size,
            "support": support_label(min(n_neg, n_pos)),
        })
    return pd.DataFrame(rows), x < thr_lo, x > thr_hi, thr_lo, thr_hi


# ---------------------------------------------------------------------------
# Triage (D3) -- criterios escritos antes de medir su efecto
# ---------------------------------------------------------------------------

TRIAGE_RULES_TEXT: tuple[tuple[str, str, str], ...] = (
    ("R1", CAT_ERROR, "La barra o la barra anterior viola un invariante fisico de TDA-00 (`bad_data`). Es la unica evidencia que DEMUESTRA un error."),
    ("R2", CAT_ROLL, "El retorno cruza un cambio de contrato (`invalid_reason = ROLL_BOUNDARY`, TDA-04)."),
    ("R3", CAT_BOUNDARY, "El retorno cruza una frontera de jornada o un hueco (`TRADING_DATE_BOUNDARY`, `NON_CONSECUTIVE_MINUTE`): no es un retorno de 1 minuto."),
    ("R4", CAT_UNDETERMINED, "Retorno valido con ALGUNA duda: (a) barra o barra anterior plana o candidata a forward-fill; (b) pegado a un hueco de causa desconocida; "
                             "(c) no existe barra siguiente contigua para comprobar si el precio se sostiene; (d) revierte mas de la mitad en la barra siguiente "
                             "(firma de TDA-03; sin ticks no se distingue un pico real de un ultimo precio anomalo); (e) volumen bajo el percentil 5 global (TDA-03)."),
    ("R5", CAT_REAL, "Retorno valido SIN ninguna de las dudas de R4: barras activas, registro continuo alrededor, precio sostenido en la barra siguiente y volumen no bajo. "
                     "Es evidencia positiva de que hubo negociacion a ese precio; no identifica la causa del movimiento."),
)


def classify_extremes(inv: pd.DataFrame) -> pd.DataFrame:
    """Aplica R1..R5 en orden (la primera que aplica decide). Devuelve `category`, `triage_rule` y `triage_evidence` (todas las dudas que aplican, separadas por `;`)."""
    n = len(inv)
    reason = inv["invalid_reason"].to_numpy()
    r1 = (inv["bad_data"] | inv["bad_data_prev"]).to_numpy()
    r2 = reason == REASON_ROLL_BOUNDARY
    r3 = np.isin(reason, [REASON_TRADING_DATE_BOUNDARY, REASON_NON_CONSECUTIVE_MINUTE])
    doubts = {
        "BARRA_NO_ACTIVA": ((inv["ff_category"] != FF_ACTIVE) | (inv["ff_category_prev"] != FF_ACTIVE)).to_numpy(),
        "JUNTO_A_HUECO_DESCONOCIDO": inv["adjacent_unknown_gap"].to_numpy(),
        "SIN_BARRA_SIGUIENTE": (~inv["next_bar_contiguous"]).to_numpy(),
        "REVIERTE_BARRA_SIGUIENTE": inv["reverts_next_bar"].to_numpy(),
        "VOLUMEN_BAJO": inv["low_volume"].to_numpy(),
    }
    any_doubt = np.zeros(n, dtype=bool)
    evidence = np.full(n, "", dtype=object)
    for name, flag in doubts.items():
        any_doubt |= flag
        evidence = np.where(flag, np.where(evidence == "", name, evidence + ";" + name), evidence)
    category = np.where(r1, CAT_ERROR, np.where(r2, CAT_ROLL, np.where(r3, CAT_BOUNDARY, np.where(any_doubt, CAT_UNDETERMINED, CAT_REAL))))
    rule = np.where(r1, "R1", np.where(r2, "R2", np.where(r3, "R3", np.where(any_doubt, "R4", "R5"))))
    ev = np.where(r1, "bad_data(TDA-00)", np.where(r2 | r3, reason, np.where(any_doubt, evidence, "SIN_DUDAS_R4")))
    out = inv.copy()
    out["category"], out["triage_rule"], out["triage_evidence"] = category, rule, ev
    return out


# ---------------------------------------------------------------------------
# Agrupamiento: conteos observados y esperanzas exactas bajo independencia (D4, D5)
# ---------------------------------------------------------------------------

def continuity_flags(block_ids: np.ndarray) -> np.ndarray:
    """`cont[i] = 1.0` si la barra `i` continua el bloque de la barra `i-1` (mismo bloque de continuidad), 0.0 si abre bloque."""
    b = np.asarray(block_ids)
    cont = np.zeros(b.size, dtype=float)
    if b.size > 1:
        cont[1:] = (b[1:] == b[:-1]).astype(float)
    return cont


def restrict_block_ids(block_ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Bloques de continuidad de la subpoblacion `mask`, sin unir nunca dos tramos separados por una barra excluida (segunda auditoria, H5).

    Filtrar sin mas (`block_ids[mask]`) dejaria en el mismo bloque a dos
    barras entre las que se ha quitado otra: pasarian por contiguas y el
    tiempo entre excedencias se contaria de menos. Aqui dos barras
    seleccionadas consecutivas siguen en el mismo bloque solo si tenian
    el mismo `block_id` Y eran adyacentes en la poblacion original.
    Equivale a recalcular los bloques sobre la subpoblacion por marcas
    de tiempo (lo que hace la tabla principal).
    """
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return np.zeros(0, dtype=np.int64)
    b = np.asarray(block_ids)[idx]
    new = np.ones(idx.size, dtype=bool)
    new[1:] = (b[1:] != b[:-1]) | (np.diff(idx) != 1)
    return np.cumsum(new) - 1


def observed_counts(exceed: np.ndarray, block_ids: np.ndarray, day: np.ndarray, n_days: int,
                    ks: tuple[int, ...] = GAP_BIN_EDGES, run_max: int = RUN_LENGTH_MAX) -> dict:
    """Conteos POR JORNADA de lo observado. Un tiempo entre excedencias solo existe entre dos excedencias del MISMO bloque de continuidad.

    Claves: `n` (excedencias), `b` (tiempos entre excedencias), `a[k]`
    (tiempos <= k barras), `runs` (matriz jornada x longitud de racha:
    1..run_max-1 y >=run_max), `n_runs`, `in_runs2` (excedencias en
    rachas de 2 o mas barras consecutivas).
    """
    exc = np.asarray(exceed, dtype=bool)
    idx = np.flatnonzero(exc)
    out = {"n": np.bincount(day[idx], minlength=n_days).astype(float), "b": np.zeros(n_days), "a": {k: np.zeros(n_days) for k in ks},
           "runs": np.zeros((n_days, run_max)), "n_runs": np.zeros(n_days), "in_runs2": np.zeros(n_days)}
    if idx.size == 0:
        return out
    blk = np.asarray(block_ids)[idx]
    gaps = np.diff(idx)
    same = blk[1:] == blk[:-1]
    g, gd = gaps[same], day[idx[1:]][same]
    out["b"] = np.bincount(gd, minlength=n_days).astype(float)
    for k in ks:
        out["a"][k] = np.bincount(gd[g <= k], minlength=n_days).astype(float)
    start = np.ones(idx.size, dtype=bool)
    start[1:] = ~(same & (gaps == 1))
    run_id = np.cumsum(start) - 1
    lengths = np.bincount(run_id)
    run_day = day[idx[start]]
    out["n_runs"] = np.bincount(run_day, minlength=n_days).astype(float)
    col = np.minimum(lengths, run_max) - 1
    np.add.at(out["runs"], (run_day, col), 1.0)
    out["in_runs2"] = np.bincount(day[idx], weights=(lengths[run_id] >= 2).astype(float), minlength=n_days)
    return out


def null_expectations(p: np.ndarray, block_ids: np.ndarray, day: np.ndarray, n_days: int,
                      ks: tuple[int, ...] = GAP_BIN_EDGES, run_max: int = RUN_LENGTH_MAX) -> dict:
    """Esperanza EXACTA, por jornada, de los mismos conteos si cada barra `i` fuese una excedencia independiente con probabilidad `p[i]`.

    Es la referencia de "Poisson de la misma intensidad" en tiempo
    discreto (barras de 1 minuto), sin aproximacion continua, y respeta
    los bloques de continuidad: un tiempo de `d` barras solo cuenta si
    las `d+1` barras pertenecen al mismo bloque.

    - `a[k]`: suma sobre `i` y `d<=k` de `p_i * p_(i+d) * prod(1-p)` de
      las barras intermedias.
    - `b`: por bloque, `sum(p) - (1 - prod(1-p))` (excedencias esperadas
      menos la probabilidad de que haya al menos una).
    - rachas de longitud `>= l` que empiezan en `i`:
      `(1 - p_(i-1)) * p_i * ... * p_(i+l-1)`.
    - `var`: varianza del conteo diario, `sum p(1-p)`.
    """
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0 - 1e-12)
    n = p.size
    cont = continuity_flags(block_ids)
    q = 1.0 - p
    out = {"n": np.bincount(day, weights=p, minlength=n_days), "var": np.bincount(day, weights=p * q, minlength=n_days), "a": {}}
    acc = np.zeros(n_days)
    carry = np.ones(n)
    for d in range(1, max(ks) + 1):
        m = n - d
        if m <= 0:
            if d in ks:
                out["a"][d] = acc.copy()
            continue
        c = carry[:m] * cont[d:]
        acc = acc + np.bincount(day[d:], weights=p[:m] * c * p[d:], minlength=n_days)
        carry = c * q[d:]
        if d in ks:
            out["a"][d] = acc.copy()
    b = np.asarray(block_ids)
    _, first, inv = np.unique(b, return_index=True, return_inverse=True)
    sum_p = np.bincount(inv, weights=p)
    log_q = np.bincount(inv, weights=np.log1p(-p))
    out["b"] = np.bincount(day[first], weights=sum_p - (1.0 - np.exp(log_q)), minlength=n_days)

    prev_p = np.concatenate([[0.0], p[:-1]])
    startf = 1.0 - prev_p * cont
    prod = p.copy()
    ge = [np.bincount(day, weights=startf * prod, minlength=n_days)]
    for length in range(2, run_max + 1):
        m = n - (length - 1)
        if m <= 0:
            ge.append(np.zeros(n_days))
            continue
        prod = prod[:m] * cont[length - 1:] * p[length - 1:]
        ge.append(np.bincount(day[:m], weights=startf[:m] * prod, minlength=n_days))
    runs = np.zeros((n_days, run_max))
    for j in range(run_max - 1):
        runs[:, j] = ge[j] - ge[j + 1]
    runs[:, run_max - 1] = ge[run_max - 1]
    out["runs"], out["n_runs"] = runs, ge[0]
    out["in_runs2"] = out["n"] - runs[:, 0]
    return out


def _safe_div(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / b, np.nan)


def cluster_statistics(obs: dict, null: dict, weights: np.ndarray, windows: tuple[int, ...] = CLUSTER_WINDOWS) -> dict:
    """Estadisticos de agrupamiento (observado, referencia y cociente) para cada fila de `weights` (matriz replicas x jornadas; una fila de unos = estimacion puntual)."""
    w = np.atleast_2d(np.asarray(weights, dtype=float))
    res = {}
    b, b0 = w @ obs["b"], w @ null["b"]
    for k in windows:
        if k not in obs["a"] or k not in null["a"]:
            continue
        s, s0 = _safe_div(w @ obs["a"][k], b), _safe_div(w @ null["a"][k], b0)
        res[f"share_le_{k}"], res[f"share_le_{k}_ref"], res[f"ratio_share_le_{k}"] = s, s0, _safe_div(s, s0)
    n, n0 = w @ obs["n"], w @ null["n"]
    f, f0 = _safe_div(w @ obs["in_runs2"], n), _safe_div(w @ null["in_runs2"], n0)
    res["frac_in_runs_ge2"], res["frac_in_runs_ge2_ref"], res["ratio_frac_in_runs_ge2"] = f, f0, _safe_div(f, f0)
    m, m0 = _safe_div(n, w @ obs["n_runs"]), _safe_div(n0, w @ null["n_runs"])
    res["mean_run_length"], res["mean_run_length_ref"], res["ratio_mean_run_length"] = m, m0, _safe_div(m, m0)
    wsum = w.sum(axis=1)
    mean, mean0 = n / wsum, n0 / wsum
    var = (w @ obs["n"] ** 2) / wsum - mean ** 2
    var0 = (w @ null["var"]) / wsum + (w @ null["n"] ** 2) / wsum - mean0 ** 2
    d, d0 = _safe_div(var, mean), _safe_div(var0, mean0)
    res["daily_dispersion"], res["daily_dispersion_ref"], res["ratio_daily_dispersion"] = d, d0, _safe_div(d, d0)
    res["n_exceed"], res["n_intervals"] = n, b
    return res


def fit_two_way_rates(exceed: np.ndarray, a: np.ndarray, b: np.ndarray, weights: np.ndarray | None = None,
                       max_iter: int = IPF_MAX_ITER, tol: float = IPF_TOL) -> tuple[np.ndarray, dict]:
    """Probabilidad multiplicativa `p = u(a) * v(b)` cuyos totales esperados por nivel de `a` y de `b` igualan a los observados (ajuste proporcional iterativo).

    Dada la exposicion `N[a,b]` (numero de barras de cada combinacion,
    ponderado por `weights` si se pasa) se buscan `u`, `v` tales que
    ``sum_b N[a,b] u_a v_b = O_a`` y ``sum_a N[a,b] u_a v_b = O_b``, con
    `O` las excedencias observadas. Es el estimador de maxima
    verosimilitud del modelo de efectos principales para conteos, y la
    forma mas simple de respetar ambos margenes cuando la composicion de
    `a` difiere entre niveles de `b`.

    Casos limite (documentados, no parcheados):
    - nivel de `a` o de `b` sin ninguna excedencia -> su factor es 0 y
      `p = 0` en esas barras (lo observado: ninguna);
    - combinacion `(a,b)` sin barras -> no interviene (exposicion 0);
    - `p >= 1` -> se recorta a `1 - 1e-9` y se cuenta en `n_clipped`
      (rompe la conservacion exacta; se reporta).

    Salida: `(p por barra, info)` con iteraciones, error maximo de cada
    margen (en excedencias), intensidad total esperada y observada.
    """
    exc = np.asarray(exceed, dtype=float)
    w = np.ones(exc.size) if weights is None else np.asarray(weights, dtype=float)
    ac, _ = pd.factorize(np.asarray(a), sort=True)
    bc, _ = pd.factorize(np.asarray(b), sort=True)
    na, nb = (int(ac.max()) + 1, int(bc.max()) + 1) if exc.size else (0, 0)
    info = {"iterations": 0, "converged": True, "max_abs_error_a": 0.0, "max_abs_error_b": 0.0, "n_clipped": 0,
            "total_observed": float((w * exc).sum()), "total_expected": 0.0, "n_bars_p_zero": 0}
    if exc.size == 0:
        return np.zeros(0), info
    flat = ac * nb + bc
    expo = np.bincount(flat, weights=w, minlength=na * nb).reshape(na, nb)
    oa, ob = np.bincount(ac, weights=w * exc, minlength=na), np.bincount(bc, weights=w * exc, minlength=nb)
    u, v = np.ones(na), np.ones(nb)
    err = np.inf
    for it in range(1, max_iter + 1):
        da = expo @ v
        u = np.where(da > 0, oa / np.where(da > 0, da, 1.0), 0.0)
        db = expo.T @ u
        v = np.where(db > 0, ob / np.where(db > 0, db, 1.0), 0.0)
        err_a, err_b = float(np.abs(u * (expo @ v) - oa).max()), float(np.abs(v * (expo.T @ u) - ob).max())
        err = max(err_a, err_b)
        info.update({"iterations": it, "max_abs_error_a": err_a, "max_abs_error_b": err_b})
        if err < tol:
            break
    info["converged"] = bool(err < tol)
    p = u[ac] * v[bc]
    clipped = p >= 1.0
    info["n_clipped"] = int(clipped.sum())
    p = np.where(clipped, 1.0 - 1e-9, p)
    info["total_expected"] = float((w * p).sum())
    info["n_bars_p_zero"] = int((p == 0).sum())
    return p, info


def reference_probabilities(exceed: np.ndarray, minute: np.ndarray, year: np.ndarray, kind: str, weights: np.ndarray | None = None) -> np.ndarray:
    """Probabilidad de excedencia por barra bajo cada referencia (D4), con la MISMA intensidad total que lo observado.

    `RELOJ`: frecuencia observada por minuto del reloj (controla la
    estacionalidad horaria, tambien dentro de un segmento).
    `RELOJ_Y_ANIO` (primaria): `fit_two_way_rates` sobre (minuto, año):
    conserva el total de excedencias de cada minuto Y de cada año.
    `RELOJ_X_FACTOR_ANIO_PREAUDITORIA`: la formula original
    ``p(minuto) * tasa(año)/tasa(global)``, que NO conserva esos
    margenes; solo para comparar. `HOMOGENEA`: una sola tasa.
    RETROSPECTIVAS todas. `weights` (multiplicidad de cada barra) permite
    re-estimar la referencia dentro de un bootstrap.
    """
    exc = np.asarray(exceed, dtype=float)
    w = np.ones(exc.size) if weights is None else np.asarray(weights, dtype=float)
    wsum = float(w.sum())
    rate = float((w * exc).sum() / wsum) if wsum > 0 else 0.0
    if kind == NULL_HOMOGENEOUS:
        return np.full(exc.size, rate)
    if kind == NULL_CLOCK_YEAR:
        return fit_two_way_rates(exc, minute, year, weights)[0]
    m = np.asarray(minute).astype(int)
    p = _safe_div(np.bincount(m, weights=w * exc, minlength=1440), np.bincount(m, weights=w, minlength=1440))[m]
    p = np.nan_to_num(p, nan=0.0)
    if kind == NULL_CLOCK:
        return p
    if kind == NULL_CLOCK_YEAR_LEGACY:
        codes, _ = pd.factorize(np.asarray(year), sort=True)
        yr = _safe_div(np.bincount(codes, weights=w * exc), np.bincount(codes, weights=w))
        factor = yr[codes] / rate if rate > 0 else np.ones(exc.size)
        return np.clip(p * factor, 0.0, 1.0 - 1e-9)
    raise ValueError(f"referencia desconocida: {kind}")


def reference_conservation(exceed: np.ndarray, minute: np.ndarray, year: np.ndarray, kinds: tuple[str, ...] = NULLS) -> pd.DataFrame:
    """Comprueba que conserva cada referencia: intensidad total y totales por minuto y por año (esperado frente a observado)."""
    exc = np.asarray(exceed, dtype=float)
    m = np.asarray(minute).astype(int)
    yc, yu = pd.factorize(np.asarray(year), sort=True)
    om, oy = np.bincount(m, weights=exc, minlength=1440), np.bincount(yc, weights=exc)
    rows = []
    for kind in kinds:
        p = reference_probabilities(exc, minute, year, kind)
        em, ey = np.bincount(m, weights=p, minlength=1440), np.bincount(yc, weights=p)
        worst = int(np.argmax(np.abs(ey - oy)))
        rows.append({
            "reference": kind, "n_observed": float(exc.sum()), "n_expected": float(p.sum()), "total_error": float(p.sum() - exc.sum()),
            "max_abs_error_by_minute": float(np.abs(em - om).max()), "sum_abs_error_by_minute": float(np.abs(em - om).sum()),
            "max_abs_error_by_year": float(np.abs(ey - oy).max()), "worst_year": str(yu[worst]),
            "worst_year_rel_error": float((ey[worst] - oy[worst]) / oy[worst]) if oy[worst] > 0 else np.nan,
            "n_bars_p_zero": int((p == 0).sum()), "p_max": float(p.max()) if p.size else np.nan,
        })
    return pd.DataFrame(rows)


def bootstrap_day_weights(n_days: int, n_boot: int, seed: int) -> np.ndarray:
    """Multiplicidades de un bootstrap de jornadas completas con reemplazo (mismo esquema que TDA-07/TDA-12): matriz `n_boot x n_days`."""
    rng = np.random.default_rng(seed)
    w = np.zeros((n_boot, n_days))
    for b in range(n_boot):
        w[b] = np.bincount(rng.integers(0, n_days, size=n_days), minlength=n_days)
    return w


def _ci(samples: np.ndarray) -> tuple[float, float]:
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 10:
        return float("nan"), float("nan")
    a = (1.0 - CI_LEVEL) / 2.0
    return float(np.percentile(s, 100 * a)), float(np.percentile(s, 100 * (1 - a)))


def simulate_null_band(p: np.ndarray, block_ids: np.ndarray, day: np.ndarray, n_days: int, null: dict, n_sim: int, seed: int,
                       ks: tuple[int, ...] = GAP_BIN_EDGES) -> dict:
    """Variabilidad de cada cociente cuando la referencia es CIERTA: `n_sim` series de excedencias independientes con probabilidad `p`, pasadas por el mismo calculo."""
    rng = np.random.default_rng(seed)
    ones = np.ones((1, n_days))
    acc: dict[str, list] = {}
    for _ in range(n_sim):
        sim = rng.random(p.size) < p
        st = cluster_statistics(observed_counts(sim, block_ids, day, n_days, ks), null, ones)
        for key, val in st.items():
            if key.startswith("ratio_"):
                acc.setdefault(key, []).append(float(val[0]))
    return {k: np.asarray(v) for k, v in acc.items()}


def analyze_variant(exceed: np.ndarray, minute: np.ndarray, year: np.ndarray, block_ids: np.ndarray, day: np.ndarray, n_days: int,
                    boot_w: np.ndarray, nulls: tuple[str, ...] = (PRIMARY_NULL,), n_sim: int = 0, seed: int = NULL_SIM_SEED,
                    ks: tuple[int, ...] = GAP_BIN_EDGES, row_mask: np.ndarray | None = None) -> tuple[list[dict], dict]:
    """Agrupamiento de UNA serie de excedencias frente a cada referencia. Devuelve filas largas (una por referencia y estadistico) y los conteos para las tablas de distribucion.

    `row_mask` restringe a un subconjunto de barras (un año, un segmento):
    las probabilidades de referencia se calculan con TODA la muestra y
    los bloques se parten donde el subconjunto se interrumpe.
    """
    exc = np.asarray(exceed, dtype=bool)
    probs = {name: reference_probabilities(exc, minute, year, name) for name in nulls}
    blk = np.asarray(block_ids)
    if row_mask is not None:
        keep = np.flatnonzero(row_mask)
        pos_break = np.concatenate([[True], np.diff(keep) != 1])
        blk = np.cumsum(pos_break | np.concatenate([[True], blk[keep][1:] != blk[keep][:-1]]))
        exc, day = exc[keep], day[keep]
        probs = {name: pr[keep] for name, pr in probs.items()}
    obs = observed_counts(exc, blk, day, n_days, ks)
    ones = np.ones((1, n_days))
    rows, detail = [], {"obs": obs, "null": {}}
    for name in nulls:
        null = null_expectations(probs[name], blk, day, n_days, ks)
        detail["null"][name] = null
        point = cluster_statistics(obs, null, ones)
        boot = cluster_statistics(obs, null, boot_w)
        band = simulate_null_band(probs[name], blk, day, n_days, null, n_sim, seed, ks) if (n_sim > 0 and name == PRIMARY_NULL) else {}
        for key in [k for k in point if k.startswith("ratio_")]:
            stat = key[len("ratio_"):]
            lo, hi = _ci(boot[key])
            sim = band.get(key)
            rows.append({
                "reference": name, "statistic": stat,
                "observed": float(point[stat][0]), "expected": float(point[f"{stat}_ref"][0]),
                "ratio": float(point[key][0]), "ratio_ci_lo": lo, "ratio_ci_hi": hi,
                "null_sim_p2_5": float(np.percentile(sim, 100 - NULL_PERCENTILE)) if sim is not None and sim.size else np.nan,
                "null_sim_p97_5": float(np.percentile(sim, NULL_PERCENTILE)) if sim is not None and sim.size else np.nan,
                "n_null_sim": int(sim.size) if sim is not None else 0,
                "n_exceed": int(point["n_exceed"][0]), "n_intervals": int(point["n_intervals"][0]),
            })
    return rows, detail


def is_distinguishable(row: dict | pd.Series) -> bool | None:
    """Criterio PREVIO a la auditoria (ya no decide): IC 95 % CONDICIONAL por encima de 1 Y valor observado por encima del percentil 97,5 de la banda Bernoulli. Se conserva para comparar; la regla vigente es `decide_th27` con la banda calibrada. `None` si falta alguno de los dos."""
    lo, hi = row["ratio_ci_lo"], row["null_sim_p97_5"]
    if not (np.isfinite(lo) and np.isfinite(hi) and np.isfinite(row["ratio"])):
        return None
    return bool(lo > 1.0 and row["ratio"] > hi)


def distribution_tables(detail: dict, null_names: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Distribucion de tiempos entre excedencias (por tramos de `GAP_BIN_EDGES`) y de longitudes de racha: observado frente a cada referencia."""
    obs = detail["obs"]
    edges = list(GAP_BIN_EDGES)
    labels = [f"{'1' if i == 0 else str(edges[i - 1] + 1) + '-' + str(e) if edges[i - 1] + 1 != e else str(e)}" for i, e in enumerate(edges)] + [f">{edges[-1]}"]
    b = float(obs["b"].sum())
    cum = [float(obs["a"][k].sum()) for k in edges]
    counts = np.diff([0.0] + cum + [b])
    gap_rows, run_rows = [], []
    for name in null_names:
        null = detail["null"][name]
        b0 = float(null["b"].sum())
        counts0 = np.diff([0.0] + [float(null["a"][k].sum()) for k in edges] + [b0])
        for lab, c, c0 in zip(labels, counts, counts0):
            gap_rows.append({"reference": name, "gap_bars": lab, "n_observed": c, "share_observed": c / b if b else np.nan,
                             "n_expected": c0, "share_expected": c0 / b0 if b0 else np.nan,
                             "share_ratio": (c / b) / (c0 / b0) if b and b0 and c0 else np.nan})
        r, r0 = obs["runs"].sum(axis=0), null["runs"].sum(axis=0)
        for j in range(RUN_LENGTH_MAX):
            lab = str(j + 1) if j < RUN_LENGTH_MAX - 1 else f">={RUN_LENGTH_MAX}"
            run_rows.append({"reference": name, "run_length": lab, "n_observed": float(r[j]), "share_observed": r[j] / r.sum() if r.sum() else np.nan,
                             "n_expected": float(r0[j]), "share_expected": r0[j] / r0.sum() if r0.sum() else np.nan,
                             "count_ratio": r[j] / r0[j] if r0[j] > 0 else np.nan})
    return pd.DataFrame(gap_rows), pd.DataFrame(run_rows)


# ---------------------------------------------------------------------------
# Pipeline de excedencias sobre una serie de retornos (real o de calibracion)
# ---------------------------------------------------------------------------

@dataclass
class ExceedanceSet:
    """Poblacion comun (D2) y excedencias de `r` y `z` para todas las definiciones y niveles."""
    eligible: np.ndarray                       # mascara sobre la poblacion valida
    z: np.ndarray                              # z_t (poblacion valida completa)
    sigma_prev: np.ndarray
    decile: np.ndarray
    flags: dict = field(default_factory=dict)        # (serie, definicion, nivel) -> (neg, pos) sobre la poblacion ELEGIBLE
    thresholds: dict = field(default_factory=dict)   # (serie, definicion, nivel) -> (tabla por celda, thr_lo, thr_hi)


def standardize_primary(r_valid: np.ndarray, halflife: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """`sigma_hat_(t-1)` y `z_t` con las funciones de TDA-10, sin modificarlas (estimador primario salvo que se pida otra semivida de TDA-10)."""
    fam, par, _ = PRIMARY_ESTIMATOR
    par = par if halflife is None else halflife
    sigma = build_sigma_hat(r_valid, fam, par)
    return sigma, standardize_return(r_valid, sigma, burn_in_length(fam, par))


def causal_volatility_decile_fast(sigma_hat: np.ndarray, trading_date: np.ndarray, valid: np.ndarray | None = None,
                                  n_deciles: int = VOLATILITY_DECILES, chunk_days: int = 32) -> np.ndarray:
    """MISMO resultado, valor a valor, que `causal_volatility_decile` de TDA-12, calculado de otra forma para que miles de replicas sean viables.

    TDA-12 recalcula en cada jornada una suma acumulada sobre toda la
    muestra (coste jornadas x barras). Aqui la historia se guarda
    ordenada y, para cada barra, se cuentan por busqueda binaria los
    valores historicos ESTRICTAMENTE menores: los de jornadas anteriores
    al tramo actual (`hist`) mas los de las jornadas anteriores dentro
    del tramo (`recent`). Misma definicion (`floor(n_deciles * menores /
    historia)`, tope `n_deciles-1`), mismo minimo de historia
    (`n_deciles*20`) y misma mascara de validez. TDA-12 no se modifica;
    la igualdad se comprueba en los tests y, sobre los datos reales, en
    cada ejecucion.
    """
    s = np.asarray(sigma_hat, dtype=float)
    dates = np.asarray(trading_date)
    ok = np.isfinite(s) & (s > 0)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    out = np.full(s.size, np.nan)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return out
    sv, dv = s[idx], dates[idx]
    change = np.concatenate([[True], dv[1:] != dv[:-1]])
    day_start = np.flatnonzero(change)
    day_end = np.append(day_start[1:], sv.size)
    min_history = n_deciles * 20
    dec = np.full(sv.size, np.nan)
    hist, recent, n_prior = np.empty(0), np.empty(0), 0
    for j, (a, b) in enumerate(zip(day_start, day_end)):
        if j % chunk_days == 0 and recent.size:
            hist = np.insert(hist, np.searchsorted(hist, recent), recent)      # fusion de dos listas ordenadas
            recent = np.empty(0)
        x = sv[a:b]
        if n_prior >= min_history:
            less = np.searchsorted(hist, x, side="left") + np.searchsorted(recent, x, side="left")
            dec[a:b] = np.minimum(np.floor(n_deciles * (less / float(n_prior))), n_deciles - 1)
        xs = np.sort(x)
        recent = np.insert(recent, np.searchsorted(recent, xs), xs)
        n_prior += b - a
    out[idx] = dec
    return out


def build_exceedances(r_valid: np.ndarray, trading_date: np.ndarray, segment: np.ndarray,
                      levels: tuple[float, ...] = THRESHOLD_LEVELS, definitions: tuple[str, ...] = DEFINITIONS,
                      halflife: float | None = None, eligible: np.ndarray | None = None, decile: np.ndarray | None = None,
                      decile_fn=causal_volatility_decile) -> ExceedanceSet:
    """Estandariza, asigna el estado causal de volatilidad, fija la poblacion comun y calcula umbrales y excedencias.

    Si se pasan `eligible` y `decile` (los del estimador primario) se
    reutilizan: asi una `z` de sensibilidad se compara sobre las mismas
    barras y las mismas celdas que la primaria.
    """
    r = np.asarray(r_valid, dtype=float)
    sigma, z = standardize_primary(r, halflife)
    if decile is None:
        decile = decile_fn(sigma, trading_date, valid=np.isfinite(z))
    if eligible is None:
        eligible = np.isfinite(z) & np.isfinite(decile) & (np.nan_to_num(decile, nan=-1.0) >= 0)
    eligible = eligible & np.isfinite(z)
    out = ExceedanceSet(eligible=eligible, z=z, sigma_prev=sigma, decile=decile)
    seg_e, dec_e = np.asarray(segment)[eligible], decile[eligible]
    series = {SERIES_R: r[eligible], "z": z[eligible]}
    for definition in definitions:
        cells = cell_codes(definition, seg_e, dec_e)
        for name, x in series.items():
            for lvl in levels:
                table, neg, pos, lo, hi = compute_thresholds(x, cells, lvl)
                out.flags[(name, definition, lvl)] = (neg, pos)
                out.thresholds[(name, definition, lvl)] = (table, lo, hi)
    return out


def side_flags(neg: np.ndarray, pos: np.ndarray, side: str) -> np.ndarray:
    return neg if side == SIDE_NEG else pos if side == SIDE_POS else (neg | pos)


def permute_within_groups(values: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permuta `values` al azar DENTRO de cada grupo (p. ej. minuto del reloj): conserva la distribucion de cada grupo y destruye el orden temporal."""
    v = np.asarray(values)
    order = np.lexsort((rng.random(v.size), np.asarray(groups)))
    sorted_groups_order = np.argsort(np.asarray(groups), kind="stable")
    out = np.empty_like(v)
    out[sorted_groups_order] = v[order]
    return out


# ---------------------------------------------------------------------------
# Efecto del triage sobre curtosis y cuantiles (E)
# ---------------------------------------------------------------------------

TRIAGE_STAT_NAMES: tuple[str, ...] = ("n", "std", "kurtosis_excess", "kurtosis_excess_trimmed", "q0.001", "q0.01", "q0.99", "q0.999")
_TRIAGE_Q = (0.001, 0.01, 0.99, 0.999)


def stats_from_sorted(xs: np.ndarray) -> np.ndarray:
    """`TRIAGE_STAT_NAMES` de una muestra YA ordenada, con las convenciones exactas de `compute_moments_quantiles` de TDA-07 (curtosis poblacional, recorte de 0,05 % por cola con cortes inclusivos, cuantil lineal)."""
    n = xs.size
    out = np.full(len(TRIAGE_STAT_NAMES), np.nan)
    out[0] = n
    if n < 2:
        return out

    def quant(p):
        h = (n - 1) * p
        k = int(np.floor(h))
        return xs[k] + (h - k) * (xs[min(k + 1, n - 1)] - xs[k])

    def kurt(v):
        e = v - v.mean()
        m2 = float(np.mean(e * e))
        return float(np.mean(e ** 4) / m2 ** 2 - 3.0) if m2 > 0 else np.nan

    out[1] = float(xs.std(ddof=1))
    out[2] = kurt(xs)
    if n >= 20:
        lo, hi = quant(TRIM_FRACTION_EACH_TAIL), quant(1.0 - TRIM_FRACTION_EACH_TAIL)
        trimmed = xs[np.searchsorted(xs, lo, side="left"):np.searchsorted(xs, hi, side="right")]
        if trimmed.size >= 2:
            out[3] = kurt(trimmed)
    for j, p in enumerate(_TRIAGE_Q):
        out[4 + j] = quant(p)
    return out


def triage_impact_table(scenarios: dict[str, tuple[np.ndarray, np.ndarray]], meta: dict[str, dict], n_days: int, n_boot: int, seed: int) -> pd.DataFrame:
    """Curtosis y cuantiles extremos de cada escenario (valores, indice de jornada), con IC por bootstrap de jornadas (las MISMAS jornadas sorteadas para todos los escenarios).

    Tabla larga: una fila por escenario y estadistico, con el valor, su
    IC y la diferencia respecto del escenario de referencia de su serie
    (`meta[escenario]["baseline"]`).
    """
    prepared = {}
    for name, (vals, day) in scenarios.items():
        order = np.argsort(vals, kind="stable")
        prepared[name] = (np.asarray(vals, dtype=float)[order], np.asarray(day)[order])
    point = {name: stats_from_sorted(xs) for name, (xs, _) in prepared.items()}
    reps = {name: np.full((n_boot, len(TRIAGE_STAT_NAMES)), np.nan) for name in prepared}
    rng = np.random.default_rng(seed)
    for b in range(n_boot):
        mult = np.bincount(rng.integers(0, n_days, size=n_days), minlength=n_days)
        for name, (xs, ds) in prepared.items():
            reps[name][b] = stats_from_sorted(np.repeat(xs, mult[ds]))
    rows = []
    for name in prepared:
        base = meta[name]["baseline"]
        for j, stat in enumerate(TRIAGE_STAT_NAMES):
            lo, hi = _ci(reps[name][:, j]) if stat != "n" else (np.nan, np.nan)
            diff = reps[name][:, j] - reps[base][:, j]
            dlo, dhi = _ci(diff) if (stat != "n" and name != base) else (np.nan, np.nan)
            rows.append({
                "series": meta[name]["series"], "scenario": name, "description": meta[name]["description"], "baseline": base,
                "n_removed_vs_baseline": int(point[base][0] - point[name][0]),
                "statistic": stat, "value": float(point[name][j]), "ci_lo": lo, "ci_hi": hi,
                "diff_vs_baseline": float(point[name][j] - point[base][j]), "diff_ci_lo": dlo, "diff_ci_hi": dhi,
                "rel_diff_vs_baseline": float(point[name][j] / point[base][j] - 1.0) if point[base][j] not in (0.0,) and np.isfinite(point[base][j]) else np.nan,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Veredictos (D6, D7)
# ---------------------------------------------------------------------------

def decide_th06(category_counts: pd.Series, sensitivity: pd.DataFrame | None = None) -> dict:
    """Lectura de TH06 (D7): `MAYORIA_CLASIFICABLE` si menos de la mitad de los extremos queda `INDETERMINADO`; en otro caso `FRACCION_ALTA_INDETERMINADA`.

    "Clasificable" es una afirmacion OPERATIVA: dice que las reglas de
    triage asignan una categoria, no que el origen de cada movimiento
    haya sido verificado de forma independiente (con barras de 1 minuto
    no puede serlo). Por eso se devuelve tambien `robust_to_rules`: si la
    lectura se mantiene en TODAS las variantes declaradas de las reglas
    (`triage_sensitivity`), y `depends_on_accepting_operational_evidence`
    (siempre cierto: si no se acepta la evidencia favorable de R5, todo
    retorno valido queda `INDETERMINADO`).
    """
    total = int(category_counts.sum())
    counts = {c: int(category_counts.get(c, 0)) for c in CATEGORIES}
    frac = {c: (counts[c] / total if total else float("nan")) for c in CATEGORIES}
    artefacts = frac[CAT_ERROR] + frac[CAT_ROLL] + frac[CAT_BOUNDARY] if total else float("nan")
    verdict = TH06_MOSTLY_CLASSIFIABLE if total and frac[CAT_UNDETERMINED] < TH06_MAJORITY else TH06_HIGH_UNDETERMINED
    robust, rng_lo, rng_hi = None, float("nan"), float("nan")
    if sensitivity is not None and len(sensitivity):
        variants = sensitivity.loc[sensitivity["kind"] == "VARIANTE_DE_REGLA", "fraction_undetermined"]
        if len(variants):
            rng_lo, rng_hi = float(variants.min()), float(variants.max())
            robust = bool((variants < TH06_MAJORITY).all()) if verdict == TH06_MOSTLY_CLASSIFIABLE else bool((variants >= TH06_MAJORITY).all())
    return {"verdict": verdict, "n_extremes": total, "counts": counts, "fractions": frac, "fraction_artefacts_and_errors": artefacts,
            "tails_reported_as_range": True, "robust_to_rules": robust, "undetermined_range_over_rule_variants": (rng_lo, rng_hi),
            "depends_on_accepting_operational_evidence": True, "basis": dict(TH06_BASIS)}


def triage_sensitivity(inv: pd.DataFrame) -> pd.DataFrame:
    """Cuanto cambia la fraccion `INDETERMINADO` (y `EVENTO_REAL`) al variar las reglas de triage. Descriptivo: la clasificacion vigente no cambia.

    - `VIGENTE`: las reglas R1-R5 tal como estan.
    - `VARIANTE_DE_REGLA`: retirar una a una cada duda de R4, y mover la
      fraccion de reversion alrededor del 0,5 de TDA-03.
    - `COTA`: los dos extremos logicos. Sin ninguna duda de R4 todo
      retorno valido seria `EVENTO_REAL`; si no se acepta la evidencia
      favorable de R5 como prueba, todo retorno valido es
      `INDETERMINADO`. La lectura de TH06 vive entre ambas cotas.
    """
    n = len(inv)
    structural = inv["category"].isin([CAT_ERROR, CAT_ROLL, CAT_BOUNDARY]).to_numpy()
    n_valid = int((~structural).sum())
    doubts = {
        "BARRA_NO_ACTIVA": ((inv["ff_category"] != FF_ACTIVE) | (inv["ff_category_prev"] != FF_ACTIVE)).to_numpy(),
        "JUNTO_A_HUECO_DESCONOCIDO": inv["adjacent_unknown_gap"].to_numpy(dtype=bool),
        "SIN_BARRA_SIGUIENTE": (~inv["next_bar_contiguous"].astype(bool)).to_numpy(),
        "REVIERTE_BARRA_SIGUIENTE": inv["reverts_next_bar"].to_numpy(dtype=bool),
        "VOLUMEN_BAJO": inv["low_volume"].to_numpy(dtype=bool),
    }
    rows = []

    def add(kind, name, description, undetermined_mask):
        und = int((undetermined_mask & ~structural).sum())
        rows.append({"kind": kind, "variant": name, "description": description, "n_extremes": n, "n_structural": int(structural.sum()),
                     "n_undetermined": und, "n_real": n_valid - und, "fraction_undetermined": und / n if n else np.nan,
                     "fraction_real": (n_valid - und) / n if n else np.nan})

    def any_of(d):
        out = np.zeros(n, dtype=bool)
        for flag in d.values():
            out |= flag
        return out

    add("VIGENTE", "R1-R5", "Reglas vigentes", any_of(doubts))
    for name in doubts:
        add("VARIANTE_DE_REGLA", f"sin_{name}", f"R4 sin la duda {name}", any_of({k: v for k, v in doubts.items() if k != name}))
    frac = inv["reversal_fraction_next_bar"].to_numpy(dtype=float)
    for f in REVERSAL_SENSITIVITY_FRACTIONS:
        alt = dict(doubts)
        alt["REVIERTE_BARRA_SIGUIENTE"] = np.nan_to_num(frac, nan=-np.inf) > f
        add("VARIANTE_DE_REGLA", f"reversion_mayor_que_{f:g}", f"Duda de reversion con fraccion {f:g} en vez de {REVERSAL_FRACTION_TDA03:g}", any_of(alt))
    add("COTA", "sin_dudas_R4", "Cota inferior: ninguna duda de R4 (todo retorno valido seria EVENTO_REAL)", np.zeros(n, dtype=bool))
    add("COTA", "sin_aceptar_evidencia_R5", "Cota superior: la evidencia favorable de R5 no se acepta como prueba (todo retorno valido es INDETERMINADO)", np.ones(n, dtype=bool))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Auditoria correctiva: estadisticos de la variante primaria, replicas nulas, bootstrap completo
# ---------------------------------------------------------------------------

REPLICATE_WINDOWS: tuple[int, ...] = (1, PRIMARY_WINDOW)
PRIMARY_SERIES: tuple[tuple[str, str], ...] = ((SERIES_R, SERIES_R), ("z", SERIES_Z))
KIND_FULL_BOOTSTRAP = "BOOTSTRAP_CON_REESTIMACION"
KIND_BERNOULLI_DIAG = "DIAGNOSTICO_BANDA_BERNOULLI"
# Orden FIJO: la posicion de cada tipo forma parte de la semilla de sus replicas. Los tipos nuevos se añaden AL FINAL para que las replicas
# de los tipos anteriores sean identicas, bit a bit, a las de ejecuciones previas.
_TASK_KINDS: tuple[str, ...] = (PREVIOUS_INDEPENDENCE_FAMILIES + VOLATILITY_FAMILIES + (KIND_FULL_BOOTSTRAP, KIND_BERNOULLI_DIAG) + (NULL_FAMILY_MINUTE_YEAR,))


def _key(series: str, side: str, name: str) -> str:
    return f"{series}|{side}|{name}"


def primary_statistics_from_flags(flags: dict, minute: np.ndarray, year: np.ndarray, block_ids: np.ndarray, day: np.ndarray, n_days: int,
                                  day_year: np.ndarray, bar_weights: np.ndarray | None = None, day_weights: np.ndarray | None = None) -> dict:
    """Estadisticos de la regla de TH27 para un juego de excedencias `{(serie, lado): booleano}`: cociente observado/esperado (ventanas de `REPLICATE_WINDOWS`) global y por año completo.

    Es el UNICO camino de calculo de la regla: lo usan los datos reales,
    cada replica nula y cada replica del bootstrap, de modo que lo que se
    calibra es exactamente lo que se aplica. La referencia es la primaria
    (`RELOJ_Y_ANIO`), re-estimada con los datos (y los pesos) que recibe.
    El "año" de una jornada es el de su `trading_date`.
    """
    w = np.ones((1, n_days)) if day_weights is None else np.asarray(day_weights, dtype=float).reshape(1, n_days)
    out: dict[str, float] = {}
    for (series, side), exc in flags.items():
        prob = reference_probabilities(exc, minute, year, PRIMARY_NULL, weights=bar_weights)
        obs = observed_counts(exc, block_ids, day, n_days, REPLICATE_WINDOWS)
        null = null_expectations(prob, block_ids, day, n_days, REPLICATE_WINDOWS)
        st = cluster_statistics(obs, null, w, windows=REPLICATE_WINDOWS)
        for k in REPLICATE_WINDOWS:
            out[_key(series, side, f"ratio_le_{k}")] = float(st[f"ratio_share_le_{k}"][0])
        out[_key(series, side, f"observed_le_{PRIMARY_WINDOW}")] = float(st[f"share_le_{PRIMARY_WINDOW}"][0])
        out[_key(series, side, f"expected_le_{PRIMARY_WINDOW}")] = float(st[f"share_le_{PRIMARY_WINDOW}_ref"][0])
        out[_key(series, side, "n_exceed")] = float(st["n_exceed"][0])
        year_ratios = []
        for y in COMPLETE_YEARS:
            wy = w[0] * (day_year == y)
            b, b0 = float(wy @ obs["b"]), float(wy @ null["b"])
            a, a0 = float(wy @ obs["a"][PRIMARY_WINDOW]), float(wy @ null["a"][PRIMARY_WINDOW])
            ratio = (a / b) / (a0 / b0) if b > 0 and b0 > 0 and a0 > 0 else float("nan")
            out[_key(series, side, f"ratio_le_{PRIMARY_WINDOW}_year_{y}")] = ratio
            if np.isfinite(ratio):
                year_ratios.append(ratio)
        out[_key(series, side, "n_complete_years")] = float(len(year_ratios))
        out[_key(series, side, "min_year_ratio")] = float(min(year_ratios)) if year_ratios else float("nan")
    for side in SIDES:
        kr, kz = _key(SERIES_R, side, f"ratio_le_{PRIMARY_WINDOW}"), _key(SERIES_Z, side, f"ratio_le_{PRIMARY_WINDOW}")
        if kr in out and kz in out:
            er, ez = out[kr] - 1.0, out[kz] - 1.0
            out[_key("r_vs_z", side, "ratio_difference")] = out[kr] - out[kz]
            out[_key("r_vs_z", side, "fraction_removed")] = float(1.0 - ez / er) if er > 0 else float("nan")
    return out


@dataclass
class ReplicateContext:
    """Todo lo que necesita un proceso trabajador: la poblacion valida, su estructura temporal y, para el bootstrap y el diagnostico, la variante primaria real ya calculada."""
    r_valid: np.ndarray
    trading_date: np.ndarray
    segment: np.ndarray
    minute: np.ndarray
    year: np.ndarray
    block_ids: np.ndarray
    day: np.ndarray
    n_days: int
    day_year: np.ndarray
    sigma_paths: dict = field(default_factory=dict)
    eligible: np.ndarray | None = None            # poblacion comun real
    values_e: dict = field(default_factory=dict)  # {"r_1m": r elegible, "z": z elegible}
    cell_sorted: dict = field(default_factory=dict)   # serie -> lista de indices (dentro de la poblacion elegible) de cada celda, ordenados por valor
    real_flags: dict = field(default_factory=dict)    # (serie, lado) -> excedencias reales primarias
    seed: int = CALIBRATION_SEED


def primary_replicate_statistics(r_series: np.ndarray, ctx: ReplicateContext) -> dict:
    """Procedimiento COMPLETO sobre una serie de retornos (real o nula): `sigma_hat`, `z`, decil causal, poblacion comun, umbrales primarios, referencia y estadisticos de la regla."""
    ex = build_exceedances(r_series, ctx.trading_date, ctx.segment, levels=(PRIMARY_LEVEL,), definitions=(PRIMARY_DEFINITION,),
                           decile_fn=causal_volatility_decile_fast)
    e = ex.eligible
    flags = {}
    for name, label in PRIMARY_SERIES:
        neg, pos = ex.flags[(name, PRIMARY_DEFINITION, PRIMARY_LEVEL)]
        for side in SIDES:
            flags[(label, side)] = side_flags(neg, pos, side)
    out = primary_statistics_from_flags(flags, ctx.minute[e], ctx.year[e], restrict_block_ids(ctx.block_ids, e), ctx.day[e], ctx.n_days, ctx.day_year)
    out["n_eligible"] = float(e.sum())
    return out


def volatility_paths(r_valid: np.ndarray) -> dict:
    """Sendas de volatilidad para los controles de "solo volatilidad" (todas tomadas de los datos; ninguna es "la volatilidad verdadera", que no es observable, G0).

    - `VOLATILIDAD_PURA_EWMA_60`: la misma `sigma_hat` causal del estimador primario (la senda que el estimador puede seguir sin retraso propio).
    - `VOLATILIDAD_PURA_EWMA_20`: la EWMA mas rapida de TDA-10 (el estimador primario la sigue con retraso).
    - `VOLATILIDAD_PURA_LOCAL_10`: raiz de la media de `r^2` de las `PRIMARY_WINDOW` barras anteriores y posteriores, SIN la propia barra. Es
      RETROSPECTIVA (usa el futuro) y deliberadamente rapida: contiene las rafagas de pocos minutos, a la misma escala a la que se mide el agrupamiento.
    Donde una senda no es finita o es practicamente nula se usa la EWMA primaria.
    """
    r = np.asarray(r_valid, dtype=float)
    fam = PRIMARY_ESTIMATOR[0]
    primary = build_sigma_hat(r, fam, PRIMARY_ESTIMATOR[1])
    fast = build_sigma_hat(r, fam, min(EWMA_HALFLIVES_MINUTES))
    n, w = r.size, PRIMARY_WINDOW
    c = np.concatenate([[0.0], np.cumsum(r * r)])
    i = np.arange(n)
    lo, hi = np.maximum(i - w, 0), np.minimum(i + w, n - 1)
    cnt = (hi - lo).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        local = np.sqrt(np.where(cnt > 0, (c[hi + 1] - c[lo] - r * r) / np.where(cnt > 0, cnt, 1.0), np.nan))

    def clean(path):
        bad = ~np.isfinite(path) | (path <= MIN_VALID_SIGMA_HAT)
        return np.where(bad, primary, path)

    return {VOL_FAMILY_EWMA_PRIMARY: primary, VOL_FAMILY_EWMA_FAST: clean(fast), VOL_FAMILY_LOCAL: clean(local)}


def minute_year_groups(minute: np.ndarray, year: np.ndarray) -> np.ndarray:
    """Codigo de grupo (minuto del dia en Nueva York, año) de cada barra: los grupos dentro de los que permuta `PERMUTACION_MINUTO_X_ANIO`.

    `minute` y `year` son los MISMOS vectores con los que se ajusta la
    referencia `RELOJ_Y_ANIO` (`minute_of_day` y `year_ny` de TDA-06,
    ambos en hora local de Nueva York con horario de verano): la familia
    conserva exactamente la estructura que la referencia condiciona.
    """
    m = np.asarray(minute).astype(np.int64)
    if m.size and (m.min() < 0 or m.max() >= MINUTES_PER_DAY):
        raise ValueError("`minute` debe ser el minuto del dia (0..1439).")
    _, y = np.unique(np.asarray(year), return_inverse=True)
    return y.astype(np.int64) * MINUTES_PER_DAY + m


def null_family_group_codes(kind: str, minute: np.ndarray, year: np.ndarray) -> np.ndarray:
    """Grupos de intercambiabilidad de cada familia sin dependencia: un solo grupo, uno por minuto, o uno por (minuto, año)."""
    if kind == NULL_FAMILY_PERMUTATION:
        return np.zeros(np.asarray(minute).size, dtype=np.int64)
    if kind == NULL_FAMILY_CLOCK:
        return np.asarray(minute).astype(np.int64)
    if kind == NULL_FAMILY_MINUTE_YEAR:
        return minute_year_groups(minute, year)
    raise ValueError(f"no es una familia sin dependencia: {kind}")


def describe_null_family_groups(minute: np.ndarray, year: np.ndarray) -> pd.DataFrame:
    """Tamaño de los grupos de permutacion de cada familia sin dependencia (los grupos pequeños se reportan; nunca se fusionan ni se descartan)."""
    rows = []
    for fam in INDEPENDENCE_FAMILIES:
        sizes = np.bincount(np.unique(null_family_group_codes(fam, minute, year), return_inverse=True)[1])
        small = sizes < SMALL_GROUP_SIZE
        rows.append({"family": fam, "n_groups": int(sizes.size), "min_size": int(sizes.min()), "median_size": float(np.median(sizes)), "max_size": int(sizes.max()),
                     "n_groups_of_one_bar": int((sizes == 1).sum()), "n_small_groups": int(small.sum()), "small_group_size": SMALL_GROUP_SIZE,
                     "bars_in_small_groups": int(sizes[small].sum()), "share_bars_in_small_groups": float(sizes[small].sum() / sizes.sum())})
    return pd.DataFrame(rows)


def generate_null_series(kind: str, ctx: ReplicateContext, rng: np.random.Generator) -> np.ndarray:
    """Serie de control para G2.

    Sin dependencia temporal (calibran la banda). Las tres son
    PERMUTACIONES de los retornos reales dentro de grupos: conservan, sin
    error, el conjunto de valores de cada grupo (y por tanto su
    distribucion completa) y destruyen el orden temporal dentro de el.
    - `PERMUTACION_GLOBAL`: un solo grupo (todos los retornos barajados).
    - `SIMULADO_PERFIL_RELOJ`: un grupo por minuto del reloj (misma
      volatilidad y curtosis por hora, agregada entre años).
    - `PERMUTACION_MINUTO_X_ANIO`: un grupo por (minuto del reloj, año).
      Conserva ademas el nivel de cada año y que el perfil horario pueda
      tener una forma distinta cada año. ROMPE: el orden de las jornadas
      dentro de cada año (rachas de dias volatiles, dia de la semana,
      deriva dentro del año, dias de evento). Un grupo de una sola barra
      queda fijo. Hipotesis nula que representa: "dentro de un mismo
      minuto del reloj y un mismo año, los retornos son intercambiables".
    Las hipotesis estan ANIDADAS: si los retornos son intercambiables
    globalmente, lo son dentro de cada minuto; y si lo son dentro de cada
    minuto, lo son dentro de cada (minuto, año). La tercera familia es la
    hipotesis mas amplia de las tres.
    Solo volatilidad (no calibran; separan la afirmacion 2 de la 3):
    - `VOLATILIDAD_PURA_*`: ``r* = sigma_t * eps_t`` con `sigma_t` una
      senda de `volatility_paths` y `eps` los residuos `r/sigma`
      barajados dentro de cada minuto del reloj. Los retornos son
      independientes DADA la senda: todo agrupamiento que aparezca es,
      por construccion, volatilidad.
    """
    r = ctx.r_valid
    if kind == NULL_FAMILY_PERMUTATION:
        return rng.permutation(r)
    if kind == NULL_FAMILY_CLOCK:
        return permute_within_groups(r, ctx.minute, rng)
    if kind == NULL_FAMILY_MINUTE_YEAR:
        return permute_within_groups(r, minute_year_groups(ctx.minute, ctx.year), rng)
    if kind in VOLATILITY_FAMILIES:
        sig = ctx.sigma_paths[kind]
        ok = np.isfinite(sig) & (sig > MIN_VALID_SIGMA_HAT)
        out = r.copy()
        out[ok] = sig[ok] * permute_within_groups(r[ok] / sig[ok], ctx.minute[ok], rng)
        return out
    raise ValueError(f"familia de control desconocida: {kind}")


def weighted_cell_exceedances(x: np.ndarray, cell_sorted: list, bar_weights: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Excedencias de `x` con umbrales RE-ESTIMADOS por celda sobre la muestra ponderada (cada barra repetida `bar_weights` veces): mismo cuantil lineal que `compute_thresholds`, sin materializar la muestra."""
    neg, pos = np.zeros(x.size, dtype=bool), np.zeros(x.size, dtype=bool)
    for idx in cell_sorted:
        cw = np.cumsum(bar_weights[idx])
        n = float(cw[-1]) if cw.size else 0.0
        if n < 1:
            continue
        xs = x[idx]

        def quant(prob):
            h = (n - 1) * prob
            k = np.floor(h)
            x0 = xs[min(np.searchsorted(cw, k, side="right"), xs.size - 1)]
            x1 = xs[min(np.searchsorted(cw, min(k + 1, n - 1), side="right"), xs.size - 1)]
            return x0 + (h - k) * (x1 - x0)

        lo, hi = quant(level), quant(1.0 - level)
        neg[idx], pos[idx] = xs < lo, xs > hi
    return neg, pos


def attach_primary_sample(ctx: ReplicateContext, ex: ExceedanceSet) -> ReplicateContext:
    """Rellena en `ctx` lo que necesitan el bootstrap con re-estimacion y el diagnostico: poblacion comun, valores de `r` y `z`, celdas primarias ordenadas por valor y excedencias primarias. Devuelve el propio `ctx`."""
    e = ex.eligible
    values_e = {SERIES_R: np.asarray(ctx.r_valid, dtype=float)[e], "z": ex.z[e]}
    prim_cells = cell_codes(PRIMARY_DEFINITION, np.asarray(ctx.segment)[e], ex.decile[e])
    codes, _ = pd.factorize(pd.Series(prim_cells), sort=True)
    cell_sorted = {}
    for name in values_e:
        order = np.lexsort((values_e[name], codes))
        bounds = np.searchsorted(codes[order], np.arange(codes.max() + 2))
        cell_sorted[name] = [order[bounds[c]:bounds[c + 1]] for c in range(codes.max() + 1)]
    real_flags = {}
    for name, label in PRIMARY_SERIES:
        neg, pos = ex.flags[(name, PRIMARY_DEFINITION, PRIMARY_LEVEL)]
        for side in SIDES:
            real_flags[(label, side)] = side_flags(neg, pos, side)
    ctx.eligible, ctx.values_e, ctx.cell_sorted, ctx.real_flags = e, values_e, cell_sorted, real_flags
    return ctx


def evaluate_full_rule(r_series: np.ndarray, ctx: ReplicateContext, n_calibration: int, n_boot_full: int, n_workers: int = 1,
                       family_sets: dict[str, tuple[str, ...]] | None = None) -> dict:
    """Aplica a UNA serie la regla completa de TH27, tal como se aplica a los datos reales: procedimiento completo, banda calibrada con las familias sin dependencia generadas A PARTIR DE ESA SERIE, control G2, bootstrap con re-estimacion y `decide_th27`.

    Sirve para medir, sobre series sinteticas cuya verdad se conoce, con
    que frecuencia la regla entera afirma algo falso. `family_sets`
    ({etiqueta: familias}) decide con varios conjuntos de familias sobre
    las MISMAS replicas (p. ej. el procedimiento anterior y el vigente).
    No se generan controles de solo volatilidad: no intervienen en las
    afirmaciones 1 y 2.
    """
    from dataclasses import replace
    r = np.asarray(r_series, dtype=float)
    ex = build_exceedances(r, ctx.trading_date, ctx.segment, levels=(PRIMARY_LEVEL,), definitions=(PRIMARY_DEFINITION,), decile_fn=causal_volatility_decile_fast)
    c = attach_primary_sample(replace(ctx, r_valid=r, sigma_paths={}), ex)
    point = primary_replicate_statistics(r, c)
    tasks = [(fam, rep) for fam in INDEPENDENCE_FAMILIES for rep in range(int(n_calibration))] + [(KIND_FULL_BOOTSTRAP, rep) for rep in range(int(n_boot_full))]
    reps = run_tasks(c, tasks, n_workers)
    boot = summarize_full_bootstrap(point, reps)
    decisions, bands_out = {}, {}
    for label, fams in (family_sets or {"VIGENTE": INDEPENDENCE_FAMILIES}).items():
        rates, bands, g2_ok = summarize_calibration(reps, independence_families=fams)
        decisions[label] = decide_th27(point, boot, bands, g2_ok, rates, reps)
        bands_out[label] = bands
    return {"point": point, "bootstrap": boot, "decisions": decisions, "bands": bands_out, "replicates": reps}


def _full_bootstrap_replicate(ctx: ReplicateContext, rng: np.random.Generator) -> dict:
    """Una replica del bootstrap por jornadas que RE-ESTIMA umbrales (por celda) y referencia (por minuto y año) sobre la muestra remuestreada."""
    e = ctx.eligible
    day_e = ctx.day[e]
    mult = np.bincount(rng.integers(0, ctx.n_days, size=ctx.n_days), minlength=ctx.n_days).astype(float)
    wbar = mult[day_e]
    flags = {}
    for name, label in PRIMARY_SERIES:
        neg, pos = weighted_cell_exceedances(ctx.values_e[name], ctx.cell_sorted[name], wbar, PRIMARY_LEVEL)
        for side in SIDES:
            flags[(label, side)] = side_flags(neg, pos, side)
    return primary_statistics_from_flags(flags, ctx.minute[e], ctx.year[e], restrict_block_ids(ctx.block_ids, e), day_e, ctx.n_days, ctx.day_year, bar_weights=wbar, day_weights=mult)


def _bernoulli_diagnostic_replicate(ctx: ReplicateContext, rng: np.random.Generator) -> dict:
    """Diagnostico del hallazgo B: excedencias independientes con la probabilidad real como VERDAD. Cociente con esa probabilidad conocida (la banda antigua) y con la referencia re-estimada de la propia simulacion."""
    e = ctx.eligible
    minute, year, blk, day = ctx.minute[e], ctx.year[e], restrict_block_ids(ctx.block_ids, e), ctx.day[e]
    ones = np.ones((1, ctx.n_days))
    out = {}
    for (label, side), exc in ctx.real_flags.items():
        if side == SIDE_BOTH:
            continue
        p_true = reference_probabilities(exc, minute, year, PRIMARY_NULL)
        sim = rng.random(p_true.size) < p_true
        obs = observed_counts(sim, blk, day, ctx.n_days, REPLICATE_WINDOWS)
        for tag, prob in ((DIAG_BERNOULLI_KNOWN, p_true), (DIAG_BERNOULLI_REFIT, reference_probabilities(sim, minute, year, PRIMARY_NULL))):
            st = cluster_statistics(obs, null_expectations(prob, blk, day, ctx.n_days, REPLICATE_WINDOWS), ones, windows=REPLICATE_WINDOWS)
            out[_key(label, side, f"{tag}|ratio_le_{PRIMARY_WINDOW}")] = float(st[f"ratio_share_le_{PRIMARY_WINDOW}"][0])
    return out


_WORKER_CTX: ReplicateContext | None = None


def _worker_init(path: str) -> None:
    global _WORKER_CTX
    with open(path, "rb") as fh:
        _WORKER_CTX = pickle.load(fh)


def _run_task(task: tuple) -> tuple:
    kind, rep = task
    ctx = _WORKER_CTX
    rng = np.random.default_rng([ctx.seed, _TASK_KINDS.index(kind), rep])     # semilla propia de cada replica: el resultado no depende del reparto entre procesos
    if kind == KIND_FULL_BOOTSTRAP:
        return kind, rep, _full_bootstrap_replicate(ctx, rng)
    if kind == KIND_BERNOULLI_DIAG:
        return kind, rep, _bernoulli_diagnostic_replicate(ctx, rng)
    return kind, rep, primary_replicate_statistics(generate_null_series(kind, ctx, rng), ctx)


MAX_WORKERS = 8                 # tope de procesos de esta etapa (cada uno mantiene su copia de la poblacion: ~1 GB)
WORKER_MEMORY_GB = 1.0          # memoria reservada por proceso al dimensionar (el pico observado es ~0,75 GB)
MEMORY_RESERVE_GB = 4.0         # memoria que se deja siempre libre para el resto del sistema
MEMORY_USABLE_FRACTION = 0.6    # de lo que queda tras la reserva, solo se usa esta fraccion


def available_memory_gb() -> float | None:
    """Memoria fisica libre en GB (sin dependencias externas). `None` si no se puede determinar."""
    try:
        if os.name == "nt":
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = _MemStatus()
            st.dwLength = ctypes.sizeof(_MemStatus)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return None
            return st.ullAvailPhys / 2 ** 30
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2 ** 30
    except (AttributeError, OSError, ValueError):
        return None


def memory_safe_workers(requested: int, available_gb: float | None) -> int:
    """Numero de procesos que caben con margen: ``floor((libre - reserva) * fraccion / memoria_por_proceso)``, nunca mas de `requested` ni menos de 1. Si la memoria libre no se puede medir, 1 (lo seguro)."""
    if available_gb is None:
        return 1
    fit = int(np.floor(max(available_gb - MEMORY_RESERVE_GB, 0.0) * MEMORY_USABLE_FRACTION / WORKER_MEMORY_GB))
    return max(1, min(int(requested), fit))


def default_n_workers(n_jobs: int) -> int:
    """``min(MAX_WORKERS, cpu_count()-4, n_jobs)`` -- sin sobre-suscripcion (docs/project_hardware.md) y con el tope de memoria de esta etapa."""
    return max(1, min(MAX_WORKERS, (os.cpu_count() or 4) - 4, max(1, n_jobs)))


def run_tasks(ctx: ReplicateContext, tasks: list[tuple], n_workers: int) -> pd.DataFrame:
    """Ejecuta las replicas (en serie si `n_workers <= 1`, en procesos si no) y devuelve una tabla larga `kind, replicate, key, value`. El resultado es identico con cualquier numero de procesos."""
    global _WORKER_CTX
    results = []
    if n_workers <= 1 or len(tasks) <= 1:
        _WORKER_CTX = ctx
        try:
            results = [_run_task(tk) for tk in tasks]
        finally:
            _WORKER_CTX = None
    else:
        fd, path = tempfile.mkstemp(prefix="tda13_ctx_", suffix=".pkl")
        try:
            with os.fdopen(fd, "wb") as fh:
                pickle.dump(ctx, fh, protocol=pickle.HIGHEST_PROTOCOL)
            with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init, initargs=(path,)) as pool:
                results = list(pool.map(_run_task, tasks, chunksize=1))
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
    rows = [{"kind": kind, "replicate": rep, "key": key, "value": val} for kind, rep, stats in results for key, val in stats.items()]
    out = pd.DataFrame(rows, columns=["kind", "replicate", "key", "value"])
    return out.sort_values(["kind", "replicate", "key"], kind="stable").reset_index(drop=True)


def wilson_interval(k: int, n: int, level: float = G2_CI_LEVEL) -> tuple[float, float]:
    """Intervalo de Wilson para una proporcion `k/n` (incertidumbre binomial de una tasa estimada con `n` replicas)."""
    if n <= 0:
        return float("nan"), float("nan")
    zq = NormalDist().inv_cdf(0.5 + level / 2.0)
    ph = k / n
    den = 1.0 + zq * zq / n
    centre = (ph + zq * zq / (2 * n)) / den
    half = zq * np.sqrt(ph * (1 - ph) / n + zq * zq / (4 * n * n)) / den
    return float(min(ph, max(0.0, centre - half))), float(max(ph, min(1.0, centre + half)))


def _wide(replicates: pd.DataFrame, kind: str) -> pd.DataFrame:
    sub = replicates.loc[replicates["kind"] == kind]
    return sub.pivot(index="replicate", columns="key", values="value").sort_index() if len(sub) else pd.DataFrame()


def summarize_calibration(replicates: pd.DataFrame, legacy_band_hi: dict | None = None,
                          independence_families: tuple[str, ...] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, bool | None]:
    """Banda de decision calibrada y tasas de falsa deteccion (hallazgo B).

    REGLA DE INTEGRACION DE LAS FAMILIAS (segunda auditoria, H3). Una
    cola se detecta solo si el cociente supera la banda de TODAS las
    familias sin dependencia, es decir, la mayor de sus bandas.
    Justificacion:

    a. La hipotesis nula "no hay dependencia temporal" es compuesta. Un
       contraste que solo rechaza cuando rechazan todos los contrastes de
       sus componentes (interseccion-union) tiene, frente a cada
       componente, como mucho el tamaño del contraste de ESE componente,
       sin correccion por multiplicidad.
    b. Cada componente es un contraste de permutacion: si los retornos
       son intercambiables dentro de los grupos de la familia, la serie
       observada es una permutacion mas, y su estadistico -- cualquier
       funcion de la serie, con toda la re-estimacion incluida -- es
       intercambiable con los de las replicas. Entonces, con `B`
       replicas para la banda, ``P(estadistico > X_(j)) = (B-j+1)/(B+1)``
       exactamente, y como el percentil interpolado es >= `X_(j)`
       (`band_order_statistic`), la probabilidad de superar la banda es
       <= ``(B-j+1)/(B+1)`` (con B = 500: 14/501 = 2,79 %; el 2,5 % es
       nominal). `permutation_band_size_bound`.
    c. Las tres hipotesis estan anidadas (ver `generate_null_series`):
       la banda de `PERMUTACION_MINUTO_X_ANIO` es valida por si sola en
       los tres mundos. Tomar ademas el maximo con las otras dos solo
       puede subir la banda: mas conservador, menos potencia, nunca
       menos valido.
    LO QUE NO CUBRE: mundos sin dependencia que no sean intercambiables
    dentro de (minuto, año) -- p. ej. un perfil horario que cambie de un
    mes a otro, o por dia de la semana. Ahi el maximo de las tres bandas
    NO garantiza nada. Tampoco convierte a `z` en mas potente: en mundos
    heterogeneos el cociente de `z` de la tercera familia queda por
    debajo de 1 y la banda la fijan las otras dos.

    `independence_families` permite recalcular con otro conjunto de
    familias (p. ej. `PREVIOUS_INDEPENDENCE_FAMILIES`, para comparar con
    el procedimiento anterior); por defecto, `INDEPENDENCE_FAMILIES`.

    CRITERIO DE ACEPTACION DE G2 (fijado antes de volver a evaluar los
    datos reales):

    1. Las replicas de cada familia SIN dependencia se parten en dos
       mitades por paridad del indice: PARES fijan la banda, IMPARES
       miden las falsas detecciones. Ninguna replica hace las dos cosas.
    2. Banda superior de cada (serie, lado) = el MAYOR, entre las
       familias sin dependencia, del percentil 97,5 del cociente en sus
       replicas pares (la familia mas exigente manda).
    3. Sobre las impares se mide, con intervalo de Wilson al 95 %:
       la tasa por cola (cociente > banda) de `r` y de `z`, y la tasa de
       las dos reglas globales -- afirmacion 1 (`r` detectado en caidas Y
       en subidas) y afirmacion 2 (ademas `z` detectado en ambas colas y
       por encima de 1 en todos los años completos).
    4. G2 NO SE RECHAZA solo si, en TODAS las familias sin dependencia,
       el numero de detecciones por cola (`r` y `z`, caidas y subidas:
       `m` tasas) no supera `acceptance_max_detections`. Ese maximo es
       una COTA CONSERVADORA EXACTA del numero de replicas reservadas
       que superan la banda cuando ambas mitades proceden del mismo
       mundo (`max_detections_under_exchangeability`: ley beta-binomial
       del estadistico de orden inferior de los dos entre los que se
       interpola el percentil 97,5); tiene en cuenta que la propia banda
       esta estimada con un numero finito de replicas. Se toma su
       cuantil ``1 - 0.05/m`` (Bonferroni), de modo que un procedimiento
       bien calibrado pasa el criterio al menos el 95 % de las veces.
       Si alguna cola lo incumple, G2 queda RECHAZADA y las afirmaciones
       1 y 2 quedan `INCONCLUSO`. Sin replicas reservadas: no evaluable.
       ALCANCE: pasar el criterio es NO RECHAZAR la calibracion en las
       familias evaluadas, con la potencia que da el numero de replicas
       (`criterion_non_rejection_probability`); no demuestra que cada
       tasa sea el 2,5 % nominal ni cubre mundos nulos distintos de esas
       familias. El rotulo es `g2_status`.
       Las reglas globales no pueden detectar mas veces que la cola que
       menos detecta, asi que quedan cubiertas por el criterio por cola;
       se reportan con su intervalo de Wilson al 95 %, igual que cada
       tasa por cola.

    La regla aplicada a los datos reales exige ADEMAS que el IC del
    bootstrap con re-estimacion quede por encima de 1; esa condicion solo
    puede quitar detecciones, asi que las tasas medidas aqui son una cota
    superior de las de la regla completa.

    Para las familias de "solo volatilidad" se reportan las mismas tasas
    con esa banda (no son falsas detecciones de la afirmacion 1: ahi SI
    hay agrupamiento, es su control positivo) y la distribucion del
    cociente de `z`, que se compara con el real en `decide_th27`.
    `legacy_band_hi` (banda Bernoulli previa) permite cuantificar el
    defecto original sobre las mismas replicas.

    Salida: `(tasas, bandas, g2_ok)`.
    """
    stat = f"ratio_le_{PRIMARY_WINDOW}"
    fams_ind = tuple(INDEPENDENCE_FAMILIES if independence_families is None else independence_families)
    fams_all = fams_ind + VOLATILITY_FAMILIES
    wide = {fam: _wide(replicates, fam) for fam in fams_all}
    band_rows, band_hi = [], {}
    for series in (SERIES_R, SERIES_Z):
        for side in SIDES:
            key = _key(series, side, stat)
            his, los, per_family = [], [], {}
            for fam in fams_ind:
                w = wide[fam]
                if key not in w:
                    continue
                even = w.loc[w.index % 2 == 0, key].dropna().to_numpy()
                if even.size:
                    per_family[fam] = (float(np.percentile(even, 100 - NULL_PERCENTILE)), float(np.percentile(even, NULL_PERCENTILE)), float(even.mean()), float(even.std(ddof=1)) if even.size > 1 else np.nan, int(even.size))
                    los.append(per_family[fam][0])
                    his.append(per_family[fam][1])
            hi, lo = (max(his), min(los)) if his else (np.nan, np.nan)
            band_hi[(series, side)] = hi
            binding = max(per_family, key=lambda f_: per_family[f_][1]) if per_family else ""
            row = {"series": series, "side": side, "statistic": f"share_le_{PRIMARY_WINDOW}", "band_lo": lo, "band_hi": hi, "binding_family": binding,
                   "n_families": len(per_family), "legacy_bernoulli_band_hi": (legacy_band_hi or {}).get((series, side), np.nan)}
            for fam, (flo, fhi, mean, sd, n) in per_family.items():
                row.update({f"{fam}|p2_5": flo, f"{fam}|p97_5": fhi, f"{fam}|mean": mean, f"{fam}|std": sd, f"{fam}|n_band": n})
            band_rows.append(row)
    bands = pd.DataFrame(band_rows)

    def detect(w: pd.DataFrame, series: str, side: str, band: dict) -> pd.Series:
        return w[_key(series, side, stat)] > band[(series, side)]

    rate_rows = []
    for fam in fams_all:
        w = wide[fam]
        if w.empty:
            continue
        independent = fam in fams_ind
        ev = w.loc[w.index % 2 == 1] if independent else w        # las familias de volatilidad no fijan la banda: se usan todas sus replicas
        if ev.empty:
            continue
        tails = {(s, sd): detect(ev, s, sd, band_hi) for s in (SERIES_R, SERIES_Z) for sd in SIDES}
        z_years = (ev[_key(SERIES_Z, SIDE_NEG, "min_year_ratio")] > 1.0) & (ev[_key(SERIES_Z, SIDE_POS, "min_year_ratio")] > 1.0)
        claim1 = tails[(SERIES_R, SIDE_NEG)] & tails[(SERIES_R, SIDE_POS)]
        claim2 = claim1 & tails[(SERIES_Z, SIDE_NEG)] & tails[(SERIES_Z, SIDE_POS)] & z_years
        z_only = tails[(SERIES_Z, SIDE_NEG)] & tails[(SERIES_Z, SIDE_POS)] & z_years
        rules = [(f"cola: {s} {sd}", flag, s, sd) for (s, sd), flag in tails.items()]
        rules += [("regla global: afirmacion 1 (r en ambas colas)", claim1, SERIES_R, "NEG+POS"),
                  ("regla global: afirmacion 2 (r y z en ambas colas, z > 1 en todos los años completos)", claim2, SERIES_Z, "NEG+POS"),
                  ("z en ambas colas y z > 1 en todos los años completos (sin exigir r)", z_only, SERIES_Z, "NEG+POS")]
        for name, flag, series, side in rules:
            n, k = int(flag.size), int(flag.sum())
            lo, hi = wilson_interval(k, n)
            in_acceptance = independent and side in (SIDE_NEG, SIDE_POS) and name.startswith("cola")
            row = {"family": fam, "family_type": "SIN_DEPENDENCIA" if independent else "SOLO_VOLATILIDAD", "rule": name, "series": series, "side": side,
                   "n_replicates": n, "n_detections": k, "rate": k / n, "rate_ci_lo": lo, "rate_ci_hi": hi, "nominal_rate": G2_NOMINAL_RATE,
                   "in_acceptance_criterion": in_acceptance}
            if legacy_band_hi and side in SIDES and name.startswith("cola"):
                allrep = w[_key(series, side, stat)].dropna()
                kk = int((allrep > legacy_band_hi.get((series, side), np.inf)).sum())
                llo, lhi = wilson_interval(kk, int(allrep.size))
                row.update({"legacy_band_rate": kk / allrep.size if allrep.size else np.nan, "legacy_band_rate_ci_lo": llo, "legacy_band_rate_ci_hi": lhi,
                            "legacy_band_n": int(allrep.size)})
            col = w[_key(series, side, stat)] if side in SIDES else None
            if col is not None:
                row.update({"ratio_mean": float(col.mean()), "ratio_std": float(col.std(ddof=1)) if col.size > 1 else np.nan,
                            "ratio_p2_5": float(np.percentile(col.dropna(), 2.5)), "ratio_p97_5": float(np.percentile(col.dropna(), 97.5))})
            rate_rows.append(row)
    rates = pd.DataFrame(rate_rows)
    if rates.empty or not rates["in_acceptance_criterion"].any():
        return rates, bands, None
    acc = rates["in_acceptance_criterion"].to_numpy(dtype=bool)
    m = int(acc.sum())
    level = 1.0 - (1.0 - G2_CI_LEVEL) / m
    n_band = {(r_["series"], r_["side"]): max([int(r_[f"{fam}|n_band"]) for fam in fams_ind if f"{fam}|n_band" in r_ and pd.notna(r_[f"{fam}|n_band"])] or [0])
              for _, r_ in bands.iterrows()}
    kmax = [max_detections_under_exchangeability(n_band.get((s, sd), 0), int(n), level) if a_ else np.nan
            for s, sd, n, a_ in zip(rates["series"], rates["side"], rates["n_replicates"], acc)]
    rates["n_rates_in_criterion"] = m
    rates["acceptance_level"] = level
    rates["acceptance_max_detections"] = kmax
    rates["passes"] = [bool(k <= km) if a_ else None for k, km, a_ in zip(rates["n_detections"], kmax, acc)]
    return rates, bands, bool(all(v for v in rates.loc[acc, "passes"]))


def band_order_statistic(n_band: int, percentile: float = NULL_PERCENTILE) -> int:
    """Posicion (1..n_band) del estadistico de orden INFERIOR de los dos entre los que `numpy.percentile` interpola el percentil `percentile` de `n_band` valores.

    Con 500 valores y el percentil 97,5: ``0.975 * 499 = 486.525`` -> el
    percentil queda entre las posiciones 487 y 488; se devuelve 487.
    """
    return int(np.floor(percentile / 100.0 * (n_band - 1))) + 1


def max_detections_under_exchangeability(n_band: int, n_eval: int, level: float, percentile: float = NULL_PERCENTILE) -> int:
    """Maximo de replicas reservadas (de `n_eval`) por encima de la banda que se admite, al nivel `level`, cuando las dos mitades proceden de la misma distribucion.

    COTA CONSERVADORA EXACTA (segunda auditoria, H1). La banda es un
    percentil INTERPOLADO entre dos estadisticos de orden vecinos,
    `X_(j)` y `X_(j+1)`, de las `n_band` replicas que la fijan
    (`j = band_order_statistic`). La probabilidad de superar un valor
    interpolado NO sigue una ley Beta exacta; la de superar un
    estadistico de orden si: para cualquier distribucion continua,
    ``P(superar X_(j)) ~ Beta(n_band - j + 1, j)``, y el numero de
    superaciones entre `n_eval` replicas nuevas e independientes es una
    beta-binomial. Como el percentil interpolado es >= `X_(j)`, el numero
    de replicas que superan la banda es <= el numero de las que superan
    `X_(j)`. Por tanto el cuantil `level` de esa beta-binomial es una
    cota: un procedimiento cuyas dos mitades proceden del mismo mundo
    supera este maximo con probabilidad <= ``1 - level``. (Tomar ademas
    la banda como el maximo de varias familias solo puede reducir las
    superaciones.)

    "Conservadora" se refiere a ESE error (rechazar un procedimiento
    bien calibrado). El precio es potencia: el criterio tolera algo mas
    de lo que toleraria la ley exacta del percentil interpolado, que no
    es libre de distribucion. Ver `criterion_non_rejection_probability`.

    La version anterior usaba una Beta de orden fraccionario
    (``r = percentile/100*(n_band-1)+1``) y la llamaba "exacta": era una
    aproximacion. Para 500 + 500 replicas y 8 comparaciones ambas dan 29.
    """
    from math import exp, lgamma
    if n_band < 2 or n_eval < 1:
        return 0
    j = band_order_statistic(n_band, percentile)
    a, b = float(n_band - j + 1), float(j)

    def lbeta(x, y):
        return lgamma(x) + lgamma(y) - lgamma(x + y)

    cum = 0.0
    for k in range(n_eval + 1):
        cum += exp(lgamma(n_eval + 1) - lgamma(k + 1) - lgamma(n_eval - k + 1) + lbeta(k + a, n_eval - k + b) - lbeta(a, b))
        if cum >= level:
            return k
    return n_eval


def permutation_band_size_bound(n_band: int, percentile: float = NULL_PERCENTILE) -> float:
    """Cota EXACTA de la probabilidad de que un estadistico intercambiable con las `n_band` replicas que fijan la banda la supere: ``(n_band - j + 1) / (n_band + 1)``, con `j = band_order_statistic`.

    Es el tamaño real, por cola, del contraste de permutacion cuando la
    banda es el percentil interpolado (>= `X_(j)`): 14/501 = 2,79 % con
    500 replicas, no el 2,5 % nominal. Valida para cualquier estadistico
    y cualquier distribucion sin empates.
    """
    if n_band < 2:
        return float("nan")
    return float((n_band - band_order_statistic(n_band, percentile) + 1) / (n_band + 1))


def criterion_non_rejection_probability(max_detections: int, n_eval: int, true_rate: float) -> float:
    """Probabilidad de que UNA tasa por cola NO supere `max_detections` entre `n_eval` replicas si su tasa verdadera de falsa deteccion fuese `true_rate` (banda fija): ``P(Binomial(n_eval, true_rate) <= max_detections)``.

    Mide lo poco o lo mucho que distingue el criterio de G2 (segunda
    auditoria, H2): con 29/500, una tasa verdadera del 5 % (el doble del
    nominal) pasa el 82 % de las veces. No rechazar NO es demostrar que
    la tasa sea el nominal.
    """
    from math import comb
    q = float(true_rate)
    return float(sum(comb(n_eval, k) * q**k * (1.0 - q) ** (n_eval - k) for k in range(min(int(max_detections), n_eval) + 1)))


def g2_status(g2_ok: bool | None) -> str:
    """Rotulo del resultado de G2. `True` significa "no rechazada en los controles evaluados", nunca "tasa nominal demostrada"."""
    return {True: G2_STATUS_COMPATIBLE, False: G2_STATUS_INCOMPATIBLE}.get(g2_ok, G2_STATUS_NOT_EVALUABLE)


def summarize_full_bootstrap(point: dict, replicates: pd.DataFrame, conditional: dict | None = None) -> pd.DataFrame:
    """IC percentil 95 % del bootstrap con re-estimacion para cada estadistico de la regla, junto al IC condicional previo cuando existe."""
    w = _wide(replicates, KIND_FULL_BOOTSTRAP)
    rows = []
    for key, val in point.items():
        if key not in w or "|year_" in key or key.endswith("n_exceed") or key == "n_eligible" or "n_complete_years" in key:
            continue
        series, side, name = key.split("|", 2)
        lo, hi = _ci(w[key].to_numpy())
        clo, chi = (conditional or {}).get(key, (np.nan, np.nan))
        rows.append({"series": series, "side": side, "statistic": name, "point": val, "ci_lo": lo, "ci_hi": hi, "n_boot": int(w[key].notna().sum()),
                     "boot_mean": float(w[key].mean()), "conditional_ci_lo": clo, "conditional_ci_hi": chi,
                     "ci_width_ratio_full_vs_conditional": (hi - lo) / (chi - clo) if np.isfinite(clo) and chi > clo else np.nan,
                     "covers": "umbrales por celda + referencia por minuto y año + conteos (jornadas remuestreadas)",
                     "does_not_cover": "sigma_hat/z/decil causal (fijos por barra), cortes de segmento de TDA-06, eleccion del estimador"})
    return pd.DataFrame(rows)


def decide_th27(point: dict, boot: pd.DataFrame, bands: pd.DataFrame, g2_ok: bool | None, rates: pd.DataFrame, replicates: pd.DataFrame) -> dict:
    """TH27 en TRES afirmaciones, sobre la variante primaria (nivel 1 %, `SEGMENT_X_VOL`, ventana 10, EWMA 60).

    Un cociente esta DETECTADO en una cola si supera la banda calibrada
    (percentil 97,5 del procedimiento completo sobre series sin
    dependencia) Y el IC 95 % del bootstrap con re-estimacion queda por
    encima de 1.

    1. `claim_1` -- hay agrupamiento frente a la independencia:
       `DETECTADO` si `r` se detecta en caidas y en subidas;
       `NO_DETECTADO` si en ninguna; `INCONCLUSO` si discrepan, si falta
       algun dato o si G2 queda rechazada o no es evaluable.
    2. `claim_2` -- persiste al estandarizar con el EWMA primario: solo
       se evalua si (1) es `DETECTADO`. `DETECTADO` si `z` se detecta en
       ambas colas y su cociente supera 1 en todos los años completos;
       `NO_DETECTADO` si `z` no se detecta en ninguna; `INCONCLUSO` en
       otro caso.
    3. `claim_3` -- existe un agrupamiento PROPIO de los extremos, no
       explicable por cambios de volatilidad: ESTE PROCEDIMIENTO no lo
       afirma ni lo descarta. Lo que se sabe es mas modesto: dividir por
       una volatilidad estimada con retraso deja agrupamiento residual
       aunque no exista ningun fenomeno propio, de modo que la
       afirmacion 2 no basta para identificarlo. Eso es una limitacion
       de ESTE metodo y de ESTOS datos (OHLCV de 1 minuto, sin una
       medida de volatilidad independiente del propio retorno), NO una
       prueba de que los dos fenomenos sean indistinguibles en general:
       con datos mas finos u otro diseño podrian separarse. Se devuelve
       `NO_IDENTIFICADO_POR_ESTE_PROCEDIMIENTO` junto con la evidencia
       descriptiva: para cada control de "solo volatilidad", si el
       cociente real de `z` cae dentro de lo que ese control produce.

    Veredicto: (1) no detectado -> `SIN_AGRUPAMIENTO_DETECTABLE`; (1)
    detectado y (2) no -> `AGRUPAMIENTO_NO_DETECTABLE_TRAS_ESTANDARIZAR`;
    (1) y (2) detectados ->
    `AGRUPAMIENTO_PERSISTE_CON_EWMA_ORIGEN_NO_IDENTIFICADO`; cualquier
    `INCONCLUSO` -> `INCONCLUSO`.
    """
    stat = f"ratio_le_{PRIMARY_WINDOW}"
    band = {(r_["series"], r_["side"]): float(r_["band_hi"]) for _, r_ in bands.iterrows()} if bands is not None and len(bands) else {}
    bci = {(r_["series"], r_["side"], r_["statistic"]): (float(r_["ci_lo"]), float(r_["ci_hi"])) for _, r_ in boot.iterrows()} if boot is not None and len(boot) else {}
    detail, det = {}, {}
    for side in (SIDE_NEG, SIDE_POS):
        detail[side] = {}
        for series in (SERIES_R, SERIES_Z):
            ratio = point.get(_key(series, side, stat), np.nan)
            hi = band.get((series, side), np.nan)
            lo_ci, hi_ci = bci.get((series, side, stat), (np.nan, np.nan))
            ok = None if not (np.isfinite(ratio) and np.isfinite(hi) and np.isfinite(lo_ci)) else bool(ratio > hi and lo_ci > 1.0)
            det[(series, side)] = ok
            tag = "r" if series == SERIES_R else "z"
            detail[side].update({f"{tag}_ratio": float(ratio), f"{tag}_band_hi": float(hi), f"{tag}_ci": (lo_ci, hi_ci), f"{tag}_detected": ok,
                                 f"{tag}_observed": float(point.get(_key(series, side, f"observed_le_{PRIMARY_WINDOW}"), np.nan)),
                                 f"{tag}_expected": float(point.get(_key(series, side, f"expected_le_{PRIMARY_WINDOW}"), np.nan))})
        min_year = point.get(_key(SERIES_Z, side, "min_year_ratio"), np.nan)
        n_years = point.get(_key(SERIES_Z, side, "n_complete_years"), 0.0)
        years = [point.get(_key(SERIES_Z, side, f"{stat}_year_{y}"), np.nan) for y in COMPLETE_YEARS]
        fr_lo, fr_hi = bci.get(("r_vs_z", side, "fraction_removed"), (np.nan, np.nan))
        detail[side].update({"z_min_year_ratio": float(min_year), "z_years_evaluated": int(n_years),
                             "z_years_above_1": int(sum(1 for v in years if np.isfinite(v) and v > 1.0)),
                             "z_stable_across_years": bool(np.isfinite(min_year) and n_years == len(COMPLETE_YEARS) and min_year > 1.0),
                             "fraction_removed": float(point.get(_key("r_vs_z", side, "fraction_removed"), np.nan)), "fraction_removed_ci": (fr_lo, fr_hi)})

    def combine(flags: list) -> str:
        if any(f is None for f in flags):
            return CLAIM_INCONCLUSIVE
        if all(flags):
            return CLAIM_YES
        return CLAIM_NO if not any(flags) else CLAIM_INCONCLUSIVE

    claim1_raw = combine([det[(SERIES_R, SIDE_NEG)], det[(SERIES_R, SIDE_POS)]])
    claim1 = claim1_raw if g2_ok is True else CLAIM_INCONCLUSIVE
    if claim1 != CLAIM_YES:
        claim2_raw = claim2 = "NO_EVALUADA" if claim1 == CLAIM_NO else CLAIM_INCONCLUSIVE
    else:
        z_flags = [det[(SERIES_Z, SIDE_NEG)], det[(SERIES_Z, SIDE_POS)]]
        claim2_raw = combine(z_flags)
        if claim2_raw == CLAIM_YES and not (detail[SIDE_NEG]["z_stable_across_years"] and detail[SIDE_POS]["z_stable_across_years"]):
            claim2_raw = CLAIM_INCONCLUSIVE
        claim2 = claim2_raw

    # Evidencia descriptiva para la afirmacion 3: ¿el cociente real de z cabe en lo que produce cada mundo de "solo volatilidad"?
    vol_evidence = []
    for fam in VOLATILITY_FAMILIES:
        w = _wide(replicates, fam) if replicates is not None and len(replicates) else pd.DataFrame()
        for side in (SIDE_NEG, SIDE_POS):
            key = _key(SERIES_Z, side, stat)
            if w.empty or key not in w:
                continue
            vals = w[key].dropna().to_numpy()
            real = point.get(key, np.nan)
            lo, hi = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
            sub = rates.loc[(rates["family"] == fam) & (rates["rule"].str.startswith("regla global: afirmacion 2"))] if rates is not None and len(rates) else pd.DataFrame()
            vol_evidence.append({"family": fam, "side": side, "n_replicates": int(vals.size), "z_ratio_real": float(real), "z_ratio_control_mean": float(vals.mean()),
                                 "z_ratio_control_p2_5": lo, "z_ratio_control_p97_5": hi, "z_ratio_control_max": float(vals.max()),
                                 "real_within_control_range": bool(np.isfinite(real) and real <= hi),
                                 "share_of_real_excess_reproduced": float((vals.mean() - 1.0) / (real - 1.0)) if np.isfinite(real) and real > 1 else np.nan,
                                 "claim_2_rate_in_control": float(sub["rate"].iloc[0]) if len(sub) else np.nan})
    vol_evidence = pd.DataFrame(vol_evidence)
    compatible = sorted(set(vol_evidence.loc[vol_evidence.groupby("family")["real_within_control_range"].transform("all"), "family"])) if len(vol_evidence) else []

    if claim1 == CLAIM_NO:
        verdict = TH27_NONE
    elif claim1 == CLAIM_INCONCLUSIVE:
        verdict = TH27_INCONCLUSIVE
    elif claim2 == CLAIM_NO:
        verdict = TH27_EXPLAINED
    elif claim2 == CLAIM_YES:
        verdict = TH27_PERSISTS
    else:
        verdict = TH27_INCONCLUSIVE
    return {"verdict": verdict, "claim_1": claim1, "claim_2": claim2, "claim_3": CLAIM_NOT_DEMONSTRABLE,
            "claim_1_before_g2_gate": claim1_raw, "claim_2_before_g2_gate": claim2_raw, "g2_ok": g2_ok, "g2_status": g2_status(g2_ok), "detail": detail,
            "volatility_only_evidence": vol_evidence, "volatility_controls_reproducing_real_z": compatible,
            "statistic": f"share_le_{PRIMARY_WINDOW}", "window_bars": PRIMARY_WINDOW}


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------

class _StageTimer:
    def __init__(self, total: int, verbose: bool):
        self.total, self.verbose, self._i = total, verbose, 0
        self.stage_timings: dict[str, float] = {}
        self.t_start = time.perf_counter()
        self._name, self._t = None, self.t_start

    def stage(self, name: str) -> None:
        now = time.perf_counter()
        if self._name is not None:
            self.stage_timings[self._name] = now - self._t
        self._i += 1
        self._name, self._t = name, now
        if self.verbose:
            print(f"[TDA13 {self._i}/{self.total}] {name}...", flush=True)

    def finish(self) -> dict[str, float]:
        now = time.perf_counter()
        if self._name is not None:
            self.stage_timings[self._name] = now - self._t
        self.stage_timings["_total_analysis"] = now - self.t_start
        return self.stage_timings


ANALYSIS_STAGES = 9
TOTAL_STAGES = 10


@dataclass
class TDA13Result:
    n_universe: int
    n_valid: int
    n_eligible: int
    n_days: int
    population_info: pd.DataFrame = None
    causality_sigma: pd.DataFrame = None
    causality_decile: dict = field(default_factory=dict)
    thresholds: pd.DataFrame = None
    inventory: pd.DataFrame = None
    triage_summary: pd.DataFrame = None
    triage_sensitivity: pd.DataFrame = None
    cluster_table: pd.DataFrame = None
    gap_distribution: pd.DataFrame = None
    run_distribution: pd.DataFrame = None
    cluster_by_group: pd.DataFrame = None
    hourly: pd.DataFrame = None
    triage_impact: pd.DataFrame = None
    reference_conservation: pd.DataFrame = None
    primary_point: dict = field(default_factory=dict)
    null_replicates: pd.DataFrame = None           # tabla larga: familias de control, bootstrap completo y diagnostico de la banda
    calibration_rates: pd.DataFrame = None
    calibration_bands: pd.DataFrame = None
    null_family_groups: pd.DataFrame = None        # tamaño de los grupos de permutacion de cada familia sin dependencia
    bernoulli_diagnostic: pd.DataFrame = None
    full_bootstrap: pd.DataFrame = None
    reference_checks: pd.DataFrame = None
    th06: dict = field(default_factory=dict)
    th27: dict = field(default_factory=dict)
    n_boot: int = 0
    n_null_sim: int = 0
    n_calibration: int = 0
    n_volatility_control: int = 0
    n_boot_full: int = 0
    n_workers: int = 1
    z_sensitivity_skipped: list = field(default_factory=list)
    seed: int = DEFAULT_SEED
    stage_timings: dict[str, float] = field(default_factory=dict)


def _hourly_table(hour: np.ndarray, day: np.ndarray, n_days: int, flags: dict, boot_w: np.ndarray) -> pd.DataFrame:
    """Distribucion horaria (hora NY) de las excedencias: tasa por hora con IC por jornadas, parte del total y "lift" (tasa de la hora / tasa global)."""
    rows = []
    n_bars = np.zeros((n_days, 24))
    np.add.at(n_bars, (day, hour), 1.0)
    for (series, definition, side), exc in flags.items():
        cnt = np.zeros((n_days, 24))
        np.add.at(cnt, (day[exc], hour[exc]), 1.0)
        total_rate = cnt.sum() / n_bars.sum()
        boot_rate = _safe_div(boot_w @ cnt, boot_w @ n_bars)
        for h in range(24):
            nb, ne = n_bars[:, h].sum(), cnt[:, h].sum()
            if nb == 0:
                continue
            lo, hi = _ci(boot_rate[:, h])
            rows.append({"series": series, "definition": definition, "side": side, "level": PRIMARY_LEVEL, "hour_ny": h,
                         "n_bars": int(nb), "n_exceed": int(ne), "rate": ne / nb, "rate_ci_lo": lo, "rate_ci_hi": hi,
                         "share_of_exceedances": ne / cnt.sum() if cnt.sum() else np.nan, "share_of_bars": nb / n_bars.sum(),
                         "lift_vs_overall": (ne / nb) / total_rate if total_rate else np.nan})
    return pd.DataFrame(rows)


def run_tda13_analysis(config: SnapshotConfig, verbose: bool = True) -> TDA13Result:
    """Ejecuta TDA-13 Parte A completa. No escribe nada (la persistencia es de `run_tda13.py`)."""
    timer = _StageTimer(TOTAL_STAGES, verbose)
    n_boot, n_sim, n_cal = int(config.tda13_n_boot), int(config.tda13_n_null_sim), int(config.tda13_n_calibration)
    n_vol, n_boot_full = int(config.tda13_n_volatility_control), int(config.tda13_n_boot_full)
    seed = DEFAULT_SEED

    # --- 1. Universo y evidencia de contexto --------------------------------
    timer.stage("Universo de triage: serie canonica + mascaras de TDA-00/02/03/04/05")
    uni = load_universe(config)
    valid = uni["r_1m_valid"].to_numpy(dtype=bool)
    finite_naive = np.isfinite(uni["r_naive_1m"].to_numpy(dtype=float))
    pop = uni.loc[valid].reset_index(drop=True)
    r_valid = pop["r_1m"].to_numpy(dtype=float)
    trading_date = pop["trading_date"].to_numpy()

    # --- 2. z_t y estado causal (TDA-10/12) + causalidad --------------------
    timer.stage("sigma_hat causal, z_t y decil causal (TDA-10/12) + verificacion de ausencia de look-ahead")
    causality_sigma = run_causality_checks(r_valid, (PRIMARY_ESTIMATOR,))
    ex = build_exceedances(r_valid, trading_date, pop["segment_label"].to_numpy())
    causality_decile = verify_causal_decile_no_lookahead(ex.sigma_prev, trading_date, valid=np.isfinite(ex.z))
    if not np.array_equal(causal_volatility_decile_fast(ex.sigma_prev, trading_date, valid=np.isfinite(ex.z)), ex.decile, equal_nan=True):
        raise RuntimeError("Inconsistencia interna: el decil causal rapido de las replicas no reproduce el de TDA-12 sobre los datos reales.")
    e = ex.eligible
    elig = pop.loc[e].reset_index(drop=True)
    unique_days, day_all = np.unique(uni["trading_date"].to_numpy(), return_inverse=True)
    n_days = int(unique_days.size)
    day_pop = day_all[valid]
    day_e = day_pop[e]
    minute_e, year_e, hour_e = elig["minute_of_day"].to_numpy(), elig["year_ny"].to_numpy(), elig["hour_ny"].to_numpy()
    seg_e = elig["segment_label"].astype(str).to_numpy()
    block_pop = compute_block_ids_with_contract(pop["timestamp"], pop["trading_date"], pop["contract"], 60.0)
    block_e = compute_block_ids_with_contract(elig["timestamp"], elig["trading_date"], elig["contract"], 60.0)
    boot_w = bootstrap_day_weights(n_days, n_boot, seed)

    # --- 3. Umbrales e inventario -------------------------------------------
    timer.stage("Umbrales relativos por celda e inventario de extremos")
    thr_rows = []
    for (name, definition, lvl), (table, _, _) in ex.thresholds.items():
        t = table.copy()
        t.insert(0, "definition", definition)
        t.insert(0, "series", SERIES_R if name == SERIES_R else SERIES_Z)
        t["is_primary"] = (definition == PRIMARY_DEFINITION) and (lvl == PRIMARY_LEVEL)
        t["label"] = "RETROSPECTIVO"
        thr_rows.append(t)
    thresholds = pd.concat(thr_rows, ignore_index=True)
    inventory = _build_inventory(uni, valid, finite_naive, ex, config.tick_size)
    inventory = classify_extremes(inventory)
    inventory["category_basis"] = inventory["category"].map(TH06_BASIS)
    triage_summary = _triage_summary(inventory)
    triage_sens = triage_sensitivity(inventory)
    th06 = decide_th06(inventory["category"].value_counts(), triage_sens)

    # --- 4. Agrupamiento: tabla principal -----------------------------------
    timer.stage("Agrupamiento observado vs referencia independiente (r y z; definiciones, niveles y lados)")
    cluster_rows, gap_tabs, run_tabs = [], [], []
    hourly_flags = {}
    for name, label in ((SERIES_R, SERIES_R), ("z", SERIES_Z)):
        for definition in DEFINITIONS:
            for lvl in THRESHOLD_LEVELS:
                neg, pos = ex.flags[(name, definition, lvl)]
                primary_level = lvl == PRIMARY_LEVEL
                for side in SIDES:
                    exc = side_flags(neg, pos, side)
                    rows, detail = analyze_variant(exc, minute_e, year_e, block_e, day_e, n_days, boot_w,
                                                   nulls=NULLS if primary_level else (PRIMARY_NULL,), n_sim=n_sim if primary_level else 0)
                    for row in rows:
                        row.update({"series": label, "definition": definition, "level": lvl, "side": side, "z_halflife": PRIMARY_ESTIMATOR[1]})
                    cluster_rows.extend(rows)
                    if primary_level:
                        hourly_flags[(label, definition, side)] = exc
                        g, rn = distribution_tables(detail, NULLS)
                        for tab, sink in ((g, gap_tabs), (rn, run_tabs)):
                            tab.insert(0, "side", side)
                            tab.insert(0, "definition", definition)
                            tab.insert(0, "series", label)
                            sink.append(tab)
    # Sensibilidad de la estandarizacion: otras semividas EWMA de TDA-10, mismas barras y mismas celdas.
    z_sens_skipped: list[float] = []
    for hl in Z_SENSITIVITY_HALFLIVES:
        alt = build_exceedances(r_valid, trading_date, pop["segment_label"].to_numpy(), levels=(PRIMARY_LEVEL,), definitions=(PRIMARY_DEFINITION,),
                                halflife=hl, eligible=e, decile=ex.decile)
        if not np.array_equal(alt.eligible, e):
            z_sens_skipped.append(hl)   # esa semivida no cubre la poblacion comun (quemado mas largo): se omite, no se compara sobre barras distintas
            continue
        neg, pos = alt.flags[("z", PRIMARY_DEFINITION, PRIMARY_LEVEL)]
        for side in SIDES:
            rows, _ = analyze_variant(side_flags(neg, pos, side), minute_e, year_e, block_e, day_e, n_days, boot_w, n_sim=n_sim)
            for row in rows:
                row.update({"series": f"z_ewma_{hl:g}_raw", "definition": PRIMARY_DEFINITION, "level": PRIMARY_LEVEL, "side": side, "z_halflife": hl})
            cluster_rows.extend(rows)
    cluster = pd.DataFrame(cluster_rows)
    cluster["is_primary"] = ((cluster["definition"] == PRIMARY_DEFINITION) & (cluster["level"] == PRIMARY_LEVEL) & (cluster["reference"] == PRIMARY_NULL)
                             & (cluster["z_halflife"] == PRIMARY_ESTIMATOR[1]))
    # Criterio PREVIO a la auditoria (IC condicional + banda Bernoulli): se conserva como columna informativa; ya NO decide.
    cluster["distinguishable_preaudit_criterion"] = [is_distinguishable(r_) for _, r_ in cluster.iterrows()]
    cluster["ci_type"] = "CONDICIONAL (umbrales y referencia fijos)"
    cluster["label"] = "RETROSPECTIVO"
    conservation = []
    for name, label in PRIMARY_SERIES:
        neg, pos = ex.flags[(name, PRIMARY_DEFINITION, PRIMARY_LEVEL)]
        for side in SIDES:
            tab = reference_conservation(side_flags(neg, pos, side), minute_e, year_e)
            tab.insert(0, "side", side)
            tab.insert(0, "series", label)
            conservation.append(tab)
    conservation = pd.concat(conservation, ignore_index=True)

    # --- 5. Por año y por segmento (G3) -------------------------------------
    timer.stage("Agrupamiento por año y por segmento (G3)")
    group_rows = []
    ks_small = (1, PRIMARY_WINDOW)
    for name, label in ((SERIES_R, SERIES_R), ("z", SERIES_Z)):
        neg, pos = ex.flags[(name, PRIMARY_DEFINITION, PRIMARY_LEVEL)]
        for side in SIDES:
            exc = side_flags(neg, pos, side)
            for grouping, values in (("YEAR", year_e), ("SEGMENT", seg_e)):
                for gval in pd.unique(values):
                    mask = values == gval
                    rows, _ = analyze_variant(exc, minute_e, year_e, block_e, day_e, n_days, boot_w, ks=ks_small, row_mask=mask)
                    for row in rows:
                        if row["statistic"] in (f"share_le_{k}" for k in ks_small) or row["statistic"] == "frac_in_runs_ge2":
                            row.update({"series": label, "side": side, "grouping": grouping, "group": str(gval), "n_bars": int(mask.sum()),
                                        "support": support_label(int(row["n_intervals"]))})
                            group_rows.append(row)
    by_group = pd.DataFrame(group_rows).sort_values(["series", "side", "grouping", "group", "statistic"]).reset_index(drop=True)

    # --- 6. Distribucion horaria --------------------------------------------
    timer.stage("Distribucion horaria de los extremos")
    hourly = _hourly_table(hour_e, day_e, n_days, hourly_flags, boot_w)

    # --- 7. Efecto del triage ------------------------------------------------
    timer.stage("Efecto del triage sobre curtosis y cuantiles extremos (con y sin cada categoria)")
    triage_impact = _triage_impact(uni, valid, finite_naive, day_all, inventory, ex, n_days, n_boot, seed)
    reference_checks = _reference_checks(config, triage_impact)

    # --- 8. Calibracion G2 con el procedimiento completo + bootstrap con re-estimacion -------------
    timer.stage("Calibracion G2 (procedimiento completo sobre series nulas), controles de solo volatilidad y bootstrap con re-estimacion")
    day_year = pd.to_datetime(pd.Series(unique_days)).dt.year.to_numpy()
    ctx = attach_primary_sample(ReplicateContext(
        r_valid=r_valid, trading_date=trading_date, segment=pop["segment_label"].astype(str).to_numpy(), minute=pop["minute_of_day"].to_numpy(),
        year=pop["year_ny"].to_numpy(), block_ids=block_pop, day=day_pop, n_days=n_days, day_year=day_year,
        sigma_paths=volatility_paths(r_valid),
    ), ex)
    null_groups = describe_null_family_groups(ctx.minute, ctx.year)
    # Los datos reales pasan por el MISMO camino de calculo que las replicas.
    primary_point = primary_replicate_statistics(r_valid, ctx)
    prim_rows = cluster.loc[cluster["is_primary"] & (cluster["statistic"] == f"share_le_{PRIMARY_WINDOW}")]
    for _, row in prim_rows.iterrows():
        via_rule = primary_point[_key(row["series"], row["side"], f"ratio_le_{PRIMARY_WINDOW}")]
        if not np.isclose(via_rule, row["ratio"], rtol=1e-9, atol=1e-12):
            raise RuntimeError(f"Inconsistencia interna: el cociente primario de {row['series']}/{row['side']} difiere entre la tabla ({row['ratio']}) y la regla ({via_rule}).")
    tasks = [(fam, rep) for fam in INDEPENDENCE_FAMILIES for rep in range(n_cal)]
    tasks += [(fam, rep) for fam in VOLATILITY_FAMILIES for rep in range(n_vol)]
    tasks += [(KIND_BERNOULLI_DIAG, rep) for rep in range(n_sim)]
    tasks += [(KIND_FULL_BOOTSTRAP, rep) for rep in range(n_boot_full)]
    requested = min(int(config.tda13_n_workers), MAX_WORKERS) if int(config.tda13_n_workers) > 0 else default_n_workers(len(tasks))
    free_gb = available_memory_gb()
    n_workers = memory_safe_workers(requested, free_gb) if requested > 1 else 1
    if verbose:
        print(f"[TDA13] procesos: {n_workers} (solicitados {requested}; memoria libre {'desconocida' if free_gb is None else f'{free_gb:.1f} GB'})", flush=True)
    replicates = run_tasks(ctx, tasks, n_workers)

    legacy_band = {(row["series"], row["side"]): float(row["null_sim_p97_5"]) for _, row in prim_rows.iterrows()}
    rates, bands, g2_ok = summarize_calibration(replicates, legacy_band)
    conditional = {_key(row["series"], row["side"], f"ratio_le_{PRIMARY_WINDOW}"): (float(row["ratio_ci_lo"]), float(row["ratio_ci_hi"])) for _, row in prim_rows.iterrows()}
    full_boot = summarize_full_bootstrap(primary_point, replicates, conditional)
    diag = _wide(replicates, KIND_BERNOULLI_DIAG)
    diag_rows = []
    for col in diag.columns:
        series, side, tag, _ = col.split("|")
        v = diag[col].dropna().to_numpy()
        diag_rows.append({"series": series, "side": side, "simulation": tag, "n": int(v.size), "ratio_mean": float(v.mean()), "ratio_std": float(v.std(ddof=1)) if v.size > 1 else np.nan,
                          "ratio_p2_5": float(np.percentile(v, 2.5)), "ratio_p97_5": float(np.percentile(v, 97.5))})
    for fam in INDEPENDENCE_FAMILIES:
        w = _wide(replicates, fam)
        for series in (SERIES_R, SERIES_Z):
            for side in (SIDE_NEG, SIDE_POS):
                key = _key(series, side, f"ratio_le_{PRIMARY_WINDOW}")
                if len(w) and key in w:
                    v = w[key].dropna().to_numpy()
                    diag_rows.append({"series": series, "side": side, "simulation": f"PROCEDIMIENTO_COMPLETO|{fam}", "n": int(v.size), "ratio_mean": float(v.mean()),
                                      "ratio_std": float(v.std(ddof=1)) if v.size > 1 else np.nan, "ratio_p2_5": float(np.percentile(v, 2.5)), "ratio_p97_5": float(np.percentile(v, 97.5))})
    bernoulli_diag = pd.DataFrame(diag_rows)
    th27 = decide_th27(primary_point, full_boot, bands, g2_ok, rates, replicates)

    population_info = pd.DataFrame([
        {"poblacion": "Barras de la serie canonica (TDA-04)", "n": int(len(uni))},
        {"poblacion": "Universo de triage: retorno contrafactual finito (validos + cruces de frontera)", "n": int(finite_naive.sum())},
        {"poblacion": "Retornos validos (`r_1m_valid`, poblacion de TDA-07)", "n": int(valid.sum())},
        {"poblacion": "Retornos que cruzan frontera/hueco/roll (solo para el triage)", "n": int((finite_naive & ~valid).sum())},
        {"poblacion": "Poblacion comun de agrupamiento (valido, `z` finito, decil causal definido)", "n": int(e.sum())},
        {"poblacion": "Validos sin estado de volatilidad (quemado / sin historia): fuera del agrupamiento", "n": int(valid.sum() - e.sum())},
        {"poblacion": "Bloques de continuidad en la poblacion comun", "n": int(np.unique(block_e).size)},
        {"poblacion": "Jornadas", "n": n_days},
    ])
    return TDA13Result(
        n_universe=int(finite_naive.sum()), n_valid=int(valid.sum()), n_eligible=int(e.sum()), n_days=n_days,
        population_info=population_info, causality_sigma=causality_sigma, causality_decile=causality_decile,
        thresholds=thresholds, inventory=inventory, triage_summary=triage_summary, cluster_table=cluster,
        gap_distribution=pd.concat(gap_tabs, ignore_index=True), run_distribution=pd.concat(run_tabs, ignore_index=True),
        cluster_by_group=by_group, hourly=hourly, triage_impact=triage_impact, triage_sensitivity=triage_sens,
        reference_conservation=conservation, primary_point=primary_point, null_replicates=replicates, calibration_rates=rates,
        calibration_bands=bands, null_family_groups=null_groups, bernoulli_diagnostic=bernoulli_diag, full_bootstrap=full_boot,
        reference_checks=reference_checks, th06=th06, th27=th27,
        n_boot=n_boot, n_null_sim=n_sim, n_calibration=n_cal, n_volatility_control=n_vol, n_boot_full=n_boot_full, n_workers=n_workers,
        z_sensitivity_skipped=z_sens_skipped, seed=seed, stage_timings=timer.finish(),
    )


def _build_inventory(uni: pd.DataFrame, valid: np.ndarray, finite_naive: np.ndarray, ex: ExceedanceSet, tick_size: float) -> pd.DataFrame:
    """Una fila por observacion extrema (nivel primario) del universo de triage.

    Barras de la poblacion comun: umbral de su celda `SEGMENT_X_VOL`.
    Resto (retornos que cruzan una frontera, o validos sin estado de
    volatilidad): no existe `sigma_hat_(t-1)` aplicable, asi que se usa
    el umbral de su SEGMENTO (calculado sobre la poblacion comun) y se
    deja anotado en `threshold_scope`.
    """
    n = len(uni)
    e_pop = ex.eligible
    idx_valid = np.flatnonzero(valid)
    idx_elig = idx_valid[e_pop]
    x = uni["r_naive_1m"].to_numpy(dtype=float)
    lo, hi = np.full(n, np.nan), np.full(n, np.nan)
    scope = np.full(n, "", dtype=object)
    _, lo_e, hi_e = ex.thresholds[(SERIES_R, PRIMARY_DEFINITION, PRIMARY_LEVEL)]
    lo[idx_elig], hi[idx_elig] = lo_e, hi_e
    scope[idx_elig] = PRIMARY_DEFINITION
    seg_table, _, _ = ex.thresholds[(SERIES_R, DEF_SEGMENT, PRIMARY_LEVEL)]
    seg_lo, seg_hi = seg_table.set_index("cell")["threshold_lo"], seg_table.set_index("cell")["threshold_hi"]
    rest = finite_naive.copy()
    rest[idx_elig] = False
    seg_all = uni["segment_label"].astype(str)
    lo[rest], hi[rest] = seg_all[rest].map(seg_lo).to_numpy(dtype=float), seg_all[rest].map(seg_hi).to_numpy(dtype=float)
    scope[rest] = f"{DEF_SEGMENT} (sin estado de volatilidad)"
    is_neg, is_pos = finite_naive & (x < lo), finite_naive & (x > hi)
    keep = is_neg | is_pos

    full = lambda arr_pop: _scatter(arr_pop, idx_valid, n)   # noqa: E731
    z_all, sig_all, dec_all = full(ex.z), full(ex.sigma_prev), full(ex.decile)
    zneg, zpos = ex.flags[("z", PRIMARY_DEFINITION, PRIMARY_LEVEL)]
    z_ext = np.zeros(n, dtype=bool)
    z_ext[idx_elig] = zneg | zpos
    deep = np.zeros(n, dtype=bool)
    dneg, dpos = ex.flags[(SERIES_R, PRIMARY_DEFINITION, min(SENSITIVITY_LEVELS))]
    deep[idx_elig] = dneg | dpos
    population = np.where(~valid, POP_CROSSING, POP_VALID_NO_STATE).astype(object)
    population[idx_elig] = POP_ELIGIBLE

    ts = uni["timestamp"]
    close_prev = uni["close"].shift(1)
    inv = pd.DataFrame({
        "timestamp_utc": ts, "timestamp_ny": ts.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d %H:%M"),
        "trading_date": uni["trading_date"], "year_ny": uni["year_ny"], "hour_ny": uni["hour_ny"], "minute_of_day": uni["minute_of_day"],
        "source_file": uni["source_file"], "contract": uni["contract"], "segment": seg_all, "vol_decile_causal": dec_all,
        "population": population, "direction": np.where(is_neg, SIDE_NEG, SIDE_POS),
        "r": x, "r_ticks": (uni["close"] - close_prev) / tick_size, "close_prev": close_prev, "close": uni["close"], "volume": uni["volume"],
        "sigma_hat_prev": sig_all, "z": z_all,
        "level": PRIMARY_LEVEL, "threshold_scope": scope, "threshold_lo": lo, "threshold_hi": hi,
        "threshold_applied": np.where(is_neg, lo, hi), "exceedance_ratio": np.abs(x) / np.abs(np.where(is_neg, lo, hi)),
        f"also_extreme_at_{min(SENSITIVITY_LEVELS):g}": deep, "also_extreme_in_z": z_ext,
        "invalid_reason": uni["invalid_reason"], "bad_data": uni["bad_data"], "bad_data_prev": uni["bad_data_prev"],
        "ff_category": uni["ff_category"], "ff_category_prev": uni["ff_category_prev"],
        "adjacent_unknown_gap": uni["adjacent_unknown_gap"], "next_bar_contiguous": uni["next_bar_contiguous"],
        "reverts_next_bar": uni["reverts_next_bar"], "reversal_fraction_next_bar": uni["reversal_fraction_next_bar"],
        "low_volume": uni["low_volume"], "low_volume_threshold": uni["low_volume_threshold"],
        "clock_0830_window": uni["clock_0830_window"],
    })
    return inv.loc[keep].reset_index(drop=True)


def _scatter(arr_pop: np.ndarray, idx_valid: np.ndarray, n: int) -> np.ndarray:
    out = np.full(n, np.nan)
    out[idx_valid] = np.asarray(arr_pop, dtype=float)
    return out


def _triage_summary(inv: pd.DataFrame) -> pd.DataFrame:
    """Conteo de categorias: total, por direccion, por año, por segmento, por poblacion y por duda de R4."""
    rows = []

    def add(grouping, key, sub):
        total = len(sub)
        for cat in CATEGORIES:
            c = int((sub["category"] == cat).sum())
            rows.append({"grouping": grouping, "group": str(key), "category": cat, "n": c, "n_group": total, "fraction": c / total if total else np.nan})

    add("TOTAL", "ALL", inv)
    for col, name in (("direction", "DIRECTION"), ("year_ny", "YEAR"), ("segment", "SEGMENT"), ("population", "POPULATION")):
        for key, sub in inv.groupby(col, sort=True):
            add(name, key, sub)
    und = inv.loc[inv["category"] == CAT_UNDETERMINED, "triage_evidence"]
    for doubt in ("BARRA_NO_ACTIVA", "JUNTO_A_HUECO_DESCONOCIDO", "SIN_BARRA_SIGUIENTE", "REVIERTE_BARRA_SIGUIENTE", "VOLUMEN_BAJO"):
        c = int(und.str.contains(doubt, regex=False).sum())
        rows.append({"grouping": "INDETERMINADO_POR_DUDA (no excluyentes)", "group": doubt, "category": CAT_UNDETERMINED, "n": c, "n_group": len(und),
                     "fraction": c / len(und) if len(und) else np.nan})
    only_rev = int((und == "REVIERTE_BARRA_SIGUIENTE").sum())
    rows.append({"grouping": "INDETERMINADO_POR_DUDA (no excluyentes)", "group": "SOLO_REVIERTE_BARRA_SIGUIENTE", "category": CAT_UNDETERMINED, "n": only_rev,
                 "n_group": len(und), "fraction": only_rev / len(und) if len(und) else np.nan})
    return pd.DataFrame(rows)


def _triage_impact(uni, valid, finite_naive, day_all, inventory, ex, n_days, n_boot, seed) -> pd.DataFrame:
    """Escenarios "con y sin cada categoria". Solo las observaciones EXTREMAS tienen categoria: quitar una categoria retira esos extremos y nada mas."""
    ts_all = uni["timestamp"].to_numpy()
    x = uni["r_naive_1m"].to_numpy(dtype=float)
    cat_ts = {c: inventory.loc[inventory["category"] == c, "timestamp_utc"].to_numpy() for c in CATEGORIES}
    is_cat = {c: np.isin(ts_all, cat_ts[c]) for c in CATEGORIES}
    scen, meta = {}, {}

    def add(name, mask, series, baseline, description, values=x):
        scen[name] = (values[mask], day_all[mask])
        meta[name] = {"series": series, "baseline": baseline, "description": description}

    u, v = "UNIVERSO_COMPLETO", "SOLO_VALIDOS"
    add(u, finite_naive, SERIES_R, u, "Todos los retornos consecutivos de la tabla, incluidos los que cruzan roll, jornada o hueco (contrafactual de TDA-07)")
    for c in CATEGORIES:
        add(f"UNIVERSO_sin_{c}", finite_naive & ~is_cat[c], SERIES_R, u, f"Universo completo sin los extremos clasificados {c}")
    add(v, valid, SERIES_R, v, "Poblacion de TDA-07: `r_1m` valido (excluye TODOS los cruces, sean o no extremos)")
    for c in (CAT_UNDETERMINED, CAT_REAL):
        add(f"VALIDOS_sin_{c}", valid & ~is_cat[c], SERIES_R, v, f"Validos sin los extremos {c}")
    add("VALIDOS_sin_EXTREMOS", valid & ~(is_cat[CAT_UNDETERMINED] | is_cat[CAT_REAL] | is_cat[CAT_ERROR]), SERIES_R, v,
        "Validos sin NINGUN extremo inventariado (referencia: cuanto de la cola es el propio inventario)")
    idx_valid = np.flatnonzero(valid)
    z_all = _scatter(ex.z, idx_valid, len(uni))
    zfin = np.isfinite(z_all)
    zb = "Z_TODO"
    add(zb, zfin, SERIES_Z, zb, "Poblacion de TDA-12: `z_t` finito", values=z_all)
    for c in (CAT_UNDETERMINED, CAT_REAL):
        add(f"Z_sin_{c}", zfin & ~is_cat[c], SERIES_Z, zb, f"`z_t` sin las barras cuyo retorno es un extremo {c}", values=z_all)
    return triage_impact_table(scen, meta, n_days, n_boot, seed)


def _reference_checks(config: SnapshotConfig, impact: pd.DataFrame) -> pd.DataFrame:
    """Comprobacion de trazabilidad: los escenarios de referencia deben reproducir los numeros ya publicados por TDA-07 (r) y TDA-12 (z). Solo LEE esos CSV; si no existen, lo declara."""
    rows = []

    def val(scenario, stat):
        s = impact.loc[(impact["scenario"] == scenario) & (impact["statistic"] == stat), "value"]
        return float(s.iloc[0]) if len(s) else np.nan

    p07 = config.reports_dir / "TDA07_momentos_cuantiles_drift_colas.csv"
    if p07.exists():
        t = pd.read_csv(p07)
        g = t.loc[(t["series"] == "r_1m") & (t["scope"] == "GLOBAL")]
        if len(g):
            for stat in ("kurtosis_excess", "kurtosis_excess_trimmed", "q0.001", "q0.01", "q0.99", "q0.999"):
                rows.append({"source": "TDA-07", "scenario": "SOLO_VALIDOS", "statistic": stat, "published": float(g[stat].iloc[0]), "recomputed": val("SOLO_VALIDOS", stat)})
    p12 = config.reports_dir / "TDA12_cuantiles_condicionales.csv"
    if p12.exists():
        t = pd.read_csv(p12)
        for series, scenario in ((SERIES_R, "SOLO_VALIDOS"), (SERIES_Z, "Z_TODO")):
            g = t.loc[(t["series"] == series) & (t["grouping"] == "GLOBAL")]
            for lvl in _TRIAGE_Q:
                q = g.loc[np.isclose(g["level"], lvl), "q"]
                if len(q):
                    rows.append({"source": "TDA-12", "scenario": scenario, "statistic": f"q{lvl:g}", "published": float(q.iloc[0]), "recomputed": val(scenario, f"q{lvl:g}")})
    out = pd.DataFrame(rows, columns=["source", "scenario", "statistic", "published", "recomputed"])
    if len(out):
        out["abs_diff"] = (out["published"] - out["recomputed"]).abs()
        out["matches"] = np.isclose(out["published"], out["recomputed"], rtol=1e-9, atol=1e-12)
    return out
