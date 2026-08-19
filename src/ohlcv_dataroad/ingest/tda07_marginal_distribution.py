"""TDA-07 -- Distribucion marginal y por segmento.

Implementa la etapa TDA-07 del roadmap
(``docs/methodology/Tsay_OHLCV_analysis_roadmap.md``, seccion "TDA-07"):
caracteriza la distribucion de los retornos de 1 minuto -- media, escala,
asimetria, curtosis, cuantiles -- con la conciencia de que la marginal
agregada es una MEZCLA de segmentos horarios (TDA-06) y de años. Cierra,
ademas, el componente distribucional que TDA-04 dejo explicitamente
diferido de TH08 (comparacion de momentos/cuantiles con y sin las reglas
de no-cruce).

Que responde esta etapa, en una frase: ¿cuanta curtosis hay en r_1m, cuanto
depende de un puñado de observaciones, cuanto cambia por segmento y por
año, es la media distinguible de cero, difieren las colas, y cuanto habria
contaminado la distribucion agregada si las reglas de no-cruce de TDA-04
no hubieran existido?

Que NO hace este modulo (deliberadamente, fuera de alcance de TDA-07):

- No calcula ACF/PACF ni busca dependencia estocastica (TDA-08/TDA-09).
- No estudia volatility clustering ni ajusta ARCH/GARCH (TDA-09).
- No ejecuta EVT (TDA-13).
- No entrena modelos, no crea targets, no busca predictibilidad.
- No retoma TH10 (sigue diferida).
- No investiga las dos ventanas de posible efecto en media de TDA-06 (§7:
  09:31-09:35, 15:52-16:02 NY) como señal -- eso es TDA-08.
- No modifica ningun artefacto de TDA-00..TDA-06: reutiliza
  ``attach_calendar_fields`` de TDA-06 y lee la segmentacion propuesta
  como un archivo mas, sin alterarla.
- No ajusta ningun test de significancia como hallazgo (G5): todo se
  reporta como (estimador puntual, intervalo, magnitud interpretable).
- No abre ``data/raw/`` ni ``config.holdout_files``.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from statistics import NormalDist

import numpy as np
import pandas as pd

from ohlcv_dataroad.config import SnapshotConfig
from ohlcv_dataroad.ingest.holdout_guard import (
    validate_last_timestamps_before_boundary,
    validate_research_holdout_disjoint,
)
from ohlcv_dataroad.ingest.session_calendar import NY_TZ
from ohlcv_dataroad.ingest.tda06_intraday_calendar_profile import attach_calendar_fields

# ---------------------------------------------------------------------------
# Convenciones PREDECLARADAS (fijadas antes de mirar el resultado real,
# seccion "TH08"/"convenciones" de la tarea: "define las convenciones
# exactamente antes de interpretar resultados; no pruebes multiples
# variantes para elegir la que produzca la historia mas interesante").
# ---------------------------------------------------------------------------

# Cuantiles del roadmap (TDA-07, metodo minimo 3): 0.1/1/5/25/50/75/95/99/99.9,
# expresados como fraccion.
QUANTILE_LEVELS: tuple[float, ...] = (0.001, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999)

# "Recorte del 0.1% mas extremo" (roadmap, metodo minimo 2): interpretado
# como el 0.1% TOTAL de la muestra (no 0.1% por cada cola) -- la lectura
# habitual de "recorte de X%" en estadistica robusta (p.ej.
# ``scipy.stats.mstats.trimboth`` con ``proportiontocut`` por lado suma al
# total declarado). Se retira, por cada lado, la mitad: 0.05%.
TRIM_FRACTION_TOTAL = 0.001
TRIM_FRACTION_EACH_TAIL = TRIM_FRACTION_TOTAL / 2.0

# Umbral de excedencia SIMETRICO para TH13 (frecuencia de excedencias por
# lado): percentil 99 de |x| dentro de CADA grupo (global/año/segmento),
# el mismo percentil que TDA-06 uso para su bandera de extremo (0.99) --
# reutiliza el metodo, no el numero: se recalcula aqui sobre la poblacion
# de TDA-07 (r_1m o r_tilde, segun la serie), nunca se importa el valor ya
# calculado por TDA-06.
TAIL_EXCEEDANCE_QUANTILE = 0.99

# Bandwidth de Newey-West (Tsay/roadmap, docs/tsay/Tsay_Cap2_..., Ec. 2.50):
# l = floor(4*(T/100)^(2/9)). Formula CERRADA, sin ajustar tras ver el
# resultado.
def hac_bandwidth(T: int) -> int:
    if T <= 0:
        return 0
    return int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))


# Tolerancia para la invariante "r_naive_1m == r_1m en filas VALID"
# (seccion "INVARIANTES BLOQUEANTES" de la tarea). Ambas series se
# calculan con la MISMA formula (``ln(close_t/close_{t-1})``) sobre la
# MISMA columna ``close`` -- deberian coincidir bit a bit; esta tolerancia
# solo absorbe una eventual diferencia de orden de operaciones de punto
# flotante entre las dos implementaciones (TDA-04 vs este modulo), nunca
# una discrepancia de metodo.
NAIVE_MATCH_TOLERANCE = 1e-9

# Numero de puntos usados para dibujar un QQ-plot sobre una muestra de
# millones de filas (seccion outputs de la tarea): se comparan cuantiles
# empiricos en una grilla de probabilidades fija contra los cuantiles
# teoricos de la normal en esa misma grilla -- la practica estandar para
# QQ-plots de muestras grandes (graficar cada punto seria ilegible y no
# aporta nada que la grilla no muestre ya).
QQ_N_POINTS = 500

REASON_FIRST_OBSERVATION = "FIRST_OBSERVATION"
REASON_ROLL_BOUNDARY = "ROLL_BOUNDARY"
REASON_TRADING_DATE_BOUNDARY = "TRADING_DATE_BOUNDARY"
REASON_NON_CONSECUTIVE_MINUTE = "NON_CONSECUTIVE_MINUTE"
REASON_VALID = "VALID"
CONTAMINATING_REASONS = (REASON_ROLL_BOUNDARY, REASON_TRADING_DATE_BOUNDARY, REASON_NON_CONSECUTIVE_MINUTE)

CONTRAFACTUAL_LABEL = "CONTRAFACTUAL_VIOLA_NO_CRUCE"


class TimestampAlignmentError(Exception):
    """``tda04_variables_1m.parquet`` y ``tda04_return_validity_mask.parquet`` no comparten timestamps exactos.

    Invariante bloqueante #1 de esta etapa (nunca se asume por orden de
    filas, aunque ambos archivos se ordenen por ``timestamp`` antes de
    compararlos). Si se dispara, TDA-07 se detiene sin producir ningun
    resultado distribucional.
    """


class NaiveReturnContradictionError(Exception):
    """``r_naive_1m`` no coincide con ``r_1m`` de TDA-04 en alguna fila ``VALID``.

    Invariante bloqueante #2. Ambas series deben ser identicas (dentro de
    ``NAIVE_MATCH_TOLERANCE``) en toda fila donde TDA-04 certifico
    ``invalid_reason == "VALID"`` -- si no coinciden, hay una
    inconsistencia real entre la formula de TDA-04 y la de este modulo, no
    un resultado a interpretar. TDA-07 se detiene sin producir ningun
    resultado distribucional.
    """


class RTildeInvariantError(Exception):
    """``tda06_r_tilde.parquet`` no esta alineado con TDA-04, o su etiqueta no es ``RETROSPECTIVO`` en toda fila.

    Invariante bloqueante #3 (corrección puntual de cierre), exigida antes
    de analizar ``r_tilde`` en TH11/TH12/TH13: (1) mismos timestamps,
    mismo orden y mismo numero de filas que
    ``tda04_variables_1m.parquet`` -- ``r_tilde`` (TDA-06,
    ``build_r_tilde``) tiene una fila por cada fila de la serie canonica,
    no solo por las validas; (2) ``label == "RETROSPECTIVO"`` en TODAS las
    filas -- si alguna careciera de la etiqueta, se estaria a punto de
    tratar una cantidad no marcada como retrospectiva igual que una
    causal (G1). Si cualquiera falla, TDA-07 se detiene sin analizar
    ``r_tilde`` -- pero TH08/TH11-TH13 sobre ``r_1m`` YA calculados no se
    invalidan (esta invariante solo protege la rama de ``r_tilde``).
    """


# ---------------------------------------------------------------------------
# Carga de entradas
# ---------------------------------------------------------------------------

def load_tda04_inputs(config: SnapshotConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee variables y mascara de validez de TDA-04 -- sin recalcular nada.

    Se carga POR SEPARADO de ``r_tilde`` (``load_r_tilde``) para que las
    invariantes bloqueantes y el cierre de TH08 -- que solo necesitan
    TDA-04 -- puedan fallar rapido sin depender de un artefacto de TDA-06
    no relacionado con esa comprobacion.
    """
    variables = pd.read_parquet(config.tda04_variables_parquet_path).sort_values("timestamp").reset_index(drop=True)
    validity = pd.read_parquet(config.tda04_validity_mask_parquet_path).sort_values("timestamp").reset_index(drop=True)
    return variables, validity


def load_r_tilde(config: SnapshotConfig) -> pd.DataFrame:
    """Lee ``r_tilde`` (RETROSPECTIVO) de TDA-06 -- sin recalcular nada.

    TDA-07 nunca abre ``data/raw/`` ni ``config.holdout_files``: sus
    entradas son exclusivamente artefactos ya construidos por etapas
    anteriores sobre el conjunto de investigacion.
    """
    return pd.read_parquet(config.tda06_r_tilde_parquet_path).sort_values("timestamp").reset_index(drop=True)


def verify_r_tilde_invariants(r_tilde: pd.DataFrame, variables: pd.DataFrame) -> None:
    """Invariante #3 (bloqueante): ``r_tilde`` alineado con TDA-04 y etiquetado ``RETROSPECTIVO`` en toda fila.

    Se ejecuta ANTES de construir la poblacion de analisis de ``r_tilde``
    (``build_r_tilde_population``) -- si falla, ninguna tabla de
    TH11/TH12/TH13 para ``r_tilde`` se produce. No modifica
    ``tda06_r_tilde.parquet``: solo lo lee y lo compara.
    """
    r = r_tilde.sort_values("timestamp").reset_index(drop=True)
    v = variables.sort_values("timestamp").reset_index(drop=True)

    if len(r) != len(v):
        raise RTildeInvariantError(
            f"tda06_r_tilde.parquet tiene {len(r)} filas y tda04_variables_1m.parquet tiene {len(v)} -- "
            "no son la misma serie canonica (TDA-06 construye r_tilde con una fila por cada fila de "
            "variables, valida o no). TDA-07 se detiene sin analizar r_tilde."
        )

    mismatch = r["timestamp"].to_numpy() != v["timestamp"].to_numpy()
    if mismatch.any():
        n_bad = int(mismatch.sum())
        first_bad = int(np.argmax(mismatch))
        raise RTildeInvariantError(
            f"{n_bad} timestamp(s) no coinciden entre tda06_r_tilde.parquet y tda04_variables_1m.parquet "
            f"(primera discrepancia en la posicion {first_bad}: {r['timestamp'].iloc[first_bad]} vs "
            f"{v['timestamp'].iloc[first_bad]}). TDA-07 se detiene sin analizar r_tilde."
        )

    bad_label = r["label"] != "RETROSPECTIVO"
    if bad_label.any():
        n_bad = int(bad_label.sum())
        raise RTildeInvariantError(
            f"{n_bad} fila(s) de tda06_r_tilde.parquet NO tienen label='RETROSPECTIVO'. "
            "TDA-07 se detiene sin analizar r_tilde -- no se trata una cantidad sin etiquetar "
            "como si fuera retrospectiva (G1)."
        )


# ---------------------------------------------------------------------------
# Invariantes bloqueantes (seccion "INVARIANTES BLOQUEANTES" de la tarea)
# ---------------------------------------------------------------------------

def verify_timestamp_alignment(variables: pd.DataFrame, validity: pd.DataFrame) -> None:
    """Invariante #1: ``variables`` y ``validity`` comparten EXACTAMENTE los mismos timestamps, en el mismo orden.

    Nunca se asume por longitud o por orden de fila -- se compara elemento
    a elemento tras ordenar ambas tablas por ``timestamp`` (la misma
    disciplina que TDA-05/TDA-06 ya aplican con un ``assert``; aqui se
    formaliza como excepcion propia porque la tarea exige que la etapa se
    DETENGA, no solo que falle un assert interno).
    """
    v_ts = variables["timestamp"].to_numpy()
    m_ts = validity["timestamp"].to_numpy()
    if len(v_ts) != len(m_ts):
        raise TimestampAlignmentError(
            f"tda04_variables_1m.parquet tiene {len(v_ts)} filas y "
            f"tda04_return_validity_mask.parquet tiene {len(m_ts)} -- no son la misma serie. "
            "TDA-07 se detiene sin producir ningun resultado."
        )
    mismatch = v_ts != m_ts
    if mismatch.any():
        n_bad = int(mismatch.sum())
        first_bad = int(np.argmax(mismatch))
        raise TimestampAlignmentError(
            f"{n_bad} timestamp(s) no coinciden entre tda04_variables_1m.parquet y "
            f"tda04_return_validity_mask.parquet (primera discrepancia en la posicion {first_bad}: "
            f"{v_ts[first_bad]} vs {m_ts[first_bad]}). TDA-07 se detiene sin producir ningun resultado."
        )


def build_naive_return(variables: pd.DataFrame) -> pd.DataFrame:
    """``r_naive_1m = ln(close_t/close_{t-1})`` por ``shift(1)`` INCONDICIONAL sobre TODA la serie canonica.

    A diferencia de ``r_1m`` de TDA-04, aqui NO se aplica ninguna regla de
    no-cruce: la fila anterior en la tabla se usa siempre, sea cual sea su
    ``trading_date``, ``segment_id`` o separacion temporal real. Es,
    deliberadamente, la version que TDA-04 se nego a construir.

    ``close`` nunca es ``NaN`` (TDA-00 certifico 0 valores no finitos en
    los precios); por construccion, ``r_naive_1m`` es ``NaN`` EXACTAMENTE
    en la primera fila de la serie completa (no existe ``t-1``) -- ninguna
    otra fila queda ``NaN`` en esta version contrafactual, a diferencia de
    ``r_1m`` que tiene 3.520 filas ``NaN`` por las reglas de no-cruce.

    Salida: ``DataFrame`` con ``timestamp``, ``trading_date``,
    ``r_naive_1m``, ``label="CONTRAFACTUAL_VIOLA_NO_CRUCE"`` -- un
    artefacto propio de TDA-07, nunca escrito sobre ``tda04_variables_1m``.
    """
    df = variables.sort_values("timestamp").reset_index(drop=True)
    prev_close = df["close"].shift(1)
    r_naive = np.log(df["close"].to_numpy() / prev_close.to_numpy())
    return pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "trading_date": df["trading_date"],
            "r_naive_1m": r_naive,
            "label": CONTRAFACTUAL_LABEL,
        }
    )


def verify_naive_matches_valid(
    naive_1m: np.ndarray, r_1m: np.ndarray, invalid_reason: np.ndarray, tol: float = NAIVE_MATCH_TOLERANCE
) -> None:
    """Invariante #2: en toda fila ``invalid_reason == "VALID"``, ``r_naive_1m`` debe coincidir con ``r_1m``.

    Ambos arrays deben estar alineados fila a fila con ``invalid_reason``
    (mismo orden de timestamp que ``verify_timestamp_alignment`` ya
    certifico). Si alguna fila ``VALID`` difiere mas de ``tol``, la
    contradiccion se reporta y la etapa se detiene -- nunca se sigue
    adelante con un resultado parcial.
    """
    valid_mask = invalid_reason == REASON_VALID
    diff = np.abs(naive_1m[valid_mask] - r_1m[valid_mask])
    bad = diff > tol
    if bad.any():
        n_bad = int(bad.sum())
        max_diff = float(np.nanmax(diff))
        raise NaiveReturnContradictionError(
            f"{n_bad} fila(s) con invalid_reason=VALID donde r_naive_1m != r_1m "
            f"(tolerancia={tol}, diferencia maxima observada={max_diff}). "
            "Esto indica una inconsistencia real entre la formula de TDA-04 y la de TDA-07 "
            "(no un resultado a interpretar). TDA-07 se detiene sin producir ningun resultado."
        )


# ---------------------------------------------------------------------------
# TH08 -- cierre del componente distribucional
# ---------------------------------------------------------------------------

def build_contrafactual_table(variables: pd.DataFrame, validity: pd.DataFrame) -> pd.DataFrame:
    """Ensambla la tabla contrafactual completa: ``r_naive_1m`` + ``invalid_reason`` de TDA-04.

    Ejecuta, en orden, las dos invariantes bloqueantes ANTES de devolver
    nada -- si cualquiera falla, la excepcion correspondiente se propaga y
    ninguna tabla se construye.

    Salida: ``DataFrame`` con ``timestamp``, ``trading_date``, ``r_1m``
    (de TDA-04, para comparacion directa), ``r_naive_1m``,
    ``invalid_reason``, ``r_1m_valid``, ``label``.
    """
    v = variables.sort_values("timestamp").reset_index(drop=True)
    m = validity.sort_values("timestamp").reset_index(drop=True)
    verify_timestamp_alignment(v, m)

    naive = build_naive_return(v)
    verify_naive_matches_valid(
        naive["r_naive_1m"].to_numpy(), v["r_1m"].to_numpy(), m["invalid_reason"].to_numpy()
    )

    return pd.DataFrame(
        {
            "timestamp": v["timestamp"],
            "trading_date": v["trading_date"],
            "r_1m": v["r_1m"],
            "r_naive_1m": naive["r_naive_1m"],
            "invalid_reason": m["invalid_reason"],
            "r_1m_valid": m["r_1m_valid"],
            "label": CONTRAFACTUAL_LABEL,
        }
    )


def th08_global_comparison(contrafactual: pd.DataFrame) -> pd.DataFrame:
    """Comparacion PRINCIPAL de TH08: A) r_1m valido vs B) r_naive_1m global completo.

    A) = ``r_1m`` donde ``r_1m_valid=True`` (exactamente la poblacion de
    TDA-04, TDA-05, TDA-06).
    B) = ``r_naive_1m`` donde no es ``NaN`` -- que, por construccion
    (``build_naive_return``), es EXACTAMENTE toda la serie salvo la unica
    fila ``FIRST_OBSERVATION`` (sin ``t-1``, excluida de forma natural, no
    por una regla añadida aqui).

    Mide directamente cuanto cambiarian los momentos y cuantiles de la
    distribucion GLOBAL si las reglas de no-cruce de TDA-04 no hubieran
    existido -- el objetivo declarado de este cierre de TH08.

    Salida: ``DataFrame`` de dos filas (``population="A_r_1m_valid"``,
    ``population="B_r_naive_global"``) con las columnas de
    ``compute_moments_quantiles``.
    """
    a = contrafactual.loc[contrafactual["r_1m_valid"], "r_1m"].to_numpy(dtype=float)
    b = contrafactual.loc[contrafactual["r_naive_1m"].notna(), "r_naive_1m"].to_numpy(dtype=float)

    rows = [
        {"population": "A_r_1m_valid", **compute_moments_quantiles(a)},
        {"population": "B_r_naive_global", **compute_moments_quantiles(b)},
    ]
    return pd.DataFrame(rows)


def th08_secondary_by_cause(contrafactual: pd.DataFrame) -> pd.DataFrame:
    """Diagnostico SECUNDARIO de TH08: dentro de B, aisla los retornos contaminantes por ``invalid_reason``.

    Subordinado a ``th08_global_comparison`` -- caracteriza por separado
    los tres subconjuntos que la regla de no-cruce de TDA-04 excluyo:
    ``ROLL_BOUNDARY`` (n pequeño, 21 en el conjunto de investigacion),
    ``TRADING_DATE_BOUNDARY`` (n=1.398) y ``NON_CONSECUTIVE_MINUTE``
    (n=2.100). ``FIRST_OBSERVATION`` queda fuera por construccion (no
    tiene ``t-1``, nunca aparece en ``r_naive_1m``).

    Con ``n`` pequeño, los cuantiles profundos (0.1/99.9) de estos
    subconjuntos son poco fiables -- se reportan de todas formas, con
    ``n`` visible en cada fila, para que el lector juzgue (G5: la magnitud
    y su incertidumbre son el resultado, no un umbral de "suficiente
    muestra" impuesto aqui).

    Salida: ``DataFrame`` con una fila por causa contaminante.
    """
    rows = []
    for reason in CONTAMINATING_REASONS:
        subset = contrafactual.loc[contrafactual["invalid_reason"] == reason, "r_naive_1m"].to_numpy(dtype=float)
        rows.append({"invalid_reason": reason, **compute_moments_quantiles(subset)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Momentos y cuantiles -- motor generico reutilizado por TH08/TH11
# ---------------------------------------------------------------------------

def compute_moments_quantiles(x: np.ndarray) -> dict:
    """Momentos, cuantiles y curtosis con/sin recorte -- convencion UNICA, reutilizada en toda la etapa.

    Definiciones (predeclaradas, seccion "convenciones" de la tarea):

    - ``mean``, ``std`` (``ddof=1``, desviacion MUESTRAL -- la escala
      reportada).
    - ``skewness = m3 / m2^1.5`` y ``kurtosis_excess = m4 / m2^2 - 3``,
      con momentos centrales POBLACIONALES (``ddof=0``, dividir por ``n``,
      no por ``n-1``) -- la definicion estandar del tercer/cuarto momento
      estandarizado (equivalente a ``scipy.stats.skew``/``kurtosis`` con
      ``bias=True``, sin dependencia de ``scipy``).
    - ``kurtosis_excess_trimmed``: la MISMA formula de curtosis, recalculada
      sobre la muestra tras retirar el ``TRIM_FRACTION_EACH_TAIL`` (0.05%)
      mas extremo de CADA cola (0.1% total, roadmap TDA-07 metodo minimo
      2) -- la diferencia entre ambas curtosis es el resultado (si colapsa,
      el estimador sin recortar estaba dominado por un puñado de puntos).
    - Cuantiles en ``QUANTILE_LEVELS``.

    Entrada: array 1-D SIN ``NaN`` (el llamador filtra la poblacion
    correcta antes de invocar esta funcion).
    Salida: ``dict`` con ``n``, ``mean``, ``std``, ``skewness``,
    ``kurtosis_excess``, ``kurtosis_excess_trimmed``, ``min``, ``max``, y
    una clave ``q{nivel}`` por cada nivel de ``QUANTILE_LEVELS``.
    """
    x = np.asarray(x, dtype=float)
    n = int(x.size)
    if n == 0:
        out = {"n": 0, "mean": np.nan, "std": np.nan, "skewness": np.nan,
               "kurtosis_excess": np.nan, "kurtosis_excess_trimmed": np.nan,
               "min": np.nan, "max": np.nan}
        out.update({f"q{lvl}": np.nan for lvl in QUANTILE_LEVELS})
        return out

    mean = float(x.mean())
    std = float(x.std(ddof=1)) if n > 1 else float("nan")

    e = x - mean
    m2 = float(np.mean(e**2))
    m3 = float(np.mean(e**3))
    m4 = float(np.mean(e**4))
    skewness = (m3 / m2**1.5) if m2 > 0 else float("nan")
    kurtosis_excess = (m4 / m2**2 - 3.0) if m2 > 0 else float("nan")

    if n >= 20:
        lo_cut, hi_cut = np.quantile(x, [TRIM_FRACTION_EACH_TAIL, 1.0 - TRIM_FRACTION_EACH_TAIL])
        trimmed = x[(x >= lo_cut) & (x <= hi_cut)]
        if trimmed.size >= 2:
            e_t = trimmed - trimmed.mean()
            m2_t = float(np.mean(e_t**2))
            m4_t = float(np.mean(e_t**4))
            kurtosis_excess_trimmed = (m4_t / m2_t**2 - 3.0) if m2_t > 0 else float("nan")
        else:
            kurtosis_excess_trimmed = float("nan")
    else:
        # Recortar el 0.1% de una muestra pequeña (<20) retiraria una
        # fraccion de un solo punto o menos -- no es un recorte
        # interpretable (G5: se declara NaN explicito, no se fabrica un
        # numero de un recorte que en la practica no elimino nada).
        kurtosis_excess_trimmed = float("nan")

    out = {
        "n": n, "mean": mean, "std": std, "skewness": skewness,
        "kurtosis_excess": kurtosis_excess, "kurtosis_excess_trimmed": kurtosis_excess_trimmed,
        "min": float(x.min()), "max": float(x.max()),
    }
    quantile_values = np.quantile(x, QUANTILE_LEVELS)
    out.update({f"q{lvl}": float(q) for lvl, q in zip(QUANTILE_LEVELS, quantile_values)})
    return out


# ---------------------------------------------------------------------------
# HAC (Newey-West) para la media -- TH12
# ---------------------------------------------------------------------------
#
# CORRECCION PUNTUAL DE CIERRE (posterior al cierre inicial de TDA-07).
#
# Bug detectado: la version original de ``hac_mean_se`` recibia el array
# ``values`` ya COMPACTADO (filtrado a ``r_1m_valid=True``, y ademas
# restringido por año/segmento en ``analyze_group``) y calculaba
# ``gamma_j = mean(e[j:] * e[:-j])`` asumiendo que la posicion ``t`` y la
# posicion ``t-j`` del array eran observaciones temporalmente separadas
# por exactamente ``j`` MINUTOS. Eso es FALSO en general: dos filas
# ``r_1m_valid=True`` consecutivas EN EL ARRAY COMPACTADO pueden estar
# separadas, en el reloj real, por una fila invalidada por TDA-04
# (``NON_CONSECUTIVE_MINUTE``, ``TRADING_DATE_BOUNDARY``,
# ``ROLL_BOUNDARY``), por la ventana de mantenimiento, o -- en el analisis
# POR SEGMENTO -- por muchas horas o jornadas completas (el mismo tramo
# horario de dos dias distintos queda "adyacente" en el array compactado
# tras filtrar por ``segment_label``, aunque el reloj real los separe
# ~24h). Compactar y aplicar HAC sobre el array resultante fabricaba
# dependencia temporal artificial entre observaciones que TDA-04 decidio
# explicitamente NO conectar (la misma regla de no-cruce que TH08 cierra
# en esta etapa, §"cierre de TH08").
#
# Solucion elegida: en vez de suponer continuidad por posicion de array,
# se construyen explicitamente ``block_ids`` (``compute_hac_block_ids``)
# a partir de ``timestamp``/``trading_date`` de la POBLACION analizada
# (ya sea GLOBAL, un año o un segmento) -- dos observaciones consecutivas
# (por timestamp) quedan en el MISMO bloque si y solo si estan separadas
# por EXACTAMENTE 60 segundos Y comparten ``trading_date``: exactamente la
# misma condicion que TDA-04 exige para ``r_1m_valid=True`` (delta=1
# minuto, mismo trading_date, no roll). No hace falta comprobar el roll
# por separado: TDA-04 (informe, §5) certifico que TODO roll ocurre en
# una frontera de ``trading_date`` (nunca a mitad de una), asi que exigir
# ``trading_date`` identico excluye automaticamente cualquier roll.
# ``hac_mean_se`` solo permite que el par ``(t, t-j)`` contribuya a
# ``gamma_j`` cuando ``block_ids[t] == block_ids[t-j]`` -- como los ids
# son monotonos no decrecientes y constantes dentro de cada tramo
# continuo, esa igualdad garantiza que TODAS las posiciones intermedias
# tambien pertenecen al mismo bloque (no solo los dos extremos del par).
#
# La MEDIA observada no cambia (sigue siendo ``x.mean()``, sin bloques):
# lo unico que se corrige es la estimacion de su incertidumbre.

HAC_EXPECTED_DELTA_SECONDS = 60.0
HAC_DELTA_TOLERANCE_SECONDS = 1e-6


def compute_hac_block_ids(timestamp: pd.Series, trading_date: pd.Series) -> np.ndarray:
    """Bloques de continuidad temporal GENUINA de una poblacion, para HAC.

    Entrada: ``timestamp`` y ``trading_date`` de la poblacion analizada
    (mismo orden, ya filtrada a ``r_1m_valid=True`` y, si corresponde, a
    un año/segmento). Ambas se reordenan por ``timestamp`` internamente
    (defensivo).

    Salida: array de enteros, un ``block_id`` por fila, IGUAL para dos
    filas si y solo si pertenecen al mismo tramo continuo (delta=60s,
    mismo ``trading_date`` respecto de la fila anterior). La PRIMERA fila
    de cada tramo (incluida la primera fila de toda la poblacion) abre un
    bloque nuevo.
    """
    order = np.argsort(pd.to_datetime(timestamp).to_numpy())
    ts_sorted = pd.to_datetime(timestamp).to_numpy()[order]
    td_sorted = np.asarray(trading_date)[order]

    n = len(ts_sorted)
    if n == 0:
        return np.array([], dtype=int)

    delta_seconds = np.diff(ts_sorted).astype("timedelta64[ns]").astype(np.int64) / 1e9
    same_date = td_sorted[1:] == td_sorted[:-1]
    continues = (np.abs(delta_seconds - HAC_EXPECTED_DELTA_SECONDS) < HAC_DELTA_TOLERANCE_SECONDS) & same_date
    new_block = np.concatenate([[True], ~continues])
    block_ids_sorted = np.cumsum(new_block) - 1

    # Devolver en el orden ORIGINAL de entrada (no necesariamente ya
    # ordenado) -- invierte la permutacion ``order``.
    block_ids = np.empty(n, dtype=int)
    block_ids[order] = block_ids_sorted
    return block_ids


def hac_mean_se(x: np.ndarray, block_ids: np.ndarray) -> tuple[float, int]:
    """Error estandar HAC (Newey-West) de la media muestral de ``x``, respetando bloques de continuidad.

    Caso particular de la Ec. (2.50) de Tsay (``docs/tsay/Tsay_Cap2_...``)
    cuando el unico regresor es una constante: el "sandwich" se reduce a
    ``Var_HAC(mean) = (1/T) * [gamma_0 + 2*sum_{j=1}^l w_j*gamma_j]``, con
    ``gamma_j`` la autocovarianza muestral de rezago ``j`` y
    ``w_j = 1 - j/(l+1)`` (pesos de Bartlett).

    Correccion de continuidad (ver bloque de comentarios arriba): el par
    ``(x[t], x[t-j])`` SOLO contribuye a la SUMA de productos de rezago
    ``j`` si ``block_ids[t] == block_ids[t-j]`` -- es decir, si ambas
    observaciones pertenecen al MISMO tramo temporalmente continuo
    (``compute_hac_block_ids``).

    Correccion de NORMALIZACION (segunda correccion puntual, posterior a
    la de continuidad): ``gamma_j`` se define como la SUMA (no el
    promedio) de los productos ``e[t]*e[t-j]`` sobre los pares
    ``same_block``, dividida por ``T`` -- la muestra COMPLETA, nunca por
    el numero de pares que sobreviven al filtro de bloque. Derivacion:
    para regresion sobre una constante, el sandwich de Newey-West
    (Ec. 2.50 de Tsay) se reduce a
    ``Var_HAC(mean) = (1/T^2) * [sum_t e_t^2 + 2*sum_j w_j*SumProd_j]``,
    con ``SumProd_j = sum_{(t,t-j)} e_t*e_{t-j}`` (SUMA cruda, sin
    normalizar). Bajo el supuesto de que los bloques son mutuamente
    independientes (sin covarianza cruzada, la misma aproximacion
    conservadora que motiva excluir esos pares en primer lugar), esta
    suma se reduce EXACTAMENTE a sumar solo los productos de pares
    ``same_block`` -- los pares cross-block aportan CERO al numerador,
    no una "renormalizacion" del denominador. Dividir por el numero de
    pares sobrevivientes (``same_block.sum()``) en vez de por ``T``
    sobreponderaria los pares que sí sobreviven, exactamente al reves de
    lo que "aportar cero" significa. Con ``l << T`` (bandwidth tipico
    decenas, ``T`` del orden de 10^5-10^6), la diferencia entre normalizar
    por ``T`` o por ``T-j`` es numericamente despreciable; se elige ``T``
    por ser la que reproduce EXACTAMENTE el sandwich de Tsay Ec. 2.50 para
    el caso de regresor constante (ver tambien el bloque de comentarios de
    modulo, mas arriba, y el test
    ``test_hac_mean_se_normalizes_by_full_sample_not_by_surviving_pair_count``).

    ``x`` y ``block_ids`` DEBEN estar en el MISMO orden temporal (por
    ``timestamp``) -- el llamador (``analyze_group``) es responsable de
    construir ambos con ``compute_hac_block_ids`` sobre la misma
    poblacion, en el mismo orden.

    Salida: ``(se, l)`` -- error estandar HAC y bandwidth usado
    (``hac_bandwidth``, formula cerrada de Newey-West, basado en el
    TOTAL de observaciones ``T`` -- el bandwidth no cambia con ninguna de
    las dos correcciones).
    """
    x = np.asarray(x, dtype=float)
    block_ids = np.asarray(block_ids)
    T = x.size
    l = hac_bandwidth(T)
    if T <= 1:
        return float("nan"), l

    xbar = x.mean()
    e = x - xbar
    gamma0 = float(np.mean(e * e))
    s = gamma0
    for j in range(1, l + 1):
        if j >= T:
            break
        same_block = block_ids[j:] == block_ids[:-j]
        if not same_block.any():
            continue
        # SUMA (no promedio) de los pares same_block, normalizada por T
        # -- ver derivacion en el docstring: los pares cross-block ya
        # estan excluidos de la suma (aportan 0), y el denominador es la
        # muestra COMPLETA, no ``same_block.sum()``.
        sum_prod = float(np.sum((e[j:] * e[:-j])[same_block]))
        gamma_j = sum_prod / T
        w_j = 1.0 - j / (l + 1.0)
        s += 2.0 * w_j * gamma_j

    var_mean = max(s, 0.0) / T
    return float(np.sqrt(var_mean)), l


# ---------------------------------------------------------------------------
# Bootstrap de bloques por jornada -- motor generico (drift + asimetria de colas)
# ---------------------------------------------------------------------------

def _day_block_index_map(trading_date: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Precalcula, UNA vez, los indices de fila que corresponden a cada ``trading_date`` distinta.

    Mismo patron de rendimiento que ``tda05_effective_resolution.block_bootstrap_global_metrics``
    (agrupar indices una sola vez, NumPy puro, sin reconstruir un
    ``DataFrame`` por repeticion de bootstrap).
    """
    unique_dates, inverse = np.unique(trading_date, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    boundaries = np.searchsorted(sorted_inverse, np.arange(len(unique_dates) + 1))
    date_to_rows = [order[boundaries[i]:boundaries[i + 1]] for i in range(len(unique_dates))]
    return unique_dates, date_to_rows


def day_block_bootstrap(
    values: np.ndarray, trading_date: np.ndarray, stat_fn, n_boot: int, seed: int
) -> np.ndarray:
    """Bootstrap de BLOQUES de jornada (G5): remuestrea ``trading_date`` completas, con reemplazo.

    Motor UNICO reutilizado para el intervalo de la media (TH12, drift) y
    para los intervalos de asimetria de colas (TH13) -- una sola pasada de
    remuestreo por grupo produce todas las estadisticas de ese grupo a la
    vez (``stat_fn`` puede devolver un vector), en vez de bootstrapear cada
    estadistico por separado.

    ``stat_fn`` recibe el array de valores YA remuestreado (mismo orden
    interno que produce la concatenacion de bloques -- no necesita ser
    temporal, ninguna de las estadisticas usadas aqui depende del orden)
    y devuelve un escalar o un array 1-D.

    Salida: array ``(n_boot, k)`` (o ``(n_boot,)`` si ``stat_fn`` devuelve
    un escalar) con una fila por repeticion.
    """
    unique_dates, date_to_rows = _day_block_index_map(trading_date)
    n_dates = len(unique_dates)
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_boot):
        sampled = rng.integers(0, n_dates, size=n_dates)
        idx = np.concatenate([date_to_rows[i] for i in sampled])
        results.append(stat_fn(values[idx]))
    return np.asarray(results)


def _drift_and_tail_stat(x: np.ndarray, threshold_v: float) -> np.ndarray:
    """Estadistico combinado (drift + asimetria de colas) evaluado sobre UNA remuestra.

    Devuelve, en un solo vector, todo lo que TH12/TH13 necesitan de una
    misma remuestra -- evita bootstrapear el mismo grupo dos veces:

    0. ``mean`` (TH12).
    1. ``diff_0_1pct = q_0.999 - |q_0.001|`` (TH13, cola profunda). Cero si
       las colas son simetricas EN MAGNITUD a ese nivel; positivo si la
       cola derecha es mas extrema, negativo si lo es la izquierda.
    2. ``diff_1pct = q_0.99 - |q_0.01|`` (TH13, cola menos profunda, mas
       muestra que la respalda).
    3. ``freq_left = P(x <= -threshold_v)``.
    4. ``freq_right = P(x >= threshold_v)``.
    5. ``freq_diff = freq_right - freq_left`` (TH13, frecuencia de
       excedencias por lado a un umbral SIMETRICO fijo, calculado una
       unica vez sobre la muestra puntual -- nunca recalculado dentro del
       bootstrap, para que el intervalo mida incertidumbre sobre la
       frecuencia a un umbral FIJO, no sobre un umbral que tambien varia).
    """
    q_low_deep, q_high_deep = np.quantile(x, [0.001, 0.999])
    q_low, q_high = np.quantile(x, [0.01, 0.99])
    return np.array(
        [
            x.mean(),
            q_high_deep - abs(q_low_deep),
            q_high - abs(q_low),
            float((x <= -threshold_v).mean()),
            float((x >= threshold_v).mean()),
            float((x >= threshold_v).mean()) - float((x <= -threshold_v).mean()),
        ]
    )


# ---------------------------------------------------------------------------
# Traduccion a ticks (solo para r_1m -- ver nota en analyze_group)
# ---------------------------------------------------------------------------

def compute_tick_return_repr(close_values: np.ndarray, tick_size: float) -> float:
    """``ln((C_repr+tick)/C_repr)`` con ``C_repr`` = Close mediano del grupo -- misma convencion que TDA-05.

    Un tick de PUNTOS vale una fraccion de retorno distinta segun el nivel
    de precio (TDA-05, seccion 9) -- por eso la traduccion usa el Close
    mediano REPRESENTATIVO del propio grupo (global/año/segmento), no una
    unica cifra para todo el conjunto de investigacion.
    """
    close_repr = float(np.median(close_values))
    return float(np.log((close_repr + tick_size) / close_repr))


# ---------------------------------------------------------------------------
# Segmentacion de TDA-06 -- particion, nunca modificada
# ---------------------------------------------------------------------------

def load_segmentation_cutoffs(config: SnapshotConfig) -> list[int]:
    """Lee los cortes ESTABLES de ``TDA06_segmentacion_propuesta.csv`` -- nunca recalculados aqui.

    Filtra por ``stable == True`` (la columna que TDA-06 ya calculo) para
    ser robusto si una ejecucion futura de TDA-06 tuviera menos de 6
    cortes estables -- sobre el conjunto de investigacion real, esto
    conserva los 6 cortes (02:00, 03:00, 08:30, 09:30, 16:02, 20:00 NY)
    documentados en el informe de TDA-06.
    """
    seg = pd.read_csv(config.tda06_segmentation_csv_path)
    stable = seg.loc[seg["stable"].astype(bool), "minute_of_day"]
    return sorted(int(m) for m in stable.to_numpy())


def _minute_to_hhmm(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


def assign_segment_label(minute_of_day: pd.Series, cutoffs: list[int]) -> pd.Series:
    """Etiqueta cada fila con su tramo horario (``HH:MM-HH:MM``) segun los cortes de TDA-06.

    Los cortes dividen 0..1439 en ``len(cutoffs)+1`` tramos contiguos
    (TDA-06, seccion 9 del informe: "7 tramos horarios"). Esta es una
    PARTICION EMPIRICA para el analisis de TDA-07 (repetir TH11/12/13 "por
    segmento") -- no una arquitectura de ML, exactamente como advierte el
    informe de TDA-06.
    """
    edges = [0] + sorted(cutoffs) + [1440]
    labels = [f"{_minute_to_hhmm(edges[i])}-{_minute_to_hhmm(edges[i + 1])}" for i in range(len(edges) - 1)]
    categorized = pd.cut(minute_of_day, bins=edges, right=False, labels=labels, include_lowest=True)
    return categorized.astype(str), labels


# ---------------------------------------------------------------------------
# Poblaciones de analisis: r_1m (crudo) y r_tilde (ajustado, RETROSPECTIVO)
# ---------------------------------------------------------------------------

def build_r1m_population(variables: pd.DataFrame, validity: pd.DataFrame, cutoffs: list[int]) -> tuple[pd.DataFrame, list[str]]:
    """Poblacion de analisis de ``r_1m`` crudo: filas ``r_1m_valid=True``, con año/segmento adjuntos.

    Reutiliza ``attach_calendar_fields`` de TDA-06 (``minute_of_day``,
    ``year_ny``, ``weekday``, ``r_1m_valid``) SIN modificar ese modulo --
    la misma logica DST-aware de minuto-del-dia que ya certifico TDA-06.
    """
    df = attach_calendar_fields(variables, validity)
    df = df.loc[df["r_1m_valid"]].copy()
    df["segment_label"], labels = assign_segment_label(df["minute_of_day"], cutoffs)
    return df, labels


def build_r_tilde_population(r_tilde: pd.DataFrame, cutoffs: list[int]) -> tuple[pd.DataFrame, list[str]]:
    """Poblacion de analisis de ``r_tilde`` (RETROSPECTIVO): filas no nulas, con año/segmento adjuntos.

    ``r_tilde`` ya trae ``minute_of_day``/``trading_date`` (TDA-06); solo
    hace falta ``year_ny`` (misma conversion NY que TDA-06, aplicada aqui
    porque ``tda06_r_tilde.parquet`` no la persiste) y la etiqueta de
    segmento. La etiqueta ``label="RETROSPECTIVO"`` de TDA-06 se conserva
    intacta en la columna de origen -- esta funcion no la sobrescribe.
    """
    df = r_tilde.loc[r_tilde["r_tilde"].notna()].copy()
    ts_ny = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(NY_TZ)
    df["year_ny"] = ts_ny.dt.year
    df["segment_label"], labels = assign_segment_label(df["minute_of_day"], cutoffs)
    return df, labels


# ---------------------------------------------------------------------------
# Analisis por grupo (GLOBAL / año / segmento) -- TH11 + TH12 + TH13 juntos
# ---------------------------------------------------------------------------

def analyze_group(
    group_df: pd.DataFrame,
    value_col: str,
    n_boot: int,
    seed: int,
    tick_return_repr: float | None,
) -> dict:
    """Momentos/cuantiles (TH11) + drift HAC/bootstrap (TH12) + asimetria de colas (TH13) sobre UN grupo.

    ``tick_return_repr``: si se provee (siempre para ``r_1m``; ``None``
    para ``r_tilde``, ver nota mas abajo), traduce la media y su intervalo
    a TICKS. ``r_tilde`` es una cantidad DES-ESTACIONALIZADA (dividida por
    ``s(m)``, RETROSPECTIVO) -- no es literalmente un retorno de precio, y
    traducirla a ticks introduciria una convencion no estandar y discutible
    (cuantos "ticks" representa un r_tilde no es una pregunta bien
    definida). Por eso ``r_tilde`` se reporta SOLO en sus unidades propias
    (comparables en escala a ``r_1m``, TDA-06 §11: ``std(r_tilde)~std(r_1m)``),
    nunca traducido a ticks -- una medida ambigua no se inventa (evita
    exactamente el riesgo que la tarea señalo para "curtosis por lado").

    Igual razon (evitar una metrica no estandar) explica por que TH13 NO
    usa "curtosis por lado": se sustituye por dos medidas directamente
    interpretables y ya definidas mas arriba
    (``_drift_and_tail_stat``): la diferencia entre cuantiles simetricos en
    magnitud (``|q_p|`` vs ``q_{1-p}``, en las mismas unidades que la
    variable) y la frecuencia de excedencias por lado a un umbral fijo
    (una proporcion, unidad ya interpretable por si misma) -- ambas
    responden exactamente la pregunta de TH13 ("¿difieren las colas
    izquierda y derecha en magnitud y frecuencia?") sin definir un momento
    de cuarto orden partido a mitad de muestra, que no tiene una
    definicion estandar unica en la literatura.
    """
    group_df = group_df.sort_values("timestamp").reset_index(drop=True)
    values = group_df[value_col].to_numpy(dtype=float)
    trading_date = group_df["trading_date"].to_numpy()
    n = values.size

    moments = compute_moments_quantiles(values)

    if n < 2:
        drift_tail = {
            "hac_se": float("nan"), "hac_l": 0, "hac_ci_lo": float("nan"), "hac_ci_hi": float("nan"),
            "boot_mean_ci_lo": float("nan"), "boot_mean_ci_hi": float("nan"),
            "threshold_v": float("nan"),
            "diff_0_1pct": float("nan"), "diff_0_1pct_ci_lo": float("nan"), "diff_0_1pct_ci_hi": float("nan"),
            "diff_1pct": float("nan"), "diff_1pct_ci_lo": float("nan"), "diff_1pct_ci_hi": float("nan"),
            "freq_left": float("nan"), "freq_right": float("nan"),
            "freq_diff": float("nan"), "freq_diff_ci_lo": float("nan"), "freq_diff_ci_hi": float("nan"),
        }
    else:
        block_ids = compute_hac_block_ids(group_df["timestamp"], group_df["trading_date"])
        hac_se, hac_l = hac_mean_se(values, block_ids)
        mean = moments["mean"]
        hac_ci_lo, hac_ci_hi = mean - 1.96 * hac_se, mean + 1.96 * hac_se

        threshold_v = float(np.quantile(np.abs(values), TAIL_EXCEEDANCE_QUANTILE))
        point_stat = _drift_and_tail_stat(values, threshold_v)
        boot = day_block_bootstrap(
            values, trading_date, partial(_drift_and_tail_stat, threshold_v=threshold_v), n_boot, seed
        )
        ci_lo = np.percentile(boot, 2.5, axis=0)
        ci_hi = np.percentile(boot, 97.5, axis=0)

        drift_tail = {
            "hac_se": hac_se, "hac_l": hac_l, "hac_ci_lo": hac_ci_lo, "hac_ci_hi": hac_ci_hi,
            "boot_mean_ci_lo": float(ci_lo[0]), "boot_mean_ci_hi": float(ci_hi[0]),
            "threshold_v": threshold_v,
            "diff_0_1pct": float(point_stat[1]), "diff_0_1pct_ci_lo": float(ci_lo[1]), "diff_0_1pct_ci_hi": float(ci_hi[1]),
            "diff_1pct": float(point_stat[2]), "diff_1pct_ci_lo": float(ci_lo[2]), "diff_1pct_ci_hi": float(ci_hi[2]),
            "freq_left": float(point_stat[3]), "freq_right": float(point_stat[4]),
            "freq_diff": float(point_stat[5]), "freq_diff_ci_lo": float(ci_lo[5]), "freq_diff_ci_hi": float(ci_hi[5]),
        }

    if tick_return_repr is not None and tick_return_repr != 0:
        drift_tail["mean_ticks"] = moments["mean"] / tick_return_repr
        drift_tail["hac_ci_lo_ticks"] = drift_tail["hac_ci_lo"] / tick_return_repr
        drift_tail["hac_ci_hi_ticks"] = drift_tail["hac_ci_hi"] / tick_return_repr
    else:
        drift_tail["mean_ticks"] = float("nan")
        drift_tail["hac_ci_lo_ticks"] = float("nan")
        drift_tail["hac_ci_hi_ticks"] = float("nan")

    return {**moments, **drift_tail}


def _iter_scopes(df: pd.DataFrame, segment_labels: list[str]):
    """Genera ``(scope, scope_value, subframe)`` para GLOBAL, cada año y cada segmento -- orden fijo."""
    yield "GLOBAL", "GLOBAL", df
    for year in sorted(df["year_ny"].unique()):
        yield "YEAR", int(year), df.loc[df["year_ny"] == year]
    for label in segment_labels:
        yield "SEGMENT", label, df.loc[df["segment_label"] == label]


def build_distribution_tables(
    r1m_pop: pd.DataFrame, r_tilde_pop: pd.DataFrame, segment_labels: list[str],
    tick_size: float, n_boot: int, seed: int,
) -> pd.DataFrame:
    """Orquesta TH11+TH12+TH13 sobre ``r_1m`` y ``r_tilde``, GLOBAL + por año + por segmento (TDA-06).

    Para ``r_tilde``, la traduccion a ticks se omite (``tick_return_repr=None``,
    ver ``analyze_group``); para ``r_1m`` se calcula un ``tick_return_repr``
    propio de cada grupo (Close mediano DE ESE grupo, TDA-05 §"unidades").

    Salida: un unico ``DataFrame`` "largo" con columnas ``series``,
    ``scope`` (``GLOBAL``/``YEAR``/``SEGMENT``), ``scope_value``, y todas
    las columnas de ``analyze_group`` (moments + drift + tail) -- una fila
    por combinacion serie x alcance.
    """
    rows = []
    for scope, scope_value, sub in _iter_scopes(r1m_pop, segment_labels):
        tick_repr = compute_tick_return_repr(sub["close"].to_numpy(dtype=float), tick_size) if len(sub) else None
        result = analyze_group(sub, "r_1m", n_boot, seed, tick_repr)
        rows.append({"series": "r_1m", "scope": scope, "scope_value": scope_value, **result})

    for scope, scope_value, sub in _iter_scopes(r_tilde_pop, segment_labels):
        result = analyze_group(sub, "r_tilde", n_boot, seed, tick_return_repr=None)
        rows.append({"series": "r_tilde", "scope": scope, "scope_value": scope_value, **result})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# QQ-plot -- sin dependencia de scipy
# ---------------------------------------------------------------------------

def qq_points(x: np.ndarray, n_points: int = QQ_N_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """Cuantiles teoricos (normal estandar) vs empiricos ESTANDARIZADOS, en una grilla de probabilidades.

    Con muestras de millones de filas, graficar cada punto es ilegible y
    no aporta nada que una grilla de ``n_points`` probabilidades no
    muestre ya -- practica estandar para QQ-plots de muestras grandes.
    ``NormalDist().inv_cdf`` (libreria estandar de Python, sin ``scipy``)
    calcula el cuantil teorico de la normal estandar.

    Salida: ``(theoretical_q, empirical_q)``, ambos ``(n_points,)``.
    """
    x = np.asarray(x, dtype=float)
    mean, std = x.mean(), x.std(ddof=1)
    standardized = (x - mean) / std
    probs = (np.arange(1, n_points + 1) - 0.5) / n_points
    empirical_q = np.quantile(standardized, probs)
    theoretical_q = np.array([NormalDist().inv_cdf(p) for p in probs])
    return theoretical_q, empirical_q


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------

@dataclass
class TDA07Result:
    contrafactual: pd.DataFrame
    th08_global: pd.DataFrame
    th08_by_cause: pd.DataFrame
    segment_labels: list[str]
    segment_cutoffs: list[int]
    r1m_population: pd.DataFrame
    r_tilde_population: pd.DataFrame
    distribution_tables: pd.DataFrame


def run_tda07_analysis(config: SnapshotConfig) -> TDA07Result:
    """Orquesta TDA-07 completo: invariantes -> TH08 -> TH11/TH12/TH13 (r_1m y r_tilde, global/año/segmento).

    Reutiliza, sin duplicar logica: la proteccion del hold-out
    (``holdout_guard.py``), los artefactos de TDA-04 (variables + mascara
    de validez), el artefacto ``RETROSPECTIVO`` de TDA-06 (``r_tilde``) y
    ``attach_calendar_fields``/la segmentacion propuesta de TDA-06 (leidas,
    nunca modificadas).
    """
    validate_research_holdout_disjoint(config)

    variables, validity = load_tda04_inputs(config)
    last_timestamps = variables.groupby("source_file")["timestamp"].max().to_dict()
    validate_last_timestamps_before_boundary(config, last_timestamps)

    # --- Invariantes bloqueantes + TH08 (nunca se sigue si fallan) -------
    contrafactual = build_contrafactual_table(variables, validity)
    th08_global = th08_global_comparison(contrafactual)
    th08_by_cause = th08_secondary_by_cause(contrafactual)

    # --- TH11/TH12/TH13 (r_tilde se carga recien aqui) --------------------
    r_tilde = load_r_tilde(config)
    verify_r_tilde_invariants(r_tilde, variables)
    cutoffs = load_segmentation_cutoffs(config)
    r1m_pop, segment_labels = build_r1m_population(variables, validity, cutoffs)
    r_tilde_pop, _ = build_r_tilde_population(r_tilde, cutoffs)

    distribution_tables = build_distribution_tables(
        r1m_pop, r_tilde_pop, segment_labels, config.tick_size, config.tda07_n_boot, seed=0
    )

    return TDA07Result(
        contrafactual=contrafactual,
        th08_global=th08_global,
        th08_by_cause=th08_by_cause,
        segment_labels=segment_labels,
        segment_cutoffs=cutoffs,
        r1m_population=r1m_pop,
        r_tilde_population=r_tilde_pop,
        distribution_tables=distribution_tables,
    )
