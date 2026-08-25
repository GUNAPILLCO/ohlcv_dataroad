"""TDA-10 -- Escala versus forma: origen de las colas.

Implementa la etapa TDA-10 del roadmap
(``docs/methodology/Tsay_OHLCV_analysis_roadmap.md``, seccion "TDA-10"):
responde que fraccion de la NO-normalidad marginal de ``r_1m`` (colas
gruesas, exceso de curtosis, ya caracterizadas en TDA-07) es ESCALA
(heterocedasticidad -- la volatilidad cambia en el tiempo, TDA-09 ya
demostro que ese cambio tiene memoria genuina) y que fraccion es FORMA
(la distribucion condicional, incluso despues de quitar correctamente la
escala, sigue siendo anormalmente pesada).

Hereda expresamente de TDA-09 (CLOSED, `PASS_WITH_OPEN_QUESTIONS`):
``VOLATILITY_CLUSTERING_DETECTABLE`` (TH19), `CLUSTERING_GENUINO` (TH21),
`STOP-9` NO activado, y la advertencia explicita de que las fracciones de
energia de TDA-09 (``clock_attribution``) NUNCA se leen como una
descomposicion causal -- esta etapa no reutiliza ese numero para nada,
construye su propia comparacion (curtosis antes/despues de estandarizar
por volatilidad CAUSAL, no por energia de ACF).

Que es "z_t" aqui, en una frase: ``z_t = r_t / sigma_hat_{t-1}``, con
``sigma_hat_{t-1}`` una estimacion de volatilidad que usa EXCLUSIVAMENTE
informacion disponible antes de ``t`` (G1, causalidad estricta) -- nunca
el propio ``r_t``, nunca una ventana centrada, nunca un parametro
estimado sobre toda la muestra y reutilizado retrospectivamente como si
fuera causal.

CONVENCIONES PREDECLARADAS (fijadas antes de mirar el resultado real,
igual disciplina que TDA-06/07/08/09 -- ver seccion de constantes mas
abajo para el detalle y la justificacion de cada valor):

1. Dos familias minimas de estimadores de volatilidad CAUSAL (roadmap,
   metodo minimo 1): desviacion estandar rodante (``causal_rolling_sigma``)
   y EWMA (``causal_ewma_sigma``), cada una con una grilla PEQUEÑA y
   predeclarada de configuraciones -- nunca una busqueda masiva de
   hiperparametros (G4).
2. Dos series de entrada: "raw" (``r_1m``, genuinamente causal) y
   "clock_adjusted" (``r_tilde``, RETROSPECTIVO -- TDA-06 estimo ``s(m)``
   sobre TODA la muestra). La version "clock_adjusted" NUNCA se presenta
   como disponible en produccion; existe exclusivamente para separar la
   escala DETERMINISTA (hora del dia) de la escala DINAMICA (volatilidad
   reciente) -- exactamente la distincion que la tarea exige no confundir.
3. Ventana de "quemado" (burn-in): las primeras filas de cada
   configuracion, donde el estimador causal aun no tiene suficiente
   historia para ser confiable, se excluyen de TODO analisis de esta
   etapa (nunca se reporta un ``z_t`` construido con un ``sigma_hat``
   poco fiable).
4. Umbrales de clasificacion del veredicto (ESCALA DOMINA / FORMA
   SUSTANCIAL / MIXTO) -- declarados en las constantes ``*_THRESHOLD``
   mas abajo, ANTES de ejecutar el analisis sobre el conjunto de
   investigacion real. Nunca se ajustan despues de ver el resultado.

Que NO hace este modulo (deliberadamente, fuera de alcance de TDA-10):

- No entrena ningun modelo de ML, no crea targets, no crea señales, no
  hace backtest, no evalua rentabilidad.
- No ajusta GARCH ni ningun modelo parametrico de volatilidad (eso es
  TDA-11, condicional, y solo si esta etapa NO responde la pregunta).
- No calcula cuantiles condicionales completos con bootstrap por grupo y
  ``n`` por cuantil extremo (eso es TDA-12, obligatorio, TH26 completa) --
  esta etapa aporta unicamente la comparacion de PERFILES de cuantiles
  ESTANDARIZADOS que el roadmap pide como metodo minimo 5 de TDA-10, y
  que sirve como evidencia PARCIAL para TH26, nunca como su resolucion
  formal.
- No ejecuta EVT (TDA-13). No activa formalmente ``STOP-13`` (esa
  activacion formal ocurre en TDA-12, seccion "criterios de
  interpretacion" del roadmap) -- esta etapa como maximo SUGIERE si
  ``STOP-13`` es probable, documentado explicitamente como sugerencia,
  no como decision.
- No abre ``data/raw/`` ni ``config.holdout_files``.
- No modifica ningun artefacto de TDA-00..TDA-09.
- No inicia TDA-11, TDA-12, TDA-13.
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
from ohlcv_dataroad.ingest.tda07_marginal_distribution import (
    QUANTILE_LEVELS,
    RTildeInvariantError,
    TimestampAlignmentError,
    build_r1m_population,
    build_r_tilde_population,
    compute_moments_quantiles,
    day_block_bootstrap,
    load_r_tilde,
    load_segmentation_cutoffs,
    load_tda04_inputs,
    qq_points,
    verify_r_tilde_invariants,
    verify_timestamp_alignment,
)

__all__ = [
    "PopulationMismatchError",
    "LookaheadLeakageError",
    "ROLLING_WINDOWS_MINUTES",
    "EWMA_HALFLIVES_MINUTES",
    "BURN_IN_HALFLIVES",
    "INPUT_SERIES",
    "PRIMARY_ESTIMATOR",
    "PRIMARY_ESTIMATOR_CLOCK_ADJUSTED",
    "ALL_ESTIMATOR_CONFIGS",
    "STABILITY_QUANTILE_LEVELS",
    "VOLATILITY_DECILES",
    "MIN_N_FOR_STABILITY_GROUP",
    "FRACTION_REMOVED_SCALE_THRESHOLD",
    "FRACTION_REMOVED_FORM_THRESHOLD",
    "PROFILE_STABILITY_SCALE_THRESHOLD",
    "PROFILE_STABILITY_FORM_THRESHOLD",
    "ROBUSTNESS_AGREEMENT_FRACTION",
    "CAUSALITY_CHECK_N_INDICES",
    "ewma_lambda_from_halflife",
    "causal_rolling_sigma",
    "causal_ewma_sigma",
    "burn_in_length",
    "build_sigma_hat",
    "standardize_return",
    "verify_no_lookahead_generic",
    "verify_no_lookahead",
    "run_causality_checks",
    "assign_volatility_decile",
    "compute_group_quantile_table",
    "extreme_scale_from_table",
    "profile_stability_ratio",
    "excess_kurtosis",
    "build_kurtosis_row",
    "build_kurtosis_table",
    "bootstrap_kurtosis_ci",
    "classify_config",
    "decide_verdict",
    "build_qq_table",
    "verify_populations_aligned",
    "TDA10Result",
    "run_tda10_analysis",
]


# ---------------------------------------------------------------------------
# Convenciones PREDECLARADAS (nunca elegidas despues de ver el resultado)
# ---------------------------------------------------------------------------

# Familia A -- desviacion estandar rodante causal. Ventanas en NUMERO DE
# BARRAS VALIDAS (no minutos de reloj -- ver docstring de
# ``causal_rolling_sigma``): 30 (media hora, sensible a cambios rapidos),
# 120 (2 horas, intermedia), 390 (6.5 horas -- duracion de una sesion
# regular NYSE/CME_Equity, "un dia de actividad de mercado", el mismo
# orden de magnitud que la ventana de energia TH21_ENERGY_M=240 de TDA-09
# pero elegida independientemente por su propio significado economico).
ROLLING_WINDOWS_MINUTES = (30, 120, 390)

# Familia B -- EWMA causal. Half-lives en NUMERO DE BARRAS VALIDAS: 20
# (corto), 60 (medio -- PRIMARIO, 1 hora), 240 (largo, 4 horas -- mismo
# orden que TH21_ENERGY_M de TDA-09, elegido por consistencia de escala
# temporal entre etapas, no copiado del valor).
EWMA_HALFLIVES_MINUTES = (20.0, 60.0, 240.0)

# Quemado de EWMA: se descartan las primeras `BURN_IN_HALFLIVES *
# halflife` filas de cada configuracion EWMA -- con 3 halflives, el peso
# remanente de la semilla inicial (sigma2=0 en t=0, ver
# `causal_ewma_sigma`) es 0.5**3=12.5%, un quemado conservador estandar
# para filtros exponenciales. La familia rodante no necesita esta
# constante: su propio `min_periods=window` ya produce NaN durante el
# quemado (no hay parametro de semilla que decaiga).
BURN_IN_HALFLIVES = 3

# Dos series de entrada -- "raw" (r_1m, causal) y "clock_adjusted"
# (r_tilde, RETROSPECTIVO por TDA-06) -- ver docstring de modulo, punto 2.
INPUT_SERIES = ("raw", "clock_adjusted")

# Estimador PRIMARIO -- EWMA half-life=60min sobre la serie RAW (la unica
# genuinamente causal de principio a fin). Elegido por: (a) EWMA es el
# estandar de facto para un estimador de volatilidad de un solo parametro
# (RiskMetrics); (b) 60 min es el punto medio de la grilla predeclarada,
# equilibrio entre reactividad (20min) y estabilidad (240min); (c) "raw"
# es el default causal -- la version "clock_adjusted" es la comparacion
# secundaria explicita para aislar el componente deterministico.
PRIMARY_ESTIMATOR: tuple[str, float, str] = ("ewma", 60.0, "raw")
# Contraparte "clock_adjusted" del primario -- misma familia/parametro,
# input distinto -- para la comparacion determinista-vs-dinamica.
PRIMARY_ESTIMATOR_CLOCK_ADJUSTED: tuple[str, float, str] = ("ewma", 60.0, "clock_adjusted")

ALL_ESTIMATOR_CONFIGS: tuple[tuple[str, float, str], ...] = tuple(
    (family, float(param), input_series)
    for input_series in INPUT_SERIES
    for family, params in (("rolling_std", ROLLING_WINDOWS_MINUTES), ("ewma", EWMA_HALFLIVES_MINUTES))
    for param in params
)  # 2 series x (3+3) configuraciones = 12 -- grilla PEQUEÑA y predeclarada (G4).

# Niveles de cuantil usados para las tablas de estabilidad de FORMA
# (metodo minimo 5 del roadmap) -- un subconjunto simetrico de
# `QUANTILE_LEVELS` (TDA-07), suficiente para medir la "escala extrema"
# de cada grupo sin repetir los 9 niveles completos de TDA-07/12.
STABILITY_QUANTILE_LEVELS: tuple[float, ...] = (0.01, 0.05, 0.95, 0.99)
assert set(STABILITY_QUANTILE_LEVELS).issubset(set(QUANTILE_LEVELS))

VOLATILITY_DECILES = 10
# Minimo de observaciones para que un grupo (decil/segmento/año) participe
# en la comparacion de estabilidad de forma -- misma filosofia que
# `MIN_N_FOR_GROUP=200` de TDA-09 (evita que un grupo con muestra minuscula
# domine el rango observado).
MIN_N_FOR_STABILITY_GROUP = 200

# --- Umbrales del veredicto (predeclarados, roadmap seccion "criterios de
# interpretacion" de TDA-10 + regla operativa G4/G5 de esta tarea: "si
# necesitas reglas operativas adicionales, definelas antes de mirar los
# resultados") ---------------------------------------------------------
#
# Fraccion de exceso de curtosis ELIMINADA por estandarizar (r->z),
# GLOBAL, version SIN recortar (la version recortada se reporta pero no
# entra en la regla de decision -- ver informe, seccion de interpretacion,
# para el porque: el recorte purga precisamente los eventos que la
# pregunta de "forma" quiere estudiar).
FRACTION_REMOVED_SCALE_THRESHOLD = 0.80  # >=80% eliminado: "cae drasticamente" (roadmap)
FRACTION_REMOVED_FORM_THRESHOLD = 0.50   # <=50% eliminado: "sobrevive una fraccion grande" (roadmap)

# Estabilidad del PERFIL de cuantiles estandarizados de z_t entre grupos
# (decil de volatilidad / segmento / año) -- ver `profile_stability_ratio`.
# Un ratio bajo = perfiles superpuestos ("solo escala"); un ratio alto =
# divergencia sistematica ("forma").
PROFILE_STABILITY_SCALE_THRESHOLD = 0.30
PROFILE_STABILITY_FORM_THRESHOLD = 0.60

# Robustez: fraccion minima de las 12 configuraciones de sensibilidad que
# deben coincidir en la misma etiqueta para que el veredicto HOLISTICO
# adopte esa etiqueta en vez de `MIXTO` (ver `decide_verdict`).
ROBUSTNESS_AGREEMENT_FRACTION = 0.75

CAUSALITY_CHECK_N_INDICES = 6
CAUSALITY_CHECK_SEED = 42
CAUSALITY_CHECK_EXTREME_VALUE = 1.0  # retorno "extremo" sintetico (100%) -- muy fuera del rango real (~1e-4..1e-2)
CAUSALITY_CHECK_LOOKAHEAD_WINDOW = 5  # cuantas posiciones DESPUES del indice perturbado se exige que cambien

KURTOSIS_BOOTSTRAP_SEED = 0

# Piso numerico de validez para `sigma_hat` (guardia de division protegida,
# ver `standardize_return`). El retorno logaritmico mas pequeño posible por
# construccion del tick de MNQ es del orden de `tick/precio` (~0.25/17000
# ~ 1.47e-5 sobre el conjunto de investigacion real, TDA-05). Un
# `sigma_hat` ordenes de magnitud por debajo de eso (`<1e-8`, mas de 3
# ordenes de magnitud) no puede representar volatilidad de mercado
# genuina -- solo puede surgir de una ventana de precios (casi) identicos
# donde el calculo de varianza en punto flotante produce un residual
# numerico en vez de un cero exacto. Verificado empiricamente: en el
# conjunto de investigacion real, exactamente 1 de 1.914.530 filas
# (``rolling_std`` ventana=30) cae por debajo de este piso, con
# `sigma_hat~6.4e-11` -- sin el piso, ese unico punto dispara `z~3*10^7` y
# arrastra la curtosis cruda de esa configuracion a >1.9 millones (un
# artefacto numerico, no evidencia de forma). Mismo principio que la
# guardia `sigma_hat>0` que ya exisitia -- este piso solo la hace
# numericamente robusta, no cambia la definicion.
MIN_VALID_SIGMA_HAT = 1e-8


class PopulationMismatchError(Exception):
    """``r1m_pop`` y ``r_tilde_pop`` no tienen el mismo numero de filas.

    TDA-09 (informe, seccion 2) documento que ambas poblaciones tienen
    exactamente el mismo ``n`` sobre el conjunto de investigacion real
    (misma serie canonica, r_tilde=NaN exactamente donde r_1m tambien lo
    es). Esta etapa depende de esa igualdad para poder comparar "raw" vs
    "clock_adjusted" bajo la misma nocion de poblacion -- si divergiera,
    seria una señal de que TDA-06/TDA-07 cambiaron de comportamiento sin
    que esta etapa se actualizara, y TDA-10 se detiene en vez de continuar
    silenciosamente con poblaciones no comparables.
    """


class LookaheadLeakageError(Exception):
    """Un estimador de volatilidad causal declarado NO paso la verificacion de ausencia de look-ahead (G1).

    Se dispara si, al perturbar el valor de ``r`` en una posicion ``idx``
    a un valor extremo, el ``sigma_hat`` en posiciones <= ``idx`` cambia
    (deberia depender EXCLUSIVAMENTE de ``r[0..idx-1]``). Bloqueante: si
    se dispara, TDA-10 se detiene sin producir ningun resultado de escala
    vs forma -- construir ``z_t`` con un estimador no causal invalidaria
    toda la etapa.
    """


# ---------------------------------------------------------------------------
# A. Desviacion estandar rodante causal
# ---------------------------------------------------------------------------

def causal_rolling_sigma(r: np.ndarray, window: int) -> np.ndarray:
    """``sigma_hat_{t-1}`` = desviacion estandar (ddof=1) de las `window` observaciones VALIDAS previas a `t`.

    Convencion (predeclarada, ver docstring de modulo): la ventana avanza
    sobre el NUMERO DE FILAS de la poblacion (que ya excluye, por TDA-04,
    cualquier retorno que cruce roll/jornada/hueco), no sobre minutos de
    reloj -- exactamente la misma convencion que la "ventana rodante"
    mensual de persistencia de TDA-08/09 usa para el eje temporal. Esto
    significa que una ventana puede abarcar mas de una jornada de
    negociacion (usa la volatilidad reciente REAL, no solo la de la
    sesion en curso) -- una eleccion metodologica explicita, no un
    descuido: la practica estandar de un estimador de volatilidad de
    "las ultimas N observaciones" no exige continuidad de reloj, solo
    continuidad de LA SERIE DE RETORNOS ya validados.

    ``pandas.Series.rolling(window, min_periods=window).std().shift(1)``:
    el ``shift(1)`` es lo que garantiza causalidad -- la fila `t` recibe
    la desviacion estandar calculada sobre `r[t-window:t]` (EXCLUYE `r[t]`
    por construccion, nunca la incluye).
    """
    if window < 2:
        raise ValueError(f"window debe ser >=2 para un desvio estandar bien definido, recibido {window}")
    s = pd.Series(np.asarray(r, dtype=float))
    return s.rolling(window=window, min_periods=window).std(ddof=1).shift(1).to_numpy()


# ---------------------------------------------------------------------------
# B. EWMA causal (recursivo, estilo RiskMetrics)
# ---------------------------------------------------------------------------

def ewma_lambda_from_halflife(halflife: float) -> float:
    """``lambda = 0.5**(1/halflife)`` -- el factor de decaimiento que hace que el peso caiga a la mitad cada `halflife` observaciones."""
    if halflife <= 0:
        raise ValueError(f"halflife debe ser > 0, recibido {halflife}")
    return float(np.exp(np.log(0.5) / halflife))


def causal_ewma_sigma(r: np.ndarray, halflife: float) -> np.ndarray:
    """``sigma_hat_{t-1}`` = raiz de la varianza EWMA recursiva ``sigma2_t = lambda*sigma2_{t-1} + (1-lambda)*r_{t-1}^2``.

    CAUSALIDAD (verificada, no solo asumida -- ver ``verify_no_lookahead``):
    ``sigma2_t`` depende UNICAMENTE de ``r[0..t-1]``, nunca de ``r[t]`` ni
    de ninguna observacion posterior. Se implementa desplazando `r^2` una
    posicion (``r2.shift(1)``, valor ``NaN`` en la posicion 0 -- no existe
    informacion previa a la primera observacion) y aplicando
    ``pandas.Series.ewm(halflife=halflife, adjust=False).mean()`` sobre la
    serie desplazada: con ``adjust=False``, ``ewm`` reproduce exactamente
    la recursion ``y_t = lambda*y_{t-1} + (1-lambda)*x_t`` con
    ``x_t = r2.shift(1)[t] = r[t-1]^2`` -- es decir, ``y_t = sigma2_t``
    tal como se define arriba. La SEMILLA (posicion donde `x` deja de ser
    ``NaN``, es decir ``t=1``) queda fijada por pandas al primer valor no
    nulo (``sigma2[1] = r[0]^2``), sin usar informacion futura en ningun
    punto de la recursion -- se documenta como "quemado" (no como fuga):
    las primeras ``BURN_IN_HALFLIVES*halflife`` filas se excluyen de todo
    analisis (ver ``standardize_return``) porque la varianza de la semilla
    es ruidosa, no porque el calculo sea incausal.

    Salida: ``sigma_hat`` (no `sigma2`), ``NaN`` en la posicion 0.
    """
    r = np.asarray(r, dtype=float)
    r2_shift = pd.Series(r).pow(2).shift(1)
    sigma2 = r2_shift.ewm(halflife=float(halflife), adjust=False).mean().to_numpy()
    with np.errstate(invalid="ignore"):
        return np.sqrt(sigma2)


# ---------------------------------------------------------------------------
# Dispatch generico + quemado + estandarizacion
# ---------------------------------------------------------------------------

def burn_in_length(family: str, param: float) -> int:
    """Numero de filas iniciales a excluir para la configuracion `(family, param)` -- ver constantes de modulo."""
    if family == "rolling_std":
        return int(param)
    if family == "ewma":
        return int(np.ceil(BURN_IN_HALFLIVES * param))
    raise ValueError(f"familia de estimador desconocida: {family!r}")


def build_sigma_hat(r: np.ndarray, family: str, param: float) -> np.ndarray:
    """Dispatcher: `sigma_hat` causal para `(family, param)` sobre la serie `r`."""
    if family == "rolling_std":
        return causal_rolling_sigma(r, int(param))
    if family == "ewma":
        return causal_ewma_sigma(r, float(param))
    raise ValueError(f"familia de estimador desconocida: {family!r}")


def standardize_return(r: np.ndarray, sigma_hat: np.ndarray, burn_in: int) -> np.ndarray:
    """``z_t = r_t / sigma_hat_{t-1}`` -- `NaN` donde `sigma_hat` no es finito/valido, o dentro del quemado.

    Division protegida: `sigma_hat <= MIN_VALID_SIGMA_HAT` (incluye
    negativos, cero, y positivos numericamente indistinguibles de cero) o
    no finito produce `NaN` explicito (nunca `inf` ni un `z` con
    magnitudes de millones de desviaciones estandar por dividir entre un
    residual de punto flotante) -- misma disciplina que
    `tda06_intraday_calendar_profile.build_r_tilde` y
    `tda09_volatility_clustering.build_log_hl_tilde`, reforzada con el
    piso numerico `MIN_VALID_SIGMA_HAT` (ver constante de modulo).
    """
    r = np.asarray(r, dtype=float)
    sigma_hat = np.asarray(sigma_hat, dtype=float)
    valid = np.isfinite(sigma_hat) & (sigma_hat > MIN_VALID_SIGMA_HAT) & np.isfinite(r)
    z = np.where(valid, r / np.where(sigma_hat <= 0, np.nan, sigma_hat), np.nan)
    if burn_in > 0:
        z = z.copy()
        z[: min(burn_in, z.size)] = np.nan
    return z


# ---------------------------------------------------------------------------
# Verificacion explicita de ausencia de look-ahead (G1)
# ---------------------------------------------------------------------------

def verify_no_lookahead_generic(
    estimator_fn, r: np.ndarray, perturb_indices: list[int],
    extreme_value: float = CAUSALITY_CHECK_EXTREME_VALUE, lookahead_window: int = CAUSALITY_CHECK_LOOKAHEAD_WINDOW,
) -> dict:
    """Prueba de reconstruccion (G1) generica: perturbar `r[idx]` NO debe cambiar `estimator_fn(r)` en ninguna posicion `<= idx`.

    `estimator_fn`: `np.ndarray -> np.ndarray` (mismo largo que `r`) --
    inyectado explicitamente para que esta funcion sea testeable con un
    estimador DELIBERADAMENTE no causal (p.ej. una ventana CENTRADA) y
    confirmar que la prueba efectivamente lo detecta, no solo que los
    estimadores de produccion la pasan.

    Para cada `idx` en `perturb_indices`: construye `r_perturbed` (copia de
    `r` con `r_perturbed[idx] = extreme_value`), recalcula `estimator_fn`
    y compara contra el resultado sobre la serie original en dos tramos:

    1. `[0, idx]` (inclusive): DEBE ser identico -- si difiere en
       cualquier posicion, es evidencia DIRECTA de fuga de informacion
       futura y la funcion lo reporta como fallo.
    2. `(idx, idx+lookahead_window]`: DEBE cambiar (si el estimador fuera
       insensible a `r[idx]` incluso alli, no seria un estimador de
       volatilidad reciente en absoluto -- un chequeo de sanidad adicional,
       no solo de causalidad).

    Salida: `dict` con `checks` (una fila por `idx`) y `passed` (True solo
    si TODOS los `idx` pasan el tramo 1 -- el tramo 2 se reporta pero no
    hace fallar la prueba por si solo, ya que en configuraciones EWMA de
    half-life muy largo el cambio en una ventana corta puede ser diminuto
    sin que eso implique fuga).
    """
    r = np.asarray(r, dtype=float)
    n = r.size
    base_sigma = np.asarray(estimator_fn(r), dtype=float)
    checks = []
    for idx in perturb_indices:
        if idx <= 0 or idx >= n - 1:
            continue
        r_perturbed = r.copy()
        r_perturbed[idx] = extreme_value
        pert_sigma = np.asarray(estimator_fn(r_perturbed), dtype=float)

        before = base_sigma[: idx + 1]
        before_pert = pert_sigma[: idx + 1]
        unaffected_before = bool(np.allclose(before, before_pert, equal_nan=True, rtol=1e-10, atol=1e-12))

        hi = min(idx + 1 + lookahead_window, n)
        after = base_sigma[idx + 1 : hi]
        after_pert = pert_sigma[idx + 1 : hi]
        changed_after = bool(after.size > 0 and not np.allclose(after, after_pert, equal_nan=True, rtol=1e-10, atol=1e-12))

        checks.append({
            "perturb_index": int(idx),
            "unaffected_up_to_and_including_index": unaffected_before,
            "changed_shortly_after": changed_after,
        })
    passed = all(c["unaffected_up_to_and_including_index"] for c in checks) if checks else True
    return {"checks": checks, "passed": bool(passed)}


def verify_no_lookahead(
    family: str, param: float, r: np.ndarray, perturb_indices: list[int],
    extreme_value: float = CAUSALITY_CHECK_EXTREME_VALUE, lookahead_window: int = CAUSALITY_CHECK_LOOKAHEAD_WINDOW,
) -> dict:
    """Igual que `verify_no_lookahead_generic`, pero para un `(family, param)` de produccion (via `build_sigma_hat`)."""
    result = verify_no_lookahead_generic(
        lambda rr: build_sigma_hat(rr, family, param), r, perturb_indices, extreme_value, lookahead_window,
    )
    return {"family": family, "param": float(param), **result}


def run_causality_checks(
    r_raw: np.ndarray, configs: tuple[tuple[str, float, str], ...] = ALL_ESTIMATOR_CONFIGS,
    n_indices: int = CAUSALITY_CHECK_N_INDICES, seed: int = CAUSALITY_CHECK_SEED,
) -> pd.DataFrame:
    """Ejecuta `verify_no_lookahead` para CADA configuracion de `configs` (familia/parametro, sin distinguir input_series -- la logica de causalidad es identica sea cual sea el input) sobre `r_raw`.

    Bloqueante: si CUALQUIER configuracion falla, lanza `LookaheadLeakageError`
    de inmediato -- TDA-10 nunca continua construyendo `z_t` con un
    estimador que no supero esta prueba.
    """
    r_raw = np.asarray(r_raw, dtype=float)
    n = r_raw.size
    lo, hi = min(2000, max(2, n // 10)), max(n - 2000, min(2, n))
    if hi <= lo:
        lo, hi = 1, max(2, n - 1)
    rng = np.random.default_rng(seed)
    idxs = sorted(set(int(i) for i in rng.integers(lo, hi, size=n_indices))) if hi > lo else []

    seen: dict[tuple[str, float], dict] = {}
    rows = []
    for family, param, _input_series in configs:
        key = (family, float(param))
        if key not in seen:
            result = verify_no_lookahead(family, param, r_raw, idxs)
            seen[key] = result
            if not result["passed"]:
                raise LookaheadLeakageError(
                    f"Estimador causal ({family}, param={param}) FALLO la prueba de reconstruccion (G1): "
                    f"sigma_hat cambio en una posicion <= idx al perturbar r[idx]. Detalle: {result['checks']}"
                )
        rows.append({
            "family": family, "param": float(param),
            "n_indices_checked": len(seen[key]["checks"]),
            "passed": seen[key]["passed"],
            "all_changed_shortly_after": all(c["changed_shortly_after"] for c in seen[key]["checks"]) if seen[key]["checks"] else None,
        })
    return pd.DataFrame(rows).drop_duplicates(subset=["family", "param"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Estabilidad de FORMA: deciles de volatilidad / segmento / año
# ---------------------------------------------------------------------------

def assign_volatility_decile(sigma_hat: np.ndarray, n_deciles: int = VOLATILITY_DECILES) -> np.ndarray:
    """Decil (0..n_deciles-1) de `sigma_hat` en el momento t -- `NaN` donde `sigma_hat` no es finito/positivo."""
    s = pd.Series(np.asarray(sigma_hat, dtype=float))
    valid = s.notna() & (s > 0)
    out = pd.Series(np.full(len(s), np.nan))
    if int(valid.sum()) >= n_deciles * 20:
        out.loc[valid] = pd.qcut(s.loc[valid], n_deciles, labels=False, duplicates="drop").astype(float)
    return out.to_numpy()


def compute_group_quantile_table(
    z: np.ndarray, group_labels: np.ndarray, quantile_levels: tuple[float, ...] = STABILITY_QUANTILE_LEVELS,
) -> pd.DataFrame:
    """Tabla `group, n, std_z, q{nivel}...` de `z` agrupado por `group_labels` -- filas `z`/`group` no finitos se excluyen antes de agrupar."""
    z = np.asarray(z, dtype=float)
    group_labels = np.asarray(group_labels)
    finite = np.isfinite(z)
    valid_group = pd.notna(pd.Series(group_labels))
    mask = finite & valid_group.to_numpy()
    df = pd.DataFrame({"group": group_labels[mask], "z": z[mask]})

    rows = []
    for g, sub in df.groupby("group", observed=True, sort=True):
        n = len(sub)
        row: dict = {"group": g, "n": n, "std_z": float(sub["z"].std(ddof=1)) if n > 1 else float("nan")}
        if n >= MIN_N_FOR_STABILITY_GROUP:
            qs = np.quantile(sub["z"].to_numpy(), quantile_levels)
            for lvl, q in zip(quantile_levels, qs):
                row[f"q{lvl}"] = float(q)
        else:
            for lvl in quantile_levels:
                row[f"q{lvl}"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def extreme_scale_from_table(table: pd.DataFrame) -> pd.Series:
    """``(q0.99 - q0.01)/2`` por fila -- una medida simetrica de "cuan extremo" es el grupo, en unidades de `z` (ya estandarizado)."""
    return (table["q0.99"] - table["q0.01"]).abs() / 2.0


def profile_stability_ratio(table: pd.DataFrame) -> float:
    """``(max-min)/mediana`` de `extreme_scale_from_table` entre los grupos con `n>=MIN_N_FOR_STABILITY_GROUP`.

    Bajo (cerca de 0): los grupos tienen una escala extrema similar tras
    estandarizar -- el perfil de cuantiles de `z_t` se superpone entre
    grupos ("solo escala", el ideal de `ESCALA DOMINA`). Alto: la escala
    extrema difiere sistematicamente entre grupos incluso despues de
    dividir por `sigma_hat` -- evidencia de FORMA (el estimador causal no
    esta capturando toda la heterocedasticidad, o hay estructura de forma
    genuina que varia por grupo). `NaN` si menos de 2 grupos tienen
    muestra suficiente.
    """
    valid = table.dropna(subset=["q0.01", "q0.99"])
    valid = valid.loc[valid["n"] >= MIN_N_FOR_STABILITY_GROUP]
    if len(valid) < 2:
        return float("nan")
    scale = extreme_scale_from_table(valid)
    med = float(scale.median())
    if not np.isfinite(med) or med == 0:
        return float("nan")
    return float((scale.max() - scale.min()) / med)


# ---------------------------------------------------------------------------
# Curtosis: antes/despues de estandarizar, con/sin recorte
# ---------------------------------------------------------------------------

def excess_kurtosis(x: np.ndarray) -> float:
    """Exceso de curtosis (momentos poblacionales, `m4/m2^2-3`) -- version LIGERA para el bootstrap (sin cuantiles)."""
    x = np.asarray(x, dtype=float)
    if x.size < 4:
        return float("nan")
    e = x - x.mean()
    m2 = float(np.mean(e**2))
    if m2 <= 0:
        return float("nan")
    m4 = float(np.mean(e**4))
    return float(m4 / m2**2 - 3.0)


def build_kurtosis_row(family: str, param: float, input_series: str, scope: str, scope_value, r_sub: np.ndarray, z_sub: np.ndarray) -> dict:
    """Una fila de la tabla central: momentos completos (`compute_moments_quantiles`, TDA-07) de `r` y `z` restringidos a la MISMA poblacion, mas la fraccion eliminada (version completa y recortada 0.1%)."""
    mr = compute_moments_quantiles(r_sub)
    mz = compute_moments_quantiles(z_sub)

    def _fraction(k_r: float, k_z: float) -> float:
        if not np.isfinite(k_r) or k_r <= 0:
            return float("nan")
        return float(1.0 - k_z / k_r)

    return {
        "family": family, "param": param, "input_series": input_series, "scope": scope, "scope_value": scope_value,
        "n": mr["n"],
        "kurt_r": mr["kurtosis_excess"], "kurt_z": mz["kurtosis_excess"],
        "fraction_removed": _fraction(mr["kurtosis_excess"], mz["kurtosis_excess"]),
        "kurt_r_trimmed": mr["kurtosis_excess_trimmed"], "kurt_z_trimmed": mz["kurtosis_excess_trimmed"],
        "fraction_removed_trimmed": _fraction(mr["kurtosis_excess_trimmed"], mz["kurtosis_excess_trimmed"]),
    }


def build_kurtosis_table(
    family: str, param: float, input_series: str,
    r: np.ndarray, z: np.ndarray, year_ny: np.ndarray,
) -> pd.DataFrame:
    """Tabla GLOBAL + por año para una configuracion -- `r` y `z` restringidos a las filas donde `z` es finito (misma poblacion en ambas columnas, comparacion apples-to-apples)."""
    mask = np.isfinite(np.asarray(z, dtype=float))
    r_valid = np.asarray(r, dtype=float)[mask]
    z_valid = np.asarray(z, dtype=float)[mask]
    year_valid = np.asarray(year_ny)[mask]

    rows = [build_kurtosis_row(family, param, input_series, "GLOBAL", "GLOBAL", r_valid, z_valid)]
    for year in sorted(pd.unique(year_valid)):
        ymask = year_valid == year
        rows.append(build_kurtosis_row(family, param, input_series, "YEAR", int(year), r_valid[ymask], z_valid[ymask]))
    return pd.DataFrame(rows)


def bootstrap_kurtosis_ci(
    r: np.ndarray, z: np.ndarray, trading_date: np.ndarray, n_boot: int, seed: int = KURTOSIS_BOOTSTRAP_SEED,
) -> dict:
    """IC 95% por bootstrap de BLOQUES de jornada (G5) para `kurt_r`, `kurt_z` y `fraction_removed`, GLOBAL, en la poblacion restringida a `z` finito.

    Reutiliza `day_block_bootstrap` (TDA-07) DOS veces -- una para `r`, una
    para `z` -- con la MISMA `trading_date` y la MISMA semilla: como
    `day_block_bootstrap` resamplea los dias con un generador determinista
    inicializado igual en ambas llamadas, la replica `b` de `r` y la
    replica `b` de `z` corresponden EXACTAMENTE al mismo conjunto de dias
    remuestreados -- permite calcular `fraction_removed` por replica
    (`1 - kurt_z_b/kurt_r_b`) sin escribir un motor de bootstrap nuevo.
    """
    mask = np.isfinite(np.asarray(z, dtype=float))
    r_valid = np.asarray(r, dtype=float)[mask]
    z_valid = np.asarray(z, dtype=float)[mask]
    date_valid = np.asarray(trading_date)[mask]

    kurt_r_boot = day_block_bootstrap(r_valid, date_valid, excess_kurtosis, n_boot, seed)
    kurt_z_boot = day_block_bootstrap(z_valid, date_valid, excess_kurtosis, n_boot, seed)

    with np.errstate(invalid="ignore", divide="ignore"):
        frac_boot = 1.0 - kurt_z_boot / kurt_r_boot
    frac_boot = np.where((kurt_r_boot > 0) & np.isfinite(kurt_r_boot) & np.isfinite(kurt_z_boot), frac_boot, np.nan)

    def _ci(arr: np.ndarray) -> tuple[float, float]:
        finite = arr[np.isfinite(arr)]
        if finite.size < 10:
            return float("nan"), float("nan")
        return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))

    kr_lo, kr_hi = _ci(kurt_r_boot)
    kz_lo, kz_hi = _ci(kurt_z_boot)
    fr_lo, fr_hi = _ci(frac_boot)
    return {
        "n_boot": n_boot,
        "kurt_r_point": excess_kurtosis(r_valid), "kurt_r_ci_lo": kr_lo, "kurt_r_ci_hi": kr_hi,
        "kurt_z_point": excess_kurtosis(z_valid), "kurt_z_ci_lo": kz_lo, "kurt_z_ci_hi": kz_hi,
        "fraction_removed_ci_lo": fr_lo, "fraction_removed_ci_hi": fr_hi,
    }


# ---------------------------------------------------------------------------
# Veredicto: clasificacion por configuracion + decision holistica
# ---------------------------------------------------------------------------

def classify_config(fraction_removed_trimmed: float, max_profile_stability_ratio: float) -> str:
    """Etiqueta UNA configuracion (`ESCALA_DOMINA`/`FORMA_SUSTANCIAL`/`MIXTO`) segun las reglas predeclaradas (ver constantes de modulo).

    `fraction_removed_trimmed`: la fraccion de curtosis eliminada usando la
    version RECORTADA (0.1%, misma convencion que TDA-07), NUNCA la
    version completa/cruda -- decision tomada por la recomendacion
    EXPLICITA de TDA-07 (informe, seccion 12: "la curtosis recortada...
    es la cifra mas estable disponible... para juzgar cuanta de la
    no-normalidad es genuina"), reconfirmada empiricamente en esta etapa:
    la curtosis SIN recortar de `r`/`z` es un momento de cuarto orden
    -- unos pocos puntos extremos (incluidos artefactos numericos de un
    `sigma_hat` casi nulo en ventanas cortas, ver `MIN_VALID_SIGMA_HAT`)
    pueden dominarla por completo (ver TDA-07 informe, TH08: 16x de
    diferencia por 3.519 filas de 1.9 millones). La version completa se
    reporta igualmente (tabla central, roadmap) mostrando exactamente esa
    fragilidad -- pero NUNCA decide el veredicto.

    `max_profile_stability_ratio`: el MAYOR de los `profile_stability_ratio`
    calculados sobre decil de volatilidad, segmento horario y año -- un
    perfil solo cuenta como "estable" si lo es en las TRES dimensiones
    (roadmap: "compara...entre segmentos horarios...entre deciles...entre
    años" -- ninguna dimension se ignora).
    """
    if not np.isfinite(fraction_removed_trimmed) or not np.isfinite(max_profile_stability_ratio):
        return "MIXTO"
    scale_like = fraction_removed_trimmed >= FRACTION_REMOVED_SCALE_THRESHOLD and max_profile_stability_ratio <= PROFILE_STABILITY_SCALE_THRESHOLD
    form_like = fraction_removed_trimmed <= FRACTION_REMOVED_FORM_THRESHOLD or max_profile_stability_ratio >= PROFILE_STABILITY_FORM_THRESHOLD
    if scale_like and not form_like:
        return "ESCALA_DOMINA"
    if form_like and not scale_like:
        return "FORMA_SUSTANCIAL"
    return "MIXTO"


def decide_verdict(config_labels: list[str]) -> dict:
    """Veredicto HOLISTICO (TH22): la etiqueta mayoritaria si alcanza `ROBUSTNESS_AGREEMENT_FRACTION` de las configuraciones evaluadas; `MIXTO` en cualquier otro caso (incluida la propia mayoria si esta es `MIXTO`)."""
    n = len(config_labels)
    if n == 0:
        return {"verdict": "MIXTO", "agreement_fraction": float("nan"), "counts": {}, "robust": False, "n_configs": 0}
    counts: dict[str, int] = {}
    for lbl in config_labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    top_label = max(counts, key=lambda k: counts[k])
    agreement = counts[top_label] / n
    robust = agreement >= ROBUSTNESS_AGREEMENT_FRACTION
    verdict = top_label if (robust and top_label != "MIXTO") else "MIXTO"
    return {"verdict": verdict, "agreement_fraction": float(agreement), "counts": counts, "robust": bool(robust), "n_configs": n}


# ---------------------------------------------------------------------------
# QQ
# ---------------------------------------------------------------------------

def build_qq_table(r_raw: np.ndarray, z_by_config: dict[tuple[str, float, str], np.ndarray]) -> pd.DataFrame:
    """Puntos QQ (`qq_points`, TDA-07 -- sin dependencia de scipy) para `r_1m` crudo y cada `z` de `z_by_config`."""
    rows = []
    r_finite = np.asarray(r_raw, dtype=float)
    r_finite = r_finite[np.isfinite(r_finite)]
    if r_finite.size >= 100:
        theo, emp = qq_points(r_finite)
        for th, em in zip(theo, emp):
            rows.append({"series": "r_1m_raw", "config": "NA", "theoretical_q": th, "empirical_q": em})

    for (family, param, input_series), z in z_by_config.items():
        z_finite = np.asarray(z, dtype=float)
        z_finite = z_finite[np.isfinite(z_finite)]
        if z_finite.size < 100:
            continue
        theo, emp = qq_points(z_finite)
        label = f"{family}_{param:g}_{input_series}"
        for th, em in zip(theo, emp):
            rows.append({"series": "z", "config": label, "theoretical_q": th, "empirical_q": em})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Invariante: r1m_pop y r_tilde_pop deben tener el mismo n
# ---------------------------------------------------------------------------

def verify_populations_aligned(r1m_pop: pd.DataFrame, r_tilde_pop: pd.DataFrame) -> None:
    if len(r1m_pop) != len(r_tilde_pop):
        raise PopulationMismatchError(
            f"r1m_pop tiene {len(r1m_pop)} filas y r_tilde_pop tiene {len(r_tilde_pop)} -- "
            "TDA-09 (informe, seccion 2) establecio que ambas poblaciones deben coincidir en n "
            "sobre el conjunto de investigacion real. TDA-10 se detiene sin construir ninguna "
            "configuracion 'clock_adjusted'."
        )


# ---------------------------------------------------------------------------
# Progreso visible -- mismo patron que TDA-09 (_StageTimer)
# ---------------------------------------------------------------------------

class _StageTimer:
    def __init__(self, total_stages: int, verbose: bool = True):
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
                print(f"[TDA10 {self._i}/{self.total}] {self._current_name} -- completado en {elapsed_stage:.1f}s (total: {total_elapsed:.1f}s)")
        self._i += 1
        self._current_name = name
        self._last = now
        if self.verbose:
            print(f"[TDA10 {self._i}/{self.total}] {name}...")

    def finish(self) -> dict[str, float]:
        now = time.perf_counter()
        elapsed_stage = now - self._last
        self.stage_timings[self._current_name] = elapsed_stage
        if self.verbose:
            total_elapsed = now - self.t_start
            print(f"[TDA10 {self._i}/{self.total}] {self._current_name} -- completado en {elapsed_stage:.1f}s (total: {total_elapsed:.1f}s)")
        self.stage_timings["_total_analysis"] = now - self.t_start
        return self.stage_timings


ANALYSIS_STAGES = 7
TOTAL_STAGES = 8  # 7 de analisis + 1 de persistencia de artefactos (run_tda10.py)


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------

@dataclass
class TDA10Result:
    r1m_population: pd.DataFrame
    r_tilde_population: pd.DataFrame
    segment_labels: list[str]

    causality_table: pd.DataFrame
    kurtosis_table: pd.DataFrame
    kurtosis_bootstrap_ci: dict
    quantile_by_decile: pd.DataFrame   # solo configs "headline" (primario raw + clock_adjusted)
    quantile_by_segment: pd.DataFrame
    quantile_by_year: pd.DataFrame
    sensitivity_table: pd.DataFrame    # una fila por cada una de las 12 configuraciones
    qq_table: pd.DataFrame

    th22_verdict: dict = field(default_factory=dict)
    th26_status: str = "PARCIALMENTE_INFORMADA"
    stop13_suggestion: dict = field(default_factory=dict)
    stage_timings: dict[str, float] = field(default_factory=dict)


def _sigma_and_z_for_series(r: np.ndarray, family: str, param: float) -> tuple[np.ndarray, np.ndarray]:
    sigma_hat = build_sigma_hat(r, family, param)
    burn_in = burn_in_length(family, param)
    z = standardize_return(r, sigma_hat, burn_in)
    return sigma_hat, z


def run_tda10_analysis(config: SnapshotConfig, verbose: bool = True) -> TDA10Result:
    """Orquesta TDA-10 completo: invariantes -> verificacion de causalidad (G1) -> construccion de sigma_hat/z_t para las 12 configuraciones -> curtosis (global+año, completa+recortada) -> bootstrap CI (primario) -> estabilidad de forma (decil/segmento/año) -> sensibilidad -> veredicto TH22."""
    timer = _StageTimer(ANALYSIS_STAGES, verbose=verbose)

    # --- Etapa 1: carga de datos y poblaciones ------------------------------
    timer.stage("Carga de datos, poblaciones (r_1m/r_tilde) e invariantes")
    validate_research_holdout_disjoint(config)

    variables, validity = load_tda04_inputs(config)
    last_timestamps = variables.groupby("source_file")["timestamp"].max().to_dict()
    validate_last_timestamps_before_boundary(config, last_timestamps)
    verify_timestamp_alignment(variables, validity)

    cutoffs = load_segmentation_cutoffs(config)
    r1m_pop, segment_labels = build_r1m_population(variables, validity, cutoffs)
    r1m_pop = r1m_pop.sort_values("timestamp").reset_index(drop=True)

    r_tilde = load_r_tilde(config)
    verify_r_tilde_invariants(r_tilde, variables)
    r_tilde_pop, _ = build_r_tilde_population(r_tilde, cutoffs)
    r_tilde_pop = r_tilde_pop.sort_values("timestamp").reset_index(drop=True)

    verify_populations_aligned(r1m_pop, r_tilde_pop)

    r_raw = r1m_pop["r_1m"].to_numpy(dtype=float)
    r_adj = r_tilde_pop["r_tilde"].to_numpy(dtype=float)
    pop_by_series = {"raw": r1m_pop, "clock_adjusted": r_tilde_pop}
    r_by_series = {"raw": r_raw, "clock_adjusted": r_adj}

    # --- Etapa 2: verificacion explicita de causalidad (G1), bloqueante ----
    timer.stage("Verificacion de ausencia de look-ahead (G1) para las 12 configuraciones")
    causality_table = run_causality_checks(r_raw, ALL_ESTIMATOR_CONFIGS)

    # --- Etapa 3: sigma_hat/z_t para las 12 configuraciones -----------------
    timer.stage("Construccion causal de sigma_hat y z_t (12 configuraciones)")
    sigma_by_config: dict[tuple[str, float, str], np.ndarray] = {}
    z_by_config: dict[tuple[str, float, str], np.ndarray] = {}
    for family, param, input_series in ALL_ESTIMATOR_CONFIGS:
        r_input = r_by_series[input_series]
        sigma_hat, z = _sigma_and_z_for_series(r_input, family, param)
        sigma_by_config[(family, param, input_series)] = sigma_hat
        z_by_config[(family, param, input_series)] = z

    # --- Etapa 4: tabla central de curtosis (global + por año) -------------
    timer.stage("Tabla central de curtosis (global + por año, completa + recortada) para las 12 configuraciones")
    kurtosis_rows = []
    for (family, param, input_series), z in z_by_config.items():
        pop = pop_by_series[input_series]
        r_input = r_by_series[input_series]
        tab = build_kurtosis_table(family, param, input_series, r_input, z, pop["year_ny"].to_numpy())
        kurtosis_rows.append(tab)
    kurtosis_table = pd.concat(kurtosis_rows, ignore_index=True)

    # --- Etapa 5: bootstrap CI (primario GLOBAL) ----------------------------
    timer.stage("Bootstrap de bloques por jornada (IC 95%) para la configuracion primaria")
    prim_family, prim_param, prim_input = PRIMARY_ESTIMATOR
    prim_pop = pop_by_series[prim_input]
    prim_r = r_by_series[prim_input]
    prim_z = z_by_config[PRIMARY_ESTIMATOR]
    kurtosis_bootstrap_ci = bootstrap_kurtosis_ci(
        prim_r, prim_z, prim_pop["trading_date"].to_numpy(), n_boot=config.tda10_n_boot,
    )

    # --- Etapa 6: estabilidad de FORMA (decil / segmento / año) ------------
    timer.stage("Estabilidad de forma por decil de volatilidad, segmento y año (headline + sensibilidad)")
    headline_configs = (PRIMARY_ESTIMATOR, PRIMARY_ESTIMATOR_CLOCK_ADJUSTED)
    quantile_decile_rows, quantile_segment_rows, quantile_year_rows = [], [], []
    for key in headline_configs:
        family, param, input_series = key
        pop = pop_by_series[input_series]
        z = z_by_config[key]
        sigma_hat = sigma_by_config[key]

        decile = assign_volatility_decile(sigma_hat)
        d_tab = compute_group_quantile_table(z, decile)
        d_tab.insert(0, "config", f"{family}_{param:g}_{input_series}")
        quantile_decile_rows.append(d_tab)

        s_tab = compute_group_quantile_table(z, pop["segment_label"].to_numpy())
        s_tab.insert(0, "config", f"{family}_{param:g}_{input_series}")
        quantile_segment_rows.append(s_tab)

        y_tab = compute_group_quantile_table(z, pop["year_ny"].to_numpy())
        y_tab.insert(0, "config", f"{family}_{param:g}_{input_series}")
        quantile_year_rows.append(y_tab)

    quantile_by_decile = pd.concat(quantile_decile_rows, ignore_index=True)
    quantile_by_segment = pd.concat(quantile_segment_rows, ignore_index=True)
    quantile_by_year = pd.concat(quantile_year_rows, ignore_index=True)

    # Sensibilidad: ratio de estabilidad (decil/segmento/año) para las 12 --
    sensitivity_rows = []
    kurt_global = kurtosis_table.loc[kurtosis_table["scope"] == "GLOBAL"].set_index(["family", "param", "input_series"])
    for (family, param, input_series), z in z_by_config.items():
        pop = pop_by_series[input_series]
        sigma_hat = sigma_by_config[(family, param, input_series)]
        decile = assign_volatility_decile(sigma_hat)

        decile_ratio = profile_stability_ratio(compute_group_quantile_table(z, decile))
        segment_ratio = profile_stability_ratio(compute_group_quantile_table(z, pop["segment_label"].to_numpy()))
        year_ratio = profile_stability_ratio(compute_group_quantile_table(z, pop["year_ny"].to_numpy()))
        max_ratio = np.nanmax([decile_ratio, segment_ratio, year_ratio]) if np.isfinite([decile_ratio, segment_ratio, year_ratio]).any() else float("nan")

        row_key = (family, float(param), input_series)
        if row_key in kurt_global.index:
            fraction_removed_full = float(kurt_global.loc[row_key, "fraction_removed"])
            fraction_removed_trimmed = float(kurt_global.loc[row_key, "fraction_removed_trimmed"])
        else:
            fraction_removed_full = float("nan")
            fraction_removed_trimmed = float("nan")
        label = classify_config(fraction_removed_trimmed, max_ratio)

        sensitivity_rows.append({
            "family": family, "param": param, "input_series": input_series,
            "fraction_removed_full": fraction_removed_full, "fraction_removed_trimmed": fraction_removed_trimmed,
            "decile_stability_ratio": decile_ratio, "segment_stability_ratio": segment_ratio, "year_stability_ratio": year_ratio,
            "max_stability_ratio": max_ratio, "config_label": label,
        })
    sensitivity_table = pd.DataFrame(sensitivity_rows)

    # --- Etapa 7: QQ + veredicto TH22 + STOP-13 (sugerencia) ----------------
    timer.stage("QQ-plots, veredicto TH22 y sugerencia de STOP-13")
    qq_table = build_qq_table(r_raw, z_by_config)

    th22_verdict = decide_verdict(sensitivity_table["config_label"].tolist())
    th22_verdict["primary_config"] = f"{prim_family}_{prim_param:g}_{prim_input}"
    th22_verdict["primary_fraction_removed_full"] = float(
        kurt_global.loc[(prim_family, float(prim_param), prim_input), "fraction_removed"]
    )
    th22_verdict["primary_fraction_removed_trimmed"] = float(
        kurt_global.loc[(prim_family, float(prim_param), prim_input), "fraction_removed_trimmed"]
    )

    if th22_verdict["verdict"] == "ESCALA_DOMINA" and th22_verdict["robust"]:
        stop13_suggestion = {
            "suggested": True,
            "reason": (
                "TDA-10 encontro ESCALA_DOMINA de forma robusta (agreement="
                f"{th22_verdict['agreement_fraction']:.2f} de las {th22_verdict['n_configs']} configuraciones). "
                "Esto SUGIERE que STOP-13 (no ejecutar EVT) sera probable, pero la activacion FORMAL de STOP-13 "
                "corresponde a TDA-12 (roadmap, criterios de interpretacion de TDA-12), no a esta etapa."
            ),
        }
    else:
        stop13_suggestion = {
            "suggested": False,
            "reason": (
                f"TDA-10 no encontro ESCALA_DOMINA robusta (veredicto={th22_verdict['verdict']}, "
                f"agreement={th22_verdict['agreement_fraction']:.2f}). No se sugiere STOP-13; "
                "TDA-12 debe evaluar formalmente si procede EVT."
            ),
        }

    stage_timings = timer.finish()

    return TDA10Result(
        r1m_population=r1m_pop, r_tilde_population=r_tilde_pop, segment_labels=segment_labels,
        causality_table=causality_table, kurtosis_table=kurtosis_table, kurtosis_bootstrap_ci=kurtosis_bootstrap_ci,
        quantile_by_decile=quantile_by_decile, quantile_by_segment=quantile_by_segment, quantile_by_year=quantile_by_year,
        sensitivity_table=sensitivity_table, qq_table=qq_table,
        th22_verdict=th22_verdict, th26_status="PARCIALMENTE_INFORMADA", stop13_suggestion=stop13_suggestion,
        stage_timings=stage_timings,
    )
