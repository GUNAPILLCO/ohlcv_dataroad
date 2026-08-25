"""TDA-11 -- Modelo parametrico de volatilidad *(CONDICIONAL)*.

Implementa la etapa TDA-11 del roadmap
(``docs/methodology/Tsay_OHLCV_analysis_roadmap.md``, seccion "TDA-11"):
responde si un GARCH(1,1) -- el modelo parametrico MAS SIMPLE posible --
resume la persistencia genuina de volatilidad que TDA-09 encontro
(``CLUSTERING_GENUINO``, `STOP-9` NO activado) en parametros
interpretables y razonablemente estables, y si elimina dependencia
residual que los benchmarks causales simples de TDA-10 (EWMA, rodante) y
un benchmark nuevo basado en rango NO eliminan.

PREGUNTA ESCRITA que justifica esta etapa (declarada ANTES de construir
ningun benchmark ni ajustar ningun modelo -- G2/G4):

    ¿Puede el modelo parametrico de volatilidad mas simple resumir la
    persistencia genuina encontrada en TDA-09 mediante parametros
    interpretables y razonablemente estables entre subperiodos, y
    eliminar dependencia residual de volatilidad que los benchmarks
    simples no eliminan?

Preguntas secundarias: (1) valor y estabilidad de `alpha+beta` entre
años y segmentos; (2) ¿deja GARCH(1,1) menos dependencia en los residuos
estandarizados que EWMA/rodante/rango?; (3) ¿existe asimetria
descriptiva estable respecto del signo del shock previo que justifique
considerar un modelo asimetrico?

Esto sigue siendo CARACTERIZACION ESTADISTICA -- NO evalua capacidad
predictiva OOS, rentabilidad, señales, targets ni features. NO abre
``data/raw/`` ni ``config.holdout_files``.

---------------------------------------------------------------------
PUERTA DE ENTRADA (verificada, no asumida -- ver `verify_entry_gate`)
---------------------------------------------------------------------

TDA-11 es CONDICIONAL. Se ejecuta SOLO si, verificado sobre los
artefactos REALES persistidos por TDA-09 (nunca sobre texto de un
informe, siempre recomputando con el MISMO codigo que TDA-09 uso):

1. TH21 (``classify_th21``, recomputado desde
   ``TDA09_clock_attribution.csv``) = `CLUSTERING_GENUINO`.
2. `STOP-9` (``decide_stop9``, recomputado desde el mismo CSV) NO
   activado.
3. La pregunta escrita de arriba existe y pertenece a caracterizacion,
   no a prediccion (verificado por inspeccion: ninguna funcion de este
   modulo calcula error de prediccion, capacidad OOS ni metricas
   economicas).

Si la puerta NO se abre, `STOP-11` se activa de inmediato y NINGUN
benchmark ni modelo se construye.

Si la puerta se abre pero, tras construir los benchmarks simples, la
pregunta ya queda respondida sin necesidad de GARCH, tambien se activa
`STOP-11` (ver `decide_garch_usefulness`) -- **nunca se ajusta GARCH por
costumbre** (G4).

---------------------------------------------------------------------
CONVENCIONES PREDECLARADAS (fijadas antes de ejecutar sobre el
conjunto de investigacion real -- misma disciplina que TDA-06..10)
---------------------------------------------------------------------

1. **Serie principal**: `r_tilde` (TDA-06, clock-adjusted, RETROSPECTIVO
   -- `s(m)` estimado con TODA la muestra). El roadmap exige controlar
   la estacionalidad intradia ANTES de ajustar el modelo parametrico
   (metodo minimo 3 de TDA-11) -- por eso `r_tilde`, no `r_1m`, es la
   serie que responde la pregunta FORMAL (TH23/TH24/TH25).
2. **Serie secundaria**: `r_1m` (RAW/CAUSAL, TDA-10) se usa unicamente
   como sensibilidad -- alcance GLOBAL, sin bateria completa de
   subperiodos (G4: la pregunta formal ya la responde la serie
   principal; repetir la bateria completa en la serie secundaria no
   aportaria informacion nueva sobre la pregunta declarada). Nunca se
   mezcla ni se promedia con la serie principal (auditoria de TDA-10,
   problema 3, aplicada aqui de nuevo).
3. **GARCH es RETROSPECTIVO EN SI MISMO, sea cual sea la serie de
   entrada.** Esta es una distincion ADICIONAL a la de "raw vs
   clock_adjusted": los parametros `(omega, alpha, beta)` se estiman
   por maxima verosimilitud usando TODA la muestra (igual que `s(m)` en
   TDA-06) -- incluso el GARCH ajustado sobre `r_1m` (serie de entrada
   CAUSAL) es, como MODELO, RETROSPECTIVO, porque sus parametros no
   estarian disponibles en el instante `t` de ninguna observacion
   temprana sin haber visto datos futuros. Nunca se presenta ningun
   resultado de GARCH como un estimador causal disponible en
   produccion.
4. **Topologia temporal de la propia recursion GARCH/benchmarks**:
   igual convencion que TDA-10 -- el filtro (EWMA/rodante/rango/GARCH)
   avanza sobre el NUMERO DE FILAS de la poblacion (que ya excluye,
   por TDA-04, cualquier retorno que cruce roll/jornada/hueco), no
   sobre minutos de reloj. Un ajuste "por segmento" concatena las
   barras de ESE segmento horario a traves de los dias en orden
   cronologico -- el modelo puede tratar como "adyacentes" (t-1,t) dos
   barras que en el reloj real estan separadas por muchas horas (el
   cierre del segmento de un dia y la apertura del mismo segmento al
   dia siguiente). Es la MISMA simplificacion, ya predeclarada y
   documentada, que TDA-10 adopto para sus propios estimadores
   causales -- no una decision nueva de esta etapa. El DIAGNOSTICO de
   residuos (ACF/portmanteau), en cambio, SI respeta bloques de
   continuidad genuina (`compute_block_ids_with_contract`, TDA-09) para
   no fabricar dependencia artificial en el propio diagnostico.
5. **Escala numerica del ajuste GARCH** (`GARCH_SCALE_FACTOR=1e4`):
   verificado empiricamente que `arch_model` sobre retornos sin escalar
   (`std~4e-4`) dispara `DataScaleWarning`/`ConvergenceWarning` (el
   optimizador no converge de forma fiable); escalando por `1e4` el
   `std` escalado cae en el rango recomendado por la libreria (~1-10).
   `alpha`/`beta`/persistencia son invariantes a esta escala; `omega`
   se des-escala dividiendo por `GARCH_SCALE_FACTOR**2` antes de
   reportarlo.
6. **Distribucion de innovaciones**: Gaussian QMLE como especificacion
   PRIMARIA (la mas simple, roadmap G4), con inferencia ROBUSTA
   (`cov_type="robust"`, sandwich de Bollerslev-Wooldridge -- corrige
   los errores estandar cuando la verdadera distribucion no es
   gaussiana, sin cambiar la estimacion puntual). Una innovacion
   Student-t se ajusta UNICAMENTE como sensibilidad (alcance GLOBAL,
   serie principal) -- justificada por TDA-10, que encontro que `z_t`
   estandarizado por estimadores causales simples sigue teniendo
   curtosis material -- nunca como busqueda de la distribucion que
   mejor ajusta (G4).
7. **Umbrales de decision** (nunca ajustados despues de ver el
   resultado real -- ver constantes `*_THRESHOLD` mas abajo): utilidad
   informativa de GARCH frente a benchmarks, materialidad de la
   asimetria, tamaño minimo de muestra para un ajuste de subperiodo.
8. **Multi-arranque del optimizador GARCH** (`GARCH_OPTIMIZER_EPS_GRID`,
   ver `fit_garch11`): descubierto DESPUES de la primera ejecucion sobre
   el conjunto de investigacion real -- con `n~10^6`, `scipy.optimize.minimize`
   (SLSQP, interno a `arch`) puede reportar convergencia exitosa sin
   haberse movido del punto de partida (verificado: `res.params`
   identico a `GARCH.starting_values`, gradiente de cientos de miles en
   el supuesto optimo). Se corrige probando una grilla PEQUEÑA y
   predeclarada del paso de diferencias finitas (`eps`) y quedandose con
   el intento de MENOR log-verosimilitud negativa -- un criterio
   OBJETIVO (la propia funcion que el MLE maximiza), nunca una eleccion
   subjetiva ni un ajuste de umbral para cambiar una conclusion. Se
   documenta con la misma transparencia que la auditoria de TDA-10
   (`MIN_VALID_SIGMA_HAT`): es una correccion de VALIDEZ NUMERICA del
   ajuste, no una decision metodologica sobre que responde la pregunta.

Que NO hace este modulo (fuera de alcance, deliberado):

- No entrena modelos de ML, no crea targets/features/señales.
- No evalua capacidad predictiva OOS ni backtest ni rentabilidad.
- No prueba una coleccion de familias GARCH -- solo GARCH(1,1), y como
  MAXIMO una extension asimetrica (GJR-GARCH(1,1,1)) si TH25 la
  justifica.
- No afirma IGARCH, memoria larga ni "leverage effect" por analogia con
  acciones -- documenta la ambiguedad explicitamente cuando aplica.
- No abre ``data/raw/`` ni ``config.holdout_files``.
- No modifica ningun artefacto de TDA-00..TDA-10.
- No inicia TDA-12.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import arch
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import chi2

from ohlcv_dataroad.config import SnapshotConfig
from ohlcv_dataroad.ingest.holdout_guard import (
    validate_last_timestamps_before_boundary,
    validate_research_holdout_disjoint,
)
from ohlcv_dataroad.ingest.tda07_marginal_distribution import (
    build_r1m_population,
    build_r_tilde_population,
    day_block_bootstrap,
    load_r_tilde,
    load_segmentation_cutoffs,
    load_tda04_inputs,
    verify_r_tilde_invariants,
    verify_timestamp_alignment,
)
from ohlcv_dataroad.ingest.tda08_linear_mean_dependence import compute_acf, compute_portmanteau_q
from ohlcv_dataroad.ingest.tda09_volatility_clustering import (
    classify_th21,
    compute_block_ids_with_contract,
    decide_stop9,
)
from ohlcv_dataroad.ingest.tda10_scale_vs_shape import (
    burn_in_length,
    build_sigma_hat,
    causal_ewma_sigma,
    standardize_return,
    verify_populations_aligned,
)

__all__ = [
    "WRITTEN_QUESTION",
    "ARCH_VERSION",
    "GARCH_SCALE_FACTOR",
    "GARCH_OPTIMIZER_EPS_GRID",
    "GARCH_OPTIMIZER_MAXITER",
    "MIN_N_FOR_GARCH_FIT",
    "RANGE_BENCHMARK_HALFLIFE",
    "ROLLING_BENCHMARK_WINDOW",
    "BENCHMARK_CONFIGS",
    "DIAGNOSTIC_LAGS",
    "PORTMANTEAU_M",
    "RESIDUAL_COMPARISON_LAG",
    "RESIDUAL_COMPARISON_M",
    "GARCH_RESIDUAL_REDUCTION_THRESHOLD",
    "GARCH_MIN_ALPHA_FOR_SEPARATION",
    "ASYMMETRY_DECILES",
    "ASYMMETRY_MATERIALITY_RELATIVE",
    "ASYMMETRY_STABILITY_MIN_YEAR_FRACTION",
    "DEFAULT_N_BOOT",
    "DEFAULT_SEED",
    "GARCHFitError",
    "RowAlignmentError",
    "EntryGateResult",
    "verify_entry_gate",
    "verify_populations_row_aligned",
    "parkinson_variance_proxy",
    "causal_range_sigma",
    "compute_half_life",
    "is_stationary",
    "fit_garch11",
    "residual_diagnostics",
    "ljung_box_pvalue",
    "compute_benchmark_z",
    "describe_asymmetry",
    "decide_garch_usefulness",
    "decide_asymmetry_material",
    "default_n_workers",
    "fit_garch_by_group",
    "ANALYSIS_STAGES",
    "TOTAL_STAGES",
    "TDA11Result",
    "run_tda11_analysis",
]

# ---------------------------------------------------------------------------
# Pregunta escrita (declarada antes de ejecutar nada)
# ---------------------------------------------------------------------------

WRITTEN_QUESTION = (
    "¿Puede el modelo parametrico de volatilidad mas simple resumir la persistencia genuina "
    "encontrada en TDA-09 mediante parametros interpretables y razonablemente estables entre "
    "subperiodos, y eliminar dependencia residual de volatilidad que los benchmarks simples no "
    "eliminan?"
)

ARCH_VERSION = arch.__version__

# ---------------------------------------------------------------------------
# Convenciones PREDECLARADAS
# ---------------------------------------------------------------------------

# Verificado empiricamente (ver docstring de modulo, punto 5): sin escalar,
# arch_model dispara DataScaleWarning/ConvergenceWarning sobre retornos de
# MNQ (std~4e-4). alpha/beta/persistencia son invariantes a esta eleccion.
GARCH_SCALE_FACTOR = 1e4

# Grilla de multi-arranque para el paso de diferencias finitas (`eps`) de
# SLSQP -- ver docstring de `fit_garch11` para la causa raiz verificada
# (con log-verosimilitud negativa de orden 10^6-10^7, el `eps` por
# defecto de scipy queda por debajo del piso de ruido de punto flotante
# del objetivo y SLSQP puede "converger" trivialmente en el punto de
# partida). `None` = valor por defecto de scipy (se incluye porque a
# veces SI es el mejor, ver n=1.5M en la investigacion empirica). Grilla
# PEQUEÑA y predeclarada -- nunca una busqueda masiva (G4); se elige el
# intento con la MENOR log-verosimilitud negativa (el propio objetivo
# del MLE, un criterio objetivo, nunca subjetivo).
GARCH_OPTIMIZER_EPS_GRID: tuple[float | None, ...] = (None, 1e-7, 1e-6, 1e-5)
GARCH_OPTIMIZER_MAXITER = 500

# Tamaño minimo de muestra para intentar un ajuste GARCH de subperiodo
# (año o segmento) -- 2019 es un año parcial con ~7.000 filas (TDA-09,
# informe seccion 11); un ajuste GARCH con tan pocas observaciones no es
# fiable. Predeclarado antes de ver los conteos reales por subperiodo.
MIN_N_FOR_GARCH_FIT = 20_000

# Benchmark de rango: half-life EWMA -- MISMO valor que el estimador
# EWMA primario de TDA-10 (60 min) para que la comparacion entre
# benchmarks sea a igualdad de "memoria" del filtro, no confundida con
# una diferencia de ventana.
RANGE_BENCHMARK_HALFLIFE = 60.0

# Benchmark rodante: ventana intermedia de la grilla predeclarada de
# TDA-10 (30, 120, 390) -- 120 minutos, ni la mas reactiva ni la mas lenta.
ROLLING_BENCHMARK_WINDOW = 120

# Los 3 benchmarks obligatorios (roadmap, metodo minimo 1 de TDA-11):
# EWMA causal (reutiliza el PRIMARIO de TDA-10), rodante causal, y rango
# causal (nuevo en esta etapa). Grilla PEQUEÑA, nunca una busqueda.
BENCHMARK_CONFIGS = ("ewma_60", "rolling_120", "range_ewma_60")

# Grilla de rezagos para ACF de residuos estandarizados -- misma
# filosofia que STABILITY_LAGS de TDA-09 (pequeña, predeclarada).
DIAGNOSTIC_LAGS = (1, 5, 20, 60, 240)
PORTMANTEAU_M = (10, 20, 40, 60, 240)

# Rezago/ventana PRINCIPAL de comparacion GARCH-vs-benchmarks (decision
# de utilidad informativa, ver `decide_garch_usefulness`) -- declarados
# antes de ver ningun resultado.
RESIDUAL_COMPARISON_LAG = 1
RESIDUAL_COMPARISON_M = 60

# --- Umbrales de decision (predeclarados, roadmap seccion 11 de la
# tarea) -- GARCH se considera informativamente util frente a los
# benchmarks solo si TODAS estas condiciones se cumplen simultaneamente:
#
# (a) reduccion RELATIVA de Q(RESIDUAL_COMPARISON_M) de |z| de al menos
#     este umbral frente al MEJOR benchmark (menor Q(m) entre los 3);
# (b) alpha estimado (impacto inmediato del shock) supera este piso --
#     de lo contrario GARCH colapsa a una EWMA pura (beta~persistencia,
#     alpha~0) y no aporta una separacion util impacto/persistencia
#     que un EWMA causal ya no diera.
GARCH_RESIDUAL_REDUCTION_THRESHOLD = 0.20
GARCH_MIN_ALPHA_FOR_SEPARATION = 0.01

# TH25 -- asimetria: deciles de |r_{t-1}| para controlar magnitud del
# shock (misma logica que `assign_volatility_decile` de TDA-10, aplicada
# aqui sobre |r_{t-1}| en vez de sigma_hat).
ASYMMETRY_DECILES = 10
# Diferencia relativa minima (grupo shock negativo vs positivo, sobre el
# promedio de ambos) para considerar la asimetria MATERIAL -- unidad
# interpretable (G5): fraccion del nivel medio de la variable de
# magnitud usada (|r_t| siguiente).
ASYMMETRY_MATERIALITY_RELATIVE = 0.10
# Debe sostenerse en el MISMO signo en al menos esta fraccion de los
# años completos evaluados para considerarse ESTABLE (no solo material
# en el agregado global).
ASYMMETRY_STABILITY_MIN_YEAR_FRACTION = 0.70

DEFAULT_N_BOOT = 300
DEFAULT_SEED = 0

# Precision numerica: por debajo de este nivel, alpha+beta se trata como
# indistinguible de 1 para la clasificacion "estacionario vs no" en el
# texto (nunca cambia el numero reportado, solo el lenguaje).
STATIONARITY_EPS = 1e-6


class RowAlignmentError(Exception):
    """``r1m_pop`` y ``r_tilde_pop`` no comparten EXACTAMENTE los mismos timestamps, en el mismo orden.

    Mas estricta que `verify_populations_aligned` de TDA-10 (que solo
    compara `n`) -- TDA-11 necesita alinear `log_hl` (de `r1m_pop`) con
    `s_m` (de `r_tilde_pop`) POSICION A POSICION para construir
    `log_hl_tilde = log_hl/s_m` (el insumo del benchmark de rango sobre
    la serie ajustada por reloj) -- una coincidencia de longitud sola no
    garantiza eso. Bloqueante: si falla, TDA-11 se detiene sin construir
    ningun benchmark de rango sobre `clock_adjusted`.
    """


class GARCHFitError(Exception):
    """Un ajuste GARCH(1,1) no convergio o produjo parametros no finitos.

    Bloqueante para el fit especifico (nunca se reporta un parametro NaN
    o infinito como si fuera un resultado valido) -- el llamador decide
    si eso invalida toda la etapa (ajuste PRIMARIO GLOBAL) o si solo ese
    subperiodo queda marcado NOT_ESTIMABLE (año/segmento individual).
    """


# ---------------------------------------------------------------------------
# Puerta de entrada -- verificada sobre artefactos REALES de TDA-09,
# nunca asumida ni parseada de texto de informe.
# ---------------------------------------------------------------------------

@dataclass
class EntryGateResult:
    th21_verdict: dict
    stop9_decision: dict
    gate_open: bool
    reason: str


def verify_entry_gate(config: SnapshotConfig) -> EntryGateResult:
    """Recomputa TH21/STOP-9 desde ``TDA09_clock_attribution.csv`` con el MISMO codigo que TDA-09 uso.

    Nunca se asume "TDA-09 encontro CLUSTERING_GENUINO" porque lo diga
    un prompt o un informe -- se reconstruye la clasificacion desde el
    artefacto persistido, con `classify_th21`/`decide_stop9` importados
    SIN modificar de `tda09_volatility_clustering.py`.
    """
    path = config.tda09_clock_attribution_csv_path
    if not path.exists():
        return EntryGateResult(
            th21_verdict={"verdict": "INDETERMINADO"}, stop9_decision={"stop9_activated": False},
            gate_open=False,
            reason=f"No se encontro {path} -- TDA-09 debe ejecutarse antes que TDA-11. STOP-11.",
        )
    tab = pd.read_csv(path)
    rows = tab.to_dict(orient="records")

    th21 = classify_th21(rows)
    stop9 = decide_stop9(rows)

    if th21["verdict"] != "CLUSTERING_GENUINO":
        return EntryGateResult(
            th21, stop9, gate_open=False,
            reason=f"TH21 recomputado = {th21['verdict']} (no CLUSTERING_GENUINO). STOP-11: no se cumple el criterio de entrada 1.",
        )
    if stop9.get("stop9_activated"):
        return EntryGateResult(
            th21, stop9, gate_open=False,
            reason="STOP-9 recomputado = ACTIVADO (el clustering de TDA-09 colapso al ajuste de reloj). STOP-11: no se cumple el criterio de entrada 1.",
        )
    return EntryGateResult(
        th21, stop9, gate_open=True,
        reason=(
            "TH21 recomputado = CLUSTERING_GENUINO y STOP-9 = NO ACTIVADO (criterio de entrada 1 satisfecho). "
            "La pregunta escrita (WRITTEN_QUESTION) esta declarada y pertenece a caracterizacion, no a "
            "prediccion (criterios 2 y 3). Puerta ABIERTA -- se construyen los benchmarks."
        ),
    )


def verify_populations_row_aligned(r1m_pop: pd.DataFrame, r_tilde_pop: pd.DataFrame) -> None:
    if len(r1m_pop) != len(r_tilde_pop):
        raise RowAlignmentError(f"r1m_pop tiene {len(r1m_pop)} filas y r_tilde_pop tiene {len(r_tilde_pop)}.")
    a = r1m_pop["timestamp"].to_numpy()
    b = r_tilde_pop["timestamp"].to_numpy()
    mismatch = a != b
    if mismatch.any():
        n_bad = int(mismatch.sum())
        first_bad = int(np.argmax(mismatch))
        raise RowAlignmentError(
            f"{n_bad} timestamp(s) no coinciden entre r1m_pop y r_tilde_pop en la misma posicion "
            f"(primera discrepancia en la fila {first_bad}: {a[first_bad]} vs {b[first_bad]})."
        )


# ---------------------------------------------------------------------------
# Benchmark de rango (nuevo en esta etapa) -- Parkinson causal
# ---------------------------------------------------------------------------

def parkinson_variance_proxy(log_hl: np.ndarray) -> np.ndarray:
    """``sigma_park_t^2 = log_hl_t^2 / (4*ln(2))`` -- estimador de Parkinson (1980) de la varianza de UNA barra a partir de su rango.

    Formula CERRADA (roadmap/TDA-04, "Parkinson `sigma_hat^2~=0.3607*(H-L)^2`" -- `0.3607~=1/(4*ln2)`),
    aplicada sobre `log_hl=ln(H/L)` (TDA-04) -- consistente en escala
    logaritmica con `r_1m`/`r_tilde`, nunca sobre la diferencia de
    PRECIOS `H-L` (que mezclaria escalas con el retorno logaritmico).
    Disponible en el CIERRE de la barra `t` -- MISMA informacion temporal
    que `r_1m_t`, nunca antes.
    """
    log_hl = np.asarray(log_hl, dtype=float)
    return (log_hl ** 2) / (4.0 * np.log(2.0))


def causal_range_sigma(log_hl: np.ndarray, halflife: float = RANGE_BENCHMARK_HALFLIFE) -> np.ndarray:
    """``sigma_hat_(t-1)`` causal basado en rango: EWMA del proxy de Parkinson, desplazado un paso.

    Reutiliza `causal_ewma_sigma` de TDA-10 SIN MODIFICARLO: esa funcion
    ya implementa exactamente la recursion causal deseada
    (`sigma2_t = lambda*sigma2_(t-1)+(1-lambda)*x_(t-1)^2`, `shift(1)`
    explicito) pero eleva su argumento al cuadrado internamente -- como
    aqui la cantidad de interes YA es una varianza (`parkinson_variance_proxy`),
    se le pasa su RAIZ CUADRADA (`sqrt(proxy)`) para que el cuadrado
    interno de `causal_ewma_sigma` la reconstruya exactamente. Evita
    duplicar la logica de EWMA causal para un segundo tipo de insumo.
    """
    proxy = parkinson_variance_proxy(log_hl)
    with np.errstate(invalid="ignore"):
        return causal_ewma_sigma(np.sqrt(proxy), halflife)


# ---------------------------------------------------------------------------
# GARCH(1,1) -- ajuste, persistencia, vida media
# ---------------------------------------------------------------------------

def is_stationary(persistence: float) -> bool:
    """``0 < persistence < 1 - STATIONARITY_EPS`` -- extraida como funcion pura para poder testear el caso frontera `alpha+beta>=1` sin depender de un ajuste GARCH real."""
    return bool(np.isfinite(persistence) and 0.0 < persistence < 1.0 - STATIONARITY_EPS)


def compute_half_life(persistence: float) -> float | None:
    """Vida media (en barras validas) de un shock de volatilidad: ``ln(0.5)/ln(persistence)``.

    Solo definida (finita) si `is_stationary(persistence)` (roadmap: "si
    alpha+beta>=1, no fuerces una vida media finita"). Devuelve `None`
    explicito en cualquier otro caso -- nunca un numero fabricado.

    Auditoria post-primera-ejecucion (hallazgo real sobre el conjunto de
    investigacion): usar el chequeo LITERAL `persistence>=1` en vez de
    `is_stationary` dejaba pasar valores como `0.999999999999999`
    (`persistence` a distancia de punto flotante de 1, tipico de un
    ajuste con la restriccion de estacionariedad de `arch` ACTIVA en el
    borde -- el optimo SIN restriccion querria `alpha+beta>=1`) y
    calculaba una vida media de `~7*10^14` barras (~mil millones de años)
    -- un numero tecnicamente "finito" pero economicamente sin sentido,
    NUNCA una vida media real. Usar el MISMO umbral que `is_stationary`
    (`STATIONARITY_EPS`) evita reportar esa cifra fabricada.
    """
    if not is_stationary(persistence):
        return None
    return float(np.log(0.5) / np.log(persistence))


def fit_garch11(
    r: np.ndarray, dist: str = "normal", scale: float = GARCH_SCALE_FACTOR, asymmetric: bool = False,
) -> dict:
    """Ajusta GARCH(1,1) (o GJR-GARCH(1,1,1) si `asymmetric=True`) sobre `r` -- media CERO (TDA-08: dependencia en media despreciable, STOP-8a).

    `r` se reescala por `scale` antes de ajustar (ver constante de modulo,
    punto 5 del docstring) -- `alpha`/`beta`/`gamma`/persistencia se
    reportan SIN escala (invariantes); `omega` se des-escala dividiendo
    por `scale**2`. Inferencia ROBUSTA (`cov_type="robust"`, sandwich
    Bollerslev-Wooldridge) siempre, sea cual sea `dist`.

    MULTI-ARRANQUE (auditoria post-primera-ejecucion sobre el conjunto de
    investigacion real -- ver docstring de modulo, punto 8): se detecto
    que, con `n~10^6` y la curtosis genuina de MNQ, `scipy.optimize.minimize`
    (SLSQP, usado internamente por `arch`) puede reportar "convergencia
    exitosa" (`convergence_flag=0`) tras UNA sola iteracion, devolviendo
    LITERALMENTE el punto de PARTIDA sin haberse movido -- verificado
    comparando `res.params` contra `GARCH.starting_values(resid)`
    (coincidian byte a byte) y confirmando un gradiente (`jac`) de
    magnitud grande (cientos de miles) en el supuesto "optimo". La causa
    raiz (verificada empiricamente, no solo sospechada): con `n~10^6`,
    la log-verosimilitud NEGATIVA sumada es del orden de `10^6-10^7`; el
    paso de diferencias finitas POR DEFECTO de SLSQP (`eps~1.49e-8`,
    calibrado para problemas de magnitud O(1)) queda por debajo del
    "piso de ruido" de redondeo de punto flotante de un objetivo de ese
    tamaño, produciendo un gradiente por diferencias finitas dominado por
    ruido numerico en vez de la derivada real -- SLSQP entonces "converge"
    trivialmente. Fue reproducido y corregido de forma DEMOSTRABLE, no
    solo hipotetica: variando `eps` (el paso de diferencias finitas) se
    escapa de ese punto espurio, pero NINGUN valor unico de `eps`
    funciona de forma fiable en todos los subperiodos -- por eso se
    ajusta con una GRILLA PEQUEÑA y predeclarada de valores de `eps`
    (`GARCH_OPTIMIZER_EPS_GRID`) y se conserva el ajuste con la MENOR
    log-verosimilitud negativa (el criterio es la propia funcion objetivo
    del MLE, nunca una eleccion subjetiva) -- practica estandar de
    "multi-start" para verosimilitudes numericamente dificiles, NO una
    reimplementacion del optimizador (cada intento sigue siendo
    `arch_model(...).fit()` sin modificar).

    Salida: `dict` con `omega`/`alpha`/`beta`/`gamma` (si `asymmetric`),
    sus IC 95% robustos, `persistence`, `half_life`, `stationary`
    (bool), `convergence_flag` (0=exito del intento SELECCIONADO), `n`,
    `loglikelihood`, `standardized_resid` (array, MISMA longitud que `r`,
    alineado posicion a posicion), `dist`, `asymmetric`, `eps_selected`
    (que intento de la grilla gano), `n_multistart_attempts_finite`,
    `converged_cleanly` (`True` si AL MENOS UN intento de la grilla
    reporto `convergence_flag==0` -- el intento seleccionado siempre se
    toma de ese subconjunto cuando existe; `False` solo si NINGUN
    intento convergio limpiamente, en cuyo caso se usa igualmente el de
    menor log-verosimilitud entre todos, documentado como no confiable).
    Lanza `GARCHFitError` si NINGUN intento de la grilla produce
    parametros finitos.
    """
    r = np.asarray(r, dtype=float)
    n = r.size
    o = 1 if asymmetric else 0
    am = arch_model(r * scale, mean="Zero", vol="GARCH", p=1, o=o, q=1, dist=dist)

    finite_attempts: list[tuple[float, float | None, object]] = []  # (neg_llf, eps, result)
    for eps in GARCH_OPTIMIZER_EPS_GRID:
        options = {"maxiter": GARCH_OPTIMIZER_MAXITER}
        if eps is not None:
            options["eps"] = eps
        try:
            attempt = am.fit(disp="off", cov_type="robust", show_warning=False, options=options)
        except Exception:
            continue
        neg_llf = -float(attempt.loglikelihood)
        if not np.isfinite(neg_llf) or not np.isfinite(attempt.params.to_numpy()).all():
            continue
        finite_attempts.append((neg_llf, eps, attempt))

    if not finite_attempts:
        raise GARCHFitError(
            f"Ninguno de los {len(GARCH_OPTIMIZER_EPS_GRID)} intentos de la grilla de multi-arranque "
            f"(GARCH_OPTIMIZER_EPS_GRID) produjo un ajuste con log-verosimilitud finita (n={n})."
        )

    # Preferencia: entre los intentos con `convergence_flag==0` (exito
    # reportado por SLSQP), el de MENOR log-verosimilitud negativa. Solo
    # si NINGUNO reporto exito se recurre al mejor entre TODOS los
    # intentos finitos (documentado explicitamente via `converged_cleanly`
    # en la salida) -- un intento con `convergence_flag!=0` (p.ej. "Positive
    # directional derivative for linesearch") puede evaluar una
    # log-verosimilitud engañosamente baja en un punto que scipy NUNCA
    # declaro optimo; nunca se prefiere sobre un intento que si convergio.
    clean = [t for t in finite_attempts if int(t[2].convergence_flag) == 0]
    pool = clean if clean else finite_attempts
    best_neg_llf, best_eps, res = min(pool, key=lambda t: t[0])
    converged_cleanly = bool(clean)

    params = res.params
    conf = res.conf_int()

    def _get(name: str, descale_power: int = 0) -> tuple[float, float, float]:
        point = float(params[name]) / (scale ** descale_power)
        lo = float(conf.loc[name, "lower"]) / (scale ** descale_power)
        hi = float(conf.loc[name, "upper"]) / (scale ** descale_power)
        return point, lo, hi

    omega, omega_lo, omega_hi = _get("omega", descale_power=2)
    alpha, alpha_lo, alpha_hi = _get("alpha[1]", descale_power=0)
    beta, beta_lo, beta_hi = _get("beta[1]", descale_power=0)
    gamma = gamma_lo = gamma_hi = float("nan")
    if asymmetric:
        gamma, gamma_lo, gamma_hi = _get("gamma[1]", descale_power=0)
        persistence = alpha + beta + gamma / 2.0
    else:
        persistence = alpha + beta

    if not np.isfinite([omega, alpha, beta, persistence]).all():
        raise GARCHFitError(f"Ajuste GARCH produjo parametro(s) no finito(s): omega={omega}, alpha={alpha}, beta={beta}")

    standardized_resid = (r * scale) / np.asarray(res.conditional_volatility, dtype=float)

    return {
        "n": n, "dist": dist, "asymmetric": bool(asymmetric),
        "omega": omega, "omega_ci_lo": omega_lo, "omega_ci_hi": omega_hi,
        "alpha": alpha, "alpha_ci_lo": alpha_lo, "alpha_ci_hi": alpha_hi,
        "beta": beta, "beta_ci_lo": beta_lo, "beta_ci_hi": beta_hi,
        "gamma": gamma, "gamma_ci_lo": gamma_lo, "gamma_ci_hi": gamma_hi,
        "persistence": persistence,
        "stationary": is_stationary(persistence),
        "half_life": compute_half_life(persistence),
        "convergence_flag": int(res.convergence_flag),
        "loglikelihood": float(res.loglikelihood),
        "standardized_resid": standardized_resid,
        "eps_selected": best_eps,
        "n_multistart_attempts_finite": len(finite_attempts),
        "converged_cleanly": converged_cleanly,
        "n_multistart_attempts_total": len(GARCH_OPTIMIZER_EPS_GRID),
    }


# ---------------------------------------------------------------------------
# Diagnostico de residuos (ACF, ACF^2, |z|, portmanteau, Ljung-Box)
# ---------------------------------------------------------------------------

def ljung_box_pvalue(q_value: float, m: int) -> float:
    """P-valor asintotico aproximado de `Q(m)` (TDA-08 `compute_portmanteau_q`) bajo chi-cuadrado(`m`) g.l.

    `Q(m)=sum n_pairs_k*rho_k^2` NO es el estadistico de Ljung-Box
    clasico (TDA-08, docstring de `compute_portmanteau_q`: la
    correlacion es PAIRWISE, no de denominador fijo `T`) -- pero cada
    termino aproxima un `chi2(1)` bajo la hipotesis nula de no
    dependencia (varianza `~1/n_pairs_k` de una correlacion de Pearson
    sobre `n_pairs_k` pares aprox. independientes), por lo que la suma
    de `m` terminos aproxima `chi2(m)`. Se reporta como diagnostico
    NOMBRADO "Ljung-Box" (la tarea lo exige explicitamente) pero NUNCA
    como criterio de importancia practica (G5, prohibicion explicita de
    la tarea) -- la magnitud de `Q(m)`/`rho_k` es siempre el resultado
    principal; este p-valor es un complemento asintotico aproximado.
    """
    if not np.isfinite(q_value) or m <= 0:
        return float("nan")
    return float(1.0 - chi2.cdf(q_value, df=m))


def residual_diagnostics(
    z: np.ndarray, block_ids: np.ndarray, lags: tuple[int, ...] = DIAGNOSTIC_LAGS, m_values: tuple[int, ...] = PORTMANTEAU_M,
) -> dict:
    """ACF/portmanteau/Ljung-Box de `z` y de `z^2` (roadmap, metodo minimo 4 de TDA-11), respetando bloques de continuidad genuina.

    `block_ids` SIEMPRE se calcula sobre la poblacion REAL (timestamps
    genuinos), nunca sobre la topologia "por fila" que el propio filtro
    (EWMA/rodante/rango/GARCH) uso para construir `z` -- ver docstring
    de modulo, punto 4: el diagnostico no hereda la simplificacion del
    filtro.

    Auditoria post-primera-ejecucion (bug real detectado sobre el
    conjunto de investigacion real, no solo teorico): `compute_acf`
    (TDA-08) NO filtra `NaN` fila a fila dentro de un bloque -- espera
    que el llamador ya le pase una poblacion COMPACTADA (sin huecos de
    `NaN`), exactamente como TDA-09/10 siempre hacen. La version anterior
    de esta funcion dejaba las filas de "quemado" (burn-in, `NaN`) DENTRO
    del array de longitud completa antes de llamar a `compute_acf` --
    un SOLO `NaN` en cualquier posicion arrastra a `NaN` la SUMA completa
    (`np.sum`) de la que depende `rho` en TODOS los rezagos, inutilizando
    silenciosamente el diagnostico entero (confirmado: `n_pairs` correcto
    pero `rho=NaN` en cada fila). La correccion filtra `z` (y `block_ids`
    en la MISMA posicion) al subconjunto finito ANTES de llamar a
    `compute_acf` -- la misma disciplina de "poblacion compactada antes
    de calcular bloques" que TDA-07/09/10 ya aplican.

    Salida: `dict` con `acf_z` y `acf_z2` (`DataFrame`s de `compute_acf`),
    `portmanteau_z`/`portmanteau_z2` (`DataFrame`s de
    `compute_portmanteau_q` + columna `lb_pvalue`), y `rho1_abs_z`
    (rezago 1 de `ACF(|z|)`, el resumen de una sola cifra usado por
    `decide_garch_usefulness`).
    """
    z = np.asarray(z, dtype=float)
    block_ids = np.asarray(block_ids)
    max_lag = max(lags)
    finite = np.isfinite(z)

    z_compact = z[finite]
    block_ids_compact = block_ids[finite]

    acf_z = compute_acf(z_compact, block_ids_compact, max_lag)
    acf_abs_z = compute_acf(np.abs(z_compact), block_ids_compact, max_lag)
    acf_z2 = compute_acf(z_compact ** 2, block_ids_compact, max_lag)

    port_abs_z = compute_portmanteau_q(acf_abs_z, m_values)
    port_abs_z["lb_pvalue"] = [ljung_box_pvalue(q, m) for q, m in zip(port_abs_z["Q"], port_abs_z["m"])]
    port_z2 = compute_portmanteau_q(acf_z2, m_values)
    port_z2["lb_pvalue"] = [ljung_box_pvalue(q, m) for q, m in zip(port_z2["Q"], port_z2["m"])]

    rho1_row = acf_abs_z.loc[acf_abs_z["lag"] == RESIDUAL_COMPARISON_LAG]
    rho1_abs_z = float(rho1_row["rho"].iloc[0]) if not rho1_row.empty and bool(rho1_row["estimable"].iloc[0]) else float("nan")

    q_row = port_abs_z.loc[port_abs_z["m"] == RESIDUAL_COMPARISON_M]
    q_abs_z_m = float(q_row["Q"].iloc[0]) if not q_row.empty and bool(q_row["estimable"].iloc[0]) else float("nan")

    return {
        "acf_z": acf_z, "acf_abs_z": acf_abs_z, "acf_z2": acf_z2,
        "portmanteau_abs_z": port_abs_z, "portmanteau_z2": port_z2,
        "rho1_abs_z": rho1_abs_z, "q_abs_z_m": q_abs_z_m,
        "n_finite": int(finite.sum()),
    }


# ---------------------------------------------------------------------------
# Benchmarks -- reutilizan TDA-10 sin modificarlo
# ---------------------------------------------------------------------------

def compute_benchmark_z(r: np.ndarray, log_hl: np.ndarray, config_name: str) -> np.ndarray:
    """Construye `z_t` para uno de los 3 `BENCHMARK_CONFIGS`, reutilizando `tda10_scale_vs_shape` sin modificarlo."""
    if config_name == "ewma_60":
        sigma_hat = build_sigma_hat(r, "ewma", 60.0)
        burn_in = burn_in_length("ewma", 60.0)
    elif config_name == "rolling_120":
        sigma_hat = build_sigma_hat(r, "rolling_std", ROLLING_BENCHMARK_WINDOW)
        burn_in = burn_in_length("rolling_std", ROLLING_BENCHMARK_WINDOW)
    elif config_name == "range_ewma_60":
        sigma_hat = causal_range_sigma(log_hl, RANGE_BENCHMARK_HALFLIFE)
        burn_in = burn_in_length("ewma", RANGE_BENCHMARK_HALFLIFE)
    else:
        raise ValueError(f"config_name desconocido: {config_name!r}")
    return standardize_return(r, sigma_hat, burn_in)


# ---------------------------------------------------------------------------
# TH25 -- asimetria descriptiva
# ---------------------------------------------------------------------------

def _asymmetry_pooled_stat(mag: np.ndarray, sign: np.ndarray) -> float:
    pos = mag[sign > 0]
    neg = mag[sign < 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    pooled_mean = float(mag.mean())
    if pooled_mean == 0 or not np.isfinite(pooled_mean):
        return float("nan")
    return float((neg.mean() - pos.mean()) / pooled_mean)


def describe_asymmetry(
    r_prev: np.ndarray, magnitude_next: np.ndarray, trading_date_next: np.ndarray,
    n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED,
) -> dict:
    """Compara `magnitude_next` (p.ej. `|r_t|`) entre shock previo POSITIVO y NEGATIVO, controlando magnitud por deciles de `|r_(t-1)|`.

    Metodo (roadmap TH25, metodo minimo): descriptivo primero, NUNCA se
    ajusta un modelo asimetrico antes de este paso. `r_prev[i]` es el
    shock previo (`r_(t-1)`) y `magnitude_next[i]` la variable de
    respuesta (`|r_t|` o similar) -- MISMA posicion `i`, ya alineadas y
    restringidas por el llamador a pares del MISMO bloque de continuidad
    (nunca fabricados a traves de un hueco/roll -- ver
    `run_tda11_analysis`). `trading_date_next[i]` es la fecha de
    negociacion de la observacion `magnitude_next[i]`, usada por el
    bootstrap de bloques de jornada (reutiliza `day_block_bootstrap`,
    TDA-07).

    Salida: `dict` con `pooled_rel_diff`/`pooled_ci_lo`/`pooled_ci_hi`
    (diferencia relativa global negativo-positivo, con IC bootstrap de
    bloques por jornada, G5) y `by_decile` (`DataFrame`, la comparacion
    CONTROLADA por magnitud del shock, sin IC por decil -- solo punto
    estimado, misma disciplina de parsimonia que TDA-10 aplico a sus
    tablas de estabilidad por decil).
    """
    r_prev = np.asarray(r_prev, dtype=float)
    magnitude_next = np.asarray(magnitude_next, dtype=float)
    sign_prev = np.sign(r_prev)
    abs_prev = np.abs(r_prev)

    valid = np.isfinite(r_prev) & np.isfinite(magnitude_next) & (sign_prev != 0)
    rows = []
    if int(valid.sum()) >= ASYMMETRY_DECILES * 20:
        deciles = pd.qcut(pd.Series(abs_prev[valid]), ASYMMETRY_DECILES, labels=False, duplicates="drop")
        df = pd.DataFrame({"decile": deciles.to_numpy(), "sign": sign_prev[valid], "mag": magnitude_next[valid]})
        for d, sub in df.groupby("decile", observed=True):
            pos = sub.loc[sub["sign"] > 0, "mag"]
            neg = sub.loc[sub["sign"] < 0, "mag"]
            mean_pos = float(pos.mean()) if len(pos) else float("nan")
            mean_neg = float(neg.mean()) if len(neg) else float("nan")
            pooled_mean = float(sub["mag"].mean())
            diff = mean_neg - mean_pos
            rel_diff = diff / pooled_mean if pooled_mean else float("nan")
            rows.append({
                "decile": int(d), "n_pos": int(len(pos)), "n_neg": int(len(neg)),
                "mean_pos": mean_pos, "mean_neg": mean_neg, "diff_neg_minus_pos": diff, "rel_diff": rel_diff,
            })
    by_decile = pd.DataFrame(rows)

    pooled_point = _asymmetry_pooled_stat(magnitude_next[valid], sign_prev[valid])
    ci_lo = ci_hi = float("nan")
    if np.isfinite(pooled_point) and int(valid.sum()) >= 200:
        mag_v = magnitude_next[valid]
        sign_v = sign_prev[valid]
        dates_v = np.asarray(trading_date_next)[valid]

        # Empaqueta (mag, sign) en un unico array 2D para que
        # day_block_bootstrap remuestree AMBAS columnas de forma
        # consistente (mismos indices de fila, indexado por fila en un
        # array 2D) en cada replica.
        packed = np.column_stack([mag_v, sign_v])
        boot = day_block_bootstrap(packed, dates_v, lambda x: _asymmetry_pooled_stat(x[:, 0], x[:, 1]), n_boot, seed)
        finite = boot[np.isfinite(boot)]
        if finite.size >= 10:
            ci_lo, ci_hi = float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))

    return {
        "pooled_rel_diff": pooled_point, "pooled_ci_lo": ci_lo, "pooled_ci_hi": ci_hi,
        "n": int(valid.sum()), "by_decile": by_decile,
    }


def decide_asymmetry_material(by_decile: pd.DataFrame) -> dict:
    """Clasifica la asimetria como MATERIAL si la diferencia relativa mediana entre deciles supera `ASYMMETRY_MATERIALITY_RELATIVE` Y el signo es consistente en la mayoria de los deciles."""
    valid = by_decile.dropna(subset=["rel_diff"])
    if valid.empty:
        return {"material": False, "reason": "sin deciles estimables"}
    median_rel = float(valid["rel_diff"].median())
    same_sign_frac = float((np.sign(valid["rel_diff"]) == np.sign(median_rel)).mean()) if median_rel != 0 else float("nan")
    material = bool(abs(median_rel) >= ASYMMETRY_MATERIALITY_RELATIVE and same_sign_frac >= 0.5)
    return {
        "material": material, "median_rel_diff": median_rel, "same_sign_fraction": same_sign_frac,
        "direction": "NEGATIVO_MAS_VOLATIL" if median_rel > 0 else "POSITIVO_MAS_VOLATIL",
    }


# ---------------------------------------------------------------------------
# Utilidad informativa de GARCH frente a benchmarks (STOP-11 de cierre)
# ---------------------------------------------------------------------------

def decide_garch_usefulness(garch_diag: dict, benchmark_diags: dict[str, dict], garch_fit: dict) -> dict:
    """Reglas predeclaradas (roadmap seccion 11 de la tarea): GARCH es informativamente util si (a) reduce Q(m) de |z| al menos `GARCH_RESIDUAL_REDUCTION_THRESHOLD` frente al MEJOR benchmark, y (b) alpha separa impacto inmediato de persistencia."""
    best_benchmark_q = min(
        (d["q_abs_z_m"] for d in benchmark_diags.values() if np.isfinite(d["q_abs_z_m"])), default=float("nan"),
    )
    garch_q = garch_diag["q_abs_z_m"]
    if not np.isfinite(best_benchmark_q) or not np.isfinite(garch_q) or best_benchmark_q <= 0:
        relative_reduction = float("nan")
        reduces_dependence = False
    else:
        relative_reduction = float(1.0 - garch_q / best_benchmark_q)
        reduces_dependence = bool(relative_reduction >= GARCH_RESIDUAL_REDUCTION_THRESHOLD)

    separates_impact = bool(np.isfinite(garch_fit["alpha"]) and garch_fit["alpha"] >= GARCH_MIN_ALPHA_FOR_SEPARATION)
    useful = bool(reduces_dependence and separates_impact)

    return {
        "useful": useful,
        "best_benchmark_q_abs_z_m": best_benchmark_q, "garch_q_abs_z_m": garch_q,
        "relative_reduction": relative_reduction, "reduces_dependence": reduces_dependence,
        "alpha": garch_fit["alpha"], "separates_impact": separates_impact,
        "reduction_threshold": GARCH_RESIDUAL_REDUCTION_THRESHOLD, "alpha_threshold": GARCH_MIN_ALPHA_FOR_SEPARATION,
    }


# ---------------------------------------------------------------------------
# Hardware -- workers CPU, mismo criterio que docs/project_hardware.md
# ---------------------------------------------------------------------------

def default_n_workers(n_jobs: int) -> int:
    """``min(20, cpu_count()-4, n_jobs)`` -- nunca mas workers que trabajos independientes ni que el maximo de referencia del hardware del proyecto."""
    cpu = os.cpu_count() or 4
    return max(1, min(20, cpu - 4, max(1, n_jobs)))


def _fit_garch_worker(args: tuple) -> dict:
    """Worker de proceso (picklable, nivel de modulo): limita hilos BLAS internos a 1 (best-effort, ver docs/project_hardware.md) antes de ajustar.

    Calcula el diagnostico de residuos DENTRO del worker (con los
    `block_ids` ya recortados al subperiodo, calculados en el proceso
    principal) y devuelve solo el resumen COMPACTO -- nunca el array de
    residuos completo -- para evitar copiar arrays grandes de vuelta al
    proceso principal (docs/project_hardware.md, seccion 7: "evita
    copiar innecesariamente arrays de millones de filas entre procesos").
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    group_label, r, block_ids, dist, scale, min_n, lags, m_values = args
    if r.size < min_n:
        return {"group": group_label, "estimable": False, "n": int(r.size), "reason": f"n={r.size}<{min_n} (MIN_N_FOR_GARCH_FIT)"}
    try:
        fit = fit_garch11(r, dist=dist, scale=scale)
    except GARCHFitError as exc:
        return {"group": group_label, "estimable": False, "n": int(r.size), "reason": str(exc)}

    diag = residual_diagnostics(fit["standardized_resid"], block_ids, lags=lags, m_values=m_values)
    out = {k: v for k, v in fit.items() if k != "standardized_resid"}
    out["group"] = group_label
    out["estimable"] = True
    out["rho1_abs_z"] = diag["rho1_abs_z"]
    out["q_abs_z_m"] = diag["q_abs_z_m"]
    return out


def fit_garch_by_group(
    groups: dict[str, np.ndarray], block_ids_by_group: dict[str, np.ndarray], dist: str = "normal",
    scale: float = GARCH_SCALE_FACTOR, min_n: int = MIN_N_FOR_GARCH_FIT, n_workers: int | None = None,
    lags: tuple[int, ...] = (1, 5, 20, 60), m_values: tuple[int, ...] = (20, 60),
) -> pd.DataFrame:
    """Ajusta GARCH(1,1) independientemente para cada grupo de `groups` ({label: array de retornos}), en paralelo (ProcessPoolExecutor).

    Trabajos NATURALMENTE independientes (docs/project_hardware.md
    seccion 7, punto 3) -- un año o un segmento no comparte estado con
    otro. `n_workers` por defecto: `default_n_workers(len(groups))`
    (maximo ~20, nunca mas que trabajos disponibles). Reproducible: cada
    ajuste es determinista dado su array de entrada (MLE con el mismo
    optimizador, sin aleatoriedad) -- el orden de la tabla de salida SI
    esta fijado (orden de `groups`), para que dos ejecuciones produzcan
    exactamente la misma tabla.
    """
    if n_workers is None:
        n_workers = default_n_workers(len(groups))
    jobs = [(label, arr, block_ids_by_group[label], dist, scale, min_n, lags, m_values) for label, arr in groups.items()]
    if n_workers <= 1 or len(jobs) <= 1:
        results = [_fit_garch_worker(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_fit_garch_worker, jobs))
    order = {label: i for i, label in enumerate(groups.keys())}
    results.sort(key=lambda r: order[r["group"]])
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Progreso visible -- mismo patron que TDA-09/10
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
                print(f"[TDA11 {self._i}/{self.total}] {self._current_name} -- completado en {elapsed_stage:.1f}s (total: {total_elapsed:.1f}s)")
        self._i += 1
        self._current_name = name
        self._last = now
        if self.verbose:
            print(f"[TDA11 {self._i}/{self.total}] {name}...")

    def finish(self) -> dict[str, float]:
        now = time.perf_counter()
        elapsed_stage = now - self._last
        self.stage_timings[self._current_name] = elapsed_stage
        if self.verbose:
            total_elapsed = now - self.t_start
            print(f"[TDA11 {self._i}/{self.total}] {self._current_name} -- completado en {elapsed_stage:.1f}s (total: {total_elapsed:.1f}s)")
        self.stage_timings["_total_analysis"] = now - self.t_start
        return self.stage_timings


ANALYSIS_STAGES = 6
TOTAL_STAGES = 7  # 6 de analisis + 1 de persistencia de artefactos (run_tda11.py)


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------

@dataclass
class TDA11Result:
    entry_gate: EntryGateResult
    stop11_activated: bool
    stop11_reason: str

    n_raw: int
    n_clock_adjusted: int
    n_workers: int

    benchmark_diagnostics: dict = field(default_factory=dict)  # {"raw": {config: diag}, "clock_adjusted": {config: diag}}

    garch_primary_global: dict = field(default_factory=dict)          # clock_adjusted, gaussian, global
    garch_primary_global_diag: dict = field(default_factory=dict)
    garch_secondary_raw_global: dict = field(default_factory=dict)    # raw, gaussian, global (sensibilidad)
    garch_secondary_raw_global_diag: dict = field(default_factory=dict)
    garch_student_t_sensitivity: dict = field(default_factory=dict)   # clock_adjusted, student-t, global

    garch_by_year: pd.DataFrame = None
    garch_by_segment: pd.DataFrame = None

    garch_usefulness: dict = field(default_factory=dict)

    asymmetry_global: dict = field(default_factory=dict)
    asymmetry_by_year: pd.DataFrame = None
    asymmetry_decision: dict = field(default_factory=dict)
    garch_asymmetric_fit: dict | None = None
    garch_asymmetric_diag: dict | None = None

    stage_timings: dict[str, float] = field(default_factory=dict)


def _closed_result(entry_gate: EntryGateResult) -> TDA11Result:
    return TDA11Result(
        entry_gate=entry_gate, stop11_activated=True, stop11_reason=entry_gate.reason,
        n_raw=0, n_clock_adjusted=0, n_workers=0, stage_timings={},
    )


def run_tda11_analysis(config: SnapshotConfig, n_workers: int | None = None, verbose: bool = True) -> TDA11Result:
    """Orquesta TDA-11 completo: puerta de entrada -> poblaciones -> benchmarks -> GARCH(1,1) primario/secundario/sensibilidad -> estabilidad por año/segmento -> utilidad informativa/STOP-11 -> asimetria TH25."""
    validate_research_holdout_disjoint(config)
    entry_gate = verify_entry_gate(config)
    if not entry_gate.gate_open:
        if verbose:
            print(f"[TDA11] STOP-11: {entry_gate.reason}")
        return _closed_result(entry_gate)

    timer = _StageTimer(ANALYSIS_STAGES, verbose=verbose)

    # --- Etapa 1: poblaciones e invariantes ---------------------------------
    timer.stage("Poblaciones (r_1m/r_tilde), invariantes y bloques de continuidad")
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
    verify_populations_row_aligned(r1m_pop, r_tilde_pop)

    r_raw = r1m_pop["r_1m"].to_numpy(dtype=float)
    r_adj = r_tilde_pop["r_tilde"].to_numpy(dtype=float)
    log_hl_raw = r1m_pop["log_hl"].to_numpy(dtype=float)
    log_hl_tilde = log_hl_raw / r_tilde_pop["s_m"].to_numpy(dtype=float)

    block_ids_raw = compute_block_ids_with_contract(r1m_pop["timestamp"], r1m_pop["trading_date"], r1m_pop["contract"], 60.0)
    block_ids_adj = compute_block_ids_with_contract(r_tilde_pop["timestamp"], r_tilde_pop["trading_date"], r_tilde_pop["contract"], 60.0)

    if n_workers is None:
        n_workers = default_n_workers(1 + len(segment_labels))

    # --- Etapa 2: benchmarks (3 configuraciones x 2 bloques) ----------------
    timer.stage(f"Benchmarks causales (EWMA/rodante/rango) x raw+clock_adjusted")
    benchmark_diagnostics: dict[str, dict] = {"raw": {}, "clock_adjusted": {}}
    for cfg in BENCHMARK_CONFIGS:
        z_raw = compute_benchmark_z(r_raw, log_hl_raw, cfg)
        benchmark_diagnostics["raw"][cfg] = residual_diagnostics(z_raw, block_ids_raw)
        z_adj = compute_benchmark_z(r_adj, log_hl_tilde, cfg)
        benchmark_diagnostics["clock_adjusted"][cfg] = residual_diagnostics(z_adj, block_ids_adj)

    # --- Etapa 3: GARCH global -- primario, secundario, sensibilidad t -----
    timer.stage("GARCH(1,1) global -- primario (clock_adjusted), secundario (raw), sensibilidad Student-t")
    garch_primary_global = fit_garch11(r_adj, dist="normal")
    garch_primary_global_diag = residual_diagnostics(garch_primary_global["standardized_resid"], block_ids_adj)

    garch_secondary_raw_global = fit_garch11(r_raw, dist="normal")
    garch_secondary_raw_global_diag = residual_diagnostics(garch_secondary_raw_global["standardized_resid"], block_ids_raw)

    garch_student_t_sensitivity = fit_garch11(r_adj, dist="t")

    # --- Etapa 4: GARCH por año y por segmento (clock_adjusted, paralelo) --
    timer.stage(f"GARCH(1,1) por año y por segmento (clock_adjusted, {n_workers} workers)")
    year_groups = {int(y): r_adj[r_tilde_pop["year_ny"].to_numpy() == y] for y in sorted(r_tilde_pop["year_ny"].unique())}
    year_block_ids = {
        int(y): compute_block_ids_with_contract(
            r_tilde_pop.loc[r_tilde_pop["year_ny"] == y, "timestamp"],
            r_tilde_pop.loc[r_tilde_pop["year_ny"] == y, "trading_date"],
            r_tilde_pop.loc[r_tilde_pop["year_ny"] == y, "contract"], 60.0,
        )
        for y in sorted(r_tilde_pop["year_ny"].unique())
    }
    segment_groups = {s: r_adj[r_tilde_pop["segment_label"].to_numpy() == s] for s in segment_labels}
    segment_block_ids = {
        s: compute_block_ids_with_contract(
            r_tilde_pop.loc[r_tilde_pop["segment_label"] == s, "timestamp"],
            r_tilde_pop.loc[r_tilde_pop["segment_label"] == s, "trading_date"],
            r_tilde_pop.loc[r_tilde_pop["segment_label"] == s, "contract"], 60.0,
        )
        for s in segment_labels
    }
    garch_by_year = fit_garch_by_group(year_groups, year_block_ids, n_workers=n_workers)
    garch_by_segment = fit_garch_by_group(segment_groups, segment_block_ids, n_workers=n_workers)

    # --- Etapa 5: utilidad informativa de GARCH + STOP-11 -------------------
    timer.stage("Utilidad informativa de GARCH frente a benchmarks (decision STOP-11)")
    garch_usefulness = decide_garch_usefulness(
        garch_primary_global_diag, benchmark_diagnostics["clock_adjusted"], garch_primary_global,
    )
    if garch_usefulness["useful"]:
        stop11_activated, stop11_reason = False, (
            f"GARCH(1,1) aporta informacion sobre los benchmarks simples: reduccion relativa de "
            f"Q({RESIDUAL_COMPARISON_M}) de |z| = {garch_usefulness['relative_reduction']:.3f} "
            f"(umbral {GARCH_RESIDUAL_REDUCTION_THRESHOLD}), alpha={garch_usefulness['alpha']:.4f} "
            f"(umbral {GARCH_MIN_ALPHA_FOR_SEPARATION}). STOP-11 NO activado."
        )
    else:
        stop11_activated, stop11_reason = True, (
            f"GARCH(1,1) NO supera el umbral de utilidad informativa predeclarado frente a los benchmarks "
            f"simples (reduccion relativa={garch_usefulness['relative_reduction']}, alpha={garch_usefulness['alpha']}). "
            "STOP-11 activado -- resultado negativo (G6), no se ajustan modelos adicionales por este motivo."
        )

    # --- Etapa 6: TH25 -- asimetria descriptiva + GJR condicional ----------
    timer.stage("TH25 -- asimetria descriptiva (global + por año) y extension asimetrica condicional")
    same_block_prev = np.concatenate([[False], block_ids_adj[1:] == block_ids_adj[:-1]])
    r_prev = np.concatenate([[np.nan], r_adj[:-1]])
    mag_next = np.abs(r_adj)
    valid_pairs = same_block_prev
    trading_date_next = r_tilde_pop["trading_date"].to_numpy()

    asymmetry_global = describe_asymmetry(
        r_prev[valid_pairs], mag_next[valid_pairs], trading_date_next[valid_pairs], n_boot=config.tda11_n_boot,
    )

    year_arr = r_tilde_pop["year_ny"].to_numpy()
    asym_year_rows = []
    for y in sorted(r_tilde_pop["year_ny"].unique()):
        y_mask = valid_pairs & (year_arr == y)
        if int(y_mask.sum()) < ASYMMETRY_DECILES * 20:
            continue
        res_y = describe_asymmetry(r_prev[y_mask], mag_next[y_mask], trading_date_next[y_mask], n_boot=50, seed=DEFAULT_SEED)
        decision_y = decide_asymmetry_material(res_y["by_decile"])
        asym_year_rows.append({
            "year": int(y), "n": res_y["n"], "pooled_rel_diff": res_y["pooled_rel_diff"],
            "pooled_ci_lo": res_y["pooled_ci_lo"], "pooled_ci_hi": res_y["pooled_ci_hi"],
            "median_rel_diff": decision_y.get("median_rel_diff"), "direction": decision_y.get("direction"),
        })
    asymmetry_by_year = pd.DataFrame(asym_year_rows)

    asymmetry_decision = decide_asymmetry_material(asymmetry_global["by_decile"])
    if not asymmetry_by_year.empty and np.isfinite(asymmetry_decision.get("median_rel_diff", np.nan)):
        global_sign = np.sign(asymmetry_decision["median_rel_diff"])
        year_signs = np.sign(asymmetry_by_year["median_rel_diff"].dropna())
        stable_fraction = float((year_signs == global_sign).mean()) if len(year_signs) else float("nan")
        asymmetry_decision["stable_across_years_fraction"] = stable_fraction
        asymmetry_decision["stable"] = bool(np.isfinite(stable_fraction) and stable_fraction >= ASYMMETRY_STABILITY_MIN_YEAR_FRACTION)
    else:
        asymmetry_decision["stable_across_years_fraction"] = float("nan")
        asymmetry_decision["stable"] = False

    garch_asymmetric_fit = None
    garch_asymmetric_diag = None
    if asymmetry_decision["material"] and asymmetry_decision["stable"]:
        garch_asymmetric_fit = fit_garch11(r_adj, dist="normal", asymmetric=True)
        garch_asymmetric_diag = residual_diagnostics(garch_asymmetric_fit["standardized_resid"], block_ids_adj)

    stage_timings = timer.finish()

    return TDA11Result(
        entry_gate=entry_gate, stop11_activated=stop11_activated, stop11_reason=stop11_reason,
        n_raw=len(r1m_pop), n_clock_adjusted=len(r_tilde_pop), n_workers=n_workers,
        benchmark_diagnostics=benchmark_diagnostics,
        garch_primary_global=garch_primary_global, garch_primary_global_diag=garch_primary_global_diag,
        garch_secondary_raw_global=garch_secondary_raw_global, garch_secondary_raw_global_diag=garch_secondary_raw_global_diag,
        garch_student_t_sensitivity=garch_student_t_sensitivity,
        garch_by_year=garch_by_year, garch_by_segment=garch_by_segment,
        garch_usefulness=garch_usefulness,
        asymmetry_global=asymmetry_global, asymmetry_by_year=asymmetry_by_year, asymmetry_decision=asymmetry_decision,
        garch_asymmetric_fit=garch_asymmetric_fit, garch_asymmetric_diag=garch_asymmetric_diag,
        stage_timings=stage_timings,
    )
