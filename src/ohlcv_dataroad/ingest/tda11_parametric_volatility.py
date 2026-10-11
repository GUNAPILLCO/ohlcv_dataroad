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
   causales -- no una decision nueva de esta etapa (su impacto se MIDE
   desde la auditoria correctiva, ver punto 11). El DIAGNOSTICO de
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

---------------------------------------------------------------------
AUDITORIA CORRECTIVA (posterior al primer cierre de TDA-11; cada punto
fue DEMOSTRADO y MEDIDO sobre el conjunto de investigacion antes de
cambiar nada -- ver el informe, secciones 5, 6, 8, 10 y 17)
---------------------------------------------------------------------

9.  **C2 -- convergencia VERIFICADA, no declarada.** El punto 8 no
    bastaba: sobre los datos reales, los 4 intentos de la grilla de
    `eps` reportaron `convergence_flag=0` terminando en puntos MUY
    distintos (log-verosimilitud separada por decenas de miles de
    unidades). El ajuste PRIMARIO y los de año/segmento si coincidian
    con el optimo; el SECUNDARIO (`raw`) y la sensibilidad Student-t NO
    (69.663 y 11.558 unidades de log-verosimilitud por debajo del
    optimo real, con `alpha`/`beta` materialmente distintos). Desde esta
    auditoria `fit_garch11` (a) refina el mejor intento de `arch` con
    un optimizador independiente sobre la MISMA verosimilitud y la
    MISMA restriccion (`_refine_garch`), (b) exige que esa
    verosimilitud coincida con la de `arch` en el punto final
    (`GARCH_LLF_CROSSCHECK_RTOL`), y (c) declara un ajuste
    `converged_verified` SOLO si el gradiente proyectado (condicion de
    primer orden, KKT) de la log-verosimilitud media es menor que
    `GARCH_KKT_GRADIENT_TOL`. Un ajuste con parametros finitos pero sin
    esa verificacion NUNCA se usa: el primario detiene la etapa
    (`GARCHFitError`), un subperiodo queda `NOT_CONVERGED`.
10. **C1 -- persistencia en el borde de la restriccion.** `arch` impone
    `alpha+beta<=1`. Cuando el estimador queda en ese borde
    (`at_constraint_boundary=True`), el valor `1.0000` es el TOPE que
    permite el software, no una estimacion libre: no demuestra IGARCH
    ni memoria infinita, y los IC individuales de `alpha`/`beta`
    (calculados sin tener en cuenta la restriccion activa) son solo
    orientativos.
11. **H1 -- continuidad temporal.** La concatenacion del punto 4 se
    CONSERVA como procedimiento primario (medido: 0,16 % de los pares
    adyacentes de la poblacion global cruzan un hueco/sesion/contrato).
    Se añade una sensibilidad predeclarada con dos tratamientos
    alternativos de cada frontera de bloque
    (`BOUNDARY_SENSITIVITY_MODES`): `reset` (la varianza condicional
    olvida todo y vuelve a la varianza muestral -- extremo opuesto a
    concatenar) y `gap_forecast` (la varianza se propaga a traves del
    hueco con el pronostico del propio GARCH a `g` minutos de reloj --
    el tratamiento coherente con el modelo para observaciones
    faltantes). Es DESCRIPTIVA: mide cuanto dependen `alpha`/`beta` de
    la convencion, no elige "el mejor" modelo ni predice nada.
12. **H2 -- comparacion sobre poblacion COMUN.** La decision de utilidad
    informativa se toma con GARCH y los 3 benchmarks restringidos a
    EXACTAMENTE las mismas filas (interseccion de `z` finitos), mismos
    bloques y mismos pares. La diferencia medida frente a la version
    anterior (cada serie sobre su propia poblacion) fue de 180 filas
    sobre 1,9 millones (`Q(60)` cambia <0,02 %): inmaterial, pero ahora
    es igual por construccion. Se añade ademas una sensibilidad
    POST-HOC, claramente rotulada, contra un EWMA cuya velocidad de
    memoria se iguala a la `beta` estimada por GARCH: NO interviene en
    la regla predeclarada de STOP-11 (los benchmarks y umbrales no se
    tocan despues de ver el resultado), solo informa QUE aporta GARCH.
13. **C3 -- `Q(m)` no estimable.** Un GARCH puede ser estimable en un
    segmento cuyos bloques de continuidad duran <= `m` barras, donde
    `Q(m)` NO lo es. `residual_diagnostics` devuelve
    `q_abs_z_m_status` explicito; nunca se rellena ni se sustituye por
    un `Q` de menor alcance.

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
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln
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
    "GARCH_REFINE_STARTS",
    "GARCH_KKT_GRADIENT_TOL",
    "GARCH_LLF_CROSSCHECK_RTOL",
    "BOUNDARY_SENSITIVITY_MODES",
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
    "garch_backcast",
    "garch_sigma2_path",
    "garch_mean_neg_loglik",
    "kkt_projected_gradient",
    "kkt_check_garch",
    "compute_gap_minutes",
    "fit_garch11",
    "fit_garch11_boundary_sensitivity",
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

# --- Auditoria correctiva C2 (docstring de modulo, punto 9) -------------
# Arranques ADICIONALES `(alpha, alpha+beta)` del refinamiento
# independiente -- grilla pequeña y fija; se suman al mejor intento de
# `arch`. Gana la menor log-verosimilitud negativa (criterio objetivo).
GARCH_REFINE_STARTS: tuple[tuple[float, float], ...] = ((0.05, 0.98), (0.10, 0.95), (0.20, 0.995))
# Un ajuste es `converged_verified` solo si el mayor componente (en
# valor absoluto) del gradiente PROYECTADO de la log-verosimilitud
# negativa MEDIA (por observacion) es menor que este umbral. Fijado con
# la evidencia de la auditoria, en el hueco entre las dos poblaciones
# observadas y lejos de ambas: optimos reales <= 6e-5 (incluidos casos
# sinteticos IID con verosimilitud plana); falsos "exitos" de SLSQP
# >= 2,9e-2. NO es un umbral de decision cientifica (no interviene en
# STOP-11/TH23-25): es un control de validez numerica.
GARCH_KKT_GRADIENT_TOL = 1e-3
GARCH_KKT_STEP = 1e-6
# La verosimilitud propia (usada para refinar) debe coincidir con la de
# `arch` en el punto final con esta tolerancia relativa (medido: <=2e-16).
GARCH_LLF_CROSSCHECK_RTOL = 1e-8
# Valor devuelto por la verosimilitud en un punto no admisible.
_INFEASIBLE = 1e10

# --- Auditoria correctiva H1 (docstring de modulo, punto 11) ------------
BOUNDARY_SENSITIVITY_MODES = ("reset", "gap_forecast")

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


def garch_backcast(x: np.ndarray) -> float:
    """Valor de pre-muestra de la varianza -- MISMA formula que `arch` (`GARCH.backcast`): media exponencial (peso 0,94) de los primeros `min(75, n)` cuadrados.

    Se replica (en vez de llamar a `arch`) para que `garch_sigma2_path`
    sea una funcion pura y testeable; `fit_garch11` comprueba en cada
    ajuste que la verosimilitud resultante coincide con la de `arch`.
    """
    x = np.asarray(x, dtype=float)
    tau = min(75, x.size)
    w = 0.94 ** np.arange(tau)
    w = w / w.sum()
    return float(np.sum(w * x[:tau]))


def compute_gap_minutes(timestamp) -> np.ndarray:
    """Minutos de reloj transcurridos entre cada fila y la ANTERIOR de la misma poblacion (`1.0` = contiguas; primera fila = `1.0`)."""
    ts = pd.to_datetime(pd.Series(np.asarray(timestamp))).to_numpy().astype("datetime64[ns]").astype(np.int64)
    gap = np.ones(ts.size, dtype=float)
    if ts.size > 1:
        gap[1:] = np.diff(ts) / 60e9
    return gap


def _block_starts(block_ids: np.ndarray) -> np.ndarray:
    b = np.asarray(block_ids)
    if b.size == 0:
        return np.array([], dtype=int)
    return np.flatnonzero(np.concatenate([[True], b[1:] != b[:-1]]))


def garch_sigma2_path(
    x: np.ndarray, omega: float, alpha: float, beta: float, gamma: float = 0.0, neg: np.ndarray | None = None,
    block_starts: np.ndarray | None = None, boundary: str = "carry", gap_minutes: np.ndarray | None = None,
) -> np.ndarray:
    """Varianza condicional GARCH(1,1)/GJR: ``sigma2_t = omega + (alpha + gamma*1[r_(t-1)<0])*x_(t-1) + beta*sigma2_(t-1)``, con `x = r^2`.

    `boundary` decide que ocurre en el PRIMER elemento de cada bloque de
    continuidad (`block_starts`, indices de inicio):

    - ``"carry"`` (procedimiento PRIMARIO de TDA-11, identico a `arch`
      sobre la serie concatenada): la recursion ignora la frontera -- la
      ultima barra del bloque anterior se trata como `t-1`.
    - ``"reset"``: la varianza vuelve a la varianza muestral (`mean(x)`,
      media cero) -- el modelo "olvida todo" en cada hueco. Extremo
      opuesto a ``"carry"``.
    - ``"gap_forecast"``: la varianza que ``"carry"`` asignaria se
      propaga `g-1` pasos adicionales con el pronostico del propio
      GARCH (`g = gap_minutes` en el inicio del bloque, minutos de reloj
      del hueco): ``vbar + p^(g-1)*(sigma2_carry - vbar)`` con
      ``vbar = omega/(1-p)``, `p` la persistencia; en el limite `p=1`,
      ``sigma2_carry + (g-1)*omega``. Trata los minutos del hueco como
      observaciones faltantes del mismo proceso.

    La recursion dentro de cada bloque es un filtro lineal de primer
    orden (`scipy.signal.lfilter`), exacto y sin bucle por observacion.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    bc = garch_backcast(x)
    eff = alpha * x if (neg is None or gamma == 0.0) else (alpha + gamma * neg) * x
    u = np.empty(n)
    u[0] = omega + (alpha + 0.5 * gamma + beta) * bc
    u[1:] = omega + eff[:-1]
    if boundary == "carry":
        return lfilter([1.0], [1.0, -beta], u)
    if boundary not in BOUNDARY_SENSITIVITY_MODES:
        raise ValueError(f"boundary desconocido: {boundary!r}")
    if boundary == "gap_forecast" and gap_minutes is None:
        raise ValueError("boundary='gap_forecast' requiere gap_minutes")

    y = np.empty(n)
    p = alpha + beta + 0.5 * gamma
    reset_value = float(x.mean())
    starts = np.array([0], dtype=int) if block_starts is None else np.asarray(block_starts, dtype=int)
    ends = np.append(starts[1:], n)
    prev = None
    for a, b in zip(starts, ends):
        if boundary == "reset":
            init = reset_value
        elif prev is None:
            init = u[0]
        else:
            carried = u[a] + beta * prev
            g = max(float(gap_minutes[a]) - 1.0, 0.0)
            if p < 1.0:
                vbar = omega / (1.0 - p)
                init = vbar + (p ** g) * (carried - vbar)
            else:
                init = carried + g * omega
        if b - a == 1:
            y[a] = init
        else:
            seg = u[a:b].copy()
            seg[0] = init
            y[a:b] = lfilter([1.0], [1.0, -beta], seg)
        prev = y[b - 1]
    return y


def _unpack_theta(theta: np.ndarray, asymmetric: bool, dist: str) -> tuple[float, float, float, float, float, float | None]:
    """``theta = [log10(omega), alpha, (gamma), persistencia, (nu)]`` -> ``(omega, alpha, gamma, beta, persistencia, nu)``.

    Se optimiza sobre la PERSISTENCIA (no sobre `beta`) para que la
    restriccion de `arch` (`alpha+beta+gamma/2<=1`) sea una cota simple
    de una coordenada, y sobre `log10(omega)` porque `omega` recorre
    varios ordenes de magnitud.
    """
    i = 2
    gamma = 0.0
    if asymmetric:
        gamma = float(theta[i])
        i += 1
    persistence = float(theta[i])
    i += 1
    nu = float(theta[i]) if dist == "t" else None
    alpha = float(theta[1])
    return 10.0 ** float(theta[0]), alpha, gamma, persistence - alpha - 0.5 * gamma, persistence, nu


def _theta_bounds(x: np.ndarray, asymmetric: bool, dist: str) -> list[tuple[float, float]]:
    """Cotas de `theta` -- las mismas regiones admisibles que `arch` (`alpha,beta>=0`, persistencia `<=1`, `nu` en `(2.05, 500)`)."""
    log_v = float(np.log10(x.mean()))
    bounds = [(log_v - 10.0, log_v + 1.0), (0.0, 1.0)]
    if asymmetric:
        bounds.append((-1.0, 2.0))
    bounds.append((0.0, 1.0))
    if dist == "t":
        bounds.append((2.05, 500.0))
    return bounds


def garch_mean_neg_loglik(
    theta: np.ndarray, x: np.ndarray, neg: np.ndarray | None = None, asymmetric: bool = False, dist: str = "normal",
    block_starts: np.ndarray | None = None, boundary: str = "carry", gap_minutes: np.ndarray | None = None,
) -> float:
    """Log-verosimilitud negativa MEDIA (por observacion) de GARCH(1,1)/GJR, media cero, innovacion gaussiana o Student-t estandarizada.

    Se trabaja con la MEDIA (no la suma) para que el objetivo sea de
    orden 1 sea cual sea `n` -- la suma, de orden `10^6-10^7` sobre el
    conjunto de investigacion, es lo que vuelve poco fiables las
    diferencias finitas del optimizador interno de `arch` (punto 8 del
    docstring de modulo). Devuelve `_INFEASIBLE` fuera de la region
    admisible (misma region que `arch`).
    """
    omega, alpha, gamma, beta, persistence, nu = _unpack_theta(np.asarray(theta, dtype=float), asymmetric, dist)
    return _garch_mean_neg_loglik_params(omega, alpha, gamma, beta, persistence, nu, x, neg, dist, block_starts, boundary, gap_minutes)


def _garch_mean_neg_loglik_params(
    omega: float, alpha: float, gamma: float, beta: float, persistence: float, nu: float | None,
    x: np.ndarray, neg: np.ndarray | None, dist: str,
    block_starts: np.ndarray | None = None, boundary: str = "carry", gap_minutes: np.ndarray | None = None,
) -> float:
    """Nucleo de `garch_mean_neg_loglik` sobre parametros EXPLICITOS -- lo comparten la parametrizacion `theta` (optimizacion) y la parametrizacion de caja `phi` (verificacion KKT, `kkt_check_garch`)."""
    if alpha < 0.0 or beta < 0.0 or alpha + gamma < 0.0 or persistence > 1.0 or (nu is not None and nu <= 2.05):
        return _INFEASIBLE
    s2 = garch_sigma2_path(x, omega, alpha, beta, gamma, neg, block_starts, boundary, gap_minutes)
    if not np.all(s2 > 0.0):
        return _INFEASIBLE
    if dist == "t":
        ll = (
            gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(np.pi * (nu - 2.0))
            - 0.5 * np.log(s2) - (nu + 1.0) / 2.0 * np.log1p(x / (s2 * (nu - 2.0)))
        )
    elif dist == "normal":
        ll = -0.5 * (np.log(2.0 * np.pi) + np.log(s2) + x / s2)
    else:
        raise ValueError(f"dist no soportada: {dist!r}")
    value = -float(np.mean(ll))
    return value if np.isfinite(value) else _INFEASIBLE


def kkt_projected_gradient(theta: np.ndarray, objective, bounds: list[tuple[float, float]], step: float = GARCH_KKT_STEP) -> np.ndarray:
    """Gradiente PROYECTADO de `objective` en `theta` (diferencias finitas) -- la condicion de primer orden de un minimo con restricciones.

    En un minimo interior todo el gradiente es ~0. Si una coordenada
    esta sobre una cota (o sobre el borde de la region admisible, donde
    `objective` devuelve `_INFEASIBLE`) y el objetivo solo mejoraria
    SALIENDO de ella, ese componente no viola la optimalidad y se
    proyecta a 0 (condicion de Karush-Kuhn-Tucker). Un componente
    grande que apunta hacia el INTERIOR significa que el optimizador se
    detuvo antes de tiempo -- exactamente lo que `convergence_flag=0` de
    SLSQP no detecta. Devuelve `NaN` en una coordenada si no se puede
    evaluar hacia ningun lado.

    ALCANCE MATEMATICO (2a revision de la auditoria correctiva): para
    restricciones de CAJA (`lo_i <= theta_i <= hi_i`, cada una sobre una
    sola coordenada) las condiciones KKT de un minimo son EXACTAMENTE
    "gradiente proyectado = 0": `df/dtheta_i = 0` en coordenadas libres,
    `>= 0` en una cota inferior activa, `<= 0` en una superior activa
    (los multiplicadores son esas mismas derivadas, y la calificacion de
    restricciones se cumple siempre porque los gradientes de las cotas
    activas son vectores canonicos distintos). Para una restriccion que
    ACOPLA coordenadas (p. ej. `beta = p - alpha >= 0`) esta comprobacion
    coordenada a coordenada es necesaria pero NO suficiente: no examina
    las direcciones que se deslizan a lo largo de la cara activa. Por eso
    la verificacion de un ajuste GARCH nunca se hace sobre `theta`
    directamente, sino mediante `kkt_check_garch`, que primero cambia a
    coordenadas en las que TODAS las restricciones son de caja.

    Una coordenada a menos de `step` de su cota se trata como activa.
    Es una condicion de PRIMER orden: acredita un punto estacionario
    restringido (candidato a minimo local), no un minimo global.
    """
    theta = np.asarray(theta, dtype=float)
    f0 = objective(theta)
    grad = np.zeros(theta.size)
    for i, (lo, hi) in enumerate(bounds):
        f_up = f_dn = _INFEASIBLE
        if theta[i] + step <= hi:
            up = theta.copy()
            up[i] = theta[i] + step
            f_up = objective(up)
        if theta[i] - step >= lo:
            dn = theta.copy()
            dn[i] = theta[i] - step
            f_dn = objective(dn)
        ok_up, ok_dn = f_up < _INFEASIBLE, f_dn < _INFEASIBLE
        if ok_up and ok_dn:
            grad[i] = (f_up - f_dn) / (2.0 * step)
        elif ok_dn:
            # No se puede subir (cota/borde superior activo): solo viola
            # la optimalidad si BAJAR mejora el objetivo (derivada > 0).
            grad[i] = max((f0 - f_dn) / step, 0.0)
        elif ok_up:
            # No se puede bajar (cota/borde inferior activo): solo viola
            # la optimalidad si SUBIR mejora el objetivo (derivada < 0).
            grad[i] = min((f_up - f0) / step, 0.0)
        else:
            grad[i] = np.nan
    return grad


def _theta_to_phi(theta: np.ndarray, asymmetric: bool, dist: str) -> np.ndarray:
    """`theta` -> coordenadas de caja `phi = [log10(omega), w, (v), p, (nu)]`, ver `kkt_check_garch`."""
    _, alpha, gamma, _, persistence, nu = _unpack_theta(np.asarray(theta, dtype=float), asymmetric, dist)
    arch_part = alpha + 0.5 * gamma                      # = w * p
    w = float(np.clip(arch_part / persistence, 0.0, 1.0)) if persistence > 0.0 else 0.0
    phi = [float(theta[0]), w]
    if asymmetric:
        phi.append(float(np.clip(alpha / (2.0 * arch_part), 0.0, 1.0)) if arch_part > 0.0 else 0.5)
    phi.append(persistence)
    if dist == "t":
        phi.append(nu)
    return np.asarray(phi, dtype=float)


def _phi_to_params(phi: np.ndarray, asymmetric: bool, dist: str) -> tuple[float, float, float, float, float, float | None]:
    """`phi` -> `(omega, alpha, gamma, beta, persistencia, nu)`; toda `phi` dentro de su caja da parametros admisibles."""
    i = 2
    w = float(phi[1])
    v = 0.5
    if asymmetric:
        v = float(phi[i])
        i += 1
    persistence = float(phi[i])
    i += 1
    nu = float(phi[i]) if dist == "t" else None
    arch_part = w * persistence
    alpha = 2.0 * arch_part * v if asymmetric else arch_part
    gamma = 2.0 * arch_part * (1.0 - 2.0 * v) if asymmetric else 0.0
    return 10.0 ** float(phi[0]), alpha, gamma, (1.0 - w) * persistence, persistence, nu


def kkt_check_garch(
    theta: np.ndarray, x: np.ndarray, neg: np.ndarray | None = None, asymmetric: bool = False, dist: str = "normal",
    block_starts: np.ndarray | None = None, boundary: str = "carry", gap_minutes: np.ndarray | None = None,
) -> np.ndarray:
    """Gradiente proyectado (KKT) de la log-verosimilitud negativa media, evaluado en coordenadas donde TODAS las restricciones son de caja.

    La region admisible de GARCH(1,1)/GJR (la misma que impone `arch`) es
    `omega>0`, `alpha>=0`, `alpha+gamma>=0`, `beta>=0`,
    `alpha+gamma/2+beta<=1`: en `(alpha, beta)` o en `theta` hay
    restricciones que acoplan parametros. Con

        p = alpha + gamma/2 + beta          (persistencia)      en [0, 1]
        w = (alpha + gamma/2) / p           (parte ARCH de p)   en [0, 1]
        v = alpha / (2*alpha + gamma)       (solo GJR)          en [0, 1]

    cada restriccion pasa a ser la cota de UNA coordenada:
    `beta>=0 <=> w<=1`, `alpha>=0 <=> v>=0` (y `w>=0`),
    `alpha+gamma>=0 <=> v<=1`, `alpha+gamma/2+beta<=1 <=> p<=1`. El
    cambio de variables es suave e invertible para `p>0` (y `w>0` en
    GJR), asi que un punto es KKT del problema original si y solo si lo
    es en `phi` -- donde `kkt_projected_gradient` es exacto (ver su
    docstring). En los bordes degenerados `p=0` o `w=0` alguna coordenada
    deja de estar identificada y su derivada es 0: no produce un falso
    rechazo. `omega` solo tiene las cotas amplias de busqueda (10 ordenes
    de magnitud por debajo y 1 por encima de la varianza muestral), que
    no son restricciones del modelo; si quedaran activas el punto se
    trataria igualmente con la regla de cota.
    """
    phi = _theta_to_phi(theta, asymmetric, dist)
    log_v = float(np.log10(x.mean()))
    bounds = [(log_v - 10.0, log_v + 1.0), (0.0, 1.0)] + ([(0.0, 1.0)] if asymmetric else []) + [(0.0, 1.0)]
    if dist == "t":
        bounds.append((2.05, 500.0))

    def objective_phi(p_: np.ndarray) -> float:
        omega, alpha, gamma, beta, persistence, nu = _phi_to_params(p_, asymmetric, dist)
        return _garch_mean_neg_loglik_params(omega, alpha, gamma, beta, persistence, nu, x, neg, dist, block_starts, boundary, gap_minutes)

    return kkt_projected_gradient(phi, objective_phi, bounds)


def _refine_garch(
    x: np.ndarray, neg: np.ndarray | None, asymmetric: bool, dist: str, extra_starts: tuple = (),
    block_starts: np.ndarray | None = None, boundary: str = "carry", gap_minutes: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Minimiza `garch_mean_neg_loglik` -- multi-arranque L-BFGS-B (cotas) + pulido Nelder-Mead (sin derivadas) -- y devuelve `(theta, objetivo, gradiente proyectado)`.

    Determinista (sin aleatoriedad). Arranques: `extra_starts` (el mejor
    intento de `arch`, si existe) + `GARCH_REFINE_STARTS`. Nunca devuelve
    un punto peor que el mejor arranque. El gradiente proyectado se
    calcula con `kkt_check_garch` (coordenadas de caja), no sobre `theta`.
    """
    bounds = _theta_bounds(x, asymmetric, dist)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    def objective(theta: np.ndarray) -> float:
        return garch_mean_neg_loglik(theta, x, neg, asymmetric, dist, block_starts, boundary, gap_minutes)

    def objective_box(theta: np.ndarray) -> float:
        if np.any(theta < lo) or np.any(theta > hi):
            return _INFEASIBLE
        return objective(theta)

    log_v = float(np.log10(x.mean()))
    candidates = [np.clip(np.asarray(t, dtype=float), lo, hi) for t in extra_starts]
    for alpha0, p0 in GARCH_REFINE_STARTS:
        theta0 = [log_v + np.log10(1.0 - p0), alpha0] + ([0.0] if asymmetric else []) + [p0] + ([8.0] if dist == "t" else [])
        candidates.append(np.asarray(theta0, dtype=float))

    best_theta, best_value = None, np.inf
    for theta0 in candidates:
        value0 = objective(theta0)
        if value0 < best_value:
            best_theta, best_value = theta0, value0
        res = minimize(objective, theta0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500, "ftol": 1e-15, "gtol": 1e-10})
        if np.isfinite(res.fun) and res.fun < best_value:
            best_theta, best_value = np.clip(res.x, lo, hi), float(res.fun)
    if best_theta is None or best_value >= _INFEASIBLE:
        raise GARCHFitError("El refinamiento independiente no encontro ningun punto admisible con log-verosimilitud finita.")

    for _ in range(3):
        nm = minimize(objective_box, best_theta, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 4000, "maxfev": 4000})
        gain = best_value - float(nm.fun)
        if nm.fun < best_value:
            best_theta, best_value = np.asarray(nm.x, dtype=float), float(nm.fun)
        if gain < 1e-12:
            break

    return best_theta, best_value, kkt_check_garch(best_theta, x, neg, asymmetric, dist, block_starts, boundary, gap_minutes)


def _arch_multistart(am) -> tuple[object | None, float | None, int, bool]:
    """Etapa A (punto 8 del docstring de modulo): grilla de `eps` sobre `arch_model(...).fit()`. Devuelve `(mejor intento, eps, n intentos finitos, converged_cleanly)`.

    Preferencia: entre los intentos con `convergence_flag==0`, el de
    MENOR log-verosimilitud negativa; solo si NINGUNO reporto exito se
    recurre al mejor entre todos los finitos. Un intento con
    `convergence_flag!=0` nunca se prefiere sobre uno que si lo reporto.
    Desde la auditoria correctiva este resultado es solo un ARRANQUE del
    refinamiento (`_refine_garch`): `convergence_flag==0` NO acredita
    convergencia (punto 9 del docstring de modulo).
    """
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
        return None, None, 0, False
    clean = [t for t in finite_attempts if int(t[2].convergence_flag) == 0]
    pool = clean if clean else finite_attempts
    _, best_eps, best = min(pool, key=lambda t: t[0])
    return best, best_eps, len(finite_attempts), bool(clean)


def fit_garch11(
    r: np.ndarray, dist: str = "normal", scale: float = GARCH_SCALE_FACTOR, asymmetric: bool = False,
) -> dict:
    """Ajusta GARCH(1,1) (o GJR-GARCH(1,1,1) si `asymmetric=True`) sobre `r` -- media CERO (TDA-08: dependencia en media despreciable, STOP-8a).

    `r` se reescala por `scale` antes de ajustar (punto 5 del docstring
    de modulo) -- `alpha`/`beta`/`gamma`/persistencia se reportan SIN
    escala (invariantes); `omega` se des-escala dividiendo por
    `scale**2`. Inferencia ROBUSTA (sandwich Bollerslev-Wooldridge)
    siempre, sea cual sea `dist`.

    Tres pasos (puntos 8 y 9 del docstring de modulo):

    A. `_arch_multistart`: grilla de `eps` sobre el optimizador de
       `arch` (SLSQP). Auditoria correctiva: sobre los datos reales los
       4 intentos reportaron exito en puntos distintos, y en 2 de los 3
       ajustes globales NINGUNO era el optimo -- por eso este paso ya
       solo aporta un ARRANQUE.
    B. `_refine_garch`: optimizador independiente sobre la MISMA
       verosimilitud y la MISMA restriccion (`alpha+beta(+gamma/2)<=1`),
       con el objetivo normalizado por `n`.
    C. Resultado e inferencia desde `arch` en el punto final
       (`am.fix`, `am.compute_param_cov(robust=True)`), exigiendo que la
       verosimilitud propia coincida con la de `arch`
       (`GARCH_LLF_CROSSCHECK_RTOL`) -- `arch` sigue siendo la
       referencia del modelo; solo se sustituyo la busqueda del optimo.

    Salida: `dict` con `omega`/`alpha`/`beta`/`gamma`/`nu`, IC 95%
    robustos, `persistence`, `half_life`, `stationary`, `n`,
    `loglikelihood`, `standardized_resid` (misma longitud que `r`),
    `dist`, `asymmetric`, y los campos de validez numerica:

    - `converged_verified`: UNICO campo que acredita convergencia --
      gradiente proyectado maximo `< GARCH_KKT_GRADIENT_TOL`.
    - `kkt_max_abs_gradient`: ese gradiente.
    - `at_constraint_boundary`: la persistencia quedo en el tope
      `alpha+beta<=1` que impone `arch` (punto 10 del docstring).
    - `loglikelihood_arch_multistart`, `refinement_llf_gain`: cuanto
      mejoro el paso B sobre el paso A (~0 si `arch` ya estaba en el
      optimo; grande si no).
    - `convergence_flag`, `converged_cleanly`, `eps_selected`,
      `n_multistart_attempts_finite/total`: lo que reporto el paso A --
      se conservan por trazabilidad, NO acreditan convergencia.

    Lanza `GARCHFitError` si la entrada no es finita, si no existe
    ningun punto admisible, o si la verosimilitud propia y la de `arch`
    discrepan.
    """
    if dist not in ("normal", "t"):
        raise ValueError(f"dist no soportada: {dist!r}")
    r = np.asarray(r, dtype=float)
    n = r.size
    if n == 0 or not np.isfinite(r).all():
        raise GARCHFitError(f"La serie de entrada esta vacia o contiene valores no finitos (n={n}).")
    y = r * scale
    x = y ** 2
    if not float(x.mean()) > 0.0:
        raise GARCHFitError("La serie de entrada tiene varianza nula.")
    o = 1 if asymmetric else 0
    am = arch_model(y, mean="Zero", vol="GARCH", p=1, o=o, q=1, dist=dist)

    # --- A: multi-arranque de arch (solo un arranque del refinamiento) ---
    arch_best, best_eps, n_finite, converged_cleanly = _arch_multistart(am)
    extra_starts = []
    llf_arch_multistart = float("nan")
    if arch_best is not None:
        pa = arch_best.params
        gamma_arch = float(pa["gamma[1]"]) if asymmetric else 0.0
        theta_arch = [np.log10(float(pa["omega"])), float(pa["alpha[1]"])]
        if asymmetric:
            theta_arch.append(gamma_arch)
        theta_arch.append(float(pa["alpha[1]"]) + float(pa["beta[1]"]) + 0.5 * gamma_arch)
        if dist == "t":
            theta_arch.append(float(pa["nu"]))
        extra_starts.append(theta_arch)
        llf_arch_multistart = float(arch_best.loglikelihood)

    # --- B: refinamiento independiente, misma verosimilitud/restriccion ---
    neg = (y < 0.0).astype(float) if asymmetric else None
    theta, mean_nll, kkt = _refine_garch(x, neg, asymmetric, dist, extra_starts=tuple(extra_starts))
    omega_s, alpha, gamma_v, beta, persistence, nu = _unpack_theta(theta, asymmetric, dist)
    kkt_max = float(np.max(np.abs(kkt))) if np.isfinite(kkt).all() else float("nan")
    converged_verified = bool(np.isfinite(kkt_max) and kkt_max < GARCH_KKT_GRADIENT_TOL)

    # --- C: resultado e inferencia desde arch en el punto final ----------
    names = ["omega", "alpha[1]"] + (["gamma[1]"] if asymmetric else []) + ["beta[1]"] + (["nu"] if dist == "t" else [])
    values = [omega_s, alpha] + ([gamma_v] if asymmetric else []) + [beta] + ([nu] if dist == "t" else [])
    arch_params = np.asarray(values, dtype=float)
    if not np.isfinite(arch_params).all():
        raise GARCHFitError(f"Ajuste GARCH produjo parametro(s) no finito(s): {dict(zip(names, values))}")
    fixed = am.fix(arch_params)
    llf = float(fixed.loglikelihood)
    llf_own = -mean_nll * n
    if not np.isfinite(llf) or abs(llf_own - llf) > GARCH_LLF_CROSSCHECK_RTOL * max(1.0, abs(llf)):
        raise GARCHFitError(
            f"La log-verosimilitud propia ({llf_own:.6f}) y la de arch ({llf:.6f}) discrepan en el punto final "
            f"(tolerancia relativa {GARCH_LLF_CROSSCHECK_RTOL}): el refinamiento no optimizo el mismo modelo que arch."
        )
    try:
        with np.errstate(invalid="ignore"):
            std_err = np.sqrt(np.diag(np.asarray(am.compute_param_cov(arch_params, robust=True), dtype=float)))
    except Exception:
        std_err = np.full(arch_params.size, np.nan)
    z975 = 1.959963984540054
    point = dict(zip(names, arch_params))
    half = dict(zip(names, z975 * std_err))

    def _get(name: str, descale_power: int = 0) -> tuple[float, float, float]:
        d = scale ** descale_power
        return float(point[name]) / d, float(point[name] - half[name]) / d, float(point[name] + half[name]) / d

    omega, omega_lo, omega_hi = _get("omega", descale_power=2)
    alpha, alpha_lo, alpha_hi = _get("alpha[1]")
    beta, beta_lo, beta_hi = _get("beta[1]")
    gamma = gamma_lo = gamma_hi = float("nan")
    if asymmetric:
        gamma, gamma_lo, gamma_hi = _get("gamma[1]")
        persistence = alpha + beta + gamma / 2.0
    else:
        persistence = alpha + beta

    standardized_resid = y / np.asarray(fixed.conditional_volatility, dtype=float)

    return {
        "n": n, "dist": dist, "asymmetric": bool(asymmetric),
        "omega": omega, "omega_ci_lo": omega_lo, "omega_ci_hi": omega_hi,
        "alpha": alpha, "alpha_ci_lo": alpha_lo, "alpha_ci_hi": alpha_hi,
        "beta": beta, "beta_ci_lo": beta_lo, "beta_ci_hi": beta_hi,
        "gamma": gamma, "gamma_ci_lo": gamma_lo, "gamma_ci_hi": gamma_hi,
        "nu": float(nu) if nu is not None else float("nan"),
        "persistence": persistence,
        "stationary": is_stationary(persistence),
        "half_life": compute_half_life(persistence),
        "at_constraint_boundary": bool(persistence >= 1.0 - STATIONARITY_EPS),
        "converged_verified": converged_verified,
        "kkt_max_abs_gradient": kkt_max,
        "loglikelihood": llf,
        "loglikelihood_arch_multistart": llf_arch_multistart,
        "refinement_llf_gain": float(llf - llf_arch_multistart) if np.isfinite(llf_arch_multistart) else float("nan"),
        "standardized_resid": standardized_resid,
        "convergence_flag": int(arch_best.convergence_flag) if arch_best is not None else -1,
        "eps_selected": best_eps,
        "n_multistart_attempts_finite": n_finite,
        "converged_cleanly": converged_cleanly,
        "n_multistart_attempts_total": len(GARCH_OPTIMIZER_EPS_GRID),
    }


def fit_garch11_boundary_sensitivity(
    r: np.ndarray, block_ids: np.ndarray, gap_minutes: np.ndarray, scale: float = GARCH_SCALE_FACTOR,
    lags: tuple[int, ...] = (1, 5, 20, 60), m_values: tuple[int, ...] = (20, 60),
) -> dict:
    """Sensibilidad H1 (punto 11 del docstring de modulo): reestima GARCH(1,1) gaussiano con cada tratamiento alternativo de frontera de `BOUNDARY_SENSITIVITY_MODES`.

    Mismo optimizador verificado que `fit_garch11` (paso B) y misma
    restriccion; solo cambia que hace la varianza condicional al
    empezar cada bloque de continuidad (`garch_sigma2_path`). Devuelve
    un `dict` plano con, por cada modo: `alpha_<modo>`, `beta_<modo>`,
    `persistence_<modo>`, `converged_verified_<modo>`,
    `rho1_abs_z_<modo>`, `q_abs_z_m_<modo>` (diagnostico de residuos con
    los MISMOS bloques). Si un modo no converge de forma verificada sus
    parametros se devuelven `NaN` -- nunca un numero no fiable.
    """
    r = np.asarray(r, dtype=float)
    y = r * scale
    x = y ** 2
    starts = _block_starts(block_ids)
    out: dict = {}
    for mode in BOUNDARY_SENSITIVITY_MODES:
        keys = [f"{k}_{mode}" for k in ("alpha", "beta", "persistence", "rho1_abs_z", "q_abs_z_m")]
        try:
            theta, _, kkt = _refine_garch(x, None, False, "normal", block_starts=starts, boundary=mode, gap_minutes=gap_minutes)
        except GARCHFitError:
            out.update({k: float("nan") for k in keys})
            out[f"converged_verified_{mode}"] = False
            continue
        omega_s, alpha, _, beta, persistence, _ = _unpack_theta(theta, False, "normal")
        verified = bool(np.isfinite(kkt).all() and float(np.max(np.abs(kkt))) < GARCH_KKT_GRADIENT_TOL)
        out[f"converged_verified_{mode}"] = verified
        if not verified:
            out.update({k: float("nan") for k in keys})
            continue
        s2 = garch_sigma2_path(x, omega_s, alpha, beta, 0.0, None, starts, mode, gap_minutes)
        diag = residual_diagnostics(y / np.sqrt(s2), block_ids, lags=lags, m_values=m_values)
        out.update(dict(zip(keys, (alpha, beta, persistence, diag["rho1_abs_z"], diag["q_abs_z_m"]))))
    return out


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

    Auditoria correctiva (H2, defecto LATENTE -- sin efecto sobre el
    conjunto de investigacion, donde los unicos `NaN` son el quemado
    contiguo del inicio): al compactar, una fila descartada DENTRO de un
    bloque dejaba "adyacentes" a sus dos vecinas, que en realidad estan
    a 2 barras -- un par de rezago 1 fabricado. Ahora cada fila
    descartada PARTE su bloque: nunca se forma un par a traves de una
    fila eliminada.

    Auditoria correctiva (C3): `q_abs_z_m_status` distingue
    `ESTIMABLE` de `NOT_ESTIMABLE` (ningun bloque de continuidad es mas
    largo que `m` barras, asi que no existe ningun par de rezago `m`).
    En ese caso `q_abs_z_m` es `NaN` y NUNCA se sustituye por un `Q` de
    menor alcance.

    Salida: `dict` con `acf_z` y `acf_z2` (`DataFrame`s de `compute_acf`),
    `portmanteau_z`/`portmanteau_z2` (`DataFrame`s de
    `compute_portmanteau_q` + columna `lb_pvalue`), `rho1_abs_z`
    (rezago 1 de `ACF(|z|)`), `q_abs_z_m` (el resumen usado por
    `decide_garch_usefulness`), `q_abs_z_m_status`,
    `q_abs_z_m_n_pairs` (pares validos en el rezago `m`),
    `max_estimable_lag` y `n_finite`.
    """
    z = np.asarray(z, dtype=float)
    block_ids = np.asarray(block_ids)
    max_lag = max(lags)
    finite = np.isfinite(z)

    z_compact = z[finite]
    _, block_int = np.unique(block_ids[finite], return_inverse=True)
    removed_before = np.cumsum(~finite)[finite]
    block_ids_compact = block_int.astype(np.int64) * (int(removed_before.max()) + 1 if removed_before.size else 1) + removed_before

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

    estimable_lags = acf_abs_z.loc[acf_abs_z["estimable"].astype(bool), "lag"] if not acf_abs_z.empty else pd.Series(dtype=int)
    max_estimable_lag = int(estimable_lags.max()) if len(estimable_lags) else 0
    m_lag_row = acf_abs_z.loc[acf_abs_z["lag"] == RESIDUAL_COMPARISON_M] if not acf_abs_z.empty else acf_abs_z
    q_n_pairs = int(m_lag_row["n_pairs"].iloc[0]) if not m_lag_row.empty else 0
    if np.isfinite(q_abs_z_m):
        q_status = "ESTIMABLE"
    elif q_row.empty:
        q_status = f"NOT_COMPUTED -- m={RESIDUAL_COMPARISON_M} no esta en m_values"
    else:
        q_status = (
            f"NOT_ESTIMABLE -- mayor rezago con pares validos dentro de un mismo bloque de continuidad = "
            f"{max_estimable_lag} < m = {RESIDUAL_COMPARISON_M}"
        )

    return {
        "acf_z": acf_z, "acf_abs_z": acf_abs_z, "acf_z2": acf_z2,
        "portmanteau_abs_z": port_abs_z, "portmanteau_z2": port_z2,
        "rho1_abs_z": rho1_abs_z, "q_abs_z_m": q_abs_z_m,
        "q_abs_z_m_status": q_status, "q_abs_z_m_n_pairs": q_n_pairs, "max_estimable_lag": max_estimable_lag,
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

    Auditoria correctiva: (C2) un ajuste sin `converged_verified` se
    devuelve `estimable=False` con `reason="NOT_CONVERGED ..."` -- sus
    parametros NO se reportan; (C3) `q_abs_z_m_status` acompaña siempre
    a `q_abs_z_m`; (H1) si se recibe `gap_minutes`, se añade la
    sensibilidad de tratamiento de frontera
    (`fit_garch11_boundary_sensitivity`).
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    group_label, r, block_ids, dist, scale, min_n, lags, m_values, gap_minutes = args
    if r.size < min_n:
        return {"group": group_label, "estimable": False, "n": int(r.size), "reason": f"n={r.size}<{min_n} (MIN_N_FOR_GARCH_FIT)"}
    try:
        fit = fit_garch11(r, dist=dist, scale=scale)
    except GARCHFitError as exc:
        return {"group": group_label, "estimable": False, "n": int(r.size), "reason": str(exc)}
    if not fit["converged_verified"]:
        return {
            "group": group_label, "estimable": False, "n": int(r.size),
            "reason": (
                f"NOT_CONVERGED -- gradiente proyectado maximo={fit['kkt_max_abs_gradient']:.3e} >= "
                f"{GARCH_KKT_GRADIENT_TOL} (GARCH_KKT_GRADIENT_TOL): parametros finitos pero convergencia NO verificada"
            ),
            "converged_verified": False, "kkt_max_abs_gradient": fit["kkt_max_abs_gradient"],
        }

    diag = residual_diagnostics(fit["standardized_resid"], block_ids, lags=lags, m_values=m_values)
    out = {k: v for k, v in fit.items() if k != "standardized_resid"}
    out["group"] = group_label
    out["estimable"] = True
    out["rho1_abs_z"] = diag["rho1_abs_z"]
    out["q_abs_z_m"] = diag["q_abs_z_m"]
    out["q_abs_z_m_status"] = diag["q_abs_z_m_status"]
    out["q_abs_z_m_n_pairs"] = diag["q_abs_z_m_n_pairs"]
    out["max_estimable_lag"] = diag["max_estimable_lag"]
    out["n_blocks"] = int(_block_starts(block_ids).size)
    if gap_minutes is not None:
        out.update(fit_garch11_boundary_sensitivity(r, block_ids, gap_minutes, scale=scale, lags=lags, m_values=m_values))
    return out


def fit_garch_by_group(
    groups: dict[str, np.ndarray], block_ids_by_group: dict[str, np.ndarray], dist: str = "normal",
    scale: float = GARCH_SCALE_FACTOR, min_n: int = MIN_N_FOR_GARCH_FIT, n_workers: int | None = None,
    lags: tuple[int, ...] = (1, 5, 20, 60), m_values: tuple[int, ...] = (20, 60),
    gap_minutes_by_group: dict[str, np.ndarray] | None = None,
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

    `gap_minutes_by_group` (opcional, `{label: minutos de reloj desde la
    fila anterior DEL MISMO grupo}`): si se pasa, cada fila de salida
    incluye la sensibilidad H1 de tratamiento de frontera.
    """
    if n_workers is None:
        n_workers = default_n_workers(len(groups))
    jobs = [
        (label, arr, block_ids_by_group[label], dist, scale, min_n, lags, m_values,
         None if gap_minutes_by_group is None else gap_minutes_by_group[label])
        for label, arr in groups.items()
    ]
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
    # Auditoria correctiva H2: mismos diagnosticos sobre la poblacion
    # COMUN (interseccion de z finitos de GARCH y los 3 benchmarks) --
    # {"raw": {"n_common", "garch", config...}, "clock_adjusted": {...}}.
    common_population_diagnostics: dict = field(default_factory=dict)
    # Auditoria correctiva H1: sensibilidad de frontera del primario global.
    garch_primary_boundary_sensitivity: dict = field(default_factory=dict)
    block_topology: dict = field(default_factory=dict)

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

    # Auditoria correctiva H1: topologia de huecos MEDIDA (no supuesta).
    gap_minutes_adj = compute_gap_minutes(r_tilde_pop["timestamp"])
    n_blocks_adj = int(_block_starts(block_ids_adj).size)
    block_topology = {
        "n": int(len(r_tilde_pop)), "n_blocks": n_blocks_adj,
        "non_contiguous_pair_fraction": float((n_blocks_adj - 1) / max(1, len(r_tilde_pop) - 1)),
    }

    if n_workers is None:
        n_workers = default_n_workers(1 + len(segment_labels))

    # --- Etapa 2: benchmarks (3 configuraciones x 2 bloques) ----------------
    timer.stage(f"Benchmarks causales (EWMA/rodante/rango) x raw+clock_adjusted")
    benchmark_diagnostics: dict[str, dict] = {"raw": {}, "clock_adjusted": {}}
    benchmark_z: dict[str, dict] = {"raw": {}, "clock_adjusted": {}}
    for cfg in BENCHMARK_CONFIGS:
        z_raw = compute_benchmark_z(r_raw, log_hl_raw, cfg)
        benchmark_z["raw"][cfg] = z_raw
        benchmark_diagnostics["raw"][cfg] = residual_diagnostics(z_raw, block_ids_raw)
        z_adj = compute_benchmark_z(r_adj, log_hl_tilde, cfg)
        benchmark_z["clock_adjusted"][cfg] = z_adj
        benchmark_diagnostics["clock_adjusted"][cfg] = residual_diagnostics(z_adj, block_ids_adj)

    # --- Etapa 3: GARCH global -- primario, secundario, sensibilidad t -----
    timer.stage("GARCH(1,1) global -- primario (clock_adjusted), secundario (raw), sensibilidad Student-t")
    garch_primary_global = fit_garch11(r_adj, dist="normal")
    if not garch_primary_global["converged_verified"]:
        # Auditoria correctiva C2: TH23/TH24 y STOP-11 descansan sobre
        # ESTE ajuste -- parametros finitos sin convergencia verificada
        # no se declaran validos; la etapa se detiene.
        raise GARCHFitError(
            f"El ajuste GARCH(1,1) PRIMARIO no supero la verificacion de convergencia (gradiente proyectado maximo="
            f"{garch_primary_global['kkt_max_abs_gradient']:.3e} >= {GARCH_KKT_GRADIENT_TOL}). TDA-11 no declara "
            "TH23/TH24/STOP-11 sobre un ajuste no fiable."
        )
    garch_primary_global_diag = residual_diagnostics(garch_primary_global["standardized_resid"], block_ids_adj)
    garch_primary_boundary_sensitivity = fit_garch11_boundary_sensitivity(
        r_adj, block_ids_adj, gap_minutes_adj, lags=DIAGNOSTIC_LAGS, m_values=PORTMANTEAU_M,
    )

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
    ts_adj = r_tilde_pop["timestamp"].to_numpy()
    year_gaps = {int(y): compute_gap_minutes(ts_adj[r_tilde_pop["year_ny"].to_numpy() == y]) for y in sorted(r_tilde_pop["year_ny"].unique())}
    segment_gaps = {s: compute_gap_minutes(ts_adj[r_tilde_pop["segment_label"].to_numpy() == s]) for s in segment_labels}
    garch_by_year = fit_garch_by_group(year_groups, year_block_ids, n_workers=n_workers, gap_minutes_by_group=year_gaps)
    garch_by_segment = fit_garch_by_group(segment_groups, segment_block_ids, n_workers=n_workers, gap_minutes_by_group=segment_gaps)

    # --- Etapa 5: utilidad informativa de GARCH + STOP-11 -------------------
    timer.stage("Utilidad informativa de GARCH frente a benchmarks (decision STOP-11)")
    # Auditoria correctiva H2: la decision se toma sobre la poblacion
    # COMUN -- GARCH y los 3 benchmarks restringidos a EXACTAMENTE las
    # mismas filas, bloques y pares. Los diagnosticos "cada serie sobre
    # su propia poblacion" se conservan y se reportan al lado.
    common_population_diagnostics: dict[str, dict] = {}
    for block_name, z_garch, block_ids_b in (
        ("clock_adjusted", garch_primary_global["standardized_resid"], block_ids_adj),
        ("raw", garch_secondary_raw_global["standardized_resid"], block_ids_raw),
    ):
        common = np.isfinite(z_garch)
        for z_b in benchmark_z[block_name].values():
            common &= np.isfinite(z_b)
        entry = {"n_common": int(common.sum()), "garch": residual_diagnostics(np.where(common, z_garch, np.nan), block_ids_b)}
        for cfg, z_b in benchmark_z[block_name].items():
            entry[cfg] = residual_diagnostics(np.where(common, z_b, np.nan), block_ids_b)
        common_population_diagnostics[block_name] = entry
    common_adj = common_population_diagnostics["clock_adjusted"]

    garch_usefulness = decide_garch_usefulness(
        common_adj["garch"], {cfg: common_adj[cfg] for cfg in BENCHMARK_CONFIGS}, garch_primary_global,
    )
    garch_usefulness["population"] = "common"
    garch_usefulness["n_common"] = common_adj["n_common"]
    own = decide_garch_usefulness(garch_primary_global_diag, benchmark_diagnostics["clock_adjusted"], garch_primary_global)
    garch_usefulness["garch_q_abs_z_m_own_population"] = own["garch_q_abs_z_m"]
    garch_usefulness["best_benchmark_q_abs_z_m_own_population"] = own["best_benchmark_q_abs_z_m"]
    garch_usefulness["relative_reduction_own_population"] = own["relative_reduction"]
    garch_usefulness["useful_own_population"] = own["useful"]

    # Sensibilidad POST-HOC de auditoria (punto 12 del docstring de
    # modulo) -- NO interviene en `useful` ni en STOP-11: EWMA causal
    # cuya velocidad de memoria (lambda) se iguala a la beta estimada
    # por el GARCH primario, sobre la misma poblacion comun.
    beta_primary = float(garch_primary_global["beta"])
    matched_halflife = matched_q = matched_rho1 = matched_reduction = float("nan")
    if 0.0 < beta_primary < 1.0:
        matched_halflife = float(np.log(0.5) / np.log(beta_primary))
        z_matched = standardize_return(r_adj, build_sigma_hat(r_adj, "ewma", matched_halflife), burn_in_length("ewma", matched_halflife))
        common_m = np.isfinite(garch_primary_global["standardized_resid"]) & np.isfinite(z_matched)
        for z_b in benchmark_z["clock_adjusted"].values():
            common_m &= np.isfinite(z_b)
        matched_diag = residual_diagnostics(np.where(common_m, z_matched, np.nan), block_ids_adj, lags=(1, 60), m_values=(RESIDUAL_COMPARISON_M,))
        garch_diag_m = residual_diagnostics(
            np.where(common_m, garch_primary_global["standardized_resid"], np.nan), block_ids_adj, lags=(1, 60), m_values=(RESIDUAL_COMPARISON_M,),
        )
        matched_q, matched_rho1 = matched_diag["q_abs_z_m"], matched_diag["rho1_abs_z"]
        if np.isfinite(matched_q) and matched_q > 0 and np.isfinite(garch_diag_m["q_abs_z_m"]):
            matched_reduction = float(1.0 - garch_diag_m["q_abs_z_m"] / matched_q)
    garch_usefulness["posthoc_matched_ewma_halflife"] = matched_halflife
    garch_usefulness["posthoc_matched_ewma_q_abs_z_m"] = matched_q
    garch_usefulness["posthoc_matched_ewma_rho1_abs_z"] = matched_rho1
    garch_usefulness["posthoc_relative_reduction_vs_matched_ewma"] = matched_reduction
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
        common_population_diagnostics=common_population_diagnostics,
        garch_primary_boundary_sensitivity=garch_primary_boundary_sensitivity,
        block_topology=block_topology,
        garch_primary_global=garch_primary_global, garch_primary_global_diag=garch_primary_global_diag,
        garch_secondary_raw_global=garch_secondary_raw_global, garch_secondary_raw_global_diag=garch_secondary_raw_global_diag,
        garch_student_t_sensitivity=garch_student_t_sensitivity,
        garch_by_year=garch_by_year, garch_by_segment=garch_by_segment,
        garch_usefulness=garch_usefulness,
        asymmetry_global=asymmetry_global, asymmetry_by_year=asymmetry_by_year, asymmetry_decision=asymmetry_decision,
        garch_asymmetric_fit=garch_asymmetric_fit, garch_asymmetric_diag=garch_asymmetric_diag,
        stage_timings=stage_timings,
    )
