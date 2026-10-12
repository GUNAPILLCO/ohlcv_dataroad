"""TDA-12 -- Cuantiles condicionales (TH26).

Implementa la etapa TDA-12 del roadmap
(``docs/methodology/Tsay_OHLCV_analysis_roadmap.md``, seccion "TDA-12"):
caracteriza como cambian los cuantiles de los retornos de MNQ con el
contexto observable (segmento horario, estado previo de volatilidad,
año) y responde la pregunta central de TH26:

    ¿La distribucion cambia SOLO EN ESCALA o tambien EN FORMA?

Es CARACTERIZACION ESTADISTICA. No diseña targets, features, señales ni
modelos predictivos; no ejecuta TDA-13 ni EVT; no abre ``data/raw/`` ni
``config.holdout_files``; no modifica ningun artefacto de TDA-00..11.

---------------------------------------------------------------------
QUE SE HEREDA SIN REDEFINIR (decisiones ya cerradas de etapas previas)
---------------------------------------------------------------------

- ``r_t`` = ``r_1m`` valido (TDA-04, ``r_1m_valid=True``): nunca cruza
  roll, jornada ni hueco. Poblacion construida con
  ``build_r1m_population`` (TDA-07), sin modificar.
- ``sigma_hat_(t-1)`` y ``z_t = r_t / sigma_hat_(t-1)``: el estimador
  PRIMARIO de TDA-10 (``PRIMARY_ESTIMATOR`` = EWMA causal, half-life 60
  barras, sobre ``r_1m``), construido con ``build_sigma_hat`` /
  ``standardize_return`` de TDA-10 sin modificar (mismo quemado, mismo
  piso ``MIN_VALID_SIGMA_HAT``). Su causalidad se RE-VERIFICA aqui con
  ``run_causality_checks`` (bloqueante).
- Niveles de cuantil: ``QUANTILE_LEVELS`` de TDA-07 (0.1, 1, 5, 25, 50,
  75, 95, 99, 99.9 %), que son exactamente los que pide el roadmap.
- Segmentos horarios: los de TDA-06 (``load_segmentation_cutoffs``).
- Bootstrap: la politica vigente desde TDA-07 y usada en TDA-08/09/10/11
  -- bloques = JORNADAS completas (``trading_date``) remuestreadas con
  reemplazo, ``n_boot`` configurable (300 en la configuracion real),
  semilla fija, IC percentil 95 %. Una jornada nunca se parte ni se
  empalma con otra: como todos los estadisticos de esta etapa (cuantiles,
  desviaciones) son independientes del ORDEN de las observaciones, el
  remuestreo no fabrica ninguna continuidad temporal entre sesiones,
  huecos o contratos.
- Umbrales de materialidad: ``PROFILE_STABILITY_SCALE_THRESHOLD`` (0.30)
  y ``PROFILE_STABILITY_FORM_THRESHOLD`` (0.60) de TDA-10, y su
  estadistico ``(max-min)/mediana`` de la semi-amplitud 1%-99% entre
  grupos; ``ROBUSTNESS_AGREEMENT_FRACTION`` (0.75) de TDA-10;
  ``MIN_N_FOR_STABILITY_GROUP`` (200) de TDA-10; el minimo ``n>=20`` y el
  recorte ``TRIM_FRACTION_EACH_TAIL`` de TDA-07. NINGUN valor se cambia.

---------------------------------------------------------------------
VERSION: COMPLETA (no minima)
---------------------------------------------------------------------

El arbol de decision del roadmap envia a "TDA-12 en version minima" solo
si TDA-10 concluye que la curtosis "desaparece casi toda"
(``ESCALA_DOMINA``). TDA-10 concluyo ``MIXTO`` (bloque RAW/CAUSAL,
acuerdo 0.50) y dejo ``STOP-13 = NO SUGERIDO`` con TH26 "formalmente
pendiente de TDA-12". Por tanto se ejecuta la version COMPLETA.

---------------------------------------------------------------------
DECISIONES METODOLOGICAS NUEVAS DE ESTA ETAPA
(predeclaradas aqui, ANTES de ejecutar sobre el conjunto de
investigacion; NO estaban aprobadas previamente -- se listan tambien en
el informe para revision)
---------------------------------------------------------------------

N1. **Deciles de volatilidad CAUSALES.** TDA-10 asigna el decil con
    ``pd.qcut`` sobre TODA la muestra: el valor ``sigma_hat_(t-1)`` es
    causal, pero las FRONTERAS de los deciles usan informacion futura.
    Como TDA-12 exige que el propio condicionante sea causal, el estado
    primario es ``VOL_DECILE_CAUSAL``: el decil de ``sigma_hat_(t-1)``
    respecto de la distribucion empirica de ``sigma_hat`` de las
    JORNADAS ANTERIORES unicamente (``causal_volatility_decile``). No
    introduce ningun parametro: reutiliza el minimo de historia que ya
    usa ``assign_volatility_decile`` (``n_deciles*20`` observaciones).
    Consecuencia inevitable: los grupos no tienen el mismo tamaño. La
    asignacion de TDA-10 se conserva como ``VOL_DECILE_FULLSAMPLE``
    (fronteras RETROSPECTIVAS) solo para comparar.
N2. **Objeto del veredicto formal = ``z_t``.** El roadmap pide los
    cuantiles de ``r_t`` y de ``z_t``; ambos se reportan. Pero ``r_t``
    dentro de un segmento o de un año mezcla estados de volatilidad
    distintos, y una mezcla de escalas cambia la forma aunque la
    distribucion condicional sea la misma. Lo que ``SOLO ESCALA``
    afirma (roadmap: "conocer sigma_hat determina los cuantiles") se
    contrasta sobre ``z_t`` (sintesis transversal, tabla §3: "Cuantiles
    de z_t comparados entre grupos"). Las comparaciones sobre ``r_t`` son
    contexto descriptivo, nunca deciden.
N3. **Normalizacion y su robustez.** El roadmap dice "dividir cada
    conjunto de cuantiles por la desviacion del grupo". Con colas tan
    pesadas (curtosis de ``r`` ~100) la desviacion estandar la dominan
    unos pocos puntos; por eso cada comparacion se calcula con DOS
    escalas: la desviacion estandar (roadmap) y la desviacion estandar
    RECORTADA 0.1 % (misma convencion de recorte de TDA-07/TDA-10). Una
    comparacion solo recibe una etiqueta firme si AMBAS coinciden.
N4. **Regla de veredicto** (``decide_th26``), sobre el estadistico y los
    umbrales de TDA-10 aplicados a las tres agrupaciones causales
    (segmento, decil causal, año) de ``z_t``:
      - cada comparacion es ``SUPERPUESTO`` (ratio e IC 95 % completos
        <= 0.30), ``DIVERGENTE`` (>= 0.60) o ``INTERMEDIO`` (IC completo
        entre ambos), con ambas normalizaciones; si el IC cruza un
        umbral o las normalizaciones discrepan -> ``NO_CONCLUYENTE``;
      - ``ESCALA_Y_FORMA`` si ALGUNA comparacion es ``DIVERGENTE``;
        ``SOLO_ESCALA`` si TODAS son ``SUPERPUESTO``; en cualquier otro
        caso ``NO_CONCLUYENTE`` (no se fuerza una conclusion);
      - robustez (misma regla que TDA-10): si menos del 75 % de las 6
        configuraciones causales de ``sigma_hat`` de TDA-10 reproduce la
        misma conclusion puntual, el veredicto se degrada a
        ``NO_CONCLUYENTE``.
    Los umbrales son los de TDA-10 (unica escala de materialidad
    documentada para este estadistico); no se crea ninguno. El nivel
    1%-99% es el que TDA-10 definio; los demas niveles (0.1/99.9, 5/95,
    25/75) se reportan sin umbral.
N5. **Soporte de los cuantiles extremos.** Para cada cuantil se reporta
    ``n_tail`` (observaciones en o mas alla del cuantil) y
    ``n_days_tail`` (jornadas distintas que las aportan -- el tamaño de
    muestra efectivo es menor que ``n_tail`` porque los extremos se
    agrupan). La etiqueta ``support`` es DESCRIPTIVA y reutiliza dos
    constantes existentes: ``NO_FIABLE`` (< 20, minimo de TDA-07),
    ``BAJO`` (< 200, ``MIN_N_FOR_STABILITY_GROUP``), ``OK``. Un grupo
    solo participa en una comparacion de forma si su cola en ese nivel
    tiene soporte ``OK``.

---------------------------------------------------------------------
CAUSAL vs RETROSPECTIVO (misma disciplina que TDA-10/11)
---------------------------------------------------------------------

- Bloque ``raw`` (``r_1m``, ``z`` con EWMA sobre ``r_1m``, decil causal):
  responde la pregunta FORMAL. Matices declarados: (a) los cortes de los
  segmentos horarios son los de TDA-06, elegidos con toda la muestra --
  la etiqueta de segmento de una barra es reloj puro, pero DONDE se
  corta es retrospectivo; (b) la desviacion de grupo usada para
  normalizar es un estadistico descriptivo de la muestra completa del
  grupo (asi lo define el roadmap): describe, no esta disponible ex ante.
- Bloque ``clock_adjusted`` (``r_tilde`` y su ``z``): diagnostico
  SECUNDARIO y RETROSPECTIVO (``s(m)`` de TDA-06 usa toda la muestra).
  Nunca decide ni se promedia con el bloque ``raw``.

---------------------------------------------------------------------
AUDITORIA CORRECTIVA (posterior a la primera implementacion)
---------------------------------------------------------------------

Correcciones TECNICAS (no cambian la regla de veredicto ni ningun
umbral):

A1. **Grupos sin normalizacion valida.** Un grupo con soporte de cola
    suficiente pero desviacion nula (o no finita) producia una
    semi-amplitud normalizada `NaN` que `profile_ratio` ignoraba en
    silencio, mientras `n_groups` lo seguia contando. Ahora cada
    comparacion distingue grupos DISPONIBLES, EXCLUIDOS POR SOPORTE,
    SIN NORMALIZACION VALIDA y COMPARADOS (`classify_groups`); el ratio
    se calcula solo con los comparados, exige que TODOS sean finitos en
    cada replica, y si algun grupo con soporte carece de normalizacion
    valida la comparacion es `NO_CONCLUYENTE` (no se concluye sobre "los
    grupos" dejando uno fuera sin poder compararlo).
A2. **La verificacion de causalidad del decil usa el conjunto
    efectivo.** `verify_causal_decile_no_lookahead` recibe la misma
    mascara `valid` con la que se construye el decil, y añade una
    prueba de TRUNCAMIENTO (recalcular sin el futuro) a la de
    perturbacion.
A3. **El estadistico del veredicto es ciego a la asimetria** (hallazgo
    de la revision independiente, reproducido): la semi-amplitud
    `(q_hi-q_lo)/2` es identica para una distribucion y su reflejo. Dos
    grupos con la misma amplitud y asimetrias opuestas salian
    `SUPERPUESTO`. NO se cambia la regla (seria una decision nueva);
    se añaden, como evidencia DESCRIPTIVA, las comparaciones de CADA
    COLA por separado (`left_ratio`, `right_ratio`: el mismo ratio sobre
    `|q_lo|/escala` y sobre `q_hi/escala`) y el rango de asimetria entre
    grupos, y se calcula que diria una regla que exigiera superposicion
    en ambas colas (`evaluate_one_sided_proposal`) -- una PROPUESTA
    pendiente de aprobacion que no interviene en `decide_th26`.

Precisiones sobre las decisiones N3-N5 (la reutilizacion de una
constante NO equivale a que su nuevo uso estuviera aprobado):

- Los umbrales 0.30/0.60 se aprobaron en TDA-10 para el ratio de la
  semi-amplitud 1%-99% de `z` SIN dividir por la desviacion del grupo
  (normalizacion ``"none"`` de este modulo, que reproduce exactamente
  `segment_stability_ratio`/`decile_stability_ratio` de TDA-10).
  Aplicarlos al ratio dividido por la desviacion del grupo (como pide el
  roadmap de TDA-12) es una EXTENSION decidida aqui.
- `MIN_N_FOR_STABILITY_GROUP=200` era en TDA-10 un minimo de TAMAÑO DE
  GRUPO. Exigirlo a la COLA (decision N5) es mas estricto y es nuevo.
  `build_alternative_readings` reporta que ocurre con el criterio
  literal de TDA-10.
- La comprobacion de robustez de N4 compara conclusiones PUNTUALES de
  las otras configuraciones con un veredicto basado en IC: no compara
  cosas equivalentes. Se conserva (solo puede degradar) y se reporta al
  lado la comparacion homogenea (puntual contra puntual).

Sensibilidad nueva, DESCRIPTIVA: bootstrap con bloques de varias
jornadas CONSECUTIVAS (`BLOCK_LENGTH_SENSITIVITY_DAYS`); el bootstrap
aprobado (una jornada) sigue siendo el que decide.

---------------------------------------------------------------------
CIERRE DOCUMENTAL (alcance de las conclusiones; no cambia calculos)
---------------------------------------------------------------------

- ``SUPERPUESTO`` describe la estabilidad de la semi-amplitud
  normalizada entre grupos. NO demuestra la igualdad del perfil
  distribucional completo.
- Son decisiones metodologicas NUEVAS y EXPLORATORIAS de esta etapa (el
  veredicto se lee condicionado a ellas): (i) aplicar los umbrales
  0.30/0.60 de TDA-10 al estadistico NORMALIZADO; (ii) exigir 200
  observaciones EN LA COLA, en vez de 200 totales por grupo; (iii) la
  doble normalizacion y la regla de robustez que compara conclusiones
  puntuales con conclusiones basadas en IC; (iv) ``NO_CONCLUYENTE`` como
  salida prudente cuando no puede sostenerse ninguno de los dos
  veredictos que preve el roadmap.
- La comparacion de colas separadas (`left_ratio`, `right_ratio`,
  `evaluate_one_sided_proposal`) es evidencia DESCRIPTIVA. No forma
  parte del veredicto formal y no debe incorporarse a `decide_th26` sin
  aprobacion especifica.
- Alcance de las pruebas de causalidad en `run_tda12_analysis`:
  `run_causality_checks` se re-ejecuta SOLO para el estimador primario
  (los dos bloques comparten familia y parametro, asi que es una unica
  configuracion, con `r_1m` como entrada); el decil causal se verifica
  por muestreo de barras para el estimador primario de cada bloque. Las
  demas configuraciones de `build_estimator_sensitivity` no se
  re-verifican aqui: su causalidad descansa en la verificacion de TDA-10.

Unidades (G5): cada cuantil se reporta con su IC bootstrap y traducido
a multiplos de la desviacion del grupo. No se reporta ningun p-valor.
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
    TRIM_FRACTION_EACH_TAIL,
    build_r1m_population,
    build_r_tilde_population,
    load_r_tilde,
    load_segmentation_cutoffs,
    load_tda04_inputs,
    verify_r_tilde_invariants,
    verify_timestamp_alignment,
)
from ohlcv_dataroad.ingest.tda10_scale_vs_shape import (
    ALL_ESTIMATOR_CONFIGS,
    MIN_N_FOR_STABILITY_GROUP,
    PRIMARY_ESTIMATOR,
    PRIMARY_ESTIMATOR_CLOCK_ADJUSTED,
    PROFILE_STABILITY_FORM_THRESHOLD,
    PROFILE_STABILITY_SCALE_THRESHOLD,
    ROBUSTNESS_AGREEMENT_FRACTION,
    VOLATILITY_DECILES,
    assign_volatility_decile,
    build_sigma_hat,
    burn_in_length,
    run_causality_checks,
    standardize_return,
    verify_populations_aligned,
)

__all__ = [
    "QUANTILE_LEVELS",
    "SYMMETRIC_LEVEL_PAIRS",
    "VERDICT_LEVEL_PAIR",
    "NORMALIZATIONS",
    "GROUPINGS",
    "VERDICT_GROUPINGS",
    "MIN_N_FOR_TRIM",
    "MIN_TAIL_SUPPORT_UNRELIABLE",
    "MIN_TAIL_SUPPORT_LOW",
    "CI_LEVEL",
    "DEFAULT_SEED",
    "LABEL_OVERLAP",
    "LABEL_INTERMEDIATE",
    "LABEL_DIVERGENT",
    "LABEL_INCONCLUSIVE",
    "VERDICT_SCALE_ONLY",
    "VERDICT_SCALE_AND_SHAPE",
    "VERDICT_INCONCLUSIVE",
    "CausalDecileLeakageError",
    "causal_volatility_decile",
    "verify_causal_decile_no_lookahead",
    "weighted_group_stats",
    "prepare_sorted_groups",
    "joint_day_block_bootstrap",
    "tail_support",
    "support_label",
    "build_quantile_table",
    "profile_ratio",
    "classify_ratio",
    "classify_groups",
    "build_shape_comparison_table",
    "evaluate_one_sided_proposal",
    "build_alternative_readings",
    "build_block_length_sensitivity",
    "BLOCK_LENGTH_SENSITIVITY_DAYS",
    "build_asymmetry_table",
    "build_estimator_sensitivity",
    "decide_th26",
    "decide_stop13_implication",
    "ANALYSIS_STAGES",
    "TOTAL_STAGES",
    "TDA12Result",
    "run_tda12_analysis",
]

# ---------------------------------------------------------------------------
# Convenciones PREDECLARADAS (ver docstring de modulo)
# ---------------------------------------------------------------------------

# Pares simetricos (cola izquierda, cola derecha) de QUANTILE_LEVELS.
SYMMETRIC_LEVEL_PAIRS: tuple[tuple[float, float], ...] = ((0.001, 0.999), (0.01, 0.99), (0.05, 0.95), (0.25, 0.75))
# Nivel del estadistico de TDA-10 (`extreme_scale_from_table`: q0.01/q0.99)
# -- el UNICO al que se aplican sus umbrales.
VERDICT_LEVEL_PAIR: tuple[float, float] = (0.01, 0.99)
assert all(lo in QUANTILE_LEVELS and hi in QUANTILE_LEVELS for lo, hi in SYMMETRIC_LEVEL_PAIRS)
assert VERDICT_LEVEL_PAIR in SYMMETRIC_LEVEL_PAIRS

# Escalas de normalizacion (decision N3). "none" = sin normalizar: solo
# tiene sentido para z (ya en unidades de sigma_hat) y es descriptiva.
NORMALIZATIONS: tuple[str, ...] = ("std", "std_trimmed")

GROUPINGS: tuple[str, ...] = ("GLOBAL", "SEGMENT", "VOL_DECILE_CAUSAL", "VOL_DECILE_FULLSAMPLE", "YEAR")
# Agrupaciones que deciden el veredicto: las tres que pide el roadmap,
# con el estado de volatilidad CAUSAL (decision N1).
VERDICT_GROUPINGS: tuple[str, ...] = ("SEGMENT", "VOL_DECILE_CAUSAL", "YEAR")

# Reutilizados sin cambiar su valor (decision N5).
MIN_N_FOR_TRIM = 20                                   # TDA-07: recorte no interpretable con n<20
MIN_TAIL_SUPPORT_UNRELIABLE = 20                      # mismo minimo de TDA-07
MIN_TAIL_SUPPORT_LOW = MIN_N_FOR_STABILITY_GROUP      # 200, TDA-10

CI_LEVEL = 0.95

# Auditoria correctiva -- sensibilidad DESCRIPTIVA del bootstrap a la
# longitud de bloque: bloques de L jornadas CONSECUTIVAS (una semana y
# un mes de negociacion). Valores nuevos de esta etapa, sin funcion de
# decision: el bootstrap aprobado (L=1) es el unico que decide.
BLOCK_LENGTH_SENSITIVITY_DAYS: tuple[int, ...] = (5, 21)
DEFAULT_SEED = 0   # misma semilla que TDA-10 (KURTOSIS_BOOTSTRAP_SEED) y TDA-11

# `SUPERPUESTO` describe la ESTABILIDAD DE LA SEMI-AMPLITUD NORMALIZADA
# entre grupos (ratio e IC completo bajo el umbral). NO es una
# demostracion de igualdad del perfil distribucional completo: el
# estadistico no ve asimetria, percentiles interiores ni cola profunda.
LABEL_OVERLAP = "SUPERPUESTO"
LABEL_INTERMEDIATE = "INTERMEDIO"
LABEL_DIVERGENT = "DIVERGENTE"
LABEL_INCONCLUSIVE = "NO_CONCLUYENTE"

VERDICT_SCALE_ONLY = "SOLO_ESCALA"
VERDICT_SCALE_AND_SHAPE = "ESCALA_Y_FORMA"
VERDICT_INCONCLUSIVE = "NO_CONCLUYENTE"

_STAT_NAMES: tuple[str, ...] = tuple(f"q{lvl}" for lvl in QUANTILE_LEVELS) + ("std", "std_trimmed", "n")


class CausalDecileLeakageError(Exception):
    """El decil causal de volatilidad de una barra cambio al perturbar informacion de su misma jornada o posterior.

    Bloqueante: TDA-12 no construye ninguna tabla condicionada por un
    estado de volatilidad que no supero esta prueba.
    """


# ---------------------------------------------------------------------------
# Estado de volatilidad CAUSAL (decision N1)
# ---------------------------------------------------------------------------

def causal_volatility_decile(
    sigma_hat: np.ndarray, trading_date: np.ndarray, valid: np.ndarray | None = None, n_deciles: int = VOLATILITY_DECILES,
) -> np.ndarray:
    """Decil (0..n_deciles-1) de `sigma_hat_(t-1)` respecto de los `sigma_hat` de JORNADAS ANTERIORES unicamente.

    Para una barra de la jornada `d`, sea `H_d` el conjunto de valores
    validos de `sigma_hat` de todas las jornadas estrictamente anteriores
    a `d`. El decil es ``min(floor(n_deciles * F), n_deciles-1)`` con
    ``F = #{h en H_d : h < sigma_hat_t} / |H_d|`` -- la posicion de la
    volatilidad actual dentro de lo observado HASTA AYER. Ninguna barra
    de la jornada `d` ni posterior interviene, asi que el estado esta
    disponible antes de la barra que condiciona (causalidad estricta de
    las fronteras, no solo del valor).

    `NaN` si la barra no es valida o si ``|H_d| < n_deciles*20`` (mismo
    minimo de historia que `assign_volatility_decile` de TDA-10; en la
    practica solo afecta a la(s) primera(s) jornada(s)).

    Entrada: `sigma_hat` y `trading_date` alineados y en orden
    cronologico; `valid` (opcional) restringe que barras cuentan (por
    defecto: `sigma_hat` finito y positivo).

    Implementacion O(n log n + dias*n): se ordena una vez; por jornada,
    una suma acumulada sobre "valores ya vistos" da cuantos historicos
    son menores que cada valor del dia.
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

    sv = s[idx]
    # rango "denso por la izquierda": posicion del primer elemento igual
    # en el orden global -> contar historicos ESTRICTAMENTE menores.
    order = np.argsort(sv, kind="stable")
    sorted_vals = sv[order]
    left_rank = np.searchsorted(sorted_vals, sv, side="left")
    pos_in_sorted = np.empty(sv.size, dtype=np.int64)
    pos_in_sorted[order] = np.arange(sv.size)

    dv = dates[idx]
    # jornadas en orden de aparicion (la poblacion esta en orden cronologico)
    change = np.concatenate([[True], dv[1:] != dv[:-1]])
    day_start = np.flatnonzero(change)
    day_end = np.append(day_start[1:], sv.size)

    present = np.zeros(sv.size, dtype=np.int64)
    n_prior = 0
    min_history = n_deciles * 20
    dec = np.full(sv.size, np.nan)
    for a, b in zip(day_start, day_end):
        if n_prior >= min_history:
            prefix = np.cumsum(present)
            lr = left_rank[a:b]
            less = np.where(lr > 0, prefix[np.maximum(lr - 1, 0)], 0)
            frac = less / float(n_prior)
            dec[a:b] = np.minimum(np.floor(n_deciles * frac), n_deciles - 1)
        present[pos_in_sorted[a:b]] = 1
        n_prior += b - a
    out[idx] = dec
    return out


def verify_causal_decile_no_lookahead(
    sigma_hat: np.ndarray, trading_date: np.ndarray, valid: np.ndarray | None = None, n_checks: int = 6, seed: int = 42,
) -> dict:
    """Prueba de reconstruccion (G1) del estado causal, sobre el MISMO conjunto de barras elegibles (`valid`) con el que se construye el decil que se usa.

    Dos pruebas complementarias, en `n_checks` barras elegibles elegidas
    con semilla fija:

    - PERTURBACION: se multiplica `sigma_hat` de la barra por un factor
      extremo y se recalcula. Se exige deciles identicos en todas las
      barras de jornadas anteriores y en las DEMAS barras de la misma
      jornada (las fronteras de un dia no dependen de ese dia). Se
      reporta si alguna jornada posterior cambio (sanidad: el estado si
      debe reaccionar a la historia).
    - TRUNCAMIENTO: se recalcula usando solo las barras hasta el final
      de la jornada de la barra elegida (como si el futuro no
      existiera). Los deciles de ese tramo deben ser identicos a los
      calculados con la muestra completa.

    `valid` debe ser la misma mascara que se pasa a
    `causal_volatility_decile` (p. ej. excluye el quemado): verificar
    sobre un conjunto distinto del efectivo no acreditaria nada sobre el
    estado realmente usado.
    """
    s = np.asarray(sigma_hat, dtype=float)
    dates = np.asarray(trading_date)
    ok = np.isfinite(s) & (s > 0)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        ok &= valid
    base = causal_volatility_decile(s, dates, valid=valid)
    valid_idx = np.flatnonzero(ok)
    rng = np.random.default_rng(seed)
    checks = []
    if valid_idx.size > 2:
        lo, hi = valid_idx.size // 10, max(valid_idx.size // 10 + 1, valid_idx.size - valid_idx.size // 10)
        picks = np.unique(valid_idx[rng.integers(lo, hi, size=n_checks)])
        for i in picks:
            pert = s.copy()
            pert[i] = s[i] * 1e3
            new = causal_volatility_decile(pert, dates, valid=valid)
            earlier = dates < dates[i]
            same_day_others = (dates == dates[i])
            same_day_others[i] = False
            later = dates > dates[i]
            upto = dates <= dates[i]
            truncated = causal_volatility_decile(s[upto], dates[upto], valid=None if valid is None else valid[upto])
            checks.append({
                "perturb_index": int(i), "n_eligible_bars": int(valid_idx.size),
                "earlier_days_unchanged": bool(np.array_equal(base[earlier], new[earlier], equal_nan=True)),
                "same_day_other_bars_unchanged": bool(np.array_equal(base[same_day_others], new[same_day_others], equal_nan=True)),
                "some_later_day_changed": bool(not np.array_equal(base[later], new[later], equal_nan=True)),
                "truncation_identical": bool(np.array_equal(base[upto], truncated, equal_nan=True)),
            })
    passed = all(c["earlier_days_unchanged"] and c["same_day_other_bars_unchanged"] and c["truncation_identical"] for c in checks) if checks else True
    return {"checks": checks, "passed": bool(passed)}


# ---------------------------------------------------------------------------
# Estadisticos ponderados por multiplicidad -- nucleo del bootstrap
# ---------------------------------------------------------------------------

def weighted_group_stats(sorted_values: np.ndarray, weights: np.ndarray | None, levels: tuple[float, ...] = QUANTILE_LEVELS) -> np.ndarray:
    """Cuantiles, desviacion estandar, desviacion recortada y `n` de la muestra que resulta de repetir `sorted_values[i]` `weights[i]` veces.

    Es EXACTAMENTE lo que se obtendria materializando la muestra
    remuestreada y aplicando ``np.quantile`` (interpolacion lineal, la
    convencion de TDA-07), ``std(ddof=1)`` y el recorte de TDA-07
    (retirar `TRIM_FRACTION_EACH_TAIL` de cada cola y recalcular la
    desviacion), pero sin materializarla: con los valores ya ORDENADOS,
    el estadistico de orden `k` de la muestra repetida es el primer
    valor cuya multiplicidad acumulada supera `k`. `weights=None` = todos
    1 (la estimacion puntual sigue el mismo camino de codigo que cada
    replica del bootstrap).

    Salida: vector ``[q(levels)..., std, std_trimmed, n]``. Cuantiles
    `NaN` si `n == 0`; `std` `NaN` si `n < 2`; `std_trimmed` `NaN` si
    `n < MIN_N_FOR_TRIM` (convencion de TDA-07: nunca se fabrica un
    recorte que no retira nada interpretable).
    """
    x = np.asarray(sorted_values, dtype=float)
    k_levels = len(levels)
    out = np.full(k_levels + 3, np.nan)
    if x.size == 0:
        out[-1] = 0.0
        return out
    w = np.ones(x.size, dtype=np.int64) if weights is None else np.asarray(weights, dtype=np.int64)
    cw = np.cumsum(w)
    n = int(cw[-1])
    out[-1] = float(n)
    if n == 0:
        return out

    def _quantile(p: float) -> float:
        h = (n - 1) * p
        k = int(np.floor(h))
        frac = h - k
        x0 = x[np.searchsorted(cw, k, side="right")]
        if frac == 0.0:
            return float(x0)
        x1 = x[np.searchsorted(cw, min(k + 1, n - 1), side="right")]
        return float(x0 + frac * (x1 - x0))

    for j, p in enumerate(levels):
        out[j] = _quantile(p)

    wx = w * x
    if n > 1:
        s1 = float(wx.sum())
        s2 = float((wx * x).sum())
        var = (s2 - s1 * s1 / n) / (n - 1)
        out[k_levels] = float(np.sqrt(var)) if var > 0 else 0.0
    if n >= MIN_N_FOR_TRIM:
        lo_cut, hi_cut = _quantile(TRIM_FRACTION_EACH_TAIL), _quantile(1.0 - TRIM_FRACTION_EACH_TAIL)
        a = int(np.searchsorted(x, lo_cut, side="left"))
        b = int(np.searchsorted(x, hi_cut, side="right"))
        nt = int(w[a:b].sum())
        if nt > 1:
            t1 = float(wx[a:b].sum())
            t2 = float((wx[a:b] * x[a:b]).sum())
            var_t = (t2 - t1 * t1 / nt) / (nt - 1)
            out[k_levels + 1] = float(np.sqrt(var_t)) if var_t > 0 else 0.0
    return out


def prepare_sorted_groups(values: np.ndarray, day_index: np.ndarray, labels: np.ndarray) -> dict:
    """Agrupa por `labels` (se excluyen valores no finitos y etiquetas nulas) y devuelve `{grupo: (valores ordenados, indice de jornada en ese orden)}`."""
    v = np.asarray(values, dtype=float)
    d = np.asarray(day_index)
    lab = pd.Series(np.asarray(labels))
    mask = np.isfinite(v) & lab.notna().to_numpy()
    groups: dict = {}
    if not mask.any():
        return groups
    frame = pd.DataFrame({"g": lab[mask].to_numpy(), "v": v[mask], "d": d[mask]})
    for g, sub in frame.groupby("g", observed=True, sort=True):
        order = np.argsort(sub["v"].to_numpy(), kind="stable")
        groups[g] = (sub["v"].to_numpy()[order], sub["d"].to_numpy()[order])
    return groups


def joint_day_block_bootstrap(
    sorted_groups_by_grouping: dict[str, dict], n_days: int, n_boot: int, seed: int, block_days: int = 1,
) -> tuple[dict, dict]:
    """Bootstrap de bloques por JORNADA, conjunto para todas las agrupaciones y grupos de una serie.

    `block_days=1` (procedimiento APROBADO, el unico que decide): en cada
    replica se sortean `n_days` jornadas con reemplazo UNA vez (mismo
    esquema que `day_block_bootstrap` de TDA-07) y esa misma seleccion se
    aplica a todos los grupos: asi los estadisticos que comparan grupos
    (ratios de forma) heredan la dependencia entre grupos que comparten
    jornada. La jornada entra completa -- con todas sus barras, en todos
    los grupos a los que pertenecen -- o no entra.

    `block_days=L>1` (sensibilidad descriptiva): bootstrap de bloques
    MOVILES sobre la secuencia cronologica de jornadas -- se sortean
    `ceil(n_days/L)` posiciones de inicio y cada una aporta `L` jornadas
    CONSECUTIVAS completas, en su orden (se recorta el sobrante para que
    la replica tenga `n_days` jornadas). Conserva la dependencia entre
    jornadas vecinas (p. ej. una racha de dias de alta volatilidad entra
    o sale junta). Requiere que el indice de jornada sea cronologico.

    Salida: `(point, reps)` con `point[agrupacion][grupo]` = vector de
    `weighted_group_stats` y `reps[agrupacion][grupo]` = matriz
    `(n_boot, len(vector))`.
    """
    if block_days < 1:
        raise ValueError("block_days debe ser >= 1")
    point = {gk: {g: weighted_group_stats(xs, None) for g, (xs, _) in groups.items()} for gk, groups in sorted_groups_by_grouping.items()}
    reps = {gk: {g: np.full((n_boot, len(_STAT_NAMES)), np.nan) for g in groups} for gk, groups in sorted_groups_by_grouping.items()}
    rng = np.random.default_rng(seed)
    length = min(block_days, n_days)
    n_blocks = int(np.ceil(n_days / length))
    for b in range(n_boot):
        if block_days == 1:
            sampled = rng.integers(0, n_days, size=n_days)
        else:
            starts = rng.integers(0, n_days - length + 1, size=n_blocks)
            sampled = (starts[:, None] + np.arange(length)[None, :]).ravel()[:n_days]
        mult = np.bincount(sampled, minlength=n_days)
        for gk, groups in sorted_groups_by_grouping.items():
            for g, (xs, ds) in groups.items():
                reps[gk][g][b] = weighted_group_stats(xs, mult[ds])
    return point, reps


def _ci(samples: np.ndarray, min_finite: int = 10) -> tuple[float, float]:
    finite = samples[np.isfinite(samples)]
    if finite.size < min_finite:
        return float("nan"), float("nan")
    alpha = (1.0 - CI_LEVEL) / 2.0
    return float(np.percentile(finite, 100 * alpha)), float(np.percentile(finite, 100 * (1 - alpha)))


# ---------------------------------------------------------------------------
# Soporte de los cuantiles extremos (decision N5)
# ---------------------------------------------------------------------------

def tail_support(sorted_values: np.ndarray, day_index_sorted: np.ndarray, level: float, quantile_value: float) -> tuple[int, int]:
    """`(n_tail, n_days_tail)`: observaciones en o mas alla del cuantil `level` (hacia su cola mas cercana) y jornadas distintas que las aportan.

    Para `level < 0.5` cuenta `x <= q`; para `level > 0.5`, `x >= q`;
    para la mediana devuelve el menor de ambos lados. `n_days_tail`
    acota por arriba el numero de episodios independientes que sostienen
    la estimacion.
    """
    x = np.asarray(sorted_values, dtype=float)
    d = np.asarray(day_index_sorted)
    if x.size == 0 or not np.isfinite(quantile_value):
        return 0, 0
    lo_end = int(np.searchsorted(x, quantile_value, side="right"))
    hi_start = int(np.searchsorted(x, quantile_value, side="left"))
    if level < 0.5:
        sl = slice(0, lo_end)
    elif level > 0.5:
        sl = slice(hi_start, x.size)
    else:
        sl = slice(0, lo_end) if lo_end <= x.size - hi_start else slice(hi_start, x.size)
    n_tail = int(sl.stop - sl.start)
    return n_tail, int(np.unique(d[sl]).size) if n_tail else 0


def support_label(n_tail: int) -> str:
    """Etiqueta DESCRIPTIVA del soporte de un cuantil (reutiliza constantes existentes; no es un umbral de decision nuevo)."""
    if n_tail < MIN_TAIL_SUPPORT_UNRELIABLE:
        return "NO_FIABLE"
    if n_tail < MIN_TAIL_SUPPORT_LOW:
        return "BAJO"
    return "OK"


# ---------------------------------------------------------------------------
# Tabla larga de cuantiles (outputs obligatorios del roadmap)
# ---------------------------------------------------------------------------

def build_quantile_table(series: str, block: str, sorted_groups_by_grouping: dict, point: dict, reps: dict, n_days_by_group: dict) -> pd.DataFrame:
    """Una fila por (agrupacion, grupo, nivel de cuantil): estimacion, IC bootstrap, cuantil normalizado (con IC), `n`, soporte de cola y advertencia."""
    k = len(QUANTILE_LEVELS)
    rows = []
    for gk, groups in sorted_groups_by_grouping.items():
        for g, (xs, ds) in groups.items():
            pt, rp = point[gk][g], reps[gk][g]
            std, std_t, n = pt[k], pt[k + 1], int(pt[k + 2])
            std_lo, std_hi = _ci(rp[:, k])
            for j, lvl in enumerate(QUANTILE_LEVELS):
                q = pt[j]
                q_lo, q_hi = _ci(rp[:, j])
                with np.errstate(divide="ignore", invalid="ignore"):
                    norm_reps = rp[:, j] / rp[:, k]
                    norm_t_reps = rp[:, j] / rp[:, k + 1]
                qn_lo, qn_hi = _ci(norm_reps)
                qt_lo, qt_hi = _ci(norm_t_reps)
                n_tail, n_days_tail = tail_support(xs, ds, lvl, q)
                rel_width = (q_hi - q_lo) / (2.0 * abs(q)) if np.isfinite(q) and q != 0 and np.isfinite(q_lo) else float("nan")
                rows.append({
                    "series": series, "block": block, "grouping": gk, "group": g, "n": n, "n_days": n_days_by_group[gk][g],
                    "std": std, "std_ci_lo": std_lo, "std_ci_hi": std_hi, "std_trimmed": std_t,
                    "level": lvl, "q": q, "q_ci_lo": q_lo, "q_ci_hi": q_hi, "ci_rel_halfwidth": rel_width,
                    "q_over_std": q / std if np.isfinite(std) and std > 0 else float("nan"),
                    "q_over_std_ci_lo": qn_lo, "q_over_std_ci_hi": qn_hi,
                    "q_over_std_trimmed": q / std_t if np.isfinite(std_t) and std_t > 0 else float("nan"),
                    "q_over_std_trimmed_ci_lo": qt_lo, "q_over_std_trimmed_ci_hi": qt_hi,
                    "n_tail": n_tail, "n_days_tail": n_days_tail, "support": support_label(n_tail),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Escala vs forma -- perfiles normalizados
# ---------------------------------------------------------------------------

def _scale(stats: np.ndarray, normalization: str) -> np.ndarray:
    k = len(QUANTILE_LEVELS)
    s = np.atleast_2d(stats)
    if normalization == "none":
        return np.ones(s.shape[0])
    return s[:, k] if normalization == "std" else s[:, k + 1]


def _normalized(stats: np.ndarray, numerator: np.ndarray, normalization: str) -> np.ndarray:
    """`numerator / escala`, `NaN` donde la escala no es finita y positiva (nunca `inf` ni 0/0 disfrazado)."""
    scale = _scale(stats, normalization)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where((scale > 0) & np.isfinite(scale), numerator / scale, np.nan)
    return out if np.ndim(stats) > 1 else out[0]


def _half_width(stats: np.ndarray, pair: tuple[float, float], normalization: str) -> np.ndarray:
    """Semi-amplitud `(q_hi - q_lo)/2` dividida por la escala elegida; `stats` es un vector o una matriz `(n_boot, k)`."""
    j_lo, j_hi = QUANTILE_LEVELS.index(pair[0]), QUANTILE_LEVELS.index(pair[1])
    s = np.atleast_2d(stats)
    return _normalized(stats, (s[:, j_hi] - s[:, j_lo]) / 2.0, normalization)


def _tail_distance(stats: np.ndarray, level: float, normalization: str) -> np.ndarray:
    """Distancia NORMALIZADA de UN cuantil al cero: `|q_level| / escala` -- la mitad izquierda o derecha del perfil, por separado."""
    s = np.atleast_2d(stats)
    return _normalized(stats, np.abs(s[:, QUANTILE_LEVELS.index(level)]), normalization)


def _asymmetry(stats: np.ndarray, pair: tuple[float, float]) -> np.ndarray:
    j_lo, j_hi = QUANTILE_LEVELS.index(pair[0]), QUANTILE_LEVELS.index(pair[1])
    s = np.atleast_2d(stats)
    left, right = np.abs(s[:, j_lo]), s[:, j_hi]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where((left + right) > 0, (left - right) / ((left + right) / 2.0), np.nan)
    return out if np.ndim(stats) > 1 else out[0]


def profile_ratio(half_widths: np.ndarray, strict: bool = False) -> np.ndarray:
    """``(max-min)/mediana`` entre grupos -- el MISMO estadistico que `profile_stability_ratio` de TDA-10, sobre semi-amplitudes ya normalizadas.

    Bajo una familia de escala pura (todos los grupos comparten forma) la
    semi-amplitud normalizada es identica en todos los grupos y el ratio
    es 0. Entrada: vector `(n_grupos,)` o matriz `(n_boot, n_grupos)`.
    `NaN` si menos de 2 grupos finitos o mediana nula. Con `strict=True`
    (el modo que usan las comparaciones de esta etapa) tambien es `NaN`
    si ALGUN grupo no es finito: nunca se resume "los grupos" ignorando
    en silencio a uno de ellos.
    """
    h = np.atleast_2d(np.asarray(half_widths, dtype=float))
    out = np.full(h.shape[0], np.nan)
    finite = np.isfinite(h)
    ok = finite.all(axis=1) & (h.shape[1] >= 2) if strict else finite.sum(axis=1) >= 2
    if ok.any():
        hv = h[ok]
        med = np.nanmedian(hv, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[ok] = np.where(med != 0, (np.nanmax(hv, axis=1) - np.nanmin(hv, axis=1)) / med, np.nan)
    return out if np.ndim(half_widths) > 1 else out[0]


def classify_ratio(point: float, ci_lo: float, ci_hi: float) -> str:
    """Etiqueta de UNA normalizacion con los umbrales de TDA-10 (sin cambiarlos): firme solo si el IC 95 % completo cae en la misma zona que el punto.

    `SUPERPUESTO` = semi-amplitud normalizada estable entre grupos; no
    afirma igualdad del perfil completo. Aplicar estos umbrales a un
    ratio normalizado es una decision nueva y exploratoria de TDA-12.
    """
    if not np.isfinite([point, ci_lo, ci_hi]).all():
        return LABEL_INCONCLUSIVE
    lo_t, hi_t = PROFILE_STABILITY_SCALE_THRESHOLD, PROFILE_STABILITY_FORM_THRESHOLD
    if point <= lo_t and ci_hi <= lo_t:
        return LABEL_OVERLAP
    if point >= hi_t and ci_lo >= hi_t:
        return LABEL_DIVERGENT
    if lo_t < point < hi_t and ci_lo > lo_t and ci_hi < hi_t:
        return LABEL_INTERMEDIATE
    return LABEL_INCONCLUSIVE


def _point_zone(point: float) -> str:
    """Zona del estimador PUNTUAL (sin IC) -- usada solo en la sensibilidad a estimador."""
    if not np.isfinite(point):
        return LABEL_INCONCLUSIVE
    if point <= PROFILE_STABILITY_SCALE_THRESHOLD:
        return LABEL_OVERLAP
    if point >= PROFILE_STABILITY_FORM_THRESHOLD:
        return LABEL_DIVERGENT
    return LABEL_INTERMEDIATE


def classify_groups(sorted_groups: dict, point: dict, pair: tuple[float, float], normalization: str, min_tail: int | None = MIN_TAIL_SUPPORT_LOW) -> dict:
    """Reparte los grupos de UNA comparacion en cuatro conjuntos explicitos (auditoria correctiva A1).

    - `available`: todos los grupos de la agrupacion.
    - `excluded_low_support`: su cola en algun nivel del par tiene menos
      de `min_tail` observaciones (decision N5). Con `min_tail=None` se
      aplica en su lugar el criterio LITERAL de TDA-10: tamaño de grupo
      `n >= MIN_N_FOR_STABILITY_GROUP`.
    - `excluded_invalid_normalization`: tienen soporte pero su escala
      (desviacion o desviacion recortada) es nula o no finita, de modo
      que no existe un perfil normalizado que comparar.
    - `compared`: los que entran efectivamente en el ratio.
    """
    available = list(sorted_groups)
    low, invalid, compared = [], [], []
    for g, (xs, ds) in sorted_groups.items():
        if min_tail is None:
            supported = int(point[g][-1]) >= MIN_N_FOR_STABILITY_GROUP
        else:
            supported = all(tail_support(xs, ds, lvl, point[g][QUANTILE_LEVELS.index(lvl)])[0] >= min_tail for lvl in pair)
        if not supported:
            low.append(g)
        elif not np.isfinite(_half_width(point[g], pair, normalization)):
            invalid.append(g)
        else:
            compared.append(g)
    return {"available": available, "excluded_low_support": low, "excluded_invalid_normalization": invalid, "compared": compared}


def _eligible_groups(sorted_groups: dict, point: dict, pair: tuple[float, float]) -> list:
    """Grupos con soporte `OK` en ambos niveles del par (sin mirar la normalizacion) -- usado por la sensibilidad puntual a estimador."""
    sets = classify_groups(sorted_groups, point, pair, "none")
    return sets["compared"] + sets["excluded_invalid_normalization"]


def _ratio_with_ci(point_values: np.ndarray, rep_values: np.ndarray) -> tuple[float, float, float, int]:
    ratio = float(profile_ratio(point_values, strict=True))
    rep_ratios = profile_ratio(rep_values, strict=True)
    lo, hi = _ci(rep_ratios)
    return ratio, lo, hi, int(np.isfinite(rep_ratios).sum())


def build_shape_comparison_table(
    series: str, block: str, sorted_groups_by_grouping: dict, point: dict, reps: dict, is_standardized: bool,
    min_tail: int | None = MIN_TAIL_SUPPORT_LOW,
) -> pd.DataFrame:
    """Una fila por (agrupacion, par de niveles, normalizacion): comparacion de forma entre grupos.

    Columnas principales:

    - Que grupos entran: `n_groups_available`, `n_groups_compared`
      (= `n_groups`), `groups_excluded_low_support`,
      `groups_excluded_invalid_normalization`, `obs_fraction_compared`
      (fraccion de las observaciones de la agrupacion que pertenece a
      grupos comparados) y `n_boot_valid` (replicas en las que TODOS los
      grupos comparados tuvieron normalizacion valida).
    - `ratio` (+IC): `(max-min)/mediana` de la SEMI-AMPLITUD normalizada
      -- el estadistico de la regla N4. Es ciego a la asimetria.
    - `left_ratio`, `right_ratio` (+IC): el mismo ratio sobre cada cola
      por separado (`|q_lo|/escala` y `q_hi/escala`), y
      `asymmetry_range` (+IC): `max-min` de la asimetria entre grupos.
      DESCRIPTIVOS: completan lo que la semi-amplitud no ve.
    - `label` (solo en `VERDICT_LEVEL_PAIR`): etiqueta de la regla N4
      sobre `ratio`; `NO_CONCLUYENTE` si hay menos de 2 grupos comparados
      o si algun grupo con soporte carece de normalizacion valida.
    - `left_label`, `right_label` (solo en `VERDICT_LEVEL_PAIR`): la
      misma clasificacion aplicada a cada cola -- insumo de la PROPUESTA
      `evaluate_one_sided_proposal`, que no decide.
    """
    rows = []
    norms = NORMALIZATIONS + (("none",) if is_standardized else ())
    for gk, groups in sorted_groups_by_grouping.items():
        if gk == "GLOBAL":
            continue
        n_total = float(sum(point[gk][g][-1] for g in groups)) or float("nan")
        for pair in SYMMETRIC_LEVEL_PAIRS:
            for norm in norms:
                sets = classify_groups(groups, point[gk], pair, norm, min_tail)
                compared = sets["compared"]
                row = {
                    "series": series, "block": block, "grouping": gk, "level_lo": pair[0], "level_hi": pair[1], "normalization": norm,
                    "n_groups_available": len(sets["available"]), "n_groups": len(compared), "n_groups_compared": len(compared),
                    "groups_excluded_low_support": ";".join(str(g) for g in sets["excluded_low_support"]),
                    "groups_excluded_invalid_normalization": ";".join(str(g) for g in sets["excluded_invalid_normalization"]),
                    "obs_fraction_compared": float(sum(point[gk][g][-1] for g in compared) / n_total),
                    "n_boot_valid": 0,
                    "ratio": float("nan"), "ratio_ci_lo": float("nan"), "ratio_ci_hi": float("nan"),
                    "half_width_min": float("nan"), "group_min": "", "half_width_max": float("nan"), "group_max": "",
                    "half_width_median": float("nan"),
                    "left_ratio": float("nan"), "left_ratio_ci_lo": float("nan"), "left_ratio_ci_hi": float("nan"),
                    "right_ratio": float("nan"), "right_ratio_ci_lo": float("nan"), "right_ratio_ci_hi": float("nan"),
                    "asymmetry_range": float("nan"), "asymmetry_range_ci_lo": float("nan"), "asymmetry_range_ci_hi": float("nan"),
                    "label": "", "left_label": "", "right_label": "",
                }
                blocked = bool(sets["excluded_invalid_normalization"])
                if len(compared) >= 2:
                    h_point = np.array([_half_width(point[gk][g], pair, norm) for g in compared])
                    h_reps = np.column_stack([_half_width(reps[gk][g], pair, norm) for g in compared])
                    ratio, lo, hi, n_valid = _ratio_with_ci(h_point, h_reps)
                    row.update({
                        "ratio": ratio, "ratio_ci_lo": lo, "ratio_ci_hi": hi, "n_boot_valid": n_valid,
                        "half_width_min": float(np.min(h_point)), "group_min": str(compared[int(np.argmin(h_point))]),
                        "half_width_max": float(np.max(h_point)), "group_max": str(compared[int(np.argmax(h_point))]),
                        "half_width_median": float(np.median(h_point)),
                    })
                    side_stats = {}
                    for side, lvl in (("left", pair[0]), ("right", pair[1])):
                        t_point = np.array([_tail_distance(point[gk][g], lvl, norm) for g in compared])
                        t_reps = np.column_stack([_tail_distance(reps[gk][g], lvl, norm) for g in compared])
                        side_stats[side] = _ratio_with_ci(t_point, t_reps)[:3]
                        row[f"{side}_ratio"], row[f"{side}_ratio_ci_lo"], row[f"{side}_ratio_ci_hi"] = side_stats[side]
                    a_point = np.array([_asymmetry(point[gk][g], pair) for g in compared])
                    a_reps = np.column_stack([_asymmetry(reps[gk][g], pair) for g in compared])
                    if np.isfinite(a_point).all():
                        with np.errstate(invalid="ignore"):
                            rng_reps = np.where(np.isfinite(a_reps).all(axis=1), np.max(a_reps, axis=1) - np.min(a_reps, axis=1), np.nan)
                        row["asymmetry_range"] = float(np.max(a_point) - np.min(a_point))
                        row["asymmetry_range_ci_lo"], row["asymmetry_range_ci_hi"] = _ci(rng_reps)
                    if pair == VERDICT_LEVEL_PAIR:
                        row["label"] = LABEL_INCONCLUSIVE if blocked else classify_ratio(ratio, lo, hi)
                        for side in ("left", "right"):
                            row[f"{side}_label"] = LABEL_INCONCLUSIVE if blocked else classify_ratio(*side_stats[side])
                elif pair == VERDICT_LEVEL_PAIR:
                    row["label"] = row["left_label"] = row["right_label"] = LABEL_INCONCLUSIVE
                rows.append(row)
    return pd.DataFrame(rows)


def build_asymmetry_table(series: str, block: str, sorted_groups_by_grouping: dict, point: dict, reps: dict) -> pd.DataFrame:
    """Forma de las colas POR GRUPO y par de niveles (descriptiva, sin umbral), con IC bootstrap.

    - `half_width_over_std` / `half_width_over_std_trimmed`: semi-amplitud
      ``(q_hi - q_lo)/2`` en multiplos de la desviacion (o de la
      desviacion recortada) del grupo -- los valores cuya dispersion
      ENTRE grupos resume `build_shape_comparison_table`.
    - `asymmetry`: ``(|q_lo| - q_hi) / ((|q_lo| + q_hi)/2)``; positivo =
      cola izquierda mas larga que la derecha.
    """
    rows = []
    for gk, groups in sorted_groups_by_grouping.items():
        for g in groups:
            for pair in SYMMETRIC_LEVEL_PAIRS:
                lo, hi = _ci(_asymmetry(reps[gk][g], pair))
                row = {
                    "series": series, "block": block, "grouping": gk, "group": g, "level_lo": pair[0], "level_hi": pair[1],
                    "n": int(point[gk][g][-1]),
                    "asymmetry": float(_asymmetry(point[gk][g], pair)), "asymmetry_ci_lo": lo, "asymmetry_ci_hi": hi,
                }
                for norm in NORMALIZATIONS:
                    h_lo, h_hi = _ci(_half_width(reps[gk][g], pair, norm))
                    row[f"half_width_over_{norm}"] = float(_half_width(point[gk][g], pair, norm))
                    row[f"half_width_over_{norm}_ci_lo"] = h_lo
                    row[f"half_width_over_{norm}_ci_hi"] = h_hi
                xs, ds = groups[g]
                supports = [support_label(tail_support(xs, ds, lvl, point[gk][g][QUANTILE_LEVELS.index(lvl)])[0]) for lvl in pair]
                row["support"] = "OK" if all(v == "OK" for v in supports) else ("NO_FIABLE" if "NO_FIABLE" in supports else "BAJO")
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sensibilidad al estimador de volatilidad (6 configuraciones causales de TDA-10)
# ---------------------------------------------------------------------------

def _point_shape_ratios(z: np.ndarray, day_index: np.ndarray, labels_by_grouping: dict) -> dict:
    """Ratios puntuales (nivel de veredicto, ambas normalizaciones) por agrupacion -- sin bootstrap."""
    out = {}
    for gk in VERDICT_GROUPINGS:
        groups = prepare_sorted_groups(z, day_index, labels_by_grouping[gk])
        point = {g: weighted_group_stats(xs, None) for g, (xs, _) in groups.items()}
        for norm in NORMALIZATIONS:
            sets = classify_groups(groups, point, VERDICT_LEVEL_PAIR, norm)
            if len(sets["compared"]) >= 2 and not sets["excluded_invalid_normalization"]:
                out[(gk, norm)] = float(profile_ratio(np.array([_half_width(point[g], VERDICT_LEVEL_PAIR, norm) for g in sets["compared"]]), strict=True))
            else:
                out[(gk, norm)] = float("nan")
    return out


def _point_verdict(ratios: dict) -> str:
    """Conclusion PUNTUAL de una configuracion: misma logica que `decide_th26` pero sin IC."""
    zones = []
    for gk in VERDICT_GROUPINGS:
        z_std, z_trim = _point_zone(ratios[(gk, "std")]), _point_zone(ratios[(gk, "std_trimmed")])
        zones.append(z_std if z_std == z_trim else LABEL_INCONCLUSIVE)
    if LABEL_DIVERGENT in zones:
        return VERDICT_SCALE_AND_SHAPE
    if all(z == LABEL_OVERLAP for z in zones):
        return VERDICT_SCALE_ONLY
    return VERDICT_INCONCLUSIVE


def build_estimator_sensitivity(r_raw: np.ndarray, trading_date: np.ndarray, day_index: np.ndarray, segment: np.ndarray, year: np.ndarray) -> pd.DataFrame:
    """Repite la comparacion de forma de `z_t` (puntual) con cada una de las 6 configuraciones CAUSALES (`raw`) de `sigma_hat` de TDA-10.

    Cada configuracion usa SU PROPIO `sigma_hat` tanto para estandarizar
    como para el decil causal. Responde: ¿la conclusion depende de la
    ventana/familia del estimador de volatilidad? (riesgo explicito del
    roadmap de TDA-10: "concluir sobre una unica ventana").
    """
    rows = []
    for family, param, input_series in ALL_ESTIMATOR_CONFIGS:
        if input_series != "raw":
            continue
        sigma_hat = build_sigma_hat(r_raw, family, param)
        z = standardize_return(r_raw, sigma_hat, burn_in_length(family, param))
        decile = causal_volatility_decile(sigma_hat, trading_date, valid=np.isfinite(z))
        ratios = _point_shape_ratios(z, day_index, {"SEGMENT": segment, "VOL_DECILE_CAUSAL": decile, "YEAR": year})
        row = {"family": family, "param": float(param), "input_series": input_series, "is_primary": (family, float(param), input_series) == PRIMARY_ESTIMATOR}
        for (gk, norm), v in ratios.items():
            row[f"ratio_{gk}_{norm}"] = v
        row["point_verdict"] = _point_verdict(ratios)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Veredicto TH26 y consecuencia para STOP-13 (decision N4)
# ---------------------------------------------------------------------------

def decide_th26(shape_table: pd.DataFrame, sensitivity: pd.DataFrame, verdict_series: str) -> dict:
    """Veredicto formal de TH26 sobre `verdict_series` (el `z_t` primario causal), con la regla predeclarada N4.

    1. Etiqueta por agrupacion de `VERDICT_GROUPINGS`: la de la
       normalizacion `std` si coincide con la de `std_trimmed`;
       `NO_CONCLUYENTE` si discrepan.
    2. `ESCALA_Y_FORMA` si alguna es `DIVERGENTE`; `SOLO_ESCALA` si
       todas son `SUPERPUESTO`; `NO_CONCLUYENTE` en otro caso.
    3. Robustez a estimador: fraccion de las configuraciones causales
       cuya conclusion puntual coincide con el veredicto del paso 2; si
       es menor que `ROBUSTNESS_AGREEMENT_FRACTION` el veredicto final
       pasa a `NO_CONCLUYENTE` (se conserva `verdict_before_robustness`).
    """
    sub = shape_table.loc[
        (shape_table["series"] == verdict_series)
        & (shape_table["level_lo"] == VERDICT_LEVEL_PAIR[0]) & (shape_table["level_hi"] == VERDICT_LEVEL_PAIR[1])
    ]
    labels: dict[str, str] = {}
    detail = []
    for gk in VERDICT_GROUPINGS:
        by_norm = {}
        for norm in NORMALIZATIONS:
            r = sub.loc[(sub["grouping"] == gk) & (sub["normalization"] == norm)]
            by_norm[norm] = r.iloc[0] if len(r) else None
        lab_std = by_norm["std"]["label"] if by_norm["std"] is not None else LABEL_INCONCLUSIVE
        lab_trim = by_norm["std_trimmed"]["label"] if by_norm["std_trimmed"] is not None else LABEL_INCONCLUSIVE
        labels[gk] = lab_std if lab_std == lab_trim else LABEL_INCONCLUSIVE
        detail.append({
            "grouping": gk, "label": labels[gk], "label_std": lab_std, "label_std_trimmed": lab_trim,
            "ratio_std": float(by_norm["std"]["ratio"]) if by_norm["std"] is not None else float("nan"),
            "ratio_std_ci_lo": float(by_norm["std"]["ratio_ci_lo"]) if by_norm["std"] is not None else float("nan"),
            "ratio_std_ci_hi": float(by_norm["std"]["ratio_ci_hi"]) if by_norm["std"] is not None else float("nan"),
            "ratio_std_trimmed": float(by_norm["std_trimmed"]["ratio"]) if by_norm["std_trimmed"] is not None else float("nan"),
            "ratio_std_trimmed_ci_lo": float(by_norm["std_trimmed"]["ratio_ci_lo"]) if by_norm["std_trimmed"] is not None else float("nan"),
            "ratio_std_trimmed_ci_hi": float(by_norm["std_trimmed"]["ratio_ci_hi"]) if by_norm["std_trimmed"] is not None else float("nan"),
        })

    values = list(labels.values())
    if LABEL_DIVERGENT in values:
        before = VERDICT_SCALE_AND_SHAPE
    elif values and all(v == LABEL_OVERLAP for v in values):
        before = VERDICT_SCALE_ONLY
    else:
        before = VERDICT_INCONCLUSIVE

    if sensitivity is None or sensitivity.empty:
        agreement, n_configs = float("nan"), 0
    else:
        n_configs = len(sensitivity)
        agreement = float((sensitivity["point_verdict"] == before).mean())
    robust = bool(np.isfinite(agreement) and agreement >= ROBUSTNESS_AGREEMENT_FRACTION)
    final = before if (before == VERDICT_INCONCLUSIVE or robust) else VERDICT_INCONCLUSIVE

    # INFORMATIVO (auditoria correctiva): comparacion HOMOGENEA --
    # conclusion puntual del primario contra las conclusiones puntuales
    # de las demas configuraciones. No interviene en `final`.
    primary_point, point_agreement = "", float("nan")
    if sensitivity is not None and not sensitivity.empty and "is_primary" in sensitivity.columns and sensitivity["is_primary"].any():
        primary_point = str(sensitivity.loc[sensitivity["is_primary"], "point_verdict"].iloc[0])
        point_agreement = float((sensitivity["point_verdict"] == primary_point).mean())

    return {
        "verdict": final, "verdict_before_robustness": before, "verdict_series": verdict_series,
        "labels_by_grouping": labels, "detail": detail,
        "estimator_agreement_fraction": agreement, "n_estimator_configs": n_configs, "robust_to_estimator": robust,
        "primary_point_verdict": primary_point, "estimator_point_agreement_with_primary_point": point_agreement,
        "scale_threshold": PROFILE_STABILITY_SCALE_THRESHOLD, "form_threshold": PROFILE_STABILITY_FORM_THRESHOLD,
        "robustness_threshold": ROBUSTNESS_AGREEMENT_FRACTION,
    }


def _combine(labels: list[str]) -> str:
    """Misma agregacion que la regla N4, sobre una lista de etiquetas por agrupacion."""
    if LABEL_DIVERGENT in labels:
        return VERDICT_SCALE_AND_SHAPE
    if labels and all(v == LABEL_OVERLAP for v in labels):
        return VERDICT_SCALE_ONLY
    return VERDICT_INCONCLUSIVE


def _verdict_rows(shape_table: pd.DataFrame, series: str) -> pd.DataFrame:
    return shape_table.loc[
        (shape_table["series"] == series)
        & (shape_table["level_lo"] == VERDICT_LEVEL_PAIR[0]) & (shape_table["level_hi"] == VERDICT_LEVEL_PAIR[1])
    ]


def evaluate_one_sided_proposal(shape_table: pd.DataFrame, verdict_series: str) -> dict:
    """PROPUESTA pendiente de aprobacion -- NO es la regla vigente y `decide_th26` no la usa.

    Motivo (auditoria correctiva A3): la semi-amplitud no distingue una
    distribucion de su reflejo. La propuesta exige, por agrupacion y con
    las dos normalizaciones, que la clasificacion de la regla N4 se
    cumpla en CADA cola por separado (`left_label` y `right_label`): la
    etiqueta de la agrupacion es la comun a las cuatro (2 colas x 2
    normalizaciones) o `NO_CONCLUYENTE` si discrepan; salvo que alguna
    cola sea `DIVERGENTE` con ambas normalizaciones, en cuyo caso la
    agrupacion es `DIVERGENTE`. Usa los mismos umbrales, aplicados a
    `|q|/escala` de un solo cuantil -- una extension mas, no aprobada.
    """
    sub = _verdict_rows(shape_table, verdict_series)
    labels, detail = {}, []
    for gk in VERDICT_GROUPINGS:
        cells = {}
        for norm in NORMALIZATIONS:
            r = sub.loc[(sub["grouping"] == gk) & (sub["normalization"] == norm)]
            for side in ("left", "right"):
                cells[(side, norm)] = str(r[f"{side}_label"].iloc[0]) if len(r) and r[f"{side}_label"].iloc[0] else LABEL_INCONCLUSIVE
        side_div = any(all(cells[(side, norm)] == LABEL_DIVERGENT for norm in NORMALIZATIONS) for side in ("left", "right"))
        vals = set(cells.values())
        labels[gk] = LABEL_DIVERGENT if side_div else (vals.pop() if len(vals) == 1 else LABEL_INCONCLUSIVE)
        detail.append({"grouping": gk, "label": labels[gk], **{f"{side}_{norm}": v for (side, norm), v in cells.items()}})
    return {"verdict_if_adopted": _combine(list(labels.values())), "labels_by_grouping": labels, "detail": detail, "status": "PROPUESTA_NO_VIGENTE"}


def build_alternative_readings(shape_table: pd.DataFrame, shape_table_tda10_eligibility: pd.DataFrame, verdict_series: str, th26: dict) -> pd.DataFrame:
    """Que diria la misma agregacion bajo lecturas ALTERNATIVAS del estadistico (descriptivo; solo la primera fila es la regla vigente).

    Sirve para juzgar cuanto depende el veredicto de decisiones nuevas de
    esta etapa: la normalizacion (N3), el criterio de participacion (N5)
    y la ceguera a la asimetria (A3). No se aplica la degradacion por
    robustez a estimador (es `verdict_before_robustness`).
    """
    def _labels(table: pd.DataFrame, norms: tuple[str, ...]) -> dict:
        sub = _verdict_rows(table, verdict_series)
        out = {}
        for gk in VERDICT_GROUPINGS:
            labs = []
            for norm in norms:
                r = sub.loc[(sub["grouping"] == gk) & (sub["normalization"] == norm)]
                labs.append(str(r["label"].iloc[0]) if len(r) else LABEL_INCONCLUSIVE)
            out[gk] = labs[0] if len(set(labs)) == 1 else LABEL_INCONCLUSIVE
        return out

    proposal = evaluate_one_sided_proposal(shape_table, verdict_series)
    readings = [
        ("regla vigente N4: ÷std y ÷std_trimmed deben coincidir; cola >= 200", th26["labels_by_grouping"], th26["verdict_before_robustness"], "VIGENTE"),
        ("solo ÷std (lectura literal del roadmap de TDA-12)", _labels(shape_table, ("std",)), None, "alternativa"),
        ("solo ÷std_trimmed", _labels(shape_table, ("std_trimmed",)), None, "alternativa"),
        ("sin dividir por la desviacion del grupo (estadistico LITERAL de TDA-10, al que pertenecen los umbrales)", _labels(shape_table, ("none",)), None, "alternativa"),
        ("regla N4 con el criterio de participacion LITERAL de TDA-10 (tamaño de grupo >= 200)", _labels(shape_table_tda10_eligibility, NORMALIZATIONS), None, "alternativa"),
        ("PROPUESTA: cada cola por separado (sensible a asimetria)", proposal["labels_by_grouping"], proposal["verdict_if_adopted"], "propuesta"),
    ]
    rows = []
    for name, labels, verdict, status in readings:
        row = {"reading": name, "status": status, **{gk: labels[gk] for gk in VERDICT_GROUPINGS}}
        row["verdict_before_robustness"] = verdict if verdict is not None else _combine(list(labels.values()))
        rows.append(row)
    return pd.DataFrame(rows)


def build_block_length_sensitivity(
    series: str, sorted_groups_by_grouping: dict, n_days: int, n_boot: int, seed: int,
    point: dict, reps_by_block: dict[int, dict], focus: list[tuple[str, object]],
) -> pd.DataFrame:
    """Compara los IC del bootstrap aprobado (bloque = 1 jornada) con bloques de varias jornadas consecutivas (descriptivo).

    Filas: (a) el estadistico del veredicto -- ratio de forma 1%-99% por
    agrupacion y normalizacion -- y (b) para cada grupo de `focus`
    (`(agrupacion, grupo)`): su desviacion, su semi-amplitud normalizada
    y sus cuantiles 1/99 y 0.1/99.9. `width_ratio_vs_1day` = anchura del
    IC con ese bloque / anchura con el bloque de 1 jornada.
    """
    k = len(QUANTILE_LEVELS)
    rows = []

    def _add(statistic: str, grouping: str, group: str, norm: str, point_value: float, samples_by_block: dict[int, np.ndarray], label_fn=None) -> None:
        base_width = None
        for L in sorted(samples_by_block):
            lo, hi = _ci(samples_by_block[L])
            width = hi - lo
            if L == 1:
                base_width = width
            rows.append({
                "series": series, "statistic": statistic, "grouping": grouping, "group": group, "normalization": norm,
                "block_days": L, "is_approved_procedure": L == 1, "point": point_value, "ci_lo": lo, "ci_hi": hi, "ci_width": width,
                "width_ratio_vs_1day": width / base_width if base_width and np.isfinite(base_width) and base_width > 0 else float("nan"),
                "label_with_this_block": label_fn(point_value, lo, hi) if label_fn else "",
            })

    for gk in VERDICT_GROUPINGS:
        groups = sorted_groups_by_grouping[gk]
        for norm in NORMALIZATIONS:
            sets = classify_groups(groups, point[gk], VERDICT_LEVEL_PAIR, norm)
            compared = sets["compared"]
            if len(compared) < 2:
                continue
            h_point = np.array([_half_width(point[gk][g], VERDICT_LEVEL_PAIR, norm) for g in compared])
            samples = {
                L: profile_ratio(np.column_stack([_half_width(reps[gk][g], VERDICT_LEVEL_PAIR, norm) for g in compared]), strict=True)
                for L, reps in reps_by_block.items()
            }
            blocked = bool(sets["excluded_invalid_normalization"])
            _add("shape_ratio_1_99", gk, "(entre grupos)", norm, float(profile_ratio(h_point, strict=True)), samples,
                 (lambda p, lo, hi: LABEL_INCONCLUSIVE) if blocked else classify_ratio)

    for gk, g in focus:
        if gk not in sorted_groups_by_grouping or g not in sorted_groups_by_grouping[gk]:
            continue
        pt = point[gk][g]
        _add("std", gk, str(g), "", float(pt[k]), {L: reps[gk][g][:, k] for L, reps in reps_by_block.items()})
        for norm in NORMALIZATIONS:
            _add("half_width_1_99", gk, str(g), norm, float(_half_width(pt, VERDICT_LEVEL_PAIR, norm)),
                 {L: _half_width(reps[gk][g], VERDICT_LEVEL_PAIR, norm) for L, reps in reps_by_block.items()})
        for lvl in (0.001, 0.01, 0.99, 0.999):
            j = QUANTILE_LEVELS.index(lvl)
            _add(f"q{lvl}", gk, str(g), "", float(pt[j]), {L: reps[gk][g][:, j] for L, reps in reps_by_block.items()})
    columns = [
        "series", "statistic", "grouping", "group", "normalization", "block_days", "is_approved_procedure", "point",
        "ci_lo", "ci_hi", "ci_width", "width_ratio_vs_1day", "label_with_this_block",
    ]
    return pd.DataFrame(rows, columns=columns)


def decide_stop13_implication(th26: dict) -> dict:
    """Consecuencia del veredicto para `STOP-13` (roadmap, TDA-13 parte B). TDA-12 NO ejecuta TDA-13 ni EVT: solo deja escrito el estado de su criterio de entrada 1."""
    v = th26["verdict"]
    if v == VERDICT_SCALE_ONLY:
        return {
            "stop13_activated_by_tda12": True, "evt_entry_criterion_1_met": False,
            "text": "TDA-12 = SOLO_ESCALA -> STOP-13 ACTIVADO por TDA-12: la rama EVT (TDA-13 parte B) no se ejecuta. TDA-13 parte A (inventario y agrupamiento) sigue siendo obligatoria.",
        }
    if v == VERDICT_SCALE_AND_SHAPE:
        return {
            "stop13_activated_by_tda12": False, "evt_entry_criterion_1_met": True,
            "text": (
                "TDA-12 = ESCALA_Y_FORMA -> STOP-13 NO activado por TDA-12: se cumple el criterio de entrada 1 de EVT. La rama EVT sigue "
                "CONDICIONADA a los criterios 2 (razon escrita para extrapolar mas alla de lo observado) y 3 (TDA-14 no declara no "
                "estacionariedad fuerte), y a la segunda via de STOP-13 ('los cuantiles empiricos ya responden'), que decide TDA-13. "
                "TDA-13 parte A (inventario y agrupamiento) es obligatoria en cualquier caso."
            ),
        }
    return {
        "stop13_activated_by_tda12": False, "evt_entry_criterion_1_met": False,
        "text": (
            "TDA-12 = NO_CONCLUYENTE -> TDA-12 NO activa STOP-13 (no puede afirmar SOLO_ESCALA) y tampoco declara cumplido el criterio "
            "de entrada 1 de EVT (no puede afirmar ESCALA_Y_FORMA). El roadmap no preve este tercer resultado: como EVT exige que TDA-12 "
            "haya concluido ESCALA_Y_FORMA, leido literalmente la rama EVT queda CERRADA mientras ese criterio no se cumpla; es una "
            "decision que requiere confirmacion explicita antes de TDA-13 parte B. TDA-13 parte A es obligatoria en cualquier caso."
        ),
    }


# ---------------------------------------------------------------------------
# Progreso visible -- mismo patron que TDA-09/10/11
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

    def _close(self, now: float) -> None:
        elapsed = now - self._last
        self.stage_timings[self._current_name] = elapsed
        if self.verbose:
            print(f"[TDA12 {self._i}/{self.total}] {self._current_name} -- completado en {elapsed:.1f}s (total: {now - self.t_start:.1f}s)")

    def stage(self, name: str) -> None:
        now = time.perf_counter()
        if self._i > 0:
            self._close(now)
        self._i += 1
        self._current_name = name
        self._last = now
        if self.verbose:
            print(f"[TDA12 {self._i}/{self.total}] {name}...")

    def finish(self) -> dict[str, float]:
        now = time.perf_counter()
        self._close(now)
        self.stage_timings["_total_analysis"] = now - self.t_start
        return self.stage_timings


ANALYSIS_STAGES = 6
TOTAL_STAGES = 7  # 6 de analisis + 1 de persistencia de artefactos (run_tda12.py)


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------

@dataclass
class TDA12Result:
    n_population: int
    n_days: int
    primary_estimator: tuple
    series_info: pd.DataFrame = None            # una fila por serie: bloque, clasificacion causal, n finito, quemado
    causality_sigma: pd.DataFrame = None        # run_causality_checks (TDA-10), re-ejecutado
    causality_decile: dict = field(default_factory=dict)
    decile_sizes: pd.DataFrame = None           # tamaño de cada decil, causal vs muestra completa
    quantile_table: pd.DataFrame = None
    shape_table: pd.DataFrame = None
    asymmetry_table: pd.DataFrame = None
    estimator_sensitivity: pd.DataFrame = None
    # Auditoria correctiva -- evidencia descriptiva adicional (ninguna decide):
    shape_table_tda10_eligibility: pd.DataFrame = None   # misma comparacion con el criterio de participacion literal de TDA-10
    alternative_readings: pd.DataFrame = None
    one_sided_proposal: dict = field(default_factory=dict)
    block_sensitivity: pd.DataFrame = None
    th26: dict = field(default_factory=dict)
    th26_clock_adjusted: dict = field(default_factory=dict)   # diagnostico retrospectivo, nunca decide
    stop13: dict = field(default_factory=dict)
    n_boot: int = 0
    seed: int = DEFAULT_SEED
    stage_timings: dict[str, float] = field(default_factory=dict)


def _series_label(key: tuple[str, float, str]) -> str:
    family, param, input_series = key
    return f"z_{family}_{param:g}_{input_series}"


def run_tda12_analysis(config: SnapshotConfig, verbose: bool = True) -> TDA12Result:
    """Orquesta TDA-12: poblaciones -> sigma_hat/z (TDA-10) y verificacion de causalidad -> estados causales -> cuantiles + bootstrap conjunto por jornada -> escala vs forma -> sensibilidad a estimador -> veredicto TH26 / implicacion STOP-13."""
    timer = _StageTimer(ANALYSIS_STAGES, verbose=verbose)

    # --- Etapa 1: poblaciones e invariantes (identico a TDA-10/11) ----------
    timer.stage("Poblaciones (r_1m/r_tilde) e invariantes")
    validate_research_holdout_disjoint(config)
    variables, validity = load_tda04_inputs(config)
    validate_last_timestamps_before_boundary(config, variables.groupby("source_file")["timestamp"].max().to_dict())
    verify_timestamp_alignment(variables, validity)

    cutoffs = load_segmentation_cutoffs(config)
    r1m_pop, _ = build_r1m_population(variables, validity, cutoffs)
    r1m_pop = r1m_pop.sort_values("timestamp").reset_index(drop=True)
    r_tilde = load_r_tilde(config)
    verify_r_tilde_invariants(r_tilde, variables)
    r_tilde_pop, _ = build_r_tilde_population(r_tilde, cutoffs)
    r_tilde_pop = r_tilde_pop.sort_values("timestamp").reset_index(drop=True)
    verify_populations_aligned(r1m_pop, r_tilde_pop)
    if not np.array_equal(r1m_pop["timestamp"].to_numpy(), r_tilde_pop["timestamp"].to_numpy()):
        raise ValueError("r1m_pop y r_tilde_pop no comparten los mismos timestamps en el mismo orden.")

    r_raw = r1m_pop["r_1m"].to_numpy(dtype=float)
    r_adj = r_tilde_pop["r_tilde"].to_numpy(dtype=float)
    trading_date = r1m_pop["trading_date"].to_numpy()
    segment = r1m_pop["segment_label"].astype(str).to_numpy()
    year = r1m_pop["year_ny"].to_numpy()
    unique_days, day_index = np.unique(trading_date, return_inverse=True)
    n_days = int(unique_days.size)

    # --- Etapa 2: sigma_hat / z_t de TDA-10 + causalidad (bloqueante) -------
    timer.stage("sigma_hat causal y z_t (definiciones de TDA-10) + verificacion de ausencia de look-ahead")
    causality_sigma = run_causality_checks(r_raw, (PRIMARY_ESTIMATOR, PRIMARY_ESTIMATOR_CLOCK_ADJUSTED))
    fam, par, _ = PRIMARY_ESTIMATOR
    burn = burn_in_length(fam, par)
    sigma_raw = build_sigma_hat(r_raw, fam, par)
    z_raw = standardize_return(r_raw, sigma_raw, burn)
    fam_a, par_a, _ = PRIMARY_ESTIMATOR_CLOCK_ADJUSTED
    sigma_adj = build_sigma_hat(r_adj, fam_a, par_a)
    z_adj = standardize_return(r_adj, sigma_adj, burn_in_length(fam_a, par_a))

    # --- Etapa 3: estados de volatilidad (causal + muestra completa) --------
    timer.stage("Estados de volatilidad: decil CAUSAL (jornadas anteriores) y decil de muestra completa (TDA-10)")
    # Auditoria correctiva A2: la verificacion usa la MISMA mascara de
    # elegibilidad (z finito: excluye el quemado) que el decil efectivo,
    # y se ejecuta para los dos bloques.
    causality_decile = verify_causal_decile_no_lookahead(sigma_raw, trading_date, valid=np.isfinite(z_raw))
    causality_decile_adj = verify_causal_decile_no_lookahead(sigma_adj, trading_date, valid=np.isfinite(z_adj))
    for c in causality_decile["checks"]:
        c["block"] = "raw"
    for c in causality_decile_adj["checks"]:
        c["block"] = "clock_adjusted"
    causality_decile = {
        "checks": causality_decile["checks"] + causality_decile_adj["checks"],
        "passed": bool(causality_decile["passed"] and causality_decile_adj["passed"]),
    }
    if not causality_decile["passed"]:
        raise CausalDecileLeakageError(
            "El decil causal de volatilidad cambio al perturbar informacion de la misma jornada o posterior, o difiere al truncar el futuro."
        )
    deciles = {}
    for name, sigma_hat, z in (("raw", sigma_raw, z_raw), ("clock_adjusted", sigma_adj, z_adj)):
        usable = np.isfinite(z)
        causal = causal_volatility_decile(sigma_hat, trading_date, valid=usable)
        full = assign_volatility_decile(np.where(usable, sigma_hat, np.nan))
        deciles[name] = {"VOL_DECILE_CAUSAL": causal, "VOL_DECILE_FULLSAMPLE": full}
    size_rows = []
    for name, d in deciles.items():
        for gk, lab in d.items():
            counts = pd.Series(lab).dropna().astype(int).value_counts().sort_index()
            for g, c in counts.items():
                size_rows.append({"block": name, "grouping": gk, "group": int(g), "n": int(c)})
            size_rows.append({"block": name, "grouping": gk, "group": "SIN_ESTADO", "n": int(np.isnan(lab).sum())})
    decile_sizes = pd.DataFrame(size_rows)

    # --- Etapa 4: cuantiles + bootstrap conjunto por jornada ----------------
    timer.stage(f"Cuantiles incondicionales/condicionales + bootstrap de bloques por jornada (n_boot={config.tda12_n_boot}) para 4 series + sensibilidad a bloques de {BLOCK_LENGTH_SENSITIVITY_DAYS} jornadas")
    z_label, z_label_adj = _series_label(PRIMARY_ESTIMATOR), _series_label(PRIMARY_ESTIMATOR_CLOCK_ADJUSTED)
    series_defs = [
        ("r_1m", "raw", r_raw, False, "CAUSAL"),
        (z_label, "raw", z_raw, True, "CAUSAL"),
        ("r_tilde", "clock_adjusted", r_adj, False, "RETROSPECTIVO"),
        (z_label_adj, "clock_adjusted", z_adj, True, "RETROSPECTIVO"),
    ]
    quantile_tables, shape_tables, shape_tables_t10, asym_tables, info_rows, block_tables = [], [], [], [], [], []
    global_label = np.full(r_raw.size, "ALL", dtype=object)
    for name, block, values, is_std, causal_class in series_defs:
        labels = {
            "GLOBAL": global_label, "SEGMENT": segment,
            "VOL_DECILE_CAUSAL": deciles[block]["VOL_DECILE_CAUSAL"], "VOL_DECILE_FULLSAMPLE": deciles[block]["VOL_DECILE_FULLSAMPLE"],
            "YEAR": year,
        }
        groups = {gk: prepare_sorted_groups(values, day_index, labels[gk]) for gk in GROUPINGS}
        point, reps = joint_day_block_bootstrap(groups, n_days, config.tda12_n_boot, DEFAULT_SEED)
        n_days_by_group = {gk: {g: int(np.unique(ds).size) for g, (_, ds) in gr.items()} for gk, gr in groups.items()}
        quantile_tables.append(build_quantile_table(name, block, groups, point, reps, n_days_by_group))
        shape_tables.append(build_shape_comparison_table(name, block, groups, point, reps, is_std))
        shape_tables_t10.append(build_shape_comparison_table(name, block, groups, point, reps, is_std, min_tail=None))
        if block == "raw":
            # Sensibilidad a la longitud de bloque (descriptiva): mismas
            # jornadas, misma semilla, bloques de L jornadas consecutivas.
            reps_by_block = {1: reps}
            for L in BLOCK_LENGTH_SENSITIVITY_DAYS:
                _, reps_by_block[L] = joint_day_block_bootstrap(groups, n_days, config.tda12_n_boot, DEFAULT_SEED, block_days=L)
            verdict_rows = shape_tables[-1]
            verdict_rows = verdict_rows.loc[(verdict_rows["level_lo"] == VERDICT_LEVEL_PAIR[0]) & (verdict_rows["grouping"].isin(VERDICT_GROUPINGS)) & (verdict_rows["normalization"] == "std")]
            focus = []
            for _, vr in verdict_rows.iterrows():
                for key in ("group_min", "group_max"):
                    match = [g for g in groups[vr["grouping"]] if str(g) == vr[key]]
                    if match and (vr["grouping"], match[0]) not in focus:
                        focus.append((vr["grouping"], match[0]))
            block_tables.append(build_block_length_sensitivity(name, groups, n_days, config.tda12_n_boot, DEFAULT_SEED, point, reps_by_block, focus))
        asym_tables.append(build_asymmetry_table(name, block, groups, point, reps))
        info_rows.append({
            "series": name, "block": block, "standardized": is_std, "causal_class": causal_class,
            "n_finite": int(np.isfinite(values).sum()), "n_excluded_nonfinite": int((~np.isfinite(values)).sum()),
        })
    quantile_table = pd.concat(quantile_tables, ignore_index=True)
    shape_table = pd.concat(shape_tables, ignore_index=True)
    shape_table_tda10_eligibility = pd.concat(shape_tables_t10, ignore_index=True)
    block_sensitivity = pd.concat(block_tables, ignore_index=True) if block_tables else pd.DataFrame()
    asymmetry_table = pd.concat(asym_tables, ignore_index=True)

    # --- Etapa 5: sensibilidad al estimador de volatilidad -------------------
    timer.stage("Sensibilidad de la comparacion de forma a las 6 configuraciones causales de sigma_hat (TDA-10)")
    estimator_sensitivity = build_estimator_sensitivity(r_raw, trading_date, day_index, segment, year)

    # --- Etapa 6: veredicto TH26 e implicacion para STOP-13 ------------------
    timer.stage("Veredicto TH26 (escala vs forma) e implicacion para STOP-13")
    th26 = decide_th26(shape_table, estimator_sensitivity, z_label)
    th26_clock_adjusted = decide_th26(shape_table, pd.DataFrame(), z_label_adj)
    stop13 = decide_stop13_implication(th26)
    one_sided_proposal = evaluate_one_sided_proposal(shape_table, z_label)
    alternative_readings = build_alternative_readings(shape_table, shape_table_tda10_eligibility, z_label, th26)

    stage_timings = timer.finish()
    return TDA12Result(
        n_population=int(r_raw.size), n_days=n_days, primary_estimator=PRIMARY_ESTIMATOR,
        series_info=pd.DataFrame(info_rows), causality_sigma=causality_sigma, causality_decile=causality_decile,
        decile_sizes=decile_sizes, quantile_table=quantile_table, shape_table=shape_table, asymmetry_table=asymmetry_table,
        estimator_sensitivity=estimator_sensitivity, shape_table_tda10_eligibility=shape_table_tda10_eligibility,
        alternative_readings=alternative_readings, one_sided_proposal=one_sided_proposal, block_sensitivity=block_sensitivity, th26=th26, th26_clock_adjusted=th26_clock_adjusted, stop13=stop13,
        n_boot=int(config.tda12_n_boot), seed=DEFAULT_SEED, stage_timings=stage_timings,
    )
