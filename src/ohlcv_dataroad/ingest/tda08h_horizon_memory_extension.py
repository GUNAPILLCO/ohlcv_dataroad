"""TDA08-H -- Horizon Memory Extension (complemento ACOTADO de TDA-08).

Responde una pregunta que TDA-08 (CERRADA, ``PASS_WITH_OPEN_QUESTIONS /
CLOSED``, ver ``reports/mnq/TDA08_dependencia_lineal_media.md``) dejo sin
medir directamente:

    ¿Existe dependencia lineal entre retornos consecutivos NO SOLAPADOS
    de 30 minutos y de 60 minutos en MNQ?

Esto es DISTINTO de dos cosas ya calculadas en otras etapas:

- la ACF de ``r_1m`` a los rezagos 30/60 (TDA-08, `compute_acf` sobre la
  serie nativa de 1 minuto) -- mide la correlacion entre BARRAS DE 1
  MINUTO separadas por 30/60 minutos, no entre BLOQUES ACUMULADOS de
  30/60 minutos;
- el escalado de varianza de TH10 (``Var(r[h]) ~ h*Var(r[1])``) -- mide
  como crece la VARIANZA con el horizonte, nunca si un bloque de h
  minutos predice linealmente al siguiente bloque de h minutos.

TDA-08 NO se reabre ni se modifica. Este modulo IMPORTA, sin duplicar ni
reimplementar, las funciones ya validadas:

- ``build_horizon_returns``/``non_overlap_mask`` de TH10
  (``th10_horizon_scaling.py``) -- construccion de ``r[h] = ln(C_t/C_{t-h})``
  con la invariante de CADENA COMPLETA (``run_length>=h``, nunca solo los
  dos extremos) y seleccion de ventanas NO SOLAPADAS (origen reiniciado
  en cada bloque de continuidad -- nunca cruza `trading_date`/gap/roll).
- ``compute_block_ids``/``compute_acf``/``bootstrap_rho`` de TDA-08
  (``tda08_linear_mean_dependence.py``) -- misma topologia temporal (dos
  ventanas NO SOLAPADAS solo forman un par lag-1 si el delta REAL entre
  ellas es exactamente ``h*60`` segundos, lo que automaticamente excluye
  cualquier par que cruce una frontera de bloque, incluida la frontera
  entre el ULTIMO bloque no solapado de una jornada y el PRIMERO de la
  siguiente), Pearson pairwise-complete con centrado local (rho/beta
  DISTINTOS, nunca presupuestos iguales) y bootstrap de bloques de
  jornada con clave compuesta. Definiciones IDENTICAS a las de TDA-08
  CERRADA, sin reimplementar ni un solo detalle.

Alcance deliberadamente ACOTADO (secciones 10-11 de la tarea): SOLO
estimador (``rho_1``/``beta_1``) + bootstrap por ``trading_date``. NO se
extiende el sistema G2 (permutacion/sintetico/calibracion) a 30/60
minutos -- ver la seccion "G2" del informe para la justificacion
explicita. NO se repiten deciles de volumen, segmentos, años, meses,
ventanas de apertura/cierre, PACF, portmanteau completo ni analisis de
``r_tilde`` -- todo eso permanece exclusivamente en TDA-08 CERRADA.
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
from ohlcv_dataroad.ingest.tda07_marginal_distribution import (
    load_tda04_inputs,
    verify_timestamp_alignment,
)
from ohlcv_dataroad.ingest.tda08_linear_mean_dependence import (
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_N_BOOT,
    MIN_N_FOR_SUBGROUP_ACF,
    bootstrap_rho,
    compute_acf,
    compute_block_ids,
)
from ohlcv_dataroad.ingest.th10_horizon_scaling import build_horizon_returns, non_overlap_mask

# Horizontes objetivo de TDA08-H (seccion 3 de la tarea): la pregunta
# pendiente es especificamente 30 y 60 minutos; 1/5/10 se reincluyen como
# CONTROL DE REGRESION frente a TDA-08 CERRADA (deben reproducirla
# numericamente) y como contexto de la tabla comparativa completa.
#
# NOTA sobre 15 minutos (tambien presente en la grilla de TH10,
# ``HORIZONS=(1,2,5,10,15,30,60)``): NO se incluye aqui. No hay una razon
# metodologica que lo haga indispensable como punto intermedio entre 10 y
# 30 para responder la pregunta de esta extension (memoria a 30/60
# minutos) -- se documenta esta decision explicitamente (informe, seccion
# "Por que no 15 minutos") en vez de añadirlo silenciosamente. Si el
# patron 10->30 resultara ambiguo o no ordenado, 15 minutos seria el
# primer candidato natural para una extension futura, no de esta tarea.
HORIZONS: tuple[int, ...] = (1, 5, 10, 30, 60)


def compute_multi_horizon_memory(
    variables: pd.DataFrame, validity: pd.DataFrame, n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_BOOTSTRAP_SEED, horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    """``rho_1``/``beta_1`` (con IC 95% bootstrap de AMBOS) de retornos NO SOLAPADOS consecutivos, por horizonte.

    Misma logica que ``compute_multi_frequency_rho1`` de TDA-08 (que solo
    cubria h=1/5/10) extendida a h=30/60, con una diferencia deliberada:
    aqui se captura tambien el IC bootstrap de ``beta_1`` -- TDA-08 lo
    calculaba internamente (``bootstrap_rho`` siempre devuelve
    ``(rho_boot, beta_boot)``) pero lo descartaba, porque su tabla
    comparativa original solo reportaba el de ``rho_1``. TDA08-H necesita
    ambos (seccion 6, punto 7 de la tarea) sin modificar el archivo de
    TDA-08 (que permanece congelado) -- por eso esta funcion vive aqui,
    no en ``tda08_linear_mean_dependence.py``.

    Para cada horizonte ``h``: construye ``r[h]`` sobre TODA la topologia
    (``build_horizon_returns``, invariante de cadena completa), selecciona
    las ventanas NO SOLAPADAS (``non_overlap_mask`` para ``h>1``; para
    ``h=1`` cada barra de 1 minuto valida ya es, por definicion, una
    "ventana no solapada" -- no hay nada que solapar a esa resolucion),
    construye los bloques de continuidad de ESA topologia con el delta
    esperado ``h*60`` segundos (``compute_block_ids``) -- lo que garantiza
    que un par lag-1 solo se acepta si el delta REAL entre dos ventanas NO
    SOLAPADAS consecutivas es exactamente ``h*60`` segundos, excluyendo
    automaticamente cualquier par que cruce una frontera de bloque
    (incluida la frontera entre jornadas, ya que el origen de las
    ventanas no solapadas se reinicia en cada bloque nuevo) -- y calcula
    ``rho_1``/``beta_1`` (``compute_acf``) y sus IC 95% (``bootstrap_rho``,
    bloques de jornada, clave compuesta).

    Salida: ``DataFrame`` con ``h_minutes``, ``n``, ``n_trading_dates``,
    ``n_pairs_lag1``, ``rho_1``, ``beta_1``, ``rho_ci_lo``, ``rho_ci_hi``,
    ``beta_ci_lo``, ``beta_ci_hi``.
    """
    hdf = build_horizon_returns(variables, validity, horizons=horizons)
    rows = []
    for h in horizons:
        col = f"r_{h}"
        if h == 1:
            sub = hdf.dropna(subset=[col]).copy()
            expected_delta = 60.0
        else:
            mask = non_overlap_mask(hdf["run_length"].to_numpy(), h)
            sub = hdf.loc[mask].dropna(subset=[col]).copy()
            expected_delta = float(h * 60)
        sub = sub.sort_values("timestamp")
        n = len(sub)
        n_trading_dates = int(sub["trading_date"].nunique())

        if n < MIN_N_FOR_SUBGROUP_ACF:
            rows.append({
                "h_minutes": h, "n": n, "n_trading_dates": n_trading_dates, "n_pairs_lag1": 0,
                "rho_1": np.nan, "beta_1": np.nan,
                "rho_ci_lo": np.nan, "rho_ci_hi": np.nan, "beta_ci_lo": np.nan, "beta_ci_hi": np.nan,
            })
            continue

        block_ids = compute_block_ids(sub["timestamp"], sub["trading_date"], expected_delta)
        values = sub[col].to_numpy(dtype=float)
        acf1 = compute_acf(values, block_ids, max_lag=1)
        rho1 = float(acf1.loc[0, "rho"])
        beta1 = float(acf1.loc[0, "beta"])
        n_pairs = int(acf1.loc[0, "n_pairs"])

        rho_boot, beta_boot = bootstrap_rho(
            values, block_ids, sub["trading_date"].to_numpy(), lags=(1,), n_boot=n_boot, seed=seed
        )
        rho_finite = rho_boot[:, 0][~np.isnan(rho_boot[:, 0])]
        beta_finite = beta_boot[:, 0][~np.isnan(beta_boot[:, 0])]
        rho_ci_lo, rho_ci_hi = (np.percentile(rho_finite, [2.5, 97.5]) if rho_finite.size else (np.nan, np.nan))
        beta_ci_lo, beta_ci_hi = (np.percentile(beta_finite, [2.5, 97.5]) if beta_finite.size else (np.nan, np.nan))

        rows.append({
            "h_minutes": h, "n": n, "n_trading_dates": n_trading_dates, "n_pairs_lag1": n_pairs,
            "rho_1": rho1, "beta_1": beta1,
            "rho_ci_lo": float(rho_ci_lo), "rho_ci_hi": float(rho_ci_hi),
            "beta_ci_lo": float(beta_ci_lo), "beta_ci_hi": float(beta_ci_hi),
        })
    return pd.DataFrame(rows)


@dataclass
class TDA08HResult:
    multi_horizon: pd.DataFrame


def run_tda08h_analysis(config: SnapshotConfig) -> TDA08HResult:
    """Orquesta TDA08-H completo: SOLO memoria multi-horizonte (1/5/10/30/60), nada mas.

    Reutiliza, sin duplicar: proteccion de holdout, carga de TDA-04 e
    invariante de alineacion de timestamps (``tda07_marginal_distribution.py``,
    sin modificarlo), y ``compute_multi_horizon_memory`` (este modulo).
    """
    validate_research_holdout_disjoint(config)

    variables, validity = load_tda04_inputs(config)
    last_timestamps = variables.groupby("source_file")["timestamp"].max().to_dict()
    validate_last_timestamps_before_boundary(config, last_timestamps)
    verify_timestamp_alignment(variables, validity)

    n_boot = config.tda08_n_boot  # mismo n_boot vigente de TDA-08 (300) -- sin parametro nuevo
    multi_horizon = compute_multi_horizon_memory(
        variables, validity, n_boot=n_boot, seed=DEFAULT_BOOTSTRAP_SEED, horizons=HORIZONS
    )
    return TDA08HResult(multi_horizon=multi_horizon)
