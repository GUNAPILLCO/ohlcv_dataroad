"""Complemento TH10 -- Escalado de la varianza con el horizonte (pre-TDA-08).

Resuelve TH10 (``docs/methodology/Tsay_empirical_hypotheses_backlog.md``,
seccion "TH10"), diferida explicitamente desde TDA-04 (ver
``reports/mnq/TDA04_variables_analisis.md``, seccion 8) y recomendada por
TDA-04/05/06/07 para retomarse "en un punto conveniente antes de
TDA-08/TDA-09".

Pregunta que responde, en una frase: ¿``Var(r[h]) ~ h * Var(r[1])``
(escalado aproximadamente lineal, compatible con poca dependencia lineal
acumulativa), o se desvia sistematicamente -- y en que direccion (h muy
pequeño = reversion; h grande = persistencia)?

Resultado formal principal: la pendiente ``beta`` de
``log Var(r[h]) = alpha + beta * log(h)``. Diagnostico BARATO -- NO
calcula ACF/PACF, NO estudia dependencia formal (eso es TDA-08), y sus
interpretaciones (beta <1/~1/>1) son estrictamente diagnosticas, nunca
evidencia de predictibilidad ni señal de trading.

Que NO hace este modulo (deliberadamente, fuera de alcance):

- No calcula ACF/PACF ni ajusta AR/MA/ARMA (TDA-08).
- No estudia volatility clustering ni ajusta GARCH (TDA-09).
- No ejecuta EVT, no crea features ni targets, no entrena modelos.
- No optimiza los horizontes tras ver el resultado -- la grilla
  ``[1,2,5,10,15,30,60]`` esta predeclarada, fija antes de calcular nada.
- No usa ``r_tilde`` como retorno multi-horizonte (esta dividido por
  ``s(m)``, no es aditivo de forma literal a varios minutos).
- No modifica ningun artefacto de TDA-00..TDA-07: reutiliza
  ``load_tda04_inputs``/``verify_timestamp_alignment``/``compute_hac_block_ids``
  de ``tda07_marginal_distribution.py`` (misma logica de continuidad
  temporal ya auditada, sin duplicarla).
- No abre ``data/raw/`` ni ``config.holdout_files``.
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
from ohlcv_dataroad.ingest.session_calendar import NY_TZ
from ohlcv_dataroad.ingest.tda07_marginal_distribution import (
    compute_hac_block_ids as compute_continuity_block_ids,
)
from ohlcv_dataroad.ingest.tda07_marginal_distribution import (
    load_tda04_inputs,
    verify_timestamp_alignment,
)

# ---------------------------------------------------------------------------
# Convenciones PREDECLARADAS (fijadas antes de mirar el resultado real)
# ---------------------------------------------------------------------------

# Grilla diagnostica de horizontes (minutos) -- seccion 4 de la tarea.
# NO es una seleccion de horizontes de prediccion; NO se ajusta tras ver
# el resultado.
HORIZONS: tuple[int, ...] = (1, 2, 5, 10, 15, 30, 60)

# Convencion de varianza muestral -- ddof=1, la misma que
# ``compute_moments_quantiles``/``summarize_resolution`` en TDA-05/07.
VAR_DDOF = 1

DEFAULT_N_BOOT = 300

# Años completos vs parciales -- misma particion que TDA-05/06/07.
COMPLETE_YEARS = (2020, 2021, 2022, 2023, 2024)
PARTIAL_YEARS = (2019, 2025)


# ---------------------------------------------------------------------------
# Construccion de r[h] con INVARIANTE DE CADENA COMPLETA
# ---------------------------------------------------------------------------

def build_run_length(block_id_full: np.ndarray) -> np.ndarray:
    """Posicion (1-indexada) de cada fila dentro de su bloque de continuidad; 0 si la fila es invalida.

    Entrada: ``block_id_full`` -- un entero por fila de la serie canonica
    COMPLETA (no solo las validas), con ``-1`` en las filas invalidas y un
    entero >=0 CONSTANTE dentro de cada tramo de continuidad genuina
    (``compute_continuity_block_ids`` aplicado a la subpoblacion valida,
    con las filas invalidas reinsertadas como ``-1`` en su posicion
    original -- ver ``build_horizon_returns``).

    La posicion dentro del bloque ``k`` (1, 2, 3, ... contando desde el
    primer minuto valido de ESE bloque) es exactamente el numero de
    retornos de 1 minuto validos y encadenados que terminan en esa fila
    -- por construccion de ``compute_continuity_block_ids`` (bloque
    constante si y solo si delta=60s y mismo ``trading_date`` respecto de
    la fila anterior, la misma condicion que TDA-04 exige para
    ``r_1m_valid=True``). Por tanto ``r[h]`` es construible en la fila
    ``t`` si y solo si ``k >= h``.

    Salida: array de enteros, misma longitud que ``block_id_full``.
    """
    s = pd.Series(block_id_full)
    new_group = s.ne(s.shift(1)).cumsum()
    counts = (s.groupby(new_group).cumcount() + 1).to_numpy()
    return np.where(block_id_full == -1, 0, counts)


def build_horizon_returns(
    variables: pd.DataFrame, validity: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS
) -> pd.DataFrame:
    """Construye ``r[h] = ln(C_t / C_{t-h})`` para cada ``h`` de ``horizons``, con la invariante de cadena completa.

    INVARIANTE CENTRAL (seccion 3 de la tarea): ``r[h]`` en la fila ``t``
    solo se acepta si existen ``h`` retornos de 1 minuto CONSECUTIVOS y
    VALIDOS que formen una cadena continua terminando en ``t`` -- nunca
    basta con comprobar solo los dos extremos. Esto se verifica via
    ``run_length`` (``build_run_length``): la fila ``t`` soporta ``r[h]``
    si y solo si ``run_length[t] >= h``.

    Por que esto excluye TODO lo que la seccion 3 exige excluir:
    ``run_length`` se construye sobre bloques de continuidad
    (``compute_continuity_block_ids``, reutilizada de TDA-07 sin
    modificarla) que YA exigen, entre cada par de filas consecutivas del
    bloque, delta=60s exacto y mismo ``trading_date`` -- la MISMA
    condicion que TDA-04 exige para ``r_1m_valid=True``
    (``NON_CONSECUTIVE_MINUTE``/``TRADING_DATE_BOUNDARY`` rompen la
    cadena por construccion). El roll (``ROLL_BOUNDARY``) queda cubierto
    porque TDA-04 certifico (informe, §5) que TODO roll coincide con un
    cambio de ``trading_date`` -- y un cambio de ``segment_id``
    incompatible coincide, a su vez, EXACTAMENTE con
    ``is_roll_boundary`` (TDA-04, informe, §5: "``segment_changed`` =
    ``contract_changed`` = ``is_roll_boundary``" verificado sobre las 21
    fronteras reales) -- asi que excluir roll excluye tambien todo cambio
    de ``segment_id``, sin necesitar una comprobacion separada. Cualquier
    fila con ``r_1m_valid=False`` (por la razon que sea) nunca aporta un
    eslabon a ninguna cadena, porque solo las filas ``r_1m_valid=True``
    entran en el calculo de bloques.

    Entrada: ``variables``/``validity`` de TDA-04 (mismo orden de
    timestamp, verificado con ``verify_timestamp_alignment``, reutilizada
    de TDA-07).

    Salida: ``DataFrame`` con ``timestamp``, ``trading_date``,
    ``year_ny``, ``run_length``, y una columna ``r_{h}`` por cada
    horizonte -- ``NaN`` donde ``run_length < h``. Para ``h=1``, la
    poblacion no nula de ``r_1`` es EXACTAMENTE la poblacion
    ``r_1m_valid=True`` de TDA-04 (``run_length>=1`` equivale a
    ``block_id_full != -1`` equivale a ``r_1m_valid``), y sus valores
    coinciden exactamente con ``r_1m``.
    """
    df = variables.sort_values("timestamp").reset_index(drop=True)
    v = validity.sort_values("timestamp").reset_index(drop=True)
    verify_timestamp_alignment(df, v)

    r_1m_valid = v["r_1m_valid"].to_numpy()
    close = df["close"].to_numpy(dtype=float)
    n = len(df)

    valid_positions = np.where(r_1m_valid)[0]
    block_ids_valid = compute_continuity_block_ids(
        df["timestamp"].iloc[valid_positions], df["trading_date"].iloc[valid_positions]
    )
    block_id_full = np.full(n, -1, dtype=int)
    block_id_full[valid_positions] = block_ids_valid

    run_length = build_run_length(block_id_full)

    ts_ny = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(NY_TZ)

    out = {
        "timestamp": df["timestamp"],
        "trading_date": df["trading_date"],
        "year_ny": ts_ny.dt.year.to_numpy(),
        "run_length": run_length,
    }
    for h in horizons:
        prev_close = pd.Series(close).shift(h).to_numpy()
        r_h = np.log(close / prev_close)
        r_h = np.where(run_length >= h, r_h, np.nan)
        out[f"r_{h}"] = r_h

    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Var(r[h]), VR(h) y ajuste log-log
# ---------------------------------------------------------------------------

def compute_var_table(df: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS, ddof: int = VAR_DDOF) -> pd.DataFrame:
    """Tabla ``n``/``Var(r[h])``/``VR(h)``/``log(h)``/``log(Var(r[h]))`` -- una fila por horizonte.

    ``VR(h) = Var(r[h]) / (h * Var(r[1]))`` -- ayuda interpretativa
    (seccion 5 de la tarea): ``VR~1`` escalado aprox. lineal, ``VR<1``
    sublineal, ``VR>1`` superlineal. El resultado FORMAL sigue siendo la
    pendiente ``beta`` del ajuste log-log (``fit_loglog_slope``), no esta
    tabla por si sola.
    """
    var1 = float(np.nanvar(df["r_1"].to_numpy(), ddof=ddof))
    rows = []
    for h in horizons:
        x = df[f"r_{h}"].to_numpy()
        x = x[~np.isnan(x)]
        n = int(x.size)
        var_h = float(np.var(x, ddof=ddof)) if n > ddof else float("nan")
        vr_h = var_h / (h * var1) if var1 > 0 else float("nan")
        rows.append({
            "h": h, "n": n, "var": var_h, "vr": vr_h,
            "log_h": float(np.log(h)), "log_var": float(np.log(var_h)) if var_h > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def fit_loglog_slope(h_values: np.ndarray, var_values: np.ndarray) -> tuple[float, float]:
    """Pendiente ``beta`` (y ordenada ``alpha``) de ``log(var) = alpha + beta*log(h)`` por minimos cuadrados.

    Resultado FORMAL principal de TH10 (roadmap: "la pendiente es el
    resultado"). Sin ponderar ni robustecer -- 7 puntos predeclarados
    (``HORIZONS``), ninguno descartado tras ver el ajuste.
    """
    log_h = np.log(np.asarray(h_values, dtype=float))
    log_var = np.log(np.asarray(var_values, dtype=float))
    x_bar, y_bar = log_h.mean(), log_var.mean()
    dx = log_h - x_bar
    denom = np.sum(dx**2)
    beta = float(np.sum(dx * (log_var - y_bar)) / denom) if denom > 0 else float("nan")
    alpha = float(y_bar - beta * x_bar)
    return beta, alpha


# ---------------------------------------------------------------------------
# Bootstrap de bloques por jornada -- incertidumbre de beta (G5)
# ---------------------------------------------------------------------------

def _day_block_index_map(trading_date: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Precalcula los indices de fila de cada ``trading_date`` -- mismo patron que TDA-05/06/07."""
    unique_dates, inverse = np.unique(trading_date, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    boundaries = np.searchsorted(sorted_inverse, np.arange(len(unique_dates) + 1))
    date_to_rows = [order[boundaries[i]:boundaries[i + 1]] for i in range(len(unique_dates))]
    return unique_dates, date_to_rows


def bootstrap_beta(
    df: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS, n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0, ddof: int = VAR_DDOF,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap de BLOQUES de jornada (G5, mismo motor conceptual que TDA-05/06/07): remuestrea dias completos.

    Para CADA repeticion (seccion 6.A de la tarea): remuestrea
    ``trading_date`` completas con reemplazo, recalcula ``Var(r[h])``
    para TODOS los horizontes sobre esa remuestra, y vuelve a ajustar
    ``beta``. Las ventanas overlapping NUNCA se tratan como
    observaciones iid -- la unidad de remuestreo es la JORNADA, no la
    fila.

    Salida: ``(betas, var_matrix)`` -- ``betas`` es ``(n_boot,)``,
    ``var_matrix`` es ``(n_boot, len(horizons))`` (Var(r[h]) de cada
    repeticion, por si se quiere inspeccionar mas alla de beta).
    """
    trading_date = df["trading_date"].to_numpy()
    unique_dates, date_to_rows = _day_block_index_map(trading_date)
    n_dates = len(unique_dates)

    r_h_cols = {h: df[f"r_{h}"].to_numpy(dtype=float) for h in horizons}
    h_array = np.array(horizons, dtype=float)

    rng = np.random.default_rng(seed)
    betas = np.empty(n_boot)
    var_matrix = np.empty((n_boot, len(horizons)))

    for b in range(n_boot):
        sampled = rng.integers(0, n_dates, size=n_dates)
        idx = np.concatenate([date_to_rows[i] for i in sampled])
        var_h = np.empty(len(horizons))
        for i, h in enumerate(horizons):
            vals = r_h_cols[h][idx]
            vals = vals[~np.isnan(vals)]
            var_h[i] = np.var(vals, ddof=ddof) if vals.size > ddof else np.nan
        var_matrix[b] = var_h
        betas[b], _ = fit_loglog_slope(h_array, var_h)

    return betas, var_matrix


# ---------------------------------------------------------------------------
# Sensibilidad B: ventanas NO solapadas (seccion 6.B de la tarea)
# ---------------------------------------------------------------------------

def non_overlap_mask(run_length: np.ndarray, h: int) -> np.ndarray:
    """Selecciona ventanas de horizonte ``h`` NO solapadas -- convencion PREDECLARADA.

    Convencion: dentro de CADA bloque de continuidad (mayoritariamente
    una jornada), se seleccionan las observaciones en las posiciones
    ``k = h, 2h, 3h, ...`` contadas DESDE EL INICIO del bloque
    (``run_length % h == 0``) -- el origen de las ventanas se REINICIA en
    cada bloque nuevo, nunca cruza jornadas/huecos (los bloques ya
    respetan esa frontera por construccion, igual que en
    ``build_horizon_returns``). Es una convencion determinista, fijada
    antes de ver el resultado -- no se elige el origen que produzca la
    beta mas "atractiva".

    Salida: mascara booleana, ``True`` donde la fila es una observacion
    NO solapada de horizonte ``h``.
    """
    return (run_length > 0) & (run_length % h == 0)


def compute_var_table_non_overlap(
    df: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS, ddof: int = VAR_DDOF
) -> pd.DataFrame:
    """Igual que ``compute_var_table``, pero restringido a ventanas NO solapadas (``non_overlap_mask``)."""
    run_length = df["run_length"].to_numpy()
    var1_vals = df["r_1"].to_numpy()[non_overlap_mask(run_length, 1)]
    var1 = float(np.var(var1_vals, ddof=ddof)) if var1_vals.size > ddof else float("nan")

    rows = []
    for h in horizons:
        mask = non_overlap_mask(run_length, h)
        x = df[f"r_{h}"].to_numpy()[mask]
        x = x[~np.isnan(x)]
        n = int(x.size)
        var_h = float(np.var(x, ddof=ddof)) if n > ddof else float("nan")
        vr_h = var_h / (h * var1) if var1 > 0 else float("nan")
        rows.append({
            "h": h, "n": n, "var": var_h, "vr": vr_h,
            "log_h": float(np.log(h)), "log_var": float(np.log(var_h)) if var_h > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Estabilidad por año (sensibilidad secundaria, seccion 7 de la tarea)
# ---------------------------------------------------------------------------

def beta_by_year(df: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS, ddof: int = VAR_DDOF) -> pd.DataFrame:
    """``beta`` puntual por año (``year_ny``) -- sensibilidad secundaria, sin bootstrap (seccion 7 de la tarea).

    2020-2024 son años COMPLETOS (evidencia principal); 2019/2025 quedan
    identificados explicitamente como parciales, nunca con el mismo peso
    -- misma convencion que TDA-05/06/07. No se convierten diferencias
    pequeñas de beta entre años en "regimenes" (seccion 7): el proposito
    es solo comprobar si la DIRECCION (beta<1/~1/>1) es razonablemente
    estable.
    """
    rows = []
    for year in sorted(df["year_ny"].unique()):
        sub = df[df["year_ny"] == year]
        var1 = float(np.nanvar(sub["r_1"].to_numpy(), ddof=ddof))
        var_h = []
        n_h = []
        for h in horizons:
            x = sub[f"r_{h}"].to_numpy()
            x = x[~np.isnan(x)]
            n_h.append(int(x.size))
            var_h.append(float(np.var(x, ddof=ddof)) if x.size > ddof else float("nan"))
        var_h = np.array(var_h)
        if np.any(np.isnan(var_h)) or np.any(var_h <= 0):
            beta = float("nan")
        else:
            beta, _ = fit_loglog_slope(np.array(horizons, dtype=float), var_h)
        rows.append({
            "year": int(year), "is_complete_year": bool(year in COMPLETE_YEARS),
            "n_h1": n_h[0], "n_h_max": n_h[-1], "beta": beta,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------

@dataclass
class TH10Result:
    df: pd.DataFrame
    var_table: pd.DataFrame
    beta: float
    alpha: float
    beta_ci_lo: float
    beta_ci_hi: float
    beta_boot: np.ndarray
    var_table_non_overlap: pd.DataFrame
    beta_non_overlap: float
    beta_by_year_table: pd.DataFrame


def run_th10_analysis(config: SnapshotConfig) -> TH10Result:
    """Orquesta el complemento TH10 completo sobre las variables de TDA-04.

    Reutiliza, sin duplicar logica: la proteccion del hold-out
    (``holdout_guard.py``), la carga de TDA-04 y la invariante de
    alineacion de timestamps, y el detector de bloques de continuidad
    temporal (todos de ``tda07_marginal_distribution.py``, sin
    modificarlo).
    """
    validate_research_holdout_disjoint(config)

    variables, validity = load_tda04_inputs(config)
    last_timestamps = variables.groupby("source_file")["timestamp"].max().to_dict()
    validate_last_timestamps_before_boundary(config, last_timestamps)

    horizons = tuple(config.th10_horizons) if config.th10_horizons else HORIZONS
    n_boot = config.th10_n_boot

    df = build_horizon_returns(variables, validity, horizons)
    var_table = compute_var_table(df, horizons)
    beta, alpha = fit_loglog_slope(var_table["h"].to_numpy(), var_table["var"].to_numpy())

    beta_boot, _ = bootstrap_beta(df, horizons, n_boot=n_boot, seed=0)
    beta_ci_lo, beta_ci_hi = np.percentile(beta_boot, [2.5, 97.5])

    var_table_non_overlap = compute_var_table_non_overlap(df, horizons)
    beta_non_overlap, _ = fit_loglog_slope(
        var_table_non_overlap["h"].to_numpy(), var_table_non_overlap["var"].to_numpy()
    )

    beta_year_table = beta_by_year(df, horizons)

    return TH10Result(
        df=df, var_table=var_table, beta=beta, alpha=alpha,
        beta_ci_lo=float(beta_ci_lo), beta_ci_hi=float(beta_ci_hi), beta_boot=beta_boot,
        var_table_non_overlap=var_table_non_overlap, beta_non_overlap=beta_non_overlap,
        beta_by_year_table=beta_year_table,
    )
