"""Tests del complemento TH10 -- escalado de la varianza con el horizonte (pre-TDA-08).

Datos sinteticos minimos, igual que TDA-04..07: se aisla cada propiedad
(construccion de r[h] con la invariante de cadena completa, no-cruce de
trading_date/roll/hueco, conteo exacto de ventanas, ajuste log-log con
resultado conocido en casos sinteticos iid/AR(1), bootstrap reproducible,
sensibilidad no solapada, aislamiento del hold-out). No se reproducen
aqui los numeros exactos del conjunto de investigacion real (esos viven
en ``reports/mnq/TH10_escalado_varianza_horizonte.md``).
"""
from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ohlcv_dataroad.config import load_config
from ohlcv_dataroad.ingest.holdout_guard import HoldoutIsolationError
from ohlcv_dataroad.ingest.th10_horizon_scaling import (
    HORIZONS,
    beta_by_year,
    bootstrap_beta,
    build_horizon_returns,
    build_run_length,
    compute_var_table,
    compute_var_table_non_overlap,
    fit_loglog_slope,
    non_overlap_mask,
    run_th10_analysis,
)

TICK = 0.25


# ---------------------------------------------------------------------------
# Fixtures sinteticas
# ---------------------------------------------------------------------------

def _variables_and_validity(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    variables = pd.DataFrame(
        [
            {
                "timestamp": r["timestamp"], "source_file": r.get("source_file", "f1"),
                "contract": r.get("contract", "H24"), "trading_date": r["trading_date"],
                "close": r["close"],
            }
            for r in rows
        ]
    )
    validity = pd.DataFrame([{"timestamp": r["timestamp"], "r_1m_valid": r["r_1m_valid"]} for r in rows])
    return variables, validity


def _continuous_chain(
    n: int, start_ts: pd.Timestamp, trading_date: datetime.date, start_price: float = 100.0, step: float = 0.25
) -> list[dict]:
    """``n`` barras consecutivas de 1 minuto en la MISMA jornada; la primera es el ancla (invalida)."""
    ts = pd.date_range(start_ts, periods=n, freq="1min")
    rows = []
    price = start_price
    for i in range(n):
        rows.append({"timestamp": ts[i], "trading_date": trading_date, "close": price, "r_1m_valid": i > 0})
        price += step
    return rows


def _single_long_chain_with_returns(returns: np.ndarray, start_ts: pd.Timestamp, trading_date: datetime.date, start_price: float = 15000.0) -> list[dict]:
    """Cadena continua de 1 minuto cuyo ``r_1m`` en cada paso es EXACTAMENTE ``returns[i]`` (log-retorno)."""
    n = len(returns) + 1
    ts = pd.date_range(start_ts, periods=n, freq="1min")
    rows = [{"timestamp": ts[0], "trading_date": trading_date, "close": start_price, "r_1m_valid": False}]
    price = start_price
    for i, r in enumerate(returns):
        price = price * np.exp(r)
        rows.append({"timestamp": ts[i + 1], "trading_date": trading_date, "close": price, "r_1m_valid": True})
    return rows


# ---------------------------------------------------------------------------
# 1. r[h] coincide con la suma de h retornos de 1 minuto consecutivos
# ---------------------------------------------------------------------------

def test_r_h_matches_sum_of_consecutive_1m_returns():
    rows = _continuous_chain(10, pd.Timestamp("2024-01-08 10:00:00"), datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    df = build_horizon_returns(variables, validity, horizons=(1, 3))

    # r_3 en la fila 5 debe coincidir EXACTAMENTE con ln(close[5]/close[2])
    # y con la suma de r_1 en las filas 3, 4, 5.
    close = variables.sort_values("timestamp")["close"].to_numpy()
    expected_ratio = np.log(close[5] / close[2])
    expected_sum = np.log(close[3] / close[2]) + np.log(close[4] / close[3]) + np.log(close[5] / close[4])
    assert df.loc[5, "r_3"] == pytest.approx(expected_ratio, abs=1e-12)
    assert df.loc[5, "r_3"] == pytest.approx(expected_sum, abs=1e-10)


# ---------------------------------------------------------------------------
# 2 / 5. No se construye r[h] si falta un minuto intermedio (hueco / NON_CONSECUTIVE_MINUTE)
# ---------------------------------------------------------------------------

def test_r_h_is_nan_when_an_intermediate_minute_is_missing():
    ts = pd.date_range("2024-01-08 10:00:00", periods=5, freq="1min")
    rows = [
        {"timestamp": ts[0], "trading_date": datetime.date(2024, 1, 8), "close": 100.0, "r_1m_valid": False},
        {"timestamp": ts[1], "trading_date": datetime.date(2024, 1, 8), "close": 100.25, "r_1m_valid": True},
        {"timestamp": ts[2], "trading_date": datetime.date(2024, 1, 8), "close": 100.50, "r_1m_valid": True},
        # fila 3: invalidada (simula NON_CONSECUTIVE_MINUTE) -- su propio r_1m no cuenta.
        {"timestamp": ts[3], "trading_date": datetime.date(2024, 1, 8), "close": 100.75, "r_1m_valid": False},
        {"timestamp": ts[4], "trading_date": datetime.date(2024, 1, 8), "close": 101.00, "r_1m_valid": True},
    ]
    variables, validity = _variables_and_validity(rows)
    df = build_horizon_returns(variables, validity, horizons=(1, 2, 3))

    # La fila 4 (posicion 4) inicia un bloque NUEVO (run_length=1) -- r_2 y r_3
    # NO deben construirse ahi, aunque sus "endpoints" (filas 2 y 4) tengan
    # precios validos por separado.
    assert df.loc[4, "run_length"] == 1
    assert pd.isna(df.loc[4, "r_2"])
    assert pd.isna(df.loc[4, "r_3"])
    # h=1 SI es valido en la fila 4: su propio r_1m_valid es True (conecta
    # con el close de la fila 3, que sigue siendo un ancla legitima aunque
    # el r_1m PROPIO de la fila 3 -- el enlace fila2->fila3 -- fuera
    # invalido). Esa misma fila 4 inicia un bloque nuevo (run_length=1),
    # por eso h=2 y h=3 SI quedan bloqueados: atravesarian el enlace roto.
    assert not pd.isna(df.loc[4, "r_1"])


def test_r_1_is_valid_at_the_row_immediately_after_the_gap_but_starts_a_new_block():
    """La validez de r_1 en la fila t depende SOLO de la propia fila t (r_1m_valid[t]), nunca de si el r_1m DE LA FILA ANTERIOR fue valido.

    La fila 3 esta marcada ``r_1m_valid=True`` -- su r_1 = ln(close[3]/close[2])
    es valido, aunque la fila 2 (su predecesora inmediata en la tabla) haya
    sido invalidada (su PROPIO r_1m, es decir el enlace fila1->fila2, no lo
    es). El close de la fila 2 sigue existiendo y es un ancla legitima para
    el retorno de la fila 3 -- exactamente como TDA-04 define r_1m_valid
    fila por fila. Al mismo tiempo, la fila 3 SI inicia un bloque nuevo
    (run_length=1): el eslabon roto en la fila 2 impide encadenar mas alla
    de este unico retorno de 1 minuto, asi que h>=2 no seria construible
    desde aqui (ver los tests de "no cruza un hueco/NON_CONSECUTIVE_MINUTE").
    """
    ts = pd.date_range("2024-01-08 10:00:00", periods=4, freq="1min")
    rows = [
        {"timestamp": ts[0], "trading_date": datetime.date(2024, 1, 8), "close": 100.0, "r_1m_valid": False},
        {"timestamp": ts[1], "trading_date": datetime.date(2024, 1, 8), "close": 100.25, "r_1m_valid": True},
        # fila 2: invalidada (su PROPIO r_1m, fila1->fila2, no es valido) -- rompe la cadena para h>=2.
        {"timestamp": ts[2], "trading_date": datetime.date(2024, 1, 8), "close": 100.50, "r_1m_valid": False},
        {"timestamp": ts[3], "trading_date": datetime.date(2024, 1, 8), "close": 100.75, "r_1m_valid": True},
    ]
    variables, validity = _variables_and_validity(rows)
    df = build_horizon_returns(variables, validity, horizons=(1,))
    assert df.loc[3, "run_length"] == 1  # bloque nuevo -- el enlace roto esta justo detras
    assert not pd.isna(df.loc[3, "r_1"])  # pero r_1 SI es valido: depende solo de r_1m_valid[3]
    assert df.loc[3, "r_1"] == pytest.approx(np.log(100.75 / 100.50))


# ---------------------------------------------------------------------------
# 3. No cruza trading_date
# ---------------------------------------------------------------------------

def test_r_h_never_crosses_a_trading_date_boundary():
    rows_day1 = _continuous_chain(5, pd.Timestamp("2024-01-08 15:56:00"), datetime.date(2024, 1, 8), start_price=100.0)
    # Primera barra del dia 2: invalidada (TRADING_DATE_BOUNDARY), aunque el
    # delta de timestamp fuera exactamente 60s.
    rows_day2 = [
        {"timestamp": pd.Timestamp("2024-01-09 09:30:00"), "trading_date": datetime.date(2024, 1, 9), "close": 200.0, "r_1m_valid": False},
        {"timestamp": pd.Timestamp("2024-01-09 09:31:00"), "trading_date": datetime.date(2024, 1, 9), "close": 200.25, "r_1m_valid": True},
        {"timestamp": pd.Timestamp("2024-01-09 09:32:00"), "trading_date": datetime.date(2024, 1, 9), "close": 200.50, "r_1m_valid": True},
    ]
    variables, validity = _variables_and_validity(rows_day1 + rows_day2)
    df = build_horizon_returns(variables, validity, horizons=(1, 2, 5))

    last_day1_idx = 4  # ultima fila del dia 1 (posicion 4, run_length=4)
    first_day2_valid_idx = 6  # segunda fila del dia 2 (posicion 6, run_length=1)
    assert df.loc[first_day2_valid_idx, "run_length"] == 1
    assert pd.isna(df.loc[first_day2_valid_idx, "r_2"])  # cruzaria al dia 1 -- prohibido
    # r_5 tampoco existe en ningun punto: ningun bloque tiene longitud >=5.
    assert df["r_5"].notna().sum() == 0


# ---------------------------------------------------------------------------
# 4. No cruza roll / segment_id (coincide siempre con cambio de trading_date, TDA-04 S5)
# ---------------------------------------------------------------------------

def test_r_h_never_crosses_what_would_be_a_roll_boundary():
    """Un roll SIEMPRE coincide con un cambio de trading_date (TDA-04, informe S5) -- mismo mecanismo que el test anterior."""
    rows_before = _continuous_chain(3, pd.Timestamp("2024-03-14 16:57:00"), datetime.date(2024, 3, 14), start_price=100.0)
    rows_after = [
        {"timestamp": pd.Timestamp("2024-03-15 18:01:00"), "trading_date": datetime.date(2024, 3, 15), "close": 5000.0, "r_1m_valid": False},
        {"timestamp": pd.Timestamp("2024-03-15 18:02:00"), "trading_date": datetime.date(2024, 3, 15), "close": 5000.25, "r_1m_valid": True},
    ]
    variables, validity = _variables_and_validity(rows_before + rows_after)
    df = build_horizon_returns(variables, validity, horizons=(1, 2))
    last_before_idx = 2
    first_after_idx = 4
    assert df.loc[first_after_idx, "run_length"] == 1
    assert pd.isna(df.loc[first_after_idx, "r_2"])  # cruzaria el "roll" -- prohibido


# ---------------------------------------------------------------------------
# 6. h=1 reproduce exactamente la poblacion/retorno valido de TDA-04
# ---------------------------------------------------------------------------

def test_h1_reproduces_tda04_valid_population_exactly():
    rows = _continuous_chain(6, pd.Timestamp("2024-01-08 10:00:00"), datetime.date(2024, 1, 8))
    # Invalida una fila intermedia a mano.
    rows[3]["r_1m_valid"] = False
    variables, validity = _variables_and_validity(rows)
    df = build_horizon_returns(variables, validity, horizons=(1,))

    v = validity.sort_values("timestamp").reset_index(drop=True)
    expected_valid = v["r_1m_valid"].to_numpy()
    actual_valid = df["r_1"].notna().to_numpy()
    np.testing.assert_array_equal(actual_valid, expected_valid)

    c = variables.sort_values("timestamp")["close"].to_numpy()
    expected_r1 = np.log(c[1:] / c[:-1])
    np.testing.assert_allclose(df["r_1"].to_numpy()[1:][expected_valid[1:]], expected_r1[expected_valid[1:]])


def test_h1_matches_tda04_r_1m_column_directly():
    """Comparacion LITERALMENTE DIRECTA: ``TH10.r_1 == variables["r_1m"]`` -- una columna ``r_1m`` explicita en el propio DataFrame ``variables``, representativa de ``tda04_variables_1m.parquet`` (que SI tiene esa columna en el artefacto real).

    ``r_1m`` se calcula con un bucle explicito (``math.log``, fila por
    fila) -- deliberadamente un camino de codigo DISTINTO al vectorizado
    (``numpy``/``close.shift``) que usa ``build_horizon_returns`` -- para
    que la coincidencia no sea circular (comparar la misma formula contra
    si misma via el mismo codigo), sino una verificacion independiente de
    que ``TH10.r_1`` es, fila por fila, exactamente la columna ``r_1m``
    que ``variables`` ya trae incorporada -- la comparacion que exige el
    informe (§1: "h=1 reproduce exactamente la poblacion Y LOS VALORES de
    r_1m de TDA-04"), no una comparacion contra un array aparte.
    """
    import math

    rows = _continuous_chain(8, pd.Timestamp("2024-01-08 10:00:00"), datetime.date(2024, 1, 8))
    rows[4]["r_1m_valid"] = False  # invalida una fila intermedia a mano
    variables, validity = _variables_and_validity(rows)

    v = validity.sort_values("timestamp").reset_index(drop=True)
    variables = variables.sort_values("timestamp").reset_index(drop=True)
    closes = variables["close"].tolist()
    valid_flags = v["r_1m_valid"].tolist()
    # Columna r_1m AÑADIDA a variables (como en el artefacto real de
    # TDA-04): no es un array aparte, es parte del propio DataFrame que se
    # pasa a build_horizon_returns.
    variables["r_1m"] = [
        math.log(closes[i] / closes[i - 1]) if i > 0 and valid_flags[i] else np.nan
        for i in range(len(closes))
    ]

    df = build_horizon_returns(variables, validity, horizons=(1,))
    th10_r1 = df["r_1"].to_numpy()
    tda04_r1m = variables["r_1m"].to_numpy()
    valid_mask = np.array(valid_flags)

    assert np.array_equal(np.isnan(th10_r1), np.isnan(tda04_r1m))  # misma poblacion, fila por fila
    # Tolerancia estricta y justificada: ambas cantidades son
    # MATEMATICAMENTE la misma (ln(close_t/close_{t-1})), calculada por
    # dos caminos de codigo independientes -- deben coincidir hasta el
    # redondeo de punto flotante de doble precision (~1e-15 relativo),
    # nunca una tolerancia mas laxa que enmascare una discrepancia real.
    np.testing.assert_allclose(th10_r1[valid_mask], tda04_r1m[valid_mask], atol=1e-12, rtol=1e-12)
    # La comparacion literal que exige el requisito: TH10.r_1 == variables["r_1m"].
    pd.testing.assert_series_equal(
        pd.Series(th10_r1[valid_mask]), pd.Series(tda04_r1m[valid_mask]),
        check_exact=False, atol=1e-12, rtol=1e-12, check_names=False,
    )


# ---------------------------------------------------------------------------
# 7. Conteo esperado de ventanas validas sobre ejemplo sintetico conocido
# ---------------------------------------------------------------------------

def test_expected_valid_window_counts_on_a_known_chain():
    n = 10  # posicion 0 = ancla invalida; posiciones 1..9 validas (run_length 1..9)
    rows = _continuous_chain(n, pd.Timestamp("2024-01-08 10:00:00"), datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    df = build_horizon_returns(variables, validity, horizons=(1, 3, 5, 9, 10))

    for h, expected_n in [(1, 9), (3, 7), (5, 5), (9, 1), (10, 0)]:
        assert int(df[f"r_{h}"].notna().sum()) == expected_n, f"h={h}"


def test_build_run_length_matches_hand_computed_sequence():
    block_id_full = np.array([-1, 0, 0, 0, -1, 1, 1])
    run_length = build_run_length(block_id_full)
    np.testing.assert_array_equal(run_length, [0, 1, 2, 3, 0, 1, 2])


# ---------------------------------------------------------------------------
# fit_loglog_slope -- casos conocidos
# ---------------------------------------------------------------------------

def test_fit_loglog_slope_recovers_beta_exactly_on_a_perfect_power_law():
    h = np.array([1, 2, 5, 10, 15, 30, 60], dtype=float)
    true_beta, true_alpha = 1.3, 0.5
    var_h = np.exp(true_alpha + true_beta * np.log(h))
    beta, alpha = fit_loglog_slope(h, var_h)
    assert beta == pytest.approx(true_beta, abs=1e-10)
    assert alpha == pytest.approx(true_alpha, abs=1e-10)


# ---------------------------------------------------------------------------
# 9. beta ~= 1 con retornos independientes (serie sintetica suficientemente grande)
# ---------------------------------------------------------------------------

def test_beta_is_close_to_one_for_iid_returns():
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0, 0.0008, size=8000)
    rows = _single_long_chain_with_returns(returns, pd.Timestamp("2024-01-08 09:30:00"), datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    df = build_horizon_returns(variables, validity, horizons=HORIZONS)
    var_table = compute_var_table(df, HORIZONS)
    beta, _ = fit_loglog_slope(var_table["h"].to_numpy(), var_table["var"].to_numpy())
    assert abs(beta - 1.0) < 0.15


# ---------------------------------------------------------------------------
# 10 / 11. Dependencia negativa -> beta<1 ; dependencia positiva -> beta>1
# ---------------------------------------------------------------------------

def _ar1_returns(n: int, phi: float, sigma_e: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma_e, size=n)
    r = np.empty(n)
    r[0] = e[0]
    for i in range(1, n):
        r[i] = phi * r[i - 1] + e[i]
    return r


def test_beta_below_one_for_negatively_autocorrelated_returns():
    returns = _ar1_returns(8000, phi=-0.35, sigma_e=0.0008, seed=1)
    rows = _single_long_chain_with_returns(returns, pd.Timestamp("2024-01-08 09:30:00"), datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    df = build_horizon_returns(variables, validity, horizons=HORIZONS)
    var_table = compute_var_table(df, HORIZONS)
    beta, _ = fit_loglog_slope(var_table["h"].to_numpy(), var_table["var"].to_numpy())
    assert beta < 0.9


def test_beta_above_one_for_positively_autocorrelated_returns():
    returns = _ar1_returns(8000, phi=0.35, sigma_e=0.0008, seed=2)
    rows = _single_long_chain_with_returns(returns, pd.Timestamp("2024-01-08 09:30:00"), datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    df = build_horizon_returns(variables, validity, horizons=HORIZONS)
    var_table = compute_var_table(df, HORIZONS)
    beta, _ = fit_loglog_slope(var_table["h"].to_numpy(), var_table["var"].to_numpy())
    assert beta > 1.1


# ---------------------------------------------------------------------------
# 8. Bootstrap reproducible con la misma semilla
# ---------------------------------------------------------------------------

def test_bootstrap_beta_is_reproducible_with_same_seed():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.0008, size=3000)
    rows = _single_long_chain_with_returns(returns, pd.Timestamp("2024-01-08 09:30:00"), datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    df = build_horizon_returns(variables, validity, horizons=(1, 2, 5, 10))
    betas_1, _ = bootstrap_beta(df, horizons=(1, 2, 5, 10), n_boot=20, seed=7)
    betas_2, _ = bootstrap_beta(df, horizons=(1, 2, 5, 10), n_boot=20, seed=7)
    np.testing.assert_array_equal(betas_1, betas_2)


# ---------------------------------------------------------------------------
# 12. Sensibilidad no solapada: nunca introduce cruces de jornada/hueco
# ---------------------------------------------------------------------------

def test_non_overlap_selection_never_yields_a_nan_return():
    n = 12
    rows = _continuous_chain(n, pd.Timestamp("2024-01-08 10:00:00"), datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    df = build_horizon_returns(variables, validity, horizons=(3,))
    mask = non_overlap_mask(df["run_length"].to_numpy(), 3)
    selected = df.loc[mask, "r_3"]
    assert len(selected) > 0
    assert selected.notna().all()  # ninguna seleccion no-solapada produce NaN


def test_non_overlap_mask_selects_multiples_of_h_within_block_and_resets_across_blocks():
    block_id_full = np.array([-1, 0, 0, 0, 0, -1, 1, 1, 1])
    run_length = build_run_length(block_id_full)  # [0,1,2,3,4,0,1,2,3]
    mask = non_overlap_mask(run_length, 2)
    np.testing.assert_array_equal(mask, [False, False, True, False, True, False, False, True, False])


def test_var_table_non_overlap_matches_direct_ratio_formula():
    n = 12
    rows = _continuous_chain(n, pd.Timestamp("2024-01-08 10:00:00"), datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    df = build_horizon_returns(variables, validity, horizons=(1, 4))
    table = compute_var_table_non_overlap(df, horizons=(1, 4))
    row_h4 = table[table["h"] == 4].iloc[0]
    mask = non_overlap_mask(df["run_length"].to_numpy(), 4)
    expected_var = float(np.var(df.loc[mask, "r_4"].to_numpy(), ddof=1))
    assert row_h4["var"] == pytest.approx(expected_var)
    assert int(row_h4["n"]) == int(mask.sum())


# ---------------------------------------------------------------------------
# beta_by_year -- años completos vs parciales identificados
# ---------------------------------------------------------------------------

def test_beta_by_year_flags_complete_years_correctly():
    rows_2020 = _continuous_chain(50, pd.Timestamp("2020-06-01 10:00:00"), datetime.date(2020, 6, 1))
    rows_2019 = _continuous_chain(50, pd.Timestamp("2019-12-23 10:00:00"), datetime.date(2019, 12, 23))
    variables, validity = _variables_and_validity(rows_2019 + rows_2020)
    df = build_horizon_returns(variables, validity, horizons=(1, 2, 5))
    table = beta_by_year(df, horizons=(1, 2, 5))
    row_2020 = table[table["year"] == 2020].iloc[0]
    row_2019 = table[table["year"] == 2019].iloc[0]
    assert bool(row_2020["is_complete_year"]) is True
    assert bool(row_2019["is_complete_year"]) is False


# ---------------------------------------------------------------------------
# Orquestador end-to-end (config sintetica, sin abrir raw/holdout)
# ---------------------------------------------------------------------------

def _synthetic_multi_day_rows(n_days: int, start: datetime.date, bars_per_day: int = 80, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    day_i = 0
    added = 0
    price = 15000.0
    first_row = True
    while added < n_days:
        date = start + datetime.timedelta(days=day_i)
        day_i += 1
        if date.weekday() >= 5:
            continue
        base_ts = pd.Timestamp(date) + pd.Timedelta(hours=9, minutes=30)
        for b in range(bars_per_day):
            ts = base_ts + pd.Timedelta(minutes=b)
            price = price * (1.0 + rng.normal(0.0, 0.0005))
            price = round(price / TICK) * TICK
            if first_row:
                rows.append({"timestamp": ts, "close": price, "trading_date": date, "r_1m_valid": False})
                first_row = False
            else:
                rows.append({"timestamp": ts, "close": price, "trading_date": date, "r_1m_valid": True})
        added += 1
    return rows


def _make_config(tmp_path: Path, research_files, holdout_files=None, boundary_utc="2099-01-01 00:00:00"):
    config_dir = tmp_path / "configs"
    config_dir.mkdir(exist_ok=True)
    config_yaml = {
        "instrument": "MNQ",
        "raw_data": {
            "dir": "data/raw/mnq", "file_pattern": "*.Last.txt", "separator": ";",
            "columns": ["timestamp", "open", "high", "low", "close", "volume"],
            "timestamp_format": "%Y%m%d %H%M%S", "timestamp_raw_timezone": "UTC",
        },
        "instrument_spec": {"tick_size": TICK, "tick_size_source": "test"},
        "holdout": {
            "boundary_utc": boundary_utc, "boundary_source": "test",
            "research_files": research_files, "holdout_files": holdout_files or [],
        },
        "tda00": {
            "output_dir_reports": "reports/mnq", "output_dir_interim": "data/interim/mnq",
            "inventory_report": "TDA00_inventario.md", "violations_report": "TDA00_violaciones.csv",
            "per_file_summary": "TDA00_resumen_por_archivo.csv", "bad_data_mask": "tda00_bad_data_mask.parquet",
        },
        "tda04": {
            "output_dir_reports": "reports/mnq", "output_dir_interim": "data/interim/mnq",
            "report_name": "TDA04_variables_analisis.md", "variables_parquet_name": "tda04_variables_1m.parquet",
            "validity_mask_parquet_name": "tda04_return_validity_mask.parquet",
            "losses_by_cause_csv": "TDA04_perdidas_por_causa.csv", "th07_r_vs_R_csv": "TDA04_th07_r_vs_R.csv",
        },
        "th10": {
            "output_dir_reports": "reports/mnq",
            "horizons": [1, 2, 5, 10],
            "n_boot": 5,
            "report_name": "TH10_escalado_varianza_horizonte.md",
            "var_by_horizon_csv": "TH10_var_by_horizon.csv",
            "var_by_horizon_non_overlap_csv": "TH10_var_by_horizon_no_solapado.csv",
            "beta_by_year_csv": "TH10_beta_por_anio.csv",
            "var_h_plot_png": "TH10_var_h_loglog.png",
        },
    }
    config_path = config_dir / "mnq_snapshot.yaml"
    config_path.write_text(yaml.safe_dump(config_yaml), encoding="utf-8")
    return load_config(config_path)


def _write_tda04_artifacts(config, variables: pd.DataFrame, validity: pd.DataFrame) -> None:
    config.interim_dir.mkdir(parents=True, exist_ok=True)
    variables.to_parquet(config.tda04_variables_parquet_path, index=False)
    validity.to_parquet(config.tda04_validity_mask_parquet_path, index=False)


def test_run_th10_raises_when_research_and_holdout_overlap(tmp_path):
    config = _make_config(
        tmp_path, research_files=["00_mnq_03_24.Last.txt"],
        holdout_files=["00_mnq_03_24.Last.txt", "no_existe.Last.txt"],
    )
    with pytest.raises(HoldoutIsolationError, match="disjuntos"):
        run_th10_analysis(config)


def test_run_th10_never_opens_any_raw_or_holdout_file(tmp_path):
    config = _make_config(
        tmp_path, research_files=["00_mnq_03_24.Last.txt"],
        holdout_files=["archivo_que_no_existe.Last.txt"],
    )
    rows = _synthetic_multi_day_rows(n_days=15, start=datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    _write_tda04_artifacts(config, variables, validity)
    assert not (config.raw_dir / "00_mnq_03_24.Last.txt").exists()
    result = run_th10_analysis(config)
    assert len(result.df) == len(rows)
    assert set(result.var_table["h"]) == {1, 2, 5, 10}


def test_run_th10_end_to_end_produces_all_tables(tmp_path):
    config = _make_config(tmp_path, research_files=["00_mnq_03_24.Last.txt"])
    rows = _synthetic_multi_day_rows(n_days=15, start=datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    _write_tda04_artifacts(config, variables, validity)
    result = run_th10_analysis(config)

    assert not result.var_table.empty
    assert not result.var_table_non_overlap.empty
    assert not result.beta_by_year_table.empty
    assert np.isfinite(result.beta)
    assert result.beta_ci_lo <= result.beta_ci_hi
    assert len(result.beta_boot) == 5  # n_boot de la config sintetica
