"""Tests de TDA-08 -- dependencia lineal en la media (ACF/PACF/portmanteau, TH16-TH18).

CORRECCION METODOLOGICA: reescrito para la version corregida de
``tda08_linear_mean_dependence.py`` -- correlacion ACF pairwise sobre
pares validos (no atenuada por ``T``), rho/beta separados, agrupacion
por decil/segmento/año que construye los pares sobre la topologia
COMPLETA (no exige que ``t-1`` comparta el grupo de ``t``), nulls G2 que
preservan estructura de reloj/escala, rezagos largos ``NOT_ESTIMABLE``
explicito. No se reproducen aqui los numeros exactos del conjunto de
investigacion real (esos viven en
``reports/mnq/TDA08_dependencia_lineal_media.md``).
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
from ohlcv_dataroad.ingest.tda07_marginal_distribution import TimestampAlignmentError
from ohlcv_dataroad.ingest.tda08_linear_mean_dependence import (
    BOOTSTRAP_LAGS,
    WINDOW_CLOSE,
    WINDOW_OPEN,
    acf_by_group,
    acf_by_group_strict_same_block,
    analyze_window,
    annotate_portmanteau_calibration,
    assign_volume_decile,
    bartlett_se,
    bootstrap_rho,
    compute_acf,
    compute_block_ids,
    compute_multi_frequency_rho1,
    compute_portmanteau_q,
    draw_synthetic_empirical_sample,
    durbin_levinson_pacf,
    g2_combined_calibration_summary,
    g2_combined_portmanteau_summary,
    g2_null1_calibration_summary,
    g2_null1_portmanteau_summary,
    g2_permutation_null_by_minute,
    g2_synthetic_empirical_null,
    g2_synthetic_null_moment_check,
    rolling_rho1_by_month,
    run_tda08_analysis,
    translate_dependence_to_ticks,
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
                "segment_id": r.get("segment_id", 1),
                "open": r.get("open", r["close"]), "high": r.get("high", r["close"] + 0.5),
                "low": r.get("low", r["close"] - 0.5), "close": r["close"], "volume": r.get("volume", 10),
                "r_1m": r.get("r_1m", np.nan),
            }
            for r in rows
        ]
    )
    validity = pd.DataFrame(
        [{"timestamp": r["timestamp"], "r_1m_valid": r["r_1m_valid"], "invalid_reason": r.get("invalid_reason", "VALID" if r["r_1m_valid"] else "NON_CONSECUTIVE_MINUTE")} for r in rows]
    )
    return variables, validity


def _continuous_chain(n: int, start_ts: pd.Timestamp, trading_date: datetime.date, start_price: float = 100.0, step: float = 0.25, volume: int = 10) -> list[dict]:
    ts = pd.date_range(start_ts, periods=n, freq="1min")
    rows = []
    price = start_price
    prev_price = None
    for i in range(n):
        row = {"timestamp": ts[i], "trading_date": trading_date, "close": price, "r_1m_valid": i > 0, "volume": volume}
        if i > 0:
            row["r_1m"] = float(np.log(price / prev_price))
        rows.append(row)
        prev_price = price
        price += step
    return rows


def _single_long_chain_with_returns(returns: np.ndarray, start_ts: pd.Timestamp, trading_date: datetime.date, start_price: float = 15000.0, volume: int = 20) -> list[dict]:
    n = len(returns) + 1
    ts = pd.date_range(start_ts, periods=n, freq="1min")
    rows = [{"timestamp": ts[0], "trading_date": trading_date, "close": start_price, "r_1m_valid": False, "volume": volume}]
    price = start_price
    for i, r in enumerate(returns):
        price = price * np.exp(r)
        rows.append({"timestamp": ts[i + 1], "trading_date": trading_date, "close": price, "r_1m_valid": True, "volume": volume})
    return rows


def _ar1_returns(n: int, phi: float, sigma_e: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma_e, size=n)
    r = np.empty(n)
    r[0] = e[0]
    for i in range(1, n):
        r[i] = phi * r[i - 1] + e[i]
    return r


def attach_calendar_for_test(variables, validity):
    from ohlcv_dataroad.ingest.tda06_intraday_calendar_profile import attach_calendar_fields
    df = attach_calendar_fields(variables, validity)
    return df.loc[df["r_1m_valid"]].copy()


# ---------------------------------------------------------------------------
# Topologia temporal -- compute_block_ids (sin cambios respecto del cierre anterior)
# ---------------------------------------------------------------------------

def test_compute_block_ids_single_continuous_run_is_one_block():
    ts = pd.date_range("2024-01-08 09:30:00", periods=10, freq="1min")
    trading_date = np.full(10, datetime.date(2024, 1, 8), dtype=object)
    block_ids = compute_block_ids(pd.Series(ts), pd.Series(trading_date), 60.0)
    assert (block_ids == 0).all()


def test_compute_block_ids_never_crosses_trading_date_even_with_60s_delta():
    ts = pd.to_datetime(["2024-01-08 23:59:00", "2024-01-09 00:00:00"])
    trading_date = np.array([datetime.date(2024, 1, 8), datetime.date(2024, 1, 9)], dtype=object)
    block_ids = compute_block_ids(pd.Series(ts), pd.Series(trading_date), 60.0)
    np.testing.assert_array_equal(block_ids, [0, 1])


def test_compute_block_ids_never_crosses_a_gap_left_by_an_invalidated_row():
    ts = pd.to_datetime([
        "2024-01-08 09:30:00", "2024-01-08 09:31:00", "2024-01-08 09:32:00",
        "2024-01-08 11:00:00", "2024-01-08 11:01:00",
    ])
    trading_date = np.full(5, datetime.date(2024, 1, 8), dtype=object)
    block_ids = compute_block_ids(pd.Series(ts), pd.Series(trading_date), 60.0)
    np.testing.assert_array_equal(block_ids, [0, 0, 0, 1, 1])


def test_compute_block_ids_never_crosses_what_would_be_a_roll_boundary():
    ts = pd.to_datetime(["2024-03-14 16:57:00", "2024-03-15 18:01:00"])
    trading_date = np.array([datetime.date(2024, 3, 14), datetime.date(2024, 3, 15)], dtype=object)
    block_ids = compute_block_ids(pd.Series(ts), pd.Series(trading_date), 60.0)
    np.testing.assert_array_equal(block_ids, [0, 1])


def test_compute_block_ids_with_5_minute_delta_for_downsampled_series():
    ts = pd.to_datetime(["2024-01-08 09:30:00", "2024-01-08 09:35:00", "2024-01-08 09:40:00", "2024-01-08 09:50:00"])
    trading_date = np.full(4, datetime.date(2024, 1, 8), dtype=object)
    block_ids = compute_block_ids(pd.Series(ts), pd.Series(trading_date), 300.0)
    np.testing.assert_array_equal(block_ids, [0, 0, 0, 1])


# ---------------------------------------------------------------------------
# compute_acf -- correlacion PAIRWISE (CORREGIDA)
# ---------------------------------------------------------------------------

def test_compute_acf_excludes_cross_block_pairs_and_marks_not_estimable():
    values = np.array([1.0, 2.0, 3.0, 100.0, 101.0, 102.0])
    block_ids = np.array([0, 0, 0, 1, 1, 1])
    acf_tab = compute_acf(values, block_ids, max_lag=3)
    assert acf_tab.loc[2, "n_pairs"] == 0
    assert not acf_tab.loc[2, "estimable"]
    assert np.isnan(acf_tab.loc[2, "rho"])


def test_compute_acf_reports_effective_n_pairs_not_assumed_T_minus_k():
    values = np.arange(20, dtype=float)
    block_ids = np.array([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5)
    acf_tab = compute_acf(values, block_ids, max_lag=6)
    assert acf_tab.loc[0, "n_pairs"] == 16  # 19 pares totales - 3 fronteras entre 4 bloques


def test_compute_acf_matches_pairwise_correlation_formula():
    """rho_k debe coincidir EXACTAMENTE con Pearson pairwise-complete LITERAL: cada lado (t y t-k) centrado por SU PROPIA media sobre P_k -- nunca por la media global de toda la poblacion (2a revision, seccion 1)."""
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 1.0, size=3000)
    block_ids = np.zeros(len(x), dtype=int)  # un unico bloque -- todos los pares son "validos"
    acf_tab = compute_acf(x, block_ids, max_lag=5)

    for k in range(1, 6):
        x_t = x[k:]
        x_tk = x[:-k]
        e_t = x_t - x_t.mean()   # media SOLO del lado t (sobre P_k), no la media global de x
        e_tk = x_tk - x_tk.mean()  # media SOLO del lado t-k (sobre P_k)
        expected_rho = np.sum(e_t * e_tk) / np.sqrt(np.sum(e_t**2) * np.sum(e_tk**2))
        expected_beta = np.sum(e_t * e_tk) / np.sum(e_tk**2)
        assert acf_tab.loc[k - 1, "rho"] == pytest.approx(expected_rho, rel=1e-10)
        assert acf_tab.loc[k - 1, "beta"] == pytest.approx(expected_beta, rel=1e-10)


def test_compute_acf_pair_local_centering_differs_from_global_mean_centering_in_general():
    """Contraste explicito: centrar por la media GLOBAL de la poblacion (bug corregido) da un numero DISTINTO del centrado correcto por-par cuando el subconjunto de pares excluidos no es representativo de la poblacion completa."""
    rng = np.random.default_rng(6)
    # Bloques de longitud variable: el rezago 3 excluye desproporcionadamente
    # los bloques cortos (que aportan MENOS pares altos), sesgando el conjunto
    # de pares P_3 respecto de la poblacion completa.
    block_lengths = [3] * 200 + [50] * 20  # muchos bloques cortos (longitud 3) + pocos largos (longitud 50)
    block_ids = np.concatenate([np.full(length, i) for i, length in enumerate(block_lengths)])
    n = len(block_ids)
    x = rng.normal(0.0, 1.0, size=n) + np.repeat(np.arange(len(block_lengths)) * 0.01, block_lengths)  # deriva lenta entre bloques

    acf_tab = compute_acf(x, block_ids, max_lag=3)
    rho_correct = acf_tab.loc[2, "rho"]  # k=3 -- solo los bloques de longitud 50 aportan pares

    same_block = block_ids[3:] == block_ids[:-3]
    global_mean = x.mean()
    e_global = x - global_mean
    e_t_buggy = e_global[3:][same_block]
    e_tk_buggy = e_global[:-3][same_block]
    rho_buggy_global_mean = np.sum(e_t_buggy * e_tk_buggy) / np.sqrt(np.sum(e_t_buggy**2) * np.sum(e_tk_buggy**2))

    assert rho_correct != pytest.approx(rho_buggy_global_mean, rel=1e-6)


def test_compute_acf_rho_always_within_unit_interval():
    rng = np.random.default_rng(4)
    x = rng.standard_cauchy(size=2000) * 0.001  # colas muy pesadas, caso adverso
    block_ids = np.repeat(np.arange(50), 40)
    acf_tab = compute_acf(x, block_ids, max_lag=10)
    valid_rho = acf_tab.loc[acf_tab["estimable"], "rho"]
    assert (valid_rho.abs() <= 1.0 + 1e-9).all()


def test_compute_acf_rho_and_beta_are_distinct_quantities_in_general():
    """rho (correlacion) y beta (pendiente) NO deben presuponerse iguales -- se construye un caso donde difieren claramente.

    ``n_blocks`` bloques de 2 elementos cada uno: ``e0`` (la mitad "t-1"
    de cada par, escala 1) y ``e1 = rho_true*e0 + ruido de escala GRANDE``
    (la mitad "t" de cada par) -- la asimetria de escala es una
    propiedad DE CADA PAR individual (no un efecto de frontera de un
    unico bloque largo, que solo afectaria 2 puntos de N), asi que se
    preserva al agregar sobre miles de pares.
    """
    rng = np.random.default_rng(5)
    n_blocks = 3000
    rho_true = 0.3
    e0 = rng.normal(0.0, 1.0, size=n_blocks)         # mitad "t-1": escala 1
    e1 = rho_true * e0 + rng.normal(0.0, 8.0, size=n_blocks)  # mitad "t": escala mucho mayor
    values = np.empty(2 * n_blocks)
    values[0::2] = e0
    values[1::2] = e1
    block_ids = np.repeat(np.arange(n_blocks), 2)  # cada par es su propio bloque de 2 elementos

    acf_tab = compute_acf(values, block_ids, max_lag=1)
    rho = acf_tab.loc[0, "rho"]
    beta = acf_tab.loc[0, "beta"]
    assert acf_tab.loc[0, "n_pairs"] == n_blocks
    # Con escalas muy distintas entre "t" y "t-1" dentro de los pares, rho y beta deben diferir de forma material.
    assert abs(beta / rho) > 3.0


def test_compute_acf_early_stop_fills_remaining_lags_as_not_estimable():
    """Una vez que n_pairs llega a 0 en un rezago, TODOS los rezagos mayores deben quedar NOT_ESTIMABLE sin necesidad de que existan datos para calcularlos."""
    values = np.arange(10, dtype=float)
    block_ids = np.zeros(10, dtype=int)  # bloque de longitud 10: n_pairs_k = 10-k -> ultimo rezago con >=2 pares es k=8
    acf_tab = compute_acf(values, block_ids, max_lag=50)
    assert len(acf_tab) == 50
    estimable = acf_tab.set_index("lag")["estimable"]
    assert bool(estimable.loc[8])   # n_pairs=2 -- estimable
    assert not bool(estimable.loc[9])   # n_pairs=1 -- degenerado, no estimable
    assert not bool(estimable.loc[10])  # n_pairs=0
    assert not bool(estimable.loc[50])
    assert acf_tab.loc[acf_tab["lag"] >= 10, "n_pairs"].eq(0).all()


# ---------------------------------------------------------------------------
# A. ACF multi-bloque NO debe atenuar mecanicamente un AR(1) conocido
# ---------------------------------------------------------------------------

def test_acf_multiblock_does_not_attenuate_known_ar1_dependence():
    x = _ar1_returns(20000, phi=0.4, sigma_e=1.0, seed=50)
    block_single = np.zeros(len(x), dtype=int)
    rho_single = compute_acf(x, block_single, max_lag=1).loc[0, "rho"]

    block_multi = np.repeat(np.arange(500), 40)[: len(x)]
    acf_multi = compute_acf(x, block_multi, max_lag=1)
    rho_multi = acf_multi.loc[0, "rho"]
    n_pairs_multi = acf_multi.loc[0, "n_pairs"]
    n_pairs_single = compute_acf(x, block_single, max_lag=1).loc[0, "n_pairs"]

    assert abs(rho_multi - rho_single) < 0.02  # NO se atenua por fragmentar en 500 bloques
    assert n_pairs_multi < n_pairs_single       # SI se pierden los pares de frontera...
    assert n_pairs_multi > 0.9 * n_pairs_single  # ...pero solo la fraccion pequeña esperada (1/40)


def test_old_T_normalized_convention_would_attenuate_more_than_corrected():
    """Contraste explicito con la convencion ANTERIOR (bug corregido): normalizar por T atenua mas al fragmentar en bloques cortos."""
    x = _ar1_returns(20000, phi=0.4, sigma_e=1.0, seed=51)
    block_multi = np.repeat(np.arange(1000), 20)[: len(x)]  # bloques cortos -> mas perdida de pares

    same_block = block_multi[1:] == block_multi[:-1]
    e = x - x.mean()
    sum_prod = np.sum((e[1:] * e[:-1])[same_block])
    T = len(x)
    gamma0_full = np.mean(e**2)
    rho_old_buggy_convention = (sum_prod / T) / gamma0_full

    rho_new = compute_acf(x, block_multi, max_lag=1).loc[0, "rho"]

    assert abs(rho_new) > abs(rho_old_buggy_convention) * 1.02


# ---------------------------------------------------------------------------
# Comportamiento conocido: iid, AR(1)+/-
# ---------------------------------------------------------------------------

def test_acf_is_close_to_zero_for_iid_noise():
    rng = np.random.default_rng(10)
    x = rng.normal(0.0, 1.0, size=20000)
    block_ids = np.zeros(len(x), dtype=int)
    acf_tab = compute_acf(x, block_ids, max_lag=5)
    assert abs(acf_tab.loc[0, "rho"]) < 0.05


def test_acf_detects_positive_ar1_dependence():
    x = _ar1_returns(20000, phi=0.4, sigma_e=1.0, seed=11)
    block_ids = np.zeros(len(x), dtype=int)
    assert compute_acf(x, block_ids, max_lag=1).loc[0, "rho"] > 0.3


def test_acf_detects_negative_ar1_dependence():
    x = _ar1_returns(20000, phi=-0.4, sigma_e=1.0, seed=12)
    block_ids = np.zeros(len(x), dtype=int)
    assert compute_acf(x, block_ids, max_lag=1).loc[0, "rho"] < -0.3


# ---------------------------------------------------------------------------
# Bartlett y Durbin-Levinson (sin cambios metodologicos respecto del cierre anterior)
# ---------------------------------------------------------------------------

def test_bartlett_se_matches_textbook_formula():
    rho = np.array([0.3, 0.1, -0.05, 0.02, 0.01])
    T = 10000
    se = bartlett_se(rho, T)
    assert se[0] == pytest.approx(np.sqrt(1.0 / T))
    assert se[1] == pytest.approx(np.sqrt((1.0 + 2.0 * rho[0] ** 2) / T))
    assert se[2] == pytest.approx(np.sqrt((1.0 + 2.0 * (rho[0] ** 2 + rho[1] ** 2)) / T))


def test_durbin_levinson_pacf_recovers_pure_ar1_case():
    phi = 0.6
    rho = np.array([phi ** k for k in range(1, 11)])
    pacf = durbin_levinson_pacf(rho, max_lag=10)
    assert pacf[0] == pytest.approx(phi, abs=1e-10)
    np.testing.assert_allclose(pacf[1:], np.zeros(9), atol=1e-8)


# ---------------------------------------------------------------------------
# E. Portmanteau: NOT_ESTIMABLE mas alla del maximo rezago observable
# ---------------------------------------------------------------------------

def test_portmanteau_q_matches_hand_computed_formula_within_estimable_range():
    rows = _continuous_chain(30, pd.Timestamp("2024-01-08 09:30:00"), datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    df = attach_calendar_for_test(variables, validity)
    block_ids = compute_block_ids(df["timestamp"], df["trading_date"], 60.0)
    values = df["r_1m"].to_numpy(dtype=float)
    acf_tab = compute_acf(values, block_ids, max_lag=10)
    q_tab = compute_portmanteau_q(acf_tab, m_values=(3,))
    rho = acf_tab["rho"].to_numpy()
    n_pairs = acf_tab["n_pairs"].to_numpy()
    expected_q3 = float(np.sum(n_pairs[:3] * rho[:3] ** 2))
    assert q_tab.loc[q_tab["m"] == 3, "Q"].iloc[0] == pytest.approx(expected_q3)


def test_portmanteau_q_returns_not_estimable_for_m_beyond_max_observable_lag():
    values = np.arange(15, dtype=float)
    block_ids = np.zeros(15, dtype=int)  # bloque unico de longitud 15 -> max rezago con pares = 14
    acf_tab = compute_acf(values, block_ids, max_lag=20)
    q_tab = compute_portmanteau_q(acf_tab, m_values=(5, 18))
    row5 = q_tab[q_tab["m"] == 5].iloc[0]
    row18 = q_tab[q_tab["m"] == 18].iloc[0]
    assert bool(row5["estimable"])
    assert not np.isnan(row5["Q"])
    assert not bool(row18["estimable"])
    assert np.isnan(row18["Q"])
    assert "NOT_ESTIMABLE" in row18["note"]


def test_portmanteau_q_adds_row_for_actual_max_estimable_lag():
    values = np.arange(15, dtype=float)
    block_ids = np.zeros(15, dtype=int)  # n_pairs_k=15-k -> ultimo rezago con >=2 pares es k=13
    acf_tab = compute_acf(values, block_ids, max_lag=20)
    q_tab = compute_portmanteau_q(acf_tab, m_values=(5,))
    assert 13 in set(q_tab["m"])  # max rezago realmente estimable, añadido aunque no estuviera en m_values


# ---------------------------------------------------------------------------
# Portmanteau: estado de calibracion G2 (2a revision, seccion 7;
# corregido en el cierre definitivo -- seccion 3: distincion EXPLICITA
# por serie, ninguna serie sin null propio puede marcarse G2_CALIBRATED
# por accidente) -- G2_CALIBRATED / DESCRIPTIVE_UNCALIBRATED /
# NOT_G2_CALIBRATED / NOT_ESTIMABLE
# ---------------------------------------------------------------------------

def test_portmanteau_calibration_status_distinguishes_g2_calibrated_from_descriptive():
    """Serie CON calibracion Null-1 propia (equivalente a r_1m en el pipeline real)."""
    values = np.arange(80, dtype=float)
    block_ids = np.zeros(80, dtype=int)  # n_pairs_k=80-k -> estimable hasta k=78 (n_pairs=2); k=79 -> n_pairs=1, no estimable
    acf_tab = compute_acf(values, block_ids, max_lag=80)
    q_tab = compute_portmanteau_q(acf_tab, m_values=(10, 70, 79))
    annotated = annotate_portmanteau_calibration(q_tab, has_null1_calibration=True, g2_calibrated_max_m=60)
    status = annotated.set_index("m")["calibration_status"]
    assert status[10] == "G2_CALIBRATED"          # m<=60, estimable, serie CON calibracion propia
    assert status[70] == "DESCRIPTIVE_UNCALIBRATED"  # estimable pero excede lo que G2 calibra
    assert status[79] == "NOT_ESTIMABLE"           # excede el maximo rezago realmente estimable (78)


def test_portmanteau_calibration_status_never_marks_a_series_without_its_own_null_as_calibrated():
    """Serie SIN calibracion Null-1 propia (equivalente a r_tilde en el pipeline real) -- NUNCA G2_CALIBRATED, ni siquiera para m<=60 (cierre definitivo, seccion 2/3)."""
    values = np.arange(80, dtype=float)
    block_ids = np.zeros(80, dtype=int)
    acf_tab = compute_acf(values, block_ids, max_lag=80)
    q_tab = compute_portmanteau_q(acf_tab, m_values=(10, 70, 79))
    annotated = annotate_portmanteau_calibration(q_tab, has_null1_calibration=False, g2_calibrated_max_m=60)
    status = annotated.set_index("m")["calibration_status"]
    assert status[10] == "NOT_G2_CALIBRATED"   # m<=60 pero la serie NO tiene null propio -- nunca G2_CALIBRATED
    assert status[70] == "NOT_G2_CALIBRATED"
    assert status[79] == "NOT_ESTIMABLE"       # NOT_ESTIMABLE siempre tiene prioridad, sea cual sea la serie
    assert "G2_CALIBRATED" not in set(status.values)


def test_annotate_portmanteau_calibration_requires_explicit_has_null1_calibration_argument():
    """Garantia estructural: no hay valor por defecto -- omitir el argumento debe fallar, no asumir silenciosamente calibrado/no calibrado."""
    values = np.arange(20, dtype=float)
    block_ids = np.zeros(20, dtype=int)
    acf_tab = compute_acf(values, block_ids, max_lag=20)
    q_tab = compute_portmanteau_q(acf_tab, m_values=(5,))
    with pytest.raises(TypeError):
        annotate_portmanteau_calibration(q_tab)


# ---------------------------------------------------------------------------
# Bootstrap: reproducibilidad + clave compuesta
# ---------------------------------------------------------------------------

def test_bootstrap_rho_is_reproducible_with_same_seed():
    rng = np.random.default_rng(0)
    values = rng.normal(size=500)
    trading_date = np.repeat(np.arange(50), 10).astype(str)
    block_ids = np.repeat(np.arange(50), 10)
    rho1, beta1 = bootstrap_rho(values, block_ids, trading_date, lags=(1, 2), n_boot=20, seed=7)
    rho2, beta2 = bootstrap_rho(values, block_ids, trading_date, lags=(1, 2), n_boot=20, seed=7)
    np.testing.assert_array_equal(rho1, rho2)
    np.testing.assert_array_equal(beta1, beta2)


def test_bootstrap_composite_key_prevents_false_neighbor_across_repeated_day_draws():
    block_id = np.zeros(10, dtype=int)
    values = np.array([100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -100.0])

    n_dates_sim = 2
    idx = np.concatenate([np.arange(10), np.arange(10)])
    slots = np.concatenate([np.zeros(10, dtype=np.int64), np.ones(10, dtype=np.int64)])
    values_concat = values[idx]
    block_id_concat = block_id[idx]

    e = values_concat - values_concat.mean()

    same_naive = block_id_concat[1:] == block_id_concat[:-1]
    e_t_naive = e[1:][same_naive]
    e_tk_naive = e[:-1][same_naive]
    rho_naive = np.sum(e_t_naive * e_tk_naive) / np.sqrt(np.sum(e_t_naive**2) * np.sum(e_tk_naive**2))

    composite = block_id_concat.astype(np.int64) * (n_dates_sim + 1) + slots
    same_correct = composite[1:] == composite[:-1]
    e_t_ok = e[1:][same_correct]
    e_tk_ok = e[:-1][same_correct]
    rho_correct = np.sum(e_t_ok * e_tk_ok) / np.sqrt(np.sum(e_t_ok**2) * np.sum(e_tk_ok**2))

    assert rho_naive != pytest.approx(rho_correct)


def test_bootstrap_rho_only_draws_from_observed_days():
    values = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    trading_date = np.array(["d1", "d1", "d2", "d2", "d3", "d3"])
    block_ids = np.array([0, 0, 1, 1, 2, 2])
    rho_boot, beta_boot = bootstrap_rho(values, block_ids, trading_date, lags=(1,), n_boot=100, seed=0)
    finite = rho_boot[~np.isnan(rho_boot)]
    assert np.all(np.abs(finite) <= 1.0 + 1e-9)


# ---------------------------------------------------------------------------
# B. Decil de volumen / agrupacion CORREGIDA: no exige que t-1 comparta el grupo
# ---------------------------------------------------------------------------

def test_acf_by_group_does_not_require_previous_row_to_share_the_group():
    n = 2000
    rows = _continuous_chain(n, pd.Timestamp("2024-01-08 09:30:00"), datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    df = attach_calendar_for_test(variables, validity)
    block_ids = compute_block_ids(df["timestamp"], df["trading_date"], 60.0)
    values = df["r_1m"].to_numpy(dtype=float)
    group_labels = np.arange(len(df)) % 2  # alterna 0,1,0,1,... -- decile_t NUNCA coincide con decile_{t-1}

    corrected = acf_by_group(values, block_ids, group_labels, lags=(1,), min_n_pairs=10)
    n_pairs_by_group = corrected.set_index("group")["n_pairs"]
    assert n_pairs_by_group[0] > n * 0.4
    assert n_pairs_by_group[1] > n * 0.4

    df2 = df.copy()
    df2["grp"] = group_labels
    strict = acf_by_group_strict_same_block(df2, "r_1m", "grp", lags=(1,), min_n_pairs=1)
    strict_n_pairs = strict.set_index("group")["n_pairs"] if len(strict) else pd.Series(dtype=int)
    assert strict_n_pairs.get(0, 0) == 0
    assert strict_n_pairs.get(1, 0) == 0


def test_acf_by_group_labels_pair_by_the_later_row_t():
    """La etiqueta de grupo de un par (t,t-1) debe ser la de la fila t (la mas reciente), no la de t-1."""
    values = np.array([0.0, 1.0, -1.0, 2.0, -2.0])
    block_ids = np.zeros(5, dtype=int)
    group_labels = np.array(["A", "A", "B", "B", "A"])  # etiqueta de CADA fila (posicion 0..4)
    table = acf_by_group(values, block_ids, group_labels, lags=(1,), min_n_pairs=1)
    # pares (t=1,t-1=0)->grupo "A"; (t=2,t-1=1)->grupo "B"; (t=3,t-1=2)->grupo "B"; (t=4,t-1=3)->grupo "A"
    n_pairs = table.set_index("group")["n_pairs"]
    assert n_pairs["A"] == 2
    assert n_pairs["B"] == 2


def test_acf_by_group_handles_insufficient_sample_per_group():
    values = np.arange(50, dtype=float)
    block_ids = np.zeros(50, dtype=int)
    group_labels = np.where(np.arange(50) < 45, 0, 1)  # grupo 1 tiene muy pocas filas
    table = acf_by_group(values, block_ids, group_labels, lags=(1,), min_n_pairs=20)
    row1 = table[table["group"] == 1].iloc[0]
    assert bool(row1["insufficient_sample"])
    assert np.isnan(row1["rho"])


# ---------------------------------------------------------------------------
# TEST ADVERSARIAL OBLIGATORIO (2a revision, seccion 3): centrar por la
# media GLOBAL no debe fabricar dependencia serial a partir de una mera
# diferencia de medias ENTRE grupos.
# ---------------------------------------------------------------------------

def test_acf_by_group_does_not_inject_spurious_dependence_from_between_group_mean_differences():
    """Dos grupos con medias MUY distintas; DENTRO de cada grupo, x (r_t) e y (r_t-1) son independientes por construccion.

    El estimador CORRECTO (centrado POR GRUPO, sobre los pares de ESE
    grupo) debe dar rho~0 en ambos grupos. La version centrada por la
    media GLOBAL de toda la poblacion (el bug corregido) fabrica
    dependencia artificial: si el grupo A tiene media +5 y el grupo B
    media -5, centrar por una media comun (~0) deja residuos que NO son
    ruido de media cero dentro del grupo, sino "+5+ruido" y "-5+ruido" --
    su producto tiene un termino sistematico grande (+5)*(-5)=-25 que
    domina sobre el ruido genuino, fabricando una correlacion cercana a
    -1 donde la dependencia real es 0.
    """
    rng = np.random.default_rng(42)
    n_per_group = 5000
    mean_a, mean_b = 5.0, -5.0
    values_a = mean_a + rng.normal(0.0, 1.0, size=n_per_group)  # ruido INDEPENDIENTE, sin relacion serial
    values_b = mean_b + rng.normal(0.0, 1.0, size=n_per_group)  # ruido INDEPENDIENTE, sin relacion serial

    # Intercalados: el par (t, t-1) alterna entre "A precedido de B" y "B precedido de A".
    values = np.empty(2 * n_per_group)
    group_labels = np.empty(2 * n_per_group, dtype=object)
    values[0::2] = values_a
    values[1::2] = values_b
    group_labels[0::2] = "A"
    group_labels[1::2] = "B"
    block_ids = np.zeros(len(values), dtype=int)  # continuidad total -- ningun par se excluye por topologia

    table = acf_by_group(values, block_ids, group_labels, lags=(1,), min_n_pairs=100)
    for g in ("A", "B"):
        row = table[table["group"] == g].iloc[0]
        assert abs(row["rho"]) < 0.05, f"grupo {g}: rho={row['rho']} -- deberia ser ~0 (independencia real dentro del grupo)"

    # Contraste EXPLICITO con la version centrada por la media GLOBAL (el bug corregido):
    global_mean = values.mean()
    e_global = values - global_mean
    same_block = block_ids[1:] == block_ids[:-1]
    idx_t = np.arange(1, len(values))[same_block]
    e_t = e_global[idx_t]
    e_tk = e_global[idx_t - 1]
    grp = group_labels[idx_t]
    df_buggy = pd.DataFrame({"group": grp, "prod": e_t * e_tk, "et2": e_t**2, "etk2": e_tk**2})
    agg_buggy = df_buggy.groupby("group").sum()
    rho_buggy_a = agg_buggy.loc["A", "prod"] / np.sqrt(agg_buggy.loc["A", "et2"] * agg_buggy.loc["A", "etk2"])
    assert abs(rho_buggy_a) > 0.3  # la version centrada por media global SI fabrica dependencia espuria fuerte


# ---------------------------------------------------------------------------
# C. G2 -- null de permutacion "por minuto": preserva escala por minuto, destruye dependencia
# ---------------------------------------------------------------------------

def _multi_day_two_scale_minutes(n_days: int, phi: float, seed: int) -> list[dict]:
    """Cada dia tiene 3 barras: un ancla (SIEMPRE invalida, primer minuto del dia), un minuto de escala PEQUEÑA (valido, ruido puro) y un minuto de escala GRANDE (valido, con dependencia AR(1) DENTRO del mismo dia respecto del minuto pequeño). Todas las dependencias cruzando dias quedan, por diseño, fuera de cualquier bloque de continuidad (el ancla las bloquea)."""
    rng = np.random.default_rng(seed)
    rows = []
    price = 15000.0
    day_i = 0
    added = 0
    while added < n_days:
        date = datetime.date(2024, 1, 8) + datetime.timedelta(days=day_i)
        day_i += 1
        if date.weekday() >= 5:
            continue
        ts_anchor = pd.Timestamp(date) + pd.Timedelta(hours=10, minutes=0)
        ts_small = pd.Timestamp(date) + pd.Timedelta(hours=10, minutes=1)
        ts_large = pd.Timestamp(date) + pd.Timedelta(hours=10, minutes=2)

        rows.append({"timestamp": ts_anchor, "trading_date": date, "close": price, "r_1m_valid": False, "volume": 10})

        scale_small, scale_large = 0.001, 0.05
        r_small = rng.normal(0.0, scale_small)
        price = price * np.exp(r_small)
        rows.append({"timestamp": ts_small, "trading_date": date, "close": price, "r_1m_valid": True, "volume": 10, "r_1m": float(r_small)})

        # dependencia DENTRO del mismo dia (60s, mismo trading_date) -- el
        # termino "K=scale_large/scale_small" reescala la señal para que
        # no quede aplastada por la diferencia de escala entre minutos
        # (si no, con phi*r_small directamente sobre ruido de escala 50x
        # mayor, la correlacion resultante seria casi cero por pura
        # dilucion de escala, no por ausencia real de dependencia).
        K = scale_large / scale_small
        r_large = phi * r_small * K + rng.normal(0.0, scale_large * 0.75)
        price = price * np.exp(r_large)
        rows.append({"timestamp": ts_large, "trading_date": date, "close": price, "r_1m_valid": True, "volume": 10, "r_1m": float(r_large)})

        added += 1
    return rows


def test_g2_permutation_null_by_minute_preserves_scale_profile_and_destroys_dependence():
    rows = _multi_day_two_scale_minutes(n_days=400, phi=0.5, seed=60)
    variables, validity = _variables_and_validity(rows)
    df = attach_calendar_for_test(variables, validity)
    block_ids = compute_block_ids(df["timestamp"], df["trading_date"], 60.0)
    values = df["r_1m"].to_numpy(dtype=float)
    minute_of_day = df["minute_of_day"].to_numpy()

    real_acf = compute_acf(values, block_ids, max_lag=1)
    null = g2_permutation_null_by_minute(values, minute_of_day, block_ids, lags=(1,), n_perm=30, seed=1)

    # La dependencia serial real es fuerte; el null debe colapsarla.
    assert abs(real_acf.loc[0, "rho"]) > 0.2
    assert np.nanmean(np.abs(null[:, 0])) < 0.15

    # El perfil de escala POR MINUTO debe preservarse: std por minuto en
    # UNA permutacion concreta debe seguir siendo mucho mayor en el
    # minuto "B" (escala 0.05) que en el "A" (escala 0.001).
    rng = np.random.default_rng(1)
    order = np.argsort(minute_of_day, kind="stable")
    sorted_minute = minute_of_day[order]
    boundaries = np.searchsorted(sorted_minute, np.arange(1441))
    permuted_once = values.copy()
    for start_val in np.unique(minute_of_day):
        g = order[boundaries[start_val]:boundaries[start_val + 1]] if start_val < 1440 else np.array([], dtype=int)
        if len(g) > 1:
            permuted_once[g] = rng.permutation(values[g])
    # ts_small (escala 0.001) y ts_large (escala 0.05, un minuto despues)
    # pueden mapear a MAS de un minute_of_day NY cada uno si el rango de
    # 400 dias cruza una transicion de DST -- se compara max/min de la
    # desviacion estandar POR MINUTO (robusto a cuantos valores distintos
    # de minute_of_day resulten), no dos grupos fijos por posicion.
    stds_by_minute = pd.Series(permuted_once).groupby(minute_of_day).std()
    assert stds_by_minute.max() > stds_by_minute.min() * 5  # la escala por minuto se preserva tras la permutacion


def test_g2_permutation_null_by_minute_is_reproducible():
    rng = np.random.default_rng(0)
    values = rng.normal(size=2880)
    minute_of_day = np.tile(np.arange(2), 1440)[: len(values)]
    block_ids = np.zeros(len(values), dtype=int)
    null1 = g2_permutation_null_by_minute(values, minute_of_day, block_ids, lags=(1,), n_perm=5, seed=3)
    null2 = g2_permutation_null_by_minute(values, minute_of_day, block_ids, lags=(1,), n_perm=5, seed=3)
    np.testing.assert_array_equal(null1, null2)


# ---------------------------------------------------------------------------
# D. G2 -- null sintetico empirico reescalado por s(m)
# ---------------------------------------------------------------------------

def test_g2_synthetic_empirical_null_rescales_pool_by_s_m_profile():
    """Dividir el synthetic por s(m) debe recuperar aprox. la escala del pool empirico -- la funcion SI reescala por s(m), no usa sigma global."""
    rng = np.random.default_rng(0)
    pool = rng.standard_t(df=3, size=5000) * 0.001
    n = 200000
    s_m = np.where(np.arange(n) % 2 == 0, 1.0, 3.0)
    draws = rng.choice(pool, size=n, replace=True)
    synthetic = draws * s_m
    recovered = synthetic / s_m
    assert recovered.std() == pytest.approx(pool.std(), rel=0.05)


def test_g2_synthetic_empirical_null_is_reproducible_and_has_no_deliberate_acf():
    rng = np.random.default_rng(0)
    pool = rng.standard_t(df=3, size=5000) * 0.001
    n = 3000
    s_m = np.tile([1.0, 2.0], n // 2)
    block_ids = np.zeros(n, dtype=int)
    null1 = g2_synthetic_empirical_null(pool, s_m, block_ids, lags=(1, 2), n_reps=20, seed=9)
    null2 = g2_synthetic_empirical_null(pool, s_m, block_ids, lags=(1, 2), n_reps=20, seed=9)
    np.testing.assert_array_equal(null1, null2)
    assert np.nanmean(np.abs(null1)) < 0.15  # sin dependencia serial deliberada


# ---------------------------------------------------------------------------
# G2 -- diagnostico REAL de momentos del null sintetico (2a revision,
# seccion 6): la afirmacion "preserva volatilidad/curtosis" debe
# verificarse, no solo declararse.
# ---------------------------------------------------------------------------

def test_g2_synthetic_null_moment_check_passes_for_correct_construction_and_fails_for_naive_gaussian():
    rng = np.random.default_rng(77)
    pool = rng.standard_t(df=3, size=20000) * 0.0005  # cola pesada, consistente con TDA-07
    n = 60000
    minute_of_day = np.tile(np.arange(100), n // 100)
    s_m_values = 1.0 + 0.5 * (minute_of_day % 100) / 100.0  # perfil de escala variable por minuto

    real = draw_synthetic_empirical_sample(pool, s_m_values, seed=123)  # "real" para efectos del test de mecanica
    good_synthetic = draw_synthetic_empirical_sample(pool, s_m_values, seed=999)
    result_good = g2_synthetic_null_moment_check(real, good_synthetic, minute_of_day, s_m_values)
    assert result_good["var_within_tolerance"]
    assert result_good["kurt_within_tolerance"]
    assert result_good["scale_profile_within_tolerance"]

    # Construccion INGENUA (version original de TDA-08, corregida hace dos revisiones):
    # gaussiano i.i.d. con sigma GLOBAL fijo, sin perfil de escala ni cola pesada.
    global_sigma = float(np.std(real))
    naive_gaussian = rng.normal(0.0, global_sigma, size=n)
    result_bad = g2_synthetic_null_moment_check(real, naive_gaussian, minute_of_day, s_m_values)
    assert not result_bad["kurt_within_tolerance"]           # sin cola pesada -> curtosis muy distinta
    assert not result_bad["scale_profile_within_tolerance"]  # sin perfil de escala -> sin correlacion con s_m


# ---------------------------------------------------------------------------
# G2 -- calibracion SECUNDARIA combinada (400 replicas totales: 200+200,
# nunca 400x400) -- nunca la inferencia principal (3a revision)
# ---------------------------------------------------------------------------

def test_g2_combined_calibration_summary_combines_both_nulls_correctly():
    real_rho = {1: 0.5}
    perm_null = np.zeros((200, 1))
    synth_null = np.zeros((200, 1))
    summary = g2_combined_calibration_summary(real_rho, perm_null, synth_null, lags=(1,))
    assert summary.loc[0, "n_null_replicas_combinadas"] == 400  # 200+200, no 400*400
    assert bool(summary.loc[0, "exceeds_calibration_threshold"])


# ---------------------------------------------------------------------------
# 3a revision -- CIERRE FINAL: G2 PRINCIPAL usa EXCLUSIVAMENTE Null 1;
# Null 2 (que no supero el diagnostico de momentos) nunca entra
# silenciosamente en la calibracion principal.
# ---------------------------------------------------------------------------

def test_g2_null1_calibration_uses_only_null1_and_ignores_null2_even_if_null2_would_change_the_verdict():
    """Construccion deterministica: bajo SOLO Null 1 el real NO supera el umbral (95%); si Null 2 se incluyera (combinado), SI lo superaria (97,5%). La inferencia PRINCIPAL debe seguir dando "no supera"."""
    lags = (1,)
    real_rho = {1: 0.01}
    # 190 de 200 replicas de Null 1 por debajo de 0.01, 10 por encima -> percentil = 95.0% (no supera 97.5)
    perm_null = np.concatenate([np.full(190, 0.005), np.full(10, 0.05)]).reshape(200, 1)
    # Si Null 2 (200 replicas, todas en 0, todas < 0.01) se sumara, el percentil combinado subiria a (190+200)/400=97.5% (SI supera)
    synth_null_that_would_flip_verdict = np.zeros((200, 1))

    null1_only = g2_null1_calibration_summary(real_rho, perm_null, lags)
    assert null1_only.loc[0, "percentile_of_real_in_null1"] == pytest.approx(95.0)
    assert not bool(null1_only.loc[0, "exceeds_calibration_threshold"])

    combined = g2_combined_calibration_summary(real_rho, perm_null, synth_null_that_would_flip_verdict, lags)
    assert combined.loc[0, "percentile_of_real_in_null"] == pytest.approx(97.5)
    assert bool(combined.loc[0, "exceeds_calibration_threshold"])  # Null 2 SI cambiaria el veredicto si se incluyera


def test_g2_null1_calibration_summary_signature_structurally_excludes_null2():
    """Garantia estructural: la funcion PRINCIPAL ni siquiera acepta un argumento de Null 2 -- no puede colarse por error futuro."""
    import inspect
    params = inspect.signature(g2_null1_calibration_summary).parameters
    assert "synth_null" not in params
    assert set(params) == {"real_rho", "perm_null", "lags"}

    port_params = inspect.signature(g2_null1_portmanteau_summary).parameters
    assert "synth_null" not in port_params


def test_g2_null1_portmanteau_summary_matches_hand_computation_and_reports_real_percentile():
    col = np.concatenate([np.full(190, 0.01), np.full(10, 10.0)])
    perm_null = np.column_stack([col, col])  # 2 rezagos identicos, (200, 2)
    real_n_pairs = np.array([1000.0, 1000.0])
    real_q = {2: 1000.0 * (0.01 ** 2 + 0.01 ** 2)}  # Q(2) real IGUAL al valor de las 190 replicas "bajas"

    q_tab = g2_null1_portmanteau_summary(perm_null, real_n_pairs, m_values=(2,), real_q=real_q)
    row = q_tab.iloc[0]
    assert row["n_null1_replicas"] == 200
    assert row["Q_real"] == pytest.approx(real_q[2])
    # El real coincide con las 190 replicas "bajas" -- por convencion de conteo estricto (<), su percentil es 0 (no hay ninguna replica ESTRICTAMENTE menor).
    assert row["percentile_of_real_in_null1"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Traduccion a ticks -- rho_1 y beta_1 SEPARADOS
# ---------------------------------------------------------------------------

def test_translate_dependence_to_ticks_uses_beta_not_rho_for_implied_move():
    rho_1 = 0.02
    beta_1 = 0.05  # deliberadamente DISTINTO de rho_1
    sigma_r = 0.0004
    close = np.array([15000.0, 15001.0, 15002.0])
    out = translate_dependence_to_ticks(rho_1, beta_1, sigma_r, close, TICK)

    implied_return = beta_1 * sigma_r  # debe usar beta_1, NO rho_1
    close_repr = float(np.median(close))
    tick_return_repr = float(np.log((close_repr + TICK) / close_repr))

    assert out["rho_1"] == rho_1
    assert out["beta_1"] == beta_1
    assert out["movimiento_lineal_implicito_return"] == pytest.approx(implied_return)
    assert out["movimiento_lineal_implicito_return"] != pytest.approx(rho_1 * sigma_r)  # NO coincide con la version basada en rho
    assert out["conversion_ticks"] == pytest.approx(implied_return / tick_return_repr)


# ---------------------------------------------------------------------------
# 1/5/10 minutos -- no solapado (sin cambios de diseño, solo de motor de ACF interno)
# ---------------------------------------------------------------------------

def test_multi_frequency_rho1_uses_non_overlapping_windows_and_respects_continuity():
    rng = np.random.default_rng(30)
    returns = rng.normal(0.0, 0.0008, size=6000)
    rows = _single_long_chain_with_returns(returns, pd.Timestamp("2024-01-08 09:30:00"), datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    result = compute_multi_frequency_rho1(variables, validity, n_boot=10, seed=0, horizons=(1, 5, 10))

    assert set(result["h_minutes"]) == {1, 5, 10}
    n_by_h = result.set_index("h_minutes")["n"]
    assert n_by_h[5] < n_by_h[1]
    assert n_by_h[10] < n_by_h[5]
    assert (result["n_pairs_lag1"] <= result["n"]).all()
    assert "beta_1" in result.columns


# ---------------------------------------------------------------------------
# Ventanas de TDA-06 -- magnitud POR BARRA
# ---------------------------------------------------------------------------

def test_analyze_window_matches_predeclared_tda06_windows():
    assert WINDOW_OPEN == (571, 575)
    assert WINDOW_CLOSE == (952, 962)


def test_analyze_window_computes_mean_per_bar_and_rho1_on_the_correct_minute_range():
    rows = []
    ts0 = pd.Timestamp("2024-01-08 14:29:00")  # UTC -- 09:29 NY en invierno (EST, UTC-5)
    price = 100.0
    for i in range(10):
        rows.append({"timestamp": ts0 + pd.Timedelta(minutes=i), "trading_date": datetime.date(2024, 1, 8), "close": price, "r_1m_valid": i > 0, "volume": 10})
        price += 0.25
    variables, validity = _variables_and_validity(rows)
    df = attach_calendar_for_test(variables, validity)
    out = analyze_window(df, (571, 573), TICK, n_boot=5, seed=0)
    assert out["n"] == 3
    assert out["n_minutes"] == 3
    assert "mean_per_bar" in out and "mean_per_bar_ticks" in out


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
            prev_price = price
            price = price * (1.0 + rng.normal(0.0, 0.0005))
            price = round(price / TICK) * TICK
            volume = int(rng.integers(1, 500))
            if first_row:
                rows.append({"timestamp": ts, "close": price, "trading_date": date, "r_1m_valid": False, "volume": volume})
                first_row = False
            else:
                rows.append({"timestamp": ts, "close": price, "trading_date": date, "r_1m_valid": True, "volume": volume, "r_1m": float(np.log(price / prev_price))})
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
        "tda06": {
            "output_dir_reports": "reports/mnq", "output_dir_interim": "data/interim/mnq",
            "n_boot": 5, "extreme_quantile": 0.99, "smoothing_window_minutes": 15,
            "max_breakpoints": 6, "min_spacing_minutes": 60, "year_stability_tolerance_minutes": 15,
            "year_stability_min_years": 3,
            "report_name": "TDA06_perfil_intradia_calendario.md",
            "minute_global_csv": "TDA06_perfil_minuto_global.csv",
            "minute_by_year_csv": "TDA06_perfil_minuto_por_anio.csv",
            "weekday_csv": "TDA06_perfil_dia_semana.csv",
            "segmentation_csv": "TDA06_segmentacion_propuesta.csv",
            "calibration_csv": "TDA06_calibracion_null.csv",
            "profile_png": "TDA06_perfil_intradia.png",
            "s_m_parquet_name": "tda06_s_m.parquet",
            "r_tilde_parquet_name": "tda06_r_tilde.parquet",
        },
        "tda08": {
            "output_dir_reports": "reports/mnq",
            "n_boot": 5, "n_perm": 5,
            "report_name": "TDA08_dependencia_lineal_media.md",
            "acf_global_csv": "TDA08_acf_global.csv",
            "pacf_csv": "TDA08_pacf.csv",
            "bootstrap_ci_csv": "TDA08_bootstrap_ci.csv",
            "portmanteau_csv": "TDA08_portmanteau.csv",
            "g2_calibration_null1_csv": "TDA08_g2_calibracion_null1_principal.csv",
            "g2_portmanteau_null1_csv": "TDA08_g2_portmanteau_null1_principal.csv",
            "g2_calibration_csv": "TDA08_g2_calibracion_combinada_secundaria.csv",
            "g2_portmanteau_null_csv": "TDA08_g2_portmanteau_combinado_secundario.csv",
            "g2_moment_check_csv": "TDA08_g2_diagnostico_momentos.csv",
            "ticks_translation_csv": "TDA08_traduccion_ticks.csv",
            "multi_frequency_csv": "TDA08_rho1_multi_frecuencia.csv",
            "acf_by_group_csv": "TDA08_acf_por_grupo.csv",
            "volume_decile_sensitivity_csv": "TDA08_decil_volumen_sensibilidad.csv",
            "windows_csv": "TDA08_ventanas_tda06.csv",
            "windows_by_year_csv": "TDA08_ventanas_por_anio.csv",
            "windows_adjacent_csv": "TDA08_ventanas_minutos_adyacentes.csv",
            "acf_plot_png": "TDA08_acf_bartlett.png",
        },
    }
    config_path = config_dir / "mnq_snapshot.yaml"
    config_path.write_text(yaml.safe_dump(config_yaml), encoding="utf-8")
    return load_config(config_path)


def _write_tda04_artifacts(config, variables: pd.DataFrame, validity: pd.DataFrame) -> None:
    config.interim_dir.mkdir(parents=True, exist_ok=True)
    variables.to_parquet(config.tda04_variables_parquet_path, index=False)
    validity.to_parquet(config.tda04_validity_mask_parquet_path, index=False)


def _write_tda06_artifacts(config, calendar_df: pd.DataFrame, cutoffs: list[int]) -> None:
    config.interim_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    r_tilde = pd.DataFrame({
        "timestamp": calendar_df["timestamp"], "source_file": calendar_df["source_file"],
        "contract": calendar_df["contract"], "trading_date": calendar_df["trading_date"],
        "minute_of_day": calendar_df["minute_of_day"], "weekday": calendar_df["weekday"],
        "r_1m": calendar_df["r_1m"], "r_1m_valid": calendar_df["r_1m_valid"],
        "s_m": 1.0, "r_tilde": calendar_df["r_1m"], "label": "RETROSPECTIVO",
    })
    r_tilde.to_parquet(config.tda06_r_tilde_parquet_path, index=False)
    seg_rows = [{"minute_of_day": c, "n_years_supporting": 5, "years_supporting": "2020,2021,2022,2023,2024", "stable": True} for c in cutoffs]
    pd.DataFrame(seg_rows).to_csv(config.tda06_segmentation_csv_path, index=False)


def _prepare_full_fixture(config, n_days=10, start=datetime.date(2024, 1, 8), cutoffs=(600, 900)):
    rows = _synthetic_multi_day_rows(n_days=n_days, start=start)
    variables, validity = _variables_and_validity(rows)
    _write_tda04_artifacts(config, variables, validity)
    from ohlcv_dataroad.ingest.tda06_intraday_calendar_profile import attach_calendar_fields
    calendar_df = attach_calendar_fields(variables, validity)
    _write_tda06_artifacts(config, calendar_df, list(cutoffs))
    return variables, validity


def test_run_tda08_raises_when_research_and_holdout_overlap(tmp_path):
    config = _make_config(
        tmp_path, research_files=["00_mnq_03_24.Last.txt"],
        holdout_files=["00_mnq_03_24.Last.txt", "no_existe.Last.txt"],
    )
    with pytest.raises(HoldoutIsolationError, match="disjuntos"):
        run_tda08_analysis(config)


def test_run_tda08_never_opens_any_raw_or_holdout_file(tmp_path):
    config = _make_config(
        tmp_path, research_files=["00_mnq_03_24.Last.txt"],
        holdout_files=["archivo_que_no_existe.Last.txt"],
    )
    _prepare_full_fixture(config)
    assert not (config.raw_dir / "00_mnq_03_24.Last.txt").exists()
    result = run_tda08_analysis(config)
    assert len(result.r1m_population) > 0


def test_run_tda08_raises_on_misaligned_timestamps(tmp_path):
    config = _make_config(tmp_path, research_files=["00_mnq_03_24.Last.txt"])
    rows = _synthetic_multi_day_rows(n_days=5, start=datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    validity = validity.drop(index=2).reset_index(drop=True)
    _write_tda04_artifacts(config, variables, validity)
    with pytest.raises(TimestampAlignmentError):
        run_tda08_analysis(config)


def test_run_tda08_end_to_end_produces_all_result_fields(tmp_path):
    config = _make_config(tmp_path, research_files=["00_mnq_03_24.Last.txt"])
    _prepare_full_fixture(config, n_days=12)
    result = run_tda08_analysis(config)

    assert not result.acf_raw.empty
    assert not result.acf_adjusted.empty
    assert "beta" in result.acf_raw.columns and "estimable" in result.acf_raw.columns
    assert len(result.bartlett_se_raw) == len(result.acf_raw)
    assert len(result.pacf_raw) > 0 and len(result.pacf_adjusted) > 0
    assert not result.portmanteau_raw.empty and not result.portmanteau_adjusted.empty
    assert not result.bootstrap_table_raw.empty
    assert not result.g2_null1_calibration_raw.empty
    assert not result.g2_null1_portmanteau_raw.empty
    assert not result.g2_calibration_combined_secondary.empty
    assert not result.g2_portmanteau_combined_secondary.empty
    assert "n_null1_replicas" in result.g2_null1_calibration_raw.columns  # PRINCIPAL: solo Null 1
    assert "n_null_replicas_combinadas" in result.g2_calibration_combined_secondary.columns  # SECUNDARIO: combinado
    assert "beta_1" in result.ticks_translation
    assert set(result.multi_frequency["h_minutes"]) == {1, 5, 10}
    assert not result.acf_by_group_long.empty
    assert set(result.acf_by_group_long["group_type"]).issuperset({"year", "segment", "volume_decile", "year_month"})
    assert not result.volume_decile_sensitivity.empty
    assert result.window_open["window"] == WINDOW_OPEN
    assert result.window_close["window"] == WINDOW_CLOSE
    assert result.max_estimable_lag_raw > 0
    assert "kurt_within_tolerance" in result.g2_synthetic_null_moment_check
    assert "calibration_status" in result.portmanteau_raw.columns
    # Cierre definitivo: r_1m (con Null-1 propio) puede quedar G2_CALIBRATED;
    # r_tilde (sin Null-1 propio) NUNCA -- ver annotate_portmanteau_calibration.
    assert "G2_CALIBRATED" in set(result.portmanteau_raw["calibration_status"])
    assert "G2_CALIBRATED" not in set(result.portmanteau_adjusted["calibration_status"])
    estimable_adjusted = result.portmanteau_adjusted[result.portmanteau_adjusted["estimable"]]
    if not estimable_adjusted.empty:
        assert set(estimable_adjusted["calibration_status"]) <= {"NOT_G2_CALIBRATED"}


# ---------------------------------------------------------------------------
# F. Persistencia del runner (2a revision, seccion 8): los 17 artefactos
# actuales (estado del cierre definitivo -- ver ARTIFACT_PATH_ATTRS,
# fuente de verdad del conteo) deben existir de verdad tras
# persist_artifacts, y no debe
# quedar ningun nombre de artefacto obsoleto en el directorio.
# ---------------------------------------------------------------------------

def test_persist_artifacts_writes_exactly_the_declared_paths_and_nothing_obsolete(tmp_path):
    from ohlcv_dataroad.ingest.run_tda08 import ARTIFACT_PATH_ATTRS, persist_artifacts

    config = _make_config(tmp_path, research_files=["00_mnq_03_24.Last.txt"])
    _prepare_full_fixture(config, n_days=12)
    result = run_tda08_analysis(config)

    written = persist_artifacts(result, config)
    declared_paths = {getattr(config, attr) for attr in ARTIFACT_PATH_ATTRS}

    assert set(written) == declared_paths
    for p in declared_paths:
        assert p.exists(), f"artefacto declarado pero no escrito: {p}"

    tda08_files_on_disk = {p.name for p in config.reports_dir.glob("TDA08_*")}
    declared_names = {p.name for p in declared_paths}
    obsolete = tda08_files_on_disk - declared_names
    assert not obsolete, f"artefactos TDA08_* en disco que ya no estan declarados: {obsolete}"
