"""Tests de TDA-10 -- escala versus forma: origen de las colas (TH22, parte de TH26).

No se reproducen aqui los numeros exactos del conjunto de investigacion
real (esos viven en ``reports/mnq/TDA10_escala_vs_forma.md``). Se cubren:
unidad (estimadores causales, quemado, division protegida, verificacion de
look-ahead, estabilidad de forma, curtosis, clasificacion del veredicto) y
extremo a extremo (invariantes, aislamiento de hold-out, reproducibilidad,
persistencia de artefactos) con datos sinteticos pequeños.
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
from ohlcv_dataroad.ingest.tda06_intraday_calendar_profile import attach_calendar_fields
from ohlcv_dataroad.ingest.tda07_marginal_distribution import RTildeInvariantError, TimestampAlignmentError
from ohlcv_dataroad.ingest.tda10_scale_vs_shape import (
    ALL_ESTIMATOR_CONFIGS,
    BURN_IN_HALFLIVES,
    FRACTION_REMOVED_FORM_THRESHOLD,
    FRACTION_REMOVED_SCALE_THRESHOLD,
    LookaheadLeakageError,
    MIN_N_FOR_STABILITY_GROUP,
    MIN_VALID_SIGMA_HAT,
    PopulationMismatchError,
    PRIMARY_ESTIMATOR,
    PROFILE_STABILITY_FORM_THRESHOLD,
    PROFILE_STABILITY_SCALE_THRESHOLD,
    ROBUSTNESS_AGREEMENT_FRACTION,
    assign_volatility_decile,
    bootstrap_kurtosis_ci,
    build_kurtosis_row,
    build_kurtosis_table,
    build_qq_table,
    build_sigma_hat,
    burn_in_length,
    causal_ewma_sigma,
    causal_rolling_sigma,
    classify_config,
    compute_group_quantile_table,
    decide_verdict,
    excess_kurtosis,
    extreme_scale_from_table,
    ewma_lambda_from_halflife,
    profile_stability_ratio,
    run_causality_checks,
    run_tda10_analysis,
    standardize_return,
    verify_no_lookahead,
    verify_no_lookahead_generic,
    verify_populations_aligned,
)

TICK = 0.25


# ---------------------------------------------------------------------------
# A. Estimadores causales -- desviacion rodante y EWMA
# ---------------------------------------------------------------------------

def test_causal_rolling_sigma_matches_hand_computation():
    r = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = causal_rolling_sigma(r, window=3)
    # posiciones 0,1,2 son NaN (min_periods=3 + shift(1) -> primeros 3 NaN)
    assert np.isnan(out[:3]).all()
    # out[3] = std(r[0:3]) = std([1,2,3], ddof=1)
    assert out[3] == pytest.approx(np.std([1.0, 2.0, 3.0], ddof=1))
    # out[4] = std(r[1:4]) = std([2,3,4], ddof=1)
    assert out[4] == pytest.approx(np.std([2.0, 3.0, 4.0], ddof=1))
    # out[5] = std(r[2:5]) = std([3,4,5], ddof=1)
    assert out[5] == pytest.approx(np.std([3.0, 4.0, 5.0], ddof=1))


def test_causal_rolling_sigma_rejects_window_below_two():
    with pytest.raises(ValueError):
        causal_rolling_sigma(np.array([1.0, 2.0, 3.0]), window=1)


def test_causal_rolling_sigma_never_uses_current_or_future_value():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.001, size=500)
    base = causal_rolling_sigma(r, window=30)
    perturbed = r.copy()
    perturbed[250] = 5.0  # valor extremo
    pert_sigma = causal_rolling_sigma(perturbed, window=30)
    # sigma_hat en t<=250 NUNCA debe cambiar
    np.testing.assert_array_equal(base[:251], pert_sigma[:251])
    # sigma_hat en t=251..280 (dentro de la ventana de 30) SI debe cambiar
    assert not np.allclose(base[251:280], pert_sigma[251:280], equal_nan=True)


def test_ewma_lambda_from_halflife_matches_closed_form():
    lam = ewma_lambda_from_halflife(10.0)
    assert lam == pytest.approx(0.5 ** (1.0 / 10.0))
    assert lam == pytest.approx(np.exp(np.log(0.5) / 10.0))


def test_ewma_lambda_from_halflife_rejects_non_positive():
    with pytest.raises(ValueError):
        ewma_lambda_from_halflife(0.0)


def test_causal_ewma_sigma_first_value_is_nan_second_is_r0_squared():
    r = np.array([2.0, 0.0, 0.0, 0.0, 0.0])
    sigma_hat = causal_ewma_sigma(r, halflife=5.0)
    assert np.isnan(sigma_hat[0])
    # sigma2[1] = r[0]^2 = 4.0 -> sigma_hat[1] = 2.0
    assert sigma_hat[1] == pytest.approx(2.0)


def test_causal_ewma_sigma_matches_hand_computed_recursion():
    r = np.array([1.0, 2.0, 3.0, 0.5])
    halflife = 4.0
    lam = ewma_lambda_from_halflife(halflife)
    sigma_hat = causal_ewma_sigma(r, halflife)

    sigma2 = [np.nan, r[0] ** 2]
    sigma2.append(lam * sigma2[1] + (1 - lam) * r[1] ** 2)
    sigma2.append(lam * sigma2[2] + (1 - lam) * r[2] ** 2)

    assert sigma_hat[1] == pytest.approx(np.sqrt(sigma2[1]))
    assert sigma_hat[2] == pytest.approx(np.sqrt(sigma2[2]))
    assert sigma_hat[3] == pytest.approx(np.sqrt(sigma2[3]))


def test_causal_ewma_sigma_never_uses_current_or_future_value():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0, 0.001, size=500)
    base = causal_ewma_sigma(r, halflife=60.0)
    perturbed = r.copy()
    perturbed[250] = 5.0
    pert_sigma = causal_ewma_sigma(perturbed, halflife=60.0)
    np.testing.assert_array_equal(base[:251], pert_sigma[:251])
    assert not np.allclose(base[251:255], pert_sigma[251:255], equal_nan=True)


def test_causal_rolling_sigma_spans_trading_day_boundaries_by_design():
    """Convencion documentada (docstring de `causal_rolling_sigma`): la ventana avanza
    sobre FILAS validas, no sobre minutos de reloj -- un hueco de fin de semana no
    reinicia ni invalida la ventana."""
    r = np.concatenate([np.full(20, 0.001), np.full(20, 0.001)])  # simula dos jornadas separadas por un hueco no representado aqui
    out = causal_rolling_sigma(r, window=10)
    assert np.isfinite(out[15:]).all()  # ninguna reinicia por "cruzar" el punto medio


def test_burn_in_length_rolling_equals_window():
    assert burn_in_length("rolling_std", 30) == 30
    assert burn_in_length("rolling_std", 390) == 390


def test_burn_in_length_ewma_equals_ceil_3x_halflife():
    assert burn_in_length("ewma", 60.0) == int(np.ceil(BURN_IN_HALFLIVES * 60.0))
    assert burn_in_length("ewma", 20.0) == int(np.ceil(BURN_IN_HALFLIVES * 20.0))


def test_burn_in_length_rejects_unknown_family():
    with pytest.raises(ValueError):
        burn_in_length("garch", 1.0)


def test_build_sigma_hat_dispatches_correctly():
    r = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(build_sigma_hat(r, "rolling_std", 2), causal_rolling_sigma(r, 2))
    np.testing.assert_array_equal(build_sigma_hat(r, "ewma", 3.0), causal_ewma_sigma(r, 3.0))


def test_build_sigma_hat_rejects_unknown_family():
    with pytest.raises(ValueError):
        build_sigma_hat(np.array([1.0]), "unknown", 1.0)


# ---------------------------------------------------------------------------
# B. Estandarizacion: division protegida + quemado
# ---------------------------------------------------------------------------

def test_standardize_return_protects_against_zero_negative_and_nan_sigma():
    r = np.array([1.0, 1.0, 1.0, 1.0])
    sigma = np.array([2.0, 0.0, -1.0, np.nan])
    z = standardize_return(r, sigma, burn_in=0)
    assert z[0] == pytest.approx(0.5)
    assert np.isnan(z[1])  # division por cero -> NaN, nunca inf
    assert np.isnan(z[2])  # sigma negativa -> NaN (nunca valida)
    assert np.isnan(z[3])  # sigma NaN -> NaN
    assert not np.isinf(z).any()


def test_standardize_return_masks_burn_in():
    r = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    sigma = np.full(5, 2.0)
    z = standardize_return(r, sigma, burn_in=3)
    assert np.isnan(z[:3]).all()
    assert z[3] == pytest.approx(0.5)
    assert z[4] == pytest.approx(0.5)


def test_standardize_return_nan_in_r_propagates():
    r = np.array([1.0, np.nan, 1.0])
    sigma = np.full(3, 2.0)
    z = standardize_return(r, sigma, burn_in=0)
    assert np.isnan(z[1])


def test_standardize_return_rejects_sigma_numerically_indistinguishable_from_zero():
    """Un `sigma_hat` positivo pero muy por debajo de `MIN_VALID_SIGMA_HAT` (residual de punto flotante en una
    ventana de precios casi constantes) NO debe producir un z_t de millones de desviaciones estandar."""
    r = np.array([-0.0019, 1.0])
    sigma = np.array([6.4e-11, 2.0])  # el mismo orden de magnitud del caso real detectado en rolling_std_30
    z = standardize_return(r, sigma, burn_in=0)
    assert np.isnan(z[0])  # protegido por el piso, no un z de ~30 millones
    assert z[1] == pytest.approx(0.5)


def test_standardize_return_accepts_sigma_just_above_the_floor():
    r = np.array([1.0])
    sigma = np.array([MIN_VALID_SIGMA_HAT * 2])
    z = standardize_return(r, sigma, burn_in=0)
    assert np.isfinite(z[0])


# ---------------------------------------------------------------------------
# C. Verificacion explicita de ausencia de look-ahead (G1)
# ---------------------------------------------------------------------------

def test_verify_no_lookahead_generic_passes_for_a_causal_estimator():
    rng = np.random.default_rng(2)
    r = rng.normal(0.0, 1.0, size=200)
    result = verify_no_lookahead_generic(lambda rr: causal_rolling_sigma(rr, 10), r, [50, 100, 150])
    assert result["passed"] is True
    assert all(c["unaffected_up_to_and_including_index"] for c in result["checks"])


def test_verify_no_lookahead_generic_detects_a_deliberately_centered_noncausal_estimator():
    """Estimador CENTRADO (usa r[t-w:t+w]) -- deliberadamente incausal, la prueba debe detectarlo."""
    def centered_estimator(r: np.ndarray) -> np.ndarray:
        return pd.Series(r).rolling(window=11, center=True, min_periods=11).std(ddof=1).to_numpy()

    rng = np.random.default_rng(3)
    r = rng.normal(0.0, 1.0, size=200)
    result = verify_no_lookahead_generic(centered_estimator, r, [50, 100, 150])
    assert result["passed"] is False
    assert not all(c["unaffected_up_to_and_including_index"] for c in result["checks"])


def test_verify_no_lookahead_production_families_pass():
    rng = np.random.default_rng(4)
    r = rng.normal(0.0, 0.001, size=1000)
    for family, param in [("rolling_std", 30), ("ewma", 20.0)]:
        result = verify_no_lookahead(family, param, r, [200, 500, 800])
        assert result["passed"] is True, (family, param, result)


def test_run_causality_checks_raises_when_an_estimator_is_noncausal(monkeypatch):
    import ohlcv_dataroad.ingest.tda10_scale_vs_shape as mod

    def fake_dispatch(r, family, param):
        if family == "rolling_std":
            return pd.Series(r).rolling(window=int(param), center=True, min_periods=int(param)).std(ddof=1).to_numpy()
        return causal_ewma_sigma(r, param)

    monkeypatch.setattr(mod, "build_sigma_hat", fake_dispatch)
    rng = np.random.default_rng(5)
    r = rng.normal(0.0, 0.001, size=1000)
    with pytest.raises(LookaheadLeakageError):
        run_causality_checks(r, configs=(("rolling_std", 30.0, "raw"),))


def test_run_causality_checks_covers_all_declared_configs_deduplicated_by_family_param():
    rng = np.random.default_rng(6)
    r = rng.normal(0.0, 0.001, size=2000)
    tab = run_causality_checks(r, configs=ALL_ESTIMATOR_CONFIGS)
    # 6 combinaciones distintas de (family, param) -- "raw"/"clock_adjusted" comparten la logica
    assert len(tab) == 6
    assert tab["passed"].all()


# ---------------------------------------------------------------------------
# D. Estabilidad de forma: decil de volatilidad / segmento / año
# ---------------------------------------------------------------------------

def test_assign_volatility_decile_produces_ten_groups_with_enough_data():
    rng = np.random.default_rng(7)
    sigma = rng.uniform(0.1, 10.0, size=5000)
    deciles = assign_volatility_decile(sigma)
    finite = deciles[~np.isnan(deciles)]
    assert set(np.unique(finite)) <= set(range(10))
    assert len(np.unique(finite)) == 10


def test_assign_volatility_decile_nan_where_sigma_invalid():
    sigma = np.array([1.0, -1.0, 0.0, np.nan] * 100)
    deciles = assign_volatility_decile(sigma)
    assert np.isnan(deciles[1::4]).all()
    assert np.isnan(deciles[2::4]).all()
    assert np.isnan(deciles[3::4]).all()


def test_assign_volatility_decile_all_nan_with_too_little_data():
    sigma = np.array([1.0, 2.0, 3.0])
    deciles = assign_volatility_decile(sigma)
    assert np.isnan(deciles).all()


def test_compute_group_quantile_table_marks_small_groups_as_nan_quantiles():
    rng = np.random.default_rng(8)
    z_big = rng.normal(0.0, 1.0, size=MIN_N_FOR_STABILITY_GROUP + 50)
    z_small = rng.normal(0.0, 1.0, size=5)
    z = np.concatenate([z_big, z_small])
    groups = np.array(["A"] * len(z_big) + ["B"] * len(z_small))
    tab = compute_group_quantile_table(z, groups)
    row_a = tab.loc[tab["group"] == "A"].iloc[0]
    row_b = tab.loc[tab["group"] == "B"].iloc[0]
    assert row_a["n"] == len(z_big)
    assert np.isfinite(row_a["q0.99"])
    assert row_b["n"] == 5
    assert np.isnan(row_b["q0.99"])  # muestra insuficiente -> NaN explicito, no un cuantil fabricado


def test_compute_group_quantile_table_excludes_nan_z_and_nan_group():
    z = np.array([1.0, np.nan, 2.0, 3.0])
    groups = np.array(["A", "A", "A", np.nan], dtype=object)
    tab = compute_group_quantile_table(z, groups, quantile_levels=(0.5,))
    assert len(tab) == 1
    assert tab.iloc[0]["group"] == "A"
    assert tab.iloc[0]["n"] == 2  # solo z=1.0 y z=2.0 (z=3.0 tiene group=NaN)


def test_extreme_scale_from_table_hand_computed():
    tab = pd.DataFrame({"q0.01": [-2.0, -1.0], "q0.99": [2.0, 3.0]})
    out = extreme_scale_from_table(tab)
    np.testing.assert_allclose(out.to_numpy(), [2.0, 2.0])


def test_profile_stability_ratio_zero_when_all_groups_identical():
    tab = pd.DataFrame({
        "n": [300, 300, 300], "q0.01": [-2.0, -2.0, -2.0], "q0.99": [2.0, 2.0, 2.0],
    })
    assert profile_stability_ratio(tab) == pytest.approx(0.0)


def test_profile_stability_ratio_positive_when_groups_diverge():
    tab = pd.DataFrame({
        "n": [300, 300, 300], "q0.01": [-1.0, -2.0, -4.0], "q0.99": [1.0, 2.0, 4.0],
    })
    ratio = profile_stability_ratio(tab)
    assert ratio > PROFILE_STABILITY_FORM_THRESHOLD


def test_profile_stability_ratio_nan_with_insufficient_groups():
    tab = pd.DataFrame({"n": [5, 300], "q0.01": [-1.0, -1.0], "q0.99": [1.0, 1.0]})
    assert np.isnan(profile_stability_ratio(tab))


# ---------------------------------------------------------------------------
# E. Curtosis: fila / tabla / bootstrap
# ---------------------------------------------------------------------------

def test_excess_kurtosis_hand_computed_uniform_like_sample():
    # Muestra simetrica de 5 puntos con curtosis conocida analiticamente
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    e = x - x.mean()
    m2 = np.mean(e ** 2)
    m4 = np.mean(e ** 4)
    expected = m4 / m2 ** 2 - 3.0
    assert excess_kurtosis(x) == pytest.approx(expected)


def test_excess_kurtosis_nan_for_degenerate_sample():
    assert np.isnan(excess_kurtosis(np.array([1.0, 1.0, 1.0, 1.0])))  # varianza cero
    assert np.isnan(excess_kurtosis(np.array([1.0, 2.0])))  # muy pocos puntos


def test_build_kurtosis_row_fraction_removed_hand_computed():
    rng = np.random.default_rng(9)
    r = rng.standard_t(df=3, size=5000) * 0.001  # colas pesadas
    z = rng.normal(0.0, 1.0, size=5000)  # normal -- curtosis casi 0
    row = build_kurtosis_row("ewma", 60.0, "raw", "GLOBAL", "GLOBAL", r, z)
    assert row["kurt_r"] > row["kurt_z"]
    assert row["fraction_removed"] == pytest.approx(1.0 - row["kurt_z"] / row["kurt_r"])
    assert row["fraction_removed"] > 0.5


def test_build_kurtosis_row_fraction_removed_nan_when_kurt_r_not_positive():
    r = np.array([1.0] * 30)  # varianza cero -> kurt_r NaN
    z = np.random.default_rng(10).normal(size=30)
    row = build_kurtosis_row("ewma", 60.0, "raw", "GLOBAL", "GLOBAL", r, z)
    assert np.isnan(row["fraction_removed"])


def test_build_kurtosis_table_restricts_r_to_same_population_as_z():
    rng = np.random.default_rng(11)
    n = 500
    r = rng.normal(0.0, 0.001, size=n)
    z = rng.normal(0.0, 1.0, size=n)
    z[:100] = np.nan  # simula quemado
    year_ny = np.array([2020] * 250 + [2021] * 250)
    tab = build_kurtosis_table("ewma", 60.0, "raw", r, z, year_ny)
    global_row = tab.loc[tab["scope"] == "GLOBAL"].iloc[0]
    assert global_row["n"] == n - 100  # las 100 filas con z=NaN quedan fuera de AMBAS series
    assert set(tab.loc[tab["scope"] == "YEAR", "scope_value"]) == {2020, 2021}


def test_bootstrap_kurtosis_ci_is_reproducible_with_fixed_seed():
    rng = np.random.default_rng(12)
    n = 2000
    r = rng.standard_t(df=4, size=n) * 0.001
    z = rng.normal(size=n)
    dates = np.repeat(np.arange(40), n // 40)
    out1 = bootstrap_kurtosis_ci(r, z, dates, n_boot=20, seed=99)
    out2 = bootstrap_kurtosis_ci(r, z, dates, n_boot=20, seed=99)
    assert out1 == out2


def test_bootstrap_kurtosis_ci_interval_contains_point_estimate():
    rng = np.random.default_rng(13)
    n = 3000
    r = rng.standard_t(df=4, size=n) * 0.001
    z = rng.normal(size=n)
    dates = np.repeat(np.arange(60), n // 60)
    out = bootstrap_kurtosis_ci(r, z, dates, n_boot=100, seed=1)
    assert out["kurt_r_ci_lo"] <= out["kurt_r_point"] <= out["kurt_r_ci_hi"]
    assert out["kurt_z_ci_lo"] <= out["kurt_z_point"] <= out["kurt_z_ci_hi"]


# ---------------------------------------------------------------------------
# F. Clasificacion del veredicto
# ---------------------------------------------------------------------------

def test_classify_config_scale_dominates():
    label = classify_config(FRACTION_REMOVED_SCALE_THRESHOLD + 0.05, PROFILE_STABILITY_SCALE_THRESHOLD - 0.05)
    assert label == "ESCALA_DOMINA"


def test_classify_config_form_substantial_by_low_fraction_removed():
    label = classify_config(FRACTION_REMOVED_FORM_THRESHOLD - 0.05, 0.01)
    assert label == "FORMA_SUSTANCIAL"


def test_classify_config_form_substantial_by_high_instability():
    label = classify_config(0.95, PROFILE_STABILITY_FORM_THRESHOLD + 0.05)
    assert label == "FORMA_SUSTANCIAL"


def test_classify_config_mixto_in_between():
    label = classify_config(0.65, 0.45)
    assert label == "MIXTO"


def test_classify_config_mixto_when_inputs_not_finite():
    assert classify_config(np.nan, 0.1) == "MIXTO"
    assert classify_config(0.9, np.nan) == "MIXTO"


def test_decide_verdict_unanimous_scale_dominates():
    out = decide_verdict(["ESCALA_DOMINA"] * 12)
    assert out["verdict"] == "ESCALA_DOMINA"
    assert out["robust"] is True
    assert out["agreement_fraction"] == pytest.approx(1.0)


def test_decide_verdict_below_robustness_threshold_falls_back_to_mixto():
    labels = ["ESCALA_DOMINA"] * 8 + ["FORMA_SUSTANCIAL"] * 4  # 8/12 = 0.667 < 0.75
    out = decide_verdict(labels)
    assert out["verdict"] == "MIXTO"
    assert out["robust"] is False


def test_decide_verdict_above_robustness_threshold_adopts_majority():
    n = 12
    n_majority = int(np.ceil(ROBUSTNESS_AGREEMENT_FRACTION * n))
    labels = ["FORMA_SUSTANCIAL"] * n_majority + ["ESCALA_DOMINA"] * (n - n_majority)
    out = decide_verdict(labels)
    assert out["verdict"] == "FORMA_SUSTANCIAL"
    assert out["robust"] is True


def test_decide_verdict_empty_list():
    out = decide_verdict([])
    assert out["verdict"] == "MIXTO"
    assert out["n_configs"] == 0


# ---------------------------------------------------------------------------
# G. QQ table
# ---------------------------------------------------------------------------

def test_build_qq_table_includes_r_and_each_config():
    rng = np.random.default_rng(14)
    r = rng.normal(size=1000)
    z_by_config = {("ewma", 60.0, "raw"): rng.normal(size=1000), ("rolling_std", 30.0, "raw"): rng.normal(size=1000)}
    tab = build_qq_table(r, z_by_config)
    assert set(tab["config"].unique()) == {"NA", "ewma_60_raw", "rolling_std_30_raw"}
    assert (tab.loc[tab["series"] == "r_1m_raw", "config"] == "NA").all()


def test_build_qq_table_skips_configs_with_too_few_finite_points():
    rng = np.random.default_rng(15)
    r = rng.normal(size=1000)
    z_by_config = {("ewma", 60.0, "raw"): np.full(1000, np.nan)}
    tab = build_qq_table(r, z_by_config)
    assert "ewma_60_raw" not in set(tab["config"])


# ---------------------------------------------------------------------------
# H. verify_populations_aligned
# ---------------------------------------------------------------------------

def test_verify_populations_aligned_passes_when_equal_length():
    a = pd.DataFrame({"x": range(10)})
    b = pd.DataFrame({"x": range(10)})
    verify_populations_aligned(a, b)  # no debe lanzar


def test_verify_populations_aligned_raises_on_mismatch():
    a = pd.DataFrame({"x": range(10)})
    b = pd.DataFrame({"x": range(5)})
    with pytest.raises(PopulationMismatchError):
        verify_populations_aligned(a, b)


# ---------------------------------------------------------------------------
# Fixtures sinteticas para pruebas extremo a extremo (mismo estilo que
# test_tda09_volatility_clustering.py)
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
    variables["abs_r_1m"] = variables["r_1m"].abs()
    variables["r2_1m"] = variables["r_1m"] ** 2
    variables["log_hl"] = np.log(variables["high"] / variables["low"])
    validity = pd.DataFrame(
        [{"timestamp": r["timestamp"], "r_1m_valid": r["r_1m_valid"], "invalid_reason": r.get("invalid_reason", "VALID" if r["r_1m_valid"] else "NON_CONSECUTIVE_MINUTE")} for r in rows]
    )
    return variables, validity


def _synthetic_multi_day_rows(n_days: int, start: datetime.date, bars_per_day: int = 100, seed: int = 0, return_gen=None) -> list[dict]:
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
            r = return_gen(rng, added, b) if return_gen is not None else rng.normal(0.0, 0.0005)
            price = price * (1.0 + r)
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
        "tda10": {
            "output_dir_reports": "reports/mnq",
            "n_boot": 5,
            "report_name": "TDA10_escala_vs_forma.md",
            "kurtosis_csv": "TDA10_curtosis_escala_vs_forma.csv",
            "kurtosis_bootstrap_ci_csv": "TDA10_curtosis_bootstrap_ci.csv",
            "quantile_by_decile_csv": "TDA10_cuantiles_por_decil_volatilidad.csv",
            "quantile_by_segment_csv": "TDA10_cuantiles_por_segmento.csv",
            "quantile_by_year_csv": "TDA10_cuantiles_por_anio.csv",
            "sensitivity_csv": "TDA10_sensibilidad_estimador_ventana.csv",
            "causality_check_csv": "TDA10_verificacion_causalidad.csv",
            "qq_points_csv": "TDA10_qq_puntos.csv",
            "qq_primary_png": "TDA10_qq_primario.png",
            "qq_sensitivity_png": "TDA10_qq_sensibilidad.png",
            "quantile_profile_png": "TDA10_perfil_cuantiles_decil.png",
        },
    }
    config_path = config_dir / "mnq_snapshot.yaml"
    config_path.write_text(yaml.safe_dump(config_yaml), encoding="utf-8")
    return load_config(config_path)


def _write_tda04_artifacts(config, variables: pd.DataFrame, validity: pd.DataFrame) -> None:
    config.interim_dir.mkdir(parents=True, exist_ok=True)
    variables.to_parquet(config.tda04_variables_parquet_path, index=False)
    validity.to_parquet(config.tda04_validity_mask_parquet_path, index=False)


def _write_tda06_artifacts(config, calendar_df: pd.DataFrame, cutoffs: list[int], s_m_by_minute: dict[int, float] | None = None) -> None:
    config.interim_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)

    if s_m_by_minute is None:
        s_m_map = {m: 1.0 for m in calendar_df["minute_of_day"].unique()}
    else:
        s_m_map = s_m_by_minute
    s_m_series = calendar_df["minute_of_day"].map(s_m_map).astype(float)

    r_tilde = pd.DataFrame({
        "timestamp": calendar_df["timestamp"], "source_file": calendar_df["source_file"],
        "contract": calendar_df["contract"], "trading_date": calendar_df["trading_date"],
        "minute_of_day": calendar_df["minute_of_day"], "weekday": calendar_df["weekday"],
        "r_1m": calendar_df["r_1m"], "r_1m_valid": calendar_df["r_1m_valid"],
        "s_m": s_m_series,
        "r_tilde": np.where(calendar_df["r_1m_valid"], calendar_df["r_1m"] / s_m_series, np.nan),
        "label": "RETROSPECTIVO",
    })
    r_tilde.to_parquet(config.tda06_r_tilde_parquet_path, index=False)

    seg_rows = [{"minute_of_day": c, "n_years_supporting": 5, "years_supporting": "2020,2021,2022,2023,2024", "stable": True} for c in cutoffs]
    pd.DataFrame(seg_rows).to_csv(config.tda06_segmentation_csv_path, index=False)


def _prepare_full_fixture(config, n_days=25, start=datetime.date(2024, 1, 8), cutoffs=(600, 900), return_gen=None, s_m_by_minute=None, bars_per_day=100):
    rows = _synthetic_multi_day_rows(n_days=n_days, start=start, bars_per_day=bars_per_day, return_gen=return_gen)
    variables, validity = _variables_and_validity(rows)
    _write_tda04_artifacts(config, variables, validity)
    calendar_df = attach_calendar_fields(variables, validity)
    _write_tda06_artifacts(config, calendar_df, list(cutoffs), s_m_by_minute=s_m_by_minute)
    return variables, validity


# ---------------------------------------------------------------------------
# I. Extremo a extremo
# ---------------------------------------------------------------------------

def test_run_tda10_raises_when_research_and_holdout_overlap(tmp_path):
    config = _make_config(tmp_path, research_files=["a.txt"], holdout_files=["a.txt"])
    with pytest.raises(HoldoutIsolationError):
        run_tda10_analysis(config, verbose=False)


def test_run_tda10_never_opens_any_raw_or_holdout_file(tmp_path, monkeypatch):
    config = _make_config(tmp_path, research_files=["research.txt"], holdout_files=["holdout.txt"])
    _prepare_full_fixture(config)

    import ohlcv_dataroad.ingest.parsing as parsing_mod

    def _forbidden(*args, **kwargs):
        raise AssertionError("TDA-10 no debe parsear ningun archivo crudo")

    if hasattr(parsing_mod, "parse_raw_file"):
        monkeypatch.setattr(parsing_mod, "parse_raw_file", _forbidden)

    result = run_tda10_analysis(config, verbose=False)
    assert len(result.r1m_population) > 0


def test_run_tda10_raises_on_misaligned_timestamps(tmp_path):
    config = _make_config(tmp_path, research_files=["research.txt"])
    variables, validity = _prepare_full_fixture(config)
    validity_bad = validity.iloc[:-1].copy()
    validity_bad.to_parquet(config.tda04_validity_mask_parquet_path, index=False)
    with pytest.raises(TimestampAlignmentError):
        run_tda10_analysis(config, verbose=False)


def test_run_tda10_raises_on_broken_r_tilde_invariant(tmp_path):
    config = _make_config(tmp_path, research_files=["research.txt"])
    _prepare_full_fixture(config)
    r_tilde = pd.read_parquet(config.tda06_r_tilde_parquet_path)
    r_tilde["label"] = "NO_ETIQUETADO"
    r_tilde.to_parquet(config.tda06_r_tilde_parquet_path, index=False)
    with pytest.raises(RTildeInvariantError):
        run_tda10_analysis(config, verbose=False)


def test_run_tda10_end_to_end_produces_all_result_fields(tmp_path):
    config = _make_config(tmp_path, research_files=["research.txt"])
    _prepare_full_fixture(config, n_days=25, bars_per_day=120)
    result = run_tda10_analysis(config, verbose=False)

    assert len(result.r1m_population) > 0
    assert len(result.r_tilde_population) == len(result.r1m_population)
    assert not result.causality_table.empty
    assert result.causality_table["passed"].all()
    assert not result.kurtosis_table.empty
    assert set(result.kurtosis_table["input_series"].unique()) == {"raw", "clock_adjusted"}
    assert isinstance(result.kurtosis_bootstrap_ci, dict)
    assert "fraction_removed_ci_lo" in result.kurtosis_bootstrap_ci
    assert not result.sensitivity_table.empty
    assert len(result.sensitivity_table) == 12
    assert {"fraction_removed_full", "fraction_removed_trimmed"}.issubset(result.sensitivity_table.columns)
    assert set(result.sensitivity_table["config_label"].unique()) <= {"ESCALA_DOMINA", "FORMA_SUSTANCIAL", "MIXTO"}
    assert "primary_fraction_removed_trimmed" in result.th22_verdict
    assert "primary_fraction_removed_full" in result.th22_verdict
    assert result.th22_verdict["verdict"] in {"ESCALA_DOMINA", "FORMA_SUSTANCIAL", "MIXTO"}
    assert result.th26_status == "PARCIALMENTE_INFORMADA"
    assert "suggested" in result.stop13_suggestion
    assert not result.qq_table.empty
    assert len(result.stage_timings) >= 7


def test_run_tda10_analysis_is_reproducible_given_fixed_seeds(tmp_path):
    config = _make_config(tmp_path, research_files=["research.txt"])
    _prepare_full_fixture(config, n_days=20, bars_per_day=100)
    r1 = run_tda10_analysis(config, verbose=False)
    r2 = run_tda10_analysis(config, verbose=False)
    pd.testing.assert_frame_equal(r1.kurtosis_table, r2.kurtosis_table)
    pd.testing.assert_frame_equal(r1.sensitivity_table, r2.sensitivity_table)
    assert r1.th22_verdict.keys() == r2.th22_verdict.keys()
    for k in r1.th22_verdict:
        np.testing.assert_equal(r1.th22_verdict[k], r2.th22_verdict[k])  # NaN == NaN aqui


def test_run_tda10_analysis_prints_stages_with_progress(tmp_path, capsys):
    config = _make_config(tmp_path, research_files=["research.txt"])
    _prepare_full_fixture(config, n_days=15, bars_per_day=80)
    run_tda10_analysis(config, verbose=True)
    out = capsys.readouterr().out
    assert "[TDA10 1/7]" in out
    assert "[TDA10 7/7]" in out


def test_run_tda10_analysis_silent_when_verbose_false(tmp_path, capsys):
    config = _make_config(tmp_path, research_files=["research.txt"])
    _prepare_full_fixture(config, n_days=15, bars_per_day=80)
    run_tda10_analysis(config, verbose=False)
    out = capsys.readouterr().out
    assert out == ""


def test_deterministic_scale_only_series_yields_high_fraction_removed():
    """Escenario sintetico: r_t = sigma_t * epsilon_t, con epsilon_t ~ N(0,1) iid y sigma_t con memoria genuina
    (proceso GARCH-like simple) -- el estimador causal deberia recuperar la mayor parte de la escala."""
    rng = np.random.default_rng(21)
    n = 20000
    sigma2 = np.empty(n)
    r = np.empty(n)
    sigma2[0] = 1e-6
    r[0] = rng.normal(0.0, np.sqrt(sigma2[0]))
    for t in range(1, n):
        sigma2[t] = 1e-7 + 0.85 * sigma2[t - 1] + 0.10 * r[t - 1] ** 2
        r[t] = rng.normal(0.0, np.sqrt(sigma2[t]))

    sigma_hat = causal_ewma_sigma(r, halflife=60.0)
    burn_in = burn_in_length("ewma", 60.0)
    z = standardize_return(r, sigma_hat, burn_in)
    year_ny = np.full(n, 2024)
    tab = build_kurtosis_table("ewma", 60.0, "raw", r, z, year_ny)
    global_row = tab.loc[tab["scope"] == "GLOBAL"].iloc[0]
    assert global_row["fraction_removed"] > 0.5  # una fraccion sustancial de la curtosis proviene de la escala dinamica


def test_persist_artifacts_writes_exactly_the_declared_paths_and_nothing_obsolete(tmp_path):
    from ohlcv_dataroad.ingest.run_tda10 import ARTIFACT_PATH_ATTRS, persist_artifacts

    config = _make_config(tmp_path, research_files=["research.txt"])
    _prepare_full_fixture(config, n_days=15, bars_per_day=80)
    result = run_tda10_analysis(config, verbose=False)
    written = persist_artifacts(result, config, t0=0.0, run_command="test")

    expected = {getattr(config, attr) for attr in ARTIFACT_PATH_ATTRS}
    assert set(written) == expected
    for p in expected:
        assert p.exists(), p


def test_persist_artifacts_generates_the_markdown_report_automatically(tmp_path):
    from ohlcv_dataroad.ingest.run_tda10 import persist_artifacts

    config = _make_config(tmp_path, research_files=["research.txt"])
    _prepare_full_fixture(config, n_days=15, bars_per_day=80)
    result = run_tda10_analysis(config, verbose=False)
    persist_artifacts(result, config, t0=0.0, run_command="python -m ohlcv_dataroad.ingest.run_tda10")

    text = config.tda10_report_path.read_text(encoding="utf-8")
    assert "TDA-10" in text
    assert "TH22" in text
    assert result.th22_verdict["verdict"] in text
