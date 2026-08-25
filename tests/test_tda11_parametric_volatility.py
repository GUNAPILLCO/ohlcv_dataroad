"""Tests de TDA-11 -- modelo parametrico de volatilidad (TH23/TH24/TH25).

No se reproducen aqui los numeros exactos del conjunto de investigacion
real (esos viven en ``reports/mnq/TDA11_modelo_parametrico_volatilidad.md``).
Se cubren: unidad (benchmark de rango causal, GARCH(1,1)/GJR, vida media,
diagnostico de residuos, decision de utilidad informativa, asimetria,
paralelizacion) y extremo a extremo (puerta de entrada, STOP-11,
aislamiento de hold-out, reproducibilidad) con datos sinteticos.
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
from ohlcv_dataroad.ingest.tda08_linear_mean_dependence import compute_acf
from ohlcv_dataroad.ingest.tda09_volatility_clustering import (
    STOP9_FRACTION_REMOVED_THRESHOLD,
    TH21_SURVIVES_GENUINE_THRESHOLD,
)
from ohlcv_dataroad.ingest.tda11_parametric_volatility import (
    ASYMMETRY_MATERIALITY_RELATIVE,
    GARCHFitError,
    GARCH_MIN_ALPHA_FOR_SEPARATION,
    GARCH_RESIDUAL_REDUCTION_THRESHOLD,
    GARCH_SCALE_FACTOR,
    MIN_N_FOR_GARCH_FIT,
    RowAlignmentError,
    causal_range_sigma,
    compute_half_life,
    decide_asymmetry_material,
    decide_garch_usefulness,
    default_n_workers,
    describe_asymmetry,
    fit_garch11,
    fit_garch_by_group,
    is_stationary,
    ljung_box_pvalue,
    parkinson_variance_proxy,
    residual_diagnostics,
    run_tda11_analysis,
    verify_entry_gate,
    verify_populations_row_aligned,
)

TICK = 0.25


# ---------------------------------------------------------------------------
# A. Benchmark de rango causal (Parkinson + EWMA)
# ---------------------------------------------------------------------------

def test_parkinson_variance_proxy_matches_closed_form():
    log_hl = np.array([0.001, 0.002, 0.0])
    out = parkinson_variance_proxy(log_hl)
    expected = (log_hl ** 2) / (4.0 * np.log(2.0))
    np.testing.assert_allclose(out, expected)


def test_causal_range_sigma_has_nan_burn_in_like_ewma():
    log_hl = np.abs(np.random.default_rng(1).normal(0.001, 0.0003, size=100))
    out = causal_range_sigma(log_hl, halflife=10.0)
    assert np.isnan(out[0])  # sin informacion previa
    assert np.isfinite(out[1:]).all()


def test_causal_range_sigma_never_uses_current_or_future_value():
    """Ausencia de look-ahead (seccion 14 de la tarea): perturbar log_hl[idx] NO debe cambiar sigma_hat en posiciones <= idx."""
    rng = np.random.default_rng(2)
    log_hl = np.abs(rng.normal(0.001, 0.0003, size=500))
    base = causal_range_sigma(log_hl, halflife=20.0)
    perturbed = log_hl.copy()
    perturbed[250] = 5.0  # rango extremo sintetico
    pert = causal_range_sigma(perturbed, halflife=20.0)
    np.testing.assert_array_equal(base[:251], pert[:251])
    assert not np.allclose(base[251:255], pert[251:255], equal_nan=True)


def test_causal_range_sigma_matches_ewma_of_variance_proxy_by_construction():
    """Verifica el "truco sqrt": causal_range_sigma(log_hl,h)^2 == EWMA causal del proxy de Parkinson."""
    rng = np.random.default_rng(3)
    log_hl = np.abs(rng.normal(0.001, 0.0003, size=300))
    sigma = causal_range_sigma(log_hl, halflife=15.0)
    proxy = parkinson_variance_proxy(log_hl)
    # sigma[t]^2 debe ser una EWMA causal de proxy[0..t-1] -- verificado
    # indirectamente: sigma[1] (primer valor finito) coincide con
    # sqrt(proxy[0]) (semilla de la recursion EWMA, TDA-10).
    assert sigma[1] == pytest.approx(np.sqrt(proxy[0]))


# ---------------------------------------------------------------------------
# B. Vida media y estacionariedad
# ---------------------------------------------------------------------------

def test_compute_half_life_hand_computed():
    persistence = 0.95
    hl = compute_half_life(persistence)
    assert hl == pytest.approx(np.log(0.5) / np.log(0.95))
    assert hl > 0


def test_compute_half_life_none_when_persistence_geq_one():
    assert compute_half_life(1.0) is None
    assert compute_half_life(1.02) is None


def test_compute_half_life_none_when_persistence_leq_zero_or_nan():
    assert compute_half_life(0.0) is None
    assert compute_half_life(-0.1) is None
    assert compute_half_life(np.nan) is None


def test_compute_half_life_none_when_persistence_is_floating_point_epsilon_below_one():
    """Auditoria post-primera-ejecucion: `persistence=0.999999999999999` (a distancia de punto flotante de 1,
    tipico de una restriccion de estacionariedad activa) NO debe producir una vida media fabricada (~10^14)."""
    assert compute_half_life(0.999999999999999) is None
    assert compute_half_life(1.0 - 1e-12) is None
    # muy por debajo de 1 (estacionario de verdad) SI debe dar una vida media finita
    assert compute_half_life(0.90) is not None


def test_is_stationary_true_below_one():
    assert is_stationary(0.95) is True
    assert is_stationary(0.99) is True


def test_is_stationary_false_at_or_above_one():
    assert is_stationary(1.0) is False
    assert is_stationary(1.0001) is False
    assert is_stationary(np.nan) is False


# ---------------------------------------------------------------------------
# C. GARCH(1,1) -- ajuste, recuperacion de parametros conocidos, errores
# ---------------------------------------------------------------------------

def _simulate_garch11(n: int, omega: float, alpha: float, beta: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sigma2 = np.empty(n)
    r = np.empty(n)
    sigma2[0] = omega / max(1e-8, (1 - alpha - beta))
    r[0] = rng.normal(0.0, np.sqrt(sigma2[0]))
    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
        r[t] = rng.normal(0.0, np.sqrt(sigma2[t]))
    return r


def test_fit_garch11_recovers_known_parameters_reasonably():
    """Caso sintetico donde un GARCH conocido es recuperable (seccion 14 de la tarea)."""
    true_omega, true_alpha, true_beta = 1e-8, 0.10, 0.85
    r = _simulate_garch11(60_000, true_omega, true_alpha, true_beta, seed=10)
    fit = fit_garch11(r, dist="normal")
    assert fit["alpha"] == pytest.approx(true_alpha, abs=0.04)
    assert fit["beta"] == pytest.approx(true_beta, abs=0.06)
    assert fit["persistence"] == pytest.approx(fit["alpha"] + fit["beta"])
    assert fit["convergence_flag"] == 0
    assert fit["stationary"] is True
    assert fit["half_life"] is not None
    assert fit["half_life"] > 0


def test_fit_garch11_alpha_plus_beta_equals_persistence_field():
    r = _simulate_garch11(25_000, 1e-8, 0.08, 0.88, seed=11)
    fit = fit_garch11(r, dist="normal")
    assert fit["persistence"] == pytest.approx(fit["alpha"] + fit["beta"], abs=1e-12)


def test_fit_garch11_omega_is_descaled_correctly():
    r = _simulate_garch11(25_000, 1e-8, 0.08, 0.88, seed=12)
    fit_a = fit_garch11(r, dist="normal", scale=1e4)
    fit_b = fit_garch11(r, dist="normal", scale=5e3)
    # alpha/beta/persistencia son invariantes a la escala; omega, ya
    # des-escalado, tambien debe coincidir aproximadamente entre escalas.
    assert fit_a["alpha"] == pytest.approx(fit_b["alpha"], abs=0.02)
    assert fit_a["beta"] == pytest.approx(fit_b["beta"], abs=0.03)
    assert fit_a["omega"] == pytest.approx(fit_b["omega"], rel=0.2)


def test_fit_garch11_standardized_resid_matches_input_over_conditional_vol():
    r = _simulate_garch11(20_000, 1e-8, 0.09, 0.85, seed=13)
    fit = fit_garch11(r, dist="normal")
    z = fit["standardized_resid"]
    assert z.shape == r.shape
    assert np.isfinite(z).all()
    # z debe tener varianza ~1 si el modelo esta razonablemente bien
    # especificado (chequeo de sanidad, no un test exacto).
    assert 0.5 < np.nanvar(z) < 2.0


def test_fit_garch11_asymmetric_returns_gamma():
    r = _simulate_garch11(20_000, 1e-8, 0.08, 0.85, seed=14)
    fit = fit_garch11(r, dist="normal", asymmetric=True)
    assert np.isfinite(fit["gamma"])
    assert fit["persistence"] == pytest.approx(fit["alpha"] + fit["beta"] + fit["gamma"] / 2.0)


def test_fit_garch11_prefers_converged_attempt_over_lower_llf_nonconverged(monkeypatch):
    """Auditoria post-primera-ejecucion: un intento con `convergence_flag!=0` NUNCA debe ganar sobre uno con `convergence_flag==0`, incluso si reporta una log-verosimilitud negativa menor (potencialmente enganosa)."""
    import ohlcv_dataroad.ingest.tda11_parametric_volatility as mod

    class _FakeConfInt:
        def __init__(self, names):
            self._df = pd.DataFrame({"lower": [0.0] * len(names), "upper": [1.0] * len(names)}, index=names)

        @property
        def loc(self):
            return self._df.loc

    class _FakeResult:
        def __init__(self, alpha, beta, omega, neg_llf, conv_flag):
            self.params = pd.Series({"omega": omega, "alpha[1]": alpha, "beta[1]": beta})
            self.loglikelihood = -neg_llf
            self.convergence_flag = conv_flag
            self.conditional_volatility = np.full(1000, 1.0)

        def conf_int(self):
            return _FakeConfInt(["omega", "alpha[1]", "beta[1]"])

    # El "no convergido" tiene MENOR -LLF (mejor en apariencia) pero flag!=0;
    # el "convergido" tiene -LLF peor pero flag==0 -- debe ganar este ultimo.
    fake_results = [
        _FakeResult(0.05, 0.90, 0.1, neg_llf=100.0, conv_flag=8),   # mejor LLF, NO convergido
        _FakeResult(0.08, 0.85, 0.2, neg_llf=105.0, conv_flag=0),   # peor LLF, SI convergido
        _FakeResult(0.09, 0.80, 0.3, neg_llf=110.0, conv_flag=0),
    ]
    call_count = {"i": 0}

    class _FakeModel:
        def fit(self, **kwargs):
            i = call_count["i"]
            call_count["i"] += 1
            return fake_results[min(i, len(fake_results) - 1)]

    monkeypatch.setattr(mod, "arch_model", lambda *a, **k: _FakeModel())
    r = np.random.default_rng(0).normal(0, 1e-4, 1000)
    fit = mod.fit_garch11(r, dist="normal")
    assert fit["alpha"] == pytest.approx(0.08)  # el convergido con MEJOR LLF entre los convergidos
    assert fit["beta"] == pytest.approx(0.85)
    assert fit["convergence_flag"] == 0
    assert fit["converged_cleanly"] is True


def test_fit_garch11_reports_multistart_diagnostics():
    """Auditoria post-primera-ejecucion: `fit_garch11` debe reportar que intento de la grilla de `eps` gano y cuantos fueron finitos (transparencia del multi-arranque)."""
    from ohlcv_dataroad.ingest.tda11_parametric_volatility import GARCH_OPTIMIZER_EPS_GRID

    r = _simulate_garch11(25_000, 1e-8, 0.08, 0.86, seed=40)
    fit = fit_garch11(r, dist="normal")
    assert fit["eps_selected"] in GARCH_OPTIMIZER_EPS_GRID
    assert 1 <= fit["n_multistart_attempts_finite"] <= len(GARCH_OPTIMIZER_EPS_GRID)
    assert fit["n_multistart_attempts_total"] == len(GARCH_OPTIMIZER_EPS_GRID)


def test_fit_garch11_multistart_never_returns_worse_than_default_eps_alone():
    """El multi-arranque debe encontrar una log-verosimilitud AL MENOS tan buena (baja) como usar solo el eps por defecto."""
    r = _simulate_garch11(25_000, 1e-8, 0.09, 0.85, seed=41)
    multi = fit_garch11(r, dist="normal")

    scale = GARCH_SCALE_FACTOR
    from arch import arch_model as _am

    am_default = _am(r * scale, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
    default_only = am_default.fit(disp="off", cov_type="robust", show_warning=False)
    assert -multi["loglikelihood"] <= -float(default_only.loglikelihood) + 1e-6


def test_fit_garch11_raises_on_degenerate_nan_input():
    r = np.full(5000, np.nan)
    with pytest.raises((GARCHFitError, Exception)):
        fit_garch11(r, dist="normal")


# ---------------------------------------------------------------------------
# D. Diagnostico de residuos (ACF/portmanteau/Ljung-Box), respeta bloques
# ---------------------------------------------------------------------------

def test_ljung_box_pvalue_matches_scipy_chi2():
    from scipy.stats import chi2

    q, m = 12.3, 10
    expected = 1.0 - chi2.cdf(q, df=m)
    assert ljung_box_pvalue(q, m) == pytest.approx(expected)


def test_ljung_box_pvalue_nan_for_nonfinite_q():
    assert np.isnan(ljung_box_pvalue(np.nan, 10))


def test_residual_diagnostics_never_pairs_across_block_boundary():
    """Respeto de mascaras de no-cruce (seccion 14 de la tarea): dos bloques con niveles MUY distintos de |z| no deben mezclarse en rho_1."""
    rng = np.random.default_rng(15)
    block_a = rng.normal(0.0, 1.0, size=300)
    block_b = rng.normal(0.0, 50.0, size=300)  # nivel muy distinto
    z = np.concatenate([block_a, block_b])
    block_ids = np.concatenate([np.zeros(300, dtype=int), np.ones(300, dtype=int)])

    diag = residual_diagnostics(z, block_ids, lags=(1,), m_values=(1,))
    # comparacion directa: la ACF por bloque (sin cruzar) NO debe mostrar
    # la correlacion espuria que fabricaria concatenar niveles distintos
    # ignorando bloques.
    naive_block_ids = np.zeros_like(block_ids)  # simula "un solo bloque" (cruza la frontera)
    diag_naive = residual_diagnostics(z, naive_block_ids, lags=(1,), m_values=(1,))
    assert diag["acf_z"]["rho"].iloc[0] != pytest.approx(diag_naive["acf_z"]["rho"].iloc[0])


def test_residual_diagnostics_burn_in_nan_does_not_poison_the_whole_acf():
    """Auditoria post-primera-ejecucion (bug real): un puñado de `NaN` de "quemado" al INICIO del array
    (tipico de un benchmark EWMA/rodante/rango recien construido, ver `compute_benchmark_z`) NUNCA debe
    convertir `rho`/`Q(m)` en `NaN` para TODA la serie -- `compute_acf` (TDA-08) no filtra `NaN` por si solo,
    asi que `residual_diagnostics` debe compactar `z`/`block_ids` ANTES de llamarlo."""
    rng = np.random.default_rng(9)
    n = 5000
    z = rng.normal(0.0, 1.0, size=n)
    z[:180] = np.nan  # quemado tipico de un EWMA half-life=60 (3*60=180)
    block_ids = np.zeros(n, dtype=int)

    # RESIDUAL_COMPARISON_LAG=1 y RESIDUAL_COMPARISON_M=60 son constantes
    # de modulo -- se usan los `lags`/`m_values` por defecto para que
    # `rho1_abs_z`/`q_abs_z_m` (que siempre leen esas constantes) sean
    # estimables.
    diag = residual_diagnostics(z, block_ids)
    assert np.isfinite(diag["rho1_abs_z"])
    assert np.isfinite(diag["q_abs_z_m"])
    assert diag["n_finite"] == n - 180
    # ninguna fila de la ACF resultante debe ser NaN cuando estimable=True
    estimable_rows = diag["acf_abs_z"].loc[diag["acf_abs_z"]["estimable"]]
    assert estimable_rows["rho"].notna().all()


def test_residual_diagnostics_reports_n_finite():
    z = np.array([1.0, 2.0, np.nan, 3.0, -1.0])
    block_ids = np.zeros(5, dtype=int)
    diag = residual_diagnostics(z, block_ids, lags=(1,), m_values=(1,))
    assert diag["n_finite"] == 4


# ---------------------------------------------------------------------------
# E. Decision de utilidad informativa de GARCH frente a benchmarks
# ---------------------------------------------------------------------------

def test_decide_garch_usefulness_true_when_both_conditions_hold():
    garch_diag = {"q_abs_z_m": 10.0}
    benchmarks = {"a": {"q_abs_z_m": 20.0}, "b": {"q_abs_z_m": 15.0}}
    garch_fit = {"alpha": 0.10}
    out = decide_garch_usefulness(garch_diag, benchmarks, garch_fit)
    assert out["useful"] is True
    assert out["relative_reduction"] == pytest.approx(1.0 - 10.0 / 15.0)


def test_decide_garch_usefulness_false_when_reduction_insufficient():
    garch_diag = {"q_abs_z_m": 14.0}  # apenas por debajo del mejor benchmark
    benchmarks = {"a": {"q_abs_z_m": 15.0}}
    garch_fit = {"alpha": 0.10}
    out = decide_garch_usefulness(garch_diag, benchmarks, garch_fit)
    assert out["reduces_dependence"] is False
    assert out["useful"] is False


def test_decide_garch_usefulness_false_when_alpha_too_small():
    garch_diag = {"q_abs_z_m": 5.0}  # gran reduccion...
    benchmarks = {"a": {"q_abs_z_m": 15.0}}
    garch_fit = {"alpha": 0.001}  # ...pero alpha colapsa (GARCH ~ EWMA puro)
    out = decide_garch_usefulness(garch_diag, benchmarks, garch_fit)
    assert out["reduces_dependence"] is True
    assert out["separates_impact"] is False
    assert out["useful"] is False


def test_decide_garch_usefulness_nan_when_no_benchmark_estimable():
    garch_diag = {"q_abs_z_m": 5.0}
    benchmarks = {"a": {"q_abs_z_m": np.nan}}
    garch_fit = {"alpha": 0.1}
    out = decide_garch_usefulness(garch_diag, benchmarks, garch_fit)
    assert out["useful"] is False
    assert np.isnan(out["relative_reduction"])


# ---------------------------------------------------------------------------
# F. TH25 -- asimetria descriptiva
# ---------------------------------------------------------------------------

def test_describe_asymmetry_hand_computed_pooled_stat():
    r_prev = np.array([1.0, -1.0, 1.0, -1.0] * 60)
    magnitude_next = np.array([0.10, 0.20, 0.10, 0.20] * 60)  # negativo previo -> magnitud mayor
    dates = np.repeat(np.arange(60), 4)
    out = describe_asymmetry(r_prev, magnitude_next, dates, n_boot=20, seed=1)
    # media pos=0.10, neg=0.20, pooled_mean=0.15 -> rel_diff=(0.20-0.10)/0.15
    assert out["pooled_rel_diff"] == pytest.approx((0.20 - 0.10) / 0.15, abs=1e-6)
    assert out["n"] == 240


def test_describe_asymmetry_no_asymmetry_gives_near_zero_diff():
    rng = np.random.default_rng(4)
    r_prev = rng.choice([-1.0, 1.0], size=2000)
    magnitude_next = np.full(2000, 0.15) + rng.normal(0, 0.001, 2000)  # sin relacion con el signo
    dates = np.repeat(np.arange(100), 20)
    out = describe_asymmetry(r_prev, magnitude_next, dates, n_boot=30, seed=2)
    assert abs(out["pooled_rel_diff"]) < 0.02


def test_decide_asymmetry_material_true_when_large_and_consistent():
    by_decile = pd.DataFrame({"decile": range(5), "rel_diff": [0.15, 0.18, 0.20, 0.12, 0.16]})
    out = decide_asymmetry_material(by_decile)
    assert out["material"] is True
    assert out["direction"] == "NEGATIVO_MAS_VOLATIL"


def test_decide_asymmetry_material_false_when_below_threshold():
    by_decile = pd.DataFrame({"decile": range(5), "rel_diff": [0.01, -0.02, 0.015, -0.01, 0.005]})
    out = decide_asymmetry_material(by_decile)
    assert out["material"] is False


def test_decide_asymmetry_material_false_when_empty():
    out = decide_asymmetry_material(pd.DataFrame(columns=["decile", "rel_diff"]))
    assert out["material"] is False


# ---------------------------------------------------------------------------
# G. Hardware -- workers CPU y paralelizacion determinista
# ---------------------------------------------------------------------------

def test_default_n_workers_caps_at_20():
    assert default_n_workers(n_jobs=1000) <= 20


def test_default_n_workers_never_exceeds_n_jobs():
    assert default_n_workers(n_jobs=3) <= 3


def test_default_n_workers_at_least_one():
    assert default_n_workers(n_jobs=0) >= 1


def test_fit_garch_by_group_flags_insufficient_sample_as_not_estimable():
    groups = {"tiny": np.random.default_rng(5).normal(0, 0.0004, size=100)}
    block_ids = {"tiny": np.zeros(100, dtype=int)}
    out = fit_garch_by_group(groups, block_ids, min_n=MIN_N_FOR_GARCH_FIT, n_workers=1)
    assert bool(out.iloc[0]["estimable"]) is False
    assert "MIN_N_FOR_GARCH_FIT" in out.iloc[0]["reason"]


def test_fit_garch_by_group_is_reproducible_sequential():
    rng = np.random.default_rng(6)
    groups = {"g1": _simulate_garch11(25_000, 1e-8, 0.08, 0.85, seed=20), "g2": _simulate_garch11(25_000, 1e-8, 0.06, 0.90, seed=21)}
    block_ids = {k: np.zeros(len(v), dtype=int) for k, v in groups.items()}
    out1 = fit_garch_by_group(groups, block_ids, n_workers=1)
    out2 = fit_garch_by_group(groups, block_ids, n_workers=1)
    pd.testing.assert_frame_equal(out1, out2)


def test_fit_garch_by_group_preserves_input_order():
    groups = {"z_last": np.random.default_rng(7).normal(0, 0.0004, 50), "a_first": np.random.default_rng(8).normal(0, 0.0004, 50)}
    block_ids = {k: np.zeros(len(v), dtype=int) for k, v in groups.items()}
    out = fit_garch_by_group(groups, block_ids, min_n=10_000_000, n_workers=1)  # fuerza NOT_ESTIMABLE, mas rapido
    assert list(out["group"]) == ["z_last", "a_first"]


def test_fit_garch_by_group_parallel_matches_sequential_results():
    groups = {
        "g1": _simulate_garch11(22_000, 1e-8, 0.08, 0.85, seed=30),
        "g2": _simulate_garch11(22_000, 1e-8, 0.07, 0.87, seed=31),
        "g3": _simulate_garch11(22_000, 1e-8, 0.09, 0.83, seed=32),
    }
    block_ids = {k: np.zeros(len(v), dtype=int) for k, v in groups.items()}
    seq = fit_garch_by_group(groups, block_ids, n_workers=1)
    par = fit_garch_by_group(groups, block_ids, n_workers=3)
    # Numericamente equivalentes, no necesariamente bit-identicos: el
    # ajuste en un proceso separado (con hilos BLAS limitados a 1) puede
    # diferir en el orden de operaciones de punto flotante del optimizador
    # frente al proceso principal (sin ese limite) -- una diferencia de
    # ruido numerico esperable, no una falla de reproducibilidad (la
    # reproducibilidad se exige entre dos ejecuciones IDENTICAS, ver
    # test_fit_garch_by_group_is_reproducible_sequential).
    assert list(seq["group"]) == list(par["group"])
    for col in ("alpha", "beta", "persistence"):
        np.testing.assert_allclose(seq[col].to_numpy(), par[col].to_numpy(), atol=1e-3)


# ---------------------------------------------------------------------------
# H. Alineacion de poblaciones (log_hl / s_m)
# ---------------------------------------------------------------------------

def test_verify_populations_row_aligned_passes_when_aligned():
    a = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=10, freq="min")})
    b = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=10, freq="min")})
    verify_populations_row_aligned(a, b)  # no debe lanzar


def test_verify_populations_row_aligned_raises_on_length_mismatch():
    a = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=10, freq="min")})
    b = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=5, freq="min")})
    with pytest.raises(RowAlignmentError):
        verify_populations_row_aligned(a, b)


def test_verify_populations_row_aligned_raises_on_timestamp_mismatch():
    a = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=10, freq="min")})
    b = pd.DataFrame({"timestamp": pd.date_range("2024-01-01 00:05", periods=10, freq="min")})
    with pytest.raises(RowAlignmentError):
        verify_populations_row_aligned(a, b)


# ---------------------------------------------------------------------------
# Fixtures sinteticas para pruebas extremo a extremo (mismo estilo que
# test_tda10_scale_vs_shape.py, extendido con TDA09_clock_attribution.csv
# para la puerta de entrada)
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


def _garch_return_gen(true_omega=1e-8, true_alpha=0.10, true_beta=0.85):
    state = {"sigma2": true_omega / (1 - true_alpha - true_beta), "prev_r": 0.0}

    def gen(rng, day_i, bar_i):
        state["sigma2"] = true_omega + true_alpha * state["prev_r"] ** 2 + true_beta * state["sigma2"]
        r = rng.normal(0.0, np.sqrt(state["sigma2"]))
        state["prev_r"] = r
        return r

    return gen


def _make_config(tmp_path: Path, research_files, holdout_files=None, boundary_utc="2099-01-01 00:00:00", n_workers=1):
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
        "tda09": {
            "output_dir_reports": "reports/mnq",
            "n_boot": 5, "n_perm": 5, "n_perm_arch_lm": 5,
            "report_name": "TDA09_volatility_clustering.md",
            "acf_csv": "TDA09_acf_magnitud.csv",
            "bootstrap_ci_csv": "TDA09_bootstrap_ci.csv",
            "clock_attribution_csv": "TDA09_clock_attribution.csv",
            "persistence_by_year_csv": "TDA09_persistencia_por_anio.csv",
            "persistence_by_segment_csv": "TDA09_persistencia_por_segmento.csv",
            "persistence_rolling_csv": "TDA09_persistencia_ventana_rodante.csv",
            "portmanteau_csv": "TDA09_portmanteau.csv",
            "arch_lm_csv": "TDA09_arch_lm.csv",
            "g2_calibration_null1_csv": "TDA09_g2_calibracion_null1_principal.csv",
            "g2_calibration_secondary_csv": "TDA09_g2_calibracion_secundaria_global_ajustada.csv",
            "g2_synthetic_moment_check_csv": "TDA09_g2_diagnostico_momentos_sintetico.csv",
            "mean_removal_sensitivity_csv": "TDA09_sensibilidad_remocion_media.csv",
            "clock_flatness_csv": "TDA09_diagnostico_aplanamiento_reloj.csv",
            "acf_triple_png": "TDA09_acf_triple.png",
            "acf_raw_vs_adjusted_png": "TDA09_acf_raw_vs_adjustado.png",
            "decay_png": "TDA09_decay_loglog_semilog.png",
        },
        "tda11": {
            "output_dir_reports": "reports/mnq",
            "n_boot": 20, "n_workers": n_workers,
            "report_name": "TDA11_modelo_parametrico_volatilidad.md",
            "entry_gate_csv": "TDA11_puerta_de_entrada.csv",
            "benchmarks_csv": "TDA11_benchmarks_comparativa.csv",
            "garch_params_csv": "TDA11_garch_parametros.csv",
            "garch_diagnostics_csv": "TDA11_garch_diagnosticos_residuos.csv",
            "asymmetry_csv": "TDA11_asimetria_th25.csv",
            "usefulness_csv": "TDA11_utilidad_informativa.csv",
            "acf_comparison_png": "TDA11_acf_residuos_comparacion.png",
            "stability_png": "TDA11_persistencia_estabilidad.png",
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


def _write_tda09_clock_attribution(config, genuine: bool = True, stop9: bool = False) -> None:
    """Escribe un `TDA09_clock_attribution.csv` sintetico -- controla la puerta de entrada de TDA-11 en los tests."""
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    if stop9:
        fraction_survives, fraction_removed = 0.02, 0.98  # colapsa -> STOP-9 activado
    elif genuine:
        fraction_survives, fraction_removed = TH21_SURVIVES_GENUINE_THRESHOLD + 0.3, 1.0 - (TH21_SURVIVES_GENUINE_THRESHOLD + 0.3)
    else:
        fraction_survives, fraction_removed = 0.05, 0.95  # NO genuino (artefacto de reloj)
    rows = [
        {"variable": "abs_r", "m": 240, "Q_raw": 1000.0, "Q_adjusted": 1000.0 * fraction_survives, "fraction_removed": fraction_removed, "fraction_survives": fraction_survives},
        {"variable": "log_hl", "m": 240, "Q_raw": 2000.0, "Q_adjusted": 2000.0 * fraction_survives, "fraction_removed": fraction_removed, "fraction_survives": fraction_survives},
    ]
    pd.DataFrame(rows).to_csv(config.tda09_clock_attribution_csv_path, index=False)


def _prepare_full_fixture(
    config, n_days=60, start=datetime.date(2024, 1, 8), cutoffs=(600, 900), return_gen=None,
    s_m_by_minute=None, bars_per_day=100, gate_genuine=True, gate_stop9=False,
):
    rows = _synthetic_multi_day_rows(n_days=n_days, start=start, bars_per_day=bars_per_day, return_gen=return_gen)
    variables, validity = _variables_and_validity(rows)
    _write_tda04_artifacts(config, variables, validity)
    calendar_df = attach_calendar_fields(variables, validity)
    _write_tda06_artifacts(config, calendar_df, list(cutoffs), s_m_by_minute=s_m_by_minute)
    _write_tda09_clock_attribution(config, genuine=gate_genuine, stop9=gate_stop9)
    return variables, validity


# ---------------------------------------------------------------------------
# I. Puerta de entrada / STOP-11 -- extremo a extremo
# ---------------------------------------------------------------------------

def test_verify_entry_gate_missing_csv_closes_gate(tmp_path):
    config = _make_config(tmp_path, research_files=["research.txt"])
    out = verify_entry_gate(config)
    assert out.gate_open is False


def test_verify_entry_gate_opens_when_genuine_and_stop9_not_activated(tmp_path):
    config = _make_config(tmp_path, research_files=["research.txt"])
    _write_tda09_clock_attribution(config, genuine=True, stop9=False)
    out = verify_entry_gate(config)
    assert out.gate_open is True
    assert out.th21_verdict["verdict"] == "CLUSTERING_GENUINO"
    assert out.stop9_decision["stop9_activated"] is False


def test_verify_entry_gate_closed_when_not_genuine(tmp_path):
    config = _make_config(tmp_path, research_files=["research.txt"])
    _write_tda09_clock_attribution(config, genuine=False, stop9=False)
    out = verify_entry_gate(config)
    assert out.gate_open is False
    assert out.th21_verdict["verdict"] != "CLUSTERING_GENUINO"


def test_verify_entry_gate_closed_when_stop9_activated(tmp_path):
    config = _make_config(tmp_path, research_files=["research.txt"])
    _write_tda09_clock_attribution(config, genuine=True, stop9=True)
    out = verify_entry_gate(config)
    assert out.gate_open is False


def test_run_tda11_raises_when_research_and_holdout_overlap(tmp_path):
    config = _make_config(tmp_path, research_files=["a.txt"], holdout_files=["a.txt"])
    with pytest.raises(HoldoutIsolationError):
        run_tda11_analysis(config, n_workers=1, verbose=False)


def test_run_tda11_stop11_when_entry_gate_closed(tmp_path):
    config = _make_config(tmp_path, research_files=["research.txt"])
    _prepare_full_fixture(config, n_days=15, bars_per_day=50, gate_genuine=False)
    result = run_tda11_analysis(config, n_workers=1, verbose=False)
    assert result.stop11_activated is True
    assert result.entry_gate.gate_open is False
    assert result.n_raw == 0  # ningun benchmark/poblacion se construyo


def test_run_tda11_never_opens_any_raw_or_holdout_file(tmp_path, monkeypatch):
    config = _make_config(tmp_path, research_files=["research.txt"], holdout_files=["holdout.txt"])
    _prepare_full_fixture(config, n_days=15, bars_per_day=50)

    import ohlcv_dataroad.ingest.parsing as parsing_mod

    def _forbidden(*args, **kwargs):
        raise AssertionError("TDA-11 no debe parsear ningun archivo crudo")

    if hasattr(parsing_mod, "parse_raw_file"):
        monkeypatch.setattr(parsing_mod, "parse_raw_file", _forbidden)

    result = run_tda11_analysis(config, n_workers=1, verbose=False)
    assert result.entry_gate.gate_open is True


# ---------------------------------------------------------------------------
# J. Extremo a extremo -- puerta abierta (bateria completa)
# ---------------------------------------------------------------------------

def test_run_tda11_end_to_end_produces_all_result_fields(tmp_path):
    config = _make_config(tmp_path, research_files=["research.txt"], n_workers=2)
    _prepare_full_fixture(config, n_days=60, bars_per_day=150, return_gen=_garch_return_gen())
    result = run_tda11_analysis(config, n_workers=2, verbose=False)

    assert result.entry_gate.gate_open is True
    assert result.n_raw > 0
    assert result.n_clock_adjusted == result.n_raw
    assert set(result.benchmark_diagnostics.keys()) == {"raw", "clock_adjusted"}
    assert set(result.benchmark_diagnostics["raw"].keys()) == {"ewma_60", "rolling_120", "range_ewma_60"}
    assert "alpha" in result.garch_primary_global
    assert "alpha" in result.garch_secondary_raw_global
    assert "alpha" in result.garch_student_t_sensitivity
    assert result.garch_by_year is not None and len(result.garch_by_year) >= 1
    assert result.garch_by_segment is not None and len(result.garch_by_segment) == 3  # cutoffs=(600,900) -> 3 tramos en este fixture
    assert "useful" in result.garch_usefulness
    assert "pooled_rel_diff" in result.asymmetry_global
    assert isinstance(result.asymmetry_decision, dict)
    assert isinstance(result.stop11_activated, bool)
    assert len(result.stage_timings) >= 6


def test_run_tda11_analysis_is_reproducible_given_fixed_seeds(tmp_path):
    config = _make_config(tmp_path, research_files=["research.txt"], n_workers=2)
    _prepare_full_fixture(config, n_days=40, bars_per_day=100, return_gen=_garch_return_gen())
    r1 = run_tda11_analysis(config, n_workers=2, verbose=False)
    r2 = run_tda11_analysis(config, n_workers=2, verbose=False)
    assert r1.garch_primary_global["alpha"] == r2.garch_primary_global["alpha"]
    assert r1.garch_primary_global["beta"] == r2.garch_primary_global["beta"]
    pd.testing.assert_frame_equal(r1.garch_by_year, r2.garch_by_year)
    pd.testing.assert_frame_equal(r1.garch_by_segment, r2.garch_by_segment)
    assert r1.stop11_activated == r2.stop11_activated


def test_run_tda11_analysis_prints_stages_with_progress(tmp_path, capsys):
    config = _make_config(tmp_path, research_files=["research.txt"], n_workers=1)
    _prepare_full_fixture(config, n_days=30, bars_per_day=80, return_gen=_garch_return_gen())
    run_tda11_analysis(config, n_workers=1, verbose=True)
    out = capsys.readouterr().out
    assert "[TDA11 1/6]" in out
    assert "[TDA11 6/6]" in out


def test_run_tda11_analysis_silent_when_verbose_false(tmp_path, capsys):
    config = _make_config(tmp_path, research_files=["research.txt"], n_workers=1)
    _prepare_full_fixture(config, n_days=20, bars_per_day=60, return_gen=_garch_return_gen())
    run_tda11_analysis(config, n_workers=1, verbose=False)
    out = capsys.readouterr().out
    assert out == ""


def test_run_tda11_subperiods_below_min_n_are_marked_not_estimable(tmp_path):
    """Caso realista: con datos sinteticos pequeños, la mayoria de años/segmentos no alcanzan MIN_N_FOR_GARCH_FIT -- el pipeline debe marcarlo, no fallar."""
    config = _make_config(tmp_path, research_files=["research.txt"], n_workers=1)
    _prepare_full_fixture(config, n_days=20, bars_per_day=60, return_gen=_garch_return_gen())
    result = run_tda11_analysis(config, n_workers=1, verbose=False)
    assert (result.garch_by_year["estimable"] == False).any() or (result.garch_by_segment["estimable"] == False).any()


def test_run_tda11_iid_series_does_not_invent_garch_usefulness(tmp_path):
    """Caso sin clustering (seccion 14 de la tarea): el pipeline NO debe inventar utilidad de GARCH sobre retornos genuinamente IID.

    Se fuerza la puerta de entrada ABIERTA (clock_attribution sintetico
    marcado CLUSTERING_GENUINO) para poder ejercitar el pipeline
    completo, pero la serie `r_tilde` real es IID -- GARCH debe converger
    a alpha/beta ~0 y `decide_garch_usefulness` debe dar `useful=False`.
    """
    config = _make_config(tmp_path, research_files=["research.txt"], n_workers=1)
    _prepare_full_fixture(config, n_days=60, bars_per_day=150, return_gen=None)  # IID (sin return_gen -> normal iid)
    result = run_tda11_analysis(config, n_workers=1, verbose=False)
    assert result.entry_gate.gate_open is True
    assert result.garch_usefulness["useful"] is False
    assert result.stop11_activated is True


def test_persist_artifacts_writes_exactly_the_declared_paths_open_gate(tmp_path):
    from ohlcv_dataroad.ingest.run_tda11 import ARTIFACT_PATH_ATTRS, persist_artifacts

    config = _make_config(tmp_path, research_files=["research.txt"], n_workers=1)
    _prepare_full_fixture(config, n_days=30, bars_per_day=80, return_gen=_garch_return_gen())
    result = run_tda11_analysis(config, n_workers=1, verbose=False)
    written = persist_artifacts(result, config, t0=0.0, run_command="test")

    expected = {getattr(config, attr) for attr in ARTIFACT_PATH_ATTRS}
    assert set(written) == expected
    for p in expected:
        assert p.exists(), p


def test_persist_artifacts_writes_minimal_paths_when_gate_closed(tmp_path):
    from ohlcv_dataroad.ingest.run_tda11 import persist_artifacts

    config = _make_config(tmp_path, research_files=["research.txt"], n_workers=1)
    _prepare_full_fixture(config, n_days=15, bars_per_day=50, gate_genuine=False)
    result = run_tda11_analysis(config, n_workers=1, verbose=False)
    written = persist_artifacts(result, config, t0=0.0, run_command="test")

    assert config.tda11_entry_gate_csv_path in written
    assert config.tda11_report_path in written
    assert config.tda11_garch_params_csv_path not in written
    assert config.tda11_entry_gate_csv_path.exists()
    assert config.tda11_report_path.exists()


def test_persist_artifacts_generates_the_markdown_report_automatically(tmp_path):
    from ohlcv_dataroad.ingest.run_tda11 import persist_artifacts

    config = _make_config(tmp_path, research_files=["research.txt"], n_workers=1)
    _prepare_full_fixture(config, n_days=30, bars_per_day=80, return_gen=_garch_return_gen())
    result = run_tda11_analysis(config, n_workers=1, verbose=False)
    persist_artifacts(result, config, t0=0.0, run_command="python -m ohlcv_dataroad.ingest.run_tda11")

    text = config.tda11_report_path.read_text(encoding="utf-8")
    assert "TDA-11" in text
    assert "STOP-11" in text
    assert "GARCH" in text


def test_persist_artifacts_generates_stop11_report_when_gate_closed(tmp_path):
    from ohlcv_dataroad.ingest.run_tda11 import persist_artifacts

    config = _make_config(tmp_path, research_files=["research.txt"], n_workers=1)
    _prepare_full_fixture(config, n_days=15, bars_per_day=50, gate_genuine=False)
    result = run_tda11_analysis(config, n_workers=1, verbose=False)
    persist_artifacts(result, config, t0=0.0, run_command="test")

    text = config.tda11_report_path.read_text(encoding="utf-8")
    assert "STOP-11" in text
    assert "ACTIVADO" in text
