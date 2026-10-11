"""Tests de `audit_tda11_convergence.py` -- el script que conserva la evidencia de C2/C1 de la auditoria correctiva de TDA-11.

Sobre datos sinteticos pequeños: comprueban que las dos sondas miden lo
que dicen medir y que el script no altera ni depende de nada fuera de
los insumos derivados de TDA-11.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ohlcv_dataroad.ingest.audit_tda11_convergence as audit
from ohlcv_dataroad.ingest.tda11_parametric_volatility import (
    GARCH_KKT_GRADIENT_TOL,
    GARCH_OPTIMIZER_EPS_GRID,
    GARCH_SCALE_FACTOR,
    fit_garch11,
    garch_mean_neg_loglik,
)


def _simulate_garch11(n: int, omega: float, alpha: float, beta: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    s2 = omega / max(1e-8, 1 - alpha - beta)
    for t in range(n):
        r[t] = rng.normal() * np.sqrt(s2)
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    return r


def test_free_persistence_nll_equals_tda11_likelihood_inside_the_arch_constraint():
    """La sonda usa la MISMA verosimilitud que TDA-11: solo deja de rechazar persistencia>1."""
    r = _simulate_garch11(6_000, 1e-8, 0.09, 0.86, seed=1)
    x = (r * GARCH_SCALE_FACTOR) ** 2
    for log_omega, alpha, p in [(-1.0, 0.10, 0.95), (-2.0, 0.04, 1.0), (-0.5, 0.30, 0.70)]:
        assert audit.free_persistence_mean_nll(log_omega, alpha, p, x) == pytest.approx(
            garch_mean_neg_loglik(np.array([log_omega, alpha, p]), x), rel=1e-13
        )
    # ...y fuera de la restriccion TDA-11 rechaza, la sonda evalua
    assert garch_mean_neg_loglik(np.array([-2.0, 0.06, 1.005]), x) >= 1e9
    assert audit.free_persistence_mean_nll(-2.0, 0.06, 1.005, x) < 1e9


def test_persistence_profile_interior_optimum_when_constraint_is_not_binding():
    """GARCH estacionario genuino: el maximo libre coincide con el restringido (<1) y el perfil decrece a ambos lados."""
    r = _simulate_garch11(30_000, 1e-8, 0.10, 0.80, seed=2)
    fit = fit_garch11(r)
    prof = audit.audit_persistence_profile(r, fit)
    free = prof.loc[prof["kind"] == "free_persistence_optimum"].iloc[0]
    assert free["persistence"] == pytest.approx(fit["persistence"], abs=1e-4)
    assert abs(free["llf_minus_constrained_optimum"]) < 1e-3
    assert bool(free["feasible_under_arch_constraint"]) is True
    grid = prof.loc[prof["kind"] == "profile_fixed_persistence"]
    assert (grid["llf_minus_constrained_optimum"] <= 1e-6).all()   # ningun punto del perfil supera al optimo


def test_persistence_profile_detects_a_binding_constraint():
    """Volatilidad con deriva (sin dinamica GARCH): el optimo restringido queda en el tope y el maximo libre lo SUPERA por encima de 1."""
    rng = np.random.default_rng(109)
    n = 40_000
    r = rng.normal(size=n) * 1e-4 * np.exp(3.0 * np.arange(n) / n)
    fit = fit_garch11(r)
    assert fit["at_constraint_boundary"] is True
    prof = audit.audit_persistence_profile(r, fit)
    free = prof.loc[prof["kind"] == "free_persistence_optimum"].iloc[0]
    assert free["persistence"] > 1.0
    assert free["llf_minus_constrained_optimum"] > 0.0
    assert bool(free["feasible_under_arch_constraint"]) is False
    assert bool(free["probe_bound_active"]) is False
    assert set(prof["kind"]) == {"profile_fixed_persistence", "free_persistence_optimum", "constrained_optimum_tda11"}
    assert len(prof) == len(audit.PERSISTENCE_PROFILE_GRID) + 2


def test_audit_arch_attempts_table_structure_and_reference_row():
    r = _simulate_garch11(20_000, 1e-8, 0.08, 0.88, seed=3)
    tab, fit = audit.audit_arch_attempts(r, "normal", "sintetico")
    att = tab.loc[tab["source"] == "arch_slsqp_attempt"]
    ref = tab.loc[tab["source"] == "verified_optimum_fit_garch11"].iloc[0]
    assert len(att) == len(GARCH_OPTIMIZER_EPS_GRID)
    assert ref["llf_minus_verified_optimum"] == 0.0
    assert bool(ref["passes_kkt_tolerance"]) is True and ref["kkt_max_abs_gradient"] < GARCH_KKT_GRADIENT_TOL
    # ningun intento de arch puede superar al optimo verificado (misma verosimilitud, misma restriccion)
    feasible = att.loc[att["persistence"] <= 1.0]
    assert (feasible["llf_minus_verified_optimum"] <= 1e-6 * abs(fit["loglikelihood"])).all()
    # coherencia interna: un intento marcado "no pasa KKT" esta de verdad por debajo del optimo
    bad = feasible.loc[~feasible["passes_kkt_tolerance"].astype(bool)]
    assert (bad["llf_minus_verified_optimum"] < 0.0).all()


def test_audit_arch_attempts_marks_infeasible_point_instead_of_inventing_a_gradient(monkeypatch):
    """Un intento de arch que viola alpha+beta<=1 (ocurrio con Student-t sobre datos reales) no recibe un gradiente fabricado."""
    r = _simulate_garch11(8_000, 1e-8, 0.08, 0.88, seed=4)

    real_arch_model = audit.arch_model

    class _Wrapped:
        def __init__(self, am):
            self._am = am
            self.volatility = am.volatility

        def fit(self, **kwargs):
            res = self._am.fit(**kwargs)

            class _R:
                params = pd.Series({"omega": 0.05, "alpha[1]": 0.10, "beta[1]": 0.92})   # persistencia 1.02
                loglikelihood = float(res.loglikelihood)
                convergence_flag = 8
                optimization_result = res.optimization_result
            return _R()

    monkeypatch.setattr(audit, "arch_model", lambda *a, **k: _Wrapped(real_arch_model(*a, **k)))
    tab, _ = audit.audit_arch_attempts(r, "normal", "sintetico")
    att = tab.loc[tab["source"] == "arch_slsqp_attempt"]
    assert att["kkt_max_abs_gradient"].isna().all()
    assert att["kkt_note"].str.contains("fuera de la region admisible").all()
    assert not att["passes_kkt_tolerance"].astype(bool).any()


def test_render_audit_report_states_scope_and_non_conclusions():
    rng = np.random.default_rng(109)
    n = 30_000
    r = rng.normal(size=n) * 1e-4 * np.exp(3.0 * np.arange(n) / n)
    tab, fit = audit.audit_arch_attempts(r, "normal", "garch11_primary_global")
    prof = audit.audit_persistence_profile(r, fit)
    text = audit.render_audit_report(tab, prof, 1.0, "test")
    for needle in ("No forma parte del pipeline", "holdout_files", "NO significa", "IGARCH", "GARCH_KKT_GRADIENT_TOL"):
        assert needle in text


def test_audit_module_declares_only_its_own_artifacts_and_never_names_raw_inputs():
    names = {audit.ATTEMPTS_CSV_NAME, audit.PROFILE_CSV_NAME, audit.REPORT_NAME}
    assert all(n.startswith("TDA11_auditoria_") for n in names) and len(names) == 3
    import inspect

    src = inspect.getsource(audit.main) + inspect.getsource(audit.load_audit_series)
    assert "holdout_files" not in src and "raw_dir" not in src and "data/raw" not in src
    assert "validate_research_holdout_disjoint" in src and "validate_last_timestamps_before_boundary" in src
