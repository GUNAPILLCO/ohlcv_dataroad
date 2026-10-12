"""Tests de TDA-12 -- cuantiles condicionales (TH26).

Cubre: (A) el nucleo ponderado del bootstrap contra el calculo explicito;
(B) el bootstrap conjunto por jornada; (C) soporte de cuantiles extremos;
(D) el decil CAUSAL de volatilidad y sus pruebas anti-leakage; (E/F)
escala vs forma sobre casos sinteticos de respuesta conocida; (G/H) la
regla de veredicto y la implicacion para STOP-13; (I) casos limite
(muestras pequeñas, datos faltantes, desviacion nula, grupos
insuficientes); (J) extremo a extremo, reproducibilidad, hold-out y
artefactos.
"""
from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import ohlcv_dataroad.ingest.tda12_conditional_quantiles as mod
from ohlcv_dataroad.config import load_config
from ohlcv_dataroad.ingest.holdout_guard import HoldoutIsolationError
from ohlcv_dataroad.ingest.tda06_intraday_calendar_profile import attach_calendar_fields
from ohlcv_dataroad.ingest.tda07_marginal_distribution import QUANTILE_LEVELS, TRIM_FRACTION_EACH_TAIL, compute_moments_quantiles
from ohlcv_dataroad.ingest.tda10_scale_vs_shape import (
    PRIMARY_ESTIMATOR,
    PROFILE_STABILITY_FORM_THRESHOLD,
    PROFILE_STABILITY_SCALE_THRESHOLD,
    ROBUSTNESS_AGREEMENT_FRACTION,
    LookaheadLeakageError,
    assign_volatility_decile,
    build_sigma_hat,
    burn_in_length,
    standardize_return,
)
from ohlcv_dataroad.ingest.tda12_conditional_quantiles import (
    LABEL_DIVERGENT,
    LABEL_INCONCLUSIVE,
    LABEL_INTERMEDIATE,
    LABEL_OVERLAP,
    NORMALIZATIONS,
    SYMMETRIC_LEVEL_PAIRS,
    VERDICT_GROUPINGS,
    VERDICT_INCONCLUSIVE,
    VERDICT_LEVEL_PAIR,
    VERDICT_SCALE_AND_SHAPE,
    VERDICT_SCALE_ONLY,
    CausalDecileLeakageError,
    build_asymmetry_table,
    build_quantile_table,
    build_shape_comparison_table,
    causal_volatility_decile,
    classify_ratio,
    decide_stop13_implication,
    decide_th26,
    joint_day_block_bootstrap,
    prepare_sorted_groups,
    profile_ratio,
    run_tda12_analysis,
    support_label,
    tail_support,
    verify_causal_decile_no_lookahead,
    weighted_group_stats,
)

K = len(QUANTILE_LEVELS)
TICK = 0.25


# ---------------------------------------------------------------------------
# 0. Constantes heredadas: TDA-12 no cambia ningun valor aprobado
# ---------------------------------------------------------------------------

def test_inherited_constants_are_unchanged():
    assert QUANTILE_LEVELS == (0.001, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999)   # los percentiles del roadmap
    assert mod.QUANTILE_LEVELS is QUANTILE_LEVELS
    assert (PROFILE_STABILITY_SCALE_THRESHOLD, PROFILE_STABILITY_FORM_THRESHOLD, ROBUSTNESS_AGREEMENT_FRACTION) == (0.30, 0.60, 0.75)
    assert mod.PROFILE_STABILITY_SCALE_THRESHOLD is PROFILE_STABILITY_SCALE_THRESHOLD
    assert PRIMARY_ESTIMATOR == ("ewma", 60.0, "raw")
    assert VERDICT_LEVEL_PAIR == (0.01, 0.99)
    assert mod.MIN_TAIL_SUPPORT_LOW == 200 and mod.MIN_TAIL_SUPPORT_UNRELIABLE == 20
    assert set(VERDICT_GROUPINGS) == {"SEGMENT", "VOL_DECILE_CAUSAL", "YEAR"}


# ---------------------------------------------------------------------------
# A. Nucleo ponderado == calculo explicito sobre la muestra materializada
# ---------------------------------------------------------------------------

def _explicit_stats(sample: np.ndarray) -> np.ndarray:
    out = np.full(K + 3, np.nan)
    n = sample.size
    out[-1] = n
    if n == 0:
        return out
    out[:K] = np.quantile(sample, QUANTILE_LEVELS)
    if n > 1:
        out[K] = sample.std(ddof=1)
    if n >= 20:
        lo, hi = np.quantile(sample, [TRIM_FRACTION_EACH_TAIL, 1 - TRIM_FRACTION_EACH_TAIL])
        t = sample[(sample >= lo) & (sample <= hi)]
        if t.size > 1:
            out[K + 1] = t.std(ddof=1)
    return out


def test_weighted_stats_without_weights_match_numpy_and_tda07_conventions():
    rng = np.random.default_rng(1)
    x = np.sort(rng.standard_t(3, size=5_000) * 3e-4)
    got = weighted_group_stats(x, None)
    np.testing.assert_allclose(got, _explicit_stats(x), rtol=1e-12, atol=0)
    ref = compute_moments_quantiles(x)        # convencion de TDA-07
    for j, lvl in enumerate(QUANTILE_LEVELS):
        assert got[j] == pytest.approx(ref[f"q{lvl}"], rel=1e-12)
    assert got[K] == pytest.approx(ref["std"], rel=1e-12)


def test_weighted_stats_match_explicitly_repeated_sample():
    """Cada replica del bootstrap debe ser IDENTICA a materializar la muestra remuestreada y usar np.quantile/std."""
    rng = np.random.default_rng(2)
    for n in (25, 400, 3_000):
        x = np.sort(rng.normal(size=n))
        w = rng.integers(0, 5, size=n)          # multiplicidades, incluidos ceros
        np.testing.assert_allclose(weighted_group_stats(x, w), _explicit_stats(np.repeat(x, w)), rtol=1e-10, atol=1e-12)


def test_weighted_stats_with_ties_match_explicit():
    rng = np.random.default_rng(3)
    x = np.sort(np.round(rng.normal(size=2_000), 1))   # muchos empates (como retornos discretos por el tick)
    w = rng.integers(0, 4, size=x.size)
    np.testing.assert_allclose(weighted_group_stats(x, w), _explicit_stats(np.repeat(x, w)), rtol=1e-10, atol=1e-12)


def test_weighted_stats_edge_cases_small_empty_and_constant():
    empty = weighted_group_stats(np.array([]), None)
    assert empty[-1] == 0 and np.isnan(empty[:-1]).all()
    zero_weight = weighted_group_stats(np.array([1.0, 2.0]), np.array([0, 0]))
    assert zero_weight[-1] == 0 and np.isnan(zero_weight[:-1]).all()
    one = weighted_group_stats(np.array([3.0]), None)
    assert one[-1] == 1 and (one[:K] == 3.0).all() and np.isnan(one[K]) and np.isnan(one[K + 1])
    small = weighted_group_stats(np.sort(np.arange(19.0)), None)
    assert np.isfinite(small[K]) and np.isnan(small[K + 1])       # n<20: sin desviacion recortada (TDA-07)
    const = weighted_group_stats(np.full(500, 7.0), None)
    assert (const[:K] == 7.0).all() and const[K] == 0.0 and const[K + 1] == 0.0   # desviacion NULA, sin error


# ---------------------------------------------------------------------------
# B. Bootstrap conjunto por jornada
# ---------------------------------------------------------------------------

def _toy(n_days=40, per_day=60, seed=4):
    rng = np.random.default_rng(seed)
    day = np.repeat(np.arange(n_days), per_day)
    values = rng.normal(size=day.size) * (1 + (day % 3))
    grp = np.tile(np.array(["a", "b", "c"]), day.size // 3)
    return values, day, grp


def test_joint_bootstrap_equals_explicit_day_resampling():
    values, day, grp = _toy()
    n_days = int(day.max()) + 1
    groups = {"G": prepare_sorted_groups(values, day, grp), "GLOBAL": prepare_sorted_groups(values, day, np.full(day.size, "ALL", dtype=object))}
    point, reps = joint_day_block_bootstrap(groups, n_days, n_boot=7, seed=11)
    np.testing.assert_allclose(point["GLOBAL"]["ALL"], _explicit_stats(values), rtol=1e-12)

    rng = np.random.default_rng(11)
    rows_by_day = [np.flatnonzero(day == d) for d in range(n_days)]
    for b in range(7):
        sampled = rng.integers(0, n_days, size=n_days)
        idx = np.concatenate([rows_by_day[d] for d in sampled])
        np.testing.assert_allclose(reps["GLOBAL"]["ALL"][b], _explicit_stats(values[idx]), rtol=1e-10, atol=1e-12)
        for g in ("a", "b", "c"):
            sel = idx[grp[idx] == g]            # MISMAS jornadas para todos los grupos
            np.testing.assert_allclose(reps["G"][g][b], _explicit_stats(values[sel]), rtol=1e-10, atol=1e-12)


def test_joint_bootstrap_resamples_whole_days_never_partial_ones():
    """Si cada jornada es constante e igual a su indice, toda replica solo puede contener multiplos enteros del tamaño de jornada."""
    n_days, per_day = 30, 50
    day = np.repeat(np.arange(n_days), per_day)
    groups = {"GLOBAL": prepare_sorted_groups(day.astype(float), day, np.full(day.size, "ALL", dtype=object))}
    _, reps = joint_day_block_bootstrap(groups, n_days, n_boot=25, seed=5)
    assert (reps["GLOBAL"]["ALL"][:, -1] == n_days * per_day).all()      # n constante: jornadas completas


def test_joint_bootstrap_is_reproducible_and_seed_dependent():
    values, day, grp = _toy()
    groups = {"G": prepare_sorted_groups(values, day, grp)}
    _, r1 = joint_day_block_bootstrap(groups, 40, 15, seed=0)
    _, r2 = joint_day_block_bootstrap(groups, 40, 15, seed=0)
    _, r3 = joint_day_block_bootstrap(groups, 40, 15, seed=1)
    np.testing.assert_array_equal(r1["G"]["a"], r2["G"]["a"])
    assert not np.array_equal(r1["G"]["a"], r3["G"]["a"])


def test_prepare_sorted_groups_drops_missing_values_and_missing_labels():
    values = np.array([1.0, np.nan, 3.0, 2.0, np.inf, 5.0])
    labels = np.array(["a", "a", "a", None, "b", "b"], dtype=object)
    groups = prepare_sorted_groups(values, np.arange(6), labels)
    assert set(groups) == {"a", "b"}
    np.testing.assert_array_equal(groups["a"][0], [1.0, 3.0])
    np.testing.assert_array_equal(groups["b"][0], [5.0])
    float_labels = np.array([0.0, np.nan, 1.0, 1.0, 0.0, np.nan])
    assert set(prepare_sorted_groups(np.arange(6.0), np.arange(6), float_labels)) == {0.0, 1.0}


# ---------------------------------------------------------------------------
# C. Soporte de los cuantiles extremos
# ---------------------------------------------------------------------------

def test_tail_support_counts_observations_and_distinct_days():
    x = np.arange(1000.0)                 # ya ordenado
    days = np.arange(1000) // 100         # 10 jornadas de 100
    q_lo = np.quantile(x, 0.01)           # 9.99
    assert tail_support(x, days, 0.01, q_lo) == (10, 1)
    q_hi = np.quantile(x, 0.999)
    assert tail_support(x, days, 0.999, q_hi) == (1, 1)
    n_tail, n_days = tail_support(x, days, 0.25, np.quantile(x, 0.25))
    assert n_tail == 250 and n_days == 3
    assert tail_support(np.array([]), np.array([]), 0.01, np.nan) == (0, 0)


def test_tail_support_approximates_n_times_p():
    rng = np.random.default_rng(6)
    x = np.sort(rng.normal(size=84_663))                       # tamaño del segmento real mas pequeño
    n_tail, _ = tail_support(x, np.zeros(x.size, dtype=int), 0.999, np.quantile(x, 0.999))
    assert n_tail == pytest.approx(85, abs=2)                  # "decenas de puntos" (roadmap)
    assert support_label(n_tail) == "BAJO"


def test_support_label_reuses_existing_constants():
    assert [support_label(n) for n in (0, 19, 20, 199, 200, 10**6)] == ["NO_FIABLE", "NO_FIABLE", "BAJO", "BAJO", "OK", "OK"]


def test_quantile_table_flags_extreme_quantiles_of_small_groups():
    rng = np.random.default_rng(7)
    n_days, per_day = 50, 100
    day = np.repeat(np.arange(n_days), per_day)
    values = rng.normal(size=day.size)
    groups = {"GLOBAL": prepare_sorted_groups(values, day, np.full(day.size, "ALL", dtype=object))}
    point, reps = joint_day_block_bootstrap(groups, n_days, 30, 0)
    tab = build_quantile_table("x", "raw", groups, point, reps, {"GLOBAL": {"ALL": n_days}})
    by_level = tab.set_index("level")
    assert by_level.loc[0.001, "n_tail"] == 5 and by_level.loc[0.001, "support"] == "NO_FIABLE"   # 5.000 obs -> 5 puntos
    assert by_level.loc[0.01, "support"] == "BAJO" and by_level.loc[0.05, "support"] == "OK"
    assert (tab["q_ci_lo"] <= tab["q_ci_hi"]).all()
    # el IC del cuantil extremo es, en terminos relativos, mucho mas ancho que el del central
    width = (tab["q_ci_hi"] - tab["q_ci_lo"]).to_numpy()
    assert width[0] > width[2] and width[-1] > width[-3]
    # ningun cuantil extrapola mas alla de lo observado
    assert tab["q"].min() >= values.min() and tab["q"].max() <= values.max()
    assert tab["q_ci_lo"].min() >= values.min() and tab["q_ci_hi"].max() <= values.max()


# ---------------------------------------------------------------------------
# D. Decil CAUSAL de volatilidad -- pruebas anti-leakage
# ---------------------------------------------------------------------------

def _sigma_days(n_days=30, per_day=40, seed=8):
    rng = np.random.default_rng(seed)
    dates = np.repeat(np.array([datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(n_days)]), per_day)
    sigma = np.exp(rng.normal(size=dates.size) * 0.5) * 1e-4
    return sigma, dates


def test_causal_decile_matches_brute_force_definition():
    sigma, dates = _sigma_days()
    got = causal_volatility_decile(sigma, dates)
    uniq = sorted(set(dates))
    expected = np.full(sigma.size, np.nan)
    for d in uniq:
        hist = sigma[dates < d]
        if hist.size >= 200:
            for i in np.flatnonzero(dates == d):
                frac = np.sum(hist < sigma[i]) / hist.size
                expected[i] = min(np.floor(10 * frac), 9)
    np.testing.assert_array_equal(got, expected)
    assert np.isnan(got[dates < uniq[5]]).all()        # 5 jornadas * 40 = 200 -> hace falta esa historia
    assert np.isfinite(got[dates == uniq[5]]).all()
    assert set(np.unique(got[np.isfinite(got)])) <= set(range(10))


def test_causal_decile_ignores_same_day_and_future_information():
    """LEAKAGE: cambiar cualquier valor de la jornada d o posterior no puede alterar el estado de ninguna barra de jornadas <= d (salvo la propia barra)."""
    sigma, dates = _sigma_days()
    base = causal_volatility_decile(sigma, dates)
    uniq = sorted(set(dates))
    cut = uniq[15]
    future = sigma.copy()
    future[dates >= cut] *= 1e4                       # futuro radicalmente distinto
    new = causal_volatility_decile(future, dates)
    np.testing.assert_array_equal(base[dates < cut], new[dates < cut])
    # mismas fronteras para TODA la jornada `cut`: solo cambian sus barras porque cambio su propio valor,
    # y todas quedan en el decil mas alto respecto de la historia (que no cambio)
    assert (new[dates == cut] == 9).all()
    # truncar la muestra (no conocer el futuro) da exactamente los mismos estados
    truncated = causal_volatility_decile(sigma[dates < cut], dates[dates < cut])
    np.testing.assert_array_equal(base[dates < cut], truncated)


def test_full_sample_decile_of_tda10_does_leak_which_is_why_it_is_not_the_primary_state():
    """Documenta la decision N1: las fronteras de `assign_volatility_decile` (TDA-10) SI cambian estados pasados cuando cambia el futuro."""
    sigma, dates = _sigma_days()
    base = assign_volatility_decile(sigma)
    cut = sorted(set(dates))[15]
    future = sigma.copy()
    future[dates >= cut] *= 1e4
    new = assign_volatility_decile(future)
    assert not np.array_equal(base[dates < cut], new[dates < cut])


def test_causal_decile_respects_valid_mask_and_nonpositive_sigma():
    sigma, dates = _sigma_days()
    sigma[300] = np.nan
    sigma[301] = 0.0
    valid = np.ones(sigma.size, dtype=bool)
    valid[302] = False
    got = causal_volatility_decile(sigma, dates, valid=valid)
    assert np.isnan(got[[300, 301, 302]]).all()
    # una barra invalida tampoco cuenta como historia de las siguientes
    sigma2 = sigma.copy()
    sigma2[302] = 1e9
    np.testing.assert_array_equal(got, causal_volatility_decile(sigma2, dates, valid=valid))


def test_causal_decile_with_insufficient_history_is_all_nan():
    sigma, dates = _sigma_days(n_days=4, per_day=40)       # 160 < 200 de historia
    assert np.isnan(causal_volatility_decile(sigma, dates)).all()
    assert np.isnan(causal_volatility_decile(np.array([]), np.array([]))).size == 0


def test_causal_decile_groups_are_unequal_when_volatility_drifts():
    """Consecuencia declarada de la causalidad: con volatilidad creciente casi todo cae en el decil alto de la historia."""
    rng = np.random.default_rng(9)
    n_days, per_day = 60, 40
    dates = np.repeat(np.arange(n_days), per_day)
    sigma = np.exp(rng.normal(size=dates.size) * 0.1) * np.exp(dates / 10.0)
    got = causal_volatility_decile(sigma, dates)
    share_top = np.mean(got[np.isfinite(got)] == 9)
    assert share_top > 0.8
    full = assign_volatility_decile(sigma)
    assert np.mean(full[np.isfinite(full)] == 9) == pytest.approx(0.1, abs=0.01)


def test_verify_causal_decile_no_lookahead_passes_on_the_real_implementation():
    sigma, dates = _sigma_days(n_days=60)
    out = verify_causal_decile_no_lookahead(sigma, dates)
    assert out["passed"] is True and len(out["checks"]) >= 1
    assert any(c["some_later_day_changed"] for c in out["checks"])     # el estado SI reacciona a la historia


def test_verify_causal_decile_detects_a_leaky_implementation(monkeypatch):
    sigma, dates = _sigma_days(n_days=60)
    monkeypatch.setattr(mod, "causal_volatility_decile", lambda s, d, valid=None, n_deciles=10: assign_volatility_decile(s))
    assert mod.verify_causal_decile_no_lookahead(sigma, dates)["passed"] is False


# ---------------------------------------------------------------------------
# E. Estadistico de forma y su clasificacion
# ---------------------------------------------------------------------------

def test_profile_ratio_is_zero_for_identical_profiles_and_matches_formula():
    assert profile_ratio(np.array([2.3, 2.3, 2.3])) == 0.0
    assert profile_ratio(np.array([2.0, 3.0, 4.0])) == pytest.approx((4 - 2) / 3)
    assert np.isnan(profile_ratio(np.array([2.0, np.nan])))             # menos de 2 grupos
    mat = profile_ratio(np.array([[2.0, 3.0, 4.0], [1.0, 1.0, np.nan]]))
    assert mat[0] == pytest.approx(2 / 3) and mat[1] == 0.0


def test_classify_ratio_requires_point_and_whole_ci_in_the_same_zone():
    assert classify_ratio(0.10, 0.05, 0.20) == LABEL_OVERLAP
    assert classify_ratio(0.30, 0.20, 0.30) == LABEL_OVERLAP            # el umbral de TDA-10 es inclusivo (<=0.30)
    assert classify_ratio(0.29, 0.25, 0.34) == LABEL_INCONCLUSIVE       # el IC cruza 0.30
    assert classify_ratio(0.45, 0.35, 0.55) == LABEL_INTERMEDIATE
    assert classify_ratio(0.45, 0.35, 0.65) == LABEL_INCONCLUSIVE
    assert classify_ratio(0.80, 0.60, 1.20) == LABEL_DIVERGENT
    assert classify_ratio(0.80, 0.55, 1.20) == LABEL_INCONCLUSIVE
    assert classify_ratio(np.nan, 0.1, 0.2) == LABEL_INCONCLUSIVE
    assert classify_ratio(0.1, np.nan, np.nan) == LABEL_INCONCLUSIVE    # sin IC no hay etiqueta firme


# ---------------------------------------------------------------------------
# F. Escala vs forma -- casos sinteticos de respuesta conocida
# ---------------------------------------------------------------------------

def _shape_case(sampler_by_group: dict, n_days=120, per_day_group=260, n_boot=60, seed=10):
    """Construye una serie con varios grupos (todos presentes cada jornada) y devuelve la tabla de comparacion de forma."""
    rng = np.random.default_rng(seed)
    vals, days, labs = [], [], []
    for g, sampler in sampler_by_group.items():
        vals.append(sampler(rng, n_days * per_day_group))
        days.append(np.repeat(np.arange(n_days), per_day_group))
        labs.append(np.full(n_days * per_day_group, g, dtype=object))
    values, day, lab = np.concatenate(vals), np.concatenate(days), np.concatenate(labs)
    groups = {"SEGMENT": prepare_sorted_groups(values, day, lab), "GLOBAL": prepare_sorted_groups(values, day, np.full(day.size, "ALL", dtype=object))}
    point, reps = joint_day_block_bootstrap(groups, n_days, n_boot, 0)
    return build_shape_comparison_table("z", "raw", groups, point, reps, is_standardized=True), (groups, point, reps)


def _label(tab: pd.DataFrame, norm: str, pair=VERDICT_LEVEL_PAIR) -> str:
    row = tab.loc[(tab["grouping"] == "SEGMENT") & (tab["normalization"] == norm) & (tab["level_lo"] == pair[0])]
    return str(row["label"].iloc[0])


def test_pure_scale_family_is_classified_as_overlapping():
    """SOLO ESCALA sintetico: misma forma (normal), escalas 1x / 5x / 25x -> los perfiles normalizados se superponen."""
    tab, _ = _shape_case({s: (lambda rng, n, s=s: rng.normal(size=n) * s) for s in (1.0, 5.0, 25.0)})
    for norm in NORMALIZATIONS:
        row = tab.loc[(tab["normalization"] == norm) & (tab["level_lo"] == 0.01)].iloc[0]
        assert row["ratio"] < 0.05 and row["label"] == LABEL_OVERLAP and row["n_groups"] == 3
    # sin normalizar la diferencia de escala es enorme: la normalizacion es lo que la retira
    assert tab.loc[(tab["normalization"] == "none") & (tab["level_lo"] == 0.01), "ratio"].iloc[0] > 2.0
    assert set(tab["level_lo"]) == {p[0] for p in SYMMETRIC_LEVEL_PAIRS}
    assert (tab.loc[tab["level_lo"] != 0.01, "label"] == "").all()       # solo el nivel de TDA-10 lleva etiqueta


def test_same_scale_different_shape_is_classified_as_divergent():
    """ESCALA Y FORMA sintetico: misma desviacion, forma distinta (normal vs bimodal de dos picos) -> divergen aunque la escala coincida."""
    def bimodal(rng, n):
        return np.where(rng.random(n) < 0.5, -1.0, 1.0) + rng.normal(size=n) * 0.05

    tab, (groups, point, _) = _shape_case({"normal": lambda rng, n: rng.normal(size=n), "bimodal": bimodal})
    assert point["SEGMENT"]["normal"][K] == pytest.approx(point["SEGMENT"]["bimodal"][K], rel=0.02)   # MISMA escala
    for norm in NORMALIZATIONS:
        row = tab.loc[(tab["normalization"] == norm) & (tab["level_lo"] == 0.01)].iloc[0]
        assert row["ratio"] > PROFILE_STABILITY_FORM_THRESHOLD and row["label"] == LABEL_DIVERGENT
        assert row["group_min"] == "bimodal" and row["group_max"] == "normal"


def test_heavy_tail_only_beyond_the_1pct_level_is_visible_in_the_deep_tail_rows():
    """Limitacion declarada: una diferencia de forma confinada a la cola profunda apenas mueve el nivel 1/99 pero si el 0.1/99.9."""
    def contaminated(rng, n):
        x = rng.normal(size=n)
        hit = rng.random(n) < 0.004
        x[hit] *= 8.0
        return x

    tab, _ = _shape_case({"normal": lambda rng, n: rng.normal(size=n), "contaminated": contaminated}, per_day_group=2_000, n_boot=30)
    trimmed = tab.loc[tab["normalization"] == "std_trimmed"].set_index("level_lo")
    assert trimmed.loc[0.001, "ratio"] > 3 * trimmed.loc[0.01, "ratio"]


def test_groups_with_low_tail_support_are_excluded_and_too_few_groups_is_inconclusive():
    """Grupos insuficientes: un grupo pequeño no participa; con menos de 2 grupos utilizables no hay etiqueta firme."""
    rng = np.random.default_rng(12)
    n_days = 100
    big_day = np.repeat(np.arange(n_days), 300)                 # 30.000 obs -> cola 1% = 300 (OK)
    small_day = np.repeat(np.arange(n_days), 20)                # 2.000 obs -> cola 1% = 20 (BAJO)
    values = np.concatenate([rng.normal(size=big_day.size), rng.normal(size=small_day.size) * 3])
    day = np.concatenate([big_day, small_day])
    lab = np.concatenate([np.full(big_day.size, "big", dtype=object), np.full(small_day.size, "small", dtype=object)])
    groups = {"SEGMENT": prepare_sorted_groups(values, day, lab)}
    point, reps = joint_day_block_bootstrap(groups, n_days, 20, 0)
    tab = build_shape_comparison_table("z", "raw", groups, point, reps, True)
    row = tab.loc[(tab["normalization"] == "std") & (tab["level_lo"] == 0.01)].iloc[0]
    assert row["n_groups"] == 1 and row["groups_excluded_low_support"] == "small"
    assert np.isnan(row["ratio"]) and row["label"] == LABEL_INCONCLUSIVE
    mid = tab.loc[(tab["normalization"] == "std") & (tab["level_lo"] == 0.25)].iloc[0]
    assert mid["n_groups"] == 2                                  # en el centro de la distribucion si hay soporte


def test_zero_variance_group_does_not_crash_and_is_not_given_a_normalized_profile():
    rng = np.random.default_rng(13)
    n_days, per = 100, 300
    day = np.tile(np.repeat(np.arange(n_days), per), 3)
    values = np.concatenate([rng.normal(size=n_days * per), rng.normal(size=n_days * per) * 2, np.zeros(n_days * per)])
    lab = np.repeat(np.array(["a", "b", "flat"], dtype=object), n_days * per)
    groups = {"SEGMENT": prepare_sorted_groups(values, day, lab)}
    point, reps = joint_day_block_bootstrap(groups, n_days, 20, 0)
    assert point["SEGMENT"]["flat"][K] == 0.0
    tab = build_shape_comparison_table("z", "raw", groups, point, reps, True)
    row = tab.loc[(tab["normalization"] == "std") & (tab["level_lo"] == 0.01)].iloc[0]
    assert np.isfinite(row["ratio"]) and row["ratio"] < 0.1      # se compara a/b; "flat" (0/0) no fabrica un numero
    qt = build_quantile_table("z", "raw", groups, point, reps, {"SEGMENT": {g: n_days for g in groups["SEGMENT"]}})
    assert qt.loc[qt["group"] == "flat", "q_over_std"].isna().all()
    asym = build_asymmetry_table("z", "raw", groups, point, reps)
    assert asym.loc[asym["group"] == "flat", "half_width_over_std"].isna().all()


def test_asymmetry_table_sign_and_ci():
    def left_skewed(rng, n):
        x = rng.normal(size=n)
        return np.where(x < 0, 1.6 * x, x)

    _, (groups, point, reps) = _shape_case({"sym": lambda rng, n: rng.normal(size=n), "left": left_skewed})
    a = build_asymmetry_table("z", "raw", groups, point, reps)
    a = a.loc[(a["grouping"] == "SEGMENT") & (a["level_lo"] == 0.01)].set_index("group")
    assert a.loc["left", "asymmetry"] == pytest.approx(2 * (1.6 - 1) / (1.6 + 1), abs=0.03) and a.loc["left", "asymmetry_ci_lo"] > 0
    assert a.loc["sym", "asymmetry_ci_lo"] < 0 < a.loc["sym", "asymmetry_ci_hi"]
    assert a.loc["sym", "half_width_over_std"] == pytest.approx(2.326, abs=0.03)


# ---------------------------------------------------------------------------
# G. Regla de veredicto TH26
# ---------------------------------------------------------------------------

def _fake_shape(labels: dict, series="z") -> pd.DataFrame:
    """`labels[agrupacion] = (etiqueta_std, etiqueta_std_trimmed)`."""
    rows = []
    for gk, (l_std, l_trim) in labels.items():
        for norm, lab in (("std", l_std), ("std_trimmed", l_trim)):
            rows.append({"series": series, "grouping": gk, "level_lo": 0.01, "level_hi": 0.99, "normalization": norm,
                         "ratio": 0.1, "ratio_ci_lo": 0.05, "ratio_ci_hi": 0.2, "label": lab})
    return pd.DataFrame(rows)


def _sens(verdicts: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"point_verdict": verdicts})


O, I, D, N = LABEL_OVERLAP, LABEL_INTERMEDIATE, LABEL_DIVERGENT, LABEL_INCONCLUSIVE


@pytest.mark.parametrize("labels,expected", [
    ({"SEGMENT": (O, O), "VOL_DECILE_CAUSAL": (O, O), "YEAR": (O, O)}, VERDICT_SCALE_ONLY),
    ({"SEGMENT": (D, D), "VOL_DECILE_CAUSAL": (O, O), "YEAR": (O, O)}, VERDICT_SCALE_AND_SHAPE),
    ({"SEGMENT": (I, I), "VOL_DECILE_CAUSAL": (O, O), "YEAR": (O, O)}, VERDICT_INCONCLUSIVE),
    ({"SEGMENT": (N, O), "VOL_DECILE_CAUSAL": (O, O), "YEAR": (O, O)}, VERDICT_INCONCLUSIVE),   # IC cruza un umbral
    ({"SEGMENT": (D, O), "VOL_DECILE_CAUSAL": (O, O), "YEAR": (O, O)}, VERDICT_INCONCLUSIVE),   # normalizaciones discrepan
    ({"SEGMENT": (O, O), "VOL_DECILE_CAUSAL": (O, O)}, VERDICT_INCONCLUSIVE),                   # falta una agrupacion
])
def test_decide_th26_rule(labels, expected):
    out = decide_th26(_fake_shape(labels), _sens([expected] * 6), "z")
    assert out["verdict_before_robustness"] == expected and out["verdict"] == expected
    assert set(out["labels_by_grouping"]) == set(VERDICT_GROUPINGS)


def test_decide_th26_never_reads_absence_of_divergence_as_proof_of_equality():
    """Ninguna agrupacion diverge, pero una no es concluyente -> NO se declara SOLO_ESCALA."""
    out = decide_th26(_fake_shape({"SEGMENT": (N, N), "VOL_DECILE_CAUSAL": (O, O), "YEAR": (O, O)}), _sens([VERDICT_SCALE_ONLY] * 6), "z")
    assert out["verdict"] == VERDICT_INCONCLUSIVE


def test_decide_th26_is_downgraded_when_not_robust_to_the_volatility_estimator():
    shape = _fake_shape({gk: (O, O) for gk in VERDICT_GROUPINGS})
    robust = decide_th26(shape, _sens([VERDICT_SCALE_ONLY] * 5 + [VERDICT_INCONCLUSIVE]), "z")        # 5/6 >= 0.75
    assert robust["verdict"] == VERDICT_SCALE_ONLY and robust["robust_to_estimator"] is True
    fragile = decide_th26(shape, _sens([VERDICT_SCALE_ONLY] * 4 + [VERDICT_INCONCLUSIVE] * 2), "z")   # 4/6 < 0.75
    assert fragile["verdict_before_robustness"] == VERDICT_SCALE_ONLY
    assert fragile["verdict"] == VERDICT_INCONCLUSIVE and fragile["robust_to_estimator"] is False
    assert fragile["estimator_agreement_fraction"] == pytest.approx(4 / 6)


def test_decide_th26_uses_only_the_declared_series_and_the_declared_level():
    shape = pd.concat([
        _fake_shape({gk: (O, O) for gk in VERDICT_GROUPINGS}, series="z"),
        _fake_shape({gk: (D, D) for gk in VERDICT_GROUPINGS}, series="r_1m"),          # r_t es contexto: no decide
    ], ignore_index=True)
    deep = _fake_shape({gk: (D, D) for gk in VERDICT_GROUPINGS}, series="z").assign(level_lo=0.001, level_hi=0.999)
    out = decide_th26(pd.concat([shape, deep], ignore_index=True), _sens([VERDICT_SCALE_ONLY] * 6), "z")
    assert out["verdict"] == VERDICT_SCALE_ONLY and out["verdict_series"] == "z"


# ---------------------------------------------------------------------------
# H. Implicacion para STOP-13 (sin ejecutar TDA-13)
# ---------------------------------------------------------------------------

def test_stop13_implication_for_each_verdict():
    scale = decide_stop13_implication({"verdict": VERDICT_SCALE_ONLY})
    assert scale["stop13_activated_by_tda12"] is True and scale["evt_entry_criterion_1_met"] is False
    both = decide_stop13_implication({"verdict": VERDICT_SCALE_AND_SHAPE})
    assert both["stop13_activated_by_tda12"] is False and both["evt_entry_criterion_1_met"] is True
    assert "CONDICIONADA" in both["text"]                       # EVT sigue dependiendo de los criterios 2 y 3
    unk = decide_stop13_implication({"verdict": VERDICT_INCONCLUSIVE})
    assert unk["stop13_activated_by_tda12"] is False and unk["evt_entry_criterion_1_met"] is False
    assert "confirmacion explicita" in unk["text"]
    for out in (scale, both, unk):
        assert "parte A" in out["text"]                         # el inventario de extremos es obligatorio siempre


# ---------------------------------------------------------------------------
# J. Extremo a extremo
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path, research_files, holdout_files=None, boundary_utc="2099-01-01 00:00:00", n_boot=12):
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_yaml = {
        "instrument": "MNQ",
        "raw_data": {
            "dir": "data/raw/mnq", "file_pattern": "*.Last.txt", "separator": ";",
            "columns": ["timestamp", "open", "high", "low", "close", "volume"],
            "timestamp_format": "%Y%m%d %H%M%S", "timestamp_raw_timezone": "UTC",
        },
        "instrument_spec": {"tick_size": TICK, "tick_size_source": "test"},
        "holdout": {"boundary_utc": boundary_utc, "boundary_source": "test", "research_files": research_files, "holdout_files": holdout_files or []},
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
            "max_breakpoints": 6, "min_spacing_minutes": 60, "year_stability_tolerance_minutes": 15, "year_stability_min_years": 3,
            "report_name": "TDA06_perfil_intradia_calendario.md", "minute_global_csv": "TDA06_perfil_minuto_global.csv",
            "minute_by_year_csv": "TDA06_perfil_minuto_por_anio.csv", "weekday_csv": "TDA06_perfil_dia_semana.csv",
            "segmentation_csv": "TDA06_segmentacion_propuesta.csv", "calibration_csv": "TDA06_calibracion_null.csv",
            "profile_png": "TDA06_perfil_intradia.png", "s_m_parquet_name": "tda06_s_m.parquet", "r_tilde_parquet_name": "tda06_r_tilde.parquet",
        },
        "tda12": {
            "output_dir_reports": "reports/mnq", "n_boot": n_boot,
            "report_name": "TDA12_cuantiles_condicionales.md", "quantiles_csv": "TDA12_cuantiles_condicionales.csv",
            "shape_csv": "TDA12_escala_vs_forma_perfiles.csv", "asymmetry_csv": "TDA12_asimetria_colas.csv",
            "sensitivity_csv": "TDA12_sensibilidad_estimador.csv", "causality_csv": "TDA12_verificacion_causalidad.csv",
            "decile_sizes_csv": "TDA12_tamano_deciles_volatilidad.csv", "profiles_png": "TDA12_perfiles_cuantiles_normalizados.png",
            "half_width_png": "TDA12_semiamplitud_normalizada_por_grupo.png",
            "block_sensitivity_csv": "TDA12_sensibilidad_longitud_bloque.csv", "readings_csv": "TDA12_lecturas_alternativas_veredicto.csv",
        },
    }
    config_path = config_dir / "mnq_snapshot.yaml"
    config_path.write_text(yaml.safe_dump(config_yaml), encoding="utf-8")
    return load_config(config_path)


def _prepare_fixture(config, n_days=150, bars_per_day=420, seed=0, start=datetime.date(2023, 9, 4)):
    """Dos años calendario, varias sesiones; volatilidad con regimenes por jornada (para que existan estados de volatilidad distintos)."""
    rng = np.random.default_rng(seed)
    ts_list, date_list, r_list, valid_list, close_list = [], [], [], [], []
    price, day_i, added = 15000.0, 0, 0
    while added < n_days:
        date = start + datetime.timedelta(days=day_i)
        day_i += 1
        if date.weekday() >= 5:
            continue
        base = pd.Timestamp(date) + pd.Timedelta(hours=9, minutes=30)
        level = 4e-4 * np.exp(0.6 * np.sin(added / 7.0))
        rets = rng.normal(size=bars_per_day) * level
        for b in range(bars_per_day):
            prev = price
            price = round(price * (1.0 + rets[b]) / TICK) * TICK
            ts_list.append(base + pd.Timedelta(minutes=b))
            date_list.append(date)
            close_list.append(price)
            r_list.append(float(np.log(price / prev)) if b > 0 else np.nan)
            valid_list.append(b > 0)             # la primera barra de cada jornada no tiene retorno valido (no cruza jornadas)
        added += 1
    close = np.asarray(close_list)
    variables = pd.DataFrame({
        "timestamp": ts_list, "source_file": "research.txt", "contract": "H24", "trading_date": date_list, "segment_id": 1,
        "open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 10, "r_1m": r_list,
    })
    variables["abs_r_1m"] = variables["r_1m"].abs()
    variables["r2_1m"] = variables["r_1m"] ** 2
    variables["log_hl"] = np.log(variables["high"] / variables["low"])
    validity = pd.DataFrame({
        "timestamp": ts_list, "r_1m_valid": valid_list,
        "invalid_reason": ["VALID" if v else "TRADING_DATE_BOUNDARY" for v in valid_list],
    })
    config.interim_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    variables.to_parquet(config.tda04_variables_parquet_path, index=False)
    validity.to_parquet(config.tda04_validity_mask_parquet_path, index=False)

    cal = attach_calendar_fields(variables, validity)
    m0 = int(cal["minute_of_day"].min())
    cutoffs = [m0 + bars_per_day // 3, m0 + 2 * bars_per_day // 3]            # 3 segmentos de igual duracion
    s_m = pd.Series(1.0, index=cal.index)
    pd.DataFrame({
        "timestamp": cal["timestamp"], "source_file": cal["source_file"], "contract": cal["contract"], "trading_date": cal["trading_date"],
        "minute_of_day": cal["minute_of_day"], "weekday": cal["weekday"], "r_1m": cal["r_1m"], "r_1m_valid": cal["r_1m_valid"],
        "s_m": s_m, "r_tilde": np.where(cal["r_1m_valid"], cal["r_1m"] / s_m, np.nan), "label": "RETROSPECTIVO",
    }).to_parquet(config.tda06_r_tilde_parquet_path, index=False)
    pd.DataFrame([{"minute_of_day": c, "n_years_supporting": 5, "years_supporting": "2020,2021,2022,2023,2024", "stable": True} for c in cutoffs]).to_csv(
        config.tda06_segmentation_csv_path, index=False)
    return variables, validity


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tda12_e2e")
    config = _make_config(tmp, research_files=["research.txt"], holdout_files=["holdout.txt"])
    variables, validity = _prepare_fixture(config)
    result = run_tda12_analysis(config, verbose=False)
    return config, variables, validity, result


def test_e2e_result_structure(e2e):
    _, variables, validity, result = e2e
    assert result.n_population == int(validity["r_1m_valid"].sum())
    assert result.n_days == variables["trading_date"].nunique()
    q = result.quantile_table
    assert set(q["series"]) == {"r_1m", "z_ewma_60_raw", "r_tilde", "z_ewma_60_clock_adjusted"}
    assert set(q["grouping"]) == set(mod.GROUPINGS)
    assert set(q["level"]) == set(QUANTILE_LEVELS)
    assert set(q.loc[q["grouping"] == "YEAR", "group"]) == {2023, 2024}
    assert q.loc[q["grouping"] == "SEGMENT", "group"].nunique() == 3
    for col in ("n", "n_days", "q", "q_ci_lo", "q_ci_hi", "q_over_std", "n_tail", "n_days_tail", "support"):
        assert col in q.columns
    assert set(q["support"]) <= {"OK", "BAJO", "NO_FIABLE"}
    assert result.th26["verdict"] in (VERDICT_SCALE_ONLY, VERDICT_SCALE_AND_SHAPE, VERDICT_INCONCLUSIVE)
    assert result.th26["verdict_series"] == "z_ewma_60_raw"
    assert len(result.estimator_sensitivity) == 6 and result.estimator_sensitivity["is_primary"].sum() == 1
    assert isinstance(result.stop13["stop13_activated_by_tda12"], bool)
    assert len(result.stage_timings) >= mod.ANALYSIS_STAGES


def test_e2e_group_sizes_add_up_and_n_is_always_reported(e2e):
    _, _, _, result = e2e
    q = result.quantile_table
    one = q.loc[(q["series"] == "r_1m") & (q["level"] == 0.5)]
    total = int(one.loc[one["grouping"] == "GLOBAL", "n"].iloc[0])
    assert total == result.n_population
    assert int(one.loc[one["grouping"] == "SEGMENT", "n"].sum()) == total
    assert int(one.loc[one["grouping"] == "YEAR", "n"].sum()) == total
    sizes = result.decile_sizes
    for gk in ("VOL_DECILE_CAUSAL", "VOL_DECILE_FULLSAMPLE"):
        assert int(sizes.loc[(sizes["block"] == "raw") & (sizes["grouping"] == gk), "n"].sum()) == total
    assert (q["n"] > 0).all() and q["n_tail"].notna().all()


def test_e2e_z_is_exactly_the_tda10_definition(e2e):
    """Compatibilidad: TDA-12 no redefine z_t -- es la salida de las funciones de TDA-10 para su estimador primario."""
    _, variables, validity, result = e2e
    r = variables.loc[validity["r_1m_valid"].to_numpy(), "r_1m"].to_numpy(dtype=float)
    fam, par, _ = PRIMARY_ESTIMATOR
    z = standardize_return(r, build_sigma_hat(r, fam, par), burn_in_length(fam, par))
    zf = np.sort(z[np.isfinite(z)])
    row = result.quantile_table.loc[(result.quantile_table["series"] == "z_ewma_60_raw") & (result.quantile_table["grouping"] == "GLOBAL")]
    np.testing.assert_allclose(row.sort_values("level")["q"].to_numpy(), np.quantile(zf, QUANTILE_LEVELS), rtol=1e-12)
    assert int(row["n"].iloc[0]) == zf.size
    info = result.series_info.set_index("series")
    assert info.loc["z_ewma_60_raw", "n_excluded_nonfinite"] == burn_in_length(fam, par)
    assert info.loc["r_1m", "causal_class"] == "CAUSAL" and info.loc["r_tilde", "causal_class"] == "RETROSPECTIVO"


def test_e2e_unconditional_quantiles_match_direct_computation(e2e):
    _, variables, validity, result = e2e
    r = variables.loc[validity["r_1m_valid"].to_numpy(), "r_1m"].to_numpy(dtype=float)
    q = result.quantile_table
    row = q.loc[(q["series"] == "r_1m") & (q["grouping"] == "GLOBAL")].sort_values("level")
    np.testing.assert_allclose(row["q"].to_numpy(), np.quantile(r, QUANTILE_LEVELS), rtol=1e-12)
    assert row["std"].iloc[0] == pytest.approx(r.std(ddof=1), rel=1e-12)
    assert (row["q_ci_lo"] <= row["q_ci_hi"]).all()


def test_e2e_is_reproducible(e2e):
    config, _, _, result = e2e
    again = run_tda12_analysis(config, verbose=False)
    pd.testing.assert_frame_equal(result.quantile_table, again.quantile_table)
    pd.testing.assert_frame_equal(result.shape_table, again.shape_table)
    pd.testing.assert_frame_equal(result.estimator_sensitivity, again.estimator_sensitivity)
    for key in ("verdict", "verdict_before_robustness", "labels_by_grouping", "robust_to_estimator"):
        assert result.th26[key] == again.th26[key]
    pd.testing.assert_frame_equal(pd.DataFrame(result.th26["detail"]), pd.DataFrame(again.th26["detail"]))   # NaN-seguro


def test_e2e_causality_checks_ran_and_passed(e2e):
    _, _, _, result = e2e
    assert bool(result.causality_sigma["passed"].all())
    assert result.causality_decile["passed"] is True and len(result.causality_decile["checks"]) >= 1


def test_e2e_future_data_does_not_change_past_volatility_states(tmp_path):
    """LEAKAGE de extremo a extremo: truncar la muestra (no disponer del futuro) deja identicos los estados causales y z_t del pasado."""
    cfg_full = _make_config(tmp_path / "full", research_files=["research.txt"])
    variables, validity = _prepare_fixture(cfg_full, n_days=40, bars_per_day=200)
    r = variables.loc[validity["r_1m_valid"].to_numpy(), "r_1m"].to_numpy(dtype=float)
    dates = variables.loc[validity["r_1m_valid"].to_numpy(), "trading_date"].to_numpy()
    fam, par, _ = PRIMARY_ESTIMATOR
    cut = int(r.size * 0.6)
    cut = int(np.flatnonzero(dates == dates[cut])[0])                    # corte en frontera de jornada
    sig_full = build_sigma_hat(r, fam, par)
    sig_cut = build_sigma_hat(r[:cut], fam, par)
    np.testing.assert_allclose(sig_full[:cut], sig_cut, rtol=1e-12, equal_nan=True)
    z_full = standardize_return(r, sig_full, burn_in_length(fam, par))
    z_cut = standardize_return(r[:cut], sig_cut, burn_in_length(fam, par))
    np.testing.assert_allclose(z_full[:cut], z_cut, rtol=1e-12, equal_nan=True)
    np.testing.assert_array_equal(
        causal_volatility_decile(sig_full, dates, valid=np.isfinite(z_full))[:cut],
        causal_volatility_decile(sig_cut, dates[:cut], valid=np.isfinite(z_cut)),
    )


def test_run_stops_if_sigma_hat_fails_the_lookahead_check(tmp_path, monkeypatch):
    config = _make_config(tmp_path, research_files=["research.txt"])
    _prepare_fixture(config, n_days=12, bars_per_day=120)

    def _leaky(*a, **k):
        raise LookaheadLeakageError("fuga sintetica")

    monkeypatch.setattr(mod, "run_causality_checks", _leaky)
    with pytest.raises(LookaheadLeakageError):
        run_tda12_analysis(config, verbose=False)


def test_run_stops_if_causal_decile_fails_the_lookahead_check(tmp_path, monkeypatch):
    config = _make_config(tmp_path, research_files=["research.txt"])
    _prepare_fixture(config, n_days=12, bars_per_day=120)
    monkeypatch.setattr(mod, "verify_causal_decile_no_lookahead", lambda *a, **k: {"checks": [], "passed": False})
    with pytest.raises(CausalDecileLeakageError):
        run_tda12_analysis(config, verbose=False)


def test_run_raises_when_research_and_holdout_overlap(tmp_path):
    config = _make_config(tmp_path, research_files=["a.txt", "b.txt"], holdout_files=["b.txt"])
    with pytest.raises(HoldoutIsolationError):
        run_tda12_analysis(config, verbose=False)


def test_run_raises_when_research_data_reaches_the_holdout_boundary(tmp_path):
    config = _make_config(tmp_path, research_files=["research.txt"], boundary_utc="2023-09-10 00:00:00")
    _prepare_fixture(config, n_days=12, bars_per_day=120)
    with pytest.raises(HoldoutIsolationError):
        run_tda12_analysis(config, verbose=False)


def test_run_never_opens_raw_or_holdout_files(tmp_path, monkeypatch):
    config = _make_config(tmp_path, research_files=["research.txt"], holdout_files=["holdout.txt"])
    _prepare_fixture(config, n_days=12, bars_per_day=120)
    raw_dir = Path(config.raw_dir)
    opened = []
    real_open = open

    def _spy(file, *a, **k):
        opened.append(str(file))
        return real_open(file, *a, **k)

    monkeypatch.setattr("builtins.open", _spy)
    import ohlcv_dataroad.ingest.parsing as parsing_mod

    if hasattr(parsing_mod, "parse_raw_file"):
        monkeypatch.setattr(parsing_mod, "parse_raw_file", lambda *a, **k: (_ for _ in ()).throw(AssertionError("TDA-12 no debe parsear crudos")))
    run_tda12_analysis(config, verbose=False)
    assert not raw_dir.exists()                                   # ni siquiera existe: nada pudo leerse de ahi
    assert not any("holdout" in p or str(raw_dir) in p for p in opened)


def test_small_sample_run_completes_and_does_not_force_a_verdict(tmp_path):
    """Muestra pequeña: los grupos no alcanzan soporte de cola -> NO_CONCLUYENTE, nunca un veredicto fabricado."""
    config = _make_config(tmp_path, research_files=["research.txt"], n_boot=10)
    _prepare_fixture(config, n_days=12, bars_per_day=120)
    result = run_tda12_analysis(config, verbose=False)
    assert result.th26["verdict"] == VERDICT_INCONCLUSIVE
    assert result.stop13["stop13_activated_by_tda12"] is False
    q = result.quantile_table
    extreme = q.loc[q["level"].isin([0.001, 0.999]) & (q["grouping"] != "GLOBAL")]
    assert (extreme["support"] != "OK").all()


def test_missing_input_artifact_fails_cleanly(tmp_path):
    config = _make_config(tmp_path, research_files=["research.txt"])
    with pytest.raises(FileNotFoundError):
        run_tda12_analysis(config, verbose=False)


def test_progress_is_printed_only_when_verbose(tmp_path, capsys):
    config = _make_config(tmp_path, research_files=["research.txt"], n_boot=5)
    _prepare_fixture(config, n_days=12, bars_per_day=120)
    run_tda12_analysis(config, verbose=False)
    assert capsys.readouterr().out == ""
    run_tda12_analysis(config, verbose=True)
    out = capsys.readouterr().out
    assert "[TDA12 1/6]" in out and "[TDA12 6/6]" in out


# ---------------------------------------------------------------------------
# K. Artefactos e informe
# ---------------------------------------------------------------------------

def test_persist_artifacts_writes_exactly_the_declared_paths(e2e):
    from ohlcv_dataroad.ingest.run_tda12 import ARTIFACT_PATH_ATTRS, persist_artifacts

    config, _, _, result = e2e
    before = set(config.reports_dir.iterdir())
    written = persist_artifacts(result, config, t0=0.0, run_command="test")
    expected = {getattr(config, a) for a in ARTIFACT_PATH_ATTRS}
    assert set(written) == expected and len(expected) == 11
    for p in expected:
        assert p.exists() and p.stat().st_size > 0
    new_files = set(config.reports_dir.iterdir()) - before
    assert new_files <= expected                                   # nada fuera de lo declarado
    assert all(p.name.startswith("TDA12_") for p in expected)


def test_persisted_csvs_roundtrip_and_report_content(e2e):
    from ohlcv_dataroad.ingest.run_tda12 import persist_artifacts

    config, _, _, result = e2e
    persist_artifacts(result, config, t0=0.0, run_command="python -m ohlcv_dataroad.ingest.run_tda12")
    q = pd.read_csv(config.tda12_quantiles_csv_path)
    assert len(q) == len(result.quantile_table) and {"n", "n_tail", "n_days_tail", "support", "q_ci_lo", "q_ci_hi"} <= set(q.columns)
    shape = pd.read_csv(config.tda12_shape_csv_path)
    assert {"ratio", "ratio_ci_lo", "ratio_ci_hi", "label", "n_groups", "groups_excluded_low_support"} <= set(shape.columns)
    caus = pd.read_csv(config.tda12_causality_csv_path)
    assert {"sigma_hat_no_lookahead (TDA-10)", "causal_volatility_decile_no_lookahead"} <= set(caus["check"])
    assert caus["passed"].astype(bool).all()
    sens = pd.read_csv(config.tda12_sensitivity_csv_path)
    assert len(sens) == 6 and "verdict_final" in sens.columns

    text = config.tda12_report_path.read_text(encoding="utf-8")
    for needle in (
        "TH26", "Versión: COMPLETA", "bootstrap por bloques", "jornadas completas", "Decisiones metodológicas NUEVAS",
        "VOL_DECILE_CAUSAL", "n_cola", "STOP-13", "TDA-13 parte A", "Limitaciones", "Modo sencillo", "holdout",
        "NUNCA se edita a mano",
    ):
        assert needle in text, needle
    assert ("SÓLO ESCALA" in text) or ("ESCALA Y FORMA" in text)        # el informe se pronuncia sobre los dos resultados previstos
    assert "p-valor" not in text and "significativ" not in text         # G5: magnitud + IC, nunca "significativo"


def test_report_states_the_verdict_it_computed(e2e):
    from ohlcv_dataroad.ingest.run_tda12 import VERDICT_DISPLAY, render_report

    config, _, _, result = e2e
    text = render_report(result, config, 1.0, "test")
    assert f"**Veredicto: `{VERDICT_DISPLAY[result.th26['verdict']]}`**" in text
    activated = "SÍ" if result.stop13["stop13_activated_by_tda12"] else "NO"
    assert f"`STOP-13` activado por TDA-12: **{activated}**" in text


def test_tda12_does_not_modify_earlier_stage_modules():
    """TDA-12 importa TDA-07/TDA-10 sin redefinirlos: sus funciones son los MISMOS objetos."""
    import ohlcv_dataroad.ingest.tda07_marginal_distribution as t07
    import ohlcv_dataroad.ingest.tda10_scale_vs_shape as t10

    assert mod.build_sigma_hat is t10.build_sigma_hat and mod.standardize_return is t10.standardize_return
    assert mod.run_causality_checks is t10.run_causality_checks and mod.assign_volatility_decile is t10.assign_volatility_decile
    assert mod.build_r1m_population is t07.build_r1m_population and mod.TRIM_FRACTION_EACH_TAIL == t07.TRIM_FRACTION_EACH_TAIL


# ---------------------------------------------------------------------------
# L. AUDITORIA CORRECTIVA -- tests adversariales
# ---------------------------------------------------------------------------

from ohlcv_dataroad.ingest.tda12_conditional_quantiles import (  # noqa: E402
    BLOCK_LENGTH_SENSITIVITY_DAYS,
    build_alternative_readings,
    build_block_length_sensitivity,
    classify_groups,
    evaluate_one_sided_proposal,
)


def _mirror_case(n_days=120, per=400, n_boot=80):
    """Exponencial centrada y su reflejo: MISMA desviacion y MISMA semi-amplitud 1%-99%, asimetrias opuestas."""
    rng = np.random.default_rng(0)
    day = np.tile(np.repeat(np.arange(n_days), per), 2)
    values = np.concatenate([rng.exponential(size=n_days * per) - 1.0, -(rng.exponential(size=n_days * per) - 1.0)])
    lab = np.repeat(np.array(["exp", "exp_reflejada"], dtype=object), n_days * per)
    groups = {"SEGMENT": prepare_sorted_groups(values, day, lab)}
    point, reps = joint_day_block_bootstrap(groups, n_days, n_boot, 0)
    return groups, point, reps


def test_half_width_statistic_is_blind_to_asymmetry_and_this_is_now_exposed():
    """ADVERSARIAL (hallazgo A3): el estadistico del veredicto declara SUPERPUESTO dos distribuciones reflejadas; las columnas por cola y el rango de asimetria lo delatan."""
    groups, point, reps = _mirror_case()
    assert point["SEGMENT"]["exp"][K] == pytest.approx(point["SEGMENT"]["exp_reflejada"][K], rel=0.02)     # misma desviacion
    tab = build_shape_comparison_table("z", "raw", groups, point, reps, True)
    for norm in NORMALIZATIONS:
        row = tab.loc[(tab["normalization"] == norm) & (tab["level_lo"] == 0.01)].iloc[0]
        assert row["ratio"] < 0.05 and row["label"] == LABEL_OVERLAP            # la ceguera, documentada
        assert row["left_ratio"] > PROFILE_STABILITY_FORM_THRESHOLD and row["right_ratio"] > PROFILE_STABILITY_FORM_THRESHOLD
        assert row["left_label"] == LABEL_DIVERGENT and row["right_label"] == LABEL_DIVERGENT
        assert row["asymmetry_range"] > 2.0 and row["asymmetry_range_ci_lo"] > 2.0
    a = build_asymmetry_table("z", "raw", groups, point, reps)
    a = a.loc[a["level_lo"] == 0.01].set_index("group")
    assert a.loc["exp", "asymmetry_ci_hi"] < 0 < a.loc["exp_reflejada", "asymmetry_ci_lo"]                # signos opuestos, firmes
    assert a.loc["exp", "half_width_over_std"] == pytest.approx(a.loc["exp_reflejada", "half_width_over_std"], rel=0.02)


def test_current_rule_and_one_sided_proposal_disagree_on_mirrored_distributions():
    """La regla VIGENTE no cambia (daria SOLO_ESCALA); la PROPUESTA -- no vigente -- daria ESCALA_Y_FORMA. `decide_th26` nunca usa la propuesta."""
    groups, point, reps = _mirror_case()
    tab = build_shape_comparison_table("z", "raw", groups, point, reps, True)
    full = pd.concat([tab.assign(grouping=g) for g in VERDICT_GROUPINGS], ignore_index=True)
    sens = pd.DataFrame({"point_verdict": [VERDICT_SCALE_ONLY] * 6, "is_primary": [True] + [False] * 5})
    current = decide_th26(full, sens, "z")
    assert current["verdict"] == VERDICT_SCALE_ONLY
    proposal = evaluate_one_sided_proposal(full, "z")
    assert proposal["status"] == "PROPUESTA_NO_VIGENTE" and proposal["verdict_if_adopted"] == VERDICT_SCALE_AND_SHAPE
    # quitar las columnas de la propuesta no altera el veredicto vigente
    stripped = full.drop(columns=["left_label", "right_label", "left_ratio", "right_ratio"])
    assert decide_th26(stripped, sens, "z")["verdict"] == current["verdict"]
    readings = build_alternative_readings(full, full, "z", current)
    assert (readings["status"] == "VIGENTE").sum() == 1 and readings.loc[readings["status"] == "VIGENTE", "verdict_before_robustness"].iloc[0] == VERDICT_SCALE_ONLY
    assert readings.loc[readings["status"] == "propuesta", "verdict_before_robustness"].iloc[0] == VERDICT_SCALE_AND_SHAPE


def test_one_sided_columns_agree_with_half_width_for_symmetric_pure_scale_groups():
    """La evidencia por colas no inventa forma donde no la hay."""
    tab, _ = _shape_case({s: (lambda rng, n, s=s: rng.normal(size=n) * s) for s in (1.0, 5.0, 25.0)})
    row = tab.loc[(tab["normalization"] == "std") & (tab["level_lo"] == 0.01)].iloc[0]
    assert row["left_label"] == LABEL_OVERLAP and row["right_label"] == LABEL_OVERLAP and row["asymmetry_range"] < 0.1
    full = pd.concat([tab.assign(grouping=g) for g in VERDICT_GROUPINGS], ignore_index=True)
    assert evaluate_one_sided_proposal(full, "z")["verdict_if_adopted"] == VERDICT_SCALE_ONLY


def _zero_std_case(n_boot=30):
    rng = np.random.default_rng(13)
    n_days, per = 100, 300
    day = np.tile(np.repeat(np.arange(n_days), per), 4)
    values = np.concatenate([rng.normal(size=n_days * per), rng.normal(size=n_days * per) * 2, np.zeros(n_days * per), rng.normal(size=n_days * per)])
    lab = np.repeat(np.array(["a", "b", "plano", "pequeño"], dtype=object), n_days * per)
    keep = np.ones(values.size, dtype=bool)
    small = np.flatnonzero(lab == "pequeño")
    keep[small[2_000:]] = False                                   # "pequeño": 2.000 obs -> cola 1% = 20 (soporte BAJO)
    groups = {"SEGMENT": prepare_sorted_groups(values[keep], day[keep], lab[keep])}
    point, reps = joint_day_block_bootstrap(groups, n_days, n_boot, 0)
    return groups, point, reps


def test_classify_groups_separates_the_four_sets():
    groups, point, _ = _zero_std_case()
    sets = classify_groups(groups["SEGMENT"], point["SEGMENT"], VERDICT_LEVEL_PAIR, "std")
    assert sets["available"] == ["a", "b", "pequeño", "plano"]
    assert sets["excluded_low_support"] == ["pequeño"]
    assert sets["excluded_invalid_normalization"] == ["plano"]      # soporte suficiente, desviacion nula
    assert sets["compared"] == ["a", "b"]
    assert len(sets["available"]) == len(sets["excluded_low_support"]) + len(sets["excluded_invalid_normalization"]) + len(sets["compared"])
    # sin normalizar, el grupo plano SI tiene un valor (0) y se compara
    assert classify_groups(groups["SEGMENT"], point["SEGMENT"], VERDICT_LEVEL_PAIR, "none")["compared"] == ["a", "b", "plano"]
    # criterio literal de TDA-10 (tamaño de grupo >= 200): "pequeño" entra
    assert "pequeño" in classify_groups(groups["SEGMENT"], point["SEGMENT"], VERDICT_LEVEL_PAIR, "std", min_tail=None)["compared"]


def test_group_with_support_but_zero_std_is_reported_and_blocks_a_firm_label():
    """ADVERSARIAL (defecto A1): antes, `n_groups` contaba al grupo plano y el ratio salia firme (SUPERPUESTO) ignorandolo."""
    groups, point, reps = _zero_std_case()
    tab = build_shape_comparison_table("z", "raw", groups, point, reps, True)
    for norm in NORMALIZATIONS:
        row = tab.loc[(tab["normalization"] == norm) & (tab["level_lo"] == 0.01)].iloc[0]
        assert row["n_groups_available"] == 4 and row["n_groups_compared"] == 2 and row["n_groups"] == 2
        assert row["groups_excluded_invalid_normalization"] == "plano" and row["groups_excluded_low_support"] == "pequeño"
        assert np.isfinite(row["ratio"]) and row["ratio"] < 0.1          # el ratio de a/b se informa...
        assert row["label"] == LABEL_INCONCLUSIVE                        # ...pero NO se concluye sobre "los grupos"
        assert row["left_label"] == LABEL_INCONCLUSIVE and row["right_label"] == LABEL_INCONCLUSIVE
        assert row["obs_fraction_compared"] == pytest.approx(60_000 / 92_000)
        assert row["n_boot_valid"] == 30
    none = tab.loc[(tab["normalization"] == "none") & (tab["level_lo"] == 0.01)].iloc[0]
    assert none["n_groups_compared"] == 3 and none["groups_excluded_invalid_normalization"] == "" and none["label"] == LABEL_DIVERGENT
    full = pd.concat([tab.assign(grouping=g) for g in VERDICT_GROUPINGS], ignore_index=True)
    assert decide_th26(full, pd.DataFrame({"point_verdict": [VERDICT_SCALE_ONLY] * 6}), "z")["verdict"] == VERDICT_INCONCLUSIVE


def test_profile_ratio_strict_never_ignores_a_group():
    assert profile_ratio(np.array([2.0, 3.0, np.nan])) == pytest.approx(0.4)             # modo permisivo (compatibilidad)
    assert np.isnan(profile_ratio(np.array([2.0, 3.0, np.nan]), strict=True))
    mat = profile_ratio(np.array([[2.0, 3.0, 4.0], [2.0, np.nan, 4.0]]), strict=True)
    assert mat[0] == pytest.approx(2 / 3) and np.isnan(mat[1])
    assert np.isnan(profile_ratio(np.array([2.0]), strict=True))


def test_replicates_where_a_compared_group_loses_its_scale_are_counted_not_hidden():
    """Un grupo cuya desviacion es positiva en la muestra pero nula en algunas replicas: esas replicas no aportan ratio y `n_boot_valid` lo dice."""
    n_days, per = 60, 400
    rng = np.random.default_rng(21)
    day = np.tile(np.repeat(np.arange(n_days), per), 2)
    spiky = np.zeros(n_days * per)
    spiky[:per] = rng.normal(size=per) * 5.0            # toda la variacion en UNA jornada
    values = np.concatenate([rng.normal(size=n_days * per), spiky])
    lab = np.repeat(np.array(["normal", "una_jornada"], dtype=object), n_days * per)
    groups = {"SEGMENT": prepare_sorted_groups(values, day, lab)}
    point, reps = joint_day_block_bootstrap(groups, n_days, 80, 0)
    tab = build_shape_comparison_table("z", "raw", groups, point, reps, True)
    row = tab.loc[(tab["normalization"] == "std") & (tab["level_lo"] == 0.25)].iloc[0]
    assert row["n_groups_compared"] == 2
    assert 0 < row["n_boot_valid"] < 80                  # ~ (1-1/60)^60 = 36 % de replicas sin esa jornada


# --- bootstrap con bloques de varias jornadas -------------------------------

def test_multi_day_blocks_keep_consecutive_days_together():
    """Con valor = indice de jornada, cada replica solo puede contener tramos de L jornadas consecutivas completas."""
    n_days, per, L = 40, 10, 5
    day = np.repeat(np.arange(n_days), per)
    groups = {"GLOBAL": prepare_sorted_groups(day.astype(float), day, np.full(day.size, "ALL", dtype=object))}
    _, reps = joint_day_block_bootstrap(groups, n_days, 40, 3, block_days=L)
    assert (reps["GLOBAL"]["ALL"][:, -1] == n_days * per).all()            # mismo tamaño: jornadas completas
    rng = np.random.default_rng(3)
    for b in range(5):
        starts = rng.integers(0, n_days - L + 1, size=n_days // L)
        sampled = (starts[:, None] + np.arange(L)[None, :]).ravel()
        mult = np.bincount(sampled, minlength=n_days)
        np.testing.assert_allclose(reps["GLOBAL"]["ALL"][b], _explicit_stats(np.repeat(np.arange(n_days, dtype=float), mult * per)), rtol=1e-10, atol=1e-12)
        for s in starts:                                                    # cada bloque aporta L dias seguidos
            assert (mult[s:s + L] >= 1).all()


def test_block_days_one_is_identical_to_the_approved_bootstrap():
    values, day, grp = _toy()
    groups = {"G": prepare_sorted_groups(values, day, grp)}
    _, a = joint_day_block_bootstrap(groups, 40, 12, seed=0)
    _, b = joint_day_block_bootstrap(groups, 40, 12, seed=0, block_days=1)
    np.testing.assert_array_equal(a["G"]["a"], b["G"]["a"])
    with pytest.raises(ValueError):
        joint_day_block_bootstrap(groups, 40, 2, seed=0, block_days=0)


def test_multi_day_blocks_widen_intervals_when_volatility_persists_across_days():
    """Si la escala persiste entre dias (regimenes largos), el bloque de 1 jornada SUBESTIMA la incertidumbre; con ruido independiente por dia, no."""
    def width_ratio(day_scale):
        n_days, per = len(day_scale), 80
        rng = np.random.default_rng(31)
        day = np.repeat(np.arange(n_days), per)
        values = rng.normal(size=day.size) * np.repeat(day_scale, per)
        groups = {"GLOBAL": prepare_sorted_groups(values, day, np.full(day.size, "ALL", dtype=object))}
        w = {}
        for L in (1, 21):
            _, reps = joint_day_block_bootstrap(groups, n_days, 300, 0, block_days=L)
            s = reps["GLOBAL"]["ALL"][:, K]
            w[L] = np.percentile(s, 97.5) - np.percentile(s, 2.5)
        return w[21] / w[1]

    rng = np.random.default_rng(32)
    persistent = np.repeat(np.exp(rng.normal(size=20)), 21)                  # 20 regimenes de 21 dias
    independent = np.exp(rng.normal(size=420))
    assert width_ratio(persistent) > 1.8
    assert 0.6 < width_ratio(independent) < 1.5


def test_block_length_sensitivity_table_structure():
    tab_groups, (groups, point, reps) = _shape_case({s: (lambda rng, n, s=s: rng.normal(size=n) * s) for s in (1.0, 5.0)}, n_boot=20)
    groups = {gk: groups["SEGMENT"] for gk in VERDICT_GROUPINGS}
    point = {gk: point["SEGMENT"] for gk in VERDICT_GROUPINGS}
    reps_by_block = {1: {gk: reps["SEGMENT"] for gk in VERDICT_GROUPINGS}}
    for L in BLOCK_LENGTH_SENSITIVITY_DAYS:
        _, r = joint_day_block_bootstrap({"SEGMENT": groups["SEGMENT"]}, 120, 20, 0, block_days=L)
        reps_by_block[L] = {gk: r["SEGMENT"] for gk in VERDICT_GROUPINGS}
    out = build_block_length_sensitivity("z", groups, 120, 20, 0, point, reps_by_block, [("SEGMENT", 1.0)])
    assert set(out["block_days"]) == {1, *BLOCK_LENGTH_SENSITIVITY_DAYS}
    assert out.loc[out["block_days"] == 1, "is_approved_procedure"].all() and not out.loc[out["block_days"] != 1, "is_approved_procedure"].any()
    assert (out.loc[out["block_days"] == 1, "width_ratio_vs_1day"] == 1.0).all()
    ratio_rows = out.loc[out["statistic"] == "shape_ratio_1_99"]
    assert len(ratio_rows) == len(VERDICT_GROUPINGS) * len(NORMALIZATIONS) * 3 and (ratio_rows["label_with_this_block"] != "").all()
    assert {"std", "half_width_1_99", "q0.01", "q0.999"} <= set(out.loc[out["group"] == "1.0", "statistic"])
    assert out.groupby(["statistic", "grouping", "group", "normalization"])["point"].nunique().max() == 1      # el punto no depende del bloque


# --- causalidad del decil sobre el conjunto EFECTIVO ------------------------

def test_decile_verification_uses_the_effective_valid_mask():
    """ADVERSARIAL (A2): con quemado e invalidos, la verificacion debe operar sobre las barras realmente elegibles."""
    sigma, dates = _sigma_days(n_days=60)
    valid = np.ones(sigma.size, dtype=bool)
    valid[:180] = False                       # quemado
    valid[1000:1040] = False                  # una jornada entera invalida
    sigma[500] = np.nan
    out = verify_causal_decile_no_lookahead(sigma, dates, valid=valid)
    assert out["passed"] is True and len(out["checks"]) >= 1
    n_eligible = int((valid & np.isfinite(sigma) & (sigma > 0)).sum())
    for c in out["checks"]:
        assert c["n_eligible_bars"] == n_eligible
        assert valid[c["perturb_index"]] and np.isfinite(sigma[c["perturb_index"]])   # nunca se perturba una barra no elegible
        assert c["truncation_identical"] is True


def test_decile_verification_catches_an_implementation_that_ignores_the_valid_mask_for_history(monkeypatch):
    """Una variante que respeta la causalidad sobre TODAS las barras pero deja que barras NO elegibles de la misma jornada... usen el futuro, solo se detecta si se verifica con la mascara efectiva."""
    sigma, dates = _sigma_days(n_days=60)
    valid = np.ones(sigma.size, dtype=bool)
    valid[:180] = False
    real = mod.causal_volatility_decile

    def leaky_when_masked(s, d, valid=None, n_deciles=10):
        if valid is None:
            return real(s, d, valid=None, n_deciles=n_deciles)          # correcto sin mascara
        out = assign_volatility_decile(np.where(valid, s, np.nan))       # con mascara: fronteras de TODA la muestra
        return out

    monkeypatch.setattr(mod, "causal_volatility_decile", leaky_when_masked)
    assert mod.verify_causal_decile_no_lookahead(sigma, dates)["passed"] is True                 # verificar SIN mascara no lo ve
    assert mod.verify_causal_decile_no_lookahead(sigma, dates, valid=valid)["passed"] is False   # con la mascara efectiva, si


def test_decile_truncation_check_catches_leak_that_perturbation_alone_misses(monkeypatch):
    """Fuga sutil: fronteras que dependen del NUMERO de jornadas futuras (no de sus valores). Perturbar un valor no la revela; truncar si."""
    sigma, dates = _sigma_days(n_days=60)
    real = mod.causal_volatility_decile
    uniq = np.unique(dates)

    def depends_on_sample_length(s, d, valid=None, n_deciles=10):
        out = real(s, d, valid=valid, n_deciles=n_deciles)
        n_days_total = np.unique(d).size
        return np.where(np.isfinite(out), (out + (n_days_total % 2)) % n_deciles, out)

    monkeypatch.setattr(mod, "causal_volatility_decile", depends_on_sample_length)
    out = mod.verify_causal_decile_no_lookahead(sigma, dates)
    assert all(c["earlier_days_unchanged"] and c["same_day_other_bars_unchanged"] for c in out["checks"])   # la perturbacion no la ve
    assert any(not c["truncation_identical"] for c in out["checks"]) and out["passed"] is False
    assert uniq.size == 60


def test_causal_decile_with_burn_in_mask_is_invariant_to_future_and_truncation():
    sigma, dates = _sigma_days(n_days=40)
    valid = np.ones(sigma.size, dtype=bool)
    valid[:180] = False
    valid[::37] = False
    base = causal_volatility_decile(sigma, dates, valid=valid)
    assert np.isnan(base[~valid]).all()
    cut = sorted(set(dates))[25]
    future = sigma.copy()
    future[dates >= cut] *= 1e4
    np.testing.assert_array_equal(base[dates < cut], causal_volatility_decile(future, dates, valid=valid)[dates < cut])
    keep = dates < cut
    np.testing.assert_array_equal(base[keep], causal_volatility_decile(sigma[keep], dates[keep], valid=valid[keep]))
    # cambiar la validez de barras FUTURAS tampoco altera el pasado
    valid2 = valid.copy()
    valid2[dates >= cut] = False
    np.testing.assert_array_equal(base[keep], causal_volatility_decile(sigma, dates, valid=valid2)[keep])


# --- decisiones metodologicas: lo que se reutiliza y lo que es nuevo --------

def test_unnormalized_ratio_is_literally_the_tda10_statistic():
    """Los umbrales 0.30/0.60 pertenecen al estadistico SIN normalizar: la fila `none` debe reproducir `profile_stability_ratio` de TDA-10."""
    from ohlcv_dataroad.ingest.tda10_scale_vs_shape import compute_group_quantile_table, profile_stability_ratio

    rng = np.random.default_rng(41)
    n_days, per = 100, 300
    day = np.tile(np.repeat(np.arange(n_days), per), 3)
    z = np.concatenate([rng.normal(size=n_days * per) * s for s in (1.0, 1.3, 0.7)])
    lab = np.repeat(np.array(["a", "b", "c"], dtype=object), n_days * per)
    groups = {"SEGMENT": prepare_sorted_groups(z, day, lab)}
    point, reps = joint_day_block_bootstrap(groups, n_days, 10, 0)
    tab = build_shape_comparison_table("z", "raw", groups, point, reps, True)
    none = tab.loc[(tab["normalization"] == "none") & (tab["level_lo"] == 0.01), "ratio"].iloc[0]
    assert none == pytest.approx(profile_stability_ratio(compute_group_quantile_table(z, lab)), rel=1e-12)
    std = tab.loc[(tab["normalization"] == "std") & (tab["level_lo"] == 0.01), "ratio"].iloc[0]
    assert none > 0.5 and std < 0.05          # son estadisticos DISTINTOS: el normalizado es la extension de TDA-12


def test_tail_support_criterion_is_stricter_than_tda10_group_size_criterion():
    """El minimo de 200 aplicado a la COLA (decision N5) excluye grupos que el criterio de TDA-10 (tamaño de grupo) admitia."""
    rng = np.random.default_rng(42)
    n_days = 100
    big = np.repeat(np.arange(n_days), 300)
    mid = np.repeat(np.arange(n_days), 75)                    # 7.500 obs (como 2019): cola 1% = 75
    values = np.concatenate([rng.normal(size=big.size), rng.normal(size=big.size), rng.normal(size=mid.size)])
    day = np.concatenate([big, big, mid])
    lab = np.concatenate([np.full(big.size, "a", dtype=object), np.full(big.size, "b", dtype=object), np.full(mid.size, "parcial", dtype=object)])
    groups = {"YEAR": prepare_sorted_groups(values, day, lab)}
    point, reps = joint_day_block_bootstrap(groups, n_days, 20, 0)
    strict = build_shape_comparison_table("z", "raw", groups, point, reps, True)
    literal = build_shape_comparison_table("z", "raw", groups, point, reps, True, min_tail=None)
    s = strict.loc[(strict["normalization"] == "std") & (strict["level_lo"] == 0.01)].iloc[0]
    l = literal.loc[(literal["normalization"] == "std") & (literal["level_lo"] == 0.01)].iloc[0]
    assert s["groups_excluded_low_support"] == "parcial" and s["n_groups_compared"] == 2 and s["obs_fraction_compared"] < 1.0
    assert l["groups_excluded_low_support"] == "" and l["n_groups_compared"] == 3 and l["obs_fraction_compared"] == 1.0


def test_decide_th26_reports_like_for_like_estimator_agreement_without_using_it():
    shape = _fake_shape({gk: (O, O) for gk in VERDICT_GROUPINGS})
    sens = pd.DataFrame({"point_verdict": [VERDICT_SCALE_ONLY] * 3 + [VERDICT_INCONCLUSIVE] * 3, "is_primary": [True] + [False] * 5})
    out = decide_th26(shape, sens, "z")
    assert out["primary_point_verdict"] == VERDICT_SCALE_ONLY and out["estimator_point_agreement_with_primary_point"] == pytest.approx(0.5)
    assert out["verdict"] == VERDICT_INCONCLUSIVE and out["verdict_before_robustness"] == VERDICT_SCALE_ONLY   # la regla vigente no cambio
    no_flag = decide_th26(shape, _sens([VERDICT_SCALE_ONLY] * 6), "z")
    assert no_flag["primary_point_verdict"] == "" and np.isnan(no_flag["estimator_point_agreement_with_primary_point"])


def test_inconclusive_remains_an_allowed_outcome_and_is_never_upgraded_by_alternative_readings(e2e):
    _, _, _, result = e2e
    readings = result.alternative_readings
    assert list(readings["status"]).count("VIGENTE") == 1 and list(readings["status"]).count("propuesta") == 1
    assert readings.loc[readings["status"] == "VIGENTE", "verdict_before_robustness"].iloc[0] == result.th26["verdict_before_robustness"]
    assert result.one_sided_proposal["status"] == "PROPUESTA_NO_VIGENTE"
    assert set(readings["verdict_before_robustness"]) <= {VERDICT_SCALE_ONLY, VERDICT_SCALE_AND_SHAPE, VERDICT_INCONCLUSIVE}


def test_e2e_audit_outputs_are_present_and_consistent(e2e):
    from ohlcv_dataroad.ingest.run_tda12 import persist_artifacts

    config, _, _, result = e2e
    shape = result.shape_table
    for col in ("n_groups_available", "n_groups_compared", "groups_excluded_low_support", "groups_excluded_invalid_normalization",
                "obs_fraction_compared", "n_boot_valid", "left_ratio", "right_ratio", "asymmetry_range", "left_label", "right_label"):
        assert col in shape.columns
    def _count(v):
        return len(v.split(";")) if isinstance(v, str) and v else 0

    counted = shape["groups_excluded_low_support"].map(_count) + shape["groups_excluded_invalid_normalization"].map(_count) + shape["n_groups_compared"]
    assert (counted == shape["n_groups_available"]).all()                    # los cuatro conjuntos cuadran siempre
    assert (shape["n_groups"] == shape["n_groups_compared"]).all()
    assert set(result.block_sensitivity["block_days"]) == {1, *BLOCK_LENGTH_SENSITIVITY_DAYS}
    assert set(result.block_sensitivity["series"]) == {"r_1m", "z_ewma_60_raw"}
    assert {c["block"] for c in result.causality_decile["checks"]} == {"raw", "clock_adjusted"}
    assert all(c["truncation_identical"] for c in result.causality_decile["checks"])

    persist_artifacts(result, config, t0=0.0, run_command="test")
    assert len(pd.read_csv(config.tda12_readings_csv_path)) == 6
    assert "is_approved_procedure" in pd.read_csv(config.tda12_block_sensitivity_csv_path).columns
    text = config.tda12_report_path.read_text(encoding="utf-8")
    for needle in ("ciego a la asimetría", "PROPUESTA", "no está aprobada", "Tratamiento de 2019", "jornadas CONSECUTIVAS",
                   "sin normalización válida", "Reutilizar una constante no significa", "truncar la muestra"):
        assert needle in text, needle


def test_e2e_closing_report_scope_and_numbering(e2e):
    """Cierre documental: alcance de `SUPERPUESTO`, decisiones exploratorias, numeracion de limitaciones y alcance real de la causalidad."""
    import re

    from ohlcv_dataroad.ingest.run_tda12 import render_report

    config, _, _, result = e2e
    text = render_report(result, config, 1.0, "test")
    for needle in ("Qué significa `SUPERPUESTO`", "No demuestra que el perfil distribucional completo sea igual", "EXPLORATORIAS",
                   "`NO CONCLUYENTE` como salida", "no se incorporará a él sin aprobación específica", "alcance exacto de lo ejecutado",
                   "NO se re-verificaron aquí"):
        assert needle in text, needle
    assert "todas las configuraciones pasan" not in text                    # no se afirma mas de lo ejecutado
    assert f"para {len(result.causality_sigma)} configuración(es)" in text   # el recuento sale de lo realmente verificado
    lim = text.split("## 14. Limitaciones")[1].split("## 15.")[0]
    numbers = [int(m) for m in re.findall(r"^(\d+)\. ", lim, flags=re.MULTILINE)]
    assert numbers == list(range(1, len(numbers) + 1)) and len(numbers) >= 10
