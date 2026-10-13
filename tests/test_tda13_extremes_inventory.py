"""Tests de TDA-13 Parte A (inventario, triage y agrupamiento de extremos; TH06, TH27).

Cubren: umbrales relativos y excedencias, reglas de triage, conteos
observados y esperanzas exactas bajo independencia (contra formulas
cerradas y contra simulacion), controles positivos y negativos del
pipeline, efecto del triage con las convenciones de TDA-07, reglas de
veredicto, y un extremo a extremo con invariantes (holdout, datos de
entrada intactos, reproducibilidad).
"""
from __future__ import annotations

import datetime
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import ohlcv_dataroad.ingest.tda13_extremes_inventory as mod
from ohlcv_dataroad.config import load_config
from ohlcv_dataroad.ingest.holdout_guard import HoldoutIsolationError
from ohlcv_dataroad.ingest.tda06_intraday_calendar_profile import attach_calendar_fields
from ohlcv_dataroad.ingest.tda07_marginal_distribution import QUANTILE_LEVELS, compute_moments_quantiles
from ohlcv_dataroad.ingest.tda13_extremes_inventory import (
    CAT_BOUNDARY,
    CAT_ERROR,
    CAT_REAL,
    CAT_ROLL,
    CAT_UNDETERMINED,
    CATEGORIES,
    CLAIM_INCONCLUSIVE,
    CLAIM_NO,
    CLAIM_NOT_DEMONSTRABLE,
    CLAIM_YES,
    DEF_GLOBAL,
    DEF_SEGMENT,
    DEF_SEGMENT_X_VOL,
    GAP_BIN_EDGES,
    NULL_CLOCK,
    NULL_CLOCK_YEAR,
    NULL_CLOCK_YEAR_LEGACY,
    NULL_FAMILY_CLOCK,
    NULL_FAMILY_PERMUTATION,
    NULL_HOMOGENEOUS,
    PRIMARY_DEFINITION,
    PRIMARY_LEVEL,
    PRIMARY_NULL,
    PRIMARY_WINDOW,
    RUN_LENGTH_MAX,
    SERIES_R,
    SERIES_Z,
    SIDE_BOTH,
    SIDE_NEG,
    SIDE_POS,
    TH06_HIGH_UNDETERMINED,
    TH06_MOSTLY_CLASSIFIABLE,
    TH27_EXPLAINED,
    TH27_INCONCLUSIVE,
    TH27_NONE,
    TH27_PERSISTS,
    THRESHOLD_LEVELS,
    VOL_FAMILY_EWMA_FAST,
    VOL_FAMILY_EWMA_PRIMARY,
    VOL_FAMILY_LOCAL,
    MissingEvidenceError,
    ReplicateContext,
    add_context_flags,
    analyze_variant,
    bootstrap_day_weights,
    build_exceedances,
    cell_codes,
    classify_extremes,
    cluster_statistics,
    compute_thresholds,
    decide_th06,
    decide_th27,
    fit_two_way_rates,
    generate_null_series,
    is_distinguishable,
    null_expectations,
    observed_counts,
    permute_within_groups,
    primary_replicate_statistics,
    primary_statistics_from_flags,
    reference_conservation,
    reference_probabilities,
    run_tasks,
    run_tda13_analysis,
    side_flags,
    stats_from_sorted,
    summarize_calibration,
    summarize_full_bootstrap,
    triage_impact_table,
    triage_sensitivity,
    volatility_paths,
    weighted_cell_exceedances,
    wilson_interval,
)

TICK = 0.25


# ---------------------------------------------------------------------------
# Constantes predeclaradas
# ---------------------------------------------------------------------------

def test_levels_are_inherited_from_tda07_and_primary_choices_are_fixed():
    assert PRIMARY_LEVEL == 0.01 and set(THRESHOLD_LEVELS) == {0.01, 0.001, 0.05}
    for lvl in THRESHOLD_LEVELS:
        assert lvl in QUANTILE_LEVELS and (1 - lvl) in QUANTILE_LEVELS
    assert PRIMARY_DEFINITION == DEF_SEGMENT_X_VOL and PRIMARY_NULL == NULL_CLOCK_YEAR and PRIMARY_WINDOW == 10
    assert set(CATEGORIES) == {"ERROR_DATO", "ARTEFACTO_ROLL", "ARTEFACTO_FRONTERA", "EVENTO_REAL", "INDETERMINADO"}


def test_module_does_not_implement_evt():
    """Parte B no existe en este modulo: ningun ajuste de cola ni indice extremal."""
    names = [n.lower() for n in dir(mod)]
    assert not any(k in n for n in names for k in ("gpd", "pareto", "extremal_index", "hill", "mean_excess"))


# ---------------------------------------------------------------------------
# Umbrales y excedencias
# ---------------------------------------------------------------------------

def test_thresholds_are_relative_to_each_cell_not_global():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(size=20000), 10.0 * rng.normal(size=20000)])   # celda tranquila y celda agitada
    cells = np.repeat(np.array(["calma", "agitada"], dtype=object), 20000)
    table, neg, pos, lo, hi = compute_thresholds(x, cells, 0.01)
    t = table.set_index("cell")
    assert t.loc["agitada", "threshold_hi"] > 5 * t.loc["calma", "threshold_hi"]
    for c in ("calma", "agitada"):                       # misma tasa en ambas celdas: el umbral relativo no selecciona la celda agitada
        m = cells == c
        assert abs(neg[m].mean() - 0.01) < 5e-4 and abs(pos[m].mean() - 0.01) < 5e-4
    _, gneg, gpos, _, _ = compute_thresholds(x, np.full(x.size, "ALL", dtype=object), 0.01)
    assert (gneg | gpos)[cells == "calma"].mean() < 0.001   # un umbral global casi no ve la celda tranquila


def test_exceedance_is_strict_and_ties_do_not_inflate_the_rate():
    x = np.array([0.0] * 990 + [1.0] * 10)                 # el cuantil 99 % cae en un empate
    table, neg, pos, lo, hi = compute_thresholds(x, np.full(1000, "c", dtype=object), 0.01)
    assert not neg.any()                                    # nada es estrictamente menor que el minimo
    assert pos.mean() <= 0.01 and table["rate_pos"].iloc[0] <= 0.01
    assert table["n"].iloc[0] == 1000 and table["expected_per_tail"].iloc[0] == pytest.approx(10.0)


def test_cell_codes_by_definition():
    seg, dec = np.array(["a", "a", "b"]), np.array([0.0, 3.0, 3.0])
    assert list(cell_codes(DEF_GLOBAL, seg, dec)) == ["ALL"] * 3
    assert list(cell_codes(DEF_SEGMENT, seg, dec)) == ["a", "a", "b"]
    assert list(cell_codes(DEF_SEGMENT_X_VOL, seg, dec)) == ["a|D0", "a|D3", "b|D3"]
    with pytest.raises(ValueError):
        cell_codes("OTRA", seg, dec)


def test_side_flags():
    neg, pos = np.array([True, False, False]), np.array([False, True, False])
    assert list(side_flags(neg, pos, SIDE_NEG)) == [True, False, False]
    assert list(side_flags(neg, pos, SIDE_POS)) == [False, True, False]
    assert list(side_flags(neg, pos, SIDE_BOTH)) == [True, True, False]


# ---------------------------------------------------------------------------
# Triage
# ---------------------------------------------------------------------------

def _inv_row(**over):
    base = {"invalid_reason": "VALID", "bad_data": False, "bad_data_prev": False, "ff_category": "ACTIVA", "ff_category_prev": "ACTIVA",
            "adjacent_unknown_gap": False, "next_bar_contiguous": True, "reverts_next_bar": False, "low_volume": False}
    base.update(over)
    return base


def test_triage_rules_each_category_and_order():
    rows = [
        _inv_row(),                                                               # R5
        _inv_row(bad_data=True, invalid_reason="ROLL_BOUNDARY"),                   # R1 gana a R2
        _inv_row(bad_data_prev=True),                                             # R1 por la barra anterior
        _inv_row(invalid_reason="ROLL_BOUNDARY", reverts_next_bar=True),           # R2 gana a R4
        _inv_row(invalid_reason="TRADING_DATE_BOUNDARY"),                         # R3
        _inv_row(invalid_reason="NON_CONSECUTIVE_MINUTE"),                        # R3
        _inv_row(reverts_next_bar=True),                                          # R4
        _inv_row(low_volume=True, ff_category_prev="FLAT_AISLADA"),               # R4 con dos dudas
        _inv_row(next_bar_contiguous=False),                                      # R4: no se puede comprobar la persistencia
        _inv_row(adjacent_unknown_gap=True),                                      # R4
    ]
    out = classify_extremes(pd.DataFrame(rows))
    assert list(out["category"]) == [CAT_REAL, CAT_ERROR, CAT_ERROR, CAT_ROLL, CAT_BOUNDARY, CAT_BOUNDARY,
                                     CAT_UNDETERMINED, CAT_UNDETERMINED, CAT_UNDETERMINED, CAT_UNDETERMINED]
    assert list(out["triage_rule"]) == ["R5", "R1", "R1", "R2", "R3", "R3", "R4", "R4", "R4", "R4"]
    assert out["triage_evidence"].iloc[7] == "BARRA_NO_ACTIVA;VOLUMEN_BAJO"
    assert out["triage_evidence"].iloc[6] == "REVIERTE_BARRA_SIGUIENTE" and out["triage_evidence"].iloc[0] == "SIN_DUDAS_R4"


def test_real_event_requires_positive_evidence_not_absence_of_error():
    """Ningun `EVENTO_REAL` puede tener una duda de R4, y la ausencia de `bad_data` no basta."""
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame({
        "invalid_reason": "VALID", "bad_data": False, "bad_data_prev": False,
        "ff_category": rng.choice(["ACTIVA", "FLAT_AISLADA"], n, p=[0.9, 0.1]), "ff_category_prev": "ACTIVA",
        "adjacent_unknown_gap": rng.random(n) < 0.1, "next_bar_contiguous": rng.random(n) > 0.1,
        "reverts_next_bar": rng.random(n) < 0.2, "low_volume": rng.random(n) < 0.1,
    })
    out = classify_extremes(df)
    real = out["category"] == CAT_REAL
    assert real.any() and (out["category"] == CAT_UNDETERMINED).any()
    r = out.loc[real]
    assert (r["ff_category"] == "ACTIVA").all() and not r["adjacent_unknown_gap"].any() and r["next_bar_contiguous"].all()
    assert not r["reverts_next_bar"].any() and not r["low_volume"].any()
    assert not (out["category"] == CAT_ERROR).any()        # sin evidencia de TDA-00 nada se declara error


def test_context_flags_reversal_contiguity_and_gap_adjacency():
    ts = pd.to_datetime(["2024-01-02 15:00", "2024-01-02 15:01", "2024-01-02 15:02", "2024-01-02 15:10", "2024-01-02 15:11", "2024-01-02 15:12"])
    df = pd.DataFrame({
        "timestamp": ts, "trading_date": datetime.date(2024, 1, 2), "contract": "H24",
        "close": [100.0, 110.0, 104.0, 104.0, 105.0, 106.0], "volume": [100, 1, 100, 100, 100, 100],
        "bad_data": [False, False, True, False, False, False], "ff_category": ["ACTIVA", "ACTIVA", "FLAT_AISLADA", "ACTIVA", "ACTIVA", "ACTIVA"],
        "last_before_unknown_gap": [False, False, True, False, False, False], "first_after_unknown_gap": [False, False, False, True, False, False],
        "minute_of_day": [510, 511, 512, 520, 521, 522],
    })
    out = add_context_flags(df)
    assert list(out["next_bar_contiguous"]) == [True, True, False, True, True, False]
    assert bool(out["reverts_next_bar"].iloc[1])            # +10 y luego -6: revierte mas de la mitad
    assert not out["reverts_next_bar"].iloc[4]              # +1 y luego +1: se sostiene
    assert not out["reverts_next_bar"].iloc[2]              # sin barra siguiente contigua no se afirma reversion
    assert out["reversal_fraction_next_bar"].iloc[1] == pytest.approx(0.6) and out["reversal_fraction_next_bar"].iloc[4] == pytest.approx(-1.0)
    assert np.isnan(out["reversal_fraction_next_bar"].iloc[2])
    assert list(out["bad_data_prev"]) == [False, False, False, True, False, False]
    assert out["ff_category_prev"].iloc[3] == "FLAT_AISLADA"
    assert list(out["adjacent_unknown_gap"]) == [False, False, True, False, True, False]   # ultima antes del hueco; y barra cuyo previo es la primera tras el hueco
    assert list(out["clock_0830_window"]) == [True, True, True, False, False, False]
    assert bool(out["low_volume"].iloc[1]) and not out["low_volume"].iloc[0]


# ---------------------------------------------------------------------------
# Conteos observados
# ---------------------------------------------------------------------------

def test_observed_counts_hand_example_never_crosses_blocks():
    #        bloque 0 (dia 0)            | bloque 1 (dia 0)   | bloque 2 (dia 1)
    exc = np.array([1, 1, 0, 0, 1, 0, 0,   1, 0, 1, 1, 1,       0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=bool)
    block = np.array([0] * 7 + [1] * 5 + [2] * 13)
    day = np.array([0] * 12 + [1] * 13)
    c = observed_counts(exc, block, day, 2)
    assert list(c["n"]) == [7.0, 2.0]
    # tiempos: bloque 0 -> 1, 3 ; bloque 1 -> 2, 1, 1 ; bloque 2 -> 11. Nada entre bloques (4->7 y 11->13 NO cuentan).
    assert list(c["b"]) == [5.0, 1.0]
    assert list(c["a"][1]) == [3.0, 0.0] and list(c["a"][2]) == [4.0, 0.0] and list(c["a"][5]) == [5.0, 0.0] and list(c["a"][30]) == [5.0, 1.0]
    # rachas dia 0: [0,1] long 2, [4] long 1, [7] long 1, [9,10,11] long 3 ; dia 1: dos de longitud 1
    assert list(c["n_runs"]) == [4.0, 2.0]
    assert list(c["runs"][0]) == [2.0, 1.0, 1.0, 0.0, 0.0] and list(c["runs"][1]) == [2.0, 0.0, 0.0, 0.0, 0.0]
    assert list(c["in_runs2"]) == [5.0, 0.0]


def test_observed_counts_consecutive_exceedances_across_a_block_boundary_are_not_a_run():
    exc = np.array([0, 1, 1, 0], dtype=bool)
    c = observed_counts(exc, np.array([0, 0, 1, 1]), np.zeros(4, dtype=int), 1)
    assert c["b"][0] == 0 and c["n_runs"][0] == 2 and c["in_runs2"][0] == 0 and c["runs"][0, 0] == 2


def test_observed_counts_empty():
    c = observed_counts(np.zeros(10, dtype=bool), np.zeros(10, dtype=int), np.zeros(10, dtype=int), 1)
    assert c["n"][0] == 0 and c["b"][0] == 0 and c["n_runs"][0] == 0


# ---------------------------------------------------------------------------
# Esperanzas exactas bajo independencia
# ---------------------------------------------------------------------------

def test_null_expectations_closed_form_single_block_homogeneous():
    n, p = 500, 0.03
    e = null_expectations(np.full(n, p), np.zeros(n, dtype=int), np.zeros(n, dtype=int), 1)
    q = 1 - p
    assert e["n"][0] == pytest.approx(n * p)
    assert e["var"][0] == pytest.approx(n * p * q)
    assert e["b"][0] == pytest.approx(n * p - (1 - q ** n))
    for k in GAP_BIN_EDGES:
        assert e["a"][k][0] == pytest.approx(sum((n - d) * p * p * q ** (d - 1) for d in range(1, k + 1)))
    # rachas de longitud exacta l que no tocan los bordes: (1-p) p^l (1-p); con bordes se resuelve contando "longitud >= l".
    ge = [sum(p ** l * (1.0 if i == 0 else q) for i in range(n - l + 1)) for l in range(1, RUN_LENGTH_MAX + 1)]
    assert e["n_runs"][0] == pytest.approx(ge[0])
    for j in range(RUN_LENGTH_MAX - 1):
        assert e["runs"][0, j] == pytest.approx(ge[j] - ge[j + 1])
    assert e["runs"][0, -1] == pytest.approx(ge[-1])
    assert e["in_runs2"][0] == pytest.approx(n * p - (ge[0] - ge[1]))


def test_null_expectations_match_simulation_with_blocks_and_heterogeneous_probabilities():
    rng = np.random.default_rng(3)
    lengths = [40, 7, 1, 120, 60, 15, 300]
    block = np.repeat(np.arange(len(lengths)), lengths)
    day = np.repeat([0, 0, 0, 1, 1, 2, 2], lengths)
    p = rng.uniform(0.01, 0.25, size=block.size)
    e = null_expectations(p, block, day, 3)
    acc = None
    n_sim = 4000
    for _ in range(n_sim):
        c = observed_counts(rng.random(p.size) < p, block, day, 3)
        vec = np.concatenate([c["n"], c["b"], c["a"][1], c["a"][10], c["a"][60], c["n_runs"], c["in_runs2"], c["runs"].ravel()])
        acc = vec if acc is None else acc + vec
    sim = acc / n_sim
    exact = np.concatenate([e["n"], e["b"], e["a"][1], e["a"][10], e["a"][60], e["n_runs"], e["in_runs2"], e["runs"].ravel()])
    assert np.allclose(sim, exact, rtol=0.06, atol=0.02)


def test_a_block_of_one_bar_contributes_no_interval():
    e = null_expectations(np.full(5, 0.5), np.arange(5), np.zeros(5, dtype=int), 1)
    assert e["b"][0] == pytest.approx(0.0) and all(e["a"][k][0] == pytest.approx(0.0) for k in GAP_BIN_EDGES)
    assert e["in_runs2"][0] == pytest.approx(0.0)           # sin barras contiguas no hay rachas de 2


def _unequal_composition(seed=4, n_days=600, per_day=120):
    """Dos años con composicion horaria MUY distinta: el segundo año casi no opera en los minutos de tasa alta."""
    rng = np.random.default_rng(seed)
    minute = np.tile(np.arange(per_day), n_days)
    year = np.repeat(np.where(np.arange(n_days) < n_days // 2, 2021, 2022), per_day)
    keep = ~((year == 2022) & (minute < 30) & (rng.random(minute.size) < 0.85))      # 2022 pierde el 85 % de sus barras de los minutos 0-29
    minute, year = minute[keep], year[keep]
    rate = (0.004 + 0.05 * (minute < 30)) * np.where(year == 2022, 2.5, 1.0)
    return rng.random(minute.size) < rate, minute, year


def test_legacy_reference_does_not_preserve_margins_under_unequal_hourly_composition():
    """Hallazgo A reproducido: `p(minuto) * tasa(año)/tasa(global)` no conserva ni la intensidad total ni los totales por año ni por minuto."""
    exc, minute, year = _unequal_composition()
    p = reference_probabilities(exc, minute, year, NULL_CLOCK_YEAR_LEGACY)
    assert abs(p.sum() - exc.sum()) > 10                                 # no conserva la intensidad total (sobran mas de 10 excedencias esperadas)
    assert max(abs(p[year == y].sum() - exc[year == y].sum()) for y in (2021, 2022)) > 10
    tab = reference_conservation(exc, minute, year).set_index("reference")
    assert tab.loc[NULL_CLOCK_YEAR_LEGACY, "max_abs_error_by_year"] > 10 and tab.loc[NULL_CLOCK_YEAR_LEGACY, "max_abs_error_by_minute"] > 0.5


def test_corrected_reference_preserves_total_minute_and_year_margins():
    exc, minute, year = _unequal_composition()
    p, info = fit_two_way_rates(exc, minute, year)
    assert info["converged"] and info["n_clipped"] == 0 and (p >= 0).all() and (p < 1).all()
    assert p.sum() == pytest.approx(exc.sum(), abs=1e-6)
    for y in (2021, 2022):
        assert p[year == y].sum() == pytest.approx(exc[year == y].sum(), abs=1e-6)
    om, em = np.bincount(minute, weights=exc.astype(float)), np.bincount(minute, weights=p)
    assert np.allclose(om, em, atol=1e-6)
    assert np.array_equal(p, reference_probabilities(exc, minute, year, NULL_CLOCK_YEAR))
    tab = reference_conservation(exc, minute, year).set_index("reference")
    assert tab.loc[NULL_CLOCK_YEAR, "max_abs_error_by_year"] < 1e-6 and tab.loc[NULL_CLOCK_YEAR, "max_abs_error_by_minute"] < 1e-6
    assert abs(tab.loc[NULL_CLOCK_YEAR, "total_error"]) < 1e-6
    # las secundarias conservan lo que declaran: RELOJ el minuto (no el año), HOMOGENEA solo el total
    assert tab.loc[NULL_CLOCK, "max_abs_error_by_minute"] < 1e-6 and tab.loc[NULL_CLOCK, "max_abs_error_by_year"] > 10
    assert abs(tab.loc[NULL_HOMOGENEOUS, "total_error"]) < 1e-6 and tab.loc[NULL_HOMOGENEOUS, "max_abs_error_by_minute"] > 1


def test_corrected_reference_recovers_a_multiplicative_truth_and_removes_spurious_clustering():
    """Con tasa verdadera = factor(minuto) x factor(año) y composicion desigual, la referencia corregida recupera las tasas; la original las sesga y eso se traduce en un cociente distinto de 1."""
    exc, minute, year = _unequal_composition(seed=11, n_days=1200)
    truth = (0.004 + 0.05 * (minute < 30)) * np.where(year == 2022, 2.5, 1.0)
    p_new = reference_probabilities(exc, minute, year, NULL_CLOCK_YEAR)
    p_old = reference_probabilities(exc, minute, year, NULL_CLOCK_YEAR_LEGACY)
    hot22 = (year == 2022) & (minute < 30)
    assert p_new[hot22].mean() == pytest.approx(truth[hot22].mean(), rel=0.08)
    assert abs(p_old[hot22].mean() / truth[hot22].mean() - 1) > 0.15           # la formula original yerra justo donde la composicion difiere
    day = np.cumsum(np.concatenate([[1], np.diff(minute) < 0])) - 1               # una jornada por cada vuelta del reloj
    n_days = int(day.max()) + 1
    ones = np.ones((1, n_days))
    obs = observed_counts(exc, day, day, n_days)
    r_new = cluster_statistics(obs, null_expectations(p_new, day, day, n_days), ones)[f"ratio_share_le_{PRIMARY_WINDOW}"][0]
    r_old = cluster_statistics(obs, null_expectations(p_old, day, day, n_days), ones)[f"ratio_share_le_{PRIMARY_WINDOW}"][0]
    assert abs(r_new - 1) < 0.06 and abs(r_old - 1) > abs(r_new - 1)


def test_two_way_rates_edge_cases_zero_levels_and_missing_combinations():
    minute = np.array([0, 0, 1, 1, 2, 2, 2, 2])
    year = np.array([2021, 2021, 2021, 2022, 2022, 2022, 2022, 2022])           # (0, 2022) y (2, 2021) no existen
    exc = np.array([1, 0, 0, 0, 1, 0, 1, 0], dtype=bool)                          # el minuto 1 no tiene ninguna excedencia
    p, info = fit_two_way_rates(exc, minute, year)
    assert info["converged"] and np.isfinite(p).all()
    assert (p[minute == 1] == 0).all() and info["n_bars_p_zero"] == 2            # nivel sin excedencias -> probabilidad 0, declarado
    assert p.sum() == pytest.approx(exc.sum()) and p[year == 2021].sum() == pytest.approx(1.0) and p[year == 2022].sum() == pytest.approx(2.0)
    p0, info0 = fit_two_way_rates(np.zeros(8, dtype=bool), minute, year)
    assert (p0 == 0).all() and info0["total_expected"] == 0.0
    pe, infoe = fit_two_way_rates(np.zeros(0, dtype=bool), np.zeros(0, dtype=int), np.zeros(0, dtype=int))
    assert pe.size == 0 and infoe["converged"]
    with pytest.raises(ValueError):
        reference_probabilities(exc, minute, year, "OTRA")


def test_reference_weights_are_equivalent_to_repeating_bars():
    exc, minute, year = _unequal_composition(seed=5, n_days=60)
    w = np.random.default_rng(1).integers(0, 4, size=exc.size)
    rep_idx = np.repeat(np.arange(exc.size), w)
    for kind in (NULL_CLOCK_YEAR, NULL_CLOCK, NULL_HOMOGENEOUS, NULL_CLOCK_YEAR_LEGACY):
        a = reference_probabilities(exc, minute, year, kind, weights=w)[rep_idx]
        b = reference_probabilities(exc[rep_idx], minute[rep_idx], year[rep_idx], kind)
        assert np.allclose(a, b, atol=1e-9), kind


# ---------------------------------------------------------------------------
# Estadisticos: controles negativos y positivos
# ---------------------------------------------------------------------------

def _grid(n_days=120, per_day=600, years=(2021, 2022)):
    n = n_days * per_day
    day = np.repeat(np.arange(n_days), per_day)
    minute = np.tile(np.arange(per_day), n_days)
    year = np.repeat(np.array(years), n // len(years))
    return n, day, minute, year, day.copy()                # un bloque de continuidad por jornada


def test_independent_exceedances_give_ratio_one():
    """Control negativo: excedencias independientes con un perfil de reloj fuerte. La referencia primaria debe dar ~1 (promediando semillas: sin sesgo)."""
    n, day, minute, year, block = _grid(n_days=300)
    ones = np.ones((1, 300))
    ratios, homogeneous = [], []
    for seed in range(8):
        exc = np.random.default_rng(seed).random(n) < (0.01 + 0.04 * (minute < 20))
        rows, _ = analyze_variant(exc, minute, year, block, day, 300, ones, nulls=(NULL_CLOCK_YEAR, NULL_HOMOGENEOUS), ks=(1, PRIMARY_WINDOW))
        t = pd.DataFrame(rows).set_index(["reference", "statistic"])
        ratios.append([t.loc[(NULL_CLOCK_YEAR, s), "ratio"] for s in (f"share_le_{PRIMARY_WINDOW}", "share_le_1", "frac_in_runs_ge2", "daily_dispersion")])
        homogeneous.append(t.loc[(NULL_HOMOGENEOUS, f"share_le_{PRIMARY_WINDOW}"), "ratio"])
    mean = np.mean(ratios, axis=0)
    assert abs(mean[0] - 1) < 0.04 and abs(mean[1] - 1) < 0.12 and abs(mean[2] - 1) < 0.12 and abs(mean[3] - 1) < 0.08
    # La referencia homogenea confunde el perfil de reloj con agrupamiento: por eso NO es la primaria.
    assert np.mean(homogeneous) > 1.1


def test_null_band_covers_independent_data_and_distinguishability_needs_both_conditions():
    n, day, minute, year, block = _grid(n_days=200)
    exc = np.random.default_rng(21).random(n) < 0.02
    rows, _ = analyze_variant(exc, minute, year, block, day, 200, bootstrap_day_weights(200, 80, 0), n_sim=60)
    r = pd.DataFrame(rows).set_index("statistic").loc[f"share_le_{PRIMARY_WINDOW}"]
    assert r["n_null_sim"] == 60 and r["null_sim_p2_5"] < 1.0 < r["null_sim_p97_5"]
    assert r["null_sim_p2_5"] - 0.05 < r["ratio"] < r["null_sim_p97_5"] + 0.05
    assert is_distinguishable(r) is False


def test_clustered_exceedances_are_detected():
    n, day, minute, year, block = _grid()
    rng = np.random.default_rng(6)
    seeds = rng.random(n) < 0.004
    exc = seeds.copy()
    idx = np.flatnonzero(seeds)
    for shift in (1, 2, 4):                                 # cada evento arrastra otros en los minutos siguientes
        j = idx + shift
        j = j[(j < n) & (day[np.minimum(j, n - 1)] == day[np.minimum(idx, n - 1)][: j.size])]
        exc[j[rng.random(j.size) < 0.5]] = True
    w = bootstrap_day_weights(120, 60, 0)
    rows, detail = analyze_variant(exc, minute, year, block, day, 120, w, n_sim=40)
    t = pd.DataFrame(rows).set_index("statistic")
    assert t.loc[f"share_le_{PRIMARY_WINDOW}", "ratio"] > 3 and is_distinguishable(t.loc[f"share_le_{PRIMARY_WINDOW}"]) is True
    assert t.loc["frac_in_runs_ge2", "ratio"] > 3 and t.loc["mean_run_length", "ratio"] > 1.05
    gaps, runs = mod.distribution_tables(detail, (PRIMARY_NULL,))
    assert gaps["share_observed"].sum() == pytest.approx(1.0) and gaps["share_expected"].sum() == pytest.approx(1.0)
    assert runs["share_observed"].sum() == pytest.approx(1.0)
    assert list(gaps["gap_bars"]) == ["1", "2", "3-5", "6-10", "11-30", "31-60", ">60"]
    assert gaps.loc[gaps["gap_bars"] == "1", "share_ratio"].iloc[0] > 3


def test_row_mask_splits_blocks_where_the_subset_is_interrupted():
    n, day, minute, year, block = _grid(n_days=20, per_day=100, years=(2021, 2022))
    exc = np.zeros(n, dtype=bool)
    exc[[10, 12, 60, 62]] = True                            # dos pares a 2 barras, en la jornada 0
    w = np.ones((1, 20))
    mask = (minute < 50)
    rows, detail = analyze_variant(exc, minute, year, block, day, 20, w, ks=(1, PRIMARY_WINDOW), row_mask=mask)
    assert detail["obs"]["b"].sum() == 1                    # solo el par dentro del subconjunto; 12 -> 60 nunca se mide
    rows_all, detail_all = analyze_variant(exc, minute, year, block, day, 20, w, ks=(1, PRIMARY_WINDOW))
    assert detail_all["obs"]["b"].sum() == 3


def test_cluster_statistics_weights_reproduce_a_resampled_sample():
    """Ponderar jornadas equivale a repetirlas: el bootstrap por multiplicidades es exacto."""
    n, day, minute, year, block = _grid(n_days=12, per_day=200)
    rng = np.random.default_rng(7)
    exc = rng.random(n) < 0.05
    p = reference_probabilities(exc, minute, year, NULL_CLOCK_YEAR)
    obs, null = observed_counts(exc, block, day, 12), null_expectations(p, block, day, 12)
    w = np.array([[2, 0, 1, 1, 3, 0, 0, 1, 1, 1, 2, 0]], dtype=float)
    a = cluster_statistics(obs, null, w)
    rep = np.repeat(np.arange(12), w[0].astype(int))
    obs2 = {k: (v[rep] if not isinstance(v, dict) else {kk: vv[rep] for kk, vv in v.items()}) for k, v in obs.items()}
    null2 = {k: (v[rep] if not isinstance(v, dict) else {kk: vv[rep] for kk, vv in v.items()}) for k, v in null.items()}
    b = cluster_statistics(obs2, null2, np.ones((1, rep.size)))
    for key in a:
        assert a[key][0] == pytest.approx(b[key][0], rel=1e-9, abs=1e-12), key


def test_bootstrap_day_weights_are_reproducible_and_sum_to_n_days():
    w1, w2 = bootstrap_day_weights(50, 20, 0), bootstrap_day_weights(50, 20, 0)
    assert np.array_equal(w1, w2) and (w1.sum(axis=1) == 50).all() and w1.shape == (20, 50)
    assert not np.array_equal(w1, bootstrap_day_weights(50, 20, 1))


def test_permute_within_groups_preserves_each_group_and_breaks_order():
    rng = np.random.default_rng(8)
    v, g = rng.normal(size=5000), rng.integers(0, 7, size=5000)
    out = permute_within_groups(v, g, np.random.default_rng(9))
    for k in range(7):
        assert np.array_equal(np.sort(out[g == k]), np.sort(v[g == k]))
    assert not np.array_equal(out, v)


# ---------------------------------------------------------------------------
# Pipeline: volatilidad pura frente a agrupamiento propio
# ---------------------------------------------------------------------------

def _pipeline_ratio(r, trading_date, segment, minute, year, block, day, n_days):
    ex = build_exceedances(r, trading_date, segment, levels=(PRIMARY_LEVEL,), definitions=(DEF_GLOBAL, PRIMARY_DEFINITION))
    e = ex.eligible
    out = {}
    for name in (SERIES_R, "z"):
        for definition in (DEF_GLOBAL, PRIMARY_DEFINITION):
            neg, pos = ex.flags[(name, definition, PRIMARY_LEVEL)]
            rows, _ = analyze_variant(neg | pos, minute[e], year[e], block[e], day[e], n_days, np.ones((1, n_days)), ks=(1, PRIMARY_WINDOW))
            out[(name, definition)] = pd.DataFrame(rows).set_index("statistic").loc[f"share_le_{PRIMARY_WINDOW}", "ratio"]
    return out, ex


def _synthetic_days(n_days=160, per_day=500):
    dates = np.array([datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(n_days)])
    trading_date = np.repeat(dates, per_day)
    day = np.repeat(np.arange(n_days), per_day)
    minute = np.tile(np.arange(per_day), n_days)
    year = np.repeat(np.where(np.arange(n_days) < n_days // 2, 2022, 2023), per_day)
    segment = np.where(minute < per_day // 2, "seg_a", "seg_b").astype(object)
    return trading_date, day, minute, year, segment


def test_pure_slow_volatility_clusters_in_raw_global_but_not_after_standardizing():
    """Si los extremos solo se agrupan porque la volatilidad cambia DESPACIO, el agrupamiento debe verse con umbral global en crudo y desaparecer en z con umbral relativo."""
    trading_date, day, minute, year, segment = _synthetic_days()
    rng = np.random.default_rng(10)
    t_all = np.arange(day.size)
    sigma = np.exp(0.9 * np.sin(2 * np.pi * t_all / (25 * 500)))      # ciclo de 25 jornadas, continuo: el EWMA de 60 barras lo sigue bien
    r = rng.standard_t(6, size=day.size) * 1e-4 * sigma
    ratios, ex = _pipeline_ratio(r, trading_date, segment, minute, year, day.copy(), day, 160)
    assert ratios[(SERIES_R, DEF_GLOBAL)] > 1.5              # un umbral global "ve" agrupamiento: es la volatilidad
    assert 0.75 < ratios[("z", PRIMARY_DEFINITION)] < 1.15   # al estandarizar y usar umbral relativo, el exceso desaparece (muestra pequeña: ruidoso)
    assert ex.eligible.sum() < r.size                        # quemado y jornadas sin historia quedan fuera, no se rellenan


def test_abrupt_volatility_jumps_leave_residual_clustering_in_z_because_the_estimator_lags():
    """Limitacion documentada en el informe: si la volatilidad SALTA de golpe (aqui, al empezar cada jornada), `sigma_hat` va con retraso y `z` conserva agrupamiento aunque los retornos sean independientes dado sigma. "Persiste en z" no prueba un fenomeno ajeno a la volatilidad."""
    trading_date, day, minute, year, segment = _synthetic_days()
    rng = np.random.default_rng(10)
    sigma_day = np.exp(rng.normal(scale=0.8, size=160))
    r = rng.standard_t(6, size=day.size) * 1e-4 * sigma_day[day]
    ratios, _ = _pipeline_ratio(r, trading_date, segment, minute, year, day.copy(), day, 160)
    assert ratios[("z", PRIMARY_DEFINITION)] > 1.15
    assert ratios[("z", PRIMARY_DEFINITION)] < ratios[(SERIES_R, DEF_GLOBAL)]   # aun asi, estandarizar reduce el agrupamiento aparente


def test_bursts_unrelated_to_volatility_persist_after_standardizing():
    trading_date, day, minute, year, segment = _synthetic_days()
    rng = np.random.default_rng(11)
    r = rng.normal(size=day.size) * 1e-4
    starts = rng.choice(np.flatnonzero((minute > 10) & (minute < 480)), size=600, replace=False)
    for s in starts:                                        # rafagas de 3 barras extremas, de signo alterno (no elevan la volatilidad media de forma duradera)
        r[s:s + 3] = np.array([8.0, -8.0, 8.0]) * 1e-4
    ratios, _ = _pipeline_ratio(r, trading_date, segment, minute, year, day.copy(), day, 160)
    assert ratios[("z", PRIMARY_DEFINITION)] > 2.0 and ratios[(SERIES_R, PRIMARY_DEFINITION)] > 2.0


def test_build_exceedances_uses_a_common_population_and_equal_rates():
    trading_date, day, minute, year, segment = _synthetic_days(n_days=80)
    rng = np.random.default_rng(12)
    r = rng.standard_t(5, size=day.size) * 1e-4
    ex = build_exceedances(r, trading_date, segment)
    n_e = int(ex.eligible.sum())
    for key, (neg, pos) in ex.flags.items():
        assert neg.size == n_e and pos.size == n_e           # misma poblacion para r y z, todas las definiciones y niveles
        assert not (neg & pos).any()
        assert abs(neg.mean() - key[2]) < 0.004 and abs(pos.mean() - key[2]) < 0.004
    assert np.isfinite(ex.z[ex.eligible]).all() and (ex.decile[ex.eligible] >= 0).all()
    alt = build_exceedances(r, trading_date, segment, levels=(PRIMARY_LEVEL,), definitions=(PRIMARY_DEFINITION,), halflife=20.0, eligible=ex.eligible, decile=ex.decile)
    assert np.array_equal(alt.eligible, ex.eligible)         # la z de sensibilidad se compara sobre las mismas barras
    assert not np.allclose(alt.z[ex.eligible], ex.z[ex.eligible])


def test_fast_causal_decile_is_identical_to_the_tda12_definition():
    """Las replicas usan una implementacion mas rapida del decil causal: debe coincidir EXACTAMENTE con la de TDA-12 (empates, no finitos, mascara, historia corta, jornadas de tamaño desigual)."""
    from ohlcv_dataroad.ingest.tda12_conditional_quantiles import causal_volatility_decile

    rng = np.random.default_rng(0)
    for trial in range(6):
        sizes = rng.integers(1, 90, size=int(rng.integers(3, 140)))
        dates = np.repeat(np.arange(sizes.size), sizes)
        n = dates.size
        s = np.exp(rng.normal(size=n))
        if trial % 2 == 0:
            s = np.round(s, 1)                                    # muchos empates
        s[rng.random(n) < 0.03] = np.nan
        s[rng.random(n) < 0.02] = 0.0
        valid = None if trial < 2 else rng.random(n) > 0.1
        ref = causal_volatility_decile(s, dates, valid=valid)
        for chunk in (1, 7, 32, 1000):
            assert np.array_equal(mod.causal_volatility_decile_fast(s, dates, valid=valid, chunk_days=chunk), ref, equal_nan=True), (trial, chunk)
    assert np.isnan(mod.causal_volatility_decile_fast(np.array([np.nan, 0.0]), np.array([0, 1]))).all()
    assert mod.causal_volatility_decile_fast(np.zeros(0), np.zeros(0)).size == 0


@pytest.mark.parametrize("case", ["una_jornada", "todo_igual", "historia_justa", "historia_insuficiente", "jornada_enteramente_invalida", "fechas_no_ordenadas_alfabeticamente",
                                  "valores_extremos", "otro_numero_de_deciles", "empate_con_toda_la_historia"])
def test_fast_causal_decile_edge_cases_match_tda12(case):
    """Casos limite, uno a uno, contra la implementacion original de TDA-12 (que no se modifica)."""
    from ohlcv_dataroad.ingest.tda12_conditional_quantiles import causal_volatility_decile

    rng = np.random.default_rng(1)
    n_dec, valid = 10, None
    if case == "una_jornada":
        s, d = np.exp(rng.normal(size=500)), np.zeros(500, dtype=int)                       # sin jornadas previas: todo NaN
    elif case == "todo_igual":
        s, d = np.full(900, 0.37), np.repeat(np.arange(9), 100)                             # ningun historico es ESTRICTAMENTE menor: decil 0
    elif case == "historia_justa":
        s, d = np.exp(rng.normal(size=400)), np.repeat([0, 1], 200)                         # exactamente n_deciles*20 = 200 historicos: el minimo entra
    elif case == "historia_insuficiente":
        s, d = np.exp(rng.normal(size=399)), np.repeat([0, 1], [199, 200])                  # 199 historicos: no llega al minimo
    elif case == "jornada_enteramente_invalida":
        s, d = np.exp(rng.normal(size=900)), np.repeat(np.arange(3), 300)
        valid = d != 1                                                                      # la jornada central no aporta historia ni recibe decil
    elif case == "fechas_no_ordenadas_alfabeticamente":
        s = np.exp(rng.normal(size=900))
        d = np.repeat(np.array(["2021-12-31", "2022-01-03", "2022-01-02"], dtype=object), 300)   # manda el orden de aparicion, no el de la etiqueta
    elif case == "valores_extremos":
        s, d = np.exp(rng.normal(scale=12, size=1200)), np.repeat(np.arange(6), 200)
        s[5], s[700] = np.inf, -1.0
    elif case == "otro_numero_de_deciles":
        s, d, n_dec = np.exp(rng.normal(size=1500)), np.repeat(np.arange(15), 100), 4
    else:
        s, d = np.concatenate([np.full(300, 2.0), np.full(300, 2.0), np.full(300, 5.0)]), np.repeat(np.arange(3), 300)
    ref = causal_volatility_decile(s, d, valid=valid, n_deciles=n_dec)
    for chunk in (1, 2, 32):
        got = mod.causal_volatility_decile_fast(s, d, valid=valid, n_deciles=n_dec, chunk_days=chunk)
        assert np.array_equal(got, ref, equal_nan=True), (case, chunk)
    if case == "una_jornada" or case == "historia_insuficiente":
        assert np.isnan(ref).all()
    if case == "historia_justa":
        assert np.isfinite(ref[200:]).all() and np.isnan(ref[:200]).all()
    if case == "todo_igual":
        assert (ref[200:] == 0).all()
    if case == "empate_con_toda_la_historia":
        assert (ref[300:600] == 0).all() and (ref[600:] == n_dec - 1).all()


def test_fast_causal_decile_matches_tda12_on_a_realistic_sized_series():
    from ohlcv_dataroad.ingest.tda12_conditional_quantiles import causal_volatility_decile

    rng = np.random.default_rng(2)
    sizes = rng.integers(900, 1400, size=120)
    d = np.repeat(np.arange(120), sizes)
    r = rng.standard_t(4, size=d.size) * 1e-4 * np.exp(0.5 * np.sin(np.arange(d.size) / 5000.0))
    sigma, z = mod.standardize_primary(r)
    ref = causal_volatility_decile(sigma, d, valid=np.isfinite(z))
    got = mod.causal_volatility_decile_fast(sigma, d, valid=np.isfinite(z))
    assert np.array_equal(got, ref, equal_nan=True) and np.isfinite(ref).sum() > 100000
    a = build_exceedances(r, d, np.full(d.size, "s", dtype=object), levels=(PRIMARY_LEVEL,), definitions=(PRIMARY_DEFINITION,))
    b = build_exceedances(r, d, np.full(d.size, "s", dtype=object), levels=(PRIMARY_LEVEL,), definitions=(PRIMARY_DEFINITION,), decile_fn=mod.causal_volatility_decile_fast)
    assert np.array_equal(a.eligible, b.eligible)
    for key in a.flags:                                                                       # y, por tanto, las mismas excedencias
        assert np.array_equal(a.flags[key][0], b.flags[key][0]) and np.array_equal(a.flags[key][1], b.flags[key][1])


def test_memory_safe_workers_never_exceeds_what_fits():
    assert mod.MAX_WORKERS == 8 and mod.default_n_workers(10000) <= 8
    assert mod.memory_safe_workers(8, None) == 1                                              # si no se puede medir la memoria, lo seguro
    assert mod.memory_safe_workers(8, 3.0) == 1                                               # por debajo de la reserva: un solo proceso
    assert mod.memory_safe_workers(8, 14.0) == 6                                              # (14 - 4) * 0,6 / 1 GB
    assert mod.memory_safe_workers(8, 64.0) == 8 and mod.memory_safe_workers(3, 64.0) == 3   # nunca mas de lo solicitado
    free = mod.available_memory_gb()
    assert free is None or free > 0


def test_volatility_state_is_causal_future_returns_do_not_change_past_flags():
    """G1: perturbar retornos futuros no cambia `z`, el decil ni la elegibilidad de las barras anteriores."""
    trading_date, day, minute, year, segment = _synthetic_days(n_days=40)
    rng = np.random.default_rng(13)
    r = rng.normal(size=day.size) * 1e-4
    cut = 25 * 500
    r2 = r.copy()
    r2[cut:] *= 7.0
    a, b = build_exceedances(r, trading_date, segment), build_exceedances(r2, trading_date, segment)
    assert np.array_equal(a.eligible[:cut], b.eligible[:cut])
    assert np.allclose(a.z[:cut], b.z[:cut], equal_nan=True) and np.array_equal(a.decile[:cut], b.decile[:cut], equal_nan=True)


# ---------------------------------------------------------------------------
# Efecto del triage
# ---------------------------------------------------------------------------

def test_stats_from_sorted_matches_tda07_conventions():
    rng = np.random.default_rng(14)
    x = rng.standard_t(4, size=30000) * 1e-3
    s = stats_from_sorted(np.sort(x))
    ref = compute_moments_quantiles(x)
    got = dict(zip(mod.TRIAGE_STAT_NAMES, s))
    for key in ("n", "std", "kurtosis_excess", "kurtosis_excess_trimmed", "q0.001", "q0.01", "q0.99", "q0.999"):
        assert got[key] == pytest.approx(ref[key], rel=1e-10, abs=1e-15), key
    assert np.isnan(stats_from_sorted(np.sort(x[:10]))[3])    # n < 20: sin recorte, como TDA-07


def test_triage_impact_table_differences_and_reproducibility():
    rng = np.random.default_rng(15)
    n_days = 40
    day = np.repeat(np.arange(n_days), 300)
    x = rng.normal(size=day.size)
    big = np.zeros(day.size, dtype=bool)
    big[rng.choice(day.size, 30, replace=False)] = True
    x[big] *= 25.0
    scen = {"BASE": (x, day), "SIN_GRANDES": (x[~big], day[~big])}
    meta = {"BASE": {"series": "r", "baseline": "BASE", "description": "todo"}, "SIN_GRANDES": {"series": "r", "baseline": "BASE", "description": "sin 30"}}
    t1 = triage_impact_table(scen, meta, n_days, 40, 0)
    t2 = triage_impact_table(scen, meta, n_days, 40, 0)
    pd.testing.assert_frame_equal(t1, t2)
    k = t1.loc[(t1["scenario"] == "SIN_GRANDES") & (t1["statistic"] == "kurtosis_excess")].iloc[0]
    assert k["n_removed_vs_baseline"] == 30 and k["diff_vs_baseline"] < -5 and k["diff_ci_hi"] < 0
    base = t1.loc[(t1["scenario"] == "BASE") & (t1["statistic"] == "kurtosis_excess")].iloc[0]
    assert base["ci_lo"] <= base["value"] <= base["ci_hi"] and base["diff_vs_baseline"] == 0 and np.isnan(base["diff_ci_lo"])


# ---------------------------------------------------------------------------
# Veredictos
# ---------------------------------------------------------------------------

STAT = f"ratio_le_{PRIMARY_WINDOW}"
ALL_ABOVE = {2020: 1.3, 2021: 1.4, 2022: 1.2, 2023: 1.5, 2024: 1.3}


def _point(r=(1.9, 1.6), z=(1.5, 1.4), z_years=ALL_ABOVE):
    out = {}
    for side, rv, zv in ((SIDE_NEG, r[0], z[0]), (SIDE_POS, r[1], z[1])):
        for series, v in ((SERIES_R, rv), (SERIES_Z, zv)):
            out[f"{series}|{side}|{STAT}"] = v
            out[f"{series}|{side}|observed_le_{PRIMARY_WINDOW}"] = 0.2 * v
            out[f"{series}|{side}|expected_le_{PRIMARY_WINDOW}"] = 0.2
        for y, v in z_years.items():
            out[f"{SERIES_Z}|{side}|{STAT}_year_{y}"] = v
        out[f"{SERIES_Z}|{side}|min_year_ratio"] = min(z_years.values())
        out[f"{SERIES_Z}|{side}|n_complete_years"] = float(len(z_years))
        out[f"r_vs_z|{side}|fraction_removed"] = 1 - (zv - 1) / (rv - 1) if rv > 1 else np.nan
    return out


def _boot(point, half=0.05, override=None):
    rows = []
    for key, v in point.items():
        series, side, name = key.split("|", 2)
        if name in (STAT, "fraction_removed"):
            lo, hi = (override or {}).get((series, side), (v - half, v + half))
            rows.append({"series": series, "side": side, "statistic": name, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def _bands(hi=1.05):
    return pd.DataFrame([{"series": s, "side": sd, "band_hi": hi} for s in (SERIES_R, SERIES_Z) for sd in (SIDE_NEG, SIDE_POS, SIDE_BOTH)])


def _vol_replicates(z_mean, n=40, sd=0.02, families=(VOL_FAMILY_EWMA_PRIMARY, VOL_FAMILY_EWMA_FAST, VOL_FAMILY_LOCAL)):
    rng = np.random.default_rng(0)
    rows = []
    for fam in families:
        for rep_i in range(n):
            for side in (SIDE_NEG, SIDE_POS):
                rows.append({"kind": fam, "replicate": rep_i, "key": f"{SERIES_Z}|{side}|{STAT}", "value": z_mean[fam] + sd * rng.normal()})
    return pd.DataFrame(rows)


EMPTY = pd.DataFrame(columns=["kind", "replicate", "key", "value"])
NO_RATES = pd.DataFrame(columns=["family", "rule", "rate"])


def test_th27_no_clustering():
    pt = _point(r=(1.01, 1.0), z=(1.0, 1.0))
    v = decide_th27(pt, _boot(pt), _bands(), True, NO_RATES, EMPTY)
    assert v["verdict"] == TH27_NONE and v["claim_1"] == CLAIM_NO and v["claim_2"] == "NO_EVALUADA" and v["claim_3"] == CLAIM_NOT_DEMONSTRABLE


def test_th27_not_detectable_after_standardizing():
    pt = _point(z=(1.01, 1.0))
    v = decide_th27(pt, _boot(pt), _bands(), True, NO_RATES, EMPTY)
    assert v["verdict"] == TH27_EXPLAINED and v["claim_1"] == CLAIM_YES and v["claim_2"] == CLAIM_NO
    assert v["detail"][SIDE_NEG]["fraction_removed"] == pytest.approx(1 - 0.01 / 0.9)


def test_th27_persistence_never_asserts_an_own_extremal_phenomenon():
    """Afirmaciones 1 y 2 detectadas NO convierten la 3 en demostrada: el veredicto lleva el origen como no separable."""
    pt = _point()
    v = decide_th27(pt, _boot(pt), _bands(), True, NO_RATES, EMPTY)
    assert v["claim_1"] == CLAIM_YES and v["claim_2"] == CLAIM_YES and v["claim_3"] == CLAIM_NOT_DEMONSTRABLE
    assert v["verdict"] == TH27_PERSISTS and "ORIGEN_NO_IDENTIFICADO" in v["verdict"]
    assert CLAIM_NOT_DEMONSTRABLE == "NO_IDENTIFICADO_POR_ESTE_PROCEDIMIENTO"       # es una limitacion del procedimiento, no una afirmacion de imposibilidad
    assert "IMPOSIBLE" not in v["verdict"] and "SEPARABLE" not in v["verdict"] and "DEMOSTRABLE" not in v["claim_3"]


def test_th27_persists_requires_stability_across_complete_years():
    pt = _point(z_years={**ALL_ABOVE, 2022: 0.95})
    v = decide_th27(pt, _boot(pt), _bands(), True, NO_RATES, EMPTY)
    assert v["claim_2"] == CLAIM_INCONCLUSIVE and v["verdict"] == TH27_INCONCLUSIVE and v["detail"][SIDE_POS]["z_years_above_1"] == 4
    missing_year = _point(z_years={k: v_ for k, v_ in ALL_ABOVE.items() if k != 2024})
    assert decide_th27(missing_year, _boot(missing_year), _bands(), True, NO_RATES, EMPTY)["claim_2"] == CLAIM_INCONCLUSIVE   # un año completo sin dato no cuenta como estable


def test_th27_needs_both_the_calibrated_band_and_the_ci():
    pt = _point(r=(1.04, 1.04))                                   # por debajo de la banda calibrada (1,05) aunque el IC excluya 1
    assert decide_th27(pt, _boot(pt, half=0.01), _bands(), True, NO_RATES, EMPTY)["claim_1"] == CLAIM_NO
    pt2 = _point()
    touching = {(SERIES_R, SIDE_NEG): (0.99, 2.5), (SERIES_R, SIDE_POS): (0.99, 2.5)}    # por encima de la banda pero el IC con re-estimacion incluye 1
    assert decide_th27(pt2, _boot(pt2, override=touching), _bands(), True, NO_RATES, EMPTY)["claim_1"] == CLAIM_NO
    assert decide_th27(pt2, _boot(pt2), _bands(hi=2.5), True, NO_RATES, EMPTY)["claim_1"] == CLAIM_NO     # una banda mas ancha quita la deteccion: la banda manda


def test_th27_sides_disagree_gives_inconclusive():
    pt = _point(r=(1.9, 1.0))
    v = decide_th27(pt, _boot(pt), _bands(), True, NO_RATES, EMPTY)
    assert v["claim_1"] == CLAIM_INCONCLUSIVE and v["verdict"] == TH27_INCONCLUSIVE
    pt2 = _point(z=(1.5, 1.0))
    v2 = decide_th27(pt2, _boot(pt2), _bands(), True, NO_RATES, EMPTY)
    assert v2["claim_1"] == CLAIM_YES and v2["claim_2"] == CLAIM_INCONCLUSIVE and v2["verdict"] == TH27_INCONCLUSIVE


def test_th27_without_passed_calibration_nothing_is_claimed():
    pt = _point()
    for bad in (False, None):
        v = decide_th27(pt, _boot(pt), _bands(), bad, NO_RATES, EMPTY)
        assert v["verdict"] == TH27_INCONCLUSIVE and v["claim_1"] == CLAIM_INCONCLUSIVE and v["claim_1_before_g2_gate"] == CLAIM_YES
        assert v["claim_2"] == CLAIM_INCONCLUSIVE


def test_th27_missing_data_is_inconclusive_not_a_guess():
    pt = _point()
    assert decide_th27(pt, _boot(pt), _bands().iloc[0:0], True, NO_RATES, EMPTY)["verdict"] == TH27_INCONCLUSIVE     # sin banda
    assert decide_th27(pt, _boot(pt).iloc[0:0], _bands(), True, NO_RATES, EMPTY)["verdict"] == TH27_INCONCLUSIVE     # sin IC
    assert decide_th27({}, _boot(pt), _bands(), True, NO_RATES, EMPTY)["verdict"] == TH27_INCONCLUSIVE
    assert is_distinguishable({"ratio": 1.5, "ratio_ci_lo": np.nan, "null_sim_p97_5": 1.03}) is None


def test_th27_volatility_only_evidence_is_reported_but_never_decides():
    pt = _point(z=(1.5, 1.4))
    reproduces = {VOL_FAMILY_EWMA_PRIMARY: 1.0, VOL_FAMILY_EWMA_FAST: 1.15, VOL_FAMILY_LOCAL: 1.6}
    v = decide_th27(pt, _boot(pt), _bands(), True, NO_RATES, _vol_replicates(reproduces))
    ev = v["volatility_only_evidence"].set_index(["family", "side"])
    assert v["volatility_controls_reproducing_real_z"] == [VOL_FAMILY_LOCAL]
    assert bool(ev.loc[(VOL_FAMILY_LOCAL, SIDE_NEG), "real_within_control_range"]) and not ev.loc[(VOL_FAMILY_EWMA_FAST, SIDE_NEG), "real_within_control_range"]
    assert ev.loc[(VOL_FAMILY_EWMA_FAST, SIDE_NEG), "share_of_real_excess_reproduced"] == pytest.approx(0.3, abs=0.03)
    none = decide_th27(pt, _boot(pt), _bands(), True, NO_RATES, _vol_replicates({k: 1.0 for k in reproduces}))
    assert none["volatility_controls_reproducing_real_z"] == []
    for res in (v, none):                                       # en ninguno de los dos casos cambia el veredicto ni se afirma la 3
        assert res["verdict"] == TH27_PERSISTS and res["claim_3"] == CLAIM_NOT_DEMONSTRABLE


# ---------------------------------------------------------------------------
# Calibracion G2 (hallazgo B)
# ---------------------------------------------------------------------------

def test_wilson_interval():
    lo, hi = wilson_interval(0, 500)
    assert lo == 0.0 and 0.005 < hi < 0.01
    lo, hi = wilson_interval(25, 500)
    assert 0.03 < lo < 0.05 < hi < 0.08
    assert all(np.isnan(v) for v in wilson_interval(0, 0))


def _null_replicates(n=400, mean=None, sd=0.02, seed=0, odd_shift=0.0, families=(NULL_FAMILY_PERMUTATION, NULL_FAMILY_CLOCK)):
    """Replicas sinteticas del cociente: cada familia con su media; `odd_shift` desplaza SOLO la mitad reservada (simula un procedimiento mal calibrado)."""
    rng = np.random.default_rng(seed)
    mean = mean or {NULL_FAMILY_PERMUTATION: 1.0, NULL_FAMILY_CLOCK: 1.015}
    rows = []
    for fam in families:
        for rep_i in range(n):
            shift = odd_shift if rep_i % 2 == 1 else 0.0
            for series in (SERIES_R, SERIES_Z):
                for side in (SIDE_NEG, SIDE_POS, SIDE_BOTH):
                    rows.append({"kind": fam, "replicate": rep_i, "key": f"{series}|{side}|{STAT}", "value": mean[fam] + shift + sd * rng.normal()})
                    rows.append({"kind": fam, "replicate": rep_i, "key": f"{series}|{side}|min_year_ratio", "value": 0.9 + 0.2 * rng.random()})
    return pd.DataFrame(rows)


def test_calibrated_band_comes_from_even_replicates_of_the_most_demanding_family():
    reps = _null_replicates()
    rates, bands, ok = summarize_calibration(reps)
    b = bands.set_index(["series", "side"]).loc[(SERIES_R, SIDE_NEG)]
    even = reps.loc[(reps["kind"] == NULL_FAMILY_CLOCK) & (reps["replicate"] % 2 == 0) & (reps["key"] == f"{SERIES_R}|{SIDE_NEG}|{STAT}"), "value"]
    assert b["band_hi"] == pytest.approx(np.percentile(even, 97.5))                    # la familia con sesgo positivo fija la banda
    assert b[f"{NULL_FAMILY_CLOCK}|p97_5"] > b[f"{NULL_FAMILY_PERMUTATION}|p97_5"] and b[f"{NULL_FAMILY_CLOCK}|n_band"] == 200
    assert ok is True
    acc = rates.loc[rates["in_acceptance_criterion"].astype(bool)]
    assert (acc["n_replicates"] == 200).all() and (acc["n_detections"] <= acc["acceptance_max_detections"]).all()
    assert len(acc) == 8 and (acc["n_rates_in_criterion"] == 8).all() and acc["acceptance_level"].iloc[0] == pytest.approx(1 - 0.05 / 8)
    assert acc["rule"].str.startswith("cola").all()
    assert (acc["acceptance_max_detections"] == mod.max_detections_under_exchangeability(200, 200, 1 - 0.05 / 8)).all()
    assert set(acc["family"]) == {NULL_FAMILY_PERMUTATION, NULL_FAMILY_CLOCK} and not acc["side"].eq(SIDE_BOTH).any()
    tail = acc.loc[acc["family"] == NULL_FAMILY_CLOCK]
    assert tail["rate"].between(0.0, 0.07).all()                                         # ~2,5 % nominal en la familia que fija la banda
    glob = rates.loc[rates["rule"].str.startswith("regla global") & (rates["family"] == NULL_FAMILY_CLOCK)]
    assert len(glob) == 2 and (glob["rate"] <= tail["rate"].min() + 1e-12).all()         # exigir ambas colas no puede detectar mas que la cola que menos detecta


def test_calibration_fails_when_the_held_out_half_detects_too_often():
    """El criterio debe poder FALLAR: si la mitad reservada produce detecciones muy por encima del nominal, G2 no se supera (la media 'dentro de la banda' ya no basta)."""
    reps = _null_replicates(odd_shift=0.035)
    rates, bands, ok = summarize_calibration(reps)
    assert ok is False
    bad = rates.loc[rates["in_acceptance_criterion"].astype(bool) & (rates["passes"] == False)]   # noqa: E712
    assert len(bad) and (bad["n_detections"] > bad["acceptance_max_detections"]).all() and bad["rate"].min() > 0.1
    mean_all = reps.loc[reps["key"] == f"{SERIES_R}|{SIDE_NEG}|{STAT}", "value"].mean()
    b = bands.set_index(["series", "side"]).loc[(SERIES_R, SIDE_NEG)]
    assert b["band_lo"] < mean_all < b["band_hi"]                                        # con el criterio antiguo (media dentro) esto se habria dado por bueno


def test_legacy_band_false_detection_rate_is_quantified_on_the_same_replicates():
    reps = _null_replicates(sd=0.025)
    legacy = {(s, sd): 1.02 for s in (SERIES_R, SERIES_Z) for sd in (SIDE_NEG, SIDE_POS, SIDE_BOTH)}       # una banda demasiado estrecha
    rates, _, _ = summarize_calibration(reps, legacy)
    row = rates.loc[(rates["family"] == NULL_FAMILY_CLOCK) & (rates["rule"] == f"cola: {SERIES_R} {SIDE_NEG}")].iloc[0]
    assert row["legacy_band_rate"] > 0.3 and row["legacy_band_rate_ci_lo"] > mod.G2_NOMINAL_RATE and row["legacy_band_n"] == 400
    assert row["rate"] < row["legacy_band_rate"]


def test_acceptance_criterion_controls_its_own_false_alarm_rate_and_keeps_power():
    """El criterio no debe rechazar por azar un procedimiento bien calibrado (correccion por comparaciones multiples), y debe rechazar uno que detecta el doble o mas de lo nominal."""
    fails_ok = sum(summarize_calibration(_null_replicates(n=1000, seed=s))[2] is False for s in range(20))
    assert fails_ok <= 2                                                                  # <= 5 % nominal: el criterio ya descuenta el error de estimar la banda
    fails_bad = sum(summarize_calibration(_null_replicates(n=1000, seed=s, odd_shift=0.02))[2] is False for s in range(10))
    assert fails_bad >= 9                                                                 # desplazar la mitad reservada 1 desviacion tipica (~16 % de detecciones) se detecta casi siempre


def test_band_order_statistic_is_the_lower_neighbour_of_the_interpolated_percentile():
    """El percentil 97,5 de numpy se interpola entre dos estadisticos de orden; `band_order_statistic` devuelve el inferior (posicion 1..n)."""
    rng = np.random.default_rng(1)
    for n in (16, 100, 200, 500, 501):
        x = np.sort(rng.normal(size=n))
        j = mod.band_order_statistic(n)
        pct = np.percentile(x, 97.5)
        assert x[j - 1] <= pct <= x[min(j, n - 1)]
    assert mod.band_order_statistic(500) == 487 and mod.band_order_statistic(200) == 195
    assert mod.band_order_statistic(41) == 40                                             # 0,975*40 = 39 exacto: el percentil ES el estadistico de orden 40


def test_max_detections_is_an_exact_conservative_bound_for_the_interpolated_percentile():
    """Segunda auditoria, H1. La ley beta-binomial es EXACTA para el estadistico de orden inferior y, por tanto, una COTA para el percentil interpolado (que nunca queda por debajo)."""
    rng = np.random.default_rng(0)
    n_band, n_eval, n_sim = 200, 200, 20000
    j = mod.band_order_statistic(n_band)
    c_interp, c_order = np.empty(n_sim, dtype=int), np.empty(n_sim, dtype=int)
    for s in range(n_sim):
        band, new = np.sort(rng.standard_t(3, size=n_band)), rng.standard_t(3, size=n_eval)   # distribucion de colas pesadas: el resultado es libre de distribucion
        c_interp[s] = (new > np.percentile(band, 97.5)).sum()
        c_order[s] = (new > band[j - 1]).sum()
    assert (c_interp <= c_order).all()                                                    # superar la banda interpolada implica superar el estadistico de orden inferior
    for level in (0.95, 0.99):
        k = mod.max_detections_under_exchangeability(n_band, n_eval, level)
        assert (c_order <= k).mean() >= level - 0.005 and (c_order <= k - 1).mean() < level + 0.005   # exacta para el estadistico de orden
        assert (c_interp <= k).mean() >= level - 0.005                                    # cota (conservadora) para el percentil interpolado
    assert c_order.mean() / n_eval == pytest.approx((n_band - j + 1) / (n_band + 1), abs=0.001)     # media de Beta(n-j+1, j): probabilidad media de EXCEDER (no su complemento)
    naive = int(np.ceil(n_eval * 0.025 + 1.645 * np.sqrt(n_eval * 0.025 * 0.975)))          # binomial con la banda como si fuera conocida
    assert mod.max_detections_under_exchangeability(n_band, n_eval, 0.95) > naive            # estimar la banda añade dispersion: ignorarla haria fallar a un procedimiento correcto
    assert mod.max_detections_under_exchangeability(0, 10, 0.95) == 0


def test_max_detections_for_the_configured_design_is_29_of_500():
    """Con 500 replicas para la banda, 500 reservadas y Bonferroni sobre 8 tasas, la cota conservadora es 29 (posicion 487: Beta(14, 487), media 14/501 = 2,79 %)."""
    level = 1 - 0.05 / 8
    assert mod.max_detections_under_exchangeability(500, 500, level) == 29
    assert (500 - mod.band_order_statistic(500) + 1) / 501 == pytest.approx(0.027944, abs=1e-6)
    assert mod.max_detections_under_exchangeability(500, 500, 0.95) == 23                 # sin Bonferroni


def test_passing_g2_is_not_a_certificate_of_the_nominal_rate():
    """Segunda auditoria, H2. Con 29/500, una tasa verdadera del doble del nominal pasa el criterio el 82 % de las veces: no rechazar no es demostrar."""
    p = mod.criterion_non_rejection_probability
    assert p(29, 500, 0.05) == pytest.approx(0.8235, abs=5e-4)
    assert p(29, 500, 0.025) > 0.9999 and p(29, 500, 0.10) < 0.001
    probs = [p(29, 500, q) for q in mod.G2_POWER_RATES]
    assert all(a > b for a, b in zip(probs, probs[1:]))                                   # cuanto peor la calibracion, menos pasa
    rng = np.random.default_rng(3)
    assert (rng.binomial(500, 0.06, size=200000) <= 29).mean() == pytest.approx(p(29, 500, 0.06), abs=0.005)
    assert p(500, 500, 0.9) == pytest.approx(1.0) and p(0, 10, 0.0) == 1.0
    assert mod.g2_status(True) == mod.G2_STATUS_COMPATIBLE and "COMPATIBLE" in mod.G2_STATUS_COMPATIBLE and "SUPERADA" not in mod.G2_STATUS_COMPATIBLE
    assert mod.g2_status(False) == mod.G2_STATUS_INCOMPATIBLE and mod.g2_status(None) == mod.G2_STATUS_NOT_EVALUABLE


def test_restrict_block_ids_never_joins_bars_separated_by_an_excluded_bar():
    """Segunda auditoria, H5. Filtrar `block_ids[mask]` uniria dos tramos de un mismo bloque separados por una barra excluida."""
    block = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    mask = np.array([1, 1, 0, 1, 1, 1, 0, 1], dtype=bool)                                  # exclusion interior en cada bloque
    got = mod.restrict_block_ids(block, mask)
    assert got.tolist() == [0, 0, 1, 1, 2, 3]
    exc = np.array([0, 1, 1, 0, 0, 0], dtype=bool)                                         # barras 1 y 3 originales: NO son consecutivas (entre ellas esta la excluida)
    day = np.zeros(6, dtype=int)
    naive = observed_counts(exc, block[mask], day, 1, ks=(1, PRIMARY_WINDOW))
    fixed = observed_counts(exc, got, day, 1, ks=(1, PRIMARY_WINDOW))
    assert naive["b"].sum() == 1 and naive["in_runs2"].sum() == 2                          # el filtrado simple fabrica una racha de 2 que no existe
    assert fixed["b"].sum() == 0 and fixed["in_runs2"].sum() == 0
    # una exclusion solo al principio (quemado del estimador) o de bloques enteros no cambia la particion: es el caso de los datos reales
    prefix = np.array([0, 0, 1, 1, 1, 1, 1, 1], dtype=bool)
    r1, r0 = mod.restrict_block_ids(block, prefix), block[prefix]
    assert np.array_equal(r1[1:] == r1[:-1], r0[1:] == r0[:-1])
    assert mod.restrict_block_ids(block, np.zeros(8, dtype=bool)).size == 0
    assert np.array_equal(mod.restrict_block_ids(block, np.ones(8, dtype=bool)), block)


def test_replicate_statistics_use_blocks_that_respect_excluded_bars(monkeypatch):
    """El camino de las replicas pasa por `restrict_block_ids`: si la poblacion comun de una serie tuviera una exclusion interior, no se mediria ningun tiempo a traves de ella."""
    seen = {}
    real = mod.primary_statistics_from_flags

    def spy(flags, minute, year, block_ids, day, n_days, day_year, **kw):
        seen["block"] = np.asarray(block_ids).copy()
        return real(flags, minute, year, block_ids, day, n_days, day_year, **kw)

    monkeypatch.setattr(mod, "primary_statistics_from_flags", spy)
    trading_date, day, minute, year, segment = _synthetic_days(n_days=40, per_day=200)
    rng = np.random.default_rng(5)
    r = 1e-4 * rng.standard_normal(day.size)
    ctx = ReplicateContext(r_valid=r, trading_date=trading_date, segment=segment, minute=minute, year=year, block_ids=day.copy(), day=day, n_days=40,
                           day_year=np.array([d.year for d in trading_date[::200]]))
    primary_replicate_statistics(r, ctx)
    ex = build_exceedances(r, trading_date, segment, levels=(PRIMARY_LEVEL,), definitions=(PRIMARY_DEFINITION,))
    assert np.array_equal(seen["block"], mod.restrict_block_ids(day, ex.eligible))


# ---------------------------------------------------------------------------
# Segunda auditoria, H3: tercera familia de control (minuto x año) y regla de integracion
# ---------------------------------------------------------------------------

def _year_hour_world(seed, kind, days_per_year=30, per_day=240):
    """Mundo SIN dependencia temporal (retornos independientes). En `S2` cada año tiene su propia franja horaria agitada, con rafagas a horas fijas: heterogeneidad año x hora no separable."""
    years = mod.COMPLETE_YEARS
    n_days = days_per_year * len(years)
    dates = np.array([datetime.date(y, 1, 1) + datetime.timedelta(days=int(i)) for y in years for i in range(days_per_year)])
    day = np.repeat(np.arange(n_days), per_day)
    minute = np.tile(np.arange(per_day), n_days)
    yidx = np.repeat(np.repeat(np.arange(len(years)), days_per_year), per_day)
    eps = np.random.default_rng(seed).standard_normal(day.size)
    scale = 1.0
    if kind == "S2":
        scale = np.where(((minute // (per_day // len(years))) == yidx) & ((minute % 12) < 3), 5.0, 1.0)
    r = 1e-4 * scale * eps
    ctx = ReplicateContext(r_valid=r, trading_date=np.repeat(dates, per_day), segment=np.where(minute < per_day // 2, "seg_a", "seg_b").astype(object), minute=minute,
                           year=np.repeat(np.array([d.year for d in dates]), per_day), block_ids=day.copy(), day=day, n_days=n_days,
                           day_year=np.array([d.year for d in dates]), seed=100 + seed)
    return r, ctx


def test_task_kind_order_keeps_the_seeds_of_previous_replicates():
    """La posicion de cada tipo entra en la semilla: los tipos anteriores conservan su posicion y el nuevo va al final, para que sus replicas sean identicas a las de ejecuciones previas."""
    assert mod._TASK_KINDS[:7] == (NULL_FAMILY_PERMUTATION, NULL_FAMILY_CLOCK, VOL_FAMILY_EWMA_PRIMARY, VOL_FAMILY_EWMA_FAST, VOL_FAMILY_LOCAL,
                                   mod.KIND_FULL_BOOTSTRAP, mod.KIND_BERNOULLI_DIAG)
    assert mod._TASK_KINDS[7:] == (mod.NULL_FAMILY_MINUTE_YEAR,)
    assert mod.INDEPENDENCE_FAMILIES == mod.PREVIOUS_INDEPENDENCE_FAMILIES + (mod.NULL_FAMILY_MINUTE_YEAR,)


def test_minute_year_permutation_preserves_every_minute_year_group_and_only_breaks_the_order():
    r, ctx = _year_hour_world(0, "S2")
    out = generate_null_series(mod.NULL_FAMILY_MINUTE_YEAR, ctx, np.random.default_rng(1))
    g = mod.minute_year_groups(ctx.minute, ctx.year)
    assert np.unique(g).size == 240 * 5
    order = np.lexsort((r, g))
    order_out = np.lexsort((out, g))
    assert np.array_equal(r[order], out[order_out])                                       # mismo conjunto de valores en cada (minuto, año): distribucion identica, sin error
    assert not np.array_equal(r, out) and (r != out).mean() > 0.9                         # el orden temporal si cambia
    for groups in (ctx.minute, ctx.year):                                                 # por tanto tambien se conservan los margenes por minuto y por año
        assert np.allclose(np.bincount(np.unique(groups, return_inverse=True)[1], weights=out * out), np.bincount(np.unique(groups, return_inverse=True)[1], weights=r * r))
    # la permutacion solo por minuto NO conserva la estructura por año: mezcla los años
    clock = generate_null_series(NULL_FAMILY_CLOCK, ctx, np.random.default_rng(1))
    by_g = lambda x: np.bincount(np.unique(g, return_inverse=True)[1], weights=x * x)     # noqa: E731
    assert not np.allclose(by_g(clock), by_g(r), rtol=0.2)
    # dentro de un grupo el destino es uniforme: ninguna jornada queda en su sitio de forma sistematica
    same_day = (np.argsort(np.lexsort((out, g))) == np.argsort(np.lexsort((r, g))))
    assert same_day.mean() < 0.2


def test_minute_year_groups_small_groups_and_bad_input():
    minute = np.array([0, 0, 0, 1, 1, 5, 0, 0, 1])
    year = np.array([2021, 2021, 2021, 2021, 2021, 2021, 2022, 2022, 2022])
    g = mod.minute_year_groups(minute, year)
    assert len(set(g.tolist())) == 5 and g[0] == g[1] == g[2] and g[0] != g[6]
    rng = np.random.default_rng(0)
    x = np.arange(9.0)
    for _ in range(20):
        out = permute_within_groups(x, g, rng)
        assert out[5] == 5.0                                                               # un grupo de una sola barra queda fijo
        assert sorted(out[:3]) == [0.0, 1.0, 2.0] and sorted(out[6:8]) == [6.0, 7.0]
    desc = mod.describe_null_family_groups(minute, year).set_index("family")
    assert desc.loc[mod.NULL_FAMILY_MINUTE_YEAR, "n_groups"] == 5 and desc.loc[mod.NULL_FAMILY_MINUTE_YEAR, "n_groups_of_one_bar"] == 2
    assert desc.loc[NULL_FAMILY_CLOCK, "n_groups"] == 3 and desc.loc[NULL_FAMILY_PERMUTATION, "n_groups"] == 1
    assert desc.loc[mod.NULL_FAMILY_MINUTE_YEAR, "bars_in_small_groups"] == 9               # todos los grupos tienen menos de SMALL_GROUP_SIZE barras: se reporta, no se oculta
    with pytest.raises(ValueError):
        mod.minute_year_groups(np.array([1440]), np.array([2021]))
    with pytest.raises(ValueError):
        mod.null_family_group_codes(VOL_FAMILY_LOCAL, minute, year)


def test_null_hypotheses_are_nested_so_the_finest_family_is_valid_for_all():
    """Cada grupo (minuto, año) esta contenido en un unico grupo por minuto, y este en el grupo global: intercambiable en el grupo grande implica intercambiable en el pequeño."""
    _, ctx = _year_hour_world(0, "S0", days_per_year=4)
    fine = mod.null_family_group_codes(mod.NULL_FAMILY_MINUTE_YEAR, ctx.minute, ctx.year)
    mid = mod.null_family_group_codes(NULL_FAMILY_CLOCK, ctx.minute, ctx.year)
    coarse = mod.null_family_group_codes(NULL_FAMILY_PERMUTATION, ctx.minute, ctx.year)
    for a, b in ((fine, mid), (mid, coarse)):
        pairs = np.unique(np.stack([a, b], axis=1), axis=0)
        assert pairs.shape[0] == np.unique(a).size                                         # cada grupo fino cae en un solo grupo grueso
    assert np.unique(fine).size > np.unique(mid).size > np.unique(coarse).size == 1


def test_replicates_of_the_new_family_go_through_the_full_reestimation():
    """Sobre una replica de la nueva familia se re-estiman sigma/z, el estado de volatilidad, los umbrales por celda y la referencia: tasa del 1 % por cola y margenes conservados."""
    r, ctx = _year_hour_world(2, "S2")
    null = generate_null_series(mod.NULL_FAMILY_MINUTE_YEAR, ctx, np.random.default_rng(3))
    ex0 = build_exceedances(r, ctx.trading_date, ctx.segment, levels=(PRIMARY_LEVEL,), definitions=(PRIMARY_DEFINITION,))
    ex1 = build_exceedances(null, ctx.trading_date, ctx.segment, levels=(PRIMARY_LEVEL,), definitions=(PRIMARY_DEFINITION,), decile_fn=mod.causal_volatility_decile_fast)
    assert not np.allclose(ex0.z[ex0.eligible][:5000], ex1.z[ex1.eligible][:5000])           # z se recalcula con la serie permutada
    t0, t1 = ex0.thresholds[(SERIES_R, PRIMARY_DEFINITION, PRIMARY_LEVEL)][0], ex1.thresholds[(SERIES_R, PRIMARY_DEFINITION, PRIMARY_LEVEL)][0]
    assert not t0.equals(t1)                                                               # los umbrales por celda se re-estiman
    e = ex1.eligible
    for name in (SERIES_R, "z"):
        neg, pos = ex1.flags[(name, PRIMARY_DEFINITION, PRIMARY_LEVEL)]
        assert neg.mean() == pytest.approx(PRIMARY_LEVEL, abs=0.002) and pos.mean() == pytest.approx(PRIMARY_LEVEL, abs=0.002)
        cons = reference_conservation(neg, ctx.minute[e], ctx.year[e], kinds=(PRIMARY_NULL,)).iloc[0]
        assert abs(cons["total_error"]) < 1e-6 and cons["max_abs_error_by_year"] < 1e-6 and cons["max_abs_error_by_minute"] < 1e-6
    st = primary_replicate_statistics(null, ctx)
    assert all(np.isfinite(st[mod._key(s, sd, f"ratio_le_{PRIMARY_WINDOW}")]) for s in (SERIES_R, SERIES_Z) for sd in (SIDE_NEG, SIDE_POS))


def test_permutation_band_size_bound_is_exact_for_an_exchangeable_statistic():
    """Si el estadistico observado es intercambiable con las replicas que fijan la banda, la probabilidad de superarla esta entre (B-j)/(B+1) y (B-j+1)/(B+1): 2,59 %..2,79 % con 500 (el 2,5 % es nominal)."""
    assert mod.permutation_band_size_bound(500) == pytest.approx(14 / 501) and mod.permutation_band_size_bound(50) == pytest.approx(3 / 51)
    assert np.isnan(mod.permutation_band_size_bound(1))
    rng = np.random.default_rng(11)
    n_band, n_sim = 50, 200000
    x = rng.standard_t(3, size=(n_sim, n_band + 1))
    exceed = (x[:, 0] > np.percentile(x[:, 1:], 97.5, axis=1)).mean()
    j = mod.band_order_statistic(n_band)
    assert (n_band - j) / (n_band + 1) - 0.002 <= exceed <= mod.permutation_band_size_bound(n_band) + 0.002


def test_three_families_band_is_the_maximum_and_twelve_rates_enter_the_criterion():
    mean3 = {NULL_FAMILY_PERMUTATION: 1.0, NULL_FAMILY_CLOCK: 1.015, mod.NULL_FAMILY_MINUTE_YEAR: 1.12}
    reps = _null_replicates(n=400, mean=mean3, families=tuple(mean3))
    rates, bands, ok = summarize_calibration(reps)
    assert ok is True and (bands["n_families"] == 3).all()
    row = bands.loc[(bands["series"] == SERIES_R) & (bands["side"] == SIDE_NEG)].iloc[0]
    per_family = [row[f"{fam}|p97_5"] for fam in mean3]
    assert row["band_hi"] == pytest.approx(max(per_family)) and row["binding_family"] == mod.NULL_FAMILY_MINUTE_YEAR
    acc = rates.loc[rates["in_acceptance_criterion"].astype(bool)]
    assert len(acc) == 12 and set(acc["family"]) == set(mean3) and (acc["n_rates_in_criterion"] == 12).all()
    assert acc["acceptance_level"].iloc[0] == pytest.approx(1 - 0.05 / 12)
    assert (acc["acceptance_max_detections"] == mod.max_detections_under_exchangeability(200, 200, 1 - 0.05 / 12)).all()
    # con el maximo, las familias que no fijan la banda detectan MENOS que con su propia banda: el maximo nunca aumenta las falsas detecciones
    assert acc.loc[acc["family"] == NULL_FAMILY_PERMUTATION, "n_detections"].sum() == 0
    # el procedimiento anterior (2 familias), sobre las mismas replicas
    rates2, bands2, _ = summarize_calibration(reps, independence_families=mod.PREVIOUS_INDEPENDENCE_FAMILIES)
    assert (bands2["n_families"] == 2).all() and (bands2["band_hi"].to_numpy() < bands["band_hi"].to_numpy()).all()
    assert rates2["in_acceptance_criterion"].sum() == 8 and mod.NULL_FAMILY_MINUTE_YEAR not in set(rates2["family"])


def test_acceptance_limit_for_twelve_rates_is_30_of_500_and_is_verified_independently():
    """Limite de aceptacion con 3 familias x 4 colas = 12 tasas: cuantil 1 - 0,05/12 de la beta-binomial del estadistico de orden 487 (Beta(14, 487)). Se recalcula aqui con otra formula."""
    from math import comb, exp, lgamma
    level = 1 - 0.05 / 12
    assert mod.max_detections_under_exchangeability(500, 500, level) == 30
    a, b, n = 14.0, 487.0, 500

    def cdf(k):                                                                            # P(K <= k) con K | q ~ Bin(n, q), q ~ Beta(a, b): suma de C(n, i) B(i+a, n-i+b) / B(a, b)
        lb = lambda x, y: lgamma(x) + lgamma(y) - lgamma(x + y)                           # noqa: E731
        return sum(comb(n, i) * exp(lb(i + a, n - i + b) - lb(a, b)) for i in range(k + 1))

    assert cdf(30) >= level > cdf(29)
    assert cdf(29) >= 1 - 0.05 / 8                                                         # con 8 tasas el limite era 29: mas comparaciones -> limite algo mayor, por la misma regla
    rng = np.random.default_rng(5)                                                         # y por simulacion directa del estadistico de orden
    q = rng.beta(a, b, size=400000)
    assert (rng.binomial(n, q) <= 30).mean() == pytest.approx(cdf(30), abs=0.001)
    assert mod.criterion_non_rejection_probability(30, 500, 0.05) == pytest.approx(0.869, abs=0.002)   # sigue sin distinguir el doble del nominal


def test_previous_procedure_falsely_detects_clustering_under_year_by_hour_heterogeneity_and_the_new_one_does_not():
    """Segunda auditoria, H3, con la REGLA COMPLETA (banda + G2 + bootstrap con re-estimacion + `decide_th27`).

    Mundos de retornos INDEPENDIENTES. En `S2` la forma del perfil horario cambia de año: las dos familias anteriores mezclan los años y la regla
    afirma un agrupamiento que no existe; con la tercera familia la regla no lo afirma. En `S0` (homogeneo) ninguna de las dos lo afirma.
    """
    sets = {"ANTERIOR": mod.PREVIOUS_INDEPENDENCE_FAMILIES, "VIGENTE": mod.INDEPENDENCE_FAMILIES}
    for seed in (0, 1):
        r, ctx = _year_hour_world(seed, "S2")
        res = mod.evaluate_full_rule(r, ctx, n_calibration=40, n_boot_full=40, family_sets=sets)
        old, new = res["decisions"]["ANTERIOR"], res["decisions"]["VIGENTE"]
        assert old["claim_1"] == CLAIM_YES                                                 # falsa deteccion del procedimiento anterior
        assert new["claim_1"] == CLAIM_NO and new["verdict"] == mod.TH27_NONE
        assert new["claim_3"] == CLAIM_NOT_DEMONSTRABLE and old["claim_3"] == CLAIM_NOT_DEMONSTRABLE
        key = mod._key(SERIES_R, SIDE_NEG, f"ratio_le_{PRIMARY_WINDOW}")
        w = res["replicates"].loc[res["replicates"]["key"] == key].groupby("kind")["value"].mean()
        assert res["point"][key] > 2.0 and w[NULL_FAMILY_PERMUTATION] < 1.2 and w[NULL_FAMILY_CLOCK] < 1.2   # las familias anteriores no reproducen el nivel del mundo independiente
        assert abs(w[mod.NULL_FAMILY_MINUTE_YEAR] - res["point"][key]) < 0.8                # la nueva si
        b_old, b_new = res["bands"]["ANTERIOR"], res["bands"]["VIGENTE"]
        assert (b_new["band_hi"].to_numpy() >= b_old["band_hi"].to_numpy()).all()          # la banda vigente nunca es menos exigente
        assert set(b_new.loc[b_new["series"] == SERIES_R, "binding_family"]) == {mod.NULL_FAMILY_MINUTE_YEAR}
    r, ctx = _year_hour_world(0, "S0")
    res = mod.evaluate_full_rule(r, ctx, n_calibration=40, n_boot_full=40, family_sets=sets)
    assert res["decisions"]["ANTERIOR"]["claim_1"] == CLAIM_NO and res["decisions"]["VIGENTE"]["claim_1"] == CLAIM_NO


def test_calibration_without_replicates_is_not_evaluable():
    rates, bands, ok = summarize_calibration(EMPTY)
    assert ok is None and rates.empty
    only_even = _null_replicates(n=1)
    assert summarize_calibration(only_even)[2] is None                                   # sin mitad reservada no se declara nada


# ---------------------------------------------------------------------------
# Procedimiento completo: controles nulos, de solo volatilidad y positivos
# ---------------------------------------------------------------------------

def _ctx(r, n_days=160, per_day=500, seed=7):
    trading_date, day, minute, year, segment = _synthetic_days(n_days, per_day)
    day_year = np.where(np.arange(n_days) < n_days // 2, 2022, 2023)
    return ReplicateContext(r_valid=r, trading_date=trading_date, segment=segment, minute=minute, year=year, block_ids=day.copy(), day=day, n_days=n_days,
                            day_year=day_year, sigma_paths=volatility_paths(r), seed=seed)


def _clustered_volatility_returns(n_days=160, per_day=500, seed=3):
    """Retornos con perfil de reloj (apertura agitada) y volatilidad que cambia despacio y a saltos: independientes DADA la volatilidad."""
    rng = np.random.default_rng(seed)
    n = n_days * per_day
    minute = np.tile(np.arange(per_day), n_days)
    clock = 1.0 + 2.0 * np.exp(-minute / 25.0)
    slow = np.exp(0.6 * np.sin(2 * np.pi * np.arange(n) / (20 * per_day)))
    jumps = np.repeat(np.exp(rng.normal(scale=0.5, size=n_days * 5)), per_day // 5)       # cinco regimenes por jornada, con saltos bruscos
    return rng.standard_t(6, size=n) * 1e-4 * clock * slow * jumps


def test_null_families_destroy_dependence_and_volatility_families_preserve_it():
    r = _clustered_volatility_returns()
    ctx = _ctx(r)
    real = primary_replicate_statistics(r, ctx)
    key_r, key_z = f"{SERIES_R}|{SIDE_BOTH}|{STAT}", f"{SERIES_Z}|{SIDE_BOTH}|{STAT}"
    assert real[key_z] > 1.1                                                             # el EWMA rezagado deja agrupamiento residual en z (saltos bruscos)
    rng = np.random.default_rng(0)
    perm = generate_null_series(NULL_FAMILY_PERMUTATION, ctx, rng)
    clock = generate_null_series(NULL_FAMILY_CLOCK, ctx, rng)
    assert np.array_equal(np.sort(perm), np.sort(r)) and not np.array_equal(perm, r)
    for m in (0, 100, 499):                                                              # el control de reloj conserva la distribucion de cada minuto
        assert np.array_equal(np.sort(clock[ctx.minute == m]), np.sort(r[ctx.minute == m]))
    for null in (perm, clock):
        st = primary_replicate_statistics(null, ctx)
        assert abs(st[key_r] - 1) < 0.12 and abs(st[key_z] - 1) < 0.12                   # sin dependencia: el procedimiento completo no encuentra agrupamiento
    vol = generate_null_series(VOL_FAMILY_LOCAL, ctx, rng)
    sig = ctx.sigma_paths[VOL_FAMILY_LOCAL]
    assert np.allclose(np.sort((vol / sig)[ctx.minute == 100]), np.sort((r / sig)[ctx.minute == 100]))      # solo se barajan los residuos
    assert np.corrcoef(np.abs(vol), sig)[0, 1] > 0.3                                     # la senda de volatilidad se conserva
    with pytest.raises(ValueError):
        generate_null_series("OTRA", ctx, rng)


def test_volatility_only_world_reproduces_residual_clustering_in_z():
    """Control de 'solo volatilidad' con la senda rapida: los retornos son independientes dada la senda y aun asi z (EWMA 60) muestra agrupamiento. Por eso 'persiste en z' no prueba un fenomeno propio."""
    r = _clustered_volatility_returns()
    ctx = _ctx(r)
    key = f"{SERIES_Z}|{SIDE_BOTH}|{STAT}"
    rng = np.random.default_rng(1)
    local = np.mean([primary_replicate_statistics(generate_null_series(VOL_FAMILY_LOCAL, ctx, rng), ctx)[key] for _ in range(3)])
    slow = np.mean([primary_replicate_statistics(generate_null_series(VOL_FAMILY_EWMA_PRIMARY, ctx, rng), ctx)[key] for _ in range(3)])
    indep = np.mean([primary_replicate_statistics(generate_null_series(NULL_FAMILY_CLOCK, ctx, rng), ctx)[key] for _ in range(3)])
    assert local > 1.08 and local > slow and local > indep + 0.05
    assert abs(indep - 1) < 0.08


def test_volatility_paths_are_finite_positive_and_local_excludes_the_bar_itself():
    rng = np.random.default_rng(2)
    r = rng.normal(size=3000) * 1e-4
    r[1500] = 0.5                                                                        # un valor enorme aislado
    paths = volatility_paths(r)
    assert set(paths) == {VOL_FAMILY_EWMA_PRIMARY, VOL_FAMILY_EWMA_FAST, VOL_FAMILY_LOCAL}
    local = paths[VOL_FAMILY_LOCAL]
    assert np.isfinite(local).all() and (local > 0).all()
    assert local[1500] < 1e-3 and local[1499] > 0.05 and local[1501] > 0.05              # la barra no entra en su propia volatilidad; sus vecinas si la ven
    assert local[1500 + PRIMARY_WINDOW + 1] < 1e-3


def test_full_pipeline_positive_control_bursts_exceed_a_calibrated_band():
    """Control positivo extremo a extremo: con rafagas inyectadas, el cociente real supera la banda que produce el MISMO procedimiento sobre series nulas."""
    trading_date, day, minute, year, segment = _synthetic_days()
    rng = np.random.default_rng(11)
    r = rng.normal(size=day.size) * 1e-4
    for s in rng.choice(np.flatnonzero((minute > 10) & (minute < 480)), size=600, replace=False):
        r[s:s + 3] = np.array([8.0, -8.0, 8.0]) * 1e-4
    ctx = _ctx(r)
    tasks = [(fam, i) for fam in (NULL_FAMILY_PERMUTATION, NULL_FAMILY_CLOCK) for i in range(16)]
    reps = run_tasks(ctx, tasks, n_workers=1)
    _, bands, _ = summarize_calibration(reps)
    real = primary_replicate_statistics(r, ctx)
    b = bands.set_index(["series", "side"])
    for series in (SERIES_R, SERIES_Z):
        assert real[f"{series}|{SIDE_BOTH}|{STAT}"] > 1.5 * b.loc[(series, SIDE_BOTH), "band_hi"]
        assert 0.8 < b.loc[(series, SIDE_BOTH), "band_hi"] < 1.4


def test_run_tasks_is_deterministic_and_independent_of_the_number_of_workers():
    r = _clustered_volatility_returns(n_days=40, per_day=300)
    ctx = _ctx(r, n_days=40, per_day=300)
    tasks = [(NULL_FAMILY_CLOCK, 0), (NULL_FAMILY_CLOCK, 1), (VOL_FAMILY_EWMA_FAST, 0), (NULL_FAMILY_PERMUTATION, 3)]
    a = run_tasks(ctx, tasks, n_workers=1)
    b = run_tasks(ctx, list(reversed(tasks)), n_workers=1)
    pd.testing.assert_frame_equal(a, b)                                                  # el orden de ejecucion no cambia el resultado: cada replica tiene su semilla
    c = run_tasks(ctx, tasks, n_workers=2)
    pd.testing.assert_frame_equal(a, c)                                                  # ni el numero de procesos
    other = run_tasks(ReplicateContext(**{**ctx.__dict__, "seed": 99}), tasks, n_workers=1)
    assert not np.allclose(a["value"].to_numpy(), other["value"].to_numpy(), equal_nan=True)


# ---------------------------------------------------------------------------
# Bootstrap con re-estimacion (hallazgo C)
# ---------------------------------------------------------------------------

def test_weighted_cell_exceedances_equal_thresholds_on_the_materialized_resample():
    rng = np.random.default_rng(3)
    n = 6000
    x = rng.standard_t(4, size=n)
    cells = rng.integers(0, 5, size=n)
    w = rng.integers(0, 4, size=n).astype(float)
    order = np.lexsort((x, cells))
    bounds = np.searchsorted(cells[order], np.arange(6))
    cell_sorted = [order[bounds[c]:bounds[c + 1]] for c in range(5)]
    neg, pos = weighted_cell_exceedances(x, cell_sorted, w, 0.01)
    rep_idx = np.repeat(np.arange(n), w.astype(int))
    _, neg2, pos2, _, _ = compute_thresholds(x[rep_idx], cells[rep_idx].astype(str).astype(object), 0.01)
    assert np.array_equal(neg[rep_idx], neg2) and np.array_equal(pos[rep_idx], pos2)
    neg1, pos1 = weighted_cell_exceedances(x, cell_sorted, np.ones(n), 0.01)                   # pesos 1 = muestra original
    _, neg0, pos0, _, _ = compute_thresholds(x, cells.astype(str).astype(object), 0.01)
    assert np.array_equal(neg1, neg0) and np.array_equal(pos1, pos0)


def test_primary_statistics_with_day_weights_equal_the_statistics_of_repeated_days():
    n_days, per_day = 14, 200
    n = n_days * per_day
    day = np.repeat(np.arange(n_days), per_day)
    minute = np.tile(np.arange(per_day), n_days)
    year = np.repeat(np.where(np.arange(n_days) < 7, 2020, 2021), per_day)
    day_year = np.where(np.arange(n_days) < 7, 2020, 2021)
    exc = np.random.default_rng(4).random(n) < 0.05
    mult = np.array([2, 0, 1, 1, 3, 0, 1, 1, 1, 2, 0, 1, 1, 0], dtype=float)
    a = primary_statistics_from_flags({(SERIES_R, SIDE_NEG): exc}, minute, year, day, day, n_days, day_year, bar_weights=mult[day], day_weights=mult)
    rep_days = np.repeat(np.arange(n_days), mult.astype(int))
    idx = np.concatenate([np.flatnonzero(day == d) for d in rep_days])
    new_day = np.repeat(np.arange(rep_days.size), per_day)
    b = primary_statistics_from_flags({(SERIES_R, SIDE_NEG): exc[idx]}, minute[idx], year[idx], new_day, new_day, rep_days.size, day_year[rep_days])
    for key in (f"{SERIES_R}|{SIDE_NEG}|{STAT}", f"{SERIES_R}|{SIDE_NEG}|ratio_le_1", f"{SERIES_R}|{SIDE_NEG}|{STAT}_year_2020", f"{SERIES_R}|{SIDE_NEG}|n_exceed"):
        assert a[key] == pytest.approx(b[key], rel=1e-9), key


def test_full_bootstrap_reestimates_thresholds_and_reference_and_differs_from_the_conditional_interval():
    """El IC con re-estimacion y el condicional comparten el estadistico puntual pero NO son el mismo intervalo (aqui resulta mas estrecho: re-estimar el umbral fija la tasa en cada replica). Por eso se rotulan por separado."""
    r = _clustered_volatility_returns(n_days=60, per_day=300, seed=5)
    ctx = _ctx(r, n_days=60, per_day=300)
    ex = build_exceedances(r, ctx.trading_date, ctx.segment, levels=(PRIMARY_LEVEL,), definitions=(PRIMARY_DEFINITION,))
    e = ex.eligible
    cells = cell_codes(PRIMARY_DEFINITION, ctx.segment[e], ex.decile[e])
    codes, _ = pd.factorize(pd.Series(cells), sort=True)
    ctx.eligible, ctx.values_e = e, {SERIES_R: r[e], "z": ex.z[e]}
    for name in ctx.values_e:
        order = np.lexsort((ctx.values_e[name], codes))
        bounds = np.searchsorted(codes[order], np.arange(codes.max() + 2))
        ctx.cell_sorted[name] = [order[bounds[c]:bounds[c + 1]] for c in range(codes.max() + 1)]
    point = primary_replicate_statistics(r, ctx)
    reps = run_tasks(ctx, [(mod.KIND_FULL_BOOTSTRAP, i) for i in range(60)], n_workers=1)
    neg, pos = ex.flags[(SERIES_R, PRIMARY_DEFINITION, PRIMARY_LEVEL)]
    rows, _ = analyze_variant(neg | pos, ctx.minute[e], ctx.year[e], ctx.block_ids[e], ctx.day[e], 60, bootstrap_day_weights(60, 60, 0), ks=(1, PRIMARY_WINDOW))
    cond = pd.DataFrame(rows).set_index("statistic").loc[f"share_le_{PRIMARY_WINDOW}"]
    key = f"{SERIES_R}|{SIDE_BOTH}|{STAT}"
    tab = summarize_full_bootstrap(point, reps, {key: (cond["ratio_ci_lo"], cond["ratio_ci_hi"])})
    row = tab.loc[(tab["series"] == SERIES_R) & (tab["side"] == SIDE_BOTH) & (tab["statistic"] == STAT)].iloc[0]
    assert row["point"] == pytest.approx(cond["ratio"], rel=1e-9)                        # mismo estadistico puntual por los dos caminos
    assert row["ci_lo"] < row["point"] < row["ci_hi"] and row["n_boot"] == 60
    assert np.isfinite(row["ci_width_ratio_full_vs_conditional"]) and abs(row["ci_width_ratio_full_vs_conditional"] - 1.0) > 0.05
    w = mod._wide(reps, mod.KIND_FULL_BOOTSTRAP)
    assert (w[f"{SERIES_R}|{SIDE_BOTH}|n_exceed"].std() / w[f"{SERIES_R}|{SIDE_BOTH}|n_exceed"].mean()) < 0.1    # el numero de excedencias se re-estima en cada replica y queda cerca del 2 %
    assert "sigma_hat" in row["does_not_cover"] and "umbrales" in row["covers"]
    assert {"fraction_removed", "ratio_difference"} <= set(tab.loc[tab["series"] == "r_vs_z", "statistic"])     # la comparacion r/z tiene su propio IC pareado


def test_th06_reading():
    mostly = decide_th06(pd.Series({CAT_REAL: 800, CAT_UNDETERMINED: 150, CAT_BOUNDARY: 45, CAT_ROLL: 5}))
    assert mostly["verdict"] == TH06_MOSTLY_CLASSIFIABLE and mostly["n_extremes"] == 1000 and mostly["counts"][CAT_ERROR] == 0
    assert mostly["fraction_artefacts_and_errors"] == pytest.approx(0.05) and mostly["tails_reported_as_range"]
    high = decide_th06(pd.Series({CAT_REAL: 400, CAT_UNDETERMINED: 600}))
    assert high["verdict"] == TH06_HIGH_UNDETERMINED
    assert decide_th06(pd.Series({CAT_REAL: 500, CAT_UNDETERMINED: 500}))["verdict"] == TH06_HIGH_UNDETERMINED   # exactamente la mitad no es "la mayoria"
    assert decide_th06(pd.Series(dtype=float))["verdict"] == TH06_HIGH_UNDETERMINED                               # sin datos no se declara clasificable
    assert mostly["depends_on_accepting_operational_evidence"] and "NO_VERIFICADA" in mostly["basis"][CAT_REAL] and mostly["robust_to_rules"] is None


def _inventory_for_sensitivity(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    frac = rng.normal(loc=0.1, scale=0.6, size=n)
    inv = pd.DataFrame({
        "invalid_reason": "VALID", "bad_data": False, "bad_data_prev": False,
        "ff_category": rng.choice(["ACTIVA", "FLAT_AISLADA"], n, p=[0.98, 0.02]), "ff_category_prev": "ACTIVA",
        "adjacent_unknown_gap": rng.random(n) < 0.01, "next_bar_contiguous": True,
        "reversal_fraction_next_bar": frac, "reverts_next_bar": frac > 0.5, "low_volume": rng.random(n) < 0.03,
    })
    inv.loc[:19, "invalid_reason"] = "TRADING_DATE_BOUNDARY"
    return classify_extremes(inv)


def test_triage_sensitivity_bounds_variants_and_conservation():
    inv = _inventory_for_sensitivity()
    s = triage_sensitivity(inv)
    base = s.loc[s["kind"] == "VIGENTE"].iloc[0]
    assert base["n_undetermined"] == int((inv["category"] == CAT_UNDETERMINED).sum()) and base["n_real"] == int((inv["category"] == CAT_REAL).sum())
    assert (s["n_undetermined"] + s["n_real"] + s["n_structural"] == len(inv)).all()         # ninguna variante pierde ni duplica observaciones
    assert (s["n_structural"] == 20).all()                                                   # las variantes nunca reclasifican artefactos estructurales
    lo, hi = s.loc[s["variant"] == "sin_dudas_R4"].iloc[0], s.loc[s["variant"] == "sin_aceptar_evidencia_R5"].iloc[0]
    assert lo["n_undetermined"] == 0 and hi["n_real"] == 0 and hi["fraction_undetermined"] == pytest.approx(1980 / 2000)
    v = s.loc[s["kind"] == "VARIANTE_DE_REGLA"].set_index("variant")
    assert (v["n_undetermined"] <= hi["n_undetermined"]).all()
    assert v.loc["sin_REVIERTE_BARRA_SIGUIENTE", "n_undetermined"] < base["n_undetermined"]  # quitar una duda solo puede reducir los indeterminados
    assert v.loc["reversion_mayor_que_0.25", "n_undetermined"] > base["n_undetermined"] > v.loc["reversion_mayor_que_0.75", "n_undetermined"] > v.loc["reversion_mayor_que_1", "n_undetermined"]


def test_th06_reading_reports_whether_it_survives_the_rule_variants():
    inv = _inventory_for_sensitivity()
    s = triage_sensitivity(inv)
    res = decide_th06(inv["category"].value_counts(), s)
    lo, hi = res["undetermined_range_over_rule_variants"]
    assert res["verdict"] == TH06_MOSTLY_CLASSIFIABLE and res["robust_to_rules"] is True and 0 < lo < hi < 0.5
    fragile = s.copy()
    fragile.loc[fragile["variant"] == "reversion_mayor_que_0.25", "fraction_undetermined"] = 0.6
    assert decide_th06(inv["category"].value_counts(), fragile)["robust_to_rules"] is False   # si una variante cruza la mitad, la lectura NO es robusta


# ---------------------------------------------------------------------------
# Extremo a extremo
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path, research_files, holdout_files=None, boundary_utc="2099-01-01 00:00:00"):
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    rep, itm = "reports/mnq", "data/interim/mnq"
    config_yaml = {
        "instrument": "MNQ",
        "raw_data": {"dir": "data/raw/mnq", "file_pattern": "*.Last.txt", "separator": ";",
                     "columns": ["timestamp", "open", "high", "low", "close", "volume"],
                     "timestamp_format": "%Y%m%d %H%M%S", "timestamp_raw_timezone": "UTC"},
        "instrument_spec": {"tick_size": TICK, "tick_size_source": "test"},
        "holdout": {"boundary_utc": boundary_utc, "boundary_source": "test", "research_files": research_files, "holdout_files": holdout_files or []},
        "tda00": {"output_dir_reports": rep, "output_dir_interim": itm, "inventory_report": "TDA00_inventario.md", "violations_report": "TDA00_violaciones.csv",
                  "per_file_summary": "TDA00_resumen_por_archivo.csv", "bad_data_mask": "tda00_bad_data_mask.parquet"},
        "tda02": {"output_dir_reports": rep, "output_dir_interim": itm, "gaps_csv": "TDA02_huecos.csv"},
        "tda04": {"output_dir_reports": rep, "output_dir_interim": itm, "report_name": "TDA04_variables_analisis.md",
                  "variables_parquet_name": "tda04_variables_1m.parquet", "validity_mask_parquet_name": "tda04_return_validity_mask.parquet",
                  "losses_by_cause_csv": "TDA04_perdidas_por_causa.csv", "th07_r_vs_R_csv": "TDA04_th07_r_vs_R.csv"},
        "tda05": {"output_dir_reports": rep, "output_dir_interim": itm, "tick_variables_parquet_name": "tda05_tick_variables.parquet"},
        "tda06": {"output_dir_reports": rep, "output_dir_interim": itm, "segmentation_csv": "TDA06_segmentacion_propuesta.csv"},
        "tda13": {"output_dir_reports": rep, "n_boot": 15, "n_null_sim": 12, "n_calibration": 12, "n_volatility_control": 4, "n_boot_full": 12, "n_workers": 1,
                  "reference_conservation_csv": "TDA13_referencia_conservacion.csv", "false_detection_csv": "TDA13_tasas_falsa_deteccion.csv",
                  "full_bootstrap_csv": "TDA13_bootstrap_reestimacion.csv", "triage_sensitivity_csv": "TDA13_triage_sensibilidad.csv",
                  "volatility_control_csv": "TDA13_control_solo_volatilidad.csv", "audit_previous_csv": "TDA13_auditoria_resultados_previos.csv",
                  "report_name": "TDA13_extremos_inventario_agrupamiento.md", "extremes_csv": "TDA13_extremos.csv", "thresholds_csv": "TDA13_umbrales_por_celda.csv",
                  "triage_summary_csv": "TDA13_triage_resumen.csv", "clustering_csv": "TDA13_agrupamiento_vs_poisson.csv",
                  "gap_distribution_csv": "TDA13_tiempos_entre_excedencias.csv", "run_distribution_csv": "TDA13_longitud_rachas.csv",
                  "clustering_by_group_csv": "TDA13_agrupamiento_por_anio_segmento.csv", "hourly_csv": "TDA13_distribucion_horaria.csv",
                  "triage_impact_csv": "TDA13_efecto_triage.csv", "calibration_csv": "TDA13_calibracion_g2.csv",
                  "clustering_png": "TDA13_agrupamiento_crudo_vs_estandarizado.png", "gap_distribution_png": "TDA13_tiempos_entre_excedencias.png",
                  "hourly_png": "TDA13_distribucion_horaria.png", "triage_png": "TDA13_triage_categorias.png"},
    }
    config_path = config_dir / "mnq_snapshot.yaml"
    config_path.write_text(yaml.safe_dump(config_yaml), encoding="utf-8")
    return load_config(config_path)


def _prepare_fixture(config, n_days=90, bars_per_day=360, seed=0, start=datetime.date(2023, 10, 2)):
    """Dos años, dos contratos (un roll), una jornada con hueco de causa desconocida, una barra `bad_data`; volatilidad por regimenes y rafagas de extremos."""
    rng = np.random.default_rng(seed)
    rows, price, day_i, added = [], 15000.0, 0, 0
    gap_day, bad_day, roll_day = n_days // 3, (2 * n_days) // 3, n_days // 2
    while added < n_days:
        date = start + datetime.timedelta(days=day_i)
        day_i += 1
        if date.weekday() >= 5:
            continue
        base = pd.Timestamp(date) + pd.Timedelta(hours=13, minutes=30)      # 13:30 UTC
        level = 3e-4 * np.exp(0.7 * np.sin(added / 6.0))
        rets = rng.standard_t(5, size=bars_per_day) * level
        if added % 9 == 0:
            s = int(rng.integers(20, bars_per_day - 20))
            rets[s:s + 3] = np.array([9.0, -9.0, 9.0]) * level
        contract = "Z23" if added < roll_day else "H24"
        if added == roll_day:
            price += 180.0                                                   # salto de base en el roll
        minutes = np.arange(bars_per_day)
        if added == gap_day:
            minutes = np.concatenate([minutes[:100], minutes[100:] + 7])    # hueco de 7 minutos
        for b in range(bars_per_day):
            price = round(price * (1.0 + rets[b]) / TICK) * TICK
            rows.append({"timestamp": base + pd.Timedelta(minutes=int(minutes[b])), "trading_date": date, "contract": contract,
                         "source_file": "research_a.txt" if contract == "Z23" else "research_b.txt", "close": price,
                         "volume": int(rng.integers(1, 400)), "day_no": added, "bar_no": b})
        added += 1
    df = pd.DataFrame(rows)
    close = df["close"].to_numpy()
    prev_ok = (df["day_no"] == df["day_no"].shift(1)) & ((df["timestamp"] - df["timestamp"].shift(1)).dt.total_seconds() == 60)
    r = np.full(len(df), np.nan)
    r[1:] = np.log(close[1:] / close[:-1])
    valid = prev_ok.to_numpy()
    new_day = (df["day_no"] != df["day_no"].shift(1)).to_numpy()
    new_contract = (df["contract"] != df["contract"].shift(1)).to_numpy()
    reason = np.where(valid, "VALID", np.where(new_contract, "ROLL_BOUNDARY", np.where(new_day, "TRADING_DATE_BOUNDARY", "NON_CONSECUTIVE_MINUTE"))).astype(object)
    reason[0] = "FIRST_OBSERVATION"
    variables = pd.DataFrame({
        "timestamp": df["timestamp"], "source_file": df["source_file"], "contract": df["contract"], "trading_date": df["trading_date"],
        "segment_id": (df["contract"] != "Z23").astype(int), "open": close, "high": close + 0.5, "low": close - 0.5, "close": close,
        "volume": df["volume"], "r_1m": np.where(valid, r, np.nan),
    })
    validity = pd.DataFrame({"timestamp": df["timestamp"], "r_1m_valid": valid, "invalid_reason": reason})
    config.interim_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    variables.to_parquet(config.tda04_variables_parquet_path, index=False)
    validity.to_parquet(config.tda04_validity_mask_parquet_path, index=False)

    bad = np.zeros(len(df), dtype=bool)
    bad[np.flatnonzero((df["day_no"] == bad_day) & (df["bar_no"] == 200))] = True
    pd.DataFrame({"source_file": df["source_file"], "timestamp": df["timestamp"], "bad_data": bad}).to_parquet(config.bad_data_mask_path, index=False)
    ff = np.where(rng.random(len(df)) < 0.01, "FLAT_AISLADA", "ACTIVA")
    pd.DataFrame({"timestamp": df["timestamp"], "ff_category": ff}).to_parquet(config.tda05_tick_variables_parquet_path, index=False)
    g = df.loc[df["day_no"] == gap_day].reset_index(drop=True)
    pd.DataFrame([{"before_ts_utc": str(g["timestamp"].iloc[99]) + "+00:00", "after_ts_utc": str(g["timestamp"].iloc[100]) + "+00:00", "cause": "UNKNOWN"},
                  {"before_ts_utc": "2023-10-02 21:00:00+00:00", "after_ts_utc": "2023-10-03 13:30:00+00:00", "cause": "DAILY_MAINTENANCE"}]).to_csv(
        config.tda02_gaps_csv_path, index=False)
    cal = attach_calendar_fields(variables, validity)
    m0 = int(cal["minute_of_day"].min())
    pd.DataFrame([{"minute_of_day": m0 + bars_per_day // 2, "n_years_supporting": 5, "years_supporting": "2020,2021,2022,2023,2024", "stable": True}]).to_csv(
        config.tda06_segmentation_csv_path, index=False)
    return variables, validity


def _hash_inputs(config) -> dict:
    paths = [config.tda04_variables_parquet_path, config.tda04_validity_mask_parquet_path, config.bad_data_mask_path,
             config.tda05_tick_variables_parquet_path, config.tda02_gaps_csv_path, config.tda06_segmentation_csv_path]
    return {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tda13_e2e")
    config = _make_config(tmp, research_files=["research_a.txt", "research_b.txt"], holdout_files=["holdout.txt"])
    variables, validity = _prepare_fixture(config)
    before = _hash_inputs(config)
    result = run_tda13_analysis(config, verbose=False)
    return config, variables, validity, result, before


def test_e2e_populations_and_inventory_invariants(e2e):
    config, variables, validity, result, _ = e2e
    inv = result.inventory
    assert result.n_valid == int(validity["r_1m_valid"].sum()) and result.n_universe == len(variables) - 1
    assert 0 < result.n_eligible <= result.n_valid
    assert set(inv["category"]) <= set(CATEGORIES) and inv["timestamp_utc"].is_unique
    assert set(inv["direction"]) == {SIDE_NEG, SIDE_POS}
    neg, pos = inv["direction"] == SIDE_NEG, inv["direction"] == SIDE_POS
    assert (inv.loc[neg, "r"] < inv.loc[neg, "threshold_lo"]).all() and (inv.loc[pos, "r"] > inv.loc[pos, "threshold_hi"]).all()
    assert (inv["exceedance_ratio"] > 1).all()
    for col in ("timestamp_ny", "trading_date", "contract", "segment", "vol_decile_causal", "threshold_applied", "threshold_scope", "r_ticks", "z",
                "invalid_reason", "bad_data", "ff_category", "reverts_next_bar", "low_volume", "triage_rule", "triage_evidence", "clock_0830_window"):
        assert col in inv.columns, col
    cross = inv["population"] == mod.POP_CROSSING
    assert cross.any()
    assert inv.loc[cross, "category"].isin([CAT_ROLL, CAT_BOUNDARY, CAT_ERROR]).all()          # un cruce de frontera nunca es evento real ni indeterminado
    assert not inv.loc[~cross, "category"].isin([CAT_ROLL, CAT_BOUNDARY]).any()
    assert (inv.loc[inv["category"] == CAT_ROLL, "invalid_reason"] == "ROLL_BOUNDARY").all() and (inv["category"] == CAT_ROLL).sum() == 1
    elig = inv["population"] == mod.POP_ELIGIBLE
    assert (inv.loc[elig, "threshold_scope"] == PRIMARY_DEFINITION).all() and inv.loc[elig, "z"].notna().all()
    assert inv.loc[cross, "threshold_scope"].str.startswith(DEF_SEGMENT).all() and inv.loc[cross, "z"].isna().all()
    real = inv.loc[inv["category"] == CAT_REAL]
    assert len(real) and not real["reverts_next_bar"].any() and real["next_bar_contiguous"].all() and (real["invalid_reason"] == "VALID").all()
    assert result.th06["n_extremes"] == len(inv) and sum(result.th06["counts"].values()) == len(inv)


def test_e2e_inventory_matches_exceedance_flags(e2e):
    """El inventario de la poblacion comun son exactamente las excedencias primarias de r usadas en el agrupamiento."""
    _, _, _, result, _ = e2e
    c = result.cluster_table
    row = c.loc[(c["series"] == SERIES_R) & c["is_primary"] & (c["side"] == SIDE_BOTH) & (c["statistic"] == f"share_le_{PRIMARY_WINDOW}")].iloc[0]
    assert int(row["n_exceed"]) == int((result.inventory["population"] == mod.POP_ELIGIBLE).sum())
    thr = result.thresholds
    prim = thr.loc[thr["is_primary"] & (thr["series"] == SERIES_R)]
    assert int(prim["n"].sum()) == result.n_eligible
    assert int(prim["n_exceed_neg"].sum() + prim["n_exceed_pos"].sum()) == int(row["n_exceed"])
    assert (thr["label"] == "RETROSPECTIVO").all()


def test_e2e_cluster_tables_structure(e2e):
    _, _, _, result, _ = e2e
    c = result.cluster_table
    assert set(c["series"]) >= {SERIES_R, SERIES_Z} and set(c["definition"]) == {DEF_GLOBAL, DEF_SEGMENT, DEF_SEGMENT_X_VOL}
    assert set(c["level"]) == set(THRESHOLD_LEVELS) and set(c["side"]) == {SIDE_NEG, SIDE_POS, SIDE_BOTH}
    assert set(c.loc[c["level"] == PRIMARY_LEVEL, "reference"]) == {NULL_CLOCK_YEAR, NULL_CLOCK, NULL_HOMOGENEOUS, NULL_CLOCK_YEAR_LEGACY}
    assert (c["ci_type"].str.startswith("CONDICIONAL")).all() and "distinguishable_preaudit_criterion" in c.columns
    prim = c.loc[c["is_primary"]]
    assert prim["null_sim_p97_5"].notna().all() and (prim["n_null_sim"] == 12).all()
    both = c.loc[(c["side"] == SIDE_BOTH) & (c["statistic"] == f"share_le_{PRIMARY_WINDOW}") & (c["level"] == PRIMARY_LEVEL) & (c["reference"] == PRIMARY_NULL)]
    r_n = both.loc[both["series"] == SERIES_R].set_index("definition")["n_exceed"]
    z_n = both.loc[both["series"] == SERIES_Z].set_index("definition")["n_exceed"]
    assert (abs(r_n - z_n) <= 0.02 * r_n).all()                # tasas de excedencia comparables entre r y z
    assert (c["label"] == "RETROSPECTIVO").all()
    g, rn = result.gap_distribution, result.run_distribution
    for (_, _, _, _), sub in g.groupby(["series", "definition", "side", "reference"]):
        assert sub["share_observed"].sum() == pytest.approx(1.0) and sub["share_expected"].sum() == pytest.approx(1.0)
    assert set(rn["run_length"]) == {"1", "2", "3", "4", f">={RUN_LENGTH_MAX}"}
    bg = result.cluster_by_group
    assert set(bg["grouping"]) == {"YEAR", "SEGMENT"} and set(bg["group"]) >= {"2023", "2024"}
    h = result.hourly
    tot = h.groupby(["series", "definition", "side"])["share_of_exceedances"].sum()
    assert np.allclose(tot, 1.0)
    reps = result.null_replicates
    assert set(reps["kind"]) == set(mod.ALL_FAMILIES) | {mod.KIND_FULL_BOOTSTRAP, mod.KIND_BERNOULLI_DIAG}
    assert reps.loc[reps["kind"] == NULL_FAMILY_CLOCK, "replicate"].nunique() == 12 and reps.loc[reps["kind"] == VOL_FAMILY_LOCAL, "replicate"].nunique() == 4
    assert reps.loc[reps["kind"] == mod.KIND_FULL_BOOTSTRAP, "replicate"].nunique() == 12
    th27 = result.th27
    assert th27["verdict"] in {TH27_NONE, TH27_EXPLAINED, TH27_PERSISTS, TH27_INCONCLUSIVE} and th27["claim_3"] == CLAIM_NOT_DEMONSTRABLE
    assert th27["claim_1"] in {CLAIM_YES, CLAIM_NO, CLAIM_INCONCLUSIVE}


def test_e2e_rule_inputs_are_the_same_numbers_as_the_published_table(e2e):
    """Los datos reales pasan por el mismo camino que las replicas: el cociente que decide es el de la tabla."""
    _, _, _, result, _ = e2e
    c = result.cluster_table
    prim = c.loc[c["is_primary"] & (c["statistic"] == f"share_le_{PRIMARY_WINDOW}")]
    assert len(prim) == 6
    for _, row in prim.iterrows():
        assert result.primary_point[f"{row['series']}|{row['side']}|{STAT}"] == pytest.approx(row["ratio"], rel=1e-9)
    fb = result.full_bootstrap
    assert set(fb["series"]) == {SERIES_R, SERIES_Z, "r_vs_z"}
    main = fb.loc[fb["statistic"] == STAT]
    assert (main["n_boot"] == 12).all()
    assert main["conditional_ci_lo"].notna().all() and (main["ci_lo"] <= main["ci_hi"]).all()


def test_e2e_reference_conservation_and_calibration_outputs(e2e):
    _, _, _, result, _ = e2e
    cons = result.reference_conservation.set_index(["series", "side", "reference"])
    prim = cons.xs(NULL_CLOCK_YEAR, level="reference")
    assert (prim["total_error"].abs() < 1e-6).all() and (prim["max_abs_error_by_minute"] < 1e-6).all() and (prim["max_abs_error_by_year"] < 1e-6).all()
    assert (cons.xs(NULL_CLOCK, level="reference")["max_abs_error_by_minute"] < 1e-6).all()
    bands, rates = result.calibration_bands, result.calibration_rates
    assert len(bands) == 6 and bands["band_hi"].notna().all() and (bands["band_hi"] > bands["band_lo"]).all()
    ind = rates.loc[rates["family_type"] == "SIN_DEPENDENCIA"]
    assert (ind["n_replicates"] == 6).all()                                               # solo la mitad reservada mide; la otra fija la banda
    assert set(rates["family_type"]) == {"SIN_DEPENDENCIA", "SOLO_VOLATILIDAD"}
    assert rates["rate"].between(0, 1).all() and (rates["rate_ci_lo"] <= rates["rate"]).all() and (rates["rate"] <= rates["rate_ci_hi"]).all()
    diag = result.bernoulli_diagnostic
    assert {mod.DIAG_BERNOULLI_KNOWN, mod.DIAG_BERNOULLI_REFIT} <= set(diag["simulation"])
    ev = result.th27["volatility_only_evidence"]
    assert set(ev["family"]) == set(mod.VOLATILITY_FAMILIES) and set(ev["side"]) == {SIDE_NEG, SIDE_POS}
    sens = result.triage_sensitivity
    assert (sens["n_undetermined"] + sens["n_real"] + sens["n_structural"] == len(result.inventory)).all()
    assert set(result.inventory["category_basis"]) <= set(mod.TH06_BASIS.values())
    assert result.inventory.loc[result.inventory["category"] == CAT_REAL, "category_basis"].str.startswith("OPERATIVA_NO_VERIFICADA").all()


def test_e2e_triage_impact_scenarios_are_traceable(e2e):
    _, variables, validity, result, _ = e2e
    imp, inv = result.triage_impact, result.inventory
    n = imp.loc[imp["statistic"] == "n"].set_index("scenario")["value"]
    assert n["UNIVERSO_COMPLETO"] == len(variables) - 1 and n["SOLO_VALIDOS"] == int(validity["r_1m_valid"].sum())
    for cat in CATEGORIES:
        assert n["UNIVERSO_COMPLETO"] - n[f"UNIVERSO_sin_{cat}"] == int((inv["category"] == cat).sum())
    valid_inv = inv["invalid_reason"] == "VALID"
    assert n["SOLO_VALIDOS"] - n[f"VALIDOS_sin_{CAT_UNDETERMINED}"] == int((valid_inv & (inv["category"] == CAT_UNDETERMINED)).sum())
    assert n["SOLO_VALIDOS"] - n["VALIDOS_sin_EXTREMOS"] == int(valid_inv.sum())
    ref = compute_moments_quantiles(variables.loc[validity["r_1m_valid"].to_numpy(), "r_1m"].to_numpy())
    k = imp.loc[(imp["scenario"] == "SOLO_VALIDOS") & (imp["statistic"] == "kurtosis_excess"), "value"].iloc[0]
    assert k == pytest.approx(ref["kurtosis_excess"], rel=1e-10)
    kur = imp.loc[imp["statistic"] == "kurtosis_excess"].set_index("scenario")["value"]
    assert kur["VALIDOS_sin_EXTREMOS"] < kur["SOLO_VALIDOS"]     # quitar todos los extremos adelgaza la cola
    assert kur["UNIVERSO_COMPLETO"] > kur["SOLO_VALIDOS"]        # el salto del roll infla la curtosis del universo
    assert set(imp["series"]) == {SERIES_R, SERIES_Z}


def test_e2e_inputs_untouched_and_run_is_reproducible(e2e):
    config, _, _, result, before = e2e
    assert _hash_inputs(config) == before
    again = run_tda13_analysis(config, verbose=False)
    for attr in ("inventory", "thresholds", "cluster_table", "gap_distribution", "run_distribution", "cluster_by_group", "hourly", "triage_impact", "null_replicates",
                 "triage_summary", "triage_sensitivity", "calibration_rates", "calibration_bands", "full_bootstrap", "reference_conservation", "bernoulli_diagnostic"):
        pd.testing.assert_frame_equal(getattr(result, attr), getattr(again, attr), check_exact=False, rtol=1e-12)
    assert result.th27["verdict"] == again.th27["verdict"] and result.th27["claim_1"] == again.th27["claim_1"] and result.th06 == again.th06
    assert _hash_inputs(config) == before


def test_e2e_persist_artifacts_and_report(e2e):
    from ohlcv_dataroad.ingest.run_tda13 import ARTIFACT_PATH_ATTRS, TH27_DISPLAY, persist_artifacts

    config, _, _, result, before = e2e
    written = persist_artifacts(result, config, t0=0.0, run_command="test")
    assert len(written) == len(ARTIFACT_PATH_ATTRS) == 20 and all(Path(p).exists() and Path(p).stat().st_size > 0 for p in written)
    assert not config.tda13_audit_previous_csv_path.exists()          # la foto previa es una entrada opcional: persistir nunca la crea ni la reescribe
    csv = pd.read_csv(config.tda13_extremes_csv_path)
    assert len(csv) == len(result.inventory) and "category" in csv.columns and set(csv["category"]) <= set(CATEGORIES)
    text = config.tda13_report_path.read_text(encoding="utf-8")
    for needle in ("TH06", "TH27", "RETROSPECTIVO", "exploratorio", "NUNCA se edita a mano", "Parte B (EVT) no se ejecuta", "holdout",
                   "escritos antes de calcular su efecto", "Calibración de la maquinaria (G2)", "Limitaciones", "Modo sencillo",
                   "ningún extremo se atribuye a un evento", "PENDIENTE DE SEGUNDA AUDITORÍA", "no cerrada", "clasificación operativa",
                   "no es un evento verificado", "NO IDENTIFICADO POR ESTE PROCEDIMIENTO", "Lo que NO se afirma", "la separación podría ser posible", "Criterio de aceptación de G2", "IC condicional", "IC con re-estimación",
                   "Ninguno de los dos cubre", "Ausencia de prueba no es prueba de ausencia", "hallazgo A", "hallazgo B", "hallazgo C", "hallazgo D",
                   "sólo volatilidad", "No se encontró la foto de resultados previos"):
        assert needle in text, needle
    assert f"### TH27 — agrupamiento de extremos: `{TH27_DISPLAY[result.th27['verdict']]}`" in text
    assert f"### TH06 — triage de extremos: `{result.th06['verdict']}` (clasificación operativa)" in text
    cal = pd.read_csv(config.tda13_calibration_csv_path)
    assert set(cal["kind"]) == set(mod.ALL_FAMILIES)                  # el CSV de calibracion contiene las replicas de las familias de control
    fd = pd.read_csv(config.tda13_false_detection_csv_path)
    assert {"false_detection_rate", "calibrated_band", "band_diagnostic"} == set(fd["table"])
    assert "p-valor" not in text and "significativ" not in text      # G5
    assert _hash_inputs(config) == before                             # persistir no toca las entradas


def test_e2e_holdout_guard_blocks_overlap(tmp_path):
    config = _make_config(tmp_path, research_files=["research_a.txt", "research_b.txt"], holdout_files=["research_b.txt"])
    _prepare_fixture(config, n_days=12)
    with pytest.raises(HoldoutIsolationError):
        run_tda13_analysis(config, verbose=False)


def test_e2e_holdout_boundary_is_enforced(tmp_path):
    config = _make_config(tmp_path, research_files=["research_a.txt", "research_b.txt"], holdout_files=["holdout.txt"], boundary_utc="2023-10-10 00:00:00")
    _prepare_fixture(config, n_days=12)
    with pytest.raises(HoldoutIsolationError):
        run_tda13_analysis(config, verbose=False)


def test_missing_quality_evidence_stops_the_stage(tmp_path):
    config = _make_config(tmp_path, research_files=["research_a.txt", "research_b.txt"], holdout_files=["holdout.txt"])
    _prepare_fixture(config, n_days=12)
    bad = pd.read_parquet(config.bad_data_mask_path)
    bad.iloc[:-5].to_parquet(config.bad_data_mask_path, index=False)          # faltan filas de la mascara de TDA-00
    with pytest.raises(MissingEvidenceError):
        mod.load_universe(config)
    bad.to_parquet(config.bad_data_mask_path, index=False)
    ticks = pd.read_parquet(config.tda05_tick_variables_parquet_path)
    ticks.iloc[1:].to_parquet(config.tda05_tick_variables_parquet_path, index=False)
    with pytest.raises(MissingEvidenceError):
        mod.load_universe(config)


def test_bad_data_bar_is_classified_as_error_when_extreme(tmp_path):
    """Si una barra marcada `bad_data` por TDA-00 produce un extremo, R1 decide `ERROR_DATO` aunque el retorno sea valido."""
    config = _make_config(tmp_path, research_files=["research_a.txt", "research_b.txt"], holdout_files=["holdout.txt"])
    _prepare_fixture(config, n_days=12)
    uni = mod.load_universe(config)
    flagged = uni.loc[uni["bad_data"] | uni["bad_data_prev"]]
    out = classify_extremes(flagged)
    assert (out["category"] == CAT_ERROR).all() and (out["triage_rule"] == "R1").all()


# --- Cierre documental: secciones 17 (tiempos) y 20 (estado) -------------------------------------------------

def _closure_file(tmp_path, stage, start="2026-10-13 00:05:34", end="2026-10-13 09:58:55"):
    path = tmp_path / "cierre.yaml"
    path.write_text(yaml.safe_dump({"estado": "CERRADA TRAS AUDITORÍA INDEPENDIENTE", "detalle": "cierre metodológico", "auditoria": "Auditoría independiente: conforme.",
                                    "incidencias": [{"tipo": "Suspensión del sistema operativo", "inicio": start, "fin": end, "etapa_afectada": stage,
                                                     "evidencia": "registro del sistema", "efecto": "ninguno"}]}, allow_unicode=True), encoding="utf-8")
    return path


def test_timing_section_separates_elapsed_from_effective_time_when_the_system_was_suspended(tmp_path):
    import ohlcv_dataroad.ingest.run_tda13 as run

    rows, tail = [("etapa uno", 35604.0), ("etapa dos", 85.1), ("etapa tres", 5337.0)], " CPU, 6 proceso(s)."
    plain = run.timing_section(rows, 41026.1, tail)
    assert "| etapa uno | 35604.0s |" in plain and "**Tiempo total (análisis + escritura): 41026.1s (~683.8 min).** CPU, 6 proceso(s)." in plain
    assert "Incidencias" not in plain and run.status_line() == run.DEFAULT_STATUS_LINE          # sin archivo de cierre, el texto de siempre
    closure = run.load_closure(_closure_file(tmp_path, "etapa uno"))
    text = run.timing_section(rows, 41026.1, tail, closure)
    assert "| etapa uno | 35604.0s (incluye ≈ 35601 s con el sistema parado; efectivo ≈ 3 s) |" in text and "| etapa dos | 85.1s |" in text
    assert "Tiempo transcurrido (análisis + escritura): 41026.1s (~683.8 min), de los cuales ≈ 35601 s (~593.4 min) con el sistema parado." in text
    assert "Tiempo efectivo aproximado: ≈ 5425 s (~90.4 min)." in text and "NO es tiempo de cálculo" in text and "Tiempo total" not in text
    assert "`CERRADA TRAS AUDITORÍA INDEPENDIENTE` (cierre metodológico)" in run.status_line(closure) and "PENDIENTE DE SEGUNDA" not in run.status_line(closure)
    with pytest.raises(ValueError):                                                              # la incidencia debe referirse a una etapa del informe
        run.timing_section(rows, 41026.1, tail, run.load_closure(_closure_file(tmp_path, "etapa inexistente")))
    with pytest.raises(ValueError):                                                              # ... y no puede durar mas que su etapa
        run.timing_section(rows, 41026.1, tail, run.load_closure(_closure_file(tmp_path, "etapa dos")))
    with pytest.raises(ValueError):                                                              # fin anterior al inicio
        run.load_closure(_closure_file(tmp_path, "etapa uno", start="2026-10-13 10:00:00", end="2026-10-13 09:00:00"))


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _closure_with_hashes(run, tmp_path, stage, report):
    """Archivo de cierre de prueba con los SHA-256 del informe original (el que hay en disco) y del cerrado (el que produce la enmienda)."""
    path = _closure_file(tmp_path, stage)
    closed = hashlib.sha256(run.amended_report_bytes(Path(report).read_bytes(), run.load_closure(path))).hexdigest()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["informe"] = {"sha256_original": _sha(report), "sha256_cerrado": closed}
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return run.load_closure(path)


@pytest.fixture()
def amendable(e2e, tmp_path):
    """Informe de prueba con una primera etapa de 35604,0 s (como la ejecucion suspendida), los demas artefactos y sus hashes."""
    import dataclasses

    import ohlcv_dataroad.ingest.run_tda13 as run

    config, _, _, result, _ = e2e
    written = run.persist_artifacts(result, config, t0=0.0, run_command="test")
    report = config.tda13_report_path
    stage = next(name for name in result.stage_timings if not name.startswith("_"))
    long_result = dataclasses.replace(result, stage_timings={**result.stage_timings, stage: 35604.0})
    report.write_text(run.render_report(long_result, config, 41026.1, "test"), encoding="utf-8")
    others = {Path(p): _sha(p) for p in written if Path(p) != report}
    return run, config, report, stage, long_result, others


def test_e2e_amend_report_rewrites_only_sections_17_and_20_and_no_other_artifact(amendable, tmp_path):
    run, config, report, stage, long_result, others = amendable
    before = report.read_text(encoding="utf-8")
    with pytest.raises(ValueError):                                                              # sin los dos SHA-256 declarados no se enmienda
        run.amend_report(report, run.load_closure(_closure_file(tmp_path, stage)))
    assert report.read_text(encoding="utf-8") == before
    closure = _closure_with_hashes(run, tmp_path, stage, report)
    original_sha, closed_sha = run.closure_report_hashes(closure)
    assert _sha(report) == original_sha
    assert run.amend_report(report, closure) is True and _sha(report) == closed_sha
    after = report.read_text(encoding="utf-8")
    i17, i18, i20 = (before.index(f"\n{h}\n") for h in (run.TIMING_HEADING, run.AFTER_TIMING_HEADING, run.STATUS_HEADING))
    a17, a18, a20 = (after.index(f"\n{h}\n") for h in (run.TIMING_HEADING, run.AFTER_TIMING_HEADING, run.STATUS_HEADING))
    assert before[:i17] == after[:a17] and before[i18:i20] == after[a18:a20]                    # todo lo anterior a §17 y de §18 a §19: identico
    assert before[i20:].split(run.DEFAULT_STATUS_LINE) == after[a20:].split(run.status_line(closure))   # en §20 solo cambia la linea de estado
    assert "PENDIENTE DE SEGUNDA AUDITORÍA" not in after and "CERRADA TRAS AUDITORÍA INDEPENDIENTE" in after
    assert "Tiempo efectivo aproximado: ≈ 5425 s (~90.4 min)." in after and "--amend-report" in after and "sin repetir ningún cálculo" in after
    assert "`STOP-13` | sin cambios respecto de TDA-12 (no activado)" in after and "no ejecutada; no autorizada" in after
    assert {p: _sha(p) for p in others} == others                                               # ningun otro artefacto cambia
    assert run.amend_report(report, closure) is False and _sha(report) == closed_sha            # idempotente: el informe cerrado exacto se acepta y no se toca
    # Una ejecucion completa con el mismo archivo de cierre produce las mismas secciones, salvo la frase de procedencia de la enmienda.
    full = run.render_report(long_result, config, 41026.1, "test", closure)
    provenance = after[after.index(" Las secciones 17 y 20 se regeneraron"):after.index("\n\n| Incidencia |")]
    assert full == after.replace(provenance, "")


@pytest.mark.parametrize("state", ["original", "cerrado"])
@pytest.mark.parametrize("tamper", ["duracion_de_etapa", "linea_stop_13", "espacio_final"])
def test_e2e_amend_report_rejects_any_manual_alteration_of_the_report(amendable, tmp_path, state, tamper):
    """Auditoria del cierre: la validacion por fragmentos aceptaba una duracion retocada con el mismo formato, o una linea de la seccion 20 cambiada."""
    run, config, report, stage, _, others = amendable
    closure = _closure_with_hashes(run, tmp_path, stage, report)
    if state == "cerrado":
        assert run.amend_report(report, closure) is True
    good = report.read_bytes()
    text = good.decode("utf-8")
    if tamper == "duracion_de_etapa":                                                           # mismo formato, otra cifra
        assert text.count("| 35604.0s") == 1
        bad = text.replace("| 35604.0s", "| 35603.9s")
    elif tamper == "linea_stop_13":                                                             # contenido de la seccion 20 posterior a la linea de estado
        line = "| `STOP-13` | sin cambios respecto de TDA-12 (no activado) |"
        assert text.count(line) == 1
        bad = text.replace(line, "| `STOP-13` | activado |")
    else:
        bad = text + " "
    report.write_bytes(bad.encode("utf-8"))
    tampered_sha = _sha(report)
    with pytest.raises(RuntimeError, match="ni el original ni el cerrado"):
        run.amend_report(report, closure)
    assert _sha(report) == tampered_sha                                                          # rechazado sin escribir nada
    assert {p: _sha(p) for p in others} == others
    report.write_bytes(good)
    assert run.amend_report(report, closure) is (state == "original")                            # el contenido exacto si se acepta


def test_e2e_amend_report_refuses_when_the_result_would_not_be_the_registered_closed_report(amendable, tmp_path):
    run, config, report, stage, _, others = amendable
    closure = _closure_with_hashes(run, tmp_path, stage, report)
    closure["informe"]["sha256_cerrado"] = "0" * 64                                              # hash del informe cerrado mal registrado
    before = _sha(report)
    with pytest.raises(RuntimeError, match="no reproduce el informe cerrado"):
        run.amend_report(report, closure)
    assert _sha(report) == before and {p: _sha(p) for p in others} == others


def test_closure_file_of_the_repository_declares_both_report_hashes():
    import ohlcv_dataroad.ingest.run_tda13 as run

    closure = run.load_closure(Path(__file__).resolve().parents[1] / "configs" / "tda13_cierre.yaml")
    original, closed = run.closure_report_hashes(closure)
    assert original == "307825f19349ceb34d53f5d16a678c0a2b0384f58df38920d3b749a376873c25"
    assert closed == "d790eac0016e698805bca2d7d526242be346631e84ebe6733905ee62c017b1cf"
    assert closure["estado"] == "CERRADA TRAS AUDITORÍA INDEPENDIENTE" and len(closure["incidencias"]) == 1
