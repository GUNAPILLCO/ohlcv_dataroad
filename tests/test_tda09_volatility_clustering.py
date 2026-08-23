"""Tests de TDA-09 -- dependencia en magnitud / volatility clustering (TH19-TH21).

No se reproducen aqui los numeros exactos del conjunto de investigacion
real (esos viven en ``reports/mnq/TDA09_volatility_clustering.md``). Las
funciones reutilizadas SIN MODIFICAR de TDA-08 CLOSED (``compute_acf``,
``compute_block_ids``, ``bootstrap_rho``, ``acf_by_group``,
``g2_permutation_null_by_minute``, ``g2_synthetic_null_moment_check``,
``annotate_portmanteau_calibration``) no se re-testean aqui en detalle --
ya tienen su propia suite en ``test_tda08_linear_mean_dependence.py``
(50 tests) que sigue pasando intacta. Este archivo cubre exclusivamente
la logica NUEVA de TDA-09.
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
from ohlcv_dataroad.ingest.tda07_marginal_distribution import TimestampAlignmentError
from ohlcv_dataroad.ingest.tda08_linear_mean_dependence import compute_acf, compute_block_ids
from ohlcv_dataroad.ingest.tda09_volatility_clustering import (
    ARCH_LM_ORDERS,
    CLOCK_FLATNESS_RATIO_THRESHOLD,
    G2_FULL_LAGS,
    G2_LAGS,
    SMProxyMismatchError,
    STOP9_FRACTION_REMOVED_THRESHOLD,
    block_relative_position,
    build_log_hl_population,
    build_log_hl_tilde,
    calibrate_engle_lm,
    clock_attribution,
    clock_profile_flatness,
    decay_form_diagnostic,
    decide_stop9,
    dependence_energy,
    engle_lm_statistic,
    g2_global_permutation_null,
    mean_removal_sensitivity,
    run_tda09_analysis,
    same_clock_next_trading_day,
    verify_s_m_is_log_hl_proxy,
)

TICK = 0.25


# ---------------------------------------------------------------------------
# Fixtures sinteticas (mismo estilo que test_tda08_linear_mean_dependence.py)
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


def _synthetic_multi_day_rows(
    n_days: int, start: datetime.date, bars_per_day: int = 80, seed: int = 0,
    return_gen=None,
) -> list[dict]:
    """Genera ``n_days`` jornadas de ``bars_per_day`` barras, saltando fines de semana.

    ``return_gen(rng, day_i, bar_i) -> float`` opcional -- si se pasa,
    reemplaza el retorno gaussiano por defecto (para construir escenarios
    con patron de reloj y/o clustering genuino).
    """
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
        "tda09": {
            "output_dir_reports": "reports/mnq",
            "n_boot": 5, "n_perm": 5, "n_perm_arch_lm": 5,
            "report_name": "TDA09_volatility_clustering.md",
            "acf_csv": "TDA09_acf_magnitud.csv",
            "bootstrap_ci_csv": "TDA09_bootstrap_ci.csv",
            "clock_attribution_csv": "TDA09_clock_attribution.csv",
            "persistence_by_year_csv": "TDA09_persistencia_por_anio.csv",
            "persistence_by_segment_csv": "TDA09_persistencia_por_segmento.csv",
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

    s_m_table = pd.DataFrame({
        "minute_of_day": list(s_m_map.keys()), "n": 10, "point": list(s_m_map.values()),
        "s_m": list(s_m_map.values()), "proxy": "log_hl", "label": "RETROSPECTIVO",
    })
    s_m_table.to_parquet(config.tda06_s_m_parquet_path, index=False)


def _prepare_full_fixture(config, n_days=15, start=datetime.date(2024, 1, 8), cutoffs=(600, 900), return_gen=None, s_m_by_minute=None):
    rows = _synthetic_multi_day_rows(n_days=n_days, start=start, return_gen=return_gen)
    variables, validity = _variables_and_validity(rows)
    _write_tda04_artifacts(config, variables, validity)
    calendar_df = attach_calendar_fields(variables, validity)
    _write_tda06_artifacts(config, calendar_df, list(cutoffs), s_m_by_minute=s_m_by_minute)
    return variables, validity


# ---------------------------------------------------------------------------
# A. build_log_hl_population / build_log_hl_tilde / verify_s_m_is_log_hl_proxy
# ---------------------------------------------------------------------------

def test_build_log_hl_population_includes_rows_without_valid_r1m():
    rows = _synthetic_multi_day_rows(n_days=3, start=datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    cutoffs = [600, 900]
    log_hl_pop, labels = build_log_hl_population(variables, validity, cutoffs)
    # la primera fila de la serie (r_1m_valid=False) SI debe estar presente:
    # log_hl no depende de una barra anterior.
    assert (~validity["r_1m_valid"]).sum() > 0
    assert len(log_hl_pop) == len(variables)
    assert log_hl_pop["log_hl"].notna().all()
    assert set(labels) == set(log_hl_pop["segment_label"].unique()) or len(labels) >= len(log_hl_pop["segment_label"].unique())


def test_verify_s_m_is_log_hl_proxy_raises_on_wrong_proxy():
    bad_table = pd.DataFrame({"minute_of_day": [0, 1], "s_m": [1.0, 1.0], "proxy": ["abs_r_1m", "abs_r_1m"]})
    with pytest.raises(SMProxyMismatchError):
        verify_s_m_is_log_hl_proxy(bad_table)


def test_verify_s_m_is_log_hl_proxy_passes_for_log_hl():
    ok_table = pd.DataFrame({"minute_of_day": [0, 1], "s_m": [1.0, 1.0], "proxy": ["log_hl", "log_hl"]})
    verify_s_m_is_log_hl_proxy(ok_table)  # no debe lanzar


def test_build_log_hl_tilde_divides_by_s_m_and_protects_against_zero():
    log_hl_pop = pd.DataFrame({
        "minute_of_day": [0, 1, 2], "log_hl": [0.002, 0.004, 0.001],
    })
    s_m_table = pd.DataFrame({"minute_of_day": [0, 1, 2], "s_m": [2.0, 0.0, np.nan], "proxy": "log_hl"})
    out = build_log_hl_tilde(log_hl_pop, s_m_table)
    assert out["log_hl_tilde"].iloc[0] == pytest.approx(0.001)  # 0.002 / 2.0
    assert np.isnan(out["log_hl_tilde"].iloc[1])  # s_m=0 -> NaN, nunca inf
    assert np.isnan(out["log_hl_tilde"].iloc[2])  # s_m=NaN -> NaN
    assert out["label"].eq("RETROSPECTIVO").all()
    assert not np.isinf(out["log_hl_tilde"]).any()


# ---------------------------------------------------------------------------
# B. Topologia: block_relative_position / engle_lm_statistic nunca fabrican
# vecinos ni cruzan fronteras
# ---------------------------------------------------------------------------

def test_block_relative_position_resets_at_each_new_block():
    block_ids = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    pos = block_relative_position(block_ids)
    np.testing.assert_array_equal(pos, [0, 1, 2, 0, 1, 0, 1, 2, 3])


def test_engle_lm_statistic_only_uses_full_chains_within_one_block():
    # Dos bloques cortos (longitud 3) -- ningun rezago de orden 2 tiene una
    # cadena completa (posicion maxima dentro de cada bloque es 2, se
    # necesitaria posicion>=2, exactamente 1 fila por bloque -- 2 filas en
    # total, insuficiente para el minimo de "order+5").
    x = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    block_ids = np.array([0, 0, 0, 1, 1, 1])
    result = engle_lm_statistic(x, block_ids, order=2)
    assert result["estimable"] is False


def test_engle_lm_statistic_estimable_with_enough_full_chains():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=500)
    block_ids = np.zeros(500, dtype=int)  # un solo bloque continuo
    result = engle_lm_statistic(x, block_ids, order=1)
    assert result["estimable"] is True
    assert result["n_eff"] == 499
    assert np.isfinite(result["LM"])
    assert 0.0 <= result["R2"] <= 1.0 + 1e-9


def test_engle_lm_statistic_detects_genuine_arch_clustering():
    rng = np.random.default_rng(1)
    n = 4000
    sigma2 = np.empty(n)
    x = np.empty(n)
    sigma2[0] = 1.0
    x[0] = rng.normal(0.0, np.sqrt(sigma2[0]))
    for t in range(1, n):
        sigma2[t] = 0.05 + 0.85 * x[t - 1] ** 2
        x[t] = rng.normal(0.0, np.sqrt(sigma2[t]))
    block_ids = np.zeros(n, dtype=int)

    iid = rng.normal(0.0, 1.0, size=n)
    r_arch = engle_lm_statistic(x, block_ids, order=1)
    r_iid = engle_lm_statistic(iid, block_ids, order=1)
    assert r_arch["R2"] > r_iid["R2"]
    assert r_arch["LM"] > r_iid["LM"]


def test_calibrate_engle_lm_is_reproducible_with_fixed_seed():
    rng = np.random.default_rng(2)
    n = 600
    x = rng.normal(0.0, 1.0, size=n)
    block_ids = np.zeros(n, dtype=int)
    minute_of_day = np.tile(np.arange(60), n // 60 + 1)[:n]
    r1 = calibrate_engle_lm(x, block_ids, minute_of_day, order=1, n_perm=10, seed=7)
    r2 = calibrate_engle_lm(x, block_ids, minute_of_day, order=1, n_perm=10, seed=7)
    assert r1["null_p50"] == r2["null_p50"]
    assert r1["percentile_of_real"] == r2["percentile_of_real"]


# ---------------------------------------------------------------------------
# C. |r| y r^2 no dependen del signo
# ---------------------------------------------------------------------------

def test_abs_and_square_magnitude_proxies_are_sign_invariant():
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 1.0, size=300)
    block_ids = np.zeros(300, dtype=int)
    acf_abs_pos = compute_acf(np.abs(x), block_ids, max_lag=5)
    acf_abs_neg = compute_acf(np.abs(-x), block_ids, max_lag=5)
    acf_sq_pos = compute_acf(x ** 2, block_ids, max_lag=5)
    acf_sq_neg = compute_acf((-x) ** 2, block_ids, max_lag=5)
    pd.testing.assert_frame_equal(acf_abs_pos, acf_abs_neg)
    pd.testing.assert_frame_equal(acf_sq_pos, acf_sq_neg)


# ---------------------------------------------------------------------------
# D. clock_profile_flatness / g2_global_permutation_null
# ---------------------------------------------------------------------------

def test_clock_profile_flatness_is_near_zero_for_a_flat_profile():
    minute_of_day = np.tile(np.arange(10), 50)
    values = np.full(500, 5.0)  # mismo nivel en todos los minutos
    flat = clock_profile_flatness(values, minute_of_day)
    assert flat == pytest.approx(0.0, abs=1e-9)


def test_clock_profile_flatness_is_high_for_a_strong_clock_pattern():
    minute_of_day = np.tile(np.arange(10), 50)
    values = np.where(minute_of_day % 2 == 0, 10.0, 1.0)
    flat = clock_profile_flatness(values, minute_of_day)
    assert flat > CLOCK_FLATNESS_RATIO_THRESHOLD


def test_g2_global_permutation_null_is_reproducible_and_destroys_dependence():
    rng = np.random.default_rng(4)
    n = 2000
    x = np.empty(n)
    x[0] = rng.normal()
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + rng.normal(0.0, 0.3)  # AR(1) fuerte
    block_ids = np.zeros(n, dtype=int)

    null1 = g2_global_permutation_null(x, block_ids, lags=(1, 2), n_perm=20, seed=11)
    null2 = g2_global_permutation_null(x, block_ids, lags=(1, 2), n_perm=20, seed=11)
    np.testing.assert_array_equal(null1, null2)  # reproducible

    real_acf = compute_acf(x, block_ids, max_lag=2)
    real_rho1 = float(real_acf.loc[0, "rho"])
    null_rho1_abs_mean = float(np.nanmean(np.abs(null1[:, 0])))
    assert abs(real_rho1) > null_rho1_abs_mean  # la dependencia real supera claramente al null permutado


# ---------------------------------------------------------------------------
# E. same_clock_next_trading_day -- diagnostico SEPARADO de una ACF continua
# ---------------------------------------------------------------------------

def test_same_clock_next_trading_day_matches_hand_computed_correlation():
    # 4 dias, 3 minutos por dia, valores conocidos -> correlacion calculada a mano.
    dates = [datetime.date(2024, 1, i) for i in (8, 9, 10, 11)]
    minute = [0, 1, 2]
    values_by_day = {
        dates[0]: [1.0, 2.0, 3.0],
        dates[1]: [2.0, 3.0, 4.0],
        dates[2]: [4.0, 5.0, 6.0],
        dates[3]: [3.0, 2.0, 1.0],
    }
    rows = []
    for d in dates:
        for m, v in zip(minute, values_by_day[d]):
            rows.append({"trading_date": d, "minute_of_day": m, "val": v})
    pop = pd.DataFrame(rows)

    result = same_clock_next_trading_day(pop, "val", n_boot=10, seed=0)
    assert result["n_day_pairs"] == 3  # (d0,d1),(d1,d2),(d2,d3)
    assert result["n_pairs"] == 9  # 3 minutos x 3 pares de dias

    a = np.array([values_by_day[dates[0]], values_by_day[dates[1]], values_by_day[dates[2]]]).ravel()
    b = np.array([values_by_day[dates[1]], values_by_day[dates[2]], values_by_day[dates[3]]]).ravel()
    expected_rho = np.corrcoef(a, b)[0, 1]
    assert result["rho"] == pytest.approx(expected_rho)


def test_same_clock_next_trading_day_never_pairs_non_consecutive_days():
    # Si un dia intermedio no tiene NINGUNA barra en un minuto dado, ese
    # minuto simplemente no aporta un par ahi (NaN en el pivote) -- nunca
    # se fabrica un vecino saltando el dia faltante.
    rows = [
        {"trading_date": datetime.date(2024, 1, 8), "minute_of_day": 0, "val": 1.0},
        {"trading_date": datetime.date(2024, 1, 9), "minute_of_day": 0, "val": np.nan},
        {"trading_date": datetime.date(2024, 1, 10), "minute_of_day": 0, "val": 2.0},
    ]
    pop = pd.DataFrame(rows)
    result = same_clock_next_trading_day(pop, "val", n_boot=5, seed=0)
    # (dia0,dia1): dia1 es NaN -> no aporta par. (dia1,dia2): dia1 es NaN -> no aporta par.
    assert result["n_pairs"] == 0


# ---------------------------------------------------------------------------
# F. dependence_energy / clock_attribution / decide_stop9 (TH21 / STOP-9)
# ---------------------------------------------------------------------------

def _acf_tab(rhos: list[float], n_pairs: int = 1000) -> pd.DataFrame:
    return pd.DataFrame({
        "lag": range(1, len(rhos) + 1), "rho": rhos, "beta": rhos,
        "n_pairs": [n_pairs] * len(rhos), "estimable": [True] * len(rhos),
    })


def test_dependence_energy_matches_hand_computed_portmanteau():
    tab = _acf_tab([0.1, 0.05, 0.0], n_pairs=100)
    energy = dependence_energy(tab, m=3)
    expected = 100 * (0.1 ** 2 + 0.05 ** 2 + 0.0 ** 2)
    assert energy == pytest.approx(expected)


def test_clock_attribution_collapse_gives_high_fraction_removed():
    raw = _acf_tab([0.3, 0.25, 0.2, 0.15], n_pairs=1000)
    adjusted = _acf_tab([0.01, 0.005, 0.0, -0.002], n_pairs=1000)
    attr = clock_attribution(raw, adjusted, m=4)
    assert attr["fraction_removed"] > STOP9_FRACTION_REMOVED_THRESHOLD


def test_clock_attribution_survival_gives_low_fraction_removed():
    raw = _acf_tab([0.3, 0.25, 0.2, 0.15], n_pairs=1000)
    adjusted = _acf_tab([0.28, 0.22, 0.18, 0.14], n_pairs=1000)  # casi identica
    attr = clock_attribution(raw, adjusted, m=4)
    assert attr["fraction_survives"] > 0.5
    assert attr["fraction_removed"] < STOP9_FRACTION_REMOVED_THRESHOLD


def test_clock_attribution_reports_nan_when_denominator_is_unstable():
    raw = _acf_tab([0.0, 0.0, 0.0], n_pairs=1000)
    adjusted = _acf_tab([0.01, 0.0, 0.0], n_pairs=1000)
    attr = clock_attribution(raw, adjusted, m=3)
    assert np.isnan(attr["fraction_removed"])
    assert np.isnan(attr["fraction_survives"])


def test_decide_stop9_activated_only_when_all_variables_collapse():
    collapse_abs_r = {"variable": "abs_r", "fraction_removed": 0.97}
    collapse_log_hl = {"variable": "log_hl", "fraction_removed": 0.95}
    decision = decide_stop9([collapse_abs_r, collapse_log_hl])
    assert decision["stop9_activated"] is True

    survives_log_hl = {"variable": "log_hl", "fraction_removed": 0.30}
    decision2 = decide_stop9([collapse_abs_r, survives_log_hl])
    assert decision2["stop9_activated"] is False


def test_decide_stop9_conservative_default_when_no_valid_fraction():
    decision = decide_stop9([{"variable": "abs_r", "fraction_removed": float("nan")}])
    assert decision["stop9_activated"] is False


# ---------------------------------------------------------------------------
# G. decay_form_diagnostic -- log-log vs semi-log (TH20), sin estimar 'd'
# ---------------------------------------------------------------------------

def test_decay_form_diagnostic_prefers_loglog_for_a_power_law_decay():
    k = np.arange(1, 201)
    rho = 0.3 * k ** (-0.6)  # decaimiento polinomial
    tab = pd.DataFrame({"lag": k, "rho": rho, "n_pairs": 1000, "estimable": True})
    diag = decay_form_diagnostic(tab)
    assert diag["estimable"] is True
    assert diag["r2_loglog"] > diag["r2_semilog"]
    assert "log-log" in diag["form_hint"]


def test_decay_form_diagnostic_prefers_semilog_for_an_exponential_decay():
    k = np.arange(1, 201)
    rho = 0.3 * np.exp(-k / 15.0)  # decaimiento exponencial
    tab = pd.DataFrame({"lag": k, "rho": rho, "n_pairs": 1000, "estimable": True})
    diag = decay_form_diagnostic(tab)
    assert diag["estimable"] is True
    assert diag["r2_semilog"] > diag["r2_loglog"]
    assert "semi-log" in diag["form_hint"]


def test_decay_form_diagnostic_not_estimable_with_too_few_positive_lags():
    tab = pd.DataFrame({"lag": [1, 2, 3], "rho": [-0.1, -0.2, 0.05], "n_pairs": 100, "estimable": True})
    diag = decay_form_diagnostic(tab)
    assert diag["estimable"] is False


# ---------------------------------------------------------------------------
# H. mean_removal_sensitivity -- no fabrica vecinos al eliminar filas
# ---------------------------------------------------------------------------

def test_mean_removal_sensitivity_does_not_fabricate_neighbors_across_dropped_rows():
    rows = _synthetic_multi_day_rows(n_days=5, start=datetime.date(2024, 1, 8), bars_per_day=30)
    variables, validity = _variables_and_validity(rows)
    calendar_df = attach_calendar_fields(variables, validity)
    r1m_pop = calendar_df.loc[calendar_df["r_1m_valid"]].copy()
    block_ids = compute_block_ids(r1m_pop["timestamp"], r1m_pop["trading_date"], 60.0)
    out = mean_removal_sensitivity(r1m_pop, block_ids)
    assert set(out["lag"]) == {1, 5, 20, 60}
    assert np.isfinite(out["beta_1_used"]).all()
    # con beta_1 pequeño (retornos ~iid), remover la media no debe disparar
    # rho_abs_e a un valor absurdo (>1 en valor absoluto seria imposible
    # por Cauchy-Schwarz, ya garantizado por compute_acf, pero se verifica
    # que la funcion produce numeros finitos o NaN explicito, nunca inf).
    finite_or_nan = out["rho_abs_e"].apply(lambda v: np.isnan(v) or np.isfinite(v))
    assert finite_or_nan.all()


# ---------------------------------------------------------------------------
# I. Orquestador end-to-end -- holdout, alineacion, proxy de s(m)
# ---------------------------------------------------------------------------

def test_run_tda09_raises_when_research_and_holdout_overlap(tmp_path):
    config = _make_config(
        tmp_path, research_files=["00_mnq_03_24.Last.txt"],
        holdout_files=["00_mnq_03_24.Last.txt", "no_existe.Last.txt"],
    )
    with pytest.raises(HoldoutIsolationError, match="disjuntos"):
        run_tda09_analysis(config)


def test_run_tda09_never_opens_any_raw_or_holdout_file(tmp_path):
    config = _make_config(
        tmp_path, research_files=["00_mnq_03_24.Last.txt"],
        holdout_files=["archivo_que_no_existe.Last.txt"],
    )
    _prepare_full_fixture(config)
    assert not (config.raw_dir / "00_mnq_03_24.Last.txt").exists()
    result = run_tda09_analysis(config)
    assert len(result.r1m_population) > 0


def test_run_tda09_raises_on_misaligned_timestamps(tmp_path):
    config = _make_config(tmp_path, research_files=["00_mnq_03_24.Last.txt"])
    rows = _synthetic_multi_day_rows(n_days=5, start=datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    validity = validity.drop(index=2).reset_index(drop=True)
    _write_tda04_artifacts(config, variables, validity)
    with pytest.raises(TimestampAlignmentError):
        run_tda09_analysis(config)


def test_run_tda09_raises_on_s_m_proxy_mismatch(tmp_path):
    config = _make_config(tmp_path, research_files=["00_mnq_03_24.Last.txt"])
    rows = _synthetic_multi_day_rows(n_days=8, start=datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    _write_tda04_artifacts(config, variables, validity)
    calendar_df = attach_calendar_fields(variables, validity)
    _write_tda06_artifacts(config, calendar_df, [600, 900])
    # sobrescribir el parquet de s(m) con un proxy incorrecto
    bad = pd.read_parquet(config.tda06_s_m_parquet_path)
    bad["proxy"] = "abs_r_1m"
    bad.to_parquet(config.tda06_s_m_parquet_path, index=False)
    with pytest.raises(SMProxyMismatchError):
        run_tda09_analysis(config)


def test_run_tda09_end_to_end_produces_all_result_fields(tmp_path):
    config = _make_config(tmp_path, research_files=["00_mnq_03_24.Last.txt"])
    _prepare_full_fixture(config, n_days=20)
    result = run_tda09_analysis(config)

    assert not result.acf_table.empty
    assert set(result.acf_table["variable"]) == {"r", "abs_r", "r2", "log_hl"}
    assert set(result.acf_table["raw_adjusted"]) == {"raw", "adjusted"}
    assert not result.bootstrap_table.empty
    assert "same_clock_next_trading_day" in set(result.bootstrap_table["kind"])
    assert "continuous_lag" in set(result.bootstrap_table["kind"])
    assert not result.portmanteau_table.empty
    assert not result.clock_attribution_table.empty
    assert set(result.clock_attribution_table["variable"]) == {"abs_r", "log_hl"}
    assert "stop9_activated" in result.stop9_decision
    assert not result.persistence_by_year.empty
    assert not result.persistence_by_segment.empty
    assert not result.arch_lm_table.empty
    assert set(result.arch_lm_table["order"]) == set(ARCH_LM_ORDERS)
    assert not result.g2_null1_calibration.empty
    assert not result.g2_secondary_global_calibration.empty
    assert "kurt_within_tolerance" in result.g2_synthetic_moment_check
    assert not result.mean_removal_sensitivity_table.empty
    assert not result.clock_flatness_table.empty

    # r/r2 NUNCA reciben calibracion G2 propia (solo abs_r/log_hl la tienen).
    port = result.portmanteau_table
    for name in ("r", "r2"):
        sub = port.loc[port["variable"] == name]
        assert "G2_CALIBRATED" not in set(sub["calibration_status"])
    for name in ("abs_r", "log_hl"):
        sub = port.loc[(port["variable"] == name) & (port["raw_adjusted"] == "raw")]
        estimable = sub.loc[sub["estimable"]]
        if not estimable.empty:
            assert set(estimable["calibration_status"]) <= {"G2_CALIBRATED", "DESCRIPTIVE_UNCALIBRATED"}


def test_run_tda09_synthetic_null_diagnostic_is_kept_separate_from_principal_inference(tmp_path):
    config = _make_config(tmp_path, research_files=["00_mnq_03_24.Last.txt"])
    _prepare_full_fixture(config, n_days=20)
    result = run_tda09_analysis(config)

    # el diagnostico de momentos del null sintetico es un dict SEPARADO,
    # nunca fusionado silenciosamente en las tablas de calibracion
    # principal/secundaria (seccion 15/25 de la tarea: un null que no
    # supera su propio diagnostico no puede colarse en la inferencia).
    assert isinstance(result.g2_synthetic_moment_check, dict)
    for col in result.g2_null1_calibration.columns:
        assert "synthetic" not in col.lower()
    for col in result.g2_secondary_global_calibration.columns:
        assert "synthetic" not in col.lower()


# ---------------------------------------------------------------------------
# J. Persistencia exacta de los artefactos declarados
# ---------------------------------------------------------------------------

def test_persist_artifacts_writes_exactly_the_declared_paths_and_nothing_obsolete(tmp_path):
    from ohlcv_dataroad.ingest.run_tda09 import ARTIFACT_PATH_ATTRS, persist_artifacts

    config = _make_config(tmp_path, research_files=["00_mnq_03_24.Last.txt"])
    _prepare_full_fixture(config, n_days=20)
    result = run_tda09_analysis(config)

    written = persist_artifacts(result, config)
    declared_paths = {getattr(config, attr) for attr in ARTIFACT_PATH_ATTRS}

    # el decay_png solo se escribe si TH20 quedo habilitada -- se acepta
    # que "written" contenga ese extra, pero todo lo DECLARADO como base
    # debe estar presente.
    assert declared_paths <= set(written)
    for p in declared_paths:
        assert p.exists(), f"artefacto declarado pero no escrito: {p}"

    tda09_files_on_disk = {p.name for p in config.reports_dir.glob("TDA09_*")}
    declared_names = {p.name for p in written}
    obsolete = tda09_files_on_disk - declared_names
    assert not obsolete, f"artefactos TDA09_* en disco que ya no estan declarados: {obsolete}"


# ---------------------------------------------------------------------------
# K. Escenarios de comportamiento: IID, clustering genuino, patron de reloj
#    puro, y clustering + reloj combinados (TH19/TH21 -- validacion adversarial)
# ---------------------------------------------------------------------------

def test_iid_series_produces_near_zero_magnitude_acf(tmp_path):
    config = _make_config(tmp_path, research_files=["00_mnq_03_24.Last.txt"])
    _prepare_full_fixture(config, n_days=25)
    result = run_tda09_analysis(config)

    sub = result.acf_table[(result.acf_table["variable"] == "abs_r") & (result.acf_table["raw_adjusted"] == "raw")]
    sub = sub[sub["estimable"]]
    # bajo iid, la MAYORIA de los rho deberian ser pequeños -- se verifica
    # con una cota laxa (no un test de significancia): la mediana de
    # |rho_k| en los primeros 30 rezagos debe ser chica.
    early = sub[sub["lag"] <= 30]
    assert early["rho"].abs().median() < 0.15


def test_genuine_clustering_produces_clearly_positive_magnitude_acf(tmp_path):
    def _garch_like(rng, day_i, bar_i, state={"sigma2": 0.0005 ** 2}):
        state["sigma2"] = 0.05 * (0.0005 ** 2) + 0.90 * state["sigma2"]
        shock = rng.normal(0.0, np.sqrt(state["sigma2"]))
        state["sigma2"] = 0.05 * (0.0005 ** 2) + 0.90 * shock ** 2
        return shock

    config = _make_config(tmp_path, research_files=["00_mnq_03_24.Last.txt"])
    _prepare_full_fixture(config, n_days=25, return_gen=_garch_like)
    result = run_tda09_analysis(config)

    sub = result.acf_table[(result.acf_table["variable"] == "abs_r") & (result.acf_table["raw_adjusted"] == "raw")]
    rho1 = float(sub.loc[sub["lag"] == 1, "rho"].iloc[0])
    assert rho1 > 0.05  # clustering genuino de rezago corto, claramente positivo


def test_deterministic_clock_pattern_without_dynamics_collapses_after_adjustment(tmp_path):
    # Cada minuto tiene una escala FIJA (alterna alta/baja), pero cada
    # barra es un sorteo INDEPENDIENTE dado su minuto -- no hay memoria
    # genuina, solo el reloj. s(m) construido ANALITICAMENTE a partir de
    # la MISMA regla de escala (no estimado empiricamente, para aislar
    # la pregunta: ¿colapsa el ajuste si s(m) es exacto?).
    def _clock_only(rng, day_i, bar_i):
        scale = 0.0015 if bar_i % 2 == 0 else 0.0001
        return rng.normal(0.0, scale)

    config = _make_config(tmp_path, research_files=["00_mnq_03_24.Last.txt"])
    bars_per_day = 300  # >> TH21_ENERGY_M (240) para que Q(m) sea estimable dentro de un solo bloque diario
    minute_scale = {}

    rows = _synthetic_multi_day_rows(n_days=25, start=datetime.date(2024, 1, 8), bars_per_day=bars_per_day, return_gen=_clock_only)
    variables, validity = _variables_and_validity(rows)
    _write_tda04_artifacts(config, variables, validity)
    calendar_df = attach_calendar_fields(variables, validity)

    for m in calendar_df["minute_of_day"].unique():
        is_even_bar = (m - calendar_df["minute_of_day"].min()) % 2 == 0
        minute_scale[int(m)] = 15.0 if is_even_bar else 1.0  # proporcional a 0.0015 vs 0.0001, normalizado

    _write_tda06_artifacts(config, calendar_df, [600, 900], s_m_by_minute=minute_scale)
    result = run_tda09_analysis(config)

    attr = result.clock_attribution_table
    fraction_removed_abs_r = float(attr.loc[attr["variable"] == "abs_r", "fraction_removed"].iloc[0])
    assert fraction_removed_abs_r > 0.5  # una fraccion sustancial de la dependencia (puramente de reloj) desaparece


def test_genuine_clustering_plus_clock_pattern_survives_adjustment(tmp_path):
    state = {"sigma2": 0.0005 ** 2}

    def _garch_plus_clock(rng, day_i, bar_i):
        scale_mult = 3.0 if bar_i % 2 == 0 else 1.0
        state["sigma2"] = 0.05 * (0.0005 ** 2) + 0.90 * state["sigma2"]
        shock = rng.normal(0.0, np.sqrt(state["sigma2"]) * scale_mult)
        state["sigma2"] = 0.05 * (0.0005 ** 2) + 0.90 * (shock / scale_mult) ** 2
        return shock

    config = _make_config(tmp_path, research_files=["00_mnq_03_24.Last.txt"])
    bars_per_day = 300  # >> TH21_ENERGY_M (240) para que Q(m) sea estimable dentro de un solo bloque diario
    rows = _synthetic_multi_day_rows(n_days=25, start=datetime.date(2024, 1, 8), bars_per_day=bars_per_day, return_gen=_garch_plus_clock)
    variables, validity = _variables_and_validity(rows)
    _write_tda04_artifacts(config, variables, validity)
    calendar_df = attach_calendar_fields(variables, validity)

    minute_scale = {}
    for m in calendar_df["minute_of_day"].unique():
        is_even_bar = (m - calendar_df["minute_of_day"].min()) % 2 == 0
        minute_scale[int(m)] = 3.0 if is_even_bar else 1.0

    _write_tda06_artifacts(config, calendar_df, [600, 900], s_m_by_minute=minute_scale)
    result = run_tda09_analysis(config)

    sub_adj = result.acf_table[(result.acf_table["variable"] == "abs_r") & (result.acf_table["raw_adjusted"] == "adjusted")]
    rho1_adj = float(sub_adj.loc[sub_adj["lag"] == 1, "rho"].iloc[0])
    assert rho1_adj > 0.03  # la dependencia genuina sobrevive al ajuste por reloj
