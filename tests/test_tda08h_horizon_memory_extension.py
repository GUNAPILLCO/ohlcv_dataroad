"""Tests de TDA08-H -- extension acotada de TDA-08 a memoria de 30/60 minutos.

No reimplementa nada de TDA-08/TH10 -- solo verifica que
``compute_multi_horizon_memory``/``run_tda08h_analysis`` conectan
correctamente las funciones YA validadas (``build_horizon_returns``,
``non_overlap_mask``, ``compute_block_ids``, ``compute_acf``,
``bootstrap_rho``) sin romper sus invariantes, y que 1/5/10 siguen
coincidiendo con la implementacion vigente de TDA-08 (control de
regresion estructural -- no se hardcodean numeros del dataset real).
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
from ohlcv_dataroad.ingest.tda08_linear_mean_dependence import compute_multi_frequency_rho1
from ohlcv_dataroad.ingest.tda08h_horizon_memory_extension import (
    HORIZONS,
    compute_multi_horizon_memory,
    run_tda08h_analysis,
)

TICK = 0.25


# ---------------------------------------------------------------------------
# Fixtures sinteticas (mismo estilo que test_tda08_linear_mean_dependence.py,
# duplicadas aqui deliberadamente -- son helpers de TEST, no logica de
# produccion; ver seccion 15 de la tarea, que solo prohibe duplicar
# retornos/non-overlap/bootstrap/ACF/beta, todos reutilizados via import)
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


def _single_long_chain_with_returns(returns: np.ndarray, start_ts: pd.Timestamp, trading_date: datetime.date, start_price: float = 15000.0, volume: int = 20) -> list[dict]:
    n = len(returns) + 1
    ts = pd.date_range(start_ts, periods=n, freq="1min")
    rows = [{"timestamp": ts[0], "trading_date": trading_date, "close": start_price, "r_1m_valid": False, "volume": volume}]
    price = start_price
    for i, r in enumerate(returns):
        price = price * np.exp(r)
        rows.append({"timestamp": ts[i + 1], "trading_date": trading_date, "close": price, "r_1m_valid": True, "volume": volume})
    return rows


def _two_block_long_chains(n_per_block: int, phi: float, seed: int) -> list[dict]:
    """Dos bloques de continuidad LARGOS y SEPARADOS -- para verificar que ningun par lag-1 los cruza.

    ``n_per_block`` minutos pueden abarcar varios dias de reloj (p. ej.
    13000 minutos ~ 9 dias) -- el segundo bloque debe empezar
    ESTRICTAMENTE despues de que el primero termine en tiempo de reloj,
    o ``compute_block_ids`` (que ordena por timestamp) intercalaria las
    filas de ambos bloques al ordenar globalmente.
    """
    rng = np.random.default_rng(seed)

    def _ar1(n: int) -> np.ndarray:
        e = rng.normal(0.0, 0.001, size=n)
        r = np.empty(n)
        r[0] = e[0]
        for i in range(1, n):
            r[i] = phi * r[i - 1] + e[i]
        return r

    start_a = pd.Timestamp("2024-01-08 08:00:00")
    rows_a = _single_long_chain_with_returns(_ar1(n_per_block), start_a, datetime.date(2024, 1, 8))
    end_a_ts = rows_a[-1]["timestamp"]

    start_b = (end_a_ts + pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=8)
    rows_b = _single_long_chain_with_returns(_ar1(n_per_block), start_b, start_b.date())

    return rows_a + rows_b


# ---------------------------------------------------------------------------
# 1-7. Construccion de h=30/60, no-solapamiento, no-cruce, monotonicidad
# ---------------------------------------------------------------------------

def test_compute_multi_horizon_memory_builds_h30_and_h60_with_non_null_estimates():
    rows = _two_block_long_chains(n_per_block=13000, phi=0.15, seed=1)
    variables, validity = _variables_and_validity(rows)
    result = compute_multi_horizon_memory(variables, validity, n_boot=20, seed=0, horizons=(1, 5, 10, 30, 60))

    assert set(result["h_minutes"]) == {1, 5, 10, 30, 60}
    for h in (30, 60):
        row = result[result["h_minutes"] == h].iloc[0]
        assert row["n"] > 0
        assert not np.isnan(row["rho_1"])
        assert not np.isnan(row["beta_1"])
        assert not np.isnan(row["rho_ci_lo"]) and not np.isnan(row["rho_ci_hi"])
        assert not np.isnan(row["beta_ci_lo"]) and not np.isnan(row["beta_ci_hi"])


def test_n_pairs_never_exceeds_n_at_any_horizon():
    rows = _two_block_long_chains(n_per_block=13000, phi=0.0, seed=2)
    variables, validity = _variables_and_validity(rows)
    result = compute_multi_horizon_memory(variables, validity, n_boot=10, seed=0, horizons=(1, 5, 10, 30, 60))
    assert (result["n_pairs_lag1"] <= result["n"]).all()


def test_n_decreases_monotonically_with_horizon():
    """n_60 < n_30 < n_10 < n_5 < n_1 -- perdida de muestra esperada al exigir ventanas no solapadas mas largas."""
    rows = _two_block_long_chains(n_per_block=13000, phi=0.0, seed=3)
    variables, validity = _variables_and_validity(rows)
    result = compute_multi_horizon_memory(variables, validity, n_boot=10, seed=0, horizons=(1, 5, 10, 30, 60))
    n_by_h = result.set_index("h_minutes")["n"]
    assert n_by_h[60] < n_by_h[30] < n_by_h[10] < n_by_h[5] < n_by_h[1]


def test_windows_are_non_overlapping_by_construction_h30_and_h60():
    """Dos ventanas NO SOLAPADAS consecutivas de horizonte h deben estar separadas EXACTAMENTE h minutos en el tiempo (nunca 1 minuto, que seria solapamiento casi total)."""
    rows = _single_long_chain_with_returns(
        np.random.default_rng(4).normal(0.0, 0.001, size=5000), pd.Timestamp("2024-01-08 08:00:00"), datetime.date(2024, 1, 8)
    )
    variables, validity = _variables_and_validity(rows)
    from ohlcv_dataroad.ingest.th10_horizon_scaling import build_horizon_returns, non_overlap_mask

    hdf = build_horizon_returns(variables, validity, horizons=(30, 60))
    for h in (30, 60):
        mask = non_overlap_mask(hdf["run_length"].to_numpy(), h)
        sub = hdf.loc[mask].dropna(subset=[f"r_{h}"]).sort_values("timestamp")
        deltas_minutes = sub["timestamp"].diff().dropna().dt.total_seconds() / 60.0
        # dentro del MISMO bloque de continuidad, el delta entre ventanas consecutivas es SIEMPRE h minutos
        assert (deltas_minutes.round(6) % h == 0).all()
        assert (deltas_minutes >= h).all()  # nunca menos de h minutos (eso seria solapamiento)


def test_lag1_pairs_never_cross_the_boundary_between_two_separate_blocks():
    """Prueba aritmetica exacta: con dos bloques largos y SEPARADOS (jornadas distintas), n_pairs_lag1(h) debe ser EXACTAMENTE 2*(floor(n_per_block/h)-1) -- ni un par mas, lo que demuestra que ningun par cruza la frontera entre bloques (trading_date/gap/roll -- misma logica que TDA-04/TH10)."""
    n_per_block = 13000
    rows = _two_block_long_chains(n_per_block=n_per_block, phi=0.0, seed=5)
    variables, validity = _variables_and_validity(rows)
    result = compute_multi_horizon_memory(variables, validity, n_boot=5, seed=0, horizons=(30, 60))

    for h in (30, 60):
        expected_n_pairs = 2 * (n_per_block // h - 1)
        actual = int(result.loc[result["h_minutes"] == h, "n_pairs_lag1"].iloc[0])
        assert actual == expected_n_pairs, f"h={h}: se esperaban {expected_n_pairs} pares (2 bloques separados), se obtuvieron {actual} -- posible fuga entre bloques"


# ---------------------------------------------------------------------------
# 8. rho_1/beta_1 usan el estimador FINAL de TDA-08 (Pearson pairwise-complete
#    con centrado local, OLS con intercepto) -- verificado contra la formula
#    a mano, no solo "coincide con lo que devuelve compute_acf" (circular).
# ---------------------------------------------------------------------------

def test_rho_1_and_beta_1_match_the_final_tda08_pairwise_complete_formula_by_hand():
    rows = _two_block_long_chains(n_per_block=13000, phi=0.2, seed=6)
    variables, validity = _variables_and_validity(rows)

    from ohlcv_dataroad.ingest.tda08_linear_mean_dependence import compute_block_ids
    from ohlcv_dataroad.ingest.th10_horizon_scaling import build_horizon_returns, non_overlap_mask

    h = 60
    hdf = build_horizon_returns(variables, validity, horizons=(h,))
    mask = non_overlap_mask(hdf["run_length"].to_numpy(), h)
    sub = hdf.loc[mask].dropna(subset=[f"r_{h}"]).sort_values("timestamp")
    block_ids = compute_block_ids(sub["timestamp"], sub["trading_date"], float(h * 60))
    values = sub[f"r_{h}"].to_numpy(dtype=float)

    same_block = block_ids[1:] == block_ids[:-1]
    x_t = values[1:][same_block]
    x_tk = values[:-1][same_block]
    e_t = x_t - x_t.mean()   # centrado LOCAL -- por su propia media sobre P_1 (2a revision de TDA-08)
    e_tk = x_tk - x_tk.mean()
    expected_rho = float(np.sum(e_t * e_tk) / np.sqrt(np.sum(e_t**2) * np.sum(e_tk**2)))
    expected_beta = float(np.sum(e_t * e_tk) / np.sum(e_tk**2))

    result = compute_multi_horizon_memory(variables, validity, n_boot=5, seed=0, horizons=(h,))
    row = result.iloc[0]
    assert row["rho_1"] == pytest.approx(expected_rho, rel=1e-9)
    assert row["beta_1"] == pytest.approx(expected_beta, rel=1e-9)


# ---------------------------------------------------------------------------
# 9. Bootstrap reproducible con semilla fija
# ---------------------------------------------------------------------------

def test_bootstrap_ci_is_reproducible_with_the_same_seed():
    rows = _two_block_long_chains(n_per_block=13000, phi=0.1, seed=7)
    variables, validity = _variables_and_validity(rows)
    r1 = compute_multi_horizon_memory(variables, validity, n_boot=30, seed=42, horizons=(30, 60))
    r2 = compute_multi_horizon_memory(variables, validity, n_boot=30, seed=42, horizons=(30, 60))
    pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# 10. h=1/5/10 reproducen EXACTAMENTE la implementacion vigente de TDA-08
#     (control de regresion estructural -- misma logica, no numeros
#     hardcodeados del dataset real)
# ---------------------------------------------------------------------------

def test_h_1_5_10_reproduce_tda08_compute_multi_frequency_rho1_exactly():
    rows = _two_block_long_chains(n_per_block=13000, phi=0.05, seed=8)
    variables, validity = _variables_and_validity(rows)

    tda08_result = compute_multi_frequency_rho1(variables, validity, n_boot=25, seed=0, horizons=(1, 5, 10))
    tda08h_result = compute_multi_horizon_memory(variables, validity, n_boot=25, seed=0, horizons=(1, 5, 10, 30, 60))
    tda08h_1_5_10 = tda08h_result[tda08h_result["h_minutes"].isin((1, 5, 10))].reset_index(drop=True)

    for h in (1, 5, 10):
        row_08 = tda08_result[tda08_result["h_minutes"] == h].iloc[0]
        row_08h = tda08h_1_5_10[tda08h_1_5_10["h_minutes"] == h].iloc[0]
        assert row_08h["n"] == row_08["n"]
        assert row_08h["n_pairs_lag1"] == row_08["n_pairs_lag1"]
        assert row_08h["rho_1"] == pytest.approx(row_08["rho_1"], rel=1e-12, nan_ok=True)
        assert row_08h["beta_1"] == pytest.approx(row_08["beta_1"], rel=1e-12, nan_ok=True)
        assert row_08h["rho_ci_lo"] == pytest.approx(row_08["rho_ci_lo"], rel=1e-12, nan_ok=True)
        assert row_08h["rho_ci_hi"] == pytest.approx(row_08["rho_ci_hi"], rel=1e-12, nan_ok=True)


# ---------------------------------------------------------------------------
# 11. Holdout -- nunca se abre
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
        "tda08": {
            "output_dir_reports": "reports/mnq", "n_boot": 5, "n_perm": 5,
            "report_name": "TDA08_dependencia_lineal_media.md",
        },
        "tda08h": {
            "output_dir_reports": "reports/mnq",
            "report_name": "TDA08H_horizon_memory_extension.md",
            "multi_horizon_csv": "TDA08H_rho1_multi_horizon.csv",
            "plot_png": "TDA08H_rho1_multi_horizon.png",
        },
    }
    config_path = config_dir / "mnq_snapshot.yaml"
    config_path.write_text(yaml.safe_dump(config_yaml), encoding="utf-8")
    return load_config(config_path)


def _write_tda04_artifacts(config, variables: pd.DataFrame, validity: pd.DataFrame) -> None:
    config.interim_dir.mkdir(parents=True, exist_ok=True)
    variables.to_parquet(config.tda04_variables_parquet_path, index=False)
    validity.to_parquet(config.tda04_validity_mask_parquet_path, index=False)


def test_run_tda08h_raises_when_research_and_holdout_overlap(tmp_path):
    config = _make_config(
        tmp_path, research_files=["00_mnq_03_24.Last.txt"],
        holdout_files=["00_mnq_03_24.Last.txt", "no_existe.Last.txt"],
    )
    with pytest.raises(HoldoutIsolationError, match="disjuntos"):
        run_tda08h_analysis(config)


def test_run_tda08h_never_opens_any_raw_or_holdout_file(tmp_path):
    config = _make_config(
        tmp_path, research_files=["00_mnq_03_24.Last.txt"],
        holdout_files=["archivo_que_no_existe.Last.txt"],
    )
    rows = _synthetic_multi_day_rows(n_days=10, start=datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    _write_tda04_artifacts(config, variables, validity)

    assert not (config.raw_dir / "00_mnq_03_24.Last.txt").exists()
    result = run_tda08h_analysis(config)
    assert not result.multi_horizon.empty
    assert set(result.multi_horizon["h_minutes"]) == set(HORIZONS)


def test_run_tda08h_raises_on_misaligned_timestamps(tmp_path):
    config = _make_config(tmp_path, research_files=["00_mnq_03_24.Last.txt"])
    rows = _synthetic_multi_day_rows(n_days=5, start=datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    validity = validity.drop(index=2).reset_index(drop=True)
    _write_tda04_artifacts(config, variables, validity)
    with pytest.raises(TimestampAlignmentError):
        run_tda08h_analysis(config)
