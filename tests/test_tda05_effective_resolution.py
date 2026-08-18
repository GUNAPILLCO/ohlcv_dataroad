"""Tests de TDA-05 -- resolucion efectiva y discrecion del retorno de 1 minuto.

Igual que TDA-02/03/04: datos sinteticos minimos para aislar cada
propiedad (conversion puntos->ticks respetando la mascara de validez de
TDA-04, invariante de grilla de tick, hora NY vs UTC, DST, unidades
dimensionalmente correctas, forward-fill), mas la proteccion del
hold-out. No se reproducen aqui los numeros exactos del conjunto de
investigacion real (esos viven en
``reports/mnq/TDA05_resolucion_discrecion.md``, generado por
``run_tda05.py``).
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
from ohlcv_dataroad.ingest.tda05_effective_resolution import (
    CENTRAL_RANGE_TICKS,
    TICK_GRID_TOLERANCE,
    TickGridInconsistencyError,
    attach_forward_fill_flags,
    build_by_hour_table,
    build_by_year_hour_table,
    build_by_year_table,
    build_stop5_watchlist,
    compare_with_without_forward_fill,
    compute_tick_variables,
    run_tda05_analysis,
    summarize_resolution,
)

TS_FORMAT = "%Y%m%d %H%M%S"
TICK = 0.25


# ---------------------------------------------------------------------------
# Fixtures sinteticas
# ---------------------------------------------------------------------------

def _variables_and_validity(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``rows``: lista de dicts con al menos timestamp/close/high/low/r_1m_valid.
    Construye las dos tablas alineadas que TDA-04 produciria."""
    variables = pd.DataFrame(
        [
            {
                "timestamp": r["timestamp"], "source_file": r.get("source_file", "f1"),
                "contract": r.get("contract", "H24"), "trading_date": r.get("trading_date", datetime.date(2024, 1, 8)),
                "segment_id": r.get("segment_id", 1),
                "open": r.get("open", r["close"]), "high": r.get("high", r["close"] + 1),
                "low": r.get("low", r["close"] - 1), "close": r["close"], "volume": r.get("volume", 10),
                "r_1m": r.get("r_1m", np.nan),
            }
            for r in rows
        ]
    )
    validity = pd.DataFrame(
        [{"timestamp": r["timestamp"], "r_1m_valid": r["r_1m_valid"]} for r in rows]
    )
    return variables, validity


def _empty_inactive_mask() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_file": pd.Series(dtype="str"),
            "expected_ts_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "trading_date": pd.Series(dtype="object"),
            "status": pd.Series(dtype="str"),
            "category": pd.Series(dtype="str"),
        }
    )


# ---------------------------------------------------------------------------
# compute_tick_variables -- conversion puntos -> ticks, respetando la mascara
# ---------------------------------------------------------------------------

def test_plus_one_tick_from_quarter_point_move():
    ts = pd.date_range("2024-01-08 15:00:00", periods=2, freq="1min")
    variables, validity = _variables_and_validity(
        [
            {"timestamp": ts[0], "close": 100.00, "r_1m_valid": False},
            {"timestamp": ts[1], "close": 100.25, "r_1m_valid": True},
        ]
    )
    out = compute_tick_variables(variables, validity, TICK)
    assert out.iloc[1]["delta_close_points"] == pytest.approx(0.25)
    assert out.iloc[1]["delta_close_ticks"] == pytest.approx(1.0)


def test_minus_two_ticks_from_half_point_move():
    ts = pd.date_range("2024-01-08 15:00:00", periods=2, freq="1min")
    variables, validity = _variables_and_validity(
        [
            {"timestamp": ts[0], "close": 100.50, "r_1m_valid": False},
            {"timestamp": ts[1], "close": 100.00, "r_1m_valid": True},
        ]
    )
    out = compute_tick_variables(variables, validity, TICK)
    assert out.iloc[1]["delta_close_points"] == pytest.approx(-0.50)
    assert out.iloc[1]["delta_close_ticks"] == pytest.approx(-2.0)


def test_zero_points_is_zero_ticks_and_matches_zero_1m():
    ts = pd.date_range("2024-01-08 15:00:00", periods=2, freq="1min")
    variables, validity = _variables_and_validity(
        [
            {"timestamp": ts[0], "close": 100.00, "r_1m_valid": False},
            {"timestamp": ts[1], "close": 100.00, "r_1m_valid": True, "r_1m": 0.0},
        ]
    )
    out = compute_tick_variables(variables, validity, TICK)
    assert out.iloc[1]["delta_close_ticks"] == pytest.approx(0.0)
    assert out.iloc[1]["r_1m"] == 0.0


def test_ticks_only_computed_when_r_1m_valid_true():
    """Aunque el Close SI cambio (0.25 puntos), si r_1m_valid=False (p.ej.
    una frontera de TDA-04), delta_close_ticks debe ser NaN -- nunca se
    calcula con solo `close.shift(1)` sin aplicar la mascara."""
    ts = pd.date_range("2024-01-08 15:00:00", periods=2, freq="1min")
    variables, validity = _variables_and_validity(
        [
            {"timestamp": ts[0], "close": 100.00, "r_1m_valid": False},
            {"timestamp": ts[1], "close": 100.25, "r_1m_valid": False},  # frontera: NaN pese al cambio de precio
        ]
    )
    out = compute_tick_variables(variables, validity, TICK)
    assert pd.isna(out.iloc[1]["delta_close_points"])
    assert pd.isna(out.iloc[1]["delta_close_ticks"])


def test_boundary_nan_from_tda04_stays_nan_in_derived_variables():
    ts = pd.date_range("2024-01-08 15:00:00", periods=3, freq="1min")
    variables, validity = _variables_and_validity(
        [
            {"timestamp": ts[0], "close": 100.00, "r_1m_valid": False},
            {"timestamp": ts[1], "close": 101.00, "r_1m_valid": False},  # NaN de frontera
            {"timestamp": ts[2], "close": 101.25, "r_1m_valid": True},
        ]
    )
    out = compute_tick_variables(variables, validity, TICK)
    assert pd.isna(out.iloc[1]["delta_close_points"])
    assert pd.isna(out.iloc[1]["delta_close_ticks"])
    assert not pd.isna(out.iloc[2]["delta_close_ticks"])  # la siguiente fila SI enlaza normalmente


def test_non_integer_tick_movement_raises_instead_of_rounding_silently():
    """Un movimiento que NO es multiplo entero de tick (fuera de tolerancia)
    debe detener el analisis, nunca redondearse en silencio."""
    ts = pd.date_range("2024-01-08 15:00:00", periods=2, freq="1min")
    variables, validity = _variables_and_validity(
        [
            {"timestamp": ts[0], "close": 100.00, "r_1m_valid": False},
            {"timestamp": ts[1], "close": 100.10, "r_1m_valid": True},  # 0.10 no es multiplo de 0.25
        ]
    )
    with pytest.raises(TickGridInconsistencyError):
        compute_tick_variables(variables, validity, TICK)


def test_floating_point_noise_within_tolerance_does_not_raise():
    ts = pd.date_range("2024-01-08 15:00:00", periods=2, freq="1min")
    variables, validity = _variables_and_validity(
        [
            {"timestamp": ts[0], "close": 100.00, "r_1m_valid": False},
            {"timestamp": ts[1], "close": 100.25 + TICK_GRID_TOLERANCE / 10, "r_1m_valid": True},
        ]
    )
    out = compute_tick_variables(variables, validity, TICK)  # no debe lanzar
    assert out.iloc[1]["delta_close_ticks"] == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Hora NY (no UTC) y DST
# ---------------------------------------------------------------------------

def test_hour_uses_america_new_york_not_utc():
    # 22:30 UTC en enero (EST, UTC-5) = 17:30 NY -> hour_ny debe ser 17, no 22.
    ts = pd.to_datetime(["2024-01-08 22:00:00", "2024-01-08 22:30:00"])
    variables, validity = _variables_and_validity(
        [
            {"timestamp": ts[0], "close": 100.00, "r_1m_valid": False},
            {"timestamp": ts[1], "close": 100.25, "r_1m_valid": True},
        ]
    )
    out = compute_tick_variables(variables, validity, TICK)
    assert int(out.iloc[1]["hour_ny"]) == 17
    assert int(out.iloc[1]["hour_ny"]) != 22


def test_dst_spring_transition_converts_correctly_both_sides():
    """2024-03-10 es la transicion de primavera (EST->EDT) en EE. UU.
    Un timestamp UTC fijo debe convertirse a horas NY distintas antes y
    despues de la transicion, reflejando el cambio de offset."""
    before = pd.Timestamp("2024-03-09 22:00:00")  # EST (UTC-5) -> 17:00 NY
    after = pd.Timestamp("2024-03-11 22:00:00")   # EDT (UTC-4) -> 18:00 NY
    variables, validity = _variables_and_validity(
        [
            {"timestamp": before, "close": 100.00, "r_1m_valid": False},
            {"timestamp": after, "close": 100.25, "r_1m_valid": False},
        ]
    )
    out = compute_tick_variables(variables, validity, TICK)
    assert int(out.iloc[0]["hour_ny"]) == 17
    assert int(out.iloc[1]["hour_ny"]) == 18


def test_dst_fall_transition_converts_correctly_both_sides():
    """2024-11-03 es la transicion de otoño (EDT->EST)."""
    before = pd.Timestamp("2024-11-02 21:00:00")  # EDT (UTC-4) -> 17:00 NY
    after = pd.Timestamp("2024-11-04 22:00:00")   # EST (UTC-5) -> 17:00 NY
    variables, validity = _variables_and_validity(
        [
            {"timestamp": before, "close": 100.00, "r_1m_valid": False},
            {"timestamp": after, "close": 100.25, "r_1m_valid": False},
        ]
    )
    out = compute_tick_variables(variables, validity, TICK)
    assert int(out.iloc[0]["hour_ny"]) == 17
    assert int(out.iloc[1]["hour_ny"]) == 17  # misma hora NY, offset UTC distinto


def test_year_grouping_is_correct():
    ts = pd.to_datetime(["2023-12-31 23:00:00", "2024-06-01 15:00:00"])  # 18:00 NY (EST) y 11:00 NY (EDT)
    variables, validity = _variables_and_validity(
        [
            {"timestamp": ts[0], "close": 100.00, "r_1m_valid": False},
            {"timestamp": ts[1], "close": 100.25, "r_1m_valid": False},
        ]
    )
    out = compute_tick_variables(variables, validity, TICK)
    assert int(out.iloc[0]["year_ny"]) == 2023
    assert int(out.iloc[1]["year_ny"]) == 2024

    by_year = build_by_year_table(
        out.assign(r_1m=[np.nan, 0.0], r_1m_valid=[False, True]), TICK
    )
    # Ambos años aparecen en la tabla (siempre se reporta n, incluso 0);
    # solo 2024 tiene barras con r_1m_valid=True.
    assert set(by_year["year_ny"]) == {2023, 2024}
    assert int(by_year.set_index("year_ny").loc[2023, "n"]) == 0
    assert int(by_year.set_index("year_ny").loc[2024, "n"]) == 1


# ---------------------------------------------------------------------------
# Unidades dimensionalmente correctas
# ---------------------------------------------------------------------------

def _synthetic_valid_rows(closes, highs=None, lows=None, r_1ms=None):
    n = len(closes)
    ts = pd.date_range("2024-01-08 15:00:00", periods=n, freq="1min")
    highs = highs or [c + 1.0 for c in closes]
    lows = lows or [c - 1.0 for c in closes]
    r_1ms = r_1ms if r_1ms is not None else [np.nan] + [
        float(np.log(closes[i] / closes[i - 1])) for i in range(1, n)
    ]
    rows = [
        {"timestamp": ts[i], "close": closes[i], "high": highs[i], "low": lows[i],
         "r_1m_valid": i > 0, "r_1m": r_1ms[i]}
        for i in range(n)
    ]
    variables, validity = _variables_and_validity(rows)
    return compute_tick_variables(variables, validity, TICK)


def test_tick_to_sigma_points_uses_points_over_points():
    """tick_size (puntos) / std(delta_close_points) (puntos): ambos en la
    MISMA unidad -- el cociente debe coincidir con el calculo manual en
    puntos, y NO con el resultado (incorrecto) de dividir tick_size entre
    std(r_1m) directamente."""
    closes = [100.0, 100.25, 100.75, 100.50, 101.00, 100.75]
    out = _synthetic_valid_rows(closes)
    summary = summarize_resolution(out, TICK)

    manual_sigma_points = out.loc[out["r_1m_valid"], "delta_close_points"].std(ddof=1)
    assert summary["sigma_delta_close_points"] == pytest.approx(manual_sigma_points)
    assert summary["tick_to_sigma_points"] == pytest.approx(TICK / manual_sigma_points)

    # La version incorrecta (mezclar unidades) daria un numero MUY distinto
    # (r_1m es del orden de 1e-3, tick_size del orden de 0.25 -- el cociente
    # "erroneo" seria ordenes de magnitud mayor que el correcto).
    sigma_r = out.loc[out["r_1m_valid"], "r_1m"].std(ddof=1)
    wrong_ratio = TICK / sigma_r
    assert abs(summary["tick_to_sigma_points"] - wrong_ratio) > 1.0


def test_tick_to_sigma_return_uses_return_units_consistently():
    closes = [100.0, 100.25, 100.75, 100.50, 101.00, 100.75]
    out = _synthetic_valid_rows(closes)
    summary = summarize_resolution(out, TICK)

    close_repr = out.loc[out["r_1m_valid"], "close"].median()
    expected_tick_return = float(np.log((close_repr + TICK) / close_repr))
    assert summary["tick_return_representative"] == pytest.approx(expected_tick_return)
    sigma_r = out.loc[out["r_1m_valid"], "r_1m"].std(ddof=1)
    assert summary["tick_to_sigma_return"] == pytest.approx(expected_tick_return / sigma_r)


def test_tick_to_median_range_uses_points_over_points():
    closes = [100.0, 100.25, 100.75, 100.50]
    highs = [101.0, 102.0, 103.0, 101.5]
    lows = [99.0, 99.5, 100.0, 99.0]
    out = _synthetic_valid_rows(closes, highs=highs, lows=lows)
    summary = summarize_resolution(out, TICK)
    manual_median_range = out.loc[out["r_1m_valid"], "range_points"].median()
    assert summary["median_range_points"] == pytest.approx(manual_median_range)
    assert summary["tick_to_median_range"] == pytest.approx(TICK / manual_median_range)


def test_zero_sigma_gives_explicit_infinity_not_silent_division():
    """Si TODOS los movimientos de un grupo son 0 puntos, sigma=0 -- el
    cociente debe ser +inf explicito, no una excepcion ni un NaN mudo."""
    closes = [100.0, 100.0, 100.0, 100.0]
    out = _synthetic_valid_rows(closes, r_1ms=[np.nan, 0.0, 0.0, 0.0])
    summary = summarize_resolution(out, TICK)
    assert summary["sigma_delta_close_points"] == 0.0
    assert summary["tick_to_sigma_points"] == float("inf")


def test_zero_median_range_gives_explicit_infinity():
    closes = [100.0, 100.25]
    out = _synthetic_valid_rows(closes, highs=[100.0, 100.25], lows=[100.0, 100.25])
    summary = summarize_resolution(out, TICK)
    assert summary["median_range_points"] == 0.0
    assert summary["tick_to_median_range"] == float("inf")


def test_central_range_distinct_values_uses_fixed_predefined_range():
    """El rango central esta FIJO (CENTRAL_RANGE_TICKS), no se elige a partir
    de los resultados."""
    assert CENTRAL_RANGE_TICKS == 5
    closes = [100.0, 100.25, 100.50, 100.75, 101.00, 103.00]  # el ultimo salto (2 pts = 8 ticks) queda FUERA
    out = _synthetic_valid_rows(closes)
    summary = summarize_resolution(out, TICK)
    ticks_in_central = out.loc[out["r_1m_valid"] & (out["delta_close_ticks"].abs() <= 5), "delta_close_ticks"]
    assert summary["n_distinct_ticks_central"] == ticks_in_central.round().astype(int).nunique()


# ---------------------------------------------------------------------------
# Forward-fill: confirmado vs candidato
# ---------------------------------------------------------------------------

def test_forward_fill_confirmed_not_confused_with_candidate():
    closes = [100.0, 100.0, 100.0]
    out = _synthetic_valid_rows(closes, r_1ms=[np.nan, 0.0, 0.0])
    inactive_mask = pd.DataFrame(
        {
            "source_file": ["f1", "f1", "f1"],
            "expected_ts_utc": out["timestamp"].dt.tz_localize("UTC"),
            "trading_date": out["trading_date"],
            "status": ["PRESENTE"] * 3,
            "category": ["ACTIVA", "CANDIDATO_FORWARD_FILL", "FORWARD_FILL_CONFIRMADO"],
        }
    )
    tick_df = attach_forward_fill_flags(out, inactive_mask)
    comparison = compare_with_without_forward_fill(tick_df, TICK)
    assert comparison.n_confirmed_forward_fill == 1
    assert comparison.n_candidate_forward_fill == 1
    assert "1 barra(s) con FORWARD_FILL CONFIRMADO" in comparison.note


def test_forward_fill_reports_explicit_declaration_when_none_confirmed():
    closes = [100.0, 100.0, 100.0]
    out = _synthetic_valid_rows(closes, r_1ms=[np.nan, 0.0, 0.0])
    inactive_mask = pd.DataFrame(
        {
            "source_file": ["f1", "f1", "f1"],
            "expected_ts_utc": out["timestamp"].dt.tz_localize("UTC"),
            "trading_date": out["trading_date"],
            "status": ["PRESENTE"] * 3,
            "category": ["ACTIVA", "CANDIDATO_FORWARD_FILL", "ACTIVA"],
        }
    )
    tick_df = attach_forward_fill_flags(out, inactive_mask)
    comparison = compare_with_without_forward_fill(tick_df, TICK)
    assert comparison.n_confirmed_forward_fill == 0
    assert "No existen barras confirmadas como FORWARD_FILL" in comparison.note
    assert comparison.global_with_all["zero_fraction"] == comparison.global_excluding_candidates["zero_fraction"]


# ---------------------------------------------------------------------------
# Proteccion del hold-out + conservacion
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path, research_files, holdout_files=None, boundary_utc="2099-01-01 00:00:00"):
    config_dir = tmp_path / "configs"
    config_dir.mkdir(exist_ok=True)
    config_yaml = {
        "instrument": "MNQ",
        "raw_data": {
            "dir": "data/raw/mnq", "file_pattern": "*.Last.txt", "separator": ";",
            "columns": ["timestamp", "open", "high", "low", "close", "volume"],
            "timestamp_format": TS_FORMAT, "timestamp_raw_timezone": "UTC",
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
        "tda02": {
            "calendar_name": "CME_Equity", "calendar_buffer_days": 10,
            "output_dir_reports": "reports/mnq", "output_dir_interim": "data/interim/mnq",
            "coverage_report": "TDA02_cobertura.md", "coverage_by_year_csv": "TDA02_cobertura_por_anio.csv",
            "coverage_by_month_csv": "TDA02_cobertura_por_mes.csv", "gaps_csv": "TDA02_huecos.csv",
            "incomplete_days_csv": "TDA02_dias_incompletos.csv", "out_of_grid_csv": "TDA02_barras_fuera_de_grilla.csv",
            "dst_evidence_csv": "TDA02_dst_evidencia.csv", "heatmap_png": "TDA02_heatmap_completitud.png",
            "inactive_bar_mask": "tda02_barra_inactiva_mask.parquet",
        },
        "tda04": {
            "output_dir_reports": "reports/mnq", "output_dir_interim": "data/interim/mnq",
            "report_name": "TDA04_variables_analisis.md", "variables_parquet_name": "tda04_variables_1m.parquet",
            "validity_mask_parquet_name": "tda04_return_validity_mask.parquet",
            "losses_by_cause_csv": "TDA04_perdidas_por_causa.csv", "th07_r_vs_R_csv": "TDA04_th07_r_vs_R.csv",
        },
        "tda05": {
            "output_dir_reports": "reports/mnq", "output_dir_interim": "data/interim/mnq",
            "report_name": "TDA05_resolucion_discrecion.md", "global_csv": "TDA05_resolucion_global.csv",
            "by_hour_csv": "TDA05_resolucion_por_hora.csv", "by_year_csv": "TDA05_resolucion_por_anio.csv",
            "by_year_hour_csv": "TDA05_resolucion_anio_hora.csv", "histogram_png": "TDA05_histograma_ticks.png",
            "tick_variables_parquet_name": "tda05_tick_variables.parquet",
        },
    }
    config_path = config_dir / "mnq_snapshot.yaml"
    config_path.write_text(yaml.safe_dump(config_yaml), encoding="utf-8")
    return load_config(config_path)


def _write_tda04_and_tda02_artifacts(config, variables: pd.DataFrame, validity: pd.DataFrame) -> None:
    config.interim_dir.mkdir(parents=True, exist_ok=True)
    variables.to_parquet(config.tda04_variables_parquet_path, index=False)
    validity.to_parquet(config.tda04_validity_mask_parquet_path, index=False)
    _empty_inactive_mask().to_parquet(config.tda02_inactive_bar_mask_path, index=False)


def test_run_tda05_raises_when_research_and_holdout_overlap(tmp_path):
    config = _make_config(
        tmp_path, research_files=["00_mnq_03_24.Last.txt"],
        holdout_files=["00_mnq_03_24.Last.txt", "no_existe_en_disco.Last.txt"],
    )
    with pytest.raises(HoldoutIsolationError, match="disjuntos"):
        run_tda05_analysis(config)


def test_run_tda05_raises_when_row_reaches_holdout_boundary(tmp_path):
    config = _make_config(
        tmp_path, research_files=["00_mnq_03_24.Last.txt"],
        holdout_files=["no_existe_en_disco.Last.txt"],
        boundary_utc="2024-01-08 15:01:00",
    )
    closes = [100.0, 100.25]
    ts = pd.date_range("2024-01-08 15:00:00", periods=2, freq="1min")
    variables, validity = _variables_and_validity(
        [
            {"timestamp": ts[0], "close": closes[0], "r_1m_valid": False, "source_file": "00_mnq_03_24.Last.txt"},
            {"timestamp": ts[1], "close": closes[1], "r_1m_valid": True, "source_file": "00_mnq_03_24.Last.txt", "r_1m": 0.0025},
        ]
    )
    _write_tda04_and_tda02_artifacts(config, variables, validity)
    with pytest.raises(HoldoutIsolationError, match="frontera de hold-out"):
        run_tda05_analysis(config)
    assert not (config.raw_dir / "no_existe_en_disco.Last.txt").exists()


def test_run_tda05_never_opens_any_raw_or_holdout_file(tmp_path):
    config = _make_config(
        tmp_path, research_files=["00_mnq_03_24.Last.txt"],
        holdout_files=["archivo_que_no_existe_en_disco.Last.txt"],
    )
    ts = pd.date_range("2024-01-08 15:00:00", periods=3, freq="1min")
    variables, validity = _variables_and_validity(
        [
            {"timestamp": ts[0], "close": 100.0, "r_1m_valid": False, "source_file": "00_mnq_03_24.Last.txt"},
            {"timestamp": ts[1], "close": 100.25, "r_1m_valid": True, "source_file": "00_mnq_03_24.Last.txt", "r_1m": 0.0025},
            {"timestamp": ts[2], "close": 100.50, "r_1m_valid": True, "source_file": "00_mnq_03_24.Last.txt", "r_1m": 0.0025},
        ]
    )
    _write_tda04_and_tda02_artifacts(config, variables, validity)
    assert not (config.raw_dir / "00_mnq_03_24.Last.txt").exists()
    result = run_tda05_analysis(config)
    assert len(result.tick_df) == 3
    assert int(result.global_table.iloc[0]["n"]) == 2


def test_row_count_and_traceability_conserved(tmp_path):
    config = _make_config(tmp_path, research_files=["00_mnq_03_24.Last.txt"])
    ts = pd.date_range("2024-01-08 15:00:00", periods=4, freq="1min")
    variables, validity = _variables_and_validity(
        [
            {"timestamp": ts[0], "close": 100.0, "r_1m_valid": False, "source_file": "00_mnq_03_24.Last.txt", "contract": "H24"},
            {"timestamp": ts[1], "close": 100.25, "r_1m_valid": True, "source_file": "00_mnq_03_24.Last.txt", "contract": "H24", "r_1m": 0.0025},
            {"timestamp": ts[2], "close": 100.00, "r_1m_valid": True, "source_file": "00_mnq_03_24.Last.txt", "contract": "H24", "r_1m": -0.0025},
            {"timestamp": ts[3], "close": 100.50, "r_1m_valid": True, "source_file": "00_mnq_03_24.Last.txt", "contract": "H24", "r_1m": 0.005},
        ]
    )
    _write_tda04_and_tda02_artifacts(config, variables, validity)
    result = run_tda05_analysis(config)
    assert len(result.tick_df) == 4
    assert set(result.tick_df.columns) >= {"timestamp", "source_file", "contract", "trading_date", "hour_ny", "year_ny"}
    assert int(result.global_table.iloc[0]["n"]) == 3


# ---------------------------------------------------------------------------
# STOP-5 watchlist -- debe revisar tambien by_year_hour (correccion puntual)
# ---------------------------------------------------------------------------

def _watchlist_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_stop5_watchlist_flags_segment_only_visible_at_year_hour_granularity():
    """Un segmento año×hora muy discreto puede quedar diluido en las vistas por hora y por año por separado."""
    by_hour = _watchlist_frame(
        [
            {"hour_ny": 23, "n": 90000, "zero_fraction": 0.08, "tick_to_sigma_points": 0.10, "median_abs_ticks": 6.0},
        ]
    )
    by_year = _watchlist_frame(
        [
            {"year_ny": 2019, "n": 7000, "zero_fraction": 0.18, "tick_to_sigma_points": 0.22, "median_abs_ticks": 1.0},
        ]
    )
    by_year_hour = _watchlist_frame(
        [
            # Segmento localmente muy discreto (n pequeño, año parcial 2019 x hora de baja liquidez).
            {"year_ny": 2019, "hour_ny": 23, "n": 297, "zero_fraction": 0.55, "tick_to_sigma_points": 2.5, "median_abs_ticks": 1.0},
            {"year_ny": 2020, "hour_ny": 23, "n": 84000, "zero_fraction": 0.07, "tick_to_sigma_points": 0.09, "median_abs_ticks": 6.0},
        ]
    )
    watchlist = build_stop5_watchlist(by_hour, by_year, by_year_hour)
    # ni by_hour ni by_year, tomadas solas, cruzan ningun umbral -- solo la fila año×hora lo hace.
    assert len(watchlist) == 1
    row = watchlist.iloc[0]
    assert row["segment"] == "year_ny=2019 hour_ny=23"
    assert int(row["n"]) == 297
    assert row["tick_to_sigma_points"] == pytest.approx(2.5)


def test_stop5_watchlist_empty_when_no_table_crosses_thresholds():
    """Reproduce la conclusion del informe: 0 horas, 0 años y 0 pares año×hora señalados en el conjunto real."""
    by_hour = _watchlist_frame(
        [{"hour_ny": h, "n": 80000, "zero_fraction": 0.05, "tick_to_sigma_points": 0.08, "median_abs_ticks": 6.0} for h in range(2)]
    )
    by_year = _watchlist_frame(
        [{"year_ny": y, "n": 300000, "zero_fraction": 0.04, "tick_to_sigma_points": 0.05, "median_abs_ticks": 6.0} for y in (2020, 2021)]
    )
    by_year_hour = _watchlist_frame(
        [
            {"year_ny": 2019, "hour_ny": 23, "n": 297, "zero_fraction": 0.33, "tick_to_sigma_points": 0.72, "median_abs_ticks": 1.0},
            {"year_ny": 2019, "hour_ny": 17, "n": 5, "zero_fraction": 0.40, "tick_to_sigma_points": 0.67, "median_abs_ticks": 1.0},
        ]
    )
    watchlist = build_stop5_watchlist(by_hour, by_year, by_year_hour)
    assert len(watchlist) == 0


def test_by_year_hour_table_structure_from_synthetic_data():
    ts = pd.date_range("2024-01-08 15:00:00", periods=3, freq="1min")
    variables, validity = _variables_and_validity(
        [
            {"timestamp": ts[0], "close": 100.0, "r_1m_valid": False},
            {"timestamp": ts[1], "close": 100.25, "r_1m_valid": True, "r_1m": 0.0025},
            {"timestamp": ts[2], "close": 100.50, "r_1m_valid": True, "r_1m": 0.0025},
        ]
    )
    out = compute_tick_variables(variables, validity, TICK)
    table = build_by_year_hour_table(out, TICK)
    assert {"year_ny", "hour_ny", "n"} <= set(table.columns)
    assert int(table["n"].sum()) == 2
