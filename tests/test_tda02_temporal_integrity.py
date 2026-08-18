"""Tests de TDA-02 -- integridad del eje temporal y del calendario.

Igual que ``test_tda01_temporal_semantics.py``: se usan datos sinteticos
minimos para aislar cada propiedad (clasificacion causal de huecos,
cobertura, mascara de barras inactivas, STOP-2, proteccion del hold-out),
mas un puñado de pruebas que usan el calendario CME_Equity real (via
``session_calendar.py``) para verificar la clasificacion contra fechas
conocidas (un feriado real, un cierre anticipado real, un fin de semana
real).
"""
from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd
import pytest
import yaml

from ohlcv_dataroad.config import load_config
from ohlcv_dataroad.ingest.holdout_guard import HoldoutIsolationError
from ohlcv_dataroad.ingest.session_calendar import (
    build_session_schedule,
    expected_bar_grid_frame,
    full_holidays_in_range,
)
from ohlcv_dataroad.ingest.tda02_temporal_integrity import (
    CAUSE_DAILY_MAINTENANCE,
    CAUSE_EARLY_CLOSE,
    CAUSE_HOLIDAY,
    CAUSE_MISSING_TRADING_DAY,
    CAUSE_SECONDARY_MAINTENANCE_BREAK,
    CAUSE_UNKNOWN,
    CAUSE_WEEKEND,
    CONFIDENCE_HIGH,
    analyze_inactive_bar_candidates,
    check_stop2,
    classify_gaps,
    compute_file_coverage,
    find_out_of_grid_bars,
    run_tda02_analysis,
)

TS_FORMAT = "%Y%m%d %H%M%S"


def _write(tmp_path: Path, name: str, lines: list[str]) -> Path:
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _make_config(
    tmp_path: Path,
    research_files: list[str],
    holdout_files: list[str] | None = None,
    boundary_utc: str = "2099-01-01 00:00:00",
):
    config_dir = tmp_path / "configs"
    config_dir.mkdir(exist_ok=True)
    config_yaml = {
        "instrument": "MNQ",
        "raw_data": {
            "dir": "data/raw/mnq",
            "file_pattern": "*.Last.txt",
            "separator": ";",
            "columns": ["timestamp", "open", "high", "low", "close", "volume"],
            "timestamp_format": TS_FORMAT,
            "timestamp_raw_timezone": "UTC",
        },
        "instrument_spec": {"tick_size": 0.25, "tick_size_source": "test"},
        "holdout": {
            "boundary_utc": boundary_utc,
            "boundary_source": "test",
            "research_files": research_files,
            "holdout_files": holdout_files or [],
        },
        "tda00": {
            "output_dir_reports": "reports/mnq",
            "output_dir_interim": "data/interim/mnq",
            "inventory_report": "TDA00_inventario.md",
            "violations_report": "TDA00_violaciones.csv",
            "per_file_summary": "TDA00_resumen_por_archivo.csv",
            "bad_data_mask": "tda00_bad_data_mask.parquet",
        },
        "tda02": {
            "calendar_name": "CME_Equity",
            "calendar_buffer_days": 10,
            "output_dir_reports": "reports/mnq",
            "output_dir_interim": "data/interim/mnq",
            "coverage_report": "TDA02_cobertura.md",
            "coverage_by_year_csv": "TDA02_cobertura_por_anio.csv",
            "coverage_by_month_csv": "TDA02_cobertura_por_mes.csv",
            "gaps_csv": "TDA02_huecos.csv",
            "incomplete_days_csv": "TDA02_dias_incompletos.csv",
            "out_of_grid_csv": "TDA02_barras_fuera_de_grilla.csv",
            "dst_evidence_csv": "TDA02_dst_evidencia.csv",
            "heatmap_png": "TDA02_heatmap_completitud.png",
            "inactive_bar_mask": "tda02_barra_inactiva_mask.parquet",
        },
    }
    config_path = config_dir / "mnq_snapshot.yaml"
    config_path.write_text(yaml.safe_dump(config_yaml), encoding="utf-8")
    return load_config(config_path)


# ---------------------------------------------------------------------------
# classify_gaps -- cada causa candidata, con el calendario CME_Equity real
# ---------------------------------------------------------------------------

def _schedule_and_holidays(start, end):
    schedule = build_session_schedule(start, end)
    holidays = set(full_holidays_in_range(start, end))
    return schedule, holidays


def test_classify_daily_maintenance_gap():
    schedule, holidays = _schedule_and_holidays("2024-01-01", "2024-01-31")
    gaps = pd.DataFrame(
        {
            "source_file": ["a.txt"],
            "before_ts_utc": [pd.Timestamp("2024-01-09 22:00:00", tz="UTC")],  # martes 17:00 NY
            "after_ts_utc": [pd.Timestamp("2024-01-09 23:01:00", tz="UTC")],  # 18:01 NY -> miercoles
            "gap_minutes": [61.0],
        }
    )
    result = classify_gaps(gaps, schedule, holidays)
    assert result.iloc[0]["cause"] == CAUSE_DAILY_MAINTENANCE
    assert result.iloc[0]["confidence"] == CONFIDENCE_HIGH


def test_classify_weekend_gap():
    schedule, holidays = _schedule_and_holidays("2024-01-01", "2024-01-31")
    friday_close = schedule.table.loc[datetime.date(2024, 1, 5), "market_close"]
    monday_open = schedule.table.loc[datetime.date(2024, 1, 8), "market_open"] + pd.Timedelta(minutes=1)
    gaps = pd.DataFrame(
        {
            "source_file": ["a.txt"],
            "before_ts_utc": [friday_close],
            "after_ts_utc": [monday_open],
            "gap_minutes": [(monday_open - friday_close).total_seconds() / 60],
        }
    )
    result = classify_gaps(gaps, schedule, holidays)
    assert result.iloc[0]["cause"] == CAUSE_WEEKEND
    assert result.iloc[0]["confidence"] == CONFIDENCE_HIGH


def test_classify_holiday_gap_christmas_2023():
    schedule, holidays = _schedule_and_holidays("2023-12-01", "2023-12-31")
    before_date = datetime.date(2023, 12, 22)  # viernes antes de Navidad (lunes 25)
    after_date = datetime.date(2023, 12, 26)  # martes (26 dic es feriado observado)
    before_ts = schedule.table.loc[before_date, "market_close"]
    after_ts = schedule.table.loc[after_date, "market_open"] + pd.Timedelta(minutes=1)
    gaps = pd.DataFrame(
        {
            "source_file": ["a.txt"],
            "before_ts_utc": [before_ts],
            "after_ts_utc": [after_ts],
            "gap_minutes": [(after_ts - before_ts).total_seconds() / 60],
        }
    )
    result = classify_gaps(gaps, schedule, holidays)
    assert result.iloc[0]["cause"] == CAUSE_HOLIDAY


def test_classify_early_close_gap_thanksgiving_2023():
    schedule, holidays = _schedule_and_holidays("2023-11-01", "2023-11-30")
    before_date = datetime.date(2023, 11, 22)
    after_date = datetime.date(2023, 11, 23)  # Accion de Gracias, cierre anticipado
    before_ts = schedule.table.loc[before_date, "market_close"]
    after_ts = schedule.table.loc[after_date, "market_open"] + pd.Timedelta(minutes=1)
    gaps = pd.DataFrame(
        {
            "source_file": ["a.txt"],
            "before_ts_utc": [before_ts],
            "after_ts_utc": [after_ts],
            "gap_minutes": [(after_ts - before_ts).total_seconds() / 60],
        }
    )
    result = classify_gaps(gaps, schedule, holidays)
    assert result.iloc[0]["cause"] == CAUSE_DAILY_MAINTENANCE  # el 22 no es especial

    # El propio 23 (dia de cierre anticipado) -> hueco DESPUES de su cierre.
    after_date2 = datetime.date(2023, 11, 24)
    before_ts2 = schedule.table.loc[after_date, "market_close"]
    after_ts2 = schedule.table.loc[after_date2, "market_open"] + pd.Timedelta(minutes=1)
    gaps2 = pd.DataFrame(
        {
            "source_file": ["a.txt"],
            "before_ts_utc": [before_ts2],
            "after_ts_utc": [after_ts2],
            "gap_minutes": [(after_ts2 - before_ts2).total_seconds() / 60],
        }
    )
    result2 = classify_gaps(gaps2, schedule, holidays)
    assert result2.iloc[0]["cause"] == CAUSE_EARLY_CLOSE


def test_classify_secondary_maintenance_break_when_it_matches_calendar():
    schedule, holidays = _schedule_and_holidays("2020-01-01", "2020-01-31")
    row = schedule.table.loc[datetime.date(2020, 1, 9)]
    gaps = pd.DataFrame(
        {
            "source_file": ["a.txt"],
            "before_ts_utc": [row["break_start"]],
            "after_ts_utc": [row["break_end"] + pd.Timedelta(minutes=1)],
            "gap_minutes": [(row["break_end"] + pd.Timedelta(minutes=1) - row["break_start"]).total_seconds() / 60],
        }
    )
    result = classify_gaps(gaps, schedule, holidays)
    assert result.iloc[0]["cause"] == CAUSE_SECONDARY_MAINTENANCE_BREAK


def test_classify_secondary_break_not_applied_after_abolition_cutoff():
    """Un hueco con la firma horaria EXACTA del break secundario, pero en
    una fecha de negociacion posterior a la eliminacion documentada
    (2021-06-28), NO debe clasificarse como SECONDARY_MAINTENANCE_BREAK
    -- el break ya no existia estructuralmente ese dia."""
    schedule, holidays = _schedule_and_holidays("2021-06-28", "2021-06-29")
    row = schedule.table.loc[datetime.date(2021, 6, 28)]
    before_ts = row["break_start"]
    after_ts = row["break_end"] + pd.Timedelta(minutes=1)
    gaps = pd.DataFrame(
        {
            "source_file": ["a.txt"],
            "before_ts_utc": [before_ts],
            "after_ts_utc": [after_ts],
            "gap_minutes": [(after_ts - before_ts).total_seconds() / 60],
        }
    )
    result = classify_gaps(gaps, schedule, holidays)
    assert result.iloc[0]["cause"] != CAUSE_SECONDARY_MAINTENANCE_BREAK


def test_classify_unknown_when_intraday_gap_does_not_match_any_break():
    schedule, holidays = _schedule_and_holidays("2024-01-01", "2024-01-31")
    # Un hueco de 2 minutos a media mañana, mismo trading_date en ambos lados.
    base = pd.Timestamp("2024-01-09 15:00:00", tz="UTC")
    gaps = pd.DataFrame(
        {
            "source_file": ["a.txt"],
            "before_ts_utc": [base],
            "after_ts_utc": [base + pd.Timedelta(minutes=3)],
            "gap_minutes": [3.0],
        }
    )
    result = classify_gaps(gaps, schedule, holidays)
    assert result.iloc[0]["cause"] == CAUSE_UNKNOWN


def test_classify_missing_trading_day_when_a_full_valid_session_has_zero_bars():
    schedule, holidays = _schedule_and_holidays("2024-01-01", "2024-01-31")
    # antes: cierre del lunes 8; despues: apertura del miercoles 10 --
    # el martes 9 (dia habil valido) queda completamente saltado.
    before_ts = schedule.table.loc[datetime.date(2024, 1, 8), "market_close"]
    after_ts = schedule.table.loc[datetime.date(2024, 1, 10), "market_open"] + pd.Timedelta(minutes=1)
    gaps = pd.DataFrame(
        {
            "source_file": ["a.txt"],
            "before_ts_utc": [before_ts],
            "after_ts_utc": [after_ts],
            "gap_minutes": [(after_ts - before_ts).total_seconds() / 60],
        }
    )
    result = classify_gaps(gaps, schedule, holidays)
    assert result.iloc[0]["cause"] == CAUSE_MISSING_TRADING_DAY


def test_classify_unknown_when_edge_falls_outside_any_session():
    schedule, holidays = _schedule_and_holidays("2024-01-01", "2024-01-31")
    # Sabado a mediodia: no pertenece a ninguna sesion.
    before_ts = pd.Timestamp("2024-01-06 16:00:00", tz="UTC")
    after_ts = pd.Timestamp("2024-01-06 16:05:00", tz="UTC")
    gaps = pd.DataFrame(
        {
            "source_file": ["a.txt"],
            "before_ts_utc": [before_ts],
            "after_ts_utc": [after_ts],
            "gap_minutes": [5.0],
        }
    )
    result = classify_gaps(gaps, schedule, holidays)
    assert result.iloc[0]["cause"] == CAUSE_UNKNOWN


# ---------------------------------------------------------------------------
# compute_file_coverage -- recorte a [file_start, file_end]
# ---------------------------------------------------------------------------

def test_coverage_is_clipped_to_file_range_not_full_calendar():
    """Un archivo que solo cubre UNA sesion completa no debe generar cobertura
    esperada para ningun otro dia del calendario, aunque el horario
    abarque un rango mas amplio."""
    schedule, _ = _schedule_and_holidays("2024-01-08", "2024-01-08")
    open_ts = schedule.table.loc[datetime.date(2024, 1, 8), "market_open"]
    close_ts = schedule.table.loc[datetime.date(2024, 1, 8), "market_close"]

    from ohlcv_dataroad.ingest.session_calendar import expected_bar_grid_frame

    full_grid = expected_bar_grid_frame(schedule)

    all_minutes = pd.date_range(open_ts + pd.Timedelta(minutes=1), close_ts, freq="1min")
    rows_f = pd.DataFrame(
        {
            "source_file": ["a.txt"] * len(all_minutes),
            "timestamp": all_minutes.tz_localize(None),
        }
    )
    coverage = compute_file_coverage(rows_f, full_grid, schedule)
    assert coverage.expected_minutes == len(all_minutes)
    assert coverage.present_minutes == len(all_minutes)
    assert coverage.missing_minutes == 0


def test_coverage_detects_missing_minutes_within_file_range():
    schedule, _ = _schedule_and_holidays("2024-01-08", "2024-01-08")
    open_ts = schedule.table.loc[datetime.date(2024, 1, 8), "market_open"]
    close_ts = schedule.table.loc[datetime.date(2024, 1, 8), "market_close"]

    from ohlcv_dataroad.ingest.session_calendar import expected_bar_grid_frame

    full_grid = expected_bar_grid_frame(schedule)

    all_minutes = pd.date_range(open_ts + pd.Timedelta(minutes=1), close_ts, freq="1min")
    kept = all_minutes.delete(5)  # elimina UN minuto interno del rango del archivo
    rows_f = pd.DataFrame({"source_file": ["a.txt"] * len(kept), "timestamp": kept.tz_localize(None)})
    coverage = compute_file_coverage(rows_f, full_grid, schedule)
    assert coverage.missing_minutes == 1


# ---------------------------------------------------------------------------
# find_out_of_grid_bars -- incluye ahora el tramo del break secundario
# ---------------------------------------------------------------------------

def test_out_of_grid_bars_flags_bar_inside_secondary_break_window():
    """Una barra real observada DENTRO del tramo de break excluido de la
    grilla (fecha pre-corte) no debe desaparecer de toda contabilidad: debe
    aparecer aqui, con motivo SECONDARY_BREAK_WINDOW."""
    schedule, _ = _schedule_and_holidays("2020-06-01", "2020-06-01")
    full_grid = expected_bar_grid_frame(schedule)
    row = schedule.table.loc[datetime.date(2020, 6, 1)]
    break_ts = row["break_start"] + pd.Timedelta(minutes=1)  # primer minuto excluido
    rows = pd.DataFrame(
        {"source_file": ["a.txt"], "timestamp": [break_ts.tz_localize(None)]}
    )
    out = find_out_of_grid_bars(rows, full_grid, schedule)
    assert len(out) == 1
    assert out.iloc[0]["reason"] == "SECONDARY_BREAK_WINDOW"


def test_out_of_grid_bars_flags_weekend_bar_as_no_session():
    schedule, _ = _schedule_and_holidays("2024-01-01", "2024-01-10")
    full_grid = expected_bar_grid_frame(schedule)
    rows = pd.DataFrame(
        {"source_file": ["a.txt"], "timestamp": [pd.Timestamp("2024-01-06 16:00:00")]}  # sabado
    )
    out = find_out_of_grid_bars(rows, full_grid, schedule)
    assert len(out) == 1
    assert out.iloc[0]["reason"] == "NO_SESSION"


def test_out_of_grid_bars_does_not_flag_bar_inside_grid():
    schedule, _ = _schedule_and_holidays("2024-01-08", "2024-01-08")
    full_grid = expected_bar_grid_frame(schedule)
    ts = full_grid["expected_ts_utc"].iloc[10].tz_localize(None)
    rows = pd.DataFrame({"source_file": ["a.txt"], "timestamp": [ts]})
    out = find_out_of_grid_bars(rows, full_grid, schedule)
    assert len(out) == 0


# ---------------------------------------------------------------------------
# analyze_inactive_bar_candidates -- TH04
# ---------------------------------------------------------------------------

def test_isolated_flat_bar_is_not_a_long_run():
    rows = pd.DataFrame(
        {
            "source_file": ["a.txt"] * 3,
            "timestamp": pd.to_datetime(["2020-01-01 09:00:00", "2020-01-01 09:01:00", "2020-01-01 09:02:00"]),
            "open": [100.0, 100.0, 100.50],
            "high": [100.0, 100.0, 100.75],
            "low": [100.0, 100.0, 100.25],
            "close": [100.0, 100.0, 100.60],
            "volume": [5, 3, 7],
        }
    )
    evidence = analyze_inactive_bar_candidates(rows, long_run_threshold=6)
    assert len(evidence.flat_bars) == 2  # las dos primeras barras son O=H=L=C
    assert len(evidence.long_runs) == 0
    assert evidence.zero_volume_count == 0
    assert "NEGATIVO" in evidence.verdict


def test_long_run_with_constant_volume_is_flagged_as_candidate():
    ts = pd.date_range("2020-01-01 09:00:00", periods=8, freq="1min")
    rows = pd.DataFrame(
        {
            "source_file": ["a.txt"] * 8,
            "timestamp": ts,
            "open": [100.0] * 8,
            "high": [100.0] * 8,
            "low": [100.0] * 8,
            "close": [100.0] * 8,
            "volume": [0] * 8,  # volumen constante (0) dentro de la racha
        }
    )
    evidence = analyze_inactive_bar_candidates(rows, long_run_threshold=6)
    assert evidence.zero_volume_count == 8
    assert "volume == 0" in evidence.verdict


def test_long_run_with_varying_volume_stays_indeterminate_not_confirmed():
    ts = pd.date_range("2020-01-01 09:00:00", periods=8, freq="1min")
    rows = pd.DataFrame(
        {
            "source_file": ["a.txt"] * 8,
            "timestamp": ts,
            "open": [100.0] * 8,
            "high": [100.0] * 8,
            "low": [100.0] * 8,
            "close": [100.0] * 8,
            "volume": [3, 7, 1, 9, 2, 5, 4, 6],  # volumen VARIA -> no confirma forward-fill
        }
    )
    evidence = analyze_inactive_bar_candidates(rows, long_run_threshold=6)
    assert len(evidence.long_runs) == 8
    assert "INDETERMINADA" in evidence.verdict
    assert "CONFIRMADO" not in evidence.verdict.split("no se fuerza")[0]


# ---------------------------------------------------------------------------
# check_stop2
# ---------------------------------------------------------------------------

def test_stop2_not_triggered_when_calendar_gaps_match_exactly():
    gaps = pd.DataFrame(
        {"cause": [CAUSE_DAILY_MAINTENANCE] * 10, "confidence": [CONFIDENCE_HIGH] * 10}
    )
    result = check_stop2(gaps, threshold=0.90)
    assert result["triggered"] is False
    assert result["exact_match_fraction"] == 1.0


def test_stop2_triggered_when_calendar_gaps_mostly_do_not_match():
    gaps = pd.DataFrame(
        {
            "cause": [CAUSE_DAILY_MAINTENANCE] * 10,
            "confidence": ["MEDIA"] * 9 + [CONFIDENCE_HIGH] * 1,
        }
    )
    result = check_stop2(gaps, threshold=0.90)
    assert result["triggered"] is True


def test_stop2_ignores_non_calendar_causes():
    gaps = pd.DataFrame(
        {"cause": [CAUSE_UNKNOWN] * 10, "confidence": ["BAJA"] * 10}
    )
    result = check_stop2(gaps, threshold=0.90)
    assert result["calendar_gap_count"] == 0
    assert result["triggered"] is False


# ---------------------------------------------------------------------------
# run_tda02_analysis -- proteccion del hold-out (reutiliza holdout_guard.py)
# ---------------------------------------------------------------------------

def test_run_tda02_raises_when_research_and_holdout_overlap(tmp_path):
    raw_dir = tmp_path / "data" / "raw" / "mnq"
    raw_dir.mkdir(parents=True)
    _write(raw_dir, "00_mnq_03_20.Last.txt", [
        "20200102 090000;100.00;100.25;99.75;100.00;10",
    ])
    config = _make_config(
        tmp_path,
        research_files=["00_mnq_03_20.Last.txt"],
        holdout_files=["00_mnq_03_20.Last.txt", "no_existe_en_disco.Last.txt"],
    )
    with pytest.raises(HoldoutIsolationError, match="disjuntos"):
        run_tda02_analysis(config)


def test_run_tda02_raises_when_research_row_reaches_boundary(tmp_path):
    raw_dir = tmp_path / "data" / "raw" / "mnq"
    raw_dir.mkdir(parents=True)
    _write(raw_dir, "00_mnq_03_20.Last.txt", [
        "20200102 090000;100.00;100.25;99.75;100.00;10",
        "20200102 090100;100.00;100.50;99.75;100.25;20",
    ])
    config = _make_config(
        tmp_path,
        research_files=["00_mnq_03_20.Last.txt"],
        holdout_files=["no_existe_en_disco.Last.txt"],
        boundary_utc="2020-01-02 09:01:00",
    )
    with pytest.raises(HoldoutIsolationError, match="frontera de hold-out"):
        run_tda02_analysis(config)
    assert not (raw_dir / "no_existe_en_disco.Last.txt").exists()


def test_run_tda02_end_to_end_on_tiny_synthetic_file(tmp_path):
    """Integracion ligera: un archivo sintetico de un par de dias produce
    un resultado coherente (sin verificar numeros exactos del snapshot
    real, que vive en reports/mnq/TDA02_*)."""
    raw_dir = tmp_path / "data" / "raw" / "mnq"
    raw_dir.mkdir(parents=True)

    schedule = build_session_schedule("2024-01-08", "2024-01-09")
    lines = []
    for trading_date in [datetime.date(2024, 1, 8), datetime.date(2024, 1, 9)]:
        open_ts = schedule.table.loc[trading_date, "market_open"]
        close_ts = schedule.table.loc[trading_date, "market_close"]
        minutes = pd.date_range(open_ts + pd.Timedelta(minutes=1), close_ts, freq="1min")
        for m in minutes:
            lines.append(f"{m.strftime('%Y%m%d %H%M%S')};100.00;100.25;99.75;100.00;5")

    _write(raw_dir, "00_mnq_03_20.Last.txt", lines)
    config = _make_config(tmp_path, research_files=["00_mnq_03_20.Last.txt"])

    result = run_tda02_analysis(config)
    assert result.file_coverages[0].missing_minutes == 0
    assert result.stop2["triggered"] is False
    # El unico hueco interno de este archivo sintetico (limpio, sin huecos
    # dentro de cada sesion) es la transicion normal cierre->reapertura
    # entre el 8 y el 9 de enero.
    assert len(result.gaps_classified) == 1
    assert result.gaps_classified.iloc[0]["cause"] == CAUSE_DAILY_MAINTENANCE
