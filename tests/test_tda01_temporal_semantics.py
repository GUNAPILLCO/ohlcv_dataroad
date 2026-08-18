"""Tests de la evidencia forense de TDA-01 (semántica temporal).

Cada test usa datos sintéticos minúsculos para aislar una única propiedad:
detección de huecos por archivo (sin mezclar archivos), conversión DST-aware
a hora de Nueva York, y clasificación por magnitud. El objetivo no es
reproducir el hallazgo completo sobre datos reales (eso vive en
``reports/mnq/TDA01_convencion_temporal.md`` y en
``reports/mnq/TDA01_evidencia_gaps.csv``, generado por
``run_tda01_forensics.py``), sino garantizar que la lógica que produjo ese
hallazgo es correcta y no se rompe silenciosamente en el futuro.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from ohlcv_dataroad.config import load_config
from ohlcv_dataroad.ingest.holdout_guard import HoldoutIsolationError
from ohlcv_dataroad.ingest.tda01_temporal_semantics import (
    DAILY_MAINTENANCE_MAX_MINUTES,
    DAILY_MAINTENANCE_MIN_MINUTES,
    EARLY_CLOSE_AFTER_NY_TIME,
    EARLY_CLOSE_BEFORE_NY_TIME,
    LONG_GAP_MIN_MINUTES,
    attach_ny_wallclock,
    build_forensic_evidence,
    classify_gap_magnitude,
    compute_intra_file_gaps,
    identify_early_close_like_gaps,
    load_research_rows,
    summarize_boundary_labels,
)

TS_FORMAT = "%Y%m%d %H%M%S"
SEP = ";"


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
    """Construye un SnapshotConfig sintetico minimo, igual en espiritu al
    helper equivalente de tests/test_tda00.py -- se duplica aqui a
    proposito (es solo YAML de prueba, no logica de producto) para no
    acoplar los dos modulos de test entre si."""
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
    }
    config_path = config_dir / "mnq_snapshot.yaml"
    config_path.write_text(yaml.safe_dump(config_yaml), encoding="utf-8")
    return load_config(config_path)


# ---------------------------------------------------------------------------
# compute_intra_file_gaps
# ---------------------------------------------------------------------------

def test_no_gap_when_consecutive_minutes():
    rows = pd.DataFrame(
        {
            "source_file": ["a.txt", "a.txt", "a.txt"],
            "timestamp": pd.to_datetime(
                ["2020-01-01 09:00:00", "2020-01-01 09:01:00", "2020-01-01 09:02:00"]
            ),
        }
    )
    gaps = compute_intra_file_gaps(rows)
    assert len(gaps) == 0


def test_single_gap_detected_with_correct_magnitude():
    rows = pd.DataFrame(
        {
            "source_file": ["a.txt", "a.txt"],
            "timestamp": pd.to_datetime(["2020-01-01 17:00:00", "2020-01-01 18:01:00"]),
        }
    )
    gaps = compute_intra_file_gaps(rows)
    assert len(gaps) == 1
    assert gaps.iloc[0]["gap_minutes"] == 61.0


def test_gaps_are_not_computed_across_different_files():
    """El salto entre el último timestamp de un archivo y el primero del
    siguiente (una transición de contrato, no un hueco) no debe aparecer."""
    rows = pd.DataFrame(
        {
            "source_file": ["a.txt", "a.txt", "b.txt", "b.txt"],
            "timestamp": pd.to_datetime(
                [
                    "2020-01-01 09:00:00",
                    "2020-01-01 09:01:00",
                    "2025-06-23 03:01:00",  # muy lejos en el tiempo, otro archivo
                    "2025-06-23 03:02:00",
                ]
            ),
        }
    )
    gaps = compute_intra_file_gaps(rows)
    assert len(gaps) == 0


def test_multiple_gaps_within_one_file():
    rows = pd.DataFrame(
        {
            "source_file": ["a.txt"] * 4,
            "timestamp": pd.to_datetime(
                [
                    "2020-01-01 09:00:00",
                    "2020-01-01 09:01:00",
                    "2020-01-01 09:05:00",  # hueco de 4 min
                    "2020-01-01 10:06:00",  # hueco de 61 min
                ]
            ),
        }
    )
    gaps = compute_intra_file_gaps(rows)
    assert len(gaps) == 2
    assert sorted(gaps["gap_minutes"].tolist()) == [4.0, 61.0]


# ---------------------------------------------------------------------------
# attach_ny_wallclock -- lo mas importante: correccion DST-aware
# ---------------------------------------------------------------------------

def test_ny_wallclock_in_standard_time_est():
    """Enero (EST, UTC-5): 22:00 UTC debe ser 17:00 hora de Nueva York."""
    gaps = pd.DataFrame(
        {
            "source_file": ["a.txt"],
            "before_ts_utc": pd.to_datetime(["2020-01-10 22:00:00"]),
            "after_ts_utc": pd.to_datetime(["2020-01-10 23:01:00"]),
            "gap_minutes": [61.0],
        }
    )
    result = attach_ny_wallclock(gaps)
    assert result.iloc[0]["before_ny_time"] == "17:00:00"
    assert result.iloc[0]["after_ny_time"] == "18:01:00"


def test_ny_wallclock_in_daylight_time_edt():
    """Julio (EDT, UTC-4): 21:00 UTC debe ser 17:00 hora de Nueva York.

    El mismo umbral de mercado (17:00 ET) cae en una hora UTC distinta
    (21:00 en vez de 22:00) segun la epoca del año -- exactamente el
    comportamiento que exige usar zoneinfo/IANA en vez de un offset fijo.
    """
    gaps = pd.DataFrame(
        {
            "source_file": ["a.txt"],
            "before_ts_utc": pd.to_datetime(["2020-07-10 21:00:00"]),
            "after_ts_utc": pd.to_datetime(["2020-07-10 22:01:00"]),
            "gap_minutes": [61.0],
        }
    )
    result = attach_ny_wallclock(gaps)
    assert result.iloc[0]["before_ny_time"] == "17:00:00"
    assert result.iloc[0]["after_ny_time"] == "18:01:00"


def test_ny_wallclock_across_dst_spring_transition():
    """El mismo umbral de mercado debe seguir leyendose 17:00 ET a ambos
    lados de la transicion de marzo 2020 (2020-03-08), aunque el offset
    UTC correspondiente cambie de -5 a -4."""
    gaps = pd.DataFrame(
        {
            "source_file": ["a.txt", "a.txt"],
            "before_ts_utc": pd.to_datetime(["2020-03-06 22:00:00", "2020-03-09 21:00:00"]),
            "after_ts_utc": pd.to_datetime(["2020-03-06 23:01:00", "2020-03-09 22:01:00"]),
            "gap_minutes": [61.0, 61.0],
        }
    )
    result = attach_ny_wallclock(gaps)
    assert (result["before_ny_time"] == "17:00:00").all()
    assert (result["after_ny_time"] == "18:01:00").all()


# ---------------------------------------------------------------------------
# classify_gap_magnitude
# ---------------------------------------------------------------------------

def test_classify_gap_magnitude_boundaries():
    assert classify_gap_magnitude(DAILY_MAINTENANCE_MIN_MINUTES) == "daily_maintenance_like"
    assert classify_gap_magnitude(DAILY_MAINTENANCE_MAX_MINUTES) == "daily_maintenance_like"
    assert classify_gap_magnitude(61.0) == "daily_maintenance_like"
    assert classify_gap_magnitude(DAILY_MAINTENANCE_MIN_MINUTES - 1) == "other"
    assert classify_gap_magnitude(DAILY_MAINTENANCE_MAX_MINUTES + 1) == "other"
    assert classify_gap_magnitude(LONG_GAP_MIN_MINUTES + 1) == "weekly_or_long"
    assert classify_gap_magnitude(LONG_GAP_MIN_MINUTES) == "other"


# ---------------------------------------------------------------------------
# summarize_boundary_labels
# ---------------------------------------------------------------------------

def test_summarize_boundary_labels_counts_dominant_label():
    gaps = pd.DataFrame(
        {
            "before_ny_time": ["17:00:00", "17:00:00", "17:01:00"],
            "after_ny_time": ["18:01:00", "18:01:00", "18:01:00"],
            "gap_minutes": [61.0, 61.0, 60.0],
        }
    )
    summary = summarize_boundary_labels(gaps)
    assert summary.n_gaps == 3
    assert summary.before_ny_time_counts["17:00:00"] == 2
    assert summary.after_ny_time_counts["18:01:00"] == 3


# ---------------------------------------------------------------------------
# load_research_rows -- respeta el hold-out (integracion ligera con config)
# ---------------------------------------------------------------------------

def test_load_research_rows_only_reads_research_files(tmp_path):
    raw_dir = tmp_path / "data" / "raw" / "mnq"
    raw_dir.mkdir(parents=True)
    _write(raw_dir, "00_research.Last.txt", [
        "20200101 090000;100.00;100.25;99.75;100.00;10",
        "20200101 090100;100.00;100.50;99.75;100.25;20",
    ])
    # Deliberadamente NO se crea "99_holdout.Last.txt" en disco: si
    # load_research_rows intentase abrirlo, la prueba fallaria con
    # FileNotFoundError en vez de silenciosamente ignorarlo.
    config = _make_config(
        tmp_path,
        research_files=["00_research.Last.txt"],
        holdout_files=["99_holdout.Last.txt"],
    )

    rows = load_research_rows(config)
    assert len(rows) == 2
    assert set(rows["source_file"]) == {"00_research.Last.txt"}


# ---------------------------------------------------------------------------
# build_forensic_evidence -- proteccion del hold-out (compartida con TDA-00,
# via holdout_guard.py)
# ---------------------------------------------------------------------------

def test_build_forensic_evidence_raises_when_research_and_holdout_overlap(tmp_path):
    raw_dir = tmp_path / "data" / "raw" / "mnq"
    raw_dir.mkdir(parents=True)
    _write(raw_dir, "00_mnq_03_20.Last.txt", [
        "20200101 090000;100.00;100.25;99.75;100.00;10",
    ])
    config = _make_config(
        tmp_path,
        research_files=["00_mnq_03_20.Last.txt"],
        holdout_files=["00_mnq_03_20.Last.txt", "no_existe_en_disco.Last.txt"],
    )

    with pytest.raises(HoldoutIsolationError, match="disjuntos"):
        build_forensic_evidence(config)


def test_build_forensic_evidence_raises_on_duplicate_research_file(tmp_path):
    raw_dir = tmp_path / "data" / "raw" / "mnq"
    raw_dir.mkdir(parents=True)
    _write(raw_dir, "00_mnq_03_20.Last.txt", [
        "20200101 090000;100.00;100.25;99.75;100.00;10",
    ])
    config = _make_config(
        tmp_path,
        research_files=["00_mnq_03_20.Last.txt", "00_mnq_03_20.Last.txt"],
    )

    with pytest.raises(HoldoutIsolationError, match="duplicado"):
        build_forensic_evidence(config)


def test_build_forensic_evidence_raises_when_research_row_reaches_boundary(tmp_path):
    raw_dir = tmp_path / "data" / "raw" / "mnq"
    raw_dir.mkdir(parents=True)
    _write(raw_dir, "00_mnq_03_20.Last.txt", [
        "20200101 090000;100.00;100.25;99.75;100.00;10",
        "20200101 090100;100.00;100.50;99.75;100.25;20",  # timestamp = frontera exacta
    ])
    config = _make_config(
        tmp_path,
        research_files=["00_mnq_03_20.Last.txt"],
        holdout_files=["no_existe_en_disco.Last.txt"],
        boundary_utc="2020-01-01 09:01:00",
    )

    with pytest.raises(HoldoutIsolationError, match="frontera de hold-out"):
        build_forensic_evidence(config)

    # La validacion de frontera no debe haber intentado abrir el archivo
    # de holdout_files (no existe en disco; si el pipeline lo hubiera
    # intentado abrir, habria fallado con FileNotFoundError en vez de
    # HoldoutIsolationError).
    assert not (raw_dir / "no_existe_en_disco.Last.txt").exists()


def test_build_forensic_evidence_does_not_open_holdout_files(tmp_path):
    """Un hold-out inexistente en disco no debe impedir una corrida normal."""
    raw_dir = tmp_path / "data" / "raw" / "mnq"
    raw_dir.mkdir(parents=True)
    _write(raw_dir, "00_mnq_03_20.Last.txt", [
        "20200101 090000;100.00;100.25;99.75;100.00;10",
        "20200101 090100;100.00;100.50;99.75;100.25;20",
    ])
    config = _make_config(
        tmp_path,
        research_files=["00_mnq_03_20.Last.txt"],
        holdout_files=["archivo_que_no_existe_en_disco.Last.txt"],
        boundary_utc="2099-01-01 00:00:00",
    )

    gaps = build_forensic_evidence(config)
    assert isinstance(gaps, pd.DataFrame)


# ---------------------------------------------------------------------------
# identify_early_close_like_gaps -- tercera ancla forense, regla reproducible
# ---------------------------------------------------------------------------

def _gap_row(magnitude_class, before_ny_time, after_ny_time, gap_minutes=301.0, source_file="a.txt"):
    return {
        "source_file": source_file,
        "magnitude_class": magnitude_class,
        "before_ny_time": before_ny_time,
        "after_ny_time": after_ny_time,
        "gap_minutes": gap_minutes,
    }


def test_identify_early_close_like_gaps_matches_exact_signature():
    gaps = pd.DataFrame(
        [
            _gap_row("other", EARLY_CLOSE_BEFORE_NY_TIME, EARLY_CLOSE_AFTER_NY_TIME),
        ]
    )
    result = identify_early_close_like_gaps(gaps)
    assert len(result) == 1


def test_identify_early_close_like_gaps_excludes_near_misses():
    """Un minuto de diferencia en cualquiera de los dos bordes NO cuenta:
    la regla exige coincidencia exacta, no una tolerancia."""
    gaps = pd.DataFrame(
        [
            _gap_row("other", "13:01:00", EARLY_CLOSE_AFTER_NY_TIME),  # borde anterior distinto
            _gap_row("other", EARLY_CLOSE_BEFORE_NY_TIME, "18:02:00"),  # borde posterior distinto
            _gap_row("other", "12:59:00", EARLY_CLOSE_AFTER_NY_TIME),  # tampoco
        ]
    )
    result = identify_early_close_like_gaps(gaps)
    assert len(result) == 0


def test_identify_early_close_like_gaps_excludes_other_magnitude_classes():
    """Un hueco que calza con la firma horaria pero es del tamano del
    corte diario o del cierre semanal no debe contarse aqui -- ya
    pertenece a otra ancla."""
    gaps = pd.DataFrame(
        [
            _gap_row("daily_maintenance_like", EARLY_CLOSE_BEFORE_NY_TIME, EARLY_CLOSE_AFTER_NY_TIME, gap_minutes=61.0),
            _gap_row("weekly_or_long", EARLY_CLOSE_BEFORE_NY_TIME, EARLY_CLOSE_AFTER_NY_TIME, gap_minutes=3000.0),
        ]
    )
    result = identify_early_close_like_gaps(gaps)
    assert len(result) == 0


def test_identify_early_close_like_gaps_returns_matching_rows_unmodified():
    gaps = pd.DataFrame(
        [
            _gap_row("other", EARLY_CLOSE_BEFORE_NY_TIME, EARLY_CLOSE_AFTER_NY_TIME, source_file="x.txt"),
            _gap_row("other", "05:00:00", "06:00:00", source_file="y.txt"),  # no matchea
        ]
    )
    result = identify_early_close_like_gaps(gaps)
    assert len(result) == 1
    assert result.iloc[0]["source_file"] == "x.txt"
