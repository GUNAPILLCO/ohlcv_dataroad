"""Tests de TDA-00 -- integridad intra-barra e invariantes fisicos.

Cada test aisla UNA regla concreta usando un archivo sintetico minimo
(unas pocas lineas escritas a mano en ``tmp_path``), en vez de depender de
los datos reales de ``data/raw/``. Esto hace que el test:

- sea rapido (milisegundos, no segundos sobre 2M de filas reales);
- sea determinista (la regla se dispara SIEMPRE, no depende de que el
  snapshot real siga estando limpio);
- documente, por si mismo, que forma de dato dispara cada regla -- son,
  en la practica, ejemplos ejecutables de cada invariante.

Formato de las lineas sinteticas: igual que el dato real,
``timestamp;open;high;low;close;volume`` con ``;`` como separador y el
timestamp en formato ``YYYYMMDD HHMMSS``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from ohlcv_dataroad.config import load_config
from ohlcv_dataroad.ingest.parsing import parse_raw_file
from ohlcv_dataroad.ingest.tda00_integrity import (
    ALL_HARD_RULES,
    HARD_INVARIANT_RULES,
    HoldoutIsolationError,
    _hard_invariant_checks,
    process_file,
    run_tda00,
)

TS_FORMAT = "%Y%m%d %H%M%S"
SEP = ";"
TICK = 0.25


def _write(tmp_path: Path, name: str, lines: list[str]) -> Path:
    """Escribe ``lines`` como un archivo crudo en ``tmp_path`` y devuelve su ruta."""
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _violated_rules(path: Path, row_index: int) -> set[str]:
    """Ejecuta process_file y devuelve el conjunto de reglas violadas por una fila."""
    result = process_file(path, path.name, TS_FORMAT, SEP, TICK)
    hit = result.mask.loc[result.mask["row_index"] == row_index]
    assert len(hit) == 1, f"se esperaba exactamente una fila con row_index={row_index}"
    rules = hit.iloc[0]["violation_rules"]
    return set(rules.split(";")) if rules else set()


# ---------------------------------------------------------------------------
# Caso base: archivo limpio -> cero violaciones, cero errores de parseo.
# ---------------------------------------------------------------------------

def test_clean_file_has_no_violations(tmp_path):
    lines = [
        "20200101 090000;100.00;100.25;99.75;100.00;10",
        "20200101 090100;100.00;100.50;99.75;100.25;20",
        "20200101 090200;100.25;100.50;100.00;100.25;5",
    ]
    path = _write(tmp_path, "clean.Last.txt", lines)
    result = process_file(path, path.name, TS_FORMAT, SEP, TICK)

    assert result.summary["total_lines"] == 3
    assert result.summary["parsed_rows"] == 3
    assert result.summary["parse_error_rows"] == 0
    assert result.summary["hard_violation_rows"] == 0
    assert result.violations == []
    assert not result.mask["bad_data"].any()


# ---------------------------------------------------------------------------
# Errores de parseo / esquema
# ---------------------------------------------------------------------------

def test_wrong_field_count_is_schema_error(tmp_path):
    lines = [
        "20200101 090000;100.00;100.25;99.75;100.00;10",
        "20200101 090100;100.00;100.50;99.75;100.25",  # falta el campo de volumen
    ]
    path = _write(tmp_path, "badschema.Last.txt", lines)
    result = process_file(path, path.name, TS_FORMAT, SEP, TICK)

    assert result.summary["parse_error_rows"] == 1
    assert result.summary["parsed_rows"] == 1
    rules = {v["rule"] for v in result.violations}
    assert "schema_field_count" in rules


def test_non_numeric_field_is_parse_error(tmp_path):
    lines = [
        "20200101 090000;100.00;100.25;99.75;100.00;10",
        "20200101 090100;abc;100.50;99.75;100.25;20",
    ]
    path = _write(tmp_path, "badnumeric.Last.txt", lines)
    result = process_file(path, path.name, TS_FORMAT, SEP, TICK)

    assert result.summary["parse_error_rows"] == 1
    rules = {v["rule"] for v in result.violations}
    assert "parse_error_numeric" in rules


def test_bad_timestamp_format_is_parse_error(tmp_path):
    lines = [
        "2020-01-01 09:00:00;100.00;100.25;99.75;100.00;10",  # formato incorrecto
    ]
    path = _write(tmp_path, "badts.Last.txt", lines)
    result = process_file(path, path.name, TS_FORMAT, SEP, TICK)

    assert result.summary["parse_error_rows"] == 1
    rules = {v["rule"] for v in result.violations}
    assert "parse_error_timestamp" in rules


def test_empty_field_is_null_value(tmp_path):
    lines = [
        "20200101 090000;;100.25;99.75;100.00;10",  # open vacio
    ]
    path = _write(tmp_path, "nullfield.Last.txt", lines)
    result = process_file(path, path.name, TS_FORMAT, SEP, TICK)

    assert result.summary["parse_error_rows"] == 1
    rules = {v["rule"] for v in result.violations}
    assert "null_value" in rules


def test_infinite_value_is_non_finite(tmp_path):
    lines = [
        "20200101 090000;inf;100.25;99.75;100.00;10",
    ]
    path = _write(tmp_path, "inf.Last.txt", lines)
    result = process_file(path, path.name, TS_FORMAT, SEP, TICK)

    assert result.summary["parse_error_rows"] == 1
    rules = {v["rule"] for v in result.violations}
    assert "non_finite_value" in rules


def test_conservation_holds_with_mixed_good_and_bad_lines(tmp_path):
    """total_lines == parsed_rows + parse_error_rows, incluso con lineas mezcladas."""
    lines = [
        "20200101 090000;100.00;100.25;99.75;100.00;10",   # ok
        "20200101 090100;abc;100.50;99.75;100.25;20",       # parse_error_numeric
        "20200101 090200;100.25;100.50;100.00;100.25;5",   # ok
        "20200101 090300;100.25;100.50",                    # schema_field_count
    ]
    path = _write(tmp_path, "mixed.Last.txt", lines)
    result = process_file(path, path.name, TS_FORMAT, SEP, TICK)

    s = result.summary
    assert s["total_lines"] == 4
    assert s["parsed_rows"] == 2
    assert s["parse_error_rows"] == 2
    assert s["parsed_rows"] + s["parse_error_rows"] == s["total_lines"]
    # Toda linea del archivo debe estar representada en la mascara, una vez.
    assert len(result.mask) == 4


def test_trailing_blank_line_counts_as_schema_error(tmp_path):
    """Corrección 1: una línea en blanco real al final del archivo se cuenta
    como línea leída y queda trazada como schema_field_count -- ya no
    desaparece en silencio."""
    path = tmp_path / "trailing.Last.txt"
    path.write_text("20200101 090000;100.00;100.25;99.75;100.00;10\n\n", encoding="utf-8")
    parsed = parse_raw_file(path, TS_FORMAT, SEP)

    assert parsed.total_lines == 2
    assert len(parsed.rows) == 1
    assert len(parsed.parse_errors) == 1
    assert parsed.parse_errors[0].rule == "schema_field_count"
    assert parsed.parse_errors[0].row_index == 2
    assert parsed.parse_errors[0].raw_line == ""
    # Conservacion: toda linea leida termina en fila valida o en error.
    assert parsed.total_lines == len(parsed.rows) + len(parsed.parse_errors)


def test_blank_line_in_the_middle_counts_as_schema_error(tmp_path):
    """Una línea en blanco en medio del archivo (no solo al final) se trata igual."""
    path = tmp_path / "midblank.Last.txt"
    path.write_text(
        "20200101 090000;100.00;100.25;99.75;100.00;10\n"
        "\n"
        "20200101 090100;100.00;100.50;99.75;100.25;20\n",
        encoding="utf-8",
    )
    parsed = parse_raw_file(path, TS_FORMAT, SEP)

    assert parsed.total_lines == 3
    assert len(parsed.rows) == 2
    assert len(parsed.parse_errors) == 1
    assert parsed.parse_errors[0].rule == "schema_field_count"
    assert parsed.parse_errors[0].row_index == 2


def test_file_without_any_blank_line_is_unaffected(tmp_path):
    """Un archivo normal (sin línea en blanco real) no cambia de comportamiento."""
    path = _write(tmp_path, "normal.Last.txt", [
        "20200101 090000;100.00;100.25;99.75;100.00;10",
        "20200101 090100;100.00;100.50;99.75;100.25;20",
    ])
    parsed = parse_raw_file(path, TS_FORMAT, SEP)
    assert parsed.total_lines == 2
    assert len(parsed.rows) == 2
    assert parsed.parse_errors == []


# ---------------------------------------------------------------------------
# Invariantes duros sobre filas bien formadas
# ---------------------------------------------------------------------------

def test_ohlc_incoherent_high_below_low(tmp_path):
    lines = ["20200101 090000;100.00;99.00;99.50;100.00;10"]  # high < low
    path = _write(tmp_path, "ohlc1.Last.txt", lines)
    assert "ohlc_incoherent" in _violated_rules(path, 1)


def test_ohlc_incoherent_high_below_close(tmp_path):
    lines = ["20200101 090000;100.00;100.10;99.50;100.50;10"]  # high < close
    path = _write(tmp_path, "ohlc2.Last.txt", lines)
    assert "ohlc_incoherent" in _violated_rules(path, 1)


def test_ohlc_incoherent_low_above_open(tmp_path):
    lines = ["20200101 090000;100.00;100.50;100.25;100.25;10"]  # low > open
    path = _write(tmp_path, "ohlc3.Last.txt", lines)
    assert "ohlc_incoherent" in _violated_rules(path, 1)


def test_nonpositive_price(tmp_path):
    lines = ["20200101 090000;0.00;100.25;99.75;100.00;10"]  # open == 0
    path = _write(tmp_path, "nonpos.Last.txt", lines)
    assert "nonpositive_price" in _violated_rules(path, 1)


def test_negative_volume(tmp_path):
    lines = ["20200101 090000;100.00;100.25;99.75;100.00;-5"]
    path = _write(tmp_path, "negvol.Last.txt", lines)
    assert "negative_volume" in _violated_rules(path, 1)


def test_tick_grid_violation(tmp_path):
    lines = ["20200101 090000;100.10;100.35;99.85;100.10;10"]  # 100.10 no es multiplo de 0.25
    path = _write(tmp_path, "tick.Last.txt", lines)
    assert "tick_grid_violation" in _violated_rules(path, 1)


def test_tick_aligned_prices_do_not_violate_grid(tmp_path):
    lines = ["20200101 090000;100.00;100.25;99.75;100.50;10"]  # todos multiplos de 0.25
    path = _write(tmp_path, "tickok.Last.txt", lines)
    assert "tick_grid_violation" not in _violated_rules(path, 1)


def test_duplicate_timestamp_within_file(tmp_path):
    lines = [
        "20200101 090000;100.00;100.25;99.75;100.00;10",
        "20200101 090000;100.25;100.50;100.00;100.25;5",  # mismo timestamp, distinto OHLCV
    ]
    path = _write(tmp_path, "duptime.Last.txt", lines)
    assert "duplicate_timestamp_within_file" in _violated_rules(path, 1)
    assert "duplicate_timestamp_within_file" in _violated_rules(path, 2)
    # No son fila identica (OHLCV distinto): no debe marcarse duplicate_exact_row.
    assert "duplicate_exact_row" not in _violated_rules(path, 1)


def test_duplicate_exact_row(tmp_path):
    lines = [
        "20200101 090000;100.00;100.25;99.75;100.00;10",
        "20200101 090000;100.00;100.25;99.75;100.00;10",  # fila identica
    ]
    path = _write(tmp_path, "dupexact.Last.txt", lines)
    r1 = _violated_rules(path, 1)
    r2 = _violated_rules(path, 2)
    assert "duplicate_exact_row" in r1 and "duplicate_exact_row" in r2
    # Una fila exactamente duplicada tambien comparte timestamp por definicion.
    assert "duplicate_timestamp_within_file" in r1 and "duplicate_timestamp_within_file" in r2


def test_timestamp_out_of_order(tmp_path):
    lines = [
        "20200101 090100;100.00;100.25;99.75;100.00;10",
        "20200101 090000;100.25;100.50;100.00;100.25;5",  # retrocede en el tiempo
    ]
    path = _write(tmp_path, "outoforder.Last.txt", lines)
    assert "timestamp_out_of_order" in _violated_rules(path, 2)
    assert "timestamp_out_of_order" not in _violated_rules(path, 1)


def test_zero_volume_is_flagged_but_not_a_violation(tmp_path):
    lines = ["20200101 090000;100.00;100.25;99.75;100.00;0"]
    path = _write(tmp_path, "zerovol.Last.txt", lines)
    result = process_file(path, path.name, TS_FORMAT, SEP, TICK)

    assert result.summary["zero_volume_rows"] == 1
    assert result.summary["hard_violation_rows"] == 0  # NO es una violacion
    row = result.mask.iloc[0]
    assert bool(row["zero_volume"]) is True
    assert bool(row["bad_data"]) is False


def test_row_can_violate_multiple_rules_simultaneously(tmp_path):
    # high < low (ohlc_incoherent) y ademas precio no positivo.
    lines = ["20200101 090000;-1.00;-0.50;0.00;-0.75;10"]
    path = _write(tmp_path, "multi.Last.txt", lines)
    rules = _violated_rules(path, 1)
    assert "nonpositive_price" in rules
    assert "ohlc_incoherent" in rules


def test_parse_error_preserves_raw_line_end_to_end(tmp_path):
    """Corrección 3: el texto crudo de una línea que falla el parseo debe
    poder reconstruirse desde `violations`, sin volver al archivo original."""
    lines = [
        "20200101 090000;100.00;100.25;99.75;100.00;10",
        "20200101 090100;abc;100.50;99.75;100.25;20",  # parse_error_numeric
    ]
    path = _write(tmp_path, "rawline.Last.txt", lines)
    result = process_file(path, path.name, TS_FORMAT, SEP, TICK)

    parse_violations = [v for v in result.violations if v["rule"] == "parse_error_numeric"]
    assert len(parse_violations) == 1
    assert parse_violations[0]["raw_line"] == "20200101 090100;abc;100.50;99.75;100.25;20"

    # Las violaciones sobre filas YA bien formadas no necesitan raw_line
    # (son perfectamente reconstruibles desde sus columnas numericas).
    lines2 = ["20200101 090000;0.00;100.25;99.75;100.00;10"]  # nonpositive_price
    path2 = _write(tmp_path, "rawline2.Last.txt", lines2)
    result2 = process_file(path2, path2.name, TS_FORMAT, SEP, TICK)
    assert result2.violations[0]["rule"] == "nonpositive_price"
    assert result2.violations[0]["raw_line"] is None


def test_all_hard_invariant_rules_are_in_all_hard_rules():
    """Consistencia interna del catalogo de reglas usado por _hard_invariant_checks."""
    for rule in HARD_INVARIANT_RULES:
        assert rule in ALL_HARD_RULES


# ---------------------------------------------------------------------------
# _hard_invariant_checks directamente (sin pasar por parse_raw_file)
# ---------------------------------------------------------------------------

def test_hard_invariant_checks_on_synthetic_dataframe():
    df = pd.DataFrame(
        {
            "row_index": [1, 2],
            "timestamp": pd.to_datetime(["2020-01-01 09:00:00", "2020-01-01 09:01:00"]),
            "timestamp_raw": ["20200101 090000", "20200101 090100"],
            "open": [100.0, 100.0],
            "high": [100.5, 100.5],
            "low": [99.5, 99.5],
            "close": [100.0, 100.0],
            "volume": [10, 10],
        }
    )
    checks = _hard_invariant_checks(df, tick_size=0.25)
    assert not checks[HARD_INVARIANT_RULES].to_numpy().any()


# ---------------------------------------------------------------------------
# Integracion: run_tda00 de punta a punta sobre una configuracion sintetica
# ---------------------------------------------------------------------------

def _make_config_yaml(
    tmp_path: Path,
    research_files: list[str],
    holdout_files: list[str] | None = None,
    boundary_utc: str = "2099-01-01 00:00:00",
) -> Path:
    config_dir = tmp_path / "configs"
    config_dir.mkdir(exist_ok=True)
    config = {
        "instrument": "MNQ",
        "raw_data": {
            "dir": "data/raw/mnq",
            "file_pattern": "*.Last.txt",
            "separator": ";",
            "columns": ["timestamp", "open", "high", "low", "close", "volume"],
            "timestamp_format": TS_FORMAT,
            "timestamp_raw_timezone": "UTC",
        },
        "instrument_spec": {"tick_size": 0.25, "tick_size_source": "test fixture"},
        "holdout": {
            "boundary_utc": boundary_utc,
            "boundary_source": "test fixture",
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
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_run_tda00_end_to_end_clean_data_gives_pass(tmp_path):
    raw_dir = tmp_path / "data" / "raw" / "mnq"
    raw_dir.mkdir(parents=True)
    _write(raw_dir, "00_mnq_03_20.Last.txt", [
        "20200101 090000;100.00;100.25;99.75;100.00;10",
        "20200101 090100;100.00;100.50;99.75;100.25;20",
    ])
    config_path = _make_config_yaml(tmp_path, ["00_mnq_03_20.Last.txt"])
    config = load_config(config_path)

    stats = run_tda00(config)

    assert stats["status"] == "PASS"
    assert stats["hard_violation_rows"] == 0
    assert stats["total_lines"] == 2
    assert config.bad_data_mask_path.exists()
    assert config.violations_report_path.exists()
    assert config.inventory_report_path.exists()
    assert config.per_file_summary_path.exists()

    mask = pd.read_parquet(config.bad_data_mask_path)
    assert len(mask) == 2
    assert not mask["bad_data"].any()

    violations = pd.read_csv(config.violations_report_path)
    assert len(violations) == 0  # header presente, cero filas


def test_run_tda00_end_to_end_dirty_data_gives_pass_with_warnings(tmp_path):
    raw_dir = tmp_path / "data" / "raw" / "mnq"
    raw_dir.mkdir(parents=True)
    _write(raw_dir, "00_mnq_03_20.Last.txt", [
        "20200101 090000;100.00;100.25;99.75;100.00;10",
        "20200101 090100;100.00;99.00;99.75;100.25;20",  # high < low
        "20200101 090200;100.00;100.25;99.75;100.00;0",   # volumen 0 (observacional)
    ])
    config_path = _make_config_yaml(tmp_path, ["00_mnq_03_20.Last.txt"])
    config = load_config(config_path)

    stats = run_tda00(config)

    assert stats["status"] == "PASS_WITH_WARNINGS"
    assert stats["hard_violation_rows"] == 1
    assert stats["zero_volume_rows"] == 1

    violations = pd.read_csv(config.violations_report_path)
    assert len(violations) == 1
    assert violations.iloc[0]["rule"] == "ohlc_incoherent"


def test_run_tda00_raises_on_missing_research_file(tmp_path):
    raw_dir = tmp_path / "data" / "raw" / "mnq"
    raw_dir.mkdir(parents=True)
    config_path = _make_config_yaml(tmp_path, ["no_existe.Last.txt"])
    config = load_config(config_path)

    with pytest.raises(FileNotFoundError):
        run_tda00(config)


# ---------------------------------------------------------------------------
# Correccion 2: proteccion explicita del hold-out
# ---------------------------------------------------------------------------

def test_run_tda00_raises_when_research_and_holdout_overlap(tmp_path):
    """research_files y holdout_files comparten un archivo -> FAIL, sin procesar nada."""
    raw_dir = tmp_path / "data" / "raw" / "mnq"
    raw_dir.mkdir(parents=True)
    _write(raw_dir, "00_mnq_03_20.Last.txt", [
        "20200101 090000;100.00;100.25;99.75;100.00;10",
    ])
    # "00_mnq_03_20.Last.txt" aparece en ambas listas: violacion de disjuncion.
    config_path = _make_config_yaml(
        tmp_path,
        research_files=["00_mnq_03_20.Last.txt"],
        holdout_files=["00_mnq_03_20.Last.txt", "no_existe_en_disco.Last.txt"],
    )
    config = load_config(config_path)

    with pytest.raises(HoldoutIsolationError, match="disjuntos"):
        run_tda00(config)

    # No debe haberse escrito ningun artefacto: el pipeline aborta ANTES de procesar.
    assert not config.inventory_report_path.exists()
    assert not config.bad_data_mask_path.exists()


def test_run_tda00_raises_on_duplicate_research_file(tmp_path):
    raw_dir = tmp_path / "data" / "raw" / "mnq"
    raw_dir.mkdir(parents=True)
    _write(raw_dir, "00_mnq_03_20.Last.txt", [
        "20200101 090000;100.00;100.25;99.75;100.00;10",
    ])
    config_path = _make_config_yaml(
        tmp_path,
        research_files=["00_mnq_03_20.Last.txt", "00_mnq_03_20.Last.txt"],
    )
    config = load_config(config_path)

    with pytest.raises(HoldoutIsolationError, match="duplicado"):
        run_tda00(config)


def test_run_tda00_raises_when_research_row_reaches_holdout_boundary(tmp_path):
    """Una fila de investigacion con timestamp >= holdout_boundary_utc -> FAIL."""
    raw_dir = tmp_path / "data" / "raw" / "mnq"
    raw_dir.mkdir(parents=True)
    _write(raw_dir, "00_mnq_03_20.Last.txt", [
        "20200101 090000;100.00;100.25;99.75;100.00;10",
        "20200101 090100;100.00;100.50;99.75;100.25;20",  # timestamp = frontera exacta
    ])
    config_path = _make_config_yaml(
        tmp_path,
        research_files=["00_mnq_03_20.Last.txt"],
        holdout_files=["no_existe_en_disco.Last.txt"],  # nunca se abre: ver aserciones abajo
        boundary_utc="2020-01-01 09:01:00",  # igual al timestamp de la segunda fila
    )
    config = load_config(config_path)

    with pytest.raises(HoldoutIsolationError, match="frontera de hold-out"):
        run_tda00(config)

    # La validacion de frontera no abrio el archivo de holdout_files (no existe en disco;
    # si el pipeline lo hubiera intentado abrir, habria fallado con FileNotFoundError,
    # no con HoldoutIsolationError).
    assert not (raw_dir / "no_existe_en_disco.Last.txt").exists()


def test_run_tda00_does_not_open_holdout_files(tmp_path):
    """Confirma que un hold-out inexistente en disco no impide una corrida limpia."""
    raw_dir = tmp_path / "data" / "raw" / "mnq"
    raw_dir.mkdir(parents=True)
    _write(raw_dir, "00_mnq_03_20.Last.txt", [
        "20200101 090000;100.00;100.25;99.75;100.00;10",
    ])
    config_path = _make_config_yaml(
        tmp_path,
        research_files=["00_mnq_03_20.Last.txt"],
        holdout_files=["archivo_que_no_existe_en_disco.Last.txt"],
        boundary_utc="2099-01-01 00:00:00",
    )
    config = load_config(config_path)

    # Si run_tda00 intentase abrir algun archivo de holdout_files, esto lanzaria
    # FileNotFoundError (el archivo no existe en disco). Como no lo abre, la
    # corrida debe completarse en PASS con normalidad.
    stats = run_tda00(config)
    assert stats["status"] == "PASS"


def test_run_tda00_violations_csv_has_raw_line_column_populated_for_parse_errors(tmp_path):
    """Correccion 3, de punta a punta: raw_line debe aparecer en el CSV final."""
    raw_dir = tmp_path / "data" / "raw" / "mnq"
    raw_dir.mkdir(parents=True)
    _write(raw_dir, "00_mnq_03_20.Last.txt", [
        "20200101 090000;100.00;100.25;99.75;100.00;10",
        "20200101 090100;abc;100.50;99.75;100.25;20",  # parse_error_numeric
    ])
    config_path = _make_config_yaml(tmp_path, ["00_mnq_03_20.Last.txt"])
    config = load_config(config_path)

    stats = run_tda00(config)
    assert stats["status"] == "PASS_WITH_WARNINGS"

    violations = pd.read_csv(config.violations_report_path)
    assert "raw_line" in violations.columns
    row = violations.loc[violations["rule"] == "parse_error_numeric"].iloc[0]
    assert row["raw_line"] == "20200101 090100;abc;100.50;99.75;100.25;20"


def test_run_tda00_still_passes_on_clean_real_shaped_data(tmp_path):
    """Ejecucion limpia actual (sin lineas vacias, sin solapamiento) sigue dando PASS."""
    raw_dir = tmp_path / "data" / "raw" / "mnq"
    raw_dir.mkdir(parents=True)
    _write(raw_dir, "00_mnq_03_20.Last.txt", [
        "20200101 090000;100.00;100.25;99.75;100.00;10",
        "20200101 090100;100.00;100.50;99.75;100.25;20",
        "20200101 090200;100.25;100.50;100.00;100.25;5",
    ])
    config_path = _make_config_yaml(
        tmp_path,
        research_files=["00_mnq_03_20.Last.txt"],
        holdout_files=["22_mnq_09_25.Last.txt"],
        boundary_utc="2025-06-23 00:00:00",
    )
    config = load_config(config_path)

    stats = run_tda00(config)
    assert stats["status"] == "PASS"
    assert stats["hard_violation_rows"] == 0
    assert stats["parse_error_rows"] == 0
