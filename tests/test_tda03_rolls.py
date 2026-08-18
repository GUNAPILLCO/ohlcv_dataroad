"""Tests de TDA-03 -- rolls y construccion de la serie continua.

Igual que TDA-02: datos sinteticos minimos para aislar cada propiedad
(clasificacion de transiciones, causalidad de la señal de roll,
irreversibilidad, construccion de la serie canonica, mascara de roll,
basis, ajustes, tabla de invariancia), mas la proteccion del hold-out.
No se reproducen aqui los numeros exactos del conjunto de investigacion
real (eso vive en ``reports/mnq/TDA03_rolls_serie_continua.md``, generado
por ``run_tda03.py``).
"""
from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd
import pytest
import yaml

from ohlcv_dataroad.config import load_config
from ohlcv_dataroad.ingest.holdout_guard import HoldoutIsolationError
from ohlcv_dataroad.ingest.session_calendar import build_session_schedule
from ohlcv_dataroad.ingest.tda03_rolls import (
    REASON_POST_CROSSOVER,
    REASON_PRE_CROSSOVER,
    REASON_ZERO_BAR_FALLBACK,
    TRANSITION_NO_OVERLAP,
    TRANSITION_OVERLAP,
    RollPolicy,
    apply_adjustments,
    attach_trading_date_and_contract,
    build_active_contract_calendar,
    build_canonical_series,
    build_invariance_table,
    build_roll_mask,
    classify_transition,
    compute_adjustment_factors,
    compute_basis_evolution,
    compute_no_overlap_evidence,
    compute_overlap_daily_evidence,
    determine_overlap_rollover,
    find_extreme_jumps,
    parse_contract_label,
    run_tda03_analysis,
)

TS_FORMAT = "%Y%m%d %H%M%S"


def _write(tmp_path: Path, name: str, lines: list[str]) -> Path:
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _make_config(tmp_path: Path, research_files, holdout_files=None, boundary_utc="2099-01-01 00:00:00"):
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
        "tda03": {
            "min_incoming_share_shared": 0.50,
            "confirmation_sessions_required": 1,
            "extreme_jump_top_n": 10,
            "output_dir_reports": "reports/mnq",
            "output_dir_interim": "data/interim/mnq",
            "report_name": "TDA03_rolls_serie_continua.md",
            "transitions_csv": "TDA03_transiciones.csv",
            "overlap_daily_evidence_csv": "TDA03_evidencia_diaria_solapamientos.csv",
            "no_overlap_evidence_csv": "TDA03_evidencia_sin_solapamiento.csv",
            "invariance_table_csv": "TDA03_tabla_invariancia.csv",
            "discarded_rows_csv": "TDA03_filas_descartadas.csv",
            "extreme_jumps_csv": "TDA03_saltos_extremos_stop3.csv",
            "adjustment_factors_csv": "TDA03_factores_ajuste.csv",
            "roll_mask_name": "tda03_roll_mask.parquet",
            "canonical_series_name": "tda03_serie_continua.parquet",
        },
    }
    config_path = config_dir / "mnq_snapshot.yaml"
    config_path.write_text(yaml.safe_dump(config_yaml), encoding="utf-8")
    return load_config(config_path)


def _policy(**overrides):
    base = dict(min_incoming_share_shared=0.50, confirmation_sessions_required=1, extreme_jump_top_n=10)
    base.update(overrides)
    return RollPolicy(**base)


# ---------------------------------------------------------------------------
# parse_contract_label
# ---------------------------------------------------------------------------

def test_parse_contract_label():
    assert parse_contract_label("19_mnq_12_24.Last.txt") == "Z24"
    assert parse_contract_label("00_mnq_03_20.Last.txt") == "H20"
    assert parse_contract_label("21_mnq_06_25.Last.txt") == "M25"
    assert parse_contract_label("02_mnq_09_20.Last.txt") == "U20"


# ---------------------------------------------------------------------------
# Fixtures sinteticas: dos "contratos" con solapamiento controlado
# ---------------------------------------------------------------------------

def _synthetic_overlap_rows():
    """3 dias de solapamiento: dia1 saliente domina, dia2 empate, dia3 entrante domina."""
    schedule = build_session_schedule("2024-01-08", "2024-01-10")
    days = [datetime.date(2024, 1, 8), datetime.date(2024, 1, 9), datetime.date(2024, 1, 10)]
    out_shares = [0.10, 0.30, 0.70]  # share del ENTRANTE cada dia (creciente)
    records = []
    for day, share_in in zip(days, out_shares):
        open_ts = schedule.table.loc[day, "market_open"]
        close_ts = schedule.table.loc[day, "market_close"]
        minutes = pd.date_range(open_ts + pd.Timedelta(minutes=1), close_ts, freq="1min")[:5]
        for m in minutes:
            vol_in = round(100 * share_in)
            vol_out = 100 - vol_in
            records.append(
                {"source_file": "00_out_03_24.Last.txt", "timestamp": m.tz_localize(None),
                 "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": vol_out}
            )
            records.append(
                {"source_file": "01_in_06_24.Last.txt", "timestamp": m.tz_localize(None),
                 "open": 101.0, "high": 101.5, "low": 100.5, "close": 101.0, "volume": vol_in}
            )
    return pd.DataFrame(records), schedule


def _synthetic_overlap_rows_with_resolved_crossover():
    """4 dias de solapamiento con shares crecientes [0.10,0.30,0.70,0.90]:
    la señal se confirma en el dia 3 y el dia 4 (tambien >= umbral) permite
    que ``effective_date`` se resuelva DENTRO de la ventana (a diferencia
    de ``_synthetic_overlap_rows``, cuya señal cae en el ultimo dia
    disponible y deja ``effective_date=None``)."""
    schedule = build_session_schedule("2024-01-08", "2024-01-11")
    days = [datetime.date(2024, 1, 8), datetime.date(2024, 1, 9), datetime.date(2024, 1, 10), datetime.date(2024, 1, 11)]
    shares_in = [0.10, 0.30, 0.70, 0.90]
    records = []
    for day, share_in in zip(days, shares_in):
        open_ts = schedule.table.loc[day, "market_open"]
        close_ts = schedule.table.loc[day, "market_close"]
        for m in pd.date_range(open_ts + pd.Timedelta(minutes=1), close_ts, freq="1min")[:5]:
            vol_in = round(100 * share_in)
            records.append({"source_file": "00_out_03_24.Last.txt", "timestamp": m.tz_localize(None),
                             "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 100 - vol_in})
            records.append({"source_file": "01_in_06_24.Last.txt", "timestamp": m.tz_localize(None),
                             "open": 101.0, "high": 101.5, "low": 100.5, "close": 101.0, "volume": vol_in})
    return pd.DataFrame(records), schedule


def test_overlap_daily_evidence_share_shared_matches_volume_composition():
    rows, schedule = _synthetic_overlap_rows()
    rows = attach_trading_date_and_contract(rows, schedule)
    out_rows = rows[rows["source_file"] == "00_out_03_24.Last.txt"]
    in_rows = rows[rows["source_file"] == "01_in_06_24.Last.txt"]
    dates = sorted(set(out_rows["trading_date"]) & set(in_rows["trading_date"]))
    full_grid_counts = {d: 1000 for d in dates}  # valor arbitrario, no usado en esta prueba
    evidence = compute_overlap_daily_evidence(out_rows, in_rows, dates, full_grid_counts)
    shares = evidence.set_index("trading_date")["share_shared"]
    assert shares[dates[0]] == pytest.approx(0.10, abs=1e-6)
    assert shares[dates[1]] == pytest.approx(0.30, abs=1e-6)
    assert shares[dates[2]] == pytest.approx(0.70, abs=1e-6)


def test_determine_overlap_rollover_crosses_on_dominance_day():
    rows, schedule = _synthetic_overlap_rows()
    rows = attach_trading_date_and_contract(rows, schedule)
    out_rows = rows[rows["source_file"] == "00_out_03_24.Last.txt"]
    in_rows = rows[rows["source_file"] == "01_in_06_24.Last.txt"]
    dates = sorted(set(out_rows["trading_date"]) & set(in_rows["trading_date"]))
    evidence = compute_overlap_daily_evidence(out_rows, in_rows, dates, {d: 1000 for d in dates})
    decision = determine_overlap_rollover(evidence, dates, _policy())
    assert decision["signal_date"] == dates[2]  # el 3er dia (share=0.70) es el primero >= 0.50
    assert decision["effective_date"] is None  # no hay fecha posterior en 'dates' (fin de la ventana sintetica)


def test_determine_overlap_rollover_effective_is_next_observed_date_not_signal_date():
    """Regla 8 heredada: la fecha efectiva NUNCA es la misma que la de la señal."""
    rows, schedule = _synthetic_overlap_rows()
    rows = attach_trading_date_and_contract(rows, schedule)
    out_rows = rows[rows["source_file"] == "00_out_03_24.Last.txt"]
    in_rows = rows[rows["source_file"] == "01_in_06_24.Last.txt"]
    dates = sorted(set(out_rows["trading_date"]) & set(in_rows["trading_date"]))
    evidence = compute_overlap_daily_evidence(out_rows, in_rows, dates, {d: 1000 for d in dates})
    # combined_dates con una fecha extra despues del solapamiento (simula el resto del archivo entrante)
    extra_date = dates[-1] + datetime.timedelta(days=1)
    combined = dates + [extra_date]
    decision = determine_overlap_rollover(evidence, combined, _policy())
    assert decision["signal_date"] == dates[2]
    assert decision["effective_date"] == extra_date
    assert decision["effective_date"] != decision["signal_date"]


# ---------------------------------------------------------------------------
# Causalidad -- test de reconstruccion explicito (seccion 4 de la tarea)
# ---------------------------------------------------------------------------

def test_rollover_decision_is_causal_truncated_future_does_not_change_past_decision():
    """Si se recalcula la señal usando SOLO los datos hasta la fecha de la
    señal (sin ver ningun dia posterior), se obtiene EXACTAMENTE la misma
    signal_date -- el test de reconstruccion causal que exige la tarea."""
    rows, schedule = _synthetic_overlap_rows()
    rows = attach_trading_date_and_contract(rows, schedule)
    out_rows = rows[rows["source_file"] == "00_out_03_24.Last.txt"]
    in_rows = rows[rows["source_file"] == "01_in_06_24.Last.txt"]
    dates = sorted(set(out_rows["trading_date"]) & set(in_rows["trading_date"]))
    full_evidence = compute_overlap_daily_evidence(out_rows, in_rows, dates, {d: 1000 for d in dates})

    full_decision = determine_overlap_rollover(full_evidence, dates, _policy())
    signal_date = full_decision["signal_date"]

    # Trunca la evidencia a solo los dias <= signal_date (simula "solo lo conocido hasta ese instante").
    truncated_dates = [d for d in dates if d <= signal_date]
    truncated_evidence = full_evidence[full_evidence["trading_date"] <= signal_date]
    truncated_decision = determine_overlap_rollover(truncated_evidence, truncated_dates, _policy())

    assert truncated_decision["signal_date"] == signal_date


def test_rollover_decision_does_not_change_if_future_days_are_different():
    """Complementario: cambiar los datos DESPUES de la señal no debe
    cambiar la señal ya emitida -- confirma que no hay look-ahead.

    Usa una ventana de 4 dias (no la fixture de 3 dias, donde la señal cae
    justo en el ultimo dia y no dejaria ningun dia 'futuro' que mutar):
    shares crecientes [0.10, 0.30, 0.70, 0.90] -> la señal se confirma en
    el dia 3 (primer dia >= 0.50); el dia 4 es estrictamente posterior."""
    rows, schedule = _synthetic_overlap_rows_with_resolved_crossover()
    rows = attach_trading_date_and_contract(rows, schedule)
    out_rows = rows[rows["source_file"] == "00_out_03_24.Last.txt"]
    in_rows = rows[rows["source_file"] == "01_in_06_24.Last.txt"]
    dates = sorted(set(out_rows["trading_date"]) & set(in_rows["trading_date"]))
    evidence = compute_overlap_daily_evidence(out_rows, in_rows, dates, {d: 1000 for d in dates})
    decision_a = determine_overlap_rollover(evidence, dates, _policy())
    assert decision_a["signal_date"] == dates[2]  # confirma que la señal NO cae en el ultimo dia

    # Se fuerza el share del ULTIMO dia (posterior a la señal) a un valor absurdo.
    evidence_b = evidence.copy()
    evidence_b.loc[evidence_b["trading_date"] == dates[-1], "share_shared"] = 0.01
    decision_b = determine_overlap_rollover(evidence_b, dates, _policy())

    assert decision_a["signal_date"] == decision_b["signal_date"]


# ---------------------------------------------------------------------------
# classify_transition -- OVERLAP / NO_OVERLAP
# ---------------------------------------------------------------------------

def test_classify_transition_detects_overlap():
    rows, schedule = _synthetic_overlap_rows()
    rows = attach_trading_date_and_contract(rows, schedule)
    full_grid_counts = {d: 1000 for d in rows["trading_date"].dropna().unique()}
    result = classify_transition("00_out_03_24.Last.txt", "01_in_06_24.Last.txt", rows, full_grid_counts, _policy())
    assert result.transition_type == TRANSITION_OVERLAP
    assert len(result.overlap_dates) == 3
    assert result.rollover["signal_date"] is not None


def test_classify_transition_detects_no_overlap():
    schedule = build_session_schedule("2024-01-08", "2024-01-10")
    d1 = schedule.table.loc[datetime.date(2024, 1, 8)]
    d2 = schedule.table.loc[datetime.date(2024, 1, 10)]
    rows = pd.DataFrame(
        [
            {"source_file": "00_out_03_24.Last.txt", "timestamp": (d1["market_open"] + pd.Timedelta(minutes=1)).tz_localize(None),
             "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 10},
            {"source_file": "01_in_06_24.Last.txt", "timestamp": (d2["market_open"] + pd.Timedelta(minutes=1)).tz_localize(None),
             "open": 105.0, "high": 105.5, "low": 104.5, "close": 105.0, "volume": 10},
        ]
    )
    rows = attach_trading_date_and_contract(rows, schedule)
    result = classify_transition("00_out_03_24.Last.txt", "01_in_06_24.Last.txt", rows, {}, _policy())
    assert result.transition_type == TRANSITION_NO_OVERLAP
    assert result.no_overlap_evidence["confidence"] == "BAJA"
    assert result.no_overlap_evidence["apparent_diff_points"] == pytest.approx(5.0)


def test_no_overlap_evidence_never_forces_high_confidence():
    schedule = build_session_schedule("2024-01-08", "2024-01-10")
    d1 = schedule.table.loc[datetime.date(2024, 1, 8)]
    d2 = schedule.table.loc[datetime.date(2024, 1, 10)]
    out_rows = pd.DataFrame(
        [{"source_file": "00_out_03_24.Last.txt",
          "timestamp": (d1["market_open"] + pd.Timedelta(minutes=1)).tz_localize(None),
          "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 10}]
    )
    in_rows = pd.DataFrame(
        [{"source_file": "01_in_06_24.Last.txt",
          "timestamp": (d2["market_open"] + pd.Timedelta(minutes=1)).tz_localize(None),
          "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 10}]
    )
    evidence = compute_no_overlap_evidence(out_rows, in_rows)
    assert evidence["confidence"] == "BAJA"  # incluso con diff aparente = 0


# ---------------------------------------------------------------------------
# build_active_contract_calendar -- irreversibilidad + regla de respaldo
# ---------------------------------------------------------------------------

def test_active_calendar_is_pre_crossover_then_post_crossover_never_reverts():
    rows, schedule = _synthetic_overlap_rows()
    rows = attach_trading_date_and_contract(rows, schedule)
    full_grid_counts = {d: 1000 for d in rows["trading_date"].dropna().unique()}
    result = classify_transition("00_out_03_24.Last.txt", "01_in_06_24.Last.txt", rows, full_grid_counts, _policy())
    calendar = build_active_contract_calendar(rows, ["00_out_03_24.Last.txt", "01_in_06_24.Last.txt"], [result])
    calendar = calendar.sort_values("trading_date").reset_index(drop=True)

    reasons = calendar["reason"].tolist()
    # Una vez que aparece POST_CROSSOVER, nunca debe volver a PRE_CROSSOVER.
    if REASON_POST_CROSSOVER in reasons:
        first_post = reasons.index(REASON_POST_CROSSOVER)
        assert REASON_PRE_CROSSOVER not in reasons[first_post:]
    active_files = calendar["active_file"].tolist()
    # Irreversibilidad: una vez que "01_in_06_24.Last.txt" aparece, "00_out_03_24.Last.txt" no debe volver a aparecer despues.
    if "01_in_06_24.Last.txt" in active_files:
        first_in = active_files.index("01_in_06_24.Last.txt")
        assert "00_out_03_24.Last.txt" not in active_files[first_in:]


def test_zero_bar_fallback_uses_incoming_without_advancing_formal_crossover():
    """Regla de respaldo (generalizacion de la regla 11 heredada): si el
    saliente (formalmente aun activo) tiene 0 barras un dia dentro de la
    ventana de solapamiento y el entrante SI tiene, se usa el entrante
    SOLO ese dia, sin adelantar el cruce formal."""
    schedule = build_session_schedule("2024-01-08", "2024-01-11")
    days = [datetime.date(2024, 1, 8), datetime.date(2024, 1, 9), datetime.date(2024, 1, 10), datetime.date(2024, 1, 11)]
    records = []
    # dia1: solo OUT tiene barras (normal, antes del solapamiento real)
    # dia2: OUT tiene 0 barras (fallback), IN si tiene -- pero el cruce formal NO debe activarse aqui
    # dia3: ambos tienen barras, IN domina (>=50%) -> señal
    # dia4: solo IN (ya post-crossover)
    open_ts = {d: schedule.table.loc[d, "market_open"] for d in days}
    for d, has_out, has_in, in_share in zip(days, [True, False, True, False], [False, True, True, True], [0, 1.0, 0.9, 1.0]):
        m = (open_ts[d] + pd.Timedelta(minutes=1)).tz_localize(None)
        if has_out:
            records.append({"source_file": "00_out_03_24.Last.txt", "timestamp": m, "open": 100.0, "high": 100.5,
                             "low": 99.5, "close": 100.0, "volume": round(100 * (1 - in_share)) or 1})
        if has_in:
            records.append({"source_file": "01_in_06_24.Last.txt", "timestamp": m, "open": 101.0, "high": 101.5,
                             "low": 100.5, "close": 101.0, "volume": round(100 * in_share) or 1})
    rows = pd.DataFrame(records)
    rows = attach_trading_date_and_contract(rows, schedule)
    full_grid_counts = {d: 1000 for d in rows["trading_date"].dropna().unique()}
    result = classify_transition("00_out_03_24.Last.txt", "01_in_06_24.Last.txt", rows, full_grid_counts, _policy())
    calendar = build_active_contract_calendar(rows, ["00_out_03_24.Last.txt", "01_in_06_24.Last.txt"], [result])
    calendar = calendar.set_index("trading_date")

    assert calendar.loc[days[1], "reason"] == REASON_ZERO_BAR_FALLBACK
    assert calendar.loc[days[1], "active_file"] == "01_in_06_24.Last.txt"


# ---------------------------------------------------------------------------
# build_canonical_series -- unicidad, monotonicidad, conservacion
# ---------------------------------------------------------------------------

def test_canonical_series_has_no_duplicate_timestamps_and_is_monotonic():
    rows, schedule = _synthetic_overlap_rows()
    rows = attach_trading_date_and_contract(rows, schedule)
    full_grid_counts = {d: 1000 for d in rows["trading_date"].dropna().unique()}
    result = classify_transition("00_out_03_24.Last.txt", "01_in_06_24.Last.txt", rows, full_grid_counts, _policy())
    calendar = build_active_contract_calendar(rows, ["00_out_03_24.Last.txt", "01_in_06_24.Last.txt"], [result])
    canonical, discarded = build_canonical_series(rows, calendar)

    assert not canonical["timestamp"].duplicated().any()
    assert canonical["timestamp"].is_monotonic_increasing
    assert len(canonical) + len(discarded) == len(rows)
    # Exactamente un contrato activo por timestamp: nunca dos filas del mismo minuto.
    assert canonical.groupby("timestamp").size().max() == 1


def test_canonical_series_preserves_source_file_and_contract_traceability():
    rows, schedule = _synthetic_overlap_rows()
    rows = attach_trading_date_and_contract(rows, schedule)
    full_grid_counts = {d: 1000 for d in rows["trading_date"].dropna().unique()}
    result = classify_transition("00_out_03_24.Last.txt", "01_in_06_24.Last.txt", rows, full_grid_counts, _policy())
    calendar = build_active_contract_calendar(rows, ["00_out_03_24.Last.txt", "01_in_06_24.Last.txt"], [result])
    canonical, _ = build_canonical_series(rows, calendar)
    assert {"source_file", "contract", "trading_date", "open", "high", "low", "close", "volume"}.issubset(
        canonical.columns
    )


# ---------------------------------------------------------------------------
# build_roll_mask
# ---------------------------------------------------------------------------

def test_roll_mask_flags_exactly_the_contract_boundary():
    rows, schedule = _synthetic_overlap_rows_with_resolved_crossover()
    rows = attach_trading_date_and_contract(rows, schedule)
    full_grid_counts = {d: 1000 for d in rows["trading_date"].dropna().unique()}
    result = classify_transition("00_out_03_24.Last.txt", "01_in_06_24.Last.txt", rows, full_grid_counts, _policy())
    calendar = build_active_contract_calendar(rows, ["00_out_03_24.Last.txt", "01_in_06_24.Last.txt"], [result])
    canonical, _ = build_canonical_series(rows, calendar)

    transitions_df = pd.DataFrame([{
        "out_file": "00_out_03_24.Last.txt", "in_file": "01_in_06_24.Last.txt", "transition_type": result.transition_type,
        "rollover_rule": result.rollover["rule"],
    }])
    mask = build_roll_mask(canonical, transitions_df)
    boundary_rows = mask[mask["is_roll_boundary"]]
    assert len(boundary_rows) == 1
    assert boundary_rows.iloc[0]["source_file"] == "01_in_06_24.Last.txt"
    assert boundary_rows.iloc[0]["contract"] == "M24"
    assert boundary_rows.iloc[0]["prev_contract"] == "H24"


# ---------------------------------------------------------------------------
# compute_basis_evolution
# ---------------------------------------------------------------------------

def test_basis_evolution_uses_only_simultaneous_timestamps():
    ts = pd.date_range("2024-01-08 15:00:00", periods=3, freq="1min")
    out_rows = pd.DataFrame(
        {"trading_date": [datetime.date(2024, 1, 8)] * 3, "ts_utc": ts.tz_localize("UTC"), "close": [100.0, 101.0, 102.0]}
    )
    in_rows = pd.DataFrame(
        {"trading_date": [datetime.date(2024, 1, 8)] * 3, "ts_utc": ts.tz_localize("UTC"), "close": [103.0, 104.0, 106.0]}
    )
    basis = compute_basis_evolution(out_rows, in_rows, [datetime.date(2024, 1, 8)])
    assert basis.iloc[0]["n_pairs"] == 3
    assert basis.iloc[0]["diff_points_mean"] == pytest.approx((3 + 3 + 4) / 3)


def test_basis_evolution_handles_no_simultaneous_timestamps():
    out_rows = pd.DataFrame(
        {"trading_date": [datetime.date(2024, 1, 8)], "ts_utc": pd.to_datetime(["2024-01-08 15:00:00"]).tz_localize("UTC"), "close": [100.0]}
    )
    in_rows = pd.DataFrame(
        {"trading_date": [datetime.date(2024, 1, 8)], "ts_utc": pd.to_datetime(["2024-01-08 15:05:00"]).tz_localize("UTC"), "close": [103.0]}
    )
    basis = compute_basis_evolution(out_rows, in_rows, [datetime.date(2024, 1, 8)])
    assert basis.iloc[0]["n_pairs"] == 0


# ---------------------------------------------------------------------------
# compute_adjustment_factors / apply_adjustments
# ---------------------------------------------------------------------------

def test_ratio_adjustment_preserves_returns_within_segment():
    close = pd.Series([100.0, 102.0, 101.0])
    adj = close * 1.5
    returns_raw = close.pct_change().dropna()
    returns_adj = adj.pct_change().dropna()
    assert (returns_raw.round(10) == returns_adj.round(10)).all()


def test_diff_adjustment_preserves_point_differences_within_segment():
    close = pd.Series([100.0, 102.0, 101.0])
    adj = close + 50.0
    diffs_raw = close.diff().dropna()
    diffs_adj = adj.diff().dropna()
    assert (diffs_raw == diffs_adj).all()


def test_adjustment_factor_chain_breaks_at_no_overlap_transition():
    """La cadena de ajuste solo se propaga a traves de transiciones OVERLAP
    (basis medido). Con un NO_OVERLAP en medio, el archivo mas antiguo
    debe quedar SIN factor (NaN), no con uno inventado."""
    files_order = ["00_a_03_24.Last.txt", "01_b_06_24.Last.txt", "02_c_09_24.Last.txt"]

    class FakeResult:
        def __init__(self, out_file, in_file, ttype, rollover, basis_evolution=None):
            self.out_file = out_file
            self.in_file = in_file
            self.transition_type = ttype
            self.rollover = rollover
            self.basis_evolution = basis_evolution

    eff_date = datetime.date(2024, 1, 10)
    overlap_result = FakeResult(
        "01_b_06_24.Last.txt", "02_c_09_24.Last.txt", TRANSITION_OVERLAP,
        {"effective_date": eff_date},
        pd.DataFrame([{"trading_date": eff_date, "diff_points_mean": 10.0, "ratio_mean": 1.1}]),
    )
    no_overlap_result = FakeResult("00_a_03_24.Last.txt", "01_b_06_24.Last.txt", TRANSITION_NO_OVERLAP, {"effective_date": None})

    factors = compute_adjustment_factors([no_overlap_result, overlap_result], files_order)
    factors = factors.set_index("source_file")

    assert factors.loc["02_c_09_24.Last.txt", "basis_chain"] == True  # noqa: E712
    assert factors.loc["01_b_06_24.Last.txt", "basis_chain"] == True  # noqa: E712
    assert factors.loc["01_b_06_24.Last.txt", "ratio_factor"] == pytest.approx(1.1)
    assert factors.loc["01_b_06_24.Last.txt", "diff_factor"] == pytest.approx(10.0)
    assert factors.loc["00_a_03_24.Last.txt", "basis_chain"] == False  # noqa: E712
    assert pd.isna(factors.loc["00_a_03_24.Last.txt", "ratio_factor"])
    assert pd.isna(factors.loc["00_a_03_24.Last.txt", "diff_factor"])


def test_apply_adjustments_does_not_overwrite_raw_columns():
    canonical = pd.DataFrame(
        {"source_file": ["00_a_03_24.Last.txt"], "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [10]}
    )
    factors = pd.DataFrame(
        {"source_file": ["00_a_03_24.Last.txt"], "ratio_factor": [2.0], "diff_factor": [5.0], "basis_chain": [True]}
    )
    out = apply_adjustments(canonical, factors)
    assert out.loc[0, "close"] == 100.5  # cruda intacta
    assert out.loc[0, "close_adj_ratio"] == pytest.approx(201.0)
    assert out.loc[0, "close_adj_diff"] == pytest.approx(105.5)


# ---------------------------------------------------------------------------
# find_extreme_jumps / STOP-3 -- no se dispara con volumen alto y sin reversion
# ---------------------------------------------------------------------------

def test_extreme_jump_with_high_volume_and_no_reversion_is_not_suspicious():
    ts = pd.date_range("2024-01-08 15:00:00", periods=4, freq="1min")
    canonical = pd.DataFrame(
        {
            "timestamp": ts, "source_file": ["00_a_03_24.Last.txt"] * 4, "contract": ["H24"] * 4,
            "trading_date": [datetime.date(2024, 1, 8)] * 4, "segment_id": [1, 1, 1, 1],
            "close": [100.0, 150.0, 151.0, 152.0], "volume": [1000, 5000, 4000, 3000],
        }
    )
    jumps = find_extreme_jumps(canonical, _policy(extreme_jump_top_n=5))
    assert len(jumps) > 0
    assert not jumps["suspicious"].any()


def test_extreme_jump_across_roll_boundary_is_excluded():
    ts = pd.date_range("2024-01-08 15:00:00", periods=3, freq="1min")
    canonical = pd.DataFrame(
        {
            "timestamp": ts, "source_file": ["00_a_03_24.Last.txt", "01_b_06_24.Last.txt", "01_b_06_24.Last.txt"], "contract": ["H24", "M24", "M24"],
            "trading_date": [datetime.date(2024, 1, 8)] * 3, "segment_id": [1, 2, 2],
            "close": [100.0, 500.0, 501.0], "volume": [1000, 1000, 1000],
        }
    )
    jumps = find_extreme_jumps(canonical, _policy(extreme_jump_top_n=5))
    # El salto 100->500 cruza una frontera de segmento (roll): no debe aparecer.
    assert not ((jumps["close_prev"] == 100.0) & (jumps["close"] == 500.0)).any()


# ---------------------------------------------------------------------------
# build_invariance_table
# ---------------------------------------------------------------------------

def test_invariance_table_has_expected_categories_and_types():
    table = build_invariance_table()
    names = set(table["statistic"])
    assert any("etorno simple" in n for n in names)
    assert any("og-retorno" in n for n in names)
    assert any("Diferencia en puntos" in n for n in names)
    row_return = table[table["statistic"].str.contains("etorno simple")].iloc[0]
    assert bool(row_return["invariant_to_ratio"]) is True
    assert bool(row_return["invariant_to_diff"]) is False
    row_diff = table[table["statistic"].str.contains("Diferencia en puntos")].iloc[0]
    assert bool(row_diff["invariant_to_ratio"]) is False
    assert bool(row_diff["invariant_to_diff"]) is True


# ---------------------------------------------------------------------------
# Proteccion del hold-out
# ---------------------------------------------------------------------------

def test_run_tda03_raises_when_research_and_holdout_overlap(tmp_path):
    raw_dir = tmp_path / "data" / "raw" / "mnq"
    raw_dir.mkdir(parents=True)
    _write(raw_dir, "00_mnq_03_20.Last.txt", ["20200102 090000;100.00;100.25;99.75;100.00;10"])
    config = _make_config(
        tmp_path,
        research_files=["00_mnq_03_20.Last.txt"],
        holdout_files=["00_mnq_03_20.Last.txt", "no_existe_en_disco.Last.txt"],
    )
    with pytest.raises(HoldoutIsolationError, match="disjuntos"):
        run_tda03_analysis(config)


def test_run_tda03_raises_when_research_row_reaches_boundary(tmp_path):
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
        run_tda03_analysis(config)
    assert not (raw_dir / "no_existe_en_disco.Last.txt").exists()


def test_run_tda03_never_opens_holdout_only_file(tmp_path):
    raw_dir = tmp_path / "data" / "raw" / "mnq"
    raw_dir.mkdir(parents=True)
    schedule = build_session_schedule("2024-01-08", "2024-01-09")
    lines = []
    for d in [datetime.date(2024, 1, 8), datetime.date(2024, 1, 9)]:
        open_ts = schedule.table.loc[d, "market_open"]
        close_ts = schedule.table.loc[d, "market_close"]
        for m in pd.date_range(open_ts + pd.Timedelta(minutes=1), close_ts, freq="1min")[:3]:
            lines.append(f"{m.strftime('%Y%m%d %H%M%S')};100.00;100.25;99.75;100.00;5")
    _write(raw_dir, "00_mnq_03_20.Last.txt", lines)
    config = _make_config(
        tmp_path,
        research_files=["00_mnq_03_20.Last.txt"],
        holdout_files=["archivo_que_no_existe_en_disco.Last.txt"],
        boundary_utc="2099-01-01 00:00:00",
    )
    result = run_tda03_analysis(config)
    assert result.stop3["triggered"] is False
    assert len(result.transitions) == 0  # un solo archivo, 0 transiciones


# ---------------------------------------------------------------------------
# Integracion ligera end-to-end sobre un par de archivos sinteticos con overlap
# ---------------------------------------------------------------------------

def test_run_tda03_end_to_end_two_files_with_overlap(tmp_path):
    raw_dir = tmp_path / "data" / "raw" / "mnq"
    raw_dir.mkdir(parents=True)
    schedule = build_session_schedule("2024-01-08", "2024-01-12")
    days = [datetime.date(2024, 1, 8), datetime.date(2024, 1, 9), datetime.date(2024, 1, 10),
            datetime.date(2024, 1, 11), datetime.date(2024, 1, 12)]

    out_lines, in_lines = [], []
    for i, d in enumerate(days):
        open_ts = schedule.table.loc[d, "market_open"]
        close_ts = schedule.table.loc[d, "market_close"]
        minutes = pd.date_range(open_ts + pd.Timedelta(minutes=1), close_ts, freq="1min")[:4]
        in_share = [0.1, 0.2, 0.6, 0.8, 0.9][i]
        for m in minutes:
            vol_in = max(round(100 * in_share), 1)
            vol_out = max(100 - vol_in, 1)
            out_lines.append(f"{m.strftime('%Y%m%d %H%M%S')};100.00;100.25;99.75;100.00;{vol_out}")
            in_lines.append(f"{m.strftime('%Y%m%d %H%M%S')};105.00;105.25;104.75;105.00;{vol_in}")

    _write(raw_dir, "00_mnq_03_24.Last.txt", out_lines)
    _write(raw_dir, "01_mnq_06_24.Last.txt", in_lines)
    config = _make_config(
        tmp_path, research_files=["00_mnq_03_24.Last.txt", "01_mnq_06_24.Last.txt"]
    )

    result = run_tda03_analysis(config)

    assert len(result.transitions) == 1
    assert result.transitions.iloc[0]["transition_type"] == TRANSITION_OVERLAP
    assert not result.canonical["timestamp"].duplicated().any()
    assert result.canonical["timestamp"].is_monotonic_increasing
    assert len(result.canonical) + len(result.discarded) == len(result.rows)
    assert int(result.roll_mask["is_roll_boundary"].sum()) == 1
