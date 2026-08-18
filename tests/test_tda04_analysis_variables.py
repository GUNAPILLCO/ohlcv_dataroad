"""Tests de TDA-04 -- construccion y auditoria de las variables de analisis de 1 minuto.

Igual que TDA-02/03: datos sinteticos minimos para aislar cada propiedad
(reglas de no-cruce, causalidad, comprobaciones matematicas de cada
variable), mas la proteccion del hold-out. No se reproducen aqui los
numeros exactos del conjunto de investigacion real (esos viven en
``reports/mnq/TDA04_variables_analisis.md``, generado por ``run_tda04.py``).
"""
from __future__ import annotations

import datetime
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ohlcv_dataroad.config import load_config
from ohlcv_dataroad.ingest.holdout_guard import HoldoutIsolationError
from ohlcv_dataroad.ingest.tda04_analysis_variables import (
    REASON_FIRST_OBSERVATION,
    REASON_NON_CONSECUTIVE_MINUTE,
    REASON_ROLL_BOUNDARY,
    REASON_TRADING_DATE_BOUNDARY,
    REASON_VALID,
    RollConsistencyError,
    audit_losses_by_cause,
    audit_th07_r_vs_R,
    build_analysis_variables,
    build_return_validity_mask,
    run_tda04_analysis,
)

TS_FORMAT = "%Y%m%d %H%M%S"


# ---------------------------------------------------------------------------
# Config sintetica + escritura de artefactos de entrada de TDA-03
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
        "instrument_spec": {"tick_size": 0.25, "tick_size_source": "test"},
        "holdout": {
            "boundary_utc": boundary_utc, "boundary_source": "test",
            "research_files": research_files, "holdout_files": holdout_files or [],
        },
        "tda00": {
            "output_dir_reports": "reports/mnq", "output_dir_interim": "data/interim/mnq",
            "inventory_report": "TDA00_inventario.md", "violations_report": "TDA00_violaciones.csv",
            "per_file_summary": "TDA00_resumen_por_archivo.csv", "bad_data_mask": "tda00_bad_data_mask.parquet",
        },
        "tda03": {
            "min_incoming_share_shared": 0.50, "confirmation_sessions_required": 1, "extreme_jump_top_n": 10,
            "output_dir_reports": "reports/mnq", "output_dir_interim": "data/interim/mnq",
            "report_name": "TDA03_rolls_serie_continua.md", "transitions_csv": "TDA03_transiciones.csv",
            "overlap_daily_evidence_csv": "TDA03_evidencia_diaria_solapamientos.csv",
            "no_overlap_evidence_csv": "TDA03_evidencia_sin_solapamiento.csv",
            "invariance_table_csv": "TDA03_tabla_invariancia.csv", "discarded_rows_csv": "TDA03_filas_descartadas.csv",
            "extreme_jumps_csv": "TDA03_saltos_extremos_stop3.csv", "adjustment_factors_csv": "TDA03_factores_ajuste.csv",
            "roll_mask_name": "tda03_roll_mask.parquet", "canonical_series_name": "tda03_serie_continua.parquet",
        },
        "tda04": {
            "output_dir_reports": "reports/mnq", "output_dir_interim": "data/interim/mnq",
            "report_name": "TDA04_variables_analisis.md", "variables_parquet_name": "tda04_variables_1m.parquet",
            "validity_mask_parquet_name": "tda04_return_validity_mask.parquet",
            "losses_by_cause_csv": "TDA04_perdidas_por_causa.csv", "th07_r_vs_R_csv": "TDA04_th07_r_vs_R.csv",
        },
    }
    config_path = config_dir / "mnq_snapshot.yaml"
    config_path.write_text(yaml.safe_dump(config_yaml), encoding="utf-8")
    return load_config(config_path)


def _write_canonical_and_mask(config, canonical: pd.DataFrame) -> None:
    """Escribe los dos parquet de entrada de TDA-04 a partir de un DataFrame
    canonico sintetico -- la mascara de roll se deriva EXACTAMENTE como lo
    hace TDA-03 (``build_roll_mask``): primera fila de cada ``segment_id``
    nuevo. No se reimporta ``tda03_rolls`` a proposito (se prueba TDA-04
    de forma aislada de TDA-03), pero la logica replicada aqui es la
    misma, de una sola linea.
    """
    config.interim_dir.mkdir(parents=True, exist_ok=True)
    canonical = canonical.sort_values("timestamp").reset_index(drop=True)
    canonical.to_parquet(config.tda03_canonical_series_path, index=False)

    mask = canonical[["timestamp", "source_file", "contract", "trading_date", "segment_id"]].copy()
    mask["prev_contract"] = mask["contract"].shift(1)
    mask["is_roll_boundary"] = mask["contract"].ne(mask["prev_contract"]) & mask["prev_contract"].notna()
    mask["transition_type"] = None
    mask["overlap"] = False
    mask["rollover_rule"] = None
    mask.to_parquet(config.tda03_roll_mask_path, index=False)


def _bar(ts, source_file, contract, trading_date, segment_id, o, h, l, c, v):
    return {
        "timestamp": pd.Timestamp(ts), "source_file": source_file, "contract": contract,
        "trading_date": trading_date, "segment_id": segment_id,
        "open": o, "high": h, "low": l, "close": c, "volume": v,
    }


# ---------------------------------------------------------------------------
# build_return_validity_mask -- reglas de no-cruce, una por una
# ---------------------------------------------------------------------------

def test_consecutive_minute_same_trading_date_same_segment_is_valid():
    d = datetime.date(2024, 1, 8)
    canonical = pd.DataFrame([
        _bar("2024-01-08 15:00:00", "f1", "H24", d, 1, 100, 101, 99, 100, 10),
        _bar("2024-01-08 15:01:00", "f1", "H24", d, 1, 100, 101, 99, 100.5, 10),
    ])
    mask = canonical[["timestamp", "source_file", "contract", "trading_date", "segment_id"]].copy()
    mask["is_roll_boundary"] = [False, False]
    validity = build_return_validity_mask(canonical, mask)
    assert validity.iloc[1]["r_1m_valid"] == True  # noqa: E712
    assert validity.iloc[1]["invalid_reason"] == REASON_VALID


def test_same_trading_date_gap_of_two_minutes_is_invalid():
    d = datetime.date(2024, 1, 8)
    canonical = pd.DataFrame([
        _bar("2024-01-08 15:00:00", "f1", "H24", d, 1, 100, 101, 99, 100, 10),
        _bar("2024-01-08 15:03:00", "f1", "H24", d, 1, 100, 101, 99, 100.5, 10),  # hueco de 3 min
    ])
    mask = canonical[["timestamp", "source_file", "contract", "trading_date", "segment_id"]].copy()
    mask["is_roll_boundary"] = [False, False]
    validity = build_return_validity_mask(canonical, mask)
    assert validity.iloc[1]["r_1m_valid"] == False  # noqa: E712
    assert validity.iloc[1]["invalid_reason"] == REASON_NON_CONSECUTIVE_MINUTE
    assert validity.iloc[1]["delta_minutes"] == 3.0


def test_trading_date_change_is_invalid_even_if_consecutive_in_dataframe():
    d1, d2 = datetime.date(2024, 1, 8), datetime.date(2024, 1, 9)
    canonical = pd.DataFrame([
        _bar("2024-01-08 22:00:00", "f1", "H24", d1, 1, 100, 101, 99, 100, 10),
        _bar("2024-01-08 23:01:00", "f1", "H24", d2, 1, 105, 106, 104, 105, 10),  # nueva jornada, tras el corte
    ])
    mask = canonical[["timestamp", "source_file", "contract", "trading_date", "segment_id"]].copy()
    mask["is_roll_boundary"] = [False, False]
    validity = build_return_validity_mask(canonical, mask)
    assert validity.iloc[1]["r_1m_valid"] == False  # noqa: E712
    assert validity.iloc[1]["invalid_reason"] == REASON_TRADING_DATE_BOUNDARY


def test_segment_change_is_roll_boundary_invalid():
    d = datetime.date(2024, 1, 8)
    canonical = pd.DataFrame([
        _bar("2024-01-08 15:00:00", "f1", "H24", d, 1, 100, 101, 99, 100, 10),
        _bar("2024-01-08 15:01:00", "f2", "M24", d, 2, 200, 201, 199, 200, 10),  # mismo trading_date, otro contrato
    ])
    mask = canonical[["timestamp", "source_file", "contract", "trading_date", "segment_id"]].copy()
    mask["is_roll_boundary"] = [False, True]
    validity = build_return_validity_mask(canonical, mask)
    assert validity.iloc[1]["r_1m_valid"] == False  # noqa: E712
    assert validity.iloc[1]["invalid_reason"] == REASON_ROLL_BOUNDARY


def test_roll_mask_flag_is_authoritative_even_with_one_minute_delta():
    """Si tda03_roll_mask marca la frontera, es ROLL_BOUNDARY aunque el
    delta temporal sea de 1 minuto exacto (caso hipotetico defensivo)."""
    d = datetime.date(2024, 1, 8)
    canonical = pd.DataFrame([
        _bar("2024-01-08 15:00:00", "f1", "H24", d, 1, 100, 101, 99, 100, 10),
        _bar("2024-01-08 15:01:00", "f2", "M24", d, 2, 200, 201, 199, 200, 10),
    ])
    mask = canonical[["timestamp", "source_file", "contract", "trading_date", "segment_id"]].copy()
    mask["is_roll_boundary"] = [False, True]
    validity = build_return_validity_mask(canonical, mask)
    assert validity.iloc[1]["delta_minutes"] == 1.0
    assert validity.iloc[1]["invalid_reason"] == REASON_ROLL_BOUNDARY  # roll, no VALID, pese al delta de 1 min


# ---------------------------------------------------------------------------
# RollConsistencyError -- validacion bloqueante segment_id/contract/roll_mask
# ---------------------------------------------------------------------------

def test_A_segment_and_contract_change_with_roll_mask_true_is_normal_and_returns_nan():
    """A. segment_id/contract cambian + roll_mask=True -> comportamiento
    normal (sin excepcion), retorno NaN por ROLL_BOUNDARY."""
    d = datetime.date(2024, 1, 8)
    canonical = pd.DataFrame([
        _bar("2024-01-08 15:00:00", "f1", "H24", d, 1, 100, 101, 99, 100, 10),
        _bar("2024-01-08 15:01:00", "f2", "M24", d, 2, 200, 201, 199, 200, 10),
    ])
    mask = canonical[["timestamp", "source_file", "contract", "trading_date", "segment_id"]].copy()
    mask["is_roll_boundary"] = [False, True]
    validity = build_return_validity_mask(canonical, mask)  # no debe lanzar
    assert validity.iloc[1]["r_1m_valid"] == False  # noqa: E712
    assert validity.iloc[1]["invalid_reason"] == REASON_ROLL_BOUNDARY


def test_B_segment_and_contract_change_with_roll_mask_false_raises():
    """B. segment_id/contract cambian + roll_mask=False -> debe fallar la
    validacion: la mascara de TDA-03 dice que NO hay roll, pero la serie
    canonica muestra un cambio de contrato -- inconsistencia real."""
    d = datetime.date(2024, 1, 8)
    canonical = pd.DataFrame([
        _bar("2024-01-08 15:00:00", "f1", "H24", d, 1, 100, 101, 99, 100, 10),
        _bar("2024-01-08 15:01:00", "f2", "M24", d, 2, 200, 201, 199, 200, 10),
    ])
    mask = canonical[["timestamp", "source_file", "contract", "trading_date", "segment_id"]].copy()
    mask["is_roll_boundary"] = [False, False]  # inconsistente: deberia ser True
    with pytest.raises(RollConsistencyError):
        build_return_validity_mask(canonical, mask)


def test_C_roll_mask_true_without_segment_or_contract_change_raises():
    """C. roll_mask=True sin cambio de segment_id/contract -> debe fallar
    la validacion: la mascara dice que hay roll, pero la serie canonica
    no muestra ningun cambio de contrato -- inconsistencia real."""
    d = datetime.date(2024, 1, 8)
    canonical = pd.DataFrame([
        _bar("2024-01-08 15:00:00", "f1", "H24", d, 1, 100, 101, 99, 100, 10),
        _bar("2024-01-08 15:01:00", "f1", "H24", d, 1, 100, 101, 99, 100.5, 10),  # mismo segment/contract
    ])
    mask = canonical[["timestamp", "source_file", "contract", "trading_date", "segment_id"]].copy()
    mask["is_roll_boundary"] = [False, True]  # inconsistente: deberia ser False
    with pytest.raises(RollConsistencyError):
        build_return_validity_mask(canonical, mask)


def test_D_invariant_holds_on_real_tda03_artifacts_for_all_21_boundaries():
    """D. Sobre los artefactos REALES de TDA-03 (si estan disponibles en
    este checkout -- se salta la prueba si no se han generado todavia),
    la invariante segment_changed == contract_changed == is_roll_boundary
    se cumple para las 21 fronteras de roll del conjunto de
    investigacion, sin lanzar RollConsistencyError."""
    repo_root = Path(__file__).resolve().parents[1]
    canonical_path = repo_root / "data" / "interim" / "mnq" / "tda03_serie_continua.parquet"
    roll_mask_path = repo_root / "data" / "interim" / "mnq" / "tda03_roll_mask.parquet"
    if not (canonical_path.exists() and roll_mask_path.exists()):
        pytest.skip("artefactos reales de TDA-03 no generados en este checkout")

    canonical = pd.read_parquet(canonical_path)
    roll_mask = pd.read_parquet(roll_mask_path)
    validity = build_return_validity_mask(canonical, roll_mask)  # no debe lanzar
    assert int(validity["is_roll_boundary"].sum()) == 21
    assert int((validity["invalid_reason"] == REASON_ROLL_BOUNDARY).sum()) == 21


def test_first_row_of_series_is_first_observation():
    d = datetime.date(2024, 1, 8)
    canonical = pd.DataFrame([_bar("2024-01-08 15:00:00", "f1", "H24", d, 1, 100, 101, 99, 100, 10)])
    mask = canonical[["timestamp", "source_file", "contract", "trading_date", "segment_id"]].copy()
    mask["is_roll_boundary"] = [False]
    validity = build_return_validity_mask(canonical, mask)
    assert validity.iloc[0]["invalid_reason"] == REASON_FIRST_OBSERVATION
    assert validity.iloc[0]["r_1m_valid"] == False  # noqa: E712


def test_calendar_midnight_crossing_within_same_trading_date_is_valid():
    """23:59 -> 00:00 hora de calendario, pero MISMO trading_date -> valido."""
    d = datetime.date(2024, 1, 9)  # la sesion de esta trading_date abarca la noche del 8 al 9
    canonical = pd.DataFrame([
        _bar("2024-01-08 23:59:00", "f1", "H24", d, 1, 100, 101, 99, 100, 10),
        _bar("2024-01-09 00:00:00", "f1", "H24", d, 1, 100, 101, 99, 100.25, 10),
    ])
    mask = canonical[["timestamp", "source_file", "contract", "trading_date", "segment_id"]].copy()
    mask["is_roll_boundary"] = [False, False]
    validity = build_return_validity_mask(canonical, mask)
    assert validity.iloc[1]["r_1m_valid"] == True  # noqa: E712
    assert validity.iloc[1]["invalid_reason"] == REASON_VALID


def test_bars_after_maintenance_do_not_link_with_previous_trading_date():
    """Reapertura tras el corte de mantenimiento: nueva trading_date, primera
    barra de esa jornada -> invalida (TRADING_DATE_BOUNDARY), no importa
    que en el DataFrame sea la fila inmediatamente siguiente."""
    d1, d2 = datetime.date(2024, 1, 8), datetime.date(2024, 1, 9)
    canonical = pd.DataFrame([
        _bar("2024-01-08 21:59:00", "f1", "H24", d1, 1, 100, 101, 99, 100, 10),  # ultima barra antes del corte
        _bar("2024-01-08 22:00:00", "f1", "H24", d1, 1, 100, 100.5, 99.5, 100.10, 10),  # cierre de sesion (17:00 NY)
        _bar("2024-01-08 23:01:00", "f1", "H24", d2, 1, 110, 111, 109, 110, 10),  # reapertura (18:01 NY), NUEVA jornada
        _bar("2024-01-08 23:02:00", "f1", "H24", d2, 1, 110, 111, 109, 110.5, 10),
    ])
    mask = canonical[["timestamp", "source_file", "contract", "trading_date", "segment_id"]].copy()
    mask["is_roll_boundary"] = [False, False, False, False]
    validity = build_return_validity_mask(canonical, mask)
    assert validity.iloc[2]["invalid_reason"] == REASON_TRADING_DATE_BOUNDARY  # reapertura no enlaza con el cierre previo
    assert validity.iloc[3]["invalid_reason"] == REASON_VALID  # dentro de la nueva jornada, si enlaza


# ---------------------------------------------------------------------------
# build_analysis_variables -- comprobaciones matematicas
# ---------------------------------------------------------------------------

def _simple_canonical_and_validity(closes, opens=None, highs=None, lows=None, same_date=True):
    d = datetime.date(2024, 1, 8)
    n = len(closes)
    opens = opens or [c - 0.5 for c in closes]
    highs = highs or [c + 1.0 for c in closes]
    lows = lows or [c - 1.0 for c in closes]
    ts = pd.date_range("2024-01-08 15:00:00", periods=n, freq="1min")
    canonical = pd.DataFrame(
        [
            _bar(ts[i], "f1", "H24", d, 1, opens[i], highs[i], lows[i], closes[i], 10)
            for i in range(n)
        ]
    )
    mask = canonical[["timestamp", "source_file", "contract", "trading_date", "segment_id"]].copy()
    mask["is_roll_boundary"] = [False] * n
    validity = build_return_validity_mask(canonical, mask)
    return canonical, validity


def test_r_1m_matches_log_of_close_ratio_for_valid_pair():
    canonical, validity = _simple_canonical_and_validity([100.0, 102.0, 101.0])
    variables = build_analysis_variables(canonical, validity)
    assert variables.iloc[1]["r_1m"] == pytest.approx(math.log(102.0 / 100.0))
    assert variables.iloc[2]["r_1m"] == pytest.approx(math.log(101.0 / 102.0))


def test_r_1m_is_zero_when_close_does_not_change():
    canonical, validity = _simple_canonical_and_validity([100.0, 100.0])
    variables = build_analysis_variables(canonical, validity)
    assert variables.iloc[1]["r_1m"] == pytest.approx(0.0)


def test_abs_r_1m_equals_abs_of_r_1m():
    canonical, validity = _simple_canonical_and_validity([100.0, 95.0, 99.0])
    variables = build_analysis_variables(canonical, validity)
    valid = variables["r_1m"].notna()
    assert (variables.loc[valid, "abs_r_1m"] == variables.loc[valid, "r_1m"].abs()).all()


def test_r2_1m_equals_r_1m_squared():
    canonical, validity = _simple_canonical_and_validity([100.0, 95.0, 99.0])
    variables = build_analysis_variables(canonical, validity)
    valid = variables["r_1m"].notna()
    assert np.allclose(variables.loc[valid, "r2_1m"], variables.loc[valid, "r_1m"] ** 2)


def test_log_hl_equals_ln_high_over_low():
    canonical, validity = _simple_canonical_and_validity(
        [100.0], opens=[99.5], highs=[103.0], lows=[97.0]
    )
    variables = build_analysis_variables(canonical, validity)
    assert variables.iloc[0]["log_hl"] == pytest.approx(math.log(103.0 / 97.0))


def test_log_co_equals_ln_close_over_open():
    canonical, validity = _simple_canonical_and_validity(
        [100.0], opens=[98.0], highs=[103.0], lows=[97.0]
    )
    variables = build_analysis_variables(canonical, validity)
    assert variables.iloc[0]["log_co"] == pytest.approx(math.log(100.0 / 98.0))


def test_log_oc_prev_equals_ln_open_over_prev_close_only_when_comparable():
    canonical, validity = _simple_canonical_and_validity(
        [100.0, 102.0], opens=[99.5, 101.0]
    )
    variables = build_analysis_variables(canonical, validity)
    assert variables.iloc[1]["log_oc_prev"] == pytest.approx(math.log(101.0 / 100.0))

    # Ahora con una frontera de trading_date entre las dos filas: debe ser NaN.
    d1, d2 = datetime.date(2024, 1, 8), datetime.date(2024, 1, 9)
    canonical2 = pd.DataFrame([
        _bar("2024-01-08 22:00:00", "f1", "H24", d1, 1, 99.5, 101, 99, 100.0, 10),
        _bar("2024-01-08 23:01:00", "f1", "H24", d2, 1, 101.0, 102, 100, 102.0, 10),
    ])
    mask2 = canonical2[["timestamp", "source_file", "contract", "trading_date", "segment_id"]].copy()
    mask2["is_roll_boundary"] = [False, False]
    validity2 = build_return_validity_mask(canonical2, mask2)
    variables2 = build_analysis_variables(canonical2, validity2)
    assert pd.isna(variables2.iloc[1]["log_oc_prev"])
    assert pd.isna(variables2.iloc[1]["r_1m"])  # misma regla, misma fila


def test_zero_1m_is_nan_exactly_where_r_1m_is_nan():
    d1, d2 = datetime.date(2024, 1, 8), datetime.date(2024, 1, 9)
    canonical = pd.DataFrame([
        _bar("2024-01-08 22:00:00", "f1", "H24", d1, 1, 99.5, 101, 99, 100.0, 10),
        _bar("2024-01-08 23:01:00", "f1", "H24", d2, 1, 101.0, 102, 100, 100.0, 10),  # cruza frontera -> NaN
        _bar("2024-01-08 23:02:00", "f1", "H24", d2, 1, 100.0, 101, 99, 100.0, 10),   # valido, sin cambio -> zero_1m=1
    ])
    mask = canonical[["timestamp", "source_file", "contract", "trading_date", "segment_id"]].copy()
    mask["is_roll_boundary"] = [False, False, False]
    validity = build_return_validity_mask(canonical, mask)
    variables = build_analysis_variables(canonical, validity)
    assert pd.isna(variables.iloc[1]["zero_1m"])
    assert variables.iloc[2]["zero_1m"] == 1.0


def test_no_function_fills_a_boundary_nan():
    """Ninguna variable derivada (abs, cuadrado, indicador) puede convertir
    un NaN de frontera en un numero -- deben seguir siendo NaN."""
    d1, d2 = datetime.date(2024, 1, 8), datetime.date(2024, 1, 9)
    canonical = pd.DataFrame([
        _bar("2024-01-08 22:00:00", "f1", "H24", d1, 1, 99.5, 101, 99, 100.0, 10),
        _bar("2024-01-08 23:01:00", "f1", "H24", d2, 1, 101.0, 102, 100, 103.0, 10),
    ])
    mask = canonical[["timestamp", "source_file", "contract", "trading_date", "segment_id"]].copy()
    mask["is_roll_boundary"] = [False, False]
    validity = build_return_validity_mask(canonical, mask)
    variables = build_analysis_variables(canonical, validity)
    row = variables.iloc[1]
    assert pd.isna(row["r_1m"]) and pd.isna(row["R_1m"]) and pd.isna(row["abs_r_1m"])
    assert pd.isna(row["r2_1m"]) and pd.isna(row["zero_1m"]) and pd.isna(row["log_oc_prev"])


def test_log_returns_sum_correctly_across_a_valid_sequence():
    """r_t + r_{t+1} = ln(C_{t+1}/C_{t-1}) -- propiedad matematica del
    log-retorno, NO un horizonte de prediccion: las tres barras son
    pasadas/realizadas, no futuras."""
    canonical, validity = _simple_canonical_and_validity([100.0, 103.0, 97.0])
    variables = build_analysis_variables(canonical, validity)
    r1 = variables.iloc[1]["r_1m"]
    r2 = variables.iloc[2]["r_1m"]
    assert (r1 + r2) == pytest.approx(math.log(97.0 / 100.0))


# ---------------------------------------------------------------------------
# Causalidad -- sin look-ahead
# ---------------------------------------------------------------------------

def test_no_look_ahead_truncating_the_series_does_not_change_past_values():
    canonical, validity = _simple_canonical_and_validity([100.0, 103.0, 97.0, 110.0, 90.0])
    variables_full = build_analysis_variables(canonical, validity)

    truncated_canonical = canonical.iloc[:3].reset_index(drop=True)
    truncated_validity = build_return_validity_mask(
        truncated_canonical,
        truncated_canonical[["timestamp", "source_file", "contract", "trading_date", "segment_id"]].assign(
            is_roll_boundary=False
        ),
    )
    variables_truncated = build_analysis_variables(truncated_canonical, truncated_validity)

    def _assert_same(a, b):
        if pd.isna(a):
            assert pd.isna(b)
        else:
            assert a == pytest.approx(b)

    for col in ["r_1m", "R_1m", "abs_r_1m", "r2_1m", "log_hl", "log_co", "log_oc_prev"]:
        _assert_same(variables_full.iloc[1][col], variables_truncated.iloc[1][col])
        _assert_same(variables_full.iloc[2][col], variables_truncated.iloc[2][col])


# ---------------------------------------------------------------------------
# audit_losses_by_cause / audit_th07_r_vs_R
# ---------------------------------------------------------------------------

def test_audit_losses_by_cause_counts_are_exclusive_and_sum_to_total():
    d1, d2 = datetime.date(2024, 1, 8), datetime.date(2024, 1, 9)
    canonical = pd.DataFrame([
        _bar("2024-01-08 15:00:00", "f1", "H24", d1, 1, 100, 101, 99, 100, 10),
        _bar("2024-01-08 15:01:00", "f1", "H24", d1, 1, 100, 101, 99, 100.5, 10),  # valido
        _bar("2024-01-08 15:04:00", "f1", "H24", d1, 1, 100, 101, 99, 101, 10),    # NON_CONSECUTIVE
        _bar("2024-01-09 15:00:00", "f2", "M24", d2, 2, 200, 201, 199, 200, 10),   # ROLL + fecha nueva
    ])
    mask = canonical[["timestamp", "source_file", "contract", "trading_date", "segment_id"]].copy()
    mask["is_roll_boundary"] = [False, False, False, True]
    validity = build_return_validity_mask(canonical, mask)
    audit = audit_losses_by_cause(validity)
    exclusive = audit[audit["block"] == "exclusive_reason"].set_index("label")["n_rows"]
    assert exclusive.sum() == len(validity)
    assert exclusive[REASON_FIRST_OBSERVATION] == 1
    assert exclusive[REASON_NON_CONSECUTIVE_MINUTE] == 1
    assert exclusive[REASON_ROLL_BOUNDARY] == 1
    assert exclusive[REASON_VALID] == 1


def test_audit_th07_global_diff_is_tiny_for_small_moves():
    canonical, validity = _simple_canonical_and_validity([100.0, 100.01, 100.02, 99.99, 100.0, 100.03, 99.98, 100.01, 100.0, 99.99, 100.02])
    variables = build_analysis_variables(canonical, validity)
    th07 = audit_th07_r_vs_R(variables, n_deciles=3)
    global_row = th07[th07["decile"] == "GLOBAL"].iloc[0]
    assert global_row["mean_abs_diff"] < 1e-4


# ---------------------------------------------------------------------------
# Proteccion del hold-out
# ---------------------------------------------------------------------------

def test_run_tda04_raises_when_research_and_holdout_overlap(tmp_path):
    config = _make_config(
        tmp_path, research_files=["00_mnq_03_24.Last.txt"],
        holdout_files=["00_mnq_03_24.Last.txt", "no_existe_en_disco.Last.txt"],
    )
    with pytest.raises(HoldoutIsolationError, match="disjuntos"):
        run_tda04_analysis(config)


def test_run_tda04_raises_when_canonical_row_reaches_boundary(tmp_path):
    config = _make_config(
        tmp_path, research_files=["00_mnq_03_24.Last.txt"],
        holdout_files=["no_existe_en_disco.Last.txt"],
        boundary_utc="2024-01-08 15:01:00",
    )
    canonical, _ = _simple_canonical_and_validity([100.0, 101.0])
    canonical["source_file"] = "00_mnq_03_24.Last.txt"
    _write_canonical_and_mask(config, canonical)

    with pytest.raises(HoldoutIsolationError, match="frontera de hold-out"):
        run_tda04_analysis(config)
    assert not (config.raw_dir / "no_existe_en_disco.Last.txt").exists()


def test_run_tda04_never_opens_any_raw_or_holdout_file(tmp_path):
    """TDA-04 nunca abre `data/raw/` -- ni siquiera el archivo de
    investigacion declarado. Solo lee los parquet de TDA-03."""
    config = _make_config(
        tmp_path, research_files=["00_mnq_03_24.Last.txt"],
        holdout_files=["archivo_que_no_existe_en_disco.Last.txt"],
        boundary_utc="2099-01-01 00:00:00",
    )
    canonical, _ = _simple_canonical_and_validity([100.0, 101.0, 102.0])
    canonical["source_file"] = "00_mnq_03_24.Last.txt"
    _write_canonical_and_mask(config, canonical)

    assert not (config.raw_dir / "00_mnq_03_24.Last.txt").exists()  # no existe en disco
    result = run_tda04_analysis(config)  # y sin embargo la corrida funciona: nunca lo intento abrir
    assert len(result.variables) == 3


def test_row_count_is_conserved_between_input_and_validity_and_variables(tmp_path):
    config = _make_config(tmp_path, research_files=["00_mnq_03_24.Last.txt"])
    canonical, _ = _simple_canonical_and_validity([100.0, 101.0, 99.0, 100.5])
    canonical["source_file"] = "00_mnq_03_24.Last.txt"
    _write_canonical_and_mask(config, canonical)

    result = run_tda04_analysis(config)
    assert len(result.canonical) == len(result.validity) == len(result.variables) == 4
    n_valid = int((result.validity["invalid_reason"] == REASON_VALID).sum())
    n_invalid = len(result.validity) - n_valid
    assert len(result.validity) == n_valid + n_invalid
