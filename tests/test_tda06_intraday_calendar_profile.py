"""Tests de TDA-06 -- perfil determinista intradia y de calendario.

Igual que TDA-04/05: datos sinteticos minimos para aislar cada propiedad
(minute_of_day en America/New_York, DST, weekday de trading_date, no
reintroduccion de retornos invalidos, no relleno de minutos cerrados,
proteccion del hold-out, s(m)/r_tilde solo bajo la puerta STOP-6). No se
reproducen aqui los numeros exactos del conjunto de investigacion real
(esos viven en ``reports/mnq/TDA06_perfil_intradia_calendario.md``).
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
from ohlcv_dataroad.ingest.tda06_intraday_calendar_profile import (
    DEFAULT_N_SURROGATES,
    _centered_rolling_median,
    _permute_preserving_nan_mask,
    attach_calendar_fields,
    build_all_minute_profiles,
    build_r_tilde,
    build_s_m,
    build_segmentation_proposal,
    build_weekday_profile,
    calibrate_breakpoint_detector,
    choose_s_m_proxy,
    compute_composite_activity_score,
    compute_extreme_flag,
    decide_stop6,
    detect_breakpoints,
    run_tda06_analysis,
)

TICK = 0.25


# ---------------------------------------------------------------------------
# Fixtures sinteticas
# ---------------------------------------------------------------------------

def _variables_and_validity(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    variables = pd.DataFrame(
        [
            {
                "timestamp": r["timestamp"], "source_file": r.get("source_file", "f1"),
                "contract": r.get("contract", "H24"), "trading_date": r.get("trading_date", datetime.date(2024, 1, 8)),
                "segment_id": r.get("segment_id", 1),
                "open": r.get("open", 100.0), "high": r.get("high", 100.5), "low": r.get("low", 99.5),
                "close": r.get("close", 100.0), "volume": r.get("volume", 10),
                "r_1m": r.get("r_1m", np.nan), "R_1m": r.get("R_1m", np.nan),
                "abs_r_1m": r.get("abs_r_1m", np.nan), "r2_1m": r.get("r2_1m", np.nan),
                "zero_1m": r.get("zero_1m", np.nan),
                "log_hl": r.get("log_hl", 0.01), "log_co": r.get("log_co", 0.0), "log_oc_prev": r.get("log_oc_prev", np.nan),
            }
            for r in rows
        ]
    )
    validity = pd.DataFrame(
        [{"timestamp": r["timestamp"], "r_1m_valid": r["r_1m_valid"]} for r in rows]
    )
    return variables, validity


# ---------------------------------------------------------------------------
# minute_of_day / weekday / eje temporal
# ---------------------------------------------------------------------------

def test_minute_of_day_uses_ny_not_utc():
    # 2024-01-08 22:15:00 UTC -> invierno, NY = UTC-5 -> 17:15 NY -> 17*60+15=1035
    ts = pd.Timestamp("2024-01-08 22:15:00")
    variables, validity = _variables_and_validity(
        [{"timestamp": ts, "r_1m_valid": False, "trading_date": datetime.date(2024, 1, 8)}]
    )
    out = attach_calendar_fields(variables, validity)
    assert int(out.loc[0, "hour_ny"]) == 17
    assert int(out.loc[0, "minute_of_day"]) == 17 * 60 + 15


def test_minute_of_day_range_is_0_to_1439():
    ts = pd.date_range("2024-01-08 00:00:00", periods=5, freq="17min")
    variables, validity = _variables_and_validity(
        [{"timestamp": t, "r_1m_valid": False, "trading_date": datetime.date(2024, 1, 8)} for t in ts]
    )
    out = attach_calendar_fields(variables, validity)
    assert out["minute_of_day"].between(0, 1439).all()


def test_dst_spring_forward_converts_correctly():
    # 2024-03-10: DST empieza en US a las 02:00 NY -> 07:00 UTC ya es EDT (UTC-4).
    ts_before = pd.Timestamp("2024-03-10 06:59:00")  # 01:59 EST
    ts_after = pd.Timestamp("2024-03-10 07:00:00")   # 03:00 EDT (salta 02:xx)
    variables, validity = _variables_and_validity(
        [
            {"timestamp": ts_before, "r_1m_valid": False, "trading_date": datetime.date(2024, 3, 8)},
            {"timestamp": ts_after, "r_1m_valid": False, "trading_date": datetime.date(2024, 3, 8)},
        ]
    )
    out = attach_calendar_fields(variables, validity)
    assert int(out.loc[0, "minute_of_day"]) == 1 * 60 + 59
    assert int(out.loc[1, "minute_of_day"]) == 3 * 60 + 0


def test_dst_fall_back_converts_correctly():
    # 2024-11-03: DST termina a las 02:00 NY -> a partir de 06:00 UTC ya es EST (UTC-5).
    ts_edt = pd.Timestamp("2024-11-03 05:30:00")  # 01:30 EDT (UTC-4)
    ts_est = pd.Timestamp("2024-11-03 06:30:00")  # 01:30 EST (UTC-5)
    variables, validity = _variables_and_validity(
        [
            {"timestamp": ts_edt, "r_1m_valid": False, "trading_date": datetime.date(2024, 11, 1)},
            {"timestamp": ts_est, "r_1m_valid": False, "trading_date": datetime.date(2024, 11, 1)},
        ]
    )
    out = attach_calendar_fields(variables, validity)
    assert int(out.loc[0, "hour_ny"]) == 1 and int(out.loc[0, "minute_of_day"]) == 90
    assert int(out.loc[1, "hour_ny"]) == 1 and int(out.loc[1, "minute_of_day"]) == 90


def test_sunday_night_bar_gets_weekday_of_monday_trading_date():
    """Una barra del domingo por la noche (hora NY) pertenece a la trading_date del lunes -- su weekday debe ser LUNES, no domingo."""
    # Domingo 2024-01-07 23:30 UTC = domingo 18:30 NY (invierno) -- pero
    # su trading_date (TDA-02/03) es el LUNES 2024-01-08 (la sesion que
    # abre el domingo por la noche cierra -y se etiqueta- el lunes).
    ts = pd.Timestamp("2024-01-07 23:30:00")
    monday = datetime.date(2024, 1, 8)
    variables, validity = _variables_and_validity(
        [{"timestamp": ts, "r_1m_valid": False, "trading_date": monday}]
    )
    out = attach_calendar_fields(variables, validity)
    assert out.loc[0, "weekday"] == 0  # Lunes
    assert out.loc[0, "weekday"] != 6  # NO domingo, aunque el timestamp local caiga en domingo


def test_weekday_computed_from_trading_date_not_local_calendar_date():
    ts = pd.Timestamp("2024-01-08 15:00:00")  # lunes normal, sin cruce de medianoche
    variables, validity = _variables_and_validity(
        [{"timestamp": ts, "r_1m_valid": False, "trading_date": datetime.date(2024, 1, 8)}]
    )
    out = attach_calendar_fields(variables, validity)
    assert out.loc[0, "weekday"] == 0


# ---------------------------------------------------------------------------
# No reintroduccion de retornos invalidos / no relleno de minutos cerrados
# ---------------------------------------------------------------------------

def test_invalid_return_never_enters_return_based_profile():
    """Un r_1m invalido (con un valor extremo puesto a proposito) no debe afectar n ni point del perfil."""
    rows = []
    for d in range(6):
        date = datetime.date(2024, 1, 8 + d)
        ts = pd.Timestamp(f"2024-01-{8+d:02d} 15:00:00")
        rows.append({"timestamp": ts, "trading_date": date, "r_1m_valid": True, "r_1m": 0.001, "abs_r_1m": 0.001, "r2_1m": 1e-6, "zero_1m": 0.0})
    # Fila invalida con un valor absurdo -- si se colara, dominaria la mediana/CI.
    bad_ts = pd.Timestamp("2024-01-14 15:00:00")
    rows.append({"timestamp": bad_ts, "trading_date": datetime.date(2024, 1, 15), "r_1m_valid": False, "r_1m": 999.0, "abs_r_1m": 999.0, "r2_1m": 999.0, "zero_1m": np.nan})
    variables, validity = _variables_and_validity(rows)
    df = attach_calendar_fields(variables, validity)
    profiles = build_all_minute_profiles(df, years=(2024,), n_boot=5, seed=0, extreme_quantile=0.99)
    minute = 10 * 60  # 15:00 UTC (invierno) = 10:00 NY
    row = profiles.global_tables["abs_r_1m"]
    at_minute = row[row["minute_of_day"] == minute].iloc[0]
    assert int(at_minute["n"]) == 6  # NO 7 -- la fila invalida quedo fuera
    assert at_minute["point"] == pytest.approx(0.001)


def test_bar_population_variables_use_all_admissible_bars_including_first_of_session():
    """volume/log_hl NO se restringen a r_1m_valid -- una primera barra de sesion (sin r_1m) SI cuenta."""
    rows = [
        {"timestamp": pd.Timestamp("2024-01-08 09:30:00"), "trading_date": datetime.date(2024, 1, 8), "r_1m_valid": False, "volume": 50, "log_hl": 0.02},
        {"timestamp": pd.Timestamp("2024-01-09 09:30:00"), "trading_date": datetime.date(2024, 1, 9), "r_1m_valid": False, "volume": 60, "log_hl": 0.02},
    ]
    variables, validity = _variables_and_validity(rows)
    df = attach_calendar_fields(variables, validity)
    profiles = build_all_minute_profiles(df, years=(2024,), n_boot=5, seed=0, extreme_quantile=0.99)
    minute = 4 * 60 + 30  # 09:30 UTC (invierno) = 04:30 NY
    vol_row = profiles.global_tables["volume"]
    at_minute = vol_row[vol_row["minute_of_day"] == minute].iloc[0]
    assert int(at_minute["n"]) == 2  # ambas filas cuentan pese a r_1m_valid=False
    # Pero el perfil de r_1m (poblacion de retorno) SI las excluye:
    r_row = profiles.global_tables["r_1m_median"]
    at_minute_r = r_row[r_row["minute_of_day"] == minute].iloc[0]
    assert int(at_minute_r["n"]) == 0


def test_structurally_closed_minutes_are_not_filled_with_synthetic_observations():
    """Un minuto sin ninguna barra en ningun dia debe quedar con n=0 y point=NaN -- nunca inventado."""
    rows = [
        {"timestamp": pd.Timestamp("2024-01-08 09:30:00"), "trading_date": datetime.date(2024, 1, 8), "r_1m_valid": False, "volume": 50, "log_hl": 0.02},
    ]
    variables, validity = _variables_and_validity(rows)
    df = attach_calendar_fields(variables, validity)
    profiles = build_all_minute_profiles(df, years=(2024,), n_boot=5, seed=0, extreme_quantile=0.99)
    vol_row = profiles.global_tables["volume"]
    closed_minute = vol_row[vol_row["minute_of_day"] == 500].iloc[0]  # ninguna barra cae aqui
    assert int(closed_minute["n"]) == 0
    assert np.isnan(closed_minute["point"])
    assert np.isnan(closed_minute["ci_lo"]) and np.isnan(closed_minute["ci_hi"])


def test_minute_profile_table_always_has_all_1440_minutes():
    rows = [{"timestamp": pd.Timestamp("2024-01-08 09:30:00"), "trading_date": datetime.date(2024, 1, 8), "r_1m_valid": False, "volume": 50, "log_hl": 0.02}]
    variables, validity = _variables_and_validity(rows)
    df = attach_calendar_fields(variables, validity)
    profiles = build_all_minute_profiles(df, years=(2024,), n_boot=5, seed=0, extreme_quantile=0.99)
    assert len(profiles.global_tables["volume"]) == 1440
    assert set(profiles.global_tables["volume"]["minute_of_day"]) == set(range(1440))


# ---------------------------------------------------------------------------
# Agregacion minuto-a-minuto correcta (dataset sintetico, valor exacto conocido)
# ---------------------------------------------------------------------------

def test_minute_aggregation_matches_hand_computed_median():
    rows = []
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    for i, v in enumerate(values):
        date = datetime.date(2024, 1, 8 + i)
        rows.append({"timestamp": pd.Timestamp(f"2024-01-{8+i:02d} 12:00:00"), "trading_date": date, "r_1m_valid": False, "volume": v, "log_hl": 0.01})
    variables, validity = _variables_and_validity(rows)
    df = attach_calendar_fields(variables, validity)
    profiles = build_all_minute_profiles(df, years=(2024,), n_boot=5, seed=0, extreme_quantile=0.99)
    minute = 7 * 60  # 12:00 UTC (invierno) = 07:00 NY
    row = profiles.global_tables["volume"]
    at_minute = row[row["minute_of_day"] == minute].iloc[0]
    assert at_minute["n"] == 5
    assert at_minute["point"] == pytest.approx(np.median(values))


def test_zero_1m_profile_is_a_proportion_between_0_and_1():
    rows = []
    for i in range(4):
        date = datetime.date(2024, 1, 8 + i)
        zero = 1.0 if i < 3 else 0.0
        rows.append({"timestamp": pd.Timestamp(f"2024-01-{8+i:02d} 10:00:00"), "trading_date": date, "r_1m_valid": True, "r_1m": 0.0 if zero else 0.001, "abs_r_1m": 0.0 if zero else 0.001, "r2_1m": 0.0, "zero_1m": zero})
    variables, validity = _variables_and_validity(rows)
    df = attach_calendar_fields(variables, validity)
    profiles = build_all_minute_profiles(df, years=(2024,), n_boot=5, seed=0, extreme_quantile=0.99)
    row = profiles.global_tables["zero_1m"]
    at_minute = row[row["minute_of_day"] == 300].iloc[0]  # 10:00 UTC (invierno) = 05:00 NY
    assert at_minute["point"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Estabilidad por año -- sin mezclar etiquetas; años parciales
# ---------------------------------------------------------------------------

def test_by_year_tables_do_not_mix_years():
    rows = [
        {"timestamp": pd.Timestamp("2020-01-08 10:00:00"), "trading_date": datetime.date(2020, 1, 8), "r_1m_valid": False, "volume": 100, "log_hl": 0.01},
        {"timestamp": pd.Timestamp("2021-01-08 10:00:00"), "trading_date": datetime.date(2021, 1, 8), "r_1m_valid": False, "volume": 999, "log_hl": 0.01},
    ]
    variables, validity = _variables_and_validity(rows)
    df = attach_calendar_fields(variables, validity)
    profiles = build_all_minute_profiles(df, years=(2020, 2021), n_boot=5, seed=0, extreme_quantile=0.99)
    minute = 300  # 10:00 UTC (invierno) = 05:00 NY
    row_2020 = profiles.by_year_tables[2020]["volume"]
    row_2021 = profiles.by_year_tables[2021]["volume"]
    assert row_2020[row_2020["minute_of_day"] == minute].iloc[0]["point"] == pytest.approx(100)
    assert row_2021[row_2021["minute_of_day"] == minute].iloc[0]["point"] == pytest.approx(999)
    assert row_2020[row_2020["minute_of_day"] == minute].iloc[0]["n"] == 1
    assert row_2021[row_2021["minute_of_day"] == minute].iloc[0]["n"] == 1


def test_partial_years_are_present_but_distinguishable_from_complete_years():
    from ohlcv_dataroad.ingest.tda06_intraday_calendar_profile import COMPLETE_YEARS, PARTIAL_YEARS
    assert 2019 in PARTIAL_YEARS and 2025 in PARTIAL_YEARS
    assert 2019 not in COMPLETE_YEARS and 2025 not in COMPLETE_YEARS
    assert set(COMPLETE_YEARS) == {2020, 2021, 2022, 2023, 2024}


# ---------------------------------------------------------------------------
# Frecuencia de extremos -- umbral relativo predeclarado
# ---------------------------------------------------------------------------

def test_extreme_flag_uses_relative_quantile_not_absolute_points():
    rows = []
    for i in range(100):
        date = datetime.date(2024, 1, 1) + datetime.timedelta(days=i)
        r = 0.0001 * (i + 1)  # creciente, para tener un percentil bien definido
        rows.append({"timestamp": pd.Timestamp(date) + pd.Timedelta(hours=10), "trading_date": date, "r_1m_valid": True, "r_1m": r, "abs_r_1m": r, "r2_1m": r**2, "zero_1m": 0.0})
    variables, validity = _variables_and_validity(rows)
    df = attach_calendar_fields(variables, validity)
    flag, threshold = compute_extreme_flag(df, 0.99)
    assert threshold == pytest.approx(df.loc[df["r_1m_valid"], "abs_r_1m"].quantile(0.99))
    assert flag.sum() == pytest.approx((df["abs_r_1m"] >= threshold).sum())


def test_extreme_flag_is_nan_where_return_invalid():
    rows = [
        {"timestamp": pd.Timestamp("2024-01-08 10:00:00"), "trading_date": datetime.date(2024, 1, 8), "r_1m_valid": True, "r_1m": 0.001, "abs_r_1m": 0.001, "r2_1m": 1e-6, "zero_1m": 0.0},
        {"timestamp": pd.Timestamp("2024-01-08 10:01:00"), "trading_date": datetime.date(2024, 1, 8), "r_1m_valid": False, "r_1m": np.nan, "abs_r_1m": np.nan, "r2_1m": np.nan, "zero_1m": np.nan},
    ]
    variables, validity = _variables_and_validity(rows)
    df = attach_calendar_fields(variables, validity)
    flag, _ = compute_extreme_flag(df, 0.99)
    assert np.isnan(flag.iloc[1])
    assert not np.isnan(flag.iloc[0])


# ---------------------------------------------------------------------------
# STOP-6 / s(m) / r_tilde
# ---------------------------------------------------------------------------

def test_decide_stop6_activates_when_profile_is_flat():
    evidence = {
        "abs_r_1m": {"relative_range": 0.01, "spearman_median": 0.9, "spearman_by_year": [0.9] * 5},
        "log_hl": {"relative_range": 0.01, "spearman_median": 0.9, "spearman_by_year": [0.9] * 5},
    }
    assert decide_stop6(evidence) is True


def test_decide_stop6_activates_when_profile_is_unstable_across_years():
    evidence = {
        "abs_r_1m": {"relative_range": 2.0, "spearman_median": 0.05, "spearman_by_year": [0.05] * 5},
        "log_hl": {"relative_range": 2.0, "spearman_median": 0.05, "spearman_by_year": [0.05] * 5},
    }
    assert decide_stop6(evidence) is True


def test_decide_stop6_does_not_activate_when_strong_and_stable():
    evidence = {
        "abs_r_1m": {"relative_range": 1.5, "spearman_median": 0.9, "spearman_by_year": [0.85, 0.88, 0.9, 0.92, 0.95]},
        "log_hl": {"relative_range": 1.2, "spearman_median": 0.85, "spearman_by_year": [0.8, 0.83, 0.85, 0.87, 0.9]},
    }
    assert decide_stop6(evidence) is False


def test_decide_stop6_is_holistic_not_a_binary_gate_on_the_median_alone():
    """Correccion de cierre: un año individual debil debe activar STOP-6 aunque la MEDIANA por año supere la referencia."""
    evidence = {
        # spearman_median=0.9 (muy por encima de la referencia 0.5) pero
        # un año individual (0.1) esta muy por debajo -- la version
        # anterior, que solo miraba la mediana, habria dicho "no
        # activado"; la version holistica debe activarlo.
        "abs_r_1m": {"relative_range": 2.0, "spearman_median": 0.9, "spearman_by_year": [0.95, 0.95, 0.95, 0.95, 0.1]},
        "log_hl": {"relative_range": 2.0, "spearman_median": 0.9, "spearman_by_year": [0.9, 0.9, 0.9, 0.9, 0.9]},
    }
    assert decide_stop6(evidence) is True


def test_decide_stop6_activates_when_no_complete_years_available():
    evidence = {
        "abs_r_1m": {"relative_range": 2.0, "spearman_median": float("nan"), "spearman_by_year": []},
        "log_hl": {"relative_range": 2.0, "spearman_median": float("nan"), "spearman_by_year": []},
    }
    assert decide_stop6(evidence) is True


def _flat_table(n, point, ci_lo, ci_hi, minutes=10):
    return pd.DataFrame({
        "minute_of_day": np.arange(minutes), "n": [n] * minutes,
        "point": [point] * minutes, "ci_lo": [ci_lo] * minutes, "ci_hi": [ci_hi] * minutes,
    })


def test_choose_s_m_proxy_prefers_the_less_noisy_estimator():
    # log_hl: banda de bootstrap estrecha relativa al nivel -> menos ruidoso.
    global_tables = {
        "log_hl": _flat_table(100, point=1.0, ci_lo=0.98, ci_hi=1.02),
        "abs_r_1m": _flat_table(100, point=1.0, ci_lo=0.5, ci_hi=1.5),
    }
    assert choose_s_m_proxy(global_tables) == "log_hl"


def test_choose_s_m_proxy_can_prefer_abs_r_1m_when_clearly_less_noisy():
    global_tables = {
        "log_hl": _flat_table(100, point=1.0, ci_lo=0.2, ci_hi=1.8),
        "abs_r_1m": _flat_table(100, point=1.0, ci_lo=0.98, ci_hi=1.02),
    }
    assert choose_s_m_proxy(global_tables) == "abs_r_1m"


def test_choose_s_m_proxy_is_not_the_same_metric_as_relative_range():
    """La eleccion de proxy (ruido del estimador) es una comparacion DISTINTA del rango relativo del perfil (fuerza del patron, usado en STOP-6)."""
    # Mismo relative_range (0) para ambas -- perfil perfectamente plano en las dos --
    # pero anchos de banda de bootstrap muy distintos: la eleccion debe basarse en el ruido, no en el rango.
    global_tables = {
        "log_hl": _flat_table(100, point=1.0, ci_lo=0.99, ci_hi=1.01),
        "abs_r_1m": _flat_table(100, point=1.0, ci_lo=0.3, ci_hi=1.7),
    }
    assert choose_s_m_proxy(global_tables) == "log_hl"


def test_s_m_normalized_to_mean_one_over_minutes_with_data():
    global_tables = {
        "log_hl": pd.DataFrame({
            "minute_of_day": [0, 1, 2, 3],
            "n": [10, 10, 10, 0],
            "point": [1.0, 2.0, 3.0, np.nan],
            "ci_lo": [0.9, 1.9, 2.9, np.nan], "ci_hi": [1.1, 2.1, 3.1, np.nan],
        })
    }
    s_m = build_s_m(global_tables, "log_hl")
    used = s_m[s_m["n"] > 0]
    assert used["s_m"].mean() == pytest.approx(1.0)
    assert np.isnan(s_m.loc[s_m["minute_of_day"] == 3, "s_m"].iloc[0])


def test_s_m_guards_against_zero_or_non_finite_point():
    global_tables = {
        "log_hl": pd.DataFrame({
            "minute_of_day": [0, 1, 2],
            "n": [10, 10, 10],
            "point": [0.0, -1.0, np.inf],
            "ci_lo": [0.0, -1.1, np.inf], "ci_hi": [0.0, -0.9, np.inf],
        })
    }
    s_m = build_s_m(global_tables, "log_hl")
    assert s_m["s_m"].isna().all()
    assert not (s_m["s_m"] == np.inf).any()


def test_r_tilde_equals_r_over_s_m_exactly():
    rows = [
        {"timestamp": pd.Timestamp("2024-01-08 10:00:00"), "trading_date": datetime.date(2024, 1, 8), "r_1m_valid": True, "r_1m": 0.002, "abs_r_1m": 0.002, "r2_1m": 4e-6, "zero_1m": 0.0},
    ]
    variables, validity = _variables_and_validity(rows)
    df = attach_calendar_fields(variables, validity)
    minute = int(df.loc[0, "minute_of_day"])
    s_m_table = pd.DataFrame({"minute_of_day": [minute], "n": [1], "point": [0.01], "s_m": [2.0], "proxy": ["log_hl"], "label": ["RETROSPECTIVO"]})
    r_tilde = build_r_tilde(df, s_m_table)
    assert r_tilde.loc[0, "r_tilde"] == pytest.approx(0.002 / 2.0)
    assert r_tilde.loc[0, "label"] == "RETROSPECTIVO"


def test_r_tilde_is_nan_where_return_invalid_or_s_m_missing():
    rows = [
        {"timestamp": pd.Timestamp("2024-01-08 10:00:00"), "trading_date": datetime.date(2024, 1, 8), "r_1m_valid": False, "r_1m": np.nan, "abs_r_1m": np.nan, "r2_1m": np.nan, "zero_1m": np.nan},
        {"timestamp": pd.Timestamp("2024-01-08 10:01:00"), "trading_date": datetime.date(2024, 1, 8), "r_1m_valid": True, "r_1m": 0.001, "abs_r_1m": 0.001, "r2_1m": 1e-6, "zero_1m": 0.0},
    ]
    variables, validity = _variables_and_validity(rows)
    df = attach_calendar_fields(variables, validity)
    # s(m) SOLO definido para el minuto de la segunda fila -- la primera queda sin s(m).
    minute_2 = int(df.loc[1, "minute_of_day"])
    s_m_table = pd.DataFrame({"minute_of_day": [minute_2], "n": [1], "point": [0.01], "s_m": [1.5], "proxy": ["log_hl"], "label": ["RETROSPECTIVO"]})
    r_tilde = build_r_tilde(df, s_m_table)
    assert np.isnan(r_tilde.loc[0, "r_tilde"])  # invalido -> NaN
    assert r_tilde.loc[1, "r_tilde"] == pytest.approx(0.001 / 1.5)


def test_run_tda06_only_persists_s_m_and_r_tilde_when_stop6_not_activated(tmp_path, monkeypatch):
    """No corre el pipeline completo (costoso); parchea evaluate_stop6 para forzar cada rama y comprueba la puerta."""
    import ohlcv_dataroad.ingest.tda06_intraday_calendar_profile as mod

    config = _make_config(tmp_path, research_files=["00_mnq_03_24.Last.txt"])
    rows = _synthetic_multi_day_rows(n_days=10, start=datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    _write_tda04_artifacts(config, variables, validity)

    monkeypatch.setattr(mod, "decide_stop6", lambda evidence: True)
    result_stop = mod.run_tda06_analysis(config)
    assert result_stop.stop6_activated is True
    assert result_stop.s_m_table is None and result_stop.r_tilde_table is None

    monkeypatch.setattr(mod, "decide_stop6", lambda evidence: False)
    result_go = mod.run_tda06_analysis(config)
    assert result_go.stop6_activated is False
    assert result_go.s_m_table is not None and result_go.r_tilde_table is not None
    assert (result_go.s_m_table["label"] == "RETROSPECTIVO").all()
    assert (result_go.r_tilde_table["label"] == "RETROSPECTIVO").all()


# ---------------------------------------------------------------------------
# Segmentacion reproducible + calibracion sobre null/permutacion
# ---------------------------------------------------------------------------

def test_detect_breakpoints_is_deterministic_and_respects_min_spacing():
    score = np.zeros(200)
    score[50] = 10.0
    score[52] = 9.0  # muy cerca de 50 -- debe descartarse por min_spacing
    score[150] = 8.0
    breaks_1 = detect_breakpoints(score, max_breakpoints=5, min_spacing=20)
    breaks_2 = detect_breakpoints(score, max_breakpoints=5, min_spacing=20)
    assert breaks_1 == breaks_2  # reproducible, sin aleatoriedad
    assert all(abs(breaks_1[i] - breaks_1[j]) >= 20 for i in range(len(breaks_1)) for j in range(len(breaks_1)) if i != j)


def test_detect_breakpoints_respects_max_breakpoints():
    rng = np.random.default_rng(0)
    score = rng.normal(size=500)
    breaks = detect_breakpoints(score, max_breakpoints=3, min_spacing=5)
    assert len(breaks) <= 3


def _random_global_tables_with_gap(n_minutes: int, gap: range, seed: int = 0) -> dict:
    """Tablas crudas sinteticas: ruido i.i.d. en 3 proxies, con un hueco estructural (n=0/point=NaN) compartido."""
    rng = np.random.default_rng(seed)
    tables = {}
    for col in ("abs_r_1m", "log_hl", "volume"):
        point = rng.normal(loc=10.0, scale=1.0, size=n_minutes)
        n = np.full(n_minutes, 100)
        point[list(gap)] = np.nan
        n[list(gap)] = 0
        tables[col] = pd.DataFrame({
            "minute_of_day": np.arange(n_minutes), "n": n, "point": point,
            "ci_lo": point - 0.1, "ci_hi": point + 0.1,
        })
    return tables


def test_calibration_on_permuted_null_finds_few_or_no_stable_breakpoints():
    """G2: sobre datos crudos sin estructura real, el detector no deberia fabricar quiebres 'estables' de forma sistematica."""
    tables = _random_global_tables_with_gap(1440, range(1021, 1081), seed=0)
    calib = calibrate_breakpoint_detector(
        tables, n_surrogates=20, smoothing_window=15, max_breakpoints=6, min_spacing=60, tolerance=15, min_years=3, seed=1
    )
    # Sobre ruido puro, la fraccion de quiebres candidatos que sobreviven
    # el filtro de estabilidad entre-surrogates debe ser pequeña -- no se
    # exige CERO (podria haber alguna coincidencia por azar), pero no
    # debe ser la mayoria.
    row = calib.iloc[0]
    if row["n_candidate_breaks_total"] > 0:
        assert row["n_stable_breaks_total"] / row["n_candidate_breaks_total"] < 0.5


def test_calibration_stability_criterion_does_not_degenerate_with_a_large_surrogate_pool():
    """Regresion: comparar cada surrogate contra TODOS los demas de una piscina grande (en vez de grupos del mismo tamaño que los años reales) hacia que el 100% de los candidatos del null parecieran 'estables' por puro azar combinatorio -- verificado que ocurria con la implementacion intermedia antes de agrupar por ``len(COMPLETE_YEARS)``."""
    tables = _random_global_tables_with_gap(1440, range(1021, 1081), seed=3)
    calib = calibrate_breakpoint_detector(
        tables, n_surrogates=100, smoothing_window=15, max_breakpoints=6, min_spacing=60,
        tolerance=15, min_years=3, seed=5,
    )
    row = calib.iloc[0]
    assert row["group_size"] == 5  # len(COMPLETE_YEARS), no el tamaño de la piscina de surrogates
    if row["n_candidate_breaks_total"] > 0:
        assert row["stable_fraction"] < 0.5


def test_default_n_surrogates_is_meaningfully_larger_than_five():
    """Cierre de TDA-06: 5 surrogates es insuficiente para caracterizar un null sobre 1440 minutos; el costo por surrogate es bajo, asi que no hay razon para quedarse ahi."""
    assert DEFAULT_N_SURROGATES >= 50


def test_calibrate_breakpoint_detector_takes_raw_tables_not_a_presmoothed_array():
    """Correccion de cierre: la firma recibe las tablas CRUDAS (sin suavizar); el suavizado ocurre DENTRO, una vez por surrogate -- nunca se le pasa un score ya suavizado."""
    tables = _random_global_tables_with_gap(300, range(100, 130), seed=2)
    calib = calibrate_breakpoint_detector(
        tables, n_surrogates=10, smoothing_window=5, max_breakpoints=3, min_spacing=20, tolerance=5, min_years=3, seed=0
    )
    assert set(calib.columns) >= {"n_surrogates", "n_candidate_breaks_total", "n_stable_breaks_total"}
    assert calib.iloc[0]["n_surrogates"] == 10


def test_permute_preserving_nan_mask_keeps_nan_positions_fixed_and_shuffles_the_rest():
    rng = np.random.default_rng(0)
    point = np.array([1.0, 2.0, np.nan, np.nan, 3.0, 4.0, 5.0])
    surrogate = _permute_preserving_nan_mask(point, rng)
    assert np.isnan(surrogate[2]) and np.isnan(surrogate[3])
    assert not np.isnan(surrogate[[0, 1, 4, 5, 6]]).any()
    assert sorted(surrogate[~np.isnan(surrogate)]) == sorted(point[~np.isnan(point)])


# ---------------------------------------------------------------------------
# Suavizado: NO fabricar valores en minutos estructuralmente cerrados
# ---------------------------------------------------------------------------

def test_centered_rolling_median_keeps_nan_in_a_closed_block_after_smoothing():
    """Correccion de cierre: un bloque n=0 permanece NaN despues del suavizado, incluso si esta rodeado de datos."""
    x = pd.Series([10.0] * 30 + [np.nan] * 20 + [10.0] * 30)
    smoothed = _centered_rolling_median(x, window=15)
    closed_idx = np.arange(30, 50)
    assert np.isnan(smoothed[closed_idx]).all()


def test_centered_rolling_median_still_smooths_open_minutes_near_the_edge_of_a_gap():
    """Los minutos ABIERTOS cerca del borde de un hueco si deben recibir un valor suavizado (no se rompe el suavizado legitimo)."""
    x = pd.Series([10.0] * 30 + [np.nan] * 20 + [10.0] * 30)
    smoothed = _centered_rolling_median(x, window=15)
    assert not np.isnan(smoothed[0]) and not np.isnan(smoothed[-1])


def test_detect_breakpoints_never_returns_a_structurally_closed_minute():
    tables = {
        "abs_r_1m": pd.DataFrame({
            "minute_of_day": np.arange(400),
            "n": [100] * 200 + [0] * 50 + [100] * 150,
            "point": [10.0] * 200 + [np.nan] * 50 + [10.0] * 100 + [50.0] + [10.0] * 49,
            "ci_lo": [9.0] * 400, "ci_hi": [11.0] * 400,
        })
    }
    score = compute_composite_activity_score(tables, smoothing_window=15, proxy_cols=("abs_r_1m",))
    breaks = detect_breakpoints(score, max_breakpoints=5, min_spacing=20)
    closed_range = set(range(200, 250))
    assert not (set(breaks) & closed_range)


def test_closed_window_surrounded_by_data_does_not_create_artificial_breakpoint_at_its_edges():
    """Reproduce el bug real: un hueco estructural rodeado de un nivel CONSTANTE no debe generar ningun quiebre fabricado en sus bordes."""
    n = 400
    point = np.full(n, 10.0)
    gap = range(200, 250)
    point[list(gap)] = np.nan
    tables = {
        "abs_r_1m": pd.DataFrame({
            "minute_of_day": np.arange(n), "n": np.where(np.isnan(point), 0, 100), "point": point,
            "ci_lo": point - 0.1, "ci_hi": point + 0.1,
        })
    }
    score = compute_composite_activity_score(tables, smoothing_window=15, proxy_cols=("abs_r_1m",))
    breaks = detect_breakpoints(score, max_breakpoints=6, min_spacing=20)
    # Nivel constante a ambos lados del hueco -> ningun quiebre real que
    # detectar; en particular, ninguno cerca de los bordes del hueco
    # (199-200 y 249-250), que es exactamente donde el bug fabricaba uno.
    assert not any(190 <= b <= 260 for b in breaks)


def test_build_segmentation_proposal_never_proposes_a_cut_inside_or_at_the_edge_of_a_structural_gap():
    """Integracion: el pipeline completo de segmentacion (global + por año) no debe proponer un corte dentro/al borde de un hueco estructural compartido por todos los años."""
    n = 400
    gap = range(200, 250)

    def _tables_with_jump(seed):
        rng = np.random.default_rng(seed)
        point = 10.0 + rng.normal(scale=0.05, size=n)
        point[350] += 20.0  # una diferencia real, lejos del hueco
        point[list(gap)] = np.nan
        return {
            col: pd.DataFrame({
                "minute_of_day": np.arange(n), "n": np.where(np.isnan(point), 0, 100), "point": point.copy(),
                "ci_lo": point - 0.1, "ci_hi": point + 0.1,
            })
            for col in ("abs_r_1m", "log_hl", "volume")
        }

    global_tables = _tables_with_jump(seed=0)
    by_year_tables = {2020 + i: _tables_with_jump(seed=i + 1) for i in range(5)}

    stability, global_score = build_segmentation_proposal(
        global_tables, by_year_tables, smoothing_window=15, max_breakpoints=6,
        min_spacing=20, tolerance=10, min_years=3,
    )
    assert np.isnan(global_score[list(gap)]).all()
    proposed = set(stability["minute_of_day"]) if len(stability) else set()
    assert not (proposed & set(range(190, 260)))


def test_composite_score_and_segmentation_are_reproducible():
    rows = _synthetic_multi_day_rows(n_days=15, start=datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    df = attach_calendar_fields(variables, validity)
    profiles = build_all_minute_profiles(df, years=(2024,), n_boot=5, seed=0, extreme_quantile=0.99)
    score_1 = compute_composite_activity_score(profiles.global_tables, smoothing_window=15)
    score_2 = compute_composite_activity_score(profiles.global_tables, smoothing_window=15)
    np.testing.assert_array_equal(score_1[~np.isnan(score_1)], score_2[~np.isnan(score_2)])


# ---------------------------------------------------------------------------
# Perfil por dia de semana (TH15)
# ---------------------------------------------------------------------------

def test_weekday_profile_has_seven_weekdays_per_variable():
    rows = _synthetic_multi_day_rows(n_days=10, start=datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    df = attach_calendar_fields(variables, validity)
    weekday_profile = build_weekday_profile(df, n_boot=5, seed=0)
    for var in weekday_profile["variable"].unique():
        assert set(weekday_profile.loc[weekday_profile["variable"] == var, "weekday"]) == set(range(7))


# ---------------------------------------------------------------------------
# Proteccion del hold-out + conservacion
# ---------------------------------------------------------------------------

def _synthetic_multi_day_rows(n_days: int, start: datetime.date) -> list[dict]:
    rows = []
    d = start
    added = 0
    day_i = 0
    while added < n_days:
        date = start + datetime.timedelta(days=day_i)
        day_i += 1
        if date.weekday() >= 5:
            continue
        for h, m in [(10, 0), (14, 0)]:
            ts = pd.Timestamp(date) + pd.Timedelta(hours=h, minutes=m)
            rows.append({
                "timestamp": ts, "trading_date": date, "r_1m_valid": True,
                "r_1m": 0.0005, "abs_r_1m": 0.0005, "r2_1m": 0.0005 ** 2, "zero_1m": 0.0,
                "volume": 20, "log_hl": 0.008,
            })
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
    }
    config_path = config_dir / "mnq_snapshot.yaml"
    config_path.write_text(yaml.safe_dump(config_yaml), encoding="utf-8")
    return load_config(config_path)


def _write_tda04_artifacts(config, variables: pd.DataFrame, validity: pd.DataFrame) -> None:
    config.interim_dir.mkdir(parents=True, exist_ok=True)
    variables.to_parquet(config.tda04_variables_parquet_path, index=False)
    validity.to_parquet(config.tda04_validity_mask_parquet_path, index=False)


def test_run_tda06_raises_when_research_and_holdout_overlap(tmp_path):
    config = _make_config(
        tmp_path, research_files=["00_mnq_03_24.Last.txt"],
        holdout_files=["00_mnq_03_24.Last.txt", "no_existe.Last.txt"],
    )
    with pytest.raises(HoldoutIsolationError, match="disjuntos"):
        run_tda06_analysis(config)


def test_run_tda06_raises_when_row_reaches_holdout_boundary(tmp_path):
    config = _make_config(
        tmp_path, research_files=["00_mnq_03_24.Last.txt"],
        holdout_files=["no_existe.Last.txt"],
        boundary_utc="2024-01-08 15:01:00",
    )
    rows = [
        {"timestamp": pd.Timestamp("2024-01-08 15:00:00"), "trading_date": datetime.date(2024, 1, 8), "r_1m_valid": False, "source_file": "00_mnq_03_24.Last.txt"},
        {"timestamp": pd.Timestamp("2024-01-08 15:01:00"), "trading_date": datetime.date(2024, 1, 8), "r_1m_valid": True, "r_1m": 0.001, "abs_r_1m": 0.001, "r2_1m": 1e-6, "zero_1m": 0.0, "source_file": "00_mnq_03_24.Last.txt"},
    ]
    variables, validity = _variables_and_validity(rows)
    _write_tda04_artifacts(config, variables, validity)
    with pytest.raises(HoldoutIsolationError, match="frontera de hold-out"):
        run_tda06_analysis(config)
    assert not (config.raw_dir / "no_existe.Last.txt").exists()


def test_run_tda06_never_opens_any_raw_or_holdout_file(tmp_path):
    config = _make_config(
        tmp_path, research_files=["00_mnq_03_24.Last.txt"],
        holdout_files=["archivo_que_no_existe.Last.txt"],
    )
    rows = _synthetic_multi_day_rows(n_days=8, start=datetime.date(2024, 1, 8))
    for r in rows:
        r["source_file"] = "00_mnq_03_24.Last.txt"
    variables, validity = _variables_and_validity(rows)
    _write_tda04_artifacts(config, variables, validity)
    assert not (config.raw_dir / "00_mnq_03_24.Last.txt").exists()
    result = run_tda06_analysis(config)
    assert len(result.df) == len(rows)


def test_row_count_and_traceability_conserved(tmp_path):
    config = _make_config(tmp_path, research_files=["00_mnq_03_24.Last.txt"])
    rows = _synthetic_multi_day_rows(n_days=8, start=datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    _write_tda04_artifacts(config, variables, validity)
    result = run_tda06_analysis(config)
    assert len(result.df) == len(rows)
    assert set(result.df.columns) >= {"timestamp", "source_file", "contract", "trading_date", "minute_of_day", "weekday", "year_ny"}
