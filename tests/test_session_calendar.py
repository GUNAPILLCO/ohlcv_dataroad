"""Tests de ``session_calendar.py`` -- calendario nativo de CME_Equity y grilla esperada.

Se prueba contra el calendario REAL de ``pandas_market_calendars`` (no un
doble sintetico): el objetivo de este modulo es justamente verificar el
comportamiento de esa libreria externa contra hechos conocidos (horario de
mantenimiento, DST, feriados, la correccion forense de 2025-01-09), asi
que sustituirla por un doble no probaria nada util.
"""
from __future__ import annotations

import datetime

import pandas as pd
import pytest

from ohlcv_dataroad.ingest.session_calendar import (
    FORENSIC_SCHEDULE_OVERRIDES,
    NY_TZ,
    SECONDARY_BREAK_ABOLISHED_TRADING_DATE,
    SECONDARY_BREAK_LAST_TRADING_DATE,
    build_session_schedule,
    expected_bar_grid,
    expected_bar_grid_frame,
    full_holidays_in_range,
)


# ---------------------------------------------------------------------------
# build_session_schedule / trading_date_for
# ---------------------------------------------------------------------------

def test_trading_date_uses_the_session_close_date_not_the_open_date():
    """Bajo la convencion de CIERRE de barra, una sesion que abre domingo a
    la noche y cierra lunes a las 17:00 NY se etiqueta con la fecha del
    LUNES (fecha de cierre), no la del domingo (fecha de apertura)."""
    schedule = build_session_schedule("2024-01-01", "2024-01-10")
    ts = pd.Series(
        pd.to_datetime(
            [
                "2024-01-02 22:00:00",  # martes 17:00 NY (EST) -- cierre normal
                "2024-01-02 23:01:00",  # martes 18:01 NY -- primer bar del miercoles
            ]
        ).tz_localize("UTC")
    )
    dates = schedule.trading_date_for(ts)
    assert dates.iloc[0] == datetime.date(2024, 1, 2)
    assert dates.iloc[1] == datetime.date(2024, 1, 3)


def test_weekend_timestamp_has_no_trading_date():
    schedule = build_session_schedule("2024-01-01", "2024-01-10")
    ts = pd.Series(pd.to_datetime(["2024-01-06 12:00:00"]).tz_localize("UTC"))  # sabado
    dates = schedule.trading_date_for(ts)
    assert pd.isna(dates.iloc[0])


def test_early_close_day_is_flagged():
    """El dia de Accion de Gracias 2023 (2023-11-23) cierra temprano segun CME_Equity."""
    schedule = build_session_schedule("2023-11-15", "2023-11-30")
    assert schedule.is_early_close(datetime.date(2023, 11, 23)) is True
    # Un dia cualquiera sin cierre anticipado no debe estar marcado.
    assert schedule.is_early_close(datetime.date(2023, 11, 20)) is False


# ---------------------------------------------------------------------------
# expected_bar_grid / expected_bar_grid_frame -- convencion cierre de barra
# ---------------------------------------------------------------------------

def test_expected_grid_first_bar_is_open_plus_one_minute():
    schedule = build_session_schedule("2024-01-02", "2024-01-02")
    open_ts = schedule.table.loc[datetime.date(2024, 1, 2), "market_open"]
    close_ts = schedule.table.loc[datetime.date(2024, 1, 2), "market_close"]
    grid = expected_bar_grid(schedule, clip_start=open_ts, clip_end=close_ts)
    assert grid.min() == open_ts + pd.Timedelta(minutes=1)


def test_expected_grid_last_bar_is_market_close():
    schedule = build_session_schedule("2024-01-02", "2024-01-02")
    open_ts = schedule.table.loc[datetime.date(2024, 1, 2), "market_open"]
    close_ts = schedule.table.loc[datetime.date(2024, 1, 2), "market_close"]
    grid = expected_bar_grid(schedule, clip_start=open_ts, clip_end=close_ts)
    assert grid.max() == close_ts


def test_expected_grid_can_be_clipped():
    schedule = build_session_schedule("2024-01-02", "2024-01-03")
    full_grid = expected_bar_grid(schedule)
    clip_start = full_grid[10]
    clipped = expected_bar_grid(schedule, clip_start=clip_start)
    assert clipped.min() == clip_start
    assert len(clipped) == len(full_grid) - 10


def test_expected_grid_frame_attaches_correct_trading_date():
    schedule = build_session_schedule("2024-01-02", "2024-01-03")
    frame = expected_bar_grid_frame(schedule)
    row = frame[frame["trading_date"] == datetime.date(2024, 1, 2)]
    assert len(row) > 0
    assert (row["expected_ts_utc"] <= schedule.table.loc[datetime.date(2024, 1, 2), "market_close"]).all()


# ---------------------------------------------------------------------------
# DST -- el mismo umbral de mercado debe seguir siendo 17:00 NY a ambos lados
# ---------------------------------------------------------------------------

def test_session_close_stays_at_seventeen_ny_across_spring_dst_transition():
    """Confirma con el calendario real -- no solo con datos observados -- que
    el cierre de sesion sigue siendo 17:00 NY antes y despues de la
    transicion de marzo 2024 (2024-03-10), aunque el offset UTC cambie."""
    schedule = build_session_schedule("2024-03-06", "2024-03-13")
    before = schedule.table.loc[datetime.date(2024, 3, 8), "market_close"].tz_convert(NY_TZ)
    after = schedule.table.loc[datetime.date(2024, 3, 11), "market_close"].tz_convert(NY_TZ)
    assert before.time() == after.time() == datetime.time(17, 0, 0)
    assert before.utcoffset() != after.utcoffset()


# ---------------------------------------------------------------------------
# full_holidays_in_range
# ---------------------------------------------------------------------------

def test_christmas_is_a_full_holiday():
    holidays = full_holidays_in_range("2023-12-01", "2023-12-31")
    assert datetime.date(2023, 12, 25) in holidays


def test_secondary_break_minutes_excluded_from_grid_on_or_before_cutoff():
    """2021-06-25 (viernes) es la ultima fecha de negociacion con el break
    vigente segun la fuente documental citada (CME SER/Globex Notice,
    junio 2021, efecto 2021-06-28): sus minutos de break NO deben aparecer
    en la grilla esperada."""
    schedule = build_session_schedule("2021-06-24", "2021-06-25")
    assert SECONDARY_BREAK_LAST_TRADING_DATE == datetime.date(2021, 6, 25)
    row = schedule.table.loc[datetime.date(2021, 6, 25)]
    break_start, break_end = row["break_start"], row["break_end"]
    assert pd.notna(break_start) and pd.notna(break_end)

    frame = expected_bar_grid_frame(schedule)
    day_grid = frame[frame["trading_date"] == datetime.date(2021, 6, 25)]
    missing_break_minutes = pd.date_range(
        break_start + pd.Timedelta(minutes=1), break_end, freq="1min"
    )
    assert not day_grid["expected_ts_utc"].isin(missing_break_minutes).any()
    # El resto de la sesion (antes y despues del break) SI debe seguir presente.
    assert (day_grid["expected_ts_utc"] == break_start).any()
    assert (day_grid["expected_ts_utc"] == break_end + pd.Timedelta(minutes=1)).any()


def test_secondary_break_minutes_included_in_grid_after_cutoff():
    """2021-06-28 (lunes) es la primera fecha de negociacion SIN el break,
    segun la misma fuente documental: sus minutos de break DEBEN aparecer
    en la grilla esperada (el mercado ya no se detiene ahi)."""
    schedule = build_session_schedule("2021-06-28", "2021-06-29")
    assert SECONDARY_BREAK_ABOLISHED_TRADING_DATE == datetime.date(2021, 6, 28)
    row = schedule.table.loc[datetime.date(2021, 6, 28)]
    break_start, break_end = row["break_start"], row["break_end"]
    assert pd.notna(break_start) and pd.notna(break_end)  # la libreria lo sigue declarando

    frame = expected_bar_grid_frame(schedule)
    day_grid = frame[frame["trading_date"] == datetime.date(2021, 6, 28)]
    break_minutes = pd.date_range(break_start + pd.Timedelta(minutes=1), break_end, freq="1min")
    # Todos los minutos del break DEBEN estar presentes en la grilla de ese dia
    # (subconjunto, no igualdad -- day_grid tiene ademas el resto de la sesion).
    assert pd.Series(break_minutes).isin(day_grid["expected_ts_utc"]).all()


def test_expected_grid_excludes_exactly_fifteen_minutes_per_pre_cutoff_session():
    schedule = build_session_schedule("2020-06-01", "2020-06-01")
    open_ts = schedule.table.loc[datetime.date(2020, 6, 1), "market_open"]
    close_ts = schedule.table.loc[datetime.date(2020, 6, 1), "market_close"]
    full_len = int((close_ts - open_ts).total_seconds() / 60)
    grid = expected_bar_grid(schedule, clip_start=open_ts, clip_end=close_ts)
    assert full_len - len(grid) == 15


def test_forensically_overridden_date_is_excluded_from_full_holidays():
    """2025-01-09 (dia de duelo nacional) NO debe aparecer como feriado
    completo: la evidencia forense (datos reales) muestra un cierre
    anticipado real, no una ausencia total de sesion -- ver
    FORENSIC_SCHEDULE_OVERRIDES."""
    holidays = full_holidays_in_range("2025-01-01", "2025-01-15")
    assert datetime.date(2025, 1, 9) not in holidays


# ---------------------------------------------------------------------------
# Correccion forense: 2025-01-09
# ---------------------------------------------------------------------------

def test_forensic_override_produces_a_normal_open_but_early_close():
    schedule = build_session_schedule("2025-01-05", "2025-01-12")
    row = schedule.table.loc[datetime.date(2025, 1, 9)]
    assert row["market_open"] == pd.Timestamp("2025-01-08 23:00:00", tz="UTC")
    assert row["market_close"] == pd.Timestamp("2025-01-09 14:30:00", tz="UTC")
    assert schedule.is_early_close(datetime.date(2025, 1, 9)) is True


def test_forensic_override_out_of_declared_range_is_not_applied():
    """El override solo debe aplicarse si la fecha cae dentro del rango pedido."""
    schedule = build_session_schedule("2020-01-01", "2020-01-10")
    assert datetime.date(2025, 1, 9) not in schedule.table.index


def test_only_one_adhoc_mourning_holiday_falls_in_research_range():
    """Verifica el hecho citado en la documentacion de FORENSIC_SCHEDULE_OVERRIDES:
    2025-01-09 es la UNICA fecha de USNationalDaysofMourning dentro del
    rango de investigacion (2019-12-23 a 2025-06-22)."""
    import pandas_market_calendars as mcal

    cal = mcal.get_calendar("CME_Equity")
    adhoc_dates = {pd.Timestamp(d).date() for d in cal.adhoc_holidays}
    in_range = {d for d in adhoc_dates if datetime.date(2019, 12, 23) <= d <= datetime.date(2025, 6, 22)}
    assert in_range == {datetime.date(2025, 1, 9)}
    assert set(pd.Timestamp(d).date() for d in FORENSIC_SCHEDULE_OVERRIDES) == in_range
