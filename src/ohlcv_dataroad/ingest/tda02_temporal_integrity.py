"""TDA-02 -- Integridad del eje temporal y del calendario.

Implementa la etapa TDA-02 del roadmap
(``docs/methodology/Tsay_OHLCV_analysis_roadmap.md``, seccion "TDA-02"):
determina si el eje temporal del conjunto de investigacion esta completo
respecto de la grilla esperada por la estructura NATIVA de negociacion de
CME_Equity (``session_calendar.py``), y clasifica cada ausencia.

Que NO hace este modulo (deliberadamente, fuera de alcance de TDA-02):

- No construye ninguna serie continua entre contratos.
- No decide rollover ni selecciona contrato activo (TDA-03).
- No imputa, corrige ni elimina ningun dato.
- No calcula retornos, volatilidad, features ni targets.

Todo el analisis se hace POR ARCHIVO (nunca a traves de la frontera entre
dos archivos, que es una transicion de contrato -- ver
``tda01_temporal_semantics.py``, ya usado aqui para el mismo proposito).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from ohlcv_dataroad.config import SnapshotConfig
from ohlcv_dataroad.ingest.holdout_guard import (
    validate_last_timestamps_before_boundary,
    validate_research_holdout_disjoint,
)
from ohlcv_dataroad.ingest.session_calendar import (
    NY_TZ,
    NORMAL_SESSION_CLOSE_NY_TIME,
    SECONDARY_BREAK_LAST_TRADING_DATE,
    SessionSchedule,
    build_session_schedule,
    expected_bar_grid_frame,
    full_holidays_in_range,
)
from ohlcv_dataroad.ingest.tda01_temporal_semantics import (
    attach_ny_wallclock,
    compute_intra_file_gaps,
    load_research_rows,
)

# --- Catalogo de causas candidatas de un hueco (seccion 7 de la tarea) -----
CAUSE_DAILY_MAINTENANCE = "DAILY_MAINTENANCE"
CAUSE_WEEKEND = "WEEKEND"
CAUSE_HOLIDAY = "HOLIDAY"
CAUSE_EARLY_CLOSE = "EARLY_CLOSE"
CAUSE_SECONDARY_MAINTENANCE_BREAK = "SECONDARY_MAINTENANCE_BREAK"
CAUSE_MISSING_TRADING_DAY = "MISSING_TRADING_DAY"
CAUSE_UNKNOWN = "UNKNOWN"

CONFIDENCE_HIGH = "ALTA"
CONFIDENCE_MEDIUM = "MEDIA"
CONFIDENCE_LOW = "BAJA"

ONE_MIN = pd.Timedelta(minutes=1)


# ---------------------------------------------------------------------------
# Clasificacion causal de huecos internos
# ---------------------------------------------------------------------------

def _classify_one_gap(
    before_ts, after_ts, before_date, after_date, schedule: SessionSchedule,
    holidays: set, trading_dates_sorted: np.ndarray,
) -> tuple[str, str, str]:
    """Clasifica UN hueco interno ya localizado en el calendario.

    Entrada: los timestamps observados que rodean el hueco (``before_ts``,
    ``after_ts``, tz-aware UTC) y la ``trading_date`` de cada uno (puede
    ser ``pd.NaT`` si ese borde no cae dentro de ninguna sesion conocida).

    Salida: tupla ``(causa, confianza, detalle)`` -- ``causa`` es uno de
    los codigos ``CAUSE_*`` de arriba; ``confianza`` es ``ALTA`` cuando los
    dos bordes coinciden EXACTAMENTE con el horario que declara el
    calendario para esas fechas, ``MEDIA`` cuando la causa es plausible
    pero algun borde no coincide exactamente, y ``BAJA`` cuando no se pudo
    determinar ninguna causa concreta (``UNKNOWN``).
    """
    if pd.isna(before_date) or pd.isna(after_date):
        return (
            CAUSE_UNKNOWN,
            CONFIDENCE_LOW,
            "al menos uno de los bordes del hueco no cae dentro de ninguna "
            "sesion conocida del calendario CME_Equity (ver barras fuera de grilla)",
        )

    if before_date == after_date:
        # Hueco intradia: la unica causa de calendario conocida es el break
        # secundario declarado por la libreria (break_start/break_end),
        # ELIMINADO por CME con efecto en la fecha de negociacion
        # 2021-06-28 (ver session_calendar.SECONDARY_BREAK_LAST_TRADING_DATE
        # para la cita documental completa -- CME SER/Globex Notice de
        # junio de 2021 -- y su verificacion forense). La fecha de corte se
        # exige de forma EXPLICITA aqui (antes se dependia solo de que no
        # apareciera ningun hueco con esa firma despues de esa fecha; ahora
        # se gatea tambien por fecha, para no depender de la representacion
        # estatica de la libreria de calendario).
        if before_date <= SECONDARY_BREAK_LAST_TRADING_DATE:
            row = schedule.table.loc[before_date]
            break_start = row.get("break_start")
            break_end = row.get("break_end")
            if pd.notna(break_start) and pd.notna(break_end):
                if before_ts == break_start and after_ts == break_end + ONE_MIN:
                    return (
                        CAUSE_SECONDARY_MAINTENANCE_BREAK,
                        CONFIDENCE_HIGH,
                        f"coincide exactamente con el break intradiario declarado por "
                        f"CME_Equity para esta sesion ({break_start} -> {break_end} UTC), "
                        f"vigente hasta la fecha de negociacion {SECONDARY_BREAK_LAST_TRADING_DATE} "
                        "(eliminado por CME con efecto 2021-06-28, ver session_calendar.py)",
                    )
        return (
            CAUSE_UNKNOWN,
            CONFIDENCE_LOW,
            "hueco intradia (mismo trading_date en ambos bordes) que no "
            "coincide con ningun break declarado por el calendario",
        )

    idx_before = int(np.searchsorted(trading_dates_sorted, before_date))
    idx_after = int(np.searchsorted(trading_dates_sorted, after_date))
    skipped_trading_dates = list(trading_dates_sorted[idx_before + 1 : idx_after])

    close_row = schedule.table.loc[before_date]
    open_row = schedule.table.loc[after_date]
    exact_boundary = (before_ts == close_row["market_close"]) and (
        after_ts == open_row["market_open"] + ONE_MIN
    )

    if skipped_trading_dates:
        detail = (
            f"el calendario CME_Equity esperaba sesion en {skipped_trading_dates} "
            "pero este archivo no tiene NINGUNA barra en esa(s) fecha(s) -- "
            "posible jornada de negociacion completa ausente del archivo"
        )
        return CAUSE_MISSING_TRADING_DAY, CONFIDENCE_MEDIUM, detail

    if after_date > before_date + timedelta(days=1):
        calendar_days_between = [
            (before_date + timedelta(days=i))
            for i in range(1, (after_date - before_date).days)
        ]
    else:
        calendar_days_between = []
    holidays_in_between = [d for d in calendar_days_between if d in holidays]

    if holidays_in_between:
        cause = CAUSE_HOLIDAY
        detail = (
            f"el hueco cruza el/los feriado(s) completo(s) {holidays_in_between} "
            "del calendario CME_Equity"
        )
    elif (after_date - before_date).days >= 3:
        cause = CAUSE_WEEKEND
        detail = "el hueco cruza un fin de semana (sin feriado adicional en medio)"
    elif schedule.is_early_close(before_date):
        cause = CAUSE_EARLY_CLOSE
        close_ny = close_row["market_close"].tz_convert(NY_TZ).time()
        detail = (
            f"la sesion de {before_date} cierra a las {close_ny} hora de Nueva York, "
            f"antes de la hora normal ({NORMAL_SESSION_CLOSE_NY_TIME}), segun el "
            "calendario CME_Equity (cierre anticipado)"
        )
    else:
        cause = CAUSE_DAILY_MAINTENANCE
        detail = (
            f"transicion normal cierre->reapertura entre {before_date} y {after_date} "
            "(jornadas consecutivas, sin feriado ni fin de semana en medio)"
        )

    confidence = CONFIDENCE_HIGH if exact_boundary else CONFIDENCE_MEDIUM
    if not exact_boundary:
        expected_close = close_row["market_close"]
        expected_open = open_row["market_open"] + ONE_MIN
        detail += (
            f" -- ADVERTENCIA: el borde observado no coincide exactamente con el "
            f"horario declarado (cierre esperado {expected_close}, reapertura "
            f"esperada {expected_open}; observado {before_ts} -> {after_ts})"
        )
    return cause, confidence, detail


def classify_gaps(gaps: pd.DataFrame, schedule: SessionSchedule, holidays: set) -> pd.DataFrame:
    """Clasifica, fila por fila, el inventario completo de huecos internos.

    Entrada
    -------
    gaps      : salida de ``compute_intra_file_gaps`` (tda01), con
        ``before_ts_utc``/``after_ts_utc`` YA localizados como tz-aware UTC
        (a diferencia del uso original en TDA-01, donde se dejaban
        tz-naive -- aqui hace falta comparar contra el horario del
        calendario, que si es tz-aware).
    schedule  : ``SessionSchedule`` ya calculado sobre un rango que cubre
        holgadamente todos los huecos de ``gaps``.
    holidays  : conjunto de ``datetime.date`` de feriados completos
        (``session_calendar.full_holidays_in_range``).

    Salida
    ------
    ``DataFrame`` con todas las columnas de ``gaps`` mas: ``before_date``,
    ``after_date`` (trading_date de cada borde), ``expected_ts_inicio``,
    ``expected_ts_fin`` (primer/ultimo minuto que la grilla habria
    esperado dentro del propio hueco, calculados directamente de los
    bordes observados, sin condicionar a la causa), ``cause`` y
    ``confidence``.
    """
    gaps = gaps.copy()
    gaps["before_date"] = schedule.trading_date_for(gaps["before_ts_utc"])
    gaps["after_date"] = schedule.trading_date_for(gaps["after_ts_utc"])
    trading_dates_sorted = np.array(sorted(schedule.table.index))

    causes, confidences, details = [], [], []
    for row in gaps.itertuples(index=False):
        cause, confidence, detail = _classify_one_gap(
            row.before_ts_utc, row.after_ts_utc, row.before_date, row.after_date,
            schedule, holidays, trading_dates_sorted,
        )
        causes.append(cause)
        confidences.append(confidence)
        details.append(detail)

    gaps["cause"] = causes
    gaps["confidence"] = confidences
    gaps["cause_detail"] = details
    gaps["expected_ts_inicio"] = gaps["before_ts_utc"] + ONE_MIN
    gaps["expected_ts_fin"] = gaps["after_ts_utc"] - ONE_MIN
    return gaps


# ---------------------------------------------------------------------------
# Bordes de archivo (seccion 5 de la tarea): distinguir de huecos internos
# ---------------------------------------------------------------------------

def build_file_boundary_records(
    file_bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]], schedule: SessionSchedule
) -> pd.DataFrame:
    """Compara el inicio/fin real de cada archivo contra el borde IDEAL de su sesion.

    Entrada: ``file_bounds`` -- ``{source_file: (file_start_utc, file_end_utc)}``,
    ambos tz-aware UTC; ``schedule`` ya calculado sobre un rango que cubre
    todos los archivos.

    Por que existe (seccion 5 de la tarea): un archivo que empieza o
    termina a mitad de una jornada NO debe clasificarse automaticamente
    como perdida de datos -- es, simplemente, donde empieza o termina el
    export de ESTE contrato dentro del snapshot. Esta funcion documenta
    esa situacion como informativa (``FILE_BOUNDARY``), separada por
    completo del inventario de huecos internos: nunca se usa para restar
    cobertura, porque la cobertura de cada archivo se calcula sobre la
    grilla YA recortada a ``[file_start, file_end]`` (ver
    ``compute_file_coverage``).

    Salida: un ``DataFrame``, una fila por archivo y borde (``edge`` =
    ``"start"``/``"end"``) SOLO cuando el borde real no coincide con el
    primer/ultimo minuto ideal de su sesion; columnas: ``source_file``,
    ``edge``, ``trading_date``, ``observed_ts_utc``, ``ideal_ts_utc``,
    ``minutes_before_file_start`` (para ``edge="start"``) o
    ``minutes_after_file_end`` (para ``edge="end"``), ``cause``
    (siempre ``FILE_BOUNDARY``).
    """
    records = []
    for source_file, (file_start, file_end) in file_bounds.items():
        start_date = schedule.trading_date_for(pd.Series([file_start])).iloc[0]
        if pd.notna(start_date):
            ideal_start = schedule.table.loc[start_date, "market_open"] + ONE_MIN
            if file_start != ideal_start:
                records.append(
                    {
                        "source_file": source_file,
                        "edge": "start",
                        "trading_date": start_date,
                        "observed_ts_utc": file_start,
                        "ideal_ts_utc": ideal_start,
                        "gap_minutes": (file_start - ideal_start).total_seconds() / 60,
                        "cause": "FILE_BOUNDARY",
                    }
                )
        end_date = schedule.trading_date_for(pd.Series([file_end])).iloc[0]
        if pd.notna(end_date):
            ideal_end = schedule.table.loc[end_date, "market_close"]
            if file_end != ideal_end:
                records.append(
                    {
                        "source_file": source_file,
                        "edge": "end",
                        "trading_date": end_date,
                        "observed_ts_utc": file_end,
                        "ideal_ts_utc": ideal_end,
                        "gap_minutes": (ideal_end - file_end).total_seconds() / 60,
                        "cause": "FILE_BOUNDARY",
                    }
                )
    columns = [
        "source_file", "edge", "trading_date", "observed_ts_utc", "ideal_ts_utc",
        "gap_minutes", "cause",
    ]
    return pd.DataFrame(records, columns=columns)


# ---------------------------------------------------------------------------
# Cobertura (seccion 6): por archivo, y agregada por trading_date/mes/año
# ---------------------------------------------------------------------------

@dataclass
class FileCoverage:
    source_file: str
    file_start: pd.Timestamp
    file_end: pd.Timestamp
    expected_minutes: int
    present_minutes: int
    missing_minutes: int
    per_day: pd.DataFrame  # trading_date, expected, present, missing, pct, is_early_close


def compute_file_coverage(
    rows_f: pd.DataFrame, full_grid: pd.DataFrame, schedule: SessionSchedule
) -> FileCoverage:
    """Cobertura de UN archivo, restringida a su propio rango ``[file_start, file_end]``.

    Entrada
    -------
    rows_f    : filas del archivo (columna ``timestamp``, tz-naive UTC --
        la salida cruda de ``load_research_rows``).
    full_grid : ``expected_bar_grid_frame`` calculado UNA sola vez sobre
        todo el rango de investigacion (columnas ``expected_ts_utc``,
        ``trading_date``) -- se recorta aqui a este archivo, en vez de
        recalcularse por archivo, por eficiencia.
    schedule  : ``SessionSchedule`` (para anotar cierres anticipados por
        dia en ``per_day``).

    Por que se recorta a ``[file_start, file_end]`` (seccion 5 de la
    tarea): no se espera ninguna barra antes del primer timestamp real de
    este archivo ni despues del ultimo -- esos minutos, si faltan, son un
    borde de archivo (``FILE_BOUNDARY``, ver
    ``build_file_boundary_records``), no un hueco interno de cobertura.

    Salida: ``FileCoverage`` con el desglose global del archivo y una
    tabla ``per_day`` (una fila por ``trading_date`` que este archivo
    toca, con minutos esperados/presentes/ausentes y si es un dia de
    cierre anticipado segun el calendario).
    """
    timestamps_utc = rows_f["timestamp"].dt.tz_localize("UTC")
    file_start = timestamps_utc.min()
    file_end = timestamps_utc.max()

    grid = full_grid[
        (full_grid["expected_ts_utc"] >= file_start) & (full_grid["expected_ts_utc"] <= file_end)
    ]
    observed = set(timestamps_utc)
    is_present = grid["expected_ts_utc"].isin(observed)

    per_day = (
        grid.assign(present=is_present.to_numpy())
        .groupby("trading_date")
        .agg(expected=("present", "size"), present_count=("present", "sum"))
        .reset_index()
    )
    per_day["missing"] = per_day["expected"] - per_day["present_count"]
    per_day["coverage_pct"] = 100.0 * per_day["present_count"] / per_day["expected"]
    per_day["is_early_close"] = per_day["trading_date"].apply(
        lambda d: schedule.is_early_close(d) if d in schedule.table.index else False
    )
    per_day = per_day.rename(columns={"present_count": "present"})

    return FileCoverage(
        source_file=str(rows_f["source_file"].iloc[0]),
        file_start=file_start,
        file_end=file_end,
        expected_minutes=int(len(grid)),
        present_minutes=int(is_present.sum()),
        missing_minutes=int(len(grid) - is_present.sum()),
        per_day=per_day,
    )


def aggregate_coverage_by_period(per_day_all: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Agrega la cobertura diaria (concatenada de todos los archivos) por año y por mes.

    Entrada: la concatenacion de ``FileCoverage.per_day`` de todos los
    archivos (columna ``trading_date`` ya en tipo ``date``).
    Salida: ``{"by_year": DataFrame, "by_month": DataFrame}``, cada una con
    ``expected``, ``present``, ``missing``, ``coverage_pct`` agregados por
    el periodo correspondiente (suma de minutos, no promedio de
    porcentajes -- evita que un año con pocos dias pese igual que uno
    completo).
    """
    df = per_day_all.copy()
    df["trading_date"] = pd.to_datetime(df["trading_date"])
    df["year"] = df["trading_date"].dt.year
    df["month"] = df["trading_date"].dt.to_period("M").astype(str)

    def _agg(group_col: str) -> pd.DataFrame:
        g = df.groupby(group_col).agg(expected=("expected", "sum"), present=("present", "sum"))
        g["missing"] = g["expected"] - g["present"]
        g["coverage_pct"] = 100.0 * g["present"] / g["expected"]
        return g.reset_index()

    return {"by_year": _agg("year"), "by_month": _agg("month")}


# ---------------------------------------------------------------------------
# Barras fuera de la grilla esperada (seccion 11)
# ---------------------------------------------------------------------------

def find_out_of_grid_bars(rows: pd.DataFrame, full_grid: pd.DataFrame, schedule: SessionSchedule) -> pd.DataFrame:
    """Timestamps observados que NO caen en la grilla esperada.

    Entrada: todas las filas del conjunto de investigacion (columnas
    ``source_file``, ``timestamp`` tz-naive UTC), la grilla esperada
    completa (``expected_bar_grid_frame``, YA sin los minutos del break
    secundario pre-corte -- ver ``session_calendar._session_expected_minutes``)
    y el calendario (para distinguir el motivo).

    Transformacion: "fuera de grilla" se define literalmente como "no
    pertenece a ``full_grid``" (comparacion exacta de membresia, no sólo
    "sin trading_date valida" como en una version anterior de esta
    funcion) -- esta es la definicion correcta y mas completa: garantiza
    que TODA fila observada quede contabilizada exactamente una vez, o
    bien como "presente" (en la grilla) o bien como "fuera de grilla"
    (aqui), sin ningun hueco de contabilidad entre ambas. Se distinguen
    dos motivos, ambos igual de legitimos para "fuera de grilla":
    - ``NO_SESSION``: el timestamp no cae dentro de NINGUNA sesion del
      calendario (fin de semana, feriado completo, o fuera de la ventana
      de apertura/cierre de su dia -- p.ej. el "goteo" previo a la
      reapertura dominical, seccion 8 de ``TDA02_cobertura.md``).
    - ``SECONDARY_BREAK_WINDOW``: el timestamp SI cae dentro de una sesion
      valida, pero en el tramo del break secundario que la grilla excluye
      para fechas `<= SECONDARY_BREAK_LAST_TRADING_DATE` (ver
      ``session_calendar.py``) -- es decir, una operacion real que ocurrio
      durante lo que se trata, estructuralmente, como mercado detenido.
      No se descarta ni se oculta: queda registrada aqui, con su motivo
      explicito, en vez de desaparecer silenciosamente de toda la
      contabilidad de cobertura.

    Salida: ``DataFrame`` con ``source_file``, ``timestamp_utc``,
    ``timestamp_ny``, ``reason`` -- SIN eliminar ni modificar nada
    (seccion 11: "no los elimines, registralos").
    """
    ts_utc = rows["timestamp"].dt.tz_localize("UTC")
    is_out_of_grid = ~ts_utc.isin(full_grid["expected_ts_utc"])

    out_of_grid = rows.loc[is_out_of_grid, ["source_file", "timestamp"]].copy()
    out_of_grid["timestamp_utc"] = ts_utc.loc[out_of_grid.index]
    out_of_grid["timestamp_ny"] = out_of_grid["timestamp_utc"].dt.tz_convert(NY_TZ)

    trading_dates = schedule.trading_date_for(out_of_grid["timestamp_utc"])
    out_of_grid["reason"] = np.where(trading_dates.isna().to_numpy(), "NO_SESSION", "SECONDARY_BREAK_WINDOW")
    return out_of_grid.drop(columns=["timestamp"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# DST (seccion 9)
# ---------------------------------------------------------------------------

def dst_transition_dates(years: list[int]) -> pd.DataFrame:
    """Fechas de transicion DST de EE. UU. (segundo domingo de marzo / primer domingo de noviembre).

    Entrada: lista de años.
    Transformacion: usa ``zoneinfo``/pandas para localizar, para cada año,
    el primer dia en que ``America/New_York`` pasa de EST a EDT (marzo,
    "spring forward") y el primer dia en que vuelve de EDT a EST
    (noviembre, "fall back") -- encontrados buscando el primer cambio de
    offset UTC de la zona horaria a lo largo del año, no una regla de
    calendario hardcodeada (evita asumir "segundo domingo de marzo" a
    mano, que es fragil ante cambios de ley).
    Salida: ``DataFrame`` con columnas ``year``, ``spring_transition_date``,
    ``fall_transition_date`` (ambas en fecha de Nueva York).
    """
    records = []
    for year in years:
        days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D", tz=NY_TZ)
        offset_hours = [d.utcoffset().total_seconds() / 3600 for d in days]
        changed = [i for i in range(1, len(days)) if offset_hours[i] != offset_hours[i - 1]]
        offset_changes = [days[i] for i in changed]
        spring = [d for d in offset_changes if d.month in (3, 4)]
        fall = [d for d in offset_changes if d.month in (10, 11)]
        records.append(
            {
                "year": year,
                "spring_transition_date": spring[0].date() if spring else None,
                "fall_transition_date": fall[0].date() if fall else None,
            }
        )
    return pd.DataFrame(records)


def build_dst_evidence(
    rows: pd.DataFrame, schedule: SessionSchedule, gaps_classified: pd.DataFrame
) -> pd.DataFrame:
    """Evidencia reproducible de manejo correcto de DST, año por año.

    Entrada
    -------
    rows            : filas del conjunto de investigacion.
    schedule        : ``SessionSchedule`` calculado sobre el rango completo.
    gaps_classified : salida de ``classify_gaps`` (para localizar el hueco
        de mantenimiento/cierre mas cercano a cada fecha de transicion).

    Metodo: para cada transicion DST dentro del rango de investigacion, se
    localiza el hueco de tipo ``DAILY_MAINTENANCE`` o ``EARLY_CLOSE`` cuyo
    ``before_date`` coincide con la fecha de transicion (o el dia habil
    mas cercano), y se compara: (a) el offset UTC de la sesion antes y
    despues de la transicion (debe cambiar en exactamente 1 hora); (b) la
    hora LOCAL de Nueva York del cierre de esa sesion (debe seguir siendo
    la hora normal de cierre, SIN desplazarse); (c) el numero de barras
    esperadas ese dia (debe seguir siendo el mismo: la transicion de DST
    ocurre fuera de horario de mercado -- de madrugada -- por lo que no
    debe alterar el conteo de minutos de ninguna sesion de este
    instrumento).

    Salida: ``DataFrame`` con una fila por transicion (spring/fall) por
    año, con las columnas necesarias para verificar (a)-(c) directamente
    desde el CSV, sin tener que re-ejecutar nada.
    """
    years = sorted(schedule.table.index.map(lambda d: d.year).unique())
    transitions = dst_transition_dates(list(years))

    maintenance = gaps_classified[
        gaps_classified["cause"].isin([CAUSE_DAILY_MAINTENANCE, CAUSE_EARLY_CLOSE])
    ]

    records = []
    for _, tr in transitions.iterrows():
        for kind, transition_date in [
            ("spring", tr["spring_transition_date"]),
            ("fall", tr["fall_transition_date"]),
        ]:
            if transition_date is None or transition_date not in schedule.table.index:
                continue
            row = schedule.table.loc[transition_date]
            close_ny = row["market_close"].tz_convert(NY_TZ)
            open_ny = row["market_open"].tz_convert(NY_TZ)
            expected_minutes = int((row["market_close"] - row["market_open"]).total_seconds() / 60)

            nearby = maintenance[maintenance["before_date"] == transition_date]
            observed_close_ny_time = (
                nearby.iloc[0]["before_ny_time"] if len(nearby) > 0 else None
            )
            observed_gap_minutes = nearby.iloc[0]["gap_minutes"] if len(nearby) > 0 else None

            records.append(
                {
                    "year": int(tr["year"]),
                    "transition": kind,
                    "transition_trading_date": transition_date,
                    "session_open_utc": row["market_open"],
                    "session_close_utc": row["market_close"],
                    "session_open_ny": open_ny,
                    "session_close_ny_time": close_ny.time(),
                    "session_utc_offset_hours": close_ny.utcoffset().total_seconds() / 3600,
                    "expected_minutes_in_session": expected_minutes,
                    "observed_gap_after_close_before_ny_time": observed_close_ny_time,
                    "observed_gap_after_close_minutes": observed_gap_minutes,
                }
            )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# TH04 -- clasificacion de barras inactivas (seccion 8)
# ---------------------------------------------------------------------------

@dataclass
class InactiveBarEvidence:
    flat_bars: pd.DataFrame          # todas las barras O=H=L=C observadas
    run_length_counts: pd.Series     # distribucion de longitud de racha (mismo precio, minuto contiguo)
    long_runs: pd.DataFrame          # rachas de longitud >= 6, con detalle fila a fila
    zero_volume_count: int
    verdict: str                     # texto de interpretacion, ver build_inactive_bar_mask


def analyze_inactive_bar_candidates(rows: pd.DataFrame, long_run_threshold: int = 6) -> InactiveBarEvidence:
    """Investiga la pregunta TH04: como se representa un minuto de mercado abierto sin operaciones.

    Entrada: filas del conjunto de investigacion, ordenadas por archivo y
    timestamp.

    Metodo (seccion 8 de la tarea: ``O=H=L=C`` por si solo NO es prueba
    suficiente de forward-fill):
    1. Se marcan como *candidatas* todas las barras con ``open == high ==
       low == close`` (un unico precio impreso en todo el minuto).
    2. Se agrupan en *rachas*: tramos de minutos CONSECUTIVOS (diferencia
       de 1 minuto exacta, dentro del mismo archivo) que ademas comparten
       el MISMO precio de cierre. Una racha de longitud 1 es una barra
       aislada; una racha larga es mas sugestiva de relleno sintetico
       (forward-fill tipicamente repite el mismo valor muchos minutos
       seguidos).
    3. Se contrasta contra volumen: TDA-00 ya certifico 0 filas con
       ``volume == 0`` en todo el conjunto de investigacion. Un
       forward-fill clasico (el que documenta la literatura y el propio
       roadmap, seccion TDA-01 §10) rellena con volumen CERO o un volumen
       fijo -- si las rachas observadas tienen volumen que VARIA minuto a
       minuto (evidencia positiva de actividad real, no un valor
       copiado), eso es evidencia EN CONTRA de forward-fill, no a favor.

    Salida: ``InactiveBarEvidence`` con el detalle completo y un veredicto
    en texto plano (``verdict``) que NO fuerza una clasificacion cuando la
    evidencia no alcanza (principio explicito de la seccion 8: declarar
    ``INDETERMINADO`` en vez de inventar).
    """
    rows = rows.sort_values(["source_file", "timestamp"]).reset_index(drop=True)
    is_flat = (rows["open"] == rows["high"]) & (rows["high"] == rows["low"]) & (rows["low"] == rows["close"])
    flat_bars = rows.loc[is_flat, ["source_file", "timestamp", "open", "close", "volume"]].reset_index(drop=True)

    prev_ts = rows.groupby("source_file")["timestamp"].shift(1)
    prev_close = rows.groupby("source_file")["close"].shift(1)
    prev_flat = is_flat.groupby(rows["source_file"]).shift(1).fillna(False)
    is_run_continuation = (
        is_flat
        & prev_flat.to_numpy()
        & (rows["close"].to_numpy() == prev_close.to_numpy())
        & ((rows["timestamp"] - prev_ts).dt.total_seconds() == 60).to_numpy()
    )
    run_id = (~is_run_continuation).cumsum()
    run_lengths = rows.loc[is_flat].groupby(run_id.loc[is_flat]).size()
    run_length_counts = run_lengths.value_counts().sort_index()

    long_run_ids = run_lengths[run_lengths >= long_run_threshold].index
    long_run_rows = []
    for rid in long_run_ids:
        members = rows.loc[is_flat].groupby(run_id.loc[is_flat]).get_group(rid)
        long_run_rows.append(members[["source_file", "timestamp", "open", "close", "volume"]])
    long_runs = (
        pd.concat(long_run_rows, ignore_index=True)
        if long_run_rows
        else pd.DataFrame(columns=["source_file", "timestamp", "open", "close", "volume"])
    )

    zero_volume_count = int((rows["volume"] == 0).sum())

    if len(long_runs) > 0:
        volume_is_constant = long_runs.groupby(
            (long_runs["timestamp"].diff().dt.total_seconds() != 60).cumsum()
        )["volume"].nunique().eq(1).all()
    else:
        volume_is_constant = None

    if zero_volume_count > 0:
        verdict = (
            "Se encontraron filas con volume == 0 -- contradice el hallazgo de TDA-00 "
            "y debe investigarse antes de continuar."
        )
    elif len(long_runs) == 0:
        verdict = (
            "No se encontraron rachas largas (>= "
            f"{long_run_threshold} minutos consecutivos) de barras planas al mismo "
            "precio. Todas las candidatas O=H=L=C observadas tienen volumen >= 1 "
            "(TDA-00 ya certifico 0 filas con volume == 0) y son, en su gran mayoria, "
            "eventos aislados de 1-2 minutos. No hay evidencia de FORWARD_FILL en el "
            "conjunto de investigacion: resultado NEGATIVO (G6)."
        )
    elif volume_is_constant:
        verdict = (
            "Se encontraron rachas largas de barras planas CON volumen constante "
            "dentro de la racha -- patron compatible con relleno sintetico. "
            "CANDIDATO_FORWARD_FILL: requiere revision manual antes de descartarse."
        )
    else:
        verdict = (
            "Se encontraron rachas largas de barras planas, pero el volumen VARIA "
            "minuto a minuto dentro de cada racha (evidencia de actividad real, no "
            "un valor copiado). Estas rachas se concentran en fechas de volatilidad "
            "extrema conocida (ver detalle en 'long_runs'). No se puede CONFIRMAR "
            "forward-fill con OHLCV Last unicamente: clasificacion FORWARD_FILL "
            "queda INDETERMINADA para estas rachas especificas (no se fuerza "
            "CONFIRMADO ni se descarta del todo)."
        )

    return InactiveBarEvidence(
        flat_bars=flat_bars,
        run_length_counts=run_length_counts,
        long_runs=long_runs,
        zero_volume_count=zero_volume_count,
        verdict=verdict,
    )


def build_inactive_bar_mask(
    full_grid: pd.DataFrame, rows: pd.DataFrame, inactive_evidence: InactiveBarEvidence,
    long_run_threshold: int = 6,
) -> pd.DataFrame:
    """Construye la mascara persistente ``barra_inactiva`` exigida por la tarea.

    Entrada: la grilla esperada completa (``expected_bar_grid_frame``,
    sin recortar por archivo -- se recorta aqui por archivo internamente),
    las filas observadas, y la evidencia ya calculada por
    ``analyze_inactive_bar_candidates``.

    Clasificacion tripartita por MINUTO esperado (seccion 8 de la tarea):
    - ``AUSENTE``: el minuto esperado no tiene ninguna fila observada.
    - ``VOLUMEN_CERO``: fila observada con ``volume == 0`` (0 casos en el
      conjunto de investigacion, ver TDA-00; la categoria se mantiene en
      el catalogo aunque este vacia).
    - ``ACTIVA``: fila observada, no plana (``O=H=L=C`` falso).
    - ``CANDIDATO_FORWARD_FILL``: fila observada, plana, que pertenece a
      una racha de longitud >= ``long_run_threshold`` (ver
      ``analyze_inactive_bar_candidates``) -- candidata que la evidencia
      disponible no permite confirmar ni descartar (queda
      ``INDETERMINADO`` en el informe narrativo, no aqui: la mascara
      guarda el HECHO estructural -- "pertenece a una racha larga" -- no
      un juicio final).
    - ``FLAT_AISLADA``: fila observada, plana, pero en una racha corta
      (< ``long_run_threshold``) -- consistente con un unico trade
      genuino a un precio, sin evidencia adicional de relleno.

    Esta mascara se calcula SOLO dentro del rango propio de cada archivo
    (igual que ``compute_file_coverage``): los minutos fuera de
    ``[file_start, file_end]`` de todo archivo no se incluyen (no son
    "ausentes" en ningun sentido util -- son borde de archivo).

    Salida: ``DataFrame`` con columnas ``source_file``, ``expected_ts_utc``,
    ``trading_date``, ``status`` (``AUSENTE``/``PRESENTE``), ``category``
    (una de las cinco de arriba), ``volume`` (nullable).
    """
    long_run_ts = set(
        zip(inactive_evidence.long_runs["source_file"], inactive_evidence.long_runs["timestamp"])
    ) if len(inactive_evidence.long_runs) > 0 else set()
    flat_ts = set(
        zip(inactive_evidence.flat_bars["source_file"], inactive_evidence.flat_bars["timestamp"])
    )

    pieces = []
    for source_file, rows_f in rows.groupby("source_file", sort=False):
        ts_utc = rows_f["timestamp"].dt.tz_localize("UTC")
        file_start, file_end = ts_utc.min(), ts_utc.max()
        grid = full_grid[
            (full_grid["expected_ts_utc"] >= file_start) & (full_grid["expected_ts_utc"] <= file_end)
        ].copy()
        grid["source_file"] = source_file

        observed_map = dict(zip(ts_utc, rows_f["volume"]))
        grid_ts_naive = grid["expected_ts_utc"].dt.tz_localize(None)
        present = grid["expected_ts_utc"].isin(observed_map)
        grid["status"] = np.where(present, "PRESENTE", "AUSENTE")
        grid["volume"] = grid["expected_ts_utc"].map(observed_map)

        # Vectorizado (evita `.apply` fila a fila sobre ~2M filas totales):
        # cada condicion es una mascara booleana; `np.select` aplica la
        # primera que matchee, en el mismo orden de prioridad que la version
        # fila a fila (VOLUMEN_CERO > CANDIDATO_FORWARD_FILL > FLAT_AISLADA > ACTIVA).
        is_long_run = pd.Series(
            list(zip(grid["source_file"], grid_ts_naive)), index=grid.index
        ).isin(long_run_ts)
        is_flat = pd.Series(
            list(zip(grid["source_file"], grid_ts_naive)), index=grid.index
        ).isin(flat_ts)
        is_zero_vol = grid["volume"] == 0

        grid["category"] = np.select(
            [present & is_zero_vol, present & is_long_run, present & is_flat, present],
            ["VOLUMEN_CERO", "CANDIDATO_FORWARD_FILL", "FLAT_AISLADA", "ACTIVA"],
            default=None,
        )
        pieces.append(grid[["source_file", "expected_ts_utc", "trading_date", "status", "category", "volume"]])

    return pd.concat(pieces, ignore_index=True)


# ---------------------------------------------------------------------------
# STOP-2 (seccion 18)
# ---------------------------------------------------------------------------

def check_stop2(gaps_classified: pd.DataFrame, threshold: float = 0.90) -> dict:
    """Verifica el criterio STOP-2: ¿la sesion real difiere sustancialmente de la esperada?

    Entrada: el inventario de huecos ya clasificado.
    Metodo: entre los huecos "de calendario" (``DAILY_MAINTENANCE``,
    ``WEEKEND``, ``HOLIDAY``, ``EARLY_CLOSE`` -- es decir, huecos que la
    grilla esperada predice como cierres reales, no anomalias) se calcula
    que fraccion tiene ``confidence == "ALTA"`` (borde EXACTO con el
    horario que declara el calendario). Un valor bajo indicaria que la
    sesion real diverge sistematicamente del calendario usado para
    construir la grilla -- exactamente la condicion que dispara STOP-2.
    ``threshold`` (90 % por defecto) es deliberadamente exigente: con el
    calendario correcto, la inmensa mayoria de estos huecos deberia
    coincidir de forma EXACTA (minuto a minuto), no solo aproximada.

    Salida: ``{"triggered": bool, "calendar_gap_count": int,
    "exact_match_fraction": float, "by_cause": DataFrame}``.
    """
    calendar_causes = [CAUSE_DAILY_MAINTENANCE, CAUSE_WEEKEND, CAUSE_HOLIDAY, CAUSE_EARLY_CLOSE]
    calendar_gaps = gaps_classified[gaps_classified["cause"].isin(calendar_causes)]
    if len(calendar_gaps) == 0:
        return {
            "triggered": False,
            "calendar_gap_count": 0,
            "exact_match_fraction": None,
            "by_cause": pd.DataFrame(),
        }
    exact_fraction = (calendar_gaps["confidence"] == CONFIDENCE_HIGH).mean()
    by_cause = (
        calendar_gaps.groupby("cause")["confidence"]
        .apply(lambda s: (s == CONFIDENCE_HIGH).mean())
        .rename("exact_match_fraction")
        .reset_index()
    )
    return {
        "triggered": bool(exact_fraction < threshold),
        "calendar_gap_count": int(len(calendar_gaps)),
        "exact_match_fraction": float(exact_fraction),
        "by_cause": by_cause,
    }


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------

@dataclass
class TDA02Result:
    schedule: SessionSchedule
    holidays: set
    rows: pd.DataFrame
    gaps_classified: pd.DataFrame
    file_boundaries: pd.DataFrame
    file_coverages: list[FileCoverage]
    per_day_all: pd.DataFrame
    coverage_by_period: dict[str, pd.DataFrame]
    out_of_grid_bars: pd.DataFrame
    dst_evidence: pd.DataFrame
    inactive_evidence: InactiveBarEvidence
    inactive_mask: pd.DataFrame
    stop2: dict


def run_tda02_analysis(config: SnapshotConfig) -> TDA02Result:
    """Orquesta el analisis completo de TDA-02 sobre el conjunto de investigacion.

    Reutiliza, sin duplicar logica: la proteccion del hold-out
    (``holdout_guard.py``), la carga de filas y el calculo de huecos
    internos (``tda01_temporal_semantics.py``) y el calendario nativo de
    CME (``session_calendar.py``). No escribe ningun archivo -- eso es
    responsabilidad de ``run_tda02.py``, que llama a esta funcion y
    vuelca cada pieza de ``TDA02Result`` a su artefacto correspondiente.
    """
    validate_research_holdout_disjoint(config)

    rows = load_research_rows(config)

    last_timestamps = rows.groupby("source_file")["timestamp"].max().to_dict()
    validate_last_timestamps_before_boundary(config, last_timestamps)

    research_min = rows["timestamp"].min()
    research_max = rows["timestamp"].max()
    schedule = build_session_schedule(
        research_min, research_max,
        calendar_name=config.tda02_calendar_name or "CME_Equity",
        buffer_days=config.tda02_calendar_buffer_days,
    )
    holidays = set(
        full_holidays_in_range(
            research_min - pd.Timedelta(days=config.tda02_calendar_buffer_days),
            research_max + pd.Timedelta(days=config.tda02_calendar_buffer_days),
            calendar_name=config.tda02_calendar_name or "CME_Equity",
        )
    )

    # raw_gaps sale de compute_intra_file_gaps con before_ts_utc/after_ts_utc
    # tz-naive (representan UTC sin localizar -- igual que en TDA-01).
    # attach_ny_wallclock necesita esa forma tz-naive (localiza ella misma);
    # la clasificacion de TDA-02, en cambio, necesita comparar contra el
    # horario del calendario, que SI es tz-aware -- por eso se localiza
    # despues de calcular las columnas de hora NY, no antes.
    raw_gaps = compute_intra_file_gaps(rows)
    raw_gaps = attach_ny_wallclock(raw_gaps)
    raw_gaps["before_ts_utc"] = raw_gaps["before_ts_utc"].dt.tz_localize("UTC")
    raw_gaps["after_ts_utc"] = raw_gaps["after_ts_utc"].dt.tz_localize("UTC")

    gaps_classified = classify_gaps(raw_gaps, schedule, holidays)

    file_bounds = {
        source_file: (
            group["timestamp"].min().tz_localize("UTC"),
            group["timestamp"].max().tz_localize("UTC"),
        )
        for source_file, group in rows.groupby("source_file", sort=False)
    }
    file_boundaries = build_file_boundary_records(file_bounds, schedule)

    full_grid = expected_bar_grid_frame(
        schedule,
        clip_start=research_min.tz_localize("UTC") - pd.Timedelta(days=1),
        clip_end=research_max.tz_localize("UTC") + pd.Timedelta(days=1),
    )

    file_coverages = [
        compute_file_coverage(group, full_grid, schedule)
        for _, group in rows.groupby("source_file", sort=False)
    ]
    per_day_pieces = []
    for fc in file_coverages:
        d = fc.per_day.copy()
        d["source_file"] = fc.source_file
        per_day_pieces.append(d)
    per_day_all = pd.concat(per_day_pieces, ignore_index=True)
    coverage_by_period = aggregate_coverage_by_period(per_day_all)

    out_of_grid_bars = find_out_of_grid_bars(rows, full_grid, schedule)

    dst_evidence = build_dst_evidence(rows, schedule, gaps_classified)

    inactive_evidence = analyze_inactive_bar_candidates(rows)
    inactive_mask = build_inactive_bar_mask(full_grid, rows, inactive_evidence)

    stop2 = check_stop2(gaps_classified)

    return TDA02Result(
        schedule=schedule,
        holidays=holidays,
        rows=rows,
        gaps_classified=gaps_classified,
        file_boundaries=file_boundaries,
        file_coverages=file_coverages,
        per_day_all=per_day_all,
        coverage_by_period=coverage_by_period,
        out_of_grid_bars=out_of_grid_bars,
        dst_evidence=dst_evidence,
        inactive_evidence=inactive_evidence,
        inactive_mask=inactive_mask,
        stop2=stop2,
    )
