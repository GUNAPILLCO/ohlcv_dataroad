"""TDA-02 -- calendario de sesion nativo de CME y grilla esperada de barras.

Este modulo construye, de forma reproducible, la "grilla esperada" que
exige TDA-02 (``docs/methodology/Tsay_OHLCV_analysis_roadmap.md``, seccion
"TDA-02"): el conjunto de timestamps de CIERRE de barra de 1 minuto que
deberian existir si el instrumento cotizase exactamente segun la
estructura NATIVA de negociacion de CME para el complejo de futuros de
indices bursatiles (al que pertenece MNQ), sin adoptar ninguna ventana
operativa de analisis (RTH, regimenes horarios heredados, etc. -- ver
``reports/mnq/TDA02_cobertura.md``, seccion 1, para la justificacion
completa de por que esta distincion importa).

Fuente del calendario
----------------------
``pandas_market_calendars.get_calendar("CME_Equity")`` (alias tambien
``CBOT_Equity``). Esta libreria de codigo abierto codifica, para el
complejo de futuros de indices bursatiles de CME (E-mini/Micro E-mini
S&P 500, Nasdaq-100, Dow, Russell, ...): horario de apertura/cierre de
sesion, el corte de mantenimiento diario (implicito: cierre de un dia y
apertura del siguiente), un break intradiario secundario declarado, y el
calendario de feriados completos y cierres anticipados -- el propio codigo
fuente de la libreria (``pandas_market_calendars/calendars/cme.py``) cita
``http://www.cmegroup.com/tools-information/holiday-calendar.html`` como
fuente. El acceso directo a ``cmegroup.com`` estuvo bloqueado en esta
sesion (confirmado de nuevo, igual que en TDA-01 -- ver
``reports/mnq/TDA01_convencion_temporal.md``, seccion 2.3): por eso esta
libreria es la mejor fuente documental disponible, PERO -- siguiendo la
instruccion explicita de esta tarea de no tratar una libreria de
calendario como fuente de verdad sin verificarla -- cada una de sus
afirmaciones estructurales se contrasta contra la evidencia forense
independiente ya construida en TDA-01 (``TDA01_evidencia_gaps.csv``) antes
de usarse para construir la grilla. El resultado de esa verificacion
(incluida una correccion real: el break secundario que la libreria declara
como permanente en realidad fue ELIMINADO por CME con efecto en la fecha
de negociacion 2021-06-28) esta documentado en detalle en
``reports/mnq/TDA02_cobertura.md``, seccion 1.

Fuentes primarias CME citadas en esta correccion (ver constantes mas abajo
para el detalle exacto de cada una):

- Eliminacion del break intradiario de 15:15-15:30 CT para futuros de
  indices bursatiles: CME Special Executive Report / Globex Notice de
  junio de 2021, con efecto en la fecha de negociacion (trade date)
  2021-06-28.
- CME SER 9499R -- dia nacional de duelo (2025-01-09): sesion de indices
  bursatiles de EE. UU. en Globex ABREVIADA (no cerrada), cierre 08:30 CT,
  reapertura normal 17:00 CT.

El acceso directo a ``cmegroup.com`` para releer estos avisos en el propio
sitio de CME volvio a estar bloqueado en esta sesion (mismo bloqueo que en
TDA-01 y en la primera version de este informe): las dos citas de arriba
se incorporan tal como fueron provistas para esta tarea, no verificadas
por lectura directa del documento original en esta sesion. Ambas SI se
verificaron, de forma independiente, contra la evidencia forense propia
(ver mas abajo) -- la cita documental y la evidencia forense convergen en
el mismo resultado en los dos casos, lo cual es la forma mas fuerte de
confirmacion disponible sin poder abrir el documento original.

Convencion de "trading date"
-----------------------------
Bajo la convencion de CIERRE de barra ya confirmada por TDA-01, una sesion
que abre el domingo 18:00 hora de Nueva York y cierra el lunes 17:00 hora
de Nueva York se etiqueta con la fecha de calendario en la que CIERRA (la
fecha "lunes" en el ejemplo). Esto coincide exactamente con como
``pandas_market_calendars`` indexa su tabla de horario (``schedule()``):
el indice de cada fila es la fecha en la que esa sesion cierra, no en la
que abre -- ver test
``test_trading_date_uses_the_session_close_date_not_the_open_date``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal

NY_TZ = ZoneInfo("America/New_York")

# La hora de cierre de sesion "normal" (no de cierre anticipado) del
# complejo CME_Equity, en hora de Nueva York -- usada unicamente para
# distinguir EARLY_CLOSE de DAILY_MAINTENANCE al clasificar un hueco (una
# sesion que cierra antes de esta hora es, por definicion, un cierre
# anticipado). No es un numero inventado: es la hora de cierre que declara
# la propia libreria para la inmensa mayoria de las sesiones del rango de
# investigacion (16:00 America/Chicago = 17:00 America/New_York todo el
# año, por construccion de zoneinfo), y coincide exactamente con la hora
# modal encontrada de forma independiente por la evidencia forense de
# TDA-01 (91,5 % de los bordes "antes" del corte de mantenimiento caen en
# `17:00:00` NY -- ver TDA01_convencion_temporal.md, tabla de §2.2).
NORMAL_SESSION_CLOSE_NY_TIME = time(17, 0, 0)

# --- Correccion documental + forense: eliminacion del break secundario --
#
# ``CMEEquityExchangeCalendar`` (pandas_market_calendars, ``cme.py``)
# declara un break intradiario PERMANENTE de `15:15-15:30 America/Chicago`
# (`16:15-16:30` hora de Nueva York) para todo el historial del calendario.
# Fuente primaria citada para esta tarea: CME Special Executive Report /
# Globex Notice de junio de 2021 -- el break de 15 minutos de futuros de
# indices bursatiles (Equity Index) fue ELIMINADO, con efecto en la fecha
# de negociacion (trade date) **2021-06-28** (lunes). La libreria NO
# modela este cambio (lo trata como vigente siempre): por eso esta fecha
# de corte se codifica aqui explicitamente, en vez de confiar en que
# ``pandas_market_calendars`` la represente -- exactamente la instruccion
# de esta tarea de no depender de una libreria de calendario para un
# hecho que cambio en el tiempo.
#
# Verificacion forense independiente (``TDA01_evidencia_gaps.csv``): el
# hueco con la firma exacta de dos lados de este break (borde anterior
# `16:15:00` NY, borde posterior `16:31:00` NY bajo la convencion de
# cierre de barra) aparece en 353 sesiones **consecutivas**, desde el
# primer dia del conjunto de investigacion (2019-12-23) hasta el
# **viernes 2021-06-25** inclusive -- y no vuelve a aparecer ni una sola
# vez despues, en ninguna de las 1.082 sesiones restantes del conjunto de
# investigacion. La fecha forense (ultima aparicion: viernes 25/06/2021)
# y la fecha documental (eliminacion con efecto lunes 28/06/2021, el
# primer dia habil siguiente) **coinciden exactamente**: la cita
# documental y la evidencia forense se corroboran mutuamente.
#
# Consecuencia para la grilla esperada: los minutos del break
# (`break_start+1min .. break_end` de cada sesion, bajo la convencion de
# cierre de barra) se EXCLUYEN de la grilla esperada solo para sesiones
# con `trading_date <= SECONDARY_BREAK_LAST_TRADING_DATE`; a partir de
# `SECONDARY_BREAK_ABOLISHED_TRADING_DATE` la grilla es continua durante
# esa franja horaria, igual que cualquier otro minuto de sesion. Antes de
# esta correccion, esos minutos se contaban como "esperados pero
# ausentes" incluso cuando el mercado estaba estructuralmente detenido,
# sesgando la cobertura hacia abajo -- ver ``_session_expected_minutes``.
SECONDARY_BREAK_LAST_TRADING_DATE = date(2021, 6, 25)
SECONDARY_BREAK_ABOLISHED_TRADING_DATE = date(2021, 6, 28)

# --- Correccion documental + forense puntual: 2025-01-09 -----------------
#
# ``CMEEquityExchangeCalendar.adhoc_holidays`` (pandas_market_calendars,
# ``cme.py``) declara ``USNationalDaysofMourning`` -- que incluye
# 2025-01-09 (dia nacional de duelo por James "Jimmy" Carter) -- como
# CIERRE COMPLETO, sin ninguna sesion.
#
# Fuente primaria citada para esta tarea: CME SER 9499R -- la sesion de
# indices bursatiles de EE. UU. (U.S. Equity Index) en Globex ese dia fue
# ABREVIADA, no cerrada: cierre `08:30 America/Chicago`, reapertura normal
# `17:00 America/Chicago`. Traducido a Nueva York (America/Chicago esta
# siempre 1 hora por detras de America/New_York, ambas zonas siguen las
# mismas fechas de cambio de horario de EE. UU.): cierre `09:30` NY,
# reapertura `18:00` NY (primera barra completa `18:01` NY, convencion de
# cierre de barra).
#
# Verificacion forense independiente (``data/raw/mnq/20_mnq_03_25.Last.txt``):
# ese archivo tiene barras con volumen normal (decenas a miles de
# contratos por minuto, hasta 10.450) de forma practicamente continua
# desde la reapertura habitual del 2025-01-08 (`2025-01-08 23:01:00 UTC` =
# `18:01` NY = `17:01` CT, la firma exacta de reapertura normal) hasta
# `2025-01-09 14:30:00 UTC` (`09:30` NY = `08:30` CT) -- momento exacto en
# el que el archivo muestra un hueco de 511 minutos hasta la siguiente
# reapertura habitual (`18:01` NY esa misma tarde). **La cita documental
# (CME SER 9499R: cierre 08:30 CT) y la evidencia forense (ultimo bar real
# a las 08:30 CT en punto) coinciden de forma exacta, al minuto.** Esta
# fecha queda, por tanto, `CONFIRMADO` por fuente primaria + evidencia
# forense convergente -- ya no es una inferencia forense sin corroborar.
# Es, ademas, la UNICA fecha de ``USNationalDaysofMourning`` dentro del
# rango de investigacion (2019-12-23 a 2025-06-22): verificado
# explicitamente (ver test
# ``test_only_one_adhoc_mourning_holiday_falls_in_research_range``).
#
# El acceso directo a ``cmegroup.com`` para releer el SER original en el
# sitio de CME volvio a estar bloqueado en esta sesion: la cita se
# incorpora tal como fue provista para esta tarea (numero de aviso y
# horarios exactos), no releida directamente del documento en esta sesion
# -- pero la convergencia exacta al minuto con la evidencia forense propia
# es la corroboracion mas fuerte disponible sin poder abrir el documento.
FORENSIC_SCHEDULE_OVERRIDES: dict = {
    # trading_date -> (market_open_utc, market_close_utc). ``market_open``
    # es el instante crudo de apertura (UN minuto ANTES de la primera
    # barra esperada -- misma convencion que usa el resto de la tabla:
    # ver ``expected_bar_grid``); el primer bar real observado ese dia fue
    # `2025-01-08 23:01:00 UTC`, de ahi `market_open = 23:00:00` --
    # coincide con la reapertura normal `17:00 CT` que declara CME SER
    # 9499R. ``market_close`` es `2025-01-09 14:30:00 UTC` (`08:30` CT) --
    # coincide exactamente con el cierre `08:30 CT` que declara el mismo
    # SER, y con el ultimo bar real observado antes del hueco de 511
    # minutos.
    "2025-01-09": ("2025-01-08 23:00:00", "2025-01-09 14:30:00"),
}


def get_cme_equity_calendar():
    """Devuelve el objeto calendario de ``pandas_market_calendars`` para CME_Equity.

    Entrada: ninguna.
    Salida: una instancia de
    ``pandas_market_calendars.calendars.cme.CMEEquityExchangeCalendar``,
    el calendario del complejo de futuros de indices bursatiles de CME
    (aliases ``CME_Equity`` / ``CBOT_Equity``). Se llama a
    ``mcal.get_calendar`` en cada invocacion (no se cachea a nivel de
    modulo) porque el costo de construirlo es despreciable frente al costo
    de generar el horario sobre un rango de años, y evita cualquier estado
    mutable compartido entre llamadas.
    """
    return mcal.get_calendar("CME_Equity")


@dataclass(frozen=True)
class SessionSchedule:
    """Horario de sesiones de CME_Equity ya calculado sobre un rango de fechas.

    Atributos
    ---------
    table : DataFrame indexado por ``trading_date`` (``date`` de Python,
        la fecha de CIERRE de la sesion -- ver docstring del modulo),
        columnas ``market_open`` y ``market_close`` (tz-aware UTC) y,
        cuando la libreria los declara, ``break_start``/``break_end``
        (tz-aware UTC, pueden ser ``NaT``).
    normal_close_ny_time : la hora de cierre "normal" en NY, usada para
        distinguir cierres anticipados.
    """

    table: pd.DataFrame
    normal_close_ny_time: time = NORMAL_SESSION_CLOSE_NY_TIME

    def session_open_close_intervals(self) -> pd.IntervalIndex:
        """``IntervalIndex`` (market_open, market_close] para localizar sesiones.

        Se construye una sola vez y se reutiliza para todas las
        busquedas vectorizadas de "a que sesion pertenece este timestamp"
        (``trading_date_for``). El intervalo es cerrado por la derecha
        (``closed="right"``) porque, bajo la convencion de CIERRE de
        barra, el propio instante ``market_close`` SI pertenece a esa
        sesion (es la ultima barra), mientras que ``market_open`` NO
        pertenece a ninguna barra (es el instante en que empieza a
        acumularse la primera barra, que solo se etiqueta un minuto
        despues).
        """
        return pd.IntervalIndex.from_arrays(
            self.table["market_open"], self.table["market_close"], closed="right"
        )

    def trading_date_for(self, timestamps_utc: pd.Series) -> pd.Series:
        """Devuelve, para cada timestamp UTC, la ``trading_date`` de su sesion.

        Entrada: una ``Series`` de timestamps tz-aware UTC.
        Transformacion: localiza cada timestamp dentro de
        ``session_open_close_intervals()`` (busqueda vectorizada por
        intervalo, O(log n) por elemento). Un timestamp que no cae dentro
        de NINGUN intervalo de sesion (por ejemplo, un timestamp durante
        un fin de semana o un feriado completo) no tiene sesion valida.
        Salida: una ``Series`` de ``datetime.date`` (con ``pd.NaT``/``None``
        donde no hay sesion), mismo indice que la entrada.
        """
        intervals = self.session_open_close_intervals()
        positions = intervals.get_indexer(pd.DatetimeIndex(timestamps_utc))
        dates = pd.Series(pd.NaT, index=timestamps_utc.index, dtype=object)
        valid = positions >= 0
        dates.loc[valid] = self.table.index.to_numpy()[positions[valid]]
        return dates

    def is_early_close(self, trading_date) -> bool:
        """``True`` si la sesion de esa fecha cierra ANTES de la hora normal."""
        close_ny = self.table.loc[trading_date, "market_close"].tz_convert(NY_TZ)
        return close_ny.time() < self.normal_close_ny_time


def build_session_schedule(
    start_date, end_date, calendar_name: str = "CME_Equity", buffer_days: int = 10
) -> SessionSchedule:
    """Construye el horario de sesiones CME_Equity sobre ``[start_date, end_date]``.

    Entrada
    -------
    start_date, end_date : fechas (cualquier tipo aceptado por
        ``pandas.Timestamp``) que acotan el rango de interes -- tipicamente
        el primer y ultimo timestamp del conjunto de investigacion.
    calendar_name : nombre del calendario de ``pandas_market_calendars``
        (por defecto ``"CME_Equity"``, ver ``configs/mnq_snapshot.yaml``).
    buffer_days : dias de calendario añadidos a cada lado de
        ``[start_date, end_date]`` antes de pedirle el horario a la
        libreria. Es necesario para poder resolver correctamente la sesion
        del PRIMER y del ULTIMO timestamp del rango sin efecto de borde
        (si se pidiera el horario exactamente `= [start_date, end_date]`,
        una sesion que abre unos dias antes de `start_date` y cierra
        DENTRO del rango podria no aparecer completa).

    Salida
    ------
    ``SessionSchedule`` con la tabla ya calculada, en UTC.
    """
    cal = mcal.get_calendar(calendar_name)
    buffer = pd.Timedelta(days=buffer_days)
    sched = cal.schedule(
        start_date=(pd.Timestamp(start_date) - buffer).date(),
        end_date=(pd.Timestamp(end_date) + buffer).date(),
        tz="UTC",
    )
    sched.index = pd.DatetimeIndex(sched.index).date
    sched.index.name = "trading_date"

    for date_str, (open_str, close_str) in FORENSIC_SCHEDULE_OVERRIDES.items():
        override_date = pd.Timestamp(date_str).date()
        if not (pd.Timestamp(start_date).date() <= override_date <= pd.Timestamp(end_date).date()):
            continue
        open_ts = pd.Timestamp(open_str, tz="UTC")
        close_ts = pd.Timestamp(close_str, tz="UTC")
        if override_date in sched.index:
            sched.loc[override_date, "market_open"] = open_ts
            sched.loc[override_date, "market_close"] = close_ts
            sched.loc[override_date, "break_start"] = pd.NaT
            sched.loc[override_date, "break_end"] = pd.NaT
        else:
            new_row = pd.DataFrame(
                {"market_open": [open_ts], "market_close": [close_ts], "break_start": [pd.NaT], "break_end": [pd.NaT]},
                index=pd.Index([override_date], name="trading_date"),
            )
            sched = pd.concat([sched, new_row]).sort_index()

    return SessionSchedule(table=sched)


def _session_expected_minutes(trading_date, market_open, market_close, break_start, break_end) -> pd.DatetimeIndex:
    """Grilla esperada de UNA sesion, excluyendo el break secundario cuando corresponde.

    Entrada
    -------
    trading_date : fecha de cierre de la sesion (``datetime.date``).
    market_open, market_close : bordes de la sesion (tz-aware UTC).
    break_start, break_end : bordes del break intradiario que declara el
        calendario para esta sesion (tz-aware UTC, pueden ser ``NaT``).

    Transformacion
    --------------
    1. Grilla completa: ``market_open + 1min .. market_close`` (convencion
       de cierre de barra).
    2. Si ``trading_date`` es anterior o igual a
       ``SECONDARY_BREAK_LAST_TRADING_DATE`` (2021-06-25, ver la constante
       para la cita documental + forense completa) Y el calendario declara
       un break para esta sesion: se EXCLUYEN de la grilla los minutos
       ``break_start + 1min .. break_end`` -- el mercado estuvo
       estructuralmente detenido ahi, igual que en el corte de
       mantenimiento nocturno, asi que esos minutos no deben esperarse
       como barras ausentes.
    3. Para sesiones con ``trading_date >= SECONDARY_BREAK_ABOLISHED_TRADING_DATE``
       (2021-06-28), o para cualquier sesion sin break declarado, no se
       excluye nada: la grilla es continua en esa franja horaria, igual
       que el resto de la sesion.

    Salida: ``pandas.DatetimeIndex`` (tz-aware UTC) de los minutos
    esperados de esta sesion.
    """
    full = pd.date_range(start=market_open + pd.Timedelta(minutes=1), end=market_close, freq="1min")
    if (
        trading_date <= SECONDARY_BREAK_LAST_TRADING_DATE
        and pd.notna(break_start)
        and pd.notna(break_end)
    ):
        break_minutes = pd.date_range(
            start=break_start + pd.Timedelta(minutes=1), end=break_end, freq="1min"
        )
        full = full.difference(break_minutes)
    return full


def expected_bar_grid(
    schedule: SessionSchedule, clip_start=None, clip_end=None
) -> pd.DatetimeIndex:
    """Construye la grilla esperada de timestamps de CIERRE de barra de 1 minuto.

    Entrada
    -------
    schedule   : ``SessionSchedule`` ya calculado.
    clip_start, clip_end : si se pasan (tz-aware UTC), la grilla resultante
        se recorta a ``[clip_start, clip_end]`` (ambos inclusive). Se usa,
        por ejemplo, para acotar la grilla al rango realmente cubierto por
        UN archivo (``file_start``/``file_end``), de forma que no se
        esperen barras antes del primer timestamp real de ese archivo ni
        despues del ultimo -- eso es un borde de archivo (``FILE_BOUNDARY``),
        no un hueco (ver seccion 5 de la tarea y
        ``reports/mnq/TDA02_cobertura.md``, seccion 5).

    Transformacion
    --------------
    Para cada sesion de la tabla, la grilla esperada es
    ``market_open + 1 minuto .. market_close`` (convencion de cierre de
    barra ya confirmada por TDA-01), EXCLUYENDO el break secundario
    cuando corresponda -- ver ``_session_expected_minutes`` y
    ``SECONDARY_BREAK_LAST_TRADING_DATE``.

    Salida
    ------
    ``pandas.DatetimeIndex`` (tz-aware UTC), ordenado, sin duplicados, con
    todos los minutos esperados de todas las sesiones de ``schedule``
    (recortado si se pidio).
    """
    pieces = [
        _session_expected_minutes(
            trading_date, row["market_open"], row["market_close"],
            row.get("break_start"), row.get("break_end"),
        )
        for trading_date, row in schedule.table.iterrows()
    ]
    grid = pieces[0].append(pieces[1:]) if len(pieces) > 1 else pieces[0]
    grid = pd.DatetimeIndex(grid).sort_values()
    if clip_start is not None:
        grid = grid[grid >= pd.Timestamp(clip_start)]
    if clip_end is not None:
        grid = grid[grid <= pd.Timestamp(clip_end)]
    return grid


def expected_bar_grid_frame(
    schedule: SessionSchedule, clip_start=None, clip_end=None
) -> pd.DataFrame:
    """Como :func:`expected_bar_grid`, pero conservando la ``trading_date`` de cada minuto.

    Entrada / transformacion: identica a ``expected_bar_grid``, incluida
    la exclusion del break secundario para sesiones con
    ``trading_date <= SECONDARY_BREAK_LAST_TRADING_DATE``
    (``_session_expected_minutes``), salvo que en vez de construir un
    unico ``DatetimeIndex`` concatenado, se construye un ``DataFrame`` por
    sesion (columnas ``expected_ts_utc`` y ``trading_date`` repetida) y se
    concatenan todos. Esto evita tener que volver a localizar cada minuto
    contra el horario de sesiones (via ``trading_date_for``, que es mas
    cara) cuando el llamador ya sabe, por construccion, a que sesion
    pertenece cada minuto de la grilla.

    Salida: ``DataFrame`` con columnas ``expected_ts_utc`` (tz-aware UTC) y
    ``trading_date`` (``datetime.date``), ordenado por ``expected_ts_utc``,
    recortado a ``[clip_start, clip_end]`` si se pidio.
    """
    frames = []
    for trading_date, row in schedule.table.iterrows():
        idx = _session_expected_minutes(
            trading_date, row["market_open"], row["market_close"],
            row.get("break_start"), row.get("break_end"),
        )
        frames.append(pd.DataFrame({"expected_ts_utc": idx, "trading_date": trading_date}))
    grid = pd.concat(frames, ignore_index=True).sort_values("expected_ts_utc").reset_index(drop=True)
    if clip_start is not None:
        grid = grid[grid["expected_ts_utc"] >= pd.Timestamp(clip_start)]
    if clip_end is not None:
        grid = grid[grid["expected_ts_utc"] <= pd.Timestamp(clip_end)]
    return grid.reset_index(drop=True)


def full_holidays_in_range(start_date, end_date, calendar_name: str = "CME_Equity") -> list:
    """Fechas de cierre COMPLETO (sin ninguna sesion) de CME_Equity en el rango.

    Entrada: rango de fechas de interes.
    Transformacion: ``cal.holidays().holidays`` devuelve TODAS las fechas
    de feriado que produce el calendario de feriados regulares + ad-hoc de
    la libreria (para cualquier año, pasado o futuro); se filtra al rango
    pedido. Estas son, por construccion, exactamente las fechas que
    ``cal.schedule()`` omite de su indice -- no hace falta recalcularlas de
    otra forma.
    Salida: lista ordenada de ``datetime.date``.
    """
    cal = mcal.get_calendar(calendar_name)
    hol = cal.holidays()
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    overridden = {pd.Timestamp(d).date() for d in FORENSIC_SCHEDULE_OVERRIDES}
    dates = [pd.Timestamp(h).date() for h in hol.holidays]
    return sorted(d for d in dates if start <= d <= end and d not in overridden)
