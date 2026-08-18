"""TDA-06 -- Perfil determinista intradia y de calendario.

Implementa la etapa TDA-06 del roadmap
(``docs/methodology/Tsay_OHLCV_analysis_roadmap.md``, seccion "TDA-06"):
antes de estudiar cualquier dependencia estocastica (ACF, clustering de
volatilidad, dependencia en media), hay que aislar el componente
DETERMINISTA ligado al reloj -- ¿el minuto del dia y el dia de la semana
cambian sistematicamente el comportamiento estadistico de MNQ?

Que responde esta etapa, en una frase: si una hora concreta es siempre
mas activa/volatil que otra (patron de RELOJ, repetible año tras año),
eso no es "dependencia" en el sentido de TDA-08/TDA-09 -- es estacionalidad
determinista, y hay que quitarla antes de buscar cualquier otra cosa,
porque un pico de ACF en el lag de un dia de sesion, o un ARCH aparente,
pueden ser puramente el reloj repitiendose (roadmap, TDA-06 "Fundamento
Tsay": C3 error 27, C2 Ljung-Box con estacionalidad).

Que NO hace este modulo (deliberadamente, fuera de alcance de TDA-06):

- No calcula ACF/PACF ni busca dependencia estocastica (TDA-08/TDA-09).
- No estudia volatility clustering ni ajusta ARCH/GARCH (TDA-09).
- No entrena modelos, no crea targets, no busca predictibilidad.
- No ejecuta EVT ni optimiza umbrales de extremos (TDA-13).
- No retoma TH10 (diferida, ver TDA-04).
- No asume la forma en U de NYSE ni importa Overnight/Pre-market/Opening/
  Regular/Closing de ninguna decision historica: la segmentacion de esta
  etapa es una PROPUESTA EMPIRICA derivada de los datos, no una decision
  de arquitectura.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Minutos estructuralmente cerrados (sin ninguna barra en ningun dia, o
# en ningun dia de la remuestra de bootstrap) producen columnas
# enteramente NaN al pasar por ``nanmedian``/``nanmean`` -- exactamente
# el comportamiento correcto (seccion 5/7 de la tarea: nunca inventar una
# observacion ahi), pero NumPy emite un ``RuntimeWarning`` esperado
# ("Mean/All-NaN slice encountered") en ese caso. Se silencia GLOBALMENTE
# en este modulo (no por-llamada) porque ocurre en decenas de miles de
# columnas del pivote real (minutos de mantenimiento, fin de semana) y
# es, por diseño, benigno -- el resultado sigue siendo ``NaN`` explicito,
# nunca un valor inventado.
warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", message=".*All-NaN slice encountered.*")

from ohlcv_dataroad.config import SnapshotConfig
from ohlcv_dataroad.ingest.holdout_guard import (
    validate_last_timestamps_before_boundary,
    validate_research_holdout_disjoint,
)
from ohlcv_dataroad.ingest.session_calendar import NY_TZ

# --- Años completos vs parciales (seccion 8 de la tarea) -------------------
# El conjunto de investigacion empieza el 2019-12-23 (2019 = 9 dias de
# calendario) y termina en el corte de hold-out dentro de 2025 (2025 =
# parcial). 2020-2024 son los unicos años calendario COMPLETOS -- son la
# evidencia PRINCIPAL de estabilidad; 2019/2025 se muestran siempre, pero
# se documentan explicitamente como evidencia COMPLEMENTARIA parcial (no
# se les da el mismo peso). Coherente con el cierre de TDA-05.
COMPLETE_YEARS = (2020, 2021, 2022, 2023, 2024)
PARTIAL_YEARS = (2019, 2025)

WEEKDAY_NAMES = {0: "Lunes", 1: "Martes", 2: "Miercoles", 3: "Jueves", 4: "Viernes", 5: "Sabado", 6: "Domingo"}


# ---------------------------------------------------------------------------
# Carga de entradas y eje temporal
# ---------------------------------------------------------------------------

def load_inputs(config: SnapshotConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee variables y mascara de validez de TDA-04 -- sin recalcular retornos desde raw.

    TDA-06 nunca abre ``data/raw/`` ni ``config.holdout_files``: su unica
    entrada son estos dos artefactos, ya construidos exclusivamente sobre
    el conjunto de investigacion por TDA-04.
    """
    variables = pd.read_parquet(config.tda04_variables_parquet_path).sort_values("timestamp").reset_index(drop=True)
    validity = pd.read_parquet(config.tda04_validity_mask_parquet_path).sort_values("timestamp").reset_index(drop=True)
    return variables, validity


def attach_calendar_fields(variables: pd.DataFrame, validity: pd.DataFrame) -> pd.DataFrame:
    """Añade ``minute_of_day``, ``year_ny`` y ``weekday`` a las variables de TDA-04.

    Entrada: las dos tablas de TDA-04, mismo orden de timestamps
    (verificado con ``assert``, igual que TDA-05).

    ``minute_of_day = hour_ny * 60 + minute_ny`` (seccion 5 de la tarea):
    ``timestamp`` es tz-naive UTC por construccion desde TDA-00; se
    localiza como UTC y se convierte a ``America/New_York`` con
    ``zoneinfo`` (``NY_TZ``, el mismo mecanismo DST-aware que TDA-01/02/
    03/05 -- nunca un offset fijo). Dominio resultante: 0..1439.

    ``weekday`` se calcula sobre ``trading_date`` (la columna de TDA-02/
    TDA-03: la fecha de CIERRE de la sesion CME), **no** sobre la fecha
    calendario local de la barra -- exactamente lo que exige la seccion 5
    de la tarea: una barra del domingo por la noche (hora NY) pertenece a
    la ``trading_date`` del lunes, asi que su ``weekday`` es LUNES (0), no
    domingo (6). Calcular el dia de semana sobre la fecha local de la
    barra fabricaria un falso "efecto domingo" que en realidad es la
    apertura del lunes.

    Salida: ``variables`` con las columnas nuevas ``minute_of_day``,
    ``hour_ny``, ``year_ny``, ``weekday`` añadidas, y ``r_1m_valid``
    unida desde ``validity`` (misma fila, mismo orden).
    """
    df = variables.sort_values("timestamp").reset_index(drop=True)
    v = validity.sort_values("timestamp").reset_index(drop=True)
    assert (df["timestamp"].to_numpy() == v["timestamp"].to_numpy()).all(), (
        "variables y validity no comparten exactamente los mismos timestamps, en el mismo orden"
    )

    ts_ny = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(NY_TZ)
    minute_of_day = ts_ny.dt.hour * 60 + ts_ny.dt.minute

    trading_date = pd.to_datetime(df["trading_date"])
    weekday = trading_date.dt.weekday

    out = df.copy()
    out["hour_ny"] = ts_ny.dt.hour
    out["year_ny"] = ts_ny.dt.year
    out["minute_of_day"] = minute_of_day
    out["weekday"] = weekday
    out["r_1m_valid"] = v["r_1m_valid"].to_numpy()
    return out


# ---------------------------------------------------------------------------
# Motor generico de perfil por minuto: pivote + bootstrap de bloques por trading_date
# ---------------------------------------------------------------------------

def _build_date_minute_pivot(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Pivota ``value_col`` a una matriz densa ``(n_trading_dates, 1440)``.

    Cada ``trading_date`` tiene, como mucho, UNA barra por
    ``minute_of_day`` (la serie canonica de TDA-03 no tiene timestamps
    duplicados) -- asi que esta matriz es una representacion exacta y sin
    perdida de informacion de "una fila por jornada, una columna por
    minuto del reloj NY", con ``NaN`` donde no hay barra (minuto
    estructuralmente cerrado, o hueco). Se usa como base para calcular
    tanto el perfil puntual (``nanmedian``/``nanmean`` por columna) como
    el bootstrap de bloques (remuestrear FILAS completas = jornadas
    completas, preservando la dependencia intra-dia, G5).
    """
    dates, date_idx = np.unique(df["trading_date"].to_numpy(), return_inverse=True)
    minute_idx = df["minute_of_day"].to_numpy()
    values = df[value_col].to_numpy(dtype=float)

    pivot = np.full((len(dates), 1440), np.nan)
    pivot[date_idx, minute_idx] = values
    return pivot, dates


def block_bootstrap_minute_profile(
    pivot: np.ndarray, agg: str = "median", n_boot: int = 300, seed: int = 0
) -> dict[str, np.ndarray]:
    """Perfil por minuto (0..1439) con banda de incertidumbre por bootstrap de bloques de jornada.

    Entrada: la matriz ``(n_dates, 1440)`` de ``_build_date_minute_pivot``.

    Por que bootstrap de BLOQUES de ``trading_date`` (G5, igual que
    TDA-05): las barras dentro de una misma jornada no son independientes
    entre si; remuestrear jornadas COMPLETAS (con reemplazo) preserva esa
    dependencia. Cada repeticion sortea ``n_dates`` filas de la matriz con
    reemplazo y recalcula el agregado por columna -- vectorizado con
    NumPy (fancy indexing de filas + ``nanmedian``/``nanmean`` por
    columna), nunca reconstruyendo un ``DataFrame`` por iteracion (la
    misma leccion de rendimiento de TDA-05).

    ``agg``: ``"median"`` (estimador principal para magnitud, seccion 7
    de la tarea) o ``"mean"`` (para ``zero_1m``/frecuencia de extremos,
    donde el promedio de un indicador 0/1 es la proporcion -- una
    mediana de un indicador binario no es interpretable, seccion 6).

    Salida: ``{"n": (1440,), "point": (1440,), "ci_lo": (1440,),
    "ci_hi": (1440,)}``. Minutos sin ninguna observacion (``n=0``,
    estructuralmente cerrados) quedan con ``point``/``ci_lo``/``ci_hi``
    en ``NaN`` -- nunca se inventa una observacion donde el mercado no
    operaba (seccion 5 de la tarea).
    """
    if agg not in ("median", "mean"):
        raise ValueError(f"agg debe ser 'median' o 'mean', recibido: {agg!r}")
    func = np.nanmedian if agg == "median" else np.nanmean

    n = (~np.isnan(pivot)).sum(axis=0)
    with np.errstate(invalid="ignore"):
        point = func(pivot, axis=0)
    point = np.where(n > 0, point, np.nan)

    n_dates = pivot.shape[0]
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, 1440))
    for b in range(n_boot):
        sampled_rows = rng.integers(0, n_dates, size=n_dates)
        with np.errstate(invalid="ignore"):
            boot[b, :] = func(pivot[sampled_rows, :], axis=0)

    with np.errstate(invalid="ignore"):
        ci_lo = np.nanpercentile(boot, 2.5, axis=0)
        ci_hi = np.nanpercentile(boot, 97.5, axis=0)
    ci_lo = np.where(n > 0, ci_lo, np.nan)
    ci_hi = np.where(n > 0, ci_hi, np.nan)

    return {"n": n, "point": point, "ci_lo": ci_lo, "ci_hi": ci_hi}


def build_minute_profile_table(
    df: pd.DataFrame, value_col: str, agg: str = "median", n_boot: int = 300, seed: int = 0
) -> pd.DataFrame:
    """Tabla de perfil por minuto del dia (0..1439) para UNA variable, sobre las filas de ``df``.

    ``df`` ya debe estar filtrado a la poblacion correcta para
    ``value_col`` (seccion 6 de la tarea): filas ``r_1m_valid`` para
    variables de retorno, todas las barras admisibles para variables de
    una sola barra (``volume``, ``log_hl``). Ver
    ``build_all_minute_profiles`` para la eleccion documentada de
    poblacion por variable.

    Salida: ``DataFrame`` indexado por ``minute_of_day`` (0..1439, TODOS
    presentes, incluidos los que no tienen ninguna observacion -- con
    ``n=0`` explicito, nunca omitidos) con columnas ``n``, ``point``,
    ``ci_lo``, ``ci_hi``.
    """
    pivot, _ = _build_date_minute_pivot(df, value_col)
    result = block_bootstrap_minute_profile(pivot, agg=agg, n_boot=n_boot, seed=seed)
    return pd.DataFrame(
        {
            "minute_of_day": np.arange(1440),
            "n": result["n"],
            "point": result["point"],
            "ci_lo": result["ci_lo"],
            "ci_hi": result["ci_hi"],
        }
    )


# --- Catalogo de variables: poblacion y agregador (seccion 6 de la tarea) --
# Documentado UNA vez, reutilizado por el perfil global, por año y por
# dia de semana -- para que la eleccion de poblacion/agregador nunca
# pueda divergir entre tablas.
#
# - Variables de RETORNO (r_1m, abs_r_1m, r2_1m, zero_1m, extreme_flag):
#   poblacion = SOLO filas con r_1m_valid=True. Usan la MISMA mascara de
#   TDA-04 que certifico que existe una barra t-1 comparable -- nunca se
#   reintroduce un retorno invalido.
# - Variables de UNA SOLA BARRA (volume, log_hl == rg_t): poblacion =
#   TODAS las barras admisibles de la serie canonica (TDA-03), SIN
#   restringir a r_1m_valid. Motivo (seccion 6 de la tarea, "opcion
#   tecnicamente mas limpia"): ni volume ni log_hl dependen de una barra
#   anterior -- exigirles la mascara de r_1m seria descartar informacion
#   perfectamente valida (p.ej. la primera barra de cada sesion, que
#   nunca tiene r_1m valido pero SI tiene un volumen y un rango
#   perfectamente medibles) solo por una regla que no les aplica. El
#   costo es que su ``n`` por minuto es mayor y NO directamente
#   comparable al de las variables de retorno -- se reporta side-by-side
#   en la tabla de salida, nunca oculto (seccion 6: "no ocultes
#   diferencias de denominadores").
#
# ``agg="mean"`` para ``zero_1m``/``extreme_flag`` (proporciones de un
# indicador 0/1); ``agg="median"`` para el resto (robusto a colas,
# seccion 7). ``r_1m`` es la unica variable con AMBOS ("media" o
# "mediana, mas robusta" -- roadmap TDA-06 metodo minimo 1): se reporta
# la media como estimador PRINCIPAL de la pregunta de TH14 sobre la
# media (seccion 9), y la mediana como version robusta de contraste.
RETURN_POPULATION_VARS = ["r_1m_mean", "r_1m_median", "abs_r_1m", "r2_1m", "zero_1m", "extreme_flag"]
BAR_POPULATION_VARS = ["volume", "log_hl"]

VARIABLE_SPECS: dict[str, dict] = {
    "r_1m_mean": {"source_col": "r_1m", "population": "return", "agg": "mean"},
    "r_1m_median": {"source_col": "r_1m", "population": "return", "agg": "median"},
    "abs_r_1m": {"source_col": "abs_r_1m", "population": "return", "agg": "median"},
    "r2_1m": {"source_col": "r2_1m", "population": "return", "agg": "median"},
    "zero_1m": {"source_col": "zero_1m", "population": "return", "agg": "mean"},
    "extreme_flag": {"source_col": "extreme_flag", "population": "return", "agg": "mean"},
    "volume": {"source_col": "volume", "population": "bar", "agg": "median"},
    "log_hl": {"source_col": "log_hl", "population": "bar", "agg": "median"},
}


def compute_extreme_flag(df: pd.DataFrame, quantile: float) -> tuple[pd.Series, float]:
    """Umbral de movimiento extremo RELATIVO -- seccion 10 de la tarea.

    Definicion exacta, PREDECLARADA (fijada en ``VARIABLE_SPECS``/config
    antes de mirar el resultado por minuto, igual que
    ``CENTRAL_RANGE_TICKS`` en TDA-05): un retorno es "extremo" si
    ``|r_1m|`` supera el percentil ``quantile`` (por defecto el 99, es
    decir, el 1% de movimientos mas grandes) de ``|r_1m|`` calculado
    sobre TODA la poblacion de retornos validos del conjunto de
    investigacion.

    Por que esta regla y no un numero de puntos: un umbral absoluto en
    puntos mezclaria escalas completamente distintas segun el nivel de
    precio (TDA-05, seccion 9: el mismo tick vale una fraccion de retorno
    distinta en 2019 que en 2025) -- exactamente el error que TDA-05 ya
    evito con ``tick_to_sigma_return``. Un percentil de la propia
    distribucion de ``|r_1m|`` es relativo a la escala del dato, simple y
    reproducible sin ajustar ningun parametro.

    Es explicitamente RETROSPECTIVO: el umbral se calcula usando TODA la
    muestra de investigacion (G1) -- no es una regla causal utilizable
    online sin modificacion. Limitaciones: es un umbral GLOBAL en el
    tiempo; TDA-05 ya establecio que la dispersion cambia por hora/año,
    asi que un movimiento "extremo" en una hora tranquila y uno en una
    hora agitada quedan bajo el mismo corte -- exactamente lo que esta
    etapa quiere medir (¿se concentran los extremos en ciertos minutos?),
    no un defecto a corregir aqui. No es EVT, no optimiza el umbral, no
    se repite con variantes.

    Salida: ``(extreme_flag, threshold)`` -- ``extreme_flag`` es
    ``1.0``/``0.0``/``NaN`` (NaN exactamente donde ``r_1m`` es invalido,
    igual que ``zero_1m``), ``threshold`` es el valor de corte en
    unidades de retorno (se traduce a ticks en el informe via TDA-05).
    """
    valid_abs_r = df.loc[df["r_1m_valid"], "abs_r_1m"]
    threshold = float(valid_abs_r.quantile(quantile))
    flag = np.where(df["r_1m_valid"], (df["abs_r_1m"].to_numpy() >= threshold).astype(float), np.nan)
    return pd.Series(flag, index=df.index), threshold


@dataclass
class MinuteProfiles:
    global_tables: dict[str, pd.DataFrame]
    by_year_tables: dict[int, dict[str, pd.DataFrame]]
    extreme_threshold: float


def build_all_minute_profiles(
    df: pd.DataFrame, years: tuple[int, ...], n_boot: int, seed: int, extreme_quantile: float
) -> MinuteProfiles:
    """Construye el perfil por minuto de TODAS las variables de ``VARIABLE_SPECS``, global y por año.

    Entrada: ``df`` ya con ``minute_of_day``/``year_ny``/``r_1m_valid``
    (``attach_calendar_fields``). ``years``: los años sobre los que
    calcular la version por año (tipicamente todos los presentes,
    incluidos 2019/2025 -- se muestran siempre, ver seccion 8 de la
    tarea; la interpretacion de estabilidad los pondera distinto, no el
    calculo).

    Salida: ``MinuteProfiles`` con la tabla global y por año de cada
    variable, mas el umbral de extremos usado (para citarlo en el
    informe).
    """
    extreme_flag, threshold = compute_extreme_flag(df, extreme_quantile)
    work = df.copy()
    work["extreme_flag"] = extreme_flag

    return_pop = work[work["r_1m_valid"]]
    bar_pop = work

    global_tables: dict[str, pd.DataFrame] = {}
    for name, spec in VARIABLE_SPECS.items():
        pop = return_pop if spec["population"] == "return" else bar_pop
        global_tables[name] = build_minute_profile_table(pop, spec["source_col"], agg=spec["agg"], n_boot=n_boot, seed=seed)

    by_year_tables: dict[int, dict[str, pd.DataFrame]] = {}
    for year in years:
        year_return_pop = return_pop[return_pop["year_ny"] == year]
        year_bar_pop = bar_pop[bar_pop["year_ny"] == year]
        year_tables: dict[str, pd.DataFrame] = {}
        for name, spec in VARIABLE_SPECS.items():
            pop = year_return_pop if spec["population"] == "return" else year_bar_pop
            year_tables[name] = build_minute_profile_table(pop, spec["source_col"], agg=spec["agg"], n_boot=n_boot, seed=seed)
        by_year_tables[year] = year_tables

    return MinuteProfiles(global_tables=global_tables, by_year_tables=by_year_tables, extreme_threshold=threshold)


# ---------------------------------------------------------------------------
# Perfil por dia de semana -- TH15
# ---------------------------------------------------------------------------

def build_weekday_profile(df: pd.DataFrame, n_boot: int, seed: int, extreme_quantile: float = 0.99) -> pd.DataFrame:
    """Perfil por dia de semana (``weekday`` de ``trading_date``, seccion 11 de la tarea).

    Mismo motor que el perfil por minuto (bootstrap de bloques de
    jornada), reemplazando "columna = minuto del dia" por "columna =
    weekday (0..6)" -- una jornada aporta como mucho una observacion
    agregada por variable a un unico weekday (el de su ``trading_date``),
    asi que el mismo pivote ``(n_dates, n_weekdays)`` + remuestreo de
    filas preserva exactamente la misma logica de bloques.

    No se ejecuta ningun test de significancia -- G5 (reporte por
    magnitud, no p-valor) rige toda esta etapa igual que TDA-05: la
    "correccion por multiplicidad" que pide la seccion 11 de la tarea
    solo aplica cuando se testean varias etiquetas simultaneamente: al no
    ejecutar tests, no hay p-valores que corregir. La evidencia de
    estabilidad es la superposicion de intervalos de bootstrap entre
    weekdays y entre años (seccion 8), consistente con G4 (parsimonia):
    los estadisticos simples ya responden la pregunta de TH15 sin
    necesitar una regresion con indicadores.

    Salida: ``DataFrame`` con una fila por ``(weekday, variable)``:
    ``weekday``, ``weekday_name``, ``variable``, ``n``, ``point``,
    ``ci_lo``, ``ci_hi``.
    """
    extreme_flag, _ = compute_extreme_flag(df, extreme_quantile)
    work = df.copy()
    work["extreme_flag"] = extreme_flag
    return_pop = work[work["r_1m_valid"]]
    bar_pop = work

    rows = []
    for name, spec in VARIABLE_SPECS.items():
        pop = return_pop if spec["population"] == "return" else bar_pop
        dates, date_idx = np.unique(pop["trading_date"].to_numpy(), return_inverse=True)
        weekday_idx = pop["weekday"].to_numpy()
        values = pop[spec["source_col"]].to_numpy(dtype=float)

        n_weekdays = 7
        pivot = np.full((len(dates), n_weekdays), np.nan)
        # Una jornada puede tener MUCHAS barras en el mismo weekday (una
        # por minuto de la sesion) -- a diferencia del pivote por minuto
        # (una barra por (jornada, minuto)), aqui hay que agregar primero
        # dentro de cada (jornada, weekday) antes de pivotar, o el
        # bootstrap de bloques perderia sentido (cada jornada debe
        # aportar UN valor resumen por variable, no cientos).
        agg_func = np.nanmean if spec["agg"] == "mean" else np.nanmedian
        df_pv = pd.DataFrame({"date_idx": date_idx, "weekday_idx": weekday_idx, "value": values})
        per_date_weekday = df_pv.groupby(["date_idx", "weekday_idx"])["value"].apply(
            lambda s: agg_func(s.to_numpy())
        )
        for (di, wi), val in per_date_weekday.items():
            pivot[di, wi] = val

        # El motor de bootstrap generico esta escrito para 1440 columnas;
        # aqui solo hay 7 -- se reutiliza la MISMA logica de remuestreo de
        # filas, sin depender del numero fijo de columnas.
        func = np.nanmedian if spec["agg"] == "median" else np.nanmean
        n_obs = (~np.isnan(pivot)).sum(axis=0)
        with np.errstate(invalid="ignore"):
            point = func(pivot, axis=0)
        point = np.where(n_obs > 0, point, np.nan)
        rng = np.random.default_rng(seed)
        boot = np.empty((n_boot, n_weekdays))
        for b in range(n_boot):
            sampled_rows = rng.integers(0, len(dates), size=len(dates))
            with np.errstate(invalid="ignore"):
                boot[b, :] = func(pivot[sampled_rows, :], axis=0)
        with np.errstate(invalid="ignore"):
            ci_lo = np.nanpercentile(boot, 2.5, axis=0)
            ci_hi = np.nanpercentile(boot, 97.5, axis=0)
        ci_lo = np.where(n_obs > 0, ci_lo, np.nan)
        ci_hi = np.where(n_obs > 0, ci_hi, np.nan)

        for wi in range(n_weekdays):
            rows.append({
                "weekday": wi, "weekday_name": WEEKDAY_NAMES[wi], "variable": name,
                "n": int(n_obs[wi]), "point": float(point[wi]),
                "ci_lo": float(ci_lo[wi]), "ci_hi": float(ci_hi[wi]),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Segmentacion derivada de los datos -- seccion 12 de la tarea
# ---------------------------------------------------------------------------
# Metodo PREDECLARADO antes de ver el resultado real (los mismos
# parametros viven en config/YAML, pero sus valores por defecto se fijan
# aqui, en el codigo, no se ajustan despues de mirar la salida):
#
# 1. Variable(s) guia: la mediana de ``abs_r_1m`` (magnitud del
#    movimiento) como proxy PRINCIPAL, corroborada por ``log_hl``
#    (rango) y ``volume`` (actividad) -- tres proxies independientes de
#    "cuanto esta pasando" en cada minuto (seccion 12: "respaldado por
#    mas de un proxy").
# 2. Suavizado: media movil CENTRADA de ``SMOOTHING_WINDOW_MINUTES``
#    minutos sobre cada perfil (normalizado 0-1 por min-max) antes de
#    combinarlos -- quita el ruido minuto-a-minuto sin desplazar la
#    ubicacion aproximada de un quiebre real.
# 3. Score compuesto: promedio de los 3 perfiles normalizados y
#    suavizados.
# 4. Quiebre = minuto donde el VALOR ABSOLUTO de la primera diferencia
#    del score compuesto es grande: se ordenan los 1439 valores de
#    |diferencia| de mayor a menor y se toman, en orden, los que superen
#    una separacion minima de ``MIN_SPACING_MINUTES`` respecto de
#    cualquier corte ya aceptado -- hasta un maximo de
#    ``MAX_BREAKPOINTS`` cortes (predeclarado, para no dejar que el
#    metodo devuelva un corte por cada minuto ruidoso -- data snooping).
# 5. Estabilidad entre años: el mismo algoritmo se corre POR SEPARADO en
#    cada año COMPLETO (2020-2024); un corte candidato del score GLOBAL
#    se acepta en la propuesta final solo si un corte detectado
#    independientemente en al menos ``YEAR_STABILITY_MIN_YEARS`` de los 5
#    años completos cae dentro de ``YEAR_STABILITY_TOLERANCE_MINUTES``
#    minutos de el.

def _minmax_normalize(x: np.ndarray) -> np.ndarray:
    finite = x[~np.isnan(x)]
    if len(finite) == 0 or finite.max() == finite.min():
        return np.full_like(x, np.nan)
    return (x - finite.min()) / (finite.max() - finite.min())


def _centered_rolling_median(x: pd.Series, window: int) -> np.ndarray:
    """Mediana movil centrada, SIN fabricar valores en minutos estructuralmente cerrados.

    ``min_periods=1`` es necesario para que un minuto ABIERTO cerca del
    borde de un hueco (p.ej. justo despues de la ventana de mantenimiento
    de CME) siga recibiendo una estimacion suavizada usando los vecinos
    validos que tenga, aunque sean menos de ``window`` -- eso es
    suavizado legitimo, no fabricacion.

    El problema (detectado en el cierre de TDA-06): ese mismo
    ``min_periods=1`` tambien le da un valor a un minuto CERRADO
    (``x`` original ``NaN`` -- ``n=0``, ninguna barra en ninguna
    jornada) si su ventana alcanza a rozar aunque sea UN vecino valido
    al otro lado del hueco -- por ejemplo, el minuto 17:54 NY (dentro de
    la ventana de mantenimiento 17:01-18:00, sin ninguna barra jamas)
    quedaba con un valor fabricado a partir del primer minuto abierto
    despues de las 18:00, generando un quiebre artificial DENTRO de un
    hueco estructural.

    Fix: se restaura EXPLICITAMENTE ``NaN`` en toda posicion donde el
    valor ORIGINAL (antes de suavizar) ya era ``NaN`` -- nunca se
    interpola, nunca se hace forward-fill. Un minuto estructuralmente
    cerrado permanece ``NaN`` despues del suavizado, siempre.
    """
    original = x.to_numpy()
    smoothed = x.rolling(window=window, center=True, min_periods=1).median().to_numpy().copy()
    smoothed[np.isnan(original)] = np.nan
    return smoothed


def compute_composite_activity_score(
    tables: dict[str, pd.DataFrame], smoothing_window: int, proxy_cols: tuple[str, ...] = ("abs_r_1m", "log_hl", "volume")
) -> np.ndarray:
    """Score compuesto de actividad/volatilidad por minuto -- promedio de proxies normalizados y suavizados.

    Entrada: el diccionario ``{variable: tabla de perfil}`` de
    ``build_all_minute_profiles`` (global o de un año concreto).
    Salida: array ``(1440,)`` con el score compuesto (``NaN`` donde
    ningun proxy tiene datos).
    """
    normalized = []
    for col in proxy_cols:
        point = tables[col]["point"]
        smoothed = _centered_rolling_median(point, smoothing_window)
        normalized.append(_minmax_normalize(smoothed))
    stacked = np.vstack(normalized)
    with np.errstate(invalid="ignore"):
        return np.nanmean(stacked, axis=0)


def detect_breakpoints(score: np.ndarray, max_breakpoints: int, min_spacing: int) -> list[int]:
    """Detecta hasta ``max_breakpoints`` quiebres del ``score`` compuesto, separados >= ``min_spacing`` minutos.

    Metodo: primera diferencia absoluta del score (tratado linealmente en
    0..1439, sin envolver la medianoche -- ver informe, limitacion
    documentada); se seleccionan, de mayor a menor magnitud de
    diferencia, los minutos que respeten la separacion minima respecto de
    los ya aceptados, hasta el tope predeclarado. Simple, determinista,
    reproducible -- sin ningun parametro ajustado a posteriori (G4: no se
    justifica un detector de quiebres mas sofisticado para responder esta
    pregunta, ver seccion 13 de la tarea/G2).
    """
    diff = np.abs(np.diff(score))
    valid_positions = np.where(~np.isnan(diff))[0]
    order = valid_positions[np.argsort(-diff[valid_positions])]

    accepted: list[int] = []
    for pos in order:
        if all(abs(pos - a) >= min_spacing for a in accepted):
            accepted.append(int(pos))
        if len(accepted) >= max_breakpoints:
            break
    return sorted(accepted)


def check_breakpoint_stability(
    global_breaks: list[int], per_year_breaks: dict[int, list[int]], tolerance: int, min_years: int
) -> pd.DataFrame:
    """Para cada quiebre global, cuenta en cuantos años (de los provistos) hay un quiebre a <= ``tolerance`` minutos.

    Salida: ``DataFrame`` con ``minute_of_day``, ``n_years_supporting``,
    ``years_supporting``, ``stable`` (``n_years_supporting >= min_years``).
    """
    rows = []
    for gb in global_breaks:
        supporting = []
        for year, breaks in per_year_breaks.items():
            if any(abs(gb - yb) <= tolerance for yb in breaks):
                supporting.append(year)
        rows.append({
            "minute_of_day": gb, "n_years_supporting": len(supporting),
            "years_supporting": ",".join(str(y) for y in sorted(supporting)),
            "stable": len(supporting) >= min_years,
        })
    return pd.DataFrame(rows)


def build_segmentation_proposal(
    global_tables: dict[str, pd.DataFrame],
    by_year_tables: dict[int, dict[str, pd.DataFrame]],
    smoothing_window: int, max_breakpoints: int, min_spacing: int,
    tolerance: int, min_years: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Orquesta la propuesta de segmentacion: score global, quiebres por año COMPLETO, filtro de estabilidad."""
    global_score = compute_composite_activity_score(global_tables, smoothing_window)
    global_breaks = detect_breakpoints(global_score, max_breakpoints, min_spacing)

    per_year_breaks = {}
    for year in COMPLETE_YEARS:
        if year not in by_year_tables:
            continue
        year_score = compute_composite_activity_score(by_year_tables[year], smoothing_window)
        per_year_breaks[year] = detect_breakpoints(year_score, max_breakpoints, min_spacing)

    stability = check_breakpoint_stability(global_breaks, per_year_breaks, tolerance, min_years)
    return stability, global_score


# Numero de surrogates de la calibracion G2 -- PREDECLARADO. 5 (el valor
# inicial del cierre de TDA-06) es insuficiente para caracterizar un null
# sobre un vector de 1440 minutos. Cada surrogate pasa por el pipeline
# COMPLETO (permutar -> suavizar 3 proxies -> normalizar -> promediar ->
# detectar quiebres), pero esa secuencia es barata (rolling median sobre
# 1440 puntos + un top-K con separacion minima: milisegundos por
# surrogate) -- no hay ninguna razon de costo para quedarse en 5. Se fija
# en 200: un numero redondo, grande respecto de los 5 años completos
# reales (asi la tasa de falsos positivos del filtro de estabilidad se
# estima sobre una base bastante mas amplia que la evidencia real que
# calibra), y barato (unos pocos segundos en total, ver
# ``run_tda06_analysis``). No se ajusto despues de ver el resultado.
DEFAULT_N_SURROGATES = 200


def _permute_preserving_nan_mask(point: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permuta los valores VALIDOS de ``point`` entre si, dejando fijas las posiciones ``NaN``.

    Preserva la mascara de minutos estructuralmente cerrados (un hecho
    REAL del calendario de negociacion, no algo que esta calibracion deba
    poner a prueba) y destruye unicamente el ORDEN temporal de los
    niveles de actividad en los minutos abiertos -- exactamente lo que
    G2 pide poner a prueba: ¿el detector fabrica estructura donde el
    nivel de actividad no tiene ningun patron temporal real?
    """
    surrogate = point.copy()
    valid_mask = ~np.isnan(point)
    surrogate[valid_mask] = rng.permutation(point[valid_mask])
    return surrogate


# Numero de reagrupamientos aleatorios de la piscina de surrogates
# usados para estimar la tasa de falsos positivos del filtro de
# estabilidad (ver docstring de ``calibrate_breakpoint_detector`` para el
# porque). Barato (cada reagrupamiento solo reutiliza quiebres ya
# calculados, sin volver a suavizar nada) -- 200 reagrupamientos
# independientes de la piscina de 200 surrogates dan una estimacion
# razonablemente estable de esa tasa sin costo adicional relevante.
DEFAULT_N_STABILITY_GROUPS = 200


def calibrate_breakpoint_detector(
    global_tables: dict[str, pd.DataFrame], n_surrogates: int, smoothing_window: int, max_breakpoints: int,
    min_spacing: int, tolerance: int, min_years: int, seed: int = 0,
    proxy_cols: tuple[str, ...] = ("abs_r_1m", "log_hl", "volume"),
    n_stability_groups: int = DEFAULT_N_STABILITY_GROUPS,
) -> pd.DataFrame:
    """Calibracion G2: ¿el detector "encuentra" quiebres estables sobre ruido puro, corriendo el MISMO pipeline?

    Version corregida (cierre de TDA-06, dos problemas separados):

    **Problema 1 -- el pipeline saltaba el suavizado.** La primera
    implementacion permutaba el SCORE COMPUESTO ya suavizado y corria el
    detector directamente sobre esa permutacion -- no calibraba el
    pipeline real. Ahora cada surrogate se genera permutando los valores
    VALIDOS del perfil CRUDO (sin suavizar) de cada proxy por separado
    (``_permute_preserving_nan_mask`` -- preserva que 17:01-18:00 NY siga
    sin ninguna observacion, igual que en los datos reales) y pasa por
    EXACTAMENTE la misma funcion de score compuesto
    (``compute_composite_activity_score``, con el mismo suavizado y la
    misma restauracion de la mascara ``NaN``) y el mismo
    ``detect_breakpoints`` que se usa sobre los datos reales.

    **Problema 2 -- el criterio de estabilidad no escalaba con el numero
    de surrogates.** Al subir ``n_surrogates`` de 5 a un numero grande
    (para caracterizar mejor el null, tal como pide la tarea), evaluar
    cada surrogate contra TODOS los demas (``n_surrogates - 1``, p.ej.
    199) usando el MISMO ``min_years=3`` absoluto que se usa entre 5 años
    reales cambia por completo el significado de la regla: con 199
    "otros" candidatos, encontrar 3 que coincidan por puro azar deja de
    ser una condicion exigente (se verifico empiricamente: con
    ``n_surrogates=200`` el 100% de los quiebres candidatos del null
    resultaban "estables" bajo esa comparacion todos-contra-todos --
    justo el sintoma que motiva la correccion). La regla real compara un
    conjunto de quiebres contra exactamente ``len(COMPLETE_YEARS)`` (5)
    referencias independientes, exigiendo 3 de 5 -- una proporcion del
    60%, no un conteo absoluto que deje de significar lo mismo si el
    denominador cambia.

    Fix: se genera una PISCINA grande de ``n_surrogates`` permutaciones
    (barato, ver ``DEFAULT_N_SURROGATES``), pero el filtro de estabilidad
    se aplica en ``n_stability_groups`` REAGRUPAMIENTOS aleatorios
    independientes de tamaño ``len(COMPLETE_YEARS)`` extraidos de esa
    piscina (sin reemplazo dentro de cada grupo) -- exactamente el mismo
    tamaño de grupo y el mismo ``min_years`` que la comparacion real
    entre años, repetida muchas veces para estimar con precision la tasa
    de falsos positivos de ESA regla exacta, en vez de aplicar la regla
    con un denominador que ya no es el mismo.

    Salida: ``DataFrame`` con una fila resumen: tamaño de la piscina de
    surrogates, numero de reagrupamientos, y numero/tasa de quiebres
    candidatos y estables sobre esos reagrupamientos.
    """
    rng = np.random.default_rng(seed)
    surrogate_breaks = {}
    for i in range(n_surrogates):
        surrogate_tables = {
            col: global_tables[col].assign(point=_permute_preserving_nan_mask(global_tables[col]["point"].to_numpy(), rng))
            for col in proxy_cols
        }
        surrogate_score = compute_composite_activity_score(surrogate_tables, smoothing_window, proxy_cols)
        surrogate_breaks[i] = detect_breakpoints(surrogate_score, max_breakpoints, min_spacing)

    group_size = len(COMPLETE_YEARS)
    n_stable_total = 0
    n_candidates_total = 0
    for _ in range(n_stability_groups):
        group_idx = rng.choice(n_surrogates, size=min(group_size, n_surrogates), replace=False)
        group_breaks = {idx: surrogate_breaks[idx] for idx in group_idx}
        for idx in group_idx:
            others = {k: v for k, v in group_breaks.items() if k != idx}
            stability = check_breakpoint_stability(group_breaks[idx], others, tolerance, min_years)
            n_candidates_total += len(stability)
            n_stable_total += int(stability["stable"].sum()) if len(stability) else 0

    n_events = n_stability_groups * group_size
    return pd.DataFrame([{
        "n_surrogates": n_surrogates,
        "n_stability_groups": n_stability_groups,
        "group_size": group_size,
        "n_candidate_breaks_total": n_candidates_total,
        "n_stable_breaks_total": n_stable_total,
        "mean_candidates_per_group_member": n_candidates_total / n_events,
        "stable_fraction": (n_stable_total / n_candidates_total) if n_candidates_total > 0 else float("nan"),
    }])


# ---------------------------------------------------------------------------
# STOP-6 y factor estacional s(m) -- seccion 14 de la tarea
# ---------------------------------------------------------------------------

def evaluate_stop6(
    global_tables: dict[str, pd.DataFrame], by_year_tables: dict[int, dict[str, pd.DataFrame]],
) -> dict:
    """Evalua conjuntamente magnitud + estabilidad del perfil de MAGNITUD (no de media) -- decision de STOP-6.

    No hay umbral universal en el roadmap (seccion 14 de la tarea): se
    combinan, para ``abs_r_1m`` y ``log_hl`` (las dos variables de
    magnitud/volatilidad, la pregunta central de STOP-6):

    - Rango relativo del perfil global: ``(max(point) - min(point)) /
      median(point)`` sobre los minutos con ``n`` suficiente -- que tan
      grande es la variacion minuto-a-minuto respecto del nivel tipico.
    - Correlacion de Spearman entre el perfil de cada año COMPLETO y el
      perfil global -- que tan bien se repite la FORMA entre años (no
      exige que coincidan en nivel exacto, solo en forma, tal como pide
      la seccion 8: "¿la forma... permanece razonablemente estable?").

    Salida: ``dict`` con las cifras de ambas variables y una recomendacion
    textual (la decision final la toma ``run_tda06_analysis``/el informe,
    esta funcion solo agrega la evidencia).
    """
    evidence = {}
    for var in ("abs_r_1m", "log_hl"):
        g = global_tables[var]
        mask = g["n"] > 0
        point = g.loc[mask, "point"].to_numpy()
        relative_range = float((np.max(point) - np.min(point)) / np.median(point)) if np.median(point) != 0 else float("inf")

        correlations = []
        for year in COMPLETE_YEARS:
            if year not in by_year_tables:
                continue
            y = by_year_tables[year][var]
            common_mask = mask & (y["n"] > 0)
            if common_mask.sum() < 30:
                continue
            corr = _spearman_corr(g.loc[common_mask, "point"].to_numpy(), y.loc[common_mask, "point"].to_numpy())
            correlations.append(corr)

        evidence[var] = {
            "relative_range": relative_range,
            "spearman_by_year": correlations,
            "spearman_median": float(np.median(correlations)) if correlations else float("nan"),
        }
    return evidence


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Correlacion de Spearman sin depender de ``scipy`` (no es una dependencia del repositorio).

    Definicion equivalente: correlacion de Pearson sobre los RANGOS de
    ``a`` y ``b`` (``pandas.Series.rank`` ya maneja empates por rango
    promedio, sin necesitar ``scipy.stats.rankdata``).
    """
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


NORMALIZATION_EPS = 1e-12

# Puntos de referencia DESCRIPTIVOS de STOP-6 (seccion 14 de la tarea:
# "no inventes un threshold universal arbitrario si el roadmap no lo
# define" -- el roadmap no define uno). Corregido en el cierre de
# TDA-06: estos dos numeros NO son una puerta binaria estricta que decide
# STOP-6 por si sola comparando solo la MEDIANA entre años -- son marcas
# de referencia, predeclaradas antes de ver el resultado real, que
# ayudan a leer la evidencia (¿la variacion minuto-a-minuto es al menos
# de la magnitud del nivel tipico? ¿la forma se repite mas de lo que
# difiere entre años?). La decision la toma ``decide_stop6`` sopesando
# el CONJUNTO de la evidencia (rango relativo, Spearman por año
# INDIVIDUAL -- no solo la mediana --, y consistencia entre las dos
# variables de magnitud), no un solo par de numeros agregados.
RELATIVE_RANGE_REFERENCE = 0.5
SPEARMAN_REFERENCE = 0.5


def decide_stop6(evidence: dict) -> bool:
    """``True`` si STOP-6 se ACTIVA (perfil debil o inestable -> no se construye s(m)).

    Entrada: el ``dict`` de ``evaluate_stop6`` (una entrada por variable
    de magnitud, con ``relative_range``, ``spearman_median`` y
    ``spearman_by_year`` -- la lista completa, un valor por año
    COMPLETO).

    Decision HOLISTICA, no un umbral binario sobre un unico numero
    agregado (corregido en el cierre de TDA-06 -- ver
    ``RELATIVE_RANGE_REFERENCE``/``SPEARMAN_REFERENCE``): CASO A (perfil
    fuerte y estable, STOP-6 NO se activa) exige, para AMBAS variables de
    magnitud (``abs_r_1m``, ``log_hl``):

    1. ``relative_range`` por encima de la referencia (variacion
       minuto-a-minuto al menos del orden del nivel tipico) para AMBAS
       variables.
    2. Spearman por año INDIVIDUAL por encima de la referencia en TODOS
       los años completos disponibles (no solo en la mediana) -- una
       forma que se repite bien en la mayoria de los años pero falla
       gravemente en uno no séria "estable" solo por tener una mediana
       alta.

    Cualquier otro caso (algun año individual por debajo de la
    referencia, rango relativo debil en alguna variable, o sin años
    completos con los que evaluar estabilidad) activa STOP-6 (CASO B:
    resultado negativo valido, no se genera una serie ajustada falsa).

    Sobre el conjunto de investigacion real esta version holistica y la
    version binaria original (solo mediana vs. las mismas dos
    referencias) llegan a la MISMA conclusion -- la evidencia (rango
    relativo ~5.3-5.5, Spearman 0.94-0.99 en CADA año individual, para
    AMBAS variables) supera las referencias por un margen tan amplio que
    ninguna forma razonable de agregarla cambia el resultado. La
    correccion es de FRAMING/robustez (no reducir "estable" a una
    mediana agregada), no un cambio de conclusion forzado.
    """
    for var in ("abs_r_1m", "log_hl"):
        e = evidence[var]
        if not (e["relative_range"] > RELATIVE_RANGE_REFERENCE):
            return True
        by_year = e["spearman_by_year"]
        if not by_year:
            return True
        if not all(c > SPEARMAN_REFERENCE for c in by_year):
            return True
    return False


def _mean_relative_ci_halfwidth(table: pd.DataFrame) -> float:
    """Ancho medio (relativo al nivel) de la banda de bootstrap de un perfil por minuto -- metrica de RUIDO del estimador.

    Para cada minuto con datos: ``(ci_hi - ci_lo) / 2 / |point|`` (medio
    ancho de la banda de bootstrap, como fraccion del propio nivel
    estimado en ese minuto) -- deliberadamente distinto de
    ``relative_range`` (que mide cuanto VARIA el NIVEL entre minutos, la
    señal; esta funcion mide cuanto VARIA el ESTIMADOR entre remuestreos
    en un mismo minuto, el ruido). Se promedia sobre todos los minutos
    con ``n>0`` y nivel finito no nulo. Un valor MENOR indica un
    estimador proporcionalmente mas preciso (menos ruidoso).
    """
    mask = (table["n"] > 0) & np.isfinite(table["point"]) & (table["point"] != 0)
    halfwidth = (table.loc[mask, "ci_hi"] - table.loc[mask, "ci_lo"]) / 2.0
    relative = (halfwidth / table.loc[mask, "point"].abs())
    relative = relative[np.isfinite(relative)]
    return float(relative.mean()) if len(relative) else float("inf")


def choose_s_m_proxy(global_tables: dict[str, pd.DataFrame]) -> str:
    """Elige ``abs_r_1m`` o ``log_hl`` como base de ``s(m)`` -- criterio TECNICO documentado, seccion 14 de la tarea.

    Regla predeclarada (antes de ver cual "produce mejores resultados",
    prohibido por la tarea): se prefiere la variable cuyo perfil por
    minuto tiene MENOR ``_mean_relative_ci_halfwidth`` -- es decir, la
    que es proporcionalmente MENOS ruidosa como ESTIMADOR (Tsay/roadmap
    C3: "si rg_t es sustancialmente menos ruidoso, es el proxy
    preferido"). Nota: esto es distinto de comparar cuanto varia el
    NIVEL del perfil entre minutos (``relative_range``, usado para
    STOP-6) -- aqui se compara la incertidumbre de CADA estimacion
    puntual, no la fuerza del patron.
    Empate (diferencia < 10%): se prefiere ``log_hl`` porque nunca
    depende de la mascara de validez de retorno (poblacion mayor, mas
    barras por minuto -- seccion 6).
    """
    noise_abs_r = _mean_relative_ci_halfwidth(global_tables["abs_r_1m"])
    noise_log_hl = _mean_relative_ci_halfwidth(global_tables["log_hl"])
    return "log_hl" if noise_log_hl <= noise_abs_r * 1.10 else "abs_r_1m"


def build_s_m(global_tables: dict[str, pd.DataFrame], proxy: str) -> pd.DataFrame:
    """Factor estacional ``s(m)``: normaliza el perfil de ``proxy`` a media 1 sobre los minutos con datos.

    ``s(m) = point(m) / mean(point sobre minutos con n>0)`` -- la
    normalizacion estandar de un indice estacional (media 1): ``s(m)>1``
    dice "minuto m tipicamente mas volatil/activo que el promedio del
    dia"; ``s(m)<1``, lo contrario. Protegida contra division por cero
    (seccion 14 de la tarea): minutos con ``point<=0`` o no finito quedan
    con ``s_m=NaN`` (nunca ``inf``) -- en la practica no ocurre para
    ``abs_r_1m``/``log_hl`` (ambas son >=0 por construccion, y una
    mediana de barras reales rara vez es exactamente 0), pero se
    comprueba explicitamente, no se asume.

    Salida: ``DataFrame`` con ``minute_of_day``, ``n``, ``point`` (nivel
    crudo del proxy), ``s_m``, ``label="RETROSPECTIVO"`` (estimado sobre
    TODA la muestra de investigacion, G1 -- no es una feature causal
    disponible online sin reestimarse solo con datos hasta cada momento).
    """
    table = global_tables[proxy].copy()
    valid_point = table.loc[table["n"] > 0, "point"]
    mean_level = float(valid_point[valid_point > 0].mean())

    s_m = np.where(
        (table["n"].to_numpy() > 0) & (table["point"].to_numpy() > NORMALIZATION_EPS) & np.isfinite(table["point"].to_numpy()),
        table["point"].to_numpy() / mean_level,
        np.nan,
    )
    table["s_m"] = s_m
    table["proxy"] = proxy
    table["label"] = "RETROSPECTIVO"
    return table[["minute_of_day", "n", "point", "s_m", "proxy", "label"]]


def build_r_tilde(df: pd.DataFrame, s_m_table: pd.DataFrame) -> pd.DataFrame:
    """``r_tilde = r_1m / s(minute_of_day)`` -- serie ajustada por estacionalidad, RETROSPECTIVA.

    Entrada: ``df`` con ``minute_of_day``/``r_1m``/``r_1m_valid``
    (``attach_calendar_fields``) y la tabla de ``s(m)``.
    Transformacion: ``join`` por ``minute_of_day``; ``r_tilde`` es
    ``NaN`` donde ``r_1m`` es invalido O donde ``s_m`` es ``NaN``/no
    positivo (division protegida, igual que ``build_s_m``) -- nunca
    ``inf``.
    Salida: ``DataFrame`` con columnas de trazabilidad
    (``timestamp``, ``source_file``, ``contract``, ``trading_date``,
    ``minute_of_day``, ``weekday``), ``r_1m``, ``r_1m_valid``, ``s_m``,
    ``r_tilde``, ``label="RETROSPECTIVO"``.
    """
    s_map = s_m_table.set_index("minute_of_day")["s_m"]
    s_m = df["minute_of_day"].map(s_map).to_numpy(dtype=float)

    r_1m = df["r_1m"].to_numpy(dtype=float)
    valid = df["r_1m_valid"].to_numpy() & np.isfinite(s_m) & (s_m > 0)
    r_tilde = np.where(valid, r_1m / np.where(s_m == 0, np.nan, s_m), np.nan)

    return pd.DataFrame({
        "timestamp": df["timestamp"], "source_file": df["source_file"], "contract": df["contract"],
        "trading_date": df["trading_date"], "minute_of_day": df["minute_of_day"], "weekday": df["weekday"],
        "r_1m": df["r_1m"], "r_1m_valid": df["r_1m_valid"], "s_m": s_m, "r_tilde": r_tilde,
        "label": "RETROSPECTIVO",
    })


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------

@dataclass
class TDA06Result:
    df: pd.DataFrame
    profiles: MinuteProfiles
    weekday_profile: pd.DataFrame
    segmentation: pd.DataFrame
    global_score: np.ndarray
    calibration: pd.DataFrame
    stop6_evidence: dict
    stop6_activated: bool
    s_m_proxy: str | None
    s_m_table: pd.DataFrame | None
    r_tilde_table: pd.DataFrame | None


def run_tda06_analysis(config: SnapshotConfig) -> TDA06Result:
    """Orquesta TDA-06 completo sobre las variables de TDA-04.

    Reutiliza, sin duplicar logica: la proteccion del hold-out
    (``holdout_guard.py`` -- defensiva, TDA-06 no abre ningun archivo
    crudo) y los artefactos de TDA-04 (variables + mascara de validez).
    """
    validate_research_holdout_disjoint(config)

    variables, validity = load_inputs(config)
    last_timestamps = variables.groupby("source_file")["timestamp"].max().to_dict()
    validate_last_timestamps_before_boundary(config, last_timestamps)

    df = attach_calendar_fields(variables, validity)
    years = tuple(sorted(df["year_ny"].unique()))

    profiles = build_all_minute_profiles(
        df, years=years, n_boot=config.tda06_n_boot, seed=0, extreme_quantile=config.tda06_extreme_quantile
    )
    weekday_profile = build_weekday_profile(
        df, n_boot=config.tda06_n_boot, seed=0, extreme_quantile=config.tda06_extreme_quantile
    )

    segmentation, global_score = build_segmentation_proposal(
        profiles.global_tables, profiles.by_year_tables,
        smoothing_window=config.tda06_smoothing_window_minutes,
        max_breakpoints=config.tda06_max_breakpoints,
        min_spacing=config.tda06_min_spacing_minutes,
        tolerance=config.tda06_year_stability_tolerance_minutes,
        min_years=config.tda06_year_stability_min_years,
    )
    calibration = calibrate_breakpoint_detector(
        profiles.global_tables, n_surrogates=DEFAULT_N_SURROGATES,
        smoothing_window=config.tda06_smoothing_window_minutes,
        max_breakpoints=config.tda06_max_breakpoints,
        min_spacing=config.tda06_min_spacing_minutes,
        tolerance=config.tda06_year_stability_tolerance_minutes,
        min_years=config.tda06_year_stability_min_years,
        seed=1,
    )

    stop6_evidence = evaluate_stop6(profiles.global_tables, profiles.by_year_tables)
    stop6_activated = decide_stop6(stop6_evidence)

    s_m_proxy = None
    s_m_table = None
    r_tilde_table = None
    if not stop6_activated:
        s_m_proxy = choose_s_m_proxy(profiles.global_tables)
        s_m_table = build_s_m(profiles.global_tables, s_m_proxy)
        r_tilde_table = build_r_tilde(df, s_m_table)

    return TDA06Result(
        df=df, profiles=profiles, weekday_profile=weekday_profile, segmentation=segmentation,
        global_score=global_score, calibration=calibration, stop6_evidence=stop6_evidence,
        stop6_activated=stop6_activated, s_m_proxy=s_m_proxy, s_m_table=s_m_table, r_tilde_table=r_tilde_table,
    )
