"""TDA-05 -- Resolucion efectiva y discrecion del retorno de 1 minuto.

Implementa la etapa TDA-05 del roadmap
(``docs/methodology/Tsay_OHLCV_analysis_roadmap.md``, seccion "TDA-05"):
determina en que medida el retorno de 1 minuto ``r_1m`` (TDA-04) es una
variable EFECTIVAMENTE DISCRETA -- porque el precio de MNQ solo puede
moverse en multiplos de un tick de 0.25 puntos -- y como cambia esa
discrecion por hora del dia (America/New_York), por año y por combinacion
año×hora.

Que es un tick, en una frase: es el escalon minimo de precio que el
contrato puede cotizar (0.25 puntos para MNQ, especificacion externa del
contrato -- no inferida de los datos, ver ``configs/mnq_snapshot.yaml``).
Ningun precio Close puede diferir de otro Close en una cantidad que no
sea un multiplo entero de esa cantidad -- TDA-00 ya lo certifico sobre
CADA precio individual (0 violaciones de grilla de tick en todo el
conjunto de investigacion); esta etapa comprueba la consecuencia directa:
la DIFERENCIA entre dos precios en la grilla tambien debe ser un multiplo
entero del tick.

Por que el retorno puede ser decimal aunque el precio se mueva en
escalones: ``r_1m = ln(C_t/C_{t-1})`` es una funcion continua de dos
numeros que, individualmente, si estan restringidos a una grilla -- pero
el COCIENTE de dos multiplos de 0.25 no tiene por que ser, el mismo, un
numero "redondo". Por eso ``r_1m`` en si mismo parece tomar valores
decimales "distintos" en cada fila, incluso cuando el movimiento de
precio subyacente es siempre el mismo conjunto de escalones enteros de
tick -- es exactamente la ilusion de continuidad que esta etapa
desenmascara, comparando ``r_1m`` contra el movimiento en TICKS.

Que significa "variable efectivamente discreta": que, dentro de un
segmento de analisis (por ejemplo, una hora del dia, un año o una
combinacion año×hora), el retorno queda fuertemente condicionado por los
escalones discretos del precio: puede existir una fraccion importante de
barras sin movimiento (0 ticks) y una concentracion elevada de movimientos
pequeños, como ±1 o ±2 ticks. Cuando esto ocurre, los momentos, cuantiles,
histogramas y QQ-plots de etapas posteriores (TDA-07 en adelante) deben
interpretarse teniendo en cuenta que la variable subyacente no se comporta
como una variable perfectamente continua.

Que NO hace este modulo (deliberadamente, fuera de alcance de TDA-05):

- No estudia el perfil minuto-a-minuto completo (eso es TDA-06).
- No estudia la distribucion marginal, momentos, cuantiles ni QQ-plots de
  ``r_1m`` (eso es TDA-07).
- No calcula ACF/PACF ni busca dependencia.
- No repite el analisis a 5/10 minutos (metodo OPCIONAL del roadmap,
  diferido explicitamente a TDA-08).
- No retoma TH10 (sigue diferida).
- No absorbe la comparacion distribucional completa de TH08 (queda para
  TDA-07); solo usa lo estrictamente necesario para interpretar TH09.
- No convierte ``CANDIDATO_FORWARD_FILL`` (TDA-02) en forward-fill
  confirmado, ni lo excluye como si fuera sintetico.
- No introduce ninguna segmentacion de sesion (RTH, pre-market, etc.):
  agrupa por hora NY literal (0..23), por año y por combinacion año×hora,
  sin crear regímenes horarios nuevos.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ohlcv_dataroad.config import SnapshotConfig
from ohlcv_dataroad.ingest.holdout_guard import (
    validate_last_timestamps_before_boundary,
    validate_research_holdout_disjoint,
)
from ohlcv_dataroad.ingest.session_calendar import NY_TZ

# --- Rango central para "numero de valores distintos" (seccion 11) -------
# Definido UNA sola vez, ANTES de mirar ningun resultado, tal como exige
# la tarea ("elige UNA definicion sencilla; fijala antes de interpretar
# los resultados; no pruebes multiples rangos"). Se expresa en TICKS (no
# en percentiles de r_1m ni en puntos): +-5 ticks es un rango simetrico,
# redondo, centrado en cero, elegido sin mirar los datos primero -- NO
# porque cubra la mayoria de las barras: en el conjunto de investigacion
# real cubre ~40% de las barras validas (``prop_abs_le_5_ticks``, ver
# informe, seccion de resultados), una minoria sustancial pero lejos de
# ser la totalidad.
CENTRAL_RANGE_TICKS = 5

# Tolerancia numerica para el chequeo de que delta_close_ticks es entero
# -- la misma que usa TDA-00 para la grilla de tick de un precio
# individual (``tda00_integrity.TICK_GRID_TOLERANCE``), reutilizada aqui
# por coherencia: la diferencia de dos multiplos de tick con el mismo
# margen de error de punto flotante que cada uno por separado.
TICK_GRID_TOLERANCE = 1e-6

# Umbrales PURAMENTE DESCRIPTIVOS para armar una lista de segmentos
# candidatos a revision manual en la evaluacion de STOP-5 (seccion 14).
# NO son un criterio de exclusion automatica -- el roadmap prohibe
# explicitamente inventar un corte del tipo "tick/sigma > X => excluir".
# Se usan solo para que el informe no tenga que inspeccionar a mano cada
# fila de las tablas por hora/año: marcan qué filas merecen la lectura
# atenta de la seccion STOP-5, nada mas.
WATCHLIST_MIN_ZERO_FRACTION = 0.50
WATCHLIST_MAX_TICK_TO_SIGMA = 2.0


class TickGridInconsistencyError(Exception):
    """``delta_close_ticks`` no es entero dentro de tolerancia para alguna fila valida.

    TDA-00 certifico que CADA precio Close individual cae en la grilla de
    tick (0 violaciones, ``tick_grid_violation`` en todo el conjunto de
    investigacion). Si dos Close en la grilla producen una diferencia que
    NO es multiplo entero del tick, hay una inconsistencia real entre lo
    que TDA-00 certifico y lo que TDA-05 observa -- posiblemente un error
    de punto flotante mayor al esperado, o una fuga en alguna
    transformacion previa. La tarea es explicita: no redondear en
    silencio, detener el analisis y reportarlo.
    """


# ---------------------------------------------------------------------------
# Carga de entradas (TDA-04 + TDA-02, sin recalcular nada de golpe)
# ---------------------------------------------------------------------------

def load_inputs(config: SnapshotConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lee las variables y la mascara de validez de TDA-04, y la mascara de barras inactivas de TDA-02.

    Entrada: la configuracion (rutas ya declaradas por TDA-02/TDA-04, no
    redeclaradas aqui).
    Transformacion: ninguna -- lectura directa de los tres parquet, cada
    uno ordenado por su clave natural.
    Salida: ``(variables, validity, inactive_mask)``. TDA-05 nunca abre
    ningun archivo de ``data/raw/`` ni de ``config.holdout_files``: su
    entrada son exclusivamente estos tres artefactos, ya construidos
    sobre el conjunto de investigacion por etapas anteriores.
    """
    variables = pd.read_parquet(config.tda04_variables_parquet_path).sort_values("timestamp").reset_index(drop=True)
    validity = pd.read_parquet(config.tda04_validity_mask_parquet_path).sort_values("timestamp").reset_index(drop=True)
    inactive_mask = pd.read_parquet(config.tda02_inactive_bar_mask_path)
    return variables, validity, inactive_mask


# ---------------------------------------------------------------------------
# Movimiento en ticks -- el objeto principal de la etapa
# ---------------------------------------------------------------------------

def compute_tick_variables(variables: pd.DataFrame, validity: pd.DataFrame, tick_size: float) -> pd.DataFrame:
    """Construye ``delta_close_points``/``delta_close_ticks`` respetando la mascara de validez de TDA-04.

    Entrada
    -------
    variables : ``tda04_variables_1m.parquet`` (incluye ``close``, ``high``,
        ``low``, ``r_1m``, ordenado por ``timestamp``).
    validity  : ``tda04_return_validity_mask.parquet`` (incluye
        ``r_1m_valid``), MISMO orden de timestamps que ``variables`` --
        verificado con un ``assert`` antes de usarlo (igual que hace el
        propio TDA-04 entre sus dos tablas).
    tick_size : tamaño de tick del contrato (``config.tick_size``, 0.25
        para MNQ) -- especificacion externa, no recalculada aqui.

    Por que NO basta ``close.shift(1)`` a secas (seccion 4 de la tarea):
    la fila anterior en la tabla puede pertenecer a otra jornada, a otro
    contrato o estar separada por un hueco -- exactamente la misma razon
    por la que TDA-04 nunca usa ``pct_change()`` ciegamente. Aqui se
    calcula ``close.shift(1)`` (la MISMA operacion, sobre la MISMA tabla
    ordenada que usa TDA-04 para ``r_1m``) y luego se aplica
    EXPLICITAMENTE la mascara ``r_1m_valid`` de TDA-04 -- no una mascara
    nueva, la misma: donde ``r_1m`` es ``NaN`` por una regla de no-cruce,
    ``delta_close_points``/``delta_close_ticks`` tambien lo son.

    Transformacion
    --------------
    1. ``delta_close_points = close_t - close_{t-1}`` (``NaN`` donde
       ``r_1m_valid`` es ``False``).
    2. ``delta_close_ticks = delta_close_points / tick_size``.
    3. **Invariante verificada, no asumida**: para toda fila valida,
       ``delta_close_ticks`` debe ser entero dentro de
       ``TICK_GRID_TOLERANCE``. Si no lo es, se lanza
       ``TickGridInconsistencyError`` -- nunca se redondea en silencio.
    4. ``range_points = high_t - low_t`` (siempre definido, no depende de
       ninguna barra anterior -- igual que ``log_hl`` en TDA-04).
    5. ``hour_ny``/``year_ny``: se localiza ``timestamp`` (tz-naive, UTC
       por construccion desde TDA-00) como UTC y se convierte a
       ``America/New_York`` con ``zoneinfo`` (el mismo mecanismo DST-aware
       de TDA-01/TDA-02/TDA-03 -- nunca un offset fijo). ``hour_ny`` es la
       hora local literal (0..23) de esa conversion; ``year_ny`` es el año
       de calendario de esa misma hora local (no el año de ``trading_date``
       ni el de la fecha UTC cruda) -- ambos derivados de la MISMA
       conversion, para que la agrupacion por hora y por año sean
       consistentes entre si.

    Salida
    ------
    ``DataFrame`` con columnas de trazabilidad (``timestamp``,
    ``source_file``, ``contract``, ``trading_date``, ``hour_ny``,
    ``year_ny``), ``close``, ``high``, ``low``, ``r_1m``, ``r_1m_valid``,
    ``delta_close_points``, ``delta_close_ticks``, ``range_points``.
    """
    df = variables.sort_values("timestamp").reset_index(drop=True)
    v = validity.sort_values("timestamp").reset_index(drop=True)
    assert (df["timestamp"].to_numpy() == v["timestamp"].to_numpy()).all(), (
        "variables y validity no comparten exactamente los mismos timestamps, en el mismo orden"
    )

    valid = v["r_1m_valid"].to_numpy()
    prev_close = df["close"].shift(1)

    delta_close_points = (df["close"] - prev_close).where(valid, other=np.nan)
    delta_close_ticks = delta_close_points / tick_size

    residual = (delta_close_ticks - delta_close_ticks.round()).abs()
    bad = residual > TICK_GRID_TOLERANCE
    bad = bad.fillna(False)
    if bad.any():
        bad_rows = df.loc[bad, ["timestamp", "source_file"]].copy()
        bad_rows["delta_close_ticks"] = delta_close_ticks.loc[bad]
        bad_rows["residual"] = residual.loc[bad]
        raise TickGridInconsistencyError(
            f"{int(bad.sum())} fila(s) con delta_close_ticks NO entero dentro de tolerancia "
            f"({TICK_GRID_TOLERANCE}). TDA-00 certifico que cada Close individual cae en la "
            f"grilla de tick; esto indica una inconsistencia real. Primeras filas afectadas:\n"
            f"{bad_rows.head(5).to_string(index=False)}"
        )

    range_points = df["high"] - df["low"]

    ts_utc = df["timestamp"].dt.tz_localize("UTC")
    ts_ny = ts_utc.dt.tz_convert(NY_TZ)

    return pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "source_file": df["source_file"],
            "contract": df["contract"],
            "trading_date": df["trading_date"],
            "hour_ny": ts_ny.dt.hour,
            "year_ny": ts_ny.dt.year,
            "close": df["close"],
            "high": df["high"],
            "low": df["low"],
            "r_1m": df["r_1m"],
            "r_1m_valid": valid,
            "delta_close_points": delta_close_points,
            "delta_close_ticks": delta_close_ticks,
            "range_points": range_points,
        }
    )


def attach_forward_fill_flags(tick_df: pd.DataFrame, inactive_mask: pd.DataFrame) -> pd.DataFrame:
    """Une, por (``source_file``, timestamp UTC), la categoria de barra de TDA-02.

    Entrada: la tabla de ``compute_tick_variables`` y la mascara de
    barras inactivas de TDA-02 (``status``, ``category`` por
    ``(source_file, expected_ts_utc)``).
    Transformacion: ``left join`` -- toda fila de ``tick_df`` conserva su
    lugar; se añade ``category`` (``ACTIVA`` / ``FLAT_AISLADA`` /
    ``CANDIDATO_FORWARD_FILL`` / ``NaN`` -- las filas ``AUSENTE`` de
    TDA-02 no tienen `category` y, de todas formas, nunca aparecen aqui:
    son minutos SIN barra, y ``tick_df`` solo tiene filas CON barra).
    **Nunca** se convierte `CANDIDATO_FORWARD_FILL` en un forward-fill
    confirmado: se conserva el nombre exacto de la categoria de TDA-02,
    sin reinterpretarla.
    Salida: ``tick_df`` con la columna ``ff_category`` añadida.
    """
    right = inactive_mask[["source_file", "expected_ts_utc", "category"]].copy()
    right["timestamp"] = right["expected_ts_utc"].dt.tz_localize(None)
    right = right.drop(columns=["expected_ts_utc"]).rename(columns={"category": "ff_category"})
    merged = tick_df.merge(right, on=["source_file", "timestamp"], how="left")
    return merged


# ---------------------------------------------------------------------------
# Metricas de resolucion efectiva -- una funcion, reutilizada por cada agregacion
# ---------------------------------------------------------------------------

def summarize_resolution(df: pd.DataFrame, tick_size: float) -> dict:
    """Calcula el conjunto completo de indicadores de resolucion efectiva sobre UN grupo de filas.

    Entrada: un subconjunto de la tabla de ``compute_tick_variables``
    (p.ej. todas las filas, o solo las de una hora NY, o de un año).
    Solo se usan las filas con ``r_1m_valid=True`` -- exactamente la
    misma poblacion para ``n``, ``zero_fraction``, ``sigma`` y
    ``median_range``, para que todas las cifras de una misma fila de
    salida describan el MISMO conjunto de barras.

    Unidades -- por que no se puede dividir ``tick_size`` (puntos) entre
    ``std(r_1m)`` (adimensional) directamente: mezclaria una magnitud en
    PUNTOS con una en proporcion/logaritmo. Esta funcion calcula DOS
    cocientes dimensionalmente correctos y por separado:

    - ``tick_to_sigma_points = tick_size / std(delta_close_points)`` --
      ambos en PUNTOS. Responde "¿cuan grande es 1 tick frente al
      movimiento tipico de precio de 1 minuto?" directamente.
    - ``tick_to_sigma_return = tick_return_repr / std(r_1m)`` -- ambos
      ADIMENSIONALES. ``tick_return_repr = ln((C_repr+tick)/C_repr)``,
      con ``C_repr`` = precio Close MEDIANO del grupo (una unica cifra
      representativa del nivel de precio de ese grupo, NO un valor por
      fila -- el equivalente en retorno de un tick de puntos depende del
      nivel de precio, que cambia con el tiempo, seccion 13 de la tarea).

    Salida: ``dict`` con las 15 cifras exigidas por la seccion 10 de la
    tarea. Casos de division por cero (``sigma=0`` o ``median_range=0``,
    posibles en un grupo degenerado con muy pocas observaciones o sin
    ningun movimiento) se resuelven con ``np.inf`` explicito, nunca con
    una excepcion ni con un ``NaN`` silencioso indistinguible de "no
    calculable" -- ``n`` queda siempre disponible para que el lector
    juzgue si ese ``inf`` es significativo o es ruido de un grupo minusculo.
    """
    valid = df[df["r_1m_valid"]]
    n = int(len(valid))
    if n == 0:
        return {"n": 0}

    zero_fraction = float((valid["r_1m"] == 0.0).mean())

    sigma_points = float(valid["delta_close_points"].std(ddof=1)) if n > 1 else float("nan")
    tick_to_sigma_points = (tick_size / sigma_points) if sigma_points and sigma_points > 0 else (
        float("inf") if sigma_points == 0 else float("nan")
    )

    sigma_return = float(valid["r_1m"].std(ddof=1)) if n > 1 else float("nan")
    close_repr = float(valid["close"].median())
    tick_return_repr = float(np.log((close_repr + tick_size) / close_repr))
    tick_to_sigma_return = (tick_return_repr / sigma_return) if sigma_return and sigma_return > 0 else (
        float("inf") if sigma_return == 0 else float("nan")
    )

    median_range_points = float(valid["range_points"].median())
    tick_to_median_range = (tick_size / median_range_points) if median_range_points and median_range_points > 0 else (
        float("inf") if median_range_points == 0 else float("nan")
    )

    abs_ticks = valid["delta_close_ticks"].abs()
    median_abs_ticks = float(abs_ticks.median())
    prop_0_ticks = float((abs_ticks == 0).mean())
    prop_abs_1_tick = float((abs_ticks == 1).mean())
    prop_abs_le_2_ticks = float((abs_ticks <= 2).mean())
    prop_abs_le_5_ticks = float((abs_ticks <= 5).mean())

    central = valid[abs_ticks <= CENTRAL_RANGE_TICKS]
    n_distinct_r1m_central = int(central["r_1m"].nunique())
    n_distinct_ticks_central = int(valid.loc[abs_ticks <= CENTRAL_RANGE_TICKS, "delta_close_ticks"].round().astype(int).nunique())

    return {
        "n": n,
        "zero_fraction": zero_fraction,
        "sigma_delta_close_points": sigma_points,
        "tick_to_sigma_points": tick_to_sigma_points,
        "sigma_r_1m": sigma_return,
        "tick_return_representative": tick_return_repr,
        "tick_to_sigma_return": tick_to_sigma_return,
        "close_representative": close_repr,
        "median_range_points": median_range_points,
        "tick_to_median_range": tick_to_median_range,
        "median_abs_ticks": median_abs_ticks,
        "prop_0_ticks": prop_0_ticks,
        "prop_abs_1_tick": prop_abs_1_tick,
        "prop_abs_le_2_ticks": prop_abs_le_2_ticks,
        "prop_abs_le_5_ticks": prop_abs_le_5_ticks,
        "n_distinct_r1m_central": n_distinct_r1m_central,
        "n_distinct_ticks_central": n_distinct_ticks_central,
    }


def build_global_table(tick_df: pd.DataFrame, tick_size: float) -> pd.DataFrame:
    return pd.DataFrame([{"group": "GLOBAL", **summarize_resolution(tick_df, tick_size)}])


def build_by_hour_table(tick_df: pd.DataFrame, tick_size: float) -> pd.DataFrame:
    rows = []
    for hour, sub in tick_df.groupby("hour_ny", sort=True):
        rows.append({"hour_ny": int(hour), **summarize_resolution(sub, tick_size)})
    return pd.DataFrame(rows).sort_values("hour_ny").reset_index(drop=True)


def build_by_year_table(tick_df: pd.DataFrame, tick_size: float) -> pd.DataFrame:
    rows = []
    for year, sub in tick_df.groupby("year_ny", sort=True):
        rows.append({"year_ny": int(year), **summarize_resolution(sub, tick_size)})
    return pd.DataFrame(rows).sort_values("year_ny").reset_index(drop=True)


def build_by_year_hour_table(tick_df: pd.DataFrame, tick_size: float) -> pd.DataFrame:
    rows = []
    for (year, hour), sub in tick_df.groupby(["year_ny", "hour_ny"], sort=True):
        rows.append({"year_ny": int(year), "hour_ny": int(hour), **summarize_resolution(sub, tick_size)})
    return pd.DataFrame(rows).sort_values(["year_ny", "hour_ny"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Forward-fill: comparacion con/sin, sin inventar una clasificacion
# ---------------------------------------------------------------------------

@dataclass
class ForwardFillComparison:
    n_confirmed_forward_fill: int
    n_candidate_forward_fill: int
    global_with_all: dict
    global_excluding_candidates: dict
    note: str


def compare_with_without_forward_fill(tick_df: pd.DataFrame, tick_size: float) -> ForwardFillComparison:
    """Compara la fraccion de ceros con/sin barras FORWARD_FILL -- metodo minimo de TH09.

    Entrada: la tabla con ``ff_category`` ya unida
    (``attach_forward_fill_flags``).

    Por que no hay una categoria "FORWARD_FILL confirmada" que excluir
    (TDA-02, §10.2): la mascara de TDA-02 solo tiene ``ACTIVA``,
    ``FLAT_AISLADA`` y ``CANDIDATO_FORWARD_FILL`` (mas ``VOLUMEN_CERO``,
    vacia) -- CERO filas con una categoria de forward-fill CONFIRMADA. Si
    ``n_confirmed_forward_fill == 0`` (como es el caso en el conjunto de
    investigacion, verificado aqui explicitamente, no asumido), la
    comparacion "con/sin FORWARD_FILL" del roadmap es, por construccion,
    la MISMA cifra dos veces -- se declara asi explicitamente, no se
    inventa una diferencia.

    Se añade, ademas, una comparacion ETIQUETADA como
    ``SENSITIVITY_ONLY`` excluyendo ``CANDIDATO_FORWARD_FILL`` (barras
    candidatas, NUNCA confirmadas): sirve solo para mostrar que esa
    exclusion no cambia materialmente la fraccion de ceros global -- NO
    es el resultado principal de TH09 y NO participa en la evaluacion de
    STOP-5.

    Salida: ``ForwardFillComparison`` con los conteos y ambas cifras
    globales de ``zero_fraction`` (con todas las barras / excluyendo
    candidatos).
    """
    n_confirmed = int((tick_df["ff_category"] == "FORWARD_FILL_CONFIRMADO").sum())
    n_candidates = int((tick_df["ff_category"] == "CANDIDATO_FORWARD_FILL").sum())

    global_all = summarize_resolution(tick_df, tick_size)

    if n_confirmed > 0:
        without_confirmed = tick_df[tick_df["ff_category"] != "FORWARD_FILL_CONFIRMADO"]
        note = (
            f"Se encontraron {n_confirmed} barra(s) con FORWARD_FILL CONFIRMADO en TDA-02 -- "
            "la comparacion con/sin excluye exactamente esas filas."
        )
        global_excluding = summarize_resolution(without_confirmed, tick_size)
    else:
        note = (
            "No existen barras confirmadas como FORWARD_FILL en el conjunto de investigacion "
            "(TDA-02, seccion 10.2: 0 filas con categoria de forward-fill confirmada). La "
            "comparacion con/sin FORWARD_FILL no modifica el resultado: son la MISMA cifra. "
            f"(Sensibilidad SOLO informativa excluyendo los {n_candidates} CANDIDATO_FORWARD_FILL "
            "-- nunca confirmados -- se reporta aparte, etiquetada SENSITIVITY_ONLY.)"
        )
        global_excluding = dict(global_all)  # misma cifra, sin recalcular nada distinto

    return ForwardFillComparison(
        n_confirmed_forward_fill=n_confirmed,
        n_candidate_forward_fill=n_candidates,
        global_with_all=global_all,
        global_excluding_candidates=global_excluding,
        note=note,
    )


def sensitivity_excluding_candidates(tick_df: pd.DataFrame, tick_size: float) -> dict:
    """``SENSITIVITY_ONLY``: recalcula el resumen global excluyendo ``CANDIDATO_FORWARD_FILL``.

    Ver ``compare_with_without_forward_fill`` para el porque. Esta funcion
    NUNCA debe usarse como resultado principal de TDA-05 ni para decidir
    STOP-5 -- solo para mostrar, de forma transparente, cuanto (o cuan
    poco) cambiaria el resultado si esas 121 barras candidatas (nunca
    confirmadas) se excluyeran.
    """
    without_candidates = tick_df[tick_df["ff_category"] != "CANDIDATO_FORWARD_FILL"]
    return summarize_resolution(without_candidates, tick_size)


# ---------------------------------------------------------------------------
# Incertidumbre -- bootstrap por bloques de trading_date (G5)
# ---------------------------------------------------------------------------

def block_bootstrap_global_metrics(
    tick_df: pd.DataFrame, tick_size: float, n_boot: int = 300, seed: int = 0
) -> dict[str, tuple[float, float, float]]:
    """Intervalo de incertidumbre por bootstrap de bloques (jornadas completas), en una sola pasada.

    Entrada: la tabla de ticks completa y el numero de remuestreos.

    Por que remuestrear por ``trading_date`` (seccion 12 de la tarea, G5):
    las barras dentro de una misma jornada no son independientes entre si
    (mismo regimen de liquidez, mismo dia de la semana, posible
    dependencia de corto plazo) -- remuestrear FILA A FILA rompería esa
    estructura y subestimaria la incertidumbre real. Remuestrear jornadas
    COMPLETAS (con reemplazo) preserva la dependencia intra-dia y es la
    solucion mas simple compatible con G5.

    Por que esta version trabaja con arrays de NumPy en vez de repetir
    ``pandas.concat``/``groupby`` en cada iteracion (nota de rendimiento,
    no de metodologia): con ~1,9 millones de filas y ~1.420 jornadas,
    reconstruir un ``DataFrame`` completo en cada una de las ``n_boot``
    repeticiones es sustancialmente mas lento que indexar arrays NumPy
    precalculados -- el resultado estadistico es identico, solo cambia la
    velocidad. Se agrupan los indices de fila por ``trading_date`` UNA
    sola vez; cada repeticion arma la remuestra concatenando esos indices
    (no los datos) y calculando las dos metricas directamente sobre los
    arrays resultantes.

    Solo se calculan intervalos para las DOS metricas globales
    principales (``zero_fraction``, ``tick_to_sigma_points`, seccion 12
    de la tarea: "no hace falta calcular bootstrap para cada bin
    individual del histograma" -- ni, por extension, para cada fila de
    las tablas por hora/año).

    Salida: ``{"zero_fraction": (punto, p2.5, p97.5), "tick_to_sigma_points": (...)}``.
    """
    valid = tick_df[tick_df["r_1m_valid"]]
    dates = valid["trading_date"].to_numpy()
    r1m = valid["r_1m"].to_numpy()
    dpts = valid["delta_close_points"].to_numpy()

    unique_dates, inverse = np.unique(dates, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    boundaries = np.searchsorted(sorted_inverse, np.arange(len(unique_dates) + 1))
    date_to_rows = [order[boundaries[i]:boundaries[i + 1]] for i in range(len(unique_dates))]

    point_zero = float((r1m == 0).mean())
    point_sigma = float(dpts.std(ddof=1))
    point_ratio = (tick_size / point_sigma) if point_sigma > 0 else float("inf")

    rng = np.random.default_rng(seed)
    n_dates = len(unique_dates)
    boot_zero = np.empty(n_boot)
    boot_ratio = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.integers(0, n_dates, size=n_dates)
        idx = np.concatenate([date_to_rows[i] for i in sampled])
        r1m_s = r1m[idx]
        sigma_s = dpts[idx].std(ddof=1)
        boot_zero[b] = (r1m_s == 0).mean()
        boot_ratio[b] = (tick_size / sigma_s) if sigma_s > 0 else np.inf

    zero_lo, zero_hi = np.percentile(boot_zero, [2.5, 97.5])
    finite_ratio = boot_ratio[np.isfinite(boot_ratio)]
    if len(finite_ratio) > 0:
        ratio_lo, ratio_hi = np.percentile(finite_ratio, [2.5, 97.5])
    else:
        ratio_lo, ratio_hi = float("nan"), float("nan")

    return {
        "zero_fraction": (point_zero, float(zero_lo), float(zero_hi)),
        "tick_to_sigma_points": (point_ratio, float(ratio_lo), float(ratio_hi)),
    }


# ---------------------------------------------------------------------------
# STOP-5 -- lista de segmentos candidatos a revision (NO una exclusion automatica)
# ---------------------------------------------------------------------------

def build_stop5_watchlist(
    by_hour: pd.DataFrame, by_year: pd.DataFrame, by_year_hour: pd.DataFrame
) -> pd.DataFrame:
    """Señala, de forma puramente descriptiva, los segmentos que merecen lectura atenta en STOP-5.

    Entrada: las tres tablas de agregacion -- por hora, por año y por
    año×hora. Las tres se revisan: la granularidad año×hora es la que
    puede exponer un segmento localmente muy discreto (p.ej. una hora de
    un año concreto, de muestra pequeña) que ni la vista por hora
    (promediada sobre todos los años) ni la vista por año (promediada
    sobre todas las horas) alcanzarian a mostrar por separado.
    Transformacion: marca (no excluye) las filas donde
    ``zero_fraction >= WATCHLIST_MIN_ZERO_FRACTION`` **o**
    ``tick_to_sigma_points >= WATCHLIST_MAX_TICK_TO_SIGMA`` -- umbrales
    deliberadamente laxos, usados solo para no tener que inspeccionar a
    mano cada fila de las tres tablas; NO son el criterio de STOP-5 en si
    (la tarea prohibe explicitamente un corte automatico de exclusion).
    La decision final de STOP-5 se toma leyendo la evidencia completa de
    estas filas (incluido ``n`` -- un segmento año×hora señalado con
    muestra pequeña es evidencia LOCAL, no un regimen a excluir), no
    aplicando este watchlist como regla.
    Salida: subconjunto de ``by_hour``, ``by_year`` y ``by_year_hour``
    (con una columna ``segment`` identificando cada uno, p.ej.
    ``hour_ny=00``, ``year_ny=2019`` o ``year_ny=2019 hour_ny=23``) que
    cumple cualquiera de los dos criterios, ordenado por
    ``tick_to_sigma_points`` descendente.
    """
    flagged_hour = by_hour[
        (by_hour["zero_fraction"] >= WATCHLIST_MIN_ZERO_FRACTION)
        | (by_hour["tick_to_sigma_points"] >= WATCHLIST_MAX_TICK_TO_SIGMA)
    ].copy()
    flagged_hour.insert(0, "segment", flagged_hour["hour_ny"].apply(lambda h: f"hour_ny={h:02d}"))

    flagged_year = by_year[
        (by_year["zero_fraction"] >= WATCHLIST_MIN_ZERO_FRACTION)
        | (by_year["tick_to_sigma_points"] >= WATCHLIST_MAX_TICK_TO_SIGMA)
    ].copy()
    flagged_year.insert(0, "segment", flagged_year["year_ny"].apply(lambda y: f"year_ny={y}"))

    flagged_year_hour = by_year_hour[
        (by_year_hour["zero_fraction"] >= WATCHLIST_MIN_ZERO_FRACTION)
        | (by_year_hour["tick_to_sigma_points"] >= WATCHLIST_MAX_TICK_TO_SIGMA)
    ].copy()
    # ``DataFrame.apply(..., axis=1)`` sobre 0 filas devuelve un DataFrame
    # vacio (no una Series), lo que rompe ``insert`` mas abajo -- caso
    # normal cuando ningun segmento año×hora cruza el umbral (como ocurre
    # en el conjunto de investigacion real, ver informe §11).
    if len(flagged_year_hour) > 0:
        segment_labels = flagged_year_hour.apply(
            lambda r: f"year_ny={int(r['year_ny'])} hour_ny={int(r['hour_ny']):02d}", axis=1
        )
    else:
        segment_labels = pd.Series([], dtype=str)
    flagged_year_hour.insert(0, "segment", segment_labels)

    common_cols = ["segment", "n", "zero_fraction", "tick_to_sigma_points", "median_abs_ticks"]
    combined = pd.concat(
        [flagged_hour[common_cols], flagged_year[common_cols], flagged_year_hour[common_cols]],
        ignore_index=True,
    )
    return combined.sort_values("tick_to_sigma_points", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------

@dataclass
class TDA05Result:
    tick_df: pd.DataFrame
    global_table: pd.DataFrame
    by_hour: pd.DataFrame
    by_year: pd.DataFrame
    by_year_hour: pd.DataFrame
    forward_fill: ForwardFillComparison
    sensitivity_excluding_candidates: dict
    zero_fraction_ci: tuple
    tick_to_sigma_ci: tuple
    stop5_watchlist: pd.DataFrame


def run_tda05_analysis(config: SnapshotConfig) -> TDA05Result:
    """Orquesta TDA-05 completo sobre las variables de TDA-04.

    Reutiliza, sin duplicar logica: la proteccion del hold-out
    (``holdout_guard.py`` -- defensiva, TDA-05 no abre ningun archivo
    crudo) y los tres artefactos de entrada (TDA-04: variables + validez;
    TDA-02: barras inactivas). No escribe ningun archivo -- responsabilidad
    de ``run_tda05.py``.
    """
    validate_research_holdout_disjoint(config)

    variables, validity, inactive_mask = load_inputs(config)

    last_timestamps = variables.groupby("source_file")["timestamp"].max().to_dict()
    validate_last_timestamps_before_boundary(config, last_timestamps)

    tick_df = compute_tick_variables(variables, validity, config.tick_size)
    tick_df = attach_forward_fill_flags(tick_df, inactive_mask)

    global_table = build_global_table(tick_df, config.tick_size)
    by_hour = build_by_hour_table(tick_df, config.tick_size)
    by_year = build_by_year_table(tick_df, config.tick_size)
    by_year_hour = build_by_year_hour_table(tick_df, config.tick_size)

    forward_fill = compare_with_without_forward_fill(tick_df, config.tick_size)
    sensitivity = sensitivity_excluding_candidates(tick_df, config.tick_size)

    bootstrap_cis = block_bootstrap_global_metrics(tick_df, config.tick_size)
    zero_fraction_ci = bootstrap_cis["zero_fraction"]
    tick_to_sigma_ci = bootstrap_cis["tick_to_sigma_points"]

    watchlist = build_stop5_watchlist(by_hour, by_year, by_year_hour)

    return TDA05Result(
        tick_df=tick_df,
        global_table=global_table,
        by_hour=by_hour,
        by_year=by_year,
        by_year_hour=by_year_hour,
        forward_fill=forward_fill,
        sensitivity_excluding_candidates=sensitivity,
        zero_fraction_ci=zero_fraction_ci,
        tick_to_sigma_ci=tick_to_sigma_ci,
        stop5_watchlist=watchlist,
    )
