"""TDA-01 -- Semántica temporal, de sesión y de contrato: evidencia forense.

Implementa exclusivamente la parte de TDA-01
(``docs/methodology/Tsay_OHLCV_analysis_roadmap.md``, sección "TDA-01") que
requiere código para ser reproducible: el análisis forense de huecos
internos (por archivo) usado para inferir si el timestamp de una barra
marca su inicio o su cierre, y para observar cómo se representan los
minutos sin operaciones.

Qué NO hace este módulo (deliberadamente, para no invadir TDA-02/TDA-03):

- No construye ninguna grilla temporal esperada.
- No hace inventario completo de huecos ni los clasifica por causa
  calendario a calendario (feriados, cierres anticipados exactos, etc.).
- No construye ninguna sesión operativa ni ventana de análisis.
- No toca ningún archivo de ``config.holdout_files``.

Método, en una frase: los archivos del conjunto de investigación son,
según TDA-00, internamente monótonos y sin duplicados -- por lo tanto,
cualquier salto de más de 1 minuto entre dos filas consecutivas de un
mismo archivo es un hueco real. El corte de mantenimiento diario de CME
(~17:00-18:00 hora de Nueva York, casi todos los días de la semana) es un
ancla horaria conocida y de alta frecuencia (ocurre cientos de veces en el
conjunto de investigación): comparar la etiqueta de la última barra antes
del hueco y de la primera barra después, en hora local de Nueva York,
contra esa ancla, es una prueba forense directa de si el timestamp marca
inicio o cierre de barra -- sin necesidad de asumir ninguna ventana de
sesión ni resolver el inventario completo de huecos.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from ohlcv_dataroad.config import SnapshotConfig
from ohlcv_dataroad.ingest.holdout_guard import (
    validate_last_timestamps_before_boundary,
    validate_research_holdout_disjoint,
)
from ohlcv_dataroad.ingest.parsing import parse_raw_file

NY_TZ = ZoneInfo("America/New_York")

# Firma esperada de un cierre anticipado bajo la convención cierre-de-barra
# ya establecida por TDA-01 (ver reports/mnq/TDA01_convencion_temporal.md
# §1-2): si el mercado cierra anticipadamente a las 13:00 hora de Nueva
# York y reabre con normalidad esa misma tarde (en el mismo horario que
# cualquier reapertura tras el corte de mantenimiento, ver
# DAILY_MAINTENANCE_MIN_MINUTES más abajo), la última barra del tramo debe
# etiquetarse exactamente "13:00:00" NY y la primera tras la reapertura,
# exactamente "18:01:00" NY -- la misma etiqueta de reapertura que ya se
# observa en el ancla diaria. Esta es una firma de DOS lados, no solo un
# valor cercano a 13:00: exigir ambos lados a la vez reduce el riesgo de
# capturar huecos que casualmente empiezan cerca de esa hora por otro
# motivo (dato faltante, evento de mercado real, etc.).
EARLY_CLOSE_BEFORE_NY_TIME = "13:00:00"
EARLY_CLOSE_AFTER_NY_TIME = "18:01:00"

# Umbrales puramente descriptivos para agrupar huecos por orden de magnitud,
# NO una clasificación causal de calendario (eso es TDA-02). Un hueco de
# 30-90 minutos es compatible con el corte de mantenimiento diario de CME
# (~60-61 min observados); un hueco de más de 40 horas es compatible con un
# fin de semana o una transición de contrato con huecos de varios días.
DAILY_MAINTENANCE_MIN_MINUTES = 30
DAILY_MAINTENANCE_MAX_MINUTES = 90
LONG_GAP_MIN_MINUTES = 40 * 60


def load_research_rows(config: SnapshotConfig) -> pd.DataFrame:
    """Parsea todos los archivos de ``config.research_files`` y los concatena.

    Entrada: la configuración ya cargada.
    Transformación: reutiliza ``parsing.parse_raw_file`` -- la MISMA función
    que usa TDA-00 -- por cada archivo declarado en ``research_files``
    (nunca ``holdout_files``); añade una columna ``source_file``; concatena.
    Sólo se conservan las filas bien formadas (``parsed.rows``): los
    errores de parseo no son relevantes para un análisis de huecos
    temporales, y TDA-00 ya certificó que el conjunto de investigación
    tiene 0 errores de parseo y 0 violaciones de invariante.
    Salida: un DataFrame con columnas ``source_file, timestamp, open, high,
    low, close, volume``, ordenado por archivo y luego por timestamp.
    """
    frames = []
    for source_file in config.research_files:
        path = config.raw_dir / source_file
        parsed = parse_raw_file(path, config.timestamp_format, config.separator)
        df = parsed.rows.copy()
        df["source_file"] = source_file
        frames.append(df)
    all_rows = pd.concat(frames, ignore_index=True)
    return all_rows.sort_values(["source_file", "timestamp"]).reset_index(drop=True)


def compute_intra_file_gaps(rows: pd.DataFrame) -> pd.DataFrame:
    """Calcula, POR ARCHIVO, el hueco entre cada fila y la anterior.

    Entrada: el DataFrame de ``load_research_rows`` (o equivalente), con
    columnas ``source_file`` y ``timestamp``.

    Por qué por archivo y no sobre la serie global concatenada: cada
    archivo es un contrato distinto (ver TDA-00). La diferencia entre la
    última barra de un archivo y la primera del siguiente no es un "hueco"
    en el sentido de TDA-01 -- es una transición de contrato, un fenómeno
    distinto (alcance de TDA-03). Calcular los huecos con
    ``groupby("source_file")`` evita mezclar ambos fenómenos.

    Salida: un DataFrame con una fila por cada par de filas consecutivas
    DENTRO de un mismo archivo cuya diferencia de timestamp supera 1
    minuto, con columnas ``source_file, before_ts_utc, after_ts_utc,
    gap_minutes``. Huecos de exactamente 1 minuto (el caso normal, sin
    huecos) no se incluyen.
    """
    records = []
    for source_file, group in rows.groupby("source_file", sort=False):
        ts = group["timestamp"].sort_values().reset_index(drop=True)
        diffs = ts.diff()
        for i in range(1, len(ts)):
            gap_minutes = diffs.iloc[i].total_seconds() / 60
            if gap_minutes > 1:
                records.append(
                    {
                        "source_file": source_file,
                        "before_ts_utc": ts.iloc[i - 1],
                        "after_ts_utc": ts.iloc[i],
                        "gap_minutes": gap_minutes,
                    }
                )
    return pd.DataFrame(records, columns=["source_file", "before_ts_utc", "after_ts_utc", "gap_minutes"])


def attach_ny_wallclock(gaps: pd.DataFrame) -> pd.DataFrame:
    """Añade la hora local de Nueva York (DST-aware) de cada borde del hueco.

    Entrada: el DataFrame de ``compute_intra_file_gaps`` (columnas
    ``before_ts_utc``, ``after_ts_utc`` -- timestamps tz-naive que
    representan UTC, el hecho ya confirmado documentalmente para este
    dataset).

    Transformación, el paso metodológicamente importante de todo el
    módulo: se localiza cada timestamp como UTC (``tz_localize("UTC")``,
    NUNCA un offset fijo) y se convierte a ``America/New_York``
    (``tz_convert``, vía ``zoneinfo``, que consulta la base de datos IANA
    de reglas de horario de verano). Este orden -- localizar en la zona de
    origen, LUEGO convertir -- es exactamente la secuencia que la memoria
    heredada del proyecto marca como correcta
    (``docs/instruments/mnq/MNQ_DATA_PRIOR_KNOWLEDGE.md``, sección 5.6) y
    es la única forma de que el mismo código produzca la hora local
    correcta tanto en horario estándar (EST, UTC-5) como en horario de
    verano (EDT, UTC-4) sin necesitar dos ramas de código ni una fecha de
    corte hardcodeada.

    Salida: el mismo DataFrame con 4 columnas nuevas: ``before_ny``,
    ``after_ny`` (timestamps tz-aware en America/New_York) y
    ``before_ny_time``, ``after_ny_time`` (su hora en formato ``HH:MM:SS``,
    la forma en la que se agregan y cuentan en el informe).
    """
    gaps = gaps.copy()
    for col in ["before_ts_utc", "after_ts_utc"]:
        utc = pd.to_datetime(gaps[col]).dt.tz_localize("UTC")
        ny = utc.dt.tz_convert(NY_TZ)
        prefix = col.replace("_ts_utc", "")
        gaps[f"{prefix}_ny"] = ny
        gaps[f"{prefix}_ny_time"] = ny.dt.strftime("%H:%M:%S")
    return gaps


def classify_gap_magnitude(gap_minutes: float) -> str:
    """Clasifica un hueco por orden de magnitud (descriptivo, no causal).

    Entrada: la duración del hueco en minutos.
    Salida: una de tres etiquetas puramente dimensionales:
      - ``"daily_maintenance_like"``: 30-90 min -- compatible en magnitud
        con el corte de mantenimiento diario de CME.
      - ``"weekly_or_long"``: > 40 h -- compatible con un fin de semana o
        una transición de contrato con solapamiento/hueco de varios días.
      - ``"other"``: cualquier otro valor -- no se interpreta aquí; queda
        como evidencia para TDA-02 (inventario completo de huecos).
    Esta función NO afirma que un hueco "sea" el corte de mantenimiento:
    sólo agrupa por tamaño para poder mirar el subconjunto relevante para
    la pregunta de TDA-01. La confirmación de que el subconjunto
    corresponde de verdad al corte de mantenimiento es el resultado
    empírico (concentración en una única etiqueta de hora), no una regla
    de calendario impuesta de antemano.
    """
    if DAILY_MAINTENANCE_MIN_MINUTES <= gap_minutes <= DAILY_MAINTENANCE_MAX_MINUTES:
        return "daily_maintenance_like"
    if gap_minutes > LONG_GAP_MIN_MINUTES:
        return "weekly_or_long"
    return "other"


@dataclass
class BoundaryLabelSummary:
    """Resumen de a qué hora de Nueva York se concentran los bordes de un grupo de huecos."""

    n_gaps: int
    before_ny_time_counts: pd.Series
    after_ny_time_counts: pd.Series
    gap_minutes_counts: pd.Series


def summarize_boundary_labels(gaps: pd.DataFrame) -> BoundaryLabelSummary:
    """Cuenta cuántas veces se repite cada etiqueta horaria en los bordes de un grupo de huecos.

    Entrada: un subconjunto de huecos ya pasado por ``attach_ny_wallclock``
    (típicamente, sólo los de magnitud ``"daily_maintenance_like"``).
    Transformación: ``value_counts()`` sobre cada una de las tres columnas
    relevantes. No hay ningún cálculo estadístico más alla de un conteo de
    frecuencias -- la fuerza del resultado está en la concentración (una
    sola etiqueta domina con una fracción muy alta del total), no en un
    test formal.
    Salida: un ``BoundaryLabelSummary`` con los tres conteos, listo para
    imprimir o volcar a un reporte.
    """
    return BoundaryLabelSummary(
        n_gaps=len(gaps),
        before_ny_time_counts=gaps["before_ny_time"].value_counts(),
        after_ny_time_counts=gaps["after_ny_time"].value_counts(),
        gap_minutes_counts=gaps["gap_minutes"].value_counts().sort_index(),
    )


def identify_early_close_like_gaps(gaps: pd.DataFrame) -> pd.DataFrame:
    """Selecciona, de forma reproducible, la tercera ancla forense (cierres anticipados).

    Entrada: el DataFrame completo de huecos, ya pasado por
    ``attach_ny_wallclock`` y con la columna ``magnitude_class`` ya
    calculada por ``classify_gap_magnitude``.

    Regla exacta (ver constantes ``EARLY_CLOSE_BEFORE_NY_TIME`` y
    ``EARLY_CLOSE_AFTER_NY_TIME`` más arriba, y su justificación):
    1. El hueco debe pertenecer a la categoría ``"other"`` -- ni del
       tamaño del corte de mantenimiento diario ni de un fin de semana.
       Esto excluye por construcción cualquier solape con las otras dos
       anclas forenses.
    2. Su borde ANTERIOR, en hora de Nueva York, debe ser exactamente
       ``"13:00:00"``.
    3. Su borde POSTERIOR, en hora de Nueva York, debe ser exactamente
       ``"18:01:00"``.

    Es una coincidencia EXACTA de las dos condiciones, no una tolerancia
    ni un rango: un hueco cuyo borde anterior sea `"13:01:00"` o cuyo
    borde posterior sea distinto de `"18:01:00"` NO se cuenta aquí, aunque
    esté cerca. Esto es deliberado -- es la firma de dos lados que predice
    la convención cierre-de-barra ya establecida (§1-2 del informe), no un
    intento de maximizar cuántos huecos "parecen" un cierre anticipado.

    Esta función NO afirma que cada hueco seleccionado corresponda a una
    fecha de feriado concreta del calendario oficial de CME -- eso
    requeriría cruzar cada fecha contra un calendario, que es trabajo de
    TDA-02. Sólo afirma que el hueco tiene la forma (firma de dos lados)
    que la convención de cierre de barra predice para un cierre anticipado
    seguido de una reapertura normal la misma tarde.

    Salida: el subconjunto de ``gaps`` que cumple la regla, con las mismas
    columnas.
    """
    other = gaps[gaps["magnitude_class"] == "other"]
    matches = (other["before_ny_time"] == EARLY_CLOSE_BEFORE_NY_TIME) & (
        other["after_ny_time"] == EARLY_CLOSE_AFTER_NY_TIME
    )
    return other[matches].reset_index(drop=True)


def build_forensic_evidence(config: SnapshotConfig) -> pd.DataFrame:
    """Orquesta todo el análisis forense de huecos para TDA-01.

    Entrada: la configuración ya cargada.
    Flujo:
    0. ``validate_research_holdout_disjoint`` -- sin abrir ningún archivo,
       aborta si la configuración mezcla investigación y hold-out. Misma
       protección que usa TDA-00 (``holdout_guard.py``), no una copia.
    1. ``load_research_rows`` -- parsea únicamente ``config.research_files``.
    2. ``validate_last_timestamps_before_boundary`` -- con el máximo
       timestamp de cada archivo ya cargado, confirma que ninguno alcanza
       la frontera del hold-out. Sigue sin abrir ningún archivo de
       ``config.holdout_files``.
    3. ``compute_intra_file_gaps`` -> ``attach_ny_wallclock`` -> se añade
       la columna ``magnitude_class`` con ``classify_gap_magnitude``
       aplicada fila a fila.

    Salida: el DataFrame completo de huecos del conjunto de investigación,
    con toda la evidencia necesaria para las tablas del informe
    (``TDA01_convencion_temporal.md``) y para el CSV de evidencia
    persistido (``TDA01_evidencia_gaps.csv``).
    """
    validate_research_holdout_disjoint(config)

    rows = load_research_rows(config)

    last_timestamps = rows.groupby("source_file")["timestamp"].max().to_dict()
    validate_last_timestamps_before_boundary(config, last_timestamps)

    gaps = compute_intra_file_gaps(rows)
    gaps = attach_ny_wallclock(gaps)
    gaps["magnitude_class"] = gaps["gap_minutes"].apply(classify_gap_magnitude)
    return gaps.sort_values(["source_file", "before_ts_utc"]).reset_index(drop=True)
