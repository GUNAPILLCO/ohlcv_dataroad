"""TDA-03 -- Rolls y construccion de la serie continua.

Implementa la etapa TDA-03 del roadmap
(``docs/methodology/Tsay_OHLCV_analysis_roadmap.md``, seccion "TDA-03"):
caracteriza las 21 transiciones entre los 22 contratos trimestrales del
conjunto de investigacion, determina una regla CAUSAL de seleccion de
contrato activo, mide el basis de cada roll con solapamiento, construye
una serie canonica de un unico contrato activo por timestamp, y clasifica
los estadisticos previstos por el roadmap segun su invariancia a los
metodos de ajuste (ninguno / aditivo / ratio).

Que hace TDA-03, en una frase: decide, para cada ``trading_date`` del
conjunto de investigacion, cual de los dos contratos que podrian tener
datos ese dia es el "activo", de forma reproducible y sin usar nunca
informacion posterior al instante de la decision -- y deja esa decision
trazada en una mascara persistente, no escondida en el codigo.

Que NO hace este modulo (deliberadamente, fuera de alcance de TDA-03):

- No calcula retornos ni ninguna variable de TDA-04.
- No decide una unica representacion "correcta" (raw/ratio/aditivo) mas
  alla de lo que la evidencia permite justificar.
- No aplica back-adjustment automatico a todo el historial (regla 12
  heredada, y ademas: la evidencia de este conjunto de investigacion NO
  permite hacerlo de forma defendible mas que en un tramo acotado -- ver
  ``compute_adjustment_factors``).
- No abre ningun archivo de ``config.holdout_files``.
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
from ohlcv_dataroad.ingest.session_calendar import (
    SessionSchedule,
    build_session_schedule,
    expected_bar_grid_frame,
)
from ohlcv_dataroad.ingest.tda01_temporal_semantics import load_research_rows

# --- Codigo de mes de vencimiento CME/Globex (convencion de industria, --
# ya documentada como tal en TDA-01, no propia de este dataset).
MONTH_CODE = {"03": "H", "06": "M", "09": "U", "12": "Z"}

TRANSITION_OVERLAP = "OVERLAP"
TRANSITION_NO_OVERLAP = "NO_OVERLAP"

REASON_NORMAL = "NORMAL"
REASON_PRE_CROSSOVER = "PRE_CROSSOVER_OUTGOING_ACTIVE"
REASON_POST_CROSSOVER = "POST_CROSSOVER_INCOMING_ACTIVE"
REASON_ZERO_BAR_FALLBACK = "ZERO_BARS_FALLBACK"

DISCARD_NON_ACTIVE_ON_OVERLAP_DATE = "NON_ACTIVE_CONTRACT_ON_OVERLAP_DATE"


def parse_contract_label(source_file: str) -> str:
    """Convierte un nombre de archivo en su etiqueta de contrato corta.

    Entrada: ``"19_mnq_12_24.Last.txt"``.
    Transformacion: el nombre de archivo sigue el patron
    ``NN_mnq_MM_YY.Last.txt`` (ya documentado en TDA-01, §8): ``MM`` es el
    mes de vencimiento trimestral, ``YY`` el año a 2 digitos. Se traduce
    ``MM`` al codigo de mes CME/Globex (``03``→``H``, ``06``→``M``,
    ``09``→``U``, ``12``→``Z`` -- convencion estandar de la industria de
    futuros, no inferida de los datos).
    Salida: ``"Z24"``.
    """
    stem = source_file.split(".")[0]
    _, _, mm, yy = stem.split("_")
    return f"{MONTH_CODE[mm]}{yy}"


@dataclass(frozen=True)
class RollPolicy:
    """Politica de rollover, leida de ``configs/mnq_snapshot.yaml`` (seccion ``tda03``).

    Ningun parametro vive escondido en el codigo -- ver el YAML para la
    justificacion completa de cada valor (recalibrados bajo la sesion
    nativa CME_Equity de TDA-02, NO heredados de la ventana historica
    04:30-16:00).
    """

    min_incoming_share_shared: float
    confirmation_sessions_required: int
    extreme_jump_top_n: int


def load_roll_policy(config: SnapshotConfig) -> RollPolicy:
    return RollPolicy(
        min_incoming_share_shared=config.tda03_min_incoming_share_shared,
        confirmation_sessions_required=config.tda03_confirmation_sessions_required,
        extreme_jump_top_n=config.tda03_extreme_jump_top_n,
    )


def attach_trading_date_and_contract(rows: pd.DataFrame, schedule: SessionSchedule) -> pd.DataFrame:
    """Añade ``ts_utc``, ``trading_date`` y ``contract`` a las filas de investigacion.

    Entrada: la salida de ``load_research_rows`` (columnas ``source_file``,
    ``timestamp`` tz-naive UTC, OHLCV) y el ``SessionSchedule`` de TDA-02
    (reutilizado sin cambios, no reconstruido).
    Transformacion: localiza cada timestamp como UTC, lo ubica en su
    sesion (``trading_date_for``, TDA-02) y traduce ``source_file`` a su
    etiqueta de contrato corta (``parse_contract_label``).
    Salida: el mismo DataFrame con esas 3 columnas nuevas. Las filas cuyo
    timestamp no cae en ninguna sesion valida (fuera de grilla, ver
    TDA-02 §8) quedan con ``trading_date`` nulo -- se excluyen de todo
    calculo de volumen/cobertura por dia mas adelante, consistente con
    como TDA-02 ya las trata.
    """
    rows = rows.copy()
    rows["ts_utc"] = rows["timestamp"].dt.tz_localize("UTC")
    rows["trading_date"] = schedule.trading_date_for(rows["ts_utc"])
    rows["contract"] = rows["source_file"].map(parse_contract_label)
    return rows


# ---------------------------------------------------------------------------
# Evidencia diaria de volumen en el solapamiento
# ---------------------------------------------------------------------------

def compute_overlap_daily_evidence(
    out_rows: pd.DataFrame, in_rows: pd.DataFrame, overlap_dates: list, expected_by_date: dict
) -> pd.DataFrame:
    """Evidencia diaria de volumen para UNA transicion con solapamiento.

    Entrada
    -------
    out_rows, in_rows : filas (ya con ``trading_date``) del contrato
        saliente y entrante, respectivamente.
    overlap_dates : lista ordenada de ``trading_date`` en las que AMBOS
        contratos tienen al menos una barra.
    expected_by_date : ``{trading_date: minutos_esperados}`` (grilla de
        TDA-02, para poder reportar cobertura cuando sea relevante).

    Transformacion, por cada ``trading_date`` del solapamiento
    -------------------------------------------------------------
    - Volumen total de cada contrato ese dia (todas sus barras).
    - Minutos COMPARTIDOS: timestamps en los que AMBOS contratos tienen
      una barra ese dia. Volumen de cada contrato calculado SOLO sobre
      esos minutos (regla 5 de la politica heredada, §7.1 de
      ``MNQ_DATA_PRIOR_KNOWLEDGE.md`` -- comparar volumenes de periodos
      distintos no mide traspaso de liquidez).
    - ``share_shared``: fraccion del volumen compartido que corresponde
      al entrante -- la variable que decide la señal de roll
      (``determine_overlap_rollover``).
    - ``share_total``: la misma fraccion pero sobre el volumen TOTAL del
      dia (no solo minutos compartidos) -- se reporta aparte, nunca se
      usa para decidir, precisamente porque puede diferir mucho de
      ``share_shared`` cuando uno de los dos contratos tiene cobertura
      reducida ese dia (ver ejemplo real, 2025-03-17, en el informe).

    Salida: ``DataFrame``, una fila por ``trading_date`` del solapamiento.
    """
    records = []
    for d in overlap_dates:
        o_day = out_rows[out_rows["trading_date"] == d]
        i_day = in_rows[in_rows["trading_date"] == d]
        o_vol_total = int(o_day["volume"].sum())
        i_vol_total = int(i_day["volume"].sum())
        shared_minutes = set(o_day["ts_utc"]) & set(i_day["ts_utc"])
        o_vol_shared = int(o_day.loc[o_day["ts_utc"].isin(shared_minutes), "volume"].sum())
        i_vol_shared = int(i_day.loc[i_day["ts_utc"].isin(shared_minutes), "volume"].sum())
        denom_shared = o_vol_shared + i_vol_shared
        denom_total = o_vol_total + i_vol_total
        expected = expected_by_date.get(d)
        records.append(
            {
                "trading_date": d,
                "out_bars": len(o_day),
                "in_bars": len(i_day),
                "out_volume_total": o_vol_total,
                "in_volume_total": i_vol_total,
                "shared_minutes": len(shared_minutes),
                "out_volume_shared": o_vol_shared,
                "in_volume_shared": i_vol_shared,
                "share_shared": (i_vol_shared / denom_shared) if denom_shared > 0 else float("nan"),
                "share_total": (i_vol_total / denom_total) if denom_total > 0 else float("nan"),
                "in_out_volume_ratio_shared": (i_vol_shared / o_vol_shared) if o_vol_shared > 0 else float("inf"),
                "expected_minutes": expected,
                "out_coverage_pct": (100.0 * len(o_day) / expected) if expected else float("nan"),
                "in_coverage_pct": (100.0 * len(i_day) / expected) if expected else float("nan"),
            }
        )
    return pd.DataFrame(records)


def determine_overlap_rollover(daily_evidence: pd.DataFrame, combined_dates: list, policy: RollPolicy) -> dict:
    """Aplica la regla CAUSAL de crossover de volumen y devuelve la decision.

    Entrada
    -------
    daily_evidence : salida de ``compute_overlap_daily_evidence``, YA
        ordenada cronologicamente por ``trading_date``.
    combined_dates : lista ordenada de TODAS las ``trading_date`` que
        tocan el contrato saliente o el entrante (union, no solo el
        solapamiento) -- necesaria para poder aplicar la regla 8 heredada
        ("fecha efectiva = jornada siguiente OBSERVADA") incluso cuando
        la señal se emite en el ULTIMO dia del solapamiento (en ese caso,
        la jornada siguiente observada esta fuera de ``overlap_dates``,
        dentro de las fechas exclusivas del entrante).
    policy : ``RollPolicy`` (umbral y sesiones de confirmacion).

    Prueba de causalidad (test de reconstruccion, seccion 4 de la
    tarea): la señal del dia ``d`` se calcula EXCLUSIVAMENTE con datos de
    ``d`` y de dias anteriores (``compute_overlap_daily_evidence`` nunca
    mira un dia posterior). La decision de que contrato esta activo en
    la jornada ``e`` (efectiva) usa solo la señal ya confirmada en un dia
    estrictamente anterior a ``e``. Si se re-ejecutara este algoritmo
    disponiendo solo de los datos hasta el final del dia de la señal, se
    obtendria exactamente la misma decision -- nada de lo que ocurre
    DESPUES de la señal participa en calcularla.

    Transformacion
    --------------
    Recorre ``daily_evidence`` en orden cronologico, contando sesiones
    CONSECUTIVAS con ``share_shared >= policy.min_incoming_share_shared``.
    En cuanto el contador alcanza ``policy.confirmation_sessions_required``,
    esa es la ``signal_date`` (regla 7 heredada: con
    ``confirmation_sessions_required=1``, basta una sola jornada). La
    ``effective_date`` es la SIGUIENTE fecha observada en
    ``combined_dates`` despues de ``signal_date`` (regla 8 heredada: la
    señal se conoce al terminar la jornada de la señal; aplicarla ESE
    mismo dia usaria informacion que, en el instante de negociacion, aun
    no se conocia completa).

    Si ninguna jornada del solapamiento alcanza el umbral, la transicion
    se fuerza en el momento en que el archivo saliente simplemente deja
    de tener datos (fin de su exportacion) -- el solapamiento no puede
    continuar indefinidamente, y en ausencia de una señal de volumen mas
    temprana, el propio fin del archivo saliente es la unica frontera
    disponible.

    Salida: ``dict`` con ``signal_date``, ``effective_date``, ``rule``
    (texto), ``confidence`` (``ALTA``/``MEDIA``/``BAJA``).
    """
    daily_evidence = daily_evidence.sort_values("trading_date").reset_index(drop=True)
    consecutive = 0
    signal_date = None
    for _, row in daily_evidence.iterrows():
        if pd.notna(row["share_shared"]) and row["share_shared"] >= policy.min_incoming_share_shared:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= policy.confirmation_sessions_required:
            signal_date = row["trading_date"]
            break

    if signal_date is None:
        overlap_last = daily_evidence["trading_date"].max()
        after = [d for d in combined_dates if d > overlap_last]
        effective_date = after[0] if after else None
        return {
            "signal_date": None,
            "effective_date": effective_date,
            "rule": (
                f"ninguna jornada del solapamiento alcanzo share_shared >= "
                f"{policy.min_incoming_share_shared:.2f} durante "
                f"{policy.confirmation_sessions_required} sesion(es) consecutiva(s); "
                "transicion forzada al terminar el solapamiento (fin de la exportacion "
                "del contrato saliente)"
            ),
            "confidence": "BAJA",
        }

    after = [d for d in combined_dates if d > signal_date]
    effective_date = after[0] if after else None
    return {
        "signal_date": signal_date,
        "effective_date": effective_date,
        "rule": (
            f"share_shared >= {policy.min_incoming_share_shared:.2f} durante "
            f"{policy.confirmation_sessions_required} sesion(es) consecutiva(s), confirmado "
            f"en {signal_date}; efectivo desde la jornada siguiente observada"
        ),
        "confidence": "ALTA",
    }


def compute_basis_evolution(out_rows: pd.DataFrame, in_rows: pd.DataFrame, overlap_dates: list) -> pd.DataFrame:
    """Basis (diferencia y ratio de precio) entre saliente y entrante, en timestamps SIMULTANEOS.

    Entrada: filas de ambos contratos y la lista de fechas de solapamiento.
    Transformacion: para cada ``trading_date``, se emparejan las barras de
    ambos contratos que comparten EXACTAMENTE el mismo timestamp (mismo
    minuto) y se calcula, por par: ``diff_points = close_in - close_out``
    y ``ratio = close_in / close_out``. Se agregan por dia (media,
    mediana, desvio) -- exactamente lo que exige la seccion 7 de la
    tarea: "usa precios comparables del mismo timestamp siempre que sea
    posible", nunca el ultimo precio de uno contra el primero del otro en
    instantes distintos (eso seria mezclar basis con movimiento genuino
    del mercado -- ver ``compute_no_overlap_evidence`` para el caso sin
    timestamps simultaneos).
    Salida: ``DataFrame``, una fila por ``trading_date``, con las columnas
    de agregacion del basis y ``n_pairs`` (numero de minutos simultaneos
    usados ese dia -- la evidencia de cuan solida es la medicion).
    """
    records = []
    for d in overlap_dates:
        o_day = out_rows[out_rows["trading_date"] == d][["ts_utc", "close"]].rename(columns={"close": "close_out"})
        i_day = in_rows[in_rows["trading_date"] == d][["ts_utc", "close"]].rename(columns={"close": "close_in"})
        merged = o_day.merge(i_day, on="ts_utc", how="inner")
        if len(merged) == 0:
            records.append({"trading_date": d, "n_pairs": 0})
            continue
        diff = merged["close_in"] - merged["close_out"]
        ratio = merged["close_in"] / merged["close_out"]
        records.append(
            {
                "trading_date": d,
                "n_pairs": len(merged),
                "diff_points_mean": float(diff.mean()),
                "diff_points_median": float(diff.median()),
                "diff_points_std": float(diff.std()) if len(merged) > 1 else 0.0,
                "ratio_mean": float(ratio.mean()),
                "ratio_median": float(ratio.median()),
                "diff_pct_mean": float((diff / merged["close_out"]).mean() * 100),
            }
        )
    return pd.DataFrame(records)


def compute_no_overlap_evidence(out_rows: pd.DataFrame, in_rows: pd.DataFrame) -> dict:
    """Documenta una transicion SIN timestamps simultaneos (seccion 6 de la tarea).

    Entrada: todas las filas de los dos contratos de una transicion sin
    solapamiento.
    Transformacion: se toma la ULTIMA barra observada del saliente y la
    PRIMERA del entrante -- son los unicos dos puntos de referencia que
    existen. La distancia temporal y la diferencia de precio entre ambos
    se reportan con ``confidence="BAJA"`` **siempre**: sin precios
    simultaneos no puede descartarse que parte (o toda) la diferencia sea
    movimiento genuino de mercado durante el intervalo, no basis de roll
    -- exactamente la advertencia de la seccion 6 de la tarea. No se
    calcula ningun ratio/diferencia "de ajuste" a partir de esta cifra.
    Salida: ``dict`` con los timestamps/precios de referencia, la
    distancia, la diferencia aparente (SOLO descriptiva) y la
    clasificacion de confianza.
    """
    last_out = out_rows.sort_values("timestamp").iloc[-1]
    first_in = in_rows.sort_values("timestamp").iloc[0]
    distance = first_in["timestamp"] - last_out["timestamp"]
    apparent_diff = float(first_in["close"] - last_out["close"])
    apparent_pct = 100.0 * apparent_diff / last_out["close"]
    return {
        "last_out_ts": last_out["timestamp"],
        "last_out_close": float(last_out["close"]),
        "first_in_ts": first_in["timestamp"],
        "first_in_close": float(first_in["close"]),
        "distance_minutes": distance.total_seconds() / 60,
        "apparent_diff_points": apparent_diff,
        "apparent_diff_pct": apparent_pct,
        "confidence": "BAJA",
        "note": (
            "sin barras simultaneas de ambos contratos: la diferencia observada NO puede "
            "atribuirse integramente a basis de roll -- puede incluir movimiento genuino "
            "de mercado durante el intervalo. Solo descriptiva, no se usa para ajustar nada."
        ),
    }


# ---------------------------------------------------------------------------
# Tabla de transiciones (orquestador de las funciones anteriores)
# ---------------------------------------------------------------------------

@dataclass
class TransitionResult:
    out_file: str
    in_file: str
    out_contract: str
    in_contract: str
    transition_type: str
    overlap_dates: list = field(default_factory=list)
    daily_evidence: pd.DataFrame | None = None
    basis_evolution: pd.DataFrame | None = None
    rollover: dict | None = None
    no_overlap_evidence: dict | None = None


def classify_transition(
    out_file: str, in_file: str, rows: pd.DataFrame, expected_by_date: dict, policy: RollPolicy
) -> TransitionResult:
    """Caracteriza UNA transicion entre dos archivos consecutivos del conjunto de investigacion.

    Entrada: los dos nombres de archivo (saliente, entrante), TODAS las
    filas de investigacion ya con ``trading_date``/``contract``
    (``attach_trading_date_and_contract``), la grilla esperada por fecha
    (para cobertura) y la politica de rollover.

    Transformacion: separa las filas de cada archivo; calcula las fechas
    de solapamiento (interseccion de ``trading_date`` de ambos, EXCLUYENDO
    filas fuera de grilla -- ``trading_date`` nulo). Si hay alguna fecha
    en comun -> rama ``OVERLAP`` (evidencia diaria, basis, regla de
    crossover). Si no hay ninguna -> rama ``NO_OVERLAP`` (evidencia de
    borde, confianza BAJA por diseño).

    Salida: ``TransitionResult`` con todo lo calculado, listo para
    volcarse a las tablas de salida (``build_transitions_and_evidence``).
    """
    out_rows = rows[rows["source_file"] == out_file]
    in_rows = rows[rows["source_file"] == in_file]
    out_dates = set(out_rows["trading_date"].dropna())
    in_dates = set(in_rows["trading_date"].dropna())
    overlap_dates = sorted(out_dates & in_dates)

    result = TransitionResult(
        out_file=out_file,
        in_file=in_file,
        out_contract=parse_contract_label(out_file),
        in_contract=parse_contract_label(in_file),
        transition_type=TRANSITION_OVERLAP if overlap_dates else TRANSITION_NO_OVERLAP,
        overlap_dates=overlap_dates,
    )

    if overlap_dates:
        combined_dates = sorted(out_dates | in_dates)
        result.daily_evidence = compute_overlap_daily_evidence(out_rows, in_rows, overlap_dates, expected_by_date)
        result.basis_evolution = compute_basis_evolution(out_rows, in_rows, overlap_dates)
        result.rollover = determine_overlap_rollover(result.daily_evidence, combined_dates, policy)
    else:
        result.no_overlap_evidence = compute_no_overlap_evidence(out_rows, in_rows)
        result.rollover = {
            "signal_date": None,
            "effective_date": in_rows["trading_date"].dropna().min(),
            "rule": "sin solapamiento: el saliente no tiene ninguna barra desde la primera fecha del entrante -- no hay ambiguedad que resolver por volumen",
            "confidence": "ALTA",
        }

    return result


def build_transitions_and_evidence(
    rows: pd.DataFrame, files_order: list, schedule: SessionSchedule, policy: RollPolicy
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[TransitionResult]]:
    """Ejecuta ``classify_transition`` sobre las 21 transiciones consecutivas.

    Entrada: filas de investigacion completas, la lista de archivos EN SU
    ORDEN CRONOLOGICO (``config.research_files``, ya ordenada asi), el
    calendario y la politica.
    Transformacion: recorre pares consecutivos ``(files[i], files[i+1])``
    -- nunca compara archivos no adyacentes, exactamente como TDA-01 hizo
    con los huecos internos (una transicion de contrato es un fenomeno
    distinto de un hueco dentro de un mismo archivo).
    Salida: tupla ``(transitions_df, overlap_daily_evidence_df,
    no_overlap_evidence_df, transition_results)`` -- las tres tablas listas
    para CSV, mas la lista de objetos ``TransitionResult`` completos (para
    el resto del pipeline, que necesita el detalle, no solo el resumen).
    """
    full_grid = expected_bar_grid_frame(schedule)
    expected_by_date = full_grid.groupby("trading_date").size().to_dict()

    results = []
    for out_file, in_file in zip(files_order[:-1], files_order[1:]):
        results.append(classify_transition(out_file, in_file, rows, expected_by_date, policy))

    transition_rows = []
    overlap_evidence_rows = []
    no_overlap_rows = []
    for r in results:
        transition_rows.append(
            {
                "out_file": r.out_file,
                "in_file": r.in_file,
                "out_contract": r.out_contract,
                "in_contract": r.in_contract,
                "transition_type": r.transition_type,
                "n_overlap_dates": len(r.overlap_dates),
                "overlap_start": r.overlap_dates[0] if r.overlap_dates else None,
                "overlap_end": r.overlap_dates[-1] if r.overlap_dates else None,
                "signal_date": r.rollover["signal_date"],
                "effective_date": r.rollover["effective_date"],
                "rollover_rule": r.rollover["rule"],
                "rollover_confidence": r.rollover["confidence"],
            }
        )
        if r.daily_evidence is not None:
            de = r.daily_evidence.copy()
            de.insert(0, "in_file", r.in_file)
            de.insert(0, "out_file", r.out_file)
            if r.basis_evolution is not None and len(r.basis_evolution) > 0:
                de = de.merge(r.basis_evolution, on="trading_date", how="left")
            overlap_evidence_rows.append(de)
        if r.no_overlap_evidence is not None:
            row = {"out_file": r.out_file, "in_file": r.in_file, **r.no_overlap_evidence}
            no_overlap_rows.append(row)

    transitions_columns = [
        "out_file", "in_file", "out_contract", "in_contract", "transition_type",
        "n_overlap_dates", "overlap_start", "overlap_end", "signal_date", "effective_date",
        "rollover_rule", "rollover_confidence",
    ]
    transitions_df = pd.DataFrame(transition_rows, columns=transitions_columns)
    overlap_evidence_df = (
        pd.concat(overlap_evidence_rows, ignore_index=True) if overlap_evidence_rows else pd.DataFrame()
    )
    no_overlap_df = pd.DataFrame(no_overlap_rows)
    return transitions_df, overlap_evidence_df, no_overlap_df, results


# ---------------------------------------------------------------------------
# Calendario de contrato activo + serie canonica
# ---------------------------------------------------------------------------

def build_active_contract_calendar(
    rows: pd.DataFrame, files_order: list, transition_results: list[TransitionResult]
) -> pd.DataFrame:
    """Decide, para CADA ``trading_date`` del conjunto de investigacion, el contrato activo.

    Entrada: filas de investigacion, orden de archivos, y los
    ``TransitionResult`` ya calculados (con su ``effective_date``).

    Transformacion, en dos pasadas:
    1. **Fechas sin ambigüedad**: cada archivo "posee" por defecto todas
       sus propias ``trading_date`` -- correcto para las 19 transiciones
       ``NO_OVERLAP`` (nunca hay dos archivos con datos en la misma
       fecha) y para cualquier fecha de un archivo con solapamiento que
       cae FUERA de la ventana de solapamiento de ambos lados.
    2. **Fechas dentro de una ventana de solapamiento** (las 2
       transiciones ``OVERLAP``): se decide segun ``effective_date`` --
       antes de ella, el saliente esta activo (``PRE_CROSSOVER_...``);
       desde ella (inclusive), el entrante (``POST_CROSSOVER_...``).
       **Regla de respaldo** (generalizacion de la regla 11 heredada,
       §7.4 de ``MNQ_DATA_PRIOR_KNOWLEDGE.md``): si, dentro de la ventana
       de solapamiento, el contrato que DEBERIA estar formalmente activo
       esa fecha (segun el punto anterior) tiene CERO barras ese dia
       mientras el otro SI tiene, se usa la cobertura REAL del otro SOLO
       para esa fecha -- sin adelantar ni atrasar el estado formal de
       cruce, y sin mezclar ni promediar. Sobre el conjunto de
       investigacion real, esta regla de respaldo no se activa ninguna
       vez (ver informe, ambas ventanas de solapamiento tienen barras de
       los dos contratos en todas sus fechas) -- se mantiene porque es la
       logica correcta y general, no una prediccion de que vaya a
       necesitarse.

    Salida: ``DataFrame`` con columnas ``trading_date``, ``active_file``,
    ``active_contract``, ``reason``.
    """
    file_dates = {f: set(rows.loc[rows["source_file"] == f, "trading_date"].dropna()) for f in files_order}

    active: dict = {}
    for f in files_order:
        for d in file_dates[f]:
            active[d] = (f, REASON_NORMAL)

    for r in transition_results:
        if r.transition_type != TRANSITION_OVERLAP:
            continue
        out_f, in_f = r.out_file, r.in_file
        effective_date = r.rollover["effective_date"]
        overlap_dates = set(r.overlap_dates)

        for d in overlap_dates:
            if effective_date is not None and d >= effective_date:
                active[d] = (in_f, REASON_POST_CROSSOVER)
            else:
                active[d] = (out_f, REASON_PRE_CROSSOVER)

        # Regla de respaldo (generalizacion de la regla 11 heredada): fechas
        # donde SOLO uno de los dos contratos tiene barras (no estan en
        # overlap_dates, que exige ambos > 0 ese dia). No se acota por una
        # ventana adicional -- ``file_dates[in_f]``/``file_dates[out_f]``
        # ya estan, por construccion, acotadas al propio archivo (un
        # contrato trimestral), asi que cualquier fecha "solo entrante"
        # anterior a la fecha efectiva (o "solo saliente" posterior a
        # ella) es, por definicion, una fecha dentro de la transicion.
        only_in = file_dates[in_f] - file_dates[out_f]
        only_out = file_dates[out_f] - file_dates[in_f]
        for d in only_in:
            if effective_date is None or d < effective_date:
                active[d] = (in_f, REASON_ZERO_BAR_FALLBACK)
        for d in only_out:
            if effective_date is not None and d >= effective_date:
                active[d] = (out_f, REASON_ZERO_BAR_FALLBACK)

    records = [
        {"trading_date": d, "active_file": f, "active_contract": parse_contract_label(f), "reason": reason}
        for d, (f, reason) in active.items()
    ]
    return pd.DataFrame(records).sort_values("trading_date").reset_index(drop=True)


def build_canonical_series(rows: pd.DataFrame, active_calendar: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construye la serie con EXACTAMENTE un contrato activo por timestamp.

    Entrada: filas de investigacion (con ``trading_date``) y el
    calendario de contrato activo.

    Transformacion: para cada fila, se compara su ``source_file`` contra
    el ``active_file`` que le corresponde a su ``trading_date`` -- se
    conserva la fila si coinciden, se descarta (con motivo trazado) si
    no. Filas sin ``trading_date`` valida (fuera de grilla, TDA-02 §8) se
    descartan tambien, con su propio motivo -- TDA-03 no decide su
    contrato activo porque no pertenecen a ninguna sesion.

    Verificacion de conservacion (bloqueante, regla 10-11 heredada):
    ``len(rows) == len(canonical) + len(discarded)``, comprobado con un
    ``assert`` antes de devolver -- igual que TDA-00 aborta si falla su
    propia verificacion de conservacion.

    Salida: tupla ``(canonical_df, discarded_df)``. ``canonical_df`` esta
    ordenado por timestamp, con columnas originales intactas
    (``source_file``, ``contract``, ``trading_date``, OHLCV crudo) mas
    ``segment_id`` (entero que cambia en cada frontera de contrato, para
    que TDA-04 pueda agrupar sin tener que releer el archivo de
    transiciones). ``discarded_df`` tiene las mismas columnas mas
    ``discard_reason``.
    """
    calendar_map = dict(zip(active_calendar["trading_date"], active_calendar["active_file"]))

    def _expected_active_file(trading_date):
        if pd.isna(trading_date):
            return None
        return calendar_map.get(trading_date)

    rows = rows.copy()
    rows["_expected_active_file"] = rows["trading_date"].map(_expected_active_file)

    is_out_of_grid = rows["trading_date"].isna()
    is_kept = (~is_out_of_grid) & (rows["source_file"] == rows["_expected_active_file"])

    canonical = rows.loc[is_kept].drop(columns=["_expected_active_file"]).sort_values("timestamp").reset_index(drop=True)

    discarded = rows.loc[~is_kept].drop(columns=["_expected_active_file"]).copy()
    discard_reason = np.where(
        is_out_of_grid.loc[~is_kept], "OUT_OF_GRID_NO_TRADING_DATE", DISCARD_NON_ACTIVE_ON_OVERLAP_DATE
    )
    discarded["discard_reason"] = discard_reason
    discarded = discarded.sort_values("timestamp").reset_index(drop=True)

    assert len(rows) == len(canonical) + len(discarded), (
        f"fallo de conservacion en build_canonical_series: {len(rows)} != "
        f"{len(canonical)} + {len(discarded)}"
    )
    assert canonical["timestamp"].is_monotonic_increasing
    assert not canonical["timestamp"].duplicated().any()

    contract_change = canonical["contract"].ne(canonical["contract"].shift(1))
    canonical["segment_id"] = contract_change.cumsum().astype(int)

    return canonical, discarded


# ---------------------------------------------------------------------------
# Metodos de ajuste (evaluados, no forzados)
# ---------------------------------------------------------------------------

def compute_adjustment_factors(transition_results: list[TransitionResult], files_order: list) -> pd.DataFrame:
    """Factores de ajuste aditivo y por ratio, SOLO donde el basis es medible.

    Entrada: los ``TransitionResult`` (con su basis medido, si lo hay) y
    el orden de archivos.

    Transformacion -- por que NO se puede ajustar todo el historial: el
    basis (diferencia/ratio de precio en timestamps simultaneos) solo es
    medible en una transicion ``OVERLAP`` (seccion 7 de la tarea: "usa
    precios comparables del mismo timestamp"). De las 21 transiciones del
    conjunto de investigacion, solo 2 son ``OVERLAP`` (Z24→H25, H25→M25)
    -- las otras 19 no tienen NINGUN par de precios simultaneos, asi que
    cualquier "basis" que se les asignara seria, por construccion, una
    mezcla no separable de basis de roll y movimiento de mercado (la
    misma advertencia de ``compute_no_overlap_evidence``). Por tanto, el
    ajuste retrospectivo solo puede encadenarse de forma defendible a lo
    largo de un tramo CONTIGUO de transiciones con basis medido -- en este
    conjunto de investigacion, exactamente los 3 contratos mas recientes
    (Z24, H25, M25), conectados por las 2 transiciones ``OVERLAP``. El
    resto de la historia (H20...U24) NO recibe factor de ajuste: se deja
    ``NaN``, explicitamente, en vez de forzar una cifra no defendible.

    El factor se ancla en el contrato MAS RECIENTE (el ultimo de la
    cadena contigua, es decir el que ya no tiene ningun roll posterior
    medido dentro de esa cadena): su factor es 1.0 (ratio) / 0.0
    (aditivo). Cada contrato anterior de la cadena hereda el factor del
    siguiente, ACUMULADO con el basis de SU PROPIO roll hacia adelante
    -- exactamente la propiedad que cita el fundamento Tsay de esta etapa:
    "el factor de ajuste de un segmento depende de rolls posteriores,
    pero es constante DENTRO del segmento".

    Salida: ``DataFrame`` con ``contract``, ``source_file``,
    ``ratio_factor``, ``diff_factor`` (``NaN`` fuera de la cadena
    contigua con basis medido), y ``basis_chain`` (``True``/``False``).
    """
    basis_by_pair = {}
    for r in transition_results:
        if r.transition_type == TRANSITION_OVERLAP and r.rollover.get("effective_date") is not None:
            eff = r.rollover["effective_date"]
            be = r.basis_evolution
            row = be[be["trading_date"] == eff]
            if len(row) == 0:
                continue
            basis_by_pair[(r.out_file, r.in_file)] = {
                "diff": float(row.iloc[0]["diff_points_mean"]),
                "ratio": float(row.iloc[0]["ratio_mean"]),
            }

    ratio_factor = {f: float("nan") for f in files_order}
    diff_factor = {f: float("nan") for f in files_order}
    in_chain = {f: False for f in files_order}

    for i in range(len(files_order) - 1, -1, -1):
        f = files_order[i]
        if i == len(files_order) - 1:
            ratio_factor[f] = 1.0
            diff_factor[f] = 0.0
            in_chain[f] = True
            continue
        next_f = files_order[i + 1]
        pair = (f, next_f)
        if in_chain[next_f] and pair in basis_by_pair:
            b = basis_by_pair[pair]
            # close_in (next_f) ~= close_out (f) * ratio  =>  para expresar
            # el contrato f en la escala de next_f, se multiplica por ratio_factor[next_f]*b['ratio'].
            ratio_factor[f] = ratio_factor[next_f] * b["ratio"]
            diff_factor[f] = diff_factor[next_f] + b["diff"]
            in_chain[f] = True
        else:
            break  # cadena contigua rota: no seguir propagando hacia atras

    return pd.DataFrame(
        [
            {
                "source_file": f,
                "contract": parse_contract_label(f),
                "ratio_factor": ratio_factor[f],
                "diff_factor": diff_factor[f],
                "basis_chain": in_chain[f],
            }
            for f in files_order
        ]
    )


def apply_adjustments(canonical: pd.DataFrame, adjustment_factors: pd.DataFrame) -> pd.DataFrame:
    """Añade columnas ajustadas SIN sobrescribir las columnas OHLCV crudas.

    Entrada: la serie canonica y la tabla de factores.
    Transformacion: para cada fila, multiplica O/H/L/C por
    ``ratio_factor`` de su ``source_file`` (columnas ``*_adj_ratio``) y
    suma ``diff_factor`` (columnas ``*_adj_diff``). Donde el factor es
    ``NaN`` (fuera de la cadena con basis medido), las columnas ajustadas
    quedan ``NaN`` tambien -- explicito, no un valor inventado.
    Salida: el mismo DataFrame con 8 columnas nuevas
    (``open/high/low/close_adj_ratio`` y ``..._adj_diff``); ``volume`` no
    se ajusta (no es un precio).
    """
    canonical = canonical.merge(
        adjustment_factors[["source_file", "ratio_factor", "diff_factor", "basis_chain"]],
        on="source_file", how="left",
    )
    for col in ["open", "high", "low", "close"]:
        canonical[f"{col}_adj_ratio"] = canonical[col] * canonical["ratio_factor"]
        canonical[f"{col}_adj_diff"] = canonical[col] + canonical["diff_factor"]
    return canonical


# ---------------------------------------------------------------------------
# Mascara de roll
# ---------------------------------------------------------------------------

def build_roll_mask(canonical: pd.DataFrame, transitions_df: pd.DataFrame) -> pd.DataFrame:
    """Mascara persistente de frontera de contrato, reutilizable por TDA-04.

    Entrada: la serie canonica (con ``segment_id``) y la tabla de
    transiciones.
    Transformacion: la primera fila de cada ``segment_id`` (salvo el
    primero de toda la serie) es una frontera de roll -- se le asocia la
    fila de ``transitions_df`` cuyo ``in_file`` coincide con el
    ``source_file`` de esa fila. TDA-04 usara ``is_roll_boundary`` para
    forzar ``r_t = NaN`` en esa barra (el retorno hacia la barra anterior
    cruzaria la frontera de contrato -- exactamente la regla de no-cruce
    que exige TDA-04, seccion "Metodos minimos" del roadmap).
    Salida: ``DataFrame`` con una fila por cada fila de la serie canonica:
    ``timestamp``, ``source_file``, ``contract``, ``trading_date``,
    ``segment_id``, ``is_roll_boundary``, ``prev_contract``,
    ``transition_type``, ``overlap``, ``rollover_rule``.
    """
    mask = canonical[["timestamp", "source_file", "contract", "trading_date", "segment_id"]].copy()
    mask["prev_contract"] = mask["contract"].shift(1)
    mask["is_roll_boundary"] = mask["contract"].ne(mask["prev_contract"]) & mask["prev_contract"].notna()

    trans_by_in_file = transitions_df.set_index("in_file")
    mask["transition_type"] = mask["source_file"].map(trans_by_in_file["transition_type"]).where(mask["is_roll_boundary"])
    mask["overlap"] = (mask["transition_type"] == TRANSITION_OVERLAP)
    mask["rollover_rule"] = mask["source_file"].map(trans_by_in_file["rollover_rule"]).where(mask["is_roll_boundary"])
    return mask


# ---------------------------------------------------------------------------
# STOP-3
# ---------------------------------------------------------------------------

def find_extreme_jumps(canonical: pd.DataFrame, policy: RollPolicy, top_n: int | None = None) -> pd.DataFrame:
    """Rankea las mayores discontinuidades de precio NO atribuibles a un roll (STOP-3).

    Entrada: la serie canonica (con ``segment_id``) y la politica.

    Por que NO se usa un umbral global de MAD (diseño descartado durante
    esta etapa, documentado aqui para que no se reintente sin motivo): el
    precio de MNQ cambia de nivel en un factor ~3x a lo largo del
    conjunto de investigacion (de ~7.000 a ~22.000 puntos), y su
    volatilidad tiene regimenes muy distintos (marzo 2020 frente a un
    tramo tranquilo de 2023) -- la distribucion de saltos de 1 minuto es
    fuertemente no estacionaria y de colas pesadas. Un umbral fijo de
    "K veces la MAD global" (probado con K entre 15 y 50, en puntos y en
    terminos relativos) marca miles de barras perfectamente normales como
    "candidatas", precisamente el resultado que la seccion 13 de la tarea
    advierte NO forzar ("no la corrijas ni la escondas" tambien implica no
    generar una lista de miles de falsos positivos que nadie puede
    revisar). En su lugar, se usa un diseño mas simple y mas util para
    revision humana: el ranking de las ``top_n`` mayores discontinuidades
    RELATIVAS (en % del precio previo, no en puntos absolutos -- para que
    el ranking no este dominado trivialmente por los años con el precio
    mas alto), acotado a un tamaño revisable.

    Transformacion: se calcula ``|close_t - close_{t-1}| / close_{t-1}``
    SOLO entre barras CONSECUTIVAS de 1 minuto exacto y del MISMO
    ``segment_id`` (mismo contrato, sin frontera de roll de por medio --
    una barra que SI cruza una frontera de roll no es candidata a
    "discontinuidad sin explicar": ya esta explicada, es el roll, y su
    salto se mide aparte como basis en ``compute_basis_evolution``). Para
    cada una de las ``top_n``, se añade evidencia de contexto que ayuda a
    la revision: el volumen de la barra (un salto en volumen alto es
    compatible con una reaccion de mercado real; uno en volumen muy bajo
    es mas sospechoso de ser un error de dato aislado) y si el precio
    REVIERTE mas de la mitad del salto en la barra siguiente (firma
    clasica de un tick erroneo aislado que luego "se corrige" solo).

    Salida: ``DataFrame``, ``top_n`` filas ordenadas de mayor a menor
    salto relativo, con ``timestamp``, ``source_file``, ``trading_date``,
    ``close_prev``, ``close``, ``abs_diff_points``, ``rel_diff_pct``,
    ``volume``, ``reverts_next_bar``.
    """
    top_n = top_n if top_n is not None else policy.extreme_jump_top_n
    df = canonical.sort_values("timestamp").reset_index(drop=True)
    same_segment = df["segment_id"].eq(df["segment_id"].shift(1))
    one_minute_apart = (df["timestamp"] - df["timestamp"].shift(1)).dt.total_seconds().eq(60)
    candidate_mask = same_segment & one_minute_apart

    close_prev = df["close"].shift(1)
    abs_diff = (df["close"] - close_prev).abs()
    rel_diff = abs_diff / close_prev

    ranked_idx = rel_diff[candidate_mask].sort_values(ascending=False).head(top_n).index
    out = df.loc[ranked_idx, ["timestamp", "source_file", "contract", "trading_date", "close", "volume"]].copy()
    out["close_prev"] = close_prev.loc[ranked_idx]
    out["abs_diff_points"] = abs_diff.loc[ranked_idx]
    out["rel_diff_pct"] = rel_diff.loc[ranked_idx] * 100

    next_close = df["close"].shift(-1)
    next_diff = (next_close - df["close"])
    jump_diff = df["close"] - close_prev
    # Revierte si el siguiente movimiento tiene signo OPUESTO al salto y
    # recorre mas de la mitad de su magnitud -- firma de un tick aislado
    # que "vuelve" solo, no de un nuevo nivel de precio sostenido.
    reverts = (np.sign(next_diff) != np.sign(jump_diff)) & (next_diff.abs() > 0.5 * jump_diff.abs())
    out["reverts_next_bar"] = reverts.loc[ranked_idx]

    # Umbral de "volumen bajo" = percentil 5 del volumen de TODA la serie
    # canonica (no solo de los candidatos) -- un salto que ademas ocurre
    # en una barra de volumen tipicamente bajo es la combinacion mas
    # compatible con un tick aislado/erroneo; un salto en volumen alto es
    # mas compatible con una reaccion de mercado real (muchos
    # participantes operando al mismo tiempo).
    low_volume_threshold = df["volume"].quantile(0.05)
    out["low_volume"] = out["volume"] < low_volume_threshold
    out["suspicious"] = out["reverts_next_bar"] & out["low_volume"]

    return out.sort_values("rel_diff_pct", ascending=False).reset_index(drop=True)


def check_stop3(extreme_jumps: pd.DataFrame) -> dict:
    """Evalua el criterio STOP-3: ¿alguna discontinuidad grande queda sin explicar?

    Entrada: la salida de ``find_extreme_jumps`` -- por construccion, ya
    excluye las barras que cruzan una frontera de roll (esas SI estan
    explicadas: son el roll, no una discontinuidad "sin explicar").

    Criterio de disparo: ``triggered`` es ``True`` si ALGUNA de las
    ``top_n`` discontinuidades es ``suspicious`` (revierte mas de la
    mitad de su magnitud en la barra siguiente Y ocurre en volumen bajo,
    percentil 5 o menor de toda la serie -- la combinacion mas compatible
    con un tick aislado/erroneo en vez de una reaccion de mercado real).
    El mero TAMAÑO del salto NO dispara STOP-3 por si solo: un salto
    grande en volumen alto, sin reversion inmediata, es la firma esperada
    de un evento de mercado genuino (una sorpresa macro, un shock de
    volatilidad), no de un problema de datos -- exactamente la distincion
    que pide la seccion 13 de la tarea ("no la corrijas ni la escondas",
    pero tampoco fuerces STOP-3 sobre volatilidad real).

    Esta funcion NO decide con certeza absoluta si un candidato es un
    error de dato o un evento real -- eso es revision humana (TH06/TDA-13
    mas adelante, cuando exista contexto adicional). Solo aplica el
    criterio explicito de la tarea para decidir si detenerse.

    Salida: ``{"triggered": bool, "n_candidates": int, "n_suspicious": int}``.
    """
    n_suspicious = int(extreme_jumps["suspicious"].sum()) if "suspicious" in extreme_jumps.columns else 0
    return {
        "triggered": n_suspicious > 0,
        "n_candidates": int(len(extreme_jumps)),
        "n_suspicious": n_suspicious,
    }


# ---------------------------------------------------------------------------
# Tabla de invariancia (TH05)
# ---------------------------------------------------------------------------

def build_invariance_table() -> pd.DataFrame:
    """Clasifica los estadisticos previstos por el roadmap segun su invariancia al ajuste.

    Esta funcion NO calcula nada sobre los datos: es una clasificacion
    MATEMATICA, razonada por escrito, de como cambia (o no) cada
    estadistico si se multiplica un segmento entero por una constante
    (ajuste por ratio) o se le suma una constante (ajuste aditivo) --
    exactamente el "test operativo de invariancia a escala" que exige el
    fundamento Tsay de esta etapa. Se construye ANTES de ejecutar
    cualquier etapa posterior (TDA-04 en adelante), para que esas etapas
    sepan de antemano que estadisticos pueden calcular sobre la serie
    ajustada y cuales solo sobre la serie cruda por contrato.

    Regla general: un estadistico es "invariante" a una transformacion si
    su valor NUMERICO no cambia al aplicarla -- no si "sigue siendo
    calculable", sino si da EXACTAMENTE el mismo numero.
    """
    rows = [
        {
            "statistic": "Nivel de precio (C_t)",
            "invariant_to_ratio": False,
            "invariant_to_diff": False,
            "reason": "Un nivel de precio cambia por definicion bajo cualquiera de los dos ajustes (k*C, C+c).",
        },
        {
            "statistic": "Diferencia en puntos (C_t - C_{t-1}, mismo segmento)",
            "invariant_to_ratio": False,
            "invariant_to_diff": True,
            "reason": "Ratio: k*C_t - k*C_{t-1} = k*(C_t-C_{t-1}) != original salvo k=1. "
            "Aditivo: (C_t+c)-(C_{t-1}+c) = C_t-C_{t-1}, exacto -- es la propiedad que define al ajuste aditivo.",
        },
        {
            "statistic": "Retorno simple (R_t = C_t/C_{t-1} - 1, mismo segmento)",
            "invariant_to_ratio": True,
            "invariant_to_diff": False,
            "reason": "Ratio: (k C_t)/(k C_{t-1}) - 1 = R_t, exacto -- es la propiedad que define al ajuste por ratio. "
            "Aditivo: (C_t+c)/(C_{t-1}+c) - 1 != R_t en general.",
        },
        {
            "statistic": "Log-retorno (r_t = ln(C_t/C_{t-1}), mismo segmento)",
            "invariant_to_ratio": True,
            "invariant_to_diff": False,
            "reason": "Mismo argumento que el retorno simple: ln(kC_t/kC_{t-1}) = ln(C_t/C_{t-1}); "
            "ln((C_t+c)/(C_{t-1}+c)) != r_t en general.",
        },
        {
            "statistic": "Rango absoluto en puntos (H_t - L_t)",
            "invariant_to_ratio": False,
            "invariant_to_diff": True,
            "reason": "Misma logica que la diferencia en puntos: el aditivo preserva diferencias, el ratio las escala por k.",
        },
        {
            "statistic": "Rango relativo/log (ln(H_t/L_t))",
            "invariant_to_ratio": True,
            "invariant_to_diff": False,
            "reason": "Misma logica que el log-retorno: ln(kH/kL) = ln(H/L); el aditivo NO preserva el cociente.",
        },
        {
            "statistic": "Medidas expresadas en ticks (p.ej. (C_t-C_{t-1})/tick_size)",
            "invariant_to_ratio": False,
            "invariant_to_diff": True,
            "reason": "tick_size es una constante FIJA del contrato (no se reescala con el ajuste). Bajo ratio, el numerador "
            "se escala pero el tick no -- el cociente cambia, y ademas el precio ajustado puede dejar de caer en la grilla "
            "de tick exacta. Bajo aditivo, la diferencia en el numerador es invariante, luego el cociente tambien.",
        },
        {
            "statistic": "Varianza / desvio de retornos (Var(r_t), Var(R_t))",
            "invariant_to_ratio": True,
            "invariant_to_diff": False,
            "reason": "Se calcula sobre valores que ya son, uno a uno, identicos bajo ratio (ver retorno simple/log-retorno "
            "arriba) -- luego cualquier funcion de esos valores (incluida la varianza) tambien es identica. Bajo aditivo, "
            "los propios retornos cambian valor a valor, luego su varianza tambien puede cambiar.",
        },
        {
            "statistic": "Varianza / desvio de diferencias en puntos (Var(C_t-C_{t-1}))",
            "invariant_to_ratio": False,
            "invariant_to_diff": True,
            "reason": "Analogo al anterior pero para diferencias en puntos: son identicas bar a bar bajo aditivo (invariante), "
            "escaladas por k bajo ratio (Var escala por k^2, no invariante).",
        },
        {
            "statistic": "Cuantiles de retornos (simples o log)",
            "invariant_to_ratio": True,
            "invariant_to_diff": False,
            "reason": "Los retornos individuales son identicos bajo ratio -> su distribucion empirica completa (y por tanto "
            "cualquier cuantil) tambien lo es. Bajo aditivo cambian valor a valor.",
        },
        {
            "statistic": "Cuantiles de niveles de precio o de diferencias en puntos",
            "invariant_to_ratio": False,
            "invariant_to_diff": "parcial",
            "reason": "Niveles: dependientes de ambos ajustes (igual que el nivel de precio). Diferencias en puntos: "
            "invariantes al aditivo (ver fila correspondiente), dependientes del ratio.",
        },
        {
            "statistic": "Correlacion / ACF de retornos",
            "invariant_to_ratio": True,
            "invariant_to_diff": False,
            "reason": "Los retornos bajo ratio son identicos valor a valor (no solo 'proporcionales'), luego cualquier "
            "estadistico calculado sobre ellos -- incluida su autocorrelacion -- coincide exactamente. Bajo aditivo, al "
            "cambiar los propios retornos, la ACF puede cambiar tambien (aunque la formula de correlacion sea invariante "
            "a transformaciones afines de UNA variable, aqui el aditivo transforma el PRECIO, no el retorno directamente, "
            "y esa transformacion no es afin sobre el retorno).",
        },
        {
            "statistic": "Volumen (V_t)",
            "invariant_to_ratio": True,
            "invariant_to_diff": True,
            "reason": "Ninguno de los dos ajustes de PRECIO toca la columna de volumen -- permanece exactamente igual "
            "bajo cualquiera de los dos metodos (y tambien sin ajuste).",
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------

@dataclass
class TDA03Result:
    schedule: SessionSchedule
    policy: RollPolicy
    rows: pd.DataFrame
    transitions: pd.DataFrame
    overlap_daily_evidence: pd.DataFrame
    no_overlap_evidence: pd.DataFrame
    transition_results: list
    active_calendar: pd.DataFrame
    canonical: pd.DataFrame
    discarded: pd.DataFrame
    adjustment_factors: pd.DataFrame
    roll_mask: pd.DataFrame
    extreme_jumps: pd.DataFrame
    stop3: dict
    invariance_table: pd.DataFrame


def run_tda03_analysis(config: SnapshotConfig) -> TDA03Result:
    """Orquesta el analisis completo de TDA-03 sobre el conjunto de investigacion.

    Reutiliza, sin duplicar logica: la proteccion del hold-out
    (``holdout_guard.py``), la carga de filas (``tda01_temporal_semantics.
    load_research_rows``) y el calendario nativo de CME (``session_
    calendar.py``, TDA-02). No escribe ningun archivo -- eso es
    responsabilidad de ``run_tda03.py``.
    """
    validate_research_holdout_disjoint(config)

    rows = load_research_rows(config)
    last_timestamps = rows.groupby("source_file")["timestamp"].max().to_dict()
    validate_last_timestamps_before_boundary(config, last_timestamps)

    schedule = build_session_schedule(rows["timestamp"].min(), rows["timestamp"].max())
    rows = attach_trading_date_and_contract(rows, schedule)

    policy = load_roll_policy(config)
    files_order = list(config.research_files)

    transitions_df, overlap_evidence_df, no_overlap_df, transition_results = build_transitions_and_evidence(
        rows, files_order, schedule, policy
    )

    active_calendar = build_active_contract_calendar(rows, files_order, transition_results)
    canonical, discarded = build_canonical_series(rows, active_calendar)

    adjustment_factors = compute_adjustment_factors(transition_results, files_order)
    canonical = apply_adjustments(canonical, adjustment_factors)

    roll_mask = build_roll_mask(canonical, transitions_df)

    extreme_jumps = find_extreme_jumps(canonical, policy)
    stop3 = check_stop3(extreme_jumps)

    invariance_table = build_invariance_table()

    return TDA03Result(
        schedule=schedule,
        policy=policy,
        rows=rows,
        transitions=transitions_df,
        overlap_daily_evidence=overlap_evidence_df,
        no_overlap_evidence=no_overlap_df,
        transition_results=transition_results,
        active_calendar=active_calendar,
        canonical=canonical,
        discarded=discarded,
        adjustment_factors=adjustment_factors,
        roll_mask=roll_mask,
        extreme_jumps=extreme_jumps,
        stop3=stop3,
        invariance_table=invariance_table,
    )
