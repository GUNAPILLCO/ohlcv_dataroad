"""TDA-04 -- Construccion y auditoria de las variables de analisis de 1 minuto.

Implementa la etapa TDA-04 del roadmap
(``docs/methodology/Tsay_OHLCV_analysis_roadmap.md``, seccion "TDA-04"):
construye el "ladrillo estadistico" minimo sobre el que se hara toda la
caracterizacion posterior -- el retorno logaritmico de 1 minuto y un
puñado de variables relacionadas -- **con las reglas de no-cruce
aplicadas**, y audita cuantitativamente cuanto se pierde por cada regla.

Que hace TDA-04, en una frase: para cada barra de la serie canonica de
TDA-03, decide si tiene una barra "anterior comparable" (mismo
``trading_date``, mismo ``segment_id``, exactamente 1 minuto de
diferencia, sin cruzar una frontera de roll) y, solo si la tiene, calcula
el retorno; si no la tiene, dice explicitamente POR QUE no, en vez de
dejar un ``NaN`` mudo.

Que NO hace este modulo (deliberadamente, fuera de alcance de TDA-04):

- No estudia la distribucion de los retornos (momentos, cuantiles, QQ) --
  eso es TDA-07/TDA-11.
- No calcula ACF/PACF ni busca dependencia -- eso es TDA-08/TDA-09.
- No calcula ningun horizonte de prediccion (5/15/30/60/90 minutos) ni
  ningun target -- esta etapa define el retorno REALIZADO de 1 minuto,
  no un objeto para prediccion futura. Por esta razon, el diagnostico de
  escalado de varianza Var(r[h]) vs h que el roadmap lista como "metodo
  minimo" de TDA-04 (y que resuelve TH10) se **difiere explicitamente**
  a una etapa posterior: instruccion directa de esta tarea concreta,
  documentada en ``reports/mnq/TDA04_variables_analisis.md``, seccion
  "Estado de TH07, TH08 y TH10" -- no es un olvido.
- No introduce ninguna ventana horaria (RTH, pre-market, etc.) -- usa
  exclusivamente la sesion/``trading_date`` ya definida por TDA-02.
- No reintroduce las filas que TDA-03 ya descarto (fuera de grilla,
  contrato no activo en fecha de solapamiento).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ohlcv_dataroad.config import SnapshotConfig
from ohlcv_dataroad.ingest.holdout_guard import (
    validate_last_timestamps_before_boundary,
    validate_research_holdout_disjoint,
)

# --- Catalogo de causas de invalidez de r_1m (seccion 9 de la tarea) ------
# Categorias mutuamente EXCLUYENTES para el conteo principal (columna
# "invalid_reason"), en el orden de prioridad que se aplica cuando mas de
# una condicion falla a la vez (ver build_return_validity_mask). Cada
# causa tambien queda disponible como bandera booleana INDEPENDIENTE
# (columnas "is_*"), para poder medir el solapamiento entre causas sin
# perder informacion por la exclusividad del conteo principal.
REASON_FIRST_OBSERVATION = "FIRST_OBSERVATION"
REASON_ROLL_BOUNDARY = "ROLL_BOUNDARY"
REASON_TRADING_DATE_BOUNDARY = "TRADING_DATE_BOUNDARY"
REASON_NON_CONSECUTIVE_MINUTE = "NON_CONSECUTIVE_MINUTE"
REASON_VALID = "VALID"

ALL_REASONS = [
    REASON_FIRST_OBSERVATION,
    REASON_ROLL_BOUNDARY,
    REASON_TRADING_DATE_BOUNDARY,
    REASON_NON_CONSECUTIVE_MINUTE,
]


class RollConsistencyError(Exception):
    """La serie canonica y la mascara de roll de TDA-03 se contradicen.

    TDA-03 garantiza, por construccion, que ``segment_id`` cambia
    EXACTAMENTE cuando ``contract`` cambia, y que ambos cambian
    EXACTAMENTE donde ``roll_mask["is_roll_boundary"]`` es ``True`` (ver
    ``tda03_rolls.build_canonical_series``/``build_roll_mask``). TDA-04
    depende de ``is_roll_boundary`` para decidir ``ROLL_BOUNDARY`` (no
    recalcula la frontera a partir de ``segment_id``, por diseño -- ver
    docstring de ``build_return_validity_mask``), asi que esa garantia es
    un supuesto de entrada, no un hecho que TDA-04 pueda dar por sentado
    sin comprobarlo. Esta excepcion se dispara si algun cambio de
    ``segment_id``/``contract`` no coincide exactamente con
    ``is_roll_boundary`` -- indicaria que la serie canonica y la mascara
    de roll ya no son consistentes entre si (p.ej. porque una se
    regenero y la otra no), y TDA-04 debe abortar en vez de construir
    retornos sobre una base contradictoria.
    """


def load_canonical_and_mask(config: SnapshotConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee la serie canonica y la mascara de roll producidas por TDA-03.

    Entrada: la configuracion (para resolver las rutas de
    ``config.tda03_canonical_series_path`` / ``config.tda03_roll_mask_path`` --
    los MISMOS nombres de archivo que TDA-03 ya declaro, no una copia).
    Transformacion: ninguna -- lectura directa de los dos parquet.
    Salida: ``(canonical, roll_mask)``, ambos ordenados por ``timestamp``
    (la serie canonica ya sale ordenada de TDA-03, pero se reordena aqui
    de forma defensiva por si algun consumidor futuro no lo garantiza).

    TDA-04 nunca abre ningun archivo de ``data/raw/`` ni de
    ``config.holdout_files``: su unica entrada son estos dos artefactos ya
    construidos exclusivamente sobre el conjunto de investigacion.
    """
    canonical = pd.read_parquet(config.tda03_canonical_series_path).sort_values("timestamp").reset_index(drop=True)
    roll_mask = pd.read_parquet(config.tda03_roll_mask_path).sort_values("timestamp").reset_index(drop=True)
    return canonical, roll_mask


# ---------------------------------------------------------------------------
# Mascara de validez de retorno -- el nucleo metodologico de la etapa
# ---------------------------------------------------------------------------

def build_return_validity_mask(canonical: pd.DataFrame, roll_mask: pd.DataFrame) -> pd.DataFrame:
    """Decide, fila por fila, si existe una barra ``t-1`` COMPARABLE para el retorno de ``t``.

    Entrada
    -------
    canonical : serie canonica de TDA-03 (una fila por timestamp, ya con
        exactamente un contrato activo -- TDA-03 garantiza esto, aqui no
        se vuelve a comprobar).
    roll_mask : mascara de roll de TDA-03, con la misma cantidad de filas
        y los mismos timestamps (se verifica con un ``assert``).

    Antes de clasificar nada, se COMPRUEBA (no se asume) que
    ``roll_mask["is_roll_boundary"]`` coincide EXACTAMENTE, fila a fila,
    con un cambio de ``segment_id`` y con un cambio de ``contract`` --
    las tres cosas deben ser la misma frontera, tal como TDA-03 garantiza
    por construccion. Si alguna fila las contradice, se lanza
    ``RollConsistencyError`` de inmediato (ver esa clase para el porque:
    en resumen, TDA-04 usa ``is_roll_boundary`` directamente como fuente
    de verdad para ``ROLL_BOUNDARY`` -- nunca recalculado a partir de
    ``segment_id`` -- asi que esta comprobacion es lo que hace seguro
    reutilizarlo sin releer la logica de TDA-03).

    Por que "la fila anterior en el DataFrame" NO es automaticamente la
    "barra anterior de 1 minuto" -- la idea central de toda esta funcion:
    la serie canonica esta ordenada por timestamp y no tiene duplicados,
    asi que ``shift(1)`` SIEMPRE da una fila -- pero esa fila puede ser de
    OTRO dia de negociacion (tras el corte de mantenimiento nocturno, un
    fin de semana o un feriado), de OTRO contrato (tras un roll), o
    simplemente no ser el minuto inmediatamente anterior (un hueco interno
    de TDA-02, p.ej. un hueco corto durante mercado abierto). Ninguno de
    esos casos es "la barra de hace 1 minuto" aunque este en la fila justo
    encima en la tabla -- por eso NUNCA se puede usar
    ``df["close"].pct_change()`` ciegamente sobre toda la tabla: esa
    funcion calcula exactamente sobre "la fila anterior", sin ninguna de
    estas comprobaciones.

    Transformacion -- las 4 condiciones, evaluadas EN ORDEN DE PRIORIDAD
    (una fila puede fallar varias a la vez; el orden decide cual se
    reporta como ``invalid_reason``; las 4 tambien quedan disponibles como
    banderas booleanas independientes, sin la exclusividad del orden, para
    poder medir el solapamiento entre causas -- seccion 9 de la tarea):

    1. ``FIRST_OBSERVATION`` -- no existe ninguna fila anterior en toda la
       serie (la primera fila de todo el conjunto de investigacion).
       Maxima prioridad: si no hay fila anterior, ninguna otra pregunta
       tiene sentido.
    2. ``ROLL_BOUNDARY`` -- ``roll_mask["is_roll_boundary"]`` es ``True``
       para esta fila (viene de TDA-03: la primera barra de un nuevo
       ``segment_id``, distinto contrato). Se usa la mascara de TDA-03
       DIRECTAMENTE (no se recalcula aqui a partir de ``segment_id``) para
       reutilizar la fuente de verdad de esa etapa; el codigo, ademas,
       verifica por separado que ambas formas de mirarlo coinciden
       siempre (test dedicado).
    3. ``TRADING_DATE_BOUNDARY`` -- ``trading_date`` cambia respecto de la
       fila anterior, SIN ser un roll. Es el caso general de "la sesion
       anterior ya cerro": corte de mantenimiento nocturno, fin de semana,
       feriado completo, cierre anticipado -- cualquier frontera de
       ``trading_date`` de TDA-02, sea cual sea su causa de calendario.
       Nota importante (seccion 3 de la tarea): esto compara
       ``trading_date``, NO la fecha de calendario del timestamp -- una
       sesion que cruza medianoche de Nueva York (23:59 -> 00:00, mismo
       ``trading_date``) NO cae en esta categoria.
    4. ``NON_CONSECUTIVE_MINUTE`` -- mismo ``trading_date``, mismo
       ``segment_id`` (ni roll ni frontera de dia), pero la diferencia de
       timestamp NO es exactamente 60 segundos. Es el residuo: un hueco
       interno de TDA-02 (p.ej. el break secundario pre-2021-06-25, o
       cualquier hueco corto "UNKNOWN") que ocurre DENTRO de una misma
       jornada de negociacion.

    Si NINGUNA de las 4 aplica, el retorno es valido (``r_1m_valid=True``,
    ``invalid_reason="VALID"``).

    Salida
    ------
    ``DataFrame``, una fila por barra de la serie canonica, con:
    ``timestamp``, ``source_file``, ``contract``, ``trading_date``,
    ``segment_id`` (de la fila actual); ``prev_timestamp``,
    ``prev_trading_date``, ``prev_segment_id``, ``prev_contract``,
    ``delta_minutes`` (de la comparacion contra la fila anterior);
    ``is_first_observation``, ``is_roll_boundary``,
    ``is_trading_date_boundary``, ``is_non_consecutive_minute`` (banderas
    independientes); ``r_1m_valid`` (booleano) e ``invalid_reason`` (una
    de las 5 categorias, ``VALID`` incluida).
    """
    df = canonical.sort_values("timestamp").reset_index(drop=True)
    rm = roll_mask.sort_values("timestamp").reset_index(drop=True)
    assert len(df) == len(rm), "canonical y roll_mask tienen distinto numero de filas"
    assert (df["timestamp"].to_numpy() == rm["timestamp"].to_numpy()).all(), (
        "canonical y roll_mask no comparten exactamente los mismos timestamps, en el mismo orden"
    )

    prev_timestamp = df["timestamp"].shift(1)
    prev_trading_date = df["trading_date"].shift(1)
    prev_segment_id = df["segment_id"].shift(1)
    prev_contract = df["contract"].shift(1)

    delta_minutes = (df["timestamp"] - prev_timestamp).dt.total_seconds() / 60.0

    is_first = prev_timestamp.isna().to_numpy()
    is_roll = rm["is_roll_boundary"].fillna(False).to_numpy() & (~is_first)

    # --- Validacion bloqueante: la mascara de roll y la serie canonica ---
    # deben describir EXACTAMENTE la misma frontera. TDA-04 usa
    # ``is_roll`` (leido de tda03_roll_mask) como la fuente de verdad para
    # ROLL_BOUNDARY -- nunca recalcula la frontera a partir de
    # ``segment_id`` -- precisamente porque reutilizar la fuente de
    # verdad de TDA-03 es mas simple que duplicar su logica. Pero esa
    # eleccion de diseño depende de un supuesto (segment_changed ==
    # contract_changed == is_roll_boundary, garantizado por TDA-03 en su
    # propia construccion) que aqui se COMPRUEBA, no se asume: si algun
    # dia la serie canonica y la mascara se regeneraran por separado y
    # dejaran de coincidir, TDA-04 debe abortar con un error claro en vez
    # de construir retornos (o "no-retornos") sobre una base
    # contradictoria.
    segment_changed = (~is_first) & (df["segment_id"].to_numpy() != prev_segment_id.to_numpy())
    contract_changed = (~is_first) & (df["contract"].to_numpy() != prev_contract.to_numpy())
    consistent = (segment_changed == contract_changed) & (contract_changed == is_roll)
    if not consistent.all():
        bad_positions = np.where(~consistent)[0]
        preview = df.iloc[bad_positions[:5]][["timestamp", "source_file", "contract", "segment_id"]].copy()
        preview["segment_changed"] = segment_changed[bad_positions[:5]]
        preview["contract_changed"] = contract_changed[bad_positions[:5]]
        preview["is_roll_boundary"] = is_roll[bad_positions[:5]]
        raise RollConsistencyError(
            f"{len(bad_positions)} fila(s) donde segment_changed / contract_changed / "
            "is_roll_boundary NO coinciden entre si -- la serie canonica "
            "(tda03_serie_continua.parquet) y la mascara de roll "
            "(tda03_roll_mask.parquet) son inconsistentes entre si. "
            f"Primeras filas afectadas:\n{preview.to_string(index=False)}"
        )

    is_trading_date_boundary = (~is_first) & (df["trading_date"].to_numpy() != prev_trading_date.to_numpy())
    is_non_consecutive = (~is_first) & (delta_minutes.to_numpy() != 1.0)

    valid = (~is_first) & (~is_roll) & (~is_trading_date_boundary) & (~is_non_consecutive)

    invalid_reason = np.select(
        [is_first, is_roll, is_trading_date_boundary, is_non_consecutive],
        [REASON_FIRST_OBSERVATION, REASON_ROLL_BOUNDARY, REASON_TRADING_DATE_BOUNDARY, REASON_NON_CONSECUTIVE_MINUTE],
        default=REASON_VALID,
    )

    return pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "source_file": df["source_file"],
            "contract": df["contract"],
            "trading_date": df["trading_date"],
            "segment_id": df["segment_id"],
            "prev_timestamp": prev_timestamp,
            "prev_trading_date": prev_trading_date,
            "prev_segment_id": prev_segment_id,
            "prev_contract": prev_contract,
            "delta_minutes": delta_minutes,
            "is_first_observation": is_first,
            "is_roll_boundary": is_roll,
            "is_trading_date_boundary": is_trading_date_boundary,
            "is_non_consecutive_minute": is_non_consecutive,
            "r_1m_valid": valid,
            "invalid_reason": invalid_reason,
        }
    )


# ---------------------------------------------------------------------------
# Variables de analisis
# ---------------------------------------------------------------------------

def build_analysis_variables(canonical: pd.DataFrame, validity: pd.DataFrame) -> pd.DataFrame:
    """Construye las variables de TDA-04, aplicando la mascara de validez.

    Entrada: la serie canonica y la mascara de ``build_return_validity_mask``
    (mismo orden de filas, verificado con ``assert``).

    Por que Close -> Close (no Open, no un precio "tipico"): TDA-01 (§5)
    ya establecio que, en este formato de datos, los 5 campos de la barra
    ``t`` (incluido ``Close``) estan disponibles CONJUNTAMENTE en ``t`` --
    no hay una secuencia intra-barra que privilegie a ningun precio en
    particular por llegar "antes". Close -> Close es, ademas, la
    definicion estandar de Tsay (fundamento de esta etapa) y la que hace
    que los log-retornos se sumen exactamente entre barras consecutivas
    (ver seccion 11 de la tarea, comprobacion 8) -- una propiedad que NO
    se cumple, por ejemplo, para una secuencia de retornos Open->Close.

    Que es un log-retorno, en una frase: ``ln(C_t/C_{t-1})`` es,
    aproximadamente, el cambio porcentual de precio entre dos barras
    consecutivas -- pero, a diferencia del retorno simple
    ``C_t/C_{t-1}-1``, los log-retornos de una SECUENCIA VALIDA se suman
    en vez de multiplicarse (``r_t + r_{t+1} = ln(C_{t+1}/C_{t-1})``), lo
    que los hace mucho mas comodos para agregar en el tiempo (TH07/TH10,
    fundamento Tsay de esta etapa).

    Variables construidas (columnas nuevas sobre la serie canonica):

    - ``r_1m = ln(close_t/close_{t-1})`` -- retorno log de 1 minuto.
      ``NaN`` donde ``validity["r_1m_valid"]`` es ``False``.
    - ``R_1m = close_t/close_{t-1} - 1`` -- retorno simple, para TH07
      (¿cuanto difiere de ``r_1m``?). Misma regla de validez que ``r_1m``
      exactamente (mismas dos barras).
    - ``abs_r_1m = |r_1m|``, ``r2_1m = r_1m**2`` -- se derivan
      elemento a elemento de ``r_1m``: el ``NaN`` se propaga solo, sin
      necesitar ninguna regla de no-cruce propia.
    - ``zero_1m`` -- indicador (1.0/0.0/``NaN``) de ``r_1m == 0``. Vale
      ``NaN`` exactamente donde ``r_1m`` es ``NaN`` (no se puede decir si
      "no hubo movimiento" cuando no hay un retorno valido que evaluar).
    - ``log_hl = ln(high_t/low_t)`` -- rango intrabarra. Usa SOLO datos de
      la barra ``t``: no necesita ninguna barra anterior, así que
      **nunca** es ``NaN`` por una regla de no-cruce (TDA-00 ya garantiza
      ``high_t >= low_t > 0`` en el 100% de las filas -- 0 violaciones).
    - ``log_co = ln(close_t/open_t)`` -- movimiento intrabarra Open->Close.
      Igual que ``log_hl``: solo usa la barra ``t``, nunca cruza una
      frontera.
    - ``log_oc_prev = ln(open_t/close_{t-1})`` -- el "gap" heredado del
      roadmap (ahi llamado ``gap_t``): la parte del movimiento entre
      barras que ``log_co`` deliberadamente ignora. **Usa exactamente la
      misma barra anterior que ``r_1m``** -- por eso, tal como exige la
      seccion 4 de la tarea, sigue EXACTAMENTE la misma regla de validez.
    - ``volume`` -- se conserva tal cual de la serie canonica (no necesita
      ninguna regla de no-cruce: es un dato de una sola barra).

    Salida: ``DataFrame`` con las columnas de trazabilidad
    (``timestamp``, ``source_file``, ``contract``, ``trading_date``,
    ``segment_id``), el OHLCV crudo intacto, y las 8 columnas anteriores.
    """
    df = canonical.sort_values("timestamp").reset_index(drop=True)
    v = validity.sort_values("timestamp").reset_index(drop=True)
    assert (df["timestamp"].to_numpy() == v["timestamp"].to_numpy()).all(), (
        "canonical y validity no comparten exactamente los mismos timestamps, en el mismo orden"
    )

    prev_close = df["close"].shift(1)
    valid = v["r_1m_valid"].to_numpy()

    r_1m = np.log(df["close"] / prev_close)
    r_1m = r_1m.where(valid, other=np.nan)

    R_1m = (df["close"] / prev_close - 1.0)
    R_1m = R_1m.where(valid, other=np.nan)

    abs_r_1m = r_1m.abs()
    r2_1m = r_1m ** 2
    zero_1m = np.where(r_1m.isna(), np.nan, (r_1m.to_numpy() == 0.0).astype(float))

    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])

    log_oc_prev = np.log(df["open"] / prev_close)
    log_oc_prev = log_oc_prev.where(valid, other=np.nan)

    out = df[["timestamp", "source_file", "contract", "trading_date", "segment_id",
              "open", "high", "low", "close", "volume"]].copy()
    out["r_1m"] = r_1m
    out["R_1m"] = R_1m
    out["abs_r_1m"] = abs_r_1m
    out["r2_1m"] = r2_1m
    out["zero_1m"] = zero_1m
    out["log_hl"] = log_hl
    out["log_co"] = log_co
    out["log_oc_prev"] = log_oc_prev
    return out


# ---------------------------------------------------------------------------
# Auditoria cuantitativa
# ---------------------------------------------------------------------------

def audit_losses_by_cause(validity: pd.DataFrame) -> pd.DataFrame:
    """Cuenta cuantas filas caen en cada causa de invalidez, y su solapamiento.

    Entrada: la mascara de validez completa.
    Transformacion: conteo simple por ``invalid_reason`` (categoria
    prioritaria y EXCLUYENTE -- suma exactamente el total de filas) mas,
    por separado, el conteo de cada bandera booleana de forma
    INDEPENDIENTE (para poder ver cuanto se solapan -- p.ej., cuantas
    filas tienen a la vez ``is_roll_boundary`` e
    ``is_trading_date_boundary``, que por construccion de TDA-03 debería
    ser exactamente igual al total de ``is_roll_boundary`` -- todo roll
    ocurre en una frontera de jornada, nunca a mitad de una).
    Salida: ``DataFrame`` de dos bloques (categoria exclusiva / bandera
    independiente), con conteo y porcentaje sobre el total de filas.
    """
    total = len(validity)
    exclusive_counts = validity["invalid_reason"].value_counts()
    rows = []
    for reason in ALL_REASONS + [REASON_VALID]:
        n = int(exclusive_counts.get(reason, 0))
        rows.append({"block": "exclusive_reason", "label": reason, "n_rows": n, "pct_of_total": 100.0 * n / total})

    flag_columns = ["is_first_observation", "is_roll_boundary", "is_trading_date_boundary", "is_non_consecutive_minute"]
    for col in flag_columns:
        n = int(validity[col].sum())
        rows.append({"block": "independent_flag", "label": col, "n_rows": n, "pct_of_total": 100.0 * n / total})

    overlap_roll_and_date = int((validity["is_roll_boundary"] & validity["is_trading_date_boundary"]).sum())
    rows.append({
        "block": "overlap", "label": "is_roll_boundary_AND_is_trading_date_boundary",
        "n_rows": overlap_roll_and_date, "pct_of_total": 100.0 * overlap_roll_and_date / total,
    })

    return pd.DataFrame(rows)


def audit_th07_r_vs_R(variables: pd.DataFrame, n_deciles: int = 10) -> pd.DataFrame:
    """TH07 -- distribucion de |r_1m - R_1m|, global y por decil de magnitud de r_1m.

    Entrada: la tabla de variables ya construida.
    Transformacion (metodo minimo de TDA-04 para TH07, roadmap §"Metodos
    minimos" punto 3): se calcula ``|r_1m - R_1m|`` sobre las filas donde
    ambos son validos (por construccion, exactamente las mismas filas), y
    se agrega globalmente y por decil de ``|r_1m|`` (para ver si la
    diferencia se concentra en los movimientos mas grandes, tal como
    plantea la pregunta de TH07).
    Salida: ``DataFrame`` con una fila "GLOBAL" y una fila por decil
    (0=movimientos mas pequeños, 9=mas grandes), con media/mediana/max de
    la diferencia absoluta y el numero de observaciones.
    """
    valid = variables["r_1m"].notna() & variables["R_1m"].notna()
    r = variables.loc[valid, "r_1m"]
    R = variables.loc[valid, "R_1m"]
    abs_diff = (r - R).abs()

    rows = [{
        "decile": "GLOBAL", "n": int(valid.sum()),
        "mean_abs_diff": float(abs_diff.mean()), "median_abs_diff": float(abs_diff.median()),
        "max_abs_diff": float(abs_diff.max()),
    }]
    deciles = pd.qcut(r.abs(), n_deciles, labels=False, duplicates="drop")
    for d in sorted(deciles.dropna().unique()):
        sub = abs_diff[deciles == d]
        rows.append({
            "decile": int(d), "n": int(len(sub)),
            "mean_abs_diff": float(sub.mean()), "median_abs_diff": float(sub.median()),
            "max_abs_diff": float(sub.max()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------

@dataclass
class TDA04Result:
    canonical: pd.DataFrame
    roll_mask: pd.DataFrame
    validity: pd.DataFrame
    variables: pd.DataFrame
    losses_by_cause: pd.DataFrame
    th07_r_vs_R: pd.DataFrame


def run_tda04_analysis(config: SnapshotConfig) -> TDA04Result:
    """Orquesta TDA-04 completo sobre la serie canonica de TDA-03.

    Reutiliza, sin duplicar logica: la proteccion del hold-out
    (``holdout_guard.py`` -- defensiva: TDA-04 no abre ningun archivo
    crudo, pero se re-valida por consistencia con el resto del pipeline y
    para que un cambio futuro en la serie canonica no pueda colar
    silenciosamente una fila del hold-out) y los dos artefactos de TDA-03
    (serie canonica, mascara de roll) sin recalcular nada de lo que esas
    etapas ya resolvieron.
    """
    validate_research_holdout_disjoint(config)

    canonical, roll_mask = load_canonical_and_mask(config)

    last_timestamps = canonical.groupby("source_file")["timestamp"].max().to_dict()
    validate_last_timestamps_before_boundary(config, last_timestamps)

    validity = build_return_validity_mask(canonical, roll_mask)
    variables = build_analysis_variables(canonical, validity)
    losses_by_cause = audit_losses_by_cause(validity)
    th07 = audit_th07_r_vs_R(variables)

    return TDA04Result(
        canonical=canonical,
        roll_mask=roll_mask,
        validity=validity,
        variables=variables,
        losses_by_cause=losses_by_cause,
        th07_r_vs_R=th07,
    )
