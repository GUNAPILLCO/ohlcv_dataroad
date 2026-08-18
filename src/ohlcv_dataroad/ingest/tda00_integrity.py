"""TDA-00 -- Integridad intra-barra e invariantes fisicos.

Implementa la etapa TDA-00 del roadmap
(``docs/methodology/Tsay_OHLCV_analysis_roadmap.md``, seccion "TDA-00 --
Integridad intra-barra e invariantes fisicos") y las hipotesis TH01/TH06
del backlog (``docs/methodology/Tsay_empirical_hypotheses_backlog.md``).

Que hace TDA-00, en una frase: comprobar que cada fila cruda representa una
barra OHLCV admisible -- sin usar todavia ninguna informacion de calendario,
sesion o contrato (eso es TDA-01/TDA-02, fuera de alcance aqui) -- y dejar
esa comprobacion trazada, reproducible y sin borrar ni corregir un solo
dato.

Catalogo de reglas
-------------------
Reglas que se originan durante el PARSEO (ver ``parsing.py``), porque la
linea ni siquiera llega a convertirse en una fila interpretable:

    schema_field_count     -- la linea no tiene exactamente 6 campos
    parse_error_timestamp  -- el campo de timestamp no matchea el formato
    parse_error_numeric    -- un campo numerico no es convertible
    (null_value y non_finite_value tambien pueden originarse aqui)

Reglas que se comprueban sobre filas YA bien formadas (``HARD_INVARIANT_RULES``):

    null_value                       -- valor nulo (chequeo defensivo; no
                                         deberia ocurrir tras el parseo)
    non_finite_value                 -- valor no finito (chequeo defensivo)
    nonpositive_price                -- open/high/low/close <= 0
    negative_volume                  -- volume < 0
    ohlc_incoherent                  -- H<L, o H<max(O,C), o L>min(O,C)
    tick_grid_violation               -- precio no alineado a tick_size
    duplicate_timestamp_within_file  -- el mismo timestamp aparece 2+ veces
    timestamp_out_of_order           -- el timestamp retrocede respecto a
                                         la fila anterior del archivo
    duplicate_exact_row              -- timestamp+OHLCV identicos a otra fila

Todas las reglas anteriores son INVARIANTES DUROS: si se cumplen, la fila
se marca ``bad_data=True`` y cuenta para la fraccion de violaciones de
TDA-00 (criterio STOP-0 del roadmap).

Aparte, se registra una unica bandera OBSERVACIONAL, que NO es una
violacion y NO cuenta para ``bad_data``:

    zero_volume -- volume == 0. El roadmap (TDA-02, hipotesis TH04) deja
    abierta la pregunta de si una barra sin operaciones se representa como
    fila ausente, volumen 0 o forward-fill; TDA-00 solo puede observar y
    contar, no resolver esa pregunta -- por eso se registra sin tratarla
    como defecto.

Ninguna fila se elimina ni se corrige en ningun punto de este modulo.

Proteccion del hold-out
------------------------
Antes de procesar cualquier dato, ``run_tda00`` valida que
``research_files`` y ``holdout_files`` (declarados en
``configs/mnq_snapshot.yaml``) sean conjuntos disjuntos, y despues de
parsear el conjunto de investigacion valida que ninguna de sus filas
alcance la frontera del hold-out (``holdout_boundary_utc``). Ambas
validaciones lanzan ``HoldoutIsolationError`` si fallan, y ninguna de las
dos requiere abrir un archivo de ``holdout_files``. Esta proteccion vive
en ``holdout_guard.py`` (``validate_research_holdout_disjoint`` y
``validate_last_timestamps_before_boundary``), compartida con TDA-01 y
cualquier etapa futura sobre ``research_files`` -- no se duplica aqui.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ohlcv_dataroad.config import SnapshotConfig
from ohlcv_dataroad.ingest.parsing import parse_raw_file

# Reglas que solo pueden originarse durante el parseo de la linea cruda
# (la linea nunca llega a ser una fila con columnas numericas).
PARSE_TIME_RULES = ["schema_field_count", "parse_error_timestamp", "parse_error_numeric"]

# Reglas que se comprueban sobre el DataFrame de filas ya bien formadas.
# "null_value" y "non_finite_value" tambien pueden aparecer como
# PARSE_TIME_RULES (campo vacio o infinito detectado durante el parseo);
# aqui se repiten como chequeo defensivo posterior, que en condiciones
# normales siempre deberia dar 0 filas.
HARD_INVARIANT_RULES = [
    "null_value",
    "non_finite_value",
    "nonpositive_price",
    "negative_volume",
    "ohlc_incoherent",
    "tick_grid_violation",
    "duplicate_timestamp_within_file",
    "timestamp_out_of_order",
    "duplicate_exact_row",
]

# Catalogo completo de reglas duras (union de las dos listas anteriores,
# sin duplicar "null_value"/"non_finite_value"). Se usa para las columnas
# del resumen por archivo.
ALL_HARD_RULES = PARSE_TIME_RULES + HARD_INVARIANT_RULES

TICK_GRID_TOLERANCE = 1e-6

# HoldoutIsolationError vivia originalmente en este modulo. Se movio a
# holdout_guard.py para que TDA-01 (y cualquier etapa futura sobre
# research_files) pueda reutilizar exactamente la misma proteccion en vez
# de duplicarla. Se re-exporta aqui para no romper el codigo ni los tests
# existentes que hacen `from ohlcv_dataroad.ingest.tda00_integrity import
# HoldoutIsolationError`.
from ohlcv_dataroad.ingest.holdout_guard import (  # noqa: E402
    HoldoutIsolationError,
    validate_last_timestamps_before_boundary,
    validate_research_holdout_disjoint,
)


def _hard_invariant_checks(df: pd.DataFrame, tick_size: float) -> pd.DataFrame:
    """Calcula, de forma vectorizada, cada regla de ``HARD_INVARIANT_RULES``.

    Entrada
    -------
    df        : DataFrame de filas bien formadas de UN archivo (columnas
        ``timestamp, open, high, low, close, volume``, en el orden del
        archivo -- esto importa para "timestamp_out_of_order").
    tick_size : tamano de tick del contrato (0.25 para MNQ), desde config.

    Por que vectorizado y no fila a fila: un archivo tiene entre 50.000 y
    100.000 filas; operar con mascaras booleanas de pandas/numpy sobre la
    columna completa es ordenes de magnitud mas rapido que iterar en Python
    puro, y el resultado es identico. Cada linea de abajo produce una
    ``Series`` booleana del mismo largo que ``df``: ``True`` en las filas
    que violan esa regla en particular.

    Salida
    ------
    DataFrame con una columna booleana por regla (mismo indice que ``df``),
    listo para que ``bad_data = df_checks.any(axis=1)``.
    """
    price_cols = ["open", "high", "low", "close"]

    # --- Defensivo: no deberia dispararse nunca, porque parse_raw_file ya
    # filtra nulos e infinitos antes de que la fila llegue aqui. Se
    # mantiene por si en el futuro esta funcion se usa sobre un DataFrame
    # construido por otra via (p.ej. releido desde un artefacto persistido).
    null_value = df[price_cols + ["volume"]].isnull().any(axis=1) | df["timestamp"].isnull()
    finite_prices = np.isfinite(df[price_cols].to_numpy(dtype=float))
    non_finite_value = pd.Series(~finite_prices.all(axis=1), index=df.index)

    # --- Precios y volumen ---
    nonpositive_price = (df[price_cols] <= 0).any(axis=1)
    negative_volume = df["volume"] < 0

    # --- Coherencia OHLC: H y L son el maximo y el minimo de la barra, por
    # lo que deben contener a O y C. Las tres condiciones vienen
    # directamente de la definicion de barra OHLC (roadmap TDA-00,
    # fundamento C5: H/L son estadisticos de orden).
    oc_max = df[["open", "close"]].max(axis=1)
    oc_min = df[["open", "close"]].min(axis=1)
    ohlc_incoherent = (df["high"] < df["low"]) | (df["high"] < oc_max) | (df["low"] > oc_min)

    # --- Grilla de tick: (precio / tick_size) debe ser (aprox.) un entero.
    # Se compara contra una tolerancia pequena (1e-6) en vez de una
    # igualdad exacta porque la division en punto flotante puede introducir
    # errores de redondeo del orden de 1e-12..1e-9 incluso para precios que
    # SI estan alineados a la grilla.
    ratio = df[price_cols] / tick_size
    residual = (ratio - ratio.round()).abs()
    tick_grid_violation = (residual > TICK_GRID_TOLERANCE).any(axis=1)

    # --- Timestamps: duplicado = el mismo instante aparece 2+ veces en
    # cualquier posicion del archivo (keep=False marca TODAS las
    # apariciones, no solo la segunda en adelante, para poder trazar el
    # grupo completo). Fuera de orden = el timestamp de esta fila es
    # ANTERIOR al de la fila inmediatamente previa en el archivo (un
    # timestamp IGUAL no cuenta aqui como "fuera de orden": ya queda
    # cubierto por duplicate_timestamp_within_file, y mezclar ambos
    # criterios oscurecería cual de las dos cosas paso realmente).
    duplicate_timestamp_within_file = df["timestamp"].duplicated(keep=False)
    prev_ts = df["timestamp"].shift(1)
    timestamp_out_of_order = df["timestamp"] < prev_ts
    if len(timestamp_out_of_order) > 0:
        timestamp_out_of_order.iloc[0] = False  # la primera fila no tiene fila previa

    # --- Duplicado exacto: timestamp Y los 5 valores OHLCV coinciden con
    # otra fila del archivo. Es la definicion que la memoria heredada marca
    # como correcta (incluir el timestamp/indice en la comparacion; ver
    # docs/instruments/mnq/MNQ_DATA_PRIOR_KNOWLEDGE.md, seccion 4.3) tras un
    # bug historico en el que se comparaba solo OHLCV sin el timestamp.
    duplicate_exact_row = df.duplicated(subset=["timestamp"] + price_cols + ["volume"], keep=False)

    return pd.DataFrame(
        {
            "null_value": null_value,
            "non_finite_value": non_finite_value,
            "nonpositive_price": nonpositive_price,
            "negative_volume": negative_volume,
            "ohlc_incoherent": ohlc_incoherent,
            "tick_grid_violation": tick_grid_violation,
            "duplicate_timestamp_within_file": duplicate_timestamp_within_file,
            "timestamp_out_of_order": timestamp_out_of_order,
            "duplicate_exact_row": duplicate_exact_row,
        },
        index=df.index,
    )


def _build_detail(rule: str, idx, df: pd.DataFrame, tick_size: float) -> str:
    """Construye el texto explicativo de UNA violacion concreta.

    Solo se llama sobre filas ya identificadas como violatorias de ``rule``
    (subconjunto tipicamente pequeno o vacio), asi que no necesita estar
    vectorizado: prioriza la claridad del mensaje sobre el rendimiento.

    Entrada: el codigo de regla, el indice de la fila en ``df`` y el propio
    ``df`` (para poder mirar filas vecinas, p.ej. la anterior, o recontar
    cuantas filas comparten timestamp/valores).

    Salida: una frase en espanol que un revisor humano pueda leer
    directamente en ``TDA00_violaciones.csv`` sin tener que releer el
    codigo para entender que paso.
    """
    row = df.loc[idx]

    if rule == "nonpositive_price":
        bad_fields = [c for c in ["open", "high", "low", "close"] if row[c] <= 0]
        detalle = ", ".join(f"{c}={row[c]}" for c in bad_fields)
        return f"precio(s) no positivo(s): {detalle}"

    if rule == "negative_volume":
        return f"volume={row['volume']} (< 0)"

    if rule == "ohlc_incoherent":
        reasons = []
        oc_max = max(row["open"], row["close"])
        oc_min = min(row["open"], row["close"])
        if row["high"] < row["low"]:
            reasons.append(f"high({row['high']}) < low({row['low']})")
        if row["high"] < oc_max:
            reasons.append(f"high({row['high']}) < max(open,close)({oc_max})")
        if row["low"] > oc_min:
            reasons.append(f"low({row['low']}) > min(open,close)({oc_min})")
        return "; ".join(reasons)

    if rule == "tick_grid_violation":
        bad = []
        for c in ["open", "high", "low", "close"]:
            ratio = row[c] / tick_size
            residual = abs(ratio - round(ratio))
            if residual > TICK_GRID_TOLERANCE:
                bad.append(f"{c}={row[c]} (residuo={residual:.6f} ticks de {tick_size})")
        return "; ".join(bad)

    if rule == "duplicate_timestamp_within_file":
        n = int((df["timestamp"] == row["timestamp"]).sum())
        return f"timestamp {row['timestamp']} aparece {n} veces en el archivo"

    if rule == "timestamp_out_of_order":
        pos = df.index.get_loc(idx)
        prev_ts = df.iloc[pos - 1]["timestamp"] if pos > 0 else None
        return f"timestamp {row['timestamp']} es anterior al de la fila previa ({prev_ts})"

    if rule == "duplicate_exact_row":
        same = (
            (df["timestamp"] == row["timestamp"])
            & (df["open"] == row["open"])
            & (df["high"] == row["high"])
            & (df["low"] == row["low"])
            & (df["close"] == row["close"])
            & (df["volume"] == row["volume"])
        )
        n = int(same.sum())
        return f"fila (timestamp+OHLCV) identica aparece {n} veces en el archivo"

    if rule == "null_value":
        return "valor nulo detectado (verificacion defensiva posterior al parseo)"

    if rule == "non_finite_value":
        return "valor no finito detectado (verificacion defensiva posterior al parseo)"

    return "sin detalle adicional"


@dataclass
class FileResult:
    """Resultado de ejecutar TDA-00 sobre UN archivo del conjunto de investigacion."""

    source_file: str
    mask: pd.DataFrame           # una fila por linea del archivo (parseada o no)
    violations: list[dict]       # una entrada por (fila, regla) violada -- formato largo
    summary: dict = field(default_factory=dict)


def process_file(
    path: Path, source_file: str, timestamp_format: str, separator: str, tick_size: float
) -> FileResult:
    """Ejecuta TDA-00 sobre un unico archivo crudo.

    Flujo
    -----
    1. ``parse_raw_file`` separa las lineas bien formadas (``df``) de los
       errores de parseo (``parsed.parse_errors``).
    2. Sobre ``df``, ``_hard_invariant_checks`` calcula, por fila, si viola
       cada una de las reglas de ``HARD_INVARIANT_RULES``.
    3. Se construyen dos artefactos:
       - ``violations``: una entrada por cada (fila, regla) violada --
         incluye tanto los errores de parseo como las violaciones de
         invariantes sobre filas bien formadas. Formato "largo": una fila
         que viola 2 reglas genera 2 entradas.
       - ``mask``: una fila por CADA linea del archivo (bien formada o no),
         con un booleano ``bad_data`` y la lista de reglas violadas. Este
         es el artefacto que despues se persiste completo (sin filtrar) en
         ``data/interim/mnq/tda00_bad_data_mask.parquet``.
    4. Se arma ``summary``: conteos agregados para el resumen por archivo y
       para el informe de inventario.

    Nota de rendimiento: la construccion de ``violations`` y de la columna
    ``violation_rules`` de la mascara solo itera sobre las filas
    EFECTIVAMENTE marcadas como violatorias (``bad_idx``), no sobre el
    archivo completo. Con 0 violaciones (el caso esperado en este
    snapshot), ese trabajo es esencialmente gratis; si en una futura
    version del snapshot aparecieran violaciones, el costo escala con el
    numero de violaciones, no con el numero de filas del archivo.
    """
    parsed = parse_raw_file(path, timestamp_format, separator)
    df = parsed.rows
    n_rows = len(df)

    violations: list[dict] = []
    for err in parsed.parse_errors:
        violations.append(
            {
                "source_file": source_file,
                "row_index": err.row_index,
                "rule": err.rule,
                "detail": err.detail,
                "raw_line": err.raw_line,
                "timestamp_raw": None,
                "open": None,
                "high": None,
                "low": None,
                "close": None,
                "volume": None,
            }
        )

    if n_rows > 0:
        checks = _hard_invariant_checks(df, tick_size)
        bad_data_parsed = checks[HARD_INVARIANT_RULES].any(axis=1)
        zero_volume = df["volume"] == 0

        bad_idx = df.index[bad_data_parsed]
        for idx in bad_idx:
            row = df.loc[idx]
            for rule in HARD_INVARIANT_RULES:
                if bool(checks.loc[idx, rule]):
                    # raw_line=None aqui a proposito: estas violaciones ocurren
                    # sobre filas que YA se parsearon correctamente (df viene de
                    # parsed.rows), asi que su contenido original es exactamente
                    # reconstruible a partir de timestamp_raw + open/high/low/
                    # close/volume, sin ambiguedad -- no hace falta guardar el
                    # texto crudo aparte. raw_line solo aporta valor cuando la
                    # linea NO llego a convertirse en una fila interpretable
                    # (errores de parseo, arriba), que es el caso en el que el
                    # Correccion 3 de esta tarea pide preservarlo.
                    violations.append(
                        {
                            "source_file": source_file,
                            "row_index": int(row["row_index"]),
                            "rule": rule,
                            "detail": _build_detail(rule, idx, df, tick_size),
                            "raw_line": None,
                            "timestamp_raw": row["timestamp_raw"],
                            "open": row["open"],
                            "high": row["high"],
                            "low": row["low"],
                            "close": row["close"],
                            "volume": row["volume"],
                        }
                    )

        violation_rules_series = pd.Series("", index=df.index, dtype=object)
        if len(bad_idx) > 0:
            sub = checks.loc[bad_idx, HARD_INVARIANT_RULES]
            violation_rules_series.loc[bad_idx] = sub.apply(
                lambda r: ";".join(rule for rule in HARD_INVARIANT_RULES if r[rule]), axis=1
            )

        mask_parsed = pd.DataFrame(
            {
                "source_file": source_file,
                "row_index": df["row_index"].astype(int),
                "timestamp": df["timestamp"],
                "timestamp_raw": df["timestamp_raw"],
                "open": df["open"],
                "high": df["high"],
                "low": df["low"],
                "close": df["close"],
                "volume": df["volume"],
                "bad_data": bad_data_parsed.to_numpy(),
                "violation_rules": violation_rules_series.to_numpy(),
                "zero_volume": zero_volume.to_numpy(),
            }
        )
    else:
        checks = pd.DataFrame(columns=HARD_INVARIANT_RULES)
        mask_parsed = pd.DataFrame(
            columns=[
                "source_file", "row_index", "timestamp", "timestamp_raw",
                "open", "high", "low", "close", "volume",
                "bad_data", "violation_rules", "zero_volume",
            ]
        )

    if parsed.parse_errors:
        mask_errors = pd.DataFrame(
            [
                {
                    "source_file": source_file,
                    "row_index": e.row_index,
                    "timestamp": pd.NaT,
                    "timestamp_raw": None,
                    "open": None, "high": None, "low": None, "close": None, "volume": None,
                    "bad_data": True,
                    "violation_rules": e.rule,
                    "zero_volume": False,
                }
                for e in parsed.parse_errors
            ]
        )
    else:
        mask_errors = pd.DataFrame(columns=mask_parsed.columns)

    mask = pd.concat([mask_parsed, mask_errors], ignore_index=True)
    if not mask.empty:
        mask = mask.sort_values("row_index").reset_index(drop=True)

    rule_counts = {rule: sum(1 for v in violations if v["rule"] == rule) for rule in ALL_HARD_RULES}
    hard_violation_rows = int(mask["bad_data"].sum()) if not mask.empty else 0
    zero_volume_rows = int(mask["zero_volume"].sum()) if not mask.empty else 0

    summary = {
        "source_file": source_file,
        "total_lines": parsed.total_lines,
        "parsed_rows": n_rows,
        "parse_error_rows": len(parsed.parse_errors),
        "hard_violation_rows": hard_violation_rows,
        "violation_records": len(violations),
        "zero_volume_rows": zero_volume_rows,
        "first_timestamp": df["timestamp"].min() if n_rows > 0 else None,
        "last_timestamp": df["timestamp"].max() if n_rows > 0 else None,
        "min_volume": df["volume"].min() if n_rows > 0 else None,
        "max_volume": df["volume"].max() if n_rows > 0 else None,
        **{f"rule_count__{rule}": rule_counts[rule] for rule in ALL_HARD_RULES},
    }

    return FileResult(source_file=source_file, mask=mask, violations=violations, summary=summary)


def _aggregate_global_stats(file_results: list[FileResult], config: SnapshotConfig) -> dict:
    """Reduce la lista de resultados por archivo a un unico resumen global.

    Entrada: la lista de ``FileResult`` de todos los archivos procesados.
    Transformacion: sumas y min/max sencillos sobre los campos de
    ``summary`` de cada archivo (no hay nada estadistico aqui, solo
    aritmetica de conteo).
    Salida: un dict con los totales que alimentan tanto el informe de
    inventario (``TDA00_inventario.md``) como la salida por consola del
    script de linea de comandos.
    """
    total_lines = sum(r.summary["total_lines"] for r in file_results)
    parsed_rows = sum(r.summary["parsed_rows"] for r in file_results)
    parse_error_rows = sum(r.summary["parse_error_rows"] for r in file_results)
    hard_violation_rows = sum(r.summary["hard_violation_rows"] for r in file_results)
    violation_records = sum(r.summary["violation_records"] for r in file_results)
    zero_volume_rows = sum(r.summary["zero_volume_rows"] for r in file_results)

    first_timestamps = [r.summary["first_timestamp"] for r in file_results if r.summary["first_timestamp"] is not None]
    last_timestamps = [r.summary["last_timestamp"] for r in file_results if r.summary["last_timestamp"] is not None]
    min_volumes = [r.summary["min_volume"] for r in file_results if r.summary["min_volume"] is not None]
    max_volumes = [r.summary["max_volume"] for r in file_results if r.summary["max_volume"] is not None]

    rule_counts_total = {
        rule: sum(r.summary[f"rule_count__{rule}"] for r in file_results) for rule in ALL_HARD_RULES
    }

    if hard_violation_rows == 0 and zero_volume_rows == 0:
        status = "PASS"
    else:
        status = "PASS_WITH_WARNINGS"

    return {
        "status": status,
        "n_files": len(file_results),
        "total_lines": total_lines,
        "parsed_rows": parsed_rows,
        "parse_error_rows": parse_error_rows,
        "hard_violation_rows": hard_violation_rows,
        "violation_records": violation_records,
        "zero_volume_rows": zero_volume_rows,
        "fraction_hard_violation_rows": (hard_violation_rows / total_lines) if total_lines > 0 else 0.0,
        "first_timestamp": min(first_timestamps) if first_timestamps else None,
        "last_timestamp": max(last_timestamps) if last_timestamps else None,
        "min_volume": min(min_volumes) if min_volumes else None,
        "max_volume": max(max_volumes) if max_volumes else None,
        "rule_counts_total": rule_counts_total,
        "tick_size": config.tick_size,
        "tick_size_source": config.tick_size_source,
    }


def _write_inventory_report(config: SnapshotConfig, stats: dict, summary_df: pd.DataFrame) -> None:
    """Genera ``reports/mnq/TDA00_inventario.md`` a partir de las estadisticas agregadas.

    Puramente de presentacion: no calcula nada, solo formatea los numeros
    ya calculados por ``_aggregate_global_stats`` y ``process_file`` en la
    salida markdown que exige el roadmap para TDA-00 (columnas, tipos,
    rango de fechas, conteo de filas, tick size declarado y su fuente).
    """
    lines: list[str] = []
    lines.append("# TDA-00 -- Inventario e integridad intra-barra")
    lines.append("")
    lines.append("Salida obligatoria de la etapa TDA-00 "
                  "(`docs/methodology/Tsay_OHLCV_analysis_roadmap.md`). "
                  "Generado automaticamente por "
                  "`src/ohlcv_dataroad/ingest/run_tda00.py`; no editar a mano.")
    lines.append("")
    lines.append(f"**Estado final: `{stats['status']}`**")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Alcance")
    lines.append("")
    lines.append(f"- Archivos procesados: {stats['n_files']} (conjunto de investigacion; "
                  "el hold-out declarado en `configs/mnq_snapshot.yaml` no fue abierto por este pipeline).")
    lines.append(f"- Frontera de hold-out (no tocada): `{config.holdout_boundary_utc}` "
                  f"(fuente: {config.holdout_boundary_source}).")
    lines.append("")
    lines.append("## 2. Esquema y tipos")
    lines.append("")
    lines.append("| Columna | Tipo | Origen |")
    lines.append("|---|---|---|")
    lines.append("| `timestamp` | datetime (parseado con `strptime`) | campo 1, formato "
                  f"`{config.timestamp_format}` |")
    lines.append("| `open`, `high`, `low`, `close` | float | campos 2-5 |")
    lines.append("| `volume` | int | campo 6 |")
    lines.append("")
    lines.append(f"- Separador de campos: `{config.separator}`")
    lines.append(f"- Zona horaria del timestamp crudo: `{config.timestamp_raw_timezone}` "
                  "(confirmada por la fuente de los datos; ver "
                  "`reports/mnq/00_initial_repository_audit.md`, seccion 4). "
                  "TDA-00 no realiza ninguna conversion de zona horaria: solo la registra.")
    lines.append(f"- Tick size declarado: `{stats['tick_size']}` -- fuente: {stats['tick_size_source']}.")
    lines.append("")
    lines.append("## 3. Conteo de filas")
    lines.append("")
    lines.append(f"- Lineas totales leidas: **{stats['total_lines']:,}**")
    lines.append(f"- Filas bien formadas: **{stats['parsed_rows']:,}**")
    lines.append(f"- Errores de parseo (lineas descartadas del parseo, trazadas): **{stats['parse_error_rows']:,}**")
    conservation_ok = stats['parsed_rows'] + stats['parse_error_rows'] == stats['total_lines']
    lines.append(f"- Verificacion de conservacion (`total_lines == parsed_rows + parse_error_rows`): "
                  f"{'CUMPLIDA' if conservation_ok else 'FALLIDA'}")
    lines.append(f"- Rango temporal cubierto: `{stats['first_timestamp']}` -> `{stats['last_timestamp']}`")
    lines.append(f"- Volumen minimo / maximo observado (filas bien formadas): "
                  f"`{stats['min_volume']}` / `{stats['max_volume']}`")
    lines.append("")
    lines.append("## 4. Resultado de las invariantes duras")
    lines.append("")
    lines.append(f"- Filas con al menos una violacion de invariante duro: **{stats['hard_violation_rows']:,}** "
                  f"de {stats['total_lines']:,} "
                  f"({stats['fraction_hard_violation_rows']*100:.6f} %)")
    lines.append(f"- Registros de violacion (una fila puede violar mas de una regla; formato largo de "
                  f"`TDA00_violaciones.csv`): **{stats['violation_records']:,}**")
    lines.append("")
    lines.append("| Regla | Ocurrencias |")
    lines.append("|---|---:|")
    for rule in ALL_HARD_RULES:
        lines.append(f"| `{rule}` | {stats['rule_counts_total'][rule]:,} |")
    lines.append("")
    lines.append("## 5. Bandera observacional (no es violacion)")
    lines.append("")
    lines.append(f"- Barras con `volume == 0`: **{stats['zero_volume_rows']:,}**. "
                  "No se cuenta como violacion de invariante (el roadmap solo exige `volume >= 0`). "
                  "Es un dato relevante para `TH04`/`TDA-02` (semantica de barras sin actividad), "
                  "que sigue fuera de alcance de esta etapa.")
    lines.append("")
    lines.append("## 6. Resumen por archivo")
    lines.append("")
    cols = [
        "source_file", "total_lines", "parsed_rows", "parse_error_rows",
        "hard_violation_rows", "zero_volume_rows", "first_timestamp", "last_timestamp",
        "min_volume", "max_volume",
    ]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for _, row in summary_df[cols].iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    lines.append("")
    lines.append("Tabla completa, con el conteo por regla por archivo, en `TDA00_resumen_por_archivo.csv`.")
    lines.append("")
    lines.append("## 7. Interpretacion")
    lines.append("")
    if stats["hard_violation_rows"] == 0:
        lines.append("- No se encontro ninguna violacion de invariante duro. No aplica `STOP-0`: "
                      "el roadmap detiene la etapa solo si la fraccion de violaciones no es despreciable "
                      "o si se concentran en un tramo extenso -- ninguna de las dos condiciones se da "
                      "porque el conteo es cero.")
    else:
        lines.append("- **Se encontraron violaciones de invariante duro.** El roadmap "
                      "(seccion `STOP-0`) exige evaluar si la fraccion afectada es despreciable y si "
                      "las violaciones se concentran en un tramo temporal concreto antes de decidir si "
                      "se continua. Esta decision es metodologica y no la toma este pipeline de forma "
                      "automatica: revisar `TDA00_violaciones.csv` antes de avanzar a TDA-01.")
    if stats["zero_volume_rows"] > 0:
        lines.append(f"- Se encontraron {stats['zero_volume_rows']:,} barras con volumen 0. "
                      "Queda como pregunta abierta para TDA-02/TH04, no se interpreta aqui.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("**No se elimino ni se corrigio ningun dato en esta etapa.** Toda fila conflictiva "
                  "queda trazada en `TDA00_violaciones.csv` y en la mascara persistida "
                  f"`{config.bad_data_mask_name}` (`bad_data=True`), no en el archivo crudo.")
    lines.append("")

    config.inventory_report_path.write_text("\n".join(lines), encoding="utf-8")


def run_tda00(config: SnapshotConfig) -> dict:
    """Orquesta la etapa TDA-00 completa sobre el conjunto de investigacion.

    Flujo
    -----
    0. Valida, sin abrir ningun archivo, que ``research_files`` y
       ``holdout_files`` sean disjuntos (``validate_research_holdout_disjoint``,
       en ``holdout_guard.py`` -- compartida con TDA-01). Si la
       configuracion esta mal escrita, se aborta aqui, antes de tocar
       ningun dato.
    1. Crea los directorios de salida si no existen.
    2. Procesa, EN ORDEN, cada archivo declarado en
       ``config.research_files`` (nunca los de ``config.holdout_files`` --
       el hold-out no se abre en ningun punto de esta funcion).
    3. Verifica la conservacion de cada archivo
       (``total_lines == parsed_rows + parse_error_rows``); si falla para
       cualquier archivo, se aborta con ``AssertionError`` en vez de
       escribir artefactos parcialmente inconsistentes.
    3b. Valida que ninguna fila del conjunto de investigacion alcance o
        supere la frontera del hold-out
        (``validate_last_timestamps_before_boundary``, tambien compartida
        con TDA-01), mirando exclusivamente los archivos de investigacion
        ya parseados.
    4. Concatena los resultados de todos los archivos en tres artefactos:
       - mascara `bad_data` completa (parquet),
       - violaciones en formato largo (csv),
       - resumen por archivo (csv).
    5. Escribe el informe de inventario en markdown.

    Entrada: ``SnapshotConfig`` ya cargado (ver ``config.py``).
    Salida: dict de estadisticas globales (el mismo que recibe el informe
    de inventario), para que el script de linea de comandos pueda
    imprimir un resumen sin tener que releer los archivos generados.

    Efectos secundarios (los unicos de todo el modulo): escribe los
    archivos en ``config.bad_data_mask_path``, ``config.violations_report_path``,
    ``config.per_file_summary_path`` y ``config.inventory_report_path``.
    Nunca escribe ni modifica nada dentro de ``data/raw/``, y nunca abre un
    archivo de ``config.holdout_files``.
    """
    validate_research_holdout_disjoint(config)

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    config.interim_dir.mkdir(parents=True, exist_ok=True)

    file_results: list[FileResult] = []
    for source_file in config.research_files:
        path = config.raw_dir / source_file
        if not path.exists():
            raise FileNotFoundError(
                f"Archivo del conjunto de investigacion declarado en la configuracion no existe: {path}"
            )
        result = process_file(path, source_file, config.timestamp_format, config.separator, config.tick_size)
        file_results.append(result)

    for r in file_results:
        s = r.summary
        expected = s["parsed_rows"] + s["parse_error_rows"]
        if expected != s["total_lines"]:
            raise AssertionError(
                f"Fallo de conservacion en {r.source_file}: total_lines={s['total_lines']} "
                f"!= parsed_rows({s['parsed_rows']}) + parse_error_rows({s['parse_error_rows']})"
            )

    validate_last_timestamps_before_boundary(
        config, {r.source_file: r.summary["last_timestamp"] for r in file_results}
    )

    mask_all = pd.concat([r.mask for r in file_results], ignore_index=True)
    violations_columns = [
        "source_file", "row_index", "rule", "detail", "raw_line",
        "timestamp_raw", "open", "high", "low", "close", "volume",
    ]
    violations_all = pd.DataFrame(
        [v for r in file_results for v in r.violations], columns=violations_columns
    )
    summary_all = pd.DataFrame([r.summary for r in file_results])

    mask_all.to_parquet(config.bad_data_mask_path, index=False)
    violations_all.to_csv(config.violations_report_path, index=False)
    summary_all.to_csv(config.per_file_summary_path, index=False)

    stats = _aggregate_global_stats(file_results, config)
    _write_inventory_report(config, stats, summary_all)

    return stats
