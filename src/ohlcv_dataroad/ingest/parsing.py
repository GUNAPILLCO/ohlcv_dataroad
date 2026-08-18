"""Parseo linea a linea de los archivos crudos OHLCV de MNQ (TDA-00).

Este modulo hace UNA sola cosa: convertir cada linea de texto de un archivo
``NN_mnq_MM_YY.Last.txt`` en, exactamente, uno de estos dos resultados:

1. una fila bien formada (timestamp + 4 precios + volumen, todos numericos
   y finitos), o
2. un :class:`ParseError` trazable (numero de linea, contenido crudo,
   motivo exacto).

Ninguna linea desaparece sin dejar rastro: toda linea del archivo termina
en una de las dos categorias de arriba. Esto es lo que permite, mas tarde,
la verificacion de conservacion (``lineas_leidas == filas_bien_formadas +
errores_de_parseo``) que se comprueba en ``tda00_integrity.py`` -- el mismo
principio que la memoria heredada del proyecto marca como patron validado
(nunca perder una fila silenciosamente, ver
``docs/instruments/mnq/MNQ_DATA_PRIOR_KNOWLEDGE.md``, seccion 7.1, regla 10).

Por que no se usa el parser C de pandas para esto: ``pandas.read_csv`` con
``errors="coerce"`` convierte un valor no numerico en ``NaN`` de forma
silenciosa, sin registrar de que linea vino. Aqui se controla cada campo de
forma explicita para poder atribuir el motivo exacto del fallo a su numero
de linea -- ese nivel de trazabilidad es exactamente lo que pide la tarea.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path

import pandas as pd

# Columnas del DataFrame de filas bien formadas que devuelve parse_raw_file.
ROW_COLUMNS = [
    "row_index",     # numero de linea dentro del archivo (1-based)
    "timestamp",     # datetime ya parseado
    "timestamp_raw", # texto original del campo timestamp, para trazabilidad
    "open", "high", "low", "close",  # precios, como float
    "volume",        # como int
]


@dataclass(frozen=True)
class ParseError:
    """Una linea que NO pudo convertirse en una fila valida.

    Atributos
    ---------
    row_index : numero de linea dentro del archivo (1-based), para poder
        localizar la linea exacta en el archivo original sin ambiguedad.
    raw_line  : contenido textual completo de la linea, sin modificar.
    rule      : codigo corto y estable de la causa del fallo. Los valores
        posibles son "schema_field_count", "parse_error_timestamp",
        "parse_error_numeric", "null_value" y "non_finite_value" -- el
        mismo vocabulario de reglas que usan las invariantes de
        ``tda00_integrity.py``, para que la tabla de violaciones final use
        un unico catalogo de codigos.
    detail    : explicacion legible, en espanol, de por que fallo esa linea
        en concreto (que campo, que valor, que se esperaba).
    """

    row_index: int
    raw_line: str
    rule: str
    detail: str


@dataclass
class ParsedFile:
    """Resultado de parsear un archivo crudo completo.

    Atributos
    ---------
    rows         : DataFrame con una fila por linea bien formada (columnas:
        ``ROW_COLUMNS``).
    parse_errors : lista de :class:`ParseError`, una por linea descartada
        del parseo.
    total_lines  : numero de lineas efectivamente leidas del archivo. Debe
        cumplirse siempre ``total_lines == len(rows) + len(parse_errors)``;
        esa igualdad es la verificacion de conservacion de TDA-00.
    """

    rows: pd.DataFrame
    parse_errors: list[ParseError]
    total_lines: int


def _parse_price_field(
    field_name: str, raw_value: str, row_index: int, raw_line: str
) -> tuple[float | None, ParseError | None]:
    """Convierte el texto crudo de UN campo de precio a ``float``.

    Entrada: el nombre logico del campo (para poder describir el error),
    su valor crudo (todavia como texto) y la posicion de la linea (para
    poder trazar el error si el campo no es valido).

    Transformacion, en orden:
    1. Si el texto (sin espacios) esta vacio -> ``null_value``.
    2. Si ``float(...)`` lanza ``ValueError`` (texto no numerico, p.ej.
       ``"abc"``) -> ``parse_error_numeric``.
    3. Si el float resultante no es finito (``inf``, ``-inf``, ``nan`` --
       ``float("inf")`` es una conversion valida en Python, por eso hace
       falta este chequeo aparte) -> ``non_finite_value``.
    4. En cualquier otro caso, se devuelve el valor parseado y ningun error.

    Salida: una tupla ``(valor_o_None, error_o_None)``; exactamente uno de
    los dos elementos es ``None``. El llamador decide que hacer con cada
    caso (aqui: pasar a la siguiente linea si hubo error).
    """
    value = raw_value.strip()
    if value == "":
        return None, ParseError(row_index, raw_line, "null_value", f"campo '{field_name}' vacio")
    try:
        parsed = float(value)
    except ValueError:
        return None, ParseError(
            row_index, raw_line, "parse_error_numeric",
            f"campo '{field_name}'='{value}' no es numerico",
        )
    if not math.isfinite(parsed):
        return None, ParseError(
            row_index, raw_line, "non_finite_value",
            f"campo '{field_name}'='{value}' no es un valor finito (inf/-inf/nan)",
        )
    return parsed, None


def parse_raw_file(path: Path, timestamp_format: str, separator: str = ";") -> ParsedFile:
    """Parsea un archivo crudo ``.Last.txt`` linea a linea.

    Entrada
    -------
    path             : ruta al archivo crudo (texto plano, ``;``-separado).
    timestamp_format : formato ``strptime`` esperado para el campo de
        timestamp (viene de la configuracion, no esta hardcodeado aqui).
    separator        : separador de campos (por defecto ``";"``).

    Logica, linea a linea
    ----------------------
    Cada linea pasa por dos comprobaciones, en orden. La primera linea que
    falla determina el destino de toda la fila (no se sigue evaluando):

    1. **Esquema**: se divide la linea por ``separator`` y se exige que el
       resultado tenga EXACTAMENTE 6 campos (timestamp, open, high, low,
       close, volume). Si no -> ``schema_field_count``.
    2. **Tipos**: se intenta convertir cada campo a su tipo esperado, en
       este orden: timestamp (``datetime.strptime``), open, high, low,
       close (float, via ``_parse_price_field``), volume (``int``). El
       primer campo que falla decide el motivo del error de esa linea.

    Nota sobre lineas vacias: ``readlines()`` NO produce una entrada extra
    por el salto de linea final normal de un archivo de texto (un archivo
    que termina en ``"...ultima_linea\n"`` produce tantos elementos como
    lineas de datos, ni uno mas). Una entrada vacia en ``raw_lines`` solo
    aparece cuando el archivo contiene una linea EN BLANCO real (p.ej.
    ``"...ultima_linea\n\n"``). Esa linea en blanco se cuenta y se procesa
    igual que cualquier otra: al dividirla por ``separator`` produce un
    unico campo vacio (``len(fields) == 1 != 6``), por lo que queda
    trazada como ``schema_field_count`` -- no se descarta en silencio, ni
    siquiera si es la ultima linea del archivo.

    Salida
    ------
    :class:`ParsedFile` con las filas bien formadas separadas de los
    errores de parseo, ambos con trazabilidad completa a numero de linea y,
    para los errores, al contenido crudo de la linea.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw_lines = fh.readlines()

    total_lines = len(raw_lines)
    parse_errors: list[ParseError] = []
    parsed_rows: list[tuple] = []

    for row_index, raw_line_with_newline in enumerate(raw_lines, start=1):
        raw_line = raw_line_with_newline.rstrip("\r\n")
        fields = raw_line.split(separator)

        if len(fields) != 6:
            parse_errors.append(ParseError(
                row_index, raw_line, "schema_field_count",
                f"se esperaban 6 campos separados por '{separator}', se encontraron {len(fields)}",
            ))
            continue

        ts_raw, open_raw, high_raw, low_raw, close_raw, volume_raw = fields
        ts_raw_stripped = ts_raw.strip()

        try:
            timestamp = datetime.strptime(ts_raw_stripped, timestamp_format)
        except ValueError:
            parse_errors.append(ParseError(
                row_index, raw_line, "parse_error_timestamp",
                f"timestamp '{ts_raw}' no coincide con el formato esperado '{timestamp_format}'",
            ))
            continue

        open_v, err = _parse_price_field("open", open_raw, row_index, raw_line)
        if err is not None:
            parse_errors.append(err)
            continue
        high_v, err = _parse_price_field("high", high_raw, row_index, raw_line)
        if err is not None:
            parse_errors.append(err)
            continue
        low_v, err = _parse_price_field("low", low_raw, row_index, raw_line)
        if err is not None:
            parse_errors.append(err)
            continue
        close_v, err = _parse_price_field("close", close_raw, row_index, raw_line)
        if err is not None:
            parse_errors.append(err)
            continue

        volume_raw_stripped = volume_raw.strip()
        if volume_raw_stripped == "":
            parse_errors.append(ParseError(row_index, raw_line, "null_value", "campo 'volume' vacio"))
            continue
        try:
            volume_v = int(volume_raw_stripped)
        except ValueError:
            parse_errors.append(ParseError(
                row_index, raw_line, "parse_error_numeric",
                f"campo 'volume'='{volume_raw_stripped}' no es un entero",
            ))
            continue

        parsed_rows.append(
            (row_index, timestamp, ts_raw_stripped, open_v, high_v, low_v, close_v, volume_v)
        )

    rows_df = pd.DataFrame(parsed_rows, columns=ROW_COLUMNS)
    return ParsedFile(rows=rows_df, parse_errors=parse_errors, total_lines=total_lines)
