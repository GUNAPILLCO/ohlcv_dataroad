# `ohlcv_dataroad.ingest` — explicación del código de TDA-00 y TDA-01

Este documento explica, bloque por bloque, qué hace cada archivo de este
paquete. Está escrito para alguien que está aprendiendo análisis de datos
y estadística: se explica qué entra, qué transformación se aplica, qué
sale, y **por qué** esa operación es necesaria — no solo qué línea hace
qué.

Contexto en una frase: TDA-00 responde *¿cada fila del archivo crudo es una
barra OHLCV admisible?*; TDA-01 responde *¿qué significa temporalmente esa
fila — el timestamp marca inicio o cierre de la barra?*; TDA-02 responde
*¿el eje temporal está completo respecto de la grilla que la estructura
NATIVA de negociación de CME esperaría, y cómo se explica cada ausencia?*;
TDA-03 responde *¿cómo se pasa de 22 archivos, uno por contrato trimestral,
a una única serie con un contrato activo por timestamp, de forma causal y
trazable?*; TDA-04 responde *¿cuál es el retorno realizado de 1 minuto de
esa serie, y en qué barras NO puede calcularse sin violar causalidad o sin
mezclar dos jornadas/contratos distintos?*; TDA-05 responde *¿en qué medida
ese retorno es una variable efectivamente DISCRETA, por el tick mínimo del
contrato, y cómo cambia esa discreción por hora del día y por año?*; TDA-06
responde *¿existe un patrón DETERMINISTA de actividad/volatilidad ligado al
reloj — minuto del día y día de la semana —, está en la media o en la
magnitud, es estable entre años, y qué segmentación de sesión sugieren los
propios datos?* — sin estudiar todavía la distribución marginal completa,
la dependencia estocástica (ACF, clustering de volatilidad) ni ningún
horizonte de predicción (eso es TDA-07 en adelante, fuera de esta tarea).

Los archivos, en el orden en que se ejecutan:

```
config.py                       -> carga configs/mnq_snapshot.yaml
ingest/parsing.py                -> convierte texto crudo en filas o errores de parseo
ingest/holdout_guard.py          -> proteccion del hold-out, compartida por TDA-00/01/02/03/04/05/06
ingest/tda00_integrity.py        -> aplica las invariantes y escribe los informes de TDA-00
ingest/run_tda00.py              -> punto de entrada de terminal de TDA-00
ingest/tda01_temporal_semantics.py -> evidencia forense de huecos para TDA-01
ingest/run_tda01_forensics.py    -> punto de entrada de terminal de TDA-01
ingest/session_calendar.py       -> calendario nativo CME_Equity y grilla esperada (TDA-02)
ingest/tda02_temporal_integrity.py -> cobertura, clasificacion de huecos, TH04, DST, STOP-2 (TDA-02)
ingest/run_tda02.py              -> punto de entrada de terminal de TDA-02
ingest/tda03_rolls.py            -> rollover causal, basis, serie continua, invariancia, STOP-3 (TDA-03)
ingest/run_tda03.py              -> punto de entrada de terminal de TDA-03
ingest/tda04_analysis_variables.py -> retorno de 1 minuto, reglas de no-cruce, auditoria, STOP-4 (TDA-04)
ingest/run_tda04.py              -> punto de entrada de terminal de TDA-04
ingest/tda05_effective_resolution.py -> movimiento en ticks, discrecion, tick/sigma, STOP-5 (TDA-05)
ingest/run_tda05.py              -> punto de entrada de terminal de TDA-05
ingest/tda06_intraday_calendar_profile.py -> perfil por minuto/dia de semana, segmentacion, s(m), STOP-6 (TDA-06)
ingest/run_tda06.py              -> punto de entrada de terminal de TDA-06
```

---

## 1. `config.py`

### Qué problema resuelve

Antes de esta etapa, decisiones como "el tick de MNQ es 0.25" o "el
hold-out empieza el 2025-06-23" no vivían en ningún sitio versionado — se
habrían escrito como números sueltos dentro del código de análisis. Este
módulo lee esas decisiones desde `configs/mnq_snapshot.yaml` y las expone
como un objeto Python (`SnapshotConfig`), para que el resto del código
nunca tenga que abrir el YAML directamente.

### Bloque: `SnapshotConfig` (dataclass)

```python
@dataclass(frozen=True)
class SnapshotConfig:
    repo_root: Path
    raw_dir: Path
    ...
```

- **Entrada**: nada, es solo la definición de una estructura de datos.
- **Qué es un `dataclass(frozen=True)`**: una clase de Python que solo
  guarda datos (sin lógica), y que además es *inmutable* — una vez creada,
  ningún campo puede reasignarse. Esto es deliberado: la configuración con
  la que corrió un análisis no debería poder cambiar a mitad de ejecución.
- Las `@property` (`inventory_report_path`, `bad_data_mask_path`, etc.)
  son rutas *calculadas* a partir de otros campos (p.ej. `reports_dir /
  inventory_report_name`), para no tener que escribir esa concatenación en
  cada sitio del código que necesite esa ruta.

### Bloque: `load_config(config_path)`

```python
config_path = Path(config_path).resolve()
...
repo_root = config_path.parent.parent
```

- **Entrada**: la ruta al archivo `configs/mnq_snapshot.yaml`.
- **Transformación**: `config_path.parent` es la carpeta `configs/`;
  `.parent` otra vez es la raíz del repositorio. Esto permite que todas las
  rutas dentro del YAML (`data/raw/mnq`, `reports/mnq`, ...) estén escritas
  de forma relativa y legible, sin que el resultado dependa de *desde qué
  carpeta* se ejecute el script — un error común es que un script funcione
  al lanzarlo desde la raíz del repo y falle al lanzarlo desde otra carpeta.
- El resto de la función simplemente lee cada sección del YAML
  (`raw_data`, `instrument_spec`, `holdout`, `tda00`) y arma el
  `SnapshotConfig`.
- **Salida**: un `SnapshotConfig` con todas las rutas ya resueltas de
  forma absoluta.

---

## 2. `ingest/parsing.py`

### Qué problema resuelve

Convertir cada línea de texto de un archivo `.Last.txt` en, exactamente,
uno de dos resultados: **una fila válida** o **un error de parseo
trazable** (con número de línea, contenido crudo y motivo). Ninguna línea
puede desaparecer sin dejar rastro — de ahí sale la verificación de
conservación (`total_lines == filas_válidas + errores_de_parseo`) que se
comprueba más adelante.

**Por qué no usar directamente `pandas.read_csv`**: el parser rápido (en
C) de pandas, si le pides tolerancia a errores, convierte silenciosamente
un valor no numérico en `NaN` — pero no te dice de qué línea vino. Aquí se
controla cada campo a mano, precisamente para poder señalar la línea
exacta y el motivo exacto de cada fallo.

### Bloque: `ParseError` y `ParsedFile` (dataclasses)

Son solo contenedores de datos:
- `ParseError`: una línea que falló — número de línea, texto original,
  código corto del motivo (`rule`) y una frase explicativa (`detail`).
- `ParsedFile`: el resultado de parsear un archivo completo — un
  `DataFrame` de filas válidas, una lista de `ParseError`, y
  `total_lines` (cuántas líneas se leyeron en total).

### Bloque: `_parse_price_field(field_name, raw_value, row_index, raw_line)`

```python
value = raw_value.strip()
if value == "":
    return None, ParseError(..., "null_value", ...)
try:
    parsed = float(value)
except ValueError:
    return None, ParseError(..., "parse_error_numeric", ...)
if not math.isfinite(parsed):
    return None, ParseError(..., "non_finite_value", ...)
return parsed, None
```

- **Entrada**: el texto crudo de un campo de precio (p.ej. `"100.25"`),
  más los datos necesarios para poder describir un error si ocurre.
- **Transformación**, en tres pasos, cada uno con su propia razón de ser:
  1. **Vacío → `null_value`**. Un campo vacío (`""`) no es un `float`
     válido, y merece un motivo distinto de "no es numérico": no había
     ningún valor, ni siquiera uno mal escrito.
  2. **`float(value)` falla → `parse_error_numeric`**. Cubre el caso de
     texto genuinamente no numérico (p.ej. `"abc"`).
  3. **`math.isfinite(parsed)` es `False` → `non_finite_value`**. Este
     paso existe porque `float("inf")` y `float("nan")` **son
     conversiones válidas en Python** — no lanzan `ValueError` — pero un
     precio infinito o "no un número" no es un dato admisible. Sin este
     paso, un valor `"inf"` en el archivo pasaría el chequeo anterior sin
     que nadie lo notara.
- **Salida**: una tupla `(valor, error)` donde exactamente uno de los dos
  es `None`. Quien llama a esta función decide qué hacer con cada caso
  (aquí: pasar a la siguiente línea si hubo error, sin seguir evaluando
  los campos restantes de esa línea).

### Bloque: `parse_raw_file(path, timestamp_format, separator)`

```python
with open(path, "r", encoding="utf-8") as fh:
    raw_lines = fh.readlines()

total_lines = len(raw_lines)
```

- **Entrada**: la ruta del archivo, el formato de fecha esperado (p.ej.
  `"%Y%m%d %H%M%S"`) y el separador de campos (`;`).
- Se lee el archivo completo como una lista de líneas de texto, sin
  descartar nada todavía.

> **Corrección puntual (revisión posterior a la validación de TDA-00):**
> la primera versión de este archivo eliminaba en silencio la última
> línea si estaba vacía, razonando que era "el artefacto normal de que un
> archivo de texto termine con un salto de línea". Ese razonamiento era
> **incorrecto**: `readlines()` no produce una entrada extra por el salto
> de línea final normal — un archivo que termina en
> `"...ultima_linea\n"` da tantos elementos como líneas de datos, ni uno
> más. Una entrada vacía en `raw_lines` **solo** aparece cuando el
> archivo contiene una línea en blanco real (p.ej.
> `"...ultima_linea\n\n"`, con dos saltos de línea seguidos). Descartarla
> ocultaba, sin dejar rastro, una línea que sí formaba parte del archivo.
> Ahora esa línea **se cuenta** como cualquier otra y se deja que la
> compruebe el mismo camino que ya existía para el resto de líneas (ver
> el bloque siguiente): al dividir una cadena vacía por `;`, `"".split(";")`
> da `['']` — un único campo, no 6 — así que la línea en blanco queda
> trazada automáticamente como `schema_field_count`, sin necesitar ningún
> caso especial adicional. Esto no cambió ningún resultado sobre los
> datos reales de `data/raw/mnq/`: los 22 archivos del conjunto de
> investigación no contienen ninguna línea en blanco real, así que el
> recorte anterior era, en la práctica, un no-operación sobre ellos —
> pero sí protegía incorrectamente contra el caso de que apareciera una
> en una futura exportación.

```python
for row_index, raw_line_with_newline in enumerate(raw_lines, start=1):
    raw_line = raw_line_with_newline.rstrip("\r\n")
    fields = raw_line.split(separator)

    if len(fields) != 6:
        parse_errors.append(ParseError(..., "schema_field_count", ...))
        continue
```

- Se recorre cada línea, numerada desde 1 (`row_index`), que es el número
  que después aparece en `TDA00_violaciones.csv` para localizar la línea
  exacta en el archivo original.
- **Primera comprobación — esquema**: dividir por `;` debe dar
  exactamente 6 trozos (timestamp, open, high, low, close, volume). Si no,
  se registra `schema_field_count` y se pasa a la siguiente línea — el
  resto de los chequeos no tiene sentido sobre una línea que ni siquiera
  tiene la forma correcta.

```python
    try:
        timestamp = datetime.strptime(ts_raw_stripped, timestamp_format)
    except ValueError:
        parse_errors.append(ParseError(..., "parse_error_timestamp", ...))
        continue
```

- **Segunda comprobación — timestamp**: se intenta interpretar el primer
  campo con el formato declarado en la configuración. Si el texto no
  encaja (p.ej. viene con otro formato de fecha), se registra el error y
  se descarta la línea.

```python
    open_v, err = _parse_price_field("open", open_raw, row_index, raw_line)
    if err is not None:
        parse_errors.append(err); continue
    ... (igual para high, low, close)
```

- **Tercera comprobación — los 4 precios**, uno a uno, usando la función
  explicada arriba. Se corta en el primer campo que falle: si `open` ya es
  inválido, no tiene sentido seguir evaluando `high`.

```python
    volume_raw_stripped = volume_raw.strip()
    if volume_raw_stripped == "":
        parse_errors.append(ParseError(..., "null_value", ...)); continue
    try:
        volume_v = int(volume_raw_stripped)
    except ValueError:
        parse_errors.append(ParseError(..., "parse_error_numeric", ...)); continue
```

- **Cuarta comprobación — volumen**: se trata aparte de los precios porque
  es un `int`, no un `float` (el volumen es un número de contratos, un
  conteo, no una magnitud continua).

```python
    parsed_rows.append((row_index, timestamp, ts_raw_stripped, open_v, high_v, low_v, close_v, volume_v))
...
rows_df = pd.DataFrame(parsed_rows, columns=ROW_COLUMNS)
return ParsedFile(rows=rows_df, parse_errors=parse_errors, total_lines=total_lines)
```

- Si la línea pasó las cuatro comprobaciones, se guarda como tupla en una
  lista de Python (`parsed_rows`). El `DataFrame` se construye **una sola
  vez, al final**, a partir de esa lista — construirlo fila a fila dentro
  del bucle sería mucho más lento, porque cada `DataFrame` es una
  estructura pesada de crear.
- **Salida**: un `ParsedFile` con el `DataFrame` de filas válidas y la
  lista de errores, ambos trazables a su número de línea original.

---

## 3. `ingest/holdout_guard.py`

### Qué problema resuelve

Proteger el hold-out no es responsabilidad de una etapa concreta: es una
regla de gobernanza que aplica a **cualquier** etapa que procese
`config.research_files` (`docs/instruments/mnq/MNQ_DATA_PRIOR_KNOWLEDGE.md`,
sección 11). Este módulo nació de una corrección puntual: la primera
versión de TDA-01 no tenía ninguna protección propia del hold-out (sólo
"por diseño" no abría esos archivos, sin una comprobación activa), mientras
que TDA-00 sí la tenía, pero definida dentro de `tda00_integrity.py` — si
una tercera etapa hubiera necesitado la misma protección, la habría
duplicado por tercera vez. Extraerla aquí, una sola vez, y que TDA-00 y
TDA-01 la **importen** en vez de reescribirla, es lo que garantiza que
ambas comparten exactamente la misma regla — no dos reglas parecidas que
puedan divergir con el tiempo.

### Bloque: `HoldoutIsolationError`

Una excepción propia (no un `ValueError` genérico) para que quien llama
pueda distinguir "la configuración compromete el aislamiento del hold-out"
de cualquier otro tipo de error.

### Bloque: `validate_research_holdout_disjoint(config)`

```python
research_set = set(config.research_files)
holdout_set = set(config.holdout_files)
overlap = research_set & holdout_set
if overlap:
    raise HoldoutIsolationError(...)
```

- **Entrada**: únicamente las dos listas de nombres de archivo de la
  configuración — **ningún archivo se abre para esta comprobación**.
- **Transformación**: convertir cada lista a `set` y calcular la
  intersección (`&`) es la forma directa de preguntar "¿hay algún nombre
  que aparezca en las dos listas a la vez?". También se comprueba que
  ninguna lista tenga nombres repetidos dentro de sí misma
  (`len(lista) != len(set(lista))`).
- **Cuándo debe llamarse**: siempre lo primero, antes de crear cualquier
  directorio de salida o abrir cualquier archivo — así, si la
  configuración está mal escrita, no se procesa nada.

### Bloque: `validate_last_timestamps_before_boundary(config, last_timestamps)`

```python
boundary = pd.Timestamp(config.holdout_boundary_utc)
offending = [
    (name, ts) for name, ts in last_timestamps.items()
    if ts is not None and ts >= boundary
]
if offending:
    raise HoldoutIsolationError(...)
```

- **Entrada**: la configuración (para leer `holdout_boundary_utc`) y un
  **diccionario** `{nombre_de_archivo: ultimo_timestamp_o_None}` — no una
  lista de resultados de una etapa concreta. Esta es la generalización
  clave que permite compartir la función entre TDA-00 (que construye el
  diccionario a partir de sus `FileResult`) y TDA-01 (que lo construye con
  un simple `rows.groupby("source_file")["timestamp"].max().to_dict()`):
  la función no necesita saber nada de cómo cada etapa procesa sus
  archivos, sólo el resultado final (el último timestamp de cada uno).
- **Transformación**: por cada archivo, se compara su último timestamp
  contra la frontera. Basta con el máximo por archivo — si es anterior a
  la frontera, todas las filas de ese archivo también lo son.
- **Por qué nunca abre un archivo de hold-out**: no necesita hacerlo — el
  diccionario de entrada ya contiene toda la información necesaria,
  calculada por quien llama a partir de archivos de investigación que ya
  procesó.

**Uso desde TDA-00** (`tda00_integrity.py`): re-exporta
`HoldoutIsolationError` desde este módulo (para no romper código que hace
`from ohlcv_dataroad.ingest.tda00_integrity import HoldoutIsolationError`)
y llama a las dos funciones desde `run_tda00`. **Uso desde TDA-01**: igual,
desde `build_forensic_evidence` en `tda01_temporal_semantics.py` — ver
§6.

---

## 4. `ingest/tda00_integrity.py`

### Qué problema resuelve

Es el núcleo de TDA-00: define qué cuenta como "violación de invariante" y
calcula, para cada fila ya bien formada, si la viola. También orquesta
todo el pipeline (recorrer los archivos del conjunto de investigación,
verificar conservación, escribir los cuatro artefactos de salida).

### Catálogo de reglas (constantes al principio del archivo)

```python
PARSE_TIME_RULES = ["schema_field_count", "parse_error_timestamp", "parse_error_numeric"]
HARD_INVARIANT_RULES = [
    "null_value", "non_finite_value", "nonpositive_price", "negative_volume",
    "ohlc_incoherent", "tick_grid_violation", "duplicate_timestamp_within_file",
    "timestamp_out_of_order", "duplicate_exact_row",
]
ALL_HARD_RULES = PARSE_TIME_RULES + HARD_INVARIANT_RULES
```

Estas tres listas son el "vocabulario" que usa todo el módulo. Tenerlas
como constantes explícitas (en vez de strings repetidos en el código)
evita errores de tipeo y sirve como documentación: cualquiera que lea el
archivo ve de inmediato cuáles son *todas* las reglas que existen.

Aparte, hay una bandera que **no** es una regla dura:

```python
# zero_volume -- ver más abajo, en _hard_invariant_checks
```

`volume == 0` se cuenta y se reporta, pero no cuenta como violación: el
roadmap solo exige `volume >= 0`, y la semántica de una barra sin
operaciones (fila ausente vs. volumen 0 vs. relleno hacia adelante) es una
pregunta abierta de una etapa posterior (`TDA-02`/`TH04`), no algo que
TDA-00 pueda decidir.

### Bloque: `_hard_invariant_checks(df, tick_size)`

Esta es la función más importante del archivo. Recibe el `DataFrame` de
filas bien formadas de **un** archivo y devuelve, para cada fila, un
booleano por cada regla.

**Por qué está vectorizada y no en un bucle `for`**: un archivo tiene
entre 50.000 y 100.000 filas. Comparar columna contra columna con pandas
(`df["high"] < df["low"]`) evalúa la condición sobre *todas* las filas a
la vez usando código optimizado; hacerlo con un `for fila in df` sería
cientos de veces más lento sin cambiar el resultado.

```python
nonpositive_price = (df[price_cols] <= 0).any(axis=1)
negative_volume = df["volume"] < 0
```

- **Entrada**: las columnas de precios (`open, high, low, close`) y
  `volume`.
- **Transformación**: `df[price_cols] <= 0` compara las 4 columnas de
  precio contra 0 a la vez, dando una tabla de booleanos del mismo tamaño;
  `.any(axis=1)` colapsa esa tabla a **una** columna: `True` si *alguno*
  de los 4 precios de esa fila es ≤ 0.
- **Por qué es necesario**: un precio ≤ 0 no tiene sentido económico para
  un futuro cotizando en puntos de índice — sería, casi con certeza, un
  error de captura o de transmisión del dato, no un precio real.

```python
oc_max = df[["open", "close"]].max(axis=1)
oc_min = df[["open", "close"]].min(axis=1)
ohlc_incoherent = (df["high"] < df["low"]) | (df["high"] < oc_max) | (df["low"] > oc_min)
```

- **Transformación**: por definición, en una barra OHLC válida, `High` es
  el precio más alto de todo el minuto y `Low` es el más bajo — por lo
  tanto **tienen que ser mayor/igual y menor/igual que Open y Close**, que
  son solo dos puntos dentro de ese minuto (el primero y el último). Se
  comprueban las tres formas en que eso puede romperse: `High < Low`
  (imposible), `High` menor que el mayor de Open/Close, o `Low` mayor que
  el menor de Open/Close.
- **Por qué es necesario**: esta es la invariante estructural más básica
  de una barra OHLC (fundamento Tsay, capítulo 5: High y Low son
  *estadísticos de orden* — el máximo y el mínimo — de los precios
  operados en el intervalo). Si esto falla, la fila no representa una
  barra físicamente posible.

```python
ratio = df[price_cols] / tick_size
residual = (ratio - ratio.round()).abs()
tick_grid_violation = (residual > TICK_GRID_TOLERANCE).any(axis=1)
```

- **Entrada**: los 4 precios y el `tick_size` (0.25 para MNQ, desde la
  configuración — no un número inventado por el código).
- **Transformación**: si un precio es un múltiplo exacto del tick, al
  dividirlo por el tick el resultado es un entero (p.ej. `100.25 / 0.25 =
  401.0`). `ratio.round()` redondea al entero más cercano, y la diferencia
  absoluta entre el valor real y ese entero (`residual`) debería ser 0 si
  el precio está alineado a la grilla. Se compara contra una tolerancia
  pequeña (`1e-6`) en vez de exigir una igualdad exacta, porque la
  división en punto flotante puede introducir errores de redondeo del
  orden de `1e-12`–`1e-9` incluso en precios que sí están alineados —
  exigir `== 0` exacto generaría falsos positivos.
- **Por qué es necesario**: MNQ solo puede operar en múltiplos de su tick;
  un precio fuera de esa grilla sería, otra vez, indicio de un error de
  captura del dato, no de un precio real negociado.

```python
duplicate_timestamp_within_file = df["timestamp"].duplicated(keep=False)
```

- **Transformación**: `.duplicated(keep=False)` marca **todas** las
  apariciones de un valor repetido (no solo la segunda en adelante). Se
  usa `keep=False` en vez del valor por defecto para poder trazar el grupo
  completo de filas involucradas, no solo "cuál copia sobra".
- **Por qué es necesario**: en TDA-00 cada archivo es un único contrato;
  dos filas con el mismo timestamp dentro del mismo archivo no pueden ser
  ambas observaciones válidas del mismo minuto.

```python
prev_ts = df["timestamp"].shift(1)
timestamp_out_of_order = df["timestamp"] < prev_ts
timestamp_out_of_order.iloc[0] = False
```

- **Transformación**: `.shift(1)` desplaza la columna una posición hacia
  abajo, de forma que en la fila *i* queda el timestamp de la fila *i-1*.
  Comparar `timestamp < prev_ts` detecta el caso en que el tiempo
  **retrocede** respecto a la fila anterior del archivo. La primera fila
  no tiene fila anterior, así que se fuerza a `False` explícitamente (si
  no, quedaría comparada contra un valor vacío y el resultado sería
  ambiguo).
- Nota de diseño: un timestamp **igual** al anterior no se cuenta aquí
  como "fuera de orden" — ya lo captura la regla de arriba
  (`duplicate_timestamp_within_file`). Mezclar ambos criterios en una sola
  regla haría más difícil saber, al leer una violación, *cuál* de las dos
  cosas pasó realmente.

```python
duplicate_exact_row = df.duplicated(subset=["timestamp"] + price_cols + ["volume"], keep=False)
```

- **Transformación**: igual que la de duplicados de timestamp, pero
  exigiendo que **las 6 columnas** coincidan (timestamp + los 5 valores
  OHLCV), no solo el timestamp.
- **Por qué se incluye el timestamp en la comparación**: la memoria
  heredada del proyecto (`docs/instruments/mnq/MNQ_DATA_PRIOR_KNOWLEDGE.md`,
  sección 4.3) documenta un bug real de una versión anterior del pipeline,
  en el que la comprobación de "fila exactamente duplicada" comparaba solo
  las columnas OHLCV **sin el timestamp** — lo que habría marcado como
  duplicadas dos barras distintas que coincidieran por azar en sus 5
  valores numéricos. Esta implementación evita ese error deliberadamente.

**Salida de toda la función**: un `DataFrame` con una columna booleana por
regla, del mismo largo que la entrada — listo para que
`checks[HARD_INVARIANT_RULES].any(axis=1)` diga, fila por fila, si esa
fila viola *alguna* regla dura.

### Bloque: `_build_detail(rule, idx, df, tick_size)`

A diferencia de la función anterior, esta **no** está vectorizada — y es
intencional: solo se llama sobre las filas que *ya* se sabe que violan una
regla (un subconjunto normalmente pequeño o vacío), así que prioriza que
el mensaje sea claro y legible para un humano por encima de la velocidad.
Genera la frase que aparece en la columna `detail` de
`TDA00_violaciones.csv` — por ejemplo, para `ohlc_incoherent`, reconstruye
*cuál* de las tres sub-condiciones fue la que falló y con qué valores.

### Bloque: `process_file(path, source_file, timestamp_format, separator, tick_size)`

Es la función que procesa **un** archivo de principio a fin, combinando
`parse_raw_file` + `_hard_invariant_checks`:

1. Llama a `parse_raw_file` → obtiene filas bien formadas y errores de
   parseo.
2. Sobre las filas bien formadas, llama a `_hard_invariant_checks`.
3. Construye **dos** artefactos distintos, con propósitos distintos:
   - `violations` (lista, formato "largo"): una entrada por cada
     combinación (fila, regla) violada. Si una fila viola 2 reglas a la
     vez, genera 2 entradas — así lo pide el roadmap para
     `TDA00_violaciones.csv` ("una fila por violación").
   - `mask` (`DataFrame`, formato "ancho"): **una fila por cada línea del
     archivo original**, sin excepción — incluye tanto las filas válidas
     como los errores de parseo — con un booleano `bad_data` y la lista de
     reglas violadas (unidas con `;`). Este es el artefacto persistente
     completo.

> **Corrección puntual — columna `raw_line` en las violaciones.** Cada
> entrada de `violations` que se origina en un `ParseError` ahora incluye
> `"raw_line": err.raw_line` — el texto exacto de la línea que falló,
> conservado sin modificar. Antes de esta corrección, `ParseError` ya
> guardaba `raw_line` internamente (ver `parsing.py`), pero ese dato se
> perdía al construir el diccionario de violación: `TDA00_violaciones.csv`
> tenía columnas de valores numéricos (`open`, `high`, ...) que para un
> error de parseo son necesariamente `None` — precisamente porque la línea
> no pudo interpretarse — y ninguna columna con el texto original. Eso
> hacía imposible reconstruir, solo desde el CSV, qué decía exactamente
> una línea que falló por `schema_field_count` o `parse_error_numeric`.
> Para las violaciones sobre filas **ya bien formadas** (p.ej.
> `ohlc_incoherent`, `tick_grid_violation`) `raw_line` se deja en `None` a
> propósito: esas filas sí se parsearon con éxito, así que su contenido
> original es perfectamente reconstruible a partir de `timestamp_raw` +
> `open/high/low/close/volume` — guardar el texto crudo ahí sería
> redundante.

```python
violation_rules_series = pd.Series("", index=df.index, dtype=object)
if len(bad_idx) > 0:
    sub = checks.loc[bad_idx, HARD_INVARIANT_RULES]
    violation_rules_series.loc[bad_idx] = sub.apply(
        lambda r: ";".join(rule for rule in HARD_INVARIANT_RULES if r[rule]), axis=1
    )
```

- **Nota de rendimiento, explicada**: en vez de recorrer las ~90.000 filas
  del archivo para construir el texto de reglas violadas, se construye
  primero una columna vacía (`""` para todas las filas) y **solo** se
  itera sobre `bad_idx` — el subconjunto de filas efectivamente marcadas
  como malas. Con el snapshot actual (0 violaciones), ese trabajo es
  gratis; si en una futura versión del dato aparecieran violaciones, el
  costo crece con el número de violaciones, no con el tamaño del archivo.

### Bloque: `run_tda00(config)`

Es el orquestador de toda la etapa:

```python
for source_file in config.research_files:
    path = config.raw_dir / source_file
    if not path.exists():
        raise FileNotFoundError(...)
    result = process_file(path, ...)
    file_results.append(result)
```

- Recorre **únicamente** `config.research_files` — la lista de 22
  archivos declarada en `configs/mnq_snapshot.yaml`. Los 5 archivos del
  hold-out (`config.holdout_files`) nunca se mencionan en este bucle: el
  pipeline no tiene ninguna línea de código que abra esos archivos. Así se
  respeta la gobernanza del hold-out desde el primer artefacto productivo
  del repositorio.

```python
for r in file_results:
    s = r.summary
    expected = s["parsed_rows"] + s["parse_error_rows"]
    if expected != s["total_lines"]:
        raise AssertionError(...)
```

- **Verificación de conservación**: por cada archivo, el número de líneas
  leídas tiene que ser exactamente igual a filas válidas + errores de
  parseo — ninguna línea puede "perderse" entre el archivo y los
  artefactos de salida. Si esto falla, es un bug del pipeline (no un
  problema del dato), y se aborta con una excepción en vez de escribir
  artefactos parcialmente inconsistentes.

```python
mask_all = pd.concat([r.mask for r in file_results], ignore_index=True)
...
mask_all.to_parquet(config.bad_data_mask_path, index=False)
violations_all.to_csv(config.violations_report_path, index=False)
summary_all.to_csv(config.per_file_summary_path, index=False)
```

- Se concatenan los resultados de los 22 archivos en tres tablas globales
  y se escriben a disco: la máscara completa (formato `parquet`, elegido
  por ser compacto y rápido de leer para ~1.9 millones de filas), las
  violaciones en formato largo (`csv`, para revisión humana directa) y el
  resumen por archivo (`csv`).
- Por último, `_write_inventory_report` genera
  `reports/mnq/TDA00_inventario.md` a partir de las estadísticas ya
  calculadas — es una función puramente de formato, no calcula nada nuevo.

### Bloque: uso de la protección del hold-out en `run_tda00`

La lógica completa de `HoldoutIsolationError` y sus dos validaciones vive
ahora en `holdout_guard.py` (§3) — **no** en este archivo. Esto es un
cambio respecto de una versión anterior de `tda00_integrity.py`, en la que
ambas funciones estaban definidas localmente con nombres privados
(`_validate_research_holdout_disjoint`,
`_validate_research_before_holdout_boundary`); se extrajeron a un módulo
común para que TDA-01 pudiera reutilizar exactamente la misma protección
en vez de duplicarla (ver §3 para el razonamiento completo y la
explicación bloque a bloque de ambas funciones).

`tda00_integrity.py` sólo importa y usa:

```python
from ohlcv_dataroad.ingest.holdout_guard import (
    HoldoutIsolationError,
    validate_last_timestamps_before_boundary,
    validate_research_holdout_disjoint,
)
```

`HoldoutIsolationError` se re-exporta desde aquí (el `import` la deja
disponible en el espacio de nombres de este módulo) para que el código y
los tests existentes que hacían
`from ohlcv_dataroad.ingest.tda00_integrity import HoldoutIsolationError`
sigan funcionando sin cambios.

**Dónde se enganchan las dos funciones en `run_tda00`** (fragmento):

```python
def run_tda00(config: SnapshotConfig) -> dict:
    validate_research_holdout_disjoint(config)      # (1) antes de abrir nada

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    ...
    for source_file in config.research_files:
        ...                                            # (2) procesa SOLO research_files

    for r in file_results:
        ...                                            # (3) verifica conservacion

    validate_last_timestamps_before_boundary(           # (3b)
        config, {r.source_file: r.summary["last_timestamp"] for r in file_results}
    )

    mask_all = pd.concat(...)                           # (4) recien aqui se escribe algo
    ...
```

El paso (3b) construye el diccionario `{archivo: ultimo_timestamp}` a
partir de los `FileResult` que TDA-00 ya calculó — es el "adaptador"
específico de TDA-00 hacia la función genérica de `holdout_guard.py` (ver
§3 y, para el adaptador equivalente de TDA-01, §6).

**Cómo se traduce un fallo en la terminal**: `run_tda00.py` (el punto de
entrada de línea de comandos) captura `HoldoutIsolationError` junto con
`FileNotFoundError` y `AssertionError`, e imprime `TDA-00 status: FAIL`
con el motivo exacto, en vez de dejar pasar una traza de Python cruda.

---

## 5. `ingest/run_tda00.py`

Es el punto de entrada de terminal. Deliberadamente no contiene lógica de
negocio — solo:

```python
config = load_config(Path(args.config))
try:
    stats = run_tda00(config)
except (FileNotFoundError, AssertionError, HoldoutIsolationError) as exc:
    print("TDA-00 status: FAIL", file=sys.stderr)
    ...
    return 1
```

1. Carga la configuración.
2. Llama a `run_tda00`.
3. Si algo falla a nivel de *pipeline* (un archivo declarado en la config
   no existe, o falla la verificación de conservación), lo traduce a un
   mensaje `FAIL` legible y un código de salida distinto de cero — en vez
   de dejar que quien ejecute el script desde la terminal solo vea una
   traza de error de Python.
4. Si todo va bien, imprime un resumen con los números clave (líneas
   totales, violaciones, estado) y las rutas de los 4 archivos generados.

**Reproducir la ejecución**:

```bash
python -m ohlcv_dataroad.ingest.run_tda00 --config configs/mnq_snapshot.yaml
```

---

## 6. `ingest/tda01_temporal_semantics.py`

### Qué problema resuelve

TDA-01 necesita responder si el timestamp de una barra marca su **inicio**
o su **cierre**. La documentación oficial de NinjaTrader ya apunta a
"cierre" (ver `reports/mnq/TDA01_convencion_temporal.md`, §2.1), pero esa
documentación describe el comportamiento general de la plataforma, no la
exportación específica de este snapshot. Hace falta evidencia **propia**,
sacada de los datos, que corrobore o contradiga esa documentación. Este
módulo produce esa evidencia de forma reproducible: detecta huecos
temporales dentro de cada archivo y mide en qué hora local de mercado caen
sus bordes.

**Por qué esto es código "necesario" y no "código por escribir una etapa
nueva"**: sin él, la afirmación central del informe ("1.078 de 1.178
huecos tienen su borde exactamente en `17:00:00` hora de Nueva York") no
sería verificable por nadie más que releyendo miles de líneas a mano. El
módulo es la forma de que ese número se pueda recalcular con un comando.

### La idea central, en una frase

El corte de mantenimiento diario de CME es un cese de negociación real que
ocurre casi todos los días. Si el timestamp marca el *inicio* de la barra,
la última barra antes del corte debería etiquetarse **un minuto antes**
del corte (p. ej. `16:59`) y la primera tras la reapertura **en** la hora
de reapertura (`18:00`). Si marca el *cierre*, es al revés: la última
barra se etiqueta **en** la hora del corte (`17:00`) y la primera tras la
reapertura, **un minuto después** (`18:01`) — porque esa es la barra que
recién termina de cerrarse en ese instante. Contar en qué hora caen esos
bordes, sobre miles de huecos reales, es la prueba.

### Bloque: `load_research_rows(config)`

```python
frames = []
for source_file in config.research_files:
    path = config.raw_dir / source_file
    parsed = parse_raw_file(path, config.timestamp_format, config.separator)
    df = parsed.rows.copy()
    df["source_file"] = source_file
    frames.append(df)
all_rows = pd.concat(frames, ignore_index=True)
```

- **Entrada**: la configuración cargada (`SnapshotConfig`).
- **Transformación**: reutiliza `parse_raw_file` de `parsing.py` — la
  MISMA función que usa TDA-00 — sobre cada archivo de
  `config.research_files`. Reutilizarla (en vez de escribir un segundo
  parser) garantiza que TDA-01 ve exactamente las mismas filas que TDA-00
  ya certificó como bien formadas, sin duplicar lógica de parseo.
- **Por qué nunca toca `config.holdout_files`**: el `for` sólo itera
  `config.research_files`. No existe ninguna otra ruta en este módulo
  hacia un archivo de hold-out.
- **Salida**: un único `DataFrame` con todas las filas del conjunto de
  investigación, con una columna `source_file` para poder volver a separar
  por archivo más adelante.

### Bloque: `compute_intra_file_gaps(rows)`

```python
for source_file, group in rows.groupby("source_file", sort=False):
    ts = group["timestamp"].sort_values().reset_index(drop=True)
    diffs = ts.diff()
    for i in range(1, len(ts)):
        gap_minutes = diffs.iloc[i].total_seconds() / 60
        if gap_minutes > 1:
            records.append({...})
```

- **Entrada**: el `DataFrame` de `load_research_rows`.
- **Transformación**: `groupby("source_file")` separa las filas por
  archivo ANTES de calcular ninguna diferencia de tiempo. Esto es
  deliberado: el salto entre la última fila de un archivo y la primera del
  siguiente es una **transición de contrato** (dos instrumentos distintos),
  no un hueco dentro de la negociación continua de un mismo contrato.
  Calcularlo por archivo evita mezclar los dos fenómenos — si se
  calculara sobre la serie global concatenada, cada cambio de contrato
  aparecería como un "hueco" gigantesco y contaminaría por completo el
  análisis de huecos reales.
- `diffs = ts.diff()` calcula, para cada fila, la diferencia respecto de
  la fila anterior (dentro del mismo archivo, gracias al `groupby`).
  Cualquier diferencia mayor a 1 minuto es un hueco.
- **Salida**: un `DataFrame` con una fila por hueco encontrado —
  `source_file`, el timestamp justo antes del hueco, el timestamp justo
  después, y la duración en minutos.

### Bloque: `attach_ny_wallclock(gaps)` — el bloque metodológicamente más importante

```python
for col in ["before_ts_utc", "after_ts_utc"]:
    utc = pd.to_datetime(gaps[col]).dt.tz_localize("UTC")
    ny = utc.dt.tz_convert(NY_TZ)          # NY_TZ = ZoneInfo("America/New_York")
    ...
```

- **Entrada**: los timestamps de los bordes del hueco (tz-naive, pero que
  representan UTC — el hecho ya confirmado para este dataset).
- **Transformación, en dos pasos que NO pueden invertirse ni fusionarse**:
  1. `tz_localize("UTC")` — le dice a pandas "estos números ya son UTC",
     sin cambiar ningún valor, sólo *etiquetando* la zona.
  2. `tz_convert(NY_TZ)` — ahora sí recalcula la hora, aplicando las
     reglas de horario de verano de la zona `America/New_York` tal como
     están definidas en la base de datos de zonas horarias IANA (a través
     de `zoneinfo`, un módulo de la librería estándar de Python).
- **Por qué no usar un offset fijo (p. ej. restar 5 horas a mano)**:
  `America/New_York` no tiene un offset constante respecto a UTC — alterna
  entre `UTC-5` (horario estándar, EST) y `UTC-4` (horario de verano, EDT)
  dos veces al año. Un offset fijo daría la hora local correcta sólo
  durante media parte del año. `zoneinfo` conoce las fechas exactas de
  cada cambio y las aplica automáticamente; es la única forma de que el
  mismo código funcione todo el año sin una rama especial para DST. El
  test `test_ny_wallclock_across_dst_spring_transition` (en
  `tests/test_tda01_temporal_semantics.py`) verifica exactamente esto:
  la misma ancla de mercado (`17:00` NY) debe seguir apareciendo como
  `17:00` a ambos lados de una transición de marzo, aunque el offset UTC
  correspondiente cambie de `-5` a `-4`.
- **Salida**: el mismo `DataFrame` con columnas nuevas —
  `before_ny`/`after_ny` (timestamps con zona horaria adjunta) y
  `before_ny_time`/`after_ny_time` (su hora en texto `HH:MM:SS`, que es la
  forma en la que después se cuentan y agrupan en el informe).

### Bloque: `classify_gap_magnitude(gap_minutes)`

```python
if DAILY_MAINTENANCE_MIN_MINUTES <= gap_minutes <= DAILY_MAINTENANCE_MAX_MINUTES:
    return "daily_maintenance_like"
if gap_minutes > LONG_GAP_MIN_MINUTES:
    return "weekly_or_long"
return "other"
```

- **Entrada**: la duración de un hueco, en minutos.
- **Transformación**: una clasificación puramente por **tamaño** (30-90
  min, > 40 h, o cualquier otra cosa) — **no** una afirmación de que un
  hueco "es" el corte de mantenimiento. Esa afirmación no se impone antes
  de mirar los datos: es el propio resultado (la concentración de más del
  90 % de los huecos de esa categoría en una única etiqueta horaria) el
  que la respalda a posteriori. Esta función sólo agrupa por orden de
  magnitud para poder aislar el subconjunto relevante.
- **Salida**: una de tres etiquetas de texto (ver docstring para el
  significado exacto de cada una).

### Bloque: `summarize_boundary_labels(gaps)`

- **Entrada**: un subconjunto de huecos ya procesado por
  `attach_ny_wallclock` (típicamente, sólo los `"daily_maintenance_like"`).
- **Transformación**: tres `value_counts()` — sobre la hora NY del borde
  anterior, la hora NY del borde posterior, y la duración en minutos. No
  hay ningún test estadístico formal aquí: la fuerza del resultado está en
  que una única etiqueta concentra una fracción muy alta del total, algo
  que un conteo simple de frecuencias ya deja ver con claridad.
- **Salida**: un objeto `BoundaryLabelSummary` con los tres conteos —
  exactamente los números que aparecen en la tabla de evidencia de
  `TDA01_convencion_temporal.md` §2.2.

### Bloque: `identify_early_close_like_gaps(gaps)` — la tercera ancla forense

**Por qué existe este bloque (corrección puntual).** Una versión anterior
del informe (`TDA01_convencion_temporal.md`) citaba "36 candidatos, 34/36
con borde `13:00`" como tercera ancla forense (cierres anticipados), pero
esa selección se había hecho de forma ad-hoc, fuera de este módulo, y no
era reproducible con un comando. Esta función reemplaza ese análisis suelto
por una regla explícita, versionada y testeada.

```python
other = gaps[gaps["magnitude_class"] == "other"]
matches = (
    (other["before_ny_time"] == EARLY_CLOSE_BEFORE_NY_TIME)   # "13:00:00"
    & (other["after_ny_time"] == EARLY_CLOSE_AFTER_NY_TIME)   # "18:01:00"
)
return other[matches].reset_index(drop=True)
```

- **Entrada**: el `DataFrame` completo de huecos, ya con `magnitude_class`
  calculada.
- **Transformación, en dos pasos**:
  1. Se restringe a la categoría `"other"` — ni del tamaño del corte
     diario ni de un fin de semana. Esto evita, por construcción, que un
     hueco ya contado en las anclas 1 o 2 se cuente otra vez aquí.
  2. Se exige una coincidencia **exacta** de los dos bordes a la vez: el
     borde anterior debe ser exactamente `"13:00:00"` NY **y** el
     posterior exactamente `"18:01:00"` NY (la misma etiqueta de
     reapertura que ya se observa en el ancla diaria). No es una
     tolerancia ni un rango — un hueco con borde anterior `"13:01:00"` no
     cuenta, aunque esté a un minuto de distancia. Esto es deliberado: es
     la firma de dos lados que la convención de cierre de barra predice
     para "cierre anticipado a las 13:00, reapertura normal esa misma
     tarde a las 18:01" — exigir ambos lados a la vez reduce el riesgo de
     capturar, por casualidad, un hueco que empieza cerca de esa hora por
     otro motivo.
- **Qué NO afirma esta función**: no dice que cada hueco seleccionado
  corresponda a una fecha de feriado concreta del calendario oficial de
  CME — eso requeriría cruzar cada fecha contra un calendario (TDA-02).
  Sólo afirma que el hueco tiene la forma que la convención ya establecida
  predice.
- **Salida**: el subconjunto de `gaps` que cumple la regla. Sobre el
  conjunto de investigación real, esto da **31** huecos, los 31 con
  duración exacta de 301 minutos — ver
  `TDA01_convencion_temporal.md` §2.2 para el resultado completo y su
  interpretación.

### Bloque: `build_forensic_evidence(config)`

```python
validate_research_holdout_disjoint(config)                 # (0)

rows = load_research_rows(config)                           # (1)

last_timestamps = rows.groupby("source_file")["timestamp"].max().to_dict()
validate_last_timestamps_before_boundary(config, last_timestamps)  # (2)

gaps = compute_intra_file_gaps(rows)                         # (3)
gaps = attach_ny_wallclock(gaps)
gaps["magnitude_class"] = gaps["gap_minutes"].apply(classify_gap_magnitude)
```

- **Paso (0)**: protección del hold-out, primer paso — igual que en
  `run_tda00` (§4), se valida `research_files`/`holdout_files` **antes**
  de abrir cualquier archivo, reutilizando `validate_research_holdout_disjoint`
  de `holdout_guard.py` (§3), no una copia local.
  (Nota: `run_tda00` está definido en `tda00_integrity.py`, §4; el
  punto de entrada de terminal correspondiente es `run_tda00.py`, §5.)
- **Paso (1)**: se cargan únicamente los archivos de investigación (§ arriba,
  `load_research_rows`).
- **Paso (2)**: el "adaptador" específico de TDA-01 hacia la función
  genérica de frontera temporal. A diferencia de TDA-00 (que construye el
  diccionario a partir de sus `FileResult`), aquí basta una línea:
  `rows.groupby("source_file")["timestamp"].max()` da, directamente, el
  último timestamp de cada archivo ya cargado — sin releer ni reabrir
  nada. Se pasa a `validate_last_timestamps_before_boundary` (§3), la
  misma función que usa TDA-00.
- **Paso (3)**: el análisis de huecos ya descrito arriba
  (`compute_intra_file_gaps` → `attach_ny_wallclock` →
  `classify_gap_magnitude`).
- **Salida**: la tabla completa de huecos del conjunto de investigación,
  lista para escribirse a disco (`run_tda01_forensics.py`, §7) o
  resumirse (`summarize_boundary_labels`, `identify_early_close_like_gaps`).

---

## 7. `ingest/run_tda01_forensics.py`

Punto de entrada de terminal, con el mismo espíritu que `run_tda00.py`: no
contiene lógica propia, sólo carga la configuración, llama a
`build_forensic_evidence` y escribe `reports/mnq/TDA01_evidencia_gaps.csv`.
Igual que `run_tda00.py`, captura `HoldoutIsolationError` (además de
`FileNotFoundError`) y las traduce a un mensaje `FAIL` legible en vez de
una traza cruda.

Imprime un resumen por consola con las **tres** anclas forenses por
separado (corte de mantenimiento, cierre/reapertura semanal, y cierres
anticipados vía `identify_early_close_like_gaps`) — las mismas cifras que
alimentan la tabla de §2.2 del informe.

**Reproducir la ejecución**:

```bash
python -m ohlcv_dataroad.ingest.run_tda01_forensics --config configs/mnq_snapshot.yaml
```

---

## 8. `ingest/session_calendar.py`

### Qué problema resuelve

TDA-02 necesita una "grilla esperada" de minutos — es decir, saber, para
cualquier instante, si el mercado *debería* tener una barra ahí o no. Esa
grilla tiene que salir de la estructura **nativa** de negociación de CME
para el complejo de futuros de índices bursátiles (al que pertenece MNQ),
no de una ventana operativa que alguien elija más adelante (RTH, un
régimen horario heredado, etc. — ver la tarea, sección 2). Este módulo
construye esa grilla usando `pandas_market_calendars` (alias del
calendario: `CME_Equity`), y — siguiendo la instrucción explícita de la
tarea de no tratar una librería de calendario como fuente de verdad sin
verificarla — **corrige dos reglas** de esa librería que resultaron
incorrectas para este instrumento, ambas respaldadas por cita documental
primaria de CME además de evidencia forense propia (ver más abajo,
`SECONDARY_BREAK_LAST_TRADING_DATE` y `FORENSIC_SCHEDULE_OVERRIDES`).

### Bloque: `get_cme_equity_calendar()` / `build_session_schedule(start, end, ...)`

- **Entrada**: un rango de fechas (típicamente el primer y último
  timestamp del conjunto de investigación).
- **Transformación**: pide a `pandas_market_calendars` el horario
  (`market_open`, `market_close`, y — cuando la librería lo declara —
  `break_start`/`break_end`) de cada sesión en `[start - buffer, end +
  buffer]`. El `buffer` (10 días por defecto,
  `configs/mnq_snapshot.yaml`, `tda02.calendar_buffer_days`) evita efectos
  de borde al resolver la sesión del primer/último timestamp real.
- El índice de la tabla resultante es la **fecha de CIERRE** de cada
  sesión (`trading_date`), no la de apertura — una sesión que abre
  domingo a la noche y cierra lunes a las 17:00 NY se etiqueta "lunes".
  Esto coincide exactamente con la convención de barra-cierre ya
  confirmada por TDA-01 (ver docstring del módulo para el razonamiento
  completo).
- **Corrección documental + forense #1 — break secundario eliminado**
  (`SECONDARY_BREAK_LAST_TRADING_DATE` / `SECONDARY_BREAK_ABOLISHED_TRADING_DATE`):
  la librería declara un break intradiario `15:15-15:30 America/Chicago`
  (`16:15-16:30` NY) como PERMANENTE. Fuente primaria citada para esta
  tarea: CME Special Executive Report / Globex Notice de junio de 2021 —
  el break fue ELIMINADO, con efecto en la fecha de negociación
  `2021-06-28`. Verificación forense independiente
  (`TDA01_evidencia_gaps.csv`): el hueco con la firma exacta de este
  break aparece en 353 sesiones consecutivas desde 2019-12-23 hasta
  **viernes 2021-06-25** — y nunca después. La fecha forense (última
  aparición) y la documental (eliminación con efecto el lunes siguiente)
  coinciden exactamente. Esta fecha de corte se usa en dos sitios: para
  EXCLUIR los minutos del break de la grilla esperada en sesiones
  `<= 2021-06-25` (`_session_expected_minutes`, ver más abajo) y para
  impedir que `classify_gaps` (§9) asigne `SECONDARY_MAINTENANCE_BREAK` a
  ningún hueco en sesiones posteriores.
- **Corrección documental + forense #2 — `2025-01-09`**
  (`FORENSIC_SCHEDULE_OVERRIDES`): la librería declara `2025-01-09` (día
  nacional de duelo por Jimmy Carter) como cierre **completo**, vía
  `CMEEquityExchangeCalendar.adhoc_holidays` (`USNationalDaysofMourning`).
  Fuente primaria citada para esta tarea: CME SER 9499R — la sesión de
  índices bursátiles de EE. UU. en Globex ese día fue ABREVIADA (cierre
  `08:30 CT`, reapertura normal `17:00 CT`), no cerrada. Verificado contra
  `data/raw/mnq/20_mnq_03_25.Last.txt`: volumen normal (hasta 10.450
  contratos/minuto) de forma casi continua desde la reapertura habitual
  del 8 de enero hasta `2025-01-09 14:30:00 UTC` (`09:30` NY = **`08:30`
  CT exacto**, coincide al minuto con el SER) — momento en que aparece un
  hueco de 511 minutos hasta la siguiente reapertura habitual. Es la
  ÚNICA fecha de `USNationalDaysofMourning` dentro del rango de
  investigación (verificado en
  `test_only_one_adhoc_mourning_holiday_falls_in_research_range`).
  El acceso directo a `cmegroup.com` para releer ambos avisos originales
  volvió a estar bloqueado en esta sesión: las dos citas se incorporan
  tal como fueron provistas para esta tarea, corroboradas por la
  convergencia exacta con la evidencia forense propia — no releídas
  directamente del documento original en esta sesión.
- **Salida**: un `SessionSchedule` (dataclass) que envuelve la tabla y
  expone dos operaciones:
  - `trading_date_for(timestamps_utc)`: localiza vectorialmente cada
    timestamp dentro de un `IntervalIndex` de `(market_open, market_close]`
    (cerrado por la derecha — bajo la convención de cierre de barra, el
    propio `market_close` SÍ pertenece a la sesión; `market_open` no,
    porque es el instante en que empieza a acumularse la primera barra,
    que se etiqueta un minuto después). Devuelve `NaT` para cualquier
    timestamp que no caiga dentro de ninguna sesión (fin de semana,
    feriado completo, o fuera de horario).
  - `is_early_close(trading_date)`: compara la hora de cierre de esa
    sesión, en hora de Nueva York, contra `NORMAL_SESSION_CLOSE_NY_TIME`
    (`17:00`) — usado para distinguir `EARLY_CLOSE` de
    `DAILY_MAINTENANCE` al clasificar un hueco.

### Bloque: `_session_expected_minutes(trading_date, market_open, market_close, break_start, break_end)`

- **Entrada**: los bordes de UNA sesión (incluidos los del break, que
  pueden ser `NaT`).
- **Transformación**: grilla completa `market_open + 1min .. market_close`;
  si `trading_date <= SECONDARY_BREAK_LAST_TRADING_DATE` (2021-06-25) y el
  calendario declara un break para esa sesión, se EXCLUYEN los minutos
  `break_start + 1min .. break_end` — el mercado estuvo estructuralmente
  detenido ahí, igual que en el corte de mantenimiento nocturno. Para
  sesiones posteriores (o sin break declarado) no se excluye nada.
- **Por qué existe este bloque**: antes de esta corrección, `TDA-02` ya
  clasificaba 353 huecos reales como `SECONDARY_MAINTENANCE_BREAK` (una
  detención real del mercado), pero la grilla esperada seguía generando
  esos mismos minutos como "esperados" — contabilizándolos como ausentes
  y sesgando la cobertura hacia abajo. Es la misma inconsistencia que el
  roadmap prohíbe para el corte de mantenimiento nocturno.
- **Salida**: `DatetimeIndex` de los minutos esperados de esa sesión.

### Bloque: `expected_bar_grid(schedule, clip_start=None, clip_end=None)` / `expected_bar_grid_frame(...)`

- **Entrada**: un `SessionSchedule` ya calculado, y opcionalmente un
  rango `[clip_start, clip_end]`.
- **Transformación**: para cada sesión, llama a `_session_expected_minutes`
  — la grilla esperada de timestamps de CIERRE de barra, ya sin los
  minutos del break secundario cuando corresponda. `expected_bar_grid_frame`
  hace lo mismo pero conserva la columna `trading_date` de cada minuto
  (más barata que volver a localizar cada minuto contra el calendario
  cuando ya se sabe, por construcción, a qué sesión pertenece).
- El recorte (`clip_start`/`clip_end`) es lo que permite acotar la
  grilla al rango propio de UN archivo (`file_start`/`file_end`) sin
  esperar barras antes/después de donde ese archivo realmente empieza y
  termina — ver §9, `compute_file_coverage`.
- **Salida**: `DatetimeIndex` (o `DataFrame` con `trading_date`), tz-aware
  UTC, ordenado, sin duplicados.
- **Invariante de contabilidad verificada**: ningún dato desaparece por
  excluir el break de la grilla — las 89 barras reales que caen dentro de
  ese tramo (el break no fue perfectamente limpio en el 25 % de las 353
  sesiones donde aplicó) se registran explícitamente en
  `find_out_of_grid_bars` con motivo `SECONDARY_BREAK_WINDOW` (más abajo),
  no desaparecen de ninguna contabilidad.

### Bloque: `full_holidays_in_range(start, end, ...)`

- **Entrada**: rango de fechas.
- **Transformación**: `cal.holidays().holidays` filtrado al rango,
  **excluyendo** cualquier fecha de `FORENSIC_SCHEDULE_OVERRIDES` (para
  que `2025-01-09` no aparezca dos veces con estados contradictorios: ya
  tiene su propia sesión de cierre anticipado en la tabla).
- **Salida**: lista ordenada de `datetime.date` — los feriados de cierre
  COMPLETO (sin ninguna sesión), usados por `classify_gaps` (§9) para
  distinguir `HOLIDAY` de `WEEKEND`.

---

## 9. `ingest/tda02_temporal_integrity.py`

### Qué problema resuelve

Es el núcleo de TDA-02: determina si el eje temporal del conjunto de
investigación está completo respecto de la grilla de `session_calendar.py`,
y clasifica cada ausencia. Reutiliza, sin duplicar lógica: la protección
del hold-out (`holdout_guard.py`), la carga de filas y el cálculo de
huecos internos (`tda01_temporal_semantics.py` — el MISMO
`compute_intra_file_gaps` que usó TDA-01, no una copia) y el calendario
(`session_calendar.py`).

### Bloque: `classify_gaps(gaps, schedule, holidays)` — clasificación causal

- **Entrada**: la salida de `compute_intra_file_gaps` (huecos internos,
  por archivo, ya con `before_ts_utc`/`after_ts_utc` localizados como
  tz-aware UTC) y el calendario.
- **Transformación** (función auxiliar `_classify_one_gap`, fila a fila —
  con sólo ~4.000 huecos en todo el conjunto de investigación, un bucle
  de Python es instantáneo y mucho más legible que vectorizar esta
  lógica condicional):
  1. Se localiza la `trading_date` de cada borde
     (`schedule.trading_date_for`). Si algún borde no cae en ninguna
     sesión → `UNKNOWN` (ver también §11, barras fuera de grilla).
  2. Si ambos bordes caen en la MISMA `trading_date` (hueco intradía) Y
     `before_date <= SECONDARY_BREAK_LAST_TRADING_DATE`
     (`session_calendar.py`, 2021-06-25): se compara contra el
     `break_start`/`break_end` que declara el calendario para esa sesión.
     Coincidencia exacta → `SECONDARY_MAINTENANCE_BREAK`; si no →
     `UNKNOWN`. Para fechas posteriores a ese corte, la causa nunca se
     asigna — la fecha se exige de forma EXPLÍCITA, no se depende de que
     simplemente no aparezcan huecos coincidentes después de esa fecha.
  3. Si caen en fechas distintas: se buscan fechas de negociación válidas
     ENTRE ambas (`np.searchsorted` sobre las `trading_date` ordenadas
     del calendario) que no tengan NINGUNA barra en este archivo → si las
     hay, `MISSING_TRADING_DAY` (jornada completa ausente — distinto de
     un huecos normal).
  4. Si no hay jornadas saltadas: se listan los días de CALENDARIO (no de
     negociación) entre ambas fechas. Si alguno es un feriado completo
     (`holidays`) → `HOLIDAY`. Si el salto es de 3+ días de calendario
     (fin de semana, sin feriado) → `WEEKEND`. Si es de un día
     (jornadas consecutivas) y la sesión de `before_date` cierra ANTES de
     la hora normal (`schedule.is_early_close`) → `EARLY_CLOSE`. En
     cualquier otro caso → `DAILY_MAINTENANCE`.
  5. Cada causa lleva una `confidence`: `ALTA` si los dos bordes
     coinciden EXACTAMENTE con el horario que declara el calendario para
     esas fechas (`before_ts == market_close`, `after_ts == market_open +
     1 min`); `MEDIA` si la causa es plausible pero algún borde no
     coincide exactamente (se documenta la discrepancia en
     `cause_detail`); `BAJA` sólo para `UNKNOWN`.
- **Por qué esto es más preciso que la heurística de magnitud de TDA-01**
  (30-90 min ≈ mantenimiento diario, etc.): aquí se compara contra la
  hora EXACTA que declara el calendario para CADA fecha concreta
  (incluyendo cierres anticipados, que tienen una hora de cierre
  distinta cada vez), no contra un rango fijo — por eso TDA-02 puede
  distinguir automáticamente `EARLY_CLOSE` de `DAILY_MAINTENANCE` sin
  necesitar la regla ad-hoc "13:00 → 18:01" que usaba TDA-01.
- **Salida**: el mismo `DataFrame` de huecos, con columnas nuevas:
  `before_date`, `after_date`, `cause`, `confidence`, `cause_detail`,
  `expected_ts_inicio`, `expected_ts_fin` (primer/último minuto que
  faltó, calculado directamente de los bordes observados — sin
  condicionar a la causa).

### Bloque: `build_file_boundary_records(file_bounds, schedule)`

- **Entrada**: `{source_file: (file_start_utc, file_end_utc)}` y el
  calendario.
- **Transformación**: compara el primer/último timestamp REAL de cada
  archivo contra el primer/último minuto IDEAL de su propia sesión. Si no
  coinciden, genera una fila informativa (`cause="FILE_BOUNDARY"`) — un
  archivo que empieza o termina a mitad de una jornada no es una pérdida
  de datos, es simplemente dónde empieza/termina el export de ese
  contrato dentro del snapshot (sección 5 de la tarea).
- **Por qué es una tabla SEPARADA del inventario de huecos internos**:
  para que nunca se cuente como pérdida de cobertura — la cobertura de
  cada archivo se calcula ya recortada a `[file_start, file_end]`
  (siguiente bloque), así que estos bordes son, por diseño, invisibles
  para el cálculo de cobertura.
- **Salida**: `TDA02_bordes_de_archivo.csv`.

### Bloque: `compute_file_coverage(rows_f, full_grid, schedule)`

- **Entrada**: las filas de UN archivo y la grilla esperada completa (ya
  calculada una sola vez sobre todo el rango de investigación, para no
  recalcularla 22 veces).
- **Transformación**: recorta `full_grid` a `[file_start, file_end]` de
  ESE archivo, y compara contra el conjunto de timestamps observados
  (`isin`). Agrega también por `trading_date` (`per_day`), anotando si
  cada día es de cierre anticipado.
- **Salida**: un `FileCoverage` (minutos esperados/presentes/ausentes,
  global y por día) — la base de `TDA02_dias_incompletos.csv` y de
  `aggregate_coverage_by_period` (suma de minutos por año/mes, no
  promedio de porcentajes, para que un año con pocos días no pese igual
  que uno completo).

### Bloque: `find_out_of_grid_bars(rows, full_grid, schedule)`

- **Entrada**: todas las filas del conjunto de investigación y la grilla
  esperada completa (`full_grid`, ya sin los minutos del break
  secundario pre-corte).
- **Transformación**: "fuera de grilla" se define de forma literal —
  cualquier timestamp observado que NO pertenezca a `full_grid`
  (`~ts.isin(full_grid["expected_ts_utc"])`), no sólo los que caen fuera
  de toda sesión como en una versión anterior de esta función. Esto
  garantiza que TODA fila observada quede contabilizada exactamente una
  vez, como "presente" (en la grilla) o como "fuera de grilla" (aquí) —
  sin ningún hueco de contabilidad entre ambas (verificado:
  `minutos_presentes + barras_fuera_de_grilla == filas_observadas`,
  exactamente, sobre el conjunto de investigación real). Se distingue el
  motivo (`reason`): `NO_SESSION` (el timestamp no cae en ninguna sesión
  — fin de semana, feriado, fuera de horario) o
  `SECONDARY_BREAK_WINDOW` (cae en una sesión válida, pero dentro del
  tramo del break que la grilla excluye para fechas `<= 2021-06-25` — una
  operación real ocurrida durante lo que se trata, estructuralmente,
  como mercado detenido). **No se elimina nada** (sección 11 de la
  tarea): sólo se registra, con su hora NY y su motivo, para revisión.
- **Salida**: `TDA02_barras_fuera_de_grilla.csv`.

### Bloque: `dst_transition_dates(years)` / `build_dst_evidence(...)`

- **Entrada**: los años que cubre el rango de investigación.
- **Transformación**: encuentra, para cada año, el primer día en que
  `America/New_York` cambia de offset UTC en marzo (`spring`) y en
  noviembre (`fall`) — buscando el cambio real de offset con `zoneinfo`,
  no asumiendo "segundo domingo de marzo" a mano (una regla de
  calendario que podría cambiar por ley). Para cada transición, se
  compara: el offset UTC antes/después (debe diferir en exactamente 1
  hora), la hora LOCAL de cierre (debe seguir siendo `17:00` NY, sin
  desplazarse) y el número de minutos esperados esa sesión (debe ser
  igual que cualquier otro día — la transición ocurre de madrugada,
  fuera de horario de mercado, así que no debería alterar la duración de
  ninguna sesión de este instrumento).
- **Salida**: `TDA02_dst_evidencia.csv` — una fila por transición
  (spring/fall) por año, reproducible sin tener que re-ejecutar nada.

### Bloque: `analyze_inactive_bar_candidates(rows, long_run_threshold=6)` — TH04

- **Entrada**: todas las filas del conjunto de investigación.
- **Transformación**: marca como *candidatas* las barras `open == high ==
  low == close`; las agrupa en *rachas* de minutos CONSECUTIVOS con el
  MISMO precio de cierre (una racha de longitud 1 es un evento aislado).
  Para las rachas largas (≥ `long_run_threshold`, 6 por defecto), compara
  si el VOLUMEN es constante dentro de la racha (señal de relleno
  sintético) o si varía minuto a minuto (señal de actividad real — un
  forward-fill clásico repite un valor fijo, normalmente 0; TDA-00 ya
  certificó 0 filas con `volume == 0` en todo el conjunto de
  investigación).
- **Por qué `O=H=L=C` sólo no es prueba** (sección 8 de la tarea, con
  razón): una barra puede tener una única operación real a un precio, con
  volumen positivo — eso no es relleno, es una barra genuinamente plana.
  Sobre los datos reales, las rachas largas encontradas (máximo 14
  minutos consecutivos) se concentran en fechas de volatilidad extrema
  conocida (marzo de 2020) y tienen volumen que VARÍA minuto a minuto
  (1 a 19 contratos) — evidencia en contra de forward-fill, no a favor.
- **Salida**: `InactiveBarEvidence` (barras planas, distribución de
  longitud de racha, rachas largas con detalle, y un veredicto en texto
  que NO fuerza una clasificación cuando la evidencia no alcanza).
- **Alcance de esta función, precisado**: sólo investiga `FORWARD_FILL`
  (presencia de filas rellenadas hacia adelante). La pregunta más amplia
  de TH04 — cómo se representa un minuto de mercado abierto SIN
  operaciones — no puede resolverse desde aquí: un hueco corto (fila
  ausente) es compatible con esa hipótesis, pero también con una pérdida
  de dato histórico, y ninguna de las dos se puede confirmar con OHLCV de
  1 minuto únicamente. Ver `reports/mnq/TDA02_cobertura.md`, §10.1, donde
  esa distinción — "barra faltante" (hecho confirmado) frente a "minuto
  confirmado sin trades" (interpretación, `INDETERMINADO`) — se hace
  explícita.

### Bloque: `build_inactive_bar_mask(full_grid, rows, inactive_evidence)`

- **Entrada**: la grilla esperada completa, las filas observadas, y la
  evidencia de `analyze_inactive_bar_candidates`.
- **Transformación**: para cada minuto esperado (recortado por archivo,
  igual que `compute_file_coverage`), asigna `status` (`AUSENTE`/
  `PRESENTE`) y, si está presente, una `category`: `VOLUMEN_CERO` (0
  casos, pero se mantiene en el catálogo), `CANDIDATO_FORWARD_FILL`
  (pertenece a una racha larga), `FLAT_AISLADA` (plana pero aislada) o
  `ACTIVA`. Vectorizado con `np.select` sobre máscaras booleanas (evitar
  `.apply` fila a fila sobre ~2 millones de filas).
- **Salida**: `data/interim/mnq/tda02_barra_inactiva_mask.parquet` — la
  máscara persistente que exige la tarea, reutilizable por etapas
  futuras.

### Bloque: `check_stop2(gaps_classified, threshold=0.90)`

- **Entrada**: el inventario de huecos ya clasificado.
- **Transformación**: entre los huecos "de calendario"
  (`DAILY_MAINTENANCE`, `WEEKEND`, `HOLIDAY`, `EARLY_CLOSE`), calcula qué
  fracción tiene `confidence == "ALTA"` (coincidencia EXACTA con el
  horario declarado). Si esa fracción cae por debajo del umbral (90 % por
  defecto), `triggered=True` — la sesión real derivada de los datos
  difiere sustancialmente de la esperada, y la tarea exige detenerse
  (`BLOCKED_STOP_2`) en vez de forzar el calendario a encajar.
- **Salida**: `{"triggered", "calendar_gap_count", "exact_match_fraction",
  "by_cause"}`.

### Bloque: `run_tda02_analysis(config)` — orquestador

Encadena todo lo anterior (protección del hold-out → carga de filas →
calendario → clasificación de huecos → bordes de archivo → cobertura →
barras fuera de grilla → DST → TH04 → STOP-2) y devuelve un
`TDA02Result` con cada pieza. No escribe ningún archivo — eso es
responsabilidad de `run_tda02.py`.

---

## 10. `ingest/run_tda02.py`

Punto de entrada de terminal. Además de cargar la configuración, llamar a
`run_tda02_analysis` y volcar cada pieza de `TDA02Result` a su artefacto
(CSV/parquet), dibuja el **único gráfico** que exige la tarea: el heatmap
de completitud día × minuto-del-día (`_draw_completeness_heatmap`) —
una matriz booleana `(n_días, 1440)`, fecha de Nueva York en las filas,
minuto del día (hora NY) en las columnas, marcando presente cualquier
minuto con al menos una barra observada en cualquier archivo. El objetivo
es puramente visual: distinguir huecos ESTRUCTURALES (bandas verticales
alineadas con el horario — el corte de mantenimiento diario, y el break
secundario visible sólo hasta 2021-06-25) de huecos ESPORÁDICOS (puntos
aislados). No reemplaza el inventario numérico de huecos.

Como TDA-00/TDA-01, captura `HoldoutIsolationError` y la traduce a un
mensaje `FAIL` legible; si `STOP-2` se dispara, termina con código de
salida `2` (`BLOCKED_STOP_2`) en vez de `0`.

**Reproducir la ejecución**:

```bash
python -m ohlcv_dataroad.ingest.run_tda02 --config configs/mnq_snapshot.yaml
```

---

## 11. `ingest/tda03_rolls.py`

### Qué problema resuelve

Cada uno de los 22 archivos del conjunto de investigación es UN contrato
trimestral distinto (TDA-01, §8). TDA-03 decide, para cada fecha, cuál de
esos contratos es el "activo" — de forma **causal** (nunca usando
información posterior al instante de la decisión) y **trazable** (nunca
mezclando ni promediando dos contratos, nunca borrando una fila sin dejar
rastro de por qué se descartó). El resultado es una única serie con
exactamente una fila por minuto, lista para que TDA-04 calcule retornos
sin que ninguno de ellos cruce, sin darse cuenta, la frontera entre dos
instrumentos distintos.

### Bloque: `parse_contract_label(source_file)`

Traduce `"19_mnq_12_24.Last.txt"` → `"Z24"`, usando el código de mes de
vencimiento CME/Globex (`03→H, 06→M, 09→U, 12→Z`, convención de industria
ya documentada en TDA-01 — no inventada aquí).

### Bloque: `RollPolicy` / `load_roll_policy(config)`

Los tres parámetros de la política de rollover (`min_incoming_share_shared`,
`confirmation_sessions_required`, `extreme_jump_top_n`) se leen de
`configs/mnq_snapshot.yaml`, sección `tda03` — **ninguno vive escondido en
el código**. El YAML documenta, para cada uno, por qué se eligió ese valor
y por qué NO se heredó el parámetro histórico calibrado bajo la ventana
`04:30–16:00` (`docs/instruments/mnq/MNQ_DATA_PRIOR_KNOWLEDGE.md`, §7.2).

### Bloque: `attach_trading_date_and_contract(rows, schedule)`

Reutiliza el `SessionSchedule` de TDA-02 (`session_calendar.py`) para
asignar la `trading_date` de cada fila y `parse_contract_label` para su
etiqueta de contrato — ninguna lógica de calendario se reimplementa aquí.

### Bloque: `compute_overlap_daily_evidence(...)` — la evidencia que decide el roll

- **Entrada**: filas del contrato saliente y entrante, y las fechas donde
  ambos tienen barras (`overlap_dates`).
- **Transformación**: por cada fecha, calcula el volumen de cada contrato
  SOLO sobre los minutos que AMBOS comparten (`shared_minutes`) — política
  heredada (regla 5, §7.1 de `MNQ_DATA_PRIOR_KNOWLEDGE.md`): comparar
  volúmenes de tramos horarios distintos no mide traspaso de liquidez. La
  columna que decide el roll es `share_shared` = fracción de ese volumen
  compartido que es del ENTRANTE. Se reporta también `share_total` (sobre
  el volumen del día completo) — pero **nunca se usa para decidir**: puede
  divergir mucho de `share_shared` cuando uno de los dos contratos tiene
  cobertura reducida ese día (ejemplo real verificado en el informe,
  `2025-03-17`: `share_shared=40%` pero `share_total=90%`, porque el
  saliente sólo tuvo 300 de 1.380 barras esperadas esa jornada).
- **Salida**: una fila por `trading_date` del solapamiento, con volumen
  total/compartido de cada contrato, `share_shared`, `share_total` y
  cobertura respecto de la grilla de TDA-02.

### Bloque: `determine_overlap_rollover(daily_evidence, combined_dates, policy)` — la regla CAUSAL

**Prueba de causalidad explícita** (sección 4 de la tarea, "test de
reconstrucción"): la señal del día `d` se calcula únicamente con
`daily_evidence` de `d` y de días *anteriores* — la función recorre las
fechas en orden cronológico y nunca mira una fila posterior a la que está
evaluando. Verificado con un test dedicado
(`test_rollover_decision_is_causal_truncated_future_does_not_change_past_decision`):
recalcular la señal usando SOLO los datos hasta el día de la señal da
exactamente la misma fecha que usando el conjunto completo.

```python
consecutive = 0
signal_date = None
for _, row in daily_evidence.iterrows():          # orden cronologico
    if row["share_shared"] >= policy.min_incoming_share_shared:
        consecutive += 1
    else:
        consecutive = 0
    if consecutive >= policy.confirmation_sessions_required:
        signal_date = row["trading_date"]
        break
```

- **Umbral (`min_incoming_share_shared=0.50`)**: "cruce de dominancia" —
  el entrante pasa a tener MÁS volumen que el saliente. Es el umbral más
  simple posible, sin calibrar: no se copió el 55 % heredado (calibrado
  bajo una ventana que ya no aplica). Sobre los datos reales, este umbral
  reproduce EXACTAMENTE las mismas fechas de roll que el algoritmo
  heredado (calibrado de forma independiente, bajo otra ventana y otro
  umbral) — ver el informe, sección de validación.
- **Confirmación (`confirmation_sessions_required=1`)**: basta una sola
  jornada (regla 7 heredada). Justificado empíricamente: en las 2
  transiciones con solapamiento del conjunto de investigación, el share
  del entrante crece de forma MONÓTONA una vez cruza el umbral — exigir 2
  confirmaciones habría dado exactamente la misma fecha efectiva.
- **`effective_date` = la fecha siguiente OBSERVADA en `combined_dates`**
  (unión de todas las fechas del saliente y del entrante, no sólo el
  solapamiento — necesario para el caso borde en que la señal cae en el
  último día del solapamiento). Regla 8 heredada: la señal se conoce al
  TERMINAR la jornada de la señal; aplicarla ese mismo día usaría
  información que, en el instante de negociación, aún no se conocía
  completa.
- **Si ninguna jornada cruza el umbral**: la transición se fuerza al
  terminar el solapamiento (el archivo saliente simplemente deja de tener
  datos) — `confidence="BAJA"`. No ocurre en el conjunto de investigación
  real, pero el mecanismo existe y está testeado
  (`test_determine_overlap_rollover_crosses_on_dominance_day`, que usa una
  fixture donde la señal cae en el último día disponible).

### Bloque: `compute_basis_evolution(...)` / `compute_no_overlap_evidence(...)`

- **Con solapamiento**: el basis (diferencia y ratio de precio) se mide
  ÚNICAMENTE en pares de barras con el MISMO timestamp exacto de ambos
  contratos (`merge` por `ts_utc`) — nunca el último precio de uno contra
  el primero del otro en instantes distintos, que mezclaría basis con
  movimiento genuino de mercado.
- **Sin solapamiento** (19 de las 21 transiciones): no existe ningún par
  de precios simultáneos. `compute_no_overlap_evidence` documenta el
  último bar del saliente y el primero del entrante, con
  `confidence="BAJA"` **siempre** — la función ni siquiera calcula un
  "basis": sólo una diferencia aparente, explícitamente advertida como no
  atribuible en su totalidad al roll.

### Bloque: `build_active_contract_calendar(...)` — irreversibilidad + regla de respaldo

Decide, por `trading_date`, qué archivo está activo:

1. Por defecto, cada archivo "posee" sus propias fechas (correcto para
   las 19 transiciones sin solapamiento, donde nunca hay ambigüedad).
2. Dentro de una ventana de solapamiento: antes de `effective_date`, el
   saliente (`PRE_CROSSOVER_OUTGOING_ACTIVE`); desde ella, el entrante
   (`POST_CROSSOVER_INCOMING_ACTIVE`). Como `effective_date` sólo avanza
   (nunca se recalcula hacia atrás una vez fijada por archivo), la
   irreversibilidad (regla 6 heredada) es una consecuencia directa del
   diseño, no una comprobación aparte — verificado igualmente con un test
   dedicado (`test_active_calendar_is_pre_crossover_then_post_crossover_never_reverts`).
3. **Regla de respaldo** (generalización de la regla 11 heredada, §7.4 de
   `MNQ_DATA_PRIOR_KNOWLEDGE.md`): si el contrato formalmente activo tiene
   CERO barras una fecha mientras el otro sí tiene, se usa la cobertura
   REAL del otro **sólo esa fecha**, sin adelantar ni atrasar el cruce
   formal. Sobre el conjunto de investigación real, esta regla no se
   activa ninguna vez (verificado: `reason.value_counts()` no tiene
   ninguna fila `ZERO_BARS_FALLBACK`) — se mantiene porque es la lógica
   correcta y general, probada con datos sintéticos
   (`test_zero_bar_fallback_uses_incoming_without_advancing_formal_crossover`),
   no como predicción de que vaya a necesitarse.

### Bloque: `build_canonical_series(rows, active_calendar)`

Conserva una fila SÓLO si su `source_file` coincide con el archivo activo
de su `trading_date`; descarta el resto CON motivo trazado
(`NON_ACTIVE_CONTRACT_ON_OVERLAP_DATE` o `OUT_OF_GRID_NO_TRADING_DATE`).
**Verificación de conservación bloqueante** (regla 10-11 heredada): un
`assert len(rows) == len(canonical) + len(discarded)` — igual que TDA-00
aborta si su propia verificación falla. También comprueba, con
`assert`, que la serie resultante es monótona y sin duplicados: como cada
`trading_date` tiene un único archivo activo y las `trading_date` son
disjuntas por construcción (TDA-02), la unicidad temporal es una
consecuencia del diseño, verificada aquí explícitamente.

### Bloque: `compute_adjustment_factors(...)` / `apply_adjustments(...)` — ajuste evaluado, no forzado

**El hallazgo central de esta etapa**: el basis sólo es medible en una
transición con solapamiento (necesita precios simultáneos). De las 21
transiciones, sólo 2 lo son. Por tanto, un factor de ajuste retrospectivo
sólo puede encadenarse de forma defendible a lo largo de un tramo
CONTIGUO de transiciones con solapamiento — en este conjunto de
investigación, exactamente los 3 contratos más recientes (`Z24`, `H25`,
`M25`). El resto de la historia (`H20`...`U24`) queda **sin factor**
(`NaN`), explícitamente, en vez de forzar una cifra no defendible sobre un
salto que no se puede separar de movimiento genuino de mercado. Esto es
la instancia concreta, con datos reales, del principio que pide la
sección 8 de la tarea: si la evidencia no permite declarar un único
método superior en todo el rango, no se fuerza.

Los factores se anclan en el contrato MÁS RECIENTE de la cadena
(`ratio_factor=1.0`, `diff_factor=0.0`) y se propagan hacia atrás,
acumulando el basis de cada roll — la propiedad que cita el fundamento
Tsay: "el factor de ajuste de un segmento depende de rolls posteriores,
pero es constante DENTRO del segmento". `apply_adjustments` añade 8
columnas nuevas (`open/high/low/close_adj_ratio` y `..._adj_diff`) **sin
tocar** las columnas OHLCV crudas.

### Bloque: `build_roll_mask(canonical, transitions_df)`

La máscara persistente que reutilizará TDA-04: `is_roll_boundary=True` en
la primera fila de cada nuevo `segment_id` (salvo la primera de toda la
serie). TDA-04 la usará para forzar `r_t = NaN` en esa barra — el retorno
hacia la barra anterior cruzaría la frontera de contrato.

### Bloque: `find_extreme_jumps(...)` / `check_stop3(...)` — STOP-3, diseño explicado

**Un diseño descartado, documentado para que no se reintente sin motivo**:
la primera versión de esta función usaba un umbral fijo de "K veces la
MAD (desviación absoluta mediana) global de saltos de 1 minuto". Sobre los
datos reales, ese diseño marcó **26.225** barras como "candidatas" — el
precio de MNQ cambia de nivel en un factor ~3x a lo largo del conjunto de
investigación (de ~7.000 a ~22.000 puntos) y su volatilidad tiene
regímenes muy distintos, así que un umbral global no es apropiado para
una serie tan poco estacionaria.

**Diseño adoptado**: un ranking ACOTADO de las `top_n` (40 por defecto)
mayores discontinuidades RELATIVAS (`|ΔC|/C_{t-1}`, no en puntos
absolutos, para que el ranking no lo dominen trivialmente los años con el
precio más alto), sólo entre barras consecutivas de 1 minuto del MISMO
contrato (una barra que cruza una frontera de roll no es candidata: ya
está explicada, es el roll). Para cada una se añade contexto de revisión:
el volumen de esa barra y si el precio REVIERTE más de la mitad del salto
en la barra siguiente (firma clásica de un tick aislado que "se corrige"
solo). `check_stop3` sólo dispara si existe al menos un salto
`suspicious` = revierte Y ocurre en volumen bajo (percentil 5 o menor) —
el tamaño del salto por sí solo NO dispara STOP-3: un salto grande en
volumen alto, sin reversión, es la firma esperada de un evento de mercado
genuino, no de un problema de datos.

Sobre el conjunto de investigación real, los 40 candidatos se concentran
en dos patrones reconocibles: (a) saltos a las `12:31`/`13:31` UTC — las
`08:31` hora de Nueva York, un minuto después de la hora estándar de
publicaciones macroeconómicas de EE. UU. (IPC, nóminas, etc.); (b) la
semana de crisis de marzo de 2020 y la semana de shock arancelario de
abril de 2025. Ninguno revierte en volumen bajo — `STOP-3` no se activa.

### Bloque: `build_invariance_table()`

Clasificación puramente MATEMÁTICA (no calcula nada sobre los datos): para
cada estadístico que el roadmap prevé usar en etapas posteriores, si su
valor NUMÉRICO cambia al multiplicar un segmento por una constante (ratio)
o al sumarle una constante (aditivo). Ver el informe
(`TDA03_rolls_serie_continua.md`, tabla de invariancia) para la
justificación fila por fila; el resumen: retornos (simples y log), rangos
relativos, y cualquier estadístico derivado de ellos (varianza, cuantiles,
ACF) son invariantes al RATIO; diferencias en puntos, rangos absolutos y
medidas en ticks son invariantes al ADITIVO; el nivel de precio no es
invariante a ninguno de los dos.

### Bloque: `run_tda03_analysis(config)` — orquestador

Encadena todo lo anterior (protección del hold-out → carga de filas →
calendario → transiciones → calendario de contrato activo → serie
canónica → factores de ajuste → máscara de roll → STOP-3 → tabla de
invariancia). No escribe ningún archivo — eso es responsabilidad de
`run_tda03.py`.

---

## 12. `ingest/run_tda03.py`

Punto de entrada de terminal, mismo espíritu que TDA-00/01/02: carga la
configuración, llama a `run_tda03_analysis`, vuelca cada pieza a su
artefacto (CSV/parquet) e imprime un resumen. Si `STOP-3` se dispara,
termina con código de salida `3` (`BLOCKED_STOP_3`) en vez de `0` — y,
antes de eso, ya escribió todos los artefactos de evidencia (para que la
discontinuidad quede documentada, no escondida).

**Reproducir la ejecución**:

```bash
python -m ohlcv_dataroad.ingest.run_tda03 --config configs/mnq_snapshot.yaml
```

---

## 13. `ingest/tda04_analysis_variables.py`

### Qué problema resuelve

TDA-03 dejó una serie con exactamente un contrato activo por timestamp,
pero todavía **una tabla de precios**, no un objeto analizable
estadísticamente. TDA-04 construye el ladrillo mínimo — el retorno
logarítmico de 1 minuto — y, sobre todo, decide con precisión **cuándo
NO existe** un retorno válido: cada fila consecutiva del `DataFrame` no
es automáticamente un retorno de 1 minuto real.

### Por qué una fila consecutiva del `DataFrame` no basta

La serie canónica está ordenada por `timestamp` y no tiene duplicados, así
que `df["close"].shift(1)` **siempre** da un valor — nunca lanza un error.
El problema es que esa fila anterior puede ser de:
- **otra jornada de negociación** (tras el corte de mantenimiento
  nocturno, un fin de semana o un feriado — TDA-02 ya inventarió estos
  huecos);
- **otro contrato** (justo después de un roll — TDA-03 ya marcó estas
  fronteras);
- **un minuto no contiguo** dentro del mismo día (un hueco corto interno,
  también ya inventariado por TDA-02).

Por eso `pandas.Series.pct_change()` (o `np.log(df["close"]).diff()`)
aplicado ciegamente sobre toda la tabla **es incorrecto**: esas funciones
calculan exactamente "contra la fila anterior", sin ninguna de las tres
comprobaciones de arriba. TDA-04 nunca las usa directamente sobre la
tabla completa — las aplica solo *después* de filtrar por
`r_1m_valid` (`build_return_validity_mask`, más abajo).

### Por qué `trading_date` y fecha de calendario no son lo mismo

Una sesión de CME puede empezar la tarde/noche de un día de calendario y
cerrar al día siguiente (p. ej., abre domingo 18:00 NY y cierra lunes
17:00 NY). TDA-02 ya resolvió esto asignando a cada barra un
`trading_date` = la fecha en la que su sesión CIERRA, no la fecha de
calendario de su propio timestamp. Dos barras a las `23:59` y `00:00`
hora de Nueva York pueden pertenecer al MISMO `trading_date` (mismo cierre
de sesión) y su retorno es perfectamente válido; dos barras del mismo día
de calendario pero de dos sesiones distintas (rarísimo, pero posible en
teoría) no lo serían. TDA-04 compara siempre `trading_date`, nunca la
fecha del timestamp.

### Qué es un log-retorno, y por qué Close→Close

`r_1m = ln(C_t/C_{t-1})` es, aproximadamente, el cambio porcentual de
precio entre dos barras — pero con una propiedad que el retorno simple no
tiene: los log-retornos de una secuencia **válida** se SUMAN
(`r_t + r_{t+1} = ln(C_{t+1}/C_{t-1})`, verificado con un test dedicado),
lo que los hace mucho más cómodos para agregar en el tiempo. Se usa
Close→Close (no Open, no ningún precio "intermedio") porque TDA-01 (§5)
ya estableció que, en este formato de datos, los 5 campos de la barra
`t` — incluido `Close` — están disponibles **conjuntamente** en `t`, no
de forma escalonada: no hay ninguna razón para preferir otro precio, y
Close→Close es además la definición estándar del fundamento Tsay de esta
etapa.

### Bloque: `load_canonical_and_mask(config)`

Lee `tda03_serie_continua.parquet` y `tda03_roll_mask.parquet` — los
MISMOS artefactos que TDA-03 ya declaró en `configs/mnq_snapshot.yaml`
(sección `tda03`), no una copia. TDA-04 nunca abre ningún archivo de
`data/raw/`: su única entrada son estos dos parquet.

### Bloque: `RollConsistencyError` + validación bloqueante previa

Antes de clasificar nada, `build_return_validity_mask` **comprueba**
(no asume) que, salvo la primera observación, un cambio de `segment_id`
respecto de la fila anterior coincide EXACTAMENTE con un cambio de
`contract` y con `roll_mask["is_roll_boundary"]=True` — las tres cosas
deben describir la misma frontera, tal como TDA-03 garantiza por
construcción (`build_canonical_series`/`build_roll_mask`, § 11). Si
alguna fila contradice esa invariante, se lanza `RollConsistencyError`
de inmediato, con las primeras filas afectadas en el mensaje — TDA-04
aborta en vez de construir retornos (o "no-retornos") sobre una serie
canónica y una máscara de roll que ya no coinciden entre sí. Verificado
con 4 tests dedicados: cambio de contrato con `is_roll_boundary=True` es
el caso normal (no lanza nada); cambio de contrato con
`is_roll_boundary=False`, y `is_roll_boundary=True` sin ningún cambio de
contrato, lanzan ambos `RollConsistencyError`; y, sobre los artefactos
REALES de TDA-03 (`tests/test_tda04_analysis_variables.py::test_D_invariant_holds_on_real_tda03_artifacts_for_all_21_boundaries`),
la invariante se cumple exactamente para las 21 fronteras de roll del
conjunto de investigación, sin excepción.

### Bloque: `build_return_validity_mask(canonical, roll_mask)` — el núcleo de la etapa

Para cada fila, compara contra la fila **inmediatamente anterior** de la
tabla (`shift(1)`) y evalúa, EN ESTE ORDEN DE PRIORIDAD (una fila puede
fallar varias condiciones a la vez; el orden decide cuál se reporta como
`invalid_reason`; las 4 también quedan como banderas booleanas
independientes, para medir el solapamiento entre causas):

1. **`FIRST_OBSERVATION`** — no hay ninguna fila anterior en toda la
   serie (la primera barra del conjunto de investigación). Un único caso.
2. **`ROLL_BOUNDARY`** — `roll_mask["is_roll_boundary"]` es `True` para
   esta fila. Se usa la mascara de TDA-03 **directamente**, no se
   recalcula a partir de `segment_id` — reutiliza la fuente de verdad de
   esa etapa, ya con la validación bloqueante de arriba garantizando que
   ambas formas de mirarlo coinciden siempre.
3. **`TRADING_DATE_BOUNDARY`** — `trading_date` cambió respecto de la
   fila anterior, sin ser un roll: cualquier frontera de sesión de TDA-02
   (corte de mantenimiento, fin de semana, feriado, cierre anticipado),
   sea cual sea su causa de calendario.
4. **`NON_CONSECUTIVE_MINUTE`** — mismo `trading_date`, mismo
   `segment_id`, pero la diferencia de timestamp NO es exactamente 60
   segundos: un hueco corto interno de TDA-02 (p. ej. el break secundario
   pre-2021-06-25) dentro de la misma jornada.

Si ninguna aplica, `r_1m_valid=True`. La función también arma las
columnas de trazabilidad completas que exige la tarea:
`prev_timestamp`, `delta_minutes`, `prev_trading_date`,
`prev_segment_id`, `prev_contract` — todo lo necesario para auditar
CUALQUIER fila sin tener que releer la serie canónica.

### Bloque: `build_analysis_variables(canonical, validity)`

Construye las 8 columnas de variables (ver la tabla completa en
`reports/mnq/TDA04_variables_analisis.md`): `r_1m`, `R_1m` (retorno
simple, para TH07), `abs_r_1m`, `r2_1m`, `zero_1m` (indicador de
`r_1m==0`), `log_hl`, `log_co` (estas dos últimas NUNCA dependen de una
barra anterior — no tienen ninguna regla de no-cruce que aplicar, TDA-00
ya garantiza que siempre son finitas) y `log_oc_prev` (el "gap" del
roadmap: usa `open_t` Y `close_{t-1}`, así que sigue **exactamente** la
misma regla de validez que `r_1m` — mismas dos barras). Todas las
columnas derivadas de `r_1m` (`abs_r_1m`, `r2_1m`, `zero_1m`) heredan su
`NaN` por propagación aritmética simple, sin necesitar lógica propia.

### Bloque: `audit_losses_by_cause(validity)` / `audit_th07_r_vs_R(variables)`

El primero cuenta cuántas filas caen en cada causa (categoría exclusiva,
suma exactamente el total) y en cada bandera independiente (para medir
solapamiento — p. ej., cuántas filas son a la vez `is_roll_boundary` y
`is_trading_date_boundary`, que por construcción de TDA-03 debería
coincidir exactamente con el total de rolls: todo roll ocurre en una
frontera de jornada, nunca a mitad de una). El segundo resuelve el
método mínimo de TDA-04 para **TH07** (`|r_1m - R_1m|`, global y por
decil de `|r_1m|` — ver el informe para el resultado).

### Bloque: `run_tda04_analysis(config)` — orquestador

Encadena todo lo anterior. Reutiliza `holdout_guard.py` de forma
defensiva: TDA-04 no abre ningún archivo crudo (su única entrada son los
parquet de TDA-03), pero se revalida la disyunción `research`/`holdout` y
que ningún timestamp de la serie canónica alcance la frontera del
hold-out, por consistencia con el resto del pipeline y para que un
cambio futuro en la serie canónica no pueda colar silenciosamente una
fila del hold-out sin que ninguna etapa lo note.

---

## 14. `ingest/run_tda04.py`

Punto de entrada de terminal, mismo espíritu que TDA-00/01/02/03: carga
la configuración, llama a `run_tda04_analysis`, vuelca cada pieza a su
artefacto (parquet/CSV) e imprime el resumen de auditoría completo
(retenidos/perdidos, pérdida por causa, comprobaciones de sanidad, TH07).

**Reproducir la ejecución**:

```bash
python -m ohlcv_dataroad.ingest.run_tda04 --config configs/mnq_snapshot.yaml
```

---

## 15. `ingest/tda05_effective_resolution.py`

### Qué problema resuelve

MNQ solo puede cotizar en múltiplos de **0.25 puntos** (un *tick* — la
especificación externa del contrato, no algo inferido de los datos). Esto
significa que el precio se mueve en **escalones**, no de forma continua.
TDA-05 mide en qué medida esa discreción del precio hace que el retorno
de 1 minuto `r_1m` (TDA-04) sea, en la práctica, una variable con pocos
valores posibles — y cómo cambia eso por hora del día y por año.

### Qué es un tick, y por qué el retorno puede parecer decimal aunque el precio no lo sea

Un tick es el escalón mínimo de precio: dos Close consecutivos SIEMPRE
difieren en un múltiplo entero de 0.25 (TDA-00 ya certificó que cada
precio individual cae en esa grilla). Pero `r_1m = ln(C_t/C_{t-1})` es el
LOGARITMO de un COCIENTE de dos números en esa grilla — y el cociente de
dos múltiplos de 0.25 no tiene por qué ser, él mismo, un número "redondo".
Por eso `r_1m` toma miles de valores decimales distintos en la práctica,
aunque el movimiento de precio subyacente sea siempre el mismo puñado de
saltos enteros de tick — es la ilusión de continuidad que esta etapa
desenmascara comparando `r_1m` contra `delta_close_ticks` directamente.

### Qué significa "variable efectivamente discreta"

Que, dentro de un segmento, el retorno toma pocos valores posibles: una
fracción grande de barras no se mueve nada (0 ticks), y la mayoría de las
que sí se mueven lo hacen en el escalón mínimo (±1 tick). Cuando esto
ocurre, los momentos, cuantiles, histogramas y QQ-plots de TDA-07 en
adelante **no** describen una distribución continua: describen una
variable casi categórica.

### Por qué no se puede dividir 0.25 puntos entre `std(r_1m)` directamente

`tick_size` está en PUNTOS; `r_1m` es ADIMENSIONAL (un logaritmo de
cociente). Dividir uno entre el otro mezclaría unidades incompatibles —
el resultado no significaría nada. TDA-05 calcula DOS cocientes, cada uno
dimensionalmente coherente por separado:

- `tick_to_sigma_points = tick_size / std(delta_close_points)` — ambos en
  PUNTOS. Responde directamente "¿cuán grande es 1 tick frente al
  movimiento típico de 1 minuto?".
- `tick_to_sigma_return = tick_return_representative / std(r_1m)` — ambos
  ADIMENSIONALES, donde `tick_return_representative =
  ln((C_repr+tick)/C_repr)` usa el Close MEDIANO del grupo como nivel de
  precio representativo (el equivalente de un tick en unidades de retorno
  depende del nivel de precio, que cambia con el tiempo — no es una
  constante).

Sobre el conjunto de investigación real, ambos cocientes llevan a la
MISMA lectura práctica (ver el informe, tabla global) — se calculan por
separado precisamente para poder demostrarlo, no para elegir uno de los
dos a ciegas.

### Por qué una alta proporción de ceros importa, y por qué no basta con el promedio global

Una fracción alta de `r_1m = 0` en un segmento indica que ese segmento es
efectivamente discreto — momentos y cuantiles calculados ahí no describen
una distribución continua. Mirar solo el promedio GLOBAL puede ocultar
que, de noche, esa fracción es dominante, mientras que de día es
marginal (TDA-05 lo desagrega explícitamente por hora NY y por año, nunca
solo una cifra agregada — exactamente el riesgo que cita el roadmap para
esta etapa).

### Diferencia entre el análisis por HORA de TDA-05 y el perfil MINUTO-A-MINUTO de TDA-06

TDA-05 agrupa en 24 baldes (`hour_ny` = 0..23, la hora NY literal de cada
barra) para medir la discreción — **no** decide todavía ninguna
segmentación de sesión (RTH, pre-market, etc.) ni estudia la forma fina
del perfil intradía. Eso es TDA-06, que mira el perfil **minuto por
minuto** (1.440 puntos, no 24) para encontrar la forma real del patrón
determinista de mercado, incluida cualquier segmentación que emerja de
los propios datos.

### Bloque: `load_inputs(config)`

Lee `tda04_variables_1m.parquet`, `tda04_return_validity_mask.parquet` y
`tda02_barra_inactiva_mask.parquet` — los mismos artefactos que TDA-04 y
TDA-02 ya declararon, sin redeclararlos. TDA-05 nunca abre ningún archivo
de `data/raw/`.

### Bloque: `compute_tick_variables(variables, validity, tick_size)`

- `delta_close_points = close_t - close_{t-1}`, calculado con
  `close.shift(1)` sobre la MISMA tabla ordenada que usa TDA-04, y
  aplicando EXPLÍCITAMENTE `validity["r_1m_valid"]` — nunca
  `close.shift(1)` a secas (la razón es idéntica a por qué TDA-04 nunca
  usa `pct_change()` ciegamente: la fila anterior en la tabla puede ser
  de otra jornada, otro contrato o estar separada por un hueco).
- `delta_close_ticks = delta_close_points / tick_size`.
- **Invariante verificada, no asumida**: para toda fila válida,
  `delta_close_ticks` debe ser entero dentro de `TICK_GRID_TOLERANCE`
  (la misma tolerancia que usa TDA-00 para la grilla de un precio
  individual). Si no lo es, se lanza `TickGridInconsistencyError` de
  inmediato — nunca se redondea en silencio.
- `hour_ny`/`year_ny`: se localiza el timestamp (tz-naive, UTC por
  construcción) como UTC y se convierte a `America/New_York` con
  `zoneinfo` — el mismo mecanismo DST-aware de TDA-01/02/03, nunca un
  offset fijo. Ambos se derivan de la MISMA conversión, para que la
  agrupación por hora y por año sean consistentes entre sí.

### Bloque: `attach_forward_fill_flags(tick_df, inactive_mask)` / `compare_with_without_forward_fill(...)`

Une, por `(source_file, timestamp UTC)`, la categoría exacta que TDA-02
asignó a cada barra (`ACTIVA` / `FLAT_AISLADA` / `CANDIDATO_FORWARD_FILL`
— nunca una categoría "FORWARD_FILL confirmada" inventada). Sobre el
conjunto de investigación real, `n_confirmed_forward_fill == 0` (TDA-02,
§10.2, ya lo estableció): la comparación "con/sin FORWARD_FILL" que pide
el roadmap es, por construcción, la MISMA cifra dos veces — se declara
así explícitamente, verificado con un test dedicado
(`test_forward_fill_reports_explicit_declaration_when_none_confirmed`).
Aparte, y etiquetada `SENSITIVITY_ONLY`, se calcula la misma cifra
excluyendo `CANDIDATO_FORWARD_FILL` (barras candidatas, nunca
confirmadas) — solo para mostrar que esa exclusión no cambia el resultado
de forma material; nunca participa en la decisión de STOP-5.

### Bloque: `summarize_resolution(df, tick_size)` — el conjunto completo de indicadores

Reutilizada por las cuatro tablas de salida (global, por hora, por año,
año×hora): calcula las 17 cifras de la sección 10 de la tarea sobre
**exactamente las mismas filas válidas** (mismo `n` para todas las
columnas de una fila de salida). Casos de división por cero (`sigma=0` o
`median_range=0`, posibles en un grupo degenerado) dan `+inf` EXPLÍCITO,
nunca una excepción ni un `NaN` mudo — `n` queda siempre disponible para
juzgar si ese `inf` es significativo.

### Bloque: `block_bootstrap_global_metrics(tick_df, tick_size)`

Intervalo de incertidumbre por bootstrap de BLOQUES (jornadas completas
con reemplazo, no filas individuales — preserva la dependencia intra-día,
sección 12 de la tarea/G5), calculado sobre arrays de NumPy
precalculados (no reconstruyendo un `DataFrame` en cada repetición) por
razones de rendimiento: con ~1,9 millones de filas y ~1.420 jornadas, la
versión ingenua con `pandas.concat` en cada repetición resultó
impracticamente lenta. El resultado estadístico es idéntico; solo cambia
la velocidad. Solo se calculan intervalos para las DOS métricas globales
principales (`zero_fraction`, `tick_to_sigma_points`) — no para cada fila
de las tablas por hora/año (la propia tarea señala que no aporta
información suficiente para justificar el costo).

### Bloque: `build_stop5_watchlist(by_hour, by_year, by_year_hour)`

Señala, de forma PURAMENTE DESCRIPTIVA (umbrales laxos, no un criterio de
exclusión), qué horas, años y combinaciones año×hora merecen lectura atenta
en la evaluación de STOP-5 — nunca decide la exclusión por sí sola: el roadmap
prohíbe explícitamente un corte automático del tipo "tick/sigma > X => excluir".
La decisión de STOP-5 se toma leyendo la evidencia completa (§ informe),
no aplicando este watchlist como regla.

### Bloque: `run_tda05_analysis(config)` — orquestador

Encadena todo lo anterior. Reutiliza `holdout_guard.py` de forma
defensiva (TDA-05 no abre ningún archivo crudo; su entrada son los
parquet de TDA-02/TDA-04).

---

## 17. `ingest/tda06_intraday_calendar_profile.py`

### Qué problema resuelve

Antes de buscar cualquier dependencia estocástica (ACF, clustering de
volatilidad, dependencia en media — TDA-08/TDA-09), hay que aislar el
componente **determinista** ligado al reloj: ¿una hora concreta es
sistemáticamente más activa/volátil que otra, año tras año? Si es así, un
pico de ACF en el lag de una jornada, o un ARCH aparente, pueden ser
puramente el reloj repitiéndose — no dependencia real. TDA-06 mide ese
patrón (TH14) y el efecto de día de la semana (TH15), y decide si hace
falta un factor de ajuste estacional `s(m)` antes de seguir.

### Qué es `minute_of_day`, y por qué America/New_York

`minute_of_day = hour_ny * 60 + minute_ny`, con dominio 0..1439 — un
número por cada minuto del reloj de Nueva York. Se calcula igual que
`hour_ny` en TDA-05: el timestamp (tz-naive, UTC por construcción) se
localiza como UTC y se convierte con `zoneinfo` — nunca un offset fijo,
para que las transiciones de horario de verano (DST) queden bien
resueltas a ambos lados.

### Por qué `weekday` se calcula sobre `trading_date`, no sobre la fecha local de la barra

Una barra del domingo por la noche (hora NY) puede pertenecer a la
`trading_date` del LUNES — la sesión que abre el domingo cierra, y se
etiqueta, el lunes (TDA-02/TDA-03). Si el día de la semana se calculara
sobre la fecha de calendario local del timestamp, esa barra se contaría
como "domingo" — fabricando un falso "efecto domingo" que en realidad es
la apertura del lunes. Por eso `weekday = trading_date.weekday()`
siempre, nunca la fecha local del timestamp.

### El motor genérico: pivote (jornada × columna) + bootstrap de bloques

Cada jornada (`trading_date`) tiene, como mucho, UNA barra por
`minute_of_day` (la serie de TDA-03/04 no tiene timestamps duplicados) —
así que "una fila por jornada, una columna por minuto" es una matriz
densa sin pérdida de información (`_build_date_minute_pivot`). Sobre esa
matriz:

- El perfil PUNTUAL de un minuto es `nanmedian`/`nanmean` de su columna
  (ignorando `NaN` = jornadas sin barra en ese minuto).
- La banda de incertidumbre es bootstrap de BLOQUES: se remuestrean FILAS
  completas (jornadas enteras, con reemplazo) — preserva la dependencia
  intra-día, igual que TDA-05 (G5) — nunca barras individuales como si
  fueran independientes. Vectorizado con NumPy puro (sin reconstruir un
  `DataFrame` por repetición), la misma lección de rendimiento de TDA-05.
- Un minuto sin NINGUNA barra en NINGÚN día (estructuralmente cerrado —
  p.ej. el mantenimiento diario) queda con `n=0`, `point=NaN`: nunca se
  inventa una observación donde el mercado no operaba.

El mismo motor se reutiliza para el perfil por día de semana
(`build_weekday_profile`), cambiando "columna = minuto" por
"columna = weekday (0..6)", agregando primero dentro de cada
`(jornada, weekday)` para que cada jornada aporte un único valor resumen.

### Población/denominador por variable — decisión documentada, no oculta

- **Variables de RETORNO** (`r_1m`, `abs_r_1m`, `r2_1m`, `zero_1m`, la
  bandera de extremo): solo filas con `r_1m_valid=True` de TDA-04 — la
  MISMA máscara, nunca recalculada. Un retorno inválido nunca entra al
  perfil.
- **Variables de UNA SOLA BARRA** (`volume`, `log_hl` ≡ `rg_t`): TODAS
  las barras admisibles de la serie canónica, SIN restringir a
  `r_1m_valid` — ninguna de las dos depende de una barra anterior, así
  que exigirles esa máscara descartaría información perfectamente válida
  (p.ej. la primera barra de cada sesión, que nunca tiene `r_1m` válido
  pero sí tiene un volumen y un rango perfectamente medibles). El costo:
  su `n` por minuto es mayor y NO directamente comparable al de las
  variables de retorno — se reporta lado a lado en la tabla de salida,
  nunca oculto.

`rg_t ≡ log_hl = ln(H_t/L_t)`: TDA-04 ya lo calculó exactamente así
(roadmap, tabla de variables); TDA-06 lo reutiliza directamente, sin
inventar un nuevo estimador de rango.

### `r_1m`: media Y mediana, `abs_r_1m`/`r2_1m`/`log_hl`/`volume`: mediana

El roadmap permite "media (o mediana, más robusta)" como estimador
principal. Para las variables de MAGNITUD (`abs_r_1m`, `r2_1m`, `log_hl`,
`volume`) se usa la MEDIANA — robusta frente a colas, tal como pide la
tarea. Para `r_1m` (la pregunta de la MEDIA, TH14) se reportan AMBAS: la
media como estimador principal de esa pregunta específica, y la mediana
como versión robusta de contraste. `zero_1m` y la bandera de extremo son
indicadores 0/1: su "mediana" no es interpretable, así que se reporta su
MEDIA (= la proporción), igual que `zero_fraction` en TDA-05.

### La bandera de movimiento extremo — umbral RELATIVO, predeclarado

Un movimiento es "extremo" si `|r_1m|` supera el percentil 99 (1% más
grande) de `|r_1m|` sobre TODA la población de retornos válidos del
conjunto de investigación (`compute_extreme_flag`). Es explícitamente
RETROSPECTIVO (usa toda la muestra, G1) y relativo a la ESCALA del propio
dato — nunca un número de puntos fijo, que mezclaría escalas de precio
muy distintas entre 2019 y 2025 (TDA-05 ya estableció por qué). No es
EVT, no optimiza el umbral, no se prueban variantes.

### Segmentación derivada de los datos — método predeclarado, no ajustado a posteriori

1. **Proxies**: `abs_r_1m`, `log_hl`, `volume` (tres medidas
   independientes de "cuánto está pasando" en cada minuto).
2. **Suavizado**: mediana móvil CENTRADA de 15 minutos sobre cada perfil
   normalizado 0-1 (min-max) — quita el ruido minuto-a-minuto sin
   desplazar la ubicación aproximada de un quiebre real. **Corrección de
   cierre**: `_centered_rolling_median` usa `min_periods=1` para que un
   minuto ABIERTO cerca del borde de un hueco (p.ej. justo después de la
   ventana de mantenimiento) reciba igualmente una estimación suavizada
   — pero eso mismo, sin más, también le daba un valor FABRICADO a un
   minuto CERRADO (`n=0`) si su ventana llegaba a rozar un único vecino
   válido al otro lado del hueco: el minuto 17:54 NY (dentro de
   17:01-18:00, sin ninguna barra jamás) tomaba su valor casi
   íntegramente del primer minuto abierto tras las 18:00, generando un
   quiebre fabricado DENTRO del hueco. Se corrigió restaurando
   EXPLÍCITAMENTE `NaN` en toda posición donde el valor original (antes
   de suavizar) ya era `NaN` — nunca se interpola, nunca se hace
   forward-fill. Un minuto estructuralmente cerrado permanece `NaN`
   después del suavizado, siempre.
3. **Score compuesto**: promedio de los tres perfiles normalizados y
   suavizados (`compute_composite_activity_score`).
4. **Quiebres** (`detect_breakpoints`): se ordenan los 1439 valores de la
   primera diferencia absoluta del score de mayor a menor, y se aceptan
   en ese orden hasta un máximo de 6, exigiendo al menos 60 minutos de
   separación entre cortes ya aceptados — determinista, reproducible, sin
   ningún parámetro ajustado después de ver el resultado.
5. **Estabilidad entre años** (`check_breakpoint_stability`): el mismo
   algoritmo corre por separado en cada año COMPLETO (2020-2024); un
   corte del score GLOBAL se acepta en la propuesta final solo si un
   corte detectado independientemente en al menos 3 de esos 5 años cae a
   ≤15 minutos de él.

### Calibración G2 — `calibrate_breakpoint_detector`

Antes de confiar en el detector, se comprueba que NO fabrica quiebres
"estables" sobre ruido puro. **Corrección de cierre, dos problemas
separados**:

1. La primera versión permutaba el SCORE COMPUESTO ya suavizado y
   corría el detector directamente — se saltaba el paso de suavizado (y
   su interacción con los huecos) por completo, así que no calibraba el
   pipeline real. Corregido: cada surrogate permuta los valores VÁLIDOS
   del perfil CRUDO de cada proxy por separado
   (`_permute_preserving_nan_mask`, dejando fijas las posiciones `NaN`
   — el hueco estructural sigue en el mismo sitio, solo se destruye el
   orden temporal de los niveles válidos) y pasa por la MISMA
   `compute_composite_activity_score` (con el fix del suavizado) y el
   mismo `detect_breakpoints`.
2. Al aumentar el número de surrogates de 5 a 200 (barato: cada uno
   cuesta milisegundos), comparar cada surrogate contra los otros 199
   con el mismo `min_years=3` ABSOLUTO que se usa entre 5 años reales
   deja de ser una condición exigente — con 199 referencias, encontrar 3
   coincidencias por azar es casi seguro (verificado: producía 100% de
   "estabilidad" en el null, un resultado claramente roto). Corregido:
   se genera una piscina grande de 200 surrogates, pero el filtro de
   estabilidad se evalúa sobre 200 reagrupamientos aleatorios
   independientes de tamaño `len(COMPLETE_YEARS)` = 5 extraídos de esa
   piscina — el mismo tamaño de grupo y el mismo `min_years` que la
   comparación real entre años, repetida muchas veces.

Sobre el conjunto de investigación real, tras ambas correcciones: solo
**0,63%** de los eventos candidato-vs-grupo del null resultaron
"estables" — frente al **100% (6/6)** sobre los datos reales.

### STOP-6 — `evaluate_stop6` / `decide_stop6`

Para `abs_r_1m` y `log_hl` (las variables de magnitud): `relative_range`
mide cuánto varía el NIVEL del perfil entre minutos respecto de su nivel
típico (¿hay señal?); `spearman_by_year` mide cuánto se PARECE la FORMA
del perfil entre CADA año completo individual y el global (¿es estable?).
`RELATIVE_RANGE_REFERENCE`/`SPEARMAN_REFERENCE` (ambos 0.5) son puntos de
referencia DESCRIPTIVOS predeclarados — **corrección de cierre**:
`decide_stop6` ya no es una puerta binaria que mira solo la MEDIANA de
Spearman entre años; exige la referencia en el rango relativo Y en el
Spearman de **cada año individual** (no solo en la mediana agregada) para
ambas variables — una lectura más completa de la misma evidencia, sin
inventar un umbral nuevo. "Perfil fuerte y estable" (STOP-6 NO se activa)
exige esa condición para AMBAS variables de magnitud. Cualquier otro caso
activa STOP-6: no se construye `s(m)`, resultado negativo válido (G6).
Sobre el conjunto de investigación real, esta versión holística y una
comparación binaria simple sobre la mediana llegan a la MISMA conclusión
— el margen de la evidencia (rango relativo ~10× la referencia, Spearman
por encima de la referencia en los 5 años individuales, para ambas
variables) es tan amplio que ninguna forma razonable de agregarla la
revierte.

### `s(m)` — elección de proxy, normalización, protección

`choose_s_m_proxy` elige entre `abs_r_1m` y `log_hl` por su RUIDO como
estimador (`_mean_relative_ci_halfwidth`: ancho medio de la banda de
bootstrap relativo a su propio nivel, promediado por minuto) — **no** por
cuánto varía el nivel del perfil (esa es la pregunta de STOP-6, una
comparación distinta a propósito). Se prefiere la variable
proporcionalmente MENOS ruidosa (Tsay/roadmap C3: "si `rg_t` es
sustancialmente menos ruidoso, es el proxy preferido"), con empate a
favor de `log_hl` (población mayor, no depende de `r_1m_valid`).

`s(m) = point(m) / mean(point sobre minutos con datos)` — normalización
estándar de un índice estacional a media 1: `s(m)>1` = "minuto más
activo/volátil que el promedio del día". Protegida contra división por
cero: minutos con nivel `<=0` o no finito quedan con `s_m=NaN`, nunca
`inf`.

### `r_tilde` y la etiqueta `RETROSPECTIVO`

`r_tilde = r_1m / s(minute_of_day)` — `NaN` donde `r_1m` es inválido O
`s_m` no está definido, nunca `inf`. Se etiqueta explícitamente
`RETROSPECTIVO` (G1): `s(m)` se estimó usando TODA la muestra de
investigación, así que `r_tilde` no es una feature causal disponible
online sin volver a estimarse solo con datos hasta cada momento — su
versión causal, si alguna etapa posterior la necesita, se construye ahí,
no aquí.

### Bloque: `run_tda06_analysis(config)` — orquestador

Encadena todo lo anterior. Reutiliza `holdout_guard.py` de forma
defensiva (TDA-06 no abre ningún archivo crudo; su entrada son los
parquet de TDA-04).

---

## 18. `ingest/run_tda06.py`

Punto de entrada de terminal, mismo espíritu que TDA-00..05: carga la
configuración, llama a `run_tda06_analysis`, vuelca cada tabla (perfil
global, por año, por día de semana, segmentación, calibración) a su CSV,
dibuja el ÚNICO gráfico obligatorio de la etapa — el perfil intradía por
minuto superpuesto por año, para volumen, `|r|`, `rg_t` y `zero_1m` (2019
y 2025 en línea punteada y más clara, nunca con el mismo peso visual que
2020-2024) — persiste `s(m)`/`r_tilde` SOLO si STOP-6 no se activó, e
imprime el resumen completo de auditoría.

**Reproducir la ejecución**:

```bash
python -m ohlcv_dataroad.ingest.run_tda06 --config configs/mnq_snapshot.yaml
```

---

## 16. `ingest/run_tda05.py`

Punto de entrada de terminal, mismo espíritu que TDA-00..04: carga la
configuración, llama a `run_tda05_analysis`, vuelca cada tabla a su CSV,
dibuja el histograma de ticks (el único gráfico que exige el roadmap para
esta etapa — región central ±30 ticks, con el porcentaje exacto que queda
fuera citado en el propio gráfico, sección 5 de la tarea) e imprime el
resumen completo de auditoría.

**Reproducir la ejecución**:

```bash
python -m ohlcv_dataroad.ingest.run_tda05 --config configs/mnq_snapshot.yaml
```
