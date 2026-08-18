# TDA-01 — Semántica temporal, de sesión y de contrato

**Etapa:** `docs/methodology/Tsay_OHLCV_analysis_roadmap.md` § TDA-01
**Depende de:** TDA-00 (`PASS`, ver `reports/mnq/TDA00_inventario.md`)
**Alcance de datos:** exclusivamente el conjunto de investigación (22 archivos, `< 2025-06-23 00:00:00 UTC`, ver `configs/mnq_snapshot.yaml`). Ningún archivo de `holdout_files` fue abierto en esta etapa.
**Evidencia reproducible:** `reports/mnq/TDA01_evidencia_gaps.csv` (4.093 huecos internos, generado por `python -m ohlcv_dataroad.ingest.run_tda01_forensics`).

> **Este documento no toma ninguna decisión de diseño para `ohlcv_dataroad`.** Caracteriza la semántica temporal ya existente en el dato. No fija ventana operativa, no construye grilla, no decide rollover, no calcula retornos.

---

## 1. Convención del timestamp

## **CIERRE de barra (fin de intervalo).**

Un timestamp `t` etiqueta la barra que resume el intervalo `[t − 1 min, t)`. Es decir: la barra etiquetada `09:31:00` contiene las operaciones ocurridas entre `09:30:00` y `09:30:59`, y **no** la actividad que empieza en `09:31:00`.

Esta conclusión combina evidencia **documental primaria** (documentación oficial de NinjaTrader, la plataforma de origen del dato) con evidencia **forense** obtenida de forma independiente sobre el conjunto de investigación, en tres anclas horarias distintas que convergen en el mismo resultado (§2). No queda `INDETERMINADO`: es la conclusión mejor sustentada de todo este informe.

---

## 2. Evidencia utilizada

### 2.1 Evidencia documental (prioridad máxima)

| # | Fuente | Tipo | Qué establece |
|---|---|---|---|
| D1 | NinjaTrader, *"How to Export Historical Data"*, guía oficial NT8 (`ninjatrader.com/support/helpguides/nt8/exporting.htm`) | Documentación oficial del proveedor | *"[Historical data is] exported with End of Bar time stamps"*; exportación en zona horaria **UTC**. |
| D2 | NinjaTrader, *"Importing"* (`ninjatrader.com/support/helpguides/nt8/importing.htm`) | Documentación oficial del proveedor | Formato exacto de barra de 1 minuto: `yyyyMMdd HHmmss;open;high;low;close;volume` — **coincide campo a campo** con el formato real de nuestros 27 archivos crudos. La plataforma reconoce explícitamente dos convenciones posibles (inicio/fin de barra) como opciones de *importación configurables*, lo que confirma que "fin de barra" no es un accidente sino una convención deliberada y documentada del formato de **exportación**. |
| D3 | NinjaTrader, *"How Bars Are Built"*, guía oficial NT8 (`ninjatrader.com/support/helpguides/nt8/how_bars_are_built.htm`) | Documentación oficial del proveedor (fuente primaria directa) | Cita textual: *"NinjaTrader stamps a bar with the closing time of the bar."* Ejemplo textual: una barra de 1 minuto etiquetada `9:31:00 AM` contiene datos de `9:30:00 AM` a `9:30:59 AM`. Esta fuente **reemplaza** a la referencia de foro citada en una versión anterior de este informe (que no pudo re-obtenerse textualmente): es la misma afirmación, pero ahora respaldada por la guía de ayuda oficial, no por un hilo de soporte. |
| D4 | NinjaTrader, documentación y terminología oficial sobre tipos de datos históricos (`Last` / `Bid` / `Ask`) | Documentación oficial del proveedor | *"In NinjaTrader, 'last' refers to historical trades."* La plataforma distingue tres series históricas independientes — `Last` (precio de operaciones ejecutadas), `Bid`, `Ask` — y `Last` es, además, el tipo de dato histórico disponible por defecto salvo que se solicite explícitamente Bid/Ask. `Last` es la terminología propia y documentada de la plataforma para "precio de la última operación ejecutada", no una convención ambigua. |

**`price_type = Last`: `CONFIRMADO` + `DOCUMENTADO`.** D4 documenta qué significa `Last` en la terminología de la plataforma de origen (precio de operaciones ejecutadas, no cotizaciones). A eso se suma que este snapshot **fue descargado deliberadamente como tipo `Last`** — una decisión de proyecto, no una inferencia hecha a partir del sufijo del nombre de archivo. Con ambos elementos juntos (documentación de qué es `Last` + decisión deliberada de descargarlo como tal), la clasificación correcta es `CONFIRMADO + DOCUMENTADO`, no `INFERIDO`: no queda ninguna duda razonable sobre qué representa la columna de precio de este dataset.

**Nota de alcance sobre D1–D4.** Esta documentación describe el comportamiento **general y por defecto** de la plataforma NinjaTrader 8. No es una confirmación específica de la configuración exacta usada para exportar *este* snapshot concreto (versión de NinjaTrader, ajustes del usuario) — salvo en el caso de `price_type`, donde sí se cuenta con la confirmación directa de que la descarga fue deliberadamente de tipo `Last` (ítem anterior). El peso de la conclusión sobre inicio/cierre de barra (§1) no descansa únicamente en D1–D3, sino en su **convergencia** con la evidencia forense independiente de §2.2.

**Documentación ya almacenada en el repositorio consultada:** `docs/instruments/mnq/MNQ_DATA_PRIOR_KNOWLEDGE.md` (§5.1–5.6, §9.2, §10), `reports/mnq/00_initial_repository_audit.md`, `reports/mnq/TDA00_inventario.md`, `configs/mnq_snapshot.yaml`. Ninguno de estos documentos internos resuelve la pregunta inicio/cierre — todos la marcan `UNRESOLVED`/pendiente; es exactamente la pregunta que este informe responde por primera vez con evidencia directa.

### 2.2 Evidencia forense (conjunto de investigación exclusivamente)

**Método.** Se calculó, para cada uno de los 22 archivos del conjunto de investigación por separado (nunca a través de una frontera entre archivos, que es una transición de contrato, no un hueco — ver `src/ohlcv_dataroad/ingest/tda01_temporal_semantics.py`), todo salto de más de 1 minuto entre dos filas consecutivas del mismo archivo. Cada borde del hueco se convirtió a hora local de Nueva York de forma DST-aware (`zoneinfo`, `America/New_York`, localizando primero en UTC — el hecho ya confirmado — y convirtiendo después, el orden que exige la memoria heredada §5.6). Antes de calcular nada, `build_forensic_evidence` valida que `research_files`/`holdout_files` sean disjuntos y que ningún timestamp de investigación alcance la frontera del hold-out (`src/ohlcv_dataroad/ingest/holdout_guard.py`, la misma protección que usa TDA-00 — ver §2.4).

La lógica de la prueba: el corte de mantenimiento diario de CME es un cese de negociación real y conocido, de duración aproximada de una hora, que ocurre casi todos los días de la semana. Si el timestamp marca el **inicio** de la barra, la última barra antes del corte debería etiquetarse un minuto **antes** de la hora de corte (p. ej. `16:59`), y la primera barra tras la reapertura debería etiquetarse **en** la hora de reapertura (p. ej. `18:00`). Si el timestamp marca el **cierre**, ocurre lo contrario: la última barra se etiqueta **en** la hora de corte (`17:00`) y la primera tras la reapertura se etiqueta un minuto **después** de la reapertura (`18:01`), porque esa es la barra que recién se completa en ese instante.

**Resultado — tres anclas horarias independientes, mismo veredicto:**

| Ancla | Población base | N que cumple la firma cierre-de-barra | Duración dominante |
|---|---:|---|---|
| 1. Corte de mantenimiento diario (huecos de 30–90 min) | 1.178 huecos | última barra **`17:00:00`** NY en 1.078/1.178 (91,5 %); primera barra **`18:01:00`** NY en 1.118/1.178 (94,9 %) | 61 min en 1.073/1.178 (91,1 %) |
| 2. Cierre/reapertura semanal (huecos > 40 h) | 174 huecos | última barra **`17:00:00`** NY en 156/174 (89,7 %); primera barra `18:01:00` NY en 96/174 (55,2 %; más ruido por liquidez fina del domingo) | variable (fin de semana) |
| 3. Cierres anticipados (regla exacta, ver abajo) | 2.741 huecos de magnitud "other" | **31** huecos cumplen la firma exacta de dos lados (`13:00:00`→`18:01:00`) | 301 min en 31/31 (100 %) |

Las anclas 1 y 2 se leen como un porcentaje **dentro de una población definida por tamaño de hueco, independiente de la hora del borde**: el hecho de que esa hora se concentre en una única etiqueta con 90 %+ de los casos es la evidencia. La ancla 3 se construye de otra forma (ver regla exacta abajo) y por eso su columna central no es un porcentaje dentro de un grupo pre-filtrado por hora — sería circular —, sino el **número de huecos, dentro de los 2.741 no explicados por las anclas 1–2, que casan exactamente con la firma de dos lados que predice la convención de cierre**. Que existan 31 huecos reales con esa firma exacta, y que los 31 tengan además una duración idéntica (301 minutos, compatible con "cierre a las 13:00, reapertura normal a las 18:01 esa misma tarde"), es la evidencia — no un porcentaje.

Las tres anclas — un fenómeno que ocurre ~250 veces al año, uno que ocurre 52 veces al año, y uno que ocurre ~6 veces al año — **coinciden en el mismo patrón**: el borde de la barra está en la hora de corte, no un minuto antes; y la primera barra tras la reapertura aparece un minuto después de la reapertura, no en la reapertura misma. Esto es exactamente lo que predice la convención de **cierre de barra**, y lo opuesto de lo que predice inicio de barra.

**Regla exacta de la ancla 3 (reproducible, `identify_early_close_like_gaps` en `tda01_temporal_semantics.py`):** de los huecos ya clasificados como `"other"` (ni del tamaño del corte diario ni de un fin de semana), se seleccionan los que cumplen **ambas** condiciones a la vez: borde anterior en NY exactamente `13:00:00` **y** borde posterior en NY exactamente `18:01:00`. Es una coincidencia exacta de dos lados, no una tolerancia ni un rango — un hueco con borde anterior `13:01:00` o borde posterior distinto de `18:01:00` no se cuenta, aunque esté cerca. Esta regla no clasifica ningún hueco por causa de calendario (eso sigue siendo TDA-02): sólo selecciona, dentro de la evidencia ya calculada, los que tienen la forma que la convención de cierre predice para un cierre anticipado seguido de reapertura normal.

**Verificación cruzada del recuento (orden de magnitud, no validación de calendario).** La memoria heredada (`MNQ_DATA_PRIOR_KNOWLEDGE.md` §10.3) reporta 40 fechas de cierre anticipado verificadas sobre el snapshot completo (27 archivos). El conjunto de investigación cubre ~83 % del snapshot por filas; `40 × 0,83 ≈ 33`. La regla exacta reproducible encontró **31** — ligeramente por debajo de esa aproximación, una diferencia pequeña y esperable (la regla exacta de dos lados excluye, por diseño, cualquier cierre anticipado real cuya reapertura no haya seguido el patrón normal de las `18:01`, algo que puede ocurrir y que un inventario completo de TDA-02 podría investigar). El resultado es consistente en orden de magnitud, sin necesidad de abrir el hold-out ni de cruzar fecha por fecha contra un calendario oficial.

**Confirmación de manejo correcto de DST, con datos reales del conjunto de investigación:**

```
Transición de primavera 2020 (2020-03-08):
  antes  2020-03-04 22:00 UTC -> 17:00:00 NY   (EST, UTC-5)
  antes  2020-03-09 21:00 UTC -> 17:00:00 NY   (EDT, UTC-4, tras el cambio)

Transición de otoño 2020 (2020-11-01):
  antes  2020-10-28 21:00 UTC -> 17:00:00 NY   (EDT, UTC-4)
  antes  2020-11-02 22:00 UTC -> 17:00:00 NY   (EST, UTC-5, tras el cambio)
```

El offset UTC del ancla cambia exactamente 1 hora al cruzar cada transición, mientras que la etiqueta en hora local de Nueva York permanece fija en `17:00:00`. Esto sólo es posible si la conversión se hizo con reglas de huso horario conscientes de DST (`zoneinfo`/IANA) y no con un offset fijo — y confirma, con datos reales, que ese es el método correcto para cualquier conversión posterior (§7).

**Método de detección — reproducible.** `src/ohlcv_dataroad/ingest/tda01_temporal_semantics.py` (documentado en `src/ohlcv_dataroad/ingest/README.md`); comando: `python -m ohlcv_dataroad.ingest.run_tda01_forensics`; artefacto: `reports/mnq/TDA01_evidencia_gaps.csv`.

### 2.3 Evidencia de mercado consultada, no usada como base de la conclusión

Se intentó obtener confirmación directa del horario oficial de CME Globex (`cmegroup.com`) para contrastar la hora exacta del corte de mantenimiento; las solicitudes fueron bloqueadas/agotaron el tiempo de espera repetidamente, tanto en la sesión que produjo la primera versión de este informe como en esta revisión. **CME sigue registrado como fuente no confirmada** (no como corroboración) para el horario exacto del corte de mantenimiento. Fuentes secundarias (agregadores de datos de mercado, no CME directamente) sitúan el corte de mantenimiento en `17:00–18:00 ET`, consistente con el patrón observado en los datos — pero la conclusión de este informe **no depende** de esa cifra externa: se sostiene por la autoconsistencia interna del propio dato (misma etiqueta, ~1.100+ observaciones independientes, comportamiento correcto en DST) más la documentación oficial de NinjaTrader (§2.1).

### 2.4 Protección del hold-out durante el análisis forense

El cálculo de huecos de §2.2 se ejecuta a través de `build_forensic_evidence` (`tda01_temporal_semantics.py`), que reutiliza exactamente la misma protección que ya usa TDA-00 (`src/ohlcv_dataroad/ingest/holdout_guard.py`, no una copia independiente):

1. **Antes de abrir cualquier archivo**, se valida que `research_files` y `holdout_files` sean conjuntos disjuntos y que ninguna de las dos listas tenga nombres repetidos (`validate_research_holdout_disjoint`).
2. **Después de parsear el conjunto de investigación**, se valida que ningún timestamp alcance o supere `holdout_boundary_utc`, usando el máximo timestamp ya calculado por archivo — sin volver a abrir ni releer nada (`validate_last_timestamps_before_boundary`).

Ambas comprobaciones lanzan `HoldoutIsolationError` si fallan, y ninguna de las dos abre jamás un archivo de `config.holdout_files`. Cubierto por tests dedicados en `tests/test_tda01_temporal_semantics.py` (solapamiento de listas, duplicados, timestamp en la frontera, y confirmación explícita de que un hold-out inexistente en disco no impide una corrida normal).

---

## 3. Zona horaria raw: UTC — hecho confirmado

`raw timestamp timezone = UTC` se trata como hecho confirmado, tal como indica esta tarea. No se reevaluó UTC contra America/Chicago o America/New_York. La documentación de NinjaTrader (D1, §2.1) es consistente con este hecho de forma independiente (declara exportación en UTC), lo que corrobora — sin ser necesario para establecerlo — la decisión ya tomada.

---

## 4. Offset explícito en archivo: **NO**

Confirmado estructuralmente (formato `YYYYMMDD HHMMSS`, sin sufijo de offset ni de zona) en el 100 % de las 2.329.783 filas del snapshot (verificado en la auditoría inicial) y consistente con el formato de importación documentado por NinjaTrader (D2). Estado: `DOCUMENTADO` (estructura del archivo) + `CONFIRMADO` (verificación directa).

---

## 5. Disponibilidad temporal de O / H / L / C / V

Bajo la convención cierre-de-barra confirmada en §1, y dado que este es un export histórico por filas completas (no un feed en vivo con actualizaciones intra-barra):

| Campo | Momento en que el VALOR se determina económicamente | Momento en que el DATO está disponible en este archivo |
|---|---|---|
| `Open` | Al primer trade del intervalo `[t−1min, t)`, es decir, ya en `t−1min` | **En `t`** — la fila completa (incluido `Open`) sólo se emite cuando la barra cierra |
| `High` | Cuando ocurre el máximo del intervalo — no se sabe cuál es hasta que termina el intervalo | **En `t`** |
| `Low` | Análogo a `High` | **En `t`** |
| `Close` | En el último trade del intervalo, en o justo antes de `t` | **En `t`** |
| `Volume` | Es un agregado del intervalo completo | **En `t`** |

**Conclusión operativa.** No hay disponibilidad escalonada dentro de la barra en *este formato de datos*: los cinco campos de la barra etiquetada `t` están disponibles **conjuntamente, exactamente en `t`, no antes**. Esto difiere de un feed en vivo (donde `Open` sí se conocería antes que `High`/`Low`/`Close`/`Volume`) precisamente porque aquí los datos llegan como filas de barra ya cerrada, no como actualizaciones incrementales.

**Regla de causalidad resultante para etapas posteriores:** cualquier cálculo que use la barra etiquetada `t` **puede** tratarse como disponible en el instante `t` (no se necesita el desplazamiento conservador de "+1 barra" que exigiría la convención de inicio-de-barra). Sí sigue aplicando la regla general de que ningún cálculo en `t` puede usar información de `t+k`, `k>0` (barras futuras).

Clasificación: `DOCUMENTADO` (D1, D2) + `INFERIDO` (forense, §2.2), consistentes entre sí.

---

## 6. Definición y representación de la sesión

Se distingue explícitamente, como exige esta tarea, entre **(a)** el horario/estructura de negociación del mercado tal como se refleja en el dato, y **(b)** cualquier ventana operativa de análisis — que **no se decide en este documento**.

**Lo que el dato permite afirmar (a):**
- El instrumento cotiza de forma prácticamente continua, con un cese diario de negociación de ~60–61 minutos observado consistentemente alrededor de `17:00–18:01` hora de Nueva York (§2.2), y un cese más largo de fin de semana con el mismo borde de apertura semanal (última barra del viernes también en `17:00` NY).
- Existen jornadas con cierre marcadamente más temprano (última barra ≈ `13:00` NY, reapertura normal esa misma tarde), consistentes en cantidad y patrón con los cierres anticipados de feriados de CME ya documentados en la memoria heredada (§10.3), aunque esta tarea no verificó fecha por fecha contra el calendario oficial (eso es TDA-02).
- La memoria heredada (§10.4) advierte explícitamente que la ventana `04:30–16:00` de un proyecto anterior **no** es un óptimo derivado de los datos, sino una convención operativa de ese proyecto. Esta auditoría no encontró ningún motivo en la evidencia forense para tratar `04:30` o `16:00` como fronteras estructurales: la única frontera estructural con respaldo empírico directo, encontrada de forma independiente en esta tarea, es el corte de mantenimiento (~`17:00–18:01` NY).

**Lo que sigue sin decidirse (b):** si el análisis posterior trabajará sobre las ~24 h completas, sobre alguna ventana (RTH, ETH, u otra), o sobre una segmentación derivada de datos. **Esta decisión no se tomó aquí**, conforme a la instrucción explícita de esta tarea.

Clasificación: `INFERIDO` (patrón horario) + `PENDIENTE DE DECISIÓN DEL USUARIO` (ventana de análisis, fuera de alcance de TDA-01).

---

## 7. Tratamiento correcto de DST

**Regla semántica que debe usar TDA-02 (y cualquier etapa posterior) para convertir UTC a hora de mercado:**

```
1. Localizar el timestamp crudo como UTC   (tz_localize("UTC"))
   — nunca asumir un offset fijo.
2. Convertir a la zona horaria de destino usando una base de datos
   de reglas horarias IANA/tz  (tz_convert("America/New_York"),
   vía el módulo estándar zoneinfo en Python)
   — nunca aplicar un offset constante calculado a mano.
3. Sólo después de (1) y (2), asignar fecha operativa / aplicar calendario
   (orden ya validado en la memoria heredada, §5.6).
```

**Por qué "offset fijo" está prohibido, con evidencia propia:** América/New_York alterna entre EST (UTC-5) y EDT (UTC-4) dos veces al año. La evidencia de §2.2 muestra el mismo ancla de mercado (`17:00` NY) en horas UTC distintas según la época del año (`22:00` UTC en enero, `21:00` UTC en julio) — un offset fijo produciría una hora de mercado incorrecta durante media parte del año.

Esta tarea **no** ejecuta la auditoría completa de cobertura por transición DST (los 14 cambios de horario del rango completo, uno por uno) — eso pertenece a TDA-02. Lo que se deja establecido aquí es la **regla correcta**, verificada con dos ejemplos reales concretos (§2.2), que TDA-02 debe aplicar sistemáticamente.

Clasificación: `INFERIDO` (regla derivada de evidencia propia) + heredado como `PREVIOUSLY VALIDATED DATA POLICY` de la memoria del proyecto (§5.6), ahora confirmado con evidencia nueva e independiente.

---

## 8. Representación de contratos

**Lo que puede afirmarse en esta etapa, sin abrir el hold-out:**

- Cada uno de los 22 archivos del conjunto de investigación corresponde a **un único contrato trimestral**, identificado por el nombre de archivo `NN_mnq_MM_YY.Last.txt`, donde `NN` es un índice secuencial (`00`–`21`), `MM` es el mes de vencimiento (`03`,`06`,`09`,`12`) y `YY` el año de vencimiento a 2 dígitos.
- El mapeo mes→letra de vencimiento trimestral (`03→H, 06→M, 09→U, 12→Z`) es la convención estándar de la industria de futuros para meses de vencimiento (código de mes CME/Globex), no una convención propia de este dataset. Clasificación: `DOCUMENTADO` (convención de industria, de uso público y estándar en toda la industria de futuros).
- El nombre de archivo lleva la forma **corta** del contrato (p. ej. `H20`); la memoria heredada (§3.2) señala que la forma **larga** (`MNQH20`) no aparece fila a fila y debe tratarse como metadato, no como columna del dato — se hereda esta lección como guía, no como hecho nuevo verificado aquí.
- De las 21 transiciones entre archivos consecutivos del conjunto de investigación, se puede afirmar (dato ya establecido en `reports/mnq/00_initial_repository_audit.md` §3.5, sin necesidad de reabrir nada aquí): **19 transiciones limpias** (sin solapamiento) y **2 transiciones con solapamiento real** (`Z24→H25` y `H25→M25`, de ~9 y ~10 días respectivamente), ambas enteramente dentro del conjunto de investigación.
- Existe una **tercera** transición con solapamiento (`M26→U26`) documentada en la auditoría inicial del repositorio — pero esa transición involucra **exclusivamente archivos del hold-out** (`25_mnq_06_26` y `26_mnq_09_26`). Esta tarea no la reexamina ni la usa: se cita únicamente como referencia a un hallazgo ya publicado antes de que el hold-out quedara operativo como restricción de pipeline.

Clasificación: `DOCUMENTADO` (convención de nomenclatura, código de mes) + `CONFIRMADO` (transiciones limpias/solapadas, ya estructuralmente verificado).

---

## 9. Qué puede afirmarse sobre el roll en esta etapa

**Explícitamente, lo que NO se decide aquí (pertenece a TDA-03):** método de ajuste (aditivo/ratio/ninguno), regla de selección de contrato activo, construcción de cualquier serie continua.

**Lo que sí puede afirmarse:**
- El roll no está representado por ningún campo explícito en el dato — se infiere únicamente de la transición entre archivos (cambio de contrato = cambio de archivo fuente).
- En las 2 transiciones con solapamiento del conjunto de investigación, existen fechas en las que **ambos** contratos (saliente y entrante) tienen filas simultáneamente — construir cualquier serie que use esas fechas sin una regla explícita de selección de contrato produciría más de una barra por minuto. Esto confirma, a nivel de TDA-01, que TDA-03 es una etapa necesaria antes de calcular cualquier retorno que cruce esas fechas.
- La política de rollover heredada (`MNQ_DATA_PRIOR_KNOWLEDGE.md` §7.1, reglas 1–12) es una **lógica general reutilizable** (no mezclar contratos, confirmar por volumen, irreversibilidad, trazabilidad), pero sus **parámetros concretos** (umbral 55 %, 691 barras/día esperadas) están definidos sobre una ventana operativa (`04:30–16:00`) que este proyecto no ha adoptado (§6) y que, por tanto, no son directamente aplicables sin recalibración — decisión que corresponde a TDA-03, no a TDA-01.

Clasificación: `CONFIRMADO` (existencia de solapamiento) + `PENDIENTE` (método de roll, explícitamente fuera de alcance).

---

## 10. Representación de barras sin operaciones

**Hechos disponibles (verificados, no interpretados):**
- TDA-00 encontró **0 filas con `volume == 0`** en las 1.937.230 filas del conjunto de investigación (`reports/mnq/TDA00_inventario.md`, §5).
- El corte de mantenimiento diario (§2.2), un período en el que el **mercado está cerrado** (cese de negociación real y conocido, de ~60 minutos), se manifiesta en el dato como una **ausencia genuina de filas** (un hueco de ~61 minutos entre dos timestamps), **no** como una secuencia de ~60 filas con volumen 0 o con `O=H=L=C` repetido.

**Por qué esto NO resuelve la pregunta general.** Los dos hechos anteriores describen exclusivamente cómo se representa un intervalo en el que **el mercado está cerrado** — una condición estructural conocida de antemano (horario de mantenimiento), no la ausencia de operaciones dentro de un mercado abierto. Que la ausencia de filas sea la representación cuando *no se puede* operar no implica, por sí solo, cuál sería la representación de un minuto en el que el mercado *podría* operar pero, de hecho, no hubo ninguna operación (p. ej., una franja de muy baja liquidez de madrugada, con el mercado técnicamente abierto). Son dos fenómenos estructuralmente distintos, y generalizar del primero al segundo sería una extrapolación no sostenida por la evidencia reunida en esta etapa — exactamente el tipo de salto que esta tarea pidió no dar.

**Por tanto:** la representación de un minuto de **mercado abierto** sin operaciones queda `INDETERMINADO`. Verificarlo requiere el inventario completo de huecos de TDA-02 — en particular, aislar huecos cortos (de 1 a pocos minutos) que caigan dentro de horario nominal de mercado abierto y comprobar si aparecen como fila ausente, como fila de volumen 0, o de alguna otra forma. Esa comprobación se traslada íntegramente a TDA-02; no se resuelve aquí.

Clasificación de los dos hechos: `CONFIRMADO` (ambos, verificables directamente sobre datos ya procesados por TDA-00/TDA-01). Clasificación de la representación general de "minuto de mercado abierto sin operaciones": `INDETERMINADO`.

---

## 11. Tabla de confianza por afirmación

| # | Afirmación | Clasificación | Evidencia |
|---|---|---|---|
| 1 | Zona horaria raw = UTC | `CONFIRMADO` (hecho dado por esta tarea) | Fuente de los datos (declarado por el usuario); corroborado independientemente por D1 |
| 2 | Sin offset explícito en archivo | `DOCUMENTADO` + `CONFIRMADO` | D2; verificación estructural directa (100 % de las filas) |
| 3 | **Timestamp = cierre de barra** | `DOCUMENTADO` + `INFERIDO` (convergentes) | D1, D2, D3 (documental, incluye cita textual de *"How Bars Are Built"*); 3 anclas forenses independientes, ~1.380 observaciones combinadas (§2.2) |
| 4 | Disponibilidad conjunta de O/H/L/C/V en `t` | `DOCUMENTADO` + `INFERIDO` | Consecuencia directa de #3 |
| 5 | Corte de mantenimiento diario ≈ `17:00–18:01` NY | `INFERIDO` (fuerte, ~1.100 obs.) | §2.2; NO confirmado contra fuente primaria de CME (acceso bloqueado en dos sesiones distintas) |
| 6 | Cierre semanal viernes ≈ `17:00` NY, reapertura domingo ≈ `18:01` NY | `INFERIDO` | §2.2 (174 obs.) |
| 7 | Cierres anticipados con firma exacta `13:00`→`18:01` NY | `INFERIDO` | §2.2 (31 obs. por regla exacta y reproducible; orden de magnitud consistente con memoria heredada) |
| 8 | Ventana `04:30–16:00` no es frontera estructural | `INFERIDO` (heredado + no contradicho por evidencia nueva) | Memoria heredada §10.4; ausencia de soporte forense propio para esa ventana |
| 9 | Regla DST correcta = localizar UTC → convertir con zoneinfo/IANA | `INFERIDO` (con 2 ejemplos reales) + heredado como política validada | §2.2, §7 |
| 10 | 2 de 3 solapamientos de contrato están en el conjunto de investigación | `CONFIRMADO` | `reports/mnq/00_initial_repository_audit.md` §3.5 |
| 11 | Método de roll (ratio/aditivo/ninguno) | `PENDIENTE` (fuera de alcance, TDA-03) | — |
| 12a | 0 barras con `volume == 0` en el conjunto de investigación | `CONFIRMADO` | TDA-00 (`reports/mnq/TDA00_inventario.md`, §5) |
| 12b | Durante el corte de mantenimiento (mercado **cerrado**) no se generan filas | `CONFIRMADO` | §2.2/§10 — hueco genuino, no filas de volumen 0 |
| 12c | Representación de un minuto de mercado **abierto** sin operaciones | `INDETERMINADO` | §10 — el caso confirmado (12b) es de mercado cerrado y no generaliza; trasladado a TDA-02 |
| 13 | `price_type = Last` = precio de la última operación ejecutada | `CONFIRMADO` + `DOCUMENTADO` | D4 (terminología oficial de la plataforma) + descarga deliberada de tipo `Last` para este snapshot (decisión de proyecto) |
| 14 | Convención exacta de exportación usada en *este* snapshot concreto (versión/config de NinjaTrader) | `INDETERMINADO` | Ninguna fuente confirma la configuración específica de esta exportación; se infiere del comportamiento general documentado + la autoconsistencia forense |

---

## 12. Consecuencias causales para etapas posteriores

1. **La barra etiquetada `t` puede tratarse como disponible en `t`**, no en `t+1`. Esto es **menos conservador** que el supuesto por defecto que prescribe el roadmap ante indeterminación — y está justificado porque la semántica **no** quedó indeterminada. Cualquier etapa posterior que shiftee las barras "por seguridad" estaría introduciendo un sesgo no necesario y debería justificarlo explícitamente si decide hacerlo de todos modos.
2. **TDA-02** debe construir su grilla esperada de minutos consistente con la convención de cierre: la primera barra de una sesión que reabre a las `18:00` se espera etiquetada `18:01`, no `18:00` — construir la grilla con el supuesto contrario clasificaría sistemáticamente la primera barra real de cada reapertura como "faltante".
3. **TDA-02** debe determinar, mediante su inventario completo de huecos, cómo se representa un minuto **de mercado abierto** sin operaciones (§10, pregunta que TDA-01 deja explícitamente `INDETERMINADA`) — sin asumir que el patrón observado durante el corte de mantenimiento (mercado **cerrado**) se generaliza a horario abierto. Lo único que puede darse por sentado desde TDA-01 es que no hace falta buscar `volume == 0` como señal (0 casos en el conjunto de investigación); eso no resuelve si un hueco corto en horario abierto significa "sin operaciones", un dato faltante, u otra cosa.
4. **TDA-03** tiene, en las 2 fechas de solapamiento real del conjunto de investigación, evidencia directa de que sin una regla de selección de contrato la serie tendría más de una barra por minuto — confirma que TDA-03 es un prerrequisito antes de calcular cualquier retorno que cruce esas fechas.
5. **Cualquier conversión a hora local** (TDA-02 en adelante) debe implementarse con `zoneinfo`/IANA, nunca con un offset fijo, bajo pena de introducir un error sistemático de 1 hora durante la mitad del año (§7).
6. La ventana `04:30–16:00` sigue sin ningún respaldo empírico propio de este proyecto; si una etapa futura la reintroduce, debe justificarlo de nuevo, no heredarla.

---

## 13. Preguntas que siguen abiertas

1. **Configuración exacta de exportación de este snapshot concreto** (ítem 14 de §11): la evidencia es fuerte pero indirecta (comportamiento documentado de la plataforma + autoconsistencia forense), no una confirmación específica de este archivo. Riesgo residual bajo, no cero.
2. **Confirmación oficial de CME del horario exacto del corte de mantenimiento**: no se obtuvo, ni en la sesión que produjo la primera versión de este informe ni en esta revisión (bloqueo/reset de conexión repetido a `cmegroup.com`); la conclusión de este informe no depende de ella, pero cerrar esta fuente reforzaría aún más la evidencia.
3. **Representación de un minuto de mercado ABIERTO sin operaciones — `INDETERMINADO`** (§10, §11 ítem 12c). Sólo se verificó el caso de mercado cerrado (corte de mantenimiento); no generaliza a horario abierto. Su resolución se traslada íntegramente al inventario completo de huecos de TDA-02.
4. **Patrón recurrente sin explicar** en el propio conjunto de huecos "tipo mantenimiento": un pequeño número de huecos (p. ej., 4 casos con borde en `16:33` NY, 4 casos en `05:49` NY) no encaja con el corte de mantenimiento habitual. Es un volumen pequeño (≈1 % del total de huecos de esa categoría) y podría estar relacionado con el "patrón recurrente ~16:20–16:30 ET, concentrado en 2019–2021" que la memoria heredada (§10.7) ya documenta como sin explicar. No se investiga más aquí — queda para TDA-02.
5. **Verificación fecha por fecha de los 31 cierres anticipados** (regla exacta, §2.2) contra el calendario oficial de feriados de CME — sólo se hizo una verificación de orden de magnitud, no una comprobación exhaustiva por fecha (TDA-02).

Ninguna de estas preguntas es bloqueante para continuar con TDA-02, pero **todas** deben permanecer visibles: son exactamente el tipo de incertidumbre silenciosa que la síntesis metodológica (`Tsay_sintesis_transversal_OHLCV.md`, eje 7) advierte que puede invalidar un pipeline entero sin dejar rastro si se ignora.

---

## Estado final

**`PASS_WITH_OPEN_QUESTIONS`**

La pregunta bloqueante principal del roadmap para esta etapa — inicio vs. cierre de barra — **queda resuelta** con evidencia documental primaria y forense convergente, no permanece `INDETERMINADA`. No se activa el supuesto conservador por defecto del roadmap (timestamp = inicio) porque no hizo falta: la evidencia apunta de forma consistente y multi-anclada a la convención opuesta (cierre). `price_type = Last` también queda resuelto (`CONFIRMADO + DOCUMENTADO`, §2.1). Queda explícitamente `INDETERMINADA`, y trasladada a TDA-02, la representación de un minuto de mercado **abierto** sin operaciones (§10) — la evidencia disponible en esta etapa sólo cubre el caso de mercado cerrado y no debe generalizarse. Quedan, además, varias preguntas secundarias legítimamente abiertas (§13) que deben trasladarse a TDA-02 y TDA-03, no resolverse por conveniencia.

**No se avanza a TDA-02.** Este documento y su evidencia (`TDA01_evidencia_gaps.csv`) quedan a la espera de revisión y aprobación.
