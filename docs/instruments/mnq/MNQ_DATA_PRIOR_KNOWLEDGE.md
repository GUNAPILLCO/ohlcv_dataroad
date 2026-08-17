# MNQ — CONOCIMIENTO PREVIO DEL DATASET

**Repositorio:** `ohlcv_dataroad`
**Ruta:** `docs/instruments/mnq/MNQ_DATA_PRIOR_KNOWLEDGE.md`
**Instrumento:** Micro E-mini Nasdaq-100 (MNQ) — CME Globex
**Naturaleza:** memoria heredada de auditorías de datos previas. **No es un informe de análisis.**

---

# 1. Propósito y alcance

## 1.1. Qué responde este documento

> **¿Qué sabemos ya sobre los datos MNQ antes de comenzar el análisis empírico de OHLCV DataRoad?**

Existe una auditoría previa sustancial de este mismo snapshot, realizada en un proyecto anterior. Este documento traslada **únicamente el conocimiento sobre los datos** que sigue siendo relevante, para impedir que DataRoad:

- vuelva a investigar problemas ya resueltos;
- repita errores metodológicos ya identificados;
- olvide anomalías conocidas todavía sin explicar;
- pierda políticas de tratamiento de contratos ya validadas y verificadas;
- contamine accidentalmente el hold-out protegido.

## 1.2. Qué NO hace este documento

Este documento **no transporta conclusiones estadísticas**. Los proyectos anteriores midieron autocorrelación, dependencia serial, heterocedasticidad condicional, estacionalidad intradía, estabilidad temporal, distribuciones por año y por régimen, correlaciones y detección de outliers. **Nada de eso se hereda como evidencia vigente.** DataRoad debe regenerar cualquier resultado estadístico de forma independiente (ver §14).

Tampoco hereda ninguna decisión de Machine Learning: targets, features, horizontes, modelos, folds, thresholds operativos ni reglas de trading. `ohlcv_dataroad` es un repositorio de **análisis de datos**, no de ML.

## 1.3. Convención de estados — no mezclar

Cada elemento de este documento lleva exactamente uno de estos estados:

| Estado | Significado |
|---|---|
| **CONFIRMED DATA FACT** | Comprobado directamente sobre los datos del snapshot actual. |
| **PREVIOUSLY VALIDATED DATA POLICY** | Regla de tratamiento implementada, probada y validada anteriormente. |
| **UNRESOLVED DATA QUESTION** | Problema o interpretación todavía no confirmada. |
| **RESOLVED HISTORICAL ISSUE** | Problema que existió en versiones anteriores y ya no existe en el snapshot actual. |
| **INVALIDATED HISTORICAL PRACTICE** | Procedimiento anterior que sabemos que no debe reutilizarse. |
| **HOLDOUT GOVERNANCE** | Restricción normativa asociada al período protegido. |

Una variante adicional, usada sólo en §7:

> **PREVIOUSLY VALIDATED DATA POLICY — REQUIRES REVALIDATION UNDER DATAROAD SCOPE**
> Regla validada bajo una ventana operativa (04:30–16:00, 691 barras/día) que **DataRoad no ha adoptado**. Su lógica general se conserva; sus parámetros dependientes de esa ventana no.

---

# 2. Identidad del snapshot

## 2.1. Identificadores — CONFIRMED DATA FACT

| Campo | Valor |
|---|---|
| Instrumento | Micro E-mini Nasdaq-100 (MNQ), CME Globex |
| Fuente | NinjaTrader (exportación de archivos por contrato) |
| Frecuencia | Barras de 1 minuto |
| Price type | `Last` |
| Formato | Texto plano, un archivo por contrato, nombre `NN_mnq_MM_YY.Last.txt` |
| Separador | `;` — 6 campos por línea |
| Nº de archivos | **27** |
| Contratos | **27**, trimestrales, secuencia H/M/U/Z **completa sin trimestre faltante**, desde `00_mnq_03_20` (H20) hasta `26_mnq_09_26` (U26) |
| Rango temporal | **2019-12-23 03:01:00 → 2026-07-31 20:10:00** (timestamps tal como vienen exportados) |
| Filas totales | **2.329.783** |
| Bytes totales | 122.523.200 |
| `dataset_version` | `IRIS-MNQ-SRC-20191223_20260731-e382a75ac222` |
| `hash_snapshot_sha256` | `e382a75ac2222c0391a777d597459fab439e606a545daf73d2da2d512260d32c` |
| Versión anterior del mismo contenido | `IRIS-MNQ-SRC-2026-08-14-e382a75a` (mismo hash; sustituida por un esquema de `version_id` determinista) |
| Fecha de generación del manifest | 2026-08-14T03:11:30Z |

**Nota sobre el rango.** El snapshot empieza el **2019-12-23**, no en 2020. La referencia informal "2020–2026" que aparece en documentación anterior es aproximada; el primer timestamp real es de diciembre de 2019 (contrato H20, que cotiza desde el trimestre previo a su vencimiento).

**Nota sobre el `version_id`.** El esquema es determinista respecto del contenido (`IRIS-MNQ-SRC-<primer_ts:YYYYMMDD>_<ultimo_ts:YYYYMMDD>-<12 chars del hash>`): dos ejecuciones sobre el mismo contenido producen siempre el mismo identificador. Una exportación futura con más datos constituye una **nueva versión** del snapshot, no una sobrescritura conceptual de ésta.

## 2.2. Contrato incompleto — CONFIRMED DATA FACT

```
archivo: 26_mnq_09_26.Last.txt   (contrato U26)
filas:   54.015
motivo:  contrato vigente, aún no expirado en la fecha del snapshot
```

Frente a las **81.660 – 96.755 filas** de un contrato completo, U26 está aproximadamente al 55–65 % de un trimestre. **No es un defecto del dato**: es el estado natural del contrato en curso. Crecerá en una exportación futura, que será otra versión del snapshot.

**Implicación para DataRoad:** cualquier estadística agregada por contrato debe excluir U26 de las comparaciones de cobertura, o reportarlo por separado. Además, U26 cae íntegramente dentro del hold-out (§11).

## 2.3. Distribución de filas por contrato — CONFIRMED DATA FACT

Los 27 archivos tienen su `sha256`, `filas`, `bytes`, `primer_timestamp` y `ultimo_timestamp` individuales registrados en el manifest de origen. Valores de referencia:

| Estadístico | Valor |
|---|---|
| Contrato completo más pequeño | `00_mnq_03_20` (H20) — 81.660 filas |
| Contrato completo más grande | `21_mnq_06_25` (M25) — 96.755 filas |
| Contrato incompleto | `26_mnq_09_26` (U26) — 54.015 filas |

**Este rango de ~15.000 filas entre contratos completos es en sí un hecho a caracterizar**: DataRoad no debe asumir que todos los contratos tienen la misma cobertura.

---

# 3. Formato y estructura OHLCV

## 3.1. Esquema de origen — CONFIRMED DATA FACT

Cada línea contiene exactamente **6 campos** separados por `;`. Verificado línea por línea en el 100 % de las líneas de los 27 archivos.

## 3.2. Esquema persistido tras la ingestión previa — PREVIOUSLY VALIDATED DATA POLICY

El pipeline anterior persistía:

```
columnas: open, high, low, close, volume, contract
índice:   DatetimeIndex, tz-naive
```

La columna `contract` se conserva **desde la ingestión** en formato corto (`H20`, `M20`, `U20`, `Z20`). El instrumento (`MNQ`) y el contrato completo (`MNQH20`) se mantienen como **metadata separada en el manifest**, no fila a fila.

**Lección heredada (S00-03):** existieron dos convenciones de nomenclatura simultáneas (`MNQH20` y `H20`), lo que provocó problemas de portabilidad. La resolución fue una **única función de extracción** desde el nombre de archivo, sin duplicación de lógica. DataRoad debería fijar **una sola convención** desde el principio y conservar la forma larga como metadata.

**Nota de alcance:** este esquema es el que usaba el pipeline anterior. DataRoad no está obligado a reproducirlo, pero si diverge debe declararlo, porque toda la trazabilidad heredada asume estos nombres de columna.

---

# 4. Integridad confirmada

## 4.1. QA estructural sobre los 27 archivos fuente — CONFIRMED DATA FACT

Verificado línea por línea, en modo lectura, sobre los 27 archivos:

| Comprobación | Resultado |
|---|---|
| Campos por línea (separador `;`) | Exactamente 6 en el **100 %** de las líneas, en todos los archivos |
| Líneas malformadas / no parseables | **0** |
| Timestamps duplicados **dentro de un mismo archivo** | **0** |
| Timestamps fuera de orden **dentro de un mismo archivo** | **0** |
| OHLC estructuralmente inválido (`H<L`, `H<O`, `H<C`, `L>O`, `L>C`) | **0** |
| Volumen negativo o vacío | **0** |

> **No se detectó corrupción estructural en ningún archivo.**

## 4.2. Validaciones adicionales del pipeline de ingestión — CONFIRMED DATA FACT

El pipeline de ingestión anterior ejecutó, además, y **sin rechazar ninguna fila**:

```
esquema · parseo · timestamps · monotonicidad · duplicados globales ·
duplicados por (timestamp, contract) · nulos · infinitos · precios positivos ·
volumen no negativo · volumen entero · invariantes OHLC ·
filas exactamente duplicadas · archivos vacíos · transiciones de contrato
```

```
Filas rechazadas:          0
Duplicados exactos:        0
Duplicados (ts, contract): 0
Nulos:                     0
```

**Caveat de versión.** La corrida validada en detalle se ejecutó sobre el corpus de **26 archivos / 2.172.640 filas** (anterior a la actualización de fuente). Tras la actualización a 27 archivos / 2.329.783 filas, la suite completa (**26 unitarias + 10 de integración**) se reejecutó y **pasó contra el corpus actualizado**, y la QA estructural de §4.1 se ejecutó directamente sobre los 27 archivos. La integridad está confirmada para el snapshot actual; lo que corresponde a la versión de 26 archivos son los conteos intermedios, no el veredicto.

## 4.3. Lección sobre detección de duplicados — PREVIOUSLY VALIDATED DATA POLICY

Un bug real, encontrado y corregido: la comprobación de "filas exactamente duplicadas" comparaba sólo las columnas **sin incluir el índice** (`timestamp`), lo que habría marcado como duplicadas dos barras distintas que coincidieran por azar en OHLCV.

**Regla heredada:**

```
duplicado exacto              = timestamp + OHLCV + contract idénticos
duplicado (timestamp,contract) = mismo minuto, mismo contrato  -> ERROR, detiene la ingestión
mismo timestamp, contrato distinto -> NO es duplicado: es solapamiento de rollover (ver §7)
```

Confundir el tercer caso con el segundo es el error que rompe la ingestión durante un rollover. Existe una prueba de regresión explícita que fija este contrato.

## 4.4. Chequeo automatizado ausente — UNRESOLVED DATA QUESTION

El módulo de ingestión anterior **nunca implementó** un chequeo dedicado de **solapamiento de intervalos entre archivos** (distinto del chequeo de duplicados exactos por `timestamp`+`contract`).

Una auditoría manual sobre el corpus de 26 archivos concluyó **0 solapamientos**. **Ese resultado quedó superado por la actualización de fuente**: en el snapshot actual **sí existen 3 solapamientos reales** (§7). El chequeo automatizado sigue sin existir.

**Qué debe hacer DataRoad:** implementar la comprobación de solapamiento entre archivos como parte de la QA estructural, no como auditoría manual puntual — y no dar por buena la afirmación histórica de "0 solapamientos".

---

# 5. Timestamp y timezone

Ésta es la zona del conocimiento heredado donde **la distinción entre evidencia e inferencia importa más**. Nada de lo siguiente debe elevarse a hecho confirmado.

## 5.1. Estado de los timestamps crudos — CONFIRMED DATA FACT

```
timestamps de origen: tz-naive (sin información de zona horaria)
timezone_stored:      null
```

Los archivos exportados no llevan offset ni identificador de zona. Ninguna conversión de zona horaria se aplicó en la etapa de ingestión cruda, por decisión de alcance.

## 5.2. Zona horaria — UNRESOLVED DATA QUESTION (con respaldo empírico fuerte)

La hipótesis **UTC** fue evaluada programáticamente contra dos alternativas, **no asumida**. Se compararon las tres hipótesis contra la apertura 09:30 ET, el cierre 16:00 ET y el corte de mantenimiento CME (~17:00 ET), separando meses EDT y EST:

| Hipótesis | Alineación apertura | Alineación cierre | Corte de mantenimiento | Consistencia DST | Score total | Confianza |
|---|---:|---:|---:|---:|---:|---|
| **UTC** | 1 min | 0 min | 1 min | 0 min | **2.0** | **alta** |
| America/Chicago | 29 min | 29 min | 999 (no detectado) | 30 min | 1087.1 | baja |
| America/New_York | 38 min | 59 min | 999 (no detectado) | 9 min | 1105.1 | baja |

UTC gana por **~500×** y de forma consistente entre EDT y EST. Las otras dos hipótesis no logran localizar el corte de mantenimiento en absoluto y producen filas ambiguas o inexistentes en las transiciones DST (localizar directamente un índice tz-naive en una zona con DST genera ese problema; UTC, al no observar DST, no lo tiene).

Estado registrado:

```
timezone_selected:              UTC
timezone_validation_status:     empirically_supported
timezone_provider_confirmation: FALSE
timezone_evidence:              inferred_from_market_structure_and_dst
confidence_level:               high
```

> **UTC está fuertemente respaldado empíricamente. UTC NO está confirmado documentalmente por el proveedor.**

No existe en el repositorio anterior la configuración de exportación de NinjaTrader que lo confirmaría. La confirmación documental **sigue pendiente y no es resoluble con los datos**.

**Caveat de versión:** la validación de zona horaria se ejecutó sobre **2.172.640 filas** (corpus de 26 archivos). No se reejecutó sobre el snapshot actual de 2.329.783 filas. La conclusión es robusta por el margen del resultado, pero **la corrida es anterior a la actualización de fuente**.

## 5.3. Semántica del timestamp — UNRESOLVED DATA QUESTION

```
timestamp_semantics: unknown_not_confirmed
```

**No se sabe si el timestamp marca el inicio o el cierre de la barra de 1 minuto.** Se buscó evidencia en dos etapas independientes del pipeline anterior y no se encontró suficiente.

Decisión heredada, correcta y a preservar: **ninguna barra fue desplazada**. No se aplicó ningún ajuste basado en una semántica no confirmada.

Este punto afecta directamente la interpretación de `minute_of_day` y cualquier afirmación sobre disponibilidad de información en un instante dado.

## 5.4. Price type — CONFIRMED DATA FACT

```
price_type: "Last"
```

## 5.5. DST — CONFIRMED DATA FACT

Verificación explícita sobre las **14 fechas de transición DST (2020–2026)**: **0 fechas con minutos duplicados** dentro de la ventana analizada.

```
tz_localize("UTC").tz_convert("America/New_York")  ->  seguro frente a DST para este dataset
```

**Caveat:** la verificación se hizo dentro de la ventana 04:30–16:00 America/New_York. DataRoad, si trabaja sobre las ~24 h completas, debe reejecutarla sobre su propio alcance.

## 5.6. Orden correcto de operaciones — PREVIOUSLY VALIDATED DATA POLICY

Un error histórico (S01-02) fue filtrar fechas de mercado **antes** de completar la localización y conversión. El orden correcto, validado:

```
localizar en la zona horaria de origen
  -> convertir a la zona horaria de análisis
     -> asignar la fecha operativa
        -> aplicar calendario
```

Invertir cualquiera de estos pasos compromete la asignación de jornada y, con ella, todo el calendario.

---

# 6. Cobertura contractual

## 6.1. Estructura — CONFIRMED DATA FACT

```
27 contratos trimestrales, un archivo por contrato
secuencia H/M/U/Z COMPLETA, sin trimestre faltante
H20 (Mar-2020)  ...  U26 (Sep-2026)
26 transiciones posibles entre archivos consecutivos
```

## 6.2. Clasificación de las 26 transiciones — CONFIRMED DATA FACT

Conteo verificado programáticamente (no a mano), por intersección de fechas por contrato:

```
26 transiciones posibles
   23 handoffs LIMPIOS: CERO fechas con ambos contratos presentes
    3 transiciones con SOLAPAMIENTO REAL:  Z24->H25 · H25->M25 · M26->U26
```

**Corrección histórica registrada:** un borrador anterior citaba "25 transiciones" y "22 handoffs limpios". Las cifras correctas son **26 y 23**. El "25" mezclaba el total de *transiciones* con el total de *filas* de una tabla de fechas ambiguas (que sí es 25 — ver §7.3). Son dos conteos distintos que coinciden en ser cercanos por casualidad. **DataRoad no debe reintroducir esa confusión.**

## 6.3. Los tres solapamientos, a nivel de archivo — CONFIRMED DATA FACT

| Transición | Saliente termina | Entrante empieza | Solapamiento aprox. |
|---|---|---|---|
| `19_mnq_12_24` → `20_mnq_03_25` (Z24→H25) | 2024-12-20 21:30 | 2024-12-12 03:01 | **~8 días** |
| `20_mnq_03_25` → `21_mnq_06_25` (H25→M25) | 2025-03-22 15:03 | 2025-03-13 03:01 | **~9 días** |
| `25_mnq_06_26` → `26_mnq_09_26` (M26→U26) | 2026-06-18 13:30 | 2026-06-08 03:03 | **~10 días** |

**Este patrón es inconsistente con el resto del snapshot**: las otras 23 transiciones no solapan en absoluto. La causa (política de exportación distinta para esos archivos, reexportación posterior, o comportamiento del proveedor) **nunca fue explicada**.

**Nota de localización:** las dos primeras caen íntegramente en el conjunto de investigación; **la tercera (M26→U26) cae íntegramente dentro del hold-out** (§11.4).

---

# 7. Solapamientos y rollover

**Sección crítica.** Es el conocimiento operativo más valioso que se hereda, y el que más cuidado requiere al separar lo general de lo dependiente de la ventana histórica.

## 7.1. Política general de resolución — PREVIOUSLY VALIDATED DATA POLICY

Estas reglas constituyen la política de rollover previamente validada para MNQ. Algunas son invariantes de integridad claramente transferibles (no mezclar ni promediar contratos, conservar trazabilidad y verificar conservación), mientras que otras son decisiones metodológicas de construcción de la serie y deberán confirmarse dentro del alcance de DataRoad antes de adoptarse como política vigente.

| # | Regla | Fundamento |
|---|---|---|
| 1 | **Un único contrato seleccionado por fecha** | Una fecha con dos contratos produce más de una barra por minuto |
| 2 | **Nunca mezclar OHLCV de dos contratos** dentro de una misma fecha | Son dos instrumentos distintos, no dos observaciones del mismo |
| 3 | **Nunca promediar contratos** ni crear barras sintéticas | Un precio promediado no es un precio negociado |
| 4 | **Confirmación mediante el volumen del contrato entrante** | El traspaso de liquidez es la evidencia observable del roll |
| 5 | Comparación de volumen **sólo sobre minutos compartidos** | Comparar volúmenes de períodos distintos no mide traspaso |
| 6 | **Irreversibilidad**: una vez cruzado, el contrato activo no vuelve atrás | Evita oscilación entre contratos por ruido de volumen |
| 7 | **Una sola confirmación** basta (no se exige confirmación doble) | Parámetro `consecutive_confirmations_required: 1` |
| 8 | **Fecha efectiva = jornada siguiente observada**, nunca la fecha de la señal | La señal se conoce al terminar la jornada; aplicarla ese mismo día usaría información no disponible |
| 9 | **Las fechas de transición NO se hardcodean**: se detectan en los datos | Las fechas fijas sólo se usan como regresión en pruebas |
| 10 | **Trazabilidad total de las filas descartadas** — nunca borradas silenciosamente | Artefacto dedicado con motivo por fila |
| 11 | **Validación de conservación bloqueante en código productivo** | `filas_antes = filas_resueltas + filas_descartadas`, no sólo en pruebas |
| 12 | **Ningún back-adjustment automático de precios** | Los precios crudos se conservan |

**Regla 12 en detalle — PREVIOUSLY VALIDATED DATA POLICY.** La decisión vigente fue explícitamente *"no ajustar toda la serie de forma automática"*. Se resolvió la **selección de contrato**, no el **ajuste de nivel de precio**. El gap de precio en el momento del roll **nunca fue medido**: es un análisis pendiente, no una decisión tomada (§12).

## 7.2. Reglas dependientes de la ventana histórica

> ### PREVIOUSLY VALIDATED DATA POLICY — REQUIRES REVALIDATION UNDER DATAROAD SCOPE

Estas reglas funcionaron correctamente, pero sus **parámetros están atados a la ventana operativa 04:30–16:00 America/New_York con 691 barras/día esperadas**, que DataRoad **no ha adoptado** (y que, además, fue documentada por el propio proyecto anterior como una convención, no como un óptimo empírico — ver §10.4).

| Regla | Parámetro histórico | Por qué depende de la ventana |
|---|---|---|
| Criterio de sesión compartida completa | **691/691 barras** | 691 = número de minutos de 04:30 a 16:00 inclusive. Bajo otro alcance, el número cambia por completo |
| Umbral de confirmación por volumen | **≥ 55 %** del volumen del entrante (`min_incoming_share: 0.55`) | El share se calcula sobre los minutos compartidos **dentro de la ventana**; con las ~24 h completas la composición del volumen es distinta |
| Regla de respaldo 11 — condición de disparo | contrato activo con **exactamente 0 barras** esa fecha | "0 barras" significa 0 barras *dentro de la ventana*, no en la jornada real |
| Clasificación resultante de la jornada | `full_coverage` exige 691 barras **y contrato único** | Depende del conteo esperado de la ventana |

**Qué debe hacer DataRoad:** conservar la **lógica** (reglas 1–12 de §7.1) y **recalibrar los parámetros** bajo su propio alcance temporal. En particular: si DataRoad analiza las ~24 h completas, el "691" no aplica y el umbral del 55 % debe reevaluarse, no copiarse.

## 7.3. Fechas evaluadas y resultados verificados — CONFIRMED DATA FACT

El artefacto de fechas ambiguas registró **25 filas** (no confundir con las 26 transiciones):

```
23 filas: fecha con los DOS contratos presentes simultáneamente ese día
           (7 en Z24/H25, 7 en H25/M25, 9 en M26/U26)
 2 filas: fecha donde el contrato activo tiene 0 barras pero la fecha cae
           DENTRO de una ventana de solapamiento aún no confirmada
           (2025-03-16 y 2025-03-17, dentro de H25->M25)
```

Transiciones **confirmadas** por el algoritmo, verificadas por regresión contra los datos reales:

| Transición | Fecha de señal | Share del entrante | Contrato activo desde |
|---|---|---:|---|
| Z24 → H25 | 2024-12-17 | 69,10 % | **2024-12-18** |
| H25 → M25 | 2025-03-18 | 69,09 % | **2025-03-19** |
| M26 → U26 | 2026-06-15 | 76,44 % | **2026-06-16** |

Casos límite que **no** confirman, también verificados:

| Fecha | Qué ocurrió | Resultado |
|---|---|---|
| 2025-03-15 | Sesión compartida no completa (H25: 2 barras; M25: 1 barra) | No confirma |
| 2026-06-11 | M26 (activo) con cobertura parcial; U26 con share de sólo 1,3 % | No confirma; se conserva M26 con su cobertura real |

## 7.4. Regla de respaldo 11 — PREVIOUSLY VALIDATED DATA POLICY

Añadida en una revisión posterior, resuelve el caso "contrato activo sin datos, entrante con datos válidos":

```
1. El contrato activo debe tener EXACTAMENTE 0 barras esa fecha.
2. El contrato entrante debe tener barras esa fecha (cualquier cantidad).
3. NO se mezclan contratos: el activo no aporta nada ese día.
4. NO se crean ni completan barras sintéticas: se usa la cobertura REAL del entrante.
5. Motivo registrado: active_contract_no_data_fallback_to_incoming
6. NO adelanta el contrato activo formal para fechas siguientes: el cruce
   formal sigue dependiendo EXCLUSIVAMENTE de la confirmación por volumen.
```

**Caso de regresión verificado:** `2025-03-17` — H25 (activo) con 0 barras, M25 (entrante) con cobertura completa. Se conserva M25 sólo para esa fecha; **el rollover formal H25→M25 sigue confirmando el 2025-03-18 y efectivo desde el 2025-03-19, sin adelantarse**.

Segundo caso: `2025-03-16` — mismo mecanismo, pero el entrante sólo tiene 2 barras; se conservan esas 2 barras reales, sin relleno.

El punto 6 es el más importante de la regla: **separa la cobertura de datos de una fecha del estado formal del contrato activo.** Confundir ambos es lo que haría que un roll se adelantara silenciosamente.

## 7.5. Trazabilidad y conservación — CONFIRMED DATA FACT

```
filas ANTES de resolver rollover = filas resueltas + filas descartadas
1.166.364 = 1.152.510 + 13.854      ->  conservation_check_passed: true
```

Las 13.854 filas descartadas quedaron en un artefacto dedicado, **con motivo por fila, nunca borradas silenciosamente**. La validación es **bloqueante en código productivo**, no sólo en pruebas.

> **Estos conteos corresponden a la ventana 04:30–16:00 y no son transferibles a DataRoad como cifras.** Lo transferible es el **patrón**: conservación verificada de forma bloqueante + artefacto de filas descartadas con motivo.

---

# 8. Gaps históricos resueltos

## 8.1. Gap interno M23 — RESOLVED HISTORICAL ISSUE

| Campo | Valor histórico |
|---|---|
| Archivo / contrato | `13_mnq_06_23.Last.txt` (M23) |
| Tipo estructural | `intra_file` |
| Extremo anterior | 2023-04-05 18:03:00 |
| Extremo posterior | 2023-04-16 14:18:00 |
| Duración | 936.900 s = **260 h 15 min ≈ 10 d 20 h** |
| Jornadas calendario sin datos | 10 (2023-04-06 → 2023-04-15); ambos extremos con cobertura parcial |
| Clasificación | `no_resuelto`, `evidence_level: unconfirmed` |

> **RESUELTO (2026-07-31).** El archivo fuente fue **reemplazado por una versión más completa** (87.931 filas frente a 78.856 antes). En los datos vigentes, el salto máximo dentro de M23 es de **~57 h** (fin de semana ordinario), muy por debajo del umbral que definía el hallazgo como extraordinario. **El gap ya NO existe en el snapshot actual.**

## 8.2. Gap de transición H25 → M25 — RESOLVED HISTORICAL ISSUE

| Campo | Valor histórico |
|---|---|
| Archivos / contratos | `20_mnq_03_25` (H25) → `21_mnq_06_25` (M25) |
| Tipo estructural | `inter_contract` |
| Extremo anterior | 2025-03-21 13:30:00 |
| Extremo posterior | 2025-04-06 08:42:00 |
| Duración | 1.365.120 s = **379 h 12 min ≈ 15 d 19 h** |
| Jornadas calendario sin datos | 15 (2025-03-22 → 2025-04-05); ambos extremos con cobertura parcial |
| Clasificación | `no_resuelto`, `evidence_level: unconfirmed` |

> **RESUELTO (2026-07-31).** Ambos archivos fueron reemplazados. En los datos vigentes **H25 cubre 2024-12-12 → 2025-03-22 y M25 cubre 2025-03-13 → 2025-06-22: se SOLAPAN**, no hay ningún vacío entre ambos. El gap era **un artefacto de una exportación de fuente incompleta, no un fenómeno estructural del mercado**. Y ese mismo solapamiento es el que permite confirmar el rollover H25→M25 por volumen (§7.3).

## 8.3. Estado agregado de gaps — CONFIRMED DATA FACT

```
En el snapshot actual NO existe ningún gap en el bucket >100h.
Los gaps restantes del bucket 70min-100h (5 casos) son fines de semana
largos ordinarios, no anomalías sin explicar.
```

## 8.4. Lección metodológica — PREVIOUSLY VALIDATED DATA POLICY

**Dos gaps que se investigaron como anomalías estructurales del mercado resultaron ser fallas de exportación de la fuente.** La lección es doble:

1. **Antes de interpretar un gap grande como fenómeno de mercado, verificar si la exportación de origen está completa.** Es la explicación más barata y resultó ser la correcta en los dos casos.
2. **Ninguno de los dos gaps fue rellenado, interpolado ni eliminado** mientras estuvo abierto — quedaron persistidos íntegros y clasificados como `unconfirmed`. Ésa es la conducta correcta ante un gap sin explicar, y es lo que permitió que la actualización de fuente los resolviera limpiamente sin haber corrompido nada.

> **Estos dos gaps NO deben volver a tratarse como anomalías vigentes.** Aparecen extensamente en la documentación histórica; están cerrados por datos reales, no por decisión de alcance.

---

# 9. Anomalías OHLCV conocidas

## 9.1. Tres barras con volumen extremo — UNRESOLVED DATA QUESTION

Tres barras de un minuto con volumen **100–170×** el máximo habitual del resto del snapshot (que ronda **8.500 – 31.000**):

| Archivo | Timestamp | Volumen | OHLC de la barra |
|---|---|---:|---|
| `22_mnq_09_25.Last.txt` | **2025-07-01 10:08:00** | **1.258.222** | 22816.25 / 22934 / 22780 / 22831.5 |
| `22_mnq_09_25.Last.txt` | **2025-07-15 13:52:00** | **1.451.062** | 22862 / 23222.75 / 22805 / 23159.5 |
| `24_mnq_03_26.Last.txt` | **2026-01-27 03:44:00** | **1.534.923** | 25567.75 / 25977.5 / 25543.25 / 25972.25 |

**El OHLC de las tres barras es internamente válido** — no disparan ninguna comprobación de invariantes. Es **exclusivamente el campo de volumen** el que se sale del rango del resto del dataset.

> **Estado: UNRESOLVED DATA QUESTION.**
> No se ha afirmado que sean errores. No se ha afirmado que sean eventos reales de mercado. No se ha decidido que deban eliminarse. **Quedan abiertas.**

**Qué debe hacer DataRoad:** investigarlas antes de emitir cualquier conclusión que dependa de `volume` — perfil de actividad, estacionalidad de volumen, relación volumen-rango, detección de extremos basada en volumen. Mientras no se resuelvan, todo resultado sensible al volumen debería reportarse **con y sin** estas tres barras.

**Restricción crítica:** las **tres caen dentro del período reservado como hold-out** (§11.4). Su detección fue exposición pre-lock ya documentada. **Investigarlas ahora requiere autorización explícita de acceso al hold-out.**

## 9.2. Comparabilidad de métricas absolutas a lo largo del snapshot — UNRESOLVED DATA QUESTION

Un análisis histórico documentó que el **nivel de precio de MNQ creció sustancialmente a lo largo del período cubierto** (aproximadamente de ~8.700 a ~26.700 entre 2019 y 2026). Ese análisis **no se transporta como evidencia vigente** — DataRoad debe medirlo por su cuenta si lo necesita.

Lo que sí se hereda es la **consecuencia metodológica**, que es una lección de tratamiento de datos y no un resultado estadístico:

> Cualquier métrica expresada en **puntos absolutos** (rango en puntos, cuerpo de vela en puntos, umbrales de detección fijos) **no es comparable a lo largo del snapshot**, porque el denominador implícito cambió de forma sustancial. Una regla de detección calibrada globalmente sobre magnitudes absolutas selecciona preferentemente el tramo final del histórico.

La corrección adoptada anteriormente fue usar **métricas normalizadas por nivel de precio** en lugar de magnitudes absolutas. Los umbrales y resultados concretos de aquella detección **NO se heredan** (§14).

## 9.3. Cálculo de estadísticos seriales sobre series con huecos — PREVIOUSLY VALIDATED DATA POLICY

Una validación focalizada anterior encontró un defecto real y lo corrigió: los cálculos de autocorrelación construían pares entre observaciones pertenecientes a **segmentos consecutivos distintos** (es decir, a través de un hueco). El efecto medido fue pequeño pero real y estaba presente en varios lags.

**Regla heredada, aplicable a cualquier estadístico serial:**

```
Todo estadístico que empareje observaciones separadas por k posiciones
(autocovarianza, autocorrelación, tests de dependencia serial, matrices de
regresores lageados) debe respetar los límites de los segmentos consecutivos
y NUNCA formar un par que cruce un hueco.
```

Esto **no es** un resultado estadístico: es un requisito de implementación. Aplica igual sea cual sea el alcance temporal que adopte DataRoad. Requiere disponer de un identificador de segmento consecutivo (o equivalente) construido en la etapa de preparación.

**Nota:** la implementación de librería estándar de varios de estos tests **no** es consciente de huecos; construye su matriz de regresores sobre el array completo. Hubo que reimplementarlos manualmente. DataRoad debe verificarlo, no asumirlo.

---

# 10. Calendario y sesiones: conocimiento previo

## 10.1. Prácticas invalidadas — INVALIDATED HISTORICAL PRACTICE

### 10.1.1. Calendario NASDAQ aplicado a MNQ

```python
mcal.get_calendar("NASDAQ")   # <-- INVALIDADO
```

**MNQ es un futuro de CME Globex, no un instrumento del NASDAQ.** El uso del calendario NASDAQ comprometió: feriados, cierres anticipados, jornadas especiales, fechas excluidas y cobertura anual.

> **No se afirma que todas las fechas resultantes fueran incorrectas**, pero el procedimiento no es metodológicamente aceptable. Cualquier clasificación de jornadas heredada de esa etapa está comprometida.

### 10.1.2. Eliminación automática de jornadas por conteo de barras

Se eliminaron **80 jornadas** por no contener exactamente el número esperado de barras o por presentar huecos, **sin clasificarlas**.

Impacto documentado:

```
sesgo de selección de sesiones
pérdida de eventos especiales (cierres anticipados legítimos)
alteración de distribuciones
reducción artificial de escenarios extremos
```

> **INVALIDADA COMO REGLA DEFINITIVA.** Una jornada con menos barras de las esperadas **no es** por eso una jornada inválida: puede ser un cierre anticipado legítimo, una sesión especial o un día válido con horario reducido.

### 10.1.3. Filtrado temporal antes de completar la conversión de zona horaria

Ver §5.6. Orden correcto: localizar → convertir → asignar fecha operativa → aplicar calendario.

## 10.2. Calendario de referencia validado — PREVIOUSLY VALIDATED DATA POLICY

```
Calendario:  CME_Equity  (pandas_market_calendars, clase CMEEquityExchangeCalendar)
Política:    hybrid_observed_plus_calendar
```

**Política híbrida, y ésta es la parte importante:**

> El calendario se usa para **clasificar** (`día de trading` / `fin de semana` / `feriado`) y para **detectar cierres anticipados**.
> **Nunca para excluir un día que tiene datos observados.**
> **Los datos observados son la evidencia principal; el calendario es la referencia.**

**Nota de alcance honesta, heredada:** `pandas_market_calendars` es un paquete de terceros mantenido que **modela** el calendario publicado por CME. **No es una consulta en vivo a una API oficial de CME Group.** Debe documentarse así.

## 10.3. Cierres anticipados — CONFIRMED DATA FACT

Existen cierres anticipados y son verificables con **dos evidencias independientes**:

```
(a) la fecha está declarada como cierre anticipado en el calendario oficial versionado
(b) la cobertura observada calza EXACTO con el patrón de cierre anticipado
```

Resultado de la verificación anterior:

```
66 fechas de cierre anticipado declaradas por el calendario en 2020-01-01 .. 2026-12-31
40 fechas verificadas por AMBAS evidencias (patrón de datos + calendario)
   -> hora de cierre declarada 13:00 ET en las 40/40
   -> desde 2020-02-17 hasta 2026-07-03
```

Corresponden a los feriados de cierre anticipado estándar de CME: **MLK Day, Presidents Day, Memorial Day, Juneteenth, 3–5 de julio según caiga el 4, Labor Day, Thanksgiving.**

**Regla heredada:** si el calendario marca cierre anticipado pero los datos **no** calzan con el patrón, la jornada **no** se clasifica como cierre anticipado — cae a "sin determinar". No se asume el cierre anticipado sin verificación empírica.

> **El conteo de 40 y el patrón exacto de barras dependen de la ventana histórica** (§7.2). La *existencia* de cierres anticipados, la lista de feriados afectados y la hora de cierre 13:00 ET son hechos del calendario, independientes de la ventana.

## 10.4. La ventana 04:30–16:00 NO era un óptimo empírico — CONFIRMED DATA FACT

Registro explícito del proyecto anterior:

```yaml
window:
  start_time: "04:30:00"
  end_time:   "16:00:00"
  expected_minutes: 691
  is_empirically_optimal: false
  rationale: "conventional_operational_limit_not_data_driven_optimum"
```

Y la evidencia que lo sustenta:

> *"04:30 y 16:00 NO son quiebres estructurales óptimos (verificado empíricamente: la curva de cobertura es continua y suave antes de 04:30 y no cae hasta el corte de mantenimiento ~17:01). Se conservan por continuidad metodológica con etapas ya diseñadas alrededor de 691 barras/día."*

Alternativas documentadas y **no** implementadas: `08:30–16:00` (451 min) y `09:30–16:00` (391 min).

> **DataRoad NO debe heredar 04:30–16:00 como decisión.** Es una convención de un proyecto anterior, declarada por sus propios autores como no derivada de los datos. La única estructura temporal con respaldo empírico heredado es el **corte de mantenimiento CME (~17:00–18:00 ET)**, localizado independientemente en dos etapas del pipeline anterior y usado como una de las tres anclas de la validación de zona horaria (§5.2).

## 10.5. Taxonomía de jornadas — PREVIOUSLY VALIDATED DATA POLICY

Lo verdaderamente transferible no son las cifras, sino **la exigencia de distinguir categorías que no deben mezclarse**:

| Categoría | Qué es | Por qué no se puede mezclar con las demás |
|---|---|---|
| **Sesión completa** | Cobertura esperada íntegra | Población de referencia |
| **Cierre anticipado legítimo** | Sesión más corta **por calendario**, verificada por dos evidencias | Es un día válido, no un defecto |
| **Cobertura parcial** | Hay datos, pero menos de lo esperado, sin causa identificada | Puede ser dato faltante o sesión especial: **no se sabe** |
| **Ausencia de datos** | Cero barras en una fecha que el calendario marca como día de trading | Distinto de "no era día de trading" |
| **Fin de semana / feriado** | Cero barras esperadas | No es una anomalía |
| **Gap** | Discontinuidad documentada dentro de un rango | Puede abarcar varias fechas y cortar sesiones a mitad |

**Lección concreta heredada:** una cifra agregada de "N jornadas no completas" es **engañosa**, porque mezcla dos poblaciones radicalmente distintas — *fechas con datos parciales* y *fechas sin ningún dato*. Estas últimas aportan **cero filas** por definición. En el proyecto anterior, confundirlas llevó a interpretaciones erróneas del efecto de cada cambio del pipeline.

**Segunda lección:** un gap que corta una sesión a mitad genera **fechas con cobertura parcial en sus extremos** que no son errores de clasificación: son la sesión que el gap interrumpe. Deben trazarse al identificador del gap, no clasificarse como anomalía independiente.

**Tercera lección:** ninguna jornada debe desaparecer silenciosamente. Todas las fechas del rango deben quedar en una tabla de auditoría con un estado explícito y trazable, incluso las que no aportan ninguna fila.

## 10.6. Los regímenes intradía NO se heredan

El proyecto anterior definía cinco regímenes horarios (`Early_Premarket`, `Premarket`, `Opening`, `Regular`, `Closing`) con límites concretos en minutos.

> **INVALIDATED HISTORICAL PRACTICE para DataRoad — por alcance, no por defecto de implementación.**
> Esa segmentación era una **hipótesis de segmentación temporal orientada a modelado**, declarada como tal por sus autores ("no deben presentarse como regímenes económicos demostrados"). Depende íntegramente de la ventana 04:30–16:00, que no se hereda.
> **DataRoad no debe adoptar esos límites.** Si necesita segmentar el día, debe derivar la segmentación de los datos observados.

Lo único transferible de aquella experiencia es una **regla de implementación**: ningún minuto puede quedar asignado a un segmento por una **ruta por defecto silenciosa**. El bug histórico más costoso de esa etapa fue exactamente ése — una barra que no calzaba con ninguna condición recibía el valor por defecto y contaminaba todas las estadísticas de ese segmento. La construcción de la tabla de asignación debe **fallar en tiempo de construcción** si algún minuto queda sin asignar.

## 10.7. Otras observaciones de calendario abiertas — UNRESOLVED DATA QUESTION

- **Patrón recurrente ~16:20–16:30 ET, concentrado en 2019–2021.** Documentado como `s01_pattern_1620_1630`, `blocking: false`. **Nunca se investigó a fondo** porque caía fuera de la ventana histórica. **Para DataRoad, si trabaja sobre las ~24 h, este patrón SÍ está dentro de alcance y sigue sin explicación.**
- **Discrepancia entre calendarios `CME_Equity` y `"CME Globex Equity"`**: difieren en **1 día (2025-01-09)**. No se revalidó; se usó `CME_Equity`.
- **Jornadas sin causa determinada:** el proyecto anterior dejó abiertas jornadas que el calendario marca como día de trading pero cuya cobertura observada no calza con ningún patrón conocido (ni gap documentado, ni cierre anticipado). Las cifras concretas dependen de la ventana histórica y no se transportan; **el problema sí**.

---

# 11. Hold-out protegido

> ## HOLDOUT GOVERNANCE — sección normativa

## 11.1. Definición — recuperada literalmente

```yaml
estado:          LOCKED
dataset_version: IRIS-MNQ-SRC-20191223_20260731-e382a75ac222

frontera:
  inicio: "2025-06-23 00:00:00"
  fin:    "2026-07-31 20:10:00"
  fin_nota: "Último timestamp del snapshot congelado. No es una fecha de
             calendario elegida; es el límite real de los datos disponibles."

contratos_incluidos:
  - 22_mnq_09_25.Last.txt
  - 23_mnq_12_25.Last.txt
  - 24_mnq_03_26.Last.txt
  - 25_mnq_06_26.Last.txt
  - 26_mnq_09_26.Last.txt   # incompleto

proporcion_aprox:
  filas_holdout:      392.553
  filas_investigacion: 1.937.230
  filas_totales:       2.329.783
  pct_holdout:         16,9 %
  pct_investigacion:   83,1 %
```

**Nota de alcance importante:** estos conteos se calculan sobre el **snapshot crudo completo** (~24 h por jornada), no sobre ninguna ventana operativa recortada. Son directamente aplicables a DataRoad.

## 11.2. Razón de la frontera — recuperada literalmente

> Se eligió esta frontera porque **(a)** reserva aproximadamente el 16,9 % del snapshot actual, **(b)** cubre más de un año calendario de datos recientes, **(c)** **coincide con una transición de contrato SIN solapamiento temporal** entre `21_mnq_06_25` y `22_mnq_09_25` — frontera contractual limpia, a diferencia de las otras tres transiciones del snapshot que sí solapan — y **(d)** conserva aproximadamente el 83,1 % del snapshot para la fase de investigación.

**Verificable en los datos:** `21_mnq_06_25` termina el 2025-06-22 13:53 y `22_mnq_09_25` empieza el 2025-06-23 03:01. La frontera es limpia por contrato **y** por timestamp. Ningún contrato queda partido entre los dos conjuntos.

**Declaración explícita de lo NO afirmado, heredada literalmente:**

> *"No se afirma, porque no ha sido medido, que el hold-out contenga varios regímenes de mercado ni que tenga potencia estadística suficiente para ninguna conclusión: ambas propiedades quedan sin evaluar."*

## 11.3. Prohibiciones de acceso — normativo

```
NO debe usarse para exploración.
NO debe usarse para estadísticas descriptivas orientadas a decisiones.
NO debe usarse para selección metodológica.
NO debe usarse para tuning.
NO debe usarse para comparación de alternativas.
NO debe usarse para evaluación predictiva.
NO debe usarse para ninguna otra decisión de investigación.
```

**Permitido únicamente:**

> comprobaciones estructurales estrictamente necesarias (p. ej. integridad de esquema o de timestamps) y transformaciones deterministas **previamente especificadas y congeladas a partir del conjunto de investigación**, siempre que sus resultados **no se usen para modificar esas decisiones**.

**Obligaciones:**

- **Todo acceso al hold-out, del tipo que sea, debe quedar trazado en un registro de accesos.**
- Cualquier acceso fuera de los casos permitidos **requiere autorización explícita del usuario para ese acceso concreto**.

**Traducción directa a DataRoad.** El objetivo declarado de este repositorio — *"caracterizar estadísticamente series OHLCV y generar un informe empírico"* — es, casi en su totalidad, **exactamente lo que las prohibiciones impiden hacer sobre el hold-out**. La caracterización estadística de DataRoad debe ejecutarse sobre el **conjunto de investigación (1.937.230 filas, hasta 2025-06-22)**, y el informe empírico debe declarar ese alcance. Sólo la QA estructural (§4) es admisible sobre el período protegido, y aun así con trazado.

## 11.4. Exposición estructural pre-lock — declarada, no ocultada

**Existe exposición estructural previa al bloqueo. No es una omisión.** Antes de que la frontera quedara `LOCKED`, un escaneo estructural de sólo lectura recorrió los 27 archivos, **incluidos los 5 reservados hoy como hold-out**.

**Qué se hizo sobre el período del hold-out:**

| # | Acción ejecutada |
|---|---|
| 1 | Escaneo estructural completo: parseo de 6 campos por línea, detección de timestamps duplicados o fuera de orden dentro de archivo, validez de OHLC — **sin hallazgos de corrupción en estos 5 archivos** |
| 2 | Análisis de solapamientos contractuales entre archivos consecutivos, **incluido el solapamiento real M26→U26 (~10 días), que cae íntegramente dentro del hold-out** |
| 3 | Detección de las **3 barras con volumen extremo** — **las 3 dentro del período reservado**: 2 en `22_mnq_09_25` (2025-07-01 10:08 y 2025-07-15 13:52) y 1 en `24_mnq_03_26` (2026-01-27 03:44) |
| 4 | Conteo de filas, tamaño en bytes y primer/último timestamp por archivo — usados para **dimensionar** las alternativas de frontera, no para evaluarlas por contenido |

**Qué NO se hizo sobre el hold-out:**

```
Ningún análisis de retornos.
Ningún análisis de targets ni de labeling.
Ninguna construcción ni evaluación de señales.
Ninguna evaluación de estrategias.
Ningún análisis de predictibilidad.
Ninguna medición de desempeño.
```

**Clasificación:** QA pre-lock — comprobación de integridad estructural, no experimento.

**Implicación normativa, heredada literalmente:**

> *"Esta exposición previa no invalida ni modifica la frontera aprobada. A partir del estado LOCKED, cualquier acceso adicional al hold-out — incluida cualquier comprobación estructural nueva, aunque sea del mismo tipo que la ya realizada — requiere autorización explícita del usuario y debe quedar trazado."*

## 11.5. Consecuencias operativas para DataRoad

1. **El hold-out ya no es estructuralmente ciego.** Su integridad de esquema, sus solapamientos contractuales y sus tres anomalías de volumen son conocidos. Esto está documentado y **no debe negarse ni volver a "descubrirse"**.
2. **El hold-out sigue siendo ciego en todo lo demás** — distribución, dependencia, estacionalidad, colas, estabilidad. Ninguna de esas propiedades fue medida allí.
3. **Las tres anomalías de volumen no pueden investigarse** sin autorización explícita de acceso (§9.1).
4. **El solapamiento M26→U26 no puede analizarse** en profundidad sin autorización, aunque su existencia y magnitud ya estén registradas.
5. **Repetir la QA estructural sobre el hold-out —aunque sea idéntica a la ya hecha— requiere autorización nueva.** El estado `LOCKED` no otorga una licencia permanente por precedente.

---

# 12. Preguntas todavía abiertas

| Problema | Estado | Evidencia | Qué debe hacer DataRoad |
|---|---|---|---|
| **Confirmación documental de la zona horaria** | UNRESOLVED DATA QUESTION | UTC gana la comparación de 3 hipótesis por ~500× (score 2.0 vs 1087.1 / 1105.1), con `dst_consistency=0`. Pero `timezone_provider_confirmation: false`: no existe la config de exportación del proveedor en el repositorio anterior | Tratar UTC como **hipótesis fuertemente respaldada**, no como hecho. Reejecutar la validación sobre el snapshot actual de 27 archivos (la anterior corrió sobre 26). Documentar la evidencia como empírica en todo output. Buscar la config de exportación de NinjaTrader si es accesible |
| **Semántica del timestamp (inicio vs cierre de barra)** | UNRESOLVED DATA QUESTION | `timestamp_semantics: unknown_not_confirmed`. Se buscó evidencia en dos etapas independientes sin éxito. Ninguna barra fue desplazada | **No desplazar ninguna barra.** Declarar la ambigüedad en el informe empírico. Documentar qué conclusiones cambiarían bajo cada convención. Es el punto que más silenciosamente puede invalidar afirmaciones sobre disponibilidad de información |
| **`price_type = Last`** | UNRESOLVED DATA QUESTION | Inferido **únicamente** del sufijo `.Last.txt`. Sin confirmación del proveedor | Registrarlo como inferencia. No presentarlo como propiedad verificada |
| **Tres barras de volumen extremo** | UNRESOLVED DATA QUESTION | 1.258.222 · 1.451.062 · 1.534.923 frente a un máximo habitual de 8.500–31.000. OHLC internamente válido en las tres | **No decidir si son errores o eventos reales.** Investigar antes de cualquier conclusión que dependa de `volume`; reportar resultados con y sin ellas. **Requiere autorización de acceso al hold-out** |
| **Semántica de las barras sin operaciones** | UNRESOLVED DATA QUESTION | Ninguna fuente documenta si el proveedor emite una barra con volumen 0 cuando no hubo operaciones, o simplemente **omite** el minuto. Los archivos no tienen volumen vacío ni negativo, pero eso no distingue ambos casos | **Determinarlo explícitamente.** Distinguir tres situaciones que se ven idénticas al agregar: minuto ausente / barra con volumen 0 / barra rellenada. La conclusión cambia la interpretación de toda medida de actividad y de todo cálculo de cobertura |
| **Causa del patrón de solapamiento en 3 de 26 transiciones** | UNRESOLVED DATA QUESTION | 23 handoffs limpios frente a 3 solapamientos de ~8–10 días. Patrón inconsistente con el resto del snapshot, **nunca explicado** | Documentarlo como característica de la fuente. No asumir que es política uniforme del proveedor. Considerar que puede repetirse en exportaciones futuras |
| **Gap de precio en el momento del roll** | UNRESOLVED DATA QUESTION | La resolución previa cubrió la **selección de contrato**, explícitamente **no** el ajuste de nivel ni el análisis de gap de precio | Medirlo si la caracterización lo requiere. **No aplicar back-adjustment automático** (§7.1, regla 12) |
| **Patrón recurrente ~16:20–16:30 ET (2019–2021)** | UNRESOLVED DATA QUESTION | Documentado como `blocking: false`; nunca investigado porque caía fuera de la ventana histórica | **Está dentro de alcance si DataRoad analiza las ~24 h.** Caracterizarlo |
| **Jornadas con cobertura sin causa determinada** | UNRESOLVED DATA QUESTION | Fechas que el calendario marca como día de trading, con cobertura que no calza con ningún patrón conocido (ni gap ni cierre anticipado). Cifras dependientes de la ventana histórica, no transportadas | Reclasificarlas bajo el alcance de DataRoad. Mantener la categoría "sin determinar" en vez de forzar una etiqueta |
| **Discrepancia `CME_Equity` vs `"CME Globex Equity"`** | UNRESOLVED DATA QUESTION | Difieren en 1 día: **2025-01-09**. No revalidada | Verificar cuál corresponde a MNQ, o documentar el uso de una con la discrepancia declarada |
| **Ausencia de chequeo automatizado de solapamiento entre archivos** | UNRESOLVED DATA QUESTION | El módulo anterior sólo detecta duplicados exactos por `(timestamp, contract)`. La verificación de solapamiento fue manual y su resultado ("0 solapamientos") quedó **superado** por la actualización de fuente | Implementarlo dentro de la QA estructural. **No dar por válida la afirmación histórica** |
| **Comparabilidad de métricas absolutas a lo largo del snapshot** | UNRESOLVED DATA QUESTION | El nivel de precio creció sustancialmente en el período (cifra histórica, no transportada). Cualquier umbral en puntos absolutos selecciona preferentemente el tramo final | Preferir métricas normalizadas por nivel de precio. Medir el crecimiento del nivel de forma independiente si se necesita la cifra |

---

# 13. Prácticas históricas invalidadas

| Problema histórico | Qué ocurrió | Estado actual |
|---|---|---|
| **Gap interno M23 (~260 h, abr-2023)** | Se detectó un salto de 260 h 15 min dentro de `13_mnq_06_23.Last.txt` (2023-04-05 18:03 → 2023-04-16 14:18), 10 jornadas sin datos. Se clasificó `unconfirmed` y quedó abierto como anomalía estructural | **RESOLVED HISTORICAL ISSUE.** El archivo fue reemplazado por una versión más completa (87.931 filas vs 78.856). El salto máximo dentro de M23 es ahora ~57 h (fin de semana ordinario). **El gap NO existe en el snapshot actual. No reabrir** |
| **Gap de transición H25→M25 (~15 d 19 h, mar–abr 2025)** | Se detectó un vacío entre el fin de H25 (2025-03-21 13:30) y el inicio de M25 (2025-04-06 08:42), 15 jornadas sin datos. Clasificado `unconfirmed` | **RESOLVED HISTORICAL ISSUE.** Ambos archivos fueron reemplazados: ahora **se solapan** (H25 hasta 2025-03-22, M25 desde 2025-03-13). Era **un artefacto de exportación incompleta**, no un fenómeno de mercado. Ese mismo solapamiento es el que permite confirmar el rollover. **No reabrir** |
| **Calendario NASDAQ aplicado a MNQ** | Se usó `mcal.get_calendar("NASDAQ")` para un futuro de CME Globex, comprometiendo feriados, cierres anticipados y jornadas especiales | **INVALIDATED HISTORICAL PRACTICE.** Sustituido por `CME_Equity` con política híbrida (calendario como referencia, datos observados como evidencia principal). **No reintroducir un calendario de renta variable para un futuro de CME** |
| **Eliminación rígida de jornadas por conteo de barras** | 80 jornadas eliminadas sin clasificar por no tener el conteo esperado. Produjo sesgo de selección, pérdida de cierres anticipados legítimos y reducción artificial de escenarios extremos | **INVALIDATED HISTORICAL PRACTICE.** Sustituido por clasificación explícita de **todas** las fechas del rango, con estado trazable, sin eliminar ninguna. **Nunca excluir una jornada sólo por conteo de barras** |
| **Mezcla de contratos en fechas de rollover** | La serie podía contener más de una barra por minuto en fechas con solapamiento, porque no se seleccionaba contrato | **INVALIDATED HISTORICAL PRACTICE, ya resuelta.** Sustituida por la política de un único contrato por fecha, sin mezclar ni promediar, confirmada por volumen, irreversible y con trazabilidad de filas descartadas (§7). **No volver a construir una serie sin selección de contrato** |
| **Filtrado temporal antes de completar la conversión de zona horaria** | Se filtraron fechas de mercado antes de localizar y convertir | **INVALIDATED HISTORICAL PRACTICE.** Orden correcto en §5.6 |
| **Asignación de segmento por ruta por defecto silenciosa** | Una barra que no calzaba con ninguna condición recibía el valor por defecto, contaminando todas las estadísticas de ese segmento | **INVALIDATED HISTORICAL PRACTICE.** La construcción de la tabla de asignación debe **fallar** si algún minuto queda sin asignar (§10.6) |
| **Estadísticos seriales calculados a través de huecos** | Autocorrelación y tests de dependencia formaban pares entre observaciones de segmentos consecutivos distintos | **INVALIDATED HISTORICAL PRACTICE.** Todo estadístico serial debe respetar límites de segmento (§9.3) |
| **Detección de "0 solapamientos entre archivos"** | Auditoría manual sobre el corpus de 26 archivos concluyó 0 solapamientos | **SUPERADO POR LOS DATOS.** El snapshot actual tiene 3 solapamientos reales. **No citar la afirmación histórica** |
| **Reutilización silenciosa de artefactos** | El código podía cargar un artefacto existente en lugar de regenerarlo tras modificar la lógica: el código visible y el artefacto real podían pertenecer a versiones distintas | **INVALIDATED HISTORICAL PRACTICE.** Sustituida por manifest autoritativo con hash de fuente + módulo + config normalizada, `pipeline_version` y `force_rebuild` explícito. DataRoad debería adoptar el mismo patrón (§15.2) |
| **Documentación que describe operaciones no ejecutadas** | La documentación histórica afirmaba conversiones de zona horaria, filtrados y limpiezas que el código **no** realizaba | **INVALIDATED HISTORICAL PRACTICE.** Regla heredada: **la documentación de una etapa describe únicamente lo que esa etapa ejecuta.** |

---

# 14. Información que deliberadamente NO se heredó

## 14.1. Resultados estadísticos históricos — excluidos por diseño

Los proyectos anteriores produjeron una cantidad sustancial de análisis estadístico sobre este mismo dataset. **Nada de eso se transporta a DataRoad como evidencia vigente.**

Excluidos deliberadamente:

```
autocorrelación (ACF) y sus resúmenes por segmento
Ljung-Box y tests de dependencia serial
ARCH-LM y diagnósticos de heterocedasticidad condicional
volatility clustering ya medido
análisis por regímenes horarios
distribuciones históricas de OHLCV
estadísticos por año
diagnósticos de inestabilidad temporal / cambio de distribución
resúmenes de ventanas rodantes
validez de ventanas por horizonte
umbrales de detección de outliers y sus resultados
correlaciones entre variables OHLCV
estacionalidad minuto a minuto ya calculada
```

**Justificación.** Todos esos resultados se calcularon sobre una ventana operativa (04:30–16:00) que DataRoad no adopta, con una segmentación intradía que DataRoad no hereda, y en un contexto orientado a modelado predictivo. Transportarlos convertiría decisiones de alcance ajenas en conclusiones vigentes.

> **Existió un análisis estadístico histórico sobre este dataset, pero no se transporta como evidencia vigente.** DataRoad debe regenerar cualquier resultado de forma independiente y bajo su propio alcance.

## 14.2. Qué sí se conservó de esa etapa, y por qué

Tres elementos de aquel trabajo **no son resultados estadísticos sino requisitos de implementación**, y por eso sí se heredan:

| Elemento | Por qué no es un resultado estadístico |
|---|---|
| Los estadísticos seriales deben respetar límites de segmento (§9.3) | Es una condición de corrección del cálculo, no una medición |
| Las métricas en puntos absolutos no son comparables a lo largo del snapshot (§9.2) | Es una consecuencia de la estructura del dato, no una conclusión sobre el mercado |
| Ninguna asignación puede ocurrir por ruta por defecto (§10.6) | Es una invariante de construcción |

## 14.3. Contenido de Machine Learning — fuera de alcance del repositorio

Descartado íntegramente por no corresponder al objetivo de `ohlcv_dataroad`:

```
targets DIR / BAR / OPC, su codificación de clases y su mapping
thresholds por percentil (p40/p50/p60) y por régimen
configuración de barreras TP / SL
horizontes predictivos (30 / 60 / 90 minutos) y la preferencia histórica por 60
lookbacks de secuencia (30 / 60 / 90)
biblioteca de features causales y familias de features
selección de features (Mutual Information, importancia, ablación, PCA)
diseño walk-forward, folds, validación interna, purging y embargo
catálogo de modelos (lineales, árboles, boosting, MLP, CNN1D, GRU, LSTM, TCN)
métricas predictivas (macro F1, balanced accuracy, log loss, calibración)
manejo de desbalance de clases y class weights
evaluación financiera, P&L, costes, Sharpe, drawdown, backtesting
reglas de entrada, ejecución, sizing y política de ambigüedad intrabar
regímenes intradía como segmentación de modelado
```

**Nota de matiz.** Dos elementos de esa lista tocan los datos y no el modelado, y por eso aparecen en este documento reformulados como conocimiento de datos, no como decisiones de ML:

- La **ambigüedad intrabar** — con OHLCV de 1 minuto no puede determinarse el orden en que ocurrieron `high` y `low` dentro de la barra. Aquí se registra como **limitación estructural del dato** (§16.4), no como política de backtest.
- La **ventana 04:30–16:00** — aparece sólo para advertir explícitamente que **no debe heredarse** (§10.4).

## 14.4. Limitaciones estructurales del dato — sí se heredan

No son decisiones de ML: son propiedades irreducibles de un OHLCV de 1 minuto, y aplican a cualquier análisis. Con este dataset **no puede conocerse**:

```
el orden exacto en que ocurrieron high y low dentro de la barra
bid / ask
spread real
profundidad de mercado
dirección agresora de las operaciones
secuencia de operaciones intraminuto
número de operaciones por barra   (volume = contratos operados, NO número de trades)
VWAP real
```

> **Limitación estructural, no corregible con código.** Debe declararse en el informe empírico, y ninguna técnica que requiera estos datos debe simularse como si fueran observables.

---

# 15. Fuentes y trazabilidad

## 15.1. Documentos consultados para construir esta memoria

| Fuente | Qué aportó |
|---|---|
| `dataset_manifest.yaml` | Identidad del snapshot, hash, 27 archivos con sus hashes/filas/rangos individuales, contrato incompleto, esquema de `version_id` |
| `00_inventario_estructural_data_source.md` | QA estructural de los 27 archivos, los 3 solapamientos a nivel de archivo, las 3 barras de volumen extremo |
| `holdout.yaml` | Frontera, estado `LOCKED`, contratos, proporciones, razón de la elección, prohibiciones de acceso, exposición pre-lock |
| `S00_v2_report.md` | Validaciones de ingestión, los dos gaps extraordinarios y su resolución posterior, pendientes de zona horaria y semántica, actualización de fuente |
| `S01_v2_report.md` | Validación comparativa de zona horaria, tratamiento de DST, resolución de rollover, regla de respaldo 11, verificación de cierres anticipados, taxonomía de jornadas |
| `intraday_config.yaml` | Parámetros de rollover, política de calendario, declaración de que la ventana no es un óptimo empírico, hipótesis de zona horaria, `timestamp_semantics` |
| `01_CURRENT_DECISIONS.md` | Alcance de los datos, datos no disponibles, política de contratos y rollover, decisiones no vigentes |
| `02_KNOWN_ISSUES_AND_INVALIDATED_RESULTS.md` | Problemas por etapa con su estado, prácticas invalidadas, limitaciones estructurales, resolución de los gaps |

## 15.2. Patrón de trazabilidad heredado — PREVIOUSLY VALIDATED DATA POLICY

El proyecto anterior desarrolló un patrón de gobernanza de artefactos que **funcionó** y que conviene reutilizar:

```
version_id determinista respecto del contenido (nunca datetime.now())
hash SHA-256 del snapshot completo + hash por archivo individual
manifest autoritativo con hash de: fuente + módulo + config normalizada
pipeline_version explícito
force_rebuild explícito
git commit y estado dirty como METADATA de procedencia, no como invalidante
detección de staleness end-to-end: si cambia cualquier hash upstream, no se reutiliza
escritura atómica: archivo temporal -> relectura y verificación -> reemplazo
data source READ ONLY
```

Dos aprendizajes concretos de su implementación:

1. **La detección de staleness debe usarse, no sólo existir.** Hubo un caso en que la función de comparación estaba escrita pero el manifest de una etapa upstream no participaba en la comparación, de modo que un cambio upstream no invalidaba el artefacto downstream.
2. **La verificación por relectura debe ser de equivalencia lógica, no byte a byte.** La metadata interna de los formatos binarios varía entre escrituras. Un caso real: una columna de tipo lista se releía como array y rompía la comparación de igualdad; se resolvió serializándola explícitamente.

## 15.3. Regla de precedencia para este documento

Cuando exista una contradicción, prevalecen en primer lugar los datos reales del snapshot vigente y sus artefactos autoritativos (manifest, hashes y configuraciones vigentes). Este documento resume el estado consolidado conocido y debe utilizarse como punto de entrada, pero cualquier contradicción material con una fuente primaria debe verificarse antes de modificarla o descartarla.

---

# 16. Resumen ejecutivo para futuras sesiones

## 16.1. Lo que ya sabemos y no hace falta volver a establecer

1. **El snapshot está identificado y es reproducible.** 27 archivos, 2.329.783 filas, hash conocido, `version_id` determinista, hash por archivo. Rango real **2019-12-23 → 2026-07-31**.
2. **La integridad estructural está confirmada.** 0 líneas malformadas, 0 duplicados dentro de archivo, 0 desorden temporal, 0 OHLC inválido, 0 volumen negativo, 0 filas rechazadas.
3. **Los dos grandes gaps históricos ya no existen.** Eran artefactos de exportación incompleta y se resolvieron al reemplazar los archivos fuente. **No reabrirlos.**
4. **La cobertura contractual está mapeada.** 27 contratos, secuencia H/M/U/Z completa, 26 transiciones, 23 limpias, 3 con solapamiento real identificado y cuantificado.
5. **Existe una política de rollover validada y verificada por regresión** — un contrato por fecha, sin mezclar, sin promediar, confirmada por volumen, irreversible, trazable, sin back-adjustment automático.
6. **UTC está fuertemente respaldado empíricamente** por una comparación programática de tres hipótesis con un margen de ~500×, y **DST es seguro** para este dataset.
7. **Los cierres anticipados son reales y verificables** con doble evidencia (calendario oficial + patrón de datos), y corresponden a los feriados estándar de CME con cierre 13:00 ET.
8. **El calendario correcto es de futuros de CME**, no de renta variable, y debe usarse en política híbrida: nunca excluye un día con datos observados.

## 16.2. Lo que NO debe darse por hecho

1. **La zona horaria NO está confirmada documentalmente.** Es una inferencia fuerte, no un hecho.
2. **La semántica del timestamp NO se conoce.** Inicio o cierre de barra: sin determinar. Ninguna barra ha sido desplazada, y así debe seguir.
3. **Tres barras de volumen extremo siguen sin explicación** — y las tres están dentro del hold-out.
4. **No se sabe cómo representa la fuente un minuto sin operaciones.**
5. **La ventana 04:30–16:00 y los cinco regímenes intradía NO se heredan.** Fueron convenciones de un proyecto anterior, declaradas por sus propios autores como no derivadas de los datos.
6. **Ningún resultado estadístico histórico se hereda.** DataRoad regenera todo.

## 16.3. Lo primero que DataRoad debería hacer

| # | Acción | Por qué |
|---|---|---|
| 1 | **Verificar el hash del snapshot** contra `e382a75ac222...` | Confirma que se trabaja sobre el mismo dataset del que habla esta memoria. Si difiere, **casi nada de este documento aplica sin revalidación** |
| 2 | **Fijar y declarar el alcance temporal** (¿las ~24 h completas o una ventana?) | Determina cuáles de las políticas heredadas requieren recalibración (§7.2) |
| 3 | **Excluir el hold-out del análisis** desde el primer paso | La caracterización estadística sobre el período protegido está prohibida (§11.3). El conjunto de investigación son 1.937.230 filas hasta 2025-06-22 |
| 4 | **Reejecutar la QA estructural** sobre el conjunto de investigación, incluyendo el chequeo de solapamiento entre archivos que nunca se automatizó | Confirma la integridad bajo el alcance propio y cierra el hueco de §4.4 |
| 5 | **Reejecutar la validación de zona horaria** sobre el snapshot de 27 archivos | La corrida heredada usó 26 archivos |
| 6 | **Determinar la semántica de las barras sin operaciones** antes de calcular cualquier medida de cobertura o actividad | Distingue minuto ausente / volumen 0 / barra rellenada, que se ven idénticos al agregar |
| 7 | **Aplicar la política de rollover recalibrada** antes de construir cualquier serie continua | Sin selección de contrato, las fechas de solapamiento tienen más de una barra por minuto |
| 8 | **Declarar en el informe empírico** las cuatro inferencias no confirmadas (zona horaria, semántica del timestamp, price type, semántica de barras sin trades) | Son las que más silenciosamente pueden invalidar conclusiones |

## 16.4. Frontera del conocimiento — lo que este dataset no puede responder

Con OHLCV `Last` de 1 minuto, y **con independencia de cualquier técnica**, no es observable: el orden intrabarra de `high` y `low`, el bid/ask, el spread, la profundidad, la dirección agresora, la secuencia de operaciones, el número de operaciones por barra (`volume` son contratos, no trades) ni el VWAP real.

**Limitación estructural, no corregible con código.** Cualquier análisis que la ignore está infiriendo algo que el dato no contiene.

---

**MNQ PRIOR DATA KNOWLEDGE CONSOLIDATED — NO EMPIRICAL ANALYSIS EXECUTED.**
