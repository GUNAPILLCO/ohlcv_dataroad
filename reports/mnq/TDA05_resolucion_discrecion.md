# TDA-05 — Resolución efectiva y discreción del retorno de 1 minuto

**Etapa:** `docs/methodology/Tsay_OHLCV_analysis_roadmap.md` § TDA-05
**Depende de:** TDA-02 (`PASS_WITH_OPEN_QUESTIONS`), TDA-04 (`PASS_WITH_OPEN_QUESTIONS`)
**Alcance de datos:** exclusivamente los artefactos de TDA-04 (`tda04_variables_1m.parquet`, `tda04_return_validity_mask.parquet`) y la máscara de barras inactivas de TDA-02 (`tda02_barra_inactiva_mask.parquet`). Ningún archivo de `data/raw/` ni de `holdout_files` fue abierto en esta etapa.
**Evidencia reproducible:** `reports/mnq/TDA05_resolucion_global.csv`, `TDA05_resolucion_por_hora.csv`, `TDA05_resolucion_por_anio.csv`, `TDA05_resolucion_anio_hora.csv`, `data/interim/mnq/tda05_tick_variables.parquet`, `reports/mnq/TDA05_histograma_ticks.png`. Todo generado por `python -m ohlcv_dataroad.ingest.run_tda05`.

> Esta etapa NO estudia el perfil minuto-a-minuto completo, NO calcula momentos/cuantiles/QQ-plots de `r_1m`, NO busca dependencia, NO repite el análisis a 5/10 minutos, NO retoma TH10 y NO introduce ninguna segmentación de sesión. Mide, por hora NY y por año, cuán discreta es la variable retorno debido al tick del contrato.

---

## 1. Objetivo

Determinar en qué medida `r_1m` (TDA-04) es una variable efectivamente discreta por el tick mínimo de MNQ (0.25 puntos), globalmente y desagregado por hora del día (America/New_York) y por año — sin buscar predictibilidad, señales ni segmentaciones operativas.

## 2. Inputs utilizados

- `data/interim/mnq/tda04_variables_1m.parquet` — `close`, `high`, `low`, `r_1m` (fuente de verdad de TDA-04, no recalculada desde raw).
- `data/interim/mnq/tda04_return_validity_mask.parquet` — `r_1m_valid`, la máscara que decide qué barras tienen una `t-1` comparable.
- `data/interim/mnq/tda02_barra_inactiva_mask.parquet` — categoría exacta de cada barra (`ACTIVA`/`FLAT_AISLADA`/`CANDIDATO_FORWARD_FILL`).
- `configs/mnq_snapshot.yaml`, `instrument_spec.tick_size = 0.25`.

## 3. Definición matemática de cada métrica y tratamiento de unidades

| Métrica | Definición | Unidad |
|---|---|---|
| `delta_close_points` | `close_t − close_{t-1}`, con `t-1` EXACTAMENTE la barra que TDA-04 usó para `r_1m` (`close.shift(1)` + máscara `r_1m_valid` aplicada explícitamente — nunca `close.shift(1)` a secas) | puntos |
| `delta_close_ticks` | `delta_close_points / tick_size` | ticks (adimensional, entero) |
| `range_points` | `high_t − low_t` (no depende de barra anterior) | puntos |
| `zero_fraction` | `P(r_1m = 0)` sobre las filas válidas del grupo | fracción |
| `sigma_delta_close_points` | `std(delta_close_points)`, `ddof=1` | puntos |
| `tick_to_sigma_points` | `tick_size / sigma_delta_close_points` | adimensional (puntos/puntos) |
| `sigma_r_1m` | `std(r_1m)`, `ddof=1` | adimensional |
| `tick_return_representative` | `ln((C_repr + tick_size)/C_repr)`, `C_repr` = Close mediano del grupo | adimensional |
| `tick_to_sigma_return` | `tick_return_representative / sigma_r_1m` | adimensional (retorno/retorno) |
| `median_range_points` | mediana de `range_points` del grupo | puntos |
| `tick_to_median_range` | `tick_size / median_range_points` | adimensional (puntos/puntos) |

**Por qué NO se divide `tick_size` (puntos) entre `std(r_1m)` (adimensional) directamente**: mezclaría magnitudes incompatibles. Se calculan `tick_to_sigma_points` (ambos en puntos) y `tick_to_sigma_return` (ambos adimensionales, usando el Close mediano del grupo como nivel de precio representativo — el equivalente de un tick en unidades de retorno depende del nivel de precio, que cambia con el tiempo) **por separado**, y se compara su interpretación práctica (§6).

**Invariante verificada, no asumida**: para toda fila válida, `delta_close_ticks` debe ser entero dentro de `1×10⁻⁶` (misma tolerancia que TDA-00 usa para la grilla de un precio individual). **Verificado sobre el conjunto de investigación real: 0 inconsistencias** — la ejecución completa `run_tda05.py` no lanzó `TickGridInconsistencyError`.

**Rango central fijado a priori** (sección 11 de la tarea: "elige UNA definición sencilla; fíjala antes de interpretar los resultados"): `CENTRAL_RANGE_TICKS = 5` — un rango simétrico, redondo, centrado en cero, definido en el código **antes** de ejecutar el análisis sobre datos reales, no elegido después de ver los resultados. **Aclaración**: ±5 ticks no cubre la mayoría de las barras — en el conjunto de investigación real cubre `prop_abs_le_5_ticks = 39,98 %` (§4) de las barras válidas, una minoría sustancial pero no dominante; se eligió por ser un rango simple y redondo, no por representar la mayor parte de los datos.

---

## 4. Resultado global

| Indicador | Valor |
|---|---:|
| `n` (retornos válidos) | 1.914.530 |
| `zero_fraction` | **4,2935 %** (bootstrap 95 % CI: [4,1995 %, 4,3790 %]) |
| `sigma_delta_close_points` | 5,6521 puntos |
| `tick_to_sigma_points` | **0,0442** (bootstrap 95 % CI: [0,0422, 0,0463]) |
| `tick_to_sigma_return` | 0,0415 |
| `median_range_points` | 4,50 puntos |
| `tick_to_median_range` | 0,0556 |
| `median_abs_ticks` | 7,0 ticks |
| `prop_0_ticks` | 4,29 % |
| `prop_abs_1_tick` | 8,28 % |
| `prop_abs_le_2_ticks` | 20,38 % |
| `prop_abs_le_5_ticks` | 39,98 % |
| Valores distintos de `r_1m` en rango central (±5 ticks) | 299.178 |
| Valores distintos de `delta_close_ticks` en el mismo rango | 11 (los enteros −5..+5) |

**Lectura**: globalmente, MNQ **no** es una variable fuertemente discreta. El tick (0,25 puntos) es sólo ~4,4 % de la dispersión típica de 1 minuto (`tick_to_sigma_points = 0,044`) — muy distinto de los episodios clásicos de Tsay (tick 1/8 de IBM: 67 % de barras sin cambio). Aquí, el movimiento mediano ya es de 7 ticks, y solo el 4,3 % de las barras tiene retorno exactamente cero. El contraste entre 299.178 valores decimales distintos de `r_1m` y sólo 11 valores enteros posibles de `delta_close_ticks`, **en el mismo conjunto de barras**, ilustra exactamente el fenómeno que motiva esta etapa: `r_1m` parece continuo porque el nivel de precio (`close_{t-1}`) cambia de fila a fila, pero el movimiento subyacente es siempre uno de 11 escalones enteros.

---

## 5. Resultado por hora NY

`TDA05_resolucion_por_hora.csv` (24 filas, `n` entre 1.343 y 84.720). Extremos:

| hora NY | n | `zero_fraction` | `tick_to_sigma_points` | `prop_0_ticks` |
|---:|---:|---:|---:|---:|
| 23 | 84.459 | 7,44 % | 0,0963 | 7,44 % |
| 00 | 84.455 | 7,71 % | 0,0955 | 7,71 % |
| 22 | 83.670 | 6,56 % | 0,0835 | 6,56 % |
| … | | | | |
| 10 | 83.611 | 1,52 % | 0,0245 | 1,52 % |
| 09 | 84.122 | 2,35 % | 0,0265 | 2,35 % |
| 15 | 80.794 | 1,99 % | 0,0315 | 1,99 % |

**Las horas 22, 23 y 00 NY (noche/madrugada, baja liquidez) son consistentemente las MÁS discretas**; las horas 09–15 NY (sesión diurna de mayor liquidez) son las MENOS discretas, con `tick_to_sigma_points` 3-4 veces menor. Ninguna hora, ni siquiera la más discreta (00 NY, `tick_to_sigma_points = 0,0955`), se acerca a un régimen donde el tick domine el movimiento (eso requeriría un valor del orden de 0,3–0,5 o más, con `zero_fraction` de varias decenas por ciento).

**Caso especial — hora 17 NY**: `n = 1.343`, muy inferior al resto (~84.000). No es un artefacto de baja liquidez: la sesión cierra exactamente a las 17:00 NY (TDA-02), así que la hora "17" solo puede contener, como máximo, **una barra por jornada** (la del cierre) — con 1.420 jornadas en el conjunto de investigación, 1.343 es exactamente el orden de magnitud esperado. Se cita explícitamente para que no se interprete como una hora de mercado completa (sección 6 de la tarea).

## 6. Resultado por año NY

| año | n | `zero_fraction` | `tick_to_sigma_points` | `tick_to_sigma_return` | `median_range_points` |
|---:|---:|---:|---:|---:|---:|
| 2019 | 7.406 | 18,50 % | 0,2182 | 0,2184 | 0,75 |
| 2020 | 342.648 | 4,71 % | 0,0486 | 0,0425 | 4,25 |
| 2021 | 349.039 | 5,05 % | 0,0585 | 0,0567 | 3,75 |
| 2022 | 349.996 | 3,15 % | 0,0378 | 0,0385 | 6,00 |
| 2023 | 349.636 | 4,93 % | 0,0597 | 0,0553 | 3,75 |
| 2024 | 352.261 | 4,12 % | 0,0465 | 0,0465 | 4,75 |
| 2025 | 163.544 | 2,62 % | 0,0270 | 0,0245 | 7,50 |

**Estabilidad**: 2020–2024 son consistentes entre sí (`tick_to_sigma_points` entre 0,038 y 0,060, sin tendencia monótona clara — 2022 es el año MENOS discreto, 2021/2023 los más discretos de ese tramo, diferencias moderadas). **2019 muestra el `tick_to_sigma_points` más alto de la serie** (0,218, `zero_fraction = 18,5 %`), con **`n = 7.406`**, apenas 9 días de calendario (el conjunto de investigación empieza el 2019-12-23): es un año **PARCIAL**, y su evidencia tiene menor capacidad de generalización que la de un año completo — no se interpreta como un régimen de discreción distinto, pero tampoco se descarta como dato. **2025 también es un año parcial** (`n = 163.544`, cubre el conjunto de investigación hasta su corte, no un año calendario completo) y su cifra (`tick_to_sigma_points = 0,027`, la más baja de la serie) debe leerse con la misma reserva, aunque su tamaño muestral es sustancialmente mayor que el de 2019.

**Comparación de `tick_to_sigma_points` vs `tick_to_sigma_return` (sección 8 de la tarea)**: ambos cocientes, calculados por separado y en unidades distintas, llevan a la MISMA lectura práctica año por año — 2019 es el más discreto por ambas medidas (0,218 vs 0,218, prácticamente idénticos); 2022 y 2025 son de los menos discretos por ambas medidas. La correspondencia es cercana en los 7 años (diferencias de un dígito decimal, nunca un cambio de orden de magnitud ni de ranking cualitativo).

## 7. Forward-fill (TH09)

- Barras `FORWARD_FILL` **confirmadas** en TDA-02: **0**.
- Barras `CANDIDATO_FORWARD_FILL` (nunca confirmadas): 121.
- **Declaración explícita**: no existen barras confirmadas como `FORWARD_FILL` en el conjunto de investigación (TDA-02, §10.2). La comparación con/sin `FORWARD_FILL` que exige el método mínimo de TH09 no modifica el resultado: son la MISMA cifra (`zero_fraction = 4,2935 %` en ambos casos, por construcción — no hay ninguna fila que excluir).
- **`SENSITIVITY_ONLY`** (excluyendo los 121 `CANDIDATO_FORWARD_FILL`, nunca usada como resultado principal ni para STOP-5): `zero_fraction = 4,2880 %` — una diferencia de 0,0055 puntos porcentuales sobre 1,9 millones de filas, irrelevante en la práctica.

## 8. Histograma y masas principales en ticks

`reports/mnq/TDA05_histograma_ticks.png` — región central ±30 ticks (elegida porque MNQ tiene un rango de precio amplio y una dispersión de 1 minuto de varias decenas de ticks, muy distinta de las series de acción individual de Tsay; **10,03 % de las barras válidas quedan fuera de ese rango**, citado explícitamente en el propio gráfico, no ocultado). El histograma muestra una masa central suave, con pico en 0, decayendo gradualmente — **no** el pico dominante y casi degenerado de los episodios clásicos de tick grande (IBM 1/8). La cola es amplia, con movimientos extremos presentes: percentil 90 = 31 ticks, percentil 99 = 85 ticks, máximo observado = 1.754 ticks (un salto de precio real y extremo, no examinado aquí — pertenece a TH06/TDA-13). La caracterización formal de la forma de esa cola (heavy tails, familia de distribución, índice de forma) es explícitamente materia de TDA-07 y no se anticipa aquí.

## 9. Umbral operativo de relevancia (movimiento mínimo no nulo)

- **0,25 puntos** — el suelo físico absoluto, constante, especificación del contrato (`configs/mnq_snapshot.yaml`).
- Equivalente en retorno: **no es una única constante**, depende del nivel de precio (sección 13 de la tarea). Representativo global: `ln((14.365,25+0,25)/14.365,25) ≈ 1,74×10⁻⁵` (usando el Close mediano del conjunto de investigación).
- Por año, ese equivalente cae monótonamente a medida que el índice sube de nivel: de `2,9×10⁻⁵` (2019, Close mediano ≈ 8.739) a `1,2×10⁻⁵` (2025, Close mediano ≈ 21.224) — el mismo tick de 0,25 puntos representa una fracción cada vez menor del precio. Cualquier "predicción" de un movimiento inferior a 1 tick es vacía por construcción — este es el suelo de relevancia para G5 en todo el roadmap, tal como exige el criterio de interpretación de esta etapa. No se convierte en target ni en regla de trading.

## 10. Estado de TH09

**RESUELTA.** Cociente `tick/sigma` correcto (dos versiones dimensionalmente consistentes, convergentes); fracción de ceros global y desagregada por hora/año; distribución de movimientos en múltiplos de tick (histograma); comparación con/sin `FORWARD_FILL` (declarada explícitamente, sin barras confirmadas que excluir); número de valores distintos en un rango central predefinido. MNQ resulta, globalmente y en la mayoría de sus segmentos horarios/anuales, **NO** fuertemente discreta — el tick es una fracción pequeña de la dispersión típica, salvo en las horas de menor liquidez (22–01 NY) y en el tramo parcial de 2019, cuya evidencia (muestra pequeña, año incompleto) tiene menor capacidad de generalización.

## 11. Evaluación de STOP-5

El roadmap no fija un umbral numérico universal; la evaluación combina `tick/sigma`, `zero_fraction`, concentración en 0/±1/±2 ticks, número de valores distintos, estabilidad por año y tamaño muestral (sección 14 de la tarea) — sin inventar un corte de exclusión automática.

**Corrección de cierre**: la evaluación inicial de esta sección solo revisaba `TDA05_resolucion_por_hora.csv` y `TDA05_resolucion_por_anio.csv`. Se corrigió `build_stop5_watchlist()` para revisar también `TDA05_resolucion_anio_hora.csv` (`by_year_hour`) — la granularidad que podría exponer un segmento localmente muy discreto (una hora concreta dentro de un año concreto, típicamente de muestra pequeña) que las vistas por hora o por año, promediadas sobre el otro eje, no alcanzarían a mostrar por separado. Se revisó en particular los pares año×hora de 2019 más discretos (ver tabla abajo).

**Segmentos año×hora de 2019 más discretos** (`TDA05_resolucion_anio_hora.csv`, ordenados por `tick_to_sigma_points` descendente):

| hora NY | n | `zero_fraction` | `tick_to_sigma_points` | `median_abs_ticks` |
|---:|---:|---:|---:|---:|
| 23 | 297 | 33,3 % | 0,7208 | 1,0 |
| 00 | 317 | 30,9 % | 0,6963 | 1,0 |
| 17 | 5 | 40,0 % | 0,6742 | 1,0 |
| 01 | 316 | 31,3 % | 0,5277 | 1,0 |
| 22 | 333 | 30,6 % | 0,5251 | 1,0 |

Estas son, con diferencia, las cifras de `tick_to_sigma_points` más altas de todo el conjunto de investigación desagregado (por encima incluso del año 2019 completo, 0,218) — pero cada una descansa sobre `n` entre 5 y ~350 observaciones (una sola hora, dentro de un año parcial de 9 días de calendario). Es evidencia real, no se descarta, pero es evidencia **local y de muestra pequeña**: no permite inferir un régimen de discreción horario estable, solo documentar que, en esos ~9 días de diciembre de 2019, esas horas concretas mostraron una resolución más gruesa de lo habitual.

**Watchlist descriptiva** (umbrales laxos, solo para no inspeccionar cada fila a mano: `zero_fraction ≥ 50 %` o `tick_to_sigma_points ≥ 2,0`, aplicados ahora a las tres tablas — hora, año y año×hora): **0 horas, 0 años y 0 pares año×hora señalados**. Incluso los pares año×hora de 2019 de la tabla anterior (el caso más extremo del conjunto, `tick_to_sigma_points` hasta 0,72) quedan muy por debajo del umbral descriptivo de 2,0 — a más de 2,7 veces de distancia del valor más alto observado.

**`STOP-5` NO ACTIVADO.** No se identificó ningún segmento (hora NY, año, o año×hora) donde `r_1m` sea tan discreta que los análisis distribucionales continuos de TDA-06/TDA-07 pierdan su interpretación. Los casos más discretos del conjunto (los pares año×hora de 2019 de la tabla anterior) son evidencia local, de muestra pequeña y de un año parcial — documentados explícitamente, no descartados ni excluidos, pero sin capacidad de generalización suficiente para justificar la creación de un régimen o la exclusión de datos.

---

## 12. Archivos creados/modificados

**Código nuevo:**
- `src/ohlcv_dataroad/ingest/tda05_effective_resolution.py`
- `src/ohlcv_dataroad/ingest/run_tda05.py`

**Código modificado:**
- `src/ohlcv_dataroad/config.py` (campos `tda05_*`)
- `configs/mnq_snapshot.yaml` (sección `tda05`)
- `src/ohlcv_dataroad/ingest/README.md` (secciones 15-16)

**Tests nuevos:**
- `tests/test_tda05_effective_resolution.py` (26 tests: 23 del cierre inicial + 3 de la corrección de STOP-5)

**Corrección puntual de cierre** (posterior al cierre inicial): `build_stop5_watchlist()` se corrigió para revisar también `by_year_hour` (antes solo revisaba `by_hour` y `by_year`), con 3 tests nuevos (segmento año×hora flagged que no era visible en las vistas agregadas, watchlist vacía con las tres tablas, estructura de `build_by_year_hour_table`). Se corrigió además un bug de implementación expuesto por esos tests: `DataFrame.apply(..., axis=1)` sobre un subconjunto de 0 filas devuelve un `DataFrame` vacío (no una `Series`), lo que rompía `insert()` — caso normal cuando ningún segmento cruza el umbral, como ocurre en el conjunto de investigación real. Se corrigieron tres frases del informe (§3, §8, §15) y la redacción de 2019/2025 en §6, §10, §11, §15 (ver §11 para el detalle). Ningún número de las tablas cambió — solo la evaluación textual de STOP-5 y la redacción.

**Artefactos generados:**
- `reports/mnq/TDA05_resolucion_discrecion.md` (este informe)
- `reports/mnq/TDA05_resolucion_global.csv`
- `reports/mnq/TDA05_resolucion_por_hora.csv`
- `reports/mnq/TDA05_resolucion_por_anio.csv`
- `reports/mnq/TDA05_resolucion_anio_hora.csv`
- `reports/mnq/TDA05_histograma_ticks.png`
- `data/interim/mnq/tda05_tick_variables.parquet`

## 13. Tests ejecutados

```
python -m pytest -q
```

| Archivo | Tests |
|---|---:|
| `test_tda00.py` | 34 |
| `test_tda01_temporal_semantics.py` | 18 |
| `test_session_calendar.py` | 16 |
| `test_tda02_temporal_integrity.py` | 23 |
| `test_tda03_rolls.py` | 27 |
| `test_tda04_analysis_variables.py` | 29 |
| `test_tda05_effective_resolution.py` | **26** (23 del cierre inicial + 3 de la corrección de STOP-5) |
| **Total previo (tras TDA-04)** | **147** |
| **Total final** | **173** |

**Resultado: `173 passed`.**

Cobertura explícita de los 16 puntos exigidos (sección 20 de la tarea): +0,25 pts = +1 tick; −0,50 pts = −2 ticks; 0 pts = 0 ticks y coincide con `zero_1m`; ticks solo se calculan con `r_1m_valid=True` (nunca `close.shift(1)` a secas); un `NaN` de frontera de TDA-04 permanece `NaN` en las variables derivadas; ningún movimiento no entero en ticks pasa sin excepción (`TickGridInconsistencyError`, y ruido de punto flotante dentro de tolerancia SÍ pasa); hora = America/New_York, no UTC; conversión correcta a ambos lados de las transiciones de primavera y otoño de DST; agrupación por año correcta (incluye años con 0 filas válidas, mostrando `n=0` explícito); `tick/sigma` dimensionalmente correcto en ambas versiones; `tick/median_range` dimensionalmente correcto; `sigma=0`/`median_range=0` dan `inf` explícito, no excepción ni `NaN` mudo; forward-fill confirmado nunca se confunde con candidato; hold-out protegido (2 tests dedicados) y archivo exclusivamente de investigación/hold-out inexistente en disco nunca se abre; conservación de filas y trazabilidad; suite completa anterior (147 tests) sigue pasando sin cambios. Adicionalmente, la corrección de STOP-5 cubre: watchlist detecta un segmento año×hora flagged que las vistas por hora/año, tomadas solas, no exponen; watchlist vacía cuando ninguna de las tres tablas cruza el umbral (reproduciendo la conclusión real).

## 14. Regresión

`TDA-05` no modificó ningún módulo compartido de etapas anteriores (`session_calendar.py`, `tda00_integrity.py`, `tda01_temporal_semantics.py`, `tda02_temporal_integrity.py`, `tda03_rolls.py`, `tda04_analysis_variables.py` — sin cambios). `configs/mnq_snapshot.yaml` y `config.py` solo recibieron una sección `tda05` **añadida**, sin tocar las secciones existentes. Se verificó explícitamente, sin necesidad de reejecutar los pipelines costosos de TDA-00–03 (sección 24 de la tarea):

- `python -m ohlcv_dataroad.ingest.run_tda04` reproduce exactamente las mismas cifras que al cierre de TDA-04 (1.918.050 filas, 1.914.530 válidas, 3.520 inválidas, 99,8165 % retenido).
- Los 147 tests previos a TDA-05 siguen pasando (incluidos en los 173 totales).

## 15. Preguntas que permanecen abiertas

1. **2019 (y, en menor medida, 2025) son años parciales** (§6): 2019 cubre solo 9 días de calendario y 2025 cubre el conjunto de investigación hasta su corte, no un año completo. La evidencia de ambos tiene menor capacidad de generalización que la de 2020-2024 — no bloqueante, pero debe leerse con esa reserva.
2. **La cola amplia del histograma** (percentil 99 = 85 ticks, máximo = 1.754 ticks, §8): la caracterización formal de su forma (heavy tails, familia de distribución) pertenece a TDA-07; el triage de los movimientos extremos individuales pertenece a TH06/TDA-13. Ninguna de las dos se aborda en esta etapa.
3. **Las 904 barras de "goteo" dominical y las 121 `CANDIDATO_FORWARD_FILL`** (heredadas de TDA-02): siguen `INDETERMINADAS` en cuanto a su origen; no afectan materialmente ningún resultado de TDA-05 (verificado, §7).

Ninguna de estas preguntas es bloqueante para TDA-06.

## 16. Recomendación para TDA-06

- Usar `tda05_tick_variables.parquet` (o directamente `tda04_variables_1m.parquet`) como base — TDA-05 no introdujo ninguna exclusión adicional de filas.
- El perfil minuto-a-minuto completo de TDA-06 puede reportarse en versión CRUDA sin reservas de discreción: ningún segmento horario mostró una resolución tan gruesa como para invalidar una lectura continua de `|r_1m|`, `r_1m²` o `rg_t` (aunque las horas 22–01 NY sí muestran una discreción relativa mayor — un candidato natural, no forzado, a que el perfil intradía de TDA-06 muestre un quiebre de forma alrededor de esas horas).
- Retomar el diagnóstico de escalado de varianza (TH10, diferido desde TDA-04) en un punto conveniente antes de TDA-08/TDA-09, tal como recomendó ya el informe de TDA-04.

---

## Estado final

**`PASS_WITH_OPEN_QUESTIONS`**

`r_1m` se caracterizó como una variable **no fuertemente discreta**, globalmente y en la mayoría de sus segmentos horarios y anuales — el tick de 0,25 puntos es una fracción pequeña (4–10 %) de la dispersión típica de 1 minuto, con la excepción moderada de las horas de menor liquidez (22:00–01:00 NY) y del tramo parcial de 2019, cuya evidencia (muestra pequeña, año incompleto) tiene menor capacidad de generalización (2025 es también parcial, aunque con muestra sustancialmente mayor). Las dos versiones dimensionalmente correctas de `tick/sigma` (puntos y retorno) convergen en la misma lectura práctica. No existen barras `FORWARD_FILL` confirmadas — la comparación exigida por TH09 se declaró explícitamente en vez de forzarse. `TH09` queda resuelta. `STOP-5` **no se activa**, revisando explícitamente las tres granularidades (hora, año y año×hora): ningún segmento resultó tan discreto como para invalidar la caracterización distribucional continua de las etapas siguientes; los pares año×hora más discretos (2019, horas de baja liquidez) son evidencia local de muestra pequeña, documentada pero sin capacidad de generalización suficiente para activar STOP-5.

**No se avanza a TDA-06.**
