# TDA-06 — Perfil determinista intradía y de calendario

**Etapa:** `docs/methodology/Tsay_OHLCV_analysis_roadmap.md` § TDA-06
**Depende de:** TDA-04 (`PASS_WITH_OPEN_QUESTIONS`), TDA-05 (`PASS_WITH_OPEN_QUESTIONS`, STOP-5 no activado)
**Alcance de datos:** exclusivamente `tda04_variables_1m.parquet` y `tda04_return_validity_mask.parquet`. Ningún archivo de `data/raw/` ni de `holdout_files` fue abierto en esta etapa.
**Evidencia reproducible:** `reports/mnq/TDA06_perfil_minuto_global.csv`, `TDA06_perfil_minuto_por_anio.csv`, `TDA06_perfil_dia_semana.csv`, `TDA06_segmentacion_propuesta.csv`, `TDA06_calibracion_null.csv`, `TDA06_perfil_intradia.png`, `data/interim/mnq/tda06_s_m.parquet`, `data/interim/mnq/tda06_r_tilde.parquet`. Todo generado por `python -m ohlcv_dataroad.ingest.run_tda06`.

> Esta etapa aísla el componente DETERMINISTA ligado al reloj antes de estudiar cualquier dependencia estocástica. NO calcula ACF/PACF, NO estudia volatility clustering, NO ajusta ARCH/GARCH, NO entrena modelos ni crea targets, NO ejecuta EVT, NO retoma TH10 y NO asume la forma en U de NYSE ni ninguna segmentación de sesión heredada.

---

## Corrección puntual de cierre (posterior al cierre inicial)

Se detectaron y corrigieron dos problemas concretos en la segmentación/calibración; ningún otro resultado de la etapa (perfil por minuto, por año, por día de semana, umbral de extremos) se vio afectado.

**1. El suavizado fabricaba valores dentro de la ventana de mantenimiento (17:01-18:00 NY, `n=0` estructural).** `_centered_rolling_median` usaba `min_periods=1`, así que un minuto CERRADO cuya ventana de suavizado (±7 minutos) llegaba a rozar aunque fuera un único vecino válido al otro lado del hueco recibía un valor fabricado a partir de ese vecino — el minuto 17:54 NY, muy cerca del borde de cierre de la ventana (18:00), tomaba su valor casi enteramente del primer minuto abierto después de las 18:00, generando una diferencia grande y, con ella, un quiebre de segmentación **dentro** de un hueco que nunca tuvo ninguna barra. **Corrección**: se restaura explícitamente `NaN` en toda posición donde el valor original (antes de suavizar) ya era `NaN` — nunca se interpola ni se hace forward-fill. Un minuto estructuralmente cerrado permanece `NaN` después del suavizado, siempre; como consecuencia, la diferencia de primer orden a ambos lados del hueco también es `NaN`, y `detect_breakpoints` nunca puede seleccionar un minuto dentro o inmediatamente adyacente a él. **Resultado**: el corte en 17:54 desaparece de la segmentación (ver §9 corregida).

**2. La calibración G2 no ejercitaba el suavizado y, tras aumentar el número de surrogates, el criterio de estabilidad dejó de ser comparable.** La implementación original permutaba el SCORE ya suavizado y corría el detector directamente — saltaba por completo el paso de suavizado (y su interacción con los huecos), así que no calibraba el pipeline real. Se corrigió para que cada surrogate permute los valores CRUDOS (sin suavizar, antes válidos) de cada proxy por separado, preservando la máscara de minutos cerrados, y pase por la MISMA `compute_composite_activity_score` (con el fix del punto 1) y el mismo `detect_breakpoints`. Al aumentar el número de surrogates de 5 a 200 (barato: cada surrogate cuesta milisegundos) surgió un segundo problema, detectado durante la propia verificación de esta corrección: comparar cada surrogate contra los otros 199 con el mismo `min_years=3` absoluto que se usa entre 5 años reales deja de ser una condición exigente (con 199 referencias, encontrar 3 coincidencias por azar es casi seguro) — **el 100% de los quiebres candidatos del null resultaban "estables"** bajo esa comparación todos-contra-todos, un resultado claramente roto. **Corrección**: se generó una piscina de 200 surrogates (barata, deliberadamente grande), pero el filtro de estabilidad se aplica sobre 200 reagrupamientos aleatorios independientes de tamaño `len(COMPLETE_YEARS)` = 5 extraídos de esa piscina — el mismo tamaño de grupo y el mismo `min_years=3` que la comparación real entre años, repetida muchas veces para estimar con precisión la tasa de falsos positivos de esa regla exacta. **Resultado**: `stable_fraction` del null pasa de 100% (roto) a **0,63%** (ver §9 corregida).

**3. Aclaración de STOP-6 (no un cambio de conclusión).** `decide_stop6` se reescribió para dejar explícito que `RELATIVE_RANGE_REFERENCE`/`SPEARMAN_REFERENCE` (ambos 0,5) son puntos de referencia DESCRIPTIVOS predeclarados, no una puerta binaria que decide solo mirando la MEDIANA de Spearman entre años. La versión corregida exige la referencia en el rango relativo Y en el Spearman de **cada año individual** (no solo en la mediana) — una lectura más completa de la misma evidencia. Sobre el conjunto de investigación real, ambas versiones (la binaria original y la holística corregida) llegan a la MISMA conclusión: la evidencia (rango relativo ~5,3-5,5, Spearman 0,94-0,99 en cada año individual, para ambas variables de magnitud) supera las referencias por un margen tan amplio que ninguna forma razonable de agregarla cambia el resultado. Ver §10 corregida para el detalle.

**No cambiaron**: el perfil por minuto (global/por año/por día de semana), el umbral de extremos, la elección de proxy para `s(m)` (`log_hl`), ni los valores de `s(m)`/`r_tilde` (verificado: idénticos byte a byte a la ejecución anterior — ninguno de los dos problemas corregidos toca el código que los produce).

---

## 1. Qué se implementó

- `src/ohlcv_dataroad/ingest/tda06_intraday_calendar_profile.py` — módulo de análisis (perfil por minuto y por día de semana con bootstrap de bloques, umbral de extremos relativo, segmentación derivada de los datos con calibración G2, evaluación STOP-6, factor estacional `s(m)` y serie `r_tilde`).
- `src/ohlcv_dataroad/ingest/run_tda06.py` — punto de entrada de terminal; genera las 5 tablas, el gráfico obligatorio y los dos artefactos retrospectivos.
- `tests/test_tda06_intraday_calendar_profile.py` — 47 tests.
- Secciones `tda06` en `configs/mnq_snapshot.yaml` y `src/ohlcv_dataroad/config.py`.
- Secciones 17-18 de `src/ohlcv_dataroad/ingest/README.md` (pedagógicas).
- Este informe.

## 2. Inputs utilizados

- `data/interim/mnq/tda04_variables_1m.parquet` — `close`, `high`, `low`, `volume`, `r_1m`, `abs_r_1m`, `r2_1m`, `zero_1m`, `log_hl`, `trading_date`, `segment_id`.
- `data/interim/mnq/tda04_return_validity_mask.parquet` — `r_1m_valid` (la misma máscara de TDA-04, reutilizada sin recalcular).
- `configs/mnq_snapshot.yaml` (parámetros `tda06`, predeclarados antes de ejecutar sobre datos reales — ver §3).

No se usó `tda05_tick_variables.parquet`: no aporta nada a esta granularidad (minuto/día de semana) que TDA-04 no tenga ya.

## 3. Definición exacta de `minute_of_day` y `weekday`

- **`minute_of_day = hour_ny * 60 + minute_ny`**, dominio 0..1439. `timestamp` (tz-naive UTC por construcción) se localiza como UTC y se convierte a `America/New_York` con `zoneinfo` — el mismo mecanismo DST-aware de TDA-01/02/03/05, nunca un offset fijo. Verificado con tests dedicados de invierno/verano y de ambas transiciones de DST (2024-03-10, 2024-11-03).
- **`weekday = trading_date.weekday()`** (0=Lunes..6=Domingo) — **NO** la fecha calendario local del timestamp. Una barra del domingo por la noche (hora NY) pertenece a la `trading_date` del LUNES (TDA-02/03), así que su `weekday` es LUNES, no domingo. **Verificado sobre datos reales**: 0 filas con `weekday` en {5,6} (Sábado/Domingo) — el conjunto de investigación no tiene ningún "efecto domingo" fabricado por construcción (`TDA06_perfil_dia_semana.csv`: `n=0` exacto para Sábado y Domingo).

## 4. Población y denominadores por variable

| Variable | Población | `n` (global) | Motivo |
|---|---:|---:|---|
| `r_1m` (media y mediana), `abs_r_1m`, `r2_1m`, `zero_1m`, bandera de extremo | `r_1m_valid=True` (TDA-04) | 1.914.530 | Requieren una barra `t-1` comparable; usan la MISMA máscara de TDA-04, nunca recalculada. |
| `volume`, `log_hl` (≡ `rg_t = ln(H_t/L_t)`, TDA-04) | TODAS las barras admisibles | 1.918.050 | Ninguna depende de una barra anterior — restringirlas a `r_1m_valid` descartaría, sin motivo, la primera barra de cada sesión (que nunca tiene `r_1m` válido pero sí un volumen y un rango perfectamente medibles). |

La diferencia de denominador (1.918.050 vs 1.914.530, exactamente las 3.520 filas sin retorno válido de TDA-04) se reporta explícitamente en cada tabla (columna `n` por minuto/año/weekday), nunca oculta.

**Estimador central**: mediana para las variables de MAGNITUD (`abs_r_1m`, `r2_1m`, `log_hl`, `volume` — robusta a colas, roadmap TDA-06 método mínimo 1). `zero_1m` y la bandera de extremo son indicadores 0/1: se reporta su MEDIA (= proporción). `r_1m` (pregunta de la MEDIA, TH14) se reporta con AMBAS: media como estimador principal de esa pregunta específica, mediana como contraste robusto.

**Umbral de movimiento extremo** (sección 10 de la tarea, predeclarado antes de mirar el resultado): `|r_1m| >= p99(|r_1m|)` sobre toda la población válida de investigación. Valor real: **0.001590** (≈ 22,84 puntos ≈ 91,4 ticks al precio mediano de investigación — RETROSPECTIVO, relativo a la escala del propio dato, no un número de puntos fijo; ver TDA-05 sobre por qué un umbral absoluto en puntos mezclaría escalas de precio muy distintas entre 2019 y 2025).

## 5. Resultados principales del perfil intradía

El gráfico obligatorio (`TDA06_perfil_intradia.png`, cuatro paneles: `volume`, `|r_1m|`, `rg_t`, `P(r_1m=0)`, superpuestos por año) muestra, de forma inequívoca, un **patrón determinista fuerte** en las tres variables de magnitud/actividad:

- Salto abrupto en **09:30 NY** (apertura de acciones en NYSE) — el minuto más activo del día en las tres variables.
- Decaimiento gradual durante la sesión de EE. UU., con un repunte secundario cerca de **16:00-16:02 NY** (cierre/liquidación de RTH).
- Actividad baja y estable durante la noche/madrugada NY, con protuberancias menores hacia **03:00 NY** (apertura de Londres) y **08:30 NY** (horario habitual de publicaciones macro de EE. UU.).
- 60 minutos (**17:01-18:00 NY**) sin NINGUNA barra en NINGÚN día del conjunto de investigación — la ventana de mantenimiento diario de CME (TDA-02) — `n=0` explícito, nunca rellenado.

Magnitud del patrón: el proxy elegido para `s(m)` (`log_hl`, §11) varía entre **0,364** (minuto 00:00 NY, el más tranquilo) y **4,455** (minuto 09:31 NY, la apertura) tras normalizar a media 1 — es decir, el minuto más activo del día es **~12 veces** más volátil/activo que el más tranquilo.

## 6. Estabilidad por año

2020-2024 (años COMPLETOS, evidencia PRINCIPAL) muestran la MISMA forma: el salto de apertura, el decaimiento intradía y la ventana de mantenimiento aparecen en el mismo lugar y con la misma forma relativa en los cinco años, con **niveles** que sí difieren (2022, el año de mayor volatilidad de mercado del período, es consistentemente el más alto; 2019 —parcial, `n` pequeño— consistentemente el más bajo). Correlación de Spearman entre el perfil de cada año completo y el perfil global (§10): 0,94-0,99 para `abs_r_1m`, 0,96-0,99 para `log_hl` — la forma se repite casi exactamente.

2019 y 2025 se muestran siempre en el gráfico (línea punteada, más clara) pero se documentan como evidencia COMPLEMENTARIA parcial, nunca con el mismo peso que 2020-2024: 2019 (9 días de calendario) muestra un panel `P(r_1m=0)` visiblemente errático (oscila entre 0 y 1 minuto a minuto) — consistente con el hallazgo de TDA-05 de que 2019 es una muestra demasiado pequeña para estimar fracciones de forma estable, no un régimen distinto.

## 7. Resultado media vs magnitud

**Magnitud** (`abs_r_1m`, `log_hl`, `volume`): patrón fuerte y estable, descrito en §5-§6 — la varianza/actividad de 1 minuto cambia sistemáticamente con la hora del reloj, año tras año.

**Media** (`r_1m`): esencialmente plana y cercana a cero en la inmensa mayoría de los 1.440 minutos, con exactamente **dos ventanas** donde aparece una desviación no trivial:

| Ventana | minuto | media `r_1m` | IC 95% (bootstrap) | ≈ en puntos / ticks |
|---|---:|---:|---|---:|
| Apertura NYSE | 571 (09:31 NY) | +0,000119 | [0,000061, 0,000183] | ≈ +1,71 pts ≈ +6,8 ticks |
| Apertura NYSE | 575 (09:35 NY) | −0,000087 | [−0,000138, −0,000039] | ≈ −1,25 pts ≈ −5,0 ticks |
| Cierre/liquidación RTH | 952-962 (15:52-16:02 NY) | ±0,00006-0,00008 | bandas excluyen 0 en varios minutos | ≈ ±1 pt ≈ ±3-4 ticks |

Estas cifras son pequeñas en términos absolutos (0,15-0,3 desviaciones estándar de `r_1m`, una fracción de un puñado de ticks) pero **no** se descartan sin más: siguiendo la instrucción explícita de la tarea, se sometieron a escrutinio adicional. La evidencia más fuerte de que son un **artefacto de frontera de sesión** (no una "señal" de dirección) es que la **frecuencia de movimientos extremos** (§ umbral de §4) está dramáticamente concentrada exactamente en esos mismos minutos: **15,3%** de las barras en el minuto 09:31 NY superan el umbral de extremo (percentil 99 global) — frente a un promedio global de **1,0%** — una concentración ~15×. El minuto 08:31 NY (justo tras el horario habitual de publicaciones macro) también está elevado (12,4%), y 15:51 NY (13,5%). Esto es exactamente el patrón que cita el roadmap para NO interpretar un patrón de media como fenómeno de mercado antes que como artefacto de apertura/publicación macro/liquidación — **no se concluye aquí que exista un "drift" explotable**; se deja documentado como candidato a revisar si TDA-08 (dependencia en media) lo retoma.

## 8. Resultado TH15 — efecto de día de la semana

`TDA06_perfil_dia_semana.csv`. Ningún día de la semana se distingue con claridad de los demás: las bandas de bootstrap se solapan ampliamente entre Lunes-Viernes en las cuatro variables.

| Variable | Lunes | Martes | Miércoles | Jueves | Viernes |
|---|---:|---:|---:|---:|---:|
| `abs_r_1m` (mediana) | 0,000129 | 0,000124 | 0,000126 | **0,000133** | 0,000128 |
| `log_hl` (mediana) | 0,000310 | 0,000296 | 0,000297 | **0,000315** | 0,000291 |
| `volume` (mediana) | 265,0 | 271,0 | 271,0 | **274,0** | 259,75 |
| `zero_1m` (proporción) | 4,39% | 4,43% | 4,27% | 4,25% | 4,34% |
| `r_1m` (media) | +1,7e-6 | +0,4e-6 | +1,3e-6 | −0,3e-6 | +0,4e-6 |

Jueves es consistentemente el más alto en las tres variables de magnitud/actividad (~5-8% por encima de Viernes, el más bajo), pero con intervalos de bootstrap que se solapan con los demás días — una tendencia débil, no una separación clara. La media de `r_1m` por día es minúscula en todos los casos (una fracción pequeña de un tick equivalente), con bandas que en su mayoría incluyen cero. `zero_1m` es prácticamente idéntico entre días laborables.

No se ejecutó ningún test de significancia (G5 rige toda la etapa, igual que TDA-05): la "corrección por multiplicidad" que pide la tarea solo aplica cuando se testean varias etiquetas simultáneamente — al no ejecutar tests, no hay p-valores que corregir. La evidencia de (in)estabilidad es la superposición de bandas de bootstrap entre días, consistente con G4 (no se justificó una regresión con indicadores: los estadísticos simples ya responden la pregunta).

## 9. Segmentación propuesta y evidencia

Método predeclarado (§ código, `tda06_intraday_calendar_profile.py`): score compuesto (promedio normalizado y suavizado de `abs_r_1m`, `log_hl`, `volume`) → hasta 6 quiebres por magnitud de primera diferencia, con separación mínima de 60 minutos → aceptados en la propuesta final solo si un año completo independiente (2020-2024) confirma un quiebre a ≤15 minutos en al menos 3 de los 5 años.

**6 de 6 candidatos globales resultaron estables** (`TDA06_segmentacion_propuesta.csv`, tras la corrección de §"Corrección puntual de cierre" — el suavizado ya no fabrica valores dentro de la ventana de mantenimiento):

| Corte (NY) | Años que lo confirman | Soporte |
|---|---|---:|
| 02:00 | 2020, 2023, 2024 | 3/5 |
| 03:00 | 2020, 2021, 2022, 2023, 2024 | 5/5 |
| 08:30 | 2021, 2022, 2023, 2024 | 4/5 |
| 09:30 | 2020, 2021, 2022, 2023, 2024 | 5/5 |
| 16:02 | 2020, 2021, 2022, 2023, 2024 | 5/5 |
| 20:00 | 2020, 2021, 2022, 2023 | 4/5 |

El corte que antes aparecía en **17:54** (dentro de la ventana de mantenimiento 17:01-18:00 NY) **desapareció**: era un artefacto del suavizado fabricando un valor donde nunca hubo ninguna barra (§"Corrección puntual de cierre"). En su lugar, al liberarse ese cupo dentro del máximo de 6 quiebres, emergió un candidato nuevo y genuino en **02:00 NY** (soporte 3/5, el más débil de los seis).

Tres de los seis (02:00, 08:30, 20:00) tienen soporte de 3/5 o 4/5 años — más débil que los otros tres (5/5); se muestran igual, sin ocultarlos, pero con esa reserva explícita.

Los 6 cortes dividen el día en 7 tramos horarios (`00:00-02:00`, `02:00-03:00`, `03:00-08:30`, `08:30-09:30`, `09:30-16:02`, `16:02-20:00`, `20:00-24:00`). **Esto es una PROPUESTA EMPÍRICA de TDA-06, no una decisión de arquitectura ML**: se describen por límite horario, sin asignarles nombres económicos ("Overnight", "Opening", etc.) — aunque, a título puramente descriptivo, varios cortes coinciden con hitos de mercado conocidos (apertura de Londres ≈03:00, horario habitual de datos macro de EE. UU. ≈08:30, apertura de NYSE =09:30, cierre/liquidación de RTH ≈16:00-16:02, apertura de Asia ≈20:00; 02:00 NY no tiene un hito tan inmediato — se documenta como el candidato más débil, no se le busca una justificación económica forzada).

**Calibración G2** (`TDA06_calibracion_null.csv`, corregida): se genera una piscina de **200 surrogates**, cada uno permutando los valores CRUDOS (sin suavizar) de cada proxy manteniendo fija la máscara de minutos estructuralmente cerrados, y pasando por el MISMO pipeline completo (suavizado con la máscara restaurada, normalización, promedio, detección de quiebres) que los datos reales. El filtro de estabilidad se evalúa sobre **200 reagrupamientos aleatorios independientes de tamaño 5** (= `len(COMPLETE_YEARS)`) extraídos de esa piscina — el mismo tamaño de grupo y el mismo `min_years=3` que la comparación real entre años. Resultado: de 6.000 eventos candidato-vs-grupo evaluados, solo **38 (0,63%)** resultaron "estables" — frente al **100% (6/6)** sobre los datos reales. El detector no fabrica estructura sobre ruido puro cuando se calibra con el pipeline y el criterio correctos.

*(Nota metodológica, documentada explícitamente: una implementación intermedia de esta corrección, que sí ejercitaba el suavizado pero comparaba cada surrogate contra los otros 199 de la piscina completa con el mismo `min_years=3` absoluto, producía 100% de "estabilidad" también en el null — un resultado claramente roto, causado por relajar el criterio al aumentar el denominador sin ajustar el tamaño de grupo. Se corrigió antes de aceptar el resultado; ver §"Corrección puntual de cierre".)*

## 10. Evaluación STOP-6

Criterio predeclarado (sin umbral universal del roadmap): dos puntos de referencia DESCRIPTIVOS (no una puerta binaria estricta) — rango relativo del perfil > 0,5 (variación minuto a minuto al menos del orden del nivel típico) y correlación de Spearman > 0,5 entre año completo y global. La decisión final es una lectura HOLÍSTICA de la evidencia completa: se exige la referencia de Spearman en **cada año individual** (no solo en la mediana entre años) para ambas variables de magnitud, junto con la consistencia entre `abs_r_1m` y `log_hl` y la estabilidad visual del gráfico (§5-§6).

| Variable | Rango relativo | Spearman mediano (2020-2024 vs global) | Spearman por año |
|---|---:|---:|---|
| `abs_r_1m` | **5,4800** | **0,9825** | 0,937 / 0,977 / 0,983 / 0,986 / 0,983 |
| `log_hl` | **5,3182** | **0,9902** | 0,958 / 0,987 / 0,990 / 0,994 / 0,990 |

Ambas variables superan la referencia de rango relativo por un margen amplio (~10×), y superan la referencia de Spearman en **cada uno de los 5 años individuales** (mínimo observado: 0,937, casi 2× la referencia) — no solo en la mediana agregada. La lectura holística y una comparación binaria simple sobre la mediana llegan, en este caso, a la misma conclusión: la magnitud de la evidencia es tan amplia que ninguna forma razonable de agregarla la revierte.

**`STOP-6` NO ACTIVADO.** El perfil de magnitud/volatilidad es fuerte y estable — se construye `s(m)` (§11).

## 11. Factor estacional `s(m)`

**Elección de proxy** (`choose_s_m_proxy`, criterio TÉCNICO predeclarado, no elegido por "qué produce mejores resultados"): se compara el RUIDO del estimador — ancho medio de la banda de bootstrap relativo a su propio nivel, promediado por minuto — entre `abs_r_1m` y `log_hl` (deliberadamente distinto de `relative_range`, que mide la fuerza del patrón, usado en §10). Resultado real: `log_hl` resultó proporcionalmente menos ruidoso → **proxy elegido: `log_hl`** (consistente con Tsay/roadmap C3: "si `rg_t` es sustancialmente menos ruidoso, es el proxy preferido").

**Fórmula**: `s(m) = mediana(log_hl en el minuto m) / media(mediana(log_hl) sobre los minutos con datos)` — normalización estándar de un índice estacional a media 1: `s(m)>1` = minuto más activo que el promedio del día. Rango real: **0,364** (minuto 00:00 NY) a **4,455** (minuto 09:31 NY). 60 minutos (17:01-18:00 NY, ventana de mantenimiento, `n=0`) quedan sin `s(m)` definido — no hay ninguna barra ahí para ajustar, así que no es un caso relevante en la práctica.

**Protección de división por cero**: minutos con nivel `<=0` o no finito quedan con `s_m=NaN` explícito, nunca `inf` (no ocurrió en la práctica: `log_hl>=0` siempre por construcción).

`r_tilde = r_1m / s(minute_of_day)` — `NaN` donde `r_1m` es inválido o `s(m)` no está definido, nunca `inf`. **1.914.530 valores no nulos**, exactamente el mismo `n` que `r_1m_valid` — ninguna fila se pierde ni se inventa. `std(r_tilde) ≈ 4,17e-4`, muy cercano a `std(r_1m) ≈ 4,19e-4` (TDA-05) — consistente con la normalización a media 1 (`r_tilde` reequilibra la varianza CONDICIONAL por minuto sin alterar mucho la escala GLOBAL).

Ambos artefactos (`tda06_s_m.parquet`, `tda06_r_tilde.parquet`) llevan la columna `label="RETROSPECTIVO"` — `s(m)` se estimó usando TODA la muestra de investigación (G1): **no** es una feature causal disponible online sin volver a estimarse solo con datos hasta cada momento.

## 12. Archivos creados/modificados

**Código nuevo:**
- `src/ohlcv_dataroad/ingest/tda06_intraday_calendar_profile.py`
- `src/ohlcv_dataroad/ingest/run_tda06.py`

**Código modificado:**
- `src/ohlcv_dataroad/config.py` (campos `tda06_*`, sección añadida sin tocar las existentes)
- `configs/mnq_snapshot.yaml` (sección `tda06`, añadida)
- `src/ohlcv_dataroad/ingest/README.md` (intro, listado de archivos, secciones 17-18 nuevas)

**Tests nuevos:**
- `tests/test_tda06_intraday_calendar_profile.py` (47 tests: 36 del cierre inicial + 11 de la corrección puntual)

**Corrección puntual de cierre (posterior al cierre inicial)**: `_centered_rolling_median`/`compute_composite_activity_score` (restauración de la máscara `NaN`), `calibrate_breakpoint_detector`/`_permute_preserving_nan_mask` (pipeline completo + reagrupamiento por tamaño de año, `DEFAULT_N_SURROGATES=200`, `DEFAULT_N_STABILITY_GROUPS=200`) y `decide_stop6`/`RELATIVE_RANGE_REFERENCE`/`SPEARMAN_REFERENCE` (decisión holística) en `tda06_intraday_calendar_profile.py` — ningún archivo nuevo, ninguna sección de configuración nueva. Se regeneraron únicamente los artefactos de segmentación/calibración (`TDA06_segmentacion_propuesta.csv`, `TDA06_calibracion_null.csv`) y este informe; `s(m)`/`r_tilde` y las demás tablas no cambiaron (verificado).

**Artefactos generados:**
- `reports/mnq/TDA06_perfil_intradia_calendario.md` (este informe)
- `reports/mnq/TDA06_perfil_minuto_global.csv`
- `reports/mnq/TDA06_perfil_minuto_por_anio.csv`
- `reports/mnq/TDA06_perfil_dia_semana.csv`
- `reports/mnq/TDA06_segmentacion_propuesta.csv`
- `reports/mnq/TDA06_calibracion_null.csv`
- `reports/mnq/TDA06_perfil_intradia.png`
- `data/interim/mnq/tda06_s_m.parquet`
- `data/interim/mnq/tda06_r_tilde.parquet`

## 13. Tests ejecutados

```
python -m pytest -q
```

| Archivo | Tests |
|---|---:|
| Suite previa (TDA-00…05) | 173 |
| `test_tda06_intraday_calendar_profile.py` | **47** (36 del cierre inicial + 11 de la corrección puntual) |
| **Total final** | **220** |

**Resultado: `220 passed`.**

Cobertura de los 20 puntos exigidos (sección 16 de la tarea): `minute_of_day` correcto en NY (no UTC); DST manejado en ambas transiciones; domingo nocturno con `weekday` del lunes de su `trading_date` (2 tests dedicados, más verificación directa sobre datos reales: 0 filas Sáb/Dom); rango 0..1439; ningún `r_1m` inválido reintroducido en los perfiles de retorno (test con valor centinela extremo); ningún hueco puenteado ni roll cruzado (se hereda directamente de la máscara `r_1m_valid` de TDA-04, sin recalcular); ningún archivo de hold-out abierto (3 tests); minutos estructuralmente cerrados con `n=0`/`NaN` explícito, nunca rellenados NI SIQUIERA DESPUÉS DEL SUAVIZADO (4 tests nuevos de la corrección); `n` correcto por minuto (verificado a mano); agregación minuto-a-minuto exacta contra una mediana calculada a mano; estabilidad por año sin mezclar etiquetas (verificado con valores distintos por año en el mismo minuto); años parciales (2019/2025) identificados y distinguidos explícitamente de los completos; `s(m)` solo se genera si `decide_stop6` lo permite (test con la puerta forzada en ambos sentidos vía monkeypatch); normalización de `s(m)` a media 1 (verificado a mano); protección contra `s(m)=0`/no finito (nunca `inf`); `r_tilde = r/s(m)` exacto (verificado a mano) y `NaN` donde corresponde; etiqueta `RETROSPECTIVO` persistida; segmentación reproducible (mismo resultado en dos ejecuciones) y respeta la separación mínima/el tope de cortes, y nunca propone un corte dentro/al borde de un hueco estructural (2 tests nuevos); calibración del detector de quiebres sobre datos crudos sin estructura real, ejercitando el mismo pipeline (suavizado incluido); criterio de estabilidad de la calibración no degenera con una piscina de surrogates grande (test de regresión dedicado); `decide_stop6` es holístico (exige la referencia en cada año individual, no solo en la mediana — 2 tests nuevos).

## 14. Estado de TH14

**RESUELTA.** Existe un patrón determinista de actividad/volatilidad fuerte y estable, ligado al minuto del día — no a la forma en U de NYSE, sino a una forma propia de un instrumento casi 24 horas con picos claros en la apertura de NYSE, decaimiento intradía, un repunte en el cierre/liquidación, y actividad baja y estable durante la noche, estable año tras año (2020-2024, Spearman 0,94-0,99). El patrón está en la **magnitud/varianza**, no en la media: la media de `r_1m` es esencialmente plana salvo dos ventanas estrechas (apertura NYSE, cierre RTH) que se identifican como candidatas a artefacto de frontera de sesión (concentración de 15× en la frecuencia de extremos en esos mismos minutos), no como una señal direccional confirmada.

## 15. Estado de TH15

**RESUELTA (resultado débil/nulo).** No se encontró un efecto de día de la semana claramente distinguible del ruido en `r_1m`, `abs_r_1m`, `rg_t` ni `volume` — las bandas de bootstrap se solapan ampliamente entre Lunes y Viernes en las cuatro variables. Jueves muestra una tendencia mildly más alta en las tres variables de magnitud/actividad, pero sin separación clara. Consistente con la prioridad BAJA que el backlog asigna a TH15.

## 16. Preguntas abiertas

1. **Las dos ventanas de posible efecto en la media de `r_1m`** (09:31-09:35 NY, 15:52-16:02 NY, §7): identificadas y cuantificadas, pero deliberadamente NO investigadas más a fondo aquí — TDA-06 no debe anticipar TDA-08 (dependencia en media). Se recomienda que TDA-08 las trate con escrutinio adicional antes de citarlas como cualquier tipo de "señal".
2. **Los tres cortes de segmentación con soporte 3/5 o 4/5** (02:00, 08:30 y 20:00 NY, §9): más débiles que los otros tres (5/5); se documentan, no se descartan. El corte en 02:00 NY, en particular, no tiene un hito de mercado conocido tan inmediato como los demás.
3. **2019 y 2025 parciales**: como en TDA-05, su evidencia tiene menor capacidad de generalización que 2020-2024 — el panel `P(r_1m=0)` de 2019 en el gráfico principal lo ilustra visualmente de forma directa.
4. **TH10** (escalado de varianza, diferida desde TDA-04): sigue diferida; no se encontró ninguna dependencia metodológica que obligara a retomarla dentro de TDA-06.

Ninguna de estas preguntas es bloqueante para continuar.

## 17. Recomendación para el siguiente paso

- TDA-07 debe reportar la distribución marginal **cruda Y ajustada** (`r_1m` y `r_tilde`, ambos, este último etiquetado `RETROSPECTIVO`) — tal como exige el roadmap cuando STOP-6 no se activa.
- Usar la segmentación de §9 como partición adicional opcional donde sea útil, nunca como arquitectura ML definitiva — sigue siendo una PROPUESTA EMPÍRICA.
- Cuando TDA-08 (dependencia en media) se ejecute, tratar explícitamente las dos ventanas de §7/§16.1 como candidatas a artefacto de apertura/cierre a descartar o confirmar, no como una media distinta de cero per se.
- Retomar TH10 en un punto conveniente antes de TDA-08/TDA-09, como ya recomendaron TDA-04 y TDA-05.

---

## Estado final

**`PASS_WITH_OPEN_QUESTIONS`**

TH14 y TH15 quedan resueltas. El perfil de magnitud/actividad es fuerte y estable entre años completos (STOP-6 **NO ACTIVADO**, decisión holística respaldada por la magnitud completa de la evidencia, no un umbral binario ciego); se construyó y persistió `s(m)` y `r_tilde`, ambos etiquetados `RETROSPECTIVO` (idénticos a la versión pre-corrección: ningún fix de esta corrección los afecta). El patrón de media es esencialmente nulo salvo dos ventanas estrechas de apertura/cierre identificadas como candidatas a artefacto, no confirmadas como señal. La segmentación derivada de los datos (6 cortes tras la corrección — 02:00, 03:00, 08:30, 09:30, 16:02, 20:00 NY — ya no incluye el corte fabricado en 17:54 dentro de la ventana de mantenimiento) está calibrada contra un null que ahora sí ejercita el pipeline completo (0,63% de estabilidad espuria, frente al 100% real) y es una propuesta empírica, no una decisión de arquitectura. TH15 no muestra un efecto de calendario semanal claramente distinguible del ruido.

**No se avanza a TDA-07.**
