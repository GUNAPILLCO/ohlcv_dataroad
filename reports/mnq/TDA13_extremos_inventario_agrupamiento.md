# TDA-13 Parte A — Extremos: inventario, triage y agrupamiento

**Etapa:** `docs/methodology/Tsay_OHLCV_analysis_roadmap.md` § TDA-13, Parte A (Nivel 4, obligatoria). **Hipótesis:** `TH06`, `TH27`.
**Depende de:** TDA-00 (máscara `bad_data`), TDA-02 (huecos y su causa), TDA-03 (rolls; firma de "tick aislado"), TDA-04 (`r_1m`, validez), TDA-05 (categoría de barra), TDA-06 (segmentos), TDA-07 (niveles de cuantil, bootstrap, retorno contrafactual), TDA-09 (bloques de continuidad), TDA-10 (`sigma_hat`, `z_t`), TDA-12 (decil causal de volatilidad; TH26 `NO CONCLUYENTE`).
**Alcance de datos:** exclusivamente artefactos ya construidos sobre el conjunto de investigación (`data/interim/mnq/` y CSV de `reports/mnq/`). Ningún archivo de `data/raw/` ni de `holdout_files` fue abierto.
**Generado automáticamente** por `python -m ohlcv_dataroad.ingest.run_tda13 --config configs/mnq_snapshot.yaml` — este informe NUNCA se edita a mano; cualquier corrección se hace en el código (`tda13_extremes_inventory.py`/`run_tda13.py`) y se regenera.

> Esta etapa hace un inventario de los movimientos de 1 minuto más grandes de MNQ, clasifica cada uno (¿error de datos, artefacto o movimiento real?) y mide si llegan en grupos. Es caracterización estadística: NO diseña targets, señales ni modelos; NO ejecuta la Parte B (EVT, POT, GPD, índice extremal); NO abre el holdout. Todos los resultados son **exploratorios y retrospectivos** (G2): ninguno se ha confirmado en el holdout.

---

## 1. Preguntas

1. **TH06:** de los movimientos extremos, ¿cuáles son errores de datos, cuáles artefactos (de roll o de frontera) y cuáles movimientos reales?
2. **TH27:** ¿los extremos llegan más juntos de lo que llegarían si fueran independientes? ¿Sigue ocurriendo después de descontar la volatilidad?

**Palabras que se usan en este informe.**
- *Extremo / excedencia*: un retorno de 1 minuto que supera un umbral. Aquí el umbral no es un número fijo: es "estar entre el 1 % de los movimientos más grandes para esa hora y ese nivel de agitación".
- *Retorno estandarizado `z`*: el retorno dividido por la volatilidad reciente. Un `z` grande es un movimiento grande *para lo agitado que ya estaba el mercado*.
- *Agrupamiento*: que tras un extremo sea más probable de lo normal ver otro poco después.
- *Referencia de independencia (Poisson)*: lo que se vería si cada barra tuviera la misma probabilidad de ser extremo que la observada, pero sin ninguna relación con las barras vecinas.
- *IC 95 %*: intervalo que indica cuánto podría moverse el número si se repitiera el análisis con otras jornadas (bootstrap por jornadas completas).

## 2. Población

| poblacion | n |
|---:|---:|
| Barras de la serie canonica (TDA-04) | 1,918,050 |
| Universo de triage: retorno contrafactual finito (validos + cruces de frontera) | 1,918,049 |
| Retornos validos (`r_1m_valid`, poblacion de TDA-07) | 1,914,530 |
| Retornos que cruzan frontera/hueco/roll (solo para el triage) | 3,519 |
| Poblacion comun de agrupamiento (valido, `z` finito, decil causal definido) | 1,913,438 |
| Validos sin estado de volatilidad (quemado / sin historia): fuera del agrupamiento | 1,092 |
| Bloques de continuidad en la poblacion comun | 3,081 |
| Jornadas | 1,420 |

- **`r`** = `r_1m` válido de TDA-04. **`z`** = `r_t / sigma_hat_(t-1)` con el estimador primario de TDA-10 (EWMA, semivida 60 barras), sin redefinir nada. Estado de volatilidad = decil causal de TDA-12.
- **Población común.** `r` y `z` se comparan sobre las MISMAS barras (retorno válido, `z` finito, decil definido). Así las tasas de excedencia son comparables.
- **Retornos que cruzan una frontera.** TDA-04 los excluyó de `r_1m` (cruzan un roll, un cierre de sesión o un hueco). No vuelven a entrar en ningún análisis estadístico: sólo se recuperan para el triage, con el retorno contrafactual que TDA-07 ya definió, porque precisamente ahí están los candidatos a artefacto.
- **Causalidad (alcance exacto).** `sigma_hat` y `z` son causales (prueba de reconstrucción de TDA-10 re-ejecutada para el estimador primario: pasa); decil causal verificado por muestreo (6 de 6 comprobaciones pasan). **Los umbrales, las referencias y los cortes de segmento usan toda la muestra: son RETROSPECTIVOS.** Esta etapa describe; no produce ninguna cantidad utilizable en tiempo real.

## 3. Qué es un extremo aquí

Un umbral único para toda la muestra seleccionaría horas (la apertura) y épocas agitadas, no eventos. Por eso el umbral se fija **dentro de cada celda** = segmento horario (7, de TDA-06) × decil causal de volatilidad (10): un retorno es extremo negativo si queda por debajo del percentil 1 de su celda, y positivo si queda por encima del 99.

- **Definición primaria:** `SEGMENT_X_VOL`, nivel 1 % por cola.
- **Sensibilidades declaradas de antemano:** niveles 0.1 %, 5 %; definiciones `SEGMENT` (sólo segmento) y `GLOBAL` (umbral único: el anti-patrón, incluido sólo para ver su efecto).
- Caídas y subidas se analizan por separado; "ambas" se reporta como complemento.

| serie | definición | nivel | celdas | n | excedencias neg. | excedencias pos. | tasa neg. realizada | tasa pos. realizada | celdas con soporte < 200 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r_1m | GLOBAL | 1 % | 1 | 1,913,438 | 19,135 | 19,135 | 1.000 % | 1.000 % | 0 |
| r_1m | GLOBAL | 0.1 % | 1 | 1,913,438 | 1,914 | 1,914 | 0.100 % | 0.100 % | 0 |
| r_1m | GLOBAL | 5 % | 1 | 1,913,438 | 95,671 | 95,669 | 5.000 % | 5.000 % | 0 |
| z_ewma_60_raw | GLOBAL | 1 % | 1 | 1,913,438 | 19,135 | 19,135 | 1.000 % | 1.000 % | 0 |
| z_ewma_60_raw | GLOBAL | 0.1 % | 1 | 1,913,438 | 1,914 | 1,914 | 0.100 % | 0.100 % | 0 |
| z_ewma_60_raw | GLOBAL | 5 % | 1 | 1,913,438 | 95,672 | 95,672 | 5.000 % | 5.000 % | 0 |
| r_1m | SEGMENT | 1 % | 7 | 1,913,438 | 19,138 | 19,138 | 1.000 % | 1.000 % | 0 |
| r_1m | SEGMENT | 0.1 % | 7 | 1,913,438 | 1,917 | 1,917 | 0.100 % | 0.100 % | 3 |
| r_1m | SEGMENT | 5 % | 7 | 1,913,438 | 95,675 | 95,674 | 5.000 % | 5.000 % | 0 |
| z_ewma_60_raw | SEGMENT | 1 % | 7 | 1,913,438 | 19,138 | 19,138 | 1.000 % | 1.000 % | 0 |
| z_ewma_60_raw | SEGMENT | 0.1 % | 7 | 1,913,438 | 1,917 | 1,917 | 0.100 % | 0.100 % | 3 |
| z_ewma_60_raw | SEGMENT | 5 % | 7 | 1,913,438 | 95,675 | 95,675 | 5.000 % | 5.000 % | 0 |
| r_1m | SEGMENT_X_VOL | 1 % | 70 | 1,913,438 | 19,166 | 19,166 | 1.002 % | 1.002 % | 37 |
| r_1m | SEGMENT_X_VOL | 0.1 % | 70 | 1,913,438 | 1,950 | 1,950 | 0.102 % | 0.102 % | 70 |
| r_1m | SEGMENT_X_VOL | 5 % | 70 | 1,913,438 | 95,703 | 95,702 | 5.002 % | 5.002 % | 8 |
| z_ewma_60_raw | SEGMENT_X_VOL | 1 % | 70 | 1,913,438 | 19,166 | 19,166 | 1.002 % | 1.002 % | 37 |
| z_ewma_60_raw | SEGMENT_X_VOL | 0.1 % | 70 | 1,913,438 | 1,950 | 1,950 | 0.102 % | 0.102 % | 70 |
| z_ewma_60_raw | SEGMENT_X_VOL | 5 % | 70 | 1,913,438 | 95,703 | 95,703 | 5.002 % | 5.002 % | 8 |

**Grupos pequeños.** En la definición primaria hay 70 celdas; 37 tienen menos de 200 excedencias por cola (2 con menos de 20). La celda más pequeña es `02:00-03:00|D9` con 1,149 barras. En esas celdas el umbral es impreciso; con el nivel 0.1 % casi todas las celdas quedan con muy pocas excedencias, y sus resultados son sólo orientativos. Como los precios se mueven en saltos discretos (ticks), varias barras pueden empatar exactamente en el umbral; se cuentan como extremo sólo las que lo superan estrictamente, y por eso la tasa realizada puede quedar algo por debajo del nivel nominal. Detalle por celda en `TDA13_umbrales_por_celda.csv`.

## 4. Criterios de triage (escritos antes de calcular su efecto)

Reglas en orden; decide la primera que aplica.

| Regla | Categoría | Criterio |
|---|---|---|
| R1 | `ERROR_DATO` | La barra o la barra anterior viola un invariante fisico de TDA-00 (`bad_data`). Es la unica evidencia que DEMUESTRA un error. |
| R2 | `ARTEFACTO_ROLL` | El retorno cruza un cambio de contrato (`invalid_reason = ROLL_BOUNDARY`, TDA-04). |
| R3 | `ARTEFACTO_FRONTERA` | El retorno cruza una frontera de jornada o un hueco (`TRADING_DATE_BOUNDARY`, `NON_CONSECUTIVE_MINUTE`): no es un retorno de 1 minuto. |
| R4 | `INDETERMINADO` | Retorno valido con ALGUNA duda: (a) barra o barra anterior plana o candidata a forward-fill; (b) pegado a un hueco de causa desconocida; (c) no existe barra siguiente contigua para comprobar si el precio se sostiene; (d) revierte mas de la mitad en la barra siguiente (firma de TDA-03; sin ticks no se distingue un pico real de un ultimo precio anomalo); (e) volumen bajo el percentil 5 global (TDA-03). |
| R5 | `EVENTO_REAL` | Retorno valido SIN ninguna de las dudas de R4: barras activas, registro continuo alrededor, precio sostenido en la barra siguiente y volumen no bajo. Es evidencia positiva de que hubo negociacion a ese precio; no identifica la causa del movimiento. |

**Lo que estas reglas NO hacen.**
- No declaran real un movimiento sólo porque no se detecte un error: `EVENTO_REAL` exige que el precio se sostenga en la barra siguiente y que hubiera negociación normal.
- No atribuyen causas. El repositorio no contiene un calendario fechado de noticias o datos macroeconómicos, así que **ningún extremo se atribuye a un evento**. La columna `clock_0830_window` sólo marca que la barra cae en los 5 minutos desde las 08:30 de Nueva York (el único horario de publicaciones citado en el repositorio, TDA-06). Es una coincidencia de reloj: no demuestra causalidad y no interviene en la clasificación.
- No eliminan nada. El inventario clasifica; los datos originales y los resultados de TDA-07/TDA-12 quedan intactos.

**Qué es heredado y qué es nuevo.** Las máscaras son de etapas anteriores. La firma "revierte más de 0.5 del salto en la barra siguiente" y "volumen bajo el percentil 5" son las de TDA-03, sin cambiar sus valores. **Usarlas como criterio de triage de todos los extremos es una decisión nueva y exploratoria de esta etapa**, igual que el orden de las reglas.

## 5. Inventario y resultado del triage (TH06)

`TDA13_extremos.csv` contiene **38,961 extremos** (una fila por observación): 38,332 de la población común, 629 retornos que cruzan una frontera y 0 válidos sin estado de volatilidad (para estos dos últimos grupos el umbral aplicado es el de su segmento, anotado en `threshold_scope`). Cada fila lleva fecha y hora (UTC y Nueva York), contrato, segmento, estado de volatilidad, magnitud (en retorno, en ticks y en `z`), umbral aplicado, indicadores de calidad, categoría, regla que decidió y evidencia.

| categoría | extremos | % del inventario |
|---:|---:|---:|
| ERROR_DATO | 0 | 0.00 % |
| ARTEFACTO_ROLL | 21 | 0.05 % |
| ARTEFACTO_FRONTERA | 608 | 1.56 % |
| EVENTO_REAL | 33,204 | 85.22 % |
| INDETERMINADO | 5,128 | 13.16 % |

- Errores y artefactos juntos: 1.61 % del inventario.
- 28,448 de los extremos de `r` (73.0 %) lo son también en `z`.
- 635 caen en la ventana de reloj de las 08:30 NY (contexto, sin atribución).

**Por qué quedan `INDETERMINADO`** (una observación puede tener varias dudas):

| duda | extremos | % de los indeterminados |
|---:|---:|---:|
| BARRA_NO_ACTIVA | 62 | 1.21 % |
| JUNTO_A_HUECO_DESCONOCIDO | 103 | 2.01 % |
| SIN_BARRA_SIGUIENTE | 73 | 1.42 % |
| REVIERTE_BARRA_SIGUIENTE | 4,726 | 92.16 % |
| VOLUMEN_BAJO | 297 | 5.79 % |
| SOLO_REVIERTE_BARRA_SIGUIENTE | 4,673 | 91.13 % |

**Por dirección**

| grupo | ERROR_DATO | ARTEFACTO_ROLL | ARTEFACTO_FRONTERA | EVENTO_REAL | INDETERMINADO | total | % INDETERMINADO |
|---:|---:|---:|---:|---:|---:|---:|---:|
| NEG | 0 | 8 | 284 | 16,510 | 2,656 | 19,458 | 13.65 % |
| POS | 0 | 13 | 324 | 16,694 | 2,472 | 19,503 | 12.67 % |

**Por año**

| grupo | ERROR_DATO | ARTEFACTO_ROLL | ARTEFACTO_FRONTERA | EVENTO_REAL | INDETERMINADO | total | % INDETERMINADO |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2019 | 0 | 0 | 1 | 1 | 0 | 2 | 0.00 % |
| 2020 | 0 | 4 | 248 | 11,259 | 1,943 | 13,454 | 14.44 % |
| 2021 | 0 | 4 | 66 | 5,895 | 854 | 6,819 | 12.52 % |
| 2022 | 0 | 4 | 107 | 5,528 | 749 | 6,388 | 11.73 % |
| 2023 | 0 | 4 | 58 | 5,027 | 772 | 5,861 | 13.17 % |
| 2024 | 0 | 4 | 69 | 3,484 | 487 | 4,044 | 12.04 % |
| 2025 | 0 | 1 | 59 | 2,010 | 323 | 2,393 | 13.50 % |

**Por segmento horario**

| grupo | ERROR_DATO | ARTEFACTO_ROLL | ARTEFACTO_FRONTERA | EVENTO_REAL | INDETERMINADO | total | % INDETERMINADO |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 00:00-02:00 | 0 | 0 | 18 | 2,940 | 446 | 3,404 | 13.10 % |
| 02:00-03:00 | 0 | 0 | 3 | 1,492 | 208 | 1,703 | 12.21 % |
| 03:00-08:30 | 0 | 0 | 13 | 8,146 | 1,166 | 9,325 | 12.50 % |
| 08:30-09:30 | 0 | 0 | 3 | 1,440 | 262 | 1,705 | 15.37 % |
| 09:30-16:02 | 0 | 0 | 12 | 9,315 | 1,441 | 10,768 | 13.38 % |
| 16:02-20:00 | 0 | 2 | 532 | 3,944 | 820 | 5,298 | 15.48 % |
| 20:00-24:00 | 0 | 19 | 27 | 5,927 | 785 | 6,758 | 11.62 % |

**Por población**

| grupo | ERROR_DATO | ARTEFACTO_ROLL | ARTEFACTO_FRONTERA | EVENTO_REAL | INDETERMINADO | total | % INDETERMINADO |
|---:|---:|---:|---:|---:|---:|---:|---:|
| CRUCE_FRONTERA | 0 | 21 | 608 | 0 | 0 | 629 | 0.00 % |
| VALIDO_CON_ESTADO | 0 | 0 | 0 | 33,204 | 5,128 | 38,332 | 13.38 % |

**Cómo leerlo.** `INDETERMINADO` no significa "sospechoso de error": significa que con barras de 1 minuto no hay forma de comprobarlo. La duda más frecuente es que el precio retroceda más de la mitad en el minuto siguiente; eso es lo que haría un precio anómalo, pero también lo hace un mercado real que rebota. Sin datos tick a tick (G0: la secuencia dentro del minuto no es observable) las dos cosas son indistinguibles.

## 6. Cómo se mide el agrupamiento

1. Se marcan las excedencias y se mide cuántas barras pasan de una a la siguiente. **Sólo se mide dentro de un bloque continuo de negociación**: nunca a través de un cierre de sesión, la pausa diaria, un fin de semana, un hueco o un cambio de contrato.
2. Se compara con la **referencia de independencia**: cada barra es extremo con una probabilidad que depende de *su minuto del día y su año*, sin relación con las vecinas. Es el equivalente, en barras de 1 minuto, de un proceso de Poisson de la misma intensidad; su valor esperado se calcula de forma exacta y respetando los mismos bloques. Controlar el minuto del día evita llamar "agrupamiento" a que todos los días haya movimiento a la misma hora.
3. Estadístico principal: **qué fracción de los tiempos entre extremos es de 10 barras o menos**, dividida por la fracción esperada. Cociente 1 = como si fueran independientes; 2 = el doble de extremos "pegados".
4. **Cuándo se considera detectado** (regla revisada en la auditoría, §12 y §13): el cociente debe superar la *banda calibrada* —lo máximo que produce, el 97,5 % de las veces, este mismo procedimiento completo aplicado a series que no tienen ninguna dependencia— y además su IC 95 % debe quedar por encima de 1.

**Referencia primaria `RELOJ_Y_ANIO` (corregida, hallazgo A).** La probabilidad es el producto de un factor por minuto y un factor por año, ajustados para que el número esperado de extremos coincida con el observado *en cada minuto y en cada año a la vez* (ajuste proporcional iterativo). La fórmula original multiplicaba la tasa del minuto por (tasa del año / tasa global), lo que no conserva esos totales cuando los años no reparten sus barras igual entre minutos. Comprobación sobre los datos (caídas y subidas, nivel y definición primarios):

| serie | lado | referencia | observadas | esperadas | error máx. por minuto | error máx. por año | año peor (error relativo) | barras con p = 0 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r_1m | NEG | RELOJ_Y_ANIO | 19,166 | 19,166.00 | 0.000 | 0.00 | 2020 (0.00 %) | 17,260 |
| r_1m | NEG | RELOJ | 19,166 | 19,166.00 | 0.000 | 3,037.04 | 2020 (-46.85 %) | 10,982 |
| r_1m | NEG | HOMOGENEA | 19,166 | 19,166.00 | 207.057 | 3,050.86 | 2020 (-47.06 %) | 0 |
| r_1m | NEG | RELOJ_X_FACTOR_ANIO_PREAUDITORIA | 19,166 | 19,181.48 | 1.868 | 26.11 | 2020 (0.40 %) | 17,260 |
| r_1m | POS | RELOJ_Y_ANIO | 19,166 | 19,166.00 | 0.000 | 0.00 | 2020 (-0.00 %) | 5,476 |
| r_1m | POS | RELOJ | 19,166 | 19,166.00 | 0.000 | 3,271.49 | 2020 (-48.69 %) | 5,476 |
| r_1m | POS | HOMOGENEA | 19,166 | 19,166.00 | 272.087 | 3,286.86 | 2020 (-48.92 %) | 0 |
| r_1m | POS | RELOJ_X_FACTOR_ANIO_PREAUDITORIA | 19,166 | 19,184.13 | 1.742 | 30.09 | 2020 (0.45 %) | 5,476 |
| z_ewma_60_raw | NEG | RELOJ_Y_ANIO | 19,166 | 19,166.00 | 0.000 | 0.00 | 2022 (0.00 %) | 5,423 |
| z_ewma_60_raw | NEG | RELOJ | 19,166 | 19,166.00 | 0.000 | 537.22 | 2023 (18.14 %) | 5,423 |
| z_ewma_60_raw | NEG | HOMOGENEA | 19,166 | 19,166.00 | 226.057 | 540.14 | 2023 (18.24 %) | 0 |
| z_ewma_60_raw | NEG | RELOJ_X_FACTOR_ANIO_PREAUDITORIA | 19,166 | 19,167.03 | 0.227 | 10.14 | 2020 (0.28 %) | 5,423 |
| z_ewma_60_raw | POS | RELOJ_Y_ANIO | 19,166 | 19,166.00 | 0.000 | 0.00 | 2023 (0.00 %) | 16,480 |
| z_ewma_60_raw | POS | RELOJ | 19,166 | 19,166.00 | 0.000 | 579.72 | 2024 (19.68 %) | 16,480 |
| z_ewma_60_raw | POS | HOMOGENEA | 19,166 | 19,166.00 | 294.087 | 582.43 | 2024 (19.77 %) | 0 |
| z_ewma_60_raw | POS | RELOJ_X_FACTOR_ANIO_PREAUDITORIA | 19,166 | 19,166.99 | 0.270 | 10.13 | 2020 (0.27 %) | 16,480 |

- *Minutos sin ningún extremo observado*: su probabilidad es 0 (columna "barras con p = 0"). Es lo coherente con lo observado, pero significa que la referencia no puede "esperar" extremos ahí.
- *Combinaciones minuto–año sin barras*: no intervienen (no hay exposición).
- *Celdas pequeñas*: el ajuste usa sólo totales por minuto y por año (dos márgenes), no una tasa por cada combinación, precisamente para no estimar miles de tasas con dos o tres casos cada una.
- Referencias secundarias: `RELOJ` (sin año), `HOMOGENEA` (una única tasa: el Poisson literal) y `RELOJ_X_FACTOR_ANIO_PREAUDITORIA` (la fórmula original, sólo para comparar). No se estima el índice extremal (G4).

**Qué cubren los intervalos (hallazgo C).** Hay dos tipos y no son intercambiables:

- **IC condicional**: remuestrea jornadas pero deja fijos los umbrales y la referencia estimados con toda la muestra. Mide sólo la variabilidad de los conteos. Es el que acompaña a todas las tablas de sensibilidad.
- **IC con re-estimación** (sólo para la variante primaria): en cada réplica se vuelven a estimar los umbrales de cada celda y la referencia por minuto y año sobre las jornadas remuestreadas. 300 réplicas.
- **Ninguno de los dos cubre**: la estimación de `sigma_hat`, `z` y el decil de volatilidad (son atributos de cada barra que dependen de su propio pasado; remuestrear jornadas no los regenera), los cortes de segmento de TDA-06, ni la elección del estimador de volatilidad. Tampoco la dependencia entre jornadas. Son, por tanto, intervalos **condicionados a esas piezas**, no la incertidumbre total.

## 7. Resultado: crudo frente a estandarizado

**Variante primaria — la que decide** (definición `SEGMENT_X_VOL`, nivel 1 %, referencia `RELOJ_Y_ANIO`, tiempos ≤ 10 barras):

| serie | lado | observado | esperado si independientes | cociente | IC95 con re-estimación | IC95 condicional (previo) | banda calibrada (sup.) | banda Bernoulli previa (sup.) | ¿detectado? |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r_1m | NEG | 32.18 % | 17.27 % | 1.863 | [1.791, 1.908] | [1.800, 1.928] | 1.048 | 1.035 | sí |
| r_1m | POS | 29.49 % | 18.87 % | 1.563 | [1.508, 1.604] | [1.498, 1.634] | 1.057 | 1.032 | sí |
| r_1m | BOTH | 44.28 % | 29.76 % | 1.488 | [1.458, 1.514] | [1.444, 1.535] | 1.037 | 1.016 | sí |
| z_ewma_60_raw | NEG | 28.57 % | 16.48 % | 1.734 | [1.684, 1.789] | [1.693, 1.778] | 1.043 | 1.038 | sí |
| z_ewma_60_raw | POS | 25.44 % | 18.05 % | 1.409 | [1.369, 1.438] | [1.377, 1.445] | 1.044 | 1.034 | sí |
| z_ewma_60_raw | BOTH | 39.09 % | 28.69 % | 1.363 | [1.342, 1.379] | [1.343, 1.386] | 1.023 | 1.017 | sí |

La anchura del IC con re-estimación es entre 0.62 y 1.23 veces la del condicional (menor que 1 = más estrecho). Re-estimar umbrales y referencia en cada réplica no tiene por qué ensanchar el intervalo: al recalcular el umbral, cada réplica vuelve a tener la misma tasa de extremos, y eso estabiliza el cociente. Ninguno de los dos intervalos es "el bueno" en abstracto; responden a preguntas distintas (§6).

**Comparación directa entre crudo y estandarizado** (mismas jornadas remuestreadas para ambos):

| lado | cociente r − cociente z [IC95] | parte del exceso que desaparece al estandarizar [IC95] |
|---:|---:|---:|
| NEG | 0.129 [0.066, 0.170] | 15 % [8 %, 19 %] |
| POS | 0.153 [0.111, 0.190] | 27 % [21 %, 33 %] |
| BOTH | 0.125 [0.095, 0.148] | 26 % [21 %, 30 %] |

**El mismo cociente según cómo se define el extremo** (cuánto del agrupamiento aparente es hora del día y volatilidad; IC condicional):

| serie | definición | lado | observado | esperado si independientes | cociente [IC95 condicional] | excedencias | tiempos medidos |
|---:|---:|---:|---:|---:|---:|---:|---:|
| r_1m | GLOBAL | NEG | 60.98 % | 30.02 % | 2.03 [1.93, 2.12] | 19,135 | 17,830 |
| r_1m | GLOBAL | POS | 61.24 % | 30.18 % | 2.03 [1.92, 2.12] | 19,135 | 17,842 |
| r_1m | SEGMENT | NEG | 57.50 % | 19.26 % | 2.99 [2.76, 3.19] | 19,138 | 17,879 |
| r_1m | SEGMENT | POS | 56.86 % | 19.87 % | 2.86 [2.62, 3.08] | 19,138 | 17,897 |
| r_1m | SEGMENT_X_VOL | NEG | 32.18 % | 17.27 % | 1.86 [1.80, 1.93] | 19,166 | 17,575 |
| r_1m | SEGMENT_X_VOL | POS | 29.49 % | 18.87 % | 1.56 [1.50, 1.63] | 19,166 | 17,552 |
| z_ewma_60_raw | GLOBAL | NEG | 31.38 % | 21.10 % | 1.49 [1.46, 1.52] | 19,135 | 17,582 |
| z_ewma_60_raw | GLOBAL | POS | 28.31 % | 22.95 % | 1.23 [1.20, 1.26] | 19,135 | 17,571 |
| z_ewma_60_raw | SEGMENT | NEG | 29.25 % | 17.53 % | 1.67 [1.63, 1.71] | 19,138 | 17,522 |
| z_ewma_60_raw | SEGMENT | POS | 26.26 % | 19.13 % | 1.37 [1.33, 1.40] | 19,138 | 17,515 |
| z_ewma_60_raw | SEGMENT_X_VOL | NEG | 28.57 % | 16.48 % | 1.73 [1.69, 1.78] | 19,166 | 17,551 |
| z_ewma_60_raw | SEGMENT_X_VOL | POS | 25.44 % | 18.05 % | 1.41 [1.38, 1.44] | 19,166 | 17,532 |

**Otras ventanas y otros estadísticos** (definición y nivel primarios; cociente observado/esperado [IC 95 % condicional]):

| serie | lado | ≤1 barras | ≤5 barras | ≤10 barras | ≤60 barras | en rachas ≥2 | longitud media de racha | dispersión diaria |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r_1m | NEG | 3.02 [2.85, 3.18] | 2.16 [2.07, 2.24] | 1.86 [1.80, 1.93] | 1.20 [1.18, 1.22] | 2.95 [2.81, 3.12] | 1.05 [1.05, 1.06] | 2.35 [1.89, 2.76] |
| r_1m | POS | 2.16 [2.02, 2.31] | 1.71 [1.62, 1.80] | 1.56 [1.50, 1.63] | 1.16 [1.14, 1.18] | 2.14 [2.00, 2.28] | 1.03 [1.03, 1.04] | 2.26 [1.85, 2.70] |
| r_1m | BOTH | 2.20 [2.09, 2.33] | 1.70 [1.64, 1.77] | 1.49 [1.44, 1.53] | 1.07 [1.06, 1.08] | 2.15 [2.05, 2.25] | 1.07 [1.06, 1.07] | 2.48 [2.01, 2.99] |
| z_ewma_60_raw | NEG | 2.95 [2.79, 3.09] | 2.05 [1.99, 2.12] | 1.73 [1.69, 1.78] | 1.15 [1.14, 1.17] | 2.89 [2.74, 3.02] | 1.05 [1.04, 1.05] | 1.52 [1.42, 1.64] |
| z_ewma_60_raw | POS | 2.01 [1.89, 2.11] | 1.55 [1.50, 1.60] | 1.41 [1.38, 1.44] | 1.10 [1.08, 1.11] | 2.00 [1.88, 2.10] | 1.03 [1.02, 1.03] | 1.20 [1.12, 1.29] |
| z_ewma_60_raw | BOTH | 2.08 [2.01, 2.14] | 1.56 [1.53, 1.59] | 1.36 [1.34, 1.39] | 1.04 [1.03, 1.04] | 2.02 [1.96, 2.08] | 1.06 [1.05, 1.06] | 1.48 [1.38, 1.60] |

- *En rachas ≥2*: fracción de los extremos que forman parte de una racha de dos o más barras consecutivas.
- *Dispersión diaria*: variabilidad del número de extremos por jornada frente a la esperada; mayor que 1 = hay jornadas que concentran muchos más extremos de lo previsto.

**Sensibilidad al nivel de cola:**

| serie | nivel | lado | observado | esperado si independientes | cociente [IC95 condicional] | excedencias | tiempos medidos |
|---:|---:|---:|---:|---:|---:|---:|---:|
| r_1m | 1 % | NEG | 32.18 % | 17.27 % | 1.86 [1.80, 1.93] | 19,166 | 17,575 |
| r_1m | 1 % | POS | 29.49 % | 18.87 % | 1.56 [1.50, 1.63] | 19,166 | 17,552 |
| r_1m | 0.1 % | NEG | 22.86 % | 5.13 % | 4.46 [3.85, 5.09] | 1,950 | 1,041 |
| r_1m | 0.1 % | POS | 18.32 % | 6.47 % | 2.83 [2.40, 3.26] | 1,950 | 1,026 |
| r_1m | 5 % | NEG | 55.05 % | 45.58 % | 1.21 [1.19, 1.22] | 95,703 | 93,679 |
| r_1m | 5 % | POS | 53.20 % | 46.16 % | 1.15 [1.14, 1.17] | 95,702 | 93,628 |
| z_ewma_60_raw | 1 % | NEG | 28.57 % | 16.48 % | 1.73 [1.69, 1.78] | 19,166 | 17,551 |
| z_ewma_60_raw | 1 % | POS | 25.44 % | 18.05 % | 1.41 [1.38, 1.44] | 19,166 | 17,532 |
| z_ewma_60_raw | 0.1 % | NEG | 19.83 % | 5.04 % | 3.94 [3.44, 4.43] | 1,950 | 928 |
| z_ewma_60_raw | 0.1 % | POS | 15.13 % | 6.19 % | 2.44 [2.08, 2.79] | 1,950 | 932 |
| z_ewma_60_raw | 5 % | NEG | 51.61 % | 44.71 % | 1.15 [1.15, 1.16] | 95,703 | 93,695 |
| z_ewma_60_raw | 5 % | POS | 48.78 % | 45.12 % | 1.08 [1.07, 1.09] | 95,703 | 93,663 |

**Sensibilidad a la referencia** (incluye la fórmula original):

| serie | referencia | lado | observado | esperado si independientes | cociente [IC95 condicional] |
|---:|---:|---:|---:|---:|---:|
| r_1m | RELOJ_Y_ANIO | NEG | 32.18 % | 17.27 % | 1.86 [1.80, 1.93] |
| r_1m | RELOJ | NEG | 32.18 % | 15.22 % | 2.11 [2.05, 2.19] |
| r_1m | HOMOGENEA | NEG | 32.18 % | 10.45 % | 3.08 [2.98, 3.19] |
| r_1m | RELOJ_X_FACTOR_ANIO_PREAUDITORIA | NEG | 32.18 % | 17.31 % | 1.86 [1.80, 1.92] |
| r_1m | RELOJ_Y_ANIO | POS | 29.49 % | 18.87 % | 1.56 [1.50, 1.63] |
| r_1m | RELOJ | POS | 29.49 % | 16.33 % | 1.81 [1.72, 1.88] |
| r_1m | HOMOGENEA | POS | 29.49 % | 10.45 % | 2.82 [2.69, 2.95] |
| r_1m | RELOJ_X_FACTOR_ANIO_PREAUDITORIA | POS | 29.49 % | 18.92 % | 1.56 [1.49, 1.63] |
| z_ewma_60_raw | RELOJ_Y_ANIO | NEG | 28.57 % | 16.48 % | 1.73 [1.69, 1.78] |
| z_ewma_60_raw | RELOJ | NEG | 28.57 % | 16.38 % | 1.74 [1.70, 1.79] |
| z_ewma_60_raw | HOMOGENEA | NEG | 28.57 % | 10.45 % | 2.74 [2.67, 2.80] |
| z_ewma_60_raw | RELOJ_X_FACTOR_ANIO_PREAUDITORIA | NEG | 28.57 % | 16.48 % | 1.73 [1.69, 1.78] |
| z_ewma_60_raw | RELOJ_Y_ANIO | POS | 25.44 % | 18.05 % | 1.41 [1.38, 1.44] |
| z_ewma_60_raw | RELOJ | POS | 25.44 % | 17.90 % | 1.42 [1.39, 1.46] |
| z_ewma_60_raw | HOMOGENEA | POS | 25.44 % | 10.45 % | 2.44 [2.38, 2.50] |
| z_ewma_60_raw | RELOJ_X_FACTOR_ANIO_PREAUDITORIA | POS | 25.44 % | 18.06 % | 1.41 [1.38, 1.44] |

**Sensibilidad a la estandarización** (otras semividas EWMA ya definidas en TDA-10; mismas barras y mismas celdas):

| serie | semivida EWMA (min) | lado | observado | esperado si independientes | cociente [IC95 condicional] |
|---:|---:|---:|---:|---:|---:|
| z_ewma_60_raw | 60.0000 | NEG | 28.57 % | 16.48 % | 1.73 [1.69, 1.78] |
| z_ewma_60_raw | 60.0000 | POS | 25.44 % | 18.05 % | 1.41 [1.38, 1.44] |
| z_ewma_20_raw | 20.0000 | NEG | 18.88 % | 13.39 % | 1.41 [1.37, 1.45] |
| z_ewma_20_raw | 20.0000 | POS | 17.19 % | 14.51 % | 1.18 [1.14, 1.22] |
| z_ewma_240_raw | 240.0000 | NEG | 35.27 % | 17.10 % | 2.06 [2.01, 2.12] |
| z_ewma_240_raw | 240.0000 | POS | 31.12 % | 18.40 % | 1.69 [1.65, 1.74] |


Las tablas de sensibilidad son **descriptivas**: sólo la variante primaria tiene banda calibrada e IC con re-estimación.

## 8. Tiempos entre excedencias y longitud de las rachas

Definición y nivel primarios, referencia `RELOJ_Y_ANIO`. "Esperado" es lo que daría la independencia con la misma intensidad.

**`r_1m` — caídas: tiempos entre excedencias (barras)**

| barras | observado (n) | observado (%) | esperado (n) | esperado (%) | observado / esperado |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,362 | 7.75 % | 449.6 | 2.57 % | 3.03 |
| 2 | 782 | 4.45 % | 374.3 | 2.14 % | 2.09 |
| 3-5 | 1,732 | 9.85 % | 964.0 | 5.51 % | 1.80 |
| 6-10 | 1,779 | 10.12 % | 1,233.2 | 7.05 % | 1.44 |
| 11-30 | 3,456 | 19.66 % | 3,355.8 | 19.18 % | 1.03 |
| 31-60 | 2,475 | 14.08 % | 3,234.4 | 18.49 % | 0.77 |
| >60 | 5,989 | 34.08 % | 7,883.1 | 45.06 % | 0.76 |

**`r_1m` — subidas**

| barras | observado (n) | observado (%) | esperado (n) | esperado (%) | observado / esperado |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,116 | 6.36 % | 514.7 | 2.94 % | 2.17 |
| 2 | 720 | 4.10 % | 432.7 | 2.47 % | 1.66 |
| 3-5 | 1,600 | 9.12 % | 1,058.7 | 6.05 % | 1.51 |
| 6-10 | 1,740 | 9.91 % | 1,295.8 | 7.41 % | 1.34 |
| 11-30 | 3,528 | 20.10 % | 3,376.2 | 19.29 % | 1.04 |
| 31-60 | 2,717 | 15.48 % | 3,167.2 | 18.10 % | 0.86 |
| >60 | 6,131 | 34.93 % | 7,653.0 | 43.74 % | 0.80 |

**`z_ewma_60_raw` — caídas**

| barras | observado (n) | observado (%) | esperado (n) | esperado (%) | observado / esperado |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,260 | 7.18 % | 425.9 | 2.43 % | 2.96 |
| 2 | 727 | 4.14 % | 358.8 | 2.05 % | 2.03 |
| 3-5 | 1,537 | 8.76 % | 926.4 | 5.29 % | 1.66 |
| 6-10 | 1,491 | 8.50 % | 1,172.6 | 6.70 % | 1.27 |
| 11-30 | 3,128 | 17.82 % | 3,184.3 | 18.20 % | 0.98 |
| 31-60 | 2,572 | 14.65 % | 3,182.0 | 18.18 % | 0.81 |
| >60 | 6,836 | 38.95 % | 8,248.6 | 47.14 % | 0.83 |

**`z_ewma_60_raw` — subidas**

| barras | observado (n) | observado (%) | esperado (n) | esperado (%) | observado / esperado |
|---:|---:|---:|---:|---:|---:|
| 1 | 975 | 5.56 % | 483.7 | 2.76 % | 2.02 |
| 2 | 653 | 3.72 % | 415.6 | 2.37 % | 1.57 |
| 3-5 | 1,354 | 7.72 % | 1,019.3 | 5.82 % | 1.33 |
| 6-10 | 1,479 | 8.44 % | 1,240.7 | 7.09 % | 1.19 |
| 11-30 | 3,162 | 18.04 % | 3,184.7 | 18.20 % | 0.99 |
| 31-60 | 2,713 | 15.47 % | 3,077.9 | 17.59 % | 0.88 |
| >60 | 7,196 | 41.04 % | 8,078.6 | 46.16 % | 0.89 |

**Longitud de las rachas (barras extremas consecutivas, ambas colas)**

`r_1m`:

| longitud | observado (n) | observado (%) | esperado (n) | esperado (%) | observado / esperado |
|---:|---:|---:|---:|---:|---:|
| 1 | 30,631 | 89.86 % | 34,747.5 | 95.43 % | 0.88 |
| 2 | 2,863 | 8.40 % | 1,462.8 | 4.02 % | 1.96 |
| 3 | 466 | 1.37 % | 158.0 | 0.43 % | 2.95 |
| 4 | 91 | 0.27 % | 30.8 | 0.08 % | 2.95 |
| >=5 | 37 | 0.11 % | 11.4 | 0.03 % | 3.25 |

`z_ewma_60_raw`:

| longitud | observado (n) | observado (%) | esperado (n) | esperado (%) | observado / esperado |
|---:|---:|---:|---:|---:|---:|
| 1 | 31,415 | 90.90 % | 34,909.8 | 95.59 % | 0.90 |
| 2 | 2,651 | 7.67 % | 1,439.8 | 3.94 % | 1.84 |
| 3 | 395 | 1.14 % | 142.7 | 0.39 % | 2.77 |
| 4 | 70 | 0.20 % | 22.3 | 0.06 % | 3.14 |
| >=5 | 28 | 0.08 % | 4.9 | 0.01 % | 5.77 |

Tablas completas (todas las definiciones, lados y referencias) en `TDA13_tiempos_entre_excedencias.csv` y `TDA13_longitud_rachas.csv`; gráfico en `TDA13_tiempos_entre_excedencias.png`.

## 9. Por año y por segmento (G3)

Mismo cociente (tiempos ≤ 10 barras), definición y nivel primarios, IC condicional. ⚠ = menos de 200 tiempos medidos; ✖ = menos de 20.

**Por año**

| grupo | lado | r_1m: cociente [IC95] | r_1m: tiempos | z_ewma_60_raw: cociente [IC95] | z_ewma_60_raw: tiempos |
|---:|---:|---:|---:|---:|---:|
| 2019 | NEG | NaN [NaN, NaN] ✖ | 0 | 1.81 [0.87, 2.30] ⚠ | 56 |
| 2019 | POS | NaN [NaN, NaN] ✖ | 0 | 1.11 [0.77, 1.40] ⚠ | 58 |
| 2020 | NEG | 1.64 [1.56, 1.73] | 6,130 | 1.72 [1.62, 1.82] | 3,340 |
| 2020 | POS | 1.40 [1.31, 1.49] | 6,350 | 1.31 [1.23, 1.39] | 3,445 |
| 2021 | NEG | 2.03 [1.90, 2.16] | 3,163 | 1.82 [1.71, 1.93] | 3,206 |
| 2021 | POS | 1.56 [1.45, 1.65] | 2,995 | 1.40 [1.31, 1.48] | 3,107 |
| 2022 | NEG | 1.99 [1.85, 2.13] | 2,721 | 1.67 [1.58, 1.78] | 3,350 |
| 2022 | POS | 1.79 [1.68, 1.90] | 3,005 | 1.48 [1.41, 1.54] | 3,698 |
| 2023 | NEG | 1.97 [1.82, 2.10] | 2,535 | 1.77 [1.65, 1.90] | 2,694 |
| 2023 | POS | 1.72 [1.61, 1.85] | 2,713 | 1.49 [1.39, 1.58] | 2,924 |
| 2024 | NEG | 2.33 [2.12, 2.54] | 1,923 | 1.73 [1.60, 1.84] | 3,124 |
| 2024 | POS | 1.97 [1.77, 2.17] | 1,516 | 1.42 [1.31, 1.53] | 2,672 |
| 2025 | NEG | 2.50 [2.11, 2.93] | 1,103 | 1.69 [1.55, 1.82] | 1,781 |
| 2025 | POS | 2.02 [1.48, 2.63] | 973 | 1.34 [1.23, 1.45] | 1,628 |

**Por segmento horario**

| grupo | lado | r_1m: cociente [IC95] | r_1m: tiempos | z_ewma_60_raw: cociente [IC95] | z_ewma_60_raw: tiempos |
|---:|---:|---:|---:|---:|---:|
| 00:00-02:00 | NEG | 1.66 [1.54, 1.79] | 879 | 1.72 [1.58, 1.87] | 782 |
| 00:00-02:00 | POS | 1.55 [1.42, 1.69] | 850 | 1.65 [1.51, 1.77] | 749 |
| 02:00-03:00 | NEG | 1.40 [1.27, 1.53] | 311 | 1.50 [1.37, 1.66] | 287 |
| 02:00-03:00 | POS | 1.25 [1.12, 1.39] | 305 | 1.24 [1.08, 1.40] | 250 |
| 03:00-08:30 | NEG | 1.68 [1.59, 1.78] | 3,405 | 1.67 [1.58, 1.77] | 3,336 |
| 03:00-08:30 | POS | 1.39 [1.31, 1.47] | 3,388 | 1.31 [1.23, 1.38] | 3,275 |
| 08:30-09:30 | NEG | 1.64 [1.51, 1.75] | 296 | 1.63 [1.46, 1.77] | 265 |
| 08:30-09:30 | POS | 1.52 [1.39, 1.64] | 286 | 1.60 [1.46, 1.73] | 263 |
| 09:30-16:02 | NEG | 1.58 [1.50, 1.66] | 4,125 | 1.34 [1.28, 1.39] | 4,062 |
| 09:30-16:02 | POS | 1.35 [1.28, 1.42] | 4,120 | 1.12 [1.07, 1.17] | 4,046 |
| 16:02-20:00 | NEG | 2.07 [1.97, 2.18] | 1,357 | 2.17 [2.05, 2.28] | 1,285 |
| 16:02-20:00 | POS | 1.86 [1.75, 1.96] | 1,294 | 1.87 [1.76, 1.98] | 1,207 |
| 20:00-24:00 | NEG | 2.18 [2.05, 2.31] | 2,298 | 2.20 [2.06, 2.33] | 2,190 |
| 20:00-24:00 | POS | 1.77 [1.62, 1.90] | 2,284 | 1.78 [1.66, 1.90] | 2,159 |

Dentro de un segmento sólo se miden los tiempos entre extremos del mismo segmento, así que los segmentos cortos tienen pocas medidas. Detalle (incluye la ventana de 1 barra y la fracción en rachas) en `TDA13_agrupamiento_por_anio_segmento.csv`.

## 10. Distribución horaria de los extremos

Qué parte de los extremos cae en cada hora de Nueva York, según cómo se defina el umbral (ambas colas, nivel 1 %). Si los extremos no dependieran de la hora, cada columna se parecería a "% de las barras".

| hora NY | % de las barras | r · GLOBAL: % de los extremos | r · SEGMENT: % de los extremos | r · SEGMENT_X_VOL: % de los extremos | z · SEGMENT_X_VOL: % de los extremos |
|---:|---:|---:|---:|---:|---:|
| 0 | 4.4 % | 0.8 % | 3.7 % | 3.0 % | 3.0 % |
| 1 | 4.4 % | 1.1 % | 5.1 % | 5.8 % | 5.9 % |
| 2 | 4.4 % | 1.6 % | 4.4 % | 4.4 % | 4.4 % |
| 3 | 4.4 % | 2.7 % | 6.4 % | 9.0 % | 9.6 % |
| 4 | 4.4 % | 2.1 % | 4.7 % | 4.0 % | 4.0 % |
| 5 | 4.4 % | 1.4 % | 3.4 % | 2.2 % | 2.0 % |
| 6 | 4.4 % | 1.5 % | 3.4 % | 2.9 % | 2.7 % |
| 7 | 4.4 % | 1.9 % | 4.0 % | 3.8 % | 3.8 % |
| 8 | 4.4 % | 4.4 % | 5.5 % | 5.5 % | 5.4 % |
| 9 | 4.4 % | 16.0 % | 7.6 % | 12.9 % | 14.4 % |
| 10 | 4.4 % | 16.5 % | 6.3 % | 4.9 % | 5.1 % |
| 11 | 4.4 % | 9.1 % | 3.3 % | 1.8 % | 1.4 % |
| 12 | 4.4 % | 6.2 % | 2.2 % | 1.3 % | 1.0 % |
| 13 | 4.2 % | 5.9 % | 2.2 % | 1.7 % | 1.3 % |
| 14 | 4.2 % | 8.0 % | 3.4 % | 2.7 % | 2.4 % |
| 15 | 4.2 % | 9.9 % | 4.0 % | 3.5 % | 3.2 % |
| 16 | 3.9 % | 3.0 % | 5.0 % | 3.2 % | 3.4 % |
| 17 | 0.1 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| 18 | 4.2 % | 2.2 % | 4.9 % | 4.8 % | 4.6 % |
| 19 | 4.4 % | 1.1 % | 2.9 % | 4.9 % | 5.0 % |
| 20 | 4.4 % | 1.6 % | 6.0 % | 5.2 % | 5.4 % |
| 21 | 4.4 % | 1.3 % | 4.9 % | 5.2 % | 5.0 % |
| 22 | 4.4 % | 1.0 % | 3.7 % | 3.7 % | 3.6 % |
| 23 | 4.4 % | 0.8 % | 2.9 % | 3.4 % | 3.5 % |

Tasa de extremos de la hora más cargada respecto de la tasa media ("lift"), y de la menos cargada: con umbral global, de 0.07 a 3.77 veces; por segmento, de 0.37 a 1.74; con la definición primaria, de 0.29 a 2.95. Tasas con IC por hora en `TDA13_distribucion_horaria.csv`; gráfico en `TDA13_distribucion_horaria.png`.

**Cómo leerlo.** Con un umbral único, los "extremos" son sobre todo las horas de más actividad: eso no describe eventos, describe el horario. Con umbrales relativos el reparto se acerca al de las barras, pero no queda perfectamente plano: los cortes de segmento son anchos y dentro de un segmento sigue habiendo minutos más agitados que otros. Por eso la referencia de independencia usa la tasa de cada minuto del día y no una tasa única.

## 11. Efecto del triage sobre la curtosis y los cuantiles extremos

*Curtosis* mide cuánto pesan los valores extremos en la distribución (0 = una campana normal). Cada fila recalcula los estadísticos quitando una categoría; los resultados de TDA-07 y TDA-12 no se modifican. Entre corchetes, IC 95 % por jornadas. "quitadas" = observaciones retiradas respecto del escenario de referencia de su bloque.

**Retorno crudo**

| escenario | n | quitadas | curtosis | curtosis recortada | p0,1 | p1 | p99 | p99,9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UNIVERSO_COMPLETO | 1,918,049 | 0 | 1,790.9 [224.8, 4,322.9] | 11.84 [10.14, 13.79] | -0.00279 [-0.00299, -0.00260] | -0.00124 [-0.00128, -0.00118] | 0.00120 [0.00115, 0.00125] | 0.00283 [0.00259, 0.00306] |
| UNIVERSO_sin_ERROR_DATO | 1,918,049 | 0 | 1,790.9 [224.8, 4,322.9] | 11.84 [10.14, 13.79] | -0.00279 [-0.00299, -0.00260] | -0.00124 [-0.00128, -0.00118] | 0.00120 [0.00115, 0.00125] | 0.00283 [0.00259, 0.00306] |
| UNIVERSO_sin_ARTEFACTO_ROLL | 1,918,028 | 21 | 345.4 [171.8, 551.7] | 11.80 [10.12, 13.73] | -0.00278 [-0.00298, -0.00260] | -0.00124 [-0.00128, -0.00118] | 0.00120 [0.00115, 0.00125] | 0.00282 [0.00259, 0.00305] |
| UNIVERSO_sin_ARTEFACTO_FRONTERA | 1,917,441 | 608 | 1,724.2 [96.0, 4,518.4] | 11.52 [9.87, 13.37] | -0.00275 [-0.00294, -0.00256] | -0.00123 [-0.00128, -0.00118] | 0.00119 [0.00114, 0.00124] | 0.00279 [0.00255, 0.00303] |
| UNIVERSO_sin_EVENTO_REAL | 1,884,845 | 33,204 | 3,121.1 [263.9, 7,602.2] | 8.04 [7.18, 8.82] | -0.00218 [-0.00229, -0.00206] | -0.00109 [-0.00112, -0.00104] | 0.00106 [0.00102, 0.00109] | 0.00222 [0.00211, 0.00233] |
| UNIVERSO_sin_INDETERMINADO | 1,912,921 | 5,128 | 1,946.4 [236.1, 4,717.4] | 11.01 [9.64, 12.52] | -0.00267 [-0.00284, -0.00250] | -0.00121 [-0.00126, -0.00116] | 0.00118 [0.00113, 0.00122] | 0.00270 [0.00250, 0.00289] |
| SOLO_VALIDOS | 1,914,530 | 0 | 111.6 [70.0, 163.9] | 11.49 [9.86, 13.32] | -0.00275 [-0.00294, -0.00256] | -0.00123 [-0.00128, -0.00118] | 0.00119 [0.00114, 0.00124] | 0.00279 [0.00255, 0.00302] |
| VALIDOS_sin_INDETERMINADO | 1,909,402 | 5,128 | 114.1 [70.1, 169.6] | 10.68 [9.38, 12.08] | -0.00263 [-0.00279, -0.00247] | -0.00121 [-0.00125, -0.00116] | 0.00117 [0.00112, 0.00122] | 0.00265 [0.00247, 0.00284] |
| VALIDOS_sin_EVENTO_REAL | 1,881,326 | 33,204 | 23.8 [15.5, 33.1] | 7.82 [6.99, 8.57] | -0.00216 [-0.00226, -0.00203] | -0.00108 [-0.00112, -0.00104] | 0.00105 [0.00101, 0.00109] | 0.00219 [0.00207, 0.00229] |
| VALIDOS_sin_EXTREMOS | 1,876,198 | 38,332 | 9.5 [8.9, 10.1] | 7.21 [6.54, 7.78] | -0.00206 [-0.00214, -0.00195] | -0.00106 [-0.00109, -0.00102] | 0.00103 [0.00099, 0.00106] | 0.00209 [0.00199, 0.00217] |

**Retorno estandarizado `z`**

| escenario | n | quitadas | curtosis | curtosis recortada | p0,1 | p1 | p99 | p99,9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Z_TODO | 1,914,350 | 0 | 154.3 [97.5, 229.5] | 2.82 [2.76, 2.89] | -5.165 [-5.226, -5.114] | -2.728 [-2.739, -2.715] | 2.596 [2.585, 2.605] | 4.928 [4.865, 4.988] |
| Z_sin_INDETERMINADO | 1,909,222 | 5,128 | 161.6 [101.1, 241.2] | 2.60 [2.55, 2.67] | -5.028 [-5.085, -4.969] | -2.663 [-2.676, -2.650] | 2.542 [2.534, 2.552] | 4.774 [4.716, 4.839] |
| Z_sin_EVENTO_REAL | 1,881,146 | 33,204 | 4.8 [2.3, 7.8] | 0.81 [0.79, 0.83] | -3.433 [-3.470, -3.402] | -2.324 [-2.335, -2.315] | 2.235 [2.228, 2.244] | 3.243 [3.214, 3.279] |

**Lectura.**
- *Artefactos.* Con todos los retornos de la tabla, incluidos los que cruzan fronteras (`UNIVERSO_COMPLETO`), la curtosis es 1,790.9. Quitar sólo los extremos de roll la deja en 345.4, y quitar sólo los de frontera en 1,724.2. La población que usan TDA-07 y TDA-12 (`SOLO_VALIDOS`, curtosis 111.6) ya excluye todos esos retornos: sus resultados no estaban contaminados por ellos.
- *Extremos clasificados `EVENTO_REAL`.* Quitarlos de los válidos deja la curtosis en 23.8, y quitar todos los extremos inventariados, en 9.5: la cola pesada **es** el conjunto de estos movimientos, no un añadido. Borrarlos no "limpia" la serie; cambia lo que se está midiendo.
- *Sensibilidad a `INDETERMINADO`.* Sin ellos la curtosis pasa de 111.6 a 114.1. La tabla siguiente da el rango entre conservarlos y quitarlos:

| serie | estadístico | con INDETERMINADO | sin INDETERMINADO | cambio relativo |
|---:|---:|---:|---:|---:|
| r_1m | curtosis | 111.56 | 114.08 | 2.3 % |
| r_1m | curtosis recortada | 11.49 | 10.68 | -7.1 % |
| r_1m | percentil 0,1 | -0.00275 | -0.00263 | -4.2 % |
| r_1m | percentil 1 | -0.00123 | -0.00121 | -2.0 % |
| r_1m | percentil 99 | 0.00119 | 0.00117 | -1.8 % |
| r_1m | percentil 99,9 | 0.00279 | 0.00265 | -4.9 % |
| z_ewma_60_raw | curtosis | 154.35 | 161.55 | 4.7 % |
| z_ewma_60_raw | curtosis recortada | 2.82 | 2.60 | -7.7 % |
| z_ewma_60_raw | percentil 0,1 | -5.165 | -5.028 | -2.7 % |
| z_ewma_60_raw | percentil 1 | -2.728 | -2.663 | -2.4 % |
| z_ewma_60_raw | percentil 99 | 2.596 | 2.542 | -2.1 % |
| z_ewma_60_raw | percentil 99,9 | 4.928 | 4.774 | -3.1 % |

Quitar una categoría entera de extremos recorta mecánicamente la cola (son, por definición, los valores más grandes), así que estas cifras son cotas de sensibilidad, no estimaciones "corregidas".

**Trazabilidad.** 14 de 14 valores recalculados coinciden con los publicados por TDA-07 y TDA-12 (diferencia máxima 2.84e-14). Tabla completa, con diferencias e IC, en `TDA13_efecto_triage.csv`.

## 12. Calibración de la maquinaria (G2) — revisada en la auditoría

**Qué estaba mal (hallazgo B).** La versión anterior daba la calibración por buena si la *media* de los controles caía dentro de la banda. Eso no dice con qué frecuencia la regla "detecta" algo que no existe. Además, la banda salía de simular extremos independientes con probabilidades fijas, ignorando que los umbrales, los estados de volatilidad y las propias tasas se estiman de los datos.

**Diagnóstico de la causa.** Cociente (tiempos ≤ 10 barras) en mundos donde no hay ninguna dependencia:

| serie | lado | simulación | n | media | desviación | p97,5 |
|---:|---:|---:|---:|---:|---:|---:|
| r_1m | NEG | Excedencias independientes, tasa conocida (banda previa) | 200 | 0.999 | 0.015 | 1.027 |
| r_1m | NEG | Excedencias independientes, tasa re-estimada | 200 | 0.999 | 0.013 | 1.023 |
| r_1m | NEG | Procedimiento completo: PERMUTACION_GLOBAL | 1,000 | 0.999 | 0.021 | 1.039 |
| r_1m | NEG | Procedimiento completo: PERMUTACION_MINUTO_X_ANIO | 1,000 | 1.014 | 0.015 | 1.045 |
| r_1m | NEG | Procedimiento completo: SIMULADO_PERFIL_RELOJ | 1,000 | 1.011 | 0.019 | 1.048 |
| r_1m | POS | Excedencias independientes, tasa conocida (banda previa) | 200 | 1.000 | 0.016 | 1.035 |
| r_1m | POS | Excedencias independientes, tasa re-estimada | 200 | 1.001 | 0.012 | 1.024 |
| r_1m | POS | Procedimiento completo: PERMUTACION_GLOBAL | 1,000 | 0.999 | 0.021 | 1.040 |
| r_1m | POS | Procedimiento completo: PERMUTACION_MINUTO_X_ANIO | 1,000 | 1.029 | 0.015 | 1.058 |
| r_1m | POS | Procedimiento completo: SIMULADO_PERFIL_RELOJ | 1,000 | 1.016 | 0.019 | 1.051 |
| z_ewma_60_raw | NEG | Excedencias independientes, tasa conocida (banda previa) | 200 | 0.998 | 0.018 | 1.035 |
| z_ewma_60_raw | NEG | Excedencias independientes, tasa re-estimada | 200 | 0.999 | 0.015 | 1.026 |
| z_ewma_60_raw | NEG | Procedimiento completo: PERMUTACION_GLOBAL | 1,000 | 1.006 | 0.021 | 1.045 |
| z_ewma_60_raw | NEG | Procedimiento completo: PERMUTACION_MINUTO_X_ANIO | 1,000 | 0.970 | 0.017 | 1.002 |
| z_ewma_60_raw | NEG | Procedimiento completo: SIMULADO_PERFIL_RELOJ | 1,000 | 1.004 | 0.018 | 1.038 |
| z_ewma_60_raw | POS | Excedencias independientes, tasa conocida (banda previa) | 200 | 1.001 | 0.016 | 1.031 |
| z_ewma_60_raw | POS | Excedencias independientes, tasa re-estimada | 200 | 1.000 | 0.011 | 1.019 |
| z_ewma_60_raw | POS | Procedimiento completo: PERMUTACION_GLOBAL | 1,000 | 1.003 | 0.021 | 1.043 |
| z_ewma_60_raw | POS | Procedimiento completo: PERMUTACION_MINUTO_X_ANIO | 1,000 | 0.976 | 0.017 | 1.011 |
| z_ewma_60_raw | POS | Procedimiento completo: SIMULADO_PERFIL_RELOJ | 1,000 | 1.006 | 0.019 | 1.043 |

Las dos primeras filas de cada bloque son extremos independientes "de laboratorio": si el procedimiento completo (filas siguientes) tiene más media o más dispersión que ellas, la banda antigua era demasiado estrecha o estaba desplazada. Con la banda Bernoulli previa, sobre estas mismas réplicas, la tasa por cola llega al 91.1 % [89.2 %, 92.7 %] (`PERMUTACION_MINUTO_X_ANIO`, cola: r_1m BOTH), frente al 2.5 % nominal. La causa compatible con estos números es que, dentro de un mismo minuto del día, la probabilidad de superar el umbral todavía depende del estado de volatilidad de la barra, y ese estado es persistente; la referencia por minuto no lo recoge. Es una explicación coherente con el diagnóstico, no una demostración.

**Qué se hace ahora.** El procedimiento COMPLETO (estandarizar, asignar el estado de volatilidad, fijar umbrales, ajustar la referencia, medir) se aplica a series sin dependencia temporal: (i) `PERMUTACION_GLOBAL`, los retornos reales barajados; (ii) `SIMULADO_PERFIL_RELOJ`, los retornos reales barajados dentro de cada minuto del día (misma volatilidad por hora y misma curtosis, agregadas entre años); (iii) `PERMUTACION_MINUTO_X_ANIO`, los retornos reales barajados dentro de cada minuto del día **y de cada año** (añadida en la segunda auditoría). 1000 réplicas de cada una. La mitad fija la banda; la otra mitad, que no intervino, mide las falsas detecciones.

**Por qué una tercera familia (segunda auditoría, hallazgo H3).** La referencia de independencia condiciona por minuto y por año, pero las familias (i) y (ii) mezclan los años al barajar. Un diagnóstico con series sintéticas *sin ninguna dependencia temporal* mostró que, si la forma del perfil horario cambia de un año a otro, esas dos familias "detectan" agrupamiento en `r` la mayoría de las veces. La familia (iii) conserva la distribución de cada minuto en cada año, que es justo lo que faltaba.

- *Qué conserva (iii):* el conjunto exacto de retornos de cada (minuto de Nueva York, año): su volatilidad, sus colas, el nivel de cada año y la forma del perfil horario de cada año.
- *Qué rompe:* el orden de las jornadas dentro de cada año (rachas de días agitados, día de la semana, deriva dentro del año, días de evento). Eso es exactamente lo que la hipótesis "sin dependencia temporal" niega.
- *Horario de verano:* el minuto es el de la hora local de Nueva York (el mismo `minute_of_day` de TDA-06 que usa la referencia), así que una misma hora de mercado cae siempre en el mismo grupo. El día del cambio de hora de primavera no existen los minutos de la hora saltada, y el de otoño una hora local se repite: esos grupos tienen una barra menos o una más, sin otro efecto.
- *Año:* el de la hora local de la barra (`year_ny`, el de la referencia). Las barras de la tarde del 31 de diciembre pertenecen a la jornada del año siguiente pero se agrupan con su año local.
- *Grupos pequeños:* no se fusionan ni se descartan; un grupo de una sola barra queda fijo. Tamaños reales:

| familia | grupos | barras por grupo: mínimo | mediana | grupos de una sola barra | grupos con menos de 30 barras | barras en esos grupos |
|---:|---:|---:|---:|---:|---:|---:|
| PERMUTACION_GLOBAL | 1 | 1,914,530 | 1,914,530 | 0 | 0 | 0 (0.000 %) |
| SIMULADO_PERFIL_RELOJ | 1,379 | 978 | 1,395 | 0 | 0 | 0 (0.000 %) |
| PERMUTACION_MINUTO_X_ANIO | 9,637 | 3 | 253 | 0 | 1,379 | 7,496 (0.392 %) |

**Cómo se combinan las tres familias, y por qué.** Una cola se da por detectada sólo si el cociente supera la banda de **todas** las familias, es decir, la mayor de las tres. No se da por hecho que "tomar el máximo" lo arregle todo; lo que se puede demostrar es esto:

1. Cada familia es un contraste de permutación. Si los retornos son intercambiables dentro de sus grupos, la serie real es "una permutación más" y su cociente —con toda la re-estimación incluida— es intercambiable con los de las réplicas. Con 500 réplicas para la banda, la probabilidad de superarla es entonces como mucho **2.79 %** por cola (cota exacta; el 2.5 % es el valor nominal, y la diferencia viene de interpolar el percentil).
2. Las tres hipótesis están anidadas: intercambiable en todo ⊂ intercambiable dentro de cada minuto ⊂ intercambiable dentro de cada (minuto, año). Por eso la banda de la familia (iii) es válida por sí sola en los tres mundos.
3. Exigir además las bandas de (i) y (ii) sólo puede subir la banda: es más conservador y pierde potencia, pero nunca es menos válido. Se mantienen porque eran el criterio ya establecido.
4. **Lo que el máximo NO resuelve:** mundos sin dependencia que no sean intercambiables dentro de (minuto, año) —por ejemplo, un perfil horario que cambie de un mes a otro o según el día de la semana—. Para esos no hay garantía. Tampoco corrige la referencia: en un mundo con interacción año × hora el cociente "esperado si independientes" sigue estando mal centrado, y es la banda la que lo absorbe.

**Banda calibrada** (la más exigente de las tres familias):

| serie | lado | PERMUTACION_GLOBAL: media ± desv. | PERMUTACION_GLOBAL: p97,5 | SIMULADO_PERFIL_RELOJ: media ± desv. | SIMULADO_PERFIL_RELOJ: p97,5 | PERMUTACION_MINUTO_X_ANIO: media ± desv. | PERMUTACION_MINUTO_X_ANIO: p97,5 | banda calibrada (sup.) | familia que fija la banda | banda Bernoulli previa (sup.) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r_1m | NEG | 0.999 ± 0.021 | 1.040 | 1.012 ± 0.019 | 1.048 | 1.014 ± 0.015 | 1.045 | 1.048 | SIMULADO_PERFIL_RELOJ | 1.035 |
| r_1m | POS | 0.998 ± 0.020 | 1.037 | 1.016 ± 0.019 | 1.052 | 1.029 ± 0.015 | 1.057 | 1.057 | PERMUTACION_MINUTO_X_ANIO | 1.032 |
| r_1m | BOTH | 1.000 ± 0.010 | 1.019 | 1.012 ± 0.008 | 1.026 | 1.025 ± 0.007 | 1.037 | 1.037 | PERMUTACION_MINUTO_X_ANIO | 1.016 |
| z_ewma_60_raw | NEG | 1.006 ± 0.020 | 1.043 | 1.004 ± 0.018 | 1.041 | 0.970 ± 0.017 | 1.002 | 1.043 | PERMUTACION_GLOBAL | 1.038 |
| z_ewma_60_raw | POS | 1.002 ± 0.021 | 1.041 | 1.006 ± 0.020 | 1.044 | 0.977 ± 0.017 | 1.012 | 1.044 | SIMULADO_PERFIL_RELOJ | 1.034 |
| z_ewma_60_raw | BOTH | 1.004 ± 0.009 | 1.023 | 1.004 ± 0.008 | 1.020 | 0.976 ± 0.008 | 0.991 | 1.023 | PERMUTACION_GLOBAL | 1.017 |

**Criterio de aceptación de G2** (misma regla que antes; al pasar de 2 a 3 familias las tasas que entran son 12 en vez de 8 y el máximo admitido se recalcula con la misma fórmula): en las tres familias sin dependencia, el número de detecciones de cada cola (`r` y `z`, caídas y subidas: 12 tasas) entre las 500 réplicas reservadas no puede superar un máximo (columna "máx. admitido"). Ese máximo es una cota de lo que cabe esperar, con un 95 % de confianza conjunta para las 12 tasas, cuando las dos mitades proceden del mismo mundo; tiene en cuenta que la banda está estimada con un número finito de réplicas (sin eso, el simple error de estimar la banda haría fallar el criterio a un procedimiento correcto). Las reglas globales nunca pueden detectar más que la cola que menos detecta, así que quedan cubiertas; se muestran con su intervalo.

| control | regla | réplicas | detecciones | máx. admitido | tasa [IC95 binomial] | con la banda previa [IC95] | ¿cumple? |
|---:|---:|---:|---:|---:|---:|---:|---:|
| PERMUTACION_GLOBAL | cola: r_1m NEG | 500 | 3 | 30 | 0.6 % [0.2 %, 1.7 %] | 3.7 % [2.7 %, 5.1 %] | sí |
| PERMUTACION_GLOBAL | cola: r_1m POS | 500 | 5 | 30 | 1.0 % [0.4 %, 2.3 %] | 5.9 % [4.6 %, 7.5 %] | sí |
| PERMUTACION_GLOBAL | cola: r_1m BOTH | 500 | 0 | — | 0.0 % [0.0 %, 0.8 %] | 4.1 % [3.0 %, 5.5 %] | — |
| PERMUTACION_GLOBAL | cola: z_ewma_60_raw NEG | 500 | 22 | 30 | 4.4 % [2.9 %, 6.6 %] | 6.2 % [4.9 %, 7.9 %] | sí |
| PERMUTACION_GLOBAL | cola: z_ewma_60_raw POS | 500 | 18 | 30 | 3.6 % [2.3 %, 5.6 %] | 7.9 % [6.4 %, 9.7 %] | sí |
| PERMUTACION_GLOBAL | cola: z_ewma_60_raw BOTH | 500 | 10 | — | 2.0 % [1.1 %, 3.6 %] | 9.3 % [7.7 %, 11.3 %] | — |
| PERMUTACION_GLOBAL | regla global: afirmacion 1 (r en ambas colas) | 500 | 0 | — | 0.0 % [0.0 %, 0.8 %] | — | — |
| PERMUTACION_GLOBAL | regla global: afirmacion 2 (r y z en ambas colas, z > 1 en todos los años completos) | 500 | 0 | — | 0.0 % [0.0 %, 0.8 %] | — | — |
| PERMUTACION_GLOBAL | z en ambas colas y z > 1 en todos los años completos (sin exigir r) | 500 | 0 | — | 0.0 % [0.0 %, 0.8 %] | — | — |
| SIMULADO_PERFIL_RELOJ | cola: r_1m NEG | 500 | 11 | 30 | 2.2 % [1.2 %, 3.9 %] | 10.3 % [8.6 %, 12.3 %] | sí |
| SIMULADO_PERFIL_RELOJ | cola: r_1m POS | 500 | 4 | 30 | 0.8 % [0.3 %, 2.0 %] | 22.4 % [19.9 %, 25.1 %] | sí |
| SIMULADO_PERFIL_RELOJ | cola: r_1m BOTH | 500 | 0 | — | 0.0 % [0.0 %, 0.8 %] | 32.1 % [29.3 %, 35.1 %] | — |
| SIMULADO_PERFIL_RELOJ | cola: z_ewma_60_raw NEG | 500 | 5 | 30 | 1.0 % [0.4 %, 2.3 %] | 2.7 % [1.9 %, 3.9 %] | sí |
| SIMULADO_PERFIL_RELOJ | cola: z_ewma_60_raw POS | 500 | 6 | 30 | 1.2 % [0.6 %, 2.6 %] | 8.6 % [7.0 %, 10.5 %] | sí |
| SIMULADO_PERFIL_RELOJ | cola: z_ewma_60_raw BOTH | 500 | 5 | — | 1.0 % [0.4 %, 2.3 %] | 6.3 % [5.0 %, 8.0 %] | — |
| SIMULADO_PERFIL_RELOJ | regla global: afirmacion 1 (r en ambas colas) | 500 | 0 | — | 0.0 % [0.0 %, 0.8 %] | — | — |
| SIMULADO_PERFIL_RELOJ | regla global: afirmacion 2 (r y z en ambas colas, z > 1 en todos los años completos) | 500 | 0 | — | 0.0 % [0.0 %, 0.8 %] | — | — |
| SIMULADO_PERFIL_RELOJ | z en ambas colas y z > 1 en todos los años completos (sin exigir r) | 500 | 0 | — | 0.0 % [0.0 %, 0.8 %] | — | — |
| PERMUTACION_MINUTO_X_ANIO | cola: r_1m NEG | 500 | 11 | 30 | 2.2 % [1.2 %, 3.9 %] | 9.2 % [7.6 %, 11.2 %] | sí |
| PERMUTACION_MINUTO_X_ANIO | cola: r_1m POS | 500 | 14 | 30 | 2.8 % [1.7 %, 4.6 %] | 42.6 % [39.6 %, 45.7 %] | sí |
| PERMUTACION_MINUTO_X_ANIO | cola: r_1m BOTH | 500 | 15 | — | 3.0 % [1.8 %, 4.9 %] | 91.1 % [89.2 %, 92.7 %] | — |
| PERMUTACION_MINUTO_X_ANIO | cola: z_ewma_60_raw NEG | 500 | 0 | 30 | 0.0 % [0.0 %, 0.8 %] | 0.0 % [0.0 %, 0.4 %] | sí |
| PERMUTACION_MINUTO_X_ANIO | cola: z_ewma_60_raw POS | 500 | 0 | 30 | 0.0 % [0.0 %, 0.8 %] | 0.1 % [0.0 %, 0.6 %] | sí |
| PERMUTACION_MINUTO_X_ANIO | cola: z_ewma_60_raw BOTH | 500 | 0 | — | 0.0 % [0.0 %, 0.8 %] | 0.0 % [0.0 %, 0.4 %] | — |
| PERMUTACION_MINUTO_X_ANIO | regla global: afirmacion 1 (r en ambas colas) | 500 | 0 | — | 0.0 % [0.0 %, 0.8 %] | — | — |
| PERMUTACION_MINUTO_X_ANIO | regla global: afirmacion 2 (r y z en ambas colas, z > 1 en todos los años completos) | 500 | 0 | — | 0.0 % [0.0 %, 0.8 %] | — | — |
| PERMUTACION_MINUTO_X_ANIO | z en ambas colas y z > 1 en todos los años completos (sin exigir r) | 500 | 0 | — | 0.0 % [0.0 %, 0.8 %] | — | — |

**G2: COMPATIBLE CON LOS CONTROLES EVALUADOS (no rechazada).** La tasa más alta entre las que entran en el criterio es 4.4 % [2.9 %, 6.6 %] (`PERMUTACION_GLOBAL`, cola: z_ewma_60_raw NEG). "Compatible" significa que el criterio **no rechaza** la calibración en las familias evaluadas; no certifica una tasa del 2.5 % ni cubre mundos sin dependencia distintos de esas familias (§16). La lectura correcta son las tasas de la tabla con sus intervalos: en la peor cola el intervalo llega hasta 6.6 %. La regla aplicada a los datos reales exige además que el IC quede por encima de 1, lo que sólo puede reducir estas tasas. "Ambas colas juntas" y la fila "sin exigir r" se muestran pero no forman parte del criterio.

**Cuánto distingue el criterio.** El máximo admitido (30 de 500) es una cota conservadora exacta: se calcula con el estadístico de orden 487 de las 500 réplicas que fijan la banda (el percentil 97,5 se interpola entre las posiciones 487 y 488, y nunca queda por debajo de la 487). Esa cota protege de rechazar un procedimiento correcto; a cambio distingue poco. Si la tasa verdadera de una cola fuese la indicada (con la banda fija), esto es lo que ocurriría:

| tasa verdadera de falsa detección (por cola) | probabilidad de NO superar 30/500 (pasa) | probabilidad de que el criterio lo rechace |
|---:|---:|---:|
| 2.5 % | 100.0 % | 0.0 % |
| 4.0 % | 98.8 % | 1.2 % |
| 5.0 % | 86.9 % | 13.1 % |
| 6.0 % | 54.8 % | 45.2 % |
| 7.5 % | 11.5 % | 88.5 % |
| 10.0 % | 0.1 % | 99.9 % |

Es decir: una tasa del doble del nominal pasaría la mayor parte de las veces. Pasar el criterio **no demuestra** que las tasas sean del 2,5 %; sólo que no hay evidencia para rechazarlo en las familias evaluadas, con esta potencia.

**Controles de "sólo volatilidad"** (200 réplicas de cada uno). Aquí los retornos se generan como *volatilidad × ruido independiente*, con una senda de volatilidad tomada de los datos. Por construcción, todo agrupamiento que aparezca es volatilidad:

| control | regla | réplicas | detecciones | tasa [IC95 binomial] |
|---:|---:|---:|---:|---:|
| VOLATILIDAD_PURA_EWMA_60 | cola: r_1m NEG | 200 | 200 | 100.0 % [98.1 %, 100.0 %] |
| VOLATILIDAD_PURA_EWMA_60 | cola: r_1m POS | 200 | 200 | 100.0 % [98.1 %, 100.0 %] |
| VOLATILIDAD_PURA_EWMA_60 | cola: r_1m BOTH | 200 | 200 | 100.0 % [98.1 %, 100.0 %] |
| VOLATILIDAD_PURA_EWMA_60 | cola: z_ewma_60_raw NEG | 200 | 200 | 100.0 % [98.1 %, 100.0 %] |
| VOLATILIDAD_PURA_EWMA_60 | cola: z_ewma_60_raw POS | 200 | 200 | 100.0 % [98.1 %, 100.0 %] |
| VOLATILIDAD_PURA_EWMA_60 | cola: z_ewma_60_raw BOTH | 200 | 200 | 100.0 % [98.1 %, 100.0 %] |
| VOLATILIDAD_PURA_EWMA_60 | regla global: afirmacion 1 (r en ambas colas) | 200 | 200 | 100.0 % [98.1 %, 100.0 %] |
| VOLATILIDAD_PURA_EWMA_60 | regla global: afirmacion 2 (r y z en ambas colas, z > 1 en todos los años completos) | 200 | 195 | 97.5 % [94.3 %, 98.9 %] |
| VOLATILIDAD_PURA_EWMA_60 | z en ambas colas y z > 1 en todos los años completos (sin exigir r) | 200 | 195 | 97.5 % [94.3 %, 98.9 %] |
| VOLATILIDAD_PURA_EWMA_20 | cola: r_1m NEG | 200 | 200 | 100.0 % [98.1 %, 100.0 %] |
| VOLATILIDAD_PURA_EWMA_20 | cola: r_1m POS | 200 | 200 | 100.0 % [98.1 %, 100.0 %] |
| VOLATILIDAD_PURA_EWMA_20 | cola: r_1m BOTH | 200 | 200 | 100.0 % [98.1 %, 100.0 %] |
| VOLATILIDAD_PURA_EWMA_20 | cola: z_ewma_60_raw NEG | 200 | 200 | 100.0 % [98.1 %, 100.0 %] |
| VOLATILIDAD_PURA_EWMA_20 | cola: z_ewma_60_raw POS | 200 | 200 | 100.0 % [98.1 %, 100.0 %] |
| VOLATILIDAD_PURA_EWMA_20 | cola: z_ewma_60_raw BOTH | 200 | 200 | 100.0 % [98.1 %, 100.0 %] |
| VOLATILIDAD_PURA_EWMA_20 | regla global: afirmacion 1 (r en ambas colas) | 200 | 200 | 100.0 % [98.1 %, 100.0 %] |
| VOLATILIDAD_PURA_EWMA_20 | regla global: afirmacion 2 (r y z en ambas colas, z > 1 en todos los años completos) | 200 | 200 | 100.0 % [98.1 %, 100.0 %] |
| VOLATILIDAD_PURA_EWMA_20 | z en ambas colas y z > 1 en todos los años completos (sin exigir r) | 200 | 200 | 100.0 % [98.1 %, 100.0 %] |
| VOLATILIDAD_PURA_LOCAL_10 | cola: r_1m NEG | 200 | 200 | 100.0 % [98.1 %, 100.0 %] |
| VOLATILIDAD_PURA_LOCAL_10 | cola: r_1m POS | 200 | 200 | 100.0 % [98.1 %, 100.0 %] |
| VOLATILIDAD_PURA_LOCAL_10 | cola: r_1m BOTH | 200 | 200 | 100.0 % [98.1 %, 100.0 %] |
| VOLATILIDAD_PURA_LOCAL_10 | cola: z_ewma_60_raw NEG | 200 | 200 | 100.0 % [98.1 %, 100.0 %] |
| VOLATILIDAD_PURA_LOCAL_10 | cola: z_ewma_60_raw POS | 200 | 200 | 100.0 % [98.1 %, 100.0 %] |
| VOLATILIDAD_PURA_LOCAL_10 | cola: z_ewma_60_raw BOTH | 200 | 200 | 100.0 % [98.1 %, 100.0 %] |
| VOLATILIDAD_PURA_LOCAL_10 | regla global: afirmacion 1 (r en ambas colas) | 200 | 200 | 100.0 % [98.1 %, 100.0 %] |
| VOLATILIDAD_PURA_LOCAL_10 | regla global: afirmacion 2 (r y z en ambas colas, z > 1 en todos los años completos) | 200 | 200 | 100.0 % [98.1 %, 100.0 %] |
| VOLATILIDAD_PURA_LOCAL_10 | z en ambas colas y z > 1 en todos los años completos (sin exigir r) | 200 | 200 | 100.0 % [98.1 %, 100.0 %] |

- La afirmación 1 se detecta en estos mundos con frecuencia `VOLATILIDAD_PURA_EWMA_60`: 100 %; `VOLATILIDAD_PURA_EWMA_20`: 100 %; `VOLATILIDAD_PURA_LOCAL_10`: 100 %: es lo esperado (ahí sí hay agrupamiento) y sirve de control positivo.
- Lo importante es la fila de la afirmación 2: **con qué frecuencia se diría "persiste tras estandarizar" en un mundo donde, por construcción, sólo hay volatilidad.** Si esa tasa es alta, "persiste" no distingue nada.

Detalle por réplica en `TDA13_calibracion_g2.csv`; tasas y bandas en `TDA13_tasas_falsa_deteccion.csv`.

## 13. Veredictos

### TH06 — triage de extremos: `MAYORIA_CLASIFICABLE` (clasificación operativa)

De 38,961 extremos: 85.2 % `EVENTO_REAL`, 13.2 % `INDETERMINADO`, 1.56 % `ARTEFACTO_FRONTERA`, 0.054 % `ARTEFACTO_ROLL`, 0.000 % `ERROR_DATO`.

**Qué respalda cada categoría (hallazgo D):**

| Categoría | En qué se apoya |
|---|---|
| `ERROR_DATO` | DEMOSTRADO (invariante fisico de TDA-00) |
| `ARTEFACTO_ROLL` | ESTRUCTURAL (el retorno cruza un cambio de contrato) |
| `ARTEFACTO_FRONTERA` | ESTRUCTURAL (el retorno cruza una jornada o un hueco) |
| `EVENTO_REAL` | OPERATIVA_NO_VERIFICADA (sin indicios de artefacto y con evidencia favorable; no confirmado con una fuente independiente) |
| `INDETERMINADO` | SIN_EVIDENCIA_SUFICIENTE |

- **`EVENTO_REAL` no es un evento verificado.** Significa: retorno válido, barras activas, registro continuo alrededor, precio que no retrocede más de la mitad en el minuto siguiente y volumen no bajo. Eso es evidencia favorable, no una comprobación independiente. Un precio erróneo que se mantuviera en la barra siguiente, o un desplazamiento de nivel causado por la fuente de datos, pasaría estas comprobaciones. Con barras de 1 minuto y sin una segunda fuente no hay forma de descartarlo caso por caso. La columna `category_basis` del inventario lo deja escrito en cada fila.
- **`ERROR_DATO` = 0 no significa "no hay errores"**: significa que ninguna barra extrema viola los invariantes físicos que TDA-00 comprueba.
- **"Mayoría clasificable"** quiere decir que las reglas asignan una categoría distinta de `INDETERMINADO` a más de la mitad (lectura literal del backlog; el 50 % es esa lectura, no un umbral ajustado). No quiere decir que la mayoría esté verificada.
- La fracción de artefactos y errores es 1.61 %, y todos están fuera de la población de análisis desde TDA-04.

**Sensibilidad de la lectura a las reglas de triage** (descriptivo; la clasificación vigente no cambia):

| tipo | variante | % INDETERMINADO | % EVENTO_REAL |
|---:|---:|---:|---:|
| VIGENTE | Reglas vigentes | 13.2 % | 85.2 % |
| VARIANTE_DE_REGLA | R4 sin la duda BARRA_NO_ACTIVA | 13.1 % | 85.3 % |
| VARIANTE_DE_REGLA | R4 sin la duda JUNTO_A_HUECO_DESCONOCIDO | 13.0 % | 85.4 % |
| VARIANTE_DE_REGLA | R4 sin la duda SIN_BARRA_SIGUIENTE | 13.0 % | 85.4 % |
| VARIANTE_DE_REGLA | R4 sin la duda REVIERTE_BARRA_SIGUIENTE | 1.2 % | 97.2 % |
| VARIANTE_DE_REGLA | R4 sin la duda VOLUMEN_BAJO | 12.6 % | 85.8 % |
| VARIANTE_DE_REGLA | Duda de reversion con fraccion 0.25 en vez de 0.5 | 29.3 % | 69.1 % |
| VARIANTE_DE_REGLA | Duda de reversion con fraccion 0.75 en vez de 0.5 | 6.1 % | 92.3 % |
| VARIANTE_DE_REGLA | Duda de reversion con fraccion 1 en vez de 0.5 | 3.2 % | 95.1 % |
| COTA | Cota inferior: ninguna duda de R4 (todo retorno valido seria EVENTO_REAL) | 0.0 % | 98.4 % |
| COTA | Cota superior: la evidencia favorable de R5 no se acepta como prueba (todo retorno valido es INDETERMINADO) | 98.4 % | 0.0 % |

Entre las variantes de regla, `INDETERMINADO` va del 1.2 % al 29.3 %: la lectura `MAYORIA_CLASIFICABLE` se mantiene en todas. Las dos cotas muestran de qué depende en el fondo: si no se acepta la evidencia favorable de R5 como suficiente, **todo** retorno válido queda `INDETERMINADO`. TH06 es, por tanto, una conclusión condicionada a aceptar esa evidencia. La caracterización de la cola debe darse **como rango** (con y sin `INDETERMINADO`, §11).

### TH27 — agrupamiento de extremos: `AGRUPAMIENTO PERSISTE CON EL EWMA; ORIGEN NO IDENTIFICADO`

TH27 se separa en tres afirmaciones, porque la segunda no demuestra la tercera.

| Afirmación | Estado | Tipo de evidencia |
|---|---|---|
| 1. Los extremos se agrupan más de lo esperable si fueran independientes | `DETECTADO` | Banda del procedimiento completo; G2 compatible con los controles evaluados (no rechazada) |
| 2. El agrupamiento persiste después de dividir por la volatilidad estimada (EWMA, semivida 60 min) | `DETECTADO` | Calibrada frente a la independencia; **no** frente a "sólo volatilidad" |
| 3. Existe un agrupamiento propio de los extremos, no explicable por cambios de volatilidad | `NO IDENTIFICADO POR ESTE PROCEDIMIENTO` | Sólo evidencia descriptiva; este método no puede afirmarlo ni descartarlo |

- **caídas (cola negativa)**. En crudo: 32.18 % de los tiempos entre extremos es de 10 barras o menos, frente al 17.27 % esperado; cociente 1.86 (IC 95 % con re-estimación [1.79, 1.91]; banda calibrada hasta 1.048) → DETECTADO. Estandarizado: 28.57 % de los tiempos entre extremos es de 10 barras o menos, frente al 16.48 % esperado; cociente 1.73 (IC 95 % con re-estimación [1.68, 1.79]; banda calibrada hasta 1.043) → DETECTADO. El cociente de `z` supera 1 en 5 de 5 años completos (mínimo 1.67). Parte del exceso que desaparece al estandarizar: 15 % [8 %, 19 %].
- **subidas (cola positiva)**. En crudo: 29.49 % de los tiempos entre extremos es de 10 barras o menos, frente al 18.87 % esperado; cociente 1.56 (IC 95 % con re-estimación [1.51, 1.60]; banda calibrada hasta 1.057) → DETECTADO. Estandarizado: 25.44 % de los tiempos entre extremos es de 10 barras o menos, frente al 18.05 % esperado; cociente 1.41 (IC 95 % con re-estimación [1.37, 1.44]; banda calibrada hasta 1.044) → DETECTADO. El cociente de `z` supera 1 en 5 de 5 años completos (mínimo 1.31). Parte del exceso que desaparece al estandarizar: 27 % [21 %, 33 %].


**Por qué la afirmación 2 no demuestra la 3.** "Estandarizar" es dividir por una volatilidad *estimada* con una media móvil que reacciona con retraso. Si la volatilidad sube de golpe, durante unos minutos la estimación se queda corta, `z` sale grande varias barras seguidas y aparece agrupamiento aunque los retornos sean independientes dada la volatilidad verdadera. Esto se comprueba de dos maneras:

- *Sensibilidad a la semivida* (§7): cuanto más rápido reacciona el estimador, menos agrupamiento queda.
- *Controles de sólo volatilidad* (§12): en mundos donde, por construcción, sólo hay volatilidad, se compara el cociente de `z` que resulta con el observado:

| control | lado | réplicas | cociente real de z | cociente de z en el control: media [p2,5; p97,5] | parte del exceso real que el control reproduce | ¿el real cabe en el control? | tasa de 'persiste' en el control |
|---:|---:|---:|---:|---:|---:|---:|---:|
| VOLATILIDAD_PURA_EWMA_60 | NEG | 200 | 1.734 | 1.117 [1.086; 1.147] | 16 % | no | 98 % |
| VOLATILIDAD_PURA_EWMA_60 | POS | 200 | 1.409 | 1.153 [1.129; 1.181] | 37 % | no | 98 % |
| VOLATILIDAD_PURA_EWMA_20 | NEG | 200 | 1.734 | 1.322 [1.292; 1.353] | 44 % | no | 100 % |
| VOLATILIDAD_PURA_EWMA_20 | POS | 200 | 1.409 | 1.363 [1.331; 1.395] | 89 % | no | 100 % |
| VOLATILIDAD_PURA_LOCAL_10 | NEG | 200 | 1.734 | 1.597 [1.564; 1.632] | 81 % | no | 100 % |
| VOLATILIDAD_PURA_LOCAL_10 | POS | 200 | 1.409 | 1.592 [1.560; 1.623] | 145 % | sí | 100 % |

Ninguno de los controles de solo volatilidad considerados reproduce por completo el cociente real de `z`. Eso **no** demuestra un agrupamiento propio: sólo dice que estas sendas concretas no bastan. Podría ser volatilidad que cambia de otra forma o más deprisa, o podría ser un fenómeno propio; este análisis no decide entre ambas.

**Alcance exacto de esta conclusión.** Lo que queda establecido es una limitación del procedimiento, no una propiedad del mercado:

- *Lo que sí se sabe:* dividir por una volatilidad estimada con retraso deja agrupamiento residual aunque no exista ningún fenómeno propio de los extremos. Por tanto, que el agrupamiento persista en `z` no basta para identificar ese fenómeno.
- *Lo que no se sabe:* si ese fenómeno existe. Este procedimiento no lo identifica; tampoco lo descarta. **Ausencia de prueba no es prueba de ausencia**, en ninguna de las dos direcciones.
- *Lo que NO se afirma:* que los dos fenómenos sean imposibles de distinguir. Aquí no se distinguen porque la única medida de volatilidad disponible se construye con los mismos retornos de 1 minuto cuyos extremos se cuentan, y la volatilidad verdadera no es observable con OHLCV (G0). Con una medida de volatilidad independiente del retorno de la barra (por ejemplo, calculada con datos de mayor frecuencia), con otro diseño de contraste o con un modelo explícito, la separación podría ser posible. Eso queda fuera de esta etapa y no se ha intentado.

**En palabras sencillas.** Los movimientos extremos llegan en grupos, y siguen haciéndolo después de dividir por la volatilidad reciente (media móvil de 60 minutos). Lo que este análisis NO establece es si ese resto es algo distinto de la volatilidad o simplemente volatilidad que el estimador no alcanza a seguir. No lo afirma ni lo descarta: el método usado aquí no sirve para distinguirlo.

**Qué NO afirma este veredicto.**
- No afirma un fenómeno adicional al agrupamiento de volatilidad de TDA-09, ni lo descarta.
- No se ha estimado el índice extremal ni ningún modelo de cola.
- Es exploratorio y retrospectivo: no está confirmado en el holdout.

## 14. Auditoría correctiva: qué cambió y cuánto

Veredictos previos: TH06 = `MAYORIA_CLASIFICABLE`; TH27 = `AGRUPAMIENTO_PERSISTE_TRAS_ESTANDARIZAR`; G2 = `calibracion declarada satisfactoria con el criterio 'media dentro de la banda' (criterio retirado en la auditoria)`.

| Hallazgo | Causa | Corrección |
|---|---|---|
| A. Referencia `RELOJ_Y_ANIO` | La fórmula `p(minuto) × tasa(año)/tasa(global)` no conserva los totales por minuto y por año a la vez | Ajuste proporcional iterativo que conserva ambos (§6). La fórmula original queda como referencia secundaria de comparación |
| B. Calibración G2 | Criterio "la media cae dentro" + banda que ignora la estimación de umbrales, estados y tasas | Banda obtenida del procedimiento completo sobre series nulas; tasas de falsa detección con intervalo binomial en réplicas reservadas; criterio de aceptación escrito (§12) |
| C. Intervalos | El IC remuestreaba conteos con umbrales y referencia fijos | Bootstrap que re-estima umbrales y referencia para la variante primaria; el resto se rotula "condicional"; se declara lo que ninguno cubre (§6) |
| D. TH06 | `EVENTO_REAL` se leía como verificación | Las categorías no cambian; se explicita que es una clasificación operativa y se añade la sensibilidad a las reglas (§13) |
| H3 (2.ª auditoría). Familias de control | Las dos familias mezclaban los años: no cubrían un mundo sin dependencia con perfil horario distinto cada año | Tercera familia `PERMUTACION_MINUTO_X_ANIO`; banda = la mayor de las tres; criterio de G2 sobre 12 tasas (§12) |
| H1–H2 (2.ª auditoría). Criterio de G2 | El máximo admitido se llamaba "exacto" sin serlo, y "G2 superada" sugería una tasa demostrada | Cota conservadora exacta (estadístico de orden inferior); rótulo "compatible con los controles evaluados" y tabla de potencia (§12) |
| H5 (2.ª auditoría). Bloques | Filtrar los bloques con la población común podría unir tramos separados por una barra excluida | `restrict_block_ids` (sin efecto en los datos reales: las exclusiones son sólo el tramo inicial) |
| TH27 | Un único veredicto mezclaba "persiste con el EWMA" con "fenómeno propio" | Tres afirmaciones separadas; controles de sólo volatilidad (§13) |

**Antes y después, variante primaria:**

| serie | lado | antes: cociente [IC condicional] | antes: banda sup. | después: cociente [IC con re-estimación] | después: banda calibrada sup. | cambio del cociente |
|---:|---:|---:|---:|---:|---:|---:|
| r_1m | NEG | 1.858 [1.796, 1.923] | 1.036 | 1.863 [1.791, 1.908] | 1.048 | 0.0048 |
| r_1m | POS | 1.559 [1.494, 1.629] | 1.030 | 1.563 [1.508, 1.604] | 1.057 | 0.0042 |
| r_1m | BOTH | 1.485 [1.441, 1.531] | 1.016 | 1.488 [1.458, 1.514] | 1.037 | 0.0033 |
| z_ewma_60_raw | NEG | 1.734 [1.692, 1.778] | 1.037 | 1.734 [1.684, 1.789] | 1.043 | 0.0004 |
| z_ewma_60_raw | POS | 1.409 [1.377, 1.445] | 1.035 | 1.409 [1.369, 1.438] | 1.044 | 0.0003 |
| z_ewma_60_raw | BOTH | 1.362 [1.342, 1.385] | 1.017 | 1.363 [1.342, 1.379] | 1.023 | 0.0002 |
| z_ewma_20_raw | NEG | 1.409 [1.367, 1.454] | 1.043 | — | NaN | NaN |
| z_ewma_20_raw | POS | 1.184 [1.145, 1.225] | 1.038 | — | NaN | NaN |
| z_ewma_20_raw | BOTH | 1.106 [1.086, 1.129] | 1.018 | — | NaN | NaN |
| z_ewma_240_raw | NEG | 2.062 [2.010, 2.116] | 1.037 | — | NaN | NaN |
| z_ewma_240_raw | POS | 1.691 [1.651, 1.736] | 1.032 | — | NaN | NaN |
| z_ewma_240_raw | BOTH | 1.604 [1.578, 1.632] | 1.016 | — | NaN | NaN |

**Efecto de corregir sólo la referencia** (misma muestra, mismos umbrales; IC condicional):

| serie | referencia | lado | observado | esperado si independientes | cociente [IC95 condicional] |
|---:|---:|---:|---:|---:|---:|
| r_1m | RELOJ_Y_ANIO | NEG | 32.18 % | 17.27 % | 1.86 [1.80, 1.93] |
| r_1m | RELOJ_X_FACTOR_ANIO_PREAUDITORIA | NEG | 32.18 % | 17.31 % | 1.86 [1.80, 1.92] |
| r_1m | RELOJ_Y_ANIO | POS | 29.49 % | 18.87 % | 1.56 [1.50, 1.63] |
| r_1m | RELOJ_X_FACTOR_ANIO_PREAUDITORIA | POS | 29.49 % | 18.92 % | 1.56 [1.49, 1.63] |
| z_ewma_60_raw | RELOJ_Y_ANIO | NEG | 28.57 % | 16.48 % | 1.73 [1.69, 1.78] |
| z_ewma_60_raw | RELOJ_X_FACTOR_ANIO_PREAUDITORIA | NEG | 28.57 % | 16.48 % | 1.73 [1.69, 1.78] |
| z_ewma_60_raw | RELOJ_Y_ANIO | POS | 25.44 % | 18.05 % | 1.41 [1.38, 1.44] |
| z_ewma_60_raw | RELOJ_X_FACTOR_ANIO_PREAUDITORIA | POS | 25.44 % | 18.06 % | 1.41 [1.38, 1.44] |

Se conservaron sin cambios, como base de comparación: nivel 1 %, definición `SEGMENT_X_VOL`, ventana de 10 barras, series `r` y `z`, semivida 60 min y sus sensibilidades, y las reglas de triage R1–R5. La foto de los resultados previos (con los hash de los artefactos anteriores) está en `TDA13_auditoria_resultados_previos.csv`.

## 15. Consecuencias para el resto del roadmap

- **TDA-13 Parte B (EVT) no se ejecuta.** Su criterio de entrada 1 exige que TDA-12 hubiera concluido `ESCALA Y FORMA`; TDA-12 cerró `NO CONCLUYENTE`. Nada de esta etapa cambia ni reinterpreta ese resultado. `STOP-13` sigue no activado; la rama EVT sigue no autorizada.
- **Tamaño de muestra efectivo de la cola.** Las excedencias no son observaciones independientes: 20.1 % de los extremos de r_1m forma parte de una racha de dos o más barras, 18.0 % de los extremos de z_ewma_60_raw forma parte de una racha de dos o más barras. Esto es cierto sea cual sea su origen (volatilidad o no) y afecta a la incertidumbre de cualquier cuantil extremo. Los IC por jornadas de etapas anteriores ya incorporan la dependencia dentro del día, no la que cruza de un día a otro.
- **TDA-14** (estabilidad temporal) es la siguiente etapa; aquí no se anticipa.
- **Decisiones de IRIS:** ninguna.

## 16. Limitaciones

1. Todo es **exploratorio y retrospectivo**: umbrales, referencias y cortes de segmento usan toda la muestra, y nada se ha confirmado en el holdout (G2).
2. La definición de extremo, las reglas de triage, la referencia `RELOJ_Y_ANIO`, la ventana de 10 barras, las familias de control, el criterio de aceptación de G2 y la regla de veredicto son **decisiones de esta etapa** (las tres últimas, introducidas en la auditoría). Ninguna estaba aprobada previamente. Los resultados se leen condicionados a ellas.
3. La afirmación 3 de TH27 **no queda identificada por este procedimiento** (§13): no se afirma ni se descarta, y no se sostiene que sea imposible de distinguir con otros datos o métodos. Los controles de sólo volatilidad usan tres sendas elegidas en la auditoría; ninguna es "la volatilidad verdadera", y la más rápida es retrospectiva y opera a la misma escala que la ventana de medida.
4. La banda calibrada depende de qué mundos sin dependencia se consideran. Se usan tres familias de permutación (global, por minuto, por minuto y año). La garantía de §12 vale para mundos intercambiables dentro de (minuto, año); un mundo sin dependencia con otra heterogeneidad (por mes, por día de la semana, por tipo de jornada) no está cubierto y podría producir falsas detecciones. La familia por minuto y año destruye además el orden de las jornadas dentro del año, de modo que cualquier persistencia entre días —incluida la de la volatilidad— cuenta como "dependencia": la afirmación 1 no distingue entre ellas. La distancia entre los cocientes reales y la banda indica cuánto margen hay.
5. La calibración mide falsas detecciones con 500 réplicas reservadas por familia: detecta desviaciones apreciables del nominal, no de décimas de punto.
6. El IC con re-estimación no cubre la estimación de `sigma_hat`, `z` y el decil, ni los cortes de segmento, ni la dependencia entre jornadas (§6). Los demás IC son condicionales.
7. `EVENTO_REAL` es una clasificación operativa, no una verificación. Sin datos tick a tick ni una segunda fuente no se puede confirmar el origen de un extremo (G0). `ERROR_DATO` sólo detecta lo que TDA-00 detecta.
8. No hay calendario fechado de eventos: ninguna atribución causal. La marca de las 08:30 es una coincidencia de reloj.
9. La referencia pone probabilidad 0 en los minutos sin ningún extremo observado, y estima sus tasas con los mismos datos.
10. El agrupamiento sólo se mide dentro de bloques continuos: no dice nada sobre si una jornada con extremos va seguida de otra, salvo por el índice de dispersión diaria.
11. En las celdas pequeñas y en el nivel 0.1 % hay pocas excedencias; esos resultados son orientativos.
12. Reutilizar la firma de "tick aislado" de TDA-03 y los niveles de cuantil de TDA-07 no significa que su uso aquí estuviera aprobado: los valores son heredados, la aplicación es nueva.

## 17. Tiempos

| Etapa | Tiempo |
|---|---:|
| Universo de triage: serie canonica + mascaras de TDA-00/02/03/04/05 | 35604.0s (incluye ≈ 35601 s con el sistema parado; efectivo ≈ 3 s) |
| sigma_hat causal, z_t y decil causal (TDA-10/12) + verificacion de ausencia de look-ahead | 85.1s |
| Umbrales relativos por celda e inventario de extremos | 14.4s |
| Agrupamiento observado vs referencia independiente (r y z; definiciones, niveles y lados) | 216.6s |
| Agrupamiento por año y por segmento (G3) | 15.6s |
| Distribucion horaria de los extremos | 0.1s |
| Efecto del triage sobre curtosis y cuantiles extremos (con y sin cada categoria) | 330.0s |
| Calibracion G2 (procedimiento completo sobre series nulas), controles de solo volatilidad y bootstrap con re-estimacion | 4757.5s |

**Tiempo transcurrido (análisis + escritura): 41026.1s (~683.8 min), de los cuales ≈ 35601 s (~593.4 min) con el sistema parado. Tiempo efectivo aproximado: ≈ 5425 s (~90.4 min).** CPU, 6 proceso(s) en la etapa de calibración (limitados por la memoria libre al arrancar); sin GPU. `n_boot = 300` (condicional), 300 (con re-estimación); semillas fijas; IC percentil al 95 %. El resultado no depende del número de procesos.

**Incidencias de ejecución** (declaradas en `configs/tda13_cierre.yaml`; horas locales). El tiempo transcurrido NO es tiempo de cálculo: incluye los tramos en que el sistema estuvo parado. El tiempo efectivo es una resta aproximada (precisión de segundos de los registros del sistema), no una medida directa. Las secciones 17 y 20 se regeneraron con `python -m ohlcv_dataroad.ingest.run_tda13 --amend-report --closure configs/tda13_cierre.yaml`, sin repetir ningún cálculo: el resto de este informe y los demás artefactos son, sin cambios, los de la ejecución original.

| Incidencia | Inicio | Fin | Duración | Etapa afectada | Evidencia | Efecto sobre los resultados |
|---|---|---|---:|---|---|---|
| Suspensión del sistema operativo (motivo registrado: botón o tapa) | 2026-10-13 00:05:34 | 2026-10-13 09:58:55 | ≈ 35601 s (~593.4 min) | Universo de triage: serie canonica + mascaras de TDA-00/02/03/04/05 | registro System de Windows (Kernel-Power 42 a las 00:05:34; Power-Troubleshooter 1 a las 09:58:55) y lecturas del supervisor, sin ninguna entre las 00:05:35 y las 09:58:54 | ninguno: las 205.400 cifras de réplicas comunes con la ejecución anterior son idénticas, y 18 de los 21 artefactos son idénticos byte a byte |

## 18. Archivos generados

`TDA13_extremos.csv` (inventario con triage), `TDA13_umbrales_por_celda.csv`, `TDA13_triage_resumen.csv`, `TDA13_triage_sensibilidad.csv`, `TDA13_agrupamiento_vs_poisson.csv` (observado frente a la referencia: todas las series, definiciones, niveles, lados y referencias), `TDA13_tiempos_entre_excedencias.csv`, `TDA13_longitud_rachas.csv`, `TDA13_agrupamiento_por_anio_segmento.csv`, `TDA13_distribucion_horaria.csv`, `TDA13_efecto_triage.csv`, `TDA13_referencia_conservacion.csv`, `TDA13_calibracion_g2.csv` (réplicas), `TDA13_tasas_falsa_deteccion.csv` (tasas y bandas), `TDA13_bootstrap_reestimacion.csv`, `TDA13_control_solo_volatilidad.csv` (15 CSV) + `TDA13_agrupamiento_crudo_vs_estandarizado.png`, `TDA13_tiempos_entre_excedencias.png`, `TDA13_distribucion_horaria.png`, `TDA13_triage_categorias.png` (4 PNG) + este informe. `TDA13_auditoria_resultados_previos.csv` es una foto fija de los resultados previos a la auditoría: se lee, no se regenera.

## 19. Modo sencillo

**¿Qué se hizo?** Se listaron los movimientos de 1 minuto más grandes de MNQ —grandes *para su hora y para lo agitado que estaba el mercado*— y se revisó cada uno con las comprobaciones de calidad de las etapas anteriores.

**¿Son errores?** Ninguna comprobación demuestra errores de datos entre ellos (0 casos), lo que no garantiza que no los haya. Los artefactos conocidos (saltos al cambiar de contrato o al cruzar un cierre) son el 1.61 % y ya estaban fuera de todos los análisis. El 85 % no muestra ningún indicio de artefacto y tiene señales favorables (se clasifica como movimiento real, sin poder verificarlo); el 13 % restante no se puede decidir con barras de 1 minuto.

**¿Llegan en grupos?** Los movimientos extremos llegan en grupos, y siguen haciéndolo después de dividir por la volatilidad reciente (media móvil de 60 minutos). Lo que este análisis NO establece es si ese resto es algo distinto de la volatilidad o simplemente volatilidad que el estimador no alcanza a seguir. No lo afirma ni lo descarta: el método usado aquí no sirve para distinguirlo.

**¿Qué NO significa?** No es una señal ni una regla de operación, no predice nada y no decide nada del diseño de IRIS. Tampoco se ha modelado la cola (EVT): eso sigue sin autorizar.

## 20. Estado

**TDA-13 Parte A = `CERRADA TRAS AUDITORÍA INDEPENDIENTE` (cierre metodológico de la Parte A; resultados exploratorios, no confirmados en el holdout).**

Auditoría metodológica y numérica independiente (13 de octubre de 2026): conforme en el alcance comprobado. Se verificaron las 205.400 cifras de réplicas comunes entre las dos ejecuciones (idénticas), las cuatro bandas y las 12 comprobaciones de G2 (recalculadas desde las réplicas), los cocientes e intervalos que deciden TH27, los controles de sólo volatilidad, el triage y las decisiones individuales de 300 mundos sintéticos. La auditoría no reejecutó el análisis sobre los datos de MNQ ni abrió el holdout. El cierre no avala predicción, agrupamiento propio de los extremos ni EVT; `STOP-13` no activado; TDA-13 Parte B no autorizada. Estado declarado en `configs/tda13_cierre.yaml`.

| Elemento | Estado |
|---|---|
| TH06 | `MAYORIA_CLASIFICABLE` — clasificación operativa; colas reportadas como rango |
| TH27, afirmación 1 (agrupamiento frente a independencia) | `DETECTADO` |
| TH27, afirmación 2 (persiste con el EWMA de 60 min) | `DETECTADO` |
| TH27, afirmación 3 (agrupamiento propio, ajeno a la volatilidad) | `NO IDENTIFICADO POR ESTE PROCEDIMIENTO` (ni afirmado ni descartado) |
| TH27, veredicto | `AGRUPAMIENTO PERSISTE CON EL EWMA; ORIGEN NO IDENTIFICADO` |
| Calibración G2 | COMPATIBLE CON LOS CONTROLES EVALUADOS (no rechazada) |
| TDA-13 Parte B (EVT) / TH28 | no ejecutada; no autorizada |
| `STOP-13` | sin cambios respecto de TDA-12 (no activado) |

Todos los resultados son exploratorios (G2) y dependen de las decisiones listadas en §16.
