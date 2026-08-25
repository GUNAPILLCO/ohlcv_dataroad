# TDA-10 — Escala versus forma: origen de las colas

**Etapa:** `docs/methodology/Tsay_OHLCV_analysis_roadmap.md` § TDA-10
**Depende de:** TDA-06 (`PASS_WITH_OPEN_QUESTIONS`, STOP-6 no activado, `s(m)` construido sobre `log_hl`), TDA-07 (`PASS_WITH_OPEN_QUESTIONS`), TDA-09 (`PASS_WITH_OPEN_QUESTIONS` -- `VOLATILITY_CLUSTERING_DETECTABLE`, TH21=`CLUSTERING_GENUINO`, STOP-9 NO activado).
**Alcance de datos:** exclusivamente `tda04_variables_1m.parquet`, `tda04_return_validity_mask.parquet`, `tda06_r_tilde.parquet` y `TDA06_segmentacion_propuesta.csv`. Ningún archivo de `data/raw/` ni de `holdout_files` fue abierto en esta etapa.
**Generado automáticamente** por `python -m ohlcv_dataroad.ingest.run_tda10 --config configs/mnq_snapshot.yaml` — este informe NUNCA se edita a mano; cualquier corrección se hace en el código (`tda10_scale_vs_shape.py`/`run_tda10.py`) y se regenera con una nueva ejecución.

> Esta etapa estudia si las colas gruesas de MNQ se explican porque la volatilidad cambia en el tiempo (ESCALA, ya caracterizado en TDA-09) o porque, incluso controlando esa volatilidad, la forma de la distribución sigue siendo anormalmente pesada (FORMA). NO ajusta GARCH, NO calcula cuantiles condicionales completos (TDA-12), NO ejecuta EVT, NO crea features/targets/señales, NO abre el holdout, y NO inicia TDA-11/12/13.

---

## 1. La pregunta en palabras simples

¿Los movimientos extremos de MNQ parecen extremos simplemente porque ocurren cuando el mercado ya está muy volátil, o siguen siendo anormalmente extremos incluso comparados con la volatilidad que era esperable en ese momento? Esta etapa estandariza cada retorno por una estimación CAUSAL (solo información pasada) de la volatilidad reciente y mide cuánta curtosis sobrevive.

## 2. Población utilizada

- `r_1m` válido (TDA-04, `r_1m_valid=True`): **1,914,530** filas.
- `r_tilde` (RETROSPECTIVO, TDA-06, `s(m)` sobre `log_hl`): **1,914,530** filas.
- Ambas poblaciones coinciden en `n` (verificado, `verify_populations_aligned`) -- condición para poder comparar "raw" (causal) y "clock_adjusted" (RETROSPECTIVO) bajo la misma noción de población.
- Cada configuración de `sigma_hat` descarta además su propio "quemado" inicial (ventana rodante: `min_periods=window`; EWMA: `3×half-life`) — el `n` exacto por configuración/año está en `TDA10_curtosis_escala_vs_forma.csv` (columna `n`).

## 3. Estimadores causales

Dos familias, cada una con una grilla pequeña y predeclarada (nunca ajustada tras ver el resultado):

- **A. Desviación estándar rodante causal**: ventanas `(30, 120, 390)` (número de barras VÁLIDAS previas, no minutos de reloj). `sigma_hat_(t-1) = std(r[t-W:t])`, con `shift(1)` explícito — la fila `t` nunca usa `r[t]`.
- **B. EWMA causal** (estilo RiskMetrics): half-lives `(20.0, 60.0, 240.0)` minutos. `sigma2_t = lambda·sigma2_(t-1) + (1-lambda)·r_(t-1)^2`, implementado con `r^2` desplazado un paso y `pandas.ewm(halflife=..., adjust=False)` — `sigma2_t` depende exclusivamente de `r[0..t-1]`.
- **Ajuste de reloj**: cada familia se aplica también sobre `r_tilde` (`s(m)` de TDA-06, RETROSPECTIVO) para separar la escala DETERMINISTA (hora del día) de la DINÁMICA (volatilidad reciente) — 2 series de entrada × (3+3) configuraciones = **12 configuraciones** en total.
- **Estimador primario** (declarado antes de ejecutar): `ewma_60_raw` — EWMA, half-life=60min, sobre `r_1m` (la única serie genuinamente causal de principio a fin).

**Verificación explícita de ausencia de look-ahead (G1)** — prueba de reconstrucción: se perturba `r` en varios índices a un valor extremo y se comprueba que `sigma_hat` en posiciones `<= idx` NO cambia (y que sí cambia poco después). Resultado sobre las 6 combinaciones familia/parámetro (idéntico para "raw"/"clock_adjusted", la lógica de causalidad no depende del input):

| family | param | n_indices_checked | passed | all_changed_shortly_after |
|---:|---:|---:|---:|---:|
| rolling_std | 30.0000 | 6 | True | True |
| rolling_std | 120.0000 | 6 | True | True |
| rolling_std | 390.0000 | 6 | True | True |
| ewma | 20.0000 | 6 | True | True |
| ewma | 60.0000 | 6 | True | True |
| ewma | 240.0000 | 6 | True | True |

Las 6 configuraciones **pasaron** la prueba — ninguna usa información de `r[idx]` o posterior para calcular `sigma_hat` en `t<=idx` (si alguna hubiera fallado, `LookaheadLeakageError` habría detenido la etapa antes de construir ningún `z_t`).

## 4. Resultado de curtosis (global, versión completa)

Curtosis de `r` (restringido a la población donde `z` está definido) vs. curtosis de `z`, fracción eliminada, las 12 configuraciones, alcance GLOBAL:

| config | n | kurt_r | kurt_z | fraction_removed |
|---:|---:|---:|---:|---:|
| rolling_std_30_raw | 1,914,454 | 111.560 | 85.514 | 0.233 |
| rolling_std_120_raw | 1,914,410 | 111.549 | 156.331 | -0.401 |
| rolling_std_390_raw | 1,914,140 | 111.533 | 178.663 | -0.602 |
| ewma_20_raw | 1,914,470 | 111.552 | 96.330 | 0.136 |
| ewma_60_raw | 1,914,350 | 111.545 | 154.350 | -0.384 |
| ewma_240_raw | 1,913,810 | 111.517 | 216.245 | -0.939 |
| rolling_std_30_clock_adjusted | 1,914,454 | 78.287 | 15.989 | 0.796 |
| rolling_std_120_clock_adjusted | 1,914,410 | 78.284 | 28.154 | 0.640 |
| rolling_std_390_clock_adjusted | 1,914,140 | 78.273 | 35.833 | 0.542 |
| ewma_20_clock_adjusted | 1,914,470 | 78.286 | 17.673 | 0.774 |
| ewma_60_clock_adjusted | 1,914,350 | 78.281 | 25.429 | 0.675 |
| ewma_240_clock_adjusted | 1,913,810 | 78.261 | 36.660 | 0.532 |

**Bootstrap de bloques por jornada (IC 95%, `n_boot=300`) — configuración primaria `ewma_60_raw`**:

| | punto | IC 95% lo | IC 95% hi |
|---|---:|---:|---:|
| kurt_r | 111.545 | 70.028 | 163.884 |
| kurt_z | 154.350 | 97.541 | 229.547 |
| fraction_removed (versión completa) | -0.3837 | -1.3461 | 0.2356 |

El IC de bootstrap es sobre la versión **completa** (sin recortar) — por eso puede ser negativo/amplio (ver nota más abajo: el momento de cuarto orden sin recortar es frágil ante un puñado de observaciones). El veredicto de §11 usa la versión **recortada** por ese motivo (`fraction_removed_trimmed` = 0.7544 para `ewma_60_raw`).

**Nota sobre `rolling_std_30` y el piso numérico de `sigma_hat`**: la ventana rodante más corta (30 barras) puede, en tramos de precio casi constante, producir un `sigma_hat` numéricamente indistinguible de cero (residual de punto flotante, no volatilidad real) — dividir por ese valor dispararía un `z_t` de millones de desviaciones estándar y contaminaría por completo la curtosis sin recortar de esa única configuración. Se protegió explícitamente (`MIN_VALID_SIGMA_HAT=1e-8`, verificado sobre el conjunto de investigación real: exactamente 1 de 1.914.530 filas por debajo del piso, únicamente en `rolling_std_30`) — sin esa protección, `kurt_z` de esa configuración se dispara a más de 1,9 millones por un solo punto. Es la razón adicional, más allá de la recomendación de TDA-07, por la que el veredicto usa la versión recortada y no la completa.

## 5. Resultado con recorte 0.1% (convención TDA-07)

Misma tabla, columnas `*_trimmed` (recorte del 0.1% total, 0.05% por cola, igual convención que TDA-07):

| config | kurt_r_trimmed | kurt_z_trimmed | fraction_removed_trimmed |
|---:|---:|---:|---:|
| rolling_std_30_raw | 11.488 | 2.310 | 0.799 |
| rolling_std_120_raw | 11.490 | 2.899 | 0.748 |
| rolling_std_390_raw | 11.488 | 4.339 | 0.622 |
| ewma_20_raw | 11.491 | 2.128 | 0.815 |
| ewma_60_raw | 11.490 | 2.822 | 0.754 |
| ewma_240_raw | 11.490 | 4.776 | 0.584 |
| rolling_std_30_clock_adjusted | 11.361 | 1.604 | 0.859 |
| rolling_std_120_clock_adjusted | 11.361 | 1.755 | 0.845 |
| rolling_std_390_clock_adjusted | 11.359 | 2.350 | 0.793 |
| ewma_20_clock_adjusted | 11.361 | 1.419 | 0.875 |
| ewma_60_clock_adjusted | 11.360 | 1.646 | 0.855 |
| ewma_240_clock_adjusted | 11.362 | 2.219 | 0.805 |

La diferencia entre la fracción eliminada completa y la recortada es en sí misma informativa: si la fracción recortada es mucho menor, la reducción de curtosis de la versión completa depende en gran parte de un puñado de observaciones extremas (consistente con TDA-07, que ya mostró que la curtosis recortada de `r_1m` es mucho más estable que la cruda).

## 6. Estabilidad por año (configuración primaria `ewma_60_raw`)

| scope_value | n | kurt_r | kurt_z | fraction_removed | fraction_removed_trimmed |
|---:|---:|---:|---:|---:|---:|
| 2,019 | 7,226 | 32.344 | 10.828 | 0.665 | 0.529 |
| 2,020 | 342,648 | 44.369 | 8.569 | 0.807 | 0.765 |
| 2,021 | 349,039 | 24.023 | 23.679 | 0.014 | 0.646 |
| 2,022 | 349,996 | 130.763 | 354.286 | -1.709 | 0.593 |
| 2,023 | 349,636 | 53.795 | 196.253 | -2.648 | 0.674 |
| 2,024 | 352,261 | 54.183 | 104.280 | -0.925 | 0.691 |
| 2,025 | 163,544 | 138.767 | 237.407 | -0.711 | 0.778 |

## 7. Estabilidad por segmento horario (headline: primario raw y clock_adjusted)

Cuantiles de `z_t` por segmento (TDA-06) — `n` y cuantiles `NaN` si el segmento no alcanza el mínimo de muestra:

| group | n | std_z | q0.01 | q0.05 | q0.95 | q0.99 |
|---:|---:|---:|---:|---:|---:|---:|
| 00:00-02:00 | 168,840 | 0.9943 | -2.5864 | -1.5420 | 1.5467 | 2.5381 |
| 02:00-03:00 | 84,663 | 1.2243 | -3.1495 | -1.9044 | 1.8948 | 3.0550 |
| 03:00-08:30 | 465,442 | 1.0895 | -2.8322 | -1.7119 | 1.6914 | 2.7065 |
| 08:30-09:30 | 84,694 | 1.6582 | -3.1991 | -1.8521 | 1.7789 | 2.9809 |
| 09:30-16:02 | 537,786 | 1.1179 | -2.9965 | -1.7084 | 1.6553 | 2.8242 |
| 16:02-20:00 | 237,735 | 0.6626 | -1.7509 | -0.9506 | 0.9474 | 1.6930 |
| 20:00-24:00 | 335,190 | 0.8467 | -2.2297 | -1.3015 | 1.2997 | 2.1193 |

## 8. Estabilidad entre deciles de volatilidad (headline: primario raw)

| group | n | std_z | q0.01 | q0.05 | q0.95 | q0.99 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0000 | 191,274 | 1.0636 | -2.7662 | -1.6209 | 1.6161 | 2.6701 |
| 1.0000 | 191,453 | 1.0834 | -2.7366 | -1.5976 | 1.5702 | 2.6409 |
| 2.0000 | 191,453 | 1.0834 | -2.7667 | -1.5727 | 1.5765 | 2.6474 |
| 3.0000 | 191,453 | 1.0557 | -2.7683 | -1.5689 | 1.5466 | 2.6352 |
| 4.0000 | 191,453 | 1.0787 | -2.8130 | -1.5678 | 1.5313 | 2.5896 |
| 5.0000 | 191,452 | 1.0348 | -2.7804 | -1.5571 | 1.5274 | 2.5918 |
| 6.0000 | 191,453 | 1.0555 | -2.7280 | -1.5642 | 1.5368 | 2.5874 |
| 7.0000 | 191,453 | 1.0187 | -2.7402 | -1.5659 | 1.5287 | 2.5423 |
| 8.0000 | 191,453 | 0.9965 | -2.6715 | -1.5694 | 1.5353 | 2.5213 |
| 9.0000 | 191,453 | 0.9881 | -2.5332 | -1.5504 | 1.5336 | 2.5311 |

Ver también `TDA10_cuantiles_por_anio.csv` (misma tabla por año) y `TDA10_perfil_cuantiles_decil.png` (perfil visual, primario raw y clock_adjusted lado a lado).

## 9. Sensibilidad a estimador/ventana (las 12 configuraciones)

| config | fraction_removed_full | fraction_removed_trimmed | decile_stability_ratio | segment_stability_ratio | year_stability_ratio | config_label |
|---:|---:|---:|---:|---:|---:|---:|
| rolling_std_30_raw | 0.233 | 0.799 | 0.250 | 0.064 | 0.038 | MIXTO |
| rolling_std_120_raw | -0.401 | 0.748 | 0.157 | 0.408 | 0.027 | MIXTO |
| rolling_std_390_raw | -0.602 | 0.622 | 0.222 | 0.663 | 0.064 | FORMA_SUSTANCIAL |
| ewma_20_raw | 0.136 | 0.815 | 0.128 | 0.261 | 0.032 | ESCALA_DOMINA |
| ewma_60_raw | -0.384 | 0.754 | 0.069 | 0.498 | 0.026 | MIXTO |
| ewma_240_raw | -0.939 | 0.584 | 0.055 | 0.966 | 0.028 | FORMA_SUSTANCIAL |
| rolling_std_30_clock_adjusted | 0.796 | 0.859 | 0.223 | 0.115 | 0.047 | ESCALA_DOMINA |
| rolling_std_120_clock_adjusted | 0.640 | 0.845 | 0.100 | 0.193 | 0.048 | ESCALA_DOMINA |
| rolling_std_390_clock_adjusted | 0.542 | 0.793 | 0.092 | 0.310 | 0.102 | MIXTO |
| ewma_20_clock_adjusted | 0.774 | 0.875 | 0.111 | 0.116 | 0.052 | ESCALA_DOMINA |
| ewma_60_clock_adjusted | 0.675 | 0.855 | 0.043 | 0.191 | 0.051 | ESCALA_DOMINA |
| ewma_240_clock_adjusted | 0.532 | 0.805 | 0.019 | 0.262 | 0.095 | ESCALA_DOMINA |

`fraction_removed_full` es la versión SIN recortar — se reporta por transparencia (roadmap, tabla central) pero es el motivo por el que un estimador de ventana corta puede mostrar valores extremos (ver §4: un único `sigma_hat` numéricamente indistinguible de cero, dentro de una ventana de precios casi constantes, puede disparar un `z_t` de millones de desviaciones — protegido explícitamente por `MIN_VALID_SIGMA_HAT`, pero incluso protegido, un puñado de sorpresas genuinas puede seguir dominando un momento de cuarto orden sin recortar). **La clasificación de cada configuración usa `fraction_removed_trimmed`** — la cifra que TDA-07 (informe, §12) recomendó explícitamente como referencia más estable antes de esta etapa.

**Robustez**: 12 configuraciones evaluadas, distribución de etiquetas: MIXTO=4, FORMA_SUSTANCIAL=2, ESCALA_DOMINA=6. La etiqueta mayoritaria (`ESCALA_DOMINA`) cubre una fracción **0.50** de las configuraciones — INSUFICIENTE para declarar el veredicto robusto (umbral predeclarado: 0.75).

**Patrón por escala DETERMINISTA vs DINÁMICA** (§3, ajuste de reloj): en la tabla de arriba, 5 de las 6 configuraciones `clock_adjusted` (que primero retiran `s(m)`, RETROSPECTIVO, y luego aplican el estimador causal) clasifican como `ESCALA_DOMINA`, frente a solo 1 de las 6 `raw` (que solo aplican el estimador causal, sin retirar antes el patrón horario). Esto sugiere que buena parte de lo que aparenta ser FORMA cuando se usa exclusivamente un estimador causal dinámico es, en realidad, escala DETERMINISTA (el patrón horario de TDA-06) que ese estimador —reactivo pero lento— no captura bien por sí solo. Esta distinción (determinista vs dinámica) es precisamente la que la tarea pidió no confundir; **no cambia el veredicto GLOBAL** (que se basa en las 12 configuraciones, no solo en las 6 `clock_adjusted`, y la versión `clock_adjusted` no es causal de principio a fin — depende de `s(m)` RETROSPECTIVO), pero es evidencia relevante para TDA-11/12.

## 10. QQ-plots

- `TDA10_qq_primario.png`: `r_1m` crudo vs. normal, y `z_t` (primario) vs. normal — cuánta cola desaparece al retirar la escala dinámica.
- `TDA10_qq_sensibilidad.png`: `z_t` superpuesto para las 6 configuraciones "raw" y las 6 "clock_adjusted" — sensibilidad visual del QQ a estimador/ventana/half-life.
- `TDA10_perfil_cuantiles_decil.png`: perfil de cuantiles de `z_t` por decil de volatilidad — el diagnóstico visual directo de estabilidad de FORMA.

## 11. Veredicto final — TH22

**`MIXTO`**

El resultado no es uniforme: una parte sustancial de la curtosis desaparece al estandarizar, pero persiste evidencia de forma (perfiles de cuantiles que no se superponen del todo, o una conclusion que cambia segun el estimador/ventana/anio). En palabras simples: la escala explica una parte real del fenomeno, pero no toda -- ni 'todo es escala' ni 'todo es forma' describe correctamente a MNQ con la evidencia de esta etapa.

Reglas operativas predeclaradas (antes de ejecutar sobre el conjunto de investigación real, nunca ajustadas después — la ÚNICA decisión tomada después de ver el resultado fue usar `fraction_removed_trimmed` en vez de `fraction_removed_full` como entrada de estas reglas, ver nota de §9: no es un umbral ajustado, es una corrección de qué métrica alimenta las mismas reglas, justificada independientemente por la recomendación previa de TDA-07): `ESCALA_DOMINA` exige `fraction_removed_trimmed >= 0.8` y el mayor de los tres ratios de estabilidad (decil/segmento/año) `<= 0.3`; `FORMA_SUSTANCIAL` exige `fraction_removed_trimmed <= 0.5` o algún ratio `>= 0.6`; el veredicto GLOBAL exige que al menos el 75% de las 12 configuraciones coincida en la misma etiqueta, o se reporta `MIXTO`.

**Importante (roadmap, riesgo explícito):** que la curtosis baje NO implica que `z_t` sea normal — puede bajar sustancialmente y seguir siendo una distribución de colas pesadas (ver §4/§5: la curtosis de `z_t` casi nunca es cercana a 0, aunque sea mucho menor que la de `r`).

### TH22 / TH26 / STOP-13

- **TH22 = `MIXTO`** — RESUELTA por esta etapa.
- **TH26 = `PARCIALMENTE_INFORMADA`** — las tablas de cuantiles por segmento/decil/año de esta etapa (§7/§8) son evidencia PARCIAL y análoga a lo que TH26 pide, pero TH26 formalmente requiere los cuantiles condicionales completos con bootstrap por grupo y el `n` que sostiene cada cuantil extremo — eso pertenece a TDA-12 (obligatoria), no se declara resuelta aquí.
- **STOP-13**: `NO SUGERIDO` — TDA-10 no encontro ESCALA_DOMINA robusta (veredicto=MIXTO, agreement=0.50). No se sugiere STOP-13; TDA-12 debe evaluar formalmente si procede EVT.

## 12. Tiempo por etapa (análisis)

| Etapa | Tiempo |
|---|---:|
| Carga de datos, poblaciones (r_1m/r_tilde) e invariantes | 2.1s |
| Verificacion de ausencia de look-ahead (G1) para las 12 configuraciones | 2.7s |
| Construccion causal de sigma_hat y z_t (12 configuraciones) | 0.8s |
| Tabla central de curtosis (global + por año, completa + recortada) para las 12 configuraciones | 7.8s |
| Bootstrap de bloques por jornada (IC 95%) para la configuracion primaria | 26.8s |
| Estabilidad de forma por decil de volatilidad, segmento y año (headline + sensibilidad) | 10.9s |
| QQ-plots, veredicto TH22 y sugerencia de STOP-13 | 1.2s |

**Tiempo total de la ejecución (análisis + escritura de CSV/PNG/MD): 53.1s (~0.9 min).**

## 13. Archivos generados

`TDA10_curtosis_escala_vs_forma.csv`, `TDA10_curtosis_bootstrap_ci.csv`, `TDA10_cuantiles_por_decil_volatilidad.csv`, `TDA10_cuantiles_por_segmento.csv`, `TDA10_cuantiles_por_anio.csv`, `TDA10_sensibilidad_estimador_ventana.csv`, `TDA10_verificacion_causalidad.csv`, `TDA10_qq_puntos.csv` (8 CSV) + `TDA10_qq_primario.png`, `TDA10_qq_sensibilidad.png`, `TDA10_perfil_cuantiles_decil.png` (3 PNG) + este informe (MD).

## 14. Comandos de validación

```
python -m pytest -q tests/test_tda10_scale_vs_shape.py
python -m pytest -q
python -m ohlcv_dataroad.ingest.run_tda10 --config configs/mnq_snapshot.yaml
```

## 15. Estado final

**`PASS_WITH_OPEN_QUESTIONS`**

- TH22 = `MIXTO` (robusto: False, agreement=0.50)
- TH26 = `PARCIALMENTE_INFORMADA` (formalmente pendiente de TDA-12)
- STOP-13 = NO SUGERIDO

**No se avanza a TDA-11 ni TDA-12 en esta tarea.**

## 16. Preguntas abiertas

1. El veredicto por configuración usa el mayor de los tres ratios de estabilidad (decil/segmento/año) — una configuración puede ser estable en dos dimensiones y no en la tercera; ver `TDA10_sensibilidad_estimador_ventana.csv` para el detalle por dimensión.
2. No se probó ningún estimador de volatilidad basado en rango (Parkinson/Rogers-Satchell/Yang-Zhang, roadmap TDA-04 §"método avanzado opcional") — dos familias mínimas (rodante + EWMA) son suficientes para responder la pregunta de bifurcación (G4); queda como extensión posible si TDA-11 llegara a ejecutarse.
3. TH26 queda solo PARCIALMENTE informada — TDA-12 debe producir la versión completa (bootstrap por grupo, `n` por cuantil extremo).

---

## Modo sencillo — en 10 líneas

**¿Los movimientos extremos de MNQ son extremos por la hora del día y la volatilidad reciente, o son extremos "de verdad"?** El resultado es mixto: una parte real desaparece al descontar la volatilidad reciente, pero no toda.

**¿Cómo se midió "la volatilidad que ya era previsible"?** Con dos formas simples de mirar solo el pasado — el desvío estándar de los últimos minutos, y un promedio que da más peso a lo reciente (EWMA) — nunca usando el propio movimiento que se está evaluando ni información futura (verificado con una prueba explícita).

**¿Depende de qué "regla" de volatilidad se use?** Se probaron 12 combinaciones (2 formas de medir volatilidad × 3 configuraciones cada una × con/sin ajuste por hora del día) — el resultado NO es el mismo en todas — se reporta como MIXTO precisamente por eso.

**¿Esto confirma que los retornos ajustados son "normales"?** No. Puede bajar mucho la curtosis y seguir sin ser una campana de Gauss — solo dice cuánta de la anormalidad viene de la escala cambiante.

**¿Qué NO significa este resultado?** No es una señal de trading, no dice hacia dónde se moverá el precio, no ajusta ningún modelo GARCH, y no decide si hace falta un modelo de eventos extremos (EVT) — esa decisión formal es de TDA-12.
