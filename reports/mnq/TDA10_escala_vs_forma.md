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

## 4. Resultado de curtosis — versión COMPLETA (sin recortar)

Curtosis de `r` (restringido a la población donde `z` está definido) vs. curtosis de `z`, fracción eliminada **sin recortar**, las 12 configuraciones, alcance GLOBAL — se reporta íntegramente (roadmap, tabla central) pero **no decide el veredicto** (ver §6 y §13 para el porqué):

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

**Piso numérico de `sigma_hat` (`MIN_VALID_SIGMA_HAT=1e-08`)**: protege la división `z_t=r_t/sigma_hat_(t-1)` de un `sigma_hat` positivo pero numéricamente indistinguible de cero (residual de punto flotante en una ventana de precios casi constantes, no volatilidad real) — sin él, un solo punto así puede disparar un `z_t` de millones de desviaciones estándar y arrastrar la curtosis sin recortar de esa configuración a valores sin sentido. **No es un mecanismo para descartar extremos reales**: solo excluye `sigma_hat` por debajo de `1e-08` — más de 3 órdenes de magnitud por debajo del retorno más pequeño posible dado el tick de MNQ (~1,47×10⁻⁵) — nunca un retorno o una desviación grande. Filas afectadas, contadas automáticamente (columna `n_sigma_floor_excluded` de `TDA10_sensibilidad_estimador_ventana.csv`), por configuración:

| config | n_sigma_floor_excluded |
|---:|---:|
| rolling_std_30_raw | 1 |
| rolling_std_30_clock_adjusted | 11 |

Total de filas afectadas en las 12 configuraciones combinadas: **12** (sobre una población de 1,914,530 filas por configuración) — la razón por la que la curtosis SIN recortar de la(s) configuración(es) afectada(s) puede ser extrema y no es la métrica que decide el veredicto (ver §6).

## 5. Resultado de curtosis — versión RECORTADA 0.1% (convención TDA-07)

Misma tabla, columnas `*_trimmed` (recorte del 0.1% total, 0.05% por cola, igual convención que TDA-07). **Esta es la métrica que alimenta el veredicto formal** (`classify_config`, ver §13):

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

La diferencia entre la fracción eliminada completa (§4) y la recortada (aquí) es en sí misma informativa: si la recortada es mucho menor (o, como en `rolling_std_30_raw`, la completa se ve artificialmente MENOR por el artefacto numérico de §4), la reducción de curtosis de la versión completa depende en gran parte de un puñado de observaciones — consistente con TDA-07, que ya mostró que la curtosis recortada de `r_1m` es mucho más estable que la cruda.

## 6. Qué metrica decide el veredicto — auditoría de transparencia

El roadmap pide reportar la curtosis en ambas versiones; esta etapa las reporta y además **clasifica ambas por separado** (`config_label_full` vs `config_label_trimmed`, tabla completa en §10) para que quede explícito qué concluye cada una. La decisión de que `classify_config` use exclusivamente `fraction_removed_trimmed` (nunca `fraction_removed_full`) se tomó **después** de observar, sobre el conjunto de investigación real, que la versión completa de `rolling_std_30` se disparaba por el artefacto numérico de §4 — no se ocultó esa decisión, ni se ajustó ningún UMBRAL después de verla (`0.8`/`0.5`/`0.3`/`0.6` son idénticos a la primera ejecución). Está justificada, además, de forma independiente por TDA-07 (informe, §12, escrito antes de que TDA-10 existiera): "la curtosis recortada... es la cifra más estable disponible... para juzgar cuánta de la no-normalidad es genuina".

## 7. Incertidumbre bootstrap de la métrica primaria (completa + recortada)

Bootstrap de bloques por jornada (G5, `n_boot=300`), configuración primaria `ewma_60_raw` (obligatorio) y su contraparte `ewma_60_clock_adjusted` (diagnóstico retrospectivo secundario, barato con el mismo motor):

**`ewma_60_raw` (RAW/CAUSAL, la pregunta principal)**:

| | punto | IC 95% lo | IC 95% hi |
|---|---:|---:|---:|
| kurt_r (completa) | 111.545 | 70.028 | 163.884 |
| kurt_z (completa) | 154.350 | 97.541 | 229.547 |
| fraction_removed (completa) | -0.3837 | -1.3461 | 0.2356 |
| kurt_r (**recortada**) | 11.490 | 9.862 | 13.322 |
| kurt_z (**recortada**) | 2.822 | 2.764 | 2.891 |
| fraction_removed (**recortada, la que decide el veredicto**) | 0.7544 | 0.7146 | 0.7890 |

**`ewma_60_clock_adjusted` (CLOCK_ADJUSTED/RETROSPECTIVO, diagnóstico secundario)**:

| | punto | IC 95% lo | IC 95% hi |
|---|---:|---:|---:|
| kurt_r (completa) | 78.281 | 56.774 | 96.712 |
| kurt_z (completa) | 25.429 | 16.650 | 38.897 |
| fraction_removed (completa) | 0.6752 | 0.4679 | 0.8017 |
| kurt_r (recortada) | 11.360 | 8.941 | 14.232 |
| kurt_z (recortada) | 1.646 | 1.601 | 1.704 |
| fraction_removed (recortada) | 0.8551 | 0.8157 | 0.8853 |

El IC de la fracción **completa** puede ser negativo/amplio (momento de cuarto orden, frágil ante un puñado de observaciones — ver §4/§6); el IC de la fracción **recortada** (la que decide el veredicto) es la incertidumbre que corresponde a §13.

## 8. Estabilidad por año (configuración primaria `ewma_60_raw`)

| scope_value | n | kurt_r | kurt_z | fraction_removed | fraction_removed_trimmed |
|---:|---:|---:|---:|---:|---:|
| 2,019 | 7,226 | 32.344 | 10.828 | 0.665 | 0.529 |
| 2,020 | 342,648 | 44.369 | 8.569 | 0.807 | 0.765 |
| 2,021 | 349,039 | 24.023 | 23.679 | 0.014 | 0.646 |
| 2,022 | 349,996 | 130.763 | 354.286 | -1.709 | 0.593 |
| 2,023 | 349,636 | 53.795 | 196.253 | -2.648 | 0.674 |
| 2,024 | 352,261 | 54.183 | 104.280 | -0.925 | 0.691 |
| 2,025 | 163,544 | 138.767 | 237.407 | -0.711 | 0.778 |

## 9. Estabilidad por segmento horario y por decil de volatilidad (headline: primario raw)

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

Por decil de `sigma_hat_(t-1)` (0=más tranquilo, 9=más volátil):

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

## 10. Sensibilidad a estimador/ventana (las 12 configuraciones, ambas métricas)

| config | fraction_removed_full | fraction_removed_trimmed | max_stability_ratio | config_label_full | config_label_trimmed | is_borderline |
|---:|---:|---:|---:|---:|---:|---:|
| rolling_std_30_raw | 0.233 | 0.799 | 0.250 | FORMA_SUSTANCIAL | MIXTO | True |
| rolling_std_120_raw | -0.401 | 0.748 | 0.408 | FORMA_SUSTANCIAL | MIXTO | False |
| rolling_std_390_raw | -0.602 | 0.622 | 0.663 | FORMA_SUSTANCIAL | FORMA_SUSTANCIAL | False |
| ewma_20_raw | 0.136 | 0.815 | 0.261 | FORMA_SUSTANCIAL | ESCALA_DOMINA | True |
| ewma_60_raw | -0.384 | 0.754 | 0.498 | FORMA_SUSTANCIAL | MIXTO | False |
| ewma_240_raw | -0.939 | 0.584 | 0.966 | FORMA_SUSTANCIAL | FORMA_SUSTANCIAL | False |
| rolling_std_30_clock_adjusted | 0.796 | 0.859 | 0.223 | MIXTO | ESCALA_DOMINA | False |
| rolling_std_120_clock_adjusted | 0.640 | 0.845 | 0.193 | MIXTO | ESCALA_DOMINA | False |
| rolling_std_390_clock_adjusted | 0.542 | 0.793 | 0.310 | MIXTO | MIXTO | True |
| ewma_20_clock_adjusted | 0.774 | 0.875 | 0.116 | MIXTO | ESCALA_DOMINA | False |
| ewma_60_clock_adjusted | 0.675 | 0.855 | 0.191 | MIXTO | ESCALA_DOMINA | False |
| ewma_240_clock_adjusted | 0.532 | 0.805 | 0.262 | MIXTO | ESCALA_DOMINA | True |

`config_label_full`/`config_label_trimmed`: la clasificación (`classify_config`) de CADA configuración con cada métrica de curtosis — muestra explícitamente qué hubiera concluido la versión completa frente a la recortada (§6). **Solo `config_label_trimmed` alimenta el veredicto formal** (§13).

**Configuraciones BORDERLINE** (distancia a algún umbral `<= 0.02`, meta-información descriptiva — nunca cambia `classify_config` ni los umbrales): `rolling_std_30_raw` (distancia mínima=0.0011), `ewma_20_raw` (distancia mínima=0.0148), `rolling_std_390_clock_adjusted` (distancia mínima=0.0069), `ewma_240_clock_adjusted` (distancia mínima=0.0047). Estos resultados no deben leerse como diferencias fuertes frente al umbral — ver `TDA10_sensibilidad_estimador_ventana.csv` (columnas `dist_*`) para el detalle firmado por umbral.

## 11. Separación RAW/CAUSAL vs CLOCK_ADJUSTED/RETROSPECTIVO

Las 12 configuraciones NO tienen el mismo estatus epistemológico: `raw` es causal de principio a fin; `clock_adjusted` depende de `s(m)` (TDA-06, RETROSPECTIVO). Por eso el veredicto NO es una votación ciega de las 12 — se reportan tres resúmenes separados:

### A. RAW / CAUSAL — la pregunta principal

**RAW/CAUSAL** — 6 configuraciones. Distribución de etiquetas (métrica RECORTADA, oficial): MIXTO=3, FORMA_SUSTANCIAL=2, ESCALA_DOMINA=1. Acuerdo de la etiqueta mayoritaria: **0.50** (INSUFICIENTE, umbral=0.75). Veredicto del bloque: **`MIXTO`**. Diagnóstico con la métrica COMPLETA/sin recortar (transparencia, nunca decide): FORMA_SUSTANCIAL=6, acuerdo=1.00, veredicto=FORMA_SUSTANCIAL.

### B. CLOCK_ADJUSTED / RETROSPECTIVO — diagnóstico secundario (nunca disponible causalmente en producción)

**CLOCK_ADJUSTED/RETROSPECTIVO** — 6 configuraciones. Distribución de etiquetas (métrica RECORTADA, oficial): ESCALA_DOMINA=5, MIXTO=1. Acuerdo de la etiqueta mayoritaria: **0.83** (robusto). Veredicto del bloque: **`ESCALA_DOMINA`**. Diagnóstico con la métrica COMPLETA/sin recortar (transparencia, nunca decide): MIXTO=6, acuerdo=1.00, veredicto=MIXTO.

### C. GLOBAL / SÍNTESIS

El veredicto FORMAL de TH22 (§13) es **siempre** el del bloque A (RAW/CAUSAL) — nunca el de B, y nunca una mezcla aritmética de ambos. El bloque B se usa solo para calificar la interpretación: los dos bloques DIVERGEN — el bloque RAW/CAUSAL da `MIXTO` mientras que el diagnóstico CLOCK_ADJUSTED/RETROSPECTIVO da `ESCALA_DOMINA`. Esto sugiere que buena parte de lo que el bloque causal puro ve como FORMA podría deberse a la componente DETERMINISTA de reloj (patrón horario, TDA-06) que un estimador causal dinámico, aplicado solo sobre `r_1m`, no captura bien por sí solo — evidencia relevante para TDA-11/TDA-12, pero que NO cambia el veredicto formal de esta etapa (el bloque B es RETROSPECTIVO, no disponible causalmente).

## 12. QQ-plots

- `TDA10_qq_primario.png`: `r_1m` crudo vs. normal, y `z_t` (primario) vs. normal — cuánta cola desaparece al retirar la escala dinámica.
- `TDA10_qq_sensibilidad.png`: `z_t` superpuesto para las 6 configuraciones "raw" y las 6 "clock_adjusted" — sensibilidad visual del QQ a estimador/ventana/half-life.
- `TDA10_perfil_cuantiles_decil.png`: perfil de cuantiles de `z_t` por decil de volatilidad — el diagnóstico visual directo de estabilidad de FORMA.

## 13. Veredicto final — TH22

**`MIXTO`** (basado exclusivamente en el bloque RAW/CAUSAL, §11.A)

El bloque RAW/CAUSAL (la pregunta principal, la unica que usa exclusivamente sigma_hat causal) no alcanza el umbral de robustez predeclarado -- ni 'todo es escala' ni 'todo es forma' describe correctamente el resultado causal puro. En palabras simples: la escala explica una parte real del fenomeno, pero no toda, y esa conclusion no es la misma segun el estimador/ventana que se use dentro del bloque causal.

Reglas operativas predeclaradas (antes de ejecutar sobre el conjunto de investigación real; los UMBRALES nunca se ajustaron después de ver ningún resultado, en ninguna de las dos ejecuciones — ver §6 para la única decisión de METRICA, que sí se tomó después de ver un resultado y se documenta con total transparencia): `ESCALA_DOMINA` exige `fraction_removed_trimmed >= 0.8` y el mayor de los tres ratios de estabilidad (decil/segmento/año) `<= 0.3`; `FORMA_SUSTANCIAL` exige `fraction_removed_trimmed <= 0.5` o algún ratio `>= 0.6`; el veredicto de CADA BLOQUE (6 configuraciones) exige que al menos el 75% coincida en la misma etiqueta, o el bloque se reporta `MIXTO`; el veredicto GLOBAL es siempre el del bloque RAW/CAUSAL (§11).

**Importante (roadmap, riesgo explícito):** que la curtosis baje NO implica que `z_t` sea normal — puede bajar sustancialmente y seguir siendo una distribución de colas pesadas (ver §4/§5: la curtosis de `z_t` casi nunca es cercana a 0, aunque sea mucho menor que la de `r`).

### TH22 / TH26 / STOP-13

- **TH22 = `MIXTO`** — RESUELTA por esta etapa (bloque RAW/CAUSAL; diagnóstico CLOCK_ADJUSTED en §11.B/C).
- **TH26 = `PARCIALMENTE_INFORMADA`** — las tablas de cuantiles por segmento/decil/año de esta etapa (§9) son evidencia PARCIAL y análoga a lo que TH26 pide, pero TH26 formalmente requiere los cuantiles condicionales completos con bootstrap por grupo y el `n` que sostiene cada cuantil extremo — eso pertenece a TDA-12 (obligatoria), no se declara resuelta aquí.
- **STOP-13**: `NO SUGERIDO` — TDA-10 no encontro ESCALA_DOMINA robusta en el bloque RAW/CAUSAL (veredicto=MIXTO, agreement=0.50). No se sugiere STOP-13; TDA-12 debe evaluar formalmente si procede EVT.

## 14. Tiempo por etapa (análisis)

| Etapa | Tiempo |
|---|---:|
| Carga de datos, poblaciones (r_1m/r_tilde) e invariantes | 2.2s |
| Verificacion de ausencia de look-ahead (G1) para las 12 configuraciones | 2.8s |
| Construccion causal de sigma_hat y z_t (12 configuraciones) | 0.9s |
| Tabla central de curtosis (global + por año, completa + recortada) para las 12 configuraciones | 7.7s |
| Bootstrap de bloques por jornada (IC 95%, completa+recortada) para primaria raw y clock_adjusted | 140.9s |
| Estabilidad de forma por decil de volatilidad, segmento y año (headline + sensibilidad) | 10.8s |
| QQ-plots, veredicto TH22 (RAW/CAUSAL vs CLOCK_ADJUSTED) y sugerencia de STOP-13 | 1.2s |

**Tiempo total de la ejecución (análisis + escritura de CSV/PNG/MD): 167.1s (~2.8 min).**

## 15. Archivos generados

`TDA10_curtosis_escala_vs_forma.csv`, `TDA10_curtosis_bootstrap_ci.csv`, `TDA10_cuantiles_por_decil_volatilidad.csv`, `TDA10_cuantiles_por_segmento.csv`, `TDA10_cuantiles_por_anio.csv`, `TDA10_sensibilidad_estimador_ventana.csv`, `TDA10_verificacion_causalidad.csv`, `TDA10_qq_puntos.csv` (8 CSV) + `TDA10_qq_primario.png`, `TDA10_qq_sensibilidad.png`, `TDA10_perfil_cuantiles_decil.png` (3 PNG) + este informe (MD).

## 16. Comandos de validación

```
python -m pytest -q tests/test_tda10_scale_vs_shape.py
python -m pytest -q
python -m ohlcv_dataroad.ingest.run_tda10 --config configs/mnq_snapshot.yaml
```

## 17. Estado final

**`PASS_WITH_OPEN_QUESTIONS`**

- TH22 = `MIXTO` (bloque RAW/CAUSAL, robusto: False, agreement=0.50; diagnóstico CLOCK_ADJUSTED: `ESCALA_DOMINA`, DIVERGE)
- TH26 = `PARCIALMENTE_INFORMADA` (formalmente pendiente de TDA-12)
- STOP-13 = NO SUGERIDO

**No se avanza a TDA-11 ni TDA-12 en esta tarea.**

## 18. Preguntas abiertas

1. El veredicto por configuración usa el mayor de los tres ratios de estabilidad (decil/segmento/año) — una configuración puede ser estable en dos dimensiones y no en la tercera; ver `TDA10_sensibilidad_estimador_ventana.csv` para el detalle por dimensión.
2. No se probó ningún estimador de volatilidad basado en rango (Parkinson/Rogers-Satchell/Yang-Zhang, roadmap TDA-04 §"método avanzado opcional") — dos familias mínimas (rodante + EWMA) son suficientes para responder la pregunta de bifurcación (G4); queda como extensión posible si TDA-11 llegara a ejecutarse.
3. TH26 queda solo PARCIALMENTE informada — TDA-12 debe producir la versión completa (bootstrap por grupo, `n` por cuantil extremo).
4. El diagnóstico CLOCK_ADJUSTED (§11.B) no es causal de principio a fin (depende de `s(m)` RETROSPECTIVO) — si TDA-11/TDA-12 quisieran aprovechar esa señal (que gran parte de la FORMA aparente es reloj determinista), necesitarían una versión causal del ajuste horario, que esta etapa no construye.

---

## Modo sencillo — en 10 líneas

**¿Los movimientos extremos de MNQ son extremos por la hora del día y la volatilidad reciente, o son extremos "de verdad"?** El resultado es mixto: una parte real desaparece al descontar la volatilidad reciente, pero no toda.

**¿Cómo se midió "la volatilidad que ya era previsible"?** Con dos formas simples de mirar solo el pasado — el desvío estándar de los últimos minutos, y un promedio que da más peso a lo reciente (EWMA) — nunca usando el propio movimiento que se está evaluando ni información futura (verificado con una prueba explícita).

**¿Depende de qué "regla" de volatilidad se use?** Se probaron 12 combinaciones (2 formas de medir volatilidad × 3 configuraciones cada una × con/sin ajuste por hora del día). Las 6 que solo usan información causal (`RAW`) son las que deciden la respuesta oficial — NO coinciden entre sí — por eso el resultado se reporta como MIXTO. Las otras 6 (`CLOCK_ADJUSTED`) son un diagnóstico aparte que no puede usarse en producción (necesita conocer de antemano el patrón horario completo) — NO confirman la misma conclusión.

**¿Esto confirma que los retornos ajustados son "normales"?** No. Puede bajar mucho la curtosis y seguir sin ser una campana de Gauss — solo dice cuánta de la anormalidad viene de la escala cambiante.

**¿Qué NO significa este resultado?** No es una señal de trading, no dice hacia dónde se moverá el precio, no ajusta ningún modelo GARCH, y no decide si hace falta un modelo de eventos extremos (EVT) — esa decisión formal es de TDA-12.
