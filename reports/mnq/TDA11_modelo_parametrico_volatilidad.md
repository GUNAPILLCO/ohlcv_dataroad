# TDA-11 — Modelo paramétrico de volatilidad

**Etapa:** `docs/methodology/Tsay_OHLCV_analysis_roadmap.md` § TDA-11 *(CONDICIONAL)*
**Depende de:** TDA-09 (`PASS_WITH_OPEN_QUESTIONS` — TH21=`CLUSTERING_GENUINO`, STOP-9 NO activado), TDA-10 (`PASS_WITH_OPEN_QUESTIONS` — TH22=`MIXTO`), TDA-08 (STOP-8a activado — dependencia en media despreciable, por eso GARCH usa media CERO).
**Librería GARCH:** `arch` versión `8.0.0`.
**Alcance de datos:** exclusivamente `tda04_variables_1m.parquet`, `tda04_return_validity_mask.parquet`, `tda06_r_tilde.parquet`, `tda06_s_m.parquet`, `TDA06_segmentacion_propuesta.csv` y `TDA09_clock_attribution.csv`. Ningún archivo de `data/raw/` ni de `holdout_files` fue abierto en esta etapa.
**Generado automáticamente** por `python -m ohlcv_dataroad.ingest.run_tda11 --config configs/mnq_snapshot.yaml` — este informe NUNCA se edita a mano; cualquier corrección se hace en el código (`tda11_parametric_volatility.py`/`run_tda11.py`) y se regenera con una nueva ejecución.

> Esta etapa pregunta si un GARCH(1,1) — el modelo paramétrico más simple — resume la persistencia de volatilidad de MNQ (ya encontrada en TDA-09) en parámetros interpretables y estables, y si elimina dependencia residual que los benchmarks causales simples de TDA-10 no eliminan. Sigue siendo caracterización estadística: NO evalúa capacidad predictiva OOS, rentabilidad, señales, targets ni features. NO abre el holdout.

---

## 1. Puerta de entrada

- **TH21 recomputado** (desde `TDA09_clock_attribution.csv`, con el mismo código de `tda09_volatility_clustering.classify_th21`, nunca reparseado de texto): `CLUSTERING_GENUINO`.
- **STOP-9 recomputado** (`tda09_volatility_clustering.decide_stop9`): NO ACTIVADO.
- TH21 recomputado = CLUSTERING_GENUINO y STOP-9 = NO ACTIVADO (criterio de entrada 1 satisfecho). La pregunta escrita (WRITTEN_QUESTION) esta declarada y pertenece a caracterizacion, no a prediccion (criterios 2 y 3). Puerta ABIERTA -- se construyen los benchmarks.

## 2. Pregunta escrita que justificó (o no) el modelo

> ¿Puede el modelo parametrico de volatilidad mas simple resumir la persistencia genuina encontrada en TDA-09 mediante parametros interpretables y razonablemente estables entre subperiodos, y eliminar dependencia residual de volatilidad que los benchmarks simples no eliminan?

Preguntas secundarias: (1) valor y estabilidad de `alpha+beta` entre años/segmentos; (2) ¿deja GARCH(1,1) menos dependencia residual que EWMA/rodante/rango?; (3) ¿existe asimetría descriptiva estable respecto del signo del shock previo?

**Respuesta**: GARCH(1,1) APORTA información sobre los benchmarks simples (ver §8/§13 para el detalle cuantitativo).

## 3. Benchmarks simples (obligatorios antes del modelo paramétrico)

Tres familias causales, reutilizadas de TDA-10 sin modificarlas (EWMA/rodante) + una nueva (rango, Parkinson causal): `('ewma_60', 'rolling_120', 'range_ewma_60')`. `ewma_60`/`rolling_120` son exactamente `build_sigma_hat` de TDA-10; `range_ewma_60` es nuevo en esta etapa (`sigma_park_t^2 = log_hl_t^2/(4·ln 2)`, suavizado con EWMA causal half-life=60, ver `causal_range_sigma`). Diagnóstico de dependencia residual (`ACF(|z|)`, rezago 1, y `Q(60)` portmanteau) por bloque:

| bloque | config | n_finito | rho1(|z|) | Q(60)_|z| | n_comun | Q(60)_|z|_comun |
|---:|---:|---:|---:|---:|---:|---:|
| raw | ewma_60 | 1,914,350 | 0.1625 | 441,131.8399 | 1,914,350 | 441,131.8399 |
| raw | rolling_120 | 1,914,410 | 0.1646 | 355,873.0435 | 1,914,350 | 355,885.3755 |
| raw | range_ewma_60 | 1,914,350 | 0.1726 | 520,079.5455 | 1,914,350 | 520,079.5455 |
| clock_adjusted | ewma_60 | 1,914,350 | 0.1061 | 91,023.0210 | 1,914,350 | 91,023.0210 |
| clock_adjusted | rolling_120 | 1,914,410 | 0.1177 | 119,865.4358 | 1,914,350 | 119,871.0426 |
| clock_adjusted | range_ewma_60 | 1,914,350 | 0.1168 | 125,287.9095 | 1,914,350 | 125,287.9095 |

Las columnas `*_comun` repiten el diagnóstico sobre la población COMÚN con GARCH (mismas filas, bloques y pares — ver §8); son las que usa la decisión.

## 4. Especificación exacta del modelo

**Primario**: GARCH(1,1), media CERO (justificado por TDA-08/STOP-8a: dependencia en media despreciable — `0,1425 ticks`), sobre `r_tilde` (RETROSPECTIVO, clock-adjusted). Distribución de innovaciones: Gaussiana (QMLE), inferencia ROBUSTA (`cov_type="robust"`, sandwich Bollerslev-Wooldridge). Escala de ajuste: `r_tilde·10000` (verificado empíricamente necesario para convergencia estable — ver docstring de módulo; `alpha`/`beta`/persistencia son invariantes a esta elección).

**Secundario (sensibilidad)**: idéntica especificación sobre `r_1m` (RAW/CAUSAL), alcance GLOBAL únicamente (G4 — la pregunta formal ya la responde la serie principal).

**Sensibilidad de distribución**: idéntica especificación sobre `r_tilde`, innovaciones Student-t, alcance GLOBAL (justificada por la curtosis residual material encontrada en TDA-10, nunca una búsqueda de la distribución que mejor ajusta).

## 5. Parámetros e intervalos (IC 95% robustos)

| config | bloque | dist | n | omega | alpha | alpha_ci_lo | alpha_ci_hi | beta | beta_ci_lo | beta_ci_hi | nu | persistence | at_constraint_boundary | converged_verified |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| garch11_primary_global | clock_adjusted | normal | 1,914,530 | 0.00000 | 0.05191 | 0.04949 | 0.05432 | 0.94809 | 0.94575 | 0.95043 | NaN | 1.00000 | True | True |
| garch11_secondary_raw_global | raw | normal | 1,914,530 | 0.00000 | 0.07918 | 0.07478 | 0.08359 | 0.92082 | 0.91569 | 0.92595 | NaN | 1.00000 | True | True |
| garch11_student_t_sensitivity | clock_adjusted | t | 1,914,530 | 0.00000 | 0.04636 | 0.04552 | 0.04719 | 0.95364 | 0.95282 | 0.95446 | 6.39839 | 1.00000 | True | True |

**Validez numérica de cada ajuste** (auditoría correctiva, C2):

| config | loglikelihood | loglikelihood_arch_multistart | refinement_llf_gain | kkt_max_abs_gradient | converged_verified | convergence_flag | eps_selected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| garch11_primary_global | -4,685,851.01 | -4,685,851.01 | 0.00 | 1.21e-07 | True | 0 | 1e-07 |
| garch11_secondary_raw_global | -4,490,557.90 | -4,560,220.42 | 69,662.52 | 6.00e-08 | True | 0 | 1e-05 |
| garch11_student_t_sensitivity | -4,605,277.46 | -4,616,835.01 | 11,557.54 | 1.43e-07 | True | 0 | 1e-05 |

Los 3 ajustes globales superan la verificación de convergencia.

**Cómo leer esta tabla.** El `convergence_flag=0` del optimizador interno de `arch` (SLSQP) **no acredita convergencia**: sobre estos datos sus 4 intentos (grilla `GARCH_OPTIMIZER_EPS_GRID=(None, 1e-07, 1e-06, 1e-05)`) declaran éxito en puntos distintos, y en el primer cierre de TDA-11 eso dejó publicados como válidos dos ajustes (secundario `raw` y Student-t) que no eran el máximo de la verosimilitud. Desde la auditoría correctiva cada ajuste (A) parte del mejor intento de `arch`, (B) se refina con un optimizador independiente sobre la misma verosimilitud y la misma restricción `alpha+beta ≤ 1`, y (C) se evalúa de nuevo en `arch`, que sigue dando el resultado y la inferencia. `refinement_llf_gain` es cuánta log-verosimilitud ganó el paso B sobre el A (≈0 = `arch` ya estaba en el óptimo; grande = no lo estaba): primario (clock_adjusted, normal): 0.00; secundario (raw, normal): 69,662.52; sensibilidad Student-t (clock_adjusted): 11,557.54. `converged_verified` exige que el gradiente proyectado de la log-verosimilitud media (`kkt_max_abs_gradient`, condición de primer orden) sea menor que `GARCH_KKT_GRADIENT_TOL=0.001`. Un ajuste con parámetros finitos pero sin esa verificación no se declara válido: el primario detiene la etapa; un año/segmento queda `NOT_CONVERGED`.

## 6. Persistencia y traducción interpretable (configuración primaria)

`alpha+beta = 1.0000` — **en el tope de la restricción `alpha+beta ≤ 1` que impone `arch`**. Ese valor es el MÁXIMO que el software permite, no una estimación libre: el estimador "querría" ir más allá y la restricción lo detiene (por eso `beta` es exactamente `1 − alpha`). NO es una medida de que la persistencia valga exactamente 1, ni evidencia de IGARCH ni de memoria infinita. Los IC individuales de `alpha` y `beta` (§5) se calculan sin tener en cuenta la restricción activa: son orientativos.

**Vida media de un shock de volatilidad**: NO DEFINIDA (persistencia ≥ 1, ver §6).

**Advertencia metodológica explícita (roadmap)**: una persistencia cercana a 1 **NO demuestra** IGARCH ni memoria infinita — es compatible con (a) un GARCH estacionario genuinamente muy persistente, (b) cambios de nivel/régimen no modelados, o (c) otras formas de persistencia no identificadas aquí. Esta ambigüedad **no se resuelve** en esta etapa; se documenta (ver §17).

**Sensibilidad al tratamiento de los huecos temporales** (auditoría correctiva, H1). El GARCH primario se ajusta sobre los retornos válidos concatenados: en una frontera de bloque (hueco, cambio de sesión o de contrato) trata la última barra anterior como si fuera el minuto previo. Medido: la población tiene 3,096 bloques de continuidad sobre 1,914,530 filas, es decir, el 0.162% de los pares adyacentes cruza una frontera. Se reestima con dos tratamientos alternativos de esa frontera — `reset` (la varianza olvida todo y vuelve a la varianza muestral: el extremo opuesto a concatenar) y `gap_forecast` (la varianza se propaga a través del hueco con el pronóstico del propio GARCH a tantos pasos como minutos de reloj dura el hueco):

| tratamiento | alpha | beta | persistence | rho1_abs_z | Q_abs_z_m | converged_verified |
|---:|---:|---:|---:|---:|---:|---:|
| carry (PRIMARIO: concatenar) | 0.05191 | 0.94809 | 1.00000 | 0.04714 | 5,892.72664 | True |
| reset | 0.06200 | 0.93800 | 1.00000 | 0.04418 | 4,939.72597 | True |
| gap_forecast | 0.05004 | 0.94946 | 0.99950 | 0.04801 | 6,046.86894 | True |

Resumen: `alpha` entre 0.0500 y 0.0620; persistencia entre 0.9995 y 1.0000; `Q(60)` de `|z|` entre 4,939.7 y 6,046.9 (mejor benchmark: 91,023.0). Es una sensibilidad DESCRIPTIVA: mide cuánto dependen los parámetros de la convención, no elige un modelo ni evalúa predicción.

## 7. Diagnóstico de residuos estandarizados (configuración primaria)

`ACF(z)`, `ACF(|z|)`, `ACF(z²)`, portmanteau `Q(m)` (TDA-08, adaptación pairwise — nunca Ljung-Box clásico, ver docstring de `ljung_box_pvalue`) y su p-valor asintótico aproximado (NUNCA usado como criterio de importancia práctica, G5):

| m | Q | estimable | lb_pvalue |
|---:|---:|---:|---:|
| 10 | 5,189.7481 | True | 0.0000 |
| 20 | 5,637.1048 | True | 0.0000 |
| 40 | 5,842.9798 | True | 0.0000 |
| 60 | 5,892.7266 | True | 0.0000 |
| 240 | 6,762.9913 | True | 0.0000 |

`rho_1(|z|)` = 0.0471; `Q(60)` de `|z|` = 5,892.73.

Ver `TDA11_acf_residuos_comparacion.png` para la comparación visual completa (GARCH vs. los 3 benchmarks, ambos bloques).

## 8. Comparación directa contra benchmarks (decisión de utilidad informativa)

**Población de la comparación** (auditoría correctiva, H2). La decisión se toma con GARCH y los 3 benchmarks restringidos a EXACTAMENTE las mismas filas (intersección de `z` finitos: 1,914,350), los mismos bloques de continuidad y, por tanto, los mismos pares en cada rezago. `*_propia` = cada serie sobre su propia población (versión anterior), `*_comun` = población común:

| serie | n_propia | Q_propia | n_comun | pares_lag1_comun | pares_lag60_comun | rho1_comun | Q_comun |
|---:|---:|---:|---:|---:|---:|---:|---:|
| GARCH(1,1) | 1,914,530 | 5,892.7266 | 1,914,350 | 1,911,266 | 1,797,803 | 0.0471 | 5,891.9825 |
| ewma_60 | 1,914,350 | 91,023.0210 | 1,914,350 | 1,911,266 | 1,797,803 | 0.1061 | 91,023.0210 |
| rolling_120 | 1,914,410 | 119,865.4358 | 1,914,350 | 1,911,266 | 1,797,803 | 0.1177 | 119,871.0426 |
| range_ewma_60 | 1,914,350 | 125,287.9095 | 1,914,350 | 1,911,266 | 1,797,803 | 0.1168 | 125,287.9095 |

| | GARCH(1,1) primario | mejor benchmark (clock_adjusted) |
|---|---:|---:|
| `Q(60)` de `\|z\|`, población común | 5,891.98 | 91,023.02 |
| reducción relativa de GARCH | 0.9353 | umbral predeclarado: 0.2 |
| (misma reducción con poblaciones propias) | 0.9353 | — |
| alpha (separación impacto/persistencia) | 0.0519 | umbral predeclarado: 0.01 |

**Reduce dependencia**: True. **Separa impacto de persistencia**: True. **GARCH informativamente útil**: **True**.

Bloque `raw` (sensibilidad secundaria, misma disciplina de población común; no interviene en la decisión):

| serie | n_propia | Q_propia | n_comun | pares_lag60_comun | rho1_comun | Q_comun |
|---:|---:|---:|---:|---:|---:|---:|
| GARCH(1,1) | 1,914,530 | 8,315.5624 | 1,914,350 | 1,797,803 | 0.0493 | 8,307.0094 |
| ewma_60 | 1,914,350 | 441,131.8399 | 1,914,350 | 1,797,803 | 0.1625 | 441,131.8399 |
| rolling_120 | 1,914,410 | 355,873.0435 | 1,914,350 | 1,797,803 | 0.1646 | 355,885.3755 |
| range_ewma_60 | 1,914,350 | 520,079.5455 | 1,914,350 | 1,797,803 | 0.1726 | 520,079.5455 |

**Qué es y qué no es esta comparación.** Es DESCRIPTIVA y DENTRO DE MUESTRA: mide cuánta dependencia queda en `|z|` después de dividir cada retorno por la volatilidad que cada método le asigna. Los parámetros de GARCH se estimaron con toda la muestra (el modelo es retrospectivo, §12), mientras que los benchmarks tienen parámetros fijados de antemano; por construcción GARCH juega con ventaja. De aquí NO se sigue que GARCH pronostique mejor la volatilidad fuera de muestra, ni nada sobre rentabilidad: eso es Nivel 4, fuera de esta fase.

**Sensibilidad post-hoc de auditoría (NO interviene en la regla predeclarada ni en STOP-11).** Los benchmarks y umbrales se fijaron antes de ver resultados y no se tocan. Como pregunta adicional sobre QUÉ aporta GARCH: un EWMA causal con half-life = 13.0 barras (el que corresponde a la `beta` estimada por GARCH) deja `Q(60)` = 6,822.0; frente a él, la reducción de GARCH es 0.1363 (por DEBAJO del 0.2 que la regla exige frente a los benchmarks predeclarados). Casi toda la ventaja de GARCH sobre los benchmarks predeclarados se explica por la VELOCIDAD de memoria que estima, no por una estructura más rica que un EWMA. Este EWMA "igualado" usa un parámetro estimado con toda la muestra, así que es tan retrospectivo como el propio GARCH.

## 9. Estabilidad por año (TH24)

| group | n | estimable | alpha | alpha_ci_lo | alpha_ci_hi | beta | persistence | at_constraint_boundary | converged_verified | rho1_abs_z | Q(60)_|z| | pares_lag60 | reason |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,019 | 7,406 | False | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | — | — | n=7406<20000 (MIN_N_FOR_GARCH_FIT) |
| 2,020 | 342,648 | True | 0.0484 | 0.0452 | 0.0516 | 0.9516 | 1.0000 | True | True | 0.0609 | 1,554.3 | 314,305 |  |
| 2,021 | 349,039 | True | 0.0495 | 0.0459 | 0.0530 | 0.9505 | 1.0000 | True | True | 0.0557 | 1,483.3 | 325,776 |  |
| 2,022 | 349,996 | True | 0.0553 | 0.0487 | 0.0619 | 0.9438 | 0.9991 | False | True | 0.0420 | 978.3 | 331,808 |  |
| 2,023 | 349,636 | True | 0.0549 | 0.0445 | 0.0653 | 0.9446 | 0.9995 | False | True | 0.0321 | 745.6 | 331,543 |  |
| 2,024 | 352,261 | True | 0.0636 | 0.0568 | 0.0704 | 0.9362 | 0.9998 | False | True | 0.0334 | 914.2 | 334,497 |  |
| 2,025 | 163,544 | True | 0.0613 | 0.0535 | 0.0691 | 0.9387 | 1.0000 | True | True | 0.0436 | 438.0 | 155,368 |  |

Sensibilidad al tratamiento de frontera (H1) — `alpha`/persistencia/`Q(60)` con `carry` (primario), `reset` y `gap_forecast`:

| group | n_blocks | alpha | alpha_reset | alpha_gap_forecast | persistence | persistence_reset | persistence_gap_forecast | q_abs_z_m | q_abs_z_m_reset | q_abs_z_m_gap_forecast |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,020 | 987 | 0.0484 | 0.0621 | 0.0467 | 1.0000 | 1.0000 | 1.0000 | 1,554.3244 | 1,399.5395 | 1,574.5300 |
| 2,021 | 719 | 0.0495 | 0.0568 | 0.0473 | 1.0000 | 1.0000 | 0.9990 | 1,483.2736 | 1,323.7270 | 1,537.6709 |
| 2,022 | 386 | 0.0553 | 0.0571 | 0.0523 | 0.9991 | 0.9996 | 0.9983 | 978.2685 | 969.1809 | 999.5045 |
| 2,023 | 372 | 0.0549 | 0.0580 | 0.0521 | 0.9995 | 0.9999 | 0.9979 | 745.5767 | 721.7181 | 727.6553 |
| 2,024 | 346 | 0.0636 | 0.0644 | 0.0606 | 0.9998 | 1.0000 | 0.9980 | 914.1549 | 899.4269 | 875.2599 |
| 2,025 | 148 | 0.0613 | 0.0637 | 0.0594 | 1.0000 | 1.0000 | 0.9993 | 438.0090 | 421.2787 | 437.2731 |

Hechos: 6 de 7 años estimables; 3 de 6 con la persistencia en el tope de la restricción; persistencia entre 0.9991 y 1.0000; `alpha` entre 0.0484 y 0.0636; según el tratamiento de frontera, `alpha` cambia hasta un factor 1.33 (`2020`) y la persistencia nunca baja de 0.9979.

## 10. Estabilidad por segmento (TH24)

| group | n | estimable | alpha | alpha_ci_lo | alpha_ci_hi | beta | persistence | at_constraint_boundary | converged_verified | rho1_abs_z | Q(60)_|z| | pares_lag60 | reason |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00:00-02:00 | 168,922 | True | 0.0582 | 0.0517 | 0.0646 | 0.9418 | 1.0000 | True | True | 0.0574 | 791.0 | 82,683 |  |
| 02:00-03:00 | 84,663 | True | 0.0729 | 0.0646 | 0.0812 | 0.9271 | 1.0000 | True | True | 0.0414 | NOT_ESTIMABLE (rezago máx. estimable=59) | 0 |  |
| 03:00-08:30 | 465,442 | True | 0.0392 | 0.0368 | 0.0416 | 0.9608 | 1.0000 | True | True | 0.0494 | 1,681.7 | 375,869 |  |
| 08:30-09:30 | 84,694 | True | 0.0979 | 0.0569 | 0.1389 | 0.9021 | 1.0000 | True | True | 0.0434 | NOT_ESTIMABLE (rezago máx. estimable=59) | 0 |  |
| 09:30-16:02 | 537,786 | True | 0.0483 | 0.0463 | 0.0503 | 0.9517 | 1.0000 | True | True | 0.0328 | 765.7 | 453,003 |  |
| 16:02-20:00 | 237,735 | True | 0.0933 | 0.0836 | 0.1030 | 0.9067 | 1.0000 | True | True | 0.0505 | 818.7 | 78,631 |  |
| 20:00-24:00 | 335,288 | True | 0.0483 | 0.0443 | 0.0522 | 0.9517 | 1.0000 | True | True | 0.0591 | 1,695.4 | 246,386 |  |

Sensibilidad al tratamiento de frontera (H1):

| group | n_blocks | alpha | alpha_reset | alpha_gap_forecast | persistence | persistence_reset | persistence_gap_forecast | q_abs_z_m | q_abs_z_m_reset | q_abs_z_m_gap_forecast |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00:00-02:00 | 1,671 | 0.0582 | 0.1291 | 0.0458 | 1.0000 | 1.0000 | 1.0000 | 790.9816 | 639.1784 | 1,177.6629 |
| 02:00-03:00 | 1,471 | 0.0729 | 0.1566 | 0.1157 | 1.0000 | 1.0000 | 0.9937 | NaN | NaN | NaN |
| 03:00-08:30 | 1,726 | 0.0392 | 0.0683 | 0.0323 | 1.0000 | 1.0000 | 0.9999 | 1,681.6619 | 882.6821 | 2,206.2519 |
| 08:30-09:30 | 1,438 | 0.0979 | 0.1673 | 0.2277 | 1.0000 | 0.9987 | 0.9938 | NaN | NaN | NaN |
| 09:30-16:02 | 1,446 | 0.0483 | 0.0631 | 0.0416 | 1.0000 | 1.0000 | 0.9996 | 765.6553 | 673.6437 | 940.5950 |
| 16:02-20:00 | 3,338 | 0.0933 | 0.1576 | 0.0717 | 1.0000 | 1.0000 | 0.9999 | 818.7008 | 962.4921 | 1,319.0599 |
| 20:00-24:00 | 1,791 | 0.0483 | 0.0918 | 0.0378 | 1.0000 | 1.0000 | 0.9998 | 1,695.3533 | 1,031.0941 | 2,259.1408 |

Hechos: 7 de 7 segmentos estimables; 7 de 7 con la persistencia en el tope de la restricción; persistencia entre 1.0000 y 1.0000; `alpha` entre 0.0392 y 0.0979; según el tratamiento de frontera, `alpha` cambia hasta un factor 2.82 (`00:00-02:00`) y la persistencia nunca baja de 0.9937.

**Cómo leer estas tablas.** (1) Un ajuste "por segmento" concatena las barras de ese tramo horario a través de los días: cada día aporta una frontera, así que aquí la convención de frontera pesa mucho más que en el ajuste global — por eso el reparto `alpha`/`beta` de un segmento debe leerse junto a su sensibilidad, no como una cifra única. (2) **GARCH estimable no implica `Q(60)` estimable**: `Q(60)` necesita pares separados 60 barras dentro de un mismo bloque de continuidad; en un segmento cuyos bloques duran ≤ 60 barras ese par no existe y el estadístico es `NOT_ESTIMABLE` — no se rellena ni se sustituye por otro. Para esos segmentos el diagnóstico disponible es `rho1_abs_z`. (3) `NOT_CONVERGED` en `reason` significa parámetros finitos sin convergencia verificada: no se reportan. Ver `TDA11_persistencia_estabilidad.png` para el perfil visual.

## 11. Diagnóstico TH25 de asimetría

Comparación descriptiva **antes** de ajustar ningún modelo asimétrico: `|r_t|` condicionado al signo de `r_(t-1)`, controlando la magnitud del shock por deciles de `|r_(t-1)|` (población `r_tilde`, primario):

**Global**: diferencia relativa agrupada (negativo − positivo) = 0.0288 (IC 95% bootstrap bloques por jornada: [0.0236, 0.0326]), `n`=1,829,600.

| decile | n_pos | n_neg | mean_pos | mean_neg | diff_neg_minus_pos | rel_diff |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00000 | 91,486.00000 | 91,474.00000 | 0.00017 | 0.00017 | 0.00000 | 0.00918 |
| 1.00000 | 92,089.00000 | 90,871.00000 | 0.00018 | 0.00018 | 0.00000 | 0.00611 |
| 2.00000 | 92,429.00000 | 90,531.00000 | 0.00018 | 0.00018 | 0.00000 | 0.00791 |
| 3.00000 | 92,462.00000 | 90,498.00000 | 0.00019 | 0.00019 | 0.00000 | 0.02340 |
| 4.00000 | 93,057.00000 | 89,903.00000 | 0.00020 | 0.00020 | 0.00000 | 0.02442 |
| 5.00000 | 93,298.00000 | 89,662.00000 | 0.00021 | 0.00022 | 0.00001 | 0.02877 |
| 6.00000 | 93,324.00000 | 89,636.00000 | 0.00023 | 0.00024 | 0.00001 | 0.03539 |
| 7.00000 | 92,561.00000 | 90,399.00000 | 0.00026 | 0.00027 | 0.00001 | 0.03674 |
| 8.00000 | 92,086.00000 | 90,874.00000 | 0.00031 | 0.00032 | 0.00001 | 0.02820 |
| 9.00000 | 90,419.00000 | 92,541.00000 | 0.00053 | 0.00054 | 0.00001 | 0.02324 |

**Por año** (estabilidad del signo):

| year | n | pooled_rel_diff | pooled_ci_lo | pooled_ci_hi | median_rel_diff | direction |
|---:|---:|---:|---:|---:|---:|---:|
| 2,019 | 5,950 | 0.0138 | -0.0315 | 0.0519 | -0.0062 | POSITIVO_MAS_VOLATIL |
| 2,020 | 325,699 | 0.0308 | 0.0209 | 0.0413 | 0.0280 | NEGATIVO_MAS_VOLATIL |
| 2,021 | 330,755 | 0.0336 | 0.0264 | 0.0415 | 0.0229 | NEGATIVO_MAS_VOLATIL |
| 2,022 | 338,603 | 0.0041 | -0.0041 | 0.0102 | 0.0071 | NEGATIVO_MAS_VOLATIL |
| 2,023 | 332,060 | 0.0223 | 0.0163 | 0.0268 | 0.0200 | NEGATIVO_MAS_VOLATIL |
| 2,024 | 337,412 | 0.0485 | 0.0395 | 0.0588 | 0.0443 | NEGATIVO_MAS_VOLATIL |
| 2,025 | 159,121 | 0.0476 | 0.0310 | 0.0695 | 0.0316 | NEGATIVO_MAS_VOLATIL |

**Decisión TH25**: mediana de `rel_diff` entre deciles = 0.0239 (umbral de materialidad predeclarado: ±0.1); estable en el mismo signo en 0.86 de los años evaluados (umbral predeclarado: 0.7). **Asimetría NO material — resultado negativo válido (G6), NO se ajusta modelo asimétrico**.

## 12. Clasificación CAUSAL/RETROSPECTIVO de cada bloque

- **`r_1m` (RAW)**: serie de entrada CAUSAL de principio a fin.
- **`r_tilde` (CLOCK_ADJUSTED)**: serie de entrada RETROSPECTIVA — `s(m)` (TDA-06) se estimó con TODA la muestra.
- **Cualquier resultado de GARCH (sobre cualquiera de las dos series)**: el MODELO EN SÍ ES RETROSPECTIVO — `omega`/`alpha`/`beta` se estiman por máxima verosimilitud sobre TODA la muestra, igual que `s(m)`. Esto es una distinción ADICIONAL a la de la serie de entrada — incluso el GARCH ajustado sobre `r_1m` (entrada causal) es, como modelo, retrospectivo. Ningún resultado de esta etapa se presenta como disponible causalmente en producción sin reestimación continua.
- **Los 3 benchmarks causales** (`ewma_60`/`rolling_120`/`range_ewma_60`) aplicados sobre `r_1m`: CAUSALES de principio a fin (TDA-10). Aplicados sobre `r_tilde`: heredan el componente RETROSPECTIVO de la propia serie de entrada, pero el filtro en sí no estima ningún parámetro adicional sobre la muestra completa.

**La pregunta FORMAL (TH23/TH24/TH25) la responde el bloque `clock_adjusted`** (controla la estacionalidad intradía, roadmap método mínimo 3). El bloque `raw` es sensibilidad secundaria — nunca se promedia ni se mezcla con el primario (auditoría de TDA-10, problema 3, aplicada aquí de nuevo).

## 13. STOP-11

**`NO ACTIVADO`** — GARCH(1,1) aporta informacion sobre los benchmarks simples: reduccion relativa de Q(60) de |z| = 0.935 (umbral 0.2), alpha=0.0519 (umbral 0.01). STOP-11 NO activado.

Esta decisión usa ÚNICAMENTE la regla predeclarada de §8 (reducción de `Q(60)` frente al mejor de los tres benchmarks predeclarados, y `alpha`). La sensibilidad post-hoc contra un EWMA de memoria igualada (§8, §14) no interviene en ella.

## 14. TH23 — ¿qué estimador de volatilidad es suficiente?

`RESUELTA -- GARCH(1,1) aporta informacion`.

Dos comparaciones DISTINTAS, que no deben mezclarse:

| | (A) Regla predeclarada — DECIDE TH23 y STOP-11 | (B) Sensibilidad post-hoc — NO decide nada |
|---|---|---|
| Contra qué se compara GARCH | el mejor de los 3 benchmarks fijados antes de ver resultados (`('ewma_60', 'rolling_120', 'range_ewma_60')`) | un EWMA cuya memoria se igualó, DESPUÉS de ver el resultado, a la `beta` que estimó GARCH (half-life 13.0 barras) |
| `Q(60)` de `\|z\|` del comparador | 91,023.0 | 6,822.0 |
| `Q(60)` de `\|z\|` de GARCH (misma población) | 5,892.0 | 5,892.0 |
| Reducción relativa de GARCH | **0.9353** (umbral predeclarado 0.2) | 0.1363 (sin umbral: no es una regla) |
| Papel | veredicto formal | interpretación de QUÉ aporta el modelo |

El veredicto de TH23 es el de la columna (A) y no cambia por (B). La columna (B) no es un benchmark predeclarado — se construyó con un parámetro tomado del propio GARCH, así que no puede usarse para "aprobar" ni "suspender" el modelo; solo matiza la lectura: Casi toda la ventaja de GARCH sobre los benchmarks predeclarados se explica por la VELOCIDAD de memoria que estima, no por una estructura más rica que un EWMA. Ambas comparaciones son descriptivas y dentro de muestra (§8).

## 15. TH24 — persistencia y su estabilidad

`RESUELTA -- persistencia muy alta (≈1) en todos los subperiodos; reparto alpha/beta NO robusto al tratamiento de los huecos`.

Persistencia global (primario) = 1.0000 (tope de la restricción `alpha+beta ≤ 1`, no una estimación libre — §6).

**Lo que SÍ es estable y robusto — la persistencia es muy alta.** La persistencia se mantiene en un rango razonablemente estrecho entre subperíodos. Por año: 6 de 7 años estimables; 3 de 6 con la persistencia en el tope de la restricción; persistencia entre 0.9991 y 1.0000; `alpha` entre 0.0484 y 0.0636; según el tratamiento de frontera, `alpha` cambia hasta un factor 1.33 (`2020`) y la persistencia nunca baja de 0.9979. Por segmento: 7 de 7 segmentos estimables; 7 de 7 con la persistencia en el tope de la restricción; persistencia entre 1.0000 y 1.0000; `alpha` entre 0.0392 y 0.0979; según el tratamiento de frontera, `alpha` cambia hasta un factor 2.82 (`00:00-02:00`) y la persistencia nunca baja de 0.9937. Con la persistencia en el tope de la restricción, parte de esa "estabilidad" es mecánica (el tope es el mismo para todos) y la vida media no está definida.

**Lo que NO se presenta como robusto — el reparto entre `alpha` y `beta`.** Los valores de `alpha`/`beta` de §9/§10 corresponden a UNA convención (concatenar a través de los huecos). Al cambiar solo esa convención: por año, en **2 de 6** años el `alpha` de algún tratamiento alternativo cae FUERA del IC 95% del propio ajuste primario (`2020`, `2021`); `alpha` cambia hasta un factor 1.33; por segmento, en **7 de 7** segmentos el `alpha` de algún tratamiento alternativo cae FUERA del IC 95% del propio ajuste primario (`00:00-02:00`, `02:00-03:00`, `03:00-08:30`, `08:30-09:30`, `09:30-16:02`, `16:02-20:00`, `20:00-24:00`); `alpha` cambia hasta un factor 2.82. Por tanto las diferencias de `alpha` ENTRE segmentos no se interpretan como una propiedad establecida de cada tramo horario, y los IC de `alpha`/`beta` por segmento no incluyen esta fuente de variación.

## 16. TH25 — asimetría de respuesta al signo del shock

`RESUELTA (resultado negativo)`. Asimetría NO material — resultado negativo válido (G6), NO se ajusta modelo asimétrico.

## 17. Limitaciones

1. La ambigüedad IGARCH-vs-cambios-de-nivel-vs-memoria-larga (§6) **no se resuelve** en esta etapa — ninguna herramienta aquí distingue entre esas tres explicaciones. Cuando la persistencia queda en el tope de la restricción `alpha+beta ≤ 1`, el `1.0000` reportado es ese tope, no una estimación; los IC de `alpha`/`beta` no tienen en cuenta la restricción activa.
2. GARCH se ajusta sobre retornos válidos CONCATENADOS (§6, §10): en cada frontera de bloque trata como adyacentes dos barras separadas en el reloj. Su impacto está MEDIDO (sensibilidad `reset`/`gap_forecast`): pequeño en el ajuste global y por año, material para el reparto `alpha`/`beta` por segmento. No se sustituyó el procedimiento primario; ninguno de los tres tratamientos es "el correcto" — `gap_forecast` supone que los minutos sin negociación se comportan como minutos negociados.
3. GARCH es, en su conjunto, RETROSPECTIVO (§12) — ningún resultado de esta etapa es un estimador causal disponible sin reestimación continua.
4. La comparación GARCH-vs-benchmarks (§8) es descriptiva/in-sample, con GARCH estimado sobre la misma muestra en la que se evalúa — nunca una competición predictiva fuera de muestra (fuera de alcance de esta fase, Nivel 4).
5. La regla predeclarada de utilidad compara contra tres benchmarks de memoria fija; la sensibilidad post-hoc de §8 muestra cuánto de la ventaja de GARCH se debe solo a la velocidad de memoria. El criterio (b) de la regla (`alpha` ≥ 0.01) no detecta el caso en que GARCH se reduce numéricamente a un EWMA (`alpha+beta`=1 con `omega`≈0); se conserva tal como se predeclaró.
6. Solo se probó GARCH(1,1) y, condicionalmente, GJR-GARCH(1,1,1) — ninguna otra familia (EGARCH, FIGARCH, APARCH) se evaluó, por diseño (G4).
7. La convergencia de cada ajuste se VERIFICA con una condición de primer orden (§5), que acredita un óptimo LOCAL de la verosimilitud restringida; el multi-arranque reduce, pero no elimina matemáticamente, la posibilidad de otro óptimo local mejor.
8. `Q(60)` no es estimable en segmentos cuyos bloques de continuidad duran ≤ 60 barras (§10); en ellos la dependencia residual solo se diagnostica a rezagos cortos.

## 18. Tiempos y configuración de hardware

Workers CPU configurados: **8** (`default_n_workers`, máximo ~20, reservando ~4 núcleos para el sistema — docs/project_hardware.md). GPU: NO utilizada (ningún soporte GPU real disponible en `arch`/`scipy` para esta carga, ni ventaja demostrada — política del proyecto, sección 4).

| Etapa | Tiempo |
|---|---:|
| Poblaciones (r_1m/r_tilde), invariantes y bloques de continuidad | 2.9s |
| Benchmarks causales (EWMA/rodante/rango) x raw+clock_adjusted | 202.7s |
| GARCH(1,1) global -- primario (clock_adjusted), secundario (raw), sensibilidad Student-t | 482.4s |
| GARCH(1,1) por año y por segmento (clock_adjusted, 8 workers) | 123.3s |
| Utilidad informativa de GARCH frente a benchmarks (decision STOP-11) | 223.4s |
| TH25 -- asimetria descriptiva (global + por año) y extension asimetrica condicional | 16.8s |

**Tiempo total de la ejecución (análisis + escritura de CSV/PNG/MD): 1052.0s (~17.5 min).**

## 19. Archivos generados

`TDA11_puerta_de_entrada.csv`, `TDA11_benchmarks_comparativa.csv`, `TDA11_garch_parametros.csv`, `TDA11_garch_diagnosticos_residuos.csv`, `TDA11_asimetria_th25.csv`, `TDA11_utilidad_informativa.csv` (6 CSV) + `TDA11_acf_residuos_comparacion.png`, `TDA11_persistencia_estabilidad.png` (2 PNG) + este informe (MD).

## 20. Modo sencillo

**¿Qué preguntó esta etapa?** Si vale la pena resumir "la memoria de la volatilidad" de MNQ (ya confirmada en TDA-09) con un modelo estadístico (GARCH) en vez de con reglas simples (promedio móvil de volatilidad reciente).

**¿Qué encontró?** Un GARCH(1,1) converge de forma verificada y produce parámetros interpretables: `alpha≈0.052` (cuánto pesa la última sorpresa) y `beta≈0.948` (cuánto persiste la volatilidad ya acumulada), sumando `alpha+beta≈1.000` — que es el tope que permite el programa, no una medida exacta. El modelo SÍ deja menos "memoria sin explicar" en los residuos que las tres reglas simples fijadas de antemano. Matiz importante: casi toda la ventaja de GARCH sobre los benchmarks predeclarados se explica por la VELOCIDAD de memoria que estima, no por una estructura más rica que un EWMA. Dicho de otro modo, lo que GARCH enseña es sobre todo QUÉ TAN RÁPIDO olvida la volatilidad (unos 13 minutos de vida media del suavizado, no los 60 de la regla simple).

**¿Qué NO puede concluir?** Que el reparto exacto entre `alpha` y `beta` de cada tramo horario sea un número fiable — cambia mucho según cómo se traten los saltos entre un día y el siguiente (§15). Tampoco que `alpha+beta` cercano a 1 signifique "memoria infinita" — puede deberse a otras causas (cambios de régimen, por ejemplo) que esta etapa no distingue. Tampoco dice si esto sirve para predecir ni para operar — eso pertenece a otra fase, fuera de esta caracterización.

**¿La rama GARCH queda abierta o cerrada?** ABIERTA -- el modelo aporto informacion, TH23/24/25 documentadas con GARCH como resumen valido de la persistencia.
