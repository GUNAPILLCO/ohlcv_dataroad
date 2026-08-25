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

| bloque | config | n_finito | rho1(|z|) | Q(60)_|z| |
|---:|---:|---:|---:|---:|
| raw | ewma_60 | 1,914,350 | 0.1625 | 441,131.8399 |
| raw | rolling_120 | 1,914,410 | 0.1646 | 355,873.0435 |
| raw | range_ewma_60 | 1,914,350 | 0.1726 | 520,079.5455 |
| clock_adjusted | ewma_60 | 1,914,350 | 0.1061 | 91,023.0210 |
| clock_adjusted | rolling_120 | 1,914,410 | 0.1177 | 119,865.4358 |
| clock_adjusted | range_ewma_60 | 1,914,350 | 0.1168 | 125,287.9095 |

## 4. Especificación exacta del modelo

**Primario**: GARCH(1,1), media CERO (justificado por TDA-08/STOP-8a: dependencia en media despreciable — `0,1425 ticks`), sobre `r_tilde` (RETROSPECTIVO, clock-adjusted). Distribución de innovaciones: Gaussiana (QMLE), inferencia ROBUSTA (`cov_type="robust"`, sandwich Bollerslev-Wooldridge). Escala de ajuste: `r_tilde·10000` (verificado empíricamente necesario para convergencia estable — ver docstring de módulo; `alpha`/`beta`/persistencia son invariantes a esta elección).

**Secundario (sensibilidad)**: idéntica especificación sobre `r_1m` (RAW/CAUSAL), alcance GLOBAL únicamente (G4 — la pregunta formal ya la responde la serie principal).

**Sensibilidad de distribución**: idéntica especificación sobre `r_tilde`, innovaciones Student-t, alcance GLOBAL (justificada por la curtosis residual material encontrada en TDA-10, nunca una búsqueda de la distribución que mejor ajusta).

## 5. Parámetros e intervalos (IC 95% robustos)

| config | bloque | dist | n | omega | alpha | alpha_ci_lo | alpha_ci_hi | beta | beta_ci_lo | beta_ci_hi | persistence | stationary | convergence_flag | converged_cleanly | eps_selected | n_multistart_attempts_finite |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| garch11_primary_global | clock_adjusted | normal | 1,914,530 | 0.00000 | 0.05191 | 0.04949 | 0.05432 | 0.94809 | 0.94575 | 0.95043 | 1.00000 | False | 0 | True | 0.00000 | 4 |
| garch11_secondary_raw_global | raw | normal | 1,914,530 | 0.00000 | 0.21392 | 0.20750 | 0.22034 | 0.78383 | 0.77762 | 0.79004 | 0.99775 | True | 0 | True | 0.00001 | 4 |
| garch11_student_t_sensitivity | clock_adjusted | t | 1,914,530 | 0.00000 | 0.10531 | 0.09699 | 0.11363 | 0.88773 | 0.87825 | 0.89720 | 0.99304 | True | 0 | True | 0.00001 | 4 |

**Nota de validez numérica (auditoría post-primera-ejecución, transparente)**: con `n~10⁶`, se verificó empíricamente que `scipy.optimize.minimize` (SLSQP, interno a `arch`) puede reportar convergencia exitosa (`convergence_flag=0`) sin haberse movido del punto de partida — confirmado comparando los parámetros "ajustados" contra `GARCH.starting_values()` (coincidían exactamente) y un gradiente de cientos de miles en el supuesto óptimo. Causa raíz verificada: con una log-verosimilitud negativa sumada de orden 10⁶-10⁷, el paso de diferencias finitas por defecto de SLSQP queda por debajo del piso de ruido de punto flotante del objetivo. Se corrige con multi-arranque sobre una grilla predeclarada de `eps` (`GARCH_OPTIMIZER_EPS_GRID=(None, 1e-07, 1e-06, 1e-05)`), conservando el intento de **menor log-verosimilitud negativa** — un criterio objetivo (la propia función que el MLE maximiza), nunca una elección subjetiva. La columna `eps_selected` documenta qué intento ganó en cada ajuste; `n_multistart_attempts_finite` cuántos de los 4 intentos produjeron parámetros finitos.

## 6. Persistencia y traducción interpretable (configuración primaria)

`alpha+beta = 1.0000` — NO estacionario en el sentido de varianza (alpha+beta>=1).

**Vida media de un shock de volatilidad**: NO DEFINIDA (persistencia ≥ 1, ver §6).

**Advertencia metodológica explícita (roadmap)**: una persistencia cercana a 1 **NO demuestra** IGARCH ni memoria infinita — es compatible con (a) un GARCH estacionario genuinamente muy persistente, (b) cambios de nivel/régimen no modelados, o (c) otras formas de persistencia no identificadas aquí. Esta ambigüedad **no se resuelve** en esta etapa; se documenta (ver §17).

## 7. Diagnóstico de residuos estandarizados (configuración primaria)

`ACF(z)`, `ACF(|z|)`, `ACF(z²)`, portmanteau `Q(m)` (TDA-08, adaptación pairwise — nunca Ljung-Box clásico, ver docstring de `ljung_box_pvalue`) y su p-valor asintótico aproximado (NUNCA usado como criterio de importancia práctica, G5):

| m | Q | estimable | lb_pvalue |
|---:|---:|---:|---:|
| 10 | 5,189.7106 | True | 0.0000 |
| 20 | 5,637.0745 | True | 0.0000 |
| 40 | 5,842.9501 | True | 0.0000 |
| 60 | 5,892.6958 | True | 0.0000 |
| 240 | 6,762.9716 | True | 0.0000 |

`rho_1(|z|)` = 0.0471; `Q(60)` de `|z|` = 5,892.70.

Ver `TDA11_acf_residuos_comparacion.png` para la comparación visual completa (GARCH vs. los 3 benchmarks, ambos bloques).

## 8. Comparación directa contra benchmarks (decisión de utilidad informativa)

| | GARCH(1,1) primario | mejor benchmark (clock_adjusted) |
|---|---:|---:|
| `Q(60)` de `\|z\|` | 5,892.70 | 91,023.02 |
| reducción relativa de GARCH | 0.9353 | umbral predeclarado: 0.2 |
| alpha (separación impacto/persistencia) | 0.0519 | umbral predeclarado: 0.01 |

**Reduce dependencia**: True. **Separa impacto de persistencia**: True. **GARCH informativamente útil**: **True**.

## 9. Estabilidad por año (TH24)

| group | n | estimable | alpha | alpha_ci_lo | alpha_ci_hi | beta | beta_ci_lo | beta_ci_hi | persistence | rho1_abs_z | convergence_flag |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,019 | 7,406 | False | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |
| 2,020 | 342,648 | True | 0.0484 | 0.0452 | 0.0516 | 0.9516 | 0.9483 | 0.9549 | 1.0000 | 0.0609 | 0.0000 |
| 2,021 | 349,039 | True | 0.0495 | 0.0459 | 0.0531 | 0.9505 | 0.9471 | 0.9540 | 1.0000 | 0.0557 | 0.0000 |
| 2,022 | 349,996 | True | 0.0553 | 0.0487 | 0.0619 | 0.9438 | 0.9375 | 0.9502 | 0.9991 | 0.0420 | 0.0000 |
| 2,023 | 349,636 | True | 0.0549 | 0.0445 | 0.0653 | 0.9446 | 0.9351 | 0.9541 | 0.9995 | 0.0321 | 0.0000 |
| 2,024 | 352,261 | True | 0.0636 | 0.0568 | 0.0704 | 0.9362 | 0.9299 | 0.9426 | 0.9998 | 0.0334 | 0.0000 |
| 2,025 | 163,544 | True | 0.0613 | 0.0535 | 0.0691 | 0.9387 | 0.9316 | 0.9458 | 1.0000 | 0.0436 | 0.0000 |

## 10. Estabilidad por segmento (TH24)

| group | n | estimable | alpha | alpha_ci_lo | alpha_ci_hi | beta | beta_ci_lo | beta_ci_hi | persistence | rho1_abs_z | convergence_flag |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00:00-02:00 | 168,922 | True | 0.0581 | 0.0516 | 0.0645 | 0.9419 | 0.9355 | 0.9484 | 1.0000 | 0.0575 | 0 |
| 02:00-03:00 | 84,663 | True | 0.0729 | 0.0646 | 0.0812 | 0.9271 | 0.9194 | 0.9347 | 1.0000 | 0.0414 | 0 |
| 03:00-08:30 | 465,442 | True | 0.0392 | 0.0368 | 0.0416 | 0.9608 | 0.9584 | 0.9632 | 1.0000 | 0.0494 | 0 |
| 08:30-09:30 | 84,694 | True | 0.0980 | 0.0569 | 0.1390 | 0.9020 | 0.8412 | 0.9628 | 1.0000 | 0.0433 | 0 |
| 09:30-16:02 | 537,786 | True | 0.0483 | 0.0463 | 0.0503 | 0.9517 | 0.9498 | 0.9536 | 1.0000 | 0.0328 | 0 |
| 16:02-20:00 | 237,735 | True | 0.0933 | 0.0836 | 0.1030 | 0.9067 | 0.8969 | 0.9164 | 1.0000 | 0.0505 | 0 |
| 20:00-24:00 | 335,288 | True | 0.0483 | 0.0443 | 0.0523 | 0.9517 | 0.9478 | 0.9557 | 1.0000 | 0.0591 | 0 |

Convención (documentada en el módulo, punto 4): un ajuste "por segmento" concatena las barras de ese segmento horario a través de los días en orden cronológico — misma simplificación, ya predeclarada por TDA-10 para sus propios estimadores causales, no una decisión nueva de esta etapa. Ver `TDA11_persistencia_estabilidad.png` para el perfil visual.

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

## 14. TH23 — ¿qué estimador de volatilidad es suficiente?

`RESUELTA -- GARCH(1,1) aporta informacion`. Ver §8 para la comparación cuantitativa completa.

## 15. TH24 — persistencia y su estabilidad

`RESUELTA`. Persistencia global (primario) = 1.0000. Ver §9/§10 para la variación por año/segmento — la persistencia se mantiene en un rango razonablemente estrecho entre subperíodos.

## 16. TH25 — asimetría de respuesta al signo del shock

`RESUELTA (resultado negativo)`. Asimetría NO material — resultado negativo válido (G6), NO se ajusta modelo asimétrico.

## 17. Limitaciones

1. La ambigüedad IGARCH-vs-cambios-de-nivel-vs-memoria-larga (§6) **no se resuelve** en esta etapa — ninguna herramienta aquí distingue entre esas tres explicaciones.
2. El ajuste "por segmento" (§10) concatena barras de días distintos como si fueran adyacentes — simplificación predeclarada, heredada de TDA-10, no una limitación nueva pero sí real.
3. GARCH es, en su conjunto, RETROSPECTIVO (§12) — ningún resultado de esta etapa es un estimador causal disponible sin reestimación continua.
4. La comparación GARCH-vs-benchmarks (§8) es descriptiva/in-sample — nunca una competición predictiva fuera de muestra (fuera de alcance de esta fase, Nivel 4).
5. Solo se probó GARCH(1,1) y, condicionalmente, GJR-GARCH(1,1,1) — ninguna otra familia (EGARCH, FIGARCH, APARCH) se evaluó, por diseño (G4).
6. El optimizador subyacente (`scipy`/SLSQP, vía `arch`) mostró fragilidad numérica genuina a esta escala de muestra (§5, nota de validez numérica) — mitigada con multi-arranque, pero no se puede garantizar matemáticamente que el óptimo global de la verosimilitud se alcanzó en cada ajuste; se reporta el mejor resultado encontrado entre los intentos de la grilla, con el criterio objetivo de la propia verosimilitud.

## 18. Tiempos y configuración de hardware

Workers CPU configurados: **8** (`default_n_workers`, máximo ~20, reservando ~4 núcleos para el sistema — docs/project_hardware.md). GPU: NO utilizada (ningún soporte GPU real disponible en `arch`/`scipy` para esta carga, ni ventaja demostrada — política del proyecto, sección 4).

| Etapa | Tiempo |
|---|---:|
| Poblaciones (r_1m/r_tilde), invariantes y bloques de continuidad | 2.4s |
| Benchmarks causales (EWMA/rodante/rango) x raw+clock_adjusted | 147.3s |
| GARCH(1,1) global -- primario (clock_adjusted), secundario (raw), sensibilidad Student-t | 149.4s |
| GARCH(1,1) por año y por segmento (clock_adjusted, 8 workers) | 29.3s |
| Utilidad informativa de GARCH frente a benchmarks (decision STOP-11) | 0.0s |
| TH25 -- asimetria descriptiva (global + por año) y extension asimetrica condicional | 16.1s |

**Tiempo total de la ejecución (análisis + escritura de CSV/PNG/MD): 345.0s (~5.8 min).**

## 19. Archivos generados

`TDA11_puerta_de_entrada.csv`, `TDA11_benchmarks_comparativa.csv`, `TDA11_garch_parametros.csv`, `TDA11_garch_diagnosticos_residuos.csv`, `TDA11_asimetria_th25.csv`, `TDA11_utilidad_informativa.csv` (6 CSV) + `TDA11_acf_residuos_comparacion.png`, `TDA11_persistencia_estabilidad.png` (2 PNG) + este informe (MD).

## 20. Modo sencillo

**¿Qué preguntó esta etapa?** Si vale la pena resumir "la memoria de la volatilidad" de MNQ (ya confirmada en TDA-09) con un modelo estadístico (GARCH) en vez de con reglas simples (promedio móvil de volatilidad reciente).

**¿Qué encontró?** Un GARCH(1,1) sí converge y produce parámetros interpretables: `alpha≈0.052` (cuánto pesa la última sorpresa) y `beta≈0.948` (cuánto persiste la volatilidad ya acumulada), sumando `alpha+beta≈1.000`. El modelo SÍ deja menos "memoria sin explicar" en los residuos que las reglas simples.

**¿Qué NO puede concluir?** Que `alpha+beta` cercano a 1 signifique "memoria infinita" — puede deberse a otras causas (cambios de régimen, por ejemplo) que esta etapa no distingue. Tampoco dice si esto sirve para predecir ni para operar — eso pertenece a otra fase, fuera de esta caracterización.

**¿La rama GARCH queda abierta o cerrada?** ABIERTA -- el modelo aporto informacion, TH23/24/25 documentadas con GARCH como resumen valido de la persistencia.
