# Complemento TH10 — Escalado de la varianza con el horizonte (pre-TDA-08)

**Hipótesis:** `docs/methodology/Tsay_empirical_hypotheses_backlog.md` § TH10
**Diferida desde:** TDA-04 (`reports/mnq/TDA04_variables_analisis.md`, §8 — decisión explícita de secuenciación, no una limitación técnica). Recomendado retomarla "en un punto conveniente antes de TDA-08/TDA-09" por TDA-04, TDA-05, TDA-06 y TDA-07.
**Alcance de datos:** exclusivamente `tda04_variables_1m.parquet` y `tda04_return_validity_mask.parquet`. Ningún archivo de `data/raw/` ni de `holdout_files` fue abierto. No se recalculó la serie canónica.
**Evidencia reproducible:** `reports/mnq/TH10_var_by_horizon.csv`, `TH10_var_by_horizon_no_solapado.csv`, `TH10_beta_por_anio.csv`, `TH10_var_h_loglog.png`. Todo generado por `python -m ohlcv_dataroad.ingest.run_th10`.

> Este es un complemento PUNTUAL, no una etapa TDA nueva. NO calcula ACF/PACF, NO ajusta AR/MA/ARMA (eso es TDA-08), NO estudia volatility clustering ni ajusta GARCH (TDA-09), NO ejecuta EVT, NO crea features ni targets, NO usa `r_tilde` como retorno multi-horizonte, y NO modifica ningún artefacto de TDA-00…TDA-07.

---

## 1. Definición exacta de `r[h]`

`r_t[h] = ln(C_t / C_{t-h})`, formulación causal hacia atrás terminando en `t` — coherente con la convención `CAUSAL_AT_BAR_CLOSE` de TDA-04 (`r_1m` es la instancia `h=1` de esta misma familia). Matemáticamente idéntica a la suma de los `h` log-retornos de 1 minuto correspondientes cuando existe continuidad completa (identidad telescópica, verificada explícitamente — ver §9, test 1).

**Invariante central** (sin excepciones): `r_t[h]` solo se acepta si existen `h` retornos de 1 minuto **consecutivos y válidos** que formen una cadena temporal continua terminando en `t`. No basta con comprobar los dos extremos.

**Implementación**: se reutiliza `compute_hac_block_ids` de `tda07_marginal_distribution.py` (sin modificarlo) para identificar bloques de continuidad genuina sobre la población `r_1m_valid=True` — dos filas consecutivas quedan en el mismo bloque si y solo si están separadas por exactamente 60 segundos **y** comparten `trading_date` (la misma condición que TDA-04 exige para `r_1m_valid=True`). Sobre esos bloques se calcula `run_length`: la posición (1-indexada) de cada fila dentro de su bloque, es decir, el número de eslabones de 1 minuto válidos y encadenados que terminan ahí. `r[h]` en la fila `t` se acepta si y solo si `run_length[t] >= h`.

Esto excluye automáticamente, sin comprobaciones separadas: cambio de `trading_date`, `NON_CONSECUTIVE_MINUTE`, cualquier fila con `r_1m_valid=False`, y cualquier hueco. El **roll** (`ROLL_BOUNDARY`) y el cambio de `segment_id` quedan cubiertos por la misma condición de `trading_date`: TDA-04 certificó (informe, §5) que todo roll coincide exactamente con una frontera de jornada y que `segment_id`/`contract` cambian exactamente donde `is_roll_boundary=True` — no hace falta una comprobación de roll independiente.

Para `h=1`, esta construcción reproduce **exactamente** la población y los valores de `r_1m_valid=True` de TDA-04 (verificado — §9, test 6).

## 2. Horizontes utilizados

`h = [1, 2, 5, 10, 15, 30, 60]` minutos — grilla diagnóstica predeclarada (backlog, método mínimo de TH10), fijada antes de calcular cualquier resultado. No se modificó tras ver los datos.

## 3. Reglas de validez / no-cruce

Ver §1. Reutilizadas de TDA-04 (vía `r_1m_valid`/`trading_date`) sin recalcular la serie canónica ni redefinir ninguna regla. `verify_timestamp_alignment` (TDA-07) se ejecuta antes de construir cualquier `r[h]`, fail-fast si `tda04_variables_1m.parquet` y `tda04_return_validity_mask.parquet` no están perfectamente alineados.

## 4. `n` válido por horizonte

| h | n (overlapping) | n (no solapado) |
|---:|---:|---:|
| 1 | 1.914.530 | 1.914.530 |
| 2 | 1.911.434 | 956.192 |
| 5 | 1.903.157 | 381.031 |
| 10 | 1.890.880 | 189.581 |
| 15 | 1.879.471 | 125.593 |
| 30 | 1.847.740 | 61.971 |
| 60 | 1.799.414 | 30.386 |

La caída de `n` con `h` es pequeña en el caso *overlapping* (99,8%→94,0% de la población de `h=1`) — consistente con que las discontinuidades son raras (TDA-04: 0,18% de filas excluidas) y con que solo se pierden las primeras `h-1` observaciones de cada bloque de continuidad. En el caso *no solapado* la caída es, por diseño, aproximadamente proporcional a `1/h` (cada bloque aporta ~`L/h` observaciones en vez de `L-h+1`).

## 5. Var(r[h])

Convención: varianza muestral con `ddof=1` (misma convención que `compute_moments_quantiles`/`summarize_resolution`, TDA-05/07).

| h | Var(r[h]) |
|---:|---:|
| 1 | 1,757×10⁻⁷ |
| 2 | 3,530×10⁻⁷ |
| 5 | 8,746×10⁻⁷ |
| 10 | 1,735×10⁻⁶ |
| 15 | 2,596×10⁻⁶ |
| 30 | 5,195×10⁻⁶ |
| 60 | 1,039×10⁻⁵ |

## 6. VR(h) = Var(r[h]) / (h · Var(r[1]))

| h | VR(h) |
|---:|---:|
| 1 | 1,0000 |
| 2 | 1,0045 |
| 5 | 0,9956 |
| 10 | 0,9874 |
| 15 | 0,9849 |
| 30 | 0,9857 |
| 60 | 0,9854 |

`VR(h)` se mantiene extremadamente cerca de 1 en todo el rango (0,985 a 1,004) — una desviación máxima de ~1,5% respecto del escalado lineal puro, y sin una tendencia monótona clara más allá de `h=2` (el patrón se estabiliza alrededor de 0,985 para `h≥10`, no sigue cayendo). Esta es la ayuda interpretativa; el resultado formal es `beta` (§7).

## 7. Pendiente `beta` GLOBAL + intervalo bootstrap

**`beta = 0,9950`** (`alpha = -15,5522`). Intervalo 95% por bootstrap de bloques de jornada (300 remuestreos, semilla fija=0, jornadas completas remuestreadas con reemplazo, `Var(r[h])` y `beta` recalculados en cada repetición): **[0,9869, 1,0019]**.

`beta − 1 = −0,0050`. El intervalo bootstrap **incluye 1** — no hay evidencia de una desviación material del escalado lineal. Ver `TH10_var_h_loglog.png`: los 7 puntos caen prácticamente sobre la recta de referencia `beta=1`.

## 8. Sensibilidad no solapada

`beta (no solapado) = 0,9974`, prácticamente idéntico al de la versión *overlapping* (0,9950). **La conclusión sobre `beta ≈ 1` NO depende materialmente de usar ventanas solapadas** — la advertencia del backlog ("ventanas solapadas inflan la aparente precisión") se refiere a la precisión/incertidumbre reportada, no a la estimación puntual de `beta`; aquí ambas convenciones coinciden en el mismo diagnóstico. (Convención de ventanas no solapadas, predeclarada: dentro de cada bloque de continuidad, se seleccionan las posiciones `k = h, 2h, 3h, …` contadas desde el inicio del bloque — el origen se reinicia en cada bloque nuevo, nunca cruza jornadas/huecos, ver `non_overlap_mask`.)

## 9. `beta` por año y estabilidad temporal

| Año | Completo | n(h=1) | n(h=60) | beta |
|---|---|---:|---:|---:|
| 2019 | NO (parcial) | 7.406 | 4.566 | 1,1696 |
| 2020 | SÍ | 342.648 | 314.606 | 0,9792 |
| 2021 | SÍ | 349.039 | 326.064 | 1,0027 |
| 2022 | SÍ | 349.996 | 332.084 | 1,0089 |
| 2023 | SÍ | 349.636 | 331.820 | 0,9906 |
| 2024 | SÍ | 352.261 | 334.775 | 1,0064 |
| 2025 | NO (parcial) | 163.544 | 155.499 | 0,9958 |

En los 5 años **completos** (2020-2024), `beta` oscila en un rango angosto (0,979 a 1,009) — la dirección es estable: **ningún año completo se aleja de forma material de `beta≈1`**, ni hacia sublineal ni hacia superlineal (diferencias del orden de ±1%, no se interpretan como "régimen"). 2019 es parcial y tiene mucha menos muestra (`n` pequeño incluso en `h=1`, y más pequeño aún en `h=60`) que los años completos — su beta (1,17) es, por eso, menos comparable con la de 2020-2024, y no se interpreta como evidencia de un régimen distinto (mismo criterio de cautela que TDA-05/06/07 ya aplicaron a 2019). TH10 no identifica causalmente el origen de esa diferencia — solo constata que 2019 es la observación menos comparable del conjunto, consistente con su tamaño de muestra reducido, sin afirmar que esa sea la causa demostrada. 2025 (parcial) está en línea con el resto (0,996).

## 10. Interpretación

**`beta ≈ 1`** (0,9950 global, IC que incluye 1; 0,9974 en la sensibilidad no solapada; estable en 0,979-1,009 en los 5 años completos): **compatible con escalado aproximadamente lineal de la varianza / ausencia de desviación material acumulativa en el rango de horizontes estudiado (1 a 60 minutos)**.

Esto **NO** se interpreta como "los retornos de MNQ son independientes" — es un diagnóstico agregado sobre la varianza, no una prueba de independencia (TDA-08 estudiará la dependencia lineal directamente, vía ACF/PACF). Tampoco implica ausencia de estructura en magnitud (eso es TDA-09, aún no ejecutado) ni de no linealidad. Es, exactamente como advierte el backlog, "un diagnóstico barato que anticipa" el estudio formal — nada más.

## 11. Archivos creados/modificados

**Código nuevo:**
- `src/ohlcv_dataroad/ingest/th10_horizon_scaling.py`
- `src/ohlcv_dataroad/ingest/run_th10.py`
- `tests/test_th10_horizon_scaling.py` (21 tests)

**Código modificado:**
- `src/ohlcv_dataroad/config.py` (campos y propiedades `th10_*`, sección añadida sin tocar las existentes)
- `configs/mnq_snapshot.yaml` (sección `th10`, añadida)

**Artefactos generados:**
- `reports/mnq/TH10_escalado_varianza_horizonte.md` (este informe)
- `reports/mnq/TH10_var_by_horizon.csv`
- `reports/mnq/TH10_var_by_horizon_no_solapado.csv`
- `reports/mnq/TH10_beta_por_anio.csv`
- `reports/mnq/TH10_var_h_loglog.png`

**No se modificó ningún artefacto de TDA-00…TDA-07.** No se editó `docs/methodology/Tsay_empirical_hypotheses_backlog.md`: ese documento es una referencia metodológica estática (igual que para TH14/TH15 en TDA-06 y TH11/TH12/TH13 en TDA-07, ninguno de los cuales modificó el backlog al resolverse) — el estado vivo de cada hipótesis se registra en el informe de la etapa/complemento que la resuelve, no en el backlog. `reports/mnq/TDA04_variables_analisis.md` tampoco se modificó: sigue registrando correctamente que TH10 quedó **diferida** en ese momento, por instrucción explícita de esa tarea — este informe es el que documenta que la deuda quedó **resuelta** posteriormente, preservando la trazabilidad histórica.

## 12. Tests ejecutados y total de la suite

```
python -m pytest -q
```

| Archivo | Tests |
|---|---:|
| Suite previa (TDA-00…07 + correcciones HAC) | 276 |
| `test_th10_horizon_scaling.py` | **21** |
| **Total final** | **297** |

**Resultado: `297 passed`.**

Cobertura de los 12 puntos mínimos exigidos: (1) `r[h]` coincide con la suma de `h` retornos de 1 minuto consecutivos, verificado contra el cálculo directo por ratio de precios; (2)/(5) `r[h]` es `NaN` si falta cualquier minuto intermedio (hueco/`NON_CONSECUTIVE_MINUTE`), incluso cuando ambos "extremos" tienen precio válido por separado; (3) nunca cruza `trading_date` (ni siquiera con delta=60s exacto); (4) nunca cruza lo que sería un roll/`segment_id` (mismo mecanismo que `trading_date`, documentado); (6) `h=1` reproduce exactamente la población y los valores de `r_1m_valid=True` de TDA-04, incluida una comparación DIRECTA fila por fila contra una columna `r_1m` explícita en `variables` (tolerancia `1e-12`); (7) conteo exacto de ventanas válidas sobre una cadena sintética de longitud conocida (`run_length` 1..9 → conteos `9,7,5,1,0` para `h=1,3,5,9,10`); (8) bootstrap reproducible con la misma semilla (bit a bit); (9) `beta≈1` (`|beta-1|<0,15`) sobre 8.000 retornos sintéticos i.i.d.; (10) `beta<0,9` con dependencia AR(1) negativa (`phi=-0,35`); (11) `beta>1,1` con dependencia AR(1) positiva (`phi=+0,35`); (12) la selección no solapada nunca produce un `r[h]` `NaN` (nunca cruza bloque). Ningún test reproduce números hardcodeados del conjunto de investigación real.

## 13. Problemas encontrados

Ninguno. Ambas invariantes reutilizadas de TDA-07 (alineación de timestamps, bloques de continuidad) se cumplieron sin contradicciones sobre el conjunto de investigación real. No fue necesario detenerse ni consultar.

## 14. Estado final de TH10

**RESUELTA.**

`beta` global (0,9950, IC 95% [0,9869, 1,0019]) es compatible con escalado lineal de la varianza en el rango `h∈[1,60]` minutos, sin desviación material. El resultado es robusto a la elección de ventanas solapadas vs. no solapadas (0,9950 vs. 0,9974) y estable en dirección entre los 5 años completos (0,979-1,009) — 2019 (parcial, `n` pequeño) es la única excepción notable en magnitud, pero no se interpreta como un régimen distinto: es la observación menos comparable del conjunto, dado su tamaño de muestra muy inferior al de los años completos.

## 15. Recomendación sobre TDA-08

**Sí, ya se puede iniciar TDA-08.** No hay ninguna contradicción ni deuda metodológica pendiente que lo bloquee: TH08 (TDA-07), TH10 (este complemento) y TH11-TH13 (TDA-07) están resueltas; TH14/TH15 (TDA-06) están resueltas. El resultado de TH10 (`beta≈1`, sin desviación acumulativa material) no implica que la ACF de `r_1m` vaya a ser plana — es exactamente la pregunta que TDA-08 debe responder de forma directa y granular (por rezago, no agregada por horizonte); TDA-08 debe tratar este resultado como contexto (un diagnóstico agregado compatible con dependencia débil o nula en el rango 1-60 minutos), nunca como sustituto del análisis formal de ACF/PACF que le corresponde.
