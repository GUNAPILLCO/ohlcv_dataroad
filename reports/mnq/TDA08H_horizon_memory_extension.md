# TDA08-H — Horizon Memory Extension

**Relación con TDA-08:** complemento ACOTADO. TDA-08 permanece **CERRADA** (`PASS_WITH_OPEN_QUESTIONS` / `CLOSED`, ver `reports/mnq/TDA08_dependencia_lineal_media.md`) — este informe NO la modifica, NO cambia TH16/TH17/TH18, y no reabre ninguna de sus decisiones.
**Alcance de datos:** exclusivamente `tda04_variables_1m.parquet` / `tda04_return_validity_mask.parquet` (TDA-04). Ningún archivo de `data/raw/` ni de `holdout_files` fue abierto — holdout `LOCKED`.
**Evidencia reproducible:** `TDA08H_rho1_multi_horizon.csv`, `TDA08H_rho1_multi_horizon.png`. Generados por `python -m ohlcv_dataroad.ingest.run_tda08h`. **No se sobrescribió** `TDA08_dependencia_lineal_media.md` ni `TDA08_rho1_multi_frecuencia.csv`.

---

## 1. Qué se analizó

Una pregunta que TDA-08 dejó sin medir directamente: **¿existe dependencia lineal entre retornos consecutivos NO SOLAPADOS de 30 y 60 minutos en MNQ?** — distinta de dos cosas ya calculadas en otras etapas: (a) la ACF de `r_1m` a los rezagos 30/60 (TDA-08 §7: correlación entre barras de 1 minuto separadas por 30/60 minutos, no entre bloques acumulados); (b) el escalado de varianza `Var(r[h])~h·Var(r[1])` de TH10 (mide cómo crece la varianza, nunca si un bloque de h minutos predice al siguiente).

Horizontes analizados: **1, 5, 10, 30, 60 minutos**. Los tres primeros se recalculan como **control de regresión** frente a TDA-08 CLOSED (deben reproducirla exactamente); 30 y 60 son el objetivo nuevo.

**Por qué no 15 minutos** (presente en la grilla de TH10, `HORIZONS=(1,2,5,10,15,30,60)`): no hay una razón metodológica que lo haga indispensable como punto intermedio entre 10 y 30 para responder ESTA pregunta — se decidió no añadirlo silenciosamente. Como se ve en §5, el patrón entre 10 y 30 minutos no resulta ambiguo (10→30 es la caída más pronunciada de toda la serie, hacia un valor prácticamente nulo) — no emergió una necesidad real de un punto intermedio. Queda como candidato natural para una extensión futura, no de esta tarea.

## 2. Definición exacta de los retornos 30/60

`r_t[h] = ln(C_t / C_{t-h})`, **reutilizando literalmente** `build_horizon_returns` de TH10 (`th10_horizon_scaling.py`) — sin redefinir ni un detalle. Invariante de CADENA COMPLETA: `r[h]` en la fila `t` solo existe si hay `h` retornos de 1 minuto consecutivos y válidos que forman una cadena continua terminando en `t` (`run_length[t] >= h`) — nunca se comprueban solo los dos extremos. Esto hereda automáticamente todos los invariantes ya auditados de TDA-04/TH10: nunca cruza `trading_date`, gaps ni roll (TDA-04 certificó que todo roll coincide con un cambio de `trading_date`).

## 3. Convención de no-solapamiento

**Reutilizada literalmente** de TH10 (`non_overlap_mask`): dentro de cada bloque de continuidad, se seleccionan las posiciones `run_length % h == 0` — el origen se reinicia en cada bloque nuevo, nunca cruza jornadas/huecos. No se introdujo ningún offset/ancla alternativo ni búsqueda del "mejor" origen (§13 de la tarea) — se usó exactamente la convención ya validada y predeclarada en TH10, sin cuestionarla.

**Verificación mecánica de no-cruce** (nueva, específica de TDA08-H): un par lag-1 de horizonte `h` solo se acepta si el delta temporal REAL entre dos ventanas no solapadas consecutivas es exactamente `h·60` segundos (`compute_block_ids` de TDA-08, reutilizado sin modificar) — esto excluye automáticamente cualquier par que cruce una frontera de bloque, incluida la frontera entre el último bloque no solapado de una jornada y el primero de la siguiente. Confirmado con una prueba aritmética exacta en tests (`test_lag1_pairs_never_cross_the_boundary_between_two_separate_blocks`): con dos bloques sintéticos largos y separados, `n_pairs_lag1` es EXACTAMENTE `2·(⌊n/h⌋−1)` — ni un par de más.

## 4. Número de observaciones por horizonte

| Horizonte | `n` (ventanas válidas) | `n_trading_dates` | `n_pairs_lag1` |
|---:|---:|---:|---:|
| 1 min | 1.914.530 | 1.420 | 1.911.434 |
| 5 min | 381.031 | 1.420 | 378.475 |
| 10 min | 189.581 | 1.420 | 187.249 |
| 30 min | 61.971 | 1.418 | 60.282 |
| 60 min | 30.386 | 1.418 | 28.815 |

Pérdida de muestra esperada y NO tratada como error (§12 de la tarea): al exigir ventanas no solapadas, `n` cae monótonamente de ~1,9M (1 min) a ~30k (60 min) — un factor ~63×. `n_trading_dates` apenas baja de 1.420 a 1.418 (2 jornadas demasiado cortas para producir ni una sola ventana de 30/60 min completa) — la base del bootstrap de bloques de jornada sigue siendo amplia en ambos horizontes largos. `n_pairs_lag1 ≤ n` en los 5 horizontes (verificado). El tamaño muestral a 60 minutos (n=30.386, n_pairs=28.815, 1.418 jornadas) es **más que suficiente** para un bootstrap de bloques de jornada confiable — no se declara "muestra insuficiente" en ningún horizonte.

## 5. Tabla completa 1/5/10/30/60

| Horizonte | `n` | `n_pairs_lag1` | `rho_1` | `beta_1` | IC 95% `rho_1` | IC 95% `beta_1` | Signo | IC excluye 0 |
|---:|---:|---:|---:|---:|---|---|---|---|
| 1 min | 1.914.530 | 1.911.434 | 0,005921 | 0,005915 | [−0,000284, 0,012177] | [−0,000283, 0,012174] | + | No |
| 5 min | 381.031 | 378.475 | −0,011060 | −0,011042 | [−0,020217, −0,001777] | [−0,020144, −0,001776] | − | **Sí** |
| 10 min | 189.581 | 187.249 | 0,011494 | 0,011435 | [−0,001726, 0,024707] | [−0,001705, 0,024554] | + | No |
| **30 min** | **61.971** | **60.282** | **0,001043** | **0,001059** | **[−0,019535, 0,022316]** | **[−0,019786, 0,022474]** | + | **No** |
| **60 min** | **30.386** | **28.815** | **−0,006400** | **−0,006617** | **[−0,032514, 0,020402]** | **[−0,033730, 0,021039]** | − | **No** |

`rho_1`/`beta_1` calculados con la definición FINAL de TDA-08 (Pearson pairwise-complete con centrado local, `beta` = pendiente OLS con intercepto sobre exactamente los mismos pares — verificado por fórmula a mano en tests, no solo por circularidad contra `compute_acf`). Bootstrap: bloques de `trading_date`, clave compuesta, **n_boot=300, seed=0** (mismo valor vigente de TDA-08, sin cambio).

**Control de regresión**: los valores de 1/5/10 recalculados aquí coinciden EXACTAMENTE (dentro de 1e-9 para h=1, que se comparó con precisión completa; dentro de 5e-6 para h=5/10, limitados por los 6 decimales publicados en el informe de TDA-08) con los valores congelados de TDA-08 CLOSED — verificado automáticamente por el runner (`_check_regression_against_frozen_tda08`, que aborta ANTES de interpretar 30/60 si hay discrepancia). **No hubo ninguna discrepancia.**

## 6. Interpretación de 30 minutos

`rho_1 = 0,001043` — el valor **más pequeño en magnitud absoluta de los cinco horizontes** (menor incluso que el de 1 minuto). El IC 95% bootstrap ([−0,0195, 0,0223]) incluye ampliamente cero y es, en términos absolutos, ~20× más ancho que el IC de 1 minuto — reflejo directo de la caída de muestra (§4). **No hay evidencia de dependencia lineal a 30 minutos**: ni el punto estimado es distinguible de un valor prácticamente nulo, ni el intervalo excluye cero.

## 7. Interpretación de 60 minutos

`rho_1 = −0,006400` — signo negativo, magnitud pequeña (del mismo orden que 1 minuto, menor que 5/10 minutos). El IC 95% bootstrap ([−0,0325, 0,0204]) incluye cero cómodamente. **Tampoco hay evidencia de dependencia lineal a 60 minutos.** El punto estimado se aleja ligeramente de cero respecto a 30 minutos, pero el intervalo es también más ancho (menos jornadas efectivas, menos pares) — el cambio de signo entre 30 y 60 minutos, con ambos IC incluyendo cero holgadamente, es compatible con **ruido de muestreo alrededor de cero**, no con un patrón real que se esté revirtiendo.

## 8. Patrón completo 1→5→10→30→60

| Transición | `rho_1` | Signo | `|rho_1|` |
|---|---:|---|---:|
| 1 min | 0,005921 | + | 0,0059 |
| 5 min | −0,011060 | − | 0,0111 |
| 10 min | 0,011494 | + | 0,0115 |
| 30 min | 0,001043 | + | **0,0010 (mínimo)** |
| 60 min | −0,006400 | − | 0,0064 |

**No hay una evolución ordenada** (pregunta D/E de la tarea): el signo alterna en las cuatro transiciones (+,−,+,+,−) y la magnitud NO decae monótonamente — 5 y 10 minutos tienen magnitud *mayor* que 1 minuto, antes de caer bruscamente a su mínimo en 30 minutos y repuntar levemente (aún pequeño) en 60. Esto es consistente con la observación que TDA-08 ya había hecho para 1/5/10 ("no hay un patrón de dilución limpio", §8.i de ese informe) — TDA08-H **extiende esa misma conclusión** a 30/60: la dependencia, en la medida en que exista, **no se diluye de forma ordenada con el horizonte**; simplemente **nunca alcanza una magnitud material en ningún horizonte probado**, y a 30/60 minutos deja de ser distinguible de cero incluso puntualmente.

## 9. Relación conceptual con TH10

TH10 encontró que `Var(r[h])` escala aproximadamente de forma lineal con `h` (pendiente log-log cercana a 1 en el ajuste principal — ver `reports/mnq/TH10_escalado_varianza_horizonte.md`). **Esto es compatible con, pero NO equivalente a, ausencia de autocorrelación**: son dos preguntas y dos estadísticos distintos.

- **TH10 (varianza)**: si hubiera dependencia serial positiva fuerte y sostenida entre bloques de `h` minutos, `Var(r[h])` tendería a crecer **más rápido** que linealmente con `h` (persistencia agrega varianza); si hubiera reversión fuerte, crecería **más lento** (sub-lineal). Un escalado aproximadamente lineal es el patrón esperable bajo autocorrelación DÉBIL o ausente — pero por sí solo no la demuestra ni la descarta con precisión, porque `Var(r[h])` es sensible a la SUMA de covarianzas entre todos los pares dentro del bloque, no solo al lag-1 entre bloques consecutivos.
- **TDA08-H (autocorrelación)**: mide DIRECTAMENTE `Corr(r_t[h], r_{t-1}[h])` entre bloques NO solapados consecutivos — la pregunta que TH10 no responde.

**Ambas evidencias, leídas juntas, son mutuamente consistentes**: el escalado ~lineal de TH10 y el `rho_1` estadísticamente indistinguible de cero a 30/60 minutos de TDA08-H apuntan en la misma dirección (ausencia de dependencia serial material a esos horizontes), pero se trata de **dos mediciones independientes que coinciden**, no de que una implique la otra.

## 10. Tamaño de efecto, no solo significancia (G5)

Ningún `rho_1`/`beta_1` de los cinco horizontes supera **0,0115** en valor absoluto. Para contextualizar sin inventar umbrales: en TDA-08 (§14, STOP-8a), la cifra global de 1 minuto (`rho_1=0,0059`, muy similar a las de aquí) se tradujo a **0,14 ticks** — una fracción muy pequeña de un solo tick. Los `rho_1`/`beta_1` de 30 y 60 minutos (0,0010 y −0,0064) son del mismo orden o **menores** que esa cifra ya calificada como "diminuta, sin relevancia operativa" en TDA-08. **No se calculó una traducción a ticks específica para 30/60 minutos** — deliberadamente omitida: el IC de ambos horizontes incluye cero holgadamente, así que cualquier conversión del punto estimado a ticks sugeriría una precisión que los datos no respaldan, y no es necesaria para responder la pregunta (¿existe memoria?) — la respuesta es negativa por magnitud Y por significancia, sin necesitar una unidad económica adicional.

**Calificación cualitativa** (sin umbral arbitrario, por comparación directa entre horizontes): los cinco `rho_1` son, en términos absolutos, **prácticamente nulos a pequeños** — ninguno califica como moderado o materialmente grande frente a la escala de variación natural de los propios retornos.

## 11. Validaciones ejecutadas

- Verificación mecánica de no-solapamiento (delta entre ventanas consecutivas del mismo bloque = exactamente `h` minutos, nunca menos).
- Prueba aritmética exacta de no-cruce de bloques (dos bloques sintéticos separados → `n_pairs_lag1` coincide exactamente con la fórmula esperada, sin fuga).
- `rho_1`/`beta_1` verificados contra la fórmula de Pearson pairwise-complete con centrado local a mano (no solo contra `compute_acf`, para evitar circularidad).
- Reproducibilidad del bootstrap con semilla fija (`n_boot=30`, `seed=42`, dos corridas idénticas).
- **Control de regresión estructural**: 1/5/10 de `compute_multi_horizon_memory` (TDA08-H) coinciden EXACTAMENTE con `compute_multi_frequency_rho1` (TDA-08), sobre datos sintéticos, sin hardcodear números del dataset real.
- **Control de regresión sobre datos reales**: 1/5/10 recalculados aquí reproducen los valores congelados de TDA-08 CLOSED dentro de tolerancia numérica (verificado automáticamente por el runner antes de persistir nada).
- `n`, `n_pairs_lag1`, `n_trading_dates` monótonamente decrecientes con el horizonte (excepto `n_trading_dates`, que decrece solo marginalmente).
- Holdout nunca abierto (guardas `validate_research_holdout_disjoint`/`validate_last_timestamps_before_boundary`, mismas que TDA-08/TH10, sin modificar).

**Verificación de estabilidad por año**: **NO se ejecutó** (§11 de la tarea: solo obligatoria si 30/60 resulta "materialmente grande o sorprendentemente distinto" de 1/5/10). No fue el caso — 30 minutos es, de hecho, el valor MÁS PEQUEÑO de los cinco, y 60 minutos es del mismo orden que 1 minuto; ambos son exactamente el tipo de resultado "pequeño y compatible con ruido" que no amerita minería adicional de subgrupos.

## 12. Tests y resultados

```
python -m pytest -q tests/test_tda08h_horizon_memory_extension.py
python -m pytest -q
```

- `test_tda08h_horizon_memory_extension.py`: **11 passed** — construcción de h=30/60 con estimaciones no nulas; `n_pairs≤n` en los 5 horizontes; `n` monótonamente decreciente; no-solapamiento verificado mecánicamente; no-cruce de bloques verificado aritméticamente; `rho_1`/`beta_1` verificados a mano contra la fórmula pairwise-complete; bootstrap reproducible; 1/5/10 reproducen `compute_multi_frequency_rho1` de TDA-08 exactamente; holdout nunca se abre (disjunción, nunca toca `raw/`, guarda de alineación de timestamps).
- **Suite completa: 358 passed** (347 previas, incluidas las 50 de TDA-08 CLOSED — confirmadas intactas — + 11 de esta extensión).

## 13. Archivos creados/modificados

**Creados:** `src/ohlcv_dataroad/ingest/tda08h_horizon_memory_extension.py`, `src/ohlcv_dataroad/ingest/run_tda08h.py`, `tests/test_tda08h_horizon_memory_extension.py`, este informe, `TDA08H_rho1_multi_horizon.csv`, `TDA08H_rho1_multi_horizon.png`.
**Editados (aditivo, sin tocar ninguna sección de TDA-08):** `src/ohlcv_dataroad/config.py` (nueva sección `tda08h_*`: 3 campos, 3 `@property`, 3 líneas en `load_config`), `configs/mnq_snapshot.yaml` (nueva sección `tda08h:`).
**NO modificado:** `tda08_linear_mean_dependence.py`, `run_tda08.py`, `tests/test_tda08_linear_mean_dependence.py`, `th10_horizon_scaling.py`, `TDA08_dependencia_lineal_media.md`, `TDA08_rho1_multi_frecuencia.csv` — TDA-08 y TH10 permanecen exactamente como estaban al cerrarse.

## 14. Holdout

**LOCKED**, no abierto. Verificado por `validate_research_holdout_disjoint`/`validate_last_timestamps_before_boundary` (reutilizadas sin modificar) y por test explícito (`test_run_tda08h_never_opens_any_raw_or_holdout_file`).

## 15. TDA-09

**No iniciada.** Esta tarea es exclusivamente TDA08-H.

## 16. Estado TDA08-H

**`PASS_WITH_OPEN_QUESTIONS`**

La pregunta central se respondió con claridad: no hay evidencia de dependencia lineal material a 30 ni a 60 minutos (magnitud mínima del conjunto en 30 min, IC amplio incluyendo cero en ambos). El control de regresión 1/5/10 confirmó que TDA-08 CLOSED sigue intacta. La pregunta abierta que queda (heredada, no nueva) es la misma que TDA-08 ya había declarado: por qué el signo/magnitud de la dependencia de rezago corto no sigue un patrón ordenado al variar el horizonte (§8) — no es un defecto de esta extensión ni bloquea su cierre, es una característica genuina de los datos ya reconocida.

**Decisión sobre G2/nulls (§10 de la tarea)**: NO se implementó una calibración G2 para 30/60 minutos. Justificación: el Null 1 de TDA-08 (permutación dentro de `minute_of_day`) depende de que cada valor de agrupación tenga cientos/miles de días compartiendo exactamente ese minuto — a horizontes de 30/60 minutos, el equivalente sería agrupar por "posición de la ventana no solapada dentro de la jornada" (p. ej., ventana `[09:30,10:00)`, `[10:00,10:30)`, ...), de la cual solo hay unas pocas decenas de valores distintos por jornada, con una heterogeneidad de escala probablemente MENOS extrema que la de `minute_of_day` a resolución de 1 minuto (no verificado aquí). Un null metodológicamente válido y análogo SERÍA una permutación dentro de cada posición de ventana no solapada, a través de los días que comparten esa posición — pero construirlo y validarlo (verificar que preserva la heterogeneidad correcta, no solo declararlo) es trabajo metodológico genuino, no una extensión trivial de lo ya construido. Se detiene aquí, sin implementarlo, tal como exige la tarea.

**Recomendación**: **continuar a TDA-09.** No apareció ninguna contradicción material y reproducible con TDA-08 CLOSED — al contrario, los horizontes largos refuerzan la conclusión ya declarada (dependencia en media diminuta, sin patrón ordenado, sin relevancia operativa). No hay razón para proponer reabrir TDA-08.

---

## Comandos exactos utilizados

```bash
python -m pytest -q tests/test_tda08h_horizon_memory_extension.py
python -m pytest -q
python -m ohlcv_dataroad.ingest.run_tda08h --config configs/mnq_snapshot.yaml
```
