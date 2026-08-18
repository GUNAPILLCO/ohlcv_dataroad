# TDA-03 — Rolls y construcción de la serie continua

**Etapa:** `docs/methodology/Tsay_OHLCV_analysis_roadmap.md` § TDA-03
**Depende de:** TDA-00 (`PASS`), TDA-01 (`PASS_WITH_OPEN_QUESTIONS`), TDA-02 (`PASS_WITH_OPEN_QUESTIONS`)
**Alcance de datos:** exclusivamente el conjunto de investigación (22 archivos, `< 2025-06-23 00:00:00 UTC`). Ningún archivo de `holdout_files` fue abierto en esta etapa. El tercer solapamiento documentado (`M26→U26`) cae íntegramente en el hold-out y **no se reanalizó ni se usó** para calibrar ninguna regla — se cita únicamente como antecedente ya publicado.
**Evidencia reproducible:** `reports/mnq/TDA03_transiciones.csv`, `TDA03_evidencia_diaria_solapamientos.csv`, `TDA03_evidencia_sin_solapamiento.csv`, `TDA03_tabla_invariancia.csv`, `TDA03_filas_descartadas.csv`, `TDA03_saltos_extremos_stop3.csv`, `TDA03_factores_ajuste.csv`, `data/interim/mnq/tda03_roll_mask.parquet`, `data/interim/mnq/tda03_serie_continua.parquet`. Todo generado por `python -m ohlcv_dataroad.ingest.run_tda03`.

> Esta etapa NO calcula retornos, NO calcula volatilidad, NO construye features ni targets, y NO define ninguna ventana operativa de ML. Determina, de forma causal y trazable, qué contrato está activo en cada instante, mide el basis donde es medible, y evalúa (sin forzar) los métodos de ajuste.

---

## 1. Alcance heredado y qué NO se reabrió

- TDA-01: timestamp raw UTC, timestamp = cierre de barra, cada archivo = un contrato trimestral individual, 22 archivos / 21 transiciones consecutivas en el conjunto de investigación, 19 sin solapamiento, 2 con solapamiento real (`Z24→H25` ~9 días, `H25→M25` ~10 días).
- TDA-02: calendario nativo `CME_Equity`, `trading_date` = fecha de cierre de sesión, grilla esperada ya corregida (break secundario excluido hasta 2021-06-25, `2025-01-09` con cierre real confirmado). TDA-03 reutiliza el mismo `SessionSchedule` sin reconstruirlo.
- El solapamiento `M26→U26` (§6.3 de `MNQ_DATA_PRIOR_KNOWLEDGE.md`) cae íntegramente en `holdout_files` (`25_mnq_06_26.Last.txt`, `26_mnq_09_26.Last.txt`) y **no se abrió** en esta etapa — se cita en este informe únicamente como antecedente ya documentado, nunca como evidencia usada.

---

## 2. Política de rollover: lógica heredada, parámetros recalibrados

`docs/instruments/mnq/MNQ_DATA_PRIOR_KNOWLEDGE.md`, §7.1, documenta 12 reglas de una política de rollover previamente validada para MNQ. Se conservó la **lógica** de las 12 (un contrato por fecha; nunca mezclar ni promediar; confirmación por volumen del entrante; comparación sólo sobre minutos compartidos; irreversibilidad; una sola confirmación; fecha efectiva = jornada siguiente observada; fechas detectadas, no hardcodeadas; trazabilidad total; conservación bloqueante; sin back-adjustment automático). Se **descartaron explícitamente** los parámetros de §7.2 (691 barras/día, umbral 55 %), calibrados bajo la ventana histórica `04:30–16:00`, que este proyecto no adoptó (TDA-01 §6, TDA-02 §1).

Política adoptada (`configs/mnq_snapshot.yaml`, sección `tda03`):

| Parámetro | Valor | Justificación |
|---|---|---|
| `min_incoming_share_shared` | **0.50** | "Cruce de dominancia": el entrante pasa a tener MÁS volumen que el saliente, sobre minutos compartidos. El umbral más simple posible — no requiere calibración. |
| `confirmation_sessions_required` | **1** | Regla 7 heredada ("no se exige confirmación doble"). Justificado empíricamente (§4): el share del entrante crece de forma monótona una vez cruza el umbral en ambas transiciones con solapamiento — exigir 2 confirmaciones habría dado exactamente la misma fecha efectiva. |
| `extreme_jump_top_n` | **40** | Tamaño del ranking de discontinuidades revisadas por STOP-3 (§9) — ver justificación del diseño en esa sección. |

---

## 3. Test de causalidad — prueba explícita

Enunciado (sección 4 de la tarea): *"si ejecuto el algoritmo disponiendo únicamente de los datos conocidos hasta ese instante, ¿obtengo exactamente la misma decisión?"*

`determine_overlap_rollover` recorre las jornadas del solapamiento en orden cronológico y calcula la señal usando exclusivamente `daily_evidence` de ese día y de días anteriores — ninguna fila posterior participa en el cálculo. Verificado con dos tests dedicados:

1. **Reconstrucción causal** (`test_rollover_decision_is_causal_truncated_future_does_not_change_past_decision`): recalcular la señal usando SOLO los datos hasta el día de la señal da exactamente la misma `signal_date` que usando el conjunto completo.
2. **Insensibilidad al futuro** (`test_rollover_decision_does_not_change_if_future_days_are_different`): forzar un valor absurdo en el share de un día POSTERIOR a la señal no cambia la señal ya emitida.

Además, la fecha **efectiva** nunca coincide con la fecha de la **señal** (regla 8 heredada): se aplica la jornada siguiente *observada*, nunca el mismo día en que la señal se conoció — verificado en `test_determine_overlap_rollover_effective_is_next_observed_date_not_signal_date`.

**Anti-patrón evitado explícitamente**: no se buscó "mirando todo el solapamiento, cuál fue el mejor día para cambiar" (eso sería *ex post*). La señal se dispara la primera vez que se cumple la condición mirando hacia adelante desde el pasado, no la mejor fecha vista en retrospectiva.

---

## 4. Las dos transiciones con solapamiento — detalle completo

### 4.1 Z24 → H25 (`19_mnq_12_24.Last.txt` → `20_mnq_03_25.Last.txt`)

| trading_date | out_bars | in_bars | share_shared | share_total | diff_points_mean | ratio_mean | n_pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2024-12-12 | 1.380 | 987 | 1,87 % | 1,80 % | 269,67 | 1,01242 | 987 |
| 2024-12-13 | 1.380 | 1.369 | 4,34 % | 4,34 % | 273,46 | 1,01257 | 1.369 |
| 2024-12-16 | 1.380 | 1.380 | 47,87 % | 47,87 % | 293,27 | 1,01339 | 1.380 |
| **2024-12-17** | 1.380 | 1.380 | **67,73 %** | 67,73 % | 300,41 | 1,01362 | 1.380 |
| **2024-12-18** | 1.379 | 1.380 | 79,60 % | 79,60 % | **298,15** | **1,01358** | 1.379 |
| 2024-12-19 | 1.380 | 1.380 | 88,44 % | 88,44 % | 272,69 | 1,01283 | 1.380 |
| 2024-12-20 | 918 | 1.380 | 96,49 % | 98,91 % | 265,09 | 1,01265 | 918 |

- **Señal**: `2024-12-17` (primer día con `share_shared ≥ 0.50`: 67,73 %).
- **Fecha efectiva**: `2024-12-18` (jornada siguiente observada).
- **Basis representativo** (medido en la fecha efectiva, sobre 1.379 pares de timestamps simultáneos): **+298,15 puntos** (H25 por encima de Z24), ratio **1,01358** (H25 ≈ 1,36 % por encima de Z24).
- El basis es notablemente estable durante todo el solapamiento (ratio entre 1,0124 y 1,0136 en los 7 días medidos, ver `TDA03_evidencia_diaria_solapamientos.csv`) — nunca cambia de signo ni de orden de magnitud.

### 4.2 H25 → M25 (`20_mnq_03_25.Last.txt` → `21_mnq_06_25.Last.txt`)

| trading_date | out_bars | in_bars | share_shared | share_total | diff_points_mean | ratio_mean | n_pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025-03-13 | 1.380 | 1.070 | 1,93 % | 1,85 % | 206,81 | 1,01064 | 1.070 |
| 2025-03-14 | 1.380 | 1.363 | 2,55 % | 2,55 % | 205,95 | 1,01057 | 1.363 |
| 2025-03-17 | **300** | 1.380 | 39,92 % | **89,56 %** | 207,41 | 1,01058 | 300 |
| **2025-03-18** | 1.080 | 1.380 | **68,38 %** | 69,32 % | 206,64 | 1,01052 | 1.080 |
| **2025-03-19** | 1.380 | 1.380 | 79,89 % | 79,89 % | **204,98** | **1,01047** | 1.380 |
| 2025-03-20 | 1.380 | 1.380 | 87,30 % | 87,30 % | 205,21 | 1,01038 | 1.380 |
| 2025-03-21 | 921 | 1.380 | 94,65 % | 98,66 % | 202,02 | 1,01028 | 921 |

- **Señal**: `2025-03-18` (primer día con `share_shared ≥ 0.50`: 68,38 %).
- **Fecha efectiva**: `2025-03-19`.
- **Basis representativo** (fecha efectiva, 1.380 pares): **+204,98 puntos**, ratio **1,01047**.
- **`2025-03-17` es un caso de cobertura reducida, no de ambigüedad de roll**: H25 (aún formalmente activo) sólo tiene 300 de 1.380 barras esperadas esa jornada (TDA-02 ya documentó este mismo hecho, `TDA02_dias_incompletos.csv`, cobertura 21,7 %). Por eso `share_shared` (39,9 %, sobre los 300 minutos que H25 sí tuvo) y `share_total` (89,6 %, sobre el volumen total del día) divergen tanto — es exactamente el ejemplo real que motiva usar `share_shared`, no `share_total`, para decidir (§2, regla 5 heredada). Como H25 sí tiene barras ese día (300, no cero), **no** se activa la regla de respaldo: la política conserva las 300 barras reales de H25 para esa fecha, sin mezclar con M25 y sin adelantar el cruce formal.

**Validación independiente**: ambas fechas efectivas (`2024-12-18`, `2025-03-19`) coinciden **exactamente** con las fechas de la política de rollover heredada (`MNQ_DATA_PRIOR_KNOWLEDGE.md` §7.3: `2024-12-18` y `2025-03-19`), calibrada de forma independiente bajo una ventana horaria y un umbral de volumen completamente distintos (04:30–16:00, 55 %). Que dos algoritmos con parámetros diferentes, sobre datos parcialmente distintos (ventana completa vs. `04:30–16:00`), converjan en el mismo día es la corroboración más fuerte disponible de que la regla adoptada aquí no es un artefacto de la elección de umbral.

---

## 5. Las 19 transiciones sin solapamiento

`TDA03_evidencia_sin_solapamiento.csv`. Ninguna de las 19 tiene un par de precios simultáneos: la distancia entre el último bar del saliente y el primero del entrante ronda 3.600–4.300 minutos (≈ 2,5–3 días de calendario). La diferencia aparente de precio (`apparent_diff_pct`) va de −9,58 % (`H20→M20`, la transición más antigua, año 2020) a +2,08 % (`H22→M22`), sin patrón sistemático de signo — consistente con una mezcla de basis genuino y movimiento real de mercado durante el intervalo, exactamente como advierte la sección 6 de la tarea. **Ninguna de estas 19 diferencias se usa para nada**: no alimenta el factor de ajuste (§7), no se interpreta como basis de roll. Cada fila queda clasificada con `confidence="BAJA"` de forma explícita e irrevocable — no es una etiqueta que otra evidencia pueda "mejorar" después: refleja una limitación estructural del dato (no hay timestamps simultáneos), no una falta de esfuerzo de medición.

---

## 6. Serie canónica

| | valor |
|---|---:|
| Filas de investigación (raw) | 1.937.230 |
| Filas en la serie canónica | **1.918.050** |
| Filas descartadas | 19.180 |
| — `NON_ACTIVE_CONTRACT_ON_OVERLAP_DATE` | 17.667 |
| — `OUT_OF_GRID_NO_TRADING_DATE` | 1.513 |
| Verificación de conservación | `1.937.230 = 1.918.050 + 19.180` ✔ (`assert` bloqueante) |
| Timestamps monotónicos | ✔ |
| Duplicados | **0** |
| Contratos activos por timestamp | exactamente 1 (verificado: `groupby(timestamp).size().max() == 1`) |
| Fronteras de roll marcadas | **21** (una por transición, incluidas las 19 sin solapamiento) |

Las `1.513` filas `OUT_OF_GRID_NO_TRADING_DATE` son exactamente las mismas barras fuera de grilla que TDA-02 ya documentó (§8 de `TDA02_cobertura.md`, motivo `NO_SESSION`) — TDA-03 no decide su contrato activo porque no pertenecen a ninguna sesión válida; se descartan con el mismo criterio, no se reinterpretan.

**Regla de respaldo (§4.2)**: 0 activaciones sobre el conjunto de investigación real (`active_calendar["reason"].value_counts()`: `NORMAL` 1.406, `PRE_CROSSOVER_OUTGOING_ACTIVE` 8, `POST_CROSSOVER_INCOMING_ACTIVE` 6, `ZERO_BARS_FALLBACK` **0**). El mecanismo existe, está testeado con datos sintéticos, y sobre este dataset concreto no hizo falta.

**Columnas de `tda03_serie_continua.parquet`**: `source_file`, `contract`, `trading_date`, OHLCV crudo intacto, `segment_id`, y las 8 columnas ajustadas (`open/high/low/close_adj_ratio`, `..._adj_diff`, más `ratio_factor`/`diff_factor`/`basis_chain`) — nunca sobrescriben las columnas crudas.

---

## 7. Métodos de ajuste — evaluados, no forzados

### 7.1 Las tres alternativas

| Método | Qué preserva EXACTAMENTE | Qué modifica | Riesgo de causalidad |
|---|---|---|---|
| **Sin ajuste** (raw por contrato) | Todo — es el dato tal cual | Nada, pero dos saltos de nivel (uno por roll) quedan en la serie | Ninguno: es la representación más trazable |
| **Aditivo/diferencia** | Diferencias en puntos (`C_t - C_{t-1}`), rangos absolutos | Retornos simples/log, ratios, medidas relativas | El factor (una constante aditiva) depende de rolls **posteriores**, aunque sea constante dentro del segmento |
| **Ratio** | Retornos simples/log, rangos relativos | Diferencias en puntos, medidas en ticks (el precio ajustado puede dejar de caer en la grilla exacta) | Igual que el aditivo: el factor depende de rolls posteriores |

Ningún método preserva ambas propiedades a la vez (fundamento Tsay de esta etapa) — confirmado matemáticamente en la tabla de invariancia (§8) y empíricamente en el código (`test_ratio_adjustment_preserves_returns_within_segment`, `test_diff_adjustment_preserves_point_differences_within_segment`).

### 7.2 Por qué NO se puede ajustar todo el historial de forma defendible

El basis (§4) sólo es medible donde hay timestamps simultáneos — es decir, únicamente en las 2 transiciones con solapamiento. Encadenar un factor de ajuste hacia atrás requiere, en cada paso, un basis medido; con 19 de 21 transiciones sin solapamiento, la cadena se rompe casi de inmediato. **Resultado concreto sobre este conjunto de investigación** (`TDA03_factores_ajuste.csv`): sólo los 3 contratos más recientes (`Z24`, `H25`, `M25`) tienen un factor de ajuste defendible (`basis_chain=True`); los 19 contratos anteriores (`H20`...`U24`) quedan explícitamente `NaN` — no se inventó un factor a partir de una diferencia que podría ser, en parte o en su totalidad, movimiento genuino de mercado (§5).

### 7.3 Representación canónica elegida

**Se conserva el precio crudo por contrato (sin ajuste) como representación PRIMARIA**, con máxima trazabilidad — consistente con la regla 12 heredada ("ningún back-adjustment automático") y reforzado por la evidencia propia de esta etapa (§7.2: un ajuste end-to-end no es defendible con este dataset). Se proveen columnas ajustadas (ratio y aditivo) como **vista adicional, parcial y claramente etiquetada**, útil sólo dentro del tramo `Z24→H25→M25`; fuera de ese tramo, quedan `NaN`, no una aproximación. La máscara de roll (§6) permite a TDA-04 excluir cualquier frontera de contrato de cualquier cálculo, independientemente de qué representación se use.

Esta es, explícitamente, la salida que exige la sección 8 de la tarea cuando la evidencia no permite declarar un único método superior: no se fuerza una falsa certeza.

---

## 8. Tabla de invariancia (TH05)

Clasificación matemática (no calculada sobre datos): ¿cambia el VALOR NUMÉRICO del estadístico si se multiplica el segmento por una constante (ratio) o se le suma una constante (aditivo)?

| Estadístico | Invariante a ratio | Invariante a aditivo |
|---|:---:|:---:|
| Nivel de precio ($C_t$) | No | No |
| Diferencia en puntos ($C_t-C_{t-1}$) | No | **Sí** |
| Retorno simple ($R_t$) | **Sí** | No |
| Log-retorno ($r_t$) | **Sí** | No |
| Rango absoluto ($H_t-L_t$) | No | **Sí** |
| Rango relativo/log ($\ln(H_t/L_t)$) | **Sí** | No |
| Medidas en ticks | No | **Sí** |
| Varianza/desvío de retornos | **Sí** | No |
| Varianza/desvío de diferencias en puntos | No | **Sí** |
| Cuantiles de retornos | **Sí** | No |
| Cuantiles de niveles/diferencias en puntos | No | Parcial (diferencias sí, niveles no) |
| Correlación / ACF de retornos | **Sí** | No |
| Volumen ($V_t$) | **Sí** | **Sí** (ningún ajuste de precio lo toca) |

**Consecuencia operativa para TDA-04 en adelante** (criterio de interpretación del roadmap): prácticamente todo lo que las etapas siguientes necesitan — retornos, cocientes, ACF de retornos, momentos de retornos, cuantiles de retornos — es invariante al ratio. Puede calcularse indistintamente sobre la serie cruda o sobre la ajustada por ratio (dentro del tramo donde existe), **sin que el resultado cambie**. Lo que NO es invariante (umbrales de precio absolutos, distancias en puntos comparadas a través de una frontera de roll, medidas en ticks sobre precio ajustado) debe calcularse sobre la serie CRUDA por contrato, nunca sobre la ajustada.

---

## 9. STOP-3

### 9.1 Diseño (y por qué se descartó el primero)

Un primer diseño (umbral fijo de "K × MAD global" sobre el salto de precio en puntos) marcó **26.225** barras como candidatas — inservible para revisión humana. La causa: MNQ cambia de nivel ~3× a lo largo del conjunto de investigación (~7.000 → ~22.000 puntos) y su volatilidad no es estacionaria (marzo 2020 frente a 2023); un umbral global de dispersión no es apropiado. Se sustituyó por un **ranking acotado** (`top_n=40`) de las mayores discontinuidades RELATIVAS entre barras consecutivas de 1 minuto del MISMO contrato (las que cruzan una frontera de roll quedan excluidas: ya están explicadas), con dos piezas de contexto por candidato: volumen de la barra y si el precio revierte más de la mitad del salto en la barra siguiente.

### 9.2 Resultado sobre el conjunto de investigación real

**40 candidatos revisados, 0 sospechosos, `STOP-3` NO se activa.**

Los 40 se concentran en dos patrones reconocibles y no relacionados con calidad del dato:

- **~40 % ocurren a las `12:31`/`13:31` UTC** — exactamente `08:31` hora de Nueva York, un minuto después de la hora estándar de publicación de datos macroeconómicos de EE. UU. (IPC, nóminas no agrícolas, ventas minoristas...). Volumen alto en todos los casos (miles a >15.000 contratos), sin reversión.
- El resto se concentra en dos semanas de volatilidad extrema ya documentadas: **marzo de 2020** (crisis COVID, ya citada en TDA-02) y **abril de 2025** (semana del shock arancelario global, `21_mnq_06_25.Last.txt`, 7–9 de abril).

3 de los 40 revierten parcialmente en la barra siguiente, pero ninguno en volumen bajo (percentil 5 o menor) — la combinación que definiría un candidato `suspicious`. Ningún candidato es, con la evidencia disponible, distinguible de un evento de mercado genuino.

---

## 10. Estado de TH05

| Pregunta de TH05 | Respuesta |
|---|---|
| ¿Es una serie continua? | No en el archivo crudo; sí tras TDA-03 (serie canónica, un contrato por timestamp) |
| ¿Dónde están los rolls? | 21 fronteras localizadas, trazadas en la máscara persistente |
| ¿Qué magnitud tienen los saltos? | Medida con confianza ALTA en 2/21 (basis directo: +298,2 pts / +1,36 % en Z24→H25, y +205,0 pts / +1,05 % en H25→M25); con confianza BAJA (diferencia aparente, no basis puro) en 19/21 |
| ¿Qué método de ajuste se aplicó? | Ninguno de forma automática/global; ratio y aditivo evaluados y disponibles como vista parcial (§7.3) |
| ¿Qué estadísticos son utilizables sobre la serie ajustada? | Los invariantes al ratio (§8) — la inmensa mayoría de lo que el roadmap necesita |

`TH05` queda **resuelta hasta donde la evidencia lo permite**: no completamente indeterminada (2 transiciones sí tienen basis medido con alta confianza, y ese basis es notablemente estable dentro de cada solapamiento), pero tampoco con un ajuste único y global disponible (19 transiciones no tienen ningún par de precios simultáneos). Es exactamente el resultado "parcial" que la propia ficha de TH05 anticipaba (`Tsay_empirical_hypotheses_backlog.md`: "Datos disponibles: **Parcial**").

---

## 11. Protección del hold-out

`run_tda03_analysis` reutiliza, sin copiarla, `holdout_guard.py` (igual que TDA-00/01/02): `validate_research_holdout_disjoint` antes de abrir cualquier archivo, `validate_last_timestamps_before_boundary` después de parsear el conjunto de investigación. Verificado con tests dedicados que confirman explícitamente que un archivo declarado sólo en `holdout_files` (inexistente en disco en el test) nunca se intenta abrir. El solapamiento `M26→U26` (íntegramente en el hold-out) no se reanalizó ni se usó para calibrar `min_incoming_share_shared`, `confirmation_sessions_required` ni ningún otro parámetro — la calibración (en el sentido de "elección justificada", no de un ajuste numérico a los datos) se basó únicamente en las 2 transiciones del conjunto de investigación y en el razonamiento de parsimonia de §2.

---

## 12. Archivos creados/modificados

**Código nuevo:**
- `src/ohlcv_dataroad/ingest/tda03_rolls.py`
- `src/ohlcv_dataroad/ingest/run_tda03.py`

**Código modificado:**
- `src/ohlcv_dataroad/config.py` (campos `tda03_*`)
- `configs/mnq_snapshot.yaml` (sección `tda03`)
- `src/ohlcv_dataroad/ingest/README.md` (secciones 11-12)

**Tests nuevos:**
- `tests/test_tda03_rolls.py` (27 tests)

**Artefactos generados** (`reports/mnq/` y `data/interim/mnq/`):
- `TDA03_rolls_serie_continua.md` (este informe)
- `TDA03_transiciones.csv`
- `TDA03_evidencia_diaria_solapamientos.csv`
- `TDA03_evidencia_sin_solapamiento.csv`
- `TDA03_tabla_invariancia.csv`
- `TDA03_filas_descartadas.csv`
- `TDA03_saltos_extremos_stop3.csv`
- `TDA03_factores_ajuste.csv`
- `data/interim/mnq/tda03_roll_mask.parquet`
- `data/interim/mnq/tda03_serie_continua.parquet`

---

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
| `test_tda03_rolls.py` | **27** (nuevos) |
| **Total** | **118** |

**Resultado: `118 passed`.**

Cobertura explícita de los 17 puntos exigidos por la tarea (sección 15): transición con overlap (`test_classify_transition_detects_overlap`), transición sin overlap (`test_classify_transition_detects_no_overlap`), crossover de volumen (`test_determine_overlap_rollover_crosses_on_dominance_day`), causalidad (`test_rollover_decision_is_causal_truncated_future_does_not_change_past_decision`), ausencia de look-ahead (`test_rollover_decision_does_not_change_if_future_days_are_different`), irreversibilidad (`test_active_calendar_is_pre_crossover_then_post_crossover_never_reverts`), contrato único por timestamp (`test_canonical_series_has_no_duplicate_timestamps_and_is_monotonic`), ausencia de duplicados (idem), trazabilidad `source_file`/`contract` (`test_canonical_series_preserves_source_file_and_contract_traceability`), marcación de frontera en la máscara (`test_roll_mask_flags_exactly_the_contract_boundary`), basis en timestamps simultáneos (`test_basis_evolution_uses_only_simultaneous_timestamps`), comportamiento matemático de ratio/aditivo (`test_ratio_adjustment_preserves_returns_within_segment`, `test_diff_adjustment_preserves_point_differences_within_segment`), tabla de invariancia (`test_invariance_table_has_expected_categories_and_types`), protección del hold-out (3 tests dedicados), suite completa previa (91 tests de TDA-00/01/02, sin cambios).

---

## 14. Validación empírica final

```
python -m ohlcv_dataroad.ingest.run_tda00           -> TDA-00 status: PASS (idéntico a antes de TDA-03)
python -m ohlcv_dataroad.ingest.run_tda01_forensics -> 4.093 huecos, mismas 3 anclas forenses (idéntico)
python -m ohlcv_dataroad.ingest.run_tda02           -> cobertura 99,3909 %, STOP-2 no activado (idéntico)
python -m ohlcv_dataroad.ingest.run_tda03           -> ver resumen abajo
```

- 22 contratos research, **21 transiciones evaluadas**: **19 `NO_OVERLAP`**, **2 `OVERLAP`**.
- `Z24→H25`: señal `2024-12-17`, efectiva `2024-12-18`, basis directo +298,15 pts / ratio 1,01358 (factor de ajuste ENCADENADO de Z24 hasta M25, pasando por los dos rolls: +503,14 pts / ratio 1,02419 — ver §7.2).
- `H25→M25`: señal `2025-03-18`, efectiva `2025-03-19`, basis +204,98 pts / ratio 1,01047.
- Serie canónica: **1.918.050** filas, timestamps monotónicos, **0** duplicados, **21** fronteras de roll marcadas.
- Ajuste disponible (basis encadenado) en **3 de 22** contratos (`Z24`, `H25`, `M25`).
- `STOP-3`: **no activado** (40 candidatos revisados, 0 sospechosos).

TDA-00, TDA-01 y TDA-02 se re-ejecutaron íntegramente después de implementar TDA-03: **ningún resultado cambió** (mismas cifras de cobertura, huecos y violaciones que en sus respectivos informes).

---

## 15. Preguntas que permanecen abiertas

1. **Basis fuera del tramo `Z24→H25→M25`**: indeterminable con este dataset (sin timestamps simultáneos en 19 de 21 transiciones). No es una limitación de esta etapa: es una propiedad estructural de los datos (§5, §7.2).
2. **Signo y magnitud de la diferencia aparente en las 19 transiciones sin solapamiento** (§5): mezcla no separable de basis genuino y movimiento de mercado. Si una etapa futura necesitara una aproximación, tendría que declarar explícitamente ese supuesto adicional — no se resuelve aquí.
3. **Los 40 candidatos de STOP-3** (§9.2): no se verificó cada fecha contra un calendario oficial de publicaciones macroeconómicas de EE. UU. (acceso a fuentes externas fuera del alcance de esta etapa) — el patrón horario (`08:31` NY) es fuertemente sugestivo pero no confirmado fecha por fecha.

Ninguna de estas preguntas es bloqueante para TDA-04: todas están acotadas, documentadas con su evidencia completa, y no comprometen la validez de la serie canónica ni de la máscara de roll.

---

## 16. Recomendación para TDA-04

- Usar la serie canónica (`tda03_serie_continua.parquet`) como base, con las columnas OHLCV **crudas** por defecto para cualquier variable no invariante al ratio (§8).
- Aplicar la máscara de roll (`tda03_roll_mask.parquet`, columna `is_roll_boundary`) como una regla de no-cruce más, exactamente igual que las fronteras de sesión/hueco de TDA-02: `r_t = NaN` cuando la barra `t-1` tiene un `segment_id` distinto de la barra `t`.
- Para cualquier análisis que necesite comparar niveles de precio o distancias en puntos a través de una frontera de roll (fuera del alcance normal de TDA-04, pero por si surgiera), usar las columnas `*_adj_ratio` sólo dentro del tramo `Z24→H25→M25`, nunca fuera de él.
- Las filas descartadas (`TDA03_filas_descartadas.csv`) y las barras fuera de grilla (ya conocidas desde TDA-02) no deben reintroducirse en ningún cálculo de TDA-04.

---

## Estado final

**`PASS_WITH_OPEN_QUESTIONS`**

Las 21 transiciones del conjunto de investigación quedaron caracterizadas (19 sin solapamiento, 2 con solapamiento). La regla de rollover adoptada es causal (verificado con test de reconstrucción explícito), simple (un único parámetro sin calibrar, umbral de dominancia 50 %), irreversible, y reproduce exactamente las fechas de una política heredada calibrada de forma independiente bajo otros parámetros — la validación cruzada más fuerte disponible sin poder usar el hold-out. La serie canónica tiene exactamente un contrato activo por timestamp, sin duplicados, con conservación de filas verificada de forma bloqueante. El método de ajuste queda **evaluado explícitamente, no forzado**: sólo defendible en un tramo acotado (3 de 22 contratos) — resultado honesto dado que 19 de 21 transiciones no tienen ningún par de precios simultáneos. `TH05` queda resuelta hasta donde la evidencia lo permite. `STOP-3` no se activa: las 40 mayores discontinuidades de precio del conjunto de investigación son consistentes con eventos de mercado conocidos (publicaciones macroeconómicas, crisis de marzo 2020, shock arancelario de abril 2025), no con errores de datos.

**No se avanza a TDA-04.**
