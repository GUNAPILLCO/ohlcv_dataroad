# TDA-04 — Construcción y auditoría de las variables de análisis de 1 minuto

**Etapa:** `docs/methodology/Tsay_OHLCV_analysis_roadmap.md` § TDA-04
**Depende de:** TDA-00 (`PASS`), TDA-01 (`PASS_WITH_OPEN_QUESTIONS`), TDA-02 (`PASS_WITH_OPEN_QUESTIONS`), TDA-03 (`PASS_WITH_OPEN_QUESTIONS`)
**Alcance de datos:** exclusivamente la serie canónica de TDA-03 (`data/interim/mnq/tda03_serie_continua.parquet`), construida a su vez exclusivamente sobre el conjunto de investigación. Ningún archivo de `holdout_files` fue abierto en esta etapa — de hecho, TDA-04 no abre **ningún** archivo de `data/raw/`: su única entrada son los dos artefactos parquet que TDA-03 ya produjo.
**Evidencia reproducible:** `reports/mnq/TDA04_perdidas_por_causa.csv`, `TDA04_th07_r_vs_R.csv`, `data/interim/mnq/tda04_variables_1m.parquet`, `data/interim/mnq/tda04_return_validity_mask.parquet`. Todo generado por `python -m ohlcv_dataroad.ingest.run_tda04`.

> Esta etapa NO estudia la distribución de los retornos, NO calcula ACF/PACF, NO busca predictibilidad, NO define ningún horizonte de predicción ni ningún target, y NO introduce ninguna ventana horaria propia. Construye el retorno REALIZADO de 1 minuto — con sus reglas de no-cruce aplicadas — y audita cuantitativamente cuánto se pierde por cada regla.

---

## 1. Definición exacta de cada variable

| Variable | Definición | Depende de barra anterior | Regla de no-cruce |
|---|---|:---:|---|
| `r_1m` | $\ln(C_t/C_{t-1})$ | Sí | Completa (§3) |
| `R_1m` | $C_t/C_{t-1}-1$ | Sí | Idéntica a `r_1m` (mismas dos barras) |
| `abs_r_1m` | $\lvert r_{1m}\rvert$ | — | Heredada de `r_1m` por propagación |
| `r2_1m` | $r_{1m}^2$ | — | Heredada de `r_1m` |
| `zero_1m` | $\mathbf 1\{r_{1m}=0\}$ (1.0/0.0/`NaN`) | — | Heredada de `r_1m` — `NaN` exactamente donde `r_1m` es `NaN` |
| `log_hl` | $\ln(H_t/L_t)$ | **No** | Ninguna — usa solo la barra `t` |
| `log_co` | $\ln(C_t/O_t)$ | **No** | Ninguna — usa solo la barra `t` |
| `log_oc_prev` | $\ln(O_t/C_{t-1})$ | Sí | Idéntica a `r_1m` (usa la misma `C_{t-1}`) |
| `volume` | $V_t$ | — | Ninguna — dato de una sola barra |

`log_hl` y `log_co` **nunca** son `NaN` por una regla de no-cruce: TDA-00 ya certificó `H_t \ge \max(O_t,C_t) \ge \min(O_t,C_t)\ge L_t>0` en el 100 % de las filas del conjunto de investigación (0 violaciones) — condición que la serie canónica hereda intacta (TDA-03 sólo selecciona filas, nunca modifica valores). Verificado de nuevo aquí explícitamente (§7).

---

## 2. Definición exacta de "retorno válido"

Un retorno `r_1m` (y, con él, `R_1m` y `log_oc_prev`) es válido **si y sólo si** existe, para la barra `t`, una barra `t-1` que cumple **simultáneamente**:

1. `trading_date_t == trading_date_{t-1}` (misma jornada de negociación — TDA-02).
2. `segment_id_t == segment_id_{t-1}` (mismo contrato activo, sin cruzar un roll — TDA-03).
3. `timestamp_t - timestamp_{t-1} == exactamente 1 minuto`.
4. Ninguna de las dos filas fue excluida por la construcción de la serie canónica (garantizado por partir directamente de `tda03_serie_continua.parquet`: sólo contiene filas ya seleccionadas).

Si cualquiera falla: `r_1m = NaN`. **Nunca** se busca "la barra anterior disponible" saltando huecos, ni se rellena, ni se hace forward-fill, ni se une una jornada con la siguiente, ni se cruza un roll.

---

## 3. Auditoría de las reglas de no-cruce — esquema de causas

Cuatro categorías, evaluadas en este **orden de prioridad** (una fila puede fallar más de una a la vez; el orden decide cuál se reporta como causa única; las cuatro quedan también disponibles como banderas booleanas independientes, no excluyentes, para medir el solapamiento entre causas):

| Prioridad | Causa | Significa |
|---:|---|---|
| 1 | `FIRST_OBSERVATION` | No existe ninguna fila anterior en toda la serie (la primera barra del conjunto de investigación). |
| 2 | `ROLL_BOUNDARY` | `tda03_roll_mask["is_roll_boundary"]` es `True` para esta fila — leído **directamente** de la máscara de TDA-03, no recalculado, con una validación bloqueante previa que lo garantiza (§3.1). |
| 3 | `TRADING_DATE_BOUNDARY` | `trading_date` cambió respecto de la fila anterior, sin ser un roll. |
| 4 | `NON_CONSECUTIVE_MINUTE` | Mismo `trading_date`, mismo `segment_id`, pero la diferencia de timestamp no es exactamente 60 segundos. |
| — | `VALID` | Ninguna de las anteriores — `r_1m` calculable. |

### Ejemplos concretos (verificados con datos sintéticos, `tests/test_tda04_analysis_variables.py`)

- **`23:59 → 00:00` hora de Nueva York, mismo `trading_date`** (una sesión que cruza medianoche): `r_1m` **válido**. La regla compara `trading_date`, nunca la fecha de calendario del timestamp (`test_calendar_midnight_crossing_within_same_trading_date_is_valid`).
- **Reapertura tras el corte de mantenimiento nocturno**: la primera barra de la nueva `trading_date` (p. ej. `18:01` NY) es fila *inmediatamente siguiente* a la última del día anterior en el `DataFrame` — pero `trading_date` cambió: `TRADING_DATE_BOUNDARY`, `r_1m = NaN`. La segunda barra de esa misma jornada nueva sí enlaza con la primera (`test_bars_after_maintenance_do_not_link_with_previous_trading_date`).
- **Roll con delta de exactamente 1 minuto** (caso defensivo): aunque el tiempo transcurrido fuera de sólo 1 minuto, si `tda03_roll_mask` marca la frontera, `ROLL_BOUNDARY` tiene prioridad sobre "parece consecutivo" (`test_roll_mask_flag_is_authoritative_even_with_one_minute_delta`).
- **Hueco corto dentro del mismo día** (p. ej. el break secundario de TDA-02, pre-2021-06-25, o cualquier hueco interno breve): mismo `trading_date`, mismo `segment_id`, delta ≠ 1 minuto → `NON_CONSECUTIVE_MINUTE`.

### 3.1 Validación bloqueante: la máscara de roll y `segment_id`/`contract` deben coincidir

`ROLL_BOUNDARY` se decide leyendo `tda03_roll_mask["is_roll_boundary"]` **directamente**, sin recalcularlo a partir de `segment_id`. Esa elección de diseño depende de un supuesto — TDA-03 garantiza, por construcción, que `segment_id` cambia exactamente cuando `contract` cambia, y que ambos cambian exactamente donde `is_roll_boundary=True` — que TDA-04 **comprueba explícitamente antes de confiar en la máscara**, en vez de asumirlo en silencio: `build_return_validity_mask` calcula `segment_changed` y `contract_changed` respecto de la fila anterior y exige, para toda fila que no sea la primera observación, `segment_changed == contract_changed == is_roll_boundary`. Si alguna fila contradice esta invariante, se lanza `RollConsistencyError` de inmediato — la etapa aborta en vez de construir retornos sobre una serie canónica y una máscara de roll ya inconsistentes entre sí.

Verificado con 4 tests dedicados (`tests/test_tda04_analysis_variables.py`):

| Caso | Resultado esperado | Test |
|---|---|---|
| A. `segment_id`/`contract` cambian, `is_roll_boundary=True` | Comportamiento normal (sin excepción), `r_1m = NaN` por `ROLL_BOUNDARY` | `test_A_segment_and_contract_change_with_roll_mask_true_is_normal_and_returns_nan` |
| B. `segment_id`/`contract` cambian, `is_roll_boundary=False` | `RollConsistencyError` | `test_B_segment_and_contract_change_with_roll_mask_false_raises` |
| C. `is_roll_boundary=True` sin cambio de `segment_id`/`contract` | `RollConsistencyError` | `test_C_roll_mask_true_without_segment_or_contract_change_raises` |
| D. Artefactos REALES de TDA-03, las 21 fronteras de roll | La invariante se cumple para las 21, sin excepción | `test_D_invariant_holds_on_real_tda03_artifacts_for_all_21_boundaries` |

El caso D se ejecuta directamente sobre `data/interim/mnq/tda03_serie_continua.parquet` y `tda03_roll_mask.parquet` (no sobre datos sintéticos): confirma, sobre el conjunto de investigación real, que las 21 transiciones de contrato que TDA-03 caracterizó (§9 del informe TDA-03) son exactamente las 21 filas con `segment_changed=contract_changed=is_roll_boundary=True`, ni una más ni una menos.

---

## 4. Máscara de validez persistente

`data/interim/mnq/tda04_return_validity_mask.parquet` — una fila por barra de la serie canónica, con: `timestamp`, `source_file`, `contract`, `trading_date`, `segment_id` (actuales); `prev_timestamp`, `prev_trading_date`, `prev_segment_id`, `prev_contract`, `delta_minutes` (de la comparación contra la fila anterior); `is_first_observation`, `is_roll_boundary`, `is_trading_date_boundary`, `is_non_consecutive_minute` (banderas independientes); `r_1m_valid` (booleano); `invalid_reason` (categoría única, incluye `VALID`).

---

## 5. Auditoría cuantitativa — conjunto de investigación real

| | valor |
|---|---:|
| Filas de entrada (serie canónica de TDA-03) | 1.918.050 |
| Retornos válidos | **1.914.530** |
| Retornos inválidos (`r_1m = NaN`) | 3.520 |
| Porcentaje retenido | **99,8165 %** |
| Conservación | `1.918.050 = 1.914.530 + 3.520` ✔ |

### Pérdida por causa (categoría exclusiva)

| Causa | n filas | % del total |
|---|---:|---:|
| `FIRST_OBSERVATION` | 1 | 0,000052 % |
| `ROLL_BOUNDARY` | 21 | 0,001095 % |
| `TRADING_DATE_BOUNDARY` | 1.398 | 0,072887 % |
| `NON_CONSECUTIVE_MINUTE` | 2.100 | 0,109486 % |
| `VALID` | 1.914.530 | 99,816480 % |

### Banderas independientes y solapamiento entre causas

| Bandera | n filas | % del total |
|---|---:|---:|
| `is_first_observation` | 1 | 0,000052 % |
| `is_roll_boundary` | 21 | 0,001095 % |
| `is_trading_date_boundary` | 1.419 | 0,073981 % |
| `is_non_consecutive_minute` | 3.519 | 0,183468 % |
| `is_roll_boundary` **Y** `is_trading_date_boundary` | 21 | 0,001095 % |

**Interpretación de los solapamientos** (verificación de consistencia interna, no un hallazgo nuevo):
- `is_roll_boundary` (21) coincide **exactamente** con `is_roll_boundary ∧ is_trading_date_boundary` (21): **todo roll ocurre en una frontera de jornada, nunca a mitad de una** — consecuencia directa de que TDA-03 asigna el contrato activo por `trading_date` completo, nunca a mitad de un día (regla 8 heredada, "fecha efectiva = jornada siguiente observada").
- `is_trading_date_boundary` (1.419) − `is_roll_boundary` (21) = 1.398 = exactamente `TRADING_DATE_BOUNDARY` exclusivo. Y `is_trading_date_boundary` (1.419) = `1.420 trading_date distintas − 1`: la serie canónica cubre 1.420 jornadas de negociación distintas, con 1.419 fronteras entre ellas — verificado directamente (`c["trading_date"].nunique() == 1420`).
- `is_non_consecutive_minute` (3.519) − `NON_CONSECUTIVE_MINUTE` exclusivo (2.100) = 1.419 = exactamente `is_trading_date_boundary`: **toda frontera de jornada es, por definición, también un salto de más de 1 minuto** (ninguna sesión reabre exactamente 1 minuto después de cerrar) — consistencia esperada, confirmada.

### Otras comprobaciones de la sección 10 de la tarea

| Comprobación | Resultado |
|---|---|
| Precios ≤ 0 usados en los logaritmos | **0** (TDA-00 ya lo garantizaba; re-verificado aquí) |
| Valores no finitos en `log_hl`/`log_co` | **0** |
| Timestamps monotónicos | ✔ |
| Duplicados | **0** |

---

## 6. Pruebas de causalidad

**Test de reconstrucción** (sección 6 de la tarea): *"si ejecuto el cálculo con información disponible únicamente hasta `t`, ¿obtengo exactamente el mismo valor?"* Verificado explícitamente (`test_no_look_ahead_truncating_the_series_does_not_change_past_values`): recalcular todas las variables sobre una serie TRUNCADA (sin ninguna fila posterior a `t`) da exactamente los mismos valores, para toda `t`, que calcularlas sobre la serie completa — porque cada variable sólo mira `shift(1)` (la fila anterior), nunca `shift(-1)`.

| Variable | Información usada | Disponible desde (TDA-01, §5) | Clasificación |
|---|---|---|---|
| `r_1m`, `R_1m` | `close_t`, `close_{t-1}` | `t` (Close de la barra `t` sólo se conoce al cerrar `t`) | `CAUSAL_AT_BAR_CLOSE` |
| `abs_r_1m`, `r2_1m`, `zero_1m` | `r_1m` | `t` | `CAUSAL_AT_BAR_CLOSE` |
| `log_hl` | `high_t`, `low_t` | `t` | `CAUSAL_AT_BAR_CLOSE` |
| `log_co` | `close_t`, `open_t` | `t` (los 5 campos de la barra `t` llegan conjuntamente en `t`, TDA-01 §5) | `CAUSAL_AT_BAR_CLOSE` |
| `log_oc_prev` | `open_t`, `close_{t-1}` | `t` | `CAUSAL_AT_BAR_CLOSE` |
| `volume` | `volume_t` | `t` | `CAUSAL_AT_BAR_CLOSE` |

Las 8 variables son `CAUSAL_AT_BAR_CLOSE`: ninguna usa información de `t+k, k>0`. Ninguna es "puramente descriptiva retrospectiva" en el sentido de depender de toda la muestra (a diferencia, por ejemplo, de un z-score global) — todas son funciones puntuales de `t` y, a lo sumo, `t-1`.

---

## 7. Comprobaciones matemáticas (sección 11 de la tarea)

Las 8 comprobaciones exigidas están implementadas como tests dedicados y **todas pasan**:

1. Dos Close consecutivos válidos → `r_1m = ln(C_t/C_{t-1})` exacto.
2. Close sin cambio → `r_1m == 0` exacto.
3. `abs_r_1m == |r_1m|`.
4. `r2_1m == r_1m**2`.
5. `log_hl == ln(H/L)`.
6. `log_co == ln(C/O)`.
7. `log_oc_prev == ln(O/C_prev)` **sólo** cuando el par es comparable (verificado también el caso negativo: con una frontera de `trading_date` entre las dos filas, `log_oc_prev` es `NaN`, igual que `r_1m`).
8. **Suma de log-retornos**: `r_t + r_{t+1} = ln(C_{t+1}/C_{t-1})` dentro de una secuencia válida — comprobación puramente matemática (las tres barras son pasadas/realizadas), no una instrucción para construir un horizonte de 2 minutos como target.

Adicionalmente: ninguna variable derivada "rellena" un `NaN` de frontera (`test_no_function_fills_a_boundary_nan`): `r_1m`, `R_1m`, `abs_r_1m`, `r2_1m`, `zero_1m` y `log_oc_prev` son `NaN` simultáneamente en toda fila inválida.

---

## 8. Estado de TH07, TH08 y TH10

### TH07 — Equivalencia entre retorno logarítmico y simple: **RESUELTA**

Método mínimo del roadmap (distribución de `|r_1m - R_1m|`, global y por decil de magnitud de `|r_1m|`) ejecutado sobre las 1.914.530 filas válidas (`TDA04_th07_r_vs_R.csv`):

| decil (por `|r_1m|`) | n | media `|r-R|` | mediana | máximo |
|---:|---:|---:|---:|---:|
| GLOBAL | 1.914.530 | 8,78×10⁻⁸ | 8,46×10⁻⁹ | 4,75×10⁻⁴ |
| 0 (menores) | 191.453 | 7,26×10⁻¹¹ | — | 1,95×10⁻¹⁰ |
| 9 (mayores) | 191.453 | 6,99×10⁻⁷ | 3,20×10⁻⁷ | 4,75×10⁻⁴ |

La diferencia crece monótonamente con la magnitud del retorno (esperado matemáticamente: la aproximación de primer orden $r\approx R$ empeora con $|R|$), pero incluso en el decil de mayores movimientos la diferencia media es del orden de $7\times10^{-7}$ — **completamente despreciable a escala de 1 minuto** (equivalente a una fracción de un tick sobre casi cualquier nivel de precio del conjunto de investigación). **Resultado que apoya la hipótesis** (roadmap: "diferencia despreciable a escala de 1 minuto"), sin necesidad de segmentar por régimen de volatilidad para esta conclusión — la magnitud es tan pequeña en todos los deciles que no cambia ninguna decisión práctica de qué representación usar.

### TH08 — Efecto de las reglas de no-cruce: **PARCIALMENTE RESUELTA**

La parte que corresponde a esta etapa — "conteo de observaciones perdidas por causa" y "distribución de `gap_t` (`log_oc_prev`) por tipo de frontera" en su forma de auditoría de conteo — está resuelta (§5). La parte de "comparación de momentos y cuantiles con y sin las reglas" (el método mínimo completo de TH08 en el backlog) se **difiere explícitamente** a TDA-05/TDA-07: calcular momentos y cuantiles de `r_1m` es exactamente el tipo de análisis distribucional que esta tarea instruye no realizar todavía en TDA-04 (sección 15: "NO estudiar distribución de retornos"). Se documenta como pendiente, no como omitido por descuido.

### TH10 — Escalado de la varianza con el horizonte: **DIFERIDA EXPLÍCITAMENTE**

El roadmap asigna a TDA-04 el método mínimo de TH10 ($\log\mathrm{Var}(r[h])$ vs $\log h$, para varios $h$) como uno de sus outputs obligatorios. **Esta tarea instruye explícitamente, en tres secciones distintas (2, 7 y 8), no elegir ningún horizonte $h$ ni construir $\ln(C_{t+h}/C_t)$** — instrucción más específica y más reciente que el texto general del roadmap para esta iteración concreta de TDA-04. Se resuelve la tensión documentando la decisión, no silenciándola: **TH10 queda diferida**, no resuelta ni descartada, a una etapa posterior (candidata natural: un pequeño complemento previo a TDA-08/TDA-09, donde el escalado de varianza ya se necesita como diagnóstico barato). No es una decisión que cambie el significado del dataset ni de las variables ya construidas — es una decisión de secuenciación, tomada siguiendo la instrucción explícita y repetida de esta tarea.

---

## 9. Evaluación de STOP-4

**Criterio del roadmap**: *"si la regla de no-cruce elimina una fracción sustancial de la muestra, se revisa la definición de sesión antes de seguir. No se relaja la regla para 'salvar' observaciones."* El roadmap no fija un umbral numérico — es un juicio cualitativo, igual que `STOP-0` en TDA-00.

**Resultado**: **0,1835 %** de las filas (3.519 de 1.918.050, excluyendo la única fila `FIRST_OBSERVATION`) quedan sin retorno válido. Cada causa está, además, completamente explicada por hallazgos YA documentados en etapas anteriores, no por una regla nueva o sorpresiva:

- `ROLL_BOUNDARY` (21): exactamente las 21 transiciones de contrato que TDA-03 ya caracterizó una por una.
- `TRADING_DATE_BOUNDARY` (1.398 exclusivas, 1.419 totales): las fronteras entre las 1.420 jornadas de negociación distintas de la serie canónica — el número de sesiones, no un defecto.
- `NON_CONSECUTIVE_MINUTE` (2.100): huecos internos ya inventariados por TDA-02 (`TDA02_huecos.csv`) — mayoritariamente el break secundario pre-2021-06-25 (353 casos en todo el conjunto de investigación) y huecos cortos de causa `UNKNOWN` ya documentados y clasificados como compatibles con `AUSENTE` (TDA-02, §10.1).

**`STOP-4` NO se activa.** 0,18 % no es, bajo ninguna interpretación razonable, una "fracción sustancial" — y, más importante que el número en sí, ninguna de las causas es nueva o inexplicada: son exactamente las fronteras estructurales que TDA-02 y TDA-03 ya cuantificaron y justificaron. No hizo falta relajar ninguna regla para llegar a este resultado.

---

## 10. Protección del hold-out

`run_tda04_analysis` reutiliza `holdout_guard.py` de forma defensiva: aunque TDA-04 no abre ningún archivo de `data/raw/` (su única entrada son los parquet de TDA-03), se revalida `validate_research_holdout_disjoint` y `validate_last_timestamps_before_boundary` sobre los timestamps de la propia serie canónica, por consistencia con el resto del pipeline. Verificado con tests dedicados: un archivo declarado únicamente en `holdout_files` (inexistente en disco en el test) nunca se intenta abrir, y la corrida completa funciona correctamente incluso cuando el archivo de investigación declarado en la configuración tampoco existe en disco — confirma que TDA-04 verdaderamente nunca toca `data/raw/`.

---

## 11. Archivos creados/modificados

**Código nuevo:**
- `src/ohlcv_dataroad/ingest/tda04_analysis_variables.py`
- `src/ohlcv_dataroad/ingest/run_tda04.py`

**Código modificado:**
- `src/ohlcv_dataroad/config.py` (campos `tda04_*`)
- `configs/mnq_snapshot.yaml` (sección `tda04`)
- `src/ohlcv_dataroad/ingest/README.md` (secciones 13-14)

**Tests nuevos:**
- `tests/test_tda04_analysis_variables.py` (29 tests, incluidos los 4 de la corrección puntual de `RollConsistencyError`, §3.1)

**Artefactos generados:**
- `reports/mnq/TDA04_variables_analisis.md` (este informe)
- `reports/mnq/TDA04_perdidas_por_causa.csv`
- `reports/mnq/TDA04_th07_r_vs_R.csv`
- `data/interim/mnq/tda04_variables_1m.parquet`
- `data/interim/mnq/tda04_return_validity_mask.parquet`

---

## 12. Tests ejecutados

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
| `test_tda04_analysis_variables.py` | **29** (nuevos) |
| **Total previo (tras TDA-03)** | **118** |
| **Total final** | **147** |

**Resultado: `147 passed`.**

Cobertura explícita de los puntos exigidos (sección 12 de la tarea): minuto consecutivo/mismo día/mismo segmento → válido; mismo día con hueco de 2+ min → `NaN`; cambio de `trading_date` → `NaN`; cambio de `segment_id`/contrato → `NaN`; frontera marcada en `tda03_roll_mask` → `NaN` (incluso con delta de 1 minuto, caso defensivo); cruce de medianoche de calendario con mismo `trading_date` → válido; barras tras mantenimiento no enlazan con la jornada anterior; ninguna función rellena un `NaN` de frontera; ausencia de look-ahead (test de reconstrucción explícito); conservación exacta de filas; hold-out protegido (2 tests dedicados); archivo exclusivamente de investigación/hold-out inexistente en disco nunca se abre; suite completa anterior (118 tests) sigue pasando sin cambios. Corrección puntual de cierre (esta revisión): consistencia `segment_id`/`contract`/`is_roll_boundary` verificada de forma bloqueante, con 4 tests dedicados incluidos los 3 casos sintéticos A/B/C y el caso D sobre los artefactos reales de TDA-03 (§3.1).

---

## 13. Validación final

```
python -m ohlcv_dataroad.ingest.run_tda00           -> PASS (idéntico)
python -m ohlcv_dataroad.ingest.run_tda01_forensics -> 4.093 huecos, mismas anclas (idéntico)
python -m ohlcv_dataroad.ingest.run_tda02           -> cobertura 99,3909 %, STOP-2 no activado (idéntico)
python -m ohlcv_dataroad.ingest.run_tda03           -> 21 transiciones, serie canónica 1.918.050 filas (idéntico)
python -m ohlcv_dataroad.ingest.run_tda04           -> ver §5
```

TDA-00, TDA-01, TDA-02 y TDA-03 se re-ejecutaron íntegramente después de implementar TDA-04: **ningún resultado cambió**.

---

## 14. Preguntas que permanecen abiertas

1. **TH10 diferida** (§8): el diagnóstico de escalado de varianza queda pendiente para una etapa posterior, por instrucción explícita de esta tarea, no por limitación técnica.
2. **TH08, componente distribucional**: la comparación de momentos/cuantiles con y sin las reglas de no-cruce se difiere a TDA-05/TDA-07, junto con el resto del análisis distribucional que esta etapa no debía anticipar.
3. Ninguna pregunta nueva sobre la calidad del dato: todas las causas de `NaN` están completamente explicadas por hallazgos ya cerrados de TDA-02/TDA-03.

Ninguna de estas preguntas es bloqueante para una eventual TDA-05: ambas son diferimientos deliberados de alcance, no vacíos de evidencia.

---

## 15. Recomendación para TDA-05

- Usar `tda04_variables_1m.parquet` (`r_1m` y compañía) como base directa: ya tiene aplicadas todas las reglas de no-cruce, no hace falta repetir ningún filtro.
- Para cualquier pregunta sobre resolución efectiva/discreción (TDA-05 propiamente dicho, TH09), cruzar `r_1m` contra el `tick_size` declarado en `configs/mnq_snapshot.yaml` y contra `zero_1m` — la columna ya está lista.
- Retomar TH10 (escalado de varianza) como diagnóstico barato antes de TDA-08/TDA-09, tal como el propio roadmap sugiere ("anticipa TH16 y TH19"), ahora que TDA-04 ya dejó `r_1m` construido y auditado.
- No reintroducir ninguna de las 3.520 filas inválidas en ningún cálculo posterior — la máscara de validez (`tda04_return_validity_mask.parquet`) es la referencia para excluirlas de forma consistente en cualquier etapa futura.

---

## Estado final

**`PASS_WITH_OPEN_QUESTIONS`**

Las 8 variables mínimas del roadmap quedaron construidas sobre la serie canónica de TDA-03, con las reglas de no-cruce aplicadas de forma verificablemente causal (test de reconstrucción explícito) y auditadas cuantitativamente: 99,82 % de retornos válidos, con el 0,18 % restante completamente explicado por fronteras ya documentadas (roll, jornada, hueco interno) — ninguna causa nueva ni sorpresiva. Las 8 comprobaciones matemáticas exigidas pasan. `TH07` queda resuelta (diferencia `r_1m` vs `R_1m` despreciable, incluso en el decil de mayores movimientos). `TH08` queda parcialmente resuelta (conteo/auditoría sí, comparación distribucional diferida). `TH10` queda explícitamente diferida, por instrucción directa de esta tarea, a una etapa posterior. `STOP-4` no se activa. El hold-out permanece protegido y nunca se abrió ningún archivo de `data/raw/`.

**No se avanza a TDA-05.**
