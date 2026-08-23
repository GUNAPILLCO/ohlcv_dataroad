# TDA-09 — Dependencia en magnitud (volatility clustering)

**Etapa:** `docs/methodology/Tsay_OHLCV_analysis_roadmap.md` § TDA-09
**Depende de:** TDA-06 (`PASS_WITH_OPEN_QUESTIONS`, STOP-6 no activado, `s(m)` construido sobre `log_hl`), TDA-08 (`PASS_WITH_OPEN_QUESTIONS` / `CLOSED`, dependencia lineal en la media diminuta y `NOT SEPARABLE WITH OHLCV LAST`), TDA08-H (`PASS_WITH_OPEN_QUESTIONS`, sin memoria lineal material a 30/60 min).
**Alcance de datos:** exclusivamente `tda04_variables_1m.parquet`, `tda04_return_validity_mask.parquet`, `tda06_r_tilde.parquet`, `tda06_s_m.parquet` y `TDA06_segmentacion_propuesta.csv`. Ningún archivo de `data/raw/` ni de `holdout_files` fue abierto en esta etapa.
**Evidencia reproducible (14 CSV + 3 PNG):** ver §21. Todo generado por `python -m ohlcv_dataroad.ingest.run_tda09 --config configs/mnq_snapshot.yaml`.

> Esta etapa estudia si el TAMAÑO de los movimientos de MNQ tiene memoria (¿los períodos agitados siguen agitados?), separándolo de si la HORA del día explica ese patrón. NO diseña modelos predictivos, NO ajusta ARCH/GARCH como modelo productivo, NO crea features/targets/señales, NO abre el holdout, y NO inicia TDA-10 ni TDA-11.

---

## 1. La pregunta en palabras simples

Aunque TDA-08 ya mostró que el retorno pasado (`r_t`) casi no ayuda a adivinar la **dirección** del próximo movimiento, esta etapa pregunta algo distinto: ¿ayuda a adivinar el **tamaño** del próximo movimiento?

Ejemplo: una serie como `+grande, -grande, +grande, -grande` puede ser casi imposible de predecir en dirección, pero perfectamente predecible en tamaño ("hoy hay barras grandes"). A eso se le llama **agrupamiento de volatilidad** ("volatility clustering"): los movimientos grandes tienden a seguir a movimientos grandes, y los movimientos pequeños a movimientos pequeños.

La complicación es que MNQ ya tiene (TDA-06) un patrón de reloj muy fuerte: la apertura de NYSE es sistemáticamente más agitada que la madrugada, todos los días. Si no se controla por eso, cualquier "agrupamiento" que se observe podría ser simplemente "las horas activas duran varios minutos seguidos" — un artefacto del calendario, no una propiedad genuina del mercado. Esta etapa separa ambas cosas.

## 2. Datos y artefactos de entrada

- `tda04_variables_1m.parquet` / `tda04_return_validity_mask.parquet` (TDA-04, `CLOSED`): `r_1m`, `abs_r_1m`, `r2_1m`, `log_hl` y las reglas de no-cruce ya auditadas.
- `tda06_r_tilde.parquet` / `tda06_s_m.parquet` (TDA-06, `PASS_WITH_OPEN_QUESTIONS`): la serie ajustada por reloj `r_tilde = r_1m / s(m)` y la tabla del factor estacional `s(m)`, ambas `RETROSPECTIVAS` (estimadas sobre toda la muestra, no disponibles causalmente en producción).
- `TDA06_segmentacion_propuesta.csv`: los 7 tramos horarios empíricos de TDA-06, reutilizados sin modificar.
- Motor de ACF/bootstrap/G2 de TDA-08 (`tda08_linear_mean_dependence.py`), **reutilizado sin ninguna modificación**: `compute_block_ids`, `compute_acf`, `bartlett_se`, `bootstrap_rho`, `acf_by_group`, `g2_permutation_null_by_minute`, `g2_synthetic_empirical_null` + su diagnóstico de momentos, `compute_portmanteau_q`/`annotate_portmanteau_calibration`.

**Población analizada:** 1.914.530 filas con `r_1m` válido (idéntico a TDA-06/07/08); 1.918.050 filas de `log_hl` (todas las barras admisibles — `log_hl` no necesita una barra anterior, TDA-04).

## 3. Decisión: `r_t` vs. innovación/residuo de media

TDA-08 (`CLOSED`) encontró que la dependencia lineal de `r_t` es diminuta (`beta_1 = 0,0059`, equivalente a **0,14 ticks**) y no separable de microestructura con estos datos. Por parsimonia (G4), esta etapa usa `r_1m` **directamente** como "innovación" — no se ajustó ningún modelo AR nuevo.

Se ejecutó, como sensibilidad barata única (`TDA09_sensibilidad_remocion_media.csv`), la comparación entre `ACF(|r|)` y `ACF(|r − β₁·r_{t-1}|)` (residuo de un AR(1) mínimo, `β₁` recalculado sobre los datos reales, no congelado de TDA-08):

| Rezago | `rho(|r|)` | `rho(\|r − β₁r_{t-1}\|)` |
|---:|---:|---:|
| 1 | 0,4237 | 0,4234 |
| 5 | 0,3891 | 0,3891 |
| 20 | 0,3492 | 0,3495 |
| 60 | 0,3114 | 0,3117 |

**Resultado**: prácticamente idénticos en los 4 rezagos. Remover la media no cambia en nada la conclusión de magnitud — confirma que usar `r_1m` directamente es defendible y que abrir una etapa de modelado de media aquí habría sido innecesario.

## 4. Variables de magnitud analizadas

- `|r_1m|` y `r_1m²` (crudas) vs. `|r_tilde|` y `r_tilde²` (ajustadas) — identidades algebraicas directas de `r_tilde = r_1m/s(m)`, ninguna transformación nueva.
- `log_hl = ln(H/L)` (crudo) vs. `log_hl_tilde = log_hl / s(m)` (ajustado) — transformación **nueva** de esta etapa, pero dimensionalmente coherente porque `s(m)` fue estimado por TDA-06 usando exactamente `log_hl` como proxy elegido (verificado en código con `verify_s_m_is_log_hl_proxy`, que detiene la etapa si `s(m)` alguna vez proviniera de otro proxy).
- `r_1m`/`r_tilde` (dirección) también se calcularon hasta el mismo rezago máximo, únicamente como referencia de contraste para el "gráfico triple".

Todo lo etiquetado "ajustado" hereda la etiqueta `RETROSPECTIVO` de TDA-06 — nunca se presenta como disponible causalmente en producción.

## 5. Cómo se hizo el ajuste de reloj

`log_hl_tilde = log_hl / s(m)`, con protección de división por cero idéntica a la de `build_r_tilde` de TDA-06 (`s(m)=0`/no finito → `NaN` explícito, nunca `inf`). Un diagnóstico nuevo (`clock_profile_flatness`, `TDA09_diagnostico_aplanamiento_reloj.csv`) verificó, en vez de asumir, que el ajuste efectivamente aplana el perfil por minuto:

| Variable | Aplanamiento crudo | Aplanamiento ajustado | Razón | ¿Reloj removido? |
|---|---:|---:|---:|---|
| `\|r\|` | 0,584 | 0,045 | 0,078 | Sí |
| `log_hl` | 0,601 | ~0 (9,3e-17) | ~0 | Sí |

(El aplanamiento es `std(mediana por minuto)/media(mediana por minuto)` — cuanto más bajo, más "plano". `log_hl` queda perfectamente plano porque `s(m)` se estimó exactamente sobre `log_hl`; `|r|` se aplana de forma indirecta y algo menos completa, pero muy por debajo del umbral predeclarado de 0,20.)

Con el perfil de reloj confirmado como removido, se calculó además un null de permutación **global** (más simple) como sensibilidad secundaria para la serie ajustada — nunca como inferencia principal (que sigue siendo siempre la permutación dentro de cada minuto, más conservadora).

## 6. ACF de dirección vs. magnitud (gráfico triple)

| Rezago (min) | `rho(r)` crudo | `rho(\|r\|)` crudo | `rho(r²)` crudo |
|---:|---:|---:|---:|
| 1 | 0,0059 | **0,4237** | 0,1512 |
| 5 | −0,0045 | **0,3891** | 0,1018 |
| 20 | 0,0030 | **0,3492** | 0,0691 |
| 60 | 0,0004 | **0,3114** | 0,0596 |
| 240 | 0,0019 | **0,2252** | 0,0337 |
| 600 | 0,0017 | **0,1713** | 0,0413 |

`ACF(r) ≈ 0` en todos los rezagos (consistente con TDA-08: sin dependencia lineal material) mientras `ACF(|r|)` es enorme y sigue siendo **0,17 diez horas después**. Esta es exactamente la separación entre dirección y magnitud que anticipa el roadmap: "no correlacionado pero dependiente". `r²` muestra la misma historia pero mucho más ruidosa/atenuada (más impreciso como proxy, tal como advierte Tsay) — confirma que **no conviene usar solo `r²`**.

## 7. Resultado de `|r|`

Fuerte en todos los rezagos evaluados (hasta 600 minutos = 10 horas): 0,42 en el rezago 1, decae de forma **muy lenta**, todavía 0,17 a las 10 horas. Calibración G2 (`TDA09_g2_calibracion_null1_principal.csv`): el `rho` real supera el percentil 97,5 del null de permutación por minuto en el **100% de los rezagos evaluados** (1, 2, 5, 10, 20, 30, 60), con márgenes enormes (p.ej. rezago 1: real=0,424 vs. banda del null=0,119).

## 8. Resultado de `r²`

Misma dirección cualitativa (positivo, decae con el rezago) pero sistemáticamente más pequeño y más ruidoso que `|r|` en cada rezago (0,15 vs. 0,42 en el rezago 1; 0,04 vs. 0,17 a los 600 minutos) — confirma la advertencia de Tsay/roadmap: `r²` es insesgado pero mucho más impreciso. Se reporta como contraste, no como proxy principal.

## 9. Resultado de `log_hl`

El proxy **más fuerte y más persistente** de los tres: 0,76 en el rezago 1, y todavía **0,32 diez horas después**. Confirma la elección de TDA-06 de usar `log_hl` como proxy preferido (menos ruidoso, más informativo).

## 10. Cuánta dependencia desapareció al ajustar por el reloj — y cuánta sobrevivió

Métrica predeclarada (TH21): energía de dependencia `Q(m) = Σ n_pares_k·rho_k²` (la misma fórmula del portmanteau, reutilizada), evaluada en una ventana fija de **4 horas** (`m=240`, elegida antes de mirar el resultado):

| Variable | `Q` cruda | `Q` ajustada | Fracción removida | Fracción que sobrevive |
|---|---:|---:|---:|---:|
| `\|r\|` | 3,139×10⁷ | 3,180×10⁷ | **−1,3%** | 101,3% |
| `log_hl` | 1,028×10⁸ | 1,123×10⁸ | **−9,2%** | 109,2% |

**La fracción removida es negativa en ambas variables**: el ajuste por `s(m)` no solo no hizo colapsar la dependencia, sino que la energía de dependencia (a 4 horas) es ligeramente **mayor** después de ajustar. En otras palabras: el reloj no explica el agrupamiento de volatilidad de MNQ — casi toda la dependencia observada es dinámica genuina, no un patrón horario repitiéndose.

Como diagnóstico adicional, independiente y explícitamente **distinto** de una ACF continua (ver §11), se comparó el valor de `|r|`/`log_hl` en un mismo minuto del reloj entre un día de negociación y el siguiente día de negociación presente en los datos ("same-clock-position"):

| Variable | Crudo `rho` | Ajustado `rho` | IC 95% (ajustado) | `n` pares día-a-día |
|---|---:|---:|---|---:|
| `\|r\|` | 0,311 | 0,226 | [0,205, 0,246] | 1.419 pares de días |
| `log_hl` | 0,565 | 0,414 | [0,384, 0,442] | 1.419 pares de días |

También aquí la dependencia sobrevive con claridad al ajuste (0,23–0,41, intervalos que excluyen cero por un margen amplio) — un mismo minuto del reloj sigue siendo informativo de un día para el otro, más allá de lo que el reloj por sí solo explicaría.

## 11. El problema del rezago de una jornada (1.380 minutos)

Bajo la topología de no-cruce vigente (TDA-08 §1), ningún bloque continuo del conjunto de investigación alcanza 1.380 minutos — el rezago de "una jornada completa" es `NOT_ESTIMABLE` bajo el motor continuo de ACF (`TDA09_portmanteau.csv`: filas `m=1.380`/`2.760` no existen como estimables porque exceden el máximo estimable real). Siguiendo la opción (B) del roadmap, se implementó el diagnóstico **separado** de §10 ("same-clock-position entre jornadas") — nunca se llamó a eso "ACF lag 1.380": es un objeto matemático distinto (correlación entre pares de días consecutivos presentes, no una ACF de denominador continuo).

## 12. Persistencia por año

Rezago 1, `|r|` y `log_hl`, crudos y ajustados, en los 5 años completos (2020–2024):

| Año | `rho(\|r\|)` crudo | `rho(\|r\|)` ajustado | `rho(log_hl)` crudo | `rho(log_hl)` ajustado |
|---:|---:|---:|---:|---:|
| 2020 | 0,447 | 0,439 | 0,772 | 0,741 |
| 2021 | 0,388 | 0,312 | 0,741 | 0,647 |
| 2022 | 0,350 | 0,255 | 0,697 | 0,565 |
| 2023 | 0,379 | 0,250 | 0,735 | 0,592 |
| 2024 | 0,378 | 0,319 | 0,731 | 0,662 |

**Estable y positivo en los 5 años**, sin ningún cambio de signo — a diferencia de la dependencia en la MEDIA (TDA-08), donde el signo sí cambiaba entre años. El agrupamiento de volatilidad es, hasta ahora, la propiedad más estable que se ha encontrado en todo el roadmap.

## 13. Persistencia por segmento

Rezago 1, en los 7 tramos horarios de TDA-06 — fuerte y positivo en los 7, sin ningún colapso:

| Segmento | `rho(\|r\|)` ajustado | `rho(log_hl)` ajustado |
|---|---:|---:|
| 00:00–02:00 | 0,431 | 0,733 |
| 02:00–03:00 | 0,377 | 0,705 |
| 03:00–08:30 | 0,363 | 0,697 |
| 08:30–09:30 | 0,350 | 0,651 |
| 09:30–16:02 | 0,349 | 0,729 |
| 16:02–20:00 | 0,445 | 0,724 |
| 20:00–24:00 | 0,408 | 0,720 |

Sin excepciones: cada tramo del día, por separado, ya muestra agrupamiento de volatilidad claro.

## 14. Resultado ARCH-LM (Engle)

Diagnóstico complementario (nunca la única evidencia), órdenes predeclarados 1/5/20, calibrado empíricamente por permutación por minuto (60 réplicas, presupuesto reducido documentado — cada réplica exige una regresión OLS completa):

| Serie | Orden | `n` efectivo | `LM` | `R²` | Percentil en el null |
|---|---:|---:|---:|---:|---:|
| `r_1m` | 1 | 1.911.434 | 43.711 | 2,29% | 100,0 |
| `r_1m` | 5 | 1.900.601 | 92.639 | 4,87% | 100,0 |
| `r_1m` | 20 | 1.866.492 | 123.041 | 6,59% | 100,0 |
| `r_tilde` | 1 | 1.911.434 | 163.540 | 8,56% | 100,0 |
| `r_tilde` | 5 | 1.900.601 | 282.220 | 14,85% | 100,0 |
| `r_tilde` | 20 | 1.866.492 | 339.126 | 18,17% | 100,0 |

El `LM` real supera el percentil 97,5 del null en los 6 casos, por un margen enorme (con `n~1,9×10⁶` esto era esperable — el `R²`, entre 2% y 18%, es la magnitud interpretable: una fracción moderada pero nada despreciable de la varianza de `r_t²` se explica por sus propios rezagos). Interesante: `r_tilde` (ajustado) muestra un `R²` **mayor** que `r_1m` crudo en los 3 órdenes — consistente con §10: el ajuste no diluye la dependencia, la vuelve más nítida.

## 15. Resultado de la calibración G2

- **Null 1 (permutación dentro de `minute_of_day`) — PRINCIPAL**: el `rho` real supera el percentil 97,5 del null en el 100% de los rezagos evaluados (1 a 60), tanto para `|r|` y `log_hl` crudos como ajustados (§7/§9, `TDA09_g2_calibracion_null1_principal.csv`). El pipeline no fabrica esta magnitud sobre ruido con estructura de reloj realista.
- **Null global (permutación simple, secundario)** — solo para las series ajustadas, justificado por el diagnóstico de aplanamiento (§5): bandas aún más estrechas que el Null 1 (p.ej. `|r_tilde|` rezago 1: banda 0,0015 vs. real 0,398) — misma conclusión, con más margen todavía.
- **Null sintético (heredado de TDA-08, `Null 2`) — FALLIDO, EXCLUIDO de la inferencia principal**: se reevaluó sobre `|r|` con el mismo diagnóstico de momentos de TDA-08 (`TDA09_g2_diagnostico_momentos_sintetico.csv`). Resultado: varianza ~50% mayor que la real, curtosis en exceso ~2,1 veces mayor — **no pasa** sus propios criterios predeclarados (solo el perfil de escala por minuto, 0,997, sí se preserva). Se documenta como `SENSIBILIDAD FALLIDA / DIAGNÓSTICO`, exactamente como en TDA-08, y **no participa** de ninguna conclusión de esta etapa. No se construyó un null sintético propio para `log_hl` (G4): el null por permutación ya es suficiente para calibrar la inferencia principal.

## 16. Resultado TH19

**RESUELTA — VOLATILITY CLUSTERING DETECTABLE.**

- `ACF(|r|)` es órdenes de magnitud mayor y muchísimo más persistente que `ACF(r)` (0,42 vs. 0,006 en el rezago 1; sigue en 0,17 a las 10 horas mientras `ACF(r)` nunca se aleja de 0).
- `ACF(r²)` confirma la misma historia cualitativa pero dominada por más ruido (valores más chicos y menos estables) — no se usa como proxy principal.
- `log_hl` confirma y **refuerza** la historia — es el proxy más fuerte y persistente de los tres.
- La dependencia **sobrevive** al ajuste por `s(m)` casi por completo (de hecho, ligeramente amplificada en la energía a 4 horas y en el `R²` del LM de Engle).
- La memoria dura, de forma medible y por encima del null, al menos **10 horas** (600 minutos, el máximo rezago evaluado) — no se pudo determinar dónde termina exactamente porque no se buscó más allá del rango predeclarado.

## 17. Resultado TH20

**RESUELTA (habilitada).** Comparación log-log vs. semi-log sobre `ACF(|r_tilde|)` y `ACF(log_hl_tilde)` (600 puntos cada una):

| Variable | `R²` log-log | `R²` semi-log | Pendiente log-log | Conclusión descriptiva |
|---|---:|---:|---:|---|
| `\|r_tilde\|` | 0,9498 | 0,9047 | −0,128 | más compatible con log-log |
| `log_hl_tilde` | 0,9715 | 0,8663 | −0,105 | más compatible con log-log |

Ambas series ajustan mejor con una recta en escala log-log que en semi-log, y la pendiente es muy chica en valor absoluto (~−0,10 a −0,13) — es decir, el decaimiento es **muy lento**. Esto es **descriptivo**, no una prueba de "memoria larga": Tsay/roadmap advierten explícitamente que una persistencia muy lenta también puede deberse a cambios de nivel no modelados (p.ej. 2022 fue un año de volatilidad de mercado más alta, TDA-06 ya lo documentó). **Esta ambigüedad queda explícitamente abierta para TDA-14** — no se estimó ningún parámetro de memoria larga (`d`) ni se ejecutó differencing fraccional.

## 18. Resultado TH21

**RESUELTA — CLUSTERING GENUINO (dominante).**

La fracción de dependencia removida por el reloj es **negativa** para `|r|` (−1,3%) y para `log_hl` (−9,2%) a una ventana de 4 horas — el ajuste por `s(m)` no elimina el agrupamiento de volatilidad; si acaso, lo vuelve algo más nítido. El diagnóstico complementario de "mismo minuto, día siguiente" confirma dependencia sustancial (0,23–0,41) que sobrevive al ajuste. La persistencia es estable en los 5 años completos y en los 7 segmentos horarios (§12/§13), sin ninguna excepción. No se observa evidencia de que el patrón de reloj de TDA-06 explique una fracción material del agrupamiento observado.

## 19. STOP-9

**NO ACTIVADO.** Ninguna de las dos variables de magnitud (`|r|`, `log_hl`) muestra un colapso de dependencia al ajustar (§10/§18) — de hecho, ambas muestran fracción removida negativa, muy lejos del umbral predeclarado de colapso (90%). Se continúa: la etapa **no** salta TDA-11 por este motivo (aunque TDA-11 sigue siendo condicional y no se ejecuta en esta tarea).

## 20. Estabilidad — G3

Cubierta en §12 (por año) y §13 (por segmento) para `|r|` y `log_hl`, crudo y ajustado, en el rezago 1 de una grilla predeclarada (1, 5, 20, 60) — tabla completa en `TDA09_persistencia_por_anio.csv`/`TDA09_persistencia_por_segmento.csv`. Es, con diferencia, la propiedad más estable encontrada en el roadmap hasta ahora: positiva y de magnitud similar en absolutamente todos los años y segmentos evaluados, sin ningún cambio de signo.

## 21. Archivos creados/modificados

**Código nuevo:**
- `src/ohlcv_dataroad/ingest/tda09_volatility_clustering.py`
- `src/ohlcv_dataroad/ingest/run_tda09.py`
- `tests/test_tda09_volatility_clustering.py` (36 tests)

**Código modificado (aditivo, sin tocar TDA-00…TDA08-H):**
- `src/ohlcv_dataroad/config.py` (sección `tda09_*`, añadida)
- `configs/mnq_snapshot.yaml` (sección `tda09`, añadida)

**Artefactos generados (`reports/mnq/`):**
- `TDA09_volatility_clustering.md` (este informe)
- `TDA09_acf_magnitud.csv` — variable × raw/adjusted × lag × rho × beta × n_pairs × estimable (hasta 600 minutos)
- `TDA09_bootstrap_ci.csv` — IC 95% bootstrap en rezagos clave + diagnóstico "same-clock-position"
- `TDA09_clock_attribution.csv` — `Q` cruda/ajustada y fracción removida/sobrevive (TH21)
- `TDA09_persistencia_por_anio.csv` / `TDA09_persistencia_por_segmento.csv`
- `TDA09_portmanteau.csv` — `Q(m)` con `calibration_status`
- `TDA09_arch_lm.csv`
- `TDA09_g2_calibracion_null1_principal.csv` (null PRINCIPAL) / `TDA09_g2_calibracion_secundaria_global_ajustada.csv` (null secundario, solo ajustadas) / `TDA09_g2_diagnostico_momentos_sintetico.csv` (null sintético FALLIDO)
- `TDA09_sensibilidad_remocion_media.csv` / `TDA09_diagnostico_aplanamiento_reloj.csv`
- `TDA09_acf_triple.png`, `TDA09_acf_raw_vs_adjustado.png`, `TDA09_decay_loglog_semilog.png`

**Nota menor de nomenclatura**: las columnas del null secundario (`TDA09_g2_calibracion_secundaria_global_ajustada.csv`) se llaman `null1_*` porque reutilizan literalmente la función `g2_null1_calibration_summary` (nombre genérico, no específico de qué null recibe) — la columna `null_type="global_permutation_secondary"` desambigua sin ambigüedad cuál null es. No afecta ningún resultado.

**No se modificó** ningún artefacto de TDA-00…TDA08-H. Holdout permanece `LOCKED`. TDA-10/TDA-11 no se iniciaron.

## 22. Tests específicos

```
python -m pytest -q tests/test_tda09_volatility_clustering.py
```
**Resultado: 36 passed.** Cobertura: topología (ninguna ACF cruza `trading_date`/gaps/roll — heredado sin modificar de TDA-08, más verificación de que `build_log_hl_population` incluye filas sin `r_1m` válido correctamente); `block_relative_position`/`engle_lm_statistic` nunca fabrican una cadena que cruce un bloque; serie IID (ARCH-LM) vs. ARCH genuino sintético (`R²`/`LM` claramente mayor); `|r|`/`r²` invariantes al signo; `clock_profile_flatness` y G2 global reproducibles; `same_clock_next_trading_day` verificado a mano y sin fabricar pares entre días no consecutivos; `dependence_energy`/`clock_attribution`/`decide_stop9` con escenarios de colapso y de supervivencia; `decay_form_diagnostic` distingue correctamente un decaimiento polinomial sintético de uno exponencial sintético; `mean_removal_sensitivity` no fabrica vecinos al eliminar filas; `verify_s_m_is_log_hl_proxy`/`build_log_hl_tilde` (proxy incorrecto, protección de división por cero); holdout nunca abierto (3 tests); persistencia exacta de artefactos, sin nombres obsoletos; y 4 escenarios extremo-a-extremo sobre datos sintéticos completos: serie IID → ACF de magnitud cercana a cero; clustering genuino (GARCH-like) → `ACF(|r|)` claramente positiva; patrón de reloj puro sin dinámica → colapsa sustancialmente al ajustar; clustering genuino + patrón de reloj → sobrevive al ajuste.

## 23. Suite completa

```
python -m pytest -q
```
**Resultado: 394 passed** (358 previas a TDA-09, incluidas las 50 de TDA-08 y las 11 de TDA08-H — confirmadas intactas — + 36 nuevas de esta etapa).

## 24. Pipeline real

```
python -m ohlcv_dataroad.ingest.run_tda09 --config configs/mnq_snapshot.yaml
```
**Completado en 3.702,5 s (≈61,7 minutos)**, un solo proceso. Más lento que TDA-08 (~17 min) porque esta etapa calcula ACF hasta 600 rezagos (vs. 2.760 pero con escape temprano) sobre 8 series de magnitud/dirección, además de calibración G2 (rango contiguo 1–60 completo, no disperso) sobre 4 series y el LM de Engle calibrado empíricamente (60 réplicas × 3 órdenes × 2 series, cada una con una regresión OLS completa sobre ~1,9 millones de filas). Verificado tras la corrida: los 14 artefactos declarados existen; ningún `TDA09_*` obsoleto; TDA-00…TDA08-H sin cambios inesperados (verificado con `git status`, solo aparecen los archivos nuevos de esta etapa); holdout sigue `LOCKED`; TDA-10/TDA-11 no iniciadas.

## 25. Estado final

**`PASS_WITH_OPEN_QUESTIONS`**

- **TH19 = RESUELTA — `VOLATILITY CLUSTERING DETECTABLE`.**
- **TH20 = RESUELTA (habilitada) — decaimiento lento, más compatible con log-log que semi-log; ambigüedad memoria-larga-vs-cambios-de-nivel explícitamente abierta para TDA-14.**
- **TH21 = RESUELTA — `CLUSTERING GENUINO` (dominante; el reloj no explica una fracción material del agrupamiento observado).**
- **STOP-9 = `NO ACTIVADO`.**

## 26. Preguntas abiertas

1. **Forma exacta del decaimiento (TH20)**: el ajuste log-log es mejor que el semi-log, pero esto no distingue por sí solo entre memoria larga genuina y cambios de nivel no modelados (p.ej. 2022 como año de mayor volatilidad de mercado). Corresponde a TDA-14.
2. **El rango evaluado se detiene en 600 minutos (10 horas)**: la dependencia seguía siendo claramente positiva (0,17–0,20) en el borde del rango — no se sabe si continúa más allá; no se investigó por estar fuera del rango predeclarado.
3. **Nomenclatura de columnas del null secundario** (§21): cosmética, no afecta ningún resultado.
4. **Null sintético de TDA-08 (Null 2) sigue sin ser representativo** (§15) — ya documentado como fallido en TDA-08; se reconfirma aquí sobre `|r|`, no se investiga más a fondo (fuera de alcance).

Ninguna es bloqueante para continuar.

## 27. Recomendación para el siguiente paso

- TDA-10 (escala vs. forma de las colas) tiene ahora una base sólida: existe dinámica de volatilidad genuina y persistente que estandarizar puede explicar una fracción real del exceso de curtosis de TDA-07. Ejecutar TDA-10 en su versión completa (no la mínima de STOP-9, que no se activó).
- Cualquier futura consideración de TDA-11 (modelo paramétrico de volatilidad) debe partir de que la persistencia es MUY estable entre años/segmentos — una ventaja para cualquier modelo simple (EWMA, ventana rodante) frente a uno complejo.
- La ambigüedad de TH20 (§26.1) debe resolverse en TDA-14, no antes.
- Ningún resultado de esta etapa habilita ni sugiere features, señales o backtesting — esa es una fase separada y posterior.

---

## Nota administrativa (no metodológica)

Durante esta tarea se detectó un commit de git (`ab38916 "TDA-09-v0"`) que no fue creado explícitamente por el asistente mediante un comando `git commit` — apareció en el historial local después de escribir los archivos de código de esta etapa, antes de ejecutar el pipeline real. No se encontró ningún hook de Claude Code (`.claude/settings*.json`, local o global) que lo explique. Se deja constancia para que el usuario lo revise (posible automatización externa de su entorno) — no se intentó deshacer ni investigar más a fondo, siguiendo la política de no tomar acciones destructivas sobre el historial de git sin instrucción explícita.

---

## Modo sencillo — en 10 líneas

**¿La volatilidad del MNQ tiene memoria?** Sí, y muy fuerte: si el minuto pasado fue agitado, el siguiente minuto (y la hora siguiente, y varias horas después) tiende a seguir siendo más agitado de lo normal.

**¿Cuánto dura?** Al menos 10 horas — es el límite hasta donde se midió, no necesariamente donde termina.

**¿Es solo la hora del día o queda algo más?** Casi todo es "algo más": después de quitar matemáticamente el patrón de "la apertura es más movida que la madrugada" (TDA-06), la memoria sigue prácticamente intacta. El reloj casi no explica nada de este fenómeno.

**¿Qué significa esto?** Que MNQ tiene rachas genuinas de calma y de agitación, no solo un horario previsible.

**¿Qué NO significa?** No dice hacia dónde va a moverse el precio (eso ya se estudió en TDA-08 y sigue siendo casi impredecible), no es una señal de trading, y no valida todavía ningún modelo GARCH — solo confirma que existe algo que un modelo de volatilidad, más adelante, podría intentar capturar.
