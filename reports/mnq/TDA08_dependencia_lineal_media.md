# TDA-08 — Dependencia lineal en la media

**Etapa:** `docs/methodology/Tsay_OHLCV_analysis_roadmap.md` § TDA-08
**Depende de:** TDA-04, TDA-05, TDA-06, TDA-07 (todas `PASS_WITH_OPEN_QUESTIONS`), complemento TH10 (`RESUELTA / CLOSED`).
**Alcance de datos:** exclusivamente `tda04_variables_1m.parquet`, `tda04_return_validity_mask.parquet`, `tda06_r_tilde.parquet` y `TDA06_segmentacion_propuesta.csv`. Ningún archivo de `data/raw/` ni de `holdout_files` fue abierto.
**Evidencia reproducible (17 artefactos):** `TDA08_acf_global.csv`, `TDA08_pacf.csv`, `TDA08_bootstrap_ci.csv`, `TDA08_portmanteau.csv`, `TDA08_g2_calibracion_null1_principal.csv` (PRINCIPAL — ver §11.1), `TDA08_g2_portmanteau_null1_principal.csv` (PRINCIPAL), `TDA08_g2_calibracion_combinada_secundaria.csv` (SECUNDARIO — ver §11.3), `TDA08_g2_portmanteau_combinado_secundario.csv` (SECUNDARIO), `TDA08_g2_diagnostico_momentos.csv`, `TDA08_traduccion_ticks.csv`, `TDA08_rho1_multi_frecuencia.csv`, `TDA08_acf_por_grupo.csv` (año/segmento/decil de volumen/mes en una sola tabla larga), `TDA08_decil_volumen_sensibilidad.csv`, `TDA08_ventanas_tda06.csv`, `TDA08_ventanas_por_anio.csv`, `TDA08_ventanas_minutos_adyacentes.csv`, `TDA08_acf_bartlett.png`. Todo generado por `python -m ohlcv_dataroad.ingest.run_tda08`.

> Esta etapa mide la dependencia LINEAL de `r_1m` con su propio pasado y la somete OBLIGATORIAMENTE a un protocolo de refutación microestructural antes de llamarla "dependencia". NO estudia volatility clustering en `|r|`/`r²` (TDA-09), NO ajusta ARCH/GARCH, NO ejecuta EVT ni BDS, NO crea features/targets/señales, NO abre el holdout, y NO modifica ningún artefacto de TDA-00…TDA-07 ni del complemento TH10.

---

## 0. Corrección metodológica (`REVIEW_REQUIRED` → esta versión)

Una revisión externa detectó **cuatro problemas de fondo** en la primera versión de TDA-08 (código, tests y resultados numéricos consecuentes). Se corrigieron todos antes de reevaluar el estado de la etapa. Resumen diagnóstico:

| # | Problema detectado | Causa raíz | Corrección adoptada |
|---|---|---|---|
| 1 | `compute_acf` atenuaba `rho_k` mecánicamente | El numerador/denominador se normalizaban por `T` (población COMPLETA), convención heredada del HAC de TDA-07 sin verificar que aplicara a un objeto distinto (ACF ≠ HAC) | Correlación de Pearson **pairwise-complete**: numerador y denominador restringidos a los mismos pares válidos del rezago `k`; garantizado en `[-1,1]` por Cauchy-Schwarz sobre ESE conjunto; no depende mecánicamente de cuántos pares faltan (test dedicado `test_acf_multiblock_does_not_attenuate_known_ar1_dependence`) |
| 1b | `rho_1` y la pendiente de regresión se trataban como si fueran lo mismo | Nunca se demostró la igualdad, solo se presupuso | `beta_k` (pendiente OLS `Cov(r_t,r_{t-1})/Var(r_{t-1})`) calculado y reportado por separado de `rho_k` en todas las tablas; la traducción a ticks usa `beta_1`, no `rho_1` |
| 2 | Rezagos 1.379-2.760 se trataban como "calculados" | Ningún bloque de continuidad alcanza esa longitud (`n_pairs=0`); el bug de Ljung-Box de la versión anterior convertía ese `NaN` en aporte cero silencioso, produciendo `Q(1.380)=Q(2.760)` idénticos como si fueran resultados válidos | `estimable=False` explícito desde el primer rezago sin pares; `compute_portmanteau_q` devuelve `NOT_ESTIMABLE` (no un número) para `m` que exige rezagos sin pares; se reporta además el `m=max_estimable_lag` real (1.378) |
| 3 | Decil de volumen (y, por el mismo motor, año/segmento) exigía implícitamente `decile_t == decile_{t-1}` | El código filtraba filas por grupo y volvía a calcular bloques de continuidad SOBRE EL SUBCONJUNTO — una condición mucho más restrictiva que "el par es temporalmente válido" | `acf_by_group`: los pares se construyen UNA VEZ sobre la topología completa; cada par se etiqueta por `group_labels` de la fila `t` (la más reciente); se agrupan los PARES, no las filas. El método anterior se conserva SOLO como sensibilidad secundaria explícita (`acf_by_group_strict_same_block`), únicamente para decil de volumen |
| 4 | Los dos nulls de G2 no eran representativos | Null 1 permutaba GLOBALMENTE (destruía también la heterogeneidad por `minute_of_day` que TDA-06 documentó); Null 2 usaba ruido gaussiano i.i.d. con sigma global (no preservaba ni el perfil de escala `s(m)` ni la curtosis empírica de TDA-07) | Null 1: permutación DENTRO de cada `minute_of_day` (preserva el perfil de reloj, destruye la dependencia serial). Null 2: remuestreo con reemplazo de la distribución empírica de `r_tilde`, reescalado por `s(m)` en la posición de destino — diseñado con la **intención** de preservar forma/curtosis real además del perfil de escala intradía. **Nota de auditoría (estado vigente, no de esta 1a revisión)**: esa intención se sometió a diagnóstico explícito en la 2a revisión (§11.2) y se comprobó que Null 2 NO preserva varianza ni curtosis reales (solo el perfil de escala) — por eso quedó excluido de la inferencia PRINCIPAL de G2 desde la 3a revisión (§11). No leer esta fila como una afirmación vigente sobre Null 2 |

Además: portmanteau re-derivado para la correlación pairwise (`Q(m) = Σ n_pairs_k·rho_k²`, ponderado por el `n_pairs_k` propio de cada rezago, nunca por un `T` fijo); se persiste TODA la evidencia declarada (al cierre de ESTA 1a revisión: 14 artefactos, antes 10 — el conteo vigente es distinto, ver cabecera del informe y §5); la magnitud de las ventanas se cuantifica explícitamente POR BARRA; y se corrigen los siguientes datos factuales del informe anterior: "4 de 5 años completos positivos, 2023 negativo" (no "5 de 5 salvo 2023" — 2019 es año PARCIAL, no cuenta); 67 meses en la tabla rodante (no 66); "200+200=400 réplicas totales combinadas" (no "400+400"); "volumen/actividad" en vez de "liquidez" en todo el texto; no se repite "casi monótono" para el decil de volumen (la cifra corregida NO es monótona — ver §8).

**No se inició TDA-09. No se modificó ningún artefacto de TDA-00…TDA-07 ni de TH10. No se abrió el holdout.**

## 0b. Segunda corrección metodológica — centrado del estimador

Una segunda revisión, posterior a la anterior, detectó que aunque `compute_acf` ya excluía correctamente los pares que cruzan una frontera de bloque ("pairwise"), el **centrado** (`e = x - media`) seguía usando la media **GLOBAL** de toda la población, calculada una sola vez, en vez de la media del propio conjunto de pares de cada rezago — lo cual no es Pearson pairwise-complete literal (el nombre que el informe ya usaba). El problema era más grave en `acf_by_group` (año/segmento/decil de volumen/mes): centrar TODOS los grupos por la misma media global puede fabricar dependencia serial artificial cuando dos grupos tienen medias genuinamente distintas, incluso si dentro de cada grupo no existe ninguna dependencia real.

| # | Problema detectado | Corrección adoptada |
|---|---|---|
| 1 | `compute_acf` centraba por la media global, no por la media de cada conjunto de pares `P_k` | Se decidió explícitamente la opción **(A)**: Pearson pairwise-complete literal — cada lado del par (`t` y `t-k`) se centra por SU PROPIA media sobre `P_k` únicamente (mismo comportamiento que `cor(x,y,use="pairwise.complete.obs")` en R). Se descartó la opción (B) (mantener una media común bajo un supuesto de media estacionaria) porque no correspondía al nombre ya declarado y porque es la opción que evita el problema del punto 3 |
| 2 | `beta_k` no era automáticamente la pendiente OLS con intercepto sobre `P_k` | Con el centrado local del punto 1, `beta_k = Σe_t·e_{t-k}/Σe_{t-k}²` (mismas medias locales que `rho_k`) ES, literalmente, la pendiente OLS con intercepto sobre `P_k` — se separó explícitamente de cualquier estimador de media común en el docstring |
| 3 | `acf_by_group` centraba TODOS los grupos por la media GLOBAL de la población, pudiendo fabricar dependencia artificial a partir de diferencias de medias ENTRE grupos | Cada grupo calcula ahora sus propias `mean_t_g`/`mean_tk_g` (sobre los pares de ESE grupo únicamente) antes de construir `e_t`/`e_tk` — exactamente `Corr(r_t, r_{t-1} \| group_t=g)` estimado sobre la muestra condicionada. Verificado con un test adversarial obligatorio (`test_acf_by_group_does_not_inject_spurious_dependence_from_between_group_mean_differences`): dos grupos con medias ±5 (independencia real dentro de cada grupo) — el estimador corregido da `rho≈0` en ambos; la versión centrada por media global da `rho≈-0,96` (dependencia fabricada) |
| 4 | `bootstrap_rho` heredaba el mismo centrado por media global dentro de cada réplica | Reescrito con el mismo centrado local por-par que `compute_acf`, aplicado dentro de cada réplica bootstrap |
| 5 | Bartlett/PACF se presentaban sin advertir que son fórmulas clásicas para una serie continua de denominador fijo, no para la topología bloqueada de esta etapa | Docstrings reforzados: diagnóstico clásico APROXIMADO, nunca el intervalo principal — el bootstrap por `trading_date` es siempre la inferencia principal |
| 6 | El informe afirmaba que el null sintético "preserva volatilidad/curtosis comparable" sin un test que lo comprobara | Implementada la opción (A) preferida por la revisión: `g2_synthetic_null_moment_check` compara UNA realización del null sintético contra la serie real en tres criterios predeclarados (varianza, curtosis en exceso, perfil de escala por minuto vía MAD robusta) — ver §11.2, el resultado sobre datos reales es honesto y **no** confirma la afirmación en dos de los tres criterios. Esto motivó, en una revisión posterior (3a), excluir Null 2 de la inferencia PRINCIPAL de G2 — ver §11 |
| 7 | `Q(1.378)` se presentaba junto a `Q(10..60)` sin distinguir que G2 solo calibra empíricamente hasta rezago 60 | `annotate_portmanteau_calibration` añade `calibration_status` (`G2_CALIBRATED` / `DESCRIPTIVE_UNCALIBRATED` / `NOT_ESTIMABLE`) a cada fila de la tabla de portmanteau — ver §7 |
| 8 | El test de persistencia solo comprobaba campos de `TDA08Result`, no que el runner escribiera los archivos | `persist_artifacts` extraído a función propia en `run_tda08.py`; nuevo test `test_persist_artifacts_writes_exactly_the_declared_paths_and_nothing_obsolete` ejecuta la persistencia sobre una fixture pequeña y confirma que los artefactos declarados existen y que no queda ningún `TDA08_*` obsoleto en el directorio (15 en el cierre de esta 2a revisión; el test usa `ARTIFACT_PATH_ATTRS` como fuente de verdad, así que sigue siendo válido con el conteo vigente — 17, ver cabecera) |

**Resultado del recálculo**: sobre el conjunto de investigación real, el impacto numérico del centrado local es **prácticamente nulo** para la ACF global (los pares excluidos por frontera de bloque son una fracción pequeña de la población salvo en la cola de rezagos largos) y también resultó **numéricamente insignificante** para año/segmento/decil de volumen — no porque el punto 3 no fuera un bug real (el test adversarial lo demuestra con nitidez, `rho` fabricado ≈-0,96 en un escenario de medias muy separadas), sino porque en los datos reales de MNQ las diferencias de media ENTRE grupos (años, segmentos horarios, deciles de volumen) son minúsculas frente a la volatilidad DENTRO de cada grupo (a diferencia del escenario adversarial, diseñado deliberadamente con medias separadas por 5 desviaciones estándar). Ver comparación completa en §19.

**Sí hubo un hallazgo sustantivo nuevo**: el diagnóstico de momentos del null sintético (punto 6) reveló que, sobre datos reales, el null 2 tiene una curtosis en exceso muy superior a la real (401 vs 112) y una varianza ~37% mayor — la afirmación "preserva volatilidad/curtosis comparable" era **falsa** para el conjunto de investigación real, aunque el perfil de escala por minuto sí se preserva con gran fidelidad (correlación 0,997). Ver §11.2 para la interpretación. **Nota de cierre (3a revisión, posterior a este párrafo)**: este hallazgo motivó excluir Null 2 de la inferencia PRINCIPAL de G2 por completo — ver §11.

**No se inició TDA-09. No se modificó ningún artefacto de TDA-00…TDA-07 ni de TH10. No se abrió el holdout.**

## 1. Qué se revisó

Roadmap (TDA-08 completo, gobernanza G0–G6, STOP-8a/8b), backlog (TH16, TH17, TH18), informes finales de TDA-05, TDA-06, TDA-07 y del complemento TH10. Configuración vigente de MNQ (`configs/mnq_snapshot.yaml`). No se reabrió TDA-00…TDA-03.

**Conflicto documental (sin cambios respecto a la versión anterior)**: el roadmap genérico menciona confirmación por holdout; la gobernanza vigente mantiene el holdout MNQ `LOCKED` (frontera `2025-06-23 00:00:00 UTC`). Se mantuvo `LOCKED`: TDA-08 trabaja exclusivamente sobre `research_files`.

**Conflicto nuevo, hecho explícito por esta corrección — "≥2 jornadas" (roadmap) vs. gobernanza de no-cruce**: la grilla aspiracional pedía rezagos hasta `2×SESSION_MINUTES=2.760` ("dos jornadas"). Bajo las reglas de no-cruce vigentes desde TDA-02/07 (nunca se fabrica continuidad cruzando `trading_date`, huecos internos o el break secundario pre-2021), **ningún bloque de continuidad del conjunto de investigación alcanza 1.379 minutos seguidos** — el rezago genuinamente estimable máximo es **1.378**, dos minutos por debajo de una sola jornada nominal (1.380). Esto NO es un error de cómputo: es una propiedad estructural de los datos (huecos cortos + el break de TDA-02), confirmada por el nuevo motor (que ahora lo declara `NOT_ESTIMABLE` en vez de rellenarlo silenciosamente). **No se propone relajar la regla de no-cruce para alcanzar el objetivo de "2 jornadas"** — hacerlo violaría los invariantes de topología establecidos desde TDA-02/07. Se recomienda anotar el roadmap (fuera del alcance de autoridad de esta tarea) para que futuras etapas que reutilicen "rezagos largos" partan de este techo estructural (1.378), no de 2.760.

## 2. Diseño corregido de TDA-08

- **Grilla de rezagos completa**: `MAX_LAG_REQUESTED=2.760` sigue siendo el objetivo ASPIRACIONAL de la grilla solicitada; el máximo rezago REALMENTE estimable se calcula y reporta aparte (`max_estimable_lag_raw`, ver §1).
- **Rezagos cortos** para año/segmento: `SHORT_MAX_LAG=20`; decil de volumen: `DECILE_MAX_LAG=5`.
- **PACF**: `PACF_MAX_LAG=40`, diagnóstico de rezago bajo solamente.
- **Portmanteau**: `m ∈ {10, 20, 40, 60, 1.380, 2.760}` + fila adicional en `m=max_estimable_lag` real.
- **Bootstrap** (G5): bloques de jornada con clave compuesta, `n_boot=300`, semilla fija=0, rezagos `(1,2,5,10,20)`; devuelve `rho` y `beta` por separado.
- **G2** (calibración): null 1 = permutación DENTRO de `minute_of_day`; null 2 = remuestreo empírico de `r_tilde` reescalado por `s(m)`; `n_perm=200` cada uno, semillas fijas (1 y 2), `200+200=400` réplicas combinadas, calibrado hasta rezago 60.
- **Deciles de volumen**: GLOBALES, del volumen de la barra `t`; análisis PRINCIPAL condiciona solo por `decile_t` (no exige `decile_{t-1}`); sensibilidad secundaria con la convención estricta anterior.
- **1→5→10 minutos**: no solapado (TH10), sin cambios respecto a la versión anterior (no dependía del bug de agrupación).
- **Ventanas**: las dos heredadas de TDA-06 (09:31-09:35, 15:52-16:02 NY), con magnitud reportada explícitamente POR BARRA.

## 3. Qué se implementó

*(Snapshot histórico al cierre de la 2a revisión — la 3a y 4a añadieron mas funciones/artefactos; el estado vigente completo está en §20 y §23.)*

`src/ohlcv_dataroad/ingest/tda08_linear_mean_dependence.py` — motor con centrado LOCAL: `compute_acf` (pairwise-complete literal, cada lado del par centrado por su propia media sobre `P_k`, `rho`+`beta` separados, relleno `NOT_ESTIMABLE` sin recorrer rezagos vacíos), `compute_portmanteau_q` + `annotate_portmanteau_calibration` (nuevo — distingue `G2_CALIBRATED`/`DESCRIPTIVE_UNCALIBRATED`/`NOT_ESTIMABLE`; en la 4a revisión se añadió además `NOT_G2_CALIBRATED` para series sin null propio — ver §23), `bootstrap_rho` (clave compuesta + centrado local), `g2_permutation_null_by_minute` y `g2_synthetic_empirical_null` (dos nulls, sin cambios de esta revisión), `draw_synthetic_empirical_sample` + `g2_synthetic_null_moment_check` (nuevo — diagnóstico real de momentos del null 2), `acf_by_group` (centrado POR GRUPO, corrección central de esta revisión) y `acf_by_group_strict_same_block` (sensibilidad secundaria, auto-corregida al heredar el `compute_acf` corregido), `translate_dependence_to_ticks` (sin cambios, usa `beta_1`). `run_tda08.py` — `persist_artifacts` extraído como función propia, testeable; en este punto (2a revisión) persiste 15 artefactos (se añade `TDA08_g2_diagnostico_momentos.csv`) — el conteo VIGENTE es 17 (ver cabecera y §23). `config.py`/`configs/mnq_snapshot.yaml` — clave nueva `tda08_g2_moment_check_csv`.

## 4. Archivos modificados en esta corrección

*(Snapshot histórico de la 2a revisión — ver §20 y §23 para los archivos modificados en la 3a y 4a.)*

**Reescritos/editados a fondo:** `tda08_linear_mean_dependence.py` (`compute_acf`, `bootstrap_rho`, `acf_by_group` recentrados; `annotate_portmanteau_calibration`, `draw_synthetic_empirical_sample`, `g2_synthetic_null_moment_check` nuevos), `run_tda08.py` (persistencia extraída a `persist_artifacts`), `tests/test_tda08_linear_mean_dependence.py` (45 tests en este punto, 5 nuevos respecto a la 1a revisión).
**Editados:** `config.py` (campo/`@property`/`load_config` para `tda08_g2_moment_check_csv`), `configs/mnq_snapshot.yaml` (clave `g2_moment_check_csv`).
**Regenerados:** este informe + los 15 CSV/PNG (en este punto).
**No se modificó** ningún artefacto de TDA-00…TDA-07 ni del complemento TH10.

## 5. Tests ejecutados y resultado — histórico por revisión (ver §23 para el resultado FINAL vigente)

```
python -m pytest -q tests/test_tda08_linear_mean_dependence.py
python -m pytest -q
```

- `test_tda08_linear_mean_dependence.py`: **45 passed** (40 de la 1a revisión + 5 nuevos de esta 2a revisión). Categorías nuevas de esta revisión: **centrado pairwise literal** — `test_compute_acf_matches_pairwise_correlation_formula` corregido para centrar cada lado por su propia media sobre `P_k`; `test_compute_acf_pair_local_centering_differs_from_global_mean_centering_in_general` contrasta explícitamente contra el centrado por media global; **test adversarial obligatorio** `test_acf_by_group_does_not_inject_spurious_dependence_from_between_group_mean_differences` (dos grupos con medias ±5, independencia real dentro de cada uno — el estimador corregido da `rho≈0`, el centrado por media global fabrica `rho≈-0,96`); **diagnóstico de momentos G2** `test_g2_synthetic_null_moment_check_passes_for_correct_construction_and_fails_for_naive_gaussian` (el null correcto pasa los tres criterios, un gaussiano ingenuo con sigma global falla curtosis y perfil de escala); **calibración del portmanteau** `test_portmanteau_calibration_status_distinguishes_g2_calibrated_from_descriptive`; **persistencia del runner** `test_persist_artifacts_writes_exactly_the_declared_paths_and_nothing_obsolete` (ejecuta `persist_artifacts` sobre una fixture pequeña, confirma los 15 artefactos de ESE momento y ausencia de nombres obsoletos). Categorías de la 1a revisión sin cambios: topología temporal, atenuación multi-bloque, agrupación sin exigir `t-1` en el mismo grupo, nulls G2 que preservan reloj/escala, `NOT_ESTIMABLE` en rezagos largos, bootstrap con clave compuesta, protección de holdout.
- **Suite completa en este punto: 342 passed** (297 previas a TDA-08 + 45 de esta etapa). **Este NO es el conteo final** — ver §20 (3a revisión: 48 TDA-08 / 345 suite) y §23 (4a revisión / vigente: ver esa sección para las cifras reales).

## 6. Recursos computacionales

Sección nueva, exigida por la corrección (antes de reejecutar el análisis pesado): la disponibilidad de una máquina potente no amplía el alcance de TDA-08 — solo determina CÓMO se ejecuta la misma metodología, ya fijada en §0-§2.

**Hardware detectado** (`os.cpu_count()`, `numpy.show_config()`, WMI): Intel Core Ultra 9 275HX, **24 núcleos lógicos**, **≈31,4 GB RAM**, numpy 2.5.2 sobre OpenBLAS (`MAX_THREADS=24`, `DYNAMIC_ARCH`). **Sin GPU utilizable**: no hay `torch`/`cupy`/`jax` instalados — **no se usó GPU** (no se introdujo ninguna dependencia nueva solo para poder usarla).

**Benchmark previo a la 1a revisión** (array sintético de 1.900.000 filas, topología de bloques ~1.350 filas): `compute_acf` (`max_lag=2.760`) 17,5 s/serie; `g2_permutation_null_by_minute` (200 réplicas) 1,44 s/réplica (~288 s); `g2_synthetic_empirical_null` (200 réplicas) 1,50 s/réplica (~300 s); `bootstrap_rho` (5 rezagos, 300 réplicas) 0,16 s/réplica/serie (~96 s, 2 series). Total extrapolado ~750 s.

**Benchmark repetido tras el centrado local de esta 2a revisión** (mismo array sintético): `compute_acf` (`max_lag=2.760`) subió a **27,3 s/serie** (~55 s, 2 series) — el centrado por-par añade dos medias adicionales por rezago en vez de reutilizar un `e` precomputado una sola vez; `acf_by_group` (nuevo paso de `groupby.transform` para las medias por grupo) **0,78 s** para decil (5 rezagos, 10 grupos) y **3,09 s** para año (20 rezagos, 7 grupos) — coste adicional pequeño frente al resto del pipeline. Extrapolación total revisada: ~55 + 288 + 300 + 96 + ~10 (agrupaciones) + ~50 (multi-frecuencia) ≈ **~800-850 s** (~14 min), un solo hilo.

**Decisión adoptada (sin cambios respecto a la 1a revisión)**: el tiempo total, aunque algo mayor que antes, sigue siendo razonable para una corrida única de análisis (minutos, no horas) — **no se introdujo multiprocessing**. El centrado local es más costoso porque es la definición CORRECTA (no se puede volver a la convención de un único `e` precomputado sin reintroducir el problema de fondo de esta revisión) — nunca se optimiza sacrificando la corrección recién establecida. Vectorización sin cambios: `numpy` sobre arrays completos, sin bucles fila-a-fila; el único bucle Python explícito sigue siendo sobre réplicas (200-300), no sobre filas. Ninguna FFT sobre serie compactada.

**Corrida real**: `python -m ohlcv_dataroad.ingest.run_tda08 --config configs/mnq_snapshot.yaml`, **1.034,6 s (≈17,2 min)**, un solo proceso, semillas fijas sin cambios (bootstrap=0, permutación=1, sintético=2) — algo más lenta que la extrapolación del benchmark (probablemente por la carga adicional de las cuatro agrupaciones `acf_by_group` sobre la población completa: decil, año, segmento, mes — el benchmark solo midió decil y año por separado). Sigue siendo un incremento moderado (+37% de tiempo) frente a los 753,1 s de la 1a revisión, consistente con el coste extra medido del centrado local. No se midió RAM pico de forma instrumentada (no hay `psutil` instalado); no se observó degradación visible en una máquina con ~31 GB. No se introdujo ninguna ruta "optimizada" separada de la de referencia — mismo código en benchmark y corrida real.

## 7. Resultados TH16 — ACF/PACF/portmanteau/Bartlett/ticks

**Población**: 1.914.530 filas `r_1m` válidas (idéntico a TDA-04/07/versión anterior), mismo número para `r_tilde`.

**ACF global, primeros rezagos** (`r_1m` crudo, `rho` y `beta` reportados por separado):

| Rezago | `rho_k` | `beta_k` | `n_pairs` | Bartlett SE |
|---:|---:|---:|---:|---:|
| 1 | 0,005921 | 0,005915 | 1.911.434 | 0,000723 |
| 2 | −0,008445 | −0,008434 | 1.908.553 | 0,000723 |
| 3 | −0,001617 | −0,001613 | 1.905.799 | 0,000723 |
| 4 | 0,003939 | 0,003928 | 1.903.157 | 0,000723 |
| 5 | −0,004479 | −0,004464 | 1.900.601 | 0,000723 |

`rho_k` y `beta_k` prácticamente coinciden en estos rezagos (razón `beta_1/rho_1=0,999`) — consistente con varianza aproximadamente estacionaria dentro de cada conjunto de pares a rezago corto; **no se presupuso esta igualdad, se verificó** (a rezagos largos, con muchos menos pares, la razón se vuelve más ruidosa — ver `TDA08_acf_global.csv`).

**Cola de rezagos largos (1.370-1.385)**: `n_pairs` decae monótonamente (8.247 en 1.370 hasta 915 en 1.378) y colapsa a **0 en el rezago 1.379** — desde ahí, `estimable=False` para todo rezago mayor (`NOT_ESTIMABLE`, nunca un número). El **máximo rezago realmente estimable es 1.378** (ver §1, conflicto documental resuelto).

**Bartlett vs. bootstrap**: la banda de Bartlett es una referencia CLÁSICA APROXIMADA (fórmula derivada para una serie continua de denominador fijo, no para la topología bloqueada de esta etapa — §0b punto 5) — nunca el intervalo principal. Con `T~1,9×10⁶`, esa banda (±0,00142 en rezago 1) es extremadamente estrecha y trata cada barra como casi independiente. El bootstrap de bloques de jornada (respeta la dependencia intradía) es la inferencia PRINCIPAL y da:

| Rezago | `rho_k` | IC 95% bootstrap `rho` | `beta_k` | IC 95% bootstrap `beta` | ¿Excluye cero? |
|---:|---:|---|---:|---|---|
| 1 | 0,005921 | [−0,000284, 0,012177] | 0,005915 | [−0,000283, 0,012174] | No (por muy poco) |
| 2 | −0,008445 | [−0,012994, −0,003999] | −0,008434 | [−0,012971, −0,003996] | **Sí** |
| 5 | −0,004479 | [−0,008360, −0,000592] | −0,004464 | [−0,008333, −0,000591] | **Sí** |
| 10 | 0,003238 | [−0,001303, 0,007035] | 0,003225 | [−0,001301, 0,007003] | No |
| 20 | 0,003012 | [−0,001046, 0,007050] | 0,003010 | [−0,001046, 0,007047] | No |

`rho_1`/`beta_1` NO son distinguibles de cero bajo el bootstrap que respeta la dependencia intradía; rezagos 2 y 5 sí, por un margen pequeño. Conclusión sin cambios cualitativos respecto a la versión anterior.

**PACF** (40 rezagos, Durbin-Levinson): sin corte claro, magnitudes del mismo orden diminuto — sin cambios cualitativos.

**Portmanteau** (`Q(m) = Σ n_pairs_k·rho_k²`, corregido — nunca Ljung-Box clásico). Cada fila lleva un `calibration_status` explícito y **por serie** (`TDA08_portmanteau.csv` conserva la columna `series` — 2a revisión §0b punto 7; distinción por serie corregida en la 4a revisión, §11.4, tras detectar que `r_tilde` podía aparecer falsamente `G2_CALIBRATED`):

| `m` | `Q` (r_1m) | `calibration_status` (r_1m) | `Q` (r_tilde) | `calibration_status` (r_tilde) |
|---:|---:|---|---:|---|
| 10 | 378,18 | `G2_CALIBRATED` | 590,25 | `NOT_G2_CALIBRATED` |
| 20 | 445,43 | `G2_CALIBRATED` | 807,43 | `NOT_G2_CALIBRATED` |
| 40 | 612,14 | `G2_CALIBRATED` | 997,65 | `NOT_G2_CALIBRATED` |
| 60 | 718,99 | `G2_CALIBRATED` | 1.302,80 | `NOT_G2_CALIBRATED` |
| 1.380 | — | `NOT_ESTIMABLE` | — | `NOT_ESTIMABLE` |
| 2.760 | — | `NOT_ESTIMABLE` | — | `NOT_ESTIMABLE` |
| 1.378 (real) | 6.004,56 | `DESCRIPTIVE_UNCALIBRATED` | 7.171,08 | `NOT_G2_CALIBRATED` |

`r_1m` SÍ tiene calibración G2 propia (Null 1, §11.1): `m≤60` → `G2_CALIBRATED` (calibrado empíricamente); `m=1.378` → `DESCRIPTIVE_UNCALIBRATED` (calibrar G2 hasta ahí exigiría recalcular sobre las 200 réplicas de Null 1 hasta `m=1.378`, fuera del presupuesto de §6 — se persiste como magnitud descriptiva, NUNCA "significativa" contra G2); `m∈{1.380, 2.760}` → `NOT_ESTIMABLE` (sin pares). `r_tilde` NO tiene calibración G2 propia (§11.4) — todo `m` estimable queda `NOT_G2_CALIBRATED`, nunca `G2_CALIBRATED`, sin importar si `m≤60`.

`Q(m)` de `r_1m` con `m≤60` se interpreta únicamente contra la distribución empírica de Null 1 (§11.1), nunca contra una tabla chi-cuadrado clásica ni contra ningún null para `r_tilde` (que no existe). `Q(1.378)` es informativo (magnitud grande, consistente con el resto de la evidencia) pero no tiene un null calibrado a esa distancia — no se afirma genéricamente que "todos los `Q(m)` se interpretan contra G2" (corrección factual de la 2a revisión, reforzada en la 4a).

**Traducción de `rho_1`/`beta_1` a ticks**: **0,1425 ticks** (usando `beta_1`, la pendiente OLS real — ver §12 para la fórmula exacta y por qué ya no se usa `rho_1` para esta conversión).

`r_tilde` (ajustado): `rho_1=-0,005714` (cambia de signo respecto a `r_1m` crudo en este rezago específico), `rho_2=-0,013331` (mayor magnitud que crudo, igual que en la versión anterior) — el ajuste por `s(m)` no "limpia" esta estructura de rezago corto, resultado esperado y sin contradicción con TDA-06.

## 8. Refutación TH17

**(i) 1→5→10 minutos, no solapado** (sin cambios significativos — esta ruta reutiliza `build_horizon_returns`/`non_overlap_mask` de TH10, no dependía del bug de agrupación):

| h (min) | n | `n_pairs`(lag1) | `rho_1` | `beta_1` | IC 95% bootstrap `rho` |
|---:|---:|---:|---:|---:|---|
| 1 | 1.914.530 | 1.911.434 | 0,005921 | 0,005915 | [−0,000284, 0,012177] |
| 5 | 381.031 | 378.475 | −0,011060 | −0,011042 | [−0,020217, −0,001777] |
| 10 | 189.581 | 187.249 | 0,011494 | 0,011435 | [−0,001726, 0,024707] |

Sin cambios respecto a la versión anterior: no hay dilución monótona limpia; los intervalos a 5/10 minutos son más anchos (pérdida de 80-90% de la muestra por diseño de TH10).

**(ii) Por decil de volumen — RESULTADO SUSTANCIALMENTE DISTINTO tras la corrección** (decil 0 = menor actividad, decil 9 = mayor; análisis PRINCIPAL: pares construidos sobre la topología completa, condicionados solo por `decile_t`):

| Decil | `n_pairs` | `rho_1` | `beta_1` | Ticks aprox.* |
|---:|---:|---:|---:|---:|
| 0 | 192.668 | −0,131155 | −0,120147 | ≈ −2,89 |
| 1 | 191.887 | −0,100685 | −0,093626 | ≈ −2,26 |
| 2 | 190.474 | **−0,167780** | **−0,151637** | **≈ −3,65** |
| 3 | 190.641 | −0,144577 | −0,134560 | ≈ −3,24 |
| 4 | 191.333 | −0,114661 | −0,107679 | ≈ −2,59 |
| 5 | 189.516 | −0,056917 | −0,054932 | ≈ −1,32 |
| 6 | 191.222 | −0,044703 | −0,043371 | ≈ −1,04 |
| 7 | 191.004 | 0,005468 | 0,005349 | ≈ 0,13 |
| 8 | 191.399 | 0,006109 | 0,005986 | ≈ 0,14 |
| 9 | 191.290 | 0,061509 | 0,064168 | ≈ 1,55 |

*Ticks aproximados usando el mismo factor de conversión global de §7 (`ticks/beta≈24,09`); NO es un cálculo per-decil de `sigma_r`/tick_return_repr nuevo (fuera del alcance predeclarado de esta corrección) — solo una ilustración de orden de magnitud. Valores prácticamente idénticos (5ª-6ª cifra decimal) a los de la 1a revisión pese al centrado POR GRUPO de esta 2a revisión — ver §0b para la explicación (diferencias de media entre deciles minúsculas frente a la volatilidad dentro de cada decil, en los datos reales de MNQ).

**Sensibilidad secundaria** (convención estricta anterior, exige `decile_{t-1}=decile_t`, `TDA08_decil_volumen_sensibilidad.csv`): decil 0 `rho=-0,089942` (`n_pairs=122.424`), decil 2 `rho=-0,188946` (`n_pairs=50.216`), decil 9 `rho=0,036455` (`n_pairs=143.803`) — mismo signo y mismo patrón cualitativo (bajo volumen → más negativo; alto volumen → cercano a cero o positivo) que el análisis principal, con magnitudes algo distintas por construcción, y con `n_pairs` bastante menor (como es esperable al exigir una condición mucho más restrictiva).

**Corrección factual explícita respecto al informe anterior**: el patrón **NO es "casi monótono"**. En ambos métodos (principal y sensibilidad estricta) el decil 2 muestra el `rho_1` más negativo, por encima incluso del decil 0 (menor actividad) — una anomalía no monótona reproducida en dos convenciones de cálculo distintas, por lo que no parece un artefacto del método de agrupación. La magnitud global del efecto es, además, **aproximadamente el doble** de lo que reportaba la versión anterior (buggy): el decil 2 corregido implica un movimiento local de casi 4 ticks, no ~1,7 ticks como se citaba antes. Esto **fortalece** la conclusión de que la dependencia de rezago corto no es una cifra global despreciable cuando se condiciona por actividad, pero **debilita** la lectura de "gradiente limpio" que el informe anterior sugería — la anomalía del decil 2 queda como pregunta abierta (§16), no como parte de la narrativa de microestructura resuelta.

**(iii) Magnitud vs. tick**: global, 0,1425 ticks (STOP-8a, §14); en los deciles 0-4 de volumen, entre ≈2,3 y ≈3,7 ticks — magnitud claramente NO despreciable cuando se condiciona por actividad baja, más marcada que en la versión anterior.

**Por segmento de TDA-06** (motor corregido, sin cambio de convención que afecte materialmente aquí — el segmento cambia pocas veces al día, muy distinto del decil que cambia minuto a minuto): `rho_1` de −0,030949 (`16:02-20:00`) a +0,016967 (`09:30-16:02`, sesión RTH) — prácticamente sin cambio respecto a la versión anterior (−0,030 / +0,017). Confirma que el bug de agrupación (§0, punto 3) afectaba materialmente al decil de volumen (cambia cada minuto) mucho más que a segmento/año (cambian pocas veces).

**Balance de la refutación**: sin cambios en el veredicto. La evidencia de decil de volumen sigue siendo la pieza más fuerte hacia microestructura/heterogeneidad de actividad, ahora de mayor magnitud pero con una anomalía (decil 2) que impide describirla como un gradiente limpio; la dilución 1→5→10 sigue siendo ambigua; el segmento sigue siendo mixto. **Veredicto: `NOT SEPARABLE WITH OHLCV LAST`** (STOP-8b).

## 9. Estabilidad TH18

**Por año — CORRECCIÓN FACTUAL**: `rho_1` = 2019 (0,0285, año PARCIAL), 2020 (0,0071), 2021 (0,0117), 2022 (0,0098), 2023 (**−0,0087**), 2024 (0,0075), 2025 (−0,0009, año PARCIAL). De los **5 años completos** (2020-2024): **4 positivos, 2023 negativo**. La versión anterior decía "5 de 5 salvo 2023", contando incorrectamente el año parcial 2019 dentro de los "5". Corregido.

**Por segmento**: ver §8 — sin cambio material, rango `[−0,031, +0,017]`.

**Ventanas rodantes mensuales** (`TDA08_acf_por_grupo.csv`, `group_type=year_month`): **67 meses** (la versión pre-corrección decía 66 — corregido), `rho_1` mensual entre −0,050087 (2022-07) y +0,071538 (2020-07) — prácticamente el mismo rango en ambas revisiones; el centrado local (§0b) no cambia sustancialmente esta cifra porque el mes, como el segmento y el año, cambia con muy poca frecuencia frente al rezago de 1 minuto.

**Conclusión TH18**: sin cambio — la ACF agregada NO es estable en signo ni magnitud entre años, segmentos ni meses; no se cita como propiedad única de MNQ.

## 10. Ventanas 09:31-09:35 y 15:52-16:02 (TDA-06)

| | 09:31-09:35 | 15:52-16:02 |
|---|---:|---:|
| n | 6.960 | 14.817 |
| Media **por barra** | 0,0000020 | 0,0000174 |
| IC 95% bootstrap (media por barra) | [−0,0000228, 0,0000258] | [0,0000032, 0,0000296] |
| ¿Excluye cero? | No | **Sí** (por poco) |
| Media por barra en ticks | **≈0,107 ticks por barra de 1 minuto** | **≈0,980 ticks por barra de 1 minuto** |
| `rho_1` dentro de la ventana | −0,030822 | −0,022863 |

**Cuantificación explícita por barra (corrección §0)**: la ventana de cierre implica un movimiento medio de **≈0,98 ticks por cada barra de 1 minuto** dentro de la ventana — es decir, prácticamente **un tick completo por minuto**, no una fracción abstracta de tick sin contexto de escala temporal. Esto NO se traduce a una señal ni a un cálculo de PnL: es una magnitud descriptiva de la barra promedio en ese tramo horario, consistente con §14.

**Ventana de apertura**: la media NO excluye cero. Minuto exacto de apertura (571=09:31): media notablemente mayor (0,000119) que los minutos vecinos (570: 0,000029; 572: −0,000015) — pico localizado, consistente con la concentración de extremos de TDA-06.

**Ventana de cierre**: la media SÍ excluye cero (margen mínimo). Por año: 2019 (0,000142), 2020 (0,000034), 2021 (0,000003), 2022 (0,000018), 2023 (0,000031), 2024 (−0,000005), 2025 (0,000023) — mayoritariamente positivo pero de magnitudes muy dispares, sin estabilidad interanual clara.

**Veredicto de las ventanas**: sin cambio — candidatas a artefacto de apertura/cierre, no señales confirmadas.

## 11. Calibración G2 — Null 1 como inferencia PRINCIPAL (3a revisión)

**Cambio de fondo respecto a las revisiones anteriores**: la calibración de G2 usaba una combinación de dos nulls (Null 1 + Null 2) como si fueran equivalentes. El diagnóstico de momentos (§11.2) demostró que Null 2 no reproduce la varianza ni la curtosis reales — por tanto, desde esta revisión, **la inferencia PRINCIPAL de G2 usa EXCLUSIVAMENTE Null 1** (permutación de los valores reales dentro de cada `minute_of_day`), que al ser una permutación de datos reales reproduce su distribución marginal exacta sin depender de ningún supuesto de momentos que deba verificarse. Null 2 queda relegado a sensibilidad histórica/secundaria, combinado con Null 1, **nunca** como evidencia principal (ver §11.3).

### 11.1. Calibración PRINCIPAL — SOLO Null 1 (`TDA08_g2_calibracion_null1_principal.csv`, `TDA08_g2_portmanteau_null1_principal.csv`)

Percentil de `|rho_k|` real dentro de Null 1 (**200 réplicas**, calibrado hasta rezago 60):

| Rezago | `|rho_k|` real | Mediana Null 1 | P97,5 Null 1 | Percentil del real | ¿Supera el umbral? |
|---:|---:|---:|---:|---:|---|
| 1 | 0,005921 | 0,000668 | 0,002057 | 100,0 | Sí |
| 2 | 0,008445 | 0,000625 | 0,002014 | 100,0 | Sí |
| 5 | 0,004479 | 0,000664 | 0,001937 | 100,0 | Sí |
| 10 | 0,003238 | 0,000613 | 0,002165 | 100,0 | Sí |
| 20 | 0,003012 | 0,000583 | 0,002103 | 100,0 | Sí |

`Q(m)` (portmanteau) real contra la distribución de `Q(m)` bajo Null 1 exclusivamente:

| `m` | `Q_null1` mediana | `Q_null1` P97,5 | `Q` real (r_1m) | Percentil del real | ¿Supera el umbral? |
|---:|---:|---:|---:|---:|---|
| 10 | 16,83 | 36,59 | 378,18 | 100,0 | Sí |
| 20 | 34,90 | 53,05 | 445,43 | 100,0 | Sí |
| 40 | 65,77 | 98,71 | 612,14 | 100,0 | Sí |
| 60 | 95,93 | 135,63 | 718,99 | 100,0 | Sí |

**Conclusión principal (única base de calibración de G2 desde esta revisión)**: bajo Null 1 exclusivamente, `rho_k` real y `Q(m)` real superan el percentil 97,5 en **el 100% de los rezagos evaluados**, sin ninguna excepción — de hecho más nítido que la calibración combinada anterior (que daba 99,75, no 100,0, en los rezagos 10 y 20, por la cola artificialmente inflada de Null 2 ensanchando el umbral). El pipeline no fabrica esta magnitud sobre ruido con la estructura de reloj real.

### 11.2. Diagnóstico de momentos de Null 2 — por qué queda excluido de la inferencia principal

El informe de la 2a revisión afirmaba que Null 2 (empírico reescalado) "preserva volatilidad/curtosis comparable a los datos reales". Esa afirmación se verificó con `g2_synthetic_null_moment_check` (una realización del sorteo sintético contra `r_1m` real, tres criterios predeclarados):

| Criterio | Real | Sintético | Diferencia | Tolerancia predeclarada | ¿Cumple? |
|---|---:|---:|---:|---|---|
| Varianza | 1,757×10⁻⁷ | 2,403×10⁻⁷ | razón=1,368 | `1 ± 0,25` | **No** |
| Curtosis en exceso (Fisher) | 111,6 | 401,2 | diferencia relativa=2,60 | `≤0,50` relativo | **No** |
| Perfil de escala por minuto (correlación, vía MAD robusta) | — | — | 0,997 | `≥0,90` | **Sí** |

**La afirmación original era FALSA en dos de los tres criterios** — corregida en el código (`g2_synthetic_empirical_null` ya NO afirma preservar volatilidad/curtosis, solo describe lo que hace mecánicamente) y en este informe. El perfil de escala por minuto se preserva con fidelidad (correlación 0,997), pero la curtosis del sintético es ~3,6 veces mayor que la real y la varianza un ~37% mayor. Interpretación (sin sobre-explicar): al remuestrear de forma INDEPENDIENTE un valor del pool de `r_tilde` y una posición temporal con su `s(m)`, el sorteo sintético puede combinar una cola extrema del pool con un minuto de escala alta con más frecuencia que como esas dos cosas coocurren genuinamente en los datos reales.

**Corrección explícita de una afirmación no demostrada de la revisión anterior**: el informe anterior sugería que un null con más varianza/curtosis sería automáticamente "más conservador" para la correlación. **Esa relación NO está demostrada aquí y se retira del informe** — afirmarla sin demostración sería el mismo tipo de error que la afirmación original sobre Null 2. La razón real para excluir Null 2 de la inferencia principal no es un argumento sobre conservadurismo, sino más simple y más sólida: Null 2 no pasó el diagnóstico de momentos que se predeclaró para confiar en él, así que no se usa como base de una conclusión.

**Null 2 no se rediseña en esta tarea** (por decisión explícita: documentar el fallo y excluirlo es preferible a ampliar el alcance rediseñando un componente ya aprobado). Se conserva únicamente como sensibilidad histórica (§11.3), marcado con la etiqueta `SENSIBILIDAD FALLIDA / DIAGNÓSTICO`.

### 11.3. Calibración combinada — SOLO sensibilidad histórica/secundaria, NO evidencia principal

`TDA08_g2_calibracion_combinada_secundaria.csv` / `TDA08_g2_portmanteau_combinado_secundario.csv` (200+200=400 réplicas combinadas) se conservan por trazabilidad con las revisiones anteriores, pero **no se usan para ninguna conclusión de esta etapa**:

| Rezago | `|rho_k|` real | P97,5 combinado | Percentil del real | ¿Supera el umbral? |
|---:|---:|---:|---:|---|
| 1 | 0,005921 | 0,002251 | 100,00 | Sí |
| 2 | 0,008445 | 0,002373 | 100,00 | Sí |
| 5 | 0,004479 | 0,002430 | 100,00 | Sí |
| 10 | 0,003238 | 0,002414 | 99,75 | Sí |
| 20 | 0,003012 | 0,002306 | 99,75 | Sí |

Nótese que el P97,5 combinado es sistemáticamente más alto que el de Null 1 solo (p. ej. rezago 1: 0,002251 vs 0,002057) — consistente con que Null 2 aporta una cola más dispersa (mayor varianza/curtosis, §11.2) al conjunto combinado. La conclusión cualitativa (real supera el umbral) no cambia, pero **esta tabla ya no es la base de esa conclusión** — la base es exclusivamente §11.1.

### 11.4. `r_tilde` NO tiene calibración G2 propia (4a revisión — cierre definitivo)

**Problema bloqueante detectado y corregido en el cierre definitivo**: Null 1 y Null 2 se construyen EXCLUSIVAMENTE sobre `r_1m` (`values_raw`) — no existe, ni existió en ninguna revisión anterior, una calibración G2 equivalente construida específicamente para `r_tilde`. Sin embargo, `annotate_portmanteau_calibration` se aplicaba de forma genérica a ambas series, por lo que `portmanteau_adjusted` (la tabla de `r_tilde`) podía aparecer con `calibration_status="G2_CALIBRATED"` para `m≤60` — una afirmación falsa, ya que ningún null se calculó nunca sobre `r_tilde`.

**Corrección**: `annotate_portmanteau_calibration` ahora exige un argumento obligatorio `has_null1_calibration` (sin valor por defecto, para que sea estructuralmente imposible omitirlo por accidente). Desde esta revisión:

- `r_1m` (`has_null1_calibration=True`): `m≤60` → `G2_CALIBRATED`; `m>60` estimable → `DESCRIPTIVE_UNCALIBRATED`; no estimable → `NOT_ESTIMABLE`. Sin cambios respecto a §11.1.
- `r_tilde` (`has_null1_calibration=False`): **todo `m` estimable → `NOT_G2_CALIBRATED`** (nunca `G2_CALIBRATED`, sin importar si `m≤60`); no estimable → `NOT_ESTIMABLE`.

**Decisión explícita: NO se implementó una calibración G2 Null-1 propia para `r_tilde`.** `r_tilde` es, en esta etapa, un diagnóstico retrospectivo complementario (TDA-06/07) — no la base de ninguna hipótesis (TH16/17/18 se resuelven sobre `r_1m`; `r_tilde` se reporta como contraste, §7). Construir un segundo sistema de nulls (permutación + sintético + diagnóstico de momentos) solo para calibrar `r_tilde` habría ampliado el alcance de esta corrección sin que ninguna hipótesis lo exigiera. `r_tilde` conserva ACF/PACF/portmanteau descriptivos completos (`TDA08_acf_global.csv`, `TDA08_pacf.csv`, `TDA08_portmanteau.csv`, todos con `series="r_tilde"`) — simplemente no calibrados contra un null propio.

## 12. Traducción de `rho_1`/`beta_1` a ticks — fórmula exacta (corregida)

**Corrección de fondo (§0, punto 1b)**: el "movimiento lineal implícito" es, por definición de pendiente de regresión, `beta_1 × sigma_r` — EXACTO, no una aproximación bajo `Var(r_t)=Var(r_{t-1})`. `rho_1` se reporta aparte y NUNCA se usa para esta conversión.

| Cantidad | Valor |
|---|---:|
| `rho_1` | 0,005921 |
| `beta_1` (usado para la conversión) | 0,005915 |
| `beta_1/rho_1` | 0,9990 |
| `sigma_r` | 0,0004192 |
| Movimiento lineal implícito (retorno) | 2,4794×10⁻⁶ |
| Close representativo (mediana) | 14.365,25 |
| Conversión a puntos | 0,03562 |
| Retorno-equivalente de 1 tick | 1,7403×10⁻⁵ |
| **Conversión a ticks** | **0,1425** |

Sin cambio material respecto a la versión anterior (0,1422) porque `beta_1≈rho_1` en el rezago 1 (razón 0,999) — la corrección de fondo importa conceptualmente (nunca se debe presuponer la igualdad) aunque en este rezago específico el número final apenas se mueve.

## 13. Limitaciones de observabilidad

- OHLCV Last de 1 minuto: sin Bid/Ask, spread, agresor ni secuencia intrabarra — TH17 no puede confirmar el mecanismo (`NOT SEPARABLE WITH OHLCV LAST`).
- **Techo estructural de rezago**: ningún bloque alcanza 1.379 minutos seguidos — el máximo rezago estimable es 1.378, no 2.760 (ver §1). `Q(1.380)` y `Q(2.760)` son `NOT_ESTIMABLE`, no valores calculados.
- `n_perm=200` por null — suficiente para el criterio predeclarado (P97,5), no exhaustivo.
- 1→5→10 minutos pierde 80-90% de la muestra por diseño (no solapado).
- La anomalía de no-monotonicidad del decil 2 de volumen (§8) no se investiga más allá de documentarla — está fuera del alcance de esta corrección abrir una nueva línea de análisis (§0 cierre: "primero corrección metodológica, después eficiencia, nunca al revés" aplica también a no ampliar alcance).

## 14. STOP-8a

**ACTIVADO** (cifra global: 0,1425 ticks, muy por debajo de 1 tick). Matiz reforzado respecto a la versión anterior: en los deciles 0-4 de volumen, la magnitud local corregida es de **2,3 a 3,7 ticks** (antes se citaba ~1,7 ticks en el peor caso) — no contradice STOP-8a (que es sobre la cifra global), pero la heterogeneidad condicionada es más pronunciada de lo que se había reportado.

## 15. STOP-8b

**ACTIVADO.** Sin cambio de veredicto: los OHLCV Last no permiten separar bid-ask bounce de no-sincronía de una reversión económica genuina. La evidencia de decil de volumen es más fuerte en magnitud tras la corrección, pero también menos limpia (no monótona) — no inclina la balanza hacia una atribución causal. **`NOT SEPARABLE WITH OHLCV LAST`.**

## 16. Estado de las hipótesis (reevaluado desde cero, no heredado)

- **TH16 — RESUELTA.** ACF/PACF/portmanteau calculados correctamente hasta el máximo rezago genuinamente estimable (1.378), con `rho`/`beta` separados y centrados localmente por conjunto de pares (Pearson pairwise-complete literal), calibrados contra Null 1 EXCLUSIVAMENTE (§11.1 — superado en el 100% de los rezagos evaluados, sin excepción), traducidos a ticks vía `beta_1` (0,1425 ticks). Bartlett/PACF permanecen como diagnóstico clásico aproximado, nunca la inferencia principal. **Sin exagerar la evidencia (§7)**: `rho_1` global NO excluye cero bajo el bootstrap que respeta la dependencia intradía (única inferencia principal de incertidumbre); `rho_2`/`rho_5` sí excluyen cero, mostrando una estructura pequeña pero genuina. G2 (Null 1) demuestra que la magnitud observada supera claramente el null predeclarado con estructura de reloj realista — eso certifica que el pipeline no fabrica la magnitud sobre ruido, **no** que esa magnitud (0,1425 ticks global) implique predictibilidad operativa alguna (STOP-8a, §14).
- **TH17 — `NOT SEPARABLE WITH OHLCV LAST`.** Protocolo de refutación completo. La evidencia de decil de volumen es fuerte en magnitud pero no monótona (anomalía del decil 2, reproducida en dos convenciones de cálculo, y confirmada de nuevo tras el centrado por grupo de la 2a revisión) — no permite cerrar la atribución causal.
- **TH18 — RESUELTA (resultado negativo de estabilidad).** `rho_1` no es estable en signo ni magnitud entre años (4 de 5 años completos positivos, no 5 de 5), segmentos ni meses (67, no 66). El centrado por grupo de la 2a revisión no cambió esta conclusión (diferencias de media entre grupos demasiado pequeñas frente a la volatilidad intragrupo para alterar el resultado).

## 17. Estado final de TDA-08

**`PASS_WITH_OPEN_QUESTIONS`**

Reevaluado desde cero en las tres revisiones, sin asumir que el veredicto anterior siguiera vigente. El software funciona correctamente sobre una base metodológica corregida tres veces (345/345 tests, incluidos los 48 de esta etapa) y la metodología se completó íntegramente con las correcciones de fondo aplicadas (cuatro de la 1a revisión + el centrado local de la 2a + la calibración G2 basada exclusivamente en Null 1 de la 3a). El estado metodológico sigue sin ser un `PASS` limpio: TH17 se cierra como `NOT SEPARABLE` (límite declarado), y persiste la anomalía de no-monotonicidad del decil 2 como pregunta abierta. El hallazgo de que Null 2 no replica la curtosis/varianza real (§11.2) ya NO es una advertencia sobre la conclusión de calibración — desde esta revisión esa conclusión se basa exclusivamente en Null 1 (§11.1), que no tiene ese problema; Null 2 quedó fuera de la inferencia principal, no como matiz sino como exclusión.

## 18. Comparación consolidada OLD (versión con bug) vs. NEW (esta corrección)

| Cantidad | OLD | NEW | ¿Cambió y por qué? |
|---|---:|---:|---|
| `rho_1` global | 0,005903 | 0,005921 | Casi igual — a rezago 1 casi todos los pares son válidos en ambas convenciones, el bug de normalización por `T` apenas se nota |
| IC 95% bootstrap `rho_1` | [−0,000283, 0,012134] | [−0,000284, 0,012177] | Casi igual, mismo motivo |
| `rho_2` / `rho_5` | −0,008402 / −0,004434 | −0,008445 / −0,004479 | Casi igual, mismo motivo |
| Ticks (`rho_1`/`beta_1`) | 0,1422 (usaba `rho_1`) | 0,1425 (usa `beta_1`, correcto por definición) | Cambio conceptual de fondo (nunca más se presupone `beta=rho`); numéricamente casi igual porque `beta_1/rho_1=0,999` en este rezago |
| `Q(1.380)`/`Q(2.760)` | 4.254,3 (idénticos, tratados como calculados) | `NOT_ESTIMABLE` (ninguno) + `Q(1.378)=6.004,4` reportado aparte | El bug anterior propagaba un `NaN` como cero silencioso; ahora se declara explícitamente que no hay pares |
| G2 — P97,5 del null (rezago 1) | ≈0,001621 (nulls menos realistas) | ≈0,002251 (nulls preservan reloj/curtosis) | Los nulls nuevos son más exigentes; la señal real sigue superándolos |
| Decil de volumen 0, `rho_1` | −0,070984 | −0,131155 | El bug exigía implícitamente `decile_{t-1}=decile_t`, descartando la mayoría de los pares genuinos; el motor corregido casi duplica la magnitud |
| Decil de volumen, forma del patrón | "casi monótono" | NO monótono (decil 2 es el más negativo, reproducido en dos convenciones) | Corrección factual — la versión anterior describía un gradiente limpio que no sobrevive a la corrección del bug de agrupación |
| Años completos positivos | "5 de 5 salvo 2023" | 4 de 5 (2019 es parcial, no cuenta) | Error de conteo en la versión anterior |
| Meses en tabla rodante | 66 | 67 | Error de conteo en la versión anterior |
| Réplicas G2 | "400+400" (ambiguo/incorrecto) | 200+200=400 combinadas | Corrección de redacción — nunca fueron 800 ni 400² |
| Ventana de cierre, magnitud | "≈1 tick" (sin anclar a la barra) | "≈0,98 ticks **por barra de 1 minuto**" | Cuantificación explícita por barra, exigida por la revisión |
| TH16/TH17/TH18 | RESUELTA / NOT SEPARABLE / RESUELTA (negativo) | Sin cambio de categoría, pero re-derivado desde cero sobre base corregida | La corrección no cambió el veredicto cualitativo, pero sí lo puso sobre fundamentos válidos |
| Estado final TDA-08 | `PASS_WITH_OPEN_QUESTIONS` | `PASS_WITH_OPEN_QUESTIONS` | Misma categoría, alcanzada esta vez sin los cuatro bugs de fondo |

## 19. Comparación 1ª corrección vs. 2ª corrección (centrado del estimador)

| Cantidad | 1ª corrección | 2ª corrección (esta) | ¿Cambió y por qué? |
|---|---:|---:|---|
| `rho_1` global | 0,005921 | 0,005921 | Idéntico a la precisión reportada — a rezago 1 el conjunto de pares excluidos por frontera es una fracción mínima de la población, la media local del par y la media global casi coinciden |
| IC 95% bootstrap `rho_1` | [−0,000284, 0,012177] | [−0,000284, 0,012177] | Idéntico |
| Decil de volumen 0 / 2 / 9, `rho_1` | −0,131155 / −0,167773 / 0,061709 | −0,131155 / −0,167780 / 0,061509 | Diferencias en la 5ª-6ª cifra decimal — el centrado por grupo (§0b, la corrección de fondo de esta revisión) apenas mueve el número porque las diferencias de media ENTRE deciles son minúsculas frente a la volatilidad DENTRO de cada decil, en los datos reales de MNQ (a diferencia del escenario adversarial del test, diseñado con medias separadas 5 desviaciones estándar) |
| Año 2020 / 2022, `rho_1` | 0,007088 / 0,009831 | 0,007081 / 0,009820 | Diferencias en la 5ª-6ª cifra decimal, mismo motivo |
| `rho_1` ventana apertura/cierre | −0,031313 / −0,022949 | −0,030822 / −0,022863 | Cambio pequeño (4ª cifra decimal) — subconjuntos más pequeños (pocos miles de filas), el centrado local por-par tiene un efecto algo más perceptible que en la población completa, pero sigue sin alterar ninguna conclusión |
| `Q(1.378)` | 6.004,44 | 6.004,56 | Diferencia mínima |
| Afirmación "null 2 preserva volatilidad/curtosis" | Afirmada sin test explícito | **Verificada y refutada en 2 de 3 criterios** (§11.2) — curtosis sintética ~3,6x la real, varianza ~37% mayor; perfil de escala sí se preserva (correlación 0,997) | Hallazgo NUEVO de esta revisión, no una corrección de un número previamente reportado — la 1a revisión nunca había comprobado esto |
| Tabla de portmanteau | Sin distinguir calibración | `calibration_status` explícito por fila (`G2_CALIBRATED`/`DESCRIPTIVE_UNCALIBRATED`/`NOT_ESTIMABLE`) | Aclaración exigida por la revisión — `Q(1.378)` nunca tuvo un null calibrado a esa distancia, ahora se declara |
| Bartlett/PACF | Descritos como "diagnóstico clásico de referencia" | Docstrings/informe refuerzan: aproximado, nunca el intervalo principal | Aclaración de lenguaje, sin cambio de cálculo |
| Tests | 40 passed | 45 passed (+5: centrado pairwise literal, contraste global-mean, adversarial de grupo, diagnóstico de momentos G2, calibración de portmanteau, persistencia del runner) | Cobertura ampliada por la revisión |
| Tiempo de corrida real | 753,1 s | 1.034,6 s (+37%) | Coste del centrado local (dos medias por rezago/grupo en vez de un `e` precomputado una sola vez) — no se optimizó a costa de la corrección |
| TH16/TH17/TH18 | RESUELTA / NOT SEPARABLE / RESUELTA (negativo) | Sin cambio de categoría — re-derivado desde cero sobre el estimador correctamente centrado | La corrección no cambió el veredicto cualitativo en los datos reales, pero corrige un problema conceptual que SÍ importaría en un dataset con mayor heterogeneidad de medias entre grupos |
| Estado final TDA-08 | `PASS_WITH_OPEN_QUESTIONS` | `PASS_WITH_OPEN_QUESTIONS` | Misma categoría, ahora sobre un estimador formalmente correcto y con la afirmación de G2 verificada (no solo declarada) |

**Conclusión de esta comparación**: el centrado del estimador (2ª revisión) era una corrección conceptualmente necesaria — el test adversarial demuestra sin ambigüedad que el bug PODÍA fabricar dependencia espuria fuerte (`rho≈-0,96` en un escenario de medias muy separadas) — pero, sobre el conjunto de investigación real de MNQ, su impacto numérico fue insignificante porque la heterogeneidad de medias entre los grupos analizados (año, segmento, decil de volumen) es pequeña frente a la volatilidad dentro de cada grupo. El hallazgo sustantivo nuevo de esta ronda no es un cambio de `rho`/`beta`, sino la verificación (y refutación parcial) de la afirmación sobre el null sintético de G2.

## 20. Cierre final — G2 Null-1-only (3ª revisión)

Corrección puntual solicitada tras la 2ª revisión: excluir Null 2 de la inferencia PRINCIPAL de G2 (§11) y usar EXCLUSIVAMENTE Null 1. No se modificó ACF, `beta`, bootstrap, agrupaciones, 1/5/10, ventanas, ni las definiciones que producen TH16/TH17/TH18 más allá de la propia calibración de G2 — por eso todos los valores ajenos a G2 (recalculados de todas formas, al reejecutar el pipeline completo por simplicidad y para evitar una ruta de recálculo parcial ad-hoc) salen bit-a-bit idénticos a los de la 2ª revisión.

**Resultados Null-1-only vs. calibración combinada anterior:**

| Cantidad | Combinada (2ª revisión, ahora secundaria) | Null-1-only (3ª revisión, PRINCIPAL) | Diferencia |
|---|---:|---:|---|
| `|rho_k|` real, percentil en el null, rezagos 1/2/5 | 100,00 / 100,00 / 100,00 | 100,0 / 100,0 / 100,0 | Igual |
| `|rho_k|` real, percentil en el null, rezagos 10/20 | 99,75 / 99,75 | **100,0 / 100,0** | **Más nítido** — sin la cola inflada de Null 2, el real supera el 100% de las réplicas en TODOS los rezagos, no solo en 3 de 5 |
| P97,5 del null, rezago 1 | 0,002251 | 0,002057 | El null combinado es más ancho (Null 2 aporta más dispersión) — el umbral de Null-1-only es más estrecho pero el real lo sigue superando ampliamente |
| `Q(m)` real, percentil en el null, m=10/20/40/60 | No reportado con percentil explícito en la 2ª revisión | 100,0 / 100,0 / 100,0 / 100,0 | Nueva cifra (§11.1) — el `Q_null1` P97,5 (p. ej. 36,6 en m=10) es mucho menor que el `Q_null` combinado (48,0), y el `Q` real (378,2) supera ambos por un margen enorme |
| n réplicas de la calibración principal | 400 (200+200, Null 2 incluido) | 200 (solo Null 1) | Menos réplicas en la base principal, pero sin el ruido de un null que no pasó su propio diagnóstico de momentos |
| Conclusión de calibración de G2 | "El pipeline no fabrica esta magnitud sobre ruido con estructura realista" | Misma conclusión, ahora sin depender de un null (Null 2) que demostradamente no reproduce la varianza/curtosis real | Conclusión cualitativa SIN CAMBIO — la base de esa conclusión es ahora más simple y más defendible |
| Estado TH16/TH17/TH18 | RESUELTA / NOT SEPARABLE / RESUELTA (negativo) | Sin cambio | La corrección de G2 no altera ningún veredicto — refuerza la base de TH16 |
| Estado final TDA-08 | `PASS_WITH_OPEN_QUESTIONS` | `PASS_WITH_OPEN_QUESTIONS` | Sin cambio de categoría |

**Archivos modificados en esta corrección puntual**: `tda08_linear_mean_dependence.py` (`g2_null1_calibration_summary` y `g2_null1_portmanteau_summary` nuevas — inferencia PRINCIPAL; `g2_calibration_summary`/`g2_portmanteau_null_summary` renombradas a `g2_combined_calibration_summary`/`g2_combined_portmanteau_summary` — SECUNDARIAS; docstring de `g2_synthetic_empirical_null` corregido para no afirmar que preserva volatilidad/curtosis; comentario de módulo reescrito), `run_tda08.py` (`persist_artifacts` escribe 2 artefactos nuevos, `ARTIFACT_PATH_ATTRS` con 17 rutas), `config.py`/`configs/mnq_snapshot.yaml` (`tda08_g2_calibration_null1_csv`, `tda08_g2_portmanteau_null1_csv` nuevos; los combinados renombrados de archivo a `..._combinada_secundaria`/`..._combinado_secundario` para que el nombre declare su rol), `tests/test_tda08_linear_mean_dependence.py` (imports renombrados, 3 tests nuevos).

**Tests nuevos de esta revisión** (obligatorios, punto 8 de la corrección): `test_g2_null1_calibration_uses_only_null1_and_ignores_null2_even_if_null2_would_change_the_verdict` (construcción determinística donde Null 1 solo NO marca "supera el umbral" pero el combinado SÍ lo haría si Null 2 se incluyera — confirma que la inferencia principal ignora Null 2 incluso cuando eso importa); `test_g2_null1_calibration_summary_signature_structurally_excludes_null2` (garantía estructural vía introspección de la firma — Null 2 no puede colarse por error futuro); `test_g2_null1_portmanteau_summary_matches_hand_computation_and_reports_real_percentile`. Además, `test_persist_artifacts_writes_exactly_the_declared_paths_and_nothing_obsolete` (ya existente, de la 2ª revisión) verifica automáticamente que los 2 artefactos Null-1-only se persisten y que no queda ningún nombre obsoleto.

**Tests**: `test_tda08_linear_mean_dependence.py` **48 passed** (45 previos + 3 nuevos). Suite completa: **345 passed**.

**Corrida real**: 988,8 s (≈16,5 min) — del mismo orden que la 2ª revisión (1.034,6 s); no se recalculó nada ajeno a G2 con una ruta separada, se reejecutó el pipeline completo determinista por simplicidad, evitando el riesgo de una ruta de recálculo parcial que pudiera divergir de la orquestación real.

## 21. Preguntas abiertas

1. **Anomalía de no-monotonicidad en el decil 2 de volumen** (§8) — reproducida en dos convenciones de cálculo distintas y confirmada de nuevo tras el centrado por grupo. No se investiga en esta tarea (fuera de alcance); queda como candidata a revisión dedicada si una etapa futura toca microestructura/volumen.
2. **Null 2 de G2 (empírico reescalado) no replica la curtosis/varianza real** (§11.2) — curtosis sintética ~3,6x la real. Ya NO es una advertencia sobre la calibración de G2 (que desde esta revisión no depende de Null 2), pero queda como candidata a rediseño si una etapa futura necesita específicamente un null sintético con momentos verificadamente representativos (p. ej. reescalar por volatilidad LOCAL en vez de un `s(m)` de reloj puro, o resamplear bloques en vez de puntos individuales).
3. **Dilución 1→5→10 minutos sin patrón limpio** (§8.i) — podría deberse al mecanismo real o a la pérdida de 80-90% de la muestra; no se puede distinguir con los datos disponibles.
4. **Ventana de cierre con media que excluye cero por margen mínimo** (§10) — magnitud ≈0,98 ticks/barra, sin estabilidad interanual clara.
5. **Techo estructural de rezago (1.378, no 2.760)** (§1) — se recomienda anotar el roadmap para que futuras etapas no asuman "2 jornadas" como alcanzable bajo las reglas de no-cruce vigentes.
6. **`r_tilde` muestra ACF de magnitud algo mayor que `r_1m` crudo en rezagos cortos** (§7) — coherente con que `s(m)` corrige estacionalidad de reloj, no dependencia de rezago corto.

Ninguna es bloqueante para continuar.

## 22. Estado tras la 3ª revisión (histórico — SUPERADO por §23)

*Esta sección describe el estado declarado al cierre de la 3ª revisión. Se conserva por trazabilidad, pero el problema de `r_tilde` descrito en §11.4 se detectó DESPUÉS de escribirse este párrafo -- el veredicto `CLOSED` de aquí abajo NO es el vigente. La declaración de cierre autoritativa y vigente es §23.*

**Coherencia verificada (al cierre de la 3ª revisión)**: TH16/TH17/TH18 reevaluados tras las tres revisiones (§16); la calibración de G2 se sostenía ya exclusivamente en Null 1; ningún texto afirmaba que Null 2 preserva volatilidad/curtosis; 17 artefactos, 345/345 tests. `TDA-08 = PASS_WITH_OPEN_QUESTIONS / CLOSED` — declarado entonces, **revocado** por el hallazgo de §11.4 (`r_tilde` marcada falsamente `G2_CALIBRATED`), que exigió la 4ª revisión antes de poder cerrar de verdad.

## 23. Cierre definitivo (4ª revisión — metadatos, documentación, consistencia)

**Alcance de esta revisión**: exclusivamente metadatos, semántica de calibración, tests y consistencia documental — explícitamente NO una revisión metodológica general. No se modificó la definición de ACF, `beta`, bootstrap, agrupaciones, deciles, 1/5/10, ventanas, traducción a ticks, Null 1, Null 2, semillas, `n_boot`, `n_perm`, población, horizontes ni reglas de continuidad.

**1. Corrección realizada**: `r_tilde` podía aparecer falsamente `G2_CALIBRATED` en `TDA08_portmanteau.csv` porque `annotate_portmanteau_calibration` se aplicaba genéricamente a ambas series sin verificar que `r_tilde` tuviera un null propio (no lo tiene, ni lo tuvo nunca). Corregido exigiendo el argumento obligatorio `has_null1_calibration` (sin valor por defecto) en cada llamada — ver §11.4 para el detalle y la tabla `TDA08_portmanteau.csv` corregida en §7.

**2. Decisión adoptada para `r_tilde`**: etiqueta `NOT_G2_CALIBRATED` para todo `m` estimable (nunca `G2_CALIBRATED`). **No se implementó una calibración G2 Null-1 propia para `r_tilde`** — decisión explícita: `r_tilde` es un diagnóstico retrospectivo complementario en esta etapa, no la base de ninguna hipótesis (TH16/17/18 se resuelven sobre `r_1m`); no se encontró ninguna razón metodológica por la que TH16 dependa de calibrar `r_tilde` — no se reporta como blocker porque no lo es.

**3. Archivos modificados**: `tda08_linear_mean_dependence.py` (`annotate_portmanteau_calibration` con `has_null1_calibration` obligatorio y estado `NOT_G2_CALIBRATED`; docstring de `compute_portmanteau_q` corregido — ya no referencia `g2_calibration_summary` con un `statistic=` inexistente; nuevo párrafo de módulo "CIERRE DEFINITIVO -- 4a revision"), `configs/mnq_snapshot.yaml` (terminología "Ljung-Box" → "portmanteau" en comentarios), `run_tda08.py` (docstring de `persist_artifacts` corregido, ya no fija "15" como cifra vigente), `tests/test_tda08_linear_mean_dependence.py` (llamadas a `annotate_portmanteau_calibration` actualizadas; 3 tests nuevos), `reports/mnq/TDA08_dependencia_lineal_media.md` (este informe — auditoría textual completa, ver punto 9).

**4. Artefactos regenerados**: los 17 declarados (sin cambio de cantidad ni de nombres respecto a la 3ª revisión) — en particular `TDA08_portmanteau.csv`, que ahora persiste `calibration_status` correcto y distinto por `series` (§7, §11.4).

**5. Tests TDA-08**: `python -m pytest -q tests/test_tda08_linear_mean_dependence.py` → **50 passed** (48 de la 3ª revisión + 2 nuevos: `test_portmanteau_calibration_status_never_marks_a_series_without_its_own_null_as_calibrated`, `test_annotate_portmanteau_calibration_requires_explicit_has_null1_calibration_argument`; además se actualizó `test_run_tda08_end_to_end_produces_all_result_fields` para verificar que `r_1m` SÍ y `r_tilde` NUNCA aparece `G2_CALIBRATED`).

**6. Suite completa**: `python -m pytest -q` → **347 passed** (297 previas a TDA-08 + 50 de esta etapa).

**7. Pipeline real**: `python -m ohlcv_dataroad.ingest.run_tda08 --config configs/mnq_snapshot.yaml` → **completado en 978,9 s (≈16,3 min)**, del mismo orden que las corridas anteriores (988,8 s / 1.034,6 s) — sin cambios de metodología, la variación es ruido normal de máquina, no una diferencia de código estadístico. `TDA08_portmanteau.csv` verificado directamente: `r_1m` tiene estados `{G2_CALIBRATED, DESCRIPTIVE_UNCALIBRATED, NOT_ESTIMABLE}`; `r_tilde` tiene únicamente `{NOT_G2_CALIBRATED, NOT_ESTIMABLE}` — `G2_CALIBRATED` nunca aparece para `r_tilde`.

**8. Verificación de los 17 artefactos**: los 17 declarados en `ARTIFACT_PATH_ATTRS` existen tras la corrida real; no queda ningún `TDA08_*` obsoleto en `reports/mnq/` (verificado por listado directo y por `test_persist_artifacts_writes_exactly_the_declared_paths_and_nothing_obsolete`).

**9. Auditoría textual final realizada** (sección 13 de la corrección): se buscaron explícitamente, en los 4 archivos TDA-08 relevantes (módulo, runner, tests, informe), las cadenas `"15 artefactos"`, `"14 artefactos"`, `"45 passed"`, `"342 passed"`, `"Ljung-Box"`, `"g2_portmanteau_null_summary"`, `"g2_calibration_summary"`, `"preserva forma/curtosis"`, `"preserva volatilidad/curtosis"`, `"400 réplicas"`, `"G2_CALIBRATED"`. Cada coincidencia se clasificó y corrigió cuando correspondía: la fila de Null 2 en la tabla histórica de §0 (afirmaba preservar forma/curtosis como hecho vigente — corregida a "intención" + referencia cruzada a §11.2); la mención de "400 réplicas" en §7 para `Q(1.378)` (estaba mal etiquetada — corregida a 200, Null 1); las referencias a "15 artefactos"/"45 passed"/"342 passed" en §3-§5 (correctas como historia de la 2ª revisión, pero sin marcar como tal — se añadieron avisos explícitos "estado en ese punto" + puntero al conteo vigente); las menciones de `g2_calibration_summary`/`g2_portmanteau_null_summary` fuera de contexto de renombramiento (una en el docstring de `compute_portmanteau_q`, con un `statistic=` que nunca existió — corregida); terminología "Ljung-Box" en `configs/mnq_snapshot.yaml` (corregida a "portmanteau"; las menciones en código/informe que dicen explícitamente "NO es Ljung-Box clásico" son correctas y se conservaron). El resto de coincidencias (uso de "G2_CALIBRATED" como nombre de estado válido, referencias históricas ya explícitamente marcadas como tales en §18/§19/§20) se clasificaron como vigentes/correctas o históricas-correctamente-etiquetadas y no requirieron cambios.

**10. Resultados estadísticos fundamentales**: sin cambios inesperados. `rho_1=0,005921`, `beta_1=0,005915`, traducción a ticks `0,1425`, máximo rezago estimable `1.378`, y todas las cifras por año/segmento/decil/ventanas idénticas a la 3ª revisión (verificado comparando la corrida real de esta revisión contra la anterior) — consistente con que esta corrección fue exclusivamente de metadatos/semántica de calibración, nunca de la metodología estadística.

**Estado final de las hipótesis** (sin cambio respecto a §16, reafirmado):
- **TH16 = RESUELTA.**
- **TH17 = `NOT SEPARABLE WITH OHLCV LAST`.**
- **TH18 = RESUELTA (resultado negativo de estabilidad).**

**Confirmaciones explícitas**: holdout **NO** abierto (permanece `LOCKED`, frontera `2025-06-23 00:00:00 UTC`); TDA-00…TDA-07 **intactos** (ningún archivo fuera de la lista de §"Archivos a revisar" de esta tarea fue modificado); TH10 **intacto**; TDA-09 **NO iniciado**; no quedan artefactos `TDA08_*` obsoletos.

**Criterio de cierre (§16 de la tarea) — verificado punto por punto**: `r_tilde` ya no aparece falsamente `G2_CALIBRATED` ✓; `r_1m` mantiene Null 1 como inferencia principal ✓; Null 2 permanece excluido de la inferencia principal ✓; no quedan contradicciones sobre Null 2 (auditoría textual, punto 9) ✓; 17 artefactos consistentes entre runner/config/tests/informe ✓; terminología portmanteau consistente ✓; nombres vigentes de funciones G2 consistentes (`g2_null1_*` principal, `g2_combined_*` secundario) ✓; 200 (principal) vs 400 (combinado secundario) réplicas correctamente diferenciadas ✓; tests TDA-08 pasan (50/50) ✓; suite completa pasa (347/347) ✓; pipeline real finaliza correctamente ✓; resultados fundamentales sin cambios inesperados ✓; holdout `LOCKED` ✓; ninguna etapa anterior modificada ✓; TDA-09 no iniciado ✓.

**TDA-08 = `PASS_WITH_OPEN_QUESTIONS` / `CLOSED`** (declaración vigente y definitiva — reemplaza la de §22)

Las preguntas abiertas (§21) quedan documentadas como limitaciones/preguntas futuras, no como bloqueantes.

**Sí, se puede iniciar TDA-09.** Recomendaciones (sin cambio respecto a las de §22, reafirmadas): tratar TH16/TH17 como contexto de magnitud diminuta pero heterogénea por volumen, sin asumir que ausencia de estructura en media implica ausencia de estructura en magnitud; reutilizar `compute_block_ids`/la topología de esta etapa; el techo de 1.378 rezagos aplica igual a cualquier ACF de magnitud sobre la misma topología; si TDA-09 diseña su propia calibración G2-like, preferir un null que sea una permutación de datos reales sobre uno sintético, salvo verificación explícita de que reproduce lo necesario; la anomalía del decil 2 de volumen es una observación a tener presente, no una instrucción de diseño.

**No se implementó TDA-09 en esta tarea.**

## Comandos exactos utilizados

```bash
python -m pytest -q tests/test_tda08_linear_mean_dependence.py
python -m pytest -q
python -m ohlcv_dataroad.ingest.run_tda08 --config configs/mnq_snapshot.yaml
```
