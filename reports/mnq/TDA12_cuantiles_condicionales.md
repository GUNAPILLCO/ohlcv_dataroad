# TDA-12 — Cuantiles condicionales

**Etapa:** `docs/methodology/Tsay_OHLCV_analysis_roadmap.md` § TDA-12 (Nivel 4, obligatoria). **Hipótesis:** `TH26`.
**Depende de:** TDA-06 (segmentos, `r_tilde`), TDA-07 (niveles de cuantil, bootstrap por jornada), TDA-10 (`z_t`, `sigma_hat` causal, umbrales de estabilidad de forma; veredicto `MIXTO`).
**Alcance de datos:** exclusivamente `tda04_variables_1m.parquet`, `tda04_return_validity_mask.parquet`, `tda06_r_tilde.parquet` y `TDA06_segmentacion_propuesta.csv`. Ningún archivo de `data/raw/` ni de `holdout_files` fue abierto.
**Generado automáticamente** por `python -m ohlcv_dataroad.ingest.run_tda12 --config configs/mnq_snapshot.yaml` — este informe NUNCA se edita a mano; cualquier corrección se hace en el código (`tda12_conditional_quantiles.py`/`run_tda12.py`) y se regenera.

> Esta etapa describe cómo cambian los cuantiles de los retornos de MNQ según la hora, la volatilidad previa y el año, y responde si la distribución cambia sólo de ESCALA (más o menos volatilidad) o también de FORMA (colas proporcionalmente más o menos pesadas, asimetría). Es caracterización estadística: NO diseña targets, señales ni modelos; NO ejecuta TDA-13 ni EVT; NO abre el holdout.

---

## 1. Pregunta y versión ejecutada

**Pregunta central (TH26):** ¿la distribución de los retornos cambia únicamente de escala o también de forma?

**Versión: COMPLETA.** El roadmap prevé la versión mínima sólo si TDA-10 concluye que la curtosis "desaparece casi toda". TDA-10 concluyó `MIXTO` (bloque causal, acuerdo 0,50) y dejó `STOP-13 = NO SUGERIDO` con TH26 "formalmente pendiente de TDA-12". Corresponde la versión completa: cuantiles con bootstrap por grupo y el `n` que sostiene cada cuantil extremo.

## 2. Población y variables (heredadas, sin redefinir)

- **Población:** 1,914,530 retornos de 1 minuto válidos (`r_1m_valid=True`, TDA-04: ninguno cruza roll, jornada ni hueco), en 1,420 jornadas.
- **`r_t`** = `r_1m`.
- **`sigma_hat_(t-1)`** = estimador PRIMARIO de TDA-10: `ewma` con parámetro 60 barras sobre `r_1m`, construido con las funciones de TDA-10 sin modificar. Usa sólo `r[0..t-1]`.
- **`z_t`** = `r_t / sigma_hat_(t-1)` = `z_ewma_60_raw` (mismo quemado y mismo piso numérico que TDA-10).
- **Condicionantes:** segmento horario (cortes de TDA-06), decil de `sigma_hat_(t-1)`, año.
- **Percentiles:** 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9 (los del roadmap; `QUANTILE_LEVELS` de TDA-07).

| series | block | standardized | causal_class | n_finite | n_excluded_nonfinite |
|---:|---:|---:|---:|---:|---:|
| r_1m | raw | False | CAUSAL | 1,914,530 | 0 |
| z_ewma_60_raw | raw | True | CAUSAL | 1,914,350 | 180 |
| r_tilde | clock_adjusted | False | RETROSPECTIVO | 1,914,530 | 0 |
| z_ewma_60_clock_adjusted | clock_adjusted | True | RETROSPECTIVO | 1,914,350 | 180 |

`n_excluded_nonfinite` de `z` es el quemado inicial del estimador. El bloque `clock_adjusted` (`r_tilde` y su `z`) es RETROSPECTIVO (`s(m)` de TDA-06 usa toda la muestra): diagnóstico secundario, nunca decide (§11).

## 3. Método e incertidumbre

**Cuantiles empíricos** con la convención de TDA-07 (`np.quantile`, interpolación lineal). No se extrapola: ningún cuantil va más allá de lo observado.

**Intervalos de confianza — bootstrap por bloques.** Se usa la política vigente desde TDA-07 (aplicada también en TDA-08/09/10/11), sin inventar parámetros: los bloques son **jornadas completas** (`trading_date`), remuestreadas con reemplazo; `n_boot = 300` (configuración, mismo valor que las etapas anteriores); semilla fija `0`; IC percentil al 95 %. Una jornada entra entera o no entra, y nunca se empalma con otra: como los cuantiles y las desviaciones no dependen del orden de las observaciones, el remuestreo no fabrica continuidad entre sesiones, huecos ni contratos. En cada réplica se sortean las jornadas UNA vez y esa selección se aplica a todos los grupos, de modo que las comparaciones entre grupos conservan su dependencia. *Limitación:* remuestrear jornadas supone que jornadas distintas son intercambiables; la volatilidad persiste entre días (TDA-09/TDA-11), así que estos IC probablemente SUBESTIMAN la incertidumbre de los cuantiles extremos.

**Soporte de los cuantiles.** Para cada cuantil se reporta `n_cola` (observaciones en o más allá del cuantil) y `jornadas_cola` (jornadas distintas que las aportan — los extremos se agrupan, así que el número de episodios independientes es menor que `n_cola`). Etiqueta descriptiva: `NO_FIABLE` si `n_cola < 20`, `BAJO` si `< 200`, `OK` en otro caso (constantes ya existentes en TDA-07/TDA-10). En las tablas: ⚠ = `BAJO`, ✖ = `NO_FIABLE`.

**Decisiones metodológicas NUEVAS de esta etapa — EXPLORATORIAS** (fijadas en el código antes de la primera ejecución sobre el conjunto de investigación; NO estaban aprobadas previamente; el veredicto se lee condicionado a ellas):

1. **Deciles de volatilidad causales.** TDA-10 corta los deciles con toda la muestra (fronteras con información futura). Aquí el estado primario (`VOL_DECILE_CAUSAL`) es el decil de `sigma_hat_(t-1)` respecto de los valores observados en **jornadas anteriores** únicamente. No añade parámetros (mínimo de historia: el de TDA-10). Consecuencia: los grupos no tienen el mismo tamaño. La versión de TDA-10 se conserva como `VOL_DECILE_FULLSAMPLE` sólo para comparar.
2. **El veredicto se decide sobre `z_t`.** `r_t` dentro de un segmento o un año mezcla estados de volatilidad, y una mezcla de escalas cambia la forma aunque la distribución condicional sea la misma; por eso las comparaciones sobre `r_t` son contexto y no deciden.
3. **Dos normalizaciones.** El roadmap pide dividir por la desviación del grupo; con colas tan pesadas esa desviación la dominan pocos puntos, así que cada comparación se hace también con la desviación recortada 0,1 % (convención de TDA-07). Sólo hay etiqueta firme si ambas coinciden. *Nueva y exploratoria:* el roadmap sólo pide la primera.
4. **Regla de veredicto.** *Nueva y exploratoria.* Forma del estadístico y valores numéricos de los umbrales tomados de TDA-10 (los valores no se cambian; su aplicación a un estadístico normalizado sí es nueva, ver abajo): `ratio = (máx − mín)/mediana` de la semi-amplitud 1 %–99 % normalizada entre grupos; `SUPERPUESTO` si ratio e IC completo ≤ 0.3; `DIVERGENTE` si ≥ 0.6; `INTERMEDIO` si el IC completo queda entre ambos; `NO_CONCLUYENTE` si el IC cruza un umbral o las dos normalizaciones discrepan. `ESCALA Y FORMA` si alguna agrupación es `DIVERGENTE`; `SÓLO ESCALA` si todas son `SUPERPUESTO`; en otro caso `NO CONCLUYENTE`. Además, si menos del 75 % de las 6 configuraciones causales de `sigma_hat` de TDA-10 reproduce la conclusión, se degrada a `NO CONCLUYENTE`.
5. **Participación en las comparaciones.** *Nueva y exploratoria.* Un grupo sólo entra en una comparación de forma si su cola en ese nivel tiene soporte `OK` (≥ 200 observaciones EN LA COLA, no en el grupo).
6. **`NO CONCLUYENTE` como salida.** *Nueva y exploratoria.* El roadmap sólo prevé `SÓLO ESCALA` y `ESCALA Y FORMA`. Esta etapa añade `NO CONCLUYENTE` como salida prudente cuando la evidencia no sostiene con firmeza ninguno de los dos. No es un tercer hallazgo sobre los datos: significa que la regla no pudo decidir.

**Qué significa `SUPERPUESTO`.** Describe la ESTABILIDAD DE LA SEMI-AMPLITUD NORMALIZADA 1 %–99 % entre grupos (ratio e IC completo ≤ 0.3). No demuestra que el perfil distribucional completo sea igual: no ve la asimetría, ni los percentiles interiores, ni la cola más allá del 1 %/99 %. Esos aspectos se reportan aparte, de forma descriptiva (§9, §10).

**Qué estaba aprobado y qué no (auditoría correctiva).** Reutilizar una constante no significa que su nuevo uso estuviera aprobado:

- Los umbrales 0.3/0.6 se aprobaron en TDA-10 para el ratio de la semi-amplitud 1 %–99 % de `z` **sin dividir** por la desviación del grupo (fila `none` de §9, que reproduce exactamente el estadístico de TDA-10). Aplicarlos al ratio dividido por la desviación del grupo —lo que pide el roadmap de TDA-12— es una extensión decidida en esta etapa.
- El mínimo de 200 era en TDA-10 un tamaño mínimo de GRUPO. Exigirlo a la COLA (decisión 5) es más estricto y es nuevo. §11.E muestra qué cambia con el criterio literal de TDA-10.
- La doble normalización (decisión 3) y la comprobación de robustez de la decisión 4 son nuevas. La segunda compara conclusiones puntuales de otras configuraciones con un veredicto basado en IC: no compara cosas equivalentes (§11.A lo detalla y da la comparación homogénea).
- **El estadístico del veredicto es ciego a la asimetría**: una distribución y su reflejo tienen la misma semi-amplitud. "`SUPERPUESTO`" significa, por tanto, *misma amplitud normalizada*, no *mismo perfil completo*. §9 añade la comparación de cada cola por separado, que se mantiene como evidencia DESCRIPTIVA: no forma parte del veredicto formal y no se incorporará a él sin aprobación específica. §11.E calcula qué diría una regla que la tuviera en cuenta; tampoco decide.

**Causalidad (verificada, bloqueante) — alcance exacto de lo ejecutado** (`TDA12_verificacion_causalidad.csv`).

- `sigma_hat`: la prueba de reconstrucción de TDA-10 se re-ejecutó en esta etapa para 1 configuración(es) — `ewma` con parámetro 60, en 6 barras, elegidas con semilla fija, con `r_1m` como entrada — y pasa. Es el estimador primario; el bloque `clock_adjusted` usa la misma familia y parámetro, y la prueba no se repitió con `r_tilde` como entrada. Las otras 5 configuraciones de `sigma_hat` que aparecen en §11.A NO se re-verificaron aquí: su causalidad descansa en la verificación que hizo TDA-10.
- Decil causal: verificado sobre el mismo conjunto de barras elegibles con el que se construye (excluye el quemado), con dos pruebas — perturbar `sigma_hat` de una barra (no pueden cambiar los deciles de jornadas anteriores ni de las demás barras de su jornada) y truncar la muestra en esa jornada (los deciles deben ser idénticos sin conocer el futuro): 12 de 12 comprobaciones superan ambas (6 barras en el bloque `raw`, 6 barras en el bloque `clock_adjusted`, elegidas con semilla fija). Es una comprobación por muestreo sobre el estimador primario de cada bloque: no es exhaustiva y no se repitió para los deciles causales de las configuraciones alternativas de §11.A, que usan la misma función.

Matices declarados: los CORTES de los segmentos (TDA-06) se eligieron con toda la muestra, aunque la etiqueta de una barra sea reloj puro; y la desviación de grupo usada para normalizar es un descriptor de la muestra completa del grupo (así lo define el roadmap), no una cantidad disponible ex ante.

## 4. Cuantiles incondicionales

**`r_1m`**

| percentil | q | q_ci_lo | q_ci_hi | q_over_std | n_tail | n_days_tail | support |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | -0.002747 | -0.002942 | -0.002556 | -6.554 | 1,915 | 407 | OK |
| 1 | -0.001231 | -0.001279 | -0.001180 | -2.936 | 19,146 | 1,195 | OK |
| 5 | -0.000553 | -0.000569 | -0.000538 | -1.319 | 95,727 | 1,407 | OK |
| 25 | -0.000128 | -0.000131 | -0.000125 | -0.305 | 478,633 | 1,420 | OK |
| 50 | 0.000000 | 0.000000 | 0.000000 | 0.000 | 989,940 | 1,420 | OK |
| 75 | 0.000132 | 0.000130 | 0.000135 | 0.316 | 478,633 | 1,420 | OK |
| 95 | 0.000544 | 0.000529 | 0.000560 | 1.298 | 95,728 | 1,401 | OK |
| 99 | 0.001194 | 0.001143 | 0.001241 | 2.849 | 19,146 | 1,186 | OK |
| 99.9 | 0.002786 | 0.002549 | 0.003019 | 6.646 | 1,915 | 384 | OK |

**`z_ewma_60_raw`**

| percentil | q | q_ci_lo | q_ci_hi | q_over_std | n_tail | n_days_tail | support |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | -5.1648 | -5.2260 | -5.1138 | -4.936 | 1,915 | 1,056 | OK |
| 1 | -2.7282 | -2.7389 | -2.7152 | -2.607 | 19,144 | 1,420 | OK |
| 5 | -1.5737 | -1.5793 | -1.5691 | -1.504 | 95,718 | 1,420 | OK |
| 25 | -0.5323 | -0.5349 | -0.5293 | -0.509 | 478,588 | 1,420 | OK |
| 50 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 989,811 | 1,420 | OK |
| 75 | 0.5485 | 0.5467 | 0.5503 | 0.524 | 478,588 | 1,420 | OK |
| 95 | 1.5497 | 1.5463 | 1.5534 | 1.481 | 95,718 | 1,420 | OK |
| 99 | 2.5958 | 2.5847 | 2.6050 | 2.481 | 19,144 | 1,419 | OK |
| 99.9 | 4.9283 | 4.8654 | 4.9879 | 4.710 | 1,915 | 1,055 | OK |

`q_over_std` = cuantil en múltiplos de la desviación de la muestra. Referencia: una normal tendría ±2.33 en los percentiles 1/99 y ±3.09 en 0,1/99,9.

## 5. Cuantiles por segmento horario

**`r_1m` (retorno; unidades de retorno logarítmico)**

| grupo | n | jornadas | std | p0.1 | p1 | p5 | p25 | p50 | p75 | p95 | p99 | p99.9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00:00-02:00 | 168,922 | 1,415 | 0.000234 | -0.001616 ⚠ | -0.000622 | -0.000276 | -0.000075 | 0.000000 | 0.000077 | 0.000280 | 0.000628 | 0.001751 ⚠ |
| 02:00-03:00 | 84,663 | 1,415 | 0.000286 | -0.001768 ⚠ | -0.000790 | -0.000385 | -0.000108 | 0.000000 | 0.000110 | 0.000387 | 0.000807 | 0.002006 ⚠ |
| 03:00-08:30 | 465,442 | 1,417 | 0.000321 | -0.002000 | -0.000902 | -0.000445 | -0.000129 | 0.000000 | 0.000131 | 0.000437 | 0.000879 | 0.001958 |
| 08:30-09:30 | 84,694 | 1,415 | 0.000528 | -0.003127 ⚠ | -0.001252 | -0.000593 | -0.000167 | 0.000000 | 0.000167 | 0.000585 | 0.001271 | 0.003403 ⚠ |
| 09:30-16:02 | 537,786 | 1,415 | 0.000615 | -0.003452 | -0.001742 | -0.000901 | -0.000252 | 0.000000 | 0.000260 | 0.000868 | 0.001728 | 0.003641 |
| 16:02-20:00 | 237,735 | 1,418 | 0.000317 | -0.002437 | -0.000841 | -0.000350 | -0.000086 | 0.000000 | 0.000090 | 0.000350 | 0.000820 | 0.002361 |
| 20:00-24:00 | 335,288 | 1,419 | 0.000251 | -0.001821 | -0.000719 | -0.000319 | -0.000083 | 0.000000 | 0.000085 | 0.000320 | 0.000694 | 0.001678 |

**`r_1m` en múltiplos de la desviación del grupo**

| grupo | n | p0.1 | p1 | p5 | p25 | p50 | p75 | p95 | p99 | p99.9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00:00-02:00 | 168,922 | -6.897 ⚠ | -2.652 | -1.179 | -0.321 | 0.000 | 0.329 | 1.196 | 2.678 | 7.472 ⚠ |
| 02:00-03:00 | 84,663 | -6.176 ⚠ | -2.761 | -1.345 | -0.377 | 0.000 | 0.385 | 1.353 | 2.821 | 7.009 ⚠ |
| 03:00-08:30 | 465,442 | -6.222 | -2.805 | -1.384 | -0.400 | 0.000 | 0.406 | 1.359 | 2.735 | 6.090 |
| 08:30-09:30 | 84,694 | -5.922 ⚠ | -2.371 | -1.124 | -0.315 | 0.000 | 0.315 | 1.107 | 2.407 | 6.445 ⚠ |
| 09:30-16:02 | 537,786 | -5.613 | -2.832 | -1.465 | -0.410 | 0.000 | 0.424 | 1.411 | 2.809 | 5.920 |
| 16:02-20:00 | 237,735 | -7.682 | -2.650 | -1.102 | -0.272 | 0.000 | 0.284 | 1.102 | 2.585 | 7.443 |
| 20:00-24:00 | 335,288 | -7.260 | -2.868 | -1.271 | -0.333 | 0.000 | 0.340 | 1.274 | 2.770 | 6.692 |

**`z_ewma_60_raw` (retorno estandarizado; unidades de `sigma_hat`)**

| grupo | n | jornadas | std | p0.1 | p1 | p5 | p25 | p50 | p75 | p95 | p99 | p99.9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00:00-02:00 | 168,840 | 1,415 | 0.994 | -4.500 ⚠ | -2.586 | -1.542 | -0.547 | 0.000 | 0.556 | 1.547 | 2.538 | 4.567 ⚠ |
| 02:00-03:00 | 84,663 | 1,415 | 1.224 | -5.543 ⚠ | -3.149 | -1.904 | -0.693 | 0.000 | 0.712 | 1.895 | 3.055 | 5.441 ⚠ |
| 03:00-08:30 | 465,442 | 1,417 | 1.090 | -5.009 | -2.832 | -1.712 | -0.625 | 0.000 | 0.637 | 1.691 | 2.706 | 4.567 |
| 08:30-09:30 | 84,694 | 1,415 | 1.658 | -7.973 ⚠ | -3.199 | -1.852 | -0.660 | 0.000 | 0.665 | 1.779 | 2.981 | 9.948 ⚠ |
| 09:30-16:02 | 537,786 | 1,415 | 1.118 | -5.905 | -2.997 | -1.708 | -0.598 | 0.000 | 0.626 | 1.655 | 2.824 | 5.615 |
| 16:02-20:00 | 237,735 | 1,418 | 0.663 | -3.828 | -1.751 | -0.951 | -0.297 | 0.000 | 0.308 | 0.947 | 1.693 | 3.565 |
| 20:00-24:00 | 335,190 | 1,418 | 0.847 | -4.230 | -2.230 | -1.301 | -0.467 | 0.000 | 0.477 | 1.300 | 2.119 | 3.797 |

**`z_ewma_60_raw` en múltiplos de la desviación del grupo**

| grupo | n | p0.1 | p1 | p5 | p25 | p50 | p75 | p95 | p99 | p99.9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00:00-02:00 | 168,840 | -4.526 ⚠ | -2.601 | -1.551 | -0.550 | 0.000 | 0.559 | 1.555 | 2.553 | 4.594 ⚠ |
| 02:00-03:00 | 84,663 | -4.528 ⚠ | -2.573 | -1.556 | -0.566 | 0.000 | 0.582 | 1.548 | 2.495 | 4.444 ⚠ |
| 03:00-08:30 | 465,442 | -4.598 | -2.600 | -1.571 | -0.574 | 0.000 | 0.585 | 1.552 | 2.484 | 4.192 |
| 08:30-09:30 | 84,694 | -4.808 ⚠ | -1.929 | -1.117 | -0.398 | 0.000 | 0.401 | 1.073 | 1.798 | 5.999 ⚠ |
| 09:30-16:02 | 537,786 | -5.282 | -2.680 | -1.528 | -0.535 | 0.000 | 0.560 | 1.481 | 2.526 | 5.023 |
| 16:02-20:00 | 237,735 | -5.777 | -2.643 | -1.435 | -0.448 | 0.000 | 0.466 | 1.430 | 2.555 | 5.380 |
| 20:00-24:00 | 335,190 | -4.995 | -2.633 | -1.537 | -0.551 | 0.000 | 0.563 | 1.535 | 2.503 | 4.484 |

## 6. Cuantiles por estado previo de volatilidad (decil causal; 0 = más tranquilo, 9 = más volátil)

Tamaño de los estados (el decil causal no reparte por igual: compara cada barra con la historia disponible hasta el día anterior):

| decil | VOL_DECILE_CAUSAL | VOL_DECILE_FULLSAMPLE |
|---:|---:|---:|
| 0 | 229,099 | 191,435 |
| 1 | 233,702 | 191,435 |
| 2 | 208,893 | 191,435 |
| 3 | 194,902 | 191,435 |
| 4 | 193,890 | 191,435 |
| 5 | 189,625 | 191,435 |
| 6 | 188,802 | 191,435 |
| 7 | 177,878 | 191,435 |
| 8 | 158,276 | 191,435 |
| 9 | 138,371 | 191,435 |
| SIN_ESTADO | 1,092 | 180 |

**`r_1m` (retorno; unidades de retorno logarítmico)**

| grupo | n | jornadas | std | p0.1 | p1 | p5 | p25 | p50 | p75 | p95 | p99 | p99.9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 229,099 | 708 | 0.000105 | -0.000513 | -0.000273 | -0.000156 | -0.000052 | 0.000000 | 0.000053 | 0.000155 | 0.000266 | 0.000521 |
| 1 | 233,702 | 972 | 0.000155 | -0.000779 | -0.000396 | -0.000227 | -0.000078 | 0.000000 | 0.000080 | 0.000226 | 0.000388 | 0.000766 |
| 2 | 208,893 | 1,122 | 0.000193 | -0.001022 | -0.000507 | -0.000286 | -0.000095 | 0.000000 | 0.000098 | 0.000282 | 0.000477 | 0.000965 |
| 3 | 194,902 | 1,225 | 0.000242 | -0.001279 ⚠ | -0.000617 | -0.000343 | -0.000112 | 0.000000 | 0.000115 | 0.000339 | 0.000582 | 0.001159 ⚠ |
| 4 | 193,890 | 1,281 | 0.000274 | -0.001485 ⚠ | -0.000755 | -0.000414 | -0.000134 | 0.000000 | 0.000138 | 0.000404 | 0.000698 | 0.001325 ⚠ |
| 5 | 189,625 | 1,291 | 0.000339 | -0.001748 ⚠ | -0.000898 | -0.000495 | -0.000155 | 0.000000 | 0.000167 | 0.000491 | 0.000842 | 0.001580 ⚠ |
| 6 | 188,802 | 1,242 | 0.000394 | -0.002037 ⚠ | -0.001071 | -0.000603 | -0.000191 | 0.000000 | 0.000200 | 0.000590 | 0.001015 | 0.001863 ⚠ |
| 7 | 177,878 | 1,100 | 0.000477 | -0.002364 ⚠ | -0.001327 | -0.000745 | -0.000236 | 0.000000 | 0.000243 | 0.000722 | 0.001238 | 0.002350 ⚠ |
| 8 | 158,276 | 862 | 0.000602 | -0.002877 ⚠ | -0.001623 | -0.000930 | -0.000296 | 0.000000 | 0.000306 | 0.000913 | 0.001591 | 0.002921 ⚠ |
| 9 | 138,371 | 498 | 0.001026 | -0.005420 ⚠ | -0.002802 | -0.001510 | -0.000464 | 0.000000 | 0.000459 | 0.001514 | 0.002878 | 0.005817 ⚠ |

**`r_1m` en múltiplos de la desviación del grupo**

| grupo | n | p0.1 | p1 | p5 | p25 | p50 | p75 | p95 | p99 | p99.9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 229,099 | -4.904 | -2.604 | -1.486 | -0.496 | 0.000 | 0.511 | 1.476 | 2.539 | 4.981 |
| 1 | 233,702 | -5.041 | -2.564 | -1.467 | -0.503 | 0.000 | 0.519 | 1.459 | 2.507 | 4.954 |
| 2 | 208,893 | -5.308 | -2.631 | -1.486 | -0.496 | 0.000 | 0.509 | 1.463 | 2.475 | 5.009 |
| 3 | 194,902 | -5.288 ⚠ | -2.551 | -1.419 | -0.465 | 0.000 | 0.477 | 1.402 | 2.404 | 4.792 ⚠ |
| 4 | 193,890 | -5.424 ⚠ | -2.757 | -1.511 | -0.488 | 0.000 | 0.503 | 1.475 | 2.547 | 4.840 ⚠ |
| 5 | 189,625 | -5.164 ⚠ | -2.651 | -1.462 | -0.459 | 0.000 | 0.492 | 1.451 | 2.486 | 4.665 ⚠ |
| 6 | 188,802 | -5.169 ⚠ | -2.718 | -1.531 | -0.485 | 0.000 | 0.508 | 1.497 | 2.576 | 4.728 ⚠ |
| 7 | 177,878 | -4.951 ⚠ | -2.778 | -1.560 | -0.493 | 0.000 | 0.508 | 1.511 | 2.594 | 4.922 ⚠ |
| 8 | 158,276 | -4.779 ⚠ | -2.696 | -1.545 | -0.491 | 0.000 | 0.508 | 1.517 | 2.643 | 4.853 ⚠ |
| 9 | 138,371 | -5.284 ⚠ | -2.732 | -1.472 | -0.453 | 0.000 | 0.448 | 1.476 | 2.805 | 5.670 ⚠ |

**`z_ewma_60_raw` (retorno estandarizado; unidades de `sigma_hat`)**

| grupo | n | jornadas | std | p0.1 | p1 | p5 | p25 | p50 | p75 | p95 | p99 | p99.9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 229,099 | 708 | 1.071 | -5.075 | -2.760 | -1.616 | -0.566 | 0.000 | 0.577 | 1.607 | 2.685 | 5.064 |
| 1 | 233,702 | 972 | 1.076 | -5.241 | -2.740 | -1.587 | -0.548 | 0.000 | 0.563 | 1.578 | 2.642 | 5.130 |
| 2 | 208,893 | 1,122 | 1.065 | -5.562 | -2.779 | -1.580 | -0.536 | 0.000 | 0.551 | 1.556 | 2.612 | 5.161 |
| 3 | 194,902 | 1,225 | 1.097 | -5.765 ⚠ | -2.781 | -1.561 | -0.521 | 0.000 | 0.536 | 1.541 | 2.623 | 5.204 ⚠ |
| 4 | 193,890 | 1,281 | 1.032 | -5.435 ⚠ | -2.804 | -1.569 | -0.515 | 0.000 | 0.535 | 1.530 | 2.586 | 4.954 ⚠ |
| 5 | 189,625 | 1,291 | 1.064 | -5.348 ⚠ | -2.776 | -1.554 | -0.507 | 0.000 | 0.538 | 1.538 | 2.590 | 4.890 ⚠ |
| 6 | 188,802 | 1,242 | 1.024 | -5.037 ⚠ | -2.724 | -1.574 | -0.520 | 0.000 | 0.544 | 1.534 | 2.561 | 4.749 ⚠ |
| 7 | 177,878 | 1,100 | 1.004 | -4.861 ⚠ | -2.688 | -1.571 | -0.527 | 0.000 | 0.541 | 1.524 | 2.508 | 4.649 ⚠ |
| 8 | 158,276 | 862 | 0.997 | -4.553 ⚠ | -2.630 | -1.564 | -0.525 | 0.000 | 0.546 | 1.534 | 2.539 | 4.554 ⚠ |
| 9 | 138,371 | 498 | 0.982 | -4.422 ⚠ | -2.527 | -1.536 | -0.547 | 0.000 | 0.542 | 1.533 | 2.527 | 4.432 ⚠ |

**`z_ewma_60_raw` en múltiplos de la desviación del grupo**

| grupo | n | p0.1 | p1 | p5 | p25 | p50 | p75 | p95 | p99 | p99.9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 229,099 | -4.741 | -2.578 | -1.510 | -0.528 | 0.000 | 0.539 | 1.502 | 2.508 | 4.730 |
| 1 | 233,702 | -4.869 | -2.545 | -1.475 | -0.509 | 0.000 | 0.523 | 1.466 | 2.454 | 4.766 |
| 2 | 208,893 | -5.221 | -2.609 | -1.483 | -0.503 | 0.000 | 0.517 | 1.461 | 2.452 | 4.845 |
| 3 | 194,902 | -5.253 ⚠ | -2.534 | -1.422 | -0.474 | 0.000 | 0.489 | 1.404 | 2.390 | 4.742 ⚠ |
| 4 | 193,890 | -5.266 ⚠ | -2.716 | -1.520 | -0.499 | 0.000 | 0.519 | 1.482 | 2.506 | 4.800 ⚠ |
| 5 | 189,625 | -5.027 ⚠ | -2.609 | -1.460 | -0.477 | 0.000 | 0.506 | 1.446 | 2.434 | 4.596 ⚠ |
| 6 | 188,802 | -4.918 ⚠ | -2.660 | -1.537 | -0.508 | 0.000 | 0.531 | 1.498 | 2.500 | 4.637 ⚠ |
| 7 | 177,878 | -4.839 ⚠ | -2.676 | -1.564 | -0.524 | 0.000 | 0.539 | 1.518 | 2.497 | 4.628 ⚠ |
| 8 | 158,276 | -4.564 ⚠ | -2.636 | -1.568 | -0.527 | 0.000 | 0.547 | 1.538 | 2.545 | 4.566 ⚠ |
| 9 | 138,371 | -4.501 ⚠ | -2.572 | -1.563 | -0.557 | 0.000 | 0.552 | 1.561 | 2.573 | 4.512 ⚠ |

La misma tabla con las fronteras de TDA-10 (`VOL_DECILE_FULLSAMPLE`, retrospectivas) está en `TDA12_cuantiles_condicionales.csv`.

## 7. Cuantiles por año

**`r_1m` (retorno; unidades de retorno logarítmico)**

| grupo | n | jornadas | std | p0.1 | p1 | p5 | p25 | p50 | p75 | p95 | p99 | p99.9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2019 | 7,406 | 6 | 0.000131 | -0.000802 ✖ | -0.000401 ⚠ | -0.000172 | -0.000057 | 0.000000 | 0.000057 | 0.000172 | 0.000372 ⚠ | 0.000879 ✖ |
| 2020 | 342,648 | 259 | 0.000568 | -0.003828 | -0.001709 | -0.000742 | -0.000165 | 0.000000 | 0.000174 | 0.000735 | 0.001674 | 0.003973 |
| 2021 | 349,039 | 259 | 0.000303 | -0.001904 | -0.000923 | -0.000424 | -0.000106 | 0.000000 | 0.000110 | 0.000422 | 0.000887 | 0.001873 |
| 2022 | 349,996 | 258 | 0.000521 | -0.002942 | -0.001478 | -0.000739 | -0.000195 | 0.000000 | 0.000194 | 0.000728 | 0.001479 | 0.003052 |
| 2023 | 349,636 | 258 | 0.000305 | -0.001830 | -0.000902 | -0.000429 | -0.000108 | 0.000000 | 0.000113 | 0.000431 | 0.000876 | 0.001827 |
| 2024 | 352,261 | 259 | 0.000282 | -0.001800 | -0.000851 | -0.000395 | -0.000097 | 0.000000 | 0.000102 | 0.000388 | 0.000795 | 0.001615 |
| 2025 | 163,544 | 121 | 0.000481 | -0.003098 ⚠ | -0.001371 | -0.000606 | -0.000139 | 0.000000 | 0.000146 | 0.000589 | 0.001315 | 0.003329 ⚠ |

**`r_1m` en múltiplos de la desviación del grupo**

| grupo | n | p0.1 | p1 | p5 | p25 | p50 | p75 | p95 | p99 | p99.9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2019 | 7,406 | -6.120 ✖ | -3.066 ⚠ | -1.311 | -0.433 | 0.000 | 0.434 | 1.312 | 2.838 ⚠ | 6.708 ✖ |
| 2020 | 342,648 | -6.735 | -3.007 | -1.304 | -0.290 | 0.000 | 0.305 | 1.292 | 2.945 | 6.989 |
| 2021 | 349,039 | -6.281 | -3.045 | -1.400 | -0.349 | 0.000 | 0.362 | 1.392 | 2.928 | 6.179 |
| 2022 | 349,996 | -5.643 | -2.835 | -1.418 | -0.374 | 0.000 | 0.372 | 1.396 | 2.837 | 5.854 |
| 2023 | 349,636 | -5.990 | -2.952 | -1.405 | -0.354 | 0.000 | 0.370 | 1.411 | 2.867 | 5.983 |
| 2024 | 352,261 | -6.390 | -3.020 | -1.401 | -0.344 | 0.000 | 0.362 | 1.377 | 2.824 | 5.734 |
| 2025 | 163,544 | -6.447 ⚠ | -2.854 | -1.260 | -0.290 | 0.000 | 0.303 | 1.226 | 2.737 | 6.928 ⚠ |

**`z_ewma_60_raw` (retorno estandarizado; unidades de `sigma_hat`)**

| grupo | n | jornadas | std | p0.1 | p1 | p5 | p25 | p50 | p75 | p95 | p99 | p99.9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2019 | 7,226 | 6 | 1.052 | -5.622 ✖ | -2.789 ⚠ | -1.532 | -0.532 | 0.000 | 0.565 | 1.524 | 2.630 ⚠ | 6.191 ✖ |
| 2020 | 342,648 | 259 | 1.023 | -5.268 | -2.740 | -1.569 | -0.531 | 0.000 | 0.553 | 1.554 | 2.614 | 4.993 |
| 2021 | 349,039 | 259 | 1.029 | -5.397 | -2.771 | -1.562 | -0.528 | 0.000 | 0.547 | 1.546 | 2.615 | 4.951 |
| 2022 | 349,996 | 258 | 1.072 | -4.985 | -2.663 | -1.567 | -0.541 | 0.000 | 0.539 | 1.545 | 2.627 | 5.081 |
| 2023 | 349,636 | 258 | 1.051 | -4.872 | -2.680 | -1.576 | -0.538 | 0.000 | 0.551 | 1.562 | 2.599 | 5.012 |
| 2024 | 352,261 | 259 | 1.047 | -5.197 | -2.765 | -1.592 | -0.528 | 0.000 | 0.557 | 1.549 | 2.534 | 4.566 |
| 2025 | 163,544 | 121 | 1.063 | -5.304 ⚠ | -2.767 | -1.581 | -0.521 | 0.000 | 0.537 | 1.537 | 2.587 | 4.663 ⚠ |

**`z_ewma_60_raw` en múltiplos de la desviación del grupo**

| grupo | n | p0.1 | p1 | p5 | p25 | p50 | p75 | p95 | p99 | p99.9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2019 | 7,226 | -5.346 ✖ | -2.652 ⚠ | -1.456 | -0.506 | 0.000 | 0.537 | 1.449 | 2.500 ⚠ | 5.886 ✖ |
| 2020 | 342,648 | -5.152 | -2.679 | -1.535 | -0.519 | 0.000 | 0.541 | 1.520 | 2.556 | 4.883 |
| 2021 | 349,039 | -5.245 | -2.693 | -1.517 | -0.513 | 0.000 | 0.532 | 1.502 | 2.541 | 4.811 |
| 2022 | 349,996 | -4.648 | -2.483 | -1.461 | -0.504 | 0.000 | 0.503 | 1.441 | 2.449 | 4.738 |
| 2023 | 349,636 | -4.636 | -2.550 | -1.500 | -0.512 | 0.000 | 0.524 | 1.486 | 2.473 | 4.768 |
| 2024 | 352,261 | -4.962 | -2.640 | -1.520 | -0.504 | 0.000 | 0.532 | 1.479 | 2.420 | 4.360 |
| 2025 | 163,544 | -4.988 ⚠ | -2.602 | -1.487 | -0.490 | 0.000 | 0.505 | 1.445 | 2.433 | 4.386 ⚠ |

## 8. Fiabilidad de los cuantiles extremos (0,1 y 99,9)

De 100 cuantiles extremos del bloque causal (muestra completa, segmentos, deciles causales y años; `r` y `z`): 44 tienen soporte `BAJO` (< 200 observaciones de cola) y 4 `NO_FIABLE` (< 20). El peor caso: `r_1m`, YEAR = `2019`, percentil 0.1, sostenido por 8 observaciones de 2 jornadas. En los percentiles 1 y 99 —los que usa el veredicto— 4 grupo(s)-cuantil no alcanzan soporte `OK` (y quedan fuera de la comparación).

**`r_1m` por segmento**

| grupo | percentil | n | n_cola | jornadas_cola | cuantil | IC95_lo | IC95_hi | semiancho_IC_relativo | soporte |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00:00-02:00 | 0.1 | 168,922 | 169 | 49 | -0.001616 | -0.001968 | -0.001342 | 0.194 | BAJO |
| 00:00-02:00 | 99.9 | 168,922 | 169 | 43 | 0.001751 | 0.001399 | 0.002191 | 0.226 | BAJO |
| 02:00-03:00 | 0.1 | 84,663 | 85 | 39 | -0.001768 | -0.001982 | -0.001571 | 0.116 | BAJO |
| 02:00-03:00 | 99.9 | 84,663 | 85 | 30 | 0.002006 | 0.001667 | 0.002413 | 0.186 | BAJO |
| 03:00-08:30 | 0.1 | 465,442 | 466 | 122 | -0.002000 | -0.002246 | -0.001795 | 0.113 | OK |
| 03:00-08:30 | 99.9 | 465,442 | 466 | 101 | 0.001958 | 0.001737 | 0.002209 | 0.120 | OK |
| 08:30-09:30 | 0.1 | 84,694 | 85 | 59 | -0.003127 | -0.003659 | -0.002842 | 0.131 | BAJO |
| 08:30-09:30 | 99.9 | 84,694 | 85 | 67 | 0.003403 | 0.002944 | 0.003661 | 0.105 | BAJO |
| 09:30-16:02 | 0.1 | 537,786 | 538 | 158 | -0.003452 | -0.003777 | -0.003200 | 0.084 | OK |
| 09:30-16:02 | 99.9 | 537,786 | 538 | 144 | 0.003641 | 0.003276 | 0.004044 | 0.105 | OK |
| 16:02-20:00 | 0.1 | 237,735 | 238 | 93 | -0.002437 | -0.002727 | -0.002150 | 0.118 | OK |
| 16:02-20:00 | 99.9 | 237,735 | 238 | 104 | 0.002361 | 0.002023 | 0.002762 | 0.156 | OK |
| 20:00-24:00 | 0.1 | 335,288 | 336 | 86 | -0.001821 | -0.002039 | -0.001552 | 0.134 | OK |
| 20:00-24:00 | 99.9 | 335,288 | 336 | 82 | 0.001678 | 0.001486 | 0.001885 | 0.119 | OK |

**`r_1m` por año**

| grupo | percentil | n | n_cola | jornadas_cola | cuantil | IC95_lo | IC95_hi | semiancho_IC_relativo | soporte |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2019 | 0.1 | 7,406 | 8 | 2 | -0.000802 | -0.000941 | -0.000556 | 0.240 | NO_FIABLE |
| 2019 | 99.9 | 7,406 | 8 | 3 | 0.000879 | 0.000509 | 0.001239 | 0.415 | NO_FIABLE |
| 2020 | 0.1 | 342,648 | 343 | 46 | -0.003828 | -0.004325 | -0.003203 | 0.147 | OK |
| 2020 | 99.9 | 342,648 | 343 | 44 | 0.003973 | 0.003258 | 0.004609 | 0.170 | OK |
| 2021 | 0.1 | 349,039 | 350 | 106 | -0.001904 | -0.002012 | -0.001758 | 0.067 | OK |
| 2021 | 99.9 | 349,039 | 350 | 94 | 0.001873 | 0.001713 | 0.002008 | 0.079 | OK |
| 2022 | 0.1 | 349,996 | 350 | 128 | -0.002942 | -0.003091 | -0.002810 | 0.048 | OK |
| 2022 | 99.9 | 349,996 | 350 | 122 | 0.003052 | 0.002874 | 0.003198 | 0.053 | OK |
| 2023 | 0.1 | 349,636 | 350 | 119 | -0.001830 | -0.001935 | -0.001708 | 0.062 | OK |
| 2023 | 99.9 | 349,636 | 350 | 118 | 0.001827 | 0.001676 | 0.002017 | 0.093 | OK |
| 2024 | 0.1 | 352,261 | 353 | 112 | -0.001800 | -0.001961 | -0.001624 | 0.094 | OK |
| 2024 | 99.9 | 352,261 | 353 | 98 | 0.001615 | 0.001468 | 0.001829 | 0.112 | OK |
| 2025 | 0.1 | 163,544 | 164 | 28 | -0.003098 | -0.003826 | -0.002379 | 0.234 | BAJO |
| 2025 | 99.9 | 163,544 | 164 | 31 | 0.003329 | 0.002269 | 0.004376 | 0.316 | BAJO |

**`z_ewma_60_raw` por decil causal**

| grupo | percentil | n | n_cola | jornadas_cola | cuantil | IC95_lo | IC95_hi | semiancho_IC_relativo | soporte |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1 | 229,099 | 230 | 185 | -5.0754 | -5.2187 | -4.9308 | 0.028 | OK |
| 0 | 99.9 | 229,099 | 230 | 189 | 5.0635 | 4.9045 | 5.2127 | 0.030 | OK |
| 1 | 0.1 | 233,702 | 234 | 210 | -5.2409 | -5.4352 | -5.0994 | 0.032 | OK |
| 1 | 99.9 | 233,702 | 234 | 216 | 5.1301 | 4.9733 | 5.3866 | 0.040 | OK |
| 2 | 0.1 | 208,893 | 209 | 202 | -5.5620 | -5.7120 | -5.2825 | 0.039 | OK |
| 2 | 99.9 | 208,893 | 209 | 200 | 5.1607 | 4.9644 | 5.3375 | 0.036 | OK |
| 3 | 0.1 | 194,902 | 195 | 187 | -5.7651 | -5.9904 | -5.5951 | 0.034 | BAJO |
| 3 | 99.9 | 194,902 | 195 | 185 | 5.2043 | 5.0140 | 5.4453 | 0.041 | BAJO |
| 4 | 0.1 | 193,890 | 194 | 184 | -5.4353 | -5.6532 | -5.2515 | 0.037 | BAJO |
| 4 | 99.9 | 193,890 | 194 | 186 | 4.9544 | 4.7910 | 5.1148 | 0.033 | BAJO |
| 5 | 0.1 | 189,625 | 190 | 185 | -5.3483 | -5.6269 | -5.1331 | 0.046 | BAJO |
| 5 | 99.9 | 189,625 | 190 | 177 | 4.8899 | 4.7343 | 5.0412 | 0.031 | BAJO |
| 6 | 0.1 | 188,802 | 189 | 182 | -5.0365 | -5.2419 | -4.8367 | 0.040 | BAJO |
| 6 | 99.9 | 188,802 | 189 | 175 | 4.7488 | 4.5296 | 4.9805 | 0.047 | BAJO |
| 7 | 0.1 | 177,878 | 178 | 169 | -4.8605 | -5.0169 | -4.7075 | 0.032 | BAJO |
| 7 | 99.9 | 177,878 | 178 | 161 | 4.6489 | 4.4756 | 4.8292 | 0.038 | BAJO |
| 8 | 0.1 | 158,276 | 159 | 149 | -4.5526 | -4.7776 | -4.4218 | 0.039 | BAJO |
| 8 | 99.9 | 158,276 | 159 | 139 | 4.5540 | 4.4233 | 4.8100 | 0.042 | BAJO |
| 9 | 0.1 | 138,371 | 139 | 99 | -4.4219 | -4.5832 | -4.3134 | 0.031 | BAJO |
| 9 | 99.9 | 138,371 | 139 | 100 | 4.4323 | 4.2634 | 4.5759 | 0.035 | BAJO |

**Cómo leerlo.** Un percentil 0,1 o 99,9 es, por definición, el valor que deja fuera una observación de cada mil: en un grupo de 85.000 barras descansa en ~85 puntos, que además se concentran en pocas jornadas. Esos cuantiles son órdenes de magnitud, no valores precisos; `semiancho_IC_relativo` dice cuánto podría moverse (y, por lo dicho en §3, es una cota optimista). El resto de combinaciones está en `TDA12_cuantiles_condicionales.csv`.

## 9. Escala frente a forma

Si la distribución sólo cambiara de escala entre grupos, al dividir los cuantiles de cada grupo por su propia desviación todos los perfiles coincidirían. Se mide con la semi-amplitud `(q_alto − q_bajo)/2` normalizada de cada grupo, y `ratio = (máx − mín)/mediana` entre grupos (0 = misma semi-amplitud normalizada en todos los grupos). La implicación va en un solo sentido: sólo escala ⇒ misma semi-amplitud; la misma semi-amplitud NO garantiza el mismo perfil. Por eso `SUPERPUESTO` en las tablas se lee como "semi-amplitud normalizada estable", no como "distribuciones iguales". Ver `TDA12_perfiles_cuantiles_normalizados.png` (perfiles dibujados juntos) y `TDA12_semiamplitud_normalizada_por_grupo.png` (semi-amplitud grupo a grupo con IC).

**Percentiles 1 %–99 % — `z_ewma_60_raw` (DECIDE el veredicto)**

| agrupación | normalización | disponibles | comparados | ratio | ratio_ci_lo | ratio_ci_hi | mín (grupo) | máx (grupo) | excluidos (soporte) | sin normalización válida | % obs. comparadas | etiqueta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SEGMENT | std | 7 | 7 | 0.288 | 0.243 | 0.341 | 1.863 (08:30-09:30) | 2.603 (09:30-16:02) | — | — | 100.0 | NO_CONCLUYENTE |
| SEGMENT | std_trimmed | 7 | 7 | 0.104 | 0.090 | 0.126 | 2.531 (08:30-09:30) | 2.808 (16:02-20:00) | — | — | 100.0 | SUPERPUESTO |
| SEGMENT | none | 7 | 7 | 0.498 | 0.479 | 0.519 | 1.722 (16:02-20:00) | 3.102 (02:00-03:00) | — | — | 100.0 | INTERMEDIO |
| VOL_DECILE_CAUSAL | std | 10 | 10 | 0.058 | 0.040 | 0.106 | 2.462 (3) | 2.611 (4) | — | — | 100.0 | SUPERPUESTO |
| VOL_DECILE_CAUSAL | std_trimmed | 10 | 10 | 0.033 | 0.026 | 0.045 | 2.654 (9) | 2.744 (5) | — | — | 100.0 | SUPERPUESTO |
| VOL_DECILE_CAUSAL | none | 10 | 10 | 0.073 | 0.062 | 0.083 | 2.527 (9) | 2.722 (0) | — | — | 100.0 | SUPERPUESTO |
| VOL_DECILE_FULLSAMPLE | std | 10 | 10 | 0.048 | 0.038 | 0.096 | 2.482 (1) | 2.606 (8) | — | — | 100.0 | SUPERPUESTO |
| VOL_DECILE_FULLSAMPLE | std_trimmed | 10 | 10 | 0.034 | 0.026 | 0.043 | 2.653 (9) | 2.745 (5) | — | — | 100.0 | SUPERPUESTO |
| VOL_DECILE_FULLSAMPLE | none | 10 | 10 | 0.070 | 0.061 | 0.079 | 2.532 (9) | 2.719 (0) | — | — | 100.0 | SUPERPUESTO |
| YEAR | std | 7 | 6 | 0.060 | 0.044 | 0.095 | 2.466 (2022) | 2.618 (2020) | 2019 | — | 99.6 | SUPERPUESTO |
| YEAR | std_trimmed | 7 | 6 | 0.015 | 0.012 | 0.023 | 2.689 (2023) | 2.731 (2025) | 2019 | — | 99.6 | SUPERPUESTO |
| YEAR | none | 7 | 6 | 0.020 | 0.015 | 0.030 | 2.640 (2023) | 2.693 (2021) | 2019 | — | 99.6 | SUPERPUESTO |

**Cada cola por separado y asimetría entre grupos — `z_ewma_60_raw`, percentiles 1 % y 99 % (DESCRIPTIVO, no decide)**

La semi-amplitud promedia las dos colas. Esta tabla aplica el mismo ratio a `|q1|/escala` (cola izquierda) y a `q99/escala` (cola derecha), y da el rango (máx − mín) de la asimetría entre grupos:

| agrupación | normalización | comparados | cola izq.: ratio [IC95] | etiq. izq. | cola der.: ratio [IC95] | etiq. der. | rango de asimetría [IC95] |
|---:|---:|---:|---:|---:|---:|---:|---:|
| SEGMENT | std | 7 | 0.289 [0.241, 0.342] | NO_CONCLUYENTE | 0.303 [0.261, 0.361] | NO_CONCLUYENTE | 0.052 [0.032, 0.090] |
| SEGMENT | std_trimmed | 7 | 0.088 [0.069, 0.118] | SUPERPUESTO | 0.124 [0.103, 0.150] | SUPERPUESTO | 0.052 [0.032, 0.090] |
| SEGMENT | none | 7 | 0.511 [0.486, 0.539] | INTERMEDIO | 0.503 [0.481, 0.527] | INTERMEDIO | 0.052 [0.032, 0.090] |
| VOL_DECILE_CAUSAL | std | 10 | 0.070 [0.053, 0.119] | SUPERPUESTO | 0.073 [0.049, 0.121] | SUPERPUESTO | 0.081 [0.062, 0.112] |
| VOL_DECILE_CAUSAL | std_trimmed | 10 | 0.071 [0.058, 0.090] | SUPERPUESTO | 0.022 [0.016, 0.042] | SUPERPUESTO | 0.081 [0.062, 0.112] |
| VOL_DECILE_CAUSAL | none | 10 | 0.101 [0.086, 0.119] | SUPERPUESTO | 0.068 [0.055, 0.084] | SUPERPUESTO | 0.081 [0.062, 0.112] |
| YEAR | std | 6 | 0.080 [0.058, 0.116] | SUPERPUESTO | 0.055 [0.044, 0.098] | SUPERPUESTO | 0.073 [0.055, 0.094] |
| YEAR | std_trimmed | 6 | 0.037 [0.029, 0.053] | SUPERPUESTO | 0.039 [0.028, 0.050] | SUPERPUESTO | 0.073 [0.055, 0.094] |
| YEAR | none | 6 | 0.039 [0.033, 0.058] | SUPERPUESTO | 0.036 [0.028, 0.047] | SUPERPUESTO | 0.073 [0.055, 0.094] |

Las filas `none` no dividen por la desviación del grupo (cuantiles de `z` tal cual): miden si `sigma_hat` por sí solo deja a todos los grupos en la misma escala. Son descriptivas, no deciden: SEGMENT: 0.498 [0.479, 0.519]; VOL_DECILE_CAUSAL: 0.073 [0.062, 0.083]; YEAR: 0.020 [0.015, 0.030].

**Percentiles 1 %–99 % — `r_1m` (contexto, NO decide)**

| agrupación | normalización | disponibles | comparados | ratio | ratio_ci_lo | ratio_ci_hi | mín (grupo) | máx (grupo) | excluidos (soporte) | sin normalización válida | % obs. comparadas | etiqueta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SEGMENT | std | 7 | 7 | 0.156 | 0.108 | 0.235 | 2.389 (08:30-09:30) | 2.821 (09:30-16:02) | — | — | 100.0 | SUPERPUESTO |
| SEGMENT | std_trimmed | 7 | 7 | 0.031 | 0.028 | 0.060 | 2.984 (09:30-16:02) | 3.078 (20:00-24:00) | — | — | 100.0 | SUPERPUESTO |
| VOL_DECILE_CAUSAL | std | 10 | 10 | 0.111 | 0.080 | 0.156 | 2.478 (3) | 2.768 (9) | — | — | 100.0 | SUPERPUESTO |
| VOL_DECILE_CAUSAL | std_trimmed | 10 | 10 | 0.068 | 0.045 | 0.090 | 2.725 (2) | 2.913 (9) | — | — | 100.0 | SUPERPUESTO |
| VOL_DECILE_FULLSAMPLE | std | 10 | 10 | 0.084 | 0.068 | 0.136 | 2.500 (4) | 2.715 (9) | — | — | 100.0 | SUPERPUESTO |
| VOL_DECILE_FULLSAMPLE | std_trimmed | 10 | 10 | 0.060 | 0.040 | 0.085 | 2.696 (1) | 2.860 (9) | — | — | 100.0 | SUPERPUESTO |
| YEAR | std | 7 | 6 | 0.066 | 0.045 | 0.111 | 2.795 (2025) | 2.987 (2021) | 2019 | — | 99.6 | SUPERPUESTO |
| YEAR | std_trimmed | 7 | 6 | 0.047 | 0.030 | 0.069 | 3.059 (2022) | 3.207 (2020) | 2019 | — | 99.6 | SUPERPUESTO |

**Otros niveles — `z_ewma_60_raw` (descriptivo, sin umbral)**

Percentiles 5 %–95 %:

| agrupación | normalización | disponibles | comparados | ratio | ratio_ci_lo | ratio_ci_hi | mín (grupo) | máx (grupo) | excluidos (soporte) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SEGMENT | std | 7 | 7 | 0.304 | 0.257 | 0.356 | 1.095 (08:30-09:30) | 1.562 (03:00-08:30) | — |
| SEGMENT | std_trimmed | 7 | 7 | 0.074 | 0.060 | 0.095 | 1.487 (08:30-09:30) | 1.605 (03:00-08:30) | — |
| SEGMENT | none | 7 | 7 | 0.565 | 0.553 | 0.578 | 0.949 (16:02-20:00) | 1.900 (02:00-03:00) | — |
| VOL_DECILE_CAUSAL | std | 10 | 10 | 0.099 | 0.070 | 0.140 | 1.413 (3) | 1.562 (9) | — |
| VOL_DECILE_CAUSAL | std_trimmed | 10 | 10 | 0.027 | 0.023 | 0.035 | 1.568 (3) | 1.612 (9) | — |
| VOL_DECILE_CAUSAL | none | 10 | 10 | 0.050 | 0.041 | 0.059 | 1.535 (9) | 1.612 (0) | — |
| YEAR | std | 7 | 7 | 0.051 | 0.040 | 0.093 | 1.451 (2022) | 1.527 (2020) | — |
| YEAR | std_trimmed | 7 | 7 | 0.050 | 0.026 | 0.091 | 1.518 (2019) | 1.598 (2023) | — |
| YEAR | none | 7 | 7 | 0.027 | 0.011 | 0.069 | 1.528 (2019) | 1.570 (2024) | — |

Percentiles 25 %–75 %:

| agrupación | normalización | disponibles | comparados | ratio | ratio_ci_lo | ratio_ci_hi | mín (grupo) | máx (grupo) | excluidos (soporte) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SEGMENT | std | 7 | 7 | 0.324 | 0.271 | 0.378 | 0.400 (08:30-09:30) | 0.579 (03:00-08:30) | — |
| SEGMENT | std_trimmed | 7 | 7 | 0.178 | 0.169 | 0.189 | 0.494 (16:02-20:00) | 0.596 (03:00-08:30) | — |
| SEGMENT | none | 7 | 7 | 0.653 | 0.639 | 0.668 | 0.303 (16:02-20:00) | 0.702 (02:00-03:00) | — |
| VOL_DECILE_CAUSAL | std | 10 | 10 | 0.141 | 0.111 | 0.181 | 0.482 (3) | 0.554 (9) | — |
| VOL_DECILE_CAUSAL | std_trimmed | 10 | 10 | 0.069 | 0.059 | 0.083 | 0.534 (4) | 0.572 (9) | — |
| VOL_DECILE_CAUSAL | none | 10 | 10 | 0.091 | 0.082 | 0.103 | 0.523 (5) | 0.571 (0) | — |
| YEAR | std | 7 | 7 | 0.061 | 0.044 | 0.119 | 0.498 (2025) | 0.530 (2020) | — |
| YEAR | std_trimmed | 7 | 7 | 0.027 | 0.023 | 0.085 | 0.540 (2025) | 0.555 (2023) | — |
| YEAR | none | 7 | 7 | 0.036 | 0.020 | 0.094 | 0.529 (2025) | 0.549 (2019) | — |

Percentiles 0,1 %–99,9 % (cola profunda; sólo grupos con soporte `OK` — ver columna `excluidos`):

| agrupación | normalización | disponibles | comparados | ratio | ratio_ci_lo | ratio_ci_hi | mín (grupo) | máx (grupo) | excluidos (soporte) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SEGMENT | std | 7 | 4 | 0.239 | 0.201 | 0.281 | 4.395 (03:00-08:30) | 5.579 (16:02-20:00) | 00:00-02:00;02:00-03:00;08:30-09:30 |
| SEGMENT | std_trimmed | 7 | 4 | 0.295 | 0.254 | 0.336 | 4.517 (03:00-08:30) | 6.029 (16:02-20:00) | 00:00-02:00;02:00-03:00;08:30-09:30 |
| SEGMENT | none | 7 | 4 | 0.469 | 0.426 | 0.508 | 3.696 (16:02-20:00) | 5.760 (09:30-16:02) | 00:00-02:00;02:00-03:00;08:30-09:30 |
| VOL_DECILE_CAUSAL | std | 10 | 3 | 0.062 | 0.014 | 0.111 | 4.735 (0) | 5.033 (2) | 3;4;5;6;7;8;9 |
| VOL_DECILE_CAUSAL | std_trimmed | 10 | 3 | 0.076 | 0.039 | 0.105 | 5.001 (0) | 5.395 (2) | 3;4;5;6;7;8;9 |
| VOL_DECILE_CAUSAL | none | 10 | 3 | 0.056 | 0.021 | 0.085 | 5.069 (0) | 5.361 (2) | 3;4;5;6;7;8;9 |
| YEAR | std | 7 | 5 | 0.078 | 0.064 | 0.115 | 4.661 (2024) | 5.028 (2021) | 2019;2025 |
| YEAR | std_trimmed | 7 | 5 | 0.054 | 0.036 | 0.080 | 4.968 (2024) | 5.246 (2021) | 2019;2025 |
| YEAR | none | 7 | 5 | 0.058 | 0.039 | 0.085 | 4.882 (2024) | 5.174 (2021) | 2019;2025 |

**Nota sobre el IC del ratio.** `(máx − mín)/mediana` es un estadístico de rango: el ruido de muestreo sólo puede aumentarlo, así que en las réplicas bootstrap tiende a salir algo mayor que en la muestra y el IC percentil puede quedar por encima del valor puntual. La etiqueta exige que el punto Y el IC completo estén en la misma zona, lo que es conservador para declarar `SUPERPUESTO`.

## 10. Asimetría de colas

`asimetría = (|q_bajo| − q_alto) / media de ambos`: positivo = cola izquierda (caídas) más larga que la derecha. Muestra completa:

| series | par | asymmetry | asymmetry_ci_lo | asymmetry_ci_hi |
|---:|---:|---:|---:|---:|
| r_1m | 0.1/99.9 | -0.0141 | -0.0400 | 0.0101 |
| r_1m | 1/99 | 0.0302 | 0.0225 | 0.0377 |
| r_1m | 5/95 | 0.0161 | 0.0101 | 0.0218 |
| r_1m | 25/75 | -0.0326 | -0.0392 | -0.0254 |
| z_ewma_60_raw | 0.1/99.9 | 0.0469 | 0.0306 | 0.0647 |
| z_ewma_60_raw | 1/99 | 0.0498 | 0.0439 | 0.0559 |
| z_ewma_60_raw | 5/95 | 0.0154 | 0.0124 | 0.0192 |
| z_ewma_60_raw | 25/75 | -0.0300 | -0.0359 | -0.0244 |

En los pares 1/99 y 5/95, sobre 100 combinaciones grupo × serie del bloque causal, 62 tienen un IC que excluye el cero (62 con la cola izquierda más larga). La mayor en valor absoluto: `z_ewma_60_raw`, YEAR = `2024`, par 1/99: 0.087 [0.075, 0.101]. Que un IC excluya el cero con casi dos millones de observaciones dice que la asimetría es distinguible del ruido, no que sea grande: la magnitud es la que figura en la tabla. Detalle por grupo en `TDA12_asimetria_colas.csv`. Coherente con TH25 (TDA-11): asimetría presente pero pequeña.

## 11. Robustez

**A. Estimador de volatilidad** (las 6 configuraciones causales de TDA-10; ratios puntuales del nivel 1 %–99 %, sin bootstrap):

| config | ratio_SEGMENT_std | ratio_SEGMENT_std_trimmed | ratio_VOL_DECILE_CAUSAL_std | ratio_VOL_DECILE_CAUSAL_std_trimmed | ratio_YEAR_std | ratio_YEAR_std_trimmed | point_verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|
| rolling_std_30_raw | 0.263 | 0.096 | 0.033 | 0.039 | 0.042 | 0.027 | SOLO_ESCALA |
| rolling_std_120_raw | 0.318 | 0.118 | 0.054 | 0.024 | 0.065 | 0.023 | NO_CONCLUYENTE |
| rolling_std_390_raw | 0.340 | 0.107 | 0.076 | 0.061 | 0.063 | 0.018 | NO_CONCLUYENTE |
| ewma_20_raw | 0.282 | 0.092 | 0.049 | 0.031 | 0.053 | 0.028 | SOLO_ESCALA |
| ewma_60_raw (PRIMARIO) | 0.288 | 0.104 | 0.058 | 0.033 | 0.060 | 0.015 | SOLO_ESCALA |
| ewma_240_raw | 0.315 | 0.103 | 0.120 | 0.055 | 0.074 | 0.023 | NO_CONCLUYENTE |

Comparación homogénea (puntual contra puntual, informativa): 0.50 de las configuraciones da la misma conclusión puntual que el primario. Conclusiones PUNTUALES (sin IC) de las 6 configuraciones: 3 dan `SÓLO ESCALA`, 3 dan `NO CONCLUYENTE`. La del primario es `SÓLO ESCALA`; su veredicto formal, que además exige que el IC completo esté en la misma zona, es `NO CONCLUYENTE`. Fracción de configuraciones cuya conclusión puntual coincide con ese veredicto formal: 0.50 (umbral de robustez de TDA-10: 0.75; sólo puede DEGRADAR un veredicto firme a `NO CONCLUYENTE`, nunca crear uno).

**B. Fronteras de los deciles.** La fila `VOL_DECILE_FULLSAMPLE` de §9 repite la comparación con las fronteras retrospectivas de TDA-10.

**C. Bloque `clock_adjusted` (RETROSPECTIVO, nunca decide)** — misma comparación sobre `z_ewma_60_clock_adjusted`:

| agrupación | normalización | disponibles | comparados | ratio | ratio_ci_lo | ratio_ci_hi | mín (grupo) | máx (grupo) | excluidos (soporte) | sin normalización válida | % obs. comparadas | etiqueta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SEGMENT | std | 7 | 7 | 0.118 | 0.085 | 0.160 | 2.261 (08:30-09:30) | 2.556 (16:02-20:00) | — | — | 100.0 | SUPERPUESTO |
| SEGMENT | std_trimmed | 7 | 7 | 0.078 | 0.070 | 0.086 | 2.521 (09:30-16:02) | 2.726 (16:02-20:00) | — | — | 100.0 | SUPERPUESTO |
| SEGMENT | none | 7 | 7 | 0.191 | 0.176 | 0.210 | 2.436 (09:30-16:02) | 2.921 (16:02-20:00) | — | — | 100.0 | SUPERPUESTO |
| VOL_DECILE_CAUSAL | std | 10 | 10 | 0.046 | 0.031 | 0.069 | 2.444 (3) | 2.559 (9) | — | — | 100.0 | SUPERPUESTO |
| VOL_DECILE_CAUSAL | std_trimmed | 10 | 10 | 0.032 | 0.023 | 0.044 | 2.566 (1) | 2.649 (9) | — | — | 100.0 | SUPERPUESTO |
| VOL_DECILE_CAUSAL | none | 10 | 10 | 0.042 | 0.034 | 0.059 | 2.515 (8) | 2.622 (0) | — | — | 100.0 | SUPERPUESTO |
| YEAR | std | 7 | 6 | 0.049 | 0.040 | 0.065 | 2.434 (2023) | 2.557 (2020) | 2019 | — | 99.6 | SUPERPUESTO |
| YEAR | std_trimmed | 7 | 6 | 0.041 | 0.034 | 0.047 | 2.539 (2023) | 2.644 (2020) | 2019 | — | 99.6 | SUPERPUESTO |
| YEAR | none | 7 | 6 | 0.042 | 0.034 | 0.051 | 2.502 (2023) | 2.610 (2020) | 2019 | — | 99.6 | SUPERPUESTO |

Con la misma regla, este bloque daría `SÓLO ESCALA` (SEGMENT: SUPERPUESTO, VOL_DECILE_CAUSAL: SUPERPUESTO, YEAR: SUPERPUESTO). Sirve para interpretar: indica cuánto de la diferencia entre segmentos del bloque causal se debe al patrón horario determinista que `sigma_hat` causal no conoce.

**D. Longitud del bloque del bootstrap** (descriptivo; el procedimiento aprobado es el bloque de 1 jornada). Se repite el bootstrap con bloques de (5, 21) jornadas CONSECUTIVAS, que conservan la dependencia entre días vecinos. Estadístico del veredicto sobre `z_ewma_60_raw`:

| grouping | normalization | bloque (jornadas) | point | IC95 | ancho IC ÷ ancho con 1 jornada | etiqueta con ese bloque |
|---:|---:|---:|---:|---:|---:|---:|
| SEGMENT | std | 1 | 0.288 | [0.2434, 0.3408] | 1.000 | NO_CONCLUYENTE |
| SEGMENT | std | 5 | 0.288 | [0.2387, 0.3466] | 1.107 | NO_CONCLUYENTE |
| SEGMENT | std | 21 | 0.288 | [0.2292, 0.3496] | 1.236 | NO_CONCLUYENTE |
| SEGMENT | std_trimmed | 1 | 0.104 | [0.0897, 0.1260] | 1.000 | SUPERPUESTO |
| SEGMENT | std_trimmed | 5 | 0.104 | [0.0909, 0.1293] | 1.056 | SUPERPUESTO |
| SEGMENT | std_trimmed | 21 | 0.104 | [0.0831, 0.1288] | 1.260 | SUPERPUESTO |
| VOL_DECILE_CAUSAL | std | 1 | 0.058 | [0.0402, 0.1062] | 1.000 | SUPERPUESTO |
| VOL_DECILE_CAUSAL | std | 5 | 0.058 | [0.0406, 0.1027] | 0.940 | SUPERPUESTO |
| VOL_DECILE_CAUSAL | std | 21 | 0.058 | [0.0416, 0.1109] | 1.049 | SUPERPUESTO |
| VOL_DECILE_CAUSAL | std_trimmed | 1 | 0.033 | [0.0256, 0.0446] | 1.000 | SUPERPUESTO |
| VOL_DECILE_CAUSAL | std_trimmed | 5 | 0.033 | [0.0258, 0.0455] | 1.036 | SUPERPUESTO |
| VOL_DECILE_CAUSAL | std_trimmed | 21 | 0.033 | [0.0256, 0.0491] | 1.239 | SUPERPUESTO |
| YEAR | std | 1 | 0.060 | [0.0435, 0.0948] | 1.000 | SUPERPUESTO |
| YEAR | std | 5 | 0.060 | [0.0425, 0.0943] | 1.010 | SUPERPUESTO |
| YEAR | std | 21 | 0.060 | [0.0417, 0.1023] | 1.180 | SUPERPUESTO |
| YEAR | std_trimmed | 1 | 0.015 | [0.0119, 0.0230] | 1.000 | SUPERPUESTO |
| YEAR | std_trimmed | 5 | 0.015 | [0.0123, 0.0245] | 1.096 | SUPERPUESTO |
| YEAR | std_trimmed | 21 | 0.015 | [0.0122, 0.0298] | 1.574 | SUPERPUESTO |

Grupos que marcan el mínimo y el máximo de la comparación por segmento (`z_ewma_60_raw`):

| group | statistic | normalization | bloque (jornadas) | point | IC95 | ancho IC ÷ ancho con 1 jornada |
|---:|---:|---:|---:|---:|---:|---:|
| 08:30-09:30 | std |  | 1 | 1.658 | [1.5560, 1.7885] | 1.000 |
| 08:30-09:30 | std |  | 5 | 1.658 | [1.5336, 1.7869] | 1.089 |
| 08:30-09:30 | std |  | 21 | 1.658 | [1.5338, 1.8043] | 1.163 |
| 08:30-09:30 | half_width_1_99 | std | 1 | 1.863 | [1.7352, 1.9808] | 1.000 |
| 08:30-09:30 | half_width_1_99 | std | 5 | 1.863 | [1.7211, 1.9975] | 1.126 |
| 08:30-09:30 | half_width_1_99 | std | 21 | 1.863 | [1.7280, 2.0146] | 1.167 |
| 08:30-09:30 | half_width_1_99 | std_trimmed | 1 | 2.531 | [2.4757, 2.5608] | 1.000 |
| 08:30-09:30 | half_width_1_99 | std_trimmed | 5 | 2.531 | [2.4724, 2.5677] | 1.119 |
| 08:30-09:30 | half_width_1_99 | std_trimmed | 21 | 2.531 | [2.4705, 2.5764] | 1.244 |
| 08:30-09:30 | q0.001 |  | 1 | -7.973 | [-9.3677, -7.3769] | 1.000 |
| 08:30-09:30 | q0.001 |  | 5 | -7.973 | [-9.2898, -7.2098] | 1.045 |
| 08:30-09:30 | q0.001 |  | 21 | -7.973 | [-9.3488, -7.2616] | 1.048 |
| 08:30-09:30 | q0.01 |  | 1 | -3.199 | [-3.2555, -3.1157] | 1.000 |
| 08:30-09:30 | q0.01 |  | 5 | -3.199 | [-3.2715, -3.1149] | 1.120 |
| 08:30-09:30 | q0.01 |  | 21 | -3.199 | [-3.2843, -3.1111] | 1.239 |
| 08:30-09:30 | q0.99 |  | 1 | 2.981 | [2.9097, 3.0381] | 1.000 |
| 08:30-09:30 | q0.99 |  | 5 | 2.981 | [2.9149, 3.0433] | 1.000 |
| 08:30-09:30 | q0.99 |  | 21 | 2.981 | [2.9204, 3.0357] | 0.899 |
| 08:30-09:30 | q0.999 |  | 1 | 9.948 | [8.7731, 11.3582] | 1.000 |
| 08:30-09:30 | q0.999 |  | 5 | 9.948 | [8.7106, 11.3672] | 1.028 |
| 08:30-09:30 | q0.999 |  | 21 | 9.948 | [8.7157, 11.2723] | 0.989 |
| 09:30-16:02 | std |  | 1 | 1.118 | [1.1124, 1.1234] | 1.000 |
| 09:30-16:02 | std |  | 5 | 1.118 | [1.1116, 1.1249] | 1.208 |
| 09:30-16:02 | std |  | 21 | 1.118 | [1.1114, 1.1247] | 1.199 |
| 09:30-16:02 | half_width_1_99 | std | 1 | 2.603 | [2.5886, 2.6154] | 1.000 |
| 09:30-16:02 | half_width_1_99 | std | 5 | 2.603 | [2.5883, 2.6157] | 1.021 |
| 09:30-16:02 | half_width_1_99 | std | 21 | 2.603 | [2.5802, 2.6202] | 1.493 |
| 09:30-16:02 | half_width_1_99 | std_trimmed | 1 | 2.696 | [2.6840, 2.7067] | 1.000 |
| 09:30-16:02 | half_width_1_99 | std_trimmed | 5 | 2.696 | [2.6828, 2.7078] | 1.102 |
| 09:30-16:02 | half_width_1_99 | std_trimmed | 21 | 2.696 | [2.6767, 2.7113] | 1.525 |
| 09:30-16:02 | q0.001 |  | 1 | -5.905 | [-6.0287, -5.7876] | 1.000 |
| 09:30-16:02 | q0.001 |  | 5 | -5.905 | [-6.0395, -5.7857] | 1.053 |
| 09:30-16:02 | q0.001 |  | 21 | -5.905 | [-6.0609, -5.7733] | 1.193 |
| 09:30-16:02 | q0.01 |  | 1 | -2.997 | [-3.0210, -2.9691] | 1.000 |
| 09:30-16:02 | q0.01 |  | 5 | -2.997 | [-3.0255, -2.9664] | 1.138 |
| 09:30-16:02 | q0.01 |  | 21 | -2.997 | [-3.0358, -2.9536] | 1.583 |
| 09:30-16:02 | q0.99 |  | 1 | 2.824 | [2.7985, 2.8507] | 1.000 |
| 09:30-16:02 | q0.99 |  | 5 | 2.824 | [2.7923, 2.8547] | 1.196 |
| 09:30-16:02 | q0.99 |  | 21 | 2.824 | [2.7916, 2.8589] | 1.288 |
| 09:30-16:02 | q0.999 |  | 1 | 5.615 | [5.4778, 5.7507] | 1.000 |
| 09:30-16:02 | q0.999 |  | 5 | 5.615 | [5.4751, 5.7707] | 1.083 |
| 09:30-16:02 | q0.999 |  | 21 | 5.615 | [5.4526, 5.7430] | 1.064 |

En 0 de 6 comparaciones del estadístico del veredicto la etiqueta cambia al alargar el bloque. El mayor ensanchamiento de un IC de `z_ewma_60_raw` es ×2.08. Tabla completa (incluye `r_1m`) en `TDA12_sensibilidad_longitud_bloque.csv`.

**E. Lecturas alternativas del estadístico y propuesta** (descriptivo: sólo la primera fila es la regla vigente). Muestra cuánto depende el resultado de las decisiones nuevas de esta etapa:

| lectura | estado | SEGMENT | VOL_DECILE_CAUSAL | YEAR | resultado (sin degradación por robustez) |
|---:|---:|---:|---:|---:|---:|
| regla vigente N4: ÷std y ÷std_trimmed deben coincidir; cola >= 200 | VIGENTE | NO_CONCLUYENTE | SUPERPUESTO | SUPERPUESTO | NO CONCLUYENTE |
| solo ÷std (lectura literal del roadmap de TDA-12) | alternativa | NO_CONCLUYENTE | SUPERPUESTO | SUPERPUESTO | NO CONCLUYENTE |
| solo ÷std_trimmed | alternativa | SUPERPUESTO | SUPERPUESTO | SUPERPUESTO | SÓLO ESCALA |
| sin dividir por la desviacion del grupo (estadistico LITERAL de TDA-10, al que pertenecen los umbrales) | alternativa | INTERMEDIO | SUPERPUESTO | SUPERPUESTO | NO CONCLUYENTE |
| regla N4 con el criterio de participacion LITERAL de TDA-10 (tamaño de grupo >= 200) | alternativa | NO_CONCLUYENTE | SUPERPUESTO | SUPERPUESTO | NO CONCLUYENTE |
| PROPUESTA: cada cola por separado (sensible a asimetria) | propuesta | NO_CONCLUYENTE | SUPERPUESTO | SUPERPUESTO | NO CONCLUYENTE |

De las 4 lecturas alternativas, 3 dan el mismo resultado que la regla vigente. La PROPUESTA (exigir la clasificación en cada cola por separado, para que la regla no sea ciega a la asimetría) daría `NO CONCLUYENTE`; no está aprobada y no interviene en el veredicto.

**Tratamiento de 2019.** La comparación por año usa el 99.6 % de las observaciones; queda fuera `2019` (7.406 barras, año parcial) porque sus colas del 1 % tienen menos de 200 observaciones. La etiqueta de la agrupación por año describe, por tanto, los años comparados, no 2019. Con el criterio literal de TDA-10 (tamaño de grupo ≥ 200), que sí lo incluye: ÷std: 0.060 [0.044, 0.097] → `SUPERPUESTO`; ÷std_trimmed: 0.015 [0.013, 0.054] → `SUPERPUESTO`.

## 12. Veredicto TH26

| agrupación | ÷std: ratio [IC95] | label_std | ÷std recortada: ratio [IC95] | label_std_trimmed | label |
|---:|---:|---:|---:|---:|---:|
| SEGMENT | 0.288 [0.243, 0.341] | NO_CONCLUYENTE | 0.104 [0.090, 0.126] | SUPERPUESTO | NO_CONCLUYENTE |
| VOL_DECILE_CAUSAL | 0.058 [0.040, 0.106] | SUPERPUESTO | 0.033 [0.026, 0.045] | SUPERPUESTO | SUPERPUESTO |
| YEAR | 0.060 [0.044, 0.095] | SUPERPUESTO | 0.015 [0.012, 0.023] | SUPERPUESTO | SUPERPUESTO |

**Veredicto: `NO CONCLUYENTE`** (sobre `z_ewma_60_raw`, regla de §3.4).

`SUPERPUESTO` = semi-amplitud normalizada 1 %–99 % estable entre grupos; no es una demostración de igualdad del perfil completo (§3). El veredicto está condicionado a las decisiones nuevas y exploratorias de §3 (umbrales aplicados al estadístico normalizado, soporte exigido a la cola, doble normalización, regla de robustez, `NO CONCLUYENTE` como salida).

**Dónde no hay estabilidad firme de la semi-amplitud:**

- **segmento horario (TDA-06)** (`NO_CONCLUYENTE`): dividiendo por `std` la semi-amplitud 1 %–99 % va de 1.863 (`08:30-09:30`) a 2.603 (`09:30-16:02`), ratio 0.288 [0.243, 0.341] → `NO_CONCLUYENTE`; dividiendo por `std_trimmed` la semi-amplitud 1 %–99 % va de 2.531 (`08:30-09:30`) a 2.808 (`16:02-20:00`), ratio 0.104 [0.090, 0.126] → `SUPERPUESTO`. Desviación de `z` por grupo: de 0.663 (`16:02-20:00`) a 1.658 (`08:30-09:30`).

**Por qué no se fuerza una conclusión.** El roadmap prevé dos resultados (`SÓLO ESCALA` / `ESCALA Y FORMA`). La regla fijada de antemano exige evidencia firme para cualquiera de los dos; aquí al menos una agrupación no la da (su IC cruza un umbral, o las dos normalizaciones discrepan, o queda en la zona intermedia). No se interpreta la falta de divergencia como demostración de igualdad, ni al revés.

## 13. Consecuencias para `STOP-13` (sin ejecutar TDA-13)

TDA-12 = NO_CONCLUYENTE -> TDA-12 NO activa STOP-13 (no puede afirmar SOLO_ESCALA) y tampoco declara cumplido el criterio de entrada 1 de EVT (no puede afirmar ESCALA_Y_FORMA). El roadmap no preve este tercer resultado: como EVT exige que TDA-12 haya concluido ESCALA_Y_FORMA, leido literalmente la rama EVT queda CERRADA mientras ese criterio no se cumpla; es una decision que requiere confirmacion explicita antes de TDA-13 parte B. TDA-13 parte A es obligatoria en cualquier caso.

- `STOP-13` activado por TDA-12: **NO**.
- Criterio de entrada 1 de EVT (TDA-12 = `ESCALA Y FORMA`) cumplido: **NO**.

TDA-13 parte A (inventario, triage y agrupamiento de extremos) es obligatoria con cualquier resultado. Esta etapa no la inicia.

## 14. Limitaciones

1. Los IC bootstrap remuestrean jornadas como si fueran intercambiables; con volatilidad persistente entre días, la incertidumbre real de los cuantiles extremos es probablemente mayor (§11.D lo cuantifica con bloques más largos).
2. Los cuantiles 0,1 y 99,9 por grupo descansan en pocas decenas o centenas de observaciones concentradas en pocas jornadas (§8): son órdenes de magnitud. Ningún cuantil extrapola más allá de lo observado.
3. El veredicto usa el nivel 1 %–99 % (el único con umbral documentado, de TDA-10). Diferencias de forma que sólo aparezcan más allá del percentil 1/99 —la cola profunda— no las decide esta regla; se reportan en §9 sin umbral y son materia de TDA-13 parte A.
4. `SUPERPUESTO` describe la estabilidad de la semi-amplitud normalizada, no la igualdad del perfil distribucional completo. El estadístico no ve la asimetría: dos grupos con la misma amplitud y asimetrías opuestas serían `SUPERPUESTO`. La comparación de cada cola por separado (§9) lo suple sólo de forma descriptiva y no forma parte del veredicto formal.
5. Los umbrales 0.3 / 0.6 se aprobaron en TDA-10 para el estadístico SIN normalizar por la desviación del grupo. Aplicarlos al estadístico normalizado de TDA-12 es una decisión nueva y exploratoria; no existe una justificación teórica de esos valores para este uso.
6. Exigir 200 observaciones en la COLA (en TDA-10 eran 200 observaciones totales por grupo) es una decisión nueva y exploratoria. Deja fuera de la comparación por año al año parcial 2019 (§11.E).
7. La doble normalización y la regla de robustez son decisiones nuevas y exploratorias. El resultado depende de la normalización (§11.E: con sólo la desviación recortada la regla daría otro resultado), y la regla de robustez compara conclusiones puntuales con conclusiones basadas en IC, que no son equivalentes (§11.A).
8. `NO CONCLUYENTE` no es uno de los dos resultados que prevé el roadmap: es una salida prudente, nueva y exploratoria, para cuando no puede sostenerse ninguno. No equivale a "la forma es igual" ni a "la forma cambia".
9. El decil causal de volatilidad es también una decisión nueva de esta etapa (§3); los grupos resultantes no tienen el mismo tamaño y al principio de la muestra descansan en poca historia.
10. Las pruebas de causalidad son comprobaciones por muestreo (§3): se ejecutaron sobre el estimador primario, en un número reducido de barras elegidas con semilla fija. No se re-verificaron en esta etapa las demás configuraciones de `sigma_hat` de §11.A (su verificación es la de TDA-10) ni sus deciles causales.
11. La sensibilidad a la longitud de bloque (§11.D) usa dos longitudes elegidas en esta etapa; bloques más largos dejan pocos bloques independientes y sus IC son a su vez inciertos.
12. El estadístico de rango `(máx − mín)/mediana` crece con el número de grupos y está sesgado al alza por ruido de muestreo; no es comparable entre agrupaciones con distinto número de grupos.
13. Los cortes de los segmentos horarios provienen de TDA-06 (toda la muestra).
14. `z_t` depende del estimador de volatilidad; §11.A muestra cuánto cambia la conclusión con las otras configuraciones de TDA-10.
15. No se ajustó regresión cuantílica: los tres condicionantes del roadmap son categóricos o se discretizan en deciles, y no surgió ninguna pregunta que exigiera una variable continua (G4).

## 15. Tiempos

| Etapa | Tiempo |
|---|---:|
| Poblaciones (r_1m/r_tilde) e invariantes | 2.9s |
| sigma_hat causal y z_t (definiciones de TDA-10) + verificacion de ausencia de look-ahead | 0.6s |
| Estados de volatilidad: decil CAUSAL (jornadas anteriores) y decil de muestra completa (TDA-10) | 159.8s |
| Cuantiles incondicionales/condicionales + bootstrap de bloques por jornada (n_boot=300) para 4 series + sensibilidad a bloques de (5, 21) jornadas | 330.1s |
| Sensibilidad de la comparacion de forma a las 6 configuraciones causales de sigma_hat (TDA-10) | 53.8s |
| Veredicto TH26 (escala vs forma) e implicacion para STOP-13 | 0.0s |

**Tiempo total (análisis + escritura): 548.6s (~9.1 min).** CPU, un proceso; sin GPU.

## 16. Archivos generados

`TDA12_cuantiles_condicionales.csv` (todos los cuantiles: 4 series × 5 agrupaciones, con IC, normalizados, `n`, soporte), `TDA12_escala_vs_forma_perfiles.csv`, `TDA12_asimetria_colas.csv` (asimetría y semi-amplitud por grupo), `TDA12_sensibilidad_estimador.csv`, `TDA12_verificacion_causalidad.csv`, `TDA12_tamano_deciles_volatilidad.csv`, `TDA12_sensibilidad_longitud_bloque.csv`, `TDA12_lecturas_alternativas_veredicto.csv` (8 CSV) + `TDA12_perfiles_cuantiles_normalizados.png`, `TDA12_semiamplitud_normalizada_por_grupo.png` (2 PNG) + este informe.

## 17. Modo sencillo

**¿Qué preguntó esta etapa?** Los movimientos de MNQ son más grandes a unas horas que a otras, en días agitados que en días tranquilos, y en unos años que en otros. La pregunta es si eso es sólo "el mismo dibujo con otro tamaño" (escala) o si el dibujo mismo cambia (forma): por ejemplo, que en ciertos contextos los movimientos extremos sean proporcionalmente más frecuentes.

**¿Cómo se midió?** Para cada contexto se calcularon los percentiles de los movimientos y se dividieron por el tamaño típico de ese contexto. Si sólo cambia el tamaño, todos los contextos quedan con el mismo perfil.

**¿Qué salió?** No se puede afirmar ninguna de las dos cosas con las reglas fijadas de antemano: por segmento horario (TDA-06) el resultado es `NO_CONCLUYENTE`; por decil CAUSAL de volatilidad previa el resultado es `SUPERPUESTO`; por año el resultado es `SUPERPUESTO`.

**¿Qué quiere decir `SUPERPUESTO` en las tablas?** Que, una vez descontado el tamaño típico, la anchura entre los percentiles 1 y 99 es prácticamente la misma en todos los grupos comparados. Es una medida concreta de anchura; no prueba que las distribuciones sean idénticas en todos sus detalles.

**¿Qué partes del método son nuevas?** La regla que decide, los umbrales aplicados a esta medida, el mínimo de observaciones exigido en la cola, el uso de dos normalizaciones, la comprobación de robustez y la propia salida `NO CONCLUYENTE` se fijaron en esta etapa y son exploratorias (§3, §14).

**¿Qué tan fiables son los números extremos?** Los percentiles 1 y 99 están bien sostenidos. Los percentiles 0,1 y 99,9 de un contexto concreto se apoyan en pocas observaciones de pocos días: sirven como orden de magnitud, no como cifra exacta.

**¿Qué NO significa?** No es una señal ni una regla de operación, no predice nada y no decide ningún aspecto del diseño de IRIS. Tampoco ejecuta el análisis de eventos extremos (TDA-13): sólo deja escrito en qué estado queda su condición de entrada (§13).

## 18. Estado de cierre

**TDA-12 = `PASS_WITH_OPEN_QUESTIONS` / `CLOSED`** — cerrada tras la auditoría correctiva y el cierre documental, con cuestiones metodológicas abiertas y documentadas (§3 y §14).

| Elemento | Estado al cierre |
|---|---|
| TH26 | `NO CONCLUYENTE` |
| `STOP-13` | no activado por TDA-12 |
| EVT (TDA-13 parte B) | criterio de entrada 1 no cumplido; no autorizada |
| TDA-13 parte A | siguiente tarea obligatoria; no iniciada en esta etapa |

**Cuestiones abiertas que el cierre NO resuelve.** Las decisiones nuevas y exploratorias de §3 (umbrales de TDA-10 aplicados al estadístico normalizado, soporte exigido a la cola, doble normalización, regla de robustez, `NO CONCLUYENTE` como salida) no quedan validadas por cerrar la etapa: el estado de TH26 se lee condicionado a ellas y con las limitaciones de §14. La comparación de cada cola por separado sigue siendo descriptiva y no forma parte del veredicto. Nada de lo anterior es una conclusión estadísticamente demostrada sobre la igualdad o la diferencia de forma.
