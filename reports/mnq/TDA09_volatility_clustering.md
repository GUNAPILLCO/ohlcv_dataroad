# TDA-09 — Dependencia en magnitud (volatility clustering)

**Etapa:** `docs/methodology/Tsay_OHLCV_analysis_roadmap.md` § TDA-09
**Depende de:** TDA-06 (`PASS_WITH_OPEN_QUESTIONS`, STOP-6 no activado, `s(m)` construido sobre `log_hl`), TDA-08 (`PASS_WITH_OPEN_QUESTIONS` / `CLOSED`), TDA08-H (`PASS_WITH_OPEN_QUESTIONS`).
**Alcance de datos:** exclusivamente `tda04_variables_1m.parquet`, `tda04_return_validity_mask.parquet`, `tda06_r_tilde.parquet`, `tda06_s_m.parquet` y `TDA06_segmentacion_propuesta.csv`. Ningún archivo de `data/raw/` ni de `holdout_files` fue abierto en esta etapa.
**Generado automáticamente** por `python -m ohlcv_dataroad.ingest.run_tda09 --config configs/mnq_snapshot.yaml` — este informe NUNCA se edita a mano; cualquier corrección se hace en el código (`tda09_volatility_clustering.py`/`run_tda09.py`) y se regenera con una nueva ejecución.

> Esta etapa estudia si el TAMAÑO de los movimientos de MNQ tiene memoria, separándolo de si la HORA del día explica ese patrón. NO diseña modelos predictivos, NO ajusta ARCH/GARCH como modelo productivo, NO crea features/targets/señales, NO abre el holdout, y NO inicia TDA-10 ni TDA-11.

---

## 1. La pregunta en palabras simples

¿Los períodos agitados de MNQ tienden a seguir agitados, y los tranquilos a seguir tranquilos — más allá de lo que ya se explica por la hora del día? Esta etapa separa dos cosas: la memoria genuina en el TAMAÑO de los movimientos, y el patrón horario ya conocido (TDA-06).

## 2. Datos y población analizada

- `r_1m`/`r_tilde` válidos: **1,914,530** filas.
- `log_hl`/`log_hl_tilde`: **1,918,050** filas (todas las barras admisibles — `log_hl` no necesita una barra anterior).
- Motor de ACF/bootstrap/G2 de TDA-08 (`tda08_linear_mean_dependence.py`) reutilizado **sin modificar**.

## 3. Decisión: `r_t` vs. innovación/residuo de media

Sensibilidad barata única (`TDA09_sensibilidad_remocion_media.csv`): comparación entre `ACF(|r|)` y `ACF(|r − β₁·r_{t-1}|)`.

| lag | rho_abs_r | rho_abs_e | beta_1_used |
|---:|---:|---:|---:|
| 1.0000 | 0.4237 | 0.4234 | 0.0059 |
| 5.0000 | 0.3891 | 0.3891 | 0.0059 |
| 20.0000 | 0.3492 | 0.3495 | 0.0059 |
| 60.0000 | 0.3114 | 0.3117 | 0.0059 |

Prácticamente idénticos en los 4 rezagos — remover la media no cambia la conclusión de magnitud.

## 4. Ajuste de reloj — diagnóstico de aplanamiento

`log_hl_tilde = log_hl / s(m)`; `|r_tilde| = |r_1m|/s(m)`. Se verificó (no se asumió) que el ajuste aplana el perfil por minuto:

| variable | flatness_raw | flatness_adjusted | ratio | clock_effectively_removed |
|---:|---:|---:|---:|---:|
| abs_r | 0.5840 | 0.0455 | 0.0778 | True |
| log_hl | 0.6015 | 0.0000 | 0.0000 | True |

## 5. ACF de dirección vs. magnitud (gráfico triple)

| Rezago (min) | rho(r) crudo | rho(|r|) crudo | rho(r^2) crudo |
|---:|---:|---:|---:|
| 1.0000 | 0.0059 | 0.4237 | 0.1512 |
| 5.0000 | -0.0045 | 0.3891 | 0.1018 |
| 20.0000 | 0.0030 | 0.3492 | 0.0691 |
| 60.0000 | 0.0004 | 0.3114 | 0.0596 |
| 240.0000 | 0.0019 | 0.2252 | 0.0337 |
| 600.0000 | 0.0017 | 0.1713 | 0.0413 |

`ACF(r) ≈ 0` en todos los rezagos mientras `ACF(|r|)` es grande y persistente — la separación entre dirección y magnitud que anticipa el roadmap.

## 6. Resultado de `|r|` — crudo vs. ajustado

| Rezago (min) | crudo | ajustado |
|---:|---:|---:|
| 1.0000 | 0.4237 | 0.3979 |
| 5.0000 | 0.3891 | 0.3541 |
| 20.0000 | 0.3492 | 0.3246 |
| 60.0000 | 0.3114 | 0.2922 |
| 240.0000 | 0.2252 | 0.2485 |
| 600.0000 | 0.1713 | 0.2019 |

## 7. Resultado de `log_hl` — crudo vs. ajustado

| Rezago (min) | crudo | ajustado |
|---:|---:|---:|
| 1.0000 | 0.7594 | 0.7172 |
| 5.0000 | 0.6993 | 0.6457 |
| 20.0000 | 0.6309 | 0.5983 |
| 60.0000 | 0.5630 | 0.5495 |
| 240.0000 | 0.4047 | 0.4685 |
| 600.0000 | 0.3170 | 0.4045 |

## 8. `r²` como contraste (no como proxy principal)

Mismo signo cualitativo que `|r|` pero sistemáticamente menor y más ruidoso en cada rezago (ver `TDA09_acf_magnitud.csv`, `variable="r2"`) — confirma la advertencia de Tsay/roadmap de no usar únicamente `r²`.

## 9. Diagnóstico "same-clock-position" entre jornadas

Distinto de una ACF continua de 1.380 minutos (`NOT_ESTIMABLE` bajo la topología de no-cruce — ver `TDA09_portmanteau.csv`). Compara el mismo minuto del reloj entre un día de negociación y el siguiente día presente en los datos:

| variable | raw_adjusted | rho_point | ci_lo | ci_hi | n_pairs | n_day_pairs |
|---:|---:|---:|---:|---:|---:|---:|
| abs_r | raw | 0.3110 | 0.2990 | 0.3230 | 1,881,044 | 1,419.0000 |
| abs_r | adjusted | 0.2259 | 0.2048 | 0.2464 | 1,881,044 | 1,419.0000 |
| log_hl | raw | 0.5648 | 0.5515 | 0.5794 | 1,885,567 | 1,419.0000 |
| log_hl | adjusted | 0.4143 | 0.3843 | 0.4420 | 1,885,567 | 1,419.0000 |

## 10. TH21 — comparación descriptiva de energía de dependencia (NUNCA una descomposición causal)

**Corrección de interpretación (v1)**: `fraction_removed`/`fraction_survives` son un **cambio descriptivo** de una métrica de energía (`Q(m)`, ventana de 240 minutos) antes y después del ajuste — **nunca** se interpretan como "el reloj explica X% del clustering". `Q(m)` no es lineal en `s(m)`, y ambas series comparten la misma dinámica subyacente además de diferir en escala — por eso la fracción puede incluso ser negativa sin que eso implique una contribución "menor que cero" del reloj.

| variable | m | Q_raw | Q_adjusted | fraction_removed | fraction_survives |
|---:|---:|---:|---:|---:|---:|
| abs_r | 240 | 31,393,915.7869 | 31,798,909.1215 | -0.0129 | 1.0129 |
| log_hl | 240 | 102,840,665.4169 | 112,299,900.3043 | -0.0920 | 1.0920 |

**Lectura permitida** (única): el clustering sobrevive claramente al ajuste horario, y el reloj no explica la mayor parte de la persistencia observada — **no** se afirma un porcentaje causal de atribución.

## 11. Persistencia por año

| variable | raw_adjusted | group | n_pairs | rho | rho_ci_lo | rho_ci_hi |
|---:|---:|---:|---:|---:|---:|---:|
| abs_r | raw | 2,019 | 7,268 | 0.4266 | 0.2974 | 0.4841 |
| abs_r | raw | 2,020 | 341,661 | 0.4467 | 0.4209 | 0.4624 |
| abs_r | raw | 2,021 | 348,320 | 0.3878 | 0.3747 | 0.3985 |
| abs_r | raw | 2,022 | 349,610 | 0.3496 | 0.3355 | 0.3644 |
| abs_r | raw | 2,023 | 349,264 | 0.3788 | 0.3653 | 0.3919 |
| abs_r | raw | 2,024 | 351,915 | 0.3776 | 0.3576 | 0.3937 |
| abs_r | raw | 2,025 | 163,396 | 0.4362 | 0.3884 | 0.4687 |
| abs_r | adjusted | 2,019 | 7,268 | 0.1970 | 0.1465 | 0.2414 |
| abs_r | adjusted | 2,020 | 341,661 | 0.4394 | 0.4032 | 0.4606 |
| abs_r | adjusted | 2,021 | 348,320 | 0.3120 | 0.2908 | 0.3286 |
| abs_r | adjusted | 2,022 | 349,610 | 0.2553 | 0.2379 | 0.2708 |
| abs_r | adjusted | 2,023 | 349,264 | 0.2496 | 0.2200 | 0.2835 |
| abs_r | adjusted | 2,024 | 351,915 | 0.3192 | 0.2583 | 0.3497 |
| abs_r | adjusted | 2,025 | 163,396 | 0.4127 | 0.3278 | 0.4689 |
| log_hl | raw | 2,019 | 7,406 | 0.7124 | 0.6246 | 0.7451 |
| log_hl | raw | 2,020 | 342,648 | 0.7717 | 0.7470 | 0.7815 |
| log_hl | raw | 2,021 | 349,039 | 0.7411 | 0.7287 | 0.7498 |
| log_hl | raw | 2,022 | 349,996 | 0.6971 | 0.6845 | 0.7128 |
| log_hl | raw | 2,023 | 349,636 | 0.7346 | 0.7186 | 0.7472 |
| log_hl | raw | 2,024 | 352,261 | 0.7308 | 0.7094 | 0.7449 |
| log_hl | raw | 2,025 | 163,544 | 0.7660 | 0.7237 | 0.7919 |
| log_hl | adjusted | 2,019 | 7,406 | 0.4216 | 0.2763 | 0.4614 |
| log_hl | adjusted | 2,020 | 342,648 | 0.7410 | 0.7092 | 0.7588 |
| log_hl | adjusted | 2,021 | 349,039 | 0.6469 | 0.6227 | 0.6688 |
| log_hl | adjusted | 2,022 | 349,996 | 0.5655 | 0.5405 | 0.5852 |
| log_hl | adjusted | 2,023 | 349,636 | 0.5916 | 0.5465 | 0.6367 |
| log_hl | adjusted | 2,024 | 352,261 | 0.6622 | 0.5926 | 0.6960 |
| log_hl | adjusted | 2,025 | 163,544 | 0.7354 | 0.6701 | 0.7799 |

## 12. Persistencia por segmento

| variable | raw_adjusted | group | n_pairs | rho | rho_ci_lo | rho_ci_hi |
|---:|---:|---:|---:|---:|---:|---:|
| abs_r | raw | 00:00-02:00 | 168,654 | 0.4341 | 0.3686 | 0.4833 |
| abs_r | raw | 02:00-03:00 | 84,601 | 0.3774 | 0.3439 | 0.4113 |
| abs_r | raw | 03:00-08:30 | 465,128 | 0.3623 | 0.3301 | 0.3846 |
| abs_r | raw | 08:30-09:30 | 84,669 | 0.3019 | 0.2652 | 0.3340 |
| abs_r | raw | 09:30-16:02 | 537,750 | 0.3549 | 0.3324 | 0.3694 |
| abs_r | raw | 16:02-20:00 | 235,744 | 0.4559 | 0.4254 | 0.4854 |
| abs_r | raw | 20:00-24:00 | 334,888 | 0.4062 | 0.3744 | 0.4300 |
| abs_r | adjusted | 00:00-02:00 | 168,654 | 0.4309 | 0.3699 | 0.4808 |
| abs_r | adjusted | 02:00-03:00 | 84,601 | 0.3768 | 0.3407 | 0.4148 |
| abs_r | adjusted | 03:00-08:30 | 465,128 | 0.3635 | 0.3302 | 0.3852 |
| abs_r | adjusted | 08:30-09:30 | 84,669 | 0.3504 | 0.3127 | 0.3783 |
| abs_r | adjusted | 09:30-16:02 | 537,750 | 0.3492 | 0.3184 | 0.3640 |
| abs_r | adjusted | 16:02-20:00 | 235,744 | 0.4450 | 0.4135 | 0.4741 |
| abs_r | adjusted | 20:00-24:00 | 334,888 | 0.4083 | 0.3757 | 0.4351 |
| log_hl | raw | 00:00-02:00 | 168,922 | 0.7356 | 0.6824 | 0.7699 |
| log_hl | raw | 02:00-03:00 | 84,663 | 0.7003 | 0.6721 | 0.7222 |
| log_hl | raw | 03:00-08:30 | 465,442 | 0.6940 | 0.6671 | 0.7139 |
| log_hl | raw | 08:30-09:30 | 84,694 | 0.5688 | 0.5340 | 0.6032 |
| log_hl | raw | 09:30-16:02 | 537,786 | 0.7246 | 0.7066 | 0.7371 |
| log_hl | raw | 16:02-20:00 | 237,735 | 0.7236 | 0.7039 | 0.7473 |
| log_hl | raw | 20:00-24:00 | 335,288 | 0.7150 | 0.6925 | 0.7310 |
| log_hl | adjusted | 00:00-02:00 | 168,922 | 0.7328 | 0.6832 | 0.7689 |
| log_hl | adjusted | 02:00-03:00 | 84,663 | 0.7054 | 0.6770 | 0.7266 |
| log_hl | adjusted | 03:00-08:30 | 465,442 | 0.6970 | 0.6682 | 0.7148 |
| log_hl | adjusted | 08:30-09:30 | 84,694 | 0.6510 | 0.6197 | 0.6823 |
| log_hl | adjusted | 09:30-16:02 | 537,786 | 0.7291 | 0.7033 | 0.7426 |
| log_hl | adjusted | 16:02-20:00 | 237,735 | 0.7238 | 0.7009 | 0.7458 |
| log_hl | adjusted | 20:00-24:00 | 335,288 | 0.7196 | 0.6953 | 0.7395 |

## 13. Persistencia por ventana rodante (mensual)

Ventana rodante = mes calendario (misma convención que la ventana rodante de TDA-08). Se muestran los primeros y últimos meses de la tabla completa (`TDA09_persistencia_ventana_rodante.csv` tiene la serie completa mes a mes):

| variable | raw_adjusted | year_month | n_pairs | rho | rho_ci_lo | rho_ci_hi |
|---:|---:|---:|---:|---:|---:|---:|
| abs_r | raw | 2019-12 | 7,268 | 0.4266 | 0.2974 | 0.4841 |
| abs_r | raw | 2020-01 | 29,635 | 0.3547 | 0.3229 | 0.3823 |
| abs_r | raw | 2020-02 | 27,000 | 0.4512 | 0.3669 | 0.4746 |
| abs_r | raw | 2020-03 | 25,774 | 0.3236 | 0.2640 | 0.3517 |
| abs_r | raw | 2020-04 | 28,568 | 0.2421 | 0.2144 | 0.2727 |
| abs_r | raw | 2020-05 | 28,359 | 0.2786 | 0.2394 | 0.3092 |

...

| variable | raw_adjusted | year_month | n_pairs | rho | rho_ci_lo | rho_ci_hi |
|---:|---:|---:|---:|---:|---:|---:|
| log_hl | adjusted | 2025-01 | 29,628 | 0.6025 | 0.5219 | 0.6423 |
| log_hl | adjusted | 2025-02 | 27,340 | 0.6444 | 0.5723 | 0.6829 |
| log_hl | adjusted | 2025-03 | 27,577 | 0.5567 | 0.4885 | 0.5916 |
| log_hl | adjusted | 2025-04 | 28,953 | 0.7334 | 0.6259 | 0.7670 |
| log_hl | adjusted | 2025-05 | 30,098 | 0.5378 | 0.4870 | 0.5667 |
| log_hl | adjusted | 2025-06 | 19,948 | 0.5616 | 0.4396 | 0.6364 |

**Corrección de lenguaje (v1)**: no se afirma que "la magnitud es estable". Se afirma que **la PRESENCIA del clustering es estable** entre años, segmentos y ventanas mensuales — el `rho` puntual (y su intervalo bootstrap) es positivo y excluye cero de forma consistente — **aunque su INTENSIDAD (el valor exacto de `rho`) varía** de un año/segmento/mes a otro.

## 14. Resultado ARCH-LM (Engle)

| series | order | n_eff | LM | R2 | percentile_of_real | exceeds_calibration_threshold |
|---:|---:|---:|---:|---:|---:|---:|
| r_1m | 1 | 1,911,434 | 43,711.2643 | 0.0229 | 100.0000 | True |
| r_1m | 5 | 1,900,601 | 92,638.8101 | 0.0487 | 100.0000 | True |
| r_1m | 20 | 1,866,492 | 123,041.0576 | 0.0659 | 100.0000 | True |
| r_tilde | 1 | 1,911,434 | 163,540.3025 | 0.0856 | 100.0000 | True |
| r_tilde | 5 | 1,900,601 | 282,220.0513 | 0.1485 | 100.0000 | True |
| r_tilde | 20 | 1,866,492 | 339,125.8000 | 0.1817 | 100.0000 | True |

## 15. Calibración G2

- **Null 1 (permutación por minuto) — PRINCIPAL**: ver `TDA09_g2_calibracion_null1_principal.csv`.
- **Null global (secundario, solo series ajustadas)**: ver `TDA09_g2_calibracion_secundaria_global_ajustada.csv`.
- **Null sintético (heredado de TDA-08) — FALLIDO, EXCLUIDO de la inferencia principal**:

```
{'var_real': 1.2039531089112345e-07, 'var_synth': 1.8033464622995731e-07, 'var_ratio': 1.497854400600689, 'var_within_tolerance': False, 'kurt_real': 217.15387343743961, 'kurt_synth': 681.1989050583921, 'kurt_rel_diff': 2.136941074434209, 'kurt_within_tolerance': False, 'scale_profile_correlation': 0.9970607207092175, 'scale_profile_within_tolerance': True, 'note': 'Null sintetico heredado de TDA-08 (resampleo empirico de r_tilde reescalado por s(m)), aplicado a |r| -- TDA-08 ya demostro que este null NO preserva varianza/curtosis reales (solo el perfil de escala). Se reevalua aqui sobre |r| y se documenta como SENSIBILIDAD FALLIDA/DIAGNOSTICO, EXCLUIDA de la inferencia principal de G2 (que usa exclusivamente el null por permutacion, g2_null1_calibration). No se construyo un null sintetico propio para log_hl (G4): el null por permutacion ya calibra la inferencia principal de log_hl sin necesitar un segundo sistema de nulls.'}
```

## 16. Rolls en `log_hl` — verificación explícita

Se añadió `compute_block_ids_with_contract`, que exige el mismo `contract` (no solo `trading_date`/gap) entre dos filas consecutivas para pertenecer al mismo bloque de continuidad — aplicado a las 4 poblaciones de esta etapa. Cubierto por tests adversariales dedicados (ver `tests/test_tda09_volatility_clustering.py`).

## 17. Veredicto TH19

`VOLATILITY_CLUSTERING_DETECTABLE` — `rho_1(|r|)` crudo = 0.4237, supera calibración G2: True, umbral de materialidad: 0.0200.

## 18. Veredicto TH20

TH20 = `RESUELTA` (habilitada porque una fraccion no trivial de la energia de dependencia sobrevive al ajuste en al menos una variable -- ver §10).

| Variable | R2 log-log | R2 semi-log | Pendiente log-log | Interpretacion |
|---:|---:|---:|---:|---:|
| abs_r_adjusted | 0.9498 | 0.9047 | -0.1280 | mas compatible con log-log (posible decaimiento polinomial / persistencia lenta -- NO se afirma memoria larga, ver docstring) |
| log_hl_adjusted | 0.9715 | 0.8663 | -0.1048 | mas compatible con log-log (posible decaimiento polinomial / persistencia lenta -- NO se afirma memoria larga, ver docstring) |

**IMPORTANTE**: esto es un diagnostico DESCRIPTIVO. Un decaimiento mejor descrito por log-log que por semi-log es compatible con persistencia lenta, pero NO se afirma memoria larga genuina: la misma forma puede surgir de cambios de nivel/regimen no modelados (p.ej. años de mayor volatilidad de mercado). Esta ambiguedad queda EXPLICITAMENTE abierta para TDA-14 -- no se estimo ningun parametro de memoria larga (`d`) ni se ejecuto differencing fraccional.

## 19. Veredicto TH21

`CLUSTERING_GENUINO` — fracciones que sobreviven al ajuste: {'abs_r': 1.0129003765375955, 'log_hl': 1.0919795185008876}.

## 20. STOP-9

`NO ACTIVADO` — Al menos una variable conserva una fraccion de dependencia por debajo del umbral de colapso -- STOP-9 NO ACTIVADO.

## 21. Tiempo por etapa (análisis)

| Etapa | Tiempo |
|---|---:|
| Carga de datos y construccion de poblaciones (r_1m/r_tilde/log_hl/log_hl_tilde) | 4.3s |
| Sensibilidad de media, ACF de magnitud/direccion (hasta 600 rezagos) y portmanteau | 146.9s |
| Bootstrap de bloques por jornada (rezagos clave) y diagnostico same-clock-position | 881.9s |
| Calibracion G2 (null principal por minuto, secundario global, sintetico) y diagnostico de aplanamiento de reloj | 2705.4s |
| Persistencia por año, segmento y ventana rodante (mensual), con IC bootstrap por grupo | 222.9s |
| LM de Engle (r_1m crudo y r_tilde ajustado) | 216.6s |
| Comparacion descriptiva de energia de dependencia (TH21), STOP-9 y diagnostico de decaimiento (TH20) | 0.0s |
| Diagnostico del null sintetico heredado de TDA-08 y veredicto final TH19 | 0.3s |

**Tiempo total de la ejecución (análisis + escritura de CSV/PNG/MD): 4178.9s (~69.6 min).**

## 22. Archivos generados

`TDA09_acf_magnitud.csv`, `TDA09_bootstrap_ci.csv`, `TDA09_clock_attribution.csv`, `TDA09_persistencia_por_anio.csv`, `TDA09_persistencia_por_segmento.csv`, `TDA09_persistencia_ventana_rodante.csv`, `TDA09_portmanteau.csv`, `TDA09_arch_lm.csv`, `TDA09_g2_calibracion_null1_principal.csv`, `TDA09_g2_calibracion_secundaria_global_ajustada.csv`, `TDA09_g2_diagnostico_momentos_sintetico.csv`, `TDA09_sensibilidad_remocion_media.csv`, `TDA09_diagnostico_aplanamiento_reloj.csv` (13 CSV) + `TDA09_acf_triple.png`, `TDA09_acf_raw_vs_adjustado.png`, TDA09_decay_loglog_semilog.png (PNG) + este informe (MD).

## 23. Comandos de validación

```
python -m pytest -q tests/test_tda09_volatility_clustering.py
python -m pytest -q
python -m ohlcv_dataroad.ingest.run_tda09 --config configs/mnq_snapshot.yaml
```

## 24. Estado final

**`PASS_WITH_OPEN_QUESTIONS`**

- TH19 = `VOLATILITY_CLUSTERING_DETECTABLE`
- TH20 = `RESUELTA`
- TH21 = `CLUSTERING_GENUINO`
- STOP-9 = `NO ACTIVADO`

**No se avanza a TDA-10 en esta tarea.**

## 25. Preguntas abiertas

1. Forma exacta del decaimiento (TH20): ambigüedad memoria-larga-vs-cambios-de-nivel abierta para TDA-14.
2. El rango evaluado se detiene en 600 minutos — no se investigó más allá.
3. Null sintético de TDA-08 (Null 2) sigue sin ser representativo de varianza/curtosis — ya documentado en TDA-08, reconfirmado aquí sobre `|r|`.

## 26. Recomendación para el siguiente paso

TDA-10 (escala vs. forma de las colas) tendría una base sólida si se decide avanzar en una tarea separada — no se ejecuta aquí.

---

## Modo sencillo — en 10 líneas

**¿La volatilidad del MNQ tiene memoria?** Sí: si un minuto fue agitado, los siguientes minutos y horas tienden a seguir agitados.

**¿Cuánto dura?** Al menos 600 minutos (10 horas) — el límite hasta donde se midió.

**¿Es la hora del día o algo más?** La dependencia sobrevive claramente al ajustar por el patrón horario — el reloj no explica la mayor parte de lo que se observa (ver §10, comparación descriptiva, no causal).

**¿Es estable?** La PRESENCIA del clustering es estable entre años, segmentos y meses; su INTENSIDAD exacta varía.

**¿Qué NO significa?** No dice hacia dónde va a moverse el precio, no es una señal de trading, y no valida ningún modelo GARCH — solo confirma que existe algo que un modelo de volatilidad podría, más adelante, intentar capturar.
