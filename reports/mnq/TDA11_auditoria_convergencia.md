# TDA-11 — Auditoría correctiva: evidencia de convergencia (C2) y de restricción activa (C1)

**Generado automáticamente** por `python -m ohlcv_dataroad.ingest.audit_tda11_convergence --config configs/mnq_snapshot.yaml` — nunca se edita a mano. No forma parte del pipeline de TDA-11 ni de sus conclusiones: conserva de forma reproducible el diagnóstico que justificó la auditoría correctiva (ver docstring de `audit_tda11_convergence.py`).
**Alcance de datos:** los mismos insumos derivados que TDA-11; ningún archivo de `data/raw/` ni de `holdout_files` fue abierto.

## A. C2 — ¿acredita convergencia el `convergence_flag=0` de `arch`?

Cada fila `arch_slsqp_attempt` es un intento del optimizador interno de `arch` (SLSQP) con un paso de diferencias finitas `eps` de la grilla `(None, 1e-07, 1e-06, 1e-05)`; la fila `verified_optimum_fit_garch11` es el resultado que publica TDA-11. `llf_minus_verified_optimum` = cuánta log-verosimilitud le falta al intento. `kkt_max_abs_gradient` = gradiente proyectado de la log-verosimilitud media (condición de primer orden con restricciones, `kkt_check_garch`); un punto se acepta solo si es menor que `GARCH_KKT_GRADIENT_TOL=0.001`.

| fit | source | eps | convergence_flag | n_iterations | stayed_at_starting_values | alpha | beta | persistence | loglikelihood | llf_minus_verified_optimum | kkt_max_abs_gradient | passes_kkt_tolerance | kkt_note |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| garch11_primary_global | arch_slsqp_attempt | scipy_default | 0 | 5 | True | 0.20000 | 0.78000 | 0.980000 | -4,737,061.88 | -51,210.87 | 2.25e-01 | False |  |
| garch11_primary_global | arch_slsqp_attempt | 1e-07 | 0 | 20 | False | 0.05191 | 0.94809 | 1.000000 | -4,685,851.01 | -0.00 | 1.29e-06 | True |  |
| garch11_primary_global | arch_slsqp_attempt | 1e-06 | 0 | 14 | False | 0.11836 | 0.87365 | 0.992006 | -4,708,641.17 | -22,790.16 | 2.86e-02 | False |  |
| garch11_primary_global | arch_slsqp_attempt | 1e-05 | 0 | 10 | False | 0.15372 | 0.84084 | 0.994562 | -4,719,330.57 | -33,479.56 | 4.49e-02 | False |  |
| garch11_primary_global | verified_optimum_fit_garch11 |  | — | — | False | 0.05191 | 0.94809 | 1.000000 | -4,685,851.01 | 0.00 | 1.21e-07 | True |  |
| garch11_secondary_raw_global | arch_slsqp_attempt | scipy_default | 0 | 5 | True | 0.20000 | 0.78000 | 0.980000 | -4,562,461.95 | -71,904.05 | 7.85e-02 | False |  |
| garch11_secondary_raw_global | arch_slsqp_attempt | 1e-07 | 0 | 5 | True | 0.20000 | 0.78000 | 0.980000 | -4,562,461.95 | -71,904.05 | 7.85e-02 | False |  |
| garch11_secondary_raw_global | arch_slsqp_attempt | 1e-06 | 0 | 5 | True | 0.20000 | 0.78000 | 0.980000 | -4,562,461.95 | -71,904.05 | 7.85e-02 | False |  |
| garch11_secondary_raw_global | arch_slsqp_attempt | 1e-05 | 0 | 6 | False | 0.21392 | 0.78383 | 0.997748 | -4,560,220.42 | -69,662.52 | 9.25e-02 | False |  |
| garch11_secondary_raw_global | verified_optimum_fit_garch11 |  | — | — | False | 0.07918 | 0.92082 | 1.000000 | -4,490,557.90 | 0.00 | 6.00e-08 | True |  |
| garch11_student_t_sensitivity | arch_slsqp_attempt | scipy_default | 8 | 21 | False | 0.08325 | 0.92418 | 1.007433 | -4,608,781.51 | -3,504.05 | — | False | punto fuera de la region admisible (viola alpha+beta<=1 o una cota) |
| garch11_student_t_sensitivity | arch_slsqp_attempt | 1e-07 | 0 | 14 | False | 0.12293 | 0.87127 | 0.994196 | -4,631,954.00 | -26,676.54 | 1.87e-01 | False |  |
| garch11_student_t_sensitivity | arch_slsqp_attempt | 1e-06 | 0 | 13 | False | 0.15269 | 0.84036 | 0.993052 | -4,633,835.83 | -28,558.37 | 1.13e-01 | False |  |
| garch11_student_t_sensitivity | arch_slsqp_attempt | 1e-05 | 0 | 19 | False | 0.10531 | 0.88773 | 0.993037 | -4,616,835.01 | -11,557.54 | 5.84e-02 | False |  |
| garch11_student_t_sensitivity | verified_optimum_fit_garch11 |  | — | — | False | 0.04636 | 0.95364 | 1.000000 | -4,605,277.46 | 0.00 | 1.43e-07 | True |  |

**Hechos.** De 12 intentos, 11 terminaron con `convergence_flag=0` ("éxito"); de esos, **10 NO son puntos estacionarios** (no superan la verificación), y 4 ni siquiera se movieron del punto de partida.

- **garch11_primary_global**: el mejor intento que `arch` declaró exitoso queda a -0.00 unidades de log-verosimilitud del óptimo verificado (gradiente proyectado 1.29e-06; supera la verificación).
- **garch11_secondary_raw_global**: el mejor intento que `arch` declaró exitoso queda a -69,662.52 unidades de log-verosimilitud del óptimo verificado (gradiente proyectado 9.25e-02; NO supera la verificación).
- **garch11_student_t_sensitivity**: el mejor intento que `arch` declaró exitoso queda a -11,557.54 unidades de log-verosimilitud del óptimo verificado (gradiente proyectado 5.84e-02; NO supera la verificación).

## B. C1 — ¿es `alpha+beta = 1` una estimación o el tope de la restricción?

Misma verosimilitud gaussiana y misma recursion que el ajuste primario de TDA-11 (`clock_adjusted`), permitiendo `alpha+beta` hasta 1.02. `llf_minus_constrained_optimum` = log-verosimilitud menos la del óptimo restringido que publica TDA-11.

| kind | persistence | alpha | beta | loglikelihood | llf_minus_constrained_optimum | feasible_under_arch_constraint |
|---:|---:|---:|---:|---:|---:|---:|
| profile_fixed_persistence | 0.980000 | 0.07693 | 0.90307 | -4,707,892.95 | -22,041.94 | True |
| profile_fixed_persistence | 0.990000 | 0.06117 | 0.92883 | -4,695,260.11 | -9,409.09 | True |
| profile_fixed_persistence | 0.995000 | 0.05408 | 0.94092 | -4,689,554.77 | -3,703.76 | True |
| profile_fixed_persistence | 0.998000 | 0.05172 | 0.94628 | -4,686,959.63 | -1,108.62 | True |
| profile_fixed_persistence | 0.999000 | 0.05160 | 0.94740 | -4,686,332.73 | -481.71 | True |
| profile_fixed_persistence | 0.999500 | 0.05170 | 0.94780 | -4,686,073.07 | -222.05 | True |
| profile_fixed_persistence | 0.999900 | 0.05186 | 0.94804 | -4,685,892.37 | -41.36 | True |
| profile_fixed_persistence | 1.000000 | 0.05191 | 0.94809 | -4,685,851.01 | 0.00 | True |
| profile_fixed_persistence | 1.000100 | 0.05196 | 0.94814 | -4,685,811.19 | 39.82 | False |
| profile_fixed_persistence | 1.000500 | 0.05224 | 0.94826 | -4,685,667.30 | 183.71 | False |
| profile_fixed_persistence | 1.001000 | 0.05268 | 0.94832 | -4,685,522.14 | 328.87 | False |
| profile_fixed_persistence | 1.002000 | 0.05389 | 0.94811 | -4,685,345.76 | 505.25 | False |
| profile_fixed_persistence | 1.005000 | 0.05961 | 0.94539 | -4,685,628.17 | 222.85 | False |
| free_persistence_optimum | 1.002728 | 0.05502 | 0.94771 | -4,685,308.99 | 542.02 | False |
| constrained_optimum_tda11 | 1.000000 | 0.05191 | 0.94809 | -4,685,851.01 | 0.00 | True |

**Hechos.** El máximo con persistencia libre está en `alpha+beta = 1.002728` (`alpha` = 0.05502), en el interior de la sonda, con 542.02 unidades de log-verosimilitud más que el óptimo restringido (`alpha+beta = 1.000000`). En el perfil, la verosimilitud sigue creciendo al pasar de 1: la restricción `alpha+beta ≤ 1` de `arch` está **activa**, y el `1.0000` de TDA-11 es ese tope.

**Qué NO significa.** No es una estimación de que la persistencia "verdadera" sea mayor que 1, ni evidencia de IGARCH o de memoria infinita: un máximo libre por encima de 1 en un GARCH(1,1) gaussiano es un síntoma conocido de especificación insuficiente (cambios de nivel de volatilidad, colas pesadas), compatible con las tres explicaciones que TDA-11 deja abiertas. Esta sonda no sustituye la estimación de TDA-11 ni interviene en TH23/TH24/TH25/STOP-11.

---
Tiempo de ejecución: 585.1s. Archivos: `TDA11_auditoria_c2_intentos_optimizador.csv`, `TDA11_auditoria_c1_perfil_persistencia.csv`, este informe.
