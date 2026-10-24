# TDA-17 — E02, anexo de la decisión 16: relaciones N1–N21 → tipo y columnas de P

**Estado:** contenido normativo de E02 (autorización complementaria del 2026-10-19, decisión 16, R-a corregida). La versión que se aplica es la clave `identidades.verificacion_numerica.relaciones_N` de `configs/tda17_enmienda_02.yaml`; este documento la explica. E02 sigue `PENDIENTE_AUDITORIA`.

**DECISIÓN IRIS: NINGUNA.**

## Principio aprobado

Solo una dependencia **lineal afín, exacta en todas las filas y con todos sus términos en P** genera una dirección nula exacta en una matriz de correlación.
- Esas dependencias ya están registradas como L1–L11 y cuentan en `nulos_esperados`.
- **Ninguna relación N añade nulos esperados** (`nulos_esperados_adicionales: 0`). `nulos_esperados` no cambia.
- En el diagnóstico de bloqueo, las N solo señalan **candidatas** para explicar un nulo sobrante.
- El **valor numérico** de cada relación no se calcula: queda para el expediente de la enmienda de explicación (A2-a).

## Tipos y qué publica el diagnóstico (`TDA17_diagnostico_bloqueo.csv`, sección `RELACION_N`)

| Tipo | Significado | Publicación |
|---|---|---|
| LA-P | Lineal afín, todos sus términos en P | Cargas de sus columnas en cada dirección nula observada (`DIRECCION_k`) + `PENDIENTE_EXPEDIENTE` |
| LA-F | Lineal en su forma original, algún término fuera de P o de la fila | `SIN_DIRECCION_EN_P` por columna de P implicada + `PENDIENTE_EXPEDIENTE` |
| NL-P | No lineal, todos sus términos en P | Igual que LA-P |
| NL-F | No lineal, cruza paneles | Igual que LA-F |
| DES | Desigualdad | Recuento de filas de la población de la celda que la violan (`VIOLACIONES`) |
| DIAG | Diagnóstico no universal | `NO_IDENTIDAD` |

## Correspondencia

| N | Relación congelada (resumen) | Tipo | Columnas de P | Fuera de P / nota |
|---|---|---|---|---|
| N1 | `R_1m = exp(r_1m) − 1` | NL-F | `r_1m` | `R_1m` (solo tabla de identidades) |
| N2 | `abs_r_1m = \|r_1m\|`; `r2_1m`; `zero_1m` | NL-P | `abs_r_1m`, `r_1m` | `r2_1m`, `zero_1m` (resto: NL-F) |
| N3 | `%K_rapido(n) = %R(n) + 100` | LA-F | `B5_n14__wr`, `B5_n60__wr` | `%K_rapido` es interno de B4 (P contiene `%K` suavizado) |
| N4 | `pos_Donchian(n) = (%R(n) + 100)/100` | LA-P | `C5_n60__pos`, `B5_n60__wr` | Solo n = 60 coincide; **es L10** (sin nulo adicional) |
| N5 | `CMO_n = 2 RSI_Cutler_n − 100` | LA-F | `B7_n14__cmo`, `B7_n60__cmo` | `RSI_Cutler` no se publica (P tiene RSI de Wilder) |
| N6 | `ER_n = \|CMO_n\|/100` | NL-F | `B7_n14__cmo`, `B7_n60__cmo` | `ER` es interno de A5 (KAMA) |
| N7 | `beta_n = 6 (WMA_n − SMA_n)/(n − 1)` | NL-P | `A12_n*__pend_rel`, `A3_n*__dist`, `A1_n*__dist` (n = 20, 60, 240) | Lineal en niveles, exponencial con `ln(C/MA)` |
| N8 | `%B − 1/2 = (C/SMA − 1)/BW` | NL-P | `C3_n*_k2__pct_b`, `A1_n*__dist`, `C3_n*_k2__ln_bw` (n = 20, 60, 240) | Cociente |
| N9 | `pos_Keltner − 1/2 = (C/EMA − 1)/ancho` | NL-P | `C4_n*__pos`, `A2_n*__dist`, `C4_n*__ln_ancho` (n = 20, 60, 240) | Cociente |
| N10 | `dir_ST` frente a `sign(ln C/ST)` | DIAG | `A11_n10_m3__dir`, `A11_n10_m3__dist`, `A11_n60_m3__dir`, `A11_n60_m3__dist` | No es identidad (contraejemplo sintético) |
| N11 | `CLV = sign(C − O) cuerpo + mecha_inf − mecha_sup` | NL-P | `E4`, `E1`, `E2`, `E3` (`*_barra__valor`) | Lineal **a trozos** según `sign(C − O)` |
| N12 | `CMF_n = ΔADL_n / Σ_n V` | LA-F | `D3_n20__cmf`, `D3_n60__cmf`, `D3_n240__cmf` | `D4_acumulado__adl_acumulado` (Aux) |
| N13 | `flujo_OBV_n = ΔOBV_n / Σ_CONS V` | LA-F | `D1_n20__flujo`, `D1_n60__flujo`, `D1_n240__flujo` | `D1__obv_acumulado` (Aux) |
| N14 | `NATR = 25 ATR_ticks / C` | NL-F | `C2_n*__ln_natr` (n = 14, 60, 240) | `C1_n*__atr_ticks` (L) y `C` |
| N15 | `Momentum_n = C_t (1 − exp(−ROC_n))/0.25` | NL-F | `B1_n*__roc` (n = 10, 60, 240) | `B2_n*__mom_ticks` (L) |
| N16 | `rango_ticks = L (exp(log_hl) − 1)/0.25` | NL-F | `log_hl` | `E5_barra__valor` (L) |
| N17 | `Parkinson_n² = mean_n(log_hl²)/(4 ln 2)`; GK por barra | NL-P | `C7_n*__ln_sigma`, `C8_n*__ln_sigma` (n = 20, 60, 240), `log_hl`, `log_co` | Media móvil de cuadrados |
| N18 | `MACD_N_x = MACD_conv_x / C_t` | NL-F | `A6_*__n_linea`, `A6_*__n_senal`, `A6_*__n_hist` (3 configuraciones) | `A6_*__conv_*` (L) |
| N19 | `PPO = 100 MACD_N_linea C_t / EMA_s` | NL-P | `A7_f28_s60_g21__linea`, `A6_f28_s60_g21__n_linea`, `A2_n60__dist`, `A7_f111_s240_g83__linea`, `A6_f111_s240_g83__n_linea`, `A2_n240__dist` | Exacta con P solo para s = 60 y 240; multiplicativa |
| N20 | `ROC P1 = ln(C_t/C_t−n)` sin discontinuidad | LA-F | `B1_n10__roc`, `B1_n60__roc`, `B1_n240__roc`, `r_1m` | Suma temporal de `r_1m` (fuera de la fila) |
| N21 | `\|log_co\| ≤ log_hl` | DES | `log_co`, `log_hl` | Violación: `\|log_co\| > log_hl` en una fila (comparación exacta, sin tolerancia) |

## Lecturas de implementación

- **Cargas de LA-P y NL-P.** Se publican las cargas de las columnas implicadas presentes en la matriz afectada. En `B_centrada` faltan las 11 columnas `fuera_de_B`. Si una matriz afectada no tiene direcciones nulas, se publica `SIN_DIRECCIONES_NULAS`.
- **N21.** Se cuenta sobre la población de casos completos de la celda, y solo cuando la celda emite el diagnóstico de bloqueo.
- **Validación del anexo.** `tda17_analysis.relations_n` rechaza cualquier anexo distinto del implementado: ids y orden distintos de los congelados, tipos o columnas inexistentes, N4 sin identidad registrada, N21 sin sus dos columnas, nulos adicionales distintos de 0 o valor numérico distinto del expediente.
