# TDA-17 — Registro de las decisiones normativas autorizadas para E02

**Estado:** `AUTORIZACION_METODOLOGICA_REGISTRADA`.

**Fecha de la autorización:** 2026-10-18.

**Responsable:** Responsable del proyecto IRIS. Es una declaración de autorización; la identidad no está autenticada.

**Documento autorizado:** `reports/mnq/TDA17_E02_propuesta_decisiones_normativas.md`, SHA-256 (LF) `192c5a43be79835d620ee7484df1b8f8242e8ef73e49a70f182b6109cb92f0a0`.

**Instrucción recibida:** «TDA-17 — APROBACIÓN METODOLÓGICA Y AVANCE DE E02».

**Autorización complementaria (2026-10-19):** decisiones 14–19, sección 4. Documento autorizado `reports/mnq/TDA17_E02_propuesta_ambiguedades_y_salidas.md`, SHA-256 (LF) `f16672c3ac80505e3a70a0d1ef68619dff56175da9f153bab936f0c117cb46d6`; instrucción «TDA-17 — Aprobación de las 6 decisiones y avance de E02».

> **Alcance de esta autorización.**
> - Autoriza las **decisiones metodológicas** de la tabla siguiente.
> - **No** constituye la aprobación del código definitivo de E02.
> - **No** autoriza ejecutarlo sobre MNQ.
>
> El registro `reports/mnq/TDA17_enmiendas_E02.json` sigue en `PENDIENTE_AUDITORIA` con `aprobacion: null`. E02 solo será activable tras auditoría independiente y el ciclo A2 → P2 (SPEC-M2 §10). Ninguna aprobación se ha simulado en el repositorio del proyecto.
>
> **DECISIÓN IRIS: NINGUNA.**

## 1. Matriz de decisiones (propuesta §8) con la opción autorizada

| # | Decisión | Opción autorizada | Clave de E02 (`anadir`) |
|---|---|---|---|
| 1 | A1: convención del recuento de nulos y matriz degenerada | **A1-a + F5-a**: autovalores de la correlación (P/B centradas) y de los cosenos (P sin centrar); `λ ≤ τ·λ_max`; negativos por redondeo cuentan como nulos, sin recorte; valores singulares solo informativos. Matriz NO_DEFINIDA (varianza o norma cero): sin análisis matricial, incumplimiento para todas las columnas; si es GLOBAL, bloqueo de TH35 con A2 | `identidades.verificacion_numerica.convencion` |
| 2 | A4: ámbito del control | **A4-a**: bloqueante en GLOBAL, años evaluables y segmentos; informativo en 2019, año×segmento, L por contrato y réplicas G2; una sensibilidad solo se bloquea a sí misma; el bloqueo afecta a todas las conclusiones de TH35 | `identidades.verificacion_numerica.ambito` |
| 3 | A2: diagnóstico y desbloqueo | **A2-a**: diagnósticos obligatorios, expediente, levantamiento solo por enmienda formal auditada (ciclo M2) y prohibiciones explícitas | `identidades.verificacion_numerica.procedimiento_bloqueo` |
| 4 | B1: intervalo | **B1-a**: percentil tipo 7, cuantiles **0,05 / 0,95**, para R²_TODAS y Spearman; cobertura nominal | `analisis.bootstrap.intervalo` |
| 5 | B2: remuestreo | **B2-a**: por celda; unidad trading_date; segmento = parte de la jornada; circular cronológico dentro de la celda; población fija | `analisis.bootstrap.ambito` |
| 6 | B6: reproducibilidad | PCG64; `SeedSequence(1717).spawn(n_celdas)` en el orden declarado; S4 con los mismos flujos reiniciados | `analisis.bootstrap.reproducibilidad` |
| 7 | B3: estadístico de réplica | **B3-a + F2-a**: método oficial en cada réplica; 300 réplicas siempre; réplica INVALIDA (VAR_CERO, BASE_DEGENERADA, NO_FINITO, FUERA_DE_RANGO) → IC NO_CALCULABLE e incumplimiento; nunca se descarta, regenera, sustituye ni recorta | `analisis.bootstrap.estadistico_replica` |
| 8 | B4: Spearman | **F3-I + F3-I-b**: validación con 30 y, si falla, ampliada con 100; publicación con rangos fijos y 300 réplicas; si ambas fallan, no se publican los IC de Spearman (sí los coeficientes); celdas GLOBAL y ANIO_2024 | `analisis.spearman.validacion_detalle` |
| 9 | B13: celdas no publicables | **B13-a**: incumplimiento de ambas condiciones; recuento por columna | `analisis.clasificacion.celdas_no_publicables` |
| 10 | D17: auditoría causal TH37 | **D17-1a**, incluida la **semilla nueva 3717** | `analisis.auditoria_causalidad` |
| 11 | A3 + B14: hueco espectral | **APLAZADA**. Sin clave en E02. Toda publicación de EXACTA por celda que dependa de esa definición queda **bloqueada** (`PENDIENTES['HUECO_ESPECTRAL']`, `publish_exact_by_cell`) | — |
| 12 | B9: subniveles | **INCLUIDA**: R²_TODAS puntual de la celda GLOBAL | `analisis.clasificacion.subniveles_r2` |
| 13 | Checklist M2 (§7) | **Aprobado**: verificación estricta no eludible; publicación remota de P2; solo configuración oficial para resultados oficiales; sin mecanismos de omisión | (implementación) |

**Valores congelados que E02 no cambia:**
- τ = 1e-10 y `nulos_esperados`;
- umbrales 0,90 / 0,50, mínimos 5/6 y 6/7, tolerancias 1e-8 y 0,01;
- 300 réplicas, bloques de 5 jornadas, semilla 1717, 30 y 100 réplicas exactas;
- `hueco_espectral_minimo`;
- parámetros del catálogo y D1–D24.

La única cifra nueva es la semilla 3717.

## 2. Lecturas de implementación que la propuesta no fija literalmente

Se aplicaron sin cambiar ningún valor normativo y se señalan para la auditoría independiente; cualquiera puede revisarse antes de A2.

| Punto | Lectura aplicada | Motivo |
|---|---|---|
| Orden de los flujos (B6): «años ascendentes» | Todos los años **declarados** (2019 a 2025), es decir, 15 flujos: GLOBAL + 7 años + 7 segmentos | Lectura literal del orden declarado de celdas. Excluir 2019 desplazaría los flujos de los segmentos |
| Mismas réplicas para R² y Spearman en una celda | Un único juego de réplicas por celda (flujo de la celda) para ambos IC; la validación usa su prefijo | B6: un flujo por celda; F3: «prefijo del flujo de la celda» |
| Invalidez en Spearman | Solo VAR_CERO (rangos constantes) y NO_FINITO; no se define FUERA_DE_RANGO para ρ | F2 solo extiende a Spearman el caso de rangos constantes |
| Cuantiles de contrato (D17) | Barra `ceil(q·n)`, con vecindad c−1, c, c+1 (están en `cortes_deterministas`) | Regla de calendario predeclarada; aplicar la vecindad solo añade cortes |
| Cortes aleatorios (D17) | `integers(2, n, size=10)` (barras 2..n−1, con reemplazo; los repetidos se usan una vez) | «10 barras uniformes entre la 2 y la penúltima» + regla de duplicados |
| Corte fuera de 2..n−1 (D17) | NO_APLICABLE con motivo `CONTRATO_CORTO` y el detalle «corte fuera de 2..n−1» | Los motivos admitidos son solo dos; no se inventa un tercero |
| Perturbación (D17) | Toda la barra + s ticks; además H, L y C ajustados para cambiar rango, cuerpo y mechas; volumen + (k mod 4 + 1) | Una perturbación que solo desplaza el nivel no detecta fugas en las series de forma de barra (lo demostraron las pruebas) |
| Roll 3a | Factor 2, +1000 ticks y + ((fila mod 7) + 1) ticks sobre los demás contratos | En la rejilla y con las invariantes OHLC |
| Rango en celdas informativas | Estado `DIFIERE_INFORMATIVO` (solo informe) | A4-a: «solo se informa» |

## 3. Funcionalidad aislada y bloqueada (decisiones NO autorizadas o no definidas)

| Clave en `PENDIENTES` | Qué queda bloqueado | Por qué |
|---|---|---|
| `HUECO_ESPECTRAL` | Publicación de EXACTA por celda (`publish_exact_by_cell`, `spectral_gap_status`) | Decisión 11 aplazada |
| ~~`SPEARMAN_VALIDACION_REPLICA_INVALIDA`~~ | **Resuelta** por la decisión 14 (S-a) | — |
| ~~`SUBNIVEL_R2_GLOBAL_NO_PUBLICABLE`~~ | **Resuelta** por la decisión 15 (N-a) | — |
| ~~`RELACIONES_N_VALOR`~~ | **Resuelta** por la decisión 16 (R-a corregida): tipo y columnas de P; el valor numérico queda, por diseño aprobado, para el expediente | — |
| `SALIDAS_OTRAS_TABLAS` (antes `SALIDAS_ESQUEMA_VINCULACION`) | Tablas de `salidas.tablas` fuera del esquema mínimo (catálogo, ventanas, identidades, pérdidas, Pearson, transferencia, dimensión, agrupamiento, sensibilidades, G2, panel L, rvol) | Las decisiones 17–19 solo fijan cinco tablas y el diagnóstico |
| `R2_OTROS_INDICADORES`, `R2_TRANSFERENCIA` | Variante R2_OTROS_INDICADORES y transferencia | B10 y B8 no autorizadas |
| `T3_V2`, `K_PA_MAX_T_V3` | Análisis paralelo, k_PA, max-T, V3, T3/V2 y el uso de G2 en producción (`official_g2_controls`) | B11 y B12 no autorizadas |
| `S2_ESCALAS`, `S7_S8_ALCANCE` | Sensibilidades S2, S7 y S8 | C1–C3 y D3 no autorizadas |
| `REGISTRO_D7_RETRASO` | «Retraso» en barras de D7 en el registro de calentamiento | No definido |

## 4. Decisiones complementarias autorizadas el 2026-10-19

| # | Decisión | Opción autorizada | Clave de E02 (`anadir`) |
|---|---|---|---|
| 14 | Spearman: réplica INVALIDA en la validación | **S-a**: en cualquiera de los dos métodos (rangos fijos o recálculo exacto), la validación de **esa celda** falla en esa ronda; después, F3-I-b. Sin imputación ni exclusión del par | `analisis.spearman.validacion_replica_invalida` |
| 15 | Subnivel con R² global no publicable | **N-a**: `SIN_SUBNIVEL_R2_NO_PUBLICABLE`, motivo `R2_NO_PUBLICABLE_POR_COLINEALIDAD`, sin sustituir el R² | `analisis.clasificacion.subniveles_r2_global_no_publicable` |
| 16 | Relaciones N1–N21 | **R-a corregida**: tipo de cada relación (LA-P, LA-F, NL-P, NL-F, DES, DIAG) y columnas de P según el anexo `reports/mnq/TDA17_E02_anexo_relaciones_N.md`; ningún nulo esperado nuevo | `identidades.verificacion_numerica.relaciones_N` |
| 17 | Esquema de las tablas oficiales | Cinco tablas de la §4.1 de la propuesta | `salidas.esquema_minimo` |
| 18 | Vinculación | Ruta única `official_publish_th35` (solo recibe P2; calcula `official_th35` en el mismo proceso) y `TDA17_procedencia_oficial.json`; verificación posterior | `salidas.esquema_minimo` |
| 19 | Publicación con bloqueo | Solo `TDA17_auditoria_causalidad.csv` y `TDA17_diagnostico_bloqueo.csv` (§4.3); sin clasificación, subniveles ni estabilidad por celda | `salidas.diagnostico_bloqueo` |

**Valores congelados.** Las decisiones 14–19 no cambian ningún valor congelado ni ninguna decisión 1–13. No hay cifras nuevas.

**Lecturas de implementación** (revisables antes de A2):

| Punto | Lectura aplicada |
|---|---|
| Causas de invalidez en la validación (S-a) | Las de Spearman ya fijadas: rangos constantes de una columna (VAR_CERO) o correlación no finita (NO_FINITO), en cualquiera de los dos métodos |
| Directorio de las tablas | `reports/mnq` (el del informe de `salidas.informe`; §4.3 de la propuesta) |
| Serialización | CSV con separador `,`, fin de línea LF, UTF-8, sin índice; NaN como campo vacío (por defecto de pandas) |
| Sobrescritura | Prohibida: si ya existe alguna salida oficial o el registro, la publicación rechaza (`SALIDA_EXISTENTE`) |
| Diagnóstico con bloqueos sin expediente | Si una celda bloquea por matriz GLOBAL no definida o por error de identidad registrada (sin diagnóstico de rango), el diagnóstico contiene al menos las filas `AMBITO` con el texto de cada bloqueo |
| Verificación posterior | `check_official_publication` (registro, P2, efectiva, entradas, conjunto, esquema y huellas; integrada en `verificar --referencia-git P2` si existe el registro) y `reproduce_official_publication` (recalcula y compara byte a byte) |

## 5. Firma

No hay firma criptográfica. Este documento registra la autorización metodológica declarada por el responsable. La aprobación de E02 se registrará solo en `reports/mnq/TDA17_enmiendas_E02.json`, con el acta de E02, en el commit P2, tras la auditoría independiente del código en A2.
