# TDA-17 — Propuesta breve: tres ambigüedades de E02 y esquema y vinculación de las salidas oficiales

**Estado:** `PROPUESTA_PARA_DECISION`. No es normativa:
- nada de lo que aquí se propone está implementado;
- las funciones afectadas siguen bloqueadas (`PENDIENTES`);
- no se cambia ningún valor congelado ni ninguna decisión ya autorizada;
- no se ha ejecutado nada sobre MNQ.

**DECISIÓN IRIS: NINGUNA.**

Cada punto requiere tu aprobación explícita: *Aprobada*, *Otra* o *Rechazada*.

## 1. Spearman: réplica inválida durante la validación (`SPEARMAN_VALIDACION_REPLICA_INVALIDA`)

**Texto autorizado:** «una réplica INVALIDA en la validación hace que la validación falle para ese par». No fija cómo entra ese par en la mediana sobre pares y extremos.

| Opción | Regla | Consecuencia |
|---|---|---|
| **S-a (recomendada)** | Si una réplica de validación es INVALIDA para algún par, en cualquiera de los dos métodos, la validación de **esa celda falla** en esa ronda (30 o 100). Se aplica después el procedimiento ya autorizado: ronda ampliada y, si vuelve a fallar, IC de Spearman no publicados (F3-I-b) | Coherente con F2-a (tolerancia cero). No hay que asignar un valor a la diferencia de un par sin IC. Solo puede llevar a «no publicar»: es conservadora |
| S-b | El par inválido entra en la mediana con diferencia **+∞** (peor caso) | Puede pasar la validación con pocos pares inválidos. Es una imputación numérica nueva |
| S-c | Excluir el par de la mediana | No admisible: descarta información (como F2-c) |

## 2. Subnivel con R² global no publicable (`SUBNIVEL_R2_GLOBAL_NO_PUBLICABLE`)

| Opción | Regla | Consecuencia |
|---|---|---|
| **N-a (recomendada)** | La columna no registrada cuyo R²_TODAS de la celda GLOBAL es `R2_NO_PUBLICABLE_POR_COLINEALIDAD` recibe el estado `SIN_SUBNIVEL_R2_NO_PUBLICABLE`, publicado con el motivo | No inventa un R². Los subniveles son descriptivos y no afectan a la clasificación (B9) |
| N-b | Usar el R² espectral o de mínima norma | Sustituye el método oficial justo cuando las verificaciones discrepan: no recomendada |
| N-c | Usar la mediana de las celdas publicables | Cambia la fuente aprobada en B9 (celda GLOBAL): no recomendada |

## 3. Relaciones N1–N21 en el diagnóstico de bloqueo (`RELACIONES_N_VALOR`)

Las relaciones están registradas como **texto** (`identidades.no_lineales_o_fuera_de_P`). **No son todas del mismo tipo**, y solo una dependencia **lineal afín, exacta en todas las filas y con todos sus términos en P** genera una dirección nula exacta en una matriz de correlación.
- Las dependencias de ese tipo ya están registradas como L1–L11 y cuentan en `nulos_esperados`.
- Ninguna N añade direcciones nulas esperadas. Su papel en el diagnóstico es otro: señalar **cuasidependencias candidatas** cuando aparezca un nulo sobrante.

**Clasificación propuesta.** Se deriva del texto congelado y del catálogo (paneles P, L, Aux y `solo_tabla_identidades`). La correspondencia columna a columna irá en el anexo de R-a.

| Tipo | N | Términos | Qué puede producir en P |
|---|---|---|---|
| **LA-P**: lineal afín con todos sus términos en P | N4 | `C5_*__pos`, `B5_*__wr` | Dirección nula exacta solo si coincide `n` (n60). Esa dirección **ya es L10**: no añade ningún nulo |
| **LA-F**: lineal en su forma original, pero con algún término fuera de P o fuera de la fila | N3 (`%K_rapido`, interno de B4), N5 (`RSI_Cutler`, no publicado), N12 y N13 (ADL y OBV en Aux), N20 (suma temporal de `r_1m`; condicionada a ventanas sin discontinuidad) | Mezclan P con cantidades fuera de P | Ninguna dirección nula en P. Solo correlación alta |
| **NL-P**: no lineal con todos sus términos en P | N2 (solo `abs_r_1m = \|r_1m\|`), N7 (en P: `pend_rel`, `ln(C/WMA)`, `ln(C/SMA)`; lineal en niveles, exponencial en P), N8, N9, N11 (lineal **a trozos** según `sign(C − O)`), N17 (media móvil de `log_hl²`), N19 (exacta con P para s = 60 y 240, pero multiplicativa) | P | **Cuasidependencia** (autovalor pequeño, no nulo). Una relación a trozos o condicionada puede volverse **exacta en una submuestra** donde la condición se cumple siempre: esta es la causa candidata principal de un nulo sobrante en una celda |
| **NL-F**: no lineal y cruza paneles | N1 y N2 (`R_1m`, `r2_1m`, `zero_1m`: solo identidades), N6 (`ER`, interno de KAMA), N14 (`ln_natr` P frente a `atr_ticks` L), N15 (`mom_ticks` L frente a `roc` P), N16 (`E5` L frente a `log_hl` P), N18 (`n_linea` P frente a `conv_linea` L) | P con L, Aux o columnas solo de identidades | Nada en P. Solo es relevante para el diagnóstico del panel L y la transferencia |
| **DES**: desigualdad | N21 (`\|log_co\| ≤ log_hl`) | P | **Nunca** una dirección nula. Se puede comprobar como restricción: número de filas que la violan, que debe ser 0 |
| **DIAG**: diagnóstico no universal | N10 (dirección del Supertrend frente a `sign(ln C/ST)`) | P | **Nunca** una dirección nula, porque no es una identidad. Solo trazabilidad: concordancia descriptiva |

| Opción | Regla | Consecuencia |
|---|---|---|
| **R-a (recomendada, corregida)** | El diagnóstico publica, para cada N, su **tipo** (tabla anterior) y sus **columnas de P implicadas**, según un anexo N → columnas que se aprueba en E02. Lo que añade depende del tipo:<br>• **LA-P y NL-P:** las cargas de esas columnas en cada dirección nula observada;<br>• **LA-F y NL-F:** `SIN_DIRECCION_EN_P` y las columnas de P implicadas, si las hay;<br>• **DES:** el recuento de violaciones;<br>• **DIAG:** `NO_IDENTIDAD`.<br>No se evalúa ninguna fórmula no lineal; su valor numérico queda para el expediente de la enmienda de explicación (A2-a) | Ninguna relación se presenta como generadora de nulos salvo N4, ya cubierta por L10. El diagnóstico apunta a las candidatas reales (NL-P a trozos o condicionadas). El recuento de N21 es una comprobación barata y no ambigua |
| R-b | Igual que R-a y, además, evaluar ahora numéricamente las relaciones NL-P | Cada fórmula sobre columnas transformadas es una decisión técnica nueva que habría que auditar una a una |
| R-c | Mantener solo el texto de las relaciones (como ahora) | Diagnóstico menos útil, pero sin riesgo |

## 4. Esquema mínimo y vinculación de las tablas oficiales con `official_th35`

**Situación actual.** `write_official_table` está bloqueada (`SALIDAS_ESQUEMA_VINCULACION`). El protocolo nombra las tablas de `salidas.tablas` pero no fija sus columnas, salvo `TDA17_auditoria_causalidad.csv`, cuyo esquema sí fija E02 (`tabla.campos`).

### 4.1 Esquema mínimo propuesto (solo tablas que `official_th35` ya calcula)

| Tabla | Columnas | Fuente en `official_th35` |
|---|---|---|
| `TDA17_auditoria_causalidad.csv` | Las 12 de E02 (`prueba` … `etiqueta`) | `auditoria_causalidad` |
| `TDA17_estabilidad_celdas.csv` | `columna, celda, tipo, estado, estado_matriz, r2, ic_inf, ic_sup, publicable, ic_estado` | `per_cell` |
| `TDA17_r2_reconstruccion.csv` | `celda, columna, r2, r2_espectral, r2_minima_norma, dif_cruzada, publicable, etiqueta_publicacion, etiqueta_exactitud` (celdas con matriz definida) | `analisis_celdas[*]` |
| `TDA17_correlacion_spearman.csv` | `celda, columna_i, columna_j, rho, ic_inf, ic_sup, ic_estado, replicas_invalidas` | `spearman` (IC vacíos si la validación no pasa) |
| `TDA17_clasificacion_final.csv` | `columna, clasificacion, subnivel, motivo_subnivel, r2_global, MATRIZ_NO_DEFINIDA, R2_NO_PUBLICABLE_POR_COLINEALIDAD, IC_NO_CALCULABLE, total_no_publicables` | `clasificacion` + `subniveles` (solo sin bloqueos) |

Las demás tablas de `salidas.tablas` (transferencia, sensibilidades, G2, dimensión efectiva, agrupamiento, panel L, catálogo, ventanas, …) **siguen bloqueadas** hasta que se definan.

### 4.2 Vinculación propuesta

1. **Una única ruta de publicación**, `official_publish_th35(publicacion, root)`:
   - ejecuta `official_th35` en el mismo proceso (configuración re-derivada de M2 y entradas verificadas contra el manifiesto);
   - escribe **exactamente** las tablas de §4.1 a partir de los objetos que acaba de calcular.
   - **No recibe ninguna tabla.** `write_official_table` deja de ser pública.
2. **Registro de procedencia** `reports/mnq/TDA17_procedencia_oficial.json`:
   - contenido: P2, SHA remoto, `sha256_efectiva`, huellas de las entradas, huella del marco unido y SHA-256 de cada tabla escrita;
   - es un fichero **nuevo** fuera de `salidas.tablas`: requiere aprobación.
3. **Verificación posterior.** `verificar --referencia-git P2` comprobaría que cada tabla publicada tiene la huella registrada y que el registro cita la P2 vigente.
4. **Con bloqueos** (`BLOQUEADO_HASTA_EXPLICAR`):
   - solo se publican `TDA17_auditoria_causalidad.csv` y el diagnóstico definido en §4.3;
   - la clasificación y los subniveles no se escriben.
   - ¿Debe publicarse también la estabilidad por celda en ese caso? Decisión tuya; mi recomendación es **no**.

### 4.3 Salida diagnóstica durante un bloqueo (propuesta)

**Nombre.** `reports/mnq/TDA17_diagnostico_bloqueo.csv`. Es un fichero **nuevo**: no está en `salidas.tablas`, por lo que requiere aprobación (punto 6). Se escribe **solo** cuando `official_th35` devuelve `estado = BLOQUEADO_HASTA_EXPLICAR`, y nunca sin bloqueo.

**Formato.** Largo (una fila por valor). Cubre los seis diagnósticos obligatorios de `procedimiento_bloqueo`.

| Columna | Contenido |
|---|---|
| `celda` | Celda afectada (GLOBAL, ANIO_*, segmento) |
| `alcance` | `BLOQUEANTE` / `INFORMATIVA` (A4-a) |
| `matriz` | `P_sin_centrar`, `P_centrada`, `B_centrada`, o vacío si la fila no depende de la matriz |
| `seccion` | `AMBITO`, `ESPECTRO`, `DIRECCION_NULA`, `R2`, `EXACTITUD`, `RESIDUO_IDENTIDAD`, `RELACION_N` |
| `elemento` | Según la sección:<br>• `ESPECTRO` y `DIRECCION_NULA`: índice del autovalor o de la dirección;<br>• `RESIDUO_IDENTIDAD`: id L;<br>• `RELACION_N`: id N;<br>• `AMBITO`: `nulos_observados` / `nulos_esperados` |
| `columna` | Columna de P, o vacío |
| `metodo` | Según la sección:<br>• `ESPECTRO`: `AUTOVALOR_A1a` (normativo) o `SINGULAR` (informativo);<br>• `R2`: `R2_TODAS` / `R2_TDA04` × `base` / `espectral` / `minima_norma`;<br>• `RELACION_N`: tipo (LA-P … DIAG) |
| `valor` | Número: autovalor, carga, R², residuo relativo, recuento. Vacío si no hay número |
| `etiqueta` | Etiqueta textual: exactitud, publicación del R², `SIN_DIRECCION_EN_P`, `NO_IDENTIDAD`, `PENDIENTE_EXPEDIENTE` |
| `marca` | Siempre `DIAGNOSTICO_BLOQUEADO` |

**Correspondencia con `blocking_dossier`** (todo existe ya en el resultado de `official_th35`):

| Sección | Campo de `blocking_dossier` |
|---|---|
| `ESPECTRO` | `espectros` |
| `DIRECCION_NULA` | `direcciones_nulas`, con todas las cargas y sin umbral |
| `R2` y `EXACTITUD` | `r2_diagnostico` y `exactitud_diagnostico` |
| `RESIDUO_IDENTIDAD` | `residuos_identidades` |
| `RELACION_N` | `relaciones_N`, según R-a |
| `AMBITO` | `ambito_afectado` |

**Procedencia.** Es la misma que la de §4.2:
- solo lo escribe `official_publish_th35`, a partir de `analisis_celdas[*]["diagnostico_bloqueo"]` del `official_th35` que acaba de ejecutar en el mismo proceso;
- no recibe ninguna tabla;
- su SHA-256, el `estado`, la lista `bloqueos` y la P2, el SHA remoto, `sha256_efectiva` y las huellas de entradas se registran en `TDA17_procedencia_oficial.json`;
- `verificar --referencia-git P2` comprueba esa huella.

**Garantías.** Publicarlo **no** levanta el bloqueo (`levantamiento: ENMIENDA_FORMAL_POSTERIOR_AUDITADA_CICLO_M2`). No contiene la clasificación ni los subniveles (`CONSULTAR_CLASIFICACION_ANTES_DEL_LEVANTAMIENTO` sigue prohibido).

## 5. Matriz de aprobación

| # | Punto | Recomendación | Aprobación |
|---|---|---|---|
| 1 | Spearman, réplica inválida en la validación | S-a | ☐ Aprobada ☐ Otra ☐ Rechazada |
| 2 | Subnivel con R² global no publicable | N-a (`SIN_SUBNIVEL_R2_NO_PUBLICABLE`) | ☐ Aprobada ☐ Otra ☐ Rechazada |
| 3 | Relaciones N1–N21 | R-a corregida (tipo de cada N + columnas de P; cargas solo para LA-P/NL-P; recuento de violaciones de N21; N10 = `NO_IDENTIDAD`; anexo a redactar) | ☐ Aprobada ☐ Otra ☐ Rechazada |
| 4 | Esquema mínimo de 5 tablas (§4.1) | Aprobar | ☐ Aprobada ☐ Otra ☐ Rechazada |
| 5 | Ruta única `official_publish_th35` + `TDA17_procedencia_oficial.json` (§4.2) | Aprobar | ☐ Aprobada ☐ Otra ☐ Rechazada |
| 6 | Publicación con bloqueos (§4.2.4 y §4.3) | Solo auditoría causal + `TDA17_diagnostico_bloqueo.csv` (§4.3) | ☐ Aprobada ☐ Otra ☐ Rechazada |

**Instrumento.**
- Los puntos 1–3 y 6 son normativos: entrarían como claves `anadir` nuevas de E02 (por ejemplo `analisis.spearman.validacion_detalle.replica_invalida_regla`, `analisis.clasificacion.subniveles_r2.global_no_publicable`, `identidades.verificacion_numerica.procedimiento_bloqueo.relaciones_N`), revisando la lista blanca y la huella del YAML antes de A2.
- Los puntos 4–5 entrarían como clave `anadir` `salidas.esquema_minimo`.
- El esquema de §4.3 entraría con el punto 6 como clave `anadir` `salidas.diagnostico_bloqueo`.

Nada de esto se implementa hasta tu aprobación.
