# TDA-17 — Propuesta de decisiones normativas para E02 (núcleo imprescindible)

**Estado:** `PROPUESTA_PARA_DECISION`. Documento de trabajo, **no normativo**:
- no crea E02 ni su YAML;
- no modifica el protocolo congelado (`9cdea2f`), E01 (`24f9f1d`), el motor ni el inventario aprobado (revisión 3.1);
- no se ha ejecutado nada sobre MNQ ni se ha leído el holdout.

Cada recomendación es una **propuesta** que el responsable del proyecto debe aprobar, sustituir o rechazar explícitamente en la matriz final (§8).
**Arquitectura de E02:** M2 (cargador escalonado), según `reports/mnq/TDA17_especificacion_migracion_M2.md` (SPEC-M2), aprobada; sus evidencias se conservan sin cambios.
**Revisión 1** (auditoría correctiva):
- **F1:** nivel del IC, con la lectura bilateral y la unilateral separadas, 90 % frente a 95 %, el error Monte Carlo y la cobertura.
- **F2:** réplicas inválidas de R².
- **F3:** Spearman, validación frente a publicación.
- **F4:** cortes de TH37 cerca de fronteras y contratos cortos.
- **F5:** matrices degeneradas en A1 y A4.

La relación de cambios está en §9. Ninguna decisión queda aprobada.
**DECISIÓN IRIS: NINGUNA.**

## 0. Fuentes y reglas de la propuesta

| Abrev. | Fuente |
|---|---|
| YAML | `configs/tda17.yaml` (CONGELADO; citado por clave y línea) |
| E01 | `configs/tda17_enmienda_01.yaml` (APROBADA) |
| INV | `reports/mnq/TDA17_inventario_decisiones_pendientes.md` (revisión 3.1; fichas A1–A4, B1–B4, B6, B13, D17) |
| SPEC-M2 | `reports/mnq/TDA17_especificacion_migracion_M2.md` |
| T-V4A | `tests/test_tda17_diseno_v4a_sintetico.py` (diseño auditado) |
| T-PROT | `tests/test_tda17_protocol.py` |
| BACK | `docs/methodology/Tsay_empirical_hypotheses_backlog.md` (TH35 l. 691–707, TH37 l. 727–743) |
| ROAD | `docs/methodology/Tsay_OHLCV_analysis_roadmap.md` §TDA-17 (l. 1086–1119) |
| DIAG | diagnóstico **sintético** de rango (fase 2); describe mecanismos, no predice MNQ |

**Reglas que respeta toda la propuesta:**
1. **Ningún parámetro congelado cambia.** Siguen igual τ = 1e-10 (l. 372, 400), los nulos esperados 10/11/0/3 (l. 373), los umbrales 0,90 / 0,50 / 5 de 6 / 6 de 7 (l. 424–433), la tolerancia cruzada 1e-8 (l. 401), los residuos 1e-9 y 1e-4 (l. 414–415) y el bootstrap (5 jornadas, 300 réplicas, semilla 1717; l. 423), así como D1–D24.
2. **E02 solo añade claves.** No sustituye ninguna regla congelada (SPEC-M2 §7). Las rutas propuestas (§6) no están entre las prohibidas.
3. **Ningún desbloqueo es automático.** Ningún resultado de MNQ interviene en ninguna decisión.

---

## 1. A1 + A4 — Convención de autovalores nulos y ámbito del control

### 1.1 Lo que ya fija el texto congelado

- `identidades.verificacion_numerica` (l. 371–374):
  - `tau_relativa: 1.0e-10`;
  - `nulos_esperados: {P_sin_centrar: 10, P_centrada: 11, B_centrada: 0, L: 3}`;
  - `accion_si_difiere: BLOQUEAR_HASTA_EXPLICAR`.
- `analisis.exactitud.ALERTA_DEPENDENCIA_NO_REGISTRADA` (l. 410): «… o **número de autovalores nulos** distinto del registrado». La cantidad que se cuenta son **autovalores**.
- **Evidencia de diseño** (T-V4A l. 189–195 y 198–208): autovalores de `corr(P)` con `lam <= TAU * lam.max()`.
- **T-PROT l. 214–236** usa valores singulares, pero solo para comprobar algebraicamente el registro sobre columnas aleatorias con dependencias exactas, donde ambas convenciones coinciden. No define el control operativo.

### 1.2 Lo que falta (INV A1, A4)

- Qué matriz se usa para cada recuento y con qué normalización.
- Qué significa «P sin centrar», donde no existe matriz de correlación.
- La forma exacta de la desigualdad y el tratamiento de los autovalores negativos por redondeo.
- El ámbito: en qué celdas, poblaciones y paneles se aplica el control y su bloqueo.

### 1.3 Alternativas — matriz y normalización (A1)

| Alternativa | Definición | Ventajas | Riesgos y consecuencias |
|---|---|---|---|
| **A1-a** (correlación y cosenos) | Centrada: autovalores de la **correlación de Pearson** de la población de la celda (la misma matriz que usan R² y PCA). Sin centrar: autovalores de la matriz de **cosenos** `G_ij / (‖x_i‖·‖x_j‖)` | Invariante a la escala y a las unidades. Coincide con el diseño v4a y con la matriz de la PCA oficial (l. 418). Las 11 dependencias registradas se anulan exactamente | Es la convención más sensible a cuasidependencias. Con columnas normalizadas equivale a valores singulares con √τ = 1e-5 (DIAG). Puede revelar relaciones casi lineales del catálogo (N7, N18, N19) y activar A2 |
| A1-b (covarianza y productos cruzados brutos) | Autovalores de la covarianza (centrada) y de XᵀX (sin centrar) | Más cercana al rango «algebraico» de los datos | **Depende de la escala**: una columna independiente con escala relativa ≤ τ cuenta como nula (DIAG). Las columnas de P van de ~1e-4 (r_1m) a ~1e2 (CCI). Incoherente con la PCA sobre correlación |
| A1-c (convenciones conjuntas) | Exigir que coincidan autovalores y valores singulares (implementación actual) | Máxima cautela | **Sin apoyo textual**: añade una condición que el protocolo no tiene. Las dos convenciones difieren por un cuadrado, así que la discrepancia es estructural, no un error |

**Recomendación: A1-a.** Es la única que se apoya a la vez en el texto («autovalores») y en el diseño auditado (v4a), y es invariante a la escala. Los valores singulares quedan como **diagnóstico publicado y no bloqueante**.

**Aplicación exacta de τ (recomendada):**
- Autovalores con `eigh` simétrico en doble precisión.
- Un autovalor es nulo si `λ_i ≤ τ · λ_max`, con τ = 1e-10 y λ_max el mayor autovalor de esa matriz.
- Los autovalores negativos por redondeo cumplen la desigualdad y cuentan como nulos, **sin recortarlos antes de comparar**.
- El recuento se compara con `nulos_esperados` tal como están congelados.
- Se documenta el suelo de resolución: por debajo de ~1e-15·λ_max, los autovalores en doble precisión no distinguen intensidades (DIAG). Esto es documentación; τ no cambia.

### 1.4 Alternativas — ámbito (A4)

| Alternativa | Celdas en que el control **bloquea** | Ventajas | Riesgos |
|---|---|---|---|
| **A4-a** | GLOBAL, años evaluables 2020–2025 (incluido 2025 parcial) y los 7 segmentos (las celdas que usa la clasificación, l. 384–388, 425–426). Se calcula e informa, **sin bloquear**, en 2019 (NO_EVALUABLE, l. 547), en año×segmento (SOLO_DESCRIPTIVO, l. 389) y en el panel L por contrato (auxiliar, l. 567–570) | Bloquea exactamente donde se apoyan las conclusiones de TH35. No bloquea por celdas que el protocolo excluye de la inferencia | Una dependencia que solo apareciera en año×segmento se informaría sin bloquear |
| A4-b | Todas las celdas, incluidas año×segmento y 2019 | Máxima cobertura | Celdas pequeñas, con n cercano a p, dan rango deficiente sin significado. Bloquearía por celdas que el protocolo declara descriptivas o no evaluables |
| A4-c | Solo GLOBAL | Simple | No detecta dependencias específicas de un año o segmento, que sí entran en la clasificación |

**Recomendación: A4-a**, con dos precisiones:
- **Sensibilidades (S1, S3–S8).** El control se aplica a sus poblaciones y, si difiere, bloquea **solo la publicación de esa sensibilidad**, con el mismo procedimiento de A2.
- **Réplicas G2.** En los controles G2, la verificación que rige es V1 (identidades lineales con residuo ≤ 1e-9, l. 524). El recuento de nulos de las réplicas es **informativo**.

**Alcance del bloqueo.** Si el recuento difiere en una celda bloqueante, se bloquean **todas** las conclusiones de TH35 (§2.2), no solo las de esa celda. La clasificación combina todas las celdas, y «bloquear hasta explicar» no prevé bloqueos parciales.

**Alternativa A4-a′** (más permisiva): bloquear solo las conclusiones que dependan de la celda afectada. Sería defendible, pero introduce una lectura parcial que el texto no recoge.

### 1.4 bis Matrices degeneradas: matriz no definida frente a nulos distintos del registro (F5)

Son dos situaciones distintas que no deben confundirse:

| Situación | Cuándo ocurre | Qué significa |
|---|---|---|
| **Matriz NO_DEFINIDA** | Centrada: alguna columna tiene **varianza cero** en la población de la celda, de modo que su correlación no está definida (división por cero). Sin centrar: alguna columna tiene **norma cero** (todos sus valores son 0), de modo que su coseno no está definido | El control de nulos **no puede evaluarse**: no hay matriz. No es un recuento distinto del registro |
| **Matriz definida con nulos ≠ registro** | Todas las columnas tienen varianza (o norma) positiva y el recuento de A1-a difiere de `nulos_esperados` | Se aplica `BLOQUEAR_HASTA_EXPLICAR` (l. 374) con el procedimiento de A2 |

Una varianza **positiva pero diminuta** no hace que la matriz deje de estar definida: queda definida y la posible cuasidependencia entra en el segundo caso. No se usa ningún umbral nuevo de «varianza casi nula».

**Alternativas para la matriz NO_DEFINIDA:**

| Alternativa | Tratamiento | Consecuencias |
|---|---|---|
| **F5-a** | La celda queda **NO_DEFINIDA para el análisis matricial**: no se calculan R², PCA, agrupamiento ni recuento de nulos sobre la matriz de esa celda. Se informa de las columnas con varianza o norma cero. **Todas las columnas** cuentan esa celda como **incumplimiento** en la clasificación (regla de B13), sin cambiar umbrales. Si la celda NO_DEFINIDA es **GLOBAL**, se bloquean las conclusiones de TH35 con el procedimiento de A2, porque sin la celda global no hay PCA ni R² oficiales | Conservadora. No retira columnas ni altera p ni el registro. No confunde «no evaluable» con «discrepancia» |
| F5-b | Retirar de la matriz de esa celda la columna sin varianza y calcular sobre p − 1 columnas | Cambia p y deja sin sentido los nulos esperados (el registro se define sobre las 134 columnas). Requeriría un registro por subconjunto: **no recomendada** |
| F5-c | Tratarla como discrepancia (`BLOQUEAR_HASTA_EXPLICAR`) | Bloquea toda la etapa por un hecho distinto del que regula l. 374 |

**Recomendación: F5-a.**
- **Celdas informativas** (2019, año×segmento, L por contrato, réplicas G2): solo se informa NO_DEFINIDA.
- **Sensibilidades:** se aplica F5-a a la población de cada una (bloquea solo esa sensibilidad si es GLOBAL).
- **Motor actual:** se detiene ante una columna sin varianza (`cell_stats`), y la implementación de E02 lo sustituirá por este tratamiento explícito.

### 1.5 Claves E02 propuestas (A1, A4)

```yaml
identidades.verificacion_numerica.convencion:        # anadir
  cantidad: AUTOVALORES
  matrices:
    P_centrada: CORRELACION_PEARSON_POBLACION_CELDA
    B_centrada: CORRELACION_PEARSON_POBLACION_CELDA
    P_sin_centrar: COSENOS_SIN_CENTRAR
    L: CORRELACION_PEARSON_POR_CONTRATO
  desigualdad: "lambda_i <= tau_relativa * lambda_max (tau_relativa de esta misma seccion, sin cambios)"
  negativos_por_redondeo: CUENTAN_COMO_NULOS_SIN_RECORTE
  valores_singulares: DIAGNOSTICO_PUBLICADO_NO_BLOQUEANTE
  matriz_no_definida:                                # F5
    centrada: COLUMNA_CON_VARIANZA_CERO_EN_LA_POBLACION_DE_LA_CELDA
    sin_centrar: COLUMNA_CON_NORMA_CERO
    tratamiento: CELDA_NO_DEFINIDA_SIN_ANALISIS_MATRICIAL_E_INCUMPLIMIENTO_PARA_TODAS_LAS_COLUMNAS
    global_no_definida: BLOQUEO_TH35_CON_PROCEDIMIENTO_A2
    no_es: DISCREPANCIA_DE_NULOS
identidades.verificacion_numerica.ambito:             # anadir
  bloqueantes: [GLOBAL, ANIOS_EVALUABLES, SEGMENTOS]
  informativas: [ANIO_2019, ANIO_X_SEGMENTO, PANEL_L_POR_CONTRATO, REPLICAS_G2]
  sensibilidades: BLOQUEA_SOLO_LA_SENSIBILIDAD
  alcance_del_bloqueo: TODAS_LAS_CONCLUSIONES_TH35
```

**Tipo:** imprescindible (`BLOQ-EJEC`). No toca `tau_relativa` ni `nulos_esperados`.

---

## 2. A2 — Procedimiento diagnóstico y de desbloqueo

### 2.1 Lo que ya fija el texto congelado
- l. 374: `BLOQUEAR_HASTA_EXPLICAR`.
- l. 410: ALERTA por recuento distinto.
- l. 416: «identidad nueva solo como enmienda posterior al congelamiento, documentada y etiquetada».
- No existe continuación automática.

### 2.2 Qué se bloquea y qué se emite

**Conclusiones bloqueadas** mientras dure el bloqueo:
- la clasificación RECONSTRUIBLE / NUEVA_REPRESENTACION / INESTABLE;
- los subniveles;
- la etiqueta EXACTA;
- la transferencia;
- k_PA;
- las conclusiones de las sensibilidades afectadas.

**Diagnósticos que se emiten igualmente** (obligatorios; emitirlos no levanta el bloqueo):
1. El espectro completo de cada matriz afectada y su recuento, por las dos convenciones (A1-a normativa; valores singulares informativos).
2. Las direcciones nulas (autovectores con `λ ≤ τ·λ_max`), con todas sus cargas por columna, sin umbral de selección.
3. El R² de cada columna por los tres métodos (base de Cholesky, espectral y mínima norma) y su etiqueta de exactitud. Como R² y etiquetas son conclusiones sobre columnas, se publican **marcados como diagnóstico bloqueado**, no como resultado.
4. El residuo de las 11 + 3 identidades lineales registradas, y el valor numérico, en la celda, de las relaciones no lineales registradas N1–N21 que impliquen columnas con carga en las direcciones nulas sobrantes.
5. Las celdas, poblaciones y paneles afectados.

### 2.3 Alternativas de desbloqueo

| Alternativa | Instrumento | Ventajas | Riesgos y compatibilidad |
|---|---|---|---|
| **A2-a** | Enmienda formal posterior («E0x de explicación»), auditada y aprobada con el ciclo M2 (A, aprobación, P). Documenta las direcciones sobrantes, su atribución a relaciones N registradas (o «no atribuible»), las columnas implicadas y su tratamiento en la publicación | Plenamente compatible con l. 374 y l. 416. Trazable por Git y por huellas, en la misma arquitectura M2 | Coste: un ciclo de auditoría por cada bloqueo. Puede retrasar las conclusiones |
| A2-b | Acta de explicación aprobada por el responsable, sin enmienda | Más ligera | Un acta no tiene el control de integridad de una enmienda. Levantar un bloqueo cambia cómo se aplica una regla, lo cual es normativo |
| A2-c | Excepción predeclarada: lista cerrada de relaciones y criterio numérico; si se cumple, se publica automáticamente | Rápida | **Rechazada**: es un desbloqueo automático y contradice el requisito de no continuar sin decisión formal |

**Recomendación: A2-a.** E02 predeclara ahora los diagnósticos (§2.2), el contenido mínimo del expediente y las prohibiciones. El bloqueo solo se levanta con una enmienda posterior auditada.

**Contenido mínimo del expediente:**
- celdas y matrices;
- recuento observado y esperado;
- direcciones con cargas;
- relaciones N candidatas y su evidencia numérica;
- atribución por dirección;
- tratamiento propuesto para la publicación.

**Prohibiciones explícitas:**
- cambiar τ o `nulos_esperados`;
- registrar como «identidad lineal» lo que no es una identidad algebraica exacta;
- consultar o calcular resultados de clasificación TH35 antes del levantamiento;
- cualquier desbloqueo automático.

### 2.4 Distinción de efectos (INV A2, F3)

- **R² matemático.** El R² poblacional de una columna cuasidependiente está bien definido y es < 1. No depende de la convención de nulos.
- **Efectos numéricos.**
  - El R² oficial también usa τ = 1e-10 como umbral de pivote (l. 400). Con cuasidependencias del orden de τ, la base truncada puede alejar el R² calculado del matemático.
  - 1 − R² no se resuelve por debajo de ~1e-16.
  - Las verificaciones cruzadas pueden discrepar más de 1e-8.
- **Efectos de publicación.**
  - Si las verificaciones cruzadas discrepan, el R² es `R2_NO_PUBLICABLE_POR_COLINEALIDAD` (l. 412, 553).
  - Las etiquetas ALERTA y CASI se asignan sobre el R² calculado.
  - El recuento distinto bloquea las conclusiones (A2).
- **Sobre MNQ.** DIAG (sintético) muestra que el mecanismo es **posible** con relaciones del catálogo como N7, N18 y N19. **No** permite predecir si ocurrirá con MNQ.

### 2.5 Claves E02 propuestas (A2)

```yaml
identidades.verificacion_numerica.procedimiento_bloqueo:   # anadir
  estado: BLOQUEADO_HASTA_EXPLICAR
  conclusiones_bloqueadas: [CLASIFICACION_TH35, SUBNIVELES, ETIQUETA_EXACTA, TRANSFERENCIA, K_PA, SENSIBILIDAD_AFECTADA]
  diagnosticos_obligatorios: [ESPECTRO_DOS_CONVENCIONES, DIRECCIONES_NULAS_CON_CARGAS, R2_TRES_METODOS_MARCADO,
                              RESIDUOS_IDENTIDADES_REGISTRADAS, RELACIONES_N_IMPLICADAS, AMBITO_AFECTADO]
  expediente: [celdas, recuentos, direcciones, relaciones_candidatas, atribucion, tratamiento_propuesto]
  levantamiento: ENMIENDA_FORMAL_POSTERIOR_AUDITADA_CICLO_M2
  prohibido: [CAMBIAR_TAU, CAMBIAR_NULOS_ESPERADOS, REGISTRAR_CUASIDEPENDENCIA_COMO_IDENTIDAD_LINEAL,
              CONSULTAR_CLASIFICACION_ANTES_DEL_LEVANTAMIENTO, DESBLOQUEO_AUTOMATICO]
```

**Tipo:** imprescindible (`BLOQ-EJEC`). **Depende de:** A1 y A4.

---

## 3. B1 + B2 + B13 y dependencias (B3, B4, B6) — Intervalos, remuestreo y clasificación TH35

### 3.1 Lo que ya fija el texto congelado

- `analisis.bootstrap` (l. 423): bloques móviles circulares de **jornadas**, longitud 5, 300 réplicas, semilla 1717; sensibilidad con bloques de 1 y 10 (S4, l. 534).
- `analisis.clasificacion` (l. 424–433):
  - **RECONSTRUIBLE:** R²_TODAS ≥ 0,90 con «IC inferior» ≥ 0,90 en ≥ 5/6 años y ≥ 6/7 segmentos.
  - **NUEVA_REPRESENTACION:** R²_TODAS < 0,90 con «IC superior» < 0,90 con los mismos mínimos.
  - El resto es INESTABLE; nunca NUEVA_INFORMACION.
- `analisis.spearman` (l. 392–395): rangos fijos de la celda en el bootstrap; validación en las celdas global y «un año», con 30 réplicas exactas y tolerancia de 0,01 sobre la mediana de la diferencia de IC; alternativa de 100 réplicas exactas.
- D9, D21 y la regla de R² no publicable (l. 553).

### 3.2 B1 — Nivel, lateralidad y tipo del intervalo

**Bilateral y unilateral (F1).**
- Un **IC bilateral del 90 %**, [q₀,₀₅, q₀,₉₅], afirma con cobertura nominal del 90 % que el parámetro está entre ambos extremos.
- Cada extremo **usado por separado** es una cota unilateral con cobertura nominal del 95 % (q₀,₀₅ como cota inferior; q₀,₉₅ como cota superior).
- La regla de TH35 (l. 425–426) usa **un solo extremo en cada afirmación**: el inferior para RECONSTRUIBLE y el superior para NUEVA_REPRESENTACION. Por tanto, lo que fija la regla es el **nivel unilateral de cada extremo**:

| | Bilateral | Unilateral de cada extremo | Cuantiles |
|---|---|---|---|
| **B1-a** | 90 % | 95 % (α unilateral 0,05) | 0,05 y 0,95 |
| **B1-b** | 95 % | 97,5 % (α unilateral 0,025) | 0,025 y 0,975 |

**El nivel es una decisión normativa nueva.** El α = 0,05 de V3 (l. 527) y el máximo de falsas T3 (l. 439) regulan otros procedimientos (banda nula de G2-3b y tasa de falsos de T3). **No determinan** el nivel de TH35, así que no se usan como justificación.

**Comparación de 90 % y 95 %, sin datos de MNQ:**

| Criterio | B1-a (90 % bilateral) | B1-b (95 % bilateral) |
|---|---|---|
| Error nominal de cada afirmación por celda (declarar ≥ 0,90 cuando el valor verdadero es < 0,90, o al revés) | 0,05 | 0,025 |
| La clasificación exige ≥ 5/6 años y ≥ 6/7 segmentos | El error de la clasificación conjunta **no es necesariamente menor** que el de una celda, y **no se puede calcular de antemano**. Si una columna cumple de verdad la condición en algunas celdas y no en otras, basta con que unas pocas celdas den un falso positivo para alcanzar el mínimo exigido. El error depende de **cuántas celdas cumplen realmente la condición** y de la **dependencia estadística entre celdas**: años y segmentos comparten jornadas y estados de mercado | Igual |
| Efecto en la clasificación | Menos columnas INESTABLE | Más columnas INESTABLE: ambas condiciones son más difíciles de cumplir. Cuánto cambia depende de cuántas columnas tienen R² cerca de 0,90, algo que no puede conocerse sin datos |
| Réplicas en cada cola (B = 300, congelado) | ≈ 15 | ≈ 7,5 |

**Error Monte Carlo (corregido).**
- √(p(1−p)/B) es el error típico de una **proporción**: la fracción de réplicas por debajo de un valor fijo. Está en **unidades de probabilidad**: ≈ 0,013 para p = 0,05 y ≈ 0,009 para p = 0,025, con B = 300.
- **No** es el error del extremo del IC en unidades de R².
- El error del cuantil q̂ₚ en unidades de R² es aproximadamente √(p(1−p)/B) / f(qₚ), donde f es la densidad de la distribución bootstrap en ese cuantil. f no se conoce de antemano: depende de la dispersión del R² en cada celda.
- Lo único que puede afirmarse sin datos es que, con B fijo, el extremo de B1-b se apoya en la mitad de réplicas de cola que el de B1-a. Por eso es relativamente más variable.

**Cobertura.**
- La cobertura nominal (90 % o 95 %) **no es una garantía exacta**.
- El intervalo percentil del bootstrap de bloques móviles es una aproximación asintótica. Su cobertura real puede apartarse de la nominal por cuatro motivos:
  - dependencia temporal de más de 5 jornadas, que el bloque no conserva (D9, S4);
  - sesgo de muestra finita del R²;
  - la cota natural R² ≤ 1, cerca de la cual la distribución es asimétrica;
  - celdas con pocas jornadas.
- El informe debe declarar el nivel como **nominal**.

**Tipos de intervalo:**

| Alternativa | Definición | Ventajas | Riesgos |
|---|---|---|---|
| **Percentil** (con B1-a o B1-b) | Cuantiles empíricos de las 300 réplicas | Respeta el rango [0, 1] del R² y es invariante a transformaciones monótonas. Sencillo de auditar | No corrige sesgo; cobertura aproximada |
| BCa (B1-c) | Percentil corregido por sesgo y aceleración (jackknife de bloques) | Corrige sesgo y asimetría | Con B = 300 fijo, los cuantiles ajustados pueden caer en colas extremas mal estimadas. Menos establecido con bloques de jornadas |
| Studentizado (B1-d) | Bootstrap doble | Mejores propiedades teóricas | Necesita bootstrap anidado; no cabe en 300 réplicas sin cambiar el diseño congelado |

**Recomendación** (la decisión es del responsable; **las dos opciones de nivel siguen disponibles**):
- **Tipo:** percentil.
- **Nivel:** las dos opciones son defendibles.
  - **B1-a**: cuantiles 0,05/0,95 (unilateral 95 %). Se apoya en el doble de réplicas por cola con el B congelado.
  - **B1-b**: 0,025/0,975 (unilateral 97,5 %). Es más exigente contra falsos RECONSTRUIBLE y falsos NUEVA_REPRESENTACION, a costa de más INESTABLE.
  - La propuesta se inclina por **B1-a** por la estabilidad de las colas con B = 300, pero **no** por coherencia con V3 ni T3.
- El nivel y el tipo elegidos se aplican igual a R²_TODAS y a Spearman.
- Método de cuantil: empírico tipo 7 (interpolación lineal), declarado.

### 3.3 B2 — Unidad y ámbito del remuestreo por celda

| Alternativa | Definición | Ventajas | Riesgos |
|---|---|---|---|
| **B2-a** | Bootstrap **independiente por celda**. Unidad: la `trading_date` con al menos una fila en la población de la celda; en un segmento, la jornada es la parte de esa trading_date que cae en el segmento. Orden circular: trading_date cronológicas de la celda, consecutivas en esa lista (los huecos del calendario no rompen el bloque). Envolvimiento solo dentro de la celda. Réplica: ⌈D/5⌉ inicios uniformes, bloques de 5 jornadas, truncada a D. **Población fija**: casos completos de la celda, sin recalcularlos en la réplica | Los IC de cada celda solo dependen de sus datos. No hay envolvimiento entre años. Coincide con la estimación por celda de la clasificación. Cálculo exacto por estadísticos suficientes diarios (ya implementado y probado) | Los IC de celdas distintas se obtienen de forma independiente: no hay un remuestreo conjunto |
| B2-b | Un único remuestreo global de jornadas; cada celda es la restricción de la réplica global | IC conjuntos coherentes entre celdas | Las réplicas cruzan años al envolver. El número de jornadas de una celda varía entre réplicas, y una celda pequeña puede quedar casi vacía. Los IC dependen de la composición de otras celdas |

**Recomendación: B2-a.**

### 3.4 B6 (dependencia de B2) — Reproducibilidad del generador

**Recomendado**, siguiendo el modelo congelado de G2-2 (l. 467–468):
- `SeedSequence(1717).spawn(n_celdas)`, un flujo por celda, en el **orden declarado** de celdas: GLOBAL, años ascendentes, segmentos en el orden de `etiquetas`.
- `Generator(PCG64(flujo))`.
- En cada réplica, una sola llamada `integers(0, D, size=⌈D/5⌉)`.
- Las sensibilidades S4 (bloques 1 y 10) usan los mismos flujos por celda, reiniciados.
- Si cambia la versión de numpy, los IC se recalculan con el mismo procedimiento, como en l. 492. No se recalibra nada.

**Alternativa:** `default_rng(1717)` común con consumo secuencial. Es más simple, pero los IC de una celda dependerían del orden y del número de celdas procesadas antes.

### 3.5 B3 (dependencia) — Estadístico en cada réplica y réplicas inválidas (F2)

**Estadístico (sin cambios respecto de la versión anterior).**
- En cada réplica, R² con el **método oficial** (base de Cholesky con pivoteo, τ de l. 400) sobre la correlación de la réplica.
- Las verificaciones cruzadas y la publicabilidad se deciden sobre la **estimación puntual** de la celda.
- Siempre se usan **las 300 réplicas** de la celda. Si la estimación puntual es `R2_NO_PUBLICABLE`, el IC no se publica (B13).

**Réplica inválida.** Para una columna j en una celda, una réplica r es **INVALIDA** si ocurre cualquiera de estas cuatro cosas, cada una con su código:

| Código | Causa |
|---|---|
| `VAR_CERO` | Alguna columna que interviene (j o sus predictores) tiene varianza cero en la réplica, así que la correlación no está definida. Puede ocurrir si las jornadas remuestreadas no tienen variación en esa columna |
| `BASE_DEGENERADA` | La base de Cholesky de los predictores de j sale vacía, o el sistema sobre la base no tiene solución finita |
| `NO_FINITO` | R² NaN o ±∞ |
| `FUERA_DE_RANGO` | R² < 0 o R² > 1. Ambos son imposibles matemáticamente para el R² de MCO con intercepto, así que solo pueden deberse a problemas numéricos. **No se aplica ningún margen** |

Las columnas **registradas** (EXACTA) no se evalúan así: su R² es 1 por la identidad, y su IC se declara degenerado [1, 1] por registro (l. 428).

**Alternativas:**

| Alternativa | Tratamiento | Valoración |
|---|---|---|
| **F2-a** | Las 300 réplicas se conservan siempre. Si **al menos una** es INVALIDA para (columna, celda), el IC de esa columna en esa celda es **NO_CALCULABLE**. Se publica con el recuento de réplicas inválidas por código y sus índices. La celda cuenta como **incumplimiento** de ambas condiciones de la clasificación (B13). No se descarta, regenera, sustituye ni recorta ninguna réplica | Conservadora y reproducible. No introduce umbrales: la tolerancia es cero |
| F2-b | Las 300 réplicas se conservan; las inválidas se **ordenan como peores casos** explícitos (−∞ al calcular el extremo inferior, +∞ al calcular el superior) y el percentil se calcula sobre las 300 | Conservadora y aprovecha las réplicas válidas. Equivale a una imputación de valores: aunque sea explícita, es una regla de cálculo nueva que hay que aprobar |
| F2-c | Calcular el IC solo con las réplicas válidas | **No admisible**: descarta réplicas y cambia el B efectivo, con riesgo de sesgo de selección |
| F2-d | Regenerar las réplicas inválidas con nuevas extracciones | **No admisible**: cambia la secuencia reproducible y el número de réplicas |
| F2-e | Recortar el R² a [0, 1] | **No admisible**: oculta errores numéricos |

**Recomendación: F2-a.**
- La regla se aplica igual a los IC de Spearman: si en una réplica los rangos de una columna son constantes, la correlación no está definida, la réplica es INVALIDA y el IC es NO_CALCULABLE.
- Los recuentos de réplicas inválidas por celda, columna y causa se publican.

### 3.6 B4 (dependencia) — Spearman: validación frente a publicación (F3)

**Texto congelado:**
- `spearman.ic_bootstrap: RANGOS_DE_LA_CELDA_FIJOS` (l. 394);
- `spearman.validacion: {celdas: [global, un_anio], replicas_exactas: 30, tolerancia_mediana_dif_ic: 0.01, alternativa_si_falla: {replicas_exactas: 100}}` (l. 395);
- `bootstrap.replicas: 300` (l. 423).

D21: «IC con rangos fijos por celda, **validado** contra recálculo exacto».

**Hay tres conjuntos de réplicas distintos:**

| Conjunto | Para qué sirve | Réplicas |
|---|---|---|
| 1. Validación inicial | Comparar, en las celdas de validación, el IC con rangos fijos frente al IC con **re-ranking exacto** en cada réplica | Las 30 primeras del flujo de la celda |
| 2. Alternativa si falla | Lo que fija `alternativa_si_falla: {replicas_exactas: 100}` | 100 |
| 3. Bootstrap general | IC publicados (R² y Spearman) | 300 (l. 423) |

**Las dos lecturas posibles del conjunto 2:**

| Lectura | Contenido | Compatibilidad con el texto | Consecuencias computacionales |
|---|---|---|---|
| **F3-I (validación ampliada)** | Si la validación con 30 falla, se **repite la validación** con 100 réplicas exactas (las 100 primeras del flujo, mismo estadístico y tolerancia 0,01). El IC **publicado** sigue siendo el de rangos fijos con las **300** réplicas | **La más compatible.** `alternativa_si_falla` está **dentro** de `validacion` y es paralela a su `replicas_exactas: 30`: es un parámetro de la validación. El método publicado lo fija `ic_bootstrap` (rangos fijos) y el número de réplicas, `bootstrap.replicas` (300). Ninguno de los dos cambia | Validación: 30 + (si falla) 100 réplicas con re-ranking exacto en 2 celdas. Publicación: barata, por estadísticos suficientes diarios |
| F3-II (sustitución) | Si la validación falla, los IC **publicados** de todas las celdas se calculan con re-ranking exacto y 100 réplicas | Cambia a la vez el método publicado (frente a `ic_bootstrap`) y el número de réplicas (frente a 300) por una lectura del contexto que el texto no dice | Re-ranking de 134 columnas sobre las filas remuestreadas en cada réplica y celda. Con hasta ~1,9 M filas en GLOBAL, es varios órdenes de magnitud más costoso; los estadísticos diarios no sirven, porque los rangos dependen de toda la réplica. Las 300 réplicas de R² siguen igual: resultan B distintos para R² y Spearman |

**Qué falta en la lectura F3-I.** Hay que decidir qué ocurre si la validación ampliada **también** falla. El texto no lo dice. Opciones:
- **F3-I-a:** los IC de Spearman con rangos fijos se publican marcados **NO_VALIDADO**, sin cambiar el método.
- **F3-I-b:** no se publican los IC de Spearman. Se publican los coeficientes puntuales. Ni la clasificación TH35 (basada en R²) ni el resto de salidas se ven afectados.

**Recomendación: F3-I con F3-I-b.**
- Las 100 réplicas son **solo** una validación ampliada.
- El IC definitivo es siempre el de rangos fijos con 300 réplicas, si la validación (30 o 100) pasa.
- Si ambas fallan, los IC de Spearman no se publican y el informe lo declara.
- No cambia ningún valor normativo.

**Detalles de la validación** (sin cambios):
- celdas GLOBAL y ANIO_2024, elegido como último año evaluable completo por regla de calendario;
- mismas réplicas, prefijo del flujo de la celda, para los dos métodos;
- estadístico: mediana, sobre todos los pares y los dos extremos, de |IC_fijo − IC_exacto|, con el nivel de B1;
- criterio: ≤ 0,01 en ambas celdas;
- una réplica INVALIDA (F2) en la validación hace que la validación falle para ese par.

### 3.7 B13 — Celdas no publicables en la regla de clasificación

| Alternativa | Definición | Consecuencias |
|---|---|---|
| **B13-a** | Una celda cuyo R² no es publicable (l. 553), o cuyo IC no puede calcularse, **cuenta como incumplimiento** de ambas condiciones. No aporta evidencia ni de RECONSTRUIBLE ni de NUEVA_REPRESENTACION | Conserva **literalmente** los mínimos congelados («≥ 5 de 6», «≥ 6 de 7»). Desplaza hacia INESTABLE, la etiqueta conservadora. Se informa del número de celdas no publicables por columna |
| B13-b | Se descuenta del denominador (por ejemplo, 5 de 5) | **Modifica el umbral congelado**: convierte un mínimo absoluto en una proporción. No es admisible sin cambiar l. 431–432 |
| B13-c | La columna queda «NO_EVALUABLE» si hay alguna celda no publicable | Crea una etiqueta nueva no prevista en l. 425–427 |

**Recomendación: B13-a.** Es la única que no altera los umbrales ni las etiquetas congeladas.

**Consecuencia para TH35:** una columna con cuasidependencias que hagan no publicables varias celdas tiende a INESTABLE, nunca a una clasificación sin evidencia.

### 3.8 Claves E02 propuestas (B)

```yaml
analisis.bootstrap.intervalo:                       # anadir
  tipo: PERCENTIL
  cuantiles: <B1-a: [0.05, 0.95] | B1-b: [0.025, 0.975]>   # a decidir por el responsable
  interpretacion: "cada extremo se usa como cota unilateral en la regla de clasificacion; cobertura NOMINAL"
  metodo_cuantil: EMPIRICO_INTERPOLACION_LINEAL_TIPO_7
  aplica_a: [R2_TODAS, SPEARMAN]
analisis.bootstrap.ambito:                          # anadir
  por_celda: true
  unidad: TRADING_DATE_CON_FILAS_EN_LA_POBLACION_DE_LA_CELDA
  jornada_en_segmento: PARTE_DE_LA_TRADING_DATE_EN_EL_SEGMENTO
  orden_circular: CRONOLOGICO_DENTRO_DE_LA_CELDA
  poblacion_replica: FIJA_CASOS_COMPLETOS_DE_LA_CELDA
analisis.bootstrap.reproducibilidad:                # anadir (B6)
  bitgen: PCG64
  flujos: "SeedSequence(semilla).spawn(n_celdas), en el orden declarado de celdas"
  consumo: "una llamada integers(0, D, size=ceil(D / longitud_bloque_jornadas)) por replica"
  version_numpy: "si cambia la secuencia, se recalcula con el mismo procedimiento; nada se recalibra"
analisis.bootstrap.estadistico_replica:             # anadir (B3)
  r2: METODO_OFICIAL_SOBRE_LA_CORRELACION_DE_LA_REPLICA
  verificaciones_cruzadas: SOLO_ESTIMACION_PUNTUAL
  replicas_excluidas: NINGUNA
  diagnostico: TAMANO_DE_BASE_POR_REPLICA
  columnas_exactas: IC_DEGENERADO_PUBLICADO
  replica_invalida:                                 # F2
    causas: [VAR_CERO, BASE_DEGENERADA, NO_FINITO, FUERA_DE_RANGO]
    rango_admisible: "[0, 1] sin margen"
    tratamiento: IC_NO_CALCULABLE_SI_ALGUNA_INVALIDA   # F2-a; las 300 replicas se conservan
    prohibido: [DESCARTAR, REGENERAR, SUSTITUIR, RECORTAR]
    informe: RECUENTO_E_INDICES_POR_CELDA_COLUMNA_Y_CAUSA
    clasificacion: CUENTA_COMO_INCUMPLIMIENTO       # regla de B13
analisis.spearman.validacion_detalle:               # anadir (B4, F3)
  celdas: [GLOBAL, ANIO_2024]
  replicas_validacion: PRIMERAS_30_DEL_FLUJO_DE_LA_CELDA
  estadistico: "mediana sobre pares y extremos de |IC_rangos_fijos - IC_exacto|"
  alternativa_si_falla: "VALIDACION AMPLIADA con las primeras 100 replicas del flujo (mismo estadistico y tolerancia)"
  publicacion: "IC con rangos fijos y las 300 replicas de analisis.bootstrap si la validacion (30 o 100) pasa"
  si_falla_la_ampliada: IC_SPEARMAN_NO_PUBLICADOS_COEFICIENTES_SI   # F3-I-b
analisis.clasificacion.celdas_no_publicables:       # anadir (B13)
  incluye: [R2_NO_PUBLICABLE_POR_COLINEALIDAD, IC_NO_CALCULABLE, MATRIZ_NO_DEFINIDA]   # F2, F5
  tratamiento: CUENTA_COMO_INCUMPLIMIENTO_DE_AMBAS_CONDICIONES
  informe: RECUENTO_POR_COLUMNA
```

**Tipo:** B1, B2, B6, B3, B4 y B13 son imprescindibles para las salidas obligatorias (clasificación y tabla de Spearman).

---

## 4. D17 — Auditoría causal TH37

### 4.1 Fuentes y lo que falta

- **BACK l. 727–743:** TH37 es «OBLIGATORIA — bloqueante». Su método mínimo comprende la prueba de reconstrucción por transformación, la declaración CAUSAL o RETROSPECTIVO de cada salida, el origen de cada parámetro y la prueba de invariancia frente al factor de ajuste del roll. «Cualquier discrepancia… invalida los outputs afectados».
- **YAML:**
  - l. 381: series `CAUSAL_AT_BAR_CLOSE`, análisis `RETROSPECTIVO`;
  - l. 85: prohibidos los ajustes de roll y la información futura;
  - l. 595: tabla `TDA17_auditoria_causalidad.csv`, sin esquema.
- **Evidencia actual:** solo una prueba sintética (`tests/test_tda17_indicators.py` l. 568–582).
  - Compara con `pd.testing.assert_frame_equal` **con su tolerancia por defecto**: diferencias relativas por debajo de ~1e-5 se aceptan. Lo comprobé en este entorno: con pandas 3.0.5, 5e-10 se acepta y 5e-5 se rechaza.
  - Por tanto, **no es una prueba de igualdad exacta**: una fuga de información futura con peso pequeño, por ejemplo en la cola de una media exponencial, pasaría inadvertida.

### 4.2 Componente 1 — Información futura

| Alternativa | Definición | Ventajas | Riesgos |
|---|---|---|---|
| **D17-1a** | **Truncamiento y perturbación, por contrato.** Para cada punto de corte c de un contrato se hacen dos cálculos: (i) las series con los datos hasta c (truncamiento); (ii) las series con todas las barras posteriores a c perturbadas (precios y volumen alterados en la rejilla de tick, conservando las invariantes OHLC y el esqueleto). En ambos casos, las filas ≤ c deben ser **idénticas bit a bit** a las del cálculo completo, incluidas posiciones de NaN, marcas y máscaras | El truncamiento detecta dependencias del final del array; la perturbación detecta dependencias del contenido futuro aunque la longitud no cambie. Al reiniciarse por contrato (P1), basta recalcular el contrato, lo que hace el coste asumible | Coste: un recálculo por corte y por modo |
| D17-1b | Solo perturbación (como la prueba sintética actual) | Más barata | No detecta tratamientos especiales del final de los datos |
| D17-1c | Igualdad con tolerancia relativa (por ejemplo 1e-12) | Absorbe diferencias de redondeo por cambios de longitud | **Puede ocultar fugas de peso pequeño**. Además introduce un umbral nuevo |

**Recomendación: D17-1a con igualdad exacta.** Una discrepancia no se tolera: se investiga y, según BACK l. 740, invalida las salidas afectadas.

**Puntos de corte** (F4; predeclarados, todos sobre el esqueleto y nunca elegidos con resultados). Para cada contrato, el corte c se coloca en la barra **anterior**, en la **misma** y en la **siguiente** (c − 1, c, c + 1) a cada uno de estos eventos:

| Familia | Eventos (por contrato) |
|---|---|
| Discontinuidades | Primera y última FJ; primera y última FJL; primer y último HI corto; primer y último HI largo (> 90 min) |
| Reinicio y roll | Barras 1, 2 y 3 del contrato (reinicio P1) y la última barra del contrato |
| Calentamiento | Para cada K distinto del catálogo (K teóricos de l. 131–307): la **primera barra elegible** de cada clase (EXACTA, CONTADOR_CONS, VENTANA_VALIDEZ, JORNADAS), calculada sobre el esqueleto; incluye K_max = 1343 |
| Activación dependiente de los valores | Primera activación de KAMA (A5, er10 y er60) y de la salida adx de A8 (E01-A) |
| Cambio de estado recursivo | Primer y último cambio de dirección de Supertrend (A11, n10 y n60); primera vez que una ventana de VENTANA_VALIDEZ pierde y recupera la elegibilidad |
| Cuantiles | 25 %, 50 % y 75 % del contrato |
| Aleatorios | 10 barras uniformes entre la 2 y la penúltima, semilla **nueva** que debe aprobarse (se propone 3717) y `SeedSequence(3717).spawn(n_contratos)` |

- **Duplicados:** si dos reglas dan la misma barra, se usa una vez y se registran todos sus motivos.
- **Contrato demasiado corto o evento inexistente** (por ejemplo, un contrato sin FJL, o con menos barras que K_max + 1):
  - el corte **no se sustituye** por otro;
  - se registra como `NO_APLICABLE` con su motivo (`EVENTO_INEXISTENTE` o `CONTRATO_CORTO`);
  - no cuenta como discrepancia ni como conformidad;
  - el informe resume, por contrato, cuántos cortes de cada familia fueron aplicables.
  - Un corte c ≤ 1 o c ≥ número de barras del contrato no es aplicable.
- **D7 RVOL** cruza rolls: su truncamiento se hace sobre la serie **global** hasta c (no por contrato), y solo para RVOL, que es barato.
- **VWAP** se reinicia cada jornada: los cortes junto a FJ y FJL lo cubren.
- **Variables de TDA-04 que usa P:** se aplica la misma prueba con las funciones de TDA-04 sobre la serie canónica truncada.
- **Coste:** variable según los eventos; del orden de 100 a 200 recálculos de un contrato por contrato y modo.

**Alcance (F4).** Una batería finita de cortes **no es una prueba exhaustiva** de causalidad. Es una auditoría por muestreo, dirigida a los puntos donde una fuga es más probable (fronteras, calentamiento, activaciones y cambios de estado), y se complementa con:
- la comprobación estática de operaciones prohibidas (l. 85; `shift` negativo, `center=True`, `adjust=True`, …);
- la perturbación del futuro, que detecta dependencias en cualquier barra posterior al corte;
- el diseño recursivo causal del motor.

El riesgo residual (una fuga que solo se manifieste entre cortes no ensayados) se declara en el informe.

### 4.3 Componente 2 — Parámetros estimados retrospectivamente

**Recomendado:** una declaración, publicada en la tabla de causalidad, del **origen de cada parámetro**:

| Parámetro | Origen | Etiqueta | Uso |
|---|---|---|---|
| Parámetros del catálogo (n, f, s, g, k, m…) | Fijados a priori (D2, l. 131–307) | sin estimación | Series |
| Umbrales, τ, tolerancias, semillas | Protocolo congelado, antes de ver datos | sin estimación | Análisis |
| σ y curtosis recortada de G2-2 (l. 451–452) | TDA-07, muestra completa anterior al holdout | **RETROSPECTIVO** | Solo para generar el nulo G2-2, nunca series |
| Cortes de segmento (l. 386–388) | TDA-06, muestra completa | **RETROSPECTIVO** | Definen celdas, no series |
| Media por minuto × año de S5 (l. 535) | Población de análisis | **RETROSPECTIVO** (ya etiquetada) | Solo la sensibilidad S5 |
| R², PCA, bootstrap y clasificación | Población de análisis | **RETROSPECTIVO** (l. 381) | Análisis |

**Prueba asociada (estática):** ningún parámetro estimado entra en el cálculo de las series. Las únicas entradas de las series son el OHLCV y la máscara de validez de TDA-04. Se comprueba sobre el código del motor y la configuración efectiva.

### 4.4 Componente 3 — Ajustes de roll (3a; 3b fuera del núcleo)

- **3a (recomendada):** para cada contrato c, recalcular con los precios OHLC de **todos los demás contratos** transformados de tres formas: factor multiplicativo, desplazamiento aditivo y perturbación arbitraria, siempre en la rejilla y con las invariantes. Exigir que las series y marcas de c sean **idénticas bit a bit**.
- **Prueba complementaria de volumen:** perturbar el volumen de los demás contratos. Solo puede cambiar D7 RVOL, y solo en las filas con `rvol_cruza_roll = 1`. El resto, idéntico.
- Comprobar además que la entrada no contiene columnas ajustadas (l. 85) y que coincide con la del manifiesto (huellas).
- **3b (equivarianza ante reescalado o desplazamiento dentro de un contrato):** diagnóstico opcional (INV D17, OP). **No** entra en E02.

### 4.5 Componente 4 — Tabla y consecuencias

**Esquema propuesto de `TDA17_auditoria_causalidad.csv`:**

| Campo | Contenido |
|---|---|
| `prueba` | TRUNCAMIENTO, PERTURBACION_FUTURA, ROLL_PRECIO, ROLL_VOLUMEN u ORIGEN_PARAMETRO |
| `contrato`, `punto_corte` | — |
| `serie` | — |
| `filas_comparadas` | — |
| `discrepancias` | — |
| `max_abs_diff` | — |
| `resultado` | CONFORME, DISCREPANCIA o NO_APLICABLE |
| `motivo_corte` | eventos que originan el corte (§4.2) |
| `parametro`, `origen`, `etiqueta` | solo para ORIGEN_PARAMETRO |

**Consecuencia.** Cualquier DISCREPANCIA en una serie la deja fuera de las salidas oficiales: es «invalida los outputs afectados» (BACK l. 740). Además bloquea la publicación de las conclusiones de TH35 hasta corregirla y repetir la auditoría. **No hay tolerancia ni excepción automática.**

### 4.6 Claves E02 propuestas (D17)

```yaml
analisis.auditoria_causalidad:                       # anadir
  futuro:
    modos: [TRUNCAMIENTO, PERTURBACION_FUTURA]
    unidad: CONTRATO
    cortes_deterministas:                            # F4: en c-1, c y c+1 de cada evento, por contrato
      discontinuidades: [PRIMERA_Y_ULTIMA_FJ, PRIMERA_Y_ULTIMA_FJL, PRIMER_Y_ULTIMO_HI_CORTO, PRIMER_Y_ULTIMO_HI_LARGO]
      reinicio_y_roll: [BARRAS_1_2_3, ULTIMA_BARRA]
      calentamiento: PRIMERA_BARRA_ELEGIBLE_DE_CADA_K_Y_CLASE   # incluye K_max = 1343
      activacion: [A5_KAMA, A8_ADX]
      estado_recursivo: [A11_PRIMER_Y_ULTIMO_CAMBIO_DE_DIRECCION, VENTANA_VALIDEZ_PERDIDA_Y_RECUPERACION]
      cuantiles: [0.25, 0.50, 0.75]
    cortes_aleatorios: {por_contrato: 10, semilla: 3717, flujos: "SeedSequence(3717).spawn(n_contratos)"}   # semilla NUEVA: requiere aprobacion
    no_aplicable: {motivos: [EVENTO_INEXISTENTE, CONTRATO_CORTO], sustitucion: NINGUNA, cuenta_como: NI_DISCREPANCIA_NI_CONFORMIDAD}
    d7_rvol: TRUNCAMIENTO_GLOBAL
    alcance: AUDITORIA_POR_MUESTREO_NO_EXHAUSTIVA
    igualdad: EXACTA_BIT_A_BIT_INCLUIDOS_NAN_MARCAS_Y_MASCARAS
    incluye: [SERIES_CATALOGO, MARCAS, VARIABLES_TDA04_DE_P]
  origen_parametros: TABLA_DECLARADA                 # sec. 4.3
  roll:
    precio: {transformaciones: [MULTIPLICATIVA, ADITIVA, ARBITRARIA], otros_contratos: true, igualdad: EXACTA}
    volumen: {solo_puede_cambiar: D7, filas: rvol_cruza_roll}
    entrada_sin_ajustes: VERIFICAR
  tabla: {fichero: TDA17_auditoria_causalidad.csv, campos: [prueba, contrato, punto_corte, serie, filas_comparadas,
          discrepancias, max_abs_diff, resultado, parametro, origen, etiqueta]}
  consecuencia: SERIE_CON_DISCREPANCIA_FUERA_DE_SALIDAS_Y_BLOQUEO_TH35_HASTA_CORREGIR
```

**Tipo:** imprescindible (`BLOQ-EJEC`). La única cifra nueva es la semilla 3717, que debe aprobarse.

Los componentes **siguen separados**, con pruebas y filas propias en la tabla:
- 1, información futura;
- 2, origen de parámetros (`ORIGEN_PARAMETRO`);
- 3, ajuste de roll (`ROLL_PRECIO` y `ROLL_VOLUMEN`).

En la tabla se añade el campo `motivo_corte` (eventos que originan el corte) y el resultado `NO_APLICABLE`.

---

## 5. Decisiones condicionales (no imprescindibles para el núcleo)

| Id | Asunto | Recomendación | Condición para incluirla en E02 |
|---|---|---|---|
| A3 + B14 | Hueco espectral y publicación de EXACTA por celda (l. 402, 413, 554) | λ_(p−m)/λ_(p−m+1) de la matriz de A1-a, con m = nulos registrados. Si una celda no tiene separación suficiente, la etiqueta EXACTA no se publica en ella y el subnivel EXACTA global se mantiene por registro, con las celdas afectadas indicadas | Si se quiere publicar EXACTA por celda en la primera ejecución. Si no, solo se publica el registro global |
| B9 | Subniveles PARCIAL y LINEALMENTE_DISTINTA (l. 428–430) | Asignarlos con el R²_TODAS **puntual de la celda GLOBAL**: son descriptivos y no forman parte de la regla de estabilidad | Si se publican subniveles en la primera ejecución |
| A4-a′ | Bloqueo parcial por celda | No recomendada (§1.4) | Solo si el responsable prefiere un bloqueo parcial |

Los demás asuntos del inventario (B8, B10, B11, B12, C1–C3, D2–D16, …) **no** se tratan aquí: quedan para E02 o para una enmienda posterior, según el orden del inventario (§5.3). Mientras tanto, sus análisis no se publican.

---

## 6. Rutas que añadiría E02 (núcleo y condicionales) y comprobación de compatibilidad

| Ruta (`anadir`) | Decisión | ¿Toca una ruta prohibida (SPEC-M2 §7)? |
|---|---|---|
| `identidades.verificacion_numerica.convencion` | A1 | No: añade una clave nueva; no cambia `tau_relativa` ni `nulos_esperados` |
| `identidades.verificacion_numerica.ambito` | A4 | No |
| `identidades.verificacion_numerica.procedimiento_bloqueo` | A2 | No |
| `analisis.bootstrap.intervalo` | B1 | No: no cambia `replicas`, `longitud_bloque_jornadas` ni `semilla` |
| `analisis.bootstrap.ambito` | B2 | No |
| `analisis.bootstrap.reproducibilidad` | B6 | No |
| `analisis.bootstrap.estadistico_replica` | B3 | No |
| `analisis.spearman.validacion_detalle` | B4 | No: no cambia `validacion` |
| `analisis.clasificacion.celdas_no_publicables` | B13 | No: no cambia umbrales ni mínimos |
| `analisis.auditoria_causalidad` | D17 | No (clave nueva; introduce la semilla 3717, a aprobar) |
| `analisis.exactitud.hueco_espectral_definicion` | A3 + B14 (condicional) | No: no cambia `hueco_espectral_minimo` |
| `analisis.clasificacion.subniveles_r2` | B9 (condicional) | No |

---

## 7. Checklist de garantías para la implementación real de M2

Se añaden a SPEC-M2 §11, como requisitos de la implementación (no del prototipo):

1. **Verificación estricta no eludible.**
   - La ejecución oficial (`official_config_e02`) incorpora, **dentro** de su ruta y sin parámetro ni modo para omitirla, la verificación estricta oficial del protocolo: 9 ficheros, **5 entradas y 188 artefactos** (`tda17_protocol_freeze.verify_strict`) y `verify_against_git(9cdea2f)`.
   - Cualquier fallo es un rechazo.
   - **Prueba:** falta o se altera una entrada o un artefacto, y la ejecución oficial rechaza con su motivo exacto.
2. **Publicación remota de P2.**
   - Antes de autorizar MNQ, se comprueba que P2 **está publicado en el remoto**: `git fetch`, P2 alcanzable desde la rama remota publicada, y la historia remota contiene **A2 y P1** (P1 → A2 → P2).
   - Se registra el SHA remoto. Es un paso del procedimiento de autorización, verificable y registrado en el acta. No depende de la red durante el cálculo.
   - **Prueba:** con P2 solo local, o con un remoto que no contiene A2, se rechaza.
3. **Solo configuración oficial para resultados oficiales.**
   - Toda ruta que escriba resultados oficiales (tablas, informe, figuras, G2 sobre réplicas, sensibilidades) exige `require_e02_oficial(cfg)`: `ejecucion_oficial: true` y `publicacion_E02` = P2.
   - La aprobación preparatoria nunca basta.
   - **Prueba estática:** toda función de escritura llama a `require_e02_oficial`.
   - **Prueba dinámica:** con la configuración preparatoria se rechaza.
4. **Sin mecanismos de omisión.**
   - La implementación real no contiene `_DISABLED` ni ningún otro mecanismo (variable de entorno, parámetro, bandera, modo de prueba en producción) que omita o desactive controles.
   - **Prueba estática:** se buscan en el código esos identificadores y patrones.
   - La omisión de controles se demuestra **solo** con mutaciones de código fuente sobre clones Git (T-MUT).

---

## 8. Matriz final de decisiones (requiere aprobación explícita del responsable)

Marcar una opción por fila. «Otra» exige especificar la alternativa elegida y su justificación.

| # | Decisión | Recomendación | Alternativas | Claves E02 | Tipo | Depende de | Aprobación |
|---|---|---|---|---|---|---|---|
| 1 | A1 — Cantidad, matriz, normalización y aplicación de τ; matriz degenerada | **A1-a**: autovalores de la correlación de Pearson (centrada) y de los cosenos (sin centrar); `λ ≤ τ·λ_max`; negativos cuentan como nulos; valores singulares solo informativos. **F5-a**: con varianza o norma cero la matriz es NO_DEFINIDA (no es una discrepancia); la celda cuenta como incumplimiento para todas las columnas; si es GLOBAL, se bloquea TH35 con A2 | A1-b, A1-c; F5-b, F5-c | `identidades.verificacion_numerica.convencion` | Imprescindible | — | ☐ Aprobada ☐ Otra: ___ ☐ Rechazada |
| 2 | A4 — Ámbito del control y alcance del bloqueo | **A4-a**: bloquea en GLOBAL, años evaluables y segmentos; informativo en 2019, año×segmento, L y G2; las sensibilidades solo se bloquean a sí mismas; el bloqueo afecta a todas las conclusiones de TH35 | A4-b, A4-c, A4-a′ | `…verificacion_numerica.ambito` | Imprescindible | 1 | ☐ Aprobada ☐ Otra: ___ ☐ Rechazada |
| 3 | A2 — Diagnóstico y desbloqueo | **A2-a**: diagnósticos obligatorios; expediente; levantamiento solo por enmienda formal auditada (ciclo M2); prohibiciones explícitas | A2-b; A2-c (no recomendada) | `…verificacion_numerica.procedimiento_bloqueo` | Imprescindible | 1, 2 | ☐ Aprobada ☐ Otra: ___ ☐ Rechazada |
| 4 | B1 — Intervalo | Percentil (tipo 7) para R²_TODAS y Spearman; **nivel a elegir**: ☐ **B1-a** 0,05/0,95 (bilateral 90 %, unilateral 95 %; la propuesta se inclina por esta) ☐ **B1-b** 0,025/0,975 (bilateral 95 %, unilateral 97,5 %). Cobertura nominal, no exacta | B1-c (BCa), B1-d | `analisis.bootstrap.intervalo` | Imprescindible | — | ☐ Aprobada ☐ Otra: ___ ☐ Rechazada |
| 5 | B2 — Remuestreo | **B2-a**: por celda; unidad trading_date; segmentos por fracción; circular dentro de la celda; población fija | B2-b | `analisis.bootstrap.ambito` | Imprescindible | 4 | ☐ Aprobada ☐ Otra: ___ ☐ Rechazada |
| 6 | B6 — Reproducibilidad | PCG64; `SeedSequence(1717).spawn` por celda en orden declarado; S4 con los mismos flujos | `default_rng(1717)` común | `analisis.bootstrap.reproducibilidad` | Imprescindible | 5 | ☐ Aprobada ☐ Otra: ___ ☐ Rechazada |
| 7 | B3 — Estadístico de réplica y réplicas inválidas | **B3-a**: método oficial en cada réplica; 300 réplicas siempre; verificaciones solo en la estimación puntual; IC de las columnas EXACTA degenerado. **F2-a**: con alguna réplica INVALIDA (VAR_CERO, BASE_DEGENERADA, NO_FINITO, FUERA_DE_RANGO), el IC es NO_CALCULABLE y la celda cuenta como incumplimiento; nunca se descarta, regenera, sustituye ni recorta | B3-b, B3-c; F2-b (peores casos explícitos); F2-c, F2-d y F2-e no admisibles | `analisis.bootstrap.estadistico_replica` | Imprescindible | 3, 4 | ☐ Aprobada ☐ Otra: ___ ☐ Rechazada |
| 8 | B4 — Spearman: validación y publicación | **F3-I**: las 100 réplicas exactas son solo una validación ampliada; el IC publicado es el de rangos fijos con 300 réplicas si la validación (30 o 100) pasa. **F3-I-b**: si ambas fallan, no se publican los IC de Spearman (sí los coeficientes). Celdas GLOBAL y ANIO_2024 | F3-I-a (publicar marcados NO_VALIDADO); F3-II (sustituir por 100 exactas); año por sorteo | `analisis.spearman.validacion_detalle` | Imprescindible (tabla de Spearman) | 4, 5, 6 | ☐ Aprobada ☐ Otra: ___ ☐ Rechazada |
| 9 | B13 — Celdas no publicables | **B13-a**: cuentan como incumplimiento de ambas condiciones; recuento informado | B13-b (cambia el umbral: no admisible), B13-c | `analisis.clasificacion.celdas_no_publicables` | Imprescindible | 4 | ☐ Aprobada ☐ Otra: ___ ☐ Rechazada |
| 10 | D17 — Auditoría causal TH37 | **D17-1a**: truncamiento y perturbación, igualdad exacta. Cortes deterministas en c − 1, c y c + 1 de FJ, FJL, HI corto y largo, reinicio y roll, primera barra elegible por K y clase, activación de KAMA y adx, y cambios de estado de Supertrend y de VENTANA_VALIDEZ; más cuantiles y 10 aleatorios (**semilla nueva 3717**). `NO_APLICABLE` sin sustitución si el contrato es corto o falta el evento. D7 con truncamiento global. Auditoría por muestreo, no exhaustiva. Origen de parámetros y roll 3a como pruebas separadas | D17-1b, D17-1c | `analisis.auditoria_causalidad` | Imprescindible | — | ☐ Aprobada ☐ Otra: ___ ☐ Rechazada |
| 11 | A3 + B14 — Hueco espectral y EXACTA por celda | λ_(p−m)/λ_(p−m+1) de la matriz de A1-a | Menor por encima de τ / mayor por debajo; por identidad | `analisis.exactitud.hueco_espectral_definicion` | Condicional | 1 | ☐ Incluir ☐ Aplazar ☐ Otra: ___ |
| 12 | B9 — Subniveles | R²_TODAS puntual de la celda GLOBAL | Mediana de celdas; IC | `analisis.clasificacion.subniveles_r2` | Condicional | 4, 9 | ☐ Incluir ☐ Aplazar ☐ Otra: ___ |
| 13 | Checklist M2 (§7) | Incorporar las 4 garantías a SPEC-M2 §11 | — | (implementación, no YAML) | Imprescindible para autorizar MNQ | — | ☐ Aprobada ☐ Otra: ___ |

**Después de la aprobación** (fuera de esta tarea):
1. redactar el YAML de E02 solo con las rutas aprobadas;
2. implementar M2 conforme a SPEC-M2 y al checklist;
3. hacer el ciclo A2 → P2 con auditoría independiente.

Nada de esto autoriza ejecutar MNQ.

---

## 9. Relación de cambios de la revisión 1

| Punto | Sección | Cambio |
|---|---|---|
| F1 | §3.2, §3.8, matriz fila 4 | Se distingue el IC bilateral del 90 % de su uso unilateral (95 % por extremo) y se compara con 95 % bilateral (97,5 % por extremo), sin datos. Se retira el argumento de coherencia con V3/T3, que regulan otros procedimientos. Se corrige √(p(1−p)/B): es el error típico de una proporción, en unidades de probabilidad, no del extremo en unidades de R²; el error en R² depende de la densidad bootstrap, desconocida a priori. La cobertura del percentil se declara **nominal y aproximada**. Las dos opciones de nivel quedan abiertas en la matriz y en la clave `cuantiles` |
| F2 | §3.5, §3.8, matriz fila 7 | Definición de réplica INVALIDA (VAR_CERO, BASE_DEGENERADA, NO_FINITO, FUERA_DE_RANGO sin margen). Tratamiento recomendado F2-a: se conservan las 300 réplicas; si alguna es inválida, el IC es NO_CALCULABLE y la celda cuenta como incumplimiento (B13). F2-b queda como alternativa; descartar, regenerar, sustituir o recortar, como no admisibles. Se aplica también a Spearman |
| F3 | §3.6, §3.8, matriz fila 8 | Se separan los tres conjuntos de réplicas (validación con 30, alternativa con 100, bootstrap con 300). Lectura más compatible (F3-I): las 100 son solo una validación ampliada y el IC publicado sigue siendo el de rangos fijos con 300. Se exponen sus consecuencias computacionales frente a F3-II. Si ambas validaciones fallan, se recomienda no publicar los IC de Spearman (F3-I-b). Ningún valor normativo cambia |
| F4 | §4.2, §4.5, §4.6, matriz fila 10 | Cortes deterministas en c − 1, c y c + 1 de FJ, FJL, HI corto y largo, reinicio y roll, calentamiento por K y clase, activaciones (KAMA, adx) y cambios de estado recursivo (Supertrend, VENTANA_VALIDEZ). Regla `NO_APLICABLE` sin sustitución para contratos cortos o eventos inexistentes. Truncamiento global de D7. Se declara que es una auditoría por muestreo, no exhaustiva. Origen de parámetros y roll siguen siendo pruebas separadas. La tabla gana `motivo_corte` y `NO_APLICABLE` |
| F5 | §1.4 bis, §1.5, §3.8, matriz fila 1 | Se distingue la matriz NO_DEFINIDA (varianza o norma cero) de la matriz definida con nulos distintos del registro. Tratamiento recomendado F5-a: la celda queda sin análisis matricial y cuenta como incumplimiento para todas las columnas (regla de B13, sin umbrales nuevos); si es GLOBAL, se bloquea TH35 con A2. F5-b y F5-c quedan como alternativas no recomendadas. `celdas_no_publicables` incluye ahora IC_NO_CALCULABLE y MATRIZ_NO_DEFINIDA |

**Corrección de cierre (§3.2):** se elimina la afirmación de que el error de la clasificación conjunta es necesariamente menor que el de una celda; depende de cuántas celdas cumplen realmente la condición y de su dependencia. Sin cambios en niveles, umbrales ni recomendaciones.

**Corrección menor:** la cabecera remitía a la matriz como §7; es §8.

**Sin cambios:**
- las 13 filas de la matriz (se mantienen, con las filas 1, 4, 7, 8 y 10 actualizadas);
- A2, B2, B6, B13 (salvo la ampliación de `incluye`), las decisiones condicionales y el checklist M2;
- los valores congelados (τ, nulos esperados, umbrales, 300 réplicas, bloques de 5 jornadas, semilla 1717, tolerancia de Spearman 0,01, 30 y 100 réplicas exactas).

Ninguna decisión queda aprobada.
