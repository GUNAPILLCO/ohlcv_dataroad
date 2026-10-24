# TDA-17 — Inventario de decisiones metodológicas pendientes (revisión 3.1)

**Estado:** `PARA_AUDITORIA_Y_DECISION`. Documento de auditoría; **no es normativo**. No modifica el protocolo congelado (`9cdea2f`), E01 (aprobada y publicada en `24f9f1d`), parámetros, umbrales, hipótesis ni D1–D24. No se ha ejecutado nada sobre MNQ ni se ha leído el holdout. No crea ni aprueba E02.
**Revisión 3.1:** solo cambia §2 (migración), por las correcciones F4–F6 de la demostración M2:
- HEAD debe descender de A2;
- se separan la aprobación preparatoria y la ejecución oficial vinculada a P2;
- las negativas exigen tipo y motivo exactos;
- el código de salida depende también de las positivas;
- las pruebas `_DISABLED` se denominan omisiones instrumentadas.

Las fichas de decisión (§3–§5) y sus recuentos no cambian.
**Revisión 3:** M2 aprobada como arquitectura de referencia de E02 (sin autorizar E02). Incorpora F1 (§2, evidencia reproducible de la demostración v2), F2 (especificación de M2 en `reports/mnq/TDA17_especificacion_migracion_M2.md`) y F3 (D17 punto 3, B6 y A2 corregidos; B6 y D15 reclasificados como `AC`).
**Revisión 2:** incorporó las correcciones C1–C5 de la auditoría de la revisión 1:
- §2: migración E01 → E02, demostrada en clones desechables;
- A1 y A2 reformulados;
- B13 dividido en B13 y B14;
- D9 y D17 revisados;
- §5: reclasificación en cuatro categorías y conjunto mínimo de E02.

**DECISIÓN IRIS: NINGUNA.**

## 0. Fuentes y convenciones

| Abrev. | Fichero | Estado |
|---|---|---|
| **YAML** | `configs/tda17.yaml` | protocolo CONGELADO, SHA-256 `920cda43…` (manifiesto `1b14b65a…`) |
| **INF** | `reports/mnq/TDA17_protocolo_congelamiento.md` | parte del protocolo (resume; prevalece el YAML) |
| **E01** | `configs/tda17_enmienda_01.yaml` | APROBADA (`62b91747…`); commit auditado A1 = `386815b`, publicación P1 = `24f9f1d` |
| **REG** | `reports/mnq/TDA17_enmiendas.json` | registro de E01; huella canónica de la entrada E01: `3150ea3a5e6fda45…` |
| **ENM** | `src/ohlcv_dataroad/ingest/tda17_enmiendas.py` | cargador de E01, auditado (SHA-256 `6c5d96d0…`) |
| **T-PROT**, **T-V4A**, **T-V4B**, **T-G22**, **T-CAL**, **T-ST** | pruebas normativas y de diseño en `tests/test_tda17_*.py` | protegidas |
| **ROAD** | `docs/methodology/Tsay_OHLCV_analysis_roadmap.md` §TDA-17 (l. 1086–1119) | marco de la etapa |
| **BACK** | `docs/methodology/Tsay_empirical_hypotheses_backlog.md` TH35 (l. 691–707), TH37 (l. 727–743) | hipótesis |
| **AN**, **G2** | `src/ohlcv_dataroad/ingest/tda17_analysis.py`, `tda17_g2.py` | fase 2 auditada (C1–C4), sin aprobación formal |
| **DIAG** | `tests/test_tda17_rango_diagnostico.py` y su informe (fase 2) | diagnóstico **sintético** de convenciones de rango |
| **MIG** | demostración v3 en clones desechables: `mig_demo.py` (SHA-256 `027d8da5…`), `mig_demo_inyeccion.py` (`9f3e75c5…`), plantillas `m2_loader_template.py` v3 (`c70cb36d…`), v2 para la regresión (`07e782d5…`) y `m1_chain_template.py` (`55b1e075…`). Registros íntegros: `mig_demo_v3_run1.log` (`dd4cb207…`), `mig_demo_v3_run2.log` (`bf1ee8a8…`) y `mig_demo_v3_inyeccion.log` (`8492d871…`) | fuera del repositorio; reproducible (§2.6) |
| **SPEC-M2** | `reports/mnq/TDA17_especificacion_migracion_M2.md` | especificación técnica de M2 (F2), para auditoría |

**Referencias.** Las del YAML se dan como ruta de claves y líneas del fichero congelado. Las del resto, como fichero:línea.

**Bloqueo:**
- `BLOQ-EJEC`: impide ejecutar TDA-17 sobre MNQ.
- `BLOQ-ANAL`: impide solo el análisis o la publicación indicados.
- `INFORME`: necesario solo para el informe final.

**Categoría (C5):**
- `N`: decisión normativa nueva imprescindible.
- `AC`: aclaración normativa que puede agruparse con otra.
- `ID`: implementación ya determinada por las fuentes (sin enmienda).
- `OP`: ampliación opcional, que **no** debe entrar automáticamente en E02.

**Advertencia sobre los resultados sintéticos.** Las cifras de DIAG y de las pruebas de la fase 2 proceden de generadores sintéticos que no reproducen la distribución de MNQ. Muestran que un mecanismo es **posible** y cómo funciona; **no** predicen qué ocurrirá con MNQ.

---

## 1. Resumen

| Bloqueo \\ Categoría | N | AC | ID | OP | Total |
|---|---|---|---|---|---|
| BLOQ-EJEC | 3 (A2, B1, D17) | 3 (A1, A4, B2) | 0 | 0 | 6 |
| BLOQ-ANAL | 4 (B11, B12, C1, C4) | 17 | 1 (D14) | 0 | 22 |
| INFORME | 0 | 6 (B6, D8, D10, D15, D18, D19) | 7 | 3 (B5, D1, D16) | 16 |
| **Total** | **7** | **26** | **8** | **3** | **44** |

Son 44 fichas porque B13 se ha dividido en B13 y B14. C4 es una decisión de gobierno (el procedimiento de migración), no estadística, pero es imprescindible antes de redactar E02.

Conclusiones:

1. **La migración debe decidirse antes que E02 (§2).** El mecanismo actual rechaza cualquier E02 añadida de forma ingenua. «Repetir A→P» re-aprobando E01 reescribe su entrada publicada y deja de citar A1 (MIG M0.d), así que destruye la trazabilidad. Hay dos alternativas demostradas que verifican ambas aprobaciones sin tocar la de E01:
   - **M2, cargador escalonado** (aprobada como arquitectura de referencia). El cargador de E01 y sus 8 ficheros quedan intactos. Pasan 303 de 303 pruebas auditadas de E01 y 105 de 105 de la fase 2 (obligatorias, código 0). En M2 hay 40 negativas con tipo y motivo exactos y 37 omisiones instrumentadas, todas detectadas. Desde F4 se exige además que HEAD descienda de A2, y la ejecución oficial queda vinculada a P2 (SPEC-M2 §6.1).
   - **M1, aprobaciones encadenadas.** Con la cadena como vía de producción real, 205 de 303 pruebas auditadas de E01 fallan o dan error, porque su infraestructura escribe aprobaciones sintéticas de E01. Habría que reescribir pruebas y ficheros auditados de E01.
2. **Rango.** El texto congelado respalda contar **autovalores** (A1). Siguen indefinidos la matriz, la normalización, «P sin centrar», la forma exacta de aplicar τ y el ámbito (A4).
3. **A2.** Cuando el recuento de nulos difiere, el protocolo bloquea las **conclusiones**. Ninguna alternativa propuesta continúa automáticamente. Lo que falta predeclarar es qué **diagnósticos** se emiten durante el bloqueo y cómo se documenta una explicación.
4. **Núcleo mínimo de E02 (§5):** rango (A1, A4, A2, A3), inferencia de la clasificación (B1, B2, B13, B3, B4) y TH37 (D17), más la decisión de migración (C4). Es lo imprescindible para la salida obligatoria de TH35 y para TH37. El resto puede entrar en E02 o en una enmienda posterior, y los análisis afectados no se publican mientras tanto.

---

## 2. Migración E01 → E02 (C1)

### 2.1 Qué impide hoy añadir E02 (demostrado: MIG, M0)

El cargador de E01 (ENM) liga la aprobación de E01 a un conjunto auditado **global**:
- `AUDITED_FILES = [YAML de cada enmienda autorizada] + IMPLEMENTATION_FILES` (ENM l. 101–110);
- `check_approval` exige que `sha256_auditados` cubra **exactamente** ese conjunto y que cada fichero tenga, en el árbol, la huella aprobada (ENM l. 212–247);
- `require_audited_commit` exige que A1 contenga esos ficheros con esas huellas (ENM l. 261–300).

Cada paso, en un clon de P1 (MIG v2; las negativas M0.a–c exigen `ProtocolNotAuthorizedError` con el motivo citado):

| Caso | Resultado del cargador de E01 |
|---|---|
| Control (clon en P1) | carga; E01 verificada contra A1 `386815b` |
| M0.a E02 solo en el registro | **RECHAZO**: «enmiendas no autorizadas en el registro: ['E02']» |
| M0.b solo un comentario en ENM | **RECHAZO**: «E01: … ficheros distintos de la version auditada: [tda17_enmiendas.py]» |
| M0.c E02 en `AUTHORIZED_AMENDMENTS`, `WHITELIST` y `REQUIRED_AMENDMENTS` | **RECHAZO**: «E01: … sha256_auditados no cubre exactamente los ficheros auditados» (`AUDITED_FILES` pasa a incluir el YAML de E02) |
| M0.d «repetir A→P»: re-aprobar E01 y E02 sobre un commit nuevo A' (`c19a94c`) | carga, pero **E01 deja de citar A1** y su entrada publicada cambia (huella canónica distinta) |

**Conclusión.** Con el mecanismo actual, cualquier E02 invalida la aprobación de E01 o exige reescribirla. M0.d no es aceptable: la auditoría de E01 (A1, acta, evidencias y huellas publicadas) dejaría de ser lo que verifica producción.

### 2.2 Requisitos comunes a cualquier migración aceptable

- **R1.** La entrada E01 del registro y el YAML de E01 no cambian: huella canónica `3150ea3a…` y `62b91747…`.
- **R2.** La aprobación de E01 se sigue verificando contra **A1** (`386815b`), con sus huellas originales.
- **R3.** E02 se aprueba de forma independiente: commit auditado A2, publicación P2, acta propia y registro con estado propio.
- **R4.** La historia se conserva: P1 es antecesor de A2 y A2 de P2. Un commit con el mismo árbol pero sin la historia de E01 se rechaza.
- **R5.** El ancla de E01 está en **código auditado**: la huella de su entrada (y, en M2, la del cargador de E01) se fija como constante dentro de un fichero auditado en A2. Así, reescribir la aprobación de E01 de forma coherente no pasa desapercibido. El registro sigue siendo una declaración, no una firma.
- **R6.** Los 9 ficheros del protocolo y el manifiesto no cambian (`verificar --referencia-git 9cdea2f` CONFORME).
- **R7.** Los módulos de la fase 2 (`tda17_analysis.py`, `tda17_g2.py` y sus pruebas), que hoy no tienen aprobación formal, entran en el conjunto auditado de E02, porque implementan sus decisiones.

### 2.3 Alternativa M2 — Cargador escalonado (E01 intacto)

**Idea.** El cargador de E01 no se toca. Un módulo nuevo (`tda17_enmiendas_e02.py`, nombre ilustrativo) hace lo siguiente:
1. comprueba que ENM es exactamente el auditado (`6c5d96d0…`), que la entrada E01 es la publicada (`3150ea3a…`) y que P1 es antecesor;
2. llama al cargador de E01 sin modificar, que verifica E01 contra A1 como hoy;
3. verifica E02 contra **su** registro y **su** commit A2;
4. aplica la lista blanca de E02 sobre la configuración efectiva de E01, con la misma barrera de rutas cambiadas.

**Ficheros:**

| Ficheros | En M2 |
|---|---|
| 9 del protocolo, manifiesto | protegidos (sin cambios) |
| YAML de E01, REG (registro de E01), ENM, `tda17_indicators.py`, `tda17_primitives.py`, `tests/test_tda17_indicators.py`, `tests/test_tda17_enmienda_01.py`, `tests/tda17_synthetic.py` | **byte a byte iguales a P1**; los verifica el cargador de E01, como hoy |
| `configs/tda17_enmienda_02.yaml`, `tda17_enmiendas_e02.py`, `reports/mnq/TDA17_enmiendas_E02.json` (registro **propio**), `tests/test_tda17_enmienda_02.py`, módulos y pruebas de la fase 2, acta de E02 | nuevos; los auditados (todo salvo registro y acta) llevan sus huellas en la aprobación de E02 |

**Commits y registros:**
- **A2:** los ficheros nuevos, con el registro de E02 en `PENDIENTE_AUDITORIA` y `requiere: {E01: 3150ea3a…, publicacion_E01: 24f9f1d…}`.
- **Aprobación de E02:** cita A2 y las huellas de su conjunto auditado.
- **P2:** el registro de E02 aprobado y el acta.
- Verificación final con `verificar --referencia-git P2`: la del cargador de E01 (sigue CONFORME porque sus ficheros no cambian) y una equivalente para E02.

**Evidencia (MIG v3; detalle completo en SPEC-M2 §12):**
- **Commits deterministas:** A2 `c1e9b74` y P2 `e196e4f`.
- **Pruebas obligatorias:** las 303 pruebas auditadas de E01 (código 0, exactamente 303 passed) y las 105 de la fase 2 (código 0).
- **Positivas:** 17 con valor exacto. Incluyen la aprobación preparatoria (`ejecucion_oficial` false) y la ejecución oficial con HEAD = P2 o descendiente de P2 (`ejecucion_oficial` true, `publicacion_E02` = P2).
- **Negativas:** 40 en M2, con tipo **y motivo completo** exactos. Incluyen las de la ejecución oficial (`E02_P2_*`) y `E02_A2_HEAD`.
- **Omisiones instrumentadas (`_DISABLED`):** 37/37 detectadas; 12 las detecta otra barrera. **No son mutaciones de código fuente**: estas quedan a cargo de la implementación real (T-MUT).
- **Regresión F4.** La plantilla v2 **acepta** un HEAD que desciende de P1 pero no de A2 (con A2 presente y huellas válidas). La v3 lo rechaza con `[E02_A2_HEAD]`.
- **F5.** Una prueba positiva fallida inyectada, o un recuento distinto del exigido, hace terminar el guion con código 1.
- **Requisito de entorno.** En Windows, `git show` falla con rutas por encima de MAX_PATH. Los cargadores fallan de forma segura (rechazan), así que la ejecución oficial debe hacerse desde una ruta corta o con `core.longpaths`.

**Pruebas que deberá superar la implementación real:** SPEC-M2 §11 (grupos T-E01, T-POS, T-OFI, T-NEG, T-CMP, T-MUT, T-INT, T-WL, T-GIT-P2, T-REP y T-FASE2).

**Límites:**
- E02 **no puede** modificar ningún fichero auditado de E01, incluido el motor de indicadores. Ninguna decisión del inventario lo exige: todas afectan al análisis, a G2, a las sensibilidades o a las salidas. S2 se desbloquea porque `require_s2_scale_assignment` (en ENM) lee la clave `escalas_S2` de la configuración que recibe.
- Una E03 exigiría otra capa o pasar a M1.
- Hay dos registros y dos cargadores que mantener.

### 2.4 Alternativa M1 — Aprobaciones encadenadas en un único cargador

**Idea.** ENM pasa a verificar una **cadena** de aprobaciones (E01, E02, …):
- **Conjunto auditado propio.** Cada aprobación conserva su conjunto auditado (las claves de su `sha256_auditados`) y se verifica **siempre contra su propio commit**: E01 contra A1, con sus huellas originales, aunque esos ficheros hayan cambiado después.
- **Sustitución declarada.** Si una aprobación posterior cambia un fichero auditado antes, debe declararlo en `sustituye: {fichero: huella anterior}`, y la huella anterior debe coincidir con la de la aprobación que reemplaza.
- **Vínculo con la anterior.** Incluye `requiere: {E01: huella canónica de la entrada E01}`, y su commit debe descender del de la anterior.
- **Árbol de trabajo.** Cada fichero debe coincidir con su **última** versión aprobada.
- **Ancla.** La huella de la entrada E01 queda fijada en el código (`PINNED_PRIOR_APPROVALS`).

**Ficheros:**

| Ficheros | En M1 |
|---|---|
| 9 del protocolo, manifiesto, YAML de E01, entrada E01 del registro | sin cambios (la entrada E01, con la misma huella canónica) |
| ENM | **sustituido**: nueva lógica de cadena, E02 en `AUTHORIZED_AMENDMENTS` y `WHITELIST`, ancla de E01; declarado en `sustituye` |
| `tests/test_tda17_enmienda_01.py`, `tests/test_tda17_indicators.py`, `tests/tda17_synthetic.py` | **sustituidos**. En MIG v2 la cadena reemplaza a `load_effective_protocol` con el mismo nombre (vía de producción real) y 205 de 303 pruebas auditadas fallan o dan error. En 196 casos el motivo es «E01: la entrada publicada del registro ha cambiado»: la infraestructura de pruebas de E01 (`approve_registry`) reescribe la entrada E01 con aprobaciones sintéticas, y el ancla de M1 lo rechaza. Su contenido se conserva en A1 |
| `tda17_indicators.py` | sin cambios: la cadena conserva el nombre `load_effective_protocol`, de modo que `compute_catalog` queda verificada por la cadena |
| YAML de E02, entrada E02 **en el mismo registro**, pruebas y módulos de la fase 2, acta | nuevos |

**Commits y registros:** igual que M2 (A2 con E02 pendiente; P2 con la aprobación que cita A2, `requiere` y `sustituye`), pero con un solo registro y un solo cargador.

**Verificaciones demostradas (MIG v2, M1: A2 `2a4b1c6`, P2 `fe0ff2d`):**
- la entrada E01 es idéntica a la publicada;
- el cargador antiguo, conservado como `_load_effective_protocol_v1`, **rechaza** con «sha256_auditados no cubre exactamente» (fallo seguro);
- la cadena verifica E01 contra `386815b` y E02 contra `2a4b1c6`, y la versión vigente del cargador es la de E02;
- pytest de las pruebas auditadas de E01: `exit 1`, 62 failed, 98 passed y 143 errors (causa en la tabla anterior).

**Negativas con motivo exacto (11/11 superadas):** cargador antiguo tras la migración; motor de la fase 1 modificado; YAML de E01 modificado; YAML de E02 modificado; aprobación de E01 reescrita; commit inexistente; commit sin la historia de E01; E02 pendiente; sustitución no declarada; sustitución con huella anterior falsa; cargador devuelto a la versión de E01. Este último ahora se rechaza de forma **explícita**, con «enmiendas no autorizadas en el registro: ['E02']», y no por falta de una función como en la v1.

**Ventajas.** Escala a E03 y siguientes sin capas y con un solo registro. Permite, si algún día hiciera falta, cambiar ficheros auditados por E01 (incluido el motor) con sustitución declarada.

**Inconvenientes.**
- Sustituye el cargador y la infraestructura de pruebas auditada de E01: 3 ficheros de pruebas y 205 de sus 303 pruebas, que habría que reescribir y re-auditar como parte de E02.
- La lógica de verificación es más compleja (hay más superficie que auditar).

### 2.5 Comparación y recomendación

| Criterio | M2 escalonado | M1 encadenado |
|---|---|---|
| Ficheros auditados de E01 sin cambios | **los 8** | YAML, entrada y motor sí; cargador y 3 ficheros de pruebas sustituidos |
| Pruebas auditadas de E01 | 303/303 sin tocar (exit 0) | 205/303 fallan o dan error; habría que reescribir su infraestructura |
| E01 verificada contra A1 | sí, con el cargador original | sí, con el cargador nuevo |
| E02 aprobable de forma independiente | sí | sí |
| E03 y siguientes | otra capa o paso a M1 | nativo |
| Cambios en ficheros de E01 (motor) | imposible | posible, con sustitución declarada |
| Superficie de auditoría de E02 | módulo nuevo y pequeño | cargador completo |

**Recomendación (decisión del responsable).** M2 para E02. Ninguna decisión del inventario exige tocar ficheros auditados por E01, y M2 preserva la auditoría de E01 byte a byte. Si más adelante una enmienda necesitara cambiar el motor de la fase 1, migrar entonces a M1 con su propio ciclo A→P.

Los prototipos de M1 y M2 son **desechables**: solo existen en los clones del scratchpad y no deben copiarse al repositorio sin implementarse y auditarse como parte de E02.

### 2.6 Reproducibilidad de la evidencia (F1, F5)

- `python mig_demo.py <REPO> <WORKDIR>` crea clones desechables del repositorio real en P1. Sus commits llevan identidad y fechas **fijas** (2026-10-18T12:00:00+00:00), así que los SHA son deterministas.
- **Dos ejecuciones** con logs de nombre distinto (`mig_demo_v3_run1.log` y `mig_demo_v3_run2.log`):
  - 3561 líneas cada uno, idénticos salvo la etiqueta final de ejecución;
  - los 10 SHA de commits coinciden, entre ellos A2, P2, Pbad, la rama paralela de la regresión, M1 y M0;
  - código de salida **0** en ambas.
- **El código de salida es 0 solo si** se cumplen a la vez:
  - positivas sin fallo;
  - las 3 ejecuciones obligatorias de pytest con código 0 y recuento exacto;
  - negativas, omisiones instrumentadas y autopruebas del comparador sin fallo;
  - el fallo **declarado** de M1 observado;
  - la regresión F4 reproducida.
- **Inyección de fallos** (`mig_demo_inyeccion.py`, con la misma función de decisión) da (1, 1, 0), como se esperaba.
- **Defecto detectado en la v2, gracias a la exigencia de motivos exactos.** `git show` se decodificaba con cp1252, y eso alteraba la huella de la entrada E01. Se corrigió leyendo en bytes y decodificando como UTF-8.
---

## 3. Reglas ya congeladas que NO requieren enmienda

| Asunto | Regla vigente | Fuente |
|---|---|---|
| Población | Casos completos de P (134 columnas), para **todos** los análisis de `analisis` (incluido el panel L, D9) | YAML `analisis.poblacion` l. 380 |
| Celdas | Global; años 2020–2025 (2025 parcial, **cuenta** en 5/6); 2019 NO_EVALUABLE; 7 segmentos [inicio, fin); año×segmento solo descriptivo | l. 382–390; l. 547 |
| Pearson y Spearman | Pearson sobre series transformadas; Spearman = Pearson de rangos medios dentro de la celda | l. 391–393; D21 |
| Bootstrap | Bloques móviles circulares de **jornadas**, 5, 300 réplicas, semilla 1717; S4 (1 y 10) | l. 423; D9; l. 534 |
| R² | Cada columna frente a las otras 133; Cholesky con pivoteo; verificaciones cruzadas; τ 1e-10; tolerancia 1e-8; variante sobre rangos | l. 396–404; D12; D23 |
| Exactitud | Por registro y residuo; 1e-9 y 1e-4; identidad nueva solo por enmienda | l. 407–416 |
| Registro de rango | 11 identidades en P (9 + 2), 3 en L; nulos esperados 10/11/0/3; acción BLOQUEAR_HASTA_EXPLICAR | l. 322–374 |
| Cuenta de nulos | sobre **autovalores** (no valores singulares) | l. 410 |
| PCA, agrupamiento, clasificación | métricas, método y cortes; regla 5/6 años y 6/7 segmentos; nunca NUEVA_INFORMACION | l. 417–433 |
| Por celda | «todo lo anterior por segmento y por año» (incluye el agrupamiento) | ROAD l. 1101 |
| S1, marcas, S6, D7 fuera de S1/S2, E1–E5 sin marcas | — | E01 l. 55–122; YAML l. 536–542 |
| G2-2, G2-3a, G2-3b | especificación generativa y de viabilidad completa | l. 449–522; INF §5, §7 |
| NO_EVALUABLE | ninguna regla cambia umbrales tras ver resultados | l. 546–560 |

---

## 4. Fichas

### A. Rango y nulidad numérica

#### A1 — Convención del recuento de nulos · `AC` (con A4 y A3) · `BLOQ-EJEC`
- **Fuente.**
  - YAML `identidades.verificacion_numerica` (l. 371–374).
  - YAML `analisis.exactitud.ALERTA_DEPENDENCIA_NO_REGISTRADA` (l. 410).
  - T-V4A l. 189–195 (diseño auditado).
  - T-PROT l. 214–236.
  - AN l. 511–527.
- **Respaldado por el texto congelado:**
  - (i) se cuentan **autovalores** nulos (l. 410);
  - (ii) con τ = 1e-10 **relativo** (l. 372);
  - (iii) se comparan con `nulos_esperados` (l. 373);
  - (iv) si difieren, BLOQUEAR_HASTA_EXPLICAR (l. 374).

  La convención por valores singulares no tiene apoyo normativo. Solo aparece en T-PROT, que comprueba algebraicamente el registro sobre columnas aleatorias con dependencias exactas, donde ambas convenciones coinciden.
- **Indefinido** (el texto no lo fija; v4a es evidencia de diseño, no texto normativo):
  1. **Matriz:** correlación (v4a, `corr(P)`), covarianza (cuyos autovalores dependen de las unidades de las columnas) o matriz de productos cruzados.
  2. **Normalización:** con la correlación, cada columna tiene varianza 1; con la covarianza no.
  3. **«P sin centrar»** (10 nulos esperados): no existe correlación sin centrar. Las opciones son la matriz de productos cruzados bruta (dependiente de la escala) o la de cosenos (columnas normalizadas a norma 1).
  4. **Aplicación exacta de τ:** `λ ≤ τ·λ_max` (v4a) frente a `λ ≤ τ·traza` o `λ/λ_max < τ`; trato de los autovalores negativos por redondeo; suelo de resolución (~1e-15 en doble precisión, DIAG).
  5. **Ámbito:** ver A4.
- **Alternativas:**
  - **(a)** Correlación con `λ ≤ τ·λ_max`, y cosenos para «sin centrar» (coherente con v4a; invariante a la escala).
  - **(b)** Covarianza o productos cruzados brutos (dependiente de la escala).
  - **(c)** Mantener el requisito conjunto actual de la implementación, que no está en el texto.
- **Consecuencias (DIAG, sintético).**
  - Con columnas normalizadas, (a) con τ equivale a valores singulares con √τ.
  - (a) anula cuasidependencias con 1 − R² ≲ 1e-10.
  - (b) depende de la escala: una columna independiente reescalada por 1e-10 cuenta como nula.
- **Recomendación:** (a), por apoyo textual (autovalores) y de diseño (correlación, v4a). Como el texto no fija la matriz, la normalización ni «P sin centrar», recomiendo **aclararlo formalmente** en E02, junto con A4, en lugar de fijarlo solo en la implementación. No cambia τ ni los nulos esperados.
- **Dependencias:** ninguna. A3, A2 y B3 dependen de A1.

#### A2 — Bloqueo por recuento distinto: conclusiones frente a diagnósticos · `N` · `BLOQ-EJEC`
- **Fuente:** YAML l. 374, l. 410, l. 416 (`enmiendas_posteriores`); N7, N18, N19 (l. 345, 356, 357); ROAD l. 1115.
- **Lo que el protocolo ya fija.** Si el recuento difiere del registro: (i) la matriz recibe ALERTA; (ii) la etapa se bloquea **hasta explicar**; (iii) una identidad nueva solo entra por enmienda posterior, documentada y etiquetada. **No hay continuación automática, y esta revisión no la propone.**
- **Distinción necesaria.**
  - **Bloqueo de conclusiones.** Mientras dure el bloqueo no se publican la clasificación (RECONSTRUIBLE / NUEVA_REPRESENTACION / INESTABLE), los subniveles, la etiqueta EXACTA, ni k_PA o la transferencia de las celdas afectadas.
  - **Obligación de emitir diagnósticos.** Para poder «explicar», hay que producir los diagnósticos que lo permiten. Producirlos no levanta el bloqueo; es la condición para resolverlo. Son:
    - el espectro;
    - las direcciones nulas sobrantes y sus cargas;
    - R² y etiquetas por columna;
    - los recuentos por las dos convenciones.
- **Indefinido:**
  1. qué diagnósticos son obligatorios durante el bloqueo;
  2. qué constituye una explicación: por ejemplo, la atribución de cada dirección nula sobrante a una relación N registrada con un criterio numérico predeclarado;
  3. quién la aprueba y con qué instrumento (acta, enmienda);
  4. si una explicación aprobada permite publicar las conclusiones de columnas **no** implicadas o exige una enmienda que actualice el registro.
- **Alternativas** (todas exigen una decisión formal para levantar el bloqueo; ninguna continúa sola):
  - **(a) Diagnóstico obligatorio y levantamiento formal.** E02 predeclara los diagnósticos y el procedimiento de atribución. El bloqueo solo se levanta con una enmienda posterior que documente la explicación y fije cómo se tratan las columnas implicadas.
  - **(b) Excepción predeclarada en E02.** Se fija ahora una lista cerrada de relaciones (por ejemplo N7, N18, N19) y un criterio numérico de atribución. Si **todas** las direcciones sobrantes se atribuyen por ese criterio, se permite publicar las conclusiones de las columnas no implicadas, con las implicadas marcadas.

  **Compatibilidad de (b):** cambia en parte el efecto de `accion_si_difiere` (que hoy es un bloqueo sin excepciones). Solo es admisible si el responsable entiende que una excepción predeclarada antes de ver datos no contradice «bloquear hasta explicar». Si no lo entiende así, (b) no es compatible.
- **Consecuencias.**
  - (a) es plenamente compatible, pero puede retrasar indefinidamente las conclusiones si hay cuasidependencias.
  - (b) adelanta conclusiones sobre las columnas no implicadas, con el riesgo de que el criterio de atribución resulte insuficiente.
  - **Efecto de una cuasidependencia sobre el R²** (distinción F3). Hay que separar tres niveles:
    1. **R² matemático.** El R² poblacional de la columna frente a las demás está bien definido y es < 1. Ni la convención de nulos ni A2 lo cambian: la convención solo decide si la **matriz** recibe ALERTA y si se bloquean conclusiones.
    2. **Efectos numéricos.**
       - El R² oficial también usa τ = 1e-10, pero como umbral de pivote de su propia base de Cholesky (`analisis.r2.tau_relativa`, l. 400, distinto de `identidades.verificacion_numerica.tau_relativa`, l. 372). Si la cuasidependencia tiene la escala de τ, la base truncada puede dar un R² calculado distinto del matemático.
       - En doble precisión, 1 − R² no se resuelve por debajo de ~1e-16.
       - Las tres vías de cálculo (base, espectral y mínima norma) pueden discrepar más de 1e-8. DIAG, sintético, lo muestra en la franja ε ≈ 1e-13 … 1e-5.
    3. **Efectos de publicación.** Si las verificaciones cruzadas discrepan más de 1e-8, el R² afectado es `R2_NO_PUBLICABLE_POR_COLINEALIDAD` (l. 412, 553). Las etiquetas ALERTA (1 − R² ≤ 1e-9) o CASI_EXACTA (≤ 1e-4) se asignan sobre el R² **calculado**. Un recuento de nulos distinto del registro bloquea las conclusiones (A2).
  - Las métricas de dimensión apenas cambian, porque los autovalores próximos a 0 pesan poco. V1 de G2 no se ve afectado: solo exige las identidades lineales exactas.
- **Sobre MNQ.** DIAG muestra, con generadores **sintéticos**, que relaciones casi lineales del catálogo (N7, N18, N19) pueden producir nulos de más según la volatilidad relativa. **No** permite afirmar si ocurrirá con MNQ.
- **Recomendación:** (a) como mínimo imprescindible, predeclarando ya los diagnósticos y el procedimiento de atribución. (b) solo si el responsable la considera compatible con el texto congelado.
- **Dependencias:** A1, A4.

#### A3 — Definición del «hueco espectral» · `AC` (con A1) · `BLOQ-ANAL`
- **Fuente:** YAML l. 402, 413, 554.
- **Definido:** umbral 1e4; por debajo, EXACTA no se publica en la celda.
- **Falta:** el cociente y la matriz.
- **Alternativas:**
  - **(a)** λ_(p−m)/λ_(p−m+1), con m = nulos registrados, sobre la matriz de A1.
  - **(b)** El menor autovalor por encima de τ dividido por el mayor por debajo.
  - **(c)** Por identidad.
- **Consecuencias.** (a) es coherente con A1. (b) depende del recuento observado, que A2 puede alterar. (c) es local y más costosa.
- **Recomendación:** (a).
- **Dependencias:** A1. B14 depende de A3.

#### A4 — Ámbito del control de nulos · `AC` (con A1) · `BLOQ-EJEC`
- **Fuente:** YAML l. 371–374, sin ámbito; l. 389; l. 547; `panel_L` l. 567–570.
- **Falta.** Si el control y su bloqueo se aplican en la celda global, en las celdas evaluables o también en las 49 de año×segmento (descriptivas y algunas con n cercano a p), y cómo se aplican a L por contrato.
- **Alternativas:** (a) global y celdas evaluables; las descriptivas solo informan. (b) Todas las celdas. (c) Solo global.
- **Consecuencias.** (b) puede bloquear la etapa por celdas descriptivas pequeñas. (c) no detecta dependencias propias de un segmento.
- **Recomendación:** (a); L por contrato y solo como diagnóstico.
- **Dependencias:** A1.

#### A5 — Escala del residuo relativo · `ID` · `INFORME`
- **Fuente:** YAML l. 408; T-V4A l. 97–103; AN l. 486–508.
- La implementación reproduce la normalización del prototipo v4a: máx |Σcx − c₀| / máx(|c₀|, máx |c_i x_i|).
- Basta con documentarlo.

### B. Intervalos de confianza e inferencia

#### B1 — Nivel, lateralidad y tipo del intervalo · `N` · `BLOQ-EJEC`
- **Fuente:** YAML l. 423, l. 425–426; D9; D21.
- **Exigido:** IC de R²_TODAS por año y segmento (clasificación) y de Spearman (D21).
- **No fijado:** el nivel, si es bilateral o unilateral (la regla usa un extremo por dirección) y el tipo (percentil, básico, BCa o studentizado).
- **Consecuencias.**
  - Un nivel mayor produce más columnas INESTABLE.
  - Un percentil es sesgado para R² cerca de 1.
  - BCa necesita jackknife por bloques.
- **Recomendación:** no se recomienda nivel. Fijar los tres elementos a la vez.
- **Dependencias:** ninguna. B2, B3, B4, B9 y B13 dependen de B1.

#### B2 — Unidad y ámbito del remuestreo por celda · `AC` (con B1) · `BLOQ-EJEC`
- **Fuente:** YAML l. 423; l. 382–390; AN l. 604–660.
- **Falta:**
  - bootstrap por celda o derivado de uno global;
  - qué es una «jornada» en un segmento;
  - orden circular;
  - semilla por celda;
  - población fija de la réplica.
- **Consecuencias.** Por celda: IC independientes, sin envolvimiento entre años. Global: réplicas que cruzan años, número de jornadas variable por celda e IC correlacionados entre celdas.
- **Recomendación:** por celda.
- **Dependencias:** B1.

#### B3 — Estadístico en las réplicas y réplicas degeneradas · `AC` (con B1) · `BLOQ-ANAL`
- **Fuente:** YAML l. 396–404, l. 553.
- **Falta:**
  - si se aplica el método oficial y las verificaciones en cada réplica;
  - qué se hace con réplicas no publicables o con otro rango;
  - qué IC tiene una columna EXACTA.
- **Alternativas:**
  - **(a)** Método oficial; réplicas no publicables excluidas, con un límite que habría que fijar.
  - **(b)** Sin verificaciones en las réplicas.
  - **(c)** Invalidar la celda si alguna réplica falla.
- **Dependencias:** B1, A2.

#### B4 — Validación de Spearman con rangos fijos · `AC` (con B1) · `BLOQ-ANAL`
- **Fuente:** YAML l. 394–395.
- **Falta:**
  - qué año se usa;
  - sobre qué se toma la mediana de las diferencias;
  - si las réplicas exactas y las de rangos fijos son las mismas;
  - el alcance de la alternativa de 100 réplicas.
- **Dependencias:** B1, B2.

#### B5 — Intervalos para otros estadísticos · `OP` · `INFORME`
- El protocolo exige IC solo para Spearman y para la clasificación.
- Pearson, las métricas de dimensión y la estabilidad se publican sin IC salvo ampliación expresa, que **no** debe entrar automáticamente en E02.

#### B6 — Convención de reproducibilidad del bootstrap · `AC` (con B2) · `INFORME`
- **Fijado:** la semilla 1717 (l. 423).
- **No fijado:** el generador exacto, que hace falta para reproducir las réplicas bit a bit:
  - el *bit generator* (por ejemplo PCG64) y cómo se construye a partir de la semilla (`default_rng(1717)` frente a `Generator(PCG64(SeedSequence(1717)))` o flujos `spawn`, como hace G2-2 en l. 467–468);
  - si cada celda tiene su propio flujo o todas consumen uno común (depende de B2);
  - el orden de consumo: inicios de bloque por réplica y celdas en un orden determinado;
  - el comportamiento si cambia la versión de numpy (G2-2 lo prevé en l. 492; el bootstrap no).
- **Consecuencia.** No cambia la metodología, pero sin una convención declarada los IC no son reproducibles entre implementaciones ni versiones. Con la semilla sola no basta.
- **Recomendación:** declarar la convención junto con B2, siguiendo el modelo de G2-2 (PCG64, `SeedSequence` y flujos). La implementación actual (`default_rng`, un flujo por llamada; AN l. 604–618) es provisional.

#### B7 — Dependencia temporal · `ID` · `INFORME`
- Bloques de 5 jornadas (D9) y S4 con 1 y 10.
- Ya congelado; se documentan sus límites.

#### B8 — Transferencia (D19) · `AC` (con B9 y B10: «variantes de R²») · `BLOQ-ANAL`
- **Fuente:** YAML l. 405–406, l. 556.
- **Falta:**
  - qué es R²_in;
  - si la suma de cuadrados total de la celda retenida usa su media o la de entrenamiento;
  - si se reestima el intercepto;
  - qué predictores se usan;
  - el rcond de la mínima norma.
- **Consecuencias.** Con la media de entrenamiento se penalizan los cambios de nivel entre celdas; con la propia, solo se mide la forma.

#### B9 — Subniveles PARCIAL y LINEALMENTE_DISTINTA · `AC` (con B8 y B13) · `BLOQ-ANAL`
- **Fuente:** YAML l. 428–430.
- **Falta:** con qué R² se asignan (global, mediana de celdas o IC).
- **Dependencias:** B1, B13.

#### B10 — R2_OTROS_INDICADORES · `AC` (con B8) · `BLOQ-ANAL`
- **Fuente:** l. 403.
- **Falta:** si excluye solo las 6 referencias de TDA-04 o también las demás salidas y configuraciones del mismo elemento.

#### B11 — T3 y V2 · `N` · `BLOQ-ANAL` (T3 y descomposición)
- **Fuente:** YAML l. 437–439, l. 526, l. 528, l. 555.
- **Definido:** umbrales y contingencia.
- **Falta:** qué es T3, qué son las «falsas T3» y con qué controles se cuentan.
- Hay lecturas alternativas sin base textual para elegir entre ellas.
- **Dependencias:** D11 y D12 dependen de B11.

#### B12 — Análisis paralelo: k_PA, max-T y V3 · `N` · `BLOQ-ANAL` (k_PA)
- **Fuente:** YAML l. 421, l. 527, l. 549–552; INF l. 120.
- **Falta:** estandarización, cuantil y regla de parada de la banda, y el cálculo exacto de V3.
- Las demás métricas de dimensión no dependen de esto.

#### B13 — Celdas con R2_NO_PUBLICABLE en la regla de clasificación · `AC` (con B1) · `BLOQ-ANAL` (clasificación principal)
- **Fuente:** YAML l. 425–427, l. 553 (verificación cruzada > 1e-8).
- **Solo** se refiere a la no publicación del **R²** de una columna en una celda.
- **Falta:** si cuenta como incumplimiento o se descuenta del denominador.
- **Implementación actual:** cuenta como incumplimiento (AN l. 667–692).
- **Consecuencias.** Como incumplimiento, lleva hacia INESTABLE; descontada, clasifica con menos celdas.
- **Dependencias:** B1.

#### B14 — SEPARACION_NUMERICA_INSUFICIENTE en columnas EXACTA (separada de B13) · `AC` (con A3) · `BLOQ-ANAL` (subnivel EXACTA)
- **Fuente:** YAML l. 413, l. 554, l. 428 (`subniveles: {EXACTA: identidad_registrada}`).
- **Definido:**
  - la separación insuficiente **solo** impide publicar la etiqueta EXACTA en esa celda;
  - **no** afecta al R² ni a su publicabilidad, que regula B13;
  - el subnivel EXACTA se define por registro.
- **Falta:**
  - si una columna registrada conserva el subnivel EXACTA global cuando en alguna celda no se publica la etiqueta;
  - cómo se informa ese caso.
- **Dependencias:** A3.

### C. Escalas S/M/L de S2

#### C1 — Asignación S/M/L · `N` · `BLOQ-ANAL` (S2)
- **Fuente:** YAML l. 532 (S2, «escalas S y M»), l. 604 (D2, «escalas por familia: canónicas y anclas 60/240»); E01 l. 119; ENM l. 410–419.
- **Comprobado sobre el orden de configuraciones del YAML:**
  - la posición canónica, ancla 60, ancla 240 coincide con D2 en todos los elementos de 2 o 3 configuraciones;
  - **excepción:** B8 TSI `r100_s52`, que no es ancla 60 ni 240;
  - con una sola configuración: B9, D8, E1–E5 y D4;
  - D7 se mide en jornadas.
- D2 no nombra S, M ni L, y la clave `sensibilidades.escalas_S2` no existe (E01 no puede añadirla). Por tanto, aunque la lectura por posición esté respaldada, **hace falta una decisión normativa nueva**.
- **Alternativas:** (A) por posición D2; (B) por ventana de marcas de E01-B, con cortes nuevos. Difieren en 20 configuraciones.
- **Recomendación:** (A), con B8 `r100_s52` decidido expresamente.
- **Dependencias:** C4 (migración). C2 y C3 dependen de C1.

#### C2 — Panel y predictores de S2 · `AC` (con C1) · `BLOQ-ANAL`
- **Fuente:** E01 l. 119–120 («las series de escala L no entran en S2»).
- **Falta:** si S2 trabaja sobre un subpanel S+M (otro p y otro registro de rango) o sobre P completo excluyendo filas por las marcas S y M; y cómo se tratan las referencias de TDA-04, E1–E5 y D7.
- **Recomendación:** P completo con exclusión de filas por las series S y M, para que sea comparable con el análisis oficial.

#### C3 — VWAP (D8) · `AC` (con C1) · `BLOQ-ANAL`
- **Ya determinado por E01:** E1–E5 y las referencias de TDA-04 no excluyen filas (marcas = 0), y D7 está fuera de S2.
- **Falta solo:** la escala de D8 (ventana de sesión).

#### C4 — Procedimiento de migración E01 → E02 · `N` (gobierno) · `BLOQ-ANAL` (todo lo que dependa de E02)
- Ver §2. Hay que elegir M2, M1 u otra alternativa que cumpla R1–R7 antes de redactar E02.

### D. Resto del motor, sensibilidades, G2 y salidas

| ID | Asunto y fuente | Falta | Categoría · Bloqueo | Agrupa con · depende de |
|---|---|---|---|---|
| D1 | Evaluabilidad tras S1/S2 (E01 l. 120; control `s1_population_report`) | Una población mínima sería un criterio **nuevo**. Sin él, se publican los recuentos | `OP` · INFORME | — |
| D2 | Columna constante en una celda (sin regla; AN l. 359–376) | Celda NO_EVALUABLE, retirada de la columna o detención | `AC` · BLOQ-ANAL (condicional) | A4 |
| D3 | Alcance de S7 y S8 (YAML l. 543–544) | Global o por serie con unión; configuración | `AC` · BLOQ-ANAL | D4, D6 (sensibilidades) |
| D4 | Base de la media de S5 (l. 535) | Población global o de la celda | `AC` · BLOQ-ANAL | D3, D6 |
| D5 | Cuantiles de S3 (l. 533) | Método de interpolación (efecto despreciable) | `ID` · INFORME | — |
| D6 | Qué se recalcula en cada sensibilidad (l. 530–544, l. 597) | Conjunto de estadísticos e IC | `AC` · BLOQ-ANAL | D3, D4; depende de B1 |
| D7 | Agrupamiento por celda | ROAD l. 1101 lo determina: «todo… por segmento y por año». El oficial es el global (l. 422) y el resto se publica por celda | `ID` · INFORME | — |
| D8 | Contenido de `TDA17_estabilidad_celdas.csv` (l. 597) | Esquema | `AC` · INFORME | D10, D18 (salidas) |
| D9 | Población del panel L (l. 390, l. 567–570) | Ver nota 1 | `ID` · INFORME | — |
| D10 | Estadísticos de año×segmento (l. 389) | Cuáles | `AC` · INFORME | D8 |
| D11 | Descomposición G2 (l. 528) | Pearson o Spearman; 2 semillas; celdas; población de la réplica | `AC` · BLOQ-ANAL | B11 |
| D12 | Papel de G2-2 en las salidas | Uso inferencial no escrito | `AC` · BLOQ-ANAL | B11 |
| D13 | Salida de G2-3a (l. 505) | Estadístico publicado y uso | `AC` · BLOQ-ANAL | B11, B12 (grupo G2) |
| D14 | Reconstrucción de G2-1a y G2-1b (l. 447–448; G2 l. 84–130) | Confirmar la lectura «fronteras en su sitio» | `ID` (confirmación) · BLOQ-ANAL | — |
| D15 | Generador de G2-1 y de las réplicas de G2-3b (l. 447–448, 510: solo semillas) | Igual que B6: el protocolo fija la semilla, no el generador ni el orden de consumo (grupos de G2-1, réplicas 0..199 de G2-3b). Hace falta una convención de reproducibilidad declarada. La de G2 docstring l. 13–16 es provisional | `AC` · INFORME | D14 (por coherencia con B6) |
| D16 | Comprobación numérica de N1–N21 en `TDA17_identidades_exactas.csv` | El protocolo no la exige (exactitud solo para las lineales). Basta listarlas | `OP` · INFORME | — |
| D17 | Auditoría de causalidad, TH37 | Ver nota 2 | `N` · **BLOQ-EJEC** | — |
| D18 | Unidad del «retraso» de D7 (E01 l. 52) | Jornadas o barras | `AC` · INFORME | D8 |
| D19 | Rango de L por contrato | Ámbito | `AC` · INFORME | A4 |
| D20 | Celda de las figuras (l. 600) | Global | `ID` · INFORME | — |
| D21 | Cambio de versión de numpy (l. 492) | Ya previsto | `ID` · INFORME | — |

**Nota 1 — D9 (revisada, C4).** La elección de población del panel L sí tiene consecuencias numéricas:
- los casos completos de P excluyen las barras con H = L (E1–E4 NaN), con V = 0 (`ln_volume` NaN) y las anteriores al calentamiento máximo, y esas exclusiones no son aleatorias: se concentran en momentos de poca actividad;
- con pares completos se incluirían.

Pero `analisis.poblacion: CASOS_COMPLETOS_DE_P` (l. 380) es la población de toda la sección `analisis`, y el panel L se declara dentro de ella (`analisis.celdas.panel_L`, l. 390). Las fuentes, por tanto, ya determinan los casos completos de P: **no requiere enmienda**. El informe debe declarar la población usada.

**Nota 2 — D17, auditoría de causalidad (TH37, revisada, C4).** BACK l. 735–740: «OBLIGATORIA — bloqueante»; «cualquier discrepancia… invalida los outputs afectados». El alcance completo según el backlog tiene cuatro partes:
1. **Información futura.** Hace falta una prueba de reconstrucción por transformación: recalcular cada serie con los datos truncados en t y exigir igualdad en t. Hoy solo existe una prueba sintética del motor (`tests/test_tda17_indicators.py` l. 568–582). Falta fijar, para datos reales:
   - los puntos de corte (fronteras de contrato y de jornada, y una muestra con semilla);
   - su número;
   - la tolerancia (igualdad exacta o relativa);
   - el tratamiento de D7, que cruza rolls, y de VWAP, que se reinicia por jornada.
2. **Parámetros estimados globalmente.** Los del catálogo están fijados a priori (D2), así que no hay estimación. Sí son de muestra completa, y por tanto RETROSPECTIVOS:
   - σ y la curtosis de G2-2 (TDA-07), que solo se usan en el nulo, no en las series;
   - los cortes de segmento (TDA-06), que definen celdas, no series;
   - la media de S5 (YAML l. 535, ya etiquetada RETROSPECTIVO).

   Falta la **declaración del origen de cada parámetro** que exige BACK l. 737, y su formato en la tabla.
3. **Ajuste de roll** (separado de las transformaciones de precio, F3).
   - **3a. Invariancia frente a factores de ajuste de roll** (la exige BACK l. 737).
     - Un ajuste de roll es un factor, multiplicativo (`adj_ratio`) o aditivo (`adj_diff`), aplicado a los precios de los contratos **anteriores** a cada roll para encadenar una serie continua. TDA-17 lo prohíbe (l. 85) y calcula cada serie dentro de su contrato, con reinicio en cada roll (P1).
     - Por tanto, la serie de un contrato no puede depender de los precios de otros contratos: cualquier factor aplicado a los contratos anteriores no cambia las series del contrato vigente, sea cual sea la forma de cada fórmula.
     - **Prueba requerida:** alterar arbitrariamente los precios OHLC de los demás contratos (en particular, con factores de ajuste) y exigir series y marcas **idénticas** en el contrato vigente.
     - **Excepción declarada:** D7 RVOL cruza rolls, pero solo con volumen. Ningún factor de precio la altera; su marca `rvol_cruza_roll` la identifica.
     - Hay que comprobar también que la entrada de TDA-04 no contiene precios ajustados.
   - **3b. Comportamiento ante transformaciones reales de los precios OHLCV** (no es un requisito de TH37).
     - Reescalar o desplazar los precios **de un mismo contrato** no es un ajuste de roll: es una propiedad de cada fórmula.
     - Las series adimensionales de P (cocientes y logaritmos) son invariantes a un reescalado multiplicativo, pero **no** a un desplazamiento aditivo.
     - Las series del panel L en puntos o en ticks escalan con el factor, y la rejilla de tick (0,25) no se conserva salvo con factores compatibles.
     - Documentar esta equivarianza por serie sería un diagnóstico `OP`. No debe confundirse con 3a ni exigirse como invariancia.
4. **Etiquetas.** CAUSAL_AT_BAR_CLOSE para las series y RETROSPECTIVO para los análisis ya están en YAML l. 381. Falta el esquema de `TDA17_auditoria_causalidad.csv` (l. 595).

**Recomendación:** definir 1–4 en E02 antes de ejecutar; es una decisión normativa nueva.

---

## 5. Inventario mínimo de E02 y orden de resolución (C5)

### 5.1 Reclasificación completa

| Categoría | Fichas |
|---|---|
| **N — nueva imprescindible (7)** | A2, B1, B11, B12, C1, C4 (gobierno), D17 |
| **AC — aclaración agrupable (26)** | A1, A3, A4 · B2, B6, B3, B4, B13 · B8, B9, B10, B14 · C2, C3 · D2 · D3, D4, D6 · D8, D10, D18, D19 · D11, D12, D13, D15 |
| **ID — determinada por las fuentes (8)** | A5, B7, D5, D7, D9, D14 (confirmación), D20, D21 |
| **OP — ampliación opcional, fuera de E02 por defecto (3)** | B5, D1, D16 |

### 5.2 Dependencias

```
C4 (migración) ─► todo E02
A1 ─► A3 ─► B14
A1, A4 ─► A2 ─► B3
B1 ─► B2 ─► B4 ; B2 ─► B6
B1 ─► B13 ─► B9
B1 ─► B3 ; B1 ─► D6
C1 ─► C2, C3
B11 ─► D11, D12 ; B11, B12 ─► D13 ; D14 ─► D15
D17 (independiente)
```

### 5.3 Propuesta priorizada del conjunto mínimo de decisiones normativas

**Núcleo imprescindible de E02** (sin él no se puede ejecutar TDA-17 ni producir sus salidas obligatorias):

| Prioridad | Decisión | Contenido | Por qué es mínima |
|---|---|---|---|
| 0 | **C4** | Elegir el procedimiento de migración (recomendado M2) | Sin él, E02 invalida E01 (§2.1) |
| 1 | **A1 + A4** (AC) | Matriz, normalización, «P sin centrar», forma de aplicar τ, ámbito | El control de nulos decide si la etapa se bloquea |
| 2 | **A2** (N) | Diagnósticos obligatorios durante el bloqueo; procedimiento de explicación; instrumento de levantamiento | Sin él, un bloqueo no tiene salida |
| 3 | **B1** (N) + **B2, B6, B13** (AC) | Intervalo (nivel, lateralidad, tipo); remuestreo por celda y su convención de reproducibilidad; celdas no publicables en la regla | Clasificación, salida obligatoria de TH35 |
| 4 | **D17** (N) | Alcance operativo de TH37 (puntos 1–4 de la nota 2) | TH37 es obligatoria y bloqueante |

**Recomendable en la misma E02** (si se decide a tiempo; si no, los análisis afectados no se publican):

| Prioridad | Decisiones | Bloquean |
|---|---|---|
| 5 | A3, B14, B3, B4 | Etiqueta EXACTA por celda; IC de R² y de Spearman |
| 6 | B8, B9, B10 | Transferencia, subniveles, variante de R² |
| 7 | C1 + C2, C3 | S2 (sigue bloqueada mientras tanto) |
| 8 | B11 + D11, D12; B12 + D13; D15 | T3, descomposición G2, k_PA, G2-3a; reproducibilidad de G2-1 y G2-3b |
| 9 | D3, D4, D6, D2 | S5, S7, S8, tabla de sensibilidades, columna constante |
| 10 | D8, D10, D18, D19 | Solo el informe |

**Fuera de E02 por defecto:** B5, D1 y D16 (`OP`), y las fichas `ID`, que se documentan en la implementación sin enmienda: A5, B7, D5, D7, D9, D14, D20 y D21.

**Si E02 solo incluye el núcleo**, las prioridades 5–10 requerirán una E03. Con M2 eso supone otra capa; con M1 es nativo (§2.5). Esto debe pesar en la elección de C4: si se prevén más enmiendas, conviene valorar M1 desde el principio o incluir en E02 el mayor número posible de prioridades.
