# TDA-17 — Paquete de auditoría de la implementación de E02 (versión candidata a A2)

> **Versión definitiva: v11** (2026-10-23).
> - Es la v10, aprobada técnicamente por la auditoría independiente, más la corrección v11 de 3 ficheros de pruebas, que sigue pendiente de auditoría.
> - La consolidación de las rondas v5–v11, las huellas definitivas, las pruebas finales, el alcance tras aprobar E02 y el procedimiento A2 → P2 preparado están en `reports/mnq/TDA17_E02_cierre_formal.md`.
> - Ese documento **sustituye** a la sección 7 de este paquete. Las rondas v6–v10 solo afectan a la publicación oficial (`tda17_salidas.py`), a `verify` del cargador (v6 y v10) y a sus pruebas. La v11 solo afecta a 3 ficheros de pruebas. Ninguna de estas rondas se repite aquí.
> - Las secciones siguientes se conservan como historial de la auditoría.

**Estado:** `PENDIENTE_AUDITORIA_INDEPENDIENTE`. Es la implementación de las decisiones autorizadas el 2026-10-18 (`reports/mnq/TDA17_E02_registro_decisiones.md`).

**Lo que este documento NO hace:**
- no aprueba E02;
- no crea commits;
- no publica nada;
- no autoriza ejecutar sobre MNQ.

**Estado del repositorio:**
- HEAD = P1 `24f9f1d`, sin commits nuevos;
- E02 registrada como `PENDIENTE_AUDITORIA` con `aprobacion: null`;
- todas las rutas de producción y de ejecución oficial rechazan con `[E02_PENDIENTE]`.

No se ha leído `data/raw` ni el holdout. No se ha ejecutado ningún análisis sobre MNQ. Las pruebas que necesitan aprobación usan repositorios Git temporales con una aprobación simulada (`PRUEBA_SINTETICA`).

**DECISIÓN IRIS: NINGUNA.**

Las secciones 3 a 6 recogen los valores medidos en esta ejecución (huellas, pruebas y mutaciones).

## 0000. Versión v5: decisiones 14–19 aprobadas e implementadas

Esta sección **sustituye** a las anteriores cuando discrepan. El diff exacto respecto del estado v4 está en `correccion_v5.diff`.

**Autorización.** El responsable de IRIS aprobó metodológicamente, el 2026-10-19, las seis recomendaciones de `reports/mnq/TDA17_E02_propuesta_ambiguedades_y_salidas.md` (SHA-256 LF `f16672c3…`; el documento **no se ha modificado** después de la aprobación).
- La autorización está registrada en tres sitios:
  - en el YAML de E02 (`enmienda.autorizacion_complementaria`);
  - en `TDA17_E02_registro_decisiones.md` (§4, decisiones 14–19);
  - en el alcance del registro JSON.
- La autorización del 2026-10-18 y sus 11 operaciones se conservan **sin cambios**.
- E02 sigue `PENDIENTE_AUDITORIA` con `aprobacion: null`.

### 0000.1 Cambios en E02 (YAML)

**YAML:**
- 5 operaciones `anadir` nuevas (16 en total):

  | Decisión | Clave |
  |---|---|
  | 14 (S-a) | `analisis.spearman.validacion_replica_invalida` |
  | 15 (N-a) | `analisis.clasificacion.subniveles_r2_global_no_publicable` |
  | 16 (R-a corregida, con el anexo N → columnas) | `identidades.verificacion_numerica.relaciones_N` |
  | 17 + 18 (esquema mínimo y vinculación) | `salidas.esquema_minimo` |
  | 19 (publicación con bloqueo, §4.3) | `salidas.diagnostico_bloqueo` |

- Ninguna ruta prohibida.
- No cambia ningún valor congelado: τ, nulos esperados, umbrales, réplicas, semillas, tolerancias, catálogo, D1–D24 ni E01.

**Cargador:**
- `E02_SHA256` = `fd7478d5…`;
- `WHITELIST_E02` con las 16 rutas;
- `AUDITED_E02` añade `tests/test_tda17_salidas.py` (15 ficheros).

**Anexo:** `reports/mnq/TDA17_E02_anexo_relaciones_N.md`.

### 0000.2 Reglas implementadas (`tda17_analysis.py`)

- **S-a** (`spearman_invalid_rule`, `_validation_cell`, `spearman_validation`).
  - Una réplica INVALIDA en cualquiera de los dos métodos (VAR_CERO o NO_FINITO) deja la celda **sin mediana**, con `motivo: REPLICA_INVALIDA` y los índices de esas réplicas.
  - La validación de la celda falla en esa ronda y después se aplica F3-I-b: ronda de 100 y, si vuelve a fallar, IC de Spearman no publicados.
  - No hay imputación ni exclusión del par; ni con tolerancia infinita pasa.
- **N-a** (`subniveles`). Una columna no registrada con R² global no publicable recibe `SIN_SUBNIVEL_R2_NO_PUBLICABLE`, con motivo `R2_NO_PUBLICABLE_POR_COLINEALIDAD`. El R² se publica, sin sustituirlo.
- **R-a corregida** (`relations_n`, `relation_diagnostics`, `blocking_dossier`). Según el tipo de cada relación N:

  | Tipo | Diagnóstico |
  |---|---|
  | LA-P y NL-P | cargas de sus columnas de P en cada dirección nula observada |
  | LA-F y NL-F | `SIN_DIRECCION_EN_P` |
  | DES (N21) | recuento exacto de violaciones en la población de la celda |
  | DIAG (N10) | `NO_IDENTIDAD` |

  - El valor numérico queda `PENDIENTE_EXPEDIENTE`, por diseño aprobado.
  - El anexo se valida: ids y orden congelados, columnas existentes, N4 ⇔ L10, N21 con sus dos columnas y `nulos_esperados_adicionales: 0`.
- **`PENDIENTES`:**
  - salen `SPEARMAN_VALIDACION_REPLICA_INVALIDA`, `SUBNIVEL_R2_GLOBAL_NO_PUBLICABLE`, `RELACIONES_N_VALOR` y `SALIDAS_ESQUEMA_VINCULACION`;
  - entra `SALIDAS_OTRAS_TABLAS`: las 13 tablas de `salidas.tablas` sin esquema siguen bloqueadas.

### 0000.3 Publicación oficial (`tda17_salidas.py`, reescrito)

- **Ruta única `official_publish_th35(publicacion, root)`**, que solo recibe P2:
  1. `an.official_config(P2)`;
  2. `an.official_th35(P2)` **en el mismo proceso**: entradas verificadas contra el manifiesto y TH37 obligatoria;
  3. comprobación de que el resultado procede de esa configuración (`SALIDA_PROCEDENCIA`);
  4. tablas del esquema a partir de ese resultado;
  5. rechazo si ya existe alguna salida oficial (`SALIDA_EXISTENTE`);
  6. escritura de las tablas y de `TDA17_procedencia_oficial.json`.
- **`write_official_table` desaparece.** Ninguna función recibe una tabla, un resultado o una configuración para escribirlos.
  - `official_tables` y `diagnostic_table` son puras.
  - Una prueba estática exige que las dos únicas escrituras estén dentro de `official_publish_th35`, después del cálculo oficial.
- **Por estado:**
  - `SIN_BLOQUEO`: las 5 tablas de §4.1;
  - `BLOQUEADO_HASTA_EXPLICAR`: solo la auditoría causal y `TDA17_diagnostico_bloqueo.csv`;
  - un resultado incoherente (estado frente a bloqueos, o bloqueado con clasificación) rechaza con `SALIDA_ESTADO`.
- **Columnas exactas del esquema** (`SALIDA_COLUMNAS`).
- **Serialización determinista:** CSV `,`, LF, UTF-8, sin índice.
- **Esquema verificado contra E02** (`SALIDA_ESQUEMA`).
- **Procedencia:** `publicacion_E02`, `remoto_E02`, `sha256_efectiva`, `entradas`, `huella_marco`, `filas`, `estado`, `bloqueos` y la SHA-256 de cada tabla.
- **Verificación posterior:**
  - `check_official_publication`, sin recalcular. Comprueba el registro (`SALIDA_REGISTRO`), su P2 (`SALIDA_P2`), la configuración efectiva y el remoto vigentes (`SALIDA_EFECTIVA`), las entradas del manifiesto (`SALIDA_ENTRADAS`), el conjunto de tablas según el estado sin salidas extra (`SALIDA_CONJUNTO`), y la huella y la cabecera de cada tabla (`SALIDA_HUELLA`, `SALIDA_ESQUEMA`).
  - `verificar --referencia-git P2` la ejecuta si existe el registro.
  - `reproduce_official_publication` recalcula `official_th35` y exige igualdad byte a byte (`SALIDA_NO_REPRODUCIBLE`). Es la única comprobación que detecta una alteración coherente de tabla y registro.
- **Tablas fuera del esquema:** `other_official_table` devuelve `SALIDAS_OTRAS_TABLAS`; las del esquema por otra vía devuelven `SALIDA_RUTA_UNICA`.

### 0000.4 Garantías que siguen sin poder verificarse sin datos reales

- Las tablas oficiales **no se han generado nunca sobre MNQ**.
- Todas las publicaciones de las pruebas se hacen en repositorios Git temporales:
  - aprobación simulada `PRUEBA_SINTETICA`;
  - remoto temporal;
  - entradas sintéticas redirigidas antes de cualquier lectura.
- Que la ejecución real termine en `SIN_BLOQUEO` o en `BLOQUEADO_HASTA_EXPLICAR` solo se sabrá en la ejecución oficial, tras P2.
- El coste de `reproduce_official_publication` sobre MNQ (recalcula todo TH35) no está medido.

### 0000.5 Resultados (todas las pruebas sintéticas)

- **Suite completa estricta** (`TDA17_INTEGRIDAD=estricta`), después de todas las modificaciones de v5: **1763 passed, 2 skipped**, código 0 (2368 s; `suite_estricta_v5.log`).
- **Pruebas por fichero:**

  | Fichero | Pruebas |
  |---|---|
  | `test_tda17_enmienda_02.py` | 73 |
  | `test_tda17_e02_motor.py` | 49 |
  | `test_tda17_salidas.py` (nuevo) | 23 |
  | `test_tda17_causalidad.py` | 23 |
  | `test_tda17_procedencia.py` | 14 |
  | `test_tda17_analysis.py` | 51 |
  | `test_tda17_g2.py` | 34 |
  | `test_tda17_rango_diagnostico.py` | 20 |

- **Verificadores** (`verificadores_v5.txt`):
  - protocolo `9cdea2f`: CONFORME (9 ficheros, 5 entradas, 188 artefactos);
  - E01 `24f9f1d`: CONFORME;
  - E02 `fd7478d557e9` en BORRADOR, `PENDIENTE_AUDITORIA`, no activable ni oficial: CONFORME (efectiva `66ab24579c51`);
  - HEAD `24f9f1d`, 0 ficheros versionados modificados.
- **T-MUT v5** (`t_mut_v5.log`, `t_mut_resultados_v5.json`), sobre el código definitivo:
  - control sin mutar: 182 passed, código 0;
  - 65 controles (52 anteriores y 13 nuevos `SALIDA_*`): **64 detectados**;
  - los 13 controles `SALIDA_*` se detectan todos;
  - único superviviente: `E02_REMOTO_HISTORIA`, redundante por construcción (sección 00.1).
  - Una primera ejecución falló en el control sin mutar porque el clon no tenía los documentos autorizados, cuyas huellas comprueba `test_tda17_salidas.py`. El arnés ahora los copia; esa ejecución no es válida y no se usa.

### 0000.6 Huellas v5 (SHA-256 tras normalizar CRLF → LF)

| Fichero | SHA-256 | Líneas | Respecto de v4 |
|---|---|---|---|
| `configs/tda17_enmienda_02.yaml` | `fd7478d557e9a798c52b2af0806e89a6945c90cf5d8f35ea328afb581ad0d317` | 330 | MODIFICADO en v5 |
| `src/ohlcv_dataroad/ingest/tda17_enmiendas_e02.py` | `ab4e5e7fa6098d6eb8e238c65b8bf210c3f632b456f06210c78e7b3d15afba6e` | 548 | MODIFICADO en v5 |
| `tests/test_tda17_enmienda_02.py` | `cc1e68259f47f8a2c0e3249bcb728af22530c206b1b62c2e80f3802e918f622d` | 885 | MODIFICADO en v5 |
| `src/ohlcv_dataroad/ingest/tda17_analysis.py` | `f26bf380e0bbc2874c1939873a995513a693de6d4b848fdb8c1be88c26515fe0` | 1572 | MODIFICADO en v5 |
| `src/ohlcv_dataroad/ingest/tda17_g2.py` | `586b97b79cde7730a5d27b71935d8263387c2ae1f4111ae22564d20d2902fb6b` | 478 | sin cambios |
| `src/ohlcv_dataroad/ingest/tda17_causalidad.py` | `1f5b6f3b1348762d83c11ca6b581d04ee945049531a8a878ca0cc05cb3910e89` | 523 | sin cambios |
| `src/ohlcv_dataroad/ingest/tda17_salidas.py` | `5ab6643ccf354fbe3aa714734de46c28fc8befd9ec951e6a304f91461f2ca677` | 308 | MODIFICADO en v5 |
| `src/ohlcv_dataroad/ingest/tda17_entradas.py` | `fae54edd2773c7244a16d1e309643a7acf33d32fe9be3528c89f9f03134e84fa` | 81 | sin cambios |
| `tests/test_tda17_analysis.py` | `acf1278fe3a9013b51e6510990ac65f70057f45e648ddbb212db3b46f0d79eab` | 751 | sin cambios |
| `tests/test_tda17_g2.py` | `33f820ff9c4a66afadd333954af28243e45577a1c41753675be2dab3ea67cbcd` | 386 | sin cambios |
| `tests/test_tda17_rango_diagnostico.py` | `61aefb2dcbd80fe53ec43675ef4b353066faf1a94a4fc4840cfbccf3eb8d9ddf` | 120 | sin cambios |
| `tests/test_tda17_causalidad.py` | `a4508c40a223c7bdc153065fffd001c9efef91ca32c0d582620b7a006065baa0` | 368 | sin cambios |
| `tests/test_tda17_e02_motor.py` | `f8337823252d1353e5d52953ae5cd871c87b347b60b426c02acdb3999c870bbf` | 663 | MODIFICADO en v5 |
| `tests/test_tda17_procedencia.py` | `df3306e4fbb07e5ee8021336ea07827d56d0f78e6e521a59c434521568f9e9f3` | 284 | sin cambios |
| `tests/test_tda17_salidas.py` | `cad8572408f2ea8a7dc63f58301e86e337e3f1a5b6864750ca480036cf9a7d54` | 358 | NUEVO en v5 |
| `reports/mnq/TDA17_enmiendas_E02.json` | `698873be6d7e6fb1a19e35772e8fa045b2df30e82b493fa08feb4c6f25cc5159` | 25 | MODIFICADO en v5 |
| `reports/mnq/TDA17_E02_registro_decisiones.md` | `896acafb2937f0d45c89e931af1d47e54cef26205ecbda00353a7606bb9977b3` | 107 | MODIFICADO en v5 |
| `reports/mnq/TDA17_E02_anexo_relaciones_N.md` | `8cac7280d7dbb414bfc60909b162d0a585b954e9d5bdfb6efdf1a51c1112f7ac` | 56 | NUEVO en v5 |
| `reports/mnq/TDA17_E02_propuesta_ambiguedades_y_salidas.md` | `f16672c3ac80505e3a70a0d1ef68619dff56175da9f153bab936f0c117cb46d6` | 139 | sin cambios |
| `reports/mnq/TDA17_E02_propuesta_decisiones_normativas.md` | `192c5a43be79835d620ee7484df1b8f8242e8ef73e49a70f182b6109cb92f0a0` | 687 | sin cambios (anterior a v5; fuera de la instantánea previa, no entra en el diff) |
| `reports/mnq/TDA17_especificacion_migracion_M2.md` | `088504a0875a5c935ce3c00966bdb8f89338a4deff2fab97df4f0a6bd456ac69` | 304 | sin cambios (anterior a v5; fuera de la instantánea previa, no entra en el diff) |
| `reports/mnq/TDA17_inventario_decisiones_pendientes.md` | `b42e1c49bf9e20a8453d933d1008a2462453e64c6d305edf582df5559fd76ec9` | 583 | sin cambios (anterior a v5; fuera de la instantánea previa, no entra en el diff) |

Los 15 primeros son `AUDITED_E02`, cuyas huellas se aprobarán en A2. Las dos propuestas autorizadas no han cambiado: `f16672c3…` y `192c5a43…`.

### 0000.7 Qué sigue bloqueado

- **E02:** no está aprobada ni activada. Falta la auditoría independiente y el procedimiento A2 → P2 (sección 7), que no se ha ejecutado.
- **MNQ:** no se ha ejecutado nada sobre datos reales; no se ha leído `data/raw` ni el holdout.
- **Publicación con bloqueo:** si el estado es `BLOQUEADO_HASTA_EXPLICAR`, solo se publican la auditoría causal y `TDA17_diagnostico_bloqueo.csv` (decisión 19).
- **Decisiones sin autorizar, que mantienen bloqueada solo su funcionalidad:**
  - decisión 11 (`HUECO_ESPECTRAL`);
  - `R2_OTROS_INDICADORES` y `R2_TRANSFERENCIA`;
  - `T3_V2` y `K_PA_MAX_T_V3`;
  - `S2_ESCALAS` y `S7_S8_ALCANCE`;
  - `REGISTRO_D7_RETRASO`;
  - `SALIDAS_OTRAS_TABLAS` (las 13 tablas de `salidas.tablas` sin esquema).
- **Valor numérico de las relaciones N:** queda `PENDIENTE_EXPEDIENTE`, por diseño aprobado (decisión 16).

## 000. Ronda posterior a la auditoría favorable de v3 (sin ZIP nuevo todavía)

- **Identidad del repositorio autorizado** (`E02_REMOTO_NO_AUTORIZADO`, nuevo).
  - El cargador fija `AUTHORIZED_REMOTE = https://github.com/GUNAPILLCO/ohlcv_dataroad.git` en su propio código, que es un fichero auditado. No es un parámetro ni una variable de entorno.
  - La consulta (`ls-remote`) y la descarga (`fetch`) usan **esa URL**, no el nombre `origin`.
  - Además se exige que `origin` sea ese repositorio (normalizado: `git@`/`https`, mayúsculas, `/` y `.git` finales) y que no exista ninguna regla `url.*.insteadOf` / `pushInsteadOf` que pueda redirigir la URL.
  - La traza registra `remoto_E02 = {repositorio, ref, sha}`.
- **Pruebas nuevas:**
  - negativas:
    - remoto distinto con el mismo contenido y P2 publicado;
    - `origin` inexistente;
    - regla `insteadOf` hacia un espejo;
    - repositorio autorizado inaccesible;
  - positivas: identidad del repositorio del proyecto (solo configuración, sin red) y normalización.
  - En los repositorios temporales la identidad autorizada se fija al remoto temporal **solo dentro de la prueba** (`monkeypatch`).
- **Instrucciones `--fetch` obsoletas:** corregidas (la opción no existe desde v3).
- **Descripciones del remoto (cierre documental v4):**
  - Las secciones 00.1 y 7 ya no dicen `git ls-remote origin`, `git fetch origin` ni `origin/main`. Describen la consulta directa de `AUTHORIZED_REMOTE`.
  - El documento autorizado `TDA17_E02_propuesta_decisiones_normativas.md` (§7.2 dice «`git fetch`» de forma genérica) **no se modifica**: está fijado por su SHA-256 `192c5a43…` en el YAML de E02. La implementación cumple esa garantía consultando la URL autorizada.
- **Propuesta para decisión:** `reports/mnq/TDA17_E02_propuesta_ambiguedades_y_salidas.md` cubre las tres ambigüedades y el esquema y la vinculación de las salidas. No está implementada y todo sigue bloqueado.

## 00. Revisión v3: publicación remota real, publicación de tablas bloqueada y cierre de comprobaciones

Esta sección **sustituye** a las anteriores cuando discrepan. El diff exacto respecto de los ficheros del ZIP v2 está en `correccion_v3.diff`.

### 00.1 C1: verificación contra el remoto REAL

**Fallo.** `verify_remote_publication()` confiaba en la referencia local `refs/remotes/origin/main`, que puede estar desactualizada.

**Corrección.** La función consulta **siempre** el remoto real (redacción actualizada a v4, §000: la consulta va a la URL autorizada, no al nombre `origin`):
- `git ls-remote --exit-code <AUTHORIZED_REMOTE> refs/heads/main`, con `AUTHORIZED_REMOTE = https://github.com/GUNAPILLCO/ohlcv_dataroad.git` y el tiempo máximo de 60 s del acceso a Git de E01;
- usa el SHA devuelto, nunca la referencia local;
- si ese commit no está en el repositorio local, lo descarga con `git fetch --quiet <AUTHORIZED_REMOTE> refs/heads/main` para comprobar la historia.

**Fallo seguro.** Cualquier problema rechaza con `E02_REMOTO`: repositorio autorizado inaccesible (GitHub caído o sin red), rama borrada, respuesta mal formada o descarga fallida. Desde v4, `origin` inexistente o distinto del autorizado, o cualquier regla `url.*.insteadOf` / `pushInsteadOf`, rechazan antes de consultar, con `E02_REMOTO_NO_AUTORIZADO`.

**Otros cambios:**
- La opción `--fetch` desaparece: la consulta remota es obligatoria en toda ejecución oficial.
- La traza registra `remoto_E02 = {repositorio: <AUTHORIZED_REMOTE>, ref: "refs/heads/main", sha: <SHA remoto real>}` (formato v4).

**Pruebas** (remoto desnudo temporal real, nunca GitHub ni el repositorio del proyecto):
- la referencia local indica P2 publicado, pero el remoto real está en A2: rechazo;
- rama remota borrada: rechazo;
- remoto inexistente: rechazo;
- `origin` eliminado: rechazo (desde v4, con `E02_REMOTO_NO_AUTORIZADO`);
- otro clon publica un commit posterior que no está en local: con la descarga impedida, rechazo; con la descarga, aceptación y la traza registra ese SHA;
- positiva con HEAD = P2.

**`E02_REMOTO_HISTORIA`, redundancia formal.** Sea R el SHA remoto real.
- `E02_REMOTO` garantiza que P2 es antecesor de R.
- `E02_P2_A2` garantiza que A2 es antecesor estricto de P2.
- `E02_ASCENDENCIA` garantiza que P1 es antecesor de A2.

Como la relación de ascendencia de Git es transitiva, A2 y P1 son antecesores de R. Por tanto, con los controles anteriores activos, no existe un estado del repositorio que alcance `E02_REMOTO_HISTORIA`, y su mutante no puede ser detectado por ninguna prueba. Se conserva como defensa en profundidad. Los controles efectivos de historia remota se detectan en T-MUT y sus positivas pasan.

### 00.2 C2: publicación de tablas oficiales BLOQUEADA

**Fallo.** `write_official_table()` aceptaba un DataFrame arbitrario con un nombre declarado.

**Corrección.** Tras verificar la configuración oficial (re-derivada con M2) y el nombre declarado, la primitiva lanza `PendingDefinitionError('SALIDAS_ESQUEMA_VINCULACION')` y **no escribe ningún fichero**; ya no contiene ninguna operación de escritura. No se inventa ningún esquema: el protocolo no fija el de las tablas de `salidas.tablas`, ni existe vinculación verificable con `official_th35`.

**Pruebas:**
- negativa con la configuración oficial **verificada** en el repositorio temporal: tabla TH35 declarada, bloqueo exacto y fichero inexistente;
- estática: el módulo no contiene `to_csv`, `to_parquet`, `write_*` ni `open(`.

### 00.3 Resultados v3

- **Suite completa estricta, después de TODAS las modificaciones de v3:** **1724 passed, 2 skipped**, codigo 0 (1364 s; `suite_estricta_v3.log`).
  - Una ejecucion anterior en paralelo con T-MUT se interrumpio al cerrarse la sesion: `suite_estricta_v3_interrumpida.log`, conservada como evidencia y no valida.
  - Tras la interrupcion, los 70 tests del cargador pasan solos y la suite completa repetida pasa sin errores.
- **Pruebas por fichero:** `test_tda17_enmienda_02.py` 70, `test_tda17_e02_motor.py` 36, `test_tda17_causalidad.py` 23, `test_tda17_procedencia.py` 14, fase 2 adaptada 105, E01 sin modificar 303.
- **Verificadores** (`verificadores_v3.txt`):
  - protocolo `9cdea2f`: CONFORME (9 ficheros, 5 entradas, 188 artefactos);
  - E01 `24f9f1d`: CONFORME;
  - E02 en BORRADOR, `PENDIENTE_AUDITORIA`, no activable ni oficial: CONFORME;
  - HEAD `24f9f1d`, 0 ficheros versionados modificados.
- **T-MUT v3** (`t_mut_v3.log`, `t_mut_resultados_v3.json`), sobre el codigo definitivo:
  - control sin mutar: 143 passed, codigo 0;
  - 51 controles (44 del cargador, 4 TH37 y 3 de entradas): **50 detectados**;
  - `ENTRADA_NO_DECLARADA` queda **detectado** (pendiente en v2);
  - unico superviviente: `E02_REMOTO_HISTORIA`, redundante por construccion (seccion 00.1);
  - `E02_REMOTO` y los controles de historia efectivos (`E02_P2_A2`, `E02_ASCENDENCIA`) se detectan.
  - Una primera ejecucion de T-MUT v3 se interrumpio porque su control supero el limite de 1800 s del arnes al correr en paralelo: `t_mut_v3_interrumpido.log`. El limite pasa a 3600 s.

**Huellas v3** (SHA-256, LF):

| Fichero | SHA-256 | Lineas | Respecto de v2 |
|---|---|---|---|
| `configs/tda17_enmienda_02.yaml` | `5471ec30aa0df02d72cb379a697ed39f4d69a29825eda1b6b547d9a3a96bcfe2` | 199 | sin cambios |
| `reports/mnq/TDA17_enmiendas_E02.json` | `31013b33ada38cf796d9a018299016ce3f59ec24bf1b8c377a7ea21095bba0d4` | 25 | sin cambios |
| `src/ohlcv_dataroad/ingest/tda17_enmiendas_e02.py` | `6038927ff20a91d0794a058d52526d1068446d63a4a148157f47817816d428da` | 500 | MODIFICADO en v3 |
| `src/ohlcv_dataroad/ingest/tda17_analysis.py` | `e404e71572d3b6b2183b95131e8419b9aa74e2bd97ca2b170a35e1096b6326a4` | 1498 | MODIFICADO en v3 |
| `src/ohlcv_dataroad/ingest/tda17_g2.py` | `586b97b79cde7730a5d27b71935d8263387c2ae1f4111ae22564d20d2902fb6b` | 478 | sin cambios |
| `src/ohlcv_dataroad/ingest/tda17_causalidad.py` | `1f5b6f3b1348762d83c11ca6b581d04ee945049531a8a878ca0cc05cb3910e89` | 523 | sin cambios |
| `src/ohlcv_dataroad/ingest/tda17_salidas.py` | `7191ddd6fc051c3be72b887771999b9d6572781e89cb5cf69524777800dc1842` | 30 | MODIFICADO en v3 |
| `src/ohlcv_dataroad/ingest/tda17_entradas.py` | `fae54edd2773c7244a16d1e309643a7acf33d32fe9be3528c89f9f03134e84fa` | 81 | sin cambios |
| `tests/test_tda17_enmienda_02.py` | `bb6e09b6a411027706c7af86f322ce4584a1d067962b4d866aa6ac744ae97fd6` | 839 | MODIFICADO en v3 |
| `tests/test_tda17_e02_motor.py` | `26cb681b7b4e97b0dc307e9aa12f8b1a86796b04be1bd0405b49e87eb3b17ecb` | 560 | MODIFICADO en v3 |
| `tests/test_tda17_causalidad.py` | `a4508c40a223c7bdc153065fffd001c9efef91ca32c0d582620b7a006065baa0` | 368 | sin cambios |
| `tests/test_tda17_procedencia.py` | `df3306e4fbb07e5ee8021336ea07827d56d0f78e6e521a59c434521568f9e9f3` | 284 | sin cambios |
| `tests/test_tda17_analysis.py` | `acf1278fe3a9013b51e6510990ac65f70057f45e648ddbb212db3b46f0d79eab` | 751 | sin cambios |
| `tests/test_tda17_g2.py` | `33f820ff9c4a66afadd333954af28243e45577a1c41753675be2dab3ea67cbcd` | 386 | sin cambios |
| `tests/test_tda17_rango_diagnostico.py` | `61aefb2dcbd80fe53ec43675ef4b353066faf1a94a4fc4840cfbccf3eb8d9ddf` | 120 | sin cambios |

## 0. Revisión v2: correcciones C1, C2 y procedencia (auditoría GPT del ZIP v1)

Esta sección **sustituye** a las huellas (§3), los resultados (§4) y el T-MUT (§5) de la versión v1 cuando discrepan. El diff exacto respecto de los ficheros del ZIP v1 está en `correccion_v2.diff`.

### 0.1 C1: escritura oficial no falsificable

**Fallo.** `require_e02_oficial(cfg)` aceptaba un diccionario fabricado con estados APROBADA, `activable`, `ejecucion_oficial` y un SHA de P2 sintácticamente válido.

**Corrección.** `require_e02_oficial(cfg, root)`:
1. comprueba los indicadores;
2. **deriva de nuevo** la configuración oficial con `official_config_e02(root, P2)`: Git, huellas, aprobación, publicación, protocolo estricto y remoto;
3. exige que su huella canónica sea idéntica a la de la recibida.

Hay dos rechazos nuevos:
- `E02_OFICIAL_NO_VERIFICABLE`: la configuración no puede derivarse; el mensaje incluye el motivo interno.
- `E02_OFICIAL_DISTINTA`: la configuración se deriva, pero no es idéntica a la recibida.

`write_official_table` y `official_config` lo usan con la raíz del repositorio.

**Regresiones**, en las que nada se escribe:
- diccionario fabricado con P2 inexistente;
- configuración preparatoria con indicadores añadidos;
- configuración oficial con contenido alterado;
- P2 = A2;
- configuración oficial obsoleta tras volver E02 a pendiente;
- raíz que no es un repositorio.

### 0.2 C2: TH37 como barrera obligatoria de TH35

**Fallo.** `official_classification(..., bloqueos, ...)` aceptaba una lista vacía aunque TH37 no se hubiera ejecutado o tuviera discrepancias.

**Corrección.**
- Se retiran `official_classification` y `official_cell_results`.
- `th35_pipeline(df, cfg)` **ejecuta** la auditoría causal completa de su entrada y la **verifica** (`verify_causal_audit`) **antes** de calcular el catálogo. La tabla debe contener exactamente las filas del plan predeclarado (`expected_audit_rows`, derivado del esqueleto y del catálogo, nunca de la tabla), sin repetidas ni ajenas, con NO_APLICABLE solo donde el plan lo prevé y sin DISCREPANCIA.
- Rechazos: `TH37_AUSENTE`, `TH37_INCOMPLETA`, `TH37_FUERA_DEL_PLAN` y `TH37_DISCREPANCIA` (este último con `BlockedUntilExplainedError`).
- No existe ningún parámetro de bloqueos ni de resultados de TH37.

**Pruebas negativas:**
- auditoría ausente: `None`, vacía o de otro tipo;
- incompleta: falta una fila, faltan todas las de ROLL, esquema distinto, resultado incoherente o auditoría de otra entrada;
- repetida o ajena al plan;
- con discrepancias.

En cada una se comprueba además que el catálogo no llega a calcularse.

### 0.3 Procedencia de los datos de la ejecución oficial

**Hueco.** La ejecución oficial aceptaba la entrada, `per_cell` y `r2_global` del llamante.

**Barrera implementada:**
1. **Entradas.** El módulo nuevo `tda17_entradas.py` es la única vía de lectura.
   - `load_official_inputs(cfg, root)` exige primero la configuración oficial verificada (C1).
   - Lee los dos ficheros de TDA-04 declarados en `sha256_entradas` del manifiesto congelado.
   - Calcula la huella sobre **los mismos bytes** que interpreta, sin ventana entre verificar y leer, y rechaza con `ENTRADA_PROCEDENCIA` si no coincide.
   - Rechaza con `ENTRADA_PROHIBIDA` las rutas de `data/raw` o del holdout, y con `ENTRADA_NO_DECLARADA` una entrada que no esté en el manifiesto.
   - Une variables y máscara, y comprueba `entradas.esperado`.
   - Devuelve la procedencia: huellas de los ficheros, huella del marco unido, P2 y `sha256_efectiva`.
2. **Derivación interna.** `official_th35(publicacion, root)` y `official_causal_audit(publicacion, root)` **solo reciben P2**. Todo lo demás lo calcula `th35_pipeline` a partir de la entrada verificada, en este orden: TH37, catálogo y panel P por la vía de producción de la fase 1, población de casos completos, `analyze_cell` y `cell_bootstrap` por celda, `per_cell` y R² global derivados, bloqueos de todas las celdas, validación de Spearman, clasificación y subniveles.
   - Si alguna celda de validación de Spearman (GLOBAL, ANIO_2024) no tiene matriz definida, la validación es `NO_EJECUTABLE` y los IC de Spearman **no se publican**.
   - Con cualquier bloqueo, el resultado es `BLOQUEADO_HASTA_EXPLICAR`, sin clasificación ni subniveles.
3. **Pruebas estáticas.** Toda ruta `official_*` tiene exactamente los parámetros `(publicacion, root)`. Las rutas de datos llaman a `ent.load_official_inputs(cfg, root)`. Ningún módulo de la fase 2 llama al cargador de E01 para producir resultados ni a la vía privada `._compute(`.
4. **Pruebas sintéticas.** Las rutas de entrada se redirigen a ficheros parquet sintéticos en un repositorio temporal con P2. Una guarda comprueba que nunca apuntan a `data/interim`. Las pruebas cubren:
   - positiva con procedencia;
   - huella distinta de la del manifiesto;
   - fichero alterado tras fijar su huella;
   - recuentos esperados de MNQ;
   - ninguna lectura sin configuración oficial verificada;
   - `official_th35` recibe solo el marco verificado (huella), y una entrada alterada no llega al orquestador;
   - `official_causal_audit` sobre la entrada verificada;
   - orquestador bloqueado: celdas, `per_cell` derivada, Spearman no publicado y TH37 verificada;
   - rama **sin bloqueos**: población en una sola matriz y registro de la *copia* de prueba fijado a los recuentos sintéticos; clasificación de las 134 columnas, todas INESTABLE porque solo hay 1 de 6 años.

### 0.4 Garantías que NO pueden verificarse sin datos reales (no se simulan)

| Garantía | Estado |
|---|---|
| Lectura positiva de las **entradas reales** (`tda04_variables_1m.parquet`, `tda04_return_validity_mask.parquet`) con su huella del manifiesto, `entradas.esperado` sobre MNQ y `th35_pipeline` sobre MNQ | **No ejercitada** (prohibido abrir MNQ). Implementada y probada solo con entradas sintéticas redirigidas. Se ejercitará por primera vez en la ejecución oficial autorizada, tras P2. Cualquier diferencia de huella o de recuentos rechaza (fallo seguro) |
| Rama SIN_BLOQUEO con el registro real (`nulos_esperados` congelados) | Probada solo con una copia de prueba del registro. Con datos sintéticos el catálogo produce cuasidependencias no registradas y el orquestador bloquea, como debe |
| Validación de Spearman sobre GLOBAL y ANIO_2024 dentro del orquestador | Mecánica probada aparte (`spearman_validation`). En el orquestador sintético, ANIO_2024 no tiene datos: `NO_EJECUTABLE`, IC de Spearman no publicados |
| **Publicación de las tablas TH35** (qué columnas van a `TDA17_clasificacion_final.csv`, `TDA17_estabilidad_celdas.csv`, …) | **BLOQUEADA**: el protocolo no fija el esquema de esas tablas. No se implementa ninguna ruta que las escriba. `write_official_table` es solo una primitiva: exige la configuración oficial verificada (C1), pero no puede ligar criptográficamente el **contenido** de una tabla a su cálculo |
| Entradas en memoria fuera de las rutas oficiales | Las funciones de bajo nivel (`th35_pipeline`, `run_causal_audit`, …) siguen aceptando datos para las pruebas; **no** producen resultados oficiales. Las rutas oficiales no aceptan datos |

### 0.5 Pruebas, verificadores y T-MUT de v2

- **Suite completa estricta:** **1721 passed, 2 skipped**, código 0 (1301 s; log `suite_estricta_v2.log`).

| Fichero de pruebas | Pruebas |
|---|---|
| `test_tda17_enmienda_02.py` | 68 |
| `test_tda17_e02_motor.py` | 36 |
| `test_tda17_causalidad.py` | 23 |
| `test_tda17_procedencia.py` (nuevo) | 13 (+2 añadidas después de la suite) |
| `test_tda17_analysis.py` | 51 |
| `test_tda17_g2.py` | 34 |
| `test_tda17_rango_diagnostico.py` | 20 |
| E01, sin modificar | 303 (117 + 186) |

- **Verificadores** (`verificadores_v2.txt`):
  - protocolo `9cdea2f`: CONFORME;
  - E01 `24f9f1d`: CONFORME;
  - E02 en BORRADOR, `PENDIENTE_AUDITORIA`, no activable ni oficial: CONFORME;
  - HEAD `24f9f1d`, 0 ficheros versionados modificados.
- **T-MUT v2** (`t_mut_v2.log`, `t_mut_resultados_v2.json`): control sin mutar 140 passed (codigo 0); 51 controles (44 del cargador, 4 TH37, 3 de entradas); **49 detectados**. Supervivientes: `E02_REMOTO_HISTORIA` (inalcanzable por construccion, documentado) y `ENTRADA_NO_DECLARADA`, que no tenia prueba negativa: se anadio despues (`test_official_inputs_are_the_manifest_entries_or_rejected`, en verde), pero **el mutante NO se ha vuelto a ejecutar** (limite de uso): queda PENDIENTE confirmar su deteccion. Los 4 controles TH37 y `E02_OFICIAL_NO_VERIFICABLE`/`E02_OFICIAL_DISTINTA` se detectan. La suite completa (1721 passed) se ejecuto ANTES de anadir esa prueba (+2 pruebas: 15 en test_tda17_procedencia.py)

### 0.6 Huellas v2 (SHA-256, LF)

| Fichero | SHA-256 | Líneas |
|---|---|---|
| `configs/tda17_enmienda_02.yaml` (sin cambios) | `5471ec30aa0df02d72cb379a697ed39f4d69a29825eda1b6b547d9a3a96bcfe2` | 199 |
| `reports/mnq/TDA17_enmiendas_E02.json` (sin cambios) | `31013b33ada38cf796d9a018299016ce3f59ec24bf1b8c377a7ea21095bba0d4` | 25 |
| `src/ohlcv_dataroad/ingest/tda17_enmiendas_e02.py` | `b69175624eec7772c07c5a7b76d0e1ccd9faf40569f88237a0e80a80ee4b02d5` | 496 |
| `src/ohlcv_dataroad/ingest/tda17_analysis.py` | `a1b7c34cd7f84536ec58f75331bd0b8dbb54087686efb8acf4ff65b5ffd7e3a1` | 1495 |
| `src/ohlcv_dataroad/ingest/tda17_g2.py` (sin cambios) | `586b97b79cde7730a5d27b71935d8263387c2ae1f4111ae22564d20d2902fb6b` | 478 |
| `src/ohlcv_dataroad/ingest/tda17_causalidad.py` | `1f5b6f3b1348762d83c11ca6b581d04ee945049531a8a878ca0cc05cb3910e89` | 523 |
| `src/ohlcv_dataroad/ingest/tda17_salidas.py` | `4bf2230939e46d4d6a7384c86aa64e93999a028c5202dc23c5ecc84a52348dd3` | 32 |
| `src/ohlcv_dataroad/ingest/tda17_entradas.py` (nuevo) | `fae54edd2773c7244a16d1e309643a7acf33d32fe9be3528c89f9f03134e84fa` | 81 |
| `tests/test_tda17_enmienda_02.py` | `f3f547001712e8321535e72888dc5225c35511235b6f03e6fc67b063cab9b144` | 803 |
| `tests/test_tda17_e02_motor.py` | `cf455dfd3e8cbeb1b1ad3b43faeae40ffa766809b59a666231b138e2caae78bd` | 557 |
| `tests/test_tda17_causalidad.py` | `a4508c40a223c7bdc153065fffd001c9efef91ca32c0d582620b7a006065baa0` | 368 |
| `tests/test_tda17_procedencia.py` (nuevo) | `ac614e839b19c4c0118cfbc6d4dc7d715a2f86c32f89ecc24cb6d5427e59b7fa` | 271 |
| `tests/test_tda17_analysis.py` (sin cambios) | `acf1278fe3a9013b51e6510990ac65f70057f45e648ddbb212db3b46f0d79eab` | 751 |
| `tests/test_tda17_g2.py` (sin cambios) | `33f820ff9c4a66afadd333954af28243e45577a1c41753675be2dab3ea67cbcd` | 386 |
| `tests/test_tda17_rango_diagnostico.py` (sin cambios) | `61aefb2dcbd80fe53ec43675ef4b353066faf1a94a4fc4840cfbccf3eb8d9ddf` | 120 |
| `reports/mnq/TDA17_E02_registro_decisiones.md` (sin cambios) | `62145ab13e602107235b2667206f4b79976074bce478f741cbc84a12871a819f` | 80 |

`AUDITED_E02` pasa a **14 ficheros**: se añaden `tda17_entradas.py` y `tests/test_tda17_procedencia.py`. El YAML de E02, el registro, los 8 ficheros de E01 y el protocolo no cambian.

## 1. Qué se implementó

| Pieza | Fichero | Contenido |
|---|---|---|
| Contenido normativo de E02 | `configs/tda17_enmienda_02.yaml` | 11 operaciones `anadir` (decisiones 1–10 y 12). No sustituye nada y no toca ninguna ruta prohibida (SPEC-M2 §7). La decisión 11 está aplazada y no tiene clave. Única cifra nueva: la semilla 3717 |
| Registro de E02 | `reports/mnq/TDA17_enmiendas_E02.json` | `PENDIENTE_AUDITORIA`, `aprobacion: null`. Su `requiere` cita la entrada E01 (`3150ea3a…`), A1 y P1 |
| Cargador escalonado M2 | `src/ohlcv_dataroad/ingest/tda17_enmiendas_e02.py` | Controles de SPEC-M2 §6 y §6.1 con identificadores `[ID]`. Además: la verificación estricta del protocolo (`PROTOCOLO_ESTRICTO`, `PROTOCOLO_GIT`) y la publicación remota de P2 (`E02_REMOTO`, `E02_REMOTO_HISTORIA`) **dentro** de la ruta oficial (checklist 1 y 2); `require_e02` (`E02_REQUERIDA`) y `require_e02_oficial` (`E02_NO_OFICIAL`) (checklist 3); la herramienta `verificar [--referencia-git P2]`. **Sin `_DISABLED`** ni ningún otro mecanismo de omisión (checklist 4) |
| Motor de la fase 2 | `src/ohlcv_dataroad/ingest/tda17_analysis.py` | A1-a: autovalores normativos y singulares informativos. A4-a: `rank_scope`. F5-a: `undefined_columns`, matriz NO_DEFINIDA. A2-a: `blocking_dossier`, `null_directions`, `BlockedUntilExplainedError`, `require_unblocked`. B1-a: `interval_quantiles`, tipo 7. B2-a + B6: `bootstrap_stream_cells`, `cell_generator`, `cell_draws`. B3-a + F2-a: `r2_official_checked`, `bootstrap_r2`, VAR_CERO exacta por mínimo y máximo diarios. Spearman con rangos fijos (`bootstrap_spearman_fixed`) y validación F3-I + F3-I-b (`spearman_validation`, `spearman_publication`). B13-a: `classification_usable`, `non_publishable_counts`, `classify_columns`. B9: `subniveles`. Panel L informativo (`panel_l_rank_checks`). Rutas `production_config`, `official_config`, `official_cell_results`, `official_classification` |
| Auditoría causal TH37 | `src/ohlcv_dataroad/ingest/tda17_causalidad.py` (nuevo) | D17-1a: plan de cortes predeclarado (eventos ±1, cuantiles, 10 aleatorios con `SeedSequence(3717).spawn`), truncamiento y perturbación por contrato con igualdad bit a bit, D7 global, variables de TDA-04, origen de parámetros, roll 3a (precio y volumen), tabla con el esquema de E02 y consecuencias. Ruta `official_causal_audit` |
| G2 | `src/ohlcv_dataroad/ingest/tda17_g2.py` | Control de nulos de las réplicas INFORMATIVO (`g2_replica_rank_check`). Ruta oficial `official_g2_controls`, bloqueada en lo pendiente (k_PA, max-T, V3) tras verificar la configuración oficial |
| Escritura oficial | `src/ohlcv_dataroad/ingest/tda17_salidas.py` (nuevo) | Única vía de escritura: `require_e02_oficial(cfg)`, tabla declarada en `salidas.tablas`, sin sobrescribir |
| Pruebas | `tests/test_tda17_enmienda_02.py`, `tests/test_tda17_e02_motor.py`, `tests/test_tda17_causalidad.py` (nuevas); `tests/test_tda17_analysis.py`, `tests/test_tda17_g2.py`, `tests/test_tda17_rango_diagnostico.py` (adaptadas a E02) | §4 |
| Registro de decisiones | `reports/mnq/TDA17_E02_registro_decisiones.md` | Autorización, matriz, lecturas de implementación y funcionalidad bloqueada |

### 1.1 Cambios en los módulos de la fase 2 respecto de su versión auditada C1–C4

No se conservaron copias de las versiones auditadas (`be07113c…`, `3b8242eb…`, `6ddabb03…`, `8b08f469…`, `07feb8f9…`), por lo que no hay un *diff* textual. Los cambios, función a función:

- `tda17_analysis.py`:
  - **Cabecera y `PENDIENTES`:** se retiran `IC_NIVEL`, `BOOTSTRAP_CELDAS`, `SPEARMAN_VALIDACION`, `CONVENCION_NULOS` y `SUBNIVEL_R2`. Se añaden `SPEARMAN_VALIDACION_REPLICA_INVALIDA`, `SUBNIVEL_R2_GLOBAL_NO_PUBLICABLE` y `RELACIONES_N_VALOR`. Se redacta de nuevo `HUECO_ESPECTRAL`.
  - **Funciones y clases nuevas:** `BlockedUntilExplainedError`, `e02_value`, `production_config`, `official_config`, `normalized_matrix`, `check_null_convention`, `rank_scope`, `undefined_columns`, `null_directions`, `blocking_dossier`, `publish_exact_by_cell`, `bootstrap_stream_cells`, `cell_generator`, `cell_draws`, `replica_zero_variance`, `interval_quantiles`, `registered_columns`, `r2_official_checked`, `bootstrap_r2`, `bootstrap_spearman_fixed`, `day_rows`, `exact_rank_replica_correlation`, `spearman_validation`, `spearman_publication`, `cell_bootstrap`, `classification_usable`, `non_publishable_counts`, `require_unblocked`, `subniveles`, `classify_columns`, `panel_l_rank_checks`, `official_cell_results`, `official_classification`.
  - **Funciones modificadas:**
    - `rank_check` exige el `ambito`; compara solo los autovalores y añade `singulares_igual_registro`.
    - `null_counts`: misma cuenta, nueva documentación.
    - `blocking_findings` solo bloquea en `BLOQUEAR_HASTA_EXPLICAR`.
    - `bootstrap_day_draws` recibe un `Generator`, no una semilla.
    - `DailyMoments` y `daily_moments` añaden `dmin`, `dmax` y `codes`.
    - `percentile_interval` recibe los cuantiles explícitos (tipo 7).
    - `classify_column` aplica B13-a.
    - `analyze_cell` exige `celda=` y acepta `sensibilidad=`; trata F5-a y emite `diagnostico_bloqueo`.
  - **Funciones eliminadas:** `production_interval_level`, sustituida por `interval_quantiles`.
- `tda17_g2.py`: el docstring, la importación y `g2_replica_rank_check` y `official_g2_controls` añadidos.
- Pruebas adaptadas:
  - la configuración de prueba pasa a la efectiva E01 + E02 en BORRADOR;
  - `analyze_cell(..., celda="GLOBAL")`;
  - generadores PCG64;
  - las aserciones de la convención de nulos siguen A1-a (los singulares ya no bloquean);
  - `_cells` incluye `ic_estado` y `estado_matriz`;
  - los puntos resueltos dejan de figurar como pendientes.

## 2. Qué sigue bloqueado

- **Decisiones no autorizadas:**
  - B5;
  - B8 (`R2_TRANSFERENCIA`);
  - B10 (`R2_OTROS_INDICADORES`);
  - B11 (`T3_V2`);
  - B12 (`K_PA_MAX_T_V3`, uso de G2 en producción);
  - C1–C3 (`S2_ESCALAS`: S2 bloqueada);
  - D2–D16 salvo lo determinado por las fuentes (incluido `S7_S8_ALCANCE`);
  - `REGISTRO_D7_RETRASO`.
- **Aplazada:** decisión 11, `HUECO_ESPECTRAL`. La publicación de EXACTA por celda está bloqueada.
- **Ambigüedades de lo autorizado, aisladas sin inventar criterio:** `SPEARMAN_VALIDACION_REPLICA_INVALIDA`, `SUBNIVEL_R2_GLOBAL_NO_PUBLICABLE` y `RELACIONES_N_VALOR`.
- **Todo E02:** producción y ejecución oficial rechazan con `[E02_PENDIENTE]` hasta el ciclo A2 → P2 y la auditoría independiente.

## 3. Ficheros y huellas (SHA-256 tras normalizar CRLF → LF)

| Fichero | SHA-256 | Líneas | En `AUDITED_E02` |
|---|---|---|---|
| `configs/tda17_enmienda_02.yaml` | `5471ec30aa0df02d72cb379a697ed39f4d69a29825eda1b6b547d9a3a96bcfe2` (fijada en `E02_SHA256`) | 199 | sí |
| `src/ohlcv_dataroad/ingest/tda17_enmiendas_e02.py` | `e27b40c7a4445ddadbab7c15668c03ba5ff405fdc2fac47e5dc3476f2baf3078` | 485 | sí |
| `src/ohlcv_dataroad/ingest/tda17_analysis.py` | `7aa21409a0852706ab6a58ba3809b4ca22d3c6e4921dbed0ffb95ea155cee513` | 1427 | sí |
| `src/ohlcv_dataroad/ingest/tda17_g2.py` | `586b97b79cde7730a5d27b71935d8263387c2ae1f4111ae22564d20d2902fb6b` | 478 | sí |
| `src/ohlcv_dataroad/ingest/tda17_causalidad.py` | `3fadbdd934f7a5f32f9b24f3ba078da28cd07d2ee6a85ab9343536c2ac7f872a` | 432 | sí |
| `src/ohlcv_dataroad/ingest/tda17_salidas.py` | `c76d24ed6e91583d8eb4638151a5dd0244f3feb333634f4a53a0397b66f970b7` | 29 | sí |
| `tests/test_tda17_enmienda_02.py` | `22068a87b3907b53babc0ca55af3ee9ec5bcf4298dd33b2a720c1cb0e95806cf` | 770 | sí |
| `tests/test_tda17_e02_motor.py` | `45aa848fe8651d99cfa72b804b1259e4fafc1d5162cd082017f70a4faeefdfbd` | 541 | sí |
| `tests/test_tda17_causalidad.py` | `e431177cd77cd0e556bceb6ab45fe01e6ee77eb5f14a0f0ef984817233639c71` | 297 | sí |
| `tests/test_tda17_analysis.py` | `acf1278fe3a9013b51e6510990ac65f70057f45e648ddbb212db3b46f0d79eab` | 751 | sí |
| `tests/test_tda17_g2.py` | `33f820ff9c4a66afadd333954af28243e45577a1c41753675be2dab3ea67cbcd` | 386 | sí |
| `tests/test_tda17_rango_diagnostico.py` | `61aefb2dcbd80fe53ec43675ef4b353066faf1a94a4fc4840cfbccf3eb8d9ddf` | 120 | sí |
| `reports/mnq/TDA17_enmiendas_E02.json` (registro, seguido, no auditado) | `31013b33ada38cf796d9a018299016ce3f59ec24bf1b8c377a7ea21095bba0d4` | 25 | no |
| `reports/mnq/TDA17_E02_registro_decisiones.md` | `62145ab13e602107235b2667206f4b79976074bce478f741cbc84a12871a819f` | 80 | no |

**Sin cambios** (comprobado):
- los 8 ficheros de E01 (huellas de SPEC-M2 §3.1);
- el protocolo (9 ficheros, manifiesto `1b14b65a…`);
- la propuesta (`192c5a43…`), la especificación M2 (`088504a0…`) y el inventario (`b42e1c49…`);
- ningún fichero versionado modificado; HEAD = `24f9f1d`.

## 4. Resultados de las pruebas (todas sintéticas)

| Ejecución | Resultado |
|---|---|
| Suite completa estricta (`TDA17_INTEGRIDAD=estricta python -m pytest -p no:cacheprovider -q`) | **1702 passed, 2 skipped**, código 0 (972 s). Antes de E02: 1581 passed, 2 skipped |
| `tests/test_tda17_enmienda_02.py` (cargador M2) | 67: T-CMP (comparador: tipo y motivo exactos, 7 casos); T-POS; T-OFI positivas (HEAD = P2 y descendiente; preparatoria en A2); T-E01; T-GIT-P2; T-REP (A2 y P2 idénticos en dos construcciones); T-WL; 40 negativas de carga, 5 de lista blanca en borrador, `E02_PENDIENTE_CON_DATOS`, `E02_RUTAS`, `GIT` (sin repositorio, repositorio padre); T-OFI negativas (`E02_P2_*`, `PROTOCOLO_ESTRICTO`, `PROTOCOLO_GIT`, `E02_REMOTO`); prueba estática de identificadores y orden de la ruta oficial |
| `tests/test_tda17_e02_motor.py` (reglas en el motor) | 36 |
| `tests/test_tda17_causalidad.py` (TH37) | 18, incluidas fugas inyectadas: del final del array (solo la detecta el truncamiento), de la barra siguiente con peso 1e-15 (pandas la acepta con su tolerancia por defecto; la auditoría no), global, en D7 y en las variables de TDA-04; origen de parámetros; ajuste de roll |
| Fase 2 adaptada (`analysis` 51, `g2` 34, `rango_diagnostico` 20) | 105 |
| Pruebas auditadas de E01 (`test_tda17_enmienda_01.py` 117 + `test_tda17_indicators.py` 186) | **303, sin modificar**, en verde |

**Incidencia corregida durante la implementación.**
- La prueba auditada de E01 `test_c1_only_tests_use_the_synthetic_path` detectó que la primera versión de `tda17_causalidad.py` llamaba a `ind._compute(` (vía privada).
- Se corrigió el módulo, **no** la prueba:
  - la auditoría usa la vía de producción `ind.compute_catalog`;
  - `check_catalog_config` exige que el catálogo de la configuración sea el de E01;
  - la prueba estática del motor admite solo ese uso del cargador de E01.

**Verificadores en el repositorio del proyecto:**
- `tda17_protocol_freeze verificar --referencia-git 9cdea2f`: CONFORME (9 ficheros, 5 entradas, 188 artefactos).
- `tda17_enmiendas verificar --referencia-git 24f9f1d`: CONFORME (E01 APROBADA, activable).
- `tda17_enmiendas_e02 verificar`: CONFORME en **BORRADOR** (E02 PENDIENTE_AUDITORIA, `activable: NO`, `ejecucion oficial: NO`).

## 5. T-MUT: mutaciones de código fuente sobre clones Git

**Arnés** (`t_mut.py`, en el directorio de trabajo de la sesión):
- Para cada uno de los **42** identificadores de control del cargador, el mutante sustituye todas sus llamadas `_fail("ID", …)` por una función que no lanza.
- Cada mutante se ejecuta en un clon Git (historia real + ficheros de E02), con `PYTHONPATH` = clon; se comprueba que el módulo se importa del clon.
- Las pruebas construyen sus propios A2 y P2 a partir del clon mutante, con aprobaciones recalculadas.

**Resultados:**
- **Control sin mutar:** 103 passed, código 0.
- **Detectados:** **41/42**.
- **Superviviente:** `E02_REMOTO_HISTORIA`. Es redundante por construcción: si P2 es alcanzable desde la rama remota (`E02_REMOTO`), también lo son A2 y P1, que son antecesores de P2 por `E02_P2_A2` y `E02_ASCENDENCIA`. No existe un estado del repositorio que llegue a ese control. Se conserva como defensa en profundidad.

**Ejecuciones:**
- Primera ronda, sobre la versión anterior a la corrección de la incidencia: mismo resultado.
- Ronda final sobre las huellas de §3: `t_mut_final.log`, `t_mut_resultados_final.json`.

**Controles que solo alcanzan sus negativas tras fijar con `monkeypatch` la barrera anterior.** Son redundancias documentadas: `E01_REGISTRO`, `E01_ENTRADA` y `E01_A1` (tras `E01_FICHEROS`), `E02_DOC`, `E02_LISTA_BLANCA` y `E02_ANADIR` (tras `E02_HUELLA` y `E02_ARBOL`), `E02_P2_APROBADA` (tras `E02_P2_ARBOL`) y `PROTOCOLO_GIT` (tras `PROTOCOLO_ESTRICTO`). El arnés los detecta igualmente.

## 6. Diferencias con SPEC-M2 y con la propuesta, para revisar

1. **Nuevos identificadores:** `PROTOCOLO_ESTRICTO`, `PROTOCOLO_GIT`, `E02_REMOTO`, `E02_REMOTO_HISTORIA`, `E02_REQUERIDA` y `E02_NO_OFICIAL` (checklist 1–3).
   - La comprobación de P2 en el remoto está **dentro** de `official_config_e02` y consulta siempre el remoto real del repositorio GitHub autorizado (v3 y siguientes, sección 00.1); no existe ninguna opción `--fetch`.
2. **`AUDITED_E02`** añade `tda17_causalidad.py`, `tda17_salidas.py`, `test_tda17_causalidad.py` y `test_tda17_e02_motor.py` (12 ficheros en v1; 14 en v2 con `tda17_entradas.py` y `test_tda17_procedencia.py`; 15 en v5 con `test_tda17_salidas.py`; SPEC-M2 §3.3: «cualquier módulo nuevo de la fase 2»).
3. **`GIT` incluye «no es la raíz del repositorio»** (`rev-parse --show-toplevel`). Un código de retorno de Git distinto de 0/1 en `diff` o `merge-base` también es `GIT`.
4. **Lecturas de implementación:** registro de decisiones §2 (orden de 15 flujos con 2019; réplicas comunes a R² y Spearman; cuantiles de contrato con vecindad; cortes aleatorios con reemplazo; motivo `CONTRATO_CORTO` para cortes fuera de 2..n−1; perturbación que cambia la forma).
5. **El valor del YAML no es una copia literal de la propuesta en tres puntos**, todos tomados del texto aprobado de la propuesta (§3.4, §3.5, §3.6 y §5):
   - `sensibilidad_S4`, `replica_invalida.spearman` y `criterio`;
   - `subniveles_r2.caracter`;
   - `vecindad_del_evento`, `duplicados`, `cortes_aleatorios.rango` y `tabla.pruebas`.

## 7. Procedimiento futuro A2 → P2 (SPEC-M2 §10, sin ejecutar)

> **Sustituida** por la sección 6 de `TDA17_E02_cierre_formal.md` (procedimiento preparado sobre la versión definitiva v11). Se conserva como referencia.

1. **Auditoría independiente** de esta versión.
   - Las correcciones que pida se aplican antes de A2.
   - Si cambia el YAML, se fija su nueva huella en `E02_SHA256` y en el registro.
   - Si cambia `AUDITED_E02`, se cierra la lista en A2.
2. **Precondiciones.**
   - HEAD = P1 o descendiente.
   - 8 ficheros de E01 y protocolo intactos (`verificar` de los tres módulos CONFORME).
   - Suite estricta completa en verde.
   - Arnés T-MUT sin supervivientes no documentados.
3. **Commit A2.**
   - Contenido: el YAML de E02, el cargador, `tda17_analysis.py`, `tda17_g2.py`, `tda17_causalidad.py`, `tda17_salidas.py`, `tda17_entradas.py`, las 8 pruebas de `AUDITED_E02` (v5 añade `test_tda17_salidas.py`), el registro en `PENDIENTE_AUDITORIA` (`aprobacion: null`) y los documentos de `reports/mnq` de E02, incluidos la propuesta aprobada de las decisiones 14–19 (sin modificar, `f16672c3…`) y el anexo `TDA17_E02_anexo_relaciones_N.md`.
   - Comprobar que las huellas LF de A2 son las auditadas y que `git diff --quiet P1 A2 -- <8 ficheros de E01>`.
4. **Aprobación.**
   - El responsable completa en `reports/mnq/TDA17_enmiendas_E02.json`: `estado: APROBADA`, `fecha` (≥ 2026-10-18), `aprobada_por` (declaración, identidad no autenticada), `referencia_auditoria: {documento: acta de E02, commit_auditado: A2}` y `sha256_auditados` (las **15** huellas de `AUDITED_E02` en v5).
   - El registro de E01 **no se toca**.
5. **Pruebas en estado aprobado.**
   - `load_effective_protocol_e02()` activa (`ejecucion_oficial: false`).
   - Suite estricta completa.
6. **Commit P2.** Registro aprobado, acta `reports/mnq/TDA17_E02_acta_aprobacion.md` y evidencias.
7. **Verificación final.**
   - `python -m ohlcv_dataroad.ingest.tda17_enmiendas_e02 verificar --referencia-git P2`: CONFORME, con `ejecucion oficial: SI` (incluye la verificación estricta y la comprobación de P2 en la rama `refs/heads/main` del repositorio autorizado, consultada directamente en su URL).
   - `tda17_enmiendas verificar --referencia-git P2` y `tda17_protocol_freeze verificar --referencia-git 9cdea2f`: CONFORME.
   - A2 es antecesor de P2.
8. **Publicación.**
   - `git push` normal, sin `--force`, tras comprobar que el remoto no ha avanzado.
   - Después: `verificar --referencia-git P2`, que consulta el remoto real del repositorio GitHub autorizado y exige P2 en `refs/heads/main` con A2 y P1 en su historia.
   - Registrar en el acta el SHA remoto.
9. **Solo entonces** puede autorizarse, por separado, la ejecución oficial sobre MNQ.
   - La **única** vía de publicación es `python -c "from ohlcv_dataroad.ingest import tda17_salidas as s; s.official_publish_th35('<P2>')"`, sin argumentos adicionales.
   - Esa vía deriva `official_config(P2)`, ejecuta TH37 y TH35 en el mismo proceso y escribe las tablas según el estado, junto con `reports/mnq/TDA17_procedencia_oficial.json`.
   - Después:
     - `verificar --referencia-git P2` (incluye `check_official_publication`);
     - `reproduce_official_publication('<P2>')` (recálculo y comparación byte a byte);
     - el commit de resultados, con autorización explícita.
   - Si el estado es `BLOQUEADO_HASTA_EXPLICAR`, solo existirán la auditoría causal y `TDA17_diagnostico_bloqueo.csv`. El levantamiento exige una enmienda formal posterior auditada (ciclo M2).
