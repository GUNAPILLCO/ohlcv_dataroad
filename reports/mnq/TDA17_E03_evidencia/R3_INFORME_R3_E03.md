# TDA-17 — E03: informe R3 (cierre de la auditoría correctiva)

Fecha: 2026-10-25. Estado: **R3 preparada para auditoría**.

- E03 **no está aprobada ni publicada**.
- No hay commits, aprobación ni push. HEAD = `origin/main` = P2 `ff224dc`. Siguen las mismas 4 entradas sin seguimiento.
- No he usado MNQ real, ni el holdout, ni `data/raw`. Todo es sintético.
- No he modificado el protocolo, E01, E02, sus verificadores ni ningún parámetro estadístico.
- No he repetido el benchmark MD interrumpido de 20 procesos.

Cambios respecto a R2, sin contar los finales de línea (detalle en `DIFF_R2_a_R3.patch`):

| Fichero | + | − |
|---|---|---|
| `configs/tda17_enmienda_03.yaml` | 16 | 0 |
| `reports/mnq/TDA17_enmiendas_E03.json` | 1 | 1 (nueva huella del YAML) |
| `src/ohlcv_dataroad/tda17_e03/motor.py` | 68 | 15 |
| `src/ohlcv_dataroad/tda17_e03/enmienda.py` | 157 | 30 |
| `tests/tda17_e03/test_e03_motor.py` | 93 | 3 |
| `tests/tda17_e03/test_e03_gobernanza.py` | 204 | 7 |
| `tests/tda17_e03/test_e03_cobertura.py` | 22 | 3 |

La nueva huella del YAML es `E03_SHA256 = 85cbf62b6c1ac9b5e0fca7e869d6502a0f5784bcde5ed26be809d3cc8042c747`. La he regenerado con `scripts/generar_registro_e03.py`, igual que en R2.

## 1. Corrección 1: memoria (`motor.py`)

**Defecto auditado.** `_cabe()` devolvía `True` siempre que no hubiera ninguna tarea activa. Una tarea podía arrancar aunque su memoria prevista, más los datos compartidos pendientes y la reserva, superaran la memoria disponible.

**Corrección.**
- `_balance_memoria(t, w)` calcula: disponible − compartidos pendientes − crecimiento posible de las activas − memoria prevista de la tarea, y lo compara con la reserva.
- `_cabe()` evalúa ese balance **siempre**, también sin activas. Los topes de concurrencia (`max_sin_medir`, rendimiento) solo se aplican si hay activas.
- Nuevo parámetro `Recursos.memoria_estricta`. Si una tarea no cabe:
  - **con tareas activas**, espera a que terminen;
  - **sin ninguna activa y en modo estricto (ruta oficial):**
    1. cierra los workers libres, cuyas cachés ocupan memoria privada;
    2. espera hasta `espera_memoria_s` (YAML: 10 s) a que el sistema libere memoria;
    3. si aun así no cabe, **rechaza** con `E03_MEMORIA_INSUFICIENTE`, sin lanzar la tarea.
    - El mensaje y el evento `memoria_insuficiente` de `progreso.jsonl` incluyen la memoria disponible, los compartidos pendientes, la memoria prevista y la reserva.
  - **sin ninguna activa y en modo no estricto** (pruebas sintéticas y benchmarks), el comportamiento anterior no cambia: la lanza sola y lo registra (`memoria_baja_secuencial`).
- **La reserva de 4 096 MB no se elimina ni se rebaja.**
- `run_th35_e03(oficial=True)` exige memoria estricta, al menos 1 proceso y una reserva positiva. Si no, falla con `E03_RECURSOS_OFICIALES` antes de crear el directorio.
- `enmienda.official_resources` fija `memoria_estricta=True`.
- El YAML lo declara (`memoria_insuficiente: RECHAZO_EXPLICITO`, `espera_memoria_s: 10`) y `_check_doc` lo exige, junto con una reserva entera positiva.

**Pruebas nuevas (`test_e03_motor.py`).**
- `test_strict_memory_never_starts_a_worker_for_a_task_that_does_not_fit`:
  - 4 000 MB disponibles frente a la reserva de 4 096 MB;
  - se rechaza con `E03_MEMORIA_INSUFICIENTE`;
  - **no hay ningún evento `worker_inicio`**, `max_activos == 0`, y el evento `memoria_insuficiente` lleva las cifras.
- `test_strict_memory_counts_the_task_the_pending_shared_data_and_the_reserve`. Comprueba los mensajes exactos y que no se crea ningún worker en tres casos:

  | Disponible | Compartidos pendientes | Tarea | Reserva | Resultado |
  |---|---|---|---|---|
  | 5 000 | 2 500 | 2 000 | 1 000 | rechazo |
  | 5 000 | 2 500 ya residentes | 2 000 | 1 000 | cabe |
  | 5 000 | 0 | 4 500 | 1 000 | rechazo |

- `test_strict_memory_waits_while_tasks_are_running`: con activas, la tarea queda en cola; ni se lanza ni se rechaza.
- `test_strict_memory_closes_idle_workers_and_launches_only_when_it_fits`:
  - se cierran los workers libres y la tarea arranca solo cuando cabe;
  - resultados idénticos (`pickle_sha`) a los de la etapa de referencia.
- `test_an_official_run_needs_strict_memory_control`: `E03_RECURSOS_OFICIALES` en 3 configuraciones (sin memoria estricta, 0 procesos, reserva 0).
- `test_official_runs_never_resume_and_never_read_checkpoints` usa ahora recursos estrictos con 2 procesos reales, y comprueba `lanzadas_sin_memoria_suficiente == 0`.
- Guarda estática (`test_e03_cobertura.py`):
  - `_cabe` no contiene `return True` y termina en el balance;
  - el rechazo está detrás de `if not self.r.memoria_estricta:`;
  - `run_th35_e03` contiene `E03_RECURSOS_OFICIALES`;
  - `official_resources` contiene `memoria_estricta=True`.

## 2. Corrección 2: procedencia y publicación de E03 (`enmienda.py`)

**Defecto auditado.** Las tablas y el registro de E02 se publicaban dentro de la transacción. `publicacion.json`, con la trazabilidad de E03, se escribía **después y fuera de ella**, en el directorio de la ejecución. Una interrupción podía dejar una publicación confirmada sin procedencia E03.

**Corrección.**
- **Comprobante E03 dentro de la transacción.** `TDA17_procedencia_E03.json` se guarda en el directorio de salidas oficiales. Campos exactos `COMPROBANTE_FIELDS`:
  - enmienda, publicacion_E03 (P3), commit_auditado_E03 (A3), remoto_E03;
  - sha256_yaml_E03, publicacion_E02 (P2), registro_E02;
  - **sha256_registro_E02**: huella de los bytes exactos del registro de E02;
  - tablas (huellas), hilos_blas, reanudacion, motor (resumen: hilos efectivos, recursos, memoria).
- **Orden de promoción.** Se reutiliza **la misma transacción de E02 sin modificarla**:
  - `sal._exclusive` (cerrojo), `sal._stage_and_promote` (preparación con fsync, diario de huellas, promoción sin sobrescribir y deshacer de lo propio);
  - orden: **tablas → comprobante E03 → registro de E02**;
  - el registro de E02 sigue siendo el último fichero y el que confirma la publicación.
  - Por eso no puede existir una publicación confirmada sin comprobante. Sin registro, la publicación es inválida, tanto para E02 (`SALIDA_INCOMPLETA`) como para E03.
- **Bytes exactos.** Las tablas y el registro de E02 son los mismos bytes que produce `sal.official_publish_th35`. El comprobante es un fichero adicional. `OFFICIAL`, `check_official_publication` y `reproduce_official_publication` de E02 no cambian. `publicacion.json` ya no se escribe.
- **Recuperación** (`_recover_e03`, bajo el cerrojo). `sal._recover` no conoce el comprobante. Si hay una preparación sin registro y el comprobante está presente:
  1. primero se demuestra que el comprobante es propio, con `sal._owned`: misma identidad de fichero que su preparado y huella del diario;
  2. después se ejecuta `sal._recover`;
  3. solo entonces se retira el comprobante.
  - Un comprobante ajeno, aunque tenga los mismos bytes en otro fichero, bloquea la recuperación sin tocar nada (`E03_SALIDA_RECUPERACION`).
  - Un comprobante huérfano sin preparación cuenta como salida existente: nunca se borra ni se sobrescribe.
- **Verificador** `check_official_publication_e03(P3)`:
  1. cadena oficial de E03 (`official_e03`: registro, A3, P3, remoto y E02);
  2. `sal.check_official_publication(P2)` sin cambios;
  3. `_check_comprobante`:
     - fichero regular y JSON con los campos exactos (`E03_COMPROBANTE`);
     - cada campo vinculado (`E03_COMPROBANTE_VINCULO`, que lista los campos distintos): P3, A3, huella del YAML, P2, nombre y **huella exacta** del registro de E02, huellas de las tablas, hilos = 1, reanudación DESHABILITADA, motor con hilos efectivos = 1 y `memoria_estricta` = true;
     - el SHA remoto registrado debe ser un commit existente que contenga P3, porque la rama puede avanzar.
  - `reproduce_official_publication_e03` y `reproduce_with_e02_engine_e03` llaman antes a este verificador.
- **Límite que declaro.** El comprobante, como el registro de E02, no está firmado. Quien pueda escribir en `reports/mnq` puede fabricar un par coherente. La verificación sin recálculo garantiza **integridad y vínculo** (comprobante ↔ registro E02 ↔ tablas ↔ P3/A3/remoto), no la autenticidad del cálculo. La autenticidad del cálculo la da la **reproducción** con recálculo byte a byte, con el motor E03 y con el motor E02, que es la misma garantía de E02.

**Pruebas nuevas (`test_e03_gobernanza.py`).** Sin cálculo, sobre la transacción real:
- `test_the_e03_receipt_is_promoted_after_the_tables_and_before_the_e02_record`: orden exacto [tablas…, comprobante, registro], y una segunda publicación se rechaza con `SALIDA_EXISTENTE`.
- `test_a_failure_during_the_e03_publication_leaves_nothing_published[3]`: fallo en la tabla, en el comprobante o en el registro → no queda nada publicado y una nueva publicación funciona.
- `test_a_publication_killed_before_the_e02_record_is_recovered_including_the_receipt[2]`: muerte sin deshacer, antes del comprobante y antes del registro → la siguiente publicación recupera (también el comprobante) y publica.
- `test_a_foreign_receipt_blocks_the_recovery_and_nothing_is_removed`.
- `test_an_orphan_receipt_is_never_overwritten_or_removed`.

Con la ejecución oficial sintética completa (repositorio temporal sobre el P2 real, A3/P3 simulados, remoto temporal):
- `test_official_e03_route_publishes_the_same_bytes_as_the_e02_route`:
  - publica y verifica con E02 y con E03, y comprueba el contenido del comprobante;
  - **12 alteraciones o ausencias del comprobante**, cada una rechazada con su campo exacto: P3, A3, remoto que no contiene P3, YAML, huella del registro E02, tablas, reanudación, hilos 24, memoria no estricta, campo eliminado, JSON truncado y fichero ausente;
  - reproducciones con E03 y con E02;
  - la ruta E02 publica los mismos bytes;
  - sin comprobante, E03 no verifica.
- `test_an_e03_publication_interrupted_between_the_receipt_and_the_e02_record_is_never_valid`: muerte simulada tras promover tablas y comprobante → `check_official_publication_e03` rechaza con `SALIDA_INCOMPLETA`, y la recuperación lo retira todo.

## 3. Mejora: rechazo de salidas existentes antes del cálculo

- `official_publish_th35_e03` pasa a `_official_run_e03` una comprobación `antes_del_calculo`, que se ejecuta:
  - **después** de verificar la cadena oficial;
  - **antes** de cargar entradas o crear el directorio de la ejecución.
- Con el cerrojo, recupera una publicación interrumpida y rechaza con `SALIDA_EXISTENTE` si hay cualquier salida oficial o el comprobante.
- **La protección transaccional contra carreras se mantiene.** `_publish_files_e03` repite la recuperación y la comprobación bajo el cerrojo al publicar, y `sal._promote` sigue sin sobrescribir.
- Pruebas:
  - `test_existing_outputs_are_rejected_before_any_computation`: no se cargan entradas y no se crea el directorio;
  - el final de la prueba de publicación completa (tras publicar por la ruta E02) comprueba que E03 rechaza sin crear `ej2`.

## 4. Validación

### 4.1 Pruebas focalizadas
- **Cobertura y motor** (memoria, guardas, ejecución oficial con 2 procesos reales): 24 pasadas. La cobertura se repitió tras el último cambio: 9 pasadas.
- **Gobernanza completa** (`evidencia/pruebas_R3/gobernanza_R3.log`): 44 pasadas y **2 fallidas, ambas por defectos de las pruebas, no del código**:
  1. un `monkeypatch.undo()` deshacía también los parches del fixture, incluido el remoto autorizado;
  2. la preparación interrumpida que dejó esa prueba al fallar persistía en la siguiente, porque `git clean` no borra los ficheros ignorados.
  - Corregidas: `pytest.MonkeyPatch.context()` y limpieza de restos en el fixture `official3`, solo en el repositorio temporal.
  - Las 10 pruebas de publicación, repetidas: 10 pasadas.
- El log se conserva tal cual. La suite completa (§4.4) incluye la versión corregida.

### 4.2 Extremo a extremo sintético, 300 réplicas, 8 procesos, memoria estricta, BLAS = 1
Benchmark `scripts/benchmark_e03.py e2e --estricta`. El E02 secuencial y el E03 se ejecutaron en el mismo proceso, con `OPENBLAS_NUM_THREADS=1` (hilos efectivos 1/1 registrados).

La comparación cubre:
- los bytes de cada tabla oficial (`cmp` sobre los CSV guardados);
- **el resultado completo**, con `deep_equal` exacto sin tolerancias y huella normalizada, incluidos `bootstrap` y `validacion_spearman`.

| Caso | Filas | Estado | Tareas bootstrap R² | Tareas validación Spearman | E02 | E03 (8 proc.) | Tablas | Resultado completo |
|---|---|---|---|---|---|---|---|---|
| **MD** (6 contratos × 21 jornadas × 950) | 119 700 | BLOQUEADO_HASTA_EXPLICAR | **270** | **60** (VALIDADA) | 1 814,1 s | 304,3 s (**5,96×**) | 2/2 idénticas | **idéntico** (`deep_equal` = []) |
| **SB** (caso sin bloqueo de las pruebas) | 2 750 | **SIN_BLOQUEO** | **90** | no ejecutable (1 año y 1 segmento) | 307,8 s | 61,7 s (4,99×) | **5/5 idénticas** (R², Spearman, estabilidad, clasificación, causalidad) | **idéntico** |

Huellas (E02 = E03):
- MD: resultado `e00047b8…`, bootstrap `6c743f25…`, validación `0c6fc820…`.
- SB: resultado `e9503cf4…`, bootstrap `fdaaf858…`, validación `3c0f39da…`.

**Memoria.** `lanzadas_sin_memoria_suficiente = 0` en los dos casos.
- MD: 8 workers activos, pico privado total de los workers 5 645 MB, mínimo disponible 12 984 MB.
- SB: mínimo disponible 17 278 MB.

**Corrección de mi informe R2.** En el estado BLOQUEADO, las tablas oficiales son solo `auditoria_causalidad` y `diagnostico_bloqueo`, y **no contienen el bootstrap R² ni la validación de Spearman**. Por eso el "tablas idénticas" del SD de R2 y de la primera pasada MD de R3 (`evidencia/benchmark_R3/MD_e2e_p8_estricta_solo_tablas/`, 1 867,9 s frente a 304,3 s, 2/2 tablas idénticas) **no cubría esos cálculos**. R3 añade la comparación del resultado completo (MD) y un caso SIN_BLOQUEO con las 5 tablas (SB).

**Corrección del recuento de R2.** Las 79 pruebas E03 de R2 eran 9 de cobertura, 36 de gobernanza y 34 de motor, no "17 y 53". El total era correcto.

Una primera pasada de SB terminó **sin informe**: el guardia del benchmark exigía tareas de validación de Spearman, que en SB no existen. Sus 5 tablas ya eran idénticas. Relajé el guardia solo para SB y la repetí (`SB_e2e_p8_estricta_2`). El log de la primera pasada se incluye.

### 4.3 Comparación contra E02 con BLAS = 1
- Benchmarks: §4.2.
- Pruebas: la publicación oficial E03 y la ruta E02 con 1 hilo fijado dan los mismos bytes.
- Pruebas de equivalencia en subproceso: entorno fijo, entorno heredado y 1 hilo fijado en ejecución.

### 4.4 Suite estricta completa
Ejecución única, completa y sin interrupciones (`TDA17_INTEGRIDAD=estricta`, `--basetemp=C:/tm19/r3/suite`), con la versión final del código:

**1890 passed, 13 skipped, 213 warnings en 8445,40 s (2:20:45), rc=0** (`logs/suite_completa_R3.log`).

1890 = 1796 previas + 94 de E03 (9 cobertura, 46 gobernanza, 39 motor). R2 también tuvo 13 omitidos. Las advertencias son RuntimeWarning de numpy en TDA-06/08 y ya aparecían antes.

**Normalización de fin de línea (después de la suite).** En Windows, `write_text` había dejado CRLF en 7 ficheros E03: yaml, enmienda, motor, e03_equivalencia y los tres tests. Se convirtieron a LF, y solo cambió el fin de línea:
- La huella del YAML no depende del fin de línea. Con LF, su SHA-256 de bytes coincide además con `E03_SHA256` (`85cbf62b…`).
- El repositorio usa `* text=auto`.
- Después se repitieron cobertura (9), el test del registro del proyecto y el test de huella/contenido del YAML: **11 passed, rc=0** (`logs/post_lf_R3.log`).
- Los hashes del manifiesto corresponden a los ficheros ya en LF.

## 5. Pendiente y decisiones del responsable
- **A3, aprobación, P3, push y ejecución sobre MNQ real:** cada paso necesita autorización explícita. E03 sigue PENDIENTE_AUDITORIA.
- **Memoria en la ejecución oficial.** Con memoria estricta, si MNQ real no cabe con la reserva de 4 GB, la ejecución **se rechaza** en vez de degradarse. La protección de memoria de Claude Code puede actuar antes (ver R2). Lanzar la ejecución oficial fuera de Claude Code, o con más margen, es una decisión del responsable.
- **El comprobante no está firmado** (§2). Es la misma clase de garantía que el registro de E02.
