# TDA-17 — E03: informe R4 (prueba de estado en ambos estados + corrección del bloqueo del motor)

Fecha: 2026-10-26. Estado: **R4 preparada para auditoría independiente**.

**Qué no se ha hecho:**
- No hay A4 ni P3 reales, ni una nueva aprobación oficial, ni push.
- No se ha usado MNQ real, ni el holdout, ni `data/raw`.
- No se han modificado E01, E02, el protocolo, el YAML ni los parámetros estadísticos.

**Repositorio del proyecto:** HEAD = A3 `fbd7ef64bb9bd038c0b49c36c65eab169eab9036`, origin/main = P2 `ff224dc7e30f7631dc46eea17db5a60dd44b07cc`, árbol limpio (verificado al reanudar y otra vez al empaquetar).

**Ensayo:** solo en el clon temporal `C:/tm19/r4/clon`, con el push deshabilitado:

| Commit de ensayo | SHA | Padre | Contenido |
|---|---|---|---|
| A4' | `1c973603e982e9df1f69686bbb9979a0cac64e9d` | A3 | los 4 ficheros de `codigo_A4prima/` |
| P3' | `84915b309b5ba8894bf6e1360699499c3b39d3c3` | A4' | aprobación SINTÉTICA que cita A4', y acta de ensayo |

## 1. Alcance de R4

Hay dos encargos autorizados.

1. **Prueba de estado (2026-10-26).** La prueba del registro del proyecto admite exactamente PENDIENTE_AUDITORIA y APROBADA:
   - En PENDIENTE se conservan todas las comprobaciones originales, incluidos los rechazos de la ruta oficial.
   - En APROBADA se exigen:
     - una aprobación válida;
     - huellas exactas del commit auditado;
     - que el commit auditado tenga el registro en PENDIENTE;
     - que el acta exista;
     - que la ejecución oficial se rechace sin P3 sin llegar a crear el directorio de ejecución.
   - El mismo control (`check_project_state`) se ejerce en un repositorio Git temporal en ambos estados. Allí una P3 publicada correctamente habilita la vía oficial, y las aprobaciones falsificadas o incoherentes (`FORGED`) se rechazan.

2. **Bloqueo del motor (ampliación del 2026-10-26).** En `motor.py` de A3, el envío síncrono `w.conn.send` podía esperar indefinidamente a un worker muerto. Ocurre cuando el worker muere antes de tomar su extremo del pipe y el mensaje supera el búfer del pipe, por ejemplo una tarea TH37O de 79 864 bytes.

Diff único A3 → A4': `diffs/DIFF_R4_A3_a_A4prima.patch` (4 ficheros, +446 / −23):

| Fichero | + | − |
|---|---|---|
| `src/ohlcv_dataroad/tda17_e03/motor.py` | 89 | 9 |
| `tests/tda17_e03/test_e03_motor.py` | 168 | 3 |
| `tests/tda17_e03/test_e03_gobernanza.py` | 180 | 11 |
| `tests/tda17_e03/test_e03_cobertura.py` | 9 | 0 |

Ficheros sin cambios: el YAML (`E03_SHA256 = 85cbf62b…`), `enmienda.py`, `__init__.py`, `e03_equivalencia.py` y el registro (PENDIENTE). Las huellas de los 8 ficheros de `AUDITED_E03` en A3 y en A4' están en `HUELLAS_E03_A3_vs_A4prima.json`.

## 2. Corrección del motor (`motor.py`)

- **Confirmación de arranque.** Cada worker envía `("listo", pid)` en cuanto posee su extremo del pipe. No se le envía ninguna tarea antes. Un worker que no confirma en `espera_arranque_s` (300 s) produce `E03_TRABAJADOR`. Si una tarea se asigna antes de la confirmación, queda en `w.mensaje` y se envía al confirmar.
- **Transferencias con plazo** (`_transferir`). El envío de cada tarea y la recepción de cada mensaje se hacen en un hilo daemon (`e03-pipe-<pid>`), con plazo `espera_envio_s` (300 s) y vigilando la vida del worker. Si el worker muere, el principal espera 2 s más, porque lo ya escrito sigue siendo legible, y después rechaza. `EOFError` y `OSError` se convierten en `E03_TRABAJADOR`.
- **Workers libres muertos.** Antes de despachar se comprueban los workers libres: uno muerto sin tarea también es un fallo.
- **Un único camino de fallo** (`_fallo_worker`): registra el evento `worker_fallo`, termina el worker y lanza `E03_TRABAJADOR`. `_abortar` termina los demás workers. Nunca hay resultado ni publicación parcial; en la ruta oficial lo prueba `test_a_dead_worker_during_the_official_e03_run_publishes_nothing`.
- **Sin efecto sobre el cálculo.** No cambian las tareas, el orden de cálculo ni los parámetros estadísticos. Los mensajes enviados son los mismos objetos; solo cambia cuándo y cómo se transfieren.

## 3. Evidencia de la causa (antes de la corrección)

| Reproducción | Resultado | Fichero |
|---|---|---|
| Motor de A3 sin corregir | B1 (worker suspendido y muerto 1 s tras asignar) y B2 (muerto antes del envío): **bloqueados** (> 90 s) en todas las repeticiones. B3 (el asesino de la prueba original): bloqueado en 1 de 5 y en 2 de 5. | `repro/repro_A4_sin_corregir*.log` |
| Pila del bloqueo | `motor.py:1059` `w.conn.send` → `_send_bytes` → `WaitForMultipleObjects` | `repro/repro_A4_sin_corregir_pilas.log` |
| Mecanismo en un pipe puro | Hijo muerto **antes** de tomar su extremo + 80 KB: bloqueo 3/3. Con 1 KB: el envío vuelve. Hijo muerto **después** de tomarlo: `BrokenPipeError` 3/3. | `repro/repro_mecanismo.log` |
| Motor A4' | Los 12 escenarios del motor (B1, B2 y B3.0–B3.9) terminan en 1,2–2,3 s con `E03_TRABAJADOR`, sin procesos hijos vivos | `repro/repro_A4prima_final.log` |

## 4. Pruebas nuevas

- **`test_e03_motor.py`** (deterministas, con tareas TH37 reales; las de envío usan mensajes > 4 MB):
  - `test_a_worker_that_dies_before_receiving_a_large_task_fails_explicitly`
  - `test_a_worker_that_dies_during_the_send_of_a_large_task_fails_explicitly`
  - `test_a_blocked_worker_that_never_reads_its_task_fails_explicitly`
  - `test_a_worker_that_never_confirms_its_start_fails_explicitly`
  - `test_a_worker_that_dies_while_idle_fails_explicitly`
  - `test_the_pipe_transfers_are_bounded_and_watch_the_worker`

  `test_a_dead_worker_fails_the_whole_run` se conserva y se refuerza, sin omitirla ni ampliar sus tiempos.
- **`test_e03_gobernanza.py`:**
  - `test_project_registry_is_pending_or_approved_and_the_official_e03_route_needs_a_valid_p3`, que sustituye a la prueba que solo admitía PENDIENTE;
  - `test_the_project_state_check_in_a_temporary_repository_in_both_states`;
  - `test_a_forged_or_inconsistent_project_approval_is_always_rejected` (parametrizada);
  - `test_a_dead_worker_during_the_official_e03_run_publishes_nothing`.
- **`test_e03_cobertura.py`:** guardas estáticas de la corrección del motor.

E03 pasa de 94 pruebas (R3) a **121**.

## 5. Mutantes

| Conjunto | PENDIENTE (A4') | APROBADA (P3') | Logs |
|---|---|---|---|
| Aprobación M1–M8 | 8/8 detectados, control verde | 8/8 detectados, control verde | `logs/mutantes_PENDIENTE_A4prima.log`, `logs/mutantes_APROBADA_P3prima.log`, `logs/mutantes/` |
| Motor MM1–MM8 | 8/8 detectados, control verde | 8/8 detectados, control verde | `logs/mutantes_motor_PENDIENTE.log`, `logs/mutantes_motor_APROBADA.log`, `logs/mutantes_motor/` |
| MM0 = comportamiento de R3 (envío directo, sin confirmación) | detectado: 5 pruebas fallan, una de ellas por el vigilante con "bloqueo indefinido" | — | `logs/mutantes_motor_R3_combinado.log` |

Mutantes del motor:
- MM1: envío directo;
- MM2: sin confirmación de arranque;
- MM3: transferencia sin plazo;
- MM4: transferencia sin vigilar la vida del worker;
- MM5: worker libre muerto ignorado;
- MM6: fallo sin evento;
- MM7: fallo sin terminar el worker;
- MM8: sin plazo de arranque.

## 6. Ejecuciones pesadas (de una en una, con puerta de memoria ≥ 12 GB y límite externo `timeout`)

| # | Qué | Estado (HEAD del clon) | RAM libre al inicio | Resultado | Log |
|---|---|---|---|---|---|
| 1/4 | E03 focalizadas | PENDIENTE (A4') | 12 GB | **121 passed, rc=0** (1 h 35 min 36 s) | `logs/e03_PENDIENTE.log` |
| 2/4 | E03 focalizadas | APROBADA (P3') | 13 GB | **121 passed, rc=0** (1 h 38 min 6 s) | `logs/e03_APROBADA.log` |
| 3/4 | suite estricta completa | PENDIENTE (A4') | 17 GB | **1917 passed, 13 skipped, rc=0** (2 h 20 min 42 s) | `logs/suite_PENDIENTE.log` |
| 4/4 | suite estricta completa | APROBADA (P3') | 19 GB | **1917 passed, 13 skipped, rc=0** (2 h 19 min 53 s) | `logs/suite_APROBADA.log` |

- Cada log empieza con el estado, el HEAD del clon, el estado del registro (y en APROBADA el commit citado) y el origen de los módulos (`C:\tm19\r4\clon`). Termina con `rc=` y la RAM libre final.
- Las ejecuciones 3/4 y 4/4 se hicieron después del reinicio de Windows, con el respaldo verificado (`sha256sum -c` correcto y `git bundle verify` correcto).
- 1917 = 1796 pruebas previas + 121 de E03. Hay 13 omitidos, el mismo número que en R3.
- Las salidas de los lanzadores están en `logs/lanzador_*.txt`.
- Las **pruebas ligeras** (75 passed en cada uno de 3 estados, `logs/focalizadas_ligeras/`) son del primer ensayo, que solo cambiaba la prueba de estado. Las cubren 1/4–4/4.

**Inválidos** (conservados en `logs/INVALIDOS/`, con `LEEME.txt`; ninguno cuenta como evidencia):
- la suite del primer ensayo, colgada por el defecto del motor y detenida con autorización;
- una ejecución de desarrollo cortada por un límite de 1 h demasiado corto.

## 7. Avisos del hilo de envío (comprobados)

Las ejecuciones con A4' muestran avisos `PytestUnraisableExceptionWarning: Exception ignored in: <_io.BytesIO …> … BufferError: Existing exports of data: object cannot be re-sized`. En R3 no aparecían: el hilo de envío es nuevo. Están clasificados y explicados.

- **Causa.** `Connection.send` serializa con `ForkingPickler.dumps`, que devuelve `buf.getbuffer()`, una `memoryview` del `BytesIO`. Cuando un envío falla porque el worker murió:
  1. el `BrokenPipeError` queda guardado en `out["error"]`;
  2. su traceback retiene el frame de `_send_bytes`, con la vista;
  3. el frame de `run` retiene `out`, lo que cierra un ciclo de referencias.

  Al romper ese ciclo, el GC puede destruir el `BytesIO` antes que su vista. CPython lo comunica como una excepción no capturable, y pytest lo convierte en aviso en la prueba que esté en curso cuando actúa el GC. No afecta al resultado: el envío ya había fallado y la ejecución ya se había rechazado con `E03_TRABAJADOR`.
- **Reproducción aislada** (`avisos/repro_aviso_hilo_envio.py` → `.json`; un pipe, sin motor ni pytest):
  - envío correcto de 200 KB y de 2 MB: **0 avisos**;
  - hijo muerto durante el envío de 2 MB (`BrokenPipeError`): **exactamente ese `BufferError` sobre `BytesIO`**, 2 de 2 veces.
- **Verificación de los logs** (`avisos/verificar_avisos.py` → `avisos/AVISOS_HILO_ENVIO.json`): en 1/4, 2/4, 3/4 y 4/4 el resultado es el mismo:
  - 3 avisos del hilo de envío, todos `BufferError` sobre `BytesIO`;
  - ninguna `PytestUnhandledThreadExceptionWarning` y ningún error de un hilo `e03-pipe`;
  - el resto de avisos son los `RuntimeWarning` de numpy en TDA-06/08, ya presentes en R3 (216 = 213 + 3).
- **Pruebas a las que se atribuyen.**
  - `test_a_blocked_worker_that_never_reads_its_task_fails_explicitly` y `test_a_worker_that_dies_while_idle_fails_explicitly` provocan ellas mismas envíos fallidos.
  - `test_end_to_end_equivalence_in_the_inherited_environment` hereda el de `test_a_dead_worker_during_the_official_e03_run_publishes_nothing` (gobernanza, que mata un worker y luego le envía). La prueba que corre entre ambas (`…in_a_fixed_numeric_environment`) trabaja en un subproceso, así que el GC del proceso de pytest actúa por primera vez en la siguiente.
- **Posible mejora, no aplicada por estar fuera del alcance autorizado:** en `_transferir`, limpiar `out["error"].__traceback__` después de clasificar el error. Eliminaría el aviso sin cambiar el comportamiento. Si se quiere, iría en A4 con una nueva auditoría.

## 8. Procedimiento A4 → aprobación → P3

Está en `PROCEDIMIENTO_A4_aprobacion_P3.md`, actualizado: A4 incluye `motor.py` y las 3 pruebas, con sus huellas LF, y la aprobación definitiva debe citar A4, no A3. La aprobación preparada en `C:/tm19/r3/wt`, que cita A3, queda obsoleta.

## 9. Contenido del paquete

| Elemento | Contenido |
|---|---|
| `INFORME_R4_E03.md` | este informe |
| `PROCEDIMIENTO_A4_aprobacion_P3.md` | procedimiento propuesto |
| `diffs/` | diff único A3→A4', diff de ensayo A4'→P3' y estadísticas. Se regeneraron desde el clon y son idénticos byte a byte a los del respaldo. |
| `codigo_A4prima/` | los 4 ficheros de A4' (sus huellas coinciden con `HUELLAS_…json`) |
| `P3prima_ensayo/` | registro y acta sintéticos |
| `logs/` | logs válidos e inválidos |
| `repro/` | reproducciones del bloqueo |
| `avisos/` | análisis de los avisos del hilo de envío |
| `scripts/` | lanzador, mutantes, ensayo de aprobación, respaldo y empaquetador |
| `MANIFIESTO_SHA256.txt` | huellas de todos los ficheros del paquete |
| `EXCLUIDOS_SHA256.txt` | ficheros grandes o datos temporales de ejecuciones inválidas, no incluidos, con su huella |

## 10. Decisiones pendientes del responsable

Cada uno de estos pasos necesita autorización explícita aparte:
- A4 real;
- aprobación citando A4;
- P3;
- push;
- ejecución oficial sobre MNQ.

Sobre el worktree obsoleto `C:/tm19/r3/wt`, se propone archivarlo y eliminarlo, también con autorización. El holdout sigue cerrado.
