# TDA-17 — Acta de aprobación de la enmienda E03 (R4)

- **Fecha:** 2026-10-26
- **Enmienda:** E03 (`configs/tda17_enmienda_03.yaml`, SHA-256 `85cbf62b6c1ac9b5e0fca7e869d6502a0f5784bcde5ed26be809d3cc8042c747`).
- **Alcance:** motor paralelo E03 R4 (`alcance: IMPLEMENTACION`; `decisiones_estadisticas: NINGUNA`), sin cambios metodológicos.
  - E03 cambia solo quién calcula el resultado: el motor paralelo `tda17_e03.motor` en lugar de `an.th35_pipeline`.
  - La configuración efectiva, las entradas, las tablas, el registro de procedencia y la publicación transaccional siguen siendo los de E02.
  - Criterio de aceptación: tablas idénticas byte a byte, con 1 hilo de OpenBLAS por proceso.
- **Requiere:** E02 publicada y sin modificar (A2 `2b6ad7abdf6609e6970f1fe6ee1e2569edc698b4`, P2 `ff224dc7e30f7631dc46eea17db5a60dd44b07cc`). E02 requiere a su vez E01 publicada (P1 `24f9f1d2bb831a25e4dec6a651c0a5129f0bbca1`).
- **Protocolo base:** `TDA17_PROTOCOLO_v1`, commit `9cdea2f4657de1c54e3fccf86cf380b2eac8704e`, sin modificar.
- **Commit auditado (A4):** `7fc78bcbc28f5e94c02bb545f46c7e89201408a2`. Único padre: A3 `fbd7ef64bb9bd038c0b49c36c65eab169eab9036`, cuyo único padre es P2.
- **Aprobada por:** Responsable del proyecto IRIS (declaración de aprobación; identidad no autenticada).

Esta acta sustituye a la aprobación preparada en la ronda anterior, que citaba A3. Esa aprobación nunca se registró en el repositorio ni se publicó; queda obsoleta y no se ha reutilizado.

## Naturaleza de esta aprobación

Esta acta y el registro `reports/mnq/TDA17_enmiendas_E03.json` son una DECLARACIÓN de aprobación, no una firma criptográfica: no autentican la identidad de quien aprueba.

Sí se verifica automáticamente que lo aprobado es exactamente lo auditado:
- `load_e03_aprobada()` exige que A4:
  - exista;
  - descienda de P2;
  - sea antecesor de HEAD;
  - contenga cada fichero auditado con la huella registrada.
- `official_e03(P3)` vincula toda ejecución oficial a una publicación P3. Esa P3 debe:
  - descender estrictamente de A4;
  - contener esta aprobación citando A4;
  - estar en la rama `refs/heads/main` del repositorio autorizado.

Una aprobación local sin publicar no habilita producción.

## Auditoría independiente (resultados declarados por el responsable del proyecto)

| Ronda | Resultado |
|---|---|
| **F1** | Auditoría independiente del motor paralelo. Sus correcciones se entregaron en R2. |
| **R2** | Favorable, con dos bloqueos: el control de memoria y la procedencia de E03 escrita fuera de la transacción de E02. |
| **R3** | Corrige ambos bloqueos y rechaza las salidas existentes antes del cálculo. Auditoría favorable del paquete `PAQUETE_AUDITORIA_R3_E03.zip` (SHA-256 `4cbf5ecf914d6024c9a6425b9b6d3b5a3eaf0550fd7e7adec5d4e949dfb290f9`). Se confirmó en A3. |
| **R4** | Corrige dos problemas detectados al preparar la aprobación sobre A3 (detalle bajo esta tabla). GPT concluyó una auditoría independiente favorable del paquete `PAQUETE_AUDITORIA_R4_E03.zip` (SHA-256 `73db0a649cdd934ea05d3e0805faab2cd22c5435f3582635169af0c94f4e2d27`). |

Los dos problemas que corrige R4:
1. **Prueba de estado.** La prueba del registro del proyecto solo admitía PENDIENTE_AUDITORIA. Ahora admite exactamente PENDIENTE_AUDITORIA y APROBADA, y conserva todas las comprobaciones en los dos estados.
2. **Espera indefinida en el motor.** El proceso principal de `motor.py` podía esperar para siempre al enviar una tarea a un worker muerto. Ahora:
   - el worker confirma su arranque antes de recibir tareas;
   - los envíos y las recepciones tienen plazo y vigilan la vida del worker;
   - un fallo de worker produce `E03_TRABAJADOR`, sin resultados ni publicación parcial.

**A4** contiene exactamente los 4 ficheros auditados en R4: `motor.py`, `test_e03_motor.py`, `test_e03_gobernanza.py` y `test_e03_cobertura.py`. Su diff respecto de A3 (+446/−23) es idéntico byte a byte a `DIFF_R4_A3_a_A4prima.patch` del paquete auditado. Su árbol (`685306a85e5ec49bd002dca7c73c70abf96ac3a1`) es idéntico al del ensayo A4', sobre el que se validaron estas ejecuciones:
- pruebas de E03: 121 passed en PENDIENTE y 121 passed en APROBADA;
- suite estricta completa: 1917 passed y 13 skipped, rc=0, en PENDIENTE y en APROBADA;
- mutantes de aprobación y del motor: 8/8 y 8/8 detectados en los dos estados.

Las evidencias reproducibles están en `reports/mnq/TDA17_E03_evidencia/` (ver su README).

## Ficheros auditados (SHA-256 tras normalizar CRLF a LF; idénticos en A4)

| Fichero | SHA-256 | Cambia en R4 |
|---|---|---|
| `configs/tda17_enmienda_03.yaml` | `85cbf62b6c1ac9b5e0fca7e869d6502a0f5784bcde5ed26be809d3cc8042c747` | no |
| `src/ohlcv_dataroad/tda17_e03/__init__.py` | `ae22ee28987bbb86a5b7a5bfec53266b422702d63926978ed3b6063dc6697a8a` | no |
| `src/ohlcv_dataroad/tda17_e03/motor.py` | `8c2f9fce9825700e700635c5dc11e7f893112b3c209047859248c78f9539e752` | sí |
| `src/ohlcv_dataroad/tda17_e03/enmienda.py` | `c96b95a938a7e684de18792316116ad8afc7b8f8b657f67b80893f88f93f4a43` | no |
| `tests/tda17_e03/e03_equivalencia.py` | `daf95ae8b62567e2b41b86fbe041b39f0d72e171d1d25b7e5207640de39ca734` | no |
| `tests/tda17_e03/test_e03_motor.py` | `89d2c92d611fad3db4b0fbc162c7cdfc69cb83a6df1730ffea52b62965c3d875` | sí |
| `tests/tda17_e03/test_e03_gobernanza.py` | `51338e941e21773f175060f4e6018af87f1653e547ddc5241b8e3c86dcaf8d17` | sí |
| `tests/tda17_e03/test_e03_cobertura.py` | `3743ad9e422d5110191fc2dd050be53b1017d431e024100f762abea96e4819fc` | sí |

El registro `reports/mnq/TDA17_enmiendas_E03.json` está seguido como fichero de E03 (`TRACKED_E03`), pero no forma parte de los auditados. En A4 está en PENDIENTE_AUDITORIA, con huella `1ea1b7827a7da8d301edb95c51cfe06ab58a516cef5b01dfb15b2ae0801185a7`, la misma que en A3. Esta aprobación solo cambia su `estado` y su `aprobacion`.

## Observación: avisos `BufferError` conocidos

En cada ejecución de las pruebas de E03 aparecen exactamente 3 avisos `PytestUnraisableExceptionWarning: Exception ignored in: <_io.BytesIO …> … BufferError: Existing exports of data: object cannot be re-sized`. Por decisión del responsable se conservan documentados, sin ocultarlos y sin modificar el código.

- **Origen.** Aparecen solo después de un envío fallido a un worker que ha muerto, cuando la ejecución ya se ha rechazado con `E03_TRABAJADOR`. El error queda guardado en el hilo de transferencia abandonado y retiene la `memoryview` del mensaje serializado (`ForkingPickler.dumps` devuelve `buf.getbuffer()`). Al romper ese ciclo de referencias, el recolector de basura puede liberar el `BytesIO` antes que su vista.
- **Efecto.** No afectan a ningún resultado ni a ninguna publicación.
- **Reproducción.** Se reprodujeron de forma aislada: con envíos correctos no aparece ningún aviso; con un hijo que muere durante el envío aparece exactamente este error.
- **Comprobación.** Se verificaron en las 4 ejecuciones pesadas de R4: siempre 3 avisos de este tipo y ninguna excepción de hilos.
- **Evidencias:** `TDA17_E03_evidencia/AVISOS_HILO_ENVIO.json` y `repro_aviso_hilo_envio.*`.
- **Posible corrección:** limpiar el traceback del error en `_transferir`. Exigiría una nueva auditoría y un nuevo commit auditado.

## Condiciones vigentes

- **Publicación P3:** pendiente. Esta aprobación no está publicada: no hay commit ni push. Mientras P3 no exista en el repositorio autorizado, la ruta oficial de E03 queda bloqueada (`E03_P3_EXISTE` / `E03_P3_A3`).
- **MNQ:** esta aprobación NO autoriza la ejecución oficial sobre MNQ, que exige una autorización explícita posterior. La única vía para esa ejecución es `official_publish_th35_e03(P3, ejecucion=…)`, seguida de `check_official_publication_e03(P3)` y de la reproducción.
- **Holdout:** sigue cerrado.
- **E02:** sin cambios. La ruta oficial de E02 (`official_publish_th35(P2)`) sigue vigente y produce las mismas tablas.
- **Recursos de la ejecución oficial:**
  - control de memoria estricto, con reserva de 4096 MB y rechazo explícito por memoria insuficiente;
  - 1 hilo de OpenBLAS por proceso;
  - reanudación oficial deshabilitada;
  - un worker muerto, bloqueado o que no arranca produce `E03_TRABAJADOR` sin publicación.
