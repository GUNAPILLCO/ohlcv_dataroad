# TDA-17 E03: evidencias de la aprobación definitiva (R4)

Esta carpeta guarda copias byte a byte de las evidencias de la aprobación de E03 R4 (`.gitattributes`: `* -text`). La aprobación cita el commit auditado A4 `7fc78bcbc28f5e94c02bb545f46c7e89201408a2`.

## Origen

- `R3_*` y `R4_*`, `AVISOS_*`, `repro_aviso_*` y `verificar_avisos.py` se extrajeron de los paquetes auditados, después de verificar su SHA-256:
  - `PAQUETE_AUDITORIA_R3_E03.zip` `4cbf5ecf914d6024c9a6425b9b6d3b5a3eaf0550fd7e7adec5d4e949dfb290f9`;
  - `PAQUETE_AUDITORIA_R4_E03.zip` `73db0a649cdd934ea05d3e0805faab2cd22c5435f3582635169af0c94f4e2d27`.
- No se ha usado nada de la aprobación preparada que citaba A3 (obsoleta).
- `A4_*`: creación del commit A4. Las suites de R4 se ejecutaron sobre A4' (clon temporal, push deshabilitado), cuyo árbol es idéntico al de A4.
- El resto de ficheros corresponde a esta aprobación: registro sin commit sobre A4, en el repositorio principal.
- Todo se ha hecho con datos sintéticos. No se ha leído `data/raw` ni el holdout, y no se ha ejecutado ningún análisis sobre MNQ.

## Avisos conocidos

Cada ejecución de las pruebas de E03 produce 3 avisos `BufferError` sobre `BytesIO` del hilo de transferencia tras un envío fallido a un worker muerto. Se conservan documentados sin modificar el código (ver el acta, `AVISOS_HILO_ENVIO.json` y `repro_aviso_hilo_envio.*`).

## Ficheros

`verificadores_despues.txt` no aparece en la tabla: lo genera después `verificar_aprobacion_a4.py`, que comprueba esta tabla.

| Fichero | SHA-256 (bytes exactos) | Contenido |
|---|---|---|
| `.gitattributes` | `705fd4d6451a31d36b3df7de96f83f30ac976c9b4a6d1e51671d8e2f33e2d0da` | `* -text`: evidencias byte a byte |
| `R3_PAQUETE_AUDITORIA_R3_E03.zip.sha256` | `7ab326b28bde3e9d0632699501b7836105a7745e159bf706a7aee31bcaf8fadc` | R3: SHA-256 del paquete R3 auditado |
| `R3_MANIFIESTO_SHA256.txt` | `a639a7e91ac40d3a0dd6d3862224872c19d11fee003bde9c895184569a9406ff` | R3: manifiesto SHA-256 del paquete R3 |
| `R3_INFORME_R3_E03.md` | `987d1345bbb91069d979c13e8740e4abe802a1de7a46940b0520d6e08ae3055d` | R3: informe de la ronda R3 (auditada) |
| `R3_suite_completa_R3.log` | `f0420b61af2c1757a1f9350242fa21569e57be969e43518dc80c022d23c45ce3` | R3: suite estricta completa (1890 passed, 13 skipped, rc=0) |
| `R3_gobernanza_R3.log` | `53bb9420aa4a7a0b16760b54c16879da27de637cce7f30cb1c6349fa8f52050f` | R3: pruebas de gobernanza focalizadas |
| `R3_post_lf_R3.log` | `7bf7c4d5046c1cf1ab013acdefd7398c6ab223551784a019e83bbf528d4e1374` | R3: comprobaciones tras normalizar a LF (11 passed) |
| `R3_benchmark_MD_e2e_p8_estricta_completo.json` | `f55565a7c5e6574a2683ffe9ae7c961978232da79ecd3d7c68c83caee3cc07c4` | R3: extremo a extremo MD, 300 replicas, 8 procesos, memoria estricta, BLAS=1; resultado completo identico a E02 |
| `R3_benchmark_MD_e2e_p8_estricta_solo_tablas.json` | `badb195139079ced37a43762e47c2b17a3b5c861dc7720cf10c9b528192eee9a` | R3: primera pasada MD (solo tablas) |
| `R3_benchmark_SB_e2e_p8_estricta_2.json` | `a72f30c516c6eaea3ce4816177755ea63c3ce6990b38be99f215b975f32c854d` | R3: caso SIN_BLOQUEO; 5 tablas y resultado completo identicos a E02 |
| `R4_PAQUETE_AUDITORIA_R4_E03.zip.sha256` | `905b1ada338245d9c9b56922ce6eb61d65452a466f1b033aa7730e913762d835` | R4: SHA-256 del paquete R4 auditado |
| `R4_MANIFIESTO_SHA256.txt` | `8715f6ec5a720d661b5fd762891cb0272e53a245f56b028e146f4eb11169b6f6` | R4: manifiesto SHA-256 del paquete R4 |
| `R4_INFORME_R4_E03.md` | `78f7cacabf78ea028cd6406e5b62b630e33c2a9e724c6b9ff54c22bb4b873f0d` | R4: informe de la ronda R4 (auditada) |
| `R4_PROCEDIMIENTO_A4_aprobacion_P3.md` | `98a316e2ed055f31ff5dcafff35241b6270d9c50dba9238f943da66e482bd0d5` | R4: procedimiento auditado A4 -> aprobacion -> P3 |
| `R4_HUELLAS_E03_A3_vs_A4prima.json` | `4caced5b9e38a19114ec13ec885995e52c823f964fc23308334a5dedf3b82f94` | R4: huellas de los 8 auditados en A3 y en A4' |
| `R4_DIFF_R4_A3_a_A4prima.patch` | `105bf43d8c155b2bc2bdc0bfa17001a2c98817ac4dc0268614734af8d72cb81f` | R4: diff unico auditado A3 -> A4' (= A3 -> A4 byte a byte) |
| `R4_e03_PENDIENTE.log` | `49b3317d5c68806126f6cf5fb1b86c98ea1c8509c459ddb43c56ce3f849a139a` | R4 1/4: pruebas E03 en PENDIENTE (121 passed, rc=0) |
| `R4_e03_APROBADA.log` | `040f7b2d94d94d5ca4c1cd81d0c896ab88d94ebe97a709ef72b20f1c885b3048` | R4 2/4: pruebas E03 en APROBADA (121 passed, rc=0) |
| `R4_suite_PENDIENTE.log` | `cddff6394fb9cb6eac630a3a924b810751fb5a1f8496e35434226934590f0399` | R4 3/4: suite estricta completa en PENDIENTE (1917 passed, 13 skipped, rc=0) |
| `R4_suite_APROBADA.log` | `5a78a5939cc09fe7ba4bf70766ed56a5fdad50c44bc2ede03abd060ad56cc1e8` | R4 4/4: suite estricta completa en APROBADA (1917 passed, 13 skipped, rc=0) |
| `R4_mutantes_PENDIENTE_A4prima.log` | `386dc97a6425ab302b92ff4da07df1ac24c5ee0f3b51a056811f43985897d6ac` | R4: mutantes de aprobacion M1-M8 en PENDIENTE (8/8) |
| `R4_mutantes_APROBADA_P3prima.log` | `6e55a92a41cc91a863f67b037c95a715f014e6014eb907881abfbe6f934e9dee` | R4: mutantes de aprobacion M1-M8 en APROBADA (8/8) |
| `R4_mutantes_motor_PENDIENTE.log` | `6238b6444a5f699235591cb3f3a61abe1e35b132fd6b4a4844b084bd8f68f6ce` | R4: mutantes del motor MM1-MM8 en PENDIENTE (8/8) |
| `R4_mutantes_motor_APROBADA.log` | `728d91583ab7b0089a7ef4939a9ca72d987debd9ac624a5c3146679758e58f5f` | R4: mutantes del motor MM1-MM8 en APROBADA (8/8) |
| `R4_mutantes_motor_R3_combinado.log` | `db1a116f58f25cad2ace0204d97f7d843d7a66929bcbef1c32a4df2aa70d5897` | R4: MM0 (comportamiento de R3) detectado |
| `R4_repro_A4_sin_corregir.log` | `a0936f3e55b073697f8a113d7bdbd0c4316a0e6e1ba5c9110927b990bf22ba90` | R4: reproduccion del bloqueo antes de la correccion |
| `R4_repro_mecanismo.log` | `cee3beb9d2ddc17dfb3f86bde0cabab358981108ff848befd650fcfe5d261baf` | R4: mecanismo del bloqueo en un pipe puro |
| `R4_repro_A4prima_final.log` | `aa1568202e4690e5191c31b2c41d051a04d2d2c904da9b7a87b553ef5c23725b` | R4: reproduccion con el motor corregido (12/12 E03_TRABAJADOR) |
| `AVISOS_HILO_ENVIO.json` | `b94d8ecfec6eb253c938d9dac4e4ed5a5343e90a7cebc2250146972e40a7b2cb` | avisos BufferError conocidos: 3 por ejecucion en 1/4-4/4, sin excepciones de hilos |
| `repro_aviso_hilo_envio.py` | `42fd56a6b94affd0f5b024519ada810d1efe35dd9bfb03a7f9695d3755274624` | reproduccion aislada del aviso BufferError |
| `repro_aviso_hilo_envio.json` | `5d773b6fdd9f80737818cc0a9e3786437871f3ee55b31f4dfacf3efc1685bc79` | resultado: 0 avisos con envio correcto, BufferError con envio fallido |
| `verificar_avisos.py` | `c93866358012778a4b2b63619531f4cd1b8c9773b038595c580d36ba6a435405` | clasificador de avisos de los logs |
| `A4_verificar_a4.py` | `8209aa36fcb5ba3211cc776bb4e0b9075dfe598d7910f356072dad65030c77dc` | A4: script de verificacion del commit A4 |
| `A4_verificacion_antes.txt` | `9d58489a6c841c69a0ed194044f72dd567f8cf488c2028f92775c51b1f94f80e` | A4: antes del commit (HEAD = A3): TODO CONFORME |
| `A4_verificacion_despues.txt` | `5285411c33c423c36e0c2257e3414ad2198d03b4f66f0c81d30beb3b04daec51` | A4: despues del commit: TODO CONFORME (diff identico al auditado) |
| `verificadores_antes.txt` | `a4c7da301f64746c9aa1b5dd9db4d403a9fef6f071ad88feed0e9b45f5ad29c3` | aprobacion: estado de partida (HEAD = A4, PENDIENTE, arbol limpio): TODO CONFORME |
| `registrar_aprobacion_e03.py` | `557409d9809dce23036cbf4790305e344c08120e5aa030eec387038de3304089` | script que registra la aprobacion citando A4 (solo estado y aprobacion) |
| `registro_aprobacion_salida.json` | `e0a5cd90efb50c1805d24d4c4602581a80080c7f5b689c6b510cf093615a40c6` | aprobacion registrada (salida del script) |
| `controles_aprobada.py` | `64c5c63966ad92b9f014a52f33b671a471d1a4793f45b720f1c06e67519c635b` | controles de la aprobacion: cita de A4 y ruta oficial bloqueada sin P3 |
| `controles_aprobada.txt` | `465c87809b1e829455f14639e82774c513b47d74d406999886dc14a3a65c2943` | resultado de los controles en estado APROBADO sin P3 |
| `test_estado_proyecto_aprobada.log` | `47cc1cb01df14f62e3b88db07e5d59204e7b2c3a37215b4f4df88e27b66dc3b9` | prueba de estado del proyecto en APROBADA: 1 passed |
| `cobertura_aprobada.log` | `bd9cc827ac83afbf8c80fe334798b194feac23a40f664ebc02bd43de38c73dcc` | pruebas de cobertura de E03 en APROBADA: 9 passed |
| `verificar_aprobacion_a4.py` | `c92c546b7c181da4a853ffe98144a039a911bf1450f886600e7e336acbde0553` | script de verificacion de la aprobacion (produce verificadores_despues.txt) |
