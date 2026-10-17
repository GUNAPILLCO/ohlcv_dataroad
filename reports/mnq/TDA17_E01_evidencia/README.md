# TDA-17 E01: evidencias reproducibles

Copias byte a byte (`.gitattributes`: `-text`) de los registros de la auditoria de E01.

Alcance de las mutaciones: el arnes copia `src/` y las pruebas a un directorio SIN `.git` y ejecuta
pytest sobre esa copia. Sus controles se reprodujeron con el registro de enmiendas PENDIENTE
(`aprobacion: null`), no directamente sobre el commit P: con el registro APROBADO la via de produccion
exige el repositorio Git y el commit auditado, de modo que el control sin mutacion del arnes fallaria
en esa copia. Cada JSONL registra el comando, la huella del arnes y las huellas de los ficheros
probados (`sha256_repo`), que corresponden a la version vigente en cada ejecucion:

- M01-M54 (`mut_engine_54.py`): `tda17_enmiendas.py` 338d5a7b..., `test_tda17_enmienda_01.py` 5bee4a7c...
  (correccion F1-F3).
- M55-M57 (`mut_engine.py`): `tda17_enmiendas.py` 6c5d96d0... (version de A), `test_tda17_enmienda_01.py`
  5a48d780... (antes del parche aprobado de las 4 pruebas de estado del registro, que da 25a86dbe...).

Las rutas absolutas de los logs son las de la maquina donde se ejecutaron.

| Fichero | SHA-256 (bytes exactos) | SHA-256 (CRLF a LF) | Contenido |
|---|---|---|---|
| `suite_f2git.log` | `5b3c1ee84eaeece31320b247b667b9b8b290fc13079b5675d7eb69d3e33dfced` | `8cf2f41c310078c5957f8c600309f36290b1fee9369d7250d808f9ce119f27d9` | Suite completa estricta tras la correccion F2 (antes del parche de estado del registro) |
| `suite_pendiente_preA.log` | `875537a8e08fa787ff5878f6c7cde0f03284750a6086f3f761b93ba89660604f` | `66b505243ba36fb844ac5517b6b6c4c97e4a4b7f1d21b347e09342dd9cb19352` | Suite completa estricta con E01 PENDIENTE y el test corregido, inmediatamente antes de A |
| `mut_engine_54.py` | `433d637496f3fc73134e8dfabcd9672b1d097991df4b1b7fa404396c8a005d22` | `433d637496f3fc73134e8dfabcd9672b1d097991df4b1b7fa404396c8a005d22` | Arnes de mutaciones M01-M54 (ejecucion completa 54/54 y M46) |
| `mut_engine.py` | `38ddb03e74582505e971b04ac75661a3913ee09afdbb4864706823eb398446df` | `38ddb03e74582505e971b04ac75661a3913ee09afdbb4864706823eb398446df` | Arnes de mutaciones actualizado con M55-M57 |
| `mut_e01_completo.log` | `39de4a8ba5646e102a038d67b17068e94d2be4062c097bb1bea8fc13c9369f08` | `39de4a8ba5646e102a038d67b17068e94d2be4062c097bb1bea8fc13c9369f08` | Ejecucion completa M01-M54: 54/54 detectadas |
| `mut_evidencia.jsonl` | `a51b69971817208c2ae7e81f7156778eee8ee0ef4513045204e9e591a6f23358` | `7c995a560fa1aeb81c142f4ba69a7135d007524f2d1b5e9ef7336512291a22c4` | Evidencia JSONL de la ejecucion completa M01-M54 |
| `mut_M46.log` | `bd18d4a935e8373407d8a46dbc47db30ba93583f40213ad0b3189547375ecfd8` | `989f2fd11586e3e29359c5f9b748c63dfab16be234d0490d5445b123b1a888f5` | Ejecucion individual de M46 |
| `mut_evidencia_M46.jsonl` | `2fd5e57a3b846bb7fb0fc06fcf9be459d6877f6885752faff6c74e19b9453c28` | `5bcf12f7d5c9352a0a0b922a1adf342150faf2c6aa1a6b77ce5737239702ae66` | Evidencia JSONL de M46 |
| `mut_M55.log` | `9d7e9150d396e3fb56342b1a2dfa4b2b0e6840f2221fe31ebd67ab73d1464c68` | `d568b3d5e98322867a542fe4fadd382791f8fa269c8b8fc8bb7dbc724d4ab8cf` | Ejecucion individual de M55 (barrera Git del commit auditado) |
| `mut_evidencia_M55.jsonl` | `669ee63e73d5ad040c5fd56f61f6831fa97816dad8d24cc1ca096ce712481fc3` | `251ee75cd417dc08b5cf314e4017fff80ea29d59ce805bb8cf5fd7343f6cae99` | Evidencia JSONL de M55 |
| `mut_M56_M57.log` | `06c7c9da25e9ae0c3c25e4b9dfc5a98a772175b117cf1667e6443ad74fb5b0c2` | `01f67392892bedf5331b29cf8d4ae56993239682d096022a6bf9635f0c32eaaf` | Ejecucion de M56 y M57 |
| `mut_evidencia_M56+M57.jsonl` | `ef83fe873ead75faf2102626d39bb1f091f863a920cb2039fb6f33aea5c45f6e` | `7af2fa91b81bff9ad96d624ee7d54a9935271b241261b5ee8bbc73506054bc34` | Evidencia JSONL de M56 y M57 |
