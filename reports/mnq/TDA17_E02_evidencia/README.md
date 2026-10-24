# TDA-17 E02: evidencias reproducibles del cierre

Esta carpeta contiene copias byte a byte (`.gitattributes`: `* -text`) de los registros del cierre de E02 sobre la versión definitiva v11 (v10 + corrección v11 de 3 ficheros de pruebas). Las evidencias de v10 están en el ZIP de cierre (`evidencia_v10/`).

## Alcance de las ejecuciones

Todo se ejecutó sobre el árbol de A2 v11:
- **Estado PENDIENTE** (repositorio del proyecto, HEAD = P1 `24f9f1d`): suite completa, focalizadas, T-MUT y mutantes manuales.
- **Estado APROBADO**, solo en clones temporales con el push deshabilitado:
  - la suite completa dentro del ensayo, con HEAD = A2 y la aprobación sin commit;
  - la suite completa con HEAD = P2.
- Los controles de seguridad, en los dos estados (`controles_seguridad_v11.txt`).
- No se ha ejecutado ningún análisis sobre MNQ.
- No se ha leído `data/raw` ni el holdout.
- Las rutas absolutas de los logs son las de la máquina donde se ejecutaron.

**T-MUT (`t_mut.py`).** Mutaciones de código fuente de los 68 controles:
- 45 `_fail` del cargador `tda17_enmiendas_e02.py`;
- 23 `_reject` de `tda17_causalidad.py`, `tda17_entradas.py` y `tda17_salidas.py`.

Cada mutante se ejecuta en un clon Git con **copias** independientes de los datos del manifiesto, nunca enlaces a los originales. Se repartió en 4 particiones, con un máximo de 2 simultáneas, cada una con su propio control sin mutar. La clasificación es estricta:

| Estado | Condición | Cuenta como detectado |
|---|---|---|
| DETECTADO | pytest termina con código 1, al menos un FAILED y ningún ERROR | sí |
| SUPERVIVIENTE | código 0 | no |
| INFRAESTRUCTURA | timeout, códigos 2–5, errores de recogida o de fixture, o ninguna prueba seleccionada | no; obliga a revisar |

**Mutantes manuales v8–v10.** Son plugins de pytest (`-p mut_manual_vN`, elegidos con `TMUT_MANUAL`) que reintroducen cada defecto corregido en la publicación oficial.

**Integridad de los datos.** `datos_antes.txt` y `datos_despues.txt` registran la huella de bytes, el tamaño, el `mtime` y el inodo de los 9 ficheros `data/` del manifiesto, antes y después de todas las ejecuciones.

**Ensayo A2 → P2.** Hubo dos ensayos v11 en clones temporales con el push deshabilitado:
- `ensayo_a2_p2.log` (`C:/tm18`): ENSAYO CORRECTO. Incluye la suite estricta completa en estado APROBADO con HEAD = A2 y la aprobación sin commit.
- `ensayo_rapido_sin_suite_tm14.log` (`C:/tm14`): ENSAYO CORRECTO. Sobre su P2 se ejecutó `suite_estricta_aprobada_p2.log`.

Los logs de ejecuciones interrumpidas (por falta de memoria o por un reinicio de Windows) no son evidencia y no se incluyen.

Cada ensayo copió esta carpeta en el estado previo a él, así que ninguna de esas copias contiene los logs del ensayo ni `datos_despues.txt`.

## Ficheros

| Fichero | SHA-256 (bytes exactos) | SHA-256 (CRLF a LF) | Contenido |
|---|---|---|---|
| `.gitattributes` | `705fd4d6451a31d36b3df7de96f83f30ac976c9b4a6d1e51671d8e2f33e2d0da` | `705fd4d6451a31d36b3df7de96f83f30ac976c9b4a6d1e51671d8e2f33e2d0da` | `* -text`: evidencias byte a byte |
| `cobertura_t_mut_v10_v11.json` | `1333e9f12e84e2ea1bd0748415a1ea821b101f7391de7e8a36435ebc188618de` | `1333e9f12e84e2ea1bd0748415a1ea821b101f7391de7e8a36435ebc188618de` | T-MUT: comparacion por control de las pruebas que detectan cada mutante, v10 frente a v11 |
| `controles_seguridad_v11.txt` | `f745299c3211e6844d101a7f443f42c5ea6e8b7021557368459a873a5a5fef33` | `be3605325b050544551c5e33248c73fcd8e60a6b46b1feb94bfec7f3f82f77cf` | v11: sonda de los controles de seguridad en estado PENDIENTE (proyecto) y APROBADO (HEAD = P2, C:/tm14/c) |
| `datos_antes.txt` | `6da9981f7003b1fe9bca0a5faf4cdc6e65b5626ba7651719bbfabfafb313d8d0` | `a78779da0ad65932b38e6b0892e04819bdaa3ad5f7ab1fead975a77d652cdb55` | estado de los 9 ficheros data/ antes de las ejecuciones v11 |
| `datos_despues.txt` | `6da9981f7003b1fe9bca0a5faf4cdc6e65b5626ba7651719bbfabfafb313d8d0` | `a78779da0ad65932b38e6b0892e04819bdaa3ad5f7ab1fead975a77d652cdb55` | estado de los 9 ficheros data/ despues de las ejecuciones v11 (identico) |
| `datos_integridad.py` | `4378cc89cc8364fa3529456cc5525719ee3e36fff2f7f9442cd2c91c61751a39` | `4378cc89cc8364fa3529456cc5525719ee3e36fff2f7f9442cd2c91c61751a39` | huella de bytes, tamano, mtime e inodo de los 9 ficheros data/ del manifiesto |
| `ensayo_a2_p2.log` | `0ecf62e71d067f9510000398f35dfe129ecf5f5260984d44f6afaa1a49655409` | `14b43c5a6b47b9dedc8ea7a2c807ef6548807081bdfe4677255a7cf309d0412f` | v11: ensayo completo A2 -> P2 con la suite estricta en estado APROBADO (HEAD = A2 + aprobacion sin commit, clon C:/tm18) |
| `ensayo_rapido_sin_suite_tm14.log` | `02e7768dd9fa1fc94b7117f9c31871683bd16b42fbe28a805ef6a2a7b5ef03f9` | `937d516c40620a5874c550ed27c7eb52ecda11dc35d1dca18adb7f50945e71ec` | v11: ensayo A2 -> P2 sin suite (clon C:/tm14), base de suite_estricta_aprobada_p2.log |
| `focalizadas_pendiente.log` | `603220d38b65e4017e80c4fe761cb76c53e1babdf8acfdd52ff3b7182406b19b` | `603220d38b65e4017e80c4fe761cb76c53e1babdf8acfdd52ff3b7182406b19b` | v11: pruebas focalizadas (salidas, motor, enmienda_02, procedencia) en estado PENDIENTE |
| `mut_manual_v10.py` | `bdde76c9c6d458899a56145cbc37f4e99d7de6902e1d141b519e3eab18d65ffd` | `bdde76c9c6d458899a56145cbc37f4e99d7de6902e1d141b519e3eab18d65ffd` | mutantes manuales v10 (SIGUE_ENLACES, SIN_TIPO, EXISTS_SALIDAS, EXISTS_VERIFY) |
| `mut_manual_v8.py` | `b6a40c2af1774f0362c195d8e3dae11cdd32a4ed5d7ca47888fd84f8807190b0` | `b6a40c2af1774f0362c195d8e3dae11cdd32a4ed5d7ca47888fd84f8807190b0` | mutantes manuales v8 (C1_UNLINK, C2_SOLO_BYTES, REPLACE) |
| `mut_manual_v9.py` | `9c7174690cdbf88e86c6e395a3145beb772bedd84ac98b7bfcaaf93ee9023f3b` | `2bdbc0232961ba2e40804a377ec73902d327f60ba46d25a24bda8bde43175630` | mutantes manuales v9 (V8_STAT, SIN_TIPO, EXISTS) |
| `suite_estricta_aprobada_p2.log` | `e76ca3ba11fe3eea5f4a80e18a76d404ad85a085ed9c772e027180146e70840e` | `e76ca3ba11fe3eea5f4a80e18a76d404ad85a085ed9c772e027180146e70840e` | v11: suite completa estricta en estado APROBADO con HEAD = P2 (clon temporal C:/tm14, relanzada sola tras el reinicio) |
| `suite_estricta_pendiente.log` | `aa18614cefe2adda1d0cf8319210ba791c86f4fd3295a2bdb8b8ebef496eda13` | `aa18614cefe2adda1d0cf8319210ba791c86f4fd3295a2bdb8b8ebef496eda13` | v11: suite completa estricta en estado PENDIENTE (arbol de A2, repositorio del proyecto) |
| `t_mut.py` | `e984cab4d01eb685385f4cde052780fb9cde0ff158c8fe097721fa33a9e87d49` | `d5fff09a8cb1cfeaf3552d531004ccd22e7317f533d318b488936eabf6fffe7e` | arnes T-MUT (copias de datos, clasificacion DETECTADO / SUPERVIVIENTE / INFRAESTRUCTURA) |
| `t_mut_final_part0.log` | `176176058a4dc320787f7d8577937b21735ab32d26d778ac9b18ae9ee56c276d` | `7a87df95b6f066c1b16a0513fc3e266f8bd595b39204ad43fd235f664f0f7c22` | T-MUT final, particion 0 (log) |
| `t_mut_final_part1.log` | `55336f325c927416ac4329eb39621ac7eb2b8afb51df54d2c5ad5613100594b7` | `8a482c36a14d8b3b8318b5414a04896d60cca407e74a9b7f2d71b05386ac3586` | T-MUT final, particion 1 (log) |
| `t_mut_final_part2.log` | `4075a699fbe1ae9c61c12bb5d5b0d10d4639fd299247deeeb61d70720ba93b35` | `a03ffe833da5f6a03dd547f6f6bdc01679e0ab0d7749f96bbe4aa223b8f4af5b` | T-MUT final, particion 2 (log) |
| `t_mut_final_part3.log` | `6f5bf35285fcc9e1a7a3cd1ab8ab31eca15e2a3ec6606875ad5ee457f0e43ad3` | `8d76d9a65ed356f7ac507a75a263734f3d2ea8e50b2c157f70b6ad134ee91b37` | T-MUT final, particion 3 (log) |
| `t_mut_final_resultados_part0.json` | `705356c187130801827959677e63e9acd723736230171ca6b76532593d451e42` | `746006510fedea00b0774c7a971584c58c10a90de4da624852d4c8e4772d7e4f` | T-MUT final, particion 0 (resultados JSON) |
| `t_mut_final_resultados_part1.json` | `a4fe438ced594cc55e811b06fcbbcc653765ab8dfa6125b426c04294cef56a79` | `9774b846f89006d060218a7201bd16c41ec64cad4690b2b537847bfd12e24500` | T-MUT final, particion 1 (resultados JSON) |
| `t_mut_final_resultados_part2.json` | `63f190cedf65801ef5f19683ae755329ba1bfd5425671e661b327ce4cea8a2d2` | `e53bde0de504c01873ff60ca6a87eae3c3e30bfe3c8139c83d6299d45dff4a2f` | T-MUT final, particion 2 (resultados JSON) |
| `t_mut_final_resultados_part3.json` | `2eb8cc00a8e4b5ae1dacb9869761c8adaeec705e289092ec85c66171d55ba98a` | `87bb898ff6c470b451460708839a3506b265664a434c89ccccd50c95da8a93ff` | T-MUT final, particion 3 (resultados JSON) |
| `t_mut_resultados_final.json` | `a0f3c9c5b8aa53618189000669a10dfcb2994cad0dcecfc85d6617abb1765019` | `a0f3c9c5b8aa53618189000669a10dfcb2994cad0dcecfc85d6617abb1765019` | T-MUT v11: 68 controles, resultados de las 4 particiones unidos |
| `t_mut_v11_manuales.log` | `747b35e53f1dfacf7b5ac668ea9396f04b6601d01b328d1f43352f97fe454328` | `747b35e53f1dfacf7b5ac668ea9396f04b6601d01b328d1f43352f97fe454328` | v11: los 10 mutantes manuales v8-v10 repetidos sobre las pruebas v11 |
| `verificadores_v11.txt` | `8303d01a16213a90398d628271e8e0c7021cf30a5aded89aab1e557775375e74` | `8303d01a16213a90398d628271e8e0c7021cf30a5aded89aab1e557775375e74` | v11: verificadores del protocolo, E01 y E02 y comprobar-a2 en el repositorio del proyecto |
