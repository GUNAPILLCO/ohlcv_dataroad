# TDA-17 — Cierre formal de E02: versión definitiva, alcance y procedimiento A2 → P2

**Estado:** `PENDIENTE_AUTORIZACION_FORMAL`.
- La implementación definitiva de E02 es la versión **v11**:
  - la v10, aprobada técnicamente por la auditoría independiente;
  - más la corrección v11, autorizada y limitada a 3 ficheros de pruebas (§1), que está **pendiente de auditoría**.
- El código de producción, el YAML de E02, el protocolo y E01 son idénticos a los de v10.
- El registro sigue en `PENDIENTE_AUDITORIA` con `aprobacion: null`.
- HEAD = P1 `24f9f1d`, sin commits nuevos.

**Lo que este documento NO hace:**
- no aprueba E02;
- no crea commits;
- no publica nada;
- no autoriza ejecutar sobre MNQ.

No se ha leído `data/raw` ni el holdout. Ningún análisis se ha ejecutado sobre MNQ.

**DECISIÓN IRIS: NINGUNA.**

Este documento consolida las rondas v5–v11 y sustituye a la sección 7 del paquete de auditoría (`TDA17_E02_paquete_auditoria.md`) en lo que respecta al procedimiento A2 → P2.

## 1. Implementación definitiva (consolidación v5–v11)

| Ronda | Cambio | Ficheros | Resultado de su auditoría |
|---|---|---|---|
| v5 | Decisiones 14–19 implementadas: S-a, N-a, R-a corregida, esquema mínimo de cinco tablas, ruta única `official_publish_th35` con `TDA17_procedencia_oficial.json`, publicación con bloqueo | YAML, cargador, `tda17_analysis.py`, `tda17_salidas.py` y pruebas | favorable; pidió la publicación transaccional (v6) |
| v6 | Publicación transaccional y comparación estricta de bytes | `tda17_salidas.py`, `tda17_enmiendas_e02.py` (`verify` también ante una publicación interrumpida) y pruebas | carrera de `os.replace` (corregida en v7) |
| v7 | Promoción atómica sin sobrescritura (`os.link`), compatible con Windows | `tda17_salidas.py` y pruebas | C1 y C2 de recuperación (corregidos en v8) |
| v8 | Propiedad demostrable al recuperar o deshacer | `tda17_salidas.py`, pruebas, `.gitattributes` y `.gitignore` | enlace simbólico en `_owned` (corregido en v9) |
| v9 | Rechazo de enlaces simbólicos en la propiedad | `tda17_salidas.py` y pruebas | aprobada; enlaces en el verificador (corregidos en v10) |
| v10 | Verificador que exige ficheros regulares | `tda17_salidas.py`, `tda17_enmiendas_e02.py` (una línea en `verify`) y pruebas | **aprobada técnicamente** |
| v11 | Pruebas válidas con el registro del proyecto PENDIENTE y APROBADO; A2 sintética siempre PENDIENTE | solo `test_tda17_e02_motor.py`, `test_tda17_enmienda_02.py` (incluido `build_repo`) y `test_tda17_procedencia.py` | **pendiente de auditoría** |

Detalle de cada ronda:
- **v6:**
  - Cálculo y serialización antes de tocar el disco.
  - Cerrojo del sistema operativo (`SALIDA_CONCURRENTE`).
  - Preparación con `fsync` y diario de huellas; registro de procedencia como punto de confirmación.
  - Recuperación de restos; `SALIDA_INCOMPLETA`.
  - Comparación de bytes exactos (`SALIDA_NO_REPRODUCIBLE`).
- **v7:** `os.link`, que falla si el destino existe. Un destino creado después de la comprobación nunca se sobrescribe (`SALIDA_EXISTENTE`).
- **v8:**
  - El preparado sigue enlazado y ancla la identidad.
  - Un fichero es propio solo con la misma identidad que su preparado y los bytes de su diario.
  - Deshacer retira todo lo demostrablemente suyo.
  - La recuperación conserva y bloquea lo no demostrable (`SALIDA_RECUPERACION`).
  - Excepciones `-text` solo para las 7 salidas oficiales; el cerrojo y la preparación se ignoran.
- **v9:** la propiedad se lee con `os.lstat`. Un enlace simbólico u otro punto de reanálisis nunca es propio, y un enlace roto cuenta como salida presente.
- **v10:**
  - `check_official_publication` y `reproduce_official_publication` leen el registro y las tablas como ficheros regulares, sin seguir enlaces.
  - Los enlaces rotos son salidas presentes.
  - `verify` detecta los enlaces rotos.
- **v11** (corrige el bloqueo del ensayo v10, en el que 13 pruebas fallaban en estado APROBADO; ver §8). Diff en `diff_v10_v11_pruebas.patch` del ZIP:
  - **Funciones auxiliares** de `test_tda17_enmienda_02.py`, que importan las otras dos pruebas:
    - `project_e02_approved()` lee el registro del proyecto y falla con cualquier estado que no sea PENDIENTE o APROBADA.
    - `project_rejection(p2)` da el motivo exacto del rechazo de una P2 inexistente: `[E02_PENDIENTE]` con el registro pendiente y `[E02_P2_EXISTE]` con el aprobado.
    - `pending_registry()` devuelve la forma PENDIENTE del registro, que solo cambia `estado` y `aprobacion`.
  - **`build_repo`** escribe en la A2 sintética el registro PENDIENTE (`pending_registry`) en lugar de copiar el del árbol. Con el registro del proyecto pendiente, el resultado es idéntico byte a byte. La aprobación simulada cita siempre la A2 sintética.
  - **Pruebas sobre el repositorio del proyecto**, que admiten los dos estados como en E01:
    - Con el registro PENDIENTE, las aserciones son exactamente las de v10.
    - Con el registro APROBADO, se comprueban:
      - la aprobación: campos exactos, A2 con P1 como único padre y HEAD igual a A2 o descendiente suyo, huellas auditadas iguales a las del árbol y registro PENDIENTE en A2;
      - la configuración de producción: activable, sin ejecución oficial y sin P2;
      - los rechazos: `require_e02_oficial` rechaza con `[E02_NO_OFICIAL]` y `load_official_inputs` no lee ninguna entrada con la configuración preparatoria;
      - las rutas oficiales: con una P2 inexistente, todas rechazan con el mensaje exacto `[E02_P2_EXISTE]`, y el CLI informa «ejecución oficial: NO».
  - **Prueba nueva** `test_synthetic_a2_registry_never_depends_on_the_project_state`: la A2 sintética siempre está PENDIENTE y la aprobación simulada cita la A2 sintética, nunca la del proyecto.
  - **Sin cambios:**
    - los nombres de las pruebas existentes, de los que dependen los selectores de T-MUT;
    - el código de producción, el YAML, el protocolo y E01.

**Sin cambios desde la aprobación metodológica:**
- el YAML de E02 (`fd7478d5…`);
- las decisiones estadísticas;
- los valores congelados;
- el protocolo `9cdea2f`;
- E01 (8 ficheros);
- las propuestas autorizadas (`192c5a43…`, `f16672c3…`) y el anexo de relaciones N (`8cac7280…`).

## 2. Huellas definitivas (SHA-256 tras normalizar CRLF → LF, `sha256_text_file`)

**Los 15 ficheros de `AUDITED_E02`.** Son los que irán en `sha256_auditados`. Los 3 marcados con (v11) cambian respecto a v10, y el resto queda idéntico.

| Fichero | SHA-256 | Líneas |
|---|---|---|
| `configs/tda17_enmienda_02.yaml` | `fd7478d557e9a798c52b2af0806e89a6945c90cf5d8f35ea328afb581ad0d317` | 330 |
| `src/ohlcv_dataroad/ingest/tda17_enmiendas_e02.py` | `9484cb1ebc1bca7367a2131fa5881d059670319a230e7eb1aec81d3efa4f3c05` | 549 |
| `tests/test_tda17_enmienda_02.py` (v11) | `0a62fddbaa5b30a8a0ce0eba1200d15fa12dc5b0a738a4c25831545f7813e449` | 953 |
| `src/ohlcv_dataroad/ingest/tda17_analysis.py` | `f26bf380e0bbc2874c1939873a995513a693de6d4b848fdb8c1be88c26515fe0` | 1572 |
| `src/ohlcv_dataroad/ingest/tda17_g2.py` | `586b97b79cde7730a5d27b71935d8263387c2ae1f4111ae22564d20d2902fb6b` | 478 |
| `src/ohlcv_dataroad/ingest/tda17_causalidad.py` | `1f5b6f3b1348762d83c11ca6b581d04ee945049531a8a878ca0cc05cb3910e89` | 523 |
| `src/ohlcv_dataroad/ingest/tda17_salidas.py` | `c8c3272caacaf333f714d0abaa78243ef27f2599c58e8b782a1877fb7e0468e0` | 509 |
| `src/ohlcv_dataroad/ingest/tda17_entradas.py` | `fae54edd2773c7244a16d1e309643a7acf33d32fe9be3528c89f9f03134e84fa` | 81 |
| `tests/test_tda17_analysis.py` | `acf1278fe3a9013b51e6510990ac65f70057f45e648ddbb212db3b46f0d79eab` | 751 |
| `tests/test_tda17_g2.py` | `33f820ff9c4a66afadd333954af28243e45577a1c41753675be2dab3ea67cbcd` | 386 |
| `tests/test_tda17_rango_diagnostico.py` | `61aefb2dcbd80fe53ec43675ef4b353066faf1a94a4fc4840cfbccf3eb8d9ddf` | 120 |
| `tests/test_tda17_causalidad.py` | `a4508c40a223c7bdc153065fffd001c9efef91ca32c0d582620b7a006065baa0` | 368 |
| `tests/test_tda17_e02_motor.py` (v11) | `f9ebbfbf1e01d549310812e806b065161a548fe9a4fda7b961dac0dcfeafe8c8` | 691 |
| `tests/test_tda17_procedencia.py` (v11) | `28d1717a44eac70727cacd35946354fc04db077cbd2a64b5c3aa4e9512e01eed` | 287 |
| `tests/test_tda17_salidas.py` | `20be16897d9ef92bfe9a05a306a3809353dbbcad1aa6e761250cc722d4fcb917` | 879 |

**Otros ficheros de A2:**

| Fichero | SHA-256 | Nota |
|---|---|---|
| `reports/mnq/TDA17_enmiendas_E02.json` | `698873be6d7e6fb1a19e35772e8fa045b2df30e82b493fa08feb4c6f25cc5159` | registro, `PENDIENTE_AUDITORIA` |
| `reports/mnq/TDA17_E02_propuesta_decisiones_normativas.md` | `192c5a43be79835d620ee7484df1b8f8242e8ef73e49a70f182b6109cb92f0a0` | autorizado, sin modificar |
| `reports/mnq/TDA17_E02_propuesta_ambiguedades_y_salidas.md` | `f16672c3ac80505e3a70a0d1ef68619dff56175da9f153bab936f0c117cb46d6` | autorizado, sin modificar |
| `reports/mnq/TDA17_E02_anexo_relaciones_N.md` | `8cac7280d7dbb414bfc60909b162d0a585b954e9d5bdfb6efdf1a51c1112f7ac` | autorizado, sin modificar |
| `reports/mnq/TDA17_E02_registro_decisiones.md` | `896acafb2937f0d45c89e931af1d47e54cef26205ecbda00353a7606bb9977b3` | sin cambios |
| `reports/mnq/TDA17_especificacion_migracion_M2.md` | `088504a0875a5c935ce3c00966bdb8f89338a4deff2fab97df4f0a6bd456ac69` | SPEC-M2 |
| `reports/mnq/TDA17_inventario_decisiones_pendientes.md` | `b42e1c49bf9e20a8453d933d1008a2462453e64c6d305edf582df5559fd76ec9` | inventario |
| `.gitattributes` | `090187848747b9cdef9077d13082a6d4fae82fcb475cab6e93c94bfe664cd44c` | 7 excepciones `-text` (v8) |
| `.gitignore` | `56c8310e619d6028e70653a35efecc98a96a75dd55cc9fddab8b16aa5b90d27c` | cerrojo y preparación (v8) |
| `reports/mnq/TDA17_E02_paquete_auditoria.md` y este documento | (se recalculan en `huellas_finales.txt`) | documentación |

**Nota sobre `.gitignore`.** En `huellas_v8.txt` figura `a38eabeb…`, calculada sustituyendo `\r\n` por `\n` en los bytes. Aquí se usa `sha256_text_file` (lectura de texto), que da `56c8310e…`. Es el mismo contenido.

## 3. Pruebas finales sobre la versión definitiva (v11)

| Ejecución | Estado del registro | Resultado |
|---|---|---|
| Suite completa estricta (`suite_estricta_pendiente.log`) | PENDIENTE (repositorio del proyecto, HEAD = P1) | 1796 passed, 13 skipped, código 0 |
| Focalizadas: salidas, motor, enmienda_02, procedencia (`focalizadas_pendiente.log`) | PENDIENTE | 192 passed, 11 skipped, código 0 |
| T-MUT, 68 controles (`t_mut_resultados_final.json`) | PENDIENTE (clones `C:/tm17a..d`, máx. 2 particiones simultáneas) | 67 DETECTADO, 1 SUPERVIVIENTE (`E02_REMOTO_HISTORIA`), 0 INFRAESTRUCTURA; 4 controles sin mutar con código 0 |
| Cobertura T-MUT v10 → v11 (`cobertura_t_mut_v10_v11.json`) | — | ninguna detección perdida; único cambio: `E02_PENDIENTE`, mismas 10 pruebas (8 solo cambian de identificador de parametrización) |
| Mutantes manuales v8–v10 (`t_mut_v11_manuales.log`) | PENDIENTE | 10 de 10 detectados; 3 controles sin mutar con código 0 |
| Suite completa estricta dentro del ensayo (`ensayo_a2_p2.log`) | **APROBADO**, HEAD = A2 + aprobación sin commit (clon `C:/tm18`) | 1796 passed, 13 skipped, código 0 |
| Suite completa estricta (`suite_estricta_aprobada_p2.log`) | **APROBADO**, HEAD = P2 (clon `C:/tm14/c`) | 1796 passed, 13 skipped, código 0 |
| Controles de seguridad (`controles_seguridad_v11.txt`) | PENDIENTE y APROBADO (HEAD = P2) | PENDIENTE: todo rechazado con `[E02_PENDIENTE]`. APROBADO: configuración preparatoria activable sin ejecución oficial; rechazada por `require_e02_oficial` y `load_official_inputs` con `[E02_NO_OFICIAL]`; P2 inexistente `[E02_P2_EXISTE]`; configuración falsificada `[E02_OFICIAL_NO_VERIFICABLE]` |
| Integridad de `data/` (`datos_antes.txt`, `datos_despues.txt`) | — | 9 ficheros idénticos (bytes, tamaño, `mtime`, inodo) |

**Ejecuciones interrumpidas, no válidas y excluidas.** Dos tandas se interrumpieron y sus logs no cuentan como evidencia ni se incluyen en el ZIP:
- la primera tanda v11 de 5 procesos paralelos, que Claude Code detuvo por falta de memoria;
- la primera suite con HEAD = P2, cortada por un reinicio inesperado de Windows.

Tras el reinicio se comprobó que el repositorio real seguía igual (HEAD = P1, 26 cambios, `comprobar-a2` CORRECTO, datos idénticos), y se repitió solo esa suite, sin otras tareas en paralelo.

**Resultados de v10, sustituidos por los anteriores.** Están en la carpeta `evidencia_v10/` del ZIP:
- suite estricta: 1795 passed, 13 skipped;
- T-MUT: 67 detectados, 1 superviviente (`E02_REMOTO_HISTORIA`), 0 de infraestructura;
- mutantes manuales: 10 de 10;
- ensayo con suite: detenido con 13 failed (§8).

**Mutantes manuales:**

| Ronda | Mutantes |
|---|---|
| v8 | `C1_UNLINK`, `C2_SOLO_BYTES`, `REPLACE` |
| v9 | `V8_STAT`, `SIN_TIPO`, `EXISTS` |
| v10 | `SIGUE_ENLACES`, `SIN_TIPO`, `EXISTS_SALIDAS`, `EXISTS_VERIFY` |

**Pruebas omitidas.** Las 11 omitidas de la ronda de enlaces son las variantes con enlaces simbólicos reales: este Windows no tiene el privilegio para crearlos (`WinError 1314`). Las cubren variantes emuladas, que se ejecutan siempre. La auditoría independiente comprobó con enlaces reales el rechazo de `_owned()` (v9).

## 4. Qué se podrá ejecutar tras aprobar E02

Aprobar E02 (P2 publicado) **no** ejecuta nada ni autoriza MNQ. Habilita técnicamente una sola vía oficial, que exige además una **autorización explícita posterior** de la ejecución sobre MNQ:

```
python -c "from ohlcv_dataroad.ingest import tda17_salidas as s; s.official_publish_th35('<P2>')"
```

Esa vía:
1. Deriva `official_config(P2)`: E01 y E02 verificadas, P2 en el remoto autorizado y protocolo estricto.
2. Lee las 5 entradas del manifiesto y las verifica (`tda17_entradas`); nunca lee `data/raw` ni el holdout.
3. Ejecuta en el mismo proceso:
   - **TH37:** auditoría causal completa y su verificación contra el plan predeclarado (semilla 3717).
   - **TH35:** catálogo y panel P por la vía de producción; celdas GLOBAL, años y segmentos; R² de cada columna (R2_TODAS) con verificación cruzada; recuento de nulos con la convención A1-a; bootstrap por celda (IC 0,05/0,95) de R² y de Spearman con la validación F3-I/F3-I-b; clasificación RECONSTRUIBLE / NUEVA_REPRESENTACION / INESTABLE con celdas no publicables (B13-a) y subniveles con el R² global (B9, N-a).
4. Publica según el estado:
   - **`SIN_BLOQUEO`:** las 5 tablas del esquema mínimo, junto con `TDA17_procedencia_oficial.json`:
     - `TDA17_auditoria_causalidad.csv`;
     - `TDA17_estabilidad_celdas.csv`;
     - `TDA17_r2_reconstruccion.csv`;
     - `TDA17_correlacion_spearman.csv`;
     - `TDA17_clasificacion_final.csv`.
   - **`BLOQUEADO_HASTA_EXPLICAR`:** solo `TDA17_auditoria_causalidad.csv` y `TDA17_diagnostico_bloqueo.csv`, junto con el registro de procedencia. No hay clasificación, subniveles ni estabilidad por celda. El levantamiento exige una enmienda formal posterior auditada (ciclo M2).
5. Después: `verificar --referencia-git P2` (incluye `check_official_publication`), `reproduce_official_publication('<P2>')` y, con autorización explícita, el commit de resultados.

Las funciones de bajo nivel ya implementadas que no forman parte de esa vía no tienen esquema de salida, así que no pueden publicarse como resultado oficial (`SALIDAS_OTRAS_TABLAS`):
- PCA;
- agrupamiento;
- sensibilidades S1, S3, S5 y S6;
- diagnóstico del panel L;
- Pearson y Pearson winsorizado.

## 5. Qué sigue bloqueado tras aprobar E02

Cada punto lanza `PendingDefinitionError` (`PENDIENTES` en `tda17_analysis.py`) o un rechazo propio. Solo una enmienda posterior auditada puede levantarlo.

| Bloqueo | Qué no puede ejecutarse o publicarse | Motivo |
|---|---|---|
| `HUECO_ESPECTRAL` | Publicación de EXACTA por celda (`publish_exact_by_cell`, `spectral_gap_status`) | Decisión 11 aplazada |
| `R2_OTROS_INDICADORES` | Variante R² frente a los demás indicadores | B10 no autorizada |
| `R2_TRANSFERENCIA` | Transferencia (R²_in / R²_transf) | B8 no autorizada |
| `T3_V2` | T3 y criterio V2 | B11 no autorizada |
| `K_PA_MAX_T_V3` | Análisis paralelo, k_PA, banda max-T, V3 y uso de G2 en producción (`official_g2_controls`) | B12 no autorizada |
| `S2_ESCALAS` | Sensibilidad S2 (falta asignar las escalas S/M/L) | C1–C3 no autorizadas |
| `S7_S8_ALCANCE` | Sensibilidades S7 y S8 | D3 no autorizada |
| `REGISTRO_D7_RETRASO` | «Retraso» en barras de D7 en el registro de calentamiento (queda NaN) | No definido |
| `SALIDAS_OTRAS_TABLAS` | Las 13 tablas de `salidas.tablas` sin esquema: catálogo, ventanas, identidades, pérdidas, Pearson, transferencia, dimensión, agrupamiento, sensibilidades, G2, panel L, rvol, y su publicación por cualquier vía (`SALIDA_RUTA_UNICA`) | Las decisiones 17–19 solo fijan cinco tablas y el diagnóstico |
| `BLOQUEADO_HASTA_EXPLICAR` (si ocurre) | Clasificación, subniveles y estabilidad por celda de TH35 | A2-a: solo una enmienda formal auditada lo levanta |
| Valor numérico de las relaciones N | Queda `PENDIENTE_EXPEDIENTE` en el diagnóstico | Por diseño aprobado (decisión 16) |
| Holdout y `data/raw` | Ningún acceso | Protocolo |

## 6. Procedimiento A2 → P2 (preparado, NO ejecutado)

La herramienta `tda17_e02_cierre.py` (del ZIP de cierre; no se versiona) automatiza las comprobaciones. Por defecto solo **muestra** cada paso; actúa únicamente con `--ejecutar` o `--escribir`. Nunca hace push. Se ejecuta desde la raíz del repositorio.

| Paso | Comando | Qué hace y qué comprueba |
|---|---|---|
| 0 | `python tda17_e02_cierre.py comprobar-a2` | **Solo lectura.** Comprueba: HEAD = P1 en `main`; índice vacío; cambios del árbol = exactamente los 26 ficheros de A2 (§7); huellas de §2; registro pendiente; sin acta ni evidencias; E01 intacta; los tres verificadores CONFORME. |
| 1 | `... a2 --ejecutar` | `git add` de los 26 ficheros y `git commit -F mensaje_A2.txt`. Comprueba después: padre = P1; A2 cambia exactamente esos ficheros; E01 sin cambios entre P1 y A2; los 15 auditados en A2 con su huella; árbol limpio. |
| 2 | Auditoría independiente de A2 (opcional si se acepta que A2 = v11 por huellas) | El auditor puede comprobar las 15 huellas en A2 con `git show A2:<fichero>`. |
| 3 | `... aprobar --a2 <A2> --fecha <AAAA-MM-DD> --aprobada-por "<responsable>" --evidencias <dir> --escribir` | Rellena el registro (`estado: APROBADA`; `aprobacion`: fecha, responsable, `referencia_auditoria` = acta + A2, `sha256_auditados` leídos de A2), escribe el acta desde la plantilla y copia las evidencias. Comprueba `load_effective_protocol_e02`: activable, sin ejecución oficial. |
| 4 | Suite completa estricta en estado aprobado | `TDA17_INTEGRIDAD=estricta python -m pytest -p no:cacheprovider -q`, en verde. |
| 5 | `... p2 --a2 <A2> --ejecutar` | Comprueba: el registro solo cambia `estado` y `aprobacion`; los cambios son el registro, el acta y las evidencias; las huellas. Hace `git commit -F mensaje_P2.txt`. |
| 6 | `... comprobar-p2 --a2 <A2>` | Comprueba: padre de P2 = A2; árbol limpio; `verify_against_git_e02(P2)` vacío; los 15 auditados sin cambios entre A2 y P2; E02 activable; los tres verificadores CONFORME. |
| 7 | `... comprobar-remoto` | **Solo lectura.** `refs/heads/main` del remoto autorizado = P1, de modo que el push es de avance rápido. |
| 8 | `git push origin main` | **Manual**, sin `--force`. |
| 9 | `python -m ohlcv_dataroad.ingest.tda17_enmiendas_e02 verificar --referencia-git <P2>` | CONFORME con «ejecución oficial: SI». Consulta el remoto real y exige P2 en `refs/heads/main` con A2 y P1 en su historia. Registrar el SHA remoto. |
| 10 | Autorización separada de la ejecución oficial sobre MNQ | Ver §4. No forma parte del cierre de E02. |

**Si algo falla:**
- **Antes del push:** se corrige sin reescribir historia publicada. Antes de P2 basta `git reset --soft P1` (A2 no publicado) o un commit correctivo nuevo.
- **Después del push:** no se usa `--force`; se corrige con commits nuevos y, si afecta a ficheros auditados, con una nueva ronda de auditoría.

## 7. Commits preparados (no ejecutados)

**A2.** `feat(tda17): E02 amendment, audited version (commit A2, registry pending)`; mensaje completo en `mensaje_A2.txt`. Contiene exactamente 26 rutas, todas nuevas salvo `.gitattributes` y `.gitignore`:

- `configs/tda17_enmienda_02.yaml`;
- `src/ohlcv_dataroad/ingest/`: `tda17_analysis.py`, `tda17_causalidad.py`, `tda17_enmiendas_e02.py`, `tda17_entradas.py`, `tda17_g2.py`, `tda17_salidas.py`;
- `tests/`: `test_tda17_analysis.py`, `test_tda17_causalidad.py`, `test_tda17_e02_motor.py`, `test_tda17_enmienda_02.py`, `test_tda17_g2.py`, `test_tda17_procedencia.py`, `test_tda17_rango_diagnostico.py`, `test_tda17_salidas.py`;
- `reports/mnq/`: `TDA17_enmiendas_E02.json` (registro pendiente), `TDA17_E02_anexo_relaciones_N.md`, `TDA17_E02_cierre_formal.md`, `TDA17_E02_paquete_auditoria.md`, `TDA17_E02_propuesta_ambiguedades_y_salidas.md`, `TDA17_E02_propuesta_decisiones_normativas.md`, `TDA17_E02_registro_decisiones.md`, `TDA17_especificacion_migracion_M2.md`, `TDA17_inventario_decisiones_pendientes.md`;
- `.gitattributes` y `.gitignore` (modificados).

**P2.** `docs(tda17): approve E02 citing audited commit A2 (commit P2)`; mensaje completo en `mensaje_P2.txt`. Contiene:
- `reports/mnq/TDA17_enmiendas_E02.json` (aprobado);
- `reports/mnq/TDA17_E02_acta_aprobacion.md`;
- `reports/mnq/TDA17_E02_evidencia/` (v11):
  - README y `.gitattributes` `* -text`;
  - suites estrictas en estado PENDIENTE y en estado APROBADO con HEAD = P2;
  - focalizadas;
  - T-MUT con sus resultados y la comparación de cobertura v10 → v11, junto con el arnés;
  - los mutantes manuales v8–v10 y su ejecución v11;
  - los ensayos del procedimiento;
  - la integridad de `data/`;
  - los verificadores.

**Datos que solo puede aportar el responsable:**
- la fecha de aprobación, entre 2026-10-18 y el día de la aprobación;
- el texto de `aprobada_por`, que es una declaración y no una identidad autenticada;
- la autorización de cada commit y del push.

## 8. Ensayo del procedimiento en un clon temporal

Condiciones comunes a todos los ensayos:
- Clones temporales con la historia real hasta P1 y los 26 ficheros de A2 copiados del árbol.
- Copias independientes de los datos del manifiesto (`samefile` comprobado).
- `origin` = URL autorizada, con push URL `no-push://deshabilitado`.
- Ninguno hizo commit ni push en el repositorio real.
- La aprobación de los ensayos dice «ENSAYO del procedimiento (no es una aprobacion)».

**Versión v11 (definitiva):**

- **Con la suite estricta en estado APROBADO** (`C:/tm18`, `ensayo_a2_p2.log`): **ENSAYO CORRECTO**, con A2 = `b769ba4cd0dba2de3c365814a2f09417b1fc3281` y P2 = `ee95da316582372851eff8af2403c57857abe31e`, solo en el clon. Pasos:
  1. `comprobar-a2` CORRECTO;
  2. commit A2;
  3. `aprobar --escribir`;
  4. `verificar`: CONFORME, modo PRODUCCION, activable: SI, ejecución oficial: NO;
  5. **suite estricta en estado APROBADO: 1796 passed, 13 skipped, código 0**;
  6. commit P2;
  7. `comprobar-p2` CORRECTO;
  8. `verificar --referencia-git P2`: se detiene solo en `E02_REMOTO` (P2 no publicado), como se esperaba;
  9. push imposible (`no-push://deshabilitado`).
- **Sin suite** (`C:/tm14`, `ensayo_rapido_sin_suite_tm14.log`): ENSAYO CORRECTO, con A2 = `efaccf88cf3916f9d73c383e0ead1798b060faca` y P2 = `d76472e3a903bfb17daa1d6adb98825c91b80be3`. Sobre ese P2 se ejecutó la suite completa estricta con HEAD = P2 (`suite_estricta_aprobada_p2.log`): 1796 passed, 13 skipped, código 0.
- Los SHA de A2 y P2 difieren entre ensayos solo por la fecha de los commits; el árbol de A2 es el mismo.

**Historia: ensayo de v10 e incidencia corregida en v11.** Las evidencias están en `evidencia_v10/` del ZIP.
- **Sin suite** (`C:/tm13b`): ENSAYO CORRECTO.
- **Con la suite estricta en estado APROBADO** (`C:/tm13`): DETENIDO con 13 failed, 1782 passed y 13 skipped.

Las 13 pruebas daban por hecho que el registro del proyecto estaba PENDIENTE:
- 8 + 1 en `test_tda17_e02_motor.py`;
- 3 en `test_tda17_enmienda_02.py`, una de ellas porque `build_repo` copiaba el registro ya aprobado en la A2 sintética;
- 1 en `test_tda17_procedencia.py`.

El código se comportaba según el diseño. La aprobación es preparatoria y activable, y las rutas oficiales seguían rechazadas, aunque por el motivo siguiente de la cadena (`[E02_P2_EXISTE]`). La auditoría independiente confirmó el bloqueo y autorizó la ronda v11 (§1), limitada a esos 3 ficheros de pruebas y a `build_repo`.

## 9. Riesgos residuales documentados (aceptados sin ampliar la infraestructura)

1. **Recuperación y deshacer.** Entre el último `lstat` de `_owned` y el `unlink`, un proceso ajeno que no toma el cerrojo podría sustituir el fichero. Python no ofrece un borrado condicionado a la identidad. Ventana mínima.
2. **Verificador.**
   - Entre el primer `lstat` y la apertura, una sustitución momentánea por un enlace que se restaure antes del segundo `lstat` haría que la lectura siguiera ese enlace.
   - El verificador solo lee: no puede borrar ni sobrescribir.
   - `reproduce_official_publication` compara byte a byte con el recálculo.
3. **Sistemas de ficheros sin enlaces duros.** La publicación falla de forma segura, sin sobrescribir ni borrar nada ajeno.
4. **Enlaces simbólicos reales sin probar en este equipo.** No se han ejecutado aquí por falta de privilegio en Windows; las variantes emuladas sí. La auditoría independiente comprobó con enlaces reales el rechazo de `_owned()` (v9). Las variantes reales del verificador (v10) se ejecutarán en cualquier entorno POSIX o en un Windows con ese privilegio.
5. **Sin datos reales.** Las garantías que dependen de datos reales no se han verificado: el estado final de TH35 sobre MNQ y el coste de `reproduce_official_publication` solo se conocerán en la ejecución oficial autorizada.
6. **Superviviente de T-MUT redundante por construcción.** `E02_REMOTO_HISTORIA`: si P2 es alcanzable desde la rama remota, también lo son A2 y P1.
7. **T-MUT se ejecuta sobre el árbol de A2, con el registro PENDIENTE.** En estado APROBADO no es aplicable: mutar un fichero auditado invalida por diseño la aprobación real (`sha256_auditados`).
   - Las ramas APROBADAS de las pruebas sobre el proyecto se validan con las suites completas en estado aprobado (§3, §8).
   - Las pruebas sobre repositorios sintéticos, que detectan casi todos los mutantes, no dependen del estado del proyecto desde v11: `build_repo` construye siempre A2 PENDIENTE.
