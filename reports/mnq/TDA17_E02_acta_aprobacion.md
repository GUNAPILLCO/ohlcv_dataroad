# TDA-17 — Acta de aprobación de la enmienda E02

- **Fecha:** 2026-10-23
- **Enmienda:** E02 (`configs/tda17_enmienda_02.yaml`, SHA-256 `fd7478d557e9a798c52b2af0806e89a6945c90cf5d8f35ea328afb581ad0d317`).
- **Alcance:** decisiones normativas 1–10 y 12 (autorizadas el 2026-10-18) y 14–19 (2026-10-19), según `reports/mnq/TDA17_E02_registro_decisiones.md`. La decisión 11 (hueco espectral) sigue aplazada.
- **Requiere:** E01 publicada (A1 `386815bde75833ec64763f65b70495d0f535c44d`, P1 `24f9f1d2bb831a25e4dec6a651c0a5129f0bbca1`), sin modificar.
- **Protocolo base:** `TDA17_PROTOCOLO_v1`, commit `9cdea2f4657de1c54e3fccf86cf380b2eac8704e`, sin modificar.
- **Commit auditado (A2):** `2b6ad7abdf6609e6970f1fe6ee1e2569edc698b4`
- **Aprobada por:** Responsable del proyecto IRIS (declaración de aprobación; identidad no autenticada)

## Naturaleza de esta aprobación

Esta acta y el registro `reports/mnq/TDA17_enmiendas_E02.json` son una DECLARACIÓN de aprobación, no una firma criptográfica: no autentican la identidad de quien aprueba.

Lo que sí se verifica automáticamente es que lo aprobado es exactamente lo auditado:
- `load_effective_protocol_e02()` exige que el commit A2 exista, descienda de P1 y contenga cada fichero auditado con la huella registrada.
- `official_config_e02(P2)` vincula toda ejecución oficial a esta publicación P2 en la rama `refs/heads/main` del repositorio autorizado.

## Auditoría independiente (resultados declarados por el responsable del proyecto)

- **Implementación de E02 (v5):** las decisiones 1–19 aprobadas e implementadas, con auditoría favorable.
- **Publicación oficial:**
  - **v6–v7:** transaccional, con cerrojo y promoción sin sobrescritura;
  - **v8:** recuperación con propiedad por identidad y bytes;
  - **v9:** rechazo de enlaces simbólicos en la propiedad;
  - **v10:** verificador que exige ficheros regulares.
- **v11:**
  - pruebas válidas con el registro del proyecto PENDIENTE y APROBADO;
  - repositorios sintéticos con A2 siempre PENDIENTE;
  - solo cambian 3 ficheros de pruebas, sin cambios de producción.

Cada ronda fue auditada. La corrección v10 quedó aprobada técnicamente. Esta aprobación presupone la auditoría favorable de v11.
- **Evidencias reproducibles:** `reports/mnq/TDA17_E02_evidencia/` (ver su README) y `reports/mnq/TDA17_E02_cierre_formal.md`.

## Ficheros auditados (SHA-256 tras normalizar CRLF a LF; idénticos en A2)

| Fichero | SHA-256 |
|---|---|
| `configs/tda17_enmienda_02.yaml` | `fd7478d557e9a798c52b2af0806e89a6945c90cf5d8f35ea328afb581ad0d317` |
| `src/ohlcv_dataroad/ingest/tda17_enmiendas_e02.py` | `9484cb1ebc1bca7367a2131fa5881d059670319a230e7eb1aec81d3efa4f3c05` |
| `tests/test_tda17_enmienda_02.py` | `0a62fddbaa5b30a8a0ce0eba1200d15fa12dc5b0a738a4c25831545f7813e449` |
| `src/ohlcv_dataroad/ingest/tda17_analysis.py` | `f26bf380e0bbc2874c1939873a995513a693de6d4b848fdb8c1be88c26515fe0` |
| `src/ohlcv_dataroad/ingest/tda17_g2.py` | `586b97b79cde7730a5d27b71935d8263387c2ae1f4111ae22564d20d2902fb6b` |
| `src/ohlcv_dataroad/ingest/tda17_causalidad.py` | `1f5b6f3b1348762d83c11ca6b581d04ee945049531a8a878ca0cc05cb3910e89` |
| `src/ohlcv_dataroad/ingest/tda17_salidas.py` | `c8c3272caacaf333f714d0abaa78243ef27f2599c58e8b782a1877fb7e0468e0` |
| `src/ohlcv_dataroad/ingest/tda17_entradas.py` | `fae54edd2773c7244a16d1e309643a7acf33d32fe9be3528c89f9f03134e84fa` |
| `tests/test_tda17_analysis.py` | `acf1278fe3a9013b51e6510990ac65f70057f45e648ddbb212db3b46f0d79eab` |
| `tests/test_tda17_g2.py` | `33f820ff9c4a66afadd333954af28243e45577a1c41753675be2dab3ea67cbcd` |
| `tests/test_tda17_rango_diagnostico.py` | `61aefb2dcbd80fe53ec43675ef4b353066faf1a94a4fc4840cfbccf3eb8d9ddf` |
| `tests/test_tda17_causalidad.py` | `a4508c40a223c7bdc153065fffd001c9efef91ca32c0d582620b7a006065baa0` |
| `tests/test_tda17_e02_motor.py` | `f9ebbfbf1e01d549310812e806b065161a548fe9a4fda7b961dac0dcfeafe8c8` |
| `tests/test_tda17_procedencia.py` | `28d1717a44eac70727cacd35946354fc04db077cbd2a64b5c3aa4e9512e01eed` |
| `tests/test_tda17_salidas.py` | `20be16897d9ef92bfe9a05a306a3809353dbbcad1aa6e761250cc722d4fcb917` |

## Condiciones vigentes

- **MNQ:** esta aprobación NO autoriza la ejecución oficial sobre MNQ. La ejecución exige una autorización explícita posterior. Su única vía es `official_publish_th35(P2)`, seguida de `verificar --referencia-git P2` y `reproduce_official_publication(P2)`.
- **Holdout:** sigue cerrado.
- **Funcionalidad sin definición normativa:** sigue bloqueada (`PENDIENTES`), según la sección 5 de `TDA17_E02_cierre_formal.md`.
- **Estado `BLOQUEADO_HASTA_EXPLICAR`:** si se produce, solo se publican la auditoría causal y `TDA17_diagnostico_bloqueo.csv`. Su levantamiento exige una enmienda formal posterior auditada.
- **Riesgos residuales:** las carreras residuales documentadas (sección 9 de `TDA17_E02_cierre_formal.md`) se aceptan sin ampliar la infraestructura.
