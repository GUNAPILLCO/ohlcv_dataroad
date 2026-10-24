# TDA-17 — Especificación técnica de la migración M2 (cargador escalonado E01 + E02)

**Estado:** `ESPECIFICACION_PARA_AUDITORIA`. M2 está aprobada como **arquitectura de referencia** para E02. Este documento **no** crea E02, no la activa, no fija ninguna decisión normativa (las decisiones de E02 están en `reports/mnq/TDA17_inventario_decisiones_pendientes.md`) y no modifica el protocolo, E01 ni el motor.
**Revisión F4–F6:**
- se exige que HEAD descienda de A2 (`E02_A2_HEAD`);
- se separan la aprobación preparatoria y la ejecución oficial vinculada a P2 (§6.1);
- las negativas exigen tipo y motivo exactos;
- las pruebas `_DISABLED` se denominan **omisiones instrumentadas**, no mutaciones.

**Evidencia:** prototipo desechable en clones (`mig_demo.py` v3, `m2_loader_template.py` v3); resultados en §12. El prototipo **no** es la implementación: sirve para demostrar que la especificación es realizable y que sus controles son detectables.
**DECISIÓN IRIS: NINGUNA.**

## 1. Objeto

Permitir aprobar y activar una enmienda E02 de forma **independiente** de E01, sin modificar ni re-aprobar E01. La verificación de producción debe comprobar a la vez:
- que E01 sigue exactamente como se auditó y publicó (A1 `386815b`, P1 `24f9f1d`);
- que E02 está aprobada sobre su propio commit auditado A2 y publicada en P2.

## 2. Invariantes

| Id | Invariante |
|---|---|
| I1 | Los **8 ficheros** de E01 (§3.1) son byte a byte los de P1: huella SHA-256 (CRLF → LF) igual a la fijada y `git diff --quiet P1 HEAD -- <8 ficheros>` |
| I2 | La entrada E01 del registro de E01 es la publicada (huella canónica fijada) y su aprobación cita **A1** |
| I3 | El cargador de E01 (`tda17_enmiendas.py`) se ejecuta **sin cambios** y verifica E01 contra A1, como hoy |
| I4 | E02 tiene YAML, registro, aprobación, acta, commit auditado A2 y publicación P2 **propios**. El registro de E01 no contiene E02 |
| I5 | La historia se conserva: P1 es antecesor de A2 y de HEAD, **A2 es antecesor de HEAD**, y A2 es antecesor estricto de P2 |
| I9 | Solo la **ejecución oficial** (§6.1) produce resultados para el informe. Exige la publicación P2, que P2 sea HEAD o antecesora de HEAD y que la verificación estricta contra P2 no dé errores. No hay ningún modo que omita controles |
| I6 | Los 9 ficheros del protocolo y el manifiesto no cambian (`tda17_protocol_freeze verificar --referencia-git 9cdea2f` CONFORME) |
| I7 | Mientras E02 esté pendiente, ninguna ruta de producción de la fase 2 es activable; la fase 1 sigue usando E01 |
| I8 | Todo fichero que implemente o verifique decisiones de E02 (motor estadístico de la fase 2 y sus pruebas) está en el conjunto auditado de E02 |

## 3. Ficheros

### 3.1 Protegidos de E01 (invariables; huellas CRLF → LF en P1)

| Fichero | SHA-256 |
|---|---|
| `configs/tda17_enmienda_01.yaml` | `62b91747d5102aef51cc2c51f014e20d3ba1e635e98cddee7c57ecebefa5d5cb` |
| `reports/mnq/TDA17_enmiendas.json` | `f286b9cf6c3ed2865ad006b7fdd5e1ee3df518feaaab7e96e9a363ab37877785` |
| `src/ohlcv_dataroad/ingest/tda17_enmiendas.py` | `6c5d96d0513b5e0962f476c7175529b061c2c4ea6c81340c86b1918a9575b569` |
| `src/ohlcv_dataroad/ingest/tda17_indicators.py` | `9d197a52576f18a19f6ebf51a6fb5f3bf966253515a1ec3439e3a44b4042dd1a` |
| `src/ohlcv_dataroad/ingest/tda17_primitives.py` | `11f90df582baf823a8c523eff0ad268ea720207fa8eb9f8bc8227d49f1a9fb32` |
| `tests/test_tda17_indicators.py` | `c3dc92f122beec2b4621e76c9012f178906814cf0c24130291f738f0ee83c417` |
| `tests/test_tda17_enmienda_01.py` | `25a86dbe393501c6262d628dffdde7fd1df3aa8e6e54f0834b1908fc6d611790` |
| `tests/tda17_synthetic.py` | `23e4f30a13069c8b717526d63bad029b50ad73493f7eca001a9b96f54d6ec64e` |

Otras constantes fijadas en el cargador de E02:
- huella canónica de la entrada E01: `3150ea3a5e6fda45…` (64 hexadecimales; se calcula con `tda17_enmiendas.canonical_sha256` sobre la entrada publicada en P1);
- A1 = `386815bde75833ec64763f65b70495d0f535c44d`;
- P1 = `24f9f1d2bb831a25e4dec6a651c0a5129f0bbca1`.

### 3.2 Protocolo base (invariables)
Los 9 ficheros de `PROTOCOL_FILES` y el manifiesto `reports/mnq/TDA17_manifiesto_protocolo.json` (SHA-256 `1b14b65a…`).

### 3.3 Nuevos de E02

| Fichero | Auditado (en `sha256_auditados`) | Seguido en la verificación contra P2 |
|---|---|---|
| `configs/tda17_enmienda_02.yaml` (contenido normativo, sin estado) | sí | sí |
| `src/ohlcv_dataroad/ingest/tda17_enmiendas_e02.py` (cargador de E02) | sí | sí |
| `tests/test_tda17_enmienda_02.py` (pruebas de E02 y de la migración, §10) | sí | sí |
| `src/ohlcv_dataroad/ingest/tda17_analysis.py` | sí | sí |
| `src/ohlcv_dataroad/ingest/tda17_g2.py` | sí | sí |
| `tests/test_tda17_analysis.py`, `tests/test_tda17_g2.py`, `tests/test_tda17_rango_diagnostico.py` | sí | sí |
| `reports/mnq/TDA17_enmiendas_E02.json` (registro de E02) | no (declaración de estado) | sí |
| `reports/mnq/TDA17_E02_acta_aprobacion.md` y evidencias | no | sí (en P2) |
| Cualquier módulo nuevo de la fase 2 (orquestación, tablas) | sí; la lista `AUDITED_E02` se cierra en A2 | sí |

**Huellas actuales de la fase 2**, solo como referencia. Cambiarán al implementar las decisiones de E02; las válidas son las registradas en A2.

| Fichero | SHA-256 hoy |
|---|---|
| `tda17_analysis.py` | `be07113c3514148c90f8b5d89cbed082fa66c9a90898082a0113621ae03b5ae6` |
| `tda17_g2.py` | `3b8242ebd5ce43e29023a2f3db8356b935aee2f31e7a9cbd5e862be8f403f9e2` |
| `tests/test_tda17_analysis.py` | `6ddabb0386f56022d0c9572c789b04b9d86f4fed715b68e6f781d097fe397087` |
| `tests/test_tda17_g2.py` | `8b08f469cc408859504e921aa58caf81734990da9da4b27cc4b036c9d70a1e2c` |
| `tests/test_tda17_rango_diagnostico.py` | `07feb8f9dc9160b83c6a2f61fadd3706f0215e8d08225952fb24321c30835ca6` |

**Sin ciclos.**
- El YAML de E02 cita el protocolo base y declara `requiere: E01`.
- El cargador de E02 fija la huella del YAML de E02 y las de E01.
- El registro de E02 cita las huellas de los ficheros auditados y A2.
- Ningún fichero auditado cita la huella del registro de E02 ni la de su propio commit.

## 4. YAML de E02 (estructura)

```yaml
enmienda:
  id: E02
  requiere: E01
  protocolo_base: {id: TDA17_PROTOCOLO_v1, commit: 9cdea2f…, sha256_yaml: 920cda43…, sha256_manifiesto: 1b14b65a…}
  motivo: {...}
  no_cambia: [...]
cambios:            # cada elemento: {op, ruta, valor}; op en {anadir, sustituir}
  - {op: anadir, ruta: <ruta de la lista blanca>, valor: <decision aprobada>}
```

## 5. Registro de E02 (`reports/mnq/TDA17_enmiendas_E02.json`)

**Campos del registro, exactamente estos:**
- `descripcion`: debe declarar que es una declaración de aprobación, no una firma.
- `protocolo_base`: igual a la referencia del protocolo base que usa E01.
- `requiere`: `{"E01": <huella canónica de la entrada E01>, "commit_auditado_E01": A1, "publicacion_E01": P1}`.
- `enmiendas`: una sola entrada.

**Campos de la entrada, exactamente estos:**
- `id`: `"E02"`.
- `fichero`: `configs/tda17_enmienda_02.yaml`.
- `sha256`: igual a la huella autorizada en el cargador.
- `alcance`.
- `estado`: `PENDIENTE_AUDITORIA` o `APROBADA`.
- `fecha_preparacion`: fecha ISO.
- `aprobacion`: `null` si está pendiente.

**Aprobación (si `APROBADA`), con exactamente los campos `fecha`, `aprobada_por`, `referencia_auditoria` y `sha256_auditados`:**
- `fecha`: ISO AAAA-MM-DD válida, con `fecha_preparacion ≤ fecha ≤ hoy`.
- `aprobada_por`: texto no vacío. Es una identidad **declarada**, no autenticada.
- `referencia_auditoria`: `{documento: texto no vacío (el acta de E02), commit_auditado: A2 en 40 hexadecimales}`.
- `sha256_auditados`: exactamente `AUDITED_E02`, cada huella igual a la del árbol de trabajo y a la del commit A2.

## 6. Cargador de E02: orden de verificación y controles

`load_effective_protocol_e02(root)` es la **única** vía de producción de la fase 2. Cada control rechaza con `ProtocolNotAuthorizedError` y un identificador `[ID]` en el mensaje. Ningún control tiene excepción ni vuelta al comportamiento anterior.

| Paso | Id | Comprobación |
|---|---|---|
| 1 | `E01_FICHEROS` | Los 8 ficheros de E01 existen y su huella LF es la fijada (§3.1) |
| 2 | `E01_BYTES` | `git diff --quiet P1 HEAD -- <8 ficheros>`: HEAD no confirma cambios en E01 |
| 3 | `E01_HISTORIA` | P1 es antecesor de HEAD |
| 4 | `E01_REGISTRO` | El registro de E01 contiene solo E01 |
| 5 | `E01_ENTRADA` | La huella canónica de la entrada E01 es la fijada |
| 6 | `E01_A1` | La aprobación de E01 cita A1 |
| 7 | (E01) | Se ejecuta `tda17_enmiendas.load_effective_protocol(root)` **sin cambios**: E01 verificada contra A1, protocolo base y lista blanca de E01 |
| 8 | `E02_REGISTRO`, `E02_REG_ESTRUCTURA`, `E02_BASE`, `E02_REQUIERE`, `E02_ENTRADA` | Registro de E02 presente, con campos exactos, protocolo base, `requiere` y una única entrada bien formada |
| 9 | `E02_HUELLA` | El YAML de E02 y la huella del registro coinciden con la autorizada en el cargador |
| 10 | `E02_DOC` | El YAML declara `id: E02`, el protocolo base y `requiere: E01` |
| 11 | `E02_ESTADO`, `E02_PENDIENTE`, `E02_PENDIENTE_CON_DATOS` | Estado conocido; en producción exige APROBADA; pendiente sin datos de aprobación |
| 12 | `E02_CAMPOS`, `E02_FECHA`, `E02_RESPONSABLE`, `E02_REFERENCIA`, `E02_COMMIT_FORMATO`, `E02_COBERTURA` | Estructura y contenido de la aprobación (§5) |
| 13 | `E02_ARBOL` | Cada fichero auditado tiene en el árbol la huella aprobada |
| 14 | `E02_COMMIT` | A2 existe y contiene cada fichero auditado con esa huella |
| 15 | `E02_ASCENDENCIA` | P1 es antecesor de A2 |
| 15b | `E02_A2_HEAD` | A2 es antecesor de HEAD (F4). Que el objeto A2 exista en el repositorio y que los ficheros tengan huellas válidas **no basta**: una rama que desciende de P1 pero no de A2 se rechaza |
| 16 | `E02_LISTA_BLANCA`, `E02_ANADIR` | Operaciones del YAML idénticas a `WHITELIST_E02`; `anadir` solo sobre una clave inexistente y `sustituir` solo sobre una existente con valor distinto (como E01) |
| 17 | `E02_RUTAS` | Barrera independiente: la efectiva difiere de la de E01 exactamente en las rutas autorizadas |
| — | `GIT` | Cualquier fallo de acceso a Git (no instalado, tiempo agotado, error de lectura, carpeta que no es la raíz del repositorio) es un rechazo |

**Modo borrador.** `effective_document_e02(root, borrador=True)` permite inspeccionar un E02 pendiente. Aplica los pasos 1–11 y 16–17, devuelve `modo: BORRADOR`, `activable: false` y `ejecucion_oficial: false`, y **nunca** se usa en producción.

### 6.1 Aprobación preparatoria frente a ejecución oficial (F4)

| | Aprobación preparatoria | Ejecución oficial |
|---|---|---|
| Uso | Pruebas antes de P2, en A2 o en repositorios temporales con la aprobación aún sin publicar | Única vía para producir resultados del informe (tablas, clasificación, G2 sobre réplicas, sensibilidades) |
| Función | `load_effective_protocol_e02(root)` | `official_config_e02(root, publicacion=P2)`; en la fase 2, `tda17_analysis.official_config(P2)` |
| Controles | Pasos 1–17 de §6, incluido `E02_A2_HEAD` | Los mismos y además los de la tabla siguiente |
| Traza | `activable: true`, `ejecucion_oficial: false`, `publicacion_E02: null` | `activable: true`, `ejecucion_oficial: true`, `publicacion_E02: P2` |

Controles adicionales de la ejecución oficial, en este orden y después del cargador:

| Id | Comprobación |
|---|---|
| `E02_P2_FORMATO` | P2 se da explícitamente como SHA-1 completo (40 hexadecimales). Sin P2 no hay ejecución oficial |
| `E02_P2_EXISTE` | P2 existe y es un commit |
| `E02_P2_A2` | A2 es antecesor **estricto** de P2 (P2 ≠ A2) |
| `E02_P2_HEAD` | P2 es HEAD o antecesora de HEAD. Una rama que desciende de A2 pero no de P2 se rechaza |
| `E02_P2_ARBOL` | `verify_against_git_e02(P2)` sin errores: los 8 ficheros de E01 y los seguidos de E02 son idénticos a P2. Se permiten commits posteriores ajenos a ellos |
| `E02_P2_APROBADA` | El registro de E02 **publicado en P2** está APROBADA y cita A2 |

**Sin desvíos de producción.** Ninguna función de producción acepta un parámetro, una variable de entorno o un modo que omita controles. El gancho `_DISABLED` del prototipo está prohibido en la implementación real (§11). La aprobación preparatoria no produce resultados oficiales: las funciones que escriben salidas del informe exigen `ejecucion_oficial: true` (`require_e02_oficial`).

## 7. Lista blanca de E02

- `WHITELIST_E02` está fijada **en el código** del cargador de E02, que es un fichero auditado, como lista de pares `(op, ruta)`. Sus valores los determinan las decisiones aprobadas del inventario; hoy no hay ninguna.
- **Rutas que la lista blanca nunca puede contener.** Una prueba estática lo comprueba (§10, T-WL). Son rutas que cambiarían parámetros, umbrales, hipótesis o D1–D24:
  - `catalogo.*.configuraciones` y `catalogo.*.formula`;
  - `analisis.umbrales`, `analisis.clasificacion.umbral_r2`, `umbral_parcial`, `anios_min` y `segmentos_min`;
  - `analisis.r2.tau_relativa`, `tolerancia_cruzada` y `hueco_espectral_minimo`;
  - `identidades.lineales`, `identidades.rango` e `identidades.verificacion_numerica.tau_relativa` y `nulos_esperados`;
  - `analisis.exactitud.residuo_max` y `casi_exacta_max`;
  - `analisis.bootstrap.replicas`, `longitud_bloque_jornadas` y `semilla`;
  - `g2.*.semillas`, `semilla`, `replicas`, `df` y tolerancias;
  - `sensibilidades.S6.fechas`, `decisiones`, `protocolo` y `entradas`.

  Una E02 **añade** aclaraciones (por ejemplo, `analisis.bootstrap.intervalo` o `sensibilidades.escalas_S2`); no sustituye reglas congeladas.
- Las operaciones del YAML deben ser exactamente las de la lista blanca, ni más ni menos.
- La configuración efectiva final difiere de la de E01 exactamente en esas rutas (`E02_RUTAS`).

## 8. Trazabilidad conjunta

`trazabilidad_protocolo` de la configuración efectiva de E02 contiene:
- la de E01 tal como la produce su cargador: base, E01 con su aprobación y A1, huella del registro de E01;
- y, añadidos:
  - la entrada `{id: E02, fichero, sha256, estado, aprobacion}`, con A2;
  - `sha256_registro_E02`;
  - `sha256_efectiva` (la de la configuración E01 + E02);
  - `modo` (`PRODUCCION` o `BORRADOR`);
  - `activable`, que solo es true en producción con E02 APROBADA.

## 9. Integración en las rutas de producción

- **Fase 2.** Toda función que produzca resultados para el informe (orquestación por celdas, tablas, G2 sobre réplicas, sensibilidades) obtiene la configuración **solo** mediante `tda17_analysis.official_config(P2)`, que llama a `official_config_e02(root, P2)`.
  - Las funciones de bajo nivel siguen recibiendo `cfg` explícito para las pruebas.
  - Las de producción deben llamar a `require_e02(cfg)`, que rechaza una configuración sin la entrada E02 APROBADA en su trazabilidad o con `activable` false, igual que `require_e01` en la fase 1.
  - Las que escriben salidas oficiales del informe llaman además a `require_e02_oficial(cfg)`, que exige `ejecucion_oficial: true` y `publicacion_E02` (§6.1).
- **Fase 1.** `compute_catalog` y `build_panel_p` siguen usando el cargador de E01 sin cambios: E02 no puede alterar el cálculo de indicadores (I1).
- **Herramienta de verificación** de E02:

  ```
  python -m ohlcv_dataroad.ingest.tda17_enmiendas_e02 verificar [--referencia-git P2]
  ```

  Ejecuta:
  1. la verificación estricta del protocolo base y de E01, que sigue CONFORME;
  2. el cargador de E02 en modo borrador, o en producción si está aprobada;
  3. con `--referencia-git`, la comparación de los 8 ficheros de E01 y de los seguidos de E02 con ese commit.

  Es **obligatoria** antes de cualquier ejecución sobre MNQ.

## 10. Procedimiento de aprobación A2 → P2

1. **Precondiciones.**
   - HEAD desciende de P1.
   - Los 8 ficheros de E01, el protocolo y el manifiesto no cambian.
   - Suite completa estricta en verde.
   - Decisiones de E02 aprobadas por el responsable y redactadas en su YAML.
2. **Commit A2.** YAML de E02, cargador de E02, pruebas de E02, módulos y pruebas de la fase 2, y registro de E02 en `PENDIENTE_AUDITORIA`, con `aprobacion: null`.
3. **Comprobaciones en A2.** Huellas de A2 iguales a las auditadas; `git diff --quiet P1 A2 -- <8 ficheros de E01>`.
4. **Aprobación.** El registro de E02 pasa a `APROBADA`, con aprobación que cita A2 y el acta; el registro de E01 **no se toca**.
5. **Pruebas en estado aprobado.** El cargador de E02 activa y las negativas siguen pasando.
6. **Commit P2.** Registro de E02 aprobado, acta y evidencias.
7. **Verificación final.**
   - `verificar --referencia-git P2` de E02 y de E01;
   - `official_config_e02(root, P2)` con HEAD = P2: `ejecucion_oficial: true`;
   - suite completa estricta;
   - A2 antecesor de P2.
8. **Publicación.** Push normal, sin `--force`, tras comprobar que el remoto no ha avanzado.

## 11. Pruebas que deberá superar la implementación real

**Requisitos para todas:**
- Todas sintéticas, sin MNQ.
- Las que necesitan aprobación usan **repositorios Git temporales reales**, con A2 y P2 y fechas fijas para SHA reproducibles.
- **Negativas.** Una negativa solo es válida si lanza **exactamente** `ProtocolNotAuthorizedError` y su mensaje **completo** coincide con la causa esperada (identificador y detalle, comparación anclada). Un rechazo con el identificador correcto pero otro motivo, otro tipo de excepción o texto adicional **no** se acepta. El comparador se autoprueba con esos casos (T-CMP).
- **Positivas.** Se comparan con el valor esperado exacto. El conjunto de pruebas solo es conforme si las obligatorias terminan con código 0 y el número exacto de pruebas superadas.
- **Mutaciones reales.** La implementación real **no** contiene ganchos de desactivación. La omisión de controles se comprueba con un arnés de **mutación de código fuente** sobre clones Git con A2 y P2 reales, con aprobaciones recalculadas para cada mutante (lección de E01: un arnés sobre copias sin Git no puede verificar aprobaciones).
- **Omisiones instrumentadas.** Las del prototipo (`_DISABLED`) son evidencia de diseño, **no** mutaciones, y no sustituyen a T-MUT.

| Grupo | Pruebas |
|---|---|
| **T-E01: preservación de E01** | Las 303 pruebas auditadas de E01 (`test_tda17_enmienda_01.py` y `test_tda17_indicators.py`) pasan **sin modificación**. `git diff --quiet P1 HEAD -- <8 ficheros>`. Huellas LF = §3.1. El cargador de E01 verifica E01 contra A1. `tda17_enmiendas verificar --referencia-git P2` CONFORME |
| **T-POS: positivas** | Carga en producción con E02 aprobada: traza conjunta (E01 contra A1, E02 contra A2), `modo` PRODUCCION, `activable` true, rutas cambiadas = lista blanca. Borrador con E02 pendiente: `activable` false |
| **T-NEG: negativas, una por control** | `E01_FICHEROS` (fichero de E01 modificado en el árbol); `E01_BYTES` (cambio confirmado y árbol restaurado); `E01_HISTORIA` (HEAD con el mismo árbol y sin historia); `E01_REGISTRO` (E02 en el registro de E01); `E01_ENTRADA` (entrada E01 alterada en un campo que su cargador no comprueba); `E01_A1` (aprobación de E01 reescrita sobre otro commit coherente); `E02_REGISTRO`, `E02_REG_ESTRUCTURA`, `E02_BASE`, `E02_REQUIERE`, `E02_ENTRADA`; `E02_HUELLA`; `E02_DOC`; `E02_ESTADO`; `E02_PENDIENTE`; `E02_PENDIENTE_CON_DATOS`; `E02_CAMPOS` (falta un campo o sobra uno); `E02_FECHA` (futura, anterior a la preparación, no ISO, inexistente); `E02_RESPONSABLE`; `E02_REFERENCIA`; `E02_COMMIT_FORMATO` (abreviado, no hexadecimal); `E02_COBERTURA` (falta un fichero de la fase 2, sobra uno); `E02_ARBOL` (motor de la fase 2 modificado); `E02_COMMIT` (inexistente; existente sin los ficheros); `E02_ASCENDENCIA` (mismo árbol sin historia); `E02_LISTA_BLANCA`; `E02_ANADIR`; `E02_RUTAS` (barrera independiente); `GIT` (no ejecutable, tiempo agotado, error de lectura, sin repositorio, repositorio padre) |
| **T-MUT: mutaciones de código fuente** | Por cada control, la mutación que elimina o invierte su condición en el código fuente, ejecutada en un clon Git, hace fallar al menos una prueba de T-NEG. Si solo la detecta otra barrera, se documenta como redundancia, sin rebajar la prueba |
| **T-CMP: comparador** | Acepta tipo y motivo exactos. Rechaza el identificador correcto con otro motivo, otro tipo de excepción, texto adicional y la ausencia de rechazo |
| **T-OFI: ejecución oficial (F4)** | Positivas: HEAD = P2 y HEAD descendiente de P2 con cambios ajenos a E01/E02 (`ejecucion_oficial: true`). Preparatoria en A2: `activable: true`, `ejecucion_oficial: false`. Negativas: `E02_A2_HEAD` (rama desde P1 con el árbol de P2), `E02_P2_FORMATO`, `E02_P2_EXISTE`, `E02_P2_A2`, `E02_P2_HEAD` (HEAD = A2; rama desde A2 con el árbol de P2), `E02_P2_ARBOL` y `E02_P2_APROBADA` |
| **T-INT: integración** | Toda ruta de producción de la fase 2 pasa por `production_config()`. Una prueba estática comprueba que ningún módulo de la fase 2 llama a `load_effective_protocol(` de E01 para producir resultados. Con E02 pendiente, `production_config()` rechaza con `[E02_PENDIENTE]`. `require_e02` rechaza configuraciones sin E02 |
| **T-WL: lista blanca** | La lista blanca no contiene ninguna ruta prohibida (§7); las operaciones del YAML son exactamente las de la lista blanca |
| **T-GIT-P2: verificación final** | `verify_against_git_e02(P2)` sin errores en P2. Cualquier cambio posterior en un fichero seguido (de E01 o de E02) se informa. Un commit inexistente nunca da CONFORME |
| **T-REP: reproducibilidad** | La demostración con fechas e identidad Git fijas reproduce los mismos A2 y P2 |
| **T-FASE2** | Las pruebas de la fase 2 (analysis, g2 y diagnóstico de rango) pasan con el cargador de E02 |

## 12. Evidencia del prototipo (clones desechables)

**Material de la demostración:**
- `m2_loader_template.py` v3 (SHA-256 `c70cb36d…`), ejecutado por `mig_demo.py` v3 (`027d8da5…`) en clones desechables.
- Para la regresión F4 se usa la plantilla v2 (`07e782d5…`), la que no tiene `E02_A2_HEAD`.
- Commits deterministas: A2 `c1e9b74`, P2 `e196e4f`, Pbad `ad2e62d`, commit posterior `c00f746`; regresión: A2 `00ee5df`, P2 `4fa71ed`, rama paralela `34e598f`.
- Registros íntegros: `mig_demo_v3_run1.log` (`dd4cb207…`) y `mig_demo_v3_run2.log` (`bf1ee8a8…`), 3561 líneas cada uno.
- Son idénticos salvo la etiqueta final de ejecución, y los 10 SHA de commits coinciden entre ejecuciones.
- Código de salida de ambas ejecuciones: **0**.
- Inyección de fallos (F5): `mig_demo_inyeccion.py` (`9f3e75c5…`), registro `mig_demo_v3_inyeccion.log` (`8492d871…`).

| Comprobación | Resultado |
|---|---|
| T-CMP: autopruebas del comparador | 7/7: acepta tipo y motivo exactos. Rechaza: identificador correcto con otro motivo; identificador correcto con otro texto; otro tipo de excepción; texto adicional; ausencia de rechazo; y el rechazo **real** de «fecha futura» comparado con el motivo de «fecha no ISO» |
| T-E01: 303 pruebas auditadas de E01 (control en P1 y clon M2) | **obligatorio**: código de salida 0 y exactamente 303 passed, en ambos |
| T-E01: `git diff --quiet P1 P2 -- <8 ficheros>`, huellas LF, cargador de E01 contra A1, `verify_against_git(P2)` de E01 | conforme |
| T-FASE2: pruebas de la fase 2 en M2 | **obligatorio**: código de salida 0 y exactamente 105 passed |
| T-POS y T-OFI: 17 positivas con valor exacto | Preparatoria en P2 y en A2 (aprobación sin publicar): `activable` true, `ejecucion_oficial` false. Oficial con HEAD = P2 y con HEAD descendiente de P2: `ejecucion_oficial` true, `publicacion_E02` = P2. Ruta oficial de la fase 2 (`tda17_analysis` y `tda17_g2`). Borrador pendiente: `BORRADOR`, no activable ni oficial |
| T-NEG y T-OFI: negativas con tipo y motivo exactos | 54/54 en total (M0 3, M1 11, M2 40). En M2: los 30 identificadores de §6 (`E02_FECHA` ×3, `E02_COMMIT` ×2) y `E02_A2_HEAD`; los 6 `E02_P2_*`; `E02_P2_HEAD` también con HEAD = A2; `E02_RUTAS`; `GIT`; ruta oficial con E02 pendiente |
| Regresión F4 | Plantilla v2: con HEAD en una rama que desciende de P1 pero no de A2 (objeto A2 presente, ficheros y huellas válidos), el cargador **acepta** (`activable` true): la vulnerabilidad se reproduce. Plantilla v3: rechaza con `[E02_A2_HEAD] HEAD no desciende del commit auditado A2 de E02` |
| Omisiones **instrumentadas** (`_DISABLED`, no son mutaciones de código fuente) | 37/37 detectadas. En 25 casos el escenario pasa a aceptarse. En 12 lo detiene otra barrera: `E01_FICHEROS` y `E01_REGISTRO` (cargador de E01); `E01_HISTORIA` y `E02_ASCENDENCIA` (`E02_A2_HEAD`); `E02_HUELLA` y `E02_COBERTURA` (`E02_ARBOL`); `E02_ESTADO` (`E02_PENDIENTE`); `E02_COMMIT` inexistente (`E02_ASCENDENCIA`); `E02_LISTA_BLANCA` (`E02_RUTAS`); `E02_P2_FORMATO` (`E02_P2_EXISTE`); `E02_P2_EXISTE` (`E02_P2_A2`); `E02_P2_A2` (`E02_P2_ARBOL`) |
| F5: código de salida dependiente de las positivas | Con todo lo demás conforme: una prueba fallida inyectada (pytest con código 1) da `final_code() = 1`; un recuento distinto del exigido con pytest en código 0 (304 frente a 303) da `final_code() = 1`; el control da `final_code() = 0` |
| M1 (comparativa) | Fallo **esperado y declarado** de sus pruebas auditadas de E01: código de salida 1, 62 failed, 98 passed y 143 errors. Si no fallaran, el guion lo marcaría INESPERADO y terminaría con código 1 |
| T-REP | Dos ejecuciones idénticas, incluidos los SHA de A2 y P2 |
| T-MUT (mutaciones de código fuente) | **No demostrado por el prototipo**: queda a cargo de la implementación real (§11) |

**Requisito de entorno observado (Windows).**
- `git show <commit>:<ruta>` falla con «Filename too long» cuando la ruta absoluta del repositorio más la del fichero supera MAX_PATH. Se reprodujo en un clon de inyección con un nombre de directorio largo.
- El cargador de E01 lo trata como commit no verificable y **rechaza**: falla de forma segura, sin aceptar nunca nada indebido. El de E02 hace lo mismo.
- La ejecución oficial debe hacerse desde una ruta corta (como la del repositorio real) o con `core.longpaths` activado en la configuración de Git del sistema.
- Esto no exige ningún cambio en el código de E01. La implementación real debe probar que el fallo es seguro en este caso.

**Fuera del prototipo, a cargo de la implementación real:**
- `sustituir` en la lista blanca de E02;
- `require_e02` y `require_e02_oficial`;
- la prueba estática de rutas de producción;
- la prueba de rutas prohibidas de la lista blanca (T-WL);
- los demás modos de fallo de Git (tiempo agotado, error de lectura, sin repositorio, repositorio padre, rutas largas);
- la herramienta `verificar` de E02;
- el arnés de **mutación de código fuente** sobre clones Git (T-MUT).

El gancho `_DISABLED` del prototipo es **solo de demostración** y está prohibido en la implementación real (§11).
