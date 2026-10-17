# TDA-17 — Acta de aprobación de la enmienda E01

- Fecha: 2026-10-17
- Enmienda: E01 (`configs/tda17_enmienda_01.yaml`, SHA-256 `62b91747d5102aef51cc2c51f014e20d3ba1e635e98cddee7c57ecebefa5d5cb`)
- Alcance: E01-A (A8 ADX con DX no definido, opción B: activación tras n + k(1/n) DX finitas) y E01-B
  (ventanas separadas de nivel y de par para las marcas, marca de sesión acumulada de VWAP y combinación para S1/S2).
- Protocolo base: `TDA17_PROTOCOLO_v1`, commit `9cdea2f4657de1c54e3fccf86cf380b2eac8704e`, sin modificar.
- Commit auditado (A): `386815bde75833ec64763f65b70495d0f535c44d`
- Aprobada por: Responsable del proyecto IRIS (declaración de aprobación; identidad no autenticada)

## Naturaleza de esta aprobación

Esta acta y el registro `reports/mnq/TDA17_enmiendas.json` son una DECLARACIÓN de aprobación, no una firma
criptográfica: no autentican la identidad de quien aprueba. Lo que sí se verifica automáticamente es que lo
aprobado es exactamente lo auditado: `load_effective_protocol()` exige que el commit A exista en el repositorio y
contenga cada fichero auditado con la huella registrada, y `verificar --referencia-git P` comprueba el commit
publicado.

## Auditoría independiente (resultados declarados por el responsable del proyecto)

- E01-A y E01-B aprobadas metodológicamente; ADX opción B y marcas C4/VWAP sin nuevos problemas.
- F1 (verificación Git de motor, primitivas y pruebas) y F3 (evidencia de mutaciones) aprobadas.
- F2 (autorización): la vulnerabilidad de commit auditado inexistente quedó corregida; hashes 3/3, mutaciones
  M55-M57 3/3 detectadas, 5/5 pruebas Git independientes, borrador no activable confirmado.
- Parche de las 4 pruebas de estado del registro aprobado antes de A (`tests/test_tda17_enmienda_01.py`,
  SHA-256 `25a86dbe393501c6262d628dffdde7fd1df3aa8e6e54f0834b1908fc6d611790`).

## Ficheros auditados (SHA-256 tras normalizar CRLF a LF; idénticos en A)

| Fichero | SHA-256 |
|---|---|
| `configs/tda17_enmienda_01.yaml` | `62b91747d5102aef51cc2c51f014e20d3ba1e635e98cddee7c57ecebefa5d5cb` |
| `src/ohlcv_dataroad/ingest/tda17_enmiendas.py` | `6c5d96d0513b5e0962f476c7175529b061c2c4ea6c81340c86b1918a9575b569` |
| `src/ohlcv_dataroad/ingest/tda17_indicators.py` | `9d197a52576f18a19f6ebf51a6fb5f3bf966253515a1ec3439e3a44b4042dd1a` |
| `src/ohlcv_dataroad/ingest/tda17_primitives.py` | `11f90df582baf823a8c523eff0ad268ea720207fa8eb9f8bc8227d49f1a9fb32` |
| `tests/test_tda17_indicators.py` | `c3dc92f122beec2b4621e76c9012f178906814cf0c24130291f738f0ee83c417` |
| `tests/test_tda17_enmienda_01.py` | `25a86dbe393501c6262d628dffdde7fd1df3aa8e6e54f0834b1908fc6d611790` |
| `tests/tda17_synthetic.py` | `23e4f30a13069c8b717526d63bad029b50ad73493f7eca001a9b96f54d6ec64e` |

## Condiciones vigentes

- No se autoriza todavía ejecutar el análisis sobre MNQ ni acceder al holdout.
- S2 sigue bloqueado hasta que una enmienda aprobada asigne las configuraciones a las escalas S/M/L.
- La regla de unión de marcas en cálculos multiserie deberá acompañarse de controles de población efectiva y
  evaluabilidad, sin modificar umbrales congelados.
- Evidencias reproducibles: `reports/mnq/TDA17_E01_evidencia/` (ver su README).
