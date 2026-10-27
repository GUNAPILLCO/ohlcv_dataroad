# TDA-17 E03 — Procedimiento propuesto: A4 → aprobación → P3 (versión R4 final)

Esta es una **propuesta**. Cada paso necesita una autorización explícita aparte. Nada de esto se ha ejecutado en el repositorio del proyecto. Solo se ha ensayado en el clon temporal `C:/tm19/r4/clon`, que tiene el push deshabilitado:
- A4' `1c973603e982e9df1f69686bbb9979a0cac64e9d`;
- P3' `84915b309b5ba8894bf6e1360699499c3b39d3c3`.

**Cambio respecto a la versión anterior de este procedimiento.** Antes, A4 contenía solo `test_e03_gobernanza.py`. Ahora contiene **4 ficheros**: la corrección de producción de `motor.py` y sus tres ficheros de pruebas. La aprobación definitiva debe citar esta A4, no A3.

## 0. Punto de partida (verificado el 2026-10-26)

**Repositorio del proyecto:**
- HEAD = A3 `fbd7ef64bb9bd038c0b49c36c65eab169eab9036` (único padre: P2).
- origin/main = P2 `ff224dc7e30f7631dc46eea17db5a60dd44b07cc`.
- Árbol limpio. Registro de E03 en PENDIENTE_AUDITORIA.

**Worktree `C:/tm19/r3/wt` (rama local `tda17-e03-aprobacion`):**
- Contiene la aprobación preparada que cita A3 y no se ha confirmado con un commit.
- **Está obsoleta**: cita A3, y A3 no incluye la corrección de `motor.py` ni la prueba de estado nueva.
- Propuesta: archivar su contenido fuera del repositorio y, con autorización, eliminar el worktree y la rama.

## 1. Commit auditado A4 (después de una auditoría favorable de R4)

1. **Comprobar en el proyecto:**
   - HEAD = A3 y origin/main = P2;
   - el índice está vacío y el árbol limpio;
   - las 15 huellas de E02, el protocolo y E01 están intactos (con los verificadores oficiales).

2. **Copiar los 4 ficheros desde `codigo_A4prima/` del paquete R4.** Su SHA-256 con fin de línea LF debe ser:

   | Fichero | SHA-256 (LF) |
   |---|---|
   | `src/ohlcv_dataroad/tda17_e03/motor.py` | `8c2f9fce9825700e700635c5dc11e7f893112b3c209047859248c78f9539e752` |
   | `tests/tda17_e03/test_e03_motor.py` | `89d2c92d611fad3db4b0fbc162c7cdfc69cb83a6df1730ffea52b62965c3d875` |
   | `tests/tda17_e03/test_e03_gobernanza.py` | `51338e941e21773f175060f4e6018af87f1653e547ddc5241b8e3c86dcaf8d17` |
   | `tests/tda17_e03/test_e03_cobertura.py` | `3743ad9e422d5110191fc2dd050be53b1017d431e024100f762abea96e4819fc` |

3. **Comprobar el alcance del cambio.** `git diff --stat` debe mostrar solo esos 4 ficheros, con +446 / −23 líneas en total, igual que `diffs/DIFF_R4_A3_a_A4prima.patch`. No cambian:
   - el YAML;
   - `E03_SHA256` (`85cbf62b…`);
   - `enmienda.py`, `__init__.py` y `e03_equivalencia.py`;
   - el registro, que sigue en PENDIENTE_AUDITORIA.

4. **Crear el commit A4 solo con esos 4 ficheros** y verificar después:
   - A4 tiene un único padre, A3;
   - A4 contiene exactamente esos 4 ficheros;
   - los 8 ficheros de `AUDITED_E03` tienen en A4 las huellas de `HUELLAS_E03_A3_vs_A4prima.json` (4 iguales a las de A3 y 4 nuevas);
   - el árbol está limpio.

   El árbol de A4 debe coincidir con el de A4'. Solo pueden cambiar el autor, la fecha y el mensaje, así que el SHA del commit será distinto.

5. **En A4 (estado PENDIENTE)**, ejecutar:
   - los verificadores oficiales: E03 PENDIENTE y ruta oficial bloqueada por `E03_PENDIENTE`;
   - las pruebas de E03.

   La suite completa ya se ensayó sobre el mismo árbol (A4'). Si la auditoría lo exige, se repite sobre A4 con la puerta de memoria (≥ 12 GB) y un límite externo.

## 2. Aprobación de E03 citando A4 (sin commit)

1. **Registrar la aprobación** con `registrar_aprobacion_e03.py` (ronda R3), poniendo A4 en lugar de A3. Solo cambian `estado` y `aprobacion` de la entrada E03:
   - `fecha`: la fecha de la aprobación;
   - `aprobada_por`: "Responsable del proyecto IRIS (declaración de aprobación; identidad no autenticada)";
   - `referencia_auditoria`: `{documento: reports/mnq/TDA17_E03_acta_aprobacion.md, commit_auditado: A4}`;
   - `sha256_auditados`: las huellas de los 8 ficheros auditados en A4.

   El ensayo equivalente es `scripts/ensayo_aprobacion.py`, que genera P3'.

2. **Rehacer el acta y las evidencias** citando A4 y el paquete R4. El conflicto de la prueba de estado y el bloqueo del envío a workers quedan resueltos por A4.

3. **Comprobar:**
   - los verificadores oficiales dan CONFORME, con E03 APROBADA y ruta oficial `BLOQUEADA: [E03_P3_EXISTE]…`;
   - `test_project_registry_is_pending_or_approved_and_the_official_e03_route_needs_a_valid_p3` pasa en estado APROBADA, y con él `check_project_state`:
     - la aprobación es válida;
     - las huellas coinciden con A4;
     - el registro de A4 está en PENDIENTE;
     - el acta existe;
     - sin P3, la ruta oficial se rechaza sin crear el directorio de ejecución.

## 3. P3 (commit de la aprobación, sin push)

1. Crear el commit P3 con:
   - el registro aprobado;
   - el acta;
   - `reports/mnq/TDA17_E03_evidencia/`.

   P3 tiene un único padre, A4.
2. Ejecutar la suite estricta completa con HEAD = P3:
   - una sola ejecución a la vez;
   - con la puerta de memoria (≥ 12 GB);
   - con límite externo (`timeout`).

   Solo valen los logs completos con `rc=`. El ensayo equivalente es la ejecución 4/4 sobre P3'.
3. Comprobar que, mientras P3 no esté en el remoto autorizado, la ruta oficial sigue bloqueada: `official_e03(P3)` rechaza con `[E03_REMOTO]`.

## 4. Push de A4 + P3 (solo con autorización explícita)

1. `git push origin main` (P2 → A3 → A4 → P3).
2. `python -m ohlcv_dataroad.tda17_e03.enmienda verificar --referencia-git P3` debe dar CONFORME y la ruta oficial ACTIVABLE.
3. La ejecución oficial sobre MNQ necesita otra autorización explícita. El holdout sigue cerrado.
