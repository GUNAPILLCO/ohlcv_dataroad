# TDA-15 — No linealidad residual (TH32, TH33)

**Etapa:** `docs/methodology/Tsay_OHLCV_analysis_roadmap.md` § TDA-15 (Nivel 5, *condicional*).
**Hipótesis:** `TH32`, `TH33` (`docs/methodology/Tsay_empirical_hypotheses_backlog.md`).
**Referencia:** Tsay, *Analysis of Financial Time Series*, capítulo 4 (`docs/tsay/Tsay_Cap4_analisis_futuros_ML.md`).
**Base:** commit `c4efa271c785ef4c10d1884cfecd5937125c6e1d` (cierre de TDA-14).
**Registro de cierre:** `configs/tda15_cierre.yaml`. Las evidencias citadas abajo (E01–E20 y X01–X04) están en ese archivo con su fuente y una cita literal; `tests/test_tda15_cierre.py` comprueba que cada cita aparece en su fuente.

**Estado: TDA-15: CERRADA METODOLÓGICAMENTE — STOP-15a ACTIVADO — ANÁLISIS NO EJECUTADO.**

> Este cierre **no es un resultado empírico negativo** sobre la existencia de no linealidad. Es una decisión metodológica: no se ejecutan pruebas cuyas condiciones de entrada no se cumplen. No se leyeron datos del MNQ, no se abrió el holdout y no se calculó ningún estadístico.

---

## 1. Objetivo original de TDA-15

El roadmap (§ TDA-15) pregunta tres cosas: si queda estructura que no explica el modelo «lineal más volatilidad», si esa estructura es estable entre subperíodos y si depende de la frecuencia de muestreo.

El método mínimo previsto era:
1. construir un residuo estandarizado $\hat\epsilon_t = (r_t - \hat\mu)/(\hat\sigma_{t-1}\cdot s(m_t))$ con componentes causales y auditados;
2. aplicarle BDS y el F-test de Tsay, con un número de variantes acotado y declarado de antemano;
3. calibrarlo por permutación y por simulación (G2);
4. repetirlo por subperíodo y a 1, 5 y 10 minutos.

Hipótesis previstas:
- **TH32:** ¿queda estructura no lineal tras filtrar media, volatilidad y estacionalidad?
- **TH33:** ¿depende esa evidencia de la frecuencia?

## 2. Alcance de la auditoría

**Qué fue la auditoría:**
- una revisión de viabilidad **solo documental**, sobre la evidencia acumulada en TDA-00 a TDA-14;
- no se ejecutó ningún análisis de TDA-15;
- no se leyeron datos reales ni el holdout;
- no se modificó ningún artefacto anterior.

**Qué se consultó:**
- documentación, informes oficiales y código;
- la implementación instalada de `statsmodels`, solo leyendo su código fuente.

Lo segundo no forma parte de la justificación de este cierre.

## 3. Dependencias y documentos revisados

| Documento | Uso |
|---|---|
| `docs/methodology/Tsay_OHLCV_analysis_roadmap.md` (§2 G0–G6, § TDA-15, §8) | Criterios de entrada, `STOP-15a`, gobernanza |
| `docs/methodology/Tsay_empirical_hypotheses_backlog.md` (TH32, TH33) | Variable requerida, dependencias |
| `docs/methodology/Tsay_sintesis_transversal_OHLCV.md` | Riesgos de los tests con $n$ grande y volatilidad agrupada |
| `docs/tsay/Tsay_Cap4_analisis_futuros_ML.md` (§9–§12) | BDS, F-test de Tsay, *guardrail* y multiplicidad |
| `reports/mnq/TDA04_variables_analisis.md` | `r_1m` causal y reglas de no cruce |
| `reports/mnq/TDA06_perfil_intradia_calendario.md` | `s(m)` y `r_tilde` |
| `reports/mnq/TDA08_dependencia_lineal_media.md` | Dependencia lineal en media, STOP-8a y STOP-8b, TH18 |
| `reports/mnq/TDA09_volatility_clustering.md` | Clustering de volatilidad y su materialidad |
| `reports/mnq/TDA10_escala_vs_forma.md` | Estimadores causales de volatilidad |
| `reports/mnq/TDA11_modelo_parametrico_volatilidad.md` | GARCH(1,1), benchmarks y dependencia residual |
| `configs/tda14_cierre.yaml`, `reports/mnq/TDA14_estabilidad_temporal.md` | STOP-14 y estado por familia |

## 4. Matriz de las cuatro condiciones de entrada

| Condición | Estado | Evidencia principal |
|---|---|---|
| A — Residuo limpio | **NO CUMPLIDA** | E01–E13 (sección 5) |
| B — Estabilidad temporal | **CUMPLIDA FORMALMENTE, CON LIMITACIONES** | E14–E20 |
| C — Pregunta científica | **PENDIENTE — NO DESARROLLADA** | No se aprobó ninguna hipótesis sobre la forma de la dependencia residual |
| D — Multiplicidad y calibración | **PENDIENTE — NO DESARROLLADA** | No se congelaron variantes, parámetros, corrección por multiplicidad ni controles G2 de TDA-15 |

**Condición B en detalle:**
- `STOP-14 = NO_ACTIVADO` (E14) satisface formalmente el criterio 2 del roadmap, pero **no demuestra estabilidad general**.
- RELOJ es INESTABLE (E15), con un recorrido de `log_amplitud_perfil` de 0,32 frente a una materialidad de 0,22 (E16).
- ESCALA (E17), DEPENDENCIA_MAGNITUD (E18) y DISCRECION (E19) son NO_EVALUABLE.
- La volatilidad varía mucho de un año a otro, sin una ruptura estructural demostrada (E20).

**Condiciones C y D:** no se desarrollan porque la condición A bloquea la ejecución. **No son pruebas fallidas.**

## 5. Evidencia que fundamenta STOP-15a (condición A)

**Qué exigen el roadmap y el backlog:** un residuo limpio de media, volatilidad y estacionalidad (E01), con todos sus componentes causales (E02). Tsay pide además eliminar la dependencia lineal antes de aplicar BDS (E03).

| Componente | Evidencia | Situación |
|---|---|---|
| Media | TDA-08: $\rho_1$ no se distingue de cero; los rezagos 2 y 5 sí, por un margen pequeño (E04). STOP-8a activado (0,1425 ticks; la media no se modeló) (E05). La dependencia lineal no es estable entre años (TH18, E06) | Pequeña, pero no completamente nula; no existe un filtro de media auditado y estable |
| Volatilidad (causal) | TDA-10: estimadores rodante y EWMA causales verificados (E07). Sus residuos, con entrada causal `r_1m`, dejan $\rho_1(\lvert z\rvert)$ entre 0,1625 y 0,1726 (TDA-11 §3; E08, E09) | Dependencia residual de magnitud |
| Volatilidad (GARCH) | TDA-11: $\rho_1(\lvert z\rvert)$ = 0,0471 y $Q(60)$ = 5.892,73 (E10). Modelo retrospectivo, estimado con toda la muestra (E11). Persistencia en el tope de la restricción (E12) | Mejora la depuración, pero sigue dejando dependencia residual y no es causal |
| Estacionalidad | `s(m)` y `r_tilde` etiquetados `RETROSPECTIVO` (E13). RELOJ inestable en TDA-14 (E15, E16) | Sin versión causal; un `s(m)` fijo no representa todos los años |

**Contexto, no normativo:**
- **X01:** el umbral 0,02 de $\rho_1(\lvert r\rvert)$ procede de TDA-09, que lo usó para su propio veredicto (TH19). Aquí sirve solo para situar descriptivamente las cifras de dependencia residual. **No fue predeclarado como criterio de aceptación de TDA-15 y no se adopta como regla.** La decisión no depende de ningún umbral numérico nuevo.
- **X02:** la ventaja del GARCH sobre los benchmarks se explica sobre todo por su velocidad de memoria.
- **X03:** sus parámetros son sensibles al tratamiento de las fronteras.
- **X04:** el bloque ajustado por reloj deja entre 0,1061 y 0,1177, pero depende de `s(m)`, que es retrospectivo.

**Conclusión.** Con los componentes auditados disponibles no se puede construir un residuo que sea a la vez causal y limpio de media, volatilidad y estacionalidad:
- el único filtro de volatilidad que reduce más la dependencia (GARCH) es retrospectivo y aun así la deja;
- los filtros causales la dejan mayor;
- el factor estacional es retrospectivo y el reloj es inestable.

El roadmap lo resuelve expresamente: «si no existe residuo limpio, no se testea: el test detectaría lo ya conocido» (§ TDA-15, `STOP-15a`). **STOP-15a se justifica solo con evidencia previa. No se obtuvo ningún resultado nuevo.**

**Corrección respecto de la auditoría previa.** La auditoría de viabilidad resumió como «filtros causales» un rango de 0,106–0,17. Contrastado con el informe oficial de TDA-11 §3:
- solo el bloque `raw` (0,1625–0,1726) es totalmente causal;
- el bloque `clock_adjusted` (0,1061–0,1177) depende de `s(m)`, que es retrospectivo (X04).

Este informe usa la distinción correcta.

## 6. Limitaciones e incertidumbres

- **El roadmap no define numéricamente qué es un residuo «limpio».** La decisión se apoya en hechos cualitativos verificables:
  - componentes retrospectivos;
  - dependencia residual de magnitud documentada;
  - media no filtrada;
  - inestabilidad de RELOJ.

  No se usa ningún umbral nuevo.
- **Que hoy no exista un residuo limpio no demuestra que no pueda construirse.** Construirlo exigiría un trabajo metodológico nuevo (un factor estacional causal y un filtro de volatilidad causal validado), fuera del alcance del roadmap actual y no autorizado.
- **No se afirma qué resultado habrían dado BDS o el F-test.** Lo único que se afirma es que un rechazo no podría atribuirse a no linealidad mientras la dependencia de magnitud conocida siga sin eliminarse.
- **Las dificultades de implementación no justifican este cierre y no son una imposibilidad matemática.** Por ejemplo, la memoria que requiere una implementación directa de BDS sobre series muy largas.
- **Ningún resultado de TDA-15 puede citarse como propiedad del MNQ.**

## 7. Estado final de las hipótesis

- **TH32 — NO EVALUADA POR STOP-15a.** No se ejecutaron BDS ni F-test de Tsay. No existe evidencia nueva para afirmar o negar la presencia de no linealidad residual.
- **TH33 — NO EVALUADA POR DEPENDENCIA DE TH32.** No se realizaron pruebas de no linealidad a 1, 5 y 10 minutos. No se puede concluir nada nuevo sobre su comportamiento al cambiar la frecuencia.

No se asigna ninguna de las tres categorías de veredicto que el roadmap reserva para pruebas realmente ejecutadas (§ TDA-15, «Outputs obligatorios»).

## 8. Pruebas y cálculos expresamente no realizados

- Construcción del residuo estandarizado de TDA-15 (ni causal ni retrospectivo).
- Test BDS (con ningún valor de k ni de delta).
- F-test de Tsay, RESET y cualquier otro test de no linealidad.
- Calibración por permutación o por simulación, y cualquier métrica G2 específica de TDA-15.
- Corrección por pruebas múltiples.
- Análisis a 1, 5 y 10 minutos.
- Análisis por subperíodo.
- Benchmarks de tiempo o de memoria.
- Lectura de datos del MNQ y del holdout.
- Ajuste de TAR, STAR, Markov switching o redes neuronales.
- Nuevos modelos de volatilidad o de estacionalidad.

Este cierre no genera tablas CSV, gráficos, estadísticos ni simulaciones.

## 9. Restricciones para etapas posteriores

- TDA-15 no aporta ninguna afirmación sobre la presencia o ausencia de no linealidad residual en el MNQ.
- Ninguna etapa posterior puede citar TDA-15 como evidencia de linealidad, de no linealidad ni de su dependencia con la frecuencia.
- No se justifica modelar TAR, STAR, Markov switching ni redes a partir de TDA-15.
- Se mantienen todas las restricciones del cierre de TDA-14 (`configs/tda14_cierre.yaml`).
- Una investigación futura de no linealidad requiere un protocolo nuevo y autorización explícita; no implica reabrir automáticamente TDA-15.
- Holdout cerrado; G0–G6 vigentes.

## 10. Veredicto oficial

- **TDA-15: CERRADA METODOLÓGICAMENTE — STOP-15a ACTIVADO — ANÁLISIS NO EJECUTADO.**
- **TH32: NO EVALUADA.**
- **TH33: NO EVALUADA.**
- **DECISIONES DE DISEÑO IRIS: NINGUNA.**

La rama queda cerrada dentro del roadmap actual.

**En palabras sencillas.** TDA-15 quería comprobar si los movimientos del MNQ esconden algún patrón «curvo» que no sea ni lineal ni la conocida memoria de la volatilidad. Para preguntarlo con rigor, primero hay que quitar de los datos todo lo ya conocido: la pequeña dependencia lineal, la memoria de la volatilidad y el patrón de la hora del día. Hoy no podemos hacer esa limpieza de forma fiable:
- los filtros disponibles dejan parte de la memoria de la volatilidad;
- el mejor de ellos solo puede calcularse mirando toda la muestra;
- el patrón horario cambió entre años.

Hacer el test sin esa limpieza confundiría lo ya conocido con algo nuevo. Por eso no se hace. No hemos descubierto que el MNQ sea lineal ni que no lo sea: simplemente, la pregunta queda sin responder.
