# 00 — AUDITORÍA INICIAL DEL REPOSITORIO Y DE LOS DATOS

**Proyecto:** OHLCV_DATAROAD — MNQ OHLCV 1 minuto
**Fecha:** 2026-08-16
**Tipo de documento:** Auditoría de arranque. **No es análisis empírico, no contiene hallazgos estadísticos, no fija targets ni decisiones de ML.**
**Alcance de verificación:** Todo lo marcado `VERIFICADO EN ESTA AUDITORÍA` fue comprobado directamente contra los 27 archivos de `data/raw/mnq/` en esta sesión, con `pandas`, sin usar librerías de terceros para el parseo. Todo lo marcado `HEREDADO (no reverificado)` proviene de `docs/instruments/mnq/MNQ_DATA_PRIOR_KNOWLEDGE.md` y no fue reproducido de forma independiente en esta tarea.

---

## 1. Estado actual del repositorio

| Directorio | Estado | Detalle |
|---|---|---|
| `README.md` | Prácticamente vacío | Sólo el título `# ohlcv_dataroad`. Sin descripción, sin instrucciones de instalación. |
| `docs/instruments/mnq/` | Completo | `MNQ_DATA_PRIOR_KNOWLEDGE.md` (948 líneas) — memoria heredada de auditorías previas, no evidencia vigente. |
| `docs/methodology/` | Completo | Síntesis transversal, roadmap (`TDA-00`…`TDA-18`) y backlog de hipótesis (`TH01`…`TH37`). Documentos de diseño, no ejecutados. |
| `docs/tsay/` | Completo | Capítulos 1, 2, 3, 4, 5, 7, 11 presentes como informes propios. |
| `docs/books_and_references/` | Presente | PDF fuente de Tsay (3ª ed.). |
| `configs/` | **Vacío** | Ningún archivo de configuración. Nada que fijar ventana horaria, tick size, calendario, etc. de forma reproducible. |
| `data/raw/mnq/` | Presente y poblado | 27 archivos `.Last.txt`, 117 MB. Único contenido real de datos del repositorio. |
| `data/interim/` | **Vacío** | Ninguna etapa de preparación ejecutada todavía. |
| `data/processed/` | **Vacío** | Ninguna serie continua ni dataset derivado existe. |
| `notebooks/mnq/` | **Vacío** | Ningún notebook. |
| `reports/mnq/` | **Vacío antes de esta tarea** | Este documento es el primer artefacto que se escribe en la carpeta. |
| `src/ohlcv_dataroad/` | **Vacío** | Ni siquiera `__init__.py`. No existe paquete Python instalable. |
| Raíz del repo | Sin tooling | No hay `pyproject.toml`, `setup.py`, `requirements.txt` ni `environment.yml`. `pandas` no estaba instalado en el entorno al empezar esta auditoría (se instaló ad-hoc para poder verificar). |

**Conclusión de esta sección.** El repositorio es, en la práctica, **documentación + datos crudos**. No existe todavía ninguna línea de código de ingestión, ningún artefacto intermedio, ningún notebook y ninguna configuración versionada. El roadmap metodológico (`TDA-00`…`TDA-18`) está completamente diseñado pero **cero etapas ejecutadas**. Esto es coherente con la instrucción del proyecto: "antes de implementar análisis, modelos o notebooks, auditar".

---

## 2. Inventario de datos

**VERIFICADO EN ESTA AUDITORÍA**

| Campo | Valor |
|---|---|
| Archivos | 27, en `data/raw/mnq/`, patrón `NN_mnq_MM_YY.Last.txt` |
| Separador | `;`, 6 campos: `timestamp;open;high;low;close;volume` |
| Formato de timestamp crudo | YYYYMMDD HHMMSS. Los timestamps corresponden a UTC, confirmado por la fuente de los datos; el archivo raw no incluye offset explícito. |
| Filas totales | **2.329.783** |
| Bytes totales | 117 MB en disco (`du -sh`) |
| Rango temporal global | **2019-12-23 03:01:00 → 2026-07-31 20:10:00** (tz-naive, tal como vienen) |
| Secuencia de contratos | H20, M20, U20, Z20, H21, M21, U21, Z21, H22, M22, U22, Z22, H23, M23, U23, Z23, H24, M24, U24, Z24, H25, M25, U25, Z25, H26, M26, U26 — **secuencia trimestral H/M/U/Z completa, sin hueco de trimestre**, 27 contratos |
| Contrato incompleto | `26_mnq_09_26.Last.txt` (U26) — 54.015 filas, vigente y sin expirar en la fecha del snapshot |
| Filas por contrato completo | mínimo 81.660 (H20), máximo 96.755 (M25); media ≈ 87.530, desviación ≈ 3.254 |

Estas cifras **coinciden exactamente** con las reportadas en `MNQ_DATA_PRIOR_KNOWLEDGE.md` §2, lo cual confirma que se está trabajando sobre el mismo snapshot que documenta esa memoria heredada. No se verificó el hash SHA-256 exacto citado en la memoria (`e382a75ac222…`) porque el procedimiento de hashing del manifest original no está documentado en detalle (¿hash por archivo concatenado en qué orden, con o sin metadata? no está especificado) y reconstruirlo a ciegas sería una decisión metodológica nueva, no una verificación. Esto queda como **pendiente abierto** (sección 6).

---

## 3. Auditoría básica del dataset

Todo lo siguiente fue calculado de forma independiente en esta sesión, leyendo los 27 archivos crudos con `pandas`.

### 3.1 Esquema y tipos
- 6 campos por línea en el 100 % de las filas leídas (pandas no reportó errores de parseo con `sep=";"`, sin encabezado).
- Columnas resultantes: `timestamp` (datetime, parseado con formato fijo `%Y%m%d %H%M%S`), `open`, `high`, `low`, `close` (float), `volume` (entero).
- **0 valores nulos** en ninguna columna, en ningún archivo.

### 3.2 Integridad OHLC y de volumen
- **0 violaciones** de `H≥L`, `H≥max(O,C)`, `L≤min(O,C)` en las 2.329.783 filas.
- **0 precios ≤ 0.**
- **0 volúmenes negativos.** Volumen mínimo observado: 1 (no hay volumen = 0 en ningún archivo — ver nota en §5).
- **0 violaciones de grilla de tick** con tick = 0.25 en `open`, `high`, `low`, `close` (verificado sobre las cuatro columnas de precio de las 2.329.783 filas).

### 3.3 Timestamps y duplicados
- **0 timestamps duplicados dentro de un mismo archivo**, en los 27 archivos.
- Los 27 archivos son **monótonos crecientes** internamente.
- **0 duplicados exactos** (timestamp + OHLCV + archivo idénticos) en el dataset completo.
- **0 duplicados de (timestamp, archivo)** — coherente con la monotonicidad interna.

### 3.4 Continuidad temporal y gaps
- Hueco interno máximo por archivo: entre 1 día 23h y 3 días 1h en todos los casos — compatibles con fines de semana largos ordinarios (viernes de cierre → domingo de apertura, o feriados de 3 días).
- **0 gaps > 100 horas** en todo el dataset (ni dentro de archivo ni en las transiciones entre archivos consecutivos).
- El archivo M23 (`13_mnq_06_23.Last.txt`), que la memoria heredada documenta como sede de un gap histórico de ~260 h ya resuelto, tiene en el snapshot actual un hueco interno máximo de **2 días 8h 46min** — consistente con la resolución reportada. No se encontró rastro del gap de ~260h.

### 3.5 Solapamientos entre contratos consecutivos
De las 26 transiciones entre archivos consecutivos:
- **23 transiciones limpias** (el último timestamp del saliente es anterior al primero del entrante).
- **3 transiciones con solapamiento real**, verificadas de forma independiente:

| Transición | Fin del saliente | Inicio del entrante | Solapamiento |
|---|---|---|---|
| `19_mnq_12_24` → `20_mnq_03_25` (Z24→H25) | 2024-12-20 21:30 | 2024-12-12 03:01 | ~9 días |
| `20_mnq_03_25` → `21_mnq_06_25` (H25→M25) | 2025-03-22 15:03 | 2025-03-13 03:01 | ~10 días |
| `25_mnq_06_26` → `26_mnq_09_26` (M26→U26) | 2026-06-18 13:30 | 2026-06-08 03:03 | ~11 días |

Estas tres transiciones y sus magnitudes aproximadas coinciden con lo documentado en la memoria heredada. Las otras 23 no solapan, lo que confirma que el patrón es la excepción, no la norma, y que sigue sin explicación conocida.

### 3.6 Volumen: valores extremos
Las 3 barras con volumen extremo documentadas en la memoria heredada se reprodujeron de forma exacta e independiente:

| Timestamp | Volumen | Contrato | OHLC |
|---|---:|---|---|
| 2026-01-27 03:44:00 | 1.534.923 | `24_mnq_03_26` | 25567.75 / 25977.50 / 25543.25 / 25972.25 |
| 2025-07-15 13:52:00 | 1.451.062 | `22_mnq_09_25` | 22862.00 / 23222.75 / 22805.00 / 23159.50 |
| 2025-07-01 10:08:00 | 1.258.222 | `22_mnq_09_25` | 22816.25 / 22934.00 / 22780.00 / 22831.50 |

El resto del dataset tiene un volumen máximo por barra de orden 8.500–31.400 (el máximo "normal" más alto observado es 31.379, en `26_mnq_09_26`). Las tres barras extremas caen a **100–170×** ese rango. El OHLC de las tres es internamente válido (no dispara ninguna invariante de §3.2). **Las tres caen dentro del rango 2025-06-23 → 2026-07-31.**

### 3.7 Frontera de hold-out declarada (sólo verificación aritmética, sin análisis de contenido)
Usando la frontera documentada en la memoria heredada (`2025-06-23 00:00:00`), la partición reproduce exactamente las cifras citadas:

```
filas con timestamp <  2025-06-23 00:00:00:  1.937.230
filas con timestamp >= 2025-06-23 00:00:00:    392.553
total:                                        2.329.783
% reservado:                                     16,85 %
```

**Esta es una comprobación puramente aritmética sobre la partición, no un análisis de contenido del período reservado.** Las tres barras de volumen extremo (§3.6) y el solapamiento M26→U26 (§3.5) caen dentro de ese 16,85 %; su existencia y ubicación ya eran conocidas antes de esta sesión (memoria heredada §11.4) y no se investigaron aquí más allá de reproducir su ubicación temporal.

### 3.8 Lo que esta auditoría **no** verificó de forma independiente
- El hash SHA-256 exacto del snapshot documentado en la memoria heredada.
- No se realizó una revalidación programática de la zona horaria. La fuente de los datos confirma que los timestamps del dataset raw corresponden a UTC.
- La clasificación tripartita de barras inactivas (ausente / volumen 0 / forward-fill) — **no se encontró ningún volumen = 0 en el dataset**, lo cual en sí mismo es un dato relevante: o bien el proveedor nunca emite una barra de volumen 0, o bien las barras sin operaciones simplemente no aparecen como fila. Esto es exactamente la pregunta abierta que documenta `TH04`/`TDA-02`, y esta auditoría no la resuelve, sólo confirma que sigue abierta y aporta un dato adicional (0 filas con `volume=0` en 2.329.783).
- Cobertura por jornada, feriados, cierres anticipados, DST.
- Cualquier estadístico de distribución, dependencia o volatilidad — fuera de alcance de esta tarea por instrucción explícita.

---

## 4. Correspondencia entre datos y roadmap

Contraste entre lo que existe realmente en el dataset (secciones 2–3) y lo que exige el roadmap (`Tsay_OHLCV_analysis_roadmap.md`) y el backlog (`Tsay_empirical_hypotheses_backlog.md`).

| Ítem del roadmap | Clasificación | Justificación |
|---|---|---|
| `TDA-00` — Integridad intra-barra e invariantes físicos | **EJECUTABLE AHORA** | Todos los insumos existen en `data/raw/mnq/`. Esta auditoría ya adelantó una verificación equivalente (§3.1–3.3) con resultado limpio; falta formalizarla como artefacto (`TDA00_inventario.md`, `TDA00_violaciones.csv`, máscara `bad_data`) dentro de un pipeline versionado. |
| `TDA-01` — Semántica temporal, de sesión y de contrato | **REQUIERE DECISIÓN METODOLÓGICA** | La semántica del timestamp (inicio vs cierre de barra) es `UNRESOLVED DATA QUESTION` heredada y esta auditoría no aporta evidencia nueva. El roadmap ya prescribe qué hacer si queda indeterminada (supuesto conservador + marca de pregunta bloqueante), pero **la decisión de adoptar ese supuesto debe tomarse explícitamente**, no heredarse en silencio. |
| `TDA-02` — Integridad del eje temporal y calendario | **REQUIERE PREPARACIÓN PREVIA** | Depende de `TDA-01` (necesita la sesión definida). Los datos permiten calcularlo (hay timestamp completo, sin huecos estructurales sorpresivos según §3.4), pero no puede ejecutarse honestamente antes de fijar la semántica de sesión. |
| `TDA-03` — Rolls y serie continua | **REQUIERE DECISIÓN METODOLÓGICA** | Los 3 solapamientos reales (§3.5) están localizados, pero el **método de ajuste de contrato continuo** (ninguno / ratio / aditivo) no está decidido ni por la memoria heredada ni por esta auditoría. La política de rollover heredada (§7.1 de la memoria) es reutilizable como **lógica**, pero sus parámetros (umbral de volumen 55 %, 691 barras/día) están atados a una ventana horaria que OHLCV_DATAROAD no ha adoptado y **deben recalibrarse bajo el alcance propio (24h completas o ventana a definir)**. |
| `TDA-04` — Construcción de variables de análisis | **REQUIERE PREPARACIÓN PREVIA** | Depende de `TDA-01`–`TDA-03`. Ningún bloqueo de datos: se puede calcular en cuanto exista la serie con reglas de no-cruce definidas. |
| `TDA-05` — Resolución efectiva y discreción | **REQUIERE PREPARACIÓN PREVIA** | Depende de `TDA-04`. Dato base disponible (tick=0.25 confirmado en §3.2). |
| `TDA-06` — Perfil determinista intradía | **REQUIERE PREPARACIÓN PREVIA** | Depende de tener retornos causales bien definidos (`TDA-04`/`05`). Es la etapa que, según la síntesis, debe preceder a toda dependencia/volatilidad — no puede adelantarse. |
| `TDA-07`…`TDA-10` (distribución, dependencia, escala/forma) | **NO EVALUABLE TODAVÍA** | Dependen en cadena de todo lo anterior. Los datos existen; el trabajo previo no. |
| `TDA-11`, `TDA-15`, `TDA-16` (GARCH, no linealidad, estados latentes) | **CONDICIONAL / NO EVALUABLE TODAVÍA** | Por diseño del propio roadmap, sólo se ejecutan si una etapa previa genera la pregunta que las justifica. Nada que decidir ahora. |
| `TDA-13` EVT | **CONDICIONAL** | El inventario/triage de extremos (obligatorio) sí es ejecutable en el conjunto de investigación; la rama EVT depende de resultados de `TDA-12`, que no existen aún. |
| Hipótesis `TH01`–`TH06` (Nivel 0) | **EJECUTABLE AHORA**, salvo `TH02` que es **DECISIÓN METODOLÓGICA** | Coincide con `TDA-00`–`TDA-03`. |
| Hipótesis `TH07`+ (Nivel 1 en adelante) | **REQUIERE PREPARACIÓN PREVIA** | Ninguna es ejecutable sin que exista primero una serie de retornos causal y auditada. |
| Zona horaria | **RESUELTA — UTC CONFIRMADO** | La fuente de los datos confirma que los timestamps del dataset raw corresponden a UTC. No debe tratarse como hipótesis ni como decisión metodológica pendiente en TDA-01. |
| Hold-out — exploración/estadística sobre el período reservado | **PROHIBIDO por gobernanza** | Confirmado en la memoria heredada §11.3 y respetado en esta auditoría: sólo se hizo aritmética de partición (§3.7), ninguna caracterización de contenido. |

**Lectura general.** El dataset es estructuralmente apto para arrancar el roadmap desde `TDA-00`. No hay ningún hallazgo en esta auditoría que sugiera `STOP-0`: 0 violaciones de invariantes, 0 nulos, 0 duplicados, grilla de tick limpia. El cuello de botella no es la calidad del dato — es que **no existe ningún código, configuración ni artefacto** que ejecute siquiera la primera etapa.

---

## 5. Problemas o riesgos detectados

1. **Ausencia total de tooling reproducible.** Sin `pyproject.toml`/`requirements.txt`, cualquier ejecución futura depende de que el entorno se arme manualmente (como se hizo aquí, instalando `pandas` ad-hoc). Riesgo de irreproducibilidad entre sesiones.
2. **`configs/` vacío.** No hay ningún lugar versionado donde fijar decisiones como tick size, alcance temporal (24h vs ventana), método de ajuste de roll, etc. Si estas decisiones se toman "en el código" sin quedar en config, se repite el patrón que la memoria heredada marca como error histórico (documentación que describe operaciones no ejecutadas, o decisiones que no quedan trazadas).
3. **Barras con `volume=0` inexistentes en el dataset (0 de 2.329.783).** No es necesariamente una anomalía, pero es un dato nuevo que la memoria heredada no reporta explícitamente en estos términos. Puede significar que el proveedor omite el minuto en vez de emitirlo con volumen cero, lo cual es relevante para `TH04`. **No se investigó más porque escapa al alcance de "auditoría de inventario".**
4. **Tres barras de volumen extremo dentro del hold-out.** Confirmado que siguen sin investigar (correctamente, según la gobernanza) y que su investigación requiere autorización explícita de acceso al hold-out — no implícita por el hecho de haber sido "redescubiertas" en esta auditoría.
5. **Hash SHA-256 del snapshot no reverificado.** No se puede afirmar con esta auditoría que el snapshot es *exactamente* el mismo que describe la memoria heredada a nivel de bytes, sólo que coincide en todas las cifras estructurales comprobables (filas, rango, contratos, solapamientos, anomalías de volumen). Es una coincidencia fuerte pero no una prueba criptográfica.
6. **La política de rollover todavía no está fijada para OHLCV_DATAROAD.** Es prerequisito de TDA-03 y requiere una decisión metodológica explícita. La zona horaria del dataset raw, en cambio, está confirmada como UTC y no constituye una decisión pendiente.

---

## 6. Decisiones pendientes

Estas son decisiones que **no pueden resolverse ni con los datos ni con la documentación existente**, y que esta auditoría señala explícitamente en vez de asumir:

1. **Alcance temporal de OHLCV_DATAROAD**: ¿se analizan las ~24 h completas de cada jornada, o se adopta alguna ventana operativa? Determina si la política de rollover heredada (umbral 55 %, 691 barras/día) es directamente reutilizable o debe recalibrarse por completo.
2. **Semántica del timestamp** (inicio vs cierre de barra): sigue indeterminada. El roadmap prescribe un supuesto conservador por defecto (inicio de barra) si no se resuelve documentalmente — **pero adoptarlo es una decisión que el usuario debe confirmar**, no algo que este documento decide.
3. **Método de ajuste para la serie continua** (ninguno / ratio / aditivo) en las tres transiciones con solapamiento real: no está decidido. Afecta directamente qué estadísticos son utilizables (invariantes a escala) en cualquier análisis posterior.
4. **Verificación del hash del snapshot**: ¿se reconstruye el procedimiento de hashing exacto de la memoria heredada para confirmar que este es el mismo snapshot al byte, o se acepta la coincidencia estructural (secc. 2–3) como suficiente para proceder?
5. **Qué constituye "grilla esperada" para `TDA-02`**: depende directamente de la decisión (1). No se puede definir sin ella.

Ninguna de estas cinco decisiones se tomó en esta tarea.

---

## 7. Propuesta concreta de la primera etapa empírica a implementar

**Propuesta: `TDA-00` — Integridad intra-barra e invariantes físicos, formalizada como pipeline reproducible sobre el conjunto de investigación (excluyendo el hold-out desde el primer paso).**

Razones:
- Es la raíz del roadmap (sin dependencias) y la única etapa que no requiere ninguna de las 5 decisiones pendientes de la sección 6 — el chequeo de invariantes físicos no depende de zona horaria, semántica de timestamp ni política de roll.
- Esta auditoría ya adelantó su resultado esperado de forma ad-hoc (§3.1–3.3): 0 violaciones. Formalizarla como etapa productiva convierte un chequeo manual en un artefacto reproducible, versionado y con máscara persistida — exactamente el patrón que la memoria heredada marca como validado (§15.2) y cuya ausencia es el problema real del repositorio hoy (sección 5, punto 1–2).
- Al ejecutarla exclusivamente sobre el conjunto de investigación (< 2025-06-23), respeta la gobernanza del hold-out desde el primer artefacto productivo, en vez de dejarlo como una regla a aplicar "después".
- Da además la oportunidad de fijar, en el mismo movimiento, el **tooling mínimo** que hoy no existe (entorno reproducible, estructura de config), sin lo cual ninguna etapa posterior puede ejecutarse de forma trazable.

No se propone ejecutar `TDA-01` todavía en la misma etapa, porque `TDA-01` requiere primero decidir o documentar la semántica del timestamp (decisión pendiente #2), y mezclar ambas etapas oscurecería cuál resultado depende de qué supuesto.

---

## 8. Archivos que deberían crearse o modificarse en la siguiente etapa

**No se crean en esta tarea — se listan como propuesta para aprobación.**

| Archivo | Propósito |
|---|---|
| `pyproject.toml` (o `requirements.txt`) | Fijar dependencias reproducibles (pandas como mínimo) — hoy inexistente. |
| `configs/mnq_snapshot.yaml` (o similar) | Declarar identidad del snapshot: nº de archivos, rango esperado, hold-out boundary, tick size — como config versionada, no como constantes en código. |
| `src/ohlcv_dataroad/ingest/tda00_integrity.py` (o equivalente) | Lógica de `TDA-00`: parseo, invariantes duros, grilla de tick, unicidad/monotonicidad de timestamps, localización de violaciones. |
| `src/ohlcv_dataroad/__init__.py` | Convertir `src/ohlcv_dataroad/` en paquete real (hoy vacío). |
| `data/interim/mnq/tda00_bad_data_mask.parquet` (o csv) | Máscara persistida de filas con violaciones — vacía o casi vacía según lo observado, pero como artefacto formal, no como print de una sesión de auditoría. |
| `reports/mnq/TDA00_inventario.md` | Inventario de columnas, tipos, rango, conteo — salida obligatoria de la etapa según el roadmap. |
| `reports/mnq/TDA00_violaciones.csv` | Una fila por violación (se espera vacío dado lo observado en esta auditoría, pero el artefacto debe existir y poder estar vacío de forma legítima). |

---

## Resumen de hallazgos para revisión

- El dataset crudo (27 archivos, 2.329.783 filas) es estructuralmente limpio: 0 nulos, 0 violaciones OHLC, 0 duplicados, 0 desviaciones de la grilla de tick, verificado de forma independiente en esta sesión.
- Todas las cifras estructurales clave de la memoria heredada (rango temporal, contratos, 3 solapamientos, 3 barras de volumen extremo, partición hold-out) se **reprodujeron exactamente** de forma independiente.
- El repositorio no tiene código, configuración, artefactos intermedios ni notebooks — sólo documentación y datos crudos.
- Ninguna etapa del roadmap se ejecutó formalmente todavía; ninguna hipótesis del backlog se convirtió en hallazgo.
- 5 decisiones metodológicas quedan explícitamente pendientes y sin resolver (sección 6).
- Se propone `TDA-00` como primera etapa formal a implementar, sin ejecutarla en esta tarea.

**Queda a la espera de aprobación antes de implementar `TDA-00` o cualquier otra etapa.**
