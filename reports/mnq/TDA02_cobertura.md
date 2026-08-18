# TDA-02 — Integridad del eje temporal y del calendario

**Etapa:** `docs/methodology/Tsay_OHLCV_analysis_roadmap.md` § TDA-02
**Depende de:** TDA-00 (`PASS`), TDA-01 (`PASS_WITH_OPEN_QUESTIONS`)
**Alcance de datos:** exclusivamente el conjunto de investigación (22 archivos, `< 2025-06-23 00:00:00 UTC`, ver `configs/mnq_snapshot.yaml`). Ningún archivo de `holdout_files` fue abierto en esta etapa.
**Evidencia reproducible:** `reports/mnq/TDA02_huecos.csv`, `TDA02_cobertura_por_anio.csv`, `TDA02_cobertura_por_mes.csv`, `TDA02_dias_incompletos.csv`, `TDA02_barras_fuera_de_grilla.csv`, `TDA02_dst_evidencia.csv`, `TDA02_bordes_de_archivo.csv`, `TDA02_barras_planas_candidatas.csv`, `TDA02_rachas_planas_largas.csv`, `TDA02_heatmap_completitud.png`, `data/interim/mnq/tda02_barra_inactiva_mask.parquet`. Todo generado por `python -m ohlcv_dataroad.ingest.run_tda02`.

> **Revisión 2 de este documento.** Corrige dos puntos de la primera versión, señalados en revisión: (1) la grilla esperada no excluía el tramo del break secundario (16:15–16:30 NY) a pesar de que TDA-02 ya clasificaba 353 huecos reales con esa causa — sesgaba la cobertura hacia abajo; (2) la conclusión de §10 sobre la representación de un minuto de mercado abierto sin operaciones se afirmaba como `AUSENTE confirmado`, cuando la evidencia disponible (huecos cortos intra-sesión, sin ninguna fuente documental que describa explícitamente ese caso) sólo permite decir que es **compatible** con `AUSENTE`, no que lo confirma — con OHLCV de 1 minuto no puede distinguirse una ausencia de operación real de una pérdida del dato histórico. Ambas correcciones están detalladas en las secciones correspondientes, marcadas explícitamente como revisión.

> **Este documento no construye ninguna serie continua, no decide rollover, no selecciona contrato activo y no elige ninguna ventana operativa de análisis.** Audita la integridad del eje temporal de cada archivo por separado, contra la grilla que la estructura nativa de negociación de CME para el complejo de futuros de índices bursátiles predice.

---

## 1. Fuente del calendario y su verificación

### 1.1 Por qué no se adoptó ninguna ventana operativa

TDA-01 (§6) ya advirtió que la ventana `04:30–16:00` (u otra ventana heredada de proyectos anteriores) no tiene respaldo empírico propio de este proyecto. Esta tarea instruye explícitamente no adoptarla, ni RTH, ni los antiguos regímenes horarios, ni ninguna ventana arbitraria. "Grilla esperada" significa aquí la grilla de la estructura **nativa** de negociación del complejo al que pertenece MNQ (Micro E-mini Nasdaq-100): un instrumento que cotiza casi continuamente, con un cese diario de negociación conocido.

### 1.2 Fuente elegida: `pandas_market_calendars`, calendario `CME_Equity`

Prioridad de evidencia exigida por la tarea: (1) documentación oficial CME, (2) documentación oficial NinjaTrader, (3) evidencia ya establecida por TDA-01.

**CME (`cmegroup.com`) volvió a estar bloqueado en esta sesión** (mismo bloqueo/reset de conexión que documentó TDA-01, §2.3 — reintentado explícitamente con `WebFetch`, en dos revisiones distintas de este documento, sobre la página de especificación del contrato, el calendario de feriados de CME y un aviso Special Executive Report; los tres intentos fallaron con `ECONNRESET`/timeout). NinjaTrader no publica un calendario de sesión de mercado — no es una fuente aplicable a esta pregunta.

Ante ese bloqueo, se usó la mejor evidencia documental disponible: la librería de código abierto `pandas_market_calendars` (versión 5.4.0, instalada para esta etapa — ver `pyproject.toml`), calendario `CME_Equity` (aliases `CME_Equity`/`CBOT_Equity`; `pandas_market_calendars/calendars/cme.py`) — el calendario que esta librería usa para el complejo de futuros de **índices bursátiles** de CME (E-mini/Micro E-mini S&P 500, Nasdaq-100, Dow, Russell...), al que pertenece MNQ. El propio código fuente de la librería cita `http://www.cmegroup.com/tools-information/holiday-calendar.html` como fuente de sus reglas de feriados — es una **codificación de terceros de la especificación de CME**, no una fuente independiente.

**Esta tarea exige explícitamente no tratar una librería de calendario como fuente de verdad sin verificarla.** Se hizo, en cuatro frentes:

**(a) Corte de mantenimiento diario y cierre semanal.** El calendario declara la sesión como `17:00 → 16:00 America/Chicago` (apertura del día anterior). Traducido a Nueva York (DST-aware): cierre `17:00` NY, reapertura `18:00` NY (primera barra completa `18:01`, convención de cierre de barra de TDA-01). Coincide **exactamente** con las dos primeras anclas forenses de TDA-01 (última barra en `17:00:00` NY en 91,5 % de 1.178 huecos de mantenimiento; primera barra en `18:01:00` NY en 94,9 %).

**(b) Cierres anticipados — verificación fecha por fecha, la pregunta que TDA-01 dejó pendiente (§13, ítem 5).** Los 31 huecos con firma exacta `13:00→18:01` NY que TDA-01 encontró como candidatos forenses se cruzaron, uno por uno, contra las fechas y horas de cierre anticipado que predice `CME_Equity`. **Resultado: 31/31 coinciden exactamente.** Cierra la pregunta abierta de TDA-01.

**(c) Corrección documental + forense: el break secundario NO es permanente — fue eliminado por CME.** `CME_Equity` declara un break intradiario permanente de `15:15–15:30 America/Chicago` (`16:15–16:30` NY). **Fuente primaria citada para esta tarea (revisión 2): CME Special Executive Report / Globex Notice de junio de 2021 — el break de 15 minutos de futuros de índices bursátiles (Equity Index) fue ELIMINADO, con efecto en la fecha de negociación (trade date) `2021-06-28` (lunes).** La librería no modela este cambio (lo trata como vigente siempre para todo el historial).

Verificación forense independiente, ya construida en la revisión 1 de este documento: el hueco con la firma exacta de dos lados del break (`16:15:00`→`16:31:00` NY) aparece en **353 sesiones consecutivas**, desde 2019-12-23 hasta el **viernes 2021-06-25** inclusive, y no vuelve a aparecer en ninguna de las 1.082 sesiones restantes. **La fecha forense (última aparición: viernes 25/06/2021) y la fecha documental (eliminación con efecto lunes 28/06/2021, el primer día hábil siguiente) coinciden exactamente** — la cita documental y la evidencia forense se corroboran mutuamente. Esta fecha de corte (`SECONDARY_BREAK_LAST_TRADING_DATE = 2021-06-25`, `SECONDARY_BREAK_ABOLISHED_TRADING_DATE = 2021-06-28`) queda codificada de forma **explícita** en `session_calendar.py`, y se usa tanto para **excluir los minutos del break de la grilla esperada** en sesiones `<= 2021-06-25` (corrección de esta revisión — ver §2 y §3) como para **impedir** que la causa `SECONDARY_MAINTENANCE_BREAK` se asigne a ningún hueco en sesiones posteriores, sin depender de que la librería represente el cambio correctamente.

**(d) Corrección documental + forense puntual: `2025-01-09` — ahora CONFIRMADA, no sólo inferida.** `CME_Equity` declara este día (duelo nacional por James "Jimmy" Carter) como cierre completo, vía `adhoc_holidays` (`USNationalDaysofMourning`). **Fuente primaria citada para esta tarea (revisión 2): CME SER 9499R — la sesión de índices bursátiles de EE. UU. (U.S. Equity Index) en Globex ese día fue ABREVIADA, no cerrada: cierre `08:30 America/Chicago`, reapertura normal `17:00 America/Chicago`.**

Verificación forense independiente (`data/raw/mnq/20_mnq_03_25.Last.txt`): volumen normal (hasta 10.450 contratos/minuto) de forma continua desde la reapertura habitual del 8 de enero (`23:01:00 UTC` = `18:01` NY = `17:01` CT) hasta `2025-01-09 14:30:00 UTC` (`09:30` NY = **`08:30` CT exacto**) — momento en el que aparece un hueco de 511 minutos hasta la siguiente reapertura habitual. **La hora de cierre que declara el SER (`08:30 CT`) y la hora del último bar real observado (`08:30 CT` exacto) coinciden al minuto.** Esta fecha queda `CONFIRMADO` por fuente primaria + evidencia forense convergente — ya no es una inferencia forense sin corroborar, como se presentó en la revisión 1. Es, además, la única fecha de `USNationalDaysofMourning` dentro del rango de investigación (verificado explícitamente).

**(e) Discrepancia menor no corregida, documentada.** En varias fechas de cierre anticipado "post-feriado" (Black Friday, víspera de Navidad, 3 de julio), aparecen sistemáticamente ~15 minutos de barras reales **después** de la hora de cierre que declara la librería (ejemplo: `2020-11-27`, 15 barras reales de `13:01` a `13:15` NY). Pequeña, acotada, no afecta cobertura (esas barras se registran como fuera de grilla, nunca se pierden ni se cuentan como ausentes — §8). No se corrige el calendario por una diferencia de este tamaño.

**Nota sobre el nivel de confirmación de (c) y (d).** El acceso directo a `cmegroup.com` para releer los avisos originales en el propio sitio de CME estuvo bloqueado en esta sesión (igual que en las dos revisiones anteriores de este documento y en TDA-01): las citas de (c) y (d) se incorporan tal como fueron provistas para esta tarea (números de aviso, fechas y horarios exactos), no releídas directamente del documento original en esta sesión. En ambos casos, sin embargo, la cita documental y la evidencia forense propia **convergen de forma exacta, al minuto** — la forma más fuerte de corroboración disponible sin poder abrir el documento original.

**Conclusión de la verificación.** El calendario `CME_Equity`, con las dos correcciones documentadas de (c) y (d), es una fuente confiable para construir la grilla esperada de este instrumento: coincide exactamente con las tres anclas forenses independientes de TDA-01 y con el 99,67 % de los huecos de calendario del propio conjunto de investigación (§12, STOP-2).

---

## 2. Definición exacta de sesión y grilla

Procedimiento aplicado, en este orden (obligatorio, TDA-01 §7):

```
1. Timestamp crudo, localizado como UTC (tz_localize("UTC"), nunca offset fijo).
2. Horario de sesion de CME_Equity obtenido en UTC directamente (DST resuelto
   internamente por pandas_market_calendars via zoneinfo/IANA sobre
   America/Chicago; ningun offset manual).
3. Se localiza cada timestamp observado dentro de (market_open, market_close]
   de cada sesion (IntervalIndex cerrado por la derecha) para obtener su
   trading_date -- la fecha de CIERRE de esa sesion.
4. La grilla esperada de CADA sesion es open+1min .. close (inclusive),
   EXCLUYENDO -- correccion de esta revision -- el tramo
   break_start+1min .. break_end cuando trading_date <= 2021-06-25
   (session_calendar.SECONDARY_BREAK_LAST_TRADING_DATE, §1.2.c). Para
   trading_date >= 2021-06-28 no se excluye nada: la grilla es continua
   en esa franja horaria, igual que el resto de la sesion.
```

**Por qué esta corrección era necesaria.** El mercado estuvo estructuralmente detenido durante el break secundario, exactamente igual que durante el corte de mantenimiento nocturno — TDA-02 ya clasificaba ese hueco como `SECONDARY_MAINTENANCE_BREAK` en 353 sesiones, reconociéndolo como un cierre real. Sin embargo, la grilla esperada (`expected_bar_grid`/`expected_bar_grid_frame`) seguía generando esos 15 minutos por sesión como "esperados", contabilizándolos como ausentes en la cobertura — la misma inconsistencia que el roadmap prohíbe explícitamente para el corte de mantenimiento nocturno (nunca se cuenta como hueco de cobertura). Corregido en `session_calendar._session_expected_minutes` (ver `tests/test_session_calendar.py::test_secondary_break_minutes_excluded_from_grid_on_or_before_cutoff` y `test_secondary_break_minutes_included_in_grid_after_cutoff`).

**Efecto en la contabilidad — sin pérdida de datos.** Se verificó explícitamente que 89 barras reales (de 1.937.230) caen dentro del tramo ahora excluido de la grilla, en sesiones anteriores a 2021-06-25 (el break no fue perfectamente limpio todos los días: ocasionalmente hay un trade aislado justo al minuto siguiente de `break_start`). Estas 89 barras NO desaparecen de ninguna contabilidad: `find_out_of_grid_bars` se redefinió para capturarlas explícitamente con motivo `SECONDARY_BREAK_WINDOW` (§8), distinto de `NO_SESSION` (fin de semana/feriado/fuera de sesión). Se verificó la invariante de contabilidad completa: **minutos presentes (1.935.628) + barras fuera de grilla (1.602) = 1.937.230 filas observadas, exactamente.**

`trading_date` queda definida de forma explícita y testeada (`tests/test_session_calendar.py::test_trading_date_uses_the_session_close_date_not_the_open_date`).

**Contratos, no serie continua.** La grilla y la cobertura se calculan **por archivo**, recortadas a `[file_start, file_end]` de cada archivo — nunca a través de la frontera entre dos archivos (transición de contrato, alcance de TDA-03). Un archivo que empieza o termina a mitad de una jornada se compara contra el borde ideal de su propia sesión y se documenta por separado, con causa `FILE_BOUNDARY` (§7), completamente excluido del cálculo de cobertura.

---

## 3. Cobertura global y por subperíodo (recalculada tras la corrección de §2)

| Minutos esperados | Minutos presentes | Minutos ausentes | Cobertura |
|---:|---:|---:|---:|
| 1.947.490 | 1.935.628 | 11.862 | **99,3909 %** |

(Antes de la corrección de §2: 1.953.055 esperados, 17.338 ausentes, 99,1123 % — la corrección elimina de la grilla esperada 5.565 minutos que correspondían a mercado estructuralmente detenido, no a datos perdidos; los 89 minutos con barra real dentro de ese tramo se reclasifican a `SECONDARY_BREAK_WINDOW` en §8, no se cuentan como "presentes" en la cobertura de la grilla — ver la invariante de contabilidad de §2.)

### 3.1 Por año (`TDA02_cobertura_por_anio.csv`)

| año | esperados | presentes | ausentes | cobertura |
|---:|---:|---:|---:|---:|
| 2019 | 7.725 | 7.557 | 168 | 97,83 % |
| 2020 | 348.165 | 343.832 | 4.333 | 98,76 % |
| 2021 | 350.550 | 349.851 | 699 | 99,80 % |
| 2022 | 351.205 | 350.391 | 814 | 99,77 % |
| 2023 | 350.475 | 350.027 | 448 | 99,87 % |
| 2024 | 363.525 | 361.404 | 2.121 | 99,42 % |
| 2025 | 175.845 | 172.566 | 3.279 | 98,14 % |

2019 y 2025 son años parciales. 2020 mejora de 97,74 % a 98,76 % tras la corrección del break (era el año que más break contenía: todo el año cae antes del corte de 2021-06-25). Ningún año cae por debajo de 97,8 %.

### 3.2 Por mes (`TDA02_cobertura_por_mes.csv`, 67 meses)

Los 8 meses de menor cobertura:

| mes | cobertura |
|---|---:|
| 2020-03 | **90,31 %** |
| 2025-03 | 95,13 % |
| 2024-03 | 96,68 % |
| 2020-10 | 96,80 % |
| 2025-04 | 96,93 % |
| 2025-06 | 97,68 % |
| 2019-12 | 97,83 % |
| 2022-11 | 97,88 % |

2020-03 sigue siendo, por un margen amplio, el mes con menor cobertura — la semana de crisis de marzo de 2020 (§6) no está relacionada con el break secundario, así que la corrección de §2 no la mueve significativamente. El resto de los 67 meses supera 96,6 %; la mediana mensual es 99,96 %.

---

## 4. Inventario de huecos y clasificación causal

**4.093 huecos internos**, cada uno clasificado contra el calendario `CME_Equity` (`TDA02_huecos.csv`) — sin cambios en los conteos por causa respecto de la revisión 1 (la corrección de §2 afecta la grilla/cobertura, no la clasificación de los huecos ya observados; la única salvedad, ya presente desde la revisión 1, es que `SECONDARY_MAINTENANCE_BREAK` ahora se asigna con una fecha de corte explícita, §1.2.c, en vez de depender de que no aparezcan huecos coincidentes después de 2021-06-25):

| causa | n | ¿qué significa |
|---|---:|---|
| `UNKNOWN` | 2.543 | no coincide con ninguna estructura de calendario conocida — ver desglose en §4.1 |
| `DAILY_MAINTENANCE` | 1.076 | transición normal cierre→reapertura entre jornadas consecutivas |
| `SECONDARY_MAINTENANCE_BREAK` | 353 | break intradiario, `trading_date <= 2021-06-25` (eliminado por CME con efecto 2021-06-28, §1.2.c) |
| `WEEKEND` | 84 | cierre/reapertura de fin de semana |
| `EARLY_CLOSE` | 32 | cierre anticipado (31 verificados fecha-por-fecha + `2025-01-09`, ahora confirmado por CME SER 9499R) |
| `HOLIDAY` | 5 | el hueco cruza uno o más feriados completos de CME_Equity |
| `MISSING_TRADING_DAY` | 0 | ninguna jornada de negociación válida aparece con cero barras dentro de un archivo |

### 4.1 Los 2.543 huecos `UNKNOWN`, desglosados

El 69,9 % (1.777) son huecos **cortos** (≤ 5 minutos), y de ésos, el 99,4 % (1.766) tienen **ambos** bordes dentro de una sesión de negociación válida — genuinamente internos a un día de mercado abierto, no un problema de calendario ni de borde de archivo. Su distribución horaria se concentra en horas de menor liquidez (00–08h y 18–23h NY) pero aparece también, en menor número, en horario diurno de mayor liquidez (9h–16h NY: 71 casos). **Sobre qué significan exactamente estos 1.766 huecos — ver §10 (revisado): la evidencia es compatible con `AUSENTE` como representación de un minuto sin operaciones, pero no lo confirma por sí sola.**

Los 766 huecos `UNKNOWN` restantes incluyen dos hallazgos concretos y bien acotados en el tiempo:

**(a) Semana anómala del 18 al 22 de octubre de 2020.** Cuatro días de negociación consecutivos muestran el mismo patrón de dos huecos (`05:04`→`05:19` NY y `05:49`→`06:49` NY) que no aparece en ningún otro tramo del conjunto de investigación; el domingo previo también muestra una reapertura real anómala a las `06:49` NY. Localiza con precisión la pista que TDA-01 (§13, pregunta 4) dejó abierta, pero no se pudo confirmar su causa institucional — acceso a CME bloqueado. Queda `UNKNOWN`.

**(b) Semana de crisis de marzo de 2020.** Ver §6: huecos largos (hasta ~18 horas) sin causa de calendario identificable, concentrados en 2020-03-09, 16, 17 y 18 — la semana de mayor volatilidad histórica del período, con múltiples paradas de negociación por circuit-breaker de nivel de mercado ampliamente documentadas en la industria para esas fechas. Contexto públicamente conocido, no confirmado contra fuente primaria de CME en esta sesión. Queda `UNKNOWN`.

Ninguno de estos dos hallazgos afecta más de 5 fechas de negociación (de 1.435 totales).

---

## 5. Bordes de archivo (no son huecos)

`TDA02_bordes_de_archivo.csv`: **41** de los 44 bordes posibles (22 archivos × inicio/fin) no coinciden con el primer/último minuto ideal de su propia sesión — comportamiento esperado (cada archivo exporta la historia de UN contrato trimestral, que empieza y termina donde el snapshot lo decidió). **Ninguno de estos bordes se cuenta como hueco ni resta cobertura** (no afectado por la corrección de §2).

---

## 6. Días incompletos

`TDA02_dias_incompletos.csv`: 1.436 filas (archivo × `trading_date`). De los días que NO son de cierre anticipado, **131** tienen cobertura < 100 % (antes de la corrección de §2: 451 — la diferencia, 320 días, eran días cuya única "incompletitud" era el tramo del break secundario, ahora correctamente excluido de su propio conteo esperado) y **34** tienen cobertura < 95 % (sin cambio: ninguno de los días severamente incompletos está relacionado con el break). Los días más severamente incompletos:

| archivo | trading_date | esperados | presentes | cobertura |
|---|---|---:|---:|---:|
| `00_mnq_03_20.Last.txt` | 2020-03-18 | 1.365 | 273 | **20,00 %** |
| `20_mnq_03_25.Last.txt` | 2025-03-17 | 1.380 | 300 | **21,74 %** |
| `00_mnq_03_20.Last.txt` | 2020-03-16 | 1.365 | 606 | 44,40 % |
| `03_mnq_12_20.Last.txt` | 2020-10-23 | 1.365 | 695 | 50,92 % |
| `11_mnq_12_22.Last.txt` | 2022-11-07 | 1.380 | 747 | 54,13 % |

Las tres fechas de marzo de 2020 (`16`, `17`, `18`) son la semana de crisis ya citada en §4.1(b). `2025-03-17` está dominado por un único hueco de 1.014 minutos ligado al tramo previo a la apertura dominical (§8). Ninguna de estas fechas se compara contra el conteo de una jornada normal sin justificarlo.

---

## 7. Heatmap de completitud (único gráfico exigido)

`reports/mnq/TDA02_heatmap_completitud.png` — matriz booleana día (hora de Nueva York) × minuto del día, agregada sobre todos los archivos (construida directamente sobre las barras observadas; no depende de la grilla esperada, por lo que no cambia con la corrección de §2). Se observan con claridad: la banda estructural del corte de mantenimiento diario (`~17:00`–`~18:00` NY, constante en toda la historia); la banda del break secundario (`~16:15`–`~16:30` NY, presente sólo hasta mediados de 2021); y puntos/rayas esporádicas dispersas (más densas en 2020 y en la semana de octubre de 2020).

---

## 8. Barras fuera de la grilla esperada (revisado: ahora incluye el tramo del break)

`TDA02_barras_fuera_de_grilla.csv`: **1.602** timestamps observados que no caen en la grilla esperada (§2 redefine "fuera de grilla" de forma literal: cualquier bar que no pertenezca a la grilla, no sólo los que caen fuera de toda sesión — ver `find_out_of_grid_bars`). Se distinguen dos motivos:

| motivo | n | qué significa |
|---|---:|---|
| `NO_SESSION` | 1.513 | el timestamp no cae dentro de ninguna sesión del calendario |
| `SECONDARY_BREAK_WINDOW` | 89 | el timestamp SÍ cae en una sesión válida, pero dentro del tramo del break secundario que la grilla excluye para fechas `<= 2021-06-25` (§2) |

Los `NO_SESSION` (1.513) se concentran en dos patrones ya documentados:

1. **"Goteo" previo a la reapertura dominical** (904 de 1.513, el 60 %): barras aisladas, de un solo contrato de volumen y a un único precio (el cierre del viernes), esparcidas de forma no periódica entre la medianoche y las 18:00 NY del domingo. Ejemplo verificado (`21_mnq_06_25.Last.txt`, domingo 2025-05-25): 13 barras de volumen 1, todas al mismo precio, entre las `02:55` y las `15:11` NY, seguidas de la reapertura normal a las `18:01` NY. No se puede determinar si son operaciones genuinas de muy baja liquidez o un artefacto de la plataforma de origen. `INDETERMINADO`.
2. **~15 minutos "extra" en fechas de cierre anticipado post-feriado** (§1.2.e): imprecisión menor y acotada del calendario, sin efecto sobre la cobertura.

Los `SECONDARY_BREAK_WINDOW` (89) son, en su mayoría, un único trade aislado justo al minuto siguiente de `break_start` — el break no fue perfectamente limpio en el 25 % de las 353 sesiones donde aplicó. Se registran aquí, no se descartan ni se cuentan como cobertura perdida ni ganada.

---

## 9. Verificación de DST

Sin cambios respecto de la revisión 1 (no afectado por la corrección de §2: ninguna transición DST coincide con el tramo del break secundario). `TDA02_dst_evidencia.csv`: **11 transiciones** dentro del rango de investigación, todas verificadas correctas: offset UTC cambia exactamente 1 hora, hora local de cierre fija en `17:00:00` NY, 1.380 minutos esperados por sesión sin excepción, hueco de mantenimiento con firma exacta `17:00:00`→`61 min` en las 11 transiciones. Manejo de DST **verificado correcto** en todo el rango de investigación.

---

## 10. Investigación de barras inactivas (TH04) — **revisado**

**Hecho previo (TDA-00/TDA-01), re-confirmado aquí de forma independiente:** 0 filas con `volume == 0` en las 1.937.230 filas del conjunto de investigación (`analyze_inactive_bar_candidates`: `zero_volume_count = 0`).

### 10.1 Representación de un minuto de mercado ABIERTO sin operaciones — **permanece `INDETERMINADO`**

> **Corrección respecto de la revisión 1.** La versión anterior de este documento concluía "AUSENTE confirmado" a partir de 1.766 huecos internos cortos (1–5 minutos) con ambos bordes dentro de una sesión válida. Esa conclusión iba más allá de lo que la evidencia permite. Con OHLCV de 1 minuto **no puede distinguirse**, para un hueco corto intra-sesión:
>
> - (A) que realmente no hubo ninguna operación ejecutada en ese minuto concreto, o
> - (B) que la barra está ausente por una pérdida o falta de captura del dato histórico (un problema del proveedor/plataforma, no del mercado).
>
> No se encontró, en esta revisión, documentación oficial de NinjaTrader (o de otro proveedor relevante) que establezca explícitamente cómo se representa un minuto de mercado abierto sin trades — TDA-01 (§10, D1–D4) ya agotó la documentación de NinjaTrader disponible sobre la convención de barra y sobre `Last` como tipo de dato, sin encontrar esa afirmación específica; no hay evidencia nueva en esta revisión que la resuelva.

**Lo que SÍ queda confirmado, con evidencia directa:**

- **0 filas con `volume == 0`** en todo el conjunto de investigación (TDA-00, re-confirmado aquí).
- **Existen 1.766 huecos internos cortos (1–5 minutos) con ambos bordes dentro de una sesión de negociación válida** — ni fin de semana, ni feriado, ni corte de mantenimiento, ni break secundario. Es un hecho estructural del dato, verificado y reproducible (`TDA02_huecos.csv`, `cause=UNKNOWN`, `gap_minutes<=5`, `before_date`/`after_date` ambos no nulos).
- **No hay evidencia de `FORWARD_FILL`** que compita como explicación alternativa para esos 1.766 huecos (§10.2): si esos minutos estuvieran rellenados hacia adelante en vez de ausentes, aparecerían como filas con `O=H=L=C` y volumen fijo/cero en el lugar del hueco — no aparece ninguna fila así (el hueco es, precisamente, la ausencia de una fila, no la presencia de una rellenada).

**Lo que queda `INDETERMINADO`:** si esos 1.766 huecos representan minutos genuinamente sin operaciones (compatible con la hipótesis `AUSENTE`) o si una fracción de ellos es pérdida de dato histórico no atribuible al mercado. La evidencia disponible es **compatible con** `AUSENTE`, pero no la **confirma** — se necesitaría una fuente documental explícita del proveedor (no encontrada) o una segunda fuente de datos independiente para el mismo período (fuera del alcance de esta etapa) para cerrar la pregunta. **Distinción explícita, tal como exige esta corrección:** "barra faltante" (el hecho observado: no hay fila en ese minuto) es un hallazgo `CONFIRMADO`; "minuto confirmado sin trades" (la interpretación causal de por qué falta) es `INDETERMINADO`. No se equiparan.

### 10.2 Candidatos a `FORWARD_FILL`

Sin cambios respecto de la revisión 1 (no afectado por la corrección de TH04: la pregunta de forward-fill es distinta de la pregunta de representación de minutos sin trades, y su evidencia no cambió). Metodología (`O=H=L=C` sólo NO es prueba suficiente): **5.179 barras planas** (0,27 % de las filas), agrupadas en rachas; 96,7 % de longitud 1 (evento aislado, volumen ≥ 1). Racha más larga: 14 minutos consecutivos. Las 12 rachas de longitud ≥ 6 (`TDA02_rachas_planas_largas.csv`) se concentran todas en la semana de crisis de marzo de 2020, con volumen que **varía** minuto a minuto dentro de cada racha (1 a 19 contratos) — evidencia de actividad real repetida al mismo precio (compatible con un bloqueo de precio por circuit-breaker/límite), no de un valor sintético copiado.

**Veredicto: no hay evidencia de `FORWARD_FILL` en el conjunto de investigación** (resultado negativo, G6). Las candidatas aisladas se clasifican `FLAT_AISLADA`. Las 12 rachas largas quedan `INDETERMINADO` respecto de forward-fill específicamente, con la evidencia de volumen variable pesando en contra de esa hipótesis, no a favor.

La máscara persistente `barra_inactiva` (`data/interim/mnq/tda02_barra_inactiva_mask.parquet`) clasifica cada minuto esperado de cada archivo en `AUSENTE` (fila no observada — término puramente descriptivo de presencia/ausencia de fila, sin implicar una interpretación causal sobre por qué falta, ver §10.1), `VOLUMEN_CERO` (vacío), `CANDIDATO_FORWARD_FILL` (las 12 rachas largas), `FLAT_AISLADA` o `ACTIVA`.

---

## 11. Protección del hold-out

`run_tda02_analysis` reutiliza la misma protección que TDA-00/TDA-01 (`holdout_guard.py`). Ningún archivo de `holdout_files` fue abierto — verificado con tests dedicados. Ningún hecho ya conocido del hold-out se usó para definir ninguna regla de esta etapa.

---

## 12. STOP-2

Criterio del roadmap: si la sesión real derivada de los datos difiere **sustancialmente** de la sesión/calendario esperado, detenerse sin "arreglar" el calendario para que encaje.

Verificación (`check_stop2`, umbral 90 %): de los **1.550 huecos "de calendario"** (`DAILY_MAINTENANCE` + `WEEKEND` + `HOLIDAY` + `EARLY_CLOSE`), el **99,67 %** coincide EXACTAMENTE con el horario que declara `CME_Equity` — sin cambios respecto de la revisión 1 (la corrección de §2 afecta la grilla/cobertura, no la clasificación de huecos que alimenta esta verificación).

**`STOP-2` NO se activa.** Las dos desviaciones encontradas (§4.1.a, §4.1.b) son incidentes puntuales y bien acotados (9 fechas de negociación en total, de 1.435), no una discrepancia sistemática de la definición de sesión. Las dos correcciones aplicadas al calendario (break secundario, `2025-01-09`) están ahora respaldadas por cita documental primaria además de evidencia forense — ver §1.2(c)-(d) —, no son "correcciones silenciosas de TDA-01": TDA-01 no se modifica.

---

## 13. Preguntas que permanecen `INDETERMINADAS`

1. **Representación de un minuto de mercado abierto sin operaciones (§10.1).** Compatible con `AUSENTE`, no confirmada: no puede distinguirse de una pérdida de dato histórico con OHLCV de 1 minuto únicamente, y no se encontró documentación explícita del proveedor sobre este caso concreto.
2. **Semana anómala del 18-22 de octubre de 2020 (§4.1.a).** Patrón preciso y repetido, causa institucional desconocida.
3. **Huecos largos sin explicar de la semana de crisis de marzo de 2020 (§4.1.b, §6).** Contexto histórico plausible (circuit-breakers de nivel de mercado), no confirmado contra fuente primaria en esta sesión.
4. **"Goteo" de barras aisladas antes de la reapertura dominical (§8).** 904 barras de volumen 1 al precio de cierre previo, esparcidas en horario nominalmente cerrado — no se pudo determinar si son operaciones genuinas o un artefacto de la plataforma de origen.
5. **Imprecisión de ~15 minutos del calendario en cierres anticipados post-feriado (§1.2.e).** Pequeña, documentada, no corregida (no afecta cobertura).
6. **Las 12 rachas largas de barras planas de marzo de 2020 (§10.2).** `FORWARD_FILL` no puede confirmarse ni descartarse con certeza absoluta usando sólo OHLCV Last.

Ítems ya **cerrados** en esta revisión (no permanecen abiertos): la causa institucional del break secundario y su fecha exacta de eliminación (§1.2.c, confirmado por CME SER/Globex Notice + evidencia forense convergente); la naturaleza del cierre de `2025-01-09` (§1.2.d, confirmado por CME SER 9499R + evidencia forense convergente al minuto).

Ninguna de las preguntas restantes es bloqueante para TDA-03: todas están acotadas en el tiempo o en alcance, documentadas con su evidencia completa, y no comprometen la validez general del calendario usado para la grilla (§12, STOP-2 no activado).

---

## Estado final

**`PASS_WITH_OPEN_QUESTIONS`**

La grilla esperada se construyó sobre la estructura nativa de negociación de CME_Equity y se corrigió en dos puntos verificables, ambos ahora respaldados por cita documental primaria además de evidencia forense propia (§1.2.c-d): la eliminación del break secundario con efecto 2021-06-28, y el cierre abreviado (no completo) del 2025-01-09. La grilla esperada excluye correctamente el tramo del break secundario para las fechas en las que estuvo vigente (§2), sin perder de vista ninguna barra real observada en ese tramo (§2, §8). La cobertura global corregida es **99,39 %**. Los 31 cierres anticipados candidatos de TDA-01 quedan verificados fecha por fecha (§1.2.b). No hay evidencia de `FORWARD_FILL` (resultado negativo, G6). `STOP-2` no se activa. La pregunta sobre la representación de un minuto de mercado abierto sin operaciones **permanece `INDETERMINADA`** (§10.1, corregido respecto de la revisión 1: la evidencia es compatible con `AUSENTE` pero no la confirma). Quedan, además, varias preguntas secundarias legítimamente abiertas (§13), acotadas y documentadas, que se trasladan a TDA-03 y etapas posteriores sin resolverse por conveniencia.

**No se avanza a TDA-03.**
