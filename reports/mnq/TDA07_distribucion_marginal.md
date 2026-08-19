# TDA-07 — Distribución marginal y por segmento

**Etapa:** `docs/methodology/Tsay_OHLCV_analysis_roadmap.md` § TDA-07
**Depende de:** TDA-04 (`PASS_WITH_OPEN_QUESTIONS`), TDA-05 (`PASS_WITH_OPEN_QUESTIONS`), TDA-06 (`PASS_WITH_OPEN_QUESTIONS`, STOP-6 no activado)
**Alcance de datos:** exclusivamente `tda04_variables_1m.parquet`, `tda04_return_validity_mask.parquet`, `tda06_r_tilde.parquet` y `TDA06_segmentacion_propuesta.csv`. Ningún archivo de `data/raw/` ni de `holdout_files` fue abierto en esta etapa.
**Evidencia reproducible:** `reports/mnq/TDA07_th08_contrafactual_global.csv`, `TDA07_th08_contrafactual_por_causa.csv`, `TDA07_momentos_cuantiles_drift_colas.csv`, `TDA07_qq_global.png`, `TDA07_qq_por_segmento.png`, `TDA07_qq_th08_contrafactual.png`. Todo generado por `python -m ohlcv_dataroad.ingest.run_tda07`.

> Esta etapa caracteriza la distribución de `r_1m` (y `r_tilde`, RETROSPECTIVO) — momentos, cuantiles, drift, asimetría de colas — global, por año y por la segmentación empírica de TDA-06, y cierra el componente distribucional que TDA-04 dejó explícitamente pendiente de TH08. NO calcula ACF/PACF, NO estudia volatility clustering, NO ajusta ARCH/GARCH, NO ejecuta EVT, NO retoma TH10, NO investiga las ventanas de apertura/cierre de TDA-06 §7 como señal (eso es TDA-08), y NO modifica ningún artefacto de TDA-00…TDA-06.

---

## Corrección puntual de cierre (posterior al cierre inicial)

Se detectó y corrigió un problema concreto en el cálculo de HAC (Newey-West) de TH12; ningún otro resultado de la etapa (TH08, TH11, TH13, `r_tilde`) se vio afectado.

**El HAC original podía cruzar discontinuidades temporales.** `hac_mean_se` recibía el array de valores ya *compactado* (filtrado a `r_1m_valid=True`, y además restringido por año/segmento en `analyze_group`) y calculaba la autocovarianza de rezago `j` como `mean(e[j:]*e[:-j])`, asumiendo implícitamente que la posición `t` y la posición `t-j` del array representaban observaciones separadas por exactamente `j` minutos de reloj. Eso es falso en general: dos filas `r_1m_valid=True` consecutivas en el array compactado pueden estar separadas, en el reloj real, por una fila que TDA-04 invalidó (`NON_CONSECUTIVE_MINUTE`, `TRADING_DATE_BOUNDARY`, `ROLL_BOUNDARY`) o, en el análisis **por segmento**, por horas o jornadas completas (el mismo tramo horario de dos días distintos queda "adyacente" en el array tras filtrar por `segment_label`, aunque el reloj real los separe ~24h). El HAC original podía así fabricar dependencia temporal artificial entre observaciones que TDA-04 decidió explícitamente no conectar — exactamente la misma regla de no-cruce que TH08 cierra en esta etapa.

**Corrección**: se construyen explícitamente `block_ids` (`compute_hac_block_ids`) a partir de `timestamp`/`trading_date` de la población analizada (global, un año o un segmento) — dos observaciones consecutivas por timestamp quedan en el mismo bloque si y solo si están separadas por exactamente 60 segundos **y** comparten `trading_date` — la misma condición que TDA-04 exige para `r_1m_valid=True`. No hace falta comprobar el roll por separado: TDA-04 (informe, §5) certificó que todo roll ocurre en una frontera de `trading_date` (nunca a mitad de una), así que exigir `trading_date` idéntico excluye automáticamente cualquier roll. `hac_mean_se` solo permite que el par `(t, t-j)` contribuya a la autocovarianza de rezago `j` si `block_ids[t] == block_ids[t-j]` — como los ids son monótonos no decrecientes y constantes dentro de cada tramo continuo, esa igualdad garantiza que **todas** las posiciones intermedias también pertenecen al mismo bloque, no solo los dos extremos del par. Si ningún par a un rezago dado cae dentro del mismo bloque, ese término se omite (equivalente a asumir esa autocovarianza como 0, una aproximación conservadora documentada, nunca una fabricación de dato). La media observada **no cambió** — solo el cálculo de su incertidumbre HAC, y por tanto sus intervalos.

**Resultado**: el efecto numérico fue pequeño (variaciones de `hac_se` del orden de ±0,2% a ±1,6% según el grupo) porque las discontinuidades excluidas por TDA-04 son una fracción muy pequeña del total (0,18% de las filas) — pero la corrección es necesaria por principio, y en el análisis por segmento en particular estaba fabricando dependencia cruzada entre jornadas distintas del mismo tramo horario, algo que nunca debió ocurrir. **Ninguna de las conclusiones cualitativas de TH12 cambió** (ver §4/§7 corregidas): el intervalo HAC global sigue excluyendo cero por un margen mínimo, siguen siendo exactamente 2 de 7 años (2020, 2023) y 1 de 7 segmentos (`00:00-02:00`) los que excluyen cero, y la magnitud sigue siendo una fracción diminuta de un tick en todos los casos.

**No cambiaron**: TH08 (cierre distribucional, no depende de HAC), TH11 (momentos/cuantiles, no depende de HAC), TH13 (asimetría de colas, usa bootstrap de bloques por jornada, no HAC), ni `r_tilde`/`s(m)` (artefactos de TDA-06, no tocados). Se añadió, además, una invariante bloqueante nueva sobre `tda06_r_tilde.parquet` (alineación de timestamps con TDA-04 + etiqueta `RETROSPECTIVO` en toda fila) — ver §1.

### Segunda corrección puntual: normalización de la autocovarianza HAC

Tras la corrección de continuidad anterior, `gamma_j` se calculaba como el **promedio** de los productos `e[t]*e[t-j]` sobre los pares que sobrevivían al filtro de bloque (`mean(...[same_block])`, es decir, dividido por `same_block.sum()`). Esto excluye correctamente los pares cross-block de la SUMA, pero al normalizar por el número de pares *sobrevivientes* en vez de por la muestra completa, sobrepondera los pares que sí quedan — los pares excluidos deben aportar **cero** al numerador, no hacer que el denominador se encoja y los demás pesen más.

**Derivación**: para una regresión sobre una constante, el sandwich de Newey-West (Ec. 2.50 de Tsay) se reduce a `Var_HAC(mean) = (1/T²)·[Σₜeₜ² + 2·Σⱼ wⱼ·SumProdⱼ]`, con `SumProdⱼ` la SUMA cruda (sin normalizar) de los productos `eₜ·eₜ₋ⱼ`. Asumiendo bloques mutuamente independientes (la misma aproximación conservadora que motiva excluir esos pares), `SumProdⱼ` se reduce exactamente a la suma de los productos de pares `same_block` — los pares cross-block aportan cero al numerador, y el denominador correcto es **T** (la muestra completa), nunca el conteo de pares sobrevivientes. Con `l≪T` (bandwidth de decenas, `T` del orden de 10⁵-10⁶), la diferencia entre normalizar por `T` o por `T−j` es numéricamente despreciable — la corrección importante es no normalizar por `same_block.sum()`.

**Corrección**: `gamma_j = sum((e[j:]*e[:-j])[same_block]) / T` (suma, no promedio; dividida por `T`, no por el número de pares sobrevivientes).

**Resultado sobre el conjunto de investigación real**: el efecto numérico fue, de nuevo, muy pequeño (`hac_se` cambió en la 5ª-6ª cifra significativa, tanto GLOBAL como por año/segmento) porque las discontinuidades excluidas son una fracción pequeña del total en todos los grupos analizados — pero la corrección es matemáticamente necesaria, no cosmética. **Ninguna conclusión de TH12 cambió**: siguen siendo exactamente los mismos 2 de 7 años y 1 de 7 segmentos que excluyen cero (ver §4/§7, valores actualizados a la versión final).

Test añadido (`test_hac_mean_se_normalizes_by_full_sample_not_by_surviving_pair_count`): con 3 bloques cortos de niveles muy distintos (1-3, 10-12, 100-102), calcula de forma independiente tanto la referencia correcta (suma de pares `same_block` ÷ T) como la incorrecta (÷ `same_block.sum()`), y comprueba que `hac_mean_se` coincide exactamente con la primera y difiere de la segunda.

**Aclaración documental**: cuando este informe dice que el drift por año/segmento es "por debajo de X ticks", se refiere siempre a la **media puntual** (`mean_ticks`), no al intervalo HAC completo — el extremo superior del intervalo HAC supera 0,1 ticks en varios años/segmentos (p. ej. 2025: hasta 0,240 ticks; `08:30-09:30`: hasta 0,169 ticks), reflejando la incertidumbre de la estimación, no la magnitud del efecto puntual.

---

## 1. Invariantes bloqueantes — verificación previa

Antes de producir cualquier resultado distribucional, TDA-07 ejecuta tres invariantes bloqueantes (fail-fast; si cualquiera falla, la etapa se detiene sin escribir artefactos):

1. **Alineación exacta de timestamps** entre `tda04_variables_1m.parquet` y `tda04_return_validity_mask.parquet` — comparación elemento a elemento tras ordenar ambas tablas por `timestamp`, nunca asumida por orden de fila.
2. **Coincidencia numérica** entre `r_naive_1m` (contrafactual, `shift(1)` incondicional sobre `close`) y `r_1m` de TDA-04 en toda fila `invalid_reason == "VALID"`, dentro de una tolerancia explícita `1e-9` (ambas series usan la misma fórmula sobre la misma columna `close`; la tolerancia solo absorbe orden de operaciones de punto flotante, no un error de método).
3. **(Corrección puntual)** `tda06_r_tilde.parquet` alineado exactamente (mismos timestamps, mismo número de filas) con `tda04_variables_1m.parquet`, y `label == "RETROSPECTIVO"` en absolutamente todas las filas — verificado justo antes de construir la población de análisis de `r_tilde`, para no tratar una cantidad sin etiquetar como si fuera retrospectiva (G1).

**Resultado sobre el conjunto de investigación real: las tres invariantes se cumplen.** 1.918.050 filas alineadas exactamente entre `tda04_variables_1m.parquet` y `tda04_return_validity_mask.parquet`; las 1.914.530 filas `VALID` de `r_naive_1m` coinciden con `r_1m` de TDA-04 dentro de tolerancia; `tda06_r_tilde.parquet` tiene las mismas 1.918.050 filas, mismos timestamps, y `label == "RETROSPECTIVO"` en el 100% de ellas. Ninguna contradicción detectada — la etapa procedió a producir resultados.

## 2. Cierre de TH08 — componente distribucional

**Comparación PRINCIPAL** (`TDA07_th08_contrafactual_global.csv`): A) `r_1m` válido (TDA-04, reglas de no-cruce aplicadas) vs B) `r_naive_1m` global completo (`shift(1)` incondicional, excluye únicamente `FIRST_OBSERVATION` por construcción — no existe `t-1`).

| | A) r_1m válido | B) r_naive_1m global |
|---|---:|---:|
| n | 1.914.530 | 1.918.049 |
| mean | 6,21×10⁻⁷ | 4,79×10⁻⁷ |
| std | 0,000419 | 0,000438 (+4,7%) |
| skewness | −0,209 | **−8,823** |
| kurtosis_excess (cruda) | 111,6 | **1.790,9** (16×) |
| kurtosis_excess (recortada 0,1%) | 11,49 | 11,84 |
| min | −0,0310 | **−0,1007** |
| max | 0,0256 | 0,0256 |

**Conclusión**: las reglas de no-cruce de TDA-04 protegen la distribución de una contaminación severa. La curtosis cruda se dispara 16× y la asimetría se vuelve extremadamente negativa cuando se permite que el retorno cruce fronteras de roll/jornada/hueco — pero la curtosis **recortada** (0,1% más extremo) es casi idéntica entre A y B (11,49 vs 11,84). Esto confirma que la contaminación está concentrada casi enteramente en un puñado de observaciones (las 3.519 filas no-VALID de B, el 0,18% de la muestra) — exactamente lo que TDA-04 excluyó por diseño, no un efecto difuso sobre toda la distribución. El mínimo cae de −3,1% (A) a **−10,1%** (B) — un movimiento de un solo salto de roll.

**Diagnóstico SECUNDARIO por causa** (`TDA07_th08_contrafactual_por_causa.csv`), subordinado al resultado principal:

| Causa | n | mean | std | skewness | kurtosis_excess | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| ROLL_BOUNDARY | 21 | −0,000972 | 0,0258 (~62× normal) | −2,89 | 8,85 | −0,1007 | 0,0206 |
| TRADING_DATE_BOUNDARY | 1.398 | −0,000074 | 0,00287 (~6,9× normal) | −5,63 | 76,84 | −0,0460 | 0,0175 |
| NON_CONSECUTIVE_MINUTE | 2.100 | −0,000070 | 0,00173 (~4,1× normal) | −17,86 | **436,9** | −0,0454 | 0,0128 |

Los tres subconjuntos son, cada uno, muchísimo más dispersos y asimétricos que la población válida — consistente con su naturaleza (saltos de contrato, gaps overnight/fin de semana, huecos internos). Con `n` tan pequeño (21 a 2.100), los momentos de cada subconjunto son poco fiables individualmente (G5: se reportan con `n` visible, no como parámetros estables) — el resultado que sí es robusto es el de la comparación principal A vs B.

**TH08 queda RESUELTA**: la comparación de momentos y cuantiles con y sin las reglas de no-cruce, pendiente desde TDA-04, está completa. Las reglas de no-cruce eran necesarias y su ausencia habría introducido una contaminación de cola severa y concentrada.

Nota sobre `TDA07_qq_th08_contrafactual.png`: los dos paneles (A y B) se ven casi idénticos a simple vista — esto es CONSISTENTE con el resultado, no una contradicción. El QQ-plot compara una grilla fija de percentiles (el más profundo, p=0,1%, ≈ el rango 1.918 de 1.918.049), mientras que la contaminación de B son unos pocos puntos EXTREMOS dentro de ese 0,1% más profundo (min −10,1% vs −3,1%) — invisibles para una comparación por percentiles fijos, pero dominantes para un momento de cuarto orden (curtosis), que pondera cada desviación a la cuarta potencia. Es la misma razón por la que la curtosis recortada apenas cambia entre A y B: los cuantiles (y el QQ) son robustos a un puñado de puntos; los momentos crudos, no.

## 3. TH11 — Distribución marginal (momentos y cuantiles)

**Global** (`TDA07_momentos_cuantiles_drift_colas.csv`, scope=GLOBAL):

| | r_1m (crudo) | r_tilde (ajustado, RETROSPECTIVO) |
|---|---:|---:|
| n | 1.914.530 | 1.914.530 |
| mean | 6,21×10⁻⁷ | 6,72×10⁻⁷ |
| std | 0,000419 | 0,000417 |
| skewness | −0,209 | −0,191 |
| kurtosis_excess (cruda) | 111,6 | 78,3 |
| kurtosis_excess (recortada 0,1%) | 11,49 | 11,36 |
| q0,001 / q0,999 | −0,002747 / 0,002786 | −0,002775 / 0,002762 |
| q0,01 / q0,99 | −0,001231 / 0,001194 | −0,001135 / 0,001119 |

La curtosis cruda es enorme (111,6) — MNQ a 1 minuto está muy lejos de la normalidad, dominado por eventos extremos (consistente con Tsay C1: exceso de curtosis alto y generalizado en series financieras de alta frecuencia). La curtosis **recortada** (11,49) sigue siendo alta pero mucho menor: una fracción sustancial de la curtosis cruda depende de un puñado de observaciones extremas — un resultado esperado a esta granularidad, no un artefacto (no se ejecuta ningún test de normalidad como hallazgo, G5; ver `TDA07_qq_global.png` para la forma completa de la desviación).

**Por año y por segmento**: ver §7 — ninguna de estas cifras es estable entre subperíodos, así que la marginal GLOBAL no describe ningún estado concreto del mercado (roadmap TDA-07, "criterios de interpretación").

## 4. TH12 — Drift (media, HAC + bootstrap)

*(Recalculado dos veces tras el cierre inicial — ver §"Corrección puntual de cierre": primero por continuidad temporal, después por normalización de la autocovarianza. La media no cambió en ninguna de las dos; los intervalos HAC cambiaron en un margen pequeño en ambas (±0,2-1,6% y, en la segunda, variaciones en la 5ª-6ª cifra significativa). Ninguna conclusión cualitativa cambió en ninguna de las dos correcciones.)*

**Global**, `r_1m`: media puntual = 6,21×10⁻⁷ (≈0,036 ticks, sin cambios en ninguna corrección). Intervalo HAC 95% (Newey-West, bandwidth `l=35` sobre T=1.914.530, respetando bloques de continuidad temporal y normalizado por T): **[2,84×10⁻⁸, 1,213×10⁻⁶]** ≈ [0,0016, 0,0698] ticks. Intervalo por bootstrap de bloques de jornada (300 remuestreos, sin cambios en ninguna corrección — no depende de HAC): **[3,65×10⁻⁹, 1,31×10⁻⁶]**.

Ambos intervalos siguen excluyendo cero, por un margen mínimo, y la magnitud de la **media puntual** (0,036 ticks) sigue siendo una fracción diminuta de un solo tick. Esto es exactamente el patrón que el roadmap anticipa (Tsay C1/C5: con `n~10⁶`, un drift "significativo" puede tener una magnitud muy inferior al tick) — **el resultado principal es la magnitud de la media, no la exclusión de cero**: un drift de este tamaño no es operativamente relevante y no se interpreta como señal. El **intervalo** HAC completo, en cambio, sí llega hasta ≈0,070 ticks en su extremo superior — una magnitud aún pequeña, pero mayor que la media puntual (ver aclaración de unidades más abajo).

`r_tilde` (ajustado): media puntual = 6,72×10⁻⁷ (sin cambios), HAC 95% CI [8,70×10⁻⁸, 1,256×10⁻⁶] — mismo orden de magnitud, mismo `hac_l=35`. No se traduce a ticks (ver §6).

**Por año** (`r_1m`, scope=YEAR): el intervalo HAC sigue excluyendo cero en exactamente los mismos **2 de 7 años**: 2020 (CI [2,46×10⁻⁷, 4,04×10⁻⁶] ≈ [0,010, 0,167] ticks) y 2023 (CI [2,20×10⁻⁷, 2,21×10⁻⁶] ≈ [0,013, 0,131] ticks) — ambos positivos. En 2019, 2021, 2022, 2024 y 2025 el intervalo **incluye cero**. Es decir: **el drift NO es estable entre años** (G3) — ni siquiera su signo se sostiene de forma consistente, y solo 2 de 7 años excluyen cero. La **media puntual** está siempre por debajo de 0,09 ticks en valor absoluto en los 7 años (máximo: 2020, 0,089 ticks) — pero el **extremo superior del intervalo HAC** supera 0,1 ticks en varios años (2020: 0,167; 2023: 0,131; 2019: 0,129; 2025: 0,240), reflejando la incertidumbre de la estimación en años con `n` menor, no una media puntual mayor. No se cita ningún drift como propiedad del dataset.

## 5. TH13 — Asimetría de colas

Se sustituyó, deliberadamente, la "curtosis por lado" que sugiere literalmente el texto del roadmap por dos medidas directamente interpretables y sin ambigüedad de definición (instrucción explícita de esta tarea): (a) diferencia entre cuantiles simétricos en magnitud, `q_{1-p} − |q_p|`, en unidades de retorno; y (b) frecuencia de excedencias por lado a un umbral simétrico fijo (percentil 99 de `|x|` del propio grupo), una proporción ya interpretable por sí misma.

**Global**, `r_1m`:

| Nivel | diff = q_(1-p) − \|q_p\| | IC 95% (bootstrap bloques) | ¿Excluye cero? |
|---|---:|---|---|
| p=0,01 | −0,000037 | [−0,000045, −0,000027] | Sí (negativo) |
| p=0,001 | +0,000039 | [−0,000026, +0,000115] | No |

| | freq_left | freq_right | freq_diff | IC 95% |
|---|---:|---:|---:|---|
| umbral = p99(\|r_1m\|) = 0,001590 | 0,005124 | 0,004876 | −0,000248 | [−0,000380, −0,000101] (excluye cero) |

A profundidad p=0,01 (con mucha muestra detrás), la cola izquierda es sistemáticamente algo más extrema que la derecha (diferencia estable, CI que excluye cero) — consistente con la frecuencia de excedencias, también estable y negativa: las barras extremas negativas ocurren ligeramente más seguido que las positivas (0,5124% vs 0,4876%, una diferencia de 0,0248 puntos porcentuales). A profundidad p=0,001 (mucha menos muestra por cuantil), el intervalo ya no excluye cero — la evidencia más profunda es demasiado ruidosa para sostener la misma conclusión. Ambos resultados son pequeños en magnitud (fracciones de un tick, unas pocas centésimas de punto porcentual de frecuencia) — se reportan como propiedad caracterizada, no como señal explotable.

**Por segmento** (§7): el signo de `freq_diff` **no es uniforme** — positivo (cola derecha más frecuente) en 00:00-02:00, 02:00-03:00, 08:30-09:30 y 09:30-16:02; negativo (cola izquierda más frecuente) en 03:00-08:30, 16:02-20:00 y, más marcadamente, 20:00-24:00 (−0,000552, el doble que el agregado global). La asimetría global negativa está impulsada principalmente por los segmentos de la tarde/noche NY (16:02-24:00), no por un efecto uniforme del día — un resultado que la marginal agregada, sin desagregar, habría ocultado.

## 6. Diferencias r_1m vs r_tilde

`r_tilde` (RETROSPECTIVO, TDA-06 §11) reduce la curtosis cruda de 111,6 a 78,3 (−30%) sin cambiar materialmente la escala (`std` 0,000419 vs 0,000417) ni los cuantiles centrales — consistente con que gran parte de la curtosis cruda de `r_1m` proviene de la mezcla de regímenes de volatilidad intradía que `s(m)` ya captura (TDA-06). La curtosis **recortada** apenas cambia (11,49 vs 11,36): el ajuste estacional no toca la parte de la curtosis que depende de un puñado de eventos extremos genuinos, solo la que provenía de la heterocedasticidad determinista del reloj.

`r_tilde` **no se traduce a ticks**: es una cantidad des-estacionalizada (dividida por `s(m)`, un factor RETROSPECTIVO cercano a 1 pero variable), no un retorno de precio literal — "cuántos ticks representa un `r_tilde`" no es una pregunta bien definida, y esta etapa se negó explícitamente a inventar esa convención (misma razón por la que se evitó la "curtosis por lado" en TH13, §5).

## 7. Resultados globales, por año y por segmento

**Estabilidad por año** (`r_1m`, `scope=YEAR`): la curtosis cruda oscila entre 24,0 (2021) y 138,8 (2025); la recortada, mucho más establemente, entre 6,9 (2022) y 13,1 (2025). El **signo de la asimetría cambia entre años**: positiva en 2019 (+0,92), 2020 (+0,35) y 2025 (+0,87); negativa en 2021 (−0,26), 2022 (−1,22, la más negativa), 2023 (−0,18) y 2024 (−0,32). Ninguna cifra de asimetría o curtosis cruda se cita como propiedad estable del dataset (G3) — la curtosis recortada es la única cifra razonablemente estable entre años.

**Por segmento** (partición empírica de TDA-06, §9 de su informe — 7 tramos: `00:00-02:00`, `02:00-03:00`, `03:00-08:30`, `08:30-09:30`, `09:30-16:02`, `16:02-20:00`, `20:00-24:00`): la curtosis cruda es más extrema en `08:30-09:30` (485,2 — probablemente dominada por eventos puntuales de datos macro) y en `16:02-20:00` (172,4); la recortada es más alta en `16:02-20:00` (22,2) y `00:00-02:00` (16,6), más baja en `09:30-16:02` (6,3, el tramo con más muestra). El drift (HAC 95% CI, recalculado con las dos correcciones puntuales — ver §"Corrección puntual de cierre") sigue excluyendo cero en exactamente el mismo **1 de 7 segmentos**: `00:00-02:00` (media puntual ≈0,072 ticks, CI ≈[0,009, 0,135] ticks) — el resto de los tramos incluyen cero. Ningún tramo horario muestra un drift operativamente relevante en su media puntual (todas por debajo de 0,1 ticks en valor absoluto).

La tabla completa (`TDA07_momentos_cuantiles_drift_colas.csv`) tiene una fila por combinación `serie × {GLOBAL, cada año, cada segmento}` — 2 series × 15 alcances = 30 filas, con las 9 columnas de cuantiles, momentos, drift (HAC + bootstrap, crudo y en ticks donde aplica) y asimetría de colas (diferencia de cuantiles + frecuencia de excedencias, ambas con IC) por fila.

## 8. Archivos creados/modificados

**Código nuevo:**
- `src/ohlcv_dataroad/ingest/tda07_marginal_distribution.py`
- `src/ohlcv_dataroad/ingest/run_tda07.py`
- `tests/test_tda07_marginal_distribution.py` (56 tests: 42 del cierre inicial + 13 de la 1ª corrección puntual (continuidad) + 1 de la 2ª (normalización))

**Código modificado:**
- `src/ohlcv_dataroad/config.py` (campos y propiedades `tda07_*`, sección añadida sin tocar las existentes)
- `configs/mnq_snapshot.yaml` (sección `tda07`, añadida)
- `src/ohlcv_dataroad/ingest/README.md` (listado de archivos + secciones nuevas para `tda07_marginal_distribution.py`/`run_tda07.py`)

**Correcciones puntuales de cierre (posteriores al cierre inicial), dos**: (1) `hac_mean_se`/`compute_hac_block_ids` (nueva, HAC respeta bloques de continuidad temporal) y `verify_r_tilde_invariants`/`RTildeInvariantError` (nueva, invariante bloqueante #3), más `analyze_group` (construye y pasa `block_ids`, ordena por `timestamp` defensivamente) y `run_tda07.py` (captura `RTildeInvariantError`); (2) `hac_mean_se` — la normalización de `gamma_j` cambia de `mean(...[same_block])` (dividir por el número de pares sobrevivientes) a `sum(...[same_block]) / T` (dividir por la muestra completa) — todo en `tda07_marginal_distribution.py`. Ningún archivo nuevo, ninguna sección de configuración nueva en ninguna de las dos. Se regeneraron únicamente los artefactos afectados (`TDA07_momentos_cuantiles_drift_colas.csv` y este informe, dos veces); `TDA07_th08_contrafactual_global.csv`, `TDA07_th08_contrafactual_por_causa.csv` y los tres PNG de QQ no cambiaron en ninguna corrección (verificado: TH08/TH11/TH13 no dependen de HAC).

**Artefactos generados:**
- `reports/mnq/TDA07_distribucion_marginal.md` (este informe)
- `reports/mnq/TDA07_th08_contrafactual_global.csv`
- `reports/mnq/TDA07_th08_contrafactual_por_causa.csv`
- `reports/mnq/TDA07_momentos_cuantiles_drift_colas.csv`
- `reports/mnq/TDA07_qq_global.png`
- `reports/mnq/TDA07_qq_por_segmento.png`
- `reports/mnq/TDA07_qq_th08_contrafactual.png`

Ningún artefacto de TDA-00…TDA-06 fue modificado. `tda06_r_tilde.parquet` y `TDA06_segmentacion_propuesta.csv` se leyeron sin alterarlos (verificado con test dedicado de que `load_segmentation_cutoffs` no modifica el archivo fuente).

## 9. Tests ejecutados

```
python -m pytest -q
```

| Archivo | Tests |
|---|---:|
| Suite previa (TDA-00…06) | 220 |
| `test_tda07_marginal_distribution.py` | **56** (42 del cierre inicial + 13 de la 1ª corrección puntual + 1 de la 2ª) |
| **Total final** | **276** |

**Resultado: `276 passed`.**

Cobertura de las validaciones exigidas: alineación de timestamps (pasa/falla, `TimestampAlignmentError`); contradicción `r_naive` vs `r_1m` en `VALID` (pasa/falla, respeta tolerancia, `NaiveReturnContradictionError`); construcción del contrafactual (fórmula exacta, `shift(1)` incondicional verificado contra cálculo manual); exclusión natural de `FIRST_OBSERVATION` (única fila `NaN` en `r_naive_1m` de una serie sin fronteras); aislamiento por `invalid_reason` (cada causa cuenta exactamente sus filas, con valor esperado verificado a mano); conservación de etiquetas `RETROSPECTIVO` (`r_tilde`) y `CONTRAFACTUAL_VIOLA_NO_CRUCE` (`r_naive_1m`); segmentación leída sin alterar el CSV de TDA-06 (hash de archivo antes/después) y filtrando solo cortes `stable=True`; aislamiento del hold-out (3 tests, mismo patrón que TDA-00…06: overlap de listas, fila que alcanza la frontera, ningún archivo raw/holdout abierto); momentos y cuantiles con resultado conocido a mano (distribución simétrica con curtosis calculada analíticamente; recorte que efectivamente remueve un outlier situado fuera del percentil de corte); reproducibilidad del bootstrap de bloques y de HAC (misma semilla → mismo resultado, bit a bit); HAC con dependencia positiva fuerte produce un error estándar mayor que el ingenuo iid (dentro de un único bloque continuo); `hac_bandwidth` verificado contra la fórmula cerrada de Newey-West para varios `T`, incluyendo el ejemplo citado en la documentación (T=100 → l=4); extremo a extremo con configuración sintética (invariantes, TH08, segmentación, tablas GLOBAL/YEAR/SEGMENT) sin abrir ningún archivo raw ni de hold-out. **Corrección puntual — tests del bug de HAC** (13 nuevos): `compute_hac_block_ids` nunca enlaza dos jornadas distintas ni siquiera con delta=60s exacto; nunca enlaza a través de un hueco dejado por una fila invalidada (`NON_CONSECUTIVE_MINUTE`) ni de lo que sería una frontera de `ROLL_BOUNDARY`; una población tipo segmento (mismo minuto de días distintos) nunca vuelve vecinas dos observaciones separadas ~24h; `hac_mean_se` con dos bloques de niveles muy distintos produce un resultado DIFERENTE al cálculo ingenuo que trataría el array como un único bloque (reproduce el bug original y confirma la corrección); `hac_mean_se` coincide EXACTAMENTE (tolerancia 1e-10) con la fórmula anterior cuando la serie es genuinamente continua (un solo bloque); `analyze_group` con una población tipo segmento (40 días, 1 observación por día) reduce el HAC exactamente al término `gamma_0/T` (ningún rezago encuentra pareja dentro del mismo bloque); `verify_r_tilde_invariants` pasa/falla en los tres casos (alineación de longitud, alineación de timestamps, etiqueta `RETROSPECTIVO`). **Corrección puntual — test de normalización** (1 nuevo): con 3 bloques cortos de niveles muy distintos, la suma de covarianzas permitidas se conoce de forma independiente en el test; se comprueba que `hac_mean_se` coincide exactamente con la normalización por `T` y difiere de la normalización por el conteo de pares sobrevivientes (que sobrepondera y produce un e.e. mayor en este ejemplo).

## 10. Preguntas abiertas

1. **Drift por año/segmento no es estable** (§4, §7): 2 de 7 años y 1 de 7 segmentos excluyen cero en el intervalo HAC. La **media puntual** está siempre por debajo de 0,1 ticks (máximo 0,093, segmento `02:00-03:00`); el **extremo superior del intervalo HAC**, en cambio, supera 0,1 ticks en varios años/segmentos (hasta 0,240 en 2025) — se aclara esta distinción porque son magnitudes distintas (efecto puntual vs. incertidumbre de su estimación). No se cita como propiedad del dataset — candidato a revisar si TDA-08 (dependencia en media) encuentra algo relacionado, pero no es una instrucción para esa etapa, solo una observación.
2. **El signo de la asimetría (`skewness`) cambia entre años** (§7): positiva en 3 de 7 años, negativa en 4 de 7 — no es una propiedad estable, se documenta el rango completo, no un signo único.
3. **`freq_diff` (TH13) tiene signo distinto por segmento** (§5, §7): la asimetría de colas GLOBAL está dominada por los segmentos de la tarde/noche NY; los segmentos de mañana muestran el signo contrario. No se investiga la causa aquí (fuera de alcance de TDA-07).
4. **La asimetría profunda (p=0,001) no es estadísticamente distinguible de cero** (§5), a diferencia de la de p=0,01 — consistente con menos muestra efectiva en la cola más profunda; no se fuerza una conclusión con evidencia insuficiente.
5. **TH10** (escalado de varianza, diferida desde TDA-04/TDA-06): sigue diferida; TDA-07 no encontró ninguna dependencia metodológica que obligara a retomarla.

Ninguna de estas preguntas es bloqueante para continuar.

## 11. Estado final

**`PASS_WITH_OPEN_QUESTIONS`**

Las tres invariantes bloqueantes se cumplieron sobre el conjunto de investigación real (incluida la nueva invariante de `r_tilde` añadida en la corrección puntual de cierre). TH08 queda **RESUELTA**: las reglas de no-cruce de TDA-04 evitan una contaminación de cola severa y concentrada (curtosis cruda 16× mayor sin ellas; curtosis recortada casi idéntica, confirmando que el efecto proviene casi enteramente de las 3.519 filas excluidas). TH11 (distribución marginal), TH12 (drift) y TH13 (asimetría de colas) quedan **RESUELTAS**, con el resultado explícito de que ninguna de las cifras de asimetría/curtosis cruda es estable entre años o segmentos (G3) — la marginal agregada de MNQ es, tal como advierte el roadmap, una mezcla, y se reporta como tal, nunca como una única propiedad del dataset. `r_tilde` reduce la curtosis cruda (−30%) sin alterar la curtosis recortada, consistente con que la estacionalidad de reloj (TDA-06) explica una parte real pero no dominante del exceso de curtosis. TH10 sigue diferida.

**No se avanza a TDA-08.**

## 12. Recomendación para el siguiente paso

- TDA-08 (Dependencia lineal en la media) debe tratar el drift caracterizado en §4/§10.1 exactamente como candidato a artefacto de microestructura o frontera de sesión, no como señal — la magnitud de su media puntual (<0,1 ticks en todos los grupos) y su inestabilidad entre años ya lo desaconsejan como hallazgo por sí solo.
- TDA-08 es también la etapa natural para investigar las dos ventanas de apertura/cierre identificadas por TDA-06 (§7 de ese informe), explícitamente diferidas hasta ahora.
- Usar la tabla `TDA07_momentos_cuantiles_drift_colas.csv` (columnas `q0.001`…`q0.999`) como referencia de escala antes de cualquier estandarización en TDA-09/TDA-10 — la curtosis recortada (11,3-11,5 global) es la cifra más estable disponible hasta ahora para juzgar cuánta de la no-normalidad es genuina.
- Retomar TH10 en un punto conveniente antes de TDA-08/TDA-09, como ya recomendaron TDA-04, TDA-05 y TDA-06.
- Mantener la segmentación de TDA-06 como partición opcional de análisis (TDA-07 la usó sin modificarla) — sigue sin ser una decisión de arquitectura ML.
