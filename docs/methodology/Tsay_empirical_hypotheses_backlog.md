# TSAY EMPIRICAL HYPOTHESIS BACKLOG
## Consolidación de las hipótesis de los Capítulos 1, 2, 3, 4, 5, 7 y 11 — aplicadas a MNQ OHLCV 1-min

**Documentos hermanos:** `Tsay_sintesis_transversal_OHLCV.md` · `Tsay_OHLCV_analysis_roadmap.md`
**Datos disponibles:** MNQ · OHLCV precio Last · barras de 1 minuto · ~2020–2026 · TZ declarada America/New_York · **un solo instrumento**

> **DECISIÓN IRIS: NINGUNA.** La columna "¿Afecta diseño IRIS?" dice `FUTURA` en todas las entradas. No se decide nada aquí.

---

# 1. Recuento de la consolidación

| Origen | Hipótesis originales | Comentario |
|---|---|---|
| C1 — Características de series financieras | 33 | 26 preguntas empíricas numeradas + 7 hipótesis `H-Vol-1…7` |
| C2 — Dependencia temporal | 14 | `H2.1…H2.14` |
| C3 — Volatilidad | 17 | `H3.1…H3.17` |
| C4 — No linealidad | 19 | `H-NL1…H-NL19` |
| C5 — Microestructura | 11 | `H5.1…H5.11` |
| C7 — Cuantiles y extremos | 15 | `H7.1…H7.15` |
| C11 — Estados no observados | 15 | `H11.1…H11.15` |
| **TOTAL BRUTO** | **124** | |

| Destino | Cantidad |
|---|---|
| **Descartadas — NO OBSERVABLE** con OHLCV Last 1-min de un solo instrumento | 12 |
| **Descartadas — FUERA DE FASE** (Nivel 4/5: predictibilidad OOS, utilidad económica, diseño de ML) | 38 |
| Reconvertidas en **reglas de gobernanza** del roadmap (no son hipótesis sobre el dataset) | 7 |
| Entran a consolidación | 67 |
| **Suma de destinos** | **124** ✔ |

| Consolidación de las 67 restantes | Cantidad |
|---|---|
| **Hipótesis consolidadas** (`TH01…TH37`) | **37** |
| Absorbidas por duplicación o equivalencia | 30 |

**Reducción efectiva: 124 hipótesis originales → 37 consolidadas (–70%).**
De las 37, **8 son condicionales** y pueden no ejecutarse nunca.

---

# 2. Prioridades y niveles

| Prioridad | Significado |
|---|---|
| **OBLIGATORIA** | Sin su respuesta, alguna etapa posterior no es interpretable, o queda sin responder una de las 20 preguntas del *Empirical Profile*. |
| **ALTA** | No bloqueante, pero de alto valor informativo y bajo costo. |
| **CONDICIONAL** | Se responde sólo si un resultado previo la habilita. |
| **BAJA** | Se responde si sobra presupuesto de análisis; su ausencia no compromete el perfil. |

**Nota transversal.** Toda hipótesis se responde bajo las reglas de gobernanza `G0–G6` del roadmap: observabilidad declarada, exploración vs confirmación, causalidad estricta, estabilidad obligatoria, parsimonia, reporte por magnitud y validez del resultado negativo.

---

# 3. Backlog consolidado

## NIVEL 0 — Admisibilidad del dato

### TH01 — Invariantes físicos de la barra

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Cumple cada fila los invariantes de una barra OHLC, la grilla del tick, y tiene timestamps únicos y monótonos? |
| **Origen** | C1 (discreción del tick), C5 (H/L como estadísticos de orden), C7 (extremo ≠ error) |
| **Variable requerida** | `timestamp`, `O`, `H`, `L`, `C`, `V`, tick size |
| **Datos disponibles** | **Sí** |
| **Prioridad** | **OBLIGATORIA** |
| **Dependencia** | Ninguna. Es la raíz del roadmap. |
| **Método mínimo** | Chequeo de invariantes ($H\ge L$, $H\ge\max(O,C)$, $L\le\min(O,C)$, $P>0$, $V\ge0$); verificación de múltiplos del tick; unicidad y monotonicidad de timestamps; localización temporal de cada violación |
| **Método avanzado opcional** | Contraste con una segunda fuente de datos |
| **Resultado que apoyaría** | Cero o casi cero violaciones, sin concentración temporal |
| **Resultado que debilitaría** | Violaciones frecuentes o concentradas en un tramo → `STOP-0`, ningún resultado posterior sería atribuible al mercado |
| **Riesgos** | Borrar filas en vez de marcarlas (destruye trazabilidad y sesga la cola); asumir un tick incorrecto; confundir precisión de punto flotante con violación de grilla |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-00 |

### TH02 — Semántica del timestamp y disponibilidad de cada campo

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿El timestamp marca el inicio o el cierre de la barra, y en qué instante estuvo disponible cada campo? |
| **Origen** | C11 (filtering/prediction/smoothing dependen de qué está en $\mathcal F_t$), C1 (factorización condicional), C5 (el Close es la última transacción) |
| **Variable requerida** | `timestamp` + **documentación del proveedor** |
| **Datos disponibles** | **Parcial** — no es resoluble sólo con estadística |
| **Prioridad** | **OBLIGATORIA** |
| **Dependencia** | TH01 |
| **Método mínimo** | Documental (prioritario). Si falta: forense — etiqueta de la primera y la última barra de sesión contra el horario oficial; firma de barras inactivas; comportamiento en los cambios de DST |
| **Método avanzado opcional** | Contraste de barras puntuales contra un registro externo |
| **Resultado que apoyaría** | Convención documentada y confirmada forensemente |
| **Resultado que debilitaría** | Convención `INDETERMINADO` → continuar bajo el supuesto conservador y marcar como **bloqueante para la fase de ML** |
| **Riesgos** | Asumir la convención por costumbre; una diferencia de un minuto invalida un pipeline sin dejar rastro |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-01 |

### TH03 — Completitud del eje temporal y calendario

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Está completa la grilla esperada? ¿Dónde están los huecos, de qué tamaño y por qué causa? ¿Cómo se comportan DST, feriados y medias sesiones? |
| **Origen** | C11 (los valores faltantes tienen tratamiento propio) |
| **Variable requerida** | `timestamp`, calendario de sesión derivado de TH02 |
| **Datos disponibles** | **Sí** |
| **Prioridad** | **OBLIGATORIA** |
| **Dependencia** | TH02 (sin definición de sesión no existe "grilla esperada") |
| **Método mínimo** | Cobertura por año/mes/día; inventario de huecos con causa candidata; conteo de barras por jornada; barras fuera de horario; verificación de los saltos de DST |
| **Método avanzado opcional** | Contraste con calendario oficial de feriados de CME |
| **Resultado que apoyaría** | Huecos estructurales alineados con el horario, sin huecos esporádicos relevantes |
| **Resultado que debilitaría** | Sesión real distinta de la declarada → `STOP-2`, revisar TH02 |
| **Riesgos** | Confundir el mantenimiento diario del mercado con un defecto; **imputar huecos (prohibido: crea datos)** |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-02 |

### TH04 — Naturaleza de las barras inactivas

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Una barra sin actividad se representa como fila ausente, como volumen cero, o como *forward-fill* ($O=H=L=C$)? |
| **Origen** | C11 (barra faltante ≠ volumen cero ≠ valor imputado), C5 (fracción de "sin cambio") |
| **Variable requerida** | Presencia/ausencia de barras, $V_t$, firma $O=H=L=C$ |
| **Datos disponibles** | **Sí** |
| **Prioridad** | **OBLIGATORIA** |
| **Dependencia** | TH02, TH03 |
| **Método mínimo** | Clasificación tripartita `AUSENTE` / `VOLUMEN_CERO` / `FORWARD_FILL`; conteo y distribución horaria de cada categoría |
| **Método avanzado opcional** | — |
| **Resultado que apoyaría** | Ausencia de forward-fill; barras inactivas identificables sin ambigüedad |
| **Resultado que debilitaría** | Presencia de forward-fill → genera $r_t=0$ artificiales, deprime la volatilidad estimada, infla la fracción de ceros y puede generar ACF espuria. **Obliga a reportar toda propiedad con y sin esas barras.** |
| **Riesgos** | Tratar las tres categorías como homogéneas; se ven idénticas en un DataFrame |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-02 |

### TH05 — Rolls y construcción de la serie continua

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Es una serie continua? ¿Dónde están los rolls, de qué magnitud, y qué método de ajuste se aplicó? ¿Qué estadísticos son utilizables sobre ella? |
| **Origen** | C1 (ratio preserva retornos; aditivo preserva puntos; ninguno ambos; el factor de ajuste es constante dentro del segmento; regla de roll ex post = no causal) |
| **Variable requerida** | $C_t$, $V_t$, calendario de vencimientos del Nasdaq-100 |
| **Datos disponibles** | **Parcial** — el método de ajuste puede ser indeterminable sin documentación |
| **Prioridad** | **OBLIGATORIA** |
| **Dependencia** | TH02, TH03 |
| **Método mínimo** | Detección de saltos atípicos en fechas del ciclo trimestral; firma en volumen; magnitud del salto en puntos y en %; **test de invariancia a escala** de cada estadístico previsto; máscara `roll` |
| **Método avanzado opcional** | Reconstrucción intra-contrato y comparación, si se dispusiera de los contratos individuales |
| **Resultado que apoyaría** | Rolls localizados, saltos acotados, y todos los estadígrafos previstos clasificados como invariantes a escala |
| **Resultado que debilitaría** | Discontinuidades no atribuibles a rolls ni a eventos conocidos → `STOP-3`, volver a TH01 |
| **Riesgos** | Confundir un evento real con un roll y viceversa; usar estadísticos dependientes del nivel sobre una serie ajustada |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-03 |

### TH06 — Triage de extremos: dato erróneo, artefacto o evento real

| Campo | Contenido |
|---|---|
| **Pregunta** | De los movimientos extremos observados, ¿cuáles son errores de datos, cuáles artefactos de roll o de frontera, y cuáles eventos reales? |
| **Origen** | C7 (extremo ≠ error de datos, y tampoco a la inversa), C5 (contaminación de microestructura), C1 (los extremos son el fenómeno, no ruido) |
| **Variable requerida** | $r_t$, máscaras de TH01/TH05, calendario de eventos macro conocidos |
| **Datos disponibles** | **Parcial** — sin ticks no se puede confirmar el origen microestructural de un extremo |
| **Prioridad** | **OBLIGATORIA** |
| **Dependencia** | TH01, TH03, TH05, TH14 (umbral relativo, no absoluto) |
| **Método mínimo** | Inventario con contexto; clasificación `ERROR_DATO`/`ARTEFACTO_ROLL`/`ARTEFACTO_FRONTERA`/`EVENTO_REAL`/`INDETERMINADO`, **con criterio escrito antes de mirar su efecto**; recálculo de curtosis y cuantiles con y sin cada categoría |
| **Método avanzado opcional** | Contraste con una fuente de noticias con marca temporal |
| **Resultado que apoyaría** | Fracción pequeña de artefactos, con la mayoría clasificable |
| **Resultado que debilitaría** | Fracción alta de `INDETERMINADO` → la caracterización de colas debe reportarse como un rango, no como un valor |
| **Riesgos** | Descartar extremos (sesga la cola al optimismo); conservarlos sin triage (mete artefactos en toda la caracterización) |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-00 (parcial) + TDA-13A |

---

## NIVEL 1 — Definición del objeto de análisis

### TH07 — Equivalencia entre retorno logarítmico y simple

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Cuánto difieren $r_t$ y $R_t$, y se concentra la diferencia en los tramos de alta volatilidad? |
| **Origen** | C1 (la precisión de la aproximación de primer orden "puede no ser suficiente"; simple aditivo en cartera, log aditivo en el tiempo) |
| **Variable requerida** | $r_t$, $R_t$ |
| **Datos disponibles** | **Sí** |
| **Prioridad** | **ALTA** (costo casi nulo) |
| **Dependencia** | TH05 (máscaras) |
| **Método mínimo** | Distribución de $\lvert r_t-R_t\rvert$, global y por decil de magnitud |
| **Método avanzado opcional** | — |
| **Resultado que apoyaría** | Diferencia despreciable a escala de 1 minuto |
| **Resultado que debilitaría** | Diferencia concentrada en los tramos de mayor volatilidad → la elección de representación **interactúa con el régimen**, justo donde más importa |
| **Riesgos** | Concluir "da lo mismo cuál usar" a partir de estadísticos descriptivos: la diferencia importa al **acumular** |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-04 |

### TH08 — Efecto de las reglas de no-cruce

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Cuántas observaciones y qué propiedades cambian al impedir que $r_t$ cruce fronteras de día, sesión, hueco y roll? ¿Qué magnitud tiene el gap entre barras? |
| **Origen** | C1 (un salto de roll no es un retorno), C3 (gap overnight/fin de semana; calcular volatilidad sobre barras que abarcan un roll es un error) |
| **Variable requerida** | $r_t$ con y sin reglas; $\text{gap}_t=\ln(O_t/C_{t-1})$ |
| **Datos disponibles** | **Sí** |
| **Prioridad** | **OBLIGATORIA** |
| **Dependencia** | TH03, TH05 |
| **Método mínimo** | Conteo de observaciones perdidas por causa; comparación de momentos y cuantiles con y sin reglas; distribución de $\text{gap}_t$ por tipo de frontera |
| **Método avanzado opcional** | — |
| **Resultado que apoyaría** | Pérdida pequeña de observaciones y cambio material en las colas al aplicar las reglas (confirma que los cruces eran artefactos) |
| **Resultado que debilitaría** | Pérdida sustancial de observaciones → `STOP-4`, revisar la definición de sesión, **no** relajar la regla |
| **Riesgos** | Rellenar los `NaN` de frontera; tratar un retorno que cruza el hueco nocturno como comparable con uno de 1 minuto |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-04 |

### TH09 — Resolución efectiva y discreción

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Cuál es el cociente $\text{tick}/\hat\sigma_r$ y la fracción de barras con $r_t=0$, por segmento horario y por año? ¿Es $r_t$ efectivamente discreta? |
| **Origen** | C1 ("para retornos de alta frecuencia, la discreción se vuelve un problema"), C5 (los tres episodios de tick vs fracción de "sin cambio"; el tick sigue restringiendo O/H/L/C tras la agregación) |
| **Variable requerida** | $r_t$, tick, $\hat\sigma$ local, $\text{zero}_t$ |
| **Datos disponibles** | **Sí** |
| **Prioridad** | **OBLIGATORIA** |
| **Dependencia** | TH04 (separar ceros por forward-fill), TH08 |
| **Método mínimo** | Histograma de $\Delta C_t$ en múltiplos de tick; fracción de ceros **con y sin** barras forward-fill; $\text{tick}/\hat\sigma$ y $\text{tick}/\text{mediana}(\text{rg})$; todo desagregado por hora y por año; número de valores distintos de $r_t$ |
| **Método avanzado opcional** | Repetir a 5 y 10 minutos |
| **Resultado que apoyaría** | Resolución fina en el segmento principal → interpretación continua defendible allí |
| **Resultado que debilitaría** | Resolución gruesa en algún segmento → allí momentos, densidades y QQ **no** se interpretan como continuos; `STOP-5` |
| **Riesgos** | Reportar una única fracción agregada que oculta que de noche la variable es casi degenerada |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora — pero **fija el suelo de relevancia** de todo el roadmap |
| **Etapa** | TDA-05 |

### TH10 — Escalado de la varianza con el horizonte

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Se cumple $\text{Var}(r_t[h])\approx h\cdot\text{Var}(r_t)$? ¿A partir de qué $h$ se rompe y en qué dirección? |
| **Origen** | C1 (agregación aditiva del log-retorno; la desviación señala autocorrelación o memoria), C7 (regla $\sqrt k$ como consecuencia de supuestos, no ley) |
| **Variable requerida** | $r_t[h]$ para varios $h$ |
| **Datos disponibles** | **Sí** |
| **Prioridad** | **ALTA** (diagnóstico barato que anticipa TH16 y TH19) |
| **Dependencia** | TH08 |
| **Método mínimo** | $\log\text{Var}(r[h])$ vs $\log h$; la pendiente es el resultado |
| **Método avanzado opcional** | Estimación del exponente de escalado por subperíodo |
| **Resultado que apoyaría** | Pendiente $\approx1$ → consistente con independencia |
| **Resultado que debilitaría** | Pendiente $<1$ → reversión de corto plazo, refuerza la sospecha de microestructura (TH17). Pendiente $>1$ → persistencia |
| **Riesgos** | Usar ventanas solapadas sin advertir que inflan la aparente precisión |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-04 |

### TH11 — Distribución marginal y magnitud de la no normalidad

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Cuáles son momentos y cuantiles de $r_t$? ¿Cuán lejos está de la normal y **dónde**? ¿Cuánta curtosis depende de un puñado de observaciones? |
| **Origen** | C1 (exceso de curtosis alto y generalizado; densidad con pico alto y colas gruesas; JB supone iid y su p-valor está distorsionado con clustering; si el 4º momento no existe, la curtosis muestral no converge) |
| **Variable requerida** | $r_t$, $\text{rg}_t$, $V_t$ |
| **Datos disponibles** | **Sí** |
| **Prioridad** | **OBLIGATORIA** |
| **Dependencia** | TH09 (interpretación bajo discreción), TH14 (segmentación) |
| **Método mínimo** | Momentos; **curtosis con y sin recorte del 0.1% extremo**; cuantiles 0.1–99.9; QQ-plot; todo global, por segmento y por año |
| **Método avanzado opcional** | Ajuste de familias paramétricas — **desaconsejado antes de TH22** (se estaría ajustando una mezcla conocida) |
| **Resultado que apoyaría** | Curtosis estable ante el recorte → descriptor utilizable |
| **Resultado que debilitaría** | Curtosis que colapsa al recortar → el estimador está dominado por pocos puntos y **no es un parámetro** |
| **Riesgos** | Reportar el rechazo de normalidad como hallazgo: con una muestra muy grande, los tests pueden detectar desviaciones extremadamente pequeñas, por lo que el rechazo aislado no informa sobre su magnitud o importancia práctica; presentar la marginal agregada como si describiera un estado homogéneo |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-07 |

### TH12 — Drift: ¿es la media distinguible de cero?

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Es el drift por barra distinguible de cero, y cuál es la relación señal/ruido del primer momento? |
| **Origen** | C1 (media diaria de IBM indistinguible de cero: $t=1.51$, $p=0.13$ con $T=9{,}845$; ratio media/desviación ≈0.024) |
| **Variable requerida** | $r_t$ |
| **Datos disponibles** | **Sí** |
| **Prioridad** | **ALTA** |
| **Dependencia** | TH08 |
| **Método mínimo** | Media con **errores estándar HAC** y con **bootstrap por bloques**; reportar el intervalo y la traducción a ticks, nunca el p-valor solo |
| **Método avanzado opcional** | Media por segmento y por año |
| **Resultado que apoyaría** | Intervalo que contiene cero, o drift material y estable |
| **Resultado que debilitaría** | Drift "significativo" con magnitud muy inferior al tick → establece la escala del problema, no una señal |
| **Riesgos** | Con una muestra muy grande, incluso un drift de magnitud muy pequeña puede producir un t-estadístico elevado; por eso **la magnitud, el intervalo de incertidumbre y su estabilidad son el resultado principal**, no el p-valor aislado |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-07 |

### TH13 — Asimetría entre cola izquierda y cola derecha

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Difieren las colas izquierda y derecha en magnitud, frecuencia y forma? |
| **Origen** | C1 ("la asimetría no es un problema serio" **a la vez** que se rechaza simetría con $p=0.013$; los extremos negativos importan al riesgo y los positivos a las posiciones cortas), C7 (`H7.1`) |
| **Variable requerida** | $r_t$ |
| **Datos disponibles** | **Sí** |
| **Prioridad** | **ALTA** |
| **Dependencia** | TH06 (triage), TH11 |
| **Método mínimo** | Comparación de cuantiles simétricos ($\lvert q_{0.01}\rvert$ vs $q_{0.99}$; $\lvert q_{0.001}\rvert$ vs $q_{0.999}$); curtosis y frecuencia de excedencias por lado; con intervalos por bootstrap |
| **Método avanzado opcional** | Índice de cola por lado (sólo si TH28 se habilita) |
| **Resultado que apoyaría** | Asimetría material y estable entre subperíodos |
| **Resultado que debilitaría** | Asimetría dentro del error de estimación → simetría no rechazable en la práctica |
| **Riesgos** | Trasladar la conclusión de Tsay ("no es un problema serio") sin verificarla: es una afirmación sobre **descripción**, y sus efectos sobre las colas de una estrategia pueden ser grandes |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-07 |

### TH14 — Perfil determinista intradía

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Existe un patrón determinista de actividad y volatilidad asociado a la hora? ¿Qué forma tiene? ¿Es estable entre años? ¿Está en la media, en la varianza, o en ninguna? |
| **Origen** | C5 (patrón diurno con periodicidad igual a la jornada; forma en U en NYSE), C3 (`H3.17`: estacionalidad vs dinámica ARCH), C2 (`H2.14`), C7 (`H7.4`) |
| **Variable requerida** | $V_t$, $\lvert r_t\rvert$, $r_t^2$, $\text{rg}_t$, $\text{zero}_t$, frecuencia de extremos, $r_t$ |
| **Datos disponibles** | **Sí** |
| **Prioridad** | **OBLIGATORIA** — es el prerequisito de TH16, TH19, TH26, TH27 |
| **Dependencia** | TH08, TH09 |
| **Método mínimo** | Perfil por minuto del día (mediana + bandas) de cada variable; **superposición por año** para verificar estabilidad; comparación del perfil de media contra el de varianza; segmentación derivada de los quiebres observados; factor de escala $s(m)$ etiquetado como `RETROSPECTIVO` |
| **Método avanzado opcional** | Descomposición en componentes vía state-space — **explícitamente desaconsejada** (C11: no unicidad; un perfil promedio responde lo mismo con supuestos mucho menores) |
| **Resultado que apoyaría** | Perfil fuerte y estable en varianza → todas las etapas posteriores se reportan crudo y ajustado |
| **Resultado que debilitaría** | Perfil débil o inestable → `STOP-6`, **no** se construye $s(m)$; simplifica todo el roadmap. Resultado negativo válido |
| **Riesgos** | Asumir la forma en U de NYSE en un mercado casi 24h; importar la segmentación de una decisión histórica del repositorio (**prohibido**); un perfil en la **media** es candidato a artefacto de frontera antes que a fenómeno |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-06 |

### TH15 — Efectos de calendario semanal

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Hay efectos de día de la semana, y están en la media, en la varianza o en ninguna? |
| **Origen** | C2 (`H2.14`, efectos de calendario; molde de los ejercicios de regresión con indicadores) |
| **Variable requerida** | $r_t$, $\lvert r_t\rvert$, $\text{rg}_t$, $V_t$ |
| **Datos disponibles** | **Sí** |
| **Prioridad** | **BAJA** |
| **Dependencia** | TH14 |
| **Método mínimo** | Estadísticos por día de la semana con intervalos; corrección por multiplicidad al testear varias etiquetas |
| **Método avanzado opcional** | Regresión con indicadores y errores estándar HAC, verificando la ACF de residuos |
| **Resultado que apoyaría** | Efecto estable entre años |
| **Resultado que debilitaría** | Efecto que cambia de signo entre años → ruido |
| **Riesgos** | Testear muchas etiquetas de calendario aumenta la probabilidad de encontrar alguna aparentemente "significativa" por azar (G2) |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-06 |

---

## NIVEL 2 — Dependencia

### TH16 — Dependencia lineal en la media

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Es la ACF de $r_t$ distinguible de cero, y **de qué tamaño en ticks**? |
| **Origen** | C1 (`#12`), C2 (`H2.4`, `H2.9`, `H2.10`: ACF, Ljung–Box, bandas de Bartlett; $R^2=2.46\%$ con coeficientes significativos al 1%) |
| **Variable requerida** | $r_t$, $\tilde r_t$ |
| **Datos disponibles** | **Sí** |
| **Prioridad** | **OBLIGATORIA** |
| **Dependencia** | TH14 (sin ajuste de reloj la ACF tiene picos espurios), TH09 (para la traducción a ticks) |
| **Método mínimo** | ACF hasta ≥2 jornadas con **bandas de Bartlett**; PACF en rezagos bajos; Ljung–Box para varios $m$ incluyendo múltiplos de la jornada; **traducción de $\hat\rho_1$ a movimiento esperado en ticks**; calibración por permutación |
| **Método avanzado opcional** | AR($p$) de referencia con verificación de residuos, sólo para cuantificar el $R^2$ alcanzable con estructura lineal — **nunca como modelo predictivo** |
| **Resultado que apoyaría** | ACF materialmente distinta de cero y estable |
| **Resultado que debilitaría** | ACF dentro de bandas, o magnitud implícita muy inferior al tick → `STOP-8a` |
| **Riesgos** | Usar bandas $1/T$ (asumen iid); reportar significancia con $n\sim10^6$; leer una ACF pequeña como señal |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-08 |

### TH17 — Atribución de la autocorrelación de rezago 1

| Campo | Contenido |
|---|---|
| **Pregunta** | Si hay $\hat\rho_1<0$, ¿es huella residual de bid–ask bounce, efecto de trading no sincrónico, o reversión económica genuina? |
| **Origen** | C5 (`H5.6`, `H5.1`, `H5.7`: modelo de Roll con $\rho_1=-0.5$; con valor fundamental de camino aleatorio $\rho_1=-\frac{S^2/4}{S^2/2+\sigma^2}$; no sincronía con $\text{Cov}=-\mu^2\pi$), C2 (`H2.5`) |
| **Variable requerida** | $r_t$ a 1/5/10 min; $V_t$; segmento horario; tick |
| **Datos disponibles** | **Parcial** — **la atribución concluyente requiere Bid/Ask o ticks, que NO se tienen** |
| **Prioridad** | **OBLIGATORIA** (el intento de refutación es obligatorio; la conclusión puede ser "no separable") |
| **Dependencia** | TH16 |
| **Método mínimo** | Protocolo de refutación: (i) comportamiento de $\hat\rho_1$ al agregar 1→5→10 min; (ii) por decil de volumen y por segmento; (iii) magnitud implícita comparada con el tick; (iv) consistencia con la fracción de barras sin cambio |
| **Método avanzado opcional** | Ninguno posible con los datos disponibles |
| **Resultado que apoyaría** | Efecto que **no** se diluye al agregar y **no** empeora en tramos ilíquidos → menos compatible con microestructura |
| **Resultado que debilitaría** | Efecto que se diluye rápidamente y se concentra en tramos ilíquidos → compatible con microestructura → `STOP-8b` |
| **Riesgos** | Presentar como resuelta una atribución que los datos no permiten cerrar. **Declarar el límite es el resultado correcto.** |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-08 |

### TH18 — Estabilidad de la dependencia lineal

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Es la ACF de $r_t$ estable entre segmentos y entre años, o cambia de signo? |
| **Origen** | C2 (`H2.6`) |
| **Variable requerida** | $r_t$ |
| **Datos disponibles** | **Sí** |
| **Prioridad** | **ALTA** |
| **Dependencia** | TH16 |
| **Método mínimo** | ACF por año y por segmento, superpuesta; ventanas rodantes de $\hat\rho_1$ |
| **Método avanzado opcional** | — |
| **Resultado que apoyaría** | Estabilidad de signo y magnitud |
| **Resultado que debilitaría** | Cambio de signo entre subperíodos → la dependencia agregada es un promedio de fenómenos distintos y no es citable |
| **Riesgos** | Reportar una ACF agregada de 6 años como "la ACF del dataset" |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-08 |

### TH19 — Dependencia en magnitud (volatility clustering)

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Existe volatility clustering? ¿Sobrevive al ajuste por el perfil de reloj? ¿Cuánta más estructura hay en magnitud que en dirección? |
| **Origen** | C1 (`#13`), C2 (`H2.7`, `H2.8`: ACF de $\lvert r\rvert$ significativa tras 300 rezagos mientras la de $r$ es nula), C3 (`H3.1`, `H3.2`: Ljung–Box y LM de Engle sobre $a_t^2$; ejemplo Intel $Q(12)=18.26$ vs $89.85$) |
| **Variable requerida** | $\lvert r_t\rvert$, $r_t^2$, $\text{rg}_t$ — cada una cruda y ajustada |
| **Datos disponibles** | **Sí** |
| **Prioridad** | **OBLIGATORIA** |
| **Dependencia** | TH14 (**indispensable**), TH16 (media removida) |
| **Método mínimo** | ACF hasta varios cientos de rezagos con bandas de Bartlett; **comparación crudo vs ajustado**; verificación de picos en múltiplos de la jornada; Ljung–Box y LM de Engle; **gráfico triple** de ACF ($r$, $\lvert r\rvert$, $r^2$) |
| **Método avanzado opcional** | Estimadores de rango con supuestos (Parkinson, GK, RS, YZ) si $\ln(H/L)$ crudo resulta insuficiente |
| **Resultado que apoyaría** | ACF de magnitud que **sobrevive** al ajuste, con decaimiento lento |
| **Resultado que debilitaría** | ACF que **colapsa** al ajustar → el hallazgo era el reloj; `STOP-9`, se salta TH23/TH24 |
| **Riesgos** | Ejecutar sin ajuste estacional y concluir ARCH (error explícito de C3); usar sólo $r_t^2$, que es muy ruidoso; leer clustering como señal direccional |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-09 |

### TH20 — Forma del decaimiento: memoria larga o persistencia corta

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Decae la ACF de $\lvert r_t\rvert$ de forma polinomial o exponencial? |
| **Origen** | C2 (`H2.7`: decaimiento polinomial $\rho_k\sim k^{2d-1}$), C3 (`H3.12`) |
| **Variable requerida** | $\lvert r_t\rvert$ ajustada |
| **Datos disponibles** | **Sí** |
| **Prioridad** | **ALTA** |
| **Dependencia** | TH19 |
| **Método mínimo** | ACF en escala log-log frente a semi-log — distingue las dos formas sin ajustar ningún modelo |
| **Método avanzado opcional** | Estimación del parámetro de diferenciación fraccional |
| **Resultado que apoyaría** | Decaimiento claramente identificable y estable entre subperíodos |
| **Resultado que debilitaría** | Forma que cambia entre subperíodos → compatible con cambios de nivel no modelados, no con memoria larga genuina |
| **Riesgos** | C3 advierte que persistencia extrema puede ser artefacto de cambios ocasionales de nivel; **esta ambigüedad no se resuelve en esta fase** |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-09 |

### TH21 — Estacionalidad versus dinámica ARCH genuina

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Qué fracción de la dependencia observada en magnitud es atribuible al perfil determinista de reloj y qué fracción a dinámica estocástica? |
| **Origen** | C3 (`H3.17` + error explícito 27), C5 (patrón diurno) |
| **Variable requerida** | $\lvert r_t\rvert$, $r_t^2$ crudas y ajustadas; $s(m)$ |
| **Datos disponibles** | **Sí** |
| **Prioridad** | **OBLIGATORIA** |
| **Dependencia** | TH14, TH19 |
| **Método mínimo** | Diferencia de la ACF y de $Q(m)$ antes y después del ajuste; presencia/ausencia de picos en múltiplos de la jornada |
| **Método avanzado opcional** | — |
| **Resultado que apoyaría** | Fracción sustancial que sobrevive → dinámica genuina |
| **Resultado que debilitaría** | La ACF se explica esencialmente por el reloj → `STOP-9` |
| **Riesgos** | Es la confusión más fácil de cometer y la más consecuente: contamina TH19, TH22, TH26, TH27 y TH28 simultáneamente |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-09 |

---

## NIVEL 3 — Forma de la distribución condicional

### TH22 — Escala versus forma: origen de las colas *(nodo de bifurcación)*

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Qué fracción del exceso de curtosis marginal desaparece al estandarizar por una estimación **causal** de la volatilidad? |
| **Origen** | C1 (`#9`: mixtura de escala de normales como puente), C3 (`H3.13`: un GARCH gaussiano genera curtosis por sí solo, $K_a^{(g)}=6\alpha_1^2/[1-2\alpha_1^2-(\alpha_1+\beta_1)^2]$; pero las colas de GARCH-t siguen siendo cortas), C7 (descomposición $r=\mu+\sigma\epsilon$) |
| **Variable requerida** | $r_t$, $z_t=r_t/\hat\sigma_{t-1}$ con $\hat\sigma$ estrictamente causal |
| **Datos disponibles** | **Sí** |
| **Prioridad** | **OBLIGATORIA — es la hipótesis más rentable del backlog** |
| **Dependencia** | TH19 (debe existir dinámica en la volatilidad para que estandarizar tenga sentido) |
| **Método mínimo** | Curtosis de $r_t$ vs de $z_t$ con al menos dos estimadores simples (ventana rodante y EWMA) y varias ventanas; con y sin recorte; QQ superpuestos; **sensibilidad a la ventana** |
| **Método avanzado opcional** | Estandarizar además por el perfil de reloj, para separar los dos componentes de escala |
| **Resultado que apoyaría** | Caída drástica de la curtosis → `ESCALA DOMINA`; la agenda es la dinámica de $\sigma_t$ y **EVT es innecesaria** (`STOP-13`) |
| **Resultado que debilitaría** | Fracción grande que sobrevive → `FORMA SUSTANCIAL`; abre (aún condicionada) la puerta a TH28 |
| **Riesgos** | Usar $\hat\sigma$ no causal (G1); concluir con una única ventana; confundir "la curtosis bajó" con "la condicional es normal" |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-10 |

### TH23 — ¿Qué estimador de volatilidad es suficiente? *(CONDICIONAL)*

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Existe alguna pregunta de caracterización que los diagnósticos simples (ACF, EWMA causal, ventana rodante, rango) **no** respondan y que requiera un modelo paramétrico? |
| **Origen** | C3 (`H3.8`, `H3.10`, `H3.3`, `H3.7`, `H3.9`: GARCH ≡ ARMA sobre $a_t^2$; eficiencias relativas de los estimadores de rango entre ~2 y ~8; el ranking de modelos depende del proxy usado) |
| **Variable requerida** | $\lvert r_t\rvert$, $\text{rg}_t$, $r_t^2$, EWMA, ventana rodante |
| **Datos disponibles** | **Sí** |
| **Prioridad** | **CONDICIONAL** — sólo si TH19 dio `CLUSTERING GENUINO` |
| **Dependencia** | TH19, TH22 |
| **Método mínimo** | Comparación descriptiva de los estimadores simples entre sí (correlación, ruido relativo, comportamiento por segmento); **declaración escrita** de qué pregunta queda sin responder |
| **Método avanzado opcional** | GARCH(1,1) sobre la serie ajustada, con benchmark obligatorio contra los simples y diagnóstico de residuos estandarizados |
| **Resultado que apoyaría** | Existe una pregunta concreta que sólo un parámetro interpretable responde |
| **Resultado que debilitaría** | Los diagnósticos simples bastan → `STOP-11`, **no se ajusta GARCH** |
| **Riesgos** | Ajustar GARCH por costumbre; comparar pronósticos contra $r_t^2$ y concluir mal modelo por $R^2$ bajo (C3 lo advierte explícitamente) |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-11 |

### TH24 — Persistencia de la volatilidad y su estabilidad *(CONDICIONAL)*

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Cuál es la persistencia de la volatilidad, y es estable entre subperíodos o artefacto de mezclar regímenes? |
| **Origen** | C3 (`H3.3`, `H3.4`: persistencia $\hat\alpha_1+\hat\beta_1=0.9772$ en S&P mensual; **advertencia explícita**: "el fenómeno IGARCH podría ser causado por cambios ocasionales de nivel") |
| **Variable requerida** | Serie de volatilidad estimada; parámetros del modelo si se ajusta |
| **Datos disponibles** | **Sí** |
| **Prioridad** | **CONDICIONAL** |
| **Dependencia** | TH23 |
| **Método mínimo** | Persistencia medida por ventanas rodantes y por año, sin modelo paramétrico |
| **Método avanzado opcional** | Estimación de $\alpha_1+\beta_1$ por subperíodo, con intervalos |
| **Resultado que apoyaría** | Persistencia estable entre subperíodos |
| **Resultado que debilitaría** | Persistencia que varía mucho → la persistencia global es artefacto de mezcla |
| **Riesgos** | Interpretar $\alpha+\beta\approx1$ como demostración de IGARCH — es compatible con al menos tres explicaciones distintas |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-11 |

### TH25 — Asimetría de respuesta al signo del shock *(CONDICIONAL)*

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Responde la volatilidad de MNQ asimétricamente al signo del shock previo, y en qué dirección? |
| **Origen** | C3 (`H3.5`: efecto apalancamiento en IBM, 0.0658 tras subida vs 0.1501 tras caída; **el mecanismo corporativo no existe en otras clases de activo y el signo podría invertirse**) |
| **Variable requerida** | $\hat\sigma$ causal, $\text{sign}(r_{t-1})$ |
| **Datos disponibles** | **Sí** |
| **Prioridad** | **CONDICIONAL** |
| **Dependencia** | TH19, TH22 |
| **Método mínimo** | **Descriptivo primero**: volatilidad media condicionada al signo del shock previo, por magnitud del shock, con intervalos. Si la asimetría descriptiva es nula, **no se ajusta ningún modelo asimétrico** (G4) |
| **Método avanzado opcional** | EGARCH o TGARCH, sólo si la asimetría descriptiva es material |
| **Resultado que apoyaría** | Asimetría material y estable entre subperíodos |
| **Resultado que debilitaría** | Asimetría dentro del error, o inestable → resultado negativo válido |
| **Riesgos** | Trasladar el efecto apalancamiento desde acciones sin verificarlo; MNQ es un índice accionario, lo que lo hace más plausible que en commodities, **pero el mecanismo podría ser cobertura de opciones y no apalancamiento contable** |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-11 |

### TH26 — Cuantiles condicionales: ¿escala o forma?

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Cambian los cuantiles sólo en escala con el contexto, o también en forma? |
| **Origen** | C7 (`H7.2`, `H7.3`, `H7.5`: el cuantil empírico no extrapola; su varianza asintótica $p(1-p)/\{n[f(x_p)]^2\}$ explota en la cola profunda), C3 (`H3.14`: distribución condicional de los residuos estandarizados) |
| **Variable requerida** | $r_t$, $z_t$; condicionantes causales: segmento horario, decil de $\hat\sigma_{t-1}$, año |
| **Datos disponibles** | **Sí** |
| **Prioridad** | **OBLIGATORIA** |
| **Dependencia** | TH22 |
| **Método mínimo** | Cuantiles 0.1–99.9 con bootstrap por bloques, por segmento, por decil de volatilidad y por año; **perfiles de cuantiles normalizados superpuestos** (superposición = sólo escala; divergencia = forma); **reportar el $n$ que sostiene cada cuantil extremo** |
| **Método avanzado opcional** | Regresión cuantílica lineal, si hace falta condicionar por una variable continua |
| **Resultado que apoyaría** | Perfiles normalizados que se superponen → `SÓLO ESCALA`, resultado maximamente simplificador; activa `STOP-13` |
| **Resultado que debilitaría** | Divergencia sistemática entre deciles → `ESCALA Y FORMA` |
| **Riesgos** | Reportar un cuantil 99.9 de un segmento pequeño sin advertir que descansa en decenas de observaciones; condicionar por variables no causales |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-12 |

### TH27 — Agrupamiento de extremos

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Se agrupan los extremos más de lo esperable bajo independencia? ¿El agrupamiento sobrevive al estandarizar por volatilidad? |
| **Origen** | C7 (`H7.8`, `H7.9`: IBM con ACF≈0 y sin embargo $\hat\theta\approx0.82$; ignorarlo subestima el VaR ~7%; **no confundir volatility clustering con extremal clustering**) |
| **Variable requerida** | Excedencias de $r_t$ y de $z_t$ sobre umbrales relativos por segmento |
| **Datos disponibles** | **Sí** |
| **Prioridad** | **OBLIGATORIA** |
| **Dependencia** | TH06 (triage), TH14 (umbral relativo), TH22 |
| **Método mínimo** | Distribución de tiempos entre excedencias y de longitud de rachas, comparada con un proceso de Poisson de la misma intensidad — **crudo y estandarizado**. Este es el análisis mínimo suficiente (G4) |
| **Método avanzado opcional** | Índice extremal $\hat\theta$ por método de bloques o de rachas, con sensibilidad al umbral y al tamaño de bloque |
| **Resultado que apoyaría** | Agrupamiento que **sobrevive** a la estandarización → fenómeno adicional que reduce el tamaño de muestra efectivo de la cola |
| **Resultado que debilitaría** | Agrupamiento que **desaparece** al estandarizar → era volatility clustering, ya caracterizado en TH19; no aporta información nueva |
| **Riesgos** | Umbral absoluto sobre una serie con perfil diurno (selecciona horas, no eventos); el estimador de $\hat\theta$ puede caer fuera de $(0,1]$ en muestra finita, artefacto reconocido en C7 |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-13A |

### TH28 — Caracterización paramétrica de la cola *(CONDICIONAL)*

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Aporta EVT algo sobre los cuantiles empíricos para caracterizar esta serie? ¿Es el índice de cola estable entre umbrales, entre subperíodos y entre frecuencias? |
| **Origen** | C7 (`H7.6`, `H7.7`, `H7.10`: $\xi$ varía 63% entre umbrales 2% y 3%, con el error estándar duplicándose; varía 70% entre bloques de 21 y 63; con $n=252$ el e.e. llega al 70% del valor; invariancia del índice de cola bajo agregación como predicción falsable) |
| **Variable requerida** | Excesos sobre umbral, por segmento y por lado de la cola; $r_t$ a varias frecuencias |
| **Datos disponibles** | **Sí** (pero con las limitaciones de estacionariedad) |
| **Prioridad** | **CONDICIONAL — default: NO EJECUTAR** |
| **Dependencia** | TH26 (debe haber dado `ESCALA Y FORMA`), TH29 (estacionariedad), TH06 (triage), **y una razón escrita para necesitar extrapolar más allá del peor evento observado** |
| **Método mínimo** | Si se ejecuta: mean excess plot; umbral con la regla de ≈5% de excedencias **y** verificación de estabilidad; GPD; **análisis de sensibilidad obligatorio** al umbral; diagnósticos de tres partes (tasa de excedencia, distribución de excesos, independencia); **dentro de segmentos homogéneos**; colas izquierda y derecha por separado |
| **Método avanzado opcional** | Parámetros de cola como función de variables explicativas causales (Ec. 7.39 de Tsay) |
| **Resultado que apoyaría** | Índice de cola estable entre umbrales razonables y consistente entre frecuencias |
| **Resultado que debilitaría** | Variación sustancial con el umbral → `STOP-13b`, reportar el **rango**, no un valor. Diagnósticos fallidos → `STOP-13c`, descartar el modelo |
| **Riesgos** | Reportar $\xi$ como parámetro conocido; aplicar EVT clásica a una serie no estacionaria; ejecutar EVT cuando los cuantiles empíricos ya respondían la pregunta |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-13B |

---

## NIVEL 4 — Estabilidad y estructura

### TH29 — Estabilidad de las propiedades entre subperíodos

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Qué propiedades son estables entre años, subperíodos y ventanas rodantes, y cuáles no? |
| **Origen** | C2 (`H2.1`: receta de Tsay — "dividir los datos en submuestras y comprobar la consistencia"), C1 (`#10`), C3 (`H3.4`), C7 (`H7.7`), C11 (no estacionariedad) |
| **Variable requerida** | Lista **acotada y predefinida** de estadísticos centrales |
| **Datos disponibles** | **Sí** |
| **Prioridad** | **OBLIGATORIA** — es la condición de citabilidad de todo el perfil |
| **Dependencia** | TH11 a TH28 (consolida sus versiones por subperíodo) |
| **Método mínimo** | Tabla maestra estadístico × año con intervalos; ventanas rodantes; veredicto por propiedad: `ESTABLE`/`TENDENCIA`/`CAMBIO ABRUPTO`/`INESTABLE` |
| **Método avanzado opcional** | — |
| **Resultado que apoyaría** | Propiedades centrales estables → citables como propiedades del dataset |
| **Resultado que debilitaría** | Inestabilidad fuerte → `STOP-14`; ninguna propiedad agregada es citable, y el Nivel 5 se ejecuta por subperíodo o no se ejecuta |
| **Riesgos** | Buscar inestabilidad sin una lista predefinida aumenta sustancialmente el riesgo de encontrar cambios aparentes por selección ex post; etiquetar los períodos con narrativas económicas |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-14 |

### TH30 — Estacionariedad: precios versus retornos

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Son los retornos sustancialmente más cercanos a estacionarios que los precios, y en qué medida? |
| **Origen** | C2 (`H2.2`, `H2.3`: ADF sobre log S&P 500 con estadístico $-1.998$, $p=0.602$; **no rechazar no demuestra la raíz unitaria**) |
| **Variable requerida** | $C_t$, $\ln C_t$, $r_t$ |
| **Datos disponibles** | **Parcial** — un ADF riguroso debería aplicarse a un contrato individual dentro de su vida activa, no a una serie continua ajustada |
| **Prioridad** | **ALTA** |
| **Dependencia** | TH05, TH29 |
| **Método mínimo** | Comparación de la estabilidad de momentos de $C_t$ frente a los de $r_t$ entre submuestras (la receta de Tsay, que **no** requiere ADF) |
| **Método avanzado opcional** | ADF con varias especificaciones del término determinista y varios órdenes, reportando la robustez de la conclusión |
| **Resultado que apoyaría** | Momentos de $r_t$ mucho más estables que los de $C_t$ |
| **Resultado que debilitaría** | Momentos de $r_t$ también inestables → refuerza `STOP-14` |
| **Riesgos** | Deslizar de "no puede rechazarse la raíz unitaria" a "la serie contiene una raíz unitaria" — deslizamiento que comete el propio texto de Tsay y que C2 registra explícitamente; elegir la especificación del ADF que da el resultado deseado |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-14 |

### TH31 — Cambio estructural y eventos institucionales

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Hay fechas en las que varias propiedades cambian simultáneamente? ¿Corresponden a cambios institucionales conocidos del contrato? |
| **Origen** | C5 (`H5.10`: intervention analysis de Box–Tiao sobre la decimalización de Apple, coeficiente significativo al 1%), C3 (cambios de nivel como explicación alternativa de la persistencia) |
| **Variable requerida** | Estadísticos centrales por ventana; **lista predefinida de fechas candidatas** |
| **Datos disponibles** | **Parcial** — requiere una fuente externa sobre cambios de especificación del contrato |
| **Prioridad** | **ALTA** |
| **Dependencia** | TH29 |
| **Método mínimo** | Identificación de fechas donde **varios** estadísticos se mueven a la vez (uno solo es ruido); contraste con la lista predefinida de eventos institucionales |
| **Método avanzado opcional** | Intervention analysis sobre el rango o la volatilidad, si se identifica un evento con fecha conocida |
| **Resultado que apoyaría** | Coincidencia entre cambios estadísticos y eventos institucionales documentados |
| **Resultado que debilitaría** | Cambios sin correlato institucional → se describen, **no se nombran** (G3) |
| **Riesgos** | Buscar cambios sin lista predefinida; confundir un cambio institucional con un cambio de dinámica del activo |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-14 |

### TH32 — No linealidad residual *(CONDICIONAL)*

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Queda estructura no explicada tras filtrar media, volatilidad y estacionalidad? ¿Es estable entre subperíodos? |
| **Origen** | C4 (`H-NL1`, `H-NL3`: BDS requiere remover primero la dependencia lineal; BDS detecta cualquier desviación de iid y **no identifica la fuente**; **evidencia decisiva**: BDS significativo en IBM mientras la red 8-4-1 no supera al random walk, $\chi^2=0.137$, $p=0.71$) |
| **Variable requerida** | Residuo estandarizado $\hat\epsilon_t=(r_t-\hat\mu)/(\hat\sigma_{t-1}\cdot s(m_t))$, todo causal |
| **Datos disponibles** | **Sí** |
| **Prioridad** | **CONDICIONAL** |
| **Dependencia** | TH16, TH19, TH22, TH29 (y `STOP-15a` si no hay residuo limpio) |
| **Método mínimo** | BDS y F-test de Tsay sobre $\hat\epsilon_t$, con número acotado y declarado de valores de $\delta$ y $k$; **calibración por permutación (G2)**; repetición **por subperíodo** |
| **Método avanzado opcional** | Ninguno en esta fase. **Identificar la forma (TAR/STAR/Markov switching) pertenece al diseño de modelos, no a la caracterización** |
| **Resultado que apoyaría** | Rechazo que se sostiene en subperíodos independientes y no colapsa al agregar la frecuencia |
| **Resultado que debilitaría** | Sin evidencia, o evidencia inestable → `STOP-15b`, resultado negativo válido |
| **Riesgos** | Con una muestra muy grande, BDS puede tener potencia suficiente para detectar desviaciones extremadamente pequeñas; por ello, **un rechazo aislado no es suficiente** y debe evaluarse su magnitud, estabilidad, calibración y reproducibilidad por subperíodo; aplicar BDS sin filtrar puede detectar simplemente estructura ya conocida |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-15 |

### TH33 — Dependencia de la evidencia de no linealidad con la frecuencia *(CONDICIONAL)*

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Depende la evidencia de no linealidad de la frecuencia de muestreo? |
| **Origen** | C4 (`H-NL2`, `H-NL19`), C5 (`H5.1`, `H5.7`: qué estructura se atenúa al agregar) |
| **Variable requerida** | $\hat\epsilon_t$ a 1, 5 y 10 minutos |
| **Datos disponibles** | **Sí** |
| **Prioridad** | **CONDICIONAL** |
| **Dependencia** | TH32 |
| **Método mínimo** | Repetición de los tests a las tres frecuencias, con el mismo protocolo |
| **Método avanzado opcional** | — |
| **Resultado que apoyaría** | Evidencia que persiste al agregar → menos compatible con microestructura |
| **Resultado que debilitaría** | Evidencia que desaparece al agregar → candidata a artefacto de microestructura |
| **Riesgos** | Interpretar la desaparición como ausencia de fenómeno cuando puede ser pérdida de potencia por menor $n$ |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-15 |

### TH34 — ¿Existe una hipótesis latente que justifique state-space? *(CONDICIONAL)*

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Existe alguna cantidad latente concreta y falsable, postulable sobre esta serie, que una EMA causal no responda? |
| **Origen** | C11 (`H11.13`, `H11.14`, `H11.7`: local trend ≡ ARIMA(0,1,1) ≡ suavizado exponencial; en estado estacionario el Kalman **es** una EMA con $K_\infty\approx0.142=1-\hat\theta$; infinitas descomposiciones válidas; el CAPM dinámico sobre GM estima $\hat\sigma_\eta=4.9\times10^{-5}$, "essentially constant") |
| **Variable requerida** | Formulación escrita previa; comparación contra EMA / ventana rodante / rango |
| **Datos disponibles** | **Sí** (los datos no son la restricción; la hipótesis lo es) |
| **Prioridad** | **CONDICIONAL — default: NO EJECUTAR** |
| **Dependencia** | TH29, TH35 |
| **Método mínimo** | Redacción de la hipótesis: qué cantidad se postula, por qué su dinámica tendría esa forma, y qué observación la refutaría. **Si no puede redactarse, `STOP-16` y la rama se cierra.** |
| **Método avanzado opcional** | Ajuste del modelo con benchmark obligatorio; sólo filtering, nunca smoothing; parámetros estimados por subperíodo y comparados con la estimación global |
| **Resultado que apoyaría** | Hipótesis falsable formulable y no cubierta por alternativas simples |
| **Resultado que debilitaría** | No formulable (caso esperado) → `STOP-16`. O bien la estimación encuentra dinámica ≈0 → `STOP-16c`, la hipótesis queda **refutada** |
| **Riesgos** | Aplicar Kalman por haberlo estudiado; postular un "precio verdadero" — cadena de razonamiento **no presente en Tsay y explícitamente rechazada** |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-16 |

---

## NIVEL 5 — Representación y auditoría

### TH35 — Redundancia informacional entre transformaciones

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Cuántas dimensiones efectivas contiene el conjunto de transformaciones derivables del OHLCV? ¿Qué transformaciones son reconstruibles a partir de otras? |
| **Origen** | C11 (`H11.14`: representación ≠ información; fabricar transformaciones sin verificar redundancia infla el espacio de búsqueda), C3 (GARCH ≡ ARMA sobre $a_t^2$) |
| **Variable requerida** | Conjunto **acotado y cerrado de antemano** de transformaciones causales |
| **Datos disponibles** | **Sí** |
| **Prioridad** | **OBLIGATORIA** |
| **Dependencia** | TH07 a TH26 (las transformaciones deben estar definidas y auditadas) |
| **Método mínimo** | Matriz de correlación (Pearson y de rangos); **$R^2$ de cada transformación regresada sobre las demás**; número de dimensiones efectivas; agrupamiento por similitud; todo por segmento y por año |
| **Método avanzado opcional** | Medidas de dependencia no lineal entre transformaciones, sólo si TH32 encontró evidencia estable |
| **Resultado que apoyaría** | Dimensión efectiva baja frente a un número alto de transformaciones — **el argumento cuantitativo contra la proliferación de indicadores** |
| **Resultado que debilitaría** | Dimensión efectiva alta → cada transformación aporta una dirección propia (poco probable; sería un hallazgo notable) |
| **Riesgos** | Convertir esto en selección de features (**prohibido en esta fase**); ampliar la lista después de ver los resultados; interpretar correlación baja como información independiente cuando puede ser ruido |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora — pero determina cómo contar pruebas independientes a efectos de multiplicidad |
| **Etapa** | TDA-17 |

### TH36 — Equivalencia entre estado filtrado y suavizadores simples *(CONDICIONAL)*

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Aporta un estado filtrado algo sobre una EMA causal calibrada, una ventana rodante o una medida de rango? |
| **Origen** | C11 (`H11.2`, `H11.3`, `H11.10`: en estado estacionario el Kalman **es** una EMA; "the decision of using ARIMA models or linear state-space models is not critical") |
| **Variable requerida** | Estado filtrado vs EMA vs ventana rodante |
| **Datos disponibles** | **Sí** |
| **Prioridad** | **CONDICIONAL** — sólo si TH34 pasó |
| **Dependencia** | TH34, TH35 |
| **Método mínimo** | Correlación y diferencia punto a punto entre estado filtrado y alternativas; sensibilidad al cociente $q=\sigma_\eta^2/\sigma_e^2$; magnitud de $\lvert s_{t\mid T}-s_{t\mid t}\rvert$ como medida de cuánto "hace trampa" el suavizado |
| **Método avanzado opcional** | Diagnóstico de innovaciones estandarizadas (Ljung–Box y LM de ARCH sobre $\tilde v_t$) |
| **Resultado que apoyaría** | Diferencia material y sistemática respecto de las alternativas simples |
| **Resultado que debilitaría** | Equivalencia práctica → `STOP-16b`. **Es el resultado esperado a priori según C11** |
| **Riesgos** | Usar el estado suavizado como si fuera causal; ignorar que los parámetros del filtro se estimaron sobre toda la muestra |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora |
| **Etapa** | TDA-16 |

### TH37 — Auditoría de causalidad de todas las transformaciones

| Campo | Contenido |
|---|---|
| **Pregunta** | ¿Es toda cantidad presentada como disponible en $t$ reconstruible usando exclusivamente datos hasta $t$? |
| **Origen** | C11 (`H11.5`, `H11.6`: leakage por smoothing; leakage por parámetros estimados sobre toda la muestra — en el local trend model la secuencia entera de ganancias está determinada **sólo** por los parámetros), C1 (leakage por el factor de ajuste del roll; test de invariancia a escala) |
| **Variable requerida** | Todas las transformaciones y todos los parámetros usados |
| **Datos disponibles** | **Sí** |
| **Prioridad** | **OBLIGATORIA — bloqueante** |
| **Dependencia** | Transversal: se aplica en cada etapa (regla `G1`), y se consolida al final |
| **Método mínimo** | Test de reconstrucción por transformación; declaración `CAUSAL` / `RETROSPECTIVO` en cada output; declaración del origen de cada parámetro; test de invariancia a escala frente al factor de ajuste del roll |
| **Método avanzado opcional** | Comparación cuantitativa entre parámetros estimados globalmente y por subperíodo, para medir la magnitud del leakage por parámetros |
| **Resultado que apoyaría** | Todas las transformaciones causales verifican la reconstrucción exacta |
| **Resultado que debilitaría** | Cualquier discrepancia señala leakage y **invalida** los outputs afectados, no sólo su interpretación |
| **Riesgos** | Las tres vías de fuga (smoothing, parámetros, ajuste de roll) **no dejan marca visible en el código y producen resultados buenos-pero-no-absurdos**, que es exactamente por lo que convencen |
| **¿Afecta diseño IRIS?** | FUTURA, no decidir ahora — pero es prerequisito absoluto de cualquier fase posterior |
| **Etapa** | Transversal (G1) + TDA-18 |

---

# 4. Hipótesis descartadas

## 4.1 NO OBSERVABLE con OHLCV Last de 1 minuto, un solo instrumento (12)

| Origen | Enunciado | Motivo del descarte |
|---|---|---|
| C5 `H5.5` | ¿Cambiarían las propiedades usando Bid/Ask/Mid en vez de Last? | **No hay Bid ni Ask.** No hay proxy válido: el Mid ni siquiera es un precio ejecutable |
| C5 `H5.9` | ¿Aportaría el número de trades por barra sobre el volumen? | **No hay columna de número de trades.** Volumen ≠ número de trades |
| C5 `H5.8` | ¿Sobrevive un lead-lag aparente a la sincronización rigurosa? | Requiere **al menos dos instrumentos** |
| C2 `H2.13` | ¿Existe relación rezagada entre dos futuros? | Requiere al menos dos instrumentos |
| C1 `#21` | Matriz de correlación entre futuros candidatos y número de factores | Requiere múltiples instrumentos |
| C1 `#22` | ¿Es estable la correlación entre pares clave? | Requiere múltiples instrumentos |
| C1 `#23` | ¿Existe lead-lag real o desaparece al controlar por horarios? | Requiere múltiples instrumentos |
| C1 `#24` | ¿Predice un modelo del instrumento A al instrumento B? | Requiere múltiples instrumentos + está en el Nivel 4 |
| C1 `#25` | ¿Mejora la performance OOS al añadir cada mercado? | Requiere múltiples instrumentos + está en el Nivel 4 |
| C1 `#26` | ¿Cuál es el tamaño de muestra efectivo de un dataset apilado? | Requiere múltiples instrumentos |
| C1 (Micro/E-mini) | ¿Son las series de MNQ y NQ estadísticamente indistinguibles? | Requiere datos de **NQ**, no disponibles |
| C1 `#1` (parcial) | ¿Difiere la serie intra-contrato de la back-adjusted? | Requiere los **contratos individuales**. La parte respondible con la serie disponible se absorbió en `TH05` |

## 4.2 FUERA DE FASE — Nivel 4 o 5 de la jerarquía, o diseño de ML (38)

| Origen | Enunciado | Motivo |
|---|---|---|
| C1 `#14` | ¿Supera un modelo condicional al incondicional en log-verosimilitud OOS? | Nivel 4. La parte de caracterización se absorbió en `TH22`/`TH23` |
| C1 `#16` | ¿Supera el edge estimado al spread + comisión + slippage? | Nivel 5. **El tick como escala de referencia** se conserva en `TH09` como filtro de relevancia, no como evaluación económica |
| C1 `#19` | ¿Es $r_{t+h}/\hat\sigma_t$ un target más aprendible? | Diseño de target — prohibido |
| C1 `H-Vol-1` | ¿Difiere la relación $X_t\to y_{t+h}$ entre regímenes de volatilidad? | Nivel 4 |
| C1 `H-Vol-2` | ¿La volatilidad predice la dirección? | Nivel 4 |
| C1 `H-Vol-3` | ¿Conviene modelar dirección y magnitud por separado? | Diseño de modelo |
| C1 `H-Vol-4` | ¿Deben los extremos recibir tratamiento especial en el modelo? | Diseño de modelo. La parte de caracterización está en `TH06`/`TH27` |
| C1 `H-Vol-6` | ¿Es $r/\hat\sigma$ un target con distribución más estable? | Diseño de target. La parte distribucional está en `TH22` |
| C1 `H-Vol-7` | ¿Es la volatilidad un buen condicionante para asignación de capital? | Risk management, otra fase |
| C1 `#20` (parcial) | ¿Son los extremos precedidos por señales observables? | Nivel 4 (predicción de extremos) |
| C2 `H2.9` | ¿Supera un AR($p$) al pronóstico incondicional OOS? | Nivel 4. El $R^2$ alcanzable in-sample se conserva como método opcional en `TH16` |
| C2 `H2.10` | ¿Cuál es el $R^2$ alcanzable con estructura lineal? | Absorbida como método opcional de `TH16` |
| C3 `H3.6` | ¿Mejora EGARCH/TGARCH el pronóstico OOS? | Nivel 4 |
| C3 `H3.8` (parte OOS) | ¿Supera un modelo de volatilidad a benchmarks simples fuera de muestra? | Nivel 4. La comparación **descriptiva** se conserva en `TH23` |
| C3 `H3.15`, `H3.16` | ¿Son comparables los parámetros GARCH entre instrumentos y frecuencias? | Instrumentos: no observable. Frecuencias: absorbido en `TH28`/`TH33` |
| C4 `H-NL4…H-NL11` (8) | Ajuste y comparación de TAR, STAR, Markov switching, coeficientes funcionales, interacciones | Identificar la **forma** de la no linealidad es diseño de modelos. `TH32` se detiene en "evidencia estable, forma no identificada" |
| C4 `H-NL12`, `H-NL13` | ¿Supera una red neuronal a un AR/regresión, y capta mejor la no linealidad? | Nivel 4, diseño de modelo |
| C4 `H-NL14`, `H-NL15`, `H-NL17` | Predicción de signo, MSE/MAD OOS, persistencia de la ventaja in-sample fuera de muestra | Nivel 4 |
| C4 `H-NL16` | ¿Está bien calibrada la distribución predictiva del bootstrap paramétrico? | Nivel 4 |
| C4 `H-NL18` | ¿Aparece la misma no linealidad en instrumentos similares? | No observable (un solo instrumento) |
| C7 `H7.11` | ¿Logran los cuantiles condicionales cobertura adecuada fuera de muestra? | Nivel 4 |
| C7 `H7.12` | ¿Hay variables que informan poco sobre la media pero sí sobre las colas? | Nivel 4. C7 la marca además como de **alto riesgo de resultado trivial** por el efecto de escala; `TH26` cubre la parte descriptiva |
| C7 `H7.15`, C5 `H5.11`, C11 `H11.15`, C3 (utilidad) | ¿Tiene utilidad operativa medible? ¿Supera la señal al spread? ¿Sobrevive a costos? | **Nivel 5.** Fuera de la fase de caracterización por definición |
| C11 `H11.1` | ¿Aporta una representación filtrada información predictiva incremental? | Nivel 4. La parte de redundancia está en `TH35`/`TH36` |
| C11 `H11.8` | ¿Aporta la incertidumbre del estado información incremental? | Nivel 4, y C11 la marca como **vacua** en modelos invariantes ($\Sigma_{t\mid t}$ converge a constante) |
| C11 `H11.11` | ¿Separa un modelo de componentes un patrón intradiario estable? | Sustituida por `TH14` con un método mucho más simple (G4); C11 advierte de la no unicidad |

## 4.3 Reconvertidas en reglas de gobernanza — no son hipótesis sobre el dataset (7)

| Origen | Enunciado | Se convirtió en |
|---|---|---|
| C1 `#15` | ¿Colapsa la performance del pipeline sobre retornos permutados y caminos aleatorios simulados? | **`G2`** — calibración obligatoria de la maquinaria de detección |
| C1 `#6` | ¿Se rechaza normalidad, y con qué magnitud? | **`G5`** + parte descriptiva absorbida en `TH11`. El rechazo con $n\sim10^6$ no es un hallazgo |
| C2 `H2.11` | ¿Pasan los residuos de un modelo el Ljung–Box? | **Método obligatorio** de diagnóstico dentro de `TH23`, `TH32`, `TH36` |
| C2 `H2.12` | ¿Cuánto cambia la evidencia al usar errores estándar HC/HAC? | **Método obligatorio** en `TH12` y en toda regresión (`G5`) |
| C11 `H11.5`, `H11.6` | ¿Es toda feature reconstruible con información disponible? ¿Cambian los estados si los parámetros se estiman por fold? | **`G1`** + formalizadas como `TH37` (auditoría bloqueante) |
| C11 `H11.12` | ¿Producen state-space y ARIMA equivalentes pronósticos idénticos? | Prueba de **validación de código**, no de investigación. C11 lo señala explícitamente |

## 4.4 Absorbidas por duplicación o equivalencia (30)

Se agruparon por responder la misma pregunta subyacente. Los casos principales:

| Hipótesis fusionadas | Consolidada en |
|---|---|
| C1 `#5`, `#8` · C3 `H3.14` (parcial) | `TH11` (distribución marginal y magnitud de la no normalidad) |
| C1 `#7` | `TH12` |
| C1 `#11` · C7 `H7.1` | `TH13` |
| C1 `#4` · C5 `H5.2`, `H5.3` | `TH09` |
| C1 `#2` | `TH10` |
| C1 `#3` | `TH07` |
| C1 `#10` · C2 `H2.1` · C3 `H3.4` · C7 `H7.7` | `TH29` |
| C1 `#12` · C2 `H2.4` | `TH16` |
| C1 `#13` · C2 `H2.7`, `H2.8` · C3 `H3.1`, `H3.2`, `H3.12` | `TH19`, `TH20` |
| C1 `#9` · C3 `H3.13` | `TH22` |
| C1 `#17`, `#18` (parte descriptiva) | `TH26` (cuantiles por estado de volatilidad) |
| C1 `#20` (parte descriptiva) · C7 `H7.8`, `H7.9` | `TH27` |
| C2 `H2.2`, `H2.3` | `TH30` |
| C2 `H2.5` · C5 `H5.1`, `H5.6`, `H5.7` | `TH17` |
| C2 `H2.6` | `TH18` |
| C2 `H2.14` · C3 `H3.17` · C5 `H5.4` · C7 `H7.4` | `TH14`, `TH15`, `TH21` |
| C3 `H3.3` | `TH24` |
| C3 `H3.5` | `TH25` |
| C3 `H3.7`, `H3.9`, `H3.10` | `TH23` |
| C3 `H3.11` | `TH08` (gap entre barras y fronteras) |
| C4 `H-NL1`, `H-NL3` | `TH32` |
| C4 `H-NL2`, `H-NL19` | `TH33` |
| C5 `H5.10` | `TH31` |
| C7 `H7.2`, `H7.3`, `H7.5` | `TH26` |
| C7 `H7.6`, `H7.10` | `TH28` |
| C7 `H7.13` | `TH06` (parte respondible) + `TH17` (parte no separable) |
| C7 `H7.14` | `TH10` (escalado con el horizonte) |
| C11 `H11.3`, `H11.10` | `TH36` |
| C11 `H11.4` | `TH36` (magnitud de $\lvert s_{t\mid T}-s_{t\mid t}\rvert$) |
| C11 `H11.9` | `TH03`, `TH04` (tratamiento de faltantes) |
| C11 `H11.13`, `H11.14` | `TH34`, `TH35` |
| C11 `H11.7` | `TH34` (el caso de GM como prior escéptico) |

---

# 5. Orden de resolución y prioridades

| Bloque | Hipótesis | Estatus |
|---|---|---|
| **Bloqueante — sin esto nada es interpretable** | `TH01` `TH02` `TH03` `TH04` `TH05` `TH37` | Obligatorias, primero |
| **Definición del objeto** | `TH07` `TH08` `TH09` `TH10` | Obligatorias / Altas |
| **Caracterización básica** | `TH11` `TH12` `TH13` `TH14` `TH15` | Obligatorias / Altas / Baja (`TH15`) |
| **Dependencia** | `TH16` `TH17` `TH18` `TH19` `TH20` `TH21` | Obligatorias / Altas |
| **Distribución condicional** | `TH22` `TH26` | Obligatorias — `TH22` es el nodo de bifurcación |
| **Extremos** | `TH06` `TH27` | Obligatorias |
| **Estabilidad** | `TH29` `TH30` `TH31` | Obligatoria / Altas |
| **Representación** | `TH35` | Obligatoria |
| **Condicionales — pueden no ejecutarse nunca** | `TH23` `TH24` `TH25` `TH28` `TH32` `TH33` `TH34` `TH36` | **8 de 37.** Que ninguna se ejecute es un desenlace válido y probable |

**Camino crítico mínimo:** `TH01 → TH02 → TH03 → TH04 → TH05 → TH08 → TH09 → TH14 → TH11 → TH16 → TH19 → TH22 → TH26 → TH27 → TH29 → TH35 → TH37`

Diecisiete hipótesis. **Ése es el conjunto mínimo y suficiente** para producir el *Empirical Profile of MNQ OHLCV*. Las veinte restantes son o bien complementos de bajo costo y alto valor (`TH07`, `TH10`, `TH12`, `TH13`, `TH18`, `TH20`, `TH21`, `TH30`, `TH31`, `TH06`, `TH15`), o bien condicionales que pueden no ejecutarse.

---

**DECISIÓN IRIS: NINGUNA**
