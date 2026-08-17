# SÍNTESIS TRANSVERSAL DE TSAY APLICADA A UN DATASET OHLCV INTRADIARIO

**Proyecto:** IRIS — Intelligent Recognition of Intraday Signals
**Instrumento:** Micro E-mini Nasdaq-100 (MNQ)
**Objeto de estudio:** barras OHLCV de 1 minuto, precio Last, ~2020–2026, zona horaria declarada America/New_York
**Fase:** DATA UNDERSTANDING / EMPIRICAL CHARACTERIZATION
**Fuente primaria:** informes propios de los Capítulos 1, 2, 3, 4, 5, 7 y 11 de Ruey S. Tsay, *Analysis of Financial Time Series*, 3ª ed.
**Capítulos deliberadamente diferidos:** 6, 8, 9, 10, 12

> **DECISIÓN IRIS: NINGUNA.**
> Este documento no adopta targets, features, horizontes, arquitecturas, umbrales operativos ni reglas de trading. Toda idea con potencial de diseño se marca como `POSSIBLE FUTURE ML IMPLICATION` y no se convierte en decisión.

---

## 0. Convención de evidencia

| Marca | Significado |
|---|---|
| **[A] TSAY** | Concepto o resultado directamente respaldado por el texto de Tsay (o por los informes en tanto reportan a Tsay). |
| **[B] ADAPTACIÓN OHLCV/FUTUROS** | Interpretación nuestra al trasladar el concepto a MNQ, futuros, barras de 1 minuto o análisis previo a ML. |
| **HECHO DEL DATASET** | Propiedad conocida directamente de nuestros datos o de su documentación. |
| **HECHO DECLARADO (NO VERIFICADO)** | Propiedad afirmada por nosotros pero aún no comprobada contra el archivo. |
| **HIPÓTESIS EMPÍRICA** | Debe comprobarse con datos. Referencia al backlog `THxx`. |
| **PREGUNTA ABIERTA** | Todavía no resoluble con la información y los capítulos disponibles. |

**Advertencia de estatus de los "hechos".** Al auditar la documentación del proyecto se constató que casi ninguna propiedad del dataset está verificada. Rango temporal, zona horaria, frecuencia, tipo de precio, criterio de contrato continuo, método de roll, definición de sesión y número de barras figuran como **declarados** o como preguntas abiertas, no como hechos comprobados. En consecuencia, en la Matriz Final de Conocimiento (§7) prácticamente toda la columna "qué sabemos de nuestros datos" está en estado `EMPIRICAL TEST REQUIRED`. **Esto no es un defecto de esta síntesis: es su primer hallazgo, y es lo que justifica que el roadmap empiece en integridad y semántica y no en estadística.**

---

## 1. La tesis central de esta síntesis

Los siete capítulos estudiados, leídos como un solo cuerpo, no producen una lista de técnicas. Producen **una manera distinta de mirar una fila de un archivo OHLCV**.

Antes de Tsay, una barra de 1 minuto se lee como *"lo que hizo el precio en ese minuto"*: cinco números y una hora.

Después de Tsay, la misma fila se lee como:

> **una realización, discretizada por el tick y resumida por cuatro estadísticos de orden, de un proceso cuya distribución condicional cambia en el tiempo, observada a través de un mecanismo de negociación que introduce estructura propia, y etiquetada con un timestamp cuya semántica determina qué información estaba realmente disponible en ese instante.**

Cada cláusula de esa frase proviene de un capítulo distinto, y **cada cláusula genera un análisis obligatorio previo a cualquier modelado**:

| Cláusula | Capítulo | Consecuencia para el análisis |
|---|---|---|
| "una realización … de un proceso" | C1 | Se modela la distribución condicional, no la trayectoria. |
| "cuya distribución condicional cambia en el tiempo" | C1, C3, C11 | La estabilidad temporal es una propiedad a medir, no un supuesto. |
| "discretizada por el tick" | C1, C5 | La resolución efectiva acota lo que puede llamarse "movimiento". |
| "resumida por cuatro estadísticos de orden" | C5 | La barra destruye el orden intraminuto: O/H/L/C no reconstruyen la secuencia. |
| "observada a través de un mecanismo de negociación" | C5 | Parte de la estructura observable la genera el mercado como mecanismo, no como información. |
| "etiquetada con un timestamp cuya semántica…" | C11, C5 | Causalidad y fuga de información se deciden antes de calcular nada. |

El resto de este documento desarrolla las conexiones entre capítulos que hacen que esa lectura sea inevitable.

---

## 2. Los siete capítulos como un único argumento

No se presentan siete resúmenes. Se presentan **nueve ejes de integración**, cada uno construido a partir de al menos dos capítulos, y cada uno con una consecuencia directa sobre el orden del roadmap.

### 2.1 Eje vertebral — la factorización condicional es el problema entero

**[A]** C1 establece la identidad que sostiene todo lo demás:

$$F(r_1,\dots,r_T;\theta)=F(r_1)\prod_{t=2}^{T}F(r_t\mid r_{t-1},\dots,r_1)$$

y define la hipótesis nula del proyecto: si $F(r_t\mid \text{pasado})=F(r_t)$, los retornos son temporalmente independientes y **no predecibles**.

**[A]** Cada capítulo posterior es una forma de preguntar *en qué aspecto* la condicional difiere de la marginal:

| Capítulo | Aspecto de $F(r_t\mid \mathcal F_{t-1})$ que interroga |
|---|---|
| C2 | La media condicional $\mu_t$, restringida a formas lineales |
| C3 | La varianza condicional $\sigma_t^2$ |
| C4 | La forma funcional de $\mu_t$ y $\sigma_t^2$ cuando no es lineal |
| C7 | Los cuantiles y el comportamiento de cola de la condicional |
| C11 | La existencia de un estado latente $s_t$ que gobierne $\mathcal F_{t-1}$ |
| C5 | Qué parte de la diferencia observada la genera el mecanismo de negociación y no el proceso |

**[B] Consecuencia de orden.** Los capítulos no son técnicas alternativas: son **coordenadas distintas del mismo objeto**. Por lo tanto el roadmap debe recorrerlas por dependencia informacional (qué debo saber antes de poder interpretar lo siguiente), no por orden de aparición en el libro.

**[B] Consecuencia de método.** La factorización impone que toda estimación sea **secuencial y causal**: cualquier estadístico calculado en $t$ que use $y_{t+k}$ no estima ninguna cantidad definida en este marco. Esto vale ya en la fase de caracterización, no sólo en validación de modelos.

---

### 2.2 Primer eje — dirección y magnitud son dos preguntas, no una

Éste es el eje más importante para un dataset intradiario.

**[A] C1** documenta que la media de un retorno diario es estadísticamente indistinguible de cero (IBM: $t=1.51$, $p=0.13$ con $T=9{,}845$), y simultáneamente que las variabilidades "aparecen en clusters".
**[A] C2** muestra que la ACF de $r_t$ puede ser prácticamente nula mientras la ACF de $|r_t|$ es significativa **hasta 300 rezagos** en índices CRSP diarios.
**[A] C3** formaliza esa asimetría: $r_t=\mu_t+a_t$ con $a_t=\sigma_t\epsilon_t$ y $\epsilon_t$ iid; la persistencia vive en $\sigma_t$, no en $\epsilon_t$. Ejemplo Intel mensual: $Q(12)=18.26$ ($p=0.11$) sobre $r_t$ frente a $Q(12)=89.85$ ($p=5.3\times10^{-14}$) sobre $a_t^2$.

**[A] Integración exacta.** $\text{ACF}(r_t)\approx 0$ y $\text{ACF}(|r_t|)>0$ **no se contradicen**: la primera afirma ausencia de dependencia lineal en la media; la segunda, presencia de dependencia en la escala. Ninguna de las dos dice nada sobre dependencia no lineal en la media, que es la pregunta de C4.

**[A] C3 refuerza el corte:** modelar volatilidad **no** es predecir dirección. En la propia estructura del modelo, $\epsilon_t$ es iid por construcción.

**[B] Consecuencia para el roadmap.** El análisis de dependencia debe ejecutarse **en dos ramas paralelas y explícitamente separadas** (dirección / magnitud), con criterios de interpretación distintos y umbrales de relevancia distintos. Fusionarlas en un solo "análisis de autocorrelación" es el error que hace que un hallazgo en magnitud se lea como señal direccional.

**[B] Consecuencia de expectativa.** Es razonable *esperar* más estructura detectable en magnitud que en dirección, pero eso es una expectativa trasladada de acciones e índices accionarios de EE.UU. a frecuencia diaria/mensual. Para MNQ intradiario es `HIPÓTESIS EMPÍRICA` (TH16, TH19), no hallazgo heredado.

---

### 2.3 Segundo eje — la jerarquía de dependencia y por qué ningún nivel implica el siguiente

**[A]** Combinando C1, C2, C3, C4 y C7 se obtiene una jerarquía estricta, donde cada nivel implica el anterior pero **nunca** al revés:

```
Nivel 1  Dependencia lineal en la media            ACF(r) != 0                     [C2]
   ⊂
Nivel 2  Dependencia estadística de cualquier tipo  F(r|pasado) != F(r)            [C1, C3, C4, C7]
            2a  en la escala          ACF(|r|), ACF(r²) != 0        [C3]
            2b  en la forma funcional  tests de no linealidad        [C4]
            2c  en los cuantiles/colas Q_τ(r|X) variable, θ<1        [C7]
   ⊂
Nivel 3  Estabilidad de esa dependencia            se sostiene entre subperíodos   [C2 §submuestras, C11]
   ⊂
Nivel 4  Predictibilidad fuera de muestra          supera un benchmark, OOS        [C4]
   ⊂
Nivel 5  Utilidad económica neta de costos         supera tick + comisión + slippage [ninguno: fuera de Tsay]
```

**[A] Evidencia de que los saltos no son automáticos, dentro del propio Tsay.**
- C4 detecta no linealidad significativa por BDS en retornos mensuales de IBM y de dos índices CRSP, y sin embargo la red neuronal 8-4-1 entrenada sobre esa misma serie de IBM **no supera** un random walk con deriva (tasa de acierto 0.58, $\chi^2=0.137$, $p=0.71$); y la red 3-2-1 gana in-sample (6.56 vs 6.61) y **empata** OOS (91.74 vs 91.70 del AR(1) y 91.85 del RW). Es la demostración, dentro del libro, de que Nivel 2 no implica Nivel 4.
- C2 encuentra un AR(3) con coeficientes significativos al 1% y $R^2=2.46\%$ — Nivel 1 sin magnitud relevante.
- C11 permite que un coeficiente varíe en el tiempo (CAPM dinámico sobre GM) y la máxima verosimilitud estima $\hat\sigma_\eta=4.9\times10^{-5}$: **dar libertad no produce dinámica**.

**[B] Consecuencia para el roadmap.** Los Niveles 4 y 5 **no pertenecen a esta fase**. La caracterización se detiene en el Nivel 3. Todo análisis que pretenda saltar al Nivel 4 es diseño de ML disfrazado de análisis exploratorio, y debe rechazarse en la auditoría del roadmap.

**[B] Consecuencia de gobernanza.** Cada resultado positivo debe declarar **en qué nivel de la jerarquía vive**. Un hallazgo sin nivel declarado es un hallazgo que se leerá un nivel más arriba del que corresponde.

---

### 2.4 Tercer eje — escala versus forma: de dónde vienen las colas pesadas

**[A] C1** documenta exceso de curtosis alto y generalizado (S&P diario simple: 22.81; Citigroup: 55.25) y presenta la familia de **mixturas de escala de normales** como la que reconcilia colas pesadas con momentos finitos.
**[A] C3** cierra el círculo: un GARCH(1,1) **con innovaciones gaussianas** genera exceso de curtosis por sí solo,
$$K_a^{(g)}=\frac{6\alpha_1^2}{1-2\alpha_1^2-(\alpha_1+\beta_1)^2},$$
y si $\alpha_1=0$ entonces $K_a^{(g)}=0$. Es decir: **las colas pesadas marginales no prueban colas pesadas condicionales**.
**[A] C3** matiza en la dirección opuesta: "el comportamiento de cola de los modelos GARCH sigue siendo demasiado corto incluso con innovaciones Student-t estandarizadas". Ni todo es heterocedasticidad, ni todo es forma.
**[A] C7** provee el instrumento que separa ambos orígenes: $r_t=\mu_t+\sigma_t\epsilon_t$ permite preguntar si un extremo observado se debe a un $\epsilon_t$ grande (forma) o a un $\sigma_t$ grande (escala), y de hecho usa la volatilidad GARCH como **variable explicativa de los parámetros de cola** (Ec. 7.39), con efecto significativo sobre IBM.

**[A] Integración exacta.** La pregunta correcta después de C1 no es "¿qué distribución tienen los retornos?" sino:

> **¿qué fracción del exceso de curtosis marginal desaparece al estandarizar por una estimación causal de la volatilidad?**

Es una sola medición, barata, sin ajustar ningún modelo paramétrico, y **reordena todo lo que viene después**:

- si desaparece casi todo → el fenómeno es heterocedasticidad; la agenda es la dinámica de $\sigma_t$ y **EVT es probablemente innecesaria** para caracterizar;
- si sobrevive una parte grande → hay colas condicionales genuinas; entonces cuantiles condicionales y, sólo entonces, quizá EVT.

**[B] Consecuencia de orden.** Esta medición debe ocurrir **antes** de decidir si se estudian colas, y **antes** de elegir cualquier distribución condicional. Es el nodo de bifurcación más rentable de todo el roadmap (TH22).

**[B] Consecuencia sobre "outliers".** C7 lo enuncia sin ambigüedad: extremo estadístico ≠ error de datos, y tampoco a la inversa. Descartar extremos automáticamente sesga toda estimación de cola hacia el optimismo; modelar cualquier extremo como evento real ignora contaminación por microestructura, roll y bad ticks. La única salida es un **triage documentado, por observación**, ejecutado en la etapa de integridad y no en la de estadística.

---

### 2.5 Cuarto eje — determinismo de reloj versus dinámica estocástica

Este eje es el que más cambia el **orden** del roadmap respecto de la secuencia natural del libro.

**[A] C5** documenta el patrón diurno: la ACF del número de transacciones por intervalos de 5 minutos exhibe periodicidad exacta igual a la longitud de la jornada (78 en NYSE), con forma en U.
**[A] C3** advierte, en su listado de errores, contra "atribuir a dinámica ARCH lo que es estacionalidad intradía determinista".
**[A] C2** advierte que Ljung–Box debe modificarse en presencia de estacionalidad y que hay que mirar los rezagos múltiplos del período.
**[A] C7** exige estacionariedad (estricta, para el índice extremal) antes de aplicar EVT; un patrón diurno de volatilidad la viola de plano.

**[A] Integración exacta.** Un ciclo determinista ligado al reloj **produce autocorrelación en $|r_t|$ y $r_t^2$ con picos en múltiplos de la jornada, produce colas marginales gruesas por mezcla de regímenes horarios, y produce "clustering de extremos" que es simplemente concentración horaria**. Los tres efectos son indistinguibles de dinámica estocástica genuina si no se controla primero el reloj.

**[B] Consecuencia de orden — decisión metodológica central de este roadmap.**

> **El perfil intradía determinista debe caracterizarse ANTES de analizar dependencia, volatilidad, cuantiles y extremos, y todo análisis posterior debe reportarse en dos versiones: sobre la variable cruda y sobre la variable ajustada por el perfil.**

No se propone "desestacionalizar y olvidar el crudo": la comparación entre ambas versiones **es** el resultado. Si la ACF de $r_t^2$ colapsa al ajustar, el hallazgo era el reloj; si sobrevive, hay dinámica.

**[B] Advertencia contra el traslado literal.** La forma en U de NYSE **no se traslada** a un instrumento que cotiza casi 24 horas en Globex. El perfil de MNQ debe medirse, no asumirse (TH14). Y la segmentación de sesión debe derivarse del perfil medido, no importarse de una convención previa.

---

### 2.6 Quinto eje — el mecanismo de negociación como generador de estructura

**[A] C5** demuestra, con el modelo de Roll, que el propio mecanismo de cotización genera dependencia: con $P_t=P_t^*+I_t(S/2)$ e $I_t=\pm1$ equiprobable, resulta $\rho_1=-0.5$ y $\rho_j=0$ para $j>1$; con valor fundamental de camino aleatorio, $\rho_1=-\frac{S^2/4}{S^2/2+\sigma^2}\le 0$. Evidencia empírica sobre IBM: ACF de la serie direccional con pico único en el rezago 1 de $-0.389$.
**[A] C5** demuestra además que el **trading no sincrónico** produce $\text{Cov}(r_t^o,r_{t-1}^o)=-\mu^2\pi^j$: autocorrelación negativa espuria **aunque la serie verdadera sea independiente**, siempre que $\mu\ne 0$.
**[A] C2** enseña a detectar autocorrelación; **C5 provee dos explicaciones alternativas que no son información**.

**[A] Integración exacta.** Ante una ACF de rezago 1 negativa y significativa en barras de 1 minuto de MNQ existen **al menos tres explicaciones mutuamente compatibles**: (i) huella residual de bid–ask bounce transmitida por el Close (última transacción del minuto); (ii) efecto de no sincronía en tramos de baja actividad; (iii) reversión económica genuina. **Con OHLCV Last puro no se pueden separar de forma concluyente.**

**[B] Consecuencia — el guardrail más duro de esta síntesis.** Un hallazgo de dependencia lineal de corto rezago en datos intradiarios **no puede reportarse como "predictibilidad" sin haber intentado y documentado la refutación por microestructura**. Los instrumentos disponibles con OHLCV son indirectos pero reales: comportamiento del efecto al agregar la frecuencia (1→5→10 min), comportamiento por segmento horario (la no sincronía debería ser peor en tramos ilíquidos), comparación de la magnitud implícita con el tick, y consistencia con la fracción de barras sin cambio.

**[B] Consecuencia de escala.** Aunque el efecto fuese real, el criterio de tamaño es previo al de significancia: si el movimiento esperado implícito es una fracción del tick, el hallazgo es estadístico y no operativo. Ese criterio se aplica **en la fase de caracterización**, como filtro de relevancia, no como evaluación económica (que está fuera de alcance).

---

### 2.7 Sexto eje — información versus representación

**[A] C11** demuestra que el local trend model **es** un ARIMA(0,1,1), que **es** el suavizado exponencial simple, con conversión explícita $(1+\theta^2)\sigma_a^2=2\sigma_e^2+\sigma_\eta^2$ y $\theta\sigma_a^2=\sigma_e^2$; que un state-space lineal invariante con estado $m$-dimensional produce observaciones ARMA($m,m$); y textualmente, que "based on the data alone, the decision of using ARIMA models or linear state-space models is not critical".
**[A] C11** demuestra además que en estado estacionario el filtro de Kalman del local trend model **es una EMA** con $\alpha=K_\infty$; verificación numérica sobre Alcoa: $K_\infty\approx 0.142 = 1-\hat\theta = 1-0.858$.
**[A] C11** advierte sobre **no unicidad**: "there are infinitely many ways to decompose an observed time series into unobserved components… care must be exercised in interpreting the estimated components".
**[A] C3** aporta el análogo en volatilidad: GARCH es formalmente un ARMA sobre $a_t^2$.

**[A] Integración exacta.** Reescribir la misma serie en otra representación es una **transformación determinista de la información ya disponible**. Cambia la facilidad de extracción, no el contenido informativo de $\{y_1,\dots,y_t\}$.

**[B] Consecuencia — el marco de redundancia.** OHLCV admite miles de transformaciones (retornos, momentum, ROC, EMA, MACD, RSI, ATR, Bollinger, estado de Kalman, volatilidad de rango). Todas son funciones del mismo vector de cinco columnas. **La cantidad de transformaciones no es la cantidad de información.** La distinción operativa que debe establecerse ahora, y aplicarse después, es:

| Categoría | Definición operativa | Ejemplo del propio Tsay |
|---|---|---|
| **Nueva información** | Requiere datos que no están en el OHLCV histórico observado hasta $t$ | Bid/Ask, order flow, otro instrumento, dato macro |
| **Nueva representación** | Función determinista de OHLCV hasta $t$ | Kalman(local trend) ≡ EMA ≡ ARIMA(0,1,1) [C11]; GARCH ≡ ARMA sobre $a_t^2$ [C3] |
| **Información futura (leakage)** | Función que involucra $y_{t+k}$ | Smoothing $s_{t\mid T}$ [C11]; normalización global; regla de roll definida ex post [C1] |

**[B] Consecuencia de método.** La redundancia debe **medirse**, no argumentarse: correlación entre transformaciones, dimensión efectiva del conjunto, $R^2$ de cada transformación regresada sobre las demás. Es un análisis de la *representación*, no de ML, y por tanto pertenece a esta fase. Lo que **no** pertenece a esta fase es seleccionar features.

**[B] Consecuencia sobre Kalman.** El default correcto es **no ejecutar** state-space. C11 no autoriza a postular un "precio verdadero" detrás del precio observado: el precio es el número al que se opera. La cadena "microestructura ⟹ el precio es ruidoso ⟹ existe un precio limpio ⟹ Kalman lo recupera" no está en Tsay y debe rechazarse explícitamente.

---

### 2.8 Séptimo eje — causalidad temporal y fuga de información

**[A] C11** distingue con precisión tres objetos: *filtering* ($\mu_t\mid F_t$), *prediction* ($\mu_{t+h}\mid F_t$) y *smoothing* ($\mu_t\mid F_T$, $T>t$), y muestra que el suavizado se calcula con "a weighted linear combination of the innovations $\{v_t,\dots,v_T\}$" — literalmente posteriores a $t$.
**[A] C11** identifica una segunda vía, más sutil: aunque la recursión del filtro sea causal, los parámetros $\hat\sigma_e,\hat\sigma_\eta$ se estiman por MV sobre toda la muestra; y en el local trend model la secuencia entera de ganancias **está determinada únicamente por esos parámetros**. El filtro "causal" hereda el futuro por la vía de sus hiperparámetros.
**[A] C1** identifica una tercera vía en futuros: el factor de ajuste del contrato continuo depende de rolls posteriores. El test operativo es de invariancia: si el valor del estadístico cambiaría al cambiar el factor de ajuste, incorpora información futura; si no, no.

**[A] Integración exacta.** Las tres vías tienen la misma firma clínica: **no dejan marca visible en el código, producen resultados buenos pero no absurdos, y la ventaja es genuina en el sentido retrospectivo** — por eso convencen.

**[B] Consecuencia — regla dura para esta fase.** Toda cantidad calculada en $t$ durante la caracterización debe pasar el test de reconstrucción: recalcularla usando exclusivamente datos hasta $t$ debe dar exactamente el mismo número. Esto aplica a estimaciones de volatilidad, a estandarizaciones, a umbrales de extremos, a perfiles estacionales y a cualquier parámetro. **Un perfil intradía estimado sobre toda la muestra y aplicado a toda la muestra es aceptable como descripción, y es leakage si se usa como insumo de una evaluación predictiva** — de ahí que la distinción deba declararse en cada output.

---

### 2.9 Octavo eje — significancia, magnitud y utilidad con $n$ enorme

**[A]** Tres capítulos convergen en la misma advertencia desde ángulos distintos:
- **C2**: AR(3) con coeficientes ≈0.11 significativos al 1% y $R^2=2.46\%$. Textual: "la dependencia serial es débil, aunque sea estadísticamente significativa".
- **C5**: con 206,794 observaciones, "casi todos los t-estadísticos son grandes casi por definición del tamaño muestral".
- **C1**: Jarque–Bera de 60,921 sobre IBM diario — rechazo abrumador que, además, con clustering de volatilidad **ni siquiera está bien calibrado**, porque JB supone iid.

**[B] Consecuencia para nuestro caso.** Un dataset de 1 minuto de ~6 años puede contener del orden de $10^6$ barras. Con muestras tan grandes, muchos tests adquieren una **potencia estadística muy elevada** y pueden detectar desviaciones extremadamente pequeñas de una hipótesis nula. Por ello, un p-valor pequeño no basta para establecer que el efecto sea importante: debe evaluarse también su magnitud, incertidumbre, estabilidad y relevancia práctica.

**[B] Regla de reporte obligatoria.** Ningún resultado de esta fase se reporta como "significativo/no significativo". Cada resultado se reporta como una terna:

$$\boxed{\ (\text{estimador puntual},\ \text{intervalo por bootstrap por bloques},\ \text{traducción a unidades interpretables})\ }$$

donde "unidades interpretables" significa ticks, porcentaje de la desviación estándar del segmento, o porcentaje de la variabilidad total explicada — nunca sólo un p-valor.

**[B] Corolario sobre normalidad.** Con muestras extremadamente grandes, los tests de normalidad pueden detectar desviaciones muy pequeñas de la distribución normal y producir rechazos estadísticos aun cuando algunas diferencias tengan poca importancia práctica. Por ello, el resultado principal no debe ser simplemente "rechazar o no rechazar normalidad", sino **cuantificar la desviación**: exceso de curtosis con y sin recorte del 0.1% extremo, cuantiles empíricos frente a los normales correspondientes y forma del QQ-plot. El objetivo es medir cuán lejos está la distribución de la referencia y **dónde**.

---

### 2.10 Noveno eje — resultados negativos y condiciones de parada

**[A]** Tsay reporta resultados negativos como resultados: la red neuronal que no supera al random walk (C4), el GARCH-M cuyo parámetro de prima de riesgo no es significativo en dos ejemplos (C3), el CAPM dinámico cuya varianza de estado es "essentially constant" (C11), el modelo de Poisson homogéneo que **falla** los diagnósticos (C7).

**[A]** Y en varios puntos declara que la elección no importa: "no hay evidencia que sugiera que un enfoque supere al otro" (AIC vs BIC, C2); "the decision of using ARIMA models or linear state-space models is not critical" (C11); "we recommend that one applies several methods to gain insight into the range of VaR" — es decir, no declara ganador (C7).

**[B] Consecuencia.** Un roadmap honesto debe poder terminar diciendo *"no hay nada más que buscar por aquí"*. Esto exige **condiciones de STOP escritas antes de ejecutar**, no después de ver los resultados. El roadmap adjunto las define por rama.

**[B] Corolario de parsimonia.** Se aplica la regla: usar el análisis más simple que responda correctamente la pregunta. En la práctica, dado lo anterior:

| Pregunta | Herramienta suficiente | Herramienta que **no** hace falta salvo justificación |
|---|---|---|
| ¿Hay clustering de volatilidad? | ACF de $\lvert r\rvert$ y de $r^2$ | GARCH |
| ¿Cuánto persiste? | ACF + ventanas rodantes | Estimación de $\alpha_1+\beta_1$ |
| ¿Cómo son las colas? | Cuantiles empíricos + exceso de curtosis con/sin recorte | EVT/GPD |
| ¿Los extremos se agrupan? | Conteo de runs sobre excedencias, comparado con Poisson | Índice extremal |
| ¿Hay un nivel subyacente suave? | EMA causal | Filtro de Kalman |
| ¿La relación cambia con el contexto? | Estadísticos por segmento/subperíodo | TAR / STAR / Markov switching |

---

## 3. ¿Qué observamos realmente? Anatomía de una barra OHLCV Last de 1 minuto

### 3.1 Datos observados

`timestamp`, `Open`, `High`, `Low`, `Close`, `Volume`.

Cinco números y una etiqueta temporal. **Nada más.**

**[A] C5** aporta lo que esto significa: $H_t$ y $L_t$ son **estadísticos de orden** (máximo y mínimo) de los precios operados en el intervalo; $O_t$ y $C_t$ son **estadísticos de posición** (primero y último). Ninguno describe la trayectoria.

### 3.2 Transformaciones observables (funciones deterministas de lo anterior)

| Transformación | Interpretación | Riesgo principal |
|---|---|---|
| $r_t=\ln(C_t/C_{t-1})$ | Retorno logarítmico close-to-close | Cruza fronteras de sesión, día y roll si no se restringe |
| $R_t=C_t/C_{t-1}-1$ | Retorno simple | Aditividad de cartera, no temporal |
| $\lvert r_t\rvert$, $r_t^2$ | Proxies de magnitud/varianza | $r_t^2$ es insesgado pero muy impreciso [C3] |
| $\ln(H_t/L_t)$ | Rango logarítmico intrabarra | Subestima el rango verdadero por discreción y baja actividad [C3, C5] |
| $\ln(C_t/O_t)$ | Retorno intrabarra | No cruza fronteras; ignora el gap entre barras |
| $\ln(O_t/C_{t-1})$ | Gap entre barras | Aísla precisamente lo que $\ln(C/O)$ omite |
| Estimadores de rango (Parkinson, Garman–Klass, Rogers–Satchell, Yang–Zhang) | Volatilidad por barra | Cada uno con supuestos propios: difusión sin drift, tratamiento del gap, volatilidad constante en la ventana [C3] |
| $V_t$ y su perfil por hora | Actividad | **No es número de trades** [C5] |
| Volatilidad rodante causal, EWMA | Escala local | Elección de ventana; debe ser estrictamente causal |
| Fracción de barras con $r_t=0$; $\text{tick}/\hat\sigma$ | Resolución efectiva | Depende del segmento horario |

### 3.3 Cantidades NO observadas — no deben presentarse como hechos

| Cantidad | Estatus con OHLCV Last 1-min |
|---|---|
| Precio eficiente / "precio verdadero" $P_t^*$ | **No observable.** Construcción teórica [C5]. C11 no autoriza a postularlo. |
| Spread bid–ask | **No observable.** No hay Bid ni Ask. |
| Bid–ask bounce identificado | **No observable directamente.** Sólo inferible por su huella, y de forma no concluyente. |
| Signo/agresor de cada operación | **No observable.** Se destruye en la agregación. |
| Duración entre transacciones | **No observable.** La separación entre barras es constante por construcción; aplicar ACD sobre timestamps de barras es un error categorial [C5]. |
| Número exacto de transacciones | **No observable** salvo columna adicional. Volumen ≠ número de trades. |
| Profundidad del libro / order flow | **No observable.** |
| Secuencia intraminuto de precios | **No observable.** Dos trayectorias distintas producen el mismo OHLC. |
| Volatilidad "verdadera" $\sigma_t$ | **No observable** [C3]. Sólo proxies. |
| Estado latente / régimen de mercado | **No observable y no único** [C11, C4]. Existe dentro de una especificación, no en el mundo. |
| Cuantil condicional $Q_\tau(r_{t+h}\mid X_t)$ | **No observable.** Es una cantidad a estimar, con incertidumbre. |
| Probabilidad de movimiento futuro | **No observable.** |

### 3.4 Qué destruye la agregación en barras — y qué no

| Fenómeno de C5 | ¿Sobrevive en barras de 1 min? |
|---|---|
| Discreción del tick | **Sí, plenamente.** Restringe O/H/L/C y por tanto los retornos. |
| Patrón diurno de actividad | **Sí.** Directamente observable agregando por hora. |
| Huella de bid–ask bounce | **Posiblemente, atenuada,** vía el Close (última transacción del minuto). Magnitud desconocida → `HIPÓTESIS EMPÍRICA` TH17. |
| No sincronía | **Sí en contexto multi-instrumento**; y en tramos de muy baja actividad puede afectar la propia serie. |
| Dirección de trade, duraciones, profundidad | **No. Irrecuperables.** |
| Rango como medida de volatilidad | **Sí.** Es la variable de C5/C3 con transferencia más limpia a barras. |

---

## 4. Las cinco confusiones que esta síntesis existe para impedir

1. **Confundir ausencia de ACF con ausencia de dependencia.** C2 lo dice; C3 y C4 lo refutan empíricamente. Sólo bajo normalidad conjunta correlación cero implica independencia, y C1 rechaza normalidad de forma abrumadora.
2. **Confundir dependencia en magnitud con señal direccional.** C3 lo separa formalmente: $\epsilon_t$ es iid por construcción.
3. **Confundir estructura generada por el mercado con información sobre el mercado.** C5: el bid–ask bounce produce $\rho_1<0$ "predecible" e inejecutable; la no sincronía produce autocorrelación negativa **aunque la serie verdadera sea independiente**.
4. **Confundir determinismo de reloj con dinámica estocástica.** El patrón diurno genera ACF en $r^2$, colas gruesas y agrupamiento de extremos por sí solo.
5. **Confundir representación con información.** C11: Kalman(local trend) ≡ EMA ≡ ARIMA(0,1,1). Reescribir no agrega.

Y una sexta, transversal a todas: **confundir significancia con magnitud** cuando $n\sim 10^6$.

---

## 5. Cómo cambia esta síntesis nuestra forma de interpretar un dataset OHLCV intradiario

Resumido en siete reformulaciones. Cada una reemplaza una lectura ingenua por una lectura correcta.

| Lectura ingenua | Lectura después de Tsay |
|---|---|
| "El Close es el precio del minuto." | El Close es la última transacción del minuto, discretizada por el tick, potencialmente posicionada en un lado del spread; su semántica de disponibilidad depende de si el timestamp marca apertura o cierre de barra. |
| "El volumen mide actividad." | El volumen mide contratos operados. La actividad (número de operaciones) **no es observable**. Un trade de 100 y 100 trades de 1 son idénticos en esta columna. |
| "High−Low mide la volatilidad del minuto." | Es un estimador de volatilidad con supuestos explícitos (difusión sin drift, tratamiento del gap) y con **sesgo negativo** por discreción y por baja frecuencia de negociación; su magnitud depende del segmento horario. |
| "Los outliers hay que limpiarlos." | Los extremos son el fenómeno, no el ruido. Se hace triage documentado por observación: error de datos / artefacto de roll / evento real. Descartarlos automáticamente sesga la caracterización. |
| "Hay que testear si la serie es normal." | Con una muestra extremadamente grande, los tests de normalidad pueden detectar desviaciones muy pequeñas y producir rechazos que no necesariamente implican una diferencia importante en términos prácticos. Lo informativo es *cuánto* y *dónde* difiere la distribución, y qué fracción de esa diferencia puede atribuirse a heterocedasticidad. |
| "El histórico es la muestra." | El histórico es una mezcla de regímenes horarios, contratos, y condiciones de mercado de ~6 años. Toda propiedad debe reportarse con su versión por subperíodo, o no está reportada. |
| "Más indicadores = más información." | Todos los indicadores son funciones del mismo OHLCV. La redundancia se mide; no se argumenta. |

---

## 6. Matriz Tsay → pregunta empírica sobre MNQ

Concisa y sin redundancias. `THxx` remite al backlog consolidado.

| Concepto Tsay | Pregunta empírica para MNQ | Variable | Test mínimo | Test avanzado si se justifica |
|---|---|---|---|---|
| Retorno libre de escala [C1] | ¿Cuánto difieren $r_t$ y $R_t$, y dónde se concentra la diferencia? | $r_t$, $R_t$ | $\lvert r_t-R_t\rvert$ por decil de volatilidad | — |
| Agregación temporal aditiva [C1] | ¿Escala $\text{Var}(r_t[h])$ como $h\cdot\text{Var}(r_t)$? | $r_t[h]$ | Varianza vs $h$ en log-log | Análisis de memoria larga |
| Discreción por tick [C1, C5] | ¿Cuál es la resolución efectiva por segmento horario? | $r_t$, tick, $\hat\sigma$ | Fracción de $r_t=0$; $\text{tick}/\hat\sigma$; histograma en múltiplos de tick | — |
| Media ≈ 0 [C1] | ¿Es el drift por barra distinguible de cero? | $r_t$ | t-test con e.e. HAC + bootstrap por bloques | — |
| Colas pesadas [C1] | ¿Cuánta curtosis sobrevive a estandarizar por volatilidad causal? | $r_t$, $z_t=r_t/\hat\sigma_{t-1}$ | Curtosis con/sin recorte, antes/después | Ajuste de distribución condicional |
| Estacionariedad débil [C2] | ¿Son estables los momentos entre submuestras? | $r_t$, $\lvert r_t\rvert$ | Momentos por año/segmento (receta de submuestras de Tsay) | ADF sobre log-precio de contrato individual |
| ACF y Ljung–Box [C2] | ¿Es la ACF de $r_t$ distinguible de cero, y de qué tamaño? | $r_t$ | ACF con bandas de Bartlett; $Q(m)$, $m\approx\ln T$ | PACF; modelo AR de referencia |
| Autocorrelación de microestructura [C5] | ¿El $\rho_1$ observado sobrevive a la agregación y al control por liquidez? | $r_t$ a 1/5/10 min | $\rho_1$ por frecuencia y por segmento; magnitud en ticks | — |
| No sincronía [C5] | ¿Se concentra el efecto en tramos de baja actividad? | $r_t$, $V_t$ | $\rho_1$ por decil de volumen | — |
| Clustering de volatilidad [C1, C3] | ¿Hay dependencia en magnitud, y cuánto dura? | $\lvert r_t\rvert$, $r_t^2$, $\ln(H/L)$ | ACF hasta ≥ 2 jornadas; $Q(m)$; LM de Engle | GARCH; comparación de persistencia |
| Estacionalidad vs ARCH [C3, C5] | ¿Sobrevive la ACF de $r_t^2$ al ajustar por perfil intradía? | $r_t^2$ crudo vs ajustado | ACF antes/después | — |
| Memoria larga [C2, C3] | ¿Decae la ACF de $\lvert r_t\rvert$ polinomial o exponencialmente? | $\lvert r_t\rvert$ | ACF hasta cientos de rezagos, escala log-log | Estimación de $d$ |
| Efecto apalancamiento [C3] | ¿Responde la volatilidad de MNQ asimétricamente al signo del shock? | $\hat\sigma$, signo de $r_{t-1}$ | Volatilidad media condicionada al signo | EGARCH/TGARCH |
| Eficiencia de estimadores de rango [C3] | ¿Se replican las ganancias de eficiencia de OHLC sobre close-to-close? | Parkinson, GK, RS, YZ vs $r^2$ | Comparación contra RV de referencia | — |
| Volatilidad realizada y ruido [C3, C5] | ¿A qué frecuencia domina el ruido de microestructura? | RV a varias frecuencias | Volatility signature plot | — |
| No linealidad [C4] | ¿Queda estructura tras filtrar media, volatilidad y estacionalidad? | Residuos estandarizados | BDS y F-test de Tsay sobre residuos | TAR/STAR |
| Cuantiles condicionales [C7] | ¿Cambia la distribución sólo en escala o también en forma? | $z_t$ por segmento y por estado de volatilidad | Cuantiles de $z_t$ comparados entre grupos | Regresión cuantílica |
| Asimetría de colas [C7] | ¿Difieren las colas izquierda y derecha? | $r_t$ | Cuantiles simétricos comparados; curtosis por cola | Índice de cola por lado |
| Clustering de extremos [C7] | ¿Se agrupan los extremos más allá de lo esperado bajo independencia? | Excedencias sobre umbral | Distribución de runs vs Poisson | Índice extremal $\theta$ |
| EVT / GPD [C7] | ¿Aporta EVT sobre cuantiles empíricos para caracterizar? | Excesos sobre umbral | Mean excess plot; sensibilidad al umbral | POT/GPD con diagnósticos |
| Invariancia del índice de cola [C7] | ¿Es el índice de cola estable entre frecuencias? | $r_t$ a 1/5/10 min | Comparación de índice de cola | — |
| Filtering vs smoothing [C11] | ¿Toda transformación es reconstruible con datos hasta $t$? | Todas | Test de reconstrucción causal | — |
| Kalman ≡ EMA [C11] | ¿Aporta un estado filtrado sobre una EMA calibrada? | Estado vs EMA | Correlación y diferencia entre ambos | Ajuste state-space con MV |
| No unicidad de estados [C11] | ¿Existe una hipótesis latente concreta y falsable? | — | Formulación escrita previa | — |
| Redundancia de representaciones [C11] | ¿Cuántas dimensiones efectivas hay en el conjunto de transformaciones? | Conjunto de transformaciones | Matriz de correlación; $R^2$ de cada una sobre las demás | Análisis de componentes |

---

## 7. Matriz final de conocimiento

Estados usados: `CONFIRMED THEORY`, `DATA FACT`, `EMPIRICAL TEST REQUIRED`, `NOT OBSERVABLE WITH CURRENT DATA`, `CONDITIONAL ANALYSIS`, `DEFERRED`.

| Tema | Qué sabemos por Tsay | Qué sabemos de nuestros datos | Qué debemos comprobar | Limitación | Estado |
|---|---|---|---|---|---|
| Instrumento y frecuencia | — | MNQ, barras de 1 min, Last, ~2020–2026, TZ declarada America/New_York — **declarado, no verificado** | Rango exacto, número de barras, unicidad y monotonicidad de timestamps | La documentación del proyecto marca estos ítems como preguntas abiertas | EMPIRICAL TEST REQUIRED |
| Invariantes de la barra | $H,L$ son estadísticos de orden [C5] | Nada verificado | $H\ge\max(O,C)$, $L\le\min(O,C)$, $H\ge L$, precios $>0$, $V\ge0$, múltiplos de tick | Ninguna | EMPIRICAL TEST REQUIRED |
| Semántica del timestamp | Disponibilidad de información define causalidad [C11] | Nada verificado | ¿Inicio o cierre de barra? ¿Cuándo está disponible cada campo? | Requiere documentación del proveedor, no sólo estadística | EMPIRICAL TEST REQUIRED |
| Barras sin actividad | — | Nada verificado | ¿Ausencia de fila, volumen 0, o forward-fill ($O=H=L=C$)? | Los tres se ven casi igual en un DataFrame [C11] | EMPIRICAL TEST REQUIRED |
| Calendario y DST | — | Nada verificado | Cobertura por jornada, feriados, medias sesiones, saltos de DST | Ninguna | EMPIRICAL TEST REQUIRED |
| Contrato continuo y roll | Ratio preserva retornos; aditivo preserva puntos; ninguno ambos [C1] | No verificado si la serie es continua ni con qué método | Detección de rolls, magnitud del salto, invariancia a escala de cada estadístico | Regla de roll definida ex post = no causal | EMPIRICAL TEST REQUIRED |
| Definición de retorno | Log aditivo temporal; simple aditivo en cartera [C1] | — | Efecto de reglas de no-cruce (día/sesión/roll) sobre la serie | — | EMPIRICAL TEST REQUIRED |
| Resolución por tick | La discreción invalida continuidad a alta frecuencia [C1, C5] | Tick de MNQ conocido por especificación de contrato | $\text{tick}/\hat\sigma$ y fracción $r_t=0$ por segmento horario | — | EMPIRICAL TEST REQUIRED |
| Distribución marginal | Colas pesadas generalizadas; densidad con pico alto y colas gruesas [C1] | — | Momentos, cuantiles, curtosis con/sin recorte, QQ | Con $n\sim10^6$ los tests de normalidad no informan | CONFIRMED THEORY + EMPIRICAL TEST REQUIRED |
| Drift | Media diaria indistinguible de cero [C1] | — | t-test HAC + bootstrap por bloques sobre $r_t$ | Traslado desde acciones/índices diarios | EMPIRICAL TEST REQUIRED |
| Estacionalidad intradía | Patrón diurno en U documentado en NYSE [C5] | — | Perfil por minuto del día de $V$, $\lvert r\rvert$, rango, extremos | Forma en U de NYSE **no** trasladable a un mercado de 24h | EMPIRICAL TEST REQUIRED |
| Dependencia lineal | ACF ≈ 0 en retornos; significancia ≠ magnitud [C2] | — | ACF con bandas de Bartlett, $Q(m)$, por segmento y subperíodo | Bandas $1/T$ por defecto asumen iid | EMPIRICAL TEST REQUIRED |
| Origen del $\rho_1$ | Roll: $\rho_1=-0.5$; no sincronía: $-\mu^2\pi^j$ [C5] | — | Comportamiento por frecuencia, por segmento, magnitud en ticks | **No separable de forma concluyente sin Bid/Ask ni ticks** | NOT OBSERVABLE WITH CURRENT DATA (atribución) + EMPIRICAL TEST REQUIRED (huella) |
| Clustering de volatilidad | Documentado y persistente; ACF de $\lvert r\rvert$ significativa hasta 300 rezagos [C1, C2, C3] | — | ACF de $\lvert r\rvert$, $r^2$, rango; crudo vs ajustado por estacionalidad | Estacionalidad puede simular ARCH | CONFIRMED THEORY + EMPIRICAL TEST REQUIRED |
| Origen de las colas | GARCH gaussiano genera curtosis; pero GARCH-t sigue teniendo colas cortas [C3] | — | Fracción de curtosis que sobrevive a estandarizar por $\hat\sigma$ causal | — | EMPIRICAL TEST REQUIRED (nodo de bifurcación) |
| Modelo paramétrico de volatilidad | GARCH ≡ ARMA sobre $a_t^2$ [C3] | — | Sólo si diagnósticos simples resultan insuficientes | Complejidad sin pregunta que la exija | CONDITIONAL ANALYSIS |
| Efecto apalancamiento | Documentado en acciones; mecanismo corporativo ausente en otras clases [C3] | — | Asimetría de $\hat\sigma$ según signo del shock previo | Traslado desde acciones no autorizado | EMPIRICAL TEST REQUIRED |
| No linealidad | Rechazar linealidad ≠ predictibilidad OOS [C4] | — | BDS/F sobre residuos ya filtrados de media, volatilidad y estacionalidad | Potencia excesiva con $n$ grande; BDS no identifica la fuente | CONDITIONAL ANALYSIS |
| Regímenes | TAR usa umbral observable; Markov switching, estado latente; etiquetas se asignan a posteriori [C4] | — | Nada en esta fase | Un régimen estimado no prueba que exista | DEFERRED |
| Cuantiles condicionales | Cuantil empírico no extrapola; varianza asintótica explota en la cola [C7] | — | Cuantiles de $z_t$ por segmento y por estado de volatilidad | — | EMPIRICAL TEST REQUIRED |
| Extremos: triage | Extremo ≠ error de datos, y viceversa [C7] | — | Clasificación documentada observación por observación | — | EMPIRICAL TEST REQUIRED |
| Clustering de extremos | $\hat\theta\approx0.82$ en IBM pese a ACF≈0; ignorarlo subestima el VaR ~7% [C7] | — | Distribución de runs de excedencias vs Poisson | Índice extremal requiere estacionariedad estricta | EMPIRICAL TEST REQUIRED |
| EVT / GPD | $\xi$ varía 63% entre umbrales 2% y 3%; e.e. hasta 70% del valor [C7] | — | Sólo si los cuantiles empíricos resultan insuficientes | Muy sensible a umbral y a tamaño de bloque | CONDITIONAL ANALYSIS |
| Estabilidad temporal | Receta de submuestras [C2]; no unicidad y no estacionariedad [C11] | ~6 años de regímenes heterogéneos | Toda propiedad por año/subperíodo/rodante | Cambio estadístico ≠ régimen económico identificado | EMPIRICAL TEST REQUIRED |
| Estados latentes | Kalman(local trend) ≡ EMA ≡ ARIMA(0,1,1); infinitas descomposiciones válidas [C11] | — | Sólo si existe una hipótesis latente concreta y falsable | El estado no es observable ni verificable | CONDITIONAL ANALYSIS (default: no ejecutar) |
| Fuga por suavizado | Smoothing usa $\{v_t,\dots,v_T\}$ [C11] | — | Test de reconstrucción causal sobre toda transformación | — | CONFIRMED THEORY |
| Fuga por parámetros | Las ganancias del filtro dependen sólo de parámetros estimados sobre toda la muestra [C11] | — | Declarar, en cada output, si los parámetros son globales o causales | — | CONFIRMED THEORY |
| Redundancia de transformaciones | Representación ≠ información [C11]; GARCH ≡ ARMA sobre $a_t^2$ [C3] | — | Correlación y dimensión efectiva del conjunto de transformaciones | — | EMPIRICAL TEST REQUIRED |
| Spread, order flow, trade sign, duraciones, profundidad, nº de trades | Todo el aparato de C5 los requiere | Ninguno disponible | Nada: no son medibles | **Irrecuperables tras la agregación en barras** | NOT OBSERVABLE WITH CURRENT DATA |
| Comparabilidad Micro / E-mini | Retornos plausiblemente comparables; volumen y microestructura no [C1] | Sólo tenemos MNQ | Requiere datos de NQ | — | DEFERRED |
| Relación con otros mercados | Lead-lag y no sincronía [C1, C5] | Sólo un instrumento | Requiere otros instrumentos | — | DEFERRED |
| Excess return / tasa libre de riesgo | Depende de qué serie se construyó [C1] | — | Requiere serie de tasas y definición de la construcción | Capítulo 6 diferido | DEFERRED |
| Costos, spread efectivo, viabilidad | Tsay los excluye por diseño | — | Fuera de la fase de caracterización | — | DEFERRED |

---

## 8. Matriz de riesgos metodológicos

| Riesgo | Origen | Cómo podría engañarnos | Prevención |
|---|---|---|---|
| **Datos erróneos** | Proveedor, feed, construcción de la barra | Un precio corrupto se convierte en el evento extremo que define la cola, la curtosis y el "hallazgo" | TDA-00 antes de todo; invariantes duros; triage documentado y reversible de cada extremo, nunca borrado silencioso |
| **Semántica del timestamp** | C11, C5 | Si el timestamp marca el inicio de la barra, todo lo calculado "en $t$" usa información de hasta 59 s en el futuro | TDA-01: verificación forense + documentación del proveedor; declarar la convención en cada output |
| **Barras rellenadas o ausentes** | Construcción del dataset | $O=H=L=C$ con volumen 0 produce $r_t=0$ artificial, deprime la volatilidad estimada y genera ACF espuria | Distinguir tres categorías (ausente / volumen cero / forward-fill) en TDA-02; reportar toda propiedad con y sin ellas |
| **Rolls de contrato** | Estructura de futuros [C1] | El salto del roll se lee como un retorno extremo; contamina colas, curtosis, volatilidad y ACF | TDA-03: detección, máscara de barras contaminadas, test de invariancia a escala de cada estadístico |
| **Regla de roll definida ex post** | C1 | El "mejor" contrato elegido retrospectivamente introduce información futura invisible | Documentar la regla como causal o marcar la serie como no apta para conclusiones causales |
| **Estacionalidad intradía** | C5, C3, C7 | Produce ACF en $r^2$, colas gruesas y agrupamiento de extremos sin ninguna dinámica estocástica detrás | TDA-05 antes de TDA-07/08/10/11; todo resultado reportado crudo **y** ajustado |
| **Microestructura sin Bid/Ask** | C5 | Un $\rho_1<0$ se lee como reversión explotable cuando puede ser el mecanismo de negociación | Refutación obligatoria: comportamiento por frecuencia, por segmento, magnitud en ticks; si no se puede separar, declararlo `NOT OBSERVABLE` |
| **Discreción del tick** | C1, C5 | Con retornos discretos, momentos, densidades y tests cambian de interpretación; movimientos "predichos" por debajo del tick no existen | Medir resolución efectiva por segmento antes de interpretar cualquier distribución |
| **No estacionariedad** | C2, C11 | Una propiedad medida sobre ~6 años describe una mezcla que no corresponde a ningún período concreto | Regla transversal: ninguna propiedad se reporta sin su versión por subperíodo |
| **Clustering de volatilidad** | C1, C3 | Invalida los supuestos iid de JB, Ljung–Box, BDS y EVT clásica; infla la significancia | Bootstrap por bloques; ajustar los grados de libertad; reportar tamaño de muestra efectivo |
| **Observaciones extremas** | C1, C7 | Recortarlas sesga la cola al optimismo; conservarlas sin triage mete artefactos en el modelo | Triage documentado con criterio escrito antes de mirar el efecto sobre los resultados |
| **Multiple testing / data snooping** | Transversal | Con una muestra muy grande y muchas variantes analizadas, aumenta fuertemente la probabilidad de encontrar patrones aparentemente interesantes por azar | Protocolo exploración/confirmación (§ roadmap G2): hipótesis preregistradas, número acotado de variantes, holdout temporal reservado |
| **Tests demasiado potentes con $n$ enorme** | C2, C5 | Una muestra enorme puede hacer detectables desviaciones estadísticamente muy pequeñas, por lo que un p-valor bajo puede exagerar la importancia aparente de un efecto | Prohibición de reportar sólo p-valores; terna obligatoria (estimador, intervalo, unidad interpretable) |
| **Look-ahead por normalización** | C1, C11 | Estandarizar con estadísticos globales filtra el futuro hacia el pasado | Toda estandarización usada en contexto predictivo debe ser causal; declarar el estatus en cada output |
| **Fuga por suavizado** | C11 | $s_{t\mid T}$ usa innovaciones posteriores; el resultado es bueno-pero-no-absurdo y por eso convence | Prohibición de smoothing como insumo de cualquier evaluación causal; permitido sólo como análisis retrospectivo etiquetado |
| **Fuga por parámetros** | C11 | La recursión es causal pero sus hiperparámetros vienen de toda la muestra | Declarar el origen de todo parámetro; test de reconstrucción causal |
| **Transformaciones redundantes** | C11, C3 | Mil indicadores se leen como mil fuentes; infla el espacio de búsqueda y por tanto los falsos positivos | Medir dimensión efectiva antes de ampliar el conjunto |
| **Sobreinterpretación de estados latentes** | C11, C4 | Un estado estimado se lee como un régimen descubierto; hay infinitas descomposiciones válidas | Exigir hipótesis latente escrita y falsable antes de ajustar; prohibir etiquetas económicas a posteriori |
| **Generalización desde los ejemplos de Tsay** | Transversal | Todos los ejemplos son acciones, índices y FX de EE.UU., diarios o mensuales, 1926–2008. Ninguno es un futuro ni intradiario | Toda expectativa heredada entra al backlog como hipótesis, nunca como hallazgo |
| **Confundir descriptivo con predictivo** | C4, C3, C11 | Un buen ajuste in-sample se lee como capacidad predictiva | Esta fase se detiene en el Nivel 3 de la jerarquía; el Nivel 4 pertenece a otra fase |
| **Herencia de decisiones históricas** | Repositorio | Reutilizar targets, horarios o ventanas previas convierte supuestos en resultados | Únicos supuestos admitidos: MNQ, OHLCV Last, 1 min, ~2020–2026, TZ declarada. Todo lo demás se analiza o queda abierto |

---

## 9. Qué queda abierto tras esta síntesis

**PREGUNTA ABIERTA 1.** Cuál es la semántica exacta del timestamp y de la construcción de la barra. No es resoluble estadísticamente en solitario: requiere documentación del proveedor. Es el mayor riesgo latente porque contamina *todo* lo posterior de forma silenciosa.

**PREGUNTA ABIERTA 2.** Si la serie es un contrato continuo y con qué método. Determina si los retornos que calculemos son retornos de mercado.

**PREGUNTA ABIERTA 3.** Si la huella de bid–ask bounce en barras de 1 minuto es separable de la reversión económica genuina con OHLCV Last. La respuesta más probable, a la luz de C5, es **no de forma concluyente**; en ese caso corresponde declarar el límite en vez de simular haberlo resuelto.

**PREGUNTA ABIERTA 4.** Si el índice extremal y la EVT clásica son siquiera aplicables a una serie con patrón diurno fuerte, dado que requieren estacionariedad. Puede que la respuesta correcta sea que la caracterización de colas deba hacerse **dentro de segmentos horarios homogéneos**, o no hacerse.

**PREGUNTA ABIERTA 5.** Si existe alguna cantidad latente concreta que justifique una representación state-space y que no se responda con una EMA causal. El default, según C11, es que no.

**PREGUNTA ABIERTA 6.** Qué constituye "utilidad económica" en esta fase. Se ha usado el tick como escala de referencia mínima; eso es un filtro de relevancia, no una evaluación económica. La evaluación pertenece a otra fase.

---

## 10. Puente hacia el roadmap

De esta síntesis se derivan **siete reglas de gobernanza** y **un orden**:

**Reglas** — desarrolladas en `Tsay_OHLCV_analysis_roadmap.md` §2 (G0–G6): observabilidad declarada, causalidad estricta, exploración vs confirmación, estabilidad obligatoria, parsimonia, reporte por magnitud y no sólo por p-valor, y validez del resultado negativo.

**Orden** — derivado de las dependencias informacionales, no del índice del libro:

```
integridad física de la barra
   → semántica temporal, de sesión y de contrato
      → integridad del eje temporal
         → auditoría de rolls
            → definición y auditoría de las variables de análisis
               → perfil determinista intradía (reloj)
                  → resolución efectiva y distribución marginal
                     → dependencia en media   ≠   dependencia en magnitud
                        → escala vs forma de la distribución condicional
                           → cuantiles → [colas] → [no linealidad] → [estados latentes]
                              → estabilidad temporal (transversal + consolidación)
                                 → redundancia informacional
                                    → EMPIRICAL PROFILE OF MNQ OHLCV
```

Lo entre corchetes es **condicional**: se ejecuta sólo si el análisis previo genera la pregunta que lo justifica, y se detiene según las condiciones de STOP definidas en el roadmap.

---

**DECISIÓN IRIS: NINGUNA.**
