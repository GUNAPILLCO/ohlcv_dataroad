# Tsay — Capítulo 11: State-Space Models and Kalman Filter
## Informe de estudio para el Proyecto IRIS

**Convenciones de este documento**

- **[A]** = definición, resultado, ejemplo empírico o afirmación que proviene directamente de Tsay (*Analysis of Financial Time Series*, 3ª ed., Capítulo 11).
- **[B]** = interpretación, extensión o adaptación propia hacia futuros, datos OHLCV, trading o Machine Learning. Nunca es una decisión de diseño; es una hipótesis a evaluar.
- **PREGUNTA ABIERTA** = cuestión que este capítulo no permite resolver con la información disponible.
- Cada resultado empírico [A] indica activo/serie, frecuencia, período, tamaño de muestra, variable analizada, método, resultado y limitaciones. Ningún resultado sobre Alcoa (volatilidad realizada diaria 2003–2004), General Motors (retornos mensuales 1990–2003) o Johnson & Johnson (EPS trimestral 1960–1980) se generaliza automáticamente a MNQ, a futuros ni a datos intradiarios de 1 minuto.
- Este informe continúa los estudios de los Capítulos 1, 2, 3, 4, 5 y 7. **Los Capítulos 6, 8, 9 y 10 fueron deliberadamente diferidos por menor prioridad**; no se estudian aquí ni se intenta compensar su omisión. Cuando Tsay reenvía a alguno de ellos (Apéndice B del Cap. 8 para propiedades de la normal multivariante; Cap. 9 para el origen de los datos de GM; Cap. 10 para la interpretación de la descomposición de Cholesky), se da únicamente el contexto mínimo local y se marca como tal.
- Esta fase es exclusivamente de **comprensión y adquisición de conocimiento**. No se implementa ningún Kalman Filter, no se construyen features filtradas ni suavizadas, no se modifican datasets, notebooks, targets, folds ni modelos, y no se ejecuta ningún experimento.

---

## 1. Resumen ejecutivo

Los capítulos anteriores nos enseñaron a preguntar por objetos que **se calculan a partir de lo observado**: la media condicional (Cap. 2), la varianza condicional (Cap. 3), el régimen o la no linealidad (Cap. 4), la distorsión de microestructura (Cap. 5), los cuantiles y las colas (Cap. 7). El Capítulo 11 cambia el tipo de pregunta. Ya no pregunta *qué función de los datos calculo*, sino:

> ¿Y si postulo explícitamente que existe una cantidad que **cambia con el tiempo**, que **no observo**, y de la que sólo veo **mediciones contaminadas**? ¿Cómo actualizo racionalmente mi estimación de esa cantidad cada vez que llega un dato nuevo?

**[A]** Tsay presenta el *state-space model* como "un enfoque flexible para el análisis de series temporales, especialmente para simplificar la estimación por máxima verosimilitud y para manejar valores faltantes". Nótese el énfasis: Tsay **no** lo vende como una nueva fuente de poder predictivo. Lo vende como (i) una **representación** muy general, (ii) un **algoritmo de cálculo** eficiente (el Kalman filter) y (iii) una forma natural de manejar **datos faltantes**.

El capítulo se organiza así:

1. Un modelo mínimo, el **local trend model** (o *local level model*), con exactamente dos ecuaciones y dos parámetros de varianza.
2. Sobre ese modelo mínimo, todo el aparato conceptual: **filtering, prediction, smoothing**, la **innovation** $v_t$, el **Kalman gain** $K_t$, la **incertidumbre del estado** $\Sigma_{t|t-1}$, la **inicialización** y la **estimación de parámetros**.
3. La generalización a un **modelo lineal-gaussiano general** con matrices de sistema.
4. La demostración de que muchísimos modelos conocidos (ARMA, regresión, regresión con errores ARMA, CAPM con coeficientes variables, modelos de componentes no observados) **se pueden reescribir** en forma state-space.
5. El Kalman filter y los smoothers para el caso general.
6. Missing values y forecasting.
7. Tres aplicaciones empíricas.

Los tres resultados que más importan para IRIS son, en orden de importancia:

**Primero — la advertencia de no unicidad.** **[A]** Al cerrar la aplicación de Johnson & Johnson, Tsay escribe literalmente: *"It should be noted that the estimated components in Figure 11.6 are not unique. They depend on the model specified and constraints used. In fact, there are infinitely many ways to decompose an observed time series into unobserved components. […] Thus, care must be exercised in interpreting the estimated components."* Esto es el guardrail epistemológico central del capítulo:

$$\boxed{\text{una descomposición state-space útil} \neq \text{la única descomposición verdadera}}$$

**Segundo — state-space es una representación, no información nueva.** **[A]** Tsay demuestra que el local trend model **es** un ARIMA(0,1,1), que a su vez **es** el modelo de suavizado exponencial simple del Capítulo 2; que un ARMA cualquiera admite **varias** representaciones state-space distintas (Akaike, Harvey, Aoki); y que, recíprocamente, un state-space lineal invariante en el tiempo con estado de dimensión $m$ produce observaciones que siguen un ARMA($m,m$). Y comenta explícitamente: *"In practice, what one observes is the $y_t$ series. Thus, based on the data alone, the decision of using ARIMA models or linear state-space models is not critical."* Reforzado empíricamente en el Ejemplo 11.2, donde estimar el market model de GM como state-space reproduce **exactamente** el resultado de mínimos cuadrados ordinarios, hasta el sexto decimal.

$$\boxed{\text{state-space representation} \neq \text{nueva fuente de predictibilidad}}$$

**Tercero — filtering y smoothing no son el mismo objeto, y sólo uno es causal.** **[A]** Tsay define con precisión: *filtering* estima $\mu_t \mid F_t$; *prediction* estima $\mu_{t+h}$ o $y_{t+h} \mid F_t$; *smoothing* estima $\mu_t \mid F_T$ con $T > t$. Y la fórmula del smoother lo hace inevitable: el estado suavizado es el estado filtrado más una **suma ponderada de las innovaciones futuras** $v_t, v_{t+1}, \ldots, v_T$. **[B]** Por tanto, un estado suavizado usado como feature de una decisión tomada en $t$ es *look-ahead leakage* por construcción, no por descuido de implementación.

$$\boxed{s_{t|T}\ (T>t)\ \text{puede ser excelente retrospectivamente y completamente inválido como feature causal en } t}$$

Los tres ejemplos empíricos del capítulo son, en conjunto, notablemente **desinflacionarios**:

- **Alcoa** [A]: en el logaritmo de la volatilidad realizada diaria, el ruido de medición resulta mucho mayor que la innovación del estado ($\hat\sigma_e = 0.4803$ vs. $\hat\sigma_\eta = 0.0735$). El filtro resultante es muy suave y muy poco reactivo.
- **General Motors** [A]: se permite explícitamente que $\alpha_t$ y $\beta_t$ varíen en el tiempo, y la estimación por máxima verosimilitud concluye que **prácticamente no varían** ($\hat\sigma_\eta = 4.91\times10^{-5}$, $\hat\sigma_\epsilon = 1.22\times10^{-2}$). Dar libertad al modelo para encontrar dinámica no produce dinámica.
- **Johnson & Johnson** [A]: la descomposición funciona bien, y Tsay inmediatamente advierte que **no es única**.

El estado final de este capítulo es, deliberadamente:

**KNOWLEDGE ACQUIRED — NO DESIGN DECISIONS ADOPTED.**

---

## 2. Qué problema resuelve un state-space model

### 2.1 El problema, en una frase

Hay una cantidad que me interesa. Cambia con el tiempo. No la puedo ver. Lo único que puedo ver son mediciones relacionadas con ella, y esas mediciones tienen error. Quiero, cada vez que llega una medición nueva, **actualizar de forma racional mi estimación** de la cantidad invisible.

### 2.2 Ejemplo cotidiano extremadamente sencillo: el horno

Tenemos un horno. Adentro hay una temperatura real. Tenemos un termómetro barato pegado a la puerta.

- Cuando el termómetro marca 178°, no creemos que la temperatura sea exactamente 178°. El termómetro tiene error.
- Pero tampoco ignoramos el 178°. Es información.
- Y además sabemos algo del horno: la temperatura **no salta**. Si hace un minuto estimábamos 180°, la temperatura de ahora está cerca de 180°, no en 40°.

Entonces tenemos dos fuentes de conocimiento:

1. **Lo que sé del comportamiento del horno** (la temperatura evoluciona suavemente).
2. **Lo que acabo de medir** (178°).

Y la pregunta es exactamente: *¿cuánto le hago caso a cada una?*

Si el termómetro es muy malo y el horno muy estable → le hago poco caso a la medición.
Si el termómetro es muy bueno y el horno es un desastre que salta → le hago mucho caso.

Ese "cuánto le hago caso" es, literalmente, el **Kalman gain**.

### 2.3 Traslado a lo financiero — con mucho cuidado

**[A]** Tsay usa el modelo para analizar la **volatilidad realizada** de un activo. Ahí, dice, $\mu_t$ representa la log-volatilidad subyacente del precio del activo, y $y_t$ es el logaritmo de la volatilidad realizada. En sus palabras: *"The true log volatility is not directly observed but evolves over time according to a random-walk model."* Y añade que $y_t$ se construye a partir de datos de transacciones de alta frecuencia y está sujeta a la influencia de ruidos de microestructura de mercado; la desviación estándar de $e_t$ mide la escala de ese impacto.

**Aquí hay que frenar y ser preciso**, porque es el punto donde es más fácil sobre-interpretar.

- Tsay **sí** postula, para ese ejemplo concreto, que existe una log-volatilidad no observada. Eso es una **especificación de modelo** que le resulta razonable para ese objeto, no un descubrimiento.
- Tsay **no** dice que exista un "precio verdadero" oculto detrás de la serie de precios, ni que el Kalman Filter lo recupere.
- El Capítulo 5 nos enseñó que el precio observado sufre efectos de microestructura (*bid–ask bounce*, discretización). **[B]** Sería tentador encadenar: "microestructura ⟹ el precio observado es ruidoso ⟹ existe un precio limpio ⟹ el Kalman lo recupera". **Ese encadenamiento no está en Tsay y no debe hacerse.** El Capítulo 5 describe un mecanismo específico y medible; no autoriza a postular un estado latente arbitrario y llamarlo "el precio verdadero".

**PREGUNTA ABIERTA.** ¿Qué objetos de una serie de futuros, si alguno, admiten razonablemente la interpretación "cantidad latente medida con ruido"? La volatilidad tiene un argumento a favor (nunca se observa directamente, sólo se estima). El precio no lo tiene de la misma forma: el precio *es* el dato, es el número al que se opera. Esta distinción no se resuelve en este capítulo.

### 2.4 Qué NO es un state-space model

- **No** es un método de predicción con poder propio. Es una forma de escribir modelos y un algoritmo para calcular con ellos.
- **No** garantiza recuperar nada "verdadero". Recupera *el estado que el modelo especificado considera más plausible dados los datos y los parámetros*.
- **No** es lo mismo que "modelo de variables latentes" en el sentido de deep learning (embeddings, autoencoders). Aquí el estado tiene una dinámica explícita y una interpretación dentro de un modelo probabilístico especificado a mano.
- **No** requiere que el estado sea unidimensional ni que las matrices sean constantes.

### 2.5 Relación con capítulos anteriores

| Capítulo | Qué aportó | Cómo se conecta con el Cap. 11 |
|---|---|---|
| **C1** | Distinguir observación, distribución, condicionalidad | El estado es una cantidad sobre la que tenemos una *distribución condicional*, no un número |
| **C2** | AR/MA/ARMA, forecasting, suavizado exponencial | Tsay demuestra que ARIMA y state-space son representaciones intercambiables |
| **C3** | La volatilidad condicional no se observa directamente | Primer ejemplo natural de "cantidad latente inferida" |
| **C4** | Modelos no lineales, Markov switching, estados latentes discretos | El Cap. 11 formaliza el caso **lineal** con estado **continuo** |
| **C5** | El precio observado no es el valor económico perfecto | Motiva el concepto de *measurement error* — pero **no** autoriza a postular un "precio verdadero" |
| **C7** | Distintas partes de la distribución pueden depender del estado | Aquí preguntamos algo más básico: ¿podemos *representar* ese estado y actualizarlo? |

---

## 3. Estado observado vs. estado latente

### 3.1 ¿Qué es un "estado"?

Un **estado** es un resumen de la situación del sistema en un momento dado, suficiente para describir hacia dónde va.

En la vida cotidiana: el estado de un auto en la ruta podría ser (posición, velocidad). Con esos dos números y las reglas del movimiento, puedo decir aproximadamente dónde estará dentro de un segundo. No necesito saber el color del auto.

**[A]** En el modelo de Tsay, la ecuación de estado *"describe una cadena de Markov de primer orden que gobierna la transición del estado"*. "Markov de primer orden" significa: **dado el estado de hoy, el pasado más lejano no aporta nada más sobre el estado de mañana**. Toda la memoria del sistema está comprimida en el estado actual.

### 3.2 ¿Qué significa "latente"?

**Latente** = no observado directamente. Punto.

No significa:
- "real pero escondido";
- "la verdad que el mercado oculta";
- "una variable que existe en la naturaleza y que descubrimos".

Significa: **dentro de este modelo, postulo una variable que no está en los datos, y la infiero a partir de los datos que sí tengo.**

La diferencia es enorme. Comparemos:

| Afirmación | Estatus |
|---|---|
| "$\mu_t$ es una variable del modelo que no observo y que estimo" | Correcta, es la definición |
| "$\mu_t$ es la tendencia real que el mercado tenía y que el ruido nos ocultaba" | Injustificada sin evidencia adicional |

Tsay mismo lo deja claro al cerrar el capítulo: **[A]** existen infinitas formas de descomponer una serie observada en componentes no observados. Si hay infinitas descomposiciones válidas, ninguna de ellas es "la verdad descubierta".

### 3.3 Estado ≠ régimen

Ver también la Sección 29 de este informe. Adelanto la distinción porque es una confusión frecuente:

- **Cap. 4, Markov switching**: el estado es una **etiqueta discreta**, $S_t \in \{1, 2, \ldots, K\}$. "Estamos en régimen 1 o en régimen 2." La incertidumbre es sobre *cuál de las categorías*.
- **Cap. 11, state-space lineal**: el estado es un **vector continuo**, $s_t \in \mathbb{R}^m$. "La tendencia latente vale aproximadamente 0.87." La incertidumbre es sobre *el valor numérico*.

**[B]** No debemos llamar "régimen" a cualquier variable de estado. Un $\beta_t$ que se mueve suavemente entre 1.02 y 1.06 no describe regímenes; describe un coeficiente que deriva lentamente.

### 3.4 Relevancia para Machine Learning [B]

En ML llamamos "representación latente" a cosas muy distintas (capas ocultas, embeddings, estados de un LSTM). Hay un parentesco conceptual real —un LSTM también mantiene un estado que se actualiza con cada observación nueva— pero también diferencias importantes:

- El estado de un state-space lineal tiene una **dinámica especificada a mano** y una **distribución de probabilidad** asociada, con incertidumbre cuantificada.
- El estado de un LSTM se aprende sin especificación y no viene con una covarianza interpretable.

**PREGUNTA ABIERTA.** ¿Un estado con dinámica impuesta e incertidumbre cuantificada aporta algo que un estado aprendido libremente no aporte, en un problema con relación señal/ruido baja? El capítulo no lo responde y no lo aborda.

---

## 4. Local Trend Model desde cero

### 4.1 Las dos ecuaciones

**[A]** El modelo (ecuaciones 11.1 y 11.2 de Tsay):

$$y_t = \mu_t + e_t, \qquad e_t \sim N(0, \sigma_e^2) \tag{11.1}$$

$$\mu_{t+1} = \mu_t + \eta_t, \qquad \eta_t \sim N(0, \sigma_\eta^2) \tag{11.2}$$

**Todos los símbolos, uno por uno:**

| Símbolo | Nombre | Qué es |
|---|---|---|
| $y_t$ | observación | El dato que efectivamente tenemos en el instante $t$ |
| $\mu_t$ | estado (*state*) | La cantidad latente que nos interesa. **No se observa.** |
| $e_t$ | *measurement error* / *observation error* | La diferencia entre lo que vemos y el estado |
| $\eta_t$ | *state innovation* / *process noise* | El cambio genuino del estado entre $t$ y $t+1$ |
| $\sigma_e^2$ | varianza del error de medición | Cuánto "miente" típicamente la observación |
| $\sigma_\eta^2$ | varianza de la innovación del estado | Cuánto se mueve típicamente el estado |
| $T$ | longitud de la muestra | $t = 1, \ldots, T$ |

**[A]** Condiciones del modelo: $\{e_t\}$ y $\{\eta_t\}$ son dos series de ruido blanco gaussiano **independientes entre sí**. El valor inicial $\mu_1$ está dado o sigue una distribución conocida, y es independiente de $\{e_t\}$ y $\{\eta_t\}$ para $t > 0$.

**[A]** Terminología que fija Tsay:
- La ecuación (11.1) es la **observation equation** (o *measurement equation*), y su ruido $e_t$ es el **measurement error**.
- La ecuación (11.2) es la **state equation** (o *state transition equation*), y su ruido $\eta_t$ es la **innovation**.
- $\mu_t$ se llama en la literatura la **tendencia** de la serie, y **no es directamente observable**; $y_t$ es el dato observado con ruido observacional $e_t$.
- El modelo se llama también **local-level model** en Durbin y Koopman (2001, Cap. 2), y es un caso simple del *structural time series model* de Harvey (1993).

### 4.2 Leyendo las ecuaciones en castellano

**Ecuación de observación:** *"Lo que veo hoy es el estado de hoy, más un error."*

**Ecuación de estado:** *"El estado de mañana es el estado de hoy, más un empujón aleatorio."*

Nótese lo que dice la segunda: el estado es un **paseo aleatorio puro**. **[A]** Tsay lo señala explícitamente: $\mu_t$ es un paseo aleatorio del Capítulo 2 con valor inicial $\mu_1$. No tiene tendencia hacia ningún valor de equilibrio, no revierte a la media, no tiene deriva. Simplemente deambula.

**Esto es una elección de modelo, no un hecho.** Es la elección más simple posible para "algo que cambia lentamente y sin dirección preferida". Se podría haber elegido una AR(1) estacionaria, una tendencia con pendiente, etc.

### 4.3 De dónde viene la dinámica

**[A]** Observación importante de Tsay: *"The dynamic dependence of $y_t$ is governed by that of $\mu_t$ because $\{e_t\}$ is not serially correlated."*

Es decir: **toda la memoria de la serie observada está en el estado**. El ruido de medición no tiene memoria; es blanco. Si $y_t$ tiene autocorrelación, es porque $\mu_t$ la tiene.

**[B]** Esto tiene una implicación de diseño incómoda que conviene ver desde ahora: si postulamos que gran parte de la variación observada es ruido de medición sin memoria, estamos postulando —por construcción del modelo— que **esa parte no es predecible en absoluto**. No es un resultado empírico; es lo que asumimos al escribir el modelo así. Un modelo distinto podría atribuir esa misma variación a un estado con dinámica.

### 4.4 Qué NO dice el local trend model

- **No** dice que $y_t$ sea "$\mu_t$ contaminado y que $\mu_t$ sea lo importante". Dice que *si* modelamos así, esto es lo que se deduce.
- **No** dice que $e_t$ sea ruido de microestructura. En el ejemplo de volatilidad realizada Tsay lo interpreta así **para ese caso concreto**; en general $e_t$ es simplemente "la parte de la observación que el modelo no atribuye al estado".
- **No** implica que $\mu_t$ sea suave "en la realidad". La suavidad del $\hat\mu_t$ estimado es consecuencia de la razón $\sigma_e^2/\sigma_\eta^2$ que estimamos, no una propiedad del mundo.

---

## 5. Observation noise vs. state innovation

Ésta es, probablemente, la distinción conceptual más importante para no malinterpretar todo el resto del capítulo.

### 5.1 La pregunta que el filtro tiene que responder

Supongamos que ayer estimábamos que el estado valía 100. Hoy la observación llega y marca **107**.

Hay tres explicaciones posibles:

1. **El estado cambió de verdad.** Pasó algo, el nivel subyacente subió. Debería actualizar bastante mi estimación.
2. **La medición se desvió.** El estado sigue cerca de 100, pero esta medición en particular salió alta por ruido. No debería moverme mucho.
3. **Una mezcla de ambas.** Lo habitual.

$\eta_t$ representa la explicación 1. $e_t$ representa la explicación 2.

**El Kalman Filter no descubre cuál de las tres es la verdadera.** Lo que hace es repartir la sorpresa entre las dos explicaciones **según la proporción que los parámetros del modelo le dictan**. Si le decimos que el ruido de medición es grande, atribuirá la sorpresa mayormente a ruido. Si le decimos lo contrario, la atribuirá al estado.

### 5.2 Las dos diferencias estructurales

| | $e_t$ (measurement error) | $\eta_t$ (state innovation) |
|---|---|---|
| ¿Dónde vive? | En la ecuación de observación | En la ecuación de estado |
| ¿Qué contamina? | Sólo la observación de hoy | El estado de mañana y **todos los siguientes** |
| ¿Persiste? | **No.** Se olvida inmediatamente. | **Sí.** Se acumula para siempre (el estado es un paseo aleatorio) |
| Efecto en la serie observada | Ruido sin memoria | Fuente de toda la dependencia dinámica |

Ese renglón de "¿persiste?" es la clave. Un error de medición de hoy no afecta a mañana. Una innovación del estado de hoy **se queda incorporada al estado** y afecta a todo el futuro.

### 5.3 Ejemplo cotidiano

Volvemos al horno, pero cambiemos el ejemplo para hacerlo más nítido: **una balanza y una persona a dieta**.

- **Peso real** = estado. Cambia lentamente, día a día, y los cambios se acumulan (si bajaste 300 g, mañana partís de ahí).
- **Lectura de la balanza** = observación. Depende del peso real, pero también de si te pesaste antes o después de desayunar, con qué ropa, si la balanza está en el azulejo o en la alfombra.

Si un día la balanza marca 2 kg más que ayer, es muchísimo más probable que sea la alfombra que un aumento real de 2 kg en un día. Un buen "filtro" mental descuenta casi todo ese salto. Pero si la balanza marca 200 g más cada día durante dos semanas, el filtro **sí** debería moverse: catorce mediciones consistentes no se explican por ruido.

Nótese que el filtro no hace nada mágico: usa el hecho de que **el ruido no tiene memoria y el estado sí**. Muchas mediciones apuntando en la misma dirección son evidencia de cambio de estado; una sola medición extrema no lo es.

### 5.4 Ejemplo financiero sencillo

**[A]** En el ejemplo de Alcoa: $y_t$ = log de la volatilidad realizada diaria (calculada a partir de retornos de 10 minutos); $\mu_t$ = log-volatilidad latente. Un día la volatilidad realizada sale muy alta.

- Puede ser que el régimen de volatilidad efectivamente subió ($\eta_t$ grande).
- Puede ser que ese día, con esa muestra finita de 10-minutos, el estimador de volatilidad realizada salió alto ($e_t$ grande). La volatilidad realizada es un **estimador**, y los estimadores tienen error muestral.

Esta segunda interpretación es la que Tsay hace explícita: la volatilidad realizada se construye de datos de alta frecuencia y sufre ruido de microestructura.

### 5.5 Qué NO significa

- **[B]** $e_t$ **no** es necesariamente "ruido de microestructura". Es simplemente el residuo de la ecuación de observación bajo el modelo elegido. Puede contener microestructura, error muestral del estimador, o dinámica real que el modelo no supo representar.
- **[B]** Que el modelo atribuya una fracción $X$ de la varianza a $e_t$ **no demuestra** que esa fracción sea impredecible. Demuestra que *bajo ese modelo* esa fracción no tiene estructura dinámica. Un modelo distinto podría encontrar estructura ahí.

---

## 6. Las dos varianzas y el trade-off smooth/reactive

### 6.1 La razón señal-ruido

Todo el comportamiento del local trend model depende de **una sola cantidad**: la razón

$$q = \frac{\sigma_\eta^2}{\sigma_e^2}$$

conocida como *signal-to-noise ratio* en la literatura de state-space. No dependen de los valores absolutos, sino de su cociente.

### 6.2 Los dos extremos, intuitivamente

**Caso A: mucho ruido de medición.** $\sigma_e^2 \gg \sigma_\eta^2$, es decir $q \approx 0$.

> "Mis mediciones son malas; el estado se mueve poco. Cuando una observación se aparta de lo que esperaba, casi seguro es error de medición."

Consecuencia: el estado estimado se mueve **poco** en respuesta a cada dato nuevo. El filtro es **suave y perezoso**. Necesita muchas observaciones consistentes para cambiar de opinión.

**Caso B: mucho ruido de estado.** $\sigma_\eta^2 \gg \sigma_e^2$, es decir $q$ grande.

> "Mis mediciones son buenas; el estado sí se mueve rápido. Cuando una observación sorprende, probablemente el estado cambió."

Consecuencia: el estado estimado **sigue de cerca** a la observación. El filtro es **reactivo**. En el límite $\sigma_e = 0$, el estado estimado *es* la observación: no hay nada que filtrar.

### 6.3 Ahora con la fórmula

**[A]** El Kalman gain del local trend model (de la ecuación 11.14 de Tsay) es:

$$K_t = \frac{\Sigma_{t|t-1}}{V_t} = \frac{\Sigma_{t|t-1}}{\Sigma_{t|t-1} + \sigma_e^2}$$

donde $\Sigma_{t|t-1} = \mathrm{Var}(\mu_t \mid F_{t-1})$ es la incertidumbre que tenía sobre el estado **antes** de ver $y_t$.

Léase en castellano:

$$K_t = \frac{\text{incertidumbre previa sobre el estado}}{\text{incertidumbre previa sobre el estado} + \text{ruido de medición}}$$

Es una fracción entre 0 y 1, siempre. Y su lógica es transparente: **compara mi ignorancia sobre el estado contra el ruido de la medición**. Gana el que sea más grande.

- Si estoy muy inseguro del estado ($\Sigma_{t|t-1}$ grande) frente a un instrumento decente ($\sigma_e^2$ chico) → $K_t \to 1$ → le hago mucho caso al dato nuevo.
- Si estoy muy seguro del estado ($\Sigma_{t|t-1}$ chico) frente a un instrumento malo ($\sigma_e^2$ grande) → $K_t \to 0$ → casi ignoro el dato nuevo.

¿Y de dónde sale $\Sigma_{t|t-1}$? De la recursión (11.14): $\Sigma_{t+1|t} = \Sigma_{t|t-1}(1-K_t) + \sigma_\eta^2$. Es decir: la incertidumbre **baja** cuando observo (el término $1-K_t$) y **sube** cuando el tiempo pasa (el término $+\sigma_\eta^2$). En equilibrio entre esas dos fuerzas, $\Sigma$ se estabiliza — y con ella, $K$.

### 6.4 Un hecho estructural crucial

**[A]** Tsay lo dice de forma explícita en la Sección 11.1.3: *"the Kalman gain $K_t$ does not depend on $\mu_{1|0}$ or the data $\{y_1, \ldots, y_t\}$; it depends on $\Sigma_{1|0}$ and $\sigma_e^2$ and $\sigma_\eta^2$."*

**Esto merece que nos detengamos.** En el local trend model, **la secuencia de ganancias es completamente determinista**. No depende de los datos. Se puede calcular antes de ver un solo dato, sabiendo únicamente la inicialización y las dos varianzas.

**[B]** Consecuencia directa: el estado filtrado $\mu_{t|t}$ es una **combinación lineal de los datos con pesos fijos y conocidos de antemano**. No es un objeto adaptativo en el sentido de "aprende de los datos". Es un filtro lineal con coeficientes que dependen sólo de los parámetros. En la Sección 31 volvemos sobre lo que eso significa frente a una EMA.

**[A]** Además, la Sección 11.4.1 (*Steady State*) señala que si el modelo es invariante en el tiempo, $\Sigma_{t|t-1}$ **converge** a una matriz constante $\Sigma_*$, y una vez alcanzado el estado estacionario, $V_t$, $K_t$ y $\Sigma_{t+1|t}$ son todos constantes.

### 6.5 "¿Qué hace que un Kalman Filter sea suave o reactivo?"

Resumen operativo:

| Si… | Entonces $K$ tiende a… | El filtro es… |
|---|---|---|
| $\sigma_e^2$ grande relativo a $\sigma_\eta^2$ | 0 | Muy suave, lento, poco reactivo |
| $\sigma_e^2$ chico relativo a $\sigma_\eta^2$ | 1 | Muy reactivo, sigue el dato |
| $\sigma_e = 0$ | 1 exactamente | El estado *es* la observación |
| $\sigma_\eta = 0$ | → 0 con $t$ | El estado es una constante; el filtro converge a la media muestral |
| Incertidumbre inicial enorme (difusa) | ≈1 al principio | Reactivo al arranque, se va calmando |

**[B]** Advertencia importante: nada de esto es una regla del mundo. Es aritmética del modelo. Que un filtro estimado resulte "suave" nos dice que **la razón de varianzas estimada fue baja**, no que el mercado sea suave.

### 6.6 Qué NO significa

- **No** significa que un filtro suave sea "mejor" ni "más limpio". Significa que descarta más variación como ruido.
- **No** significa que un filtro reactivo capture mejor la señal. Significa que atribuye más variación al estado.
- **[B]** Suavizar **no** aumenta la predictibilidad por sí mismo. Reduce varianza a costa de introducir retraso. Que ese intercambio sea favorable para una tarea concreta es una pregunta empírica, no una consecuencia matemática.

---

## 7. Ejemplo Alcoa — registro detallado

**[A] Todos los datos de esta sección provienen del Ejemplo 11.1 de Tsay.**

### 7.1 Ficha del experimento

| Campo | Valor |
|---|---|
| **Activo** | Acción de Alcoa |
| **Fuente de datos** | Base TAQ del NYSE (datos de transacciones) |
| **Período** | 2 de enero de 2003 – 7 de mayo de 2004 |
| **Observaciones** | 340 (diarias) |
| **Variable construida** | Volatilidad realizada diaria = suma de cuadrados de log-retornos intradiarios de **10 minutos**, medidos en porcentaje |
| **Exclusiones** | **No** se usan retornos overnight ni los primeros 10 minutos intradiarios |
| **Variable analizada** | $y_t = \log(\text{volatilidad realizada diaria})$ |

**Es fundamental ser preciso sobre qué objeto se modela.** No es el precio. No es un retorno de 1 minuto. No es tick data. No es "la volatilidad verdadera". Es **el logaritmo de un estimador diario de volatilidad, construido agregando los retornos de 10 minutos de cada sesión** (excluyendo el overnight y el primer intervalo del día).

### 7.2 Modelo ARIMA ajustado

**[A]** Ajustando ARIMA al log de la volatilidad realizada se obtiene (ecuación 11.6):

$$(1-B)y_t = (1 - 0.858B)a_t, \qquad \hat\sigma_a = 0.5184$$

- Error estándar de $\hat\theta$: **0.028**
- Diagnóstico de residuos: $Q(12) = 12.4$, p-valor **0.33** → sin autocorrelación serial significativa
- Residuos al cuadrado: $Q(12) = 8.2$, p-valor **0.77** → sin efectos ARCH

### 7.3 Conversión a local trend model

**[A]** Como $\hat\theta > 0$, el ARIMA(0,1,1) se puede transformar en local trend model. Los estimadores de máxima verosimilitud de los dos parámetros son:

$$\hat\sigma_\eta = 0.0735, \qquad \hat\sigma_e = 0.4803$$

**[A]** Comentario textual de Tsay: *"The measurement errors have a larger variance than the state innovations, confirming that intraday high-frequency returns are subject to measurement errors."*

**[A]** Además, usando el ARIMA (11.6) y las relaciones (11.4)–(11.5) directamente, se obtiene $\sigma_e = 0.480$ y $\sigma_\eta = 0.0736$ — *"These values are close to the MLE shown above."* Es decir: las dos rutas (ajustar ARIMA y convertir, vs. estimar el state-space directamente) coinciden. Verificado independientemente en este informe: partiendo de $\theta = 0.858$, $\sigma_a = 0.5184$, las ecuaciones (11.4)–(11.5) dan $\sigma_e = 0.4802$ y $\sigma_\eta = 0.0736$.

**Magnitud de la asimetría** (aritmética sobre cifras [A]): $\hat\sigma_e / \hat\sigma_\eta \approx 6.5$, es decir $\hat\sigma_e^2 / \hat\sigma_\eta^2 \approx 43$. El ruido de medición tiene aproximadamente **43 veces** la varianza de la innovación del estado.

**[A]** Inicialización usada: $\Sigma_{1|0} = \infty$ (difusa; en el código, `P1 = -1`) y $\mu_{1|0} = 0$.

### 7.4 Resultados del filtrado

**[A]**
- Figura 11.2(a): estado filtrado $\mu_{t|t}$. Figura 11.2(b): error de predicción 1-paso $v_t$.
- *"Compared with Figure 11.1, the filtered states are smoother. The forecast errors appear to be stable and center around zero. These forecast errors are out-of-sample 1-step-ahead prediction errors."*
- Figura 11.3: estado filtrado con intervalo de confianza puntual del 95%.
- Figura 11.4: estado suavizado $\mu_{t|T}$ con intervalo del 95%.
- *"As expected, the smoothed state variables are smoother than the filtered state variables. The confidence intervals for the smoothed state variables are also narrower than those of the filtered state variables."*
- *"Note that the width of the 95% confidence interval of $\mu_{1|1}$ depends on the initial value $\Sigma_{1|0}$."*

### 7.5 Verificación del modelo

**[A]** Se calcula el error de predicción estandarizado $\tilde v_t = v_t / \sqrt{V_t}$ y se examinan sus autocorrelaciones y efectos ARCH:

- $Q(25) = 23.37$, p-valor **0.56**
- Test LM de efecto ARCH: $18.48$, p-valor **0.82**, para 25 rezagos

**[A]** Conclusión de Tsay: *"the fitted local trend model is adequate based on residual analysis."*

### 7.6 Qué se puede y qué NO se puede concluir de este ejemplo

**Se puede [A]:**
- Para *esta* serie, *este* activo, *este* período y *esta* construcción de variable, la varianza del error de medición estimada excede a la de la innovación del estado.
- El modelo pasa los diagnósticos de adecuación disponibles.
- Las dos rutas de estimación (ARIMA convertido y MLE del state-space) coinciden numéricamente.

**NO se puede:**
- ❌ "Los datos intradiarios siempre tienen principalmente ruido de medición." Es un resultado sobre un estimador diario de volatilidad de una acción en 340 días.
- ❌ "El precio intradiario de MNQ es mayormente ruido." No se estudió ningún precio; se estudió el log de un estimador de volatilidad diaria.
- ❌ "Esto valida el Kalman Filter como generador de señal." No se hizo ninguna evaluación de utilidad económica, de rentabilidad, ni de forecasting fuera de muestra más allá de los errores 1-paso.
- ❌ "43× de ruido significa que el 97.7% de la variación es impredecible." Significa que **bajo el modelo local trend especificado**, esa proporción se asigna al término sin memoria. Otro modelo podría asignarla de otra forma.

**PREGUNTA ABIERTA.** ¿Qué relación de varianzas resultaría al modelar una medida de volatilidad de MNQ construida con datos intradiarios? El capítulo no lo dice y no debe extrapolarse. Además, la respuesta dependería fuertemente de la ventana de agregación elegida (10 min vs. 1 min vs. 5 min), que en el ejemplo de Tsay es una decisión previa al modelado.

**Dato adicional relevante [A]:** el Ejercicio 11.2 del libro propone repetir el análisis con volatilidad realizada construida con retornos de **20 minutos** en lugar de 10. Tsay no reporta el resultado, pero la existencia misma del ejercicio confirma que **la elección de la ventana intradiaria es un parámetro libre que afecta el resultado**.

---

## 8. Relación State-Space ↔ ARIMA

Ésta es, para nuestros fines, la sección más importante del capítulo después de la advertencia de no unicidad.

### 8.1 El resultado

**[A]** Tsay demuestra, en pocas líneas de álgebra:

- Si $\sigma_e = 0$ (sin error de medición), entonces $y_t = \mu_t$, que es un **ARIMA(0,1,0)** (paseo aleatorio).
- Si $\sigma_e > 0$, entonces $y_t$ es un **ARIMA(0,1,1)**:
$$(1-B)y_t = (1 - \theta B)a_t \tag{11.3}$$
donde $\theta$ y $\sigma_a$ quedan determinados por $\sigma_e$ y $\sigma_\eta$ mediante:
$$(1+\theta^2)\sigma_a^2 = 2\sigma_e^2 + \sigma_\eta^2 \tag{11.4}$$
$$\theta\sigma_a^2 = \sigma_e^2 \tag{11.5}$$

**[A]** Y remata: *"the state-space model in Eqs. (11.1) and (11.2) is also an ARIMA(0,1,1) model, **which is the simple exponential smoothing model of Chapter 2**."*

### 8.2 La derivación en castellano, sin álgebra

¿Por qué dos modelos que se ven tan distintos describen la misma dinámica?

Tomemos la diferencia primera de la observación: $w_t = y_t - y_{t-1}$.

$$w_t = (\mu_t + e_t) - (\mu_{t-1} + e_{t-1}) = \underbrace{\eta_{t-1}}_{\text{cambio del estado}} + \underbrace{e_t - e_{t-1}}_{\text{diferencia de ruidos}}$$

Ahora miremos qué memoria tiene $w_t$:

- $w_t$ y $w_{t-1}$ **comparten** el término $e_{t-1}$ (con signos opuestos). Están correlacionados.
- $w_t$ y $w_{t-2}$ **no comparten nada**. No están correlacionados.

Una serie que está correlacionada sólo con su vecino inmediato y con nadie más es, por definición, un **MA(1)**. Y una serie cuya diferencia primera es MA(1) es un ARIMA(0,1,1). Fin.

**Lo que hay que ver aquí:** no hubo magia. El estado latente no aportó nada que no estuviera ya en la estructura de autocovarianzas de los datos. Simplemente, **la misma estructura de autocovarianzas admite dos descripciones verbales distintas**:

- Descripción A: "hay un nivel oculto que camina y lo veo con ruido".
- Descripción B: "las diferencias de la serie tienen memoria de un paso".

Ambas producen exactamente la misma distribución de probabilidad para los datos.

### 8.3 Detalles finos

**[A]**
- La ecuación resultante en $\theta$ es cuadrática y tiene dos soluciones; hay que elegir la que satisface $|\theta| < 1$ (invertibilidad).
- Para un ARIMA(0,1,1) con $\theta$ **positivo**, se pueden resolver las identidades hacia atrás y obtener un local trend model. **Si $\theta$ es negativo**, el modelo se puede poner igualmente en forma state-space, pero **sin error de observación** ($\sigma_e = 0$). Es decir: la correspondencia no es biyectiva sobre todo el espacio de parámetros.

### 8.4 La cita que hay que recordar

**[A]** *"In practice, what one observes is the $y_t$ series. Thus, based on the data alone, the decision of using ARIMA models or linear state-space models is not critical. Both model representations have pros and cons. The objective of data analysis, substantive issues, and experience all play a role in choosing a statistical model."*

Traducción del espíritu: **los datos no te dicen cuál representación usar, porque ambas les calzan igual de bien.** La elección es por conveniencia de cálculo, de interpretación o de propósito — no por ajuste.

### 8.5 La dirección recíproca

**[A]** En la Sección 11.3.2, Tsay señala el resultado inverso: para un state-space lineal con coeficientes invariantes en el tiempo (ecuaciones 11.26–11.27), se puede usar el teorema de Cayley–Hamilton para demostrar que la observación $y_t$ sigue un **ARMA($m,m$)**, donde $m$ es la dimensión del vector de estado.

Es decir: **la correspondencia va en ambos sentidos**. No hay una clase de dinámicas que sólo el state-space pueda representar (dentro del marco lineal-gaussiano invariante).

### 8.6 Implicación para Machine Learning [B]

$$\boxed{\text{transformar algo a state-space no crea información nueva}}$$

Si tomamos una serie, la escribimos como state-space, corremos el Kalman Filter y obtenemos $\hat\mu_{t|t}$, lo que tenemos es **una transformación determinista y lineal de los datos que ya teníamos**. Los datos de entrada son los mismos. El contenido informativo del conjunto $\{y_1, \ldots, y_t\}$ no cambió por haberlo mirado de otra forma.

Lo que **sí** puede cambiar es la facilidad con la que un modelo posterior extrae una relación de esa representación. Eso es una pregunta empírica legítima y no trivial (la ingeniería de features existe por eso). Pero es una pregunta sobre **facilidad de extracción**, no sobre **cantidad de información**.

**PREGUNTA ABIERTA.** ¿Existe alguna tarea predictiva sobre MNQ donde una representación filtrada facilite la extracción de una relación estable que features causales más simples no faciliten? No se responde aquí.

---

## 9. Filtering, Prediction y Smoothing

**PRIORIDAD MÁXIMA.** Ésta es la sección que hay que entender perfectamente antes de tocar nada.

### 9.1 Las definiciones exactas de Tsay

**[A]** Sea $F_t = \{y_1, \ldots, y_t\}$ la información disponible en el tiempo $t$ (inclusive), y supóngase que el modelo es conocido, incluidos todos sus parámetros. Los tres tipos de inferencia son:

- **Filtering.** *"Filtering means to recover the state variable $\mu_t$ given $F_t$, that is, to remove the measurement errors from the data."*
- **Prediction.** *"Prediction means to forecast $\mu_{t+h}$ or $y_{t+h}$ for $h > 0$ given $F_t$, where $t$ is the forecast origin."*
- **Smoothing.** *"Smoothing is to estimate $\mu_t$ given $F_T$, where $T > t$."*

### 9.2 La analogía de Tsay: leer una nota manuscrita

**[A]** Tsay ofrece una analogía que vale la pena parafrasear porque es muy buena.

Imaginemos que estamos leyendo una nota escrita a mano con letra difícil.

- **Filtering** es descifrar la palabra que estamos leyendo ahora, usando todo lo que entendimos desde el comienzo de la nota hasta este punto.
- **Prediction** es adivinar cuál será la palabra siguiente, que todavía no vimos.
- **Smoothing** es volver sobre una palabra que quedó dudosa **después de haber leído la nota entera**, y reinterpretarla a la luz del texto completo.

La analogía deja ver inmediatamente por qué el smoothing es más preciso: por supuesto que es más fácil descifrar una palabra ambigua cuando ya leíste el párrafo entero. **Y deja ver, con igual claridad, por qué esa ventaja es inútil para tomar una decisión mientras leés.** Cuando estás en la mitad de la nota, todavía no leíste el final.

### 9.3 La tabla operativa

| Operación | Qué queremos estimar | Notación | Información utilizada | ¿Causal en $t$? [B] |
|---|---|---|---|---|
| **Filtering** | Estado **actual** | $\mu_{t\mid t}$ | $F_t = \{y_1,\ldots,y_t\}$ | **Sí** (después de que $y_t$ está disponible) |
| **Prediction** | Estado u observación **futura** | $\mu_{t+h\mid t}$, $y_{t+h\mid t}$ | $F_t$ | **Sí** |
| **Smoothing** | Estado **pasado**, revisado | $\mu_{t\mid T}$, $T > t$ | $F_T$, incluye $y_{t+1},\ldots,y_T$ | **NO** para uso online en $t$ |

La última columna es adaptación **[B]** hacia el uso en ML. Tsay no discute leakage; discute inferencia estadística.

### 9.4 Notación de Tsay que hay que dominar

**[A]**

| Símbolo | Definición |
|---|---|
| $F_t$ | $\{y_1, \ldots, y_t\}$ — información hasta $t$ inclusive |
| $\mu_{t\mid j}$ | $E(\mu_t \mid F_j)$ — media condicional del estado en $t$ dada info hasta $j$ |
| $\Sigma_{t\mid j}$ | $\mathrm{Var}(\mu_t \mid F_j)$ — varianza condicional correspondiente |
| $y_{t\mid j}$ | $E(y_t \mid F_j)$ — media condicional de la observación |
| $v_t$ | $y_t - y_{t\mid t-1}$ — error de predicción 1-paso (*innovation*) |
| $V_t$ | $\mathrm{Var}(v_t \mid F_{t-1})$ — su varianza |

La convención de subíndices es: **primer índice = de qué momento hablo; segundo índice = hasta cuándo miré.**

- $\mu_{3|3}$: el estado de $t=3$, mirando hasta $t=3$. **Filtered.**
- $\mu_{3|2}$: el estado de $t=3$, mirando hasta $t=2$. **Predicted (one-step).**
- $\mu_{3|10}$: el estado de $t=3$, mirando hasta $t=10$. **Smoothed.**

### 9.5 Un ejemplo temporal concreto

Supongamos que tenemos datos $y_1, y_2, y_3, y_4, y_5$ y queremos hablar del estado en $t=3$.

**Estado filtrado:**
$$\mu_{3|3} = E[\mu_3 \mid y_1, y_2, y_3]$$
Usa tres observaciones. Es lo mejor que se puede decir sobre $\mu_3$ **en el momento en que $y_3$ acaba de llegar**.

**Estado suavizado:**
$$\mu_{3|5} = E[\mu_3 \mid y_1, y_2, y_3, y_4, y_5]$$
Usa cinco observaciones. Es lo mejor que se puede decir sobre $\mu_3$ **una vez terminada la muestra**.

**¿Por qué el segundo es más preciso?** Porque $y_4$ e $y_5$ contienen información sobre $\mu_3$. Recordemos: el estado es un paseo aleatorio, así que $\mu_4$ y $\mu_5$ están cerca de $\mu_3$. Si $y_4$ e $y_5$ salen ambos altos, eso es evidencia de que $\mu_3$ probablemente ya estaba alto. **Estamos usando el futuro para hablar del pasado.**

**[A]** Esto no es una intuición: es literalmente lo que dice la fórmula del smoother. De la ecuación (11.17)–(11.20) de Tsay:

$$\mu_{t|T} = \mu_{t|t-1} + \Sigma_{t|t-1}\,q_{t-1}, \qquad q_{t-1} = \frac{v_t}{V_t} + L_t q_t$$

donde la recursión hacia atrás arranca en $q_T = 0$. Desarrollada, $q_{t-1}$ es:

$$q_{t-1} = \frac{v_t}{V_t} + L_t\frac{v_{t+1}}{V_{t+1}} + L_tL_{t+1}\frac{v_{t+2}}{V_{t+2}} + \cdots + \left(\prod_{j=t}^{T-1}L_j\right)\frac{v_T}{V_T}$$

**[A]** En palabras de Tsay: $q_{t-1}$ es *"a weighted linear combination of the innovations $\{v_t, \ldots, v_T\}$."*

Ahí está, escrito: **el estado suavizado en $t$ es una función explícita de las innovaciones de $t+1, t+2, \ldots, T$.** Todas posteriores a $t$.

### 9.6 El ejemplo con horarios

Tomemos barras de 1 minuto:

```
10:00    10:01    10:02    10:03
```

Queremos estimar el estado correspondiente a **10:01**.

- **Filtro** ($\mu_{10:01 \mid 10:01}$): puede usar 10:00 y 10:01. Nada más existe todavía.
- **Smoother** ($\mu_{10:01 \mid 10:03}$): usa además 10:02 y 10:03.

**[B]** Si a las 10:01 hay que tomar una decisión operativa, el smoother **no es utilizable**. No porque sea impreciso —al contrario, es *más* preciso—, sino porque **usa números que a las 10:01 todavía no existían en el mundo**.

Un backtest que use estados suavizados producirá resultados espectaculares y completamente falsos. Y el error no será evidente al mirar el código: la función `smooth()` no tiene ninguna marca que diga "esto ve el futuro". Hay que saberlo del modelo.

### 9.7 Filtering ≠ Prediction

**[B]** Un error frecuente es decir "el Kalman Filter predice". No. El Kalman Filter **filtra**, y *además* produce predicciones como subproducto. Son operaciones distintas:

A las 10:00, con toda la información hasta 10:00:

- **Estado filtrado**: "mi mejor estimación de la cantidad latente **de las 10:00**". Es una afirmación sobre el **presente**.
- **Forecast**: "mi mejor estimación de lo que ocurrirá a las 10:01, 10:02...". Es una afirmación sobre el **futuro**.

En el local trend model resultan estar relacionados de forma trivial —**[A]** la ecuación (11.12) dice $\mu_{t+1|t} = \mu_{t|t}$, porque el estado es un paseo aleatorio y su mejor predicción es su valor actual— pero eso es una peculiaridad de *este* modelo, no una identidad general. En un modelo con $T_t \neq I$, filtrar y predecir dan números distintos.

Y aunque el estado predicho coincida con el filtrado, **las incertidumbres no coinciden**: $\Sigma_{t+1|t} = \Sigma_{t|t} + \sigma_\eta^2$. Predecir el futuro siempre es más incierto que estimar el presente.

---

## 10. Innovation / Forecast Error

### 10.1 La definición

**[A]** (ecuación 11.7):

$$v_t = y_t - y_{t|t-1} = y_t - \mu_{t|t-1}$$

En castellano:

> **Lo que realmente observé, menos lo que esperaba observar antes de verlo.**

Es la **sorpresa**. Es la parte de $y_t$ que no era anticipable con la información previa.

**[A]** La segunda igualdad viene de que, en el local trend model, $y_{t|t-1} = E(\mu_t + e_t \mid F_{t-1}) = \mu_{t|t-1}$: como el error de medición tiene media cero y no tiene memoria, la mejor predicción de la observación *es* la predicción del estado.

### 10.2 Su varianza

**[A]** (ecuación 11.8):

$$V_t = \mathrm{Var}(v_t \mid F_{t-1}) = \Sigma_{t|t-1} + \sigma_e^2$$

Léase: **la sorpresa tiene dos fuentes.** Una parte porque no sabía bien dónde estaba el estado ($\Sigma_{t|t-1}$); otra parte porque la medición tiene ruido ($\sigma_e^2$). Se suman porque son independientes.

Nótese que esto es exactamente el denominador del Kalman gain. No es casualidad: la ganancia es la fracción de la sorpresa total que es atribuible a incertidumbre sobre el estado.

### 10.3 Propiedades bajo el modelo correctamente especificado

**[A]** Tsay demuestra que, si el modelo es correcto:

1. **Media cero:** $E(v_t) = 0$.
2. **No correlacionado con el pasado:** $\mathrm{Cov}(v_t, y_j) = 0$ para $j < t$.
3. **Mutuamente independientes:** $\{v_t\}$ son mutuamente independientes bajo el supuesto de normalidad (Sección 11.1.3).
4. **Varianza conocida y variable:** cada $v_t$ tiene su propia varianza $V_t$, que el filtro calcula.
5. **[A]** Tsay las llama, en el Ejemplo 11.1, *"out-of-sample 1-step-ahead prediction errors"*.

Esa última observación es importante: aunque el modelo se haya estimado con toda la muestra, la recursión del filtro construye $v_t$ usando sólo información hasta $t-1$. **Los $v_t$ son errores 1-paso genuinos en el sentido de la recursión** — con la salvedad crucial de que **los parámetros $\sigma_e, \sigma_\eta$ sí se estimaron con toda la muestra**. Volveremos sobre esto en la Sección 32; es una fuente de leakage sutil.

### 10.4 Cómo se usa como diagnóstico

**[A]** El error estandarizado es $\tilde v_t = v_t/\sqrt{V_t}$. En el caso general (Sección 11.6), $D_t^{-1/2}v_t$ con $D_t = \mathrm{diag}\{V_t(1,1),\ldots,V_t(k,k)\}$.

Si el modelo está bien especificado, $\{\tilde v_t\}$ debería parecer ruido blanco estándar:
- Sin autocorrelación serial → test Ljung–Box.
- Sin efectos ARCH → test LM.

**[A]** En Alcoa: $Q(25) = 23.37$ (p = 0.56) y LM ARCH $= 18.48$ (p = 0.82) para 25 rezagos. Ambos no significativos → modelo adecuado.

**Qué significa que haya estructura residual:** que el modelo dejó dinámica sin explicar. Si $\tilde v_t$ tuviera autocorrelación, significaría que la sorpresa de hoy es parcialmente anticipable con la sorpresa de ayer — y por definición, entonces, no era una sorpresa genuina y el modelo se puede mejorar.

### 10.5 Qué NO significa

**[B]** Éste es un punto que hay que subrayar:

$$\text{innovations} \approx \text{ruido blanco} \nRightarrow \text{el modelo es verdadero}$$

Pasar el test de Ljung–Box significa que **no encontramos autocorrelación lineal en los residuos con la potencia estadística disponible**. Es un diagnóstico de adecuación **en ciertas dimensiones**:

- Detecta dependencia **lineal** en los residuos.
- Detecta heterocedasticidad de tipo ARCH.
- **No** detecta dependencia no lineal de otras formas.
- **No** dice nada sobre capacidad predictiva fuera de muestra.
- **No** dice nada sobre si otro modelo ajustaría mejor.
- **No** valida la interpretación del estado.

### 10.6 Conexión con capítulos anteriores

| Capítulo | Objeto análogo | Diferencia de contexto |
|---|---|---|
| **C2** | $a_t$, el shock/residuo de un ARMA | En ARMA, $a_t$ es *el* ruido del proceso. Aquí, $v_t$ mezcla dos ruidos ($e_t$ y la incertidumbre sobre el estado) |
| **C3** | La innovación estandarizada de un GARCH | Allí se usa para modelar la varianza condicional. Aquí, $V_t$ viene dada por la recursión del filtro, no por un modelo GARCH |

**Importante:** aunque $V_t$ varía con $t$ (y por tanto los $v_t$ tienen varianza cambiante), **esto no es volatilidad condicional en el sentido del Capítulo 3**. En el local trend model, $V_t$ evoluciona de forma **determinista**, sin depender de los datos — es una consecuencia de la recursión de $\Sigma_{t|t-1}$, que como vimos en la Sección 6.4 no depende de las observaciones. Un GARCH, en cambio, tiene una varianza condicional que sí es función de los shocks pasados observados. **No confundir.**

### 10.7 Un rol adicional, técnico pero relevante

**[A]** La ecuación (11.15) muestra que el vector de innovaciones se obtiene del vector de datos mediante una transformación triangular inferior con unos en la diagonal: $v = K(y - \mu_{1|0}\mathbf{1}_T)$. Tsay observa que esto constituye una **descomposición de Cholesky** de la matriz de covarianzas de $y$: $K\Sigma K' = \mathrm{diag}\{V_1,\ldots,V_T\}$.

**Contexto mínimo (el Cap. 10 fue diferido):** la idea es que el filtro convierte una serie con dependencia temporal en una secuencia de sorpresas independientes, y esa conversión es exactamente el mismo objeto matemático que "blanquear" la matriz de covarianzas. No necesitamos más que eso para seguir el capítulo.

---

## 11. Kalman Gain desde cero

**PRIORIDAD MÁXIMA.** Esta sección está escrita para que quede comprensible sin álgebra.

### 11.1 La pregunta que responde

> Llega una observación nueva que **contradice** lo que yo esperaba. ¿Cuánto debo mover mi estimación?

Todo el Kalman gain es eso. Nada más.

### 11.2 Los dos extremos, sin fórmulas

**Situación 1 — sé muy bien dónde está el estado, y el instrumento es malo.**

Vengo midiendo un objeto muy estable durante meses. Estoy segurísimo de que vale ~100. El instrumento tiene un ruido notorio. Hoy marca 107.

Reacción sensata: apenas mover la estimación. Quizá a 100.3. La medición es un dato, pero un dato barato frente a mi certeza acumulada.

→ **Ganancia baja.**

**Situación 2 — no tengo idea de dónde está el estado, y el instrumento es bueno.**

Acabo de empezar a medir. No sé si vale 20 o 500. El instrumento es de precisión. Hoy marca 107.

Reacción sensata: mi estimación pasa a ser ~107. La medición es lo único que tengo, y es confiable.

→ **Ganancia alta.**

### 11.3 La fórmula

**[A]** (de la ecuación 11.14 de Tsay, para el local trend model):

$$K_t = \frac{\Sigma_{t|t-1}}{V_t} = \frac{\Sigma_{t|t-1}}{\Sigma_{t|t-1} + \sigma_e^2}$$

En castellano:

$$K_t = \frac{\text{mi incertidumbre sobre el estado antes de ver el dato}}{\text{mi incertidumbre sobre el estado} + \text{el ruido del instrumento}}$$

**Símbolos:**

| Símbolo | Significado |
|---|---|
| $K_t$ | Kalman gain en $t$. Número entre 0 y 1 |
| $\Sigma_{t\mid t-1}$ | Varianza del estado antes de observar $y_t$ — "cuán perdido estoy" |
| $\sigma_e^2$ | Varianza del error de medición — "cuán malo es el instrumento" |
| $V_t$ | Varianza total de la sorpresa, $= \Sigma_{t\mid t-1} + \sigma_e^2$ |

Es una **proporción entre dos incertidumbres**. Está acotada entre 0 y 1 automáticamente, porque el numerador es parte del denominador.

### 11.4 Cómo entra en la actualización

**[A]** (ecuaciones 11.10 y 11.11):

$$\mu_{t|t} = \mu_{t|t-1} + K_t v_t$$

$$\Sigma_{t|t} = \Sigma_{t|t-1}(1 - K_t)$$

La primera dice: **estimación nueva = estimación vieja + $K_t$ × sorpresa.**

- Si $K_t = 0$: $\mu_{t|t} = \mu_{t|t-1}$. La observación no cambió nada.
- Si $K_t = 1$: $\mu_{t|t} = \mu_{t|t-1} + v_t = y_t$. La estimación **es** la observación.
- Si $K_t = 0.3$: incorporo el 30% de la sorpresa.

La segunda dice: **la incertidumbre siempre baja al observar**, y baja tanto más cuanto mayor sea la ganancia. Observar nunca empeora nuestro conocimiento del estado actual (dentro del modelo).

### 11.5 La formulación que hay que preferir

**Formulación imprecisa (evitar):**
> "El Kalman gain es la confianza que le tengo a la observación."

**Formulación correcta:**
> **El Kalman gain determina qué fracción de la innovation entra en la actualización del estado.**

¿Por qué importa la diferencia? Porque "confianza en la observación" sugiere que la ganancia mide una propiedad del instrumento. No es así: mide una **relación** entre dos incertidumbres. El mismo instrumento (mismo $\sigma_e$) puede producir ganancia alta o baja según cuán perdido esté yo respecto del estado.

**[A]** Tsay lo formula así: la ganancia *"is the regression coefficient of $\mu_t$ on $v_t$"* y *"is the factor that governs the contribution of the new shock $v_t$ to the state variable $\mu_t$."*

Que sea un **coeficiente de regresión** es literal y esclarecedor: la actualización del estado es, matemáticamente, la regresión del estado sobre la sorpresa.

### 11.6 ¿Bajo qué condiciones exactamente ocurre cada caso?

$K_t \approx 0$ requiere $\Sigma_{t|t-1} \ll \sigma_e^2$:
- El estado se mueve poco ($\sigma_\eta^2$ chico) **y**
- llevamos muchas observaciones acumuladas (la incertidumbre ya se estabilizó bajo) **y**
- el ruido de medición es grande.

$K_t \approx 1$ requiere $\Sigma_{t|t-1} \gg \sigma_e^2$:
- El estado se mueve mucho ($\sigma_\eta^2$ grande), **o**
- estamos al comienzo con inicialización difusa ($\Sigma_{1|0}$ enorme), **o**
- el ruido de medición es despreciable.

**[A]** Nótese lo señalado en la Sección 6.4: en el local trend model, $K_t$ **no depende de los datos**. Depende únicamente de $\Sigma_{1|0}$, $\sigma_e^2$ y $\sigma_\eta^2$. La secuencia $K_1, K_2, K_3, \ldots$ es determinista y calculable de antemano.

### 11.7 Ejemplo numérico: el caso Alcoa [B, aritmética sobre cifras A]

Con $\hat\sigma_e = 0.4803$ y $\hat\sigma_\eta = 0.0735$, la razón señal-ruido es $q = \sigma_\eta^2/\sigma_e^2 \approx 0.0234$.

Resolviendo la recursión de $\Sigma$ hasta su punto fijo, la ganancia de estado estacionario resulta $K_\infty \approx 0.142$.

**Comparación reveladora:** el ARIMA(0,1,1) ajustado a la misma serie tiene $\hat\theta = 0.858$, y $1 - \hat\theta = 0.142$. **Coinciden.**

Esto no es coincidencia: es la manifestación numérica de la equivalencia que Tsay declara — **[A]** el local trend model *es* el modelo de suavizado exponencial simple. La ganancia de estado estacionario del Kalman **es** el parámetro de suavizado de una EMA.

Interpretación en castellano: en régimen estacionario, este Kalman Filter incorpora aproximadamente el **14%** de cada sorpresa nueva. Es un filtro muy conservador, muy suave. Y es, funcionalmente, una media móvil exponencial.

**[B] Ésta es una lección importante y desinflacionaria**: para este modelo y esta serie, "correr un Kalman Filter" produce, en régimen, algo indistinguible de aplicar una EMA con un parámetro concreto. Volvemos sobre esto en la Sección 31.

### 11.8 Qué NO significa

- ❌ Ganancia alta ≠ "el filtro está funcionando bien". Significa que el modelo atribuye la variación al estado.
- ❌ Ganancia baja ≠ "el filtro es cauteloso e inteligente". Significa que el modelo atribuye la variación a ruido.
- ❌ La ganancia **no** se adapta a los datos en este modelo. No es que "el filtro detecta un cambio y reacciona". La secuencia está fijada por los parámetros.
- ❌ La ganancia **no** mide calidad de señal en un sentido predictivo.

---

## 12. Ciclo Predict → Observe → Update

### 12.1 Los siete pasos

**Paso 1 — Predict (estado).** Antes de ver $y_t$: ¿qué estado espero?
$$\mu_{t|t-1}$$

**Paso 2 — Predict (incertidumbre).** ¿Cuán inseguro estoy de esa predicción?
$$\Sigma_{t|t-1}$$

**Paso 3 — Observe.** Llega el dato: $y_t$.

**Paso 4 — Innovation.** ¿Cuánto se desvió la realidad de lo que esperaba?
$$v_t = y_t - \mu_{t|t-1}, \qquad V_t = \Sigma_{t|t-1} + \sigma_e^2$$

**Paso 5 — Kalman gain.** ¿Qué fracción de esa sorpresa incorporo al estado?
$$K_t = \Sigma_{t|t-1}/V_t$$

**Paso 6 — Update.** Actualizo estado e incertidumbre.
$$\mu_{t|t} = \mu_{t|t-1} + K_t v_t, \qquad \Sigma_{t|t} = \Sigma_{t|t-1}(1-K_t)$$

**Paso 7 — Move forward.** Proyecto al período siguiente usando la ecuación de estado.
$$\mu_{t+1|t} = \mu_{t|t}, \qquad \Sigma_{t+1|t} = \Sigma_{t|t} + \sigma_\eta^2$$

Y vuelvo al Paso 1.

**[A]** Los pasos 6 y 7 combinados son exactamente el algoritmo (11.14) de Tsay:

$$v_t = y_t - \mu_{t|t-1}$$
$$V_t = \Sigma_{t|t-1} + \sigma_e^2$$
$$K_t = \Sigma_{t|t-1}/V_t$$
$$\mu_{t+1|t} = \mu_{t|t-1} + K_t v_t$$
$$\Sigma_{t+1|t} = \Sigma_{t|t-1}(1 - K_t) + \sigma_\eta^2, \qquad t = 1,\ldots,T$$

### 12.2 Las dos fuerzas sobre la incertidumbre

Vale la pena aislar esto, porque es el corazón del algoritmo:

- **Observar reduce la incertidumbre:** $\Sigma_{t|t} = \Sigma_{t|t-1}(1-K_t) \leq \Sigma_{t|t-1}$.
- **El tiempo aumenta la incertidumbre:** $\Sigma_{t+1|t} = \Sigma_{t|t} + \sigma_\eta^2 > \Sigma_{t|t}$.

El filtro vive en el equilibrio entre esas dos fuerzas. Si el estado no se moviera ($\sigma_\eta = 0$), la incertidumbre caería monotónicamente hacia cero y el filtro convergería a la media muestral. Como el estado se mueve, la incertidumbre se estabiliza en un nivel positivo. **Nunca llegamos a saber el estado con certeza, por muchos datos que acumulemos.**

### 12.3 Ejemplo numérico completo, sin matrices

Tomemos un modelo de juguete: $\sigma_e^2 = 4$ (instrumento con desviación 2), $\sigma_\eta^2 = 1$ (el estado se mueve con desviación 1). Arrancamos con $\mu_{1|0} = 100$ y $\Sigma_{1|0} = 9$ (bastante inseguros, desviación 3).

**$t=1$. Llega $y_1 = 106$.**

| Paso | Cálculo | Resultado |
|---|---|---|
| Predicción del estado | dado | 100 |
| Incertidumbre previa | dada | 9 |
| Innovation | $106 - 100$ | **+6** |
| Varianza de la sorpresa | $9 + 4$ | 13 |
| Kalman gain | $9/13$ | **0.692** |
| Estado actualizado | $100 + 0.692 \times 6$ | **104.15** |
| Incertidumbre actualizada | $9 \times (1-0.692)$ | 2.77 |
| Predicción para $t=2$ | $= \mu_{1\mid1}$ | 104.15 |
| Incertidumbre para $t=2$ | $2.77 + 1$ | 3.77 |

*Comentario:* estábamos muy inseguros, así que incorporamos el 69% de una sorpresa de +6. Nos movimos casi hasta la observación.

**$t=2$. Llega $y_2 = 103$.**

| Paso | Cálculo | Resultado |
|---|---|---|
| Predicción | de antes | 104.15 |
| Incertidumbre previa | de antes | 3.77 |
| Innovation | $103 - 104.15$ | **−1.15** |
| Varianza de la sorpresa | $3.77+4$ | 7.77 |
| Kalman gain | $3.77/7.77$ | **0.485** |
| Estado actualizado | $104.15 - 0.485\times1.15$ | **103.59** |
| Incertidumbre actualizada | $3.77\times(1-0.485)$ | 1.94 |
| Para $t=3$ | $1.94+1$ | 2.94 |

*Comentario:* ya estamos menos inseguros, así que la ganancia bajó de 0.69 a 0.49.

**$t=3$. Llega $y_3 = 112$ (un salto grande).**

| Paso | Cálculo | Resultado |
|---|---|---|
| Predicción | de antes | 103.59 |
| Incertidumbre previa | de antes | 2.94 |
| Innovation | $112 - 103.59$ | **+8.41** |
| Varianza de la sorpresa | $2.94+4$ | 6.94 |
| Kalman gain | $2.94/6.94$ | **0.424** |
| Estado actualizado | $103.59 + 0.424\times8.41$ | **107.15** |
| Incertidumbre actualizada | $2.94\times(1-0.424)$ | 1.69 |

*Comentario crucial:* la observación fue **112**, pero el estado filtrado quedó en **107.15**. El filtro incorporó menos de la mitad del salto. **No dijo que 112 estuviera "mal"**; dijo que, bajo este modelo, una sorpresa de +8.4 se explica en parte por ruido de medición y en parte por movimiento del estado, y la repartió según las varianzas especificadas.

Si hubiéramos puesto $\sigma_e^2 = 0.5$ en lugar de 4, la ganancia habría sido mucho mayor y el estado habría saltado casi hasta 112. **Mismos datos, distinto estado estimado.** Ése es el punto.

### 12.4 Nota sobre la convergencia de la ganancia

Las ganancias fueron 0.692 → 0.485 → 0.424 → ... Están descendiendo hacia el punto fijo. Con $q = \sigma_\eta^2/\sigma_e^2 = 0.25$, la ganancia estacionaria es $K_\infty \approx 0.390$. Después de unas pocas iteraciones, el filtro se estabiliza y aplica siempre el mismo peso.

**[A]** Esto es exactamente lo que Tsay describe como *steady state* en la Sección 11.4.1.

---

## 13. Incertidumbre del estado

### 13.1 El filtro no entrega un número; entrega una distribución

Éste es un punto que se pierde con facilidad. El Kalman Filter no produce sólo $\hat\mu_t$. Produce:

$$\mu_t \mid F_t \sim N(\mu_{t|t},\ \Sigma_{t|t})$$

Es decir: una **media y una varianza**. Ignorar la varianza es tirar la mitad de la salida.

### 13.2 Por qué importa

Dos situaciones con la misma estimación puntual:

| | Situación A | Situación B |
|---|---|---|
| $\hat\mu_t$ | 100 | 100 |
| $SD$ | 0.1 | 10 |
| Interpretación | "está entre 99.8 y 100.2" | "está entre 80 y 120" |

**Son afirmaciones radicalmente distintas.** La primera es una afirmación fuerte; la segunda es casi no decir nada. Y sin embargo, si sólo reportamos "100", se ven idénticas.

### 13.3 De dónde sale la incertidumbre

**[A]** De la misma recursión que el estado:

$$\Sigma_{t|t} = \Sigma_{t|t-1}(1-K_t), \qquad \Sigma_{t+1|t} = \Sigma_{t|t} + \sigma_\eta^2$$

Y en el caso general (ecuaciones 11.62–11.63):

$$\Sigma_{t|t} = \Sigma_{t|t-1} - \Sigma_{t|t-1}Z_t'V_t^{-1}Z_t\Sigma_{t|t-1}$$
$$\Sigma_{t+1|t} = T_t\Sigma_{t|t-1}L_t' + R_tQ_tR_t'$$

### 13.4 Cómo la usa Tsay en la práctica

**[A]** En el Ejemplo 11.1, Tsay grafica el estado filtrado (Figura 11.3) y el estado suavizado (Figura 11.4) **junto con sus intervalos de confianza puntuales del 95%**, construidos como $\hat\mu \pm 2\sqrt{\hat\Sigma}$. Y observa que *"the confidence intervals for the smoothed state variables are also narrower than those of the filtered state variables."*

**[A]** En el Ejemplo 11.3 (J&J) hace lo mismo para las componentes de tendencia y estacionalidad.

### 13.5 Una advertencia sobre esas bandas [B]

Los intervalos que produce el filtro son **intervalos condicionales al modelo y a los parámetros**. Suponen que:
- el modelo es correcto;
- los parámetros son los verdaderos (aunque en la práctica se estimaron);
- las perturbaciones son gaussianas.

**No** son intervalos que incorporen incertidumbre sobre la especificación del modelo ni sobre la estimación de $\sigma_e, \sigma_\eta$. Son, en ese sentido, **optimistas**. Una banda del 95% que sale de un modelo mal especificado no cubre el 95% de nada.

### 13.6 Relevancia para ML [B]

**PREGUNTA ABIERTA.** ¿La incertidumbre estimada del estado (o del forecast) aporta información **incremental** respecto del valor puntual, para alguna tarea de decisión?

Hay un argumento a favor y uno en contra:

- **A favor:** una señal acompañada de "y estoy muy inseguro" es operativamente distinta de la misma señal con alta confianza. El proyecto IRIS explícitamente quiere producir un nivel de confianza.
- **En contra:** en el local trend model, $\Sigma_{t|t}$ **no depende de los datos**. Es una secuencia determinista que converge a una constante. Después del transitorio inicial, **la incertidumbre es la misma todos los días**, y por tanto no aporta información variable alguna.

Ese "en contra" es específico del local trend model y de los modelos lineales-gaussianos invariantes en el tiempo en general. En modelos con matrices de sistema variables en el tiempo (como el CAPM dinámico, donde $Z_t = (1, r_{M,t})$ depende de los datos), $\Sigma_{t|t}$ **sí** varía. Pero varía en función de los regresores, no de los residuos.

**No adoptamos nada.** Sólo registramos que "usar la incertidumbre del Kalman como feature" es una idea que hay que verificar caso por caso, porque en el caso más simple es literalmente una constante.

---

## 14. Initialization

### 14.1 El problema

El filtro es una recursión. Toda recursión necesita un punto de partida. Antes de ver $y_1$, ¿qué sabemos de $\mu_1$?

**[A]** Tsay parametriza esto como $\mu_1 \sim N(\mu_{1|0}, \Sigma_{1|0})$, con dos objetos:

- $\mu_{1|0}$ = **media inicial del estado**: nuestra mejor conjetura sobre dónde arranca.
- $\Sigma_{1|0}$ = **varianza inicial del estado**: cuánta confianza tenemos en esa conjetura.

### 14.2 Diffuse initialization

**[A]** ¿Qué pasa si no sabemos nada? Hacemos $\Sigma_{1|0} \to \infty$.

Tsay calcula qué ocurre en el límite:

$$\mu_{2|1} = \mu_{1|0} + \frac{\Sigma_{1|0}}{\Sigma_{1|0}+\sigma_e^2}(y_1 - \mu_{1|0}) \xrightarrow{\Sigma_{1|0}\to\infty} y_1$$

$$\Sigma_{2|1} = \frac{\Sigma_{1|0}}{\Sigma_{1|0}+\sigma_e^2}\sigma_e^2 + \sigma_\eta^2 \xrightarrow{\Sigma_{1|0}\to\infty} \sigma_e^2 + \sigma_\eta^2$$

**[A]** *"This is equivalent to treating $y_1$ as fixed and assuming $\mu_1 \sim N(y_1, \sigma_e^2)$. In the literature, this approach to initializing the Kalman filter is called **diffuse initialization** because a very large $\Sigma_{1|0}$ means one is uncertain about the initial condition."*

**En castellano:** "no sé nada del estado inicial" se implementa diciendo "mi incertidumbre inicial es infinita". El resultado práctico es elegante: **el filtro simplemente adopta la primera observación como punto de partida**. Con $\Sigma_{1|0}$ infinito, la ganancia inicial es 1: le hago caso completo al primer dato, porque no tengo nada más.

### 14.3 ¿Cuánto importa la inicialización?

Ésta es la pregunta práctica.

**Sobre el filtrado:**
- Al comienzo, mucho. La ganancia $K_1$ depende directamente de $\Sigma_{1|0}$.
- El efecto **se disipa**: la recursión de $\Sigma$ converge hacia su punto fijo independientemente de dónde empezó (siempre que el modelo sea estable). **[A]** Ésa es la lógica del *steady state* de la Sección 11.4.1.
- Pero **[A]** Tsay señala explícitamente en el Ejemplo 11.1 que *"the width of the 95% confidence interval of $\mu_{1|1}$ depends on the initial value $\Sigma_{1|0}$."* Es decir: **los primeros estados filtrados sí están afectados**.

**Sobre el suavizado:**
- **[A]** *"It is obvious that based on the results of Kalman filtering, state smoothing is not affected by the diffuse initialization for $t = T, \ldots, 2$."* Sólo $\mu_1$ dado $F_T$ queda afectado, y en el límite difuso resulta $\mu_{1|T} = y_1 + \sigma_e^2 q_1$, con $\Sigma_{1|T} = \sigma_e^2 - \sigma_e^4 M_1$.

**Conclusión [A]:** el efecto de la inicialización **se desvanece**, pero **no es cero al principio**. No se debe asumir automáticamente que "la inicialización nunca importa".

### 14.4 La alternativa que sugiere Tsay

**[A]** *"we suggest using diffuse initialization when little is known about the initial value $\mu_1$. However, it might be hard to justify the use of a random variable with infinite variance in real applications. If necessary, one can treat $\mu_1$ as an additional parameter of the state-space model and estimate it jointly with other parameters. This latter approach is closely related to the exact maximum-likelihood estimation of Chapters 2 and 8."*

Es decir: hay dos caminos honestos. Declarar ignorancia total (difusa), o tratar $\mu_1$ como un parámetro más y estimarlo. Lo que **no** es honesto es fingir una certeza inicial que no se tiene.

### 14.5 Un caso donde la inicialización es obligatoriamente difusa

**[A]** En la Sección 11.3.3, al representar una regresión lineal ordinaria como state-space (donde $s_t = \beta$ es constante), Tsay observa: *"Since the state vector is fixed, a diffuse initialization should be used."*

Tiene sentido: si el estado es un coeficiente de regresión desconocido y constante, no tenemos ninguna información previa sobre él. La inicialización difusa reproduce exactamente el estimador de mínimos cuadrados — y en efecto, en el Ejemplo 11.2, coincide con OLS hasta el sexto decimal.

**[A]** Caso contrario: cuando el modelo es **estacionario**, la inicialización correcta no es difusa sino la distribución estacionaria. Tsay lo ilustra con un AR(1) $y_t = 0.6y_{t-1}+a_t$, $\sigma_a=0.4$: el programa usa $\Sigma_{1|0} = \mathrm{Var}(y_t) = 0.4^2/(1-0.6^2) = 0.25$ y $\mu_{1|0}=0$.

### 14.6 Relevancia para ML [B]

**[B]** Si alguna vez se calculara una feature filtrada en un esquema walk-forward, la elección de inicialización en cada fold afectaría los primeros valores de cada tramo. Esto introduce una decisión adicional que hay que documentar y auditar:

- ¿Se reinicia el filtro en cada fold (efecto de inicialización en cada tramo)?
- ¿Se arrastra el estado entre folds (¿y entonces qué información cruza la frontera del fold)?

**PREGUNTA ABIERTA.** No hay respuesta obvia y el capítulo no la aborda. Queda registrada como un punto de diseño que exigiría decisión explícita **si alguna vez** se llegara a esa etapa.

---

## 15. Estimación de parámetros

### 15.1 La distinción fundamental

Ésta es una confusión frecuentísima y hay que despejarla.

| | **Parámetros** | **Estados** |
|---|---|---|
| Ejemplo | $\sigma_e^2$, $\sigma_\eta^2$, $\theta$, matrices del sistema | $\mu_t$, $s_t$ |
| ¿Cambian con $t$? | **No.** Son fijos para toda la muestra | **Sí.** Uno por cada instante |
| ¿Cuántos hay? | Pocos (2 en el local trend model) | $T$ (uno por observación) |
| ¿Cómo se obtienen? | Máxima verosimilitud sobre toda la muestra | Recursión del Kalman Filter, dado el modelo y los parámetros |
| Analogía en ML | Hiperparámetros / pesos entrenados | Activaciones para un input dado |

**Estimar parámetros ≠ filtrar estados.** Son dos operaciones distintas, en dos etapas distintas.

### 15.2 El flujo típico

**[A]** El orden de operaciones que sigue Tsay es:

1. **Proponer la estructura del modelo** (¿qué es el estado? ¿cómo evoluciona? ¿cómo se relaciona con la observación?).
2. **Estimar los parámetros** por máxima verosimilitud.
3. **Dado modelo + parámetros**, ejecutar el Kalman Filter.
4. **Obtener** likelihood, estados filtrados, estados suavizados, forecasts, diagnósticos.

En el Ejemplo 11.1, Tsay dice explícitamente: *"Here we treat the two estimates as given and use the model to demonstrate application of the Kalman filter."*

### 15.3 Cómo el Kalman Filter facilita la estimación

Aquí está el truco elegante del capítulo. **[A]** La verosimilitud de los datos se factoriza como:

$$p(y_1,\ldots,y_T \mid \sigma_e,\sigma_\eta) = p(y_1\mid\cdot)\prod_{t=2}^{T} p(y_t \mid F_{t-1}, \cdot) = p(y_1\mid\cdot)\prod_{t=2}^{T} p(v_t \mid F_{t-1}, \cdot)$$

y como cada $v_t \sim N(0, V_t)$, tomando logaritmos:

$$\ln L(\sigma_e,\sigma_\eta) = -\frac{T}{2}\ln(2\pi) - \frac{1}{2}\sum_{t=1}^{T}\left[\ln(V_t) + \frac{v_t^2}{V_t}\right] \tag{11.25}$$

**Esto es hermoso y merece explicación.** La verosimilitud de una serie con dependencia temporal compleja se ha reducido a **una suma sobre errores de predicción independientes**. Y el Kalman Filter produce exactamente $v_t$ y $V_t$ en cada paso.

Por eso Tsay abre el capítulo diciendo que el state-space es útil *"especially for simplifying maximum-likelihood estimation"*. No es un comentario menor: es una de las dos razones prácticas principales del enfoque.

En la literatura esto se conoce como **prediction error decomposition**.

**[A]** Y funciona igual con valores faltantes: *"the log-likelihood function, including cases with missing values, can be evaluated recursively via the Kalman filter."*

### 15.4 Software

**[A]** Tsay menciona Matlab, RATS y S-Plus. Usa **SsfPack** (Koopman, Shephard y Doornik, 1999), disponible en S-Plus y OX.

*(Detalles de sintaxis SsfPack omitidos deliberadamente en este informe — ver Sección 60.)*

### 15.5 El riesgo de leakage escondido aquí [B]

**Éste es el punto que más importa para IRIS de toda esta sección.**

La recursión del Kalman Filter **es causal**: $\mu_{t|t}$ usa sólo $y_1,\ldots,y_t$. Eso es cierto.

Pero los parámetros $\sigma_e, \sigma_\eta$ que alimentan esa recursión se estimaron **maximizando la verosimilitud sobre toda la muestra $y_1,\ldots,y_T$**.

Entonces, estrictamente:

$$\mu_{t|t} = f(y_1,\ldots,y_t;\ \hat\sigma_e(y_1,\ldots,y_T),\ \hat\sigma_\eta(y_1,\ldots,y_T))$$

**El estado "filtrado" depende del futuro a través de los parámetros.**

Para el propósito de Tsay —análisis estadístico de una muestra histórica— esto es completamente estándar y correcto. No es un error. Pero **[B]** para construir una feature causal en un backtest, es una fuente de contaminación real.

Se desarrolla en detalle en la Sección 32.

---

## 16. General Linear State-Space Model

### 16.1 Primero la idea, sin matrices

Un state-space general dice dos cosas:

**Ecuación de estado:** *"Cómo evoluciona internamente el sistema."*
> El estado de mañana es una transformación del estado de hoy, más un empujón aleatorio, más quizá un término conocido.

**Ecuación de observación:** *"Cómo el estado produce lo que veo."*
> Lo que observo es una transformación del estado de hoy, más un error de medición, más quizá un término conocido.

Todo lo demás son detalles de cómo se escriben esas dos transformaciones.

### 16.2 Las ecuaciones

**[A]** (ecuaciones 11.26 y 11.27):

$$s_{t+1} = d_t + T_t s_t + R_t\eta_t \tag{11.26}$$

$$y_t = c_t + Z_t s_t + e_t \tag{11.27}$$

con $\eta_t \sim N(0, Q_t)$ y $e_t \sim N(0, H_t)$, $Q_t$ y $H_t$ definidas positivas, $\{e_t\}$ y $\{\eta_t\}$ independientes (*"but this condition can be relaxed if necessary"*), y $s_1 \sim N(\mu_{1|0}, \Sigma_{1|0})$ dado e independiente de $e_t$, $\eta_t$ para $t>0$.

### 16.3 La función de cada objeto

**No hace falta memorizar dimensiones. Hace falta entender qué hace cada uno.**

| Objeto | Nombre | ¿Qué hace? | Analogía |
|---|---|---|---|
| $s_t$ | **estado** ($m$-dim) | La cantidad latente que seguimos | "Dónde estoy y a qué velocidad voy" |
| $y_t$ | **observación** ($k$-dim) | Lo que medimos | "Lo que marca el GPS" |
| $T_t$ | **matriz de transición** ($m\times m$) | Cómo el estado de hoy se convierte en el de mañana | "Las leyes del movimiento" |
| $Z_t$ | **matriz de observación** ($k\times m$) | Cómo el estado genera lo observado | "Cómo el instrumento lee el sistema" |
| $R_t$ | **matriz de selección** ($m\times n$) | Qué componentes del estado reciben shocks aleatorios | "Dónde entra el ruido del proceso" |
| $Q_t$ | **covarianza del ruido de estado** | Cuánto se mueve el estado por su cuenta | "Cuán errático es el sistema" |
| $H_t$ | **covarianza del ruido de observación** | Cuánto miente el instrumento | "Cuán malo es el sensor" |
| $d_t$ | **término determinista del estado** | Deriva o efecto conocido en la transición | "Una corriente constante que me empuja" |
| $c_t$ | **término determinista de la observación** | Offset conocido en la medición | "Un sesgo fijo del instrumento" |

**[A]** Tsay llama a $T_t, R_t, Q_t, Z_t, H_t$ las **system matrices**, y observa: *"These matrices are often sparse, and they can be functions of some parameters $\theta$, which can be estimated by the maximum-likelihood method."*

**[A]** Y sobre (11.26): *"Equation (11.26) is the state or transition equation that describes a **first-order Markov Chain** to govern the state transition with innovation $\eta_t$."*

Esa observación de "cadena de Markov de primer orden" es la que fija el significado de "estado": **toda la memoria del sistema está en $s_t$**. Nada anterior a $s_t$ aporta información adicional sobre $s_{t+1}$.

### 16.4 Verificación: el local trend model es un caso particular

$$m = 1,\quad k=1,\quad T_t = 1,\quad Z_t = 1,\quad R_t = 1,\quad Q_t = \sigma_\eta^2,\quad H_t = \sigma_e^2,\quad d_t = c_t = 0$$

Sustituyendo: $s_{t+1} = s_t + \eta_t$ e $y_t = s_t + e_t$. Exactamente (11.1)–(11.2). ✓

### 16.5 Time-invariant vs. time-varying

**[A]** *"In many applications, the system matrices are time invariant. However, these matrices can be time varying, making the state-space model flexible."*

Ésta es la fuente de casi toda la flexibilidad del enfoque. Dos ejemplos concretos del capítulo:

- **CAPM con coeficientes variables:** $Z_t = (1, r_{M,t})$ — la matriz de observación **contiene un dato**. Cambia en cada instante.
- **Regresión lineal:** $Z_t = x_t'$ — igual.

Nótese algo sutil: en estos casos, las "matrices del sistema" incorporan variables observadas. El modelo no es estrictamente "invariante"; es condicionalmente lineal dado el regresor.

### 16.6 Inicialización difusa en el caso general

**[A]** Se implementa como $\Sigma_{1|0} = \Sigma_* + \lambda\Sigma_\infty$, con $\Sigma_*$ y $\Sigma_\infty$ simétricas definidas positivas $m\times m$ y $\lambda$ un número real grande que puede tender a infinito. Permite que **algunas** componentes del estado sean difusas y otras no.

### 16.7 Qué modelos entran en este marco

**[A]** Tsay enumera: *"Examples include the ARIMA models, dynamic linear models with unobserved components, time-varying regression models, and stochastic volatility models."*

La mención de **stochastic volatility models** es un puente hacia el Capítulo 12 (ver Sección 30 de este informe).

---

## 17. Model Transformation

### 17.1 El objetivo

**[A]** Tsay lo enuncia así: *"To appreciate the flexibility of the state-space model, we rewrite some well-known econometric and financial models in state-space form."*

El objetivo es apreciar **flexibilidad del lenguaje**, no descubrir modelos nuevos. Es como decir "mirá todas las cosas que se pueden escribir en esta notación".

$$\boxed{\text{entender la flexibilidad del lenguaje state-space}}$$

### 17.2 La tabla de transformaciones

| Modelo original | ¿Qué se convierte en estado? | ¿Qué ganamos con la representación? |
|---|---|---|
| **CAPM con coef. variables** (11.3.1) | $s_t = (\alpha_t,\beta_t)'$ — los propios coeficientes | Permitir que evolucionen; estimarlos y suavizarlos con la maquinaria estándar |
| **ARMA($p,q$)** (11.3.2) | Depende del enfoque: pronósticos futuros (Akaike), combinaciones de $y$ y $a$ (Harvey), shocks rezagados (Aoki) | Estimación por ML vía Kalman; manejo de faltantes; un solo algoritmo para muchos modelos |
| **Regresión lineal** (11.3.3) | $s_t = \beta$, **constante** en $t$ | Recursividad (estimación online); base para permitir $\beta$ variable |
| **Regresión con errores ARMA** (11.3.4) | $s_t^* = (s_t', \beta_t')'$ — estado del ARMA *más* los coeficientes | Tratar coeficientes y dinámica del error en un solo marco |
| **Unobserved components / STSM** (11.3.5) | $s_t$ = (tendencia, pendiente, estacional, ciclo) | Estimar componentes no observadas simultáneamente, con incertidumbre |

### 17.3 La distinción que hay que mantener

$$\text{representación más conveniente} \ \neq\ \text{nuevo modelo estadístico}$$

Éste es el eje de toda la sección 11.3. Observemos los dos casos extremos:

**Caso 1 — Sólo cambia la representación.** Una regresión lineal escrita como state-space **es la misma regresión lineal**. Mismo modelo, misma verosimilitud, mismas estimaciones. **[A]** El Ejemplo 11.2 lo verifica numéricamente: OLS da $(0.1982, 1.0457)$ con errores estándar $(0.6302, 0.1453)$; el state-space da $(0.1982025, 1.045702)$ con $(0.6302091, 0.1453139)$. *"As expected, the result is in total agreement with that of the OLS method."*

**Caso 2 — Cambia el modelo.** Un CAPM con $\beta_t$ que evoluciona como paseo aleatorio **no es** el CAPM de coeficientes fijos. Es un modelo **distinto**, con parámetros adicionales ($\sigma_\eta$, $\sigma_\epsilon$) y con implicaciones estadísticas diferentes. El state-space es el vehículo para escribirlo, pero el cambio de modelo es real.

**[B] Regla operativa:** al leer "lo pasamos a state-space", siempre hay que preguntar: *¿esto cambió el modelo, o sólo la notación?* La respuesta determina si esperar resultados diferentes o idénticos.

### 17.4 La lección de las múltiples representaciones de ARMA

**[A]** Tsay presenta **tres** formas distintas de escribir el mismo ARMA como state-space:

- **Akaike (1975):** el estado es *"the minimum collection of variables that contains all the information needed to produce forecasts at the forecast origin $t$"*, concretamente $s_t = (y_{t|t}, y_{t+1|t}, \ldots, y_{t+m-1|t})'$ — es decir, **los pronósticos futuros son el estado**. Dimensión $m = \max(p, q+1)$.
- **Harvey (1993):** estado $m$-dimensional cuyo primer elemento es $y_t$; los demás se construyen recursivamente. **Sin error de medición.** *"It has an advantage that the AR and MA coefficients are directly used in the system matrices."*
- **Aoki (1987):** para un MA, $s_t = (a_{t-q},\ldots,a_{t-1})'$ — los **shocks rezagados** son el estado. Aquí *"$a_t$ appears in both state and measurement equations"*.

**[A]** Conclusión de Tsay: *"In summary, there are many state-space representations for an ARMA model. Each representation has its pros and cons. For estimation and forecasting purposes, one can choose any one of those representations."*

**Ésta es una lección conceptual de primer orden:**

$$\boxed{\text{una misma serie observada puede admitir distintas representaciones de estado}}$$

Tres modelos distintos, tres definiciones de "estado" radicalmente diferentes (pronósticos futuros / valores observados / shocks pasados), y **la misma serie observada, con la misma distribución de probabilidad**.

**[B]** Por tanto: preguntar "¿cuál es el estado del mercado?" es una pregunta mal planteada si no se especifica primero el modelo. **Los estados internos dependen de la representación elegida.** No hay un "estado verdadero" esperando ser descubierto; hay estados definidos por especificaciones.

Esto anticipa perfectamente la advertencia de no unicidad de la Sección 28.

### 17.5 La dirección recíproca

**[A]** *"for a time-invariant coefficient state-space model in Eqs. (11.26) and (11.27), one can use the Cayley–Hamilton theorem to show that the observation $y_t$ follows an ARMA($m,m$) model, where $m$ is the dimension of the state vector."*

Es decir: cualquier state-space lineal invariante produce observaciones ARMA. **Ninguna de las dos clases contiene dinámicas que la otra no pueda representar.** La equivalencia es completa.

---

## 18. Time-Varying Coefficients

### 18.1 La idea, que es más importante que el ejemplo

**[A]** Tsay usa el CAPM como vehículo, pero la idea central no es el CAPM. Es:

$$\boxed{\text{un coeficiente puede ser un ESTADO que evoluciona en el tiempo}}$$

### 18.2 El modelo

**[A]** (ecuación 11.29):

$$r_t = \alpha_t + \beta_t r_{M,t} + e_t, \qquad e_t \sim N(0,\sigma_e^2)$$
$$\alpha_{t+1} = \alpha_t + \eta_t, \qquad \eta_t \sim N(0,\sigma_\eta^2)$$
$$\beta_{t+1} = \beta_t + \epsilon_t, \qquad \epsilon_t \sim N(0,\sigma_\epsilon^2)$$

donde $r_t$ es el exceso de retorno de un activo, $r_{M,t}$ el exceso de retorno del mercado, y las innovaciones $\{e_t,\eta_t,\epsilon_t\}$ son mutuamente independientes.

**[A]** *"This CAPM allows for time-varying $\alpha$ and $\beta$ that evolve as a random walk over time."*

En forma state-space: $s_t = (\alpha_t,\beta_t)'$, $T_t = R_t = I_2$, $d_t = c_t = 0$, $Z_t = (1, r_{M,t})$, $H_t = \sigma_e^2$, $Q_t = \mathrm{diag}\{\sigma_\eta^2, \sigma_\epsilon^2\}$.

**Detalle notable:** $Z_t$ contiene el retorno de mercado. Es una matriz de observación que **depende de los datos**.

### 18.3 La comparación en castellano

| | Regresión tradicional | State-space con coeficientes variables |
|---|---|---|
| Coeficiente | $\beta$ = constante para toda la historia | $\beta_t$ = cambia lentamente |
| Supuesto | "La relación entre X e Y fue siempre la misma" | "La relación pudo derivar" |
| Parámetros | $\beta$, $\sigma_e$ | $\sigma_\eta$, $\sigma_\epsilon$, $\sigma_e$ (los $\beta_t$ son estados, no parámetros) |
| ¿Qué se estima? | El valor de $\beta$ | **Cuánto se mueve** $\beta$, y luego los $\beta_t$ se filtran |

En modo sencillo:

> "La relación entre X e Y no tiene por qué tener exactamente el mismo coeficiente durante toda la historia."

**Observación importante:** el modelo de coeficientes variables **contiene** al de coeficientes fijos como caso particular. Si $\sigma_\eta = \sigma_\epsilon = 0$, entonces $\alpha$ y $\beta$ no se mueven nunca y recuperamos la regresión clásica. **[A]** Tsay hace esta observación explícita en la Sección 11.3.3: *"If $\sigma_i = 0$, then $\beta_i$ is time invariant."*

Esto es metodológicamente valioso: permite **testear** si los coeficientes varían, en lugar de asumirlo. Y es exactamente lo que hace en el Ejemplo 11.2 — con un resultado que conviene leer con atención (Sección 19).

### 18.4 Conexión con capítulos anteriores

- **Cap. 4 (no estacionariedad, cambios estructurales, Markov switching):** el Cap. 4 ofrecía una forma de "la regla cambia" con estados **discretos**. Aquí tenemos otra, con deriva **continua y suave**. Son alternativas, no la misma cosa.
- **Cap. 2 (estacionariedad):** un $\beta_t$ que sigue un paseo aleatorio es no estacionario por construcción. Puede deambular sin límite.

### 18.5 Qué NO significa un coeficiente variable

Ésta es la advertencia central de la sección.

- ❌ **No** demuestra que exista un "régimen" que cambió. Un paseo aleatorio suave no tiene regímenes; tiene deriva.
- ❌ **No** demuestra que la relación real haya cambiado. Demuestra que **un modelo que permite cambio produce una trayectoria estimada**. Un modelo que permite variación casi siempre produce alguna variación estimada, aunque la verdadera relación sea constante.
- ❌ **No** significa que el coeficiente estimado en $t$ fuera conocible en $t$, si se obtuvo por smoothing.
- ❌ Y sobre todo: **la existencia de una herramienta para modelar coeficientes variables no es evidencia de que los coeficientes varíen.** El Ejemplo 11.2 lo demuestra en la dirección contraria.

### 18.6 Como posible feature — y las preguntas que abre [B]

**[B]** Una aplicación futura *concebible* sería usar $\hat\beta_{t|t}$ (filtrado, no suavizado) como descripción dinámica de una relación entre dos series.

Antes de considerar siquiera esa posibilidad, habría que responder:

1. ¿Es **estable**? ¿O es ruido con apariencia de tendencia?
2. ¿Es **causal**? ¿Se usó $\hat\beta_{t|t}$ o $\hat\beta_{t|T}$? ¿Con qué parámetros estimados y con qué datos?
3. ¿Aporta información **adicional** respecto a una beta rolling calculada con una ventana simple?
4. ¿Es sólo una **transformación complicada del mismo input**?
5. ¿Depende excesivamente de $\sigma_\eta, \sigma_\epsilon$? (Si esos parámetros controlan cuánto se mueve la beta, **nosotros** estamos eligiendo cuánto se mueve.)
6. ¿Es **robusto fuera de muestra**?
7. ¿Y qué par de series tendría sentido en MNQ? El CAPM tiene una justificación económica; una beta arbitraria entre dos series de futuros, no necesariamente.

**NO adoptamos ninguna feature.** Todas éstas son PREGUNTAS ABIERTAS.

---

## 19. Ejemplo CAPM/GM — registro detallado

**[A] Todos los datos de esta sección provienen del Ejemplo 11.2 de Tsay.**

Este ejemplo es, para nuestros fines, **el más instructivo de los tres**, y por una razón que se pierde con facilidad: es un ejemplo donde **la respuesta fue "no".**

### 19.1 Ficha del experimento

| Campo | Valor |
|---|---|
| **Activo** | Acción de General Motors (GM) |
| **Variable** | Exceso de retorno simple **mensual**, en porcentaje |
| **Período** | Enero 1990 – Diciembre 2003 |
| **Observaciones** | 168 |
| **Retorno de mercado** | Exceso de retorno simple mensual del índice compuesto S&P 500 |
| **Origen de los datos** | Los mismos usados en el Cap. 9 del libro (diferido en nuestro estudio; irrelevante para entender este ejemplo) |

### 19.2 Paso 1 — Market model de coeficientes fijos, por OLS

**[A]**

$$r_t = \alpha + \beta r_{M,t} + e_t$$

| Coeficiente | Valor | Error estándar | $t$ | $p$ |
|---|---|---|---|---|
| Intercepto ($\alpha$) | 0.1982 | 0.6302 | 0.3145 | 0.7535 |
| Pendiente ($\beta$, sobre S&P) | 1.0457 | 0.1453 | 7.1962 | 0.0000 |

Diagnósticos:

| Estadístico | Valor |
|---|---|
| $R^2$ | 0.2378 |
| $R^2$ ajustado | 0.2332 |
| Durbin–Watson | 2.0290 |
| Jarque–Bera | 2.5348 ($p$ = 0.2816) |
| Ljung–Box | 24.2132 ($p$ = 0.3362) |
| Error estándar residual | 8.13 (166 g.l.) |

**[A]** Modelo ajustado: $r_t = 0.20 + 1.0457\,r_{M,t} + e_t$, $\hat\sigma_e = 8.13$. *"Based on the residual diagnostics, the model appears to be adequate for the GM stock returns with adjusted $R^2 = 23.3\%$."*

**Nótese:** el intercepto no es significativo ($p = 0.75$). La beta sí lo es y está cerca de 1.

### 19.3 Paso 2 — El mismo modelo, estimado como state-space

**[A]** Estimando el modelo de coeficientes fijos por máxima verosimilitud en forma state-space:

- $\hat\sigma_e = 8.130114$
- Estado suavizado: $(0.1982025,\ 1.045702)$
- Raíces de las varianzas del estado: $(0.6302091,\ 0.1453139)$

**[A]** *"As expected, the result is in total agreement with that of the OLS method."*

**Este resultado merece detenerse.** Coinciden hasta el sexto decimal: coeficientes y errores estándar. Es una **demostración numérica** de la tesis central de la Sección 8: escribir un modelo en forma state-space **no cambia el modelo**. Es la misma verosimilitud, calculada de otra manera.

**[B]** Si alguien nos dijera "usamos un Kalman Filter para estimar la relación", esta comparación muestra que eso, por sí solo, no significa nada. Un Kalman Filter aplicado a un modelo de coeficientes fijos **es** mínimos cuadrados ordinarios.

### 19.4 Paso 3 — CAPM con coeficientes variables

**[A]** Ahora se estima el modelo (11.29), permitiendo que $\alpha_t$ y $\beta_t$ deriven como paseos aleatorios. Estimaciones de máxima verosimilitud:

$$\hat\sigma_\eta = 4.907845\times10^{-5} \approx 4.91\times10^{-5}$$
$$\hat\sigma_\epsilon = 1.219885\times10^{-2} \approx 1.22\times10^{-2}$$
$$\hat\sigma_e = 8.125213$$

**[A]** Comentario textual de Tsay: *"Note that estimates of $\sigma_\eta$ and $\sigma_\varepsilon$ are $4.91\times10^{-5}$ and $1.22\times10^{-2}$, respectively. These estimates are close to zero, indicating that $\alpha_t$ and $\beta_t$ of the time-varying market model are **essentially constant** for the GM stock returns. This is in agreement with the fact that the fixed-coefficient market model fits the data well."*

**[A]** Figura 11.5: (a) los excesos de retorno mensuales; (b) los retornos esperados $r_{t|T}$; (c) la estimación de $\alpha_t$; (d) la estimación de $\beta_t$. En el gráfico publicado, el eje vertical de $\alpha_t$ muestra el mismo valor (0.20643) en ambos extremos, y el de $\beta_t$ recorre aproximadamente 1.02 a 1.06.

**[A]** *"Given the tightness in the vertical scale, these two time plots confirm the assertion that a fixed-coefficient market model is adequate for the monthly GM stock return."*

### 19.5 Qué se puede y qué NO se puede concluir

**Se puede [A]:**
- Para GM, retornos mensuales, 1990–2003, contra S&P 500: **los coeficientes son esencialmente constantes**.
- Permitir explícitamente variación temporal y estimar por ML **no produjo variación**. La estimación de máxima verosimilitud "eligió" $\sigma \approx 0$.
- El modelo de coeficientes fijos ya era adecuado según sus diagnósticos.
- Escribir la regresión como state-space reproduce OLS exactamente.

**NO se puede:**
- ❌ "Los betas financieros son variables." Este ejemplo dice lo contrario para este caso.
- ❌ "Los betas financieros son fijos." Este ejemplo es **un** caso: una acción, frecuencia mensual, 14 años, contra un índice.
- ❌ Extrapolar a MNQ, a otra frecuencia o a otro par de series. Es plausible que a frecuencia intradiaria el comportamiento sea distinto — **y también es plausible que no**. No lo sabemos.

### 19.6 La lección metodológica [B]

Este ejemplo es un antídoto excelente contra un sesgo muy común en trabajo cuantitativo:

> "El mercado cambia constantemente, por lo tanto los modelos deben tener parámetros adaptativos."

Aquí se dio libertad completa al modelo para adaptarse. **Y no la usó.** La verosimilitud prefirió coeficientes constantes.

$$\boxed{\text{permitir dinámica} \neq \text{encontrar dinámica}}$$

**[B]** El corolario para IRIS es directo: si en algún momento se considerara un modelo con parámetros dinámicos, el benchmark obligatorio es **el mismo modelo con parámetros fijos**. Si el dinámico no supera al fijo, la complejidad adicional no se justifica. Y hay una manera limpia de testear esto dentro del propio marco: estimar $\sigma_\eta$ y ver si es distinguible de cero.

**Detalle adicional que conviene notar:** Tsay usa **smoothing** (`SsfCondDens` con tarea `STSMO`) para producir los gráficos de $\alpha_t$ y $\beta_t$. Es decir, las trayectorias de la Figura 11.5(c)-(d) son $\alpha_{t|T}$ y $\beta_{t|T}$: **usan toda la muestra**. Es perfectamente apropiado para el propósito de Tsay (describir retrospectivamente si los coeficientes se movieron), pero **[B]** esas trayectorias **no** son lo que se habría conocido en tiempo real.

---

## 20. ARMA y regresiones en state-space

**PRIORIDAD CONCEPTUAL.** Esta sección se estudia por su idea, no por sus matrices. Las derivaciones algebraicas de las tres representaciones ARMA se omiten deliberadamente (ver Sección 60).

### 20.1 Por qué existen varias representaciones del mismo ARMA

**[A]** Ya vimos las tres (Sección 17.4). La pregunta interesante es: **¿por qué son distintas si describen lo mismo?**

Porque "estado" no es un objeto único. Es cualquier colección de variables que cumpla dos condiciones:

1. Contiene toda la memoria relevante (propiedad markoviana de primer orden).
2. Genera la observación mediante la ecuación de observación.

Y esas dos condiciones **no determinan una única elección**. Podemos elegir:

- Los pronósticos futuros como estado (Akaike).
- Combinaciones de valores observados y shocks (Harvey).
- Los shocks rezagados (Aoki).

Cada elección resume la misma información de una forma diferente. Es como describir la posición de un punto en coordenadas cartesianas o polares: distintas coordenadas, mismo punto.

**[A]** *"Each representation has its pros and cons. For estimation and forecasting purposes, one can choose any one of those representations."*

### 20.2 Una idea que sí vale la pena registrar: la fórmula de actualización de pronósticos

**[A]** Al derivar el enfoque de Akaike, Tsay obtiene (ecuación 11.35):

$$y_{t+j|t+1} = y_{t+j|t} + \psi_{j-1}a_{t+1}$$

**[A]** *"This result is referred to as the forecast updating formula of ARMA models. It provides a simple way to update the forecast from origin $t$ to origin $t+1$ when $y_{t+1}$ becomes available. The new information of $y_{t+1}$ is contained in the innovation $a_{t+1}$, and the time-$t$ forecast is revised based on this new information with weight $\psi_{j-1}$."*

Ésta es **exactamente la misma lógica que el Kalman Filter**: pronóstico nuevo = pronóstico viejo + peso × sorpresa. El $\psi_{j-1}$ juega el papel del Kalman gain. No es casualidad: es la misma idea vista desde el lenguaje ARMA.

**[B]** Vale la pena registrarlo porque desmitifica el Kalman Filter: la estructura "actualizar una creencia proporcionalmente a la sorpresa" no es exclusiva del state-space. Ya estaba en el Capítulo 2.

### 20.3 Regresión lineal como state-space

**[A]** (Sección 11.3.3). El modelo $y_t = x_t'\beta + e_t$ se escribe con $s_t = \beta$ **para todo $t$**, es decir, un estado que no se mueve:

$$T_t = I_p,\quad Z_t = x_t',\quad d_t = c_t = 0,\quad Q_t = 0,\quad H_t = \sigma_e^2$$

Nótese $Q_t = 0$: **sin ruido en la ecuación de estado**. El estado es literalmente constante.

**[A]** *"Since the state vector is fixed, a diffuse initialization should be used."*

**[A]** Extensión natural: $\beta_{t+1} = \beta_t + R_t\eta_t$, con $R_t = (\sigma_1,\ldots,\sigma_p)$ y $\sigma_i \geq 0$. *"If $\sigma_i = 0$, then $\beta_i$ is time invariant."*

**Esto es elegante y vale la pena notar:** permite que **algunos** coeficientes sean fijos y **otros** variables, dentro del mismo modelo. La regresión clásica es el caso $\sigma_i = 0 \ \forall i$.

**[B]** Interés metodológico: convierte "¿este coeficiente varía?" en una pregunta con respuesta numérica ($\hat\sigma_i$ ¿es distinguible de 0?). Eso es más limpio que ajustar regresiones en subperíodos y comparar a ojo. Registrado como observación; **no adoptado**.

### 20.4 Regresión con errores ARMA

**[A]** (Sección 11.3.4). Modelo (11.47): $y_t = x_t'\beta + z_t$ con $\phi(B)z_t = \theta(B)a_t$.

La construcción es apilar: $s_t^* = (s_t', \beta_t')'$, donde $s_t$ es el estado del ARMA para $z_t$ y $\beta_t = \beta$ constante.

**La idea en castellano:** si hay dos cosas que queremos rastrear (los coeficientes de regresión y la dinámica del error), **las metemos ambas en el vector de estado**. El state-space no obliga a elegir; permite apilar componentes conceptualmente distintas en un mismo vector.

**[B]** Ésta es probablemente la propiedad más útil del lenguaje state-space desde el punto de vista de la ingeniería: es **componible**. Se pueden apilar bloques (una tendencia + una estacionalidad + un ciclo + unos coeficientes) y el mismo algoritmo los maneja a todos.

### 20.5 Lo que NO hay que sacar de esta sección

- ❌ Que hay que memorizar las matrices de Akaike, Harvey o Aoki.
- ❌ Que alguna representación es "la correcta".
- ❌ Que representar un ARMA como state-space mejora sus pronósticos. **[A]** Tsay dice lo contrario: para estimación y forecasting se puede usar cualquiera. Son equivalentes.

### 20.6 Lo que SÍ hay que llevarse

$$\boxed{\text{los estados internos dependen de la representación elegida, no del mercado}}$$

---

## 21. Unobserved Component Models

### 21.1 La idea

**[A]** El *unobserved component model*, o *structural time series model* (STSM), en su forma escalar (ecuación 11.51):

$$y_t = \mu_t + \gamma_t + \omega_t + e_t$$

donde:

| Componente | Nombre | Idea |
|---|---|---|
| $\mu_t$ | **trend** | El nivel de fondo, que se mueve lentamente |
| $\gamma_t$ | **seasonal** | Un patrón que se repite con período $s$ |
| $\omega_t$ | **cycle** | Una oscilación de período más largo, no necesariamente exacto |
| $e_t$ | **irregular** | Lo que sobra; ruido sin estructura |

**En castellano:** *"lo que observo es la suma de varias cosas que no puedo ver por separado."*

**Ejemplo cotidiano:** la factura de electricidad de una casa. Sube en verano y en invierno (estacionalidad), sube lentamente año a año porque compramos más aparatos (tendencia), y tiene variación mes a mes por razones idiosincráticas (irregular). Nunca vemos las tres por separado; vemos la suma.

### 21.2 Las especificaciones de cada componente

**[A]** **Tendencia** (11.52), con doble raíz unitaria posible:

$$\mu_{t+1} = \mu_t + \beta_t + \eta_t, \qquad \eta_t \sim N(0,\sigma_\eta^2)$$
$$\beta_t = \beta_{t-1} + \varsigma_t, \qquad \varsigma_t \sim N(0,\sigma_\varsigma^2)$$

con $\mu_1 \sim N(0,\xi)$, $\beta_1 \sim N(0,\xi)$, $\xi$ un número grande (por ejemplo $10^8$).

Aquí $\beta_t$ es una **pendiente** que también evoluciona. **[A]** Los casos anidados:

- Si $\sigma_\varsigma = 0$: $\mu_t$ es un paseo aleatorio **con deriva** $\beta_1$.
- Si $\sigma_\varsigma = \sigma_\eta = 0$: $\mu_t$ es una **tendencia lineal determinista**.
- Si además $\beta_t \equiv 0$: recuperamos el local trend model de la Sección 11.1.

**[A]** **Estacionalidad** (11.53):

$$(1 + B + \cdots + B^{s-1})\gamma_t = \omega_t, \qquad \omega_t \sim N(0,\sigma_\omega^2)$$

donde $s$ es el número de estaciones. Léase: **la suma de $s$ estaciones consecutivas es aproximadamente cero, más ruido.** Es la formalización de "la estacionalidad redistribuye, no crea". Si $\sigma_\omega = 0$, el patrón estacional es **determinista** (exactamente el mismo todos los años).

**[A]** **Ciclo** (11.54): una rotación amortiguada,

$$\begin{pmatrix}\omega_{t+1}\\ \omega^*_{t+1}\end{pmatrix} = \delta\begin{pmatrix}\cos\lambda_c & \sin\lambda_c\\ -\sin\lambda_c & \cos\lambda_c\end{pmatrix}\begin{pmatrix}\omega_t\\ \omega^*_t\end{pmatrix} + \begin{pmatrix}\varepsilon_t\\ \varepsilon^*_t\end{pmatrix}$$

con $\delta \in (0,1]$ el **factor de amortiguamiento** y $\lambda_c = 2\pi/q$ la frecuencia, $q$ el período. Si $\delta = 1$, el ciclo es una onda seno–coseno determinista.

**En castellano:** el ciclo se representa como un punto girando en un círculo, que se va achicando hacia el centro (por $\delta<1$) y recibe empujones aleatorios. Eso produce oscilaciones de período aproximadamente $q$ pero de amplitud y fase que derivan.

### 21.3 El guardrail — inmediatamente

$$\boxed{\text{descomposición estadística} \neq \text{descomposición física única del mercado}}$$

**[A]** Tsay lo dice sin ambigüedad al final del Ejemplo 11.3 (ver Sección 28): *"there are infinitely many ways to decompose an observed time series into unobserved components."*

Que un modelo produzca una "tendencia" y una "estacionalidad" **no significa** que la serie estuviera compuesta de esas dos cosas. Significa que **impusimos** esa descomposición y el algoritmo encontró los valores que mejor la ajustan.

### 21.4 Trend latente — advertencia especial [B]

**Repito esta advertencia explícitamente, tal como corresponde:**

Es tentador pensar:

$$\text{Price}_t = \text{TrueTrend}_t + \text{Noise}_t$$

**NO debemos asumir que el mercado realmente tiene esa estructura.**

Un local trend model **impone** una forma concreta de evolución al estado: paseo aleatorio, incrementos gaussianos, varianza constante, independiente del ruido de observación. Todas ésas son **decisiones nuestras**, no descubrimientos.

La tendencia filtrada es:

> **el estado que el modelo especificado considera compatible con los datos observados, dados sus parámetros.**

**No** es:

> ~~"la tendencia real que el mercado estaba escondiendo".~~

Y lo repito una vez más porque es el error más fácil de cometer en todo este capítulo: **si cambiamos el modelo, cambia la tendencia estimada. Si hubiera una tendencia verdadera, no dependería de qué modelo elegimos para buscarla.**

### 21.5 Relevancia potencial para futuros [B]

**PREGUNTA ABIERTA.** ¿Tiene sentido postular una descomposición de tipo componentes no observados para datos intradiarios de MNQ?

Hay un argumento a favor y varios en contra que registrar:

**A favor:** existe una estacionalidad intradiaria bien documentada en la literatura de microestructura (patrones de volatilidad y volumen por hora del día: apertura, mediodía, cierre). Un modelo de componentes podría, en principio, separar ese patrón determinista del resto.

**En contra / cautelas:**
- La estacionalidad intradiaria es más plausible en **volatilidad y volumen** que en **retornos**. Aplicarla a precios o retornos requiere un argumento propio.
- El período $s$ tendría que definirse (¿barras por día? ¿por sesión regular?), y esa decisión no es neutral.
- Hay días festivos, medias sesiones, cambios de horario de verano y rolls de contrato. La "estacionalidad" no es limpia.
- Y sobre todo: separar una estacionalidad **no es lo mismo que predecir**. Un patrón perfectamente conocido por todos los participantes no genera oportunidad por el solo hecho de existir.

**NO se adopta nada.** Registrado como H11.11 en el backlog.

---

## 22. Kalman Filter general

**[A]** Tsay advierte al abrir la Sección 11.4: *"For readers interested in applications, this section can be skipped at the first read."* Y en la introducción del capítulo: *"The derivation of Kalman filter and smoothing algorithms necessarily involves heavy notation."*

**No omitimos la idea.** Omitimos las manipulaciones matriciales que no añaden comprensión.

### 22.1 Qué entra y qué sale

**Entra:**
- El modelo especificado (las matrices $T_t, Z_t, R_t, Q_t, H_t, d_t, c_t$).
- Los valores iniciales $s_{1|0}$ y $\Sigma_{1|0}$.
- La secuencia de observaciones $y_1, y_2, \ldots$

**Sale, en cada $t$:**
- $s_{t|t-1}$, $\Sigma_{t|t-1}$ — estado predicho e incertidumbre antes de observar.
- $v_t$, $V_t$ — sorpresa y su covarianza.
- $K_t$ — la ganancia.
- $s_{t|t}$, $\Sigma_{t|t}$ — estado filtrado e incertidumbre después de observar.
- La contribución a la log-verosimilitud.

### 22.2 La tabla paso a paso

**[A]** El algoritmo (11.64) de Tsay:

| Paso | Fórmula | En castellano |
|---|---|---|
| **1. Innovation** | $v_t = y_t - c_t - Z_t s_{t\mid t-1}$ | "Lo que observé menos lo que esperaba observar" |
| **2. Covarianza de la innovation** | $V_t = Z_t\Sigma_{t\mid t-1}Z_t' + H_t$ | "La sorpresa tiene dos fuentes: mi ignorancia sobre el estado (proyectada al espacio de observaciones) y el ruido del instrumento" |
| **3. Kalman gain** | $K_t = T_t\Sigma_{t\mid t-1}Z_t'V_t^{-1}$ | "Cuánto peso le doy a la sorpresa al proyectar el estado hacia adelante" |
| **4. Matriz $L$** | $L_t = T_t - K_tZ_t$ | "La dinámica efectiva del error de estimación después de corregir" |
| **5. Estado predicho** | $s_{t+1\mid t} = d_t + T_ts_{t\mid t-1} + K_tv_t$ | "Proyecto y corrijo, en un solo paso" |
| **6. Incertidumbre predicha** | $\Sigma_{t+1\mid t} = T_t\Sigma_{t\mid t-1}L_t' + R_tQ_tR_t'$ | "La incertidumbre baja por haber observado y sube por el ruido del proceso" |

Y si además queremos las cantidades **filtradas contemporáneas** (que son las que más nos interesarían para uso causal), **[A]** Tsay da la variante:

| Paso | Fórmula | En castellano |
|---|---|---|
| Innovation | $v_t = y_t - c_t - Z_ts_{t\mid t-1}$ | idem |
| Covarianza cruzada | $C_t = \Sigma_{t\mid t-1}Z_t'$ | "Cómo se relacionan el estado y la sorpresa" |
| Covarianza de la innovation | $V_t = Z_tC_t + H_t$ | idem |
| **Estado filtrado** | $s_{t\mid t} = s_{t\mid t-1} + C_tV_t^{-1}v_t$ | **"Mi mejor estimación del estado AHORA"** |
| **Incertidumbre filtrada** | $\Sigma_{t\mid t} = \Sigma_{t\mid t-1} - C_tV_t^{-1}C_t'$ | "Cuánto me queda de ignorancia después de observar" |
| Proyección | $s_{t+1\mid t} = d_t + T_ts_{t\mid t}$ | "Muevo el estado al futuro con la dinámica" |
| Proyección de incertidumbre | $\Sigma_{t+1\mid t} = T_t\Sigma_{t\mid t}T_t' + R_tQ_tR_t'$ | "La incertidumbre crece por el ruido del proceso" |

**Nota sobre notación que puede confundir:** el $K_t$ de la ecuación (11.64) incluye $T_t$ — es una ganancia "combinada" que hace predicción y corrección en un paso. La ganancia de la actualización contemporánea es $C_tV_t^{-1} = \Sigma_{t|t-1}Z_t'V_t^{-1}$. En el local trend model, donde $T_t = Z_t = 1$, ambas coinciden y valen $\Sigma_{t|t-1}/V_t$.

### 22.3 Qué se predice, qué se actualiza, cómo evoluciona la incertidumbre

Resumiendo la lógica en tres frases:

1. **Se predice** el estado usando la ecuación de estado. La incertidumbre **crece** ($+R_tQ_tR_t'$).
2. **Se compara** la observación con lo que el estado predicho implicaba. La diferencia es $v_t$.
3. **Se actualiza** el estado moviendo una fracción de $v_t$. La incertidumbre **baja**.

Y se repite.

### 22.4 El estado estacionario

**[A]** *"If the state-space model is time invariant, that is, all system matrices are time invariant, then the matrices $\Sigma_{t|t-1}$ converge to a constant matrix $\Sigma_*$"*, solución de la ecuación matricial

$$\Sigma_* = T\Sigma_*T' - T\Sigma_*Z'V^{-1}Z\Sigma_*T' + RQR', \qquad V = Z\Sigma_*Z' + H$$

**[A]** *"Once the steady state is reached, $V_t$, $K_t$, and $\Sigma_{t+1|t}$ are all constant. This can lead to considerable saving in computing time."*

**[B] Ésta es una observación con consecuencias conceptuales importantes, no sólo computacionales.** En estado estacionario, el Kalman Filter de un modelo invariante es **un filtro lineal con coeficientes constantes**. La recursión $s_{t+1|t} = Ts_{t|t-1} + Kv_t$ con $K$ constante es una fórmula fija.

Es decir: después del transitorio, **el "adaptativo" Kalman Filter no se adapta a nada**. Aplica siempre el mismo peso a la sorpresa. Su reputación de método sofisticado y adaptativo viene de aplicaciones donde las matrices sí varían (navegación, control), no de este caso.

### 22.5 Qué NO hay que memorizar

Deliberadamente **no** reproducimos aquí: las derivaciones del Teorema 11.1, las manipulaciones de covarianzas de las ecuaciones (11.55)–(11.63), ni las verificaciones algebraicas de independencia. No añaden comprensión conceptual y están en el libro para quien las necesite.

---

## 23. Filtering vs. Smoothing y riesgo de leakage

**Ésta es la sección más importante del informe para el puente hacia ML.**

### 23.1 El algoritmo de smoothing, conceptualmente

**[A]** El *fixed interval smoother* (de Jong, 1989) tiene dos etapas:

**Etapa 1 — pasada hacia adelante.** Se corre el Kalman Filter de $t=1$ a $t=T$, **guardando** $v_t$, $V_t$, $K_t$, $s_{t|t-1}$ y $\Sigma_{t|t-1}$.

**Etapa 2 — pasada hacia atrás.** Se recorre de $t=T$ a $t=1$ acumulando información de las innovaciones futuras:

$$q_{t-1} = Z_t'V_t^{-1}v_t + L_t'q_t$$
$$s_{t|T} = s_{t|t-1} + \Sigma_{t|t-1}q_{t-1}$$
$$M_{t-1} = Z_t'V_t^{-1}Z_t + L_t'M_tL_t$$
$$\Sigma_{t|T} = \Sigma_{t|t-1} - \Sigma_{t|t-1}M_{t-1}\Sigma_{t|t-1}$$

con $q_T = 0$ y $M_T = 0$.

**Miremos la segunda pasada con atención.** Arranca en $T$ (el final de la muestra) y va hacia atrás. El objeto $q_{t-1}$ **acumula las innovaciones de $t$ hasta $T$**. Por construcción, para calcular $s_{t|T}$ hay que haber visto todos los datos hasta $T$.

**No es un detalle de implementación. Es la definición del algoritmo.** No existe una versión "causal" del smoother; si existiera, sería el filtro.

### 23.2 Qué problema resuelve el smoothing y por qué reduce la incertidumbre

**Problema que resuelve:** cuando se hace análisis histórico, ¿por qué usar sólo la mitad de los datos para hablar de cada momento?

**Por qué reduce la incertidumbre:** porque usa más información. $\Sigma_{t|T} \leq \Sigma_{t|t}$ siempre, porque condicionar sobre un conjunto de información mayor nunca aumenta la varianza condicional.

**[A]** Verificado empíricamente en el Ejemplo 11.1: *"The confidence intervals for the smoothed state variables are also narrower than those of the filtered state variables."*

**Para qué sirve legítimamente:**
- Reconstruir históricamente qué pasó.
- Diagnosticar el modelo (residuos suavizados).
- Estimar componentes no observadas para interpretación (como en J&J).
- Interpolar valores faltantes en el interior de la muestra.

### 23.3 Por qué NO es causal

Recordemos las tres cantidades para un mismo instante $t$:

| Objeto | Información usada | ¿Existía en $t$? |
|---|---|---|
| $s_{t\mid t-1}$ | $y_1,\ldots,y_{t-1}$ | **Sí** — y además antes de $y_t$ |
| $s_{t\mid t}$ | $y_1,\ldots,y_t$ | **Sí** — una vez que $y_t$ está disponible |
| $s_{t\mid T}$ | $y_1,\ldots,y_T$ con $T>t$ | **NO** — usa datos que aún no ocurrieron |

### 23.4 El ejemplo temporal con horarios, otra vez

```
10:00 ──── 10:01 ──── 10:02 ──── 10:03
             ▲
             │ queremos el estado de este momento
```

- **Filtro** ($s_{10:01|10:01}$): usa 10:00 y 10:01. **Disponible a las 10:01.**
- **Smoother** ($s_{10:01|10:03}$): usa 10:00, 10:01, 10:02 y 10:03. **NO disponible a las 10:01.**

**[B]** Y entonces:

$$\boxed{\text{smoothed state usado como feature histórica} \Rightarrow \text{riesgo directo de look-ahead leakage}}$$

### 23.5 Por qué esto es especialmente traicionero

Hay cuatro razones por las que este tipo de leakage es más peligroso que otros:

**1. No se ve en el código.** Una línea que dice `state = smoother.smooth(y)` no tiene ninguna marca visible de que mira el futuro. Comparado con un bug obvio como `df['target'].shift(-1)`, esto pasa desapercibido.

**2. Los resultados son buenos, no absurdos.** Si un backtest da 400% anual, sospechamos. Un estado suavizado no produce números absurdos: produce números **razonablemente buenos**, lo suficiente para convencer.

**3. La ventaja es genuina y por tanto convincente.** El smoother realmente es mejor estimador del estado. La mejora en el backtest es real — sólo que es real **usando información que no existía**.

**4. La nomenclatura invita al error.** "Estado filtrado" y "estado suavizado" suenan a variantes del mismo objeto, con "suavizado" sonando incluso a "más limpio, mejor". Nada en el nombre advierte que uno mira al futuro.

### 23.6 Smoothing no es "mejor" en todos los contextos

$$\boxed{\text{mejor reconstrucción histórica} \neq \text{feature válida para forecasting causal}}$$

| Contexto | ¿Smoothing es apropiado? |
|---|---|
| Describir qué pasó históricamente | **Sí**, es lo correcto |
| Diagnosticar el modelo | **Sí** |
| Interpolar faltantes en el interior de la muestra | **Sí** |
| Estimar componentes para interpretación económica | **Sí** |
| Feature para una decisión en tiempo real | **NO** |
| Feature en un backtest que simula tiempo real | **NO** |
| Construir un target/label | **NO**, salvo justificación explícita y muy cuidadosa (ver Sección 32.5) |

### 23.7 State estimation error vs. forecast error

**[A]** (Sección 11.4.2). Dos errores distintos que no hay que confundir:

**Error de estimación del estado:**
$$x_t = s_t - s_{t|t-1}$$

**Este error NO es observable.** Depende de $s_t$, que es el estado verdadero, que nunca vemos. Sabemos su varianza bajo el modelo ($\Sigma_{t|t-1}$), pero nunca su valor.

**Error de predicción de la observación:**
$$v_t = y_t - y_{t|t-1}$$

**Este error SÍ es observable**, en cuanto llega $y_t$.

**[A]** La relación entre ambos (ecuación 11.65):
$$v_t = Z_tx_t + e_t, \qquad x_{t+1} = L_tx_t + R_t\eta_t - K_te_t$$

Es decir: la sorpresa observable es el error de estado (proyectado) **más** el ruido de observación. **No podemos separarlos.** Aunque veamos $v_t$, no sabemos qué parte era error de estado y qué parte era ruido de medición.

**Por qué importa para el diagnóstico [B]:**

Sólo podemos diagnosticar el modelo con lo que observamos, es decir con $\{v_t\}$. Podemos verificar que las innovaciones parecen ruido blanco. **No podemos verificar directamente que el estado estimado esté cerca del estado verdadero**, porque el estado verdadero no existe fuera del modelo.

Esto refuerza el guardrail central:

$$\boxed{\text{estado estimado} \neq \text{estado verdadero descubierto}}$$

No hay ningún test empírico en este capítulo que valide la interpretación del estado. Los tests validan que las **predicciones de las observaciones** sean coherentes. Es una cosa distinta.

---

## 24. Disturbance Smoothing

**PRIORIDAD CONCEPTUAL.** Sólo la idea.

### 24.1 La pregunta

En vez de reconstruir sólo el estado, ¿podemos reconstruir también **los shocks** que el modelo estima que ocurrieron?

### 24.2 Los dos objetos

**[A]** Tsay define:

$$e_{t|T} = E(e_t \mid F_T) \qquad \text{(smoothed observation disturbance)}$$
$$\eta_{t|T} = E(\eta_t \mid F_T) \qquad \text{(smoothed state disturbance)}$$

En castellano:

- $e_{t|T}$ = "mirando toda la muestra, ¿cuánto creo que el instrumento se desvió en el momento $t$?"
- $\eta_{t|T}$ = "mirando toda la muestra, ¿cuánto creo que el estado se movió genuinamente entre $t$ y $t+1$?"

**[A]** Las fórmulas resultantes (11.82):

$$e_{t|T} = H_t(V_t^{-1}v_t - K_t'q_t), \qquad \eta_{t|T} = Q_tR_t'q_t$$

con las mismas recursiones hacia atrás de $q_t$ y $M_t$ del smoothing de estados.

**[A]** Tsay llama $o_t = V_t^{-1}v_t - K_t'q_t$ la *"smoothing measurement error"*.

### 24.3 Para qué sirve

**[A]** *"These smoothed disturbances are useful in many applications, for example, in **model checking**."*

Ése es el uso principal que menciona Tsay: **diagnóstico**. Si el modelo es correcto, los shocks reconstruidos deberían verse compatibles con las distribuciones asumidas. Anomalías grandes señalan períodos donde el modelo no encaja (outliers, cambios estructurales).

**[A]** Segundo uso: Koopman (1993) usa $\eta_{t|T}$ para derivar una recursión alternativa (más eficiente) para calcular $s_{t|T}$:

$$s_{t+1|T} = d_t + T_ts_{t|T} + R_tQ_tR_t'q_t \tag{11.81}$$

**[A]** Tercer uso, en la práctica: en el Ejemplo 11.3 Tsay grafica los *smoothed response residuals* junto a los errores de predicción 1-paso, como diagnóstico visual del ajuste (Figura 11.7).

### 24.4 El guardrail obvio [B]

Todo lo que dijimos sobre smoothing se aplica **idénticamente** aquí. $e_{t|T}$ y $\eta_{t|T}$ **usan toda la muestra**. Son objetos retrospectivos. No son features causales.

De hecho, son *más* peligrosos conceptualmente, porque suenan a "los shocks que realmente ocurrieron" — cuando son, otra vez, "los shocks que el modelo, mirando todo, estima que ocurrieron".

**Deliberadamente omitimos** las derivaciones de $\mathrm{Var}(e_t|F_T)$ y $\mathrm{Var}(\eta_t|F_T)$; son álgebra matricial que no añade comprensión conceptual.

---

## 25. Missing Values

### 25.1 La intuición

**[A]** Tsay presenta el manejo de faltantes como una de las dos ventajas prácticas principales del enfoque: *"An advantage of the state-space model is in handling missing values."* Y cierra la Sección 11.5 con: *"the ease in handling missing values is a nice feature of the state-space model."*

**La lógica es transparente.** El filtro tiene dos partes:

1. **Predecir** el estado usando la ecuación de estado (no necesita datos).
2. **Actualizar** usando la observación (necesita datos).

Si en $t$ no hay observación, simplemente **no hacemos el paso 2**. El estado sigue propagándose con su dinámica, y la incertidumbre crece porque no hubo corrección.

### 25.2 La formalización

**[A]** Caso 1: $y_t$ enteramente faltante para $t = \ell+1,\ldots,\ell+h$.

Se hace $v_t = 0$ y $K_t = 0$, y el filtro continúa:

$$s_{t+1|t} = d_t + T_ts_{t|t-1}, \qquad \Sigma_{t+1|t} = T_t\Sigma_{t|t-1}T_t' + R_tQ_tR_t'$$

**[A]** Comentario de Tsay para el local trend model: *"This is rather natural because when $y_t$ is missing, there is no new innovation or new Kalman gain so that $v_t = 0$ and $K_t = 0$."*

Para el local trend, esto da (11.24): $\mu_{t|t-1} = \mu_{t-1|t-2}$ y $\Sigma_{t|t-1} = \Sigma_{t-1|t-2} + \sigma_\eta^2$.

**Léase:** el estado estimado **se congela** (porque es un paseo aleatorio y no hay información nueva), pero la incertidumbre **crece linealmente** con la cantidad de períodos faltantes. Cuanto más tiempo sin datos, menos sabemos.

**[A]** Para el smoothing durante el hueco: $q_{t-1} = T_t'q_t$ y $M_{t-1} = T_t'M_tT_t$.

**[A]** Caso 2: **algunas componentes** de $y_t$ faltantes (sólo relevante en el caso multivariado). Se define $y_t^* = Jy_t$ con $J$ una matriz indicadora cuyas filas son un subconjunto de las de la identidad, y se usa la ecuación de observación modificada con $c_t^* = Jc_t$, $Z_t^* = JZ_t$, $H_t^* = JH_tJ'$. El resto del algoritmo no cambia.

### 25.3 Por qué esto es elegante

Comparemos con qué hace uno normalmente ante un dato faltante:

| Enfoque | Qué hace | Problema |
|---|---|---|
| Imputar con el valor anterior | Rellena con $y_{t-1}$ | Inventa un dato. El modelo lo trata como observación real y **reduce su incertidumbre** indebidamente |
| Imputar con interpolación | Rellena con un promedio de vecinos | **Usa el futuro.** Leakage directo |
| Eliminar la fila | Descarta | Rompe la estructura temporal; el modelo cree que $t+1$ vino inmediatamente después de $t-1$ |
| **State-space** | No actualiza | **Reconoce explícitamente que no sabe**, y su incertidumbre lo refleja |

El punto clave: el state-space **no inventa un valor**. Propaga incertidumbre honestamente.

### 25.4 Distinciones que hay que mantener [B]

Ésta es una advertencia propia, no de Tsay, y es importante para datos de futuros:

| Situación | Qué es | ¿Es lo mismo? |
|---|---|---|
| **Barra realmente ausente** | No hubo dato. Feed caído, mercado cerrado, gap | El caso que Tsay trata |
| **Barra existente con volumen cero** | Hubo un intervalo de tiempo, no hubo transacciones. **El dato existe**: el precio no se movió porque nadie operó | **NO es lo mismo.** Es información real: "nadie quiso operar" |
| **Valor imputado** | Alguien rellenó el hueco | **NO es lo mismo.** Es una construcción nuestra, no un dato |

**Estas tres cosas se ven idénticas en un DataFrame** si no se lleva registro. Y significan cosas completamente diferentes.

**[B]** Una barra de volumen cero en MNQ es un evento informativo (baja liquidez, horario nocturno, festivo). Tratarla como "faltante" descarta esa información. Tratarla como observación normal puede ser correcto o no, según qué se esté modelando.

**NO diseñamos ninguna política de imputación.** Sólo registramos que:
1. El state-space ofrece un mecanismo coherente para faltantes genuinos.
2. Ese mecanismo requiere primero decidir **qué cuenta como faltante**, y esa decisión es previa y no trivial.
3. Los datos de futuros tienen gaps de varios tipos (overnight, fin de semana, festivos, rolls de contrato, halts) que no son todos el mismo fenómeno.

**PREGUNTA ABIERTA.** ¿El gap overnight de MNQ debe tratarse como observaciones faltantes (propagando incertidumbre a través de la noche), como un salto genuino del estado, o como una discontinuidad que requiere reinicializar? Las tres son defendibles y producen resultados distintos. El capítulo no lo resuelve.

---

## 26. Forecasting

**PRIORIDAD MÁXIMA.**

### 26.1 La definición

**[A]** Sea $t$ el origen del pronóstico. El pronóstico a $j$ pasos, bajo el criterio de mínimo error cuadrático medio, es:

$$y_t(j) = E(y_{t+j} \mid F_t)$$

Igual que en los modelos ARMA del Capítulo 2. **[A]** *"Similar to the ARMA models, the $j$-step-ahead forecast $y_t(j)$ turns out to be the expected value of $y_{t+j}$ given $F_t$ and the model."*

### 26.2 La idea elegante: el futuro como valores faltantes

**[A]** Esto es lo más bonito de la sección, y Tsay lo enuncia así: *"we show that these forecasts and the covariance matrices of the associated forecast errors can be obtained via the Kalman filter in Eq. (11.64) by **treating $\{y_{t+1},\ldots,y_{t+h}\}$ as missing values**."*

**Vale la pena detenerse en por qué esto funciona, porque es conceptualmente hermoso.**

¿Qué tienen en común un dato futuro y un dato faltante? **Que no lo tenemos.** Desde el punto de vista del filtro, no hay ninguna diferencia entre:

- "el dato de las 10:05 se perdió porque el feed se cayó", y
- "el dato de las 10:05 todavía no ocurrió".

En ambos casos, el filtro no puede actualizar. Sólo puede propagar.

Y como ya vimos (Sección 25) que el filtro maneja faltantes poniendo $v_t = 0$ y $K_t = 0$, **pronosticar es simplemente correr el filtro hacia adelante con las ganancias apagadas.**

$$\boxed{\text{forecasting} = \text{filtrar hacia adelante sin nada que observar}}$$

No hace falta un algoritmo separado. Es el mismo algoritmo, con los datos ausentes.

### 26.3 Las fórmulas

**[A]** Pronóstico a 1 paso:

$$y_t(1) = c_{t+1} + Z_{t+1}s_{t+1|t}$$
$$e_t(1) = y_{t+1} - y_t(1) = Z_{t+1}(s_{t+1}-s_{t+1|t}) + e_{t+1}$$
$$\mathrm{Var}[e_t(1)] = Z_{t+1}\Sigma_{t+1|t}Z_{t+1}' + H_{t+1}$$

**[A]** *"This is precisely the covariance matrix $V_{t+1}$ of the Kalman filter."* Es decir: **la varianza del error de pronóstico a 1 paso ya está calculada por el filtro**. No es un cálculo adicional.

**[A]** Pronóstico a $j$ pasos (11.83)–(11.85):

$$y_t(j) = c_{t+j} + Z_{t+j}s_{t+j|t}$$
$$\mathrm{Var}[e_t(j)] = Z_{t+j}\Sigma_{t+j|t}Z_{t+j}' + H_{t+j}$$
$$s_{t+j+1|t} = d_{t+j} + T_{t+j}s_{t+j|t}$$
$$\Sigma_{t+j+1|t} = T_{t+j}\Sigma_{t+j|t}T_{t+j}' + R_{t+j}Q_{t+j}R_{t+j}'$$

### 26.4 Cómo crece la incertidumbre

Miremos la última ecuación. En cada paso hacia el futuro, la covarianza del estado se transforma por la dinámica **y se le suma** $R_tQ_tR_t'$. **La incertidumbre crece monotónicamente con el horizonte.**

**Caso concreto: el local trend model.** Con $T=1$, $R=1$, $Q=\sigma_\eta^2$:

$$\Sigma_{t+j|t} = \Sigma_{t+1|t} + (j-1)\sigma_\eta^2$$

y por tanto:

$$\mathrm{Var}[e_t(j)] = \Sigma_{t+1|t} + (j-1)\sigma_\eta^2 + \sigma_e^2$$

**La varianza del error de pronóstico crece LINEALMENTE con el horizonte.** Y el pronóstico puntual, en cambio, **es constante**: $y_t(j) = \mu_{t|t}$ para todo $j$, porque el estado es un paseo aleatorio sin deriva.

**Léase en castellano:** este modelo dice *"mi mejor pronóstico para dentro de 1 minuto, 1 hora o 1 día es exactamente el mismo número — y mi incertidumbre crece sin límite."*

**[B]** Es un recordatorio saludable. El local trend model, aplicado a algo cuyo estado es un paseo aleatorio, **no predice dirección en absoluto**. Todo lo que aporta es una estimación del nivel actual y una cuantificación de cuán rápido se degrada nuestro conocimiento. Correr un Kalman Filter sobre una serie con estado de paseo aleatorio y esperar señal direccional es esperar algo que el modelo, por construcción, no puede dar.

### 26.5 Forecasting ≠ Filtering — con otro ejemplo

A las **10:00**, con toda la información disponible hasta ese momento:

| Objeto | Pregunta que responde | Qué es |
|---|---|---|
| **Estado filtrado** $s_{10:00\mid10:00}$ | "¿Cuánto vale la cantidad latente **ahora**?" | Afirmación sobre el **presente** |
| **Forecast** $y_{10:00}(1)$ | "¿Cuánto valdrá la observación a las **10:01**?" | Afirmación sobre el **futuro** |
| **Forecast** $y_{10:00}(5)$ | "¿Y a las **10:05**?" | Futuro más lejano, más incierto |

**No se debe decir:** *"el Kalman Filter predice porque filtra."*

**Se debe decir:** *"el Kalman Filter filtra, y el mismo mecanismo puede propagarse hacia adelante para producir pronósticos, con incertidumbre creciente."*

Están relacionados porque el forecast **parte** del estado filtrado. Pero son objetos distintos, con incertidumbres distintas, y responden preguntas distintas.

### 26.6 Diagnóstico

**[A]** *"the prediction error series $\{v_t\}$ can be used to evaluate the likelihood function for estimation and the standardized prediction errors $D_t^{-1/2}v_t$ can be used for model checking, where $D_t = \mathrm{diag}\{V_t(1,1),\ldots,V_t(k,k)\}$."*

Es decir: el mismo objeto $\{v_t\}$ sirve para tres cosas — estimar parámetros, diagnosticar el modelo y medir el error de pronóstico a 1 paso. Es el objeto más versátil del capítulo.

### 26.7 Lo que este capítulo NO dice sobre forecasting [B]

- **No** hay ninguna evaluación de pronósticos **fuera de muestra** en el sentido de walk-forward.
- **No** hay ninguna comparación de la capacidad predictiva del state-space contra benchmarks (naïve, EMA, ARIMA) sobre datos de test.
- **No** hay ninguna evaluación de **utilidad económica**.
- **No** hay ninguna aplicación a series de precios ni a decisiones de trading.

$$\text{forecast estadístico} \neq \text{utilidad económica}$$

**PREGUNTA ABIERTA (H11.10).** ¿Un modelo state-space mejora el forecasting OOS de algún objetivo relevante en MNQ frente a benchmarks comparables? No se responde aquí ni se intenta.

---

## 27. Application: Johnson & Johnson

**[A] Todos los datos de esta sección provienen del Ejemplo 11.3 de Tsay.**

### 27.1 Ficha del experimento

| Campo | Valor |
|---|---|
| **Serie** | Earnings per share (EPS) trimestral de Johnson & Johnson |
| **Período** | 1960 – 1980 |
| **Frecuencia** | Trimestral |
| **Observaciones** | 84 ($T = 84$) |
| **Transformación** | $y_t = \log(\text{EPS observado})$ |
| **Referencia** | Los mismos datos analizados en el Cap. 2 del libro |

**Nótese:** la serie está en **logaritmos**. Esto no es un detalle menor — una descomposición aditiva en logs equivale a una descomposición multiplicativa en niveles. La estacionalidad se interpreta como un porcentaje sobre la tendencia, no como una cantidad fija de dólares.

### 27.2 El modelo

**[A]** (ecuación 11.87):

$$y_t = \mu_t + \gamma_t + e_t, \qquad e_t\sim N(0,\sigma_e^2)$$
$$\mu_{t+1} = \mu_t + \eta_t, \qquad \eta_t\sim N(0,\sigma_\eta^2)$$
$$(1+B+B^2+B^3)\gamma_t = \omega_t, \qquad \omega_t\sim N(0,\sigma_\omega^2)$$

La última se lee $\gamma_t = -\sum_{j=1}^{3}\gamma_{t-j} + \omega_t$: **la suma de cuatro trimestres consecutivos de componente estacional es aproximadamente cero.**

**Tres componentes, tres parámetros:**
- $\mu_t$ = **tendencia** (local level, paseo aleatorio)
- $\gamma_t$ = **estacionalidad** trimestral ($s=4$), estocástica
- $e_t$ = **irregular**

*(Nota: no hay componente de ciclo en este modelo — Tsay usa la versión con trend + seasonal + irregular únicamente.)*

### 27.3 La forma state-space

**[A]** El estado es de dimensión 4: $s_t = (\mu_t, \gamma_t, \gamma_{t-1}, \gamma_{t-2})'$

$$\begin{pmatrix}\mu_{t+1}\\\gamma_{t+1}\\\gamma_t\\\gamma_{t-1}\end{pmatrix} = \begin{pmatrix}1&0&0&0\\0&-1&-1&-1\\0&1&0&0\\0&0&1&0\end{pmatrix}\begin{pmatrix}\mu_t\\\gamma_t\\\gamma_{t-1}\\\gamma_{t-2}\end{pmatrix} + \begin{pmatrix}1&0\\0&1\\0&0\\0&0\end{pmatrix}\begin{pmatrix}\eta_t\\\omega_t\end{pmatrix}$$

$$y_t = [1,1,0,0]s_t + e_t$$

con $\mathrm{Cov}(\eta_t,\omega_t) = \mathrm{diag}\{\sigma_\eta^2,\sigma_\omega^2\}$.

**Léase la matriz de transición:**
- Fila 1: la tendencia se copia a sí misma (paseo aleatorio).
- Fila 2: la estacionalidad nueva es menos la suma de las tres anteriores.
- Filas 3–4: **son una cinta transportadora**. Simplemente mueven los valores viejos un lugar hacia atrás para tenerlos disponibles.

**Esto ilustra algo importante:** no todas las componentes del vector de estado son "cantidades latentes con significado". Las filas 3 y 4 son **memoria mecánica**, necesaria para que el sistema sea markoviano de primer orden. Sin ellas, la ecuación de estacionalidad necesitaría tres rezagos, y eso no sería Markov de primer orden.

**[B]** Es un buen antídoto contra la mística del "estado latente": buena parte de un vector de estado suele ser bookkeeping.

### 27.4 Los resultados

**[A]** Estimación por máxima verosimilitud:

$$(\hat\sigma_e,\ \hat\sigma_\eta,\ \hat\sigma_\omega) = (2.04\times10^{-6},\ 7.27\times10^{-2},\ 2.93\times10^{-2})$$

Valores exactos del output: `2.044516e-06`, `7.269655e-02`, `2.931691e-02`.

**Observación sobre estos números (aritmética [B] sobre cifras [A]):** $\hat\sigma_e$ es del orden de $10^{-6}$ — **prácticamente cero**. El modelo asigna esencialmente **nada** a la componente irregular. Toda la variación se reparte entre tendencia y estacionalidad. Y de esas dos, la tendencia se mueve aproximadamente 2.5 veces más que la estacionalidad ($0.0727$ vs. $0.0293$).

**[B]** Éste es el espejo del ejemplo de Alcoa, y por eso vale la pena verlos juntos:

| Ejemplo | Serie | $\hat\sigma_e$ (obs.) | Innovaciones de estado | Interpretación |
|---|---|---|---|---|
| **11.1 Alcoa** | log realized volatility diaria | 0.4803 | $\hat\sigma_\eta = 0.0735$ | Ruido de medición **domina** |
| **11.3 J&J** | log EPS trimestral | $2.04\times10^{-6}$ | $\hat\sigma_\eta = 0.0727$, $\hat\sigma_\omega = 0.0293$ | Ruido de medición **desaparece** |

**Dos series, dos resultados opuestos.** Esto refuerza que la razón señal-ruido es una propiedad de **la serie concreta y del modelo elegido**, no un hecho general sobre datos financieros.

Tiene sentido: un EPS reportado es una cifra contable auditada, no un estimador ruidoso. Una volatilidad realizada es un estimador con error muestral. Que uno tenga "ruido de medición" y el otro no, es coherente con la naturaleza de los objetos.

### 27.5 Resultados gráficos y diagnósticos

**[A]**
- **Figura 11.6:** estimaciones **suavizadas** de las componentes de tendencia ($\mu_{t|T}$) y estacionalidad ($\gamma_{t|T}$), con $T = 84$, junto con regiones de confianza puntuales del 95%.
- **[A]** *"Of particular interest is that the seasonal pattern seems to evolve over time."* Es decir: la estacionalidad **no es constante**; deriva a lo largo de las dos décadas. Coherente con $\hat\sigma_\omega > 0$.
- Rangos de las bandas reportados en el código: tendencia, máximo 2.7957 y mínimo −0.5949; estacionalidad, máximo 0.3789 y mínimo −0.3552.
- **Figura 11.7:** (a) errores de predicción 1-paso $v_t$ del Kalman filter; (b) residuos suavizados de la variable de respuesta.

**[A]** Conclusión de Tsay: *"state-space modeling provides an alternative approach for analyzing seasonal time series."*

Nótese la palabra: **alternative**. No "mejor". Alternativa.

### 27.6 Detalle metodológico importante [B]

Las componentes graficadas son **suavizadas** ($\mu_{t|T}$, $\gamma_{t|T}$ con $T=84$). Usan toda la muestra 1960–1980 para estimar cada punto, incluidos los de 1960.

Es completamente apropiado para el propósito de Tsay (describir retrospectivamente la estructura de la serie). Pero **[B]**: si alguien tomara la Figura 11.6 y dijera "en 1965 la tendencia valía X, por lo tanto un analista de 1965 podía saberlo", estaría equivocado. El analista de 1965 no tenía los datos de 1966–1980.

### 27.7 Qué NO se puede concluir

- ❌ Que los modelos de componentes no observados funcionen bien en datos financieros de alta frecuencia. Esto es EPS **trimestral** de una empresa, 84 observaciones, 20 años.
- ❌ Que la descomposición encontrada sea "la correcta". Tsay dice explícitamente lo contrario (siguiente sección).
- ❌ Que la estacionalidad de un negocio farmacéutico tenga algo que ver con la estacionalidad intradiaria de un futuro de índice.
- ❌ Que $\hat\sigma_e \approx 0$ signifique "los datos son limpios". Significa que **este modelo**, con **estas tres componentes**, no necesitó un término irregular para ajustar.

---

## 28. No unicidad de componentes latentes

**ESTA ES LA SECCIÓN MÁS IMPORTANTE DEL CAPÍTULO desde el punto de vista epistemológico.**

### 28.1 La cita completa

**[A]** Cerrando el Ejemplo 11.3, Tsay escribe:

> *"It should be noted that the estimated components in Figure 11.6 are **not unique**. They depend on the model specified and constraints used. In fact, **there are infinitely many ways to decompose an observed time series into unobserved components**. For instance, one can use a different specification for the seasonal component, for example, `seasonalTrig` in SsfPack, to obtain another decomposition for the earnings series of Johnson & Johnson. Thus, **care must be exercised in interpreting the estimated components**. However, for forecasting purposes, the choice of decomposition does not matter provided that the chosen one is a valid decomposition."*

### 28.2 Qué significa exactamente

Traduzcamos con precisión.

Tenemos una serie observada: $y_1, y_2, \ldots, y_T$. Un solo objeto, perfectamente definido, que todos podemos ver.

Escribimos: $y_t = \mu_t + \gamma_t + e_t$.

**Hay infinitas maneras de repartir cada $y_t$ entre tres sumandos.** Podríamos poner más en la tendencia y menos en la estacionalidad, o al revés. Los datos **no determinan** el reparto. Lo que determina el reparto es:

- **el modelo** que impusimos a cada componente (¿cómo se mueve la tendencia? ¿es la estacionalidad determinista o estocástica?);
- **las restricciones** que pusimos (¿la estacionalidad suma cero? ¿en cuántos períodos?);
- **los parámetros** estimados;
- la **inicialización**.

Cambiando cualquiera de esos elementos, cambian las componentes estimadas — **con los mismos datos**.

**[A]** Tsay da un ejemplo concreto de esto: usando `seasonalTrig` (representación trigonométrica de la estacionalidad) en lugar de `seasonalDummy`, se obtiene **otra descomposición** de la misma serie de J&J.

### 28.3 El guardrail

$$\boxed{\text{una descomposición state-space útil} \neq \text{la única descomposición verdadera}}$$

**Repito, porque es fácil de olvidar en el momento de mirar un gráfico bonito:**

Cuando vemos un gráfico de "la tendencia latente" de una serie, lo que estamos viendo es:

> *La tendencia que este modelo, con estas restricciones y estos parámetros, considera compatible con estos datos.*

**No** estamos viendo:

> ~~*La tendencia que la serie tenía y que el ruido nos ocultaba.*~~

Si hubiera una tendencia verdadera, no dependería de qué modelo elegimos para buscarla. El hecho de que dependa es prueba de que es un constructo del modelo.

### 28.4 La segunda parte de la cita — y por qué es la más fácil de sobre-interpretar

Tsay añade: *"However, for forecasting purposes, the choice of decomposition does not matter provided that the chosen one is a valid decomposition."*

**Esta frase requiere lectura cuidadosa, y la marcamos como potencialmente ambigua.**

Interpretación razonable **[B]**: si dos descomposiciones distintas implican **la misma distribución conjunta** para $\{y_t\}$ (es decir, son reparametrizaciones del mismo modelo observacional), entonces producirán los mismos pronósticos. Eso es correcto y coherente con el resto del capítulo — es el mismo argumento de las múltiples representaciones ARMA.

Interpretación **incorrecta** que hay que evitar: "da igual qué modelo elijas, los pronósticos serán los mismos". Eso sería falso. Un modelo con estacionalidad determinista y otro con estacionalidad estocástica **son modelos distintos** y producen pronósticos distintos.

La cláusula "provided that the chosen one is a valid decomposition" hace todo el trabajo, y Tsay no define formalmente "válida" en este punto. **[B]** Interpretamos que se refiere a descomposiciones que son reparametrizaciones equivalentes del mismo modelo observacional, no a "cualquier modelo razonable".

**Registrado en la Sección 40 como afirmación fácil de sobre-interpretar.**

### 28.5 La consecuencia práctica más importante [B]

Esto tiene una implicación directa que conviene enunciar sin rodeos:

> **La interpretación de las componentes y su utilidad predictiva son cosas independientes.**

Un modelo puede producir componentes que interpretamos como "tendencia" y "estacionalidad" y ser útil para pronosticar. Y **otro** modelo puede producir componentes completamente distintas y ser igual de útil. La utilidad predictiva no valida la interpretación, y la interpretación no garantiza utilidad.

**[B]** Traducción a nuestro contexto: si alguna vez obtuviéramos un "estado latente de tendencia" para MNQ y resultara predictivo, eso **no** demostraría que existe una tendencia latente en MNQ. Demostraría que esa transformación particular de los datos tenía contenido predictivo. Y viceversa: que la interpretación sea atractiva no es evidencia de nada.

### 28.6 Relación con las tres representaciones ARMA

Esta advertencia es **la misma lección** que aparece en la Sección 11.3.2, vista desde otro ángulo:

| Sección | Manifestación |
|---|---|
| **11.3.2** | Un mismo ARMA admite (al menos) tres representaciones state-space, con estados completamente distintos |
| **11.7** | Una misma serie admite infinitas descomposiciones en componentes no observados |

En ambos casos: **la observación es única, el estado no lo es.** Es una propiedad estructural del enfoque, no una limitación de un ejemplo particular.

---

## 29. State-Space vs. Markov Switching

Comparación con el Capítulo 4. Objetivo: **no llamar "régimen" a cualquier variable de estado.**

### 29.1 La tabla

| | **State-space / Kalman (Cap. 11)** | **Markov Switching (Cap. 4)** |
|---|---|---|
| Naturaleza del estado | **Continuo**: $s_t \in \mathbb{R}^m$ | **Discreto**: $S_t \in \{1,2,\ldots,K\}$ |
| Evolución | Lineal: $s_{t+1} = d_t + T_ts_t + R_t\eta_t$ | Probabilística entre categorías: matriz de transición $p_{ij}$ |
| Tipo de incertidumbre | Sobre el **valor numérico** del estado (una varianza) | Sobre **cuál categoría** es la actual (probabilidades que suman 1) |
| Qué produce la inferencia | Media y covarianza condicionales | Probabilidades de estar en cada régimen |
| Algoritmo | Kalman filtering (marco lineal-gaussiano) | Hamilton filtering / probabilidades de estado |
| Pregunta que responde | "¿Cuánto vale la cantidad latente?" | "¿En qué régimen estamos?" |

*(No profundizamos en el Hamilton filter; es material del Cap. 4 y no del 11.)*

### 29.2 Por qué la distinción importa

Ambos son "modelos con estado latente". Pero el tipo de afirmación que producen es radicalmente distinto.

**Markov switching dice:** *"Hay 70% de probabilidad de que estemos en el régimen de alta volatilidad."* Es una afirmación sobre **pertenencia a una categoría**. Presupone que existen categorías discretas.

**State-space lineal dice:** *"La tendencia latente vale 0.87, con desviación estándar 0.12."* Es una afirmación sobre **un valor numérico**. No presupone categorías.

**[B]** Consecuencia terminológica que hay que respetar: **un estado continuo que deriva suavemente no describe regímenes.** El $\beta_t$ del CAPM dinámico de GM, que se mueve entre 1.02 y 1.06 a lo largo de 14 años, **no** identifica regímenes de mercado. Describe una deriva lenta. Llamarlo "régimen" es un error categorial.

### 29.3 Qué NO son

- ❌ **No** son el mismo modelo con distinta notación. Son familias distintas, con supuestos distintos.
- ❌ **No** es cierto que uno "generalice" al otro dentro del marco lineal-gaussiano de este capítulo.
- ❌ **No** es cierto que se puedan combinar sin costo. Existen modelos híbridos (state-space con switching), pero requieren métodos de inferencia distintos porque el filtro exacto se vuelve intratable — **[A]** Tsay menciona a Kim y Nelson (1999) como referencia que *"focus on economic applications and regime switching"*, pero **no desarrolla estos métodos en este capítulo**.

### 29.4 Lo que sí comparten

Ambos responden a la misma intuición de fondo: **hay algo que gobierna el comportamiento observado y que no vemos directamente.** La diferencia está en cómo se modela ese algo — como un número que se mueve, o como una etiqueta que salta.

**[B] PREGUNTA ABIERTA.** Para un futuro de índice, ¿es más natural pensar en un continuo (la volatilidad esperada es un número que deriva) o en categorías (el mercado está "en tendencia" o "en rango")? La intuición de los operadores tiende a las categorías; la evidencia estadística no obliga a ninguna de las dos. Ninguno de los capítulos estudiados lo resuelve.

---

## 30. State-Space vs. GARCH

Conexión con el Capítulo 3.

### 30.1 La diferencia estructural

| | **GARCH (Cap. 3)** | **State-space estocástico (Cap. 11)** |
|---|---|---|
| ¿La cantidad latente tiene ruido propio? | **NO** | **SÍ** |
| Cómo evoluciona | $\sigma_t^2 = \alpha_0 + \alpha_1a_{t-1}^2 + \beta_1\sigma_{t-1}^2$ — **función determinista del pasado observado**, dados los parámetros | $s_{t+1} = T_ts_t + R_t\eta_t$ — **tiene su propia innovación aleatoria** $\eta_t$ |
| ¿Se puede calcular exactamente? | **Sí.** Dados los parámetros y los datos, $\sigma_t^2$ queda determinada | **No.** Sólo se puede estimar su distribución condicional |
| ¿Cuántos ruidos? | Uno ($a_t$) | Dos ($e_t$ y $\eta_t$) |

**La diferencia en castellano:**

- En un GARCH, la volatilidad condicional es "no observada" sólo en el sentido de que **no está en la columna de datos**. Pero es **calculable**: si conocemos los parámetros y los retornos pasados, sale un número exacto. No hay incertidumbre residual sobre su valor.
- En un modelo de volatilidad estocástica en forma state-space, la volatilidad tiene **su propio ruido**, independiente del ruido de los retornos. Nunca se puede calcular exactamente, sólo estimar con incertidumbre.

**[B]** Ésa es la distinción conceptual clave entre "determinista dado el pasado" y "genuinamente latente".

### 30.2 Precisión importante

**El local trend model del Capítulo 11 NO es un modelo de volatilidad estocástica.**

Es un modelo de nivel local. En el Ejemplo 11.1 se **aplica** al log de una medida de volatilidad realizada, pero eso no lo convierte en un modelo de volatilidad estocástica en sentido técnico. Es un local level model aplicado a una serie que resulta ser una medida de volatilidad.

La diferencia: un modelo de volatilidad estocástica pone la volatilidad latente **dentro** de la ecuación de los retornos. Aquí, la volatilidad realizada es la **observación**, no un parámetro de escala de otra ecuación.

### 30.3 Por qué el state-space es la puerta hacia el Capítulo 12

**[A]** Al enumerar los modelos representables en forma state-space, Tsay incluye explícitamente los **stochastic volatility models**.

**[A]** Y el Capítulo 12 del libro (*Markov Chain Monte Carlo Methods with Applications*) usa maquinaria de state-space/Kalman para tratar volatilidad estocástica mediante MCMC.

**Por qué hace falta MCMC:** un modelo de volatilidad estocástica típico es de la forma $r_t = \exp(h_t/2)\epsilon_t$ con $h_{t+1} = \phi h_t + \eta_t$. La primera ecuación **no es lineal** en el estado $h_t$. El Kalman Filter clásico no aplica directamente. De ahí la necesidad de métodos de simulación.

**[B]** Registramos esto sólo como mapa: el Capítulo 11 nos da el **lenguaje** (estados latentes, ecuaciones de observación y transición) que el Capítulo 12 usará con herramientas más pesadas. No estudiamos el Capítulo 12 aquí.

### 30.4 Un puente conceptual entre C3 y C11

Vale la pena registrar el paralelo, porque es esclarecedor:

| | GARCH | Kalman (local trend) |
|---|---|---|
| Actualización | $\sigma_t^2$ nuevo = combinación de $\sigma_{t-1}^2$ y $a_{t-1}^2$ | $\mu_{t\|t}$ = combinación de $\mu_{t\|t-1}$ y $y_t$ |
| Estructura | "Creencia previa + sorpresa reciente, con pesos" | "Creencia previa + sorpresa reciente, con pesos" |
| ¿Los pesos dependen de los datos? | Fijos (dados los parámetros) | Fijos (dados los parámetros y la inicialización) |

**Ambos son recursiones de "creencia previa + peso × información nueva".** Esa estructura es genérica y aparece por todos lados en estadística de series temporales — incluyendo, como vimos en la Sección 20.2, en la fórmula de actualización de pronósticos de un ARMA. El Kalman Filter es una instancia particularmente clara y general de esa idea, **no** una idea nueva.

---

## 31. Kalman vs. EMA / smoothers [B]

Comparación conceptual. **Sin decir que uno es mejor.**

### 31.1 Las tres herramientas

**Moving Average (media móvil simple)**

> Combina las últimas $n$ observaciones con pesos iguales, mediante una ventana.

$$\text{MA}_t = \frac{1}{n}\sum_{i=0}^{n-1}y_{t-i}$$

Parámetro libre: el largo de la ventana $n$. Todo lo anterior a $n$ períodos pesa exactamente cero; todo lo dentro de la ventana pesa exactamente igual.

**EMA (media móvil exponencial)**

> Pesa las observaciones recientes de forma decreciente, sin corte abrupto.

$$\text{EMA}_t = \alpha y_t + (1-\alpha)\text{EMA}_{t-1}$$

Parámetro libre: $\alpha \in (0,1)$. Los pesos decaen geométricamente hacia el pasado; nada pesa exactamente cero.

**Kalman Filter (local level)**

> Actualiza un estado según un modelo dinámico explícito, la incertidumbre del estado y la incertidumbre de la observación.

$$\mu_{t|t} = \mu_{t|t-1} + K_t(y_t - \mu_{t|t-1})$$

Parámetros libres: $\sigma_e^2$, $\sigma_\eta^2$ (y la inicialización).

### 31.2 La equivalencia que Tsay establece [A]

**[A]** Tsay dice literalmente, al derivar la relación con ARIMA:

> *"the state-space model in Eqs. (11.1) and (11.2) is also an ARIMA(0,1,1) model, **which is the simple exponential smoothing model of Chapter 2**."*

**Esto es una equivalencia establecida por Tsay, no una analogía nuestra.** El local trend model **es** el modelo de suavizado exponencial simple.

**Verificación numérica [B, aritmética sobre cifras A]:** en el ejemplo de Alcoa, el ARIMA(0,1,1) tiene $\hat\theta = 0.858$, lo que en la parametrización de suavizado exponencial corresponde a un parámetro $\alpha = 1-\theta = 0.142$. Y la ganancia de estado estacionario del Kalman, calculada desde $\hat\sigma_e = 0.4803$ y $\hat\sigma_\eta = 0.0735$, resulta $K_\infty \approx 0.142$. **Coinciden.**

### 31.3 Comparación honesta

| Aspecto | MA simple | EMA | Kalman (local level) |
|---|---|---|---|
| Pesos sobre el pasado | Iguales dentro de la ventana, cero fuera | Decaimiento geométrico | Decaimiento geométrico (en estado estacionario) |
| Parámetros a elegir | 1 ($n$) | 1 ($\alpha$) | 2 ($\sigma_e^2$, $\sigma_\eta^2$) — pero sólo importa el cociente |
| ¿Cómo se eligen? | A ojo / grid search | A ojo / grid search | **Máxima verosimilitud bajo un modelo** |
| ¿Produce incertidumbre? | No | No | **Sí** ($\Sigma_{t\mid t}$) |
| ¿Maneja faltantes? | Ad hoc | Ad hoc | **Naturalmente** |
| ¿Se generaliza a más componentes? | No | No | **Sí** (tendencia + pendiente + estacionalidad + ...) |
| ¿Se generaliza a multivariado? | Torpemente | Torpemente | **Sí, naturalmente** |
| ¿Los pesos dependen de los datos? | No | No | **No** (en el local level; ver Sección 6.4) |

### 31.4 Dónde está la diferencia real, y dónde no

**Donde NO hay diferencia (caso local level, estado estacionario):**

$$\boxed{\text{el Kalman local-level en régimen estacionario ES una EMA}}$$

Con un $\alpha$ específico determinado por la razón $\sigma_\eta^2/\sigma_e^2$. Funcionalmente, aplicar uno u otro produce la misma serie de salida.

**Donde SÍ hay diferencias reales:**

1. **Cómo se elige el parámetro.** La EMA requiere elegir $\alpha$ (típicamente a ojo o por grid search sobre alguna métrica). El Kalman lo deriva de un modelo estimado por máxima verosimilitud. **[B]** Esto es una ventaja metodológica genuina: hay un criterio principiado en lugar de una elección arbitraria. Pero atención — el criterio de máxima verosimilitud optimiza el **ajuste del modelo**, que no es lo mismo que optimizar una métrica de trading.

2. **La incertidumbre.** El Kalman produce $\Sigma_{t|t}$. La EMA no produce nada equivalente. (Aunque, como notamos en la Sección 13.6, en el local level $\Sigma_{t|t}$ converge a una constante, con lo que su valor informativo es limitado.)

3. **La generalidad.** La EMA hace una cosa. El state-space maneja múltiples componentes, múltiples series, matrices variables en el tiempo, faltantes, y todo con el mismo algoritmo. **[B]** Ésta es probablemente la ventaja más real: no es que el Kalman haga mejor lo que hace la EMA, es que hace muchas más cosas.

4. **El transitorio.** Al arranque, con inicialización difusa, la ganancia del Kalman es alta y va bajando. Una EMA con $\alpha$ fijo no tiene esa fase de "aprendizaje". **[B]** Es una diferencia real pero de efecto acotado en muestras largas.

### 31.5 El guardrail

$$\boxed{\text{si un Kalman local-level termina comportándose como un smoother adaptativo, no debemos venderlo como una fuente misteriosa de señal}}$$

**[B]** Y con más precisión: en el caso local level, ni siquiera es "adaptativo" después del transitorio. Es un smoother de pesos fijos.

Esto no es una crítica al método. Es una descripción honesta de lo que hace. La sofisticación del Kalman Filter reside en su **generalidad** y en su **fundamento probabilístico**, no en una capacidad mágica de extraer señal donde no la hay.

### 31.6 Qué NO afirmamos

- ❌ **No** afirmamos que el Kalman sea mejor que la EMA.
- ❌ **No** afirmamos que la EMA sea mejor que el Kalman.
- ❌ **No** afirmamos que suavizar mejore la predictibilidad. Suavizar reduce varianza a costa de introducir retraso; si ese intercambio es favorable depende de la tarea.
- ❌ **No** afirmamos que un filtro con más parámetros ajuste mejor fuera de muestra.

**PREGUNTA ABIERTA (H11.2).** ¿Un estado filtrado supera de forma estable a una EMA o a estadísticos rolling comparables, en alguna tarea concreta y con evaluación fuera de muestra honesta? Dado el resultado de la Sección 31.2 —que en el caso más simple **son el mismo objeto**— el prior razonable es que la diferencia, si existe, será pequeña y difícil de establecer. Y el benchmark obligatorio para cualquier feature Kalman es una EMA con parámetro comparable.

---

## 32. Causalidad temporal para ML [B]

**Toda esta sección es [B]. Tsay no discute leakage; discute inferencia estadística sobre muestras históricas.**

### 32.1 Los tres estados y su disponibilidad

Si alguna vez se utilizara un estado como feature, **debe especificarse exactamente cuál**:

| Estado | Notación | Información usada | ¿Uso causal posible? |
|---|---|---|---|
| **Predicho** | $\hat s_{t\mid t-1}$ | $y_1,\ldots,y_{t-1}$ | **Sí**, incluso antes de observar $t$ |
| **Filtrado** | $\hat s_{t\mid t}$ | $y_1,\ldots,y_t$ | **Sí**, una vez que $y_t$ está disponible |
| **Suavizado** | $\hat s_{t\mid T}$ | $y_1,\ldots,y_T$, con $T>t$ | **NO** para una decisión en $t$ |

Escribir "el estado Kalman" sin especificar cuál de los tres es una ambigüedad que puede costar un proyecto entero.

### 32.2 "Disponible en $t$" depende de cuándo se toma la decisión

Ésta es una sutileza que no es específica del Kalman pero que se agrava con él.

Supongamos una barra de 1 minuto etiquetada **10:00**. ¿Qué contiene?

En la convención habitual, contiene los datos **acumulados entre 10:00:00 y 10:00:59**. Es decir, su precio de cierre, su máximo, su mínimo y su volumen **no se conocen hasta las 10:00:59**.

Por lo tanto:

- $\hat s_{10:00|10:00}$ (estado filtrado de la barra 10:00) **no está disponible a las 10:00:00**.
- **Sí** está disponible a partir de las 10:01:00 (o 10:00:59, según el detalle del feed).

**[B]** Consecuencia: una decisión que se toma "en la barra de las 10:00" tiene que ser precisa sobre si se toma al abrir la barra (y entonces sólo puede usar información hasta 09:59:59) o al cerrarla.

**No diseñamos ninguna regla concreta.** Sólo señalamos la necesidad de **semántica temporal precisa**: cada feature necesita una respuesta explícita a "¿en qué segundo exactamente existía este número?".

Esto ya lo sabíamos del Capítulo 5, pero se agrava aquí porque las salidas de un filtro tienen un índice temporal $t$ que sugiere disponibilidad en $t$, y no siempre es así.

### 32.3 Causalidad del filtro ≠ causalidad de los parámetros

**Éste es el riesgo más sutil de todo el capítulo, y el más fácil de pasar por alto.**

La recursión del Kalman Filter es causal. $\hat s_{t|t}$ sólo usa $y_1,\ldots,y_t$. Eso es verdad y es verificable línea por línea en el código.

Pero: ¿de dónde salieron $\sigma_e$, $\sigma_\eta$ y las demás matrices?

**[A]** En Tsay, de maximizar la verosimilitud **sobre toda la muestra**. Es lo correcto para su propósito.

**[B]** Pero entonces, estrictamente:

$$\hat s_{t|t} = f\big(\underbrace{y_1,\ldots,y_t}_{\text{causal}};\ \underbrace{\hat\theta(y_1,\ldots,y_T)}_{\textbf{NO causal}}\big)$$

**Ejemplo concreto:** queremos una feature para una barra de marzo de 2022. Corremos el filtro con parámetros estimados sobre 2019–2026. El filtro, en marzo de 2022, sólo usó datos hasta marzo de 2022. Pero los parámetros que usó **incorporan información de 2023, 2024, 2025 y 2026**.

**No podemos afirmar que esa feature "estaba disponible históricamente".** No lo estaba: en marzo de 2022, nadie podía conocer $\hat\sigma_e$ estimado con datos hasta 2026.

$$\boxed{\text{causalidad del filtro} + \text{causalidad del ajuste de parámetros}}$$

**deben evaluarse por separado.** Cumplir la primera no implica cumplir la segunda.

### 32.4 Por qué esto importa cuantitativamente

Se podría objetar: "pero los parámetros son sólo dos números; ¿cuánto puede afectar?".

**[B]** Bastante, potencialmente. Recordemos de la Sección 6.4 que en el local trend model **la secuencia entera de ganancias está determinada por los parámetros y no depende de los datos**. Los parámetros no son un ajuste menor: **son la totalidad de lo que define el filtro**. Cambiar la razón $\sigma_\eta^2/\sigma_e^2$ cambia todos los pesos, y por tanto toda la serie filtrada.

Un parámetro estimado con conocimiento del futuro es, esencialmente, "elegir el grado de suavizado que sabemos que funcionó bien en el período completo". Es una forma de overfitting difícil de ver.

**PREGUNTA ABIERTA (H11.6).** ¿Cuánto cambian los estados si los parámetros se estiman únicamente con datos disponibles hasta cada fold, frente a estimaciones globales? Es una pregunta empírica con respuesta medible. **No se ejecuta aquí.**

### 32.5 State estimate como target — advertencia adicional

**[B]** Un riesgo distinto y aún más grave: usar un estado **suavizado** como *target* de un modelo de ML.

La tentación es evidente y suena razonable: "el precio observado es ruidoso; entrenemos contra la tendencia suavizada, que es más limpia y más fácil de aprender".

**El problema:** $\hat s_{t|T}$ incorpora $y_{t+1},\ldots,y_T$. Si entrenamos un modelo para predecir $\hat s_{t|T}$ a partir de información hasta $t$, estamos entrenándolo para predecir **algo que contiene el futuro**. En el mejor caso el modelo aprende a aproximar una media móvil centrada; en el peor, las métricas de validación son directamente inválidas.

**Un caso especialmente insidioso:** si el conjunto de validación también usa targets suavizados calculados sobre toda la muestra, **la contaminación está en ambos lados** y ninguna métrica lo detectará. El modelo se verá excelente hasta el día que se opere en vivo.

**[B]** Regla que registramos (no adoptamos, porque no hay nada que adoptar todavía): **si un target o una feature se deriva de un smoother que vio datos futuros, existe contaminación, y debe justificarse explícitamente por qué no invalida la evaluación.** No basta con que "el modelo se ve bien".

### 32.6 Checklist de auditoría temporal [B]

Para uso futuro **si alguna vez** se llegara a considerar una feature de tipo state-space. No es un plan; es una lista de verificación registrada.

| # | Pregunta | Por qué |
|---|---|---|
| 1 | ¿Es $\hat s_{t\mid t-1}$, $\hat s_{t\mid t}$ o $\hat s_{t\mid T}$? | El tercero es leakage por construcción |
| 2 | ¿En qué instante exacto (hora:min:seg) existió este número? | La barra $t$ no está completa al inicio de $t$ |
| 3 | ¿Con qué datos se estimaron los parámetros del modelo? | Sección 32.3 |
| 4 | ¿Se reestiman los parámetros en cada fold, o son globales? | Determina si la feature es reproducible históricamente |
| 5 | ¿Cómo se inicializa el filtro en cada fold? | Sección 14.6 |
| 6 | ¿El target se derivó de algún suavizado? | Sección 32.5 |
| 7 | ¿Se puede reconstruir esta feature usando **exclusivamente** información disponible en su momento? | La pregunta resumen (H11.5) |
| 8 | ¿La feature es redundante con una EMA o un rolling stat? | Sección 31; H11.14 |

---

## 33. Implicancias para futuros [B]

**Toda esta sección es [B]. Ninguna afirmación aquí está respaldada por Tsay para futuros ni para MNQ.**

### 33.1 Qué NO nos dice este capítulo sobre futuros

Empecemos por lo que falta, que es mucho:

- Tsay **no** analiza ningún futuro en este capítulo.
- **No** analiza ningún dato intradiario de precios (el ejemplo intradiario es una **agregación diaria** de retornos de 10 minutos).
- **No** analiza ninguna serie con más de 340 observaciones (Alcoa 340, GM 168, J&J 84).
- **No** evalúa capacidad predictiva fuera de muestra.
- **No** evalúa utilidad económica.
- **No** considera costos de transacción.

Un dataset de MNQ de 1 minuto tiene del orden de $10^6$ observaciones. Ninguno de los ejemplos del capítulo se acerca ni remotamente a ese régimen. **Todo lo que sigue son preguntas, no extrapolaciones.**

### 33.2 Qué preguntas nos permite formular

**A. Sobre volatilidad.** El único ejemplo del capítulo con datos de alta frecuencia trata sobre una **medida de volatilidad**, no sobre precios. Y encuentra que el ruido de medición domina. **PREGUNTA ABIERTA:** ¿una medida de volatilidad de MNQ construida a partir de datos intradiarios admite una descomposición similar? ¿Con qué ventana de agregación?

**B. Sobre estacionalidad intradiaria.** Los modelos de componentes no observados están diseñados para separar patrones periódicos. La estacionalidad intradiaria (patrones por hora del día en volatilidad y volumen) es un fenómeno documentado. **PREGUNTA ABIERTA:** ¿un modelo de componentes puede separar ese patrón de forma estable? ¿Y separarlo aporta algo, o el patrón ya es conocido por todos y por tanto no explotable?

**C. Sobre gaps.** MNQ tiene interrupciones de varios tipos: cierre diario, fines de semana, festivos, rolls de contrato, halts. El state-space ofrece un mecanismo coherente para faltantes. **PREGUNTA ABIERTA:** ¿es mejor que las alternativas actuales? ¿Y qué gaps deben tratarse como "faltantes" y cuáles como algo distinto?

**D. Sobre relaciones dinámicas.** El resultado de GM sugiere cautela: dar libertad a los coeficientes no produjo variación. **PREGUNTA ABIERTA:** ¿alguna relación relevante en el contexto de MNQ presenta evidencia reproducible de coeficientes variables frente a un benchmark fijo?

### 33.3 La advertencia sobre "true price" — repetida deliberadamente

$$\text{NO debemos asumir que } \text{Price}_t = \text{TruePrice}_t + \text{Noise}_t$$

El Capítulo 5 mostró que el precio observado tiene efectos de microestructura. **Eso no implica que exista un precio "verdadero" recuperable.** El precio observado es, entre otras cosas, **el precio al que efectivamente se opera**. Un "precio verdadero" filtrado no es un precio al que se pueda ejecutar.

**[B]** Y hay un problema práctico además del conceptual: si construimos una señal a partir de la desviación entre el precio observado y un "precio filtrado", estamos construyendo una estrategia de reversión a la media respecto de una media móvil. Eso es una estrategia perfectamente conocida, con décadas de historia, cuya rentabilidad no es un descubrimiento del Kalman Filter. Presentarla como "el filtro detecta desviaciones del valor verdadero" sería una descripción engañosa de lo que hace.

### 33.4 Escala y régimen de datos

**[B]** Una observación sobre el cambio de escala que conviene registrar:

Con 340 observaciones (Alcoa), estimar 2 parámetros es cómodo. Con $10^6$ observaciones, se podrían estimar modelos mucho más ricos. Pero:

- Más observaciones **no** resuelven la no unicidad de la descomposición. Ese problema es estructural, no de tamaño muestral.
- Más observaciones **sí** permiten detectar violaciones de supuestos (no normalidad, no linealidad, no estacionariedad) que en muestras chicas pasan desapercibidas. Con $10^6$ observaciones, **cualquier** test de bondad de ajuste rechaza. Los diagnósticos de Tsay (Ljung–Box con p=0.56) no son transportables a ese régimen.
- Los datos intradiarios de futuros son notoriamente no gaussianos (colas pesadas, agrupamiento de volatilidad), y el Kalman clásico asume normalidad. Ver Sección 34.

---

## 34. Implicancias para Machine Learning [B]

**Toda esta sección es [B].**

### 34.0 Primero: qué supone el Kalman Filter clásico

Antes de las categorías, es necesario delimitar el alcance del método.

**[A]** El Kalman Filter que desarrolla Tsay opera en un marco **lineal state-space con perturbaciones gaussianas**.

**¿Qué significa "lineal"?** Las ecuaciones combinan los estados mediante matrices y coeficientes lineales:
$$s_{t+1} = d_t + T_ts_t + R_t\eta_t, \qquad y_t = c_t + Z_ts_t + e_t$$
No hay $s_t^2$, ni $\exp(s_t)$, ni una red neuronal aplicada a $s_t$. La observación es una **combinación lineal** del estado más ruido.

**¿Qué significa "gaussiano"?** Las perturbaciones $\eta_t$ y $e_t$ tienen distribución normal bajo el modelo. Esto es lo que permite que toda la inferencia se reduzca a medias y covarianzas (por el Teorema 11.1: si todo es conjuntamente normal, las condicionales son normales y se calculan con álgebra lineal).

**Qué NO se sigue de esto:**

- ❌ **No** se sigue que "todos los estados financieros sean gaussianos". Los retornos financieros de alta frecuencia tienen colas notoriamente pesadas. Suponer normalidad es una **conveniencia matemática**, y su violación tiene consecuencias: el filtro sobrerreacciona o subreacciona ante outliers, y las bandas de confianza son incorrectas.
- ❌ **No** se sigue que "el Kalman Filter sirva para cualquier modelo no lineal". No sirve. Fuera del marco lineal-gaussiano, deja de ser el estimador óptimo y hay que recurrir a otra cosa.

**Contexto de delimitación [B], en un párrafo, sólo para saber dónde termina el alcance:** existen extensiones para el caso no lineal — el *Extended Kalman Filter* (lineariza localmente), el *Unscented Kalman Filter* (propaga puntos representativos), y los *particle filters* (aproximan la distribución por simulación). **No estudiamos ninguno de estos métodos.** Se mencionan únicamente para dejar claro que el Kalman clásico del Capítulo 11 no los cubre y que el capítulo no los desarrolla.

**[A]** Vale notar que Tsay mismo apunta hacia esa frontera al mencionar los modelos de volatilidad estocástica: son un caso donde la no linealidad obliga a métodos de simulación, y por eso aparecen en el Capítulo 12 (MCMC) y no aquí.

### 34.A Dynamic features

**Pregunta:** ¿podría un estado filtrado **causal** resumir dinámicamente información histórica de una forma útil?

**Argumento a favor:** un estado filtrado es un resumen recursivo de todo el pasado, con memoria decreciente controlada por un parámetro estimado en lugar de elegido a mano.

**Argumento en contra:** en el caso local level, es una EMA (Sección 31.2). En casos más complejos, es una combinación lineal de EMAs.

**PREGUNTA ABIERTA (H11.1, H11.14).**

### 34.B Denoising

**Pregunta:** ¿separar estado y ruido de medición mejora alguna tarea predictiva?

**Argumento a favor:** menos varianza en el input podría facilitar el aprendizaje.

**Argumento en contra, importante:** "ruido de medición" es una **categoría del modelo**, no una propiedad verificada de los datos. Que el modelo asigne el 97% de la varianza a $e_t$ no prueba que ese 97% sea impredecible; prueba que **ese modelo** no le encontró estructura dinámica. Y hay un riesgo simétrico: si parte de la señal está en lo que el modelo llamó ruido, suavizar **destruye** información.

$$\text{denoising} \neq \text{aumento automático de predictibilidad}$$

**PREGUNTA ABIERTA (H11.1).**

### 34.C Time-varying relationships

**Pregunta:** ¿alguna relación entre variables cambia lo suficiente como para justificar coeficientes dinámicos?

**Evidencia disponible [A]:** en el único caso que Tsay examina (GM vs. S&P, mensual, 1990–2003), **no**. Los coeficientes resultaron esencialmente constantes.

**PREGUNTA ABIERTA (H11.7).** Con benchmark obligatorio de coeficientes fijos.

### 34.D Missing data

**Pregunta:** ¿el state-space ofrece una forma más coherente de manejar faltantes?

**Argumento a favor [A]:** propaga incertidumbre en lugar de inventar valores. Es conceptualmente más limpio que cualquier imputación.

**Cautela [B]:** requiere primero decidir qué es un "faltante" en datos de futuros, y esa decisión es previa y no trivial (Sección 25.4).

**PREGUNTA ABIERTA (H11.9).**

### 34.E Forecast uncertainty

**Pregunta:** ¿la incertidumbre del estado o del forecast aporta información útil además del valor puntual?

**Argumento a favor:** el proyecto IRIS explícitamente quiere producir un nivel de confianza junto con la señal.

**Argumento en contra, fuerte:** en un modelo lineal-gaussiano invariante en el tiempo, $\Sigma_{t|t}$ **converge a una constante y no depende de los datos** (Secciones 6.4 y 22.4). Una feature constante no aporta nada. Sólo en modelos con matrices variables ($Z_t$ dependiente de datos) la incertidumbre varía — y varía en función de los regresores, no de los errores.

**PREGUNTA ABIERTA (H11.8).** Con la observación de que en el caso más simple la respuesta es trivialmente "no aporta nada, porque es constante".

### 34.F Feature engineering

**Pregunta:** ¿un estado filtrado contiene información **incremental** respecto de EMAs, rolling statistics u otras transformaciones causales de los mismos datos?

Ésta es la pregunta correcta, y se responde con una comparación empírica, no con un argumento.

**PREGUNTA ABIERTA (H11.14).**

### 34.G Leakage — **RIESGO CRÍTICO**

**Pregunta:** ¿se está usando algún estado suavizado que contiene futuro?

**No es una pregunta abierta. Es un riesgo a auditar.**

Incluye:
- $\hat s_{t|T}$ como feature.
- $\hat s_{t|T}$ como target.
- Disturbances suavizados ($e_{t|T}$, $\eta_{t|T}$) en cualquier rol.
- Parámetros estimados con datos futuros (Sección 32.3).
- Cualquier función de una librería llamada `smooth`, `RTS`, `fixed_interval`, `KalmanSmoother` sin auditar qué información usa.

**Checklist en Sección 32.6.**

### 34.H El Kalman Filter no puede inventar información

**Guardrail especial, que engloba a todos los anteriores.**

Supongamos que todas nuestras entradas derivan del mismo OHLCV. Construir:

- una EMA;
- una media rolling;
- un nivel filtrado;
- una pendiente filtrada;
- un estado Kalman multivariado;

**no significa haber añadido fuentes nuevas de información.** Son:

$$\boxed{\text{transformaciones de información ya disponible}}$$

Todas son funciones de $\{y_1,\ldots,y_t\}$. El contenido informativo del conjunto no cambia porque lo miremos de otra forma. **[A]** Es exactamente lo que Tsay establece al mostrar la equivalencia state-space ↔ ARIMA y al reproducir OLS con un filtro de Kalman.

La pregunta real es:

> ¿**alguna representación hace más fácil** para un modelo extraer una relación **estable fuera de muestra**?

Y **no**:

> ~~¿cuántas features nuevas puedo fabricar?~~

**[B]** Esta segunda pregunta tiene una respuesta trivial (infinitas) y es activamente dañina: multiplicar transformaciones del mismo input infla el espacio de búsqueda, aumenta la probabilidad de encontrar relaciones espurias, y no aporta información. En un problema con relación señal/ruido baja, ése es el modo de fallo más probable.

### 34.I Una nota sobre "buen fit" y OOS

$$\text{buen fit in-sample} \neq \text{capacidad de forecasting OOS}$$

Ninguno de los tres ejemplos del capítulo evalúa capacidad predictiva fuera de muestra. Todos evalúan **adecuación del modelo a la muestra** (residuos, Ljung–Box, ARCH LM). Son cosas distintas y no se implican.

**[B]** Un modelo puede pasar todos los diagnósticos de residuos y no tener ningún valor predictivo. De hecho, el local trend model aplicado a un paseo aleatorio es exactamente eso: ajusta perfectamente y predice que mañana será igual a hoy.

---

## 35. Hipótesis para backlog

**Ninguna de estas hipótesis se ejecuta.** Se registran para una fase futura de experimentación, si alguna vez se decide abordarla.

---

### H11.1 — Filtered state vs. raw observation

**Pregunta.** ¿Una representación causal filtrada contiene información predictiva incremental respecto de la observación original?

**Datos necesarios.** Serie objetivo de MNQ (a definir), con marcas temporales precisas. Un objetivo de predicción definido con horizonte explícito.

**Método.** Comparar un modelo predictivo alimentado con (a) la observación cruda, (b) el estado filtrado $\hat s_{t|t}$, (c) ambos. Evaluación estrictamente OOS con separación temporal.

**Resultado que la apoyaría.** (b) o (c) supera consistentemente a (a) en múltiples folds y múltiples definiciones del objetivo, con magnitud que sobrevive a corrección por comparaciones múltiples.

**Resultado que la refutaría.** Diferencias no distinguibles de ruido; o mejora que desaparece al cambiar de período.

**Riesgos/limitaciones.** El estado filtrado es una función determinista de las observaciones; cualquier mejora sería por facilidad de extracción, no por información nueva. Riesgo de leakage vía parámetros (H11.6). Riesgo de comparar contra un benchmark débil.

---

### H11.2 — Kalman vs. simple smoothers

**Pregunta.** ¿Un estado filtrado supera de forma estable a EMA/rolling statistics comparables?

**Datos necesarios.** Los mismos que H11.1.

**Método.** Benchmark explícito: EMA con $\alpha$ optimizado en train, media rolling con ventana optimizada en train, versus estado filtrado. Comparación OOS.

**Resultado que la apoyaría.** El estado filtrado supera a ambos benchmarks de forma reproducible.

**Resultado que la refutaría.** Rendimiento equivalente — **que es el resultado esperado a priori**, dado que en el local level son literalmente el mismo objeto (Sección 31.2).

**Riesgos/limitaciones.** Si el benchmark EMA se elige mal (parámetro arbitrario), la comparación es injusta a favor del Kalman. Debe optimizarse el benchmark con el mismo esfuerzo.

---

### H11.3 — Measurement/state-noise ratio

**Pregunta.** ¿Qué sensibilidad tienen los estados estimados a la razón entre ruido de medición e innovación del estado?

**Datos necesarios.** Una serie y un rango de valores de $q = \sigma_\eta^2/\sigma_e^2$.

**Método.** Análisis de sensibilidad: variar $q$ sobre varios órdenes de magnitud y medir cuánto cambian los estados filtrados y cualquier métrica derivada.

**Resultado que la apoyaría (como advertencia).** Alta sensibilidad → cualquier conclusión depende críticamente de un parámetro estimado.

**Resultado que la refutaría.** Baja sensibilidad en el rango relevante → conclusiones robustas.

**Riesgos/limitaciones.** Es un estudio de sensibilidad, no una prueba de utilidad. Su valor es diagnóstico.

---

### H11.4 — Filter vs. smoother

**Pregunta.** ¿Cuánto difieren los estados filtrados y suavizados, y qué revela esa diferencia sobre la incertidumbre retrospectiva?

**Datos necesarios.** Una serie y un modelo especificado.

**Método.** Calcular ambos, medir la magnitud de $\hat s_{t|T} - \hat s_{t|t}$ y comparar $\Sigma_{t|T}$ con $\Sigma_{t|t}$.

**Resultado que la apoyaría (como advertencia).** Diferencia grande → cuantifica exactamente cuánto "hace trampa" un smoother, y por tanto cuánto se inflaría un backtest contaminado.

**Resultado que la refutaría.** Diferencia despreciable.

**Riesgos/limitaciones.** **NO utilizar el smoother como feature causal bajo ninguna circunstancia.** Este experimento es puramente diagnóstico: mide la magnitud del leakage potencial, no lo explota.

---

### H11.5 — Leakage audit

**Pregunta.** ¿Todas las variables state-space hipotéticas pueden reconstruirse utilizando **exclusivamente** información disponible en el momento correspondiente?

**Datos necesarios.** La especificación completa de cada feature candidata.

**Método.** Auditoría según el checklist de la Sección 32.6. Test de reproducibilidad: recalcular cada feature en $t$ usando **sólo** datos hasta $t$ y verificar que el valor coincide con el del pipeline.

**Resultado que la apoyaría.** Coincidencia exacta.

**Resultado que la refutaría.** Cualquier discrepancia → hay leakage.

**Riesgos/limitaciones.** Es la auditoría más importante de la lista. Debería ser **bloqueante**: ninguna feature state-space debería entrar en un experimento sin pasarla.

---

### H11.6 — Parameter-estimation causality

**Pregunta.** ¿Cuánto cambian los estados si los parámetros se estiman únicamente con datos disponibles hasta cada fold, frente a estimaciones globales?

**Datos necesarios.** Un esquema walk-forward definido.

**Método.** Calcular la feature de dos formas —parámetros globales vs. parámetros re-estimados por fold— y comparar tanto los valores como el rendimiento del modelo posterior.

**Resultado que la apoyaría (como advertencia).** Diferencia material → la estimación global es una fuente real de contaminación.

**Resultado que la refutaría.** Diferencia despreciable → la estimación global es aceptable como aproximación.

**Riesgos/limitaciones.** La re-estimación por fold introduce sus propios problemas (folds tempranos con pocos datos, inestabilidad de las estimaciones). Ver también Sección 14.6 sobre inicialización por fold.

---

### H11.7 — Time-varying coefficients

**Pregunta.** ¿Alguna relación observable presenta evidencia reproducible de coeficientes variables en el tiempo frente a un benchmark fijo?

**Datos necesarios.** Un par de series con relación económicamente justificable (no una relación arbitraria).

**Método.** Estimar el modelo de coeficientes variables y testear si $\hat\sigma_\eta$, $\hat\sigma_\epsilon$ son distinguibles de cero. **[A]** Es exactamente el procedimiento del Ejemplo 11.2.

**Resultado que la apoyaría.** Varianzas de innovación significativamente distintas de cero, reproducible en subperíodos.

**Resultado que la refutaría.** Varianzas cercanas a cero — **el resultado que Tsay obtuvo para GM**.

**Riesgos/limitaciones.** El prior debe ser escéptico. Y el par de series debe justificarse antes de mirar los datos, no elegirse por resultado.

---

### H11.8 — State uncertainty

**Pregunta.** ¿La incertidumbre del estado aporta información adicional respecto del estado puntual?

**Datos necesarios.** Un modelo donde $\Sigma_{t|t}$ **no** sea constante (es decir, con matrices de sistema variables).

**Método.** Incluir $\Sigma_{t|t}$ (o su raíz) como feature adicional y medir aporte incremental OOS.

**Resultado que la apoyaría.** Mejora incremental reproducible.

**Resultado que la refutaría.** Sin aporte.

**Riesgos/limitaciones.** **Verificar primero que $\Sigma_{t|t}$ no sea constante.** En modelos invariantes converge a una constante (Secciones 6.4, 22.4) y la hipótesis es vacua por construcción.

---

### H11.9 — Missing observations

**Pregunta.** ¿Cómo se comporta una representación state-space frente a gaps reales/simulados en los datos?

**Datos necesarios.** Serie con gaps naturales (overnight, festivos, rolls) y una serie con gaps artificiales inyectados.

**Método.** Comparar propagación de estado e incertidumbre frente a estrategias de imputación alternativas. Medir el error de reconstrucción sobre los gaps artificiales (donde sí conocemos la verdad).

**Resultado que la apoyaría.** Mejor calibración de la incertidumbre y menor error de reconstrucción.

**Resultado que la refutaría.** Sin ventaja sobre alternativas simples.

**Riesgos/limitaciones.** Requiere haber decidido antes qué gaps son "faltantes" (Sección 25.4). Los gaps artificiales no reproducen las propiedades de los reales.

---

### H11.10 — Forecast benchmark

**Pregunta.** ¿Un modelo state-space mejora el forecasting OOS frente a ARIMA/EMA/naïve comparables?

**Datos necesarios.** Serie objetivo, horizonte definido, esquema walk-forward.

**Método.** Comparación directa contra benchmarks, con métricas de error y tests de significancia apropiados para pronósticos anidados.

**Resultado que la apoyaría.** Mejora consistente y significativa.

**Resultado que la refutaría.** Equivalencia — **el resultado esperado**, dada la equivalencia matemática establecida en la Sección 8.

**Riesgos/limitaciones.** Si el state-space es matemáticamente equivalente al ARIMA de comparación, no puede haber diferencia salvo por detalles numéricos. Ver H11.12.

---

### H11.11 — Intraday seasonality

**Pregunta.** ¿Un modelo de componentes no observados puede separar un patrón temporal determinista de la dinámica residual de forma estable?

**Datos necesarios.** Serie intradiaria de MNQ con marca de hora del día; definición del período estacional.

**Método.** Modelo de componentes con estacionalidad de período correspondiente a la sesión. Verificar estabilidad del patrón estimado entre subperíodos.

**Resultado que la apoyaría.** Patrón estacional estimado estable, reproducible en subperíodos independientes.

**Resultado que la refutaría.** Patrón inestable o sensible a la especificación.

**Riesgos/limitaciones.** **Crítico:** la descomposición **no es única** (Sección 28). Un patrón estacional estimado no es "el patrón real". Además: días festivos, medias sesiones, cambios de horario, rolls. Y aunque el patrón exista y sea estable, **no se sigue que sea explotable** — un patrón conocido por todos los participantes ya está en los precios.

---

### H11.12 — Representation equivalence

**Pregunta.** Si un state-space es matemáticamente equivalente a un ARIMA, ¿produce alguna ventaja predictiva, o únicamente computacional/interpretativa?

**Datos necesarios.** Un modelo local trend y su ARIMA(0,1,1) equivalente.

**Método.** Ajustar ambos, verificar que producen pronósticos idénticos (dentro de tolerancia numérica), y comparar tiempo de cómputo, manejo de faltantes y facilidad de extensión.

**Resultado esperado [A].** **Equivalencia predictiva exacta.** Tsay ya lo establece teóricamente y lo verifica numéricamente en el Ejemplo 11.2 para el caso de regresión.

**Valor del experimento.** No es descubrir algo nuevo; es **verificar que nuestra implementación es correcta**. Si los dos no coinciden, hay un bug.

**Riesgos/limitaciones.** Es una prueba de validación de código, no una hipótesis de investigación.

---

### H11.13 — Dynamic volatility state

**Pregunta.** ¿Una representación latente de una medida de volatilidad aporta información adicional respecto de indicadores de volatilidad observables?

**Datos necesarios.** Una medida de volatilidad realizada de MNQ (con ventana de agregación a definir) y benchmarks observables (ATR, desviación estándar rolling, volatilidad GARCH del Cap. 3).

**Método.** Modelo local trend sobre el log de la medida (siguiendo el Ejemplo 11.1), comparación del estado filtrado contra los benchmarks para alguna tarea de predicción de volatilidad.

**Resultado que la apoyaría.** Aporte incremental OOS.

**Resultado que la refutaría.** Redundancia con benchmarks más simples.

**Riesgos/limitaciones.** El Ejemplo 11.1 es el precedente más cercano, pero es sobre una acción, con 340 días y agregación de 10 minutos. La elección de la ventana de agregación es un grado de libertad importante (ver Ejercicio 11.2 del libro). Y **[A]** el propio ejemplo de Alcoa sugiere que la mayor parte de la variación se clasifica como ruido de medición, lo que limita cuánto puede aportar el estado filtrado.

---

### H11.14 — Feature redundancy

**Pregunta.** ¿Estados filtrados derivados de OHLCV aportan información incremental sobre features causales más simples derivadas del **mismo** OHLCV?

**Datos necesarios.** Un conjunto de features simples ya establecido y las features state-space candidatas.

**Método.** Análisis de información incremental: correlación entre features, importancia condicional, comparación de modelos con y sin las features Kalman.

**Resultado que la apoyaría.** Aporte incremental medible y estable.

**Resultado que la refutaría.** Redundancia — alta correlación con EMAs existentes y sin mejora al añadirlas.

**Riesgos/limitaciones.** **Ésta es probablemente la hipótesis más importante de la lista**, porque su respuesta esperada a priori es "redundante" (Sección 31.2) y confirmarlo ahorraría trabajo. Debe evaluarse antes que H11.1 y H11.2.

---

### H11.15 — Economic relevance

**Pregunta.** Si existe una mejora estadística del forecast, ¿sobrevive a una evaluación económica independiente?

**Datos necesarios.** Cualquier mejora estadística previamente establecida, más una especificación realista de costos.

**Método.** Traducir la mejora estadística a una evaluación con costos de transacción, slippage y restricciones operativas realistas.

**Resultado que la apoyaría.** La mejora sobrevive.

**Resultado que la refutaría.** La mejora se evapora al incorporar costos — **el resultado más común en la práctica**.

**Riesgos/limitaciones.** Esta hipótesis es **bloqueante para todas las demás**: ninguna mejora estadística tiene valor para el proyecto si no sobrevive aquí. Debe ser el filtro final, no un pensamiento posterior.

---

## 36. Auditoría de las 40 afirmaciones

Clasificación: **CORRECTA** / **INCORRECTA** / **REQUIERE CONDICIONES**, con explicación sencilla.

---

**1. "Un state-space model descubre el verdadero estado oculto del mercado."**

**INCORRECTA.** Un state-space model **postula** un estado y lo **estima dentro de su propia especificación**. No descubre nada preexistente. **[A]** Tsay advierte que existen infinitas descomposiciones posibles de una serie en componentes no observados; si hubiera un estado verdadero, no dependería de qué modelo eligiéramos para buscarlo. Además, la palabra "mercado" agrava el error: los ejemplos del capítulo son una volatilidad realizada, una beta de una acción y un EPS trimestral.

---

**2. "Una variable latente es una variable que no observamos directamente."**

**CORRECTA.** Ésa es exactamente la definición, y no dice nada más. En particular, no dice que sea "real pero escondida". Es una variable del modelo que no aparece en los datos.

---

**3. "La observation equation describe cómo el estado se relaciona con lo observado."**

**CORRECTA.** **[A]** Tsay: la ecuación (11.1) *"provides the link between the data $y_t$ and the state $\mu_t$ and is called the observation equation with measurement error $e_t$."* En el caso general, $y_t = c_t + Z_ts_t + e_t$.

---

**4. "La state equation describe cómo evoluciona el estado."**

**CORRECTA.** **[A]** Tsay: la ecuación (11.2) *"governs the time evolution of the state variable and is the state equation (or state transition equation) with innovation $\eta_t$."* Y en el caso general, *"describes a first-order Markov Chain to govern the state transition."*

---

**5. "Measurement noise y state innovation son lo mismo."**

**INCORRECTA.** Son estructuralmente distintos y ésa es la distinción central del modelo. El error de medición ($e_t$) afecta sólo la observación de hoy y **no persiste**. La innovación del estado ($\eta_t$) mueve el estado y **queda incorporada para siempre**. En el local trend model, toda la memoria de la serie viene de $\eta$; $e$ no aporta memoria alguna.

---

**6. "Todo cambio observado debe atribuirse a un cambio real del estado."**

**INCORRECTA.** Ése es precisamente el problema que el filtro resuelve. Un cambio en la observación puede deberse al estado, al ruido de medición, o a una mezcla. El Kalman gain determina el reparto **según los parámetros del modelo**. **[A]** En el ejemplo de Alcoa, la mayor parte de la variación se atribuye a error de medición ($\hat\sigma_e^2 \approx 43\hat\sigma_\eta^2$).

---

**7. "Kalman gain determina cuánto de la nueva innovation entra en la actualización."**

**CORRECTA.** Es la formulación preferible. **[A]** Tsay: *"Kalman gain is the factor that governs the contribution of the new shock $v_t$ to the state variable $\mu_t$."* Formalmente, $\mu_{t|t} = \mu_{t|t-1} + K_tv_t$.

---

**8. "Kalman gain alto significa, bajo el modelo, que la nueva información tiene gran peso en el update."**

**CORRECTA**, con la calificación "bajo el modelo" que la afirmación ya incluye. Si $K_t \approx 1$, la estimación se mueve casi hasta la observación. Ocurre cuando la incertidumbre previa sobre el estado es grande frente al ruido de medición. **No** significa que la observación sea "confiable" en un sentido absoluto; significa que **relativa a mi ignorancia**, aporta mucho.

---

**9. "Filtrado y forecasting son sinónimos."**

**INCORRECTA.** **[A]** Tsay los define por separado: filtering estima $\mu_t \mid F_t$ (el **presente**); prediction estima $\mu_{t+h}$ o $y_{t+h} \mid F_t$ (el **futuro**). En el local trend model resulta que $\mu_{t+1|t} = \mu_{t|t}$ (el valor puntual coincide, porque el estado es un paseo aleatorio), pero **las incertidumbres difieren** ($\Sigma_{t+1|t} = \Sigma_{t|t} + \sigma_\eta^2$), y en modelos con $T_t \neq I$ ni siquiera coinciden los valores.

---

**10. "Filtering utiliza observaciones futuras."**

**INCORRECTA.** Por definición, $\mu_{t|t} = E(\mu_t \mid F_t)$ con $F_t = \{y_1,\ldots,y_t\}$. Sólo pasado y presente. **La recursión es causal.** (Lo que sí puede no ser causal son los parámetros — ver #13 y #14.)

---

**11. "Smoothing puede utilizar observaciones futuras respecto del estado estimado."**

**CORRECTA**, y de hecho **necesariamente** las utiliza. **[A]** Tsay define smoothing como estimar $\mu_t$ dado $F_T$ con $T > t$. La fórmula lo hace explícito: $\mu_{t|T} = \mu_{t|t-1} + \Sigma_{t|t-1}q_{t-1}$, donde $q_{t-1}$ es *"a weighted linear combination of the innovations $\{v_t,\ldots,v_T\}$"* — todas posteriores a $t$.

---

**12. "Un smoothed state histórico es automáticamente válido como feature de trading."**

**INCORRECTA**, y es el error más peligroso del capítulo. Un estado suavizado en $t$ contiene información de $t+1,\ldots,T$. Usarlo como feature de una decisión tomada en $t$ es look-ahead leakage **por construcción**, no por descuido. Y es traicionero porque produce resultados buenos-pero-no-absurdos, y no deja rastro visible en el código.

---

**13. "Filtering es necesariamente libre de leakage."**

**REQUIERE CONDICIONES.** La **recursión** del filtro es causal: $\hat s_{t|t}$ sólo usa $y_1,\ldots,y_t$. Pero los **parámetros** que alimentan esa recursión ($\sigma_e$, $\sigma_\eta$, las matrices) típicamente se estimaron sobre toda la muestra. Si es así, $\hat s_{t|t}$ depende del futuro a través de los parámetros. Para que filtering sea genuinamente libre de leakage se necesita **además** que los parámetros se hayan estimado sólo con información disponible en su momento. Ver #14.

---

**14. "Un filtro causal puede seguir contaminado si sus parámetros se ajustaron usando datos futuros."**

**CORRECTA.** Es el complemento exacto de #13.

$$\hat s_{t|t} = f\big(y_1,\ldots,y_t;\ \hat\theta(y_1,\ldots,y_T)\big)$$

Y en el local trend model esto es especialmente grave, porque **[A]** la secuencia entera de ganancias depende **únicamente** de los parámetros y de la inicialización, no de los datos. Los parámetros no son un ajuste menor: son todo lo que define el filtro.

---

**15. "State-space siempre proporciona una representación estadística distinta de ARIMA."**

**INCORRECTA.** La palabra "siempre" la vuelve falsa. **[A]** Tsay demuestra que el local trend model **es** un ARIMA(0,1,1); que cualquier ARMA admite varias representaciones state-space; y que, recíprocamente, un state-space lineal invariante con estado $m$-dimensional produce observaciones ARMA($m,m$). Y comenta: *"based on the data alone, the decision of using ARIMA models or linear state-space models is not critical."*

---

**16. "Un ARIMA puede escribirse en forma state-space."**

**CORRECTA.** **[A]** Tsay presenta **tres** formas distintas de hacerlo (Akaike, Harvey, Aoki), en la Sección 11.3.2.

---

**17. "Representar un ARIMA como state-space crea información predictiva adicional."**

**INCORRECTA.** Es la misma distribución de probabilidad escrita de otra manera. **[A]** *"For estimation and forecasting purposes, one can choose any one of those representations."* Verificado numéricamente en el Ejemplo 11.2: estimar una regresión como state-space reproduce OLS hasta el sexto decimal.

---

**18. "Kalman Filter es simplemente una media móvil."**

**REQUIERE CONDICIONES.** En el caso **específico** del local level model **en estado estacionario**, es funcionalmente equivalente a una media móvil exponencial con un parámetro determinado por la razón de varianzas — **[A]** Tsay establece explícitamente que el local trend model *"is the simple exponential smoothing model of Chapter 2"*. Pero en general, **no**: el Kalman Filter maneja estados multidimensionales, matrices variables en el tiempo, múltiples series observadas, faltantes, y produce incertidumbre. Ninguna media móvil hace eso. La afirmación es cierta en un caso particular y falsa como generalización.

---

**19. "Kalman Filter y EMA pueden tener relaciones conceptuales en casos específicos."**

**CORRECTA**, y de hecho la relación es **más fuerte que "conceptual"** en el caso del local level: es una equivalencia matemática establecida por Tsay, verificable numéricamente ($K_\infty \approx 0.142 = 1-\hat\theta$ en el ejemplo de Alcoa).

---

**20. "Filtered state es el valor verdadero sin ruido."**

**INCORRECTA.** Es la esperanza condicional del estado **bajo el modelo especificado, con los parámetros estimados**. Si $y_t = 105$ y $\mu_{t|t} = 101$, eso **no** significa que 105 estuviera "mal" y 101 fuera "lo correcto". Significa que, combinando la predicción previa, las incertidumbres y la nueva observación, la mejor estimación condicional bajo ese modelo es 101. Con otros parámetros habría sido otro número.

---

**21. "Smoothing produce una reconstrucción retrospectiva usando más información."**

**CORRECTA.** Y por eso es más preciso: **[A]** *"The confidence intervals for the smoothed state variables are also narrower than those of the filtered state variables."* La palabra clave es **retrospectiva**.

---

**22. "Más información retrospectiva implica mejor feature causal."**

**INCORRECTA**, y la contradicción es directa: si la información es **retrospectiva**, entonces no estaba disponible en el momento de la decisión, y por tanto no puede ser una feature **causal**. Los dos adjetivos son incompatibles.

$$\text{mejor reconstrucción histórica} \neq \text{feature válida para forecasting causal}$$

---

**23. "Un state-space model puede tener coeficientes que varían en el tiempo."**

**CORRECTA.** **[A]** Sección 11.3.1: el CAPM con $\alpha_t$ y $\beta_t$ evolucionando como paseos aleatorios. Y en 11.3.3, la extensión de la regresión con $\beta_{t+1} = \beta_t + R_t\eta_t$, donde $\sigma_i = 0$ recupera el coeficiente fijo.

---

**24. "Un coeficiente time-varying demuestra que el mercado cambió de régimen."**

**INCORRECTA**, por dos razones independientes:

1. **Confusión categorial.** Un coeficiente que deriva suavemente (paseo aleatorio continuo) no describe regímenes. Los regímenes son categorías discretas (Cap. 4, Markov switching). Son objetos distintos.
2. **Confusión entre modelo y realidad.** Que un modelo *permita* variación no demuestra que la variación *exista*. **[A]** El Ejemplo 11.2 lo muestra en la dirección contraria: se permitió variación y la estimación por máxima verosimilitud concluyó que $\hat\sigma_\eta = 4.91\times10^{-5}$ y $\hat\sigma_\epsilon = 1.22\times10^{-2}$, *"essentially constant"*.

---

**25. "State-space y Markov Switching son el mismo modelo."**

**INCORRECTA.** Estado continuo vs. discreto; evolución lineal vs. transiciones probabilísticas entre categorías; incertidumbre sobre un valor vs. sobre una categoría; Kalman filtering vs. Hamilton filtering. Son familias distintas.

---

**26. "Un estado Kalman normalmente es continuo mientras un régimen Markov Switching es discreto."**

**CORRECTA.** Es exactamente la distinción de #25. El adverbio "normalmente" es apropiado: en el marco lineal-gaussiano de este capítulo el estado es continuo por construcción.

---

**27. "Una descomposición trend + seasonal + irregular es única."**

**INCORRECTA**, y Tsay lo dice con máxima claridad. **[A]** *"the estimated components in Figure 11.6 are not unique. They depend on the model specified and constraints used. In fact, there are infinitely many ways to decompose an observed time series into unobserved components."*

---

**28. "Tsay advierte que distintas descomposiciones de componentes no observados pueden ser posibles."**

**CORRECTA**, y de hecho la advertencia de Tsay es **más fuerte** que "pueden ser posibles": dice que hay **infinitas**, y da un ejemplo concreto (usar `seasonalTrig` en lugar de `seasonalDummy` produce otra descomposición de la misma serie de J&J).

---

**29. "Kalman Filter clásico en este capítulo se desarrolla en un marco lineal-gaussiano."**

**CORRECTA.** **[A]** El modelo general (11.26)–(11.27) es lineal en el estado, con $\eta_t \sim N(0,Q_t)$ y $e_t \sim N(0,H_t)$. Toda la derivación se apoya en el Teorema 11.1 (propiedades de la normal multivariante), que requiere normalidad conjunta.

---

**30. "Por usar Kalman ya podemos modelar cualquier no linealidad."**

**INCORRECTA.** El Kalman clásico requiere linealidad. Fuera de ese marco deja de ser óptimo y hay que recurrir a otros métodos (EKF, UKF, particle filters, MCMC), **ninguno de los cuales se estudia en este capítulo**. **[A]** Los modelos de volatilidad estocástica son precisamente un caso donde la no linealidad obliga a métodos de simulación, y por eso aparecen en el Capítulo 12.

---

**31. "Missing values pueden manejarse propagando el estado sin update de la observación faltante."**

**CORRECTA.** **[A]** Se hace $v_t = 0$ y $K_t = 0$; el estado se propaga con $s_{t+1|t} = d_t + T_ts_{t|t-1}$ y la incertidumbre crece. Tsay: *"when $y_t$ is missing, there is no new innovation or new Kalman gain."*

---

**32. "Una barra faltante y una barra de volumen cero son necesariamente equivalentes."**

**INCORRECTA.** Una barra faltante significa "no tengo el dato". Una barra con volumen cero significa "tengo el dato: nadie operó". La segunda **es información**, y borrarla o tratarla como faltante la descarta. Tsay no trata esta distinción (no analiza datos de futuros); es una precisión **[B]** necesaria en nuestro contexto.

---

**33. "Forecast covariance aporta información sobre incertidumbre además del forecast puntual."**

**CORRECTA** como afirmación sobre lo que el modelo produce. **[A]** El filtro entrega $\mathrm{Var}[e_t(j)]$ junto con $y_t(j)$, y esa varianza crece con el horizonte.

**Matiz necesario [B]:** que aporte información *sobre la incertidumbre bajo el modelo* no implica que aporte información **explotable**. En un modelo lineal-gaussiano invariante, esa varianza es una función determinista del horizonte, idéntica todos los días. Ver #39.

---

**34. "Si un filtered state se ve más suave, necesariamente es más predictivo."**

**INCORRECTA.** "Más suave" significa que se descartó más variación como ruido. Eso reduce varianza e introduce retraso. Si el intercambio es favorable para una tarea concreta es una pregunta empírica. En el límite, un filtro infinitamente suave es una constante: perfectamente suave y sin ningún poder predictivo.

---

**35. "Denoising garantiza aumentar signal-to-noise para cualquier target."**

**INCORRECTA.** La palabra "garantiza" la vuelve indefendible, y "cualquier target" la agrava. Lo que el modelo llama "ruido" es una **categoría del modelo**, no una propiedad verificada. Si parte de la señal relevante para un target concreto estaba en lo que el modelo clasificó como ruido, el denoising la **destruye**.

---

**36. "Un modelo state-space bien ajustado in-sample garantiza forecasting OOS."**

**INCORRECTA.** No hay ninguna evaluación fuera de muestra en todo el capítulo. Los diagnósticos de Tsay (Ljung–Box, ARCH LM sobre residuos estandarizados) miden **adecuación a la muestra**, no capacidad predictiva futura. Son cosas distintas y no se implican.

---

**37. "Una feature Kalman puede ser redundante con transformaciones más simples."**

**CORRECTA**, y en el caso del local level es más que "puede": **[A]** es equivalente al suavizado exponencial simple. La redundancia con una EMA no es una posibilidad remota; es el caso base.

---

**38. "Una representación compleja debe justificar valor incremental OOS."**

**CORRECTA** como principio metodológico **[B]**. Es una afirmación normativa, no un resultado de Tsay, pero es coherente con todo el capítulo: si la representación compleja es matemáticamente equivalente a la simple, no puede haber valor incremental; si no es equivalente, el valor debe demostrarse, no asumirse.

---

**39. "La incertidumbre estimada del estado puede ser tan importante como su valor puntual."**

**REQUIERE CONDICIONES.** Como principio general es razonable: dos estimaciones idénticas con incertidumbres muy distintas son afirmaciones muy distintas, y **[A]** Tsay siempre grafica los estados con sus bandas del 95%.

**Pero:** en un modelo lineal-gaussiano invariante, $\Sigma_{t|t}$ **converge a una constante y no depende de los datos**. Una feature constante no aporta nada. La afirmación sólo tiene contenido operativo en modelos donde la incertidumbre efectivamente varía (matrices de sistema dependientes de datos). **Verificar antes de asumir.**

---

**40. "State-space puede ser útil aunque finalmente no se utilice Kalman como feature de ML."**

**CORRECTA**, y probablemente sea la conclusión más honesta del capítulo. Los usos que Tsay destaca —**[A]** simplificar la estimación por máxima verosimilitud, manejar valores faltantes de forma coherente, unificar modelos aparentemente distintos bajo un solo lenguaje, cuantificar incertidumbre— son valiosos **independientemente** de que un estado filtrado llegue alguna vez a ser una feature. El valor puede ser conceptual (entender qué significa "estado", "latente", "filtrado" vs. "suavizado") y metodológico (el marco de auditoría de leakage de la Sección 32) sin ninguna implementación.

---

## 37. Preguntas abiertas

Consolidadas de todo el informe. **Ninguna se responde aquí.**

### Sobre la aplicabilidad del marco

1. ¿Qué objetos de una serie de futuros, si alguno, admiten razonablemente la interpretación "cantidad latente medida con ruido"? La volatilidad tiene un argumento; el precio no lo tiene de la misma forma.
2. ¿Tiene sentido postular una descomposición de componentes no observados para datos intradiarios de MNQ? ¿Con qué período estacional, dados festivos, medias sesiones y rolls?
3. ¿El supuesto gaussiano es sostenible en datos intradiarios de futuros, con sus colas pesadas documentadas? ¿Qué se rompe si no lo es?

### Sobre features y representación

4. ¿Existe alguna tarea predictiva sobre MNQ donde una representación filtrada facilite la extracción de una relación estable que features causales más simples no faciliten? (H11.1, H11.14)
5. Dado que el Kalman local-level **es** una EMA, ¿qué justificaría preferirlo? ¿La estimación principiada del parámetro? ¿La generalización a más componentes? (H11.2)
6. ¿Un estado con dinámica impuesta e incertidumbre cuantificada aporta algo que un estado aprendido libremente (LSTM, etc.) no aporte, en un problema con relación señal/ruido baja?

### Sobre causalidad temporal

7. ¿Cuánto cambian los estados si los parámetros se estiman por fold en lugar de globalmente? (H11.6)
8. En un esquema walk-forward, ¿se reinicia el filtro en cada fold o se arrastra el estado? Ambas opciones tienen problemas distintos.
9. ¿Cómo se define con precisión "disponible en $t$" para las salidas de un filtro, dada la semántica de las barras?

### Sobre datos faltantes

10. ¿El gap overnight de MNQ debe tratarse como observaciones faltantes, como un salto genuino del estado, o como una discontinuidad que exige reinicializar? Las tres son defendibles y dan resultados distintos.
11. ¿Cómo distinguir sistemáticamente barras ausentes, barras de volumen cero y valores imputados en el pipeline actual?

### Sobre incertidumbre

12. ¿Existe alguna especificación donde $\Sigma_{t|t}$ **no** sea constante y donde su variación sea informativa? (H11.8)

### Sobre relaciones dinámicas

13. ¿Alguna relación relevante en MNQ presenta evidencia reproducible de coeficientes variables frente a un benchmark fijo? (H11.7) Prior escéptico, dado el resultado de GM.

### Sobre volatilidad

14. ¿Una medida de volatilidad de MNQ admite una descomposición similar a la de Alcoa? ¿Con qué ventana de agregación? (H11.13)
15. ¿Qué relación conceptual conviene establecer entre un estado latente de volatilidad y los modelos GARCH del Capítulo 3, que ya tenemos estudiados?

### Sobre evaluación

16. ¿Alguna mejora estadística de forecasting, si existiera, sobreviviría a una evaluación económica con costos realistas? (H11.15) **Ésta es la pregunta bloqueante.**

### Sobre el capítulo mismo

17. ¿Qué significa exactamente "descomposición válida" en la frase de Tsay *"for forecasting purposes, the choice of decomposition does not matter provided that the chosen one is a valid decomposition"*? Tsay no lo define formalmente en ese punto. Ver Sección 28.4 y Sección 40.

---

## 38. Checklist de conocimientos adquiridos

Verificación de que el lector puede explicar cada punto **sin mirar las fórmulas**.

| # | Pregunta | Respuesta breve |
|---|---|---|
| 1 | ¿Qué es un state-space model? | Dos ecuaciones: una que dice cómo evoluciona una cantidad que no veo, otra que dice cómo esa cantidad genera lo que sí veo |
| 2 | ¿Qué es un estado? | Un resumen de la situación del sistema que contiene toda la memoria relevante: dado el estado de hoy, el pasado más lejano no aporta nada más |
| 3 | ¿Qué significa latent? | No observado directamente. **Nada más.** No significa "real pero escondido" |
| 4 | ¿Qué es una observation equation? | Cómo el estado produce lo que medimos, más un error de medición |
| 5 | ¿Qué es una state equation? | Cómo el estado de hoy se convierte en el de mañana, más un empujón aleatorio |
| 6 | ¿Diferencia entre measurement noise y state innovation? | El error de medición contamina sólo la observación de hoy y se olvida; la innovación del estado mueve el estado y queda para siempre |
| 7 | ¿Qué significan sus varianzas? | Cuánto miente típicamente el instrumento ($\sigma_e^2$) y cuánto se mueve típicamente el estado ($\sigma_\eta^2$). Sólo importa el cociente |
| 8 | ¿Qué hace filtering? | Estima el estado **actual** usando todo lo observado **hasta ahora** |
| 9 | ¿Qué hace prediction? | Estima estados u observaciones **futuras** usando lo observado hasta ahora |
| 10 | ¿Qué hace smoothing? | Reestima estados **pasados** usando **toda** la muestra, incluido lo que pasó después |
| 11 | ¿Por qué smoothing usa futuro? | Porque los datos posteriores a $t$ contienen información sobre el estado en $t$: si el estado se mueve poco, lo que pasó después dice algo de lo que pasaba antes |
| 12 | ¿Por qué eso genera leakage? | Porque una decisión tomada en $t$ no puede usar datos de $t+1$. Es leakage **por construcción del algoritmo**, no por un bug |
| 13 | ¿Qué es una innovation? | La sorpresa: lo que observé menos lo que esperaba observar antes de verlo |
| 14 | ¿Qué es el Kalman gain? | La fracción de esa sorpresa que incorporo a mi estimación del estado |
| 15 | ¿Qué significa un Kalman gain grande? | Que estaba muy inseguro del estado frente al ruido del instrumento → le hago mucho caso al dato nuevo |
| 16 | ¿Y uno pequeño? | Que estaba muy seguro del estado frente a un instrumento ruidoso → casi ignoro el dato nuevo |
| 17 | ¿Cómo funciona Predict → Update? | Predigo estado e incertidumbre (la incertidumbre crece), observo, calculo la sorpresa, incorporo una fracción de ella (la incertidumbre baja), avanzo |
| 18 | ¿Qué es la uncertainty del estado? | La varianza de mi estimación. "100 ± 0.1" y "100 ± 10" son afirmaciones distintísimas |
| 19 | ¿Qué papel cumple initialization? | El filtro necesita empezar en algún lado. Afecta los primeros estados; el efecto se disipa pero **no es cero al principio** |
| 20 | ¿Qué significa diffuse initialization? | "No sé nada del estado inicial" = varianza inicial infinita. En la práctica el filtro adopta la primera observación como punto de partida |
| 21 | ¿Diferencia entre parameter estimation y state estimation? | Los parámetros son pocos y fijos (se estiman por máxima verosimilitud sobre toda la muestra); los estados son uno por instante (se filtran dado el modelo y los parámetros) |
| 22 | ¿Por qué un ARIMA puede escribirse como state-space? | Porque "estado" es cualquier resumen que contenga la memoria del proceso, y un ARMA tiene memoria finita que se puede resumir de varias maneras |
| 23 | ¿Eso añade información nueva? | **No.** Es la misma distribución de probabilidad escrita de otra forma |
| 24 | ¿Qué es un time-varying coefficient? | Un coeficiente de regresión tratado como estado que evoluciona en el tiempo |
| 25 | ¿Por qué no implica automáticamente régimen? | Porque un régimen es una categoría discreta y un coeficiente que deriva es un continuo. Y porque permitir dinámica no demuestra que exista dinámica |
| 26 | ¿Qué es un unobserved-component model? | Un modelo que escribe lo observado como suma de tendencia + estacionalidad + ciclo + irregular, ninguno de los cuales se ve por separado |
| 27 | ¿La descomposición de componentes es única? | **No. Hay infinitas.** Es la advertencia más importante del capítulo |
| 28 | ¿Cómo maneja missing values? | No actualiza (ganancia cero), sólo propaga el estado y deja crecer la incertidumbre. **No inventa un valor** |
| 29 | ¿Cómo produce forecasts? | Tratando los valores futuros como si fueran datos faltantes y propagando el filtro hacia adelante |
| 30 | ¿Diferencia entre filtered y smoothed feature? | La filtrada usa hasta $t$ (usable causalmente); la suavizada usa hasta $T>t$ (**no** usable causalmente) |
| 31 | ¿Por qué el Kalman no recupera "el precio verdadero"? | Porque no hay tal cosa dentro del modelo. Hay un estado postulado, y si cambio el modelo cambia el estado estimado. Lo que es verdadero no depende del modelo que uso para buscarlo |
| 32 | ¿Qué relación tiene con EMA? | En el caso local level en estado estacionario, **es** una EMA con un parámetro determinado por la razón de varianzas. Tsay lo dice explícitamente |
| 33 | ¿Qué relación tiene con GARCH? | En GARCH la volatilidad es calculable exactamente del pasado; en un state-space estocástico el estado tiene su propio ruido y nunca se calcula exactamente, sólo se estima |
| 34 | ¿En qué se diferencia de Markov Switching? | Estado continuo vs. discreto; incertidumbre sobre un valor vs. sobre una categoría |
| 35 | ¿Qué podría aportar a ML? | Un lenguaje para pensar features dinámicas, un marco coherente para faltantes, cuantificación de incertidumbre, y —sobre todo— un vocabulario preciso para auditar causalidad temporal |
| 36 | ¿Qué riesgos de leakage introduce? | Estados suavizados como features o targets; disturbances suavizados; parámetros estimados con datos futuros; ambigüedad sobre cuándo existió cada número |
| 37 | ¿Qué cosas NO hemos decidido todavía? | **Todas.** No se adoptó ninguna arquitectura, feature, target, transformación ni política de datos |

---

## 39. Conclusiones

### 39.1 Lo que el Capítulo 11 realmente aporta

Después de recorrerlo entero, el aporte se puede resumir en tres cosas — ninguna de las cuales es "una nueva forma de predecir".

**Primero: un lenguaje.** El state-space da vocabulario preciso para hablar de estados, latencia, medición, transición, filtrado, predicción y suavizado. Ese vocabulario **ya es valioso por sí mismo**, porque permite formular preguntas que antes eran vagas. "¿Esta feature usa el futuro?" es una pregunta que ahora sabemos cómo responder con precisión.

**Segundo: un algoritmo.** El Kalman Filter resuelve eficientemente la actualización recursiva de creencias en el marco lineal-gaussiano, y de paso hace tratable la estimación por máxima verosimilitud (la *prediction error decomposition*) y el manejo de faltantes. Son ventajas prácticas reales.

**Tercero, y más importante para nosotros: una lección epistemológica.** El estado existe **dentro del modelo**, no en el mundo. Una serie observada admite infinitas descomposiciones en componentes no observados. Cualquier "descubrimiento" sobre el estado es un descubrimiento sobre nuestra especificación.

### 39.2 Los cuatro guardrails, en su forma final

$$\boxed{\text{Kalman estimate} \neq \text{ground truth}}$$

$$\boxed{\text{smooth} \neq \text{predictive}}$$

$$\boxed{\text{state-space representation} \neq \text{new information}}$$

$$\boxed{s_{t|T}\ (T>t) \text{ puede ser excelente retrospectivamente y completamente inválido como feature causal en } t}$$

### 39.3 Lo que aprendimos a mirar

Al terminar este capítulo, mirando una serie $y_t$, entendemos que un state-space model plantea conceptualmente:

$$y_t = \text{función del estado}_t + \text{ruido de observación}$$

mientras:

$$\text{state}_{t+1} = \text{función de state}_t + \text{innovación del estado}$$

y que el Kalman Filter responde iterativamente a una única pregunta:

> **"Dado lo que esperaba, lo que acabo de observar, y la incertidumbre de ambas cosas, ¿cómo debo actualizar mi estimación del estado?"**

Y entendemos también que la respuesta a esa pregunta **depende enteramente de los parámetros que le dimos al modelo**, y que esos parámetros los elegimos nosotros.

### 39.4 El mapa de los capítulos estudiados

| Cap. | Pregunta central | Objeto añadido |
|---|---|---|
| **C1** | ¿Qué estamos observando? | Distribución, condicionalidad |
| **C2** | ¿Qué memoria lineal existe? | Media condicional, autocorrelación, forecasting |
| **C3** | ¿Cómo cambia la incertidumbre? | Varianza condicional |
| **C4** | ¿La regla puede cambiar o depender de estados? | No linealidad, regímenes discretos |
| **C5** | ¿Parte de lo observado es efecto del mecanismo de negociación? | Microestructura, distorsión del precio observado |
| **C7** | ¿Cómo cambia la distribución y sus colas? | Cuantiles, extremos, VaR |
| **C11** | **¿Podemos inferir dinámicamente una cantidad que no observamos directamente?** | **Estado latente continuo, incertidumbre del estado, distinción filtrado/suavizado** |

Y la advertencia que atraviesa todo:

$$\boxed{\text{latente} \neq \text{mágico}}$$

El estado latente existe dentro de una especificación estadística. Fuera de ella, no significa nada.

### 39.5 Lo que este capítulo NO nos autoriza a hacer

Ninguna de las siguientes cosas se hizo, y ninguna está justificada por lo aprendido:

- Implementar un Kalman Filter sobre datos del proyecto.
- Crear features filtradas o suavizadas.
- Modificar datasets, notebooks, targets, modelos, lookback, validación o folds.
- Seleccionar tendencia, pendiente, beta dinámica o volatilidad latente como features.
- Usar smoothing para generar labels.
- Imputar barras faltantes.
- Hacer tuning, ejecutar experimentos o comparar rentabilidad.
- Asumir que el precio contiene un "true price" recuperable.
- Adoptar state-space como arquitectura del proyecto.

$$\boxed{\text{KNOWLEDGE ACQUISITION, NOT DESIGN}}$$

### 39.6 Estado final

**KNOWLEDGE ACQUIRED — NO DESIGN DECISIONS ADOPTED.**

---

## 40. Registro de revisión crítica

Afirmaciones sensibles que aparecen en este informe o que se derivan naturalmente del capítulo, con su riesgo asociado y el estado en que quedan.

| Afirmación sensible | Riesgo | Estado |
|---|---|---|
| "El local trend model es un ARIMA(0,1,1), que es el suavizado exponencial simple" | Ninguno — es cita directa de Tsay [A] | **MANTENER** |
| "El Kalman local-level en estado estacionario es una EMA" | Podría leerse como descalificación del método en general | **MATIZAR**: cierto en ese caso particular; falso como generalización (ver #18 de la auditoría) |
| "$K_\infty \approx 0.142 = 1-\hat\theta$ en Alcoa" | Es aritmética propia sobre cifras [A], no una cifra publicada por Tsay | **MANTENER** con la marca explícita de que es verificación propia |
| "En Alcoa el ruido de medición domina" | Riesgo alto de generalizar a intradiario de futuros | **MATIZAR**: es sobre log de volatilidad realizada diaria de una acción, 340 obs., agregación de 10 min |
| "Los datos intradiarios tienen principalmente ruido de medición" | Generalización injustificada del ejemplo de Alcoa | **NO SE AFIRMA** — explícitamente rechazada en Sección 7.6 |
| "En GM los coeficientes son esencialmente constantes" | Riesgo de generalizar a "los betas son fijos" | **MATIZAR**: un activo, mensual, 168 obs., contra S&P 500 |
| "Permitir dinámica no implica encontrar dinámica" | Es una lección legítima, pero de un solo ejemplo | **MANTENER** con la aclaración de que es un caso, no una ley |
| "El Kalman gain no depende de los datos en el local trend model" | Cita directa de Tsay [A]; podría sobre-generalizarse a todos los state-space | **MATIZAR**: vale para el local trend model y para modelos invariantes; **no** cuando $Z_t$ depende de datos |
| "$\Sigma_{t\mid t}$ converge a una constante" | Cierto para modelos invariantes; falso en general | **MATIZAR**: se aclara en Secciones 13.6, 22.4, 34.E y #39 de la auditoría |
| "Smoothing es leakage" | Podría leerse como que smoothing es siempre malo | **MATIZAR**: es leakage **como feature causal**; es la herramienta correcta para análisis retrospectivo |
| "Un filtro causal puede estar contaminado vía parámetros" | Adaptación [B] sobre material [A]; Tsay no discute leakage | **MANTENER** como [B], claramente marcado |
| "Existe un precio verdadero detrás del precio observado" | Extrapolación injustificada del Cap. 5 al Cap. 11 | **NO SE AFIRMA** — explícitamente rechazada en Secciones 2.3, 21.4 y 33.3 |
| "El Kalman Filter recupera la señal limpia" | Confunde estado estimado con ground truth | **NO SE AFIRMA** — rechazada en #20 de la auditoría |
| "Denoising mejora la predictibilidad" | "Ruido" es categoría del modelo, no propiedad verificada | **NO SE AFIRMA** — rechazada en #35 |
| "La descomposición trend/seasonal es única" | Contradice explícitamente a Tsay | **NO SE AFIRMA** — rechazada en #27 |
| Tsay: *"for forecasting purposes, the choice of decomposition does not matter provided that the chosen one is a valid decomposition"* | **Fácil de sobre-interpretar** como "da igual qué modelo elijas" | **PREGUNTA ABIERTA**: Tsay no define "valid decomposition" en ese punto. Interpretación razonable [B]: se refiere a reparametrizaciones del mismo modelo observacional. Ver Sección 28.4 |
| Tsay: *"Filtering means to recover the state variable $\mu_t$"* | La palabra "recover" sugiere que hay algo preexistente que se recupera | **MATIZAR**: en el contexto de Tsay significa "estimar dentro del modelo". Fuera de contexto invita al error de #1 |
| Tsay: *"confirming that intraday high-frequency returns are subject to measurement errors"* | El verbo "confirming" es fuerte para un resultado de un activo y 340 días | **MATIZAR**: es consistente con la hipótesis de microestructura para esa serie; no la establece en general |
| "$e_t$ es ruido de microestructura" | Tsay lo interpreta así **sólo** para el ejemplo de volatilidad realizada | **MATIZAR**: en general, $e_t$ es sólo el residuo de la ecuación de observación bajo el modelo elegido |
| "Buen fit in-sample implica forecasting OOS" | Ninguna evaluación OOS en el capítulo | **NO SE AFIRMA** — rechazada en #36 |
| "Un state variable es un régimen" | Confusión categorial continuo/discreto | **NO SE AFIRMA** — rechazada en #24, #25, #26 |
| "Una barra de volumen cero es una barra faltante" | Es una precisión [B] sobre datos de futuros; Tsay no la trata | **MANTENER** como [B] |
| Referencia cruzada en Tsay: *"The transformation in Eq. (11.5) has several important implications"* (Sección 11.1.3) | El contexto indica que se refiere a la Eq. (11.15), no (11.5) | **OBSERVACIÓN MENOR**: aparente errata tipográfica del libro; no afecta el contenido. Puede también ser artefacto de la extracción de texto del PDF |
| Cifras de $\hat\sigma_e = 0.4803$ vs. $\hat\sigma_e = 0.480$ | Tsay reporta ambas (MLE directa y vía ARIMA) con precisión distinta | **MANTENER**: no es inconsistencia; son dos rutas de cálculo que coinciden aproximadamente, como el propio Tsay señala |
| Aplicaciones modernas de ML (LSTM, embeddings, feature stores) | Tsay no las desarrolla ni las menciona | **MARCADAS COMO [B]** en todos los casos donde aparecen |
| Cualquier extrapolación a MNQ o a futuros | Tsay no analiza ningún futuro en este capítulo | **MARCADAS COMO [B] o PREGUNTA ABIERTA** sin excepción |

---

*Fin del informe — Capítulo 11.*

**KNOWLEDGE ACQUIRED — NO DESIGN DECISIONS ADOPTED.**
