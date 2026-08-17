# Tsay, *Analysis of Financial Time Series* (3ª ed.) — Capítulo 3
## *Conditional Heteroscedastic Models*
### Estudio orientado a un sistema de Machine Learning para trading de futuros

**Fuente:** Ruey S. Tsay, *Analysis of Financial Time Series, Third Edition*, Wiley (2010), Capítulo 3, pp. 109–171. Omitido: apéndice de programas RATS y salidas de software.

**Convención de atribución** (idéntica a los informes de los Capítulos 1 y 2):

- **[A]** = afirmación, definición, resultado, ejemplo o demostración que proviene directamente de Tsay.
- **[B]** = interpretación, extensión o adaptación propia a futuros, trading cuantitativo o ML.
- **PREGUNTA ABIERTA** = cuestión que este capítulo no permite resolver.

**Criterio heredado y reforzado:**

> Una observación razonable no debe convertirse en regla normativa sin explicitar las condiciones bajo las cuales es válida.

**Criterio nuevo, adoptado tras la revisión del Capítulo 2:**

> Cada vez que un hallazgo empírico de Tsay se invoque para futuros, debe indicarse el instrumento, la frecuencia y el período sobre los que fue establecido.

**Nota sobre el estilo de este informe.** A diferencia de los dos anteriores, este documento está escrito para alguien que entiende Machine Learning y mercados pero **no** econometría. Cada concepto se explica antes de usarse, cada símbolo se define, y las ecuaciones aparecen sólo cuando ayudan. Si una explicación resulta densa, es un defecto del informe, no del lector.

---

# 1. Resumen ejecutivo

## 1.1 La idea central, en una frase

**Es muy difícil saber hacia dónde se moverá un mercado, pero es mucho más fácil saber cuánto se moverá.** El Capítulo 3 trata sobre cómo medir, modelar y pronosticar ese "cuánto".

## 1.2 De dónde venimos

El Capítulo 1 documentó dos hechos sobre los retornos financieros: tienen **colas pesadas** (los movimientos extremos ocurren mucho más seguido de lo que predice una campana de Gauss) y la volatilidad **se agrupa** (los períodos agitados vienen seguidos de más días agitados).

El Capítulo 2 afinó eso con una observación notable: si uno mide la memoria de los retornos con signo, casi no hay ninguna. Pero si mide la memoria de los retornos **en valor absoluto** —es decir, sin importar si subió o bajó, sólo cuánto se movió— la memoria dura cientos de períodos.

El Capítulo 3 toma esa observación y construye modelos sobre ella.

## 1.3 Los diez puntos que importan

**1. La volatilidad no se observa directamente** [A]. Tsay lo dice en la primera línea del capítulo: es una característica especial de la volatilidad. En un día de trading hay un solo retorno diario; con un solo número no se puede medir cuánta variabilidad había. Esto tiene una consecuencia práctica enorme, que Tsay señala explícitamente: **"la no observabilidad de la volatilidad hace difícil evaluar el desempeño predictivo de los modelos de heterocedasticidad condicional"**.

**2. Tsay lista cuatro características de la volatilidad** [A], observadas en retornos de activos: (i) hay **clusters** de volatilidad; (ii) la volatilidad evoluciona de forma **continua** —los saltos son raros—; (iii) la volatilidad **no diverge a infinito**, varía dentro de un rango fijo, lo que estadísticamente significa que suele ser estacionaria; (iv) la volatilidad **parece reaccionar de forma distinta** ante una subida grande que ante una caída grande, fenómeno llamado *leverage effect*.

**3. Un modelo de volatilidad se construye en dos piezas** [A]: una **ecuación de media** que predice hacia dónde, y una **ecuación de volatilidad** que predice cuánta incertidumbre hay alrededor de esa predicción. Son dos problemas distintos y el capítulo los mantiene separados en todo momento.

**4. La receta de construcción tiene cuatro pasos** [A]: modelar la media, testear si queda estructura en los residuos al cuadrado, especificar el modelo de volatilidad si la hay, y verificar. Y una observación práctica que resume el estado de la cuestión: **"para la mayoría de las series de retornos de activos, las correlaciones seriales son débiles, si es que existen. Por lo tanto, construir una ecuación de media equivale a restar la media muestral de los datos"** [A].

**5. ARCH fue la primera idea sistemática y no fue suficiente** [A]. La idea: la volatilidad de hoy depende del tamaño de los shocks recientes. Tsay enumera cuatro debilidades explícitas, entre ellas que trata igual a un shock positivo y a uno negativo, que necesita demasiados parámetros, y que **"tiende a sobrepredecir la volatilidad porque responde lentamente a shocks grandes y aislados"**.

**6. GARCH resuelve el problema de parsimonia** [A]. La volatilidad esperada de hoy = un nivel base + el efecto del shock de ayer + parte de la volatilidad que ya veníamos teniendo. Con tres números captura lo que ARCH necesitaba nueve o más para describir. Y Tsay establece la equivalencia clave: **un GARCH es un ARMA aplicado a la serie de residuos al cuadrado**, lo que conecta directamente con todo el aparato del Capítulo 2.

**7. La persistencia se mide con una sola suma** [A]. En un GARCH(1,1), el número $\alpha_1+\beta_1$ dice cuánto dura el efecto de un shock de volatilidad. En el ejemplo del S&P 500 mensual de Tsay, ese número es **0.9772** — muy cerca de 1, lo que significa que los shocks de volatilidad se disipan muy lentamente. Tsay observa que **"este fenómeno se observa comúnmente en la práctica"**.

**8. Evaluar un pronóstico de volatilidad es un problema genuino, no un detalle** [A]. Como no se observa la volatilidad verdadera, muchos investigadores comparan el pronóstico con el retorno al cuadrado observado, obtienen correlaciones bajas, y concluyen que el modelo es malo. Tsay dice que esa conclusión es incorrecta: el retorno al cuadrado es un estimador **consistente pero muy impreciso**, y **"estrictamente hablando, tal enfoque para evaluar el desempeño predictivo de modelos de volatilidad no es apropiado"**.

**9. La volatilidad variable puede, por sí sola, generar colas pesadas** [A]. Tsay demuestra que un ARCH(1) con shocks perfectamente normales puede generar retornos con exceso de kurtosis positivo. Por tanto, no es necesario que los shocks básicos tengan colas pesadas para que la distribución de los retornos presente colas pesadas: la variación temporal de la volatilidad puede generarlas dentro de este tipo de modelos. Esto no implica que toda la kurtosis observada en datos financieros provenga exclusivamente de la volatilidad variable; el propio capítulo permite que ésta coexista con shocks cuya distribución condicional también tenga colas pesadas.

**10. Existen alternativas a GARCH que no requieren estimar nada** [A]. Si se dispone de datos intradía, la **volatilidad realizada** —la suma de los retornos intradía al cuadrado— estima directamente cuánto se movió el mercado. Y si sólo se tienen barras con apertura, máximo, mínimo y cierre, hay estimadores (Parkinson, Rogers–Satchell, Yang–Zhang) que usan el rango del día y son entre 5 y 8 veces más eficientes que usar sólo el cierre, bajo los supuestos del modelo que analizan Garman y Klass.

## 1.4 Lo que este capítulo NO dice

- **No dice que la volatilidad predecible produzca dinero.** Saber que mañana habrá movimiento grande no dice de qué lado.
- **No dice que exista una relación estable entre riesgo y retorno.** Tsay prueba un modelo (GARCH-M) que lo permitiría y encuentra que el parámetro **no es estadísticamente significativo** en su ejemplo.
- **No dice nada sobre futuros.** Todos sus ejemplos son acciones estadounidenses, índices accionarios y un tipo de cambio.
- **No dice nada sobre costos de transacción, validación out-of-sample ni sobreajuste.**
- **No resuelve si un modelo más complejo pronostica mejor.** Sobre volatilidad estocástica dice literalmente que las mejoras en ajuste no se tradujeron claramente en mejores pronósticos: **"sus contribuciones a los pronósticos de volatilidad fuera de muestra recibieron resultados mixtos"** [A].

## 1.5 Qué deberíamos poder explicar al terminar

Que la volatilidad es una cantidad que no vemos pero que podemos estimar; que cambia con el tiempo de forma parcialmente predecible; que esa predictibilidad es un fenómeno distinto de la predictibilidad direccional; que hay al menos tres familias de métodos para estimarla (modelos tipo GARCH, volatilidad estocástica, y medidas directas como la volatilidad realizada o los estimadores OHLC); y que **todo lo anterior es conocimiento sobre riesgo, no automáticamente sobre rentabilidad**.

---

# 2. Conceptos básicos antes de empezar

Esta sección define todo el vocabulario que se usará después. Ningún término aparecerá más adelante sin haber sido definido aquí o en el punto donde se introduce.

## 2.1 Retorno, varianza y volatilidad no son lo mismo

**Retorno ($r_t$).** Cuánto cambió el precio de un período al siguiente, en términos relativos. Es un número **con signo**: +1.2% significa que subió; −1.2% significa que bajó. Se observa directamente: se calcula a partir de dos precios.

**Varianza.** Una medida de cuánto se dispersan los retornos alrededor de su promedio. Se calcula elevando al cuadrado las desviaciones respecto de la media y promediándolas. Al elevar al cuadrado, **se pierde el signo**: sólo queda la magnitud.

**Volatilidad ($\sigma_t$).** La raíz cuadrada de la varianza. Se usa porque está en las mismas unidades que el retorno: si los retornos están en porcentaje, la volatilidad también. Una volatilidad diaria de 1% significa, hablando informalmente, que los movimientos típicos del día son del orden de 1%.

**La diferencia esencial:** el retorno responde *"¿hacia dónde y cuánto se movió?"*. La volatilidad responde *"¿de qué tamaño son los movimientos típicos en este momento?"*.

### Ejemplo: dos períodos con la misma dirección impredecible

**Período A:** +0.1%, −0.2%, +0.1%, −0.1%
**Período B:** +2%, −3%, +1.5%, −2%

En ambos períodos, si uno intentara adivinar el signo del próximo movimiento, estaría prácticamente adivinando: suben y bajan alternadamente sin patrón obvio. **La dirección parece igual de impredecible en los dos.**

Pero los dos períodos son radicalmente distintos. En el A, los movimientos son de una o dos décimas de punto porcentual. En el B, son de dos o tres puntos enteros: **entre 15 y 20 veces más grandes**.

Si uno tuviera que apostar sobre *el tamaño* del próximo movimiento, en el período A diría "pequeño" y en el B diría "grande" — y probablemente acertaría. Eso es exactamente lo que significa que la volatilidad sea predecible aunque la dirección no lo sea.

**Consecuencia práctica inmediata:** una posición del mismo tamaño produce en el período B un P&L unas 20 veces más variable que en el período A. Si el sistema no distingue entre ambos regímenes, está tomando cantidades de riesgo radicalmente distintas sin saberlo.

## 2.2 Por qué la volatilidad no se observa

**[A] Qué dice Tsay.** Textualmente: *"Una característica especial de la volatilidad de una acción es que no es directamente observable."* Y da el motivo con un ejemplo concreto: considerando los log-retornos diarios de IBM, **"la volatilidad diaria no es directamente observable a partir de los datos de retorno porque hay una sola observación en un día de trading"**.

**Explicación intuitiva [B].** Imaginemos que queremos medir cuánto varía la altura de las personas de una ciudad. Si medimos a mil personas, podemos calcular la dispersión. Si medimos a **una sola persona**, tenemos su altura pero no tenemos ninguna medida de dispersión: con un solo dato no hay variabilidad que medir.

Con la volatilidad diaria pasa lo mismo. El día de hoy produjo un solo retorno diario. Ese número nos dice cuánto se movió el precio de cierre a cierre, pero no nos dice si ese movimiento fue un caso típico de un día tranquilo o un caso moderado de un día muy agitado.

**[A] Cómo se puede rodear el problema.** Tsay menciona dos vías:
1. **Datos intradía.** *"Si hay datos intradía disponibles, como retornos de 10 minutos, entonces uno puede estimar la volatilidad diaria."* Con muchas observaciones dentro del día, sí hay dispersión que medir. Pero advierte: *"la precisión de tal estimación merece un estudio cuidadoso"*, porque la volatilidad de una acción consiste en volatilidad intradía **y volatilidad overnight** (la variación entre días de trading), y **"los retornos intradía de alta frecuencia contienen sólo información muy limitada sobre la volatilidad overnight"**.
2. **Volatilidad implícita.** [A] Si se acepta que los precios de opciones se rigen por un modelo como Black–Scholes, se puede despejar la desviación estándar implícita en el precio de la opción. Tsay señala la crítica habitual: el método **depende de un modelo específico** basado en supuestos que pueden no cumplirse. Y añade un dato empírico: *"la experiencia muestra que la volatilidad implícita de un retorno de un activo tiende a ser mayor que la obtenida usando un modelo de volatilidad tipo GARCH"*, posiblemente por una prima de riesgo por volatilidad. Menciona que **el VIX del CBOE es una volatilidad implícita**.

**Qué NO significa esto [B].** No significa que la volatilidad sea una construcción teórica sin realidad. Significa que es una **cantidad latente**: existe, afecta a lo que observamos, pero sólo accedemos a ella a través de estimaciones. Es la misma situación que la temperatura medida con un termómetro impreciso: la temperatura existe, pero cada lectura tiene error.

## 2.3 Varianza incondicional vs varianza condicional

Ésta es probablemente la distinción más importante de todo el capítulo, y conviene explicarla despacio.

**Varianza incondicional.** Es la variabilidad *promedio de toda la historia*. Un solo número que resume cuánto se mueve el instrumento en general. Si tomamos diez años de retornos diarios de un futuro y calculamos su desviación estándar, obtenemos algo cercano a la volatilidad incondicional.

**Varianza condicional ($\sigma_t^2$).** Es la variabilidad **esperada para el próximo período específico, dado todo lo que sabemos hasta ahora**. No es un número fijo: cambia cada día. Después de una semana tranquila será baja; después de un crash será alta.

**Notación.** Se escribe:

$$\sigma_t^2 = \text{Var}(r_t \mid \mathcal{F}_{t-1})$$

Desarmemos esa expresión símbolo por símbolo:

- **$\text{Var}(\cdot)$** = "la varianza de".
- **$r_t$** = el retorno en el período $t$ (el que todavía no ocurrió).
- **La barra vertical $\mid$** se lee **"dado que sabemos"** o **"condicional a"**.
- **$\mathcal{F}_{t-1}$** (efe caligráfica sub t menos uno) = **el conjunto de toda la información disponible hasta el período anterior**. Tsay lo define así explícitamente [A]: *"$\mathcal{F}_{t-1}$ denota el conjunto de información disponible en el tiempo $t-1$"*, y añade que *"típicamente, $\mathcal{F}_{t-1}$ consiste en todas las funciones lineales de los retornos pasados"*.
- **$\sigma_t^2$** (sigma sub t al cuadrado) = el resultado: la varianza esperada para $t$.

Leída completa: **"la varianza condicional en $t$ es la variabilidad que esperamos para el retorno del período $t$, dado todo lo que sabemos hasta el final del período $t-1$."**

**Analogía sencilla [B].** La varianza incondicional es como decir "en esta ciudad llueve 800 mm al año". La varianza condicional es como decir "hoy, viendo el cielo, la probabilidad de lluvia es alta". El promedio anual no cambia; el pronóstico de hoy sí.

**Por qué esto importa tanto [B].** Todo el Capítulo 3 trata sobre **modelar cómo evoluciona $\sigma_t^2$**. Tsay lo dice explícitamente [A]: *"Los modelos de heterocedasticidad condicional de este capítulo se ocupan de la evolución de $\sigma_t^2$. La manera en que $\sigma_t^2$ evoluciona en el tiempo distingue un modelo de volatilidad de otro."*

**La palabra "heterocedasticidad".** Suena complicada y significa algo simple: *hetero* = distinto, *cedasticidad* = dispersión. **Heterocedasticidad = la dispersión no es siempre la misma.** *Heterocedasticidad condicional* = la dispersión esperada cambia según la información disponible. Lo contrario, homocedasticidad, sería que la varianza es siempre la misma.

## 2.4 Residuo, shock e innovación

Estos tres términos se usan casi como sinónimos y conviene fijarlos.

**Punto de partida.** Descomponemos el retorno observado en dos partes:

$$r_t = \mu_t + a_t$$

- **$r_t$** = el retorno que efectivamente ocurrió.
- **$\mu_t$** (mu sub t) = la parte **esperada**: lo que nuestro modelo predecía en promedio, dada la información previa. Técnicamente $\mu_t = E(r_t\mid\mathcal{F}_{t-1})$, donde $E(\cdot)$ significa "el valor esperado de" o, informalmente, "el promedio de".
- **$a_t$** = **la sorpresa**: la parte que el modelo de media no explicó.

**Cómo se llama $a_t$ [A].** Tsay es explícito: *"A lo largo del libro, $a_t$ se denomina el **shock** o **innovación** del retorno de un activo en el tiempo $t$."* En la práctica también se le llama **residuo**, porque es lo que queda después de restar la predicción:

$$a_t = r_t - \hat\mu_t$$

donde el sombrero en $\hat\mu_t$ indica que es una *estimación* de $\mu_t$, no el valor verdadero.

**Ejemplo numérico [B].** Supongamos que nuestro modelo de media predice que el retorno de mañana será +0.02%, y mañana el retorno resulta ser +1.5%. Entonces:
- $\hat\mu_t = 0.02\%$ (lo que esperábamos)
- $r_t = 1.5\%$ (lo que ocurrió)
- $a_t = 1.5\% - 0.02\% = 1.48\%$ (la sorpresa)

**Por qué "innovación".** Porque es la **información nueva** que llegó al mercado en ese período y que no estaba contenida en lo que ya sabíamos. En el Capítulo 2 vimos exactamente el mismo concepto: $a_t$ era el término de white noise de los modelos AR y MA.

**El punto clave [A].** Como $\mu_t$ es conocido dado el pasado, restar una constante conocida no cambia la varianza. Por eso Tsay establece:

$$\sigma_t^2 = \text{Var}(r_t\mid\mathcal{F}_{t-1}) = \text{Var}(a_t\mid\mathcal{F}_{t-1})$$

**Traducción:** modelar la incertidumbre del retorno es lo mismo que modelar la incertidumbre del shock. Por eso todos los modelos de volatilidad se escriben en términos de $a_t$ y no de $r_t$.

## 2.5 La ecuación que define un shock: $a_t = \sigma_t \epsilon_t$

Ésta es la ecuación central del capítulo. Vale la pena entenderla bien.

$$a_t = \sigma_t\,\epsilon_t$$

- **$a_t$** = el shock observado (la sorpresa).
- **$\sigma_t$** = la volatilidad condicional: el "tamaño típico" de las sorpresas en este momento.
- **$\epsilon_t$** (épsilon sub t) = el **shock estandarizado**. Es una variable aleatoria con **media 0 y varianza 1**, independiente e idénticamente distribuida a lo largo del tiempo. Tsay lo especifica así [A]: *"$\{\epsilon_t\}$ es una secuencia de variables aleatorias independientes e idénticamente distribuidas (iid) con media cero y varianza 1"*.

**Interpretación intuitiva [B].** La ecuación descompone cada sorpresa en dos factores independientes:

- **$\sigma_t$ es la "escala del día"**: cuán grandes son las sorpresas en este entorno. Se conoce (o se estima) a partir del pasado.
- **$\epsilon_t$ es el "dado que se tira"**: la componente puramente aleatoria, que no depende del pasado. Da el signo y la desviación relativa.

**Analogía [B].** Pensemos en el tamaño de las olas en una playa. $\sigma_t$ es el estado del mar de hoy: calmo, moderado o agitado. Se puede saber mirando el clima reciente. $\epsilon_t$ es cuán grande resulta esta ola en particular respecto de las olas típicas de hoy. Ninguna cantidad de información sobre el clima permite predecir si la próxima ola será algo mayor o menor que el promedio de hoy — pero sí permite saber si estamos en un día de olas de medio metro o de tres metros.

**Ejemplo numérico [B].**
- Día tranquilo: $\sigma_t = 0.3\%$. Si $\epsilon_t = 1.5$, entonces $a_t = 0.45\%$.
- Día agitado: $\sigma_t = 2.5\%$. Con el **mismo** $\epsilon_t = 1.5$, entonces $a_t = 3.75\%$.

El mismo "dado" produce sorpresas ocho veces mayores según la escala del día. Esto es, en una línea, todo el fenómeno del volatility clustering.

**Qué NO significa esta ecuación [B].** No significa que la volatilidad *cause* los movimientos. Es una descomposición contable de una variable aleatoria en escala × forma. Tampoco significa que $\sigma_t$ sea observable: en la práctica sólo se observa $a_t$, y hay que inferir $\sigma_t$.

**Qué asume [A].** Que $\epsilon_t$ es iid — es decir, que **toda la dependencia temporal está en $\sigma_t$**, y ninguna en $\epsilon_t$. Ésta es una restricción real del marco, no una formalidad. **PREGUNTA ABIERTA:** ¿es cierto en futuros que, una vez descontada la volatilidad, lo que queda es verdaderamente iid? El Capítulo 4 (no linealidad) es donde eso se examina.

## 2.6 Volatilidad realizada: la idea en una frase

Se desarrolla en detalle en la sección 7 de este informe, pero conviene tener la intuición desde ahora.

**El problema:** no observamos la volatilidad diaria porque el día produce un solo retorno.

**La idea:** si tenemos muchos retornos **dentro** del día —por ejemplo, uno cada 5 minutos— sí tenemos dispersión que medir. Sumando esos retornos intradía al cuadrado obtenemos una estimación directa de cuánto se movió realmente el mercado ese día.

$$RV_t = \sum_{i=1}^{n} r_{t,i}^2$$

donde $r_{t,i}$ es el $i$-ésimo retorno intradía del día $t$, y $n$ es cuántos retornos intradía hay. Tsay la define exactamente así [A] y la llama **realized volatility**.

**Por qué es distinto de GARCH [B].** GARCH *infiere* la volatilidad a partir de la dinámica de los retornos, con un modelo y parámetros estimados. La volatilidad realizada la *mide* directamente, sin modelo. Son enfoques conceptualmente distintos, y no compiten necesariamente: uno puede medir con volatilidad realizada y luego modelar la dinámica de esa medida.

## 2.7 Dirección vs volatilidad: los cuatro escenarios

Ésta es la distinción que da sentido a todo el capítulo. Reformulemos las dos preguntas en lenguaje llano:

$$E(r_{t+1}\mid X_t) \quad\longrightarrow\quad \textit{¿hacia dónde y cuánto esperamos que se mueva, en promedio?}$$

$$\text{Var}(r_{t+1}\mid X_t) \quad\longrightarrow\quad \textit{¿cuánta incertidumbre hay alrededor de esa expectativa?}$$

donde $X_t$ es la información disponible al momento de decidir.

Estas dos preguntas pueden tener respuestas independientes. Los cuatro escenarios posibles:

| | **Volatilidad predecible** | **Volatilidad impredecible** |
|---|---|---|
| **Dirección predecible** | **Escenario 1** | **Escenario 2** |
| **Dirección impredecible** | **Escenario 3** | **Escenario 4** |

**Escenario 1 — dirección y volatilidad predecibles.** El caso ideal y el menos probable. Sabemos hacia dónde y con cuánto riesgo. Permitiría tanto decidir la posición como dimensionarla.

**Escenario 2 — dirección predecible, volatilidad impredecible.** Sabemos hacia dónde, pero no cuánto riesgo estamos tomando. Se puede operar pero es difícil dimensionar bien; el riesgo de la cartera será errático.

**Escenario 3 — dirección impredecible, volatilidad predecible.** **Éste es el escenario que la evidencia de los Capítulos 2 y 3 sugiere que es el más relevante en mercados financieros.** No sabemos si subirá o bajará, pero sabemos bastante bien cuánto se moverá.

Vale la pena detenerse aquí, porque es contraintuitivo y es el punto que el capítulo entero ilumina.

**¿De qué sirve el escenario 3?** [B] — Tsay no discute utilidad para trading, así que todo lo siguiente es interpretación propia:

- **Sirve para dimensionar.** Si sabemos que mañana la volatilidad será el doble de lo normal, podemos operar la mitad del tamaño y mantener el riesgo constante. Esto no genera señal, pero **estabiliza el riesgo**, que es un objetivo legítimo por sí mismo.
- **Sirve para saber cuándo no operar.** Si un sistema tiene un edge pequeño, hay regímenes donde el ruido lo sepulta.
- **Sirve para calibrar probabilidades.** Un modelo que dice "70% de probabilidad de subir" está diciendo algo distinto en un régimen tranquilo que en uno agitado, si el umbral se define en unidades absolutas.
- **Sirve para no engañarse.** Un backtest que ganó dinero durante un régimen de baja volatilidad y perdió en uno alto no tiene un problema de señal; tiene un problema de riesgo.
- **Sirve, potencialmente, para hacer más aprendible el problema direccional.** Si dividimos los retornos por una estimación de volatilidad, la variable resultante tiene una distribución más estable. Si eso ayuda o no a un modelo es una **PREGUNTA ABIERTA** empírica, no una conclusión.

**Lo que el escenario 3 NO garantiza [B]:** no garantiza rentabilidad. Se puede tener un pronóstico de volatilidad excelente y no tener ninguna estrategia rentable, porque **la volatilidad no dice de qué lado ponerse**. Convertir conocimiento sobre riesgo en rentabilidad requiere un mecanismo adicional — dimensionamiento, opciones, o una relación entre volatilidad y retorno esperado — y esa última posibilidad es justamente la que el modelo GARCH-M explora y que en el ejemplo de Tsay resulta **no significativa**.

**Escenario 4 — ambas impredecibles.** El mercado es un random walk homocedástico. La evidencia de Tsay sugiere que este escenario no describe bien los datos que él examina: el efecto ARCH aparece con p-valores prácticamente nulos.

---

# 3. Análisis sección por sección

## 3.1 Sección 3.1 — Characteristics of Volatility

### Qué problema intenta resolver

Antes de construir cualquier modelo, hay que saber **qué se está tratando de modelar**. Esta sección describe cómo se comporta la volatilidad en los datos, para que los modelos posteriores puedan juzgarse por si capturan o no ese comportamiento.

### [A] Qué dice Tsay

Tras establecer que la volatilidad no es directamente observable, enuncia **cuatro características que se ven comúnmente en retornos de activos**:

1. **Existen clusters de volatilidad**, es decir, *"la volatilidad puede ser alta durante ciertos períodos y baja durante otros"*.
2. **La volatilidad evoluciona de manera continua** — *"los saltos de volatilidad son raros"*.
3. **La volatilidad no diverge a infinito**: *"varía dentro de un rango fijo. Estadísticamente hablando, esto significa que la volatilidad es a menudo estacionaria"*.
4. **La volatilidad parece reaccionar de forma distinta ante una gran subida de precio que ante una gran caída**, fenómeno que se conoce como **leverage effect**.

Y añade [A]: *"Estas propiedades juegan un papel importante en el desarrollo de modelos de volatilidad. Algunos modelos de volatilidad fueron propuestos específicamente para corregir las debilidades de los existentes por su incapacidad de capturar las características mencionadas. Por ejemplo, el modelo EGARCH fue desarrollado para capturar la asimetría en volatilidad inducida por retornos 'positivos' y 'negativos' grandes."*

### Explicación simple de cada característica

**Clusters.** Los días agitados vienen en rachas y los tranquilos también. No es que la volatilidad alterne al azar entre alta y baja; se queda en un régimen durante un tiempo.

**Continuidad.** La volatilidad cambia gradualmente. Puede pasar de baja a alta en pocos días, pero no salta instantáneamente de un nivel a otro completamente distinto.

**No diverge.** La volatilidad puede triplicarse en una crisis, pero luego vuelve. No crece indefinidamente. En el lenguaje del Capítulo 2, se comporta como una serie estacionaria: fluctúa alrededor de un nivel.

**Asimetría (leverage effect).** Una caída grande parece elevar la volatilidad futura más que una subida grande de la misma magnitud.

### ¿Por qué un día muy volátil aumenta la probabilidad de que los siguientes también lo sean?

Ésta es la pregunta central de la sección. **Tsay describe el fenómeno pero no ofrece una explicación causal** — de hecho, más adelante lo señala como una debilidad de los modelos ARCH: *"no provee ninguna intuición nueva para entender la fuente de las variaciones de una serie temporal financiera. Simplemente provee una forma mecánica de describir el comportamiento de la varianza condicional. No da ninguna indicación sobre qué causa que ese comportamiento ocurra"* [A].

Lo siguiente es **[B]**, mecanismos plausibles mencionados en la literatura financiera general, no en este capítulo:

- **La información llega en rachas.** Una crisis, una decisión de política monetaria o un shock geopolítico no se resuelve en un día: genera un flujo continuado de noticias durante semanas.
- **La incertidumbre tarda en resolverse.** Mientras los participantes no saben qué va a pasar, discrepan más sobre el precio justo, y esa discrepancia se traduce en movimientos mayores.
- **Reacciones en cadena mecánicas.** Movimientos grandes disparan llamados de margen, liquidaciones forzadas y ajustes de riesgo, que a su vez generan más movimiento.
- **La liquidez se retira cuando hay volatilidad**, lo que amplifica el impacto de cada orden.

**Importante:** ninguno de estos mecanismos es necesario para *usar* los modelos del capítulo. Los modelos describen el fenómeno sin explicarlo. Pero conviene tener presente que **son descripciones, no teorías causales** — y eso limita cuánta confianza podemos tener en que el patrón se mantenga cuando el mercado cambie estructuralmente.

### ¿Hechos estilizados o propiedades teóricas?

| Característica | ¿Qué es? |
|---|---|
| Clusters | **Hecho estilizado observado** en los datos que Tsay examina (acciones e índices de EE.UU., un tipo de cambio) |
| Evolución continua | **Hecho estilizado observado**, con la salvedad de que "los saltos son raros" es una afirmación empírica que puede fallar en instrumentos o eventos concretos |
| No diverge / estacionaria | **Observación empírica** presentada con lenguaje estadístico ("a menudo estacionaria"). Nótese el "a menudo": no es una ley |
| Leverage effect | **Hecho estilizado observado en acciones e índices accionarios.** Tsay dice "parece reaccionar de forma distinta" — lenguaje deliberadamente prudente |

### [B] Adaptación a futuros — con cautela

**Lo que probablemente se traslade.** El clustering de volatilidad es un fenómeno tan general en activos financieros que sería sorprendente no encontrarlo en futuros. Pero **eso es una expectativa, no un resultado**: los ejemplos de Tsay son acciones estadounidenses (IBM, Intel), índices accionarios (S&P 500, CRSP) y un tipo de cambio (marco alemán/dólar) — ninguno es un futuro.

**Lo que NO debe trasladarse automáticamente: el leverage effect.** Ésta es la advertencia más importante de la sección para nuestro proyecto. El nombre mismo del efecto viene de una explicación específica de acciones: cuando el precio de una acción cae, la empresa queda más apalancada (la deuda pesa más respecto del capital), y por tanto más riesgosa. Ese mecanismo **no existe** en:

- **Futuros de commodities.** Un shock de oferta puede elevar la volatilidad al alza tanto o más que a la baja. En energía y agrícolas, las crisis suelen ser de escasez, es decir, movimientos **al alza** violentos.
- **Futuros de divisas.** ¿Qué es una "caída" en EUR/USD? Depende de qué lado del par se mire. La asimetría, si existe, no puede tener el mismo origen.
- **Futuros de tasas de interés.** La relación entre dirección y volatilidad tiene que ver con el nivel de tasas y con la política monetaria, no con apalancamiento corporativo.
- **Futuros sobre índices accionarios.** Aquí es más plausible que se traslade, porque el subyacente es un conjunto de acciones. Pero incluso así, el mecanismo puede ser distinto (cobertura de opciones, flujos de riesgo) más que apalancamiento contable.

**Formulación correcta [B]:** *si existe asimetría en la respuesta de la volatilidad a shocks de distinto signo en un futuro dado, es una cuestión empírica a medir por instrumento; su magnitud, su signo y su estabilidad no pueden deducirse de la evidencia sobre acciones.*

### [B] Implicancias potenciales para ML

- Si el clustering existe, entonces **el orden temporal de las observaciones contiene información**, y cualquier procedimiento que lo destruya (barajar los datos) destruye esa información.
- Si la volatilidad es aproximadamente estacionaria pero varía mucho, entonces **la varianza del target no es constante**, lo que afecta a cualquier función de pérdida que promedie errores al cuadrado.
- Si la asimetría existe, entonces features que sólo miren magnitudes (como $|r_t|$) están descartando información potencialmente útil.

### Qué NO podemos concluir

- Que estas cuatro características se cumplan en cualquier futuro, en cualquier frecuencia y en cualquier período.
- Que exista una explicación causal establecida para el clustering.
- Que la asimetría documentada en acciones tenga el mismo signo o magnitud en otros mercados.

### Preguntas abiertas

- ¿Qué causa realmente la persistencia de la volatilidad? Tsay dice explícitamente en §3.6 que *"la causa real de la persistencia en volatilidad merece una investigación cuidadosa"* [A]. → Parcialmente **Caps. 11–12** (cambios de régimen).
- ¿Hay saltos genuinos en volatilidad, distintos de la evolución continua? → **Cap. 7** (extremos) y literatura de jumps.

---

## 3.2 Sección 3.2 — Structure of a Model

### Qué problema intenta resolver

Necesitamos una forma de escribir un modelo que prediga **dos cosas a la vez**: hacia dónde va el retorno y cuánta incertidumbre hay. Esta sección establece esa estructura de dos piezas.

### [A] Qué dice Tsay

**El punto de partida empírico.** *"La idea básica detrás del estudio de volatilidad es que la serie $\{r_t\}$ está o bien serialmente no correlacionada, o bien con correlaciones seriales menores de orden bajo, **pero es una serie dependiente**."*

Lo ilustra con los log-retornos mensuales de Intel Corporation (enero 1973 – diciembre 2008), Figura 3.2:
- La ACF de los retornos **no sugiere correlaciones seriales significativas**, salvo una menor en el lag 7.
- La ACF de los **retornos absolutos** $|r_t|$ y la de los **retornos al cuadrado** $r_t^2$ *"sugieren claramente que los log-retornos mensuales no son serialmente independientes"*.
- Conclusión de Tsay: *"combinando los tres gráficos, parece que los log-retornos son en efecto serialmente **no correlacionados pero dependientes**. Los modelos de volatilidad intentan capturar tal dependencia en la serie de retornos."*

**Las dos cantidades centrales** (Ec. 3.2):

$$\mu_t = E(r_t\mid\mathcal{F}_{t-1}),\qquad \sigma_t^2=\text{Var}(r_t\mid\mathcal{F}_{t-1})=E[(r_t-\mu_t)^2\mid\mathcal{F}_{t-1}]$$

**La forma de la ecuación de media** (Ec. 3.3). Tsay dice que *"como muestran los ejemplos empíricos del Capítulo 2 y la Figura 3.2, la dependencia serial de una serie de retornos de acciones es débil si es que existe. Por lo tanto, la ecuación para $\mu_t$ debería ser **simple**"*. Propone un ARMA con posibles variables explicativas:

$$r_t=\mu_t+a_t,\qquad \mu_t=\sum_{i=1}^{p}\phi_i y_{t-i}-\sum_{i=1}^{q}\theta_i a_{t-i},\qquad y_t=r_t-\phi_0-\sum_{i=1}^{k}\beta_i x_{it}$$

donde los $x_{it}$ son **variables explicativas**. Tsay señala que son flexibles: *"por ejemplo, puede usarse una variable dummy para los lunes para estudiar el efecto del fin de semana sobre los retornos diarios"*, y que en el CAPM la ecuación de media sería $r_t=\phi_0+\beta r_{m,t}+a_t$ con $r_{m,t}$ el retorno del mercado. Observa además que *"el orden $(p,q)$ de un modelo ARMA puede depender de la frecuencia de la serie de retornos"*: los retornos diarios de un índice suelen mostrar correlaciones seriales menores, mientras que los mensuales pueden no contener ninguna.

**La ecuación clave** (Ec. 3.4):

$$\sigma_t^2=\text{Var}(r_t\mid\mathcal{F}_{t-1})=\text{Var}(a_t\mid\mathcal{F}_{t-1})$$

**La nomenclatura** [A]: *"El modelo para $\mu_t$ en la Ec. (3.3) se denomina la **ecuación de media** para $r_t$ y el modelo para $\sigma_t^2$ es la **ecuación de volatilidad** para $r_t$. Por lo tanto, modelar la heterocedasticidad condicional equivale a **añadir una ecuación dinámica**, que gobierna la evolución temporal de la varianza condicional del retorno del activo, **a un modelo de series temporales**."*

**Dos categorías de modelos** [A]: *"Los modelos de heterocedasticidad condicional pueden clasificarse en dos categorías generales. Los de la primera categoría usan una **función exacta** para gobernar la evolución de $\sigma_t^2$, mientras que los de la segunda usan una **ecuación estocástica** para describir $\sigma_t^2$. El modelo GARCH pertenece a la primera categoría, mientras que el modelo de volatilidad estocástica está en la segunda."*

### Explicación simple

Un modelo completo tiene dos partes que trabajan juntas:

**Parte 1 — la ecuación de media.** Responde: *dado lo que sé, ¿cuál es mi mejor apuesta sobre el retorno de mañana?* En finanzas, según lo que muestran los Capítulos 2 y 3, la respuesta suele ser "casi cero" y la ecuación resulta muy simple.

**Parte 2 — la ecuación de volatilidad.** Responde: *¿cuánto puedo equivocarme en esa apuesta?* Aquí sí hay estructura rica.

Un modelo con las dos partes emite, para cada período, un par de números: **$(\hat\mu_t, \hat\sigma_t)$** — dónde creo que caerá el retorno y cuán ancha es mi incertidumbre.

**"No correlacionado pero dependiente" [B].** Esta frase de Tsay es la definición precisa de lo que veníamos persiguiendo desde el Capítulo 2. *No correlacionado* significa: no hay relación lineal entre el retorno de hoy y el de ayer. *Dependiente* significa: el pasado sí contiene información sobre el presente — sólo que no sobre su dirección, sino sobre su escala. Las dos cosas son perfectamente compatibles, y la Figura 3.2 de Tsay lo muestra en un solo conjunto de gráficos.

### [B] Adaptación a futuros

- La observación de que el orden ARMA de la ecuación de media **depende de la frecuencia** [A] es directamente relevante: la ecuación de media apropiada para un futuro en barras de 1 minuto puede no ser la misma que en barras diarias. Debe examinarse por frecuencia.
- Las **variables explicativas** en la ecuación de media están contempladas explícitamente por Tsay [A] con el ejemplo de la dummy de lunes. Esto abre la puerta formal a variables de calendario y de sesión, aunque el capítulo no las estudia en futuros.
- La estructura de dos ecuaciones se traslada sin cambios conceptuales a cualquier instrumento. Lo que **no** se traslada automáticamente es cuál será la forma concreta de cada ecuación.

### [B] Implicancias potenciales para ML — planteadas como preguntas

La estructura de dos ecuaciones sugiere naturalmente una pregunta de diseño: **¿conviene que un sistema de ML tenga dos componentes, uno para la media y otro para la volatilidad?**

**No se concluye que deban ser dos modelos separados.** Hay al menos tres arquitecturas conceptualmente distintas que respetan esta estructura, y elegir entre ellas requiere evidencia que todavía no tenemos:

1. **Dos modelos independientes**: uno predice $\hat\mu_t$, otro predice $\hat\sigma_t$.
2. **Un modelo con dos salidas**: la misma red o el mismo estimador emite el par $(\hat\mu_t,\hat\sigma_t)$, compartiendo representaciones internas.
3. **Un modelo de media con la volatilidad usada externamente**: por ejemplo, la volatilidad no se predice sino que se estima con un método directo y se usa para normalizar o dimensionar.

Cada una tiene ventajas distintas y ninguna está justificada por este capítulo. **PREGUNTA ABIERTA.**

Lo que **sí** queda establecido conceptualmente es que **son dos preguntas distintas**, y que un sistema que sólo responde la primera está ignorando la parte del problema donde la evidencia sugiere que hay más estructura.

### Qué NO podemos concluir

- Que la ecuación de media deba ser siempre simple. Tsay lo dice para series de retornos de acciones a frecuencias bajas [A]. Para futuros a otras frecuencias es una hipótesis.
- Que dos modelos separados sean mejores que uno conjunto.
- Que la descomposición en media y volatilidad agote la estructura. Puede haber dependencia en momentos superiores (asimetría condicional, kurtosis condicional) que esta estructura no captura. → **PREGUNTA ABIERTA.**

---

## 3.3 Sección 3.3 — Model Building

### Qué problema intenta resolver

Da un procedimiento ordenado para pasar de datos crudos a un modelo de volatilidad verificado.

### [A] Los cuatro pasos, textuales

> Construir un modelo de volatilidad para una serie de retornos consiste en cuatro pasos:
> 1. **Especificar una ecuación de media** testeando la dependencia serial en los datos y, si es necesario, construyendo un modelo econométrico (p. ej., ARMA) para la serie de retornos **para remover cualquier dependencia lineal**.
> 2. **Usar los residuos de la ecuación de media para testear efectos ARCH.**
> 3. **Especificar un modelo de volatilidad** si los efectos ARCH son estadísticamente significativos, y realizar una **estimación conjunta** de las ecuaciones de media y volatilidad.
> 4. **Verificar cuidadosamente el modelo ajustado** y refinarlo si es necesario.

Y añade la observación práctica ya citada [A]: *"Para la mayoría de las series de retornos de activos, las correlaciones seriales son débiles, si es que existen. Así, construir una ecuación de media equivale a remover la media muestral de los datos si la media muestral es significativamente distinta de cero. Para algunas series de retornos diarios, podría necesitarse un modelo AR simple. En algunos casos, la ecuación de media puede emplear variables explicativas como una variable indicadora para efectos de fin de semana o de enero."*

### La secuencia, visualizada

$$\text{Datos} \rightarrow \text{Modelo de media} \rightarrow \text{Residuos } a_t \rightarrow \text{Buscar estructura en } a_t^2 \rightarrow \text{Modelo de volatilidad} \rightarrow \text{Diagnóstico}$$

### ¿Por qué primero la media?

**Explicación simple [B].** Porque queremos que $a_t$ sea **sorpresa pura**. Si no removemos la parte predecible del retorno, entonces $a_t$ contiene dos cosas mezcladas: la sorpresa genuina y la parte que sí era predecible pero que no modelamos. Al elevar $a_t$ al cuadrado, esa contaminación se propaga a la ecuación de volatilidad y puede hacernos creer que hay estructura de volatilidad donde en realidad hay estructura de media mal capturada.

**Analogía [B].** Es como medir el ruido de un motor. Primero hay que restar el sonido de fondo conocido (la media); lo que queda es el ruido que queremos caracterizar. Si no restamos el fondo, confundiremos el volumen del fondo con el volumen del ruido.

**La otra cara [B].** El paso 3 dice "estimación **conjunta**" [A], y no es un detalle: estimar primero la media y luego la volatilidad por separado da resultados aproximados. Tsay incluso dedica una subsección (3.5.3) a un "método de estimación en dos pasos" precisamente porque estimar por separado es una aproximación, y observa que *"tales estimaciones son obviamente aproximaciones a los parámetros verdaderos y sus propiedades estadísticas no han sido investigadas rigurosamente"*, aunque *"la experiencia limitada muestra que este enfoque simple a menudo provee buenas aproximaciones, especialmente cuando el tamaño de muestra es moderado o grande"* [A].

### ¿Por qué elevar los residuos al cuadrado?

**El problema.** Queremos medir si los movimientos **grandes** vienen seguidos de movimientos **grandes**. Pero "grande" no tiene signo: un shock de +3% y uno de −3% son igual de grandes.

**La solución.** Elevar al cuadrado:

$$(+3)^2 = 9, \qquad (-3)^2 = 9$$

El cuadrado **conserva la magnitud y elimina el signo**. Ahora, si medimos la correlación entre $a_t^2$ y $a_{t-1}^2$, estamos preguntando: *¿los períodos de shocks grandes vienen seguidos de más shocks grandes?* — que es exactamente la pregunta del volatility clustering.

**Ejemplo concreto [B].** Consideremos dos secuencias de residuos:

- Secuencia X: +0.1, −0.1, +0.1, −0.1, +3.0, −2.8, +3.1, −2.9
- Secuencia Y: +0.1, +3.0, −0.1, −2.8, +0.1, +3.1, −0.1, −2.9

Ambas contienen exactamente los mismos ocho números y ambas tienen media cercana a cero y la misma varianza total. Pero en X los movimientos grandes están **agrupados** al final; en Y están **alternados**. La correlación de $a_t$ con $a_{t-1}$ es negativa en ambas (por la alternancia de signos), y no distingue los dos casos. La correlación de $a_t^2$ con $a_{t-1}^2$ sí los distingue claramente: alta en X, negativa en Y. **Eso es lo que el cuadrado permite ver.**

### Limitaciones de la idea del cuadrado [B]

Tsay no las enumera en este punto, pero se desprenden del resto del capítulo:

1. **El cuadrado amplifica los extremos.** Un shock de 5σ contribuye 25 veces más que uno de 1σ. Con colas pesadas —que el Capítulo 1 documentó— esto significa que las medidas basadas en $a_t^2$ están dominadas por pocas observaciones y son ruidosas. El uso de $|a_t|$ en lugar de $a_t^2$ es una alternativa menos sensible a extremos, y de hecho Tsay usa ambas en distintos lugares (la ACF de $|r_t|$ en la Figura 3.2 y en la §3.13, y $a_t^2$ en los tests).
2. **El cuadrado descarta el signo, que puede contener información.** Ésta es exactamente la debilidad que motivará EGARCH y TGARCH.
3. **$a_t^2$ es un estimador muy impreciso de $\sigma_t^2$.** Tsay lo dice explícitamente en §3.4.3: *"para una muestra dada, $a_t^2$ es un estimador **insesgado** de $\sigma_t^2$"* pero *"un solo $a_t^2$ generalmente **no es un estimador eficiente** de $\sigma_t^2$"*. Insesgado significa que en promedio acierta; ineficiente significa que cada observación individual puede estar muy lejos.

Este último punto es crucial y reaparecerá en §3.5.2 (evaluación de pronósticos).

### [B] Implicancias potenciales para ML

- El procedimiento de cuatro pasos es **transferible como protocolo de diagnóstico** a un modelo de ML: predecir, obtener residuos, examinar $a_t$ y $a_t^2$ por separado, y verificar.
- El orden importa: sin quitar primero la estructura de media, el diagnóstico de volatilidad puede dar falsos positivos.
- **Pregunta abierta de diseño:** en un modelo de ML que emite $(\hat\mu_t,\hat\sigma_t)$ y se entrena conjuntamente, ¿existe el mismo riesgo de contaminación que el procedimiento en dos etapas busca evitar, o la estimación conjunta lo resuelve? **PREGUNTA ABIERTA.**

---

## 3.4 Sección 3.3.1 — Testing for ARCH Effect

### Qué problema intenta resolver

Antes de construir un modelo de volatilidad, hay que verificar que **haga falta**. Esta sección da dos tests para responder: *¿queda estructura temporal en la magnitud de los errores que el modelo de media no explicó?*

### [A] Los dos tests

Sea $a_t=r_t-\mu_t$ el residuo de la ecuación de media. La serie $a_t^2$ se usa para verificar heterocedasticidad condicional, *"lo que también se conoce como **efectos ARCH**"*.

**Test 1 — Ljung–Box sobre $a_t^2$.** Es el mismo estadístico $Q(m)$ del Capítulo 2, aplicado a la serie de residuos al cuadrado (referencia: McLeod y Li, 1983). **La hipótesis nula es que los primeros $m$ lags de la ACF de $a_t^2$ son cero.**

**Test 2 — Test del multiplicador de Lagrange de Engle (1982).** Tsay lo describe como *"equivalente al estadístico F usual para testear $\alpha_i=0$ ($i=1,\dots,m$)"* en la regresión lineal:

$$a_t^2=\alpha_0+\alpha_1a_{t-1}^2+\dots+\alpha_ma_{t-m}^2+e_t$$

- **Hipótesis nula:** $H_0:\alpha_1=\dots=\alpha_m=0$ — ninguno de los residuos al cuadrado pasados ayuda a explicar el residuo al cuadrado actual.
- El estadístico $F$ se construye comparando la suma de cuadrados sin y con los regresores, y **se distribuye asintóticamente como una $\chi^2$ con $m$ grados de libertad bajo la nula**.
- Regla de decisión: rechazar si $F$ supera el percentil correspondiente, o si el p-valor es menor que el nivel de significancia elegido.

### La lógica del test en palabras [B]

El test LM de Engle es, en el fondo, una pregunta muy simple: **si intento predecir el tamaño del error de hoy usando los tamaños de los errores de días recientes, ¿lo consigo?** Si la respuesta es sí, hay efectos ARCH: el tamaño de los errores tiene memoria.

### [A] Ejemplo de Tsay — log-retornos mensuales de Intel, 1973–2008

Este ejemplo es la ilustración perfecta del punto central del capítulo:

| Test | Estadístico | p-valor | Qué significa |
|---|---|---|---|
| Ljung–Box sobre $r_t$ (los retornos) | $Q(12)=18.26$ | **0.11** | **No** hay correlación serial en la dirección |
| Ljung–Box sobre $a_t^2$ | $Q(12)=89.85$ | **5.3 × 10⁻¹⁴** | **Sí** hay estructura fortísima en la magnitud |
| LM de Engle | $F\approx53.62$ | **≈ 0.0000** | **Sí**, efectos ARCH fuertes |

Tsay resume: *"la serie no tiene correlaciones seriales significativas, de modo que puede usarse directamente para testear el efecto ARCH. En efecto, los estadísticos $Q(m)$ de la serie de retornos dan $Q(12)=18.26$ con un p-valor de 0.11, confirmando que no hay correlaciones seriales en los datos. Por otro lado, el test del multiplicador de Lagrange muestra **efectos ARCH fuertes**."*

**Un p-valor de 5×10⁻¹⁴ frente a uno de 0.11, sobre exactamente los mismos datos.** Eso es la diferencia entre buscar estructura en la dirección y buscarla en la magnitud.

### $ACF(a_t)$ vs $ACF(a_t^2)$ — la distinción clave

| Objeto | Qué pregunta responde | Si es significativa |
|---|---|---|
| **$ACF(a_t)$** | ¿Los errores tienen memoria en su **valor con signo**? | Puede quedar estructura en la **media** que el modelo no capturó |
| **$ACF(a_t^2)$** | ¿Los errores tienen memoria en su **magnitud**? | Puede quedar estructura en la **varianza** |

**La palabra "puede" es deliberada [B].** No es una equivalencia absoluta, por dos razones:

1. **Un $ACF(a_t^2)$ significativo puede tener otras causas.** Por ejemplo, un cambio estructural en el nivel de volatilidad —un período tranquilo seguido de uno agitado, sin dinámica genuina— produce autocorrelación en $a_t^2$ sin que exista un proceso ARCH. Tsay alude a esto en §3.6 cuando dice que *"el fenómeno IGARCH podría ser causado por cambios ocasionales de nivel en la volatilidad"* [A].
2. **Un $ACF(a_t)$ significativo no siempre indica estructura de media aprovechable.** Podría ser un artefacto de microestructura, como se estableció en el Capítulo 2. → **Cap. 5.**

### Interpretación de los resultados del test

**Qué significa rechazar la nula.** Hay evidencia de que la magnitud de los errores tiene memoria. Es decir: *vale la pena modelar la volatilidad*.

**Qué significa NO rechazar la nula.** **No significa que no haya efectos ARCH.** Igual que con el test ADF del Capítulo 2, la ausencia de evidencia no es evidencia de ausencia. Puede ocurrir por:
- **Muestra pequeña.** Con pocos datos, el test tiene poca potencia.
- **Elección desafortunada de $m$.** Si la dependencia está en lags más allá de $m$, el test no la ve.
- **Dependencia de una forma que el test no capta.** El test LM busca una relación **lineal** entre $a_t^2$ y sus rezagos. Una dependencia no lineal en la magnitud podría escapársele.
- **Una ecuación de media mal especificada** puede enmascarar o inventar estructura.

**Formulación correcta [B], en paralelo exacto con lo que establecimos para ADF:** *no rechazar significa que los datos son compatibles con la ausencia de efectos ARCH detectables con esta configuración del test; no demuestra que la volatilidad sea constante.*

### Limitaciones adicionales [B]

- Ambos tests suponen condiciones sobre los momentos que, con colas pesadas, pueden no cumplirse bien. El Ljung–Box aplicado a $a_t^2$ está trabajando con una variable cuya varianza depende del **cuarto momento** de $a_t$, que como vimos en el Capítulo 1 puede ser enorme o no existir.
- La elección de $m$ es un grado de libertad del analista, igual que en el Capítulo 2.
- Con series muy largas —y las series intradía de futuros lo son— **casi cualquier test rechaza**. Un p-valor de 10⁻¹⁴ con 100,000 observaciones dice que el efecto existe, no que sea grande. **Significancia estadística ≠ magnitud económica**, otra vez.

### [B] Adaptación a futuros

El test es directamente aplicable. Preguntas específicas que surgen y que quedan como hipótesis:
- ¿Aparecen efectos ARCH en todos los futuros y en todas las frecuencias, o hay diferencias sistemáticas por clase de activo?
- ¿La magnitud del efecto (no sólo su significancia) es comparable entre instrumentos?
- ¿Los efectos ARCH detectados a frecuencias muy altas son genuinos o consecuencia de patrones intradía deterministas (la forma de U de la volatilidad a lo largo de la sesión)? **Esta última es importante:** un patrón determinista de volatilidad por hora del día produciría autocorrelación en $a_t^2$ que no es dinámica ARCH sino estacionalidad. **PREGUNTA ABIERTA.**

### [B] Implicancias potenciales para ML

- Aplicar el test LM o Ljung–Box sobre $a_t^2$ a los residuos de un modelo de ML es un diagnóstico directo de si el modelo dejó estructura de volatilidad sin capturar.
- Un modelo de ML que predice bien la media pero cuyos residuos al cuadrado tienen fuerte autocorrelación está emitiendo predicciones puntuales razonables con **incertidumbre mal caracterizada**.

---
# 4. ARCH explicado desde cero

## 4.1 El problema que ARCH intenta resolver

Sabemos que la volatilidad se agrupa: días agitados vienen seguidos de días agitados. Queremos una **fórmula** que capture eso — que, dado lo que pasó recientemente, produzca un número: la volatilidad esperada para el próximo período.

## 4.2 La idea, en una frase

**La volatilidad esperada de hoy depende del tamaño de los shocks recientes.** Si en los últimos días hubo movimientos grandes, esperamos que hoy también lo haya.

## 4.3 [A] La definición de Tsay

*"El primer modelo que provee un marco sistemático para el modelado de volatilidad es el modelo ARCH de Engle (1982). La idea básica de los modelos ARCH es que (a) el shock $a_t$ del retorno de un activo es **serialmente no correlacionado, pero dependiente**, y (b) la dependencia de $a_t$ puede describirse mediante una **función cuadrática simple de sus valores rezagados**."*

Un modelo **ARCH($m$)** (Ec. 3.5):

$$a_t=\sigma_t\epsilon_t,\qquad \sigma_t^2=\alpha_0+\alpha_1a_{t-1}^2+\dots+\alpha_ma_{t-m}^2$$

Con las condiciones [A]: $\{\epsilon_t\}$ es una secuencia iid con media cero y varianza 1; $\alpha_0>0$; $\alpha_i\ge0$ para $i>0$. *"Los coeficientes $\alpha_i$ deben satisfacer algunas condiciones de regularidad para asegurar que la varianza incondicional de $a_t$ sea finita."* Y: *"en la práctica, $\epsilon_t$ suele suponerse normal estándar, o una Student-t estandarizada, o una distribución de error generalizada."*

## 4.4 Qué significa cada símbolo

| Símbolo | Nombre | Qué es, en palabras |
|---|---|---|
| $a_t$ | shock / innovación / residuo | La sorpresa del período $t$: lo que el modelo de media no predijo |
| $\sigma_t^2$ | varianza condicional | Cuánta variabilidad esperamos para $t$, dado el pasado |
| $\sigma_t$ | volatilidad condicional | La raíz cuadrada de lo anterior; está en las mismas unidades que el retorno |
| $\epsilon_t$ | shock estandarizado | El "dado aleatorio": media 0, varianza 1, independiente del pasado |
| $\alpha_0$ | constante / nivel base | El piso de volatilidad: cuánta variabilidad hay incluso sin shocks recientes. Debe ser **positivo** |
| $\alpha_i$ | coeficientes ARCH | Cuánto influye el shock de hace $i$ períodos. Deben ser **no negativos** |
| $a_{t-i}^2$ | shock pasado al cuadrado | La magnitud del shock de hace $i$ períodos, sin signo |
| $m$ | orden del modelo | Cuántos shocks pasados se consideran |

## 4.5 ¿Por qué esas condiciones sobre los coeficientes?

**$\alpha_0>0$ y $\alpha_i\ge0$:** porque $\sigma_t^2$ es una **varianza**, y una varianza no puede ser negativa. Si permitiéramos coeficientes negativos, una combinación desafortunada de shocks pasados podría producir una "varianza" negativa, que no tiene sentido. [A] Tsay señala que la condición $\alpha_i\ge0$ *"puede relajarse. Es una condición para asegurar que la varianza condicional $\sigma_t^2$ sea positiva para todo $t$"*, y muestra una reformulación matricial (Ec. 3.6) que garantiza positividad de forma más general.

## 4.6 [A] Cómo funciona: la mecánica del clustering

Tsay lo explica así: *"A partir de la estructura del modelo, se ve que shocks pasados al cuadrado grandes $\{a_{t-i}^2\}$ implican una varianza condicional $\sigma_t^2$ grande para la innovación $a_t$. Consecuentemente, $a_t$ **tiende** a asumir un valor grande (en módulo). Esto significa que, bajo el marco ARCH, **shocks grandes tienden a ser seguidos por otro shock grande**."*

Y añade una precisión importante [A]: *"Aquí uso la palabra **tiende** porque una varianza grande no necesariamente produce una realización grande. Sólo dice que la probabilidad de obtener un valor grande es mayor que con una varianza menor."*

**Esta precisión es central [B].** El modelo no dice "después de un día volátil vendrá un día volátil". Dice "después de un día volátil, la **probabilidad** de otro día volátil es mayor". La diferencia importa: un modelo de volatilidad no predice qué pasará, sino cómo cambia la distribución de lo que puede pasar.

### Ejemplo numérico [B]

Tomemos un ARCH(1) con $\alpha_0=0.00002$ y $\alpha_1=0.3$:

$$\sigma_t^2=0.00002+0.3\,a_{t-1}^2$$

| Situación | $a_{t-1}$ | $a_{t-1}^2$ | $\sigma_t^2$ | $\sigma_t$ |
|---|---|---|---|---|
| Ayer fue tranquilo | 0.2% = 0.002 | 0.000004 | 0.0000212 | **0.46%** |
| Ayer fue normal | 0.8% = 0.008 | 0.000064 | 0.0000392 | **0.63%** |
| Ayer hubo un shock grande | 3% = 0.03 | 0.0009 | 0.00029 | **1.70%** |

Después de un movimiento del 3%, la volatilidad esperada se multiplica casi por cuatro respecto de un día tranquilo. **Y nótese que el signo del movimiento de ayer no aparece en ninguna parte**: un −3% habría dado exactamente el mismo resultado. Ésta será la debilidad número uno.

## 4.7 [A] Propiedades importantes de ARCH(1)

**La media incondicional del shock es cero.** $E(a_t)=0$.

**La varianza incondicional** es:

$$\text{Var}(a_t)=\frac{\alpha_0}{1-\alpha_1}$$

**Qué es la varianza incondicional [B].** Es el nivel de volatilidad "de largo plazo" al que el modelo tiende. Si el mercado estuviera mucho tiempo sin shocks anormales, la volatilidad se asentaría en ese valor.

**La condición que impone** [A]: *"Como la varianza de $a_t$ debe ser positiva, requerimos $0\le\alpha_1<1$."* Si $\alpha_1$ fuera 1 o más, el denominador sería cero o negativo y la varianza incondicional no existiría. **Interpretación [B]:** un $\alpha_1$ cercano a 1 significa que los shocks de volatilidad casi no se disipan; la volatilidad "recuerda" muy fuertemente.

**El cuarto momento y la kurtosis.** Éste es el resultado más importante de la subsección y merece su propia sección de este informe (ver §8). Bajo el supuesto de que $\epsilon_t$ es **normal**, Tsay deriva:

$$\frac{E(a_t^4)}{[\text{Var}(a_t)]^2}=3\,\frac{1-\alpha_1^2}{1-3\alpha_1^2}>3$$

y señala dos implicaciones [A]:
- **(a)** $\alpha_1$ debe satisfacer $0\le\alpha_1^2<\tfrac13$ para que el cuarto momento exista.
- **(b)** *"El exceso de kurtosis de $a_t$ es positivo y la distribución de cola de $a_t$ es **más pesada** que la de una distribución normal. En otras palabras, el shock $a_t$ de un modelo ARCH(1) **condicionalmente gaussiano** es más propenso que una serie de white noise gaussiano a producir 'outliers'. Esto concuerda con el hallazgo empírico de que los 'outliers' aparecen más a menudo en los retornos de activos de lo implicado por una secuencia iid de variables normales."*

**Traducción [B]:** aunque cada shock estandarizado $\epsilon_t$ sea perfectamente normal, **el retorno resultante tiene colas pesadas**. La volatilidad variable, por sí sola, las genera. Éste es el resultado que responde la pregunta abierta del Capítulo 1.

## 4.8 [A] Las cuatro debilidades de ARCH

Tsay las enumera explícitamente:

**1. Trata igual a shocks positivos y negativos.** *"El modelo asume que shocks positivos y negativos tienen los mismos efectos sobre la volatilidad porque depende del cuadrado de los shocks previos. En la práctica, es bien sabido que el precio de un activo financiero responde de forma distinta a shocks positivos y negativos."*

**2. Es restrictivo.** *"Por ejemplo, $\alpha_1^2$ de un modelo ARCH(1) debe estar en el intervalo $[0,\tfrac13]$ si la serie tiene un cuarto momento finito. La restricción se vuelve complicada para modelos ARCH de orden superior. En la práctica, **limita la capacidad de los modelos ARCH con innovaciones gaussianas de capturar exceso de kurtosis**."*

**3. No explica nada.** *"El modelo ARCH no provee ninguna intuición nueva para entender la fuente de las variaciones de una serie temporal financiera. Meramente provee una forma mecánica de describir el comportamiento de la varianza condicional. **No da ninguna indicación sobre qué causa que ese comportamiento ocurra**."*

**4. Sobrepredice.** *"Los modelos ARCH son propensos a **sobrepredecir** la volatilidad porque **responden lentamente a shocks grandes y aislados** en la serie de retornos."*

### Explicación de la debilidad 4 [B]

Ésta merece desarrollo porque es la menos obvia. En un ARCH($m$), el shock de hace $i$ períodos entra en la fórmula con peso $\alpha_i$ **mientras $i\le m$, y desaparece de golpe cuando $i>m$**. No hay decaimiento suave: el shock pesa lo mismo el día 1 que el día $m$, y luego cae a cero abruptamente.

Consecuencia: tras un evento aislado y extremo —un crash de un día—, el modelo mantiene la volatilidad elevada durante exactamente $m$ períodos y luego la deja caer bruscamente. Si el mercado en realidad se calmó al día siguiente, el modelo estuvo sobrepredicendo la volatilidad durante $m-1$ días.

### La debilidad de fondo: necesita demasiados lags [B]

Si la volatilidad tiene memoria larga —y el Capítulo 2 mostró que la ACF de $|r_t|$ es significativa tras cientos de lags—, entonces un ARCH necesitaría un $m$ enorme para capturarla. Cada lag adicional es un parámetro más que estimar. Tsay lo constata en un ejemplo concreto [A]: para los retornos excedentes mensuales del S&P 500, *"se necesita un modelo ARCH(9) para el proceso de volatilidad. Debe buscarse algún modelo alternativo."*

**Nueve parámetros para modelar la volatilidad de una serie mensual.** Ése es el problema que GARCH resuelve con dos.

## 4.9 ¿Por qué ARCH fue una buena idea y por qué no fue suficiente?

**Por qué fue buena idea [B]:** fue la primera formulación que convirtió una observación cualitativa ("la volatilidad se agrupa") en un modelo estadístico estimable, con parámetros interpretables, capaz de producir pronósticos y de ser verificado. Además, reveló algo no obvio: que la volatilidad variable **genera** colas pesadas, lo que unificó dos hechos estilizados que parecían separados.

**Por qué no fue suficiente [B]:** porque es una descripción cruda de un fenómeno que tiene memoria larga y respuesta asimétrica. Necesita muchos parámetros, no distingue el signo del shock, y su respuesta a los shocks es de encendido/apagado en lugar de gradual.

## 4.10 [A] Cómo se elige el orden $m$

Tsay recomienda usar la **PACF de $a_t^2$** y da la justificación: definiendo $\eta_t=a_t^2-\sigma_t^2$, se puede demostrar que $\{\eta_t\}$ es una serie no correlacionada de media 0, y el modelo ARCH se reescribe como

$$a_t^2=\alpha_0+\alpha_1a_{t-1}^2+\dots+\alpha_ma_{t-m}^2+\eta_t$$

*"que tiene la forma de un modelo AR($m$) para $a_t^2$, excepto que $\{\eta_t\}$ no es una serie iid."* Por tanto, **la PACF de $a_t^2$ sirve para determinar $m$**, igual que la PACF servía para determinar el orden AR en el Capítulo 2.

Pero con dos advertencias explícitas [A]: *"como los $\{\eta_t\}$ no están idénticamente distribuidos, las estimaciones por mínimos cuadrados del modelo anterior son **consistentes pero no eficientes**"* y *"**la PACF de $a_t^2$ puede no ser efectiva cuando el tamaño de muestra es pequeño**"*.

**[B] Advertencia adicional, heredada del Capítulo 2:** la PACF es una herramienta de diagnóstico dentro de un marco AR, no un selector automático. Todo lo que se dijo en el Capítulo 2 sobre no convertir la PACF en un selector de lookback aplica igual aquí.

---

# 5. GARCH explicado desde cero

## 5.1 El problema que GARCH intenta resolver

ARCH necesita demasiados parámetros. Queremos capturar una memoria larga de volatilidad **con pocos números**.

## 5.2 La idea, en una frase

**La volatilidad esperada de hoy = un nivel base + el efecto del shock de ayer + una porción de la volatilidad que ya veníamos teniendo.**

El truco es la tercera pieza: en lugar de listar explícitamente todos los shocks pasados, **reutilizamos la volatilidad de ayer, que ya los resume**.

## 5.3 [A] La definición

Tsay presenta el GARCH($m,s$) de Bollerslev (1986), Ec. (3.14):

$$a_t=\sigma_t\epsilon_t,\qquad \sigma_t^2=\alpha_0+\sum_{i=1}^{m}\alpha_ia_{t-i}^2+\sum_{j=1}^{s}\beta_j\sigma_{t-j}^2$$

con $\alpha_0>0$, $\alpha_i\ge0$, $\beta_j\ge0$, y $\sum_{i=1}^{\max(m,s)}(\alpha_i+\beta_i)<1$. *"Esta última restricción sobre $\alpha_i+\beta_i$ implica que **la varianza incondicional de $a_t$ es finita**, mientras que su varianza condicional $\sigma_t^2$ **evoluciona en el tiempo**."* Los $\alpha_i$ se llaman **parámetros ARCH** y los $\beta_j$, **parámetros GARCH**. Si $s=0$, se reduce a un ARCH puro.

**El caso que importa en la práctica**, GARCH(1,1), Ec. (3.16):

$$\sigma_t^2=\alpha_0+\alpha_1a_{t-1}^2+\beta_1\sigma_{t-1}^2,\qquad 0\le\alpha_1,\beta_1\le1,\ (\alpha_1+\beta_1)<1$$

## 5.4 Traducción a palabras

$$\underbrace{\sigma_t^2}_{\substack{\text{volatilidad}\\ \text{esperada hoy}}} = \underbrace{\alpha_0}_{\substack{\text{nivel}\\ \text{base}}} + \underbrace{\alpha_1 a_{t-1}^2}_{\substack{\text{efecto del}\\ \text{shock de ayer}}} + \underbrace{\beta_1\sigma_{t-1}^2}_{\substack{\text{parte de la volatilidad}\\ \text{que ya teníamos}}}$$

## 5.5 La diferencia entre $\alpha_1$ y $\beta_1$ — la analogía

Ésta es la intuición que hay que retener:

- **$\alpha_1$ mide cuánto REACCIONA la volatilidad a una noticia nueva.** Es la sensibilidad al shock más reciente. Un $\alpha_1$ alto significa un mercado nervioso: cualquier sorpresa dispara la volatilidad esperada.

- **$\beta_1$ mide cuánto PERSISTE la volatilidad existente.** Es cuánto de la volatilidad de ayer se arrastra a hoy. Un $\beta_1$ alto significa que la volatilidad, una vez elevada, tarda mucho en volver a la normalidad.

**Analogía [B].** Pensemos en la temperatura de una habitación:

- **$\alpha_1$ es cuán potente es la estufa.** Cuánto sube la temperatura cuando la encendemos.
- **$\beta_1$ es cuán buena es la aislación.** Cuánto calor conserva la habitación de un momento al siguiente.
- **$\alpha_0$ es la temperatura exterior.** El nivel al que la habitación tiende si no hacemos nada.

Una habitación con estufa potente y mala aislación ($\alpha$ alto, $\beta$ bajo) se calienta rápido y se enfría rápido: la volatilidad reacciona bruscamente pero vuelve pronto a la normalidad. Una con estufa débil y buena aislación ($\alpha$ bajo, $\beta$ alto) cambia poco de golpe pero mantiene el nivel durante muchísimo tiempo — que es, empíricamente, lo que se observa en los mercados.

### Valores típicos [A]

En el ejemplo de Tsay con retornos excedentes mensuales del S&P 500 (792 observaciones, desde 1926):

$$\sigma_t^2=0.000086+0.1216\,a_{t-1}^2+0.8511\,\sigma_{t-1}^2$$

$\alpha_1\approx0.12$, $\beta_1\approx0.85$. **La aislación es mucho más importante que la estufa.** Esto es característico: los valores publicados en la literatura suelen estar en ese rango.

## 5.6 Qué significa $\alpha_1+\beta_1$

Esta suma se llama **persistencia**, y es probablemente el número más informativo de todo el modelo.

**Interpretación intuitiva [B].** Es la fracción del "exceso de volatilidad" que sobrevive de un período al siguiente. Si $\alpha_1+\beta_1=0.9$, entonces tras un shock, el 90% del exceso de volatilidad permanece un período después, el 81% dos períodos después, y así sucesivamente.

| $\alpha_1+\beta_1$ | Interpretación | Comportamiento |
|---|---|---|
| **Baja (≈0.5)** | Poca persistencia | Tras un shock, la volatilidad vuelve a la normalidad en pocos períodos |
| **Alta (≈0.9)** | Mucha persistencia | Un shock eleva la volatilidad durante decenas de períodos |
| **Muy cercana a 1 (≈0.98)** | Persistencia extrema | El efecto de un shock casi no se disipa |
| **Igual a 1** | Caso límite → IGARCH | El efecto **no** se disipa; la varianza incondicional no existe |

**La varianza incondicional** [A] resulta:

$$E(a_t^2)=\frac{\alpha_0}{1-(\alpha_1+\beta_1)}$$

Nótese que si $\alpha_1+\beta_1$ se acerca a 1, el denominador se acerca a 0 y la varianza de largo plazo se dispara. En el ejemplo del S&P 500: $0.000086/(1-0.8511-0.1216)=0.00314$, que Tsay señala que *"es cercana a"* la varianza estimada por el modelo de media sin GARCH.

**En el ejemplo de Tsay:** $\hat\alpha_1+\hat\beta_1=0.9772$. Y comenta [A]: *"este fenómeno se observa comúnmente en la práctica y lleva a imponer la restricción $\alpha_1+\beta_1=1$ en un modelo GARCH(1,1), resultando en un modelo GARCH integrado (o IGARCH)."*

**Advertencia [B], siguiendo el criterio del proyecto:** que $\hat\alpha_1+\hat\beta_1$ resulte 0.977 **no demuestra** que la persistencia verdadera sea 1. Ver §6.3 sobre IGARCH.

## 5.7 [A] La equivalencia GARCH ↔ ARMA

Ésta es la conexión con el Capítulo 2 y merece explicarse con cuidado.

**El truco algebraico.** Definimos $\eta_t=a_t^2-\sigma_t^2$: la diferencia entre el shock al cuadrado observado y la varianza condicional que el modelo esperaba. Sustituyendo $\sigma_{t-i}^2=a_{t-i}^2-\eta_{t-i}$ en la ecuación GARCH, se obtiene (Ec. 3.15):

$$a_t^2=\alpha_0+\sum_{i=1}^{\max(m,s)}(\alpha_i+\beta_i)a_{t-i}^2+\eta_t-\sum_{j=1}^{s}\beta_j\eta_{t-j}$$

Tsay lo resume [A]: *"La Ecuación (3.15) es **una forma ARMA para la serie al cuadrado $a_t^2$**. Así, **un modelo GARCH puede considerarse una aplicación de la idea ARMA a la serie al cuadrado $a_t^2$**."*

Y precisa la naturaleza de $\eta_t$ [A]: *"es fácil verificar que $\{\eta_t\}$ es una serie de diferencias de martingala [es decir, $E(\eta_t)=0$ y $\text{cov}(\eta_t,\eta_{t-j})=0$ para $j\ge1$]. **Sin embargo, $\{\eta_t\}$ en general no es una secuencia iid.**"

### Traducción sencilla [B]

> **En el Capítulo 2 modelábamos cómo persistía el retorno o el shock. En el Capítulo 3 usamos exactamente la misma lógica para modelar cómo persiste la volatilidad.**

Todo lo que aprendimos en el Capítulo 2 —persistencia, raíces características, estacionariedad, decaimiento de la ACF, reversión a la media, convergencia de los pronósticos— **se reutiliza aquí**, aplicado a $a_t^2$ en lugar de a $r_t$. Ésta es la razón por la que valía la pena estudiar ARMA aunque Tsay dijera que casi no se usa para retornos.

**Un detalle importante [A+B]:** el término $\eta_t$ es una diferencia de martingala pero **no es iid**. Eso significa que la analogía ARMA es formal pero no perfecta: los resultados estándar de ARMA que dependen de innovaciones iid no se trasladan automáticamente. Es una de esas diferencias que conviene registrar en lugar de pasar por alto.

## 5.8 Pronóstico con GARCH

### La pregunta

Estamos en el período $h$ y queremos predecir la volatilidad de $h+1$, $h+2$, ..., $h+\ell$.

### [A] Pronóstico a 1 paso

Trivial, porque todo lo necesario ya se conoce:

$$\sigma_h^2(1)=\alpha_0+\alpha_1a_h^2+\beta_1\sigma_h^2$$

*"donde $a_h$ y $\sigma_h^2$ son conocidos en el índice temporal $h$"* [A].

### [A] Pronóstico a múltiples pasos

Para $\ell>1$, Tsay deriva la fórmula recursiva (Ec. 3.17):

$$\sigma_h^2(\ell)=\alpha_0+(\alpha_1+\beta_1)\,\sigma_h^2(\ell-1)$$

Y observa [A]: *"Este resultado es exactamente el mismo que el de un modelo ARMA(1,1) con polinomio AR $1-(\alpha_1+\beta_1)B$."*

Resolviendo la recursión:

$$\sigma_h^2(\ell)=\frac{\alpha_0[1-(\alpha_1+\beta_1)^{\ell-1}]}{1-\alpha_1-\beta_1}+(\alpha_1+\beta_1)^{\ell-1}\sigma_h^2(1)$$

**Y el resultado clave** [A]:

$$\sigma_h^2(\ell)\longrightarrow\frac{\alpha_0}{1-\alpha_1-\beta_1}\qquad\text{cuando }\ell\to\infty$$

*"siempre que $\alpha_1+\beta_1<1$. Consecuentemente, **los pronósticos de volatilidad a múltiples pasos de un modelo GARCH(1,1) convergen a la varianza incondicional de $a_t$** a medida que el horizonte de pronóstico crece, siempre que $\text{Var}(a_t)$ exista."*

### Explicación intuitiva [B]

Esto es exactamente la **reversión a la media** del Capítulo 2, aplicada a la volatilidad.

Cuanto más lejos miramos, menos importa lo que está pasando ahora, y más nos acercamos a decir "la volatilidad será la normal de largo plazo". El factor $(\alpha_1+\beta_1)^{\ell-1}$ mide cuánto pesa todavía la información actual a $\ell$ períodos vista, y como $\alpha_1+\beta_1<1$, ese peso se va a cero.

**La velocidad de esa convergencia es lo que mide la persistencia.** Con $\alpha_1+\beta_1=0.5$, el peso de la información actual a 10 períodos es $0.5^9\approx0.002$: prácticamente nada. Con $\alpha_1+\beta_1=0.98$, es $0.98^9\approx0.83$: la información actual sigue dominando.

### [A] Ejemplo numérico de Tsay (Tabla 3.1)

Pronósticos para los retornos excedentes mensuales del S&P 500, origen $h=792$ (diciembre 1991):

| Horizonte | 1 | 2 | 3 | 4 | 5 | ∞ |
|---|---|---|---|---|---|---|
| Retorno | 0.0076 | 0.0076 | 0.0076 | 0.0076 | 0.0076 | 0.0076 |
| Volatilidad | 0.0536 | 0.0537 | 0.0537 | 0.0538 | 0.0538 | **0.0560** |

**Dos lecturas [B]:**
1. **El pronóstico de retorno es constante** en todos los horizontes: 0.76% mensual. La ecuación de media es sólo una constante, así que no hay nada dinámico que predecir en la dirección.
2. **El pronóstico de volatilidad sube gradualmente** de 5.36% hacia su valor de largo plazo de 5.60%. En este caso particular la convergencia es lenta y el punto de partida ya estaba cerca del nivel de largo plazo, así que el efecto es pequeño. En un momento de crisis, con $\sigma_h$ muy por encima del nivel de largo plazo, la trayectoria sería descendente y mucho más pronunciada.

## 5.9 [A] Verificación del modelo

*"El modelo ajustado puede verificarse usando el **residuo estandarizado** $\tilde a_t=a_t/\sigma_t$ y su proceso al cuadrado."*

**Qué es el residuo estandarizado [B].** Es el shock dividido por la volatilidad que el modelo esperaba para ese momento. Si el modelo capturó bien la volatilidad, entonces $\tilde a_t$ debería ser aproximadamente el $\epsilon_t$ de la ecuación $a_t=\sigma_t\epsilon_t$: media 0, varianza 1, **sin memoria de ningún tipo**.

**Los dos tests [A]:**
- ACF y Ljung–Box sobre $\tilde a_t$ → ¿queda estructura en la media?
- ACF y Ljung–Box sobre $\tilde a_t^2$ → ¿queda estructura en la volatilidad?

**En el ejemplo del S&P 500** [A]: $Q(12)=11.99$ (p = 0.45) y $Q(24)=28.52$ (p = 0.24) para $\tilde a_t$; $Q(12)=13.11$ (p = 0.36) y $Q(24)=26.45$ (p = 0.33) para $\tilde a_t^2$. Tsay concluye: *"Así, el modelo parece ser adecuado para describir la dependencia lineal en las series de retorno y de volatilidad."*

**[B] Nota de prudencia, coherente con el Capítulo 2:** que los residuos estandarizados pasen Ljung–Box significa que no queda dependencia **lineal detectable** en la media ni en la magnitud. No significa que sean iid. Sigue habiendo la posibilidad de dependencia no lineal → **Cap. 4**.

## 5.10 [A] La distribución de $\epsilon_t$

Tsay reestima el mismo modelo suponiendo que $\epsilon_t$ sigue una **Student-t estandarizada con 5 grados de libertad** en lugar de una normal:

$$r_t=0.0085+a_t,\qquad \sigma_t^2=0.00012+0.1121a_{t-1}^2+0.8432\sigma_{t-1}^2$$

Los coeficientes cambian poco. Al **estimar** los grados de libertad en lugar de fijarlos, obtiene **7.02** con error estándar 1.78, y concluye [A] que *"no podemos rechazar la hipótesis de usar una Student-t estandarizada con 5 grados de libertad al nivel de significancia del 5%"*.

**Qué es la Student-t en este contexto [B].** Es una distribución con forma de campana pero con **colas más gruesas** que la normal. El parámetro "grados de libertad" ($v$) controla cuán gruesas: cuanto menor $v$, más pesadas las colas. Con $v$ grande, se parece cada vez más a la normal.

**Por qué se usa [B].** Porque, como veremos en §8, GARCH con shocks normales genera colas pesadas **pero puede no generar suficientes**. Usar una Student-t para $\epsilon_t$ añade colas pesadas adicionales.

**Lo que Tsay observa sobre esto** [A], en un comentario que conviene retener: *"estudios empíricos recientes de series financieras de alta frecuencia indican que **el comportamiento de cola de los modelos GARCH sigue siendo demasiado corto incluso con innovaciones Student-t estandarizadas**."*

---
# 6. La familia GARCH

Cada miembro de esta familia existe porque alguien detectó una limitación del anterior. Conviene verlos así: como una secuencia de reparaciones.

## 6.1 Sección 3.5.2 — Forecasting Evaluation: cómo evaluar lo que no se ve

Esta sección es breve en el libro pero conceptualmente crítica, y por eso se trata antes que el resto de la familia.

### La pregunta

**¿Cómo evaluamos una predicción de volatilidad si la volatilidad verdadera no se observa?**

Con un pronóstico de retorno, la evaluación es directa: predijimos +0.3%, ocurrió −0.1%, el error es 0.4%. Con la volatilidad no podemos hacer eso, porque nunca sabremos cuál era la volatilidad verdadera del período.

### [A] Qué dice Tsay

*"Como la volatilidad del retorno de un activo no es directamente observable, comparar el desempeño predictivo de distintos modelos de volatilidad es **un desafío** para los analistas de datos. En la literatura, algunos investigadores usan pronósticos fuera de muestra y comparan los pronósticos de volatilidad $\sigma_h^2(\ell)$ con **el shock $a_{h+\ell}^2$** en la muestra de pronóstico. **Este enfoque a menudo encuentra un coeficiente de correlación bajo entre $a_{h+\ell}^2$ y $\sigma_h^2(\ell)$, es decir, un $R^2$ bajo.**"*

Y entonces viene la parte importante [A]:

*"Sin embargo, **tal hallazgo no es sorprendente** porque $a_{h+\ell}^2$ por sí solo **no es una medida adecuada** de la volatilidad en el índice temporal $h+\ell$. Consideremos los pronósticos a 1 paso. Desde un punto de vista estadístico, $E(a_{h+1}^2\mid\mathcal{F}_h)=\sigma_{h+1}^2$, de modo que $a_{h+1}^2$ es **un estimador consistente** de $\sigma_{h+1}^2$. **Pero no es un estimador preciso** de $\sigma_{h+1}^2$ porque **una sola observación de una variable aleatoria con media conocida no puede proveer una estimación precisa de su varianza**. Consecuentemente, tal enfoque para evaluar el desempeño predictivo de los modelos de volatilidad **estrictamente hablando no es apropiado**."*

Referencia para más información: Andersen y Bollerslev (1998).

### Explicación en lenguaje llano [B]

Esto es sutil y vale la pena desarmarlo.

**"Consistente"** significa: si pudiéramos promediar infinitas realizaciones de $a_{h+1}^2$ bajo las mismas condiciones, obtendríamos exactamente $\sigma_{h+1}^2$. **En promedio, acierta.**

**"No preciso"** significa: una sola realización puede estar muy lejos del valor verdadero. **Individualmente, es ruidosísimo.**

**Analogía [B].** Supongamos que queremos medir la altura promedio de los habitantes de una ciudad, y sólo se nos permite medir a **una** persona elegida al azar. Esa medición es *insesgada*: si repitiéramos el experimento muchas veces, el promedio de las mediciones daría la altura promedio real. Pero con una sola persona, la estimación puede errar por 30 centímetros.

Ahora imaginemos que alguien evalúa nuestro método de estimación comparando nuestra predicción (digamos, 1.70 m) con esa única medición (que resultó ser 1.95 m porque tocó un jugador de básquet) y concluye que nuestro método es terrible. **El método puede ser excelente; el patrón de comparación es el que es ruidoso.**

Eso es exactamente lo que ocurre al evaluar $\hat\sigma_t^2$ contra $r_t^2$. El $R^2$ bajo puede reflejar el ruido del patrón de comparación, no la calidad del pronóstico.

### La jerarquía de conceptos

| Concepto | Qué es | ¿Se observa? |
|---|---|---|
| **Volatilidad latente $\sigma_t$** | La cantidad verdadera que queremos conocer | **No** |
| **Proxy de volatilidad** | Cualquier cantidad observable que estima $\sigma_t$ | Sí, pero con error |
| **Retorno al cuadrado $r_t^2$** | El proxy más simple: una sola observación | Sí; **insesgado pero muy impreciso** |
| **Volatilidad realizada $RV_t$** | Suma de retornos intradía al cuadrado: muchas observaciones | Sí; **mucho más preciso**, pero con sus propios problemas (ver §7.3) |
| **Estimadores OHLC** | Usan apertura, máximo, mínimo, cierre del período | Sí; más preciso que sólo el cierre bajo ciertos supuestos (ver §7.4) |

### [B] Limitaciones de evaluar $\hat\sigma_t^2$ contra $r_t^2$

1. **Ruido dominante.** Bajo normalidad condicional, la varianza de $r_t^2$ alrededor de $\sigma_t^2$ es enorme: la razón señal/ruido del proxy es intrínsecamente baja.
2. **Un $R^2$ bajo puede aparecer incluso con un buen pronóstico de volatilidad [B].** Como $r_t^2$ es un proxy extremadamente ruidoso de la varianza condicional, comparar $\sigma_t^2$ contra una sola realización de $r_t^2$ puede producir una correlación y un $R^2$ bajos incluso cuando el pronóstico contiene información útil. El resultado cuantitativo específico sobre cuánto puede estar limitado el $R^2$ no es desarrollado por Tsay; pertenece a la literatura de evaluación de volatilidad a la que remite
3. **Colas pesadas.** El proxy $r_t^2$ está dominado por los pocos días extremos. Una métrica de error cuadrático sobre $r_t^2$ está midiendo, esencialmente, el desempeño en esos días.
4. **Elegir la métrica cambia el ganador.** Distintas funciones de pérdida (error cuadrático sobre varianzas, sobre volatilidades, sobre logaritmos, QLIKE) pueden ordenar los modelos de forma distinta. Tsay no desarrolla esto.

### [B] Implicancia para el proyecto

**No se adopta ninguna métrica de evaluación de volatilidad en esta etapa.** Lo que queda establecido es que:
- La elección del proxy es una decisión metodológica de primer orden, **no un detalle técnico**.
- Un $R^2$ bajo al evaluar un pronóstico de volatilidad contra retornos al cuadrado **no es evidencia de que el pronóstico sea malo**.
- Si se dispone de datos intradía, existe un proxy mucho mejor. → §7.3.

**PREGUNTA ABIERTA:** ¿qué proxy y qué función de pérdida son apropiados para evaluar pronósticos de volatilidad en futuros? Requiere el Cap. 5 (para entender el ruido de microestructura) y literatura específica de evaluación de pronósticos de volatilidad.

## 6.2 Sección 3.6 — IGARCH: cuando la persistencia llega al límite

### El problema

En la práctica, $\hat\alpha_1+\hat\beta_1$ sale casi siempre muy cerca de 1. ¿Qué pasa exactamente en el límite?

### [A] La definición

*"Si el polinomio AR de la representación GARCH en la Ec. (3.15) tiene una raíz unitaria, entonces tenemos un modelo IGARCH. Así, **los modelos IGARCH son modelos GARCH con raíz unitaria**. Similar a los modelos ARIMA, una característica clave de los modelos IGARCH es que **el impacto de los shocks al cuadrado pasados sobre $a_t^2$ es persistente**."*

Un IGARCH(1,1) [A]:

$$\sigma_t^2=\alpha_0+\beta_1\sigma_{t-1}^2+(1-\beta_1)a_{t-1}^2$$

Nótese que los dos coeficientes suman exactamente 1 por construcción.

### [A] Las consecuencias

**1. La varianza incondicional no existe.** *"La varianza incondicional de $a_t$, y por tanto la de $r_t$, **no está definida** bajo el modelo IGARCH(1,1) anterior. Esto parece difícil de justificar para una serie de retornos excedentes."*

**2. Los pronósticos no revierten.** Con $\alpha_1+\beta_1=1$, la fórmula recursiva da (Ec. 3.22):

$$\sigma_h^2(\ell)=\sigma_h^2(1)+(\ell-1)\alpha_0,\qquad \ell\ge1$$

*"Consecuentemente, el efecto de $\sigma_h^2(1)$ sobre las volatilidades futuras es también persistente, y **los pronósticos de volatilidad forman una línea recta con pendiente $\alpha_0$**."*

**3. El caso especial $\alpha_0=0$.** *"En este caso, los pronósticos de volatilidad son simplemente $\sigma_h^2(1)$ para todos los horizontes... Este modelo IGARCH(1,1) especial es **el modelo de volatilidad usado en RiskMetrics**, que es un enfoque para calcular el value at risk; ver el Capítulo 7. El modelo es también **un modelo de suavizado exponencial** para la serie $\{a_t^2\}$."*

Tsay lo demuestra con sustituciones repetidas:

$$\sigma_t^2=(1-\beta_1)(a_{t-1}^2+\beta_1a_{t-2}^2+\beta_1^2a_{t-3}^2+\cdots)$$

*"que es la conocida formación de suavizado exponencial con $\beta_1$ como factor de descuento"*.

**Traducción [B].** El IGARCH sin constante es exactamente **un promedio móvil exponencialmente ponderado de los retornos al cuadrado**. Es decir: una media móvil de la volatilidad reciente donde los datos más antiguos pesan cada vez menos. Es una herramienta que muchos practicantes usan sin saber que están usando un IGARCH.

**4. Propiedades estadísticas raras.** [A] Citando a Nelson (1990): *"el proceso $\sigma_t^2$ es una martingala... Bajo ciertas condiciones, el proceso de volatilidad es **estrictamente estacionario pero no débilmente estacionario** porque no tiene los primeros dos momentos."*

**Esto es curioso y vale la pena notarlo [B]:** en el Capítulo 2 vimos que estacionariedad estricta más momentos finitos implica estacionariedad débil. Aquí tenemos el caso donde los momentos no existen y por tanto la implicación no aplica.

### El paralelo y la diferencia con el Capítulo 2

| | **Raíz unitaria en AR (Cap. 2)** | **IGARCH (Cap. 3)** |
|---|---|---|
| ¿Qué es lo persistente? | El **nivel** de la serie | La **varianza condicional** |
| Ejemplo | El log-precio de un activo | La volatilidad |
| Efecto del shock | Permanente sobre el nivel | Permanente sobre la volatilidad |
| Qué no existe | La media incondicional (no hay nivel fijo) | La varianza incondicional (no hay nivel fijo de volatilidad) |
| Pronóstico multipaso | Constante = último valor | Línea recta con pendiente $\alpha_0$ (o constante si $\alpha_0=0$) |

**Son objetos distintos [B].** Un mercado puede tener raíz unitaria en el precio (que es lo esperable) y **no** tener IGARCH en la volatilidad, o viceversa. Uno afecta al nivel, el otro a la dispersión. La analogía formal es útil para entender la matemática, pero confundirlos sería un error conceptual serio.

### Por qué $\hat\alpha+\hat\beta\approx1$ NO demuestra IGARCH

Ésta es la advertencia que el proyecto pidió explícitamente, y **Tsay mismo la sugiere** [A]:

> *"Desde un punto de vista teórico, **el fenómeno IGARCH podría ser causado por cambios ocasionales de nivel en la volatilidad**. La causa real de la persistencia en volatilidad **merece una investigación cuidadosa**."*

**Desarrollo [B].** Hay al menos cuatro razones para no dar el salto:

1. **Cambios de régimen mal especificados.** Ésta es la que Tsay señala. Si la volatilidad verdaderamente tuviera dos niveles (uno tranquilo y uno de crisis) y saltara ocasionalmente entre ellos, un GARCH que no contempla regímenes intentaría explicar esos saltos con persistencia altísima. El $\hat\alpha+\hat\beta\approx1$ sería un **artefacto de mala especificación**, no una propiedad del mercado. → **PREGUNTA ABIERTA, Caps. 11–12** (Markov switching).

2. **Es una estimación puntual con error estándar.** En el ejemplo de Tsay, $\hat\alpha_1=0.1216$ (e.e. 0.0197) y $\hat\beta_1=0.8511$ (e.e. 0.0190). La suma 0.9727 tiene su propia incertidumbre; no es un valor conocido con precisión infinita.

3. **La distinción es empíricamente difícil**, exactamente igual que distinguir $\phi_1=0.99$ de $\phi_1=1$ en el Capítulo 2. Los tests tienen poca potencia contra alternativas cercanas.

4. **La consecuencia de equivocarse es asimétrica.** Si asumimos IGARCH y en realidad hay reversión, nuestros pronósticos de largo horizonte serán sistemáticamente erróneos: predeciremos volatilidad alta indefinidamente después de una crisis, cuando en realidad va a bajar.

**Formulación correcta [B]:** *un $\hat\alpha_1+\hat\beta_1$ cercano a 1 indica persistencia empírica muy alta y es compatible con un IGARCH; también es compatible con un GARCH estacionario de persistencia alta y con un proceso con cambios de nivel. La elección entre ellos requiere evidencia adicional que este capítulo no provee.*

## 6.3 Sección 3.7 — GARCH-M: ¿la volatilidad predice el retorno?

### La pregunta conceptual

Hasta ahora, la volatilidad y la media eran dos problemas separados. **¿Y si la volatilidad esperada influyera sobre el retorno esperado?**

La intuición financiera clásica dice que sí debería: si un activo es más riesgoso, los inversores deberían exigir mayor retorno esperado para tenerlo. Eso se llama **prima de riesgo**.

### [A] El modelo

*"En finanzas, el retorno de un título puede depender de su volatilidad. Para modelar tal fenómeno, uno puede considerar el modelo GARCH-M, donde M significa GARCH-in-the-mean."* Ec. (3.23):

$$r_t=\mu+c\,\sigma_t^2+a_t,\qquad a_t=\sigma_t\epsilon_t,\qquad \sigma_t^2=\alpha_0+\alpha_1a_{t-1}^2+\beta_1\sigma_{t-1}^2$$

**El símbolo nuevo es $c$**, llamado por Tsay [A] *"el **parámetro de prima de riesgo**. Un $c$ positivo indica que el retorno está positivamente relacionado con su volatilidad."*

Menciona [A] que también se usan otras especificaciones: $r_t=\mu+c\sigma_t+a_t$ y $r_t=\mu+c\ln(\sigma_t^2)+a_t$.

**Una consecuencia interesante** [A]: *"La formulación del modelo GARCH-M implica que **hay correlaciones seriales en la serie de retornos $r_t$. Estas correlaciones seriales son introducidas por las del proceso de volatilidad** $\{\sigma_t^2\}$. La existencia de prima de riesgo es, por lo tanto, **otra razón por la que algunos retornos históricos de acciones tienen correlaciones seriales**."*

**Esto es notable [B]:** ofrece una explicación de por qué podríamos ver algo de autocorrelación en retornos sin que sea una ineficiencia explotable. Si la volatilidad es persistente y el retorno esperado depende de la volatilidad, entonces el retorno esperado también será persistente — pero eso es compensación por riesgo, no una anomalía.

### [A] El resultado empírico

Para los retornos excedentes mensuales del S&P 500 (enero 1926 – diciembre 1991):

$$r_t=0.0055+1.09\,\sigma_t^2+a_t,\qquad \sigma_t^2=8.76\times10^{-5}+0.123a_{t-1}^2+0.849\sigma_{t-1}^2$$

El error estándar del coeficiente 1.09 es **0.818**. Tsay concluye [A]: *"La prima de riesgo estimada para el retorno del índice es **positiva pero no es estadísticamente significativa al nivel del 5%**."*

**Nota:** $1.09/0.818\approx1.33$, un t-ratio muy por debajo del umbral convencional de ~1.96.

Y en el otro ejemplo del capítulo (§3.8.2, EGARCH sobre el índice CRSP diario, 6,408 observaciones), la estimación de $c$ es **−3.361 con error estándar 2.026** [A]: *"La prima de riesgo estimada es **negativa, pero estadísticamente insignificante**."*

### La distinción crítica [B]

$$\underbrace{\textbf{Predecir la volatilidad}}_{\text{¿cuánto se moverá?}} \qquad\neq\qquad \underbrace{\textbf{Usar la volatilidad para predecir el retorno}}_{\text{¿hacia dónde se moverá?}}$$

La primera es lo que hace GARCH y sobre lo que hay evidencia fuerte. La segunda es lo que intenta GARCH-M, y **en los dos ejemplos de Tsay no funciona**: en uno el coeficiente es positivo y no significativo, en el otro es negativo y no significativo.

**No debe concluirse que la relación no existe** — dos ejemplos sobre índices accionarios estadounidenses no cierran una pregunta que la literatura financiera lleva décadas debatiendo. Pero **tampoco debe asumirse que existe, ni que es positiva, ni que es estable.**

### [B] Preguntas que tendría sentido comprobar en futuros

Todas como hipótesis, ninguna como decisión:
- ¿El coeficiente de prima de riesgo es significativo en algún futuro concreto? ¿Con qué signo?
- ¿Es estable entre subperíodos, o cambia de signo?
- ¿Difiere sistemáticamente entre clases de activo (índices vs commodities vs tasas)?
- ¿La relación es lineal en $\sigma_t^2$, en $\sigma_t$, en $\ln\sigma_t^2$, o no lineal? Tsay muestra que hay al menos tres especificaciones en uso [A], lo que sugiere que la forma funcional no está establecida.
- Si existiera, ¿su magnitud sería suficiente para superar costos?

## 6.4 Sección 3.8 — EGARCH: permitir que el signo importe

### El problema que resuelve

En GARCH, la volatilidad depende de $a_{t-1}^2$. Y como

$$(+3)^2 = (-3)^2 = 9$$

**para un GARCH estándar, un shock de +3% y uno de −3% tienen exactamente el mismo efecto sobre la volatilidad futura.** Pero la característica número 4 de la §3.1 decía que en los datos no parece ser así.

### [A] La solución de Nelson (1991)

*"Para superar algunas debilidades del modelo GARCH en el manejo de series financieras, Nelson (1991) propone el modelo EGARCH. En particular, **para permitir efectos asimétricos entre retornos positivos y negativos**, consideró la innovación ponderada"* (Ec. 3.24):

$$g(\epsilon_t)=\theta\,\epsilon_t+\gamma\,[\,|\epsilon_t|-E(|\epsilon_t|)\,]$$

donde $\theta$ y $\gamma$ son constantes reales.

**Desarmemos esta función [B].** Tiene dos piezas:
- **$\theta\epsilon_t$** — depende del **signo** del shock estandarizado. Si $\theta$ es negativo, un shock negativo (caída) aporta positivamente a la volatilidad.
- **$\gamma[|\epsilon_t|-E(|\epsilon_t|)]$** — depende sólo de la **magnitud**: cuánto se desvió el shock respecto de su magnitud típica. Es la pieza "simétrica", análoga a lo que hace GARCH.

Y Tsay muestra la asimetría explícitamente [A]:

$$g(\epsilon_t)=\begin{cases}(\theta+\gamma)\epsilon_t-\gamma E(|\epsilon_t|) & \text{si }\epsilon_t\ge0\\[2pt](\theta-\gamma)\epsilon_t-\gamma E(|\epsilon_t|) & \text{si }\epsilon_t<0\end{cases}$$

**Los coeficientes $(\theta+\gamma)$ y $(\theta-\gamma)$ son distintos.** Ahí está la asimetría: la pendiente de la respuesta cambia según el signo del shock.

*(Nota técnica [A]: para $\epsilon_t$ normal estándar, $E(|\epsilon_t|)=\sqrt{2/\pi}\approx0.7979$.)*

### [A] El modelo EGARCH($m,s$)

Ec. (3.25):

$$a_t=\sigma_t\epsilon_t,\qquad \ln(\sigma_t^2)=\alpha_0+\frac{1+\beta_1B+\dots+\beta_{s-1}B^{s-1}}{1-\alpha_1B-\dots-\alpha_mB^m}\,g(\epsilon_{t-1})$$

donde $B$ es el operador de rezago del Capítulo 2.

**Las dos diferencias con GARCH, según Tsay** [A]:
1. *"Usa la **varianza condicional logarítmica** para relajar la restricción de positividad de los coeficientes del modelo."*
2. *"El uso de $g(\epsilon_t)$ permite al modelo **responder asimétricamente** a valores rezagados positivos y negativos de $a_t$."*

### Por qué el logaritmo [B]

En GARCH había que imponer $\alpha_i\ge0$, $\beta_j\ge0$ para garantizar que $\sigma_t^2>0$. Al modelar $\ln(\sigma_t^2)$ en lugar de $\sigma_t^2$, el problema desaparece: **cualquier** número real, al pasarlo por la exponencial, da un valor positivo. Los coeficientes quedan libres.

Es la misma razón por la que en ML se usa una capa de salida exponencial o softplus cuando se quiere predecir una cantidad que debe ser positiva.

### [A] Ejemplo concreto: AR(1)–EGARCH(1,1) sobre IBM mensual (1926–1997)

Tsay ajusta y luego traduce el resultado a una forma muy legible:

$$\ln(\sigma_t^2)=-1.001+0.856\ln(\sigma_{t-1}^2)+\begin{cases}0.185\,\epsilon_{t-1} & \text{si }\epsilon_{t-1}\ge0\\[2pt]-0.344\,\epsilon_{t-1} & \text{si }\epsilon_{t-1}<0\end{cases}$$

**Cómo leer esto [B].** Consideremos un shock estandarizado de magnitud 1:
- Si fue **positivo** ($\epsilon_{t-1}=+1$): la contribución a $\ln\sigma_t^2$ es $+0.185$.
- Si fue **negativo** ($\epsilon_{t-1}=-1$): la contribución es $-0.344\times(-1)=+0.344$.

**Un shock negativo eleva la log-volatilidad casi el doble que un shock positivo del mismo tamaño.** Eso es el leverage effect medido: 0.344 vs 0.185.

Tsay verifica el modelo [A]: $Q(10)=6.31$ (p = 0.71) y $Q(20)=21.4$ (p = 0.32) para $\tilde a_t$; $Q(10)=4.13$ (p = 0.90) y $Q(20)=15.93$ (p = 0.66) para $\tilde a_t^2$. *"Por lo tanto, no hay correlación serial ni heterocedasticidad condicional en los residuos estandarizados del modelo ajustado."*

### [A] La forma alternativa

Ec. (3.28), que es la implementada en varios paquetes:

$$\ln(\sigma_t^2)=\alpha_0+\sum_{i=1}^{s}\alpha_i\frac{|a_{t-i}|+\gamma_ia_{t-i}}{\sigma_{t-i}}+\sum_{j=1}^{m}\beta_j\ln(\sigma_{t-j}^2)$$

*"Aquí un $a_{t-i}$ positivo contribuye $\alpha_i(1+\gamma_i)|\epsilon_{t-i}|$ a la log-volatilidad, mientras que un $a_{t-i}$ negativo da $\alpha_i(1-\gamma_i)|\epsilon_{t-i}|$. **El parámetro $\gamma_i$ significa por tanto el efecto leverage de $a_{t-i}$. Nuevamente, esperamos que $\gamma_i$ sea negativo en aplicaciones reales.**"*

### Qué dice exactamente Tsay sobre el leverage effect

Conviene ser preciso, porque es donde más riesgo hay de sobregeneralizar:

- [A] Lo lista como característica observada: *"la volatilidad **parece** reaccionar de forma distinta a una gran subida de precio o a una gran caída"* (§3.1). **Lenguaje prudente: "parece".**
- [A] *"Como los shocks negativos **tienden** a tener impactos mayores, **esperamos** que $\theta$ sea negativo"* (§3.8). **"Tienden", "esperamos": es una expectativa basada en la literatura, no una demostración.**
- [A] *"Esperamos que $\gamma_i$ sea negativo **en aplicaciones reales**"* (§3.8.1).
- [A] En el ejemplo de TGARCH sobre IBM (§3.9): *"Basándose en el modelo ajustado, **el efecto leverage es significativo al nivel del 5%**."*

**Los datos sobre los que se establece:** acciones estadounidenses individuales (IBM) e índices accionarios estadounidenses (CRSP value-weighted), a frecuencia mensual y diaria, en períodos que van de 1926 a 2003.

### [B] Adaptación a futuros — la advertencia central

**No debe asumirse que el leverage effect documentado en acciones se traslade de la misma forma a futuros.** Repitiendo y ampliando lo dicho en §3.1:

| Clase de futuro | ¿Es plausible el mecanismo de apalancamiento? | Comentario [B] |
|---|---|---|
| **Índices accionarios** (ES, NQ, RTY, YM) | Parcialmente | El subyacente son acciones, pero el mecanismo puede ser flujos de cobertura más que apalancamiento contable |
| **Tasas de interés** (ZN, ZB, ZT) | No | ¿Qué es una "caída" en un futuro de bonos? Sube el rendimiento. La relación con volatilidad depende del nivel de tasas y de la política monetaria |
| **Divisas** (6E, 6J) | No | La dirección es una convención del par. Una "caída" del EUR/USD es una subida del USD/EUR |
| **Commodities energéticos** (CL, NG) | No, y posiblemente al revés | Las crisis suelen ser de escasez: los shocks violentos tienden a ser **al alza** |
| **Metales** (GC, SI) | No claro | El oro a menudo sube en episodios de estrés, junto con la volatilidad |
| **Agrícolas** | No, y posiblemente al revés | Sequías y shocks de oferta producen subidas violentas |

**Formulación correcta [B]:** *la existencia, el signo y la magnitud de una respuesta asimétrica de la volatilidad a shocks de distinto signo son cuestiones empíricas que deben medirse por instrumento. En algunos futuros cabe esperar que el signo sea el opuesto al de acciones. Nada en el Capítulo 3 permite predecir el resultado.*

### [A] Pronóstico con EGARCH

Tsay deriva la fórmula recursiva de pronóstico a $j$ pasos y la ilustra con IBM. Los pronósticos son $\hat\sigma^2_{864}(1)=6.05\times10^{-3}$, $(2)=5.82\times10^{-3}$, $(3)=5.63\times10^{-3}$, $(10)=4.94\times10^{-3}$, y observa [A] que *"estos pronósticos convergen gradualmente a la varianza muestral $4.37\times10^{-3}$ del proceso de shock"*. **Misma reversión a la media que en GARCH.**

## 6.5 Sección 3.9 — TGARCH: un interruptor según el signo

### El problema que resuelve

El mismo que EGARCH: permitir asimetría. Pero con un mecanismo más simple y directo.

### [A] El modelo

*"Otro modelo de volatilidad comúnmente usado para manejar efectos leverage es el modelo threshold GARCH (o TGARCH)"* (Glosten, Jagannathan y Runkle, 1993; Zakoian, 1994). Ec. (3.34):

$$\sigma_t^2=\alpha_0+\sum_{i=1}^{s}(\alpha_i+\gamma_iN_{t-i})a_{t-i}^2+\sum_{j=1}^{m}\beta_j\sigma_{t-j}^2$$

donde $N_{t-i}$ es un **indicador de shock negativo**:

$$N_{t-i}=\begin{cases}1 & \text{si } a_{t-i}<0\\ 0 & \text{si } a_{t-i}\ge0\end{cases}$$

Tsay explica [A]: *"del modelo se ve que un $a_{t-i}$ positivo contribuye $\alpha_ia_{t-i}^2$ a $\sigma_t^2$, mientras que un $a_{t-i}$ negativo tiene un impacto mayor $(\alpha_i+\gamma_i)a_{t-i}^2$ con $\gamma_i>0$. **El modelo usa el cero como su umbral (threshold)** para separar los impactos de los shocks pasados. Otros valores de umbral también pueden usarse."* Se le llama también **modelo GJR**.

### Explicación en lenguaje llano [B]

Es un **interruptor**. La fórmula es la misma que la de GARCH, salvo que el coeficiente que multiplica al shock pasado tiene dos valores posibles:
- Si el shock de ayer fue positivo: se usa $\alpha_1$.
- Si el shock de ayer fue negativo: se usa $\alpha_1+\gamma_1$, que es mayor.

**Qué significa "threshold" (umbral):** el valor que separa los dos regímenes. Aquí es cero (positivo vs negativo), pero Tsay señala que podrían usarse otros.

### [A] Ejemplo: IBM mensual 1926–2003

$$\sigma_t^2=3.45\times10^{-4}+(0.0658+0.0843\,N_{t-1})\,a_{t-1}^2+0.8182\,\sigma_{t-1}^2$$

**Lectura [B]:** el coeficiente del shock pasado es **0.0658 tras una subida** y **0.0658+0.0843 = 0.1501 tras una caída**. Es decir, más del doble. Tsay verifica el modelo ($Q(12)=18.34$, p = 0.106 para $\tilde a_t$; $Q(12)=5.36$, p = 0.95 para $\tilde a_t^2$) y concluye [A] que *"el efecto leverage es significativo al nivel del 5%"*.

### EGARCH vs TGARCH — diferencia conceptual [B]

| | **EGARCH** | **TGARCH** |
|---|---|---|
| Qué modela | $\ln(\sigma_t^2)$ | $\sigma_t^2$ directamente |
| Cómo introduce asimetría | Una función continua $g(\epsilon_t)$ con pendientes distintas a cada lado del cero | Un **interruptor** que activa un coeficiente extra |
| Positividad de $\sigma_t^2$ | Automática (por la exponencial) | Requiere restricciones sobre los coeficientes |
| Restricciones sobre parámetros | Ninguna, gracias al log | Los coeficientes deben ser no negativos |
| Efecto de un shock | Multiplicativo sobre $\sigma_t^2$ | Aditivo sobre $\sigma_t^2$ |
| Interpretabilidad | Menor: hay que exponenciar para interpretar | Mayor: se lee directamente "el coeficiente se duplica tras una caída" |
| Sensibilidad a extremos | Menor: el log comprime | Mayor: usa $a_{t-1}^2$ sin comprimir |

**Ninguno es "mejor" en abstracto.** Son parametrizaciones distintas del mismo fenómeno.

## 6.6 Tabla resumen de la familia

| Modelo | Idea principal | Qué problema del anterior resuelve |
|---|---|---|
| **ARCH** | La volatilidad depende de los shocks pasados al cuadrado | Es el primero: convierte el clustering en un modelo estimable |
| **GARCH** | Shocks pasados **+ volatilidad pasada** | **Parsimonia**: captura memoria larga con pocos parámetros |
| **IGARCH** | Caso límite: persistencia exactamente 1 | Formaliza lo que se observa empíricamente ($\alpha+\beta\approx1$); es la base de RiskMetrics |
| **GARCH-M** | La volatilidad puede entrar en la **ecuación de media** | Permite que exista prima de riesgo; en los ejemplos de Tsay **no resulta significativa** |
| **EGARCH** | Respuesta **asimétrica** usando log-volatilidad | Distingue shocks positivos de negativos; elimina restricciones de positividad |
| **TGARCH (GJR)** | Coeficiente distinto según el **signo** del shock | Misma asimetría que EGARCH, con parametrización más simple e interpretable |

## 6.7 Secciones 3.10–3.11 — CHARMA y RCA (nivel conceptual)

### CHARMA — coeficientes aleatorios en el shock

**Qué limitación de GARCH intenta resolver [B].** GARCH usa una función fija y determinista de los shocks pasados al cuadrado. CHARMA introduce **coeficientes que son ellos mismos aleatorios**.

**[A] Qué dice Tsay.** Ec. (3.36):

$$r_t=\mu_t+a_t,\qquad a_t=\delta_{1t}a_{t-1}+\delta_{2t}a_{t-2}+\dots+\delta_{mt}a_{t-m}+\eta_t$$

donde $\{\eta_t\}$ es white noise gaussiano y **$\{\delta_t\}$ es una secuencia de vectores aleatorios iid de media cero** con matriz de covarianza $\Omega$, independiente de $\{\eta_t\}$. Tsay señala que *"el modelo CHARMA no es el mismo que el modelo ARCH, pero los dos modelos tienen propiedades condicionales de segundo orden similares"*.

La varianza condicional resultante (Ec. 3.37) es una **forma cuadrática** en los shocks pasados. Con $m=1$ se reduce exactamente a un ARCH(1). Con $m=2$:

$$\sigma_t^2=\sigma_\eta^2+\omega_{11}a_{t-1}^2+2\omega_{12}a_{t-1}a_{t-2}+\omega_{22}a_{t-2}^2$$

*"que difiere de un modelo ARCH(2) **por el término de producto cruzado $a_{t-1}a_{t-2}$**"*.

**Dos ventajas que Tsay señala** [A]:
1. **La positividad de $\sigma_t^2$ se satisface automáticamente**, porque $\Omega$ es una matriz de covarianza (no negativa definida) y $\sigma_\eta^2>0$.
2. Los términos cruzados *"podrían ser útiles en algunas aplicaciones. Por ejemplo, al modelar una serie de retornos, los términos de producto cruzado denotan **interacciones** entre shocks previos"*.

**Qué es un "coeficiente aleatorio" [B].** En un modelo normal, los coeficientes son números fijos que estimamos una vez. En un modelo de coeficientes aleatorios, el coeficiente **cambia en cada período**, sorteado de una distribución. La idea de fondo: *la relación entre el pasado y el presente no es siempre exactamente la misma*.

### [A] Variables explicativas en la ecuación de volatilidad (§3.10.1)

Ésta es la parte más directamente relevante para nuestro proyecto. Tsay generaliza (Ec. 3.38):

$$r_t=\mu_t+a_t,\qquad a_t=\sum_{i=1}^{m}\delta_{it}x_{i,t-1}+\eta_t$$

*"donde $\{x_{it}\}$ son $m$ variables explicativas disponibles en el tiempo $t$"*, resultando en

$$\sigma_t^2=\sigma_\eta^2+(x_{1,t-1},\dots,x_{m,t-1})\,\Omega\,(x_{1,t-1},\dots,x_{m,t-1})'$$

*"En aplicaciones, las variables explicativas pueden incluir algunos valores rezagados de $a_t$."*

**Traducción [B]:** el marco permite que **la volatilidad esperada dependa de información externa**, no sólo de la historia propia de los retornos. Ésa es la puerta formal para features exógenos en la ecuación de volatilidad.

**[A] Y Tsay lo demuestra empíricamente en §3.14**, con un ejemplo importante: modela la volatilidad del S&P 500 mensual usando **el retorno rezagado de IBM** como variable explicativa:

$$\sigma_t^2=1.069+0.148a_{t-2}^2+0.834\sigma_{t-1}^2-0.007(x_{t-1}-1.24)^2$$

y concluye [A]: *"Como el p-valor para testear $\gamma=0$ es 0.0039, **la contribución del retorno rezagado de la acción de IBM a la volatilidad del índice S&P 500 es estadísticamente significativa al nivel del 1%**."* Interpreta el signo negativo como que un retorno de IBM alejado de su media **reduce** la volatilidad estimada del índice.

**Esto es un ejemplo cross-asset genuino en el propio libro [B].** Tsay muestra que la volatilidad de un instrumento puede depender de información de otro instrumento. Es la evidencia más cercana que el capítulo ofrece a una aplicación cross-market — aunque sea entre una acción y su índice, no entre futuros.

### RCA — coeficientes aleatorios en la media

**[A]** Ec. (3.39):

$$r_t=\phi_0+\sum_{i=1}^{p}(\phi_i+\delta_{it})r_{t-i}+a_t$$

*"Clasificamos el modelo RCA como un modelo de heterocedasticidad condicional, pero **históricamente se usa para obtener una mejor descripción de la ecuación de media condicional del proceso, permitiendo que los parámetros evolucionen en el tiempo**."*

La varianza condicional resultante es cuadrática en los **valores rezagados observados** $r_{t-i}$, y Tsay señala la diferencia sutil con CHARMA [A]: *"Para el modelo RCA, la volatilidad es una función cuadrática de los **valores rezagados observados** $r_{t-i}$. Sin embargo, la volatilidad es una función cuadrática de las **innovaciones rezagadas** $a_{t-i}$ en un modelo CHARMA."*

### ¿Por qué tendría sentido imaginar coeficientes que cambian? [B]

Porque asumir un coeficiente constante equivale a asumir que **la relación entre el pasado y el presente del mercado es la misma en 1990 que en 2025, en calma que en crisis**. Eso es una hipótesis fuerte y probablemente falsa.

Un coeficiente aleatorio es una forma modesta de relajarla: en lugar de un valor fijo, un valor que fluctúa alrededor de una media. No modela **regímenes** —no dice "en crisis el coeficiente es X y en calma es Y"— sino variabilidad aleatoria.

**PREGUNTA ABIERTA:** la modelización explícita de regímenes, con estados discretos y transiciones, es materia de los **Capítulos 11–12** (state-space y Markov switching). No debe anticiparse aquí.

### [B] Relevancia para futuros — como preguntas, no como propuestas

El marco de variables explicativas en la ecuación de volatilidad [A] abre preguntas sobre qué información externa podría informar la volatilidad de un futuro:

- Volatilidad reciente de otros mercados relacionados.
- Hora del día y sesión (la volatilidad intradía de futuros tiene patrones marcados).
- Proximidad a eventos de calendario conocidos (datos macro, decisiones de bancos centrales, vencimientos).
- Volumen y actividad.
- Volatilidad implícita, si está disponible.

**Ninguna de estas se propone como feature.** Son hipótesis a examinar, y varias requieren capítulos posteriores para tratarse correctamente: la relación entre mercados es **Cap. 8**, la covarianza dinámica es **Cap. 10**, y los efectos intradía son **Cap. 5**.

---
# 7. Alternativas a GARCH

## 7.1 Sección 3.12 — Stochastic Volatility (SV)

### La diferencia fundamental con GARCH

Ésta es la distinción que hay que entender, y es más sutil de lo que parece.

**En GARCH,** una vez que conocemos los parámetros y la historia, **$\sigma_t^2$ está completamente determinada**. No hay ninguna aleatoriedad en la volatilidad: es una fórmula exacta aplicada al pasado. Toda la aleatoriedad está en $\epsilon_t$.

**En SV,** la volatilidad **tiene su propio shock aleatorio**. Aunque conociéramos los parámetros y toda la historia, no sabríamos exactamente cuál es $\sigma_t^2$: hay una fuente de incertidumbre adicional.

Tsay lo había anticipado en §3.2 [A]: los modelos de la primera categoría *"usan una función exacta para gobernar la evolución de $\sigma_t^2$"*, los de la segunda *"usan una ecuación estocástica"*.

### [A] El modelo

Ec. (3.40):

$$a_t=\sigma_t\epsilon_t,\qquad (1-\alpha_1B-\dots-\alpha_mB^m)\ln(\sigma_t^2)=\alpha_0+v_t$$

*"donde los $\epsilon_t$ son iid $N(0,1)$, los $v_t$ son iid $N(0,\sigma_v^2)$, $\{\epsilon_t\}$ y $\{v_t\}$ son independientes, $\alpha_0$ es una constante, y todos los ceros del polinomio son mayores que 1 en módulo."*

En el caso más simple ($m=1$) esto es:

$$\ln\sigma_t^2=\alpha_0+\alpha_1\ln\sigma_{t-1}^2+v_t$$

**Qué es $v_t$ [B]:** es el **shock de la volatilidad** — la sorpresa en el nivel de volatilidad misma. Es una variable aleatoria normal de media cero y varianza $\sigma_v^2$, independiente del shock del retorno. Si $\sigma_v^2=0$, el modelo colapsa a una fórmula determinista.

**Nótese que se modela $\ln\sigma_t^2$, no $\sigma_t^2$.** [A] Tsay señala la razón: *"Similar a los modelos EGARCH, **para asegurar la positividad de la varianza condicional**, los modelos SV usan $\ln(\sigma_t^2)$ en lugar de $\sigma_t^2$."*

### Interpretación intuitiva [B]

Volviendo a la analogía de la temperatura: en GARCH, la temperatura de mañana está totalmente determinada por la de hoy y por si encendimos la estufa. En SV, además hay **corrientes de aire impredecibles** que afectan la temperatura por su cuenta.

En términos de mercado: GARCH dice que la volatilidad de mañana se sigue mecánicamente de lo que pasó. SV dice que **el propio nivel de riesgo del mercado puede cambiar por razones que no vemos en los retornos** — por ejemplo, un cambio en la composición de participantes, o una acumulación de incertidumbre que aún no se manifestó en movimientos de precio.

### [A] Ventajas y desventajas

**Ventaja:** *"Agregar la innovación $v_t$ **aumenta sustancialmente la flexibilidad** del modelo para describir la evolución de $\sigma_t^2$."*

**Desventaja:** *"pero también aumenta la dificultad en la estimación de parámetros. Para estimar un modelo SV necesitamos un **método de cuasi-verosimilitud vía filtro de Kalman** o un **método de Monte Carlo**."* Y da la razón de fondo [A]: *"La dificultad para estimar un modelo SV es comprensible porque **para cada shock $a_t$ el modelo usa dos innovaciones $\epsilon_t$ y $v_t$**."*

**Por qué eso complica todo [B].** En GARCH, dado el pasado y los parámetros, podemos calcular $\sigma_t^2$ exactamente, y por tanto podemos escribir la verosimilitud directamente. En SV, $\sigma_t^2$ es una **variable latente que nunca observamos**, y calcular la verosimilitud requiere integrar sobre todos sus valores posibles — un problema computacionalmente duro. De ahí Kalman y MCMC.

**Sobre Kalman y MCMC [B], sin profundizar:** son técnicas para trabajar con variables latentes. El filtro de Kalman produce una estimación secuencial y aproximada; MCMC es un método de muestreo que explora la distribución de las cantidades no observadas. Tsay remite explícitamente al **Capítulo 12** para MCMC. **PREGUNTA ABIERTA hasta ese capítulo.**

### [A] El resultado empírico más importante de la sección

> *"La experiencia limitada muestra que los modelos SV a menudo **proveyeron mejoras en el ajuste del modelo**, pero **sus contribuciones a los pronósticos de volatilidad fuera de muestra recibieron resultados mixtos**."*

### La distinción in-sample vs out-of-sample [B]

Ésta es la lección metodológica de la sección y trasciende el modelo concreto.

- **Buen ajuste in-sample** significa que el modelo describe bien los datos con los que fue estimado. Un modelo más flexible **siempre** puede ajustar mejor: tiene más parámetros o más libertad.
- **Buen pronóstico out-of-sample** significa que el modelo funciona sobre datos que no vio.

**Que un modelo más flexible mejore el ajuste y no el pronóstico es exactamente el fenómeno del sobreajuste.** Y Tsay lo constata en un caso concreto y publicado, con un modelo estadístico clásico de pocos parámetros — no con una red neuronal de millones.

**Implicancia para el proyecto [B]:** si un modelo con dos fuentes de aleatoriedad y estimación por MCMC no logra mejorar consistentemente los pronósticos de volatilidad respecto de un GARCH(1,1) de tres parámetros, es razonable ser escéptico ante la idea de que más complejidad produzca automáticamente mejores pronósticos. **No es un argumento contra la complejidad**; es un argumento a favor de exigirle evidencia out-of-sample.

## 7.2 Sección 3.13 — Long-Memory Stochastic Volatility (LMSV)

### La conexión con el Capítulo 2

En el Capítulo 2, §2.11, Tsay mostró que la ACF de $|r_t|$ de índices diarios es **significativa incluso después de 300 lags**, con decaimiento polinomial. Ahora retoma exactamente eso.

### [A] Qué dice Tsay

*"La extensión a modelos de memoria larga en el estudio de volatilidad está motivada por el hecho de que **la función de autocorrelación de la serie al cuadrado o en valor absoluto del retorno de un activo a menudo decae lentamente, aunque la serie de retornos no tenga correlación serial**"* (referencia: Ding, Granger y Engle, 1993).

**Nueva evidencia empírica en este capítulo** [A], Figura 3.10: la ACF muestral de los **retornos absolutos diarios** de IBM y del S&P 500, desde el 3 de julio de 1962 hasta el 31 de diciembre de 2003. *"Estas ACF muestrales son positivas con magnitud moderada pero decaen lentamente."* El gráfico llega hasta el lag 200.

**El modelo LMSV** (Ec. 3.41):

$$a_t=\sigma_t\epsilon_t,\qquad \sigma_t=\sigma\exp(u_t/2),\qquad (1-B)^d u_t=\eta_t$$

con $\sigma>0$, $\epsilon_t$ iid $N(0,1)$, $\eta_t$ iid $N(0,\sigma_\eta^2)$ e independiente de $\epsilon_t$, y **$0<d<0.5$**.

*"La característica de memoria larga proviene de la **diferencia fraccional $(1-B)^d$**, que implica que la ACF de $u_t$ decae lentamente a tasa hiperbólica, en lugar de exponencial, a medida que el lag aumenta."*

**Una transformación reveladora** [A]:

$$\ln(a_t^2)=\mu+u_t+e_t$$

*"Así, la serie $\ln(a_t^2)$ es **una señal gaussiana de memoria larga más un white noise no gaussiano**"* (Breidt, Crato y de Lima, 1998).

**Estimación** [A]: *"La estimación del modelo LMSV es complicada, pero el parámetro de diferencia fraccional $d$ puede estimarse usando un método de cuasi-máxima verosimilitud o un método de regresión."*

**Un dato empírico concreto** [A]: usando la serie logarítmica de retornos diarios al cuadrado de compañías del S&P 500, Bollerslev y Jubinski (1999) y Ray y Tsay (2000) *"encontraron que la **estimación mediana de $d$ es aproximadamente 0.38**"*.

Y una observación cross-asset [A]: Ray y Tsay (2000) *"estudiaron componentes comunes de memoria larga en volatilidades diarias de acciones agrupadas por diversas características. Encontraron que **las compañías en el mismo sector industrial o de negocios tienden a tener más componentes comunes de memoria larga**"* (por ejemplo, grandes bancos e instituciones financieras estadounidenses).

### ¿Qué significa "memoria larga" en volatilidad? [B]

Recordando el Capítulo 2, hay tres regímenes de decaimiento de la ACF:

| Tipo de memoria | Cómo decae la ACF | Ejemplo |
|---|---|---|
| **Memoria corta** | Exponencialmente: cae rápido y muere | Un ARMA estacionario |
| **Memoria larga** | Polinomialmente: cae lento y sigue detectable durante cientos de lags | La ACF de $\|r_t\|$ |
| **Raíz unitaria** | No decae: la ACF muestral tiende a 1 | Un log-precio |

**Memoria larga en volatilidad significa que un episodio de alta volatilidad deja una huella detectable durante muchísimo tiempo** — no décadas literalmente, pero sí un orden de magnitud más que lo que un GARCH estándar predice.

### GARCH de persistencia muy alta vs verdadera memoria larga [B]

Ésta es una distinción importante y sutil.

- **Un GARCH(1,1) con $\alpha_1+\beta_1=0.99$** tiene decaimiento **exponencial**: el efecto de un shock se multiplica por 0.99 cada período. Es lento, pero es exponencial. A 500 períodos, el efecto es $0.99^{500}\approx0.007$: prácticamente cero.
- **Un proceso de memoria larga con $d=0.38$** tiene decaimiento **polinomial**: $\rho_k\sim k^{2d-1}=k^{-0.24}$. A 500 lags, eso es $500^{-0.24}\approx0.23$ del valor inicial. **Treinta veces más que el GARCH.**

**Son formas funcionales distintas de la persistencia**, y la diferencia se vuelve enorme en horizontes largos aunque sea casi invisible en horizontes cortos. Ésta es probablemente una de las razones por las que $\hat\alpha+\hat\beta$ sale tan cerca de 1: **un GARCH forzado a aproximar memoria larga con decaimiento exponencial elige el decaimiento exponencial más lento posible.** *(Esta interpretación específica es [B]; Tsay no la enuncia, aunque sí señala que la causa de la persistencia "merece una investigación cuidadosa".)*

### Cómo entra la diferenciación fraccional [B]

Es exactamente la herramienta del Capítulo 2, §2.11. El parámetro $d$ permite un continuo entre "sin memoria" ($d=0$) y "raíz unitaria" ($d=1$). Con $0<d<0.5$ tenemos un proceso estacionario con memoria que decae lentamente.

**La diferencia con el uso del Capítulo 2 [B]:** allí se planteaba aplicar diferenciación fraccional a la **serie de precios o de retornos**. Aquí se aplica a la **volatilidad latente** $u_t$. Es un uso distinto del mismo aparato matemático.

### Preguntas abiertas

- ¿Es la memoria larga en volatilidad un fenómeno genuino, o es un artefacto de cambios de régimen no modelados? Es la misma pregunta que planteamos sobre IGARCH, y Tsay la deja abierta. → **Caps. 11–12.**
- ¿La estimación mediana de $d\approx0.38$ en acciones del S&P 500 se traslada a futuros? **No hay evidencia en el capítulo.** Es una hipótesis a medir.
- ¿Vale la pena la complejidad adicional de LMSV frente a un GARCH bien especificado, en términos de pronóstico out-of-sample? Dado lo que Tsay dice sobre SV en general, **es una pregunta abierta y no una conclusión favorable**.

## 7.3 Sección 3.15.1 — Volatilidad realizada: medir en lugar de inferir

### El cambio de enfoque

Todo lo anterior —ARCH, GARCH, SV— **infiere** una volatilidad latente a partir de la dinámica de los retornos, con un modelo y parámetros estimados.

**La volatilidad realizada hace algo distinto: la mide directamente**, usando el movimiento observado dentro del período.

### La idea intuitiva

> Si durante un día tengo muchas observaciones intradía, puedo sumar los movimientos al cuadrado y usar esa suma como estimación de cuánto se movió realmente el mercado ese día.

No hay modelo. No hay parámetros. Sólo una suma.

### [A] La definición

Tsay parte del caso mensual: si $r_t^m$ es el log-retorno mensual y hay $n$ días de trading en el mes, entonces por la aditividad de los log-retornos (Capítulo 1):

$$r_t^m=\sum_{i=1}^{n}r_{t,i}$$

Y por tanto (Ec. 3.48):

$$\text{Var}(r_t^m\mid\mathcal{F}_{t-1})=\sum_{i=1}^{n}\text{Var}(r_{t,i}\mid\mathcal{F}_{t-1})+2\sum_{i<j}\text{Cov}[(r_{t,i},r_{t,j})\mid\mathcal{F}_{t-1}]$$

Si se supone que los retornos diarios son white noise, los términos de covarianza desaparecen y queda (Ec. 3.49):

$$\hat\sigma_m^2=\frac{n}{n-1}\sum_{i=1}^{n}(r_{t,i}-\bar r_t)^2$$

Si se supone que siguen un MA(1), hay un término de corrección adicional (Ec. 3.50).

**Y luego generaliza a la versión intradía** [A]: sea $r_t$ el log-retorno diario y supongamos que hay $n$ log-retornos intradía igualmente espaciados tales que $r_t=\sum_{i=1}^{n}r_{t,i}$. La cantidad

$$RV_t=\sum_{i=1}^{n}r_{t,i}^2$$

*"se denomina **volatilidad realizada** de $r_t$"* (Andersen et al., 2001a,b). *"Matemáticamente, la volatilidad realizada es una **variación cuadrática** de $r_t$, y **asume que $\{r_{t,i}\}_{i=1}^{n}$ forma una secuencia iid con media cero y varianza finita**."*

**Un dato práctico** [A]: *"La experiencia limitada indica que $\ln(RV_t)$ a menudo sigue aproximadamente un modelo ARIMA(0,1,q) gaussiano, que puede usarse para producir pronósticos."*

### Qué significa cada símbolo

| Símbolo | Qué es |
|---|---|
| $RV_t$ | La volatilidad realizada del período $t$ (día, típicamente) |
| $n$ (o $M$) | Cuántas observaciones intradía hay dentro del período |
| $r_{t,i}$ | El $i$-ésimo retorno intradía del período $t$ |
| $r_{t,i}^2$ | Ese retorno al cuadrado: su magnitud sin signo |

**Nota terminológica [B]:** estrictamente, $RV_t$ es una **varianza** realizada; la volatilidad realizada sería $\sqrt{RV_t}$. Tsay usa "realized volatility" para la suma, como es habitual en la literatura. Conviene tener presente la ambigüedad al leer otras fuentes.

### Por qué más datos intradía pueden mejorar la estimación [B]

Volvemos al problema del principio: con **una** observación no hay dispersión que medir. Con **78** observaciones de 5 minutos en una sesión, sí la hay.

En términos estadísticos: cada $r_{t,i}^2$ es un estimador ruidoso de la varianza de ese intervalo. Al sumar muchos, **el ruido se promedia** y la estimación mejora. Es el mismo principio por el que medir a 1,000 personas da una estimación mejor de la altura promedio que medir a una.

### [A] Los tres problemas — y son serios

**Problema 1: microstructure noise.** Textualmente: *"Intuitivamente, uno querría usar tanta información como sea posible eligiendo un $n$ grande. **Sin embargo, cuando el intervalo temporal entre $r_{t,i}$ es pequeño, los retornos están sujetos a los efectos de la microestructura de mercado, por ejemplo el bid–ask bounce, lo que a menudo resulta en una estimación sesgada de la volatilidad.**"*

**Explicación en lenguaje llano [B].** A frecuencias muy altas, el precio observado no refleja sólo el valor del activo: refleja también el mecanismo de negociación. Si las operaciones se ejecutan alternadamente contra el bid y contra el ask, el precio "rebota" entre dos niveles aunque el valor no cambie. Ese rebote es movimiento observado que **no es volatilidad genuina**, y al sumar cuadrados, se suma también. **Cuanta mayor la frecuencia, mayor la proporción de ruido.**

Hay entonces una **tensión fundamental**: más observaciones reducen el error estadístico, pero aumentan la contaminación por microestructura.

**Problema 2: la frecuencia óptima.** [A] *"El problema de elegir un intervalo temporal óptimo para construir la volatilidad realizada ha atraído mucha investigación recientemente. **Para activos muy negociados en Estados Unidos, a menudo se usa un intervalo de 4–15 minutos.**"*

**Nota importante [B]:** ese rango de 4–15 minutos es una regla práctica reportada por Tsay **para activos estadounidenses muy negociados** — no para futuros específicamente, no para instrumentos menos líquidos, y no como resultado derivado. La frecuencia apropiada depende de la liquidez y de la microestructura del instrumento concreto. **PREGUNTA ABIERTA → Cap. 5.**

**Problema 3: el retorno overnight.** [A] *"Otro problema de usar volatilidad realizada para retornos de acciones es que **el retorno overnight, que es el retorno desde el precio de cierre del día $t-1$ hasta el precio de apertura de $t$, tiende a ser sustancial. Ignorar los retornos overnight puede subestimar seriamente la volatilidad.**"*

Pero añade un matiz relevante [A]: *"Por otro lado, nuestra experiencia limitada muestra que **los retornos overnight parecen ser pequeños para retornos de índices o de tipos de cambio**."*

Esto conecta con lo que dijo en §3.1 [A]: la volatilidad consiste en volatilidad intradía **y** overnight, y *"los retornos intradía de alta frecuencia contienen sólo información muy limitada sobre la volatilidad overnight"*.

### [B] Relevancia para futuros — con cautela

**Lo que probablemente juegue a favor.** Muchos futuros líquidos operan casi 24 horas, lo que reduce el problema del gap overnight comparado con acciones: el "cierre" y la "apertura" están separados por una pausa breve (típicamente 60 minutos en CME), no por 17.5 horas. Esto sugiere que la volatilidad realizada podría ser una medida más completa en futuros que en acciones.

**Pero esto es una expectativa, no un resultado.** Y viene con complicaciones propias:
- La actividad dentro de las 24 horas es **extremadamente heterogénea**. Sumar cuadrados de 5 minutos durante la madrugada asiática y durante la apertura de Nueva York mezcla regímenes de liquidez radicalmente distintos.
- El **gap de fin de semana** sigue existiendo y puede ser sustancial.
- El **roll de contratos** introduce discontinuidades que no son movimientos de mercado.
- El **tick size** limita la resolución: si a 5 segundos el precio se mueve típicamente 0 o 1 ticks, la suma de cuadrados está midiendo la discreción de la grilla más que la volatilidad.

**Nada de esto está en el Capítulo 3.** Todo requiere el **Cap. 5** para tratarse correctamente. → **PREGUNTA ABIERTA.**

### [A] Un dato del ejemplo 3.6 que conviene retener

Tsay compara tres estimaciones de la volatilidad mensual del S&P 500 (1980–1999): dos basadas en retornos diarios y una basada en un GARCH(1,1) sobre retornos mensuales. Observa: *"**Claramente las volatilidades estimadas basadas en retornos diarios son mucho más altas que las basadas en retornos mensuales y un modelo GARCH(1,1).** En particular, la volatilidad estimada para octubre de 1987 fue de aproximadamente 680 cuando se usan retornos diarios."*

**Lectura [B]:** distintos métodos de estimar "la volatilidad del mes" dan resultados que difieren **sustancialmente**, no marginalmente. Esto refuerza el punto de §3.5.2: **la elección del método de medición no es un detalle técnico**; cambia el número que se obtiene y por tanto cualquier decisión basada en él.

## 7.4 Sección 3.15.2 — Estimadores basados en Open, High, Low, Close

### El problema que resuelven

Si tenemos barras OHLC pero no datos tick a tick, **el cierre-a-cierre desperdicia información**. El máximo y el mínimo del período contienen información sobre cuánto se movió el precio *durante* el período, no sólo dónde terminó.

### La intuición

Imaginemos dos días:
- **Día A:** abre en 100, sube a 105, baja a 95, cierra en 100.
- **Día B:** abre en 100, oscila entre 99.8 y 100.2, cierra en 100.

**Cierre a cierre, ambos días tuvieron retorno exactamente cero.** Un estimador basado sólo en el cierre les asigna la misma volatilidad: cero.

Pero el día A fue violentísimo y el B fue plano. **El rango (máximo − mínimo) captura esa diferencia inmediatamente**: 10 puntos vs 0.4 puntos.

### [A] Las definiciones de Tsay

$$C_t=\text{cierre},\quad O_t=\text{apertura},\quad H_t=\text{máximo},\quad L_t=\text{mínimo}$$
$$f=\text{fracción del día (en [0,1]) en que el mercado está cerrado}$$

**La varianza convencional** [A]: $\sigma_t^2=E[(C_t-C_{t-1})^2\mid\mathcal{F}_{t-1}]$.

Garman y Klass (1980) consideraron varios estimadores **suponiendo que el precio sigue un modelo de difusión simple sin drift**. Tsay lista [A]:

| Estimador | Fórmula | Qué información usa |
|---|---|---|
| $\hat\sigma_{0,t}^2$ | $(C_t-C_{t-1})^2$ | **Sólo el cierre.** El punto de referencia |
| $\hat\sigma_{1,t}^2$ | $\dfrac{(O_t-C_{t-1})^2}{2f}+\dfrac{(C_t-O_t)^2}{2(1-f)}$ | Separa el movimiento overnight del intradía |
| $\hat\sigma_{2,t}^2$ | $\dfrac{(H_t-L_t)^2}{4\ln 2}\approx0.3607(H_t-L_t)^2$ | **Sólo el rango.** Éste es el estimador de **Parkinson (1980)** con $f=0$ |
| $\hat\sigma_{3,t}^2$ | $0.17\dfrac{(O_t-C_{t-1})^2}{f}+0.83\dfrac{(H_t-L_t)^2}{(1-f)4\ln 2}$ | Rango + gap overnight |
| $\hat\sigma_{5,t}^2$ | $0.5(H_t-L_t)^2-0.386(C_t-O_t)^2$ | Rango + movimiento neto de la sesión |
| $\hat\sigma_{6,t}^2$ | $0.12\dfrac{(O_t-C_{t-1})^2}{f}+0.88\dfrac{\hat\sigma_{5,t}^2}{1-f}$ | Todo lo anterior combinado |

### [A] La eficiencia — el resultado cuantitativo clave

Tsay define el **factor de eficiencia** como

$$\text{Eff}(\hat\sigma_{i,t}^2)=\frac{\text{Var}(\hat\sigma_{0,t}^2)}{\text{Var}(\hat\sigma_{i,t}^2)}$$

y reporta [A]: *"Garman y Klass (1980) encontraron que $\text{Eff}(\hat\sigma_{i,t}^2)$ es aproximadamente **2, 5.2, 6.2, 7.4 y 8.4** para $i=1,2,3,5$ y 6, respectivamente, **para el modelo de difusión simple considerado**."*

**Qué significa esto en lenguaje llano [B].** Un factor de eficiencia de 5.2 significa que el estimador de Parkinson tiene aproximadamente **una quinta parte de la varianza** del estimador cierre-a-cierre. Dicho de otro modo: **con un solo día de datos OHLC se obtiene aproximadamente la misma precisión que con cinco días de datos de cierre.**

Ésta es una ganancia enorme, obtenida sin datos intradía, sólo usando información que ya está en cualquier barra OHLC.

**Advertencia crítical [A+B]:** esos factores se derivan **bajo el supuesto del modelo de difusión simple sin drift** que Garman y Klass entretienen. Tsay lo dice explícitamente. **No son garantías universales.**

### [A] Rogers–Satchell y Yang–Zhang

Tsay pasa a log-retornos y define:

$$o_t=\ln(O_t)-\ln(C_{t-1})\ \text{(apertura normalizada)}$$
$$u_t=\ln(H_t)-\ln(O_t)\ \text{(máximo normalizado)}$$
$$d_t=\ln(L_t)-\ln(O_t)\ \text{(mínimo normalizado)}$$
$$c_t=\ln(C_t)-\ln(O_t)\ \text{(cierre normalizado)}$$

**El estimador de Yang–Zhang (2000)**, que Tsay presenta como *"un estimador robusto de la volatilidad"* suponiendo $n$ días de datos y volatilidad constante en el período:

$$\hat\sigma_{yz}^2=\hat\sigma_o^2+k\,\hat\sigma_c^2+(1-k)\,\hat\sigma_{rs}^2$$

donde:
- $\hat\sigma_o^2$ es la varianza muestral de las **aperturas normalizadas** $o_t$ → captura el **gap overnight**;
- $\hat\sigma_c^2$ es la varianza muestral de los **cierres normalizados** $c_t$ → captura el movimiento neto de la sesión;
- $\hat\sigma_{rs}^2=\frac1n\sum[u_t(u_t-c_t)+d_t(d_t-c_t)]$ es el estimador de **Rogers–Satchell (1991)** → usa máximo y mínimo;
- $k=\dfrac{0.34}{1.34+(n+1)/(n-1)}$, elegido *"para minimizar la varianza del estimador"*.

### Tabla comparativa de estimadores

| Estimador | Información que usa | Ventaja que busca | Supuesto que requiere | Problema | ¿Maneja overnight? |
|---|---|---|---|---|---|
| **Cierre a cierre** | Sólo $C_t, C_{t-1}$ | Simplicidad; ninguna información se pierde en el sentido de que es el retorno real | Ninguno especial | **Muy ineficiente**: ignora todo el movimiento intraperíodo | Sí, implícitamente (el gap está en el retorno) |
| **Parkinson** | $H_t, L_t$ | Usar el rango; ~5× más eficiente | Difusión sin drift, sin gaps | **Ignora completamente el gap overnight**; sensible a que el rango observado subestime el verdadero | **No** |
| **Garman–Klass** | $O,H,L,C$ + $f$ | Combinar rango y gap; hasta ~8× más eficiente | Difusión sin drift | Requiere conocer $f$; sensible al supuesto de difusión | Sí, con el término de apertura |
| **Rogers–Satchell** | $O,H,L,C$ | **Ser robusto a la presencia de drift** — a diferencia de Parkinson y GK | Difusión, permite drift | No maneja gaps por sí solo | **No** |
| **Yang–Zhang** | $O,H,L,C$ de $n$ días | Combinar overnight + sesión + rango, minimizando varianza | Volatilidad constante en el período de $n$ días | Más complejo; el peso $k$ depende de $n$; requiere una ventana | **Sí**, explícitamente vía $\hat\sigma_o^2$ |

*(La caracterización de Rogers–Satchell como robusto al drift no está enunciada así por Tsay; es conocimiento estándar sobre ese estimador y se marca como [B].)*

### [A] Tick size y frecuencia de negociación — la advertencia final

Ésta es una advertencia que Tsay hace explícitamente y que es directamente relevante para futuros:

> *"El rango $H_t-L_t$ se denomina el **rango** del precio en el día $t$. Este estimador ha llevado al uso de estimaciones de volatilidad basadas en rango. **En la práctica, los precios de las acciones sólo se observan en puntos temporales discretos. Como tal, el máximo diario observado es probablemente menor que $H_t$ y el mínimo diario observado es probablemente mayor que $L_t$. Consecuentemente, el rango de precio diario observado tiende a subestimar el rango real y, por tanto, puede llevar a subestimar la volatilidad. Este sesgo en la estimación de volatilidad depende de la frecuencia de negociación y del tick size de las acciones. Para acciones intensamente negociadas, el sesgo debería ser despreciable. Para otras acciones, se necesita más estudio.**"*

**Explicación en lenguaje llano [B].** El máximo verdadero del día es el punto más alto que el precio "habría alcanzado" si se observara continuamente. Pero sólo vemos las transacciones que efectivamente ocurrieron. Si un instrumento opera poco, el precio pudo haber pasado por un nivel alto sin que ninguna transacción lo registrara. **El rango observado es siempre menor o igual al rango verdadero**, y por tanto el estimador subestima sistemáticamente.

**Cuánto importa depende de:**
- **La frecuencia de negociación:** cuantas más transacciones por día, más probable es que el verdadero extremo quede registrado.
- **El tick size:** si el tick es grueso respecto del movimiento típico, el precio se mueve en escalones y el rango observado está cuantizado.

### [B] Relevancia para futuros — con las condiciones explícitas

**Lo que juega a favor.** Los futuros líquidos de índices son de los instrumentos más negociados del mundo, con miles de transacciones por minuto en la sesión activa. Según el propio criterio de Tsay —*"para acciones intensamente negociadas, el sesgo debería ser despreciable"*— cabría esperar que el sesgo sea pequeño en esos contratos durante la sesión activa.

**Lo que juega en contra, y no está en Tsay:**
- **Durante las sesiones de baja actividad** (madrugada, festivos), la frecuencia de negociación cae drásticamente y el sesgo puede volverse relevante.
- **En futuros menos líquidos** (contratos lejanos, mercados de nicho) el problema es mayor.
- **El tick size relativo** varía mucho entre contratos: hay que medir $\text{tick}/P$ por instrumento, como establecimos en el Capítulo 1.
- **Los gaps de roll** contaminan el máximo y el mínimo si la barra abarca un cambio de contrato.
- **La definición del "día"** en un mercado de 24 horas es una convención. ¿Cuál es el máximo del día si la sesión empieza a las 18:00 del día anterior? Los estimadores OHLC dependen de esa convención.

**Ninguna conclusión sobre qué estimador usar.** Lo que queda establecido es que:
1. **Las barras OHLC contienen información sustancialmente más rica que los cierres** — con factores de eficiencia de 5 a 8 bajo los supuestos de Garman y Klass.
2. Los estimadores difieren en **qué componentes capturan** (rango, gap, drift) y en **qué supuestos requieren**.
3. La cuestión del **gap overnight** es la que más los diferencia, y en futuros de 24 horas su relevancia es distinta que en acciones.
4. Todo depende de la liquidez y el tick size, que varían por instrumento y por hora del día.

**PREGUNTA ABIERTA:** ¿qué estimador es apropiado para cada futuro y cada frecuencia? Requiere evidencia empírica y, para el problema de discreción, el **Cap. 5**.

---

# 8. Heavy tails y volatilidad: la respuesta a la pregunta del Capítulo 1

## 8.1 La pregunta que quedó abierta

En el informe del Capítulo 1 dejamos planteada, como la hipótesis más valiosa a resolver, esta pregunta:

> **¿Los retornos tienen colas pesadas porque los shocks básicos tienen colas pesadas, o porque la volatilidad cambia en el tiempo?**

O, en la formulación que usamos entonces: *¿cuánta de la excess kurtosis marginal desaparece cuando estandarizamos por volatilidad?*

**El Capítulo 3 responde la parte teórica de esa pregunta.** La parte empírica sigue siendo una hipótesis a comprobar.

## 8.2 Recordatorio: qué es la kurtosis

**Kurtosis** es una medida de cuán pesadas son las colas de una distribución — es decir, con qué frecuencia aparecen valores muy alejados de la media.

- Una distribución **normal** tiene kurtosis exactamente **3**.
- Se define el **exceso de kurtosis** = kurtosis − 3.
- **Exceso positivo** = colas más pesadas que la normal = **los eventos extremos son más frecuentes** de lo que la campana de Gauss predice.

En el Capítulo 1 vimos excesos de kurtosis de 9.92 (IBM diario), 22.81 (S&P 500 diario) y 55.25 (Citigroup diario). Enormes.

## 8.3 Los tres caminos hacia un retorno extremo

Partimos de la ecuación fundamental:

$$a_t=\sigma_t\,\epsilon_t$$

Un retorno extremo —digamos, un movimiento de −7% en un día— puede producirse de tres formas:

**Caso A — El dado salió extremo.** $\sigma_t$ era normal (digamos 1%), pero $\epsilon_t$ resultó ser −7. Es decir, un shock estandarizado de siete desviaciones estándar. Bajo normalidad, esto es astronómicamente improbable (del orden de una vez cada 10²⁰ observaciones).

**Caso B — La escala ya era alta.** $\epsilon_t$ fue perfectamente ordinario (digamos −2, algo que ocurre con regularidad), pero $\sigma_t$ ya estaba en 3.5% porque estábamos en medio de una crisis. Resultado: $3.5\%\times(-2)=-7\%$. **Nada extraordinario ocurrió en el dado; simplemente estábamos en un régimen de alta volatilidad.**

**Caso C — Ambas cosas.** $\sigma_t$ elevada y $\epsilon_t$ moderadamente grande.

## 8.4 Por qué mezclar regímenes genera colas pesadas

Ésta es la intuición central, y merece un ejemplo concreto.

**Supongamos un mercado con dos regímenes** [B, ejemplo ilustrativo propio]:
- **95% del tiempo:** régimen tranquilo, $\sigma=0.5\%$.
- **5% del tiempo:** régimen agitado, $\sigma=3\%$.

Y supongamos que **$\epsilon_t$ es perfectamente normal** en ambos regímenes. Nada raro en el dado.

Ahora preguntemos: ¿con qué frecuencia veremos un movimiento de más de 3%?

- **En el régimen tranquilo:** 3% son 6 desviaciones estándar ($3/0.5$). Bajo normalidad, prácticamente nunca.
- **En el régimen agitado:** 3% es 1 desviación estándar ($3/3$). Ocurre en cerca de un tercio de los días de ese régimen.

**Combinando:** si el 5% del tiempo estamos en el régimen agitado y en él un tercio de los días superan el 3%, entonces globalmente veremos movimientos de +3% en aproximadamente el **1.7%** de los días.

**Y ahora comparemos con lo que predeciría una normal única.** La volatilidad promedio de esta mezcla es aproximadamente $\sqrt{0.95\times0.5^2+0.05\times3^2}\approx0.94\%$. Una normal con esa desviación estándar predice movimientos de 3% (≈3.2σ) en aproximadamente el **0.14%** de los días.

**1.7% observado contra 0.14% predicho: doce veces más eventos extremos**, y sin que ningún shock estandarizado haya sido raro.

**Ésa es la mecánica completa.** Mezclar períodos de baja y alta volatilidad produce una distribución marginal con exceso de kurtosis **aunque cada período individual sea perfectamente normal**.

**Conexión con el Capítulo 1 [B]:** esto es exactamente la "mixtura de escala de normales" que Tsay presentaba en §1.2.2 como una de las cuatro familias de distribuciones candidatas. En aquel momento notamos que era conceptualmente el puente hacia GARCH. Ahora vemos el puente completo: **GARCH es una mixtura de escala donde la escala es predecible desde el pasado.**

## 8.5 [A] Qué demuestra Tsay

No es sólo intuición: Tsay lo deriva formalmente en dos lugares.

**En §3.4.1, para ARCH(1) con $\epsilon_t$ normal:**

$$\frac{E(a_t^4)}{[\text{Var}(a_t)]^2}=3\,\frac{1-\alpha_1^2}{1-3\alpha_1^2}\;>\;3$$

Y concluye [A]: *"el exceso de kurtosis de $a_t$ es positivo y la distribución de cola de $a_t$ es más pesada que la de una distribución normal. En otras palabras, **el shock $a_t$ de un modelo ARCH(1) condicionalmente gaussiano es más propenso que una serie de white noise gaussiano a producir 'outliers'**."*

**En §3.5, para GARCH(1,1):** *"puede demostrarse que si $1-2\alpha_1^2-(\alpha_1+\beta_1)^2>0$, entonces"*

$$\frac{E(a_t^4)}{[E(a_t^2)]^2}=\frac{3[1-(\alpha_1+\beta_1)^2]}{1-(\alpha_1+\beta_1)^2-2\alpha_1^2}>3$$

*"Consecuentemente, similar a los modelos ARCH, la distribución de cola de un proceso GARCH(1,1) es más pesada que la de una distribución normal."*

**La respuesta a la pregunta del Capítulo 1, entonces, es: NO, las colas pesadas de los retornos no requieren que los shocks básicos tengan colas pesadas.** La volatilidad variable las genera por sí sola.

## 8.6 Sección 3.16 — El resultado completo sobre kurtosis

### Por qué esta sección importa

Tsay la abre con una frase que vale la pena citar [A]: *"**La incertidumbre en la estimación de volatilidad es un tema importante, pero a menudo se pasa por alto.** Para evaluar la variabilidad de una volatilidad estimada, uno debe considerar la kurtosis de un modelo de volatilidad."*

### [A] El planteo

Se considera un GARCH(1,1) donde $\{\epsilon_t\}$ es iid con

$$E(\epsilon_t)=0,\qquad \text{Var}(\epsilon_t)=1,\qquad E(\epsilon_t^4)=K_\epsilon+3$$

**donde $K_\epsilon$ es el exceso de kurtosis de la innovación estandarizada $\epsilon_t$.**

- Si $\epsilon_t$ es **normal**, entonces $K_\epsilon=0$.
- Si $\epsilon_t$ es **Student-t estandarizada** con $v$ grados de libertad, entonces [A] $K_\epsilon=6/(v-4)$ para $v>4$.

### [A] El caso gaussiano

Cuando $\epsilon_t$ es normal ($K_\epsilon=0$), el exceso de kurtosis del shock resulta:

$$K_a^{(g)}=\frac{6\alpha_1^2}{1-2\alpha_1^2-(\alpha_1+\beta_1)^2}$$

Tsay señala **dos implicaciones importantes** [A]:

> **(a) La kurtosis de $a_t$ existe si $1-2\alpha_1^2-(\alpha_1+\beta_1)^2>0$.**
>
> **(b) Si $\alpha_1=0$, entonces $K_a^{(g)}=0$, lo que significa que el modelo GARCH(1,1) correspondiente NO tiene colas pesadas.**

### La lectura del punto (b) [B] — es más importante de lo que parece

**El parámetro que genera las colas pesadas es $\alpha_1$, no $\beta_1$.**

Si $\alpha_1=0$, el modelo se reduce a $\sigma_t^2=\alpha_0+\beta_1\sigma_{t-1}^2$, que es una recursión **determinista** que converge a una constante: la volatilidad deja de depender de los shocks y por tanto deja de variar aleatoriamente. Sin variación aleatoria en la escala, no hay mixtura, y sin mixtura no hay colas pesadas.

Tsay lo resume [A]: *"el coeficiente $\alpha_1$ juega un papel crítico en determinar el comportamiento de cola de $a_t$. Si $\alpha_1=0$, entonces $K_a^{(g)}=0$ y $K_a=K_\epsilon$. En este caso, **el comportamiento de cola de $a_t$ es similar al del ruido estandarizado $\epsilon_t$**. Sin embargo, si $\alpha_1>0$, entonces $K_a^{(g)}>0$ y **el proceso $a_t$ tiene colas pesadas**."*

**En términos de la analogía de la habitación:** es la **estufa** ($\alpha$) la que genera las colas pesadas, no la **aislación** ($\beta$). La aislación hace que la volatilidad persista, pero es la reacción a los shocks la que hace que la escala varíe de forma impredecible, y esa variación es la que produce los extremos.

### [A] El caso general (no gaussiano)

Cuando $\epsilon_t$ tiene sus propias colas pesadas, Tsay presenta un resultado atribuido a George C. Tiao:

$$K_a=\frac{K_\epsilon+K_a^{(g)}+\tfrac56 K_\epsilon K_a^{(g)}}{1-\tfrac16 K_\epsilon K_a^{(g)}}$$

*"Vale para todos los modelos GARCH siempre que la kurtosis exista."*

**Cómo leer esta fórmula [B].** El exceso de kurtosis total del retorno tiene tres contribuciones:
1. **$K_\epsilon$** — las colas pesadas que trae el propio shock estandarizado.
2. **$K_a^{(g)}$** — las colas pesadas generadas por la volatilidad variable, aunque el shock fuera normal.
3. **Un término de interacción** — las dos fuentes se refuerzan mutuamente; no simplemente se suman.

Y el denominador $1-\tfrac16 K_\epsilon K_a^{(g)}$ **puede acercarse a cero**, en cuyo caso $K_a$ explota. Es decir: hay combinaciones de parámetros para las que la kurtosis teórica del modelo es infinita o no existe.

## 8.7 ¿Cuándo existe la kurtosis teórica y qué significa que no exista?

**Cuándo existe [A].** Para GARCH(1,1) gaussiano, la condición es:

$$1-2\alpha_1^2-(\alpha_1+\beta_1)^2>0$$

Nótese que esto es **más restrictivo** que la condición de estacionariedad $\alpha_1+\beta_1<1$. Un modelo puede ser estacionario (varianza finita) y sin embargo **no tener cuarto momento finito**.

**Qué significa que no exista [B].** Esto es sutil y conviene explicarlo bien.

Que la kurtosis teórica no exista **no** significa que el proceso sea imposible o que los datos no puedan generarse. Significa que:

1. **La kurtosis muestral no converge.** Si calculamos la kurtosis sobre 1,000 observaciones y luego sobre 10,000, y luego sobre 100,000, **el número no se estabiliza: sigue creciendo**. No hay ningún valor poblacional al que se acerque.

2. **El estimador está dominado por la observación más extrema.** El valor que obtengamos dependerá esencialmente de si nuestra muestra incluyó o no un evento gigantesco.

3. **Cualquier inferencia basada en el cuarto momento es inválida.** Esto incluye los errores estándar de muchos estimadores, y también la varianza de $a_t^2$, que es lo que usan los tests sobre residuos al cuadrado.

**Conexión con el informe del Capítulo 1 [B]:** allí ya habíamos señalado, como extensión propia, que las kurtosis muestrales enormes (55.25 para Citigroup) no debían leerse como estimaciones estables de un parámetro poblacional. El Capítulo 3 confirma que esa preocupación tiene fundamento teórico preciso: existen procesos plausibles y estacionarios cuya kurtosis **no existe**.

**Y una advertencia [A] que refuerza todo esto:** Tsay observa en §3.5 que *"estudios empíricos recientes de series financieras de alta frecuencia indican que **el comportamiento de cola de los modelos GARCH sigue siendo demasiado corto incluso con innovaciones Student-t estandarizadas**"*. Es decir: ni siquiera GARCH con colas pesadas explícitas alcanza a reproducir las colas observadas en datos de alta frecuencia.

## 8.8 La respuesta completa a la pregunta del Capítulo 1

**Lo que el Capítulo 3 establece [A]:**
- La volatilidad variable **genera** exceso de kurtosis, aunque los shocks estandarizados sean perfectamente normales. Está demostrado para ARCH(1) y GARCH(1,1).
- El parámetro responsable es $\alpha_1$ (la reacción a los shocks), no $\beta_1$ (la persistencia).
- Las dos fuentes posibles de colas pesadas —shocks con colas pesadas y volatilidad variable— **coexisten y se refuerzan**, con una fórmula que las combina.
- Empíricamente, para series de alta frecuencia, ni siquiera ambas juntas parecen bastar.

**Lo que sigue siendo una pregunta empírica [B]:**

> **¿Qué fracción del exceso de kurtosis de un futuro concreto desaparece al estandarizar por una estimación de volatilidad?**

Es decir: calcular la kurtosis de $r_t$ y la de $\tilde a_t=a_t/\hat\sigma_t$, y comparar. Si la segunda es mucho menor, buena parte de las colas pesadas era heterocedasticidad. Si sigue siendo alta, hay colas pesadas genuinas en la distribución condicional.

**Esta hipótesis ahora tiene un método concreto asociado** (estimar un modelo de volatilidad y examinar los residuos estandarizados), que en el Capítulo 1 no teníamos. Sigue en el backlog, sin ejecutar.

---
# 9. Mapa: Capítulo 1 → Capítulo 2 → Capítulo 3

## 9.1 La historia en tres actos

### CAPÍTULO 1 — Observamos algo raro

**Qué encontramos:**
- Los retornos tienen **colas pesadas**: los movimientos extremos son mucho más frecuentes de lo que predice una campana de Gauss. (Exceso de kurtosis de 10 a 55 en series diarias.)
- La volatilidad **se agrupa**: los períodos agitados vienen en rachas.
- La **media** de un retorno diario es estadísticamente indistinguible de cero.

**La pregunta que quedó:**
> ¿Por qué ocurre esto? ¿Y son dos fenómenos separados o el mismo?

### CAPÍTULO 2 — Localizamos dónde está la estructura

**Qué encontramos:**
- $ACF(r_t)\approx0$ — casi no hay memoria en la **dirección**.
- $ACF(|r_t|)$ significativa **más allá de 300 lags** — hay muchísima memoria en la **magnitud**.
- Ambas cosas son compatibles porque la ausencia de correlación sólo implica independencia bajo normalidad, y los retornos no son normales.

**La pregunta que quedó:**
> Si no hay mucha memoria en la dirección pero sí en la magnitud, **¿cómo modelamos esa magnitud?**

### CAPÍTULO 3 — Modelamos la magnitud

**Qué hacemos:**
- Introducimos $\sigma_t^2$ —la varianza condicional— como una **variable dinámica**, algo que cambia en el tiempo y que puede modelarse.
- Separamos el problema en dos ecuaciones: **media** (¿hacia dónde?) y **volatilidad** (¿cuánto?).
- Construimos modelos: ARCH, GARCH y sus variantes; volatilidad estocástica; y medidas directas como la volatilidad realizada y los estimadores OHLC.

**Y de paso, respondemos la pregunta del Capítulo 1:**
> La volatilidad variable puede generar colas pesadas incluso cuando los shocks estandarizados son normales. Por tanto, volatility clustering y heavy tails pueden estar estrechamente relacionados, pero **no son necesariamente el mismo fenómeno ni tienen una única causa.** Las colas pesadas también pueden provenir de la propia distribución condicional de los shocks.

## 9.2 El mapa visual

```
CAPÍTULO 1
  Observación: colas pesadas + volatility clustering
  Herramientas: momentos, kurtosis, distribuciones marginales
  Pregunta: ¿por qué?
                    │
                    ▼
CAPÍTULO 2
  Localización: ACF(r) ≈ 0  PERO  ACF(|r|) persistente
  Herramientas: ACF, PACF, Ljung-Box, ARMA, raíces unitarias
  Pregunta: ¿cómo modelo la magnitud?
                    │
                    ▼
CAPÍTULO 3
  Modelado: σ²ₜ como variable dinámica
  Estructura: rₜ = μₜ + aₜ ,  aₜ = σₜ εₜ
  Herramientas: ARCH → GARCH → {IGARCH, EGARCH, TGARCH, GARCH-M}
                SV, LMSV, Realized Volatility, OHLC
  Respuesta al Cap.1: la volatilidad variable PUEDE GENERAR parte de las colas pesadas observadas.
  Preguntas nuevas: ¿es explotable? ¿cómo se evalúa? ¿qué proxy?
```

## 9.3 Las herramientas que se reutilizan

Éste es el punto que da coherencia a los tres capítulos: **nada de lo aprendido se descarta, todo se recicla aplicado a un objeto distinto.**

| Herramienta | Cap. 1: aplicada a | Cap. 2: aplicada a | Cap. 3: aplicada a |
|---|---|---|---|
| **Log-retorno / aditividad** | Definir la variable | Diferenciar el log-precio | Construir $RV_t$ sumando retornos intradía |
| **Momentos y kurtosis** | Describir la marginal | — | Derivar las colas de GARCH (§3.16) |
| **Verosimilitud** | Marco general de estimación | AIC/BIC | Estimar GARCH; base de la loss |
| **ACF** | — | Detectar memoria en $r_t$ | Detectar efectos ARCH en $a_t^2$; verificar $\tilde a_t$ |
| **PACF** | — | Elegir orden AR | Elegir orden ARCH sobre $a_t^2$ |
| **Ljung–Box** | — | Testear white noise en $r_t$ | Testear efectos ARCH; verificar residuos estandarizados |
| **ARMA** | — | Modelo de la media | **GARCH = ARMA sobre $a_t^2$** |
| **Persistencia / raíz unitaria** | — | $\phi_1$, ADF | $\alpha_1+\beta_1$, IGARCH |
| **Reversión a la media** | — | Pronóstico AR → media | Pronóstico GARCH → varianza incondicional |
| **Diferenciación fraccional** | — | Memoria larga en $\|r_t\|$ | LMSV: memoria larga en la volatilidad latente |
| **Mixtura de escala de normales** | Una de las 4 familias candidatas | — | **Es lo que GARCH implementa dinámicamente** |

## 9.4 Lo que cada capítulo NO resolvió

| Capítulo | Dejó abierto |
|---|---|
| **1** | ¿Por qué las colas pesadas? → **respondido en el Cap. 3** |
| **1** | ¿Hay predictibilidad? → parcialmente respondido: no en la media, sí en la magnitud |
| **2** | ¿Cómo modelar la memoria de la magnitud? → **respondido en el Cap. 3** |
| **2** | ¿Cuánta autocorrelación es microestructura? → **Cap. 5** |
| **3** | ¿Es explotable la volatilidad predecible? → **fuera de Tsay** |
| **3** | ¿Cómo evaluar pronósticos de volatilidad correctamente? → literatura específica |
| **3** | ¿La persistencia es real o son cambios de régimen? → **Caps. 11–12** |
| **3** | ¿Qué queda después de estandarizar por volatilidad? → **Cap. 4** (no linealidad) |
| **3** | ¿Cómo se relacionan las volatilidades de varios instrumentos? → **Cap. 10** |

---

# 10. Implicancias para mercados de futuros

## 10.1 Lo que Tsay establece, sin referencia a futuros — [A]

Con la anotación del instrumento, frecuencia y período de cada hallazgo, según el criterio adoptado:

1. **La volatilidad no es directamente observable**; con una observación por período no hay dispersión que medir. *(Afirmación general, ilustrada con IBM diario.)*
2. **Cuatro características de la volatilidad**: clusters, evolución continua, no divergencia (estacionariedad), y asimetría (leverage effect). *(Observadas en retornos de activos; los ejemplos del capítulo son acciones e índices estadounidenses y un tipo de cambio.)*
3. **Los retornos son "no correlacionados pero dependientes"**. *(Intel mensual, 1973–2008.)*
4. **Estructura de dos ecuaciones**: media y volatilidad, con $\sigma_t^2=\text{Var}(a_t\mid\mathcal{F}_{t-1})$.
5. **Procedimiento de construcción en cuatro pasos**, con la observación de que la ecuación de media suele ser simple.
6. **Dos tests de efectos ARCH**: Ljung–Box sobre $a_t^2$ y LM de Engle. *(Intel mensual: $Q(12)=18.26$, p = 0.11 sobre $r_t$; $Q(12)=89.85$, p = 5×10⁻¹⁴ sobre $a_t^2$; LM $F=53.62$, p ≈ 0.)*
7. **Cuatro debilidades de ARCH**, incluida la sobrepredicción tras shocks aislados.
8. **GARCH es un ARMA sobre $a_t^2$**, y sus pronósticos revierten a la varianza incondicional.
9. **Persistencia empírica muy alta**: $\hat\alpha_1+\hat\beta_1=0.9772$. *(S&P 500, retornos excedentes mensuales, 792 obs. desde 1926.)*
10. **Evaluar pronósticos contra $r_t^2$ es inapropiado**: proxy insesgado pero muy impreciso.
11. **IGARCH puede deberse a cambios de nivel en la volatilidad**; la causa real "merece investigación cuidadosa".
12. **La prima de riesgo (GARCH-M) no resulta significativa** en sus dos ejemplos. *(S&P 500 mensual: $c=1.09$, e.e. 0.818. CRSP diario: $c=-3.361$, e.e. 2.026.)*
13. **Leverage effect significativo al 5%.** *(IBM mensual, 1926–2003, TGARCH: coeficiente 0.0658 tras subida vs 0.1501 tras caída.)*
14. **SV mejora el ajuste pero sus contribuciones al pronóstico out-of-sample "recibieron resultados mixtos"**.
15. **Memoria larga en volatilidad**: ACF de $|r_t|$ decae lentamente hasta el lag 200. *(IBM y S&P 500 diarios, 1962–2003.)* Estimación mediana de $d\approx0.38$. *(Acciones del S&P 500.)*
16. **Volatilidad realizada**: suma de retornos intradía al cuadrado; ruido de microestructura a frecuencias muy altas; 4–15 minutos como intervalo habitual *para activos estadounidenses muy negociados*; overnight sustancial en acciones, pequeño en índices y FX.
17. **Estimadores OHLC**: factores de eficiencia de ~2 a ~8 respecto del cierre a cierre, *bajo el modelo de difusión simple de Garman y Klass*. El rango observado subestima el verdadero, con sesgo que depende de frecuencia de negociación y tick size.
18. **La volatilidad variable genera colas pesadas**; el parámetro responsable es $\alpha_1$; la kurtosis teórica puede no existir.
19. **Las colas de GARCH siguen siendo demasiado cortas** para series de alta frecuencia, incluso con innovaciones Student-t.
20. **La volatilidad de un instrumento puede depender de información de otro**. *(S&P 500 mensual con retorno rezagado de IBM: p = 0.0039.)*

## 10.2 Adaptación a futuros — [B], enteramente propia

**Sobre la estructura general**

- La estructura de dos ecuaciones (media y volatilidad) se traslada sin cambios conceptuales a cualquier instrumento. Lo que **no** se traslada es la forma concreta de cada una.
- El orden de la ecuación de media **depende de la frecuencia** [A]; en futuros hay que examinarlo por frecuencia, no asumirlo.
- El procedimiento de cuatro pasos es transferible como protocolo de diagnóstico.

**Sobre el leverage effect — la advertencia principal**

- **No debe asumirse que el leverage effect de acciones se traslade a futuros.** El mecanismo que le da nombre (apalancamiento corporativo) no existe en commodities, divisas ni tasas.
- En **commodities energéticos y agrícolas** cabe esperar que el signo pueda ser el **opuesto**: los shocks de escasez son al alza.
- En **divisas**, la dirección es una convención del par, así que la asimetría no puede tener el mismo origen.
- En **futuros de índices accionarios** es más plausible, pero el mecanismo puede ser cobertura de opciones y flujos de riesgo más que apalancamiento contable.
- **Formulación correcta:** la existencia, signo y magnitud de la asimetría son cuestiones empíricas por instrumento.

**Sobre la medición de volatilidad**

- Los futuros líquidos operan casi 24 horas, lo que **reduce** el problema del gap overnight comparado con acciones. Es una expectativa razonable, no un resultado.
- Pero la actividad dentro de las 24 horas es **extremadamente heterogénea**: sumar cuadrados de la madrugada asiática y de la apertura de Nueva York mezcla regímenes de liquidez distintos.
- El **gap de fin de semana** sigue existiendo.
- El **roll de contratos** introduce discontinuidades que no son movimiento de mercado; deben excluirse de cualquier cálculo de volatilidad, igual que se estableció para retornos en el Cap. 1.
- La **definición del "día"** en un mercado de 24 horas es una convención que afecta directamente a los estimadores OHLC.
- El **tick size relativo** ($\text{tick}/P$, medido en el Cap. 1) determina cuánto sesgo hay en el rango observado y a partir de qué frecuencia la volatilidad realizada mide discreción en lugar de movimiento.

**Sobre los patrones intradía**

- Un patrón determinista de volatilidad por hora del día (la forma de U típica de las sesiones) **produciría autocorrelación en $a_t^2$ que no es dinámica ARCH sino estacionalidad**. Confundirlos llevaría a modelar mal el fenómeno. → **PREGUNTA ABIERTA**, conecta con §2.8 del Cap. 2 y con el **Cap. 5**.
- Lo mismo vale para efectos de calendario: datos macro, vencimientos, subastas.

**Sobre múltiples instrumentos**

- El marco de variables explicativas en la ecuación de volatilidad [A] abre formalmente la puerta a información cross-market, y Tsay lo demuestra con un ejemplo (IBM → S&P 500).
- Pero el tratamiento correcto de la volatilidad conjunta de varios instrumentos es **Cap. 10** (covarianza condicional dinámica). → **PREGUNTA ABIERTA.**
- El hallazgo de Ray y Tsay sobre componentes comunes de memoria larga en sectores [A] sugiere que instrumentos relacionados pueden compartir dinámica de volatilidad. En futuros esto plantearía preguntas sobre grupos (índices accionarios entre sí, curva de tasas, complejo energético) — pero es hipótesis, no resultado.

---

# 11. Implicancias para Machine Learning

**Todas las filas de esta sección son [B] salvo donde se indique. Ninguna constituye una decisión.**

## 11.1 Data representation

| # | Implicancia potencial | Origen | Condición de validez |
|---|---|---|---|
| D1 | La descomposición $r_t=\mu_t+a_t$ con $a_t=\sigma_t\epsilon_t$ da un vocabulario preciso para separar "señal", "escala" y "ruido" en cualquier arquitectura | [A] §3.2 | Es una descomposición, no una teoría causal |
| D2 | Si la volatilidad varía, el orden temporal contiene información; barajar los datos la destruye | [A] §3.1 + [B] | — |
| D3 | Existen al menos cuatro representaciones de la volatilidad: modelada (GARCH), latente estocástica (SV), realizada (intradía), y basada en rango (OHLC) | [A] §§3.5, 3.12, 3.15 | Cada una mide algo ligeramente distinto |
| D4 | La elección del método de medición cambia el número obtenido sustancialmente, no marginalmente | [A] Ejemplo 3.6 | Demostrado en S&P 500 mensual |
| D5 | En futuros, cualquier medida de volatilidad debe excluir gaps de roll y tratar explícitamente los gaps de fin de semana | [B] | No está en Tsay |

## 11.2 Feature engineering

| # | Implicancia potencial | Origen | Condición |
|---|---|---|---|
| F1 | **¿Podría una estimación de volatilidad ser un feature?** El marco lo permite formalmente y Tsay muestra un caso donde información externa mejora la ecuación de volatilidad | [A] §3.10.1, §3.14 | **PREGUNTA ABIERTA**: que sea posible no implica que ayude |
| F2 | Distintos estimadores de volatilidad (RV, Parkinson, YZ, GARCH) capturan aspectos distintos y no son intercambiables | [A] §3.15 + [B] | Los factores de eficiencia son bajo el modelo de difusión de Garman–Klass |
| F3 | Features basados en $\lvert a_t\rvert$ son menos sensibles a extremos que los basados en $a_t^2$ | [B] apoyado en que Tsay usa ambos | — |
| F4 | Features que descartan el signo pierden la información que motiva EGARCH y TGARCH | [A] §§3.8, 3.9 + [B] | Sólo relevante si hay asimetría real en el instrumento |
| F5 | La volatilidad realizada requiere elegir una frecuencia de muestreo, y esa elección es un hiperparámetro con tensión explícita (ruido estadístico vs microestructura) | [A] §3.15.1 | **PREGUNTA ABIERTA → Cap. 5** |
| F6 | Cualquier feature de volatilidad debe calcularse **causalmente**, con información hasta $t$ | [B] | No está en Tsay; es requisito de validación |

## 11.3 Target design

| # | Implicancia potencial | Origen | Condición |
|---|---|---|---|
| T1 | **¿Podrían modelarse $r_t$ y $\sigma_t$ como objetivos diferentes?** La estructura de dos ecuaciones sugiere que son problemas distintos | [A] §3.2 | **No se concluye que deban ser dos modelos separados**; hay al menos tres arquitecturas compatibles |
| T2 | La volatilidad es un target **latente**: no hay etiqueta verdadera contra la cual entrenar | [A] §3.1 | Requiere elegir un proxy, y esa elección es sustantiva |
| T3 | **¿Podría usarse $r_t/\hat\sigma_t$ para estabilizar la distribución?** Es exactamente el residuo estandarizado que Tsay usa para diagnóstico | [A] §3.5.1 + [B] | **PREGUNTA ABIERTA**: que estabilice la distribución no implica que mejore el aprendizaje |
| T4 | Un target de magnitud ($\lvert r\rvert$, $r^2$, RV) tiene mucha más estructura detectable que uno de dirección | [A] §§3.3.1, 3.13 | Verificado en acciones e índices; hipótesis en futuros |
| T5 | Umbrales de clasificación en unidades absolutas significan cosas distintas según el régimen de volatilidad | [B] | — |

## 11.4 Model architecture

| # | Implicancia potencial | Origen | Condición |
|---|---|---|---|
| A1 | **¿Podría un modelo emitir $(\hat\mu_t,\hat\sigma_t)$ en lugar de sólo $\hat r_t$?** Es exactamente la estructura de Tsay | [A] §3.2 | Es una posibilidad, no una decisión |
| A2 | GARCH es un modelo de tres parámetros; SV con dos fuentes de aleatoriedad y estimación por MCMC **no mejoró consistentemente el pronóstico** | [A] §3.12 | Evidencia sobre estos modelos concretos, no un argumento general contra la complejidad |
| A3 | La familia GARCH ofrece una jerarquía natural de benchmarks de volatilidad, análoga a la jerarquía de benchmarks de media del Cap. 2 | [B] | — |
| A4 | La asimetría (EGARCH/TGARCH) es una forma de no linealidad ya presente en el marco lineal-cuadrático | [A] §3.8 | Tsay señala que EGARCH es "similar a un modelo TAR"; la no linealidad general es **Cap. 4** |

**Jerarquía de benchmarks de volatilidad que se desprende [B]:**

| Nivel | Benchmark | Qué asume |
|---|---|---|
| 0 | Volatilidad constante = desviación estándar histórica | Homocedasticidad |
| 1 | Media móvil de $r^2$ en ventana fija | Persistencia sin forma funcional |
| 2 | EWMA / IGARCH sin constante (RiskMetrics) | Persistencia exponencial, sin reversión |
| 3 | GARCH(1,1) | Persistencia + reversión a media |
| 4 | EGARCH / TGARCH | Lo anterior + asimetría |
| 5 | Volatilidad realizada (si hay datos intradía) | Medición directa |

Un modelo complejo debería justificarse contra el nivel apropiado, no contra el nivel 0.

## 11.5 Loss functions

| # | Implicancia potencial | Origen | Condición |
|---|---|---|---|
| L1 | **Conexión con el Cap. 1:** la Gaussian NLL con $\mu_t$ y $\sigma_t^2$ variables es exactamente la Ec. 1.18 de Tsay, y es la verosimilitud que GARCH maximiza | [A] Cap. 1 §1.2.4 + Cap. 3 | La correspondencia es exacta |
| L2 | Un modelo entrenado con Gaussian NLL emitiendo $(\hat\mu_t,\hat\sigma_t)$ **es** un modelo de volatilidad condicional, con forma funcional no paramétrica en lugar de GARCH | [B] | — |
| L3 | Elegir la distribución de $\epsilon_t$ (normal, Student-t, GED) es elegir la loss. Tsay lo hace explícitamente y compara | [A] §3.5.1 | Con Student-t estimó $v=7.02$ (e.e. 1.78) |
| L4 | **Corrección heredada del Cap. 1:** MSE sigue siendo consistente para la media condicional bajo heterocedasticidad. Lo que se pierde es eficiencia e inferencia válida | [B], corrección ya aplicada al informe del Cap. 1 | No repetir el error de declarar MSE inválido |
| L5 | Distintas funciones de pérdida sobre volatilidad (error cuadrático sobre varianzas, sobre volatilidades, sobre logs, QLIKE) pueden ordenar los modelos de forma distinta | [B] | No está en Tsay; **PREGUNTA ABIERTA** |

## 11.6 Probabilistic forecasting

| # | Implicancia potencial | Origen | Condición |
|---|---|---|---|
| P1 | Un pronóstico completo es una **distribución**, no un punto. El par $(\hat\mu_t,\hat\sigma_t)$ más una familia distribucional define esa distribución | [A] §3.2 | — |
| P2 | Las colas de GARCH pueden ser insuficientes incluso con Student-t, según estudios de alta frecuencia | [A] §3.5 | Registrado como limitación conocida |
| P3 | La calibración de probabilidades depende de la volatilidad condicional: la misma predicción puntual implica probabilidades distintas según el régimen | [B] | — |
| P4 | **La incertidumbre sobre la propia estimación de volatilidad importa** y "a menudo se pasa por alto" | [A] §3.16, textual | Es la motivación de la sección de kurtosis |

## 11.7 Model diagnostics

| # | Implicancia potencial | Origen | Condición |
|---|---|---|---|
| G1 | **¿Deberíamos examinar $ACF(a_t)$ y $ACF(a_t^2)$?** Sí como diagnóstico: responden preguntas distintas y ambas son transferibles a modelos de ML | [A] §§3.3.1, 3.5.1 | Es diagnóstico, no criterio de diseño |
| G2 | El **residuo estandarizado** $\tilde a_t=a_t/\hat\sigma_t$ es el objeto correcto a examinar cuando el modelo predice volatilidad | [A] §3.5.1 | — |
| G3 | Comparar la kurtosis de $r_t$ con la de $\tilde a_t$ cuantifica cuánto de las colas pesadas era heterocedasticidad | [A] §3.16 + [B] | Es la hipótesis que quedó del Cap. 1 |
| G4 | Pasar Ljung–Box sobre $\tilde a_t$ y $\tilde a_t^2$ **no establece independencia**: sólo ausencia de dependencia lineal detectable | [B], heredado del Cap. 2 | → **Cap. 4** |
| G5 | Un modelo con buena predicción puntual y residuos al cuadrado autocorrelacionados está emitiendo **incertidumbre mal caracterizada** | [B] | — |

## 11.8 Validation

| # | Implicancia potencial | Origen | Condición |
|---|---|---|---|
| V1 | **Buen ajuste in-sample ≠ buen pronóstico out-of-sample.** Tsay lo constata en un caso concreto y publicado | [A] §3.12, textual | El caso es SV vs GARCH |
| V2 | Tsay advierte que $r_t^2$ es un proxy muy impreciso de la volatilidad y que una correlación o $R^2$ bajos contra ese proxy no bastan para concluir que el pronóstico sea malo | [A] §3.5.2 | La afirmación más fuerte sobre límites específicos del $R^2$ pertenece a [B] / literatura adicional |
| V3 | La elección del proxy de volatilidad para evaluar es una decisión metodológica de primer orden | [A] §3.5.2 + [B] | **No se adopta ninguno todavía** |
| V4 | Con persistencia alta y posible memoria larga en volatilidad, las observaciones cercanas son muy dependientes; el tamaño de muestra efectivo es menor que el nominal | [B], extensión de §2.10 del Cap. 2 | Afecta embargo y significancia |
| V5 | Evaluar por régimen de volatilidad, no sólo en agregado | [B] | — |

## 11.9 Risk management

| # | Implicancia potencial | Origen | Condición |
|---|---|---|---|
| R1 | **¿Una buena predicción de volatilidad puede ser útil aunque no mejore la predicción direccional?** Sí: permite estabilizar el riesgo, que es un objetivo legítimo | [B] | Tsay no discute trading; esto es enteramente interpretación |
| R2 | El IGARCH sin constante es exactamente EWMA y es el modelo de RiskMetrics para VaR | [A] §3.6 | → **Cap. 7** para VaR |
| R3 | Los pronósticos de volatilidad revierten a la varianza incondicional; en horizontes largos, el modelo converge al benchmark trivial | [A] §3.5 | Bajo estacionariedad ($\alpha+\beta<1$) |
| R4 | Bajo IGARCH, los pronósticos **no** revierten: forman una recta. La diferencia con GARCH importa mucho en horizontes largos | [A] §3.6 | — |
| R5 | La kurtosis teórica puede no existir; las estimaciones muestrales de kurtosis pueden no converger | [A] §3.16 | Afecta a toda inferencia basada en el cuarto momento |

## 11.10 Position sizing

| # | Implicancia potencial | Origen | Condición |
|---|---|---|---|
| S1 | Si la volatilidad varía por un factor de 5, una posición de tamaño fijo implica riesgo variable por un factor de 5 | [B] | Consecuencia aritmética de §2.1 de este informe |
| S2 | El dimensionamiento por volatilidad es una decisión de **riesgo**, no de **predicción**. No debe confundirse con generar señal | [B] | — |
| S3 | Si el pronóstico de volatilidad es sesgado (p. ej., ARCH sobrepredice tras shocks aislados), el dimensionamiento hereda ese sesgo | [A] §3.4.2 + [B] | — |

## 11.11 Multivariate modeling

| # | Implicancia potencial | Origen | Condición |
|---|---|---|---|
| M1 | La volatilidad de un instrumento puede depender de información de otro; Tsay lo demuestra con un ejemplo significativo al 1% | [A] §3.14 | Un ejemplo: IBM → S&P 500, mensual |
| M2 | Instrumentos relacionados pueden compartir componentes de memoria larga en volatilidad | [A] §3.13, Ray y Tsay (2000) | Establecido para sectores de acciones estadounidenses |
| M3 | El tratamiento correcto de la volatilidad conjunta requiere modelos multivariados | [A] §3.0, remite al Cap. 10 | **PREGUNTA ABIERTA → Cap. 10** |
| M4 | Apilar instrumentos correlacionados no multiplica la información independiente | [B], heredado del Cap. 2 | — |

## 11.12 Hipótesis a comprobar

Ver sección 12 completa.

---

# 12. Hipótesis a comprobar empíricamente

**Backlog. No ejecutar todavía**, conforme a la restricción del encargo. La razón es explícita: estamos adquiriendo conocimiento y queremos evitar diseñar experimentos antes de terminar de comprender las herramientas relevantes — en particular el Cap. 4 (no linealidad) y el Cap. 5 (microestructura), que afectan directamente a varias de estas preguntas.

### Bloque A — Existencia de efectos ARCH

**H3.1 — ¿Hay efectos ARCH en los residuos de un modelo de media simple?**
1. *Pregunta:* ¿la magnitud de los errores tiene memoria?
2. *Método:* ajustar una ecuación de media simple; aplicar Ljung–Box sobre $a_t^2$ y el test LM de Engle, con varios $m$.
3. *Apoya:* rechazo claro y consistente entre subperíodos.
4. *Refuta:* no rechazo robusto a la elección de $m$ y del período.
5. *Limitaciones:* con series intradía largas, **casi cualquier test rechaza** — la significancia no mide magnitud. El test supone condiciones de momentos que con colas pesadas pueden fallar. Un patrón intradía determinista podría producir rechazo sin dinámica ARCH genuina.

**H3.2 — ¿La ACF de $a_t^2$ y la de $|a_t|$ cuentan la misma historia?**
1. *Pregunta:* ¿cuánto cambia el diagnóstico según se use el cuadrado o el valor absoluto?
2. *Método:* comparar ambas ACF sobre el mismo instrumento y período.
3. *Apoya la equivalencia:* patrones similares.
4. *Refuta:* la ACF de $a_t^2$ dominada por pocos eventos, con forma distinta.
5. *Limitaciones:* no hay un criterio establecido para decidir cuál es "correcta"; miden cosas ligeramente distintas.

### Bloque B — Persistencia

**H3.3 — ¿Cuál es la persistencia $\hat\alpha_1+\hat\beta_1$ en futuros, por instrumento y frecuencia?**
1. *Pregunta:* ¿se replica el valor cercano a 1 que Tsay obtiene en índices accionarios mensuales?
2. *Método:* estimar GARCH(1,1) y reportar $\hat\alpha_1$, $\hat\beta_1$, su suma y sus errores estándar.
3. *Apoya:* valores en el rango 0.95–0.99, consistentes entre instrumentos.
4. *Refuta:* dispersión grande entre instrumentos o valores claramente inferiores.
5. *Limitaciones:* la estimación puntual no distingue GARCH persistente de IGARCH de cambios de régimen. **No convertir un valor cercano a 1 en "demostramos IGARCH".**

**H3.4 — ¿La persistencia es estable entre subperíodos?**
1. *Pregunta:* ¿es un parámetro o un artefacto de mezclar regímenes?
2. *Método:* estimar el mismo modelo en ventanas disjuntas y comparar.
3. *Apoya la estabilidad:* estimaciones consistentes.
4. *Refuta:* variación grande, sobre todo si $\hat\alpha+\hat\beta$ baja al estimar dentro de un régimen homogéneo — lo que apoyaría la hipótesis de cambios de nivel que Tsay menciona.
5. *Limitaciones:* ventanas cortas dan estimaciones ruidosas; la partición en "regímenes" es en sí misma una decisión.

### Bloque C — Asimetría

**H3.5 — ¿Existe respuesta asimétrica a shocks de distinto signo, y con qué signo?**
1. *Pregunta:* ¿se traslada el leverage effect a futuros, y con qué signo por clase de activo?
2. *Método:* estimar TGARCH y EGARCH; examinar $\hat\gamma$ y su significancia; comparar entre clases de futuros.
3. *Apoya el leverage clásico:* $\hat\gamma>0$ en TGARCH (shocks negativos elevan más la volatilidad), significativo y estable.
4. *Refuta:* $\hat\gamma$ no significativo, o **de signo contrario** — lo que sería el resultado esperable en algunos commodities.
5. *Limitaciones:* la definición de "negativo" es convencional en divisas. El resultado puede depender de la frecuencia. **No asumir el signo de acciones.**

**H3.6 — ¿Mejora EGARCH o TGARCH el pronóstico out-of-sample respecto de GARCH?**
1. *Pregunta:* ¿la asimetría, si existe, mejora la predicción o sólo el ajuste?
2. *Método:* comparación out-of-sample estricta con un proxy de volatilidad definido de antemano.
3. *Apoya:* mejora consistente entre períodos e instrumentos.
4. *Refuta:* mejora sólo in-sample.
5. *Limitaciones:* **el resultado depende del proxy y de la función de pérdida elegidos**; distintas elecciones pueden ordenar los modelos de forma distinta.

### Bloque D — Evaluación y pronóstico

**H3.7 — ¿Cuánto cambia el ranking de modelos según el proxy de volatilidad?**
1. *Pregunta:* ¿es robusta la comparación entre modelos de volatilidad?
2. *Método:* evaluar los mismos modelos contra $r_t^2$, contra volatilidad realizada, y contra un estimador OHLC.
3. *Apoya la robustez:* mismo ranking con los tres proxies.
4. *Refuta:* rankings distintos.
5. *Limitaciones:* si los proxies discrepan, el capítulo no da criterio para elegir. **PREGUNTA ABIERTA.**

**H3.8 — ¿Supera un modelo de volatilidad a los benchmarks simples fuera de muestra?**
1. *Pregunta:* ¿vale la pena la complejidad?
2. *Método:* comparar GARCH contra volatilidad constante, media móvil de $r^2$ y EWMA, out-of-sample.
3. *Apoya:* mejora clara y estable sobre EWMA (no sólo sobre volatilidad constante).
4. *Refuta:* EWMA iguala o supera a GARCH.
5. *Limitaciones:* EWMA es un IGARCH sin constante [A], así que la comparación es entre modelos anidados; hay que ser cuidadoso con la interpretación.

### Bloque E — Medición de volatilidad

**H3.9 — ¿Cuál es la frecuencia de muestreo apropiada para volatilidad realizada en cada futuro?**
1. *Pregunta:* ¿dónde está el punto de equilibrio entre ruido estadístico y microestructura?
2. *Método:* calcular RV a varias frecuencias y examinar cómo cambia sistemáticamente (el "volatility signature plot").
3. *Apoya una frecuencia dada:* estabilización de la estimación en un rango.
4. *Refuta:* crecimiento monótono al aumentar la frecuencia, señal de contaminación por microestructura.
5. *Limitaciones:* **requiere el Cap. 5** para interpretarse correctamente. El rango 4–15 minutos de Tsay es para activos estadounidenses muy negociados, no para futuros específicamente.

**H3.10 — ¿Qué aporta la información OHLC respecto del cierre a cierre en futuros?**
1. *Pregunta:* ¿se replican los factores de eficiencia de 5 a 8?
2. *Método:* comparar la varianza de distintos estimadores contra una referencia común (idealmente volatilidad realizada de alta frecuencia).
3. *Apoya:* reducción sustancial de varianza usando el rango.
4. *Refuta:* ganancia marginal, o sesgo sistemático por discreción del tick.
5. *Limitaciones:* los factores de Garman–Klass son **bajo su modelo de difusión**; el sesgo del rango depende de liquidez y tick size, que varían por instrumento y hora del día. La definición del "día" en mercados de 24 horas es convencional.

**H3.11 — ¿Cuán grande es el gap overnight/fin de semana en futuros?**
1. *Pregunta:* ¿es despreciable como en índices y FX [A], o sustancial como en acciones?
2. *Método:* comparar la varianza del retorno de gap con la de la sesión.
3. *Apoya que sea despreciable:* fracción pequeña de la varianza total.
4. *Refuta:* fracción sustancial, sobre todo en fines de semana o alrededor de eventos.
5. *Limitaciones:* depende de la definición de sesión adoptada.

### Bloque F — Memoria larga

**H3.12 — ¿Hay memoria larga en la volatilidad de futuros?**
1. *Pregunta:* ¿la ACF de $|r_t|$ decae polinomial o exponencialmente?
2. *Método:* ACF de $|r_t|$ hasta 200–300 lags; comparar el ajuste de un decaimiento exponencial vs uno polinomial.
3. *Apoya:* decaimiento lento, significativo a lags altos, mejor descrito por una potencia.
4. *Refuta:* decaimiento exponencial que muere en decenas de lags.
5. *Limitaciones:* **la memoria larga aparente puede ser un artefacto de cambios de régimen** → Caps. 11–12. El $d\approx0.38$ de Tsay es para acciones del S&P 500, no para futuros.

### Bloque G — Colas pesadas (la pregunta del Capítulo 1)

**H3.13 — ¿Qué fracción del exceso de kurtosis desaparece al estandarizar por volatilidad?**
1. *Pregunta:* ¿las colas pesadas son heterocedasticidad o son condicionales?
2. *Método:* calcular la kurtosis de $r_t$ y la de $\tilde a_t=a_t/\hat\sigma_t$ usando varias estimaciones de $\hat\sigma_t$ (GARCH, EWMA, volatilidad realizada rezagada).
3. *Apoya que sean heterocedasticidad:* caída drástica de la kurtosis al estandarizar.
4. *Refuta:* kurtosis del residuo estandarizado todavía alta — colas pesadas genuinas en la condicional.
5. *Limitaciones:* **la kurtosis teórica puede no existir** [A §3.16], en cuyo caso el estimador no converge y la comparación es inestable. El resultado depende del estimador de $\hat\sigma_t$ usado. Tsay advierte que las colas de GARCH pueden ser insuficientes incluso con Student-t.

**H3.14 — ¿Qué distribución condicional describe mejor los residuos estandarizados?**
1. *Pregunta:* ¿normal, Student-t, GED, u otra?
2. *Método:* estimar los grados de libertad de una Student-t, como hace Tsay ($\hat v=7.02$, e.e. 1.78, para el S&P 500 mensual).
3. *Apoya normalidad:* $\hat v$ grande con error estándar que no excluye valores altos.
4. *Refuta:* $\hat v$ pequeño y precisamente estimado.
5. *Limitaciones:* la estimación de $v$ es notoriamente imprecisa; el error estándar de 1.78 sobre 7.02 en el ejemplo de Tsay lo ilustra.

### Bloque H — Estabilidad y comparabilidad

**H3.15 — ¿Son comparables los parámetros GARCH entre instrumentos?**
1. *Pregunta:* ¿la dinámica de volatilidad es similar entre clases de futuros?
2. *Método:* estimar el mismo modelo en varios instrumentos y comparar $\hat\alpha_1$, $\hat\beta_1$ y la asimetría.
3. *Apoya:* parámetros en rangos similares.
4. *Refuta:* diferencias sistemáticas por clase de activo.
5. *Limitaciones:* la comparación requiere la misma frecuencia, el mismo período y el mismo tratamiento del roll. Un resultado negativo tendría implicancias directas para cualquier arquitectura que agrupe instrumentos.

**H3.16 — ¿Son estables los parámetros entre frecuencias?**
1. *Pregunta:* ¿la dinámica de volatilidad a 5 minutos es la misma que a diario?
2. *Método:* estimar a varias frecuencias y comparar persistencia implícita ajustada por horizonte.
3. *Apoya:* consistencia una vez ajustada la escala temporal.
4. *Refuta:* dinámicas cualitativamente distintas.
5. *Limitaciones:* a alta frecuencia, los efectos de microestructura y los patrones intradía contaminan la estimación → **Cap. 5**.

**H3.17 — ¿La estacionalidad intradía se confunde con dinámica ARCH?**
1. *Pregunta:* ¿la autocorrelación de $a_t^2$ a alta frecuencia refleja dinámica o patrón horario determinista?
2. *Método:* estimar el patrón medio de volatilidad por hora del día, desestacionalizar, y repetir los tests ARCH sobre la serie desestacionalizada.
3. *Apoya que sea dinámica genuina:* los efectos ARCH persisten tras desestacionalizar.
4. *Refuta:* los efectos desaparecen o se reducen drásticamente.
5. *Limitaciones:* el patrón intradía puede no ser estable entre años; desestacionalizar usando toda la muestra sería un canal de leak. Conecta con §2.8 del Cap. 2 y con el **Cap. 5**.

---
# 13. Errores metodológicos a evitar

## 13.1 Auditoría de las dieciocho afirmaciones

| # | Afirmación | Veredicto |
|---|---|---|
| 1 | "La volatilidad es directamente observable" | **INCORRECTA** |
| 2 | "Si ACF($r_t$) ≈ 0, no hay nada que predecir" | **INCORRECTA** |
| 3 | "Si ACF($r_t^2$) es significativa, existe una estrategia rentable" | **INCORRECTA** |
| 4 | "ARCH effect ⇒ debemos usar GARCH" | **REQUIERE CONDICIONES** |
| 5 | "Un GARCH con buen ajuste in-sample tendrá buen forecast" | **INCORRECTA** |
| 6 | "$\alpha+\beta\approx1$ demuestra IGARCH" | **INCORRECTA** |
| 7 | "GARCH predice dirección" | **INCORRECTA** |
| 8 | "Un shock positivo y uno negativo afectan igual a todos los mercados" | **INCORRECTA** |
| 9 | "Leverage effect en acciones implica leverage effect idéntico en futuros" | **INCORRECTA** |
| 10 | "EGARCH es mejor que GARCH porque es más complejo" | **INCORRECTA** |
| 11 | "Realized volatility es la volatilidad verdadera" | **INCORRECTA** |
| 12 | "Mayor frecuencia intradía siempre mejora realized volatility" | **INCORRECTA** |
| 13 | "OHLC contiene más información, por lo que cualquier estimador OHLC será mejor" | **REQUIERE CONDICIONES** |
| 14 | "Heavy tails implican necesariamente shocks Student-t" | **INCORRECTA** |
| 15 | "Si GARCH explica kurtosis, ya no necesitamos distribuciones heavy-tailed" | **INCORRECTA** |
| 16 | "Volatilidad predecible implica retorno predecible" | **INCORRECTA** |
| 17 | "Volatilidad predecible implica estrategia rentable" | **INCORRECTA** |
| 18 | "Una feature de volatilidad necesariamente mejorará un modelo de ML" | **INCORRECTA** |

---

**1. "La volatilidad es directamente observable" — INCORRECTA.**
Tsay abre el capítulo diciendo lo contrario [A]: con una sola observación por período no hay dispersión que medir. Lo que sí se observa son **proxies**: retornos al cuadrado (muy imprecisos), volatilidad realizada (si hay datos intradía), estimadores basados en rango, o volatilidad implícita (que depende de un modelo de pricing). Todos tienen error. Confundir el proxy con la cantidad verdadera es el error de fondo del que dependen varios de los siguientes.

**2. "Si ACF($r_t$) ≈ 0, no hay nada que predecir" — INCORRECTA.**
Es el mismo error auditado en el Capítulo 2, ahora con evidencia adicional. La ACF mide dependencia **lineal en la dirección**. El ejemplo de Intel de Tsay es demoledor: sobre los mismos datos, Ljung–Box sobre $r_t$ da p = 0.11 (nada), y sobre $a_t^2$ da p = 5×10⁻¹⁴ (muchísimo). La estructura existe; está en la magnitud.

**3. "Si ACF($r_t^2$) es significativa, existe una estrategia rentable" — INCORRECTA.**
Confunde tres niveles distintos, y hay que separarlos con cuidado:
- **Existe dependencia estadística en la magnitud** — sí, la ACF lo muestra.
- **La magnitud es predecible** — plausible, pero requiere demostrarlo out-of-sample.
- **Existe una estrategia rentable** — **no se sigue de lo anterior**, porque la volatilidad no dice de qué lado ponerse. Y aunque hubiera un mecanismo (opciones, dimensionamiento), falta el filtro de costos que Tsay nunca trata.
Añádase que con series largas la significancia estadística no mide magnitud.

**4. "ARCH effect ⇒ debemos usar GARCH" — REQUIERE CONDICIONES.**
Que exista efecto ARCH significa que **vale la pena modelar la volatilidad**. No significa que GARCH sea la herramienta correcta. Alternativas dentro del propio capítulo: EWMA/IGARCH, EGARCH, TGARCH, volatilidad estocástica, volatilidad realizada, estimadores OHLC. Y el efecto ARCH detectado puede tener otras causas: Tsay menciona los **cambios de nivel en la volatilidad** [A], y en datos intradía habría que descartar el patrón horario determinista antes de atribuirlo a dinámica ARCH. **El diagnóstico indica que hay algo; no dicta el modelo.**

**5. "Un GARCH con buen ajuste in-sample tendrá buen forecast" — INCORRECTA.**
Tsay ofrece el contraejemplo directamente [A]: sobre volatilidad estocástica, *"a menudo proveyeron mejoras en el ajuste del modelo, pero sus contribuciones a los pronósticos de volatilidad fuera de muestra recibieron **resultados mixtos**"*. Un modelo más flexible siempre puede ajustar mejor los datos que ya vio. **Que la evidencia esté en el propio libro, con modelos estadísticos clásicos de pocos parámetros, es un aviso relevante para cualquier arquitectura de ML.**

**6. "$\alpha+\beta\approx1$ demuestra IGARCH" — INCORRECTA.**
Cuatro razones, y la primera es del propio Tsay [A]: *"el fenómeno IGARCH **podría ser causado por cambios ocasionales de nivel** en la volatilidad. La causa real de la persistencia en volatilidad merece una investigación cuidadosa"*.
Las otras tres [B]: es una estimación puntual con error estándar; distinguir 0.977 de 1.000 es empíricamente muy difícil (mismo problema de potencia que con ADF en el Cap. 2); y una serie con memoria larga verdadera, forzada a ajustarse con un decaimiento exponencial, elegirá el exponencial más lento posible.
**Formulación correcta:** un valor cercano a 1 indica persistencia empírica muy alta, compatible con IGARCH, con GARCH estacionario muy persistente, con memoria larga y con cambios de régimen.

**7. "GARCH predice dirección" — INCORRECTA.**
GARCH modela $\sigma_t^2=\text{Var}(a_t\mid\mathcal{F}_{t-1})$: la **dispersión** del shock, no su signo. En la descomposición $a_t=\sigma_t\epsilon_t$, el signo lo aporta $\epsilon_t$, que es **iid por construcción del modelo** — es decir, el modelo asume explícitamente que el signo es impredecible.
La única forma en que la volatilidad entra en la ecuación de media es vía GARCH-M, y en los dos ejemplos de Tsay el coeficiente resulta **no significativo** (uno positivo, otro negativo).

**8. "Un shock positivo y uno negativo afectan igual a todos los mercados" — INCORRECTA.**
Ésa es precisamente la **debilidad número 1 de ARCH y GARCH** que Tsay enumera [A], y la razón de existir de EGARCH y TGARCH. En su ejemplo de IBM con TGARCH, el coeficiente pasa de 0.0658 tras una subida a 0.1501 tras una caída: más del doble, significativo al 5%.
**Pero atención a la palabra "todos":** la afirmación es incorrecta porque no siempre afectan igual, no porque siempre afecten distinto en el mismo sentido. Ver el punto 9.

**9. "Leverage effect en acciones implica leverage effect idéntico en futuros" — INCORRECTA.**
El mecanismo que da nombre al efecto —cuando cae el precio de una acción, la empresa queda más apalancada y por tanto más riesgosa— **no existe en commodities, divisas ni tasas de interés**. En commodities de oferta restringida, cabe esperar que los shocks violentos sean **al alza**, lo que produciría asimetría de signo contrario. En divisas, la dirección es una convención del par.
La evidencia de Tsay sobre leverage effect proviene de: IBM (acción individual, mensual, 1926–2003) y el índice CRSP value-weighted (diario, 1962–1987). **Nada permite extrapolar a futuros de commodities, tasas o divisas.**
Además, Tsay usa lenguaje deliberadamente prudente: *"parece reaccionar de forma distinta"*, *"tienden a tener impactos mayores"*, *"esperamos que $\gamma_i$ sea negativo en aplicaciones reales"*.

**10. "EGARCH es mejor que GARCH porque es más complejo" — INCORRECTA.**
La complejidad no es un criterio de calidad. EGARCH resuelve un problema específico (permitir asimetría y eliminar restricciones de positividad). **Si en un instrumento dado no hay asimetría, EGARCH añade parámetros sin añadir información**, con el consiguiente riesgo de sobreajuste. Y la comparación correcta es out-of-sample, no por ajuste. Nótese además que la afirmación 5 ya establece que mejor ajuste no implica mejor pronóstico.

**11. "Realized volatility es la volatilidad verdadera" — INCORRECTA.**
$RV_t$ es un **estimador** de la volatilidad del período, no la volatilidad misma. Tiene al menos tres fuentes de error que Tsay señala [A]:
- **Sesgo por microestructura** a frecuencias altas (bid–ask bounce).
- **Omisión del gap overnight**, que en acciones *"puede subestimar seriamente la volatilidad"*.
- **Error de muestreo**: con $n$ finito, sigue siendo una estimación.
Es un proxy **mucho mejor** que $r_t^2$, pero sigue siendo un proxy.

**12. "Mayor frecuencia intradía siempre mejora realized volatility" — INCORRECTA.**
Ésta es la tensión central de §3.15.1 y Tsay la enuncia explícitamente [A]: *"intuitivamente, uno querría usar tanta información como sea posible eligiendo un $n$ grande. **Sin embargo**, cuando el intervalo temporal entre $r_{t,i}$ es pequeño, los retornos están sujetos a los efectos de la microestructura de mercado... **lo que a menudo resulta en una estimación sesgada**"*.
Hay un **trade-off**: más observaciones reducen el error estadístico pero aumentan la contaminación. Por eso existe una frecuencia intermedia razonable — Tsay reporta 4–15 minutos *para activos estadounidenses muy negociados*, que no es una constante universal.
Añádase, para futuros [B]: si el tick size es grueso respecto del movimiento típico en el intervalo elegido, la suma de cuadrados mide la discreción de la grilla más que la volatilidad.

**13. "OHLC contiene más información, por lo que cualquier estimador OHLC será mejor" — REQUIERE CONDICIONES.**
La premisa es correcta: OHLC **sí** contiene más información que el cierre solo, y los factores de eficiencia de Garman–Klass (2 a 8.4) lo cuantifican. Pero la conclusión no se sigue automáticamente, por cuatro razones:
- **Los factores de eficiencia son bajo el modelo de difusión simple sin drift** que Garman y Klass entretienen [A]. No son garantías universales.
- **Los estimadores difieren en qué capturan.** Parkinson ignora completamente el gap overnight; Rogers–Satchell tolera drift pero tampoco maneja gaps; Yang–Zhang combina las tres piezas. **Elegir mal el estimador para el problema puede ser peor que usar el cierre.**
- **El rango observado subestima el verdadero** [A], con sesgo que depende de la frecuencia de negociación y del tick size.
- **Yang–Zhang supone volatilidad constante durante la ventana de $n$ días** [A] — un supuesto que este mismo capítulo dedica cien páginas a refutar.

**14. "Heavy tails implican necesariamente shocks Student-t" — INCORRECTA.**
Tsay muestra en distintos puntos del capítulo que la dinámica de volatilidad puede generar exceso de kurtosis incluso con innovaciones gaussianas: lo deriva para ARCH(1) en §3.4.1, para GARCH(1,1) en §3.5, y profundiza la relación entre la kurtosis de las innovaciones y la del proceso GARCH en §3.16. Por tanto, las heavy tails observadas no requieren necesariamente shocks básicos con colas pesadas: una volatilidad condicional variable puede ser una de sus fuentes. Cuál de las fuentes domina en un instrumento concreto sigue siendo una pregunta empírica (H3.13).

**15. "Si GARCH explica kurtosis, ya no necesitamos distribuciones heavy-tailed" — INCORRECTA.**
Es el error opuesto al anterior, y Tsay también lo previene [A]: *"estudios empíricos recientes de series financieras de alta frecuencia indican que **el comportamiento de cola de los modelos GARCH sigue siendo demasiado corto incluso con innovaciones Student-t estandarizadas**"*.
Es decir: ni siquiera GARCH **con** Student-t alcanza a reproducir las colas observadas en alta frecuencia. Que GARCH genere *algo* de exceso de kurtosis no significa que genere *suficiente*.
Además, la fórmula de §3.16 muestra que las dos fuentes **coexisten y se refuerzan** con un término de interacción; no son alternativas excluyentes.

**16. "Volatilidad predecible implica retorno predecible" — INCORRECTA.**
Son objetos matemáticamente distintos: $\text{Var}(r_{t+1}\mid X_t)$ y $E(r_{t+1}\mid X_t)$. En la descomposición $a_t=\sigma_t\epsilon_t$, saber $\sigma_t$ no dice nada sobre el signo de $\epsilon_t$, que es iid por construcción.
La única conexión posible sería una prima de riesgo (GARCH-M), y en los dos ejemplos de Tsay **no resulta significativa**. Esto no cierra la pregunta —dos ejemplos sobre índices estadounidenses no resuelven un debate de décadas— pero tampoco autoriza a asumir que la relación exista.

**17. "Volatilidad predecible implica estrategia rentable" — INCORRECTA.**
Aunque el pronóstico de volatilidad fuera excelente, hace falta un **mecanismo** que lo convierta en P&L, y ese mecanismo no está en el capítulo:
- **Dimensionamiento**: estabiliza el riesgo pero no genera retorno esperado por sí solo.
- **Opciones**: permitiría operar la volatilidad directamente, pero requiere que el pronóstico supere al que ya está incorporado en los precios de las opciones. Nótese el dato de Tsay [A]: *"la volatilidad implícita tiende a ser mayor que la obtenida con un modelo GARCH"* — hay una diferencia sistemática cuya explicación (prima de riesgo por volatilidad) no está resuelta.
- **Prima de riesgo**: no significativa en los ejemplos de Tsay.
Y en todos los casos falta el filtro de costos.

**18. "Una feature de volatilidad necesariamente mejorará un modelo de ML" — INCORRECTA.**
Cuatro razones [B]:
- Que exista información en la volatilidad no implica que **ese** modelo, con **esos** datos, la aproveche.
- Una feature de volatilidad mal calculada (no causal, contaminada por el roll, o dependiente de un proxy inapropiado) puede **empeorar** el modelo o introducir leak.
- Si el modelo ya extrae implícitamente información equivalente de los retornos rezagados, la feature es redundante.
- La palabra **"necesariamente"** es lo que hace la afirmación incorrecta. Es una hipótesis razonable y comprobable, **no una certeza**, y es exactamente el tipo de salto que el criterio metodológico del proyecto busca evitar.

## 13.2 Otros errores metodológicos del capítulo

19. **Evaluar un pronóstico de volatilidad contra $r_t^2$ y concluir que el modelo es malo por el $R^2$ bajo** [A §3.5.2]. El proxy es insesgado pero muy impreciso; el $R^2$ bajo es esperable incluso con un pronóstico perfecto.

20. **Estimar la media y la volatilidad por separado sin considerar que la estimación conjunta es la correcta** [A §3.3, paso 3]. Tsay presenta el método en dos pasos como una **aproximación**.

21. **No verificar los residuos estandarizados** [A §3.5.1]. Un modelo de volatilidad puede parecer razonable y dejar estructura evidente en $\tilde a_t$ o $\tilde a_t^2$.

22. **Interpretar la kurtosis muestral como un parámetro estable** [A §3.16 + B]. Puede no existir teóricamente, en cuyo caso el estimador no converge.

23. **Confundir la volatilidad implícita con la volatilidad esperada** [A §3.1 + B]. Tsay señala que la implícita tiende a ser mayor que la de modelos GARCH, posiblemente por prima de riesgo. Son objetos distintos.

24. **Ignorar el gap overnight o de fin de semana al calcular volatilidad realizada** [A §3.15.1]. En acciones, *"puede subestimar seriamente la volatilidad"*.

25. **Calcular volatilidad sobre barras que abarcan un roll de contrato** [B]. El salto de roll no es movimiento de mercado. Heredado del criterio del Cap. 1.

26. **Estandarizar por una volatilidad estimada con información futura** [B]. Si $\hat\sigma_t$ se calcula con una ventana centrada o con parámetros estimados sobre toda la muestra, la estandarización introduce leak. Debe ser estrictamente causal.

27. **Atribuir a dinámica ARCH lo que puede ser estacionalidad intradía determinista** [B]. Un patrón horario de volatilidad produce autocorrelación en $a_t^2$ sin que haya dinámica genuina.

28. **Confundir "estacionariedad de la volatilidad" con "volatilidad constante"** [B]. Tsay dice que la volatilidad *"a menudo es estacionaria"* [A] — es decir, fluctúa alrededor de un nivel. Eso es lo contrario de constante.

29. **Usar los factores de eficiencia de Garman–Klass como si fueran universales** [A + B]. Son bajo su modelo de difusión simple.

30. **Extrapolar los parámetros de un instrumento a otro** [B]. Los valores $\hat\alpha_1\approx0.12$, $\hat\beta_1\approx0.85$ de Tsay son para retornos excedentes mensuales del S&P 500 desde 1926. No son constantes de la naturaleza.

---

# 14. Preguntas abiertas

Registro explícito de lo que **no** debe resolverse todavía.

| Cuestión surgida en el Cap. 3 | Dónde se resuelve |
|---|---|
| ¿Queda dependencia no lineal en los residuos estandarizados $\tilde a_t$ tras un modelo de volatilidad? | **Cap. 4** |
| ¿Es realmente iid el shock estandarizado $\epsilon_t$, como supone todo el marco? | **Cap. 4** |
| EGARCH es "similar a un modelo TAR" [A]; ¿qué otras formas de no linealidad hay? | **Cap. 4** |
| ¿Cuánta de la autocorrelación de $a_t^2$ a alta frecuencia es microestructura? | **Cap. 5** |
| ¿Cuál es la frecuencia óptima para volatilidad realizada en cada futuro? | **Cap. 5** |
| ¿Cómo afecta el bid–ask bounce al sesgo de la volatilidad realizada? | **Cap. 5** |
| ¿Cómo afecta la discreción del tick al rango observado y a los estimadores OHLC? | **Cap. 5** |
| ¿Se distinguen los patrones intradía deterministas de la dinámica ARCH? | **Cap. 5** |
| ¿Cómo se modela el comportamiento de los extremos, dado que las colas de GARCH parecen insuficientes? | **Cap. 7** |
| ¿Cómo se usa un pronóstico de volatilidad para calcular VaR? (IGARCH/EWMA es RiskMetrics) | **Cap. 7** |
| ¿Cómo se relacionan las volatilidades de varios instrumentos? | **Cap. 8, Cap. 10** |
| ¿Cómo evoluciona la covarianza condicional entre futuros? | **Cap. 10** |
| ¿Cuántos factores comunes de volatilidad hay en un conjunto de futuros? | **Cap. 9, Cap. 10** |
| ¿La persistencia extrema es real o son cambios de régimen? | **Caps. 11–12** |
| ¿La memoria larga en volatilidad es genuina o artefacto de regímenes? | **Caps. 11–12** |
| ¿Cómo se estima un modelo de volatilidad estocástica (MCMC)? | **Cap. 12** |
| ¿Qué proxy y qué función de pérdida son correctos para evaluar pronósticos de volatilidad? | Literatura específica; **fuera de Tsay** |
| ¿Cómo se convierte un pronóstico de volatilidad en decisión de trading? | **Fuera de Tsay** |
| ¿Existe prima de riesgo estable en futuros? | Empírica; **no resuelta en el capítulo** |
| ¿Cuál es la definición apropiada del "día" en mercados de 24 horas para estimadores OHLC? | **Fuera de Tsay**; convención a decidir empíricamente |

---

# 15. Checklist de conocimientos adquiridos

Debe poder responderse cada punto **con palabras propias y sin mirar** antes de pasar al Capítulo 4.

## Conceptos fundamentales
- [ ] Explicar qué es la volatilidad y en qué se diferencia del retorno.
- [ ] Explicar **por qué la volatilidad no se observa directamente** y qué consecuencia tiene eso.
- [ ] Distinguir varianza incondicional de varianza condicional, con un ejemplo.
- [ ] Explicar qué significa la notación $\text{Var}(r_t\mid\mathcal{F}_{t-1})$, símbolo por símbolo.
- [ ] Explicar qué significa "heterocedasticidad condicional" sin usar jerga.
- [ ] Definir residuo, shock e innovación, y explicar por qué son el mismo objeto.
- [ ] Explicar la ecuación $a_t=\sigma_t\epsilon_t$ y qué papel juega cada factor.
- [ ] Explicar por qué modelar $\text{Var}(r_t\mid\mathcal{F}_{t-1})$ es lo mismo que modelar $\text{Var}(a_t\mid\mathcal{F}_{t-1})$.

## Características de la volatilidad
- [ ] Enumerar las cuatro características que lista Tsay.
- [ ] Explicar el volatility clustering y por qué es compatible con estacionariedad.
- [ ] Saber que Tsay **no ofrece una explicación causal** del clustering, y que lo señala como debilidad de ARCH.
- [ ] Explicar qué es el leverage effect y **por qué no debe extrapolarse a todos los futuros**.

## Estructura y construcción
- [ ] Distinguir ecuación de media de ecuación de volatilidad.
- [ ] Enumerar los cuatro pasos del procedimiento de construcción.
- [ ] Explicar por qué se modela primero la media.
- [ ] Explicar por qué se elevan los residuos al cuadrado, y tres limitaciones de esa idea.
- [ ] Saber que $a_t^2$ es un estimador **insesgado pero ineficiente** de $\sigma_t^2$.

## Tests
- [ ] Explicar qué pregunta responde el Ljung–Box sobre $a_t^2$.
- [ ] Explicar la lógica del test LM de Engle en una frase.
- [ ] Enunciar la hipótesis nula de ambos tests.
- [ ] Explicar por qué **no rechazar no demuestra ausencia de efectos ARCH**.
- [ ] Distinguir $ACF(a_t)$ de $ACF(a_t^2)$ y qué indica cada una.
- [ ] Recordar los números del ejemplo de Intel: p = 0.11 sobre $r_t$, p = 5×10⁻¹⁴ sobre $a_t^2$.

## ARCH
- [ ] Escribir y explicar la ecuación ARCH($m$) en palabras.
- [ ] Explicar qué representa cada parámetro y por qué deben ser no negativos.
- [ ] Explicar la mecánica del clustering en ARCH, incluida la precisión de "tiende a".
- [ ] Calcular la varianza incondicional $\alpha_0/(1-\alpha_1)$ y explicar qué significa.
- [ ] Enumerar las cuatro debilidades que lista Tsay.
- [ ] Explicar por qué ARCH "sobrepredice tras shocks aislados".
- [ ] Saber que la PACF de $a_t^2$ sirve para elegir el orden, y sus dos limitaciones.

## GARCH
- [ ] Escribir GARCH(1,1) y traducirlo a palabras.
- [ ] Explicar la diferencia entre $\alpha_1$ y $\beta_1$ con una analogía.
- [ ] Explicar qué mide $\alpha_1+\beta_1$ y qué significan valores bajos, altos y cercanos a 1.
- [ ] Explicar la equivalencia GARCH ↔ ARMA sobre $a_t^2$ y por qué importa.
- [ ] Saber que $\eta_t$ es diferencia de martingala pero **no** iid.
- [ ] Explicar el pronóstico a 1 paso y a múltiples pasos.
- [ ] Explicar hacia qué valor convergen los pronósticos y bajo qué condición.
- [ ] Explicar qué es el residuo estandarizado y cómo se usa para verificar.

## Familia GARCH
- [ ] Explicar qué problema resuelve cada miembro: IGARCH, GARCH-M, EGARCH, TGARCH.
- [ ] Explicar por qué IGARCH sin constante es EWMA y es RiskMetrics.
- [ ] Explicar por qué **$\hat\alpha+\hat\beta\approx1$ no demuestra IGARCH** (cuatro razones).
- [ ] Distinguir raíz unitaria en el nivel (Cap. 2) de IGARCH (Cap. 3).
- [ ] Distinguir "predecir la volatilidad" de "usar la volatilidad para predecir el retorno".
- [ ] Saber que la prima de riesgo **no resultó significativa** en los ejemplos de Tsay.
- [ ] Explicar por qué EGARCH usa logaritmos.
- [ ] Explicar la diferencia conceptual entre EGARCH y TGARCH.
- [ ] Leer los números del ejemplo de TGARCH sobre IBM: 0.0658 vs 0.1501.

## Evaluación
- [ ] Explicar por qué evaluar un pronóstico de volatilidad es difícil.
- [ ] Explicar la diferencia entre "consistente" y "preciso" con el ejemplo de la altura.
- [ ] Enumerar los proxies de volatilidad disponibles y sus limitaciones.
- [ ] Explicar por qué un $R^2$ bajo contra $r_t^2$ **no** es evidencia de mal pronóstico.
- [ ] Distinguir buen ajuste in-sample de buen pronóstico out-of-sample, y citar el caso de SV.

## Alternativas
- [ ] Explicar la diferencia fundamental entre GARCH y volatilidad estocástica.
- [ ] Explicar qué es $v_t$ y por qué complica la estimación.
- [ ] Saber que las contribuciones de SV al pronóstico out-of-sample fueron "mixtas".
- [ ] Explicar qué es la memoria larga en volatilidad y en qué se diferencia de persistencia GARCH alta.
- [ ] Saber que la estimación mediana de $d$ es ≈0.38 **en acciones del S&P 500**.
- [ ] Escribir y explicar la fórmula de la volatilidad realizada.
- [ ] Explicar el trade-off entre frecuencia de muestreo y ruido de microestructura.
- [ ] Explicar el problema del gap overnight y en qué activos es menor.
- [ ] Explicar qué información adicional aporta OHLC respecto del cierre.
- [ ] Explicar qué captura cada estimador (Parkinson, Rogers–Satchell, Yang–Zhang) y qué supone.
- [ ] Explicar el sesgo del rango observado y de qué depende.

## Colas pesadas
- [ ] Explicar los tres caminos hacia un retorno extremo (casos A, B, C).
- [ ] Explicar con un ejemplo numérico cómo mezclar regímenes genera exceso de kurtosis.
- [ ] Saber que un ARCH(1) **con shocks normales** genera colas pesadas.
- [ ] Explicar por qué **$\alpha_1$, y no $\beta_1$**, es el parámetro responsable de las colas.
- [ ] Explicar qué significa que la kurtosis teórica **no exista**.
- [ ] Saber que las colas de GARCH pueden ser insuficientes incluso con Student-t.
- [ ] Enunciar la hipótesis empírica que queda: ¿cuánta kurtosis desaparece al estandarizar?

## Lo que NO se sabe todavía
- [ ] Todo lo listado en la sección 14.

---

# 16. Conclusiones

## 16.1 Qué aportó realmente el Capítulo 3

**1. Convirtió una observación en un objeto modelable.** Los Capítulos 1 y 2 dijeron que la magnitud de los retornos tiene memoria. El Capítulo 3 introduce $\sigma_t^2$ como **variable dinámica**, con ecuaciones, parámetros interpretables, pronósticos derivables y diagnósticos verificables. Es el paso de "observamos un fenómeno" a "tenemos un lenguaje para describirlo".

**2. Separó formalmente dos problemas que suelen confundirse.** La estructura $r_t=\mu_t+a_t$ con $a_t=\sigma_t\epsilon_t$ deja explícito que predecir hacia dónde y predecir cuánto son preguntas distintas. Toda la evidencia del capítulo sugiere que la segunda tiene respuesta y la primera mucho menos — el escenario 3 de nuestra tabla. **Ésta es la contribución conceptual más importante para el proyecto**, porque reorienta qué es razonable esperar de un sistema predictivo.

**3. Respondió la pregunta que dejamos abierta en el Capítulo 1.** El Capítulo 3 demuestra que una volatilidad condicional variable puede generar exceso de kurtosis incluso cuando los shocks estandarizados son normales. Esto establece un vínculo directo entre volatility clustering y heavy tails, pero no demuestra que todas las colas pesadas provengan exclusivamente de la volatilidad variable. El propio capítulo permite que coexistan dos fuentes: dinámica de volatilidad y shocks cuya distribución condicional ya tenga colas pesadas. Por eso, la pregunta empírica más útil pasa a ser: “¿cuánta de la no normalidad marginal desaparece al condicionar o estandarizar por volatilidad?”.

**4. Entregó una familia de herramientas ordenada por el problema que resuelve.** ARCH resuelve la formalización. GARCH resuelve la parsimonia. IGARCH formaliza el caso límite y resulta ser EWMA. EGARCH y TGARCH resuelven la asimetría. GARCH-M explora la conexión con el retorno. Cada uno existe porque alguien detectó un límite del anterior — y saber **qué límite** es más útil que saber la fórmula.

**5. Mostró que hay más de una forma de acceder a la volatilidad.** GARCH la infiere; la volatilidad estocástica le da su propia aleatoriedad; la volatilidad realizada y los estimadores OHLC **la miden directamente**. Esta última familia es especialmente relevante porque no requiere estimar parámetros y porque los factores de eficiencia de 5 a 8 respecto del cierre-a-cierre son sustanciales.

**6. Fue honesto sobre lo que no funciona.** Tres ejemplos que conviene retener: la prima de riesgo de GARCH-M **no resultó significativa**; la volatilidad estocástica mejoró el ajuste pero sus contribuciones al pronóstico fueron **"mixtas"**; y las colas de GARCH siguen siendo **"demasiado cortas"** para alta frecuencia incluso con Student-t. Un capítulo que sólo mostrara éxitos sería menos útil.

**7. Planteó el problema de la evaluación como problema genuino.** No se puede evaluar un pronóstico contra algo que no se observa. La observación de que el $R^2$ bajo contra $r_t^2$ **no es evidencia de mal pronóstico** es de las que evitan conclusiones equivocadas más adelante.

## 16.2 Qué todavía no sabemos

**Sobre el fenómeno.** No sabemos por qué la volatilidad se agrupa. Tsay lo señala como debilidad de ARCH y dice explícitamente que la causa de la persistencia *"merece una investigación cuidadosa"*. Los modelos describen sin explicar, y eso limita cuánta confianza podemos tener en que los patrones persistan ante cambios estructurales.

**Sobre la persistencia.** No sabemos si la persistencia extremadamente alta que se observa es una propiedad genuina, memoria larga verdadera, o un artefacto de cambios de régimen no modelados. Las tres explicaciones son compatibles con los datos y llevan a pronósticos de largo horizonte muy distintos. → **Caps. 11–12.**

**Sobre las colas.** Sabemos que la volatilidad variable genera colas pesadas. **No sabemos cuánta** de la kurtosis observada en un futuro concreto es atribuible a eso y cuánta a colas condicionales genuinas. Es la hipótesis H3.13, que ahora tiene un método concreto asociado.

**Sobre la medición.** No sabemos qué proxy de volatilidad es apropiado para futuros, a qué frecuencia, ni cómo tratar el gap de fin de semana, el roll, la heterogeneidad de las 24 horas y la discreción del tick. Casi todo eso requiere el **Cap. 5**.

**Sobre la utilidad.** **No sabemos si nada de esto produce dinero.** El capítulo entrega conocimiento sobre riesgo. Convertirlo en rentabilidad requiere un mecanismo que el capítulo no provee y que, en el único caso donde lo explora (GARCH-M), no resulta significativo.

**Sobre lo que queda después.** No sabemos si el shock estandarizado $\epsilon_t$ es realmente iid, como todo el marco supone. Si no lo fuera, quedaría estructura que ningún modelo de este capítulo captura. → **Cap. 4.**

## 16.3 El aporte neto para el proyecto

El Capítulo 3 no entrega un modelo de trading, y no debe adoptarse ninguno de sus modelos como componente del sistema en esta etapa. Entrega tres cosas:

- **Un vocabulario preciso** para hablar de incertidumbre condicional, que resulta ser exactamente el vocabulario que un modelo probabilístico de ML necesita: $(\hat\mu_t,\hat\sigma_t)$, verosimilitud, residuo estandarizado.
- **Una jerarquía de benchmarks de volatilidad**, desde volatilidad constante hasta volatilidad realizada, contra la cual cualquier propuesta debe justificarse.
- **Una lista de lo que no se puede concluir**: que la volatilidad predecible implique retorno predecible; que la asimetría de acciones se traslade a futuros; que más frecuencia siempre mejore la medición; que mejor ajuste implique mejor pronóstico; que las colas pesadas requieran shocks de colas pesadas.

La tercera es, otra vez, la más valiosa. Los errores que enumera son exactamente los que producen sistemas que parecen sofisticados y descansan sobre inferencias inválidas.

---

# 17. Registro de revisión crítica

Auditoría de las afirmaciones **[B]** sensibles, con atención especial a los términos "debe", "siempre", "demuestra", "óptimo", "necesariamente", "implica" e "invalida".

| Afirmación [B] sensible | Riesgo de sobreinterpretación | Estado final |
|---|---|---|
| "El escenario 3 (dirección impredecible, volatilidad predecible) es el más relevante en mercados financieros" | Convertir la evidencia de Tsay sobre acciones e índices estadounidenses en una ley sobre mercados en general | **MATIZAR** — el texto ya dice "la evidencia de los Caps. 2 y 3 **sugiere** que es el más relevante"; en futuros es hipótesis (H3.1, H3.8) |
| "La volatilidad predecible sirve para dimensionar, para saber cuándo no operar, para calibrar probabilidades" | Presentar usos hipotéticos como beneficios establecidos | **MANTENER** — marcado explícitamente como interpretación propia, con la acotación de que no garantiza rentabilidad |
| "El leverage effect puede tener signo opuesto en commodities" | Afirmar un resultado empírico no verificado | **MATIZAR** → ya formulado como "cabe esperar" y "es plausible"; convertido en hipótesis H3.5 con ambos signos posibles |
| "En futuros de 24 horas el problema del gap overnight es menor" | Convertir una expectativa estructural en hallazgo | **PREGUNTA ABIERTA** — marcado como expectativa; convertido en H3.11 |
| "GARCH es una mixtura de escala donde la escala es predecible desde el pasado" | Atribuir a Tsay una conexión que él no enuncia | **MANTENER** marcado como [B] — es una lectura del propio material, señalada como interpretación |
| "Un GARCH forzado a aproximar memoria larga elegirá el decaimiento exponencial más lento posible" | Explicación causal de $\hat\alpha+\hat\beta\approx1$ no demostrada | **MANTENER** marcado como [B] con la nota explícita de que Tsay no lo enuncia; presentado como una de cuatro explicaciones posibles |
| "Es la estufa ($\alpha$) y no la aislación ($\beta$) la que genera las colas pesadas" | La analogía podría sugerir más de lo que el resultado dice | **MANTENER** — el resultado subyacente es [A] textual: "si $\alpha_1=0$... el modelo no tiene colas pesadas" |
| "Con un día de datos OHLC se obtiene la precisión de cinco días de cierres" | Presentar un factor teórico como garantía práctica | **MATIZAR** → el texto ya condiciona a "bajo el modelo de difusión simple de Garman y Klass"; añadida la advertencia en el error 13 |
| "Rogers–Satchell es robusto a la presencia de drift" | Atribuir a Tsay una caracterización que él no hace | **MANTENER** marcado como [B] con nota explícita de que no está enunciado así por Tsay |
| "El ejemplo IBM → S&P 500 es evidencia cross-asset" | Extrapolar de una acción y su índice a relaciones entre futuros | **MATIZAR** → el texto ya aclara "aunque sea entre una acción y su índice, no entre futuros" |
| "Los tests con series intradía largas casi siempre rechazan" | Afirmación sobre potencia de tests no verificada en nuestros datos | **MANTENER** — es una propiedad general de los tests de significancia con $T$ grande, señalada como limitación en H3.1 |
| "Un patrón intradía determinista produciría autocorrelación en $a_t^2$ que no es dinámica ARCH" | Afirmar un mecanismo no verificado | **PREGUNTA ABIERTA** — convertido en H3.17, con remisión al Cap. 5 |
| "La estandarización por volatilidad debe ser causal o hay leak" | Uso de "debe": regla normativa | **MANTENER** — es un requisito lógico de la validación temporal, no una preferencia empírica; la condición (que se evalúe out-of-sample) está explícita |
| "El residuo estandarizado $r_t/\hat\sigma_t$ podría estabilizar la distribución y hacer el problema más aprendible" | Convertir una posibilidad en propuesta | **PREGUNTA ABIERTA** — formulado como pregunta en T3, con la advertencia de que estabilizar no implica mejorar el aprendizaje |
| "Un modelo con predicción puntual buena y residuos al cuadrado autocorrelacionados emite incertidumbre mal caracterizada" | Uso de "mal caracterizada" como si fuera categórico | **MANTENER** — se sigue directamente de la definición de $\sigma_t^2$; no afirma que las predicciones puntuales sean malas |
| "Que SV no mejore el pronóstico es un aviso para arquitecturas de ML" | Extrapolar de modelos econométricos clásicos a ML | **MATIZAR** → el texto ya aclara "no es un argumento contra la complejidad; es un argumento a favor de exigirle evidencia out-of-sample" |
| "Ninguno de EGARCH o TGARCH es mejor en abstracto" | Podría leerse como que da igual cuál usar | **MANTENER** — el punto es que son parametrizaciones distintas del mismo fenómeno; la elección es empírica (H3.6) |
| "Los valores $\hat\alpha_1\approx0.12$, $\hat\beta_1\approx0.85$ no son constantes de la naturaleza" | Ninguno; es una acotación restrictiva | **MANTENER** |
| "La jerarquía de benchmarks de volatilidad (niveles 0 a 5)" | Presentar una construcción propia como si fuera de Tsay | **MANTENER** marcado como [B] — cada nivel corresponde a un modelo que Tsay sí describe, pero la ordenación es propia |
| "En futuros, la definición del día es una convención que afecta a los estimadores OHLC" | Afirmación sobre un efecto no cuantificado | **MANTENER** — es una consecuencia lógica de que los estimadores usan $O$, $H$, $L$, $C$ de un período definido; la magnitud queda como PREGUNTA ABIERTA |

## Patrón detectado en esta revisión

En el Capítulo 1 el riesgo dominante fue convertir observaciones en reglas normativas. En el Capítulo 2 fue generalizar los ejemplos empíricos de Tsay —todos sobre acciones estadounidenses— a futuros y alta frecuencia.

**En el Capítulo 3 el riesgo dominante es distinto y más sutil: confundir "existe estructura predecible" con "existe utilidad económica".** El capítulo demuestra convincentemente que la volatilidad tiene estructura predecible, y esa demostración es tan sólida que invita a dar el paso siguiente sin justificarlo. Seis de las dieciocho afirmaciones auditadas en §13 son variantes de ese salto (números 3, 16, 17, 18, y parcialmente 4 y 5).

**Criterio adoptado para el Capítulo 4:** cada vez que se afirme que existe estructura, dependencia o predictibilidad de algún tipo, indicar explícitamente **(i)** en qué momento condicional vive esa estructura, **(ii)** si se ha verificado fuera de muestra, y **(iii)** qué mecanismo la convertiría en decisión de trading — dejando claro cuando no hay ninguno.
