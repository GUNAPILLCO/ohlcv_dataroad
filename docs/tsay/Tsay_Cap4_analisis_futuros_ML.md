# Tsay — Capítulo 4: Nonlinear Models and Their Applications
## Informe de estudio para el Proyecto IRIS

**Convenciones de este documento**

- **[A]** = definición, resultado, ejemplo empírico o afirmación que proviene directamente de Tsay (Analysis of Financial Time Series, 3ª ed., Capítulo 4).
- **[B]** = interpretación, extensión o adaptación propia hacia futuros, trading o Machine Learning. Nunca es una decisión de diseño; es una hipótesis a evaluar.
- **PREGUNTA ABIERTA** = cuestión que este capítulo no permite resolver con la información disponible.
- Cada resultado empírico [A] de Tsay indica, cuando está disponible: serie/instrumento, frecuencia y período. Ningún resultado sobre otra serie (desempleo, acciones, índices) se generaliza automáticamente a futuros.
- Este informe es exclusivamente de **comprensión y adquisición de conocimiento**. No define features, targets, arquitecturas ni protocolos de validación para IRIS.

---

## 1. Resumen ejecutivo

Los Capítulos 2 y 3 nos enseñaron a mirar dos cosas separadas en una serie de retornos: la parte que se puede predecir en promedio (la media condicional) y la parte que cambia en cuánto se mueve el precio (la varianza condicional, tratada por ARCH/GARCH). En ambos casos asumíamos, implícitamente, que la "regla" que gobierna esa media o esa varianza es siempre la misma, sin importar el contexto: los mismos coeficientes, aplicados de la misma manera, todo el tiempo.

El Capítulo 4 rompe ese supuesto. Se pregunta: **¿y si la regla misma cambia según el contexto?** [B] En otras palabras: ¿y si el mercado no responde siempre igual al mismo estímulo? Un modelo lineal, del tipo "cada unidad de $x$ produce siempre el mismo efecto sobre $y$", puede ser una aproximación pobre si en realidad existen umbrales, transiciones graduales, regímenes ocultos o interacciones entre variables.

Tsay presenta una familia de modelos que intentan capturar ese tipo de estructura: modelos de umbral (TAR), modelos de transición suave (STAR), modelos de cambio de régimen oculto (Markov Switching), métodos no paramétricos (que dejan que los datos sugieran la forma de la relación), modelos donde el coeficiente depende del contexto (functional coefficient), modelos aditivos no lineales, una introducción conceptual a los modelos de espacio de estados no lineales, y finalmente las redes neuronales feed-forward, presentadas como una manera flexible y genérica de aproximar funciones no lineales.

**[A]** Tsay muestra que no existe un único test que domine a los demás para detectar no linealidad y que detectar no linealidad no determina automáticamente qué modelo no lineal debe utilizarse. [B] Además, esa evidencia tampoco demuestra por sí sola que la relación será estable fuera de muestra ni que tendrá utilidad económica. Tsay ilustra esta dificultad con un ejemplo muy instructivo: una red neuronal entrenada para predecir la dirección de los retornos mensuales de la acción de IBM, a pesar de utilizar una arquitectura capaz de representar relaciones no lineales, no logra superar a un modelo trivial de "camino aleatorio con deriva" en la muestra de evaluación fuera de muestra (Ejemplo 4.8, retornos mensuales de IBM, 1926–1999, evaluación en 1998–1999).

**[B]** Para el Proyecto IRIS, la lectura general de este capítulo no es "hay que usar redes neuronales" ni "hay que modelar regímenes". Es más modesta y más importante: existe un vocabulario preciso para describir *qué tipo* de estructura no lineal podría existir en un mercado de futuros, existen herramientas para *testear* si esa estructura está presente, y existe una advertencia reiterada de que **detectar estructura no es lo mismo que poder explotarla económicamente**. Ese es el guardrail que debe acompañar cualquier decisión futura sobre arquitectura de modelo.

Las cuatro nociones que hay que mantener separadas durante todo el capítulo son:

1. **No linealidad** — ¿la regla que conecta el pasado con el presente depende del contexto, en vez de ser una combinación lineal fija?
2. **Dependencia** — ¿hay alguna relación estadística, lineal o no, entre el pasado y el presente?
3. **Predictibilidad fuera de muestra** — ¿esa relación se mantiene y es explotable en datos que el modelo no vio durante el ajuste?
4. **Utilidad económica** — ¿esa predictibilidad, una vez descontados costos operativos y riesgo, produce un resultado económicamente valioso?

> Rechazar linealidad ≠ demostrar predictibilidad ≠ demostrar rentabilidad.

Este boxed statement aparece repetidas veces a lo largo del informe porque es, junto con la distinción entre régimen observable y régimen latente, el hilo conductor de todo el capítulo.

---

## 2. Qué significa realmente "no lineal"

### 2.1 El punto de partida: un modelo lineal

**[A]** Tsay define formalmente una serie lineal como aquella que puede escribirse como una combinación lineal (ponderada) de shocks pasados independientes:

$$x_t = \mu + \sum_{i=0}^{\infty} \psi_i a_{t-i}$$

donde $\mu$ es una constante, $\psi_i$ son números reales fijos ($\psi_0 = 1$) y $\{a_t\}$ es una secuencia de shocks independientes e idénticamente distribuidos. **Cualquier proceso que no pueda escribirse así se considera, por definición, no lineal** [A]. Los modelos ARMA del Capítulo 2 son un caso particular de esta forma lineal.

Antes de llegar a esa definición formal, conviene partir de la versión más simple posible:

$$y = a + b \cdot x$$

En palabras: el efecto de $x$ sobre $y$ es siempre proporcional, y se rige por la misma regla $b$, sin importar en qué valor de $x$ estemos, en qué momento del tiempo, o en qué contexto. Si $x$ sube una unidad, $y$ sube siempre $b$ unidades — hoy, ayer, en un mercado tranquilo o en un mercado agitado.

### 2.2 Cuatro formas en que esa regla puede dejar de ser fija

**Umbral.** La regla cambia de golpe al cruzar un valor determinado: "si $x < 0$ ocurre una cosa; si $x \ge 0$ ocurre otra". No hay una única pendiente $b$; hay (al menos) dos, y el paso de una a otra es abrupto.

**Saturación.** Cambios pequeños de $x$ importan, pero cambios muy grandes dejan de tener un efecto proporcional — la relación se "aplana" en los extremos. Es el tipo de comportamiento que producen funciones como la logística, que aparece más adelante tanto en STAR como en redes neuronales.

**Interacción.** El efecto de $x$ sobre $y$ depende además del valor de otra variable $z$. No hay una pendiente única para $x$: la pendiente misma es una función de $z$. El modelo bilineal de la Sección 4.1.1 es, en esencia, una manera simple de introducir este tipo de término.

**Régimen.** La misma variable puede tener efecto positivo en un estado del sistema y negativo en otro. No se trata de un umbral sobre la propia variable explicativa, sino de un "estado" más general del sistema (observable o no) que determina qué regla aplicar.

**[A]** Formalmente, Tsay reescribe el problema en términos de la media y la varianza condicionales dado todo lo que se sabe hasta $t-1$ (la información disponible, que denota $\mathcal{F}_{t-1}$):

$$\mu_t = E(x_t \mid \mathcal{F}_{t-1}) \equiv g(\mathcal{F}_{t-1}), \qquad \sigma_t^2 = \mathrm{Var}(x_t \mid \mathcal{F}_{t-1}) \equiv h(\mathcal{F}_{t-1})$$

Si $g(\cdot)$ es una función lineal de la información pasada, la media es lineal; si $g(\cdot)$ no lo es, decimos que la serie **es no lineal en media**. Si $h(\cdot)$ cambia en el tiempo, decimos que la serie **es no lineal en varianza** (esto es exactamente lo que estudiaron los modelos ARCH/GARCH del Capítulo 3).

### 2.3 Qué NO significa "no lineal"

Es importante despejar una confusión común: **"no lineal" no significa "caótico", "complejo" ni "impredecible"** [A, con interpretación B añadida para enfatizar el punto]. Un modelo TAR de dos regímenes, por ejemplo, es perfectamente determinístico y fácil de escribir con lápiz y papel; simplemente no puede describirse con una única combinación lineal de coeficientes constantes. La no linealidad es una propiedad de la *forma* de la relación, no una medida de cuán difícil es de predecir, ni de cuán "inteligente" debe ser el modelo que se use.

**[B] Relevancia para futuros y ML.** Este vocabulario permite hacer una pregunta muy concreta sobre cualquier instrumento de futuros: ¿la relación entre el estado reciente del mercado y el retorno siguiente es razonablemente estable (una sola regla, aproximadamente constante), o hay evidencia de que esa regla depende del contexto (volatilidad, hora del día, dirección previa, magnitud del movimiento, etc.)? Es una pregunta empírica, no una que se pueda responder por intuición. Las secciones siguientes (4.1 a 4.2) dan el vocabulario y las herramientas para plantearla con precisión — sin, todavía, responderla para ningún instrumento en particular.

---

## 3. No linealidad en media vs. no linealidad en volatilidad (puente con el Capítulo 3)

**[A]** El propio Tsay conecta explícitamente ambos capítulos: los modelos de heterocedasticidad condicional del Capítulo 3 (ARCH/GARCH y variantes) son, en el lenguaje de la Sección 2.2 de este informe, modelos donde $h(\cdot)$ —la varianza condicional— cambia en el tiempo, mientras que $g(\cdot)$ —la media condicional— permanece simple. El Capítulo 4, en cambio, se ocupa mayoritariamente de extensiones donde **$g(\cdot)$, la ecuación de la media, es la que se vuelve no lineal**. La única excepción parcial que el propio Tsay señala son los modelos GARCH-M, donde la media depende de la varianza y por lo tanto también varía con el tiempo.

En palabras simples:

> **Capítulo 3:** la cantidad de ruido cambia (el mercado a veces es más agitado, a veces más tranquilo), pero la regla que intenta predecir la dirección sigue siendo la misma.
>
> **Capítulo 4:** la propia regla que intenta predecir el retorno puede cambiar — no solo cuánto ruido hay, sino cómo el pasado se traduce en una expectativa sobre el futuro.

**[A]** Ambos fenómenos pueden coexistir perfectamente, y de hecho varios de los ejemplos del capítulo los combinan explícitamente: el Ejemplo 4.3 construye un modelo AR(2)–TAR–GARCH(1,1) sobre los retornos diarios de IBM (1962–2003), donde hay un componente TAR *dentro* de la ecuación de varianza (para capturar la asimetría entre shocks positivos y negativos), no en la media. Esto ilustra que "TAR" o "umbral" no son sinónimos de "no linealidad en la media": la misma idea de umbral puede aplicarse tanto a $g(\cdot)$ como a $h(\cdot)$.

**[B]** Para un sistema de trading esto importa porque son dos preguntas distintas con implicancias distintas: "¿cambia la dirección esperada según el contexto?" (relevante para señales de entrada) versus "¿cambia el tamaño esperado del movimiento según el contexto?" (relevante para sizing, stops, y filtros de volatilidad). Un instrumento puede tener mucha estructura no lineal en uno de los dos aspectos y poca en el otro.

---

## 4. Modelos no lineales, sección por sección

### 4.1 Modelo bilineal (4.1.1)

**Problema que intenta resolver.** Si un modelo lineal es la aproximación de primer orden de una función general, ¿cuál es la aproximación de segundo orden? La respuesta natural es agregar términos que sean productos —no solo sumas— de las variables.

**Intuición.** Un modelo lineal solo puede sumar efectos: "un poco de esto, más un poco de aquello". Un término bilineal permite que el efecto de una variable dependa de otra, mediante un producto entre ambas. Esto es exactamente la idea de una **interacción**, presentada en la Sección 2 de este informe.

**[A] Definición técnica.** El modelo bilineal general es:

$$x_t = c + \sum_{i=1}^{p}\phi_i x_{t-i} - \sum_{j=1}^{q}\theta_j a_{t-j} + \sum_{i=1}^{m}\sum_{j=1}^{s}\beta_{ij}\, x_{t-i}\, a_{t-j} + a_t$$

El término nuevo respecto de un ARMA común es $\sum \beta_{ij}\, x_{t-i}\, a_{t-j}$: productos entre valores pasados de la serie y shocks pasados. **Cada símbolo**: $c$ es una constante; $\phi_i$ son los coeficientes autorregresivos habituales; $\theta_j$ son los coeficientes de medias móviles; $\beta_{ij}$ son los coeficientes bilineales que multiplican el producto $x_{t-i}a_{t-j}$; $a_t$ es el shock en $t$.

**Por qué un producto $x_t z_t$ introduce no linealidad.** Si el coeficiente de $x_t$ fuera fijo, el efecto de $x_t$ sobre la variable de interés sería siempre el mismo. Pero si el término efectivo es $\beta \cdot x_t \cdot z_t$, entonces el efecto de $x_t$ (la derivada parcial) es $\beta \cdot z_t$: depende del valor de $z_t$. El coeficiente "efectivo" de $x_t$ ya no es un número fijo, sino que varía según el valor de la otra variable.

**[A] Ejemplo empírico (Ejemplo 4.1).** Retornos mensuales simples del índice CRSP equal-weighted, enero 1926 a diciembre 2008 (996 observaciones). Tsay ajusta un modelo bilineal especial —con heterocedasticidad condicional incorporada vía términos $\beta_i a_{t-i}$— y lo compara con un AR(3)–ARCH(3) y, finalmente, señala que **un modelo AR(1)–GARCH(1,1) ajusta mejor los datos que ambos**. La conclusión explícita de Tsay en este ejemplo es que los modelos son "similares", pero no concluye que el bilineal sea superior; de hecho, termina prefiriendo GARCH.

**Qué NO concluye este ejemplo.** No demuestra que los términos bilineales sean generalmente útiles para retornos financieros, ni que superen a alternativas más simples (como GARCH) fuera de este caso puntual.

**[B] Relación con ML.** Un término bilineal $x_t z_t$ es conceptualmente el mismo objeto que una *interaction feature* en Machine Learning: una nueva variable construida como producto de otras dos, que permite a un modelo lineal capturar parte de una relación no lineal sin cambiar de familia de modelo. **No se propone aquí crear features bilineales para IRIS** — solo se señala la analogía conceptual.

**Preguntas abiertas.** ¿Qué combinaciones de variables (si las hay) tendrían una interpretación económica razonable en un contrato de futuros intradiario? Esto no se puede responder desde este capítulo.

---

### 4.2 Threshold Autoregressive Model — TAR (4.1.2)

Esta es, junto con Neural Networks y los tests de no linealidad, una de las secciones centrales del capítulo.

**¿Qué problema intenta resolver?** Muchos procesos económicos y financieros muestran asimetrías: suben distinto de como bajan, reaccionan distinto según si el shock previo fue positivo o negativo, etc. **[A]** Tsay lo introduce explícitamente motivado por "asimetría en los patrones de caída y de subida de un proceso". Un modelo TAR usa varios modelos lineales simples, cada uno válido solo en una región del espacio, en vez de forzar una única regla para todo el rango de valores.

**¿Qué es un threshold (umbral)?** Es un valor que divide el espacio de una variable en regiones. Por ejemplo: $z_t < c$ frente a $z_t \ge c$. No es un límite en el tiempo (como "antes de 2020" y "después de 2020"), sino un límite en el valor de una variable.

**¿Qué es un régimen?** Es el estado del sistema en el que se aplica una dinámica (una ecuación) diferente. **[B]** Un ejemplo puramente intuitivo, para fijar ideas y sin ninguna base empírica todavía: "régimen A = mercado tranquilo, régimen B = mercado alterado". Este es solo un ejemplo pedagógico; en TAR, el régimen se define matemáticamente por si una variable observada supera o no un umbral, no por una etiqueta cualitativa como "tranquilo" o "alterado".

**[A] Ejemplo mínimo dado por Tsay** (Ecuación 4.8), un TAR(1) de 2 regímenes simulado:

$$x_t = \begin{cases} -1.5\, x_{t-1} + a_t & \text{si } x_{t-1} < 0 \\ 0.5\, x_{t-1} + a_t & \text{si } x_{t-1} \ge 0 \end{cases}$$

con $a_t$ ruido blanco gaussiano. Aquí la variable de umbral es $x_{t-1}$ (el propio pasado inmediato de la serie), el delay es 1 (se usa el valor de un paso atrás) y el umbral (threshold) es 0.

**[A] Hallazgos de Tsay sobre este ejemplo simulado, que valen solo para este proceso concreto:** aunque el coeficiente del primer régimen (−1.5) sería explosivo si se aplicara siempre, el proceso completo es estacionario ("ergódico") porque el segundo régimen (0.5) es estable y "atrae" la serie de vuelta; el proceso resultante es asimétrico (tarda distinto en subir que en bajar) y no es "reversible en el tiempo" (no se ve igual si se lo mira al derecho o al revés); y, notablemente, aunque el modelo no tiene término constante, la media de la serie **no** es cero — depende de cuánto tiempo pasa el proceso en cada régimen.

**[A] Definición técnica general (SETAR — self-exciting TAR).** Un proceso $x_t$ sigue un modelo SETAR de $k$ regímenes con variable de umbral $x_{t-d}$ si:

$$x_t = \phi_0^{(j)} + \phi_1^{(j)} x_{t-1} + \cdots + \phi_p^{(j)} x_{t-p} + a_t^{(j)}, \quad \text{si } \gamma_{j-1} \le x_{t-d} < \gamma_j$$

**Cada símbolo**: $j = 1, \dots, k$ identifica el régimen; $d$ es el *delay* (cuántos pasos atrás se mira para decidir en qué régimen estamos); los $\gamma_j$ son los umbrales que dividen el espacio (con $\gamma_0 = -\infty$ y $\gamma_k = +\infty$); el superíndice $(j)$ indica que cada régimen tiene su propia constante $\phi_0^{(j)}$, sus propios coeficientes autorregresivos $\phi_i^{(j)}$ y su propio término de error $a_t^{(j)}$, con varianza $\sigma_j^2$ propia. El nombre "self-exciting" viene de que la variable de umbral es la propia serie rezagada ($x_{t-d}$); Tsay también menciona la generalización a un umbral basado en otra variable $z_t$ (el "open-loop TAR").

**Respondiendo las cinco preguntas pedidas:**

1. **¿Quién determina el régimen?** El valor de la variable de umbral en $t-d$ comparado contra los cortes $\gamma_j$. No hay ambigüedad ni estimación: es una regla determinística una vez fijados los umbrales y el delay.
2. **¿El régimen es observable?** Sí. En cuanto se conoce $x_{t-d}$ (o $z_{t-d}$), se sabe con certeza en qué régimen está el proceso en $t$. Esto es fundamental y se retoma en la Sección 5 de este informe.
3. **¿La transición es brusca?** Sí — es la característica definitoria de TAR/SETAR frente a STAR (ver Sección 4.3 más abajo). La ecuación de la media condicional tiene un "salto" exactamente en el umbral.
4. **¿Cada régimen puede tener coeficientes diferentes?** Sí, por diseño: constante, pendientes AR y varianza del error pueden diferir completamente entre regímenes. Si fueran iguales entre regímenes, el modelo se reduciría a un AR lineal común (Tsay lo señala explícitamente: "de otro modo, el número de regímenes podría reducirse").
5. **¿Qué significa que una relación cambie de signo entre regímenes?** En el ejemplo simulado de arriba, el coeficiente pasa de −1.5 a +0.5: el mismo rezago $x_{t-1}$ tiene un efecto de sentido opuesto según el régimen. Esto es precisamente lo que un modelo lineal único no puede representar: una única pendiente $\phi$ no puede ser simultáneamente −1.5 y +0.5.

**[A] Ejemplo 4.2 — Tasa de desempleo civil de EE.UU., mensual, ajustada estacionalmente, enero 1948 a marzo 2009 (735 observaciones).** Sobre la serie diferenciada, Tsay ajusta un TAR de 2 regímenes con umbral 0.1 sobre $y_{t-1}$ (el cambio mensual del mes anterior): cuando el cambio previo es pequeño ($y_{t-1} \le 0.1$) la dinámica es una, y cuando es grande ($y_{t-1} > 0.1$) es otra, con 460 y 262 observaciones en cada régimen respectivamente. La interpretación que da Tsay —marcada explícitamente como tal, no como hecho general— es que la dependencia dinámica parece más fuerte cuando el desempleo sube con fuerza, posiblemente porque eso dispara intervención de política económica.

**[A] Ejemplo 4.3 — Retornos diarios de IBM, en porcentaje, con dividendos, del 3 de julio de 1962 al 31 de diciembre de 2003 (10.446 observaciones).** Aquí el TAR se usa **en la ecuación de varianza**, no en la de media: un modelo AR(2)–TAR–GARCH(1,1) permite que el coeficiente de $\sigma^2_{t-1}$ y de $a^2_{t-1}$ cambien según el signo del shock anterior (indicador $N_{t-1}$ para $a_{t-1}<0$). Tsay reporta que este modelo captura la asimetría en volatilidad de forma más fuerte que un modelo GJR estándar, y que es el único de los tres modelos comparados (GARCH simple, TGARCH, AR-TAR-GARCH) cuyos residuos al cuadrado no muestran autocorrelación significativa remanente.

**⚠️ Advertencia explícita del enunciado, reforzada aquí [B].** No hay que confundir automáticamente un "régimen" TAR con los regímenes horarios definidos manualmente en un sistema de trading (por ejemplo, "apertura", "sesión regular", "overnight"). Un régimen TAR surge porque una variable *medida* supera un umbral y eso cambia, de forma verificada estadísticamente, la dinámica de la serie. Un régimen horario es una partición definida por el analista según el reloj, sin que medie ningún test estadístico. Ambos son formas legítimas de segmentar datos, pero son conceptualmente distintos y no deben mezclarse bajo la misma palabra sin aclarar cuál es cuál. Cualquier vínculo entre ambos debe marcarse **[B]**.

**Qué NO demuestra esta sección.** Que un modelo TAR ajuste mejor que un lineal en una serie particular (desempleo de EE.UU., o IBM) no implica que un TAR vaya a ajustar mejor en cualquier otra serie, ni en datos intradiarios, ni en otro instrumento.

**[B] Relevancia para futuros y ML.** El vocabulario de TAR (variable de umbral, delay, régimen observable, coeficientes por régimen) es directamente análogo a lo que en ML se llamaría un "modelo por segmentos" o, de forma más flexible, un árbol de decisión de profundidad 1 aplicado sobre una variable de contexto. Esto es una observación conceptual, no una recomendación de arquitectura.

---

### 4.3 Smooth Transition AR — STAR (4.1.3)

**¿Qué problema intenta resolver?** Un modelo TAR tiene una discontinuidad exactamente en el umbral: la media condicional "salta" de una fórmula a otra sin transición. **[A]** Tsay señala que esta discontinuidad ha sido criticada en la literatura (Chan y Tong, 1986; Teräsvirta, 1994) porque en muchos procesos económicos es más razonable pensar en una transición gradual que en un cambio abrupto.

**Analogía.**

> **TAR** funciona como un interruptor: ON u OFF, sin punto intermedio.
> **STAR** funciona como una perilla regulable: se puede estar parcialmente en un estado y parcialmente en otro, y el paso de uno a otro es continuo.

**[A] Definición técnica.** Un modelo STAR(p) de 2 regímenes:

$$x_t = c_0 + \sum_{i=1}^{p}\phi_{0,i}x_{t-i} + F\!\left(\frac{x_{t-d}-\theta}{s}\right)\left[c_1 + \sum_{i=1}^{p}\phi_{1,i}x_{t-i}\right] + a_t$$

**Cada símbolo**: $d$ es el delay; $\theta$ y $s$ son parámetros que fijan, respectivamente, la ubicación y la escala de la transición; $F(\cdot)$ es la **función de transición**, que toma valores entre 0 y 1 y suele tomar forma logística, exponencial, o de función de distribución acumulada. Cuando $F \to 0$, el modelo se comporta como el primer régimen puro; cuando $F \to 1$, se comporta como el segundo régimen puro; en el medio, es una **mezcla continua ponderada** de ambos.

**Qué determina la velocidad de la transición.** El parámetro $s$ (escala): valores pequeños de $s$ hacen que $F(\cdot)$ pase muy rápido de 0 a 1 (aproximándose a un salto tipo TAR), mientras que valores grandes producen una transición más lenta y gradual.

**[A] Dificultades de estimación que menciona Tsay.** La experiencia empírica muestra que los parámetros de transición $\theta$ y $s$ son difíciles de estimar con precisión: los errores estándar suelen ser grandes, con estadísticos t de alrededor de 1.0 — es decir, casi nunca claramente significativos —, lo cual complica la interpretación del modelo estimado. Esto es una limitación práctica señalada explícitamente por Tsay (citando a Teräsvirta, 1994), no una opinión externa.

**[A] Ejemplo 4.4 — Retornos mensuales simples de la acción de 3M, febrero 1946 a diciembre 2008.** Tsay compara un ARCH(2) simple con un modelo donde la varianza condicional combina un ARCH(2) "base" más un término de transición logística (con el parámetro de escala fijado a priori en 1000, para simplificar la estimación) que captura la asimetría entre shocks positivos y negativos. Para shocks negativos grandes, el modelo se aproxima a un ARCH(2) con coeficientes positivos "normales"; para shocks positivos grandes, se aproxima a un ARCH(2) con un coeficiente negativo (contraintuitivo, aunque de magnitud pequeña según el propio Tsay). Nuevamente, este es un modelo de **volatilidad**, no de la media.

**Diferencias TAR vs. STAR, de forma resumida:**

| Aspecto | TAR | STAR |
|---|---|---|
| Transición | Abrupta, en un punto exacto | Gradual, mediante una función continua |
| Continuidad de la media condicional | No es continua en el umbral | Es diferenciable |
| Número efectivo de "regímenes puros" | Exactamente $k$ | Dos extremos, con un continuo de mezclas entre ellos |
| Facilidad de estimación | Relativamente más simple | Los parámetros de transición suelen ser difíciles de estimar con precisión |

**Qué NO concluye Tsay.** El texto no afirma en ningún momento que STAR sea, en general, superior a TAR. De hecho, subraya la dificultad práctica de estimar sus parámetros de transición, lo cual es un costo real frente a la simplicidad de TAR. La elección depende del problema y debe evaluarse caso por caso.

**[B] Relevancia para futuros y ML.** La función de transición logística de STAR es matemáticamente la misma función de activación que aparece más adelante en redes neuronales (Sección 4.7 de este informe). Esto no es casualidad: ambas ideas —transición suave entre "estados" y activación no lineal de una neurona— provienen de la misma familia matemática.

---

### 4.4 Markov Switching Model (4.1.4)

Esta es una de las secciones conceptualmente más importantes del capítulo, porque introduce una distinción que se usará constantemente en el resto del estudio: **régimen observable vs. régimen latente (oculto)**.

**Distinción central.**

> **TAR / STAR:** el régimen depende de una variable que efectivamente observamos (por ejemplo, $x_{t-d}$). En cuanto conocemos ese valor, sabemos con certeza en qué régimen estamos.
>
> **Markov Switching:** el régimen es un **estado oculto**, denotado $S_t$, que **no observamos directamente**. Solo observamos los datos, y a partir de ellos tratamos de *inferir* una probabilidad de estar en cada estado: $P(S_t = 1 \mid \text{datos})$ o $P(S_t = 2 \mid \text{datos})$.

**[A] Definición técnica.** Un modelo de Markov Switching Autoregressive (MSA) de 2 estados:

$$x_t = \begin{cases} c_1 + \sum_{i=1}^{p}\phi_{1,i}x_{t-i} + a_{1t} & \text{si } s_t = 1 \\ c_2 + \sum_{i=1}^{p}\phi_{2,i}x_{t-i} + a_{2t} & \text{si } s_t = 2 \end{cases}$$

donde $s_t$ es una **cadena de Markov de primer orden** con probabilidades de transición $P(s_t = 2 \mid s_{t-1} = 1) = w_1$ y $P(s_t = 1 \mid s_{t-1} = 2) = w_2$.

**Qué es un estado latente.** Es una variable que gobierna el comportamiento del sistema pero que no forma parte de los datos observados; solo se puede inferir indirectamente, a partir de cómo se comportan los datos que sí vemos.

**Qué es una transición entre estados y qué es una matriz de transición.** La cadena de Markov describe la probabilidad de pasar de un estado a otro entre $t-1$ y $t$. Con solo 2 estados, esa "matriz" queda resumida por los dos números $w_1$ y $w_2$: la probabilidad de salir del estado 1 hacia el 2, y la probabilidad de salir del estado 2 hacia el 1.

**Qué significa persistencia de un régimen y probabilidad de permanecer/cambiar.** **[A]** Un $w_i$ pequeño significa que el proceso tiende a **permanecer** más tiempo en el estado $i$ — es decir, ese régimen es persistente. Tsay da la fórmula exacta: la duración esperada en el estado $i$ es $1/w_i$. Un $w_i$ grande implica cambios frecuentes; un $w_i$ pequeño implica regímenes largos y estables.

**[A] Diferencia práctica crucial en el pronóstico**, señalada explícitamente por Tsay: en un SETAR, si se observa $x_{t-d}$, el régimen en $t$ se conoce con certeza, y el pronóstico proviene de un único submodelo (mientras el horizonte no exceda el delay). En un MSA, en cambio, **nunca hay certeza** sobre en qué estado se está; el pronóstico es siempre una combinación ponderada (por las probabilidades de estado) de los pronósticos de los submodelos individuales.

**[A] Dificultad de estimación.** Tsay señala que estimar un MSA es más difícil que otros modelos porque los estados no son observables directamente; menciona el algoritmo EM (Hamilton, 1990) y métodos de Markov Chain Monte Carlo — MCMC (McCulloch y Tsay, 1994) como las dos vías usadas en la literatura. **PREGUNTA ABIERTA → Caps. 11–12**: el detalle de estos métodos de estimación (filtros, MCMC) queda fuera del alcance de este capítulo y se retomará en capítulos posteriores del libro.

**[A] Ejemplo intuitivo dado por el propio Tsay (Ejemplo 4.5) — tasa de crecimiento trimestral del PIB real de EE.UU., 1947.II a 1991.I, ajustada estacionalmente.** McCulloch y Tsay (1994) ajustan un MSA con $p=4$ vía MCMC. Los resultados: el estado 1 tiene una media de crecimiento de aproximadamente +0.965% (interpretado como "expansión") y el estado 2 de aproximadamente −1.288% ("contracción"). Las duraciones esperadas son de aproximadamente 11.31 trimestres para el estado de expansión y 3.69 trimestres para el de contracción — es decir, las expansiones duran, en promedio, unos 3 años y las contracciones cerca de un año. **Es importante notar que Tsay pone las etiquetas "expansión" y "contracción" *después* de ver los resultados**, no las impuso de antemano: el modelo solo entrega dos estados numerados; la interpretación económica es posterior. Este es exactamente el punto que el enunciado pide destacar: el nombre del estado es una interpretación posterior, no una propiedad matemática del modelo.

**Distinción crítica, reiterada por requerimiento explícito del estudio.** Que exista una familia de modelos donde la dinámica depende de un estado no observable **no implica** que "debamos detectar regímenes de mercado" como paso obligado de ningún sistema de trading. Es un objeto estadístico con supuestos, dificultades de estimación y limitaciones propias — no una receta.

---

## 5. Tres tipos de "régimen": una tabla para no confundir el término

**[B]** Dado que la palabra "régimen" aparece con significados distintos en TAR, STAR, Markov Switching y en el lenguaje común del trading, conviene una tabla comparativa explícita:

| Tipo de régimen | ¿Cómo se define? | ¿Lo observamos directamente? | Ejemplo |
|---|---|---|---|
| **Definido manualmente** | El analista traza un límite arbitrario, normalmente basado en el reloj o en una convención de mercado | Sí, trivialmente — es una regla que el analista fija | Sesión de apertura, sesión regular, sesión overnight |
| **Observable basado en umbral (TAR)** | Una variable observada se compara con uno o más umbrales y eso determina qué ecuación se aplica | Sí — conocida la variable de umbral en $t-d$, el régimen queda determinado | $x_{t-1} \ge 0$ en el ejemplo TAR simulado de la Sección 4.2 |
| **Transición suave observable (STAR)** | Una variable observada determina un peso de transición entre dos comportamientos extremos mediante una función continua | Sí conocemos la variable que gobierna la transición, pero el modelo puede estar parcialmente entre ambos comportamientos en vez de pertenecer 100% a uno | Una función de transición puede dar, intuitivamente, 80% de un comportamiento y 20% del otro |
| **Latente (Markov Switching)** | Un estado oculto que sigue una cadena de Markov; se infiere probabilísticamente a partir de los datos observados | No — solo se dispone de una probabilidad estimada de estar en cada estado | Estado 1 / Estado 2 en el Ejemplo 4.5 del PIB de EE.UU. |

Esta tabla es una construcción propia [B] a partir de las definiciones de Tsay; el objetivo es evitar que "régimen" se use como si siempre significara lo mismo dentro del Proyecto IRIS.

---

## 6. Métodos flexibles y no paramétricos

### 6.1 Métodos no paramétricos (4.1.5) — estudio serio pero selectivo

**Idea central.** **[A]** En lugar de imponer de antemano una forma funcional concreta (como "la relación es lineal" o "la relación es logística"), se deja que los propios datos sugieran la forma de la función. Formalmente, Tsay plantea el problema como $Y_t = m(X_t) + a_t$, donde $m(\cdot)$ es una función suave pero desconocida que se quiere estimar sin asumir su forma.

**Paramétrico vs. no paramétrico.** Un modelo paramétrico (un AR, un TAR, un STAR) tiene un número fijo de parámetros que hay que estimar, y una forma funcional ya decidida por el analista. Un método no paramétrico no fija la forma: en el límite, "aprende" la función a partir de promedios locales de los datos cercanos al punto de interés. **[A]** Tsay lo presenta mediante la idea de una media ponderada local: para estimar $m(x)$ en un punto $x$, se promedian los valores observados de $y_t$ cuyo $x_t$ está cerca de $x$, dando más peso a las observaciones más cercanas (kernel regression) o ajustando una recta local ponderada por cercanía (local linear regression).

**Qué ganamos en flexibilidad y qué perdemos en interpretabilidad y eficiencia.** **[A]** Tsay advierte explícitamente que los métodos no paramétricos "no están libres de costo": son altamente dependientes de los datos y pueden llevar fácilmente a sobreajuste (overfitting). Ganamos la capacidad de capturar formas que ningún modelo paramétrico simple anticiparía; perdemos una fórmula compacta e interpretable, y en general necesitamos muchos más datos para obtener una estimación confiable en cada región del espacio de variables.

**Efecto del tamaño de muestra y el "ancho de banda" (bandwidth).** **[A]** El parámetro clave de estos métodos es el *bandwidth* $h$: con $h$ muy pequeño, el estimador prácticamente reproduce los datos originales (sobreajuste); con $h$ muy grande, el estimador colapsa hacia el promedio simple de toda la muestra (subajuste, se pierde toda la estructura). Elegir $h$ bien —Tsay menciona el método plug-in y la validación cruzada leave-one-out— es en sí mismo un problema no trivial.

**Curse of dimensionality.** **[A]** Tsay señala este problema directamente en la Sección 4.1.7 (Nonlinear Additive AR, ver más abajo): cuando el número de variables explicativas crece, el "suavizado" (smoothing) necesario para estimar bien una función se vuelve exponencialmente más difícil, porque los datos se dispersan por un espacio cada vez más grande y cada vez hay menos observaciones "cerca" de cualquier punto dado.

**[A] Ejemplo 4.6 — tasa semanal de las letras del Tesoro de EE.UU. a 3 meses, 1970–1997 (1461 observaciones).** Tsay usa `lowess` para estimar de forma no paramétrica la media y la volatilidad condicional de los cambios semanales de tasa, en función del nivel de tasa de la semana anterior. Encuentra que la media condicional estimada es positiva para niveles bajos de tasa y se vuelve negativa para niveles altos — compatible con la intuición de reversión hacia un nivel medio, aunque el propio efecto estimado es de magnitud muy pequeña ("esencialmente cero" en la escala gruesa del gráfico).

**[B] Conexión conceptual con ML** (solo como analogía, no como propuesta): la idea de "dejar que los datos determinen la forma de la función" es el principio detrás de casi cualquier modelo moderno de Machine Learning no lineal (árboles, bosques aleatorios, redes neuronales). El kernel regression de Tsay es, en cierto sentido, un antepasado directo de métodos como k-nearest-neighbors o de los kernels usados en Support Vector Machines.

### 6.2 Functional Coefficient AR — FAR (4.1.6)

**Idea.** En un AR simple, $x_t = \phi\, x_{t-1} + a_t$, el coeficiente $\phi$ es un número fijo. **[A]** En un modelo de coeficiente funcional, ese coeficiente se convierte en una función de otras variables observadas:

$$x_t = f_1(X_{t-1})\, x_{t-1} + \cdots + f_p(X_{t-1})\, x_{t-p} + a_t$$

donde $X_{t-1}$ es un vector de variables (que puede incluir rezagos de la propia serie u otras variables explicativas conocidas en $t-1$), y las funciones $f_i(\cdot)$ se estiman de forma no paramétrica (kernel o local linear regression), típicamente cuando la dimensión de $X_{t-1}$ es baja.

En palabras simples: **la influencia del pasado sobre el presente depende del contexto**, y esa dependencia no está restringida a una forma concreta (un umbral, una logística), sino que se deja libre.

**Cómo difiere de TAR y de STAR.** TAR y STAR imponen una forma específica para cómo cambia el coeficiente (un salto en TAR, una función logística/exponencial en STAR). FAR no impone ninguna forma concreta: el propio $f_i(\cdot)$ se estima de los datos, sin comprometerse a que sea un salto o una curva suave de una familia particular.

**[B] Ejemplo puramente ilustrativo, sin base empírica en Tsay para este capítulo**, marcado explícitamente como interpretación propia: "en baja volatilidad, un rezago podría tener un coeficiente de +0.1; en alta volatilidad, podría ser −0.2". Esto no es un resultado de Tsay; es solo un ejemplo para ilustrar qué significaría, en un contexto financiero, que $\phi$ dependiera de una variable de contexto como la volatilidad.

**[A]** Tsay cita a Cai, Fan y Yao (2000), quienes aplican el método de regresión local lineal para estimar $f_i(\cdot)$ y reportan mejoras sustanciales en pronósticos a un paso — pero esto se reporta como una referencia bibliográfica del propio Tsay, no se reproduce el detalle del estudio en el capítulo, por lo que no se puede evaluar aquí el instrumento, la frecuencia ni el período de esos datos.

### 6.3 Nonlinear Additive AR — NAAR (4.1.7)

**Problema que resuelve.** Aplicar métodos no paramétricos directamente a un AR(p) general, $x_t = f(x_{t-1}, \dots, x_{t-p}) + a_t$, requiere "suavizar" en un espacio de $p$ dimensiones simultáneamente — el problema de la maldición de la dimensionalidad mencionado arriba. **[A]** La solución que describe Tsay es asumir que la función se puede descomponer en una suma de funciones univariadas, una por cada rezago:

$$x_t = f_0(t) + \sum_{i=1}^{p} f_i(x_{t-i}) + a_t$$

En palabras: **cada rezago puede tener su propia relación no lineal con el presente**, pero esas relaciones se suman en vez de combinarse de forma más compleja. Como cada $f_i(\cdot)$ depende de una sola variable, puede estimarse con suavizado unidimensional — mucho más simple y con menos necesidad de datos que un suavizado conjunto de $p$ dimensiones.

**AR lineal vs. efecto no lineal, para el mismo rezago:**

| | AR lineal | NAAR |
|---|---|---|
| Contribución del rezago $x_{t-1}$ | $\phi \cdot x_{t-1}$ (proporcional, pendiente fija) | $f_1(x_{t-1})$ (forma libre, puede no ser proporcional) |

**Limitación explícita.** **[A]** Tsay señala que el supuesto de aditividad ("las contribuciones de cada rezago se suman, sin interactuar entre sí") es restrictivo y debe examinarse con cuidado en cada aplicación; cita tests de aditividad desarrollados por Chen, Liu y Tsay (1995). Es decir, NAAR resuelve el problema de dimensionalidad al costo de prohibir, por construcción, que dos rezagos interactúen entre sí (lo cual sí permitiría, por ejemplo, un modelo bilineal o una red neuronal).

### 6.4 Nonlinear State-Space Model (4.1.8) — nivel conceptual

**[A]** Tsay presenta este modelo en un nivel introductorio, siguiendo a Carlin, Polson y Stoffer (1992):

$$S_t = f_t(S_{t-1}) + u_t, \qquad x_t = g_t(S_t) + v_t$$

**Cada componente:**

- $x_t$: la **variable observada** — lo que efectivamente medimos.
- $S_t$: el **estado oculto** — una variable no observada directamente que resume la condición interna del sistema en $t$.
- La primera ecuación ($S_t = f_t(S_{t-1}) + u_t$) es la **ecuación de estado**: describe cómo evoluciona el estado oculto de un período a otro.
- La segunda ecuación ($x_t = g_t(S_t) + v_t$) es la **ecuación de observación**: describe cómo el estado oculto se traduce en lo que efectivamente observamos, con ruido de medición $v_t$.

**[A]** Tsay señala que hacer esto de forma no lineal exige métodos de Monte Carlo, porque —a diferencia del caso lineal-gaussiano (el filtro de Kalman clásico)— hace falta la distribución condicional completa del estado, no solo su media y varianza. También señala una limitación práctica importante: el método asume que se **conocen** las funciones $f_t(\cdot)$ y $g_t(\cdot)$, lo cual puede ser poco realista; sugiere que los métodos no paramétricos de FAR/NAAR podrían usarse para especificar esas funciones antes de aplicar el modelo de espacio de estados.

**PREGUNTA ABIERTA → Caps. 11 (state-space y Kalman) y 12 (MCMC).** El detalle técnico de cómo se estima este tipo de modelo (filtros, partículas, MCMC) queda explícitamente fuera del alcance de este capítulo, tal como indica el propio Tsay al remitir a capítulos posteriores.

**[B] Conexión conceptual.** Este marco de "estado oculto + observación ruidosa" es el mismo lenguaje general que sustenta tanto la volatilidad estocástica (donde el estado oculto es la volatilidad misma) como el Markov Switching (donde el estado oculto es discreto, en vez de continuo). Los tres comparten la misma estructura conceptual: hay algo que no vemos directamente, y tratamos de inferirlo a partir de lo que sí vemos.

---

## 7. Neural Networks (4.1.9)

Esta es una de las secciones más importantes del informe, tal como lo pide el estudio.

### 7.1 Desde cero: los componentes de una red feed-forward

**[A]** Tsay presenta una red neuronal *feed-forward* (hacia adelante) con una capa oculta, ilustrada con un ejemplo concreto: dos entradas (*input*), tres nodos en la capa oculta (*hidden layer*), una salida (*output*) — lo que él llama una red "2–3–1".

- **Input (entrada):** los valores que alimentan la red — por ejemplo, rezagos de retornos pasados ($r_{t-1}, r_{t-2}, \dots$).
- **Weight (peso):** un número que multiplica a cada entrada antes de sumarla; determina cuánta influencia tiene esa entrada sobre el nodo al que llega.
- **Bias (sesgo o constante):** un número que se suma independientemente de las entradas — el equivalente de la constante $c_0$ en un modelo AR.
- **Hidden unit / hidden layer (unidad / capa oculta):** nodos intermedios entre la entrada y la salida, cada uno de los cuales combina las entradas (pesos + sesgo) y les aplica una función de activación.
- **Activation function (función de activación):** una función no lineal que se aplica al resultado de la combinación ponderada de entradas. **[A]** Tsay usa típicamente la función logística: $f(z) = \dfrac{\exp(z)}{1+\exp(z)}$.
- **Output (salida):** el resultado final de la red, obtenido combinando las salidas de la capa oculta (con sus propios pesos y sesgo), y opcionalmente pasado por otra función de activación.

**[A] Definición técnica del nodo $j$ de la capa oculta:**

$$h_j = f_j\!\left(\alpha_{0j} + \sum_{i \to j} w_{ij}\, x_i\right)$$

Para la salida, si la activación es lineal:

$$o = \alpha_{0o} + \sum_{j \to o} w_{jo}\, h_j$$

**Flujo general:** entrada → combinación ponderada + sesgo en cada nodo oculto → activación no lineal → combinación ponderada de las salidas ocultas → salida final. Es decir: $X \to \text{capa oculta} \to \text{predicción}$.

**[A]** Tsay también menciona las redes "skip-layer", que además permiten una conexión directa de las entradas a la salida (sin pasar por la capa oculta), y señala que si la activación de salida es lineal, esas conexiones directas representan, dentro de la red, exactamente un modelo lineal — es decir, la red generaliza al modelo lineal como caso particular.

### 7.2 Por qué la no linealidad depende de la función de activación

**Por qué una red sin activaciones no lineales colapsaría en una transformación lineal.** Si $f_j(\cdot)$ fuera una función lineal (por ejemplo, $f_j(z) = z$), entonces cada nodo oculto sería una combinación lineal de las entradas, y la salida —una combinación lineal de esos nodos— sería, en definitiva, **una combinación lineal de las entradas originales**, sin importar cuántas capas o nodos se agreguen. Componer transformaciones lineales entre sí siempre da como resultado otra transformación lineal. **[B]** Esta es una consecuencia matemática directa de la estructura descrita por Tsay, no una afirmación textual explícita del capítulo, pero se sigue necesariamente de las ecuaciones (4.30)-(4.34) que él mismo presenta.

**Por qué las activaciones permiten representar relaciones más complejas.** Al introducir una función no lineal (como la logística) entre las capas, la salida deja de ser una combinación lineal de las entradas: cada nodo oculto puede "activarse" o "no activarse" de forma diferente según la región del espacio de entradas, lo cual permite que la red represente umbrales, saturaciones e interacciones —exactamente las cuatro formas de no linealidad descritas en la Sección 2 de este informe—, todas combinadas de forma flexible.

### 7.3 Universal approximation — con mucho cuidado

**[A]** Tsay afirma que las redes feed-forward pueden aproximar de manera uniforme, sobre conjuntos compactos, cualquier función continua, aumentando el número de nodos en la capa oculta (citando a Hornik, Stinchcombe y White, 1989; Hornik, 1993; Chen y Chen, 1995). Esta es la llamada **propiedad de aproximación universal** de los perceptrones multicapa.

En palabras simples:

> Una red con suficiente capacidad (suficientes nodos ocultos) puede, en principio, representar o aproximar una familia muy amplia de funciones — no está limitada, como un modelo lineal, a una única forma de relación.

**Esto es una afirmación sobre *capacidad de representación matemática*, no sobre lo que ocurre al entrenar una red con datos reales y finitos.** El propio enunciado del estudio exige remarcar, y este informe lo hace explícito como guardrail central:

> **Capacidad de representar ≠ capacidad de aprender.**
>
> **Capacidad de aprender in-sample ≠ capacidad de generalizar fuera de muestra.**
>
> **Generalizar estadísticamente ≠ ser económicamente útil.**

**[A]** Esto no es solo una precaución teórica abstracta: Tsay lo demuestra empíricamente en su propio capítulo. En el Ejemplo 4.8 (ver Sección 7.5 más abajo), una red 8–4–1 entrenada para predecir la dirección de los retornos mensuales de IBM **no logra superar, en la muestra de evaluación fuera de muestra, a un modelo trivial de camino aleatorio con deriva**, a pesar de tener capacidad de representación muy superior a la de ese modelo trivial (49 parámetros frente a 1).

### 7.4 Entrenamiento y selección

**[A]** Tsay describe el entrenamiento de una red como un proceso de dos etapas: (1) construir la red — decidir el número de nodos, y estimar (entrenar) sus pesos y sesgos — y (2) usarla para inferencia, especialmente pronóstico. Los datos se dividen en dos subconjuntos no superpuestos: uno para **estimar** los parámetros de una red dada (*estimation sample*), y otro para evaluar su desempeño de pronóstico y compararlo entre configuraciones distintas (lo que Tsay llama, explícitamente, "la idea de cross-validation ampliamente usada en la selección de modelos estadísticos").

**[A] Función objetivo.** El criterio de ajuste típico que menciona Tsay es la suma de cuadrados de los errores, $S^2 = \sum_t (r_t - o_t)^2$, minimizada mediante métodos iterativos; menciona específicamente el algoritmo de retropropagación (*back propagation*, Bryson y Ho, 1969) como el método popular en la literatura de redes neuronales para este propósito.

**[A] Selección del número de nodos ocultos.** Tsay no da una regla cerrada; el criterio implícito en sus ejemplos es comparar el desempeño de pronóstico fuera de muestra entre configuraciones con distinto número de nodos, seleccionando la que mejor generaliza — es decir, el mismo principio de validación cruzada mencionado arriba.

**[A] [B] Separación estricta requerida por el enunciado.**

- **[A] Lo que Tsay hace en su contexto:** divide la serie en un tramo de estimación y un tramo de evaluación, en orden temporal (para IBM: 1926–1997 para entrenar, 1998–1999 para evaluar). Esa separación ya respeta el orden temporal en los ejemplos concretos del capítulo.
- **[B] Lo que sería necesario adaptar para series financieras temporales, y que Tsay no desarrolla en detalle en este capítulo:** el capítulo no discute explícitamente validación cruzada *aleatoria* (mezclando observaciones fuera de orden) para series temporales, ni tampoco advierte en esta sección sobre fuga de información entre pliegues. **No debe asumirse que un esquema de validación cruzada aleatoria estándar —de los que mezclan observaciones sin respetar el orden temporal— sea apropiado para datos financieros temporales.** Esto es una precaución añadida por este informe, coherente con el hecho de que los propios ejemplos de Tsay usan siempre una partición temporal ordenada, pero no es una afirmación textual de Tsay sobre el tema.

No se diseña aquí ningún protocolo definitivo de validación para el Proyecto IRIS; esa decisión queda fuera del alcance de este estudio.

### 7.5 Los dos ejemplos empíricos de Tsay sobre redes neuronales

**[A] Ejemplo 4.7 — retornos mensuales logarítmicos de IBM, en porcentaje, con dividendos, enero 1926 a diciembre 1999.** Estimación: enero 1926 a diciembre 1997 (864 observaciones). Evaluación fuera de muestra: 1998–1999. Red 3–2–1 (tres entradas: $r_{t-1}, r_{t-2}, r_{t-3}$; dos nodos ocultos). El error estándar residual de la red dentro de la muestra de estimación (6.56) es apenas menor que el de un AR(1) simple (6.61). En la evaluación fuera de muestra, comparando el error cuadrático medio de pronóstico (MSFE) a un paso: el "benchmark" de camino aleatorio con deriva (media muestral) obtiene 91.85; el AR(1) obtiene 91.70; la red 3–2–1 obtiene 91.74 — **prácticamente indistinguible del AR(1) simple, y sin ventaja clara sobre el benchmark trivial**. Tsay agrega, además, una observación relevante sobre inestabilidad: repitiendo la estimación de la misma red 3–2–1 con distintos puntos de partida, el MSFE fuera de muestra varió entre 89.46 y 93.65 — es decir, **el resultado depende sensiblemente de detalles de la optimización**, no solo de los datos y la arquitectura.

**[A] Ejemplo 4.8 — misma serie de IBM, pero pronosticando la *dirección* del retorno** (variable binaria: sube o no sube). Red 8–4–1 (ocho entradas: cuatro rezagos de $r_t$ y cuatro de la variable de dirección $d_t$; cuatro nodos ocultos; 49 parámetros en total). Con un umbral de decisión de 0.5, la red logra una tasa de acierto de 0.58 en una corrida puntual. Pero Tsay va más allá y hace una **simulación de 500 corridas** de la misma red, comparándola contra 500 simulaciones de un camino aleatorio con deriva. **Resultado explícito de Tsay: "la red neuronal feed-forward 8–4–1 no supera al modelo simple que asume un camino aleatorio con deriva para el precio logarítmico mensual de IBM."** La tabla de contingencia 2×2 de esa corrida puntual arroja un estadístico chi-cuadrado de 0.137 (p = 0.71): **no hay evidencia de que la red supere significativamente al azar** en esa evaluación direccional.

**Por qué este resultado es importante para el informe.** Es el ejemplo más directo y explícito que da Tsay de la brecha entre "capacidad de representar relaciones no lineales" y "utilidad predictiva real fuera de muestra". No se trata de una advertencia abstracta: es un resultado empírico documentado por el propio autor, sobre una serie real (retornos mensuales de IBM), que debe registrarse tal cual es — sin generalizarlo automáticamente a "las redes neuronales no sirven para mercados financieros" (eso sería sobregeneralizar un resultado puntual, sobre una serie, frecuencia y arquitectura específicas, a una afirmación universal).

### 7.6 Modelos clásicos vs. redes neuronales — tabla comparativa

**[B]** Esta tabla organiza, con fines didácticos, la progresión de modelos vista en el capítulo según quién determina la forma funcional, cuánta flexibilidad ofrece, y cuán interpretable es el resultado:

| Modelo | ¿Quién especifica la forma? | Flexibilidad | Interpretabilidad |
|---|---|---|---|
| AR | El analista | Baja | Alta |
| TAR | El analista fija la forma; el umbral puede estimarse de los datos | Media | Alta |
| STAR | El analista | Media / Alta | Media |
| Functional coefficient | Parcialmente los datos | Alta | Media |
| Nonparametric (kernel, local linear) | Los datos | Alta | Menor |
| Neural network | Los datos, dentro de una arquitectura fijada por el analista | Muy alta | Menor |

**Advertencia explícita, coherente con los resultados de Tsay reportados arriba:** mayor flexibilidad **no implica automáticamente** mejor pronóstico. El Ejemplo 4.8 es la evidencia empírica directa de esto dentro del propio capítulo: la red, con 49 parámetros, no superó a un modelo de un solo parámetro (la deriva media).

---

## 8. Capacidad del modelo vs. información disponible

**[B] Sección especial, tal como lo requiere el enunciado.**

Una red neuronal puede, en principio (por la propiedad de aproximación universal discutida arriba), ser capaz de representar una función extremadamente compleja. Pero esa capacidad de representación no crea información donde no la hay. Si el conjunto de variables de entrada $X$ no contiene información suficiente sobre la variable objetivo $Y$ —si, en el mercado real, el pasado reciente simplemente no determina de forma sistemática el retorno futuro—, **ningún modelo, por más flexible que sea, puede inventar esa relación**. Solo puede, en el peor caso, ajustarse al ruido de la muestra de entrenamiento, produciendo una ilusión de ajuste que desaparece fuera de muestra.

> **Modelo más flexible ⇏ más señal.**
>
> **Bajo error de entrenamiento ⇏ predictibilidad real.**

**[A]** Esta idea está respaldada indirectamente por el propio Ejemplo 4.7 de Tsay: el error residual *dentro* de la muestra de estimación de la red (6.56) es apenas mejor que el de un AR(1) (6.61) — una diferencia pequeña —, y esa pequeña ventaja **desaparece** en la evaluación fuera de muestra (91.74 contra 91.70, prácticamente empatados). La mayor flexibilidad de la red no se tradujo en una ventaja predictiva genuina, ni siquiera pequeña, una vez evaluada honestamente fuera de la muestra de ajuste.

**[B]** Este es, junto con la distinción entre las cuatro nociones de la Sección 1 (no linealidad, dependencia, predictibilidad, utilidad económica), uno de los dos mensajes centrales que este capítulo aporta al Proyecto IRIS: antes de preguntarse *qué arquitectura de modelo usar*, hay que preguntarse *si existe, en los datos disponibles, información suficiente sobre el objetivo que se quiere predecir*. Ningún modelo resuelve la ausencia de señal.

---

## 9. Tests de no linealidad (4.2)

### 9.1 La pregunta que abre la sección

**[A]** Tsay organiza esta sección alrededor de una pregunta operativa: ¿cómo sabemos, con evidencia estadística y no solo con intuición, que necesitamos considerar un modelo no lineal? El esquema general de todos los tests que siguen es:

$$H_0: \text{la estructura lineal es suficiente} \quad \text{vs.} \quad H_1: \text{queda estructura no lineal sin explicar}$$

**[A]** Tsay agrupa los tests en dos familias: **no paramétricos** (Q de residuos al cuadrado, test bispectral, BDS), que no asumen una forma concreta de no linealidad bajo la alternativa, y **paramétricos** (RESET y sus variantes, tests F de Tsay, tests de umbral), que sí especifican una alternativa concreta contra la cual comparar. Tsay señala explícitamente que **no existe un test único que domine a los demás**, porque la no linealidad puede manifestarse de muchas formas distintas, y cada test tiene poder distinto contra cada tipo de alternativa.

### 9.2 Cada test, explicado con la misma estructura de seis preguntas

**Q-statistic de residuos al cuadrado (McLeod y Li, 1983).**
1. *¿Qué pregunta hace?* Después de ajustar un ARMA y quedarnos con los residuos, ¿esos residuos al cuadrado siguen teniendo autocorrelación?
2. *¿Qué tipo de no linealidad detecta?* Principalmente heterocedasticidad condicional (estructura tipo ARCH) — es decir, no linealidad en varianza más que en media.
3. *¿Qué supone?* Que el modelo lineal (ARMA) ya está bien especificado en la media.
4. *¿Qué significa rechazar?* Que queda estructura de segundo orden (varianza) no explicada por el modelo lineal.
5. *¿Qué NO significa rechazar?* No identifica qué tipo específico de modelo de volatilidad hace falta, ni dice nada directo sobre la media condicional.
6. *Limitaciones.* Está diseñado para un tipo particular de no linealidad (cuadrática); puede no tener poder frente a otras formas.

**Test bispectral.**
1. *¿Qué pregunta hace?* ¿El bispectro (la "huella" de tercer orden, vía momentos de tercer orden, de la serie) es constante en todas las frecuencias, como debería ser bajo linealidad?
2. *¿Qué tipo de no linealidad detecta?* Desviaciones de linealidad y de gaussianidad relacionadas con la asimetría de la distribución (momentos de tercer orden).
3. *¿Qué supone?* Estacionariedad y la existencia de momentos hasta tercer orden.
4. *¿Qué significa rechazar?* Que el bispectro no es constante — evidencia contra linealidad/gaussianidad conjunta.
5. *¿Qué NO significa rechazar?* No indica la forma concreta de la no linealidad.
6. *Limitaciones.* **[A]** Tsay señala que la experiencia limitada muestra que el test tiene poder decente solo con muestras grandes.

**BDS (Brock, Dechert y Scheinkman).** Desarrollado con detalle en la Sección 10 de este informe, dado su peso especial en el enunciado.

**RESET (Ramsey, 1969) y sus variantes (Keenan 1985, F-test de Tsay 1986).**
1. *¿Qué pregunta hace?* Si agrego, a un modelo AR(p) ya ajustado, términos adicionales construidos a partir del propio pronóstico ajustado (potencias del ajuste, o de los regresores), ¿esos términos ayudan a explicar el residuo?
2. *¿Qué tipo de no linealidad detecta?* Formas polinómicas / cuadráticas de desvío respecto de la linealidad en la media.
3. *¿Qué supone?* Que se puede especificar razonablemente qué términos adicionales probar; normalidad para la distribución exacta del estadístico F.
4. *¿Qué significa rechazar?* Que esos términos adicionales tienen poder explicativo sobre el residuo — evidencia de no linealidad en la media.
5. *¿Qué NO significa rechazar?* No dice qué modelo no lineal específico usar; el test F de Tsay (1986) usa productos cruzados de regresores ($M_{t-1} = \mathrm{vech}(X_{t-1}X_{t-1}')$) como una manera concreta, pero no única, de operacionalizar "términos adicionales".
6. *Limitaciones.* **[A]** Tsay señala el problema de multicolinealidad entre los términos ajustados originales y los nuevos, y cómo Keenan (1985) y el propio Tsay (1986) modifican el procedimiento para mitigarlo.

**Tests de umbral (likelihood ratio para SETAR, y el TAR-F de Tsay 1989).**
1. *¿Qué pregunta hace?* ¿Existe evidencia de que el proceso cambia de régimen según una variable de umbral concreta?
2. *¿Qué tipo de no linealidad detecta?* Específicamente estructura tipo TAR/SETAR.
3. *¿Qué supone?* El test de razón de verosimilitud enfrenta una dificultad técnica real: **el umbral no está definido bajo la hipótesis nula** (si no hay régimen, no hay "punto de corte" que estimar), lo que Tsay llama un "parámetro molesto" (nuisance parameter) y que invalida la distribución asintótica estándar del estadístico. El TAR-F test de Tsay (1989) evita este problema reformulando el test como una regresión ordenada (arranged autoregression) y verificando si los residuos predictivos recursivos están correlacionados con los regresores.
4. *¿Qué significa rechazar?* Evidencia de estructura tipo umbral.
5. *¿Qué NO significa rechazar?* No identifica el número de regímenes ni el umbral óptimo por sí solo.
6. *Limitaciones.* **[A]** Tsay indica explícitamente que el TAR-F test, aunque evita el problema del parámetro molesto, **es menos potente** que el test de razón de verosimilitud cuando el modelo verdadero es efectivamente un SETAR de 2 regímenes con distribución conocida — es decir, hay una compensación entre robustez del procedimiento y poder estadístico.

### 9.3 Applications — 4.2.3, con precisión de serie/frecuencia/período

**[A]** Tsay aplica varios de estos tests a cinco series (Tabla 4.2):

1. Una serie simulada iid N(0,1), 500 observaciones.
2. Una serie simulada iid t-Student con 6 grados de libertad, 500 observaciones.
3. Residuos de un AR ajustado a los retornos mensuales logarítmicos del índice CRSP equal-weighted, 1926–1997, 864 observaciones.
4. Residuos de un AR ajustado a los retornos mensuales logarítmicos del índice CRSP value-weighted, 1926–1997, 864 observaciones.
5. Residuos de un AR(1) ajustado a los retornos mensuales logarítmicos de la acción de IBM, 1926–1997, 864 observaciones.

**[A] Resultado exacto reportado por Tsay:** para las dos series simuladas (genuinamente lineales por construcción), tanto el BDS como los tests F resultan **no significativos** al 5%, como debería ser. Para los retornos de los índices (equal-weighted y value-weighted), el BDS **y** los tests F resultan **significativos**. Para los retornos de IBM, el BDS es significativo, pero **los tests F no logran detectar no linealidad**. La conclusión textual de Tsay: los tests confirman que las series simuladas son lineales, y sugieren que los retornos accionarios (en estos tres casos) son no lineales — aunque distintos tests capturan distintos aspectos, y no siempre coinciden entre sí (como el caso de IBM ilustra).

**[B] Qué podemos aprender metodológicamente** (no qué conclusión trasladar): (1) distintos tests pueden dar resultados distintos sobre la misma serie, lo cual confirma que no hay un test universal; (2) conviene aplicar varios tests, no uno solo, y ser explícito sobre en cuáles hay coincidencia y en cuáles no; (3) el hecho de que el BDS sea significativo en retornos accionarios mensuales de 1926-1997 (índices y una acción particular) **no se traslada automáticamente** a retornos intradiarios de futuros de índices: la frecuencia, el instrumento, el período y el contexto de mercado son completamente distintos, y Tsay no analiza NQ, MNQ, ni ningún futuro en este capítulo.

---

## 10. El test BDS, en detalle

**Pregunta que responde, antes de cualquier matemática.** Después de quitarle a una serie toda la estructura lineal que pudiéramos explicar (ajustando un buen modelo AR/ARMA y quedándonos con los residuos), **¿esos residuos siguen comportándose como observaciones verdaderamente independientes entre sí, o queda algún patrón —de cualquier tipo— que un modelo lineal no pudo explicar?**

**[A]** El BDS (Brock, Dechert y Scheinkman, 1987) es distinto de los demás tests de esta sección porque no se enfoca en un tipo particular de estructura (como la cuadrática de la Q de residuos al cuadrado, o la de tercer orden del bispectral): busca **cualquier** desviación de la hipótesis de independencia e idéntica distribución (iid), usando una herramienta tomada de la teoría de series caóticas: la **integral de correlación**.

**Intuición antes de la matemática.** Se toman "ventanas" de $k$ observaciones consecutivas (llamadas $k$-historias) y se las trata como puntos en un espacio de $k$ dimensiones. Si la serie original fuera realmente iid, esas $k$-historias no deberían mostrar ningún patrón de agrupamiento en ese espacio: la probabilidad de que dos $k$-historias estén "cerca" entre sí debería ser, aproximadamente, la probabilidad de que dos observaciones individuales estén cerca, elevada a la potencia $k$. Cualquier desviación sistemática de esa relación es evidencia de que las observaciones no son realmente independientes.

**[A] Matemática indispensable.** La integral de correlación:

$$C_k(\delta) = \lim_{T_k \to \infty} \frac{2}{T_k(T_k-1)} \sum_{i<j} I_\delta(X_i, X_j)$$

donde $I_\delta(u,v)=1$ si $\|u-v\| < \delta$ (es decir, si los dos puntos están a una distancia menor que $\delta$) y 0 en caso contrario. Bajo la hipótesis nula de que $\{x_t\}$ es iid, se cumple $C_k(\delta) = [C_1(\delta)]^k$ para cualquier $k$ y $\delta$ fijos. El estadístico BDS mide, de forma estandarizada, qué tan lejos está la relación empírica de esa igualdad esperada:

$$D_k(\delta, T) = \frac{\sqrt{T}\,\{C_k(\delta,T) - [C_1(\delta,T)]^k\}}{\sigma_k(\delta,T)}$$

que, bajo la hipótesis nula, se distribuye aproximadamente como una normal estándar.

**Qué NO significa un rechazo del BDS.** **[A]** El propio Tsay advierte dos cosas explícitamente: primero, que en la práctica **conviene remover cualquier dependencia lineal antes de aplicar el BDS** (aplicarlo directamente sobre datos con estructura lineal no removida confunde ambas cosas); segundo, que el test **puede ser sensible a las elecciones de $\delta$ y $k$**, especialmente cuando $k$ es grande. Más allá de lo que dice Tsay explícitamente, se sigue directamente de la definición del test: **[B]** dado que el BDS es sensible a *cualquier* forma de dependencia (no solo la que un analista tiene en mente al construir un modelo no lineal específico), un rechazo del BDS es evidencia de que "algo" queda sin explicar, pero **no dice qué es ese algo, ni qué modelo debería usarse para capturarlo**. Puede deberse a no linealidad genuina en la media, a heterocedasticidad residual no completamente capturada, o a otras formas de dependencia.

---

## 11. Guardrail principal de los tests de no linealidad

Este bloque se marca de forma explícita porque el enunciado lo exige como guardrail central del capítulo.

Un resultado del tipo $p < 0.05$ en cualquiera de los tests anteriores permite afirmar, como máximo, algo del tipo: **"hay evidencia contra la hipótesis lineal considerada, bajo las condiciones específicas de este test, en esta muestra"**. 

**No permite concluir, automáticamente, ninguna de las siguientes cosas:**

- que existe una señal fuerte;
- que se conoce la forma de la no linealidad;
- que una red neuronal (u otro modelo no lineal específico) será mejor;
- que la relación detectada será estable en el tiempo;
- que funcionará fuera de la muestra en la que se testeó;
- que será económicamente explotable, una vez descontados costos operativos.

> **Test detecta estructura ≠ Modelo sabe explotarla.**

**[A]** Esta lectura es coherente con el propio capítulo: Tsay encuentra evidencia significativa de no linealidad (vía BDS) en retornos mensuales de índices accionarios y de IBM (Sección 9.3 de este informe), pero eso no le impide, más adelante, mostrar en el Ejemplo 4.8 que una red neuronal entrenada sobre la misma serie de IBM **no logra superar a un modelo trivial** en pronóstico direccional fuera de muestra. Detectar estructura no lineal y construir un modelo capaz de explotarla con ventaja económica son, empíricamente, dos logros distintos — y el segundo es mucho más difícil de alcanzar que el primero.

---

## 12. Múltiples tests y data snooping

**[A]** Tsay no desarrolla este problema de forma explícita ni sistemática dentro del Capítulo 4; se limita a presentar cada test individualmente.

**[B]** Sin embargo, es una consecuencia lógica directa de lo visto en la Sección 9.3: si se prueban muchos tests, muchos rezagos, muchos umbrales candidatos, muchas especificaciones de variables, eventualmente **alguno resultará "significativo" por puro azar**, incluso si no existiera ninguna estructura real. Cuantas más combinaciones se prueben sobre el mismo conjunto de datos, mayor es la probabilidad de encontrar al menos un resultado significativo por casualidad — este es el problema conocido como *data snooping* o *multiple testing*.

**No debe interpretarse un conjunto de p-valores bajos como evidencia robusta de no linealidad si hubo una búsqueda extensa (muchos tests, muchos umbrales, muchas variables) sin ningún ajuste por comparaciones múltiples.** Este informe deja explícitamente el tema de data snooping / multiple testing como una cuestión que **requiere literatura adicional** más allá de este capítulo de Tsay — no se resuelve aquí, y no se ofrece ningún procedimiento correctivo definitivo.

---

## 13. Modeling — de detectar no linealidad a elegir un modelo (4.3)

**Pregunta que responde esta sección.** Detectamos evidencia de no linealidad. ¿Y ahora qué?

**[A]** Tsay es explícito en que el modelado no lineal "necesariamente involucra juicio subjetivo" (no hay un procedimiento automático), aunque ofrece lineamientos generales: (1) primero construir un modelo lineal adecuado, sobre el cual se basan los tests de no linealidad; (2) usar los tests de la Sección 4.2 para verificar si queda estructura no explicada; (3) si la no linealidad es estadísticamente significativa, elegir una **familia** de modelos no lineales a considerar — decisión que, según el propio Tsay, "puede depender de la experiencia del analista y de la naturaleza sustantiva del problema"; (4) para modelos de volatilidad, usar la función de autocorrelación parcial de la serie al cuadrado para sugerir el orden; para GARCH/EGARCH, en la práctica solo se consideran órdenes bajos (1,1), (1,2), (2,1), porque órdenes más altos son difíciles de estimar e interpretar; (5) para TAR, usar los procedimientos de Tong (1990) y Tsay (1989, 1998); (6) cuando el tamaño de muestra lo permite, usar métodos no paramétricos para explorar la forma antes de comprometerse con un modelo paramétrico concreto; (7) usar criterios de información como AIC, y comparaciones tipo "odds ratio generalizado" (Chen, McCulloch y Tsay, 1997) para discriminar entre modelos no lineales que compiten.

**Por qué rechazar linealidad no identifica automáticamente qué modelo usar.** Rechazar $H_0$: "el modelo lineal es adecuado" **solo dice que algo no está capturado** — no dice si ese algo es un umbral, una transición suave, un estado latente, una interacción, o una forma completamente distinta que ninguno de los modelos de este capítulo contempla. Elegir entre TAR, STAR, Markov Switching, red neuronal o un modelo no paramétrico exige criterio adicional, no se deduce del resultado de un test de linealidad.

**[A]** Recordatorio explícito de Tsay, y reforzado aquí como guardrail: el modelo elegido debe ser cuidadosamente diagnosticado (verificar que sus residuos ya no muestren la estructura detectada) antes de usarse para predicción — no basta con que "ajuste mejor" en la muestra de estimación.

**Guardrail final de esta sección**, coherente con lo señalado desde el Capítulo 3: **un mejor AIC/BIC (o verosimilitud) dentro de la muestra no equivale a un mejor pronóstico fuera de muestra.** Tsay no lo dice en estas palabras exactas en esta sección puntual, pero es exactamente lo que sus propios ejemplos (Sección 7.5, Ejemplo 4.7) demuestran: la red tuvo mejor ajuste dentro de muestra que el AR(1), y esa ventaja desapareció fuera de muestra.

---

## 14. Forecasting (4.4)

### 14.1 Por qué pronosticar con modelos no lineales es más difícil

**[A]** A diferencia de los modelos lineales (ARMA), para la mayoría de los modelos no lineales **no existen fórmulas cerradas** para calcular pronósticos a más de un paso. La razón matemática de fondo, que Tsay no explicita con esta fórmula exacta pero que es la justificación estándar del método de bootstrap paramétrico que sí describe, es que, para una función no lineal $f(\cdot)$ en general:

$$E[f(X)] \neq f(E[X])$$

**Ejemplo simple, tal como lo pide el enunciado.** Si $f(x) = x^2$, entonces $E[X^2]$ **no** es lo mismo que $(E[X])^2$ — de hecho, $E[X^2] - (E[X])^2 = \mathrm{Var}(X) \ge 0$, es decir, siempre son distintos salvo que $X$ no tenga varianza. Esto explica por qué **simplemente "enchufar" el pronóstico puntual de la media dentro de una función no lineal puede dar un resultado incorrecto**: el pronóstico correcto de $f(X)$ no es, en general, $f$ evaluado en el pronóstico de $X$.

**1-step ahead vs. multi-step ahead.** A un paso, muchos modelos no lineales (como TAR, si se conoce la variable de umbral) sí tienen una fórmula simple, porque toda la información relevante ya es conocida. A varios pasos, hay que promediar sobre la incertidumbre acumulada de los pasos intermedios, y ahí es donde aparece el problema de $E[f(X)] \neq f(E[X])$.

### 14.2 Bootstrap paramétrico (4.4.1)

**Idea, en una frase.** Generar muchos futuros posibles, simulando repetidamente el modelo hacia adelante, y observar la distribución resultante de esos futuros simulados.

**[A] Secuencia exacta que describe Tsay:**

1. Estamos parados en el momento $T$ (el origen del pronóstico).
2. El modelo ya fue estimado y se considera adecuado (diagnosticado correctamente).
3. Se genera un nuevo shock aleatorio, extraído de la distribución de innovaciones especificada por el modelo.
4. Usando el modelo, los datos, y los pronósticos previos ya generados, se calcula el siguiente valor simulado de la serie.
5. Se repite este proceso hasta llegar al horizonte de interés, obteniendo una trayectoria simulada completa.
6. Se repite todo el procedimiento $M$ veces (Tsay usa $M=3000$ en algunas aplicaciones), obteniendo $M$ trayectorias distintas.

**Qué se obtiene de esas $M$ simulaciones.** **[A]** El pronóstico puntual se define como el promedio de las $M$ realizaciones simuladas en el horizonte de interés. Pero, más allá del promedio, se obtiene una **distribución empírica completa** de los valores futuros posibles, de la cual pueden extraerse la mediana, percentiles específicos (para construir intervalos de confianza o predicción), la probabilidad empírica de que el resultado tenga signo positivo, o cualquier otra cantidad de interés que se derive de esa distribución simulada.

**[B] Conexión con probabilistic forecasting.** Esta idea —no dar un único número como pronóstico, sino una distribución completa de resultados posibles— es exactamente el principio detrás de lo que en la práctica moderna de forecasting (y en trading cuantitativo) se llama pronóstico probabilístico. No se ejecuta ningún bootstrap en este informe; se documenta solo el procedimiento conceptual que describe Tsay.

### 14.3 Forecasting Evaluation (4.4.2) — prioridad máxima

**[A]** Tsay distingue explícitamente tres tipos de objetivo de pronóstico, cada uno con su propia pregunta y su propia forma de evaluarse:

**Pronóstico direccional — $P(r_{t+h} > 0)$.** Pregunta: ¿sube o baja? **[A]** Se evalúa con una tabla de contingencia 2×2 de aciertos y errores (subida/bajada, predicha/observada), y un estadístico chi-cuadrado que compara el desempeño del modelo contra lo esperable por azar puro. Tsay ilustra esto exactamente con la red del Ejemplo 4.8: la tabla de contingencia arroja $\chi^2 = 0.137$ ($p=0.71$) — **la red no supera significativamente al azar** en esa evaluación.

**Pronóstico puntual / de magnitud — $\hat r_{t+h}$.** Pregunta: ¿cuál será, aproximadamente, el valor? **[A]** Se evalúa con MSE (error cuadrático medio), MAD (desviación absoluta media) o MAPE (error porcentual absoluto medio) sobre la muestra de evaluación. Tsay advierte que distintas métricas pueden llevar a elegir modelos distintos, y que estas medidas tienen limitaciones adicionales (cita a Clements y Hendry, 1993) que no desarrolla en detalle en este capítulo.

**Pronóstico distribucional — $F(r_{t+h} \mid X_t)$.** Pregunta: ¿qué distribución completa de resultados futuros se espera, no solo un punto o una dirección? **[A]** Se evalúa comparando los percentiles empíricos observados (obtenidos de la distribución predictiva simulada por bootstrap) contra una distribución uniforme teórica, usando por ejemplo el estadístico de Kolmogorov–Smirnov: si el modelo es adecuado, esos percentiles deberían distribuirse uniformemente en [0,1].

**[A]** Tsay agrega una observación metodológica importante: **no existe una única medida de comparación ampliamente aceptada** entre modelos; puede hacer falta una función de utilidad basada en el objetivo específico del pronóstico para entender realmente qué comparación importa en cada caso concreto.

---

## 15. Estimation sample vs. forecasting sample

**[A]** Tal como describe Tsay: el conjunto de datos disponible se divide en dos subconjuntos. Una parte (*estimation sample*) sirve para construir y ajustar el modelo. La otra parte (*forecasting sample*) sirve, exclusivamente, para comprobar si el modelo efectivamente generaliza a datos que no participaron de su ajuste. Tsay también menciona el uso de un **procedimiento rolling** (renovando el origen del pronóstico y volviendo a estimar a medida que avanza el tiempo), tal como se aplica en el Ejemplo de la Sección 4.5 (ver más abajo).

En palabras simples:

> Una parte de los datos sirve para aprender.
> Otra parte, que el modelo nunca vio durante el aprendizaje, sirve para comprobar si realmente aprendimos algo que funciona en datos nuevos.

**[B] Precaución añadida para series financieras temporales, no una afirmación de Tsay en este capítulo:** esta separación debe respetar el orden temporal (entrenar con el pasado, evaluar con el futuro relativo a ese entrenamiento) para que la comprobación sea honesta; de lo contrario se corre el riesgo de que información del "futuro" se filtre, directa o indirectamente, hacia el proceso de ajuste.

**Explícitamente fuera de alcance de este informe:** no se define aquí ningún protocolo definitivo de walk-forward, purging, embargo, ni ningún hiperparámetro definitivo para el Proyecto IRIS. El procedimiento de Tsay (dividir en estimación/pronóstico, eventualmente con rolling) se documenta como lo que es —un procedimiento ilustrativo dentro de un libro de texto— y no se convierte automáticamente en el protocolo final de validación de IRIS.

---

## 16. Application — el caso completo de Tsay (4.5)

**[A] Serie e instrumento:** tasa de desempleo civil trimestral de EE.UU., ajustada estacionalmente, 1948–1993 (analizada previamente en detalle por Montgomery et al., 1998; Tsay reproduce parte de ese análisis con modelos no lineales).

**Procedimiento completo, siguiendo exactamente el esquema pedido:**

1. **Modelo lineal.** Sobre la serie diferenciada $\Delta x_t$, se ajusta un modelo estacional ARMA (Ecuación 4.52) como benchmark.
2. **Diagnóstico.** Se observa que el ajuste estacional no eliminó del todo la estacionalidad aparente, y se registra la asimetría conocida de la serie: el desempleo sube rápido y baja lento.
3. **Tests de no linealidad.** Con un AR(5) sobre $\Delta x_t$, se aplican varios tests de la Sección 4.2 (Tabla 4.3: Ori-F, LST, TAR con distintos delays). **[A] Resultado exacto: todos los tests rechazan la hipótesis de linealidad**, y esto se mantiene para modelos AR de orden 2 a 10 — es decir, el rechazo de linealidad es robusto a la elección del orden autorregresivo del modelo base, no un artefacto de una elección particular.
4. **Selección de modelo.** Se opta por comparar un TAR de 2 regímenes y un Markov Switching de 2 estados, siguiendo procedimientos ya vistos en el capítulo.
5. **Estimación.** **[A]** El TAR resultante (Ecuación 4.53) usa como variable de umbral $x_{t-2}$ con corte en 0.1: en el régimen de cambios pequeños/negativos, la dinámica es casi un AR(1) simple; en el régimen de saltos grandes (≥0.1), aparece un AR(2) con raíces complejas (indicando comportamiento cíclico) y una constante positiva que sugiere tendencia ascendente durante contracciones económicas. El Markov Switching (Ecuación 4.54) da una historia cualitativamente similar: un estado con media condicional negativa (expansión, desempleo bajando levemente) y otro con media condicional positiva y estructura AR(2) con raíces complejas (contracción).
6. **Diagnóstico.** Ambos modelos no lineales se comparan y se interpretan económicamente: el segundo régimen/estado de ambos modelos coincide en estructura (AR(2) con raíces complejas, indicando ciclos cortos), lo cual Tsay señala como una coincidencia interesante entre dos formas de modelar la no linealidad de la misma serie.
7. **Forecast.** **[A]** Se usa un procedimiento rolling, comenzando en el origen 1968.II, reestimando y pronosticando de 1 a 5 pasos adelante en cada punto, usando bootstrap paramétrico (Sección 4.4.1) para los modelos no lineales.
8. **Evaluación.** **[A] Resultado exacto (Tabla 4.4), MSE relativo al modelo lineal:** en comparación general, el TAR queda muy cerca del modelo lineal en MSE pero con sesgos menores; el MSA tiene el MSE más alto de los tres (peor) pero los sesgos más pequeños. **Desagregando por el estado de la economía en el origen del pronóstico:** en orígenes durante **contracciones económicas**, el TAR mejora claramente al lineal en MSE y en sesgo (por ejemplo, MSE relativo de 0.72 a 5 pasos, es decir, 28% menor que el lineal); en orígenes durante **expansiones**, en cambio, **el modelo lineal supera a ambos modelos no lineales**.

**[A] Conclusión textual exacta de Tsay:** las contribuciones de los modelos no lineales sobre el lineal, en el pronóstico de la tasa de desempleo trimestral de EE.UU., se concentran principalmente en los períodos de contracción económica — precisamente los períodos en que la intervención de política económica es más probable, y en los que las personas prestan más atención a los pronósticos.

**[B] Qué podemos aprender metodológicamente de este caso** (no su resultado económico): el proceso completo —modelo lineal → diagnóstico → tests → selección → estimación → diagnóstico → forecast → evaluación desagregada por contexto— es un patrón replicable. En particular, la práctica de **desagregar la evaluación por régimen o contexto** (aquí: expansión vs. contracción) en vez de reportar solo un número agregado es una idea metodológica valiosa y trasladable.

**No se generaliza el resultado económico de este caso a mercados de futuros.** Que un TAR haya mejorado el pronóstico del desempleo trimestral de EE.UU. durante contracciones no implica nada, por sí solo, sobre el comportamiento de MNQ o de cualquier otro futuro, en ninguna frecuencia.

---

## 17. Mapa Capítulos 1 → 2 → 3 → 4

**[B]** Organización propia, construida sobre el contexto mínimo provisto y el contenido de este capítulo:

- **Capítulo 1 — ¿qué propiedades tienen los retornos?** Colas pesadas, agrupamiento de volatilidad, baja relación señal/ruido en la media.
- **Capítulo 2 — ¿qué estructura lineal temporal existe?** Se distinguió dependencia lineal, dependencia estadística general, predictibilidad y utilidad económica; $\mathrm{ACF}(r_t) \approx 0$ no implica ausencia de estructura.
- **Capítulo 3 — ¿qué estructura existe en la volatilidad?** Se separó la media condicional de la varianza condicional; ARCH/GARCH mostraron que puede existir mucha estructura en la varianza aunque haya poca en la dirección.
- **Capítulo 4 — ¿qué estructura no lineal puede quedar en la media, o en la dinámica general?** Se amplió la pregunta del Capítulo 2: ya no solo "¿hay dependencia lineal?", sino "¿hay dependencia cuya forma no es una combinación lineal fija de coeficientes constantes?".

**Mapa deseado:**

$$\text{Distribución} \rightarrow \text{Dependencia lineal} \rightarrow \text{Volatilidad dinámica} \rightarrow \text{Dependencia no lineal}$$

**[B] Aclaración importante:** estos capítulos no son mutuamente excluyentes ni secuenciales en la práctica — el propio Ejemplo 4.3 de Tsay (AR–TAR–GARCH sobre IBM) combina explícitamente estructura no lineal (TAR) *dentro* de una ecuación de volatilidad (GARCH), mostrando que estas cuatro capas de estructura pueden coexistir en un mismo modelo, no compiten entre sí.

---

## 18. Implicancias para futuros — [A] separado de [B]

**[A] Lo que Tsay efectivamente muestra**, sin generalizar a futuros: (1) en varias series financieras reales que analiza (retornos accionarios mensuales de índices y de una acción individual, EE.UU., 1926–1997), tests de no linealidad como el BDS resultan significativos; (2) en al menos un caso muy documentado (retornos mensuales de IBM, 1926–1999), una red neuronal con capacidad de representación no lineal genuina no logró superar a un modelo trivial de camino aleatorio con deriva en pronóstico fuera de muestra, tanto en magnitud como en dirección; (3) en un caso macroeconómico (desempleo trimestral de EE.UU.), modelos no lineales (TAR, Markov Switching) sí mejoraron el pronóstico del modelo lineal, pero **solo en un subconjunto de contextos** (contracciones), no de forma uniforme.

**[B] Lecturas propias para futuros, todas como hipótesis a evaluar, no como conclusiones:**

- La posibilidad de que exista estructura no lineal en la media condicional de un futuro de índices (más allá de la ya conocida estructura no lineal en la varianza, vista en el Capítulo 3) es una pregunta empírica abierta, no una certeza que este capítulo demuestre para MNQ ni para ningún otro contrato.
- El patrón "los modelos no lineales ayudan más en algunos contextos que en otros" (visto en el ejemplo del desempleo) sugiere, como hipótesis, que si existiera valor en modelos no lineales para futuros, ese valor podría no ser uniforme en el tiempo — podría concentrarse en condiciones específicas de mercado (por ejemplo, alta volatilidad, cerca de eventos, etc.). Esto es una hipótesis, no un hallazgo trasladado.
- El resultado de la red neuronal de IBM es una advertencia concreta y documentada contra la expectativa de que "más flexibilidad = mejor pronóstico" se cumpla automáticamente, y debería pesar como precedente al evaluar, en el futuro, cualquier modelo flexible sobre datos de futuros.

---

## 19. Implicancias para Machine Learning — [B], por categorías

Todo lo que sigue es interpretación propia [B], no una propuesta a adoptar, presentada por categorías tal como pide el enunciado.

**Data representation.** El capítulo sugiere que, antes de fijar una representación de los datos, conviene preguntarse si la relación entre "estado reciente del mercado" y "retorno futuro" podría depender del contexto (volatilidad, dirección previa, magnitud del movimiento) de forma no lineal — una pregunta testeable, no asumible de antemano.

**Feature engineering.** La idea de términos bilineales (interacciones) y de coeficientes funcionales (FAR) sugiere, como categoría general a explorar más adelante, que el efecto de una variable sobre el retorno podría no ser constante sino condicional a otra variable — sin que esto implique construir, todavía, ninguna feature concreta.

**Target design.** La distinción entre pronóstico direccional, de magnitud y distribucional (Sección 14.3) es directamente relevante para decidir, en el futuro, qué tipo de variable objetivo tiene más sentido para IRIS — sin decidirlo aquí.

**Model architecture.** La progresión TAR → STAR → Markov Switching → no paramétrico → red neuronal es un espectro de compromisos entre interpretabilidad y flexibilidad (Sección 7.6), útil como marco conceptual para evaluar arquitecturas más adelante.

**Regime modeling.** La distinción de la Sección 5 (régimen manual, observable por umbral, latente) es una taxonomía útil para no confundir tres formas muy distintas de segmentar el comportamiento del mercado.

**Interaction effects.** El modelo bilineal y FAR sugieren, conceptualmente, que la relación entre dos variables explicativas y el objetivo podría no ser aditiva — una hipótesis a comprobar, no una feature a construir todavía.

**Neural networks.** El capítulo aporta tanto el vocabulario básico (input, peso, sesgo, activación, capa oculta) como una advertencia empírica concreta (Ejemplo 4.8) sobre la brecha entre capacidad de representación y utilidad predictiva real.

**Model diagnostics.** El patrón repetido de Tsay —ajustar, diagnosticar residuos, testear, y solo entonces considerar el modelo como candidato válido— es un flujo de trabajo replicable como principio general.

**Validation.** La separación estimation/forecasting sample, y la advertencia (añadida en este informe, no textual de Tsay) sobre no usar validación cruzada aleatoria sin orden temporal, son relevantes para cualquier validación futura de IRIS.

**Probabilistic forecasting.** El bootstrap paramétrico de Tsay (Sección 14.2) es conceptualmente el mismo principio detrás de generar una distribución de resultados posibles en vez de un único número — relevante para la pregunta "¿qué nivel de confianza tiene el modelo?" mencionada en la descripción del Proyecto IRIS.

**Model complexity.** La tabla de la Sección 7.6 y el resultado del Ejemplo 4.8 son un recordatorio de que mayor complejidad no es, por sí sola, una ventaja.

**Multivariate modeling.** Ninguna de las secciones estudiadas en este capítulo (que es predominantemente univariado) resuelve cómo extender estas ideas a un contexto multivariado (múltiples instrumentos o variables de contexto simultáneas); eso queda para capítulos posteriores del libro (ver Sección 21 de este informe).

**Hipótesis a comprobar.** Se desarrollan formalmente en la Sección 20 siguiente.

---

## 20. Hipótesis empíricas para backlog — NO ejecutar

Cada hipótesis se presenta con pregunta, método, resultado que la apoyaría, resultado que la refutaría, y limitaciones. Ninguna se ejecuta en este informe.

**H-NL1. Evidencia de no linealidad residual tras remover estructura AR y de volatilidad.**
- *Pregunta:* una vez ajustado un modelo lineal para la media y un modelo GARCH para la varianza, ¿los residuos estandarizados siguen mostrando estructura no explicada?
- *Método:* tests de la Sección 9 (Q de residuos al cuadrado, BDS, RESET/F) aplicados a los residuos estandarizados del mejor modelo lineal+GARCH disponible.
- *Resultado que la apoyaría:* rechazo consistente de linealidad en varios tests, con corrección por comparaciones múltiples.
- *Resultado que la refutaría:* no rechazo en la mayoría de los tests tras la corrección.
- *Limitaciones:* sensibilidad del BDS a $\delta$ y $k$; necesidad de remover primero toda la estructura conocida (Capítulos 2 y 3) para no confundir hallazgos.

**H-NL2. BDS sobre residuos, con especial atención a la frecuencia.**
- *Pregunta:* ¿el resultado de BDS depende sistemáticamente de la frecuencia de muestreo (diaria vs. intradiaria)?
- *Método:* aplicar BDS sobre residuos a distintas frecuencias del mismo instrumento.
- *Resultado que la apoyaría:* resultados sistemáticamente distintos por frecuencia.
- *Resultado que la refutaría:* resultados estables entre frecuencias.
- *Limitaciones:* alta frecuencia introduce complicaciones de microestructura no cubiertas por este capítulo (ver Capítulo 5, Sección 21).

**H-NL3. Estabilidad temporal de la no linealidad detectada.**
- *Pregunta:* si se detecta no linealidad en un período, ¿se mantiene en subperíodos posteriores?
- *Método:* repetir los tests de la Sección 9 en ventanas móviles o subperíodos no superpuestos.
- *Resultado que la apoyaría:* rechazo de linealidad consistente entre subperíodos.
- *Resultado que la refutaría:* resultados que aparecen y desaparecen sin patrón entre subperíodos (sugiriendo hallazgo espurio o inestable).
- *Limitaciones:* menor tamaño de muestra por subperíodo reduce el poder de los tests.

**H-NL4. TAR vs. modelo lineal.**
- *Pregunta:* ¿un TAR con una variable de umbral razonable (por ejemplo, un rezago propio o una medida de volatilidad reciente) mejora el ajuste y el pronóstico fuera de muestra frente a un AR comparable?
- *Método:* comparación en estimation/forecasting sample, siguiendo el esquema de la Sección 16, con MSE/MAD fuera de muestra.
- *Resultado que la apoyaría:* mejora consistente fuera de muestra, no solo dentro de muestra.
- *Resultado que la refutaría:* mejora solo dentro de muestra, que desaparece fuera de muestra (como el Ejemplo 4.7 de la red neuronal).
- *Limitaciones:* elección del umbral y del delay puede introducir data snooping si se prueban muchas combinaciones (ver Sección 12).

**H-NL5. STAR vs. TAR.**
- *Pregunta:* ¿una transición suave describe mejor los datos que una transición abrupta?
- *Método:* comparar ajuste y estabilidad de parámetros entre ambas especificaciones.
- *Resultado que la apoyaría:* parámetros de transición de STAR razonablemente bien estimados (errores estándar no excesivos) y mejora de ajuste/pronóstico.
- *Resultado que la refutaría:* parámetros de transición con errores estándar muy grandes (t≈1, como advierte Tsay) y sin mejora de pronóstico.
- *Limitaciones:* dificultad de estimación de STAR, documentada explícitamente por Tsay.

**H-NL6. Estabilidad del umbral estimado.**
- *Pregunta:* si se estima un umbral TAR en una ventana de datos, ¿ese mismo umbral sigue siendo razonable en otra ventana?
- *Método:* reestimar el umbral en subperíodos distintos y comparar.
- *Resultado que la apoyaría:* umbral estable entre ventanas.
- *Resultado que la refutaría:* umbral que varía sustancialmente entre ventanas.
- *Limitaciones:* el propio procedimiento de búsqueda del umbral óptimo (grid search sobre $r_1$) es susceptible de sobreajuste si no se valida fuera de muestra.

**H-NL7. Número de regímenes.**
- *Pregunta:* ¿dos regímenes son suficientes, o hace falta un número mayor?
- *Método:* comparación de especificaciones con distinto número de regímenes vía criterios de información y pronóstico fuera de muestra.
- *Resultado que la apoyaría:* mejora clara y estable al aumentar el número de regímenes.
- *Resultado que la refutaría:* mejora solo dentro de muestra, sin persistir fuera de muestra.
- *Limitaciones:* más regímenes implica menos datos por régimen y mayor riesgo de sobreajuste.

**H-NL8. Régimen observable vs. régimen latente.**
- *Pregunta:* ¿un régimen basado en una variable observable (TAR) captura tanto como un estado latente (Markov Switching), o hay información adicional en tratar el régimen como no observado?
- *Método:* comparar el desempeño de pronóstico entre ambas familias sobre los mismos datos.
- *Resultado que la apoyaría:* el modelo latente mejora sustancialmente sobre el observable.
- *Resultado que la refutaría:* ambos dan resultados similares, sugiriendo que la variable observable ya captura la mayor parte de la información relevante.
- *Limitaciones:* la estimación de modelos latentes es sustancialmente más compleja (Sección 4.1.4), lo cual eleva el costo de esta comparación.

**H-NL9. Markov Switching como descripción, no como "detector de verdad".**
- *Pregunta:* ¿los estados estimados por un Markov Switching se corresponden con alguna noción externa razonable de "régimen de mercado" (por ejemplo, alta/baja volatilidad medida independientemente)?
- *Método:* comparar la secuencia de probabilidades de estado estimadas contra una medida de contexto externa e independiente.
- *Resultado que la apoyaría:* correspondencia sistemática y estable.
- *Resultado que la refutaría:* correspondencia débil, inestable o dependiente de la especificación del modelo.
- *Limitaciones:* el nombre que se le da a un estado es siempre una interpretación posterior (Sección 4.4 de este informe), nunca una propiedad garantizada del modelo.

**H-NL10. Interacciones entre features.**
- *Pregunta:* ¿el efecto de una variable de contexto sobre el retorno futuro depende del valor de otra variable de contexto?
- *Método:* comparar un modelo aditivo simple contra uno que permite interacciones explícitas (o contra un modelo no paramétrico/flexible), evaluado fuera de muestra.
- *Resultado que la apoyaría:* mejora de pronóstico fuera de muestra al permitir interacciones.
- *Resultado que la refutaría:* sin mejora, o mejora solo dentro de muestra.
- *Limitaciones:* el espacio de interacciones posibles crece muy rápido; probar muchas aumenta el riesgo de data snooping.

**H-NL11. Functional coefficients.**
- *Pregunta:* ¿el coeficiente que conecta un rezago con el presente varía sistemáticamente según alguna variable de contexto observable?
- *Método:* estimación no paramétrica del coeficiente en función del contexto, comparada contra el coeficiente fijo de un AR simple, evaluada fuera de muestra.
- *Resultado que la apoyaría:* mejora de pronóstico y coeficiente estimado con forma interpretable y estable.
- *Resultado que la refutaría:* coeficiente estimado inestable o sin mejora fuera de muestra.
- *Limitaciones:* requiere muestras razonablemente grandes para estimar bien la función (Sección 6.1).

**H-NL12. Red neuronal vs. benchmark lineal.**
- *Pregunta:* ¿una red neuronal simple supera, fuera de muestra, a un AR o a una regresión lineal comparable?
- *Método:* comparación siguiendo exactamente el esquema del Ejemplo 4.7/4.8 de Tsay (estimation/forecasting sample ordenado en el tiempo, múltiples corridas para evaluar estabilidad).
- *Resultado que la apoyaría:* mejora consistente y estable en múltiples corridas.
- *Resultado que la refutaría:* resultado similar al benchmark, o mejora que varía mucho entre corridas (como Tsay documenta para IBM).
- *Limitaciones:* alta sensibilidad a inicialización y arquitectura, documentada explícitamente por Tsay.

**H-NL13. Red neuronal vs. TAR/STAR.**
- *Pregunta:* si existe estructura no lineal, ¿una red neuronal la captura mejor que un modelo TAR/STAR más simple e interpretable?
- *Método:* comparación directa de pronóstico fuera de muestra entre ambas familias sobre el mismo conjunto de datos.
- *Resultado que la apoyaría:* la red supera de forma consistente y estable.
- *Resultado que la refutaría:* desempeño similar o inferior, con mucha menor interpretabilidad como costo adicional.
- *Limitaciones:* ninguno de los ejemplos de Tsay en este capítulo compara directamente red neuronal contra TAR/STAR sobre la misma serie, por lo que no hay precedente textual directo que orientar esta comparación.

**H-NL14. Forecast direccional.**
- *Pregunta:* ¿algún modelo no lineal mejora significativamente la predicción de signo del retorno futuro frente al azar?
- *Método:* tabla de contingencia 2×2 y estadístico chi-cuadrado, exactamente como en el Ejemplo 4.8.
- *Resultado que la apoyaría:* chi-cuadrado significativo y estable entre subperíodos.
- *Resultado que la refutaría:* resultado no significativo, como el que Tsay documenta para la red de IBM.
- *Limitaciones:* la tasa base de aciertos por azar depende de la proporción real de subidas/bajadas en la muestra, hay que tenerlo en cuenta.

**H-NL15. Forecast puntual.**
- *Pregunta:* ¿algún modelo no lineal reduce el MSE/MAD de pronóstico frente a un benchmark simple (media, AR)?
- *Método:* comparación de MSE/MAD/MAPE fuera de muestra, tal como en el Ejemplo 4.7.
- *Resultado que la apoyaría:* reducción consistente y estable del error.
- *Resultado que la refutaría:* error similar o mayor, como en el resultado de Tsay para la red de IBM (91.74 vs. 91.70 del AR(1)).
- *Limitaciones:* la elección de métrica (MSE vs. MAD vs. MAPE) puede cambiar el modelo "ganador".

**H-NL16. Forecast probabilístico.**
- *Pregunta:* ¿la distribución predictiva generada por bootstrap paramétrico está bien calibrada (los percentiles observados se distribuyen uniformemente)?
- *Método:* test de Kolmogorov–Smirnov sobre los percentiles empíricos, como describe la Sección 14.3.
- *Resultado que la apoyaría:* no rechazo de uniformidad.
- *Resultado que la refutaría:* rechazo claro de uniformidad (mala calibración).
- *Limitaciones:* requiere un número suficientemente grande de pronósticos evaluados para tener poder estadístico razonable.

**H-NL17. Desempeño in-sample vs. out-of-sample.**
- *Pregunta:* ¿la ventaja de un modelo no lineal observada dentro de la muestra de ajuste se mantiene fuera de muestra?
- *Método:* comparar explícitamente ambas métricas, no reportar solo una.
- *Resultado que la apoyaría:* ventaja similar dentro y fuera de muestra.
- *Resultado que la refutaría:* ventaja grande dentro de muestra que se reduce o desaparece fuera de muestra (patrón documentado por Tsay en el Ejemplo 4.7).
- *Limitaciones:* ninguna — esta comparación es, en sí misma, el control de calidad más básico y no debería omitirse nunca.

**H-NL18. Estabilidad entre instrumentos.**
- *Pregunta:* si se encuentra estructura no lineal en un instrumento de futuros, ¿aparece también en otros instrumentos similares?
- *Método:* repetir el mismo procedimiento de detección en varios instrumentos y comparar.
- *Resultado que la apoyaría:* estructura similar y consistente entre instrumentos comparables.
- *Resultado que la refutaría:* estructura presente en uno y ausente en otros, sin explicación clara.
- *Limitaciones:* diferencias de liquidez, tamaño de contrato y microestructura entre instrumentos pueden confundir la comparación (temas de capítulos posteriores).

**H-NL19. Estabilidad entre frecuencias.**
- *Pregunta:* ¿la estructura no lineal detectada a una frecuencia (por ejemplo, diaria) se mantiene a otra frecuencia (por ejemplo, intradiaria)?
- *Método:* repetir los tests y modelos en distintas frecuencias del mismo instrumento.
- *Resultado que la apoyaría:* resultados cualitativamente similares entre frecuencias.
- *Resultado que la refutaría:* resultados que cambian sustancialmente con la frecuencia.
- *Limitaciones:* a mayor frecuencia, aparecen efectos de microestructura no cubiertos por este capítulo (Capítulo 5).

---

## 21. Errores metodológicos — auditoría de las 20 afirmaciones

| # | Afirmación | Veredicto | Por qué |
|---|---|---|---|
| 1 | "Si ACF ≈ 0, un modelo no lineal no puede predecir nada." | **INCORRECTA** | Es exactamente el punto del Capítulo 2 y de este capítulo: ACF≈0 mide solo dependencia *lineal*. Puede existir dependencia no lineal genuina con ACF≈0. |
| 2 | "Si rechazamos linealidad, existe predictibilidad." | **INCORRECTA** | Rechazar $H_0$ de linealidad es evidencia de estructura no explicada por un modelo lineal, no evidencia de que esa estructura sea predecible de forma útil fuera de muestra (Sección 11). |
| 3 | "Si detectamos no linealidad, una neural network será mejor." | **INCORRECTA** | Detectar no linealidad no identifica la forma correcta (Sección 13); además, el propio Ejemplo 4.8 de Tsay muestra una red que no mejora un benchmark trivial pese a evidencia de no linealidad en la serie. |
| 4 | "Una red neuronal puede encontrar cualquier señal que exista." | **REQUIERE CONDICIONES** | Solo si esa señal está genuinamente contenida en las variables de entrada, y solo si el proceso de entrenamiento logra encontrarla y generalizarla fuera de muestra (Sección 8) — ninguna de las dos cosas está garantizada. |
| 5 | "Universal approximation significa que una NN puede predecir cualquier serie." | **INCORRECTA** | Universal approximation es una propiedad de *representación* de funciones continuas conocidas, no una garantía de *aprendizaje* ni de predictibilidad de un proceso estocástico con datos finitos (Sección 7.3). |
| 6 | "Más hidden units siempre mejoran el modelo." | **INCORRECTA** | Tsay señala explícitamente que ampliar el número de nodos aumenta la variabilidad del resultado (rango de MSE más amplio en su propio ejemplo) sin garantía de mejora fuera de muestra. |
| 7 | "Un mejor ajuste in-sample implica mejor forecast." | **INCORRECTA** | Es precisamente lo que el Ejemplo 4.7 refuta empíricamente: mejor ajuste dentro de muestra (red vs. AR(1)) no se tradujo en mejor MSFE fuera de muestra. |
| 8 | "TAR demuestra que los mercados tienen regímenes." | **REQUIERE CONDICIONES** | TAR es un modelo aplicable *si* se verifica estadísticamente que una variable de umbral cambia la dinámica; su buen ajuste en una serie concreta no demuestra nada sobre "los mercados" en general, y menos sobre un instrumento no analizado por Tsay. |
| 9 | "Los regímenes de TAR son lo mismo que sesiones horarias." | **INCORRECTA** | Son dos conceptos distintos (Sección 5): uno surge de un umbral estadísticamente definido sobre una variable medida; el otro es una partición manual basada en el reloj. |
| 10 | "Markov Switching detecta automáticamente los verdaderos estados económicos." | **INCORRECTA** | El modelo entrega estados numerados sin etiqueta; la interpretación económica ("expansión", "contracción") es siempre posterior y depende del analista (Sección 4.4). |
| 11 | "Si un modelo encuentra dos regímenes, esos regímenes existen realmente." | **REQUIERE CONDICIONES** | El modelo encuentra una partición estadísticamente útil dado los datos y la especificación; que corresponda a un fenómeno "real" y estable es una hipótesis adicional que debe validarse independientemente (Sección 20, H-NL9). |
| 12 | "STAR siempre es mejor que TAR porque es más flexible." | **INCORRECTA** | Tsay no lo afirma; además señala una desventaja concreta de STAR (dificultad de estimar los parámetros de transición, con t-ratios cercanos a 1). |
| 13 | "Un test BDS significativo identifica qué no linealidad existe." | **INCORRECTA** | El propio Tsay señala que el BDS detecta desviaciones de iid en general, sin identificar la fuente ni la forma concreta (Sección 10). |
| 14 | "Un p-valor pequeño mide cuánto de fuerte es la señal." | **INCORRECTA** | Un p-valor mide la evidencia contra la hipótesis nula bajo los supuestos del test, no la magnitud ni la utilidad económica de la estructura detectada (Sección 11). |
| 15 | "Muchos tests significativos implican robustez." | **REQUIERE CONDICIONES** | Solo si no hubo una búsqueda extensa sin corrección por comparaciones múltiples; de lo contrario, aumenta el riesgo de falsos positivos por azar (Sección 12, data snooping). |
| 16 | "Una interacción estadística es necesariamente útil económicamente." | **INCORRECTA** | Es exactamente la distinción entre las cuatro nociones de la Sección 1: dependencia (incluida la de tipo interacción) no equivale a utilidad económica. |
| 17 | "Un forecast direccional correcto implica rentabilidad." | **INCORRECTA** | No contempla costos operativos, tamaño de la posición, ni la magnitud del movimiento — un acierto direccional pequeño puede no cubrir costos de transacción. |
| 18 | "Un modelo probabilístico es automáticamente mejor que uno puntual." | **REQUIERE CONDICIONES** | Depende de la calibración real de la distribución predictiva (Sección 14.3, test de Kolmogorov-Smirnov) y del uso que se le dé; un modelo probabilístico mal calibrado no es mejor por el solo hecho de dar una distribución. |
| 19 | "Cross-validation aleatoria es apropiada para cualquier serie temporal." | **INCORRECTA** | Precaución añadida en este informe [B] (Sección 7.4): mezclar observaciones sin respetar el orden temporal arriesga fuga de información; los propios ejemplos de Tsay usan siempre partición temporal ordenada. |
| 20 | "Si la relación cambia entre regímenes, debemos construir un modelo de régimen." | **REQUIERE CONDICIONES** | Es una opción disponible, no una obligación automática; depende de si esa estructura es estadísticamente robusta, estable fuera de muestra, y económicamente relevante una vez evaluada con el rigor de las Secciones 9-15. |

---

## 22. Preguntas abiertas

**Dentro del alcance de este capítulo, sin resolver:**

- ¿Existe evidencia de dependencia no lineal en retornos de futuros, una vez removida la estructura lineal (Capítulo 2) y la estructura de volatilidad (Capítulo 3)? Este capítulo da las herramientas para testearlo, pero no lo testea para ningún futuro.
- ¿El efecto de una feature sobre el retorno futuro puede depender de otra feature, en el contexto específico de un futuro intradiario? Es una pregunta abierta, no examinada aquí.
- ¿Una relación predictiva cambia según volatilidad, hora, dirección previa, magnitud del movimiento, o un estado latente, en un instrumento de futuros? Ninguno de los ejemplos de Tsay en este capítulo trata un futuro.
- ¿Una red neuronal mejoraría realmente sobre benchmarks simples (media incondicional, AR, regresión lineal, TAR/STAR) para datos de futuros? El único precedente textual (IBM, retornos mensuales) es negativo para la red, pero no es generalizable a otro instrumento, frecuencia o arquitectura.
- ¿Conviene, para IRIS, un enfoque de clasificación, de regresión, o de distribución completa? El capítulo describe las tres alternativas y su forma de evaluación, pero no elige entre ellas.

**Explícitamente diferidas a capítulos posteriores del libro, sin resolver anticipadamente:**

- **Capítulo 5** — microestructura y dependencia aparente a alta frecuencia: relevante para entender si la no linealidad detectada a frecuencias intradiarias podría deberse, parcial o totalmente, a efectos de microestructura y no a estructura "económica" genuina.
- **Capítulo 8** — relaciones multivariadas, VAR, cointegración, relaciones lead-lag: relevante si en algún momento se considera incorporar más de un instrumento o serie de contexto simultáneamente.
- **Capítulo 10** — volatilidad multivariada: extensión natural de los modelos de volatilidad del Capítulo 3 al caso de múltiples series.
- **Capítulo 11** — modelos de espacio de estados y filtro de Kalman: necesario para profundizar en la Sección 4.1.8 de este capítulo (nonlinear state-space), que aquí se dejó deliberadamente a nivel conceptual.
- **Capítulo 12** — Markov Chain Monte Carlo y modelos latentes: necesario para profundizar en la estimación de Markov Switching (Sección 4.1.4) y en los modelos de espacio de estados no lineales, ambos mencionados por Tsay solo a nivel introductorio en este capítulo.

---

## 23. Checklist de conocimientos adquiridos

Lo que este informe permite explicar, en lenguaje simple, sin necesidad de repetir una fórmula:

- [x] Qué significa que una relación sea no lineal (umbral, saturación, interacción, régimen — todas formas en que el efecto de $x$ deja de ser una pendiente fija).
- [x] Cómo difiere la no linealidad en la media de la heterocedasticidad (cambia la regla que predice, vs. cambia cuánto ruido hay alrededor de esa regla).
- [x] Qué es un threshold (un valor que divide el espacio de una variable en regiones).
- [x] Qué es un régimen (un estado en el que se aplica una dinámica distinta), y por qué "régimen" no siempre significa lo mismo (manual, observable por umbral, latente).
- [x] La diferencia entre régimen observable (TAR/STAR) y régimen latente (Markov Switching), y por qué esa diferencia cambia cómo se calcula un pronóstico.
- [x] Las diferencias prácticas entre TAR (transición abrupta), STAR (transición gradual, con dificultades de estimación) y Markov Switching (estado oculto, inferido probabilísticamente).
- [x] Qué aportan los modelos no paramétricos (dejar que los datos sugieran la forma, a costa de interpretabilidad, eficiencia y necesidad de más datos).
- [x] Qué significa que un coeficiente dependa del contexto (functional coefficient: la influencia del pasado varía según otra variable, sin forma prefijada).
- [x] Cómo una red neuronal feed-forward representa relaciones no lineales (entradas, pesos, sesgos, capa oculta, activación no lineal, salida).
- [x] Qué significa realmente universal approximation (capacidad de representar, no de aprender ni de generalizar).
- [x] Por qué la capacidad del modelo no equivale a la señal disponible en los datos (un modelo flexible no inventa información que no está en $X$).
- [x] Cómo se testea la no linealidad (Q de residuos al cuadrado, bispectral, BDS, RESET/F, tests de umbral), y qué responde y qué no responde cada uno.
- [x] Por qué rechazar linealidad no demuestra predictibilidad (el test detecta estructura, no dice si es explotable).
- [x] Por qué detectar predictibilidad no demuestra rentabilidad (falta descontar costos, y verificar estabilidad y magnitud).
- [x] Cómo se evalúa el pronóstico direccional, puntual y probabilístico, con sus métricas respectivas.
- [x] Por qué el desempeño out-of-sample sigue siendo indispensable (el propio ejemplo de la red de IBM lo demuestra: gana in-sample, empata out-of-sample).
- [x] Qué preguntas quedan abiertas antes de diseñar cualquier sistema (Sección 22).

---

## 24. Conclusiones

Este capítulo amplía el vocabulario adquirido en los Capítulos 1 a 3 con una familia de herramientas para describir y testear estructura que un modelo lineal no puede representar — ya sea en la media (TAR, STAR, Markov Switching, modelos no paramétricos, redes neuronales) o combinada con estructura de volatilidad (como en el propio ejemplo de Tsay con IBM). Se aprendió a distinguir con precisión entre régimen observable y régimen latente, entre transición abrupta y transición gradual, y entre la capacidad de un modelo para *representar* funciones complejas y su capacidad real para *aprender* algo útil de datos finitos y *generalizarlo* fuera de muestra.

El hallazgo más importante de todo el capítulo, sostenido por el propio Tsay con un ejemplo empírico documentado (la red neuronal sobre retornos de IBM que no logra superar a un camino aleatorio con deriva), es que **detectar no linealidad, y construir un modelo con capacidad de representarla, no garantiza ninguna ventaja predictiva real, y mucho menos una ventaja económica**. Esto no es una opinión pesimista sobre los modelos no lineales; es simplemente el resultado de aplicar, con rigor, la misma pregunta que atravesó los tres capítulos anteriores: ¿esto que encontramos se sostiene fuera de la muestra en la que lo encontramos?

**Lo que sabemos, con el respaldo directo del capítulo:** existe un vocabulario preciso y una caja de herramientas estadística para preguntar si la regla que conecta el pasado con el futuro de una serie depende del contexto; existen tests que pueden detectar evidencia de esa dependencia, cada uno sensible a formas distintas de no linealidad; existe una manera formal de pensar en pronósticos no lineales mediante simulación (bootstrap paramétrico); y existe un conjunto claro de formas de evaluar si un pronóstico es útil, según el objetivo (dirección, magnitud, o distribución completa).

**Lo que NO sabemos todavía, y este capítulo no permite saber:** si existe estructura no lineal genuina en algún instrumento de futuros en particular, en qué horizonte, bajo qué condiciones, y si esa estructura —de existir— sería explotable una vez descontados costos operativos. Tampoco sabemos, porque no fue estudiado aquí, cómo se extienden estas ideas al caso multivariado, ni los detalles de estimación de los modelos latentes o de espacio de estados no lineales, que quedan para capítulos posteriores del libro.

La prioridad, tal como lo exige el objetivo de este estudio, sigue siendo comprender qué estructura podría existir antes de elegir cómo modelarla — y este capítulo, más que ningún otro hasta ahora, deja claro que ese "podría existir" necesita evidencia concreta, evaluada honestamente fuera de muestra, antes de convertirse en una decisión de diseño.

---

## 25. Registro de revisión crítica

| Afirmación [B] sensible | Riesgo de sobreinterpretación | Estado |
|---|---|---|
| Analogía entre régimen TAR/STAR y "regímenes horarios" de un sistema de trading | Alto — es fácil, en la práctica, empezar a llamar "régimen TAR" a una partición manual por horario sin verificación estadística | MANTENER, con la advertencia explícita ya incluida en la Sección 4.2 |
| Ejemplo de functional coefficient ("+0.1 en baja volatilidad, −0.2 en alta volatilidad") | Medio — podría leerse como un resultado empírico real de Tsay | MATIZAR — ya está marcado explícitamente como ejemplo sin base empírica, se mantiene esa aclaración |
| Conexión entre STAR (función logística) y activación de redes neuronales | Bajo-medio — es una observación matemática correcta, pero podría malinterpretarse como que ambos modelos son "lo mismo" | MANTENER, dejando claro que es una analogía estructural, no una equivalencia de modelos |
| Extrapolación de "el TAR mejora el pronóstico en contracciones económicas" (desempleo EE.UU.) hacia futuros | Alto — es tentador pensar "entonces en alta volatilidad los modelos no lineales ayudan más" | PREGUNTA ABIERTA — se dejó explícitamente como hipótesis no trasladable en la Sección 18 |
| Advertencia sobre validación cruzada aleatoria en series temporales financieras | Bajo — es una precaución estándar y ampliamente aceptada en la literatura de series temporales, aunque no es textual de este capítulo de Tsay | MANTENER, con la aclaración de que no es una cita textual de Tsay sino una precaución añadida |
| Tabla comparativa de "tres tipos de régimen" (Sección 5) | Bajo — es una síntesis organizativa propia, no un resultado de Tsay | MANTENER, ya está marcada como construcción propia [B] |
| Lectura de que "el valor de modelos no lineales podría no ser uniforme en el tiempo" para futuros | Medio — podría leerse como una recomendación de "buscar regímenes de alta volatilidad para aplicar modelos no lineales" | PREGUNTA ABIERTA — se mantiene explícitamente como hipótesis, no como recomendación |
| Comparación conceptual entre bootstrap paramétrico y "probabilistic forecasting" moderno | Bajo — es una conexión terminológica razonable y ampliamente aceptada | MANTENER |

---

## Informe de cierre

**Secciones efectivamente estudiadas en profundidad (con lectura directa del texto de Tsay):** introducción del Capítulo 4 (definición de linealidad/no linealidad, marco de medias y varianzas condicionales); 4.1.1 Bilinear Model (con Ejemplo 4.1); 4.1.2 TAR/SETAR (con Ejemplos 4.2 y 4.3); 4.1.3 STAR (con Ejemplo 4.4); 4.1.4 Markov Switching (con Ejemplo 4.5); 4.1.5 Nonparametric Methods (kernel regression, bandwidth selection, local linear regression, con Ejemplo 4.6); 4.1.6 Functional Coefficient AR; 4.1.7 Nonlinear Additive AR; 4.1.9 Neural Networks (con Ejemplos 4.7 y 4.8, estudiada con especial profundidad); 4.2.1 Tests no paramétricos (Q de residuos al cuadrado, bispectral, BDS); 4.2.2 Tests paramétricos (RESET, Keenan, F-test de Tsay, tests de umbral); 4.2.3 Applications (Tabla 4.2, cinco series); 4.3 Modeling; 4.4 Forecasting (4.4.1 Parametric Bootstrap, 4.4.2 Forecasting Evaluation); 4.5 Application (caso completo del desempleo trimestral de EE.UU., con Tablas 4.3 y 4.4).

**Secciones tratadas a nivel conceptual, según lo pedido explícitamente por el enunciado:** 4.1.8 Nonlinear State-Space Model (se presentó la estructura general —estado, ecuación de estado, ecuación de observación— sin profundizar en los métodos de estimación por Monte Carlo, que Tsay mismo remite a capítulos posteriores).

**Material omitido, y por qué:** los programas RATS/S-Plus/R de los Apéndices A y B del capítulo se omitieron por instrucción explícita del enunciado (código específico obsoleto); los ejercicios 4.1 a 4.5 del final del capítulo se omitieron por ser mecánicos y fuera del alcance de "comprensión y adquisición de conocimiento"; las derivaciones algebraicas completas de la regresión local lineal ponderada (Sección 4.1.5) se resumieron en su lógica general en vez de reproducirse paso a paso, porque no aportaban comprensión adicional más allá de la ya explicada; las referencias bibliográficas extensas al final del capítulo no se listaron individualmente, solo se citaron los autores relevantes en el cuerpo del texto donde correspondía a un resultado específico.

**Archivo generado:** `Tsay_Cap4_analisis_futuros_ML.md`.

**Principales hallazgos [A]:** (1) la definición formal de no linealidad como cualquier desviación de la representación lineal de Wold; (2) la taxonomía completa TAR/STAR/Markov Switching con sus diferencias de estimación y de cálculo de pronóstico; (3) la propiedad de aproximación universal de las redes feed-forward, y su contraparte empírica —una red que no logra superar a un camino aleatorio con deriva en pronóstico de retornos mensuales de IBM, tanto en magnitud como en dirección—; (4) el catálogo de tests de no linealidad (Q, bispectral, BDS, RESET/F, tests de umbral), cada uno con su propio alcance y limitaciones, y sin que ninguno domine a los demás; (5) el caso completo del desempleo trimestral de EE.UU., donde los modelos no lineales mejoran el pronóstico lineal solo en un subconjunto de contextos (contracciones económicas), no de forma general.

**Principales interpretaciones [B]:** (1) la distinción entre las cuatro nociones —no linealidad, dependencia, predictibilidad, utilidad económica— como guardrail central para cualquier desarrollo futuro de IRIS; (2) la tabla de tres tipos de régimen (manual, observable, latente) como herramienta para evitar confusiones terminológicas; (3) la advertencia sobre validación cruzada aleatoria en series temporales financieras, coherente con —pero no textual de— los ejemplos de Tsay; (4) la idea de que, si existiera valor en modelos no lineales para futuros, ese valor podría no ser uniforme en el tiempo, como hipótesis y no como hallazgo.

**Preguntas abiertas:** documentadas en detalle en la Sección 22, con separación explícita entre las que quedan abiertas dentro del propio Capítulo 4 y las que se difieren formalmente a los Capítulos 5, 8, 10, 11 y 12.

**Afirmaciones especialmente susceptibles de sobreinterpretación**, señaladas también en la tabla de la Sección 25: la analogía entre régimen TAR y régimen horario (por el riesgo de usarse sin la verificación estadística que la distingue); la extrapolación del resultado del desempleo de EE.UU. ("los modelos no lineales ayudan más en contracciones") hacia una expectativa sobre futuros en alta volatilidad; y, de forma más general, cualquier lectura que convierta el resultado negativo de la red neuronal de IBM en una conclusión universal sobre redes neuronales en mercados financieros — el capítulo documenta un caso, no una ley general, en ninguna de las dos direcciones (ni a favor ni en contra de las redes neuronales).
