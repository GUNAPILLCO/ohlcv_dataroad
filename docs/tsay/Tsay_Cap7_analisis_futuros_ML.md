# Tsay — Capítulo 7: Extreme Values, Quantiles, and Value at Risk
## Informe de estudio para el Proyecto IRIS

**Convenciones de este documento**

- **[A]** = definición, resultado, ejemplo empírico o afirmación que proviene directamente de Tsay (*Analysis of Financial Time Series*, 3ª ed., Capítulo 7).
- **[B]** = interpretación, extensión o adaptación propia hacia futuros, datos OHLCV, trading o Machine Learning. Nunca es una decisión de diseño; es una hipótesis a evaluar.
- **PREGUNTA ABIERTA** = cuestión que este capítulo no permite resolver con la información disponible.
- Cada resultado empírico [A] indica, cuando está disponible: activo/serie, frecuencia, período, tamaño de muestra, variable analizada, método, resultado y limitaciones. Ningún resultado sobre acciones (IBM, Intel), datos diarios de 1962–2008, se generaliza automáticamente a MNQ, a futuros ni a datos intradiarios.
- Este informe continúa los estudios de los Capítulos 1, 2, 3, 4 y 5. **El Capítulo 6 fue deliberadamente diferido por baja prioridad**; no se estudia aquí ni se intenta compensar su omisión. La única mención de material del Capítulo 6 es el reenvío que hace el propio Tsay a la definición de proceso de Poisson (Sección 6.9 del libro), que se explica aquí de forma autocontenida y mínima.
- Esta fase es exclusivamente de **comprensión y adquisición de conocimiento**. No se implementa VaR, ES, quantile regression ni EVT; no se seleccionan thresholds, features, targets ni arquitecturas; no se ejecuta ningún experimento ni se descargan datos.

---

## 1. Resumen ejecutivo

Hasta este punto del estudio, casi todo lo que aprendimos a preguntarle a los datos giraba en torno a dos objetos: **dónde está el centro** de la distribución futura (la media condicional, Capítulos 2 y 4) y **qué tan ancha es** esa distribución (la varianza condicional, Capítulo 3). El Capítulo 7 introduce una familia de preguntas distintas, que no se responden con ninguno de esos dos números:

> Dadas las condiciones actuales, ¿**dónde está un percentil determinado** de la distribución futura? ¿Qué probabilidad hay de entrar en una zona muy mala? Y si entramos, ¿**qué tan severo** puede ser el resultado?

Formalmente, el capítulo agrega a $E(r_{t+h}\mid X_t)$ y $\mathrm{Var}(r_{t+h}\mid X_t)$ un tercer objeto:

$$Q_\alpha(r_{t+h}\mid X_t)$$

y, más allá de él, todo el aparato conceptual necesario para hablar de lo que ocurre **en la cola**: eventos raros, pérdidas severas, y la incómoda realidad de que justamente ahí es donde tenemos menos datos.

**[A]** El vehículo que Tsay usa para organizar todo esto es el **Value at Risk (VaR)**, que define de forma completamente general como un **cuantil de la distribución de pérdidas**: dada una probabilidad de cola $p$ y un horizonte $\ell$, VaR es el valor tal que la probabilidad de perder esa cantidad o más es $p$. A partir de esa definición, el capítulo recorre cinco maneras distintas de estimar ese cuantil —RiskMetrics, modelos econométricos tipo AR–GARCH, cuantil empírico, teoría de valores extremos tradicional (block maxima), y el enfoque moderno de peaks-over-threshold— y las compara sobre la misma serie: **retornos logarítmicos diarios de la acción de IBM, 3 de julio de 1962 al 31 de diciembre de 1998, 9190 observaciones**, para una posición larga de 10 millones de dólares.

El resultado más importante de esa comparación, y probablemente el mensaje metodológico central del capítulo, es que **los cinco métodos producen números sustancialmente distintos sobre exactamente los mismos datos**. **[A]** Para una probabilidad de cola de 1%, los VaR obtenidos van de \$340.013 (EVT tradicional con bloques mensuales) a \$475.943 (AR–GARCH con Student-t de 5 grados de libertad). Para una probabilidad de 0.1%, el rango se amplía de \$546.641 (AR–GARCH gaussiano) a \$836.341 (AR–GARCH Student-t). **[A]** Tsay comenta esto sin ambigüedad: *"There are substantial differences among different approaches. This is not surprising because there exists substantial uncertainty in estimating tail behavior of a statistical distribution. Since there is no true VaR available to compare the accuracy of different approaches, we recommend that one applies several methods to gain insight into the range of VaR."* No declara ningún ganador.

Ese es el guardrail que atraviesa todo este informe:

$$\boxed{\text{el extremo que queremos estimar es precisamente donde tenemos menos datos}}$$

Junto con dos más, igualmente presentes en cada sección:

$$\boxed{\text{modelar la cola} \neq \text{predecir dirección} \neq \text{tener una estrategia rentable}}$$

$$\boxed{\text{un estimador de un evento muy raro puede tener enorme incertidumbre}}$$

Para el Proyecto IRIS, la parte del capítulo que abre las preguntas más interesantes **no es el VaR en sí**. El VaR es una herramienta de gestión clásica de riesgo, y este proyecto no está construyendo un sistema de gestión de riesgo de cartera. Lo que sí resulta conceptualmente valioso son tres cosas:

1. **La distinción entre cuantil incondicional y cuantil condicional.** Un percentil calculado sobre "todos los retornos mezclados" es un objeto distinto de un percentil calculado "dadas las condiciones actuales". **[A]** Tsay hace esta distinción explícita desde la introducción del capítulo (enfoques *unconditional* vs. *conditional*) y la lleva hasta el final, cuando permite que los parámetros que describen la cola dependan de variables explicativas conocidas antes de $t$ —entre ellas, explícitamente, la volatilidad condicional del Capítulo 3.

2. **La quantile regression** (Koenker y Bassett, 1978), que Tsay presenta como una manera de estimar directamente un cuantil condicional **sin** especificar toda la distribución, usando una función de pérdida asimétrica. **[B]** Ésta es la conexión conceptual más directa hacia Machine Learning de todo el capítulo: un modelo que se entrena para estimar $Q_{0.10}(Y\mid X)$ está resolviendo un problema distinto de uno que se entrena para estimar $E[Y\mid X]$, y nada garantiza que ambos problemas tengan la misma cantidad de señal disponible en los mismos datos.

3. **La idea de que una misma variable puede informar de manera muy distinta sobre la media, sobre la volatilidad y sobre la cola.** **[A]** Tsay lo muestra empíricamente en la Sección 7.7.8, donde la volatilidad GARCH y una medida cualitativa de volatilidad reciente resultan significativas para los parámetros de escala, ubicación y forma de la distribución de los extremos de IBM, mientras que un indicador de "pánico vendedor" del día anterior y un indicador de cuarto trimestre no lo son.

Las preguntas que este capítulo permite **formular** —sin responder para ningún futuro, y menos aún para MNQ, que Tsay nunca analiza— son:

- ¿Es posible que nuestros datos contengan poca información sobre la **dirección media** del próximo movimiento y, sin embargo, información más estable sobre la **dispersión**, los **cuantiles** o la **probabilidad de movimientos extremos**?
- ¿Una feature puede ser predictiva de la cola sin ser predictiva de la dirección?
- ¿Millones de barras de 1 minuto resuelven automáticamente el problema de estimar eventos raros?

Ninguna se responde aquí. El estado final de este capítulo es, deliberadamente:

**KNOWLEDGE ACQUIRED — NO DESIGN DECISIONS ADOPTED.**

---

## 2. Las colas desde cero

Antes de hablar de VaR, de cuantiles o de teoría de valores extremos, hace falta tener una imagen mental muy concreta de qué es una "cola".

### 2.1 La imagen básica

Imaginemos que anotamos el retorno de cada día durante varios años y armamos un histograma: cuántas veces el retorno cayó cerca de 0%, cuántas veces cerca de +1%, cuántas veces cerca de −3%, y así.

El resultado típico tiene tres zonas:

- **El centro.** La gran mayoría de los días. Movimientos "normales", del tamaño habitual. Aquí está la enorme masa de las observaciones.
- **La cola izquierda.** Muy lejos hacia la izquierda: los días de **pérdidas muy grandes**. Pocos días, pero de gran magnitud.
- **La cola derecha.** Muy lejos hacia la derecha: los días de **ganancias muy grandes**. También pocos.

La palabra "cola" es una metáfora visual directa: el histograma se ve como un cuerpo grueso en el medio y dos extremos que se van adelgazando y estirando hacia los costados.

### 2.2 Tail probability (probabilidad de cola)

**Qué problema resuelve:** necesitamos una manera de decir "qué tan lejos vamos a mirar hacia el extremo" que no dependa de la unidad de medida ni del activo.

**Intuición:** en vez de decir "los días en que se perdió más de 3%", decimos "el 1% de los días peores". La *tail probability* es esa fracción: la probabilidad de estar en la zona extrema que estamos mirando.

**Ejemplo numérico extremadamente sencillo:** tenemos 1000 días de datos. Si nos interesa una tail probability de 1%, estamos hablando de aproximadamente los 10 días peores. Si nos interesa una tail probability de 0.1%, estamos hablando de aproximadamente **1** día. Con 1000 observaciones, la cola de 0.1% se apoya, nominalmente, en una sola observación.

**Definición técnica:** dada una variable aleatoria $L$ (por ejemplo, la pérdida) y un umbral $x$, la tail probability asociada a ese umbral es $\Pr(L \ge x)$.

**Qué NO significa:** una tail probability de 1% no significa que "exactamente 1 de cada 100 días" vaya a caer en esa zona. Significa que, **bajo el modelo o la distribución que estamos usando**, esa es la probabilidad asignada a esa región. Los conteos observados fluctúan alrededor de esa probabilidad, y además pueden agruparse en el tiempo (ver Sección 25).

### 2.3 Evento raro y evento extremo

**[A]** Tsay usa las expresiones *rare event* y *extraordinary event* de forma esencialmente intercambiable en la introducción del capítulo, y menciona ejemplos concretos: el crash de Wall Street de octubre de 1987, la crisis de Long-Term Capital Management, la quiebra de Lehman Brothers, y el aumento sustancial de volatilidad de la crisis financiera reciente (medida por el índice VIX del CBOE).

Conviene, sin embargo, separar dos ideas [B]:

- **Evento raro** = evento con **probabilidad baja** de ocurrir en un período dado.
- **Evento extremo** = evento de **magnitud muy grande**, situado lejos en la cola.

En la práctica financiera casi siempre coinciden (los movimientos muy grandes son poco frecuentes), pero conceptualmente son cosas distintas: la rareza es una afirmación sobre frecuencia, y lo extremo es una afirmación sobre magnitud. Esta distinción vuelve a aparecer en la Sección 29.6 ("no confundir rareza con importancia económica").

### 2.4 Relación con skewness, kurtosis y heavy tails (Capítulo 1)

En el Capítulo 1 aprendimos tres descriptores de la forma de una distribución:

- **Skewness (asimetría):** si la distribución se estira más hacia un lado que hacia el otro.
- **Kurtosis (curtosis):** cuánto peso relativo tienen, en conjunto, las zonas alejadas del centro respecto de lo que tendría una normal.
- **Heavy tails (colas pesadas):** la propiedad de que los valores extremos ocurren **más frecuentemente** de lo que predeciría una distribución normal con la misma media y varianza.

**Aclaración importante, exigida por el propio marco del capítulo:**

> La curtosis resume una propiedad **global** de la distribución (es un promedio ponderado que involucra a *todas* las observaciones, con más peso en las alejadas). La teoría de valores extremos, en cambio, se concentra **deliberadamente y solo** en los extremos, descartando el resto de los datos.

Son dos maneras muy distintas de mirar el mismo fenómeno. Un número de curtosis alto es una señal de que las colas podrían ser pesadas, pero no describe *cómo* se comporta la cola: no dice si la cola izquierda y la derecha son iguales, ni con qué velocidad decae la probabilidad a medida que nos alejamos, ni qué forma funcional tiene ese decaimiento. Toda la Sección 15 en adelante existe precisamente para responder esas preguntas que la curtosis no responde.

**[A]** Tsay conecta explícitamente ambos mundos en la Sección 7.5.1: la familia **Fréchet** de distribuciones límite de extremos "incluye a las distribuciones estables y Student-t", mientras que la familia **Gumbel** "consiste en distribuciones de colas delgadas como la normal y la lognormal". Es decir: la forma de la cola que uno asume en el Capítulo 1 (normal vs. Student-t) determina en qué familia de distribuciones extremas cae el máximo.

### 2.5 Qué NO es una cola

- Una cola **no** es "los outliers del dataset". Un outlier es una observación que, por algún criterio, consideramos anómala o sospechosa; una observación en la cola puede ser perfectamente legítima y esperable dentro del modelo. La distinción entre extremo estadístico y dato erróneo se desarrolla en la Sección 26.5.
- Una cola **no** es "la parte del gráfico donde hay poca información". Al contrario: para muchas preguntas de riesgo, es la parte que más importa. Lo que sí es cierto es que es la parte donde hay **pocos datos**, que es un problema distinto y central en todo este capítulo.
- Una cola pesada **no** implica predictibilidad. Que los eventos extremos ocurran más seguido de lo que predice una normal no dice absolutamente nada sobre nuestra capacidad de anticipar **cuándo** van a ocurrir.

---

## 3. Long, short, pérdidas y convención de signos

Ésta es una sección deliberadamente pedante, porque el VaR genera confusiones de signo con una facilidad notable, y esas confusiones invalidan cualquier interpretación posterior.

### 3.1 Posición LONG (comprada)

Estamos comprados: tenemos el activo (o un contrato que se comporta como si lo tuviéramos).

- Si el precio **sube**, ganamos.
- Si el precio **baja**, perdemos.

Por lo tanto, para una posición long, **lo que nos interesa es la cola izquierda del retorno**: los retornos muy negativos.

### 3.2 Posición SHORT (vendida)

Estamos vendidos: nos comprometimos a entregar el activo más adelante, apostando a que baje.

- Si el precio **baja**, ganamos.
- Si el precio **sube**, perdemos.

Por lo tanto, para una posición short, **lo que nos interesa es la cola derecha del retorno**: los retornos muy positivos.

### 3.3 Por qué Tsay trabaja con $-r_t$

**[A]** Tsay define el VaR sobre una **función de pérdida** $L(\ell)$, y expresa su definición en términos de la **cola superior** de esa función de pérdida (Ec. 7.1). En sus propias palabras (Remark de la Sección 7.1):

> *"The definition of VaR in Eq. (7.1) is based on the upper tail of a loss function. For a long financial position, loss occurs when the returns are negative. Therefore, we shall use negative returns in data analysis for a long financial position."*

La razón es puramente de comodidad: si trabajamos con **pérdidas expresadas como números positivos**, entonces "pérdida grande" = "número grande", y todas las herramientas de teoría de valores extremos, que están escritas para máximos y colas superiores, se aplican directamente sin cambiar de signo a cada paso.

**[A]** Tsay lo dice explícitamente también en la Sección 7.5: las propiedades del **mínimo** de una serie se obtienen de las del **máximo** mediante un simple cambio de signo, ya que $r_{(1)} = -\max_{1\le j\le n}\{-r_j\}$.

### 3.4 Ejemplo numérico pequeño

Supongamos cinco días con estos retornos logarítmicos, en porcentaje:

$$-2.0,\quad +0.5,\quad -0.3,\quad +1.2,\quad -3.5$$

**Para una posición LONG**, la pérdida de cada día es $-r_t$:

$$+2.0,\quad -0.5,\quad +0.3,\quad -1.2,\quad +3.5$$

La peor pérdida es 3.5 (el día en que el retorno fue −3.5%). Esa es la **cola superior de la pérdida** = **cola inferior del retorno**.

**Para una posición SHORT**, la pérdida de cada día es $+r_t$:

$$-2.0,\quad +0.5,\quad -0.3,\quad +1.2,\quad -3.5$$

La peor pérdida es 1.2 (el día en que el retorno fue +1.2%). Esa es la **cola superior de la pérdida** = **cola superior del retorno**.

### 3.5 Cuatro expresiones que NO deben mezclarse

Ésta es la fuente de casi todos los errores:

| Expresión | Qué significa exactamente |
|---|---|
| **percentil 1% del retorno** | El valor por debajo del cual queda el 1% de los retornos. Para retornos, es un número **negativo** grande (una caída fuerte). |
| **percentil 99% de la pérdida** | El valor por debajo del cual queda el 99% de las pérdidas. Es un número **positivo** grande. Para una posición long, es exactamente el mismo evento que el anterior, con el signo cambiado. |
| **"VaR al 99%"** | Nombre usual del VaR calculado con **tail probability $p=0.01$**. El "99%" se refiere al nivel de confianza $1-p$, no a la cola. |
| **tail probability = 1%** | La probabilidad $p$ asignada a la región de pérdidas peores que el VaR. |

Las cuatro descripciones apuntan, para una posición long, al **mismo umbral**. Pero decir "el percentil 99%" cuando uno quiere decir "el percentil 1% del retorno" es un error si no se aclaró antes que estamos trabajando con pérdidas y no con retornos.

**Regla práctica:** antes de interpretar cualquier número de VaR, hay que responder dos preguntas: *(1) ¿esto está expresado en retornos o en pérdidas?* y *(2) ¿el porcentaje que aparece en el nombre es la tail probability $p$ o el nivel de confianza $1-p$?*

### 3.6 Colas asimétricas: por qué long y short no son simétricos

**[A]** En la aplicación de EVT a IBM (Sección 7.5.3 del libro), Tsay estima el parámetro de forma por separado para los extremos positivos y los negativos, y encuentra que **no son iguales**: el parámetro de forma "aparece ser mayor para los extremos negativos, indicando que el retorno logarítmico diario puede tener una cola izquierda más pesada".

Esto tiene una consecuencia directa e importante:

$$\text{left tail} \neq \text{right tail} \quad \text{en general}$$

**[B]** Por lo tanto, un modelo de cola ajustado para una posición long no es automáticamente válido para una posición short del mismo instrumento, y viceversa. Son **dos problemas de estimación separados**, sobre dos subconjuntos de datos distintos, que pueden dar resultados distintos. Esto se conecta directamente con la **skewness** del Capítulo 1: una distribución asimétrica es, por definición, una distribución cuyas dos colas no son espejo una de la otra.

**No debe asumirse simetría** en ningún momento del análisis de colas, ni siquiera como aproximación de trabajo, sin haberlo verificado sobre los datos concretos.

---

## 4. Value at Risk

### 4.1 ¿Qué problema intenta resolver?

Una institución financiera tiene una posición en el mercado. Quiere una respuesta a una pregunta muy concreta: **"si las cosas salen mal, ¿de qué orden de magnitud estamos hablando?"** No quiere una descripción completa de la distribución de resultados posibles; quiere **un número** que resuma la magnitud del riesgo, que se pueda reportar, comparar entre mesas de operaciones, y usar para fijar requisitos de capital.

**[A]** Tsay lo contextualiza: existen tres categorías principales de riesgo financiero —riesgo de crédito, riesgo operacional y riesgo de mercado—, y el VaR se ocupa principalmente del **riesgo de mercado**, aunque el concepto es aplicable a los otros. **[A]** Puede usarse por instituciones financieras para evaluar su propio riesgo, o por un comité regulatorio para fijar requisitos de margen. En ambos casos, el propósito es asegurar que la institución **siga en pie después de un evento catastrófico**.

**[A]** Un detalle conceptual que Tsay señala y que vale la pena registrar: desde el punto de vista de una institución, el VaR se define como *la pérdida máxima de una posición durante un período dado, para una probabilidad dada*; desde el punto de vista de un regulador, se define como *la pérdida mínima bajo circunstancias de mercado extraordinarias*. **Ambas definiciones conducen a la misma medida**, aunque los conceptos parezcan distintos.

**⚠️ Atención a esa frase "pérdida máxima".** En esa definición informal, "máxima" está calificado por *"for a given probability"*, y remite a la idea de "el peor caso **dentro del $(1-p)$ de escenarios que consideramos normales**". **No significa** que la pérdida no pueda superar ese valor: la definición formal que sigue (Ec. 7.1) deja explícito que hay una probabilidad $p$ de superarlo. Es exactamente la confusión que la Sección 4.4 desarma punto por punto.

### 4.2 Explicación intuitiva

Imaginemos que dibujamos la distribución de todas las pérdidas posibles de mañana. El VaR es un **punto marcado en esa distribución**: el punto tal que, hacia la derecha (pérdidas peores), queda solo una fracción $p$ de la probabilidad total.

Es, literalmente, **la frontera de la zona mala**. Nada más que eso.

### 4.3 Ejemplo numérico

> **"VaR diario al 99% = 1.000 USD."**

Lectura correcta:

> Bajo el modelo utilizado, aproximadamente el 99% de las veces la pérdida diaria no debería superar los 1.000 USD; queda aproximadamente un 1% de probabilidad en la cola más allá de ese umbral.

O, equivalentemente: bajo ese modelo, hay aproximadamente 1% de probabilidad de perder 1.000 USD o más en un día.

### 4.4 Qué NO significa — sección obligatoria

Esta lista es la parte más importante de toda la sección:

1. **NO es la pérdida máxima posible.** Es un cuantil, un punto de corte. Por construcción, hay una probabilidad $p$ de estar **más allá** de él.

2. **NO significa que solo se puedan perder 1.000 USD.** Se puede perder mucho más. El VaR simplemente no dice cuánto.

3. **NO dice cuánto perderemos si el VaR es superado.** **[A]** Éste es exactamente el punto que Tsay señala en su tercer Remark de la Sección 7.1: *"VaR is just a quantile of the loss function. It does not fully describe the upper tail behavior of the loss function. In practice, two assets may have the same VaR yet encounter different losses when the VaR is exceeded."* Ésta es precisamente la motivación del Expected Shortfall (Sección 7).

4. **NO garantiza que exactamente 1 de cada 100 días futuros exceda el VaR.** Es una afirmación probabilística bajo un modelo, sujeta a fluctuación muestral y —muy importante— a la posibilidad de que las excedencias se **agrupen en el tiempo** (Sección 25).

5. **Depende completamente del modelo, la distribución asumida, la ventana de datos y el horizonte.** Toda la Sección 18 y 19 de este informe existe para documentar exactamente cuánto puede variar el número según esas elecciones.

6. **[A] No es sub-aditivo.** Tsay lo señala explícitamente: el VaR no satisface la propiedad de sub-aditividad, que establecería que la medida de riesgo de dos carteras fusionadas no debería ser mayor que la suma de sus medidas de riesgo por separado. *"Therefore, care must be exercised in using VaR to measure risk."* Es un defecto técnico conocido de la medida.

7. **[A] Ignora, en la práctica, la incertidumbre de los parámetros.** Tsay dedica un Remark completo a esto: el VaR debería calcularse usando la **distribución predictiva** de los retornos futuros, que sí tendría en cuenta la incertidumbre de los parámetros de un modelo correctamente especificado. Pero *"predictive distribution is hard to obtain, and most of the available methods for VaR calculation ignore the effects of parameter uncertainty."*

### 4.5 Definición técnica

**[A]** Sea $t$ el índice temporal actual, y consideremos el riesgo de una posición financiera para los próximos $\ell$ períodos. Sea $\Delta V(\ell)$ el cambio de valor de los activos subyacentes entre $t$ y $t+\ell$, y sea $L(\ell)$ la función de pérdida asociada (positiva o negativa según la posición sea short o long). Sea $F(x)$ la función de distribución acumulada (CDF) de $L(\ell)$. Entonces el VaR con tail probability $p$ se define por:

$$p = \Pr[L(\ell) \ge \mathrm{VaR}] = 1 - \Pr[L(\ell) < \mathrm{VaR}] \tag{7.1}$$

**Explicación de cada símbolo:**

- $\ell$: el **horizonte temporal** (cuántos períodos hacia adelante miramos).
- $L(\ell)$: la **pérdida** acumulada en ese horizonte. Es una variable aleatoria en el momento $t$: todavía no sabemos cuánto valdrá.
- $F(x)$: la **función de distribución acumulada** de esa pérdida, es decir, $F(x)=\Pr[L(\ell)\le x]$. Nos dice qué probabilidad hay de que la pérdida sea menor o igual a $x$.
- $p$: la **tail probability**. Típicamente 0.05, 0.01, o 0.001.
- $\mathrm{VaR}$: el número que buscamos.

### 4.6 VaR como cuantil — la relación central

**[A]** Para cualquier CDF univariada $F(x)$ y cualquier probabilidad $q$ con $0<q<1$, la cantidad

$$x_q = \inf\{x \mid F(x)\ge q\}$$

se llama el **$q$-ésimo cuantil** de $F(x)$, donde $\inf$ denota el menor número real $x$ que satisface $F(x)\ge q$. Si $L(\ell)$ es continua, entonces $q = \Pr[L(\ell)\le x_q]$.

**[A]** Por lo tanto, si conociéramos $F(x)$, tendríamos $1-p = \Pr[L(\ell) < \mathrm{VaR}]$, de modo que:

$$\boxed{\mathrm{VaR} = x_{1-p} = \text{el } (1-p)\text{-ésimo cuantil de la distribución de pérdidas}}$$

**[A]** Tsay señala que a veces se llama al VaR "el cuantil superior $p$-ésimo", porque $p$ es la probabilidad de la cola superior de la distribución de pérdidas. Y agrega la observación decisiva: **en la práctica, la CDF es desconocida**. Por lo tanto:

> *"Studies of VaR are essentially concerned with estimation of the CDF and/or its quantile, especially the upper tail behavior of the loss CDF."*

Es decir: todo el capítulo, desde RiskMetrics hasta el extremal index, es una colección de estrategias distintas para estimar el mismo objeto desconocido.

### 4.7 Los cinco ingredientes de un cálculo de VaR

**[A]** Tsay enumera explícitamente los factores que intervienen:

1. **La probabilidad de interés $p$**, por ejemplo $p=0.01$ para gestión de riesgo y $p=0.001$ para stress testing.
2. **El horizonte temporal $\ell$**, que puede ser fijado por un regulador: 1 día o 10 días para riesgo de mercado, 1 año o 5 años para riesgo de crédito.
3. **La frecuencia de los datos**, que puede no coincidir con el horizonte $\ell$. Observaciones diarias son habituales en riesgo de mercado.
4. **La CDF $F(x)$ o sus cuantiles.**
5. **El monto de la posición** o el valor mark-to-market de la cartera.

**[A]** De estos cinco, "la CDF $F(x)$ es el foco del modelado econométrico. Distintos métodos para estimar la CDF dan lugar a distintos enfoques de cálculo de VaR."

### 4.8 De porcentaje a dólares

**[A]** Una nota práctica de Tsay: como los retornos logarítmicos corresponden aproximadamente a cambios porcentuales en el valor del activo, si se calcula el VaR sobre la distribución de $r_{t+1}$ el resultado está **en porcentaje**. El monto en dólares es entonces:

$$\mathrm{VaR}_{\$} = \text{Valor de la posición} \times \mathrm{VaR}_{\text{(log returns)}}$$

y, si se necesita mayor precisión, puede usarse la aproximación $\mathrm{VaR}_{\$} = \text{Valor} \times [\exp(\mathrm{VaR}_{\text{log}})-1]$.

### 4.9 Relevancia posible para futuros / ML [B]

**PREGUNTA ABIERTA, no conclusión.** El VaR como medida regulatoria de riesgo de mercado no es un objetivo del Proyecto IRIS. Lo que sí es transferible es la **estructura conceptual**: la idea de que estimar un punto específico de la cola de una distribución futura condicional es un problema estadístico bien definido, distinto de estimar su media, y con sus propias dificultades (pocos datos, sensibilidad al modelo). Esa estructura es la que se explota en las Secciones 12–14 y 22 de este informe.

**No se adopta el VaR como target, ni como métrica de evaluación, ni como componente de ninguna señal.**

---

## 5. VaR vs volatilidad

Esta distinción merece sección propia porque es fácil colapsar ambos conceptos en "una medida de cuánto se mueve el precio", y no son lo mismo.

### 5.1 Dos preguntas distintas

| | Pregunta que responde |
|---|---|
| **Volatilidad** ($\sigma$) | "¿Qué tan **ancha** es la distribución?" Es una medida de dispersión global: promedia la magnitud de todas las desviaciones respecto del centro. |
| **VaR** | "¿**Dónde está un punto específico** de una cola?" Es una coordenada, un punto de corte, no una medida de dispersión. |

Una analogía [B]: la volatilidad es como decir "esta ruta tiene curvas de una cerradura promedio de tal magnitud"; el VaR es como decir "en el kilómetro 87 hay una curva de este radio". El primero resume todo el recorrido; el segundo señala un lugar específico.

### 5.2 Dos distribuciones con la misma volatilidad y distinto VaR

Éste es el punto central, y se sigue directamente de lo que aprendimos en el Capítulo 1.

Consideremos dos distribuciones con **exactamente la misma media (0) y la misma desviación estándar (1%)**:

- **Distribución A: normal.** El 99% de la masa queda por debajo de aproximadamente $+2.326\sigma = 2.33\%$.
- **Distribución B: Student-t estandarizada con 5 grados de libertad.** La misma varianza, pero con colas más pesadas. El percentil 99% de una $t_5$ es 3.3649, y de su versión estandarizada es $3.3649/\sqrt{5/3} = 2.606$. Es decir, el 99% de la masa queda por debajo de aproximadamente $2.61\%$.

**Misma volatilidad. VaR al 99% distinto: 2.33% vs 2.61%.** Y la diferencia crece cuanto más adentro de la cola vamos: a una tail probability de 0.1%, la separación entre ambas distribuciones es mucho mayor.

**[A]** Este fenómeno está documentado numéricamente en el propio capítulo. En el Ejemplo 7.3, sobre IBM, el modelo AR(2)–GARCH(1,1) gaussiano y el AR(2)–GARCH(1,1) con Student-t de 5 grados de libertad producen VaR al 5% prácticamente idénticos (\$287.700 vs \$283.520), pero al 1% ya difieren (\$409.738 vs \$475.943), y al 0.1% la diferencia es grande (\$546.641 vs \$836.341). **[A]** Tsay comenta: *"we see the heavy-tail effect of using a Student-t distribution with 5 degrees of freedom; it increases the VaR when the tail probability becomes smaller."*

$$\boxed{\text{misma volatilidad} \nRightarrow \text{mismo VaR}}$$

### 5.3 Conexión con el Capítulo 3

En el Capítulo 3 aprendimos la descomposición:

$$r_t = \mu_t + a_t, \qquad a_t = \sigma_t \epsilon_t$$

Esta descomposición separa **dos fuentes distintas** de un movimiento extremo observado:

1. **La distribución de $\epsilon_t$** — la forma de la distribución del shock estandarizado. Si $\epsilon_t$ tiene colas pesadas (por ejemplo, Student-t), habrá shocks estandarizados grandes con más frecuencia que bajo normalidad.
2. **El nivel de $\sigma_t$** — la escala. Aunque $\epsilon_t$ fuera perfectamente normal, un valor de $\sigma_t$ muy alto produce retornos observados muy grandes en términos absolutos.

**Consecuencia clave [A]/[B]:** una cola extrema observada en la serie de retornos **puede provenir de cualquiera de las dos fuentes, o de ambas**. Un día de −5% puede ser un shock estandarizado de −5 en un régimen de volatilidad de 1%, o un shock estandarizado perfectamente normal de −1.7 en un régimen de volatilidad de 3%.

**[A]** Ésta es exactamente la razón por la cual el enfoque econométrico de la Sección 7.3 de Tsay produce un VaR **condicional al estado actual de volatilidad**, mientras que el enfoque de cuantil empírico o de EVT tradicional produce un número **incondicional**, que mezcla todos los regímenes de volatilidad de la muestra.

**[B]** Y es también la razón por la cual la sección 7.7.6 (variables explicativas en el modelo de extremos) es conceptualmente tan interesante: permite que los parámetros de la cola dependan de la volatilidad actual, en vez de asumir una cola única para todos los estados del mercado.
---

## 6. RiskMetrics y sus supuestos

**Estudio selectivo, por instrucción explícita.** El objetivo no es aprender a implementar RiskMetrics, sino entender **qué supuestos hacen que funcione** y, sobre todo, **qué se rompe cuando esos supuestos no se cumplen**.

### 6.1 Qué es

**[A]** RiskMetrics es una metodología de cálculo de VaR desarrollada por J. P. Morgan (Longerstaey y More, 1995). En su forma simple, asume que el retorno logarítmico diario de una cartera sigue una **distribución normal condicional**.

### 6.2 Los supuestos, uno por uno

**[A]** Sea $r_t$ el retorno logarítmico diario y $F_{t-1}$ el conjunto de información disponible en $t-1$. RiskMetrics asume:

$$r_t \mid F_{t-1} \sim N(\mu_t, \sigma_t^2)$$

con:

$$\mu_t = 0, \qquad \sigma_t^2 = \alpha\sigma_{t-1}^2 + (1-\alpha)r_{t-1}^2, \qquad 1 > \alpha > 0 \tag{7.2}$$

Los tres supuestos son, entonces:

1. **Normalidad condicional.** La distribución de $r_t$ dado el pasado es normal.
2. **Media condicional cero.** $\mu_t = 0$, siempre.
3. **Un modelo de volatilidad muy específico.** La ecuación de $\sigma_t^2$ es un **IGARCH(1,1) sin drift**: un GARCH integrado donde los coeficientes suman exactamente 1 y no hay constante. **[A]** Tsay lo dice explícitamente: el logaritmo del precio satisface $p_t - p_{t-1} = a_t$ con $a_t = \sigma_t\epsilon_t$ siguiendo un IGARCH(1,1) sin drift. El valor de $\alpha$ suele estar entre 0.9 y 1, con un valor típico de **0.94**.

**Explicación de la ecuación de volatilidad en palabras:** la varianza de hoy es un promedio ponderado entre la varianza de ayer (con peso $\alpha$, grande) y el retorno de ayer al cuadrado (con peso $1-\alpha$, chico). Es un promedio móvil exponencialmente ponderado (EWMA) de los retornos al cuadrado. No tiene término constante, lo cual significa que no hay ningún nivel de varianza "de largo plazo" al que la volatilidad tienda a volver.

### 6.3 VaR a 1 período

**[A]** Con esos supuestos, si la tail probability es 5%, entonces:

$$\mathrm{VaR} = 1.65\,\sigma_{t+1}$$

que es el cuantil superior del 5% (o percentil 95) de una $N(0,\sigma_{t+1}^2)$. Si la tail probability es 1%:

$$\mathrm{VaR} = 2.326\,\sigma_{t+1}$$

**[A]** Nota importante que hace el propio Tsay: como RiskMetrics asume retornos normales con media cero, **la función de pérdida es simétrica y el VaR es el mismo para posiciones long y short**. Esto es una consecuencia directa de los supuestos, no una propiedad de los mercados.

### 6.4 La regla de la raíz cuadrada del tiempo — de dónde sale

**[A]** Bajo el IGARCH(1,1) sin drift de la Ec. (7.2), la distribución condicional del retorno acumulado a $k$ períodos, $r_t[k] = r_{t+1}+\cdots+r_{t+k}$, se obtiene fácilmente. Tsay lo deriva paso a paso:

- Reescribiendo la ecuación de volatilidad: $\sigma_t^2 = \sigma_{t-1}^2 + (1-\alpha)\sigma_{t-1}^2(\epsilon_{t-1}^2-1)$.
- Como $E(\epsilon_{t+i-1}^2 - 1 \mid F_t) = 0$ para $i\ge2$, resulta que $E(\sigma_{t+i}^2\mid F_t) = E(\sigma_{t+i-1}^2\mid F_t)$ para todo $i\ge 2$.
- Es decir: **el pronóstico de varianza es plano hacia adelante**. $\mathrm{Var}(r_{t+i}\mid F_t) = \sigma_{t+1}^2$ para todo $i\ge1$.
- Por lo tanto $\sigma_t^2[k] = k\,\sigma_{t+1}^2$, y $r_t[k]\mid F_t \sim N(0, k\sigma_{t+1}^2)$.

De ahí, la desviación estándar a $k$ períodos es $\sqrt{k}\,\sigma_{t+1}$, y:

$$\mathrm{VaR}(k) = \sqrt{k}\times\mathrm{VaR}$$

**[A]** Esto es lo que se conoce como **square root of time rule** en el cálculo de VaR **bajo RiskMetrics**.

### 6.5 GUARDRAIL OBLIGATORIO: por qué $\sqrt{k}$ NO es una ley universal

$$\boxed{\mathrm{VaR}(k)=\sqrt{k}\,\mathrm{VaR}(1) \text{ es una CONSECUENCIA de supuestos específicos, no una ley de los mercados}}$$

**[A]** Tsay es explícito en la Sección 7.2.1: *"The square root of time rule is a consequence of the special model used by RiskMetrics. If either the zero mean assumption or the special IGARCH(1,1) model assumption of the log returns fails, then the rule is invalid."*

Y lo demuestra con un contraejemplo mínimo. **[A]** Consideremos el mismo modelo pero con media no nula:

$$r_t = \mu + a_t,\quad a_t=\sigma_t\epsilon_t,\quad \mu\neq0,\quad \sigma_t^2=\alpha\sigma_{t-1}^2+(1-\alpha)a_{t-1}^2$$

**[A]** Tsay señala que el supuesto $\mu\neq0$ *"holds for returns of many heavily traded stocks on the NYSE"* (ver Capítulo 1). Con este modelo, la distribución de $r_t[k]$ dado $F_t$ es $N(k\mu, k\sigma_{t+1}^2)$, y el cuantil del 95% usado para el VaR a $k$ períodos es:

$$k\mu + 1.65\sqrt{k}\,\sigma_{t+1} = \sqrt{k}\left(\sqrt{k}\,\mu + 1.65\,\sigma_{t+1}\right)$$

que **no** es $\sqrt{k}$ veces el cuantil de 1 período ($\mu + 1.65\sigma_{t+1}$). **[A]** Conclusión textual: *"Consequently, $\mathrm{VaR}(k)\neq\sqrt{k}\times\mathrm{VaR}$ when the mean return is not zero. It is also easy to show that the rule fails when the volatility model of the return is not an IGARCH(1,1) model without drift."*

**Qué puede romper la regla, en lenguaje simple:**

| Supuesto que falla | Por qué rompe la regla |
|---|---|
| **Media no nula** ($\mu\neq0$) | La media se acumula **linealmente** con $k$, mientras que la desviación estándar se acumula como $\sqrt{k}$. Dos velocidades distintas no pueden resumirse en un solo factor $\sqrt{k}$. |
| **Modelo de volatilidad distinto** | En un GARCH(1,1) estacionario (con $\alpha_1+\beta_1<1$), el pronóstico de varianza **no** es plano: converge hacia la varianza incondicional. La suma de las varianzas a $k$ pasos ya no es $k\sigma_{t+1}^2$. |
| **Dependencia en los retornos** | Si hay autocorrelación (un ARMA en la media), los errores de pronóstico a distintos horizontes se acumulan con pesos $\psi_i$ no triviales (Ec. 7.8 de Tsay), no simplemente sumándose. |
| **Distribución distinta** | El factor multiplicativo del cuantil (1.65, 2.326) es específico de la normal. Con otra distribución, el factor cambia — y puede cambiar de forma distinta según el horizonte. |
| **Colas pesadas / agregación temporal** | **[A]** Bajo teoría de valores extremos, Tsay muestra en la Sección 7.6.2 que la relación correcta es $\mathrm{VaR}(\ell) = \ell^{\xi}\mathrm{VaR}$, la "regla de la raíz $\alpha$ del tiempo", que solo coincide con $\sqrt{\ell}$ si $\xi = 0.5$. |

**Nunca presentar "square-root-of-time" como una regla general de mercados.** En este informe se usa exclusivamente como ejemplo de una regla derivada de supuestos, no como herramienta.

### 6.6 Ejemplos empíricos de Tsay

**[A] Ejemplo 7.1.** Serie: tipo de cambio marco alemán / dólar estadounidense. Frecuencia: diaria (retornos continuamente compuestos). Período de referencia: junio de 1997. Variable: desviación estándar muestral del retorno diario ≈ **0.53%**. Posición: long por 10 millones de dólares.

- VaR al 5%, horizonte 1 día: $10.000.000 \times (1.65\times0.0053) = \mathbf{\$87.450}$.
- VaR al 5%, horizonte 10 días: $10.000.000 \times (\sqrt{10}\times1.65\times0.0053) \approx \mathbf{\$276.541}$.

*Limitación:* este ejemplo aplica directamente la regla $\sqrt{k}$, cuya validez depende de los supuestos discutidos arriba.

**[A] Ejemplo 7.2.** Serie: retornos logarítmicos diarios de la acción de **IBM**. Período: 3 de julio de 1962 – 31 de diciembre de 1998. Tamaño de muestra: **9190 observaciones**. Método: IGARCH(1,1) sin drift, con media condicional asumida cero *para fines de demostración* — **[A]** Tsay aclara explícitamente que *"the sample mean of the returns is significantly different from zero"* (Capítulo 1), y que asume media cero solo para ilustrar RiskMetrics.

Modelo ajustado:

$$r_t = a_t,\quad a_t=\sigma_t\epsilon_t,\quad \sigma_t^2 = 0.9396\,\sigma_{t-1}^2 + (1-0.9396)\,a_{t-1}^2 \tag{7.4}$$

*Limitación reconocida por el propio Tsay:* **"As expected, this model is rejected by the Q statistics"** — el estadístico de Ljung-Box sobre los residuos estandarizados al cuadrado es $Q(10)=56.19$, altamente significativo. Es decir, **el modelo no ajusta bien**, y Tsay lo usa igual, con fines ilustrativos.

Resultados: con $r_{9190}=-0.0128$ y $\hat\sigma_{9190}^2 = 0.0003472$, el pronóstico a 1 paso es $\hat\sigma_{9190}^2(1)=0.000336$. Para una posición long de 10 millones:

- 5% VaR (1 día) = $10.000.000\times 1.65\sqrt{0.000336}$ = $10.000.000\times0.03025$ = **\$302.500**
- 1% VaR (1 día) = $10.000.000\times 2.326\sqrt{0.000336}$ = **\$426.500**

**[A]** Un Remark relevante: usando el comando `ewma1` de S-Plus, el estimado de $\alpha$ es 0.964 (en vez de 0.9396) y el pronóstico de volatilidad a 1 paso es 0.01888, lo que da VaR de **\$311.520** y **\$439.187** respectivamente. Tsay señala que *"these two values are slightly higher than those of Example 7.2, which are based on estimates of the RATS package."*

**[B] Observación metodológica:** el mismo método, sobre los mismos datos, con dos paquetes de software distintos, produce estimaciones de $\alpha$ distintas (0.9396 vs 0.964) y VaR que difieren en aproximadamente 3%. Esto es una primera muestra, muy modesta, de la incertidumbre de estimación que domina todo el capítulo.

### 6.7 Ventajas y desventajas según Tsay

**[A] Ventajas:**
- **Simplicidad.** Es fácil de entender y aplicar.
- **Transparencia.** Hace el riesgo más transparente en los mercados financieros.

**[A] Desventaja principal:**
- *"as security returns tend to have heavy tails (or fat tails), the normality assumption used often results in underestimation of VaR."*

### 6.8 Múltiples posiciones — prioridad baja

**[A]** Para varias posiciones, RiskMetrics adopta un enfoque simple bajo el supuesto de que los retornos logarítmicos diarios de cada posición siguen un IGARCH(1,1) tipo random-walk. Para dos posiciones:

$$\mathrm{VaR}=\sqrt{\mathrm{VaR}_1^2+\mathrm{VaR}_2^2+2\rho_{12}\mathrm{VaR}_1\mathrm{VaR}_2}$$

con generalización directa a $m$ instrumentos. **[A]** La fórmula se obtiene asumiendo que la distribución conjunta de los retornos logarítmicos es **normal multivariada** con media cero.

**No se profundiza aquí**, por instrucción explícita: este proyecto no está estudiando optimización de carteras multiactivo. Lo único que vale registrar [B] es que la fórmula depende críticamente del supuesto de normalidad multivariada, que es considerablemente más fuerte que el de normalidad marginal.

---

## 7. Expected Shortfall

**PRIORIDAD MÁXIMA.** Ésta es una de las secciones conceptualmente más importantes del capítulo para el proyecto.

### 7.1 ¿Qué problema intenta resolver?

El VaR responde:

> **"¿Dónde empieza aproximadamente la zona mala?"**

Pero deja completamente sin responder:

> **"Una vez que entramos en esa zona mala, ¿cuánto perdemos en promedio?"**

**[A]** Tsay plantea el problema exactamente así en la Sección 7.2.3: *"Given a tail probability $p$, VaR is simply the $(1-p)$th quantile of the loss function. In practice, the actual loss, if it occurs, can be greater than VaR. In this sense, VaR may underestimate the actual loss. To have a better assessment of the potential loss, one can consider the expected value of the loss function if the VaR is exceeded. This consideration leads to the concept of expected shortfall (ES)."*

**[A]** Y ya en la Sección 7.1 había anticipado el problema: *"two assets may have the same VaR yet encounter different losses when the VaR is exceeded."*

### 7.2 Ejemplo numérico sencillo

Supongamos que el VaR al 99% es **1.000**. Miramos los días en que efectivamente se superó ese umbral, y las pérdidas fueron:

$$1.100,\quad 1.200,\quad 2.000,\quad 5.000$$

El VaR nos dice que la frontera está en 1.000. No nos dice nada sobre estos cuatro números. El Expected Shortfall intenta resumir precisamente **la severidad de esos casos**: en este ejemplo, el promedio de las pérdidas que superaron el VaR es:

$$\frac{1.100+1.200+2.000+5.000}{4} = 2.325$$

Nótese la magnitud del contraste: el VaR decía "1.000"; el promedio de lo que realmente pasa cuando se cruza ese umbral es más del doble.

### 7.3 Definición

$$\mathrm{VaR}_q = \text{el }q\text{-ésimo cuantil de la distribución de pérdidas}$$

$$\mathrm{ES}_q = E[\,L \mid L > \mathrm{VaR}_q\,]$$

**Explicación de cada símbolo:** $L$ es la pérdida (variable aleatoria); $\mathrm{VaR}_q$ es el umbral; la barra vertical $\mid$ significa "dado que"; $E[\cdot]$ es el valor esperado. En palabras: **el promedio de las pérdidas, restringido a los casos en que la pérdida superó el VaR.**

**[A]** Tsay usa exactamente esta convención en la Ec. (7.38): $\mathrm{ES}_q = E(r\mid r>\mathrm{VaR}_q)$, donde $r$ es la serie de pérdidas (retornos negativos para posición long).

**[A]** Nota terminológica: Tsay señala que el expected shortfall también se conoce como **conditional value at risk (CVaR)**.

### 7.4 Qué información pierde VaR que ES intenta recuperar

| | Qué captura | Qué ignora |
|---|---|---|
| **VaR** | La **ubicación** de la frontera de la zona mala | Todo lo que ocurre más allá de esa frontera: cuán mala es la peor parte |
| **ES** | La **severidad promedio** de lo que ocurre más allá de la frontera | La forma completa de esa cola; sigue siendo un solo número resumen |

**Lo importante:** dos activos pueden tener **exactamente el mismo VaR** y, sin embargo, ES muy distintos, porque uno tiene una cola que se corta rápidamente después del umbral y el otro tiene una cola que sigue estirándose mucho más allá.

### 7.5 Advertencia obligatoria: ES no es "siempre mejor"

**NO se afirma aquí que "ES es mejor que VaR".**

Son **medidas distintas que responden preguntas distintas**:

- VaR responde: ¿dónde está el umbral?
- ES responde: dado que cruzamos el umbral, ¿cuánto en promedio?

ES **incorpora información sobre la severidad más allá del umbral** que VaR, por construcción, no puede incorporar. Esa es una propiedad objetiva. Pero:

- ES requiere estimar el comportamiento **más adentro** de la cola que VaR, es decir, en la región donde hay **aún menos datos**. Todo lo dicho en la Sección 19 sobre incertidumbre de estimación aplica con **mayor** fuerza al ES que al VaR.
- ES depende igualmente del modelo, de la distribución asumida y de la ventana de datos.
- **[A]** Tsay no declara superioridad universal de ES. Lo presenta como *"an alternative to measuring risk"* que ayuda a superar debilidades específicas de VaR, y como una medida útil asociada a un VaR dado.

### 7.6 Cálculo bajo normalidad condicional (RiskMetrics)

**[A]** Bajo RiskMetrics, la función de pérdida es normal, de modo que la distribución condicional de la pérdida **dado que se excedió el VaR** es una **normal truncada por abajo**. Para $X\sim N(0,1)$, dada una probabilidad de cola superior $p$ y $q=1-p$:

$$E(X\mid X>\mathrm{VaR}_q) = \frac{f(\mathrm{VaR}_q)}{p}$$

donde $f(x)=(1/\sqrt{2\pi})\exp(-x^2/2)$ es la densidad de la normal estándar. Entonces, para un retorno con distribución condicional $N(0,\sigma_t^2)$:

$$\mathrm{ES}_q = \frac{f(\mathrm{VaR}_q)}{p}\,\sigma_t$$

**[A] Valores concretos:**
- Si $p=0.05$: $\mathrm{VaR}_{0.95}\approx1.645$ y $f(1.645)/0.05 = 2.0627$, de modo que $\mathrm{ES}_{0.95}=2.0627\,\sigma_t$.
- Si $p=0.01$: $\mathrm{ES}_{0.99}=2.6652\,\sigma_t$.

Comparemos: el VaR al 1% es $2.326\sigma_t$ y el ES correspondiente es $2.665\sigma_t$. **Bajo normalidad**, el ES está aproximadamente 15% por encima del VaR.

**[A]** Bajo normalidad condicional con media no nula (Sección 7.3.2): $\mathrm{ES}_q = \mu_t + \dfrac{f(x_q)}{p}\sigma_t$, con $x_q$ el $q$-ésimo cuantil de la normal estándar; para $p=0.01$, $\mathrm{ES}_{0.99}=\mu_t+2.6652\sigma_t$.

### 7.7 Cálculo empírico

**[A]** El ES también puede estimarse directamente de la muestra. Sea $\hat x_q$ el cuantil empírico $q$-ésimo:

$$\mathrm{ES}_q = \frac{1}{N_q}\sum_{i=1}^{n} x_{(i)}\,I[x_{(i)}>\hat x_q] \tag{7.32 análogo}$$

donde $I[\cdot]$ vale 1 si la condición se cumple y 0 si no, y $N_q$ es el número de observaciones que superan $\hat x_q$. En palabras: **el promedio simple de todas las observaciones que superaron el cuantil**.

**[A] Ejemplo empírico:** serie de retornos logarítmicos diarios **negativos** de IBM (para posición long), 1962–1998, 9190 observaciones, medidos en porcentaje. Con $p=0.01$: $\hat x_{0.99}=3.630$, y por lo tanto $\mathrm{ES}_{0.99}=\mathbf{5.097}$.

**Lectura:** el VaR al 1% dice "3.63%". El promedio de lo que efectivamente pasó los días en que se cruzó ese umbral fue **5.10%** — un 40% peor. **[B]** Nótese que este ES empírico se calcula sobre aproximadamente 92 observaciones (el 1% de 9190), lo cual ya es una muestra pequeña para estimar un promedio de valores muy dispersos.

### 7.8 Cálculo bajo GPD

**[A]** Para la distribución generalizada de Pareto (ver Sección 20 de este informe), el ES tiene forma cerrada. Definiendo $\mathrm{ES}_q = \mathrm{VaR}_q + E(r-\mathrm{VaR}_q\mid r>\mathrm{VaR}_q)$ y usando las propiedades de la GPD:

$$E(r-\mathrm{VaR}_q\mid r>\mathrm{VaR}_q) = \frac{\psi(\eta)+\xi(\mathrm{VaR}_q-\eta)}{1-\xi}$$

**siempre que $0<\xi<1$**, de modo que:

$$\mathrm{ES}_q = \frac{\mathrm{VaR}_q}{1-\xi} + \frac{\psi(\eta)-\xi\eta}{1-\xi}$$

**Nota importante [A]:** la condición $\xi<1$ no es un tecnicismo. Si $\xi\ge1$, la media de la distribución de excesos **no existe** (es infinita), y por lo tanto el Expected Shortfall no está definido. Es decir: hay distribuciones con colas suficientemente pesadas para las cuales la pregunta "¿cuánto perdemos en promedio dado que superamos el umbral?" **no tiene respuesta finita**.

**[A] Resultado empírico (IBM, negativos de log returns, threshold 2.5%, 1962–1998):** el comando `riskmeasures` produce:

| $p$ (nivel $q$) | VaR (cuantil) | ES (shortfall) |
|---|---|---|
| 0.95 | 0.02209 | 0.03163 |
| 0.99 | 0.03616 | 0.05075 |
| 0.999 | 0.07019 | 0.09700 |

Para una posición de 10 millones: VaR de **\$220.889** y **\$361.661** para tail probabilities de 0.05 y 0.01; ES de **\$316.272** y **\$507.576** respectivamente.

**Observación [B]:** el ES es sistemáticamente 38–43% mayor que el VaR correspondiente en este ejemplo (43% al 5%, 40% al 1%, 38% al 0.1%). Y la brecha se mantiene relativamente estable a los tres niveles, lo cual es una consecuencia de la forma funcional de la GPD, no un hecho empírico independiente.

### 7.9 Relevancia posible para futuros / ML [B]

**PREGUNTA ABIERTA.** El ES no se adopta como target ni como métrica. Lo conceptualmente valioso es la distinción de objetos:

$$P(Y<c\mid X) \quad\text{vs.}\quad E[Y\mid Y<c, X]$$

"¿Qué probabilidad hay de entrar en la cola?" es una pregunta distinta de "si entramos, ¿qué tan severo puede ser?". Un modelo puede tener información sobre una y no sobre la otra. Ambas son distintas de $E[Y\mid X]$ y de $\mathrm{Var}(Y\mid X)$.

**Expected Shortfall ≠ target recomendado.**

---

## 8. Econometric VaR y conexión con GARCH

**Estudio serio pero selectivo.** No reproducimos toda la estimación; el objetivo es entender **una idea central**.

### 8.1 La idea central

$$\boxed{\text{el VaR puede ser CONDICIONAL al estado actual del mercado, no un número histórico fijo}}$$

**[A]** Tsay lo formula así: un enfoque general al cálculo de VaR es usar los modelos econométricos de series de tiempo de los Capítulos 2 a 4. Los modelos del Capítulo 2 modelan la ecuación de la media, y los modelos de heteroscedasticidad condicional del Capítulo 3 (o los no lineales del Capítulo 4) manejan la volatilidad.

### 8.2 La estructura

**[A]** Un modelo general para $r_t$:

$$r_t = \phi_0 + \sum_{i=1}^{p}\phi_i r_{t-i} + a_t - \sum_{j=1}^{q}\theta_j a_{t-j} \tag{7.5}$$

$$a_t = \sigma_t\epsilon_t, \qquad \sigma_t^2 = \alpha_0+\sum_{i=1}^{u}\alpha_i a_{t-i}^2 + \sum_{j=1}^{v}\beta_j\sigma_{t-j}^2 \tag{7.6}$$

Es decir: **ARMA para la media, GARCH para la volatilidad** — exactamente el aparato del Capítulo 3.

Estas dos ecuaciones producen pronósticos a 1 paso de la media condicional $\hat r_t(1)$ y de la varianza condicional $\hat\sigma_t^2(1)$.

### 8.3 De dónde sale el cuantil

Aquí está el punto conceptual clave, y conecta directamente con el Capítulo 3:

$$a_t = \sigma_t\epsilon_t$$

La distribución futura de $r_{t+1}$ dado $F_t$ depende de **dos cosas separadas**:

1. **El pronóstico de $\sigma_t$** — cuánta escala esperamos.
2. **La distribución que asumimos para $\epsilon_t$** — qué forma tiene el shock estandarizado.

**[A] Si $\epsilon_t$ es gaussiano**, la distribución condicional de $r_{t+1}$ es $N[\hat r_t(1), \hat\sigma_t^2(1)]$, y el cuantil del 95% es:

$$\hat r_t(1)+1.65\,\hat\sigma_t(1)$$

**[A] Si $\epsilon_t$ es una Student-t estandarizada con $v$ grados de libertad**, el cuantil es:

$$\hat r_t(1) + \frac{t_v(1-p)}{\sqrt{v/(v-2)}}\,\hat\sigma_t(1)$$

donde $t_v(1-p)$ es el cuantil $(1-p)$ de una Student-t con $v$ grados de libertad (no estandarizada). **[A]** Tsay deriva explícitamente la relación entre los cuantiles de una Student-t y los de su versión estandarizada: si $q$ es el cuantil $p$ de una $t_v$, entonces $q/\sqrt{v/(v-2)}$ es el cuantil $p$ de la $t_v$ estandarizada (requiere $v>2$).

**El mismo pronóstico de volatilidad, con dos supuestos distintos sobre $\epsilon_t$, produce dos cuantiles de riesgo distintos.**

### 8.4 Ejemplo empírico de Tsay: Ejemplo 7.3

**[A] Serie:** retornos logarítmicos diarios de IBM. **Período:** 3 jul 1962 – 31 dic 1998. **Muestra:** 9190. **Variable analizada:** $r_t = -r_t^c$ (negativos de los log returns, porque la posición es **long**). **Posición:** long de \$10 millones, horizonte 1 día, origen $t=9190$.

**CASO 1 — $\epsilon_t$ normal estándar.** Modelo ajustado:

$$r_t = -0.00066 - 0.0247\,r_{t-2}+a_t,\qquad \sigma_t^2 = 0.00000389+0.0799\,a_{t-1}^2+0.9073\,\sigma_{t-1}^2$$

Con $r_{9189}=0.00201$, $r_{9190}=0.0128$, $\sigma_{9190}^2=0.00033455$, los pronósticos son $\hat r_{9190}(1)=-0.00071$ y $\hat\sigma_{9190}^2(1)=0.0003211$.

- Cuantil 95%: $-0.00071+1.6449\sqrt{0.0003211}=0.02877$ → **VaR(5%) = \$287.700**
- Cuantil 99%: $-0.00071+2.3262\sqrt{0.0003211}=0.0409738$ → **VaR(1%) = \$409.738**

**CASO 2 — $\epsilon_t$ Student-t estandarizada con 5 grados de libertad.** Modelo ajustado:

$$r_t=-0.0003-0.0335\,r_{t-2}+a_t,\qquad \sigma_t^2=0.000003+0.0559\,a_{t-1}^2+0.9350\,\sigma_{t-1}^2$$

Pronósticos: $\hat r_{9190}(1)=-0.000367$, $\hat\sigma_{9190}^2(1)=0.0003386$.

- Cuantil 95%: el percentil 95 de una $t_5$ es 2.015, y de su versión estandarizada es $2.015/\sqrt{5/3}=1.5608$. Entonces $-0.000367+1.5608\sqrt{0.0003386}=0.028354$ → **VaR(5%) = \$283.520**
- Cuantil 99%: $-0.000367+(3.3649/\sqrt{5/3})\sqrt{0.0003386}=0.0475943$ → **VaR(1%) = \$475.943**

**[A] Interpretación textual de Tsay:** al 5%, el VaR con Student-t es *"essentially the same"* que bajo normalidad. Al 1%, en cambio, *"we see the heavy-tail effect of using a Student-t distribution with 5 degrees of freedom; it increases the VaR when the tail probability becomes smaller."*

**Lección [B]:** el efecto de la elección de distribución **crece a medida que nos adentramos en la cola**. Cerca del centro (5%) casi no importa; en la cola (1%, 0.1%) es determinante. Esto es coherente con todo el resto del capítulo.

### 8.5 Heavy tail vs. normal VaR — sección obligatoria

**[A]** Tsay señala en la Sección 7.2.1 que la normalidad *"often results in underestimation of VaR"* cuando los retornos tienen colas pesadas.

**Explicación desde cero.** Supongamos que la distribución normal, ajustada a la volatilidad observada, dice:

> "Un movimiento de −5% en un día debería tener una probabilidad de aproximadamente 1 en 3 millones. Es prácticamente imposible."

Y los datos reales dicen:

> "Movimientos de esa magnitud ocurrieron 4 veces en 9190 días."

Cuando esto pasa, un VaR calculado bajo normalidad queda **demasiado cerca del centro**: el modelo pone la frontera de la "zona mala" en un lugar donde en realidad todavía hay bastante masa de probabilidad más allá.

**Guardrail obligatorio:**

$$\text{NO decir: "el VaR normal siempre subestima el riesgo"}$$

Esa afirmación es demasiado fuerte. Lo correcto es:

> Cuando la distribución real tiene colas más pesadas que la normal, un VaR calculado bajo normalidad tenderá a quedar por debajo del valor correcto, **y ese efecto es más pronunciado cuanto menor sea la tail probability**. Si la distribución real no tuviera colas más pesadas que la normal, o si el efecto de la volatilidad condicional ya absorbe buena parte de la aparente pesadez de colas de la distribución incondicional, la conclusión no aplica automáticamente.

**[A]** El propio capítulo lo ilustra: en el Ejemplo 7.3 al 5%, el VaR bajo Student-t es **ligeramente menor** que bajo normalidad (\$283.520 vs \$287.700). Es decir, ni siquiera es cierto que la distribución de colas pesadas siempre dé un VaR mayor: **depende del nivel de la cola que se esté mirando.**

### 8.6 Conexión conceptual: incondicional vs. condicional

**[A]** Éste es el punto que Tsay anticipa en la introducción del capítulo: *"Both unconditional and conditional concepts of extreme values are discussed. The unconditional approach to VaR calculation for a financial position uses the historical returns of the instruments involved to compute VaR. On the other hand, a conditional approach uses the historical data and explanatory variables to calculate VaR."*

**Lo que esto significa en la práctica [B]:**

- El **cuantil empírico** (Sección 11) produce un número **fijo**: el percentil de toda la historia. Es el mismo hoy que mañana, hasta que se agreguen datos nuevos.
- El **VaR econométrico** produce un número que **cambia todos los días**, porque $\hat\sigma_t(1)$ cambia todos los días. En un día de alta volatilidad, el VaR se agranda; en un día tranquilo, se achica.

Ésta es la primera aparición explícita de la distinción **cuantil incondicional vs. cuantil condicional**, que se desarrolla en profundidad en la Sección 13 y es el puente central hacia ML.

---

## 9. Multiperiod VaR y square-root-of-time

**Estudio conceptual.**

### 9.1 La distinción básica

- **1-step (un paso):** ¿qué puede pasar de acá al próximo período?
- **$h$-step ($h$ pasos):** ¿qué puede pasar de acá a $h$ períodos adelante, considerando el resultado **acumulado**?

No son el mismo problema. El segundo requiere acumular incertidumbre a lo largo de varios pasos, y esa acumulación depende de la estructura del modelo.

### 9.2 Por qué no basta con multiplicar por $\sqrt{h}$

**[A]** Tsay lo desarrolla en la Sección 7.3.1. La variable de interés es el retorno acumulado $r_h[k]=r_{h+1}+\cdots+r_{h+k}$. Su media condicional se obtiene sumando los pronósticos individuales:

$$\hat r_h[k]=r_h(1)+\cdots+r_h(k)$$

Y el error de pronóstico acumulado, usando la representación MA del modelo ARMA, es:

$$e_h[k]=a_{h+k}+(1+\psi_1)a_{h+k-1}+\cdots+\left(\sum_{i=0}^{k-1}\psi_i\right)a_{h+1} \tag{7.7}$$

**Explicación de la fórmula en palabras:** los shocks más cercanos al origen del pronóstico (los $a_{h+1}$) entran con un peso **mayor** —la suma acumulada de todos los coeficientes $\psi$—, porque su efecto se propaga a través de todos los períodos siguientes. Los shocks más lejanos entran con peso 1. **Los shocks no se suman todos con el mismo peso.**

La varianza correspondiente es:

$$V_h(e_h[k])=\sigma_h^2(k)+(1+\psi_1)^2\sigma_h^2(k-1)+\cdots+\left(\sum_{i=0}^{k-1}\psi_i\right)^2\sigma_h^2(1) \tag{7.8}$$

**Las tres razones por las que $\sqrt{h}$ falla, en lenguaje simple:**

1. **Los pesos $\psi_i$ no son todos iguales a cero.** Si hay estructura ARMA en la media, los shocks se propagan y se acumulan con pesos distintos.
2. **Los pronósticos de volatilidad a distintos horizontes no son iguales.** En un GARCH(1,1) estacionario, $\sigma_h^2(\ell)$ converge hacia la varianza incondicional a medida que $\ell$ crece — no se mantiene constante como en el IGARCH de RiskMetrics.
3. **La media se acumula linealmente.** Si $\mu\neq0$, la media a $k$ períodos es $k\mu$, mientras que la desviación estándar crece más lentamente.

**[A]** Para el caso simple donde $\psi_i=0$ para todo $i>0$ (sin estructura ARMA), y con GARCH(1,1), Tsay deriva la fórmula cerrada:

$$\mathrm{Var}(e_h[k]\mid F_h)=\frac{\alpha_0}{1-\phi}\left(k-\frac{1-\phi^k}{1-\phi}\right)+\frac{1-\phi^k}{1-\phi}\,\sigma_h^2(1) \tag{7.10}$$

donde $\phi=\alpha_1+\beta_1<1$ es la persistencia del GARCH.

**[A]** Y señala: si la distribución condicional de $a_t$ no es gaussiana (Student-t, error generalizado), *"simulation can be used to obtain the multiperiod VaR"* — es decir, no hay forma cerrada.

### 9.3 Ejemplo empírico de Tsay

**[A] Serie:** IBM, retornos logarítmicos diarios, 1962–1998, 9190 obs. **Modelo:** AR(2)–GARCH(1,1) gaussiano del Ejemplo 7.3. **Horizonte:** 15 días, desde el origen 9190 (31 de diciembre de 1998). **Posición:** long \$10 millones.

- Media condicional del retorno acumulado a 15 días: $-0.00998$
- Varianza condicional: $0.0047948$ (obtenida por la recursión de la Ec. 7.9)
- Cuantil 95%: $-0.00998+1.6449\sqrt{0.0047948}=0.1039191$
- **VaR(5%, 15 días) = \$1.039.191**

**Comparación con la regla $\sqrt{k}$:** $\$287.700 \times \sqrt{15} = \$1.114.257$.

**[A] Conclusión textual de Tsay:** *"This amount is smaller than \$287,700 × √15 = \$1,114,257. This example further demonstrates that the square root of time rule used by RiskMetrics holds only for the special white noise IGARCH(1,1) model used. When the conditional mean is not zero, proper steps must be taken to compute the k-horizon VaR."*

**Magnitud del error:** la regla $\sqrt{k}$ **sobreestima** en aproximadamente 7% en este caso concreto. **[B]** Nótese que sobreestima, no subestima — el signo del error depende de la estructura del modelo, no es predecible sin hacer el cálculo.

### 9.4 La regla alternativa bajo EVT

**[A]** En la Sección 7.6.2, Tsay presenta la relación correcta bajo teoría de valores extremos (Danielsson y de Vries, 1997a):

$$\mathrm{VaR}(\ell)=\ell^{1/\alpha}\,\mathrm{VaR}=\ell^{\xi}\,\mathrm{VaR}$$

donde $\alpha=1/\xi$ es el **tail index** y $\xi$ el parámetro de forma. **[A]** Esto se conoce como la **"$\alpha$ root of time rule"**. Nótese que aquí $\alpha$ es el tail index, **no** el parámetro de escala $\alpha_n$ de la GEV — una colisión de notación desafortunada que Tsay señala explícitamente.

**[A] Ejemplo:** IBM, $p=0.01$, resultados con $n=63$ ($\xi=0.335$), horizonte de 30 días:

$$\mathrm{VaR}(30)=30^{0.335}\times\mathrm{VaR}=3.125\times\$304.969=\$952.997$$

**[A]** Y observa: *"Because 0.335 < 0.5, the α root of time rule produces lower ℓ-day horizon VaR than the square root of time rule does."*

Comparemos: $\sqrt{30}=5.48$ vs $30^{0.335}=3.125$. **La diferencia de factor es del 75%.** Escalar temporalmente el riesgo con la regla incorrecta produce errores de esta magnitud.

### 9.5 Guardrail

$$\boxed{\text{NO adoptar ninguna regla de escalamiento temporal para barras de futuros}}$$

Este informe registra dos reglas ($\sqrt{\ell}$ y $\ell^\xi$), ambas derivadas bajo supuestos específicos y **ninguna de ellas validada** para datos de futuros intradiarios. La lección transferible no es "usar $\ell^\xi$"; es que **el escalamiento temporal del riesgo es una pregunta empírica, no una constante universal**. Se registra como hipótesis H7.14 en la Sección 30.

---

## 10. Quantiles y order statistics

**PRIORIDAD MÁXIMA.** Aquí cambia parcialmente el foco del capítulo.

### 10.1 ¿Qué problema intenta resolver?

Todo lo visto hasta ahora requería **especificar una distribución completa**: normal (RiskMetrics), normal o Student-t condicional (econométrico). Pero:

> **¿Podemos estimar directamente un percentil de la distribución sin especificar toda la distribución?**

**[A]** Tsay lo plantea así: *"Quantile estimation provides a nonparametric approach to VaR calculation. It makes no specific distributional assumption on the return of a portfolio except that the distribution continues to hold within the prediction period."*

Nótese ese "except": no hay supuesto sobre la **forma** de la distribución, pero sí hay un supuesto —y es fuerte— sobre su **estabilidad en el tiempo**. Volveremos a él en la Sección 11.

**[A]** Tsay distingue dos tipos de método de cuantil: el **cuantil empírico directo** (7.4.1) y la **quantile regression** (7.4.2).

### 10.2 Quantile / percentile

**Intuición:** un cuantil es simplemente un **punto de corte** en la distribución que deja una fracción determinada de la probabilidad por debajo.

**Ejemplo numérico:** el cuantil 0.05 (o percentil 5) de una serie de retornos es el valor tal que el 5% de los retornos quedan por debajo de él.

**Definición técnica [A]:** para una CDF $F(x)$ y una probabilidad $q$ con $0<q<1$:

$$x_q = \inf\{x\mid F(x)\ge q\}$$

donde $\inf$ ("ínfimo") es el **menor número real** $x$ que satisface $F(x)\ge q$. Si la variable es continua, esto se simplifica a $q=\Pr[L\le x_q]$.

**Terminología:** "cuantil" y "percentil" son esencialmente lo mismo, expresado en distintas unidades. El cuantil 0.05 es el percentil 5. Es solo una convención de si se escribe la fracción o el porcentaje.

### 10.3 Order statistics (estadísticos de orden)

**¿Qué problema resuelven?** Necesitamos una forma sistemática de referirnos a "la observación más chica", "la segunda más chica", etc.

**Intuición:** ordenamos todas las observaciones de menor a mayor y les damos un índice según su posición en ese orden.

**Definición técnica [A]:** sean $r_1,\dots,r_n$ los retornos de una cartera en el período muestral. Los **order statistics** de la muestra son esos mismos valores ordenados de forma creciente:

$$r_{(1)}\le r_{(2)}\le\cdots\le r_{(n)}$$

**[A]** En particular, $r_{(1)}$ es el **mínimo muestral** y $r_{(n)}$ el **máximo muestral**.

**Ejemplo numérico:** con los retornos $\{-2.0, +0.5, -0.3, +1.2, -3.5\}$, los order statistics son:

$$r_{(1)}=-3.5,\quad r_{(2)}=-2.0,\quad r_{(3)}=-0.3,\quad r_{(4)}=+0.5,\quad r_{(5)}=+1.2$$

**Qué NO son:** los order statistics no son los datos "en orden cronológico". Se pierde deliberadamente toda la información temporal al ordenarlos. Ésta es una observación importante y volverá en la Sección 25 (extremal index): el ordenamiento por magnitud destruye la información sobre si los extremos estaban agrupados en el tiempo o dispersos.

### 10.4 Empirical quantile (cuantil empírico)

**Intuición:** si queremos el percentil 5% y tenemos 100 observaciones, miramos alrededor de la 5ª observación más baja.

**Ejemplo extremadamente sencillo:** 100 retornos ordenados. El percentil 5% está cerca de las 5 observaciones más bajas: concretamente, $r_{(5)}$ es una estimación natural del cuantil 0.05.

**[A] Resultado asintótico** (Cox y Hinkley, 1974): asumiendo que los retornos son iid con distribución continua de densidad $f(x)$ y CDF $F(x)$, y siendo $x_p$ el cuantil $p$-ésimo con $f(x_p)\neq0$, entonces el order statistic $r_{(\ell)}$ con $\ell=np$ es asintóticamente normal:

$$r_{(\ell)}\sim N\left(x_p,\ \frac{p(1-p)}{n[f(x_p)]^2}\right),\qquad \ell=np \tag{7.11}$$

**Explicación de cada símbolo:** $n$ es el tamaño de muestra; $p$ es la probabilidad del cuantil que estimamos; $x_p$ es el cuantil verdadero (desconocido); $f(x_p)$ es la **densidad evaluada en ese cuantil**.

**Esta fórmula es extremadamente informativa, pero debe interpretarse con cuidado.** La varianza asintótica del estimador es:

$$\frac{p(1-p)}{n\,[f(x_p)]^2}$$

Su precisión depende conjuntamente de tres elementos:

- $n$: el tamaño de muestra; en general, más observaciones reducen la varianza.
- $p(1-p)$: cambia con el cuantil que queremos estimar.
- $f(x_p)$: la densidad de la distribución precisamente en el cuantil de interés.

En regiones muy extremas de una distribución, $f(x_p)$ suele ser pequeña, lo que puede aumentar fuertemente la incertidumbre de la estimación. Sin embargo, **no debe interpretarse esta fórmula como una ley universal según la cual avanzar siempre hacia una cola más profunda aumenta monótonamente la varianza**, porque $p(1-p)$ y $f(x_p)$ cambian simultáneamente y su comportamiento exacto depende de la distribución.

**[A]** La conclusión práctica que sí enfatiza Tsay es que, cuando la probabilidad de cola es muy pequeña, el cuantil empírico se vuelve una estimación menos fiable del cuantil teórico.

$$\boxed{\text{cuantiles muy extremos} \;\Rightarrow\; \text{pocas observaciones relevantes y potencialmente gran incertidumbre}}$$

### 10.5 Interpolación cuando $np$ no es entero

**[A]** En la práctica, $np$ rara vez es un entero. Sean $\ell_1$ y $\ell_2$ los dos enteros vecinos tales que $\ell_1<np<\ell_2$, y definamos $p_i=\ell_i/n$. Entonces:

$$\hat x_p = \frac{p_2-p}{p_2-p_1}\,r_{(\ell_1)}+\frac{p-p_1}{p_2-p_1}\,r_{(\ell_2)} \tag{7.12}$$

En palabras: **una interpolación lineal simple** entre los dos order statistics vecinos, con pesos proporcionales a la cercanía.

### 10.6 Ejemplos empíricos de Tsay

**[A] Ejemplo 7.4.** Serie: retornos logarítmicos diarios de **Intel**. Período: 15 de diciembre de 1972 – 31 de diciembre de 2008. Muestra: **9096 observaciones**. Variable: negativos de los log returns (posición long). Método: cuantil empírico con interpolación.

Como $9096\times0.95 = 8641.2$, tenemos $\ell_1=8641$, $\ell_2=8642$:

$$\hat x_{0.95}=0.8\,r_{(8641)}+0.2\,r_{(8642)}=\mathbf{4.2952\%}$$

con $r_{(8641)}=4.2951\%$ y $r_{(8642)}=4.2954\%$.

**[A] Ejemplo 7.5.** Serie: IBM, log returns diarios, 1962–1998, 9190 obs. Variable: negativos de log returns.

- **Cuantil 95%:** $np=9190\times0.95=8730.5$, de modo que $\hat x_{0.95}=(r_{(8730)}+r_{(8731)})/2 = 0.021603$. Para una posición long de \$10 millones: **VaR = \$216.030**. **[A]** Tsay observa que esto es *"much smaller than those obtained by the econometric approach discussed before"* (\$287.700 y \$283.520).
- **Cuantil 99%:** con interpolación, $\hat x_{0.99}\approx 3.630$ (en porcentaje) → **VaR ≈ \$363.000**. *(Nota: en la tabla comparativa de la Sección 7.6.1, Tsay reporta \$365.709 para este mismo método al 1%. El texto del capítulo no explica la discrepancia entre ambos valores, por lo que se registra como una **inconsistencia no resuelta** del libro y no se atribuye a una causa específica. **[B]** Esta diferencia también sirve como recordatorio de que la precisión aparente de los números reportados no debe confundirse con certeza estadística.)*

**[A]** Y observa nuevamente: *"Again this amount is lower than those obtained before by other methods."*

---

## 11. Empirical quantiles — ventajas y limitaciones

### 11.1 Ventajas según Tsay

**[A]** *"Advantages of using the empirical quantile method to VaR calculation include (a) simplicity and (b) using no specific distributional assumption."*

Son ventajas reales y no menores: no hay que elegir entre normal y Student-t, no hay que ajustar un GARCH, no hay que estimar nada por máxima verosimilitud. Se ordenan los datos y se lee un número.

### 11.2 Las cuatro limitaciones

**[A]** Tsay enumera tres explícitamente; la cuarta aparece en la discusión de la Sección 7.4.2.

**1. Supone que la distribución no cambia entre el período muestral y el de predicción.**

**[A]** *"First, it assumes that the distribution of the return $r_t$ remains unchanged from the sample period to the prediction period."*

Éste es el supuesto que reemplaza al supuesto de forma distribucional. No es más débil; es de otro tipo. En un mercado donde la volatilidad cambia (Capítulo 3), asumir que la distribución **incondicional** de toda la historia describe adecuadamente la distribución de mañana es un supuesto fuerte.

**2. No extrapola más allá de lo observado.**

**[A]** *"Given that VaR is concerned mainly with tail probability, this assumption implies that the predicted loss cannot be greater than that of the historical loss. It is definitely not so in practice."*

Ésta es, quizás, la limitación más importante conceptualmente. **El cuantil empírico está acotado, por construcción, por el peor evento observado en la muestra.** Nunca puede predecir una pérdida mayor que la peor pérdida histórica, porque literalmente no tiene con qué construirla. Y sin embargo, ninguna serie histórica contiene el peor evento posible.

**3. Es ineficiente cuando $p$ es pequeño.**

**[A]** *"Second, when the tail probability $p$ is small, the empirical quantile is not an efficient estimate of the theoretical quantile."*

Es exactamente lo que dice la Ec. (7.11): la varianza del estimador explota cuando la densidad en el cuantil es pequeña.

**4. No incorpora variables explicativas.**

**[A]** *"Third, the direct quantile estimation fails to take into account the effect of explanatory variables that are relevant to the portfolio under study."*

**[A] Conclusión práctica de Tsay:** *"In real application, VaR obtained by the empirical quantile can serve as a lower bound for the actual VaR."*

**[A] Matiz importante**, añadido más adelante en la Sección 7.6.1: esta interpretación de "cota inferior" solo es razonable cuando la muestra es grande **relativa a la probabilidad de interés**. Con 9190 observaciones, los cuantiles empíricos de 5% y 1% son *"decent estimates"*. Pero: *"When the tail probability is small (e.g., 0.1%), the empirical quantile is a less reliable estimate of the true quantile. The VaR based on empirical quantiles can no longer serve as a lower bound of the true VaR."*

### 11.3 "Muchos datos" no significa "muchos extremos" — SECCIÓN ESPECIAL

Ésta es una de las secciones más importantes del informe, y su contenido es en gran medida **[B]** (una elaboración propia sobre las limitaciones que Tsay señala).

**El razonamiento aritmético.**

Supongamos que tenemos **100.000 observaciones**. Suena a una cantidad enorme de datos. Ahora preguntemos cuántas observaciones tenemos efectivamente disponibles para estimar distintas regiones de la cola:

| Región de la cola | Observaciones nominales esperadas |
|---|---|
| 5% | 5.000 |
| 1% | 1.000 |
| 0.1% | **100** |
| 0.01% | **10** |
| 0.001% | **1** |

**Con 100.000 observaciones, una cola de 0.1% se apoya en aproximadamente 100 observaciones.** No 100.000. Cien.

Y la cosa empeora en dos direcciones:

**Primera: el peor evento.** Para estimar bien el comportamiento **dentro** de la cola de 0.1% —es decir, para responder preguntas como "¿cuánto peor puede ser?"— esas 100 observaciones son la muestra completa. El "peor" de esos 100 valores es un solo dato.

**Segunda, y más grave: la dependencia.** **[B]** Si los extremos están **agrupados en el tiempo** (lo que el capítulo llama *clustering of extremes*, ver Sección 25), la cantidad de **información efectiva** es todavía menor que el conteo nominal. Si de esas 100 excedencias, 60 vienen en 15 rachas de 4 días consecutivos y las otras 40 están dispersas, no tenemos 100 eventos independientes; tenemos algo más cercano a 55 episodios de mercado distintos.

**[A]** El extremal index $\theta$ de la Sección 7.8 formaliza precisamente esta idea: bajo condiciones específicas, y para un estimador concreto ($\hat\theta_b^{(2)}$), $1/\theta$ se interpreta como el tamaño **medio** de cluster del proceso límite. **[B]** Aplicar esa aritmética al $\hat\theta_b^{(1)}\approx0.823$ que Tsay estima para IBM sería un abuso de la interpretación —es otro estimador, y el resultado es asintótico—, de modo que **aquí no se calcula ningún tamaño de cluster para IBM**. Lo único que se retiene es la dirección del efecto: $\hat\theta<1$ indica que los extremos no se comportan como si fueran independientes, y por lo tanto que el conteo nominal de excedencias sobrestima la información efectiva. (Ver la Sección 25.6 para las condiciones exactas de esta interpretación y las advertencias correspondientes.)

$$\boxed{\text{muestra total enorme} \nRightarrow \text{muestra enorme de eventos extremos}}$$

**[B] Consecuencia directa para datos intradiarios:** un dataset de barras de 1 minuto de varios años puede tener millones de filas. Es tentador concluir que "con tantos datos, la estimación de colas ya no es un problema". Ese razonamiento es incorrecto por tres motivos acumulativos:

1. La aritmética anterior aplica igual: la cola de 0.01% de 2 millones de barras son 200 barras.
2. Los extremos intradiarios podrían estar **fuertemente agrupados** (varias barras extremas consecutivas dentro del mismo episodio de mercado), reduciendo la información efectiva.
3. Un número no despreciable de esos "extremos" podría no ser información económica en absoluto (ver Sección 26.5 sobre microestructura y errores de datos, conectando con el Capítulo 5).

**Ninguna de estas tres afirmaciones se verifica aquí para MNQ.** Son hipótesis (H7.13, H7.8).

### 11.4 Ejemplo de ES empírico y su fragilidad [B]

Ya vimos que para IBM, $\mathrm{ES}_{0.99}=5.097$ se calcula promediando las observaciones que superan $\hat x_{0.99}=3.630$. Con 9190 observaciones, eso son aproximadamente 92 valores.

**[B]** Promediar 92 valores muy dispersos (que van desde poco más de 3.63 hasta el crash de octubre de 1987, que en la Figura 7.3 de Tsay aparece como un extremo de magnitud superior a 20%) produce un promedio dominado por unos pocos valores. **Si se quitara la observación del crash de 1987, el ES cambiaría notablemente.** Ésta es la fragilidad característica de cualquier estimador de cola: unas pocas observaciones tienen influencia desproporcionada.

Esto no invalida el ES; simplemente indica que el número debe leerse con una banda de incertidumbre amplia, aunque el software reporte seis decimales.
---

## 12. Quantile regression

**PRIORIDAD MÁXIMA.**

### 12.1 ¿Qué problema intenta resolver?

El cuantil empírico tiene una limitación que Tsay señala explícitamente: **no puede incorporar variables explicativas**. Da un solo número, calculado sobre todos los datos mezclados.

Pero muchas veces tenemos información relevante disponible. **[A]** Tsay lo motiva con un ejemplo concreto: *"the action taken by Federal Reserve Banks on interest rates could have important impacts on the returns of U.S. stocks. It is then more appropriate to consider the distribution function $r_{t+1}\mid F_t$, where $F_t$ includes the explanatory variables. In other words, we are interested in the quantiles of the distribution function of $r_{t+1}$ given $F_t$."*

**[A]** Un cuantil de ese tipo se llama, en la literatura, un **regression quantile** (Koenker y Bassett, 1978).

### 12.2 La diferencia con la regresión tradicional

**Regresión con pérdida cuadrática (OLS y, más generalmente, modelos de regresión entrenados con MSE, bajo las condiciones correspondientes):**

$$E[Y\mid X]$$

> "Dadas estas condiciones, ¿cuál es el valor **promedio** esperado del resultado?"

**Quantile regression:**

$$Q_\tau(Y\mid X)$$

> "Dadas estas condiciones, ¿dónde está el **percentil $\tau$** del resultado?"

**Por qué esto es genuinamente distinto, con un ejemplo [B]:**

Supongamos un conjunto de features $X$ que describen el estado del mercado. Podría ocurrir que:

- $E[Y\mid X]$ sea prácticamente constante en 0 para todos los valores de $X$ — es decir, las features no dicen nada sobre la dirección esperada.
- Y sin embargo, $Q_{0.10}(Y\mid X)$ varíe fuertemente con $X$: en unas condiciones el percentil 10 está en −0.3%, y en otras en −1.8%.

**Las features no dicen nada sobre el promedio y dicen mucho sobre la cola inferior.** No hay ninguna contradicción en esto; son propiedades distintas de la misma distribución condicional.

### 12.3 La función objetivo — primero en palabras

Antes de la fórmula: ¿cómo se le enseña a un modelo a estimar un cuantil en vez de una media?

**La respuesta es: cambiando la función de pérdida a una asimétrica.**

**Intuición.** Si queremos estimar la **media**, penalizamos los errores de forma simétrica (típicamente al cuadrado): equivocarse 3 unidades por arriba cuesta lo mismo que equivocarse 3 unidades por abajo. El valor que minimiza esa penalización es la media.

Si queremos estimar la **mediana** (el cuantil 0.5), penalizamos el valor absoluto del error, también simétricamente. El valor que minimiza eso es la mediana.

Y si queremos estimar el **cuantil 0.10**, penalizamos de forma **asimétrica**: quedarse corto y quedarse largo tienen costos diferentes. Concretamente, para un cuantil bajo, quedarse **por encima** del valor real debe penalizarse más que quedarse por debajo, de modo que la solución óptima se "empuje" hacia abajo, hasta el punto exacto donde el 10% de las observaciones queda por debajo.

**En una frase:**

> Equivocarse por arriba y por abajo tiene penalizaciones diferentes, y la asimetría exacta de esa penalización determina qué cuantil termina estimando el modelo.

### 12.4 La fórmula

**[A]** Tsay presenta primero el cuantil empírico **como un problema de optimización**, lo cual es exactamente el puente que necesitamos. Para una probabilidad dada $p$, el cuantil $p$-ésimo de $\{r_t\}$ se obtiene como:

$$\hat x_p = \arg\min_\beta \sum_{i=1}^{n} w_p(r_i-\beta)$$

donde $w_p(z)$ se define como:

$$w_p(z)=\begin{cases} p\,z & \text{si } z\ge0\\ (p-1)\,z & \text{si } z<0\end{cases}$$

**Explicación de cada símbolo:**

- $r_i$: la observación $i$-ésima.
- $\beta$: el candidato a cuantil (el número que estamos buscando).
- $z = r_i-\beta$: el **residuo** — cuánto se desvía la observación del candidato.
- $w_p(z)$: la **función de pérdida asimétrica** (también llamada *check function* o *quantile loss* en la literatura moderna; **[B]** esos nombres modernos no aparecen en Tsay, que la presenta simplemente como $w_p$).

**Verificación de que la asimetría funciona.** Tomemos $p=0.10$:

- Si la observación quedó **por encima** del candidato ($z>0$, subestimamos): el costo es $0.10\times z$ — **bajo**.
- Si la observación quedó **por debajo** del candidato ($z<0$, sobrestimamos): el costo es $(0.10-1)\times z = -0.9z$, y como $z<0$, esto es $0.9|z|$ — **nueve veces mayor**.

El optimizador, para minimizar el costo total, va a **bajar** el candidato hasta el punto donde el 10% de las observaciones queda por debajo. Ése es exactamente el cuantil 0.10.

Con $p=0.5$, ambos costos son $0.5|z|$: simétrico, y la solución es la mediana. Coherente.

### 12.5 La generalización a regresión

**[A]** Supongamos la regresión lineal:

$$r_t = \beta' x_t + a_t \tag{7.13}$$

donde $\beta$ es un vector $k$-dimensional de parámetros y $x_t$ es un vector de **predictores que son elementos de $F_{t-1}$** — es decir, información conocida **antes** del período que se predice. **[A]** Tsay señala que la distribución condicional de $r_t$ dado $F_{t-1}$ es una **traslación** de la distribución de $a_t$, porque $\beta'x_t$ es conocido.

**[A]** Koenker y Bassett (1978) proponen estimar el cuantil condicional $x_p\mid F_{t-1}$ como:

$$\hat x_p\mid F_{t-1}\equiv\inf\{\beta_o'x\mid R_p(\beta_o)=\min\} \tag{7.14}$$

donde $\beta_o$ se obtiene por:

$$\beta_o=\arg\min_\beta\sum_{t=1}^{n}w_p(r_t-\beta'x_t)$$

con $w_p(\cdot)$ definida como antes.

**En palabras:** exactamente la misma pérdida asimétrica, pero ahora el candidato ya no es un número fijo $\beta$, sino una **función de las features**: $\beta'x_t$. Cada combinación de features produce su propio cuantil estimado.

**[A]** Referencias de implementación que menciona Tsay: Koenker y D'Orey (1987) para el programa original; el paquete **`quantreg`** de R para el análisis de quantile regression.

### 12.6 Lo que Tsay NO desarrolla

**[A]** Es importante registrar el alcance real de esta subsección en el libro: **son dos páginas**. Tsay presenta la definición, la función de pérdida, la generalización lineal, y la referencia al software. **No presenta ningún ejemplo empírico de quantile regression** en este capítulo, ni discute selección de features, ni evaluación fuera de muestra, ni ninguna de las cuestiones modernas.

**Por lo tanto:**

$$\boxed{\text{quantile regression de Tsay} \neq \text{neural quantile regression}}$$

La formulación de Tsay es **lineal** (Ec. 7.13). Cualquier extensión a modelos no lineales, redes neuronales, gradient boosting con pérdida quantile, o estimación simultánea de múltiples cuantiles es **[B]** — desarrollo posterior a este texto, no atribuible a Tsay.

---

## 13. Cuantil incondicional vs cuantil condicional

**PRIORIDAD MUY ALTA.** Éste es uno de los puentes centrales del capítulo hacia Machine Learning.

### 13.1 Las dos preguntas

**Cuantil incondicional:**

$$Q_{0.95}(r)$$

> "Mirando **todos** los retornos mezclados, sin distinguir en qué condiciones ocurrieron, ¿cuál es el percentil 95?"

Es **un solo número** para toda la historia. Es lo que produce el cuantil empírico de la Sección 11.

**Cuantil condicional:**

$$Q_{0.95}(r_{t+1}\mid X_t)$$

> "**Dadas las condiciones actuales** $X_t$, ¿cuál es ahora el percentil 95 de lo que puede pasar mañana?"

Es un número **que cambia** con el estado del mercado.

### 13.2 Ejemplo intuitivo con números

Supongamos dos contextos de mercado:

**Contexto A — mercado tranquilo.** Volatilidad reciente baja, rangos chicos, volumen normal.

$$Q_{0.95}(r_{t+1}\mid A) = +0.4\%$$

**Contexto B — mercado muy volátil.** Volatilidad reciente alta, rangos amplios.

$$Q_{0.95}(r_{t+1}\mid B) = +1.5\%$$

**El punto clave, que es la parte más importante de esta sección:**

$$E[r_{t+1}\mid A]\approx E[r_{t+1}\mid B]\approx 0$$

**La media puede ser prácticamente la misma en ambos contextos, mientras los cuantiles difieren en un factor de casi 4.**

Un modelo entrenado para predecir la media no tendría casi nada que aprender de la diferencia entre A y B. Un modelo entrenado para predecir el cuantil 0.95 tendría muchísimo que aprender.

### 13.3 Conexión directa con el Capítulo 3

Esto no es una idea exótica: es la consecuencia directa de lo que ya sabemos. Si:

$$r_t = \mu_t + \sigma_t\epsilon_t$$

y $\epsilon_t$ tiene cuantil 0.95 igual a $z_{0.95}$, entonces:

$$Q_{0.95}(r_t\mid F_{t-1}) = \mu_t + \sigma_t\, z_{0.95}$$

**El cuantil condicional se mueve con $\sigma_t$ aunque $\mu_t$ sea constante.** Ésa es exactamente la razón por la cual el VaR econométrico de la Sección 8 cambia día a día mientras el cuantil empírico de la Sección 11 no cambia nunca.

**[A]** Tsay hace esta conexión explícita: todo el enfoque econométrico de la Sección 7.3 consiste en construir cuantiles condicionales a partir de pronósticos de volatilidad.

### 13.4 Por qué el cuantil incondicional puede ser engañoso

**[B]** Un cuantil incondicional calculado sobre una historia larga es un **promedio implícito sobre todos los regímenes de mercado presentes en esa historia**. Si la historia contiene un 10% de días de crisis y un 90% de días normales, el percentil 5 incondicional es una mezcla de ambos: será demasiado conservador para un día tranquilo y demasiado optimista para un día de crisis.

$$\boxed{\text{un cuantil incondicional no describe correctamente ningún régimen individual}}$$

Ésta es la misma lección que aprendimos en el Capítulo 3 sobre la volatilidad (la volatilidad incondicional no describe bien ningún período concreto), aplicada ahora a los cuantiles.

### 13.5 Qué NO se concluye

**No se concluye aquí que IRIS deba estimar cuantiles condicionales.** Lo que se registra es que:

1. Son objetos distintos del cuantil incondicional.
2. Pueden contener información que la media no contiene.
3. Existe maquinaria estadística clásica (quantile regression) para estimarlos directamente.

Nada de esto implica ninguna decisión sobre el target de IRIS.

---

## 14. Puente de quantile regression hacia Machine Learning [B]

**Toda esta sección es [B].** Tsay no desarrolla nada de esto; es una elaboración conceptual propia sobre la base de lo que sí desarrolla.

### 14.1 La idea

Un modelo de Machine Learning podría, en principio, entrenarse para estimar directamente:

$$Q_{0.10}(Y\mid X),\qquad Q_{0.50}(Y\mid X),\qquad Q_{0.90}(Y\mid X)$$

en vez de estimar únicamente:

$$E[Y\mid X]$$

**Mecánicamente**, esto solo requiere cambiar la función de pérdida del entrenamiento: en vez de error cuadrático medio (que estima la media condicional), usar la pérdida asimétrica $w_\tau$ de la Sección 12.4 (que estima el cuantil $\tau$ condicional). Cualquier arquitectura que se entrene por descenso de gradiente puede, en principio, entrenarse con esa pérdida.

**[B]** Esto es una extensión natural del marco de Koenker y Bassett a modelos no lineales, ampliamente usada en la literatura moderna de ML, pero **no descrita ni respaldada por Tsay**, cuya formulación es estrictamente lineal.

### 14.2 En qué se diferenciaría del enfoque habitual

| | Qué produce | Qué información entrega |
|---|---|---|
| **Modelo de media** | Un número: la predicción puntual | "Esperamos aproximadamente esto" |
| **Modelo de tres cuantiles** | Tres números: 0.10, 0.50, 0.90 | "Esperamos aproximadamente esto, y el rango plausible es de aquí a aquí" — una noción rudimentaria de **incertidumbre condicional** |

**[B]** La diferencia práctica es que el segundo modelo produce, en cada instante, una descripción de cuán **incierto** es el resultado, y esa incertidumbre puede variar con las condiciones de mercado. Un modelo de media, por sí solo, no distingue entre "predicción de 0% en un mercado tranquilo" y "predicción de 0% en un mercado caótico", aunque esas dos situaciones sean operativamente muy distintas.

### 14.3 Preguntas que quedan explícitamente ABIERTAS

Éstas son preguntas, **no propuestas**:

1. **¿Podrían los features contener más información sobre dispersión o colas que sobre la media?** El Capítulo 3 ya mostró que la estructura de volatilidad es mucho más marcada que la estructura de media en retornos financieros. ¿Se extiende ese patrón a los cuantiles condicionales?

2. **¿Sería útil predecir varios cuantiles del resultado futuro?** ¿Aportaría información operativamente relevante, o solo complejidad adicional?

3. **¿Podrían esos cuantiles representar incertidumbre condicional de forma útil?** ¿Cómo se relacionaría eso con la pregunta del proyecto sobre "qué nivel de confianza tiene el modelo en la señal"?

4. **¿Cómo se evaluaría su calibración fuera de muestra?** (Ver Sección 29.5.)

5. **¿Qué relación tendría un cuantil estimado con una decisión operativa?** Ninguna automáticamente — ver el guardrail siguiente.

**NO se decide que IRIS debe usar quantile regression. NO se cambia ninguna loss function. NO se adopta ningún target.**

### 14.4 GUARDRAIL: un cuantil no es una distribución completa

Ésta es una advertencia obligatoria, y es fácil de olvidar.

Predecir:

$$Q_{0.90}(Y\mid X)$$

**no** significa conocer:

$$F(Y\mid X)$$

Un cuantil es **un punto** de la función de distribución. Conocer un punto no equivale a conocer la función.

Y —esto es lo menos obvio— **predecir tres cuantiles tampoco equivale a haber estimado correctamente toda la distribución**. Tres puntos de una curva no determinan la curva. En particular:

- No dicen nada sobre lo que ocurre **más allá** del cuantil más extremo estimado. Si estimamos $Q_{0.90}$, no sabemos nada sobre el comportamiento del 10% superior más que su frontera inferior.
- No garantizan ninguna forma funcional entre los cuantiles estimados.
- **[B]** Nada garantiza siquiera que los cuantiles estimados sean **coherentes entre sí**. Un modelo que estima $Q_{0.10}$, $Q_{0.50}$ y $Q_{0.90}$ por separado podría producir, en algún punto del espacio de features, una estimación de $Q_{0.10}$ **mayor** que la de $Q_{0.50}$, lo cual es lógicamente imposible para cuantiles verdaderos. Este fenómeno se conoce en la literatura moderna como **quantile crossing**. **[B]** Tsay no lo menciona; se registra aquí como cuestión secundaria, solo para dejar constancia de que existe.

$$\boxed{Q_\tau(Y\mid X) \text{ para algunos } \tau \quad\neq\quad F(Y\mid X)}$$

---

## 15. Extreme Value Theory desde cero

**PRIORIDAD CONCEPTUAL ALTA.**

### 15.1 La pregunta fundamental

> **Si solo me interesan los eventos más extremos, ¿por qué debería modelar toda la distribución?**

Ésta es la pregunta que motiva toda la teoría de valores extremos (EVT).

Todos los métodos anteriores del capítulo intentan, de una forma u otra, describir la distribución completa (o al menos una parte sustancial de ella) y **luego** leer un punto de la cola. La EVT invierte el enfoque: **describe directamente el comportamiento de los extremos**, ignorando deliberadamente el resto.

### 15.2 Una analogía

Si quiero diseñar un dique para proteger una ciudad de inundaciones excepcionales, ¿qué debería modelar?

- ¿La altura del río **todos los días del año**, incluidos los 340 días en que está perfectamente normal?
- ¿O las **mayores crecidas** históricas, que son las únicas que ponen en riesgo el dique?

Modelar todos los días normales con exquisito detalle no ayuda a decidir la altura del dique. Peor: si ajusto un modelo a todos los datos, el ajuste va a estar dominado por los 340 días normales (que son la enorme mayoría de las observaciones), y el modelo resultante puede describir muy mal precisamente los 3 días que importan.

**[A]** Tsay lo formula así en la introducción: la teoría de valores extremos fue desarrollada en la literatura estadística *"for studying rare (or extraordinary) events"*.

### 15.3 El punto de partida formal

**[A]** Consideremos $n$ retornos $\{r_1,\dots,r_n\}$. El máximo es $r_{(n)}=\max_{1\le j\le n}\{r_j\}$ y el mínimo $r_{(1)}=\min_{1\le j\le n}\{r_j\}$.

**[A]** Tsay se enfoca en el **máximo**, señalando que las propiedades del mínimo se obtienen por simple cambio de signo: $r_{(1)}=-\max_{1\le j\le n}\{-r_j\}$. Y recuerda: *"The minimum return is relevant to holding a long financial position. As before, we shall use negative log returns, instead of the log returns, to perform VaR calculation for a long position."*

**[A] Primer resultado.** Asumiendo que los retornos son **serialmente independientes** con CDF común $F(x)$, la CDF del máximo es:

$$F_{n,n}(x)=\Pr[r_{(n)}\le x]=\prod_{j=1}^{n}\Pr(r_j\le x)=[F(x)]^n \tag{7.15}$$

**Explicación en palabras:** "el máximo de $n$ observaciones es menor o igual a $x$" es exactamente lo mismo que "**todas** las observaciones son menores o iguales a $x$". Bajo independencia, la probabilidad de que todas cumplan la condición es el producto de las probabilidades individuales, es decir, $[F(x)]^n$.

### 15.4 El problema de la degeneración

**[A]** En la práctica $F(x)$ es desconocida, y por lo tanto $F_{n,n}(x)$ también. Pero hay un problema peor: **a medida que $n$ crece, $F_{n,n}(x)$ degenera**. Si $u$ es el extremo superior del rango de $r_t$:

$$F_{n,n}(x)\to 0 \text{ si } x<u,\qquad F_{n,n}(x)\to1 \text{ si } x\ge u$$

**En palabras:** si tomamos muchísimas observaciones, el máximo tiende con probabilidad 1 a acercarse al valor más alto posible. La distribución límite es una función escalón sin información útil. **[A]** *"This degenerated CDF has no practical value."*

**[A] La solución: normalizar.** La teoría de valores extremos busca dos sucesiones $\{\beta_n\}$ y $\{\alpha_n\}$ (con $\alpha_n>0$) tales que la distribución de:

$$r_{(n)}^*\equiv\frac{r_{(n)}-\beta_n}{\alpha_n}$$

converja a una distribución **no degenerada**. La sucesión $\{\beta_n\}$ es de **ubicación** (location) y $\{\alpha_n\}$ de **escala** (scaling factors).

**Analogía [B]:** es exactamente la misma idea que el Teorema Central del Límite. La suma de $n$ variables no converge a nada útil por sí sola (crece sin límite), pero **estandarizada** (restando la media y dividiendo por la desviación estándar) converge a una normal. Aquí ocurre lo mismo con el máximo: no converge a nada útil sin normalizar, y sí converge a algo útil una vez normalizado.

### 15.5 Las dos implicaciones importantes

**[A]** Tsay destaca dos consecuencias de la teoría:

**1. Es el comportamiento de la cola, no la distribución específica, lo que determina el límite.**

*"the tail behavior of the CDF $F(x)$ of $r_t$, not the specific distribution, determines the limiting distribution $F_*(x)$ of the (normalized) maximum. Thus, the theory is generally applicable to a wide range of distributions for the return $r_t$."*

Ésta es una propiedad poderosa: no necesitamos saber si los retornos son Student-t, estables, o cualquier otra cosa. Lo único que determina la distribución límite del máximo es cómo se comporta la cola.

**2. El tail index es invariante bajo agregación temporal.**

**[A]** Feller (1971, p. 279) muestra que el tail index $\xi$ **no depende del intervalo temporal de $r_t$**. *"That is, the tail index (or equivalently the shape parameter) is invariant under time aggregation."*

**[B] Esto es potencialmente muy relevante para datos intradiarios**, pero exige distinguir entre el **parámetro teórico** y su **estimación en una muestra finita**. Bajo los supuestos de la teoría, el tail index —o parámetro de forma equivalente— es una propiedad asintótica que debería preservarse bajo agregación temporal. Esto **no implica** que las estimaciones obtenidas sobre retornos de 1 minuto, 5 minutos o diarios deban coincidir exactamente: cada estimación está afectada por tamaño de muestra, elección de threshold o bloques, dependencia temporal, microestructura y error de estimación.

Por tanto, la implicación empírica razonable no es exigir igualdad exacta entre $\hat\xi_{1m}$, $\hat\xi_{5m}$ y $\hat\xi_{diario}$, sino preguntar si sus diferencias son compatibles con la incertidumbre de estimación y con los supuestos bajo los cuales se deriva la invariancia. Esto se registra como hipótesis H7.10. **No debe darse por sentado que la propiedad teórica se observará exactamente en estimaciones de muestras finitas.**

### 15.6 Extensión a datos dependientes

**[A]** La teoría se ha extendido a observaciones serialmente dependientes *"provided that the dependence is weak"*. Tsay cita dos referencias concretas:

- **Berman (1964):** la misma forma de distribución límite de valores extremos vale para secuencias normales estacionarias siempre que la función de autocorrelación de $r_t$ sea **cuadrado-sumable**, es decir, $\sum_{i=1}^\infty\rho_i^2<\infty$.
- **Leadbetter, Lindgren y Rootzén (1983, Cap. 3)** para resultados más generales.

**[A]** Y anticipa: *"We shall discuss extremal index for a strictly stationary time series later in Section 7.8."* Ése es el tema de la Sección 25 de este informe.

---

## 16. Block maxima/minima y GEV

### 16.1 Block maxima, explicado sin distribuciones límite

**El problema práctico [A]:** *"For a given sample, there is only a single minimum or maximum, and we cannot estimate the three parameters with only an extreme observation. Alternative ideas must be used."*

Es decir: la teoría describe la distribución del máximo, pero de una muestra dada tenemos **un solo máximo**. No se pueden estimar tres parámetros con un dato.

**La solución [A]:** dividir la muestra en subperíodos y aplicar la teoría a cada uno.

**Ejemplo concreto:** tenemos 10 años de datos diarios. Los dividimos en **meses**. De cada mes, tomamos **la peor pérdida de ese mes**. Terminamos con:

$$10 \text{ años} \times 12 \text{ meses} = 120 \text{ extremos}$$

Ahora sí tenemos 120 observaciones de "el máximo de un bloque", y podemos ajustar una distribución a ellas.

**[A] Formalización.** Con $T$ retornos disponibles, se divide la muestra en $g$ subperíodos no solapados de $n$ observaciones cada uno ($T=ng$):

$$\{r_1,\dots,r_n \mid r_{n+1},\dots,r_{2n}\mid\cdots\mid r_{(g-1)n+1},\dots,r_{ng}\}$$

y el máximo del $i$-ésimo bloque es:

$$r_{n,i}=\max_{1\le j\le n}\{r_{(i-1)n+j}\},\qquad i=1,\dots,g \tag{7.20}$$

**[A]** La colección $\{r_{n,i}\}$ es el conjunto de datos que se usa para estimar los parámetros. *"Clearly, the estimates obtained may depend on the choice of subperiod length $n$."*

**[A] Elecciones habituales:** para retornos diarios, $n=21$ corresponde aproximadamente a un mes de negociación, y $n=63$ a un trimestre.

**[A] Nota práctica:** cuando $T$ no es múltiplo de $n$, se puede o bien permitir que el último bloque sea más chico, o bien descartar las primeras observaciones.

### 16.2 Qué información se descarta

Ésta es la parte que hay que tener muy presente.

**Con 10 años de datos diarios (aproximadamente 2520 observaciones) y bloques mensuales, nos quedamos con 120 números y descartamos 2400.**

Lo que se descarta incluye:

- Todas las observaciones "no extremas" de cada mes (obviamente).
- Pero también: **el segundo, tercero y cuarto peor día de cada mes**, aunque hayan sido casi tan malos como el peor.
- Y también: la información sobre **si los días malos estaban agrupados o dispersos** dentro del mes.

**[B]** El tercer punto es especialmente importante. Si un mes tuvo un episodio de cinco días consecutivos de caídas grandes, block maxima registra **un solo número**: el peor de esos cinco. Si otro mes tuvo un único día malo aislado de magnitud similar, block maxima registra un número prácticamente idéntico. **Los dos meses se ven iguales para el método**, aunque económicamente sean muy distintos. Ésta es exactamente la debilidad que motiva el enfoque de peaks-over-threshold (Sección 20) y el extremal index (Sección 25).

### 16.3 El trade-off de block size

$$\boxed{\text{bloques grandes: mejor aproximación asintótica} \quad\leftrightarrow\quad \text{bloques chicos: más observaciones extremas}}$$

**[A]** Tsay lo formula explícitamente en el Remark de la Sección 7.6:

> *"the VaR calculation based on the traditional extreme value theory depends on the choice of $n$, which is the length of subperiods. For the limiting extreme value distribution to hold, one would prefer a large $n$. But a larger $n$ means a smaller $g$ when the sample size $T$ is fixed, where $g$ is the effective sample size used in estimating the three parameters $\alpha_n$, $\beta_n$, and $\xi_n$. Therefore, some compromise between the choices of $n$ and $g$ is needed. A proper choice may depend on the returns of the asset under study. We recommend that one should check the stability of the resulting VaR in applying the traditional extreme value theory."*

**Traducción práctica:**

- **$n$ grande** (por ejemplo, bloques anuales): la teoría asintótica se aproxima mejor (porque el máximo de muchas observaciones se parece más a su distribución límite), **pero** con $T$ fijo hay muy pocos bloques y por lo tanto muy pocos datos para estimar los tres parámetros.
- **$n$ chico** (por ejemplo, bloques mensuales): muchos bloques, muchos datos para estimar, **pero** la aproximación asintótica puede ser mala.

**[A] Evidencia numérica de este trade-off en la Tabla 7.2 de Tsay:** con $n=252$ (bloques anuales), quedan solo $g=36$ bloques, y el error estándar del parámetro de forma sube a 0.127 (mínimos) y 0.186 (máximos), frente a 0.036 con bloques mensuales ($g=437$). **[A]** El propio Tsay lo observa: *"The results for $n=252$ have higher variabilities as the number of subperiods $g$ is relatively small."*

**No hay una respuesta "correcta" a la elección de $n$. Es un compromiso, y Tsay recomienda verificar la estabilidad del resultado.**

### 16.4 La distribución GEV

**[A]** Bajo el supuesto de independencia, la distribución límite del máximo normalizado $r_{(n)}^*$ es:

$$F_*(x)=\begin{cases}\exp\left[-(1+\xi x)^{-1/\xi}\right] & \text{si } \xi\neq0\\ \exp[-\exp(-x)] & \text{si } \xi=0\end{cases} \tag{7.16}$$

para $x<-1/\xi$ si $\xi<0$ y $x>-1/\xi$ si $\xi>0$. El caso $\xi=0$ se toma como el límite cuando $\xi\to0$.

**[A]** Ésta es la **generalized extreme value (GEV) distribution** de Jenkinson (1955).

**Los tres parámetros [A]:**

| Parámetro | Nombre | Qué controla |
|---|---|---|
| $\beta_n$ | **location** (ubicación) | Dónde está centrada la distribución de los máximos de bloque. Desplaza la distribución sin cambiarle la forma. |
| $\alpha_n$ | **scale** (escala) | Cuán dispersa está la distribución de los máximos. Debe ser positivo. |
| $\xi$ | **shape** (forma) | **Cómo se comporta la cola.** Es el parámetro decisivo. |

**[A]** El parámetro $\alpha=1/\xi$ se llama el **tail index** de la distribución. (Colisión de notación: $\alpha$ como tail index vs. $\alpha_n$ como parámetro de escala. Tsay usa ambos y advierte sobre la confusión en la Sección 7.6.2.)

**[A] La densidad**, obtenida por diferenciación:

$$f_*(x)=\begin{cases}(1+\xi x)^{-1/\xi-1}\exp\left[-(1+\xi x)^{-1/\xi}\right] & \text{si }\xi\neq0\\ \exp[-x-\exp(-x)] & \text{si }\xi=0\end{cases} \tag{7.19}$$

### 16.5 Métodos de estimación

**[A]** Tsay presenta tres enfoques, que se registran aquí de forma resumida:

**Máxima verosimilitud.** Se asume que los máximos de subperíodo $\{r_{n,i}\}$ siguen una GEV, se escribe la densidad de $r_{n,i}$ por transformación, y se maximiza el producto de densidades bajo supuesto de independencia entre bloques. **[A]** *"These estimates are unbiased, asymptotically normal, and of minimum variance under proper assumptions."* Referencias: Embrechts et al. (1997), Coles (2001). **Es el método que Tsay usa en el capítulo.**

**Método de regresión** (Gumbel, 1958). Usa propiedades de los order statistics: $E\{F_*[r_{n(i)}]\}=i/(g+1)$. Tomando logaritmo dos veces se obtiene una ecuación de regresión (Ec. 7.23) que puede resolverse por mínimos cuadrados. **[A]** *"The least-squares estimates are consistent but less efficient than the likelihood estimates."*

**Métodos no paramétricos:** los estimadores de **Hill** (1975) y **Pickands** (1975), que estiman solo el parámetro de forma $\xi$ y se aplican directamente a los retornos, **sin necesidad de dividir en subgrupos**:

$$\xi_p(q)=\frac{1}{\ln(2)}\ln\left(\frac{r_{(T-q+1)}-r_{(T-2q+1)}}{r_{(T-2q+1)}-r_{(T-4q+1)}}\right),\quad q\le T/4 \tag{7.24}$$

$$\xi_h(q)=\frac{1}{q}\sum_{i=1}^{q}\left[\ln(r_{(T-i+1)})-\ln(r_{(T-q)})\right] \tag{7.25}$$

**[A] Puntos importantes sobre estos estimadores:**

- Ambos dependen de la elección de $q$ (cuántos order statistics extremos se usan). *"The choice of $q$ differs between Hill and Pickands estimators. It has been investigated by several researchers, but **there is no general consensus on the best choice available**."*
- El estimador de Hill *"is applicable to the Frechet distribution only, but it is more efficient than the Pickands estimator when applicable."*
- **Procedimiento práctico [A]:** *"one may plot the Hill estimator $\xi_h(q)$ against $q$ and find a proper $q$ such that the estimate appears to be stable."*

**[B]** Nótese el patrón, que se repetirá con la elección de threshold en la Sección 21: no hay una regla automática; hay un **gráfico diagnóstico** y un juicio humano sobre dónde el resultado "parece estable". Esto es una fuente estructural de subjetividad en toda la estimación de colas.

---

## 17. Shape parameter y comportamiento de colas

### 17.1 Las tres familias

**[A]** La GEV encompassa los tres tipos de distribución límite de Gnedenko (1943):

**Tipo I: $\xi=0$ — familia Gumbel.**

$$F_*(x)=\exp[-\exp(-x)],\qquad -\infty<x<\infty$$

**[A]** *"The Gumbel family consists of thin-tailed distributions such as normal and lognormal distributions."* La cola derecha de $F(x)$ **decae exponencialmente**.

**Tipo II: $\xi>0$ — familia Fréchet.**

$$F_*(x)=\begin{cases}\exp[-(1+\xi x)^{-1/\xi}] & \text{si } x>-1/\xi\\ 0 & \text{en otro caso}\end{cases}$$

**[A]** La cola decae **por una función de potencia** (más lentamente que exponencial → cola pesada). *"For risk management, we are mainly interested in the Frechet family, which includes stable and Student-t distributions."*

**Tipo III: $\xi<0$ — familia Weibull.**

$$F_*(x)=\begin{cases}\exp[-(1+\xi x)^{-1/\xi}] & \text{si } x<-1/\xi\\ 1 & \text{en otro caso}\end{cases}$$

**[A]** La cola es **finita**: existe un valor máximo que la variable no puede superar.

### 17.2 Interpretación en lenguaje simple

| $\xi$ | Familia | Qué significa la cola |
|---|---|---|
| $\xi>0$ | Fréchet | **Cola pesada.** No hay límite superior. La probabilidad de valores extremos decae lentamente (como una potencia). Incluye Student-t y distribuciones estables. |
| $\xi=0$ | Gumbel | **Cola de decaimiento exponencial.** Tampoco hay límite superior estricto, pero la probabilidad de valores muy grandes cae muy rápido. Incluye normal y lognormal. |
| $\xi<0$ | Weibull | **Cola finita.** Existe un valor máximo que no puede superarse. |

**[B]** Una intuición útil sobre $\xi$: cuanto **más grande** es $\xi$, más pesada es la cola, es decir, más probable es encontrar valores mucho más allá de lo típico. El tail index $\alpha=1/\xi$ funciona al revés: un $\alpha$ chico corresponde a una cola muy pesada.

### 17.3 GUARDRAIL: no asumir $\xi>0$

**No debe asumirse que "las finanzas siempre tienen $\xi>0$".**

**[A]** Tsay dice, con precisión: *"For risk management, we are mainly interested in the Frechet family"* — es decir, es el caso de interés práctico habitual. Y en la Sección 7.6 (Parte I): *"In financial applications, the case of $\xi_n\neq0$ is of major interest."*

Eso **no** es lo mismo que decir que $\xi>0$ es un hecho universal de todos los mercados. Es una observación sobre qué caso resulta más relevante en las aplicaciones típicas de riesgo, más los resultados empíricos que él mismo obtiene sobre IBM.

$$\boxed{\xi \text{ debe estimarse para cada cola y cada dataset, no asumirse}}$$

**[A] Evidencia de que esto importa:** en la Tabla 7.4 de Tsay (modelo con variables explicativas), el parámetro de forma **estimado para el 31 de diciembre de 1998** resulta $\xi_{9190}=0.01195$ — es decir, prácticamente **cero**, sugiriendo comportamiento tipo Gumbel para ese día concreto. Mientras que para el 30 de diciembre de 1998, con condiciones de mercado distintas, el mismo modelo da $\xi_{9189}=0.2500$. **El parámetro de forma, en un modelo condicional, cambia de un día al siguiente.**

### 17.4 Extremo izquierdo y extremo derecho

Ya introducido en la Sección 3.6, pero merece registrarse aquí con la evidencia empírica.

**[A] Resultado sobre IBM (Sección 7.5.3):** *"The plots also indicate that the shape parameter $\xi$ appears to be larger for the negative extremes, indicating that the daily log return may have a heavier left tail."*

**[A] Los números concretos (Tabla 7.1, estimador de Hill):**

| $q$ | 190 | 200 | 210 |
|---|---|---|---|
| $r_t$ (extremos positivos) | 0.300 (0.022) | 0.299 (0.021) | 0.305 (0.021) |
| $-r_t$ (extremos negativos) | 0.290 (0.021) | 0.292 (0.021) | 0.289 (0.020) |

*(Errores estándar entre paréntesis.)*

**[B] Observación cuidadosa:** en esta tabla específica, los valores para $r_t$ y $-r_t$ son **muy similares** (0.30 vs 0.29), y la diferencia es claramente menor que los errores estándar. La afirmación de Tsay sobre la cola izquierda más pesada proviene de la inspección de los **gráficos** del estimador de Hill (Figura 7.4) y de la Tabla 7.2 (MLE por bloques), donde con $n\ge63$ los estimados para mínimos son ≈0.33 y estables, mientras los de máximos son menos estables. **Se registra la afirmación de Tsay, pero también se registra que la evidencia numérica de la Tabla 7.1 por sí sola no la sustenta con holgura.**

**[A]** Lo que sí es un resultado sólido de la Tabla 7.1: los estimados son *"stable"* entre las tres elecciones de $q$, y *"significantly different from zero at the asymptotic 5% level"* — es decir, la evidencia contra $\xi=0$ (contra normalidad de colas) es clara. **[A]** *"Overall, the result indicates that the distribution of daily log returns of IBM stock belongs to the Frechet family. The analysis thus rejects the normality assumption commonly used in practice."* En concordancia con Longin (1996), que usó un índice del mercado accionario estadounidense.

### 17.5 Resultados completos de la estimación MLE por bloques

**[A] Tabla 7.2.** Serie: IBM, log returns diarios en porcentaje, 3 jul 1962 – 31 dic 1998, $T=9190$. Método: máxima verosimilitud sobre máximos de bloque.

**Minimal Returns** (es decir, máximos de los retornos negativos → relevante para posición **long**):

| Longitud del subperíodo | Escala $\alpha_n$ | Ubicación $\beta_n$ | Forma $\xi_n$ |
|---|---|---|---|
| 1 mes ($n=21$, $g=437$) | 0.823 (0.035) | 1.902 (0.044) | 0.197 (0.036) |
| 1 trimestre ($n=63$, $g=145$) | 0.945 (0.077) | 2.583 (0.090) | 0.335 (0.076) |
| 6 meses ($n=126$, $g=72$) | 1.147 (0.131) | 3.141 (0.153) | 0.330 (0.101) |
| 1 año ($n=252$, $g=36$) | 1.542 (0.242) | 3.761 (0.285) | 0.322 (0.127) |

**Maximal Returns** (relevante para posición **short**):

| Longitud del subperíodo | Escala $\alpha_n$ | Ubicación $\beta_n$ | Forma $\xi_n$ |
|---|---|---|---|
| 1 mes ($n=21$, $g=437$) | 0.931 (0.039) | 2.184 (0.050) | 0.168 (0.036) |
| 1 trimestre ($n=63$, $g=145$) | 1.157 (0.087) | 3.012 (0.108) | 0.217 (0.066) |
| 6 meses ($n=126$, $g=72$) | 1.292 (0.158) | 3.471 (0.181) | 0.349 (0.130) |
| 1 año ($n=252$, $g=36$) | 1.624 (0.271) | 4.475 (0.325) | 0.264 (0.186) |

**[A] Las cuatro observaciones que hace Tsay:**

1. Los estimados de ubicación y escala **crecen en módulo a medida que $n$ aumenta**. Es lo esperado: el mínimo o máximo de un período más largo es, por construcción, más extremo.
2. Los estimados del parámetro de forma son **estables para los extremos negativos cuando $n\ge63$**, aproximadamente **0.33**.
3. Los estimados son **menos estables para los extremos positivos**. Son menores en magnitud pero siguen siendo significativamente distintos de cero.
4. Los resultados con $n=252$ tienen **mayor variabilidad** porque $g$ es relativamente chico.

**[B] Lo que esto muestra sobre el trade-off de block size:** para los extremos negativos, pasar de $n=21$ a $n=63$ cambia el parámetro de forma de 0.197 a 0.335 — **un cambio de 70%**. Y ese parámetro es exactamente el que gobierna el comportamiento de la cola. La elección de block size no es un detalle técnico.

### 17.6 Un dato empírico adicional

**[A]** La Figura 7.3 de Tsay muestra los extremos diarios de IBM con subperíodos de 21 días. *"The October 1987 crash is clearly seen from the plot. Excluding the 1987 crash, the range of extreme daily log returns is between 0.5 and 13%."*

**[B]** Vale la pena detenerse en esto: **una sola observación** (el crash del 19 de octubre de 1987) está tan lejos del resto que Tsay tiene que describirla por separado del rango de todo el resto de los extremos. En una muestra de 9190 días. Ése es el tipo de observación que domina cualquier estimación de cola.

---

## 18. Traditional EVT para VaR

### 18.1 El procedimiento

**[A]** Tsay presenta un enfoque similar al de Longin (1999a,b), en dos partes.

**Parte I: obtener el cuantil de la GEV.** Con las estimaciones MLE de $\beta_n$, $\alpha_n$ y $\xi_n$, y siendo $p^*$ una probabilidad de cola superior pequeña para el máximo de subperíodo, se despeja el cuantil $r_n^*$:

$$r_n^*=\begin{cases}\beta_n-\dfrac{\alpha_n}{\xi_n}\left\{1-\left[-\ln(1-p^*)\right]^{-\xi_n}\right\} & \text{si }\xi_n\neq0\\ \beta_n-\alpha_n\ln[-\ln(1-p^*)] & \text{si }\xi_n=0\end{cases} \tag{7.26}$$

**Parte II: pasar del máximo de subperíodo al retorno individual.** **[A]** Aquí está el paso conceptual clave. Como la mayoría de los retornos de activos son serialmente no correlacionados o tienen correlación serial débil, se usa la relación (7.15):

$$1-p^*=P(r_{n,i}\le r_n^*)=[P(r_t\le r_n^*)]^n \tag{7.27}$$

**En palabras:** "el máximo del bloque es menor o igual a $r_n^*$" equivale a "**todas** las $n$ observaciones del bloque son menores o iguales a $r_n^*$". Esta ecuación permite traducir una probabilidad sobre máximos de bloque en una probabilidad sobre retornos individuales.

Sustituyendo, se obtiene la fórmula final de VaR:

$$\mathrm{VaR}=\begin{cases}\beta_n-\dfrac{\alpha_n}{\xi_n}\left\{1-\left[-n\ln(1-p)\right]^{-\xi_n}\right\} & \text{si }\xi_n\neq0\\\beta_n-\alpha_n\ln[-n\ln(1-p)] & \text{si }\xi_n=0\end{cases} \tag{7.28}$$

donde $n$ es la longitud del subperíodo.

**[A] Los cuatro pasos del procedimiento resumido:**

1. Elegir la longitud del subperíodo $n$ y obtener los máximos $\{r_{n,i}\}$, con $g=[T/n]$.
2. Obtener los estimados MLE de $\beta_n$, $\alpha_n$, $\xi_n$.
3. **Verificar la adecuación del modelo ajustado** (ver Sección 23 de este informe).
4. Si el modelo es adecuado, aplicar la Ec. (7.28).

**[A] Recordatorio de signos:** *"Since we focus on loss function so that maxima of log returns are used in the derivation. Keep in mind that for a long financial position, the return series used in loss function is the negative log returns, not the traditional log returns."*

### 18.2 Ejemplo empírico: Ejemplo 7.6

**[A] Serie:** IBM, log returns diarios en porcentaje, 1962–1998, 9190 obs. **Posición:** long \$10 millones.

**Con $n=63$** ($\hat\alpha_n=0.945$, $\hat\beta_n=2.583$, $\hat\xi_n=0.335$):

$$\mathrm{VaR}=2.583-\frac{0.945}{0.335}\left\{1-[-63\ln(1-0.01)]^{-0.335}\right\}=3.04969$$

Para \$10 millones: **VaR(1%) = \$304.969**. Con $p=0.05$: **VaR(5%) = \$166.641**.

**Con $n=21$** ($\hat\alpha_n=0.823$, $\hat\beta_n=1.902$, $\hat\xi_n=0.197$):

$$\mathrm{VaR}=1.902-\frac{0.823}{0.197}\left\{1-[-21\ln(1-0.01)]^{-0.197}\right\}=3.40013$$

**VaR(1%) = \$340.013**; **VaR(5%) = \$184.127**.

**[A]** *"In this particular case, the choice of $n=21$ gives higher VaR values."*

**[A] Y el comentario más importante:**

> *"It is somewhat surprising to see that the VaR values obtained in Example 7.6 using the extreme value theory are smaller than those of Example 7.3 that uses a GARCH(1,1) model. In fact, the VaR values of Example 7.6 are even smaller than those based on the empirical quantile in Example 7.5."*

**[A]** Y explica las razones: en parte por la elección de probabilidad 0.05, y en parte porque *"the VaR obtained here via the traditional extreme value theory may not be adequate because the independent assumption of daily log returns is often rejected by statistical testings. Finally, the use of subperiod maxima overlooks the fact of volatility clustering in the daily log returns."*

**[A]** A $p=0.001$ la comparación se invierte: \$546.641 para el AR(2)–GARCH gaussiano vs **\$666.590** para EVT con $n=21$.

### 18.3 Tabla comparativa de métodos

**[A] Los resultados completos de la Sección 7.6.1** — misma serie (IBM diaria, 1962–1998, 9190 obs), misma posición (long \$10 millones), mismo horizonte (1 día):

| Método | $p=5\%$ | $p=1\%$ | $p=0.1\%$ |
|---|---|---|---|
| RiskMetrics | \$302.500 | \$426.500 | \$566.443 |
| AR(2)–GARCH(1,1) gaussiano | \$287.200 | \$409.738 | \$546.641 |
| AR(2)–GARCH(1,1) Student-t(5) | \$283.520 | \$475.943 | \$836.341 |
| Cuantil empírico | \$216.030 | \$365.709 | \$780.712 |
| EVT tradicional ($n=21$) | \$184.127 | \$340.013 | \$666.590 |

*(Nota: Tsay escribe \$287.200 en esta lista y \$287.700 en el Ejemplo 7.3 para el mismo cálculo. Se registran ambos.)*

**Tabla conceptual solicitada** — qué supone cada método, qué usa de los datos, ventajas y limitaciones. **[A]** para lo que Tsay afirma explícitamente, **[B]** para las síntesis propias:

| Método | Qué supone | Qué usa de los datos | Ventaja | Riesgo / limitación |
|---|---|---|---|---|
| **RiskMetrics** | Normalidad condicional; media cero; IGARCH(1,1) sin drift [A] | Toda la serie, para estimar $\alpha$ del EWMA de volatilidad [A] | Simplicidad, transparencia [A]; VaR condicional a la volatilidad actual [B] | Normalidad puede subestimar VaR con colas pesadas [A]; supuesto de media cero es fuerte [A]; la regla $\sqrt{k}$ depende de sus supuestos [A] |
| **AR–GARCH gaussiano** | Estructura ARMA + GARCH; $\epsilon_t$ normal [A] | Toda la serie, para estimar media y volatilidad condicionales [A] | VaR condicional al estado actual; permite media no nula y estructura ARMA [A] | La normalidad de $\epsilon_t$ puede ser inadecuada en la cola [A]; depende de la especificación correcta del modelo [B] |
| **AR–GARCH Student-t** | Igual, con $\epsilon_t$ Student-t estandarizada de $v$ g.l. [A] | Igual [A] | Captura colas más pesadas; efecto notable cuando $p$ es chico [A] | Requiere elegir/estimar $v$ [B]; usar Student-t no garantiza un VaR correcto [B] |
| **Cuantil empírico** | Que la distribución no cambia del período muestral al de predicción [A] | Los order statistics; en la práctica, unos pocos alrededor del cuantil [A] | Sin supuesto distribucional; simplicidad [A]; puede servir como cota inferior si $p$ no es muy chico [A] | No extrapola más allá del peor evento observado [A]; ineficiente con $p$ chico [A]; no admite variables explicativas [A] |
| **EVT tradicional (block maxima)** | Independencia entre observaciones; GEV como distribución límite del máximo de bloque [A] | **Solo los máximos de cada bloque** — descarta el resto [A] | Se enfoca directamente en la cola; extrapola más allá de lo observado [B] | Muy sensible a la elección de $n$ [A]; el supuesto de independencia suele rechazarse [A]; ignora volatility clustering [A] |
| **EVT nuevo (POT / Poisson 2D)** | GPD para los excesos sobre threshold; proceso de Poisson para los tiempos [A] | **Todas las excedencias** sobre un threshold, con sus tiempos [A] | VaR más estable que EVT tradicional [A]; admite variables explicativas [A]; no requiere elegir $n$ [A] | Requiere elegir threshold $\eta$ [A]; el modelo homogéneo puede fallar los diagnósticos [A] |

**GUARDRAIL: no se declara ningún ganador universal.**

**[A]** Tsay explícitamente no lo hace: *"Since there is no true VaR available to compare the accuracy of different approaches, we recommend that one applies several methods to gain insight into the range of VaR."*

---

## 19. Incertidumbre y sensibilidad de estimación de colas

**Sección especial.** Ésta es, probablemente, la lección metodológica más transferible de todo el capítulo.

### 19.1 El mensaje central

$$\boxed{\text{el extremo que queremos estimar es precisamente donde tenemos menos datos}}$$

**[A]** Tsay lo dice sin rodeos: *"There are substantial differences among different approaches. This is not surprising because there exists substantial uncertainty in estimating tail behavior of a statistical distribution."*

### 19.2 La magnitud de la discrepancia

Vale la pena cuantificar cuán "sustanciales" son esas diferencias. Con **exactamente los mismos datos**:

| Nivel | Mínimo | Máximo | Diferencia relativa |
|---|---|---|---|
| $p=5\%$ | \$184.127 (EVT) | \$302.500 (RiskMetrics) | **64%** |
| $p=1\%$ | \$340.013 (EVT) | \$475.943 (GARCH-t) | **40%** |
| $p=0.1\%$ | \$546.641 (GARCH normal) | \$836.341 (GARCH-t) | **53%** |

**[B]** No estamos hablando de diferencias del 2% o 5%. Estamos hablando de que la respuesta a "¿cuánto podemos perder?" cambia en un factor de 1.4 a 1.6 según qué método razonable elijamos, sobre la misma serie de 9190 observaciones de una de las acciones más líquidas y mejor documentadas del mundo.

### 19.3 La precisión aparente vs. la precisión epistemológica

Un número como:

$$\mathrm{VaR}=0.0738421$$

tiene siete dígitos significativos. **No significa que tengamos siete dígitos de conocimiento.**

**[B]** Los decimales de un output de software reflejan la precisión aritmética del cálculo, no la precisión del conocimiento subyacente. Dado que métodos alternativos razonables discrepan en 40–60%, reportar un VaR con más de dos cifras significativas es, epistemológicamente, engañoso.

**Ejemplo concreto del propio capítulo [A]:** en el Ejemplo 7.6, Tsay reporta $\mathrm{VaR}=3.04969$ (seis cifras) para $n=63$ y $\mathrm{VaR}=3.40013$ para $n=21$. Los dos números tienen seis cifras cada uno, y difieren entre sí en el **segundo dígito**, únicamente por haber cambiado la longitud del bloque de 63 a 21 días.

### 19.4 Tres tipos de incertidumbre

Es útil distinguirlas [B], aunque Tsay no las nombre con estas etiquetas:

**1. Estimation uncertainty (incertidumbre de estimación).**

Dado un modelo fijo, los parámetros se estiman con error muestral. **[A]** Los errores estándar de la Tabla 7.2 lo cuantifican: para $n=252$, el error estándar del parámetro de forma de los máximos es 0.186 sobre un estimado de 0.264 — es decir, **el error estándar es el 70% del estimado**. Un intervalo de confianza al 95% incluiría cómodamente el cero.

**2. Model uncertainty (incertidumbre de modelo).**

Dado los mismos datos, distintos modelos razonables dan respuestas distintas. Es exactamente lo que muestra la tabla comparativa de la Sección 18.3. **Esta fuente de incertidumbre normalmente no aparece en ningún intervalo de confianza**, porque los intervalos de confianza se calculan **dado** el modelo.

**3. Threshold / block-size uncertainty.**

Dado el mismo modelo y los mismos datos, distintas elecciones de un hiperparámetro (el $n$ de block maxima, el $\eta$ de POT, el $q$ del estimador de Hill) dan respuestas distintas.

**[A] Evidencia numérica de cada una en el capítulo:**

- *Estimación:* errores estándar de las Tablas 7.1, 7.2, 7.3, 7.4.
- *Modelo:* la tabla comparativa de 5 métodos de la Sección 7.6.1.
- *Block size:* $\xi=0.197$ con $n=21$ vs $\xi=0.335$ con $n=63$ (Tabla 7.2); VaR de \$340.013 vs \$304.969 (Ejemplo 7.6).
- *Threshold:* $\xi=0.307$ con $\eta=3.0\%$, $\xi=0.264$ con $\eta=2.5\%$, $\xi=0.188$ con $\eta=2.0\%$ (Tabla 7.3).

**Nótese ese último caso: bajar el threshold de 3.0% a 2.0% cambia el parámetro de forma en un 39%.**

### 19.5 La recomendación de Tsay

**[A]** Dos recomendaciones explícitas, ambas del mismo espíritu:

1. *"we recommend that one applies several methods to gain insight into the range of VaR"* (Sección 7.6.1).
2. *"We recommend that one should check the stability of the resulting VaR in applying the traditional extreme value theory"* (Remark de la Sección 7.6).

**[B] La lectura metodológica general**, transferible más allá del VaR: cuando se estima cualquier cosa sobre la cola de una distribución, **el resultado no es un número; es un rango**. Reportar un solo número sin explorar la sensibilidad a las elecciones de modelo y de hiperparámetros oculta la parte más importante del resultado.

**No se entra aquí en inferencia bayesiana**, por instrucción explícita. Se registra únicamente que existe una diferencia entre "incertidumbre dentro del modelo" e "incertidumbre sobre qué modelo".

### 19.6 Un matiz sobre el cuantil empírico como cota inferior

**[A]** Tsay ofrece un razonamiento útil: con 9190 observaciones, los cuantiles empíricos de 5% y 1% son *"decent estimates"*, de modo que pueden tratarse como estimaciones conservadoras (cotas inferiores) del VaR verdadero. Bajo esa lectura, *"the approach based on the traditional extreme value theory seems to underestimate the VaR for the daily log returns of IBM stock."*

**[A]** Pero inmediatamente matiza: *"When the tail probability is small (e.g., 0.1%), the empirical quantile is a less reliable estimate of the true quantile. The VaR based on empirical quantiles can no longer serve as a lower bound of the true VaR."*

**[B]** Es decir: incluso el "punto de referencia" que se usa para juzgar a los demás métodos **deja de ser confiable exactamente en la región donde más lo necesitaríamos**. No hay un patrón oro contra el cual comparar.
---

## 20. Peaks/Exceedances over Threshold y GPD

**PRIORIDAD CONCEPTUAL.** No se desarrolla la matemática completa del proceso de Poisson bidimensional, por instrucción explícita.

### 20.1 ¿Qué problema intenta resolver respecto de block maxima?

**[A]** Tsay identifica dos dificultades del enfoque tradicional:

> *"First, the choice of subperiod length $n$ is not clearly defined. Second, the approach is unconditional and, hence, does not take into consideration effects of other explanatory variables."*

**[A]** Para superarlas, la literatura estadística moderna (Davison y Smith, 1990; Smith, 1989) propone un enfoque distinto:

> *"Instead of focusing on the extremes (maximum or minimum), the new approach focuses on **exceedances of the measurement over some high threshold** and the **times at which the exceedances occur**. Thus, this new approach is also referred to as **peaks over thresholds (POT)**."*

**[B] La intuición del cambio, en una frase:** en vez de preguntar "¿cuál fue el peor día de cada mes?", preguntamos "¿qué días superaron este nivel, cuándo ocurrieron, y por cuánto lo superaron?".

**Las tres ventajas conceptuales [A]:**

1. **Aprovecha más observaciones extremas.** Si un mes tuvo cinco días malos, los cinco entran en el análisis, no solo el peor.
2. **Modela cuándo y cuánto.** Los tiempos de ocurrencia $\{t_i\}$ *"provide useful information about the intensity of the occurrence of important 'rare events'... A cluster of $t_i$ indicates a period of large market declines."*
3. **Permite extensiones con variables explicativas** (Sección 22 de este informe).

### 20.2 Threshold, exceedance, excess: tres cosas distintas

Ésta es una distinción de vocabulario que hay que tener perfectamente clara.

**Ejemplo, con threshold $\eta = 2\%$.** Movimientos adversos observados:

$$1.0\%,\quad 1.5\%,\quad 2.1\%,\quad 2.4\%,\quad 5.0\%$$

**Threshold:** el nivel elegido = **2%**. Es un número que nosotros fijamos, no algo que los datos determinen automáticamente.

**Exceedances:** las observaciones que **superan** el threshold:

$$2.1\%,\quad 2.4\%,\quad 5.0\%$$

Son 3 observaciones (de 5). El threshold nos dejó con el 60% de estos datos de ejemplo; en la práctica, se elige para quedarse con una fracción muy pequeña.

**Excesses over threshold:** cuánto lo superaron, es decir, la diferencia:

$$2.1-2.0=0.1,\qquad 2.4-2.0=0.4,\qquad 5.0-2.0=3.0$$

**[A]** Tsay usa exactamente esta notación: si la $i$-ésima excedencia ocurre en el día $t_i$, el nuevo enfoque se centra en los datos $(t_i,\ r_{t_i}-\eta)$, donde $r_{t_i}-\eta$ es *"the exceedance over the threshold $\eta$"* y $t_i$ es el momento en que ocurre.

*(Nota terminológica [B]: Tsay usa "exceedance" tanto para la observación que supera el threshold como para el monto del exceso, según el contexto. En este informe se usa "excedencia" para el evento y "exceso" para la magnitud $r_{t_i}-\eta$, que es más preciso.)*

**Para posiciones short [A]:** el mismo procedimiento con el signo apropiado. *"Similarly, for a short position, we may choose $\eta=2\%$ and focus on the data $(t_i, r_{t_i}-\eta)$ for which $r_{t_i}\ge\eta$."*

### 20.3 El cambio de pensamiento estadístico

**[A]** Tsay señala que el POT implica *"a fundamental change in statistical thinking"*:

> *"Instead of using the marginal distribution (e.g., the limiting distribution of the minimum or maximum), the new approach employs a **conditional distribution** to handle the magnitude of exceedance **given that** the measurement exceeds a threshold. The chance of exceeding the threshold is governed by a probability law."*

Es decir, se separan **dos preguntas**:

1. **¿Con qué frecuencia se supera el threshold?** — gobernado por un proceso puntual (un proceso de Poisson).
2. **Dado que se superó, ¿por cuánto?** — gobernado por una distribución condicional (la GPD).

**[A] Sobre el proceso de Poisson:** Tsay remite a la Sección 6.9 del libro para su definición formal, que aquí no estudiamos. La explicación mínima suficiente [B]: un proceso de Poisson describe la ocurrencia de eventos en el tiempo mediante un único parámetro, la **intensidad** $\lambda$ (cuántos eventos por unidad de tiempo, en promedio). **[A]** Si $\lambda$ es constante en el tiempo, el proceso es **homogéneo**; si varía en el tiempo, es **no homogéneo**. Esta distinción es exactamente la que separa la Sección 7.7.3 (modelo homogéneo) de la 7.7.6 (modelo con variables explicativas, no homogéneo).

### 20.4 La distribución generalizada de Pareto (GPD)

**[A]** El desarrollo formal parte de la distribución condicional de $r$ dado $r>\eta$:

$$\Pr(r\le x+\eta\mid r>\eta)=\frac{\Pr(r\le x+\eta)-\Pr(r\le\eta)}{1-\Pr(r\le\eta)} \tag{7.29}$$

y, usando la CDF límite $F_*(\cdot)$ de la GEV y la aproximación $e^{-y}\approx1-y$, se llega a:

$$\Pr(r\le x+\eta\mid r>\eta)\approx1-\left[1+\frac{\xi x}{\alpha+\xi(\eta-\beta)}\right]^{-1/\xi} \tag{7.30}$$

**[A]** La distribución con función de distribución acumulada:

$$G_{\xi,\psi(\eta)}(x)=\begin{cases}1-\left[1+\dfrac{\xi x}{\psi(\eta)}\right]^{-1/\xi} & \text{para }\xi\neq0\\ 1-\exp[-x/\psi(\eta)] & \text{para }\xi=0\end{cases} \tag{7.31}$$

se llama la **generalized Pareto distribution (GPD)**.

**Explicación de cada símbolo:**

| Símbolo | Nombre | Qué es | Qué controla |
|---|---|---|---|
| $x$ | exceso | $x = r-\eta$, cuánto se superó el threshold. Siempre $x\ge0$ cuando $\xi\ge0$ | La variable de la distribución |
| $\eta$ | **threshold** | El nivel que elegimos. **No es un parámetro estimado: es una decisión nuestra** | Qué observaciones entran en el análisis |
| $\psi(\eta)$ | **scale** (escala) | $\psi(\eta)=\alpha+\xi(\eta-\beta)$, un número positivo que depende del threshold elegido | **Cuán grandes** son típicamente los excesos |
| $\xi$ | **shape** (forma) | El mismo parámetro de forma de la GEV | **Cómo decae** la probabilidad de excesos muy grandes |

**[A] Resultado central:** *"the conditional distribution of $r$ given $r>\eta$ is well approximated by a GPD with parameters $\xi$ and $\psi(\eta)=\alpha+\xi(\eta-\beta)$."*

Es decir: **la distribución de los excesos sobre un threshold alto es aproximadamente GPD, casi sin importar cuál sea la distribución original**. Ésta es la contrapartida, para el enfoque POT, de lo que la GEV era para block maxima.

### 20.5 Propiedad de estabilidad del threshold

**[A]** Una propiedad importante: si la distribución de excesos sobre un threshold $\eta_o$ es GPD con forma $\xi$ y escala $\psi(\eta_o)$, entonces para **cualquier threshold mayor** $\eta>\eta_o$, la distribución de excesos también es GPD, con **la misma forma $\xi$** y escala:

$$\psi(\eta)=\psi(\eta_o)+\xi(\eta-\eta_o)$$

**[B] Por qué esto importa:** significa que, **si el modelo es correcto**, el parámetro de forma estimado debería ser **estable** al variar el threshold (por encima de un cierto nivel). Y por lo tanto, observar que $\xi$ **cambia mucho** con el threshold es evidencia de que el modelo no está describiendo bien la cola en ese rango.

**[A] Y eso es exactamente lo que ocurre con IBM en la Tabla 7.3:** $\xi = 0.307$, $0.264$, $0.188$ para thresholds de 3.0%, 2.5% y 2.0%. **No es estable.** Se registra como hipótesis H7.6.

### 20.6 El caso $\xi=0$ y el QQ plot exponencial

**[A]** Cuando $\xi=0$, la GPD se reduce a una **distribución exponencial**. *"This result motivates the use of a QQ plot of excess returns over a threshold against exponential distribution to infer the tail behavior of the returns. If $\xi=0$, then the QQ plot should be linear."*

**[A] Resultado empírico:** la Figura 7.6(a) de Tsay muestra el QQ plot de los negativos de los log returns diarios de IBM con threshold 0.025. *"The nonlinear feature of the plot clearly shows that the left tail of the daily IBM log returns is heavier than that of a normal distribution, that is, $\xi\neq0$."*

**[B]** Éste es un diagnóstico visual muy simple y directamente aplicable a cualquier serie: graficar los excesos ordenados contra los cuantiles de una exponencial. Si la curva se dobla hacia arriba, la cola es más pesada que exponencial.

### 20.7 El modelo de Poisson bidimensional — solo lo esencial

**[A]** Smith (1989) propone modelar conjuntamente $(t_i, r_{t_i})$ como un **proceso de Poisson bidimensional**, enfoque que Tsay (1999) usó para VaR. La medida de intensidad es:

$$\Lambda[(D_2,D_1)\times(r,\infty)]=\frac{D_2-D_1}{D}\,S(r;\xi,\alpha,\beta) \tag{7.33}$$

con $S(r;\xi,\alpha,\beta)=\left[1+\dfrac{\xi(r-\beta)}{\alpha}\right]_+^{-1/\xi}$ una función de supervivencia.

**Explicación conceptual, sin entrar en el álgebra:**

- $D$ es el **intervalo de tiempo base**, típicamente un año. **[A]** En Estados Unidos se usa $D=252$ porque hay aproximadamente 252 días de negociación en un año.
- El primer factor, $(D_2-D_1)/D$, dice que **la ocurrencia de excedencias es proporcional a la longitud del intervalo de tiempo** considerado.
- El segundo factor, $S(r;\cdot)$, dice que **la probabilidad de superar un nivel $r$ está gobernada por una función de supervivencia** con la misma forma que el exponente de la CDF de la GEV.

**[A]** Tsay muestra que la probabilidad condicional implicada por esta medida de intensidad es exactamente la función de supervivencia de la Ec. (7.30) — *"The relationship between the limiting extreme value distribution in Eq. (7.16) and the intensity measure in Eq. (7.33) directly connects the new approach of extreme value theory to the traditional one."*

**Los parámetros se estiman maximizando el logaritmo de una función de verosimilitud (Ec. 7.35)**, que no reproducimos aquí. **[A]** Nota práctica: como el parámetro de escala $\alpha$ es no negativo, se estima $\ln(\alpha)$ en la práctica.

**El álgebra detallada del proceso de Poisson bidimensional se omite por instrucción explícita.**

### 20.8 Ejemplo empírico: Ejemplo 7.7 y Tabla 7.3

**[A] Serie:** IBM, log returns diarios, 3 jul 1962 – 31 dic 1998, 9190 obs. **Variable:** negativos de los log returns (posición long). **Método:** proceso de Poisson bidimensional homogéneo, MLE. **Intervalo base:** $D=252$.

| Threshold | Excedencias | Forma $\xi$ | $\ln(\alpha)$ | Ubicación $\beta$ |
|---|---|---|---|---|
| **Log returns originales** | | | | |
| 3.0% | 175 | 0.30697 (0.09015) | 0.30699 (0.12380) | 4.69204 (0.19058) |
| 2.5% | 310 | 0.26418 (0.06501) | 0.31529 (0.11277) | 4.74062 (0.18041) |
| 2.0% | 554 | 0.18751 (0.04394) | 0.27655 (0.09867) | 4.81003 (0.17209) |
| **Removiendo la media muestral** | | | | |
| 3.0% | 184 | 0.30516 (0.08824) | 0.30807 (0.12395) | 4.73804 (0.19151) |
| 2.5% | 334 | 0.28179 (0.06737) | 0.31968 (0.12065) | 4.76808 (0.18533) |
| 2.0% | 590 | 0.19260 (0.04357) | 0.27917 (0.09913) | 4.84859 (0.17255) |

*(Errores estándar entre paréntesis.)*

**[A] Observaciones de Tsay:**

- *"the chance of dropping 2.5% or more in a day for IBM stock occurred with probability $310/9190\approx3.4\%$."*
- *"removing the sample mean has little impact on the parameter estimates."*
- *"in a real application one needs to check carefully the adequacy of a fitted Poisson model."*

**[B] Observación adicional sobre el trade-off:** nótese la relación entre threshold y precisión. Con $\eta=3.0\%$ hay 175 excedencias y el error estándar de $\xi$ es 0.090; con $\eta=2.0\%$ hay 554 excedencias y el error estándar baja a 0.044 — **la mitad**. Más datos, más precisión. Pero al mismo tiempo el **valor** estimado se movió de 0.307 a 0.188. Más precisión alrededor de un número posiblemente sesgado. Ése es exactamente el trade-off de la Sección 21.

### 20.9 VaR bajo el nuevo enfoque

**[A]** Como el modelo de Poisson bidimensional tiene los mismos parámetros que la GEV, se usa una fórmula análoga a la (7.28):

$$\mathrm{VaR}=\begin{cases}\beta-\dfrac{\alpha}{\xi}\left\{1-\left[-D\ln(1-p)\right]^{-\xi}\right\} & \text{si }\xi\neq0\\\beta-\alpha\ln[-D\ln(1-p)] & \text{si }\xi=0\end{cases} \tag{7.36}$$

donde $D$ es el intervalo base usado en la estimación (252 en EE.UU.).

**[A] Ejemplo 7.8** — IBM, long \$10 millones, 1 día:

| | $\eta=3.0\%$ | $\eta=2.5\%$ | $\eta=2.0\%$ |
|---|---|---|---|
| **Log returns originales** | | | |
| VaR(5%) | \$228.239 | \$219.106 | \$212.981 |
| VaR(1%) | \$359.303 | \$361.119 | \$368.552 |
| **Media removida** | | | |
| VaR(5%) | \$232.094 | \$225.782 | \$217.740 |
| VaR(1%) | \$363.697 | \$364.254 | \$372.372 |

**[A] Comentarios de Tsay:**

- *"As expected, removing the sample mean, which is positive, slightly increases the VaR."*
- *"the VaR is rather stable among the three threshold values used."*
- *"In practice, we recommend that one removes the sample mean first before applying this new approach to VaR calculation."*
- **[A] Discussion:** *"Compared with the VaR of Example 7.6 that uses the traditional extreme value theory, the new approach provides a more stable VaR calculation. The traditional approach is rather sensitive to the choice of the subperiod length $n$."*

**[B] Observación importante y algo sutil:** los **parámetros** ($\xi$) NO son estables entre thresholds (varían de 0.19 a 0.31), pero el **VaR resultante** SÍ es relativamente estable (varía de \$359.303 a \$368.552 al 1%, un 2.6%). Esto ocurre porque los tres parámetros se compensan parcialmente entre sí. **La estabilidad del output no implica estabilidad de los parámetros**, y viceversa. Vale la pena tenerlo presente: chequear la estabilidad de la cantidad que efectivamente nos importa, no solo la de los parámetros intermedios.

### 20.10 Parameterización alternativa y VaR/ES vía GPD

**[A]** La GPD puede parametrizarse directamente por $(\xi, \psi(\eta))$ en vez de $(\xi,\alpha,\beta)$ — es la parametrización que usa el paquete `evir` de R. Para IBM con $\eta=2.5\%$: $\xi=0.26418$ y $\psi(\eta)=\exp(0.31529)+(0.26418)(2.5-4.7406)=0.77873$ (en porcentaje), equivalente a 0.007787 en log returns.

**[A]** Estimando $F(\eta)$ por la CDF empírica, $\hat F(\eta)=(T-N_\eta)/T$, se obtiene una estimación alternativa del cuantil:

$$\mathrm{VaR}_q=\eta-\frac{\psi(\eta)}{\xi}\left\{1-\left[\frac{T}{N_\eta}(1-q)\right]^{-\xi}\right\} \tag{7.37}$$

donde $T$ es el tamaño de muestra y $N_\eta$ el número de excedencias. **[A]** *"This method to VaR calculation is used in R and S-Plus."*

Los resultados numéricos de este método y del ES asociado ya se registraron en la Sección 7.8 de este informe.

**[A] Diagnóstico visual (Figura 7.7):** los gráficos diagnósticos del ajuste GPD a los negativos de IBM muestran *"some minor deviation from a straight line, indicating further improvement is possible."*

---

## 21. Selección de threshold y Mean Excess Function

**Sección destacada.**

### 21.1 El trade-off fundamental

$$\boxed{\text{threshold bajo: posible sesgo} \quad\leftrightarrow\quad \text{threshold alto: alta varianza/incertidumbre}}$$

**Threshold demasiado bajo:**

> Tenemos muchos datos (muchas excedencias), y por lo tanto estimaciones con errores estándar pequeños. **Pero** quizá todavía no estamos realmente "en la cola" — la aproximación GPD es un resultado asintótico válido para thresholds **altos**, y con un threshold bajo estamos incluyendo observaciones del cuerpo de la distribución donde esa aproximación puede no valer. **Riesgo: sesgo.**

**Threshold demasiado alto:**

> Estamos claramente en la cola, la aproximación GPD debería ser buena. **Pero** tenemos poquísimas observaciones. **Riesgo: enorme varianza de estimación.**

**[A] Evidencia numérica directa de ambos lados en la Tabla 7.3:**

| Threshold | Excedencias | $\hat\xi$ | Error estándar de $\hat\xi$ |
|---|---|---|---|
| 2.0% | 554 | 0.188 | 0.044 (bajo) |
| 2.5% | 310 | 0.264 | 0.065 |
| 3.0% | 175 | 0.307 | 0.090 (alto) |

El error estándar **se duplica** al pasar de 2.0% a 3.0%. Y el valor estimado cambia en 63%. Ambos efectos operan simultáneamente en direcciones opuestas.

### 21.2 Cómo presenta Tsay el problema de elegir el threshold

Es importante registrar **exactamente** cómo lo plantea, sin importar terminología moderna no respaldada.

**[A]** Tsay dice:

> *"the new approach does not require the choice of a subperiod length $n$, but it requires the specification of threshold $\eta$. Different choices of the threshold $\eta$ lead to different estimates of the shape parameter... In the literature, some researchers believe that **the choice of $\eta$ is a statistical problem as well as a financial one, and it cannot be determined based purely on statistical theory**. For example, different financial institutions (or investors) have different risk tolerances. As such, they may select different thresholds even for an identical financial position."*

**[A]** Y da tres guías prácticas:

1. *"For the daily log returns of IBM stock considered in this chapter, the calculated VaR is not sensitive to the choice of $\eta$."* (Nótese: **para esta serie concreta**.)
2. *"The choice of threshold $\eta$ also depends on the observed log returns. For a stable return series, $\eta=2.5\%$ may fare well for a long position. For a volatile return series (e.g., daily returns of a dot-com stock), $\eta$ may be as high as 10%."*
3. *"Limited experience shows that $\eta$ can be chosen so that the **number of exceedances is sufficiently large (e.g., about 5% of the sample)**."*

**[A]** Para un estudio más formal, remite a Danielsson y de Vries (1997b).

**[B] Observación sobre el lenguaje de Tsay:** él no usa las palabras "bias" y "variance" para describir este trade-off en esta sección. Presenta el problema como una decisión que combina consideraciones estadísticas y financieras, y ofrece una regla práctica (≈5% de la muestra). La formulación explícita como trade-off sesgo–varianza que se usa en el recuadro de arriba es **[B]**, una lectura estándar en la literatura moderna de EVT que es consistente con lo que Tsay describe pero que él no formula en esos términos.

**[B] Un detalle interesante que vale registrar:** Tsay dice explícitamente que la elección del threshold no es puramente estadística, sino también **financiera** — distintos inversores con distinta tolerancia al riesgo pueden legítimamente elegir thresholds distintos para la misma posición. Es una observación honesta que resiste la tentación de convertir la elección en un procedimiento automático.

### 21.3 Mean Excess Function — primero sin fórmula

**PRIORIDAD ALTA.**

**La pregunta que responde, en palabras:**

> Para cada threshold que podría elegirse: **"De los casos que ya superaron este umbral, ¿cuánto lo superaron en promedio?"**

Es decir: no cuántos lo superaron (eso es el conteo de excedencias), sino **por cuánto**.

**Ejemplo numérico extremadamente sencillo.** Threshold = 2. Observaciones extremas:

$$2.5,\quad 3,\quad 7$$

Excesos sobre el threshold:

$$0.5,\quad 1,\quad 5$$

Mean excess:

$$\frac{0.5+1+5}{3}=\frac{6.5}{3}\approx2.17$$

Es decir: los casos que superaron el nivel 2, lo superaron en promedio por 2.17.

### 21.4 La fórmula y por qué es útil

**[A]** Dado un threshold alto $\eta_o$, si el exceso $r-\eta_o$ sigue una GPD con parámetros $\xi$ y $\psi(\eta_o)$ con $0<\xi<1$, entonces:

$$E(r-\eta_o\mid r>\eta_o)=\frac{\psi(\eta_o)}{1-\xi}$$

**[A]** Y para cualquier $\eta>\eta_o$, la **mean excess function** es:

$$e(\eta)=E(r-\eta\mid r>\eta)=\frac{\psi(\eta_o)+\xi(\eta-\eta_o)}{1-\xi}$$

o, equivalentemente, para $y>0$:

$$e(\eta_o+y)=\frac{\psi(\eta_o)+\xi y}{1-\xi}$$

**[A] El resultado clave:** *"Thus, for a fixed $\xi$, the mean excess function is a **linear function** of $y=\eta-\eta_o$."*

**Por qué esto es útil, en palabras:** si la GPD describe correctamente la cola por encima de cierto nivel $\eta_o$, entonces graficar el mean excess contra el threshold debería producir una **línea recta** a partir de $\eta_o$ hacia la derecha. Si el gráfico se curva por debajo de cierto punto y se endereza a partir de ahí, ese punto de "enderezamiento" es una indicación visual de dónde empieza a valer la aproximación GPD.

**[A] La versión empírica:**

$$e_T(\eta)=\frac{1}{N_\eta}\sum_{i=1}^{N_\eta}(r_{t_i}-\eta) \tag{7.32}$$

donde $N_\eta$ es el número de retornos que superan $\eta$ y $r_{t_i}$ son los valores correspondientes.

**[A]** El gráfico de $e_T(\eta)$ contra $\eta$ se llama **mean excess plot**, *"which should be linear in $\eta$ for $\eta>\eta_o$ under the GPD. The plot is also called **mean residual life plot**."* Comando `meplot` en el paquete `evir`.

### 21.5 Resultado empírico

**[A]** La Figura 7.6(b) muestra el mean excess plot de los negativos de los log returns diarios de IBM (1962–1998). *"It shows that, among others, **a threshold of about 3% is reasonable** for the negative return series."*

**[B] Observación sobre la frase "among others":** el propio Tsay está señalando que el gráfico no apunta a un único valor. Sugiere que 3% es razonable, entre otras posibilidades. Y nótese que en el resto del capítulo, el threshold que efectivamente usa en la mayoría de los cálculos y diagnósticos es **2.5%**, no 3%.

### 21.6 GUARDRAIL: el mean excess plot no es una máquina automática

**No convertir el gráfico en un procedimiento automático de selección de threshold.**

Razones [B]:

1. El gráfico requiere **juicio visual** sobre dónde "empieza a ser lineal". Distintos analistas pueden ver puntos distintos.
2. En la cola, el mean excess empírico se calcula sobre **muy pocas observaciones**, de modo que el extremo derecho del gráfico es extremadamente ruidoso — precisamente donde más querríamos leer información.
3. **[A]** El propio Tsay dice que la elección del threshold *"cannot be determined based purely on statistical theory"* y que involucra consideraciones financieras además de estadísticas.
4. Es un diagnóstico entre varios, no un criterio único: se complementa con el QQ plot exponencial, con la estabilidad de $\hat\xi$ ante distintos thresholds, y con la estabilidad de la cantidad final de interés.

**Se registra la selección de threshold como una PREGUNTA ABIERTA estructural** de todo el enfoque POT, no como un problema resuelto. Ver hipótesis H7.6.

---

## 22. Explanatory Variables en EVT

**MUY IMPORTANTE PARA EL PUENTE HACIA ML.** Ésta es la sección del capítulo que más directamente abre preguntas para el Proyecto IRIS.

### 22.1 Qué propone exactamente Tsay

**[A]** El modelo de Poisson bidimensional de la Sección 7.7.3 es **homogéneo**: los tres parámetros $\xi$, $\alpha$, $\beta$ son constantes en el tiempo. Tsay señala:

> *"In practice, such a model may not be adequate. Furthermore, some explanatory variables are often available that may influence the behavior of the log returns $r_t$. **A nice feature of the new extreme value theory approach to VaR calculation is that it can easily take explanatory variables into consideration.**"*

**[A] La propuesta concreta.** Sea $x_t=(x_{1t},\dots,x_{vt})'$ un vector de $v$ variables explicativas **disponibles antes del momento $t$**. Se postula que los tres parámetros son variables en el tiempo y funciones lineales de esas variables:

$$\xi_t=\gamma_0+\gamma_1x_{1t}+\cdots+\gamma_vx_{vt}\equiv\gamma_0+\gamma'x_t$$
$$\ln(\alpha_t)=\delta_0+\delta_1x_{1t}+\cdots+\delta_vx_{vt}\equiv\delta_0+\delta'x_t \tag{7.39}$$
$$\beta_t=\theta_0+\theta_1x_{1t}+\cdots+\theta_vx_{vt}\equiv\theta_0+\theta'x_t$$

**Explicación en palabras:** en vez de un solo parámetro de forma para toda la muestra, tenemos un parámetro de forma **que cambia día a día** en función de las condiciones observables. Lo mismo con la escala y la ubicación. **Los tres parámetros que describen la cola se vuelven condicionales al estado del mercado.**

**[A]** Detalles:
- Si $\gamma=0$, entonces $\xi_t=\gamma_0$ es invariante en el tiempo. *"Thus, testing the significance of $\gamma$ can provide information about the contribution of the explanatory variables to the shape parameter."*
- Se usa $\ln(\alpha_t)$ para satisfacer la restricción de positividad de la escala.
- *"In Eq. (7.39), we use the same explanatory variables for all three parameters... In an application, different explanatory variables may be used for different parameters."*
- Con parámetros variables en el tiempo, el proceso es un **proceso de Poisson no homogéneo**, con medida de intensidad (7.40) y verosimilitud (7.41).

**[A] Remark conceptualmente valioso:**

> *"The parameterization in Eq. (7.39) is **similar to that of the volatility models of Chapter 3** in the sense that the three parameters are exact functions of the available information at time $t$."*

Es decir: Tsay mismo hace la conexión con GARCH. Igual que $\sigma_t^2$ en un GARCH es una función determinista de la información pasada, aquí $\xi_t$, $\alpha_t$ y $\beta_t$ son funciones deterministas de la información pasada.

### 22.2 La conexión con volatilidad — registrada con precisión

**[A]** Tsay menciona explícitamente qué tipo de variables explicativas tiene en mente:

> *"For asset returns, **the volatility $\sigma_t^2$ of $r_t$ discussed in Chapter 3 is an example of explanatory variables**. Another example of explanatory variables in the U.S. equity markets is an indicator variable denoting the meetings of the Federal Open Market Committee."*

**Ésta es la conexión [A] fundamental de esta sección:**

$$\text{volatilidad actual (Cap. 3)} \longrightarrow \text{forma/escala/ubicación de la cola futura}$$

**PERO — guardrail obligatorio:**

- Esto es una **propuesta de modelización** que Tsay hace, no un resultado demostrado universalmente.
- Tsay **la verifica empíricamente en una sola serie** (IBM diaria, 1962–1998), donde efectivamente resulta significativa.
- **Tsay NO ha demostrado nada de esto para futuros**, ni para datos intradiarios, ni para ningún otro instrumento.

### 22.3 La implicación conceptual central [B]

Ésta es la formulación que más importa para el proyecto:

> **Una feature podría no cambiar mucho la media esperada y, sin embargo, cambiar fuertemente la probabilidad o la severidad de movimientos extremos.**

En el marco del modelo de Tsay, esto es literalmente posible: la volatilidad $\sigma_t$ puede entrar en la ecuación de $\xi_t$, $\alpha_t$ y $\beta_t$ (afectando la cola) sin aparecer en absoluto en ninguna ecuación de la media condicional. Son ecuaciones separadas, con parámetros separados, estimados sobre información distinta.

### 22.4 Media, volatilidad y cuantiles pueden contener información distinta

**Sección especial [B].**

Supongamos dos contextos de mercado A y B. Y supongamos que:

$$E[r\mid A]\approx E[r\mid B]\approx 0$$

Es decir: **la dirección media esperada es la misma** en ambos contextos. Un modelo de media no distinguiría A de B.

Pero supongamos que:

$$Q_{0.01}(r\mid A)=-0.5\%\qquad\text{y}\qquad Q_{0.01}(r\mid B)=-2.0\%$$

**La dirección media no cambió. La cola sí — por un factor de 4.**

**Consecuencia [B]:** una misma variable puede ser:

| | ¿Aporta información? |
|---|---|
| Para predecir la **media** | Posiblemente no |
| Para predecir la **volatilidad** | Posiblemente sí |
| Para predecir un **cuantil moderado** (0.10, 0.90) | Posiblemente sí, posiblemente de forma distinta que para la volatilidad |
| Para predecir un **evento de cola extrema** (0.001) | Posiblemente sí, y posiblemente de forma distinta de todas las anteriores |

$$\boxed{\text{no asumir que estas cuatro tareas tienen la misma señal disponible}}$$

**[B]** Este cuadro no es una conclusión sobre IRIS. Es la enumeración de cuatro **problemas de predicción distintos**, que hasta ahora en este estudio tendíamos a colapsar implícitamente en "predecir el mercado".

### 22.5 Los resultados empíricos de Tsay: Sección 7.7.8

**[A] Serie:** IBM, log returns diarios en porcentaje, mean-corrected ($r_t^o = r_t-\bar r$), 3 jul 1962 – 31 dic 1998, 9190 obs. **Threshold:** 2.5% (334 excedencias) y 3.0% (184 excedencias). **Posición:** long \$10 millones. **Método:** proceso de Poisson bidimensional **no homogéneo**.

**[A] Las cinco variables explicativas consideradas, todas disponibles en $t-1$:**

| Variable | Definición | Qué intenta capturar |
|---|---|---|
| $x_{1t}$ | Indicador de octubre, noviembre y diciembre | Efecto de cuarto trimestre / fin de año |
| $x_{2t}$ | Indicador de que el día anterior tuvo $r_{t-1}^o\le-2.5\%$ | Posibilidad de **pánico vendedor** tras una caída fuerte |
| $x_{3t}$ | Número de días entre $t-1$ y $t-5$ con $\lvert r_{t-i}^o\rvert\ge2.5\%$ | **Medida cualitativa de volatilidad** reciente |
| $x_{4t}$ | Tendencia anual: (año en $t$ − 1961)/38 | Tendencia temporal en el comportamiento de los extremos |
| $x_{5t}$ | $\sigma_t$ de un GARCH(1,1) gaussiano ajustado a $r_t^o$ | **Volatilidad condicional del Capítulo 3** |

**[A]** El GARCH(1,1) usado para $x_{5t}$ es: $\sigma_t^2=0.04565+0.0807a_{t-1}^2+0.9031\sigma_{t-1}^2$. **[A]** Tsay señala que se usan **dos** medidas de volatilidad ($x_{3t}$ y $x_{5t}$) *"to study the effect of market volatility on VaR"*, y que como las correlaciones seriales en $r_t$ son débiles (Ejemplo 7.3), no se modela ninguna estructura ARMA para la media.

**[A] Tabla 7.4 — resultados con threshold 2.5% (334 excedencias):**

| Parámetro | Constante | Coef. de $x_{3t}$ | Coef. de $x_{4t}$ | Coef. de $x_{5t}$ |
|---|---|---|---|---|
| $\beta_t$ (ubicación) | 0.3202 (0.3387) | — | 1.4772 (0.3222) | 2.1991 (0.2450) |
| $\ln(\alpha_t)$ (escala) | −0.8119 (0.1798) | 0.3305 (0.0826) | 1.0324 (0.2619) | — |
| $\xi_t$ (forma) | 0.1805 (0.1290) | 0.2118 (0.0580) | 0.3551 (0.1503) | −0.2602 (0.0461) |

**[A] Con threshold 3.0% (184 excedencias):**

| Parámetro | Constante | Coef. de $x_{4t}$ | Coef. de $x_{5t}$ |
|---|---|---|---|
| $\beta_t$ | 1.1569 (0.4082) | 2.1918 (0.2909) | — |
| $\ln(\alpha_t)$ | −0.0316 (0.1201) | 0.3336 (0.0861) | — |
| $\xi_t$ | 0.6008 (0.1454) | 0.2480 (0.0731) | −0.3175 (0.0685) |

*(La asignación de columnas para $\beta_t$ y $\ln(\alpha_t)$ en la tabla del threshold 2.5% se infiere del texto de Tsay, que indica qué variables afectan a qué parámetros; se marca como lectura **[B]** de la tabla original.)*

**[A] Las cuatro observaciones que hace Tsay (threshold 2.5%):**

1. *"All three parameters of the intensity function depend significantly on the annual time trend. In particular, the shape parameter has a negative annual trend, indicating that the log returns of IBM stock are moving farther away from normality as time passes. Both the location and scale parameters increase over time."*
2. *"Indicators for the fourth quarter, $x_{1t}$, and for panic selling, $x_{2t}$, are **not significant** for all three parameters."*
3. *"The location and shape parameters are positively affected by the volatility of the GARCH(1,1) model; see the coefficients of $x_{5t}$. This is understandable because the variability of log returns increases when the volatility is high. Consequently, the dependence of log returns on the tail index is reduced."*
4. *"The scale and shape parameters depend significantly on the qualitative measure of volatility. Signs of the estimates are also plausible."*

**⚠️ ADVERTENCIA DE LECTURA [B].** Hay dos puntos donde el texto narrativo de Tsay y los números de la Tabla 7.4 **no coinciden claramente**:

- El punto 1 dice que el parámetro de forma tiene una tendencia anual **negativa**, pero el coeficiente de $x_{4t}$ en la fila de $\xi_t$ es **+0.3551** (positivo) para threshold 2.5% y **+0.2480** para threshold 3.0%.
- El punto 3 dice que ubicación y forma son afectados **positivamente** por la volatilidad GARCH, pero el coeficiente de $x_{5t}$ en la fila de $\xi_t$ es **−0.2602** (negativo). La frase siguiente ("la dependencia de los log returns respecto del tail index se reduce") sugiere que Tsay está describiendo un efecto sobre el tail index $1/\xi$, no sobre $\xi$ directamente, lo cual invertiría el signo.

**Se registran ambas cosas: los números de la tabla y la interpretación narrativa del autor.** No se resuelve la discrepancia aquí. **[B]** Y se extrae una lección metodológica: incluso en un texto cuidadosamente revisado, la interpretación de los signos de parámetros de un modelo de cola con reparametrizaciones múltiples ($\xi$ vs $1/\xi$, $\alpha$ vs $\ln\alpha$) es propensa a confusión. Es un motivo adicional para no tomar un signo de coeficiente aislado como evidencia fuerte de nada.

### 22.6 El resultado más interesante: adaptación a las condiciones de mercado

**[A]** Tsay calcula el VaR para dos días consecutivos, y ésta es la parte más ilustrativa de toda la sección.

**Para el 31 de diciembre de 1998**, las variables explicativas valían $x_{3,9190}=0$ (ningún día de los últimos 5 con movimiento ≥2.5%), $x_{4,9190}=0.9737$, $x_{5,9190}=1.9766$. El modelo produce:

$$\xi_{9190}=0.01195,\qquad \ln(\alpha_{9190})=0.19331,\qquad \beta_{9190}=6.105$$

- VaR(5%) = 3.03756% → **\$303.756**
- VaR(1%) = **\$497.425**

**Para el 30 de diciembre de 1998**, con $x_{3,9189}=1$ (un día de los últimos 5 con movimiento ≥2.5%), $x_{4,9189}=0.9737$, $x_{5,9189}=1.8757$:

$$\xi_{9189}=0.2500,\qquad \ln(\alpha_{9189})=0.52385,\qquad \beta_{9189}=5.8834$$

- VaR(5%) = 2.69139% → **\$269.139**
- VaR(1%) = **\$448.323**

**[A] El comentario de Tsay:** *"An advantage of using explanatory variables is that **the parameters are adaptive to the change in market conditions**."*

**[A]** Y su conclusión comparativa: *"The 5% VaR is slightly larger than that of Example 7.3, which uses a Gaussian AR(2)–GARCH(1,1) model. The 1% VaR is larger than that of Case 1 of Example 7.3. Again, as expected, the effect of extreme values (i.e., heavy tails) on VaR is more pronounced when the tail probability used is small."* Y: *"Based on this example, the homogeneous Poisson model shown in Example 7.8 seems to underestimate the VaR."*

**[B] Lo notable de estos dos días.** El parámetro de forma pasó de 0.25 a 0.012 **de un día al siguiente**, únicamente porque cambió el valor de $x_3$ (de 1 a 0) y ligeramente el de $x_5$. Es decir: **la forma estimada de la cola es condicional al estado del mercado y puede cambiar sustancialmente día a día**. Eso es exactamente el tipo de comportamiento que un cuantil incondicional no puede capturar.

**Al mismo tiempo [B]:** que un parámetro cambie tanto ante un cambio pequeño en una variable explicativa también invita a preguntarse cuán estable sería ese modelo fuera de muestra. Tsay no evalúa el modelo fuera de muestra en ningún momento del capítulo.

### 22.7 Qué NO se concluye

- **NO** se concluye que la volatilidad reciente sea una feature seleccionada para IRIS.
- **NO** se concluye que la cola de MNQ dependa de la volatilidad.
- **NO** se concluye que un modelo condicional de cola sea apropiado para el proyecto.
- **[A]** La única conexión directa que puede registrarse como resultado de Tsay es: **en la serie de IBM diaria 1962–1998, con threshold de 2.5%, las dos medidas de volatilidad ($x_3$ y $x_5$) resultaron estadísticamente significativas para los parámetros que describen la cola, mientras que un indicador de estacionalidad de cuarto trimestre y un indicador de "pánico vendedor" del día anterior no lo resultaron.**

Todo lo demás es hipótesis (H7.3, H7.12).

---

## 23. Model Checking

**Estudiar seriamente.**

$$\boxed{\text{un modelo de cola también necesita diagnóstico}}$$

### 23.1 Las tres cosas que hay que verificar

**[A]** Tsay estructura el chequeo del modelo de Poisson bidimensional en tres partes:

> *"Checking an entertained two-dimensional Poisson process model for exceedance times and excesses involves examining **three key features** of the model. The first feature is to verify the adequacy of the **exceedance rate**, the second feature is to examine the **distribution of exceedances**, and the final feature is to check the **independence assumption** of the model."*

### 23.2 Feature 1: Exceedance rate (¿con qué frecuencia se cruza el umbral?)

**Intuición primero.** Si el modelo dice "las excedencias ocurren a tal ritmo", entonces los **tiempos de espera entre excedencias consecutivas** deberían comportarse como corresponde a ese ritmo. Un proceso de Poisson tiene una propiedad muy característica: **los tiempos entre eventos consecutivos son independientes y siguen una distribución exponencial**.

**[A]** Smith y Shively (1995) proponen examinar las duraciones entre excedencias consecutivas. Si el modelo es apropiado, la duración entre la excedencia $i$ y la $(i-1)$ debería seguir una exponencial. Concretamente, con $t_0=0$, se espera que:

$$z_{t_i}=\int_{t_{i-1}}^{t_i}\frac{1}{D}g(\eta;\xi_s,\alpha_s,\beta_s)\,ds,\qquad i=1,2,\dots$$

sean iid con distribución **exponencial estándar**. Como los retornos diarios son observaciones discretas, se usa la versión discretizada:

$$z_{t_i}=\frac{1}{D}\sum_{t=t_{i-1}+1}^{t_i}S(\eta;\xi_t,\alpha_t,\beta_t) \tag{7.42}$$

**[A] Diagnóstico:** QQ plot de los $z_{t_i}$ contra la exponencial estándar. *"If the model is adequate, the QQ plot should show a **straight line through the origin with unit slope**."*

### 23.3 Feature 2: Distribution of excesses (¿de qué tamaño son los excesos?)

**Intuición.** El modelo dice que los excesos sobre el threshold siguen una GPD. Hay que verificar si efectivamente es así.

**[A]** Bajo el modelo, la distribución condicional del exceso $x_t=r_t-\eta$ es GPD con forma $\xi_t$ y escala $\psi_t=\alpha_t+\xi_t(\eta-\beta_t)$. Usando la relación entre la exponencial estándar y la GPD, se define:

$$w_{t_i}=\begin{cases}\dfrac{1}{\xi_{t_i}}\ln\left[1+\xi_{t_i}\dfrac{r_{t_i}-\eta}{\psi_{t_i}}\right] & \text{si }\xi_{t_i}\neq0\\\dfrac{r_{t_i}-\eta}{\psi_{t_i}} & \text{si }\xi_{t_i}=0\end{cases} \tag{7.43}$$

**[A]** *"If the model is adequate, $\{w_{t_i}\}$ are independent and exponentially distributed with mean 1."* Diagnóstico: nuevamente, QQ plot contra la exponencial.

**[B] La lógica común a ambos diagnósticos:** se transforman los datos observados de modo que, **si el modelo fuera correcto**, la transformación debería producir variables con una distribución simple y conocida (exponencial estándar). Luego se verifica visualmente si efectivamente la tienen. Es exactamente la misma lógica de los residuos estandarizados de un GARCH del Capítulo 3.

### 23.4 Feature 3: Independence (¿hay dependencia residual?)

**[A]** *"A simple way to check the independence assumption, after adjusting for the effects of explanatory variables, is to examine the **sample autocorrelation functions** of $z_{t_i}$ and $w_{t_i}$. Under the independence assumption, we expect that both $z_{t_i}$ and $w_{t_i}$ have no serial correlations."*

**[B]** Éste es el diagnóstico más directamente conectado con todo lo que aprendimos en los Capítulos 2 y 3: si después de ajustar el modelo todavía queda autocorrelación en los residuos transformados, hay estructura temporal que el modelo no está capturando.

### 23.5 Resultado empírico: el modelo homogéneo FALLA

**[A]** Éste es un resultado importante y hay que registrarlo con claridad:

> *"We begin by pointing out that the two-dimensional **homogeneous** model of Example 7.7 **needs further refinements because the fitted model fails to pass the model checking statistics** of the previous section."*

**[A] Concretamente:** las funciones de autocorrelación de $z_{t_i}$ y $w_{t_i}$ del modelo homogéneo con threshold 2.5% *"have some significant serial correlations"*. Y el QQ plot de $z_{t_i}$ *"shows some discrepancy"*.

**[A] Y después del refinamiento con variables explicativas:** *"All autocorrelation functions of $z_{t_i}$ and $w_{t_i}$ are within the asymptotic two standard error limits. The QQ plots also show marked improvements as they indicate no model inadequacy. Based on these checking results, **the inhomogeneous model seems adequate**."*

**[B] La lectura metodológica, que es la parte transferible:**

1. El modelo homogéneo producía números perfectamente presentables (VaR de \$219.106, \$361.119, con seis cifras significativas). **Y estaba mal especificado.**
2. Lo que reveló el problema fue el **diagnóstico**, no el output.
3. La corrección —permitir que los parámetros dependan de variables explicativas— **cambió sustancialmente los resultados**: el VaR al 1% pasó de aproximadamente \$361.000 (homogéneo) a \$497.425 (no homogéneo). **Una diferencia de 38%.**
4. **[A]** Tsay concluye que el modelo homogéneo *"seems to underestimate the VaR"*.

$$\boxed{\text{un modelo que produce números plausibles puede estar mal especificado; solo el diagnóstico lo revela}}$$

### 23.6 Lo que Tsay NO hace, y que sería necesario en un contexto de ML [B]

**Guardrail obligatorio, marcado explícitamente como [B]:**

$$\boxed{\text{además del diagnóstico in-sample, un modelo para ML necesitaría validación temporal fuera de muestra}}$$

**[A]** Todos los diagnósticos de la Sección 7.7.7 son **in-sample**: se ajusta el modelo sobre las 9190 observaciones y se verifica si los residuos transformados se comportan como el modelo predice **sobre esas mismas observaciones**. En ningún momento del Capítulo 7 Tsay separa un período de entrenamiento de un período de evaluación, ni evalúa el desempeño predictivo de ningún modelo de VaR fuera de muestra.

**Esto no es una crítica al texto** — es un libro de econometría de series de tiempo financieras, no de Machine Learning aplicado, y el diagnóstico in-sample de la especificación es una práctica estadística estándar y correcta para su propósito.

**Pero para el Proyecto IRIS [B]**, esto significa que:

- Ningún resultado de este capítulo constituye evidencia de capacidad predictiva fuera de muestra de ningún método de cola.
- La afirmación "el modelo no homogéneo es adecuado" significa "ajusta bien a los datos que se usaron para ajustarlo", que es una afirmación mucho más débil que "generaliza a períodos futuros".
- Cualquier uso futuro de estas ideas en IRIS requeriría, además de los diagnósticos de Tsay, una validación con **separación temporal estricta** entre entrenamiento y evaluación, exactamente como se estableció en el estudio del Capítulo 4.

---

## 24. Return Level

**Estudio conceptual.**

### 24.1 Qué es

**[A]** *"Another risk measure based on the extreme values of subperiods is the **return level**. The $g$ $n$-subperiod return level, $L_{n,g}$, is defined as **the level that is exceeded in one out of every $g$ subperiods of length $n$**."*

Formalmente:

$$P(r_{n,i}>L_{n,g})=\frac{1}{g}$$

donde $r_{n,i}$ denota el máximo del subperíodo.

**[A]** *"The subperiod in which the return level is exceeded is called a **stress period**."*

### 24.2 Ejemplo intuitivo

Si trabajamos con bloques mensuales ($n=21$) y $g=12$, el return level es:

> El nivel que se espera que sea superado, aproximadamente, **una vez cada 12 meses** — es decir, aproximadamente una vez al año.

En hidrología (donde el concepto tiene su origen), esto se llamaría "la crecida de período de retorno de un año".

### 24.3 La fórmula

**[A]** Si $n$ es suficientemente grande para que el máximo normalizado siga la GEV:

$$L_{n,g}=\beta_n-\frac{\alpha_n}{\xi_n}\left\{1-\left[-\ln\left(1-\frac{1}{g}\right)\right]^{-\xi_n}\right\}$$

para $\xi_n\neq0$.

**[A]** Y una observación importante: *"Note that this is precisely the quantile of extreme value distribution given in Eq. (7.26) with tail probability $p^*=1/g$, even though we write it in a slightly different way."*

### 24.4 La diferencia crucial con el VaR

**[A]** *"Thus, **return level applies to the subperiod maximum, not to the underlying returns**. This marks the difference between VaR and return level."*

Esto es sutil pero importante:

| | Se refiere a |
|---|---|
| **VaR** | Un cuantil de la distribución de **los retornos individuales** (la Ec. 7.28 hace explícitamente la traducción de máximos de bloque a retornos individuales) |
| **Return level** | Un cuantil de la distribución de **los máximos de bloque** — no se traduce a retornos individuales |

### 24.5 Resultado empírico

**[A]** Para los negativos de los log returns diarios de IBM con subperíodos de 21 días y $g=12$: el return level es **4.4835%**.

**[A]** El output de R proporciona además un intervalo: `[1] 4.177923 4.481976 4.858102` — es decir, un intervalo de confianza aproximado de **4.18% a 4.86%** alrededor del estimado puntual de 4.48%.

**[B] Nótese el detalle:** el intervalo tiene un ancho de aproximadamente ±8% relativo alrededor del punto. Es un buen ejemplo de por qué reportar "4.4835%" con cuatro decimales es engañoso.

### 24.6 GUARDRAIL: qué NO significa "período de retorno"

Ésta es la parte más importante de esta sección, porque el lenguaje de "período de retorno" es notoriamente mal interpretado, dentro y fuera de las finanzas.

Una afirmación del estilo:

> "nivel con período de retorno de 100 años"

**NO significa:**

- ❌ "ocurre exactamente una vez cada 100 años";
- ❌ "si ocurrió este año, no volverá a ocurrir en 99 años";
- ❌ "hay garantía de que ocurrirá dentro de los próximos 100 años".

**SÍ significa (aproximadamente):**

- ✅ "en cada período, la probabilidad de superar ese nivel es de aproximadamente 1/100".

**[B]** La consecuencia contraintuitiva de esto es que **dos eventos de "período de retorno de 100 años" pueden ocurrir en años consecutivos** sin que eso contradiga en absoluto la afirmación probabilística. Si cada año tiene probabilidad 1/100 independientemente, la probabilidad de que ocurra en dos años consecutivos es 1/10.000 — baja, pero perfectamente posible. Y si los eventos **no** son independientes (que es exactamente lo que sugiere el clustering de extremos de la Sección 25), la probabilidad de ocurrencias consecutivas es **mayor** que bajo independencia.

**Es una interpretación probabilística de frecuencia o de riesgo, no una afirmación determinista sobre calendarios.**

**[A]** Nota de precisión: la definición exacta de Tsay es $P(r_{n,i}>L_{n,g})=1/g$, referida al **máximo de subperíodo**. Se sigue esa definición.

---

## 25. Extremal Index y clustering de extremos

**PRIORIDAD CONCEPTUAL ALTA.** Éste es, probablemente, el concepto más interesante del capítulo desde el punto de vista de series de tiempo.

### 25.1 La pregunta

> **¿Los eventos extremos aparecen de forma independiente en el tiempo, o tienden a venir agrupados?**

**[A]** Tsay abre la sección planteando exactamente el problema:

> *"So far our discussions of extreme values are based on the assumption that the data are **iid** random variables. However, **in reality extremal events tend to occur in clusters because of the serial dependence in the data**. For instance, we often observe large returns (both positive and negative) of an asset after some news event."*

**[A]** El concepto que formaliza esto es el **extremal index**, *"which allows one to characterize the relationship between the dependence structure of the data and their extremal behavior."*

### 25.2 El argumento heurístico

**[A]** Tsay lo desarrolla de forma muy accesible antes de la matemática. Supongamos que la dependencia serial de una serie estacionaria $x_i$ decae rápidamente, de modo que $x_i$ y $x_{i+\ell}$ son esencialmente independientes cuando $\ell$ es suficientemente grande. Dividamos los datos en bloques disjuntos de tamaño $k$, con máximos $x_{k,i}$. Entonces:

$$x_{(n)}=\max_{i=1,\dots,g+1}x_{k,i} \tag{7.44}$$

**En palabras:** el máximo de toda la muestra es también el máximo de los máximos de bloque.

**[A]** *"If the block size $k$ is sufficiently large and the block maximum $x_{k,i}$ does not occur near the end of the $i$th block, then $x_{k,i}$ and $x_{k,i+1}$ are sufficiently far apart and essentially independent... Consequently, $\{x_{k,i}\}$ can be regarded as a sample of iid random variables, and the limiting distribution of its maximum... should be the extreme value distribution."*

**Resultado [A]:** bajo condiciones apropiadas, **la distribución límite del máximo de una serie temporal estrictamente estacionaria también es una distribución de valores extremos**.

**[A] Pero — y aquí está el punto:** *"the parameters associated with the limiting distribution, however, will not be the same as those when $\{x_i\}$ are iid random samples because the limiting distribution depends on the marginal distribution of the underlying sequences."*

### 25.3 La condición $D(u_n)$ — solo a nivel conceptual

**[A]** La condición apropiada para que el máximo de una serie estrictamente estacionaria tenga distribución límite de valores extremos fue obtenida por Leadbetter (1974) y se conoce como la **condición $D(u_n)$**.

**Explicación conceptual [B], sin la formulación exacta:** la condición $D(u_n)$ establece, en esencia, que **eventos separados por suficiente distancia temporal deben volverse asintóticamente independientes**. Formalmente, se toman dos conjuntos de índices temporales $A_1$ y $A_2$ separados por al menos $\ell_n$ períodos (donde $\ell_n/n\to0$), y se exige que la probabilidad conjunta de que ambos conjuntos no superen el umbral se aproxime al producto de las probabilidades individuales, con el error tendiendo a cero cuando $n\to\infty$ (Ec. 7.45).

**[A] Es una condición débil.** Tsay lo señala explícitamente: *"The $D(u_n)$ condition looks complicated, but it is relatively weak. For instance, consider Gaussian sequences with autocorrelation $\rho_n$ for lag $n$. The $D(u_n)$ condition is satisfied if $\rho_n\ln(n)\to0$ as $n\to\infty$; see Berman (1964)."*

**[A] Leadbetter's Theorem 1:** si $\{x_i\}$ es estrictamente estacionaria, el máximo normalizado converge a una distribución no degenerada $F_*(\cdot)$, y se cumple $D(u_n)$, entonces $F_*(x)$ **es** una distribución de valores extremos.

**No se profundiza más en $D(u_n)$**, por instrucción explícita. Lo que hay que retener es: **es una condición de dependencia suficientemente débil y separada como para que la teoría asintótica de extremos pueda extenderse de datos iid a ciertas series temporales dependientes.**

### 25.4 El extremal index $\theta$

**[A] Leadbetter's Theorem 2.** Sea $\{\tilde x_i\}$ una secuencia iid con la misma distribución marginal que la serie estacionaria $\{x_i\}$, y sea $\tilde x_{(n)}$ su máximo. Si el máximo normalizado de la secuencia iid converge a $\tilde F_*(x)$, se cumple $D(u_n)$, y el máximo normalizado de la serie dependiente converge, entonces:

$$P\left(\frac{x_{(n)}-\beta_n}{\alpha_n}\le x\right)\to_d F_*(x)=\tilde F_*^{\,\theta}(x)$$

**para alguna constante $\theta\in(0,1]$.**

**[A]** *"The constant $\theta$ is called the **extremal index**."*

**Explicación en palabras:** la distribución límite del máximo de la serie **dependiente** es la distribución límite del máximo de su contrapartida **iid**, elevada a la potencia $\theta$. Como $\theta\le1$ y $\tilde F_*\le1$, elevar a una potencia menor que 1 **aumenta** el valor de la función, lo cual significa que el máximo de la serie dependiente tiende a ser **menor** que el de la serie iid equivalente.

**Rango de valores [A]:** $\theta\in(0,1]$.

**[A] Efecto sobre los parámetros:** Tsay deriva que, para $\xi\neq0$:

$$\xi_*=\xi,\qquad \alpha_*=\alpha\theta^\xi,\qquad \beta_*=\beta-\frac{\alpha(1-\theta^\xi)}{\xi}$$

**Es decir: el parámetro de FORMA no cambia** (es el mismo que el de la secuencia iid); **la escala y la ubicación sí cambian**, ambas afectadas por $\theta$. Para $\xi=0$: $\alpha_*=\alpha$ y $\beta_*=\beta+\alpha\ln(\theta)$.

**[A] Definición formal alternativa:** $\theta$ es el número no negativo tal que, para todo $\tau>0$, existe una sucesión de umbrales $u_n$ con:

$$\lim_{n\to\infty}n[1-F(u_n)]=\tau \tag{7.47}$$
$$\lim_{n\to\infty}P(x_{(n)}\le u_n)=\exp(-\theta\tau) \tag{7.48}$$

**[A]** Y Tsay muestra que, para la secuencia iid correspondiente, el mismo límite es $\exp(-\tau)$ — lo cual *"highlights the role played by the extremal index $\theta$"*.

### 25.5 Interpretación: $\theta\approx1$ vs $\theta<1$

| Valor | Qué sugiere |
|---|---|
| $\theta\approx1$ | El comportamiento extremo de la serie dependiente es **similar** al de una secuencia iid con la misma distribución marginal. Poco o ningún agrupamiento de extremos. |
| $\theta<1$ | El comportamiento extremo **difiere** del de una secuencia iid. Cuanto menor $\theta$, mayor el efecto de la dependencia sobre los extremos. |

**[A] La interpretación de tamaño de cluster.** Uno de los estimadores del extremal index es:

$$\hat\theta_b^{(2)}=\frac{1}{k}\cdot\frac{G(u_n)/g}{N(u_n)/n}=\frac{G(u_n)}{N(u_n)}$$

donde $N(u_n)$ es el número de **excedencias** de la muestra sobre el umbral y $G(u_n)$ es el número de **bloques** que contienen al menos una excedencia. **[A]** Y Tsay señala: *"Based on the results of Hsing et al. (1988), this estimator can also be interpreted as **the reciprocal of the mean cluster size of the limiting compound Poisson process** $N(u_n)$."*

### 25.6 GUARDRAIL: qué NO significa $1/\theta$

**Ésta es una advertencia crítica, y hay que ser muy preciso.**

$$\boxed{1/\theta \neq \text{"el tamaño exacto de todos los clusters"}}$$

Las condiciones exactas de la interpretación son:

1. **[A]** Es una interpretación del **estimador** $\hat\theta_b^{(2)}$ específicamente, no una propiedad general de $\theta$.
2. **[A]** Es el **tamaño MEDIO** de cluster, no el tamaño de cada cluster individual. Los clusters reales varían en tamaño; $1/\theta$ es un promedio.
3. **[A]** Se refiere al **proceso de Poisson compuesto LÍMITE**, no a los clusters observados en una muestra finita. Es un resultado asintótico ($n\to\infty$, $u_n$ creciendo apropiadamente).
4. **[A]** Depende de los resultados de Hsing et al. (1988), bajo sus supuestos.

**Ejemplo de la interpretación incorrecta que hay que evitar:**

> ❌ "$\theta=0.5$ significa que cada cluster contiene exactamente dos eventos."

**Correcto:**

> ✅ "$\theta=0.5$ es compatible, bajo las condiciones asintóticas del resultado de Hsing et al. y para la elección concreta de umbral y tamaño de bloque usada, con un tamaño **medio** de cluster de aproximadamente 2 en el proceso límite. Clusters individuales pueden ser de 1, de 5, o de cualquier otro tamaño."

Similarmente:

> ❌ "$\theta\approx1$ demuestra que la serie completa es independiente."

**Correcto:**

> ✅ "$\theta\approx1$ sugiere que el comportamiento de los **extremos** se parece al de una secuencia iid con la misma marginal. **No dice nada sobre la independencia del resto de la serie.** Una serie puede tener fuerte dependencia en su cuerpo central y $\theta$ cercano a 1."

### 25.7 Métodos de estimación

**[A]** Tsay presenta dos familias de estimadores, *"Each estimation method is associated with an interpretation of the extremal index."*

**Blocks method.** A partir de $\lim_{n\to\infty}\dfrac{\ln P(x_{(n)}\le u_n)}{n\ln F(u_n)}=\theta$ (Ec. 7.49), estimando numerador y denominador con la muestra:

$$\hat\theta_b^{(1)}=\frac{1}{k}\cdot\frac{\ln[1-G(u_n)/g]}{\ln[1-N(u_n)/n]} \tag{7.50}$$

y su aproximación por Taylor $\hat\theta_b^{(2)}=G(u_n)/N(u_n)$.

**Runs method.** Basado en un resultado de O'Brien (1987): $\lim_{n\to\infty}P(x_{(n)}^*\le u_n\mid x_1>u_n)=\theta$, donde $x^*_{(n)}=\max_{2\le i\le s}x_i$. El estimador es:

$$\hat\theta_r^{(3)}=\frac{\sum_{i=1}^{n-k}I(A_{i,n})}{N(u_n)}$$

donde $A_{i,n}=\{x_i>u_n,\ x_{i+1}\le u_n,\dots,x_{i+k}\le u_n\}$ — es decir, **el evento de que una excedencia sea seguida por una racha de $k$ observaciones por debajo del umbral**.

**[B]** La intuición del runs method es directa: si los extremos vinieran aislados, cada excedencia estaría seguida de una racha de calma. Si vienen agrupados, muchas excedencias están seguidas inmediatamente por otras. La proporción de excedencias "aisladas" estima $\theta$.

**[A]** Tsay menciona que existen otros estimadores en la literatura (Beirlant et al., 2004). **No se profundiza en las derivaciones**, por instrucción explícita.

### 25.8 Ejemplo empírico del extremal index de IBM

**Registrado con toda la precisión disponible.**

**[A]**
- **Serie:** retornos logarítmicos diarios **negativos** de la acción de IBM.
- **Frecuencia:** diaria.
- **Período:** 3 de julio de 1962 – 31 de diciembre de 1998.
- **Tamaño de muestra:** 9190.
- **Block size:** $k=10$. **[A]** *"We chose $k=10$ because the daily log returns have weak serial dependence."*
- **Número de bloques:** 919.
- **Threshold:** 0.025 (2.5%).
- **Método:** blocks method, estimador $\hat\theta_b^{(1)}$.
- **Estimación:** $\hat\theta_b^{(1)}\approx\mathbf{0.82}$. **[A]** *"Indeed, a simple direct calculation using $k=10$ and threshold 0.025 gives $\hat\theta_b^{(1)}=0.823$."*

**[A] SENSIBILIDAD — registrada explícitamente por Tsay:**

> *"The plot also shows that the estimate $\hat\theta_b^{(1)}$ of the extremal index **might be sensitive to the choices of threshold and block size $k$**."*

**[A]** El rango vertical del gráfico (Figura 7.10) va de aproximadamente 0.8 a 1.1, es decir, el estimador toma valores en ese rango según el umbral elegido — incluyendo valores **por encima de 1**, lo cual está fuera del rango teórico $(0,1]$ y es un artefacto del estimador en muestras finitas.

**GUARDRAIL — no generalizar este número:**

$$\boxed{\hat\theta\approx0.82 \text{ es un resultado sobre IBM diaria, con } k=10 \text{ y umbral } 2.5\%}$$

**NO se generaliza a:**
- ❌ otras acciones;
- ❌ futuros;
- ❌ otras frecuencias;
- ❌ otros thresholds;
- ❌ otros tamaños de bloque;
- ❌ otros períodos.

### 25.9 VaR para una serie estacionaria

**[A]** El extremal index puede incorporarse al cálculo de VaR. Como $P(x_{(n)}\le u_n)\approx[F(x)]^{n\theta}$, el cuantil $(1-p)$ de $F(x)$ es el cuantil $(1-p)^{n\theta}$ de la distribución límite del máximo, de modo que la Ec. (7.28) se convierte en:

$$\mathrm{VaR}=\begin{cases}\beta_n-\dfrac{\alpha_n}{\xi_n}\left\{1-\left[-n\theta\ln(1-p)\right]^{-\xi_n}\right\} & \text{si }\xi_n\neq0\\\beta_n-\alpha_n\ln[-n\theta\ln(1-p)] & \text{si }\xi_n=0\end{cases} \tag{7.51}$$

**[A] La consecuencia práctica, en las palabras de Tsay:** *"From the formula, **we risk underestimating the VaR if the extremal index is overlooked**."*

**[A] Resultado numérico:** para IBM (negativos, 1962–1998), con $\hat\theta_b^{(1)}=0.823$ y $n=63$ en la estimación de parámetros, el VaR al 1% para el día siguiente pasa a ser **3.2714%**, frente a **3.0497%** del Ejemplo 7.6 cuando el extremal index se ignora.

**Magnitud del efecto: +7.3%.** Ignorar el agrupamiento de extremos subestima el VaR en aproximadamente 7% en este caso concreto.

### 25.10 Extremal index ≠ volatility clustering

**Sección obligatoria.** Están relacionados, pero **no son lo mismo**, y confundirlos es un error conceptual.

| | Definición |
|---|---|
| **Volatility clustering** (Cap. 3) | Los períodos de **magnitud grande** (de cualquier signo) tienden a agruparse en el tiempo. Se manifiesta como autocorrelación positiva en $\lvert r_t\rvert$ o $r_t^2$. Es una propiedad de **toda la serie**, no solo de sus extremos. |
| **Extremal clustering** (Cap. 7) | Las observaciones que **superan un umbral extremo específico** tienden a agruparse. Se cuantifica mediante el extremal index. Es una propiedad **exclusivamente de la cola**, condicional a un umbral elegido. |

**Por qué pueden estar relacionados [A]/[B]:**

**[A]** Tsay conecta ambos: los extremos tienden a ocurrir en clusters *"because of the serial dependence in the data"*, y en la Sección 7.6 señala que *"the use of subperiod maxima overlooks the fact of volatility clustering in the daily log returns"*. Es decir, reconoce que el volatility clustering es un mecanismo plausible detrás del clustering de extremos.

**[B]** El razonamiento intuitivo es claro: si la volatilidad está agrupada (períodos de alta volatilidad seguidos de más alta volatilidad), entonces durante esos períodos hay mayor probabilidad de superar cualquier umbral fijo. Los extremos aparecerían agrupados **como consecuencia** del agrupamiento de volatilidad.

**Por qué NO son equivalentes [B]:**

1. **Distinto objeto de medición.** El volatility clustering se mide sobre la serie completa (ACF de $r_t^2$); el extremal clustering se mide solo sobre las excedencias de un umbral.
2. **El extremal index depende del umbral y del tamaño de bloque**; la autocorrelación de $r_t^2$ no depende de ninguna elección de umbral.
3. **Podría haber volatility clustering sin que eso implique un extremal index sustancialmente menor que 1**, si los extremos, dentro de los períodos de alta volatilidad, ocurrieran de forma suficientemente dispersa.
4. **Podría haber clustering de extremos por mecanismos distintos del volatility clustering** — por ejemplo, por un evento de mercado que produce varios días consecutivos de movimientos grandes por razones que un modelo GARCH no captura, o por efectos de microestructura o de datos (ver Sección 26.5).

$$\boxed{\text{volatility clustering} \neq \text{extremal clustering, aunque puedan estar relacionados}}$$

**[B] Nota importante y contraintuitiva:** una serie puede tener **ACF de los retornos prácticamente cero** (es decir, sin correlación lineal detectable, lo que a veces se resume erróneamente como "los retornos son impredecibles / independientes") y, sin embargo, tener extremos claramente agrupados. Los retornos de IBM son un ejemplo: **[A]** Tsay señala que *"the serial correlations in $r_t$ are weak"*, y sin embargo $\hat\theta\approx0.82<1$. **ACF ≈ 0 no implica extremos independientes.** Ésta es exactamente la lección del Capítulo 2 (autocorrelación ≠ independencia), reapareciendo en el dominio de las colas.

---

## 26. Adaptación conceptual a barras intradía [B]

**Toda esta sección es [B].** Tsay no analiza datos intradiarios en el Capítulo 7 — todos sus ejemplos son **diarios**, sobre acciones. Ninguna de estas preguntas se resuelve aquí para ningún futuro.

Cada pregunta se clasifica como:
- **CONOCIMIENTO TRANSFERIBLE** — el concepto aplica igual, es un hecho matemático o lógico general.
- **HIPÓTESIS** — es plausible que aplique, pero requiere verificación empírica.
- **PREGUNTA ABIERTA** — no puede determinarse sin evidencia sobre el dataset concreto.

### 26.1 ¿Qué cambia con barras de 1 minuto en vez de retornos diarios?

**Lo que NO cambia — CONOCIMIENTO TRANSFERIBLE:**

- La definición de cuantil, de VaR, de ES, de exceso sobre threshold. Son definiciones matemáticas, independientes de la frecuencia.
- El trade-off de block size y de threshold. Es estructural.
- La aritmética de "muchos datos ≠ muchos extremos" (Sección 11.3).
- El hecho de que el diagnóstico del modelo es necesario.
- La distinción entre cuantil condicional e incondicional.

**Lo que SÍ podría cambiar — HIPÓTESIS y PREGUNTAS ABIERTAS:**

- La forma efectiva de la distribución a esa frecuencia.
- La magnitud del clustering de extremos.
- El peso relativo de efectos de microestructura (Capítulo 5) en lo que se observa como "extremo".
- La existencia de patrones sistemáticos ligados a la hora del día.

### 26.2 ¿Una muestra de millones de barras resuelve automáticamente el problema de eventos raros?

**Clasificación: la parte aritmética es CONOCIMIENTO TRANSFERIBLE; la magnitud del efecto es HIPÓTESIS.**

**Respuesta corta: no.**

Como se estableció en la Sección 11.3, la aritmética es inescapable: con 2 millones de barras, la cola de 0.01% tiene aproximadamente 200 observaciones nominales. **Y ese conteo nominal es una cota superior optimista de la información efectiva**, por dos razones acumulativas:

1. **Dependencia (HIPÓTESIS).** Si los extremos intradiarios están agrupados —lo cual es plausible si el volatility clustering opera también a escala intradía—, varias de esas 200 observaciones pertenecen al mismo episodio de mercado. La información efectiva es menor. **No se ha medido para MNQ.**

2. **Contaminación (HIPÓTESIS).** Como se estableció en el estudio del Capítulo 5, algunos movimientos aparentemente extremos a alta frecuencia podrían provenir de microestructura o de problemas de datos, no de información económica. **No se ha medido para MNQ.**

$$\boxed{\text{millones de barras} \nRightarrow \text{el problema de estimación de colas está resuelto}}$$

### 26.3 ¿Qué ocurre con la dependencia temporal, la estacionalidad y el volatility clustering?

**Dependencia temporal — HIPÓTESIS.** **[A]** Toda la teoría EVT clásica de Tsay asume independencia o dependencia débil (condición $D(u_n)$). La condición es débil pero **no vacía**: exige que la dependencia decaiga suficientemente rápido. **PREGUNTA ABIERTA:** ¿se cumple algo análogo en datos intradiarios de futuros, donde la dependencia de corto plazo podría ser considerablemente más fuerte que en datos diarios?

**Estacionalidad intradía — HIPÓTESIS con base en el Capítulo 5.** El estudio del Capítulo 5 documentó, para acciones de NYSE, un patrón diurno marcado en la actividad de mercado. **[B] PREGUNTA ABIERTA:** si la volatilidad varía sistemáticamente con la hora del día en un futuro, entonces:

- **La distribución de los retornos NO sería estacionaria dentro del día.** Y toda la teoría de este capítulo (incluido el extremal index, que requiere **estricta estacionariedad**) asume estacionariedad.
- Un modelo de cola único para todo el día estaría mezclando regímenes distintos, de la misma forma en que un cuantil incondicional mezcla regímenes de volatilidad (Sección 13.4).

Se registra como hipótesis H7.4.

**Volatility clustering — HIPÓTESIS.** **[A]** Tsay señala explícitamente que el enfoque de block maxima *"overlooks the fact of volatility clustering"*. Si el volatility clustering es más pronunciado a frecuencias intradiarias, este problema sería más severo, no menos. **No verificado.**

### 26.4 ¿Tiene sentido un mismo threshold absoluto en períodos de distinta volatilidad?

**PREGUNTA ABIERTA — y probablemente una de las más importantes de esta sección.**

**[A]** Tsay usa thresholds **absolutos** (2.0%, 2.5%, 3.0%) sobre toda la muestra de 36 años de IBM. Y él mismo señala que la elección apropiada depende de la serie: *"For a stable return series, $\eta=2.5\%$ may fare well... For a volatile return series (e.g., daily returns of a dot-com stock), $\eta$ may be as high as 10%."*

**[B] La pregunta que esto abre:** si un umbral apropiado depende del nivel de volatilidad de la serie, ¿qué pasa cuando la volatilidad **de la misma serie** cambia sustancialmente a lo largo del tiempo, o incluso a lo largo del día?

Un threshold absoluto de, digamos, 0.3% en una barra de 1 minuto:

- En un período de volatilidad muy baja, podría ser cruzado casi nunca → pocas excedencias, estimación imposible.
- En un período de volatilidad muy alta, podría ser cruzado constantemente → las "excedencias" ya no son eventos extremos, son movimientos ordinarios de ese régimen.

**[B] Alternativas conceptuales que esto sugiere** (ninguna adoptada, ninguna respaldada por Tsay):

- Threshold **relativo a la volatilidad reciente** (por ejemplo, $\eta_t = c\cdot\hat\sigma_t$).
- Threshold **relativo a la hora del día**.
- Modelo con parámetros dependientes de variables explicativas (Sección 22), lo cual es lo que Tsay efectivamente hace — pero manteniendo el threshold fijo.

**Nótese que ni siquiera Tsay hace threshold variable:** en su modelo no homogéneo, los **parámetros** varían con las condiciones, pero el **threshold** sigue siendo fijo en 2.5%. La idea de un threshold adaptativo es [B] y no está respaldada por el texto.

### 26.5 Outlier vs extremo real — conectando Capítulos 5 y 7

**Sección obligatoria. [B]**

Una observación de magnitud gigantesca en datos de alta frecuencia podría ser:

1. **Un evento de mercado real.** Una noticia, un shock macroeconómico, una liquidación forzada.
2. **Un efecto de microestructura.** Como se estableció en el estudio del Capítulo 5: bid–ask bounce, un trade ejecutado lejos del mid en un momento de baja liquidez.
3. **Un bad tick / error de datos.** Un precio erróneo registrado por el proveedor.
4. **Un problema de ajuste o de roll.** En futuros, el paso de un contrato al siguiente puede producir un salto de precio que no corresponde a ningún movimiento económico.
5. **Otro artefacto.** Un hueco por interrupción de sesión, un problema de sincronización, etc.

**Los dos guardrails, ambos necesarios:**

$$\boxed{\text{extremo estadístico} \neq \text{dato erróneo}}$$

No todo lo que parece anómalo lo es. Descartar automáticamente las observaciones extremas es **exactamente el error opuesto** al que se quiere evitar: eliminaría precisamente los datos que la teoría de valores extremos necesita, y produciría estimaciones de cola sistemáticamente demasiado optimistas.

$$\boxed{\text{extremo estadístico} \neq \text{evento económico real confirmado}}$$

Y tampoco todo lo que aparece como extremo en los datos es información económica genuina.

**La consecuencia práctica [B]:**

> **Antes de eliminar o de modelar un extremo, debemos saber qué representa.**

Y esto es más difícil de lo que suena, porque en la cola cada observación individual pesa muchísimo. Como se vio en la Sección 17.6, en la muestra de IBM el crash de 1987 es una sola observación que domina visualmente todo el conjunto de extremos. **[B]** En un dataset de barras intradiarias, una decisión de limpieza de datos que afecte a un puñado de observaciones extremas puede cambiar sustancialmente cualquier estimación de cola.

**PREGUNTA ABIERTA para IRIS:** ¿qué proporción de los movimientos aparentemente extremos en los datos de barras de MNQ corresponde a cada una de las cinco categorías anteriores? Se registra como hipótesis H7.13. **No se investiga aquí.**

### 26.6 Resumen de clasificación

| Pregunta | Clasificación |
|---|---|
| Definiciones de cuantil, VaR, ES, threshold, exceso | CONOCIMIENTO TRANSFERIBLE |
| Aritmética "muchos datos ≠ muchos extremos" | CONOCIMIENTO TRANSFERIBLE |
| Trade-offs de block size y threshold | CONOCIMIENTO TRANSFERIBLE |
| Necesidad de diagnóstico del modelo | CONOCIMIENTO TRANSFERIBLE |
| Distinción cuantil condicional/incondicional | CONOCIMIENTO TRANSFERIBLE |
| Invariancia del tail index bajo agregación temporal | HIPÓTESIS (resultado asintótico bajo supuestos que los datos pueden violar) |
| Magnitud del clustering de extremos intradía | HIPÓTESIS |
| Dependencia de la cola respecto de la hora del día | PREGUNTA ABIERTA |
| Validez de un threshold absoluto entre regímenes de volatilidad | PREGUNTA ABIERTA |
| Proporción de "extremos" atribuible a microestructura o datos | PREGUNTA ABIERTA |
| Aplicabilidad de la condición $D(u_n)$ a datos intradiarios | PREGUNTA ABIERTA |

---

## 27. Mapa Capítulos 1 → 7

**[B]** Organización propia.

| Capítulo | Pregunta central |
|---|---|
| **1** | ¿Qué distribución tienen los retornos? (colas pesadas, asimetría, curtosis, cuantiles básicos) |
| **2** | ¿Qué dependencia lineal existe? (y por qué dependencia ≠ predictibilidad ≠ significancia) |
| **3** | ¿Cómo cambia la volatilidad en el tiempo? ($r_t=\mu_t+\sigma_t\epsilon_t$) |
| **4** | ¿Qué estructura no lineal puede existir? (y por qué estructura ≠ predictibilidad OOS ≠ utilidad económica) |
| **5** | ¿Qué parte de lo observado puede provenir del propio mecanismo de negociación? |
| **6** | **DIFERIDO por baja prioridad actual.** |
| **7** | ¿Cómo describimos **cuantiles**, **pérdidas severas** y **eventos extremos**, especialmente cuando la cola no se comporta como una normal simple? |

**El mapa conceptual:**

$$\text{distribución}\rightarrow\text{dependencia}\rightarrow\text{volatilidad}\rightarrow\text{no linealidad}\rightarrow\text{microestructura}\rightarrow\boxed{\text{cuantiles y extremos}}$$

### 27.1 Cómo el Capítulo 7 usa cada capítulo anterior

**Del Capítulo 1** toma los conceptos de cola pesada, asimetría y cuantil, y los **profundiza**: donde el Capítulo 1 decía "los retornos tienen colas más pesadas que la normal", el Capítulo 7 pregunta "¿exactamente cuánto más pesadas, con qué forma funcional, y son iguales las dos colas?". **[A]** La respuesta para IBM: familia Fréchet, $\xi\approx0.30$ por el estimador de Hill, y colas posiblemente asimétricas.

**Del Capítulo 2** toma la lección de que dependencia, predictibilidad y significancia no son equivalentes, y la **extiende al dominio de las colas**: el extremal index muestra que una serie con ACF prácticamente cero puede tener extremos claramente dependientes.

**Del Capítulo 3** toma la descomposición $a_t=\sigma_t\epsilon_t$, que es lo que permite distinguir **dos orígenes** de un extremo observado (la forma de $\epsilon_t$ vs. el nivel de $\sigma_t$), y lo que hace posible el **VaR condicional**. **[A]** Y toma explícitamente la volatilidad condicional como candidata a variable explicativa del modelo de cola (Sección 7.7.6).

**Del Capítulo 4** toma la advertencia central: estructura ≠ predictibilidad fuera de muestra ≠ utilidad económica. Y también la idea de forecasting distribucional, que aquí se refina: no solo "predecir la distribución" sino "predecir puntos específicos de ella".

**Del Capítulo 5** toma la advertencia de que un movimiento observado a alta frecuencia puede contener información económica, microestructura, o errores de datos — lo cual, aplicado a los extremos, produce el guardrail de la Sección 26.5.

### 27.2 Qué agrega el Capítulo 7 que ninguno de los anteriores tenía

**El reconocimiento explícito de que hay preguntas distintas sobre la misma distribución futura, y que responderlas requiere herramientas distintas y enfrenta dificultades distintas.**

Concretamente, agrega:

1. El **cuantil condicional** como objeto de estimación por derecho propio ($Q_\tau(Y\mid X)$).
2. La **quantile regression** como método directo para estimarlo, con su función de pérdida asimétrica.
3. Toda la **teoría de valores extremos**, que modela deliberadamente solo la cola.
4. La **cuantificación del clustering de extremos** (extremal index) como propiedad distinta del volatility clustering.
5. Y, sobre todo, una lección metodológica que ninguno de los capítulos anteriores había necesitado enunciar con tanta fuerza: **cuanto más profundo el objeto que queremos estimar, menos datos tenemos y más importa la elección del modelo.**
---

## 28. Implicancias para futuros — [A] separado de [B]

### 28.1 [A] Lo que Tsay efectivamente muestra

Todo lo siguiente es sobre **acciones estadounidenses, datos diarios**, en los períodos indicados. Nada de esto es sobre futuros.

1. **El VaR es, por definición, un cuantil de la distribución de pérdidas** (Ec. 7.1). No es una pérdida máxima, no dice qué pasa más allá del umbral, y no es sub-aditivo.

2. **La regla $\sqrt{k}$ es consecuencia de supuestos específicos** (media cero + IGARCH(1,1) sin drift + normalidad), y falla cuando cualquiera de ellos falla. Demostrado algebraicamente y verificado numéricamente: para IBM a 15 días, la regla sobreestima en 7%.

3. **Cinco métodos razonables producen VaR sustancialmente distintos sobre los mismos datos** (IBM diaria, 1962–1998, 9190 obs): diferencias de 40–64% entre el mínimo y el máximo según el nivel de cola.

4. **El cuantil empírico no puede extrapolar más allá del peor evento observado**, y su eficiencia se degrada cuando $p$ es chico (Ec. 7.11: la varianza es inversamente proporcional a la densidad al cuadrado en el punto estimado).

5. **La distribución de los log returns diarios de IBM pertenece a la familia Fréchet** ($\xi>0$, estimado ≈0.30 por Hill, ≈0.33 por MLE con bloques trimestrales o mayores para los extremos negativos), **rechazando la normalidad**. Consistente con Longin (1996) sobre un índice del mercado estadounidense.

6. **El parámetro de forma puede diferir entre la cola izquierda y la derecha** de la misma serie. Para IBM, Tsay reporta indicios de una cola izquierda más pesada.

7. **La elección de block size cambia sustancialmente los parámetros estimados:** $\xi$ pasa de 0.197 ($n=21$) a 0.335 ($n=63$) para los extremos negativos de IBM.

8. **La elección de threshold también:** $\xi$ pasa de 0.188 ($\eta=2.0\%$) a 0.307 ($\eta=3.0\%$) en el modelo POT sobre la misma serie.

9. **El enfoque POT produce un VaR más estable ante la elección de threshold** que el enfoque de block maxima ante la elección de $n$, aun cuando los parámetros individuales no sean estables.

10. **Los parámetros que describen la cola pueden hacerse depender de variables explicativas conocidas antes de $t$** (Ec. 7.39), incluyendo la volatilidad condicional del Capítulo 3. Para IBM, dos medidas de volatilidad ($x_3$ cualitativa y $x_5$ GARCH) resultaron significativas; un indicador de cuarto trimestre y un indicador de "pánico vendedor" del día anterior, no.

11. **El modelo de Poisson homogéneo FALLÓ los diagnósticos** sobre IBM (autocorrelación significativa en $z_{t_i}$ y $w_{t_i}$), y el modelo con variables explicativas los pasó. Corregir la especificación cambió el VaR al 1% en aproximadamente 38%.

12. **Los extremos de IBM presentan agrupamiento:** $\hat\theta_b^{(1)}\approx0.823$ con $k=10$ y umbral 2.5%, sensible a ambas elecciones. Ignorar el extremal index subestima el VaR al 1% en aproximadamente 7% en ese caso.

13. **Los retornos de IBM tienen correlación serial débil y, sin embargo, extremos agrupados.** ACF≈0 no implica extremos independientes.

### 28.2 [B] Lecturas propias hacia futuros — todas como hipótesis

- **El conocimiento estructural (definiciones, trade-offs, la aritmética de las colas, la necesidad de diagnóstico) es transferible**, porque son propiedades matemáticas y lógicas, no hallazgos empíricos sobre un activo particular.

- **Ningún número es transferible.** $\xi\approx0.30$, $\theta\approx0.82$, threshold de 2.5%, block size de 21 o 63 días — todos son resultados sobre una acción concreta, a frecuencia diaria, en un período concreto, con estructura institucional propia de la NYSE en 1962–1998. **No se traslada ninguno a MNQ ni a ningún futuro.**

- **La hipótesis más interesante para IRIS** es que la información disponible en los datos podría ser asimétrica entre tareas: poca sobre la dirección media, más sobre la dispersión, y posiblemente distinta sobre los cuantiles y las colas. **[A]** Tsay muestra que, para IBM, la volatilidad reciente informa sobre los parámetros de la cola. **[B]** Que eso ocurra también en un futuro intradiario es una **hipótesis falsable**, registrada como H7.3 y H7.12.

- **La estacionariedad es un supuesto delicado en datos intradiarios.** El extremal index requiere **estricta estacionariedad**; la EVT clásica requiere al menos $D(u_n)$. Si existe un patrón diurno marcado en la volatilidad de un futuro, la serie no sería estacionaria dentro del día, y aplicar estas herramientas sin ajuste sería incorrecto. Hipótesis H7.4.

- **La contaminación de los extremos por microestructura y por problemas de datos es potencialmente más grave a alta frecuencia** que a frecuencia diaria (conectando con el Capítulo 5). Hipótesis H7.13.

### 28.3 Por qué NO se afirma nada de esto sobre MNQ

**[A]** En ningún momento del Capítulo 7 Tsay analiza:

- un contrato de futuros;
- datos intradiarios de ningún instrumento;
- datos posteriores a 2008;
- ningún mercado electrónico de negociación continua.

**Todos los ejemplos empíricos del capítulo son acciones individuales estadounidenses (IBM, Intel) o un tipo de cambio (DEM/USD), a frecuencia diaria, en períodos que van de 1962 a 2008.** Cualquier lectura hacia futuros intradiarios debe tratarse como hipótesis, nunca como hecho demostrado.

---

## 29. Implicancias para Machine Learning [B]

**Todo lo de esta sección es [B].** Ninguna decisión de diseño se adopta.

### 29.1 Conexión con target design — SOLO COMO PREGUNTAS

El capítulo permite formular una lista de **problemas de predicción distintos** sobre el mismo objeto (el retorno futuro). No se diseña ningún target; se enumeran las opciones conceptuales.

**Mean target**

$$E[r_{t+h}\mid X_t]$$

> "¿Cuál es el retorno promedio esperado?" — el objetivo por defecto de una regresión estándar.

**Direction target**

$$P(r_{t+h}>0\mid X_t)$$

> "¿Qué probabilidad hay de que el movimiento sea hacia arriba?" — un problema de clasificación binaria.

**Quantile target**

$$Q_{0.10}(r_{t+h}\mid X_t)$$

> "¿Dónde está el percentil 10 de lo que puede pasar?" — lo que estima una quantile regression.

**Tail-event target**

$$P(r_{t+h}<c\mid X_t)$$

> "¿Qué probabilidad hay de que el movimiento sea peor que un umbral $c$?" — un problema de clasificación binaria sobre un evento raro.

**Distributional target**

$$F(r_{t+h}\mid X_t)$$

> "¿Cuál es toda la distribución condicional?" — el objetivo más ambicioso.

**Son problemas diferentes.** Difieren en:

| | Qué se predice | Función de pérdida natural | Cuántos datos informan efectivamente |
|---|---|---|---|
| Mean | Un número (el centro) | Error cuadrático | Todas las observaciones |
| Direction | Una probabilidad | Log-loss / entropía cruzada | Todas las observaciones |
| Quantile | Un número (un punto de la curva) | Pérdida asimétrica $w_\tau$ | Ponderadas asimétricamente |
| Tail-event | Una probabilidad de evento raro | Log-loss, con clases muy desbalanceadas | **Muy pocas observaciones positivas** |
| Distributional | Una función completa | Varias posibles (CRPS, verosimilitud) | Todas, pero con más parámetros |

**NO se afirma que uno sea mejor. NO se adopta ninguno.** El único punto es que colapsarlos mentalmente en "predecir el mercado" oculta que son problemas estadísticamente distintos, con distinta cantidad de señal disponible y distinta dificultad de evaluación.

### 29.2 Conexión con features — SOLO COMO PREGUNTAS

**La pregunta central:**

> **¿Una feature puede ser predictiva de extremos aunque no sea predictiva de dirección?**

**[A] La única conexión directa que Tsay establece** es la de la Sección 7.7.6: propone permitir que los parámetros del modelo de extremos dependan de variables explicativas conocidas antes de $t$, menciona explícitamente la volatilidad condicional del Capítulo 3 como ejemplo, y verifica empíricamente sobre IBM que dos medidas de volatilidad resultan significativas mientras que un indicador estacional y un indicador de pánico vendedor no lo son.

**[B] Conceptualmente**, las siguientes son categorías de variables que **podrían** informar sobre la cola sin informar sobre la dirección:

- volatilidad reciente;
- rango (High − Low);
- volumen;
- hora del día;
- contexto de mercado más amplio.

**NINGUNA de éstas se declara feature seleccionada para IRIS.** Son categorías de la misma naturaleza que las que Tsay usó para IBM ($x_3$ = conteo de días recientes con movimiento grande, $x_5$ = volatilidad GARCH), adaptadas conceptualmente. Que sean informativas para MNQ es una hipótesis (H7.12), no un hallazgo.

### 29.3 La asimetría de información entre tareas

**[B]** Ésta es la observación más importante de toda la sección, y la que se quiere conservar al terminar el capítulo:

$$\boxed{\text{media}\neq\text{volatilidad}\neq\text{cuantil}\neq\text{cola extrema}}$$

Un mismo conjunto de variables puede contener **cantidades diferentes de información** sobre cada uno de estos objetos.

**La pregunta que queda abierta:**

> ¿Es posible que nuestros datos contengan poca información sobre la dirección media del próximo movimiento, pero sí información estable sobre la dispersión, los cuantiles o la probabilidad de movimientos extremos?

**Ésta es una PREGUNTA ABIERTA, no una conclusión.** Y tiene una contrapartida igual de importante, que se desarrolla en la Sección 29.7: incluso si la respuesta fuera afirmativa, no se seguiría automáticamente que eso sea operativamente útil.

### 29.4 Un cuantil no es una distribución — recordatorio

Ya desarrollado en la Sección 14.4, se repite aquí porque es fácil de olvidar en la práctica: estimar varios cuantiles **no** equivale a haber estimado la distribución condicional. En particular, no dice nada sobre el comportamiento más allá del cuantil más extremo estimado, ni garantiza coherencia entre los cuantiles estimados (quantile crossing).

### 29.5 Calibración de cuantiles [B]

**Toda esta subsección es [B].** Tsay no desarrolla esta terminología ni este tipo de evaluación.

**La idea básica.** Si un modelo dice que cierto valor es el cuantil 10%:

$$Q_{0.10,t}$$

entonces, en una evaluación adecuada, aproximadamente el **10% de las observaciones** debería quedar por debajo de ese valor. Si en la práctica el 25% queda por debajo, el modelo está mal calibrado: sus "cuantiles 10%" no son cuantiles 10%.

Esto se conoce conceptualmente como **coverage** o **calibration** (cobertura o calibración).

**Matices importantes:**

1. **"Aproximadamente" hace mucho trabajo.** Con una muestra finita, la proporción observada fluctúa alrededor del 10% incluso si el modelo es perfecto. Hay que comparar contra una banda de tolerancia, no contra el número exacto.

2. **La dependencia complica la evaluación.** Si las violaciones del cuantil están **agrupadas en el tiempo** (que es exactamente lo que el extremal index de la Sección 25 sugiere que puede ocurrir), entonces las observaciones no son independientes y las bandas de tolerancia habituales, calculadas bajo independencia, serían demasiado estrechas. Un modelo podría parecer bien calibrado en promedio y aun así fallar sistemáticamente durante períodos completos.

3. **Calibración de cuantiles extremos requiere muchísimos datos.** Evaluar la cobertura de un cuantil 0.1% requiere suficientes observaciones para que el conteo esperado de violaciones sea informativo. Con 1000 observaciones de evaluación, se esperaría 1 violación. **No se puede evaluar la calibración de un cuantil 0.1% con 1000 observaciones.**

**Distinción crucial:**

$$\boxed{\text{cuantil bien calibrado} \neq \text{cuantil con buen valor económico}}$$

Un modelo podría estar perfectamente calibrado y ser económicamente inútil (por ejemplo, si sus cuantiles son bien calibrados pero prácticamente constantes, no aportan ninguna información condicional accionable). E inversamente, un modelo mal calibrado podría todavía tener algún contenido informativo aprovechable tras recalibración.

**Son dos propiedades distintas, y hay que evaluarlas por separado.**

### 29.6 No confundir rareza con importancia económica

**[B]**

Un evento con probabilidad:

$$0.1\%$$

puede ser **económicamente gigantesco**. Un movimiento del 5% en contra de una posición apalancada puede ser terminal, aunque ocurra raramente.

Pero también puede haber eventos raros **irrelevantes** para una estrategia concreta. Un evento de probabilidad 0.1% que produce un movimiento adverso pequeño, o que ocurre en un horario en que no hay posición abierta, o cuyo efecto se revierte antes de que haya ninguna consecuencia, es estadísticamente raro y económicamente inconsecuente.

$$\boxed{\text{rareza estadística} \neq \text{importancia económica automáticamente}}$$

**[B] La consecuencia práctica:** identificar que algo es "un evento de cola" no es, por sí mismo, un argumento para modelarlo. La pregunta relevante es qué consecuencia económica concreta tendría, dada la estrategia y el horizonte considerados — y eso no está determinado por su probabilidad.

### 29.7 GUARDRAIL FINAL: modelar la cola ≠ predecir dirección ≠ estrategia rentable

$$\boxed{\text{modelar la cola}\neq\text{predecir dirección}\neq\text{tener una estrategia rentable}}$$

Este guardrail se ha mantenido en cada sección y se cierra aquí explícitamente:

- **Modelar bien la cola** significa: describir correctamente cómo se comporta la distribución en su región extrema. Es una afirmación sobre la **forma de una distribución condicional**.
- **Predecir dirección** significa: anticipar el signo del próximo movimiento. Es una afirmación sobre la **media condicional** (o sobre $P(r>0\mid X)$).
- **Tener una estrategia rentable** significa: que la información disponible, después de descontar costos de transacción, slippage, y el riesgo asumido, produzca un resultado económico positivo de forma estable fuera de muestra.

**Las tres son cosas distintas, y ninguna implica la siguiente.**

Un modelo que estima perfectamente el percentil 1% de la distribución condicional puede no tener absolutamente ninguna información sobre si el próximo movimiento será hacia arriba o hacia abajo. Y un modelo que tuviera ambas cosas todavía tendría que superar los costos operativos para ser útil.

**[A]** Nada en el Capítulo 7 de Tsay aborda la rentabilidad de ninguna estrategia. El capítulo es enteramente sobre **medición de riesgo**, no sobre generación de señales.

$$\boxed{\mathrm{VaR}\neq\text{señal de trading}}$$
$$\boxed{\text{cola pesada}\neq\text{predictibilidad}}$$
$$\boxed{\text{extremo}\neq\text{oportunidad operativa}}$$
$$\boxed{\text{probabilidad condicional}\neq\text{causalidad}}$$
$$\boxed{\text{significancia estadística}\neq\text{utilidad económica}}$$

---

## 30. Hipótesis empíricas para backlog — NO ejecutar

Cada hipótesis incluye: pregunta, datos necesarios, método, resultado que la apoyaría, resultado que la refutaría, y limitaciones. **Ninguna se ejecuta en este informe.**

### H7.1 Tail asymmetry

- **Pregunta:** ¿la cola negativa y la positiva de los retornos de MNQ presentan comportamiento distinto?
- **Datos necesarios:** serie de retornos de MNQ a la frecuencia elegida, de longitud suficiente para tener un número razonable de extremos en cada cola.
- **Método:** estimación separada del parámetro de forma para $r_t$ y $-r_t$, por estimador de Hill (con gráfico de $\hat\xi$ contra $q$) y/o por MLE sobre bloques, replicando la Tabla 7.1/7.2 de Tsay.
- **Apoyaría:** estimados de $\xi$ que difieren entre colas por más que sus errores estándar, de forma estable ante distintas elecciones de $q$ o de block size.
- **Refutaría:** estimados indistinguibles entre colas dentro del error de estimación.
- **Limitaciones:** con muestras finitas, los errores estándar del parámetro de forma son grandes (para IBM, hasta el 70% del estimado con $n=252$). Detectar una diferencia moderada puede requerir muchísimos datos. **[A]** Nótese que la propia Tabla 7.1 de Tsay no muestra una diferencia holgada entre colas.

### H7.2 Conditional quantiles

- **Pregunta:** ¿los cuantiles futuros de MNQ cambian sistemáticamente con información disponible en $t$?
- **Datos necesarios:** serie de retornos de MNQ más un conjunto de variables candidatas construidas solo con información pasada.
- **Método:** comparar la cobertura empírica de un cuantil incondicional (fijo) contra la de un cuantil condicional estimado (por ejemplo, por quantile regression lineal, como en Koenker y Bassett), con **separación temporal estricta** entre estimación y evaluación.
- **Apoyaría:** el cuantil condicional logra mejor cobertura y/o menor pérdida asimétrica fuera de muestra que el incondicional, de forma estable entre subperíodos.
- **Refutaría:** no hay mejora, o la mejora desaparece fuera de muestra.
- **Limitaciones:** la mejora podría deberse enteramente al efecto de escala de la volatilidad (Sección 13.3), en cuyo caso no habría información "nueva" más allá de lo que ya sabemos del Capítulo 3. Convendría incluir un benchmark que sea simplemente "cuantil incondicional escalado por volatilidad estimada".

### H7.3 Volatility conditioning

- **Pregunta:** ¿condicionar por volatilidad reciente cambia materialmente los cuantiles extremos de MNQ?
- **Datos necesarios:** retornos de MNQ + una estimación de volatilidad condicional (GARCH u otra) construida solo con información pasada.
- **Método:** replicar conceptualmente la Sección 7.7.6 de Tsay — permitir que los parámetros de la cola dependan de la volatilidad — y/o comparar cuantiles empíricos calculados por terciles de volatilidad reciente.
- **Apoyaría:** los cuantiles extremos estimados difieren sustancialmente entre regímenes de volatilidad, y el modelo condicional mejora sobre el incondicional fuera de muestra.
- **Refutaría:** los cuantiles estandarizados por volatilidad son esencialmente iguales entre regímenes, indicando que toda la variación de la cola se explica por escala.
- **Limitaciones:** ésta es probablemente la hipótesis con **mayor riesgo de resultado trivial**: es casi seguro que los cuantiles absolutos cambian con la volatilidad (es aritmética de escala). La pregunta no trivial es si cambia la **forma** de la cola, no solo su escala.

### H7.4 Intraday conditional tails

- **Pregunta:** ¿los cuantiles y los extremos de MNQ cambian sistemáticamente según la hora del día?
- **Datos necesarios:** barras intradiarias de MNQ con timestamp, de longitud suficiente para promediar por franja horaria.
- **Método:** cuantiles empíricos y conteos de excedencias sobre un threshold, agregados por hora del día; comparación entre franjas.
- **Apoyaría:** patrón sistemático y estable entre subperíodos, en la línea de lo documentado en el estudio del Capítulo 5 sobre patrones diurnos.
- **Refutaría:** ausencia de patrón, o patrón inestable entre subperíodos.
- **Limitaciones:** si existe tal patrón, **la serie no sería estacionaria dentro del día**, lo cual invalidaría la aplicación directa de la teoría de extremos (que requiere estacionariedad, estricta en el caso del extremal index) sin un ajuste previo. Es decir: **confirmar esta hipótesis complicaría, no simplificaría, todo lo demás.**

### H7.5 Empirical vs parametric quantiles

- **Pregunta:** ¿cuánto difieren, sobre MNQ, los cuantiles obtenidos por cuantil empírico, por distribuciones paramétricas (normal, Student-t) y por métodos EVT?
- **Datos necesarios:** serie de retornos de MNQ.
- **Método:** replicar la comparación de la Sección 7.6.1 de Tsay: calcular el mismo cuantil por los cinco métodos y tabular.
- **Apoyaría / refutaría:** no es una hipótesis binaria; es una **medición de la incertidumbre de modelo** para esta serie concreta. El resultado informativo es la magnitud de la dispersión entre métodos.
- **Limitaciones:** no hay "cuantil verdadero" contra el cual comparar. **[A]** Como señala Tsay, *"there is no true VaR available to compare the accuracy of different approaches"*. El ejercicio mide dispersión, no exactitud.

### H7.6 Threshold stability

- **Pregunta:** ¿las estimaciones EVT sobre MNQ son estables ante thresholds razonables distintos?
- **Datos necesarios:** serie de retornos de MNQ.
- **Método:** estimar $\xi$ y $\psi(\eta)$ para un rango de thresholds; verificar si $\hat\xi$ es aproximadamente constante (como predice la propiedad de estabilidad de la GPD, Sección 20.5); examinar el mean excess plot.
- **Apoyaría (estabilidad):** $\hat\xi$ aproximadamente constante en un rango razonable de thresholds, dentro de sus errores estándar.
- **Refutaría:** $\hat\xi$ varía sistemáticamente con el threshold, como ocurre con IBM (0.188 → 0.307).
- **Limitaciones:** un $\hat\xi$ inestable puede indicar que el threshold es demasiado bajo (aún no estamos en la cola), o que el modelo GPD no describe bien esta cola, o simplemente ruido de estimación. Las tres explicaciones son difíciles de separar.

### H7.7 Subperiod stability

- **Pregunta:** ¿los parámetros de cola de MNQ se mantienen entre distintos períodos temporales?
- **Datos necesarios:** serie de MNQ suficientemente larga para dividir en subperíodos con extremos suficientes en cada uno.
- **Método:** estimar los parámetros de cola por separado en subperíodos consecutivos y comparar, con sus errores estándar.
- **Apoyaría (estabilidad):** parámetros estadísticamente indistinguibles entre subperíodos.
- **Refutaría:** cambios sistemáticos, especialmente si son monótonos (que sugerirían una tendencia, como el término $x_4$ que Tsay incorporó para IBM).
- **Limitaciones:** dividir la muestra reduce el número de extremos en cada mitad, aumentando los errores estándar y reduciendo la potencia para detectar diferencias reales. Existe un conflicto directo entre "tener suficientes subperíodos" y "tener suficientes extremos en cada uno".

### H7.8 Extreme clustering

- **Pregunta:** ¿los eventos extremos de MNQ aparecen agrupados en el tiempo?
- **Datos necesarios:** serie de MNQ con orden temporal preservado.
- **Método:** conteo de excedencias sobre un threshold y análisis de su distribución temporal; comparación con lo esperado bajo un proceso de Poisson homogéneo; ACF de la serie indicadora de excedencia.
- **Apoyaría:** excedencias significativamente más agrupadas que lo esperado bajo independencia.
- **Refutaría:** distribución temporal de excedencias compatible con un proceso de Poisson homogéneo.
- **Limitaciones:** el resultado depende del threshold elegido. Y un agrupamiento observado no distingue entre agrupamiento causado por volatility clustering, por un episodio de mercado único, o por artefactos de datos (Sección 26.5).

### H7.9 Extremal index

- **Pregunta:** ¿el extremal index de MNQ es materialmente menor que 1, y estable ante elecciones razonables de threshold y block size?
- **Datos necesarios:** serie de MNQ.
- **Método:** estimadores de blocks y de runs (Sección 25.7), replicando la Figura 7.10 de Tsay: graficar $\hat\theta$ contra el threshold para varios block sizes.
- **Apoyaría:** $\hat\theta$ consistentemente y sustancialmente menor que 1, estable en un rango de elecciones.
- **Refutaría:** $\hat\theta\approx1$, o estimados tan sensibles a las elecciones que no permiten ninguna conclusión.
- **Limitaciones:** **[A]** Tsay documenta explícitamente que el estimador es sensible al threshold y al block size, y su propio gráfico produce valores por encima de 1 (fuera del rango teórico) para algunos umbrales. Además, requiere **estricta estacionariedad**, que es dudosa en datos intradiarios con patrón diurno (ver H7.4). **Esta hipótesis podría no ser respondible de forma limpia sobre datos intradiarios sin resolver antes H7.4.**

### H7.10 Aggregation / frequency

- **Pregunta:** ¿cómo cambia el comportamiento de las colas de MNQ entre frecuencias (1m, 5m, 10m, etc.)?
- **Datos necesarios:** barras de MNQ agregadas a varias frecuencias, construidas sobre el mismo período temporal.
- **Método:** estimar el parámetro de forma a cada frecuencia, cuantificar su incertidumbre y estudiar su estabilidad ante elecciones razonables de threshold o block size. **[A]** La teoría citada por Tsay establece que el tail index teórico es invariante bajo agregación temporal bajo las condiciones correspondientes.
- **Apoyaría la compatibilidad con la teoría:** estimaciones de $\xi$ entre frecuencias que sean razonablemente compatibles entre sí una vez considerada su incertidumbre estadística y que permanezcan estables ante elecciones metodológicas razonables.
- **Pondría en duda la aplicabilidad de la invariancia:** diferencias sistemáticas, robustas y mayores que la incertidumbre de estimación entre las frecuencias analizadas. Esto podría indicar incumplimiento de los supuestos de la teoría, pero no permitiría identificar automáticamente cuál de ellos falla.
- **Limitaciones:** la invariancia se refiere al parámetro teórico/asintótico, no a la igualdad exacta de estimaciones obtenidas con muestras finitas. Al cambiar la frecuencia también cambian el tamaño de muestra, la dependencia temporal, la importancia de la microestructura y potencialmente la calidad de la aproximación EVT. Por tanto, diferencias entre $\hat\xi$ no constituyen por sí solas una refutación de la teoría.

### H7.11 Quantile calibration

- **Pregunta:** ¿los cuantiles condicionales estimados sobre MNQ consiguen la cobertura esperada fuera de muestra?
- **Datos necesarios:** serie de MNQ con separación temporal estricta entre estimación y evaluación.
- **Método:** contar la proporción de observaciones del período de evaluación que caen por debajo del cuantil estimado; comparar contra el nivel nominal, con bandas de tolerancia apropiadas.
- **Apoyaría:** cobertura empírica cercana al nivel nominal, y sin agrupamiento sistemático de las violaciones.
- **Refutaría:** cobertura sistemáticamente distinta del nivel nominal, o violaciones fuertemente agrupadas en el tiempo.
- **Limitaciones:** **[B]** las bandas de tolerancia calculadas bajo independencia serían inválidas si las violaciones están agrupadas (ver H7.8). Y evaluar cuantiles muy extremos requiere períodos de evaluación enormes. La calibración es necesaria pero **no suficiente** para utilidad (Sección 29.5).

### H7.12 Mean vs tail information

- **Pregunta:** ¿existen variables que aportan poca información a la media pero sí a los cuantiles o a las probabilidades de eventos de cola de MNQ?
- **Datos necesarios:** retornos de MNQ + un conjunto de variables candidatas.
- **Método:** para cada variable candidata, evaluar su contribución a cuatro tareas distintas (media, volatilidad, cuantil, probabilidad de evento de cola) con la métrica apropiada a cada una, con separación temporal estricta.
- **Apoyaría:** existen variables cuya contribución fuera de muestra es despreciable para la media y no despreciable para al menos una de las otras tareas.
- **Refutaría:** la contribución de cada variable es aproximadamente proporcional entre las cuatro tareas, o nula en todas.
- **Limitaciones:** **[B]** riesgo alto de resultado trivial. Es casi seguro que la volatilidad reciente informa sobre la volatilidad futura y por lo tanto sobre los cuantiles absolutos futuros — eso ya lo sabemos del Capítulo 3 y no sería un hallazgo nuevo. La pregunta interesante es si hay información **más allá** del efecto de escala. Requiere un benchmark cuidadoso. **Ésta es la hipótesis conceptualmente más importante del conjunto y también la más fácil de responder mal.**

### H7.13 Microstructure extremes

- **Pregunta:** ¿qué proporción de los movimientos aparentemente extremos de MNQ a alta frecuencia podría estar relacionada con microestructura o con problemas de datos, en vez de con información económica?
- **Datos necesarios:** barras de MNQ; idealmente, contexto adicional (volumen, número de trades si estuviera disponible, fechas de roll de contrato, calendario de eventos).
- **Método:** inspección sistemática de las observaciones más extremas, clasificándolas según las cinco categorías de la Sección 26.5; verificación de si coinciden con roll de contrato, huecos de sesión, o momentos de bajísimo volumen.
- **Apoyaría:** una proporción no despreciable de los extremos coincide con circunstancias sospechosas.
- **Refutaría:** los extremos coinciden mayoritariamente con eventos de mercado identificables.
- **Limitaciones:** **[B]** conecta directamente con el estudio del Capítulo 5. Requiere criterios de clasificación definidos de antemano para no caer en racionalización posterior. Y hay un riesgo real: **eliminar extremos "sospechosos" sesga sistemáticamente las estimaciones de cola hacia abajo.** Cualquier limpieza debe documentarse y su efecto debe medirse.

### H7.14 Multiperiod scaling

- **Pregunta:** ¿la cola de MNQ a varios horizontes es compatible con una regla simple de raíz cuadrada del tiempo, con la regla $\ell^\xi$, o con ninguna de las dos?
- **Datos necesarios:** retornos de MNQ acumulados a varios horizontes.
- **Método:** estimar cuantiles extremos directamente a cada horizonte y compararlos con lo que predirían ambas reglas a partir del horizonte de 1 paso.
- **Apoyaría (una regla):** los cuantiles observados a horizontes largos coinciden con los predichos por esa regla, dentro del error de estimación.
- **Refutaría:** discrepancias sistemáticas con ambas reglas.
- **Limitaciones:** a horizontes largos, el número de observaciones no solapadas se reduce drásticamente. Usar ventanas solapadas introduce dependencia entre observaciones que invalida los errores estándar habituales. **No adoptar ninguna regla temporal para barras de futuros sobre la base de este estudio.**

### H7.15 Economic relevance

- **Pregunta:** si existiera predictibilidad de cola en MNQ, ¿tendría una utilidad operativa o de gestión de riesgo medible?
- **Datos necesarios:** todo lo anterior, más una estimación de costos de transacción y una definición explícita de qué decisión operativa se tomaría con esa información.
- **Método:** no definido aquí. **Requiere primero definir la decisión**, y esa definición es una decisión de diseño que no corresponde a esta fase.
- **Apoyaría / refutaría:** no aplicable hasta que se defina la decisión.
- **Limitaciones:** **[B]** ésta es deliberadamente la hipótesis menos especificada del conjunto, porque especificarla equivaldría a tomar decisiones de diseño que esta fase excluye. Se registra únicamente para dejar constancia de que **ninguna de las catorce hipótesis anteriores, aunque se confirmara, constituiría por sí misma evidencia de utilidad económica.**

---

## 31. Auditoría de las 40 afirmaciones

| # | Afirmación | Veredicto | Explicación |
|---|---|---|---|
| 1 | "VaR al 99% es la pérdida máxima posible." | **INCORRECTA** | El VaR es un **cuantil**: un punto de corte. Por construcción, queda un 1% de probabilidad de perder **más** que él. **[A]** Tsay lo señala explícitamente al discutir el ES. |
| 2 | "VaR dice cuánto perderemos en promedio una vez superado el umbral." | **INCORRECTA** | Eso es exactamente lo que el VaR **no** dice, y es la motivación del Expected Shortfall. **[A]** *"VaR is just a quantile... It does not fully describe the upper tail behavior."* |
| 3 | "Expected Shortfall intenta medir la severidad más allá del VaR." | **CORRECTA** | **[A]** Es precisamente su definición: $\mathrm{ES}_q=E[L\mid L>\mathrm{VaR}_q]$, el valor esperado de la pérdida dado que se superó el umbral. |
| 4 | "Un VaR de 1% significa que exactamente cada 100 períodos habrá una violación." | **INCORRECTA** | Es una afirmación probabilística, no determinista. Las violaciones fluctúan y —peor— pueden **agruparse** en el tiempo (extremal index, Sección 25). Ver también la advertencia sobre "período de retorno" en la Sección 24.6. |
| 5 | "VaR y volatilidad son lo mismo." | **INCORRECTA** | La volatilidad mide **el ancho** de la distribución; el VaR marca **un punto** de una cola. Dos distribuciones con la misma volatilidad pueden tener VaR muy distintos (Sección 5.2). |
| 6 | "Dos distribuciones con igual volatilidad tienen necesariamente el mismo VaR." | **INCORRECTA** | Contraejemplo directo: normal vs Student-t(5) estandarizada, misma varianza, cuantil 99% de 2.33 vs 2.61, y la brecha crece hacia la cola. **[A]** Documentado numéricamente en el Ejemplo 7.3. |
| 7 | "RiskMetrics demuestra que todo riesgo escala con la raíz cuadrada del tiempo." | **INCORRECTA** | **[A]** Tsay: *"The square root of time rule is a consequence of the special model used by RiskMetrics. If either the zero mean assumption or the special IGARCH(1,1) model assumption... fails, then the rule is invalid."* No es una ley de los mercados. |
| 8 | "La regla de raíz cuadrada del tiempo requiere supuestos." | **CORRECTA** | Requiere media condicional cero, IGARCH(1,1) sin drift, y normalidad. **[A]** Tsay lo demuestra algebraicamente y lo verifica numéricamente (Ejemplo 7.3 continuado, error del 7% a 15 días). |
| 9 | "Una distribución normal siempre describe adecuadamente las colas financieras." | **INCORRECTA** | **[A]** El análisis EVT de IBM *"rejects the normality assumption commonly used in practice"*: el parámetro de forma es significativamente distinto de cero, ubicando la distribución en la familia Fréchet. |
| 10 | "Usar Student-t garantiza un VaR correcto." | **INCORRECTA** | Es otra elección de modelo, con sus propios supuestos (incluida la elección de los grados de libertad). **[A]** En el Ejemplo 7.3, la Student-t(5) da un VaR al 5% **menor** que la normal. Que sea distinto no significa que sea correcto. No hay VaR verdadero contra el cual verificar. |
| 11 | "El cuantil empírico no necesita asumir una distribución paramétrica." | **CORRECTA** | **[A]** *"It makes no specific distributional assumption."* **Pero requiere condiciones:** sí asume que **la distribución no cambia** entre el período muestral y el de predicción — un supuesto distinto, no ausente. |
| 12 | "El cuantil empírico puede estimar de forma fiable probabilidades arbitrariamente pequeñas si la muestra total es grande." | **INCORRECTA** | La Ec. (7.11) muestra que la varianza del estimador crece con $1/[f(x_p)]^2$, y $f(x_p)$ es diminuta en la cola profunda. **[A]** Tsay: con $p=0.1\%$ el cuantil empírico *"is a less reliable estimate"*. Ver también la aritmética de la Sección 11.3. |
| 13 | "El histórico permite extrapolar naturalmente pérdidas mayores que cualquier pérdida observada." | **INCORRECTA** | Es exactamente lo contrario. **[A]** *"this assumption implies that the predicted loss cannot be greater than that of the historical loss. It is definitely not so in practice."* El cuantil empírico está acotado por el peor evento observado. |
| 14 | "Quantile regression estima la media condicional." | **INCORRECTA** | Estima un **cuantil** condicional. La diferencia está en la función de pérdida: asimétrica $w_\tau$ en vez de cuadrática. **[A]** Ec. (7.14). |
| 15 | "Quantile regression puede estimar un cuantil condicional." | **CORRECTA** | **[A]** Es precisamente su propósito: *"we are interested in the quantiles of the distribution function of $r_{t+1}$ given $F_t$. Such a quantile is referred to as a regression quantile"* (Koenker y Bassett, 1978). |
| 16 | "Una variable puede no ayudar a predecir la media y sí ayudar a predecir un cuantil." | **CORRECTA** | Es conceptualmente posible y estructuralmente contemplado por el marco: la Ec. (7.39) permite que una variable entre en las ecuaciones de $\xi_t$, $\alpha_t$, $\beta_t$ sin aparecer en ninguna ecuación de la media. **[A]** Verificado empíricamente para IBM: las medidas de volatilidad son significativas para los parámetros de cola. **[B]** Que ocurra en un caso concreto es una pregunta empírica (H7.12). |
| 17 | "Predecir un cuantil equivale a conocer toda la distribución." | **INCORRECTA** | Un cuantil es **un punto** de la CDF. Ni siquiera varios cuantiles determinan la distribución completa; en particular, no dicen nada sobre el comportamiento más allá del cuantil más extremo estimado (Sección 14.4). |
| 18 | "EVT intenta modelar principalmente el centro de la distribución." | **INCORRECTA** | Es exactamente lo opuesto: la EVT descarta deliberadamente el centro. Block maxima conserva un dato por bloque; POT conserva solo las excedencias sobre un threshold alto. |
| 19 | "EVT se concentra en el comportamiento extremo." | **CORRECTA** | **[A]** La teoría fue desarrollada *"for studying rare (or extraordinary) events"*, y toda su maquinaria (GEV, GPD) describe distribuciones límite de máximos o de excesos, no de la distribución completa. |
| 20 | "Block maxima conserva toda la información de la muestra." | **INCORRECTA** | Conserva **un dato por bloque** y descarta el resto. Con 2520 días y bloques mensuales, conserva 120 y descarta 2400 — incluyendo el segundo y tercer peor día de cada bloque, y toda la información sobre agrupamiento dentro del bloque. |
| 21 | "Un bloque más grande siempre mejora la estimación." | **INCORRECTA** | Es un **trade-off**. **[A]** *"For the limiting extreme value distribution to hold, one would prefer a large $n$. But a larger $n$ means a smaller $g$... Therefore, some compromise between the choices of $n$ and $g$ is needed."* Con $n=252$ para IBM, el error estándar del parámetro de forma llega al 70% del estimado. |
| 22 | "Un threshold más alto siempre es mejor." | **INCORRECTA** | Mismo tipo de trade-off. Threshold más alto = más cerca del régimen asintótico pero muchísimos menos datos. **[A]** Para IBM, el error estándar de $\hat\xi$ se duplica al pasar de $\eta=2.0\%$ a $\eta=3.0\%$. **[A]** Tsay ofrece una guía práctica (≈5% de la muestra), no una regla de "cuanto más alto mejor". |
| 23 | "Peaks over threshold utiliza observaciones que exceden un nivel elegido." | **CORRECTA** | **[A]** Es exactamente su definición: *"the new approach focuses on exceedances of the measurement over some high threshold and the times at which the exceedances occur."* |
| 24 | "$\xi>0$ debe aparecer necesariamente en cualquier mercado financiero." | **INCORRECTA** | **[A]** Tsay dice que la familia Fréchet ($\xi>0$) es la de **principal interés** para gestión de riesgo, y la encuentra empíricamente en IBM. Eso no es una ley universal. **[A]** De hecho, en el propio modelo condicional de la Sección 7.7.8, $\xi_{9190}=0.01195$ — prácticamente cero — para un día concreto. $\xi$ debe estimarse en cada cola y cada dataset. |
| 25 | "Una cola pesada implica predictibilidad de eventos extremos." | **INCORRECTA** | Cola pesada es una afirmación sobre **la forma de una distribución** (con qué frecuencia relativa ocurren valores grandes). Predictibilidad es una afirmación sobre **anticipar cuándo** ocurren. Nada en el capítulo conecta ambas. |
| 26 | "Un evento extremo es necesariamente un error de datos." | **INCORRECTA** | Descartar automáticamente los extremos eliminaría precisamente los datos que la EVT necesita y sesgaría toda estimación de cola hacia el optimismo. $\text{extremo estadístico}\neq\text{dato erróneo}$ (Sección 26.5). |
| 27 | "Un error de datos extremo debe modelarse como si fuera un movimiento real." | **INCORRECTA** | El guardrail opera en ambas direcciones: $\text{extremo estadístico}\neq\text{evento económico real confirmado}$. Antes de eliminar **o** de modelar un extremo, hay que saber qué representa (Sección 26.5). |
| 28 | "Volatility clustering puede estar relacionado con clustering de extremos." | **CORRECTA** | **[A]** Tsay conecta ambos: los extremos se agrupan *"because of the serial dependence in the data"*, y señala que el enfoque de block maxima *"overlooks the fact of volatility clustering"*. **REQUIERE CONDICIONES en un punto:** relacionados **no** significa equivalentes; son objetos de medición distintos (Sección 25.10). |
| 29 | "Retornos con ACF cercana a cero tienen extremos independientes." | **INCORRECTA** | Contraejemplo directo en el propio capítulo: **[A]** los retornos de IBM tienen correlación serial débil y, sin embargo, $\hat\theta\approx0.82<1$, indicando agrupamiento de extremos. Es la lección del Capítulo 2 (ACF ≠ independencia) reapareciendo en el dominio de las colas. |
| 30 | "Extremal index ayuda a describir dependencia/clustering de extremos." | **CORRECTA** | **[A]** Es exactamente su propósito: *"the extremal index... allows one to characterize the relationship between the dependence structure of the data and their extremal behavior."* |
| 31 | "$\theta=0.5$ significa que cada cluster contiene exactamente dos eventos." | **INCORRECTA** | **[A]** La interpretación de $1/\theta$ es como **tamaño MEDIO** de cluster del **proceso de Poisson compuesto LÍMITE**, bajo los resultados de Hsing et al. (1988), y aplica específicamente al estimador $\hat\theta_b^{(2)}$. Clusters individuales pueden tener cualquier tamaño (Sección 25.6). |
| 32 | "$\theta\approx1$ demuestra independencia completa de toda la serie." | **INCORRECTA** | $\theta$ describe únicamente el comportamiento de los **extremos** en relación con el de una secuencia iid de igual marginal. No dice nada sobre la dependencia del resto de la serie, que puede ser fuerte. |
| 33 | "Una muestra de un millón de barras implica un millón de observaciones útiles para estimar una cola de 0.01%." | **INCORRECTA** | Una cola de 0.01% de un millón de barras son **100** observaciones nominales. Y si los extremos están agrupados, la información efectiva es **aún menor** (Sección 11.3). $\text{muestra total enorme}\nRightarrow\text{muestra enorme de extremos}$. |
| 34 | "Expected Shortfall es siempre la métrica correcta para cualquier objetivo." | **INCORRECTA** | ES y VaR responden preguntas distintas; ninguna es universalmente correcta. Además, ES requiere estimar **más adentro** de la cola (más incertidumbre), y **[A]** ni siquiera está definido si $\xi\ge1$ (la media de los excesos no existe). Tsay nunca declara superioridad universal. |
| 35 | "Modelar bien riesgo de cola implica poder predecir dirección." | **INCORRECTA** | Guardrail central: $\text{modelar la cola}\neq\text{predecir dirección}$. Describir la forma de una distribución condicional en su región extrema no dice nada sobre el signo del próximo movimiento. |
| 36 | "Un modelo que estima bien el percentil 1% necesariamente genera una estrategia rentable." | **INCORRECTA** | Segundo eslabón del guardrail: $\text{predecir dirección}\neq\text{estrategia rentable}$, y aquí ni siquiera hay predicción de dirección. Una estrategia rentable requiere además superar costos de transacción y sostenerse fuera de muestra. **[A]** Nada en el Capítulo 7 aborda rentabilidad de estrategias. |
| 37 | "La cola izquierda y derecha deben tener la misma forma." | **INCORRECTA** | **[A]** Tsay estima ambas por separado para IBM y reporta indicios de una cola izquierda más pesada. Conecta directamente con la asimetría (skewness) del Capítulo 1. Para posiciones long y short importan colas distintas, y son dos problemas de estimación separados. |
| 38 | "Una feature puede modificar la cola condicional aunque no cambie mucho la media." | **CORRECTA** | Estructuralmente contemplado por la Ec. (7.39), donde los tres parámetros de la cola son funciones de variables explicativas independientemente de cualquier ecuación de la media. **[A]** Verificado para IBM. **[B]** Que ocurra en un caso concreto es pregunta empírica. |
| 39 | "Un modelo EVT bien ajustado in-sample garantiza estabilidad futura." | **INCORRECTA** | **[A]** Todos los diagnósticos de la Sección 7.7.7 son in-sample; Tsay nunca evalúa fuera de muestra en este capítulo. "Adecuado" significa "ajusta bien a los datos usados para ajustarlo", que es mucho más débil que "generaliza". **[B]** Para ML se requeriría validación con separación temporal estricta. |
| 40 | "Los cuantiles condicionales pueden ser útiles para ML sin que eso obligue a adoptarlos como target." | **CORRECTA** | Es precisamente la posición de este informe. Los cuantiles condicionales son un **objeto conceptual** distinto de la media y de la volatilidad, con su propia maquinaria de estimación. Registrar que existen y que podrían contener información distinta **no** implica ninguna decisión de diseño para IRIS. |

**Resumen de la auditoría:** **11 CORRECTAS** (#3, 8, 11, 15, 16, 19, 23, 28, 30, 38, 40) y **29 INCORRECTAS**. Dos de las correctas llevan además un matiz explícito de "requiere condiciones": la **#11**, correcta en que el cuantil empírico no asume una forma distribucional, pero que sí asume estabilidad de la distribución entre el período muestral y el de predicción; y la **#28**, correcta en afirmar una relación entre volatility clustering y clustering de extremos, pero que exige no confundir relación con equivalencia (Sección 25.10).

**Observación transversal [B]:** de las 29 afirmaciones incorrectas, la mayoría comparte un mismo patrón de error — **convertir un resultado condicional en una ley universal**: "$\sqrt{t}$ siempre", "más alto siempre mejor", "necesariamente $\xi>0$", "garantiza", "demuestra". El capítulo entero es, en buena medida, un ejercicio de resistir esa tentación en un dominio donde los datos son escasos y los números parecen precisos.
---

## 32. Preguntas abiertas

### 32.1 Dentro del alcance de este capítulo, sin resolver

**Sobre las colas de MNQ (ninguna investigada):**

- ¿Son las colas izquierda y derecha de MNQ distintas entre sí? (H7.1)
- ¿El parámetro de forma es estable ante distintos thresholds y block sizes? (H7.6)
- ¿Es estable entre subperíodos temporales? (H7.7)
- ¿Cómo cambia entre frecuencias de barra? (H7.10)
- ¿Los extremos aparecen agrupados? ¿Cuánto? (H7.8, H7.9)

**Sobre estacionariedad y estructura intradía:**

- ¿La distribución de los retornos de MNQ es estacionaria dentro del día, o existe un patrón diurno que la haga no estacionaria? (H7.4) **Esta pregunta es lógicamente previa a casi todas las demás**, porque la teoría de extremos requiere estacionariedad (estricta, en el caso del extremal index).
- ¿Tiene sentido un threshold absoluto en una serie cuya volatilidad varía sistemáticamente por hora y por régimen? (Sección 26.4)

**Sobre la naturaleza de los extremos observados:**

- ¿Qué proporción de los movimientos extremos en barras intradiarias corresponde a información económica, a microestructura, a errores de datos, a roll de contrato, o a otros artefactos? (H7.13)
- ¿Cómo debería tratarse cada categoría, sin sesgar sistemáticamente las estimaciones de cola?

**Sobre la asimetría de información entre tareas:**

- ¿Existen variables que informen sobre los cuantiles o las colas de MNQ **más allá** del efecto trivial de escala por volatilidad? (H7.12)
- ¿Los cuantiles condicionales estimados lograrían cobertura adecuada fuera de muestra? (H7.11)

**La pregunta que se quiere conservar al terminar el capítulo:**

> ¿Es posible que nuestros datos contengan poca información sobre la dirección media del próximo movimiento, pero sí información estable sobre la dispersión, los cuantiles o la probabilidad de movimientos extremos?

**Ésta es una PREGUNTA ABIERTA, no una conclusión.**

### 32.2 Preguntas que el capítulo NO puede responder por su propia naturaleza

- **¿Cuál método de estimación de cola es "el correcto"?** **[A]** Tsay explícitamente dice que no hay VaR verdadero contra el cual comparar, y recomienda aplicar varios métodos para conocer el rango.
- **¿Qué threshold elegir?** **[A]** *"cannot be determined based purely on statistical theory"* — involucra consideraciones financieras y tolerancia al riesgo, además de las estadísticas.
- **¿Un modelo de cola generaliza fuera de muestra?** El capítulo no evalúa esto en ningún momento.
- **¿Tiene alguna utilidad económica todo esto para una estrategia?** El capítulo es sobre medición de riesgo, no sobre generación de señales ni sobre rentabilidad.

### 32.3 Diferidas explícitamente

- **Capítulo 6** — **DIFERIDO por baja prioridad actual**, por instrucción explícita. La única dependencia que el Capítulo 7 tiene de él es la definición formal de proceso de Poisson (Sección 6.9 del libro), que se explicó aquí de forma mínima y autocontenida.
- **Inferencia bayesiana** — mencionada como no abordada al discutir la incertidumbre de estimación (Sección 19.4). Tsay remite al Capítulo 12 (MCMC) en otros contextos.
- **Modelos multivariados de cola / dependencia de colas entre instrumentos** — no cubierto por este capítulo más allá de la fórmula de RiskMetrics bajo normalidad multivariada. Cualquier tratamiento serio requeriría el Capítulo 8 y literatura de cópulas, no cubierta por Tsay en este capítulo.

---

## 33. Checklist de conocimientos adquiridos

Verificación de que el informe permite responder cada pregunta **sin matemática avanzada**:

| # | Pregunta | ¿Dónde se responde? |
|---|---|---|
| 1 | ¿Qué es una cola? | §2.1 — la zona alejada del centro de la distribución, donde están los movimientos muy grandes. Hay dos: izquierda (pérdidas) y derecha (ganancias). |
| 2 | ¿Qué es un evento extremo? | §2.3 — un evento de magnitud muy grande, situado lejos en la cola. Distinto conceptualmente de "evento raro" (baja frecuencia), aunque en finanzas suelen coincidir. |
| 3 | ¿Qué significa tail probability? | §2.2 — la probabilidad asignada a la región extrema que estamos mirando. Con 1000 días y tail probability 1%, hablamos de aproximadamente los 10 peores días. |
| 4 | ¿Qué es un cuantil? | §10.2 — un punto de corte que deja una fracción determinada de la probabilidad por debajo. El cuantil 0.05 deja el 5% de las observaciones por debajo. |
| 5 | ¿Qué significa VaR 99%? | §4.3 — bajo el modelo usado, aproximadamente el 99% de las veces la pérdida no debería superar ese valor; queda ~1% de probabilidad más allá. |
| 6 | ¿Qué NO significa? | §4.4 — no es la pérdida máxima; no dice cuánto se pierde si se supera; no garantiza exactamente 1 violación de cada 100; depende completamente del modelo. |
| 7 | ¿Diferencia entre VaR y volatilidad? | §5.1 — la volatilidad mide **cuán ancha** es la distribución; el VaR marca **dónde está un punto** de la cola. Misma volatilidad puede dar VaR distintos. |
| 8 | ¿Diferencia entre VaR y ES? | §7.1 — VaR dice **dónde empieza** la zona mala; ES dice **cuánto se pierde en promedio** una vez dentro de ella. |
| 9 | ¿Por qué importan las heavy tails para VaR? | §8.5 — si la distribución real tiene colas más pesadas que la normal, un VaR calculado bajo normalidad queda demasiado cerca del centro. El efecto crece cuanto menor es la tail probability. |
| 10 | ¿Por qué la regla $\sqrt{t}$ no es universal? | §6.5 — sale de tres supuestos específicos (media cero, IGARCH sin drift, normalidad). Con media no nula, la media se acumula linealmente y la desviación como $\sqrt{k}$: dos velocidades distintas. |
| 11 | ¿Qué es un cuantil empírico? | §10.4 — ordenar todas las observaciones y leer el valor en la posición correspondiente al percentil buscado. |
| 12 | ¿Por qué falla en colas profundas? | §10.4, §11.3 — la varianza del estimador crece con $1/[f(x_p)]^2$, y la densidad en la cola es diminuta. Y aritméticamente: la cola de 0.1% de 100.000 datos son ~100 observaciones. |
| 13 | ¿Qué es un cuantil condicional? | §13.1 — el percentil de la distribución futura **dadas las condiciones actuales**, en vez del percentil de todos los datos mezclados. Cambia con el estado del mercado. |
| 14 | ¿Qué hace quantile regression? | §12 — estima un cuantil condicional en función de variables explicativas, usando una función de pérdida **asimétrica** en vez de una simétrica. |
| 15 | ¿Diferencia entre regresión de media y de cuantil? | §12.2, §12.3 — la de media responde "¿cuál es el valor promedio?"; la de cuantil, "¿dónde está el percentil $\tau$?". La diferencia mecánica está en la función de pérdida. |
| 16 | ¿Un cuantil constituye toda una distribución? | §14.4 — no. Es un punto de la curva. Ni siquiera varios cuantiles determinan la distribución completa. |
| 17 | ¿Qué intenta hacer EVT? | §15.1 — describir directamente el comportamiento de los eventos extremos, en vez de modelar toda la distribución y luego leer un punto de la cola. Analogía: modelar las crecidas del río, no todos los días normales. |
| 18 | ¿Qué es block maxima? | §16.1 — dividir la muestra en bloques (por ejemplo, meses) y quedarse con el peor valor de cada bloque, para tener varias observaciones de "un extremo". |
| 19 | ¿Qué información pierde? | §16.2 — todo excepto un dato por bloque: el segundo y tercer peor día, y toda la información sobre si los días malos estaban agrupados o dispersos dentro del bloque. |
| 20 | ¿Qué es GEV? | §16.4 — la distribución límite del máximo de bloque normalizado. Tiene tres parámetros: ubicación, escala y forma. |
| 21 | ¿Qué significa el shape parameter? | §17.1, §17.2 — gobierna el comportamiento de la cola. $\xi>0$: cola pesada (Fréchet, incluye Student-t). $\xi=0$: decaimiento exponencial (Gumbel, incluye normal). $\xi<0$: cola finita (Weibull). |
| 22 | ¿Qué es un threshold? | §20.2 — un nivel que **nosotros elegimos**, por encima del cual consideramos que una observación es extrema. No lo determinan los datos. |
| 23 | ¿Qué es una exceedance? | §20.2 — una observación que supera el threshold. |
| 24 | ¿Qué es excess over threshold? | §20.2 — **por cuánto** lo supera: la diferencia entre la observación y el threshold. Con threshold 2% y observación 2.4%, el exceso es 0.4. |
| 25 | ¿Qué es GPD? | §20.4 — la distribución que describe aproximadamente los **excesos** sobre un threshold alto, casi sin importar cuál sea la distribución original. Tiene forma ($\xi$) y escala ($\psi(\eta)$). |
| 26 | ¿Por qué elegir threshold es difícil? | §21.1 — trade-off: threshold bajo da muchos datos pero quizá aún no estamos en la cola (sesgo); threshold alto está claramente en la cola pero deja poquísimos datos (varianza). **[A]** Y Tsay señala que además hay una componente financiera, no solo estadística. |
| 27 | ¿Qué es mean excess? | §21.3 — de los casos que superaron el umbral, **por cuánto lo superaron en promedio**. Con excesos 0.5, 1 y 5, el mean excess es 2.17. |
| 28 | ¿Cómo pueden variables explicativas modificar una cola? | §22.1 — permitiendo que los tres parámetros que describen la cola (forma, escala, ubicación) sean funciones de variables conocidas antes de $t$, como la volatilidad reciente. Los parámetros pasan a cambiar día a día. |
| 29 | ¿Qué es extremal clustering? | §25.1, §25.10 — la tendencia de las observaciones que superan un umbral extremo a **aparecer agrupadas en el tiempo**, en vez de dispersas al azar. |
| 30 | ¿Qué es extremal index? | §25.4 — un número $\theta\in(0,1]$ que cuantifica cuánto difiere el comportamiento extremo de una serie dependiente respecto de una secuencia iid con la misma distribución marginal. $\theta\approx1$: parecido a iid. $\theta<1$: efecto de la dependencia. |
| 31 | ¿Por qué extremos dependientes son distintos de iid? | §25.2, §25.9 — porque la distribución límite del máximo cambia (escala y ubicación se modifican por $\theta$), y porque el número **efectivo** de eventos independientes es menor que el conteo nominal. Ignorarlo subestima el VaR. |
| 32 | ¿Por qué millones de barras no eliminan la incertidumbre de las colas? | §11.3, §26.2 — porque la cola de 0.01% de un millón de barras son 100 observaciones; porque el agrupamiento reduce la información efectiva; y porque parte de esos "extremos" podría no ser información económica. |
| 33 | ¿Por qué modelar colas no implica predecir dirección? | §29.7 — modelar la cola describe **la forma de una distribución** en su región extrema; predecir dirección es anticipar **el signo del próximo movimiento**. Son afirmaciones sobre objetos distintos. |
| 34 | ¿Qué podría aportar todo esto conceptualmente a ML? | §14, §22.4, §29 — la idea de que hay **varios problemas de predicción distintos** sobre la misma distribución futura (media, volatilidad, cuantil, evento de cola, distribución completa), que pueden tener cantidades muy distintas de señal disponible en los mismos datos. |
| 35 | ¿Qué NO hemos decidido todavía? | §34 — absolutamente nada. No se adoptó ningún target, feature, loss function, arquitectura, threshold, frecuencia ni protocolo de validación. |

---

## 34. Conclusiones

El Capítulo 7 agrega, a las preguntas que ya sabíamos hacerle a una distribución futura, una familia entera de preguntas nuevas.

Hasta aquí, el estudio se había organizado alrededor de dos objetos: **dónde está el centro** ($E[r_{t+h}\mid X_t]$) y **qué tan ancha es** la distribución ($\mathrm{Var}(r_{t+h}\mid X_t)$). El Capítulo 7 agrega, como mínimo, tres más:

$$Q_\tau(r_{t+h}\mid X_t) \qquad P(r_{t+h}<c\mid X_t) \qquad E[r_{t+h}\mid r_{t+h}<c,\ X_t]$$

"¿Dónde está determinado percentil?", "¿qué probabilidad hay de entrar en cierta cola?", y "si entramos, ¿qué tan severo puede ser?". Y el mensaje que sostiene todo el capítulo es que **éstas son preguntas distintas**, que requieren herramientas distintas, y sobre las cuales un mismo conjunto de datos puede contener cantidades de información muy diferentes.

**Lo que sabemos, con respaldo directo del capítulo [A]:**

Existe un vocabulario preciso y una maquinaria estadística madura para describir el comportamiento de una distribución en su región extrema: cuantiles y estadísticos de orden, quantile regression con su función de pérdida asimétrica, distribuciones límite de máximos (GEV) y de excesos sobre umbral (GPD), y una manera de cuantificar cuánto se agrupan los extremos en el tiempo (el extremal index). Sabemos que la distribución de los log returns diarios de IBM entre 1962 y 1998 pertenece a la familia Fréchet, con parámetro de forma en torno a 0.30, rechazando la normalidad. Sabemos que sus extremos están agrupados ($\hat\theta\approx0.82$ para una elección concreta de umbral y bloque), y que ignorar ese agrupamiento subestima el VaR. Y sabemos que los parámetros que describen esa cola pueden hacerse depender de la volatilidad reciente, y que al hacerlo el modelo pasa los diagnósticos que el modelo homogéneo fallaba.

**Y sabemos, sobre todo, algo incómodo:** cinco métodos perfectamente razonables, aplicados a la misma serie de 9190 observaciones de una de las acciones más estudiadas del mundo, producen respuestas que difieren entre sí en un 40 a 64%. **[A]** Tsay no elige un ganador; recomienda aplicar varios métodos para conocer el rango. Ésa es, probablemente, la lección más honesta y más transferible del capítulo:

$$\boxed{\text{el extremo que queremos estimar es precisamente donde tenemos menos datos}}$$

**Lo que NO sabemos, y este capítulo no permite saber:**

No sabemos nada sobre las colas de MNQ. Tsay no analiza ningún futuro, ningún dato intradiario, ningún mercado electrónico moderno. No sabemos si el patrón diurno haría que la serie sea no estacionaria dentro del día, lo cual invalidaría la aplicación directa de casi toda la maquinaria del capítulo. No sabemos qué proporción de los movimientos aparentemente extremos en barras de 1 minuto corresponde a información económica y qué proporción a microestructura o problemas de datos. No sabemos si un modelo de cola ajustado sobre estos datos generalizaría fuera de muestra, porque Tsay no evalúa ningún modelo fuera de muestra en este capítulo.

**Y no sabemos —ésta es la pregunta que queremos conservar— si nuestros datos contienen poca información sobre la dirección media del próximo movimiento y, sin embargo, información más estable sobre la dispersión, los cuantiles o la probabilidad de movimientos extremos.** Es una pregunta atractiva y podría ser importante. También es exactamente el tipo de pregunta que resulta fácil de responder mal: la mayor parte de la variación de los cuantiles absolutos con el estado del mercado es un efecto trivial de escala por volatilidad, algo que ya sabemos desde el Capítulo 3 y que no constituiría ningún hallazgo nuevo. La pregunta interesante es si hay algo **más allá** de ese efecto de escala, y responderla requiere un diseño experimental cuidadoso que esta fase no aborda.

Los tres guardrails que se mantuvieron a lo largo de todo el informe se cierran aquí:

$$\boxed{\text{modelar la cola}\neq\text{predecir dirección}\neq\text{tener una estrategia rentable}}$$

$$\boxed{\text{un estimador de un evento muy raro puede tener enorme incertidumbre}}$$

$$\boxed{\text{media}\neq\text{volatilidad}\neq\text{cuantil}\neq\text{cola extrema}}$$

La prioridad sigue siendo **comprender antes de decidir**. El Capítulo 7 agrega a esa prioridad una advertencia adicional, que es quizás la más difícil de internalizar: en la región de la distribución que más nos podría importar, el conocimiento disponible es estructuralmente más pobre que en cualquier otra, y ninguna cantidad de datos totales cambia esa aritmética.

---

## 35. Registro de revisión crítica

| Afirmación [B] sensible | Riesgo de sobreinterpretación | Estado |
|---|---|---|
| "Los datos podrían tener más información sobre cuantiles y colas que sobre la media" (§29.3) | **Alto.** Es la idea más atractiva del capítulo para el proyecto y la más fácil de convertir en una convicción no verificada. La mayor parte del efecto podría ser escala trivial por volatilidad, ya conocida desde el Cap. 3. | **PREGUNTA ABIERTA** — registrada como H7.12 con advertencia explícita sobre el riesgo de resultado trivial y la necesidad de un benchmark de escala |
| "Un modelo de ML podría estimar directamente cuantiles condicionales" (§14) | **Alto.** Tsay presenta quantile regression **lineal**; extenderla a modelos no lineales, redes o boosting es desarrollo posterior no atribuible a él. Además, riesgo de convertir una posibilidad conceptual en una propuesta de diseño. | **MANTENER**, con la advertencia explícita $\text{quantile regression de Tsay}\neq\text{neural quantile regression}$ y sin ninguna adopción |
| "Invariancia del tail index bajo agregación temporal aplicable a barras intradiarias" (§15.5, §26.1) | **Medio-alto.** Es un resultado [A] de Feller citado por Tsay, pero derivado bajo supuestos (independencia / dependencia débil) que datos intradiarios podrían violar claramente. | **MATIZAR** — clasificado como HIPÓTESIS, no como conocimiento transferible; registrado como H7.10 con el resultado informativo en ambas direcciones |
| Lectura del trade-off de threshold como "sesgo vs varianza" (§21.1, §21.2) | **Medio.** Tsay no usa esa terminología; presenta la elección como estadística **y** financiera, con una guía práctica (≈5% de la muestra). | **MANTENER**, marcado explícitamente como lectura [B] estándar en la literatura moderna, con la formulación de Tsay citada textualmente al lado |
| Asignación de columnas en la Tabla 7.4 y la discrepancia de signos con el texto narrativo (§22.5) | **Medio.** Se infirió qué variable corresponde a qué coeficiente a partir del texto; y hay dos puntos donde texto y tabla parecen no coincidir en el signo. | **MATIZAR** — se registraron ambos (números y narrativa), se marcó la inferencia como [B], y se extrajo la lección de no tomar signos aislados como evidencia fuerte |
| "Threshold adaptativo a la volatilidad o a la hora del día" (§26.4) | **Medio.** Es una idea razonable que surge naturalmente del texto, pero **Tsay no la hace**: en su modelo no homogéneo los parámetros varían y el threshold sigue fijo. | **PREGUNTA ABIERTA** — marcada explícitamente como no respaldada por el texto |
| Interpretación de $1/\theta$ como tamaño de cluster (§25.5, §25.6) | **Alto.** Es el punto del capítulo más propenso a simplificación errónea ("$\theta=0.5$ ⇒ clusters de 2"). | **MANTENER** con las cuatro condiciones explícitas (estimador específico, tamaño **medio**, proceso **límite**, supuestos de Hsing et al.) y un ejemplo explícito de la interpretación incorrecta |
| "El agrupamiento reduce la información efectiva de la muestra de extremos" (§11.3) | **Medio.** Es una consecuencia razonable del extremal index, pero la cuantificación exacta de "información efectiva" no es algo que Tsay desarrolle. | **MANTENER** como razonamiento [B], sin cuantificación numérica y sin trasladar el $\theta\approx0.82$ de IBM a ningún otro contexto |
| "Millones de barras no resuelven el problema de las colas" (§26.2) | **Bajo** en su parte aritmética (es inescapable); **medio** en su parte sobre dependencia y contaminación, que son hipótesis no verificadas para MNQ. | **MANTENER**, con la parte aritmética marcada como CONOCIMIENTO TRANSFERIBLE y las otras dos como HIPÓTESIS (H7.8, H7.13) |
| "Se requeriría validación temporal fuera de muestra" (§23.6) | **Bajo.** Es coherente con lo establecido en el estudio del Capítulo 4 y con las buenas prácticas de ML, pero **Tsay no lo desarrolla** en este capítulo. | **MANTENER**, marcado explícitamente como [B] y sin presentarlo como crítica al texto |
| Distinción entre volatility clustering y extremal clustering (§25.10) | **Bajo.** La relación entre ambos está sugerida por Tsay; la distinción precisa entre ellos es elaboración propia razonable. | **MANTENER**, con la conexión [A] de Tsay citada y la distinción [B] claramente separada |
| Las cinco categorías de origen de un "extremo" observado (§26.5) | **Medio.** La lista es razonable y coherente con el estudio del Capítulo 5, pero ninguna de las cinco fue verificada para MNQ, ni sus proporciones relativas. | **PREGUNTA ABIERTA** — registrada como H7.13, con la advertencia explícita de que limpiar extremos sesga las estimaciones de cola |
| Quantile crossing como problema de estimar cuantiles por separado (§14.4) | **Bajo.** Es un fenómeno real y conocido, mencionado como cuestión secundaria. | **MANTENER**, marcado como [B] y explícitamente secundario, tal como pide el enunciado |

---

## Informe de cierre

**Secciones estudiadas en profundidad** (prioridad máxima según el enunciado): **7.1 Value at Risk** (definición probabilística completa, VaR como cuantil, los cinco factores del cálculo, los tres Remarks sobre distribución predictiva, sub-aditividad y descripción incompleta de la cola); **7.2.3 Expected Shortfall** (definición, cálculo bajo normalidad, cálculo empírico, cálculo bajo GPD con la condición $0<\xi<1$); **7.4 Quantile Estimation** completa, incluyendo **7.4.1 Quantile and Order Statistics** (con la Ec. 7.11 de varianza asintótica y su lectura como formalización del guardrail central) y **7.4.2 Quantile Regression** (con la función de pérdida asimétrica y la generalización de Koenker–Bassett); los conceptos fundamentales de **7.5 Extreme Value Theory** (degeneración, normalización, GEV, las tres familias, las dos implicaciones); **7.7.2 Mean Excess Function**; la idea central de **7.7.3 Peaks/Exceedances over Threshold**; **7.7.6 Use of Explanatory Variables** (registrada con detalle por ser el puente más directo hacia ML); **7.7.7 Model Checking** (los tres features y el resultado de que el modelo homogéneo falla); e interpretación conceptual completa de **7.8 The Extremal Index**.

**Secciones estudiadas seriamente pero de forma selectiva:** **7.2 RiskMetrics** y **7.2.1 Discussion** (supuestos, derivación de la regla $\sqrt{k}$ y su contraejemplo con media no nula, Ejemplos 7.1 y 7.2); **7.3 Econometric Approach** y **7.3.1 Multiple Periods** (estructura ARMA-GARCH, cuantiles bajo normal y bajo Student-t, Ec. 7.7–7.10, Ejemplo 7.3 con sus dos casos y su continuación a 15 días); **7.3.2 ES under Conditional Normality**; **7.5.2 Empirical Estimation** (MLE, método de regresión, estimadores de Hill y Pickands); **7.5.3 Application to Stock Returns** (Tablas 7.1 y 7.2 completas); **7.6 Extreme Value Approach to VaR** y **7.6.1 Discussion** (Ecs. 7.26–7.28, Ejemplo 7.6, tabla comparativa completa de los cinco métodos a tres niveles de cola); **7.6.2 Multiperiod VaR** (regla $\alpha$-root of time); **7.6.3 Return Level**; **7.7.1 Statistical Theory** (GPD, propiedad de estabilidad del threshold, QQ plot exponencial); **7.7.4 VaR Calculation** (Ejemplo 7.8 completo); **7.7.5 Alternative Parameterization** (Ec. 7.37 y ES bajo GPD); **7.7.8 An Illustration** (las cinco variables explicativas, Tabla 7.4, los dos días consecutivos con parámetros distintos); y los métodos de estimación del extremal index solo en la medida necesaria para comprenderlo.

**Secciones tratadas solo a nivel conceptual, por instrucción explícita:** **7.2.2 Multiple Positions** (registrada la fórmula y su dependencia del supuesto de normalidad multivariada, sin profundizar); el **proceso de Poisson bidimensional** (medida de intensidad y su interpretación, sin el álgebra ni la derivación de la verosimilitud); la condición **$D(u_n)$** (explicada como condición de dependencia débil suficientemente separada, sin la formulación exacta ni demostraciones); los **estimadores del extremal index** (blocks y runs presentados con su intuición, sin derivaciones asintóticas).

**Material omitido, y por qué:** las derivaciones algebraicas completas de la verosimilitud del proceso de Poisson bidimensional (Ecs. 7.34–7.35, 7.40–7.41) — omitidas por instrucción explícita, registrando solo su estructura conceptual; el detalle algebraico de la derivación de $F_*(x)=\tilde F_*^{\,\theta}(x)$ hacia los parámetros $(\xi_*,\alpha_*,\beta_*)$ — se registró el resultado sin los pasos intermedios; las demostraciones asintóticas de consistencia y normalidad de los estimadores de Hill y Pickands — se registraron las varianzas asintóticas sin derivarlas; la formulación matemática exacta de $D(u_n)$ (Ec. 7.45) con sus conjuntos de índices $A_1$, $A_2$ y la sucesión $\ell_n$ — reducida a explicación conceptual; los outputs completos de R/S-Plus — conservados solo los valores numéricos relevantes; los ejercicios 7.1 a 7.8 del final del capítulo — fuera del alcance de "comprensión y adquisición de conocimiento"; y **el Capítulo 6 completo**, diferido por baja prioridad según instrucción explícita, sin intentar compensar su omisión.

**Archivo generado:** `Tsay_Cap7_analisis_futuros_ML.md`.

**Principales hallazgos [A]:** (1) el VaR es, por definición, un cuantil de la distribución de pérdidas — no una pérdida máxima, no informativo sobre lo que ocurre más allá del umbral, y no sub-aditivo; (2) la regla $\sqrt{k}$ es una consecuencia de supuestos específicos de RiskMetrics y falla cuando cualquiera de ellos falla, verificado numéricamente con un error del 7% a 15 días sobre IBM; (3) cinco métodos razonables producen VaR que difieren entre 40% y 64% sobre la misma serie, y Tsay explícitamente no declara ganador; (4) el cuantil empírico no puede extrapolar más allá del peor evento observado y su varianza crece con el inverso del cuadrado de la densidad en el punto estimado; (5) la distribución de IBM diaria pertenece a la familia Fréchet con $\xi\approx0.30$, rechazando normalidad, con indicios de cola izquierda más pesada; (6) block size y threshold cambian sustancialmente los parámetros estimados (63% y 70% respectivamente en los casos documentados); (7) los parámetros que describen la cola pueden hacerse depender de variables explicativas conocidas antes de $t$, y para IBM dos medidas de volatilidad resultan significativas mientras que un indicador estacional y uno de pánico vendedor no lo son; (8) el modelo homogéneo **falla** los diagnósticos, y corregirlo cambia el VaR al 1% en 38%; (9) los extremos de IBM están agrupados ($\hat\theta\approx0.823$), sensible a threshold y block size, e ignorar el agrupamiento subestima el VaR en ~7%; (10) los retornos de IBM tienen ACF débil y, sin embargo, extremos dependientes.

**Principales interpretaciones [B]:** (1) el guardrail triple —modelar la cola ≠ predecir dirección ≠ estrategia rentable— aplicado sistemáticamente en cada sección; (2) la aritmética de "muchos datos no significa muchos extremos", con su extensión a datos intradiarios; (3) la distinción entre cuantil incondicional y condicional como puente central hacia ML, y la observación de que la media puede ser idéntica entre dos contextos mientras los cuantiles difieren por un factor de 4; (4) la enumeración de cinco targets conceptualmente distintos (media, dirección, cuantil, evento de cola, distribución) como problemas estadísticos diferentes; (5) la calibración de cuantiles como propiedad necesaria pero no suficiente, distinta del valor económico; (6) las cinco categorías posibles de origen de un "extremo" observado, conectando con el Capítulo 5; (7) la observación de que la estacionariedad intradía es un supuesto delicado que es lógicamente previo a casi todo lo demás.

**Resultados empíricos de Tsay que vale recordar:** IBM diaria 1962–1998 (9190 obs) es la serie de referencia de todo el capítulo; $\hat\xi\approx0.30$ por Hill y $\approx0.33$ por MLE con bloques trimestrales o mayores; la tabla comparativa de cinco métodos a $p=5\%$, $1\%$ y $0.1\%$; $\hat\theta_b^{(1)}=0.823$ con $k=10$ y umbral 2.5%; $\mathrm{ES}_{0.99}=5.097$ frente a $\mathrm{VaR}_{0.99}=3.630$ (empíricos, en porcentaje); la probabilidad de una caída diaria de 2.5% o más en IBM ≈ 3.4%; y los dos días consecutivos (30 y 31 de diciembre de 1998) donde el parámetro de forma condicional pasa de 0.25 a 0.012.

**Principales limitaciones:** ningún ejemplo del capítulo involucra futuros, datos intradiarios, ni mercados electrónicos modernos; todos los diagnósticos son in-sample, sin ninguna evaluación fuera de muestra; no hay "VaR verdadero" contra el cual medir exactitud, de modo que las comparaciones entre métodos miden dispersión, no precisión; la elección de threshold, según el propio autor, no puede determinarse por teoría estadística únicamente; y la interpretación narrativa de los signos de la Tabla 7.4 presenta discrepancias con los números tabulados, que se registraron sin resolver.

**Hipótesis incorporadas al backlog:** quince (H7.1 a H7.15), cubriendo asimetría de colas, cuantiles condicionales, condicionamiento por volatilidad, colas condicionales intradía, comparación de métodos, estabilidad de threshold, estabilidad entre subperíodos, clustering de extremos, extremal index, agregación temporal, calibración de cuantiles, información media-vs-cola, extremos de microestructura, escalamiento multiperíodo, y relevancia económica. **Ninguna se ejecutó.** Se señalaron explícitamente las que tienen mayor riesgo de producir un resultado trivial (H7.3, H7.12) y la que es lógicamente previa a varias otras (H7.4).

**Afirmaciones sensibles a sobreinterpretación:** detalladas en la Sección 35. Las tres de mayor riesgo son: la idea de que los datos podrían tener más información sobre colas que sobre la media (atractiva y fácil de convertir en convicción sin verificar); la extensión de quantile regression lineal hacia modelos de ML; y la interpretación de $1/\theta$ como tamaño de cluster.

**Preguntas diferidas:** el Capítulo 6 completo (por instrucción); la inferencia bayesiana sobre incertidumbre de parámetros; y cualquier tratamiento de dependencia de colas entre múltiples instrumentos, que requeriría el Capítulo 8 y literatura no cubierta por Tsay aquí.

**Conceptos especialmente relevantes para conectar posteriormente con Machine Learning:** (1) el **cuantil condicional** $Q_\tau(Y\mid X)$ como objeto de estimación distinto de $E[Y\mid X]$; (2) la **función de pérdida asimétrica** como el mecanismo concreto que hace que un modelo estime un cuantil en vez de una media — mecánicamente aplicable a cualquier modelo entrenado por gradiente; (3) la **Ec. 7.39 de Tsay**, donde los parámetros que describen la cola son funciones de variables explicativas conocidas antes de $t$, incluida la volatilidad condicional del Capítulo 3 — la conexión [A] más directa entre features y comportamiento de cola que ofrece el libro; (4) la observación de que **media, volatilidad, cuantil y cola extrema son cuatro tareas distintas** que pueden tener cantidades muy distintas de señal disponible en los mismos datos; y (5) la **calibración** como criterio de evaluación específico de predicciones cuantílicas, distinto tanto de las métricas de error puntual como del valor económico.

---

**Estado final del capítulo:**

$$\boxed{\textbf{KNOWLEDGE ACQUIRED — NO DESIGN DECISIONS ADOPTED.}}$$
