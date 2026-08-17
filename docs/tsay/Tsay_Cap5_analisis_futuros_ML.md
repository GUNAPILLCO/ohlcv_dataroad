# Tsay — Capítulo 5: High-Frequency Data Analysis and Market Microstructure
## Informe de estudio para el Proyecto IRIS

**Convenciones de este documento**

- **[A]** = definición, resultado, ejemplo empírico o afirmación que proviene directamente de Tsay (*Analysis of Financial Time Series*, 3ª ed., Capítulo 5).
- **[B]** = interpretación, extensión o adaptación propia hacia futuros, datos OHLCV, trading o Machine Learning. Nunca es una decisión de diseño; es una hipótesis a evaluar.
- **PREGUNTA ABIERTA** = cuestión que este capítulo no permite resolver con la información disponible.
- Cada resultado empírico [A] de Tsay indica, cuando está disponible: instrumento/serie, mercado, frecuencia, período y tamaño de muestra. Ningún resultado sobre acciones de NYSE (IBM, Boeing, Apple), datos de 1990–2008, se generaliza automáticamente a MNQ ni a ningún futuro electrónico actual.
- Este informe continúa los estudios de los Capítulos 1 a 4 y es exclusivamente de **comprensión y adquisición de conocimiento**. No define features, targets, arquitecturas ni protocolos de validación para IRIS. No se descargan datos nuevos, no se ejecuta ningún experimento.

---

## 1. Resumen ejecutivo

Los capítulos anteriores trataron los datos de mercado como si fueran mediciones limpias de "cuánto vale el activo": estudiamos su distribución (Cap. 1), su dependencia lineal en el tiempo (Cap. 2), su volatilidad cambiante (Cap. 3) y su posible no linealidad (Cap. 4). En ningún momento nos preguntamos algo más básico: **¿el precio que estamos midiendo es en sí mismo un objeto limpio, o ya viene "contaminado" por la forma en que el mercado organiza las compras y las ventas?**

El Capítulo 5 de Tsay responde exactamente esa pregunta. Se llama *market microstructure* al conjunto de reglas, mecanismos e infraestructura mediante los cuales compradores y vendedores efectivamente se encuentran y acuerdan un precio: quién puede comprar, quién puede vender, a qué precio, con qué información, y qué huella deja ese mecanismo en los datos que terminamos observando. **[A]** Tsay muestra, con matemática explícita y ejemplos empíricos concretos (acciones IBM, Boeing y Apple, mercado NYSE, distintos períodos entre 1990 y 2007), que dos fenómenos del propio mecanismo de negociación —el **nonsynchronous trading** (que los activos no cotizan todos al mismo instante exacto) y el **bid–ask bounce** (que el precio observado salta entre el precio de compra y el precio de venta aunque el "valor" del activo no se haya movido)— pueden **producir autocorrelación en los retornos sin que exista ninguna relación económica real entre esos retornos**.

Éste es el guardrail central de todo el capítulo, y se mantiene explícito en cada sección de este informe:

$$\boxed{\text{estructura estadística observada} \neq \text{información económica explotable}}$$

Un segundo tema, igual de importante para el Proyecto IRIS, es la distinción entre **datos de transacciones** (*transaction data*, donde cada operación individual tiene su propio instante, precio, volumen y contexto de bid/ask) y **barras de tiempo fijo** (como el OHLCV de 1 minuto que usa IRIS), que agregan muchas transacciones en solo cinco números. **[A]** La mayor parte del Capítulo 5 —el modelo ordered probit, el modelo de descomposición, los modelos de duración ACD— está construida sobre datos de transacciones irregulares, no sobre barras. Esto significa que buena parte de la maquinaria matemática del capítulo **no se traslada directamente** a un dataset de barras OHLCV de 1 minuto como el que usa IRIS: la información que esos modelos consumen (el instante exacto de cada trade, la duración entre trades, el bid y el ask vigentes) generalmente ya no existe una vez que los datos fueron agregados en barras.

Esto no vuelve inútil al capítulo para IRIS. Al contrario: entender *qué* información se pierde al construir una barra OHLCV, y *qué* fenómenos de microestructura podrían dejar una huella residual incluso después de la agregación, es exactamente el tipo de conocimiento que se necesita antes de decidir qué variable usar como precio de referencia (Last, Mid, ¿Bid/Ask?), qué frecuencia de muestreo elegir, o cómo interpretar una autocorrelación de corto plazo que aparezca en los datos de futuros de IRIS.

**[B]** La lectura general para el Proyecto IRIS no es "hay que conseguir datos de bid/ask" ni "hay que modelar duraciones". Es más modesta: existe la posibilidad real de que parte de cualquier estructura estadística de corto plazo que se observe en datos intradiarios de MNQ no provenga de información económica genuina, sino del propio mecanismo de negociación o de cómo se construyó la barra. Esa posibilidad debe quedar como hipótesis a examinar empíricamente más adelante, no como una certeza ni como algo a descartar de antemano.

Las preguntas que este capítulo permite empezar a formular —sin resolverlas para MNQ, porque Tsay no analiza ningún futuro— son:

1. ¿Qué mide realmente "Last" en un dataset de OHLCV, y en qué se diferencia de Bid, Ask y Mid?
2. ¿Puede el bid–ask bounce, documentado por Tsay a nivel de transacción, dejar una huella en una barra de 1 minuto?
3. ¿Qué información destruye la agregación de transacciones en una barra (duración entre trades, número de trades, spread vigente)?
4. ¿Qué tan seguro es asumir que "más frecuencia = más señal", si a mayor frecuencia también aumenta el peso relativo de estos efectos de microestructura?

Ninguna de estas preguntas se responde en este informe. El objetivo, como en los capítulos anteriores, es **comprender antes de decidir**.

---

## 2. Market microstructure desde cero

### 2.1 ¿Qué problema intenta describir?

Hasta ahora, en este estudio, tratamos "el precio" como un número único en cada instante: el precio de cierre de una barra, el retorno entre dos cierres. Pero un mercado real no funciona así. En cualquier instante dado, no hay "un" precio: hay gente que quiere comprar y ofrece un precio, gente que quiere vender y pide otro precio, y un mecanismo (una bolsa, un sistema electrónico) que decide cuándo y cómo esas dos partes se encuentran y ejecutan una operación.

La pregunta que abre todo el capítulo es: **¿por qué el precio que observamos no es necesariamente una medición perfecta del "valor verdadero" del activo?**

### 2.2 Explicación intuitiva

Pensemos en un mercado de manzanas. Un comprador está dispuesto a pagar hasta 100 pesos por una manzana; un vendedor está dispuesto a aceptar 100.25 pesos como mínimo. Mientras nadie ceda, no hay operación. Si alguien finalmente compra a 100.25 (porque tiene apuro) o alguien vende a 100.00 (porque también tiene apuro), el precio "observado" de esa manzana salta entre 100.00 y 100.25 dependiendo de quién tuvo más urgencia — aunque el "valor real" de una manzana, en ese momento, sea razonablemente estable en algún punto intermedio.

Esto es, en esencia, lo que ocurre en cualquier mercado financiero organizado: existen precios de compra y de venta distintos, existe un mecanismo que decide cómo se cruzan las órdenes, y el precio que finalmente queda registrado como "el precio de la operación" refleja tanto el valor del activo como los detalles de ese mecanismo.

### 2.3 Vocabulario fundamental, desde cero

**[B]** Ninguno de estos términos requiere matemática para entenderse; son parte del vocabulario básico de cómo funciona un mercado organizado.

- **Comprador / vendedor.** Participantes que quieren, respectivamente, adquirir o desprenderse del activo.
- **Orden (order).** Una instrucción enviada al mercado: "quiero comprar N contratos a tal precio" o "quiero vender N contratos a tal precio".
- **Bid.** El mejor precio al que alguien, en este momento, está dispuesto a *comprar*. Es lo que un vendedor recibiría si vendiera *ahora mismo*.
- **Ask (u "offer").** El mejor precio al que alguien, en este momento, está dispuesto a *vender*. Es lo que un comprador pagaría si comprara *ahora mismo*.
- **Spread.** La diferencia entre ask y bid. Es, en cierto sentido, el "costo de la inmediatez": cuánto más caro es comprar ya mismo respecto de vender ya mismo.
- **Trade / transaction.** El momento en que una orden de compra y una de venta efectivamente se cruzan y se ejecuta una operación a un precio concreto.
- **Transaction price ("Last").** El precio al que se ejecutó la última operación. Es un hecho consumado, no una oferta.
- **Liquidez (market liquidity).** **[A]** Tsay la define explícitamente como la capacidad de comprar o vender cantidades significativas de un activo de forma rápida, anónima y con poco impacto en el precio. Un mercado líquido tiene compradores y vendedores dispuestos a operar en cualquier momento, con spreads pequeños; un mercado poco líquido puede tener spreads amplios o pocos participantes dispuestos a operar en un momento dado.
- **Tick size.** El incremento mínimo permitido entre dos precios distintos. Un precio no puede moverse en cualquier cantidad; solo puede moverse en múltiplos de este valor mínimo.
- **Price discovery (descubrimiento de precio).** **[A]** Tsay menciona este concepto al señalar que los datos de alta frecuencia sirven para comparar la eficiencia de distintos sistemas de negociación en la tarea de "descubrir" el precio — es decir, en qué tan rápido y correctamente el mecanismo de mercado hace que el precio observado refleje la información disponible.

### 2.4 Ejemplo sencillo, con números

Supongamos que en un instante dado:

- Bid = 100.00 (alguien está dispuesto a comprar hasta 100.00)
- Ask = 100.25 (alguien está dispuesto a vender desde 100.25)

Si en ese momento llega un comprador apurado, su operación se ejecuta contra el ask: paga 100.25. Si en cambio llega un vendedor apurado, su operación se ejecuta contra el bid: recibe 100.00. El "precio de la última transacción" (Last) puede entonces ser 100.00 o 100.25 según quién haya llegado primero — sin que el "valor" subyacente del activo se haya movido un centavo.

### 2.5 Definición técnica y qué NO significa

**[A]** Formalmente, Tsay introduce, en el contexto del modelo de Roll (1984) que se retoma en la Sección 6 de este informe, un precio "fundamental" teórico $P_t^*$ —el valor del activo en un mercado sin fricciones— y un precio observado $P_t$ que se desvía de $P_t^*$ precisamente por efecto del bid–ask spread. Esta distinción entre "valor fundamental" y "precio observado" es un **esquema conceptual [B]**, útil para razonar sobre el problema, y no debe leerse como que Tsay ofrece una manera de medir directamente ese valor fundamental — es una construcción teórica dentro de un modelo simplificado, no una cantidad observable.

**Qué NO significa esto:**

- No significa que el precio de mercado sea "falso" o "erróneo" en algún sentido moral; es simplemente el resultado de un mecanismo de negociación real, con sus propias reglas.
- No significa que exista siempre un "valor verdadero" único y objetivo escondido detrás del precio observado — es un supuesto de modelización, útil para razonar, no un hecho verificado.
- No significa que toda la variación del precio observado sea "ruido de microestructura": una parte genuina de esa variación puede ser información económica real.

### 2.6 Por qué importa para futuros, barras intradía y ML

**[B]** El mensaje conceptual central, que se desarrollará en detalle en las secciones siguientes, es que cualquier movimiento observado en los datos puede contener, esquemáticamente:

$$\text{movimiento observado} \approx \text{información económica} + \text{efectos de microestructura}$$

Esta no es una identidad matemática exacta ni una fórmula de Tsay; es un esquema conceptual **[B]** para tener presente durante todo el resto del capítulo. La pregunta relevante para IRIS —sin responderla aquí— es qué proporción de la variación de corto plazo en los datos de MNQ podría deberse a cada uno de esos dos componentes, y si esa distinción cambia según la frecuencia de muestreo.

---

## 3. Transaction data vs. barras temporales

Ésta es, por instrucción explícita del estudio, una de las secciones más importantes del informe, porque determina qué parte de la maquinaria matemática del Capítulo 5 puede, en principio, trasladarse a un dataset de barras OHLCV como el de IRIS, y qué parte no.

### 3.1 Qué son los datos de transacciones (transaction / tick data)

**[A]** Tsay define, en la Sección 5.3, la notación base que usa en gran parte del capítulo: sea $t_i$ el instante calendario (medido en segundos desde la medianoche) en que ocurre la $i$-ésima transacción de un activo. A cada transacción se le asocian variables como el precio de la operación, el volumen operado, y el bid y ask vigentes en ese momento. **[A]** Tsay llama a esta colección de $t_i$ y sus variables asociadas *transactions data*.

Ejemplo intuitivo, con timestamps ficticios:

```
10:00:00.100 → trade a 100.00, 2 contratos
10:00:00.270 → trade a 100.25, 1 contrato
10:00:01.800 → trade a 100.00, 5 contratos
10:00:08.400 → trade a 100.25, 1 contrato
```

En este tipo de datos, cada fila **es** una operación real, con su propio instante exacto. Tenemos información directa sobre:

- el orden temporal exacto de cada operación;
- la duración entre una operación y la siguiente ($\Delta t_i = t_i - t_{i-1}$, ver Sección 12 de este informe);
- la secuencia exacta de precios operados, con toda su irregularidad;
- la intensidad de negociación (cuántas operaciones ocurren por unidad de tiempo, y cómo varía eso).

### 3.2 Qué es una barra de tiempo fijo (por ejemplo, 1 minuto)

Una barra de 1 minuto resume todas las operaciones ocurridas en un intervalo fijo de reloj (por ejemplo, de 10:00:00 a 10:01:00) en cinco números:

- **Open**: precio de la primera operación del intervalo.
- **High**: precio máximo operado en el intervalo.
- **Low**: precio mínimo operado en el intervalo.
- **Close**: precio de la última operación del intervalo.
- **Volume**: cantidad total operada en el intervalo.

Si en ese minuto hubo, por ejemplo, 340 operaciones individuales con sus propios instantes, precios y volúmenes, todas esas 340 filas de datos de transacción quedan comprimidas en esos cinco números.

### 3.3 La distinción central

$$\boxed{\text{transaction data} \neq \text{1-minute OHLCV}}$$

Una barra de 1 minuto **proviene** de información de alta frecuencia (en el sentido de que, en algún momento, existieron transacciones individuales detrás de ella), pero **ya está agregada**, y en el proceso de agregación se pierde casi toda la microestructura original: el orden exacto de los precios dentro del minuto (más allá de min/max/primero/último), la duración entre transacciones individuales, el número exacto de transacciones, y el bid/ask vigente en cada una de ellas — a menos que el proveedor de datos incluya explícitamente alguna de esas variables adicionales en el propio dataset de barras.

**[B]** Esta distinción no es un tecnicismo: es el eje que determina, para cada concepto del capítulo, si la herramienta de Tsay puede aplicarse directamente sobre datos de barras, si sobrevive parcialmente, o si simplemente no es observable sin volver a datos de transacciones. Cada concepto del capítulo se clasifica, en este informe, según una de estas cuatro categorías:

- **DIRECTAMENTE RELEVANTE** — el concepto puede razonarse y, en principio, medirse directamente sobre barras OHLCV.
- **PARCIALMENTE RELEVANTE** — el concepto puede dejar una huella indirecta en las barras, pero no es observable en su forma original.
- **NO DIRECTAMENTE OBSERVABLE CON OHLCV DE BARRAS** — el concepto depende de información (timestamps exactos, bid/ask, conteo de trades) que la agregación destruye por completo, salvo que el proveedor la incluya aparte.
- **PREGUNTA ABIERTA** — no puede determinarse sin evidencia empírica adicional sobre el dataset concreto de IRIS.

Esta clasificación se aplica sistemáticamente en las secciones siguientes y se resume en la tabla de la Sección 16.

---

## 4. Last, Bid, Ask, Mid y Spread — explicación especial

Esta sección responde, de la forma más simple posible, una pregunta que puede sonar trivial pero no lo es: **cuando IRIS usa el precio de "Close" de una barra, ¿qué está usando exactamente?**

### 4.1 Bid

El mejor precio disponible al que alguien, ahora mismo, está dispuesto a **comprar** el activo. Si querés vender inmediatamente, es el precio que vas a recibir.

### 4.2 Ask

El mejor precio disponible al que alguien, ahora mismo, está dispuesto a **vender** el activo. Si querés comprar inmediatamente, es el precio que vas a pagar.

### 4.3 Spread

**[A]** Tsay lo define, en el contexto del mercado con creadores de mercado (*market makers*) de la Sección 5.2, como la diferencia entre el precio de venta (ask, $P_a$) y el precio de compra (bid, $P_b$) que ofrece el market maker:

$$Spread = P_a - P_b$$

**[A]** Tsay señala que este spread es, típicamente, pequeño en magnitud (del orden de uno o dos centavos en los ejemplos de acciones NYSE que discute), y que constituye la **principal fuente de compensación** para quien provee liquidez al mercado (el market maker).

### 4.4 Mid

$$Mid = \frac{Bid + Ask}{2}$$

Es un precio **calculado**, un punto intermedio teórico entre lo que alguien pagaría y lo que alguien recibiría en este instante. **No es, en sí mismo, el precio de ninguna operación real.**

### 4.5 Last

El precio al que se ejecutó la **última transacción real**. A diferencia de Bid, Ask y Mid (que describen el estado del mercado en este instante, incluso si nadie opera), Last es un hecho consumado: alguien efectivamente compró o vendió a ese precio, en algún momento (posiblemente hace un instante, posiblemente hace más tiempo si el activo no se opera con frecuencia).

### 4.6 Ejemplo con números

Supongamos:

- Bid = 100.00
- Ask = 100.25
- Mid = 100.125

Ahora llega una orden de compra apurada: se ejecuta contra el ask, a 100.25. El nuevo Last es 100.25. Un instante después llega una orden de venta apurada: se ejecuta contra el bid, a 100.00. El nuevo Last es 100.00. El Mid, mientras tanto, puede no haberse movido en absoluto si el bid y el ask siguen siendo 100.00 y 100.25.

### 4.7 Tres objetos distintos que no deben confundirse

| Objeto | Qué es | ¿Refleja una operación real? |
|---|---|---|
| **Last** | Precio de la última transacción ejecutada | Sí — es un hecho consumado |
| **Bid / Ask** | Estado actual del libro de órdenes: mejores precios de compra/venta disponibles | Son ofertas, no operaciones — pueden cambiar o desaparecer sin que nadie opere a ese precio |
| **Mid** | Promedio calculado entre Bid y Ask | No — es un número teórico, construido a partir de dos ofertas |

### 4.8 Lo que NO se concluye todavía

**[A]** Tsay usa el concepto de spread (Bid/Ask) para explicar un fenómeno estadístico concreto (el bid–ask bounce, ver Sección 6). **[B]** A partir de eso, hay una tentación natural de concluir cosas como "Mid es mejor que Last" o "deberíamos conseguir datos de Bid/Ask para IRIS". Por instrucción explícita de este estudio, **ninguna de esas conclusiones se adopta aquí**:

- **Mid no es necesariamente un precio ejecutable.** Es un promedio teórico entre dos ofertas; nada garantiza que alguien pueda efectivamente comprar o vender exactamente al precio Mid.
- **Last puede contener movimiento provocado por la simple alternancia entre operaciones ejecutadas contra el bid y contra el ask** (esto es, precisamente, el bid–ask bounce que se desarrolla en la Sección 6), sin que eso implique que el "valor" del activo se haya movido.

No se concluye aquí que Last sea "malo", que Mid sea "mejor", que haya que descargar datos de Bid/Ask, ni que haya que cambiar la fuente de datos del proyecto. Todo esto queda explícitamente como **PREGUNTA ABIERTA / hipótesis [B] a evaluar en el futuro**, y se retoma con una lista completa de preguntas en la Sección 17 de este informe.

**Clasificación de transferencia a barras de 1 minuto [B]:** el propio concepto de Last, Bid, Ask y Mid es **DIRECTAMENTE RELEVANTE** para pensar el problema (cualquier dataset de barras usa alguna convención de precio — típicamente Last, ya que Open/High/Low/Close de un dataset OHLCV estándar suelen construirse a partir del precio de transacción, no del bid/ask). Pero el **Bid y el Ask vigentes en cada instante** son, en un dataset OHLCV estándar sin quotes adicionales, **NO DIRECTAMENTE OBSERVABLES**: la barra no dice cuál era el spread en cada momento del minuto, solo qué rango de precios se operó.

---

## 5. Nonsynchronous Trading

### 5.1 El problema, con un ejemplo sencillo primero

Imaginemos dos activos, A y B. Hoy, la última operación de A ocurrió a las 10:00:59; la última operación de B ocurrió a las 10:00:31. Si construimos una barra para "las 10:01" usando el último precio disponible de cada uno, estamos tratando ambos precios como si fueran simultáneos — como si ambos reflejaran información hasta exactamente el mismo instante. En realidad, el precio de B es "más viejo" en casi medio minuto: no incorpora nada de lo que pasó entre las 10:00:31 y las 10:00:59.

### 5.2 Qué significa "nonsynchronous" y por qué ocurre

**[A]** Tsay abre la Sección 5.1 señalando que las operaciones bursátiles (usa como ejemplo la NYSE) no ocurren de manera sincrónica: distintas acciones tienen distinta frecuencia de negociación, y aun para una misma acción la intensidad de negociación varía de hora en hora y de día en día. Sin embargo, es habitual analizar series de retornos en intervalos fijos (diario, semanal, mensual). **[A]** Para series diarias, el precio "de cierre" de una acción es el precio de su última transacción en el día — y ese instante exacto varía de un día a otro. Tratar los retornos diarios como si fueran una serie equiespaciada de 24 horas es, en este sentido, una simplificación que puede no corresponderse con la realidad del mecanismo de negociación.

**Qué es un "stale price".** **[B]** Aunque Tsay no usa literalmente este término en el capítulo, el concepto que describe es exactamente ese: un precio "desactualizado" — la última operación registrada de un activo, que puede haber ocurrido bastante antes del instante que estamos usando como referencia (por ejemplo, "el cierre del día" o "las 10:01"), y que por lo tanto no refleja información más reciente que sí está disponible para otros activos negociados con más frecuencia.

### 5.3 Cómo puede aparecer autocorrelación espuria

**[A]** Tsay señala tres consecuencias documentadas de este fenómeno sobre retornos diarios de acciones: (a) correlación cruzada de rezago 1 (*lag-1 cross correlation*) entre los retornos de dos acciones distintas; (b) correlación serial de rezago 1 en el retorno de una cartera (*portfolio*); y (c), en algunas situaciones, autocorrelación serial **negativa** en el retorno de una sola acción individual.

**[A] El mecanismo, explicado por Tsay con un ejemplo intuitivo antes de la matemática:** consideremos las acciones A y B, verdaderamente independientes entre sí, donde A se negocia con más frecuencia que B. Si llega una noticia relevante para el mercado cerca de la hora de cierre, A —por negociarse más seguido— tiene más probabilidad de reflejar el efecto de esa noticia el mismo día. El efecto sobre B eventualmente aparecerá, pero puede quedar demorado hasta el día siguiente. Si esto ocurre sistemáticamente, el retorno de A **parece** anticipar (liderar) al retorno de B — apareciendo una correlación cruzada de rezago 1 significativa entre A y B **aunque ambas acciones sean, por construcción, independientes**. Para una cartera que contiene tanto A como B, esa correlación cruzada entre acciones se transforma en autocorrelación serial dentro del propio retorno de la cartera.

**[A] El modelo formal (versión simplificada de Lo y MacKinlay, 1990), aplicado a una sola acción:** sea $r_t$ el retorno compuesto continuo de un activo en el período $t$, con $\{r_t\}$ iid, media $E(r_t)=\mu$ y varianza $\mathrm{Var}(r_t)=\sigma^2$. En cada período, la probabilidad de que el activo **no** se negocie es $\pi$ (constante e independiente de $r_t$). Sea $r_t^o$ el retorno **observado**: si no hubo operación en $t$, $r_t^o=0$ (no hay información disponible); si hubo operación en $t$, $r_t^o$ es el retorno **acumulado** desde la última operación anterior:

$$r_t^o = \begin{cases} 0 & \text{con probabilidad } \pi \\ r_t & \text{con probabilidad } (1-\pi)^2 \\ r_t + r_{t-1} & \text{con probabilidad } (1-\pi)^2\pi \\ r_t + r_{t-1} + r_{t-2} & \text{con probabilidad } (1-\pi)^2\pi^2 \\ \vdots & \end{cases}$$

**Cada símbolo:** $r_t$ es el retorno "verdadero" del período (el que existiría si hubiera negociación continua); $r_t^o$ es lo que efectivamente se observa; $\pi$ es la probabilidad de que un período dado no tenga operación. La lógica es simple: $r_t^o = r_t$ solo si hubo operación tanto en $t$ como en $t-1$; si hubo operación en $t$ pero no en $t-1$, el retorno observado "acumula" el retorno no registrado del período anterior, y así sucesivamente.

**[A] Resultado matemático central (derivado por Tsay paso a paso, Ecs. 5.1–5.8):** aunque $E(r_t^o)=\mu$ (la media no cambia), la **autocovarianza de rezago 1** resulta:

$$\mathrm{Cov}(r_t^o, r_{t-1}^o) = -\pi\mu^2$$

y, en general, $\mathrm{Cov}(r_t^o, r_{t-j}^o) = -\mu^2\pi^j$ para $j\geq 1$. **[A]** Tsay concluye explícitamente: **siempre que $\mu \neq 0$, el nonsynchronous trading induce autocorrelaciones negativas en la serie de retornos observada** — incluso cuando la serie verdadera $\{r_t\}$ es, por construcción, independiente. La magnitud depende de $\mu$, $\pi$ y $\sigma$, y puede ser sustancial.

**Por qué esto importa, en una frase simple:** una autocorrelación negativa de rezago 1 en un retorno diario puede no significar "el mercado revierte" — puede significar, simplemente, que algunos días el activo no cotizó y el "retorno" de ese día se metió, sin querer, dentro del retorno del día siguiente.

### 5.4 Conexión con el Capítulo 2

En el Capítulo 2 aprendimos que $\mathrm{ACF}\neq 0$ no implica ni independencia ni predictibilidad económica. El Capítulo 5 agrega una tercera posibilidad, más específica: una $\mathrm{ACF}\neq 0$ observada puede no reflejar ninguna dinámica del proceso económico subyacente en absoluto, sino simplemente el hecho de que **estamos midiendo el "mismo" instante de reloj con precios que en realidad fueron fijados en momentos distintos**. La ecuación (5.8) de Tsay, derivada bajo el supuesto explícito de que $\{r_t\}$ es *verdaderamente* independiente, es la demostración matemática de que esto puede ocurrir incluso en el caso más favorable posible (sin ninguna dependencia económica real de por medio).

### 5.5 Sincronización multivariada: ¿qué pasa con varios instrumentos?

**[B]** Aunque el Capítulo 5 de Tsay es mayoritariamente univariado (se enfoca en una sola serie o, como mucho, en un par de activos), la lógica del modelo de la Sección 5.3 anterior se extiende naturalmente a la pregunta: ¿qué ocurre si en algún momento IRIS utiliza varios instrumentos como inputs simultáneos (por ejemplo, MNQ, ES, oro, bonos)?

Una barra con timestamp "10:00" en cada uno de esos instrumentos no garantiza que los últimos precios de todos ellos incorporen información hasta exactamente el mismo microinstante. Un instrumento puede haber operado por última vez a las 09:59:59.998, y otro a las 09:59:58.400 — ambos "cierran" la barra de las 10:00, pero uno de los dos está reflejando información algo más vieja que el otro.

**[A]** Tsay señala explícitamente, al cierre de la Sección 5.1, que esta discusión "puede generalizarse al retorno de una cartera que consiste en $N$ activos" y que, en la literatura de series de tiempo, los efectos del nonsynchronous trading sobre el retorno de un solo activo son equivalentes a los de una **agregación temporal aleatoria** sobre una serie de tiempo, con la probabilidad de negociación $\pi$ gobernando el mecanismo de esa agregación. Tsay no desarrolla en este capítulo el caso multivariado en detalle — remite a Campbell, Lo y MacKinlay (1997, Cap. 3).

**[B] Consecuencias hipotéticas para un contexto multi-instrumento** (sin evidencia sobre ningún futuro concreto):

- Puede aparecer un **lead-lag aparente**: un instrumento que en realidad no anticipa a otro puede *parecer* que lo hace, simplemente porque se actualiza con más frecuencia.
- Puede aparecer **correlación cruzada retrasada** entre instrumentos verdaderamente independientes o con una relación económica distinta a la observada.
- Puede aparecer una **falsa capacidad predictiva cross-market**: un modelo que usa el "último precio" de un instrumento poco líquido para predecir el movimiento de otro más líquido podría estar, en realidad, capturando solo el desfasaje de actualización entre ambos, no una relación económica genuina.

**[B] Relevancia diferenciada según el caso:**

- **Un único futuro muy líquido** (como podría ser MNQ en su sesión más activa): el propio Tsay señala que la magnitud del efecto depende de $\pi$ (la probabilidad de no negociar); un instrumento muy líquido, con negociación casi continua, tendría —en principio— un $\pi$ bajo, lo cual, según la fórmula anterior, reduciría la magnitud del efecto. Esto es una inferencia lógica a partir de la fórmula de Tsay, **no** una medición sobre MNQ.
- **Múltiples futuros con distinta liquidez:** si se combinaran instrumentos con muy distinta frecuencia de negociación, el mecanismo descrito en la Sección 5.3 anterior sería, en principio, más relevante para el instrumento menos líquido del conjunto.
- **Barras de 1 minuto:** agregar en barras de 1 minuto reduce, pero no necesariamente elimina, el problema — dentro del propio minuto puede haber igualmente instantes de última operación distintos entre instrumentos.
- **Horarios de actividad reducida** (aperturas de otras plazas, overnight, feriados parciales): son, en principio, los momentos en que $\pi$ sería más alto para cualquier instrumento, y por lo tanto donde el fenómeno descrito sería, hipotéticamente, más relevante.

**No se afirma que exista un problema material de nonsynchronous trading en MNQ, en ninguna franja horaria, sin medirlo.** Toda la discusión de este apartado es una extensión conceptual **[B]**; la modelización multivariada formal (VAR, lead-lag, cointegración) queda explícitamente diferida al Capítulo 8 del libro, que este estudio no ha abordado todavía.

### 5.6 Clasificación de transferencia a barras de tiempo fijo

**[B]**

- **Nonsynchronous trading como fenómeno:** **PARCIALMENTE RELEVANTE** para barras de un único instrumento (dentro de una barra de 1 minuto, el "cierre" sigue siendo el precio de la última transacción real de ese minuto, con toda la posible desactualización que eso implica respecto de otros instrumentos).
- **Nonsynchronous trading multivariado (sincronización entre instrumentos):** **PREGUNTA ABIERTA** — no se puede clasificar sin datos concretos sobre la frecuencia relativa de actualización de cada instrumento considerado.

---

## 6. Bid–Ask Spread y Bid–Ask Bounce

### 6.1 El rol del market maker y el spread (5.2)

**[A]** Tsay introduce esta sección en el contexto de mercados como la NYSE, donde los *market makers* cumplen un rol central: proveen liquidez comprometiéndose a comprar o vender en cualquier momento que el público lo desee. **[A]** A cambio de ese servicio, el mercado les otorga el derecho de cotizar precios distintos para compra y venta: compran al precio bid $P_b$ y venden a un precio ask más alto $P_a$. La diferencia $P_a - P_b$ es el **bid–ask spread**, y es la fuente primaria de compensación del market maker. **[A]** Tsay señala que, típicamente, este spread es pequeño en magnitud — del orden de uno o dos centavos en sus ejemplos.

**⚠️ Advertencia de contexto institucional [B], reforzada aquí por instrucción explícita del estudio:** este esquema de "market maker con obligación de cotizar" describe la estructura institucional de mercados como la NYSE en el período que Tsay analiza (hasta 2010, con datos de 1990–2008). Los mercados electrónicos de futuros modernos (como los de CME Globex, donde se negocia MNQ) operan mayormente mediante un **libro de órdenes electrónico** (*order book*) donde cualquier participante puede colocar órdenes límite y, de hecho, actuar como proveedor de liquidez, sin que exista necesariamente un market maker designado con obligaciones formales idénticas a las que describe Tsay para la NYSE de los años 90. **No se generaliza aquí la estructura institucional específica de NYSE a MNQ ni a ningún mercado electrónico de futuros actual.** El concepto de **spread** (diferencia entre el mejor precio de compra y el mejor precio de venta disponibles) es, sin embargo, un concepto de mercado general que existe también en un libro de órdenes electrónico, independientemente de si hay o no un market maker formal detrás.

### 6.2 El modelo de Roll (1984): cómo el spread introduce autocorrelación negativa

Ésta es, junto con la Sección 5 anterior, una de las piezas centrales del capítulo.

**Intuición antes de la matemática.** Supongamos que el "valor central" del mercado no se mueve en absoluto durante un breve intervalo, pero el bid y el ask siguen siendo 100.00 y 100.25. Cada operación se ejecuta o bien contra el bid (100.00, cuando alguien vende con apuro) o bien contra el ask (100.25, cuando alguien compra con apuro), de manera más o menos aleatoria según quién llegue al mercado. Una secuencia típica de precios operados podría verse así:

$$100.00 \to 100.25 \to 100.00 \to 100.25 \to \dots$$

aunque el "centro" del mercado (aproximadamente 100.125) no se haya movido casi nada. Los **retornos** calculados sobre esta secuencia de precios observados son entonces:

$$+0.25 \to -0.25 \to +0.25 \to -0.25 \to \dots$$

Es decir: **positivo, negativo, positivo, negativo**, de forma sistemática — no porque el activo esté "revirtiendo" económicamente, sino porque el precio observado simplemente rebota entre dos valores fijos (bid y ask) sin que el valor subyacente cambie.

**[A] El modelo formal de Roll (1984), tal como lo presenta Tsay (Ecs. 5.9–5.14).** El precio observado $P_t$ se modela como:

$$P_t = P_t^* + I_t \frac{S}{2}$$

**Cada símbolo:** $S = P_a - P_b$ es el bid–ask spread; $P_t^*$ es el valor "fundamental" del activo en el instante $t$, en un mercado sin fricciones (un objeto teórico, no observable); $I_t$ es una secuencia de variables binarias independientes, con $I_t=+1$ con probabilidad 0.5 (transacción iniciada por el comprador, se ejecuta cerca del ask) e $I_t=-1$ con probabilidad 0.5 (transacción iniciada por el vendedor, se ejecuta cerca del bid).

Si $P_t^*$ no cambia entre $t-1$ y $t$ (es decir, no hay ninguna novedad económica), el cambio de precio observado es:

$$\Delta P_t = (I_t - I_{t-1})\frac{S}{2}$$

**[A] Resultado exacto que deriva Tsay:** bajo estos supuestos, $E(\Delta P_t)=0$, $\mathrm{Var}(\Delta P_t) = S^2/2$, $\mathrm{Cov}(\Delta P_t, \Delta P_{t-1}) = -S^2/4$, y $\mathrm{Cov}(\Delta P_t, \Delta P_{t-j})=0$ para $j>1$. Por lo tanto, la función de autocorrelación de $\Delta P_t$ es:

$$\rho_j(\Delta P_t) = \begin{cases} -0.5 & \text{si } j=1 \\ 0 & \text{si } j>1 \end{cases}$$

**[A] Conclusión textual de Tsay:** "el bid–ask spread introduce una correlación serial negativa de rezago 1 en la serie de cambios de precio observados. Esto se conoce en la literatura de finanzas como *bid–ask bounce*." Tsay explica la intuición exactamente en los mismos términos que el ejemplo numérico de arriba: si el precio previo observado fue el ask (el valor más alto), el siguiente precio observado será, o bien igual, o bien más bajo (el bid) — nunca más alto. Simétricamente, si el precio previo fue el bid, el siguiente será igual o más alto. Esto genera, mecánicamente, la correlación negativa de rezago 1 — **y el spread no introduce ninguna correlación serial más allá del rezago 1**, según el propio resultado de Tsay.

**[A] Extensión más realista.** Tsay también considera el caso en que el valor fundamental $P_t^*$ sigue un camino aleatorio (es decir, sí cambia genuinamente con el tiempo, con incrementos iid de varianza $\sigma^2$, independientes de $I_t$). En ese caso, $\mathrm{Var}(\Delta P_t) = \sigma^2 + S^2/2$ pero la covarianza de rezago 1 no cambia, de modo que:

$$\rho_1(\Delta P_t) = \frac{-S^2/4}{S^2/2+\sigma^2} \leq 0$$

**[A]** La magnitud de la autocorrelación negativa se reduce (porque ahora hay verdadera variación económica de fondo, $\sigma^2$, compitiendo con el efecto del spread), pero el signo negativo persiste mientras $S>0$.

### 6.3 Frase central del fenómeno

$$\boxed{\text{bid–ask bounce puede parecer mean reversion sin ser una oportunidad de arbitraje}}$$

**[A]** El modelo de Roll (1984), tal como lo presenta Tsay, demuestra que el mecanismo bid–ask puede generar por sí mismo autocorrelación negativa de rezago 1 en los cambios de precio observados, incluso sin que el precio fundamental presente esa dinámica. **[B]** Por tanto, observar una reversión estadística de muy corto plazo compatible con este mecanismo no demuestra automáticamente que exista una reversión económica explotable ni una oportunidad de arbitraje. La expresión "puede parecer mean reversion" se utiliza aquí como interpretación conceptual para conectar el resultado con el vocabulario de los capítulos anteriores.

### 6.4 Bid–ask bounce multivariado

**[A]** Tsay extiende brevemente el resultado al caso bivariado: si $I_t = (I_{1t}, I_{2t})'$ denota los indicadores de tipo de orden de dos activos simultáneamente, y $I_{1t}$ e $I_{2t}$ están **contemporáneamente correlacionados de forma positiva** (por ejemplo, porque ambos activos tienden a recibir órdenes de compra o de venta al mismo tiempo, quizás por estar relacionados sectorial o macroeconómicamente), entonces los bid–ask spreads pueden introducir **correlaciones cruzadas negativas de rezago 1** entre ambos activos. **[B]** Esto es relevante conceptualmente para la Sección 5.5 de este informe (sincronización multivariada): el bid–ask bounce no es solo un fenómeno de una sola serie, también puede contaminar relaciones cruzadas entre instrumentos, sumándose al problema de nonsynchronous trading como otra fuente posible de correlación espuria entre activos.

### 6.5 Evidencia empírica de Tsay sobre bounce en transaction data

**[A]** En la Sección 5.3 del libro (desarrollada en detalle en la Sección 7 de este informe), Tsay documenta evidencia directa de bid–ask bounce en datos reales de transacciones de IBM (TORQ, noviembre 1990 – enero 1991, 59.837 pares de trades consecutivos con clasificación de dirección): la tabla de transición de movimientos de precio (arriba/sin cambio/abajo) muestra que **subidas consecutivas** y **bajadas consecutivas** son comparativamente raras (0.74% y 0.69% de los casos, respectivamente), mientras que una subida suele ir seguida de "sin cambio" o de una **bajada**. **[A]** La serie direccional $D_i$ (que vale $+1$, $0$, $-1$ según haya subida, sin cambio o bajada en cada transacción) tiene una ACF con un único pico en el rezago 1, de valor $-0.389$, altamente significativo para un tamaño de muestra de 59.837 — lo cual Tsay presenta explícitamente como **confirmación empírica de reversión de precio en transacciones consecutivas**, consistente con el bid–ask bounce.

**[A] Serie/instrumento/frecuencia/período de este resultado:** acción IBM, NYSE, datos TORQ transacción-a-transacción, 1 de noviembre de 1990 al 31 de enero de 1991, 63 días de negociación, 59.837 pares de trades consecutivos analizados.

### 6.6 Bid–ask bounce en barras de 1 minuto — PREGUNTA EMPÍRICA

Esta pregunta es, por instrucción explícita del enunciado, obligatoria y **no se responde aquí de forma universal**:

> ¿Cuánto del bid–ask bounce transacción-a-transacción puede sobrevivir después de agregar muchas transacciones a una barra de 1 minuto?

**[B]** Se trata de una **PREGUNTA EMPÍRICA**, no resoluble desde este capítulo de Tsay (que no analiza barras agregadas de ningún activo en esta sección) ni desde ningún razonamiento puramente teórico. Lo único que puede hacerse aquí es enumerar, conceptualmente, los factores que —en principio— deberían influir en la respuesta, sin prejuzgar cuál domina:

- **Número de trades por barra.** Una barra de 1 minuto puede contener desde pocas hasta muchas transacciones, pero eso no implica que el bid–ask bounce quede automáticamente "promediado" en el **Close**, porque el Close sigue siendo simplemente el precio de la última transacción del intervalo. Al aumentar el horizonte de agregación, la importancia relativa de los movimientos de un solo tick o del spread puede disminuir frente al movimiento total del precio, pero cuánto bounce permanece en una serie de Closes agregados sigue siendo una **PREGUNTA EMPÍRICA**.
- **Liquidez del instrumento.** Una mayor liquidez suele implicar más actividad y puede hacer que el spread y los movimientos mínimos representen una fracción menor del movimiento total observado durante una barra. Sin embargo, más transacciones no eliminan mecánicamente el posible efecto del bid–ask bounce sobre el Close, porque éste continúa siendo la última transacción del intervalo. Su importancia efectiva debe medirse.
- **Ancho del spread relativo al movimiento típico de precio en la barra.** Si el spread es una fracción muy pequeña del rango típico de la barra (High−Low), el bounce debería ser proporcionalmente menos relevante; si el spread es comparable al rango típico, podría dominar.
- **Tick size.** Un spread mínimo de un solo tick interactúa directamente con la discreción del precio (ver Sección 7.2 de este informe); cuanto más grande el spread en múltiplos de tick, mayor la amplitud potencial del rebote.
- **Uso de Close = Last.** Si el Close de la barra es, como es habitual, el precio de la última transacción del minuto (Last), el Close hereda directamente la posible "posición" (cerca del bid o del ask) de esa última transacción — lo cual es exactamente el mecanismo que produce bounce en la serie de Closes de barras consecutivas.
- **Frecuencia de las barras.** Cuantas más barras por unidad de tiempo (1 minuto vs. 5 minutos vs. 1 segundo), menos transacciones caen dentro de cada barra, y —en principio— menor el "promediado" y mayor la posible persistencia del bounce.
- **Horarios de baja actividad.** En tramos con pocas transacciones por minuto, cada barra se parece más a una transacción individual (menos promediado), lo cual sugiere, como hipótesis, que el bounce residual podría ser proporcionalmente más relevante en esos tramos.

**No se ejecuta ningún experimento para responder esta pregunta en este informe.** Queda explícitamente como hipótesis para backlog (ver H5.6 en la Sección 21).

### 6.7 Spread y costo de ejecución

Es necesario distinguir con precisión dos preguntas que suenan parecidas pero son completamente distintas:

**Señal estadística:** "el modelo estima que el precio puede moverse en una dirección determinada, con cierta magnitud esperada."

**Posibilidad económica:** "¿ese movimiento esperado supera el spread, el slippage y los demás costos de ejecutar la operación?"

**[B]** Esta distinción no es un resultado textual específico de esta sección de Tsay, pero es una consecuencia directa e inevitable de lo explicado arriba: si el spread bid–ask (por pequeño que sea) representa un costo real para cruzar de comprador a vendedor, entonces un movimiento de precio predicho que sea **menor** que ese spread —más cualquier otro costo de transacción (comisiones, slippage, impacto de mercado)— no puede traducirse en una operación rentable, aunque la predicción estadística sea "correcta" en promedio. Esto es especialmente relevante a alta frecuencia, donde los movimientos típicos por unidad de tiempo tienden a ser pequeños en relación con el spread.

**No se construye aquí ningún backtest ni modelo de costos.** El único objetivo de este apartado es dejar establecida, conceptualmente, la diferencia entre "movimiento predecible pequeño" y "movimiento operable" — una distinción que se retoma con más detalle en la Sección 8 (qué información pierde una barra OHLCV) y en la Sección 20 (implicancias para ML).

### 6.8 Clasificación de transferencia a barras de tiempo fijo

**[B]**

- **Spread bid–ask como cantidad medible en cada instante:** **NO DIRECTAMENTE OBSERVABLE CON OHLCV DE BARRAS** (una barra estándar de Open/High/Low/Close/Volume no incluye el spread vigente en ningún momento del intervalo, salvo que el dataset lo agregue explícitamente).
- **Bid–ask bounce como fenómeno en la serie de Closes de barras consecutivas:** **PARCIALMENTE RELEVANTE** — puede dejar una huella residual (ver Sección 6.6), pero de magnitud desconocida sin medición empírica directa sobre el dataset de IRIS.
- **Costo de ejecución (spread + slippage) como límite a la utilidad económica de una señal:** **DIRECTAMENTE RELEVANTE** como concepto, aunque su magnitud exacta para MNQ no está determinada por este capítulo.

---

## 7. Características empíricas de transacciones data (5.3)

Prioridad máxima según el enunciado. Tsay identifica, en la Sección 5.3, cuatro características que existen en datos de transacciones y que **no existen de la misma forma** en datos agregados por tiempo.

### 7.1 Unequally spaced time intervals (intervalos desigualmente espaciados)

**[A]** Los trades no ocurren a intervalos regulares de tiempo. La secuencia de precios transados, por lo tanto, no forma una serie de tiempo equiespaciada. **[A]** Tsay define $\Delta t_i = t_i - t_{i-1}$, la **duración** entre la transacción $i-1$ y la $i$, y señala explícitamente que esta duración puede contener información útil sobre microestructura de mercado (por ejemplo, sobre la intensidad de negociación).

En palabras simples: si una transacción ocurre a las 10:00:00.100 y la siguiente a las 10:00:00.270, esa duración de 0.17 segundos es, en sí misma, un dato — algo que solo existe cuando se conservan los timestamps individuales de cada operación. Este tema se retoma en profundidad en la Sección 12 de este informe (Duration Models).

### 7.2 Discrete-valued prices (precios discretos) y el tick size — sección especial

**El problema.** El precio de un activo no puede tomar cualquier valor real; solo puede moverse en múltiplos de un incremento mínimo, llamado **tick size**.

**Ejemplo sencillo.** Si el tick size es 0.25, el precio puede pasar de 100.00 a 100.25 o a 100.50, pero **nunca** puede pasar, por ejemplo, de 100.00 a 100.13 — ese valor simplemente no existe como precio posible.

**[A] Evidencia histórica que documenta Tsay, con instrumento/frecuencia/período exactos:**

- **IBM, TORQ, NYSE, 1 de noviembre de 1990 al 31 de enero de 1991** (63 días, tick size $\$1/8 = \$0.125$ vigente en ese período — antes del cambio de junio de 1997). Tabla 5.1 de Tsay, sobre las transacciones intraday de esos 63 días: aproximadamente **dos tercios** de las transacciones (67.06%) no tuvieron cambio de precio; el precio cambió en exactamente un tick en aproximadamente **29%** de los casos (14.53% subidas + 14.53% bajadas); solo 2.6% de las transacciones tuvieron cambios de dos ticks; y solo alrededor de 1.3% tuvo cambios de tres ticks o más. La distribución de cambios positivos y negativos fue aproximadamente simétrica.
- **IBM, TAQ, NYSE, diciembre de 1999** (tick size reducido a $\$1/16 = \$0.0625$, vigente antes de la decimalización de enero de 2001). **[A]** Tsay reporta que, comparado con los datos de 1990–1991, (a) el número de transacciones intradiarias se multiplicó por seis (134.120 transacciones solo en diciembre de 1999), (b) el porcentaje de trades con duración cero (múltiples transacciones en el mismo segundo) se duplicó a 22.98% —llegando a un extremo de 42 transacciones en un mismo segundo, en dos ocasiones el 3 de diciembre de 1999—, y (c) el porcentaje de transacciones **sin** cambio de precio, dentro del horario regular, bajó a 45.8% (sobre 133.475 transacciones), sustancialmente menor que en 1990–1991. **[A]** Tsay interpreta explícitamente esta reducción como consistente con la reducción del tick size: un tick más chico "debería reducir el bid–ask spread" y, en los datos, efectivamente aumentó la proporción de transacciones con algún cambio de precio.
- **Boeing, NYSE, 1 de diciembre de 2008** (ya bajo el sistema decimal, tick mínimo de 1 centavo). 43.894 transacciones en horario regular. **[A]** El histograma de cambios de precio (Figura 5.6 y Tabla 5.4 de Tsay) muestra que 58.5% de las transacciones no tuvo cambio de precio, que los cambios son aproximadamente simétricos respecto de cero, y que se concentran claramente en múltiplos de un centavo — con solo 4.59% de los cambios fuera de esos múltiplos exactos (atribuible a mecanismos de ejecución con sub-penny pricing en ciertos casos, no desarrollado en detalle por Tsay en esta sección).

**[A] Patrón general que emerge de estos tres ejemplos (siempre dentro de las series concretas analizadas, sin generalizar a otros mercados):** a medida que el tick size se redujo (de $1/8 a $1/16, y luego al sistema decimal), la proporción de transacciones **sin** cambio de precio disminuyó, y la discreción relativa del precio se hizo menos restrictiva.

**Definición técnica.** El tick size es el incremento mínimo permitido entre dos precios distintos consecutivos de un instrumento. Antes de la decimalización (29 de enero de 2001 en EE.UU.), **[A]** Tsay señala que el tick de la NYSE fue de un octavo de dólar hasta junio de 1997, y de un dieciseisavo de dólar hasta enero de 2001.

**Cómo afecta la discreción a distintos aspectos de los datos:**

- **Distribución de los cambios de precio:** en vez de una distribución continua, los cambios de precio forman una distribución sobre un conjunto discreto y finito de valores (múltiplos del tick).
- **Proporción de movimientos cero:** como muestran los tres ejemplos anteriores, una fracción considerable —a veces mayoritaria— de las transacciones individuales no implica ningún cambio de precio respecto de la anterior.
- **Retornos mínimos observables:** el retorno más pequeño posible entre dos transacciones consecutivas está acotado por abajo por el tick size relativo al nivel de precio — no puede haber un movimiento "infinitesimal".
- **Clasificación de movimientos:** cualquier esquema que clasifique cambios de precio en categorías (como el modelo ordered probit de la Sección 9 de este informe, con categorías en múltiplos de tick) depende directamente de esta discreción.
- **Diferencia entre precio continuo idealizado y precio transado:** muchos modelos teóricos de finanzas (como los de la Sección 6.2 de este informe, o los modelos de tiempo continuo del Capítulo 6 del libro) asumen un precio que se mueve de forma continua; el precio efectivamente transado, en cambio, es un objeto discreto por construcción.

**[B] El cociente tick/price como forma de pensar la importancia relativa de la discreción.** Aunque Tsay no propone explícitamente esta métrica en el capítulo, es una consecuencia razonable de lo que él sí documenta: el mismo tick size absoluto ($1/8, $1/16, un centavo) representa una fracción muy distinta del precio según el nivel del activo. Un tick de un centavo sobre una acción de \$5 es una fracción mucho mayor del precio (0.2%) que el mismo tick sobre una acción de \$500 (0.002%). El cociente $\text{tick}/\text{price}$ puede servir, como idea conceptual, para pensar qué tan restrictiva es la discreción del precio **en términos relativos**, no en términos absolutos. **No se adopta aquí ninguna métrica concreta ni ningún umbral para MNQ**; es solo una forma de razonar sobre el problema.

### 7.3 Daily periodic / diurnal pattern

**[A]** Bajo condiciones normales de negociación, la actividad de mercado exhibe un patrón periódico dentro del día. **[A]** Tsay documenta esto directamente con los datos de IBM (TORQ, 1990–1991): contando el número de transacciones en intervalos de 5 minutos ($x_t$), la ACF de esa serie muestra un patrón cíclico claro con periodicidad de 78 —exactamente el número de intervalos de 5 minutos en una jornada de negociación de la NYSE—, confirmando que el número de transacciones exhibe un patrón diario. **[A]** Promediando el número de transacciones por intervalo de 5 minutos a lo largo de los 63 días de la muestra, Tsay obtiene una forma "de sonrisa" o **forma en U**: negociación más intensa al principio y al final de la sesión, y más liviana durante la hora del almuerzo (Figura 5.2 de Tsay). **[A]** Consecuentemente, las duraciones entre transacciones también exhiben este patrón cíclico diario (ver Sección 12 de este informe, donde Tsay ajusta explícitamente este patrón antes de modelar duraciones).

**No se extrapola aquí la forma exacta "en U" de la NYSE (mercado con horario limitado, apertura y cierre definidos, alta concentración de órdenes institucionales al cierre) a futuros como MNQ**, que operan en un régimen de casi 24 horas, con estructura horaria, participantes y mecanismos de generación de volumen sustancialmente distintos.

**Distinción conceptual [B]:** conviene separar dos ideas que pueden confundirse:

- **Seasonality / efecto determinístico de la hora del día.** Un patrón sistemático y repetitivo, ligado al reloj: por ejemplo, "siempre hay más volumen en los primeros 15 minutos de sesión". Es, en principio, predecible de antemano sin necesidad de ningún modelo estadístico sofisticado — basta con saber la hora.
- **Régimen estadístico latente.** Un estado no observable directamente (como el de un modelo Markov Switching, visto en el Capítulo 4) que también cambia la dinámica de los datos, pero que no está atado a una hora fija del reloj — se infiere de los propios datos.

**[B] Pregunta hipotética para futuros, sin resolver:** ¿qué propiedades (volumen, número de trades, rango de precio, volatilidad, spread) cambian sistemáticamente según la hora del día en MNQ, y esos cambios corresponden más a un patrón determinístico ligado al reloj o hay evidencia de algo más parecido a un régimen latente superpuesto? Esto se deja explícitamente como **PREGUNTA ABIERTA / hipótesis a evaluar**, no como una conclusión.

### 7.4 Multiple transactions within the same second

**[A]** Es posible que ocurran múltiples transacciones, incluso a precios distintos, dentro del mismo segundo. Esto es, en parte, consecuencia de que el tiempo se mide en segundos, lo cual puede ser una escala demasiado gruesa en períodos de negociación intensa. **[A]** Tsay documenta esto explícitamente: para IBM (TORQ, 1990–1991), hubo 6.531 intervalos de duración cero entre transacciones consecutivas (10.91% del total), de los cuales 1.002 (1.67% del total) tuvieron además **precios distintos** entre sí. Para IBM en diciembre de 1999, este porcentaje de duración cero **se duplicó a 22.98%**, con un extremo de 42 transacciones dentro de un mismo segundo (dos veces, el 3 de diciembre de 1999).

**[B] Por qué esto importa:** incluso un timestamp medido en segundos —una resolución que podría parecer suficientemente fina— puede ser demasiado grueso para capturar el verdadero orden temporal de las operaciones durante períodos de actividad intensa. Esto refuerza la idea general de que la "resolución" de un dato de mercado (segundos, milisegundos, microsegundos) no es un detalle técnico menor: determina directamente qué fenómenos pueden observarse y cuáles quedan mezclados entre sí.

### 7.5 Nota sobre calidad de datos (Remark de Tsay)

**[A]** Tsay incluye una advertencia explícita, con motivo del ejemplo de Boeing (2008): el registro de datos de alta frecuencia suele ser de peor calidad que el de datos de baja frecuencia. Pueden faltar observaciones, la exactitud del timestamp de algunas transacciones puede ser cuestionable (por ejemplo, transacciones registradas después del cierre oficial de la sesión, antes incluso de la apertura del after-hours), y la limpieza de estos datos ("data cleaning") requiere un conocimiento profundo de cómo opera efectivamente el mercado en cuestión. **[A]** Tsay concluye que es importante especificar clara y precisamente los métodos usados en la limpieza de datos, y que esas decisiones deben tenerse en cuenta al hacer inferencia. **[B]** Esta advertencia, aunque formulada por Tsay en el contexto de datos de transacciones de acciones, es una precaución general razonable a tener presente para cualquier dataset de alta frecuencia, incluidos los de futuros — sin que esto implique ningún diagnóstico específico sobre la calidad de los datos de MNQ usados en IRIS.

### 7.6 Clasificación de transferencia a barras de tiempo fijo

**[B]**

- **Unequally spaced intervals / duración entre trades:** **NO DIRECTAMENTE OBSERVABLE CON OHLCV DE BARRAS** — una barra de tiempo fijo, por definición, tiene siempre la misma separación temporal respecto de la anterior (60 segundos, si son barras de 1 minuto); la duración irregular entre transacciones individuales queda destruida por la propia construcción de la barra.
- **Discreteness / tick size:** **DIRECTAMENTE RELEVANTE** — el tick size sigue aplicando a los precios Open/High/Low/Close de cualquier barra, ya que estos son, en última instancia, precios de transacciones reales.
- **Diurnal pattern:** **DIRECTAMENTE RELEVANTE** — un patrón horario en volumen, número de trades (si está disponible) o volatilidad puede observarse y medirse directamente sobre una serie de barras, agregando por hora del día.
- **Multiple transactions within the same second:** **NO DIRECTAMENTE OBSERVABLE CON OHLCV DE BARRAS** — el conteo exacto de transacciones simultáneas o casi simultáneas desaparece en la agregación, salvo que el dataset incluya explícitamente un conteo de número de trades por barra (ver Sección 8.2 de este informe).

---

## 8. Qué información pierde una barra OHLCV — sección especial

**[B]** Esta sección sintetiza, de forma explícita, todo lo perdido en el paso de transacción individual a barra de tiempo fijo, tal como lo exige el enunciado.

### 8.1 Lo que una barra de 1 minuto SÍ conserva

- El precio de la primera transacción del intervalo (Open).
- El precio máximo y mínimo transados dentro del intervalo (High, Low).
- El precio de la última transacción del intervalo (Close = Last del minuto).
- El volumen total operado en el intervalo (si el dataset lo incluye correctamente agregado).

### 8.2 Lo que una barra de 1 minuto NO conserva (salvo que el dataset lo agregue explícitamente)

- El **orden exacto** de todos los precios operados dentro del minuto, más allá de Open/High/Low/Close (por ejemplo, si el precio subió, bajó, volvió a subir y luego cerró en el máximo, o si simplemente subió de forma monótona hasta el máximo y ahí cerró — ambos escenarios pueden producir el mismo OHLC).
- La **duración exacta entre transacciones** dentro del minuto.
- El **número de transacciones** ocurridas dentro del minuto (a menos que el proveedor de datos lo agregue como una columna adicional, como en el ejemplo de AlgoSeek/NASDAQ ITCH descrito en la literatura de trading algorítmico, no en Tsay).
- El **bid y el ask vigentes** en cada instante del minuto — y, por lo tanto, si cada transacción individual se ejecutó cerca del bid o del ask.
- La **dirección de iniciación de cada trade** (si fue iniciado por el comprador o por el vendedor).

### 8.3 Volumen en barras vs. trading intensity — distinción obligatoria

**[B]** Una barra puede reportar, por ejemplo, $Volume = 10.000$. Eso **no** nos dice, por sí solo:

- cuántas transacciones individuales compusieron ese volumen;
- la duración entre esas transacciones;
- la secuencia temporal exacta en que ocurrieron;
- el tamaño de cada transacción individual.

**Ejemplo numérico, tal como lo pide el enunciado:** 100 transacciones de 1 contrato cada una producen el mismo volumen total (100) que 1 sola transacción de 100 contratos. Sin embargo, la estructura de actividad de mercado detrás de ambos escenarios es completamente distinta: en el primer caso hubo intensa actividad de muchos participantes pequeños; en el segundo, una única operación de gran tamaño (posiblemente institucional). **No debe asumirse que $Volume = \text{número de trades}$.**

**[A]** Aunque Tsay no analiza esta distinción exactamente en estos términos en el Capítulo 5, todo el desarrollo de las Secciones 5.3, 5.5 y 5.7 del libro está construido, precisamente, sobre variables como el **número de transacciones** ($N_i$, usado en el modelo PCD de la Sección 5.7) y la **duración entre transacciones** ($\Delta t_i$), tratadas como objetos claramente distintos del volumen operado. El propio modelo PCD (ver Sección 14 de este informe) usa $N_i$ —el número de trades sin cambio de precio dentro de un intervalo— como una variable explicativa separada del tamaño del cambio de precio, precisamente porque contiene información distinta.

**[B] Relevancia para IRIS:** si en algún momento futuro el dataset de MNQ incluyera, además de Volume, un conteo de número de transacciones por barra, esa sería una variable **conceptualmente distinta** del volumen y potencialmente informativa por sí misma sobre la estructura de la actividad de mercado (muchos trades chicos vs. pocos trades grandes) — sin que esto implique que deba incorporarse como feature; es solo una distinción conceptual a tener presente.

---

## 9. Models for Price Changes (5.4)

**Estudio serio pero selectivo**, según el enunciado. El objetivo aquí es comprender el problema que se intenta resolver, no memorizar toda la estimación.

**[A]** Tsay abre la Sección 5.4 señalando el problema central: la discreción de los precios y la concentración de casos "sin cambio" (documentadas en la Sección 7 de este informe) hacen difícil modelar los cambios de precio transacción a transacción con las herramientas estadísticas habituales (por ejemplo, una regresión lineal continua no es apropiada para una variable que toma solo un puñado de valores discretos, con una masa de probabilidad enorme en el valor cero).

**[A]** Transacción a transacción, el cambio de precio puede pensarse como una variable discreta, medida en múltiplos de tick:

$$\dots, -2, -1, 0, +1, +2, \dots \quad \text{(en ticks)}$$

Esta variable combina, en realidad, tres preguntas distintas que se pueden separar conceptualmente (y que Tsay efectivamente separa en el modelo de descomposición de la Sección 5.4.2):

1. **¿Hubo o no hubo cambio de precio?** (ausencia de cambio, sí/no)
2. **Si hubo cambio, ¿fue hacia arriba o hacia abajo?** (dirección)
3. **Si hubo cambio, ¿de qué magnitud fue?** (tamaño, en múltiplos de tick)

**[A]** Tsay menciona que Campbell, Lo y MacKinlay (1997) discuten varios modelos econométricos propuestos en la literatura para este problema, y presenta en detalle dos que tienen la ventaja de incorporar variables explicativas para estudiar los movimientos de precio intradiario: el **modelo ordered probit** (Hauseman, Lo y MacKinlay, 1992) y un **modelo de descomposición** (versión simplificada de Rydberg y Shephard, 2003, desarrollada por McCulloch y Tsay, 2000).

---

## 10. Ordered Probit (5.4.1)

### 10.1 Desde cero: qué problema resuelve

**El resultado que intentamos predecir pertenece a categorías con un orden natural.** No es una clasificación cualquiera (como "gato" vs. "perro", donde no hay un orden implícito); es una clasificación donde las categorías están ordenadas de menor a mayor.

**Ejemplo:** el cambio de precio de una transacción puede caer en una de estas categorías ordenadas:

$$-2 \text{ ticks} \;<\; -1 \text{ tick} \;<\; 0 \;<\; +1 \text{ tick} \;<\; +2 \text{ ticks}$$

Un modelo de clasificación "genérico" (multiclase, sin orden) trataría estas cinco categorías como si no tuvieran relación entre sí. Un modelo **ordenado** aprovecha, en cambio, el hecho de que "+2" está más cerca de "+1" que de "−2".

### 10.2 Intuición: una variable latente detrás de las categorías observadas

**[A]** La idea central del ordered probit es que existe una variable **continua y no observable** —llamada variable latente, $y_i^*$— que representa el "verdadero" cambio de precio deseado o implícito, y que la categoría observada surge de comparar esa variable latente contra un conjunto de **umbrales (thresholds o cut points)**.

**Ejemplo intuitivo, sin matemática todavía:** imaginemos que existe una "presión" continua sobre el precio, que puede tomar cualquier valor real. Si esa presión es muy negativa, el precio termina bajando 2 ticks; si es moderadamente negativa, baja 1 tick; si está cerca de cero, no cambia; y así sucesivamente. La variable observada (el cambio de precio en ticks) es entonces una versión "recortada" en categorías de esa presión subyacente continua.

### 10.3 Definición técnica

**[A]** Sea $y_i^*$ el cambio de precio no observable del activo (formalmente, $y_i^* = P_{t_i}^* - P_{t_{i-1}}^*$, donde $P_t^*$ es el precio "virtual" del activo en $t$ — el mismo tipo de objeto teórico que se usó en la Sección 6.2 de este informe). El modelo ordered probit asume que $y_i^*$ es continua y sigue:

$$y_i^* = x_i\beta + \epsilon_i$$

**Cada símbolo:** $x_i$ es un vector fila de variables explicativas, disponibles en el instante $t_{i-1}$ (es decir, información conocida antes de la transacción $i$); $\beta$ es un vector de parámetros; $\epsilon_i$ es el término de error, con $E(\epsilon_i|x_i)=0$, $\mathrm{Var}(\epsilon_i|x_i)=\sigma_i^2$ (no necesariamente constante — puede depender de otras variables $w_i$, típicamente incluyendo la duración entre trades), y $\mathrm{Cov}(\epsilon_i,\epsilon_j)=0$ para $i\neq j$. Se asume, además, que la distribución condicional de $\epsilon_i$ es Gaussiana.

**[A]** El cambio de precio observado $y_i$ puede tomar $k$ valores posibles $\{s_1,\dots,s_k\}$ (en la práctica, un número finito — por ejemplo, $k=7$ si se agrupan todos los cambios de "−3 ticks o más negativo" en una sola categoría). La regla que conecta la variable latente con la categoría observada es:

$$y_i = s_j \quad \text{si} \quad \alpha_{j-1} < y_i^* \leq \alpha_j, \quad j=1,\dots,k$$

donde $-\infty = \alpha_0 < \alpha_1 < \dots < \alpha_{k-1} < \alpha_k = \infty$ son los **umbrales (cut points)** que dividen la recta real en $k$ regiones, cada una asociada a una categoría observable.

**[A]** Bajo el supuesto de normalidad condicional, la probabilidad de cada categoría se obtiene evaluando la función de distribución acumulada normal $\Phi(\cdot)$ en los umbrales estandarizados por $\sigma_i(w_i)$ (Ec. 5.18 de Tsay) — este informe no reproduce la derivación completa del likelihood, por instrucción explícita del enunciado, pero conserva la estructura conceptual: la probabilidad de observar la categoría $j$ es el "área" bajo la curva normal entre los dos umbrales que la delimitan.

### 10.4 Ejemplo empírico de Tsay: Ejemplo 5.1

**[A] Instrumento, mercado, frecuencia, período, tamaño de muestra:** Hauseman, Lo y MacKinlay (1992) aplican el ordered probit a datos de transacciones de 1988 de más de 100 acciones; Tsay reporta solo el resultado para **IBM**: 206.794 transacciones. Medias (desviaciones estándar) muestrales: cambio de precio $y_i$: $-0.0010\,(0.753)$ ticks; duración $\Delta t_i$: $27.21\,(34.13)$ segundos; spread bid–ask: $1.9470\,(1.4625)$ ticks.

**[A]** El modelo usado tiene $k=9$ categorías (de $-4$ a $+4$ ticks), y las variables explicativas incluyen (Ec. 5.19–5.20 de Tsay): la duración reescalada $\Delta t_i^* = (t_i - t_{i-1})/100$; tres rezagos del propio cambio de precio $y_{i-v}$ ($v=1,2,3$); tres rezagos de retornos de 5 minutos del futuro del S&P 500 ($SP5_{i-v}$, es decir, información de **otro mercado** — un índice de futuros relacionado); tres rezagos de un indicador de si la transacción anterior se ejecutó por encima o por debajo del punto medio bid/ask ($IBS_{i-v}$); y una interacción entre volumen (transformado con Box–Cox) e $IBS$. La varianza condicional (Ec. 5.20) depende de la duración reescalada y del spread bid–ask vigente en la transacción anterior.

**[A] Resultado exacto que reporta Tsay (Tabla 5.5), interpretado por el propio autor:**

1. Los umbrales estimados no son equiespaciados, pero son aproximadamente simétricos respecto de cero.
2. La duración entre transacciones ($\Delta t_i^*$) afecta tanto la media condicional como la varianza condicional del cambio de precio.
3. Los coeficientes de los cambios de precio rezagados son negativos y altamente significativos, indicando **reversión de precio** — consistente con el bid–ask bounce documentado en la Sección 6 de este informe.
4. Como era de esperar, el spread bid–ask vigente en la transacción anterior afecta significativamente la varianza condicional.

**[A]** Casi todos los estadísticos $t$ reportados son muy grandes; Tsay señala explícitamente que esto no es sorprendente dado el gran tamaño de muestra (206.794 observaciones) — una advertencia metodológica implícita sobre no confundir significancia estadística con magnitud económica, coherente con el guardrail del Capítulo 4.

### 10.5 Qué variables aparecen y por qué podrían ser relevantes (sin adoptarlas)

**[B]** Registrando exactamente qué intenta predecir este ejemplo y qué variables usa (por instrucción del enunciado, sin convertirlas automáticamente en features recomendadas):

- **Target:** categoría ordenada de cambio de precio en la próxima transacción (9 categorías, −4 a +4 ticks).
- **Duración** ($\Delta t_i^*$): podría ser relevante porque, conceptualmente, mide cuánto tiempo pasó desde la última operación — potencialmente relacionado con la intensidad de negociación o la "novedad" de la información.
- **Spread bid–ask rezagado:** podría ser relevante porque el ancho del spread condiciona directamente cuán "lejos" puede moverse el precio observado por el simple mecanismo de bounce.
- **Cambios de precio rezagados:** podrían ser relevantes por el efecto de reversión documentado (bid–ask bounce).
- **Retornos de otro mercado relacionado (SP5):** podrían ser relevantes como proxy de información de mercado más amplia que afecta a IBM.
- **Indicador de posición relativa al bid/ask (IBS) y volumen:** podrían ser relevantes como indicadores de la "agresividad" de la orden.

**Ninguna de estas variables se convierte aquí en una feature recomendada para IRIS.** Son, únicamente, las variables que Hauseman, Lo y MacKinlay (1992) usaron para un problema distinto (predecir el cambio de precio transacción a transacción de una acción NYSE en 1988), en un contexto institucional distinto (con market makers formales, spreads cotizados explícitamente, y datos de transacción individuales con bid/ask asociado — información que, como se estableció en la Sección 3, generalmente no está disponible en un dataset OHLCV de barras).

### 10.6 Conexión con ML — comparación conceptual, sin decidir el target

**[B]** El ordered probit es, conceptualmente, el antepasado estadístico directo de lo que en Machine Learning se llama **clasificación ordinal** (*ordinal classification*): un problema de clasificación donde las clases tienen un orden natural, y donde tratarlo como clasificación multiclase "genérica" (sin orden) desperdicia información. Se distingue de:

- **Clasificación binaria:** solo dos categorías, sin problema de orden porque no hay más de dos clases para ordenar.
- **Clasificación multiclase genérica:** varias categorías, pero sin ningún orden asumido entre ellas (por ejemplo, clasificar el "tipo de día de mercado" en categorías sin jerarquía).
- **Clasificación ordinal:** varias categorías con un orden natural, exactamente como en el ordered probit — el equivalente moderno más directo.

**No se afirma aquí que el target de IRIS deba ser ordinal.** Esta sección es puramente conceptual: el propósito es que quede clara la analogía entre "categorías con orden natural" (5.4.1 de Tsay) y "clasificación ordinal" (ML moderno), sin que eso implique ninguna decisión sobre cómo debe formularse el problema de predicción de IRIS.

### 10.7 Clasificación de transferencia a barras de tiempo fijo

**[B]** El **modelo ordered probit en su forma original** —transacción a transacción, con variables como duración exacta entre trades y bid/ask vigente— es **NO DIRECTAMENTE OBSERVABLE CON OHLCV DE BARRAS** (depende de información que la agregación destruye). La **idea general** de tratar un cambio de precio discreto y ordenado como target de un modelo probabilístico es **PARCIALMENTE RELEVANTE**: podría, en principio, reformularse sobre el cambio de precio entre barras consecutivas (por ejemplo, "el Close subió, bajó o se mantuvo igual respecto del Close anterior, en múltiplos de tick"), aunque eso sería una adaptación conceptual [B], no algo que Tsay describa ni recomiende en este capítulo.

---

## 11. Decomposition Model (5.4.2)

### 11.1 Pregunta central

**¿Podemos descomponer un cambio de precio en varias preguntas más sencillas, en vez de modelarlo como un único objeto complejo?**

### 11.2 La descomposición real de Tsay (versión simplificada de Rydberg y Shephard, 2003)

**[A]** Tsay presenta, en la Sección 5.4.2, el modelo ADS (Ai-Di-Si), que descompone el cambio de precio de la $i$-ésima transacción, $y_i \equiv P_{t_i}-P_{t_{i-1}}$, como el producto de tres componentes:

$$y_i = A_i D_i S_i$$

**Cada símbolo:**

- $A_i$: variable binaria — **¿hubo o no hubo cambio de precio?** ($A_i=1$ si hubo cambio, $A_i=0$ si no).
- $D_i$: variable discreta, **definida solo si $A_i=1$** — **dirección del cambio** ($D_i=+1$ si el precio subió, $D_i=-1$ si bajó).
- $S_i$: **tamaño del cambio en ticks, definido solo si $A_i=1$** — un entero positivo, con $S_i=0$ si $A_i=0$.

**[A]** Esto **sí corresponde exactamente** a la descomposición intuitiva planteada en el enunciado (ocurrencia → dirección → magnitud), con una precisión adicional que aporta el propio Tsay: existe un **orden natural en la descomposición**. $D_i$ solo está definido cuando $A_i=1$ (no tiene sentido preguntar la dirección de un cambio que no ocurrió), y $S_i$ solo tiene sentido dado $A_i=1$ y conocido $D_i$. **[A]** Tsay lo expresa formalmente factorizando la probabilidad condicional:

$$P(y_i|\mathcal{F}_{i-1}) = P(A_iD_iS_i|\mathcal{F}_{i-1}) = P(S_i|D_i,A_i,\mathcal{F}_{i-1})\,P(D_i|A_i,\mathcal{F}_{i-1})\,P(A_i|\mathcal{F}_{i-1})$$

donde $\mathcal{F}_{i-1}$ es la información disponible hasta la transacción $i-1$.

### 11.3 Cómo se modela cada pieza

**[A]** Cada una de las tres piezas se modela con una **regresión logística** (para $A_i$ y $D_i$, variables binarias) sobre variables explicativas conocidas en $\mathcal{F}_{i-1}$, y con una **distribución geométrica** (para $S_i$, dado que es un tamaño entero positivo), con parámetros que también dependen logísticamente de variables explicativas — permitiendo, además, distintos parámetros según la dirección del cambio (asimetría entre subidas y bajadas).

### 11.4 Distinción, conectada con Capítulos 3–4

$$\text{dirección} \neq \text{magnitud} \neq \text{ocurrencia del cambio}$$

**[B]** Esta es la misma lógica de separar preguntas distintas que ya apareció en capítulos anteriores: en el Capítulo 3, la dirección del retorno ($r_t$) y su magnitud/volatilidad ($|r_t|$, $r_t^2$) mostraron estructuras estadísticas muy distintas (poca estructura en dirección, mucha en volatilidad); aquí, Tsay aplica la misma lógica de separación a un objeto distinto (el cambio de precio transacción a transacción), descomponiéndolo en ocurrencia, dirección y tamaño, cada uno modelado por separado.

### 11.5 Ejemplo empírico de Tsay: Ejemplo 5.2

**[A] Instrumento, mercado, frecuencia, período, tamaño de muestra:** IBM, TORQ, NYSE, 1 de noviembre de 1990 al 31 de enero de 1991, 63 días de negociación, 59.838 transacciones intradiarias en horario regular (59.775 tras descartar la primera observación de cada bloque de variables rezagadas).

**[A]** Las variables explicativas consideradas fueron el indicador de acción anterior ($A_{i-1}$), la dirección anterior ($D_{i-1}$), el tamaño anterior ($S_{i-1}$), el volumen anterior, la duración anterior y el spread bid–ask vigente. **[A]** Tsay reporta que **solo $A_{i-1}$, $D_{i-1}$ y $S_{i-1}$ resultaron estadísticamente significativos** para el modelo entretenido; el volumen, la duración y el spread bid–ask no lo fueron en esta especificación concreta.

**[A] Resultados exactos que reporta Tsay:**

1. **Probabilidad de cambio de precio, condicionada al pasado inmediato:** $P(A_i=1|A_{i-1}=0) = 0.258$ vs. $P(A_i=1|A_{i-1}=1) = 0.476$. Interpretación textual de Tsay: los cambios de precio pueden ocurrir en **clusters** — cuando la transacción anterior tuvo cambio de precio, la probabilidad de que la siguiente también lo tenga casi se duplica.

2. **Dirección del cambio, condicionada al pasado:** $P(D_i=1|D_{i-1}=0)=0.483$ (probabilidades de subida/bajada aproximadamente parejas si la transacción previa no tuvo cambio); $P(D_i=1|D_{i-1}=1, A_i=1)=0.085$ (muy baja probabilidad de subida consecutiva); $P(D_i=1|D_{i-1}=-1, A_i=1)=0.904$ (muy alta probabilidad de subida tras una bajada). **[A]** Tsay interpreta esto explícitamente como una confirmación del **efecto de bid–ask bounce** y de la reversión de precio en negociación de alta frecuencia — exactamente coherente con lo desarrollado en la Sección 6 de este informe.

3. **Tamaño del cambio:** evidencia débil de que cambios grandes tiendan a seguir a otros cambios grandes. Con el parámetro geométrico $\lambda_{u,i} = 2.235 - 0.670\,S_{i-1}$, la probabilidad de que una subida sea de exactamente 1 tick es 0.827 si $S_{i-1}=1$, baja a 0.709 si $S_{i-1}=2$, y a 0.556 si $S_{i-1}=3$ — es decir, cuanto más grande fue el cambio anterior, menor la probabilidad de que el próximo cambio (si sube) sea de un solo tick, lo que es consistente con una relación positiva entre el tamaño anterior y el tamaño esperado del próximo cambio.

**[A]** Tsay señala una ventaja del modelo ADS respecto del ordered probit: **no requiere truncar ni agrupar el tamaño del cambio de precio en categorías predefinidas** — el tamaño se modela con una distribución geométrica sobre enteros positivos sin límite superior fijo, en vez de forzarlo a un número finito de categorías.

### 11.6 Clasificación de transferencia a barras de tiempo fijo

**[B]** Igual que el ordered probit, el modelo de descomposición ADS en su forma original es **NO DIRECTAMENTE OBSERVABLE CON OHLCV DE BARRAS** (depende de la secuencia completa de transacciones individuales). Sin embargo, **la lógica de descomposición en sí** (ocurrencia / dirección / magnitud) es **DIRECTAMENTE RELEVANTE** como esquema conceptual aplicable, en principio, al cambio de precio entre barras consecutivas — sin que esto implique adoptar ningún target concreto para IRIS.

---

## 12. Duration Models y ACD (5.5, 5.5.1–5.5.3)

**Tratado principalmente a nivel conceptual**, según el enunciado, evitando derivaciones extensas de estimación/simulación que dependen de timestamps transacción-a-transacción inexistentes en barras.

### 12.1 Qué es una duration

**[A]** Sea $t_i$ el instante calendario de la $i$-ésima transacción. La **duración** es:

$$\Delta t_i = t_i - t_{i-1}$$

En palabras simples: **el tiempo transcurrido entre una operación y la siguiente.**

**[A]** Tsay motiva el interés en esta variable con una interpretación económica explícita: "duraciones más largas indican falta de actividad de negociación, lo cual a su vez señala un período sin nueva información. El comportamiento dinámico de las duraciones contiene, por lo tanto, información útil sobre la actividad de mercado intradiaria."

**⚠️ Advertencia metodológica, exigida explícitamente por el enunciado:** esta es la **formulación e interpretación textual de Tsay [A]** — no es una ley universal comprobada. Convertir "no hubo trades" en "sabemos con certeza que no llegó nueva información" sería una afirmación causal más fuerte de lo que el propio texto sostiene. **[B]** Es razonable pensar que ausencia de negociación *podría* correlacionarse con ausencia de noticias relevantes, pero también podría deberse a otros factores (horario de baja actividad estructural, feriados, ausencia temporal de contraparte, mecanismos de mercado específicos) que no tienen relación directa con el flujo de información económica. Se separa aquí explícitamente: **[A]** la interpretación que ofrece Tsay como motivación del modelo, de **[B]** cualquier lectura causal más fuerte, que este capítulo no establece ni sería prudente adoptar sin evidencia adicional.

**Vocabulario relacionado:**

- **Trading intensity (intensidad de negociación):** qué tan seguido ocurren transacciones — el inverso conceptual de la duración esperada.
- **Heavy trading (negociación intensa):** duraciones cortas, muchas transacciones por unidad de tiempo.
- **Thin trading (negociación liviana):** duraciones largas, pocas transacciones por unidad de tiempo.
- **Duración esperada condicional** ($\psi_i$, ver más abajo): la expectativa, dado todo lo conocido hasta la transacción anterior, de cuánto va a durar hasta la próxima transacción.

### 12.2 Por qué duration no existe directamente en barras de tiempo fijo

Sección obligatoria por instrucción del enunciado.

Si usamos barras de 1 minuto, su separación temporal respecto de la barra anterior es **siempre** la misma:

$$60\text{s}, \quad 60\text{s}, \quad 60\text{s}, \dots$$

Eso **no es** la trade duration del Capítulo 5. La verdadera información de duración —$0.1\text{s}, 0.8\text{s}, 5\text{s}, 0.03\text{s}, \dots$, con toda su irregularidad— ya fue perdida en el momento en que las transacciones individuales se agregaron dentro de cada barra de 60 segundos.

$$\boxed{\text{ACD sobre transaction duration} \not\equiv \text{ACD sobre timestamps de barras de 1 minuto}}$$

**[B]** Un modelo ACD aplicado a "los timestamps de las barras de 1 minuto" no tendría ningún sentido: esos timestamps son, por construcción del propio dataset, siempre equiespaciados en exactamente 60 segundos. No hay ninguna variación que modelar en esa secuencia. Lo único que podría, en principio, sobrevivir de la idea de ACD en un contexto de barras sería aplicar la misma lógica de "persistencia de intensidad" a alguna variable derivada —por ejemplo, el número de trades por barra, si el dataset lo incluyera— pero eso ya no sería un ACD sobre duraciones transacción-a-transacción, sino una adaptación conceptual distinta [B], no descrita ni recomendada por Tsay.

### 12.3 Ajuste del patrón diurno antes de modelar duraciones

**[A]** Como se documentó en la Sección 7.3 de este informe, las transacciones intradiarias exhiben un patrón cíclico diario. Por eso, Tsay no modela la duración cruda $t_i$, sino la **duración ajustada**:

$$\Delta t_i^* = \frac{\Delta t_i}{f(t_i)}$$

donde $f(t_i)$ es una función determinística de la **hora del día** en la que ocurre la transacción $i$, utilizada para capturar el patrón cíclico intradiario de las duraciones. **[A]** Tsay señala que $f(\cdot)$ depende del activo concreto y del comportamiento sistemático del mercado, y que existen muchas formas de estimarla (por ejemplo, splines de suavizado), sin que ningún método domine claramente a los demás; en su ejemplo concreto usa funciones cuadráticas simples más variables indicadoras (Ec. 5.32), ajustadas por mínimos cuadrados sobre $\ln(\Delta t_i)$.

**Por qué esto importa, con el ejemplo del enunciado:** si al mediodía los trades son sistemáticamente más espaciados debido a la estructura habitual del mercado (menor actividad en el almuerzo, documentada en la Sección 7.3), eso **no debería confundirse** con "un shock de baja actividad" — es, simplemente, el patrón horario normal repitiéndose todos los días. Esquemáticamente (no como identidad matemática exacta, **[B]**):

$$\text{duración observada} \approx \text{patrón horario} + \text{dinámica adicional}$$

**[A]** La forma exacta que usa Tsay no es una suma simple sino una razón: $\Delta t_i^* = \Delta t_i / f(t_i)$, con $f(\cdot) = \exp[d(\cdot)]$ y $d(\cdot)$ una combinación lineal de funciones cuadráticas e indicadores (Ec. 5.32 de Tsay). El esquema aditivo de arriba es solo una forma intuitiva de transmitir la idea; **no debe leerse como la fórmula real de Tsay**.

**[A] Evidencia empírica de que el ajuste funciona:** para IBM (TORQ, 1990–1991), Tsay muestra (Figura 5.8) que las duraciones promedio en intervalos de 5 minutos, antes del ajuste, exhiben claramente el patrón diurno; después de dividir por $f(\Delta t_i)$, ese patrón queda "en gran medida removido".

### 12.4 El modelo ACD: analogía con GARCH

**Pregunta que hace GARCH (Capítulo 3):** ¿la volatilidad alta tiende a persistir en el tiempo?

**Pregunta que hace ACD:** ¿los períodos de intensidad de negociación alta o baja tienden a persistir en el tiempo?

**[A]** Tsay presenta el modelo autoregressive conditional duration (ACD, Engle y Russell, 1998) explícitamente como una adaptación de la idea de GARCH al problema de las duraciones. Sea $x_i = t_i^*$ la duración ajustada, y sea $\psi_i = E(x_i|\mathcal{F}_{i-1})$ la **duración esperada condicional** — el análogo directo de $\sigma_t^2$ en GARCH, pero para duración en vez de varianza. El modelo básico es:

$$x_i = \psi_i \varepsilon_i$$

donde $\{\varepsilon_i\}$ es una secuencia de variables no negativas, iid, con $E(\varepsilon_i)=1$ (típicamente exponencial estándar → modelo **EACD**, o Weibull estandarizada → modelo **WACD**), y:

$$\psi_i = \omega + \sum_{j=1}^{r}\gamma_j x_{i-j} + \sum_{j=1}^{s}\omega_j \psi_{i-j}$$

**Cada símbolo:** $x_i$ es la **duración observada** (ajustada); $\psi_i$ es la **duración esperada condicional** — lo que el modelo predice, dado el pasado, para la próxima duración; $\varepsilon_i$ es el "shock" multiplicativo de duración, análogo al shock estandarizado de GARCH; $\omega,\gamma_j,\omega_j$ son parámetros a estimar; $r,s$ son los órdenes del modelo (número de rezagos de duración observada y de duración esperada, respectivamente).

**Similitud conceptual con ARCH/GARCH, explícita en el propio Tsay:** así como GARCH modela $\sigma_t^2$ como una combinación de shocks al cuadrado pasados y varianzas condicionales pasadas, ACD modela $\psi_i$ como una combinación de duraciones observadas pasadas y duraciones esperadas pasadas — **misma estructura matemática, aplicada a un objeto distinto** (duración en vez de varianza).

**[A]** Tsay también reescribe el modelo ACD(r,s) en una forma tipo ARMA con innovaciones no gaussianas (Ec. 5.35), lo cual permite derivar condiciones de estacionariedad de forma análoga a como se hace con ARMA/GARCH — este informe no reproduce esa derivación en detalle, siguiendo la instrucción de mantener la sección a nivel conceptual, salvo el caso EACD(1,1) más simple, cuyo resultado clave es que la media incondicional de la duración es $\mu_x = \omega/(1-\gamma_1-\omega_1)$ (Ec. 5.38), un resultado directamente análogo a la media incondicional de un GARCH(1,1).

**[A] Extensión con distribución generalizada gamma (GACD).** Tsay menciona brevemente que, en la literatura estadística, la intensidad de negociación suele expresarse en términos de una **función de hazard** (explicada solo a nivel conceptual imprescindible, siguiendo la instrucción de reducir al mínimo el desarrollo de hazard): el hazard de un modelo EACD es constante en el tiempo, y el de un WACD es una función monótona (siempre creciente o siempre decreciente, según el parámetro de forma $\alpha$ sea mayor o menor a 1). **[A]** Zhang, Russell y Tsay (2001) proponen usar una distribución generalizada gamma para $\varepsilon_i$ (modelo **GACD**), que permite formas de hazard más flexibles, incluyendo forma de U o de U invertida — útil porque, en la práctica, la intensidad de negociación de una acción no necesariamente es monótona en el tiempo.

### 12.5 Simulación y estimación — resumen mínimo

**[A]** Tsay ilustra el comportamiento de un ACD(1,1) simulando 500 observaciones del proceso $\psi_i = 0.3 + 0.2x_{i-1} + 0.7\psi_{i-1}$, bajo dos distribuciones distintas para $\varepsilon_i$ (Weibull estandarizada con $\alpha=1.5$, y generalizada gamma con $\kappa=1.5,\alpha=0.5$), mostrando que ambas series simuladas presentan dependencia serial clara y formas de distribución visiblemente distintas entre sí. La estimación se realiza por **máxima verosimilitud condicional**, con funciones de log-verosimilitud específicas para WACD (Ec. 5.41) y GACD (Ec. 5.42); Tsay muestra que, al re-estimar el mismo proceso simulado, los parámetros recuperados son razonablemente cercanos a los verdaderos (Tabla 5.7). **No se reproduce aquí el desarrollo matemático completo de la estimación**, siguiendo la instrucción explícita de omitir/reducir al mínimo este contenido si no aporta comprensión transferible más allá de lo ya explicado.

### 12.6 Ejemplo empírico real: duraciones de IBM (Ejemplo 5.4)

**[A] Instrumento, mercado, frecuencia, período, tamaño de muestra:** IBM, TORQ, NYSE, cinco días de negociación consecutivos (1 al 7 de noviembre de 1990), 3.534 duraciones positivas (transacción-a-transacción, ya ajustadas por el patrón diurno de la Sección 12.3).

**[A] Modelo WACD(1,1) ajustado (Ec. 5.43):**

$$x_i = \psi_i\varepsilon_i, \qquad \psi_i = 0.169 + 0.064\,x_{i-1} + 0.885\,\psi_{i-1}$$

con $\varepsilon_i$ siguiendo una Weibull estandarizada de parámetro $\hat\alpha=0.879\,(0.012)$. Todos los coeficientes son significativos al 1% (t-ratios $>4.2$). La **persistencia** ($\hat\gamma_1+\hat\omega_1 \approx 0.949$) es alta — indicando que los períodos de intensidad de negociación alta o baja tienden a mantenerse por un tiempo, no a disolverse instantáneamente. La duración esperada incondicional del modelo, $0.169/(1-0.064-0.885)=3.31$ segundos, es cercana a la media muestral observada (3.29 segundos). **[A]** Con $\hat\alpha=0.879<1$, la función de hazard es monótonamente decreciente (a ritmo lento): cuanto más tiempo pasó ya sin una transacción, menor la probabilidad instantánea de que ocurra la próxima en el siguiente instante — un patrón compatible con "cuanto más silencio, más silencio esperado".

**[A]** Los residuos estandarizados $\hat\varepsilon_i = x_i/\hat\psi_i$ no muestran autocorrelación serial significativa (Ljung–Box $Q(10)=4.96$, $Q(20)=10.75$), ni tampoco su serie al cuadrado ($Q(10)=6.20$, $Q(20)=11.16$) — el modelo se considera adecuado según estos diagnósticos.

**[A]** Un modelo GACD(1,1) alternativo da resultados cualitativamente similares (duración esperada 3.52 segundos, persistencia 0.96), con parámetros de forma $\kappa=4.248\,(1.046)$, $\alpha=0.395\,(0.053)$.

### 12.7 Clasificación de transferencia a barras de tiempo fijo

**[B]**

- **Duración transacción-a-transacción y modelo ACD en su forma original:** **NO DIRECTAMENTE OBSERVABLE CON OHLCV DE BARRAS** — es, literalmente, la información que la construcción de una barra de tiempo fijo destruye por definición (Sección 12.2).
- **Idea conceptual de "persistencia de intensidad" (análoga a persistencia de volatilidad en GARCH):** **PARCIALMENTE RELEVANTE** — podría, en principio, aplicarse a otras variables agregadas por barra que sí estén disponibles (por ejemplo, número de trades por barra, si existiera esa columna), pero eso ya no sería el mismo objeto que estudia Tsay.
- **Ajuste de patrón diurno:** **DIRECTAMENTE RELEVANTE** — la lógica de remover un componente determinístico ligado a la hora del día antes de estudiar la dinámica restante es aplicable a cualquier variable de barras (volumen, rango, volatilidad), no solo a duraciones.

---

## 13. Nonlinear Duration Models (5.6)

**Estudiado a nivel conceptual**, conectando con el Capítulo 4.

### 13.1 La pregunta

**¿La dinámica de la duración puede cambiar según estemos en un período de trading intenso o lento?**

Esto es exactamente la misma pregunta que el Capítulo 4 planteó para la media condicional de los retornos (TAR, STAR, Markov Switching), pero aplicada ahora a la duración esperada condicional $\psi_i$ en vez de a un retorno.

### 13.2 Evidencia empírica de Tsay: nonlinealidad en las duraciones de IBM

**[A] Instrumento, mercado, frecuencia, período:** mismo dataset que la Sección 12.6 — IBM, TORQ, NYSE, 1 al 7 de noviembre de 1990.

**[A]** Tsay aplica los tests de no linealidad del Capítulo 4 (ver el informe de ese capítulo) a los **residuos normalizados** $\hat\varepsilon_i$ del modelo WACD(1,1) ajustado en la Sección 12.6. Usando un AR(4) como base, el test **Ori-F** (variante del F-test de Tsay, orientado a no linealidad cuadrática tipo umbral) **no detecta** no linealidad (p-valores entre 0.915 y 0.998 según el rezago probado), lo cual es coherente con que el modelo lineal (WACD) ya diagnosticó bien la serie. **Sin embargo**, el test **TAR-F**, con distintos delays, **sí sugiere no linealidad fuerte**: p-valores de 0.006, 0.008 y 0.008 para delays 1, 2 y 3 respectivamente (Tabla 5.8, parte a, de Tsay), aunque no para delay 4 (p=0.915).

### 13.3 El modelo threshold duration resultante

**[A]** Con esta evidencia, Tsay ajusta un **modelo de duración con umbral de 2 regímenes**, con variable de umbral $x_{i-1}$ (la duración ajustada del período inmediatamente anterior) y umbral estimado en $3.79$:

$$\psi_i = \begin{cases} 0.020 + 0.257\,x_{i-1} + 0.847\,\psi_{i-1}, \;\; \varepsilon_i \sim W(0.901) & \text{si } x_{i-1}\leq 3.79 \\ 1.808 + 0.027\,x_{i-1} + 0.501\,\psi_{i-1}, \;\; \varepsilon_i \sim W(0.845) & \text{si } x_{i-1}>3.79 \end{cases}$$

con 2.503 observaciones en el primer régimen y 1.030 en el segundo. **[A]** Tras ajustar este modelo, los residuos normalizados ya **no** muestran evidencia de no linealidad residual (Tabla 5.8, parte b) — ni tampoco autocorrelación serial significativa en los propios residuos ni en su serie al cuadrado (Ljung–Box con p-valores altos). El modelo se refina eliminando parámetros no significativos, obteniendo una versión más simple con los mismos dos regímenes.

**[A] Interpretación textual de Tsay:** si se clasifican los dos regímenes como "heavy trading" (negociación intensa, duraciones cortas) y "thin trading" (negociación liviana, duraciones largas), el modelo sugiere que **la dinámica de las duraciones intradiarias es distinta entre períodos de negociación intensa y liviana**, incluso después de haber removido el patrón diurno determinístico. Tsay atribuye esto, de forma general y sin verificación causal directa dentro de esta sección, a que la actividad de mercado suele estar impulsada por la llegada de noticias y otra información.

### 13.4 Concepto: threshold duration, heavy/thin trading

- **Threshold duration:** el punto de corte (aquí, 3.79 unidades de duración ajustada) que separa los dos regímenes de dinámica de duración.
- **Heavy trading:** régimen de duraciones cortas — negociación frecuente.
- **Thin trading:** régimen de duraciones largas — negociación esporádica.
- **Modelo por regímenes aplicado a duración:** exactamente la misma lógica TAR del Capítulo 4 (Sección 4.2 del informe de ese capítulo), pero con la variable de umbral y la variable modelada siendo ambas duraciones, no retornos.

### 13.5 No trasladar automáticamente estos regímenes a barras temporales

**[B]** El "régimen" de este modelo threshold ACD está definido por el valor de la duración inmediatamente anterior — un objeto que, como se estableció en la Sección 12.2, **no existe** en un dataset de barras de tiempo fijo. Por lo tanto, este resultado concreto (el umbral de 3.79, los dos conjuntos de coeficientes) **no puede aplicarse** a un contexto de barras OHLCV sin antes redefinir por completo qué sería la "variable de duración" en ese contexto — cosa que este capítulo no hace ni recomienda.

### 13.6 Clasificación de transferencia a barras de tiempo fijo

**[B]** **NO DIRECTAMENTE OBSERVABLE CON OHLCV DE BARRAS**, con la misma razón que la Sección 12.7: depende enteramente de duraciones transacción-a-transacción. La idea general —que la dinámica de una variable de intensidad puede cambiar según el propio nivel reciente de esa variable— es, como mucho, **PARCIALMENTE RELEVANTE** como esquema conceptual trasladable a otras variables agregadas por barra (por ejemplo, volumen o rango), sin que Tsay lo demuestre ni lo sugiera para ese contexto.

---

## 14. Price Change + Duration — Bivariate Models (5.7)

**Estudio conceptual pero serio**, según el enunciado.

### 14.1 Motivación

**[A]** Muchas transacciones intradiarias no producen cambio de precio (documentado extensamente en la Sección 7 de este informe). Esas transacciones "sin cambio" son muy relevantes para entender la **intensidad de negociación**, pero no aportan información directa sobre el **movimiento del precio**. Para simplificar el problema, Tsay propone enfocarse únicamente en las transacciones que **sí** resultan en un cambio de precio, y modelar conjuntamente el precio y el tiempo hasta ese cambio: el modelo **PCD** (*price change and duration*).

**[A] Redefinición de notación para esta sección:** ahora $t_i$ es el instante calendario del $i$-ésimo **cambio de precio** (no de cualquier transacción), $\Delta t_i = t_i - t_{i-1}$ es la duración **entre cambios de precio**, $N_i$ es el **número de transacciones sin cambio de precio** ocurridas en el intervalo $(t_{i-1}, t_i)$ (una variable nueva, que representa la intensidad de negociación durante el período "silencioso"), $D_i$ es la dirección del $i$-ésimo cambio de precio, y $S_i$ es su tamaño en ticks. El precio evoluciona como $P_{t_i} = P_{t_{i-1}} + D_iS_i$.

**[A] Remark de Tsay sobre el tamaño de muestra:** enfocarse solo en transacciones con cambio de precio reduce dramáticamente el tamaño de la muestra. Para IBM (TORQ, noviembre 1990 – enero 1991), de 60.265 transacciones intradiarias totales, **solo 19.022 resultaron en un cambio de precio**. **[A]** Tsay señala además que, a diferencia de la duración transacción-a-transacción (que sí tiene patrón diurno, ver Sección 7.3), **no hay patrón diurno en la duración entre cambios de precio** — una distinción empírica específica de este subconjunto de datos.

### 14.2 La estructura real del modelo PCD

**[A]** La descomposición conjunta que usa Tsay (Ec. 5.47) es:

$$f(t_i, N_i, D_i, S_i|\mathcal{F}_{i-1}) = f(S_i|D_i,N_i,t_i,\mathcal{F}_{i-1})\,f(D_i|N_i,t_i,\mathcal{F}_{i-1})\,f(N_i|t_i,\mathcal{F}_{i-1})\,f(t_i|\mathcal{F}_{i-1})$$

Es decir: **cuatro** modelos condicionales encadenados, no tres. La descomposición conceptual del enunciado (ocurrencia → dirección → tamaño, Sección 11) sigue apareciendo, pero incorporada dentro de un esquema más amplio que **primero** modela **cuándo** ocurre el próximo cambio de precio ($t_i$), **luego cuánta actividad "silenciosa" hubo mientras tanto** ($N_i$), y **recién después** la dirección y el tamaño del cambio. **[A]** Esta estructura corrige y completa la formulación simplificada planteada en el enunciado del estudio: no es solo "¿cambia?, ¿sube o baja?, ¿qué tamaño?", sino que incorpora explícitamente el **tiempo hasta el próximo cambio** y el **número de trades sin cambio en el medio** como parte de la misma cadena de modelos.

**[A] Cada sub-modelo, brevemente (sin reproducir la estimación completa, por instrucción del enunciado):**

- **Duración entre cambios de precio** ($\ln t_i$): una regresión lineal simple con rezagos de $\ln(\Delta t_{i-1})$ y del tamaño anterior $S_{i-1}$ (Ec. 5.48).
- **$N_i$ (número de trades sin cambio):** modelo en dos partes — un modelo logit para la probabilidad de que $N_i=0$ (Ec. 5.49), y, condicional a $N_i>0$, una distribución geométrica para su magnitud (Ec. 5.50).
- **Dirección $D_i$:** modelada mediante el signo de una variable normal con media $\mu_i$ (que depende de $D_{i-1}$, capturando reversión) y varianza $\sigma_i^2$ (que depende de la suma de los últimos cuatro signos, capturando "tendencia local" — ver más abajo).
- **Tamaño $S_i$:** distribución de Poisson desplazada en 1, con parámetros distintos según la dirección (asimetría entre subidas y bajadas), dependientes de $N_i$, la duración y el tamaño anterior.

**[A] Detalle relevante del modelo de dirección:** Tsay explica que la **media** de la variable normal que determina el signo de $D_i$ depende negativamente de $D_{i-1}$ (capturando reversión de precio, igual que en las secciones anteriores), mientras que la **varianza** depende del valor absoluto de la suma de los últimos cuatro signos de dirección. La intuición que da el propio Tsay: para una distribución normal con media fija, aumentar la varianza aumenta la probabilidad de que una extracción aleatoria tenga el mismo signo que las anteriores — lo cual permite que el modelo capture **tendencias locales ocasionales** en el precio (rachas de subidas o de bajadas), incluso mientras mantiene, en promedio, el efecto de reversión de la media.

### 14.3 Qué intenta separar el modelo

$$\text{cuándo cambia el precio} \quad \text{puede contener información relacionada con} \quad \text{cómo cambia el precio}$$

**[B]** Ésta es la idea central del PCD, formulada en los términos del enunciado: el modelo no trata la duración entre cambios de precio como un dato aislado de la dirección y el tamaño de esos cambios — los modela **conjuntamente**, permitiendo que la duración (y el número de trades sin cambio mientras tanto, $N_i$) informe sobre la dirección y magnitud del próximo cambio, y viceversa.

### 14.4 Ejemplo empírico de Tsay: Ejemplo 5.5

**[A] Instrumento, mercado, frecuencia, período, tamaño de muestra:** IBM, TORQ, NYSE, **21 de noviembre de 1990** (un solo día de negociación), 726 transacciones totales en horario regular, de las cuales **194 resultaron en cambio de precio** (la base de este ejemplo). Estimación por **MCMC** (Markov Chain Monte Carlo, técnica que el propio Tsay remite al Capítulo 12 del libro, no desarrollada aquí), con 9.500 iteraciones.

**[A] Resultados exactos reportados por Tsay:**

- **Duración entre cambios de precio:** $\ln(\Delta t_i) = 4.023 + 0.032\ln(\Delta t_{i-1}) - 0.025\,S_{i-1} + 1.403\,\epsilon_i$, con desviaciones estándar de los coeficientes de 0.415, 0.073, 0.384 y 0.073 respectivamente. **[A]** Tsay indica explícitamente que este modelo **no mostró dependencia dinámica** significativa en la duración (los coeficientes de $\ln(\Delta t_{i-1})$ y $S_{i-1}$ no son grandes en relación con sus errores estándar).
- **$N_i$ (trades sin cambio):** $\Pr(N_i>0|t_i,\mathcal{F}_{i-1}) = \mathrm{logit}[-0.637 + 1.740\ln(\Delta t_i)]$ — el coeficiente positivo y significativo de $\ln(\Delta t_i)$ confirma, como es de esperar, que cuanto más larga la duración hasta el próximo cambio de precio, más trades "silenciosos" ocurren mientras tanto.
- **Dirección $D_i$:** $\mu_i = 0.049 - 0.840\,D_{i-1} - 0.004\ln(\Delta t_i)$; $\ln(\sigma_i) = 0.244\,|D_{i-1}+D_{i-2}+D_{i-3}+D_{i-4}|$. El coeficiente de $D_{i-1}$ en la media es negativo, grande en magnitud y altamente significativo — **reversión de precio clara**, consistente con el bid–ask bounce. El coeficiente de la ecuación de varianza es marginalmente significativo, exactamente como Tsay anticipa conceptualmente.
- **Tamaño $S_i$:** para bajadas, $\ln(\lambda_{d,i}) = 1.024 - 0.327\,N_i + 0.412\ln(\Delta t_i) - 4.474\,S_{i-1}$; para subidas, $\ln(\lambda_{u,i}) = -3.683 - 1.542\,N_i + 0.419\ln(\Delta t_i) + 0.921\,S_{i-1}$. **[A]** Tsay destaca el **coeficiente negativo de $N_i$** en ambas ecuaciones: cuantos más trades "sin cambio" hubo en el intervalo (interpretado por Tsay como evidencia de que no llegó nueva información relevante en ese lapso), menor el tamaño esperado del próximo cambio de precio cuando finalmente ocurre.

**[A]** Tsay reconoce explícitamente que una muestra de 194 observaciones de un solo día puede no contener suficiente información sobre la dinámica de negociación de IBM, pero considera que los resultados son "razonables". **[A]** McCulloch y Tsay (2000) extienden este modelo a un marco jerárquico para manejar los datos completos de los 63 días (más de 19.000 observaciones), donde varios parámetros —insignificantes en la muestra de un solo día— se vuelven significativos; por ejemplo, el coeficiente de $\ln(\Delta t_{i-1})$ en el modelo de duración resulta, en la muestra ampliada, pequeño pero significativo (entre 0.04 y 0.1 según la especificación).

### 14.5 Qué información se pierde si solo se dispone de OHLCV agregado

**[B]** El modelo PCD depende, en su totalidad, de la secuencia completa de transacciones individuales con cambio de precio, del conteo exacto de trades "silenciosos" entre cada cambio, y del instante calendario exacto de cada cambio. **Ninguno** de estos tres insumos está disponible en un dataset OHLCV de barras estándar. La variable $N_i$ en particular —el número de trades sin cambio de precio dentro de un intervalo, usada por Tsay como proxy de "ausencia de nueva información"— es un ejemplo directo de la distinción "volumen vs. número de trades" discutida en la Sección 8.3 de este informe: es información sobre la **estructura** de la actividad de mercado que un dataset de solo Open/High/Low/Close/Volume no puede reconstruir.

### 14.6 Clasificación de transferencia a barras de tiempo fijo

**[B]** **NO DIRECTAMENTE OBSERVABLE CON OHLCV DE BARRAS.** El modelo PCD completo depende de información (timestamps exactos de cambios de precio, conteo de trades sin cambio entre ellos) que no existe en un dataset OHLCV estándar. La única idea que sobrevive como esquema conceptual **[B]** es que "cuándo" y "cómo" cambia el precio pueden estar relacionados — algo que, en principio, podría explorarse de forma completamente distinta sobre barras (por ejemplo, relacionando el tiempo desde el último movimiento significativo de precio con la magnitud del próximo movimiento), pero eso sería una adaptación propia, no algo que Tsay describa.

---

## 15. Application e Intervention Analysis (5.8)

**Estudiado seriamente pero de forma selectiva**, según el enunciado.

### 15.1 Qué variable modela Tsay y por qué

**[A] Instrumento, mercado, frecuencia, período, tamaño de muestra:** rango diario del logaritmo del precio (*daily range of log price*) de la acción de **Apple**, del 4 de enero de 1999 al 20 de noviembre de 2007 (Yahoo Finance), 2.235 observaciones. Esta serie fue analizada previamente en Tsay (2009). Apple tuvo dos splits 2-por-1 durante el período (21 de junio de 2000 y 28 de febrero de 2005), pero **no requieren ajuste** porque el rango del logaritmo del precio (una diferencia de logaritmos) es invariante a splits proporcionales.

**Por qué usar el rango como medida de volatilidad.** **[A]** Tsay señala que el rango diario del precio en logaritmo ha sido utilizado en la literatura como una alternativa robusta para modelar volatilidad (remite a la discusión del Capítulo 3 del propio libro y a Chou, 2005, sobre el modelo CARR — *conditional autoregressive range*). **[B]** Una forma sencilla de entender su atractivo es que el rango utiliza información sobre el movimiento ocurrido dentro del período —el máximo y el mínimo— en lugar de describir la variabilidad únicamente mediante el cambio entre dos cierres. Esto no significa que High−Low sea inmune a datos erróneos o valores atípicos: un High o Low incorrecto puede alterar directamente el rango.

**Estadísticos descriptivos.** **[A]** Media 0.0407, desvío estándar 0.0218, mínimo 0.0068, máximo 0.1468; asimetría muestral 1.3, curtosis en exceso 2.13. La ACF del rango decae **lentamente** y es altamente significativa — consistente con la persistencia de volatilidad ya documentada en el Capítulo 3.

### 15.2 Qué modelos compara

**[A]** Tsay ajusta EACD(1,1), WACD(1,1) y GACD(1,1) —los mismos modelos ACD de la Sección 12 de este informe, ahora aplicados no a duraciones entre trades sino al **rango diario**, tratado como una serie de valores positivos con estructura de dependencia similar (Tabla 5.9 de Tsay). Los parámetros de la ecuación de "duración" (aquí, del rango esperado condicional) son estables entre los tres modelos, excepto que la constante del EACD resulta estadísticamente insignificante y ese modelo ajusta ligeramente peor que los otros dos. **[A]** Entre WACD y GACD, Tsay prefiere **levemente** el GACD, por mejor ajuste y mayor flexibilidad — no por una diferencia dramática.

**[A] Hallazgo interpretativo de Tsay:** los parámetros de forma estimados ($\hat\alpha$) son **mayores a 1** tanto para WACD como para GACD, lo cual implica una función de hazard **monótonamente creciente** — Tsay conecta esto explícitamente con la idea de **agrupamiento de volatilidad** (*volatility clustering*, ya visto en el Capítulo 3): un rango grande tiende a ser seguido por otro rango grande, así como, en un ACD de duraciones con hazard creciente, cuanto más tiempo pasó sin una transacción, mayor la probabilidad instantánea de que ocurra la próxima ya inminente.

### 15.3 Refinamiento: modelo threshold ACD (TWACD)

**[A]** Tsay refina el GACD(1,1) usando un **modelo threshold WACD de 2 regímenes** (TWACD), donde la diferencia principal entre regímenes resulta estar en el **parámetro de forma** de la distribución Weibull, no en los coeficientes de la ecuación del rango esperado. **[A]** El umbral se selecciona por búsqueda sobre percentiles muestrales (60 a 95, Tabla 5.10 de Tsay), maximizando la verosimilitud; el umbral elegido corresponde al **percentil 70** (valor 0.04753), con delay $d=1$ (comparado explícitamente contra $d=2,3$, que dan verosimilitudes menores). El modelo final:

$$\psi_i = 0.0013 + 0.1539\,x_{i-1} + 0.8131\,\psi_{i-1}, \qquad \varepsilon_i \sim \begin{cases} W(2.2756) & \text{si } x_{i-1}\leq 0.04753 \\ W(2.7119) & \text{si } x_{i-1}>0.04753 \end{cases}$$

**[A]** Los residuos estandarizados de este modelo no muestran autocorrelación serial significativa, ni en la serie original ni en su serie al cuadrado — diagnóstico adecuado.

### 15.4 Qué es intervention analysis, explicado desde cero

**Idea simple:** algo externo cambia las reglas del sistema en una fecha conocida. Antes de la fecha $t_0$ rige una regla A; después, rige una regla B — y queremos medir el efecto estadístico de ese cambio de reglas.

**[A]** Tsay usa el marco de intervention analysis de Box y Tiao (1975), aplicado aquí al cambio institucional de **decimalización** del mercado de acciones de EE.UU.: antes del 29 de enero de 2001, los precios se movían en múltiplos de $\$1/16$; a partir de esa fecha, en múltiplos de un centavo (sistema decimal) — el mismo cambio de tick size ya documentado en la Sección 7.2 de este informe con los datos de IBM.

### 15.5 El cambio institucional concreto y el efecto estadístico estudiado

**[A]** Para Apple, el momento de intervención es $t_0=522$, correspondiente al **26 de enero de 2001** (el último día de negociación antes del cambio de tick size). Se define un indicador $I_i^{(t_0)}=1$ si $i\leq t_0$ (antes del cambio) y $0$ en caso contrario. **[A]** La hipótesis de Tsay, razonable a priori: un tick size más grande tiende a aumentar el rango diario observado del precio, por lo que se espera que el rango esperado condicional sea **mayor antes** de la intervención. El modelo (Ec. 5.54):

$$\psi_i = \alpha_0 + \gamma\,I_i^{(t_0)} + \alpha_1 x_{i-1} + \beta_1\psi_{i-1}$$

donde $\gamma$ representa la disminución en la duración/rango esperado debida a la decimalización; se espera $\gamma>0$.

**[A] Resultado exacto:**

$$\psi_i = 0.0021 + 0.0011\,I_i^{(522)} + 0.1595\,x_{i-1} + 0.7828\,\psi_{i-1}$$

con errores estándar 0.0004, 0.0003, 0.0177 y 0.0264 respectivamente. **El coeficiente $\hat\gamma$ es significativo al 1%**, y **positivo, como se esperaba**: la decimalización efectivamente redujo el rango esperado (la medida de volatilidad usada) de la acción de Apple. Los residuos estandarizados del modelo con intervención no muestran autocorrelación serial significativa.

**[A] Conclusión textual de Tsay:** "esta sencilla analysis muestra que, como se esperaba, adoptar el sistema decimal redujo la volatilidad de la acción de Apple."

### 15.6 Diagnóstico y qué NO puede concluirse causalmente

**[A]** El diagnóstico de este modelo se basa en la falta de autocorrelación serial residual (Ljung–Box en residuos y residuos al cuadrado), tal como en los ejemplos anteriores del capítulo.

**[B] Qué NO puede concluirse sin condiciones adicionales**, aunque el resultado sea estadísticamente sólido dentro de este ejemplo concreto:

- No se demuestra aquí una relación causal universal "menor tick size → siempre menor volatilidad" para cualquier activo o mercado; se trata de un resultado documentado sobre **una acción concreta** (Apple), en un **evento institucional concreto** (la decimalización de EE.UU. de 2001), con un **modelo estadístico concreto** (TWACD con intervención).
- El propio diseño del intervention analysis asume que **no hubo otros cambios simultáneos** relevantes en torno a esa fecha que pudieran confundirse con el efecto atribuido a la decimalización — un supuesto razonable dado que se trata de un cambio regulatorio de fecha conocida y aplicado a todo el mercado, pero no verificado exhaustivamente dentro de esta sección del libro.
- El resultado es específico de la serie, frecuencia (diaria) y período analizados; no se generaliza aquí a ningún otro instrumento, y mucho menos a futuros electrónicos modernos, que no tuvieron un evento de decimalización equivalente al de acciones de EE.UU. en 2001.

### 15.7 Conexión conceptual con structural break y cambios regulatorios

**[B]** El intervention analysis es, conceptualmente, una forma de **structural break** con fecha de quiebre **conocida de antemano** (por venir de una decisión regulatoria documentada), en contraste con un structural break genérico, donde la fecha del quiebre podría ser desconocida y debería estimarse. La lectura general, coherente con —pero no textual de— este ejemplo de Tsay: **una propiedad estadística observada en datos financieros puede cambiar porque cambió la infraestructura o las reglas de negociación del mercado, no necesariamente porque cambió algo en la "psicología" o el comportamiento económico fundamental del activo.** Esta distinción es relevante como advertencia general para cualquier análisis de series de tiempo financieras de largo plazo, sin que implique nada específico sobre MNQ o sobre ningún cambio institucional en particular en los mercados de futuros — que no se investiga en esta tarea.

### 15.8 Clasificación de transferencia a barras de tiempo fijo

**[B]** El **rango diario de log-precio** usado como variable modelada en esta sección **es DIRECTAMENTE RELEVANTE y directamente calculable** sobre cualquier barra OHLCV (rango = High − Low, o su versión logarítmica), a cualquier frecuencia, incluidas barras de 1 minuto — a diferencia de casi todo lo demás visto en este capítulo. El **marco de intervention analysis** (indicador de antes/después de una fecha de cambio institucional conocida) es también **DIRECTAMENTE RELEVANTE** como herramienta metodológica general, aplicable a cualquier cambio de reglas de mercado documentado (cambio de tick size, cambio de horario de sesión, cambio de especificación de contrato) que pudiera existir en la historia de MNQ — sin que este capítulo identifique ningún evento concreto de ese tipo para MNQ.

---

## 16. Qué conceptos sobreviven a la agregación a barras de tiempo

**[B]** Tabla conceptual solicitada por el enunciado, revisada críticamente a la luz de todo lo desarrollado en las secciones anteriores. Toda la tabla es una construcción propia [B]; ninguna fila proviene textualmente de Tsay.

| Fenómeno | Transaction data | ¿Puede dejar huella en barra 1m? | ¿Se observa directamente con OHLCV? |
|---|---|---|---|
| Bid–ask bounce | Sí (Sec. 6.2, 6.5) | Posiblemente, vía el Close (última transacción del minuto) — magnitud desconocida sin medición (PREGUNTA EMPÍRICA, Sec. 6.6) | No directamente |
| Spread bid–ask vigente | Sí, con quotes (Sec. 6.1) | Puede influir en el rango y en el bounce del Close, pero no queda registrado como cantidad propia | No, salvo que el dataset incluya quotes por separado |
| Tick size / discreción del precio | Sí (Sec. 7.2) | Sí — afecta directamente Open/High/Low/Close | Sí, indirectamente (los precios de la barra siguen siendo múltiplos de tick) |
| Trade duration (Δt entre transacciones) | Sí (Sec. 12.2) | Información agregada/perdida por completo | No |
| Number of trades por intervalo | Sí (Sec. 7.4, 8.3) | Puede resumirse si el proveedor de datos lo incluye aparte | No necesariamente (no es estándar en OHLCV puro) |
| Volume | Sí | Sí, se preserva como suma agregada | Sí |
| Diurnal pattern | Sí (Sec. 7.3) | Sí — observable agregando por hora del día | Sí |
| Nonsynchronous trading (entre instrumentos) | Sí (Sec. 5) | Puede afectar barras multi-instrumento con el mismo timestamp de reloj | Depende del caso — no verificable sin datos de varios instrumentos a resolución fina |
| Ordered probit / decomposition (dirección y tamaño del cambio transacción a transacción) | Sí (Sec. 10, 11) | La lógica conceptual (ocurrencia/dirección/magnitud) puede reformularse sobre Close a Close de barras consecutivas | No en su forma original; sí como adaptación conceptual [B] |
| ACD / duración esperada condicional | Sí (Sec. 12) | No aplicable directamente — la separación entre barras es siempre constante | No |
| PCD (precio + duración conjuntos) | Sí (Sec. 14) | No aplicable — depende de N_i y timestamps exactos de cambios de precio | No |
| Rango diario / intradía como proxy de volatilidad | Sí (Sec. 15) | Sí — directamente calculable (High − Low) a cualquier frecuencia | Sí |
| Intervention analysis (cambios institucionales) | Sí (Sec. 15.4–15.7) | Sí — la técnica es aplicable a cualquier serie de barras si se conoce la fecha del cambio | Sí, como metodología (no depende de transaction data) |

**Revisión crítica de la tabla, según lo pedido:** a diferencia de una primera intuición simplificada, esta versión distingue explícitamente entre "puede dejar huella" (columna intermedia, indicando persistencia posible pero de magnitud desconocida) y "se observa directamente" (columna final, indicando si el propio dato de barras alcanza para medirlo sin suposiciones adicionales). Varias filas —bid–ask bounce, spread, nonsynchronous trading multivariado— tienen respuestas condicionales ("depende", "posiblemente") porque el propio capítulo de Tsay no analiza datos agregados en barras en ningún momento; cualquier afirmación categórica sobre su persistencia sería una extrapolación no respaldada por el texto.

---

## 17. Qué preguntas abre este capítulo sobre Last, Bid y Ask

Tal como exige el enunciado, esta sección responde **preguntas**, no decisiones. Todo lo que sigue es **PREGUNTA ABIERTA / EXPERIMENTO FUTURO**, sin adoptar ninguna conclusión.

1. **¿Qué mide Last?** El precio de la transacción más reciente efectivamente ejecutada — un hecho consumado, pero potencialmente "contaminado" por si esa transacción se ejecutó cerca del bid o del ask (Sección 4, 6).

2. **¿Qué mide Bid?** La mejor oferta de compra disponible en este instante — una intención, no una operación ejecutada.

3. **¿Qué mide Ask?** La mejor oferta de venta disponible en este instante — también una intención, no una operación ejecutada.

4. **¿Qué información ofrece el spread?** Una medida del "costo de la inmediatez" en ese instante — cuánto más caro es comprar ya mismo respecto de vender ya mismo (Sección 6.1, 6.7).

5. **¿Qué información se pierde si solo se conserva Last?** El estado completo del libro de órdenes en cada instante (bid, ask, profundidad); la posibilidad de distinguir si un movimiento de Last se debió a nueva información o a simple alternancia bid/ask (bounce); y, en consecuencia, la posibilidad de filtrar ese bounce antes de calcular retornos.

6. **¿Qué ruido puede introducir Last?** El bid–ask bounce documentado en la Sección 6: alternancia entre precios de compra y venta que puede generar autocorrelación negativa aparente sin que haya cambiado el "valor" del activo.

7. **¿Qué información adicional tendría Bid/Ask?** El spread vigente en cada instante, y —potencialmente— la posibilidad de calcular un Mid como referencia menos afectada por el bounce (aunque con las limitaciones de la pregunta 9).

8. **¿Por qué Mid tampoco es automáticamente la respuesta?** Porque Mid es un promedio **calculado**, no necesariamente un precio al que se puede ejecutar una operación real; usar Mid como si fuera el "precio verdadero ejecutable" podría subestimar el costo real de operar (Sección 4.8).

9. **¿Qué precio sería relevante para modelar** (es decir, para intentar predecir el "estado" del mercado o su dirección futura)? No se determina aquí — podría argumentarse a favor de Mid (menos ruido de bounce) o de Last (lo que efectivamente se transa), pero cualquiera de esas dos posturas es una hipótesis, no una conclusión de este capítulo.

10. **¿Qué precio sería relevante para simular ejecución** (es decir, para estimar cuánto costaría realmente entrar o salir de una posición)? Tampoco se determina aquí — intuitivamente, la ejecución real cruza el spread (se compra al ask, se vende al bid), por lo que Mid subestimaría el costo, pero esto queda como hipótesis a evaluar, no como conclusión adoptada.

**No se decide aquí cuál de estos precios (Last, Bid, Ask, Mid) debe usarse para IRIS.** Esta sección deja constancia de las preguntas que abre el capítulo, sin resolverlas.

---

## 18. Mapa Capítulos 1 → 2 → 3 → 4 → 5

**[B]** Organización propia, construida sobre el contexto mínimo provisto para este estudio y sobre el contenido de este capítulo.

- **Capítulo 1 — ¿qué variable y distribución estudiamos?** Retornos y sus propiedades distribucionales: colas pesadas, asimetría, baja relación señal/ruido en la media.
- **Capítulo 2 — ¿qué dependencia lineal existe?** Se estableció: $\text{autocorrelación} \neq \text{independencia} \neq \text{predictibilidad económica}$.
- **Capítulo 3 — ¿cómo cambia la volatilidad?** Se mostró que puede existir mucha estructura en $|r_t|$, $r_t^2$, $\sigma_t$ aunque exista poca en la dirección.
- **Capítulo 4 — ¿qué dependencia no lineal puede existir?** Se estableció: $\text{detectar estructura} \neq \text{saber modelarla} \neq \text{predecir} \neq \text{ganar dinero}$.
- **Capítulo 5 — ¿cuánta de la estructura observada depende del propio mecanismo de negociación y de cómo medimos el mercado?** Se introdujo una pregunta todavía más básica que las anteriores: antes de preguntarse si hay dependencia lineal, no lineal, o volatilidad cambiante en los datos, hay que preguntarse si el propio dato observado (Last, la barra, el retorno calculado) es una medición limpia del mercado o si ya viene parcialmente moldeado por el mecanismo de negociación mismo (bid–ask bounce, nonsynchronous trading, discreción del tick, patrón diurno).

**Mapa:**

$$\text{retornos} \rightarrow \text{dependencia} \rightarrow \text{volatilidad} \rightarrow \text{no linealidad} \rightarrow \text{microestructura}$$

**[B] Aclaración importante, coherente con el propio capítulo:** estas capas no son independientes ni estrictamente secuenciales en la práctica. El propio bid–ask bounce del Capítulo 5, por ejemplo, produce exactamente el tipo de autocorrelación negativa de rezago 1 que el Capítulo 2 enseñó a detectar (Sección 5–6 de este informe) — es decir, la Sección 5 del mapa no reemplaza a la Sección 2, sino que **agrega una explicación alternativa** para un fenómeno que ya sabíamos detectar: una autocorrelación negativa de rezago 1 podría deberse a dependencia económica genuina (Capítulo 2), o podría deberse, total o parcialmente, a bid–ask bounce o nonsynchronous trading (Capítulo 5) — y este capítulo no permite, por sí solo, distinguir entre ambas posibilidades para ningún instrumento concreto sin evidencia adicional.

---

## 19. Implicancias para futuros — [A] separado de [B]

### 19.1 [A] Lo que Tsay efectivamente muestra, sin generalizar a futuros

1. En un modelo teórico simplificado (Roll, 1984), y bajo supuestos explícitos, el bid–ask spread introduce matemáticamente una autocorrelación negativa exacta de $-0.5$ en el rezago 1 de los cambios de precio observados, y ninguna autocorrelación en rezagos mayores (Sección 6.2).
2. En datos reales de transacciones de **IBM** (NYSE, 1990–1991), la serie direccional de cambios de precio muestra una ACF con un único pico significativo en el rezago 1, de valor $-0.389$, consistente con bid–ask bounce (Sección 6.5).
3. En el mismo tipo de datos, un modelo de nonsynchronous trading simplificado (Lo y MacKinlay, 1990) demuestra matemáticamente que, incluso si el proceso económico subyacente es **verdaderamente independiente**, el hecho de que un activo no cotice en todos los períodos introduce autocorrelación negativa espuria en el retorno observado (Sección 5.3).
4. En datos de transacciones de **IBM** en dos períodos distintos (1990–1991 con tick $\$1/8$, y diciembre de 1999 con tick $\$1/16$), la reducción del tick size coincidió con una reducción sustancial de la proporción de transacciones sin cambio de precio (Sección 7.2).
5. En la acción de **Apple** (1999–2007), un evento institucional de fecha conocida (la decimalización de enero de 2001) coincidió con una reducción estadísticamente significativa del rango diario de log-precio, medido con un modelo de intervención sobre un TWACD (Sección 15.5).
6. Los modelos de duración (ACD) y sus extensiones no lineales (threshold ACD) muestran, sobre datos de transacciones de IBM, evidencia de **persistencia** en la intensidad de negociación y de **regímenes distintos** de dinámica de duración según el nivel de actividad reciente (Secciones 12.6, 13.3).

### 19.2 [B] Lecturas propias para futuros — todas como hipótesis, no como hallazgos

- La posibilidad de que parte de cualquier autocorrelación de muy corto plazo observada en datos intradiarios de MNQ (a alta frecuencia, o incluso en barras de 1 minuto) se deba, parcial o totalmente, a efectos de microestructura (bid–ask bounce residual en el Close, nonsynchronous trading entre instrumentos si se usa un contexto multivariado) es una **hipótesis empírica abierta**, no una certeza que este capítulo demuestre para MNQ ni para ningún otro contrato de futuros.
- El patrón "la dinámica de duración/actividad cambia según el nivel reciente de actividad" (documentado por Tsay para duraciones de IBM y para el rango de Apple) sugiere, como hipótesis, que un fenómeno análogo —regímenes de actividad con dinámica distinta— podría existir también en la actividad de MNQ, sin que esto implique que deba modelarse con ACD ni con ningún modelo concreto.
- La reducción de tick size documentada por Tsay para IBM y Apple sugiere, como hipótesis general, que cambios en la infraestructura de un mercado (tick size, sesión de negociación, especificación de contrato) pueden alterar propiedades estadísticas de los datos sin que eso refleje ningún cambio en el comportamiento económico fundamental del activo — una advertencia general a tener presente si en algún momento se estudia la historia completa de MNQ, que podría incluir cambios de este tipo (por ejemplo, cambios de tick size o de horario de sesión a lo largo del tiempo), sin que este informe identifique ninguno concreto.
- La observación de que, transacción a transacción, existe evidencia clara de reversión de precio de muy corto plazo en IBM (Secciones 6.5, 10.4, 11.5) es un resultado documentado por Tsay para **acciones NYSE en los años 90**, con un mecanismo institucional de creadores de mercado específico de esa época y ese mercado. **No se traslada aquí ninguna expectativa concreta sobre reversión de corto plazo en MNQ**, que opera con un mecanismo de mercado electrónico distinto (libro de órdenes, sin market maker formal con las mismas obligaciones).

### 19.3 Por qué NO se afirma nada de esto sobre MNQ

**[A]** En ningún momento del Capítulo 5 Tsay analiza un contrato de futuros, un instrumento electrónico moderno, ni datos posteriores a 2008. Todos los ejemplos empíricos del capítulo son acciones individuales (IBM, Boeing, Apple) negociadas en la NYSE, en un rango de fechas que va de 1990 a 2007, bajo estructuras institucionales (market makers, tick sizes específicos, TAQ/TORQ como fuente de datos) que **no corresponden** al mecanismo de negociación electrónico de un futuro como MNQ en CME Globex. Cualquier lectura de este capítulo hacia futuros debe, por tanto, tratarse como una hipótesis a verificar empíricamente, nunca como un hecho ya demostrado.

---

## 20. Implicancias para Machine Learning — [B], por categorías

Todo lo que sigue es interpretación propia **[B]**, no una propuesta a adoptar, presentada por categorías tal como pide el enunciado. Ninguna decisión de diseño se toma en esta sección.

**Data representation.** El capítulo sugiere que, antes de fijar qué variable de precio usar como referencia, conviene preguntarse explícitamente qué mide esa variable (Last, un precio realmente transado, potencialmente afectado por bounce; frente a un eventual Mid, un promedio no ejecutable) y qué información pierde respecto de un dataset de transacciones o de quotes.

**Sampling frequency.** La discusión de discreción de precio (tick size, Sección 7.2) y de patrón diurno (Sección 7.3) sugiere que la frecuencia de muestreo interactúa con ambos fenómenos: a mayor frecuencia, el tick size se vuelve relativamente más importante respecto del movimiento típico por barra, y el patrón diurno se vuelve más pronunciado en relación con la escala de tiempo de la barra.

**Price representation.** Pregunta especialmente destacada por el enunciado: **¿un modelo aprende movimientos económicos genuinos, o aprende (parcialmente) la alternancia entre operaciones ejecutadas cerca del bid y cerca del ask?** Esta pregunta no puede responderse desde este capítulo para MNQ; requiere evidencia empírica directa sobre el dataset concreto.

**Feature engineering.** Pregunta especialmente destacada: **¿el volumen agregado por barra captura información suficiente sobre la actividad de mercado, o faltaría el conteo de trades o la duración entre ellos** (la distinción de la Sección 8.3), si estuvieran disponibles?

**Target construction.** La descomposición de Tsay en ocurrencia/dirección/magnitud (Secciones 9–11) es relevante como marco conceptual para pensar, en el futuro, qué tipo de variable objetivo tiene sentido para IRIS (¿predecir si habrá movimiento?, ¿en qué dirección?, ¿de qué magnitud?) — sin decidir aquí ninguna de esas alternativas.

**Label noise.** Pregunta especialmente destacada: **¿un target de magnitud muy pequeña (por ejemplo, el cambio de precio de una barra a la siguiente en un instrumento de tick relativamente grande) está dominado por el propio tick size y por el posible bounce residual del Close, más que por información económica genuina?**

**Model diagnostics.** El patrón repetido de Tsay a lo largo de todo el capítulo —ajustar un modelo, verificar residuos con Ljung–Box, aplicar tests de no linealidad, refinar— es un flujo de trabajo replicable como principio general, coherente con lo ya observado en los Capítulos 3 y 4.

**Cross-market features.** Pregunta especialmente destacada: **¿un aparente lead-lag entre dos mercados (por ejemplo, MNQ y otro instrumento relacionado) desaparece al sincronizar correctamente los timestamps de última actualización de cada uno?** (Sección 5.5 de este informe). No se responde aquí; se difiere formalmente al Capítulo 8 del libro.

**Intraday seasonality.** El patrón diurno documentado extensamente por Tsay para acciones NYSE (Sección 7.3) sugiere, como categoría general a explorar, que volumen, número de trades (si disponible), rango y volatilidad podrían tener un componente sistemático ligado a la hora del día en MNQ — sin asumir que la forma exacta (en U, u otra) sea la misma que la de una acción NYSE con horario limitado.

**Execution modeling.** La distinción entre "señal estadística" y "posibilidad económica" (Sección 6.7) es directamente relevante aquí: cualquier futuro modelo de ejecución debería, en principio, contemplar el spread y otros costos de transacción antes de considerar una señal como operable — sin que esto implique construir aquí ningún modelo de costos.

**Validation.** La lección metodológica más general de todo el capítulo —que gran parte de la estructura de alta frecuencia puede no ser información económica— sugiere que cualquier validación futura de IRIS debería, como mínimo, contemplar la posibilidad de que un patrón detectado sea un artefacto de microestructura o de construcción de la barra, y no asumir automáticamente que es señal económica solo porque es estadísticamente significativo.

**Regime/context modeling.** Los modelos de duración con umbral (Sección 13) muestran, para IBM, que la dinámica de una variable de actividad puede cambiar según el nivel reciente de esa misma variable — un patrón conceptualmente similar (aunque no idéntico) a los modelos TAR del Capítulo 4, ahora aplicado a intensidad de negociación en vez de a retornos.

**Probabilistic forecasting.** No desarrollado como tema central en este capítulo (a diferencia del Capítulo 4); el modelo ordered probit (Sección 10) sí produce, por construcción, una distribución de probabilidad sobre categorías de cambio de precio, lo cual es conceptualmente relevante para la pregunta de IRIS sobre "qué nivel de confianza tiene el modelo en una señal", sin que esto implique adoptar ordered probit como arquitectura.

**Hipótesis a comprobar.** Se desarrollan formalmente en la Sección 21 siguiente.

---

## 21. Hipótesis empíricas para backlog — NO ejecutar

Cada hipótesis se presenta con pregunta, datos necesarios, método, resultado que la apoyaría, resultado que la refutaría, y limitaciones. **Ninguna se ejecuta en este informe.**

### H5.1 Frecuencia

- **Pregunta:** ¿cómo cambian la ACF de corto plazo, la distribución de retornos y la predictibilidad direccional aparente al comparar barras de 1, 5 y 10 minutos (u otras frecuencias) del mismo instrumento?
- **Datos necesarios:** serie de barras de MNQ a varias frecuencias, del mismo período.
- **Método:** ACF, tests de no linealidad (Capítulo 4), y comparación de predictibilidad direccional simple entre frecuencias.
- **Resultado que la apoyaría:** cambios sistemáticos y consistentes en la estructura estadística al variar la frecuencia.
- **Resultado que la refutaría:** estructura estable entre frecuencias.
- **Limitaciones:** a mayor frecuencia, se mezclan efectos de microestructura (Secciones 5–7) con la señal económica; conviene evaluar ambos posibles orígenes en conjunto, no solo la frecuencia.

### H5.2 Tick discreteness

- **Pregunta:** ¿qué fracción representa un tick de MNQ respecto de la variación típica de precio por barra, a distintas frecuencias?
- **Datos necesarios:** especificación de tick size de MNQ; distribución de rangos (High−Low) o de retornos por barra a distintas frecuencias.
- **Método:** cálculo del cociente tick/variación típica (Sección 7.2), comparado entre frecuencias.
- **Resultado que la apoyaría / refutaría:** no aplica en sentido estricto — es una medición descriptiva, no una hipótesis binaria; el resultado informa cuán relevante es la discreción relativa según la frecuencia.
- **Limitaciones:** ninguna métrica de discreción relativa está adoptada de antemano; esta medición es exploratoria.

### H5.3 Zero/small movement

- **Pregunta:** ¿qué proporción de barras de MNQ, a distintas frecuencias, no cambia de precio (Close = Close anterior), cambia 1 tick, 2 ticks, etc.?
- **Datos necesarios:** serie de barras de MNQ a varias frecuencias.
- **Método:** histograma de cambios de Close en múltiplos de tick, análogo a la Tabla 5.1 de Tsay para IBM.
- **Resultado que la apoyaría:** concentración fuerte en 0/1 tick a alta frecuencia, decreciente a menor frecuencia — coherente con lo observado por Tsay a nivel transacción.
- **Resultado que la refutaría:** distribución sin concentración marcada, sugiriendo que el tick size no es una restricción relevante para MNQ a las frecuencias consideradas.
- **Limitaciones:** el resultado depende del tick size relativo del contrato en el período estudiado; no comparable directamente con los resultados de acciones NYSE de Tsay.

### H5.4 Intraday seasonality

- **Pregunta:** ¿cómo cambian volumen, rango, volatilidad y retornos de MNQ según la hora del día (o la sesión: apertura de EE.UU., overlap con Europa, horario asiático, etc.)?
- **Datos necesarios:** serie de barras de MNQ con timestamp, suficientemente larga para promediar por hora del día.
- **Método:** promedios y ACF por hora del día, análogo a la Figura 5.2 de Tsay.
- **Resultado que la apoyaría:** patrón sistemático y estable por hora del día.
- **Resultado que la refutaría:** ausencia de patrón sistemático, o patrón inestable entre subperíodos.
- **Limitaciones:** MNQ opera en un régimen de casi 24 horas, sin el mismo tipo de apertura/cierre fijo que una acción NYSE; no debe asumirse una forma en U idéntica a la de Tsay.

### H5.5 Last vs. alternative price representations

- **Pregunta:** si en algún momento se dispusiera de datos de Bid/Ask/Mid para MNQ, ¿cambiarían de forma relevante las propiedades estadísticas de los retornos calculados (ACF de corto plazo, distribución) respecto de usar solo Last?
- **Datos necesarios:** datos de bid/ask de MNQ, no disponibles actualmente en el proyecto.
- **Método:** comparación de ACF y distribución de retornos calculados sobre Last vs. Mid, para el mismo período.
- **Resultado que la apoyaría:** diferencias sistemáticas (por ejemplo, menor autocorrelación negativa de corto plazo en Mid que en Last).
- **Resultado que la refutaría:** resultados prácticamente idénticos entre ambas representaciones.
- **Limitaciones:** requiere datos que el proyecto no tiene actualmente; no se descargan aquí.

### H5.6 Bid–ask footprint

- **Pregunta:** ¿hay evidencia, en los datos de barras de MNQ ya disponibles, de autocorrelación de muy corto plazo (rezago 1) en los retornos de Close a Close, compatible con un residuo de bid–ask bounce (Sección 6.6)?
- **Datos necesarios:** serie de barras de MNQ a alta frecuencia (por ejemplo, 1 minuto).
- **Método:** ACF de rezago 1 sobre retornos de Close, ya cubierto metodológicamente por el Capítulo 2; la novedad aquí es la **interpretación** candidata (microestructura) frente a la interpretación económica.
- **Resultado que la apoyaría:** autocorrelación negativa de rezago 1, de magnitud decreciente al agregar a frecuencias menores (coherente con un origen de microestructura que se diluye con la agregación).
- **Resultado que la refutaría:** ausencia de autocorrelación negativa de rezago 1, o autocorrelación que no decrece con la frecuencia (lo cual sugeriría un origen distinto al bounce).
- **Limitaciones:** una autocorrelación negativa de rezago 1, por sí sola, **no distingue** entre bid–ask bounce, nonsynchronous trading, y una genuina reversión económica de corto plazo (Sección 22, afirmación 6–7); se requeriría evidencia adicional para separar estas explicaciones.

### H5.7 Aggregation

- **Pregunta:** ¿qué estructura estadística (ACF, no linealidad detectada por los tests del Capítulo 4) desaparece o se atenúa al pasar de una frecuencia más alta (por ejemplo, 1 minuto) a una más baja (5 o 10 minutos) en MNQ?
- **Datos necesarios:** serie de barras de MNQ a varias frecuencias.
- **Método:** repetir el mismo conjunto de tests a cada frecuencia y comparar.
- **Resultado que la apoyaría:** atenuación sistemática de la estructura de corto plazo al agregar — coherente con un origen de microestructura.
- **Resultado que la refutaría:** estructura estable o creciente al agregar — menos coherente con un origen puramente de microestructura.
- **Limitaciones:** agregar reduce también el tamaño efectivo de muestra por unidad de tiempo calendario, lo cual puede afectar la potencia de los tests independientemente del fenómeno subyacente.

### H5.8 Synchronization

- **Pregunta:** en un eventual contexto multivariado (MNQ junto con otros instrumentos), ¿un lead-lag aparente entre instrumentos sobrevive a una sincronización rigurosa de timestamps de última actualización?
- **Datos necesarios:** datos de al menos dos instrumentos con timestamps de alta resolución.
- **Método:** comparar la correlación cruzada retrasada antes y después de un procedimiento de sincronización explícito.
- **Resultado que la apoyaría (de un lead-lag genuino):** la correlación cruzada retrasada persiste tras la sincronización.
- **Resultado que la refutaría:** la correlación cruzada retrasada desaparece o se reduce sustancialmente tras sincronizar — sugiriendo que era, al menos en parte, un artefacto de nonsynchronous trading (Sección 5.5).
- **Limitaciones:** el diseño formal de este tipo de análisis (VAR, lead-lag) corresponde al Capítulo 8 del libro, no cubierto todavía en este estudio.

### H5.9 Volume vs. activity

- **Pregunta:** si se dispusiera de un conteo de número de trades por barra para MNQ, ¿aportaría información adicional respecto del volumen agregado ya disponible (Sección 8.3)?
- **Datos necesarios:** dataset de MNQ con columna de número de trades por barra, no confirmada como disponible actualmente en el proyecto.
- **Método:** comparación de la relación de cada variable (volumen, número de trades) con alguna medida de volatilidad o de movimiento de precio subsecuente.
- **Resultado que la apoyaría:** el número de trades aporta información no capturada por el volumen solo.
- **Resultado que la refutaría:** el número de trades está tan correlacionado con el volumen que no aporta información adicional relevante.
- **Limitaciones:** depende de la disponibilidad de esta variable en el dataset concreto de IRIS, no verificada en este informe.

### H5.10 Structural change

- **Pregunta:** ¿existen, en la historia de MNQ, cambios contractuales, de horario de sesión, de tick size o de infraestructura de negociación conocidos, que pudieran alterar las propiedades estadísticas de los datos de forma análoga al caso de decimalización de Apple (Sección 15.4–15.7)?
- **Datos necesarios:** documentación institucional/histórica de CME sobre el contrato MNQ (especificaciones, cambios de tick size, cambios de horario).
- **Método:** intervention analysis (Sección 15.4), análogo al de Tsay, aplicado a la fecha de cualquier cambio institucional identificado.
- **Resultado que la apoyaría:** cambio estadísticamente significativo y coherente con la dirección esperada del cambio institucional.
- **Resultado que la refutaría:** ausencia de cambio detectable en torno a la fecha del evento institucional.
- **Limitaciones:** requiere primero identificar si existió algún evento de este tipo en la historia de MNQ — no investigado en esta tarea.

### H5.11 Executability

- **Pregunta:** ¿la magnitud de cualquier señal estadística detectada en datos de MNQ supera el spread y otros costos operativos plausibles (Sección 6.7)?
- **Datos necesarios:** estimación de spread y costos de transacción típicos para MNQ (no cubierta por este capítulo ni por este informe).
- **Método:** comparación de la magnitud esperada de cualquier señal candidata contra una estimación de costos — no un backtest completo, solo un chequeo de plausibilidad económica mínima.
- **Resultado que la apoyaría:** la magnitud esperada de la señal supera, con margen razonable, los costos estimados.
- **Resultado que la refutaría:** la magnitud esperada de la señal es comparable o menor a los costos estimados.
- **Limitaciones:** requiere datos de costos de transacción reales de MNQ, no disponibles en el contexto de este estudio; no se estima ningún costo concreto en este informe.

---

## 22. Auditoría de las 30 afirmaciones

| # | Afirmación | Veredicto | Por qué |
|---|---|---|---|
| 1 | "Datos de 1 minuto son equivalentes a transaction data." | **INCORRECTA** | Es exactamente la distinción central de la Sección 3: una barra agrega muchas transacciones en cinco números, perdiendo orden interno, duración entre trades, número de trades y bid/ask vigente. |
| 2 | "Más frecuencia siempre significa más información útil." | **INCORRECTA** | A mayor frecuencia, aumenta el peso relativo del tick size (Sección 7.2) y de posibles efectos de microestructura (bounce, nonsynchronous trading); "más información" no es lo mismo que "más señal económica útil". |
| 3 | "Last representa perfectamente el valor verdadero del mercado." | **INCORRECTA** | Last es el precio de la última transacción real, que puede haberse ejecutado cerca del bid o del ask (bounce); no hay garantía de que coincida con ningún "valor verdadero" (Secciones 4, 6). |
| 4 | "Mid siempre es mejor que Last." | **REQUIERE CONDICIONES** | Mid puede tener menos ruido de bounce, pero no es un precio necesariamente ejecutable (Sección 4.8); "mejor" depende del uso (modelar vs. simular ejecución) — no se decide aquí. |
| 5 | "Mid es siempre un precio ejecutable." | **INCORRECTA** | Mid es un promedio calculado entre dos ofertas (bid y ask); nada garantiza que exista una contraparte dispuesta a operar exactamente a ese precio (Sección 4.8). |
| 6 | "Bid–ask bounce puede producir autocorrelación." | **CORRECTA** | Es exactamente el resultado matemático de Roll (1984) que reproduce Tsay: bounce produce $\rho_1=-0.5$ bajo los supuestos del modelo, y $\rho_j=0$ para $j>1$ (Sección 6.2). |
| 7 | "Autocorrelación negativa a lag 1 demuestra mean reversion explotable." | **INCORRECTA** | Es precisamente lo que el modelo de Roll refuta: una autocorrelación negativa de rezago 1 puede producirse íntegramente por bounce, sin ningún valor "revirtiendo" económicamente y sin ninguna oportunidad de arbitraje (Sección 6.3). |
| 8 | "Si hay bid–ask bounce en ticks, necesariamente dominará barras de 1 minuto." | **INCORRECTA** | Es justamente la Sección 6.6: se trata de una PREGUNTA EMPÍRICA, no resuelta por este capítulo; depende de número de trades por barra, liquidez, ancho relativo del spread, y otros factores — no hay ninguna garantía universal de dominancia. |
| 9 | "Nonsynchronous trading solo importa en acciones ilíquidas." | **REQUIERE CONDICIONES** | El modelo de Lo–MacKinlay (Sección 5.3) muestra que el efecto depende de la probabilidad de no negociar ($\pi$); es más pronunciado cuanto mayor $\pi$, lo cual típicamente se asocia a menor liquidez, pero el capítulo no restringe el fenómeno exclusivamente a "acciones", ni excluye que pueda aparecer en cualquier activo con $\pi>0$, incluidos horarios de baja actividad de instrumentos por lo demás líquidos. |
| 10 | "Dos barras con el mismo timestamp representan exactamente el mismo instante económico." | **INCORRECTA** | Es el punto central de la Sección 5.5 (sincronización multivariada): el timestamp de una barra no garantiza que el último precio de cada instrumento incorpore información hasta exactamente el mismo microinstante. |
| 11 | "Un lead-lag entre dos activos demuestra que uno predice al otro." | **INCORRECTA** | El propio ejemplo de Lo–MacKinlay (Sección 5.3) muestra que un lead-lag aparente puede surgir entre activos genuinamente independientes, por la sola diferencia de frecuencia de negociación — sin relación predictiva económica real. |
| 12 | "Tick size es irrelevante una vez que calculamos retornos." | **INCORRECTA** | El tick size sigue restringiendo los valores posibles de precio (y por lo tanto de retorno) después de calcular retornos; la discreción no desaparece por transformar el precio en retorno (Sección 7.2). |
| 13 | "El precio en alta frecuencia es esencialmente continuo." | **INCORRECTA** | Es exactamente lo opuesto a lo que documenta Tsay en la Sección 5.3: a mayor frecuencia (transacción a transacción), el precio es marcadamente discreto, con una fracción grande de "sin cambio" (Sección 7.2). |
| 14 | "Volume y número de trades son lo mismo." | **INCORRECTA** | Es la distinción explícita de la Sección 8.3: 100 trades de 1 contrato y 1 trade de 100 contratos dan el mismo volumen total pero estructuras de actividad completamente distintas. |
| 15 | "Una barra de 1 minuto conserva la duración entre trades." | **INCORRECTA** | Es exactamente lo contrario de lo establecido en la Sección 12.2: la separación entre barras es siempre constante (60 segundos); la duración irregular entre transacciones individuales se pierde por completo en la agregación. |
| 16 | "Un ACD puede aplicarse directamente a timestamps de barras de un minuto." | **INCORRECTA** | Los timestamps de barras de 1 minuto son, por construcción, siempre equiespaciados; no hay ninguna variación de duración que un ACD pueda modelar sobre ellos (Sección 12.2). |
| 17 | "Duración larga demuestra que no llegó nueva información." | **REQUIERE CONDICIONES** | Es la interpretación motivacional que da Tsay [A] (Sección 12.1), pero convertirla en una ley causal universal es una afirmación más fuerte que el capítulo no sostiene [B]; otros factores (horario, mecanismo de mercado) también pueden producir duraciones largas sin relación directa con ausencia de noticias. |
| 18 | "Patrones intradía implican regímenes latentes." | **INCORRECTA** | La Sección 7.3 distingue explícitamente entre un efecto determinístico ligado al reloj (seasonality) y un régimen estadístico latente (no observable, inferido de los datos); un patrón diurno es, por definición, del primer tipo, no del segundo. |
| 19 | "Si hay más operaciones al abrir y cerrar, cualquier mercado tendrá necesariamente forma de U." | **INCORRECTA** | La forma en U que documenta Tsay es específica de la NYSE en el período estudiado, con su horario y estructura institucional particulares; no se generaliza aquí a MNQ (mercado de casi 24 horas) ni a ningún otro mercado sin evidencia directa (Sección 7.3). |
| 20 | "Ordered probit demuestra que debemos usar clasificación." | **INCORRECTA** | El ordered probit es un modelo que Tsay usa para un problema específico (categorías ordenadas de cambio de precio transacción a transacción, con variables de transacción no disponibles en barras); no implica ninguna decisión sobre el target o la arquitectura de IRIS (Sección 10.6). |
| 21 | "Más categorías de movimiento producen un target mejor." | **INCORRECTA** | No hay ningún resultado en el capítulo que sostenga esto; el número de categorías es una decisión de diseño con sus propios costos (más categorías, menos observaciones por categoría) — no evaluada aquí. |
| 22 | "Una relación significativa en transaction data sobrevivirá a la agregación." | **INCORRECTA** | Es exactamente lo contrario de lo que sugiere la lógica de la Sección 3 y la tabla de la Sección 16: buena parte de la estructura documentada por Tsay depende de información (duración exacta, bid/ask, número de trades) que la agregación destruye; no hay ninguna garantía de que sobreviva. |
| 23 | "Si un modelo predice Last correctamente, puede ejecutarse a ese mismo precio." | **INCORRECTA** | Es exactamente el punto de la Sección 6.7 y de la advertencia final del capítulo: ejecutar una operación real cruza el spread y otros costos; predecir Last correctamente no equivale a poder operar a ese precio exacto. |
| 24 | "El spread solo importa para backtesting, no para ML." | **INCORRECTA** | El spread condiciona directamente qué parte de una señal estadística detectada por un modelo de ML es económicamente explotable (Sección 6.7, 20); ignorarlo durante el desarrollo del modelo no lo vuelve irrelevante, solo posterga el problema. |
| 25 | "Cambios de tick size pueden modificar propiedades estadísticas observadas." | **CORRECTA** | Es exactamente lo que documenta Tsay con los datos de IBM (1990–91 vs. 1999) y de Apple (decimalización de 2001, Sección 15.5): la proporción de "sin cambio" y el rango esperado condicional cambiaron de forma medible y significativa. |
| 26 | "Un cambio estructural del mercado puede confundirse con cambio en la dinámica del activo." | **CORRECTA** | Es la lectura explícita de la Sección 15.7: una propiedad observada puede cambiar porque cambió la infraestructura/reglas del mercado, no necesariamente porque cambió el comportamiento económico fundamental del activo. |
| 27 | "Los resultados de NYSE de los años 1990 se trasladan directamente a futuros electrónicos actuales." | **INCORRECTA** | Es la advertencia central de la Sección 19.3: Tsay no analiza ningún futuro ni ningún mercado electrónico moderno; la estructura institucional (market makers, tick sizes, mecanismo de negociación) es sustancialmente distinta. |
| 28 | "Si solo usamos OHLCV, la microestructura deja de importar." | **INCORRECTA** | La Sección 16 muestra que varios fenómenos de microestructura (bounce residual en el Close, discreción de tick, patrón diurno) pueden seguir dejando huella en OHLCV, aunque de forma parcial o de magnitud desconocida; "no observable directamente" no es lo mismo que "irrelevante". |
| 29 | "Una red neuronal puede aprender artefactos de microestructura." | **CORRECTA** | Es una consecuencia lógica directa de lo visto en el Capítulo 4 (un modelo flexible aprende cualquier patrón presente en los datos de entrenamiento, sin distinguir su origen) combinado con este capítulo (algunos patrones de los datos pueden originarse en el mecanismo de negociación, no en información económica). |
| 30 | "Si aprende esos artefactos fuera de muestra, necesariamente son rentables." | **INCORRECTA** | Es exactamente el guardrail central de todo el capítulo y de este informe: estructura estadística observada (incluso si es estable fuera de muestra) no equivale a información económica explotable, una vez descontados spread, slippage y demás costos operativos (Secciones 1, 6.7, 20). |

---

## 23. Preguntas abiertas

**Dentro del alcance de este capítulo, sin resolver:**

- ¿Qué proporción de las barras de MNQ, a la frecuencia usada en IRIS, no cambia de precio, cambia 1 tick, 2 ticks, etc.? (H5.3)
- ¿Hay evidencia, en los datos ya disponibles de MNQ, de autocorrelación de muy corto plazo compatible con un residuo de bid–ask bounce en el Close de barras consecutivas? (H5.6)
- ¿Cómo cambian volumen, rango, volatilidad y retornos de MNQ según la hora del día, y esa variación corresponde más a un efecto determinístico ligado al reloj o hay algo más parecido a un régimen latente superpuesto? (H5.4, Sección 7.3)
- ¿Qué estructura estadística de corto plazo se atenúa o desaparece al agregar de una frecuencia más alta a una más baja en MNQ? (H5.7)
- ¿Qué precio de referencia (Last, o eventualmente Mid si se dispusiera de datos de bid/ask) sería más apropiado para modelar el estado del mercado, y cuál para simular el costo de ejecución? (Sección 17, preguntas 9–10)
- ¿Existió en la historia de MNQ algún cambio institucional (tick size, horario, especificación de contrato) análogo a la decimalización de acciones de 2001, cuyo efecto estadístico pudiera medirse con intervention analysis? (H5.10)

**Explícitamente diferidas a capítulos posteriores del libro, sin resolver anticipadamente:**

- **Capítulo 6** — modelos de tiempo continuo y de difusión: relevante para pensar de forma más formal la distinción entre el "precio fundamental" teórico $P_t^*$ usado en el modelo de Roll (Sección 6.2) y el precio efectivamente observado.
- **Capítulo 8** — relaciones multivariadas, VAR, cointegración, lead-lag: necesario para tratar con rigor la pregunta de sincronización multivariada planteada en la Sección 5.5 de este informe, y la hipótesis H5.8.
- **Capítulo 11** — modelos de espacio de estados y filtro de Kalman: potencialmente relevante para pensar el "precio fundamental" no observable como un estado latente, en la misma línea conceptual que el modelo de espacio de estados no lineal visto en el Capítulo 4.
- **Capítulo 12** — Markov Chain Monte Carlo: necesario para profundizar en la estimación del modelo PCD (Sección 14) y del ordered probit, que Tsay menciona solo como posibilidad de estimación en este capítulo, sin desarrollarla en detalle.

---

## 24. Checklist de conocimientos adquiridos

Lo que este informe permite explicar, en lenguaje simple, sin necesidad de repetir una fórmula:

- [x] Qué es market microstructure y por qué el precio observado no es necesariamente una medición perfecta del valor del activo.
- [x] La diferencia entre Last (hecho consumado, precio de una transacción real), Bid/Ask (ofertas vigentes, no ejecutadas) y Mid (promedio calculado, no necesariamente ejecutable).
- [x] Qué es el spread y qué representa (el costo de la inmediatez, y la compensación de quien provee liquidez).
- [x] Qué es el bid–ask bounce: la alternancia entre operaciones ejecutadas cerca del bid y cerca del ask, que hace que el precio observado "rebote" aunque el valor del activo no se haya movido.
- [x] Cómo el bid–ask bounce puede producir autocorrelación negativa aparente en rezago 1 (demostrado matemáticamente por Tsay vía el modelo de Roll), sin que exista ninguna reversión económica genuina ni oportunidad de arbitraje.
- [x] Qué es nonsynchronous trading: que distintos activos (o el mismo activo en distintos momentos) no se negocian de manera sincrónica, por lo que tratar sus últimos precios como simultáneos es una simplificación.
- [x] Cómo el nonsynchronous trading puede producir lead-lag aparente entre activos genuinamente independientes, simplemente porque uno se negocia con más frecuencia que el otro.
- [x] Qué es el tick size: el incremento mínimo permitido entre dos precios distintos, y por qué hace que el precio sea un objeto discreto, no continuo.
- [x] Por qué el tick size importa proporcionalmente más a frecuencias más altas (donde los movimientos típicos por operación o por barra son más pequeños).
- [x] La diferencia entre **tick size**, **transaction/tick data** y una barra de 1 minuto: el tick size es el incremento mínimo permitido del precio; transaction/tick data registra eventos o transacciones individuales con alta resolución; una barra de 1 minuto agrega muchas de esas observaciones en Open, High, Low, Close y Volume.
- [x] Qué información se pierde al construir un OHLCV: orden interno detallado, duración exacta entre transacciones, número de transacciones, bid/ask vigente en cada instante, dirección de iniciación de cada trade.
- [x] Qué es un patrón diurno: un componente sistemático y repetitivo de la actividad de mercado, ligado a la hora del día, que debe distinguirse de un régimen estadístico latente.
- [x] Que Volume y número de trades son conceptos distintos: el mismo volumen total puede provenir de estructuras de actividad completamente diferentes (muchos trades chicos vs. pocos trades grandes).
- [x] Qué es una trade duration: el tiempo transcurrido entre una transacción y la siguiente.
- [x] Por qué una barra de tiempo fijo no conserva esa duration: la separación entre barras consecutivas es siempre constante por construcción, mientras que la duración real entre transacciones es irregular y se pierde en la agregación.
- [x] Qué intenta modelar ACD: la persistencia de la intensidad de negociación (duraciones cortas o largas tienden a agruparse), con una estructura matemática análoga a GARCH pero aplicada a duración en vez de a varianza.
- [x] Qué intenta modelar el ordered probit: la probabilidad de cada categoría (ordenada) de cambio de precio, a partir de una variable latente continua no observable comparada contra un conjunto de umbrales.
- [x] Qué intenta separar el modelo de descomposición (ADS/PCD): si hubo cambio de precio, en qué dirección, y de qué tamaño — como tres preguntas encadenadas, no como un único objeto.
- [x] Por qué una relación estadística de alta frecuencia puede no ser ejecutable: porque cruzar de comprador a vendedor tiene un costo real (spread, slippage), que puede superar la magnitud del movimiento predicho.
- [x] Por qué Last no es automáticamente "el precio correcto" para todo: puede contener ruido de bounce, y no refleja el estado completo del libro de órdenes en cada instante.
- [x] Por qué Mid tampoco es automáticamente la solución: es un promedio calculado, no necesariamente un precio al que se pueda ejecutar una operación real.
- [x] Por qué transaction data y OHLCV requieren preguntas distintas: gran parte de la maquinaria de este capítulo (ACD, PCD, ordered probit en su forma original) depende de información que la agregación en barras destruye por completo.
- [x] Qué partes del capítulo son directamente aplicables a barras (tick size, patrón diurno, volumen, rango diario, intervention analysis) y cuáles no lo son en absoluto sin adaptación (ACD, PCD, ordered probit transacción a transacción, en su forma original).

---

## 25. Conclusiones

Este capítulo introduce una pregunta más fundamental que las de los cuatro capítulos anteriores: antes de preguntarse qué dependencia lineal, qué estructura de volatilidad o qué no linealidad existe en una serie de retornos, hay que preguntarse si el propio dato observado —el precio que efectivamente se registra como "el precio del activo en ese instante"— es una medición limpia del mercado o si ya viene parcialmente moldeado por el mecanismo mismo de negociación.

**[A]** Tsay muestra, mediante dos modelos simplificados, mecanismos concretos por los cuales la microestructura puede generar dependencia estadística sin que ésta represente necesariamente la dinámica económica del activo. En el modelo básico de **bid–ask bounce** de Roll (1984), bajo sus supuestos específicos —entre ellos, spread constante, operaciones igualmente probables en bid/ask y ausencia de movimiento del precio fundamental—, la autocorrelación de rezago 1 de los cambios de precio observados es exactamente $-0.5$. Cuando se permite que el precio fundamental también cambie, la autocorrelación continúa siendo negativa bajo el modelo, pero su magnitud ya no tiene por qué ser $-0.5$. Por otra parte, el modelo de **nonsynchronous trading** de Lo y MacKinlay (1990) muestra que pueden aparecer autocorrelaciones y correlaciones cruzadas espurias incluso cuando el proceso económico subyacente es, por construcción, independiente.

El segundo hallazgo central, igual de importante para el Proyecto IRIS, es la **distinción radical entre datos de transacciones y barras de tiempo fijo**. Buena parte de la maquinaria estadística que Tsay desarrolla en este capítulo —el modelo ordered probit, el modelo de descomposición ADS, los modelos de duración ACD y sus extensiones no lineales, el modelo bivariado PCD— está construida sobre información (timestamps exactos de cada transacción, duración irregular entre ellas, bid/ask vigente, número de trades sin cambio de precio) que **la agregación en un dataset OHLCV de 1 minuto destruye por completo**. Esto no vuelve inútil al capítulo, pero sí exige mucha precisión sobre qué partes son directamente trasladables a barras (tick size, patrón diurno, rango diario, intervention analysis sobre cambios institucionales) y cuáles no lo son sin una adaptación conceptual explícita que este capítulo no proporciona (ACD, PCD, ordered probit en su forma original).

**Lo que sabemos, con el respaldo directo del capítulo:** existe un vocabulario preciso y matemáticamente fundamentado para describir cómo el mecanismo de negociación —no solo la economía subyacente— puede dejar una huella estadística en los datos de mercado; existe evidencia empírica documentada (en acciones NYSE de los años 90 y 2000) de que esa huella puede ser sustancial (autocorrelaciones de rezago 1 de magnitud considerable, atribuibles en gran parte a bounce); y existe un conjunto de herramientas (ordered probit, modelos de descomposición, ACD, intervention analysis) para estudiar formalmente estos fenómenos cuando se dispone de datos de transacciones o de quotes.

**Lo que NO sabemos todavía, y este capítulo no permite saber:** cuánta de la estructura de corto plazo que eventualmente se observe en los datos intradiarios de MNQ se debe a microestructura y cuánta a información económica genuina; si el bid–ask bounce documentado por Tsay a nivel transacción deja alguna huella medible en barras de 1 minuto de MNQ; si existió algún cambio institucional relevante en la historia de MNQ análogo a la decimalización estudiada para Apple; y, en términos más generales, si alguna de las relaciones estadísticas que puedan detectarse en los datos de IRIS sobreviviría, una vez descontados spread y costos operativos, como una oportunidad económicamente explotable. Ninguna de estas preguntas se responde en este informe ni requiere resolverse antes de continuar el estudio del libro.

La prioridad, tal como lo exige el objetivo de este estudio, sigue siendo comprender antes de decidir — y este capítulo agrega una capa adicional, quizás la más incómoda de todas, a esa prioridad: antes de intentar predecir el mercado, hay que entender qué parte de nuestros propios datos proviene del mercado, y qué parte proviene de cómo se negocia y se mide.

---

## 26. Registro de revisión crítica

| Afirmación [B] sensible | Riesgo de sobreinterpretación | Estado |
|---|---|---|
| Extrapolación de la reversión de precio de IBM (bid–ask bounce, 1990-91) hacia una expectativa sobre reversión de corto plazo en MNQ | Alto — es tentador pensar "entonces en futuros también hay reversión de corto plazo explotable" | PREGUNTA ABIERTA — se dejó explícitamente como hipótesis no trasladable en la Sección 19.2–19.3 |
| Analogía entre el mecanismo de market maker de NYSE (años 90) y el libro de órdenes electrónico de un futuro moderno | Alto — es fácil, en la práctica, asumir que "spread" y "bid–ask bounce" funcionan igual en ambos contextos institucionales sin verificarlo | MANTENER, con la advertencia explícita ya incluida en la Sección 6.1 |
| Cociente tick/price como forma de pensar la importancia relativa de la discreción | Medio — podría leerse como una métrica ya adoptada para el proyecto | MATIZAR — ya está marcado explícitamente como idea conceptual [B], sin adopción de ningún umbral concreto (Sección 7.2) |
| Factores hipotéticos que influirían en la persistencia del bid–ask bounce en barras de 1 minuto (Sección 6.6) | Medio — podría leerse como una lista de features ya validadas, cuando es una lista de hipótesis sin medir | MANTENER — ya está marcada explícitamente como PREGUNTA EMPÍRICA, sin ejecutar ningún experimento |
| Lectura de que "los cambios institucionales pueden confundirse con cambios de dinámica del activo" aplicada de forma genérica a futuros | Medio — podría interpretarse como una afirmación sobre algún cambio concreto en MNQ, cuando no se identificó ninguno | PREGUNTA ABIERTA — se mantiene como hipótesis general (H5.10), sin ningún evento concreto identificado |
| Tabla de "qué sobrevive a la agregación" (Sección 16) | Medio — varias filas usan "posiblemente" o "depende", lo cual podría simplificarse indebidamente en una lectura rápida | MANTENER, con la aclaración explícita de que ninguna fila proviene textualmente de Tsay y que el capítulo no analiza datos agregados |
| Analogía entre ordered probit / ADS y clasificación ordinal en ML | Bajo — es una observación terminológica razonable y ampliamente aceptada | MANTENER |
| Conexión entre intervention analysis y "structural break" con fecha conocida | Bajo — es una lectura metodológica estándar, coherente con la literatura general de series de tiempo | MANTENER |

---

## Informe de cierre

**Secciones estudiadas en profundidad** (con lectura directa del texto de Tsay, prioridad máxima según el enunciado): introducción del capítulo (definición de high-frequency data, motivación general); **5.1 Nonsynchronous Trading** (con la derivación completa del modelo de Lo–MacKinlay simplificado, Ecs. 5.1–5.8); **5.2 Bid–Ask Spread** (con la derivación completa del modelo de Roll, Ecs. 5.9–5.14, y su extensión con valor fundamental de camino aleatorio); **5.3 Empirical Characteristics of Transactions Data** (con los tres ejemplos empíricos completos: IBM 1990–91, IBM diciembre 1999, Boeing diciembre 2008).

**Secciones estudiadas seriamente pero de forma selectiva:** **5.4 Models for Price Changes**, **5.4.1 Ordered Probit** (con el Ejemplo 5.1 completo de Hauseman, Lo y MacKinlay, 1992, sobre IBM 1988), **5.4.2 Decomposition Model** (con el Ejemplo 5.2 completo sobre IBM 1990–91), y **5.8 Application** (con el caso completo de Apple 1999–2007, incluyendo el modelo threshold ACD y el intervention analysis de la decimalización).

**Secciones tratadas principalmente a nivel conceptual, con derivaciones matemáticas reducidas al mínimo por instrucción explícita del enunciado:** **5.5 Duration Models** (concepto de duración, ajuste de patrón diurno, estructura del ACD y su analogía con GARCH, con el Ejemplo 5.4 empírico reportado pero sin reproducir la derivación completa de momentos ni de la función de verosimilitud); **5.5.2 Simulation** y **5.5.3 Estimation** (resumidas en su lógica general, sin reproducir las fórmulas de log-verosimilitud en detalle); **5.6 Nonlinear Duration Models** (con el resultado empírico del threshold WACD sobre IBM, sin desarrollar la teoría general de tests de umbral más allá de lo ya cubierto en el Capítulo 4); **5.7 Bivariate Models for Price Change and Duration** (con la estructura completa de la descomposición PCD y el Ejemplo 5.5 sobre IBM, sin reproducir la estimación MCMC en detalle).

**Material omitido, y por qué:** el Apéndice A (revisión de distribuciones de probabilidad — exponencial, gamma, Weibull, gamma generalizada) se omitió por instrucción explícita, salvo las menciones mínimas imprescindibles para entender EACD/WACD/GACD dentro del cuerpo del informe; el Apéndice B (función de hazard) se redujo a la explicación conceptual imprescindible (qué es, por qué EACD tiene hazard constante y WACD monótono), sin desarrollar el aparato matemático completo; el Apéndice C (programas RATS) se omitió por instrucción explícita (código específico obsoleto); los ejercicios 5.1 a 5.10 del final del capítulo se omitieron por ser mecánicos y fuera del alcance de "comprensión y adquisición de conocimiento"; las derivaciones algebraicas completas de momentos del EACD(1,1) (Ecs. 5.36–5.39) se resumieron en su resultado final (media incondicional) sin reproducir el álgebra intermedia paso a paso.

**Archivo generado:** `Tsay_Cap5_analisis_futuros_ML.md`.

**Principales hallazgos [A]:** (1) el modelo básico de Roll (1984), que demuestra que bajo sus supuestos simplificados el bid–ask spread puede generar por sí mismo una autocorrelación negativa de rezago 1, igual a $-0.5$ en el caso básico sin movimiento del precio fundamental; con evidencia empírica en IBM de una ACF direccional de $-0.389$, consistente con bid–ask bounce; (2) el modelo de nonsynchronous trading de Lo–MacKinlay (1990) que demuestra que activos verdaderamente independientes pueden mostrar correlación cruzada y autocorrelación espurias por el solo hecho de negociarse con distinta frecuencia; (3) la evidencia empírica de discreción de precio y su relación con el tick size en tres episodios distintos (IBM 1990–91, IBM 1999, Boeing 2008), mostrando que reducir el tick size redujo la proporción de transacciones sin cambio de precio; (4) el catálogo de modelos de transacción (ordered probit, ADS, ACD, PCD) con sus resultados empíricos concretos sobre IBM, todos dependientes de información de transacciones no disponible en OHLCV agregado; (5) el caso completo de Apple, donde un evento institucional de fecha conocida (decimalización de 2001) produjo un efecto estadísticamente significativo y económicamente sensato sobre el rango diario de precios, vía intervention analysis.

**Principales interpretaciones [B]:** (1) el guardrail central de todo el informe —estructura estadística observada ≠ información económica explotable— aplicado sistemáticamente a cada fenómeno del capítulo; (2) la clasificación de cada concepto en DIRECTAMENTE RELEVANTE / PARCIALMENTE RELEVANTE / NO DIRECTAMENTE OBSERVABLE / PREGUNTA ABIERTA para barras de tiempo fijo, resumida en la tabla de la Sección 16; (3) la lista de diez preguntas abiertas sobre Last/Bid/Ask/Mid (Sección 17), explícitamente sin resolver; (4) la extensión conceptual a un contexto multivariado (sincronización entre instrumentos, Sección 5.5), diferida formalmente al Capítulo 8.

**Preguntas abiertas:** documentadas en detalle en la Sección 23, con separación explícita entre las que quedan abiertas dentro del propio Capítulo 5 y las que se difieren formalmente a los Capítulos 6, 8, 11 y 12.

**Hipótesis incorporadas al backlog:** once hipótesis (H5.1 a H5.11, Sección 21), cubriendo frecuencia de muestreo, discreción de tick, distribución de movimientos pequeños, estacionalidad intradía, representación de precio (Last vs. Mid), huella de bid–ask bounce en barras, efecto de la agregación, sincronización multivariada, volumen vs. número de trades, cambio estructural, y ejecutabilidad. Ninguna se ejecutó en este informe.

**Afirmaciones especialmente sensibles a sobreinterpretación**, señaladas también en la tabla de la Sección 26: la extrapolación de reversión de precio de IBM (acciones NYSE, 1990–91) hacia una expectativa de reversión de corto plazo en MNQ; la analogía entre el mecanismo de market maker de NYSE y el libro de órdenes electrónico de un futuro moderno; y, de forma más general, cualquier lectura que convierta una "huella posible" de microestructura en barras (Sección 16, columna intermedia) en una "huella confirmada" sin haber medido nada sobre los datos concretos de MNQ.

**Puntos donde transaction data y barras OHLCV no son directamente comparables:** duración entre transacciones (Sección 12.2); número de transacciones sin cambio de precio, $N_i$, del modelo PCD (Sección 14.5); bid/ask vigente en cada instante (Sección 4.7, 6.8); y, en consecuencia, los tres modelos centrales de estimación de este capítulo —ordered probit, ADS/decomposition, ACD/PCD— en su formulación original tal como los presenta Tsay.
