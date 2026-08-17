# Tsay, *Analysis of Financial Time Series* (3ª ed.) — Capítulo 1
## Estudio orientado a un sistema de Machine Learning para trading de futuros

**Fuente analizada:** Ruey S. Tsay, *Analysis of Financial Time Series, Third Edition*, Wiley (2010), Capítulo 1 "Financial Time Series and Their Characteristics", pp. 1–28 (secciones 1.1, 1.2, 1.2.1–1.2.5, 1.3, Apéndice R y Ejercicios).

**Convención de atribución usada en todo el informe:**

- **[A]** = lo que Tsay afirma, demuestra o muestra explícitamente en el capítulo.
- **[B]** = interpretación, extensión o adaptación propia a mercados de futuros y a Machine Learning. Tsay **no** habla de futuros, ni de Machine Learning, ni de costos de transacción, ni de roll de contratos, ni de validación out-of-sample en este capítulo.

Cuando una conclusión requiere capítulos posteriores para poder evaluarse, se marca como **PREGUNTA ABIERTA**.

---

# 1. Resumen ejecutivo

El Capítulo 1 es un capítulo de **definiciones y hechos estilizados**, no de modelos. Su valor para un proyecto de ML no está en las técnicas que enseña (casi ninguna) sino en que **fija correctamente el objeto de estudio**: qué variable se modela, qué distribución tiene, y qué significa exactamente decir que esa variable es predecible. Diez ideas concentran casi todo el valor:

1. **Se modelan retornos, no precios** [A]. Tsay cita a Campbell, Lo y MacKinlay (1997): el retorno es un resumen *completo y libre de escala* de la oportunidad de inversión, y las series de retornos tienen propiedades estadísticas más manejables que las de precios. Para ML esto es la primera decisión de representación: los precios no son comparables entre instrumentos ni en el tiempo; los retornos sí, aproximadamente. **[B]** En futuros esto se vuelve más agudo, no menos: precios nominales, tick size y point value distintos hacen que trabajar con precios crudos sea directamente inválido para modelos multi-instrumento.

2. **Log-retorno vs retorno simple es una decisión sobre qué agregación se quiere que sea exacta** [A]. Los log-retornos son **aditivos en el tiempo** ($r_t[k]=r_t+\dots+r_{t-k+1}$); los retornos simples son **aditivos en la cartera** ($R_{p,t}=\sum w_i R_{it}$). Ninguna representación tiene las dos propiedades. **[B]** Como un target de ML a horizonte $h$ es una agregación temporal y un backtest de cartera es una agregación cross-sectional, la elección no es cosmética: define qué se puede sumar sin error.

3. **La distribución conjunta se factoriza en distribuciones condicionales** [A]: $F(r_1,\dots,r_T)=F(r_1)\prod_{t=2}^{T}F(r_t\mid r_{t-1},\dots,r_1)$ (Ec. 1.15). Tsay dice explícitamente que **las distribuciones condicionales son más relevantes que las marginales**. **[B]** Esta ecuación *es* el problema de ML: aprender $F(y_{t+h}\mid X_t)$. Todo el proyecto vive dentro de esa factorización.

4. **La hipótesis de random walk tiene una definición estadística precisa y comprobable** [A]: una versión de la hipótesis de camino aleatorio es que $F(r_t\mid r_{t-1},\dots)=F(r_t)$, es decir, los retornos son temporalmente independientes y **por lo tanto no predecibles**. **[B]** Esto define la hipótesis nula del proyecto entero. Un modelo de ML no "predice el mercado": rechaza (o no) esta nula, con evidencia fuera de muestra y neta de costos.

5. **La normalidad es empíricamente falsa y Tsay lo demuestra con datos** [A]. Excess kurtosis alto en todas las series diarias (Tabla 1.2: IBM 9.92, S&P 22.81, Citigroup 55.25 en retornos simples diarios); Jarque–Bera para log-retornos diarios de IBM = 60,921.93 con p-valor 0.00; la densidad empírica es "más alta y más delgada, pero con soporte más ancho" que la normal (Fig. 1.4). **[B]** Toda métrica, intervalo de confianza, loss function o test que asuma normalidad hereda ese error.

6. **La media de un retorno diario es indistinguible de cero** [A]. Tsay muestra (Ejemplo 1.2) que el t-test de media cero sobre log-retornos diarios de IBM da t = 1.51, p = 0.1304, con casi 10,000 observaciones; y observa que "la media de una serie de retornos diarios es cercana a cero". **[B]** Este es probablemente el hecho más importante del capítulo para ML: la señal en el primer momento es minúscula frente al ruido. Con ~10⁴ observaciones no se puede ni siquiera detectar la media incondicional. Cualquier arquitectura que "encuentre" señal fuerte en retornos diarios debe ser sospechosa antes que celebrada.

7. **La varianza no es constante y se agrupa** [A]. En 1.3 Tsay señala que las variabilidades de los retornos cambian en el tiempo y **aparecen en clusters**, y define el proceso de volatilidad como la evolución de la *varianza condicional*. **[B]** Implica que el problema es no estacionario en el segundo momento: la distribución del target cambia con el tiempo. Esto afecta targets, normalización de features, esquema de validación y sizing.

8. **Los extremos son sustanciales y merecen tratamiento propio** [A]. Tsay muestra mínimos y máximos diarios de −22.96% a +13.16% (IBM) y hasta +57.82% (Citigroup), y dice que los extremos negativos importan para risk management y los positivos para posiciones cortas. **[B]** En un sistema de trading, una fracción diminuta de observaciones determina el P&L y el riesgo de ruina; los modelos entrenados con pérdidas que promedian errores tratan a esas observaciones como outliers a descartar, cuando son el fenómeno.

9. **La verosimilitud es el puente entre "asumir una distribución condicional" y "elegir una función de pérdida"** [A]. Tsay deriva la log-verosimilitud gaussiana con media $\mu_t$ y varianza $\sigma_t^2$ variables en el tiempo (Ec. 1.18) y señala que se construye igual con distribuciones condicionales no normales. **[B]** Minimizar MSE equivale a MLE gaussiano homocedástico; usar MAE equivale a Laplace; log-loss equivale a Bernoulli. Elegir la pérdida *es* asumir una distribución condicional, se sea consciente o no.

10. **El caso multivariado se plantea, pero se posterga** [A]. En 1.2.3 Tsay define el vector de retornos, su media y su matriz de covarianza, aclara que los estimadores muestrales son consistentes **siempre que la matriz de covarianza exista**, y remite a los Capítulos 8 y 10 para la dinámica de la esperanza y la covarianza condicionales. **[B]** La advertencia sobre existencia de momentos es crítica: con colas pesadas, las correlaciones entre futuros estimadas sobre ventanas cortas son inestables, y toda arquitectura cross-market descansa sobre ellas.

**Lo que el Capítulo 1 *no* provee** y que no debe darse por resuelto: estacionariedad y autocorrelación (Cap. 2), modelos de volatilidad (Cap. 3), no linealidad (Cap. 4), microestructura y alta frecuencia (Cap. 5), teoría de valores extremos (Cap. 7), lead-lag y cointegración (Cap. 8), covarianzas condicionales (Cap. 10). Y nunca provee: costos de transacción, roll de futuros, apalancamiento, ni metodología de validación.

---

# 2. Análisis sección por sección

## 2.1 Sección 1.1 — Asset Returns

### 2.1.1 Por qué retornos y no precios

**[A] Qué afirma Tsay.** La mayoría de los estudios financieros usan retornos en lugar de precios. Cita dos razones de Campbell, Lo y MacKinlay (1997): (i) para el inversor promedio, el retorno de un activo es un resumen *completo y libre de escala* (scale-free) de la oportunidad de inversión; (ii) las series de retornos son más fáciles de manejar porque tienen propiedades estadísticas más atractivas. Tsay no desarrolla cuáles son esas propiedades en 1.1; lo hace implícitamente en 1.2.5 y explícitamente recién en el Capítulo 2.

**[A] Significado estadístico.** "Libre de escala" significa invariante ante el nivel de precio: duplicar $P$ no cambia $R$. Es una transformación que elimina la dependencia del nivel nominal.

**[B] Interpretación estadística ampliada.** La razón profunda, que Tsay deja para el Cap. 2, es que la serie de precios típicamente **no es estacionaria** (tiene raíz unitaria) mientras que la serie de retornos se comporta mucho más cerca de estacionaria en media. Un precio tiene media y varianza que dependen de $t$; un retorno, aproximadamente no. **PREGUNTA ABIERTA:** la estacionariedad no se define ni se testea en el Capítulo 1, así que en esta etapa esto es una expectativa, no un resultado.

**[B] Traslado a futuros.** El argumento se refuerza y se complica a la vez:

- *Se refuerza:* un contrato de NQ cotizando en 18,000 y uno de GC en 2,400 no son comparables en unidades de precio. Ni siquiera la misma serie es comparable consigo misma a 10 años de distancia. Los retornos hacen comparables ambas cosas, aproximadamente.
- *Se complica:* en futuros **no existe un capital invertido natural**. El retorno $P_t/P_{t-1}-1$ calculado sobre el precio del futuro no es el retorno sobre el capital del trader, que depende del margen. El retorno sobre precio del futuro debe entenderse como un **retorno normalizado del subyacente sintético**, no como el rendimiento de la posición. Tsay no discute esto en absoluto.
- *Efecto adicional:* el precio de un futuro incorpora cost of carry y converge al spot en el vencimiento; su "drift" no es el del subyacente. **PREGUNTA ABIERTA** para el Cap. 6 (modelos de tiempo continuo) y para literatura específica de futuros.

**[B] Implicancia para ML.** La primera transformación del pipeline (precio → retorno) no es preprocesamiento trivial: es la decisión que hace que un modelo entrenado en un régimen de precios pueda generalizar a otro. Un modelo alimentado con precios crudos aprende niveles, y los niveles **no se repiten**: el conjunto de test está literalmente fuera del rango de valores del conjunto de entrenamiento (extrapolación pura). Los modelos basados en árboles (que particionan por umbrales absolutos) fallan de forma catastrófica y silenciosa en este escenario.

**[B] Qué comprobar empíricamente.** Verificar en la serie objetivo que: media y varianza de los precios en ventanas sucesivas divergen sistemáticamente; que las mismas estadísticas sobre retornos son mucho más estables; y que ningún feature usado retiene el nivel de precio salvo por diseño explícito y justificado.

### 2.1.2 Retorno simple, multiperiodo y anualizado

**[A] Qué afirma Tsay.**
- Gross return simple: $1+R_t = P_t/P_{t-1}$; net return: $R_t = (P_t-P_{t-1})/P_{t-1}$ (Ecs. 1.1–1.2).
- Multiperiodo: $1+R_t[k]=\prod_{j=0}^{k-1}(1+R_{t-j})$, llamado *compound return*. El retorno neto de $k$ períodos es $(P_t-P_{t-k})/P_{t-k}$.
- Anualizado: media **geométrica** de los gross returns, $\left[\prod(1+R_{t-j})\right]^{1/k}-1$, que puede calcularse como $\exp\left(\frac1k\sum\ln(1+R_{t-j})\right)-1$.
- Aproximación de primer orden (Taylor): Anualizado $\approx \frac1k\sum R_{t-j}$ (Ec. 1.3), y advierte explícitamente que **la precisión de esta aproximación puede no ser suficiente en algunas aplicaciones**.
- El intervalo temporal importa siempre: si no se especifica, se asume anual.

**[A] Significado estadístico.** La agregación temporal de retornos simples es **multiplicativa**. Multiplicar variables aleatorias no preserva familias de distribución (salvo la lognormal) y hace que la distribución del retorno de $k$ períodos sea difícil de derivar de la de un período.

**[B] Por qué importa para ML.** Un target a horizonte $h$ construido con retornos simples es un producto de $h$ factores; su distribución es más asimétrica que la de sus componentes y su varianza no escala de forma simple. Si además se quiere comparar targets a 1, 5 y 20 barras, con retornos simples los tres viven en escalas distintas de manera no lineal.

**[B] Traslado a futuros.** La advertencia de Tsay sobre la aproximación de Taylor es más importante de lo que parece: es válida cuando los retornos son pequeños. En futuros de índices intradía, los retornos por barra son minúsculos y la aproximación es excelente; en commodities energéticos con gaps, o en una barra que contiene un dato macro, no lo es. **La calidad de todas las aproximaciones "log ≈ simple" depende del régimen de volatilidad**, es decir, deja de ser homogénea a lo largo de la muestra.

**[B] Qué comprobar.** Medir la distribución de $|r_t - R_t|$ en la serie objetivo, en total y condicionada a deciles de volatilidad. Si el error de aproximación se concentra en los días de mayor volatilidad, entonces la elección de representación interactúa con el régimen, que es exactamente el subconjunto de datos que más importa.

### 2.1.3 Composición continua y log-retornos

**[A] Qué afirma Tsay.**
- Muestra el efecto de composición (Tabla 1.1): $1(1+0.1/m)^m \to e^{0.1}=1.10517$ cuando $m\to\infty$. Valor futuro $A=C\exp(rn)$ (Ec. 1.4) y valor presente $C=A\exp(-rn)$ (Ec. 1.5).
- Define el **continuously compounded return** o **log return**: $r_t=\ln(1+R_t)=\ln(P_t/P_{t-1})=p_t-p_{t-1}$ con $p_t=\ln P_t$ (Ec. 1.6).
- Enuncia **dos ventajas** del log-retorno sobre el retorno simple: (i) el retorno multiperiodo es la **suma** de los retornos de un período, $r_t[k]=r_t+r_{t-1}+\dots+r_{t-k+1}$; (ii) **las propiedades estadísticas de los log-retornos son más tratables**.
- Relaciones exactas: $r_t=\ln(1+R_t)$, $R_t=e^{r_t}-1$; y en porcentajes $r_t=100\ln(1+R_t/100)$, $R_t=100(e^{r_t/100}-1)$.
- Ejemplo 1.1: un log-retorno mensual de 4.46% corresponde a un simple de 4.56%; tres log-retornos mensuales de 4.46%, −7.34% y 10.77% dan un log-retorno trimestral de 7.89% (suma directa).

**[A] Contrapartida — Portfolio Return.** El retorno simple de una cartera **es** el promedio ponderado de los retornos simples de los activos: $R_{p,t}=\sum_{i=1}^N w_i R_{it}$. Tsay dice explícitamente que **los log-retornos no tienen esta propiedad conveniente**, y que sólo si los $R_{it}$ son pequeños vale la aproximación $r_{p,t}\approx\sum w_i r_{it}$.

**[A] Significado estadístico.** Esto establece una **dualidad exacta y no negociable**:

| Propiedad | Retorno simple $R_t$ | Log-retorno $r_t$ |
|---|---|---|
| Agregación **temporal** | multiplicativa (producto) | **aditiva (suma)** |
| Agregación **cross-sectional** (cartera) | **aditiva (ponderada)** | sólo aproximada |
| Cota inferior | −1 (bien definida) | ninguna ($-\infty$) |
| Distribución bajo iid normal | multiperiodo **no** normal | multiperiodo **sí** normal |

**[B] Implicancia para ML — targets.** La aditividad temporal del log-retorno es una propiedad de diseño de targets de primer orden:
- Un target a $h$ barras es simplemente $\sum_{j=1}^{h} r_{t+j}$, y su varianza, bajo independencia, escala con $h$ (regla $\sqrt{h}$). Eso permite comparar y normalizar targets de horizontes distintos de forma coherente.
- Permite descomponer un target de horizonte largo en contribuciones por barra, útil para diagnosticar *dónde* dentro del horizonte está la señal.
- Con retornos simples nada de esto es limpio.

**[B] Implicancia para ML — evaluación.** La contrapartida de cartera importa en el momento del backtest: al agregar P&L de varios contratos simultáneos, sumar log-retornos ponderados introduce un sesgo. La regla práctica: **entrenar y modelar en log-retornos; contabilizar P&L en retornos simples o directamente en unidades monetarias** (puntos × point value). Esto no está en Tsay, pero se deduce directamente de su tabla de propiedades.

**[B] Traslado a futuros — la trampa del contrato continuo.** Éste es el punto donde el marco de Tsay se rompe si se aplica ingenuamente, y no está en el libro:
- Los futuros tienen vencimiento. Una serie histórica larga es una serie **construida** mediante empalme de contratos (roll).
- **Serie no ajustada (precios reales por contrato).** Precios verdaderos y negociables, sin ningún riesgo de leak. Pero la serie es discontinua en el roll: el "retorno" calculado entre dos contratos distintos no es un retorno de mercado.
- **Retornos intra-contrato encadenados.** Equivalen a la serie de retornos de una estrategia de roll. Bien definidos y libres de leak **siempre que la regla de roll sea causal** (calendario fijo, o volumen/open interest observados hasta $t$).
- **Ajuste por ratio (proporcional).** Aplica un **factor multiplicativo constante** a todo el segmento previo a cada roll. Consecuencia clave: **los retornos calculados sobre la serie ratio-ajustada son idénticos a los intra-contrato encadenados** (salvo la barra del roll). No hay distorsión.
- **Ajuste por diferencia (panama/aditivo).** Preserva **exactamente las diferencias en puntos** — que es justo lo que se necesita para contabilizar P&L en puntos × point value — pero **distorsiona los retornos**: $(P_t+c)/(P_{t-1}+c)-1$ no es el retorno verdadero, y si los precios desplazados llegan a cero o a valores negativos, $\ln P_t$ no existe.
- **Criterio corregido (no normativo):** no se trata de que los retornos "deban" calcularse intra-contrato, sino de **elegir la transformación según qué cantidad debe preservarse exactamente**. Ratio preserva retornos; aditivo preserva diferencias en puntos; **ninguna preserva ambas**. La barra del roll debe tratarse explícitamente en cualquier caso, porque un salto de roll no es un retorno de mercado.
- **Dónde hay riesgo real de look-ahead, con precisión.** El factor de ajuste de un segmento depende de rolls posteriores, pero es un factor **constante** aplicado uniformemente a ese segmento. Por lo tanto:
  - Features **invariantes a escala** (retornos, cocientes, distancia porcentual a una media móvil, rankings) son **insensibles al factor: no hay leak**.
  - Features **dependientes del nivel** (umbrales de precio absolutos, distancias en puntos, comparaciones de nivel a través de la frontera de un roll) **sí incorporan información futura**.
  - **Test operativo de leak:** ¿el valor del feature cambiaría si cambiara el factor de ajuste? Si la respuesta es no, la transformación preserva correctamente la información.
  - Riesgo aparte, independiente del método de ajuste: si la **regla de roll** se define ex post (elegir retrospectivamente el contrato más líquido de cada fecha), eso sí es no causal.
- **No se adopta todavía una metodología definitiva**: la elección depende de qué features y qué contabilidad de P&L termine usando el sistema.

**[B] Traslado a futuros — tick size y point value.** Tsay menciona el tick size sólo en 1.2.2 y como problema de discreción para alta frecuencia. En futuros la implicancia es doble:
- El **retorno mínimo no nulo** de un instrumento es $\approx \text{tick}/P$. En ES (~5,000 puntos, tick 0.25) eso es 5·10⁻⁵; en un contrato de precio bajo puede ser un orden de magnitud mayor. Los retornos no son continuos, y para barras cortas la grilla discreta domina.
- El **point value** (multiplicador) no afecta al retorno pero sí al P&L. Micro y E-mini comparten subyacente y tick expresado en puntos de índice, y difieren en granularidad de posición y en valor por punto. Por eso, **los retornos son en principio comparables entre Micro y E-mini; el P&L y el riesgo no lo son**.
- **Matiz — la intercambiabilidad para entrenamiento es una hipótesis empírica, no un hecho.** El vínculo entre ambos es arbitraje, no identidad. Diferencias que pueden importar:
  - A **nivel de tick** los precios divergen momentáneamente: timestamps de operaciones distintos, profundidad de libro muy distinta y dinámica de bid–ask bounce propia.
  - Volumen y open interest difieren en órdenes de magnitud, de modo que **cualquier feature de volumen, order flow o microestructura no es intercambiable**, aunque el retorno lo sea.
  - La **historia disponible** difiere: los Micro son mucho más recientes, y entrenar con ellos trunca la muestra — costoso dada la relación señal/ruido documentada en 1.2.5(b).
  - Spread efectivo y slippage difieren, lo que afecta al backtest aun tras normalizar el point value.
  - **Formulación como hipótesis a testear:** *a frecuencias de barra media y baja, y usando exclusivamente features derivados de precio, las series de retorno de Micro y E-mini son estadísticamente indistinguibles.* Plausible, pero debe verificarse; a alta frecuencia o con features de volumen/microestructura, es probablemente falsa.

**[B] Qué comprobar antes de usar esto.**
1. Reconstruir la serie de retornos intra-contrato y compararla con la que produce la serie back-adjusted; cuantificar la discrepancia acumulada.
2. Verificar que ningún log-precio de la serie ajustada sea indefinido o negativo.
3. Medir $\text{tick}/P_t$ por instrumento y por época, y compararlo con la desviación estándar del retorno por barra: si el cociente es grande, la serie está dominada por discreción.
4. Comprobar empíricamente que $\text{Var}(r_t[h])\approx h\,\text{Var}(r_t)$ o cuantificar la desviación (esto anticipa el Cap. 2: desviaciones sistemáticas señalan autocorrelación o memoria).

### 2.1.4 Dividendos y excess returns

**[A] Qué afirma Tsay.** Con dividendos: $R_t=(P_t+D_t)/P_{t-1}-1$ y $r_t=\ln(P_t+D_t)-\ln P_{t-1}$. El **excess return** es la diferencia entre el retorno del activo y el de un activo de referencia, típicamente libre de riesgo (T-bill de corto plazo): $Z_t=R_t-R_{0t}$, $z_t=r_t-r_{0t}$ (Ec. 1.7). En la literatura financiera, el excess return se interpreta como el payoff de una **cartera de arbitraje** que va larga en el activo y corta en el activo de referencia, sin inversión neta inicial. Añade un *Remark* explicando qué son las posiciones larga y corta y las obligaciones del vendedor en corto (devolver acciones, pagar dividendos al prestamista).

**[B] Relevancia para futuros.** Aquí hay una conexión que Tsay no hace pero que es estructural:
- Los futuros **no pagan dividendos**; el precio del futuro ya descuenta los dividendos esperados del subyacente vía cost of carry. Es decir, la corrección de dividendos de Tsay ya está incorporada en el precio.
- La estructura "posición sin inversión neta inicial, financiada al tipo libre de riesgo" **guarda una analogía** con la estructura económica de un futuro. **[B] Es una analogía, no una identidad**, y del Capítulo 1 no se deriva: Tsay define excess return sólo para un activo con precio y capital invertido, y no menciona futuros en ningún momento.
- **Qué puede afirmarse con rigor** (y requiere teoría de pricing, no el Cap. 1): bajo cost of carry, $F_t=S_t e^{(r-q)(T-t)}$, y la variación porcentual del precio del futuro **no es idénticamente** el excess return del spot — depende del efecto de vencimiento y de la convergencia. El enunciado defendible es más estrecho: el retorno de una posición **totalmente colateralizada** ≈ retorno del colateral (≈ $r$) + variación del precio del futuro sobre el nocional; de ahí que la variación porcentual del precio del futuro **aproxime** el excess return de esa posición, bajo convenciones específicas de colateralización en efectivo remunerada a $r$.
- **Qué impide cerrarlo:** con tasas estocásticas, futuros ≠ forwards por el marking-to-market diario; en commodities, convenience yield y almacenamiento no son observables; y el roll yield de una posición continuamente rolada no coincide con el excess return del spot.
- **Implicancia práctica — condicionada, no normativa.** Si se trabaja con la variación del precio del futuro, restar $r$ **probablemente** duplica el descuento. Si en cambio se construyó un índice de retorno total colateralizado, restar $r$ **sí** es lo correcto. **La respuesta depende de qué serie se construyó, y no puede resolverse con el Capítulo 1 → PREGUNTA ABIERTA** (Cap. 6 y literatura específica de futuros).
- **Lo que sí es sólido y se mantiene:** no deben compararse directamente retornos de futuros con retornos totales de un índice de acciones, que incluyen dividendos (como los datos CRSP que usa Tsay). Esa incompatibilidad no depende de supuestos de pricing.
- Implicancia para features cross-market: si se usa un ETF o un índice cash como variable explicativa junto al futuro, sus definiciones de retorno **no son homogéneas**. Hay que homogeneizar antes de mezclar.

**[B] Simetría long/short en futuros.** El *Remark* de Tsay sobre las fricciones del short selling (préstamo de acciones, obligación de pagar dividendos) es un recordatorio de que en equities long y short **no son simétricos**. En futuros esa asimetría institucional prácticamente desaparece: ir corto es tan simple como ir largo. **[B]** Esto tiene una consecuencia de modelado concreta: en futuros es legítimo plantear targets y estrategias simétricas respecto de cero. Pero la simetría *institucional* no implica simetría *estadística*: la Tabla 1.2 de Tsay muestra skewness negativa persistente en índices, y en 1.3 Tsay señala que los extremos positivos son críticos justamente para quien está corto.

**[B] Qué comprobar.** Medir skewness y kurtosis por separado en la cola izquierda y derecha del instrumento objetivo, y comparar el desempeño potencial de reglas simétricas long/short. Verificar si las rachas de volatilidad son más severas tras movimientos negativos (efecto apalancamiento) → esto conecta con el Cap. 3, **PREGUNTA ABIERTA** en esta etapa.

### 2.1.5 Comparación de representaciones (síntesis solicitada)

Tsay **no** hace esta comparación de forma explícita ni recomienda una representación. Lo que sigue es **[B]**, construido a partir de las propiedades que sí establece.

| Representación | Propiedades estadísticas | Ventajas | Desventajas | Cuándo puede ser preferible |
|---|---|---|---|---|
| **Precio $P_t$** | No estacionario; media y varianza dependen del tiempo; escala arbitraria | Preserva niveles, soportes/resistencias, distancia a extremos | Incomparable en el tiempo y entre instrumentos; extrapolación fuera de muestra; interacción severa con roll adjustment | Sólo como insumo de features relativos (distancia normalizada a un nivel), nunca como input crudo |
| **Diferencia $P_t-P_{t-1}$** | Elimina el nivel, pero **no la escala**: su varianza crece con el nivel de precio | Aditiva en el tiempo de forma exacta; se traduce directamente a P&L (puntos × point value) | Varianza no homogénea a lo largo de la historia; incomparable entre instrumentos; una diferencia de 10 puntos no significa lo mismo en 2010 que en 2025 | Contabilidad de P&L; análisis de microestructura donde el tick es la unidad natural |
| **Retorno simple $R_t$** | Libre de escala; cota inferior −1; agregación temporal multiplicativa; agregación de cartera exacta | Interpretación económica directa; correcto para agregar carteras | Multiperiodo no tratable; asimetría inducida por el producto | Cálculo de P&L de cartera; comunicación de resultados |
| **Log-retorno $r_t$** | Libre de escala; sin cota inferior; **agregación temporal aditiva exacta**; estadísticamente más tratable [A] | Targets multi-horizonte coherentes; varianza escala con $h$; simetría de tratamiento long/short | No aditivo en cartera; para retornos grandes se aleja del simple | Modelado, features, targets, tests estadísticos |
| **Retorno acumulado $r_t[k]$** | Suma de $k$ log-retornos; varianza $\approx k\sigma^2$ bajo independencia; solapamiento induce autocorrelación artificial si las ventanas se superponen | Suaviza ruido; mejora la relación señal/ruido si la señal persiste | **Ventanas solapadas violan la independencia entre observaciones** y inflan la significancia estadística | Targets a horizonte, con tratamiento explícito del solapamiento en validación |

**[B] Criterio general que se desprende:** *la representación debe elegirse por la propiedad de agregación que uno necesite que sea exacta en el paso siguiente del pipeline*. No hay una representación óptima global; hay una representación correcta por etapa. Y una regla dura: **la representación usada para entrenar y la usada para contabilizar el P&L no tienen por qué ser la misma, pero la conversión entre ambas debe ser exacta y auditada.**

**[B] Advertencia sobre el target acumulado solapado.** Es el error más frecuente y más caro. Si el target es $\sum_{j=1}^{20} r_{t+j}$ y las observaciones son diarias, dos observaciones consecutivas comparten 19 de 20 términos. La correlación entre targets vecinos es ~0.95, el tamaño de muestra efectivo es ~$T/20$, y cualquier métrica de validación que asuma observaciones independientes está inflada por un factor grande. Tsay no discute esto (no habla de validación), pero es una consecuencia directa de su Ec. de agregación temporal.

---

## 2.2 Sección 1.2 — Distributional Properties of Returns (marco general)

**[A] Qué afirma Tsay.** Para estudiar retornos conviene empezar por sus propiedades distribucionales. El objetivo es entender el comportamiento de los retornos **a través de activos y a lo largo del tiempo**. Define el objeto de estudio como una colección de $N$ activos observados durante $T$ períodos: $\{r_{it}; i=1,\dots,N;\ t=1,\dots,T\}$, y menciona que también pueden considerarse retornos simples $\{R_{it}\}$ y excess log returns $\{z_{it}\}$.

**[B] Por qué esta formulación importa para el proyecto.** Tsay plantea el problema desde el inicio como un **panel** (activos × tiempo), no como una serie única. Esto es exactamente la estructura de un dataset de futuros múltiples. Las dos direcciones del panel corresponden a las dos arquitecturas que el proyecto contempla:
- Dirección $t$ (dinámica de un activo) → Enfoque B: un futuro objetivo, historia propia como información.
- Dirección $i$ (relación entre activos en un mismo $t$) → Enfoque A y features cross-market.

Tsay señalará en 1.2.2 que teorías como CAPM se ocupan de la distribución conjunta de $N$ retornos en un **único** $t$, mientras otras se ocupan de la dinámica de un activo individual, y que el libro cubre ambas. **[B]** Es útil tener presente esta distinción desde el principio, porque un dataset "apilado" de varios futuros mezcla las dos direcciones y las consecuencias estadísticas de esa mezcla son distintas en cada eje.

## 2.3 Sección 1.2.1 — Review of Statistical Distributions and Their Moments

Esta subsección es el andamiaje formal. Su valor no es novedad conceptual sino **precisión de vocabulario**: define los objetos que después se usan en todo el libro y, por extensión, todo lo que un sistema de ML necesita nombrar correctamente.

### 2.3.1 Conjunta, marginal, condicional, independencia

**[A] Qué afirma Tsay.**
- Distribución conjunta: $F_{X,Y}(x,y;\theta)=P(X\le x, Y\le y;\theta)$, con densidad conjunta si existe.
- Marginal: se obtiene integrando la otra variable, $F_X(x;\theta)=F_{X,Y}(x,\infty,\dots,\infty;\theta)$.
- CDF (caso escalar): no decreciente, $F(-\infty)=0$, $F(\infty)=1$. Se usa para calcular p-valores.
- **Cuantil**: para una probabilidad $p$, el cuantil $100p$-ésimo es $x_p=\inf_x\{x\mid p\le F_X(x)\}$.
- Densidad condicional (Ec. 1.8): $f_{x|y}(x;\theta)=f_{x,y}(x,y;\theta)/f_y(y;\theta)$.
- **Identidad fundamental** (Ec. 1.9): $f_{x,y}(x,y;\theta)=f_{x|y}(x;\theta)\times f_y(y;\theta)$. Tsay dice explícitamente que esta identidad **se usa extensivamente en análisis de series de tiempo, por ejemplo en estimación por máxima verosimilitud**.
- **Independencia**: $X$ e $Y$ son independientes si y sólo si $f_{x|y}(x;\theta)=f_x(x;\theta)$, en cuyo caso la conjunta factoriza en el producto de marginales.

**[A] Significado estadístico.** La definición de independencia vía "la condicional es igual a la marginal" es la definición operativa que Tsay usará en 1.2.2 para caracterizar el random walk. Es más fuerte que ausencia de correlación: no correlación restringe sólo un momento; independencia restringe la distribución entera.

**[B] Por qué es central para ML.** Esta distinción es **la definición formal de "existe información predictiva"**:
- Si $f(r_{t+h}\mid X_t)=f(r_{t+h})$ para todo $X_t$ construido con información hasta $t$, entonces **no hay nada que aprender**: el mejor modelo posible es la distribución incondicional, y cualquier modelo que parezca superarla lo está haciendo por sobreajuste.
- Si $f(r_{t+h}\mid X_t)\neq f(r_{t+h})$, hay dependencia, y **entonces** cabe preguntarse si es explotable.
- Nótese que la diferencia puede estar en **cualquier** aspecto de la distribución: media, varianza, asimetría, probabilidad de cola. Un modelo de ML entrenado sólo para predecir la media condicional puede ser ciego a una dependencia real que vive en el segundo momento. Esto es exactamente lo que Tsay anticipa en 1.3.

**[B] Cuantiles y ML.** Tsay define cuantiles para calcular p-valores, pero el objeto es directamente reutilizable: predecir cuantiles condicionales $x_p(X_t)$ (regresión cuantílica) es una forma de aprender sobre $f(r_{t+h}\mid X_t)$ sin comprometerse con una familia paramétrica. Es una alternativa natural a predecir la media cuando la distribución tiene colas pesadas. **[B]** Tsay no lo dice; lo conecta implícitamente en el Cap. 7 (VaR).

### 2.3.2 Momentos, skewness, kurtosis

**[A] Qué afirma Tsay.**
- Momento $\ell$-ésimo: $m_\ell'=E(X^\ell)$; momento central: $m_\ell=E[(X-\mu_x)^\ell]$, **siempre que la integral exista**.
- El primer momento (media) mide localización central; el segundo central (varianza $\sigma_x^2$) mide variabilidad; su raíz positiva es la desviación estándar.
- **"Los primeros dos momentos determinan unívocamente una distribución normal. Para otras distribuciones, los momentos de orden superior también son de interés."**
- Tercer momento central → simetría; cuarto → comportamiento de colas.
- Skewness $S(x)=E[(X-\mu_x)^3/\sigma_x^3]$; Kurtosis $K(x)=E[(X-\mu_x)^4/\sigma_x^4]$.
- **Excess kurtosis** $=K(x)-3$, porque $K=3$ para la normal. Excess kurtosis positiva ⇒ **colas pesadas**: la distribución pone más masa en las colas que la normal, y "en la práctica esto significa que una muestra aleatoria de tal distribución tiende a contener más valores extremos". Se llama **leptocúrtica**. Excess kurtosis negativa ⇒ colas cortas, **platicúrtica** (ej.: uniforme).
- Estimadores muestrales: media (1.10), varianza con $T-1$ (1.11), skewness (1.12) y kurtosis (1.13), ambas estandarizadas por $\hat\sigma^3$ y $\hat\sigma^4$ y divididas por $(T-1)$.

**[A] Tests de normalidad.** Bajo normalidad, $\hat S(x)$ y $\hat K(x)-3$ son asintóticamente normales con media cero y varianzas $6/T$ y $24/T$ respectivamente (Snedecor y Cochran, 1980). De ahí:
- $t=\hat S(r)/\sqrt{6/T}$ para $H_0: S(r)=0$;
- $t=(\hat K(r)-3)/\sqrt{24/T}$ para $H_0: K(r)-3=0$;
- **Jarque–Bera**: $JB=\dfrac{\hat S^2(r)}{6/T}+\dfrac{[\hat K(r)-3]^2}{24/T}$, asintóticamente $\chi^2_2$.

**[A] Ejemplo 1.2 (IBM diario, T = 9,845).** Skewness muestral 0.0614 → $t=0.0614/0.0247=2.49$, p ≈ 0.013 ⇒ los retornos simples diarios de IBM están significativamente sesgados a la derecha al 5%. Sobre log-retornos: t-test de media cero da **t = 1.5126, p = 0.1304** (IC 95%: −0.0076 a 0.0593) ⇒ **no se rechaza que la media sea cero**. Jarque–Bera = 60,921.93 con p-valor 0.00 ⇒ normalidad rechazada de forma abrumadora.

**[B] Significado ampliado — la existencia de momentos no es gratuita.** Tsay incluye la cláusula "provided that the integral exists" casi al pasar, pero es una advertencia sustantiva. Si la distribución verdadera tiene colas suficientemente pesadas, el cuarto momento (o incluso el segundo) puede no existir. En ese caso:
- La kurtosis muestral **no converge a nada**; simplemente crece con el tamaño de muestra y está dominada por la observación más extrema.
- La "kurtosis de 55.25" de Citigroup no debe leerse como una estimación estable de un parámetro poblacional, sino como un indicador de que hubo observaciones enormes.
- Los tests de skewness/kurtosis basados en $6/T$ y $24/T$ suponen normalidad bajo la nula; una vez rechazada, esas varianzas asintóticas ya no describen la incertidumbre real de los estimadores.

**[B] Advertencia crítica sobre JB en datos financieros.** JB y los tests de momentos suponen observaciones **iid**. Los retornos financieros presentan volatility clustering (el propio Tsay lo afirma en 1.3), es decir, **no son iid** aunque no estén correlacionados. Bajo dependencia en varianza, el p-valor de JB está severamente distorsionado. La conclusión "los retornos no son normales" sobrevive de todos modos (el estadístico es astronómico), pero **el test no distingue entre "distribución marginal no normal" y "distribución condicional normal con varianza variable"**. Ésta es una de las confusiones más importantes de todo el capítulo, y Tsay la deja implícita:

> Una serie con distribución condicional **normal** pero varianza condicional variable en el tiempo produce una distribución **marginal** con colas pesadas.

Es decir, la leptocurtosis observada **no prueba** que la distribución condicional sea de colas pesadas. Puede ser una mixtura de normales generada por el propio clustering de volatilidad — que es justamente el modelo de "scale mixture of normals" que Tsay presenta en 1.2.2. **PREGUNTA ABIERTA hasta el Cap. 3:** ¿cuánto de la kurtosis de un futuro dado desaparece al estandarizar por una estimación de volatilidad condicional? Es una pregunta empírica concreta y de altísimo valor para el diseño del sistema.

**[B] Implicancias para ML.**
1. **Sobre pérdidas.** MSE penaliza el error al cuadrado; bajo colas pesadas, el gradiente del entrenamiento queda dominado por un puñado de observaciones. Un modelo entrenado con MSE sobre retornos crudos está, en la práctica, ajustando los pocos días extremos. Alternativas: Huber, pérdidas sobre retornos estandarizados por volatilidad, o pérdidas explícitamente robustas. **La elección debe ser consciente y testeada, no heredada por defecto.**
2. **Sobre métricas.** Un Sharpe ratio, un t-stat de una estrategia, o un intervalo de confianza calculado con fórmulas gaussianas subestiman groseramente la incertidumbre cuando la kurtosis es 10–50. Un Sharpe de 1.5 estimado sobre 2 años de datos con colas pesadas puede no ser distinguible de cero.
3. **Sobre el hallazgo de la media.** El hecho de que en ~10⁴ observaciones diarias no se pueda rechazar media cero establece la escala del problema: **la relación señal/ruido en el primer momento de retornos diarios es del orden de 10⁻²**. Un sistema de ML que busca señal direccional está buscando una perturbación pequeñísima en la media condicional. Esto determina cuánta data se necesita, cuánta capacidad de modelo es admisible, y cuán severo debe ser el control de sobreajuste.
4. **Sobre estandarización de features.** Estandarizar por media y desviación estándar (z-score) es un procedimiento que sólo captura información completa si la distribución es normal. Con leptocurtosis, el z-score no acota los valores y produce features con outliers extremos. Rank-transform, winsorización o normalización por volatilidad condicional son alternativas a evaluar empíricamente.

**[B] Qué comprobar empíricamente sobre futuros.**
- Calcular media, sd, skewness, excess kurtosis, min, max, y cuantiles (1%, 5%, 25%, 50%, 75%, 95%, 99%) por instrumento y por frecuencia (1min, 5min, 15min, 1h, diario).
- Repetir por sub-períodos (por año, por régimen) para ver si esos momentos son estables — es decir, un test informal de estacionariedad en los momentos.
- Recalcular la kurtosis excluyendo el 0.1% de observaciones más extremas: si cae drásticamente, la estimación estaba dominada por unos pocos puntos y no es un parámetro confiable.
- Repetir la kurtosis sobre retornos **estandarizados por una volatilidad realizada rezagada**: cuantifica qué fracción de la leptocurtosis es atribuible a heterocedasticidad.
- Aplicar el t-test de media cero (como hace Tsay) sobre el instrumento objetivo, para medir cuán cerca de cero está el drift y con qué incertidumbre.

## 2.4 Sección 1.2.2 — Distributions of Returns

Ésta es, para el proyecto, **la sección más importante del capítulo**.

### 2.4.1 El modelo general y su factorización

**[A] Qué afirma Tsay.**
- El modelo más general para los log-retornos es su distribución conjunta (Ec. 1.14):
  $$F_r(r_{11},\dots,r_{N1};\ r_{12},\dots,r_{N2};\ \dots;\ r_{1T},\dots,r_{NT};\ Y;\ \theta)$$
  donde **$Y$ es un vector de estado** que contiene variables que resumen el entorno en el que se determinan los retornos, y $\theta$ es el vector de parámetros. En muchos estudios financieros $Y$ se toma como dado y el interés está en la distribución **condicional** de $\{r_{it}\}$ dado $Y$.
- Dice explícitamente que **este modelo es demasiado general para tener valor práctico**, pero que provee el marco donde ubicar cualquier modelo econométrico de retornos.
- Distingue: CAPM (Sharpe, 1964) se centra en la distribución conjunta de $N$ retornos en un único $t$; otras teorías, en la dinámica de un activo individual. El libro cubre ambas.
- **Factorización (Ec. 1.15):**
  $$F(r_{i1},\dots,r_{iT};\theta)=F(r_{i1})\prod_{t=2}^{T}F(r_{it}\mid r_{i,t-1},\dots,r_{i1})$$
  y su versión en densidades (Ec. 1.16). Dice que esta partición **"resalta las dependencias temporales"** y que **"el asunto principal es entonces la especificación de la distribución condicional $F(r_{it}\mid r_{i,t-1},\cdot)$, en particular, cómo evoluciona en el tiempo"**.
- **"En finanzas, distintas especificaciones distribucionales conducen a distintas teorías."**
- **Random walk:** "una versión de la hipótesis de camino aleatorio es que la distribución condicional $F(r_{it}\mid r_{i,t-1},\dots,r_{i1})$ es igual a la distribución marginal $F(r_{it})$. En ese caso, los retornos son **temporalmente independientes** y, por lo tanto, **no predecibles**."
- Cierra el punto: **"la Ecuación (1.16) sugiere que las distribuciones condicionales son más relevantes que las marginales para estudiar retornos de activos."** Pero matiza: las marginales siguen siendo de interés porque son **más fáciles de estimar** con retornos pasados, y porque **en algunos casos los retornos tienen correlaciones seriales empíricas débiles y entonces sus marginales están cerca de sus condicionales**.

**[B] Lectura estructural: esta sección define el problema de ML.**

Un sistema que aprende $X_t \to y_{t+h}$ está estimando un funcional de $F(r_{t+h}\mid \mathcal{F}_t)$ donde $\mathcal{F}_t$ es la información disponible en $t$. Traducción término a término:

| Objeto en Tsay | Equivalente en el sistema de ML |
|---|---|
| $F(r_t \mid r_{t-1},\dots)$ | La distribución que el modelo intenta aproximar |
| $Y$ (vector de estado del entorno) | **Features exógenos**: otros futuros, macro, calendario, volatilidad implícita, hora del día |
| $\theta$ | Parámetros del modelo |
| Factorización 1.15 | Justificación formal de la predicción secuencial one-step-ahead y del walk-forward |
| $F$ condicional $=$ marginal | **La hipótesis nula del proyecto**: no hay señal |
| Especificación de la condicional | La elección conjunta de arquitectura + función de pérdida |

**[B] El vector de estado $Y$ es el permiso teórico para features cross-market.** Tsay introduce $Y$ como "variables que resumen el entorno en el que se determinan los retornos" y lo deja deliberadamente abstracto. Es el hueco formal donde entra toda la ingeniería de features: retornos de otros futuros, spreads, term structure, volatilidad de otros mercados, variables de calendario. Lo importante es que **$Y$ está dentro del modelo general desde el inicio**: usar información externa a la serie propia no es una desviación del marco, es parte de él. Lo que el marco exige es que $Y$ esté disponible en $t$ — es decir, la restricción de no usar información futura está codificada en la propia estructura condicional, aunque Tsay no la enuncie como advertencia.

**[B] La factorización justifica el esquema de validación.** Si la conjunta se descompone en un producto de condicionales secuenciales, la forma natural de evaluar un modelo es **predecir cada $t$ usando sólo información anterior a $t$ y acumular el resultado**. Eso es exactamente walk-forward / expanding window / purged time-series CV. La validación cruzada aleatoria (k-fold barajado) rompe la factorización: entrena con $t+5$ para predecir $t$, lo cual no corresponde a ninguna cantidad estadística bien definida en este marco. **Tsay no discute validación**, pero su Ec. 1.15 hace que el error sea evidente.

### 2.4.2 Qué significa realmente "hay información predictiva"

Esta es la parte que el proyecto pidió desarrollar con especial cuidado. Se separa estrictamente.

**[A] Lo que Tsay establece.**
1. $F(r_t)$ es la distribución **marginal** (incondicional): describe cómo se ven los retornos sin usar ninguna información.
2. $F(r_t\mid r_{t-1},r_{t-2},\dots)$ es la **condicional**: cómo se ven los retornos dado el pasado.
3. Si ambas coinciden ⇒ independencia temporal ⇒ **no predecibilidad**.
4. Cuando las correlaciones seriales empíricas son débiles, marginal y condicional están *cerca*.

**[B] Desarrollo: cuatro nociones que se confunden habitualmente.**

Es útil ordenarlas como una jerarquía, de la más débil a la más exigente. Cada nivel implica el anterior, pero no al revés.

**Nivel 1 — Autocorrelación (dependencia lineal en la media).**
$\text{Corr}(r_t, r_{t-k})\neq 0$. Es la forma más restrictiva y más fácil de medir. Es lo que estudia el Cap. 2.
- Autocorrelación $\neq 0$ ⇒ dependencia. Correcto.
- Autocorrelación $=0$ ⇒ **NO** implica independencia. Éste es el error más importante.
- **[A] Tsay lo sugiere** al decir que las marginales están "cerca" de las condicionales cuando la correlación serial es débil — nótese que dice *cerca*, no *iguales*, y el matiz es deliberado.
- **[B] En futuros líquidos** cabe esperar autocorrelación de retornos cercana a cero en frecuencias diarias (la competencia arbitra la dependencia lineal simple). Eso **no** cierra la discusión.

**Nivel 2 — Dependencia estadística (cualquier orden, cualquier momento).**
$F(r_t\mid \text{pasado})\neq F(r_t)$. Puede manifestarse en varianza (GARCH), en asimetría, en probabilidad de cola, en relaciones no lineales, en interacciones con otros mercados.
- **[A] El clustering de volatilidad que Tsay describe en 1.3 es exactamente esto**: dependencia sin necesidad de autocorrelación en la media.
- **[B] Consecuencia central para el proyecto:** un mercado puede ser simultáneamente **no predecible en dirección** y **muy predecible en magnitud**. Estas son dos preguntas separadas y deben tratarse como tales.

**Nivel 3 — Predecibilidad económica.**
La dependencia existe y además tiene la forma y el tamaño necesarios para producir un retorno esperado positivo con una regla implementable.
- Requiere que la dependencia sea **estable** (no un artefacto de un sub-período), **direccional o traducible a una posición**, y de magnitud superior al ruido en horizonte útil.
- **[B]** Una dependencia real puede ser económicamente inútil: por ejemplo, saber que la varianza de mañana será alta no dice si conviene ir largo o corto, aunque sí dice cuánto arriesgar.

**Nivel 4 — Predecibilidad explotable neta de costos.**
El retorno esperado supera spread + comisión + slippage + costos de roll + impacto, en el tamaño y frecuencia con que se opera.
- **[A] Tsay no menciona costos de transacción en el Capítulo 1**, salvo el Ejercicio 1.3(b) que dice explícitamente "assume that there were no transaction costs" — es decir, el libro los excluye por diseño en este punto.
- **[B]** Esta es enteramente una extensión. Pero es la que determina si el proyecto tiene sentido. En futuros, el filtro es cuantificable: si el edge por trade es menor que el tick, la señal no existe operativamente. Un modelo con un edge de 0.3 ticks es estadísticamente interesante y económicamente nulo.

**[B] La trampa del razonamiento inverso.** El sistema de ML busca patrones históricos. Encontrar un patrón en la muestra **no** es evidencia de que $F(r_{t+h}\mid X_t)\neq F(r_{t+h})$ en la población, por dos razones:
1. **Multiplicidad.** Probando suficientes features, modelos y horizontes, aparecen patrones en datos generados por un random walk puro. La probabilidad de encontrar "algo" con p < 0.05 tras 100 pruebas independientes es 99.4%.
2. **No estacionariedad.** Aunque la dependencia haya existido, puede no existir ya. La factorización de Tsay asume implícitamente un $\theta$ fijo; si $\theta$ cambia con el tiempo, un patrón real de 2010 puede ser ruido en 2025. **PREGUNTA ABIERTA:** este es el terreno de los modelos Markov switching y state-space (Caps. 11–12).

**[B] Prueba de sanidad recomendada.** Antes de creer cualquier resultado: correr exactamente el mismo pipeline sobre (i) series de retornos permutadas aleatoriamente, y (ii) random walks simulados con la misma volatilidad y kurtosis. Si el pipeline "encuentra" señal ahí, el pipeline está roto, no el mercado.

### 2.4.3 Discreción de precios y alta frecuencia

**[A] Qué afirma Tsay.** Es costumbre tratar los retornos como variables continuas, **especialmente para índices o acciones a baja frecuencia**. Pero: "para retornos de alta frecuencia, la discreción se vuelve un problema". Los precios en NYSE cambian en múltiplos del **tick size**: 1/8 de dólar antes de julio 1997, 1/16 entre julio 1997 y enero 2001, y decimalización completa desde el 29 de enero de 2001 (con programas piloto desde agosto/septiembre/diciembre de 2000). Por lo tanto **el retorno tick-a-tick de una acción individual no es continuo**. Remite al Cap. 5.

**[A] Además**, en la introducción del capítulo Tsay adelanta que el Cap. 5 muestra que **el trading no sincrónico y el bid–ask bounce pueden introducir correlaciones seriales en el retorno de una acción**.

**[B] Qué problemas deja planteados el Capítulo 1 para alta frecuencia en futuros.** Sin adelantar el Cap. 5:

1. **La grilla de precios acota la resolución de la señal.** Si el tick es 0.25 y el precio 5,000, el retorno mínimo no nulo es 5·10⁻⁵. En barras de 1 minuto donde la desviación estándar del retorno es del orden de 2–5 ticks, la variable objetivo toma un número pequeño de valores distintos. La distribución condicional es efectivamente **discreta**, y modelarla como continua es un supuesto que se degrada al bajar de frecuencia temporal.
2. **La discreción induce artefactos estadísticos.** Momentos, tests de normalidad y densidades estimadas sobre una variable discreta con pocos valores tienen interpretaciones distintas. La kurtosis de retornos tick-a-tick no mide lo mismo que la kurtosis de retornos diarios.
3. **Correlación serial de origen microestructural.** [A] Tsay anticipa que el bid–ask bounce genera autocorrelación. **[B]** Esto es una trampa directa para ML: es autocorrelación negativa de primer orden **que no es predecibilidad económica** — refleja el rebote entre bid y ask, no información. Un modelo que la detecte y la explote "en backtest" estará comprando al bid y vendiendo al ask en la simulación, cosa imposible en la práctica. **Esta es probablemente la principal fuente de falsos positivos en ML de alta frecuencia.**
4. **Trading no sincrónico entre instrumentos.** [A] Tsay lo menciona para acciones. **[B]** En futuros el análogo es distinto pero real: distintos contratos tienen horarios de sesión, liquidez y momentos de mayor actividad diferentes (ES 24h, GC con sesión asiática activa, Treasuries con dinámica ligada a subastas y datos macro). Alinear barras de dos futuros por timestamp no garantiza que ambas contengan la misma cantidad de información. **Las relaciones lead-lag aparentes pueden ser puro artefacto de sincronización.**
5. **Frecuencia de muestreo y agregación temporal.** [A] La agregación de log-retornos es aditiva exacta. **[B]** Esto significa que la frecuencia de muestreo es una decisión libre y reversible hacia abajo (se puede agregar de 1min a 5min, no al revés). Pero la agregación **cambia las propiedades**: promedia el ruido de microestructura, reduce kurtosis, y puede destruir o revelar señal. La Tabla 1.2 de Tsay muestra un caso concreto de este efecto: la kurtosis de los retornos **mensuales** es mucho menor que la de los **diarios** (IBM: 9.92 diario vs 3.43 mensual, retornos simples). **[B] La frecuencia de muestreo es un hiperparámetro del problema, no una propiedad del dato**, y debería tratarse como tal.
6. **Barras alternativas.** **[B]** La aditividad de log-retornos permite construir barras por volumen, por número de ticks o por cantidad de movimiento (dollar/volume/imbalance bars) sin romper la coherencia de los retornos. Esto queda como una posibilidad abierta a evaluar; nada en Tsay lo sugiere ni lo contradice.

### 2.4.4 Las cuatro familias de distribuciones marginales

**[A] Tsay presenta cuatro candidatas y evalúa cada una:**

**(1) Normal.** El supuesto tradicional: los retornos simples $\{R_{it}\}$ son iid normales con media y varianza fijas. Ventaja: hace tratables las propiedades estadísticas. **Tres dificultades explícitas:**
   - (i) la cota inferior de un retorno simple es −1, pero la normal tiene soporte en toda la recta real y por lo tanto **no tiene cota inferior**;
   - (ii) si $R_{it}$ es normal, el retorno multiperiodo $R_{it}[k]$ **no** es normal, porque es un producto de retornos de un período;
   - (iii) **el supuesto de normalidad no es respaldado por muchos retornos empíricos, que tienden a tener excess kurtosis positiva**.

**(2) Lognormal.** Supuesto alternativo: los log-retornos $r_t$ son iid $N(\mu,\sigma^2)$, con lo cual los retornos simples son iid lognormales, con (Ec. 1.17):
$$E(R_t)=\exp\!\left(\mu+\frac{\sigma^2}{2}\right)-1,\qquad \text{Var}(R_t)=\exp(2\mu+\sigma^2)[\exp(\sigma^2)-1]$$
Y la relación inversa, si $m_1$ y $m_2$ son media y varianza del retorno simple:
$$E(r_t)=\ln\frac{m_1+1}{\sqrt{1+m_2/(1+m_1)^2}},\qquad \text{Var}(r_t)=\ln\left[1+\frac{m_2}{(1+m_1)^2}\right]$$
Tsay dice que estas dos ecuaciones **son útiles al estudiar retornos, por ejemplo al pronosticar usando modelos construidos sobre log-retornos**.
   - Ventajas: como la suma de normales iid es normal, $r_t[k]$ también es normal ⇒ la agregación temporal es consistente; no hay cota inferior para $r_t$; y la cota de $R_t$ se satisface automáticamente vía $1+R_t=\exp(r_t)$.
   - **Dificultad:** el supuesto lognormal **no es consistente con todas las propiedades de los retornos históricos**; en particular, muchos retornos accionarios exhiben excess kurtosis positiva.

**(3) Stable distributions.** Generalización natural de la normal: son **estables bajo adición**, lo que encaja con la necesidad de agregar log-retornos. Además **son capaces de capturar la excess kurtosis** observada.
   - **Problemas graves:** las estables no normales **no tienen varianza finita**, lo cual entra en conflicto con la mayoría de las teorías financieras; y el modelado estadístico con ellas es difícil. Ejemplo: la Cauchy, simétrica respecto de su mediana pero con varianza infinita, densidad $f(x)=1/[\pi(1+x^2)]$.

**(4) Scale mixture / finite mixture of normals.** Los estudios recientes tienden a usar esta familia. Bajo mixtura de escala, $r_t\sim N(\mu,\sigma^2)$ **pero $\sigma^2$ es una variable aleatoria** con distribución positiva (p. ej. $\sigma^{-2}$ gamma). Ejemplo de mixtura finita:
$$r_t\sim(1-X)N(\mu,\sigma_1^2)+X\,N(\mu,\sigma_2^2),\qquad X\sim\text{Bernoulli}(\alpha)$$
con $\sigma_1^2$ pequeña y $\sigma_2^2$ relativamente grande. Con $\alpha=0.05$: el 95% de los retornos vienen de la normal de varianza chica y el 5% de la de varianza grande, lo que pone más masa en las colas.
   - **Ventajas:** mantienen la tratabilidad de la normal, tienen **momentos de orden superior finitos**, y capturan la excess kurtosis.
   - **Desventaja:** los parámetros de la mixtura son difíciles de estimar (p. ej. $\alpha$).
   - **Figura 1.1:** compara mixtura finita $(1-X)N(0,1)+X\,N(0,16)$ con $\alpha=0.05$, Cauchy y normal estándar. **Cauchy tiene colas más gruesas que la mixtura, que a su vez las tiene más gruesas que la normal.**

**[B] Lectura del conjunto: es un menú de trade-offs, no una recomendación.** Tsay describe cuatro familias y muestra que **ninguna satisface simultáneamente todo lo deseable**:

| Familia | Cota inferior de $R$ | Agregación temporal | Captura colas pesadas | Varianza finita | Facilidad de estimación |
|---|---|---|---|---|---|
| Normal sobre $R_t$ | ✗ | ✗ (producto) | ✗ | ✓ | ✓✓ |
| Lognormal (normal sobre $r_t$) | ✓ | ✓ (suma) | ✗ | ✓ | ✓✓ |
| Estable no normal | ✓ | ✓ | ✓ | **✗** | ✗ |
| Mixtura de normales | ✓ | parcial | ✓ | ✓ | ✗ (difícil) |

**[B] La mixtura de escala es conceptualmente el puente hacia GARCH.** "Normal con varianza aleatoria" es formalmente casi lo mismo que "normal con varianza condicional variable en el tiempo". La diferencia es que en una mixtura la varianza se sortea de forma independiente en cada $t$, mientras que en un modelo de volatilidad condicional la varianza es **predecible** desde el pasado. **[B] Esto es una observación mía, no de Tsay**, pero explica por qué el capítulo presenta esta familia y no otra: es la que conecta el hecho estilizado (colas pesadas) con el modelo que vendrá (Cap. 3). Reformulado como pregunta empírica: *¿las colas pesadas del futuro objetivo son una propiedad intrínseca de la distribución condicional, o son el subproducto de una varianza condicional que sí se puede predecir?* La respuesta cambia por completo el diseño del sistema.

**[B] Consecuencias de asumir normalidad incorrectamente — desglose por área solicitada.**

- **Targets.** Si el target es el retorno crudo y se asume normalidad implícita, se subestima la frecuencia de valores extremos. Un target "retorno a 20 barras" tendrá, en la práctica, muchas más observaciones a ±4σ de las que la normal predice (la normal predice ~1 cada 15,000; los datos financieros pueden dar 1 cada pocos cientos). Cualquier binning por desviaciones estándar (p. ej. clases "movimiento grande / pequeño") queda mal calibrado.
- **Funciones de pérdida.** MSE = MLE gaussiano. Bajo colas pesadas, el estimador resultante no es eficiente y, peor, el entrenamiento se concentra en pocos puntos. El modelo aprende a ajustar los crashes y desatiende el 99% del comportamiento normal — o, si se regulariza fuerte, hace lo contrario.
- **Probabilidades estimadas.** Un clasificador que produce $P(\text{sube})$ calibrado sobre datos con distribución cambiante estará mal calibrado en el régimen que importa. Y una probabilidad convertida a tamaño de posición vía fórmulas gaussianas (Kelly con normal, por ejemplo) sobredimensiona sistemáticamente.
- **Intervalos de confianza.** Un IC de 95% construido como $\hat\mu\pm1.96\hat\sigma$ subestima el ancho real de la cola. Para VaR el error es directo y grave: el VaR al 99% calculado con normal puede subestimar la pérdida real en un factor considerable. Tsay dedica el Cap. 7 a esto.
- **Detección de movimientos extremos.** Definir "extremo" como ">3σ" bajo supuesto normal implica esperar ~1 evento cada 370 observaciones. Con kurtosis de 20, esos eventos ocurren mucho más seguido, y una lógica de "circuit breaker" calibrada bajo normalidad se disparará constantemente.
- **Evaluación de modelos de ML.** Los tests de significancia de diferencias entre modelos (¿el modelo A es mejor que el B?) basados en normalidad de los errores están mal calibrados. Y las métricas de performance financiera (Sharpe, ratio de Sortino) tienen distribuciones muestrales mucho más anchas de lo habitualmente asumido.

**[B] Lo que NO se debe concluir todavía.** Tsay no elige una distribución, y el proyecto no debe hacerlo en esta etapa. La pregunta correcta a esta altura no es "¿qué distribución tienen los futuros?" sino **"¿qué parte de la no normalidad marginal desaparece al condicionar?"**. Sólo tras el Cap. 3 se podrá responder.

## 2.5 Sección 1.2.3 — Multivariate Returns

**[A] Qué afirma Tsay** (la sección es breve y deliberadamente introductoria):
- Sea $\mathbf{r}_t=(r_{1t},\dots,r_{Nt})'$ el vector de log-retornos de $N$ activos en $t$. Los análisis multivariados de los Caps. 8 y 10 se ocupan de la distribución conjunta de $\{\mathbf{r}_t\}_{t=1}^T$.
- **Esta conjunta se particiona de la misma manera que la Ec. 1.15**, y el análisis se centra en la especificación de $F(\mathbf{r}_t\mid \mathbf{r}_{t-1},\dots,\mathbf{r}_1,\theta)$.
- **"En particular, cómo evolucionan en el tiempo la esperanza condicional y la matriz de covarianza condicional de $\mathbf{r}_t$ constituyen los temas principales de los Capítulos 8 y 10."**
- Define vector de medias $E(X)=\mu_x$ y matriz de covarianza $\text{Cov}(X)=\Sigma_x=E[(X-\mu_x)(X-\mu_x)']$; y sus contrapartes muestrales.
- **"Estos estadísticos muestrales son estimadores consistentes de sus contrapartes teóricas siempre que la matriz de covarianza de $X$ exista."**
- "En la literatura financiera, la distribución normal multivariada se usa a menudo para el log-retorno $\mathbf{r}_t$."

**[B] Qué cambia al pasar de una serie a varias.** Cuatro cosas, en orden de dificultad creciente:

1. **El objeto a estimar crece cuadráticamente.** Con $N$ instrumentos, la covarianza tiene $N(N+1)/2$ parámetros. Con 7 futuros son 28 parámetros; con 30, son 465. Cada uno estimado sobre datos ruidosos.
2. **La estructura de dependencia contemporánea entra en juego.** Ya no basta con la dinámica propia: importa cómo se mueven juntos. Esta es la información que un modelo cross-market pretende explotar.
3. **La dependencia puede ser rezagada (lead-lag).** [A] Tsay lo remite al Cap. 8, que menciona explícitamente como dedicado a "la relación lead–lag entre series de tiempo", además de cointegración y pairs trading.
4. **La covarianza es condicional y variable.** [A] Tsay lo dice explícitamente: es el tema del Cap. 10. **[B]** Es decir: la correlación entre ES y Gold **no es un número**, es un proceso. Cualquier feature construido sobre una correlación estimada en ventana fija está midiendo un objeto que se mueve.

**[B] La advertencia sobre existencia de momentos es la más importante de la sección.** "Consistentes **siempre que la matriz de covarianza exista**". Combinado con lo que el propio Tsay muestra sobre colas pesadas, esto es una advertencia práctica seria:
- Con colas pesadas, la covarianza muestral converge **lentamente** y está dominada por los días extremos.
- La correlación entre dos futuros estimada sobre 60 días puede cambiar radicalmente por un único día de crisis.
- Las correlaciones tienden a subir justamente en los eventos extremos (fenómeno bien documentado en la literatura de riesgo; **Tsay no lo afirma en el Cap. 1**), lo cual significa que la diversificación aparente en datos normales desaparece cuando más se la necesita.

**[B] Traslado al conjunto hipotético de futuros.** Considerando Nasdaq-100, S&P 500, Russell 2000, Dow Jones, Treasuries, Gold, Crude Oil (**sólo como ejemplos; no se asume que deban usarse todos, ni que añadir mercados mejore un modelo**):

- Los cuatro índices accionarios presentan correlaciones contemporáneas altas. **Pero hay que separar tres nociones que no son equivalentes:**
  - **Correlación alta** es una propiedad contemporánea de segundo momento. Con $\rho=0.9$ la varianza residual es $1-\rho^2=19\%$: decir "casi el mismo activo" sobreestima la implicación.
  - **Redundancia estadística** (colinealidad) afecta la identificación y la estabilidad de coeficientes en modelos lineales, y hace que las importancias de features sean poco fiables. Es un riesgo real de estimación, pero **no implica** contribución predictiva nula, especialmente en modelos no lineales.
  - **Información predictiva incremental** es una pregunta sobre relaciones **rezagadas**, no sobre correlación contemporánea. Dos series pueden estar muy correlacionadas a la vez y aun así aportar (o no) información sobre el futuro de la otra. Sólo se resuelve midiendo el aporte out-of-sample, con corrección por multiplicidad.
- **La hipótesis de que "la información marginal está en los spreads" (NQ−ES, RTY−ES) es una conjetura de reparametrización, no un resultado.** Un spread es una combinación lineal específica que un modelo con ambas series podría aprender por sí solo; imponerla es un sesgo inductivo que puede ayudar en régimen de baja señal/ruido pero que debe demostrarse. Conecta con la reducción de dimensión del Cap. 9 → **PREGUNTA ABIERTA**.
- Treasuries, Gold y Crude Oil aportan direcciones de variación genuinamente distintas, pero cada uno responde a drivers propios y su relación con equities **cambia de signo según el régimen** (correlación stock-bond, por ejemplo, no es estable históricamente).
- **La pregunta previa correcta no es "¿qué mercados agrego?" sino "¿cuántas dimensiones efectivas hay en este conjunto?"**. [A] Tsay remite justamente a esto: el Cap. 9 "discute formas de simplificar la estructura dinámica de una serie multivariada y métodos para reducir la dimensión", con modelos de factores. **PREGUNTA ABIERTA hasta el Cap. 9.**

**[B] Enfoque A vs Enfoque B: qué hay que resolver antes.**

**Enfoque A — un modelo entrenado con observaciones de múltiples futuros (pooling).**
La premisa implícita es que los distintos instrumentos comparten la misma función $f$ tal que $y=f(X)+\varepsilon$. Preguntas estadísticas previas:
1. **¿Las distribuciones condicionales son suficientemente similares?** Si la relación feature→target difiere entre ES y CL, el pooling promedia relaciones distintas y produce un modelo que no sirve para ninguno. Test: entrenar por separado y comparar coeficientes/predicciones cruzadas.
2. **¿Cómo se normaliza?** Sin normalización, los instrumentos de mayor volatilidad dominan la función de pérdida (aportan errores más grandes). Candidato natural: estandarizar retornos por volatilidad propia, $\tilde r_{it}=r_{it}/\hat\sigma_{it}$. Esto es **[B]**, pero se apoya directamente en la observación de Tsay de que los instrumentos difieren en desviación estándar (Tabla 1.2: índices menor sd que acciones individuales).
3. **¿Cuál es el tamaño de muestra efectivo?** Apilar 7 futuros altamente correlacionados **no** multiplica por 7 la información. Si los residuos están correlacionados contemporáneamente, el número efectivo de observaciones independientes es mucho menor. Este es un error de conteo que infla toda medida de significancia.
4. **¿Cómo se evita el leak entre instrumentos?** Si ES está en train y NQ en validation **para la misma fecha**, hay fuga: su alta correlación contemporánea hace que una observación informe sobre la otra. La partición debe ser **por tiempo**, nunca por instrumento.
5. **¿Hay survivorship / selección de instrumentos?** Elegir los futuros "que funcionaron" es data snooping a nivel de universo.

**Enfoque B — un futuro objetivo, otros futuros como información explicativa.**
Encaja limpiamente en el marco de Tsay: los otros mercados son componentes del vector de estado $Y$ de la Ec. 1.14. Preguntas previas:
1. **¿La información es contemporánea o rezagada?** Un feature contemporáneo (retorno de NQ en la misma barra $t$ para predecir ES en $t$) es **inutilizable**: no está disponible cuando hay que decidir. Sólo sirven rezagos estrictos, y su utilidad depende de que exista lead-lag real. → Cap. 8.
2. **¿La aparente relación lead-lag es real o de sincronización?** [B] Si un mercado cierra o se vuelve ilíquido antes que otro, aparecerá un lead-lag espurio. Hay que verificar con horarios de sesión y liquidez, no sólo con timestamps.
3. **¿La relación es estable?** Estimar la correlación rodante y examinar su variabilidad. Si cambia de signo, un feature lineal basado en ella es una fuente de ruido.
4. **¿Cuánta información marginal aporta cada mercado?** Un análisis incremental honesto: partir de un modelo con la propia serie, y medir la mejora out-of-sample al añadir cada mercado, con corrección por multiplicidad de pruebas.
5. **¿Se puede reducir dimensión primero?** Extraer 2–3 factores (nivel de riesgo equity, tasa, dólar/commodity) puede ser más robusto que 7 series correlacionadas. → **PREGUNTA ABIERTA, Cap. 9.**

**[B] Diferencia conceptual clave entre A y B.** Se corresponde con la distinción que Tsay hace en 1.2.2 entre teorías cross-sectional (CAPM: distribución conjunta en un único $t$) y teorías de dinámica individual. El Enfoque A apuesta a que existe una **regularidad común** entre mercados; el Enfoque B apuesta a que existe **transmisión de información** entre mercados. Son hipótesis distintas, requieren evidencia distinta, y **pueden ser ambas falsas o ambas ciertas de forma independiente**.


## 2.6 Sección 1.2.4 — Likelihood Function of Returns

**[A] Qué afirma Tsay.**
- La partición de la Ec. 1.15 permite obtener la función de verosimilitud de los log-retornos $\{r_1,\dots,r_T\}$.
- **Si la distribución condicional $f(r_t\mid r_{t-1},\dots,r_1,\theta)$ es normal con media $\mu_t$ y varianza $\sigma_t^2$**, entonces $\theta$ contiene los parámetros de $\mu_t$ y $\sigma_t^2$, y la verosimilitud es (Ec. 1.18):
  $$f(r_1,\dots,r_T;\theta)=f(r_1;\theta)\prod_{t=2}^{T}\frac{1}{\sqrt{2\pi}\,\sigma_t}\exp\!\left[\frac{-(r_t-\mu_t)^2}{2\sigma_t^2}\right]$$
  donde $f(r_1;\theta)$ es la densidad marginal de la primera observación.
- El valor de $\theta$ que maximiza esta función es el **estimador de máxima verosimilitud (MLE)**.
- Como el logaritmo es monótono, se maximiza la **log-verosimilitud**, "más fácil de manejar en la práctica":
  $$\ln f(r_1,\dots,r_T;\theta)=\ln f(r_1;\theta)-\frac{1}{2}\sum_{t=2}^{T}\left[\ln(2\pi)+\ln(\sigma_t^2)+\frac{(r_t-\mu_t)^2}{\sigma_t^2}\right]$$
- **"La log-verosimilitud puede obtenerse de manera similar si la distribución condicional no es normal."**

**[A] Significado estadístico.** Nótese lo que Tsay hace en tres líneas: convierte una **hipótesis distribucional** en un **objetivo de optimización**. Y crucialmente, $\mu_t$ y $\sigma_t^2$ llevan subíndice $t$: la formulación ya admite media y varianza variables en el tiempo. Toda la econometría posterior del libro (ARMA, GARCH, state-space, Markov switching) consiste en dar formas funcionales concretas a $\mu_t$ y $\sigma_t^2$ y maximizar esta misma expresión.

**[B] Por qué la verosimilitud aparece constantemente en series de tiempo.** Tres razones que se desprenden del texto:
1. **Es el único mecanismo general** para estimar parámetros de modelos donde las observaciones no son independientes. La factorización condicional convierte $T$ observaciones dependientes en un producto de $T$ términos manejables.
2. **Provee inferencia, no sólo estimación**: errores estándar, tests de razón de verosimilitud, y criterios de selección de modelo (AIC/BIC son log-verosimilitud penalizada).
3. **Es composicional**: se pueden acoplar un modelo de media y uno de varianza y estimarlos conjuntamente en la misma expresión.

**[B] El término $f(r_1;\theta)$ y la verosimilitud condicional.** Tsay incluye explícitamente la densidad marginal de la primera observación. En la práctica, casi todos los modelos de series de tiempo la descartan y maximizan la **verosimilitud condicional** (condicionada a los valores iniciales), porque su contribución es despreciable cuando $T$ es grande y su forma exacta es difícil. **[B] Tsay no llama a esto "conditional likelihood" en el Capítulo 1** — lo hace en capítulos posteriores —, pero la distinción ya está latente en la ecuación. Consecuencia práctica: los primeros valores de cualquier serie modelada requieren un tratamiento de arranque, y con muestras cortas eso puede sesgar la estimación.

**[B] Relación log-verosimilitud ↔ funciones de pérdida en ML.** Esta correspondencia es matemáticamente exacta y es probablemente el aporte más directamente aplicable del capítulo al diseño del sistema. **Minimizar una pérdida es maximizar la log-verosimilitud de una distribución condicional implícita**:

| Función de pérdida en ML | Distribución condicional implícita | Qué se está estimando |
|---|---|---|
| **MSE** $(y-\hat y)^2$ | Normal con $\sigma^2$ **constante** | Media condicional |
| **MSE ponderado** $(y-\hat y)^2/\sigma_t^2$ | Normal heterocedástica ($\sigma_t^2$ conocida) | Media condicional, ponderando por precisión |
| **Gaussian NLL** $\ln\sigma_t^2+(y-\mu_t)^2/\sigma_t^2$ | Normal con $\mu_t$ **y** $\sigma_t^2$ modeladas | Media **y** varianza condicionales (Ec. 1.18 exactamente) |
| **MAE** $\lvert y-\hat y\rvert$ | Laplace | **Mediana** condicional |
| **Pinball / quantile loss** | Laplace asimétrica | Cuantil condicional $x_p$ |
| **Log-loss / cross-entropy** | Bernoulli / Categórica | Probabilidad condicional de clase |
| **Huber** | Híbrida normal-Laplace | Media robusta |
| **Student-t NLL** | $t$ con $\nu$ grados de libertad | Media con **colas pesadas explícitas** |

Consecuencias directas para el proyecto:
- **Sobre MSE — matiz importante, la correspondencia va en un solo sentido.** *Si* se asume condicional normal homocedástica, *entonces* el MLE coincide con mínimos cuadrados. La recíproca **no** es cierta: usar MSE no obliga a asumir normalidad ni homocedasticidad. Conviene separar cuatro cuestiones:
  - **Consistencia.** El minimizador de $E[(Y-f(X))^2]$ sobre funciones medibles es $E[Y\mid X]$ **para cualquier distribución con segundo momento finito**. MSE es una regla de scoring propia para la media condicional (pertenece a la clase de divergencias de Bregman, que son exactamente las pérdidas que elicitan la media). **La heterocedasticidad no rompe esto**: MSE sigue apuntando al objeto correcto.
  - **Eficiencia.** Aquí sí hay pérdida. Bajo varianza condicional variable, MSE deja de ser eficiente; la ponderación por $1/\sigma_t^2$ es la que alcanza la cota. Salvedad práctica: $\hat\sigma_t$ se estima con error, y una ponderación mal estimada puede empeorar el resultado.
  - **Inferencia.** Éste es el daño real, y es el que Tsay señala: los errores estándar calculados bajo homocedasticidad son incorrectos. En la introducción del capítulo Tsay anuncia que el Cap. 2 provee "métodos para estimación consistente de la matriz de covarianza en presencia de heterocedasticidad condicional y correlaciones seriales" [A]. **La heterocedasticidad invalida la inferencia, no la estimación puntual.**
  - **Robustez ante colas pesadas.** Argumento independiente: la varianza del estimador basado en MSE depende del cuarto momento, de modo que en muestras finitas queda dominada por pocas observaciones. Es un problema de eficiencia y estabilidad, no de sesgo.

  **Conclusión corregida:** MSE **no queda invalidado** por lo que muestra Tsay. Lo que queda invalidado es (i) leerlo como un modelo completo de la distribución condicional, (ii) su eficiencia, y (iii) los errores estándar ingenuos. La elección de pérdida sigue siendo una decisión sustantiva, pero por estas razones y no porque MSE "asuma" algo falso.
- La tercera fila de la tabla es literalmente la Ec. 1.18 de Tsay. Un modelo de ML que emite $(\hat\mu_t, \hat\sigma_t)$ y se entrena con Gaussian NLL **es** un modelo de la distribución condicional en el sentido de Tsay, sólo que con una forma funcional no paramétrica en vez de ARMA-GARCH.
- Si se sospecha de colas pesadas, la NLL de Student-t es la traducción directa de esa creencia a la función objetivo.
- **La log-verosimilitud out-of-sample es una métrica de evaluación legítima y quizás superior a las habituales**: evalúa la distribución predictiva completa, no sólo un punto. En un problema donde la señal en la media es minúscula pero la señal en la varianza puede ser sustancial, evaluar sólo el error en la media descarta la mitad del fenómeno.

**[B] Qué comprobar.** Comparar, sobre datos reales del futuro objetivo y en evaluación out-of-sample estricta: (a) un modelo con $\sigma$ constante, (b) uno con $\sigma_t$ modelada, (c) uno con distribución condicional de colas pesadas. La comparación por log-verosimilitud out-of-sample dice cuál de las tres hipótesis distribucionales describe mejor los datos, **antes** de discutir arquitecturas.

## 2.7 Sección 1.2.5 — Empirical Properties of Returns

**[A] Datos y figuras.** Datos de CRSP (University of Chicago); **los pagos de dividendos, si los hay, están incluidos en los retornos**. Figura 1.2: retornos mensuales simples y log de IBM, ene-1926 a dic-2008. Figura 1.3: lo mismo para el índice value-weighted. **"Como es de esperar, los gráficos muestran que los patrones básicos de retornos simples y log son similares."**

**[A] Las seis observaciones sobre la Tabla 1.2** (textualmente, resumidas):
- **(a)** Los retornos **diarios** de índices y acciones individuales tienden a tener **excess kurtosis alta**. En series mensuales, los retornos de índices tienen mayor excess kurtosis que las acciones individuales.
- **(b)** La media de una serie de retornos **diarios es cercana a cero**, mientras que la de una serie mensual es algo mayor.
- **(c)** Los retornos mensuales tienen **mayores desviaciones estándar** que los diarios.
- **(d)** Entre los retornos diarios, los índices de mercado tienen **menores desviaciones estándar** que las acciones individuales. Tsay comenta que esto concuerda con el sentido común.
- **(e)** **La asimetría no es un problema serio** ni para retornos diarios ni mensuales.
- **(f)** **La diferencia entre retornos simples y log no es sustancial** (a nivel de estadísticos descriptivos).

**[A] Figura 1.4.** Densidades empíricas de retornos mensuales simples y log de IBM (1926–2008), superpuestas con la densidad normal usando la media y sd muestrales. **"Los gráficos indican que el supuesto de normalidad es cuestionable para los retornos mensuales de IBM. La densidad empírica tiene un pico más alto alrededor de su media, pero colas más gruesas que la normal correspondiente. En otras palabras, la densidad empírica es más alta y más delgada, pero con un soporte más ancho que la densidad normal correspondiente."**

**[A] Cifras destacadas de la Tabla 1.2** (retornos en %):

| Serie | Frecuencia | Media | Sd | Skew | Ex. kurt | Mín | Máx |
|---|---|---|---|---|---|---|---|
| S&P compuesto | diario simple | 0.029 | 1.056 | −0.73 | **22.81** | −20.47 | 11.58 |
| IBM | diario simple | 0.040 | 1.693 | 0.06 | 9.92 | −22.96 | 13.16 |
| Citigroup | diario simple | 0.067 | 2.602 | 1.80 | **55.25** | −26.41 | 57.82 |
| S&P compuesto | diario log | 0.023 | 1.062 | −1.17 | **30.20** | −22.90 | 10.96 |
| IBM | mensual simple | 1.35 | 7.15 | 0.44 | 3.43 | −26.19 | 47.06 |
| IBM | mensual log | 1.09 | 7.03 | −0.07 | 2.62 | −30.37 | 38.57 |

**[B] Lecturas para el proyecto — punto por punto.**

- **Sobre (a) y la kurtosis a distintas frecuencias — corrección de atribución.** [A] Tsay observa que los retornos diarios tienen excess kurtosis alta, y reporta valores menores en las series mensuales de la Tabla 1.2 (IBM: 9.92 diario vs 3.43 mensual, simples). **[B] Pero Tsay nunca enuncia una ley general de agregación → reducción de kurtosis**, y la comparación no es un experimento limpio: en la Tabla 1.2 la serie diaria de IBM empieza en 1970 (T = 9,845) y la mensual en 1926 (T = 996). **Son muestras y períodos distintos, no la misma serie agregada.**
- **[B] Qué se puede decir teóricamente, y bajo qué condiciones.** Para variables **iid con varianza finita**, el exceso de kurtosis de la suma de $k$ términos decae como $1/k$. Pero los retornos no son iid: con clustering de volatilidad el decaimiento es mucho más lento, y con memoria larga en volatilidad la kurtosis puede persistir. Y hay un contraejemplo dentro del propio capítulo: Tsay señala que **las distribuciones estables son estables bajo adición** [A] — agregar no adelgaza sus colas en absoluto.
- **Formulación correcta:** *la frecuencia de muestreo puede alterar sustancialmente la kurtosis observada, y en las series de Tsay la kurtosis es menor a frecuencia mensual que a frecuencia diaria; la magnitud y hasta el signo de ese efecto dependen de la estructura de dependencia y deben medirse en el instrumento objetivo, no suponerse.* Esto no es argumento para elegir baja frecuencia — la señal puede vivir en alta frecuencia — pero sí para saber qué se está aceptando.

- **Sobre (b) — el hallazgo más importante para ML.** Media diaria ≈ 0.04% con sd 1.69% (IBM). El ratio media/sd por observación es ~0.024. Combinado con el t-test explícito de Tsay (t = 1.51, p = 0.13 con T = 9,845), el mensaje es contundente: **con casi 40 años de datos diarios no se detecta ni siquiera el drift incondicional**. Cualquier ventaja predictiva realista será una fracción pequeña de esa magnitud. Consecuencias:
  - El número de observaciones necesarias para distinguir señal de ruido es enorme.
  - Modelos de alta capacidad sobre datasets modestos van a sobreajustar por construcción.
  - Las diferencias de performance entre dos modelos en un backtest de 1–2 años son, casi siempre, ruido.
  - Es un argumento fuerte para que el sistema apunte también al **segundo momento** (donde la señal es mucho más fuerte y detectable) y no exclusivamente a la dirección.

- **Sobre (c) — mensual tiene mayor sd que diario.** Es la agregación: bajo independencia, $\sigma_h\approx\sigma_1\sqrt{h}$. **[B]** Verificar empíricamente si la escala $\sqrt{h}$ se cumple en el futuro objetivo es un test informal, barato y muy informativo: desviaciones sistemáticas señalan autocorrelación o memoria larga (Cap. 2). Con la aditividad de log-retornos, este test es directo.

- **Sobre (d) — los índices son menos volátiles que sus componentes.** Diversificación. **[B]** Trasladado a futuros: un futuro sobre índice amplio (ES) tiene menor volatilidad que un futuro de commodity individual (CL). Cualquier comparación de performance o de calidad de señal entre instrumentos **debe normalizarse por volatilidad**, o se estará comparando magnitudes distintas de riesgo. Este es el fundamento estadístico de por qué el "point value" y la escala nominal no bastan para comparar.

- **Sobre (e) — la asimetría no es un problema serio.** Nótese que Tsay lo dice a la vez que reporta un test que **rechaza** simetría para IBM (p = 0.013) y skewness de −1.17 para el S&P diario log. La lectura correcta: la skewness es **estadísticamente detectable pero pequeña en magnitud** comparada con la magnitud del problema de kurtosis. **[B] Para trading la conclusión no se traslada tal cual**: una skewness moderada tiene efectos económicos grandes sobre drawdowns y sobre la asimetría de las colas de una estrategia. Lo que es "no serio" para caracterizar una distribución puede ser decisivo para dimensionar posiciones.

- **Sobre (f) — simple y log casi no difieren.** A nivel de estadísticos descriptivos, cierto. **[B]** Pero es una afirmación sobre *descripción*, no sobre *agregación*: la diferencia entre simple y log es sustancial precisamente cuando se acumulan muchos períodos o cuando los retornos son grandes, que es donde se decide el P&L compuesto. La observación (f) no debe leerse como "da lo mismo cuál usar".

**[B] Programa de análisis exploratorio recomendado para una serie de futuros** (qué medir y por qué, sin código):

| Qué medir | Por qué | Fundamento en Tsay |
|---|---|---|
| Time plot de precio y de retorno | Detectar rupturas, gaps de roll, datos corruptos, cambios visibles de régimen | Figs. 1.2, 1.3, 1.6 [A] |
| Media, sd, skew, excess kurtosis, min, max | Caracterizar la distribución marginal | Tabla 1.2, Ecs. 1.10–1.13 [A] |
| Cuantiles (0.1, 1, 5, 25, 50, 75, 95, 99, 99.9) | Describir colas sin depender de momentos que quizá no existan | Definición de cuantil en 1.2.1 [A] + [B] |
| t-test de media cero | Cuantificar la relación señal/ruido en el primer momento | Ejemplo 1.2 [A] |
| Jarque–Bera y tests de skew/kurtosis | Documentar el rechazo de normalidad (con la advertencia sobre iid) | 1.2.1 [A] |
| Densidad empírica vs normal ajustada | Ver la forma del desvío (pico + colas) | Fig. 1.4 [A] |
| **Los mismos estadísticos por sub-período** (año, régimen) | Detectar inestabilidad de los momentos | **[B]** — no en Tsay |
| **Los mismos estadísticos por frecuencia de muestreo** | Ver el efecto de agregación sobre kurtosis y sd | derivado de Tabla 1.2 [A] + [B] |
| **Kurtosis de retornos estandarizados por vol rezagada** | Separar colas pesadas "reales" de heterocedasticidad | **[B]** |
| **Fracción de barras con retorno = 0** y $\text{tick}/P$ | Medir el impacto de la discreción | 1.2.2 discreción [A] + [B] |
| **Estadísticos por hora del día / sesión** | Los futuros operan casi 24h con actividad muy heterogénea | **[B]** — no en Tsay |
| **Conteo y fechas de eventos extremos** (\|r\| > 5σ) | Verificar cuántos hay vs lo que predice la normal; identificar si se agrupan | 1.3 extremos [A] + [B] |

**[B] Sobre los ejercicios del capítulo.** Los Ejercicios 1.1, 1.2, 1.4 y 1.5 son plantillas metodológicas útiles y directamente reutilizables: computar descriptivos sobre simple y log en porcentajes, transformar entre representaciones, testear media cero, testear skewness y excess kurtosis por separado, y hacerlo sobre **varios instrumentos en paralelo** (tres acciones; cuatro tipos de cambio). El Ejercicio 1.5 sobre tipos de cambio es el más cercano al caso de futuros, porque los FX no pagan dividendos y su análisis descriptivo es directamente análogo. El Ejercicio 1.3 (valor de $1 invertido, sin costos de transacción) es útil sólo como recordatorio de la diferencia entre media aritmética y geométrica: **el capital compuesto depende del producto de gross returns, no de la suma de retornos** — un punto con consecuencias reales sobre la evaluación de estrategias.

## 2.8 Sección 1.3 — Processes Considered

**[A] Qué afirma Tsay.**
- Además de los retornos, el libro considera el **proceso de volatilidad** y el **comportamiento de los retornos extremos**.
- **"El proceso de volatilidad se ocupa de la evolución de la varianza condicional del retorno a lo largo del tiempo. Éste es un tema de interés porque, como se muestra en las Figuras 1.2 y 1.3, las variabilidades de los retornos varían en el tiempo y aparecen en clusters."**
- **"En aplicaciones, la volatilidad juega un rol importante en la valuación de opciones y en la gestión de riesgo."**
- **Extremos**: son los retornos grandes positivos o negativos. La Tabla 1.2 muestra que **mínimos y máximos pueden ser sustanciales**. **"Los retornos extremos negativos son importantes en gestión de riesgo, mientras que los extremos positivos son críticos para mantener una posición corta."** Se estudian en el Cap. 7: frecuencia de ocurrencia, tamaño del extremo, e impacto de variables económicas sobre los extremos.
- Otras series consideradas en el libro: tasas de interés, tipos de cambio, rendimientos de bonos, y ganancias trimestrales por acción.
- Fig. 1.5: tasas de Tesoro a 10 y 1 año, abril 1953–febrero 2009. **"Como es de esperar, las dos tasas se movieron al unísono, pero la de 1 año parece más volátil."**
- Fig. 1.6: USD/JPY diario, ene-2000 a mar-2009. **"El tipo de cambio encontró cambios grandes ocasionales en el período muestral."**
- **Tabla 1.3:** para las tasas de interés, las **medias muestrales son proporcionales al plazo, pero las desviaciones estándar son inversamente proporcionales al plazo**. Para los retornos de bonos, **las desviaciones estándar están positivamente relacionadas con el plazo, mientras que las medias permanecen estables para todos los plazos**. **"La mayoría de las series consideradas tienen excess kurtosis positiva."**
- Hoja de ruta: Caps. 2–4 se centran en los primeros cuatro momentos; Cap. 7 en el comportamiento de mínimos y máximos; Caps. 8 y 10 en momentos y relaciones entre múltiples activos; Cap. 5 en las propiedades cuando el intervalo temporal es pequeño; Cap. 6 introduce finanzas matemáticas.

**[B] Por qué esta sección corta es desproporcionadamente importante.** Es donde el capítulo pasa de "los retornos tienen esta distribución" a "**esa distribución no es la misma en todo momento**". El clustering de volatilidad es la primera evidencia, dentro del capítulo, de que existe **dependencia temporal genuina** aunque la media no sea predecible. Es decir: la hipótesis de random walk de 1.2.2, en su versión fuerte (independencia), **está empíricamente rechazada** por las propias figuras del capítulo. Lo que puede sobrevivir es la versión débil (no predecibilidad de la media).

**[B] Traslado a futuros — Tabla 1.3 como recordatorio de que la escala no es comparable entre instrumentos.** El hallazgo de que la sd de las tasas es *inversamente* proporcional al plazo mientras que la sd de los retornos de bonos es *directamente* proporcional al plazo es un ejemplo concreto y verificable de que **instrumentos del mismo mercado tienen escalas de riesgo completamente distintas**. En futuros: ZT, ZF, ZN y ZB (2, 5, 10, 30 años) tienen duraciones y volatilidades en puntos radicalmente distintas. Compararlos, apilarlos o combinarlos sin normalización adecuada es un error de medición, no de modelado. Y el "point value" no resuelve esto: normaliza el dinero por punto, no el riesgo por contrato.

**[B] Preguntas de investigación que se desprenden de 1.3.** Se dejan explícitamente como **hipótesis**, no como decisiones de diseño:

- **H-Vol-1.** ¿La dinámica del mercado cambia según el nivel de volatilidad? Es decir, ¿la relación $X_t\to y_{t+h}$ es distinta en regímenes de alta y baja volatilidad? Si lo es, un modelo único entrenado sobre todo el histórico está promediando dos funciones distintas.
- **H-Vol-2.** ¿La volatilidad contiene información predictiva sobre la **dirección**, o sólo sobre la **magnitud**? Son preguntas separables y la respuesta a la segunda es casi seguramente sí (clustering), mientras que la primera está abierta.
- **H-Vol-3.** ¿Conviene modelar dirección y magnitud por separado ($\text{sign}$ y $|r|$, o $\mu_t$ y $\sigma_t$)? La estructura de la Ec. 1.18 sugiere que son componentes distintos del mismo objeto; nada obliga a que el mismo modelo, con la misma capacidad, deba estimar los dos.
- **H-Vol-4.** ¿Los movimientos extremos deben recibir tratamiento especial — modelo aparte, target aparte, exclusión, winsorización, o una distribución de colas pesadas explícita? Tsay dedica un capítulo entero al tema, lo que sugiere que la respuesta por defecto ("son outliers, se recortan") es inadecuada.
- **H-Vol-5.** ¿La distribución objetivo cambia con el tiempo (más allá de la varianza)? ¿Cambian también la asimetría y la kurtosis condicionales? Esto conecta con Markov switching (Cap. 12).
- **H-Vol-6.** ¿La volatilidad es un buen **normalizador** del target? Es decir, ¿es $r_{t+h}/\hat\sigma_t$ un objetivo más aprendible que $r_{t+h}$, por tener una distribución más estable? **[B]** Es una hipótesis atractiva y comprobable, no una conclusión.
- **H-Vol-7.** ¿La volatilidad es un buen **condicionante** para la asignación de capital, independientemente de si mejora la predicción? Aun sin señal direccional, ajustar el tamaño por volatilidad estabiliza el riesgo del portafolio. Esto es risk management, no predicción, y no debe confundirse con predicción.

---

# 3. Conceptos matemáticos esenciales

Sólo las ecuaciones necesarias para trabajar correctamente con el capítulo.

**Retornos.**
$$1+R_t=\frac{P_t}{P_{t-1}},\qquad R_t=\frac{P_t-P_{t-1}}{P_{t-1}},\qquad r_t=\ln(1+R_t)=p_t-p_{t-1}$$
$$r_t=\ln(1+R_t),\qquad R_t=e^{r_t}-1$$

**Agregación temporal (la propiedad central).**
$$1+R_t[k]=\prod_{j=0}^{k-1}(1+R_{t-j}),\qquad\qquad \boxed{\;r_t[k]=\sum_{j=0}^{k-1} r_{t-j}\;}$$

**Agregación de cartera.**
$$R_{p,t}=\sum_{i=1}^{N}w_i R_{it}\quad\text{(exacto)},\qquad r_{p,t}\approx\sum_{i=1}^{N}w_i r_{it}\quad\text{(sólo si los }R_{it}\text{ son pequeños)}$$

**Excess return.** $Z_t=R_t-R_{0t}$, $z_t=r_t-r_{0t}$.

**Momentos y su estimación.**
$$S(x)=E\!\left[\frac{(X-\mu_x)^3}{\sigma_x^3}\right],\qquad K(x)=E\!\left[\frac{(X-\mu_x)^4}{\sigma_x^4}\right],\qquad \text{excess kurtosis}=K(x)-3$$
$$\hat\mu_x=\frac1T\sum x_t,\quad \hat\sigma_x^2=\frac{1}{T-1}\sum(x_t-\hat\mu_x)^2,\quad \hat S=\frac{\sum(x_t-\hat\mu_x)^3}{(T-1)\hat\sigma_x^3},\quad \hat K=\frac{\sum(x_t-\hat\mu_x)^4}{(T-1)\hat\sigma_x^4}$$

**Tests de normalidad** (bajo $H_0$ de normalidad e **iid**):
$$\hat S\sim N(0,6/T),\qquad \hat K-3\sim N(0,24/T),\qquad JB=\frac{\hat S^2}{6/T}+\frac{(\hat K-3)^2}{24/T}\sim\chi^2_2$$

**Factorización condicional — la ecuación que define el problema de ML.**
$$F(r_1,\dots,r_T;\theta)=F(r_1)\prod_{t=2}^{T}F(r_t\mid r_{t-1},\dots,r_1)$$

**Hipótesis de random walk (versión de Tsay).**
$$F(r_t\mid r_{t-1},\dots,r_1)=F(r_t)\ \Longrightarrow\ \text{independencia temporal}\ \Longrightarrow\ \text{no predecibilidad}$$

**Log-verosimilitud con condicional normal (Ec. 1.18).**
$$\ln f(r_1,\dots,r_T;\theta)=\ln f(r_1;\theta)-\frac12\sum_{t=2}^{T}\left[\ln(2\pi)+\ln\sigma_t^2+\frac{(r_t-\mu_t)^2}{\sigma_t^2}\right]$$

**Lognormal — conversión de momentos.**
$$E(R_t)=e^{\mu+\sigma^2/2}-1,\qquad \text{Var}(R_t)=e^{2\mu+\sigma^2}\left(e^{\sigma^2}-1\right)$$

**Mixtura finita de normales.**
$$r_t\sim(1-X)\,N(\mu,\sigma_1^2)+X\,N(\mu,\sigma_2^2),\qquad X\sim\text{Bernoulli}(\alpha),\ \sigma_2^2\gg\sigma_1^2$$

**Caso multivariado.**
$$\boldsymbol\mu_x=E(\mathbf X),\qquad \Sigma_x=E[(\mathbf X-\boldsymbol\mu_x)(\mathbf X-\boldsymbol\mu_x)'],\qquad F(\mathbf r_t\mid \mathbf r_{t-1},\dots,\mathbf r_1,\theta)$$
(estimadores muestrales consistentes **sólo si $\Sigma_x$ existe**).

---

# 4. Implicancias para mercados de futuros

## 4.1 Lo que Tsay concluye (general, sin futuros) — [A]

1. Se trabaja con retornos porque son libres de escala y estadísticamente más manejables.
2. Log-retornos: aditivos en el tiempo y estadísticamente más tratables. Retornos simples: aditivos en cartera.
3. La distribución condicional es más relevante que la marginal; la marginal es más fácil de estimar y a veces está cerca de la condicional (correlación serial débil).
4. La normalidad es rechazada empíricamente: excess kurtosis positiva generalizada, densidad con pico alto y colas gruesas.
5. La media de retornos diarios es cercana a cero; la asimetría no es un problema serio; simple y log casi no difieren en descriptivos.
6. En las series de la Tabla 1.2, los retornos mensuales tienen mayor desviación estándar que los diarios; y los valores de excess kurtosis reportados a frecuencia mensual son menores que los diarios (**Tsay no enuncia esto como una ley general de agregación**, y las series diarias y mensuales de la tabla cubren períodos distintos — ver matiz en 2.7).
7. Los índices son menos volátiles que los activos individuales.
8. La varianza no es constante en el tiempo y aparece en clusters ⇒ hay un proceso de volatilidad que modelar.
9. Los extremos son sustanciales y merecen estudio propio (Cap. 7).
10. Distintas series financieras (tasas, bonos, FX) tienen escalas de media y varianza estructuralmente distintas y casi todas exhiben excess kurtosis positiva.
11. La discreción de precios (tick size) invalida el supuesto de continuidad en alta frecuencia.
12. En el caso multivariado, el objeto central es la evolución de la media y la covarianza **condicionales**.

## 4.2 Adaptación a futuros — [B], enteramente propia

**Definición del retorno.**
- Elegir el tratamiento del contrato continuo según qué cantidad deba preservarse exactamente: **ratio** preserva retornos (equivalente a intra-contrato encadenado), **aditivo** preserva diferencias en puntos, ninguno preserva ambas. No usar ajuste aditivo para calcular retornos.
- Tratar la barra del roll explícitamente (excluirla o usar el contrato que se mantiene). Un salto de roll no es un retorno.
- Si restar o no la tasa libre de riesgo depende de qué serie se construyó (variación del precio del futuro vs índice de retorno total colateralizado). No resoluble con el Cap. 1 → **PREGUNTA ABIERTA** (Cap. 6 y literatura de futuros).
- No mezclar retornos de futuros con retornos totales de índices cash (que incluyen dividendos, como los datos CRSP de Tsay) sin homogeneizar.

**Comparabilidad entre instrumentos.**
- El retorno elimina el nivel nominal de precio, pero **no la volatilidad**. Dos futuros con retorno de 1% no representan el mismo evento estadístico si uno tiene sd diaria de 0.5% y el otro de 2.5%.
- Normalizar por volatilidad ($r/\hat\sigma$) es el candidato natural para hacer comparables instrumentos, y es imprescindible bajo el Enfoque A. **Debe validarse, no asumirse.**
- Micro / E-mini: mismo subyacente, distinto point value, y **no comparten P&L, riesgo ni granularidad de posición**. Que compartan features y modelo es una **hipótesis a testear**, plausible a baja frecuencia con features de precio, dudosa a alta frecuencia o con features de volumen/microestructura, y limitada por la historia más corta de los Micro.
- Comparar performance entre instrumentos exige normalizar por riesgo, no por dinero.

**Tick size y discreción.**
- Medir $\text{tick}/P$ por instrumento y época: define el retorno mínimo observable y, por tanto, el umbral por debajo del cual "predicción" carece de sentido operativo.
- En barras cortas, el retorno es efectivamente discreto; los momentos y tests de normalidad tienen otra interpretación.
- El bid–ask bounce genera autocorrelación negativa que **no es señal**. [A] Tsay lo anticipa para acciones en el Cap. 5.

**Estructura de la serie.**
- Los futuros operan casi 24 horas con actividad extremadamente heterogénea. Los momentos calculados sobre "todo el histórico" mezclan sesiones con propiedades distintas.
- El calendario importa: vencimientos, roll, datos macro, subastas de Tesoro, expiración de opciones. Nada de esto está en Tsay y todo puede generar estructura espuria o real.

**Multivariado.**
- Los índices accionarios están muy correlacionados entre sí, lo que genera riesgo de colinealidad y sobreestimación del tamaño de muestra efectivo. Que su aporte predictivo incremental sea nulo, y que la información esté en los spreads, son **hipótesis a medir**, no conclusiones.
- Las correlaciones cross-market son procesos, no constantes (Cap. 10).
- Los lead-lag aparentes pueden ser artefactos de sincronización y liquidez.
- Con colas pesadas, las covarianzas estimadas son inestables; esto degrada cualquier arquitectura que las use como feature.

---

# 5. Implicancias para Machine Learning

## 5.1 Datos / Representación

| # | Implicancia | Origen |
|---|---|---|
| D1 | Modelar retornos, no precios. Ningún feature debe depender del nivel absoluto salvo por diseño explícito. | [A] 1.1 |
| D2 | Log-retornos para modelado (aditividad temporal); retornos simples o dinero para contabilidad de P&L. | [A] 1.1 + [B] |
| D3 | Transformación del contrato continuo elegida según la cantidad a preservar (ratio→retornos, aditivo→puntos); roll tratado explícitamente. | [B] |
| D4 | La frecuencia de muestreo es un hiperparámetro: cambia kurtosis, sd, discreción y relación señal/ruido. | [A] Tabla 1.2 + [B] |
| D5 | Con múltiples instrumentos, normalizar por volatilidad antes de mezclar. | [B] apoyado en [A] 1.2.5(d) y Tabla 1.3 |
| D6 | La discreción de precios acota la resolución del target en alta frecuencia. | [A] 1.2.2 |
| D7 | Las escalas nominales y el point value no hacen comparables dos instrumentos; el riesgo sí. | [B] |

## 5.2 Feature engineering

| # | Implicancia | Origen |
|---|---|---|
| F1 | El vector de estado $Y$ de la Ec. 1.14 es el lugar formal de los features exógenos: otros mercados, macro, calendario. Usar información externa está dentro del marco. | [A] 1.2.2 |
| F2 | Todo feature debe ser función de información disponible en $t$. La estructura condicional lo exige. | [A] 1.2.2 estructura + [B] explícito |
| F3 | Features de volatilidad son candidatos naturales de primer orden: la varianza es la dimensión donde el capítulo documenta dependencia real. | [A] 1.3 |
| F4 | Estandarización: el z-score presupone normalidad. Con leptocurtosis, considerar rank, winsorización o normalización por volatilidad. | [B] a partir de [A] 1.2.1/1.2.5 |
| F5 | Features de otros futuros sólo con rezago estricto, y previa verificación de que el lead-lag no es artefacto de sincronización. | [B] (Cap. 8 abierto) |
| F6 | Features basados en correlación rodante están midiendo un objeto que varía en el tiempo (covarianza condicional). Tratarlos como tales. | [A] 1.2.3 + [B] |

## 5.3 Targets

| # | Implicancia | Origen |
|---|---|---|
| T1 | Un target a horizonte $h$ es $\sum_{j=1}^{h}r_{t+j}$ con log-retornos: exacto y comparable entre horizontes. | [A] 1.1 |
| T2 | Targets con ventanas solapadas violan independencia entre observaciones; el tamaño de muestra efectivo cae y las métricas se inflan. | [B] |
| T3 | Bajo colas pesadas, un target de retorno crudo tiene su varianza dominada por pocos eventos. Considerar targets normalizados por volatilidad. | [B] a partir de [A] |
| T4 | Direccion y magnitud son objetivos separables: $\text{sign}(r)$ y $|r|$, o $\mu_t$ y $\sigma_t$. La Ec. 1.18 los trata como componentes distintos. | [A] 1.2.4 + [B] |
| T5 | Umbrales de clasificación definidos en múltiplos de σ están mal calibrados si σ es no constante; deben ser condicionales. | [B] a partir de [A] 1.3 |
| T6 | El drift diario es indistinguible de cero: no esperar targets con señal fuerte en la media. | [A] Ejemplo 1.2 |

## 5.4 Modelado

| # | Implicancia | Origen |
|---|---|---|
| M1 | Toda pérdida tiene una lectura probabilística (MSE ↔ normal homocedástica), pero la implicación va en un solo sentido: MSE sigue siendo **consistente** para la media condicional bajo heterocedasticidad y no normalidad. Lo que se pierde es eficiencia e inferencia válida, no el objetivo. Ver matiz en 2.6. | [A] 1.2.4 + [B] |
| M2 | Gaussian NLL con $\mu_t$ y $\sigma_t$ es la Ec. 1.18 exactamente: un modelo que emite media y varianza es un modelo de la distribución condicional. | [A] 1.2.4 |
| M3 | Con colas pesadas, considerar pérdidas robustas (Huber) o NLL de Student-t; comparar empíricamente. | [B] |
| M4 | Predecir cuantiles condicionales es una alternativa a predecir la media que no requiere supuesto paramétrico. | [B] a partir de [A] 1.2.1 |
| M5 | Baja relación señal/ruido ⇒ restringir capacidad, priorizar regularización, desconfiar de mejoras grandes. | [A] Ejemplo 1.2 + [B] |
| M6 | La hipótesis nula del proyecto es el random walk: condicional = marginal. El benchmark obligatorio es un modelo incondicional. | [A] 1.2.2 |
| M7 | Antes de elegir arquitectura, decidir qué componente de $F(r_{t+h}\mid X_t)$ se está modelando. | [A] 1.2.2/1.2.4 |

## 5.5 Validación

| # | Implicancia | Origen |
|---|---|---|
| V1 | La factorización 1.15 impone evaluación secuencial: walk-forward / expanding window. La CV aleatoria no corresponde a ninguna cantidad bien definida aquí. | [A] 1.2.2 estructura + [B] conclusión |
| V2 | Con targets solapados, purgar y aplicar embargo entre train y test. | [B] |
| V3 | Bajo el Enfoque A, particionar por **tiempo**, nunca por instrumento (instrumentos correlacionados ⇒ leak). | [B] |
| V4 | Métricas y tests basados en normalidad subestiman la incertidumbre; usar bootstrap por bloques o inferencia robusta. | [B] a partir de [A] 1.2.1/1.2.5 |
| V5 | Log-verosimilitud out-of-sample como métrica: evalúa la distribución predictiva completa, no un punto. | [A] 1.2.4 + [B] |
| V6 | Corregir por multiplicidad: probar muchos features/modelos/horizontes garantiza falsos positivos. | [B] |
| V7 | Test de sanidad: correr el pipeline sobre retornos permutados y sobre random walks simulados con misma vol y kurtosis. | [B] |
| V8 | Evaluar por régimen de volatilidad, no sólo en agregado: un modelo puede tener performance agregada positiva y ser destructivo en el régimen que concentra el riesgo. | [B] a partir de [A] 1.3 |

## 5.6 Risk management

| # | Implicancia | Origen |
|---|---|---|
| R1 | Excess kurtosis 10–50 ⇒ los eventos de cola son mucho más frecuentes de lo que sugiere la normal. El dimensionamiento debe reflejarlo. | [A] Tabla 1.2 |
| R2 | VaR e intervalos gaussianos subestiman la pérdida de cola. (Cap. 7 lo desarrolla.) | [A] 1.3 + [B] |
| R3 | Los extremos negativos importan para el riesgo y los positivos para las posiciones cortas: el riesgo es bilateral. | [A] 1.3 |
| R4 | Volatility targeting es una decisión de riesgo válida aunque no exista señal direccional. No confundirla con predicción. | [B] |
| R5 | Los momentos estimados sobre ventanas cortas con colas pesadas son inestables; los límites de riesgo no deben depender de un único estimador. | [A] 1.2.3 (existencia de momentos) + [B] |
| R6 | El P&L compuesto depende del producto de gross returns, no de la suma de retornos: una serie de retornos con media positiva puede tener capital final decreciente. | [A] agregación multiplicativa + Ej. 1.3 |

## 5.7 Análisis exploratorio

Ver la tabla de la sección 2.7 ("Programa de análisis exploratorio recomendado"). Principio rector **[B]**: el EDA no es para "conocer los datos" en abstracto, sino para **medir cada supuesto que el modelo va a hacer implícitamente**, antes de que lo haga.

---

# 6. Hipótesis a comprobar empíricamente

Lista concreta y testeable sobre datos reales de futuros. Cada una es falsable.

**Sobre representación**
1. ¿La serie de retornos calculada intra-contrato difiere materialmente de la calculada sobre la serie back-adjusted? ¿Cuánto se acumula la discrepancia?
2. ¿$\text{Var}(r_t[h])\approx h\cdot\text{Var}(r_t)$? ¿A partir de qué $h$ se rompe? (Desviación ⇒ autocorrelación o memoria; Cap. 2.)
3. ¿Cuál es la magnitud típica de $|r_t - R_t|$, y se concentra en los días de alta volatilidad?
4. ¿Qué fracción de barras tiene retorno exactamente cero, y cómo cambia con la frecuencia? ¿Cuál es el cociente $\text{tick}/P$?

**Sobre distribución**
5. ¿Cuáles son media, sd, skewness, excess kurtosis, min, max y cuantiles del instrumento objetivo, por frecuencia y por sub-período?
6. ¿Se rechaza normalidad (JB)? ¿Con qué magnitud? (Se espera que sí, contundentemente.)
7. ¿Se rechaza que la media sea cero? ¿Con qué t-stat y qué IC? (Réplica directa del Ejemplo 1.2.)
8. ¿La excess kurtosis cae drásticamente al excluir el 0.1% más extremo? (Si sí, la estimación no es un parámetro estable.)
9. **¿Qué fracción de la excess kurtosis desaparece al estandarizar por volatilidad realizada rezagada?** (Separa colas condicionales de heterocedasticidad. Pregunta central.)
10. ¿Los momentos son estables entre sub-períodos, o cambian sistemáticamente?
11. ¿La asimetría es distinta en cola izquierda vs derecha?

**Sobre predecibilidad**
12. ¿Hay autocorrelación significativa en los retornos del instrumento objetivo, a distintas frecuencias? (Cap. 2.)
13. ¿Hay autocorrelación en $|r_t|$ o $r_t^2$? (Debería haberla: clustering.) ¿Cuánto persiste?
14. ¿Un modelo condicional supera al benchmark incondicional en log-verosimilitud out-of-sample? Separadamente para media y para varianza.
15. ¿La performance sobrevive cuando el pipeline se aplica a retornos permutados y a random walks simulados? (Debe colapsar a cero.)
16. ¿El edge estimado por trade supera el spread + comisión + slippage típicos del instrumento?

**Sobre regímenes y volatilidad**
17. ¿La relación $X_t\to y_{t+h}$ difiere entre regímenes de alta y baja volatilidad? (Entrenar/evaluar por régimen.)
18. ¿La volatilidad rezagada predice la magnitud del movimiento futuro? ¿Y su dirección?
19. ¿$r_{t+h}/\hat\sigma_t$ es un target con distribución más estable y más aprendible que $r_{t+h}$?
20. ¿Los eventos extremos se agrupan en el tiempo? ¿Son precedidos por señales observables?

**Sobre múltiples instrumentos**
21. ¿Cuál es la matriz de correlación entre los futuros candidatos, y cuántos factores explican la mayor parte de la varianza?
22. ¿La correlación entre pares clave es estable en ventanas rodantes, o cambia de signo?
23. ¿Existe lead-lag rezagado real entre mercados, o desaparece al controlar por horarios de sesión y liquidez?
24. ¿Un modelo entrenado en el instrumento A predice el instrumento B mejor que el azar? (Test directo de la premisa del Enfoque A.)
25. ¿Añadir cada mercado adicional mejora la performance out-of-sample, corrigiendo por multiplicidad de pruebas?
26. ¿Cuál es el tamaño de muestra **efectivo** de un dataset apilado de $N$ futuros correlacionados?

---

# 7. Errores metodológicos a evitar

Se marca cuál proviene de una advertencia de Tsay y cuál es extensión propia.

**Derivados de advertencias explícitas de Tsay**

1. **Asumir normalidad.** [A] Rechazada empíricamente y de forma masiva. Afecta pérdidas, métricas, intervalos, VaR y calibración de probabilidades.
2. **Ignorar colas pesadas.** [A] Excess kurtosis de 10 a 55 en series diarias. Los eventos de cola son el fenómeno, no ruido a limpiar.
3. **Confundir agregación temporal con agregación de cartera.** [A] Log aditivo en el tiempo, simple aditivo en cartera. Mezclarlas produce errores de contabilidad reales.
4. **Usar la aproximación $r\approx R$ sin verificar.** [A] Tsay advierte que la precisión de la aproximación de primer orden puede no ser suficiente.
5. **Trabajar con marginales cuando el objeto relevante es la condicional.** [A] "Las distribuciones condicionales son más relevantes que las marginales."
6. **Suponer varianza constante.** [A] Las variabilidades varían en el tiempo y aparecen en clusters.
7. **Tratar los retornos de alta frecuencia como continuos.** [A] La discreción del tick lo invalida.
8. **Estimar covarianzas sin verificar existencia de momentos.** [A] La consistencia de los estimadores requiere que $\Sigma$ exista.

**Extensiones propias — no tratadas por Tsay en este capítulo** [B]

9. **Confundir autocorrelación con predecibilidad económica.** Autocorrelación ≠ 0 no implica edge explotable; autocorrelación = 0 no implica ausencia de dependencia.
10. **Confundir dependencia con causalidad.** Que NQ y ES co-varíen no dice qué mueve a qué; ambos pueden responder a un factor común.
11. **Ignorar los costos de transacción.** Tsay los excluye explícitamente (Ej. 1.3: "assume no transaction costs"). Un edge menor que el tick no existe operativamente.
12. **Explotar autocorrelación de microestructura.** El bid–ask bounce produce autocorrelación negativa "predecible" e inejecutable. Fuente principal de falsos positivos en alta frecuencia.
13. **Usar información futura por la vía del roll adjustment — con precisión.** El factor de ajuste de cada segmento depende de rolls posteriores, pero es constante dentro del segmento: los features **invariantes a escala** no sufren leak; los **dependientes del nivel** sí. Riesgo aparte: definir la regla de roll ex post.
14. **Usar información futura por la vía de la normalización.** Estandarizar todo el dataset con media y sd calculadas sobre el conjunto completo filtra información de test hacia train. La normalización debe ser causal (rolling).
15. **Usar CV aleatoria en series temporales.** Rompe la factorización condicional de la Ec. 1.15.
16. **Targets solapados sin purga ni embargo.** Infla la significancia por un factor cercano al horizonte de solapamiento.
17. **Comparar instrumentos sin normalizar por riesgo.** El punto no vale lo mismo entre contratos, ni la volatilidad es comparable entre mercados.
18. **Data snooping / multiplicidad.** Probar muchas combinaciones garantiza hallazgos espurios. Requiere corrección o registro previo de hipótesis.
19. **Selección de instrumentos o de período ex post.** Elegir el futuro y las fechas donde "funciona" es sobreajuste a nivel de diseño experimental.
20. **Asumir estacionariedad sin verificarla.** El Capítulo 1 **no** define estacionariedad (llega en el Cap. 2) y sin embargo todo lo demás la presupone implícitamente. Es la omisión más importante a tener presente.
21. **Confundir un buen ajuste distribucional con un buen sistema de trading.** Modelar bien $F(r_{t+h}\mid X_t)$ es necesario y no suficiente: falta la traducción a posiciones, costos y riesgo.
22. **Interpretar la kurtosis muestral como un parámetro estable.** Si el cuarto momento no existe, el estimador no converge.
23. **Aplicar JB o tests de momentos a datos con clustering de volatilidad como si fueran iid.** Los p-valores están distorsionados; el test no distingue no-normalidad marginal de heterocedasticidad con condicional normal.

---

# 8. Checklist de conocimientos adquiridos

Antes de pasar al Capítulo 2, deberían estar comprendidos:

**Retornos**
- [ ] Diferencia exacta entre gross return, net return simple y log-retorno, y las conversiones entre ellos.
- [ ] Por qué el log-retorno es aditivo en el tiempo y el simple es aditivo en cartera, y por qué ninguno es ambas cosas.
- [ ] Retorno multiperiodo, retorno anualizado (media geométrica) y la aproximación de Taylor con su límite de validez.
- [ ] Composición continua y la relación $A=Ce^{rn}$.
- [ ] Ajuste por dividendos y definición de excess return; su interpretación como cartera de arbitraje.
- [ ] Por qué se prefieren retornos a precios, y qué se pierde al hacerlo.

**Distribuciones**
- [ ] Distribución conjunta, marginal y condicional; la identidad $f_{x,y}=f_{x|y}f_y$.
- [ ] Definición de independencia vía condicional = marginal.
- [ ] Definición de cuantil.
- [ ] Media, varianza, skewness, kurtosis y excess kurtosis; leptocúrtica vs platicúrtica.
- [ ] Estimadores muestrales de los cuatro momentos y la condición de existencia de los momentos.
- [ ] Tests de skewness, de excess kurtosis y Jarque–Bera, con sus supuestos (normalidad **e iid**).
- [ ] Las cuatro familias candidatas y el trade-off de cada una: normal, lognormal, estable, mixtura de normales.
- [ ] Por qué las estables no normales tienen varianza infinita y por qué eso es un problema.
- [ ] Por qué una mixtura de escala genera colas pesadas y qué relación conceptual tiene con la volatilidad variable.

**Estructura del problema**
- [ ] La Ec. 1.14 y el rol del vector de estado $Y$.
- [ ] La factorización de la Ec. 1.15 y por qué la condicional es lo relevante.
- [ ] La hipótesis de random walk como igualdad entre condicional y marginal, y su consecuencia (no predecibilidad).
- [ ] La jerarquía: autocorrelación ⊂ dependencia ⊂ predecibilidad económica ⊂ predecibilidad neta de costos.
- [ ] Verosimilitud, log-verosimilitud, MLE, y el rol de $f(r_1;\theta)$.
- [ ] La correspondencia entre log-verosimilitud y funciones de pérdida.

**Hechos empíricos**
- [ ] Excess kurtosis alta y generalizada; media diaria ≈ 0; agregación reduce kurtosis; índices menos volátiles que componentes.
- [ ] Densidad empírica: pico más alto, colas más gruesas, soporte más ancho que la normal.
- [ ] Clustering de volatilidad y su significado (varianza condicional variable).
- [ ] Magnitud sustancial de mínimos y máximos, y su relevancia asimétrica para posiciones largas y cortas.
- [ ] Discreción de precios por tick size y su relevancia en alta frecuencia.

**Multivariado (nivel introductorio)**
- [ ] Vector de retornos, vector de medias, matriz de covarianza y sus estimadores.
- [ ] Que el objeto central es la **covarianza condicional**, y que su dinámica es materia de los Caps. 8 y 10.
- [ ] La distinción entre análisis cross-sectional (CAPM, un $t$) y análisis dinámico (un activo, muchos $t$).

**Lo que NO se sabe todavía y no debe darse por sabido**
- [ ] Estacionariedad y ACF → Cap. 2.
- [ ] Modelos de volatilidad condicional → Cap. 3.
- [ ] No linealidad y tests de no linealidad → Cap. 4.
- [ ] Microestructura, bid–ask bounce, trading no sincrónico → Cap. 5.
- [ ] Teoría de valores extremos y VaR → Cap. 7.
- [ ] Lead-lag y cointegración → Cap. 8.
- [ ] Reducción de dimensión y modelos de factores → Cap. 9.
- [ ] Covarianzas condicionales multivariadas → Cap. 10.
- [ ] State-space y Markov switching → Caps. 11–12.

---

# 9. Conclusiones

**Cuáles son los aportes realmente fundamentales del Capítulo 1 para construir un sistema serio de ML sobre futuros.**

**1. Define la variable correcta.** Modelar retornos y no precios es la decisión de representación que hace posible cualquier generalización fuera de muestra. Y la dualidad log/simple —aditividad temporal vs aditividad de cartera— determina qué se puede sumar sin error en cada etapa del pipeline. Es un aporte pequeño en apariencia y de consecuencias grandes en la práctica, especialmente en futuros, donde el roll y la escala nominal hacen que trabajar con precios sea directamente inválido.

**2. Define el problema.** La factorización $F(r_1,\dots,r_T)=F(r_1)\prod F(r_t\mid \text{pasado})$ es, literalmente, el problema que un sistema de ML intenta resolver. Y la definición de random walk como "condicional = marginal" provee la hipótesis nula. Sin esa nula explícita, un proyecto de ML financiero no tiene contra qué comparar y termina evaluando modelos contra otros modelos en vez de contra la posibilidad de que no haya nada que aprender. **El aporte no es una técnica: es la formulación correcta de la pregunta.**

**3. Establece la escala del desafío.** La media de un retorno diario es estadísticamente indistinguible de cero con ~10⁴ observaciones. Esto no es un detalle: es la restricción que condiciona el tamaño de dataset necesario, la capacidad admisible del modelo, la severidad del control de sobreajuste y el escepticismo con que debe recibirse cualquier resultado positivo. Un proyecto que no internaliza este número tomará decisiones de diseño demasiado optimistas en todas las etapas.

**4. Destruye el supuesto de normalidad con evidencia, no con retórica.** Y al hacerlo desactiva la elección por defecto de casi todas las herramientas estándar: MSE, z-scores, intervalos gaussianos, tests t, Sharpe con inferencia normal. Cada una de esas herramientas es utilizable, pero ninguna es neutral: cada una es una hipótesis distribucional disfrazada de convención técnica.

**5. Convierte la elección de función de pérdida en una decisión estadística explícita.** Vía la verosimilitud, "qué pérdida uso" y "qué distribución condicional asumo" son la misma pregunta. Ésta es probablemente la conexión más útil del capítulo para un practicante de ML, porque transforma un hiperparámetro que se suele elegir por costumbre en una hipótesis modelable y testeable.

**6. Muestra dónde vive la dependencia real.** El clustering de volatilidad es la evidencia, dentro del propio capítulo, de que la versión fuerte del random walk es falsa. La media puede no ser predecible; la varianza claramente lo es. Esto abre la posibilidad —que el proyecto debe tratar como hipótesis y no como decisión— de que el sistema deba modelar dirección y magnitud como problemas separados, y de que buena parte del valor esté en el segundo momento aunque el instinto lleve al primero.

**7. Anticipa correctamente el problema multivariado sin resolverlo.** La advertencia de que los estimadores de covarianza son consistentes sólo si los momentos existen, combinada con la evidencia de colas pesadas, deja planteado que toda arquitectura cross-market descansa sobre estimaciones frágiles. Y la remisión a los Caps. 8, 9 y 10 indica que el camino correcto pasa por reducción de dimensión y covarianza condicional, no por apilar mercados.

**Qué NO aporta el capítulo, y conviene no fingir que aporta.** No dice nada sobre estacionariedad, autocorrelación, modelos, validación, costos, roll, apalancamiento ni sobreajuste. La mayoría de los riesgos metodológicos que hunden proyectos de ML financiero —look-ahead bias, data snooping, targets solapados, CV mal diseñada— son extensiones que hay que traer de otra parte. El Capítulo 1 provee el vocabulario y los hechos estilizados; la metodología experimental es responsabilidad del proyecto.

**Aporte neto.** El Capítulo 1 no enseña a predecir nada. Enseña qué se está prediciendo, contra qué nula, con qué distribución, en qué escala y con qué relación señal/ruido. Esas cinco definiciones, tomadas en serio, evitan la mayoría de los errores de diseño que ocurren antes de escribir la primera línea de código de un modelo — que es exactamente el objetivo declarado de esta etapa del proyecto.

---

# Anexo — Registro de revisión crítica (post-redacción)

Revisión de seis afirmaciones marcadas **[B]**, realizada antes de pasar al Capítulo 2. Ninguna corrección altera las conclusiones de la Sección 9.

| # | Afirmación original | Evaluación | Corrección aplicada |
|---|---|---|---|
| 1 | MSE equivale a asumir normal homocedástica, contradicha por 1.3 | REQUIERE MATIZ | MSE es consistente para la media condicional bajo cualquier distribución con 2º momento finito. Se pierde eficiencia e inferencia válida, no el objetivo. §2.6 y M1 |
| 2 | La estructura de un futuro *es* un excess return; no restar $r$ | CORREGIR | Degradado a analogía; el enunciado riguroso aplica a posiciones totalmente colateralizadas y depende de la serie construida → PREGUNTA ABIERTA. §2.1.4 y §4.2 |
| 3 | Entrenar indistintamente con MNQ o NQ | REQUIERE MATIZ | Reformulado como hipótesis empírica con condiciones de frecuencia y tipo de feature. §2.1.3 y §4.2 |
| 4 | Los índices son "casi el mismo activo"; la información está en los spreads | CORREGIR | Separadas correlación / redundancia / aporte predictivo incremental; spreads degradados a conjetura de reparametrización. §2.5 y §4.2 |
| 5 | La agregación temporal reduce la kurtosis (listado como [A]) | CORREGIR | Error de atribución: Tsay no enuncia esa ley, y las series diarias y mensuales cubren períodos distintos. Condicionado al supuesto iid, con el contraejemplo de las distribuciones estables. §2.7 y §4.1 |
| 6 | Los retornos "deben" calcularse intra-contrato | REQUIERE MATIZ | Criterio reemplazado por "preservar la cantidad que importa" (ratio→retornos, aditivo→puntos); leak acotado a features dependientes del nivel, con test de invariancia. §2.1.3, §4.2 y error 13 |

**Patrón detectado.** Cinco de los seis problemas tienen la misma forma: una observación correcta convertida en regla normativa sin la condición que la hace válida. Conviene aplicar el mismo filtro a las afirmaciones [B] de los capítulos siguientes.
