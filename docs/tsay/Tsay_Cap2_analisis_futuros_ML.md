# Tsay, *Analysis of Financial Time Series* (3ª ed.) — Capítulo 2
## *Linear Time Series Analysis and Its Applications*
### Estudio orientado a un sistema de Machine Learning para trading de futuros

**Fuente analizada:** Ruey S. Tsay, *Analysis of Financial Time Series, Third Edition*, Wiley (2010), Capítulo 2, pp. 29–107 (secciones 2.1–2.11, ejercicios y referencias). Omitido: Apéndice de comandos SCA y salidas de software obsoleto.

**Convención de atribución** (idéntica al informe del Capítulo 1):

- **[A]** = afirmación, resultado, definición, demostración o ejemplo que proviene directamente de Tsay.
- **[B]** = interpretación, extrapolación o adaptación propia a futuros, trading cuantitativo o ML.
- **PREGUNTA ABIERTA** = cuestión no resoluble con este capítulo.

**Filtro heredado de la revisión del Capítulo 1**, aplicado en todo el documento:

> Una observación razonable no debe convertirse en regla normativa sin especificar las condiciones bajo las cuales es válida.

---

# 1. Resumen ejecutivo

El Capítulo 1 planteó el problema como $F(r_{t+h}\mid X_t)$ y dejó abierta la pregunta de si la condicional difiere de la marginal. El Capítulo 2 responde una versión **restringida** de esa pregunta: qué parte de esa diferencia es **lineal**, cómo se detecta, cómo se modela y cómo se verifica que fue capturada. Esa restricción es su mayor virtud y su mayor límite: todo el aparato del capítulo —ACF, AR, MA, ARMA, raíces unitarias— mide y modela **dependencia lineal en la media condicional**, nada más.

Diez ideas concentran el valor:

1. **La estacionariedad débil es un supuesto sobre los dos primeros momentos, no sobre la serie** [A]. $E(r_t)=\mu$ constante y $\text{Cov}(r_t,r_{t-\ell})=\gamma_\ell$ dependiente sólo de $\ell$. Tsay dice explícitamente que la estacionariedad estricta es "una condición muy fuerte, difícil de verificar empíricamente" y que el libro trabaja principalmente con estacionariedad débil. **[B]** Esto importa porque casi todo lo que sigue —incluidas las ACF y sus errores estándar— sólo está definido bajo ese supuesto.

2. **Tsay da un procedimiento empírico concreto para verificarla** [A]: "dividir los datos en submuestras y comprobar la consistencia de los resultados obtenidos entre submuestras". Es la única receta de verificación que ofrece, y es directamente aplicable.

3. **La correlación mide dependencia lineal, y sólo bajo normalidad conjunta la ausencia de correlación implica independencia** [A]. Tsay lo enuncia de forma exacta: "si tanto $X$ como $Y$ son variables normales, entonces $\rho_{x,y}=0$ si y sólo si $X$ e $Y$ son independientes". **[B]** Como el Capítulo 1 estableció que los retornos **no** son normales, la condición no se cumple: en datos financieros, ACF ≈ 0 no autoriza a concluir independencia. Éste es el eje de todo el informe.

4. **Los tests de autocorrelación suponen iid** [A]. Ljung–Box es $\chi^2_m$ "bajo el supuesto de que $\{r_t\}$ es una secuencia iid con ciertas condiciones de momentos", y Tsay advierte que muchos paquetes usan $1/T$ como varianza asintótica de $\hat\rho_\ell$, lo que "esencialmente asume que la serie subyacente es iid". **[B]** En series con clustering de volatilidad ese supuesto falla y los p-valores están mal calibrados.

5. **El propio Tsay muestra que la dependencia lineal detectable es minúscula** [A]. El AR(3) ajustado al índice value-weighted mensual tiene **$R^2 = 2.46\%$** y coeficientes de ~0.11; Tsay comenta que "los coeficientes AR del modelo ajustado son pequeños, indicando que la dependencia serial de la serie es débil, **aunque sea estadísticamente significativa al nivel del 1%**". **[B]** Este resultado refuerza, desde la perspectiva de la dependencia lineal temporal, el mensaje del Capítulo 1 sobre la baja relación señal/ruido en la media de los retornos: una dependencia puede ser estadísticamente significativa y, al mismo tiempo, explicar una fracción muy pequeña de la variabilidad observada. Por tanto, significancia estadística, capacidad predictiva y relevancia económica deben evaluarse por separado.

6. **$R^2$ es una métrica traicionera con series no estacionarias** [A]. Tsay lo dice sin ambigüedad: para una serie con raíz unitaria, "el $R^2$ de un ajuste AR(1) converge a uno cuando el tamaño de muestra crece, **independientemente del verdadero modelo subyacente**". **[B]** Cualquier métrica de bondad de ajuste evaluada sobre precios en vez de retornos puede ser espectacular y completamente vacía.

7. **El random walk se define y se testea, pero nunca se "demuestra"** [A]. Tsay aplica ADF al log del S&P 500 diario (14,462 obs., 1950–2008) y obtiene estadístico −1.998 con p = 0.602: "la hipótesis de raíz unitaria **no puede rechazarse** a ningún nivel de significancia razonable". La distribución del estadístico es no estándar y sus valores críticos se obtienen por simulación. **[B]** No rechazar no es demostrar.

8. **ARMA es explícitamente poco útil para retornos, pero central para volatilidad** [A]. Cita textual: "para las series de retornos en finanzas, la probabilidad de usar modelos ARMA es baja. Sin embargo, el concepto de modelos ARMA es altamente relevante en el modelado de volatilidad. De hecho, el modelo GARCH puede considerarse un modelo ARMA, si bien no estándar, para la serie $a_t^2$". Esto redirige el capítulo entero hacia el Capítulo 3.

9. **HAC no es un detalle técnico: en el ejemplo de Tsay el t-ratio cae de 107.91 a 39.92** [A]. Ignorar correlación serial y heterocedasticidad infla el t-ratio en un factor de ~2.7 en un caso concreto y publicado. Tsay dice que el estimador convencional es inconsistente "resultando a menudo en la inflación de los t-ratios".

10. **La ACF de $|r_t|$ es significativa incluso después de 300 lags** [A]. Sobre retornos diarios de índices CRSP (1970–2008): "las ACF son relativamente pequeñas en magnitud pero decaen muy lentamente; parecen significativas al 5% incluso después de 300 lags". Simultáneamente, la ACF de $r_t$ es prácticamente nula. **[B]** Ésta es, en un solo par de gráficos, la evidencia de que dirección y magnitud son problemas estadísticamente distintos, y de que el segundo tiene muchísima más estructura detectable.

**La conclusión estructural del capítulo para el proyecto**, y es [B]: el Capítulo 2 entrega un **conjunto de herramientas de diagnóstico y una familia de benchmarks**, no una familia de modelos predictivos para retornos. Su uso correcto en un pipeline de ML es (i) verificar supuestos, (ii) construir baselines honestos que un modelo complejo debe superar, (iii) diagnosticar residuos, y (iv) corregir inferencia. Su uso incorrecto sería adoptar AR/ARMA como modelo de trading o convertir PACF en un selector automático de lookback.

**Lo que el capítulo NO resuelve:** heterocedasticidad condicional (Cap. 3), dependencia no lineal (Cap. 4), microestructura (Cap. 5), lead-lag y cointegración multivariada (Cap. 8), factores (Cap. 9), covarianza condicional dinámica (Cap. 10), estados latentes y cambios de régimen (Caps. 11–12). Y nunca: costos de transacción, validación out-of-sample, ni sobreajuste por búsqueda múltiple.

---

# 2. Análisis sección por sección

## 2.1 Sección 2.1 — Stationarity

### [A] Qué afirma Tsay

Abre con una frase categórica: **"El fundamento del análisis de series temporales es la estacionariedad."**

**Estacionariedad estricta.** $\{r_t\}$ es estrictamente estacionaria si la distribución conjunta de $(r_{t_1},\dots,r_{t_k})$ es idéntica a la de $(r_{t_1+t},\dots,r_{t_k+t})$ para todo $t$, cualquier entero positivo $k$ y cualquier colección $(t_1,\dots,t_k)$. Es decir, la conjunta es **invariante ante traslación temporal**. Tsay la califica de "condición muy fuerte, difícil de verificar empíricamente".

**Estacionariedad débil (o de covarianza).** $\{r_t\}$ es débilmente estacionaria si:
- (a) $E(r_t)=\mu$, constante;
- (b) $\text{Cov}(r_t,r_{t-\ell})=\gamma_\ell$, que depende **sólo de $\ell$**, no de $t$.

Implícitamente se asume que los **dos primeros momentos son finitos**.

**Relaciones.** Si $r_t$ es estrictamente estacionaria y sus dos primeros momentos son finitos, entonces es débilmente estacionaria. **El recíproco no es cierto en general.** Sin embargo, **si $r_t$ es normalmente distribuida, estacionariedad débil ⟺ estacionariedad estricta**. Tsay declara que el libro se ocupa principalmente de series débilmente estacionarias.

**Interpretación gráfica.** La estacionariedad débil implica que el time plot de los $T$ valores mostraría fluctuaciones **con variación constante alrededor de un nivel fijo**.

**Para qué sirve.** "En aplicaciones, la estacionariedad débil permite hacer inferencia sobre observaciones futuras (por ejemplo, predicción)."

**Autocovarianza.** $\gamma_\ell=\text{Cov}(r_t,r_{t-\ell})$ es la autocovarianza de lag $\ell$. Dos propiedades: (a) $\gamma_0=\text{Var}(r_t)$; (b) $\gamma_{-\ell}=\gamma_\ell$.

**Verificación empírica.** "En la literatura financiera, es común asumir que una serie de retornos de un activo es débilmente estacionaria. Este supuesto **puede comprobarse empíricamente** siempre que se disponga de un número suficiente de retornos históricos. Por ejemplo, uno puede **dividir los datos en submuestras y comprobar la consistencia de los resultados obtenidos a través de las submuestras**."

### Significado estadístico

La estacionariedad débil no es una propiedad de la trayectoria observada sino del **proceso generador**. Es lo que permite que un único camino muestral sirva para estimar cantidades poblacionales: si los momentos cambiaran con $t$, cada observación provendría de una distribución diferente y no habría con qué promediar. Es, en esencia, la condición que hace posible la **inferencia estadística a partir de una sola realización**.

### [B] 2.1.1 ¿Por qué importa? Qué pasa si los momentos cambian con el tiempo

Tsay no desarrolla explícitamente qué falla; enuncia que la estacionariedad "permite hacer inferencia". La siguiente elaboración es [B]:

**Si $E(r_t)$ cambia con $t$.** El estimador $\hat\mu=\frac1T\sum r_t$ estima un promedio temporal de medias distintas, que no corresponde a la media de ningún período concreto —y en particular, no a la del período futuro que interesa. Un modelo que aprenda ese nivel medio estará sistemáticamente sesgado en cualquier subperíodo.

**Si $\text{Var}(r_t)$ cambia con $t$.** $\hat\sigma^2$ es un promedio de varianzas heterogéneas. Consecuencias en cascada: (i) toda estandarización $z=(r-\hat\mu)/\hat\sigma$ queda mal escalada en ambos extremos —comprime los períodos volátiles y amplifica los tranquilos—; (ii) los errores estándar y los intervalos de confianza son incorrectos; (iii) un target definido en múltiplos de $\hat\sigma$ global significa cosas distintas en 2010 y en 2020.

**Si $\text{Corr}(r_t,r_{t-k})$ cambia con $t$.** El objeto $\rho_k$ **ni siquiera está definido**: la ACF presupone que la covarianza depende sólo del lag. Estimar una ACF sobre una serie con dependencia variable en el tiempo produce un número que es un promedio de estructuras distintas, y ese promedio puede ser cercano a cero aunque en cada subperíodo haya dependencia fuerte de signos opuestos. **Un mercado con dependencia real que alterna de signo por régimen se ve, en una ACF global, exactamente igual que un mercado sin dependencia alguna.**

Esta última consecuencia es la más peligrosa para el proyecto y no está en Tsay.

### [B] 2.1.2 Qué NO significa estacionariedad

Se marca el origen de cada aclaración:

| Aclaración | Origen |
|---|---|
| **No significa que la serie sea constante.** La estacionariedad describe momentos, no trayectorias; una serie estacionaria fluctúa. | **[A] implícito** — Tsay describe la serie estacionaria como fluctuando "con variación constante alrededor de un nivel fijo". |
| **No significa que no existan shocks.** El modelo lineal $r_t=\mu+\sum\psi_i a_{t-i}$ es estacionario y está compuesto enteramente de shocks. | **[A]** — Ecs. (2.4)–(2.5). |
| **No significa que los shocks no importen.** Importan, pero su efecto **decae**: para una serie estacionaria "el impacto del shock remoto $a_{t-i}$ sobre el retorno $r_t$ se desvanece a medida que $i$ crece". | **[A]** — texto tras Ec. (2.5). |
| **No significa que el mercado nunca cambie.** Ésta es una extensión: la estacionariedad es una hipótesis sobre el proceso generador durante el período muestral, no una garantía sobre el futuro. | **[B]** |
| **No excluye heterocedasticidad condicional.** Una serie puede tener $\text{Var}(r_t)$ incondicional constante y sin embargo $\text{Var}(r_t\mid \mathcal{F}_{t-1})$ variable en el tiempo. | **[B]**, aunque **[A] parcialmente**: Tsay menciona heterocedasticidad condicional en 2.10 y le dedica el Cap. 3 sin retirar el supuesto de estacionariedad débil. |
| **No implica que todas las distribuciones condicionales sean idénticas.** La estacionariedad débil sólo restringe media y autocovarianza; la estricta sí restringe la conjunta, pero no es el supuesto de trabajo del libro. | **[B]**, derivado directamente de que **[A]** débil ⇏ estricta. |

**[B] Corolario importante:** el hecho estilizado del Capítulo 1 —volatility clustering— es **compatible** con estacionariedad débil. Un GARCH estacionario tiene varianza incondicional constante y varianza condicional variable. Por lo tanto, observar clustering **no refuta** la estacionariedad débil, y confundir ambas cosas es un error frecuente.

### [B] 2.1.3 Estacionariedad de la variable vs estabilidad de la función predictiva

Ésta es una distinción que **Tsay no formula** —no habla de modelos de aprendizaje ni de funciones predictivas aprendidas— y que conviene mantener separada con rigor:

**Estacionariedad de la variable** es una propiedad marginal/de segundo momento de una serie: ¿son $E(r_t)$ y $\gamma_\ell$ invariantes en $t$?

**Estabilidad de la función predictiva** es una propiedad de la relación conjunta: ¿es la aplicación $X_t\mapsto F(y_{t+h}\mid X_t)$ la misma en todo $t$?

Son lógicamente independientes en ambas direcciones:

- Todas las variables pueden ser estacionarias y la relación entre ellas puede cambiar. Ejemplo conceptual: dos futuros con retornos estacionarios cuya correlación cambia de signo por régimen. Cada serie pasa cualquier test de estacionariedad univariado; el modelo que aprendió la relación en un régimen falla en el otro.
- La relación puede ser estable y las variables no estacionarias. Ejemplo: una relación de equilibrio de largo plazo entre dos series con raíz unitaria — que es precisamente el fenómeno de la cointegración. **PREGUNTA ABIERTA → Cap. 8.**

**Implicancia para el proyecto [B]:** testear estacionariedad de features y de target es **necesario pero no suficiente**. La pregunta que realmente determina si un modelo generalizará es la segunda, y el Capítulo 2 no ofrece herramientas para ella. Lo más cercano que ofrece es la receta de submuestras [A], que puede extenderse (extensión [B]) a: *estimar la relación en submuestras disjuntas y comparar*. Esto no es validación out-of-sample; es un diagnóstico de estabilidad.

### [B] Adaptación a mercados de futuros

- **Precios de futuros:** no cabe esperar estacionariedad —no hay nivel fijo [A, ver 2.7]. Esto es más marcado aún en futuros por el roll: una serie continua ajustada no tiene siquiera un nivel económicamente interpretable.
- **Retornos de futuros:** candidatos plausibles a estacionariedad débil, pero **es una hipótesis a verificar por instrumento y por período**, no una propiedad garantizada. Tsay dice que "en finanzas se cree comúnmente" que la serie de retornos es estacionaria — es una creencia, y él la marca como tal.
- **Submuestras naturales en futuros [B]:** año calendario, régimen de volatilidad, sesión (asiática/europea/americana), pre y post cambios estructurales de mercado (decimalización, cambios de horario de CME, aparición de contratos Micro). La receta de Tsay aplicada a estos cortes es un diagnóstico barato y muy informativo.
- **Series que casi con certeza no serán estacionarias en niveles [B]:** open interest, volumen (tendencia secular y estacionalidad de contrato), spreads de calendario, volatilidad implícita en niveles. Usar estas variables crudas como features arrastra el problema al modelo.

### [B] Implicancias potenciales para ML

- Es razonable **estudiar** si conviene transformar variables antes de usarlas como features. No es una decisión: es una pregunta empírica cuyo criterio debería ser el desempeño out-of-sample y la estabilidad, no el resultado de un test.
- Los modelos que particionan por umbrales absolutos (árboles) son especialmente frágiles ante no estacionariedad en nivel, porque el umbral aprendido puede quedar fuera del rango de los datos futuros. Los modelos lineales fallan de otra forma (extrapolación). **Ninguna de las dos observaciones proviene de Tsay.**
- La estandarización de features debe ser **causal** (calculada con información hasta $t$). Con no estacionariedad, una estandarización global es además un canal de leak. **[B]**

### Preguntas abiertas de esta sección

- ¿Es la volatilidad condicional lo que rompe la apariencia de estacionariedad, o hay no estacionariedad también en la media? → **Cap. 3.**
- ¿Existen cambios de régimen discretos y modelables? → **Caps. 11–12.**
- ¿La no estacionariedad de una serie puede compensarse con la de otra (cointegración)? → **Cap. 8.**

---

## 2.2 Sección 2.2 — Correlation and Autocorrelation Function

### [A] Qué afirma Tsay

**Correlación.** $\rho_{x,y}=\text{Cov}(X,Y)/\sqrt{\text{Var}(X)\text{Var}(Y)}$, asumiendo que las varianzas existen. **"Este coeficiente mide la fuerza de la dependencia LINEAL entre $X$ e $Y$."** Se cumple $-1\le\rho\le1$ y $\rho_{x,y}=\rho_{y,x}$. Las variables son no correlacionadas si $\rho=0$.

**La condición clave, textual:** "Además, **si tanto $X$ como $Y$ son variables aleatorias normales, entonces $\rho_{x,y}=0$ si y sólo si $X$ e $Y$ son independientes**."

**ACF.** Para una serie débilmente estacionaria, la autocorrelación de lag $\ell$ es (Ec. 2.1):
$$\rho_\ell=\frac{\text{Cov}(r_t,r_{t-\ell})}{\sqrt{\text{Var}(r_t)\text{Var}(r_{t-\ell})}}=\frac{\gamma_\ell}{\gamma_0}$$
usando $\text{Var}(r_t)=\text{Var}(r_{t-\ell})$ por estacionariedad débil. Propiedades: $\rho_0=1$, $\rho_\ell=\rho_{-\ell}$, $-1\le\rho_\ell\le1$. Y: **"una serie débilmente estacionaria $r_t$ no está serialmente correlacionada si y sólo si $\rho_\ell=0$ para todo $\ell>0$."**

**ACF muestral** (Ec. 2.2): $\hat\rho_\ell=\dfrac{\sum_{t=\ell+1}^{T}(r_t-\bar r)(r_{t-\ell}-\bar r)}{\sum_{t=1}^{T}(r_t-\bar r)^2}$, $0\le\ell<T-1$.

**Distribución asintótica.** Si $\{r_t\}$ es **iid** con $E(r_t^2)<\infty$, entonces $\hat\rho_\ell$ es asintóticamente normal con media cero y **varianza $1/T$** (Brockwell y Davis, 1991, Teorema 7.2.2). Test: $t=\sqrt{T}\hat\rho_1$, asintóticamente normal estándar.

**Fórmula de Bartlett.** Más generalmente, si $r_t$ es débilmente estacionaria y satisface $r_t=\mu+\sum_{i=0}^{q}\psi_i a_{t-i}$ con $\psi_0=1$ y $\{a_j\}$ iid de media cero, entonces $\hat\rho_\ell$ es asintóticamente normal con media cero y varianza $\left(1+2\sum_{i=1}^{q}\rho_i^2\right)/T$ para $\ell>q$.

**Test individual.** $t\text{-ratio}=\dfrac{\hat\rho_\ell}{\sqrt{(1+2\sum_{i=1}^{\ell-1}\hat\rho_i^2)/T}}$; asintóticamente normal estándar **si $\{r_t\}$ es una serie gaussiana estacionaria con $\rho_j=0$ para $j>\ell$**.

**Advertencia explícita sobre el software:** "Por simplicidad, muchos paquetes de software usan $1/T$ como varianza asintótica de $\hat\rho_\ell$ para todo $\ell\neq0$. **Esencialmente están asumiendo que la serie subyacente es una secuencia iid.**"

**Sesgo en muestras finitas:** $\hat\rho_\ell$ es un estimador sesgado de $\rho_\ell$; el sesgo es de orden $1/T$, "que puede ser sustancial cuando el tamaño de muestra $T$ es pequeño. En la mayoría de aplicaciones financieras, $T$ es relativamente grande de modo que el sesgo no es serio".

**Tests conjuntos.**
- Box–Pierce (1970): $Q^*(m)=T\sum_{\ell=1}^{m}\hat\rho_\ell^2$, para $H_0:\rho_1=\dots=\rho_m=0$. "Bajo el supuesto de que $\{r_t\}$ es una secuencia **iid** con ciertas condiciones de momentos, $Q^*(m)$ es asintóticamente $\chi^2$ con $m$ grados de libertad."
- Ljung–Box (1978), Ec. (2.3): $Q(m)=T(T+2)\sum_{\ell=1}^{m}\dfrac{\hat\rho_\ell^2}{T-\ell}$, modificación "para aumentar la potencia del test en muestras finitas".
- **Elección de $m$:** "En la práctica, la elección de $m$ puede afectar el desempeño del estadístico $Q(m)$. Se usan a menudo varios valores de $m$. Estudios de simulación sugieren que la elección $m\approx\ln(T)$ proporciona mejor desempeño en potencia. Esta regla general necesita modificación en el análisis de series estacionales, para las cuales las autocorrelaciones con lags múltiplos de la estacionalidad son más importantes."

**Evidencia empírica** (Figuras 2.1 y 2.2):
- **IBM mensual** (1926–2008): todas las ACF muestrales dentro de los límites de dos errores estándar; simples $Q(5)=3.37$ (p = 0.64), $Q(10)=13.99$ (p = 0.17); log $Q(5)=3.52$ (p = 0.62), $Q(10)=13.39$ (p = 0.20). "las correlaciones seriales de los retornos mensuales de IBM son muy pequeñas, si es que existen."
- **Índice CRSP value-weighted mensual**: hay correlaciones seriales significativas al 5%. $Q(5)=29.71$, $Q(10)=39.55$ (simples); $Q(5)=28.38$, $Q(10)=36.16$ (log). Todos con **p < 0.0001**. "el retorno mensual del índice de mercado parece tener dependencia serial más fuerte que los retornos de acciones individuales."

**Conexión con eficiencia de mercado, textual:** "En la literatura financiera, una versión de la teoría CAPM es que el retorno $\{r_t\}$ de un activo no es predecible y no debería tener autocorrelaciones. Testear autocorrelaciones nulas se ha usado como herramienta para verificar el supuesto de mercado eficiente. **Sin embargo, la manera en que se determinan los precios de las acciones y se calculan los retornos de los índices podría introducir autocorrelaciones en la serie de retornos observada. Esto es particularmente así en el análisis de datos financieros de alta frecuencia.** Discutimos algunas de estas cuestiones, como el bid–ask bounce y el trading no sincrónico, en el Capítulo 5."

### Significado estadístico: qué pregunta responde exactamente $\rho_k$

$\rho_k=\text{Corr}(r_t,r_{t-k})$ responde: **¿cuánto de la variación de $r_t$ es explicable por una función lineal de $r_{t-k}$ únicamente, en promedio sobre toda la muestra?** Tres restricciones fuertes están contenidas ahí:

1. **Lineal.** No detecta relaciones en U, umbrales, ni interacciones.
2. **Bivariada y marginal en el resto.** Mide $r_t$ contra $r_{t-k}$ ignorando los lags intermedios (a diferencia de la PACF).
3. **Promediada sobre toda la muestra.** Si la relación existe pero cambia de signo entre subperíodos, $\rho_k$ tiende a cero.

**Qué significa $\rho_k\neq0$:** existe dependencia lineal en la media, promediada sobre la muestra. Es evidencia de que la condicional difiere de la marginal — el nivel más débil de los cuatro que definimos en el Capítulo 1.

**Qué significa $\rho_k\approx0$:** que **no hay dependencia lineal detectable de ese tipo, en promedio**. No dice nada sobre dependencia no lineal, dependencia en varianza, ni dependencia condicionada a otra variable.

### [B] La jerarquía, revisada con las herramientas del Capítulo 2

Retomando la jerarquía del Capítulo 1, ahora con contenido preciso:

| Nivel | Objeto | Herramienta del Cap. 2 | Implica el nivel anterior |
|---|---|---|---|
| **Ausencia de autocorrelación** | $\rho_k=0\ \forall k$ | ACF, Ljung–Box | — |
| **Ausencia de dependencia** | $F(r_t\mid\text{pasado})=F(r_t)$ | Ninguna en este capítulo | Dependencia ⇒ puede haber $\rho_k=0$ |
| **Ausencia de predecibilidad** | La condicional no aporta ventaja | Comparación out-of-sample vs benchmark | — |
| **Ausencia de predecibilidad explotable** | Ventaja < costos | Nada en Tsay | — |

**El punto lógico exacto [B]:** $\rho_k=0\ \forall k$ implica independencia **sólo bajo normalidad conjunta** [A]. El Capítulo 1 estableció que los retornos financieros no son normales, con evidencia abrumadora (JB = 60,922 para IBM diario). **Por lo tanto la condición que autorizaría el salto de "no correlacionado" a "independiente" está empíricamente refutada en el propio libro.** Ésta es la inferencia más importante que hay que evitar, y Tsay entrega ambos ingredientes —la condición y su refutación— sin unirlos explícitamente en un párrafo.

**Segunda observación [B], de sentido contrario:** $\rho_k\neq0$ tampoco implica predecibilidad económica. El propio Tsay advierte que la forma en que se determinan los precios y se calculan los retornos de índices **puede introducir** autocorrelación. Un índice equal-weighted contiene acciones ilíquidas cuyos precios de cierre son rezagados; la autocorrelación resultante es un artefacto de agregación, no información. Es exactamente el fenómeno que hace que "los retornos del índice tengan dependencia serial más fuerte que los de acciones individuales" [A].

### [B] Adaptación a mercados de futuros

- **Expectativa razonable, no resultado:** en futuros líquidos de índices, cabe esperar autocorrelación de retornos diarios muy cercana a cero. Es una expectativa basada en el argumento de arbitraje, no un hallazgo de Tsay ni un dato del capítulo. **Debe medirse.**
- **En frecuencias altas, la advertencia de Tsay es directamente aplicable:** parte de cualquier autocorrelación observada puede ser microestructural. Tsay lo señala para acciones y remite al Cap. 5; en futuros el mecanismo análogo existe (bid–ask bounce en el libro de órdenes). **No adelantar el Cap. 5**, pero sí registrar que **una ACF significativa a alta frecuencia no es, por sí sola, evidencia de señal**.
- **Los futuros no tienen el problema del índice sintético** que Tsay señala (agregación de precios no sincrónicos de componentes): un futuro es un instrumento que cotiza directamente. Esto **[B]** sugiere que la autocorrelación espuria por agregación debería ser menor en futuros que en índices cash — pero es una hipótesis, no un hecho establecido.
- **Un futuro sobre índice y su índice cash subyacente no son intercambiables** a estos efectos: el índice cash hereda el problema de agregación, el futuro no. Mezclarlos en un mismo análisis de ACF confunde dos fenómenos distintos.

### [B] Cómo usar la ACF conceptualmente sobre distintos objetos

Sin ejecutar nada todavía, éste es el mapa de qué pregunta responde la ACF aplicada a cada objeto:

| Objeto | Qué pregunta responde | Expectativa a priori [B] |
|---|---|---|
| **$r_t$** (retornos) | ¿Hay dependencia lineal en la dirección/magnitud con signo? | Débil o nula en futuros líquidos |
| **$\lvert r_t\rvert$** | ¿Hay dependencia lineal en la magnitud? | **Fuerte y persistente** [A, §2.11] |
| **$r_t^2$** | Idem, con más peso en extremos | Fuerte, pero más ruidosa que $\lvert r_t\rvert$ por la kurtosis |
| **Volatilidad realizada** | Persistencia de la volatilidad medida | Fuerte → Cap. 3 |
| **Features** | ¿Cuánta memoria tiene el propio feature? Un feature muy autocorrelacionado aporta pocas observaciones efectivas | Depende del diseño |
| **Residuos de un modelo** | ¿Queda estructura lineal sin capturar? | Debe ser ≈ 0 si el modelo capturó lo lineal |

**Advertencia sobre la última fila [B]:** que los residuos de un modelo de ML no tengan autocorrelación **no** significa que no quede información. Significa que no queda información **lineal en la media, detectable por este test**. Es una condición necesaria de adecuación, no suficiente.

### [B] Implicancias potenciales para ML

- La ACF y Ljung–Box son **herramientas de diagnóstico**, no criterios de diseño. Convertirlas en criterio de diseño es uno de los errores catalogados en §8.
- Los errores estándar por defecto de casi todo el software asumen iid [A]. En series financieras con clustering de volatilidad, **las bandas de confianza dibujadas en un gráfico de ACF son demasiado estrechas**, y por tanto se declararán significativos lags que no lo son. Corolario práctico [B]: desconfiar de significancias marginales en gráficos de ACF de retornos.
- La elección de $m$ en Ljung–Box importa [A] y es un grado de libertad del analista: probar muchos $m$ y reportar el que da el resultado deseado es data snooping.

### Preguntas abiertas

- ¿Cuánta de la autocorrelación observable a alta frecuencia es microestructural? → **Cap. 5.**
- ¿Cómo se generaliza la ACF a relaciones entre instrumentos (cross-correlation, lead-lag)? → **Cap. 8.**
- ¿Cómo se detecta dependencia no lineal que la ACF no ve? → **Cap. 4.**

---

## 2.3 Sección 2.3 — White Noise and Linear Time Series

### [A] Qué afirma Tsay

**Definición de white noise, textual:** "Una serie temporal $r_t$ se llama **white noise** si $\{r_t\}$ es una secuencia de variables aleatorias **independientes e idénticamente distribuidas** con media y varianza finitas. En particular, si $r_t$ está normalmente distribuida con media cero y varianza $\sigma^2$, la serie se llama **Gaussian white noise**. Para una serie de white noise, **todas las ACF son cero**."

Y luego, la frase práctica: **"En la práctica, si todas las ACF muestrales están cerca de cero, entonces la serie es una serie de white noise."** Aplicado a sus ejemplos: "los retornos mensuales de IBM están cerca de white noise, mientras que los del índice value-weighted no lo están".

**Serie lineal** (Ec. 2.4): $r_t$ es lineal si puede escribirse
$$r_t=\mu+\sum_{i=0}^{\infty}\psi_i a_{t-i}$$
donde $\mu=E(r_t)$, $\psi_0=1$ y $\{a_t\}$ es una secuencia de variables iid de media cero con distribución bien definida (es decir, $\{a_t\}$ es white noise). **"$a_t$ denota la nueva información en el tiempo $t$ de la serie y se le llama a menudo la innovación o shock en el tiempo $t$."** Los $\psi_i$ son los "pesos $\psi$".

**"No todas las series financieras son lineales, sin embargo. Estudiamos no linealidad y modelos no lineales en el Capítulo 4."**

**Momentos** (Ec. 2.5): $E(r_t)=\mu$, $\text{Var}(r_t)=\sigma_a^2\sum_{i=0}^{\infty}\psi_i^2$. Como $\text{Var}(r_t)<\infty$, la secuencia $\{\psi_i^2\}$ debe converger, es decir $\psi_i^2\to0$. **"Consecuentemente, para una serie estacionaria, el impacto del shock remoto $a_{t-i}$ sobre el retorno $r_t$ se desvanece a medida que $i$ crece."**

**Autocovarianza** (Ec. 2.6): $\gamma_\ell=\sigma_a^2\sum_{j=0}^{\infty}\psi_j\psi_{j+\ell}$, y (Ec. 2.7) $\rho_\ell=\dfrac{\sum_{i=0}^{\infty}\psi_i\psi_{i+\ell}}{1+\sum_{i=1}^{\infty}\psi_i^2}$.

**"Los modelos de series temporales lineales son modelos econométricos y estadísticos usados para describir el patrón de los pesos $\psi$ de $r_t$."** Para una serie débilmente estacionaria, $\psi_i\to0$ y por tanto $\rho_\ell\to0$ cuando $\ell$ crece: "para retornos de activos, esto significa que, como es de esperar, la dependencia lineal del retorno actual $r_t$ sobre el retorno remoto $r_{t-\ell}$ disminuye para $\ell$ grande".

### [B] Observación crítica: la definición de Tsay es más fuerte de lo habitual

**Tsay define white noise como iid.** En buena parte de la literatura econométrica, "white noise" significa sólo **media cero, varianza constante y no correlacionado** — una condición estrictamente más débil. La diferencia es exactamente la que separa las nociones que este informe insiste en no confundir:

- **White noise en sentido débil (no correlacionado):** $\rho_\ell=0\ \forall\ell$. Un proceso GARCH cumple esto.
- **White noise en sentido de Tsay (iid):** además, ninguna dependencia de ningún orden. Un proceso GARCH **no** cumple esto.

**Esto genera una tensión interna en el capítulo [B].** La frase práctica de Tsay —"si todas las ACF muestrales están cerca de cero, entonces la serie es white noise"— es, tomada literalmente y con su propia definición de white noise, **una inferencia inválida**: las ACF muestrales sólo verifican la no correlación, no la independencia. Y el propio libro la refuta dos veces:

1. En §2.11 [A], la ACF de $\lvert r_t\rvert$ de índices diarios es significativa hasta 300 lags, mientras que la de $r_t$ es prácticamente nula. Una serie cuya ACF es nula pero cuya ACF de valores absolutos no lo es **no puede ser iid**.
2. En el diagnóstico del AR(3) [A], los residuos tienen Jarque–Bera = 1656 con p = 0.0000. No son gaussianos, y su comportamiento en varianza es el objeto del Capítulo 3.

**Conclusión [B]:** la frase de Tsay debe leerse como una heurística práctica dentro del alcance del capítulo (modelado lineal de la media), no como un criterio de independencia. Para el proyecto, la formulación correcta es: *ACF muestral ≈ 0 ⇒ la serie es compatible con white noise en el sentido no correlacionado; establecer independencia requiere herramientas que este capítulo no provee.*

### [B] La cadena serie → modelo → residuo

El objetivo del modelado lineal, tal como el capítulo lo plantea, es una **descomposición**:
$$r_t=\underbrace{\hat\mu_t}_{\text{parte explicable linealmente por el pasado}}+\underbrace{\hat a_t}_{\text{residuo}}$$
y el criterio de adecuación es que $\hat a_t$ se comporte como white noise [A, §2.4.2 "Model Checking": "si el modelo es adecuado, entonces la serie residual debería comportarse como white noise"].

**Por qué éste es el objetivo natural [B]:** si el residuo conserva estructura temporal sistemática, entonces por construcción existe información en el pasado que el modelo **no** usó. Es una condición de completitud respecto de la clase de estructuras que el modelo puede representar. Nótese la restricción: "respecto de la clase de estructuras que el modelo puede representar". Un modelo lineal cuyos residuos son no correlacionados ha agotado la estructura **lineal en la media**; no ha agotado nada más.

### [B] Los cuatro niveles de "buenos residuos", que no son equivalentes

Esta distinción responde directamente a la pregunta planteada: *si un modelo de ML da buenos resultados pero sus errores conservan dependencia temporal, ¿qué significa?*

| Propiedad de los residuos | Qué se verifica con | Qué implica si falla | Se implican entre sí |
|---|---|---|---|
| **No correlacionados** | ACF, Ljung–Box | Queda estructura **lineal** sin capturar: el modelo es mejorable de forma directa | La más débil |
| **Independientes** | No hay test en este capítulo | Queda estructura de algún tipo (no lineal, en varianza, en colas) | Independientes ⇒ no correlacionados; el recíproco requiere normalidad [A] |
| **Homocedásticos** | No en este capítulo → Cap. 3 | La incertidumbre del modelo varía en el tiempo y no está modelada | Independiente de las otras dos |
| **Gaussianos** | Jarque–Bera (Cap. 1) | Los intervalos de predicción y los tests basados en normalidad están mal calibrados | Independiente de las otras dos |

**Respuesta a la pregunta [B]:** que un modelo de ML tenga buen desempeño y residuos con dependencia temporal significa que **existe estructura remanente aprovechable**, pero de qué tipo depende de dónde esté la dependencia:

- Si la dependencia está en **la media de los residuos** (ACF de $\hat a_t$ significativa), el modelo está dejando señal lineal sobre la mesa. Es la situación más simple y la más fácil de corregir.
- Si la dependencia está en **la magnitud de los residuos** (ACF de $\lvert\hat a_t\rvert$ significativa) pero no en su media, el modelo capturó bien la dirección y **no capturó la incertidumbre**. Sus predicciones puntuales pueden ser correctas mientras que sus intervalos, probabilidades y cualquier dimensionamiento de posición derivado son incorrectos. Ésta es, dada la evidencia de §2.11, la situación **esperable** en datos financieros.
- Si la dependencia aparece sólo en ciertos períodos, hay inestabilidad de régimen.

**Consecuencia adicional [B]:** residuos con dependencia temporal invalidan la inferencia estadística sobre el propio modelo — es exactamente el problema de §2.10. Los intervalos de confianza sobre métricas de desempeño calculados asumiendo observaciones independientes serán demasiado estrechos.

### [B] Implicancias potenciales para ML

- El diagnóstico de residuos **es transferible** a modelos de ML: nada en la definición de la ACF exige que $\hat\mu_t$ provenga de un modelo lineal. Es una de las herramientas más directamente reutilizables del capítulo.
- La descomposición $r_t = \hat\mu_t + \hat a_t$ sugiere una pregunta de diseño que **no debe decidirse aquí**: si la parte predecible de la media es tan pequeña como sugieren los datos de Tsay ($R^2\approx2.5\%$), ¿tiene sentido dedicar toda la capacidad del modelo a estimarla? **PREGUNTA ABIERTA** hasta tener evidencia empírica sobre futuros y hasta el Cap. 3.
- La representación lineal $r_t=\mu+\sum\psi_i a_{t-i}$ como combinación de innovaciones es el lenguaje en que se expresarán las funciones de respuesta a impulso y la propagación de shocks. Es fundamento conceptual, no herramienta operativa en esta etapa.

### Preguntas abiertas

- ¿Cómo se testea dependencia no lineal en residuos? → **Cap. 4.**
- ¿Cómo se modela la dependencia en la varianza de los residuos? → **Cap. 3.**

---

## 2.4 Sección 2.4 — Simple AR Models

### [A] Qué afirma Tsay

**Motivación empírica.** El capítulo llega al AR **porque los datos lo pidieron**: "el hecho de que el retorno mensual $r_t$ del índice CRSP value-weighted tenga una autocorrelación de lag-1 estadísticamente significativa indica que el retorno rezagado $r_{t-1}$ **podría ser útil** para predecir $r_t$".

**AR(1)** (Ec. 2.8): $r_t=\phi_0+\phi_1 r_{t-1}+a_t$, con $\{a_t\}$ white noise de media cero y varianza $\sigma_a^2$. "Este modelo tiene la misma forma que el conocido modelo de regresión lineal simple en el que $r_t$ es la variable dependiente y $r_{t-1}$ la explicativa." Se usa también en modelado de volatilidad estocástica reemplazando $r_t$ por su log-volatilidad (Caps. 3 y 12).

**Media y varianza condicionales:**
$$E(r_t\mid r_{t-1})=\phi_0+\phi_1 r_{t-1},\qquad \text{Var}(r_t\mid r_{t-1})=\text{Var}(a_t)=\sigma_a^2$$
**"Ésta es una propiedad de Markov tal que, condicional en $r_{t-1}$, el retorno $r_t$ no está correlacionado con $r_{t-i}$ para $i>1$."** Y advierte: "obviamente, hay situaciones en las que $r_{t-1}$ solo no puede determinar la esperanza condicional de $r_t$ y debe buscarse un modelo más flexible".

**AR(p)** (Ec. 2.9): $r_t=\phi_0+\phi_1r_{t-1}+\dots+\phi_p r_{t-p}+a_t$. "Este modelo dice que las $p$ variables pasadas **determinan conjuntamente** la esperanza condicional de $r_t$ dado el pasado. El modelo AR($p$) tiene la misma forma que un modelo de regresión lineal múltiple con valores rezagados como variables explicativas."

**Estacionariedad de AR(1):** condición **necesaria y suficiente** $|\phi_1|<1$. Bajo ella:
$$\mu=E(r_t)=\frac{\phi_0}{1-\phi_1},\qquad \text{Var}(r_t)=\frac{\sigma_a^2}{1-\phi_1^2}$$
Reescritura: $r_t=(1-\phi_1)\mu+\phi_1 r_{t-1}+a_t$; "**este modelo se usa a menudo en la literatura financiera con $\phi_1$ midiendo la persistencia** de la dependencia dinámica de una serie AR(1)".

**Representación como serie lineal** (Ec. 2.11): $r_t-\mu=\sum_{i=0}^{\infty}\phi_1^i a_{t-i}$, es decir $\psi_i=\phi_1^i$.

**ACF de AR(1):** $\rho_\ell=\phi_1\rho_{\ell-1}$, y como $\rho_0=1$, **$\rho_\ell=\phi_1^\ell$**: "la ACF de una serie AR(1) débilmente estacionaria **decae exponencialmente** con tasa $\phi_1$". Con $\phi_1>0$, decaimiento exponencial limpio; con $\phi_1<0$, dos decaimientos exponenciales alternantes con tasa $\phi_1^2$.

**AR(2):** ecuación de momentos $\rho_\ell=\phi_1\rho_{\ell-1}+\phi_2\rho_{\ell-2}$ (Ec. 2.13), con $\rho_1=\phi_1/(1-\phi_2)$. Ecuación característica $1-\phi_1x-\phi_2x^2=0$; los inversos de las soluciones son las **raíces características**. Si $\phi_1^2+4\phi_2<0$ las raíces son un par complejo conjugado y "el gráfico de la ACF mostraría ondas seno y coseno amortiguadas". **"En aplicaciones de negocios y economía, las raíces características complejas son importantes. Dan lugar al comportamiento de los ciclos económicos."** Longitud media de los ciclos estocásticos: $k=2\pi/\cos^{-1}[\phi_1/(2\sqrt{-\phi_2})]$.

**Ejemplo 2.1 (GNP real trimestral EE.UU., 1947.II–1991.I):** AR(3) ajustado
$$r_t=0.0047+0.348r_{t-1}+0.179r_{t-2}-0.142r_{t-3}+a_t,\quad \hat\sigma_a=0.0097$$
Factorización aproximada $(1+0.521B)(1-0.869B+0.274B^2)=0$; el segundo factor tiene raíces complejas y confirma ciclos económicos estocásticos de longitud media ≈ 10.62 trimestres (≈ 3 años).

**AR(p) general:** $E(r_t)=\phi_0/(1-\phi_1-\dots-\phi_p)$; ecuación característica $1-\phi_1x-\dots-\phi_px^p=0$; **estacionariedad requiere que todas las raíces características sean menores que 1 en módulo** (equivalentemente, todas las soluciones mayores que 1 en módulo). La ACF muestra "una mezcla de patrones de seno y coseno amortiguados y decaimientos exponenciales según la naturaleza de sus raíces características".

### [A] 2.4.2 Identificación en la práctica

**PACF.** Se define mediante una secuencia de regresiones AR de orden creciente; $\hat\phi_{\ell,\ell}$ es la PACF muestral de lag $\ell$. "La PACF de lag-2 $\hat\phi_{2,2}$ muestra **la contribución añadida** de $r_{t-2}$ a $r_t$ **por encima** del modelo AR(1)". Propiedades para un AR($p$) gaussiano estacionario:
- $\hat\phi_{p,p}\to\phi_p$ cuando $T\to\infty$;
- $\hat\phi_{\ell,\ell}\to0$ para todo $\ell>p$;
- **la varianza asintótica de $\hat\phi_{\ell,\ell}$ es $1/T$ para $\ell>p$.**

"Estos resultados dicen que, para una serie AR($p$), **la PACF muestral se corta (cuts off) en el lag $p$**."

**Ejemplo (Tabla 2.1, índice VW mensual, T = 996).** Error estándar asintótico ≈ 0.032. Al 5% se identifica un AR(3) **o** un AR(9); al 1%, un AR(3).

**Criterios de información.** Todos basados en verosimilitud.
$$\text{AIC}=\frac{-2}{T}\ln(\text{verosimilitud})+\frac{2}{T}\times(\text{n° de parámetros})$$
Para un AR($\ell$) gaussiano: $\text{AIC}(\ell)=\ln(\tilde\sigma_\ell^2)+\dfrac{2\ell}{T}$ y $\text{BIC}(\ell)=\ln(\tilde\sigma_\ell^2)+\dfrac{\ell\ln(T)}{T}$. "La penalización por cada parámetro usado es 2 para AIC y $\ln(T)$ para BIC. Así, comparado con AIC, **BIC tiende a seleccionar un modelo AR de orden menor** cuando el tamaño de muestra es moderado o grande."

**Resultado en el ejemplo, y la conclusión de Tsay, textual:** AIC alcanza su mínimo en $p=9$; BIC en $p=1$ (con $p=3$ muy cerca). **"Este ejemplo muestra que diferentes enfoques o criterios de determinación del orden pueden resultar en diferentes elecciones de $p$. No hay evidencia que sugiera que un enfoque supere al otro en una aplicación real. La información sustantiva del problema bajo estudio y la simplicidad son dos factores que también juegan un papel importante en la elección de un modelo AR."**

**Estimación.** Mínimos cuadrados condicionales, empezando en la observación $p+1$. Residuo $\hat a_t=r_t-\hat r_t$.

**Modelo ajustado al índice VW mensual:**
$$r_t=0.0091+0.116r_{t-1}-0.019r_{t-2}-0.104r_{t-3}+\hat a_t,\quad \hat\sigma_a=0.054$$
Errores estándar: 0.002, 0.032, 0.032, 0.032. Todos significativos al 1% excepto el lag-2.

**Comentario de Tsay sobre la magnitud, textual:** **"Para este ejemplo, los coeficientes AR del modelo ajustado son pequeños, indicando que la dependencia serial de la serie es débil, aunque sea estadísticamente significativa al nivel del 1%."** Y sobre la constante: $\hat\mu=0.009$, "pequeña pero con una implicancia importante de largo plazo": retorno anual simple bruto promedio ≈ 9.3%; un dólar invertido a comienzos de 1926 valdría ≈ $1,593 a fines de 2008.

**Model checking.** "Si el modelo es adecuado, entonces la serie residual debería comportarse como white noise." Se usan ACF y Ljung–Box de los residuos. **Ajuste de grados de libertad:** para un AR($p$), $Q(m)$ sigue asintóticamente una $\chi^2$ con **$m-g$** grados de libertad, donde $g$ es el número de coeficientes AR. "Nota: la mayoría de los paquetes de series temporales no ajustan los grados de libertad al aplicar Ljung–Box a una serie residual."

En el ejemplo: $Q(12)=16.35$ con p = 0.060 usando 9 df. Refinamiento eliminando el lag-2 no significativo:
$$r_t=0.0088+0.114r_{t-1}-0.106r_{t-3}+a_t,\quad\hat\sigma_a=0.0536$$
con $Q(12)=16.83$, p = 0.078. "El modelo es adecuado para modelar la dependencia lineal dinámica de los datos."

**Diagnósticos completos del ajuste (salida S-Plus reproducida por Tsay):**
- **R-Squared 0.0246**, Adjusted R-Squared 0.0216
- Durbin–Watson 1.9913
- **Jarque–Bera 1656.39, p = 0.0000**
- Ljung–Box 50.13, p = 0.0087

### [A] 2.4.3 Goodness of Fit

$R^2=1-\dfrac{\text{suma de cuadrados de residuos}}{\text{suma de cuadrados total}}$, con $0\le R^2\le1$. "Típicamente, un $R^2$ mayor indica que el modelo proporciona un ajuste más cercano a los datos."

**La advertencia crítica, textual:** **"Sin embargo, esto sólo es cierto para una serie temporal estacionaria. Para la serie no estacionaria con raíz unitaria discutida más adelante en este capítulo, el $R^2$ de un ajuste AR(1) converge a uno cuando el tamaño de muestra crece hacia infinito, independientemente del verdadero modelo subyacente de $r_t$."**

$R^2$ ajustado $=1-\hat\sigma_a^2/\hat\sigma_r^2$; "ya no está entre 0 y 1".

### [A] 2.4.4 Forecasting

**Definición del problema.** Origen de pronóstico $h$, horizonte $\ell$. $\hat r_h(\ell)$ se elige bajo **pérdida de error cuadrático mínimo**:
$$E\{[r_{h+\ell}-\hat r_h(\ell)]^2\mid F_h\}\le\min_g E[(r_{h+\ell}-g)^2\mid F_h]$$
donde $g$ es función de la información disponible en $h$.

**1 paso:** $\hat r_h(1)=E(r_{h+1}\mid F_h)=\phi_0+\sum_{i=1}^p\phi_i r_{h+1-i}$; error $e_h(1)=a_{h+1}$; $\text{Var}[e_h(1)]=\sigma_a^2$. Si $a_t$ es normal, intervalo al 95%: $\hat r_h(1)\pm1.96\sigma_a$. "$a_{t+1}$ se conoce en la literatura econométrica como el **shock** de la serie en $t+1$."

**Advertencia sobre incertidumbre de parámetros:** "En la práctica, se usan a menudo parámetros estimados... Esto resulta en un **pronóstico condicional** porque tal pronóstico **no toma en consideración la incertidumbre en las estimaciones de los parámetros**. En teoría uno puede considerar la incertidumbre de parámetros al pronosticar, pero es mucho más complicado. Una forma natural de considerar la incertidumbre de parámetros y de modelo es el pronóstico bayesiano con MCMC. Ver el Capítulo 12."

**2 pasos:** $\text{Var}[e_h(2)]=(1+\phi_1^2)\sigma_a^2\ge\text{Var}[e_h(1)]$: "a medida que el horizonte de pronóstico aumenta, **la incertidumbre del pronóstico también aumenta**".

**Multipaso:** $\hat r_h(\ell)=\phi_0+\sum_{i=1}^p\phi_i\hat r_h(\ell-i)$, calculable recursivamente. **"Puede demostrarse que para un modelo AR($p$) estacionario, $\hat r_h(\ell)$ converge a $E(r_t)$ cuando $\ell\to\infty$, lo que significa que para tal serie el pronóstico puntual de largo plazo se aproxima a su media incondicional. Esta propiedad se conoce como reversión a la media (mean reversion) en la literatura financiera."** Para AR(1), la velocidad se mide por la **vida media**: $\ell=\ln(0.5)/\ln(|\phi_1|)$. La varianza del error de pronóstico se aproxima a la varianza incondicional.

**Tabla 2.2 (VW mensual, origen $h=984$ = diciembre 2007).** Modelo reestimado con las primeras 984 obs. Los pronósticos y las desviaciones estándar **convergen rápidamente** a la media y desviación muestrales (0.0095 y 0.0540) "debido a la débil dependencia serial de la serie". Los retornos reales de 2008 incluyen −0.0786, −0.0981, **−0.1847**, −0.0852. Tsay observa: "excepto por el retorno de octubre de 2008, todos los retornos reales están dentro de los intervalos de predicción al 95%".

### [B] Significado e interpretación para el proyecto

**El AR es literalmente una regresión lineal sobre lags.** Tsay lo dice dos veces [A]. Esto tiene una consecuencia inmediata: **un modelo de ML que usa lags de retornos como features y produce una predicción de la media condicional está en la misma familia conceptual que un AR; lo que aporta de más es la posibilidad de no linealidad e interacciones.** Esa es la comparación honesta.

**El dato de $R^2 = 2.46\%$ es el ancla de realismo del capítulo [B].** Ese es el poder explicativo de la mejor estructura lineal identificada sobre 83 años de retornos mensuales de un índice, con coeficientes significativos al 1%. Cualquier resultado de ML sobre retornos que reporte un $R^2$ mucho mayor sobre datos comparables debe considerarse, en primera instancia, un indicio de error metodológico (leak, target solapado, evaluación in-sample) antes que un hallazgo. **Esto no es una afirmación de Tsay**; es la inferencia que extraigo de su ejemplo, y su validez está limitada a: retornos, frecuencia baja, índices accionarios, modelos lineales. No se traslada automáticamente a futuros ni a alta frecuencia.

**La advertencia sobre $R^2$ y raíces unitarias es una de las más útiles del capítulo [A].** Su traducción operativa **[B]**: cualquier métrica de bondad de ajuste calculada sobre una variable no estacionaria puede ser arbitrariamente alta sin contenido. Esto incluye evaluar un modelo por su capacidad de "predecir el precio" en lugar del retorno: un modelo que predice $\hat P_{t+1}=P_t$ tendrá $R^2$ altísimo sobre precios y cero información.

**La reversión a la media del pronóstico multipaso [A] tiene una lectura importante [B]:** el pronóstico óptimo de un modelo estacionario converge a la media incondicional. Es decir, **cuanto más largo el horizonte, más se parece el mejor modelo posible al benchmark trivial**. Combinado con que la varianza del error converge a la varianza incondicional, la conclusión es que en horizontes largos la ventaja de cualquier modelo lineal se desvanece por construcción. **PREGUNTA ABIERTA:** ¿ocurre lo mismo con modelos no lineales? El Cap. 4 es el lugar.

**La Tabla 2.2 ilustra involuntariamente un punto central [B].** Que "todos los retornos reales salvo uno caigan dentro del intervalo al 95%" suena a buen desempeño, pero los intervalos tienen ancho $\pm1.96\times0.054\approx\pm10.6\%$ mensual sobre pronósticos puntuales de ~1%. El intervalo es diez veces el pronóstico. **Un modelo puede estar "bien calibrado" y ser inútil para decidir**; calibración y utilidad económica son criterios distintos.

### [B] AR como benchmark: qué debe aportar un modelo complejo

Ésta es una interpretación propia. La jerarquía natural de benchmarks que se desprende del capítulo:

| Nivel | Benchmark | Qué asume | Qué debe superar un modelo complejo |
|---|---|---|---|
| 0 | **Pronóstico incondicional** $\hat r=\hat\mu$ | Estacionariedad, sin dependencia | Que exista *alguna* información condicional |
| 1 | **Random walk** ($\hat r_{t+1}=0$ en retornos; $\hat P_{t+1}=P_t$ en precios) | No predecibilidad | La hipótesis nula del Cap. 1, ahora operativa |
| 2 | **AR($p$)** con orden por AIC/BIC | Dependencia lineal en la media | Que la información no sea puramente lineal en lags propios |
| 3 | **ARMA** | Dependencia lineal parsimoniosa | Idem, con memoria de shocks |
| 4 | **Regresión con errores ARMA** (§2.9) | Dependencia lineal con exógenas | Que la información de otros mercados no sea puramente lineal |

**Advertencias sobre esta tabla [B]:**
- Los niveles 0 y 1 **casi coinciden** en retornos, porque $\hat\mu\approx0$. La distinción importa más en precios.
- Superar un benchmark **debe medirse out-of-sample y con una métrica que refleje el objetivo**, no por $R^2$ in-sample. El Capítulo 2 no dice nada sobre esto.
- Un modelo complejo que no supera al AR **no es necesariamente inútil**: puede estar capturando la misma información con más varianza. Lo relevante es si aporta información que el AR no puede representar.
- **No se propone adoptar AR como modelo final ni como componente del sistema.** Se propone usarlo como referencia de comparación, que es su función natural dado lo que el capítulo establece.

### [B] Adaptación a mercados de futuros

- La estructura de ciclos vía raíces complejas [A] es un fenómeno **macroeconómico** en el ejemplo de Tsay (GNP, ciclos de ~3 años). No hay razón en el capítulo para esperar ciclos análogos en retornos de futuros. Trasladarlo sería generalizar un ejemplo. **Lo que sí es transferible es la herramienta**: si apareciera un patrón cíclico en alguna serie derivada (volatilidad, volumen, open interest), las raíces características son la forma de caracterizarlo.
- **Persistencia $\phi_1$ [A]** es un concepto directamente útil para series **derivadas** de futuros que sí son persistentes: volatilidad realizada, spreads de calendario, open interest. Para retornos, cabe esperar $\phi_1\approx0$.
- **PACF y profundidad temporal:** la PACF responde "¿cuánto añade el lag $\ell$ sobre los $\ell-1$ anteriores, linealmente y en la media?". Es información útil pero **estrictamente limitada a esa pregunta**. Ver §8 para por qué no es un selector de lookback.

### [B] Implicancias potenciales para ML

- El **ajuste de grados de libertad en Ljung–Box** [A] tiene un análogo conceptual al aplicar el test a residuos de un modelo de ML: el test asume que los residuos provienen de un modelo con $g$ parámetros estimados. Con un modelo de ML de miles de parámetros efectivos, la corrección exacta no está definida. **PREGUNTA ABIERTA / limitación metodológica** a registrar.
- AIC y BIC son verosimilitud penalizada [A] — conecta directamente con el puente verosimilitud↔pérdida del Capítulo 1. Su uso en modelos de ML no paramétricos es problemático porque el "número de parámetros" no está bien definido. No adoptar acríticamente.
- La discrepancia AIC vs BIC en el propio ejemplo de Tsay (9 vs 1) es un recordatorio [A] de que **la selección de complejidad no tiene una respuesta única**, y que él mismo apela a "simplicidad e información sustantiva del problema".

### Preguntas abiertas

- ¿La media condicional de un futuro tiene estructura no lineal que un AR no capta? → **Cap. 4.**
- ¿La incertidumbre de parámetros importa cuantitativamente? → **Cap. 12** (bayesiano/MCMC).
- ¿Qué benchmark es el correcto cuando el objetivo es una decisión de trading y no un pronóstico puntual? → **fuera de Tsay; PREGUNTA ABIERTA del proyecto.**

---

## 2.5 Sección 2.5 — Simple MA Models

### [A] Qué afirma Tsay

**La conexión con microestructura, textual y en la primera línea de la sección:** **"Como se muestra en el Capítulo 5, el bid–ask bounce en el trading de acciones puede introducir una estructura MA(1) en una serie de retornos."**

**Derivación.** Tsay introduce el MA como un AR de orden infinito con restricciones: partiendo de $r_t=\phi_0+\phi_1r_{t-1}+\phi_2r_{t-2}+\dots+a_t$ (irrealizable por tener infinitos parámetros), impone $\phi_i=-\theta_1^i$, lo que exige $|\theta_1|<1$ para que la serie no explote. Manipulando, obtiene
$$r_t=c_0+a_t-\theta_1a_{t-1}\qquad\text{(MA(1), Ec. 2.20)}$$
"excepto por el término constante, $r_t$ es un promedio ponderado de los shocks $a_t$ y $a_{t-1}$". General (Ec. 2.22): $r_t=c_0+a_t-\theta_1a_{t-1}-\dots-\theta_qa_{t-q}$.

**Estacionariedad.** **"Los modelos de media móvil son SIEMPRE débilmente estacionarios porque son combinaciones lineales finitas de una secuencia de white noise para la cual los dos primeros momentos son invariantes en el tiempo."** $E(r_t)=c_0$ (el término constante **es** la media) y $\text{Var}(r_t)=(1+\theta_1^2+\dots+\theta_q^2)\sigma_a^2$.

**ACF.** Para MA(1): $\rho_1=\dfrac{-\theta_1}{1+\theta_1^2}$, $\rho_\ell=0$ para $\ell>1$. **"La ACF de un modelo MA(1) se corta (cuts off) en el lag 1."** Para MA($q$): $\rho_q\neq0$ pero $\rho_\ell=0$ para $\ell>q$. **"Consecuentemente, una serie MA($q$) está linealmente relacionada sólo con sus primeros $q$ valores rezagados y por tanto es un modelo de 'memoria finita'."**

**Invertibilidad.** Reescribiendo $a_t=r_t+\theta_1r_{t-1}+\theta_1^2r_{t-2}+\dots$: "intuitivamente, $\theta_1^j$ debería ir a cero cuando $j$ crece porque el retorno remoto $r_{t-j}$ debería tener muy poco impacto sobre el shock actual". Por tanto se requiere $|\theta_1|<1$ para que el MA(1) sea **invertible**; si $|\theta_1|=1$, es no invertible.

**Identificación del orden:** "La ACF es útil para identificar el orden de un modelo MA. Si $\rho_q\neq0$ pero $\rho_\ell=0$ para $\ell>q$, entonces $r_t$ sigue un modelo MA($q$)." Ejemplo: índice equal-weighted mensual con ACF significativa en lags 1, 3 y 9 → MA(9).

**Pronóstico.** "Como el modelo tiene memoria finita, sus pronósticos puntuales van a la media de la serie rápidamente." Para MA(1): $\hat r_h(1)=c_0-\theta_1a_h$ y $\hat r_h(\ell)=c_0$ para $\ell\ge2$. **"Para modelos MA(1), la reversión a la media toma sólo un período."** En general, para MA($q$), los pronósticos multipaso van a la media después de los primeros $q$ pasos.

**Resumen que ofrece Tsay** al cerrar la sección:
- Para modelos MA, la **ACF** es útil para especificar el orden porque se corta en el lag $q$.
- Para modelos AR, la **PACF** es útil porque se corta en el lag $p$.
- Una serie MA es **siempre** estacionaria; para que una AR lo sea, todas sus raíces características deben ser menores que 1 en módulo.
- Para una serie estacionaria, los pronósticos multipaso convergen a la media y las varianzas de los errores de pronóstico convergen a la varianza de la serie.

### [B] Diferencia conceptual AR vs MA

| | AR($p$) | MA($q$) |
|---|---|---|
| Depende de | **Valores pasados observables** de la serie | **Shocks pasados no observables** |
| Memoria | Infinita (decae exponencialmente) | **Finita**: exactamente $q$ períodos |
| ACF | Decae | **Se corta en $q$** |
| PACF | **Se corta en $p$** | Decae |
| Estacionariedad | Condicional a raíces características | **Siempre** |
| Interpretación | El pasado del precio contiene información | El pasado de las *sorpresas* contiene información |
| Reversión a la media del pronóstico | Gradual, gobernada por $\phi$ | Inmediata tras $q$ pasos |

**Lectura para el proyecto [B]:** la distinción operativa relevante es que **los shocks $a_t$ no son observables** — se estiman como residuos de un modelo. Un feature basado en "el shock de ayer" es un objeto derivado de un modelo, no un dato. Esto introduce una dependencia de especificación que un feature basado en retornos rezagados no tiene.

### [B] La cuestión central: estructura económica vs artefacto microestructural

Tsay afirma [A] que el bid–ask bounce **puede introducir** una estructura MA(1) en los retornos, y remite al Cap. 5 para el desarrollo. También afirma [A, §2.2] que "la manera en que se determinan los precios y se calculan los retornos de los índices podría introducir autocorrelaciones". No desarrolla ninguna de las dos cosas en este capítulo.

Sin adelantar el Cap. 5, lo que queda establecido es la **existencia de al menos tres orígenes posibles** para una estructura temporal detectable:

1. **Propiedad económica.** Información que se incorpora gradualmente al precio, ajuste lento, flujos con inercia. Potencialmente explotable.
2. **Artefacto de medición o microestructura.** El precio observado no es "el" precio sino una realización del proceso de transacción. Genera dependencia estadística real en la serie observada que **no corresponde a información sobre el valor**, y que típicamente no es explotable porque explotarla requeriría operar a precios que no están disponibles.
3. **Combinación.** El caso realista: una parte de cada uno, en proporciones que dependen de la frecuencia y del instrumento.

**Consecuencia metodológica [B]:** detectar una estructura MA(1) en los retornos de un futuro a alta frecuencia **no es, por sí solo, evidencia de señal**. La pregunta que hay que hacerse antes es: *¿esta estructura sobrevive si mido los retornos con precios mid en lugar de precios de transacción? ¿Cambia con la frecuencia de muestreo? ¿Su magnitud es del orden de un tick?* Ninguna de estas preguntas se responde con el Capítulo 2. **PREGUNTA ABIERTA → Cap. 5.**

**Nota específica de futuros [B]:** el mecanismo de bid–ask bounce descrito por Tsay para acciones aplica a cualquier instrumento con spread cotizado, incluidos los futuros. Pero los futuros líquidos de índices operan típicamente con spread de un tick durante la sesión activa, lo que **acota la magnitud** del efecto a un orden de $\text{tick}/P$. Esta acotación es una expectativa, no un resultado.

### [B] Implicancias potenciales para ML

- Un MA(1) con $\theta_1$ pequeño en retornos de alta frecuencia es un **candidato a hallazgo espurio de primer orden**. Debería tratarse como una hipótesis a descartar antes que como señal.
- Los residuos $\hat a_t$ de un modelo pueden usarse como features ("sorpresa de ayer"), pero son estimados y dependen del modelo generador. Es una posibilidad, no una recomendación.
- La reversión inmediata a la media de los pronósticos MA [A] refuerza el punto de §2.4: en horizontes más allá de la memoria del modelo, el pronóstico óptimo es el benchmark trivial.

---

## 2.6 Sección 2.6 — Simple ARMA Models

### [A] Qué afirma Tsay

**Motivación: parsimonia.** "En algunas aplicaciones, los modelos AR o MA se vuelven engorrosos porque uno puede necesitar un modelo de orden alto con muchos parámetros para describir adecuadamente la estructura dinámica de los datos. Para superar esta dificultad se introducen los modelos ARMA... combina las ideas de los modelos AR y MA en una forma compacta de modo que el número de parámetros usado se mantiene pequeño, **logrando parsimonia en la parametrización**."

**La afirmación decisiva para el proyecto, textual:** **"Para las series de retornos en finanzas, la probabilidad de usar modelos ARMA es baja. Sin embargo, el concepto de modelos ARMA es altamente relevante en el modelado de volatilidad. De hecho, el modelo GARCH puede considerarse un modelo ARMA, si bien no estándar, para la serie $a_t^2$; ver el Capítulo 3 para detalles."**

**ARMA(1,1)** (Ec. 2.25): $r_t-\phi_1r_{t-1}=\phi_0+a_t-\theta_1a_{t-1}$. "Para que este modelo sea significativo, necesitamos $\phi_1\neq\theta_1$; de lo contrario hay una cancelación en la ecuación y el proceso se reduce a una serie de white noise."

**Propiedades.** $E(r_t)=\phi_0/(1-\phi_1)$, **exactamente igual que en AR(1)**. $\text{Var}(r_t)=\dfrac{(1-2\phi_1\theta_1+\theta_1^2)\sigma_a^2}{1-\phi_1^2}$, lo que requiere $|\phi_1|<1$: **"de nuevo, ésta es precisamente la misma condición de estacionariedad que la del modelo AR(1)"**.

**ACF y PACF.** $\rho_1=\phi_1-\dfrac{\theta_1\sigma_a^2}{\gamma_0}$ y $\rho_\ell=\phi_1\rho_{\ell-1}$ para $\ell>1$. "La ACF de un modelo ARMA(1,1) se comporta muy parecido a la de un AR(1) **excepto que el decaimiento exponencial empieza en el lag 2**. Consecuentemente, la ACF de un ARMA(1,1) **no se corta en ningún lag finito**." La PACF tampoco se corta.

**Identificación:** **"La ACF y la PACF no son informativas para determinar el orden de un modelo ARMA."** Tsay y Tiao (1984) proponen la **EACF** (extended autocorrelation function).

**ARMA($p,q$) general** (Ec. 2.28): $(1-\phi_1B-\dots-\phi_pB^p)r_t=\phi_0+(1-\theta_1B-\dots-\theta_qB^q)a_t$. Se exige que no haya factores comunes entre los polinomios AR y MA. Si todas las soluciones de la ecuación característica son menores que 1 en valor absoluto, el modelo es débilmente estacionario.

### [A] 2.6.5 Tres representaciones — la parte conceptualmente más valiosa

**1. Representación ARMA** (Ec. 2.28): compacta, útil para **estimación de parámetros** y para calcular pronósticos multipaso recursivamente.

**2. Representación AR** (Ec. 2.31): $r_t=\dfrac{\phi_0}{1-\theta_1-\dots-\theta_q}+\pi_1r_{t-1}+\pi_2r_{t-2}+\dots+a_t$. "Muestra la dependencia del retorno actual $r_t$ sobre los retornos pasados." Los $\pi_i$ son los **pesos $\pi$**. **Invertibilidad:** el modelo es invertible si los $\pi_i$ decaen a cero; condición suficiente: todos los ceros del polinomio $\theta(B)$ mayores que 1 en módulo. Para un AR puro, $\pi_i=0$ para $i>p$: siempre invertible.

**3. Representación MA** (Ec. 2.32): $r_t=\mu+a_t+\psi_1a_{t-1}+\psi_2a_{t-2}+\dots$. "Muestra explícitamente el impacto del shock pasado $a_{t-i}$ sobre el retorno actual." Los $\psi_i$ son la **función de respuesta a impulso** (impulse response function). "Para una serie débilmente estacionaria, los coeficientes $\psi_i$ decaen exponencialmente... **Así, para un modelo ARMA estacionario, el shock $a_{t-i}$ no tiene un impacto permanente sobre la serie.**"

**Utilidad de la representación MA:** da la varianza del error de pronóstico (Ec. 2.34):
$$\text{Var}[e_h(\ell)]=(1+\psi_1^2+\dots+\psi_{\ell-1}^2)\sigma_a^2$$
"que, como es de esperar, es una función no decreciente del horizonte $\ell$". Y da la prueba simple de la reversión a la media: la estacionariedad implica $\psi_i\to0$, luego $\hat r_h(\ell)\to\mu$. **"La velocidad con la que $\hat r_h(\ell)$ se aproxima a $\mu$ determina la velocidad de reversión a la media."**

### [B] Interpretación: qué es realmente ARMA para este proyecto

**No es un candidato a modelo predictivo.** Tsay lo dice explícitamente [A]. Es la afirmación más útil de la sección y evita perder tiempo.

**Es cuatro cosas, en este orden de importancia para el proyecto [B]:**

1. **El puente hacia el Capítulo 3.** La equivalencia GARCH ↔ ARMA sobre $a_t^2$ [A] significa que todo el vocabulario aprendido aquí —persistencia, raíces características, estacionariedad, decaimiento de ACF, reversión a la media— **se reutilizará aplicado a la volatilidad**, donde hay mucha más estructura que en la media. Aprender ARMA es, en la práctica, aprender el lenguaje de GARCH.

2. **Un lenguaje para describir dependencia temporal.** Las tres representaciones dan tres preguntas distintas y bien definidas sobre cualquier proceso: ¿cómo depende del pasado observable ($\pi$)? ¿cómo se propagan los shocks ($\psi$)? ¿cómo se estima compactamente? Estas preguntas son aplicables fuera del marco ARMA.

3. **Un benchmark.** Con la salvedad de que, dado lo que Tsay afirma sobre su baja utilidad en retornos, un ARMA probablemente no sea un benchmark más exigente que un AR simple.

4. **El concepto de función de respuesta a impulso [A]**, que es la forma correcta de plantear la pregunta "¿cuánto dura el efecto de un evento?". **[B]** Es directamente relevante para preguntas de futuros como: ¿cuánto dura el efecto de un dato macro sobre la dinámica del mercado? Pero responderla requiere el marco multivariado. **PREGUNTA ABIERTA → Cap. 8.**

**La condición $\phi_1\neq\theta_1$ [A] tiene una moraleja general [B]:** un modelo sobreparametrizado puede contener componentes que se cancelan, produciendo un ajuste que parece rico y equivale a ruido blanco. Es la versión analítica del problema de identificabilidad, y su análogo en ML es la sobreparametrización con componentes redundantes.

**Que ACF y PACF no identifiquen el orden de un ARMA [A] es una advertencia transferible [B]:** las herramientas de diagnóstico gráfico tienen alcance limitado incluso dentro de la familia lineal. Extenderlas por analogía a la selección de arquitecturas de ML es injustificado.

### Preguntas abiertas

- ¿Cuál es la forma exacta de la equivalencia GARCH ↔ ARMA($a_t^2$)? → **Cap. 3.**
- ¿La persistencia de la volatilidad es tan alta que roza la no estacionariedad (IGARCH)? → **Cap. 3.**

---

## 2.7 Sección 2.7 — Unit-Root Nonstationarity

### [A] Qué afirma Tsay

**Planteo.** "Hasta ahora nos hemos enfocado en series de retornos que son estacionarias. En algunos estudios, las tasas de interés, los tipos de cambio o **la serie de precios de un activo** son de interés. **Estas series tienden a ser no estacionarias.** Para una serie de precios, la no estacionariedad se debe principalmente al hecho de que **no hay un nivel fijo para el precio**." El ejemplo canónico es el random walk.

**Random walk** (Ec. 2.35): $p_t=p_{t-1}+a_t$. "Si tratamos el modelo de random walk como un modelo AR(1) especial, entonces el coeficiente de $p_{t-1}$ es la unidad, que **no satisface la condición de estacionariedad débil**. Una serie de random walk es, por tanto, no estacionaria, y la llamamos serie temporal no estacionaria con raíz unitaria."

**Consecuencias, todas [A]:**
- "El modelo de random walk ha sido ampliamente considerado como modelo estadístico para el movimiento de los precios logarítmicos de acciones. **Bajo tal modelo, el precio de la acción no es predecible ni revierte a la media.**"
- Pronóstico: $\hat p_h(\ell)=p_h$ **para todo horizonte** $\ell>0$. "Tal pronóstico no tiene valor práctico."
- Representación MA: $p_t=a_t+a_{t-1}+a_{t-2}+\dots$, con **$\psi_i=1$ para todo $i$**.
- $\text{Var}[e_h(\ell)]=\ell\sigma_a^2$, que **diverge a infinito** cuando $\ell\to\infty$. "La longitud de un intervalo de pronóstico de $p_{h+\ell}$ se aproximará a infinito a medida que el horizonte aumenta. Este resultado dice que **la utilidad del pronóstico puntual disminuye a medida que $\ell$ crece**, lo que de nuevo implica que el modelo no es predecible."
- "La varianza incondicional de $p_t$ es no acotada... teóricamente, esto significa que $p_t$ puede asumir cualquier valor real para un $t$ suficientemente grande. **Para el log-precio $p_t$ de una acción individual esto es plausible. Sin embargo, para índices de mercado, un log-precio negativo es muy raro, si es que ocurre. En este sentido, la adecuación de un modelo de random walk para índices de mercado es cuestionable.**"
- "El impacto de cualquier shock pasado $a_{t-i}$ sobre $p_t$ **no decae con el tiempo**. Consecuentemente, la serie tiene **memoria fuerte** porque recuerda todos los shocks pasados. En economía se dice que los shocks tienen un **efecto permanente** sobre la serie."
- "La memoria fuerte de una serie con raíz unitaria puede verse en la ACF muestral de la serie observada. **Las ACF muestrales se aproximan todas a 1 a medida que el tamaño de muestra crece.**"

**Random walk with drift** (Ec. 2.36): $p_t=\mu+p_{t-1}+a_t$, con $\mu=E(p_t-p_{t-1})$. Desarrollando:
$$p_t=t\mu+p_0+a_t+a_{t-1}+\dots+a_1$$
"El log-precio consiste en una **tendencia temporal $t\mu$** y un proceso de random walk puro $\sum a_i$." Como $\text{Var}(\sum_{i=1}^t a_i)=t\sigma_a^2$, la desviación estándar condicional de $p_t$ es $\sqrt t\sigma_a$, **"que crece a una tasa menor que la esperanza condicional de $p_t$"**. Por tanto el gráfico de $p_t$ contra $t$ muestra una tendencia de pendiente $\mu$. "$\mu$ positivo implica que el log-precio eventualmente va a infinito."

**Ejemplo 3M (log-retornos mensuales, feb-1946 a dic-2008):** sin correlación serial significativa (según EACF), de modo que $r_t=0.0103+a_t$ con $\hat\sigma_a=0.0637$; error estándar de la media 0.0023; t = 4.44, p ≈ 0. **"La media de los log-retornos mensuales de 3M es, por tanto, significativamente distinta de cero al nivel del 1%."**

**Interpretación del término constante — sección propia [A]:**
- Para un **MA($q$)**, el término constante es simplemente **la media de la serie**.
- Para un **AR($p$)/ARMA($p,q$) estacionario**, se relaciona con la media vía $\mu=\phi_0/(1-\phi_1-\dots-\phi_p)$.
- Para un **random walk con drift**, el término constante **se convierte en la pendiente temporal de la serie**.
- "Estas diferentes interpretaciones para el término constante en un modelo de series temporales **destacan claramente la diferencia entre modelos dinámicos y modelos de regresión lineal usuales**."
- Otra diferencia: en $r_t=\phi_0+\phi_1r_{t-1}+a_t$, "para que el modelo AR(1) sea significativo, el coeficiente $\phi_1$ debe satisfacer $|\phi_1|\le1$. Sin embargo, el coeficiente $\beta_1$ [de una regresión $y_t=\beta_0+\beta_1x_t+a_t$] **puede asumir cualquier número real fijo**."

**2.7.3 Trend-stationary:** $p_t=\beta_0+\beta_1t+r_t$, con $r_t$ estacionaria. "Aquí $p_t$ crece linealmente en el tiempo a tasa $\beta_1$ y por tanto **puede exhibir un comportamiento similar al de un random walk con drift. Sin embargo, hay una diferencia mayor entre los dos modelos**":

| | Random walk con drift | Trend-stationary |
|---|---|---|
| $E(p_t)$ | $p_0+\mu t$ — depende del tiempo | $\beta_0+\beta_1 t$ — depende del tiempo |
| $\text{Var}(p_t)$ | $t\sigma_a^2$ — **depende del tiempo** | $\text{Var}(r_t)$ — **finita e invariante en el tiempo** |
| Efecto de los shocks | Permanente | Transitorio |
| Cómo se estacionariza | **Diferenciando** | **Removiendo la tendencia** por regresión lineal |

**2.7.4 Modelos generales con raíz unitaria — ARIMA.** "Si uno extiende el modelo [ARMA] permitiendo que el polinomio AR tenga a 1 como raíz característica, entonces el modelo se convierte en el conocido modelo ARIMA. Un modelo ARIMA se dice no estacionario con raíz unitaria porque su polinomio AR tiene una raíz unitaria. Como un modelo de random walk, un modelo ARIMA tiene **memoria fuerte** porque los coeficientes $\psi_i$ en su representación MA no decaen a cero, lo que implica que el shock pasado $a_{t-i}$ tiene un **efecto permanente** sobre la serie."

**Diferenciación.** $y_t$ es ARIMA($p,1,q$) si $c_t=y_t-y_{t-1}=(1-B)y_t$ sigue un ARMA($p,q$) estacionario e invertible. Y textualmente: **"En finanzas, se cree comúnmente que las series de precios son no estacionarias, pero que la serie de log-retornos $r_t=\ln(P_t)-\ln(P_{t-1})$ es estacionaria. En este caso, la serie de log-precios es no estacionaria con raíz unitaria y por tanto puede tratarse como un proceso ARIMA."** También menciona la posibilidad de raíces unitarias múltiples (segunda diferencia).

### [A] 2.7.5 Unit-Root Test

**Dickey–Fuller.** Modelos (2.38) $p_t=\phi_1p_{t-1}+e_t$ y (2.39) $p_t=\phi_0+\phi_1p_{t-1}+e_t$. **Hipótesis: $H_0:\phi_1=1$ vs $H_a:\phi_1<1$.** El estadístico es el t-ratio del estimador LS:
$$\text{DF}=\frac{\hat\phi_1-1}{\text{std}(\hat\phi_1)}$$

**La distribución.** "Si $\{e_t\}$ es una serie de white noise con momentos finitos de orden ligeramente mayor que 2, entonces el estadístico DF **converge a una función del movimiento browniano estándar** cuando $T\to\infty$." Consecuencias que Tsay detalla:
- Si $\phi_0$ es cero pero se emplea la Ec. (2.39) de todos modos, el t-ratio converge a **otra distribución asintótica no estándar**.
- **"En cualquier caso, se usa simulación para obtener los valores críticos de los estadísticos de test"** (Fuller, 1976, Cap. 8).
- "Sin embargo, si $\phi_0\neq0$ y se usa la Ec. (2.39), entonces el t-ratio para testear $\phi_1=1$ es **asintóticamente normal**. Pero **se necesitan tamaños de muestra grandes** para que la distribución asintótica normal se cumpla."

**ADF** (Ec. 2.40): $x_t=c_t+\beta x_{t-1}+\sum_{i=1}^{p-1}\phi_i\Delta x_{t-i}+e_t$, donde **$c_t$ es una función determinista del índice temporal**: "en la práctica, $c_t$ puede ser cero, o una constante, o $c_t=\omega_0+\omega_1t$". Estadístico $=(\hat\beta-1)/\text{std}(\hat\beta)$. Reparametrización equivalente con $\beta_c=\beta-1$: $H_0:\beta_c=0$ vs $H_a:\beta_c<0$.

**Ejemplo 2.2 (log GDP trimestral EE.UU., 1947.I–2008.IV).** Se elige $p=10$ basándose en la PACF muestral de la serie diferenciada. ADF = **−1.701**, p = **0.4297**: "indicando que **la hipótesis de raíz unitaria no puede rechazarse**". Y: **"Se usan también otros valores de $p$, pero no alteran la conclusión del test."** $\hat\beta=1-0.0008=0.9992$.

**Ejemplo S&P 500 (log del índice diario, 3-ene-1950 a 16-abr-2008, 14,462 observaciones).** Se usa $c_t=\omega_0+\omega_1t$ y $p=15$ (basado en la PACF de la primera diferencia). Estadístico = **−1.998**, p = **0.602**. "Así, **la hipótesis de raíz unitaria no puede rechazarse a ningún nivel de significancia razonable**. El término constante es estadísticamente significativo, mientras que la estimación de la tendencia temporal **no** lo es al nivel usual del 5%... es significativa al nivel del 10%. **En resumen, para el período de enero 1950 a abril 2008, la serie logarítmica del índice S&P 500 contiene una raíz unitaria y un drift positivo, pero no hay evidencia fuerte de una tendencia temporal.**" (Con $p=2$: estadístico −2.0179, p = 0.5708. En la regresión ADF, $R^2=0.0081$.)

### [B] Por qué $P_t$ puede ser no estacionario y $r_t$ no — con sus condiciones

El argumento **[A]** es: si $p_t$ es ARIMA($p,1,q$), entonces por definición su primera diferencia es un ARMA estacionario. Y como $r_t=p_t-p_{t-1}$ **es exactamente** la primera diferencia del log-precio [Cap. 1, Ec. 1.6], la diferenciación que estacionariza el log-precio **produce precisamente los log-retornos**. Ésta es la conexión más elegante entre los dos capítulos: la transformación que el Capítulo 1 justificaba por comparabilidad y tratabilidad estadística es, en el lenguaje del Capítulo 2, la operación que remueve la raíz unitaria.

**Condiciones bajo las que esto vale, que no deben omitirse [B]:**
1. Que el log-precio tenga **exactamente una** raíz unitaria. Con dos, la primera diferencia sigue siendo no estacionaria [A menciona el caso].
2. Que la no estacionariedad sea del tipo **raíz unitaria** y no **trend-stationary**. Si fuera trend-stationary, la transformación correcta sería remover la tendencia, no diferenciar [A, §2.7.3]. **Sobrediferenciar una serie trend-stationary introduce una estructura MA no invertible** — esto último es [B], no está en Tsay, pero se sigue de que diferenciar $r_t$ estacionario introduce un factor $(1-B)$ en el polinomio MA.
3. Que la estacionariedad de los retornos sea **débil**, es decir, referida a los dos primeros momentos. No garantiza estabilidad de la distribución completa.
4. Que se cumpla **durante el período muestral**. No es una propiedad eterna.

**Por eso no debe formularse como regla universal.** Tsay mismo usa el lenguaje "se cree comúnmente" [A].

### [B] "No rechazar" ≠ "demostrar" — el punto que el proyecto pidió aclarar

**Lo que el test hace [A]:** ADF plantea $H_0$: existe raíz unitaria. Un p-valor alto significa que los datos **no aportan evidencia suficiente contra** $H_0$. Nótese la asimetría estructural: la raíz unitaria está en la **nula**, de modo que la ausencia de evidencia se convierte en "no rechazo", nunca en confirmación.

**Formulación correcta [B]:** *no rechazar la raíz unitaria significa que la serie es compatible con un random walk, y también con un AR(1) de $\phi_1=0.999$, y con un AR(1) de $\phi_1=0.98$ si la muestra es corta.* El test no distingue entre ellos.

**Tres razones por las que el no rechazo es débil:**

1. **Potencia baja contra alternativas cercanas [B].** Un proceso con $\phi_1=0.99$ es estacionario pero prácticamente indistinguible de un random walk en muestras finitas. Los tests de raíz unitaria son notoriamente poco potentes en esta región. **Tsay no discute la potencia del test**; es una extensión, aunque bien establecida en la literatura econométrica. Nótese que el propio ejemplo de Tsay tiene $\hat\beta=0.9992$ [A]: un valor que es indistinguible de 1 y que, si fuera el verdadero, describiría una serie estacionaria con una vida media enorme.

2. **Sensibilidad a la especificación determinista [A parcialmente].** Tsay muestra que $c_t$ puede ser cero, constante, o constante más tendencia, y que la distribución asintótica del estadístico **cambia según lo que se incluya**, hasta el punto de que en un caso es no estándar y en otro es asintóticamente normal. La elección de $c_t$ no es inocua: es una decisión del analista que afecta la conclusión. Tsay lo trata con cuidado —en el S&P incluye tendencia y luego reporta que la tendencia no es significativa al 5%— pero no advierte explícitamente sobre el riesgo de elegir la especificación que da el resultado deseado. **Esa advertencia es [B].**

3. **Cambios estructurales [B].** Una serie estacionaria con un cambio de nivel en el medio de la muestra puede producir un no rechazo de raíz unitaria. **Tsay no trata este problema en el Capítulo 2**; queda como extensión y como limitación conocida a registrar.

**Una observación de honestidad sobre el propio texto [B]:** Tsay escribe con cuidado "la hipótesis de raíz unitaria no puede rechazarse", pero luego resume: "la serie logarítmica del índice S&P 500 **contiene** una raíz unitaria y un drift positivo". Esa frase de resumen es más fuerte que lo que el test autoriza. Es una forma de hablar habitual en econometría aplicada, pero conviene registrarla: **incluso en un texto cuidadoso, la conclusión se desliza de "no rechazado" a "contiene".** Ese deslizamiento es exactamente el que hay que evitar en el proyecto.

### [B] Adaptación a mercados de futuros

- **Precios de futuros:** no cabe esperar estacionariedad, por la misma razón que da Tsay para acciones (no hay nivel fijo) [A], más razones propias de futuros: el precio incorpora cost of carry y el contrato vence. Una serie continua ajustada añade un problema adicional —el nivel es artificial— que hace que la interpretación de un test de raíz unitaria sobre ella sea ambigua. **[B]**
- **Qué serie testear, entonces [B]:** un test ADF sobre una serie back-adjusted por diferencias no está testeando el precio de ningún instrumento real. Si se quisiera testear, lo mínimo sería hacerlo sobre precios de un contrato individual dentro de su vida activa, o sobre la serie ratio-ajustada. Esto conecta directamente con la corrección 6 del informe del Capítulo 1.
- **Series de futuros donde la pregunta de raíz unitaria es genuinamente interesante [B]:** spreads de calendario, diferenciales entre instrumentos relacionados (ES−NQ, ZN−ZB), bases spot-futuro. Éstas son las candidatas naturales a ser estacionarias mientras sus componentes no lo son. **Pero eso es cointegración, y es PREGUNTA ABIERTA → Cap. 8.** El Capítulo 2 sólo entrega el aparato univariado; el propio Tsay, al encontrar residuos con raíz unitaria en la regresión de tasas de interés, remite al Cap. 8 [A, §2.9].
- **Trend-stationary vs raíz unitaria en futuros [B]:** la distinción tiene consecuencias operativas concretas. Bajo trend-stationary, las desviaciones respecto de la tendencia son transitorias y hay reversión; bajo raíz unitaria con drift, no la hay. Es la diferencia entre "hay mean reversion alrededor de una tendencia" y "no la hay". **Distinguirlas empíricamente es difícil, y el Capítulo 2 muestra por qué: ambas producen gráficos con aspecto de tendencia.**

### [B] Implicancias potenciales para ML

- **La advertencia de $R^2$ [A, §2.4.3] se conecta aquí y es la implicancia más importante:** evaluar un modelo sobre una variable con raíz unitaria puede producir métricas excelentes y vacías. Esto no es exclusivo del $R^2$: cualquier métrica que compare $\hat y$ con $y$ en niveles hereda el problema.
- **Los tests de raíz unitaria son diagnóstico, no criterio de diseño [B].** Que ADF rechace o no rechace no determina qué transformación usar; informa. La decisión debería basarse en qué representación produce una relación estable y evaluable, y esa es una cuestión empírica del proyecto.
- **La distinción entre efecto permanente y transitorio de los shocks [A]** es conceptualmente relevante para diseñar features con ventana: si los shocks son permanentes, una feature de "distancia al máximo de $N$ días" mide algo que no revierte; si son transitorios, mide una desviación que sí revierte. La interpretación del feature depende del régimen de estacionariedad de la serie subyacente.

### Preguntas abiertas

- ¿Existen combinaciones lineales estacionarias entre futuros no estacionarios? → **Cap. 8 (cointegración).**
- ¿Cómo tratar cambios estructurales que confunden a los tests de raíz unitaria? → **fuera de Tsay Cap. 2; parcialmente Caps. 11–12.**
- ¿La aparente raíz unitaria en volatilidad (IGARCH) es real o artefacto? → **Cap. 3.**

---

## 2.8 Sección 2.8 — Seasonal Models

### [A] Qué afirma Tsay

**Definición y ejemplo.** "Algunas series financieras, como las ganancias trimestrales por acción de una compañía, exhiben cierto comportamiento cíclico o periódico. Tal serie se llama serie temporal estacional." Ejemplo central: ganancias trimestrales de Johnson & Johnson, 1960–1980, periodicidad 4. Con datos mensuales, periodicidad 12.

**Mención directa de futuros [A]:** **"Los modelos de series temporales estacionales son también útiles en la valuación de derivados relacionados con el clima y de futuros de energía, porque la mayoría de las series temporales medioambientales exhiben un comportamiento estacional fuerte."** Es la única mención explícita de futuros en el capítulo.

**Dos enfoques.** "En algunas aplicaciones, la estacionalidad es de importancia secundaria y se remueve de los datos, resultando en una serie ajustada estacionalmente... En otras aplicaciones como el pronóstico, **la estacionalidad es tan importante como otras características de los datos y debe manejarse en consecuencia. Como el pronóstico es un objetivo mayor del análisis de series financieras, nos enfocamos en este último enfoque.**"

**Transformación logarítmica [A], con dos razones explícitas:** "Tomamos la transformación logarítmica por dos razones. Primero, se usa para manejar el crecimiento exponencial de la serie... Segundo, la transformación se usa para **estabilizar la variabilidad** de la serie."

**Diferenciación estacional.** Para una serie con periodicidad $s$: $\Delta_s y_t=y_t-y_{t-s}=(1-B^s)y_t$. La diferencia convencional $\Delta y_t=y_t-y_{t-1}$ se llama **diferenciación regular**. En el ejemplo, la ACF de $\Delta x_t$ "es fuerte cuando el lag es múltiplo de la periodicidad 4. **Éste es un comportamiento bien documentado de la ACF muestral de una serie temporal estacional.**"

**Modelo multiplicativo ("airline model", Ec. 2.41):**
$$(1-B^s)(1-B)x_t=(1-\theta B)(1-\Theta B^s)a_t$$
con $|\theta|<1$, $|\Theta|<1$. "La parte AR del modelo consiste simplemente en las diferencias regular y estacional, mientras que la parte MA involucra dos parámetros." "Ha resultado ser ampliamente aplicable en el modelado de series estacionales."

### [B] La distinción que el proyecto pidió: determinística vs estocástica

Tsay **no hace explícitamente** esta distinción en la sección; presenta el enfoque estocástico (diferenciación estacional y modelos multiplicativos) porque su ejemplo lo requiere. La siguiente separación es [B]:

**A. Estacionalidad determinística.** El patrón está asociado a una etiqueta observable y conocida de antemano: hora del día, día de la semana, mes, proximidad a un evento de calendario. Se modela con **variables indicadoras** en una regresión. Nótese que **Tsay sí usa este enfoque**, pero lo ubica en otro lugar: los Ejercicios 2.7–2.9 [A] piden estudiar los **efectos de día de la semana** sobre retornos de índices y de IBM usando un modelo de regresión con indicadores M/T/W/H/F, verificar los residuos con $Q(12)$, aplicar el estimador HAC y, si hace falta, construir una regresión con errores de series temporales.

**B. Dependencia estocástica estacional.** El patrón es una propiedad de la estructura de dependencia: $\text{Corr}(r_t,r_{t-s})\neq0$ con $s$ la periodicidad, o incluso una raíz unitaria estacional que requiere $\Delta_s$. Se modela con AR/MA estacionales.

**Diferencias operativas [B]:**

| | Determinística | Estocástica |
|---|---|---|
| El patrón se conoce | **Antes** de observar la serie | Se descubre en la ACF |
| Se representa con | Indicadores / variables de calendario | Operadores $B^s$, $\Delta_s$ |
| Es un feature | **Sí, y trivialmente disponible en $t$** | No directamente |
| Riesgo principal | Data snooping sobre muchas etiquetas posibles | Sobrediferenciación |
| Herramienta de Tsay | Regresión + HAC (Ejercicios 2.7–2.9) | §2.8 |

### [B] Qué preguntar antes de tratar un patrón intradía como estacionalidad modelable

**No se concluye que un mercado intradía requiera SARIMA.** El ejemplo de Tsay es de ganancias trimestrales corporativas, un fenómeno con una causa institucional clara (ciclo contable). Trasladar el modelo a datos intradía de futuros sería generalizar un ejemplo. Las preguntas previas [B]:

1. **¿El patrón tiene un mecanismo causal identificable?** Apertura de sesiones, horarios de datos macro, subastas, cierres de otros mercados, expiración de opciones. Un patrón con mecanismo es mucho más creíble que uno hallado en la ACF.
2. **¿Está en la media o en la varianza?** Ésta es la pregunta más importante. Es plausible que la **volatilidad** intradía de futuros tenga un patrón fuerte (forma de U) y que la **media** no tenga ninguno. Son fenómenos distintos con implicancias distintas: el primero afecta normalización, sizing y evaluación; el segundo sería señal direccional. Confundirlos es un error grave. **PREGUNTA ABIERTA → Cap. 3.**
3. **¿Es estable a lo largo de los años?** Los horarios de mercado, la microestructura y la composición de participantes cambian. Un patrón intradía estimado sobre 10 años puede ser un promedio de patrones distintos.
4. **¿Sobrevive a la corrección por multiplicidad?** Con 24 horas, 5 días y varios instrumentos, el número de "efectos calendario" testeables es grande. Los Ejercicios 2.7–2.9 de Tsay son un buen molde metodológico precisamente porque incluyen HAC y diagnóstico de residuos, pero no incluyen corrección por multiplicidad. **[B]**
5. **¿La magnitud supera los costos?** Un efecto de calendario estadísticamente significativo de magnitud sub-tick no es explotable.
6. **¿Conviene modelarlo o condicionar sobre él?** Alternativa a modelar la estacionalidad: usar la etiqueta temporal como feature y dejar que el modelo decida. Son enfoques distintos y ninguno está decidido aquí.

**Sobre la mención de futuros de energía [A]:** Tsay indica que las series medioambientales tienen estacionalidad fuerte y que eso importa para futuros de energía. Es una observación específica y creíble sobre **fundamentales** (demanda de gas natural, electricidad). **No implica que los retornos de esos futuros sean estacionales** — el precio del futuro descuenta la estacionalidad conocida. Confundir estacionalidad del subyacente físico con estacionalidad de los retornos del futuro sería un error de razonamiento. **[B]**

### [B] Implicancias potenciales para ML

- La transformación logarítmica para estabilizar variabilidad [A] es un caso particular de un principio general: **transformar para homogeneizar la varianza antes de modelar**. Conecta con la discusión de normalización por volatilidad del Capítulo 1. Sigue siendo hipótesis.
- Las variables de calendario son features **causalmente disponibles y de coste cero** — se conocen con antelación infinita. Eso las hace atractivas y a la vez peligrosas: son el tipo de feature sobre el que es más fácil hacer data snooping.
- La advertencia de Tsay sobre Ljung–Box en series estacionales [A, §2.2] —"esta regla general [$m\approx\ln T$] necesita modificación... las autocorrelaciones con lags múltiplos de la estacionalidad son más importantes"— es directamente aplicable al diagnóstico de residuos en datos intradía: hay que mirar los lags que corresponden a un día completo.

---

## 2.9 Sección 2.9 — Regression Models with Time Series Errors

### [A] Qué afirma Tsay

**Planteo y advertencia inicial.** "En muchas aplicaciones, la relación entre dos series temporales es de mayor interés. Un ejemplo obvio es el **modelo de mercado** en finanzas, que relaciona el exceso de retorno de una acción individual con el de un índice de mercado. La **estructura temporal de tasas de interés** es otro ejemplo."

Modelo (Ec. 2.43): $y_t=\alpha+\beta x_t+e_t$. "El método de mínimos cuadrados se usa a menudo para estimar el modelo. **Si $\{e_t\}$ es una serie de white noise, entonces el método LS produce estimaciones consistentes.** En la práctica, sin embargo, es común ver que el término de error $e_t$ está serialmente correlacionado. En este caso tenemos un modelo de regresión con errores de series temporales, **y las estimaciones LS de $\alpha$ y $\beta$ pueden no ser consistentes**."

**La advertencia, textual:** **"Un modelo de regresión con errores de series temporales es ampliamente aplicable en economía y finanzas, pero es uno de los modelos econométricos más comúnmente mal utilizados porque la dependencia serial en $e_t$ es a menudo pasada por alto. Vale la pena estudiar el modelo cuidadosamente."**

**El ejemplo completo.** Dos tasas del Tesoro semanales (1-año $r_{1t}$ y 3-años $r_{3t}$), 2,467 observaciones, 5-ene-1962 a 10-abr-2009.

**Paso 1 — regresión en niveles** (Ec. 2.44):
$$r_{3t}=0.832+0.930r_{1t}+e_t,\qquad \hat\sigma_e=0.523,\qquad \mathbf{R^2=96.5\%}$$
con errores estándar 0.024 y 0.004. **"Sin embargo, el modelo es seriamente inadecuado"**: la ACF muestral de los residuos es altamente significativa y **decae lentamente, mostrando el patrón de una serie no estacionaria con raíz unitaria**. "El comportamiento de los residuos sugiere que existen diferencias marcadas entre las dos tasas de interés. Usando la terminología econométrica moderna, si uno asume que ambas series son no estacionarias con raíz unitaria, entonces el comportamiento de los residuos indica que las dos tasas **no están cointegradas**; ver el Capítulo 8... los datos no respaldan la hipótesis de que exista un equilibrio de largo plazo entre las dos tasas." Menciona que el fenómeno de "curva de rendimientos invertida" ocurrió durante el período muestral.

**Paso 2 — regresión en diferencias** (Ec. 2.45): con $c_{1t}=\Delta r_{1t}$, $c_{3t}=\Delta r_{3t}$:
$$c_{3t}=0.792c_{1t}+e_t,\qquad \hat\sigma_e=0.0690,\qquad R^2=82.5\%$$
error estándar 0.0073. Los residuos "muestran aún algunas correlaciones seriales significativas, pero **las magnitudes de las correlaciones son mucho menores**".

**Paso 3 — regresión con errores MA(1)** (Ec. 2.47):
$$c_{3t}=0.794c_{1t}+e_t,\qquad e_t=a_t+0.1823a_{t-1},\qquad \hat\sigma_a=0.0678,\qquad R^2=83.1\%$$
"El modelo ya no tiene una ACF residual de lag-1 significativa, aunque permanecen algunas correlaciones seriales residuales menores en los lags 4, 6 y 7."

**Las tres observaciones de Tsay al comparar los modelos, textuales:**
1. **"El alto $R^2$ de 96.5% y el coeficiente 0.930 del modelo (2.44) son ENGAÑOSOS porque los residuos del modelo muestran correlaciones seriales fuertes."**
2. "Para las series de cambios, el $R^2$ y el coeficiente de los modelos (2.45) y (2.47) son cercanos... añadir el modelo MA(1) sólo proporciona una mejora marginal."
3. **"El análisis demuestra que es importante verificar la dependencia serial de los residuos en el análisis de regresión lineal."**

**Procedimiento general que propone [A]:**
1. Ajustar el modelo de regresión lineal y **verificar las correlaciones seriales de los residuos**.
2. Si la serie residual es no estacionaria con raíz unitaria, **tomar la primera diferencia de las variables dependiente y explicativas**, y volver al paso 1. Si la serie residual parece estacionaria, identificar un modelo ARMA para los residuos y modificar el modelo de regresión en consecuencia.
3. Realizar una **estimación conjunta por máxima verosimilitud** y verificar el modelo ajustado.

**Sobre qué test usar:** "Para verificar las correlaciones seriales de los residuos, **recomendamos que se usen los estadísticos de Ljung–Box en lugar del estadístico de Durbin–Watson (DW), porque este último sólo considera la correlación serial de lag-1.** Hay casos en los que la dependencia serial en los residuos aparece en lags de orden superior. Esto es particularmente así cuando la serie involucrada exhibe algún comportamiento estacional." Con $\text{DW}\approx2(1-\hat\rho_1)$.

### [B] Por qué los coeficientes pueden parecer significativos con inferencia mal especificada

El ejemplo de Tsay es la ilustración perfecta, pero conviene desarmar el mecanismo [B]:

**El problema no es (necesariamente) el estimador puntual, sino su error estándar.** La fórmula estándar $\text{Var}(\hat\beta)=\sigma_e^2(\sum x_tx_t')^{-1}$ [A, §2.10] se deriva bajo el supuesto de que los $e_t$ son no correlacionados y homocedásticos. Si los residuos están positivamente autocorrelacionados, **cada observación aporta menos información nueva de la que la fórmula supone**: el tamaño de muestra efectivo es menor que $T$. La fórmula sigue dividiendo por $T$, subestima la varianza, y el t-ratio resultante es demasiado grande.

**El caso de la regresión espuria [B]:** cuando ambas series tienen raíz unitaria y no están cointegradas, el problema es más severo que la mera subestimación del error estándar: el estimador mismo no converge a un valor poblacional significativo. El $R^2$ de 96.5% no mide una relación económica sino la circunstancia de que dos series con tendencia estocástica se mueven juntas por construcción durante períodos largos. **Tsay diagnostica el síntoma correctamente** —residuos con ACF que decae lentamente, patrón de raíz unitaria— y da el remedio correcto —diferenciar— pero **no usa el término "regresión espuria"** ni desarrolla la teoría. Su marco es la cointegración, que remite al Cap. 8.

**El diagnóstico clave que Tsay entrega [A]:** *la forma de la ACF de los residuos dice qué está mal*. ACF que decae lentamente y no muere → probable raíz unitaria en los residuos → diferenciar. ACF pequeña que se corta pronto → modelar los residuos con ARMA. Es un árbol de decisión concreto y transferible.

### [B] Relevancia para hipótesis multivariadas entre futuros

La pregunta $r^{ES}_{t-k}\rightarrow r^{NQ}_t$ y sus análogas caen exactamente en este marco. **No se determina aquí si tal relación existe.** Lo que el capítulo establece es qué hay que hacer para que la pregunta esté bien planteada:

1. **Trabajar sobre retornos, no sobre precios.** El ejemplo de Tsay muestra que la regresión en niveles entre dos series con raíz unitaria produce $R^2$ = 96.5% y es "seriamente inadecuada". Una regresión entre precios de ES y NQ produciría lo mismo, con la misma vacuidad. Esto es la aplicación directa de [A].
2. **Verificar la ACF de los residuos, siempre.** Es el paso 1 del procedimiento de Tsay [A] y el que "más comúnmente se pasa por alto".
3. **Distinguir contemporáneo de rezagado.** El ejemplo de Tsay es **contemporáneo** ($c_{3t}$ sobre $c_{1t}$, mismo $t$): mide co-movimiento, no predicción. Una relación contemporánea, por fuerte que sea, **no es utilizable para predecir** porque ambas variables se observan simultáneamente. La hipótesis lead-lag requiere $k\ge1$ y es un objeto distinto. **[B]**
4. **Corregir la inferencia.** Ver §2.10: en el propio ejemplo, el t-ratio cae de 107.91 a 39.92.
5. **La estructura correcta puede ser multivariada, no de regresión.** Tsay lo admite explícitamente [A]: "estrictamente hablando, deberíamos modelar las dos series de interés conjuntamente usando análisis multivariado de series temporales en el Capítulo 8. Sin embargo, por simplicidad, nos enfocamos aquí en un análisis tipo regresión e **ignoramos el problema de la simultaneidad**." Registrar esa limitación: la regresión impone una dirección causal que los datos no establecen.

**El Ejercicio 2.14 es directamente relevante [A]:** pide construir una regresión con errores de series temporales entre los log-precios del **futuro del S&P 500 y su spot**, con datos de 1 minuto intradía de mayo de 1993, usando $y_t=f_t-f_{t-1}$ y $x_t=s_t-s_{t-1}$. Es el molde metodológico exacto para una pregunta de relación entre dos instrumentos relacionados a alta frecuencia: **diferenciar primero, regresar después, modelar los residuos**. Tsay señala que "varios autores usaron los datos para estudiar arbitraje de futuros de índices".

### [B] Implicancias potenciales para ML

- El procedimiento de tres pasos [A] es transferible como **protocolo de diagnóstico** para cualquier modelo predictivo con features exógenas, incluidos modelos de ML: ajustar, examinar residuos, corregir especificación o inferencia, reestimar.
- La preferencia por Ljung–Box sobre Durbin–Watson [A] importa en la práctica: DW sólo ve el lag 1. En datos con estructura intradía o semanal, la dependencia relevante puede estar en lags mayores.
- **La distinción entre "el modelo predice bien" y "la inferencia sobre el modelo es válida" [B]** es la lección de fondo. Un modelo de ML puede tener buen desempeño out-of-sample y a la vez toda la inferencia sobre importancias de features, significancia de mejoras y comparación entre modelos puede estar mal calibrada por dependencia residual.

### Preguntas abiertas

- ¿Están cointegrados dos futuros relacionados? ¿Qué implicaría? → **Cap. 8.**
- ¿Cómo tratar la simultaneidad, que Tsay explícitamente ignora aquí? → **Cap. 8 (VAR).**
- ¿Existe lead-lag genuino entre futuros y, de existir, sobrevive a los costos? → **Cap. 8 + evidencia empírica; PREGUNTA ABIERTA.**

---

## 2.10 Sección 2.10 — Consistent Covariance Matrix Estimation

### [A] Qué afirma Tsay

**Planteo, con una condición importante.** "Puede haber situaciones en las que el término de error $e_t$ tenga correlaciones seriales y/o heterocedasticidad condicional, pero **el objetivo principal del análisis sea hacer inferencia sobre los coeficientes de regresión $\alpha$ y $\beta$**. **En situaciones bajo las cuales las estimaciones OLS de los coeficientes siguen siendo consistentes**, hay métodos disponibles para proporcionar una estimación consistente de la matriz de covarianza de las estimaciones de los coeficientes."

Nótese la condición: HAC/HC corrigen **la covarianza**, presuponiendo que el estimador puntual sigue siendo consistente.

**Notación** (Ec. 2.48): $y_t=\boldsymbol x_t'\boldsymbol\beta+e_t$, con estimador LS y covarianza convencional
$$\hat{\boldsymbol\beta}=\left(\sum \boldsymbol x_t\boldsymbol x_t'\right)^{-1}\sum \boldsymbol x_ty_t,\qquad \text{Cov}(\hat{\boldsymbol\beta})=\sigma_e^2\left(\sum \boldsymbol x_t\boldsymbol x_t'\right)^{-1}$$

**El problema, textual:** **"En presencia de correlaciones seriales o heterocedasticidad condicional, el estimador de la matriz de covarianza anterior es INCONSISTENTE, resultando a menudo en la INFLACIÓN de los t-ratios de $\hat\beta$."**

**Estimador HC** (White, 1980; Eicker, 1967), Ec. (2.49):
$$\text{Cov}(\hat{\boldsymbol\beta})_{HC}=\left(\sum \boldsymbol x_t\boldsymbol x_t'\right)^{-1}\left(\sum \hat e_t^2\boldsymbol x_t\boldsymbol x_t'\right)\left(\sum \boldsymbol x_t\boldsymbol x_t'\right)^{-1}$$

**Estimador HAC** (Newey y West, 1987), Ec. (2.50): misma estructura de sándwich con
$$\hat C_{HAC}=\sum_{t=1}^{T}\hat e_t^2\boldsymbol x_t\boldsymbol x_t'+\sum_{j=1}^{\ell}w_j\sum_{t=j+1}^{T}\left(\boldsymbol x_t\hat e_t\hat e_{t-j}\boldsymbol x_{t-j}'+\boldsymbol x_{t-j}\hat e_{t-j}\hat e_t\boldsymbol x_t'\right)$$
donde $\ell$ es un **parámetro de truncamiento** y $w_j$ una función de pesos, como la **de Bartlett**: $w_j=1-\dfrac{j}{\ell+1}$. "Pueden usarse otras funciones de peso."

**Selección del bandwidth [A]:** **"Newey y West (1987) sugieren elegir $\ell$ como la parte entera de $4(T/100)^{2/9}$."** "Este estimador esencialmente usa un método no paramétrico para estimar la matriz de covarianza de $\{\sum\hat e_t\boldsymbol x_t\}$."

**La ilustración numérica, textual:** **"Para ilustración, empleamos la serie de tasas de interés en primeras diferencias de la Ec. (2.45). El t-ratio del coeficiente de $c_{1t}$ es 107.91 si tanto la correlación serial como la heterocedasticidad en los residuos se ignoran, se convierte en 48.44 cuando se usa el estimador HC, y se reduce a 39.92 cuando se usa el estimador HAC."**

En la misma salida [A]: los residuos de esa regresión tienen **Ljung–Box = 230.05 (p = 0.0000)** y **Jarque–Bera = 1644.61 (p = 0.0000)** — es decir, ambos supuestos violados simultáneamente.

Alternativa mostrada [A]: incluir valores rezagados $c_{1,t-1}$ y $c_{3,t-1}$ como regresores "para hacerse cargo de las correlaciones seriales en los residuos". Tras hacerlo, DW pasa de 1.6456 a 1.9865, aunque Ljung–Box sigue siendo 131.60 con p = 0.0000. "Uno puede también aplicar el estimador HC o HAC al modelo ajustado para refinar los t-ratios."

**Fórmula auxiliar [A]:** cuando $k>1$, la varianza HC de $\hat\beta_j$ puede obtenerse mediante una regresión auxiliar de $x_{jt}$ sobre el resto de los regresores, con residuo $\hat v_t$:
$$\text{Var}(\hat\beta_j)_{HC}=\frac{\sum\hat e_t^2\hat v_t^2}{\left(\sum\hat v_t^2\right)^2}$$

### Significado estadístico: por qué el t-stat ingenuo exagera

**El mecanismo, elaborado [B]** (Tsay enuncia el hecho pero no lo desarrolla):

La fórmula convencional supone que $\text{Var}(\sum \boldsymbol x_te_t)=\sigma_e^2\sum \boldsymbol x_t\boldsymbol x_t'$, lo que requiere que los términos $\boldsymbol x_te_t$ sean **no correlacionados entre sí y de varianza constante**. Si están positivamente autocorrelacionados, la varianza de la suma incluye términos cruzados positivos que la fórmula omite:
$$\text{Var}\left(\sum_t \boldsymbol x_te_t\right)=\sum_t\text{Var}(\boldsymbol x_te_t)+\underbrace{2\sum_{j\ge1}\sum_t\text{Cov}(\boldsymbol x_te_t,\boldsymbol x_{t-j}e_{t-j})}_{\text{ignorado por la fórmula convencional}}$$
El estimador HAC estima precisamente esos términos cruzados, hasta el lag $\ell$, con pesos decrecientes. **La interpretación intuitiva es que $T$ observaciones dependientes contienen menos información que $T$ observaciones independientes**: el tamaño de muestra efectivo es menor, y todo lo que dependa de $\sqrt T$ está inflado en consecuencia.

**La magnitud del efecto no es marginal [A]:** un factor de 2.7 en el t-ratio del ejemplo. Traducido a p-valores, un t de 3.0 corregido a 1.1 pasa de "altamente significativo" a "no significativo".

### [B] Dónde aparece este problema en un proyecto de ML sobre futuros

Tsay lo plantea sólo para regresión lineal. La extensión a un pipeline de ML es enteramente [B], y el problema aparece en al menos cinco lugares:

**1. Análisis univariado de features.** Calcular la correlación entre un feature y el retorno futuro, y evaluar su significancia. Si tanto el feature como el target son series con dependencia temporal —y lo son—, la significancia ingenua está inflada. Es el escenario más común y más subestimado.

**2. Regresiones predictivas.** Exactamente el caso de Tsay. Si además el target es un retorno acumulado con ventanas solapadas, la autocorrelación residual es enorme por construcción y la inflación del t-ratio, severa.

**3. Evaluación de una estrategia.** El t-stat de un Sharpe ratio se calcula habitualmente asumiendo retornos de estrategia iid. Si los retornos de la estrategia están autocorrelacionados —por persistencia de posiciones, por exposición a volatilidad, o por overlapping— el error estándar es incorrecto en la misma dirección: demasiado pequeño. **La corrección conceptual es la misma que HAC.**

**4. Comparación entre modelos.** Testear si el modelo A supera al modelo B usando la serie de diferencias de error. Esa serie hereda la dependencia temporal de los datos.

**5. Análisis de los retornos de una señal.** Agrupar retornos por decil de la señal y testear diferencias entre grupos. Las observaciones dentro de un decil no son independientes: tienden a agruparse temporalmente, porque la señal misma es persistente.

**Advertencia importante [B]:** HAC corrige errores estándar bajo el supuesto de que el estimador puntual es consistente [A lo condiciona explícitamente]. **No corrige sesgo, no corrige mala especificación, y no corrige look-ahead bias.** No es un sustituto de un esquema de validación temporal correcto. Ver §8, punto 10.

**Sobre el bandwidth [B]:** la regla $\ell=\lfloor4(T/100)^{2/9}\rfloor$ [A] crece muy lentamente con $T$ — para $T=10^5$ da $\ell\approx13$. Con targets solapados de horizonte $h$, el truncamiento debería al menos cubrir $h$. Esto es una consideración práctica, no una recomendación cerrada: **PREGUNTA ABIERTA** sobre cuál es el bandwidth apropiado en cada aplicación del proyecto.

### [B] Implicancias potenciales para ML

- **No adoptar todavía una metodología de inferencia definitiva.** Lo que sí queda establecido es que **algún tipo de corrección es necesaria**, y que la magnitud del error sin ella puede ser un factor de 2–3 en los t-ratios.
- Alternativas conceptuales al HAC que el capítulo no menciona pero que abordan el mismo problema **[B]**: bootstrap por bloques, inferencia basada en submuestras, o simplemente reducir la frecuencia de evaluación a observaciones aproximadamente independientes. Ninguna está decidida.
- La observación de que añadir regresores rezagados reduce la autocorrelación residual [A] sugiere una alternativa estructural: **en lugar de corregir la inferencia, especificar mejor el modelo**. Tsay muestra ambos caminos y no declara uno superior.

### Preguntas abiertas

- ¿Cuál es el bandwidth apropiado para las aplicaciones concretas del proyecto? → **empírica.**
- ¿Cómo se corrige la inferencia cuando el estimador puntual **no** es consistente? → fuera de este capítulo.
- ¿Cómo interactúa la heterocedasticidad condicional con todo esto? → **Cap. 3.**

---

## 2.11 Sección 2.11 — Long-Memory Models

### [A] Qué afirma Tsay

**Taxonomía del decaimiento de la ACF.** Tsay ordena tres regímenes:
- **Serie estacionaria (memoria corta):** la ACF decae **exponencialmente** a cero.
- **Serie no estacionaria con raíz unitaria:** "puede demostrarse que la ACF muestral **converge a 1 para todos los lags fijos** a medida que el tamaño de muestra crece" (Chan y Wei, 1988; Tiao y Tsay, 1983).
- **Memoria larga:** "Existen algunas series temporales cuya ACF decae lentamente a cero **a tasa polinomial** a medida que el lag aumenta. Estos procesos se conocen como series temporales de memoria larga."

**Proceso fraccionalmente diferenciado** (Ec. 2.52): $(1-B)^dx_t=a_t$, con $-0.5<d<0.5$ y $\{a_t\}$ white noise. Propiedades (Hosking, 1981):

1. Si $d<0.5$, $x_t$ es **débilmente estacionario** y tiene representación MA infinita con $\psi_k=\dfrac{d(1+d)\cdots(k-1+d)}{k!}$.
2. Si $d>-0.5$, $x_t$ es **invertible** y tiene representación AR infinita con $\pi_k=\dfrac{-d(1-d)\cdots(k-1-d)}{k!}$.
3. Para $-0.5<d<0.5$, la ACF es $\rho_k=\dfrac{d(1+d)\cdots(k-1+d)}{(1-d)(2-d)\cdots(k-d)}$, con $\rho_1=\dfrac{d}{1-d}$ y, asintóticamente,
$$\boxed{\rho_k\approx\frac{(-d)!}{(d-1)!}k^{2d-1}\quad\text{cuando }k\to\infty}$$
4. La PACF es $\phi_{k,k}=\dfrac{d}{k-d}$.
5. La densidad espectral satisface $f(\omega)\sim\omega^{-2d}$ cuando $\omega\to0$ (Ec. 2.53).

**La propiedad definitoria, textual:** "De particular interés aquí es el comportamiento de la ACF de $x_t$ cuando $d<0.5$. La propiedad dice que $\rho_k\sim k^{2d-1}$, que decae a **tasa polinomial en lugar de exponencial**. Por esta razón, tal proceso $x_t$ se llama serie temporal de memoria larga." Y sobre el espectro: "diverge a infinito cuando $\omega\to0$. Sin embargo, la densidad espectral de un proceso ARMA estacionario **está acotada para todo $\omega$**."

**Criterio práctico de detección [A]:** **"En la práctica, si la ACF muestral de una serie temporal NO ES GRANDE EN MAGNITUD, PERO DECAE LENTAMENTE, entonces la serie puede tener memoria larga."**

**LA EVIDENCIA EMPÍRICA CENTRAL [A].** Figura 2.22: ACF muestrales de **la serie de valores absolutos** de los retornos simples diarios de los índices CRSP value-weighted y equal-weighted, 2-ene-1970 a 31-dic-2008. Textual:

> **"Las ACF son relativamente pequeñas en magnitud pero decaen muy lentamente; parecen ser significativas al nivel del 5% incluso después de 300 lags."**

Referencia: Ding, Granger y Engle (1993).

**ARFIMA.** Si la serie fraccionalmente diferenciada $(1-B)^dx_t$ sigue un ARMA($p,q$), entonces $x_t$ es un proceso **ARFIMA($p,d,q$)**, "un modelo ARIMA generalizado que permite $d$ no entero". Estimación de $d$: "máxima verosimilitud o un método de regresión con periodograma logarítmico en las frecuencias bajas".

**Cierre:** "los modelos de memoria larga han atraído cierta atención en la literatura financiera, en parte debido al trabajo sobre movimiento browniano fraccional en los modelos de tiempo continuo".

**Ejercicio 2.5 [A]:** pide computar los primeros 100 lags de la ACF de **la serie de valores absolutos** de los retornos simples diarios de IBM (1970–2008) y responder si hay evidencia de dependencia de largo alcance.

### [B] La observación central: por qué $\text{Corr}(r_t,r_{t-k})\approx0$ y $\text{Corr}(|r_t|,|r_{t-k}|)>0$ simultáneamente

Éste es, para el proyecto, el hallazgo empírico más importante del capítulo. Los dos hechos son [A]:

- ACF de $r_t$ para índices mensuales: dentro de las bandas, Ljung–Box no significativo para IBM [A, §2.2].
- ACF de $|r_t|$ para índices diarios: significativa **hasta más allá de 300 lags** [A, §2.11].

**Por qué no hay contradicción [B].** La ACF de $r_t$ mide dependencia lineal en la variable **con signo**. La ACF de $|r_t|$ mide dependencia lineal en la **magnitud**. Un proceso de la forma
$$r_t=\sigma_t\varepsilon_t,\qquad \varepsilon_t \text{ iid, media 0},\qquad \sigma_t \text{ función del pasado}$$
tiene, si $\varepsilon_t$ es independiente de $\sigma_t$:
- $\text{Corr}(r_t,r_{t-k})=0$, porque el signo de $\varepsilon_t$ es impredecible y $E(r_t\mid\mathcal F_{t-1})=0$;
- $\text{Corr}(|r_t|,|r_{t-k}|)>0$, porque $|r_t|=\sigma_t|\varepsilon_t|$ hereda toda la persistencia de $\sigma_t$.

**Es exactamente la situación descrita en el Capítulo 1** como volatility clustering, ahora con una caracterización cuantitativa: la persistencia no sólo existe, **decae a tasa polinomial**, es decir, mucho más lentamente que la de cualquier ARMA estacionario.

**Esto cierra el argumento sobre "ACF ≈ 0 ⇒ impredecible" [B]:** una serie puede tener ACF nula en todos los lags y contener una cantidad enorme de estructura predecible. La estructura simplemente no está en la media. Y como se estableció en §2.3, esta serie **no es white noise en el sentido de Tsay (iid)**, aunque todas sus ACF sean cero.

**Cuantificación de la asimetría [B]:** la comparación entre "ACF de retornos dentro de las bandas" y "ACF de |retornos| significativa después de 300 lags" sugiere una diferencia de órdenes de magnitud en la cantidad de estructura detectable entre dirección y magnitud. **Pero cuidado: "más estructura detectable" no es lo mismo que "más rentabilidad".** La volatilidad predecible no dice de qué lado ponerse. Es información sobre riesgo, no sobre dirección. Convertirla en rentabilidad requiere un mecanismo adicional (dimensionamiento, opciones, o una relación entre volatilidad y retorno esperado) que este capítulo no provee. **PREGUNTA ABIERTA.**

### [B] Fractional differencing: qué problema resuelve y cuándo podría ser relevante

**No se propone usarlo.** Lo que sigue es la caracterización del problema que aborda.

**El problema [B].** Existe un dilema entre dos objetivos:
- **Estacionariedad**, que la diferenciación entera ($d=1$) garantiza pero al precio de destruir toda la información de nivel: $r_t=p_t-p_{t-1}$ no contiene ninguna memoria de dónde estaba el precio.
- **Memoria**, que la serie sin diferenciar ($d=0$) conserva íntegramente pero es no estacionaria.

La diferenciación fraccional [A] permite $0<d<1$: un continuo entre ambos extremos. Con $d<0.5$ la serie resultante es **estacionaria** [A, propiedad 1] y con $d>0$ conserva una ACF que decae polinomialmente, es decir, **memoria larga** [A, propiedad 3].

**Qué preserva y qué elimina [B], derivado de las propiedades [A]:**
- *Elimina:* la no estacionariedad, si $d$ es suficientemente grande.
- *Preserva:* dependencia de largo alcance, que la diferenciación entera destruye.
- *Coste:* la serie transformada no tiene interpretación económica directa; $d$ es un parámetro más a estimar, con su propio error de estimación; y la transformación requiere una ventana histórica larga (los pesos $\pi_k$ decaen lentamente por construcción).

**Cuándo podría ser relevante en este proyecto [B]:**
- Como transformación de features de **nivel** que se quiere conservar parcialmente sin arrastrar no estacionariedad.
- Para modelar directamente series de **volatilidad**, donde la memoria larga es el fenómeno documentado [A].
- **No** para el retorno mismo, que ya es $d=1$ sobre el log-precio y para el cual la evidencia de Tsay no sugiere memoria larga.

**Advertencia crítica [B]:** la transformación fraccional usa toda la historia previa con pesos decrecientes. En un pipeline con validación temporal, debe calcularse **causalmente** (sólo con datos hasta $t$) y con la ventana truncada de forma consistente entre train y test. Es un canal de leak si se implementa descuidadamente. **Nada de esto está en Tsay.**

**Queda como hipótesis metodológica abierta**, no como decisión.

### [B] Implicancias potenciales para ML

- **Diagnóstico prioritario, no requisito:** examinar la ACF de $|r_t|$ y de $r_t^2$ junto a la de $r_t$ es barato y, según la evidencia de Tsay sobre índices accionarios diarios, es donde aparece la estructura. Se recomienda como diagnóstico de alta prioridad; que sea informativo en futuros y a otras frecuencias es hipótesis (H2.7).
- **Memoria larga en el target vs en el feature:** si un feature tiene memoria larga, las observaciones consecutivas son altamente redundantes y el tamaño de muestra efectivo es mucho menor que $T$. Esto agrava todos los problemas de inferencia de §2.10.
- **La memoria larga en volatilidad tiene consecuencias sobre la validación [B]:** con dependencia que persiste cientos de lags, un embargo corto entre train y test es insuficiente para garantizar independencia. La longitud del embargo debería relacionarse con la persistencia observada, no elegirse arbitrariamente. **PREGUNTA ABIERTA.**
- **"Long memory ⇒ señal direccional" es un no sequitur** — ver §8.

### Preguntas abiertas

- ¿La memoria larga en $|r_t|$ se modela mejor con ARFIMA, con GARCH de alta persistencia, o con componentes múltiples? → **Cap. 3.**
- ¿Es la memoria larga aparente un artefacto de cambios de régimen en la volatilidad? → **Caps. 3, 11–12.**
- ¿Cómo se traduce persistencia de volatilidad en decisiones de trading? → **fuera de Tsay; PREGUNTA ABIERTA del proyecto.**

---

# 3. Conceptos matemáticos esenciales

Sólo lo indispensable.

**Estacionariedad débil.** $E(r_t)=\mu$ constante; $\text{Cov}(r_t,r_{t-\ell})=\gamma_\ell$ depende sólo de $\ell$; primeros dos momentos finitos. $\gamma_0=\text{Var}(r_t)$, $\gamma_{-\ell}=\gamma_\ell$.

**ACF y su estimación.**
$$\rho_\ell=\frac{\gamma_\ell}{\gamma_0},\qquad \hat\rho_\ell=\frac{\sum_{t=\ell+1}^{T}(r_t-\bar r)(r_{t-\ell}-\bar r)}{\sum_{t=1}^{T}(r_t-\bar r)^2}$$
Bajo **iid** con $E(r_t^2)<\infty$: $\hat\rho_\ell\overset{a}{\sim}N(0,1/T)$. Bartlett (para $\ell>q$): varianza $\left(1+2\sum_{i=1}^{q}\rho_i^2\right)/T$.

**Ljung–Box** (Ec. 2.3):
$$Q(m)=T(T+2)\sum_{\ell=1}^{m}\frac{\hat\rho_\ell^2}{T-\ell}\ \overset{a}{\sim}\ \chi^2_m\quad\text{bajo }H_0\text{ e iid}$$
Sobre residuos de un AR($p$): $\chi^2_{m-g}$, con $g$ = n° de coeficientes AR. Regla sugerida: $m\approx\ln(T)$.

**Serie lineal** (Ec. 2.4): $r_t=\mu+\sum_{i=0}^{\infty}\psi_ia_{t-i}$, $\psi_0=1$, $\{a_t\}$ white noise (iid). $\text{Var}(r_t)=\sigma_a^2\sum\psi_i^2$.

**AR(1) y AR(p).**
$$r_t=\phi_0+\phi_1r_{t-1}+a_t,\qquad E(r_t\mid r_{t-1})=\phi_0+\phi_1r_{t-1},\qquad \text{Var}(r_t\mid r_{t-1})=\sigma_a^2$$
Estacionariedad AR(1): $|\phi_1|<1$ (necesaria y suficiente). $\mu=\phi_0/(1-\phi_1)$, $\text{Var}=\sigma_a^2/(1-\phi_1^2)$, $\rho_\ell=\phi_1^\ell$.
AR($p$): estacionariedad ⟺ todas las raíces características menores que 1 en módulo, donde las raíces son los inversos de las soluciones de $1-\phi_1x-\dots-\phi_px^p=0$.

**MA(q).** $r_t=c_0+a_t-\theta_1a_{t-1}-\dots-\theta_qa_{t-q}$. Siempre estacionario. $E(r_t)=c_0$, $\text{Var}(r_t)=(1+\sum\theta_i^2)\sigma_a^2$, **$\rho_\ell=0$ para $\ell>q$**. Invertibilidad MA(1): $|\theta_1|<1$.

**Identificación.** ACF se corta en $q$ ⇒ MA($q$). PACF se corta en $p$ ⇒ AR($p$). Ninguna se corta ⇒ ARMA (usar EACF).

**Criterios de información.**
$$\text{AIC}(\ell)=\ln(\tilde\sigma_\ell^2)+\frac{2\ell}{T},\qquad \text{BIC}(\ell)=\ln(\tilde\sigma_\ell^2)+\frac{\ell\ln T}{T}$$

**Pronóstico.** $\hat r_h(\ell)=E(r_{h+\ell}\mid F_h)$ bajo pérdida cuadrática. Error 1 paso $=a_{h+1}$. Varianza del error (Ec. 2.34):
$$\text{Var}[e_h(\ell)]=(1+\psi_1^2+\dots+\psi_{\ell-1}^2)\sigma_a^2\quad\text{no decreciente en }\ell$$
Reversión a la media: $\hat r_h(\ell)\to\mu$; vida media AR(1): $\ell=\ln(0.5)/\ln|\phi_1|$.

**Bondad de ajuste.** $R^2=1-\text{RSS}/\text{TSS}$; **válido sólo para series estacionarias** — con raíz unitaria, $R^2\to1$ independientemente del modelo verdadero.

**Random walk (con y sin drift).**
$$p_t=p_{t-1}+a_t\ \Rightarrow\ \hat p_h(\ell)=p_h\ \forall\ell,\quad \text{Var}[e_h(\ell)]=\ell\sigma_a^2\to\infty,\quad \psi_i=1\ \forall i$$
$$p_t=\mu+p_{t-1}+a_t\ \Rightarrow\ p_t=t\mu+p_0+\sum_{i=1}^{t}a_i$$
**Trend-stationary:** $p_t=\beta_0+\beta_1t+r_t$ con $\text{Var}(p_t)=\text{Var}(r_t)$ finita e invariante.

**ADF** (Ec. 2.40):
$$x_t=c_t+\beta x_{t-1}+\sum_{i=1}^{p-1}\phi_i\Delta x_{t-i}+e_t,\qquad H_0:\beta=1\ (\beta_c=0)\ \text{vs}\ H_a:\beta<1$$
$c_t\in\{0,\ \omega_0,\ \omega_0+\omega_1t\}$. Distribución **no estándar**; valores críticos por simulación.

**Diferenciación estacional.** $\Delta_sy_t=y_t-y_{t-s}=(1-B^s)y_t$. Airline model: $(1-B^s)(1-B)x_t=(1-\theta B)(1-\Theta B^s)a_t$.

**HAC / Newey–West** (Ec. 2.50): sándwich con $w_j=1-\dfrac{j}{\ell+1}$ y $\ell=\left\lfloor4(T/100)^{2/9}\right\rfloor$.

**Memoria larga** (Ec. 2.52): $(1-B)^dx_t=a_t$, $-0.5<d<0.5$;
$$\rho_k\approx C\,k^{2d-1}\ \ (\text{polinomial}),\qquad \phi_{k,k}=\frac{d}{k-d},\qquad f(\omega)\sim\omega^{-2d}$$

---

# 4. Mapa conceptual del capítulo

## Cadena 1 — El ciclo de modelado lineal

$$\textbf{Stationarity}\ \rightarrow\ \textbf{ACF/PACF}\ \rightarrow\ \textbf{AR/MA/ARMA}\ \rightarrow\ \textbf{Residuals}\ \rightarrow\ \textbf{White Noise}$$

| Eslabón | Qué problema resuelve | Cómo se conecta con el siguiente |
|---|---|---|
| **Stationarity** | Hace posible estimar cantidades poblacionales desde una sola realización | Sin ella, $\rho_\ell$ **ni siquiera está definida** |
| **ACF/PACF** | Detectan y caracterizan la dependencia lineal; ACF identifica orden MA, PACF orden AR | El patrón observado sugiere la clase de modelo |
| **AR/MA/ARMA** | Representan esa dependencia con pocos parámetros | Producen un residuo $\hat a_t$ |
| **Residuals** | Aíslan lo que el modelo no explicó | Se les aplica ACF y Ljung–Box |
| **White noise** | Criterio de adecuación: si $\hat a_t$ es white noise, no queda estructura **lineal** | Si falla, se vuelve al paso 3 |

**El ciclo es iterativo [A]:** Tsay lo aplica explícitamente al AR(3) del índice VW —detecta un coeficiente no significativo, refina, revuelve a testear. **Y es circular en un sentido peligroso [B]:** el mismo dataset se usa para identificar, estimar y verificar. Cada iteración consume grados de libertad que ningún test contabiliza.

## Cadena 2 — El camino de la no estacionariedad

$$\textbf{Unit Root}\ \rightarrow\ \textbf{Differencing}\ \rightarrow\ \textbf{Stationarity}$$

| Eslabón | Qué problema resuelve | Condiciones y advertencias |
|---|---|---|
| **Unit root** | Diagnostica que no hay nivel fijo; los shocks son permanentes; ACF muestral → 1 | ADF: no rechazar ≠ demostrar. Distribución no estándar |
| **Differencing** | Convierte ARIMA($p,1,q$) en ARMA($p,q$) | Sobrediferenciar una serie trend-stationary es un error. Ver alternativa: remover tendencia |
| **Stationarity** | Restaura la validez de todo el aparato de la Cadena 1 | Es estacionariedad **débil**; no garantiza estabilidad de la distribución ni de la relación predictiva |

**Punto de unión con el Capítulo 1 [A+B]:** $r_t=p_t-p_{t-1}$ es simultáneamente la definición del log-retorno (Cap. 1) y la primera diferencia que estacionariza el log-precio (Cap. 2). Las dos justificaciones —comparabilidad/tratabilidad y remoción de raíz unitaria— son la misma operación.

## Cadena 3 — Donde vive la estructura

$$\textbf{ACF}(r_t)\approx0\ \ +\ \ \textbf{ACF}(|r_t|)\ \text{persistente}\ \rightarrow\ \textbf{Volatility Dynamics}$$

| Eslabón | Evidencia | Consecuencia |
|---|---|---|
| **ACF($r_t$) ≈ 0** | [A] IBM mensual: dentro de bandas, $Q(5)=3.37$, p = 0.64 | La dirección tiene poca o ninguna estructura lineal |
| **ACF($\lvert r_t\rvert$) persistente** | [A] Índices CRSP diarios: significativa **tras 300 lags**, decaimiento polinomial | La magnitud tiene muchísima estructura |
| **Ambas cosas a la vez** | Compatible con $r_t=\sigma_t\varepsilon_t$ | La serie **no es iid** aunque su ACF sea nula |
| **→ Volatility dynamics** | [A] "GARCH puede considerarse un modelo ARMA para $a_t^2$" | Todo el aparato del Cap. 2 se reutiliza en el Cap. 3 |

**Ésta es la cadena que da sentido al capítulo entero [B]:** el Capítulo 2 enseña un lenguaje que, aplicado a los retornos, encuentra poco; aplicado a la magnitud de los retornos, encuentra mucho. Y el Capítulo 3 es la aplicación sistemática de ese lenguaje al segundo objeto.

## Los tres nodos transversales

- **§2.9 Regression with time series errors** conecta el marco univariado con el multivariado, y muestra que ignorar la dependencia residual produce $R^2$ = 96.5% engañosos. Remite al Cap. 8.
- **§2.10 HAC** es el corrector de inferencia que se aplica a **todo lo anterior**: cualquier t-ratio calculado sobre datos con dependencia está inflado.
- **§2.11 Long memory** es el diagnóstico que revela que la persistencia en volatilidad es de un orden distinto al de cualquier ARMA estacionario.


---

# 5. Implicancias para mercados de futuros

## 5.1 Lo que Tsay establece, sin referencia a futuros — [A]

1. La estacionariedad es el fundamento del análisis de series temporales; el libro trabaja con estacionariedad **débil** (media constante, autocovarianza dependiente sólo del lag, primeros dos momentos finitos).
2. La estacionariedad débil puede comprobarse empíricamente **dividiendo los datos en submuestras y verificando la consistencia de los resultados**.
3. La correlación mide dependencia **lineal**. La ausencia de correlación implica independencia **si y sólo si** las variables son normales.
4. Los tests de autocorrelación (individual con $1/T$, Box–Pierce, Ljung–Box) suponen **iid**; muchos paquetes usan $1/T$ asumiendo implícitamente iid.
5. Las series de precios de activos "tienden a ser no estacionarias" porque no hay un nivel fijo para el precio.
6. Los log-retornos son la primera diferencia del log-precio; si el log-precio es ARIMA($p,1,q$), los log-retornos son ARMA estacionarios. Tsay lo enuncia como creencia común en finanzas, no como resultado.
7. Bajo random walk, el precio **no es predecible ni revierte a la media**; el pronóstico óptimo a cualquier horizonte es el valor actual, y la varianza del error de pronóstico diverge.
8. El $R^2$ es válido como bondad de ajuste **sólo para series estacionarias**: con raíz unitaria converge a 1 independientemente del modelo verdadero.
9. La dependencia lineal detectable en retornos de índices es **estadísticamente significativa pero de magnitud pequeña** (AR(3) sobre índice VW mensual: $R^2=2.46\%$, coeficientes ≈ 0.11).
10. Para series de retornos en finanzas "la probabilidad de usar modelos ARMA es baja"; el concepto es relevante para volatilidad, donde GARCH es un ARMA sobre $a_t^2$.
11. El bid–ask bounce puede introducir una estructura MA(1) en una serie de retornos; la forma de determinar precios y calcular retornos de índices puede introducir autocorrelaciones, especialmente en alta frecuencia.
12. Ignorar la dependencia serial en los residuos de una regresión es uno de los errores econométricos más comunes, y produce $R^2$ y coeficientes engañosos.
13. Ignorar correlación serial y heterocedasticidad **infla los t-ratios**; en un ejemplo concreto, de 107.91 a 39.92.
14. La ACF de $|r_t|$ de índices diarios es pequeña en magnitud pero **significativa incluso después de 300 lags**, con decaimiento polinomial (memoria larga), mientras la ACF de $r_t$ es prácticamente nula.
15. Los modelos estacionales son útiles para valuar derivados climáticos y **futuros de energía**, porque las series medioambientales tienen estacionalidad fuerte. *(Única mención explícita de futuros en el capítulo.)*

## 5.2 Adaptación a mercados de futuros — [B], enteramente propia

**Sobre estacionariedad**

- No cabe esperar estacionariedad en precios de futuros, por la misma razón que da Tsay para acciones, más dos razones propias: el precio incorpora cost of carry y el contrato vence. La estacionariedad de los **retornos** es una hipótesis a verificar por instrumento y por período, no una propiedad garantizada.
- La receta de submuestras [A] se traduce en cortes naturales de futuros: año calendario, régimen de volatilidad, sesión (asiática/europea/americana), y épocas separadas por cambios estructurales de mercado. Es el diagnóstico más barato y más informativo del capítulo.
- **Un test de raíz unitaria sobre una serie continua ajustada no testea el precio de ningún instrumento real.** Si se testeara, lo mínimo sería hacerlo sobre precios de un contrato dentro de su vida activa o sobre la serie ratio-ajustada. Conecta con la corrección 6 del informe del Capítulo 1.
- Las series de futuros donde la pregunta de raíz unitaria es genuinamente interesante son los **spreads y diferenciales** (calendario, ES−NQ, ZN−ZB, base spot-futuro): candidatas naturales a ser estacionarias mientras sus componentes no lo son. Pero eso es cointegración → **PREGUNTA ABIERTA, Cap. 8.**
- Distinguir empíricamente **trend-stationary** de **raíz unitaria con drift** es difícil y tiene consecuencias operativas opuestas (hay o no reversión alrededor de una tendencia). El Capítulo 2 muestra por qué es difícil: ambas producen gráficos con aspecto de tendencia.

**Sobre autocorrelación**

- Cabe **esperar** autocorrelación cercana a cero en retornos diarios de futuros líquidos, por argumento de arbitraje. Es una expectativa, no un hallazgo del capítulo, y debe medirse.
- La advertencia de Tsay sobre autocorrelación inducida por la construcción del dato es directamente aplicable a alta frecuencia. **Una ACF significativa a alta frecuencia no es, por sí sola, evidencia de señal.** → **PREGUNTA ABIERTA, Cap. 5.**
- Un futuro sobre índice y su índice cash subyacente **no son intercambiables** a efectos de ACF: el índice cash hereda el problema de agregación de componentes no sincrónicos que Tsay señala; el futuro cotiza directamente. Mezclarlos confunde dos fenómenos.
- El mecanismo de bid–ask bounce aplica a futuros, pero en futuros líquidos con spread de un tick su magnitud está acotada al orden de $\text{tick}/P$. Expectativa, no resultado.

**Sobre la estructura donde vive la información**

- El contraste ACF($r_t$) ≈ 0 vs ACF($|r_t|$) persistente [A] es el hallazgo con más consecuencias para el proyecto. Sugiere que en futuros la magnitud tendrá mucha más estructura detectable que la dirección — **pero eso debe verificarse por instrumento y frecuencia, no asumirse por analogía con índices accionarios de Tsay.**
- **Más estructura detectable no es más rentabilidad.** La volatilidad predecible informa sobre riesgo, no sobre dirección. Convertirla en P&L requiere un mecanismo adicional que este capítulo no provee. → **PREGUNTA ABIERTA.**

**Sobre relaciones entre instrumentos**

- La hipótesis $r^{ES}_{t-k}\rightarrow r^{NQ}_t$ cae en el marco de §2.9. El capítulo no dice si existe; dice cómo plantearla sin error: trabajar sobre retornos y no sobre precios, verificar siempre la ACF de los residuos, distinguir contemporáneo de rezagado, y corregir la inferencia.
- El ejemplo de tasas de Tsay es **contemporáneo**: mide co-movimiento, no predicción. Una relación contemporánea, por fuerte que sea, no es utilizable para predecir.
- Tsay admite que ignora el problema de simultaneidad y que lo correcto sería un análisis multivariado → **PREGUNTA ABIERTA, Cap. 8.**
- El **Ejercicio 2.14** [A] es el molde metodológico más cercano a nuestro caso: regresión con errores de series temporales entre log-precios del futuro del S&P 500 y su spot, datos de 1 minuto. La secuencia es: diferenciar primero, regresar después, modelar los residuos.

**Sobre estacionalidad y calendario**

- La mención de futuros de energía [A] se refiere a estacionalidad de los **fundamentales**, no de los retornos del futuro. El precio del futuro descuenta la estacionalidad conocida. Confundir ambas cosas sería un error de razonamiento.
- En futuros, la pregunta previa más importante sobre patrones intradía es **si están en la media o en la varianza**. Es plausible que la volatilidad tenga un patrón fuerte y la media ninguno. → **PREGUNTA ABIERTA, Cap. 3.**
- Las variables de calendario son features causalmente disponibles y de coste cero, lo que las hace atractivas y peligrosas: son el terreno más fácil para data snooping.

---

# 6. Implicancias para Machine Learning

Todas las filas de esta sección son **[B]** salvo donde se indique el origen. Ninguna constituye una decisión metodológica.

## 6.1 Data representation

| # | Implicancia potencial | Origen | Condición de validez |
|---|---|---|---|
| D1 | Modelar sobre variables plausiblemente estacionarias; los precios no lo son | [A] §2.7 + [B] | La estacionariedad de los retornos es hipótesis, no garantía |
| D2 | $r_t=p_t-p_{t-1}$ es simultáneamente el log-retorno y la diferencia que remueve la raíz unitaria | [A] §2.7.4 | Vale si hay **exactamente una** raíz unitaria y no es trend-stationary |
| D3 | Diferenciar no es la única forma de estacionarizar: para trend-stationary corresponde remover la tendencia | [A] §2.7.3 | Distinguir ambos casos es difícil empíricamente |
| D4 | La diferenciación fraccional ofrece un continuo entre memoria y estacionariedad | [A] §2.11 | **Hipótesis metodológica abierta**, no decisión. Requiere cálculo causal o es canal de leak |
| D5 | Series derivadas de futuros (volumen, open interest, VI en niveles) casi con certeza no son estacionarias en niveles | [B] | No verificado |
| D6 | La transformación logarítmica estabiliza la variabilidad | [A] §2.8 | Caso particular del principio general de homogeneizar varianza |

## 6.2 Feature engineering

| # | Implicancia potencial | Origen | Condición |
|---|---|---|---|
| F1 | La ACF de un feature indica cuánta redundancia hay entre observaciones consecutivas | [A] §2.2 + [B] | Un feature muy autocorrelacionado aporta pocas observaciones efectivas |
| F2 | Features de volatilidad/magnitud son candidatos de primer orden: es donde hay estructura documentada | [A] §2.11 | Verificado en índices accionarios de Tsay, **no en futuros** |
| F3 | Los shocks $a_t$ pueden usarse como features, pero son objetos **estimados**, no datos | [A] §2.3 + [B] | Introducen dependencia de especificación |
| F4 | Las variables de calendario son features causalmente disponibles | [A] Ejercicios 2.7–2.9 + [B] | Alto riesgo de data snooping por multiplicidad |
| F5 | La estandarización debe ser causal; con no estacionariedad, una estandarización global es canal de leak | [B] | No está en Tsay |
| F6 | Features de nivel sobre series ratio-ajustadas: ver la distinción escala-invariante del informe del Cap. 1 | [B] | — |

## 6.3 Target design

| # | Implicancia potencial | Origen | Condición |
|---|---|---|---|
| T1 | El pronóstico óptimo bajo pérdida cuadrática es $E(r_{h+\ell}\mid F_h)$ | [A] §2.4.4 | Define qué se está estimando al usar MSE |
| T2 | En horizontes largos, el pronóstico óptimo de un modelo estacionario **converge al benchmark trivial** (media incondicional) | [A] §2.4.4, §2.6.5 | Demostrado para modelos **lineales estacionarios**. ¿Vale para no lineales? → **PREGUNTA ABIERTA, Cap. 4** |
| T3 | La varianza del error de pronóstico es no decreciente en el horizonte | [A] Ec. 2.34 | Bajo el modelo; no es una ley empírica |
| T4 | Dirección y magnitud son objetivos con cantidades de estructura muy distintas | [A] §2.2 vs §2.11 + [B] | Verificado en índices accionarios; hipótesis en futuros |
| T5 | Targets solapados generan autocorrelación residual severa por construcción, agravando §2.10 | [B] | No está en Tsay |
| T6 | Un modelo puede estar bien calibrado y ser inútil: en la Tabla 2.2, el intervalo al 95% es ~10× el pronóstico puntual | [A] + [B] | Calibración ≠ utilidad económica |

## 6.4 Baseline models

Jerarquía de benchmarks que se desprende del capítulo. **No se propone adoptar ninguno como modelo del sistema.**

| Nivel | Benchmark | Qué asume | Qué probaría superarlo |
|---|---|---|---|
| 0 | Pronóstico incondicional $\hat r=\hat\mu$ | Estacionariedad, sin dependencia | Que existe *alguna* información condicional |
| 1 | Random walk ($\hat P_{t+1}=P_t$) | No predecibilidad | La nula del Cap. 1, ahora operativa |
| 2 | AR($p$), orden por AIC/BIC | Dependencia lineal en la media | Que la información no es puramente lineal en lags propios |
| 3 | ARMA | Dependencia lineal parsimoniosa | Idem, con memoria de shocks |
| 4 | Regresión con errores ARMA (§2.9) | Dependencia lineal con exógenas | Que la información cross-market no es puramente lineal |

**Advertencias [B]:** los niveles 0 y 1 casi coinciden en retornos porque $\hat\mu\approx0$. Superar un benchmark debe medirse **out-of-sample**, no por $R^2$ in-sample — y el Capítulo 2 no dice nada sobre validación. Un modelo complejo que no supera al AR no es necesariamente inútil: puede capturar lo mismo con más varianza.

## 6.5 Model diagnostics

| # | Implicancia potencial | Origen | Condición |
|---|---|---|---|
| G1 | La ACF y Ljung–Box de los residuos son **transferibles a modelos de ML**: nada exige que $\hat\mu_t$ venga de un modelo lineal | [A] §2.4.2 + [B] | Es la herramienta más directamente reutilizable del capítulo |
| G2 | Examinar ACF de $\hat a_t$, de $\lvert\hat a_t\rvert$ y de $\hat a_t^2$ por separado: responden preguntas distintas | [A] §2.11 + [B] | — |
| G3 | El ajuste de grados de libertad de Ljung–Box sobre residuos ($m-g$) **no está definido** para un modelo de ML con parámetros efectivos ambiguos | [A] §2.4.2 + [B] | **Limitación metodológica a registrar** |
| G4 | Preferir Ljung–Box sobre Durbin–Watson: DW sólo ve el lag 1 | [A] §2.9 | Relevante con estructura intradía o semanal |
| G5 | En series con estacionalidad, mirar los lags múltiplos del período | [A] §2.2 | — |
| G6 | La elección de $m$ afecta el resultado y es un grado de libertad del analista | [A] §2.2 | Probar muchos $m$ y reportar el favorable es data snooping |
| G7 | El $R^2$ evaluado sobre variables no estacionarias puede ser espectacular y vacío | [A] §2.4.3 | Aplica a cualquier métrica comparada en niveles |

## 6.6 Validation

| # | Implicancia potencial | Origen | Condición |
|---|---|---|---|
| V1 | El ciclo identificar→estimar→verificar de Tsay es **circular sobre el mismo dataset** y consume grados de libertad que ningún test contabiliza | [A] + [B] | No está en Tsay; es crítica propia |
| V2 | La receta de submuestras [A] extendida a *estimar la relación en submuestras disjuntas y comparar* es un diagnóstico de **estabilidad**, no validación out-of-sample | [A] §2.1 + [B] | No confundir ambas cosas |
| V3 | Con memoria larga en volatilidad (300+ lags), un embargo corto entre train y test puede ser insuficiente | [A] §2.11 + [B] | La longitud debería relacionarse con la persistencia observada → **PREGUNTA ABIERTA** |
| V4 | Estacionariedad de las variables ≠ estabilidad de la función predictiva: testear la primera es necesario pero no suficiente | [B] | Tsay no formula esta distinción |
| V5 | Evaluar por submuestra/régimen, no sólo en agregado | [B] | — |

**No se adopta ningún esquema de validación en esta etapa.**

## 6.7 Statistical inference

| # | Implicancia potencial | Origen | Condición |
|---|---|---|---|
| I1 | Los t-ratios ingenuos están inflados en presencia de correlación serial y/o heterocedasticidad; factor ~2.7 en el ejemplo de Tsay | [A] §2.10 | Magnitud del ejemplo, no ley general |
| I2 | HAC/HC corrigen la **covarianza**, presuponiendo que el estimador puntual sigue siendo consistente | [A] §2.10, explícito | No corrigen sesgo, mala especificación ni look-ahead |
| I3 | El problema aparece en: análisis univariado de features, regresiones predictivas, t-stat de un Sharpe, comparación entre modelos, análisis por deciles de señal | [B] | Extensión propia; Tsay sólo trata regresión |
| I4 | Alternativas conceptuales: bootstrap por bloques, inferencia por submuestras, reducir frecuencia de evaluación | [B] | Ninguna decidida |
| I5 | Alternativa estructural a corregir la inferencia: **especificar mejor el modelo** (añadir rezagos reduce la autocorrelación residual) | [A] §2.10 | Tsay muestra ambos caminos sin declarar uno superior |
| I6 | La regla $\ell=\lfloor4(T/100)^{2/9}\rfloor$ crece muy lentamente; con targets solapados de horizonte $h$, el truncamiento debería cubrir al menos $h$ | [A] + [B] | **PREGUNTA ABIERTA** sobre el bandwidth apropiado |

## 6.8 Multivariate modeling

| # | Implicancia potencial | Origen | Condición |
|---|---|---|---|
| M1 | Regresión entre precios de dos futuros reproduce el error del $R^2=96.5\%$ "seriamente inadecuado" de Tsay | [A] §2.9 + [B] | Directamente aplicable |
| M2 | Verificar la ACF de los residuos es el paso 1 y "el que más comúnmente se pasa por alto" | [A] §2.9 | — |
| M3 | La forma de la ACF residual indica el remedio: decaimiento lento → diferenciar; ACF pequeña que se corta → modelar con ARMA | [A] §2.9 | Árbol de decisión concreto y transferible |
| M4 | Relación contemporánea ≠ relación predictiva | [B] | El ejemplo de Tsay es contemporáneo |
| M5 | La regresión impone una dirección causal que los datos no establecen; Tsay admite ignorar la simultaneidad | [A] §2.9 | → **PREGUNTA ABIERTA, Cap. 8** |
| M6 | Apilar instrumentos correlacionados **no** multiplica el tamaño de muestra independiente | [B] | Extensión de la lógica de §2.10; no está en Tsay |

## 6.9 Risk management

| # | Implicancia potencial | Origen | Condición |
|---|---|---|---|
| R1 | La persistencia de la volatilidad es documentada y de decaimiento polinomial: la incertidumbre del mañana es parcialmente previsible | [A] §2.11 | En índices accionarios diarios |
| R2 | La varianza del error de pronóstico crece con el horizonte y converge a la varianza incondicional | [A] §2.4.4, Ec. 2.34 | Bajo modelo estacionario |
| R3 | Bajo random walk, el intervalo de pronóstico diverge: el riesgo de horizonte largo no está acotado por el modelo | [A] §2.7.1 | — |
| R4 | Un modelo de ML con residuos heterocedásticos tiene predicciones puntuales potencialmente correctas y **dimensionamiento incorrecto** | [B] | Consecuencia de la tabla de §2.3 |
| R5 | Los intervalos de la Tabla 2.2 son ~10× el pronóstico puntual: la incertidumbre domina la señal | [A] + [B] | En retornos mensuales de índice |

---

# 7. Hipótesis a comprobar empíricamente

Backlog para después de completar el estudio teórico. **No ejecutar todavía.**

### Bloque A — Estacionariedad

**H2.1 — ¿Son estables los momentos de los retornos entre submuestras?**
1. *Pregunta:* ¿Se sostiene la estacionariedad débil de los retornos del instrumento objetivo?
2. *Método:* Receta de Tsay [A]: partir en submuestras (por año, por régimen, por sesión) y comparar media, varianza y ACF.
3. *Apoya:* momentos consistentes entre submuestras.
4. *Refuta:* divergencia sistemática, especialmente en varianza.
5. *Limitación:* la varianza divergirá casi con certeza por clustering — lo cual **no** refuta estacionariedad débil, sólo indica heterocedasticidad condicional. El test no distingue ambas cosas → Cap. 3.

**H2.2 — ¿Rechaza ADF la raíz unitaria en el log-precio de un contrato individual?**
1. *Pregunta:* ¿Es el log-precio compatible con un random walk?
2. *Método:* ADF con $c_t$ constante y con constante+tendencia, varios $p$.
3. *Apoya (raíz unitaria):* no rechazo robusto a la elección de $p$ y de $c_t$ — como en el ejemplo del S&P de Tsay.
4. *Refuta:* rechazo consistente.
5. *Limitación:* **potencia baja contra $\phi_1$ cercano a 1**; sensibilidad a $c_t$; cambios estructurales; y sobre una serie continua ajustada el objeto testeado no es un precio real.

**H2.3 — ¿Son los retornos más cercanos a estacionarios que los precios, y en qué sentido?**
1. *Pregunta:* cuantificar la diferencia, no asumirla.
2. *Método:* comparar estabilidad de momentos entre submuestras para $P_t$ y para $r_t$; ADF sobre ambos.
3. *Apoya:* estabilidad claramente mayor en retornos.
4. *Refuta:* retornos también inestables en media entre submuestras.
5. *Limitación:* no distingue estacionariedad débil de estricta.

### Bloque B — ACF y white noise

**H2.4 — ¿Es la ACF de los retornos indistinguible de cero?**
1. *Pregunta:* ¿hay dependencia lineal en la media?
2. *Método:* ACF con bandas de Bartlett (no $1/T$), Ljung–Box con varios $m$ incluyendo $m\approx\ln T$.
3. *Apoya:* ACF dentro de bandas, $Q(m)$ no significativo.
4. *Refuta:* lags significativos con Bartlett y consistentes entre submuestras.
5. *Limitación:* **el test supone iid** [A]; con clustering de volatilidad los p-valores están mal calibrados. Un rechazo puede reflejar heterocedasticidad, no autocorrelación.

**H2.5 — ¿Es la ACF significativa a alta frecuencia y desaparece al agregar?**
1. *Pregunta:* ¿la dependencia lineal observada a frecuencias altas es microestructural?
2. *Método:* ACF de retornos a 1min, 5min, 15min, 1h, diario; comparar precios de transacción vs mid.
3. *Apoya (origen microestructural):* estructura MA(1) negativa que se atenúa al agregar y al usar mid.
4. *Refuta:* estructura que persiste al agregar y es independiente del tipo de precio.
5. *Limitación:* la interpretación completa requiere el **Cap. 5**. En esta etapa sólo se describe el fenómeno.

**H2.6 — ¿Es la ACF estable entre submuestras o cambia de signo?**
1. *Pregunta:* ¿una ACF global cercana a cero está ocultando dependencia que alterna por régimen?
2. *Método:* ACF por submuestra; comparar signos y magnitudes.
3. *Apoya (inestabilidad):* signos opuestos entre submuestras con ACF global ≈ 0.
4. *Refuta:* ACF consistentemente nula en todas las submuestras.
5. *Limitación:* con submuestras cortas la varianza de $\hat\rho_\ell$ crece; puede confundirse ruido con cambio de régimen.

### Bloque C — Magnitud y memoria larga

**H2.7 — ¿Presenta $|r_t|$ memoria larga en futuros?**
1. *Pregunta:* ¿se replica el hallazgo de Tsay [A] fuera de índices accionarios?
2. *Método:* ACF de $|r_t|$ y $r_t^2$ hasta 100–300 lags (molde del Ejercicio 2.5 [A]); examinar si el decaimiento es polinomial o exponencial.
3. *Apoya:* ACF pequeña en magnitud que decae lentamente y sigue significativa a lags altos.
4. *Refuta:* decaimiento exponencial que muere en pocas decenas de lags.
5. *Limitación:* memoria larga aparente puede ser artefacto de cambios de régimen en volatilidad → **Caps. 3, 11–12**. La estimación de $d$ no se aborda aquí.

**H2.8 — ¿Cuánta más estructura hay en magnitud que en dirección?**
1. *Pregunta:* cuantificar la asimetría del contraste ACF($r_t$) vs ACF($|r_t|$).
2. *Método:* comparar $Q(m)$ y magnitudes de ACF para ambos objetos, mismo $m$, mismo período.
3. *Apoya:* diferencia de órdenes de magnitud.
4. *Refuta:* estructuras comparables.
5. *Limitación:* **más estructura detectable no es más rentabilidad**; la magnitud no dice de qué lado ponerse.

### Bloque D — Benchmarks y modelado

**H2.9 — ¿Supera un AR($p$) al pronóstico incondicional fuera de muestra?**
1. *Pregunta:* ¿existe información lineal explotable en lags propios?
2. *Método:* AR con orden por AIC y por BIC; evaluación estrictamente out-of-sample contra $\hat\mu$ y contra random walk.
3. *Apoya:* mejora consistente y estable entre períodos.
4. *Refuta:* mejora nula o negativa fuera de muestra.
5. *Limitación:* AIC y BIC pueden discrepar mucho (9 vs 1 en el ejemplo de Tsay [A]); el $R^2$ in-sample no es el criterio.

**H2.10 — ¿Cuál es la magnitud del $R^2$ alcanzable con estructura lineal en futuros?**
1. *Pregunta:* ¿es del orden del 2.46% que Tsay obtiene en índices mensuales, mayor o menor?
2. *Método:* AR/ARMA sobre retornos de futuros a varias frecuencias, $R^2$ y $R^2$ ajustado.
3. *Apoya el orden de magnitud:* valores de unidades porcentuales o menos.
4. *Refuta:* valores muy superiores — que deberían entonces auditarse como posible error metodológico antes que como hallazgo.
5. *Limitación:* comparar $R^2$ entre frecuencias e instrumentos requiere cuidado; y el $R^2$ sobre variables no estacionarias no es interpretable [A].

**H2.11 — ¿Los residuos de un modelo pasan Ljung–Box, y qué queda en $|\hat a_t|$?**
1. *Pregunta:* ¿el modelo agotó la estructura lineal en la media? ¿queda estructura en la magnitud?
2. *Método:* ACF y $Q(m)$ sobre $\hat a_t$, $|\hat a_t|$ y $\hat a_t^2$.
3. *Apoya adecuación lineal:* $Q(m)$ no significativo sobre $\hat a_t$.
4. *Refuta:* autocorrelación residual significativa.
5. *Limitación:* pasar Ljung–Box **no** establece independencia; y el ajuste de grados de libertad no está definido para modelos de ML.

### Bloque E — Inferencia y relaciones entre instrumentos

**H2.12 — ¿Cuánto cambia la evidencia estadística al usar HC/HAC?**
1. *Pregunta:* ¿cuál es el factor de inflación de los t-ratios en nuestras aplicaciones?
2. *Método:* replicar el ejercicio de Tsay: mismo coeficiente con errores estándar convencional, HC y HAC.
3. *Apoya la relevancia del problema:* reducción sustancial del t-ratio, del orden del factor 2–3 del ejemplo.
4. *Refuta:* cambio marginal.
5. *Limitación:* HAC no corrige sesgo ni mala especificación; la elección del bandwidth afecta el resultado.

**H2.13 — ¿Existe relación rezagada entre dos futuros, tras diferenciar y corregir la inferencia?**
1. *Pregunta:* ¿hay lead-lag lineal detectable?
2. *Método:* molde del Ejercicio 2.14 [A]: diferenciar, regresar con rezagos estrictos, verificar ACF de residuos, aplicar HAC.
3. *Apoya:* coeficiente rezagado significativo con HAC y estable entre submuestras.
4. *Refuta:* significancia que desaparece con HAC o entre submuestras.
5. *Limitación:* la regresión ignora la simultaneidad [A]; el marco correcto es multivariado → **Cap. 8**. Posibles artefactos de sincronización de sesiones.

**H2.14 — ¿Hay efectos de calendario en media, en varianza, o en ninguna?**
1. *Pregunta:* ¿existen patrones de día de semana / hora del día?
2. *Método:* molde de los Ejercicios 2.7–2.9 [A]: regresión con indicadores, $Q(12)$ sobre residuos, estimador HAC, y si hace falta regresión con errores de series temporales. Repetir con $|r_t|$ como dependiente.
3. *Apoya:* efecto significativo con HAC, estable entre años, con mecanismo identificable.
4. *Refuta:* significancia que no sobrevive a HAC ni a la corrección por multiplicidad.
5. *Limitación:* los ejercicios de Tsay **no incluyen corrección por multiplicidad** [B]; con muchas horas, días e instrumentos, los falsos positivos están garantizados sin ella.

---

# 8. Errores metodológicos a evitar

## 8.1 Auditoría de las once afirmaciones

| # | Afirmación | Veredicto |
|---|---|---|
| 1 | "ACF ≈ 0 ⇒ mercado impredecible" | **INCORRECTA** |
| 2 | "ADF rechaza raíz unitaria ⇒ la serie permanecerá estacionaria" | **INCORRECTA** |
| 3 | "ADF no rechaza ⇒ demostramos raíz unitaria" | **INCORRECTA** |
| 4 | "AR significativo ⇒ estrategia rentable" | **INCORRECTA** |
| 5 | "PACF muestra $p$ lags ⇒ un modelo ML debe usar lookback $p$" | **INCORRECTA** |
| 6 | "Residuos sin autocorrelación ⇒ residuos independientes" | **REQUIERE CONDICIONES** (falsa salvo normalidad conjunta) |
| 7 | "Ljung–Box no significativo ⇒ no queda ninguna información" | **INCORRECTA** |
| 8 | "Long memory ⇒ señal direccional" | **INCORRECTA** |
| 9 | "Seasonality ⇒ debemos usar SARIMA" | **REQUIERE CONDICIONES** |
| 10 | "Newey–West soluciona cualquier dependencia temporal" | **INCORRECTA** |
| 11 | "Más instrumentos ⇒ mayor tamaño de muestra independiente" | **INCORRECTA** |

**1. "ACF ≈ 0 ⇒ mercado impredecible" — INCORRECTA.**
La ACF mide **sólo dependencia lineal en la media**, promediada sobre la muestra. La ausencia de correlación implica independencia **si y sólo si** las variables son normales [A, §2.2] — y el Capítulo 1 refuta la normalidad de forma abrumadora. El contraejemplo está en el propio libro: la ACF de $|r_t|$ es significativa tras 300 lags mientras la de $r_t$ es nula [A, §2.11]. Una serie así **no es iid** aunque todas sus ACF sean cero.

**2. "ADF rechaza raíz unitaria ⇒ la serie permanecerá estacionaria" — INCORRECTA.**
Dos errores superpuestos. Primero, el rechazo es evidencia **contra** la raíz unitaria durante el período muestral; no es una afirmación sobre el futuro. Segundo, rechazar la raíz unitaria no identifica el proceso alternativo: puede ser un AR estacionario, pero también puede ser trend-stationary, y la conclusión depende de qué $c_t$ se incluyó [A, §2.7.5]. La estacionariedad es una propiedad del proceso generador dentro de la muestra, no una garantía perpetua.

**3. "ADF no rechaza ⇒ demostramos raíz unitaria" — INCORRECTA.**
La raíz unitaria está en la **hipótesis nula**, de modo que la ausencia de evidencia se convierte en "no rechazo", nunca en confirmación. No rechazar significa que la serie es compatible con un random walk **y también** con un AR(1) de $\phi_1=0.999$, y con uno de $\phi_1=0.98$ si la muestra es corta. El propio ejemplo de Tsay tiene $\hat\beta=0.9992$ [A]. La potencia contra alternativas cercanas es baja — **esto último es [B]**, no lo discute Tsay. Nótese que el propio texto se desliza: escribe "no puede rechazarse" y luego resume que la serie "contiene una raíz unitaria" [A]. Ese deslizamiento es el que hay que evitar.

**4. "AR significativo ⇒ estrategia rentable" — INCORRECTA.**
El contraejemplo es del propio Tsay: el AR(3) sobre el índice VW tiene coeficientes significativos al 1% y $R^2=2.46\%$ con coeficientes ≈ 0.11, y Tsay comenta explícitamente que "la dependencia serial de la serie es **débil, aunque sea estadísticamente significativa** al nivel del 1%" [A]. Significancia estadística mide confianza en que el efecto no es cero; no mide su magnitud. A eso se añaden [B] los costos de transacción, que el capítulo no trata en absoluto.

**5. "PACF muestra $p$ lags ⇒ un modelo ML debe usar lookback $p$" — INCORRECTA.**
La PACF responde una pregunta muy específica: *la contribución lineal añadida del lag $\ell$ sobre los $\ell-1$ anteriores, en la media condicional, bajo estacionariedad, dentro de un marco AR* [A]. Cuatro razones por las que no se traslada:
- Un modelo no lineal puede extraer información de lags donde la PACF es cero.
- La propiedad de corte es para AR($p$) gaussiano [A]; para un ARMA, **la PACF no se corta en ningún lag finito** [A, §2.6.1].
- Los criterios discrepan entre sí: en el ejemplo de Tsay, AIC elige 9 y BIC elige 1 [A].
- El lookback de un modelo de ML es un hiperparámetro con trade-offs (capacidad, muestra efectiva, coste) que la PACF no contempla.
La PACF es **informativa** sobre profundidad temporal razonable a explorar. No es un selector.

**6. "Residuos sin autocorrelación ⇒ residuos independientes" — REQUIERE CONDICIONES, y las condiciones no se cumplen.**
Es válido **si y sólo si** los residuos son conjuntamente normales [A, §2.2]. En datos financieros no lo son: los residuos del AR(3) de Tsay tienen **Jarque–Bera = 1656 con p = 0.0000** [A]. Nótese que Tsay define *white noise* como **iid** [A, §2.3], más fuerte que "no correlacionado", y sin embargo escribe la heurística práctica "si todas las ACF muestrales están cerca de cero, entonces la serie es white noise". Tomada literalmente con su propia definición, esa frase es una inferencia inválida, y su §2.11 la refuta. Debe leerse como heurística dentro del alcance del modelado lineal de la media.

**7. "Ljung–Box no significativo ⇒ no queda ninguna información" — INCORRECTA.**
$Q(m)$ testea conjuntamente $\rho_1=\dots=\rho_m=0$: sólo autocorrelación lineal, sólo hasta el lag $m$, **bajo supuesto iid** [A]. Cuatro limitaciones: la elección de $m$ afecta el resultado [A]; el supuesto iid falla con clustering de volatilidad; sobre residuos se requiere ajustar grados de libertad y la mayoría del software no lo hace [A]; y el test tiene potencia limitada. No pasar el test es informativo; pasarlo sólo dice que no hay estructura **lineal en la media detectable con esa configuración**.

**8. "Long memory ⇒ señal direccional" — INCORRECTA.**
La memoria larga que Tsay documenta está en $|r_t|$ [A, §2.11], no en $r_t$. Es persistencia de la **magnitud**. Un proceso $r_t=\sigma_t\varepsilon_t$ con $\sigma_t$ persistente y $\varepsilon_t$ iid de media cero tiene memoria larga en $|r_t|$ y esperanza condicional **nula**. Saber que mañana será volátil no dice de qué lado ponerse. Es información sobre riesgo; convertirla en dirección requiere un mecanismo adicional → **PREGUNTA ABIERTA**.

**9. "Seasonality ⇒ debemos usar SARIMA" — REQUIERE CONDICIONES.**
El ejemplo de Tsay es de ganancias trimestrales corporativas, con causa institucional clara [A]. Antes de trasladarlo hay que distinguir estacionalidad **determinística** (etiqueta conocida de antemano: hora, día, calendario → regresión con indicadores, que es el enfoque de los propios Ejercicios 2.7–2.9 [A]) de **dependencia estocástica estacional** (correlación en lags múltiplos del período → operadores $\Delta_s$). Además hay que preguntar si el patrón está en la media o en la varianza, si es estable entre años, si sobrevive a la multiplicidad, y si su magnitud supera los costos. Y una alternativa a modelar la estacionalidad es **condicionar sobre ella** usando la etiqueta como feature. Ninguna de estas opciones está decidida.

**10. "Newey–West soluciona cualquier dependencia temporal" — INCORRECTA.**
Tsay condiciona explícitamente: HAC provee una estimación consistente de la covarianza **"en situaciones bajo las cuales las estimaciones OLS de los coeficientes siguen siendo consistentes"** [A, §2.10]. Corrige **errores estándar**, no el estimador puntual. No corrige sesgo, ni mala especificación, ni look-ahead bias, ni un esquema de validación temporal defectuoso, ni data snooping. Es un corrector de inferencia, no un sustituto de un diseño experimental correcto. **[B]**

**11. "Más instrumentos ⇒ mayor tamaño de muestra independiente" — INCORRECTA.**
Ésta es **enteramente [B]**: Tsay no la trata en el Capítulo 2. Pero la lógica de §2.10 se extiende directamente: si las observaciones están correlacionadas, el tamaño de muestra efectivo es menor que el nominal. Apilar $N$ futuros altamente correlacionados contemporáneamente no multiplica por $N$ la información independiente, y toda medida de significancia calculada sobre el conjunto apilado estará inflada. Conecta con la corrección 4 del informe del Capítulo 1.

## 8.2 Otros errores metodológicos del capítulo

12. **Evaluar bondad de ajuste sobre variables no estacionarias** [A, §2.4.3]: el $R^2$ converge a 1 con raíz unitaria, independientemente del modelo verdadero.
13. **Ignorar la dependencia serial de los residuos en una regresión** [A, §2.9]: "uno de los modelos econométricos más comúnmente mal utilizados".
14. **Confiar en las bandas de confianza por defecto de un gráfico de ACF** [A, §2.2]: casi todo el software usa $1/T$, asumiendo iid. Con clustering, las bandas son demasiado estrechas.
15. **Usar Durbin–Watson como diagnóstico general** [A, §2.9]: sólo ve el lag 1.
16. **Sobrediferenciar una serie trend-stationary** [B, derivado de A §2.7.3]: introduce un factor $(1-B)$ en el polinomio MA, con problemas de invertibilidad. La transformación correcta sería remover la tendencia.
17. **El ciclo identificar→estimar→verificar sobre el mismo dataset** [B]: cada iteración consume grados de libertad que ningún test contabiliza. Es data snooping de bajo grado, sistemático y no reportado.
18. **Elegir la especificación de $c_t$ en ADF que da el resultado deseado** [B, sobre A §2.7.5]: la distribución asintótica cambia según lo que se incluya.
19. **Confundir estacionalidad del subyacente físico con estacionalidad de los retornos del futuro** [B]: el precio del futuro descuenta la estacionalidad conocida.
20. **Confundir calibración con utilidad** [B, sobre A Tabla 2.2]: intervalos diez veces más anchos que el pronóstico pueden estar perfectamente calibrados y ser inútiles para decidir.
21. **Tratar la estacionariedad de las variables como si garantizara estabilidad de la relación predictiva** [B]: son propiedades lógicamente independientes.

---

# 9. Preguntas abiertas

Registro explícito de lo que **no** debe resolverse todavía.

| Cuestión surgida en el Cap. 2 | Dónde se resuelve |
|---|---|
| ¿Qué fracción de la aparente no estacionariedad es heterocedasticidad condicional? | **Cap. 3** |
| ¿Cuál es la forma exacta de la equivalencia GARCH ↔ ARMA sobre $a_t^2$? | **Cap. 3** |
| ¿La persistencia de volatilidad roza la no estacionariedad (IGARCH)? | **Cap. 3** |
| ¿La memoria larga en $\lvert r_t\rvert$ se modela con ARFIMA, GARCH persistente o componentes múltiples? | **Cap. 3** |
| ¿El patrón intradía está en la media o en la varianza? | **Cap. 3** |
| ¿Hay dependencia no lineal que la ACF no ve? ¿Cómo se testea? | **Cap. 4** |
| ¿Converge también al benchmark trivial el pronóstico de largo horizonte de un modelo **no lineal**? | **Cap. 4** |
| ¿Cuánta autocorrelación de alta frecuencia es microestructural (bid–ask bounce, no sincronía)? | **Cap. 5** |
| ¿Una estructura MA(1) en retornos intradía es señal o artefacto? | **Cap. 5** |
| ¿Están cointegrados dos futuros relacionados? ¿Qué implicaría? | **Cap. 8** |
| ¿Cómo tratar la simultaneidad, que Tsay explícitamente ignora en §2.9? | **Cap. 8 (VAR)** |
| ¿Existe lead-lag genuino entre futuros? | **Cap. 8** + evidencia empírica |
| ¿Cuántas dimensiones efectivas tiene un conjunto de futuros correlacionados? | **Cap. 9** |
| ¿Es la correlación entre instrumentos un proceso, y cómo evoluciona? | **Cap. 10** |
| ¿La memoria larga aparente es artefacto de cambios de régimen? | **Caps. 3, 11–12** |
| ¿Importa cuantitativamente la incertidumbre de parámetros en el pronóstico? | **Cap. 12** (bayesiano/MCMC) |
| ¿Cómo se traduce persistencia de volatilidad en decisiones de trading? | **Fuera de Tsay** |
| ¿Qué benchmark corresponde cuando el objetivo es una decisión y no un pronóstico puntual? | **Fuera de Tsay** |
| ¿Qué bandwidth HAC es apropiado en cada aplicación del proyecto? | **Empírica** |
| ¿Qué longitud de embargo exige la persistencia observada? | **Empírica** |

---

# 10. Checklist de conocimientos adquiridos

Antes de pasar al Capítulo 3 deberían estar comprendidos:

**Estacionariedad**
- [ ] Definición de estacionariedad estricta y por qué es difícil de verificar.
- [ ] Definición de estacionariedad débil: las dos condiciones exactas y el supuesto implícito de momentos finitos.
- [ ] Relación entre ambas: estricta + momentos finitos ⇒ débil; el recíproco falla; bajo normalidad son equivalentes.
- [ ] Propiedades de la autocovarianza: $\gamma_0=\text{Var}$, $\gamma_{-\ell}=\gamma_\ell$.
- [ ] La receta de submuestras como verificación empírica.
- [ ] Qué **no** significa estacionariedad, y en particular que es compatible con heterocedasticidad condicional.
- [ ] La distinción entre estacionariedad de una variable y estabilidad de una función predictiva.

**ACF y tests**
- [ ] Que la correlación mide dependencia **lineal**, y la condición de normalidad para que $\rho=0\Rightarrow$ independencia.
- [ ] Definición de $\rho_\ell$ y su estimador muestral.
- [ ] Distribución asintótica bajo iid ($1/T$) y fórmula de Bartlett.
- [ ] Que el software por defecto asume iid.
- [ ] Box–Pierce y Ljung–Box: hipótesis nula, distribución, supuestos, elección de $m$, ajuste de grados de libertad sobre residuos.
- [ ] La jerarquía autocorrelación ⊂ dependencia ⊂ predecibilidad ⊂ predecibilidad neta de costos.

**White noise y series lineales**
- [ ] La definición de white noise de Tsay (**iid**) y en qué difiere de la definición débil habitual.
- [ ] Representación lineal $r_t=\mu+\sum\psi_ia_{t-i}$; qué es una innovación/shock.
- [ ] Que para una serie estacionaria el efecto de shocks remotos se desvanece.
- [ ] La cadena serie → modelo → residuo y el criterio de adecuación.
- [ ] Los cuatro niveles de "buenos residuos" y su no equivalencia.

**AR / MA / ARMA**
- [ ] AR(1) y AR($p$): media y varianza condicionales, propiedad de Markov, condición de estacionariedad, ACF exponencial, persistencia.
- [ ] Raíces características, incluido el caso complejo y los ciclos estocásticos.
- [ ] PACF: qué mide exactamente y su propiedad de corte.
- [ ] AIC y BIC: forma, diferencia de penalización, y que pueden discrepar.
- [ ] Estimación por LS condicional y diagnóstico de residuos.
- [ ] $R^2$ y su invalidez bajo raíz unitaria.
- [ ] Pronóstico: esperanza condicional bajo pérdida cuadrática, error de 1 paso = shock, varianza creciente con el horizonte, reversión a la media, vida media.
- [ ] MA($q$): siempre estacionario, ACF que se corta, memoria finita, invertibilidad.
- [ ] La conexión MA(1) ↔ bid–ask bounce que Tsay anuncia.
- [ ] ARMA: parsimonia, condición de estacionariedad, que ACF/PACF no identifican el orden.
- [ ] Las tres representaciones y la función de respuesta a impulso.
- [ ] Que Tsay considera **poco probable** el uso de ARMA en retornos, y central en volatilidad.

**Raíces unitarias**
- [ ] Random walk: no predecible, no revierte, shocks permanentes, $\psi_i=1$, varianza del error divergente, ACF muestral → 1.
- [ ] Random walk con drift: descomposición en tendencia $t\mu$ más random walk puro.
- [ ] Trend-stationary vs difference-stationary: la diferencia está en la **varianza** y en la persistencia de los shocks.
- [ ] Las tres interpretaciones del término constante.
- [ ] ARIMA y diferenciación; que $r_t$ es la primera diferencia del log-precio.
- [ ] DF y ADF: hipótesis, papel de $c_t$, distribución no estándar, valores críticos por simulación.
- [ ] Que "no rechazar" no es "demostrar".

**Regresión, HAC y memoria larga**
- [ ] Por qué la dependencia serial en los residuos invalida la inferencia y puede invalidar la consistencia.
- [ ] El procedimiento de tres pasos y el diagnóstico por la forma de la ACF residual.
- [ ] Preferencia de Ljung–Box sobre Durbin–Watson.
- [ ] HC y HAC: estructura de sándwich, pesos de Bartlett, regla del bandwidth, y la condición de consistencia del estimador puntual.
- [ ] La magnitud del efecto: 107.91 → 48.44 → 39.92.
- [ ] Los tres regímenes de decaimiento de la ACF: exponencial, polinomial, y convergencia a 1.
- [ ] Diferenciación fraccional: rango de $d$, estacionariedad, invertibilidad, ACF y PACF, densidad espectral.
- [ ] El hallazgo de la ACF de $|r_t|$ significativa tras 300 lags.
- [ ] Por qué $\text{Corr}(r_t,r_{t-k})\approx0$ y $\text{Corr}(|r_t|,|r_{t-k}|)>0$ son compatibles.

**Lo que NO se sabe todavía**
- [ ] Todo lo listado en la sección 9.

---

# 11. Conclusiones

**Cuáles son los aportes fundamentales del Capítulo 2 para construir un sistema serio de ML sobre futuros.**

**1. Define las condiciones bajo las cuales la estimación tiene sentido.** La estacionariedad débil no es un tecnicismo: es lo que permite que una sola realización sirva para estimar cantidades poblacionales. Sin ella, la ACF ni siquiera está definida, el $R^2$ converge a 1 por construcción, y los errores estándar son ficción. El Capítulo 1 dijo qué variable modelar; el Capítulo 2 dice bajo qué condiciones se la puede estudiar.

**2. Acota con precisión lo que las herramientas lineales pueden y no pueden ver.** La ACF mide dependencia lineal en la media, promediada sobre la muestra. La condición exacta que autorizaría el salto de "no correlacionado" a "independiente" —normalidad conjunta— está enunciada en el capítulo y refutada empíricamente en el anterior. Ésa es la pieza lógica que impide el error más caro del análisis financiero cuantitativo: leer una ACF plana como ausencia de estructura.

**3. Entrega el contraejemplo que reorienta el proyecto.** Tsay muestra, en distintos ejemplos, que los retornos pueden presentar muy poca autocorrelación lineal mientras que los retornos absolutos pueden exhibir persistencia durante cientos de lags. Es la demostración, dentro del propio libro, de que la ausencia de autocorrelación es compatible con una cantidad enorme de estructura, y de que esa estructura vive en la magnitud. La consecuencia para el diseño no está decidida —y no debe decidirse aquí— pero la pregunta queda formulada con precisión: ¿cuánto del esfuerzo debe ir a la dirección y cuánto a la incertidumbre?

**4. Fija una escala de realismo para la media condicional.** $R^2=2.46\%$ con coeficientes significativos al 1%. Es la traducción, en el aparato del Capítulo 2, del hallazgo del Capítulo 1 sobre el t-test de media cero. Ambos dicen lo mismo desde ángulos distintos: **la señal en el primer momento es minúscula y la significancia estadística no la agranda**. Cualquier resultado muy superior sobre datos comparables debe auditarse antes que celebrarse.

**5. Convierte "el mercado es impredecible" en una hipótesis nula testeable y en una familia de benchmarks.** El random walk deja de ser una postura filosófica y pasa a ser un modelo con propiedades derivables —pronóstico constante, varianza divergente, shocks permanentes— y un test asociado, con su distribución no estándar y sus limitaciones. Y la jerarquía media incondicional → random walk → AR → ARMA → regresión con exógenas da la escalera contra la cual un modelo complejo debe justificarse.

**6. Provee el protocolo de diagnóstico de residuos, que es lo más directamente transferible a ML.** Nada en la ACF ni en Ljung–Box exige que la predicción provenga de un modelo lineal. Examinar $\hat a_t$, $|\hat a_t|$ y $\hat a_t^2$ por separado responde tres preguntas distintas sobre qué capturó y qué no capturó un modelo, sea cual sea su arquitectura. Con dos limitaciones registradas: el ajuste de grados de libertad no está definido para modelos de ML, y pasar el test no establece independencia.

**7. Cuantifica cuánto se exagera la evidencia estadística cuando se ignora la dependencia.** Un factor de 2.7 en un ejemplo publicado, con ambos supuestos violados simultáneamente. Esto afecta todo lo que el proyecto vaya a medir: significancia de features, comparación entre modelos, t-stat de un Sharpe, análisis por deciles de señal. Y viene con la advertencia de su propio alcance: HAC corrige la covarianza suponiendo que el estimador puntual es consistente. No es un sustituto de un diseño de validación correcto.

**8. Muestra el error de la regresión en niveles con un ejemplo que no se olvida.** $R^2=96.5\%$, "seriamente inadecuado". Es el molde exacto de lo que ocurriría al regresar precios de dos futuros relacionados, y el capítulo entrega tanto el diagnóstico —la forma de la ACF residual— como el remedio —diferenciar— y el marco correcto al que remitir —cointegración, Cap. 8.

**9. Entrega el vocabulario del Capítulo 3.** La afirmación de que GARCH es un ARMA sobre $a_t^2$ significa que persistencia, raíces características, decaimiento de ACF, estacionariedad y reversión a la media se reutilizarán aplicados a la volatilidad. Aprender ARMA no es aprender un modelo de trading; es aprender el idioma en que estará escrito el capítulo donde efectivamente hay estructura.

**Lo que el capítulo no aporta, y conviene no fingir que aporta.** No dice nada sobre validación out-of-sample, sobreajuste por búsqueda múltiple, costos de transacción, look-ahead bias, ni sobre cómo traducir un pronóstico en una decisión. Su ciclo de modelado —identificar, estimar, verificar sobre el mismo dataset— es metodológicamente circular y consume grados de libertad que ningún test contabiliza. Esas garantías hay que traerlas de otra parte.

**Aporte neto.** El Capítulo 2 no entrega un modelo predictivo para retornos; el propio Tsay dice que la probabilidad de usar ARMA en retornos es baja. Entrega tres cosas: **un conjunto de diagnósticos aplicables a cualquier modelo**, **una jerarquía de benchmarks honestos**, y **una lista precisa de lo que no puede concluirse a partir de esos diagnósticos**. La tercera es probablemente la más valiosa, porque los errores que enumera —ACF plana leída como impredecibilidad, no rechazo leído como demostración, significancia leída como rentabilidad, PACF leída como lookback— son exactamente los que producen sistemas que funcionan en backtest y fracasan en producción.

---

# 12. Registro de revisión crítica

Auditoría de las afirmaciones **[B]** más sensibles, con atención a los términos "debe", "siempre", "demuestra", "óptimo", "necesariamente", "implica" e "invalida".

| Afirmación [B] sensible | Riesgo de sobreinterpretación | Estado final |
|---|---|---|
| "El $R^2$ de 2.46% de Tsay es el ancla de realismo; un resultado mucho mayor debe auditarse" | Generalizar un ejemplo (índice accionario mensual, modelo lineal) a futuros y a alta frecuencia | **MATIZAR** — ya limitado en el texto a: retornos, baja frecuencia, índices accionarios, modelos lineales. No se traslada automáticamente |
| "En futuros líquidos cabe esperar autocorrelación ≈ 0 en retornos diarios" | Convertir una expectativa teórica en hallazgo | **PREGUNTA ABIERTA** — marcada como expectativa a medir (H2.4) |
| "La magnitud tiene mucha más estructura que la dirección" | Trasladar el hallazgo de índices CRSP a futuros sin verificar | **MATIZAR** — [A] verificado en índices accionarios diarios; en futuros es hipótesis (H2.7, H2.8) |
| "Más estructura detectable no es más rentabilidad" | Ninguno; es una acotación restrictiva | **MANTENER** |
| "MSE/pérdida cuadrática define qué se estima" | Riesgo de repetir el error corregido en el Cap. 1 (declarar MSE inválido) | **MANTENER** — formulado como "define qué se estima", sin juicio de validez |
| "Un test de raíz unitaria sobre serie continua ajustada no testea un precio real" | Palabra fuerte, pero es analíticamente correcta: la serie ajustada no es negociable | **MANTENER** — con la aclaración ya presente de qué serie sí testear |
| "Los tests de raíz unitaria tienen potencia baja contra $\phi_1$ cercano a 1" | Atribuir a Tsay algo que no dice | **MANTENER** marcado como [B] — es resultado establecido en econometría, explícitamente señalado como extensión |
| "Tsay se desliza de 'no rechazado' a 'contiene una raíz unitaria'" | Crítica al autor; debe ser verificable | **MANTENER** — ambas frases están citadas textualmente del mismo pasaje |
| "La frase 'si las ACF muestrales están cerca de cero, la serie es white noise' es una inferencia inválida con la propia definición de Tsay" | Crítica fuerte al texto | **MANTENER** — con la lectura caritativa ya incluida: heurística dentro del alcance del modelado lineal de la media |
| "Sobrediferenciar una serie trend-stationary introduce un MA no invertible" | Afirmación técnica no presente en Tsay | **MANTENER** marcado como [B] — se sigue de introducir un factor $(1-B)$ en el polinomio MA |
| "El ciclo identificar→estimar→verificar es data snooping de bajo grado" | Crítica metodológica no formulada por Tsay | **MANTENER** marcado como [B] — no invalida el procedimiento, señala un coste no contabilizado |
| "Apilar $N$ futuros no multiplica por $N$ la información independiente" | Extender §2.10 a un contexto que Tsay no trata | **MANTENER** marcado como [B] — la lógica del tamaño de muestra efectivo es la misma |
| "Con memoria larga, un embargo corto puede ser insuficiente" | Convertir un diagnóstico en regla de validación | **PREGUNTA ABIERTA** — la longitud apropiada es empírica, no derivable del capítulo |
| "La ACF de $\lvert\hat a_t\rvert$ debería examinarse siempre" | Uso de "siempre"; convierte diagnóstico en obligación | **MATIZAR** → reformulado como recomendación de diagnóstico prioritario, no como requisito |
| "HAC no corrige look-ahead bias ni validación defectuosa" | Ninguno; es acotación del alcance, y Tsay condiciona explícitamente | **MANTENER** |
| "El pronóstico de largo horizonte converge al benchmark trivial" | Generalizar de modelos lineales estacionarios a cualquier modelo | **MATIZAR** — ya limitado en el texto a modelos lineales estacionarios; el caso no lineal es **PREGUNTA ABIERTA** (Cap. 4) |
| "Estacionariedad de la variable ≠ estabilidad de la relación predictiva" | Atribuir la distinción a Tsay | **MANTENER** — explícitamente marcado como distinción que Tsay no formula |
| "Las variables de calendario son features de coste cero pero de alto riesgo de snooping" | Ninguno significativo | **MANTENER** |
| "La estacionalidad de futuros de energía es del subyacente, no de los retornos del futuro" | Afirmación sobre pricing no verificada en este capítulo | **MATIZAR** → el argumento (el precio descuenta lo conocido) es estándar pero depende de teoría de pricing; queda como **PREGUNTA ABIERTA** parcial, coherente con la corrección 2 del Cap. 1 |
| "Un modelo puede estar bien calibrado y ser inútil" | Ninguno; ilustrado con la propia Tabla 2.2 | **MANTENER** |

**Patrón detectado en esta revisión.** A diferencia del Capítulo 1, donde el riesgo dominante era convertir observaciones en reglas normativas, aquí el riesgo dominante es **generalizar los ejemplos empíricos de Tsay —todos sobre acciones e índices accionarios estadounidenses a frecuencia mensual o diaria— a futuros y a alta frecuencia**. Cinco de las afirmaciones matizadas caen en ese patrón. Criterio adoptado para el Capítulo 3: cada vez que un hallazgo empírico de Tsay se invoque para futuros, indicar explícitamente el instrumento, la frecuencia y el período sobre los que fue establecido.

