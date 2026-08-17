# TSAY-GROUNDED EMPIRICAL DATA ANALYSIS ROADMAP
## Caracterización empírica de un dataset OHLCV intradiario de 1 minuto — MNQ

**Fase:** DATA UNDERSTANDING / EMPIRICAL CHARACTERIZATION
**Documento hermano:** `Tsay_sintesis_transversal_OHLCV.md` (fundamentación) · `Tsay_empirical_hypotheses_backlog.md` (hipótesis `THxx`)
**Prefijo de etapas:** `TDA-nn` (*Tsay Data Analysis*) — elegido deliberadamente para **no colisionar** con la nomenclatura ya existente en el repositorio (`KB01–KB03`, `FASE 0–7`, o eventuales `Sxx`). Este roadmap **no reutiliza ni hereda** ninguna de esas etapas.

> **DECISIÓN IRIS: NINGUNA** — en todas y cada una de las etapas.
> **ESTE DOCUMENTO NO SE EJECUTA EN ESTA TAREA.** Primero se audita.

---

# 1. Qué responde este roadmap y qué no

## 1.1 Pregunta rectora

> ¿Cuál es el conjunto **mínimo y suficiente** de análisis empíricos necesarios para caracterizar correctamente un dataset OHLCV intradiario de 1 minuto antes de comenzar a diseñar un problema de Machine Learning?

**Respuesta estructural del roadmap:** 19 etapas (`TDA-00`…`TDA-18`), de las cuales **15 son obligatorias**, **1 es mixta** (`TDA-13`: inventario obligatorio, EVT condicional) y **3 son enteramente condicionales** (`TDA-11`, `TDA-15`, `TDA-16`). Ninguna etapa existe porque la técnica aparezca en Tsay; cada una existe porque su ausencia haría **no interpretable** alguna etapa posterior, o porque responde una de las 20 preguntas del *Empirical Profile* final.

## 1.2 Frontera dura

Este roadmap se detiene en el **Nivel 3** de la jerarquía de dependencia (`síntesis §2.3`): estructura estadística **estable y caracterizada**. No cruza al Nivel 4 (predictibilidad fuera de muestra) ni al Nivel 5 (utilidad económica neta). Cualquier etapa que requiriese entrenar un modelo para *predecir* y evaluarlo OOS **no pertenece aquí**.

## 1.3 Supuestos admitidos

Los únicos supuestos básicos son: instrumento MNQ · OHLCV con precio Last · barras de 1 minuto · aproximadamente 2020–2026 · zona horaria declarada America/New_York. **Los cinco son declaraciones no verificadas** y su verificación es literalmente el contenido de `TDA-00` a `TDA-02`. Todo lo demás se analiza o queda abierto. Las decisiones históricas del repositorio (targets, OPC, ventanas, regímenes horarios, features, folds, modelos) se tratan como **antecedentes no vinculantes** y no entran como supuestos.

---

# 2. Gobernanza transversal (G0–G6)

Estas reglas **no son etapas**: aplican simultáneamente a todas. Violarlas invalida el output de la etapa, no sólo su interpretación.

### G0 — Registro de observabilidad
Antes de diseñar cualquier análisis se consulta la tabla de **cantidades no observables** (`síntesis §3.3`). Está prohibido diseñar un análisis cuyo objeto sea no medible con OHLCV Last de 1 minuto. Cuando exista un *proxy*, debe declararse como proxy, con el supuesto que lo conecta con la cantidad de interés.

**Lista corta de lo no observable:** spread bid–ask · signo/agresor de la operación · duración entre transacciones · número de trades · profundidad del libro · secuencia intraminuto · precio eficiente $P^*$ · volatilidad verdadera · estado/régimen latente. **Volumen ≠ número de trades.**

### G1 — Causalidad estricta
Toda cantidad calculada "en $t$" debe pasar el **test de reconstrucción**: recalcularla usando exclusivamente datos hasta $t$ debe producir exactamente el mismo número. Prohibido como insumo de cualquier evaluación causal: *smoothing* ($s_{t\mid T}$), normalización con estadísticos globales, umbrales de extremos derivados de toda la muestra, y perfiles estacionales estimados sobre toda la muestra.
**Excepción etiquetada:** en esta fase muchos análisis son **descriptivos retrospectivos** y usan legítimamente toda la muestra. La regla no es "prohibido usar toda la muestra"; es **"cada output declara si es causal o retrospectivo"**. Un output sin esa etiqueta se considera no entregado.

### G2 — Exploración vs confirmación
Con una muestra muy grande y decenas de variantes posibles (transformaciones × ventanas × segmentaciones × tests), la probabilidad de encontrar algún patrón aparentemente interesante por azar puede aumentar sustancialmente.

| Modo | Reglas |
|---|---|
| **Exploración** | Libre, pero todo hallazgo se marca `EXPLORATORIO`. No puede citarse como propiedad del dataset. |
| **Confirmación** | Requiere: (a) hipótesis escrita **antes** de mirar el resultado, con su ID `THxx`; (b) número de variantes acotado y declarado **por adelantado**; (c) el resultado se sostiene en subperíodos independientes; (d) el resultado se sostiene en el **holdout temporal reservado**. |

**Holdout temporal reservado para caracterización:** el tramo final destinado a esta fase se aparta y **no se mira** durante la exploración. Se utiliza una sola vez para confirmar las propiedades empíricas que se pretendan declarar estables. Su longitud se fija antes de empezar y no se renegocia.

**Importante:** una vez utilizado para confirmar esta caracterización, este período deja de considerarse completamente virgen respecto del proyecto, porque sus resultados pueden influir indirectamente en decisiones metodológicas posteriores. Por tanto:

\[
\boxed{\text{holdout de caracterización} \neq \text{holdout final de evaluación de ML}}
\]

Si posteriormente se diseña y evalúa IRIS, deberá definirse un período de evaluación final que no haya sido utilizado para seleccionar, confirmar o modificar decisiones a partir de esta caracterización, o establecerse un protocolo temporal equivalente que preserve una evaluación realmente no contaminada.

**Calibración de la maquinaria:** todo pipeline de detección debe ejecutarse también sobre (i) la serie de retornos permutada aleatoriamente y (ii) un camino aleatorio simulado con volatilidad y curtosis comparables. Si el pipeline "encuentra" estructura ahí, el pipeline está roto.

### G3 — Estabilidad obligatoria
Ninguna propiedad se reporta sin su versión por subperíodo. Mínimo: por año calendario y por segmento de sesión. **Un número agregado sobre ~6 años, sin su desagregación, no está reportado.**
Corolario: `cambio estadístico ≠ régimen económico identificado`. Se describe el cambio; no se le pone nombre.

### G4 — Parsimonia
Se usa **el análisis más simple que responde correctamente la pregunta**. Una técnica compleja debe declarar por escrito, antes de ejecutarse, **qué información adicional aporta** respecto de la simple. Si no puede declararlo, no se ejecuta.
Tabla operativa en `síntesis §2.10`.

### G5 — Reporte por magnitud, no por p-valor
Prohibido reportar un resultado como "significativo". Todo resultado se reporta como terna:
$$(\text{estimador puntual},\ \text{intervalo por bootstrap por bloques},\ \text{traducción a unidad interpretable})$$
Unidades interpretables admitidas: **ticks**, **múltiplos de la desviación estándar del segmento**, **fracción de variabilidad explicada**, **número de barras afectadas**.
Motivo: con muestras del orden de $n\sim10^6$, muchos tests pueden alcanzar una potencia estadística muy elevada y detectar efectos extremadamente pequeños. Por eso, el p-valor por sí solo no permite juzgar la importancia práctica del resultado (`síntesis §2.9`).

### G6 — Resultado negativo = resultado
"No hay dependencia lineal detectable", "el ajuste estacional explica toda la ACF de $r^2$", "el estado de Kalman coincide con una EMA", "EVT no aporta sobre cuantiles empíricos" son **resultados entregables** que cierran ramas del roadmap. Ninguna etapa se diseña con la expectativa de encontrar señal.

---

# 3. Niveles metodológicos

Derivados por **dependencia informacional**, no por orden del libro.

| Nivel | Nombre | Qué garantiza | Sin él… |
|---|---|---|---|
| **N0** | Admisibilidad del dato | Que las filas representen lo que creemos | ningún estadístico posterior significa nada |
| **N1** | Definición del objeto | Que la variable analizada esté bien construida y su resolución sea conocida | se miden artefactos de construcción |
| **N2** | Estructura determinista | Que el componente de reloj esté separado | el reloj se confunde con dinámica |
| **N3** | Dependencia | Qué hay en media y qué hay en magnitud | no se sabe qué se está caracterizando |
| **N4** | Forma de la distribución condicional | Escala vs forma; cuantiles; colas | se confunde heterocedasticidad con colas |
| **N5** | Estabilidad y estructura avanzada | Si las propiedades se sostienen; si hay no linealidad o estados | se generaliza una mezcla de regímenes |
| **N6** | Consolidación | Redundancia y perfil empírico final | se acumulan análisis sin conclusión |

> **Nota de diseño:** N2 (estructura determinista) aparece **antes** de N3 (dependencia). Ésta es la desviación más importante respecto del orden natural del libro y está fundamentada en `síntesis §2.5`: un patrón diurno genera por sí solo ACF en $r^2$, colas gruesas y agrupamiento de extremos.

---

# 4. Mapa de etapas

| ID | Etapa | Nivel | Estatus | Depende de |
|---|---|---|---|---|
| **TDA-00** | Integridad intra-barra e invariantes físicos | N0 | **Obligatoria** | — |
| **TDA-01** | Semántica temporal, de sesión y de contrato | N0 | **Obligatoria** | TDA-00 |
| **TDA-02** | Integridad del eje temporal y del calendario | N0 | **Obligatoria** | TDA-01 |
| **TDA-03** | Rolls y construcción de la serie continua | N0 | **Obligatoria** | TDA-01, TDA-02 |
| **TDA-04** | Construcción y auditoría de las variables de análisis | N1 | **Obligatoria** | TDA-02, TDA-03 |
| **TDA-05** | Resolución efectiva y discreción | N1 | **Obligatoria** | TDA-04 |
| **TDA-06** | Perfil determinista intradía y de calendario | N2 | **Obligatoria** | TDA-04, TDA-05 |
| **TDA-07** | Distribución marginal y por segmento | N2 | **Obligatoria** | TDA-05, TDA-06 |
| **TDA-08** | Dependencia lineal en la media | N3 | **Obligatoria** | TDA-06, TDA-07 |
| **TDA-09** | Dependencia en magnitud (volatility clustering) | N3 | **Obligatoria** | TDA-06, TDA-07 |
| **TDA-10** | Escala vs forma: origen de las colas | N4 | **Obligatoria** | TDA-09 |
| **TDA-11** | Modelo paramétrico de volatilidad | N4 | *Condicional* | TDA-09, TDA-10 |
| **TDA-12** | Cuantiles condicionales | N4 | **Obligatoria** | TDA-10 |
| **TDA-13** | Extremos: inventario y agrupamiento · EVT | N4 | **Obligatoria** (inventario) / *Condicional* (EVT) | TDA-00, TDA-03, TDA-06, TDA-12 |
| **TDA-14** | Estabilidad temporal y cambio estructural | N5 | **Obligatoria** | TDA-07…TDA-13 |
| **TDA-15** | No linealidad residual | N5 | *Condicional* | TDA-08, TDA-09, TDA-10, TDA-14 |
| **TDA-16** | Estados latentes / state-space | N5 | *Condicional* (default: **no ejecutar**) | TDA-14, TDA-17 |
| **TDA-17** | Redundancia informacional entre transformaciones | N6 | **Obligatoria** | TDA-04…TDA-12 |
| **TDA-18** | Empirical Profile of MNQ OHLCV | N6 | **Obligatoria** | todas |

**Recuento:** 19 etapas. **Obligatorias (15):** `00, 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 12, 14, 17, 18`. **Mixta (1):** `13` — inventario y agrupamiento de extremos obligatorios, EVT condicional. **Condicionales (3):** `11`, `15`, `16`. Contando `13-EVT` como rama propia, hay **4 ramas condicionales**, y las cuatro tienen STOP por defecto **en contra** de ejecutarlas.

> **Sobre "mínimo".** Se evaluó fusionar etapas. Se mantienen separadas `TDA-00/01/02` porque `TDA-02` (completitud del calendario) **requiere** conocer la semántica de sesión de `TDA-01`, que a su vez sólo tiene sentido sobre filas admisibles de `TDA-00`. Se mantienen separadas `TDA-08` y `TDA-09` porque su fusión es precisamente el error que hace leer magnitud como dirección (`síntesis §2.2`). Todo lo demás que podía fusionarse, se fusionó.

---

# 5. Tabla de dependencias — con el razonamiento

| Análisis | Debe ejecutarse después de | ¿Por qué? |
|---|---|---|
| Cualquier estadístico | TDA-00 | Un $H<L$ o un precio $\le 0$ produce retornos indefinidos o extremos falsos que dominan momentos y colas. |
| Detección de barras faltantes | TDA-01 | "Falta una barra" sólo es definible contra una grilla esperada, que depende del calendario de sesión y de la convención del timestamp. |
| Definición de retornos | TDA-01, TDA-03 | Si el timestamp marca el inicio de la barra, "el retorno en $t$" usa información posterior. Si hay roll, la diferencia entre contratos no es un retorno de mercado. |
| Definición de retornos | TDA-02 | Un retorno calculado a través de un hueco de 14 horas no es comparable con uno de 1 minuto. |
| Interpretación de momentos y densidades | TDA-05 | Si $\text{tick}/\hat\sigma$ es grande, la variable es efectivamente discreta y curtosis, densidad y QQ cambian de interpretación. |
| ACF de $r_t$ | TDA-06 | Un ciclo determinista de reloj produce picos de ACF en múltiplos de la jornada que no son dependencia estocástica. |
| ACF de $\lvert r_t\rvert$ y $r_t^2$ | TDA-06 | **Crítico.** La estacionalidad de volatilidad simula ARCH de forma casi perfecta. Sin la comparación crudo/ajustado el resultado no es interpretable. |
| Interpretación del $\rho_1$ | TDA-05, TDA-06 | Su magnitud debe compararse con el tick, y su variación por segmento es la evidencia sobre no sincronía. |
| Cualquier test que asuma iid (JB, Ljung–Box estándar, BDS, EVT clásica) | TDA-09 | Si hay clustering de volatilidad, el supuesto iid está roto y la calibración del test es incorrecta. Se requiere bootstrap por bloques o ajuste. |
| Distribución condicional y cuantiles | TDA-10 | Antes de preguntar por la forma hay que saber cuánto del fenómeno es escala. |
| Cualquier análisis de colas | TDA-00, TDA-03 | Un extremo puede ser un bad tick o un salto de roll. Sin triage previo, se modela el artefacto. |
| Cualquier análisis de colas | TDA-06 | Un umbral absoluto sobre una serie con patrón diurno selecciona horas, no eventos. |
| Cualquier análisis de colas | TDA-12 | Si los cuantiles empíricos ya responden la pregunta, EVT es innecesaria (G4). |
| Índice extremal / EVT clásica | TDA-14 | Requieren estacionariedad; si la serie no es siquiera aproximadamente estacionaria, el estimador no estima lo que se cree. |
| Tests de no linealidad | TDA-08, TDA-09, TDA-06 | BDS y afines detectan *cualquier* desviación de iid: sin filtrar media, volatilidad y estacionalidad, detectan lo ya conocido y se leen como hallazgo nuevo. |
| Tests de no linealidad | TDA-14 | Una no linealidad que sólo aparece en un subperíodo es cambio estructural, no forma funcional. |
| State-space / Kalman | TDA-17 | Si el estado filtrado es una función casi lineal de transformaciones que ya se tienen, no aporta representación nueva. |
| State-space / Kalman | TDA-14 | Requiere una hipótesis latente concreta; sin evidencia de dinámica que las alternativas simples no capturen, no hay hipótesis. |
| Redundancia (TDA-17) | TDA-04…TDA-12 | Sólo se puede medir redundancia sobre un conjunto de transformaciones ya definido y auditado. |
| Empirical Profile | todas | Es la consolidación, no un análisis. |

---

# 6. Analysis decision tree

```text
[TDA-00] INVARIANTES DE BARRA  (H>=max(O,C), L<=min(O,C), H>=L, P>0, V>=0, múltiplos de tick)
   |
   +-- violaciones NO explicables --> CUANTIFICAR, AISLAR, DOCUMENTAR
   |        |
   |        +-- afectan una fracción no despreciable de la muestra
   |        |        --> STOP-0: no continuar. Escalar al proveedor de datos.
   |        |             Ningún resultado estadístico sería atribuible al mercado.
   |        +-- fracción despreciable y aislable --> máscara documentada --> continuar
   |
   +-- OK
        |
        v
[TDA-01] SEMÁNTICA: ¿timestamp = inicio o cierre de barra? ¿qué campo está disponible cuándo?
   |
   +-- INDETERMINABLE (sin doc del proveedor y sin evidencia forense concluyente)
   |        --> continuar bajo el supuesto MÁS CONSERVADOR (timestamp = inicio;
   |            la barra t sólo está completa al final de t) y MARCAR como
   |            PREGUNTA ABIERTA BLOQUEANTE para la fase de ML.
   +-- DETERMINADA --> registrar convención --> continuar
        |
        v
[TDA-02] EJE TEMPORAL: cobertura, huecos, DST, feriados, días incompletos,
         y clasificación de barras inactivas en {ausente | volumen 0 | forward-fill}
   |
   +-- ¿hay forward-fill? --> SÍ --> las barras rellenadas producen r=0 artificial.
   |                                  Toda propiedad se reporta CON y SIN ellas. Obligatorio.
   +-- NO --> continuar
        |
        v
[TDA-03] ROLLS: ¿la serie es continua? ¿se detectan saltos de roll?
   |
   +-- SÍ, y el método de ajuste es DESCONOCIDO
   |        --> test de invariancia a escala sobre cada estadístico previsto.
   |            Estadísticos invariantes a escala --> utilizables.
   |            Estadísticos dependientes del nivel --> NO utilizables. Documentar.
   +-- SÍ, método conocido --> máscara de barras de roll --> continuar
   +-- NO hay rolls detectables --> verificar que sea plausible --> continuar
        |
        v
[TDA-04] VARIABLES: r_t, |r|, r², ln(H/L), ln(C/O), ln(O/C_{t-1}), V, estimadores de rango
         + REGLAS DE NO-CRUCE (día / sesión / roll) declaradas y auditadas
        |
        v
[TDA-05] RESOLUCIÓN: tick/sigma y fracción de r=0, POR SEGMENTO HORARIO
   |
   +-- tick/sigma GRANDE en algún segmento
   |        --> ese segmento es efectivamente DISCRETO.
   |            En él: momentos, densidades y QQ NO se interpretan como continuos.
   |            Puede corresponder ANALIZARLO POR SEPARADO o excluirlo de la
   |            caracterización distribucional (decisión documentada, no automática).
   +-- pequeño en todos --> continuar
        |
        v
[TDA-06] PERFIL DE RELOJ: patrón por minuto-del-día y por día-de-semana
         sobre V, |r|, ln(H/L), frecuencia de extremos, fracción de r=0
   |
   +-- ¿el perfil es fuerte y estable entre años?
   |        |
   |        +-- SÍ --> (a) derivar segmentación de sesión DE LOS DATOS
   |        |          (b) construir factor de escala estacional s(m)
   |        |          (c) TODO análisis posterior se reporta CRUDO y AJUSTADO
   |        |
   |        +-- NO / débil --> documentar el resultado negativo (G6);
   |                            análisis posteriores sólo en versión cruda;
   |                            NO construir s(m) (G4)
        |
        v
[TDA-07] DISTRIBUCIÓN MARGINAL: momentos, cuantiles, QQ, curtosis con/sin recorte
         — global, por segmento, por año.  NO reportar tests de normalidad como hallazgo.
        |
        v
    +---+------------------------------+
    |                                  |
    v                                  v
[TDA-08] MEDIA                    [TDA-09] MAGNITUD
ACF(r), PACF, Q(m), por           ACF(|r|), ACF(r²), ACF(ln(H/L)),
segmento y subperíodo             CRUDO vs AJUSTADO por s(m)
    |                                  |
    +-- |rho_1| implica un              +-- la ACF COLAPSA al ajustar
    |   movimiento << tick              |     --> el hallazgo ERA EL RELOJ.
    |     --> STOP-8a: irrelevante      |         Registrar (G6). No modelar volatilidad.
    |         operativamente.           |         --> saltar TDA-11.
    |         Documentar como           |
    |         propiedad, no señal.      +-- la ACF SOBREVIVE
    |                                   |     --> hay dependencia en magnitud.
    +-- rho_1 material                  |         Medir persistencia y su decaimiento.
        --> INTENTO OBLIGATORIO DE      |
            REFUTACIÓN MICROESTRUCTURAL:|
            (i) ¿se atenúa al agregar   v
                1->5->10 min?      [TDA-10] ESCALA vs FORMA
            (ii) ¿es peor en tramos     ¿qué fracción del exceso de curtosis
                 de baja actividad?     sobrevive a estandarizar por sigma causal?
            (iii) ¿su magnitud es           |
                  del orden del tick?       +-- DESAPARECE CASI TODA
            |                               |     --> el fenómeno es HETEROCEDASTICIDAD.
            +-- compatible con                    |     La distribución condicional es
            |   microestructura            |         aproximadamente estable en forma.
            |     --> STOP-8b: declarar    |         --> STOP-13: EVT NO se ejecuta.
            |         NOT SEPARABLE con    |         --> TDA-12 en versión mínima.
            |         OHLCV Last.          |
            +-- NO compatible              +-- SOBREVIVE UNA FRACCIÓN GRANDE
                  --> registrar como             --> hay estructura de FORMA.
                      dependencia lineal              --> TDA-12 completo
                      caracterizada.                  --> puerta a TDA-13-EVT abierta
                      (No pasar al Nivel 4.)                (aún condicionada)
                                            |
                                            v
                                   [TDA-11] ¿MODELO PARAMÉTRICO DE VOLATILIDAD?
                                       |
                                       +-- ¿hay una pregunta que los diagnósticos
                                       |   simples (ACF, EWMA causal, rango, ventanas
                                       |   rodantes) NO respondan?
                                       |     |
                                       |     +-- NO --> STOP-11: no ajustar GARCH. (G4)
                                       |     +-- SÍ --> ajustar el modelo MÁS SIMPLE que
                                       |               la responda; benchmark obligatorio
                                       |               contra EWMA y contra volatilidad de
                                       |               rango; diagnosticar residuos
                                       |               estandarizados; reportar estabilidad.
                                       v
                                   [TDA-12] CUANTILES CONDICIONALES
                                   ¿la distribución cambia sólo en ESCALA o también en FORMA?
                                       |
                                       v
                                   [TDA-13] EXTREMOS
                                   OBLIGATORIO: inventario + triage (error/roll/real)
                                                + agrupamiento (runs vs Poisson)
                                       |
                                       +-- ¿los cuantiles empíricos de TDA-12 responden
                                       |   ya la pregunta de caracterización?
                                       |     +-- SÍ --> STOP-13: EVT NO se ejecuta. (G4)
                                       |     +-- NO (se necesita extrapolar más allá del
                                       |            peor evento observado, con una razón
                                       |            escrita) --> EVT condicional:
                                       |            mean excess plot, sensibilidad al umbral,
                                       |            GPD con diagnósticos, dentro de segmentos
                                       |            horarios homogéneos.
                                       |            Si xi cambia >~50% entre umbrales
                                       |            razonables --> STOP-13b: reportar el
                                       |            rango, no un valor. No usar como parámetro.
                                       v
                                   [TDA-14] ESTABILIDAD TEMPORAL Y CAMBIO ESTRUCTURAL
                                   (consolidación de las versiones por subperíodo de todo lo anterior)
                                       |
                                       +-- ¿alguna propiedad central es INESTABLE?
                                       |     --> ninguna propiedad agregada de la muestra
                                       |         completa puede citarse como "la propiedad
                                       |         del dataset". Reportar por régimen temporal.
                                       |         --> los análisis de N5 se ejecutan POR SUBPERÍODO
                                       |             o no se ejecutan.
                                       v
                                   [TDA-15] ¿NO LINEALIDAD?
                                       |
                                       +-- ¿existe residuo estandarizado limpio de
                                       |   media + volatilidad + estacionalidad?
                                       |     +-- NO --> STOP-15a: no testear. El test
                                       |     |          detectaría lo ya conocido.
                                       |     +-- SÍ --> BDS / F-test sobre ese residuo,
                                       |               con corrección por multiplicidad y
                                       |               calibración por permutación (G2)
                                       |         |
                                       |         +-- no significativo, o significativo pero
                                       |         |   inestable entre subperíodos
                                       |         |     --> STOP-15b: registrar resultado
                                       |         |         negativo. NO modelar TAR/STAR/MS.
                                       |         +-- estable en subperíodos independientes
                                       |               --> registrar como propiedad
                                       |                   caracterizada. NO modelar aquí:
                                       |                   identificar la forma pertenece
                                       |                   a otra fase.
                                       v
                                   [TDA-16] ¿ESTADO LATENTE?
                                       |
                                       +-- ¿existe una hipótesis latente CONCRETA y FALSABLE,
                                       |   escrita antes, que una EMA causal NO responda?
                                       |     +-- NO (default) --> STOP-16: no ejecutar.
                                       |     +-- SÍ --> comparación obligatoria contra
                                       |               EMA/rolling/rango. Si la diferencia
                                       |               es marginal --> STOP-16b (G4, G6).
                                       v
                                   [TDA-17] REDUNDANCIA INFORMACIONAL
                                   dimensión efectiva del conjunto de transformaciones
                                       v
                                   [TDA-18] EMPIRICAL PROFILE OF MNQ OHLCV
                                   (responde las 20 preguntas; declara hipótesis
                                    respondidas, rechazadas y abiertas)
                                       v
                                   >>> FIN DE LA FASE.
                                       Apertura de una fase SEPARADA de ML PROBLEM DESIGN.
```

---

# 7. Etapas

---

## TDA-00 — Integridad intra-barra e invariantes físicos

**Objetivo.** Establecer que cada fila es una barra admisible, sin usar ninguna información sobre calendario, sesión o contrato.

**Preguntas.** ¿Se cumplen los invariantes de una barra OHLC? ¿Hay precios no positivos, volúmenes negativos, valores nulos o no numéricos? ¿Todos los precios caen en la grilla del tick? ¿Hay timestamps duplicados o no monótonos? ¿Cuántas filas hay realmente y qué columnas existen además de OHLCV?

**Fundamento Tsay.** [A] C5: $H$ y $L$ son estadísticos de orden del conjunto de precios operados; por construcción $L\le O,C\le H$. [A] C1/C5: los precios cotizan en múltiplos del tick, la discreción es una propiedad del dato, no un defecto. [A] C7: un extremo puede ser un dato erróneo y descartarlo automáticamente sesga la cola — por eso el triage empieza aquí y es documentado.

**Inputs.** El archivo crudo, sin transformar.

**Variables analizadas.** `timestamp`, `O`, `H`, `L`, `C`, `V` y cualquier columna adicional presente.

**Métodos mínimos.**
1. Inventario de columnas, tipos, rango de fechas real, conteo de filas.
2. Invariantes duros: $H\ge L$; $H\ge\max(O,C)$; $L\le\min(O,C)$; $O,H,L,C>0$; $V\ge0$; ausencia de NaN/inf.
3. Grilla de tick: verificar que $(P - P_{\min})/\text{tick}$ es entero para los cuatro precios; contar y localizar violaciones.
4. Timestamps: unicidad, monotonicidad estricta, resolución (¿segundos, milisegundos?), formato y presencia/ausencia de offset de zona horaria.
5. Localización temporal de cada violación (¿se concentran en fechas o en horarios concretos?).

**Métodos opcionales.** Comparación de un tramo contra una segunda fuente, si existiera.

**Outputs obligatorios.**
- `TDA00_inventario.md`: columnas, tipos, rango, conteo.
- `TDA00_violaciones.csv`: una fila por violación, con timestamp, tipo, valores.
- **Máscara `bad_data`** persistida — no se borran filas, se marcan.
- Declaración explícita del tick size usado y su fuente.

**Criterios de interpretación.** Una violación de invariante **no es** un evento de mercado: es un defecto de construcción del dato. Su concentración temporal es informativa (un tramo con muchas violaciones sugiere un cambio de proveedor o de método de construcción). La ausencia total de violaciones es también informativa: puede indicar que el proveedor ya aplicó una limpieza no documentada.

**Dependencias.** Ninguna. Es la raíz.

**Riesgos.** Borrar en vez de marcar (se pierde la trazabilidad y se sesga la cola). Asumir un tick size incorrecto. Confundir precisión de punto flotante con violación de grilla.

**Condiciones para avanzar.** Violaciones cuantificadas, localizadas y aisladas en una máscara; fracción afectada documentada.

**Condiciones para detenerse — `STOP-0`.** Si la fracción de filas con violaciones no explicables no es despreciable, o si las violaciones se concentran en un tramo extenso, **se detiene el roadmap** y se escala al proveedor. Ningún resultado estadístico posterior sería atribuible al mercado.

**Hipótesis del backlog resueltas.** `TH01`, y parcialmente `TH06`.

**Decisiones IRIS.** `NINGUNA`

---

## TDA-01 — Semántica temporal, de sesión y de contrato

**Objetivo.** Establecer **qué significa** cada timestamp y cada campo, y en qué instante estuvo disponible cada uno. Es la etapa que determina la causalidad de todo lo posterior.

**Preguntas.** ¿El timestamp marca el **inicio** o el **cierre** de la barra? ¿Cuándo está disponible el Close? ¿Y el High y el Low — sólo al terminar la barra? ¿Y el Volume? ¿Qué zona horaria está aplicada y hay offset explícito? ¿Cómo se representa el cambio de sesión? ¿Cómo se representa el roll? ¿Cómo se construye una barra sin operaciones? ¿Existen barras rellenadas hacia adelante?

**Fundamento Tsay.** [A] C11: la distinción *filtering / prediction / smoothing* depende enteramente de qué información está en $\mathcal F_t$; "disponible en $t$" es una definición, no una obviedad. [A] C1: la factorización condicional exige que todo insumo esté disponible en $t$. [A] C5: la agregación en barras destruye el orden intraminuto — el Close es la última transacción, no un resumen.

**Inputs.** TDA-00 (filas admisibles) + **documentación del proveedor**.

**Variables analizadas.** `timestamp` y la estructura de la barra.

**Métodos mínimos.**
1. **Documental (prioritario).** Obtener y citar la especificación del proveedor. Esta etapa **no es puramente estadística**.
2. **Forense, si la documentación falta o es ambigua:**
   - Frontera de sesión: si el timestamp marcase el cierre, la primera barra de una sesión que abre a las 09:30 aparecería como 09:31, no 09:30. Verificar contra el horario oficial del contrato.
   - Última barra de la jornada: simétricamente, su etiqueta revela la convención.
   - Barras inactivas: buscar filas con $O=H=L=C$ y $V=0$ (firma de forward-fill) frente a ausencia de fila (firma de "no hubo operaciones").
   - Cambios de DST: verificar si la jornada cambia de longitud en el archivo o si el offset absorbe el cambio.
3. Registrar la convención adoptada en un documento de una página que se cita en todos los outputs posteriores.

**Métodos opcionales.** Comparación de un puñado de barras contra un registro de referencia externo.

**Outputs obligatorios.**
- `TDA01_convencion_temporal.md`: convención del timestamp, disponibilidad de cada campo, zona horaria, tratamiento de DST, definición de sesión, representación del roll y de barras inactivas. **Cada afirmación con su evidencia: documental o forense.**
- Etiqueta de confianza por afirmación: `DOCUMENTADO` / `INFERIDO` / `INDETERMINADO`.

**Criterios de interpretación.** Si el timestamp marca el **inicio** de la barra, entonces la barra etiquetada $t$ no está completa hasta $t+59\text{s}$, y usar su Close "en $t$" es fuga de información de hasta un minuto. Esta única distinción puede invalidar un pipeline entero sin dejar rastro visible.

**Dependencias.** TDA-00.

**Riesgos.** Asumir la convención por costumbre. Confundir "el proveedor lo llama así" con "así se construyó". Suponer que la zona horaria declarada coincide con la aplicada en el archivo.

**Condiciones para avanzar.** Convención registrada con etiqueta de confianza.

**Condiciones para detenerse.** No hay STOP duro: si la convención resulta `INDETERMINADO`, se **continúa bajo el supuesto más conservador** (timestamp = inicio de barra; la barra $t$ sólo está disponible al final de $t$) y se marca como **PREGUNTA ABIERTA BLOQUEANTE** para la fase de ML. Un roadmap de caracterización puede continuar con esa incertidumbre; un diseño de ML no.

**Hipótesis del backlog resueltas.** `TH02`, `TH04` (parcialmente).

**Decisiones IRIS.** `NINGUNA`

---

## TDA-02 — Integridad del eje temporal y del calendario

**Objetivo.** Determinar si el eje temporal está completo respecto de la grilla esperada, y clasificar cada ausencia.

**Preguntas.** ¿Qué fracción de la grilla esperada está presente? ¿Dónde están los huecos y de qué tamaño? ¿Coinciden con feriados, medias sesiones, mantenimientos diarios de CME, fines de semana? ¿Hay días incompletos? ¿Hay barras fuera del horario esperado? ¿Cómo se comportan los cambios de DST? ¿Las barras inactivas son ausencias, volúmenes cero o forward-fill?

**Fundamento Tsay.** [A] C11: los valores faltantes tienen tratamiento propio ($v_t=0$, $K_t=0$; la incertidumbre crece) y **no equivalen** a valores imputados. [B] C11: barra faltante ≠ barra de volumen cero ≠ valor imputado — los tres se ven idénticos en un DataFrame y significan cosas distintas.

**Inputs.** TDA-00, TDA-01 (definición de sesión y calendario esperado).

**Variables analizadas.** Presencia/ausencia de barras; $V_t$; firma $O=H=L=C$.

**Métodos mínimos.**
1. Construir la grilla esperada de minutos a partir del calendario de sesión declarado en TDA-01.
2. Cobertura: fracción presente, global y por año/mes/día.
3. Inventario de huecos: inicio, fin, longitud, y **causa candidata** (fin de semana / feriado / media sesión / mantenimiento / desconocida).
4. Clasificación tripartita de barras inactivas: `AUSENTE` / `VOLUMEN_CERO` / `FORWARD_FILL` (firma $O=H=L=C$).
5. Días incompletos: conteo de barras por jornada frente a lo esperado; distribución de ese conteo.
6. Barras fuera del horario esperado.
7. Verificación de los saltos de DST en cada año del rango.

**Métodos opcionales.** Contraste con un calendario oficial de feriados de CME (fuente externa, a citar).

**Outputs obligatorios.**
- `TDA02_cobertura.md` + tabla de cobertura por año/mes.
- `TDA02_huecos.csv` con causa candidata por hueco.
- **Máscara `barra_inactiva`** con su categoría tripartita.
- **Heatmap de completitud** (día × minuto-del-día) — es el único gráfico de esta etapa y responde una pregunta concreta: ¿los huecos son estructurales (bandas verticales = horario) o esporádicos (puntos)?

**Criterios de interpretación.** Huecos estructurales alineados con el horario son **información** (define la sesión real, que puede diferir de la declarada). Huecos esporádicos son **defecto**. La presencia de forward-fill es el hallazgo más consecuente: genera $r_t=0$ artificiales, deprime la volatilidad estimada, infla la fracción de barras sin cambio y puede generar autocorrelación espuria.

**Dependencias.** TDA-01 (sin definición de sesión no existe "grilla esperada").

**Riesgos.** Confundir el mantenimiento diario del mercado con un defecto del dato. Tratar todos los huecos como homogéneos. Imputar huecos (**prohibido en esta fase**: la imputación crea datos).

**Condiciones para avanzar.** Cobertura cuantificada, huecos clasificados, máscara tripartita construida.

**Condiciones para detenerse — `STOP-2`.** Si la sesión real derivada de los datos difiere sustancialmente de la declarada, se detiene y se revisa TDA-01 antes de continuar. No se "arregla" el calendario: se corrige la definición.

**Hipótesis del backlog resueltas.** `TH03`, `TH04`.

**Decisiones IRIS.** `NINGUNA`

---

## TDA-03 — Rolls y construcción de la serie continua

**Objetivo.** Determinar si la serie es un contrato continuo, localizar los rolls, medir su efecto y establecer qué estadísticos son utilizables sobre ella.

**Preguntas.** ¿Es una serie continua o un único contrato? ¿Se detectan discontinuidades compatibles con rolls? ¿Con qué periodicidad? ¿Qué magnitud tienen los saltos? ¿Qué método de ajuste se aplicó (ninguno / ratio / diferencia)? ¿Qué estadísticos previstos son invariantes a escala y cuáles dependen del nivel?

**Fundamento Tsay.** [A] C1: **ratio** preserva exactamente los retornos (equivalente a intra-contrato encadenado); **aditivo** preserva exactamente las diferencias en puntos; **ninguno preserva ambos**. [A] C1: el factor de ajuste de un segmento depende de rolls posteriores, pero es **constante dentro del segmento** ⇒ los estadísticos **invariantes a escala** no sufren fuga; los **dependientes del nivel**, sí. [A] C1: un salto de roll **no es un retorno de mercado**. [A] C1: definir la regla de roll *ex post* sí es no causal, independientemente del método de ajuste.

**Inputs.** TDA-00, TDA-01, TDA-02.

**Variables analizadas.** Serie de $C_t$; $V_t$ (los rolls suelen dejar firma en volumen); diferencias entre barras consecutivas en las fronteras de sesión.

**Métodos mínimos.**
1. **Detección de rolls candidatos:** barras cuya diferencia $|C_t-C_{t-1}|$ es un valor atípico extremo *y* que ocurren en fechas compatibles con el ciclo trimestral de vencimientos (marzo, junio, septiembre, diciembre) del Nasdaq-100.
2. **Firma de volumen:** cambio de nivel del volumen alrededor de las fechas candidatas.
3. **Caracterización del salto:** magnitud en puntos y en porcentaje, por evento.
4. **Test de invariancia a escala** (test operativo de C1): para cada estadístico previsto en el roadmap, responder por escrito: *¿cambiaría su valor si el factor de ajuste del segmento cambiase?* Si no cambia → `INVARIANTE`, utilizable. Si cambia → `DEPENDIENTE DEL NIVEL`, no utilizable sobre serie ajustada.
5. **Máscara `roll`** que marca las barras contaminadas (al menos la barra del salto).

**Métodos opcionales.** Comparación entre la serie disponible y una serie reconstruida intra-contrato, si se dispusiera de los contratos individuales.

**Outputs obligatorios.**
- `TDA03_rolls.csv`: fecha, magnitud en puntos, magnitud en %, evidencia (volumen, calendario).
- **Máscara `roll`** persistida.
- **Tabla de clasificación** de cada estadístico previsto en `INVARIANTE` / `DEPENDIENTE DEL NIVEL`.
- Declaración del método de ajuste: `DOCUMENTADO` / `INFERIDO` / `INDETERMINADO`.

**Criterios de interpretación.** Si el método es indeterminado, la conclusión operativa **no es "no se puede hacer nada"**: es que sólo son utilizables los estadísticos invariantes a escala. Prácticamente todo lo que este roadmap necesita (retornos, cocientes, ACF de retornos, momentos de retornos, cuantiles de retornos) es invariante. Lo que no lo es (umbrales de precio absolutos, distancias en puntos comparadas a través de la frontera de un roll) queda excluido.

**Dependencias.** TDA-01 (fechas y sesiones), TDA-02 (huecos, para no confundir un hueco con un salto).

**Riesgos.** Confundir un evento de mercado real con un roll. Confundir un roll con un dato erróneo. Aplicar la máscara de roll y olvidar que su ausencia sesga la cola en la otra dirección (los rolls reales de un trader existen).

**Condiciones para avanzar.** Rolls localizados o descartados; máscara construida; tabla de invariancia completada.

**Condiciones para detenerse — `STOP-3`.** Si se detectan discontinuidades grandes que **no** son atribuibles ni a rolls ni a eventos de mercado conocidos, se vuelve a TDA-00: son datos, no mercado.

**Hipótesis del backlog resueltas.** `TH05`.

**Decisiones IRIS.** `NINGUNA`

---

## TDA-04 — Construcción y auditoría de las variables de análisis

**Objetivo.** Definir el **conjunto mínimo** de variables sobre las que se hará toda la caracterización, con su fundamento, sus riesgos y sus reglas de frontera. No es feature engineering: es la definición del objeto de estudio.

**Preguntas.** ¿Cómo se define el retorno? ¿Debe cruzar días, sesiones, huecos o rolls? ¿Cuánto difieren $r_t$ y $R_t$ y dónde? ¿Escala la varianza con el horizonte? ¿Qué medida de rango usar y con qué supuestos?

**Fundamento Tsay.** [A] C1: log-retorno aditivo en el tiempo, retorno simple aditivo en cartera; ninguno ambos. [A] C1: la aproximación de primer orden $r\approx R$ "puede no ser suficiente en algunas aplicaciones". [A] C1: bajo independencia $\text{Var}(r_t[h])\approx h\,\text{Var}(r_t)$; la desviación señala autocorrelación o memoria. [A] C3: $r_t^2$ es insesgado pero **muy impreciso** como proxy de $\sigma_t^2$; los estimadores de rango alcanzan eficiencias relativas de ~2 a ~8 bajo el modelo de difusión sin drift; Parkinson $\hat\sigma^2\approx0.3607(H-L)^2$; Rogers–Satchell tolera drift pero no gaps; Yang–Zhang combina overnight, sesión y rango con $k=0.34/[1.34+(n+1)/(n-1)]$. [A] C5: el rango observado **subestima** el rango verdadero, con sesgo dependiente de la frecuencia de negociación y del tick.

**Inputs.** TDA-02 (máscaras), TDA-03 (máscara de roll y tabla de invariancia).

**Conjunto mínimo de variables — y su justificación**

| Variable | Definición | Interpretación | Riesgos | ¿Ajuste de sesión? | ¿Microestructura? | ¿Cruza fronteras? |
|---|---|---|---|---|---|---|
| $r_t$ | $\ln(C_t/C_{t-1})$ | Retorno logarítmico | Cruce de fronteras; discreción | Se decide en TDA-06 | Sí (Close = última transacción) | **No** cruza día/sesión/roll/hueco |
| $R_t$ | $C_t/C_{t-1}-1$ | Retorno simple | Sólo para verificar la equivalencia con $r_t$ | — | Sí | igual que $r_t$ |
| $\lvert r_t\rvert$ | — | Proxy de magnitud | Menos ruidoso que $r_t^2$ | Sí | Sí | igual que $r_t$ |
| $r_t^2$ | — | Proxy de varianza | Insesgado pero muy impreciso [C3] | Sí | Sí | igual que $r_t$ |
| $\text{rg}_t$ | $\ln(H_t/L_t)$ | Rango intrabarra | Sesgo negativo por discreción y baja actividad [C3, C5]; **nulo si no hubo operaciones** | Sí | Sí | **No cruza nada por construcción** |
| $\text{oc}_t$ | $\ln(C_t/O_t)$ | Retorno intrabarra | No cruza fronteras; ignora el gap | Sí | Sí | No |
| $\text{gap}_t$ | $\ln(O_t/C_{t-1})$ | Salto entre barras | Aísla exactamente lo que $\text{oc}_t$ omite; contiene los rolls | Sí | Sí | **Sí, por diseño** — se analiza aparte |
| $V_t$ | — | Contratos operados | **No es número de trades** [C5] | Sí | — | No |
| $\text{zero}_t$ | $\mathbf 1\{r_t=0\}$ | Indicador de no-movimiento | Contaminado por forward-fill | Sí | Sí | igual que $r_t$ |

**Variables deliberadamente NO incluidas en el mínimo, y por qué.** Parkinson / Garman–Klass / Rogers–Satchell / Yang–Zhang: sus supuestos (difusión sin drift, tratamiento específico del gap, volatilidad constante en la ventana) son **más fuertes** que los de $\ln(H/L)$ crudo, y en esta fase la pregunta es descriptiva. Entran como **método opcional en TDA-09/TDA-11** si aparece una pregunta que $\ln(H/L)$ no responda (G4). Indicadores técnicos (EMA, MACD, RSI, ATR, Bollinger): no entran como variables de análisis; entran únicamente como objetos del estudio de redundancia en `TDA-17`.

**Métodos mínimos.**
1. Construcción de las variables con las **reglas de no-cruce** aplicadas: $r_t$ es `NaN` cuando la barra $t-1$ no existe, pertenece a otra sesión, está separada por un hueco, o está marcada por la máscara de roll o `bad_data`. Ese `NaN` es un resultado, no un problema a rellenar.
2. Auditoría de la regla de no-cruce: cuántas observaciones se pierden por cada causa. Si la fracción es grande, la definición debe revisarse.
3. Verificación $r_t$ vs $R_t$: distribución de $|r_t-R_t|$, global y por decil de magnitud.
4. Escalado de varianza: $\text{Var}(r_t[h])$ frente a $h$ para $h\in\{1,5,15,60,\ldots\}$, en escala log-log. La pendiente es el resultado.
5. Consistencia interna: $\text{rg}_t\ge|\text{oc}_t|$ siempre (test de sanidad de las máscaras).

**Métodos opcionales.** Barras alternativas (por volumen o por número de ticks) — **no en esta fase**; se registra como `POSSIBLE FUTURE ML IMPLICATION`.

**Outputs obligatorios.**
- `TDA04_definiciones.md`: tabla anterior con la regla de no-cruce final y su auditoría.
- Serie de variables construida, con máscaras aplicadas y `NaN` explícitos.
- Gráfico único: $\log\text{Var}(r[h])$ vs $\log h$, con la recta de pendiente 1 como referencia.

**Criterios de interpretación.** Pendiente $\approx1$ en el escalado de varianza es consistente con independencia; pendiente $<1$ sugiere reversión de corto plazo (y remite a la sospecha de microestructura de TDA-08); pendiente $>1$ sugiere persistencia. **Es un diagnóstico barato que anticipa TDA-08 y TDA-09 y que ayuda a interpretarlos, no a sustituirlos.**

**Dependencias.** TDA-02, TDA-03.

**Riesgos.** Rellenar los `NaN` de frontera. Usar $r_t$ que cruza el hueco nocturno como si fuera comparable con un retorno de 1 minuto. Interpretar $\ln(H/L)=0$ como volatilidad cero cuando significa "no hubo actividad".

**Condiciones para avanzar.** Variables construidas, auditadas y con pérdida por máscaras cuantificada.

**Condiciones para detenerse — `STOP-4`.** Si la regla de no-cruce elimina una fracción sustancial de la muestra, se revisa la definición de sesión (TDA-01) antes de seguir. No se relaja la regla para "salvar" observaciones.

**Hipótesis del backlog resueltas.** `TH07`, `TH08`, `TH10`.

**Decisiones IRIS.** `NINGUNA`

---

## TDA-05 — Resolución efectiva y discreción

**Objetivo.** Determinar en qué medida $r_t$ es una variable efectivamente discreta, y en qué segmentos.

**Preguntas.** ¿Cuál es el cociente $\text{tick}/\hat\sigma_r$? ¿Qué fracción de barras tiene $r_t=0$? ¿Cómo se distribuyen los movimientos en múltiplos de tick? ¿Cambia todo esto por hora del día y por año?

**Fundamento Tsay.** [A] C1: "para retornos de alta frecuencia, la discreción se vuelve un problema"; el retorno tick-a-tick **no es continuo**. [A] C5: tres episodios documentados de la relación tick ↔ fracción de "sin cambio" (IBM 1990–91 con tick 1/8: 67.06% sin cambio; IBM 1999 con tick 1/16: 45.8%; Boeing 2008 decimal: 58.5%) — **a menor tick, menor proporción de sin cambio**. [A] C5: la discreción sigue restringiendo O/H/L/C **después** de la agregación en barras.

**Inputs.** TDA-04.

**Variables analizadas.** $r_t$, $\text{zero}_t$, tick, $\hat\sigma$ local.

**Métodos mínimos.**
1. Histograma de $\Delta C_t$ expresado en **número de ticks** (análogo directo de la Tabla 5.1 de Tsay).
2. Fracción de barras con $r_t=0$ — **reportada dos veces**: incluyendo y excluyendo las barras marcadas `FORWARD_FILL` en TDA-02.
3. Cociente $\text{tick}/\hat\sigma_r$ y $\text{tick}/\text{mediana}(\text{rg}_t)$.
4. Los tres anteriores desagregados **por hora del día** y **por año**.
5. Número de valores distintos que toma $r_t$ dentro de un rango central (medida directa de cuán "continua" es la variable).

**Métodos opcionales.** Comparación de la resolución efectiva a 1, 5 y 10 minutos (anticipa TDA-08 y TDA-13; si se ejecuta aquí, se reutiliza).

**Outputs obligatorios.**
- Tabla de resolución efectiva por segmento horario y por año.
- Histograma en múltiplos de tick.
- **Declaración explícita del umbral operativo de relevancia**: el movimiento mínimo no nulo, en las unidades que se usarán en G5.

**Criterios de interpretación.** Si $\text{tick}/\hat\sigma$ es grande en un segmento, en ese segmento $r_t$ toma pocos valores distintos y **momentos, densidades, QQ-plots y tests de normalidad cambian de interpretación**: no describen una distribución continua. Además, cualquier "predicción" de un movimiento inferior a un tick es vacía por construcción — esto fija el suelo de relevancia de G5 para todo el roadmap.

**Dependencias.** TDA-04 (variables), TDA-02 (para separar forward-fill).

**Riesgos.** Reportar una única fracción de ceros agregada, ocultando que de noche puede ser dominante y de día marginal. Confundir ceros por forward-fill con ceros por ausencia de movimiento.

**Condiciones para avanzar.** Resolución efectiva cuantificada por segmento; umbral de relevancia declarado.

**Condiciones para detenerse — `STOP-5`.** Si en algún segmento la variable es tan discreta que la caracterización distribucional carece de sentido, ese segmento se analiza por separado o se excluye de las etapas distribucionales — **con la decisión documentada y su efecto sobre el tamaño de muestra**. No se excluye silenciosamente.

**Hipótesis del backlog resueltas.** `TH09`.

**Decisiones IRIS.** `NINGUNA`

---

## TDA-06 — Perfil determinista intradía y de calendario

**Objetivo.** Aislar el componente ligado al reloj antes de estudiar cualquier dinámica estocástica. Es la etapa cuyo lugar en el orden es la decisión metodológica central de este roadmap.

**Preguntas.** ¿Existe un patrón determinista de actividad y volatilidad asociado a la hora? ¿Qué forma tiene (no se asume la U de NYSE)? ¿Es estable entre años? ¿Hay efecto de día de la semana? ¿Está en la **media**, en la **varianza**, o en ninguna? ¿Qué segmentación de sesión emerge **de los datos**?

**Fundamento Tsay.** [A] C5: patrón diurno documentado, con ACF del número de transacciones periódica con la longitud de la jornada y forma en U en NYSE. [A] C3 (error 27): prohibido atribuir a dinámica ARCH lo que es estacionalidad intradía determinista. [A] C2: Ljung–Box debe modificarse ante estacionalidad; mirar los rezagos múltiplos del período. [A] C7: la EVT clásica y el índice extremal requieren estacionariedad. [B] C5: la forma en U de NYSE **no se traslada** a un mercado casi 24h.

**Inputs.** TDA-04, TDA-05.

**Variables analizadas.** $V_t$, $|r_t|$, $r_t^2$, $\text{rg}_t$, $\text{zero}_t$, frecuencia de $|r_t|$ por encima de un umbral relativo, y $r_t$ (para la pregunta de media).

**Métodos mínimos.**
1. **Perfil por minuto del día**: media (o mediana, más robusta) de cada variable, agregada sobre todas las jornadas, con bandas de incertidumbre.
2. **Perfil por día de la semana** para las mismas variables.
3. **Media vs varianza**: comparar el perfil de $r_t$ (media) con el de $|r_t|$ y $\text{rg}_t$ (varianza). La expectativa razonable — a verificar — es perfil fuerte en varianza y nulo en media.
4. **Estabilidad del perfil**: repetir el cálculo por año y superponer. Un perfil que cambia de forma entre años no es determinista.
5. **Segmentación derivada de los datos**: proponer cortes de sesión basados en los quiebres observados del perfil, no en una convención heredada.
6. **Factor de escala estacional $s(m)$** para el minuto del día $m$, construido a partir del perfil de $|r_t|$ o $\text{rg}_t$, y la variable ajustada $\tilde r_t = r_t/s(m_t)$. **Declarar explícitamente que $s(m)$ estimado sobre toda la muestra es RETROSPECTIVO (G1)**; su versión causal (estimada con datos hasta $t$) se construye sólo si alguna etapa posterior la necesita.

**Métodos opcionales.** Descomposición en componentes (tendencia + estacional + irregular) vía state-space — **explícitamente desaconsejada aquí**: C11 advierte sobre no unicidad de la descomposición, y un perfil promedio responde la misma pregunta con supuestos mucho menores (G4).

**Outputs obligatorios.**
- **Gráfico de perfil intradía** superpuesto por año, para $V$, $|r|$, $\text{rg}$ y $\text{zero}$. Es el gráfico más informativo de todo el roadmap.
- Tabla de segmentación propuesta con su criterio.
- Serie ajustada $\tilde r_t$ y el factor $s(m)$, etiquetados como `RETROSPECTIVO`.
- Declaración de si el patrón está en media, en varianza o en ninguna.

**Criterios de interpretación.** Un perfil fuerte y estable en varianza implica que **todas** las etapas posteriores deben reportarse en versión cruda y ajustada. Un perfil en la **media** sería un hallazgo mucho más fuerte y debe someterse a escrutinio adicional: es un candidato natural a artefacto (frontera de sesión, apertura, rolls, huecos) antes que a fenómeno.

**Dependencias.** TDA-04, TDA-05 (la fracción de ceros por hora es parte del perfil).

**Riesgos.** Importar la segmentación de sesión de una decisión histórica del repositorio (prohibido). Asumir la forma en U. Confundir el efecto del mantenimiento diario del mercado con un patrón de comportamiento. Usar $s(m)$ retrospectivo en un contexto que se presente como causal.

**Condiciones para avanzar.** Perfil medido, estabilidad verificada, segmentación propuesta, $s(m)$ construido y etiquetado.

**Condiciones para detenerse — `STOP-6`.** Si el perfil es débil o inestable entre años, **no se construye $s(m)$** y las etapas posteriores se ejecutan sólo en versión cruda. Es un resultado negativo válido (G6) que simplifica todo el resto del roadmap.

**Hipótesis del backlog resueltas.** `TH14`, `TH15`.

**Decisiones IRIS.** `NINGUNA`

---

## TDA-07 — Distribución marginal y por segmento

**Objetivo.** Caracterizar la distribución de los retornos, con la conciencia de que la marginal agregada es una **mezcla** de segmentos horarios y de regímenes temporales.

**Preguntas.** ¿Cuáles son media, desviación, asimetría, curtosis y cuantiles? ¿Cuán lejos está de la normal y **dónde**? ¿Es la media distinguible de cero? ¿Difieren las colas izquierda y derecha? ¿Cambia todo esto por segmento y por año?

**Fundamento Tsay.** [A] C1: exceso de curtosis alto y generalizado (S&P diario 22.81; Citigroup 55.25); densidad empírica con "pico más alto y colas más gruesas"; media diaria indistinguible de cero ($t=1.51$, $p=0.13$, $T=9{,}845$); "la asimetría no es un problema serio" **a la vez** que se rechaza simetría con $p=0.013$. [A] C1: JB supone **iid**; con clustering de volatilidad su p-valor está distorsionado y **no distingue** no normalidad marginal de heterocedasticidad con condicional normal. [A] C1: si el cuarto momento no existe, la curtosis muestral no converge y está dominada por la observación más extrema. [A] C5/C2: con $n$ enorme, todo t-estadístico es grande por construcción.

**Inputs.** TDA-04, TDA-05, TDA-06.

**Variables analizadas.** $r_t$ (y $\tilde r_t$ si existe), $\text{rg}_t$, $V_t$.

**Métodos mínimos.**
1. Momentos: media, desviación, asimetría, exceso de curtosis, mínimo, máximo.
2. **Exceso de curtosis con y sin recorte del 0.1% más extremo** — la diferencia entre ambos números es el resultado, no el número.
3. Cuantiles empíricos: 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9.
4. Densidad empírica superpuesta a la normal ajustada, y **QQ-plot** — el QQ es más informativo que cualquier test.
5. Drift: $t$-test de media cero con errores estándar **HAC** y con **bootstrap por bloques**; reportar el intervalo, no el p-valor.
6. Asimetría por cola: comparar $|q_{0.01}|$ contra $q_{0.99}$, $|q_{0.001}|$ contra $q_{0.999}$; curtosis calculada por lado.
7. **Los puntos 1–6 repetidos por segmento horario (TDA-06) y por año.**

**Métodos opcionales.** Ajuste de familias paramétricas (Student-t, mixtura de normales) — **desaconsejado en esta etapa**: la pregunta de qué distribución corresponde sólo tiene sentido **después** de TDA-10 (escala vs forma). Ejecutarlo aquí es ajustar una marginal que sabemos que es una mezcla (G4).

**Outputs obligatorios.**
- Tabla de momentos y cuantiles: global, por segmento, por año.
- QQ-plot global y por segmento.
- Tabla comparativa de colas izquierda/derecha.
- **Declaración expresa** de que un rechazo estadístico de normalidad, especialmente con una muestra muy grande, no se reporta por sí solo como hallazgo; debe informarse cuál es la **magnitud y forma** de la desviación.

**Criterios de interpretación.** El objeto de interés **no** es "¿es normal?" sino: cuánta curtosis hay, cuánta depende de un puñado de observaciones, cuánto varía entre segmentos y entre años. Si la curtosis cae drásticamente al recortar el 0.1%, el estimador está dominado por pocos puntos y **no es un parámetro estable** — con lo cual no puede usarse como descriptor. Si los momentos varían mucho entre segmentos, la marginal agregada **no describe ningún estado concreto del mercado** y debe dejar de citarse como "la distribución de MNQ".

**Dependencias.** TDA-05 (interpretación bajo discreción), TDA-06 (segmentación).

**Riesgos.** Reportar JB como hallazgo. Presentar la marginal agregada como si describiera un proceso homogéneo. Interpretar la curtosis muestral como un parámetro.

**Condiciones para avanzar.** Distribución caracterizada global y por segmento, con la magnitud de la desviación cuantificada.

**Condiciones para detenerse.** No hay STOP: es obligatoria y siempre produce resultado. Si los momentos resultan muy inestables entre segmentos, **se refuerza** la obligación de desagregar en todas las etapas siguientes.

**Hipótesis del backlog resueltas.** `TH11`, `TH12`, `TH13`.

**Decisiones IRIS.** `NINGUNA`

---

## TDA-08 — Dependencia lineal en la media

**Objetivo.** Medir la dependencia lineal de $r_t$ con su pasado, y **someterla obligatoriamente a refutación por microestructura** antes de llamarla dependencia.

**Preguntas.** ¿Es la ACF de $r_t$ distinguible de cero? ¿De qué **tamaño** en ticks? ¿Es estable por segmento y por año? Si hay $\rho_1<0$, ¿es huella de bid–ask bounce, efecto de no sincronía, o reversión económica?

**Fundamento Tsay.** [A] C2: ACF muestral, bandas de **Bartlett** (no $1/T$, que asume iid); Ljung–Box $Q(m)$ con $m\approx\ln T$ y ajuste de grados de libertad sobre residuos; "los coeficientes AR son pequeños, indicando que la dependencia serial es débil, aunque sea estadísticamente significativa al 1%" con $R^2=2.46\%$. [A] C2: ACF $=0$ **no** implica independencia salvo bajo normalidad conjunta, refutada en C1. [A] C5, modelo de Roll: $\rho_1=-0.5$ puro; con valor fundamental de camino aleatorio, $\rho_1=-\frac{S^2/4}{S^2/2+\sigma^2}\le0$; ACF de la serie direccional de IBM con pico único en el rezago 1 de $-0.389$. [A] C5, no sincronía: $\text{Cov}(r^o_t,r^o_{t-1})=-\mu^2\pi$, autocorrelación negativa **aunque la serie verdadera sea independiente**.

**Inputs.** TDA-04, TDA-05, TDA-06, TDA-07.

**Variables analizadas.** $r_t$ y $\tilde r_t$; $r_t$ agregado a 5 y 10 minutos.

**Métodos mínimos.**
1. ACF de $r_t$ hasta al menos dos jornadas de rezagos, con **bandas de Bartlett**, sobre serie cruda y ajustada.
2. PACF en los primeros rezagos.
3. Ljung–Box $Q(m)$ para varios $m$, incluyendo $m$ en los múltiplos de la jornada.
4. **Traducción a magnitud (G5)**: convertir $\hat\rho_1$ en el movimiento esperado implícito, expresado **en ticks**.
5. ACF **por segmento horario** y **por año**.
6. **Protocolo obligatorio de refutación microestructural**, si $\hat\rho_1$ es material:
   - (i) comportamiento de $\hat\rho_1$ al agregar 1 → 5 → 10 minutos (el bounce debería diluirse rápidamente);
   - (ii) comportamiento por decil de volumen y por segmento (la no sincronía debería empeorar donde hay menos actividad);
   - (iii) comparación de la magnitud implícita con el tick;
   - (iv) consistencia con la fracción de barras sin cambio de TDA-05.
7. Calibración por permutación (G2): la misma ACF sobre la serie permutada debe colapsar a las bandas.

**Métodos opcionales.** Ajustar un AR($p$) de referencia y verificar sus residuos con Ljung–Box — **sólo** para cuantificar el $R^2$ alcanzable con estructura lineal, nunca como modelo predictivo. Errores estándar HAC si se hace regresión.

**Outputs obligatorios.**
- Gráfico de ACF de $r_t$ con bandas de Bartlett, crudo y ajustado, superpuesto por año.
- Tabla de $\hat\rho_1$ por frecuencia (1/5/10 min), por segmento y por decil de volumen.
- **Tabla de traducción a ticks.**
- Veredicto explícito: `SIN DEPENDENCIA LINEAL DETECTABLE` / `DEPENDENCIA COMPATIBLE CON MICROESTRUCTURA` / `DEPENDENCIA NO ATRIBUIDA` / `NO SEPARABLE CON OHLCV LAST`.

**Criterios de interpretación.** Una ACF significativa **no** es señal. Los tres criterios que la convierten en propiedad citable son: magnitud material en ticks, estabilidad entre subperíodos, y supervivencia al intento de refutación microestructural. Si falla cualquiera de los tres, el resultado es una **propiedad del mecanismo o del ruido**, no del proceso.

**Dependencias.** TDA-06 (sin ajuste estacional la ACF tiene picos de reloj), TDA-05 (para la traducción a ticks).

**Riesgos.** Usar bandas $1/T$. Reportar significancia con $n\sim10^6$. Omitir la refutación microestructural. Leer $\rho_1<0$ como reversión explotable.

**Condiciones para avanzar.** ACF caracterizada, traducida a magnitud, y con veredicto de atribución emitido.

**Condiciones para detenerse.**
- **`STOP-8a`** — si la magnitud implícita es muy inferior al tick: se documenta como propiedad estadística sin relevancia operativa y **no se persigue**. No se ajustan modelos de media.
- **`STOP-8b`** — si el efecto es compatible con microestructura y no separable con los instrumentos disponibles: se declara `NOT SEPARABLE WITH OHLCV LAST` y se cierra la rama. **Declarar el límite es el resultado correcto**; simular haberlo resuelto no lo es.

**Hipótesis del backlog resueltas.** `TH16`, `TH17`, `TH18`.

**Decisiones IRIS.** `NINGUNA`

---

## TDA-09 — Dependencia en magnitud (volatility clustering)

**Objetivo.** Medir la dependencia en la escala, separándola del componente de reloj.

**Preguntas.** ¿Existe volatility clustering? ¿Cuánto persiste? ¿Sobrevive al ajuste estacional? ¿Decae exponencial o polinomialmente? ¿Cambia por segmento y por año? ¿Cuánta más estructura hay en magnitud que en dirección?

**Fundamento Tsay.** [A] C1: las variabilidades "aparecen en clusters". [A] C2: ACF de $|r_t|$ en índices CRSP diarios "significativa al 5% incluso después de 300 rezagos", con decaimiento polinomial $\rho_k\sim k^{2d-1}$, mientras la ACF de $r_t$ es prácticamente nula. [A] C3: tests de efecto ARCH — Ljung–Box sobre $a_t^2$ y LM de Engle; ejemplo Intel: $Q(12)=18.26$ ($p=0.11$) sobre $r_t$ vs $Q(12)=89.85$ ($p=5.3\times10^{-14}$) sobre $a_t^2$. [A] C3: **prerequisito** — hay que remover primero la estructura de media. [A] C3 (error 27): estacionalidad intradía puede simular ARCH. [A] C3: $r_t^2$ es insesgado pero muy impreciso; el rango es sustancialmente más eficiente.

**Inputs.** TDA-04, TDA-06, TDA-07, TDA-08 (residuo de media, aunque probablemente sea $r_t$ menos su media).

**Variables analizadas.** $|r_t|$, $r_t^2$, $\text{rg}_t$ — cada una **cruda y ajustada** por $s(m)$.

**Métodos mínimos.**
1. ACF de $|r_t|$, $r_t^2$ y $\text{rg}_t$ hasta **varios cientos de rezagos** (al menos dos jornadas completas), con bandas de Bartlett.
2. **Comparación crudo vs ajustado** — el resultado central de la etapa. Si la ACF colapsa al ajustar, el hallazgo era el reloj.
3. Picos en múltiplos de la jornada: verificación explícita de su presencia/ausencia antes y después del ajuste.
4. Ljung–Box y LM de Engle sobre la variable ajustada, con la advertencia de G5 sobre su interpretación.
5. **Forma del decaimiento**: ACF en escala log-log frente a escala semi-log — distingue polinomial de exponencial sin ajustar ningún modelo.
6. **Comparación magnitud vs dirección**: $Q(m)$ de $|r_t|$ frente a $Q(m)$ de $r_t$, y las ACF superpuestas en un mismo gráfico.
7. Persistencia por ventanas rodantes y por año.
8. Comparación de las tres variables de magnitud: ¿cuentan la misma historia? Si $\text{rg}_t$ es sustancialmente menos ruidoso, es el proxy preferido (C3).

**Métodos opcionales.** Estimadores de rango con supuestos (Parkinson, GK, RS, YZ) — sólo si $\ln(H/L)$ crudo resulta insuficiente para alguna pregunta concreta.

**Outputs obligatorios.**
- **Gráfico triple** de ACF: $r_t$, $|r_t|$, $r_t^2$ superpuestos — la ilustración directa de "no correlacionado pero dependiente".
- Gráfico de ACF crudo vs ajustado para $|r_t|$ y $\text{rg}_t$.
- Tabla de persistencia por año y por segmento.
- Veredicto: `CLUSTERING GENUINO` / `ARTEFACTO DE ESTACIONALIDAD` / `MIXTO`, con la fracción atribuible a cada causa.

**Criterios de interpretación.** ACF$(r)\approx0$ junto con ACF$(|r|)>0$ **no es contradicción**: es la separación entre dirección y magnitud. Si la ACF de $|r|$ colapsa al ajustar por $s(m)$, el fenómeno era determinista y **se salta TDA-11**. Si sobrevive con decaimiento lento, hay dinámica; el decaimiento polinomial sugiere memoria larga, que en C3 es también compatible con cambios de nivel no modelados — **no se resuelve esa ambigüedad aquí**.

**Dependencias.** TDA-06 (indispensable), TDA-08 (para haber removido la media).

**Riesgos.** Ejecutar sin ajuste estacional y concluir ARCH. Usar sólo $r_t^2$ (muy ruidoso). ACF demasiado corta para ver los picos de reloj. Leer clustering como señal direccional.

**Condiciones para avanzar.** Clustering caracterizado, atribuido y con persistencia medida.

**Condiciones para detenerse — `STOP-9`.** Si la ACF de magnitud colapsa completamente al ajustar por el perfil de reloj, se registra el resultado (G6), **se salta TDA-11**, y TDA-10 se ejecuta en versión mínima.

**Hipótesis del backlog resueltas.** `TH19`, `TH20`, `TH21`.

**Decisiones IRIS.** `NINGUNA`

---

## TDA-10 — Escala versus forma: origen de las colas

**Objetivo.** Responder la pregunta de bifurcación más rentable del roadmap: qué fracción de la no normalidad marginal es heterocedasticidad.

**Preguntas.** ¿Qué fracción del exceso de curtosis desaparece al estandarizar por una estimación causal de la volatilidad? ¿La distribución de $z_t=r_t/\hat\sigma_{t-1}$ es estable entre segmentos, entre estados de volatilidad y entre años? ¿La distribución cambia sólo en escala o también en forma?

**Fundamento Tsay.** [A] C1: la familia de **mixturas de escala de normales** genera colas pesadas manteniendo momentos finitos, y es "el puente" hacia la volatilidad condicional. [A] C3: un GARCH(1,1) **gaussiano** genera exceso de curtosis por sí solo, $K_a^{(g)}=6\alpha_1^2/[1-2\alpha_1^2-(\alpha_1+\beta_1)^2]$, y $K_a^{(g)}=0$ si $\alpha_1=0$. [A] C3: pero "el comportamiento de cola de los modelos GARCH sigue siendo demasiado corto incluso con innovaciones Student-t estandarizadas" — no todo es escala. [A] C7: $r_t=\mu_t+\sigma_t\epsilon_t$ permite atribuir un extremo a $\epsilon_t$ (forma) o a $\sigma_t$ (escala).

**Inputs.** TDA-07 (curtosis marginal), TDA-09 (evidencia de dependencia en magnitud).

**Variables analizadas.** $r_t$, y $z_t=r_t/\hat\sigma_{t-1}$ con $\hat\sigma_{t-1}$ **estrictamente causal**, calculada con al menos dos estimadores simples: una desviación estándar rodante y una EWMA. Si existe $s(m)$, también en su versión ajustada por reloj.

**Métodos mínimos.**
1. Construir $\hat\sigma_{t-1}$ causal (ventana rodante y EWMA; opcionalmente basada en rango). **Ningún parámetro estimado sobre toda la muestra** (G1).
2. Calcular exceso de curtosis de $r_t$ y de $z_t$; reportar la **fracción eliminada**.
3. Repetir con y sin recorte del 0.1% extremo.
4. QQ-plot de $z_t$ contra normal, y contra la referencia que corresponda.
5. **Estabilidad de forma**: comparar los cuantiles estandarizados de $z_t$ entre segmentos horarios, entre deciles de $\hat\sigma$, y entre años. Si la forma es estable, la distribución cambia **sólo en escala**.
6. Sensibilidad a la ventana del estimador de volatilidad (¿la conclusión depende de la ventana?).

**Métodos opcionales.** Estandarizar por una volatilidad ajustada por reloj además de por dinámica, para separar los dos componentes de escala.

**Outputs obligatorios.**
- **Tabla central**: curtosis de $r_t$ / curtosis de $z_t$ / fracción eliminada, por estimador y por ventana, global y por año.
- QQ-plots superpuestos de $r_t$ y $z_t$.
- Tabla de cuantiles estandarizados por decil de volatilidad — el diagnóstico directo de "escala vs forma".
- **Veredicto de bifurcación**: `ESCALA DOMINA` / `FORMA SUSTANCIAL` / `MIXTO`.

**Criterios de interpretación.** Si la curtosis cae drásticamente y los cuantiles estandarizados son estables entre deciles de volatilidad, entonces **la distribución cambia esencialmente en escala**: la agenda es la dinámica de $\sigma_t$, la caracterización de colas es un problema de escala, y **EVT no aporta** (activa `STOP-13`). Si sobrevive una fracción grande, o si los cuantiles estandarizados difieren sistemáticamente entre deciles, hay estructura de forma y la puerta a TDA-13-EVT queda abierta (aún condicionada).

**Dependencias.** TDA-09 (no tiene sentido estandarizar por una volatilidad que no tiene dinámica).

**Riesgos.** Usar $\hat\sigma$ no causal (G1). Concluir sobre una única ventana. Confundir "la curtosis bajó" con "la distribución condicional es normal" — puede bajar y seguir siendo pesada.

**Condiciones para avanzar.** Veredicto de bifurcación emitido con evidencia.

**Condiciones para detenerse.** No hay STOP propio: la etapa **activa o desactiva** STOPs de etapas posteriores.

**Hipótesis del backlog resueltas.** `TH22`, y parcialmente `TH26`.

**Decisiones IRIS.** `NINGUNA`

---

## TDA-11 — Modelo paramétrico de volatilidad *(CONDICIONAL)*

**Criterio de entrada — se ejecuta SÓLO si se cumplen las tres:**
1. TDA-09 concluyó `CLUSTERING GENUINO` (sobrevive al ajuste de reloj);
2. existe una **pregunta escrita** que los diagnósticos simples (ACF, EWMA causal, ventanas rodantes, volatilidad de rango) **no** responden;
3. esa pregunta pertenece a la caracterización, no a la predicción.

**Preguntas admisibles.** ¿Cuál es la persistencia de la volatilidad expresada en un parámetro interpretable y comparable entre subperíodos? ¿Es esa persistencia distinguible de una raíz unitaria (IGARCH) o de un cambio de nivel no modelado? ¿Existe respuesta asimétrica al signo del shock?
**Preguntas NO admisibles.** Cualquiera de la forma "¿predice mejor?" — pertenece al Nivel 4, fuera de fase.

**Fundamento Tsay.** [A] C3: GARCH es formalmente un ARMA sobre $a_t^2$; persistencia $\alpha_1+\beta_1$ (ejemplo S&P 500 mensual: 0.9772). [A] C3: **advertencia explícita** — "el fenómeno IGARCH podría ser causado por cambios ocasionales de nivel en la volatilidad"; $\hat\alpha+\hat\beta\approx1$ **no demuestra** IGARCH. [A] C3: el efecto apalancamiento tiene mecanismo corporativo; su traslado a otras clases de activo **no está autorizado** (ejemplo IBM: 0.0658 tras subida vs 0.1501 tras caída). [A] C3: comparar pronósticos contra $r_t^2$ "estrictamente hablando no es apropiado". [A] C3: hay que verificar los residuos estandarizados (ACF de $\tilde a_t$ y de $\tilde a_t^2$).

**Métodos mínimos (si se ejecuta).**
1. **Benchmark obligatorio primero**: EWMA causal, desviación rodante, volatilidad de rango. El modelo paramétrico debe declarar qué añade sobre ellos.
2. El modelo **más simple** que responda la pregunta — GARCH(1,1) antes que cualquier variante.
3. Sobre la serie **ajustada por reloj**, no la cruda.
4. Diagnóstico obligatorio de residuos estandarizados: ACF de $\tilde a_t$ y de $\tilde a_t^2$, Ljung–Box en ambos.
5. **Estabilidad**: reestimar por subperíodo. Si los parámetros varían mucho, la persistencia global es un artefacto de mezclar regímenes (advertencia explícita de C3).
6. Asimetría: comparar volatilidad condicionada al signo del shock previo **antes** de ajustar cualquier modelo asimétrico. Si la asimetría descriptiva es nula, no se ajusta EGARCH/TGARCH (G4).

**Outputs obligatorios.** Parámetros con intervalos; tabla de estabilidad por subperíodo; diagnósticos de residuos; **comparación explícita contra los benchmarks simples**; declaración de qué información aportó el modelo que los simples no daban.

**Criterios de interpretación.** Persistencia cercana a 1 **no** demuestra memoria infinita: es compatible con GARCH estacionario muy persistente, memoria larga genuina, o cambios de régimen. Esa ambigüedad **no se resuelve** en esta fase; se documenta.

**Dependencias.** TDA-09, TDA-10.

**Riesgos.** Ajustar GARCH por costumbre. Interpretar $\alpha+\beta\approx1$ como IGARCH. Trasladar el efecto apalancamiento desde acciones. Ajustar sobre la serie cruda con estacionalidad.

**Condiciones para detenerse — `STOP-11`.** Si el criterio de entrada no se cumple, **no se ejecuta**. Si al ejecutarlo no supera a los benchmarks simples en la pregunta declarada, se registra el resultado negativo (G6) y se cierra la rama.

**Hipótesis del backlog resueltas.** `TH23`, `TH24`, `TH25`.

**Decisiones IRIS.** `NINGUNA`

---

## TDA-12 — Cuantiles condicionales

**Objetivo.** Caracterizar cómo cambian los cuantiles con el contexto observable, sin diseñar ningún target.

**Preguntas.** ¿Cómo son los cuantiles incondicionales? ¿Cómo cambian por hora del día? ¿Por estado de volatilidad? ¿Por año? **¿La distribución cambia sólo en escala o también en forma?** ¿Son estables?

**Fundamento Tsay.** [A] C7: el cuantil empírico tiene varianza asintótica $p(1-p)/\{n[f(x_p)]^2\}$ — **explota** cuando la densidad es pequeña, es decir en la cola profunda; no extrapola más allá del peor evento observado; es ineficiente con $p$ pequeño; no incorpora variables explicativas. [A] C7: la regresión cuantílica de Tsay es **lineal** y se presenta sin ejemplo empírico. [A] C7: los cinco métodos de VaR sobre IBM difieren hasta un 40% entre sí y Tsay **no declara ganador**.

**Inputs.** TDA-06, TDA-07, TDA-10.

**Variables analizadas.** $r_t$ y $z_t$; condicionantes observables y **causales**: segmento horario, decil de $\hat\sigma_{t-1}$, año.

**Métodos mínimos.**
1. Cuantiles empíricos incondicionales (0.1 · 1 · 5 · 25 · 50 · 75 · 95 · 99 · 99.9) con intervalos por bootstrap por bloques.
2. Los mismos cuantiles **por segmento horario**.
3. Los mismos cuantiles **por decil de $\hat\sigma_{t-1}$ causal**.
4. Los mismos cuantiles **por año**.
5. **Prueba de escala vs forma**: dividir cada conjunto de cuantiles por la desviación del grupo. Si los perfiles normalizados se superponen → la distribución cambia **sólo en escala**. Si divergen → cambia en **forma**.
6. Número de observaciones que sostiene cada cuantil extremo — **obligatorio reportarlo**. Un cuantil 99.9 en un segmento pequeño puede descansar en decenas de puntos.

**Métodos opcionales.** Regresión cuantílica lineal — sólo si se necesita condicionar por una variable continua que la segmentación por deciles no captura, y con la advertencia de que Tsay la presenta en forma estrictamente lineal.

**Outputs obligatorios.**
- Tabla de cuantiles por segmento, por decil de volatilidad y por año, **con el $n$ que sostiene cada uno**.
- Gráfico de perfiles de cuantiles normalizados superpuestos — el diagnóstico visual directo de escala vs forma.
- Veredicto: `SÓLO ESCALA` / `ESCALA Y FORMA`.

**Criterios de interpretación.** `SÓLO ESCALA` es el resultado más simplificador posible: implica que conocer $\hat\sigma_t$ determina los cuantiles, que la caracterización de colas se reduce a caracterizar la escala, y **activa `STOP-13`** para EVT. `ESCALA Y FORMA` implica que hay estructura distribucional adicional, y sólo entonces tiene sentido plantearse EVT.

**Dependencias.** TDA-10 (la definición de $z_t$ y el veredicto de bifurcación).

**Riesgos.** Reportar cuantiles extremos sin su $n$. Condicionar por variables no causales. Presentar un cuantil estimado como un valor conocido.

**Condiciones para avanzar.** Cuantiles caracterizados y veredicto emitido.

**Condiciones para detenerse.** No hay STOP propio; activa o desactiva `STOP-13`.

**Hipótesis del backlog resueltas.** `TH26`.

**Decisiones IRIS.** `NINGUNA`

---

## TDA-13 — Extremos: inventario y agrupamiento *(obligatorio)* · EVT *(condicional)*

### Parte A — Inventario, triage y agrupamiento **(OBLIGATORIA)**

**Objetivo.** Saber qué son los extremos de esta serie antes de modelarlos, y si se agrupan.

**Preguntas.** ¿Cuántos extremos hay y dónde? ¿Cuáles son datos erróneos, cuáles artefactos de roll, cuáles eventos reales? ¿Se agrupan más de lo esperable bajo independencia? ¿Se concentran en horas concretas? ¿El agrupamiento sobrevive al ajuste por reloj y por volatilidad?

**Fundamento Tsay.** [A] C7: **extremo ≠ error de datos**, y tampoco a la inversa; descartar automáticamente sesga la cola hacia el optimismo. [A] C7: IBM tiene ACF≈0 en retornos y sin embargo $\hat\theta\approx0.82$ — **los extremos se agrupan aunque los retornos no estén correlacionados**; ignorarlo subestima el VaR ~7%. [A] C7: el índice extremal requiere **estacionariedad estricta**, y su estimador es sensible al umbral y al tamaño de bloque (llegando a valores fuera de $(0,1]$ en muestra finita). [A] C7: `evento raro ≠ económicamente relevante`. [A] C3/C7: no confundir volatility clustering con extremal clustering.

**Métodos mínimos.**
1. Definición de "extremo" **por segmento y por estado de volatilidad**, no por umbral absoluto sobre la serie completa (un umbral absoluto sobre una serie con perfil diurno selecciona horas, no eventos).
2. Inventario: fecha, hora, magnitud, contexto (¿coincide con roll? ¿con una violación de TDA-00? ¿con el borde de un hueco? ¿con un horario de dato macro conocido?).
3. **Triage documentado por observación**: `ERROR_DATO` / `ARTEFACTO_ROLL` / `ARTEFACTO_FRONTERA` / `EVENTO_REAL` / `INDETERMINADO`. Con criterio escrito **antes** de mirar el efecto sobre los resultados.
4. **Agrupamiento sin EVT**: distribución de los tiempos entre excedencias, y distribución de la longitud de rachas, comparadas con lo esperado bajo un proceso de Poisson de la misma intensidad. Este es el análisis mínimo y suficiente de clustering (G4) — el índice extremal es el método avanzado.
5. Repetir el agrupamiento sobre $z_t$ estandarizado: si desaparece, el agrupamiento **era** volatility clustering (C3/C7 lo distinguen explícitamente).
6. Distribución horaria de los extremos.

**Outputs obligatorios.** `TDA13_extremos.csv` con triage por observación; tabla de agrupamiento observado vs Poisson, en crudo y estandarizado; distribución horaria; **cuantificación del efecto del triage sobre la curtosis y sobre los cuantiles extremos de TDA-07 y TDA-12** (recalcular con y sin cada categoría).

**Criterios de interpretación.** Si el agrupamiento desaparece al estandarizar por volatilidad, no hay "clustering de extremos" propio: hay clustering de volatilidad ya caracterizado en TDA-09, y no aporta información nueva. Si sobrevive, es un fenómeno adicional y reduce el **tamaño de muestra efectivo** de la cola — con consecuencias directas sobre la incertidumbre de todo cuantil extremo.

### Parte B — EVT / POT-GPD **(CONDICIONAL)**

**Criterio de entrada — se ejecuta SÓLO si:**
1. TDA-12 concluyó `ESCALA Y FORMA` (si concluyó `SÓLO ESCALA`, se activa **`STOP-13`**); **y**
2. existe una **razón escrita** por la que se necesita extrapolar más allá del peor evento observado, en el marco de la **caracterización** (no de la gestión de riesgo de una estrategia, que pertenece a otra fase); **y**
3. TDA-14 no ha declarado la serie fuertemente no estacionaria en el segmento considerado (la EVT clásica lo requiere).

**Métodos mínimos (si se ejecuta).** Mean excess plot; elección de umbral con la regla práctica de Tsay (excedencias ≈5% de la muestra) **y** verificación de estabilidad; ajuste GPD; **análisis de sensibilidad obligatorio** del parámetro de forma a un rango de umbrales razonables; diagnósticos de tres partes (tasa de excedencia, distribución de excesos, independencia) vía QQ-plots y ACF; ejecución **dentro de segmentos horarios homogéneos**, no sobre la serie completa; cola izquierda y derecha por separado.

**Condiciones para detenerse.**
- **`STOP-13`** — si TDA-12 dio `SÓLO ESCALA`, o si los cuantiles empíricos responden ya la pregunta: **no se ejecuta EVT** (G4).
- **`STOP-13b`** — si el parámetro de forma varía sustancialmente entre umbrales razonables (en el ejemplo de IBM varía 63% entre 2% y 3%, con el error estándar duplicándose), **se reporta el rango, no un valor**, y no se usa como parámetro de nada.
- **`STOP-13c`** — si los diagnósticos de tres partes fallan, el modelo se descarta; no se reporta su salida "porque es plausible" (el modelo de Poisson homogéneo de Tsay produce salidas plausibles y **falla** los diagnósticos, cambiando el VaR en ~38%).

**Dependencias.** TDA-00 y TDA-03 (triage imposible sin máscaras), TDA-06 (umbral relativo), TDA-12 (criterio de entrada), TDA-14 (estacionariedad).

**Riesgos.** Borrar extremos. Umbral absoluto sobre serie estacional. Reportar $\xi$ como si fuera un parámetro conocido. Confundir volatility clustering con extremal clustering. Aplicar EVT a una serie no estacionaria.

**Hipótesis del backlog resueltas.** `TH06`, `TH27`, `TH28`.

**Decisiones IRIS.** `NINGUNA`

---

## TDA-14 — Estabilidad temporal y cambio estructural

**Objetivo.** Consolidar las versiones por subperíodo producidas por G3 en todas las etapas anteriores, y determinar qué propiedades son citables como "propiedades del dataset" y cuáles sólo como "propiedades de un período".

**Preguntas.** ¿Qué propiedades son estables entre años, entre subperíodos y en ventanas rodantes? ¿Hay fechas donde varias propiedades cambian a la vez? ¿Corresponden a cambios institucionales conocidos? ¿Son los precios no estacionarios y los retornos aproximadamente estacionarios, y en qué medida?

**Fundamento Tsay.** [A] C2: la receta de Tsay para estacionariedad débil es literalmente "dividir los datos en submuestras y comprobar la consistencia de los resultados". [A] C2: ADF **no rechazar** raíz unitaria no la demuestra (la nula está en la raíz unitaria); rechazarla tampoco garantiza estacionariedad futura. [A] C2: el propio texto de Tsay se desliza de "no puede rechazarse" a "contiene una raíz unitaria" — deslizamiento explícitamente a evitar. [A] C5: *intervention analysis* de Box–Tiao aplicado a la decimalización de Apple, con coeficiente significativo al 1% — plantilla para cualquier cambio institucional de fecha conocida. [A] C3: persistencia extrema puede ser artefacto de cambios de nivel.

**Inputs.** Las versiones por subperíodo de TDA-07 a TDA-13.

**Variables analizadas.** El conjunto **acotado y predefinido** de estadísticos centrales: media, desviación, exceso de curtosis, cuantiles 1/99, $\hat\rho_1$, persistencia de la ACF de $|r|$, perfil intradía, volumen medio, fracción de barras sin cambio. **La lista se fija antes de ejecutar** (G2), para que "buscar cambios estructurales" no se convierta en una búsqueda ilimitada.

**Métodos mínimos.**
1. Tabla maestra: cada estadístico × cada año, con intervalos.
2. Ventanas rodantes de los mismos estadísticos.
3. Identificación de fechas donde **varios** estadísticos se mueven simultáneamente (un solo estadístico moviéndose es ruido; varios a la vez es estructura).
4. Contraste con **fechas candidatas de cambio institucional conocido** enumeradas de antemano (cambios de especificación de contrato, de horario, de tick, cambios en el ciclo de rolls). Si no se identifica ninguna, se declara.
5. Estacionariedad: comparación de la estabilidad de momentos de $C_t$ frente a los de $r_t$. ADF sobre el log-precio es **opcional** y, si se hace, con varias especificaciones del término determinista y varios órdenes, reportando la robustez de la conclusión y **sin** deslizar de "no rechazo" a "confirmación".

**Métodos opcionales.** *Intervention analysis* sobre el rango o la volatilidad, si se identifica un evento institucional de fecha conocida.

**Outputs obligatorios.** Tabla maestra de estabilidad; gráficos de ventanas rodantes de los estadísticos centrales; lista de fechas de cambio candidato con su evidencia; **veredicto por propiedad**: `ESTABLE` / `TENDENCIA` / `CAMBIO ABRUPTO` / `INESTABLE`.

**Criterios de interpretación.** Una propiedad marcada `INESTABLE` **no puede citarse** como propiedad del dataset en el Empirical Profile: sólo como propiedad de un período. Y `cambio estadístico ≠ régimen económico identificado`: se describe el cambio, no se le pone nombre (G3).

**Dependencias.** TDA-07 a TDA-13.

**Riesgos.** Buscar cambios estructurales sin una lista predefinida aumenta sustancialmente el riesgo de detectar cambios aparentes por selección ex post. Etiquetar los períodos con narrativas económicas. Usar ADF sobre una serie continua ajustada como si testeara un instrumento real (C2 lo advierte: debería aplicarse a un contrato individual dentro de su vida activa).

**Condiciones para avanzar.** Veredicto por propiedad emitido.

**Condiciones para detenerse — `STOP-14`.** Si las propiedades centrales resultan fuertemente inestables, las etapas de Nivel 5 (TDA-15, TDA-16) **se ejecutan por subperíodo o no se ejecutan**. Buscar no linealidad o estados latentes sobre una mezcla de regímenes produce hallazgos que son el cambio de régimen mal leído.

**Hipótesis del backlog resueltas.** `TH29`, `TH30`, `TH31`.

**Decisiones IRIS.** `NINGUNA`

---

## TDA-15 — No linealidad residual *(CONDICIONAL)*

**Criterio de entrada — las cuatro:**
1. existe un residuo estandarizado limpio de media (TDA-08), volatilidad (TDA-09/TDA-10) y estacionalidad (TDA-06);
2. TDA-14 no declaró la serie fuertemente inestable (o el análisis se restringe a un subperíodo estable);
3. existe una pregunta escrita sobre la **forma** de la dependencia, no sobre su existencia;
4. el número de variantes de test está acotado y declarado de antemano (G2).

**Preguntas.** ¿Queda estructura no explicada por el modelo lineal-más-volatilidad? ¿Es estable entre subperíodos? ¿Depende de la frecuencia de muestreo?

**Fundamento Tsay.** [A] C4: una serie es lineal si admite $x_t=\mu+\sum\psi_i a_{t-i}$ con $\{a_t\}$ iid; lo demás es no lineal por definición. [A] C4: **BDS requiere remover primero la dependencia lineal**, o confunde estructura lineal no removida con no linealidad. [A] C4: BDS detecta cualquier desviación de iid y **no identifica la fuente**; es sensible a $\delta$ y $k$. [A] C4, evidencia decisiva: BDS significativo en IBM y dos índices CRSP, y sin embargo la red 8-4-1 **no supera** al random walk con deriva ($\chi^2=0.137$, $p=0.71$); la red 3-2-1 gana in-sample y empata OOS. [A] C4: el test de razón de verosimilitud para umbral tiene un parámetro de estorbo (el umbral no está definido bajo $H_0$) que invalida su distribución asintótica estándar.

**Métodos mínimos (si se ejecuta).**
1. Construcción explícita del residuo: $\hat\epsilon_t = (r_t - \hat\mu)/(\hat\sigma_{t-1}\cdot s(m_t))$, con todos los componentes causales y auditados.
2. BDS y F-test de Tsay sobre $\hat\epsilon_t$, con un número acotado de valores de $\delta$ y $k$ declarado antes.
3. **Calibración por permutación (G2)**: los mismos tests sobre $\hat\epsilon_t$ permutado y sobre datos simulados. Si el test rechaza ahí, el resultado no vale nada.
4. **Estabilidad**: repetir por subperíodo. Con una muestra muy grande, el test puede tener potencia suficiente para detectar desviaciones muy pequeñas; por tanto, **la magnitud, la estabilidad entre subperíodos y la calibración del test importan más que el rechazo aislado** (G5).
5. Dependencia con la frecuencia: repetir a 5 y 10 minutos. Si la evidencia desaparece al agregar, es candidata a microestructura (C5) más que a estructura económica.

**Métodos opcionales — explícitamente NO en esta fase.** Ajustar TAR, STAR, Markov switching o redes: **identificar la forma de la no linealidad pertenece al diseño de modelos, no a la caracterización**. C4 muestra además que detectar no linealidad no indica qué modelo usar.

**Outputs obligatorios.** Tabla de estadísticos por variante, por subperíodo y por frecuencia; resultados de la calibración por permutación; veredicto: `SIN EVIDENCIA` / `EVIDENCIA INESTABLE` / `EVIDENCIA ESTABLE, FORMA NO IDENTIFICADA`.

**Criterios de interpretación.** Rechazar linealidad ≠ predictibilidad ≠ rentabilidad. Con una muestra muy grande, un test puede detectar desviaciones de magnitud muy pequeña, por lo que un rechazo aislado **no constituye por sí solo un resultado relevante**. La evidencia debe evaluarse por su magnitud, estabilidad entre subperíodos, calibración y comportamiento al cambiar la frecuencia.

**Condiciones para detenerse.**
- **`STOP-15a`** — si no existe residuo limpio, no se testea: el test detectaría lo ya conocido.
- **`STOP-15b`** — si el resultado es no significativo, o significativo pero inestable, se registra el resultado negativo (G6) y **no se modela ninguna familia no lineal**.
- **`STOP-15c`** — aun con evidencia estable, la fase termina en "evidencia estable, forma no identificada". Identificar la forma es otra fase.

**Hipótesis del backlog resueltas.** `TH32`, `TH33`.

**Decisiones IRIS.** `NINGUNA`

---

## TDA-16 — Estados latentes / state-space *(CONDICIONAL — default: NO EJECUTAR)*

**Criterio de entrada — las tres:**
1. existe una **hipótesis latente concreta y falsable**, escrita antes: qué cantidad no observada se postula, por qué su dinámica tendría la forma supuesta, y qué observación la refutaría;
2. TDA-17 mostró que las transformaciones simples ya disponibles **no** cubren esa representación;
3. una EMA causal calibrada **no** responde la misma pregunta.

**Fundamento Tsay.** [A] C11: el local trend model **es** ARIMA(0,1,1) **es** suavizado exponencial simple; un state-space lineal invariante con estado $m$-dimensional produce ARMA($m,m$); textualmente, "based on the data alone, the decision of using ARIMA models or linear state-space models is not critical". [A] C11: en estado estacionario el filtro **es** una EMA con $\alpha=K_\infty$; verificación numérica $K_\infty\approx0.142=1-\hat\theta$. [A] C11: **no unicidad** — "there are infinitely many ways to decompose an observed time series into unobserved components… care must be exercised in interpreting the estimated components". [A] C11: dar libertad no produce dinámica (CAPM dinámico sobre GM: $\hat\sigma_\eta=4.9\times10^{-5}$, "essentially constant"). [A] C11: el smoother usa $\{v_t,\dots,v_T\}$ — **leakage por construcción**. [A] C11: el Kalman gain depende sólo de los parámetros, estimados típicamente sobre toda la muestra — **leakage por parámetros**. [B] C11: la cadena "microestructura ⟹ precio ruidoso ⟹ existe precio limpio ⟹ Kalman lo recupera" **no está en Tsay** y se rechaza.

**Métodos mínimos (si se ejecuta).** Formulación escrita de la hipótesis latente; **benchmark obligatorio** contra EMA / ventana rodante / volatilidad de rango; sólo filtering, nunca smoothing, como insumo de cualquier evaluación causal; estimación de parámetros **por subperíodo** y comparación contra la estimación global para cuantificar el leakage por parámetros; diagnóstico de innovaciones estandarizadas (Ljung–Box y LM de ARCH sobre $\tilde v_t$); análisis de sensibilidad al cociente $q=\sigma_\eta^2/\sigma_e^2$; **test formal de si el coeficiente varía**, comparando $\hat\sigma_\eta$ contra cero (procedimiento exacto del ejemplo de GM).

**Outputs obligatorios.** Hipótesis latente escrita; comparación numérica estado filtrado vs EMA; diferencia $|s_{t|T}-s_{t|t}|$ (medida directa de cuánto "hace trampa" el smoother); diagnósticos de innovaciones; declaración de causalidad de los parámetros.

**Condiciones para detenerse.**
- **`STOP-16`** *(caso por defecto)* — sin hipótesis latente concreta, **no se ejecuta**. Estudiar Kalman no es razón para aplicarlo.
- **`STOP-16b`** — si el estado filtrado resulta equivalente o casi equivalente a una EMA, se registra el resultado negativo (G6) y se cierra la rama. Es el resultado **esperado a priori** según C11.
- **`STOP-16c`** — si al permitir dinámica la estimación la encuentra ≈0, eso **refuta** la hipótesis latente. Se registra.

**Riesgos.** Aplicar Kalman por haberlo estudiado. Usar smoothing. Ignorar el leakage por parámetros. Llamar "régimen" a un estado continuo. Postular un "precio verdadero".

**Hipótesis del backlog resueltas.** `TH34`, `TH36`.

**Decisiones IRIS.** `NINGUNA`

---

## TDA-17 — Redundancia informacional entre transformaciones

**Objetivo.** Establecer, con medición y no con argumento, cuántas dimensiones efectivas contiene el conjunto de transformaciones derivables del OHLCV. **No es selección de features.**

**Preguntas.** ¿Cuánta información distinta aportan las transformaciones habituales de OHLCV? ¿Cuál es la dimensión efectiva del conjunto? ¿Qué transformaciones son reconstruibles a partir de otras?

**Fundamento Tsay.** [A] C11: Kalman(local trend) ≡ EMA ≡ ARIMA(0,1,1); representación ≠ información. [A] C3: GARCH ≡ ARMA sobre $a_t^2$. [B] C11: fabricar múltiples transformaciones del mismo input sin verificar redundancia infla el espacio de búsqueda y, con señal/ruido baja, el riesgo de relaciones espurias.

**Inputs.** El conjunto de transformaciones **causales** derivadas de TDA-04, más un conjunto acotado y declarado de transformaciones habituales (momentum/ROC, EMA a varias ventanas, medidas de volatilidad de tipo ATR, bandas de tipo Bollinger, osciladores de tipo RSI, el estado de Kalman si TDA-16 se ejecutó). **La lista se cierra antes de ejecutar** (G2).

**Métodos mínimos.**
1. Matriz de correlación (Pearson y de rangos) entre todas las transformaciones.
2. **$R^2$ de cada transformación regresada sobre todas las demás** — la medida directa de redundancia: un $R^2$ muy alto significa que esa transformación es reconstruible y no aporta dimensión.
3. Dimensión efectiva del conjunto: número de componentes necesarios para explicar una fracción alta de la varianza conjunta.
4. Agrupamiento de transformaciones por similitud.
5. **Todo lo anterior por segmento y por año** (una redundancia estable es una propiedad; una que cambia es un artefacto).

**Métodos opcionales.** Medidas de dependencia no lineal entre transformaciones — sólo si TDA-15 encontró evidencia estable de no linealidad.

**Outputs obligatorios.** Matriz de correlación; tabla de $R^2$ de reconstrucción por transformación; **número de dimensiones efectivas**; agrupamiento; y el marco conceptual resultante: para cada transformación, clasificación en `NUEVA REPRESENTACIÓN` / `RECONSTRUIBLE` — con la advertencia de que **ninguna** es `NUEVA INFORMACIÓN`, porque todas son funciones del mismo OHLCV.

**Criterios de interpretación.** Una dimensión efectiva baja frente a un número alto de transformaciones es el resultado esperado y es **el argumento cuantitativo** contra la proliferación de indicadores. Reconstruible **no** significa inútil: puede ser una representación más conveniente. Significa que no amplía el espacio de información y que, por tanto, **no debe contarse como una prueba independiente** a efectos de multiplicidad (G2).

**Dependencias.** TDA-04 a TDA-12 (las transformaciones deben estar definidas y auditadas).

**Riesgos.** Convertir esto en selección de features (**prohibido**). Ampliar la lista de transformaciones después de ver los resultados. Interpretar correlación baja como información independiente (puede ser ruido).

**Condiciones para avanzar.** Dimensión efectiva medida y transformaciones clasificadas.

**Condiciones para detenerse.** No hay STOP: siempre produce resultado.

**Hipótesis del backlog resueltas.** `TH35`, `TH37` (parcialmente).

**Decisiones IRIS.** `NINGUNA`

---

## TDA-18 — Empirical Profile of MNQ OHLCV

**Objetivo.** Consolidar el resultado de la fase en un documento que responda las 20 preguntas del perfil empírico y que declare el estado de cada hipótesis del backlog.

**Métodos mínimos.** No hay análisis nuevo. Se redacta el documento, se verifica que cada afirmación tenga su output de origen, y se auditan tres cosas: (i) que ninguna propiedad marcada `INESTABLE` en TDA-14 se cite como propiedad del dataset; (ii) que ninguna afirmación se apoye sólo en un p-valor (G5); (iii) que toda cantidad presentada como disponible en $t$ haya pasado el test de reconstrucción causal (G1).

**Outputs obligatorios.**
1. **`EMPIRICAL_PROFILE_MNQ_OHLCV.md`** respondiendo:
   1. ¿Los datos son confiables? · 2. ¿Qué representa exactamente cada observación? · 3. ¿Cómo deben construirse los retornos? · 4. ¿Cuál es su distribución? · 5. ¿Cómo cambia intradía? · 6. ¿Hay dependencia lineal? · 7. ¿Hay dependencia en magnitud? · 8. ¿Existe volatility clustering? · 9. ¿Hay estacionalidad? · 10. ¿Las propiedades son estables temporalmente? · 11. ¿Existen cambios estructurales? · 12. ¿Hay evidencia de dependencia no lineal? · 13. ¿Cómo se comportan cuantiles y colas? · 14. ¿Los extremos se agrupan? · 15. ¿Qué efectos pueden venir de microestructura? · 16. ¿Qué no podemos estudiar por no tener Bid/Ask ni ticks? · 17. ¿Qué transformaciones serían potencialmente redundantes? · 18. ¿Qué hipótesis fueron rechazadas? · 19. ¿Qué hipótesis sobrevivieron? · 20. ¿Qué preguntas permanecen abiertas?
2. **Backlog actualizado** con el estado final de cada `THxx`: `RESPONDIDA` / `RECHAZADA` / `NO OBSERVABLE` / `ABIERTA` / `DIFERIDA`.
3. **Registro de STOPs activados**, con la razón de cada uno.
4. **Lista de preguntas bloqueantes para la fase de ML** (típicamente: semántica del timestamp si quedó `INDETERMINADO`, y método de roll si quedó `INDETERMINADO`).

**Condiciones para avanzar.** Con este documento cerrado, y **sólo entonces**, se abre una fase **separada** de `ML PROBLEM DESIGN`.

**Decisiones IRIS.** `NINGUNA`

---

# 8. Registro consolidado de condiciones de STOP

| ID | Etapa | Se activa cuando | Consecuencia |
|---|---|---|---|
| `STOP-0` | TDA-00 | Violaciones de invariantes no despreciables o concentradas | **Detener el roadmap.** Escalar al proveedor. |
| `STOP-2` | TDA-02 | La sesión real difiere sustancialmente de la declarada | Volver a TDA-01 y corregir la definición. |
| `STOP-3` | TDA-03 | Discontinuidades no atribuibles a rolls ni a eventos conocidos | Volver a TDA-00: son datos, no mercado. |
| `STOP-4` | TDA-04 | La regla de no-cruce elimina una fracción sustancial | Revisar TDA-01; **no** relajar la regla. |
| `STOP-5` | TDA-05 | Un segmento es efectivamente discreto | Analizarlo aparte o excluirlo, con decisión documentada. |
| `STOP-6` | TDA-06 | El perfil de reloj es débil o inestable | **No construir $s(m)$**; el resto se ejecuta sólo en crudo. Resultado negativo válido. |
| `STOP-8a` | TDA-08 | La magnitud implícita de $\hat\rho_1$ es muy inferior al tick | Documentar como propiedad, no perseguir. No modelar la media. |
| `STOP-8b` | TDA-08 | El efecto es compatible con microestructura y no separable | Declarar `NOT SEPARABLE WITH OHLCV LAST`. Cerrar la rama. |
| `STOP-9` | TDA-09 | La ACF de magnitud colapsa al ajustar por reloj | **Saltar TDA-11.** TDA-10 en versión mínima. |
| `STOP-11` | TDA-11 | No hay pregunta que los diagnósticos simples no respondan; o el modelo no supera a los benchmarks | **No ajustar / descartar el modelo.** |
| `STOP-13` | TDA-13B | TDA-12 dio `SÓLO ESCALA`, o los cuantiles empíricos ya responden | **No ejecutar EVT.** |
| `STOP-13b` | TDA-13B | El parámetro de forma varía sustancialmente entre umbrales razonables | Reportar el **rango**, no un valor. No usarlo como parámetro. |
| `STOP-13c` | TDA-13B | Los diagnósticos de tres partes fallan | Descartar el modelo, no reportar su salida. |
| `STOP-14` | TDA-14 | Las propiedades centrales son fuertemente inestables | Nivel 5 se ejecuta **por subperíodo o no se ejecuta**. |
| `STOP-15a` | TDA-15 | No existe residuo limpio de media + volatilidad + estacionalidad | **No testear.** |
| `STOP-15b` | TDA-15 | Sin evidencia, o evidencia inestable entre subperíodos | Registrar resultado negativo. **No modelar familias no lineales.** |
| `STOP-15c` | TDA-15 | Evidencia estable | La fase termina en "forma no identificada". Identificarla es otra fase. |
| `STOP-16` | TDA-16 | **Caso por defecto:** no hay hipótesis latente concreta | **No ejecutar state-space.** |
| `STOP-16b` | TDA-16 | El estado filtrado es equivalente a una EMA | Registrar resultado negativo. Cerrar la rama. |
| `STOP-16c` | TDA-16 | La estimación encuentra dinámica ≈ 0 | La hipótesis latente queda **refutada**. Registrar. |

**Lectura del registro.** Nueve de los veinte STOPs cierran ramas **avanzadas** (TDA-11, 13B, 15, 16). Esto es intencional: **el escenario más probable y perfectamente aceptable es que ninguna de las cuatro etapas condicionales llegue a ejecutarse.** Un roadmap que se ejecuta entero es un roadmap sin condiciones de entrada reales.

---

# 9. Visualización — qué gráfico responde qué pregunta

Sólo se produce un gráfico si responde una pregunta que una tabla no responde mejor.

| Gráfico | Etapa | Pregunta que responde | Por qué no basta una tabla |
|---|---|---|---|
| Heatmap de completitud (día × minuto) | TDA-02 | ¿Los huecos son estructurales o esporádicos? | El patrón espacial es el resultado |
| Histograma en múltiplos de tick | TDA-05 | ¿Cuán discreta es la variable? | La forma de la grilla es visible sólo así |
| Perfil intradía superpuesto por año | TDA-06 | ¿Existe patrón de reloj y es estable? | Superposición = estabilidad; una tabla la esconde |
| QQ-plot de $r_t$, global y por segmento | TDA-07 | ¿**Dónde** se desvía de la normal? | Los momentos dicen cuánto, no dónde |
| $\log\text{Var}(r[h])$ vs $\log h$ | TDA-04 | ¿Escala la varianza con el horizonte? | La pendiente es el resultado |
| ACF triple ($r$, $\lvert r\rvert$, $r^2$) | TDA-09 | ¿No correlacionado pero dependiente? | La comparación visual **es** el argumento |
| ACF crudo vs ajustado | TDA-09 | ¿El clustering es reloj o dinámica? | La diferencia entre curvas es el resultado |
| QQ de $r_t$ y de $z_t$ superpuestos | TDA-10 | ¿Cuánta cola es heterocedasticidad? | Muestra el efecto en toda la distribución, no en un momento |
| Perfiles de cuantiles normalizados | TDA-12 | ¿Escala o forma? | Superposición vs divergencia es el diagnóstico |
| Ventanas rodantes de estadísticos centrales | TDA-14 | ¿Cuándo cambian las propiedades? | La localización temporal es el resultado |
| Mean excess plot | TDA-13B | ¿Qué umbral y hay linealidad? | Es un diagnóstico intrínsecamente visual |
| Mapa de correlación / agrupamiento | TDA-17 | ¿Qué transformaciones son la misma cosa? | La estructura de bloques es el resultado |

**Explícitamente NO se producen:** series de precio "para ver el mercado", gráficos de indicadores técnicos, ni ningún gráfico que ilustre una intuición sin una pregunta detrás.

---

# 10. Auditoría crítica de este roadmap

Ejecutada antes de la entrega, según el cuestionario del prompt maestro.

| Pregunta de auditoría | Veredicto | Evidencia |
|---|---|---|
| ¿Hay algún análisis incluido sólo porque aparece en Tsay? | **No.** GARCH, EVT, modelos no lineales y state-space son las cuatro técnicas "estrella" de los capítulos estudiados, y **las cuatro son condicionales**, con criterios de entrada escritos y con STOP por defecto en contra. | TDA-11, 13B, 15, 16 |
| ¿Hay técnicas avanzadas sin prerequisito empírico? | **No.** Cada condicional tiene entre 2 y 4 condiciones de entrada verificables. | §7 |
| ¿Hay análisis duplicados? | **No**, tras fusionar. Se fusionaron resolución y discreción en una sola etapa; actividad y estacionalidad en TDA-06; inventario y agrupamiento de extremos en TDA-13A. | §4 nota |
| ¿Puede responderse alguna pregunta con un método más simple? | Revisado sistemáticamente en G4 y en la tabla de `síntesis §2.10`. Se sustituyó índice extremal por conteo de runs vs Poisson como método mínimo; descomposición state-space por perfil promedio en TDA-06; estimadores de rango con supuestos por $\ln(H/L)$ crudo en TDA-04. | G4, TDA-06, TDA-13A |
| ¿El orden respeta las dependencias? | **Sí**, y se documenta el razonamiento causal de cada una. La desviación deliberada respecto del orden del libro (reloj antes que dependencia) está fundamentada. | §5, `síntesis §2.5` |
| ¿Hay riesgo de usar información futura? | Mitigado por G1 con test de reconstrucción, y por el requisito de que cada output declare `CAUSAL` o `RETROSPECTIVO`. Se identifican explícitamente las tres vías de fuga (smoothing, parámetros, ajuste de roll). | G1, TDA-03, TDA-06, TDA-16 |
| ¿Se está interpretando OHLCV como algo que no es? | Mitigado por G0 y por la tabla de no observables. Se prohíbe explícitamente ACD sobre timestamps de barras, tratar Volume como número de trades, y postular un "precio verdadero". | G0, `síntesis §3.3` |
| ¿Se confunde significancia con relevancia práctica? | Mitigado por G5 (prohibición de reportar p-valores solos) y por el umbral de relevancia en ticks fijado en TDA-05. | G5, TDA-05 |
| ¿Se confunde descriptivo con predictivo? | La frontera está en el Nivel 3 y se declara en §1.2. Ninguna etapa entrena un modelo para predecir. | §1.2 |
| ¿Se derivan decisiones de ML? | **No.** `DECISIÓN IRIS: NINGUNA` en las 19 etapas. Las ideas de diseño se registran como `POSSIBLE FUTURE ML IMPLICATION`. | §7 |
| ¿Se está intentando "encontrar señal" en vez de caracterizar? | Mitigado por G6 y por el diseño de los STOPs: la mayoría cierran ramas al no encontrar nada, y ese es el resultado esperado. | G6, §8 |
| ¿Hay condiciones claras de STOP? | 20 STOPs, cada uno con activador y consecuencia. | §8 |
| ¿Puede reducirse el plan sin perder información esencial? | Se intentó. **No** puede reducirse más sin romper una dependencia: cada etapa obligatoria es prerequisito interpretativo de otra, o responde una de las 20 preguntas del perfil. La reducción real ya está en que 4 de las técnicas más costosas son condicionales con STOP por defecto en contra. | §4, §8 |

### Riesgos residuales conocidos de este roadmap

1. **Dependencia de información no estadística.** TDA-01 no puede completarse sólo con los datos. Si la documentación del proveedor no aparece, el roadmap continúa bajo el supuesto conservador, pero la fase de ML queda bloqueada. Es el riesgo residual más serio.
2. **Imposibilidad estructural en TDA-08.** La atribución del $\rho_1$ probablemente **no** sea resoluble con OHLCV Last. El roadmap lo maneja declarando el límite (`STOP-8b`) en vez de forzar una conclusión, pero eso significa que una pregunta relevante quedará abierta por falta de datos, no por falta de análisis.
3. **Tensión entre EVT y estacionariedad.** Si el perfil diurno es fuerte, la EVT clásica puede no ser aplicable ni siquiera dentro de segmentos. El roadmap lo aborda restringiendo a segmentos homogéneos, pero cabe que la respuesta correcta sea que la caracterización de colas de esta serie **no admite** EVT clásica.
4. **Presupuesto de intentos.** G2 exige acotar el número de variantes de antemano, pero no fija ese número: depende de cuánta muestra tenga el dataset y de cuánto holdout se reserve. **Debe fijarse antes de ejecutar TDA-07**, y es uno de los puntos que conviene consultar antes de arrancar.

---

**ROADMAP DESIGNED — NO EMPIRICAL ANALYSIS EXECUTED — NO IRIS DESIGN DECISIONS ADOPTED.**
