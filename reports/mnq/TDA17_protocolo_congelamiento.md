# TDA-17 — Protocolo para el congelamiento metodológico

**Estado:** `CONGELADO`.

Mientras el estado sea `PENDIENTE_AUDITORIA_FINAL`, ni este documento ni `configs/tda17.yaml` autorizan la ejecución de TDA-17. El paso a `CONGELADO` sólo se hace con el procedimiento de §9, que cambia únicamente la línea de estado anterior y la del YAML.
**Base:** commit `6245b54` (cierre de TDA-15). Ningún indicador se ha calculado, no se ha ejecutado ningún análisis sobre MNQ y no se ha leído el holdout.
**Especificación normativa:** `configs/tda17.yaml`. Este documento la resume y la justifica. Si hay discrepancia, prevalece el YAML.
**Integridad:** `reports/mnq/TDA17_manifiesto_protocolo.json` (SHA-256 de los 9 ficheros del protocolo, de las 5 entradas y de los 188 artefactos previos), verificación estricta oficial (§9) y tests `tests/test_tda17_*.py`.
**Hipótesis:** TH35 (redundancia) y TH37 (causalidad). **Estatus G2:** `EXPLORATORIO_PREDECLARADO`. **DECISIÓN IRIS: NINGUNA.**

---

## 1. Origen del protocolo

El diseño se auditó en seis iteraciones: v1, v2, v3, v4, la adenda v4a y la salvaguarda v4b (§5). Esos documentos de trabajo no están versionados en el repositorio; sí lo están sus pruebas sintéticas, como evidencia (`tests/test_tda17_diseno_v4a_sintetico.py`, `tests/test_tda17_diseno_v4b_g23b_viabilidad.py`, con las aserciones sin cambios). Este documento y el YAML consolidan su contenido vigente sin contradicciones, más las correcciones C1–C5 y F1–F4 de la auditoría final (§6–§9).

Cambios de consolidación respecto de los documentos de trabajo (todos verificados por los tests):

1. **Convención de calentamiento K explícita.** K es el **K teórico mínimo**: índice (base 1) de la primera barra **estructuralmente elegible** tras un reinicio si no hay discontinuidades (la elegibilidad con huecos y su diferencia con la primera salida finita se definen en §6). La primera barra tras un roll no tiene par consecutivo, por lo que TR y BP son locales y los incrementos empiezan en la barra 2. Las etapas encadenadas consumen a partir de la K de la etapa anterior.
   Con esa convención se corrigen siete valores aproximados de v1/v2:

   | Elemento | Antes | Ahora |
   |---|---|---|
   | MACD/PPO | 116 / 269 / 1.068 | **115 / 268 / 1.067** |
   | TRIX | 151 / 598 | **149 / 596** |
   | TSI (25,13) | 127 | **126** (el 503 de (100,52) no cambia) |
   | Stoch RSI | 98 | **95** |
   | ATR/NATR | 78 / 336 / 1.344 | **77 / 335 / 1.343** |
   | Ultimate Oscillator | 29 / 121 | **28 / 120** |
   | Choppiness | 15 / 61 | **14 / 60** |

   K máximo del catálogo: **1.343**.
2. Choppiness queda en la versión **C2** (D22).
3. Relative Volume con `J(t)` fijo (D4).

## 2. Recuentos [ARITMÉTICO]

| Concepto | Valor |
|---|---|
| Elementos | 45 (tendencia 12, momentum 10, volatilidad 10, volumen 8, velas 5) |
| Elementos por panel | 41 con salida en P · 3 sólo en L (B2 Momentum, C1 ATR, E5 rango en ticks) · 1 sólo Aux (D4 A/D Line) |
| Configuraciones | 102 |
| Salidas calculadas | 144 = 128 P (análisis oficial completo) + 16 L (diagnóstico auxiliar, sólo por contrato) |
| Series auxiliares | 2 (OBV y A/D acumulados, sólo tests) |
| Series nuevas | 146 |
| Columnas oficiales P | 134 = 128 + 6 referencias de TDA-04 (`r_1m`, `abs_r_1m`, `log_hl`, `log_co`, `log_oc_prev`, `ln V`) |
| Identidades lineales registradas | 11 en P (9 homogéneas + 2 afines) y 3 en L |
| Columnas de P implicadas en identidades | 32 |
| Rango máximo | P sin centrar ≤ 124 · con intercepto ≤ 124 · centrada ≤ 123 · B = 123 columnas · L ≤ 13 |

## 3. Estado de las decisiones D1–D24

| Id | Estado | Contenido vigente |
|---|---|---|
| D1 | Resuelta | Política P1: reinicio sólo en roll; incrementos sólo entre barras consecutivas (`CONS`); niveles encadenados dentro del contrato y marcados; sensibilidades S1/S2 |
| D2 | Resuelta | Escalas: parámetros canónicos y anclas de 60 y 240 barras (TDA-10, TDA-08H) |
| D3 | Resuelta | Formas adimensionales en P; MACD convencional en L y `MACD_N = MACD/C` en P; PPO convencional |
| D4 | Resuelta | RVOL `k ∈ {5, 20}`; `J(t)` = las k jornadas inmediatamente anteriores, fijas y sin sustitución; cruza rolls con marca; S7; chequeo de comparabilidad descriptivo **sin consecuencias metodológicas** |
| D5 | Resuelta | VWAP anclado al inicio de `trading_date` (18:00 NY) |
| D6 | Resuelta | Cuerpo relativo sin signo |
| D7 | Resuelta | 6 referencias de TDA-04 en P; `R_1m`, `r2_1m` y `zero_1m` sólo en la tabla de identidades |
| D8 | Resuelta | Umbrales: R² 0,90 (0,50 descriptivo); \|ρ\| 0,90 y 0,70; estabilidad en ≥ 5/6 años y ≥ 6/7 segmentos; materialidad T3 0,10; falsas T3 ≤ 5 %; V3 ≤ 0,081 |
| D9 | Resuelta | Bootstrap de bloques móviles circulares de 5 jornadas, 300 réplicas, semilla fija; S4 con bloques de 1 y 10 |
| D10 | Resuelta | Celdas: años 2020–2025 (2019 `NO_EVALUABLE`; 2025 parcial) y los 7 segmentos de TDA-07; año × segmento sólo descriptivo |
| D11 | Resuelta | Controles G2-1a, G2-1b, G2-2 (generativos, V1), G2-3a y G2-3b (desplazamientos, V1′); V2; V3 |
| D12 | Resuelta | R² sobre rangos como variante secundaria |
| D13 | Resuelta | Diagnóstico de reloj S5 (RETROSPECTIVO) |
| **D14** | **Aprobada explícitamente** | `EXPLORATORIO_PREDECLARADO`. El holdout no se usa en TDA-17 y no se programa ninguna confirmación futura automática |
| D15 | Resuelta | Persistir `data/interim/mnq/tda17_indicadores_1m.parquet` (ignorado por git) |
| D16 | Resuelta | Implementación propia, sin dependencias nuevas |
| D17 | Resuelta | Panel L como diagnóstico auxiliar por contrato, fuera de las salidas obligatorias |
| D18 | Resuelta | No se añaden los σ̂ de TDA-10 |
| D19 | Resuelta | Transferencia de la reconstrucción dejando fuera un año o un segmento, sin regularización ajustada |
| D20 | Resuelta | Huecos internos tratados como discontinuidad (coherente con TDA-04) |
| D21 | Resuelta | Spearman convencional (rango medio); IC con rangos fijos por celda, validado contra recálculo exacto |
| **D22** | **Aprobada explícitamente** | Choppiness **C2** como única versión: fórmula convencional con TR según P1; valores fuera de [0, 100] conservados y marcados; S8 |
| **D23** | **Aprobada explícitamente** | R² de cada columna de P frente a las otras 133 (base específica por objetivo); etiquetas de exactitud por registro y residuo (nunca sólo por R²); PCA oficial sobre P y complementario sobre B |
| **D24** | **Aprobada explícitamente** | G2-3a sobre la rejilla (jornada × minuto) con desfases relativos; G2-3b con garantía por pares y diagnóstico de viabilidad previo (§5) |

"Resuelta" = definida en el diseño auditado (auditoría de la v4a favorable). Todas quedan sujetas a la auditoría final de este protocolo.

## 4. Reglas `NO_EVALUABLE` y de no publicación

Ninguna de estas reglas cambia umbrales, nulos ni parámetros después de ver resultados: sólo suprimen o etiquetan publicaciones.

| Ámbito | Regla |
|---|---|
| Año 2019 | Siempre `NO_EVALUABLE` |
| G2-3a | Menos de 40 desfases elegibles en `{20, …, ⌊|D|/2⌋}` con `|Ω_k| ≥ 0,5·N` → `NO_EVALUABLE`; la componente de reloj se mantiene vía G2-1b |
| `k_PA` | Separación `|D|/(2p)` < 1 jornada → `NO_EVALUABLE` |
| `k_PA` | Certificado de imposibilidad de G2-3b (§5) → `NO_EVALUABLE` **sin intentos de generación** |
| `k_PA` | Más del 10 % de réplicas G2-3b no generadas → `NO_EVALUABLE` |
| `k_PA` | V3 > 0,081 en la celda → no se publica |
| R² | Verificación cruzada > 1e-8 → `R2_NO_PUBLICABLE_POR_COLINEALIDAD` |
| EXACTA | Hueco espectral < 1e4 → `SEPARACION_NUMERICA_INSUFICIENTE` |
| T3 | V2 falla tras la contingencia de bloques de 10 jornadas → T3 no se publica |
| Transferencia | R²_in < 0,90 → `NO_APLICA` |
| RVOL | Menos de ⌈0,8k⌉ observaciones en `J(t)` → NaN |
| Ventanas de incrementos | Menos de ⌈0,8n⌉ pares `CONS` → NaN |
| G2-2 | Calibración de df sin convergencia o df ≤ 2 → `NO_EVALUABLE` (no ocurre: df = 2,554); semilla que no pasa la verificación → bloqueo por error de implementación, sin recalibrar (§7) |

## 5. Salvaguarda v4b — viabilidad de G2-3b

**Problema.** En un segmento cuyas jornadas tienen W minutos regulares, la alineación entre dos columnas desplazadas es 1 si su desfase relativo es múltiplo de W y 0 en otro caso. Exigir alineación ≤ 0,05 en todas las parejas obliga a que los p = 123 desfases tengan fases horarias distintas, cosa imposible si W < 123 (principio del palomar). Los segmentos de 60 min (02:00–03:00 y 08:30–09:30) y de 120 min (00:00–02:00) están en ese caso si sus jornadas son regulares.

**Diagnóstico previo (sólo timestamps y máscaras):**
- `Bad = {δ ≠ 0 : a(δ) > 0,05 o distancia circular(δ) < L_min}`, con `a(δ)` exacta por FFT y Bad simétrico.
- Un conjunto válido de desfases es un conjunto independiente del grafo de Cayley `Cay(Z_N, Bad)`. Para estos grafos `α·ω ≤ N`, porque el mapa `(i, c) ↦ i + c` de (independiente × clique) en `Z_N` es inyectivo.
- Se acota ω inferiormente con el bloque `{0, …, L_min−1}` y con progresiones `{0, s, …, (K−1)s}` cuyas diferencias caen todas en Bad (50 pasos candidatos de mayor alineación, en orden determinista).
- Si `⌊N/ω⌋ < p`, la generación es imposible: `k_PA = NO_EVALUABLE` sin ningún intento.
- Si no hay certificado, se genera con un máximo de 10.000 intentos por columna. Si se agotan, se aplica la regla del 10 % de réplicas.

**Pruebas sintéticas (p = 123, sin MNQ; 6 superadas):**
- jornadas regulares de 60 y 120 min → certificado de imposibilidad con 0 intentos y `α ≤ W`;
- jornadas regulares de 150 min → generación válida con las 7.503 parejas verificadas y 123 fases distintas;
- caso amplio irregular (100 jornadas de 300–400 filas) → generación válida;
- instancia pequeña (N = 20, p = 6) → la búsqueda exhaustiva confirma que el certificado es correcto y que la cota es alcanzable;
- simetría de Bad.

**Consecuencia esperada [ESTIMACIÓN]:** `k_PA` será `NO_EVALUABLE` en algunos segmentos cortos y quizá en 2025. Las demás métricas de dimensión (k80/k90/k95, ratio de participación, rango efectivo, M_eff) no dependen de G2-3b y se publican en todas las celdas.

## 6. Calentamiento con discontinuidades (corrección C2)

Las K del catálogo son **K teóricos mínimos**: la primera barra que puede ser elegible si todas las barras posteriores al reinicio tienen par `CONS`. Ningún parámetro ni ventana cambia.

**Elegible no es lo mismo que finito (corrección F4).** Una barra es **estructuralmente elegible** cuando su calentamiento está completo: la ventana está llena o se ha cumplido el requisito de actualizaciones. Depende sólo del esqueleto (reinicios y pares `CONS`), nunca de los valores. Ser elegible **no** garantiza una salida finita. En una barra elegible el valor es NaN si lo exigen las reglas de división ya existentes: denominador 0 o no finito, σ < 1e-8, H = L. Por ejemplo, una ventana plana con HH = LL o σ = 0, una suma de volumen nula, V = 0 dentro de un logaritmo o un RSI con W(dC+) + W(dC−) = 0. Por eso la primera salida finita es ≥ la primera barra elegible y se registra aparte.

**Tres niveles distintos (corrección B):**

| Nivel | Depende de | Dónde aplica |
|---|---|---|
| 1. **Elegibilidad estructural** | **Sólo** el esqueleto temporal: reinicios, barras observadas y pares `CONS` | Todas las clases |
| 2. **Activación algorítmica** | Los **valores**; se evalúa sólo en barras ya elegibles | Sólo `DEPENDIENTE_DE_DATOS` (KAMA); en las demás clases coincide con la elegibilidad |
| 3. **Primera salida finita** | Las reglas de división sobre los valores | Todas las clases |

Siempre se cumple: elegibilidad ≤ activación ≤ primera salida finita.

Cada elemento queda en una clase (45 en total):

| Clase | Elementos | Primera barra elegible |
|---|---|---|
| `EXACTA` (33) | niveles, V, ADL, variables locales, TR y BP según P1 (incluidos ATR/NATR, Keltner, UO, CHOP, MACD, PPO, TRIX, Chaikin y Supertrend) | **= K** en todo contrato: sus entradas existen en todas las barras observadas y las actualizaciones nunca se detienen. La primera salida finita puede ser posterior |
| `CONTADOR_CONS` (4) | A8 ADX/DMI, B3 RSI, B8 TSI, B9 Stoch RSI | `E = min{t : u_t ≥ U}` ≥ K, con `K = U + 1` |
| `VENTANA_VALIDEZ` (6) | B1 ROC, B2 Momentum, B7 CMO, C6 σ de retornos, D1 flujo OBV, D2 MFI | K es el mínimo para que la ventana quepa; elegible sólo con ≥ ⌈0,8n⌉ pares válidos; puede serlo más tarde y dejar de serlo |
| `DEPENDIENTE_DE_DATOS` (1) | A5 KAMA | `K_min` es la primera barra que puede ser estructuralmente elegible; la **activación** exige además una condición sobre los valores (abajo) |
| `JORNADAS` (1) | D7 RVOL | ⌈0,8k⌉ observaciones en `J(t)` |

**Activación de KAMA.**
- **KAMA.**
  - *Elegible* si t ≥ n + 1 y la ventana tiene ≥ ⌈0,8n⌉ pares `CONS`.
  - *Se activa* en la primera barra elegible con ER finito, es decir, con Σ|dC| > 0 en la ventana. En ella KAMA := C y la salida vale 0.
  - Con precios constantes ER = 0/0, así que KAMA no se activa aunque la barra sea elegible.
  - Tras activarse, en las barras con ER NaN retiene el estado y da NaN.

**Supertrend A11: convención clásica de TradingView, con el estado interno separado de la publicación.** Pasa a la clase `EXACTA` y K = 54 y 335 (los mismos valores de antes, que ahora son la **frontera de publicación**). Especificación completa en `catalogo.A11.convencion` del YAML.
- **Estado interno:** empieza con el **primer ATR disponible**, en la barra n del contrato: la semilla es la media simple de los n primeros TR y después se aplica Wilder, con TR según P1.
- **Inicialización en la barra n:** bandas finales = básicas, `(H+L)/2 ± m·ATR`, y dirección **bajista** (ST = banda superior).
- **Recursión en cada barra, también durante el calentamiento adicional:**
  - las bandas finales son las clásicas, independientes de la dirección;
  - la rama se elige comparando por **valor** si `ST_(t−1) == sup_(t−1)`, igual que TradingView, empates incluidos;
  - la comparación se hace con las bandas de t.
- **Publicación:** `dist = ln(C/ST)` y `dir` sólo desde K. Antes de K son NaN aunque el estado ya exista. **No se reinicializa en K.**
- **Signo de IRIS:** +1 alcista, −1 bajista. TradingView usa el signo opuesto.
- **Reinicio completo en cada roll.** En las discontinuidades P1, el TR es local y las bandas usan el cierre de la barra observada anterior.
- **N10 no es una identidad (corrección posterior):** `dir = sign(ln C/ST)` es la concordancia habitual, pero no se cumple siempre. Al girar la dirección, ST pasa a la banda final nueva, que puede quedar del lado contrario del cierre.
  - **Contraejemplo sintético:** barras estables alrededor de 20000 (TR = 0,5) y después una barra con H = 20020, L = 20000 y C = 20002. Da dirección +1 con ln(C/ST) < 0, tanto en n10 (ST = 20002,65) como en n60 (ST = 20007,525). La referencia Pine produce lo mismo.
  - **Clasificación:** N10 conserva su identificador por trazabilidad, pero pasa a ser `DIAGNOSTICO_NO_UNIVERSAL`. Nunca se exige: ni en V1 de los controles G2, ni en las etiquetas de exactitud, ni en las pruebas.
  - **Lo que no cambia:** la fórmula de Supertrend y las 11 identidades lineales registradas, con sus rangos (N10 nunca formó parte de ellas).

**Diferencias frente a la referencia TradingView:**
1. **Publicación desde K, no desde el primer ATR.** El estado interno es idéntico; sólo cambia desde cuándo se publican las salidas.
2. **TR local en las discontinuidades P1 y reinicio por contrato.** Un gráfico continuo de TradingView no distingue pares `CONS`.
3. **Signo invertido.**

**Diferencias frente al diseño previo:**
- Supertrend ya no espera a un primer cruce: sale de `DEPENDIENTE_DE_DATOS`, de modo que ahora sólo queda KAMA en esa clase y `EXACTA` pasa de 32 a 33 elementos.
- `K_min` / `K_data_dependiente` pasan a ser `K` (mismos valores).
- La comparación se hace con las bandas de t.
- La ambigüedad de la continuación queda resuelta.

No cambian el catálogo, los parámetros, las 102 configuraciones, las 134 columnas, las identidades lineales ni D1–D24.

**Pruebas** (`tests/test_tda17_supertrend.py`, 44 casos):
- **Referencia independiente:** traducción literal, serie a serie, de la formulación documentada de TradingView (`ta.rma`, `nz(banda[1])`, rama por valor), con su propio cálculo de TR. Coincide con la implementación de IRIS en 24 caminos con huecos.
- **Calentamiento:** primer ATR en la barra n frente a primera publicación en K. Antes de K, salidas NaN pero estado ya calculado. Las bandas y la dirección evolucionan, y la dirección gira antes de K.
- **Escenarios:** tendencia alcista, tendencia bajista, mercado lateral con cierre constante (finito y bajista desde K, nunca NaN permanente) y cambios de tendencia.
- **Contrato, huecos y tick:** reinicio por contrato, huecos con TR local y precios en la rejilla de tick.
- **N10 como diagnóstico:** regresión del contraejemplo en n10 y n60 (signos discordantes e IRIS = referencia); empate con ATR = 0; salidas no definidas antes de K.
- **Regla de empate por valor:** coincide con TradingView donde una máquina de estados pura daría otro resultado.
- **Causalidad.**
- **Inicialización incorrecta en K:** un caso en que inicializar en K da una dirección distinta en la primera publicación, frente a la referencia, que coincide con IRIS.

**Contador causal.** `u_t` = número de barras τ ≤ t del contrato, posteriores al último reinicio, cuyo par (τ−1, τ) es `CONS`. Vale 0 en la primera barra tras el reinicio, vuelve a 0 en cada roll y no se reinicia en `HI`, `FJ` ni `FJL`. Sólo usa barras hasta t.

**Inicialización.** La semilla de cada EMA/Wilder es la media simple de sus n primeras **entradas recibidas**: en `CONTADOR_CONS`, los n primeros incrementos `CONS`, aunque abarquen más de n barras. El criterio de peso residual < 0,01 se cuenta en **actualizaciones**, no en barras.

**Requisitos U** (en el YAML, junto a cada K):

| Configuración | U | K = U + 1 |
|---|---:|---:|
| RSI 14 / 60 | 77 / 335 | 78 / 336 |
| TSI (25,13) / (100,52) | 125 / 502 | 126 / 503 |
| ADX 14 / 60 | 153 / 669 | 154 / 670 |
| Stoch RSI | 77 (RSI) + 17 barras | 95 |

- Etapa simple: `U = n + k(α)`.
- Etapas encadenadas (TSI, ADX): `U = U₁ + U₂ − 1`. La etapa 2 recibe como primera entrada el valor de la etapa 1 en su primera barra elegible y después una entrada por cada actualización `CONS`.
- Stoch RSI: el RSI sigue el contador; las etapas min/max 14, SMA 3 y SMA 3 cuentan barras observadas desde la primera barra elegible del RSI, incluidos los valores retenidos.
- Una vez elegible, una serie de `CONTADOR_CONS` emite el valor retenido en `HI`/`FJ`/`FJL` y sólo pierde la elegibilidad en un roll.

**Registro.** `TDA17_perdidas_calentamiento.csv` informará, por serie y contrato, de la clase, K teórico, U, primera barra elegible, retraso, número de barras no `CONS` durante el calentamiento, primera salida finita y número de barras elegibles con NaN por regla de división.

**Pruebas sintéticas** (`tests/test_tda17_calentamiento_huecos.py`, prototipos de referencia sobre precios sintéticos):
- sin huecos, la primera barra elegible es exactamente la K declarada en RSI, TSI y ADX (las dos configuraciones de cada uno);
- con huecos en la semilla, tras la semilla, justo antes de K o en ráfaga de 25 barras, se cumplen `E = K + nº de barras no CONS` y `E = min{t : u_t ≥ U}`;
- la semilla usa los n primeros incrementos `CONS`;
- valor retenido sin decaimiento y sin vuelta a NaN;
- causalidad (alterar el futuro no cambia el pasado);
- reinicio en roll;
- Stoch RSI: sólo se retrasa por huecos anteriores a la primera barra elegible del RSI;
- la clase `EXACTA` no se retrasa (ATR 14/60/240 con huecos);
- en `VENTANA_VALIDEZ`, K es sólo una cota inferior;
- (F4) ventanas planas en Donchian, Bollinger, CCI y Choppiness: elegibles en K pero NaN, con la primera salida finita posterior; CMF con volumen nulo (y CLV := 0 con H = L, que sí es finito); Volume ROC con V = 0 en el denominador y en el logaritmo; RSI con precios planos;
- (B) KAMA: con precios variables, elegibilidad = activación = primera salida finita = K_min. Con precios constantes, la misma elegibilidad (sólo esqueleto) pero sin activación, y la activación llega con el primer incremento no nulo. Tras activarse, un tramo plano da NaN con el estado retenido. Los huecos retrasan la elegibilidad.

## 7. Control G2-2: t de Student reproducible (corrección C4)

- **Grados de libertad fijos:** `df = 2,55422176662895`. Es la raíz de `κ_rec(df) = 11,491079308573783`, donde `κ_rec` es la curtosis en exceso recortada **teórica** de la t, con la misma definición que TDA-07 (recorte en los cuantiles 0,0005 y 0,9995, momentos poblacionales del tramo central). Integración con `scipy.integrate.quad` (`epsrel` 1e-13) y raíz con `brentq` en [1, 1000] (`xtol` 1e-12). La calibración sólo usa el objetivo publicado en TDA-07, artefacto congelado en el manifiesto; ningún resultado de TDA-17 interviene. Un test la recalcula (tolerancia 1e-8).
- **Escala:** `r = σ·√((df−2)/df)·T`, con `T ~ t(df)` y σ = 4,1916e-4. Como df > 2, la varianza teórica es σ².
- **Algoritmo y semillas:** PCG64 con `SeedSequence(semilla).spawn(3)` (flujo 0 innovaciones, 1 puente, 2 volumen); semillas 301 y 302; filas en orden estable (contract, timestamp).
- **Convención de primera barra (corrección F3).** Es una aclaración: no cambia df, semillas, tolerancias ni el resto del nulo.
  - **Biyección:** fila g ↔ innovación r_g ↔ bloque de 30 normales Z_g ↔ índice de volumen I_g. Cada una se extrae en una sola llamada de tamaño n_filas; ninguna se descarta ni se reutiliza.
  - **Primera fila de un contrato c:** ancla A_c = open de su primera barra en la entrada (ya en la rejilla); `x_prev = ln A_c`; **sí usa su innovación**: `x = ln A_c + r_g`. `O = A_c` y `C = q(exp x)`, con `q(p) = 0,25·round(p/0,25)` (mitad al par). El puente va de `b_0 = ln A_c` a `b_30 = x`.
  - **Filas siguientes:** `x_g = x_{g−1} + r_g` sobre el camino **latente** (sin redondear), `O_g = C_{g−1}` redondeado, y el puente va de `x_{g−1}` a `x_g`.
  - **Cambio de contrato:** reinicio en el ancla del nuevo contrato.
  - **Extremos:** `H = max(O, C, q(exp max b))` y `L = min(O, C, q(exp min b))`, lo que garantiza las invariantes OHLC y la rejilla de tick.
  - **Pruebas** (`tests/test_tda17_g2_2_generador.py`, esqueleto sintético de 3 contratos, uno de una sola fila): primera barra, segunda barra, cambio de contrato, uso único de cada innovación (comparada con un flujo fresco de n+1 extracciones), rejilla de tick, redondeo mitad al par e invariantes OHLC.
- **Verificación sobre las innovaciones (antes del redondeo):**
  - curtosis recortada dentro de ±0,85 del objetivo;
  - RIC relativo dentro de ±0,006 de 1.
  Las tolerancias son al menos 5 desviaciones típicas, estimadas con 100 muestras sintéticas de 1.914.530 extracciones (DE 0,162 y 0,0010). Con las semillas del protocolo los valores son 11,644 y 11,296 (RIC 0,99991 y 0,99954): ambas pasan, y un test lo comprueba.
- **Fallos:**
  - Si la calibración no converge o df ≤ 2: G2-2 `NO_EVALUABLE`. No ocurre: converge con df = 2,554.
  - Si una semilla no pasa la verificación: es un error de implementación y G2-2 queda bloqueado hasta corregirlo, sin cambiar df, semillas ni tolerancias y sin recalibrar con datos.
  - Los momentos tras el redondeo a tick son sólo descriptivos.

## 8. Sensibilidad S6: lista cerrada de fechas (corrección C3)

S6 excluye todas las filas de **8 `trading_date`**:
- **2020-03-09, 2020-03-16, 2020-03-17 y 2020-03-18**: TDA-02 §4.1(b), "concentrados en 2020-03-09, 16, 17 y 18";
- **2020-10-19, 2020-10-20, 2020-10-21 y 2020-10-22**: TDA-02 §4.1(a), semana anómala del 18 al 22 de octubre. Son las cuatro jornadas con el patrón repetido de dos huecos, con 1.292 de 1.365 barras cada una en `TDA02_dias_incompletos.csv`.

La regla es "exactamente las fechas nombradas en TDA-02 §4.1". Ambas fuentes están en el manifiesto con su SHA-256. 2020-10-23 (cobertura 50,92 %) queda fuera porque no pertenece al patrón de §4.1(a). La lista es cerrada y se fija antes de ver ningún resultado de TDA-17. Un test la reconstruye a partir de los dos artefactos.

## 9. Integridad y procedimiento de congelamiento (correcciones C1 y C5)

**Verificación oficial (siempre estricta):** `python -m ohlcv_dataroad.ingest.tda17_protocol_freeze verificar`. Falla si falta o difiere cualquiera de los 9 ficheros del protocolo, de las 5 entradas obligatorias o de los 188 artefactos históricos. El manifiesto declara esos recuentos y la verificación los comprueba.

**Validación de rutas antes de abrir nada (corrección A).** Antes de calcular ningún hash, `verificar`, `manifiesto` y `congelar` validan **todas** las rutas del manifiesto, sin leer su contenido. Rechazan:
- rutas que no sean relativas POSIX normalizadas: absolutas, con `..`, con barra invertida o con unidad;
- rutas en `data/raw`;
- nombres del holdout, según `holdout_files` de `configs/mnq_snapshot.yaml`, o rutas que contengan "holdout";
- rutas que, una vez resueltas (enlaces simbólicos incluidos), salen del repositorio o caen en una ubicación prohibida.

Con una sola ruta prohibida, la verificación termina sin calcular ningún hash. Un espía sobre las lecturas y sobre la función de hash demuestra que el fichero ficticio del holdout nunca se intenta abrir. Lo comprueba en 8 casos: holdout dentro y fuera de `data/raw`, `data/raw` de investigación, `..`, ruta absoluta, barra invertida y enlace real (que se omite si el sistema no permite crearlo), más el enlace simulado hacia el holdout y hacia fuera del repositorio. Un control negativo, con la validación desactivada, confirma que el espía sí detectaría la apertura. La verificación normal sigue calculando el hash de los 9 + 5 + 188 ficheros.

**Modos de los tests** (variable `TDA17_INTEGRIDAD`):

| Modo | Ficheros ausentes | Uso |
|---|---|---|
| `portable` (por defecto) | `SKIP` con el texto "MODO PORTABLE (no válido para congelar)" | clones sin `data/interim` |
| `estricta` | **FALLO**; además ejecuta la verificación oficial | congelamiento y autorización |

**Congelamiento:** `python -m ohlcv_dataroad.ingest.tda17_protocol_freeze congelar --confirmar`.

1. **Precondiciones:** YAML, informe y manifiesto en `PENDIENTE_AUDITORIA_FINAL`; verificación estricta conforme; HEAD desciende de `base_git`; sin ficheros versionados modificados. Si falla alguna, no se escribe nada.
2. **Cálculo en memoria:** sustituye sólo dos líneas (`  estado: PENDIENTE_AUDITORIA_FINAL` del YAML y `**Estado:** \`PENDIENTE_AUDITORIA_FINAL\`.` de este documento), comprueba que ambas quedan en `CONGELADO` y construye el manifiesto nuevo con los SHA-256 recalculados de esos dos ficheros y un registro `congelamiento`: fecha UTC, HEAD, transiciones y las huellas auditadas en PENDIENTE de **los 9 ficheros del protocolo**. Los hashes de entradas y artefactos no se recalculan nunca.
3. **Escritura atómica:** `os.replace` por fichero, seguido de una verificación estricta. Ante cualquier fallo se restauran los bytes originales de los tres ficheros.
4. **Comprobación permanente (corrección F1):** en `CONGELADO`, deshacer las transiciones en YAML e informe debe reproducir exactamente sus huellas auditadas, y los otros 7 ficheros (tests y herramienta) deben coincidir byte a byte (tras normalizar el fin de línea) con las suyas. Modificar cualquiera de los 9 y recalcular su hash ordinario en el manifiesto da NO CONFORME.
5. **Referencia externa: el commit Git.** El registro vive dentro del manifiesto, así que una reescritura completa y coherente del manifiesto, registro incluido, sólo puede detectarse fuera de él. Por eso el estado congelado se ancla en el commit que lo contiene: `verificar --referencia-git <commit>` compara los 9 ficheros del protocolo y el manifiesto con los de ese commit. Los SHA-256 auditados publicados en la revisión sirven como segunda referencia.

**Regeneración del manifiesto en PENDIENTE (corrección F2):** el manifiesto candidato se verifica en memoria antes de escribirlo. Si falla antes, durante o después de la escritura, el manifiesto queda idéntico byte a byte al anterior.

Los tests no se modifican al congelar: ya aceptan los dos estados, pero `CONGELADO` sólo con el registro y la reversibilidad exacta. Así, el código de verificación auditado es el mismo que verificará el estado congelado, y su hash no cambia. El procedimiento se prueba de extremo a extremo sobre una copia temporal con entradas y artefactos sintéticos: congelamiento correcto, confirmación obligatoria, rechazo con un fichero ausente, detección de un artefacto modificado, restauración ante un fallo posterior a la escritura, detección de una edición posterior en **cada uno** de los 9 ficheros con el hash rehecho, detección de una reescritura completa del manifiesto contra un commit Git sintético, y regeneración fallida (antes y después de escribir) con el manifiesto intacto.

## 10. Qué falta para autorizar la ejecución

1. Autorización definitiva tras la corrección de Supertrend A11.
2. `python -m ohlcv_dataroad.ingest.tda17_protocol_freeze congelar --confirmar` y después `TDA17_INTEGRIDAD=estricta python -m pytest tests/test_tda17_protocol.py tests/test_tda17_calentamiento_huecos.py tests/test_tda17_diseno_v4a_sintetico.py tests/test_tda17_diseno_v4b_g23b_viabilidad.py tests/test_tda17_g2_2_generador.py tests/test_tda17_supertrend.py`. Tras el commit de congelamiento: `verificar --referencia-git <commit>`.
3. Sólo entonces, implementar TDA-17 contra este protocolo.
