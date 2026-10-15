# TDA-14 — Estabilidad temporal y cambio estructural (TH29, TH30, TH31)

Generado automaticamente: 2026-10-15T00:12:02. Orden: `python -m ohlcv_dataroad.ingest.run_tda14 analizar --config configs/mnq_snapshot.yaml --config-tda14 C:\Users\heguu\Desktop\ohlcv_dataroad_TDAs\TDA14_analisis_MNQ\config\tda14_analisis_MNQ.yaml --salida C:\Users\heguu\Desktop\ohlcv_dataroad_TDAs\TDA14_analisis_MNQ\ejecucion_1`. Duracion: 172.5 s.

## 1. Estado y alcance

- Estado de las decisiones: **COMPATIBLE_CON_CONTROLES_EVALUADOS**.
- Protocolo: **CONGELADO**. Configuracion `C:\Users\heguu\Desktop\ohlcv_dataroad_TDAs\TDA14_analisis_MNQ\config\tda14_analisis_MNQ.yaml` (SHA-256 `a6e92b7749ea518b…`).
- Analisis exploratorio y retrospectivo. Un cambio estadistico no identifica un regimen economico ni una causa.
- Umbrales calibrados y validados en G2 (secciones 9 y 10): los veredictos y las fechas se publican solo en lo que esa validacion permite; una familia sin potencia validada queda `NO_EVALUABLE` y un tipo no validado se informa como no identificado.
- El volumen (familia ACTIVIDAD) es descriptivo, no calibrado y sin veredicto oficial.
- Hold-out: no abierto (guardias de TDA-04 antes y despues de leer).

## 2. Poblacion

- Jornadas en el panel: 1.414 (usadas en ventanas: 1.414; sobrantes al final: 0).
- Año de la jornada igual al año calendario NY de todas las barras: si (barras con año distinto: 0).
- Ventanas: 65 (63 jornadas, paso 21); con cobertura suficiente: 60.

## 3. Reconciliacion con TDA-05..TDA-13

Clases predeclaradas: `REPRODUCIBLE` (misma definicion; igual hasta el redondeo), `COMPARABLE_CON_DIFERENCIA_EXPLICADA` (diferencia conocida; `_VERIFICADA` si se comprueba exactamente), `NO_COMPARABLE` (sin homologo), `DISCREPANCIA_NO_EXPLICADA` (deberia reproducirse y no lo hace), `HEREDADO_AUSENTE`.

| etapa | clase | filas |
|---|---|---|
| TDA-05 | REPRODUCIBLE | 12 |
| TDA-06 | NO_COMPARABLE | 2 |
| TDA-07 | REPRODUCIBLE | 36 |
| TDA-08 | REPRODUCIBLE | 6 |
| TDA-09 | COMPARABLE_CON_DIFERENCIA_EXPLICADA | 28 |
| TDA-09 | REPRODUCIBLE | 24 |
| TDA-10 | NO_COMPARABLE | 1 |
| TDA-11 | NO_COMPARABLE | 1 |
| TDA-12 | REPRODUCIBLE | 18 |
| TDA-13 | NO_COMPARABLE | 1 |
| TDA-14 | NO_COMPARABLE | 1 |
| TDA-14 vs TDA-05 | REPRODUCIBLE | 5 |
| TDA-14 vs TDA-07 | COMPARABLE_CON_DIFERENCIA_EXPLICADA_VERIFICADA | 15 |
| TDA-14 vs TDA-08 | COMPARABLE_CON_DIFERENCIA_EXPLICADA | 5 |
| TDA-14 vs TDA-12 | COMPARABLE_CON_DIFERENCIA_EXPLICADA_VERIFICADA | 10 |

Discrepancias no explicadas: **0**.
Detalle completo: `TDA14_reconciliacion.csv`.

## 4. Tabla anual (años completos 2020-2024)

Valor puntual e IC percentil 95 % por bootstrap circular de bloques de jornadas dentro de cada año, para bloques de 1, 5 y 20 jornadas (`TDA14_tabla_anual.csv`). Valores puntuales:

| estadistico | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|
| correlacion_perfil | 0.9725 | 0.9945 | 0.9931 | 0.9914 | 0.9904 |
| curtosis_total | 47.3689 | 27.0226 | 133.7625 | 56.7952 | 57.1827 |
| log_amplitud_perfil | -1.0832 | -0.8467 | -0.7300 | -0.6069 | -0.7275 |
| log_fraccion_cero | -3.0550 | -2.9860 | -3.4577 | -3.0101 | -3.1889 |
| log_q01_en_sigma | 1.1010 | 1.1136 | 1.0421 | 1.0824 | 1.1054 |
| log_q99_en_sigma | 1.0802 | 1.0742 | 1.0428 | 1.0532 | 1.0383 |
| log_sigma | -7.4726 | -8.1015 | -7.5590 | -8.0938 | -8.1749 |
| log_volumen_medio | 6.1577 | 6.4264 | 6.8666 | 6.5779 | 6.8045 |
| media_en_sigma | 0.0038 | 0.0020 | -0.0021 | 0.0040 | 0.0013 |
| persistencia_acf_abs | 0.3654 | 0.3006 | 0.2467 | 0.2892 | 0.2844 |
| rho1 | 0.0071 | 0.0117 | 0.0098 | -0.0087 | 0.0075 |

Celdas no evaluables (todas las longitudes de bloque): 0. Evaluabilidad de la curtosis (semiamplitud relativa <= 25 % en al menos 4 de 5 años):

| bloque_jornadas | evaluable | anios_que_cumplen | semiamplitud_relativa |
|---|---|---|---|
| 1 | no | 0 | [0.385223, 0.282049, 0.683956, 0.569502, 0.57067] |
| 5 | no | 0 | [0.391144, 0.325081, 0.697219, 0.494427, 0.539228] |
| 20 | no | 0 | [0.502424, 0.405123, 0.722796, 0.50065, 0.635937] |

Tabla año x segmento: `TDA14_tabla_anio_segmento.csv` (1155 filas; no evaluables: 90). Ventanas rodantes: `TDA14_tabla_ventanas.csv`.

## 5. TH29 — estabilidad por familia

| familia | veredicto | anotacion | soporte | veredicto_doble_materialidad | validacion |
|---|---|---|---|---|---|
| NIVEL | ESTABLE | SIN_EVIDENCIA_DE_CAMBIO | SIN_INESTABILIDAD | ESTABLE | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| ESCALA | NO_EVALUABLE | POTENCIA_INSUFICIENTE | SIN_INESTABILIDAD | ESTABLE | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| FORMA | ESTABLE | VARIACION_ESTADISTICA_NO_MATERIAL | SIN_INESTABILIDAD | ESTABLE | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| DEPENDENCIA_LINEAL | ESTABLE | VARIACION_ESTADISTICA_NO_MATERIAL | SIN_INESTABILIDAD | ESTABLE | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| DEPENDENCIA_MAGNITUD | NO_EVALUABLE | POTENCIA_INSUFICIENTE | SIN_INESTABILIDAD | ESTABLE | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| RELOJ | INESTABLE | TIPO_NO_IDENTIFICABLE | SIN_PATRON_EN_LA_MUESTRA | ESTABLE | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| DISCRECION | NO_EVALUABLE | POTENCIA_INSUFICIENTE | SIN_INESTABILIDAD | ESTABLE | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| ACTIVIDAD | SIN_VEREDICTO_OFICIAL | DESCRIPTIVO_NO_CALIBRADO |  |  | DESCRIPTIVO_NO_CALIBRADO |

Detalle por estadistico: `TDA14_th29_veredictos_estadistico.csv`.

## 6. TH31 — fechas candidatas y eventos institucionales predeclarados

_Ninguna fecha candidata publicada (sin calibracion, no validadas o sin detecciones)._

Eventos (solo distancia en jornadas; no se atribuye causalidad):

| evento | fecha_negociacion | tipo | en_muestra | distancia_jornadas_a_intervalo | dentro_de_intervalo | estado_fechas |
|---|---|---|---|---|---|---|
| LIMITES_PRECIO_REGLA_589 | 2020-10-12 | reglas de negociacion | si | - | no | NO_PUBLICADAS (no validadas o sin calibracion) |
| HORA_LIQUIDACION_DIARIA | 2020-10-26 | liquidacion | si | - | no | NO_PUBLICADAS (no validadas o sin calibracion) |
| ELIMINACION_PAUSA_1515_1530_CT | 2021-06-28 | horario de negociacion | si | - | no | NO_PUBLICADAS (no validadas o sin calibracion) |
| RANGOS_NO_REVISABLES | 2021-11-01 | rangos no revisables (anulacion de operaciones erroneas) | si | - | no | NO_PUBLICADAS (no validadas o sin calibracion) |

Cobertura de barras en la ventana de reloj predeclarada (hipotesis comprobable, no causal):

| evento | ventana_reloj_ny | anio | periodo | jornadas | jornadas_con_barras | fraccion_jornadas_con_barras | barras |
|---|---|---|---|---|---|---|---|
| ELIMINACION_PAUSA_1515_1530_CT | 16:15-16:30 | 2.020 | antes_evento | 259 | 246 | 0.9498 | 324 |
| ELIMINACION_PAUSA_1515_1530_CT | 16:15-16:30 | 2.021 | antes_evento | 125 | 119 | 0.9520 | 122 |
| ELIMINACION_PAUSA_1515_1530_CT | 16:15-16:30 | 2.021 | desde_evento | 134 | 128 | 0.9552 | 1.919 |
| ELIMINACION_PAUSA_1515_1530_CT | 16:15-16:30 | 2.022 | desde_evento | 258 | 246 | 0.9535 | 3.690 |
| ELIMINACION_PAUSA_1515_1530_CT | 16:15-16:30 | 2.023 | desde_evento | 258 | 243 | 0.9419 | 3.644 |
| ELIMINACION_PAUSA_1515_1530_CT | 16:15-16:30 | 2.024 | desde_evento | 259 | 246 | 0.9498 | 3.679 |
| ELIMINACION_PAUSA_1515_1530_CT | 16:15-16:30 | 2.025 | desde_evento | 121 | 114 | 0.9421 | 1.710 |

## 7. TH30 — log-precio frente a retornos (dentro de cada contrato)

| jornadas_subventana | conclusion | factor | desplazamiento_mediano_ln_c | desplazamiento_mediano_r | nota |
|---|---|---|---|---|---|
| 21 | MOMENTOS_DE_R_RELATIVAMENTE_MAS_ESTABLES | 383.0814 | 3.4789 | 0.0091 | no demuestra estacionariedad; compara estabilidad relativa dentro de cada contrato |
| 10 | MOMENTOS_DE_R_RELATIVAMENTE_MAS_ESTABLES | 329.1125 | 6.7009 | 0.0204 | no demuestra estacionariedad; compara estabilidad relativa dentro de cada contrato |

La conclusion se limita a la estabilidad relativa de los momentos dentro de cada contrato; no demuestra estacionariedad.

## 8. STOP-14

- Alcance: **NO_ACTIVADO**; definitivo: si.
- Familias persistentes confirmadas: []; pendientes de comprobaciones: []; contradictorias: []; cambios en el borde: []; episodios: [].
- Comprobaciones anual y de sensibilidad a rolls y huecos: **EJECUTADAS (STOP14_REGLAS_A_B_v1)**. Se CALCULAN con las reglas A y B (`tda14_stop14_rules`); la configuracion no puede fijarlas a mano. Evidencia insuficiente o mixta = `null` = STOP-14 PENDIENTE; nunca cuenta como superada.

## 9. Controles G2

Criterios de aceptacion fijados antes de validar (limite superior de Clopper-Pearson; confianza repartida por Bonferroni):

| control | n_minimo | confianza_bonferroni | tasa_a_demostrar | max_falsas_alarmas_aceptables | prob_aceptar_si_tasa_real_2_5pct | prob_aceptar_si_tasa_real_5pct |
|---|---|---|---|---|---|---|
| NEGATIVO_ESTACIONARIO | 1.000 | 0.9900 | 0.0500 | 34 | 0.9680 | 0.0093 |
| NEGATIVO_ESTACIONAL | 1.000 | 0.9900 | 0.0500 | 34 | 0.9680 | 0.0093 |
| NEGATIVO_PERSISTENCIA_BAJA | 1.000 | 0.9900 | 0.0500 | 34 | 0.9680 | 0.0093 |
| NEGATIVO_PERSISTENCIA_ALTA | 1.000 | 0.9900 | 0.0500 | 34 | 0.9680 | 0.0093 |
| NEGATIVO_PARCIAL_SOLO_ESCALA | 1.000 | 0.9900 | 0.0500 | 34 | 0.9680 | 0.0093 |

| control | decision | eventos | n | tasa | limite_confianza | criterio | estado |
|---|---|---|---|---|---|---|---|
| NEGATIVO_ESTACIONARIO | GLOBAL | 10 | 1.000 | 0.0100 | 0.0200 | 0.0500 | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| NEGATIVO_ESTACIONARIO | FECHAS | 4 | 1.000 | 0.0040 | 0.0116 | 0.0500 | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| NEGATIVO_ESTACIONARIO | STOP14 | 0 | 1.000 | 0.0000 | 0.0046 | 0.0500 | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| NEGATIVO_ESTACIONARIO | FAMILIA_NIVEL | 1 | 1.000 | 0.0010 | 0.0066 | 0.0500 | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| NEGATIVO_ESTACIONARIO | NO_EVALUABLE_NIVEL | 0 | 1.000 | 0.0000 | 0.0046 | 1.0000 | INFORMATIVO |
| NEGATIVO_ESTACIONARIO | FAMILIA_ESCALA | 0 | 1.000 | 0.0000 | 0.0046 | 0.0500 | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| NEGATIVO_ESTACIONARIO | NO_EVALUABLE_ESCALA | 0 | 1.000 | 0.0000 | 0.0046 | 1.0000 | INFORMATIVO |
| NEGATIVO_ESTACIONARIO | FAMILIA_FORMA | 5 | 1.000 | 0.0050 | 0.0131 | 0.0500 | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| NEGATIVO_ESTACIONARIO | NO_EVALUABLE_FORMA | 0 | 1.000 | 0.0000 | 0.0046 | 1.0000 | INFORMATIVO |
| NEGATIVO_ESTACIONARIO | FAMILIA_DEPENDENCIA_LINEAL | 0 | 1.000 | 0.0000 | 0.0046 | 0.0500 | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| NEGATIVO_ESTACIONARIO | NO_EVALUABLE_DEPENDENCIA_LINEAL | 0 | 1.000 | 0.0000 | 0.0046 | 1.0000 | INFORMATIVO |
| NEGATIVO_ESTACIONARIO | FAMILIA_DEPENDENCIA_MAGNITUD | 0 | 1.000 | 0.0000 | 0.0046 | 0.0500 | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| NEGATIVO_ESTACIONARIO | NO_EVALUABLE_DEPENDENCIA_MAGNITUD | 0 | 1.000 | 0.0000 | 0.0046 | 1.0000 | INFORMATIVO |
| NEGATIVO_ESTACIONARIO | FAMILIA_RELOJ | 0 | 1.000 | 0.0000 | 0.0046 | 0.0500 | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| NEGATIVO_ESTACIONARIO | NO_EVALUABLE_RELOJ | 0 | 1.000 | 0.0000 | 0.0046 | 1.0000 | INFORMATIVO |
| NEGATIVO_ESTACIONARIO | FAMILIA_DISCRECION | 0 | 1.000 | 0.0000 | 0.0046 | 0.0500 | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| NEGATIVO_ESTACIONARIO | NO_EVALUABLE_DISCRECION | 0 | 1.000 | 0.0000 | 0.0046 | 1.0000 | INFORMATIVO |
| NEGATIVO_ESTACIONAL | GLOBAL | 10 | 1.000 | 0.0100 | 0.0200 | 0.0500 | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| NEGATIVO_ESTACIONAL | FECHAS | 3 | 1.000 | 0.0030 | 0.0100 | 0.0500 | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| NEGATIVO_ESTACIONAL | STOP14 | 0 | 1.000 | 0.0000 | 0.0046 | 0.0500 | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| NEGATIVO_ESTACIONAL | FAMILIA_NIVEL | 0 | 1.000 | 0.0000 | 0.0046 | 0.0500 | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| NEGATIVO_ESTACIONAL | NO_EVALUABLE_NIVEL | 0 | 1.000 | 0.0000 | 0.0046 | 1.0000 | INFORMATIVO |
| NEGATIVO_ESTACIONAL | FAMILIA_ESCALA | 0 | 1.000 | 0.0000 | 0.0046 | 0.0500 | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| NEGATIVO_ESTACIONAL | NO_EVALUABLE_ESCALA | 0 | 1.000 | 0.0000 | 0.0046 | 1.0000 | INFORMATIVO |
| NEGATIVO_ESTACIONAL | FAMILIA_FORMA | 7 | 1.000 | 0.0070 | 0.0159 | 0.0500 | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| NEGATIVO_ESTACIONAL | NO_EVALUABLE_FORMA | 0 | 1.000 | 0.0000 | 0.0046 | 1.0000 | INFORMATIVO |
| NEGATIVO_ESTACIONAL | FAMILIA_DEPENDENCIA_LINEAL | 0 | 1.000 | 0.0000 | 0.0046 | 0.0500 | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| NEGATIVO_ESTACIONAL | NO_EVALUABLE_DEPENDENCIA_LINEAL | 0 | 1.000 | 0.0000 | 0.0046 | 1.0000 | INFORMATIVO |
| NEGATIVO_ESTACIONAL | FAMILIA_DEPENDENCIA_MAGNITUD | 0 | 1.000 | 0.0000 | 0.0046 | 0.0500 | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| NEGATIVO_ESTACIONAL | NO_EVALUABLE_DEPENDENCIA_MAGNITUD | 0 | 1.000 | 0.0000 | 0.0046 | 1.0000 | INFORMATIVO |
| NEGATIVO_ESTACIONAL | FAMILIA_RELOJ | 0 | 1.000 | 0.0000 | 0.0046 | 0.0500 | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| NEGATIVO_ESTACIONAL | NO_EVALUABLE_RELOJ | 0 | 1.000 | 0.0000 | 0.0046 | 1.0000 | INFORMATIVO |
| NEGATIVO_ESTACIONAL | FAMILIA_DISCRECION | 0 | 1.000 | 0.0000 | 0.0046 | 0.0500 | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| NEGATIVO_ESTACIONAL | NO_EVALUABLE_DISCRECION | 0 | 1.000 | 0.0000 | 0.0046 | 1.0000 | INFORMATIVO |
| NEGATIVO_PERSISTENCIA_BAJA | GLOBAL | 16 | 1.000 | 0.0160 | 0.0279 | 0.0500 | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| NEGATIVO_PERSISTENCIA_BAJA | FECHAS | 0 | 1.000 | 0.0000 | 0.0046 | 0.0500 | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| NEGATIVO_PERSISTENCIA_BAJA | STOP14 | 0 | 1.000 | 0.0000 | 0.0046 | 0.0500 | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| NEGATIVO_PERSISTENCIA_BAJA | FAMILIA_NIVEL | 1 | 1.000 | 0.0010 | 0.0066 | 0.0500 | COMPATIBLE_CON_CONTROLES_EVALUADOS |
| NEGATIVO_PERSISTENCIA_BAJA | NO_EVALUABLE_NIVEL | 0 | 1.000 | 0.0000 | 0.0046 | 1.0000 | INFORMATIVO |
| NEGATIVO_PERSISTENCIA_BAJA | FAMILIA_ESCALA | 0 | 1.000 | 0.0000 | 0.0046 | 0.0500 | COMPATIBLE_CON_CONTROLES_EVALUADOS |

_(54 filas mas en el CSV)_

La aceptacion significa compatibilidad con los controles evaluados, no una garantia para cualquier mundo real.

## 10. Trazabilidad

| parametro | valor | origen |
|---|---|---|
| configuracion_tda14 | C:\Users\heguu\Desktop\ohlcv_dataroad_TDAs\TDA14_analisis_MNQ\config\tda14_analisis_MNQ.yaml | archivo |
| configuracion_tda14_sha256 | a6e92b7749ea518b1341d8e8bae3c05b35646ae3b1da76da260ca0ee12d30a09 | archivo |
| protocolo_estado | CONGELADO | configuracion |
| modulo_tda14_sha256 | 41330b4fdacd384dac0cc7b1e20f17c55b03862fc295f2a40fededfcf15bfd56 | codigo |
| modulo_pipeline_sha256 | 9a580735a3ae463791edcacd1aa047777a70d980d23e585c5dbabcd6d613f630 | codigo |
| detector_sha256 | c074ddb428bf7b0e6ee21e1857faf903810bf261a47e228646229f3a509b70d6 | configuracion |
| fuente_datos | TDA-04 | carga |
| jornadas_panel | 1.414 | carga |
| anio_jornada_coincide_con_anio_ny | si | carga |
| objetivo_calibracion_g2 | 0.0250 | codigo |
| tasa_a_demostrar_g2 | 0.0500 | codigo |
| confianza_g2 | 0.9500 | codigo |
| capacidad_minima_tipo_provisional | 0.5000 | codigo |
| bloques_bootstrap | [1, 5, 20] | codigo |
| n_boot | 300 | configuracion |
| semilla | 0 | configuracion |
| umbrales_sha256 | 9e7a891127fecb6e166f32b2169fd8b4075f1e87b93b2634300510b906a8fa5e | calibracion |
| alpha | 0.0050 | calibracion |
| k_star | 2 | calibracion |
| margen_tipo | 0.0000 | calibracion |
| calibracion.abstencion_tipo_calibracion | - | calibracion |
| calibracion.capacidad_tipo_calibracion | - | calibracion |
| calibracion.confusion_tipo_calibracion | - | calibracion |
| calibracion.especificaciones_calibracion | 028c16f9df7bf03e8da4e3c684a51a53af983ab041ff2b9c4f12868a3516b0f3,0fd41255ae8c35bb18100c1301d8247256814fe5b534e02e91a7683697a97c00,384944fc798772b55ca7886c1859767916bb8f2342a0fccfb33fcce7522ad1cb,e1ceaf4dd90654b09057f18e3c01416aba6c9dc1761e6c49ad1ae39e87c59f32 | calibracion |
| calibracion.manifiesto_sha256 | 350b4eac372c3532df81305e110cd245f43c99b7b5d9d11e6b48f51808ca9522 | calibracion |
| calibracion.n_mundos_calibracion | 4.000 | calibracion |
| calibracion.n_mundos_tipo | 400 | calibracion |
| calibracion.objetivo_calibracion | 0.0250 | calibracion |
| calibracion.tasa_global_calibracion | 0.0170 | calibracion |
| manifiesto_validacion_sha256 | 350b4eac372c3532df81305e110cd245f43c99b7b5d9d11e6b48f51808ca9522 | validacion |
| stop14_comprobaciones | EJECUTADAS (STOP14_REGLAS_A_B_v1) | ejecucion |
| stop14_jornadas_excluidas_sensibilidad | {"S1_rolls": 84, "S2_huecos": 10, "S3_rejilla_alineada": 57} | ejecucion |

Tabla completa (umbrales por estadistico, detector, materialidades): `TDA14_trazabilidad.csv`.

## 11. Tiempos por etapa

| etapa | segundos |
|---|---|
| detector | 1.5679 |
| tabla_anual | 24.2288 |
| tabla_anio_segmento | 58.1864 |
| tabla_ventanas | 77.5942 |
| reconciliacion | 2.7455 |
| stop14_reglas_A_B | 5.2552 |

## 12. Limitaciones

- Las materialidades y la capacidad minima de tipos (50 %) son convenciones documentadas, no leyes estadisticas.
- Con 100 replicas por tipo o nivel de potencia la incertidumbre es de unos +-8 puntos.
- El error comun por lotes del detector ignora la persistencia; su escala absoluta no se interpreta (lo absorben los umbrales calibrados).
- Familias sin controles de potencia en el protocolo: DEPENDENCIA_MAGNITUD. Una familia sin potencia validada en G2 (sin controles, o con controles no superados) nunca se declara ESTABLE: queda `NO_EVALUABLE` (`POTENCIA_INSUFICIENTE`).

## 13. Estado de la etapa

**TDA-14 = `CERRADA TRAS AUDITORÍA`** (auditoría de resultados favorable; resultados exploratorios y retrospectivos, no confirmados en el holdout).

Auditoría de los resultados de TDA-14 (15 de octubre de 2026): FAVORABLE. Se aprueban G2 reproducible con validación parcial; NIVEL, FORMA y DEPENDENCIA_LINEAL estables bajo los criterios validados; RELOJ inestable sin tipo identificado; ESCALA, DISCRECION y DEPENDENCIA_MAGNITUD no evaluables por falta de potencia validada; ninguna fecha de cambio conjunto (TH31); retornos relativamente más estables que precios sin demostrar estacionariedad (TH30); `STOP-14` no activado según el protocolo congelado; reconciliación con TDA-05 a TDA-13 sin discrepancias inexplicadas; holdout cerrado. El cierre no modifica metodología, parámetros, umbrales ni veredictos, y no autoriza TDA-15.

**Validacion G2.**

- Validación PARCIAL y reproducible: dos calibraciones independientes de 11.000 mundos sintéticos (4.000 + 400 de calibración, 6.600 de validación) idénticas byte a byte. Umbrales `9e7a891127fecb6e166f32b2169fd8b4075f1e87b93b2634300510b906a8fa5e`; manifiesto `350b4eac372c3532df81305e110cd245f43c99b7b5d9d11e6b48f51808ca9522`.
- Superados: error global de falsas alarmas (peor control: 28 de 1.000, máximo aceptable 34), fechas, alcance de STOP-14, falsas alarmas de las 7 familias y control parcial; potencia de NIVEL, FORMA, DEPENDENCIA_LINEAL y RELOJ.
- No superados: potencia de ESCALA (0/100 a la materialidad, 1/100 al doble) y de DISCRECION (0/100 y 0/100); DEPENDENCIA_MAGNITUD sin controles de potencia por diseño del protocolo; identificación de tipos (1 detección entre 400 controles, todos de escala).

**Conclusiones con respaldo estadistico.**

- NIVEL, FORMA y DEPENDENCIA_LINEAL: ESTABLES bajo los criterios validados (FORMA y DEPENDENCIA_LINEAL con variación estadística por debajo de la materialidad).
- RELOJ: INESTABLE, sin tipo identificado: cambio material de la amplitud del perfil intradía, sin patrón y no robusto al doble de materialidad.
- TH31: ninguna fecha de cambio conjunto publicada; ninguna frontera fue marcada por k* = 2 familias (procedimiento de fechas validado).
- TH30: dentro de cada contrato, los momentos de los retornos son relativamente más estables que los del log-precio; no se demuestra estacionariedad.
- STOP-14: NO_ACTIVADO, decisión definitiva según el protocolo congelado y las reglas A y B ejecutadas automáticamente (`STOP14_REGLAS_A_B_v1`).
- Reconciliación con TDA-05 a TDA-13: 0 discrepancias no explicadas (101 reproducibles, 25 diferencias explicadas y verificadas, 33 explicadas, 6 no comparables).

**Resultados inconclusos.**

- ESCALA, DISCRECION y DEPENDENCIA_MAGNITUD: NO_EVALUABLE por falta de potencia validada; no puede afirmarse ni su estabilidad ni un cambio estructural.
- Tipo del cambio de RELOJ (salto, tendencia, episodio o borde): la identificación de tipos no está validada.
- Curtosis anual: no evaluable (semiamplitud relativa > 25 % en todos los años).
- Relación con los eventos institucionales: sin fechas publicadas no hay comparación; no se atribuye causalidad.

**Observaciones que deben conservarse.**

- Inestabilidad de RELOJ: `log_amplitud_perfil` recorre 0,32 (materialidad 0,22); la amplitud del perfil intradía crece de 2020 a 2023 y baja algo en 2024-2025; `correlacion_perfil` sin cambio material. S2 y S3 mantienen la inestabilidad. No es candidata a STOP-14 porque no supera el doble de la materialidad.
- Variación anual de la volatilidad: la sigma por minuto de 2020 y 2022 es casi el doble que la de 2021, 2023 y 2024, con intervalos anuales disjuntos; la regla A registra cambio anual en ESCALA y DISCRECION. NO se afirma una ruptura estructural: sin potencia validada, TDA-14 no distingue un cambio estructural de la deriva natural de una volatilidad muy persistente.
- Tipos de cambio: no identificables. La columna `modelo` de `TDA14_th29_veredictos_estadistico.csv` describe el mejor ajuste y no es un tipo validado; los veredictos por estadístico de las familias sin potencia (`ESTABLE`) no son conclusiones: manda el veredicto de la familia (`NO_EVALUABLE`).
- Sensibilidad S1 (sin rolls): no evaluable para ninguna familia (falta de ventanas evaluables). Sin efecto en este resultado, porque no hubo familias candidatas; si las hubiera, la regla B dependería solo de S2 y S3.

**Implicaciones y restricciones para las etapas siguientes.**

- STOP-14 no activado: TDA-14 no obliga a ejecutar TDA-15 o TDA-16 por subperíodos. TDA-15 no está autorizada por este cierre.
- No suponer una volatilidad constante en 2020-2025: la variación anual de la sigma es evidente aunque no sea evaluable como ruptura.
- No suponer un perfil intradía de amplitud constante en toda la muestra.
- Las propiedades normalizadas por sigma (media, forma de las colas, autocorrelación de primer orden) son estables: respaldan trabajar con retornos estandarizados por una volatilidad variable.
- Ninguna etapa posterior puede citar TDA-14 como prueba de estabilidad de ESCALA, DISCRECION o DEPENDENCIA_MAGNITUD, de una ruptura estructural de la volatilidad ni de un tipo de cambio concreto.
- Si una etapa reutiliza la regla B de STOP-14, debe tener en cuenta que S1 no es evaluable sobre el MNQ.
- Resultados exploratorios y retrospectivos sobre el conjunto de investigación (hasta el 20-06-2025); no confirmados en el holdout, que permanece cerrado.

**Ejecucion registrada.**

- Protocolo congelado en el commit `cf4b6ccebb77f0d8903a4ed24efbe0e4f9464081`; `configs/tda14.yaml` sin cambios (SHA-256 `32dfee31eeff14706794b615778200129736e23dae22883c8108a2e1c5c31398`, `autorizado_mnq: false`).
- El análisis usó una copia temporal de la configuración fuera del repositorio (SHA-256 `a6e92b7749ea518b1341d8e8bae3c05b35646ae3b1da76da260ca0ee12d30a09`) que cambia solo `analisis.autorizado_mnq` (true) y las rutas de los artefactos G2; mismo manifiesto.
- Dos ejecuciones del análisis idénticas byte a byte en todas las tablas, gráficos y evidencias de STOP-14 (el informe generado solo difería en la fecha, la orden y los tiempos). Los artefactos de `reports/mnq/TDA14_*` son los de la primera; los siete artefactos G2 son copias idénticas de la primera calibración.
- Guardias del holdout superadas antes y después de leer; última ventana analizada: 11-06-2025.

**Incidencias.**

- Calibración G2, primer intento con 4 procesos (14-10-2026): detenido a los 90 s sin escribir artefactos, porque la memoria virtual libre bajó a 1,9 GB; se relanzó con 2 procesos (el resultado no depende del número de procesos).
- Calibración G2, ejecución 1: el sistema estuvo suspendido de 17:06 a 20:12 (14-10-2026, hora local); la duración registrada (21.681 s) incluye esa pausa. Sin efecto: resultados idénticos byte a byte a los de la ejecución 2 (10.865 s).

Estado declarado en `configs/tda14_cierre.yaml`. Esta seccion y las dos lineas corregidas del generador (seccion 1 y seccion 12) se escribieron con `python -m ohlcv_dataroad.ingest.run_tda14 enmendar-informe --closure configs/tda14_cierre.yaml`, sin repetir ningun calculo: el resto del informe y los demas artefactos son, sin cambios, los de la ejecucion original.
