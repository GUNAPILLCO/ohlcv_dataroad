# TDA-00 -- Inventario e integridad intra-barra

Salida obligatoria de la etapa TDA-00 (`docs/methodology/Tsay_OHLCV_analysis_roadmap.md`). Generado automaticamente por `src/ohlcv_dataroad/ingest/run_tda00.py`; no editar a mano.

**Estado final: `PASS`**

---

## 1. Alcance

- Archivos procesados: 22 (conjunto de investigacion; el hold-out declarado en `configs/mnq_snapshot.yaml` no fue abierto por este pipeline).
- Frontera de hold-out (no tocada): `2025-06-23 00:00:00` (fuente: docs/instruments/mnq/MNQ_DATA_PRIOR_KNOWLEDGE.md, seccion 11 (HOLDOUT GOVERNANCE, LOCKED)).

## 2. Esquema y tipos

| Columna | Tipo | Origen |
|---|---|---|
| `timestamp` | datetime (parseado con `strptime`) | campo 1, formato `%Y%m%d %H%M%S` |
| `open`, `high`, `low`, `close` | float | campos 2-5 |
| `volume` | int | campo 6 |

- Separador de campos: `;`
- Zona horaria del timestamp crudo: `UTC` (confirmada por la fuente de los datos; ver `reports/mnq/00_initial_repository_audit.md`, seccion 4). TDA-00 no realiza ninguna conversion de zona horaria: solo la registra.
- Tick size declarado: `0.25` -- fuente: CME Micro E-mini Nasdaq-100 (MNQ) contract specification (fuente externa, no inferida de los datos).

## 3. Conteo de filas

- Lineas totales leidas: **1,937,230**
- Filas bien formadas: **1,937,230**
- Errores de parseo (lineas descartadas del parseo, trazadas): **0**
- Verificacion de conservacion (`total_lines == parsed_rows + parse_error_rows`): CUMPLIDA
- Rango temporal cubierto: `2019-12-23 03:01:00` -> `2025-06-22 13:53:00`
- Volumen minimo / maximo observado (filas bien formadas): `1` / `24441`

## 4. Resultado de las invariantes duras

- Filas con al menos una violacion de invariante duro: **0** de 1,937,230 (0.000000 %)
- Registros de violacion (una fila puede violar mas de una regla; formato largo de `TDA00_violaciones.csv`): **0**

| Regla | Ocurrencias |
|---|---:|
| `schema_field_count` | 0 |
| `parse_error_timestamp` | 0 |
| `parse_error_numeric` | 0 |
| `null_value` | 0 |
| `non_finite_value` | 0 |
| `nonpositive_price` | 0 |
| `negative_volume` | 0 |
| `ohlc_incoherent` | 0 |
| `tick_grid_violation` | 0 |
| `duplicate_timestamp_within_file` | 0 |
| `timestamp_out_of_order` | 0 |
| `duplicate_exact_row` | 0 |

## 5. Bandera observacional (no es violacion)

- Barras con `volume == 0`: **0**. No se cuenta como violacion de invariante (el roadmap solo exige `volume >= 0`). Es un dato relevante para `TH04`/`TDA-02` (semantica de barras sin actividad), que sigue fuera de alcance de esta etapa.

## 6. Resumen por archivo

| source_file | total_lines | parsed_rows | parse_error_rows | hard_violation_rows | zero_volume_rows | first_timestamp | last_timestamp | min_volume | max_volume |
|---|---|---|---|---|---|---|---|---|---|
| 00_mnq_03_20.Last.txt | 81660 | 81660 | 0 | 0 | 0 | 2019-12-23 03:01:00 | 2020-03-20 13:30:00 | 1 | 14965 |
| 01_mnq_06_20.Last.txt | 86069 | 86069 | 0 | 0 | 0 | 2020-03-23 03:01:00 | 2020-06-19 13:30:00 | 1 | 8553 |
| 02_mnq_09_20.Last.txt | 87482 | 87482 | 0 | 0 | 0 | 2020-06-22 03:01:00 | 2020-09-18 13:30:00 | 1 | 10725 |
| 03_mnq_12_20.Last.txt | 86824 | 86824 | 0 | 0 | 0 | 2020-09-21 03:01:00 | 2020-12-18 03:00:00 | 1 | 16309 |
| 04_mnq_03_21.Last.txt | 84599 | 84599 | 0 | 0 | 0 | 2020-12-21 03:01:00 | 2021-03-19 13:30:00 | 1 | 15756 |
| 05_mnq_06_21.Last.txt | 87090 | 87090 | 0 | 0 | 0 | 2021-03-22 03:01:00 | 2021-06-18 13:30:00 | 1 | 15023 |
| 06_mnq_09_21.Last.txt | 88205 | 88205 | 0 | 0 | 0 | 2021-06-21 03:01:00 | 2021-09-17 13:30:00 | 1 | 13455 |
| 07_mnq_12_21.Last.txt | 88318 | 88318 | 0 | 0 | 0 | 2021-09-20 03:01:00 | 2021-12-17 14:30:00 | 1 | 16898 |
| 08_mnq_03_22.Last.txt | 87112 | 87112 | 0 | 0 | 0 | 2021-12-20 03:01:00 | 2022-03-18 13:30:00 | 1 | 21466 |
| 09_mnq_06_22.Last.txt | 87336 | 87336 | 0 | 0 | 0 | 2022-03-21 03:01:00 | 2022-06-17 13:30:00 | 1 | 14805 |
| 10_mnq_09_22.Last.txt | 88207 | 88207 | 0 | 0 | 0 | 2022-06-20 03:01:00 | 2022-09-16 13:55:00 | 1 | 17182 |
| 11_mnq_12_22.Last.txt | 87830 | 87830 | 0 | 0 | 0 | 2022-09-19 03:01:00 | 2022-12-16 14:30:00 | 1 | 20169 |
| 12_mnq_03_23.Last.txt | 85736 | 85736 | 0 | 0 | 0 | 2022-12-19 03:01:00 | 2023-03-17 13:30:00 | 1 | 15437 |
| 13_mnq_06_23.Last.txt | 87931 | 87931 | 0 | 0 | 0 | 2023-03-20 03:01:00 | 2023-06-16 13:30:00 | 1 | 16638 |
| 14_mnq_09_23.Last.txt | 88002 | 88002 | 0 | 0 | 0 | 2023-06-19 03:01:00 | 2023-09-15 13:30:00 | 1 | 15352 |
| 15_mnq_12_23.Last.txt | 88448 | 88448 | 0 | 0 | 0 | 2023-09-18 03:01:00 | 2023-12-15 14:30:00 | 1 | 16805 |
| 16_mnq_03_24.Last.txt | 85777 | 85777 | 0 | 0 | 0 | 2023-12-18 03:01:00 | 2024-03-15 13:30:00 | 1 | 18431 |
| 17_mnq_06_24.Last.txt | 93916 | 93916 | 0 | 0 | 0 | 2024-03-18 03:01:00 | 2024-06-21 13:30:00 | 1 | 21032 |
| 18_mnq_09_24.Last.txt | 88216 | 88216 | 0 | 0 | 0 | 2024-06-24 03:01:00 | 2024-09-21 01:19:00 | 1 | 20927 |
| 19_mnq_12_24.Last.txt | 88498 | 88498 | 0 | 0 | 0 | 2024-09-23 03:01:00 | 2024-12-20 21:30:00 | 1 | 18077 |
| 20_mnq_03_25.Last.txt | 93219 | 93219 | 0 | 0 | 0 | 2024-12-12 03:01:00 | 2025-03-22 15:03:00 | 1 | 21838 |
| 21_mnq_06_25.Last.txt | 96755 | 96755 | 0 | 0 | 0 | 2025-03-13 03:01:00 | 2025-06-22 13:53:00 | 1 | 24441 |

Tabla completa, con el conteo por regla por archivo, en `TDA00_resumen_por_archivo.csv`.

## 7. Interpretacion

- No se encontro ninguna violacion de invariante duro. No aplica `STOP-0`: el roadmap detiene la etapa solo si la fraccion de violaciones no es despreciable o si se concentran en un tramo extenso -- ninguna de las dos condiciones se da porque el conteo es cero.

---

**No se elimino ni se corrigio ningun dato en esta etapa.** Toda fila conflictiva queda trazada en `TDA00_violaciones.csv` y en la mascara persistida `tda00_bad_data_mask.parquet` (`bad_data=True`), no en el archivo crudo.
