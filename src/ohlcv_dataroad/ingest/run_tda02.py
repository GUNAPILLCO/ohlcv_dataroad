"""Punto de entrada de linea de comandos para ejecutar TDA-02.

Uso
---
    python -m ohlcv_dataroad.ingest.run_tda02
    python -m ohlcv_dataroad.ingest.run_tda02 --config configs/mnq_snapshot.yaml

Como en TDA-00/TDA-01, este script no contiene logica de analisis propia:
llama a ``run_tda02_analysis`` (``tda02_temporal_integrity.py``), vuelca
cada pieza del resultado a su artefacto correspondiente, dibuja el
heatmap de completitud y traduce cualquier fallo de gobernanza del
hold-out (``HoldoutIsolationError``) en un mensaje ``FAIL`` legible.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ohlcv_dataroad.config import load_config
from ohlcv_dataroad.ingest.holdout_guard import HoldoutIsolationError
from ohlcv_dataroad.ingest.session_calendar import NY_TZ
from ohlcv_dataroad.ingest.tda02_temporal_integrity import run_tda02_analysis


def _draw_completeness_heatmap(result, out_path: Path) -> None:
    """Heatmap dia x minuto-del-dia (unico grafico exigido por la tarea).

    Entrada: el resultado completo de TDA-02 (para las filas observadas).
    Transformacion: se construye una matriz booleana ``(n_dias, 1440)`` --
    fecha de calendario de Nueva York en las filas, minuto del dia (hora
    NY) en las columnas -- marcando presente cualquier minuto con AL MENOS
    una barra observada en CUALQUIER archivo (una vista agregada de todo
    el conjunto de investigacion, no por archivo: el objetivo del grafico
    es puramente visual -- distinguir huecos ESTRUCTURALES, alineados con
    el horario y visibles como bandas horizontales/verticales, de huecos
    ESPORADICOS, visibles como puntos aislados -- no reemplaza el
    inventario numerico de huecos).
    Salida: escribe un PNG en ``out_path``.
    """
    ts_ny = result.rows["timestamp"].dt.tz_localize("UTC").dt.tz_convert(NY_TZ)
    dates = ts_ny.dt.date
    minutes = ts_ny.dt.hour * 60 + ts_ny.dt.minute

    unique_dates = np.array(sorted(dates.unique()))
    date_to_row = {d: i for i, d in enumerate(unique_dates)}
    row_idx = dates.map(date_to_row).to_numpy()

    matrix = np.zeros((len(unique_dates), 1440), dtype=bool)
    matrix[row_idx, minutes.to_numpy()] = True

    fig_height = max(6, len(unique_dates) / 120)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.imshow(matrix, aspect="auto", interpolation="none", cmap="Greys", origin="upper")
    ax.set_xlabel("Minuto del dia (hora America/New_York)")
    ax.set_ylabel("Fecha (hora America/New_York)")
    ax.set_title(
        "TDA-02 -- Completitud dia x minuto-del-dia (conjunto de investigacion, todos los archivos)"
    )

    xticks = list(range(0, 1440, 120))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{m // 60:02d}:{m % 60:02d}" for m in xticks], rotation=45)

    n_year_ticks = min(12, len(unique_dates))
    ytick_positions = np.linspace(0, len(unique_dates) - 1, n_year_ticks, dtype=int)
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels([str(unique_dates[i]) for i in ytick_positions])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ejecuta TDA-02 (integridad del eje temporal y del calendario) "
        "sobre el conjunto de investigacion de MNQ."
    )
    parser.add_argument("--config", default="configs/mnq_snapshot.yaml")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))

    try:
        result = run_tda02_analysis(config)
    except (FileNotFoundError, HoldoutIsolationError) as exc:
        print("TDA-02 status: FAIL", file=sys.stderr)
        print(f"Motivo: {exc}", file=sys.stderr)
        return 1

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    config.interim_dir.mkdir(parents=True, exist_ok=True)

    result.gaps_classified.to_csv(config.tda02_gaps_csv_path, index=False)
    result.file_boundaries.to_csv(
        config.reports_dir / "TDA02_bordes_de_archivo.csv", index=False
    )

    result.coverage_by_period["by_year"].to_csv(config.tda02_coverage_by_year_csv_path, index=False)
    result.coverage_by_period["by_month"].to_csv(config.tda02_coverage_by_month_csv_path, index=False)
    result.per_day_all.to_csv(config.tda02_incomplete_days_csv_path, index=False)
    result.out_of_grid_bars.to_csv(config.tda02_out_of_grid_csv_path, index=False)
    result.dst_evidence.to_csv(config.tda02_dst_evidence_csv_path, index=False)
    result.inactive_evidence.flat_bars.to_csv(
        config.reports_dir / "TDA02_barras_planas_candidatas.csv", index=False
    )
    result.inactive_evidence.long_runs.to_csv(
        config.reports_dir / "TDA02_rachas_planas_largas.csv", index=False
    )
    result.inactive_mask.to_parquet(config.tda02_inactive_bar_mask_path, index=False)

    _draw_completeness_heatmap(result, config.tda02_heatmap_png_path)

    total_expected = int(result.per_day_all["expected"].sum())
    total_present = int(result.per_day_all["present"].sum())
    global_coverage = 100.0 * total_present / total_expected

    print(f"TDA-02 status: {'BLOCKED_STOP_2' if result.stop2['triggered'] else 'PASS (analisis completado)'}")
    print(f"Archivos procesados: {result.rows['source_file'].nunique()}")
    print(f"Filas observadas: {len(result.rows):,}")
    print(f"Minutos esperados (grilla CME_Equity, recortada por archivo): {total_expected:,}")
    print(f"Minutos presentes: {total_present:,}")
    print(f"Cobertura global: {global_coverage:.4f} %")
    print(f"Huecos internos clasificados: {len(result.gaps_classified):,}")
    print(result.gaps_classified["cause"].value_counts().to_string())
    print(f"\nBarras fuera de grilla: {len(result.out_of_grid_bars):,}")
    print(f"Bordes de archivo no alineados: {len(result.file_boundaries):,}")
    print(f"\nSTOP-2: triggered={result.stop2['triggered']} "
          f"exact_match_fraction={result.stop2['exact_match_fraction']}")
    print(f"\nTH04 -- veredicto barras inactivas:\n{result.inactive_evidence.verdict}")

    print("\nArtefactos escritos:")
    for p in [
        config.tda02_gaps_csv_path,
        config.reports_dir / "TDA02_bordes_de_archivo.csv",
        config.tda02_coverage_by_year_csv_path,
        config.tda02_coverage_by_month_csv_path,
        config.tda02_incomplete_days_csv_path,
        config.tda02_out_of_grid_csv_path,
        config.tda02_dst_evidence_csv_path,
        config.tda02_heatmap_png_path,
        config.tda02_inactive_bar_mask_path,
    ]:
        print(f"  {p}")

    return 2 if result.stop2["triggered"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
