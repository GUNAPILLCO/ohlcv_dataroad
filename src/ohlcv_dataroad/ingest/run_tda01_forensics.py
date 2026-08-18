"""Punto de entrada de línea de comandos: regenera la evidencia forense de TDA-01.

Uso
---
    python -m ohlcv_dataroad.ingest.run_tda01_forensics

Genera ``reports/mnq/TDA01_evidencia_gaps.csv``: una fila por cada hueco
interno (>1 min) detectado en el conjunto de investigación, con su
clasificación por magnitud y la hora local de Nueva York de cada borde.
Es el artefacto que sustenta, de forma reproducible, las tablas y conteos
citados en ``reports/mnq/TDA01_convencion_temporal.md``.

Como en TDA-00, este script no contiene lógica de análisis propia -- sólo
carga la configuración, llama a ``build_forensic_evidence`` y escribe el
resultado. Igual que TDA-00, aborta con un mensaje claro (en vez de una
traza cruda) si la protección del hold-out (``holdout_guard.py``) detecta
una configuración inconsistente.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ohlcv_dataroad.config import load_config
from ohlcv_dataroad.ingest.holdout_guard import HoldoutIsolationError
from ohlcv_dataroad.ingest.tda01_temporal_semantics import (
    build_forensic_evidence,
    identify_early_close_like_gaps,
    summarize_boundary_labels,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenera la evidencia forense de huecos usada por TDA-01."
    )
    parser.add_argument("--config", default="configs/mnq_snapshot.yaml")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))

    try:
        gaps = build_forensic_evidence(config)
    except (FileNotFoundError, HoldoutIsolationError) as exc:
        print("TDA-01 forensics status: FAIL", file=sys.stderr)
        print(f"Motivo: {exc}", file=sys.stderr)
        return 1

    out_path = config.reports_dir / "TDA01_evidencia_gaps.csv"
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    gaps.to_csv(out_path, index=False)

    daily = gaps[gaps["magnitude_class"] == "daily_maintenance_like"]
    daily_summary = summarize_boundary_labels(daily)

    weekly = gaps[gaps["magnitude_class"] == "weekly_or_long"]
    weekly_summary = summarize_boundary_labels(weekly)

    early_close = identify_early_close_like_gaps(gaps)

    print(f"Huecos totales (>1 min) en el conjunto de investigacion: {len(gaps)}")
    print(f"\n=== Ancla 1: corte de mantenimiento diario (30-90 min): {daily_summary.n_gaps} ===")
    print("Etiqueta NY de la ultima barra antes del hueco (top 5):")
    print(daily_summary.before_ny_time_counts.head(5).to_string())
    print("Etiqueta NY de la primera barra despues del hueco (top 5):")
    print(daily_summary.after_ny_time_counts.head(5).to_string())

    print(f"\n=== Ancla 2: cierre/reapertura semanal (>40h): {weekly_summary.n_gaps} ===")
    print("Etiqueta NY de la ultima barra antes del hueco (top 5):")
    print(weekly_summary.before_ny_time_counts.head(5).to_string())

    print(f"\n=== Ancla 3: cierres anticipados (firma exacta 13:00:00 -> 18:01:00): {len(early_close)} ===")
    if len(early_close) > 0:
        print(early_close[["source_file", "before_ts_utc", "gap_minutes"]].to_string(index=False))

    print(f"\nEvidencia completa escrita en: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
