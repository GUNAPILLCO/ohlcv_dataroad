"""Punto de entrada de linea de comandos para ejecutar TDA-03.

Uso
---
    python -m ohlcv_dataroad.ingest.run_tda03
    python -m ohlcv_dataroad.ingest.run_tda03 --config configs/mnq_snapshot.yaml

Como en TDA-00/01/02, este script no contiene logica de analisis propia:
llama a ``run_tda03_analysis`` (``tda03_rolls.py``), vuelca cada pieza del
resultado a su artefacto correspondiente y traduce cualquier fallo de
gobernanza del hold-out en un mensaje ``FAIL`` legible. Si STOP-3 se
dispara, termina con codigo de salida distinto de cero (no escribe la
serie canonica: TDA-03 pide explicitamente "no la corrijas ni la
escondas", no "sigue adelante silenciosamente").
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ohlcv_dataroad.config import load_config
from ohlcv_dataroad.ingest.holdout_guard import HoldoutIsolationError
from ohlcv_dataroad.ingest.tda03_rolls import run_tda03_analysis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ejecuta TDA-03 (rolls y construccion de la serie continua) "
        "sobre el conjunto de investigacion de MNQ."
    )
    parser.add_argument("--config", default="configs/mnq_snapshot.yaml")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))

    try:
        result = run_tda03_analysis(config)
    except (FileNotFoundError, HoldoutIsolationError) as exc:
        print("TDA-03 status: FAIL", file=sys.stderr)
        print(f"Motivo: {exc}", file=sys.stderr)
        return 1

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    config.interim_dir.mkdir(parents=True, exist_ok=True)

    result.transitions.to_csv(config.tda03_transitions_csv_path, index=False)
    result.overlap_daily_evidence.to_csv(config.tda03_overlap_daily_evidence_csv_path, index=False)
    result.no_overlap_evidence.to_csv(config.tda03_no_overlap_evidence_csv_path, index=False)
    result.invariance_table.to_csv(config.tda03_invariance_table_csv_path, index=False)
    result.discarded.to_csv(config.tda03_discarded_rows_csv_path, index=False)
    result.extreme_jumps.to_csv(config.tda03_extreme_jumps_csv_path, index=False)
    result.adjustment_factors.to_csv(config.tda03_adjustment_factors_csv_path, index=False)
    result.roll_mask.to_parquet(config.tda03_roll_mask_path, index=False)
    result.canonical.to_parquet(config.tda03_canonical_series_path, index=False)

    print(f"TDA-03 status: {'BLOCKED_STOP_3' if result.stop3['triggered'] else 'analisis completado'}")
    print(f"Archivos procesados: {result.rows['source_file'].nunique()}")
    print(f"Transiciones evaluadas: {len(result.transitions)}")
    print(result.transitions["transition_type"].value_counts().to_string())
    print()
    print("Transiciones OVERLAP:")
    overlap_rows = result.transitions[result.transitions["transition_type"] == "OVERLAP"]
    print(overlap_rows[[
        "out_contract", "in_contract", "overlap_start", "overlap_end",
        "signal_date", "effective_date", "rollover_confidence",
    ]].to_string(index=False))
    print()
    print(f"Filas de investigacion (raw, con OHLCV): {len(result.rows):,}")
    print(f"Filas en la serie canonica: {len(result.canonical):,}")
    print(f"Filas descartadas: {len(result.discarded):,}")
    print(result.discarded["discard_reason"].value_counts().to_string() if len(result.discarded) else "  (ninguna)")
    print(f"Timestamps monotonicos: {result.canonical['timestamp'].is_monotonic_increasing}")
    print(f"Duplicados en la serie canonica: {int(result.canonical['timestamp'].duplicated().sum())}")
    print(f"Fronteras de roll marcadas: {int(result.roll_mask['is_roll_boundary'].sum())}")
    print()
    print(f"Contratos con basis medido y encadenado (ajuste disponible): "
          f"{result.adjustment_factors['basis_chain'].sum()} de {len(result.adjustment_factors)}")
    print()
    print(f"STOP-3: triggered={result.stop3['triggered']} "
          f"candidatos_revisados={result.stop3['n_candidates']} sospechosos={result.stop3['n_suspicious']}")
    if result.stop3["n_candidates"] > 0:
        print(result.extreme_jumps.head(10)[[
            "timestamp", "source_file", "close_prev", "close", "rel_diff_pct", "volume",
            "reverts_next_bar", "low_volume", "suspicious",
        ]].to_string(index=False))

    print("\nArtefactos escritos:")
    for p in [
        config.tda03_transitions_csv_path,
        config.tda03_overlap_daily_evidence_csv_path,
        config.tda03_no_overlap_evidence_csv_path,
        config.tda03_invariance_table_csv_path,
        config.tda03_discarded_rows_csv_path,
        config.tda03_extreme_jumps_csv_path,
        config.tda03_adjustment_factors_csv_path,
        config.tda03_roll_mask_path,
        config.tda03_canonical_series_path,
    ]:
        print(f"  {p}")

    return 3 if result.stop3["triggered"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
