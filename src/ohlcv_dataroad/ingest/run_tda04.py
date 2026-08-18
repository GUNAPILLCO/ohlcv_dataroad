"""Punto de entrada de linea de comandos para ejecutar TDA-04.

Uso
---
    python -m ohlcv_dataroad.ingest.run_tda04
    python -m ohlcv_dataroad.ingest.run_tda04 --config configs/mnq_snapshot.yaml

Como en TDA-00/01/02/03, este script no contiene logica de analisis
propia: llama a ``run_tda04_analysis`` (``tda04_analysis_variables.py``),
vuelca cada pieza del resultado a su artefacto correspondiente e imprime
un resumen de auditoria por consola.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ohlcv_dataroad.config import load_config
from ohlcv_dataroad.ingest.holdout_guard import HoldoutIsolationError
from ohlcv_dataroad.ingest.tda04_analysis_variables import (
    REASON_VALID,
    RollConsistencyError,
    run_tda04_analysis,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ejecuta TDA-04 (construccion y auditoria de variables de analisis de 1 minuto) "
        "sobre la serie canonica de TDA-03."
    )
    parser.add_argument("--config", default="configs/mnq_snapshot.yaml")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))

    try:
        result = run_tda04_analysis(config)
    except (FileNotFoundError, HoldoutIsolationError, RollConsistencyError) as exc:
        print("TDA-04 status: FAIL", file=sys.stderr)
        print(f"Motivo: {exc}", file=sys.stderr)
        return 1

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    config.interim_dir.mkdir(parents=True, exist_ok=True)

    result.variables.to_parquet(config.tda04_variables_parquet_path, index=False)
    result.validity.to_parquet(config.tda04_validity_mask_parquet_path, index=False)
    result.losses_by_cause.to_csv(config.tda04_losses_by_cause_csv_path, index=False)
    result.th07_r_vs_R.to_csv(config.tda04_th07_r_vs_R_csv_path, index=False)

    total = len(result.validity)
    n_valid = int((result.validity["invalid_reason"] == REASON_VALID).sum())
    n_invalid = total - n_valid
    pct_retained = 100.0 * n_valid / total

    print("TDA-04 status: analisis completado")
    print(f"Filas de entrada (serie canonica de TDA-03): {total:,}")
    print(f"Retornos validos (r_1m no nulo): {n_valid:,}")
    print(f"Retornos invalidos (r_1m == NaN por regla de no-cruce): {n_invalid:,}")
    print(f"Porcentaje retenido: {pct_retained:.4f} %")
    print(f"Conservacion: {total:,} == {n_valid:,} + {n_invalid:,} -> "
          f"{'OK' if total == n_valid + n_invalid else 'FALLA'}")
    print()
    print("Perdida por causa (exclusiva, prioridad FIRST_OBSERVATION > ROLL_BOUNDARY > "
          "TRADING_DATE_BOUNDARY > NON_CONSECUTIVE_MINUTE):")
    exclusive = result.losses_by_cause[result.losses_by_cause["block"] == "exclusive_reason"]
    print(exclusive[["label", "n_rows", "pct_of_total"]].to_string(index=False))
    print()
    print("Banderas independientes (para medir solapamiento entre causas):")
    other = result.losses_by_cause[result.losses_by_cause["block"] != "exclusive_reason"]
    print(other[["label", "n_rows", "pct_of_total"]].to_string(index=False))
    print()

    # Validaciones de sanidad citadas en la seccion 10 de la tarea.
    prices = result.canonical[["open", "high", "low", "close"]]
    n_nonpositive = int((prices <= 0).to_numpy().sum())
    # log_hl y log_co nunca dependen de una barra anterior (seccion "Variables
    # de analisis" de tda04_analysis_variables.py): deberian ser SIEMPRE
    # finitos, dado que TDA-00 ya garantizo precios positivos y H>=L en el
    # 100% de las filas (0 violaciones). Se comprueba aqui, no solo se asume.
    n_nonfinite_single_bar = int((~np.isfinite(result.variables[["log_hl", "log_co"]].to_numpy())).sum())
    monotonic = bool(result.canonical["timestamp"].is_monotonic_increasing)
    duplicates = int(result.canonical["timestamp"].duplicated().sum())

    print(f"Precios <= 0 usados en los logaritmos (deberia ser 0, TDA-00 ya lo garantiza): {n_nonpositive}")
    print(f"Valores no finitos en log_hl/log_co (deberia ser 0): {n_nonfinite_single_bar}")
    print(f"Timestamps monotonicos: {monotonic}")
    print(f"Duplicados en la serie de entrada: {duplicates}")
    print()

    print("TH07 (|r_1m - R_1m|), global y por decil de |r_1m|:")
    print(result.th07_r_vs_R.to_string(index=False))

    print("\nArtefactos escritos:")
    for p in [
        config.tda04_variables_parquet_path,
        config.tda04_validity_mask_parquet_path,
        config.tda04_losses_by_cause_csv_path,
        config.tda04_th07_r_vs_R_csv_path,
    ]:
        print(f"  {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
