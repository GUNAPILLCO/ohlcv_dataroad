"""Punto de entrada de linea de comandos para el complemento TH10 (pre-TDA-08).

Uso
---
    python -m ohlcv_dataroad.ingest.run_th10
    python -m ohlcv_dataroad.ingest.run_th10 --config configs/mnq_snapshot.yaml

Como en TDA-00..07, este script no contiene logica de analisis propia:
llama a ``run_th10_analysis`` (``th10_horizon_scaling.py``), vuelca cada
tabla a su CSV, dibuja el grafico log-log obligatorio, e imprime un
resumen de auditoria por consola.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ohlcv_dataroad.config import load_config
from ohlcv_dataroad.ingest.holdout_guard import HoldoutIsolationError
from ohlcv_dataroad.ingest.tda07_marginal_distribution import TimestampAlignmentError
from ohlcv_dataroad.ingest.th10_horizon_scaling import run_th10_analysis


def _draw_loglog_plot(result, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    log_h = result.var_table["log_h"].to_numpy()
    log_var = result.var_table["log_var"].to_numpy()
    ax.scatter(log_h, log_var, color="tab:blue", zorder=3, label="Var(r[h]) observada")

    x_line = np.linspace(log_h.min(), log_h.max(), 50)
    ax.plot(x_line, result.alpha + result.beta * x_line, color="tab:blue", linewidth=1.5,
            label=f"ajuste beta={result.beta:.3f}")

    ref_alpha = log_var[0] - 1.0 * log_h[0]
    ax.plot(x_line, ref_alpha + 1.0 * x_line, color="grey", linestyle="--", linewidth=1.0,
            label="referencia beta=1 (escalado lineal)")

    ax.set_xlabel("log(h)")
    ax.set_ylabel("log(Var(r[h]))")
    ax.set_title("TH10 -- Escalado de la varianza con el horizonte")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ejecuta el complemento TH10 (escalado de la varianza con el horizonte, pre-TDA-08)."
    )
    parser.add_argument("--config", default="configs/mnq_snapshot.yaml")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))

    try:
        result = run_th10_analysis(config)
    except (FileNotFoundError, HoldoutIsolationError, TimestampAlignmentError) as exc:
        print("TH10 status: FAIL", file=sys.stderr)
        print(f"Motivo: {exc}", file=sys.stderr)
        return 1

    config.reports_dir.mkdir(parents=True, exist_ok=True)

    result.var_table.to_csv(config.th10_var_by_horizon_csv_path, index=False)
    result.var_table_non_overlap.to_csv(config.th10_var_by_horizon_non_overlap_csv_path, index=False)
    result.beta_by_year_table.to_csv(config.th10_beta_by_year_csv_path, index=False)
    _draw_loglog_plot(result, config.th10_var_h_plot_png_path)

    print("TH10 status: analisis completado")
    print(f"n filas (serie canonica): {len(result.df):,}")
    print()

    print("Var(r[h]) y VR(h) -- analisis PRINCIPAL (overlapping, bootstrap de bloques por jornada):")
    print(result.var_table.to_string(index=False))
    print()
    print(f"beta GLOBAL = {result.beta:.4f}  (alpha = {result.alpha:.4f})")
    print(f"IC 95% bootstrap ({len(result.beta_boot)} remuestreos de jornada, semilla fija): [{result.beta_ci_lo:.4f}, {result.beta_ci_hi:.4f}]")
    print(f"beta - 1 = {result.beta - 1.0:.4f}")
    print()

    print("Sensibilidad B -- ventanas NO solapadas:")
    print(result.var_table_non_overlap.to_string(index=False))
    print(f"beta (no solapado) = {result.beta_non_overlap:.4f}")
    print()

    print("Estabilidad por año (beta puntual, sin bootstrap):")
    print(result.beta_by_year_table.to_string(index=False))
    print()

    print("Artefactos escritos:")
    for p in [
        config.th10_var_by_horizon_csv_path, config.th10_var_by_horizon_non_overlap_csv_path,
        config.th10_beta_by_year_csv_path, config.th10_var_h_plot_png_path,
    ]:
        print(f"  {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
