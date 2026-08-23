"""Punto de entrada de linea de comandos para ejecutar TDA-09.

Uso
---
    python -m ohlcv_dataroad.ingest.run_tda09
    python -m ohlcv_dataroad.ingest.run_tda09 --config configs/mnq_snapshot.yaml

Como en TDA-00..08/TDA08-H, este script no contiene logica de analisis
propia: llama a ``run_tda09_analysis`` (``tda09_volatility_clustering.py``),
persiste toda la evidencia declarada como reproducible (``persist_artifacts``)
y dibuja los tres graficos obligatorios (ACF triple r/|r|/r^2, ACF crudo
vs ajustado de |r| y log_hl, y el diagnostico de decaimiento log-log vs
semi-log, solo si TH20 quedo habilitada).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ohlcv_dataroad.config import SnapshotConfig, load_config
from ohlcv_dataroad.ingest.holdout_guard import HoldoutIsolationError
from ohlcv_dataroad.ingest.tda07_marginal_distribution import RTildeInvariantError, TimestampAlignmentError
from ohlcv_dataroad.ingest.tda09_volatility_clustering import (
    MAX_LAG_MAGNITUDE,
    SMProxyMismatchError,
    TDA09Result,
    run_tda09_analysis,
)

ARTIFACT_PATH_ATTRS = [
    "tda09_acf_csv_path",
    "tda09_bootstrap_ci_csv_path",
    "tda09_clock_attribution_csv_path",
    "tda09_persistence_by_year_csv_path",
    "tda09_persistence_by_segment_csv_path",
    "tda09_portmanteau_csv_path",
    "tda09_arch_lm_csv_path",
    "tda09_g2_calibration_null1_csv_path",
    "tda09_g2_calibration_secondary_csv_path",
    "tda09_g2_synthetic_moment_check_csv_path",
    "tda09_mean_removal_sensitivity_csv_path",
    "tda09_clock_flatness_csv_path",
    "tda09_acf_triple_png_path",
    "tda09_acf_raw_vs_adjusted_png_path",
]


def _draw_triple_plot(result: TDA09Result, out_path: Path, plot_max_lag: int = 60) -> None:
    """ACF(r), ACF(|r|), ACF(r^2), crudo, superpuestas -- la ilustracion directa de "no correlacionado pero dependiente"."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for var, color in [("r", "tab:gray"), ("abs_r", "tab:blue"), ("r2", "tab:red")]:
        sub = result.acf_table[(result.acf_table["variable"] == var) & (result.acf_table["raw_adjusted"] == "raw")]
        sub = sub.sort_values("lag").iloc[:plot_max_lag]
        ax.plot(sub["lag"], sub["rho"], label=f"ACF({var}) crudo", color=color, marker=".", markersize=3, linewidth=1.0)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("rezago (minutos)")
    ax.set_ylabel("rho_k (correlacion pairwise-complete)")
    ax.set_title("TDA-09 -- ACF triple: direccion (r) vs magnitud (|r|, r^2), serie CRUDA")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_raw_vs_adjusted_plot(result: TDA09Result, out_path: Path, plot_max_lag: int = 120) -> None:
    """ACF crudo vs ajustado de |r| y log_hl -- el resultado central de la etapa (cuanto sobrevive al ajuste de reloj)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, var, title in zip(axes, ["abs_r", "log_hl"], ["|r_1m| / |r_tilde|", "log_hl / log_hl_tilde"]):
        for kind, color, label in [("raw", "tab:orange", "crudo"), ("adjusted", "tab:green", "ajustado (RETROSPECTIVO)")]:
            sub = result.acf_table[(result.acf_table["variable"] == var) & (result.acf_table["raw_adjusted"] == kind)]
            sub = sub.sort_values("lag").iloc[:plot_max_lag]
            ax.plot(sub["lag"], sub["rho"], label=label, color=color, linewidth=1.2)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("rezago (minutos)")
        ax.set_ylabel("rho_k")
        ax.legend()
        ax.grid(alpha=0.25)
    fig.suptitle("TDA-09 -- ACF de magnitud: CRUDO vs AJUSTADO por s(m) (RETROSPECTIVO)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_decay_plot(result: TDA09Result, out_path: Path) -> None:
    """Log-log vs semi-log de ACF(|r_tilde|) y ACF(log_hl_tilde) -- solo si TH20 quedo habilitada."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for var, color in [("abs_r", "tab:green"), ("log_hl", "tab:purple")]:
        sub = result.acf_table[(result.acf_table["variable"] == var) & (result.acf_table["raw_adjusted"] == "adjusted")]
        sub = sub[sub["estimable"] & (sub["rho"] > 0)].sort_values("lag")
        if sub.empty:
            continue
        axes[0].plot(np.log(sub["lag"]), np.log(sub["rho"]), color=color, marker=".", markersize=3, linewidth=0.8, label=var)
        axes[1].plot(sub["lag"], np.log(sub["rho"]), color=color, marker=".", markersize=3, linewidth=0.8, label=var)
    axes[0].set_title("log-log: log(rho_k) vs log(k)")
    axes[0].set_xlabel("log(k)")
    axes[0].set_ylabel("log(rho_k)")
    axes[1].set_title("semi-log: log(rho_k) vs k")
    axes[1].set_xlabel("k (minutos)")
    axes[1].set_ylabel("log(rho_k)")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.25)
    fig.suptitle("TDA-09 -- TH20: forma del decaimiento de ACF(magnitud ajustada) -- diagnostico descriptivo, NO estima memoria larga")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def persist_artifacts(result: TDA09Result, config: SnapshotConfig) -> list[Path]:
    """Escribe los artefactos declarados como evidencia reproducible y devuelve las rutas escritas."""
    config.reports_dir.mkdir(parents=True, exist_ok=True)

    result.acf_table.to_csv(config.tda09_acf_csv_path, index=False)
    result.bootstrap_table.to_csv(config.tda09_bootstrap_ci_csv_path, index=False)
    result.clock_attribution_table.to_csv(config.tda09_clock_attribution_csv_path, index=False)
    result.persistence_by_year.to_csv(config.tda09_persistence_by_year_csv_path, index=False)
    result.persistence_by_segment.to_csv(config.tda09_persistence_by_segment_csv_path, index=False)
    result.portmanteau_table.to_csv(config.tda09_portmanteau_csv_path, index=False)
    result.arch_lm_table.to_csv(config.tda09_arch_lm_csv_path, index=False)
    result.g2_null1_calibration.to_csv(config.tda09_g2_calibration_null1_csv_path, index=False)
    result.g2_secondary_global_calibration.to_csv(config.tda09_g2_calibration_secondary_csv_path, index=False)
    pd.DataFrame([result.g2_synthetic_moment_check]).to_csv(config.tda09_g2_synthetic_moment_check_csv_path, index=False)
    result.mean_removal_sensitivity_table.to_csv(config.tda09_mean_removal_sensitivity_csv_path, index=False)
    result.clock_flatness_table.to_csv(config.tda09_clock_flatness_csv_path, index=False)

    _draw_triple_plot(result, config.tda09_acf_triple_png_path)
    _draw_raw_vs_adjusted_plot(result, config.tda09_acf_raw_vs_adjusted_png_path)

    written = [getattr(config, attr) for attr in ARTIFACT_PATH_ATTRS]

    if result.th20_enabled:
        _draw_decay_plot(result, config.tda09_decay_png_path)
        written.append(config.tda09_decay_png_path)

    return written


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Ejecuta TDA-09 (dependencia en magnitud / volatility clustering, TH19-TH21) sobre las variables de TDA-04/TDA-06."
    )
    parser.add_argument("--config", default="configs/mnq_snapshot.yaml")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))

    t0 = time.perf_counter()
    try:
        result = run_tda09_analysis(config)
    except (FileNotFoundError, HoldoutIsolationError, TimestampAlignmentError, RTildeInvariantError, SMProxyMismatchError) as exc:
        print("TDA-09 status: FAIL", file=sys.stderr)
        print(f"Motivo: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - t0

    written_paths = persist_artifacts(result, config)

    print("TDA-09 status: analisis completado")
    print(f"n filas r_1m/r_tilde validas: {len(result.r1m_population):,} / {len(result.r_tilde_population):,}")
    print(f"n filas log_hl/log_hl_tilde: {len(result.log_hl_population):,} / {len(result.log_hl_tilde_population):,}")
    print(f"Rezago maximo analizado (predeclarado): {MAX_LAG_MAGNITUDE} minutos")
    print(f"Tiempo total: {elapsed:.1f}s")
    print()

    print("Sensibilidad de remocion de media (r vs r - beta_1*r_{t-1}, seccion 8):")
    print(result.mean_removal_sensitivity_table.to_string(index=False))
    print()

    print("Diagnostico de aplanamiento de reloj (¿el perfil quedo removido en la serie ajustada?):")
    print(result.clock_flatness_table.to_string(index=False))
    print()

    print("TH21 -- atribucion de reloj (Q(m) cruda vs ajustada, energia de dependencia):")
    print(result.clock_attribution_table.to_string(index=False))
    print()
    print(f"STOP-9: {result.stop9_decision}")
    print()

    print("ARCH-LM (Engle) -- r_1m crudo vs r_tilde ajustado:")
    print(result.arch_lm_table.to_string(index=False))
    print()

    print(f"TH20 habilitada: {result.th20_enabled}")
    if result.th20_enabled:
        for key, diag in result.decay_diagnostics.items():
            print(f"  {key}: {diag}")
    print()

    print("Artefactos escritos:")
    for p in written_paths:
        print(f"  {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
