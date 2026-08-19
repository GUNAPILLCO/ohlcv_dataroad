"""Punto de entrada de linea de comandos para ejecutar TDA-07.

Uso
---
    python -m ohlcv_dataroad.ingest.run_tda07
    python -m ohlcv_dataroad.ingest.run_tda07 --config configs/mnq_snapshot.yaml

Como en TDA-00..06, este script no contiene logica de analisis propia:
llama a ``run_tda07_analysis`` (``tda07_marginal_distribution.py``),
vuelca cada tabla a su artefacto correspondiente, dibuja los tres graficos
QQ que exige el alcance de la etapa e imprime un resumen de auditoria por
consola.
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
from ohlcv_dataroad.ingest.tda07_marginal_distribution import (
    NaiveReturnContradictionError,
    RTildeInvariantError,
    TimestampAlignmentError,
    qq_points,
    run_tda07_analysis,
)


def _draw_qq(ax, x: np.ndarray, title: str) -> None:
    theo, emp = qq_points(x)
    lims = [min(theo.min(), emp.min()), max(theo.max(), emp.max())]
    ax.plot(lims, lims, color="grey", linewidth=1.0, linestyle="--")
    ax.scatter(theo, emp, s=6)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Cuantil teorico (normal)")
    ax.set_ylabel("Cuantil empirico (estandarizado)")
    ax.grid(alpha=0.25)


def _draw_qq_global(result, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    _draw_qq(axes[0], result.r1m_population["r_1m"].to_numpy(dtype=float), "r_1m (crudo, GLOBAL)")
    _draw_qq(axes[1], result.r_tilde_population["r_tilde"].to_numpy(dtype=float), "r_tilde (ajustado, RETROSPECTIVO, GLOBAL)")
    fig.suptitle("TDA-07 -- QQ-plot global vs normal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_qq_by_segment(result, out_path: Path) -> None:
    labels = result.segment_labels
    n = len(labels)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()
    for ax, label in zip(axes_flat, labels):
        sub = result.r1m_population.loc[result.r1m_population["segment_label"] == label, "r_1m"].to_numpy(dtype=float)
        _draw_qq(ax, sub, f"r_1m -- segmento {label} (n={len(sub):,})")
    for ax in axes_flat[len(labels):]:
        ax.axis("off")
    fig.suptitle("TDA-07 -- QQ-plot de r_1m por segmento (TDA-06)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_qq_th08(result, out_path: Path) -> None:
    a = result.contrafactual.loc[result.contrafactual["r_1m_valid"], "r_1m"].to_numpy(dtype=float)
    b = result.contrafactual.loc[result.contrafactual["r_naive_1m"].notna(), "r_naive_1m"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    _draw_qq(axes[0], a, "A) r_1m valido (respeta reglas de no-cruce)")
    _draw_qq(axes[1], b, "B) r_naive_1m (contrafactual, sin reglas)")
    fig.suptitle("TDA-07 -- TH08: QQ-plot A vs B")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ejecuta TDA-07 (distribucion marginal y por segmento) sobre las variables de TDA-04/TDA-06."
    )
    parser.add_argument("--config", default="configs/mnq_snapshot.yaml")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))

    try:
        result = run_tda07_analysis(config)
    except (FileNotFoundError, HoldoutIsolationError, TimestampAlignmentError, NaiveReturnContradictionError, RTildeInvariantError) as exc:
        print("TDA-07 status: FAIL", file=sys.stderr)
        print(f"Motivo: {exc}", file=sys.stderr)
        return 1

    config.reports_dir.mkdir(parents=True, exist_ok=True)

    result.th08_global.to_csv(config.tda07_th08_global_csv_path, index=False)
    result.th08_by_cause.to_csv(config.tda07_th08_by_cause_csv_path, index=False)
    result.distribution_tables.to_csv(config.tda07_distribution_tables_csv_path, index=False)

    _draw_qq_global(result, config.tda07_qq_global_png_path)
    _draw_qq_by_segment(result, config.tda07_qq_by_segment_png_path)
    _draw_qq_th08(result, config.tda07_qq_th08_png_path)

    print("TDA-07 status: analisis completado")
    print(
        "Invariantes bloqueantes: alineacion de timestamps OK, r_naive_1m == r_1m en filas VALID OK, "
        "r_tilde alineado y etiquetado RETROSPECTIVO OK"
    )
    print()

    print("Segmentacion de TDA-06 usada como particion (no modificada):")
    print(f"  cortes (minuto del dia): {result.segment_cutoffs}")
    print(f"  tramos: {result.segment_labels}")
    print()

    print("TH08 -- comparacion PRINCIPAL (A: r_1m valido vs B: r_naive_1m global):")
    print(result.th08_global[["population", "n", "mean", "std", "skewness", "kurtosis_excess", "kurtosis_excess_trimmed"]].to_string(index=False))
    print()
    print("TH08 -- diagnostico SECUNDARIO por causa contaminante:")
    print(result.th08_by_cause[["invalid_reason", "n", "mean", "std", "kurtosis_excess"]].to_string(index=False))
    print()

    global_rows = result.distribution_tables[result.distribution_tables["scope"] == "GLOBAL"]
    print("TH11/TH12/TH13 -- resumen GLOBAL (r_1m vs r_tilde):")
    print(global_rows[["series", "n", "mean", "std", "skewness", "kurtosis_excess", "kurtosis_excess_trimmed",
                        "hac_se", "hac_l", "mean_ticks"]].to_string(index=False))
    print()

    print("Artefactos escritos:")
    for p in [
        config.tda07_th08_global_csv_path, config.tda07_th08_by_cause_csv_path,
        config.tda07_distribution_tables_csv_path, config.tda07_qq_global_png_path,
        config.tda07_qq_by_segment_png_path, config.tda07_qq_th08_png_path,
    ]:
        print(f"  {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
