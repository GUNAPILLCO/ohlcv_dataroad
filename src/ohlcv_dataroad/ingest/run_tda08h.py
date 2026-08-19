"""Punto de entrada de linea de comandos para TDA08-H (complemento acotado de TDA-08).

Uso
---
    python -m ohlcv_dataroad.ingest.run_tda08h
    python -m ohlcv_dataroad.ingest.run_tda08h --config configs/mnq_snapshot.yaml

No contiene logica de analisis propia: llama a ``run_tda08h_analysis``
(``tda08h_horizon_memory_extension.py``), verifica un CONTROL DE
REGRESION obligatorio (seccion 7 de la tarea: los horizontes 1/5/10 deben
reproducir los valores congelados de TDA-08 CLOSED dentro de precision
numerica -- si no lo hacen, el script se detiene sin interpretar 30/60),
y persiste la evidencia SEPARADA de TDA-08 CERRADA.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ohlcv_dataroad.config import load_config
from ohlcv_dataroad.ingest.holdout_guard import HoldoutIsolationError
from ohlcv_dataroad.ingest.tda07_marginal_distribution import TimestampAlignmentError
from ohlcv_dataroad.ingest.tda08h_horizon_memory_extension import run_tda08h_analysis

# Valores CONGELADOS de TDA-08 CLOSED para h=1/5/10 (reports/mnq/TDA08_dependencia_lineal_media.md
# §8, y TDA08_rho1_multi_frecuencia.csv -- misma corrida real que cerro TDA-08).
# Control de regresion obligatorio (seccion 7 de la tarea): si la nueva
# ejecucion no reproduce esto dentro de tolerancia, DETENERSE antes de
# interpretar 30/60 -- significaria que algo que no debia cambiar, cambio.
FROZEN_TDA08_1_5_10 = {
    1: {"rho_1": 0.005921100675958302, "beta_1": 0.005915184560903213},
    5: {"rho_1": -0.011060, "beta_1": -0.011042},
    10: {"rho_1": 0.011494, "beta_1": 0.011435},
}
REGRESSION_TOLERANCE_PRECISE = 1e-9   # para h=1 (valor de precision completa disponible)
REGRESSION_TOLERANCE_DISPLAYED = 5e-6  # para h=5/10 (solo 6 decimales disponibles del informe)


def _check_regression_against_frozen_tda08(multi_horizon: pd.DataFrame) -> list[str]:
    """Compara h=1/5/10 contra los valores congelados de TDA-08 CLOSED. Devuelve la lista de discrepancias (vacia si todo coincide)."""
    problems = []
    idx = multi_horizon.set_index("h_minutes")
    for h, frozen in FROZEN_TDA08_1_5_10.items():
        if h not in idx.index:
            problems.append(f"h={h}: ausente de la tabla recalculada")
            continue
        tol = REGRESSION_TOLERANCE_PRECISE if h == 1 else REGRESSION_TOLERANCE_DISPLAYED
        row = idx.loc[h]
        for stat, frozen_val in frozen.items():
            new_val = float(row[stat])
            if abs(new_val - frozen_val) > tol:
                problems.append(
                    f"h={h} {stat}: congelado={frozen_val!r} vs recalculado={new_val!r} (diferencia={abs(new_val - frozen_val):.3e} > tolerancia={tol:.1e})"
                )
    return problems


def _draw_multi_horizon_plot(multi_horizon: pd.DataFrame, out_path: Path) -> None:
    df = multi_horizon.sort_values("h_minutes")
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(df))
    ax.errorbar(
        x, df["rho_1"], yerr=[df["rho_1"] - df["rho_ci_lo"], df["rho_ci_hi"] - df["rho_1"]],
        fmt="o-", color="tab:blue", capsize=4, label="rho_1 (IC 95% bootstrap)",
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{int(h)} min" for h in df["h_minutes"]])
    ax.set_xlabel("horizonte (retornos NO solapados consecutivos)")
    ax.set_ylabel("rho_1")
    ax.set_title("TDA08-H -- memoria lineal por horizonte (1/5/10/30/60 min, NO solapado)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Ejecuta TDA08-H (extension de memoria lineal a horizontes de 30/60 minutos, complemento acotado de TDA-08)."
    )
    parser.add_argument("--config", default="configs/mnq_snapshot.yaml")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))

    try:
        result = run_tda08h_analysis(config)
    except (FileNotFoundError, HoldoutIsolationError, TimestampAlignmentError) as exc:
        print("TDA08-H status: FAIL", file=sys.stderr)
        print(f"Motivo: {exc}", file=sys.stderr)
        return 1

    print("TDA08-H -- tabla multi-horizonte (1/5/10/30/60 min, NO solapado):")
    print(result.multi_horizon.to_string(index=False))
    print()

    regression_problems = _check_regression_against_frozen_tda08(result.multi_horizon)
    if regression_problems:
        print("TDA08-H status: FAIL -- CONTROL DE REGRESION FALLIDO", file=sys.stderr)
        print("Los horizontes 1/5/10 NO reproducen los valores congelados de TDA-08 CLOSED:", file=sys.stderr)
        for p in regression_problems:
            print(f"  - {p}", file=sys.stderr)
        print("DETENIDO -- no se interpretan 30/60 ni se persisten artefactos hasta identificar la causa.", file=sys.stderr)
        return 1

    print("Control de regresion 1/5/10 vs. TDA-08 CLOSED: OK (dentro de tolerancia numerica).")
    print()

    config.reports_dir.mkdir(parents=True, exist_ok=True)

    out_df = result.multi_horizon.copy()
    out_df["signo"] = out_df["rho_1"].apply(lambda v: "positivo" if v > 0 else ("negativo" if v < 0 else "cero"))
    out_df["rho_ci_excludes_zero"] = (out_df["rho_ci_lo"] > 0) | (out_df["rho_ci_hi"] < 0)
    out_df["beta_ci_excludes_zero"] = (out_df["beta_ci_lo"] > 0) | (out_df["beta_ci_hi"] < 0)
    out_df.to_csv(config.tda08h_multi_horizon_csv_path, index=False)

    _draw_multi_horizon_plot(result.multi_horizon, config.tda08h_plot_png_path)

    print("TDA08-H status: analisis completado")
    print("\nArtefactos escritos:")
    for p in [config.tda08h_multi_horizon_csv_path, config.tda08h_plot_png_path]:
        print(f"  {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
