"""Punto de entrada de linea de comandos para ejecutar TDA-08 (version corregida).

Uso
---
    python -m ohlcv_dataroad.ingest.run_tda08
    python -m ohlcv_dataroad.ingest.run_tda08 --config configs/mnq_snapshot.yaml

Como en TDA-00..07/TH10, este script no contiene logica de analisis
propia: llama a ``run_tda08_analysis`` (``tda08_linear_mean_dependence.py``),
persiste TODA la evidencia declarada como reproducible (``persist_artifacts``,
seccion 8 de la 2a revision: la persistencia debe quedar probada por un
test de runner, no solo por inspeccion visual) y dibuja el grafico de
ACF con bandas de Bartlett (referencia clasica aproximada, no el
intervalo principal -- ver ``bartlett_se``).
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
from ohlcv_dataroad.ingest.tda08_linear_mean_dependence import (
    PACF_MAX_LAG,
    TDA08Result,
    run_tda08_analysis,
)

# Los 17 artefactos declarados como evidencia reproducible (informe,
# seccion cabecera + seccion 6 de la 2a revision + seccion de la 3a
# revision que separa la calibracion G2 PRINCIPAL, solo Null 1, de la
# SECUNDARIA combinada). Usado tambien por el test de persistencia del
# runner para confirmar que no quedan nombres de artefactos obsoletos.
ARTIFACT_PATH_ATTRS = [
    "tda08_acf_global_csv_path",
    "tda08_pacf_csv_path",
    "tda08_bootstrap_ci_csv_path",
    "tda08_portmanteau_csv_path",
    "tda08_g2_calibration_null1_csv_path",
    "tda08_g2_portmanteau_null1_csv_path",
    "tda08_g2_calibration_csv_path",
    "tda08_g2_portmanteau_null_csv_path",
    "tda08_g2_moment_check_csv_path",
    "tda08_ticks_translation_csv_path",
    "tda08_multi_frequency_csv_path",
    "tda08_acf_by_group_csv_path",
    "tda08_volume_decile_sensitivity_csv_path",
    "tda08_windows_csv_path",
    "tda08_windows_by_year_csv_path",
    "tda08_windows_adjacent_csv_path",
    "tda08_acf_plot_png_path",
]


def _draw_acf_plot(result: TDA08Result, out_path: Path, plot_max_lag: int = 30) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (tab, se, title) in zip(
        axes,
        [
            (result.acf_raw, result.bartlett_se_raw, "r_1m (crudo)"),
            (result.acf_adjusted, result.bartlett_se_adjusted, "r_tilde (ajustado, RETROSPECTIVO)"),
        ],
    ):
        sub = tab.iloc[:plot_max_lag]
        se_sub = se[:plot_max_lag]
        lags = sub["lag"].to_numpy()
        rho = sub["rho"].to_numpy()
        ax.bar(lags, rho, color="tab:blue", width=0.6)
        ax.plot(lags, 1.96 * se_sub, color="grey", linestyle="--", linewidth=1.0, label="Bartlett 95% (referencia clasica aproximada)")
        ax.plot(lags, -1.96 * se_sub, color="grey", linestyle="--", linewidth=1.0)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("rezago (minutos)")
        ax.set_ylabel("rho_k (correlacion pairwise-complete)")
        ax.legend()
        ax.grid(alpha=0.25)
    fig.suptitle(
        "TDA-08 -- ACF con bandas de Bartlett (diagnostico clasico aproximado; "
        "la inferencia principal es el bootstrap por trading_date, ver TDA08_bootstrap_ci.csv)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def persist_artifacts(result: TDA08Result, config: SnapshotConfig) -> list[Path]:
    """Escribe los 17 artefactos declarados como evidencia reproducible (estado actual, cierre final) y devuelve las rutas escritas.

    Extraido a una funcion propia (seccion 8 de la 2a revision) para que
    un test de runner pueda invocarla sobre una fixture pequeña sin
    ejecutar el pipeline completo sobre el dataset real. El numero de
    artefactos crecio de 10 (cierre inicial) a 14 (1a revision) a 15 (2a
    revision) a 17 (3a revision, calibracion G2 Null-1-only separada de
    la combinada secundaria) -- ``len(ARTIFACT_PATH_ATTRS)`` es la fuente
    de verdad actual, no un numero fijo en este docstring.
    """
    config.reports_dir.mkdir(parents=True, exist_ok=True)

    acf_raw_out = result.acf_raw.copy()
    acf_raw_out["series"] = "r_1m"
    acf_raw_out["bartlett_se"] = result.bartlett_se_raw
    acf_adj_out = result.acf_adjusted.copy()
    acf_adj_out["series"] = "r_tilde"
    acf_adj_out["bartlett_se"] = result.bartlett_se_adjusted
    pd.concat([acf_raw_out, acf_adj_out], ignore_index=True).to_csv(config.tda08_acf_global_csv_path, index=False)

    pacf_df = pd.DataFrame({
        "lag": list(range(1, PACF_MAX_LAG + 1)) * 2,
        "series": ["r_1m"] * PACF_MAX_LAG + ["r_tilde"] * PACF_MAX_LAG,
        "pacf": np.concatenate([result.pacf_raw, result.pacf_adjusted]),
    })
    pacf_df.to_csv(config.tda08_pacf_csv_path, index=False)

    boot_raw_out = result.bootstrap_table_raw.copy()
    boot_raw_out["series"] = "r_1m"
    boot_adj_out = result.bootstrap_table_adjusted.copy()
    boot_adj_out["series"] = "r_tilde"
    pd.concat([boot_raw_out, boot_adj_out], ignore_index=True).to_csv(config.tda08_bootstrap_ci_csv_path, index=False)

    port_raw_out = result.portmanteau_raw.copy()
    port_raw_out["series"] = "r_1m"
    port_adj_out = result.portmanteau_adjusted.copy()
    port_adj_out["series"] = "r_tilde"
    pd.concat([port_raw_out, port_adj_out], ignore_index=True).to_csv(config.tda08_portmanteau_csv_path, index=False)

    # G2 PRINCIPAL: SOLO Null 1 (3a revision) -- ver informe seccion 11.1/19.
    result.g2_null1_calibration_raw.to_csv(config.tda08_g2_calibration_null1_csv_path, index=False)
    result.g2_null1_portmanteau_raw.to_csv(config.tda08_g2_portmanteau_null1_csv_path, index=False)
    # G2 SECUNDARIO: sensibilidad historica combinando Null 1 + Null 2 -- NUNCA la inferencia principal.
    result.g2_calibration_combined_secondary.to_csv(config.tda08_g2_calibration_csv_path, index=False)
    result.g2_portmanteau_combined_secondary.to_csv(config.tda08_g2_portmanteau_null_csv_path, index=False)
    pd.DataFrame([result.g2_synthetic_null_moment_check]).to_csv(config.tda08_g2_moment_check_csv_path, index=False)
    pd.DataFrame([result.ticks_translation]).to_csv(config.tda08_ticks_translation_csv_path, index=False)
    result.multi_frequency.to_csv(config.tda08_multi_frequency_csv_path, index=False)
    result.acf_by_group_long.to_csv(config.tda08_acf_by_group_csv_path, index=False)
    result.volume_decile_sensitivity.to_csv(config.tda08_volume_decile_sensitivity_csv_path, index=False)

    windows_df = pd.DataFrame([
        {"window": "09:31-09:35", **{k: v for k, v in result.window_open.items() if k not in ("by_year", "adjacent_minutes", "window")}},
        {"window": "15:52-16:02", **{k: v for k, v in result.window_close.items() if k not in ("by_year", "adjacent_minutes", "window")}},
    ])
    windows_df.to_csv(config.tda08_windows_csv_path, index=False)

    by_year_open = result.window_open["by_year"].reset_index().assign(window="09:31-09:35")
    by_year_close = result.window_close["by_year"].reset_index().assign(window="15:52-16:02")
    pd.concat([by_year_open, by_year_close], ignore_index=True).to_csv(config.tda08_windows_by_year_csv_path, index=False)

    adj_open = result.window_open["adjacent_minutes"].reset_index().assign(window="09:31-09:35")
    adj_close = result.window_close["adjacent_minutes"].reset_index().assign(window="15:52-16:02")
    pd.concat([adj_open, adj_close], ignore_index=True).to_csv(config.tda08_windows_adjacent_csv_path, index=False)

    _draw_acf_plot(result, config.tda08_acf_plot_png_path)

    return [getattr(config, attr) for attr in ARTIFACT_PATH_ATTRS]


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Ejecuta TDA-08 (dependencia lineal en la media, TH16-TH18) sobre las variables de TDA-04/TDA-06."
    )
    parser.add_argument("--config", default="configs/mnq_snapshot.yaml")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))

    t0 = time.perf_counter()
    try:
        result = run_tda08_analysis(config)
    except (FileNotFoundError, HoldoutIsolationError, TimestampAlignmentError, RTildeInvariantError) as exc:
        print("TDA-08 status: FAIL", file=sys.stderr)
        print(f"Motivo: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - t0

    written_paths = persist_artifacts(result, config)

    print("TDA-08 status: analisis completado")
    print(f"n filas r_1m valido: {len(result.r1m_population):,}   n filas r_tilde: {len(result.r_tilde_population):,}")
    print(f"Maximo rezago REALMENTE estimable bajo no-cruce (r_1m): {result.max_estimable_lag_raw}")
    print(f"Tiempo total: {elapsed:.1f}s")
    print()

    rho1_raw = float(result.acf_raw.loc[result.acf_raw['lag'] == 1, 'rho'].iloc[0])
    beta1_raw = float(result.acf_raw.loc[result.acf_raw['lag'] == 1, 'beta'].iloc[0])
    boot_row = result.bootstrap_table_raw.set_index("lag").loc[1]
    print(f"TH16 -- rho_1 (crudo) = {rho1_raw:.6f}  beta_1 = {beta1_raw:.6f}")
    print(f"  IC 95% bootstrap rho_1 = [{boot_row['rho_ci_lo']:.6f}, {boot_row['rho_ci_hi']:.6f}]")
    print(f"  IC 95% bootstrap beta_1 = [{boot_row['beta_ci_lo']:.6f}, {boot_row['beta_ci_hi']:.6f}]")
    print(f"Traduccion a ticks: {result.ticks_translation}")
    print()

    print("Portmanteau (r_1m) -- calibration_status distingue G2_CALIBRATED / DESCRIPTIVE_UNCALIBRATED / NOT_ESTIMABLE:")
    print(result.portmanteau_raw.to_string(index=False))
    print()
    print(f"Calibracion G2 PRINCIPAL -- SOLO Null 1 (permutacion por minute_of_day, {config.tda08_n_perm} replicas):")
    print(result.g2_null1_calibration_raw.to_string(index=False))
    print()
    print("Portmanteau G2 PRINCIPAL -- SOLO Null 1:")
    print(result.g2_null1_portmanteau_raw.to_string(index=False))
    print()
    print("Diagnostico de momentos del null sintetico (Null 2) -- por que Null 2 quedo fuera de la inferencia principal:")
    print(result.g2_synthetic_null_moment_check)
    print()
    n_null_total = config.tda08_n_perm * 2
    print(f"[SECUNDARIO/HISTORICO, NO principal] Calibracion G2 combinada (Null 1 + Null 2, "
          f"{config.tda08_n_perm}+{config.tda08_n_perm}={n_null_total} replicas):")
    print(result.g2_calibration_combined_secondary.to_string(index=False))
    print()

    print("TH17 -- rho_1/beta_1 a 1/5/10 minutos (no solapado):")
    print(result.multi_frequency.to_string(index=False))
    print()

    decile_primary = result.acf_by_group_long[(result.acf_by_group_long["group_type"] == "volume_decile") & (result.acf_by_group_long["lag"] == 1)]
    print("TH17 -- rho_1 por decil de volumen (analisis PRINCIPAL -- decile_t, sin exigir decile_{t-1}, centrado POR DECIL):")
    print(decile_primary[["group", "n_pairs", "rho", "beta"]].to_string(index=False))
    print()

    year_table = result.acf_by_group_long[(result.acf_by_group_long["group_type"] == "year") & (result.acf_by_group_long["lag"] == 1)]
    print("TH18 -- rho_1 por año (centrado POR AÑO):")
    print(year_table[["group", "n_pairs", "rho"]].to_string(index=False))
    print()
    segment_table = result.acf_by_group_long[(result.acf_by_group_long["group_type"] == "segment") & (result.acf_by_group_long["lag"] == 1)]
    print("TH18 -- rho_1 por segmento TDA-06 (centrado POR SEGMENTO):")
    print(segment_table[["group", "n_pairs", "rho"]].to_string(index=False))
    print()

    print(f"Ventana 09:31-09:35 NY: n={result.window_open['n']:,}  media/barra={result.window_open['mean_per_bar']:.6f}  "
          f"({result.window_open['mean_per_bar_ticks']:.4f} ticks)  rho_1={result.window_open['rho_1']:.6f}")
    print(f"Ventana 15:52-16:02 NY: n={result.window_close['n']:,}  media/barra={result.window_close['mean_per_bar']:.6f}  "
          f"({result.window_close['mean_per_bar_ticks']:.4f} ticks)  rho_1={result.window_close['rho_1']:.6f}")

    print("\nArtefactos escritos:")
    for p in written_paths:
        print(f"  {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
