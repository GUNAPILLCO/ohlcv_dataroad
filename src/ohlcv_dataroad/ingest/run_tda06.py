"""Punto de entrada de linea de comandos para ejecutar TDA-06.

Uso
---
    python -m ohlcv_dataroad.ingest.run_tda06
    python -m ohlcv_dataroad.ingest.run_tda06 --config configs/mnq_snapshot.yaml

Como en TDA-00..05, este script no contiene logica de analisis propia:
llama a ``run_tda06_analysis`` (``tda06_intraday_calendar_profile.py``),
vuelca cada pieza a su artefacto correspondiente, dibuja el UNICO grafico
que exige el roadmap para esta etapa (perfil intradia superpuesto por
año) e imprime un resumen de auditoria por consola.
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
from ohlcv_dataroad.ingest.tda06_intraday_calendar_profile import (
    COMPLETE_YEARS,
    PARTIAL_YEARS,
    run_tda06_analysis,
)

PLOT_VARIABLES = [("volume", "Volumen (mediana)"), ("abs_r_1m", "|r_1m| (mediana)"),
                   ("log_hl", "rg_t = ln(H/L) (mediana)"), ("zero_1m", "P(r_1m = 0)")]


def _minute_to_hhmm(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


def _draw_intraday_overlay(profiles, out_path: Path) -> None:
    """El grafico obligatorio de la etapa: perfil por minuto, superpuesto por año.

    2020-2024 (años completos, evidencia principal) en linea solida;
    2019/2025 (parciales, evidencia complementaria) en linea punteada y
    mas clara -- nunca se les da el mismo peso visual (seccion 8 de la
    tarea).
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    years_present = sorted(profiles.by_year_tables.keys())
    cmap = plt.get_cmap("tab10")
    color_for_year = {y: cmap(i % 10) for i, y in enumerate(years_present)}

    for ax, (var, title) in zip(axes.flat, PLOT_VARIABLES):
        for year in years_present:
            table = profiles.by_year_tables[year][var]
            style = dict(color=color_for_year[year], linewidth=1.2)
            if year in PARTIAL_YEARS:
                style.update(linestyle=":", alpha=0.55, linewidth=1.0)
            ax.plot(table["minute_of_day"], table["point"], label=str(year), **style)
        ax.set_title(title)
        ax.set_xlabel("Minuto del dia (America/New_York)")
        ax.set_xticks(range(0, 1440, 180))
        ax.set_xticklabels([_minute_to_hhmm(m) for m in range(0, 1440, 180)])
        ax.grid(alpha=0.25)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(years_present), bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("TDA-06 -- Perfil intradia por minuto, superpuesto por año (linea punteada = año parcial)", y=1.08)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _melt_global_tables(global_tables: dict) -> pd.DataFrame:
    frames = []
    for var, table in global_tables.items():
        t = table.copy()
        t.insert(0, "variable", var)
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def _melt_by_year_tables(by_year_tables: dict) -> pd.DataFrame:
    frames = []
    for year, tables in by_year_tables.items():
        for var, table in tables.items():
            t = table.copy()
            t.insert(0, "variable", var)
            t.insert(0, "year_ny", year)
            frames.append(t)
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ejecuta TDA-06 (perfil determinista intradia y de calendario) sobre las variables de TDA-04."
    )
    parser.add_argument("--config", default="configs/mnq_snapshot.yaml")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))

    try:
        result = run_tda06_analysis(config)
    except (FileNotFoundError, HoldoutIsolationError) as exc:
        print("TDA-06 status: FAIL", file=sys.stderr)
        print(f"Motivo: {exc}", file=sys.stderr)
        return 1

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    config.interim_dir.mkdir(parents=True, exist_ok=True)

    _melt_global_tables(result.profiles.global_tables).to_csv(config.tda06_minute_global_csv_path, index=False)
    _melt_by_year_tables(result.profiles.by_year_tables).to_csv(config.tda06_minute_by_year_csv_path, index=False)
    result.weekday_profile.to_csv(config.tda06_weekday_csv_path, index=False)
    result.segmentation.to_csv(config.tda06_segmentation_csv_path, index=False)
    result.calibration.to_csv(config.tda06_calibration_csv_path, index=False)
    _draw_intraday_overlay(result.profiles, config.tda06_profile_png_path)

    if not result.stop6_activated:
        result.s_m_table.to_parquet(config.tda06_s_m_parquet_path, index=False)
        result.r_tilde_table.to_parquet(config.tda06_r_tilde_parquet_path, index=False)

    print("TDA-06 status: analisis completado")
    print(f"n filas: {len(result.df):,}   años presentes: {sorted(result.df['year_ny'].unique())}")
    print(f"umbral de movimiento extremo (|r_1m| >= percentil {config.tda06_extreme_quantile:.0%}): "
          f"{result.profiles.extreme_threshold:.6f}")
    print()

    print("Evidencia STOP-6 (variables de magnitud):")
    for var, ev in result.stop6_evidence.items():
        print(f"  {var}: rango_relativo={ev['relative_range']:.4f}  "
              f"spearman_mediano_vs_global={ev['spearman_median']:.4f}  "
              f"spearman_por_año={['%.3f' % c for c in ev['spearman_by_year']]}")
    print(f"STOP-6: {'ACTIVADO' if result.stop6_activated else 'NO ACTIVADO'}")
    print()

    print("Segmentacion propuesta (quiebres del score compuesto global, con soporte entre años completos):")
    print(result.segmentation.to_string(index=False))
    print()
    print("Calibracion (null, permutaciones de los perfiles crudos de los proxies):")
    print(result.calibration.to_string(index=False))
    print()

    if not result.stop6_activated:
        print(f"s(m) construido sobre proxy: {result.s_m_proxy}  (RETROSPECTIVO)")
        print(f"r_tilde no nulos: {int(result.r_tilde_table['r_tilde'].notna().sum()):,}")
    else:
        print("STOP-6 activado: no se construye s(m) ni r_tilde.")

    print("\nArtefactos escritos:")
    for p in [
        config.tda06_minute_global_csv_path, config.tda06_minute_by_year_csv_path,
        config.tda06_weekday_csv_path, config.tda06_segmentation_csv_path,
        config.tda06_calibration_csv_path, config.tda06_profile_png_path,
    ]:
        print(f"  {p}")
    if not result.stop6_activated:
        print(f"  {config.tda06_s_m_parquet_path}")
        print(f"  {config.tda06_r_tilde_parquet_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
