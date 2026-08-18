"""Punto de entrada de linea de comandos para ejecutar TDA-05.

Uso
---
    python -m ohlcv_dataroad.ingest.run_tda05
    python -m ohlcv_dataroad.ingest.run_tda05 --config configs/mnq_snapshot.yaml

Como en TDA-00..04, este script no contiene logica de analisis propia:
llama a ``run_tda05_analysis`` (``tda05_effective_resolution.py``), vuelca
cada pieza a su artefacto correspondiente, dibuja el histograma en ticks
(el unico grafico que exige el roadmap para esta etapa) e imprime un
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
from ohlcv_dataroad.ingest.tda05_effective_resolution import (
    TickGridInconsistencyError,
    run_tda05_analysis,
)

# Region central mostrada en el histograma principal. MNQ tiene un rango
# de precio amplio (~7.000 a ~22.000 puntos a lo largo del conjunto de
# investigacion) y una dispersion de 1 minuto de varias decenas de ticks
# (sigma global ~22,6 ticks) -- muy distinta de las series de accion
# individual de Tsay (Tabla 5.1), casi enteramente concentradas en 0/±1
# tick. Por eso la region central util aqui es mas ancha: ±30 ticks deja
# fuera solo ~10% de las barras (verificado, ver informe) y mantiene
# legible la masa cerca de cero, que es la pregunta que el grafico debe
# responder.
HISTOGRAM_TICK_RANGE = 30


def _draw_tick_histogram(tick_df, out_path: Path) -> float:
    """Histograma de ``delta_close_ticks`` en la region central, con la cola registrada aparte.

    Entrada: la tabla de ticks (filas validas de ``r_1m``).
    Transformacion: se recorta a
    ``[-HISTOGRAM_TICK_RANGE, +HISTOGRAM_TICK_RANGE]`` ticks (un entero
    por barra, ya verificado por ``compute_tick_variables``) y se cuenta
    la frecuencia de cada valor entero -- un histograma de barras
    discretas, no una curva continua, porque el propio roadmap pide el
    "analogo directo de la Tabla 5.1 de Tsay" (una tabla de frecuencias
    por multiplo de tick, no una densidad).
    Salida: escribe el PNG y devuelve el porcentaje de barras FUERA del
    rango mostrado (para citarlo explicitamente en el pie del grafico y
    en el informe -- seccion 5 de la tarea: "indica explicitamente que
    porcentaje queda fuera del rango mostrado").
    """
    valid = tick_df[tick_df["r_1m_valid"]]
    ticks = valid["delta_close_ticks"].round().astype(int)

    in_range = ticks[(ticks >= -HISTOGRAM_TICK_RANGE) & (ticks <= HISTOGRAM_TICK_RANGE)]
    pct_outside = 100.0 * (1 - len(in_range) / len(ticks))

    counts = in_range.value_counts().reindex(range(-HISTOGRAM_TICK_RANGE, HISTOGRAM_TICK_RANGE + 1), fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(counts.index, counts.to_numpy(), width=0.8, color="#4C72B0")
    ax.set_xlabel("Movimiento Close -> Close en multiplos de tick (delta_close_ticks)")
    ax.set_ylabel("Numero de barras (r_1m valido)")
    ax.set_title(
        "TDA-05 -- Histograma de movimientos en ticks, MNQ 1 minuto "
        f"(conjunto de investigacion, region central ±{HISTOGRAM_TICK_RANGE} ticks)"
    )
    ax.set_xticks(range(-HISTOGRAM_TICK_RANGE, HISTOGRAM_TICK_RANGE + 1, 2))
    ax.text(
        0.99, 0.97,
        f"{pct_outside:.3f}% de las barras validas quedan fuera de ±{HISTOGRAM_TICK_RANGE} ticks\n"
        f"(colas registradas aparte, no ocultadas -- ver TDA05_resolucion_global.csv)",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return pct_outside


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ejecuta TDA-05 (resolucion efectiva y discrecion del retorno de 1 minuto) "
        "sobre las variables de TDA-04."
    )
    parser.add_argument("--config", default="configs/mnq_snapshot.yaml")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))

    try:
        result = run_tda05_analysis(config)
    except (FileNotFoundError, HoldoutIsolationError, TickGridInconsistencyError) as exc:
        print("TDA-05 status: FAIL", file=sys.stderr)
        print(f"Motivo: {exc}", file=sys.stderr)
        return 1

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    config.interim_dir.mkdir(parents=True, exist_ok=True)

    result.global_table.to_csv(config.tda05_global_csv_path, index=False)
    result.by_hour.to_csv(config.tda05_by_hour_csv_path, index=False)
    result.by_year.to_csv(config.tda05_by_year_csv_path, index=False)
    result.by_year_hour.to_csv(config.tda05_by_year_hour_csv_path, index=False)
    result.tick_df.to_parquet(config.tda05_tick_variables_parquet_path, index=False)

    pct_outside = _draw_tick_histogram(result.tick_df, config.tda05_histogram_png_path)

    g = result.global_table.iloc[0]
    print("TDA-05 status: analisis completado")
    print(f"n retornos validos: {int(g['n']):,}")
    print(f"zero_fraction: {g['zero_fraction']:.4%}  "
          f"(bootstrap 95% CI: [{result.zero_fraction_ci[1]:.4%}, {result.zero_fraction_ci[2]:.4%}])")
    print(f"sigma(delta_close_points): {g['sigma_delta_close_points']:.4f} puntos")
    print(f"tick_to_sigma_points: {g['tick_to_sigma_points']:.4f}  "
          f"(bootstrap 95% CI: [{result.tick_to_sigma_ci[1]:.4f}, {result.tick_to_sigma_ci[2]:.4f}])")
    print(f"tick_to_sigma_return: {g['tick_to_sigma_return']:.4f}")
    print(f"median_range_points: {g['median_range_points']:.4f}  tick_to_median_range: {g['tick_to_median_range']:.4f}")
    print(f"median_abs_ticks: {g['median_abs_ticks']:.4f}")
    print(f"prop 0 ticks: {g['prop_0_ticks']:.4%}  prop |.|=1: {g['prop_abs_1_tick']:.4%}  "
          f"prop |.|<=2: {g['prop_abs_le_2_ticks']:.4%}  prop |.|<=5: {g['prop_abs_le_5_ticks']:.4%}")
    print(f"valores distintos de r_1m en rango central (±{5} ticks): {int(g['n_distinct_r1m_central']):,}  "
          f"(vs {int(g['n_distinct_ticks_central'])} valores distintos de ticks posibles)")
    print(f"% de barras fuera de ±{HISTOGRAM_TICK_RANGE} ticks en el histograma: {pct_outside:.4f}%")
    print()

    print("Forward-fill (TH09):")
    print(f"  barras FORWARD_FILL confirmadas: {result.forward_fill.n_confirmed_forward_fill}")
    print(f"  barras CANDIDATO_FORWARD_FILL (nunca confirmadas): {result.forward_fill.n_candidate_forward_fill}")
    print(f"  {result.forward_fill.note}")
    print(f"  zero_fraction SENSITIVITY_ONLY (excluyendo candidatos): "
          f"{result.sensitivity_excluding_candidates['zero_fraction']:.4%}")
    print()

    print("Por hora NY (extremos):")
    by_hour_sorted = result.by_hour.sort_values("tick_to_sigma_points", ascending=False)
    print(by_hour_sorted[["hour_ny", "n", "zero_fraction", "tick_to_sigma_points", "prop_0_ticks"]].head(5).to_string(index=False))
    print("...")
    print(by_hour_sorted[["hour_ny", "n", "zero_fraction", "tick_to_sigma_points", "prop_0_ticks"]].tail(5).to_string(index=False))
    print()

    print("Por año NY:")
    print(result.by_year[["year_ny", "n", "zero_fraction", "tick_to_sigma_points", "median_range_points"]].to_string(index=False))
    print()

    print(f"STOP-5 watchlist (hora / año / año×hora candidatos a revision, umbrales puramente descriptivos): "
          f"{len(result.stop5_watchlist)} segmento(s)")
    if len(result.stop5_watchlist) > 0:
        print(result.stop5_watchlist[["segment", "n", "zero_fraction", "tick_to_sigma_points"]].to_string(index=False))

    print("\nArtefactos escritos:")
    for p in [
        config.tda05_global_csv_path,
        config.tda05_by_hour_csv_path,
        config.tda05_by_year_csv_path,
        config.tda05_by_year_hour_csv_path,
        config.tda05_tick_variables_parquet_path,
        config.tda05_histogram_png_path,
    ]:
        print(f"  {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
