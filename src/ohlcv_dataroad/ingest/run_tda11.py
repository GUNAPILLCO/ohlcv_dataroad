"""Punto de entrada de linea de comandos para ejecutar TDA-11.

Uso
---
    python -m ohlcv_dataroad.ingest.run_tda11
    python -m ohlcv_dataroad.ingest.run_tda11 --config configs/mnq_snapshot.yaml
    python -m ohlcv_dataroad.ingest.run_tda11 --workers 12

Como en TDA-00..10, este script no contiene logica de analisis propia:
llama a ``run_tda11_analysis`` (``tda11_parametric_volatility.py``),
persiste TODA la evidencia declarada como reproducible
(``persist_artifacts``) -- incluido el informe final
``TDA11_modelo_parametrico_volatilidad.md``, generado automaticamente a
partir de los resultados (``render_report``) -- una sola ejecucion
produce TODO TDA-11, sin ningun paso manual posterior.
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
from ohlcv_dataroad.ingest.tda11_parametric_volatility import (
    ARCH_VERSION,
    ASYMMETRY_MATERIALITY_RELATIVE,
    ASYMMETRY_STABILITY_MIN_YEAR_FRACTION,
    BENCHMARK_CONFIGS,
    GARCH_MIN_ALPHA_FOR_SEPARATION,
    GARCH_OPTIMIZER_EPS_GRID,
    GARCH_RESIDUAL_REDUCTION_THRESHOLD,
    GARCH_SCALE_FACTOR,
    GARCHFitError,
    MIN_N_FOR_GARCH_FIT,
    PORTMANTEAU_M,
    RESIDUAL_COMPARISON_LAG,
    RESIDUAL_COMPARISON_M,
    RowAlignmentError,
    TDA11Result,
    TOTAL_STAGES,
    WRITTEN_QUESTION,
    run_tda11_analysis,
)

ARTIFACT_PATH_ATTRS = [
    "tda11_entry_gate_csv_path",
    "tda11_benchmarks_csv_path",
    "tda11_garch_params_csv_path",
    "tda11_garch_diagnostics_csv_path",
    "tda11_asymmetry_csv_path",
    "tda11_usefulness_csv_path",
    "tda11_acf_comparison_png_path",
    "tda11_stability_png_path",
    "tda11_report_path",
]
# 6 CSV + 2 PNG + 1 MD = 9 artefactos SIEMPRE escritos cuando la puerta de
# entrada esta ABIERTA. Si STOP-11 se activa en la puerta de entrada (antes
# de construir ningun benchmark), solo se escriben `entry_gate_csv` y el
# informe -- ver `persist_artifacts`.


# ---------------------------------------------------------------------------
# Formateo markdown -- mismos helpers pequeños que TDA-07/08/09/10
# ---------------------------------------------------------------------------

def _fmt(x, nd: int = 4) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    try:
        if isinstance(x, (int, np.integer)):
            return f"{int(x):,}"
        xf = float(x)
        if np.isnan(xf):
            return "NaN"
        return f"{xf:,.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _md_table(df: pd.DataFrame, columns: list[str] | None = None, float_nd: int = 4) -> str:
    if df is None or df.empty:
        return "_(sin datos)_"
    cols = columns or list(df.columns)
    cols = [c for c in cols if c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---:"] * len(cols)) + "|"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = [_fmt(row[c], float_nd) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _benchmarks_table(benchmark_diagnostics: dict) -> pd.DataFrame:
    rows = []
    for block, configs in benchmark_diagnostics.items():
        for cfg, diag in configs.items():
            rows.append({
                "bloque": block, "config": cfg, "n_finito": diag["n_finite"],
                f"rho{RESIDUAL_COMPARISON_LAG}(|z|)": diag["rho1_abs_z"], f"Q({RESIDUAL_COMPARISON_M})_|z|": diag["q_abs_z_m"],
            })
    return pd.DataFrame(rows)


def _garch_params_row(label: str, block: str, dist: str, fit: dict) -> dict:
    return {
        "config": label, "bloque": block, "dist": dist, "n": fit.get("n"),
        "omega": fit.get("omega"), "alpha": fit.get("alpha"), "alpha_ci_lo": fit.get("alpha_ci_lo"), "alpha_ci_hi": fit.get("alpha_ci_hi"),
        "beta": fit.get("beta"), "beta_ci_lo": fit.get("beta_ci_lo"), "beta_ci_hi": fit.get("beta_ci_hi"),
        "gamma": fit.get("gamma"), "persistence": fit.get("persistence"),
        "stationary": fit.get("stationary"), "half_life": fit.get("half_life"),
        "convergence_flag": fit.get("convergence_flag"),
        "eps_selected": fit.get("eps_selected"), "n_multistart_attempts_finite": fit.get("n_multistart_attempts_finite"),
        "converged_cleanly": fit.get("converged_cleanly"),
    }


# ---------------------------------------------------------------------------
# Graficos obligatorios (solo los que responden una pregunta concreta)
# ---------------------------------------------------------------------------

def _draw_acf_comparison(result: TDA11Result, out_path: Path) -> None:
    """ACF(|z|) de GARCH primario/secundario vs los 3 benchmarks, por bloque -- la comparacion central de la etapa (§8)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    plot_max_lag = 60

    ax = axes[0]
    for cfg, diag in result.benchmark_diagnostics["clock_adjusted"].items():
        sub = diag["acf_abs_z"].loc[diag["acf_abs_z"]["estimable"]].sort_values("lag").iloc[:plot_max_lag]
        ax.plot(sub["lag"], sub["rho"], label=cfg, linewidth=1.0)
    sub_g = result.garch_primary_global_diag["acf_abs_z"]
    sub_g = sub_g.loc[sub_g["estimable"]].sort_values("lag").iloc[:plot_max_lag]
    ax.plot(sub_g["lag"], sub_g["rho"], label="GARCH(1,1)", color="black", linewidth=1.8)
    ax.axhline(0.0, color="gray", linewidth=0.6)
    ax.set_title("ACF(|z|) -- bloque CLOCK_ADJUSTED (primario)")
    ax.set_xlabel("rezago (barras)")
    ax.set_ylabel("rho_k")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for cfg, diag in result.benchmark_diagnostics["raw"].items():
        sub = diag["acf_abs_z"].loc[diag["acf_abs_z"]["estimable"]].sort_values("lag").iloc[:plot_max_lag]
        ax.plot(sub["lag"], sub["rho"], label=cfg, linewidth=1.0)
    sub_g2 = result.garch_secondary_raw_global_diag["acf_abs_z"]
    sub_g2 = sub_g2.loc[sub_g2["estimable"]].sort_values("lag").iloc[:plot_max_lag]
    ax.plot(sub_g2["lag"], sub_g2["rho"], label="GARCH(1,1)", color="black", linewidth=1.8)
    ax.axhline(0.0, color="gray", linewidth=0.6)
    ax.set_title("ACF(|z|) -- bloque RAW (secundario/sensibilidad)")
    ax.set_xlabel("rezago (barras)")
    ax.set_ylabel("rho_k")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle("TDA-11 -- dependencia residual de |z| tras estandarizar: GARCH(1,1) vs. benchmarks causales")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_stability(result: TDA11Result, out_path: Path) -> None:
    """alpha, beta y persistencia por año y por segmento, con IC -- TH24 (§9/§10)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, tab, xcol, title in [
        (axes[0], result.garch_by_year, "group", "Estabilidad por año"),
        (axes[1], result.garch_by_segment, "group", "Estabilidad por segmento"),
    ]:
        sub = tab.loc[tab["estimable"] == True].copy() if tab is not None and not tab.empty else pd.DataFrame()
        if sub.empty:
            ax.set_title(f"{title} -- sin ajustes estimables")
            continue
        x = np.arange(len(sub))
        ax.errorbar(
            x, sub["alpha"], yerr=[sub["alpha"] - sub["alpha_ci_lo"], sub["alpha_ci_hi"] - sub["alpha"]],
            fmt="o", label="alpha", color="tab:orange", capsize=3,
        )
        ax.errorbar(
            x, sub["beta"], yerr=[sub["beta"] - sub["beta_ci_lo"], sub["beta_ci_hi"] - sub["beta"]],
            fmt="s", label="beta", color="tab:blue", capsize=3,
        )
        ax.plot(x, sub["persistence"], marker="d", linestyle="--", color="black", label="alpha+beta", linewidth=1.0)
        ax.axhline(1.0, color="red", linewidth=0.8, linestyle=":", label="persistencia=1")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in sub[xcol]], rotation=45, ha="right", fontsize=8)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle("TDA-11 -- TH24: estabilidad de la persistencia GARCH(1,1) por subperiodo (bloque CLOCK_ADJUSTED)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Informe final -- generado automaticamente a partir de los resultados
# ---------------------------------------------------------------------------

def render_stop11_report(result: TDA11Result, config: SnapshotConfig, run_command: str) -> str:
    """Informe cuando STOP-11 se activa EN LA PUERTA DE ENTRADA (ningun benchmark ni modelo se construyo)."""
    gate = result.entry_gate
    return f"""# TDA-11 — Modelo paramétrico de volatilidad

**Etapa:** `docs/methodology/Tsay_OHLCV_analysis_roadmap.md` § TDA-11 *(CONDICIONAL)*
**Generado automáticamente** por `{run_command}` — este informe NUNCA se edita a mano.

> TDA-11 es CONDICIONAL: se ejecuta solo si TDA-09 encontró clustering genuino de volatilidad (no un artefacto del reloj) y ese resultado sobrevivió como tal. Esta ejecución evaluó la puerta de entrada sobre los artefactos reales de TDA-09 y **NO se cumplió**.

## 1. Puerta de entrada

- **TH21 recomputado** (desde `TDA09_clock_attribution.csv`, con el mismo código que TDA-09): `{gate.th21_verdict.get('verdict')}`
- **STOP-9 recomputado**: {'ACTIVADO' if gate.stop9_decision.get('stop9_activated') else 'NO ACTIVADO'}
- **Motivo**: {gate.reason}

## 2. Pregunta escrita

> {WRITTEN_QUESTION}

No se llegó a evaluar: la puerta de entrada se cerró antes de construir ningún benchmark.

## 13. STOP-11

**`ACTIVADO`** — {gate.reason}

## 20. Modo sencillo

TDA-11 pregunta si vale la pena resumir la memoria de la volatilidad de MNQ con un modelo estadístico (GARCH). Antes de intentarlo, se revisa si TDA-09 realmente encontró esa memoria de forma genuina (no solo un efecto de la hora del día). En esta ejecución, esa condición previa no se cumplió sobre los artefactos reales — así que TDA-11 se cierra aquí mismo, sin ajustar ningún modelo, y sin inventar una necesidad de GARCH que los datos no respaldan.

**Estado final: `STOP-11` — TDA-11 cerrada sin modelado.**
"""


def render_report(result: TDA11Result, config: SnapshotConfig, elapsed_total_seconds: float, run_command: str) -> str:
    if not result.entry_gate.gate_open:
        return render_stop11_report(result, config, run_command)

    gate = result.entry_gate
    bench_tab = _benchmarks_table(result.benchmark_diagnostics)

    garch_rows = [
        _garch_params_row("garch11_primary_global", "clock_adjusted", "normal", result.garch_primary_global),
        _garch_params_row("garch11_secondary_raw_global", "raw", "normal", result.garch_secondary_raw_global),
        _garch_params_row("garch11_student_t_sensitivity", "clock_adjusted", "t", result.garch_student_t_sensitivity),
    ]
    if result.garch_asymmetric_fit is not None:
        garch_rows.append(_garch_params_row("gjr_garch111_conditional", "clock_adjusted", "normal", result.garch_asymmetric_fit))
    garch_params_tab = pd.DataFrame(garch_rows)

    by_year_tab = result.garch_by_year.copy() if result.garch_by_year is not None else pd.DataFrame()
    by_segment_tab = result.garch_by_segment.copy() if result.garch_by_segment is not None else pd.DataFrame()

    if "persistence" in by_year_tab.columns and (by_year_tab["estimable"] == True).any():  # noqa: E712
        estimable_persistence = by_year_tab.loc[by_year_tab["estimable"] == True, "persistence"]  # noqa: E712
        th24_stability_note = (
            "la persistencia se mantiene en un rango razonablemente estrecho entre subperíodos"
            if estimable_persistence.std() < 0.15 else
            "la persistencia VARÍA de forma no despreciable entre subperíodos — no se interpreta la cifra global "
            "como una propiedad única y estable de MNQ (roadmap, advertencia explícita de mezcla de regímenes)"
        )
    else:
        th24_stability_note = (
            "ningún subperíodo (año) alcanzó el tamaño mínimo de muestra (`MIN_N_FOR_GARCH_FIT`) para un ajuste "
            "propio — la estabilidad de la persistencia entre años queda SIN EVALUAR en esta ejecución"
        )

    prim = result.garch_primary_global
    prim_diag = result.garch_primary_global_diag
    useful = result.garch_usefulness

    asym = result.asymmetry_global
    asym_dec = result.asymmetry_decision

    stage_lines = "\n".join(
        f"| {name} | {seconds:.1f}s |" for name, seconds in result.stage_timings.items() if not name.startswith("_")
    )

    half_life_txt = (
        f"{_fmt(prim['half_life'], 2)} barras válidas (~{_fmt(prim['half_life']/60.0, 2)} horas)"
        if prim["half_life"] is not None else "NO DEFINIDA (persistencia ≥ 1, ver §6)"
    )

    stop11_line = (
        f"**`ACTIVADO`** — {result.stop11_reason}" if result.stop11_activated else f"**`NO ACTIVADO`** — {result.stop11_reason}"
    )

    th23_verdict = "GARCH(1,1) APORTA información sobre los benchmarks simples" if useful["useful"] else "Los benchmarks simples son SUFICIENTES — GARCH no aporta información material"

    if asym_dec.get("material") and asym_dec.get("stable"):
        th25_verdict_txt = f"Asimetría MATERIAL y ESTABLE ({asym_dec['direction']}) — se ajustó GJR-GARCH(1,1,1) condicionalmente"
    elif asym_dec.get("material"):
        th25_verdict_txt = "Asimetría material en el agregado global pero NO estable entre años — NO se ajusta modelo asimétrico (G4)"
    else:
        th25_verdict_txt = "Asimetría NO material — resultado negativo válido (G6), NO se ajusta modelo asimétrico"

    asymmetric_section = ""
    if result.garch_asymmetric_fit is not None:
        af = result.garch_asymmetric_fit
        asymmetric_section = f"""
**GJR-GARCH(1,1,1) condicional** (única extensión asimétrica ajustada, habilitada porque TH25 encontró asimetría material y estable):

| | punto | IC 95% lo | IC 95% hi |
|---|---:|---:|---:|
| alpha | {_fmt(af['alpha'],4)} | {_fmt(af['alpha_ci_lo'],4)} | {_fmt(af['alpha_ci_hi'],4)} |
| gamma (asimetría) | {_fmt(af['gamma'],4)} | {_fmt(af['gamma_ci_lo'],4)} | {_fmt(af['gamma_ci_hi'],4)} |
| beta | {_fmt(af['beta'],4)} | {_fmt(af['beta_ci_lo'],4)} | {_fmt(af['beta_ci_hi'],4)} |
| persistencia (alpha+beta+gamma/2) | {_fmt(af['persistence'],4)} | | |

No se traslada el "efecto apalancamiento" de acciones por analogía — MNQ es un índice, y el mecanismo (si existe) podría ser cobertura de opciones, no apalancamiento contable (roadmap, riesgos de TH25). Se documenta el signo y la magnitud encontrados, no un mecanismo causal.
"""

    report = f"""# TDA-11 — Modelo paramétrico de volatilidad

**Etapa:** `docs/methodology/Tsay_OHLCV_analysis_roadmap.md` § TDA-11 *(CONDICIONAL)*
**Depende de:** TDA-09 (`PASS_WITH_OPEN_QUESTIONS` — TH21=`{gate.th21_verdict.get('verdict')}`, STOP-9 NO activado), TDA-10 (`PASS_WITH_OPEN_QUESTIONS` — TH22=`MIXTO`), TDA-08 (STOP-8a activado — dependencia en media despreciable, por eso GARCH usa media CERO).
**Librería GARCH:** `arch` versión `{ARCH_VERSION}`.
**Alcance de datos:** exclusivamente `tda04_variables_1m.parquet`, `tda04_return_validity_mask.parquet`, `tda06_r_tilde.parquet`, `tda06_s_m.parquet`, `TDA06_segmentacion_propuesta.csv` y `TDA09_clock_attribution.csv`. Ningún archivo de `data/raw/` ni de `holdout_files` fue abierto en esta etapa.
**Generado automáticamente** por `{run_command}` — este informe NUNCA se edita a mano; cualquier corrección se hace en el código (`tda11_parametric_volatility.py`/`run_tda11.py`) y se regenera con una nueva ejecución.

> Esta etapa pregunta si un GARCH(1,1) — el modelo paramétrico más simple — resume la persistencia de volatilidad de MNQ (ya encontrada en TDA-09) en parámetros interpretables y estables, y si elimina dependencia residual que los benchmarks causales simples de TDA-10 no eliminan. Sigue siendo caracterización estadística: NO evalúa capacidad predictiva OOS, rentabilidad, señales, targets ni features. NO abre el holdout.

---

## 1. Puerta de entrada

- **TH21 recomputado** (desde `TDA09_clock_attribution.csv`, con el mismo código de `tda09_volatility_clustering.classify_th21`, nunca reparseado de texto): `{gate.th21_verdict.get('verdict')}`.
- **STOP-9 recomputado** (`tda09_volatility_clustering.decide_stop9`): {'ACTIVADO' if gate.stop9_decision.get('stop9_activated') else 'NO ACTIVADO'}.
- {gate.reason}

## 2. Pregunta escrita que justificó (o no) el modelo

> {WRITTEN_QUESTION}

Preguntas secundarias: (1) valor y estabilidad de `alpha+beta` entre años/segmentos; (2) ¿deja GARCH(1,1) menos dependencia residual que EWMA/rodante/rango?; (3) ¿existe asimetría descriptiva estable respecto del signo del shock previo?

**Respuesta**: {th23_verdict} (ver §8/§13 para el detalle cuantitativo).

## 3. Benchmarks simples (obligatorios antes del modelo paramétrico)

Tres familias causales, reutilizadas de TDA-10 sin modificarlas (EWMA/rodante) + una nueva (rango, Parkinson causal): `{BENCHMARK_CONFIGS}`. `ewma_60`/`rolling_120` son exactamente `build_sigma_hat` de TDA-10; `range_ewma_60` es nuevo en esta etapa (`sigma_park_t^2 = log_hl_t^2/(4·ln 2)`, suavizado con EWMA causal half-life=60, ver `causal_range_sigma`). Diagnóstico de dependencia residual (`ACF(|z|)`, rezago {RESIDUAL_COMPARISON_LAG}, y `Q({RESIDUAL_COMPARISON_M})` portmanteau) por bloque:

{_md_table(bench_tab, ["bloque", "config", "n_finito", f"rho{RESIDUAL_COMPARISON_LAG}(|z|)", f"Q({RESIDUAL_COMPARISON_M})_|z|"], float_nd=4)}

## 4. Especificación exacta del modelo

**Primario**: GARCH(1,1), media CERO (justificado por TDA-08/STOP-8a: dependencia en media despreciable — `0,1425 ticks`), sobre `r_tilde` (RETROSPECTIVO, clock-adjusted). Distribución de innovaciones: Gaussiana (QMLE), inferencia ROBUSTA (`cov_type="robust"`, sandwich Bollerslev-Wooldridge). Escala de ajuste: `r_tilde·{GARCH_SCALE_FACTOR:g}` (verificado empíricamente necesario para convergencia estable — ver docstring de módulo; `alpha`/`beta`/persistencia son invariantes a esta elección).

**Secundario (sensibilidad)**: idéntica especificación sobre `r_1m` (RAW/CAUSAL), alcance GLOBAL únicamente (G4 — la pregunta formal ya la responde la serie principal).

**Sensibilidad de distribución**: idéntica especificación sobre `r_tilde`, innovaciones Student-t, alcance GLOBAL (justificada por la curtosis residual material encontrada en TDA-10, nunca una búsqueda de la distribución que mejor ajusta).

## 5. Parámetros e intervalos (IC 95% robustos)

{_md_table(garch_params_tab, ["config", "bloque", "dist", "n", "omega", "alpha", "alpha_ci_lo", "alpha_ci_hi", "beta", "beta_ci_lo", "beta_ci_hi", "persistence", "stationary", "convergence_flag", "converged_cleanly", "eps_selected", "n_multistart_attempts_finite"], float_nd=5)}

**Nota de validez numérica (auditoría post-primera-ejecución, transparente)**: con `n~10⁶`, se verificó empíricamente que `scipy.optimize.minimize` (SLSQP, interno a `arch`) puede reportar convergencia exitosa (`convergence_flag=0`) sin haberse movido del punto de partida — confirmado comparando los parámetros "ajustados" contra `GARCH.starting_values()` (coincidían exactamente) y un gradiente de cientos de miles en el supuesto óptimo. Causa raíz verificada: con una log-verosimilitud negativa sumada de orden 10⁶-10⁷, el paso de diferencias finitas por defecto de SLSQP queda por debajo del piso de ruido de punto flotante del objetivo. Se corrige con multi-arranque sobre una grilla predeclarada de `eps` (`GARCH_OPTIMIZER_EPS_GRID={GARCH_OPTIMIZER_EPS_GRID}`), conservando el intento de **menor log-verosimilitud negativa** — un criterio objetivo (la propia función que el MLE maximiza), nunca una elección subjetiva. La columna `eps_selected` documenta qué intento ganó en cada ajuste; `n_multistart_attempts_finite` cuántos de los {len(GARCH_OPTIMIZER_EPS_GRID)} intentos produjeron parámetros finitos.

## 6. Persistencia y traducción interpretable (configuración primaria)

`alpha+beta = {_fmt(prim['persistence'],4)}` — {'ESTACIONARIO (alpha+beta<1)' if prim['stationary'] else 'NO estacionario en el sentido de varianza (alpha+beta>=1)'}.

**Vida media de un shock de volatilidad**: {half_life_txt}.

**Advertencia metodológica explícita (roadmap)**: una persistencia cercana a 1 **NO demuestra** IGARCH ni memoria infinita — es compatible con (a) un GARCH estacionario genuinamente muy persistente, (b) cambios de nivel/régimen no modelados, o (c) otras formas de persistencia no identificadas aquí. Esta ambigüedad **no se resuelve** en esta etapa; se documenta (ver §17).

## 7. Diagnóstico de residuos estandarizados (configuración primaria)

`ACF(z)`, `ACF(|z|)`, `ACF(z²)`, portmanteau `Q(m)` (TDA-08, adaptación pairwise — nunca Ljung-Box clásico, ver docstring de `ljung_box_pvalue`) y su p-valor asintótico aproximado (NUNCA usado como criterio de importancia práctica, G5):

{_md_table(prim_diag["portmanteau_abs_z"], ["m", "Q", "estimable", "lb_pvalue"], float_nd=4)}

`rho_{RESIDUAL_COMPARISON_LAG}(|z|)` = {_fmt(prim_diag['rho1_abs_z'],4)}; `Q({RESIDUAL_COMPARISON_M})` de `|z|` = {_fmt(prim_diag['q_abs_z_m'],2)}.

Ver `{config.tda11_acf_comparison_png_name}` para la comparación visual completa (GARCH vs. los 3 benchmarks, ambos bloques).

## 8. Comparación directa contra benchmarks (decisión de utilidad informativa)

| | GARCH(1,1) primario | mejor benchmark (clock_adjusted) |
|---|---:|---:|
| `Q({RESIDUAL_COMPARISON_M})` de `\\|z\\|` | {_fmt(useful['garch_q_abs_z_m'],2)} | {_fmt(useful['best_benchmark_q_abs_z_m'],2)} |
| reducción relativa de GARCH | {_fmt(useful['relative_reduction'],4)} | umbral predeclarado: {GARCH_RESIDUAL_REDUCTION_THRESHOLD} |
| alpha (separación impacto/persistencia) | {_fmt(useful['alpha'],4)} | umbral predeclarado: {GARCH_MIN_ALPHA_FOR_SEPARATION} |

**Reduce dependencia**: {useful['reduces_dependence']}. **Separa impacto de persistencia**: {useful['separates_impact']}. **GARCH informativamente útil**: **{useful['useful']}**.

## 9. Estabilidad por año (TH24)

{_md_table(by_year_tab, ["group", "n", "estimable", "alpha", "alpha_ci_lo", "alpha_ci_hi", "beta", "beta_ci_lo", "beta_ci_hi", "persistence", "rho1_abs_z", "convergence_flag"], float_nd=4)}

## 10. Estabilidad por segmento (TH24)

{_md_table(by_segment_tab, ["group", "n", "estimable", "alpha", "alpha_ci_lo", "alpha_ci_hi", "beta", "beta_ci_lo", "beta_ci_hi", "persistence", "rho1_abs_z", "convergence_flag"], float_nd=4)}

Convención (documentada en el módulo, punto 4): un ajuste "por segmento" concatena las barras de ese segmento horario a través de los días en orden cronológico — misma simplificación, ya predeclarada por TDA-10 para sus propios estimadores causales, no una decisión nueva de esta etapa. Ver `{config.tda11_stability_png_name}` para el perfil visual.

## 11. Diagnóstico TH25 de asimetría

Comparación descriptiva **antes** de ajustar ningún modelo asimétrico: `|r_t|` condicionado al signo de `r_(t-1)`, controlando la magnitud del shock por deciles de `|r_(t-1)|` (población `r_tilde`, primario):

**Global**: diferencia relativa agrupada (negativo − positivo) = {_fmt(asym['pooled_rel_diff'],4)} (IC 95% bootstrap bloques por jornada: [{_fmt(asym['pooled_ci_lo'],4)}, {_fmt(asym['pooled_ci_hi'],4)}]), `n`={_fmt(asym['n'])}.

{_md_table(asym["by_decile"], ["decile", "n_pos", "n_neg", "mean_pos", "mean_neg", "diff_neg_minus_pos", "rel_diff"], float_nd=5)}

**Por año** (estabilidad del signo):

{_md_table(result.asymmetry_by_year, ["year", "n", "pooled_rel_diff", "pooled_ci_lo", "pooled_ci_hi", "median_rel_diff", "direction"], float_nd=4)}

**Decisión TH25**: mediana de `rel_diff` entre deciles = {_fmt(asym_dec.get('median_rel_diff'),4)} (umbral de materialidad predeclarado: ±{ASYMMETRY_MATERIALITY_RELATIVE}); estable en el mismo signo en {_fmt(asym_dec.get('stable_across_years_fraction'),2)} de los años evaluados (umbral predeclarado: {ASYMMETRY_STABILITY_MIN_YEAR_FRACTION}). **{th25_verdict_txt}**.
{asymmetric_section}
## 12. Clasificación CAUSAL/RETROSPECTIVO de cada bloque

- **`r_1m` (RAW)**: serie de entrada CAUSAL de principio a fin.
- **`r_tilde` (CLOCK_ADJUSTED)**: serie de entrada RETROSPECTIVA — `s(m)` (TDA-06) se estimó con TODA la muestra.
- **Cualquier resultado de GARCH (sobre cualquiera de las dos series)**: el MODELO EN SÍ ES RETROSPECTIVO — `omega`/`alpha`/`beta` se estiman por máxima verosimilitud sobre TODA la muestra, igual que `s(m)`. Esto es una distinción ADICIONAL a la de la serie de entrada — incluso el GARCH ajustado sobre `r_1m` (entrada causal) es, como modelo, retrospectivo. Ningún resultado de esta etapa se presenta como disponible causalmente en producción sin reestimación continua.
- **Los 3 benchmarks causales** (`ewma_60`/`rolling_120`/`range_ewma_60`) aplicados sobre `r_1m`: CAUSALES de principio a fin (TDA-10). Aplicados sobre `r_tilde`: heredan el componente RETROSPECTIVO de la propia serie de entrada, pero el filtro en sí no estima ningún parámetro adicional sobre la muestra completa.

**La pregunta FORMAL (TH23/TH24/TH25) la responde el bloque `clock_adjusted`** (controla la estacionalidad intradía, roadmap método mínimo 3). El bloque `raw` es sensibilidad secundaria — nunca se promedia ni se mezcla con el primario (auditoría de TDA-10, problema 3, aplicada aquí de nuevo).

## 13. STOP-11

{stop11_line}

## 14. TH23 — ¿qué estimador de volatilidad es suficiente?

`{'RESUELTA -- GARCH(1,1) aporta informacion' if useful['useful'] else 'RESUELTA -- los benchmarks simples son suficientes (resultado negativo, G6)'}`. Ver §8 para la comparación cuantitativa completa.

## 15. TH24 — persistencia y su estabilidad

`RESUELTA`. Persistencia global (primario) = {_fmt(prim['persistence'],4)}. Ver §9/§10 para la variación por año/segmento — {th24_stability_note}.

## 16. TH25 — asimetría de respuesta al signo del shock

`RESUELTA ({'asimetria material y estable' if asym_dec.get('material') and asym_dec.get('stable') else 'resultado negativo'})`. {th25_verdict_txt}.

## 17. Limitaciones

1. La ambigüedad IGARCH-vs-cambios-de-nivel-vs-memoria-larga (§6) **no se resuelve** en esta etapa — ninguna herramienta aquí distingue entre esas tres explicaciones.
2. El ajuste "por segmento" (§10) concatena barras de días distintos como si fueran adyacentes — simplificación predeclarada, heredada de TDA-10, no una limitación nueva pero sí real.
3. GARCH es, en su conjunto, RETROSPECTIVO (§12) — ningún resultado de esta etapa es un estimador causal disponible sin reestimación continua.
4. La comparación GARCH-vs-benchmarks (§8) es descriptiva/in-sample — nunca una competición predictiva fuera de muestra (fuera de alcance de esta fase, Nivel 4).
5. Solo se probó GARCH(1,1) y, condicionalmente, GJR-GARCH(1,1,1) — ninguna otra familia (EGARCH, FIGARCH, APARCH) se evaluó, por diseño (G4).
6. El optimizador subyacente (`scipy`/SLSQP, vía `arch`) mostró fragilidad numérica genuina a esta escala de muestra (§5, nota de validez numérica) — mitigada con multi-arranque, pero no se puede garantizar matemáticamente que el óptimo global de la verosimilitud se alcanzó en cada ajuste; se reporta el mejor resultado encontrado entre los intentos de la grilla, con el criterio objetivo de la propia verosimilitud.

## 18. Tiempos y configuración de hardware

Workers CPU configurados: **{result.n_workers}** (`default_n_workers`, máximo ~20, reservando ~4 núcleos para el sistema — docs/project_hardware.md). GPU: NO utilizada (ningún soporte GPU real disponible en `arch`/`scipy` para esta carga, ni ventaja demostrada — política del proyecto, sección 4).

| Etapa | Tiempo |
|---|---:|
{stage_lines}

**Tiempo total de la ejecución (análisis + escritura de CSV/PNG/MD): {elapsed_total_seconds:.1f}s (~{elapsed_total_seconds/60:.1f} min).**

## 19. Archivos generados

`{config.tda11_entry_gate_csv_name}`, `{config.tda11_benchmarks_csv_name}`, `{config.tda11_garch_params_csv_name}`, `{config.tda11_garch_diagnostics_csv_name}`, `{config.tda11_asymmetry_csv_name}`, `{config.tda11_usefulness_csv_name}` (6 CSV) + `{config.tda11_acf_comparison_png_name}`, `{config.tda11_stability_png_name}` (2 PNG) + este informe (MD).

## 20. Modo sencillo

**¿Qué preguntó esta etapa?** Si vale la pena resumir "la memoria de la volatilidad" de MNQ (ya confirmada en TDA-09) con un modelo estadístico (GARCH) en vez de con reglas simples (promedio móvil de volatilidad reciente).

**¿Qué encontró?** Un GARCH(1,1) sí converge y produce parámetros interpretables: `alpha≈{_fmt(prim['alpha'],3)}` (cuánto pesa la última sorpresa) y `beta≈{_fmt(prim['beta'],3)}` (cuánto persiste la volatilidad ya acumulada), sumando `alpha+beta≈{_fmt(prim['persistence'],3)}`. {'El modelo SÍ deja menos "memoria sin explicar" en los residuos que las reglas simples.' if useful['useful'] else 'Pero, comparado con las reglas simples (promedio móvil, EWMA, rango), NO deja una reducción de dependencia residual lo bastante grande como para justificar el modelo más complejo.'}

**¿Qué NO puede concluir?** Que `alpha+beta` cercano a 1 signifique "memoria infinita" — puede deberse a otras causas (cambios de régimen, por ejemplo) que esta etapa no distingue. Tampoco dice si esto sirve para predecir ni para operar — eso pertenece a otra fase, fuera de esta caracterización.

**¿La rama GARCH queda abierta o cerrada?** {'ABIERTA -- el modelo aporto informacion, TH23/24/25 documentadas con GARCH como resumen valido de la persistencia.' if not result.stop11_activated else 'CERRADA (`STOP-11`) -- resultado negativo: los benchmarks simples ya respondian la pregunta sin necesitar GARCH.'}
"""
    return report


def persist_artifacts(result: TDA11Result, config: SnapshotConfig, t0: float, run_command: str) -> list[Path]:
    config.reports_dir.mkdir(parents=True, exist_ok=True)

    gate_row = {
        "th21_verdict": result.entry_gate.th21_verdict.get("verdict"),
        "stop9_activated": result.entry_gate.stop9_decision.get("stop9_activated"),
        "gate_open": result.entry_gate.gate_open, "reason": result.entry_gate.reason,
    }
    pd.DataFrame([gate_row]).to_csv(config.tda11_entry_gate_csv_path, index=False)

    if not result.entry_gate.gate_open:
        elapsed_total_seconds = time.perf_counter() - t0
        report_text = render_report(result, config, elapsed_total_seconds, run_command)
        config.tda11_report_path.write_text(report_text, encoding="utf-8")
        return [config.tda11_entry_gate_csv_path, config.tda11_report_path]

    _benchmarks_table(result.benchmark_diagnostics).to_csv(config.tda11_benchmarks_csv_path, index=False)

    garch_rows = [
        _garch_params_row("garch11_primary_global", "clock_adjusted", "normal", result.garch_primary_global),
        _garch_params_row("garch11_secondary_raw_global", "raw", "normal", result.garch_secondary_raw_global),
        _garch_params_row("garch11_student_t_sensitivity", "clock_adjusted", "t", result.garch_student_t_sensitivity),
    ]
    if result.garch_asymmetric_fit is not None:
        garch_rows.append(_garch_params_row("gjr_garch111_conditional", "clock_adjusted", "normal", result.garch_asymmetric_fit))
    pd.DataFrame(garch_rows).to_csv(config.tda11_garch_params_csv_path, index=False)

    diag_rows = []
    for label, block, diag in [
        ("garch11_primary_global", "clock_adjusted", result.garch_primary_global_diag),
        ("garch11_secondary_raw_global", "raw", result.garch_secondary_raw_global_diag),
    ]:
        diag_rows.append({"config": label, "bloque": block, "rho1_abs_z": diag["rho1_abs_z"], "q_abs_z_m": diag["q_abs_z_m"]})
    pd.concat(
        [pd.DataFrame(diag_rows), result.garch_by_year.assign(config="garch_by_year"), result.garch_by_segment.assign(config="garch_by_segment")],
        ignore_index=True,
    ).to_csv(config.tda11_garch_diagnostics_csv_path, index=False)

    asym_by_decile = result.asymmetry_global["by_decile"].copy()
    asym_by_decile.insert(0, "scope", "GLOBAL")
    asym_by_year = result.asymmetry_by_year.copy()
    if not asym_by_year.empty:
        asym_by_year.insert(0, "scope", "YEAR")
    pd.concat([asym_by_decile, asym_by_year], ignore_index=True).to_csv(config.tda11_asymmetry_csv_path, index=False)

    pd.DataFrame([result.garch_usefulness]).to_csv(config.tda11_usefulness_csv_path, index=False)

    _draw_acf_comparison(result, config.tda11_acf_comparison_png_path)
    _draw_stability(result, config.tda11_stability_png_path)

    written = [getattr(config, attr) for attr in ARTIFACT_PATH_ATTRS if attr != "tda11_report_path"]
    written = [config.tda11_entry_gate_csv_path] + written

    elapsed_total_seconds = time.perf_counter() - t0
    report_text = render_report(result, config, elapsed_total_seconds, run_command)
    config.tda11_report_path.write_text(report_text, encoding="utf-8")
    written.append(config.tda11_report_path)

    return written


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Ejecuta TDA-11 (modelo parametrico de volatilidad, TH23-25) sobre las variables de TDA-04/06/09."
    )
    parser.add_argument("--config", default="configs/mnq_snapshot.yaml")
    parser.add_argument("--workers", type=int, default=None, help="Numero de workers CPU para los ajustes por año/segmento (por defecto: automatico, ver default_n_workers).")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))
    run_command = "python -m ohlcv_dataroad.ingest.run_tda11 --config " + str(args.config)

    n_workers = args.workers if args.workers is not None else (config.tda11_n_workers or None)

    t0 = time.perf_counter()
    try:
        result = run_tda11_analysis(config, n_workers=n_workers, verbose=True)
    except (FileNotFoundError, HoldoutIsolationError, TimestampAlignmentError, RTildeInvariantError, RowAlignmentError, GARCHFitError) as exc:
        print("TDA-11 status: FAIL", file=sys.stderr)
        print(f"Motivo: {exc}", file=sys.stderr)
        return 1

    print(f"[TDA11 {TOTAL_STAGES}/{TOTAL_STAGES}] Persistencia de artefactos (CSV/PNG/MD)...")
    t_write_start = time.perf_counter()
    written_paths = persist_artifacts(result, config, t0, run_command)
    write_elapsed = time.perf_counter() - t_write_start
    total_elapsed_final = time.perf_counter() - t0
    print(f"[TDA11 {TOTAL_STAGES}/{TOTAL_STAGES}] Persistencia de artefactos -- completado en {write_elapsed:.1f}s (total: {total_elapsed_final:.1f}s)")

    print()
    print("TDA-11 status: analisis completado")
    print(f"Puerta de entrada: {'ABIERTA' if result.entry_gate.gate_open else 'CERRADA'}")
    if result.entry_gate.gate_open:
        print(f"n filas raw/clock_adjusted: {result.n_raw:,} / {result.n_clock_adjusted:,}")
        print(f"Workers CPU usados: {result.n_workers}")
        print(f"GARCH primario: alpha={result.garch_primary_global['alpha']:.4f} beta={result.garch_primary_global['beta']:.4f} persistencia={result.garch_primary_global['persistence']:.4f}")
        print(f"GARCH util frente a benchmarks: {result.garch_usefulness['useful']}")
    print(f"STOP-11 = {'ACTIVADO' if result.stop11_activated else 'NO ACTIVADO'}")
    print(f"Tiempo total: {total_elapsed_final:.1f}s (~{total_elapsed_final/60:.1f} min)")
    print()

    print("Artefactos escritos:")
    for p in written_paths:
        print(f"  {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
