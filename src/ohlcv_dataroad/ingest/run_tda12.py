"""Punto de entrada de linea de comandos para ejecutar TDA-12.

Uso
---
    python -m ohlcv_dataroad.ingest.run_tda12
    python -m ohlcv_dataroad.ingest.run_tda12 --config configs/mnq_snapshot.yaml

Como en TDA-00..11, este script no contiene logica de analisis propia:
llama a ``run_tda12_analysis`` (``tda12_conditional_quantiles.py``),
persiste TODA la evidencia declarada como reproducible
(``persist_artifacts``) -- incluido el informe final
``TDA12_cuantiles_condicionales.md``, generado automaticamente a partir
de los resultados (``render_report``). Una sola ejecucion produce TODO
TDA-12, sin ningun paso manual posterior.
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
from scipy.stats import norm

from ohlcv_dataroad.config import SnapshotConfig, load_config
from ohlcv_dataroad.ingest.holdout_guard import HoldoutIsolationError
from ohlcv_dataroad.ingest.tda07_marginal_distribution import RTildeInvariantError, TimestampAlignmentError
from ohlcv_dataroad.ingest.tda10_scale_vs_shape import LookaheadLeakageError, PopulationMismatchError
from ohlcv_dataroad.ingest.tda12_conditional_quantiles import (
    BLOCK_LENGTH_SENSITIVITY_DAYS,
    CI_LEVEL,
    MIN_N_FOR_STABILITY_GROUP,
    MIN_TAIL_SUPPORT_LOW,
    MIN_TAIL_SUPPORT_UNRELIABLE,
    NORMALIZATIONS,
    PROFILE_STABILITY_FORM_THRESHOLD,
    PROFILE_STABILITY_SCALE_THRESHOLD,
    QUANTILE_LEVELS,
    ROBUSTNESS_AGREEMENT_FRACTION,
    SYMMETRIC_LEVEL_PAIRS,
    TOTAL_STAGES,
    VERDICT_GROUPINGS,
    VERDICT_INCONCLUSIVE,
    VERDICT_LEVEL_PAIR,
    VERDICT_SCALE_AND_SHAPE,
    VERDICT_SCALE_ONLY,
    CausalDecileLeakageError,
    TDA12Result,
    run_tda12_analysis,
)

ARTIFACT_PATH_ATTRS = [
    "tda12_quantiles_csv_path",
    "tda12_shape_csv_path",
    "tda12_asymmetry_csv_path",
    "tda12_sensitivity_csv_path",
    "tda12_causality_csv_path",
    "tda12_decile_sizes_csv_path",
    "tda12_block_sensitivity_csv_path",
    "tda12_readings_csv_path",
    "tda12_profiles_png_path",
    "tda12_half_width_png_path",
    "tda12_report_path",
]
# 8 CSV + 2 PNG + 1 MD = 11 artefactos, siempre los mismos.

GROUPING_TITLES = {
    "GLOBAL": "muestra completa", "SEGMENT": "segmento horario (TDA-06)",
    "VOL_DECILE_CAUSAL": "decil CAUSAL de volatilidad previa", "VOL_DECILE_FULLSAMPLE": "decil de volatilidad con fronteras de muestra completa (TDA-10, RETROSPECTIVO)",
    "YEAR": "año",
}
VERDICT_DISPLAY = {VERDICT_SCALE_ONLY: "SÓLO ESCALA", VERDICT_SCALE_AND_SHAPE: "ESCALA Y FORMA", VERDICT_INCONCLUSIVE: "NO CONCLUYENTE"}


# ---------------------------------------------------------------------------
# Formateo markdown
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


def _group_str(g) -> str:
    """Etiqueta de grupo legible: los deciles/años llegan como float (`3.0`, `2021.0`)."""
    try:
        gf = float(g)
        if np.isfinite(gf) and gf == int(gf):
            return str(int(gf))
    except (TypeError, ValueError):
        pass
    return str(g)


def _md_table(df: pd.DataFrame, columns: list[str], float_nd: int = 4, nd_by_col: dict | None = None) -> str:
    if df is None or df.empty:
        return "_(sin datos)_"
    nd_by_col = nd_by_col or {}
    cols = [c for c in columns if c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---:"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = [_group_str(row[c]) if c in ("group", "grupo") else _fmt(row[c], nd_by_col.get(c, float_nd)) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _pct(level: float) -> str:
    return f"{100 * level:g}"


def _wide_quantiles(q: pd.DataFrame, series: str, grouping: str, value: str = "q", nd: int = 5) -> pd.DataFrame:
    """Tabla ancha (una fila por grupo, una columna por percentil) de `value`, con `n` y la escala del grupo."""
    sub = q.loc[(q["series"] == series) & (q["grouping"] == grouping)]
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for g, gs in sub.groupby("group", sort=False):
        row = {"grupo": g, "n": int(gs["n"].iloc[0]), "jornadas": int(gs["n_days"].iloc[0]), "std": float(gs["std"].iloc[0])}
        for _, r in gs.iterrows():
            cell = _fmt(r[value], nd)
            if r["support"] != "OK":
                cell += " ⚠" if r["support"] == "BAJO" else " ✖"
            row[f"p{_pct(r['level'])}"] = cell
        rows.append(row)
    return pd.DataFrame(rows)


def _extreme_support_table(q: pd.DataFrame, series: str, grouping: str) -> pd.DataFrame:
    """Soporte de los cuantiles 0.1 y 99.9 por grupo: n, n de cola, jornadas que la sostienen, IC y etiqueta."""
    lo, hi = QUANTILE_LEVELS[0], QUANTILE_LEVELS[-1]
    sub = q.loc[(q["series"] == series) & (q["grouping"] == grouping) & (q["level"].isin([lo, hi]))]
    rows = []
    for _, r in sub.iterrows():
        rows.append({
            "grupo": r["group"], "percentil": _pct(r["level"]), "n": int(r["n"]), "n_cola": int(r["n_tail"]),
            "jornadas_cola": int(r["n_days_tail"]), "cuantil": r["q"], "IC95_lo": r["q_ci_lo"], "IC95_hi": r["q_ci_hi"],
            "semiancho_IC_relativo": r["ci_rel_halfwidth"], "soporte": r["support"],
        })
    return pd.DataFrame(rows)


def _shape_rows(shape: pd.DataFrame, series: str, pair: tuple[float, float], groupings: tuple[str, ...]) -> pd.DataFrame:
    sub = shape.loc[
        (shape["series"] == series) & (shape["level_lo"] == pair[0]) & (shape["level_hi"] == pair[1]) & (shape["grouping"].isin(groupings))
    ].copy()
    sub["agrupación"] = sub["grouping"]
    sub["normalización"] = sub["normalization"]
    sub["mín (grupo)"] = [f"{_fmt(a, 3)} ({_group_str(b)})" if b != "" else "NaN" for a, b in zip(sub["half_width_min"], sub["group_min"])]
    sub["máx (grupo)"] = [f"{_fmt(a, 3)} ({_group_str(b)})" if b != "" else "NaN" for a, b in zip(sub["half_width_max"], sub["group_max"])]
    def _names(v) -> str:
        return ";".join(_group_str(x) for x in str(v).split(";")) if isinstance(v, str) and v else "—"

    sub["disponibles"] = sub["n_groups_available"]
    sub["comparados"] = sub["n_groups_compared"]
    sub["excluidos (soporte)"] = [_names(v) for v in sub["groups_excluded_low_support"]]
    sub["sin normalización válida"] = [_names(v) for v in sub["groups_excluded_invalid_normalization"]]
    sub["% obs. comparadas"] = [f"{100 * v:.1f}" if np.isfinite(v) else "NaN" for v in sub["obs_fraction_compared"]]
    sub["cola izq.: ratio [IC95]"] = [f"{_fmt(a, 3)} [{_fmt(b, 3)}, {_fmt(c, 3)}]" for a, b, c in zip(sub["left_ratio"], sub["left_ratio_ci_lo"], sub["left_ratio_ci_hi"])]
    sub["cola der.: ratio [IC95]"] = [f"{_fmt(a, 3)} [{_fmt(b, 3)}, {_fmt(c, 3)}]" for a, b, c in zip(sub["right_ratio"], sub["right_ratio_ci_lo"], sub["right_ratio_ci_hi"])]
    sub["rango de asimetría [IC95]"] = [f"{_fmt(a, 3)} [{_fmt(b, 3)}, {_fmt(c, 3)}]" for a, b, c in zip(sub["asymmetry_range"], sub["asymmetry_range_ci_lo"], sub["asymmetry_range_ci_hi"])]
    sub["etiq. izq."] = sub["left_label"].replace("", "—")
    sub["etiq. der."] = sub["right_label"].replace("", "—")
    sub["etiqueta"] = sub["label"].replace("", "—")
    return sub


# ---------------------------------------------------------------------------
# Graficos obligatorios
# ---------------------------------------------------------------------------

def _draw_profiles(result: TDA12Result, out_path: Path) -> None:
    """Perfiles de cuantiles normalizados (cuantil / desviacion del grupo) superpuestos -- el diagnostico visual directo de escala vs forma (roadmap, output obligatorio)."""
    q = result.quantile_table
    raw_series = [s for s in result.series_info.loc[result.series_info["block"] == "raw", "series"]]
    fig, axes = plt.subplots(len(raw_series), len(VERDICT_GROUPINGS), figsize=(6.2 * len(VERDICT_GROUPINGS), 4.6 * len(raw_series)), squeeze=False)
    xpos = np.arange(len(QUANTILE_LEVELS))
    ref = norm.ppf(np.asarray(QUANTILE_LEVELS))
    for i, series in enumerate(raw_series):
        for j, gk in enumerate(VERDICT_GROUPINGS):
            ax = axes[i][j]
            sub = q.loc[(q["series"] == series) & (q["grouping"] == gk)]
            groups = list(dict.fromkeys(sub["group"]))
            cmap = plt.get_cmap("viridis", max(len(groups), 2))
            for k, g in enumerate(groups):
                gs = sub.loc[sub["group"] == g].sort_values("level")
                ok = (gs["support"] == "OK").to_numpy()
                y = gs["q_over_std"].to_numpy(dtype=float)
                ax.plot(xpos, y, color=cmap(k), linewidth=1.1, label=f"{_group_str(g)} (n={int(gs['n'].iloc[0]):,})")
                ax.scatter(xpos[ok], y[ok], color=cmap(k), s=14, zorder=3)
                ax.scatter(xpos[~ok], y[~ok], facecolors="none", edgecolors=cmap(k), s=34, zorder=3)
            ax.plot(xpos, ref, color="black", linestyle="--", linewidth=0.9, label="normal")
            ax.set_xticks(xpos)
            ax.set_xticklabels([_pct(l) for l in QUANTILE_LEVELS], fontsize=8)
            ax.set_xlabel("percentil")
            ax.set_ylabel("cuantil / desviación del grupo")
            ax.set_title(f"{series} por {GROUPING_TITLES[gk]}", fontsize=9)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=6, ncol=2)
    fig.suptitle(
        "TDA-12 -- perfiles de cuantiles NORMALIZADOS dibujados juntos (bloque RAW/CAUSAL). Perfiles que coinciden son compatibles con sólo escala; divergencia = forma.\n"
        "Lectura visual, no decide: la etiqueta SUPERPUESTO del informe se refiere sólo a la semi-amplitud 1%–99% normalizada, no al perfil completo.\n"
        "Círculo hueco = cuantil con soporte de cola insuficiente (no interpretar como valor preciso)."
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _draw_half_widths(result: TDA12Result, out_path: Path) -> None:
    """Semi-amplitud 1%-99% normalizada por grupo, con IC bootstrap, para las dos normalizaciones -- el estadistico del veredicto, grupo a grupo."""
    a = result.asymmetry_table
    raw_series = [s for s in result.series_info.loc[result.series_info["block"] == "raw", "series"]]
    fig, axes = plt.subplots(len(raw_series), len(VERDICT_GROUPINGS), figsize=(6.2 * len(VERDICT_GROUPINGS), 4.2 * len(raw_series)), squeeze=False)
    for i, series in enumerate(raw_series):
        for j, gk in enumerate(VERDICT_GROUPINGS):
            ax = axes[i][j]
            sub = a.loc[
                (a["series"] == series) & (a["grouping"] == gk)
                & (a["level_lo"] == VERDICT_LEVEL_PAIR[0]) & (a["level_hi"] == VERDICT_LEVEL_PAIR[1])
            ]
            x = np.arange(len(sub))
            for norm_name, color, dx in zip(NORMALIZATIONS, ("tab:blue", "tab:orange"), (-0.12, 0.12)):
                y = sub[f"half_width_over_{norm_name}"].to_numpy(dtype=float)
                lo = sub[f"half_width_over_{norm_name}_ci_lo"].to_numpy(dtype=float)
                hi = sub[f"half_width_over_{norm_name}_ci_hi"].to_numpy(dtype=float)
                yerr = np.vstack([np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)])
                yerr = np.where(np.isfinite(yerr), yerr, 0.0)
                ax.errorbar(x + dx, y, yerr=yerr, fmt="o", color=color, capsize=3, markersize=4, label=f"÷ {norm_name}")
            low = (sub["support"] != "OK").to_numpy()
            for xi in x[low]:
                ax.axvspan(xi - 0.4, xi + 0.4, color="gray", alpha=0.15)
            ax.axhline(float(norm.ppf(VERDICT_LEVEL_PAIR[1])), color="black", linestyle="--", linewidth=0.8, label="normal")
            ax.set_xticks(x)
            ax.set_xticklabels([_group_str(g) for g in sub["group"]], rotation=45, ha="right", fontsize=8)
            ax.set_title(f"{series} por {GROUPING_TITLES[gk]}", fontsize=9)
            ax.set_ylabel("semi-amplitud 1%–99% / escala del grupo")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)
    fig.suptitle(
        "TDA-12 -- semi-amplitud 1%–99% normalizada, por grupo, con IC 95% bootstrap (bloque RAW/CAUSAL).\n"
        "Si la distribución sólo cambia de escala, todos los puntos de un panel coinciden; lo contrario no es cierto: que coincidan (SUPERPUESTO)\n"
        "indica semi-amplitud normalizada estable, no igualdad del perfil completo (no ve asimetría ni otros percentiles). Banda gris = grupo con soporte de cola insuficiente."
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------

def render_report(result: TDA12Result, config: SnapshotConfig, elapsed_total_seconds: float, run_command: str) -> str:
    q, shape, asym = result.quantile_table, result.shape_table, result.asymmetry_table
    info = result.series_info
    raw = info.loc[info["block"] == "raw"]
    r_name = str(raw.loc[~raw["standardized"], "series"].iloc[0])
    z_name = str(raw.loc[raw["standardized"], "series"].iloc[0])
    adj = info.loc[info["block"] == "clock_adjusted"]
    z_adj_name = str(adj.loc[adj["standardized"], "series"].iloc[0])
    th26, stop13 = result.th26, result.stop13
    fam, par, _ = result.primary_estimator
    pcols = [f"p{_pct(l)}" for l in QUANTILE_LEVELS]

    def _global_tab(series: str) -> pd.DataFrame:
        sub = q.loc[(q["series"] == series) & (q["grouping"] == "GLOBAL")].copy()
        sub["percentil"] = [_pct(l) for l in sub["level"]]
        return sub

    def _cond_section(grouping: str) -> str:
        return (
            f"**`{r_name}` (retorno; unidades de retorno logarítmico)**\n\n{_md_table(_wide_quantiles(q, r_name, grouping, 'q', 6), ['grupo', 'n', 'jornadas', 'std'] + pcols, 6)}\n\n"
            f"**`{r_name}` en múltiplos de la desviación del grupo**\n\n{_md_table(_wide_quantiles(q, r_name, grouping, 'q_over_std', 3), ['grupo', 'n'] + pcols, 3)}\n\n"
            f"**`{z_name}` (retorno estandarizado; unidades de `sigma_hat`)**\n\n{_md_table(_wide_quantiles(q, z_name, grouping, 'q', 3), ['grupo', 'n', 'jornadas', 'std'] + pcols, 3)}\n\n"
            f"**`{z_name}` en múltiplos de la desviación del grupo**\n\n{_md_table(_wide_quantiles(q, z_name, grouping, 'q_over_std', 3), ['grupo', 'n'] + pcols, 3)}"
        )

    shape_cols = ["agrupación", "normalización", "disponibles", "comparados", "ratio", "ratio_ci_lo", "ratio_ci_hi", "mín (grupo)", "máx (grupo)", "excluidos (soporte)", "sin normalización válida", "% obs. comparadas", "etiqueta"]
    side_cols = ["agrupación", "normalización", "comparados", "cola izq.: ratio [IC95]", "etiq. izq.", "cola der.: ratio [IC95]", "etiq. der.", "rango de asimetría [IC95]"]

    readings = result.alternative_readings.copy()
    readings["lectura"] = readings["reading"]
    readings["estado"] = readings["status"]
    readings["resultado (sin degradación por robustez)"] = [VERDICT_DISPLAY[v] for v in readings["verdict_before_robustness"]]
    n_readings_alt = int((readings["status"] == "alternativa").sum())
    alt_same = int(((readings["status"] == "alternativa") & (readings["verdict_before_robustness"] == th26["verdict_before_robustness"])).sum())
    proposal = result.one_sided_proposal

    bs = result.block_sensitivity.copy() if result.block_sensitivity is not None else pd.DataFrame()
    if not bs.empty:
        bs["IC95"] = [f"[{_fmt(a, 4)}, {_fmt(b, 4)}]" for a, b in zip(bs["ci_lo"], bs["ci_hi"])]
        bs["bloque (jornadas)"] = bs["block_days"]
        bs["ancho IC ÷ ancho con 1 jornada"] = bs["width_ratio_vs_1day"]
        bs["etiqueta con ese bloque"] = bs["label_with_this_block"].replace("", "—")
    bs_ratio = bs.loc[(bs["series"] == z_name) & (bs["statistic"] == "shape_ratio_1_99")] if not bs.empty else bs
    bs_focus = bs.loc[(bs["series"] == z_name) & (bs["statistic"] != "shape_ratio_1_99") & (bs["grouping"] == "SEGMENT")] if not bs.empty else bs
    if not bs_ratio.empty:
        piv = bs_ratio.pivot_table(index=["grouping", "normalization"], columns="block_days", values="label_with_this_block", aggfunc="first")
        n_cmp = len(piv)
        n_label_changes = int((piv.nunique(axis=1) > 1).sum())
        max_widen = float(bs.loc[bs["series"] == z_name, "width_ratio_vs_1day"].max())
        block_fact = (
            f"En {n_label_changes} de {n_cmp} comparaciones del estadístico del veredicto la etiqueta cambia al alargar el bloque. "
            f"El mayor ensanchamiento de un IC de `{z_name}` es ×{_fmt(max_widen, 2)}."
        )
    else:
        block_fact = "Sensibilidad no disponible."
    year_row = shape.loc[(shape["series"] == z_name) & (shape["grouping"] == "YEAR") & (shape["level_lo"] == VERDICT_LEVEL_PAIR[0]) & (shape["normalization"] == "std")]
    year_cov = float(year_row["obs_fraction_compared"].iloc[0]) if len(year_row) else float("nan")
    year_excl = str(year_row["groups_excluded_low_support"].iloc[0]) if len(year_row) else ""
    t10 = result.shape_table_tda10_eligibility
    year_t10 = t10.loc[(t10["series"] == z_name) & (t10["grouping"] == "YEAR") & (t10["level_lo"] == VERDICT_LEVEL_PAIR[0]) & (t10["normalization"].isin(NORMALIZATIONS))]
    year_t10_txt = "; ".join(f"÷{r_['normalization']}: {_fmt(r_['ratio'], 3)} [{_fmt(r_['ratio_ci_lo'], 3)}, {_fmt(r_['ratio_ci_hi'], 3)}] → `{r_['label']}`" for _, r_ in year_t10.iterrows())
    shape_cols_nolabel = ["agrupación", "normalización", "disponibles", "comparados", "ratio", "ratio_ci_lo", "ratio_ci_hi", "mín (grupo)", "máx (grupo)", "excluidos (soporte)"]
    verdict_groupings_all = VERDICT_GROUPINGS + ("VOL_DECILE_FULLSAMPLE",)

    detail = pd.DataFrame(th26["detail"])
    detail["agrupación"] = detail["grouping"]
    detail["÷std: ratio [IC95]"] = [f"{_fmt(a, 3)} [{_fmt(b, 3)}, {_fmt(c, 3)}]" for a, b, c in zip(detail["ratio_std"], detail["ratio_std_ci_lo"], detail["ratio_std_ci_hi"])]
    detail["÷std recortada: ratio [IC95]"] = [f"{_fmt(a, 3)} [{_fmt(b, 3)}, {_fmt(c, 3)}]" for a, b, c in zip(detail["ratio_std_trimmed"], detail["ratio_std_trimmed_ci_lo"], detail["ratio_std_trimmed_ci_hi"])]

    sens = result.estimator_sensitivity.copy()
    sens["config"] = [f"{f}_{p:g}_{i}" + (" (PRIMARIO)" if prim else "") for f, p, i, prim in zip(sens["family"], sens["param"], sens["input_series"], sens["is_primary"])]
    sens_cols = ["config"] + [f"ratio_{gk}_{n}" for gk in VERDICT_GROUPINGS for n in NORMALIZATIONS] + ["point_verdict"]

    sizes = result.decile_sizes.loc[result.decile_sizes["block"] == "raw"].pivot_table(index="group", columns="grouping", values="n", aggfunc="first")
    sizes = sizes.reset_index().rename(columns={"group": "decil"})
    sizes["decil"] = [_group_str(v) for v in sizes["decil"]]

    asym_sub = asym.loc[
        (asym["series"].isin([r_name, z_name])) & (asym["grouping"].isin(("GLOBAL",) + VERDICT_GROUPINGS))
        & (asym["level_lo"].isin([0.01, 0.05]))
    ].copy()
    asym_sub["par"] = [f"{_pct(a)}/{_pct(b)}" for a, b in zip(asym_sub["level_lo"], asym_sub["level_hi"])]
    asym_global = asym.loc[(asym["grouping"] == "GLOBAL") & (asym["series"].isin([r_name, z_name]))].copy()
    asym_global["par"] = [f"{_pct(a)}/{_pct(b)}" for a, b in zip(asym_global["level_lo"], asym_global["level_hi"])]
    n_asym = len(asym_sub)
    n_asym_excl0 = int(((asym_sub["asymmetry_ci_lo"] > 0) | (asym_sub["asymmetry_ci_hi"] < 0)).sum())
    n_asym_left = int((asym_sub["asymmetry_ci_lo"] > 0).sum())
    asym_abs_max = asym_sub.loc[asym_sub["asymmetry"].abs().idxmax()] if n_asym else None

    # --- soporte de cola: hechos agregados -------------------------------
    ext = q.loc[(q["block"] == "raw") & (q["level"].isin([QUANTILE_LEVELS[0], QUANTILE_LEVELS[-1]])) & (q["grouping"].isin(("GLOBAL",) + VERDICT_GROUPINGS))]
    n_ext = len(ext)
    n_ext_low = int((ext["support"] == "BAJO").sum())
    n_ext_bad = int((ext["support"] == "NO_FIABLE").sum())
    worst = ext.loc[ext["n_tail"].idxmin()] if n_ext else None
    p1 = q.loc[(q["block"] == "raw") & (q["level"].isin(VERDICT_LEVEL_PAIR)) & (q["grouping"].isin(VERDICT_GROUPINGS))]
    n_p1_notok = int((p1["support"] != "OK").sum())

    verdict_txt = VERDICT_DISPLAY[th26["verdict"]]
    before_txt = VERDICT_DISPLAY[th26["verdict_before_robustness"]]
    labels = th26["labels_by_grouping"]
    adj_before = VERDICT_DISPLAY[result.th26_clock_adjusted["verdict_before_robustness"]]
    adj_labels = result.th26_clock_adjusted["labels_by_grouping"]

    none_rows = shape.loc[(shape["series"] == z_name) & (shape["normalization"] == "none") & (shape["level_lo"] == VERDICT_LEVEL_PAIR[0]) & (shape["grouping"].isin(VERDICT_GROUPINGS))]
    none_txt = "; ".join(f"{r['grouping']}: {_fmt(r['ratio'], 3)} [{_fmt(r['ratio_ci_lo'], 3)}, {_fmt(r['ratio_ci_hi'], 3)}]" for _, r in none_rows.iterrows())

    far_pair = SYMMETRIC_LEVEL_PAIRS[0]
    far_rows = _shape_rows(shape, z_name, far_pair, VERDICT_GROUPINGS)

    if th26["verdict"] == VERDICT_SCALE_ONLY:
        plain_verdict = "Sí, dentro de lo que mide la regla: al descontar la volatilidad reciente, la amplitud normalizada entre los percentiles 1 y 99 es estable en todas las horas, estados de volatilidad y años comparados. Eso es compatible con un cambio sólo de escala, pero no demuestra que el perfil completo de la distribución sea igual."
    elif th26["verdict"] == VERDICT_SCALE_AND_SHAPE:
        plain_verdict = "No: incluso descontando la volatilidad reciente, la forma de la distribución cambia de manera material entre al menos un tipo de contexto."
    else:
        plain_verdict = (
            "No se puede afirmar ninguna de las dos cosas con las reglas fijadas de antemano: "
            + "; ".join(f"por {GROUPING_TITLES[gk]} el resultado es `{labels[gk]}`" for gk in VERDICT_GROUPINGS) + "."
        )

    point_counts = sens["point_verdict"].value_counts()
    point_counts_txt = ", ".join(f"{int(c)} dan `{VERDICT_DISPLAY[v]}`" for v, c in point_counts.items())
    primary_point = str(sens.loc[sens["is_primary"], "point_verdict"].iloc[0]) if sens["is_primary"].any() else VERDICT_INCONCLUSIVE

    z_verdict_rows = shape.loc[(shape["series"] == z_name) & (shape["level_lo"] == VERDICT_LEVEL_PAIR[0]) & (shape["level_hi"] == VERDICT_LEVEL_PAIR[1])]
    z_std_by_group = q.loc[(q["series"] == z_name) & (q["level"] == 0.5)].set_index(["grouping", "group"])["std"]
    why_lines = []
    for gk in VERDICT_GROUPINGS:
        if labels[gk] == "SUPERPUESTO":
            continue
        parts = []
        for norm_name in NORMALIZATIONS:
            r_ = z_verdict_rows.loc[(z_verdict_rows["grouping"] == gk) & (z_verdict_rows["normalization"] == norm_name)]
            if len(r_) and r_["group_min"].iloc[0] != "":
                r_ = r_.iloc[0]
                parts.append(
                    f"dividiendo por `{norm_name}` la semi-amplitud 1 %–99 % va de {_fmt(r_['half_width_min'], 3)} (`{_group_str(r_['group_min'])}`) a "
                    f"{_fmt(r_['half_width_max'], 3)} (`{_group_str(r_['group_max'])}`), ratio {_fmt(r_['ratio'], 3)} "
                    f"[{_fmt(r_['ratio_ci_lo'], 3)}, {_fmt(r_['ratio_ci_hi'], 3)}] → `{r_['label']}`"
                )
        std_txt = ""
        gs = z_std_by_group.loc[gk] if gk in z_std_by_group.index.get_level_values(0) else None
        if gs is not None and len(gs):
            std_txt = f" Desviación de `z` por grupo: de {_fmt(float(gs.min()), 3)} (`{_group_str(gs.idxmin())}`) a {_fmt(float(gs.max()), 3)} (`{_group_str(gs.idxmax())}`)."
        why_lines.append(f"- **{GROUPING_TITLES[gk]}** (`{labels[gk]}`): " + "; ".join(parts) + "." + std_txt)
    why_txt = "\n".join(why_lines) if why_lines else "- Todas las agrupaciones son `SUPERPUESTO`."

    stage_lines = "\n".join(f"| {name} | {seconds:.1f}s |" for name, seconds in result.stage_timings.items() if not name.startswith("_"))
    cz = result.causality_decile
    cz_ok = sum(1 for c in cz.get("checks", []) if c["earlier_days_unchanged"] and c["same_day_other_bars_unchanged"] and c["truncation_identical"])
    cz_blocks = pd.Series([c.get("block", "raw") for c in cz.get("checks", [])]).value_counts()
    cz_scope_txt = ", ".join(f"{int(n_)} barras en el bloque `{b_}`" for b_, n_ in cz_blocks.items())
    cs = result.causality_sigma
    cs_scope_txt = "; ".join(
        f"`{row_['family']}` con parámetro {float(row_['param']):g}, en {int(row_['n_indices_checked'])} barras"
        for _, row_ in cs.iterrows()
    )
    cs_all_passed = bool(cs["passed"].all()) if len(cs) else False
    n_sens_not_reverified = max(int(th26["n_estimator_configs"]) - len(cs), 0)

    return f"""# TDA-12 — Cuantiles condicionales

**Etapa:** `docs/methodology/Tsay_OHLCV_analysis_roadmap.md` § TDA-12 (Nivel 4, obligatoria). **Hipótesis:** `TH26`.
**Depende de:** TDA-06 (segmentos, `r_tilde`), TDA-07 (niveles de cuantil, bootstrap por jornada), TDA-10 (`z_t`, `sigma_hat` causal, umbrales de estabilidad de forma; veredicto `MIXTO`).
**Alcance de datos:** exclusivamente `tda04_variables_1m.parquet`, `tda04_return_validity_mask.parquet`, `tda06_r_tilde.parquet` y `TDA06_segmentacion_propuesta.csv`. Ningún archivo de `data/raw/` ni de `holdout_files` fue abierto.
**Generado automáticamente** por `{run_command}` — este informe NUNCA se edita a mano; cualquier corrección se hace en el código (`tda12_conditional_quantiles.py`/`run_tda12.py`) y se regenera.

> Esta etapa describe cómo cambian los cuantiles de los retornos de MNQ según la hora, la volatilidad previa y el año, y responde si la distribución cambia sólo de ESCALA (más o menos volatilidad) o también de FORMA (colas proporcionalmente más o menos pesadas, asimetría). Es caracterización estadística: NO diseña targets, señales ni modelos; NO ejecuta TDA-13 ni EVT; NO abre el holdout.

---

## 1. Pregunta y versión ejecutada

**Pregunta central (TH26):** ¿la distribución de los retornos cambia únicamente de escala o también de forma?

**Versión: COMPLETA.** El roadmap prevé la versión mínima sólo si TDA-10 concluye que la curtosis "desaparece casi toda". TDA-10 concluyó `MIXTO` (bloque causal, acuerdo 0,50) y dejó `STOP-13 = NO SUGERIDO` con TH26 "formalmente pendiente de TDA-12". Corresponde la versión completa: cuantiles con bootstrap por grupo y el `n` que sostiene cada cuantil extremo.

## 2. Población y variables (heredadas, sin redefinir)

- **Población:** {_fmt(result.n_population)} retornos de 1 minuto válidos (`r_1m_valid=True`, TDA-04: ninguno cruza roll, jornada ni hueco), en {_fmt(result.n_days)} jornadas.
- **`r_t`** = `{r_name}`.
- **`sigma_hat_(t-1)`** = estimador PRIMARIO de TDA-10: `{fam}` con parámetro {par:g} barras sobre `r_1m`, construido con las funciones de TDA-10 sin modificar. Usa sólo `r[0..t-1]`.
- **`z_t`** = `r_t / sigma_hat_(t-1)` = `{z_name}` (mismo quemado y mismo piso numérico que TDA-10).
- **Condicionantes:** segmento horario (cortes de TDA-06), decil de `sigma_hat_(t-1)`, año.
- **Percentiles:** {", ".join(_pct(l) for l in QUANTILE_LEVELS)} (los del roadmap; `QUANTILE_LEVELS` de TDA-07).

{_md_table(info, ["series", "block", "standardized", "causal_class", "n_finite", "n_excluded_nonfinite"])}

`n_excluded_nonfinite` de `z` es el quemado inicial del estimador. El bloque `clock_adjusted` (`r_tilde` y su `z`) es RETROSPECTIVO (`s(m)` de TDA-06 usa toda la muestra): diagnóstico secundario, nunca decide (§11).

## 3. Método e incertidumbre

**Cuantiles empíricos** con la convención de TDA-07 (`np.quantile`, interpolación lineal). No se extrapola: ningún cuantil va más allá de lo observado.

**Intervalos de confianza — bootstrap por bloques.** Se usa la política vigente desde TDA-07 (aplicada también en TDA-08/09/10/11), sin inventar parámetros: los bloques son **jornadas completas** (`trading_date`), remuestreadas con reemplazo; `n_boot = {result.n_boot}` (configuración, mismo valor que las etapas anteriores); semilla fija `{result.seed}`; IC percentil al {int(100 * CI_LEVEL)} %. Una jornada entra entera o no entra, y nunca se empalma con otra: como los cuantiles y las desviaciones no dependen del orden de las observaciones, el remuestreo no fabrica continuidad entre sesiones, huecos ni contratos. En cada réplica se sortean las jornadas UNA vez y esa selección se aplica a todos los grupos, de modo que las comparaciones entre grupos conservan su dependencia. *Limitación:* remuestrear jornadas supone que jornadas distintas son intercambiables; la volatilidad persiste entre días (TDA-09/TDA-11), así que estos IC probablemente SUBESTIMAN la incertidumbre de los cuantiles extremos.

**Soporte de los cuantiles.** Para cada cuantil se reporta `n_cola` (observaciones en o más allá del cuantil) y `jornadas_cola` (jornadas distintas que las aportan — los extremos se agrupan, así que el número de episodios independientes es menor que `n_cola`). Etiqueta descriptiva: `NO_FIABLE` si `n_cola < {MIN_TAIL_SUPPORT_UNRELIABLE}`, `BAJO` si `< {MIN_TAIL_SUPPORT_LOW}`, `OK` en otro caso (constantes ya existentes en TDA-07/TDA-10). En las tablas: ⚠ = `BAJO`, ✖ = `NO_FIABLE`.

**Decisiones metodológicas NUEVAS de esta etapa — EXPLORATORIAS** (fijadas en el código antes de la primera ejecución sobre el conjunto de investigación; NO estaban aprobadas previamente; el veredicto se lee condicionado a ellas):

1. **Deciles de volatilidad causales.** TDA-10 corta los deciles con toda la muestra (fronteras con información futura). Aquí el estado primario (`VOL_DECILE_CAUSAL`) es el decil de `sigma_hat_(t-1)` respecto de los valores observados en **jornadas anteriores** únicamente. No añade parámetros (mínimo de historia: el de TDA-10). Consecuencia: los grupos no tienen el mismo tamaño. La versión de TDA-10 se conserva como `VOL_DECILE_FULLSAMPLE` sólo para comparar.
2. **El veredicto se decide sobre `z_t`.** `r_t` dentro de un segmento o un año mezcla estados de volatilidad, y una mezcla de escalas cambia la forma aunque la distribución condicional sea la misma; por eso las comparaciones sobre `r_t` son contexto y no deciden.
3. **Dos normalizaciones.** El roadmap pide dividir por la desviación del grupo; con colas tan pesadas esa desviación la dominan pocos puntos, así que cada comparación se hace también con la desviación recortada 0,1 % (convención de TDA-07). Sólo hay etiqueta firme si ambas coinciden. *Nueva y exploratoria:* el roadmap sólo pide la primera.
4. **Regla de veredicto.** *Nueva y exploratoria.* Forma del estadístico y valores numéricos de los umbrales tomados de TDA-10 (los valores no se cambian; su aplicación a un estadístico normalizado sí es nueva, ver abajo): `ratio = (máx − mín)/mediana` de la semi-amplitud 1 %–99 % normalizada entre grupos; `SUPERPUESTO` si ratio e IC completo ≤ {PROFILE_STABILITY_SCALE_THRESHOLD}; `DIVERGENTE` si ≥ {PROFILE_STABILITY_FORM_THRESHOLD}; `INTERMEDIO` si el IC completo queda entre ambos; `NO_CONCLUYENTE` si el IC cruza un umbral o las dos normalizaciones discrepan. `ESCALA Y FORMA` si alguna agrupación es `DIVERGENTE`; `SÓLO ESCALA` si todas son `SUPERPUESTO`; en otro caso `NO CONCLUYENTE`. Además, si menos del {int(100 * ROBUSTNESS_AGREEMENT_FRACTION)} % de las 6 configuraciones causales de `sigma_hat` de TDA-10 reproduce la conclusión, se degrada a `NO CONCLUYENTE`.
5. **Participación en las comparaciones.** *Nueva y exploratoria.* Un grupo sólo entra en una comparación de forma si su cola en ese nivel tiene soporte `OK` (≥ {MIN_TAIL_SUPPORT_LOW} observaciones EN LA COLA, no en el grupo).
6. **`NO CONCLUYENTE` como salida.** *Nueva y exploratoria.* El roadmap sólo prevé `SÓLO ESCALA` y `ESCALA Y FORMA`. Esta etapa añade `NO CONCLUYENTE` como salida prudente cuando la evidencia no sostiene con firmeza ninguno de los dos. No es un tercer hallazgo sobre los datos: significa que la regla no pudo decidir.

**Qué significa `SUPERPUESTO`.** Describe la ESTABILIDAD DE LA SEMI-AMPLITUD NORMALIZADA 1 %–99 % entre grupos (ratio e IC completo ≤ {PROFILE_STABILITY_SCALE_THRESHOLD}). No demuestra que el perfil distribucional completo sea igual: no ve la asimetría, ni los percentiles interiores, ni la cola más allá del 1 %/99 %. Esos aspectos se reportan aparte, de forma descriptiva (§9, §10).

**Qué estaba aprobado y qué no (auditoría correctiva).** Reutilizar una constante no significa que su nuevo uso estuviera aprobado:

- Los umbrales {PROFILE_STABILITY_SCALE_THRESHOLD}/{PROFILE_STABILITY_FORM_THRESHOLD} se aprobaron en TDA-10 para el ratio de la semi-amplitud 1 %–99 % de `z` **sin dividir** por la desviación del grupo (fila `none` de §9, que reproduce exactamente el estadístico de TDA-10). Aplicarlos al ratio dividido por la desviación del grupo —lo que pide el roadmap de TDA-12— es una extensión decidida en esta etapa.
- El mínimo de {MIN_N_FOR_STABILITY_GROUP} era en TDA-10 un tamaño mínimo de GRUPO. Exigirlo a la COLA (decisión 5) es más estricto y es nuevo. §11.E muestra qué cambia con el criterio literal de TDA-10.
- La doble normalización (decisión 3) y la comprobación de robustez de la decisión 4 son nuevas. La segunda compara conclusiones puntuales de otras configuraciones con un veredicto basado en IC: no compara cosas equivalentes (§11.A lo detalla y da la comparación homogénea).
- **El estadístico del veredicto es ciego a la asimetría**: una distribución y su reflejo tienen la misma semi-amplitud. "`SUPERPUESTO`" significa, por tanto, *misma amplitud normalizada*, no *mismo perfil completo*. §9 añade la comparación de cada cola por separado, que se mantiene como evidencia DESCRIPTIVA: no forma parte del veredicto formal y no se incorporará a él sin aprobación específica. §11.E calcula qué diría una regla que la tuviera en cuenta; tampoco decide.

**Causalidad (verificada, bloqueante) — alcance exacto de lo ejecutado** (`{config.tda12_causality_csv_name}`).

- `sigma_hat`: la prueba de reconstrucción de TDA-10 se re-ejecutó en esta etapa para {len(cs)} configuración(es) — {cs_scope_txt}, elegidas con semilla fija, con `r_1m` como entrada — y {'pasa' if cs_all_passed else 'NO pasa'}. Es el estimador primario; el bloque `clock_adjusted` usa la misma familia y parámetro, y la prueba no se repitió con `r_tilde` como entrada. Las otras {n_sens_not_reverified} configuraciones de `sigma_hat` que aparecen en §11.A NO se re-verificaron aquí: su causalidad descansa en la verificación que hizo TDA-10.
- Decil causal: verificado sobre el mismo conjunto de barras elegibles con el que se construye (excluye el quemado), con dos pruebas — perturbar `sigma_hat` de una barra (no pueden cambiar los deciles de jornadas anteriores ni de las demás barras de su jornada) y truncar la muestra en esa jornada (los deciles deben ser idénticos sin conocer el futuro): {cz_ok} de {len(cz.get('checks', []))} comprobaciones superan ambas ({cz_scope_txt}, elegidas con semilla fija). Es una comprobación por muestreo sobre el estimador primario de cada bloque: no es exhaustiva y no se repitió para los deciles causales de las configuraciones alternativas de §11.A, que usan la misma función.

Matices declarados: los CORTES de los segmentos (TDA-06) se eligieron con toda la muestra, aunque la etiqueta de una barra sea reloj puro; y la desviación de grupo usada para normalizar es un descriptor de la muestra completa del grupo (así lo define el roadmap), no una cantidad disponible ex ante.

## 4. Cuantiles incondicionales

**`{r_name}`**

{_md_table(_global_tab(r_name), ["percentil", "q", "q_ci_lo", "q_ci_hi", "q_over_std", "n_tail", "n_days_tail", "support"], 6, {"q_over_std": 3})}

**`{z_name}`**

{_md_table(_global_tab(z_name), ["percentil", "q", "q_ci_lo", "q_ci_hi", "q_over_std", "n_tail", "n_days_tail", "support"], 4, {"q_over_std": 3})}

`q_over_std` = cuantil en múltiplos de la desviación de la muestra. Referencia: una normal tendría ±{_fmt(float(norm.ppf(0.99)), 2)} en los percentiles 1/99 y ±{_fmt(float(norm.ppf(0.999)), 2)} en 0,1/99,9.

## 5. Cuantiles por segmento horario

{_cond_section("SEGMENT")}

## 6. Cuantiles por estado previo de volatilidad (decil causal; 0 = más tranquilo, 9 = más volátil)

Tamaño de los estados (el decil causal no reparte por igual: compara cada barra con la historia disponible hasta el día anterior):

{_md_table(sizes, ["decil", "VOL_DECILE_CAUSAL", "VOL_DECILE_FULLSAMPLE"])}

{_cond_section("VOL_DECILE_CAUSAL")}

La misma tabla con las fronteras de TDA-10 (`VOL_DECILE_FULLSAMPLE`, retrospectivas) está en `{config.tda12_quantiles_csv_name}`.

## 7. Cuantiles por año

{_cond_section("YEAR")}

## 8. Fiabilidad de los cuantiles extremos (0,1 y 99,9)

De {n_ext} cuantiles extremos del bloque causal (muestra completa, segmentos, deciles causales y años; `r` y `z`): {n_ext_low} tienen soporte `BAJO` (< {MIN_TAIL_SUPPORT_LOW} observaciones de cola) y {n_ext_bad} `NO_FIABLE` (< {MIN_TAIL_SUPPORT_UNRELIABLE}). {('El peor caso: `' + str(worst['series']) + '`, ' + str(worst['grouping']) + ' = `' + _group_str(worst['group']) + '`, percentil ' + _pct(worst['level']) + ', sostenido por ' + _fmt(int(worst['n_tail'])) + ' observaciones de ' + _fmt(int(worst['n_days_tail'])) + ' jornadas.') if worst is not None else ''} En los percentiles 1 y 99 —los que usa el veredicto— {n_p1_notok} grupo(s)-cuantil no alcanzan soporte `OK` (y quedan fuera de la comparación).

**`{r_name}` por segmento**

{_md_table(_extreme_support_table(q, r_name, "SEGMENT"), ["grupo", "percentil", "n", "n_cola", "jornadas_cola", "cuantil", "IC95_lo", "IC95_hi", "semiancho_IC_relativo", "soporte"], 6, {"semiancho_IC_relativo": 3})}

**`{r_name}` por año**

{_md_table(_extreme_support_table(q, r_name, "YEAR"), ["grupo", "percentil", "n", "n_cola", "jornadas_cola", "cuantil", "IC95_lo", "IC95_hi", "semiancho_IC_relativo", "soporte"], 6, {"semiancho_IC_relativo": 3})}

**`{z_name}` por decil causal**

{_md_table(_extreme_support_table(q, z_name, "VOL_DECILE_CAUSAL"), ["grupo", "percentil", "n", "n_cola", "jornadas_cola", "cuantil", "IC95_lo", "IC95_hi", "semiancho_IC_relativo", "soporte"], 4, {"semiancho_IC_relativo": 3})}

**Cómo leerlo.** Un percentil 0,1 o 99,9 es, por definición, el valor que deja fuera una observación de cada mil: en un grupo de 85.000 barras descansa en ~85 puntos, que además se concentran en pocas jornadas. Esos cuantiles son órdenes de magnitud, no valores precisos; `semiancho_IC_relativo` dice cuánto podría moverse (y, por lo dicho en §3, es una cota optimista). El resto de combinaciones está en `{config.tda12_quantiles_csv_name}`.

## 9. Escala frente a forma

Si la distribución sólo cambiara de escala entre grupos, al dividir los cuantiles de cada grupo por su propia desviación todos los perfiles coincidirían. Se mide con la semi-amplitud `(q_alto − q_bajo)/2` normalizada de cada grupo, y `ratio = (máx − mín)/mediana` entre grupos (0 = misma semi-amplitud normalizada en todos los grupos). La implicación va en un solo sentido: sólo escala ⇒ misma semi-amplitud; la misma semi-amplitud NO garantiza el mismo perfil. Por eso `SUPERPUESTO` en las tablas se lee como "semi-amplitud normalizada estable", no como "distribuciones iguales". Ver `{config.tda12_profiles_png_name}` (perfiles dibujados juntos) y `{config.tda12_half_width_png_name}` (semi-amplitud grupo a grupo con IC).

**Percentiles 1 %–99 % — `{z_name}` (DECIDE el veredicto)**

{_md_table(_shape_rows(shape, z_name, VERDICT_LEVEL_PAIR, verdict_groupings_all), shape_cols, 3)}

**Cada cola por separado y asimetría entre grupos — `{z_name}`, percentiles 1 % y 99 % (DESCRIPTIVO, no decide)**

La semi-amplitud promedia las dos colas. Esta tabla aplica el mismo ratio a `|q1|/escala` (cola izquierda) y a `q99/escala` (cola derecha), y da el rango (máx − mín) de la asimetría entre grupos:

{_md_table(_shape_rows(shape, z_name, VERDICT_LEVEL_PAIR, VERDICT_GROUPINGS), side_cols, 3)}

Las filas `none` no dividen por la desviación del grupo (cuantiles de `z` tal cual): miden si `sigma_hat` por sí solo deja a todos los grupos en la misma escala. Son descriptivas, no deciden: {none_txt}.

**Percentiles 1 %–99 % — `{r_name}` (contexto, NO decide)**

{_md_table(_shape_rows(shape, r_name, VERDICT_LEVEL_PAIR, verdict_groupings_all), shape_cols, 3)}

**Otros niveles — `{z_name}` (descriptivo, sin umbral)**

Percentiles 5 %–95 %:

{_md_table(_shape_rows(shape, z_name, (0.05, 0.95), VERDICT_GROUPINGS), shape_cols_nolabel, 3)}

Percentiles 25 %–75 %:

{_md_table(_shape_rows(shape, z_name, (0.25, 0.75), VERDICT_GROUPINGS), shape_cols_nolabel, 3)}

Percentiles 0,1 %–99,9 % (cola profunda; sólo grupos con soporte `OK` — ver columna `excluidos`):

{_md_table(far_rows, shape_cols_nolabel, 3)}

**Nota sobre el IC del ratio.** `(máx − mín)/mediana` es un estadístico de rango: el ruido de muestreo sólo puede aumentarlo, así que en las réplicas bootstrap tiende a salir algo mayor que en la muestra y el IC percentil puede quedar por encima del valor puntual. La etiqueta exige que el punto Y el IC completo estén en la misma zona, lo que es conservador para declarar `SUPERPUESTO`.

## 10. Asimetría de colas

`asimetría = (|q_bajo| − q_alto) / media de ambos`: positivo = cola izquierda (caídas) más larga que la derecha. Muestra completa:

{_md_table(asym_global, ["series", "par", "asymmetry", "asymmetry_ci_lo", "asymmetry_ci_hi"], 4)}

En los pares 1/99 y 5/95, sobre {n_asym} combinaciones grupo × serie del bloque causal, {n_asym_excl0} tienen un IC que excluye el cero ({n_asym_left} con la cola izquierda más larga). {('La mayor en valor absoluto: `' + str(asym_abs_max['series']) + '`, ' + str(asym_abs_max['grouping']) + ' = `' + _group_str(asym_abs_max['group']) + '`, par ' + str(asym_abs_max['par']) + ': ' + _fmt(asym_abs_max['asymmetry'], 3) + ' [' + _fmt(asym_abs_max['asymmetry_ci_lo'], 3) + ', ' + _fmt(asym_abs_max['asymmetry_ci_hi'], 3) + '].') if asym_abs_max is not None else ''} Que un IC excluya el cero con casi dos millones de observaciones dice que la asimetría es distinguible del ruido, no que sea grande: la magnitud es la que figura en la tabla. Detalle por grupo en `{config.tda12_asymmetry_csv_name}`. Coherente con TH25 (TDA-11): asimetría presente pero pequeña.

## 11. Robustez

**A. Estimador de volatilidad** (las 6 configuraciones causales de TDA-10; ratios puntuales del nivel 1 %–99 %, sin bootstrap):

{_md_table(sens, sens_cols, 3)}

Comparación homogénea (puntual contra puntual, informativa): {_fmt(th26.get('estimator_point_agreement_with_primary_point'), 2)} de las configuraciones da la misma conclusión puntual que el primario. Conclusiones PUNTUALES (sin IC) de las {th26['n_estimator_configs']} configuraciones: {point_counts_txt}. La del primario es `{VERDICT_DISPLAY[primary_point]}`; su veredicto formal, que además exige que el IC completo esté en la misma zona, es `{before_txt}`. Fracción de configuraciones cuya conclusión puntual coincide con ese veredicto formal: {_fmt(th26['estimator_agreement_fraction'], 2)} (umbral de robustez de TDA-10: {ROBUSTNESS_AGREEMENT_FRACTION}; sólo puede DEGRADAR un veredicto firme a `NO CONCLUYENTE`, nunca crear uno).

**B. Fronteras de los deciles.** La fila `VOL_DECILE_FULLSAMPLE` de §9 repite la comparación con las fronteras retrospectivas de TDA-10.

**C. Bloque `clock_adjusted` (RETROSPECTIVO, nunca decide)** — misma comparación sobre `{z_adj_name}`:

{_md_table(_shape_rows(shape, z_adj_name, VERDICT_LEVEL_PAIR, VERDICT_GROUPINGS), shape_cols, 3)}

Con la misma regla, este bloque daría `{adj_before}` ({", ".join(f"{gk}: {adj_labels[gk]}" for gk in VERDICT_GROUPINGS)}). Sirve para interpretar: indica cuánto de la diferencia entre segmentos del bloque causal se debe al patrón horario determinista que `sigma_hat` causal no conoce.

**D. Longitud del bloque del bootstrap** (descriptivo; el procedimiento aprobado es el bloque de 1 jornada). Se repite el bootstrap con bloques de {BLOCK_LENGTH_SENSITIVITY_DAYS} jornadas CONSECUTIVAS, que conservan la dependencia entre días vecinos. Estadístico del veredicto sobre `{z_name}`:

{_md_table(bs_ratio, ["grouping", "normalization", "bloque (jornadas)", "point", "IC95", "ancho IC ÷ ancho con 1 jornada", "etiqueta con ese bloque"], 3)}

Grupos que marcan el mínimo y el máximo de la comparación por segmento (`{z_name}`):

{_md_table(bs_focus, ["group", "statistic", "normalization", "bloque (jornadas)", "point", "IC95", "ancho IC ÷ ancho con 1 jornada"], 3)}

{block_fact} Tabla completa (incluye `{r_name}`) en `{config.tda12_block_sensitivity_csv_name}`.

**E. Lecturas alternativas del estadístico y propuesta** (descriptivo: sólo la primera fila es la regla vigente). Muestra cuánto depende el resultado de las decisiones nuevas de esta etapa:

{_md_table(readings, ["lectura", "estado"] + list(VERDICT_GROUPINGS) + ["resultado (sin degradación por robustez)"])}

De las {n_readings_alt} lecturas alternativas, {alt_same} dan el mismo resultado que la regla vigente. La PROPUESTA (exigir la clasificación en cada cola por separado, para que la regla no sea ciega a la asimetría) daría `{VERDICT_DISPLAY[proposal['verdict_if_adopted']]}`; no está aprobada y no interviene en el veredicto.

**Tratamiento de 2019.** La comparación por año usa el {_fmt(100 * year_cov, 1)} % de las observaciones; queda fuera `{_group_str(year_excl) if year_excl else '—'}` (7.406 barras, año parcial) porque sus colas del 1 % tienen menos de {MIN_TAIL_SUPPORT_LOW} observaciones. La etiqueta de la agrupación por año describe, por tanto, los años comparados, no 2019. Con el criterio literal de TDA-10 (tamaño de grupo ≥ {MIN_N_FOR_STABILITY_GROUP}), que sí lo incluye: {year_t10_txt}.

## 12. Veredicto TH26

{_md_table(detail, ["agrupación", "÷std: ratio [IC95]", "label_std", "÷std recortada: ratio [IC95]", "label_std_trimmed", "label"], 3)}

**Veredicto: `{verdict_txt}`** (sobre `{z_name}`, regla de §3.4).{' Antes de la comprobación de robustez a estimador la regla daba `' + before_txt + '`; se degrada porque sólo ' + _fmt(th26['estimator_agreement_fraction'], 2) + ' de las configuraciones coincide.' if th26['verdict'] != th26['verdict_before_robustness'] else ''}

`SUPERPUESTO` = semi-amplitud normalizada 1 %–99 % estable entre grupos; no es una demostración de igualdad del perfil completo (§3). El veredicto está condicionado a las decisiones nuevas y exploratorias de §3 (umbrales aplicados al estadístico normalizado, soporte exigido a la cola, doble normalización, regla de robustez, `NO CONCLUYENTE` como salida).

**Dónde no hay estabilidad firme de la semi-amplitud:**

{why_txt}

{'**Por qué no se fuerza una conclusión.** El roadmap prevé dos resultados (`SÓLO ESCALA` / `ESCALA Y FORMA`). La regla fijada de antemano exige evidencia firme para cualquiera de los dos; aquí al menos una agrupación no la da (su IC cruza un umbral, o las dos normalizaciones discrepan, o queda en la zona intermedia). No se interpreta la falta de divergencia como demostración de igualdad, ni al revés.' if th26['verdict'] == VERDICT_INCONCLUSIVE else ''}

## 13. Consecuencias para `STOP-13` (sin ejecutar TDA-13)

{stop13['text']}

- `STOP-13` activado por TDA-12: **{'SÍ' if stop13['stop13_activated_by_tda12'] else 'NO'}**.
- Criterio de entrada 1 de EVT (TDA-12 = `ESCALA Y FORMA`) cumplido: **{'SÍ' if stop13['evt_entry_criterion_1_met'] else 'NO'}**.

TDA-13 parte A (inventario, triage y agrupamiento de extremos) es obligatoria con cualquier resultado. Esta etapa no la inicia.

## 14. Limitaciones

1. Los IC bootstrap remuestrean jornadas como si fueran intercambiables; con volatilidad persistente entre días, la incertidumbre real de los cuantiles extremos es probablemente mayor (§11.D lo cuantifica con bloques más largos).
2. Los cuantiles 0,1 y 99,9 por grupo descansan en pocas decenas o centenas de observaciones concentradas en pocas jornadas (§8): son órdenes de magnitud. Ningún cuantil extrapola más allá de lo observado.
3. El veredicto usa el nivel 1 %–99 % (el único con umbral documentado, de TDA-10). Diferencias de forma que sólo aparezcan más allá del percentil 1/99 —la cola profunda— no las decide esta regla; se reportan en §9 sin umbral y son materia de TDA-13 parte A.
4. `SUPERPUESTO` describe la estabilidad de la semi-amplitud normalizada, no la igualdad del perfil distribucional completo. El estadístico no ve la asimetría: dos grupos con la misma amplitud y asimetrías opuestas serían `SUPERPUESTO`. La comparación de cada cola por separado (§9) lo suple sólo de forma descriptiva y no forma parte del veredicto formal.
5. Los umbrales {PROFILE_STABILITY_SCALE_THRESHOLD} / {PROFILE_STABILITY_FORM_THRESHOLD} se aprobaron en TDA-10 para el estadístico SIN normalizar por la desviación del grupo. Aplicarlos al estadístico normalizado de TDA-12 es una decisión nueva y exploratoria; no existe una justificación teórica de esos valores para este uso.
6. Exigir {MIN_TAIL_SUPPORT_LOW} observaciones en la COLA (en TDA-10 eran {MIN_N_FOR_STABILITY_GROUP} observaciones totales por grupo) es una decisión nueva y exploratoria. Deja fuera de la comparación por año al año parcial 2019 (§11.E).
7. La doble normalización y la regla de robustez son decisiones nuevas y exploratorias. El resultado depende de la normalización (§11.E: con sólo la desviación recortada la regla daría otro resultado), y la regla de robustez compara conclusiones puntuales con conclusiones basadas en IC, que no son equivalentes (§11.A).
8. `NO CONCLUYENTE` no es uno de los dos resultados que prevé el roadmap: es una salida prudente, nueva y exploratoria, para cuando no puede sostenerse ninguno. No equivale a "la forma es igual" ni a "la forma cambia".
9. El decil causal de volatilidad es también una decisión nueva de esta etapa (§3); los grupos resultantes no tienen el mismo tamaño y al principio de la muestra descansan en poca historia.
10. Las pruebas de causalidad son comprobaciones por muestreo (§3): se ejecutaron sobre el estimador primario, en un número reducido de barras elegidas con semilla fija. No se re-verificaron en esta etapa las demás configuraciones de `sigma_hat` de §11.A (su verificación es la de TDA-10) ni sus deciles causales.
11. La sensibilidad a la longitud de bloque (§11.D) usa dos longitudes elegidas en esta etapa; bloques más largos dejan pocos bloques independientes y sus IC son a su vez inciertos.
12. El estadístico de rango `(máx − mín)/mediana` crece con el número de grupos y está sesgado al alza por ruido de muestreo; no es comparable entre agrupaciones con distinto número de grupos.
13. Los cortes de los segmentos horarios provienen de TDA-06 (toda la muestra).
14. `z_t` depende del estimador de volatilidad; §11.A muestra cuánto cambia la conclusión con las otras configuraciones de TDA-10.
15. No se ajustó regresión cuantílica: los tres condicionantes del roadmap son categóricos o se discretizan en deciles, y no surgió ninguna pregunta que exigiera una variable continua (G4).

## 15. Tiempos

| Etapa | Tiempo |
|---|---:|
{stage_lines}

**Tiempo total (análisis + escritura): {elapsed_total_seconds:.1f}s (~{elapsed_total_seconds / 60:.1f} min).** CPU, un proceso; sin GPU.

## 16. Archivos generados

`{config.tda12_quantiles_csv_name}` (todos los cuantiles: 4 series × 5 agrupaciones, con IC, normalizados, `n`, soporte), `{config.tda12_shape_csv_name}`, `{config.tda12_asymmetry_csv_name}` (asimetría y semi-amplitud por grupo), `{config.tda12_sensitivity_csv_name}`, `{config.tda12_causality_csv_name}`, `{config.tda12_decile_sizes_csv_name}`, `{config.tda12_block_sensitivity_csv_name}`, `{config.tda12_readings_csv_name}` (8 CSV) + `{config.tda12_profiles_png_name}`, `{config.tda12_half_width_png_name}` (2 PNG) + este informe.

## 17. Modo sencillo

**¿Qué preguntó esta etapa?** Los movimientos de MNQ son más grandes a unas horas que a otras, en días agitados que en días tranquilos, y en unos años que en otros. La pregunta es si eso es sólo "el mismo dibujo con otro tamaño" (escala) o si el dibujo mismo cambia (forma): por ejemplo, que en ciertos contextos los movimientos extremos sean proporcionalmente más frecuentes.

**¿Cómo se midió?** Para cada contexto se calcularon los percentiles de los movimientos y se dividieron por el tamaño típico de ese contexto. Si sólo cambia el tamaño, todos los contextos quedan con el mismo perfil.

**¿Qué salió?** {plain_verdict}

**¿Qué quiere decir `SUPERPUESTO` en las tablas?** Que, una vez descontado el tamaño típico, la anchura entre los percentiles 1 y 99 es prácticamente la misma en todos los grupos comparados. Es una medida concreta de anchura; no prueba que las distribuciones sean idénticas en todos sus detalles.

**¿Qué partes del método son nuevas?** La regla que decide, los umbrales aplicados a esta medida, el mínimo de observaciones exigido en la cola, el uso de dos normalizaciones, la comprobación de robustez y la propia salida `NO CONCLUYENTE` se fijaron en esta etapa y son exploratorias (§3, §14).

**¿Qué tan fiables son los números extremos?** Los percentiles 1 y 99 están bien sostenidos. Los percentiles 0,1 y 99,9 de un contexto concreto se apoyan en pocas observaciones de pocos días: sirven como orden de magnitud, no como cifra exacta.

**¿Qué NO significa?** No es una señal ni una regla de operación, no predice nada y no decide ningún aspecto del diseño de IRIS. Tampoco ejecuta el análisis de eventos extremos (TDA-13): sólo deja escrito en qué estado queda su condición de entrada (§13).

## 18. Estado de cierre

**TDA-12 = `PASS_WITH_OPEN_QUESTIONS` / `CLOSED`** — cerrada tras la auditoría correctiva y el cierre documental, con cuestiones metodológicas abiertas y documentadas (§3 y §14).

| Elemento | Estado al cierre |
|---|---|
| TH26 | `{verdict_txt}` |
| `STOP-13` | {'activado' if stop13['stop13_activated_by_tda12'] else 'no activado'} por TDA-12 |
| EVT (TDA-13 parte B) | criterio de entrada 1 {'cumplido' if stop13['evt_entry_criterion_1_met'] else 'no cumplido; no autorizada'} |
| TDA-13 parte A | siguiente tarea obligatoria; no iniciada en esta etapa |

**Cuestiones abiertas que el cierre NO resuelve.** Las decisiones nuevas y exploratorias de §3 (umbrales de TDA-10 aplicados al estadístico normalizado, soporte exigido a la cola, doble normalización, regla de robustez, `NO CONCLUYENTE` como salida) no quedan validadas por cerrar la etapa: el estado de TH26 se lee condicionado a ellas y con las limitaciones de §14. La comparación de cada cola por separado sigue siendo descriptiva y no forma parte del veredicto. Nada de lo anterior es una conclusión estadísticamente demostrada sobre la igualdad o la diferencia de forma.
"""


def persist_artifacts(result: TDA12Result, config: SnapshotConfig, t0: float, run_command: str) -> list[Path]:
    config.reports_dir.mkdir(parents=True, exist_ok=True)

    result.quantile_table.to_csv(config.tda12_quantiles_csv_path, index=False)
    result.shape_table.to_csv(config.tda12_shape_csv_path, index=False)
    result.asymmetry_table.to_csv(config.tda12_asymmetry_csv_path, index=False)

    sens = result.estimator_sensitivity.copy()
    sens["verdict_primary_before_robustness"] = result.th26["verdict_before_robustness"]
    sens["verdict_final"] = result.th26["verdict"]
    sens["estimator_agreement_fraction"] = result.th26["estimator_agreement_fraction"]
    sens.to_csv(config.tda12_sensitivity_csv_path, index=False)

    caus = result.causality_sigma.copy()
    caus.insert(0, "check", "sigma_hat_no_lookahead (TDA-10)")
    dec = pd.DataFrame(result.causality_decile.get("checks", []))
    if not dec.empty:
        dec.insert(0, "check", "causal_volatility_decile_no_lookahead")
        dec["passed"] = dec["earlier_days_unchanged"] & dec["same_day_other_bars_unchanged"] & dec["truncation_identical"]
    pd.concat([caus, dec], ignore_index=True).to_csv(config.tda12_causality_csv_path, index=False)

    result.decile_sizes.to_csv(config.tda12_decile_sizes_csv_path, index=False)
    result.block_sensitivity.to_csv(config.tda12_block_sensitivity_csv_path, index=False)
    result.alternative_readings.to_csv(config.tda12_readings_csv_path, index=False)

    _draw_profiles(result, config.tda12_profiles_png_path)
    _draw_half_widths(result, config.tda12_half_width_png_path)

    report_text = render_report(result, config, time.perf_counter() - t0, run_command)
    config.tda12_report_path.write_text(report_text, encoding="utf-8")
    return [getattr(config, attr) for attr in ARTIFACT_PATH_ATTRS]


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Ejecuta TDA-12 (cuantiles condicionales, TH26) sobre las variables de TDA-04/06.")
    parser.add_argument("--config", default="configs/mnq_snapshot.yaml")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))
    run_command = "python -m ohlcv_dataroad.ingest.run_tda12 --config " + str(args.config)

    t0 = time.perf_counter()
    try:
        result = run_tda12_analysis(config, verbose=True)
    except (
        FileNotFoundError, HoldoutIsolationError, TimestampAlignmentError, RTildeInvariantError,
        PopulationMismatchError, LookaheadLeakageError, CausalDecileLeakageError,
    ) as exc:
        print("TDA-12 status: FAIL", file=sys.stderr)
        print(f"Motivo: {exc}", file=sys.stderr)
        return 1

    print(f"[TDA12 {TOTAL_STAGES}/{TOTAL_STAGES}] Persistencia de artefactos (CSV/PNG/MD)...")
    t_write = time.perf_counter()
    written = persist_artifacts(result, config, t0, run_command)
    total = time.perf_counter() - t0
    print(f"[TDA12 {TOTAL_STAGES}/{TOTAL_STAGES}] Persistencia de artefactos -- completado en {time.perf_counter() - t_write:.1f}s (total: {total:.1f}s)")

    print()
    print("TDA-12 status: analisis completado")
    print(f"n poblacion: {result.n_population:,} en {result.n_days:,} jornadas; n_boot={result.n_boot}")
    print(f"TH26 = {result.th26['verdict']} (antes de robustez a estimador: {result.th26['verdict_before_robustness']})")
    for gk, lab in result.th26["labels_by_grouping"].items():
        print(f"  {gk}: {lab}")
    print(f"STOP-13 activado por TDA-12: {result.stop13['stop13_activated_by_tda12']}")
    print(f"Tiempo total: {total:.1f}s (~{total / 60:.1f} min)")
    print()
    print("Artefactos escritos:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
