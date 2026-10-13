"""Punto de entrada de linea de comandos para ejecutar TDA-13 Parte A.

Uso
---
    python -m ohlcv_dataroad.ingest.run_tda13
    python -m ohlcv_dataroad.ingest.run_tda13 --config configs/mnq_snapshot.yaml

Como en TDA-00..12, este script no contiene logica de analisis propia:
llama a ``run_tda13_analysis`` (``tda13_extremes_inventory.py``) y
persiste TODA la evidencia (``persist_artifacts``), incluido el informe
``TDA13_extremos_inventario_agrupamiento.md``, generado automaticamente a
partir de los resultados (``render_report``). La Parte B (EVT) no existe
en este codigo.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from ohlcv_dataroad.config import SnapshotConfig, load_config
from ohlcv_dataroad.ingest.holdout_guard import HoldoutIsolationError
from ohlcv_dataroad.ingest.tda07_marginal_distribution import NaiveReturnContradictionError, TimestampAlignmentError
from ohlcv_dataroad.ingest.tda10_scale_vs_shape import LookaheadLeakageError
from ohlcv_dataroad.ingest.tda12_conditional_quantiles import CI_LEVEL, MIN_TAIL_SUPPORT_LOW, MIN_TAIL_SUPPORT_UNRELIABLE, CausalDecileLeakageError
from ohlcv_dataroad.ingest.tda13_extremes_inventory import (
    CAT_BOUNDARY,
    CAT_ERROR,
    CAT_REAL,
    CAT_ROLL,
    CAT_UNDETERMINED,
    CATEGORIES,
    CLAIM_INCONCLUSIVE,
    CLAIM_NO,
    CLAIM_YES,
    CLUSTER_WINDOWS,
    DIAG_BERNOULLI_KNOWN,
    DIAG_BERNOULLI_REFIT,
    G2_NOMINAL_RATE,
    G2_POWER_RATES,
    G2_STATUS_COMPATIBLE,
    G2_STATUS_INCOMPATIBLE,
    G2_STATUS_NOT_EVALUABLE,
    NULL_FAMILY_MINUTE_YEAR,
    SMALL_GROUP_SIZE,
    permutation_band_size_bound,
    band_order_statistic,
    criterion_non_rejection_probability,
    g2_status,
    INDEPENDENCE_FAMILIES,
    KIND_BERNOULLI_DIAG,
    KIND_FULL_BOOTSTRAP,
    NULL_CLOCK_YEAR_LEGACY,
    TH06_BASIS,
    VOLATILITY_FAMILIES,
    DEF_GLOBAL,
    DEF_SEGMENT,
    DEFINITIONS,
    LOW_VOLUME_QUANTILE_TDA03,
    MACRO_CLOCK_WINDOW_MINUTES,
    NULL_CLOCK,
    NULL_HOMOGENEOUS,
    NULL_PERCENTILE,
    NULLS,
    POP_CROSSING,
    POP_ELIGIBLE,
    PRIMARY_DEFINITION,
    PRIMARY_ESTIMATOR,
    PRIMARY_LEVEL,
    PRIMARY_NULL,
    PRIMARY_WINDOW,
    REVERSAL_FRACTION_TDA03,
    SENSITIVITY_LEVELS,
    SERIES_R,
    SERIES_Z,
    SIDE_BOTH,
    SIDE_NEG,
    SIDE_POS,
    SIDES,
    TH06_MAJORITY,
    TH27_EXPLAINED,
    TH27_INCONCLUSIVE,
    TH27_NONE,
    TH27_PERSISTS,
    TOTAL_STAGES,
    TRIAGE_RULES_TEXT,
    MissingEvidenceError,
    TDA13Result,
    run_tda13_analysis,
)

ARTIFACT_PATH_ATTRS = [
    "tda13_extremes_csv_path",
    "tda13_thresholds_csv_path",
    "tda13_triage_summary_csv_path",
    "tda13_clustering_csv_path",
    "tda13_gap_distribution_csv_path",
    "tda13_run_distribution_csv_path",
    "tda13_clustering_by_group_csv_path",
    "tda13_hourly_csv_path",
    "tda13_triage_impact_csv_path",
    "tda13_calibration_csv_path",
    "tda13_reference_conservation_csv_path",
    "tda13_false_detection_csv_path",
    "tda13_full_bootstrap_csv_path",
    "tda13_triage_sensitivity_csv_path",
    "tda13_volatility_control_csv_path",
    "tda13_clustering_png_path",
    "tda13_gap_distribution_png_path",
    "tda13_hourly_png_path",
    "tda13_triage_png_path",
    "tda13_report_path",
]
# 15 CSV + 4 PNG + 1 MD = 20 artefactos, siempre los mismos. (`TDA13_auditoria_resultados_previos.csv` es una foto fija de entrada: se lee, no se escribe.)

SIDE_TITLES = {SIDE_NEG: "caídas (cola negativa)", SIDE_POS: "subidas (cola positiva)", SIDE_BOTH: "ambas colas juntas"}
DEF_TITLES = {DEF_GLOBAL: "umbral único global (anti-patrón, sólo contraste)", DEF_SEGMENT: "umbral por segmento horario",
              PRIMARY_DEFINITION: "umbral por segmento × estado de volatilidad (PRIMARIA)"}
TH27_DISPLAY = {
    TH27_NONE: "SIN AGRUPAMIENTO DETECTABLE", TH27_EXPLAINED: "AGRUPAMIENTO NO DETECTABLE TRAS ESTANDARIZAR",
    TH27_PERSISTS: "AGRUPAMIENTO PERSISTE CON EL EWMA; ORIGEN NO IDENTIFICADO", TH27_INCONCLUSIVE: "INCONCLUSO",
}
CLAIM_DISPLAY = {CLAIM_YES: "DETECTADO", CLAIM_NO: "NO DETECTADO", CLAIM_INCONCLUSIVE: "INCONCLUSO", "NO_EVALUADA": "NO EVALUADA"}
STAT_PRIMARY = f"share_le_{PRIMARY_WINDOW}"


def _fmt(x, nd: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, str):
        return x
    if isinstance(x, (bool, np.bool_)):
        return "sí" if bool(x) else "no"
    try:
        if isinstance(x, (int, np.integer)):
            return f"{int(x):,}"
        xf = float(x)
        if np.isnan(xf):
            return "NaN"
        return f"{xf:,.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _md_table(df: pd.DataFrame, columns: list[str], float_nd: int = 4, nd_by_col: dict | None = None) -> str:
    if df is None or len(df) == 0:
        return "_(sin datos)_"
    nd_by_col = nd_by_col or {}
    cols = [c for c in columns if c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---:"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c], nd_by_col.get(c, float_nd)) for c in cols) + " |")
    return "\n".join(lines)


def _pct(x: float, nd: int = 2) -> str:
    return "NaN" if not np.isfinite(x) else f"{100 * x:.{nd}f} %"


def _ratio_ci(row) -> str:
    return f"{_fmt(row['ratio'], 2)} [{_fmt(row['ratio_ci_lo'], 2)}, {_fmt(row['ratio_ci_hi'], 2)}]"


def _cluster_view(cluster: pd.DataFrame, statistic: str = STAT_PRIMARY, **filters) -> pd.DataFrame:
    sub = cluster.loc[cluster["statistic"] == statistic]
    for key, val in filters.items():
        sub = sub.loc[sub[key].isin(val) if isinstance(val, (list, tuple)) else (sub[key] == val)]
    out = sub.copy()
    out["serie"] = out["series"]
    out["definición"] = out["definition"] if "definition" in out else ""
    out["lado"] = out["side"]
    out["observado"] = [_pct(v) for v in out["observed"]]
    out["esperado si independientes"] = [_pct(v) for v in out["expected"]]
    out["cociente [IC95]"] = [_ratio_ci(r) for _, r in out.iterrows()]
    out["cociente [IC95 condicional]"] = out["cociente [IC95]"]
    out["banda Bernoulli (previa, no decide)"] = [f"[{_fmt(a, 2)}, {_fmt(b, 2)}]" if np.isfinite(a) else "—" for a, b in zip(out["null_sim_p2_5"], out["null_sim_p97_5"])]
    out["excedencias"] = out["n_exceed"]
    out["tiempos medidos"] = out["n_intervals"]
    return out


# ---------------------------------------------------------------------------
# Graficos
# ---------------------------------------------------------------------------

def _draw_clustering(result: TDA13Result, out_path: Path) -> None:
    """Cociente observado/esperado de la fraccion de tiempos entre excedencias <= ventana primaria: r frente a z, por definicion de extremo y lado."""
    c = result.cluster_table
    sub = c.loc[(c["statistic"] == STAT_PRIMARY) & (c["level"] == PRIMARY_LEVEL) & (c["reference"] == PRIMARY_NULL) & (c["z_halflife"] == PRIMARY_ESTIMATOR[1])]
    fig, axes = plt.subplots(1, len(SIDES), figsize=(5.6 * len(SIDES), 4.6), squeeze=False)
    for ax, side in zip(axes[0], SIDES):
        x = np.arange(len(DEFINITIONS))
        for series, color, dx in ((SERIES_R, "tab:blue", -0.17), (SERIES_Z, "tab:orange", 0.17)):
            rows = [sub.loc[(sub["series"] == series) & (sub["side"] == side) & (sub["definition"] == d)] for d in DEFINITIONS]
            y = np.array([r["ratio"].iloc[0] if len(r) else np.nan for r in rows])
            lo = np.array([r["ratio_ci_lo"].iloc[0] if len(r) else np.nan for r in rows])
            hi = np.array([r["ratio_ci_hi"].iloc[0] if len(r) else np.nan for r in rows])
            yerr = np.nan_to_num(np.vstack([np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)]))
            ax.bar(x + dx, y, width=0.32, color=color, yerr=yerr, capsize=3, label="retorno crudo r" if series == SERIES_R else "retorno estandarizado z")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9, label="independencia (=1)")
        ax.set_xticks(x)
        ax.set_xticklabels(["GLOBAL", "SEGMENTO", "SEGMENTO×VOL"], fontsize=8)
        ax.set_title(SIDE_TITLES[side], fontsize=9)
        ax.set_ylabel(f"cociente observado / esperado (tiempos ≤ {PRIMARY_WINDOW} barras)")
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=7)
    fig.suptitle(
        f"TDA-13A -- agrupamiento de extremos (nivel {100 * PRIMARY_LEVEL:g} % por cola). Cociente > 1 = los extremos llegan más juntos que si fueran independientes.\n"
        f"Referencia: independencia con la misma intensidad por minuto del reloj y por año. Barras de error: IC 95 % CONDICIONAL (umbrales y referencia fijos). RETROSPECTIVO."
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _draw_gaps(result: TDA13Result, out_path: Path) -> None:
    g = result.gap_distribution
    sub = g.loc[(g["definition"] == PRIMARY_DEFINITION) & (g["reference"] == PRIMARY_NULL)]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), squeeze=False)
    for i, series in enumerate((SERIES_R, SERIES_Z)):
        for j, side in enumerate((SIDE_NEG, SIDE_POS)):
            ax = axes[i][j]
            s = sub.loc[(sub["series"] == series) & (sub["side"] == side)]
            x = np.arange(len(s))
            ax.bar(x - 0.2, s["share_observed"], width=0.4, color="tab:blue", label="observado")
            ax.bar(x + 0.2, s["share_expected"], width=0.4, color="lightgray", edgecolor="gray", label="esperado si independientes")
            ax.set_xticks(x)
            ax.set_xticklabels(list(s["gap_bars"]), fontsize=8)
            ax.set_yscale("log")
            ax.set_xlabel("barras de 1 minuto hasta la siguiente excedencia (mismo bloque continuo)")
            ax.set_ylabel("fracción de los tiempos")
            ax.set_title(f"{series} — {SIDE_TITLES[side]}", fontsize=9)
            ax.grid(alpha=0.25, axis="y")
            ax.legend(fontsize=7)
    fig.suptitle("TDA-13A -- tiempos entre excedencias: observado frente a independencia de la misma intensidad (definición y nivel primarios). RETROSPECTIVO.")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _draw_hourly(result: TDA13Result, out_path: Path) -> None:
    h = result.hourly
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), squeeze=False)
    for ax, series in zip(axes[0], (SERIES_R, SERIES_Z)):
        for definition, color in zip(DEFINITIONS, ("tab:red", "tab:green", "tab:blue")):
            s = h.loc[(h["series"] == series) & (h["definition"] == definition) & (h["side"] == SIDE_BOTH)].sort_values("hour_ny")
            if s.empty:
                continue
            ax.plot(s["hour_ny"], 100 * s["rate"], marker="o", markersize=3, color=color, label=definition)
            ax.fill_between(s["hour_ny"], 100 * s["rate_ci_lo"], 100 * s["rate_ci_hi"], color=color, alpha=0.15)
        ax.axhline(100 * 2 * PRIMARY_LEVEL, color="black", linestyle="--", linewidth=0.8, label="tasa objetivo")
        ax.set_xlabel("hora de Nueva York")
        ax.set_ylabel("% de barras que son extremo (ambas colas)")
        ax.set_title(series, fontsize=9)
        ax.set_xticks(range(0, 24, 2))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("TDA-13A -- distribución horaria de los extremos según cómo se define el umbral. Un umbral global selecciona horas; uno relativo, no. RETROSPECTIVO.")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _draw_triage(result: TDA13Result, out_path: Path) -> None:
    t = result.triage_summary
    colors = {CAT_ERROR: "black", CAT_ROLL: "tab:purple", CAT_BOUNDARY: "tab:brown", CAT_REAL: "tab:green", CAT_UNDETERMINED: "tab:gray"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), squeeze=False)
    for ax, grouping, title in zip(axes[0], ("YEAR", "SEGMENT"), ("año", "segmento horario")):
        s = t.loc[t["grouping"] == grouping]
        groups = list(dict.fromkeys(s["group"]))
        bottom = np.zeros(len(groups))
        for cat in CATEGORIES:
            vals = np.array([s.loc[(s["group"] == g) & (s["category"] == cat), "n"].sum() for g in groups], dtype=float)
            ax.bar(range(len(groups)), vals, bottom=bottom, color=colors[cat], label=cat)
            bottom += vals
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("número de extremos")
        ax.set_title(f"Triage por {title}", fontsize=9)
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=7)
    fig.suptitle(f"TDA-13A -- clasificación de los extremos inventariados (nivel {100 * PRIMARY_LEVEL:g} % por cola).")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------

def _impact_wide(impact: pd.DataFrame, series: str) -> pd.DataFrame:
    sub = impact.loc[impact["series"] == series]
    rows = []
    for scen, s in sub.groupby("scenario", sort=False):
        s = s.set_index("statistic")
        row = {"escenario": scen, "n": int(s.loc["n", "value"]), "quitadas": int(s["n_removed_vs_baseline"].iloc[0])}
        for stat, name, nd in (("kurtosis_excess", "curtosis", 1), ("kurtosis_excess_trimmed", "curtosis recortada", 2),
                               ("q0.001", "p0,1", 5), ("q0.01", "p1", 5), ("q0.99", "p99", 5), ("q0.999", "p99,9", 5)):
            if series == SERIES_Z and nd == 5:
                nd = 3
            r = s.loc[stat]
            row[name] = f"{_fmt(r['value'], nd)} [{_fmt(r['ci_lo'], nd)}, {_fmt(r['ci_hi'], nd)}]"
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Secciones 17 (tiempos) y 20 (estado) y archivo de cierre
# ---------------------------------------------------------------------------
# Sin archivo de cierre, estas funciones producen EXACTAMENTE el texto de siempre. Con el, anotan las incidencias de ejecucion (p. ej. una
# suspension del sistema: el tiempo transcurrido no es tiempo de calculo) y el estado de cierre. No tocan ningun numero estadistico.

TIMING_HEADING, AFTER_TIMING_HEADING, STATUS_HEADING = "## 17. Tiempos", "## 18. Archivos generados", "## 20. Estado"
DEFAULT_STATUS_LINE = "**TDA-13 Parte A = `CORREGIDA TRAS AUDITORÍA — PENDIENTE DE SEGUNDA AUDITORÍA` (no cerrada).**"
AMEND_COMMAND = "python -m ohlcv_dataroad.ingest.run_tda13 --amend-report --closure"
_TS_FORMAT = "%Y-%m-%d %H:%M:%S"


def load_closure(path: Path | str | None) -> dict | None:
    """Archivo de cierre (YAML, opcional): `estado`, `detalle`, `auditoria` e `incidencias` (cada una con `tipo`, `inicio`, `fin`, `etapa_afectada`, `evidencia`, `efecto`)."""
    if path is None:
        return None
    with open(path, encoding="utf-8") as fh:
        closure = yaml.safe_load(fh)
    if not isinstance(closure, dict) or not str(closure.get("estado", "")).strip():
        raise ValueError(f"archivo de cierre invalido (falta `estado`): {path}")
    for inc in closure.get("incidencias") or []:
        missing = [k for k in ("tipo", "inicio", "fin", "etapa_afectada", "evidencia", "efecto") if not str(inc.get(k, "")).strip()]
        if missing:
            raise ValueError(f"incidencia incompleta en {path}: faltan {missing}")
        if _incident_seconds(inc) <= 0:
            raise ValueError(f"incidencia con fin anterior o igual al inicio en {path}")
    closure["_source"] = Path(path).as_posix()
    return closure


def _incident_seconds(inc: dict) -> float:
    return (datetime.datetime.strptime(str(inc["fin"]), _TS_FORMAT) - datetime.datetime.strptime(str(inc["inicio"]), _TS_FORMAT)).total_seconds()


def timing_tail(result: TDA13Result) -> str:
    return (f" CPU, {result.n_workers} proceso(s) en la etapa de calibración (limitados por la memoria libre al arrancar); sin GPU. `n_boot = {result.n_boot}` (condicional), "
            f"{result.n_boot_full} (con re-estimación); semillas fijas; IC percentil al {int(100 * CI_LEVEL)} %. El resultado no depende del número de procesos.")


def timing_section(stage_rows: list[tuple[str, float]], elapsed_total_seconds: float, tail: str, closure: dict | None = None, amended: bool = False) -> str:
    """Texto de la seccion 17. `stage_rows` = [(etapa, segundos)]. Con incidencias, separa tiempo transcurrido y tiempo efectivo aproximado (transcurrido menos tiempo parado)."""
    incidents = (closure or {}).get("incidencias") or []
    if not incidents:
        lines = "\n".join(f"| {name} | {seconds:.1f}s |" for name, seconds in stage_rows)
        return (f"{TIMING_HEADING}\n\n| Etapa | Tiempo |\n|---|---:|\n{lines}\n\n"
                f"**Tiempo total (análisis + escritura): {elapsed_total_seconds:.1f}s (~{elapsed_total_seconds / 60:.1f} min).**{tail}")
    names = [name for name, _ in stage_rows]
    stopped_by_stage: dict[str, float] = {}
    for inc in incidents:
        if inc["etapa_afectada"] not in names:
            raise ValueError(f"la incidencia se refiere a una etapa que no figura en el informe: {inc['etapa_afectada']}")
        stopped_by_stage[inc["etapa_afectada"]] = stopped_by_stage.get(inc["etapa_afectada"], 0.0) + _incident_seconds(inc)
    stopped = sum(stopped_by_stage.values())
    if stopped >= elapsed_total_seconds or any(stopped_by_stage[n] > sec for n, sec in stage_rows if n in stopped_by_stage):
        raise ValueError("las incidencias declaradas duran mas que el tiempo transcurrido de la ejecucion o de su etapa")
    rows = []
    for name, seconds in stage_rows:
        if name in stopped_by_stage:
            rows.append(f"| {name} | {seconds:.1f}s (incluye ≈ {stopped_by_stage[name]:.0f} s con el sistema parado; efectivo ≈ {seconds - stopped_by_stage[name]:.0f} s) |")
        else:
            rows.append(f"| {name} | {seconds:.1f}s |")
    effective = elapsed_total_seconds - stopped
    inc_rows = "\n".join(f"| {inc['tipo']} | {inc['inicio']} | {inc['fin']} | ≈ {_incident_seconds(inc):.0f} s (~{_incident_seconds(inc) / 60:.1f} min) | {inc['etapa_afectada']} | {inc['evidencia']} | {inc['efecto']} |"
                         for inc in incidents)
    provenance = (f" Las secciones 17 y 20 se regeneraron con `{AMEND_COMMAND} {closure['_source']}`, sin repetir ningún cálculo: el resto de este informe y los demás artefactos son, sin cambios, los de la ejecución original."
                  if amended else "")
    return (f"{TIMING_HEADING}\n\n| Etapa | Tiempo |\n|---|---:|\n" + "\n".join(rows) + "\n\n"
            f"**Tiempo transcurrido (análisis + escritura): {elapsed_total_seconds:.1f}s (~{elapsed_total_seconds / 60:.1f} min), de los cuales ≈ {stopped:.0f} s (~{stopped / 60:.1f} min) con el sistema parado. "
            f"Tiempo efectivo aproximado: ≈ {effective:.0f} s (~{effective / 60:.1f} min).**{tail}\n\n"
            f"**Incidencias de ejecución** (declaradas en `{closure['_source']}`; horas locales). El tiempo transcurrido NO es tiempo de cálculo: incluye los tramos en que el sistema estuvo parado. "
            f"El tiempo efectivo es una resta aproximada (precisión de segundos de los registros del sistema), no una medida directa.{provenance}\n\n"
            "| Incidencia | Inicio | Fin | Duración | Etapa afectada | Evidencia | Efecto sobre los resultados |\n|---|---|---|---:|---|---|---|\n" + inc_rows)


def status_line(closure: dict | None = None) -> str:
    """Primera linea de la seccion 20. Sin archivo de cierre, el estado de siempre (pendiente de auditoria)."""
    if closure is None:
        return DEFAULT_STATUS_LINE
    line = f"**TDA-13 Parte A = `{closure['estado']}`" + (f" ({closure['detalle']})" if str(closure.get("detalle", "")).strip() else "") + ".**"
    if str(closure.get("auditoria", "")).strip():
        line += f"\n\n{closure['auditoria']} Estado declarado en `{closure['_source']}`."
    return line


_STAGE_ROW = re.compile(r"^\| (.+?) \| (\d+\.\d)s(?: \(incluye .*\))? \|$")
_TOTAL_LINE = re.compile(r"^\*\*Tiempo (?:total|transcurrido) \(análisis \+ escritura\): (\d+\.\d)s .*?\*\*(.*)$")


def amended_report_text(text: str, closure: dict) -> str:
    """Texto del informe con las secciones 17 y 20 reescritas a partir del archivo de cierre (funcion pura: no lee ni escribe archivos).

    Comprueba que las dos secciones actuales son las que este generador produce (sin cierre, o ya con este mismo cierre). Esa comprobacion es
    solo estructural: la integridad del informe COMPLETO la garantiza `amend_report` con su SHA-256.
    """
    if text.count(f"\n{TIMING_HEADING}\n") != 1 or text.count(f"\n{AFTER_TIMING_HEADING}\n") != 1 or text.count(f"\n{STATUS_HEADING}\n") != 1:
        raise RuntimeError("el informe no tiene exactamente una seccion 17, una 18 y una 20: no se modifica")
    i17, i18 = text.index(f"\n{TIMING_HEADING}\n") + 1, text.index(f"\n{AFTER_TIMING_HEADING}\n")
    current = text[i17:i18].rstrip("\n")
    stage_rows, elapsed, tail = [], None, None
    for line in current.split("\n"):
        m = _STAGE_ROW.match(line)
        if m and m.group(1) != "Etapa":
            stage_rows.append((m.group(1), float(m.group(2))))
        m = _TOTAL_LINE.match(line)
        if m and elapsed is None:
            elapsed, tail = float(m.group(1)), m.group(2).split("\n")[0]
    if not stage_rows or elapsed is None:
        raise RuntimeError("no se pudo leer la tabla de tiempos del informe: no se modifica")
    target17 = timing_section(stage_rows, elapsed, tail, closure, amended=True)
    if current not in (timing_section(stage_rows, elapsed, tail), target17):
        raise RuntimeError("la seccion 17 del informe no coincide con lo que produce el generador (¿editada a mano?): no se modifica")
    i20 = text.index(f"\n{STATUS_HEADING}\n") + 1
    head20 = f"{STATUS_HEADING}\n\n"
    rest20 = text[i20 + len(head20):]
    target_status = status_line(closure)
    for candidate in (DEFAULT_STATUS_LINE, target_status):
        if rest20.startswith(candidate + "\n"):
            rest20 = target_status + rest20[len(candidate):]
            break
    else:
        raise RuntimeError("la seccion 20 del informe no empieza con el estado que produce el generador (¿editada a mano?): no se modifica")
    return text[:i17] + target17 + "\n" + text[i18:i20] + head20 + rest20


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def closure_report_hashes(closure: dict) -> tuple[str, str]:
    """(SHA-256 del informe original, SHA-256 del informe cerrado) declarados en el archivo de cierre; obligatorios para enmendar."""
    declared = closure.get("informe") or {}
    original, closed = str(declared.get("sha256_original", "")).strip().lower(), str(declared.get("sha256_cerrado", "")).strip().lower()
    if not _SHA256.match(original) or not _SHA256.match(closed) or original == closed:
        raise ValueError("el archivo de cierre debe declarar `informe.sha256_original` e `informe.sha256_cerrado` (dos SHA-256 distintos) para poder enmendar el informe")
    return original, closed


def amended_report_bytes(raw: bytes, closure: dict) -> bytes:
    """Bytes del informe enmendado a partir de los bytes del informe original (conserva la codificacion UTF-8 y el tipo de salto de linea)."""
    crlf = b"\r\n" in raw
    text = raw.decode("utf-8")
    new_text = amended_report_text(text.replace("\r\n", "\n") if crlf else text, closure)
    return (new_text.replace("\n", "\r\n") if crlf else new_text).encode("utf-8")


def amend_report(report_path: Path, closure: dict) -> bool:
    """Reescribe SOLO las secciones 17 y 20 de un informe ya generado, a partir del archivo de cierre. No recalcula nada ni toca ningun otro artefacto.

    Integridad del informe COMPLETO, por SHA-256 (los dos valores esperados se declaran en el archivo de cierre):
    - si el informe es exactamente el cerrado, no hace nada y devuelve `False` (idempotente);
    - si es exactamente el original, lo enmienda, comprueba que el resultado es exactamente el cerrado y solo entonces escribe; devuelve `True`;
    - cualquier otro contenido (una duracion retocada, una linea de estado cambiada, un espacio de mas) se rechaza sin modificar ningun archivo.
    """
    original, closed = closure_report_hashes(closure)
    raw = Path(report_path).read_bytes()
    found = hashlib.sha256(raw).hexdigest()
    if found == closed:
        return False
    if found != original:
        raise RuntimeError(f"el informe no es ni el original ni el cerrado registrados en el archivo de cierre (SHA-256 {found}): no se modifica")
    new_raw = amended_report_bytes(raw, closure)
    produced = hashlib.sha256(new_raw).hexdigest()
    if produced != closed:
        raise RuntimeError(f"la enmienda no reproduce el informe cerrado registrado (SHA-256 obtenido {produced}; esperado {closed}): no se modifica")
    Path(report_path).write_bytes(new_raw)
    return True


def render_report(result: TDA13Result, config: SnapshotConfig, elapsed_total_seconds: float, run_command: str, closure: dict | None = None) -> str:
    inv, cluster, th06, th27 = result.inventory, result.cluster_table, result.th06, result.th27
    thr = result.thresholds
    prim_thr = thr.loc[thr["is_primary"] & (thr["series"] == SERIES_R)]
    n_cells = len(prim_thr)
    n_low = int((prim_thr["support"] != "OK").sum())
    n_bad = int((prim_thr["support"] == "NO_FIABLE").sum())
    cell_min = prim_thr.loc[prim_thr["n"].idxmin()] if n_cells else None
    low_col = "celdas con soporte < " + str(MIN_TAIL_SUPPORT_LOW)
    rate_rows = []
    for (series, definition, level), g_ in thr.groupby(["series", "definition", "level"], sort=False):
        n_ = int(g_["n"].sum())
        rate_rows.append({"serie": series, "definición": definition, "nivel": f"{100 * level:g} %", "celdas": int(len(g_)), "n": n_,
                          "excedencias neg.": int(g_["n_exceed_neg"].sum()), "excedencias pos.": int(g_["n_exceed_pos"].sum()),
                          "tasa neg. realizada": _pct(g_["n_exceed_neg"].sum() / n_, 3), "tasa pos. realizada": _pct(g_["n_exceed_pos"].sum() / n_, 3),
                          low_col: int((g_["support"] != "OK").sum())})
    rate_tab = pd.DataFrame(rate_rows)

    ts = result.triage_summary
    tot = ts.loc[ts["grouping"] == "TOTAL"].copy()
    tot["categoría"], tot["extremos"], tot["% del inventario"] = tot["category"], tot["n"], [_pct(v) for v in tot["fraction"]]

    def _pivot(grouping):
        s = ts.loc[ts["grouping"] == grouping]
        p = s.pivot_table(index="group", columns="category", values="n", aggfunc="sum").reindex(columns=list(CATEGORIES)).fillna(0).astype(int)
        p["total"] = p.sum(axis=1)
        p[f"% {CAT_UNDETERMINED}"] = [_pct(a / b) if b else "NaN" for a, b in zip(p[CAT_UNDETERMINED], p["total"])]
        return p.reset_index().rename(columns={"group": "grupo"})

    doubts = ts.loc[ts["grouping"].str.startswith("INDETERMINADO_POR_DUDA")].copy()
    doubts["duda"], doubts["extremos"], doubts["% de los indeterminados"] = doubts["group"], doubts["n"], [_pct(v) for v in doubts["fraction"]]
    n_inv = len(inv)
    n_cross = int((inv["population"] == POP_CROSSING).sum())
    n_elig_inv = int((inv["population"] == POP_ELIGIBLE).sum())
    n_0830 = int(inv["clock_0830_window"].sum())
    n_also_z = int(inv["also_extreme_in_z"].sum())
    frac = th06["fractions"]

    prim = dict(level=PRIMARY_LEVEL, reference=PRIMARY_NULL, z_halflife=PRIMARY_ESTIMATOR[1])
    cols_main = ["serie", "definición", "lado", "observado", "esperado si independientes", "cociente [IC95 condicional]", "excedencias", "tiempos medidos"]
    main_tab = _cluster_view(cluster, definition=PRIMARY_DEFINITION, **prim)
    ladder_tab = _cluster_view(cluster, side=[SIDE_NEG, SIDE_POS], **prim)
    win_rows = []
    for series in (SERIES_R, SERIES_Z):
        for side in SIDES:
            row = {"serie": series, "lado": side}
            for k in CLUSTER_WINDOWS:
                r = cluster.loc[(cluster["series"] == series) & (cluster["side"] == side) & (cluster["definition"] == PRIMARY_DEFINITION) & (cluster["level"] == PRIMARY_LEVEL)
                                & (cluster["reference"] == PRIMARY_NULL) & (cluster["statistic"] == f"share_le_{k}")]
                row[f"≤{k} barras"] = _ratio_ci(r.iloc[0]) if len(r) else "—"
            for stat, name in (("frac_in_runs_ge2", "en rachas ≥2"), ("mean_run_length", "longitud media de racha"), ("daily_dispersion", "dispersión diaria")):
                r = cluster.loc[(cluster["series"] == series) & (cluster["side"] == side) & (cluster["definition"] == PRIMARY_DEFINITION) & (cluster["level"] == PRIMARY_LEVEL)
                                & (cluster["reference"] == PRIMARY_NULL) & (cluster["statistic"] == stat)]
                row[name] = _ratio_ci(r.iloc[0]) if len(r) else "—"
            win_rows.append(row)
    win_tab = pd.DataFrame(win_rows)
    level_tab = _cluster_view(cluster, definition=PRIMARY_DEFINITION, reference=PRIMARY_NULL, z_halflife=PRIMARY_ESTIMATOR[1], side=[SIDE_NEG, SIDE_POS])
    level_tab["nivel"] = [f"{100 * v:g} %" for v in level_tab["level"]]
    null_tab = _cluster_view(cluster, definition=PRIMARY_DEFINITION, level=PRIMARY_LEVEL, z_halflife=PRIMARY_ESTIMATOR[1], side=[SIDE_NEG, SIDE_POS])
    null_tab["referencia"] = null_tab["reference"]
    zs = cluster.loc[(cluster["series"].str.startswith("z_")) & (cluster["definition"] == PRIMARY_DEFINITION) & (cluster["level"] == PRIMARY_LEVEL) & (cluster["reference"] == PRIMARY_NULL)]
    zsens_tab = _cluster_view(zs, side=[SIDE_NEG, SIDE_POS])
    zsens_tab["semivida EWMA (min)"] = zsens_tab["z_halflife"]

    run_frac = cluster.loc[(cluster["statistic"] == "frac_in_runs_ge2") & (cluster["definition"] == PRIMARY_DEFINITION) & (cluster["level"] == PRIMARY_LEVEL)
                           & (cluster["reference"] == PRIMARY_NULL) & (cluster["z_halflife"] == PRIMARY_ESTIMATOR[1])]

    g, rn = result.gap_distribution, result.run_distribution

    def _dist(tab, series, side, key, name):
        s = tab.loc[(tab["series"] == series) & (tab["definition"] == PRIMARY_DEFINITION) & (tab["side"] == side) & (tab["reference"] == PRIMARY_NULL)].copy()
        s[name] = s[key]
        s["observado (n)"] = s["n_observed"].round().astype(int)
        s["observado (%)"] = [_pct(v) for v in s["share_observed"]]
        s["esperado (n)"] = [_fmt(v, 1) for v in s["n_expected"]]
        s["esperado (%)"] = [_pct(v) for v in s["share_expected"]]
        s["observado / esperado"] = [_fmt(a / b, 2) if b > 0 else "—" for a, b in zip(s["n_observed"], s["n_expected"])]
        return _md_table(s, [name, "observado (n)", "observado (%)", "esperado (n)", "esperado (%)", "observado / esperado"])

    bg = result.cluster_by_group

    def _group_tab(grouping):
        s = bg.loc[(bg["grouping"] == grouping) & (bg["statistic"] == STAT_PRIMARY) & (bg["side"].isin([SIDE_NEG, SIDE_POS]))]
        rows = []
        for (grp, side), sub in s.groupby(["group", "side"], sort=True):
            row = {"grupo": grp, "lado": side}
            for series in (SERIES_R, SERIES_Z):
                r = sub.loc[sub["series"] == series]
                if len(r):
                    r = r.iloc[0]
                    mark = "" if r["support"] == "OK" else (" ⚠" if r["support"] == "BAJO" else " ✖")
                    row[f"{series}: cociente [IC95]"] = _ratio_ci(r) + mark
                    row[f"{series}: tiempos"] = int(r["n_intervals"])
            rows.append(row)
        return pd.DataFrame(rows)

    h = result.hourly
    hour_rows = []
    for hr in sorted(h["hour_ny"].unique()):
        row = {"hora NY": int(hr)}
        base = h.loc[(h["hour_ny"] == hr) & (h["side"] == SIDE_BOTH)]
        if len(base):
            row["% de las barras"] = _pct(base["share_of_bars"].iloc[0], 1)
        for definition in DEFINITIONS:
            r = base.loc[(base["series"] == SERIES_R) & (base["definition"] == definition)]
            if len(r):
                row[f"r · {definition}: % de los extremos"] = _pct(r["share_of_exceedances"].iloc[0], 1)
        r = base.loc[(base["series"] == SERIES_Z) & (base["definition"] == PRIMARY_DEFINITION)]
        if len(r):
            row[f"z · {PRIMARY_DEFINITION}: % de los extremos"] = _pct(r["share_of_exceedances"].iloc[0], 1)
        hour_rows.append(row)
    hour_tab = pd.DataFrame(hour_rows)

    def _lift_range(series, definition):
        s = h.loc[(h["series"] == series) & (h["definition"] == definition) & (h["side"] == SIDE_BOTH)]
        return (float(s["lift_vs_overall"].min()), float(s["lift_vs_overall"].max())) if len(s) else (np.nan, np.nan)

    lift_g, lift_s, lift_p = _lift_range(SERIES_R, DEF_GLOBAL), _lift_range(SERIES_R, DEF_SEGMENT), _lift_range(SERIES_R, PRIMARY_DEFINITION)

    imp = result.triage_impact

    def _imp(scenario, stat, col="value"):
        s = imp.loc[(imp["scenario"] == scenario) & (imp["statistic"] == stat), col]
        return float(s.iloc[0]) if len(s) else float("nan")

    k_valid, k_no_und = _imp("SOLO_VALIDOS", "kurtosis_excess"), _imp(f"VALIDOS_sin_{CAT_UNDETERMINED}", "kurtosis_excess")
    k_univ, k_no_roll, k_no_bnd = _imp("UNIVERSO_COMPLETO", "kurtosis_excess"), _imp(f"UNIVERSO_sin_{CAT_ROLL}", "kurtosis_excess"), _imp(f"UNIVERSO_sin_{CAT_BOUNDARY}", "kurtosis_excess")
    k_no_real, k_no_ext = _imp(f"VALIDOS_sin_{CAT_REAL}", "kurtosis_excess"), _imp("VALIDOS_sin_EXTREMOS", "kurtosis_excess")
    range_rows = []
    for series, base, without in ((SERIES_R, "SOLO_VALIDOS", f"VALIDOS_sin_{CAT_UNDETERMINED}"), (SERIES_Z, "Z_TODO", f"Z_sin_{CAT_UNDETERMINED}")):
        for stat, name in (("kurtosis_excess", "curtosis"), ("kurtosis_excess_trimmed", "curtosis recortada"), ("q0.001", "percentil 0,1"), ("q0.01", "percentil 1"),
                           ("q0.99", "percentil 99"), ("q0.999", "percentil 99,9")):
            a, b = _imp(base, stat), _imp(without, stat)
            nd = 2 if "curtosis" in name else (5 if series == SERIES_R else 3)
            range_rows.append({"serie": series, "estadístico": name, "con INDETERMINADO": _fmt(a, nd), "sin INDETERMINADO": _fmt(b, nd),
                               "cambio relativo": _pct(b / a - 1.0, 1) if a else "NaN"})
    range_tab = pd.DataFrame(range_rows)
    ref = result.reference_checks
    ref_txt = ("No se encontraron los CSV publicados de TDA-07/TDA-12 para contrastar." if ref is None or len(ref) == 0 else
               f"{int(ref['matches'].sum())} de {len(ref)} valores recalculados coinciden con los publicados por TDA-07 y TDA-12 (diferencia máxima {ref['abs_diff'].max():.2e}).")

    pt, fb, bands, rates = result.primary_point, result.full_bootstrap, result.calibration_bands, result.calibration_rates
    d = th27.get("detail", {})
    stat_key = f"ratio_le_{PRIMARY_WINDOW}"

    def _fb(series, side, stat):
        s = fb.loc[(fb["series"] == series) & (fb["side"] == side) & (fb["statistic"] == stat)] if fb is not None and len(fb) else pd.DataFrame()
        return s.iloc[0] if len(s) else None

    def _band(series, side, col="band_hi"):
        s = bands.loc[(bands["series"] == series) & (bands["side"] == side), col] if bands is not None and len(bands) else pd.Series(dtype=float)
        return float(s.iloc[0]) if len(s) else float("nan")

    rule_rows = []
    for series in (SERIES_R, SERIES_Z):
        for side in SIDES:
            f_ = _fb(series, side, stat_key)
            ratio = pt.get(f"{series}|{side}|{stat_key}", np.nan)
            hi = _band(series, side)
            det = None if f_ is None or not np.isfinite(hi) else bool(ratio > hi and f_["ci_lo"] > 1.0)
            rule_rows.append({
                "serie": series, "lado": side,
                "observado": _pct(pt.get(f"{series}|{side}|observed_le_{PRIMARY_WINDOW}", np.nan)),
                "esperado si independientes": _pct(pt.get(f"{series}|{side}|expected_le_{PRIMARY_WINDOW}", np.nan)),
                "cociente": _fmt(ratio, 3),
                "IC95 con re-estimación": "—" if f_ is None else f"[{_fmt(f_['ci_lo'], 3)}, {_fmt(f_['ci_hi'], 3)}]",
                "IC95 condicional (previo)": "—" if f_ is None else f"[{_fmt(f_['conditional_ci_lo'], 3)}, {_fmt(f_['conditional_ci_hi'], 3)}]",
                "banda calibrada (sup.)": _fmt(hi, 3),
                "banda Bernoulli previa (sup.)": _fmt(_band(series, side, "legacy_bernoulli_band_hi"), 3),
                "¿detectado?": "—" if det is None else ("sí" if det else "no"),
            })
    rule_tab = pd.DataFrame(rule_rows)
    fr_rows = []
    for side in SIDES:
        a_, b_ = _fb("r_vs_z", side, "ratio_difference"), _fb("r_vs_z", side, "fraction_removed")
        if a_ is not None and b_ is not None:
            fr_rows.append({"lado": side, "cociente r − cociente z [IC95]": f"{_fmt(a_['point'], 3)} [{_fmt(a_['ci_lo'], 3)}, {_fmt(a_['ci_hi'], 3)}]",
                            "parte del exceso que desaparece al estandarizar [IC95]": f"{_pct(b_['point'], 0)} [{_pct(b_['ci_lo'], 0)}, {_pct(b_['ci_hi'], 0)}]"})
    fr_tab = pd.DataFrame(fr_rows)
    width = fb.loc[(fb["statistic"] == stat_key) & fb["ci_width_ratio_full_vs_conditional"].notna(), "ci_width_ratio_full_vs_conditional"] if fb is not None and len(fb) else pd.Series(dtype=float)
    width_txt = f"entre {_fmt(float(width.min()), 2)} y {_fmt(float(width.max()), 2)} veces la" if len(width) else "—"

    band_rows = []
    if bands is not None and len(bands):
        for _, b_ in bands.iterrows():
            row = {"serie": b_["series"], "lado": b_["side"]}
            for fam in INDEPENDENCE_FAMILIES:
                if f"{fam}|mean" in b_:
                    row[f"{fam}: media ± desv."] = f"{_fmt(b_[f'{fam}|mean'], 3)} ± {_fmt(b_[f'{fam}|std'], 3)}"
                    row[f"{fam}: p97,5"] = _fmt(b_[f"{fam}|p97_5"], 3)
            row["banda calibrada (sup.)"] = _fmt(b_["band_hi"], 3)
            row["familia que fija la banda"] = b_.get("binding_family", "—")
            row["banda Bernoulli previa (sup.)"] = _fmt(b_["legacy_bernoulli_band_hi"], 3)
            band_rows.append(row)
    band_tab = pd.DataFrame(band_rows)

    def _rate_tab(ftype):
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
        s = rates.loc[rates["family_type"] == ftype].copy()
        s["control"], s["regla"], s["réplicas"], s["detecciones"] = s["family"], s["rule"], s["n_replicates"], s["n_detections"]
        s["tasa [IC95 binomial]"] = [f"{_pct(a, 1)} [{_pct(b, 1)}, {_pct(c, 1)}]" for a, b, c in zip(s["rate"], s["rate_ci_lo"], s["rate_ci_hi"])]
        s["¿cumple?"] = ["—" if v is None or (isinstance(v, float) and np.isnan(v)) else ("sí" if v else "NO") for v in s["passes"]] if "passes" in s else "—"
        s["máx. admitido"] = ["—" if not np.isfinite(v) else str(int(v)) for v in s["acceptance_max_detections"]] if "acceptance_max_detections" in s else "—"
        if "legacy_band_rate" in s:
            s["con la banda previa [IC95]"] = [f"{_pct(a, 1)} [{_pct(b, 1)}, {_pct(c, 1)}]" if np.isfinite(a) else "—"
                                              for a, b, c in zip(s["legacy_band_rate"], s["legacy_band_rate_ci_lo"], s["legacy_band_rate_ci_hi"])]
        return s

    rates_ind, rates_vol = _rate_tab("SIN_DEPENDENCIA"), _rate_tab("SOLO_VOLATILIDAD")
    n_cal_half = int(rates_ind["n_replicates"].max()) if len(rates_ind) else 0
    n_acc = int(rates_ind["in_acceptance_criterion"].astype(bool).sum()) if len(rates_ind) else 0
    worst = rates_ind.loc[rates_ind["in_acceptance_criterion"].astype(bool)].sort_values("rate", ascending=False).head(1) if len(rates_ind) else pd.DataFrame()
    worst_txt = (f"La tasa más alta entre las que entran en el criterio es {_pct(float(worst['rate'].iloc[0]), 1)} "
                 f"[{_pct(float(worst['rate_ci_lo'].iloc[0]), 1)}, {_pct(float(worst['rate_ci_hi'].iloc[0]), 1)}] (`{worst['family'].iloc[0]}`, {worst['rule'].iloc[0]})." if len(worst) else "")
    worst_hi_txt = _pct(float(worst['rate_ci_hi'].iloc[0]), 1) if len(worst) else "—"
    legacy_worst = rates_ind.loc[rates_ind["legacy_band_rate"].notna()].sort_values("legacy_band_rate", ascending=False).head(1) if len(rates_ind) and "legacy_band_rate" in rates_ind else pd.DataFrame()
    legacy_txt = (f"Con la banda Bernoulli previa, sobre estas mismas réplicas, la tasa por cola llega al {_pct(float(legacy_worst['legacy_band_rate'].iloc[0]), 1)} "
                  f"[{_pct(float(legacy_worst['legacy_band_rate_ci_lo'].iloc[0]), 1)}, {_pct(float(legacy_worst['legacy_band_rate_ci_hi'].iloc[0]), 1)}] "
                  f"(`{legacy_worst['family'].iloc[0]}`, {legacy_worst['rule'].iloc[0]}), frente al {_pct(G2_NOMINAL_RATE, 1)} nominal." if len(legacy_worst) else "")
    diag = result.bernoulli_diagnostic
    diag_tab = diag.copy() if diag is not None and len(diag) else pd.DataFrame()
    if len(diag_tab):
        names = {DIAG_BERNOULLI_KNOWN: "Excedencias independientes, tasa conocida (banda previa)", DIAG_BERNOULLI_REFIT: "Excedencias independientes, tasa re-estimada"}
        diag_tab["simulación"] = [names.get(v, v.replace("PROCEDIMIENTO_COMPLETO|", "Procedimiento completo: ")) for v in diag_tab["simulation"]]
        diag_tab["serie"], diag_tab["lado"] = diag_tab["series"], diag_tab["side"]
        diag_tab["media"], diag_tab["desviación"], diag_tab["p97,5"] = [_fmt(v, 3) for v in diag_tab["ratio_mean"]], [_fmt(v, 3) for v in diag_tab["ratio_std"]], [_fmt(v, 3) for v in diag_tab["ratio_p97_5"]]
        diag_tab = diag_tab.sort_values(["serie", "lado", "simulation"])

    vol_ev = th27.get("volatility_only_evidence")
    vol_tab = vol_ev.copy() if vol_ev is not None and len(vol_ev) else pd.DataFrame()
    if len(vol_tab):
        vol_tab["control"], vol_tab["lado"], vol_tab["réplicas"] = vol_tab["family"], vol_tab["side"], vol_tab["n_replicates"]
        vol_tab["cociente real de z"] = [_fmt(v, 3) for v in vol_tab["z_ratio_real"]]
        vol_tab["cociente de z en el control: media [p2,5; p97,5]"] = [f"{_fmt(a, 3)} [{_fmt(b, 3)}; {_fmt(c, 3)}]" for a, b, c in zip(vol_tab["z_ratio_control_mean"], vol_tab["z_ratio_control_p2_5"], vol_tab["z_ratio_control_p97_5"])]
        vol_tab["parte del exceso real que el control reproduce"] = [_pct(v, 0) for v in vol_tab["share_of_real_excess_reproduced"]]
        vol_tab["¿el real cabe en el control?"] = ["sí" if v else "no" for v in vol_tab["real_within_control_range"]]
        vol_tab["tasa de 'persiste' en el control"] = [_pct(v, 0) for v in vol_tab["claim_2_rate_in_control"]]
    compat = th27.get("volatility_controls_reproducing_real_z", [])
    r_pos_ctrl = rates_vol.loc[rates_vol["rule"].str.startswith("regla global: afirmacion 1")] if len(rates_vol) else pd.DataFrame()
    r_pos_txt = "; ".join(f"`{a}`: {_pct(b, 0)}" for a, b in zip(r_pos_ctrl["family"], r_pos_ctrl["rate"])) if len(r_pos_ctrl) else "—"

    cons = result.reference_conservation
    cons_tab = cons.loc[cons["side"].isin([SIDE_NEG, SIDE_POS])].copy() if cons is not None and len(cons) else pd.DataFrame()
    if len(cons_tab):
        cons_tab["serie"], cons_tab["lado"], cons_tab["referencia"] = cons_tab["series"], cons_tab["side"], cons_tab["reference"]
        cons_tab["observadas"], cons_tab["esperadas"] = cons_tab["n_observed"].round().astype(int), [_fmt(v, 2) for v in cons_tab["n_expected"]]
        cons_tab["error máx. por minuto"] = [_fmt(v, 3) for v in cons_tab["max_abs_error_by_minute"]]
        cons_tab["error máx. por año"] = [_fmt(v, 2) for v in cons_tab["max_abs_error_by_year"]]
        cons_tab["año peor (error relativo)"] = [f"{a} ({_pct(b, 2)})" for a, b in zip(cons_tab["worst_year"], cons_tab["worst_year_rel_error"])]
        cons_tab["barras con p = 0"] = cons_tab["n_bars_p_zero"]
    legacy_cmp = _cluster_view(cluster, definition=PRIMARY_DEFINITION, level=PRIMARY_LEVEL, z_halflife=PRIMARY_ESTIMATOR[1], side=[SIDE_NEG, SIDE_POS], reference=[PRIMARY_NULL, NULL_CLOCK_YEAR_LEGACY])
    legacy_cmp["referencia"] = legacy_cmp["reference"]

    prev_path = config.tda13_audit_previous_csv_path
    prev_tab, prev_note = pd.DataFrame(), "No se encontró la foto de resultados previos a la auditoría."
    if config.tda13_audit_previous_csv_name and prev_path.exists():
        prev = pd.read_csv(prev_path)
        pc = prev.loc[(prev["table"] == "clustering") & (prev["statistic"] == f"share_le_{PRIMARY_WINDOW}")]
        rows_ = []
        for _, r_ in pc.iterrows():
            f_ = _fb(r_["series"], r_["side"], stat_key)
            now = pt.get(f"{r_['series']}|{r_['side']}|{stat_key}", np.nan)
            rows_.append({"serie": r_["series"], "lado": r_["side"],
                          "antes: cociente [IC condicional]": f"{_fmt(r_['ratio'], 3)} [{_fmt(r_['ratio_ci_lo'], 3)}, {_fmt(r_['ratio_ci_hi'], 3)}]",
                          "antes: banda sup.": _fmt(r_["band_hi"], 3),
                          "después: cociente [IC con re-estimación]": "—" if f_ is None else f"{_fmt(now, 3)} [{_fmt(f_['ci_lo'], 3)}, {_fmt(f_['ci_hi'], 3)}]",
                          "después: banda calibrada sup.": _fmt(_band(r_["series"], r_["side"]), 3),
                          "cambio del cociente": _fmt(now - r_["ratio"], 4)})
        prev_tab = pd.DataFrame(rows_)
        pv = prev.loc[prev["table"] == "verdict"]
        prev_note = "Veredictos previos: " + "; ".join(f"{a} = `{b}`" for a, b in zip(pv["artifact"], pv["value_text"])) + "."

    sens = result.triage_sensitivity
    sens_tab = sens.copy() if sens is not None and len(sens) else pd.DataFrame()
    if len(sens_tab):
        sens_tab["tipo"], sens_tab["variante"] = sens_tab["kind"], sens_tab["description"]
        sens_tab["% INDETERMINADO"], sens_tab["% EVENTO_REAL"] = [_pct(v, 1) for v in sens_tab["fraction_undetermined"]], [_pct(v, 1) for v in sens_tab["fraction_real"]]
    und_lo, und_hi = th06.get("undetermined_range_over_rule_variants", (np.nan, np.nan))
    basis_md = "\n".join(f"| `{c_}` | {TH06_BASIS[c_]} |" for c_ in CATEGORIES)

    c1, c2 = th27["claim_1"], th27["claim_2"]

    def _side_line(side):
        if side not in d:
            return f"- **{SIDE_TITLES[side]}**: sin datos suficientes."
        s = d[side]

        def one(tag, name):
            det = s[f"{tag}_detected"]
            return (f"{name}: {_pct(s[f'{tag}_observed'])} de los tiempos entre extremos es de {PRIMARY_WINDOW} barras o menos, frente al {_pct(s[f'{tag}_expected'])} esperado; "
                    f"cociente {_fmt(s[f'{tag}_ratio'], 2)} (IC 95 % con re-estimación [{_fmt(s[f'{tag}_ci'][0], 2)}, {_fmt(s[f'{tag}_ci'][1], 2)}]; banda calibrada hasta {_fmt(s[f'{tag}_band_hi'], 3)}) → "
                    f"{'sin datos' if det is None else ('DETECTADO' if det else 'no detectado')}")

        return (f"- **{SIDE_TITLES[side]}**. {one('r', 'En crudo')}. {one('z', 'Estandarizado')}. El cociente de `z` supera 1 en {s['z_years_above_1']} de {s['z_years_evaluated']} años completos "
                f"(mínimo {_fmt(s['z_min_year_ratio'], 2)}). Parte del exceso que desaparece al estandarizar: {_pct(s['fraction_removed'], 0)} "
                f"[{_pct(s['fraction_removed_ci'][0], 0)}, {_pct(s['fraction_removed_ci'][1], 0)}].")

    verdict27 = th27["verdict"]
    plain27 = {
        TH27_NONE: "Los movimientos extremos no llegan más juntos de lo que llegarían por puro azar, una vez descontadas la hora del día y el año.",
        TH27_EXPLAINED: "Los movimientos extremos llegan en grupos, pero al dividir por la volatilidad reciente ya no se detecta agrupamiento.",
        TH27_PERSISTS: ("Los movimientos extremos llegan en grupos, y siguen haciéndolo después de dividir por la volatilidad reciente (media móvil de 60 minutos). "
                        "Lo que este análisis NO establece es si ese resto es algo distinto de la volatilidad o simplemente volatilidad que el estimador no alcanza a seguir. No lo afirma ni lo descarta: el método usado aquí no sirve para distinguirlo."),
        TH27_INCONCLUSIVE: "Con las reglas fijadas de antemano y la calibración obtenida, no se puede afirmar el resultado.",
    }[verdict27]
    gate_txt = ""
    if th27.get("g2_ok") is not True:
        gate_txt = (f" Sin la condición de calibración, la afirmación 1 habría sido `{CLAIM_DISPLAY.get(th27['claim_1_before_g2_gate'], th27['claim_1_before_g2_gate'])}`; "
                    "se deja en `INCONCLUSO` porque G2 queda rechazada o no es evaluable (§12).")
    claim3_txt = ((f"El cociente real de `z` **cabe dentro** de lo que producen estos controles de solo volatilidad: {', '.join('`' + c_ + '`' for c_ in compat)}. "
                   "Es decir, una volatilidad que cambia a esa velocidad, sin ningún agrupamiento propio de los extremos, basta para reproducir lo observado. Eso muestra que la explicación por volatilidad es suficiente, no que sea la verdadera.") if compat else
                  ("Ninguno de los controles de solo volatilidad considerados reproduce por completo el cociente real de `z`. Eso **no** demuestra un agrupamiento propio: "
                   "sólo dice que estas sendas concretas no bastan. Podría ser volatilidad que cambia de otra forma o más deprisa, o podría ser un fenómeno propio; este análisis no decide entre ambas."))

    timing_text = timing_section([(name, seconds) for name, seconds in result.stage_timings.items() if not name.startswith("_")], elapsed_total_seconds, timing_tail(result), closure)
    cz = result.causality_decile.get("checks", [])
    cz_ok = sum(1 for c_ in cz if c_["earlier_days_unchanged"] and c_["same_day_other_bars_unchanged"] and c_["truncation_identical"])
    rules_md = "\n".join(f"| {rid} | `{cat}` | {txt} |" for rid, cat, txt in TRIAGE_RULES_TEXT)
    zskip = (" No se pudo evaluar la semivida " + ", ".join(f"{x:g}" for x in result.z_sensitivity_skipped) + " sobre las mismas barras (su quemado es más largo); se omite en vez de comparar poblaciones distintas."
             if result.z_sensitivity_skipped else "")
    ng = result.null_family_groups
    if ng is not None and len(ng):
        ng_tab = pd.DataFrame({"familia": ng["family"], "grupos": [f"{int(v):,}" for v in ng["n_groups"]], "barras por grupo: mínimo": ng["min_size"].astype(int),
                               "mediana": [f"{v:,.0f}" for v in ng["median_size"]], "grupos de una sola barra": ng["n_groups_of_one_bar"].astype(int),
                               f"grupos con menos de {SMALL_GROUP_SIZE} barras": ng["n_small_groups"].astype(int),
                               "barras en esos grupos": [f"{int(a):,} ({_pct(b, 3)})" for a, b in zip(ng["bars_in_small_groups"], ng["share_bars_in_small_groups"])]})
        groups_md = _md_table(ng_tab, list(ng_tab.columns))
    else:
        groups_md = "(no disponible)"
    size_bound_txt = _pct(permutation_band_size_bound(n_cal_half), 2) if n_cal_half >= 2 else "—"
    cal_ok_txt = {G2_STATUS_COMPATIBLE: "COMPATIBLE CON LOS CONTROLES EVALUADOS (no rechazada)", G2_STATUS_INCOMPATIBLE: "INCOMPATIBLE CON LOS CONTROLES EVALUADOS (rechazada)",
                  G2_STATUS_NOT_EVALUABLE: "NO EVALUABLE"}[g2_status(th27.get("g2_ok"))]
    acc_rows = rates.loc[rates["in_acceptance_criterion"].astype(bool)] if "in_acceptance_criterion" in rates and "acceptance_max_detections" in rates else rates.iloc[0:0]
    if len(acc_rows):
        kmax_g2, n_eval_g2 = int(acc_rows["acceptance_max_detections"].max()), int(acc_rows["n_replicates"].max())
        order_j = band_order_statistic(n_cal_half)
        power_md = _md_table(pd.DataFrame({
            "tasa verdadera de falsa detección (por cola)": [_pct(q, 1) for q in G2_POWER_RATES],
            f"probabilidad de NO superar {kmax_g2}/{n_eval_g2} (pasa)": [_pct(criterion_non_rejection_probability(kmax_g2, n_eval_g2, q), 1) for q in G2_POWER_RATES],
            "probabilidad de que el criterio lo rechace": [_pct(1.0 - criterion_non_rejection_probability(kmax_g2, n_eval_g2, q), 1) for q in G2_POWER_RATES],
        }), ["tasa verdadera de falsa detección (por cola)", f"probabilidad de NO superar {kmax_g2}/{n_eval_g2} (pasa)", "probabilidad de que el criterio lo rechace"])
        power_txt = (f"**Cuánto distingue el criterio.** El máximo admitido ({kmax_g2} de {n_eval_g2}) es una cota conservadora exacta: se calcula con el estadístico de orden {order_j} de las {n_cal_half} réplicas que fijan la banda "
                     f"(el percentil 97,5 se interpola entre las posiciones {order_j} y {order_j + 1}, y nunca queda por debajo de la {order_j}). Esa cota protege de rechazar un procedimiento correcto; a cambio distingue poco. "
                     "Si la tasa verdadera de una cola fuese la indicada (con la banda fija), esto es lo que ocurriría:\n\n" + power_md +
                     "\n\nEs decir: una tasa del doble del nominal pasaría la mayor parte de las veces. Pasar el criterio **no demuestra** que las tasas sean del 2,5 %; sólo que no hay evidencia para rechazarlo en las familias evaluadas, con esta potencia.")
    else:
        power_txt = ""

    return f"""# TDA-13 Parte A — Extremos: inventario, triage y agrupamiento

**Etapa:** `docs/methodology/Tsay_OHLCV_analysis_roadmap.md` § TDA-13, Parte A (Nivel 4, obligatoria). **Hipótesis:** `TH06`, `TH27`.
**Depende de:** TDA-00 (máscara `bad_data`), TDA-02 (huecos y su causa), TDA-03 (rolls; firma de "tick aislado"), TDA-04 (`r_1m`, validez), TDA-05 (categoría de barra), TDA-06 (segmentos), TDA-07 (niveles de cuantil, bootstrap, retorno contrafactual), TDA-09 (bloques de continuidad), TDA-10 (`sigma_hat`, `z_t`), TDA-12 (decil causal de volatilidad; TH26 `NO CONCLUYENTE`).
**Alcance de datos:** exclusivamente artefactos ya construidos sobre el conjunto de investigación (`data/interim/mnq/` y CSV de `reports/mnq/`). Ningún archivo de `data/raw/` ni de `holdout_files` fue abierto.
**Generado automáticamente** por `{run_command}` — este informe NUNCA se edita a mano; cualquier corrección se hace en el código (`tda13_extremes_inventory.py`/`run_tda13.py`) y se regenera.

> Esta etapa hace un inventario de los movimientos de 1 minuto más grandes de MNQ, clasifica cada uno (¿error de datos, artefacto o movimiento real?) y mide si llegan en grupos. Es caracterización estadística: NO diseña targets, señales ni modelos; NO ejecuta la Parte B (EVT, POT, GPD, índice extremal); NO abre el holdout. Todos los resultados son **exploratorios y retrospectivos** (G2): ninguno se ha confirmado en el holdout.

---

## 1. Preguntas

1. **TH06:** de los movimientos extremos, ¿cuáles son errores de datos, cuáles artefactos (de roll o de frontera) y cuáles movimientos reales?
2. **TH27:** ¿los extremos llegan más juntos de lo que llegarían si fueran independientes? ¿Sigue ocurriendo después de descontar la volatilidad?

**Palabras que se usan en este informe.**
- *Extremo / excedencia*: un retorno de 1 minuto que supera un umbral. Aquí el umbral no es un número fijo: es "estar entre el {100 * PRIMARY_LEVEL:g} % de los movimientos más grandes para esa hora y ese nivel de agitación".
- *Retorno estandarizado `z`*: el retorno dividido por la volatilidad reciente. Un `z` grande es un movimiento grande *para lo agitado que ya estaba el mercado*.
- *Agrupamiento*: que tras un extremo sea más probable de lo normal ver otro poco después.
- *Referencia de independencia (Poisson)*: lo que se vería si cada barra tuviera la misma probabilidad de ser extremo que la observada, pero sin ninguna relación con las barras vecinas.
- *IC 95 %*: intervalo que indica cuánto podría moverse el número si se repitiera el análisis con otras jornadas (bootstrap por jornadas completas).

## 2. Población

{_md_table(result.population_info, ["poblacion", "n"])}

- **`r`** = `r_1m` válido de TDA-04. **`z`** = `r_t / sigma_hat_(t-1)` con el estimador primario de TDA-10 (EWMA, semivida {PRIMARY_ESTIMATOR[1]:g} barras), sin redefinir nada. Estado de volatilidad = decil causal de TDA-12.
- **Población común.** `r` y `z` se comparan sobre las MISMAS barras (retorno válido, `z` finito, decil definido). Así las tasas de excedencia son comparables.
- **Retornos que cruzan una frontera.** TDA-04 los excluyó de `r_1m` (cruzan un roll, un cierre de sesión o un hueco). No vuelven a entrar en ningún análisis estadístico: sólo se recuperan para el triage, con el retorno contrafactual que TDA-07 ya definió, porque precisamente ahí están los candidatos a artefacto.
- **Causalidad (alcance exacto).** `sigma_hat` y `z` son causales (prueba de reconstrucción de TDA-10 re-ejecutada para el estimador primario: {'pasa' if bool(result.causality_sigma['passed'].all()) else 'NO pasa'}); decil causal verificado por muestreo ({cz_ok} de {len(cz)} comprobaciones pasan). **Los umbrales, las referencias y los cortes de segmento usan toda la muestra: son RETROSPECTIVOS.** Esta etapa describe; no produce ninguna cantidad utilizable en tiempo real.

## 3. Qué es un extremo aquí

Un umbral único para toda la muestra seleccionaría horas (la apertura) y épocas agitadas, no eventos. Por eso el umbral se fija **dentro de cada celda** = segmento horario (7, de TDA-06) × decil causal de volatilidad (10): un retorno es extremo negativo si queda por debajo del percentil {100 * PRIMARY_LEVEL:g} de su celda, y positivo si queda por encima del {100 * (1 - PRIMARY_LEVEL):g}.

- **Definición primaria:** `{PRIMARY_DEFINITION}`, nivel {100 * PRIMARY_LEVEL:g} % por cola.
- **Sensibilidades declaradas de antemano:** niveles {", ".join(f"{100 * v:g} %" for v in SENSITIVITY_LEVELS)}; definiciones `{DEF_SEGMENT}` (sólo segmento) y `{DEF_GLOBAL}` (umbral único: el anti-patrón, incluido sólo para ver su efecto).
- Caídas y subidas se analizan por separado; "ambas" se reporta como complemento.

{_md_table(rate_tab, ["serie", "definición", "nivel", "celdas", "n", "excedencias neg.", "excedencias pos.", "tasa neg. realizada", "tasa pos. realizada", low_col])}

**Grupos pequeños.** En la definición primaria hay {n_cells} celdas; {n_low} tienen menos de {MIN_TAIL_SUPPORT_LOW} excedencias por cola ({n_bad} con menos de {MIN_TAIL_SUPPORT_UNRELIABLE}).{(' La celda más pequeña es `' + str(cell_min['cell']) + '` con ' + _fmt(int(cell_min['n'])) + ' barras.') if cell_min is not None else ''} En esas celdas el umbral es impreciso; con el nivel {100 * min(SENSITIVITY_LEVELS):g} % casi todas las celdas quedan con muy pocas excedencias, y sus resultados son sólo orientativos. Como los precios se mueven en saltos discretos (ticks), varias barras pueden empatar exactamente en el umbral; se cuentan como extremo sólo las que lo superan estrictamente, y por eso la tasa realizada puede quedar algo por debajo del nivel nominal. Detalle por celda en `{config.tda13_thresholds_csv_name}`.

## 4. Criterios de triage (escritos antes de calcular su efecto)

Reglas en orden; decide la primera que aplica.

| Regla | Categoría | Criterio |
|---|---|---|
{rules_md}

**Lo que estas reglas NO hacen.**
- No declaran real un movimiento sólo porque no se detecte un error: `EVENTO_REAL` exige que el precio se sostenga en la barra siguiente y que hubiera negociación normal.
- No atribuyen causas. El repositorio no contiene un calendario fechado de noticias o datos macroeconómicos, así que **ningún extremo se atribuye a un evento**. La columna `clock_0830_window` sólo marca que la barra cae en los {MACRO_CLOCK_WINDOW_MINUTES} minutos desde las 08:30 de Nueva York (el único horario de publicaciones citado en el repositorio, TDA-06). Es una coincidencia de reloj: no demuestra causalidad y no interviene en la clasificación.
- No eliminan nada. El inventario clasifica; los datos originales y los resultados de TDA-07/TDA-12 quedan intactos.

**Qué es heredado y qué es nuevo.** Las máscaras son de etapas anteriores. La firma "revierte más de {REVERSAL_FRACTION_TDA03:g} del salto en la barra siguiente" y "volumen bajo el percentil {100 * LOW_VOLUME_QUANTILE_TDA03:g}" son las de TDA-03, sin cambiar sus valores. **Usarlas como criterio de triage de todos los extremos es una decisión nueva y exploratoria de esta etapa**, igual que el orden de las reglas.

## 5. Inventario y resultado del triage (TH06)

`{config.tda13_extremes_csv_name}` contiene **{_fmt(n_inv)} extremos** (una fila por observación): {_fmt(n_elig_inv)} de la población común, {_fmt(n_cross)} retornos que cruzan una frontera y {_fmt(n_inv - n_elig_inv - n_cross)} válidos sin estado de volatilidad (para estos dos últimos grupos el umbral aplicado es el de su segmento, anotado en `threshold_scope`). Cada fila lleva fecha y hora (UTC y Nueva York), contrato, segmento, estado de volatilidad, magnitud (en retorno, en ticks y en `z`), umbral aplicado, indicadores de calidad, categoría, regla que decidió y evidencia.

{_md_table(tot, ["categoría", "extremos", "% del inventario"])}

- Errores y artefactos juntos: {_pct(th06['fraction_artefacts_and_errors'])} del inventario.
- {_fmt(n_also_z)} de los extremos de `r` ({_pct(n_also_z / n_inv if n_inv else float('nan'), 1)}) lo son también en `z`.
- {_fmt(n_0830)} caen en la ventana de reloj de las 08:30 NY (contexto, sin atribución).

**Por qué quedan `INDETERMINADO`** (una observación puede tener varias dudas):

{_md_table(doubts, ["duda", "extremos", "% de los indeterminados"])}

**Por dirección**

{_md_table(_pivot("DIRECTION"), ["grupo"] + list(CATEGORIES) + ["total", f"% {CAT_UNDETERMINED}"])}

**Por año**

{_md_table(_pivot("YEAR"), ["grupo"] + list(CATEGORIES) + ["total", f"% {CAT_UNDETERMINED}"])}

**Por segmento horario**

{_md_table(_pivot("SEGMENT"), ["grupo"] + list(CATEGORIES) + ["total", f"% {CAT_UNDETERMINED}"])}

**Por población**

{_md_table(_pivot("POPULATION"), ["grupo"] + list(CATEGORIES) + ["total", f"% {CAT_UNDETERMINED}"])}

**Cómo leerlo.** `INDETERMINADO` no significa "sospechoso de error": significa que con barras de 1 minuto no hay forma de comprobarlo. La duda más frecuente es que el precio retroceda más de la mitad en el minuto siguiente; eso es lo que haría un precio anómalo, pero también lo hace un mercado real que rebota. Sin datos tick a tick (G0: la secuencia dentro del minuto no es observable) las dos cosas son indistinguibles.

## 6. Cómo se mide el agrupamiento

1. Se marcan las excedencias y se mide cuántas barras pasan de una a la siguiente. **Sólo se mide dentro de un bloque continuo de negociación**: nunca a través de un cierre de sesión, la pausa diaria, un fin de semana, un hueco o un cambio de contrato.
2. Se compara con la **referencia de independencia**: cada barra es extremo con una probabilidad que depende de *su minuto del día y su año*, sin relación con las vecinas. Es el equivalente, en barras de 1 minuto, de un proceso de Poisson de la misma intensidad; su valor esperado se calcula de forma exacta y respetando los mismos bloques. Controlar el minuto del día evita llamar "agrupamiento" a que todos los días haya movimiento a la misma hora.
3. Estadístico principal: **qué fracción de los tiempos entre extremos es de {PRIMARY_WINDOW} barras o menos**, dividida por la fracción esperada. Cociente 1 = como si fueran independientes; 2 = el doble de extremos "pegados".
4. **Cuándo se considera detectado** (regla revisada en la auditoría, §12 y §13): el cociente debe superar la *banda calibrada* —lo máximo que produce, el 97,5 % de las veces, este mismo procedimiento completo aplicado a series que no tienen ninguna dependencia— y además su IC 95 % debe quedar por encima de 1.

**Referencia primaria `{PRIMARY_NULL}` (corregida, hallazgo A).** La probabilidad es el producto de un factor por minuto y un factor por año, ajustados para que el número esperado de extremos coincida con el observado *en cada minuto y en cada año a la vez* (ajuste proporcional iterativo). La fórmula original multiplicaba la tasa del minuto por (tasa del año / tasa global), lo que no conserva esos totales cuando los años no reparten sus barras igual entre minutos. Comprobación sobre los datos (caídas y subidas, nivel y definición primarios):

{_md_table(cons_tab, ["serie", "lado", "referencia", "observadas", "esperadas", "error máx. por minuto", "error máx. por año", "año peor (error relativo)", "barras con p = 0"])}

- *Minutos sin ningún extremo observado*: su probabilidad es 0 (columna "barras con p = 0"). Es lo coherente con lo observado, pero significa que la referencia no puede "esperar" extremos ahí.
- *Combinaciones minuto–año sin barras*: no intervienen (no hay exposición).
- *Celdas pequeñas*: el ajuste usa sólo totales por minuto y por año (dos márgenes), no una tasa por cada combinación, precisamente para no estimar miles de tasas con dos o tres casos cada una.
- Referencias secundarias: `{NULL_CLOCK}` (sin año), `{NULL_HOMOGENEOUS}` (una única tasa: el Poisson literal) y `{NULL_CLOCK_YEAR_LEGACY}` (la fórmula original, sólo para comparar). No se estima el índice extremal (G4).

**Qué cubren los intervalos (hallazgo C).** Hay dos tipos y no son intercambiables:

- **IC condicional**: remuestrea jornadas pero deja fijos los umbrales y la referencia estimados con toda la muestra. Mide sólo la variabilidad de los conteos. Es el que acompaña a todas las tablas de sensibilidad.
- **IC con re-estimación** (sólo para la variante primaria): en cada réplica se vuelven a estimar los umbrales de cada celda y la referencia por minuto y año sobre las jornadas remuestreadas. {result.n_boot_full} réplicas.
- **Ninguno de los dos cubre**: la estimación de `sigma_hat`, `z` y el decil de volatilidad (son atributos de cada barra que dependen de su propio pasado; remuestrear jornadas no los regenera), los cortes de segmento de TDA-06, ni la elección del estimador de volatilidad. Tampoco la dependencia entre jornadas. Son, por tanto, intervalos **condicionados a esas piezas**, no la incertidumbre total.

## 7. Resultado: crudo frente a estandarizado

**Variante primaria — la que decide** (definición `{PRIMARY_DEFINITION}`, nivel {100 * PRIMARY_LEVEL:g} %, referencia `{PRIMARY_NULL}`, tiempos ≤ {PRIMARY_WINDOW} barras):

{_md_table(rule_tab, list(rule_tab.columns))}

La anchura del IC con re-estimación es {width_txt} del condicional (menor que 1 = más estrecho). Re-estimar umbrales y referencia en cada réplica no tiene por qué ensanchar el intervalo: al recalcular el umbral, cada réplica vuelve a tener la misma tasa de extremos, y eso estabiliza el cociente. Ninguno de los dos intervalos es "el bueno" en abstracto; responden a preguntas distintas (§6).

**Comparación directa entre crudo y estandarizado** (mismas jornadas remuestreadas para ambos):

{_md_table(fr_tab, list(fr_tab.columns))}

**El mismo cociente según cómo se define el extremo** (cuánto del agrupamiento aparente es hora del día y volatilidad; IC condicional):

{_md_table(ladder_tab, cols_main)}

**Otras ventanas y otros estadísticos** (definición y nivel primarios; cociente observado/esperado [IC 95 % condicional]):

{_md_table(win_tab, list(win_tab.columns))}

- *En rachas ≥2*: fracción de los extremos que forman parte de una racha de dos o más barras consecutivas.
- *Dispersión diaria*: variabilidad del número de extremos por jornada frente a la esperada; mayor que 1 = hay jornadas que concentran muchos más extremos de lo previsto.

**Sensibilidad al nivel de cola:**

{_md_table(level_tab, ["serie", "nivel", "lado", "observado", "esperado si independientes", "cociente [IC95 condicional]", "excedencias", "tiempos medidos"])}

**Sensibilidad a la referencia** (incluye la fórmula original):

{_md_table(null_tab, ["serie", "referencia", "lado", "observado", "esperado si independientes", "cociente [IC95 condicional]"])}

**Sensibilidad a la estandarización** (otras semividas EWMA ya definidas en TDA-10; mismas barras y mismas celdas):

{_md_table(zsens_tab, ["serie", "semivida EWMA (min)", "lado", "observado", "esperado si independientes", "cociente [IC95 condicional]"])}
{zskip}

Las tablas de sensibilidad son **descriptivas**: sólo la variante primaria tiene banda calibrada e IC con re-estimación.

## 8. Tiempos entre excedencias y longitud de las rachas

Definición y nivel primarios, referencia `{PRIMARY_NULL}`. "Esperado" es lo que daría la independencia con la misma intensidad.

**`{SERIES_R}` — caídas: tiempos entre excedencias (barras)**

{_dist(g, SERIES_R, SIDE_NEG, "gap_bars", "barras")}

**`{SERIES_R}` — subidas**

{_dist(g, SERIES_R, SIDE_POS, "gap_bars", "barras")}

**`{SERIES_Z}` — caídas**

{_dist(g, SERIES_Z, SIDE_NEG, "gap_bars", "barras")}

**`{SERIES_Z}` — subidas**

{_dist(g, SERIES_Z, SIDE_POS, "gap_bars", "barras")}

**Longitud de las rachas (barras extremas consecutivas, ambas colas)**

`{SERIES_R}`:

{_dist(rn, SERIES_R, SIDE_BOTH, "run_length", "longitud")}

`{SERIES_Z}`:

{_dist(rn, SERIES_Z, SIDE_BOTH, "run_length", "longitud")}

Tablas completas (todas las definiciones, lados y referencias) en `{config.tda13_gap_distribution_csv_name}` y `{config.tda13_run_distribution_csv_name}`; gráfico en `{config.tda13_gap_distribution_png_name}`.

## 9. Por año y por segmento (G3)

Mismo cociente (tiempos ≤ {PRIMARY_WINDOW} barras), definición y nivel primarios, IC condicional. ⚠ = menos de {MIN_TAIL_SUPPORT_LOW} tiempos medidos; ✖ = menos de {MIN_TAIL_SUPPORT_UNRELIABLE}.

**Por año**

{_md_table(_group_tab("YEAR"), ["grupo", "lado", f"{SERIES_R}: cociente [IC95]", f"{SERIES_R}: tiempos", f"{SERIES_Z}: cociente [IC95]", f"{SERIES_Z}: tiempos"])}

**Por segmento horario**

{_md_table(_group_tab("SEGMENT"), ["grupo", "lado", f"{SERIES_R}: cociente [IC95]", f"{SERIES_R}: tiempos", f"{SERIES_Z}: cociente [IC95]", f"{SERIES_Z}: tiempos"])}

Dentro de un segmento sólo se miden los tiempos entre extremos del mismo segmento, así que los segmentos cortos tienen pocas medidas. Detalle (incluye la ventana de 1 barra y la fracción en rachas) en `{config.tda13_clustering_by_group_csv_name}`.

## 10. Distribución horaria de los extremos

Qué parte de los extremos cae en cada hora de Nueva York, según cómo se defina el umbral (ambas colas, nivel {100 * PRIMARY_LEVEL:g} %). Si los extremos no dependieran de la hora, cada columna se parecería a "% de las barras".

{_md_table(hour_tab, list(hour_tab.columns))}

Tasa de extremos de la hora más cargada respecto de la tasa media ("lift"), y de la menos cargada: con umbral global, de {_fmt(lift_g[0], 2)} a {_fmt(lift_g[1], 2)} veces; por segmento, de {_fmt(lift_s[0], 2)} a {_fmt(lift_s[1], 2)}; con la definición primaria, de {_fmt(lift_p[0], 2)} a {_fmt(lift_p[1], 2)}. Tasas con IC por hora en `{config.tda13_hourly_csv_name}`; gráfico en `{config.tda13_hourly_png_name}`.

**Cómo leerlo.** Con un umbral único, los "extremos" son sobre todo las horas de más actividad: eso no describe eventos, describe el horario. Con umbrales relativos el reparto se acerca al de las barras, pero no queda perfectamente plano: los cortes de segmento son anchos y dentro de un segmento sigue habiendo minutos más agitados que otros. Por eso la referencia de independencia usa la tasa de cada minuto del día y no una tasa única.

## 11. Efecto del triage sobre la curtosis y los cuantiles extremos

*Curtosis* mide cuánto pesan los valores extremos en la distribución (0 = una campana normal). Cada fila recalcula los estadísticos quitando una categoría; los resultados de TDA-07 y TDA-12 no se modifican. Entre corchetes, IC 95 % por jornadas. "quitadas" = observaciones retiradas respecto del escenario de referencia de su bloque.

**Retorno crudo**

{_md_table(_impact_wide(imp, SERIES_R), ["escenario", "n", "quitadas", "curtosis", "curtosis recortada", "p0,1", "p1", "p99", "p99,9"])}

**Retorno estandarizado `z`**

{_md_table(_impact_wide(imp, SERIES_Z), ["escenario", "n", "quitadas", "curtosis", "curtosis recortada", "p0,1", "p1", "p99", "p99,9"])}

**Lectura.**
- *Artefactos.* Con todos los retornos de la tabla, incluidos los que cruzan fronteras (`UNIVERSO_COMPLETO`), la curtosis es {_fmt(k_univ, 1)}. Quitar sólo los extremos de roll la deja en {_fmt(k_no_roll, 1)}, y quitar sólo los de frontera en {_fmt(k_no_bnd, 1)}. La población que usan TDA-07 y TDA-12 (`SOLO_VALIDOS`, curtosis {_fmt(k_valid, 1)}) ya excluye todos esos retornos: sus resultados no estaban contaminados por ellos.
- *Extremos clasificados `EVENTO_REAL`.* Quitarlos de los válidos deja la curtosis en {_fmt(k_no_real, 1)}, y quitar todos los extremos inventariados, en {_fmt(k_no_ext, 1)}: la cola pesada **es** el conjunto de estos movimientos, no un añadido. Borrarlos no "limpia" la serie; cambia lo que se está midiendo.
- *Sensibilidad a `INDETERMINADO`.* Sin ellos la curtosis pasa de {_fmt(k_valid, 1)} a {_fmt(k_no_und, 1)}. La tabla siguiente da el rango entre conservarlos y quitarlos:

{_md_table(range_tab, ["serie", "estadístico", "con INDETERMINADO", "sin INDETERMINADO", "cambio relativo"])}

Quitar una categoría entera de extremos recorta mecánicamente la cola (son, por definición, los valores más grandes), así que estas cifras son cotas de sensibilidad, no estimaciones "corregidas".

**Trazabilidad.** {ref_txt} Tabla completa, con diferencias e IC, en `{config.tda13_triage_impact_csv_name}`.

## 12. Calibración de la maquinaria (G2) — revisada en la auditoría

**Qué estaba mal (hallazgo B).** La versión anterior daba la calibración por buena si la *media* de los controles caía dentro de la banda. Eso no dice con qué frecuencia la regla "detecta" algo que no existe. Además, la banda salía de simular extremos independientes con probabilidades fijas, ignorando que los umbrales, los estados de volatilidad y las propias tasas se estiman de los datos.

**Diagnóstico de la causa.** Cociente (tiempos ≤ {PRIMARY_WINDOW} barras) en mundos donde no hay ninguna dependencia:

{_md_table(diag_tab, ["serie", "lado", "simulación", "n", "media", "desviación", "p97,5"])}

Las dos primeras filas de cada bloque son extremos independientes "de laboratorio": si el procedimiento completo (filas siguientes) tiene más media o más dispersión que ellas, la banda antigua era demasiado estrecha o estaba desplazada. {legacy_txt} La causa compatible con estos números es que, dentro de un mismo minuto del día, la probabilidad de superar el umbral todavía depende del estado de volatilidad de la barra, y ese estado es persistente; la referencia por minuto no lo recoge. Es una explicación coherente con el diagnóstico, no una demostración.

**Qué se hace ahora.** El procedimiento COMPLETO (estandarizar, asignar el estado de volatilidad, fijar umbrales, ajustar la referencia, medir) se aplica a series sin dependencia temporal: (i) `PERMUTACION_GLOBAL`, los retornos reales barajados; (ii) `SIMULADO_PERFIL_RELOJ`, los retornos reales barajados dentro de cada minuto del día (misma volatilidad por hora y misma curtosis, agregadas entre años); (iii) `{NULL_FAMILY_MINUTE_YEAR}`, los retornos reales barajados dentro de cada minuto del día **y de cada año** (añadida en la segunda auditoría). {result.n_calibration} réplicas de cada una. La mitad fija la banda; la otra mitad, que no intervino, mide las falsas detecciones.

**Por qué una tercera familia (segunda auditoría, hallazgo H3).** La referencia de independencia condiciona por minuto y por año, pero las familias (i) y (ii) mezclan los años al barajar. Un diagnóstico con series sintéticas *sin ninguna dependencia temporal* mostró que, si la forma del perfil horario cambia de un año a otro, esas dos familias "detectan" agrupamiento en `r` la mayoría de las veces. La familia (iii) conserva la distribución de cada minuto en cada año, que es justo lo que faltaba.

- *Qué conserva (iii):* el conjunto exacto de retornos de cada (minuto de Nueva York, año): su volatilidad, sus colas, el nivel de cada año y la forma del perfil horario de cada año.
- *Qué rompe:* el orden de las jornadas dentro de cada año (rachas de días agitados, día de la semana, deriva dentro del año, días de evento). Eso es exactamente lo que la hipótesis "sin dependencia temporal" niega.
- *Horario de verano:* el minuto es el de la hora local de Nueva York (el mismo `minute_of_day` de TDA-06 que usa la referencia), así que una misma hora de mercado cae siempre en el mismo grupo. El día del cambio de hora de primavera no existen los minutos de la hora saltada, y el de otoño una hora local se repite: esos grupos tienen una barra menos o una más, sin otro efecto.
- *Año:* el de la hora local de la barra (`year_ny`, el de la referencia). Las barras de la tarde del 31 de diciembre pertenecen a la jornada del año siguiente pero se agrupan con su año local.
- *Grupos pequeños:* no se fusionan ni se descartan; un grupo de una sola barra queda fijo. Tamaños reales:

{groups_md}

**Cómo se combinan las tres familias, y por qué.** Una cola se da por detectada sólo si el cociente supera la banda de **todas** las familias, es decir, la mayor de las tres. No se da por hecho que "tomar el máximo" lo arregle todo; lo que se puede demostrar es esto:

1. Cada familia es un contraste de permutación. Si los retornos son intercambiables dentro de sus grupos, la serie real es "una permutación más" y su cociente —con toda la re-estimación incluida— es intercambiable con los de las réplicas. Con {n_cal_half} réplicas para la banda, la probabilidad de superarla es entonces como mucho **{size_bound_txt}** por cola (cota exacta; el {_pct(G2_NOMINAL_RATE, 1)} es el valor nominal, y la diferencia viene de interpolar el percentil).
2. Las tres hipótesis están anidadas: intercambiable en todo ⊂ intercambiable dentro de cada minuto ⊂ intercambiable dentro de cada (minuto, año). Por eso la banda de la familia (iii) es válida por sí sola en los tres mundos.
3. Exigir además las bandas de (i) y (ii) sólo puede subir la banda: es más conservador y pierde potencia, pero nunca es menos válido. Se mantienen porque eran el criterio ya establecido.
4. **Lo que el máximo NO resuelve:** mundos sin dependencia que no sean intercambiables dentro de (minuto, año) —por ejemplo, un perfil horario que cambie de un mes a otro o según el día de la semana—. Para esos no hay garantía. Tampoco corrige la referencia: en un mundo con interacción año × hora el cociente "esperado si independientes" sigue estando mal centrado, y es la banda la que lo absorbe.

**Banda calibrada** (la más exigente de las tres familias):

{_md_table(band_tab, list(band_tab.columns))}

**Criterio de aceptación de G2** (misma regla que antes; al pasar de 2 a 3 familias las tasas que entran son 12 en vez de 8 y el máximo admitido se recalcula con la misma fórmula): en las tres familias sin dependencia, el número de detecciones de cada cola (`r` y `z`, caídas y subidas: {n_acc} tasas) entre las {n_cal_half} réplicas reservadas no puede superar un máximo (columna "máx. admitido"). Ese máximo es una cota de lo que cabe esperar, con un 95 % de confianza conjunta para las {n_acc} tasas, cuando las dos mitades proceden del mismo mundo; tiene en cuenta que la banda está estimada con un número finito de réplicas (sin eso, el simple error de estimar la banda haría fallar el criterio a un procedimiento correcto). Las reglas globales nunca pueden detectar más que la cola que menos detecta, así que quedan cubiertas; se muestran con su intervalo.

{_md_table(rates_ind, ["control", "regla", "réplicas", "detecciones", "máx. admitido", "tasa [IC95 binomial]", "con la banda previa [IC95]", "¿cumple?"])}

**G2: {cal_ok_txt}.** {worst_txt} "Compatible" significa que el criterio **no rechaza** la calibración en las familias evaluadas; no certifica una tasa del {_pct(G2_NOMINAL_RATE, 1)} ni cubre mundos sin dependencia distintos de esas familias (§16). La lectura correcta son las tasas de la tabla con sus intervalos: en la peor cola el intervalo llega hasta {worst_hi_txt}. La regla aplicada a los datos reales exige además que el IC quede por encima de 1, lo que sólo puede reducir estas tasas. "Ambas colas juntas" y la fila "sin exigir r" se muestran pero no forman parte del criterio.

{power_txt}

**Controles de "sólo volatilidad"** ({result.n_volatility_control} réplicas de cada uno). Aquí los retornos se generan como *volatilidad × ruido independiente*, con una senda de volatilidad tomada de los datos. Por construcción, todo agrupamiento que aparezca es volatilidad:

{_md_table(rates_vol, ["control", "regla", "réplicas", "detecciones", "tasa [IC95 binomial]"])}

- La afirmación 1 se detecta en estos mundos con frecuencia {r_pos_txt}: es lo esperado (ahí sí hay agrupamiento) y sirve de control positivo.
- Lo importante es la fila de la afirmación 2: **con qué frecuencia se diría "persiste tras estandarizar" en un mundo donde, por construcción, sólo hay volatilidad.** Si esa tasa es alta, "persiste" no distingue nada.

Detalle por réplica en `{config.tda13_calibration_csv_name}`; tasas y bandas en `{config.tda13_false_detection_csv_name}`.

## 13. Veredictos

### TH06 — triage de extremos: `{th06['verdict']}` (clasificación operativa)

De {_fmt(th06['n_extremes'])} extremos: {_pct(frac[CAT_REAL], 1)} `EVENTO_REAL`, {_pct(frac[CAT_UNDETERMINED], 1)} `INDETERMINADO`, {_pct(frac[CAT_BOUNDARY], 2)} `ARTEFACTO_FRONTERA`, {_pct(frac[CAT_ROLL], 3)} `ARTEFACTO_ROLL`, {_pct(frac[CAT_ERROR], 3)} `ERROR_DATO`.

**Qué respalda cada categoría (hallazgo D):**

| Categoría | En qué se apoya |
|---|---|
{basis_md}

- **`EVENTO_REAL` no es un evento verificado.** Significa: retorno válido, barras activas, registro continuo alrededor, precio que no retrocede más de la mitad en el minuto siguiente y volumen no bajo. Eso es evidencia favorable, no una comprobación independiente. Un precio erróneo que se mantuviera en la barra siguiente, o un desplazamiento de nivel causado por la fuente de datos, pasaría estas comprobaciones. Con barras de 1 minuto y sin una segunda fuente no hay forma de descartarlo caso por caso. La columna `category_basis` del inventario lo deja escrito en cada fila.
- **`ERROR_DATO` = 0 no significa "no hay errores"**: significa que ninguna barra extrema viola los invariantes físicos que TDA-00 comprueba.
- **"Mayoría clasificable"** quiere decir que las reglas asignan una categoría distinta de `INDETERMINADO` a más de la mitad (lectura literal del backlog; el {100 * TH06_MAJORITY:g} % es esa lectura, no un umbral ajustado). No quiere decir que la mayoría esté verificada.
- La fracción de artefactos y errores es {_pct(th06['fraction_artefacts_and_errors'])}, y todos están fuera de la población de análisis desde TDA-04.

**Sensibilidad de la lectura a las reglas de triage** (descriptivo; la clasificación vigente no cambia):

{_md_table(sens_tab, ["tipo", "variante", "% INDETERMINADO", "% EVENTO_REAL"])}

Entre las variantes de regla, `INDETERMINADO` va del {_pct(und_lo, 1)} al {_pct(und_hi, 1)}: la lectura `{th06['verdict']}` {'se mantiene en todas' if th06.get('robust_to_rules') else 'NO se mantiene en todas'}. Las dos cotas muestran de qué depende en el fondo: si no se acepta la evidencia favorable de R5 como suficiente, **todo** retorno válido queda `INDETERMINADO`. TH06 es, por tanto, una conclusión condicionada a aceptar esa evidencia. La caracterización de la cola debe darse **como rango** (con y sin `INDETERMINADO`, §11).

### TH27 — agrupamiento de extremos: `{TH27_DISPLAY[verdict27]}`

TH27 se separa en tres afirmaciones, porque la segunda no demuestra la tercera.

| Afirmación | Estado | Tipo de evidencia |
|---|---|---|
| 1. Los extremos se agrupan más de lo esperable si fueran independientes | `{CLAIM_DISPLAY.get(c1, c1)}` | Banda del procedimiento completo; G2 {cal_ok_txt.lower()} |
| 2. El agrupamiento persiste después de dividir por la volatilidad estimada (EWMA, semivida {PRIMARY_ESTIMATOR[1]:g} min) | `{CLAIM_DISPLAY.get(c2, c2)}` | Calibrada frente a la independencia; **no** frente a "sólo volatilidad" |
| 3. Existe un agrupamiento propio de los extremos, no explicable por cambios de volatilidad | `NO IDENTIFICADO POR ESTE PROCEDIMIENTO` | Sólo evidencia descriptiva; este método no puede afirmarlo ni descartarlo |

{_side_line(SIDE_NEG)}
{_side_line(SIDE_POS)}
{gate_txt}

**Por qué la afirmación 2 no demuestra la 3.** "Estandarizar" es dividir por una volatilidad *estimada* con una media móvil que reacciona con retraso. Si la volatilidad sube de golpe, durante unos minutos la estimación se queda corta, `z` sale grande varias barras seguidas y aparece agrupamiento aunque los retornos sean independientes dada la volatilidad verdadera. Esto se comprueba de dos maneras:

- *Sensibilidad a la semivida* (§7): cuanto más rápido reacciona el estimador, menos agrupamiento queda.
- *Controles de sólo volatilidad* (§12): en mundos donde, por construcción, sólo hay volatilidad, se compara el cociente de `z` que resulta con el observado:

{_md_table(vol_tab, ["control", "lado", "réplicas", "cociente real de z", "cociente de z en el control: media [p2,5; p97,5]", "parte del exceso real que el control reproduce", "¿el real cabe en el control?", "tasa de 'persiste' en el control"])}

{claim3_txt}

**Alcance exacto de esta conclusión.** Lo que queda establecido es una limitación del procedimiento, no una propiedad del mercado:

- *Lo que sí se sabe:* dividir por una volatilidad estimada con retraso deja agrupamiento residual aunque no exista ningún fenómeno propio de los extremos. Por tanto, que el agrupamiento persista en `z` no basta para identificar ese fenómeno.
- *Lo que no se sabe:* si ese fenómeno existe. Este procedimiento no lo identifica; tampoco lo descarta. **Ausencia de prueba no es prueba de ausencia**, en ninguna de las dos direcciones.
- *Lo que NO se afirma:* que los dos fenómenos sean imposibles de distinguir. Aquí no se distinguen porque la única medida de volatilidad disponible se construye con los mismos retornos de 1 minuto cuyos extremos se cuentan, y la volatilidad verdadera no es observable con OHLCV (G0). Con una medida de volatilidad independiente del retorno de la barra (por ejemplo, calculada con datos de mayor frecuencia), con otro diseño de contraste o con un modelo explícito, la separación podría ser posible. Eso queda fuera de esta etapa y no se ha intentado.

**En palabras sencillas.** {plain27}

**Qué NO afirma este veredicto.**
- No afirma un fenómeno adicional al agrupamiento de volatilidad de TDA-09, ni lo descarta.
- No se ha estimado el índice extremal ni ningún modelo de cola.
- Es exploratorio y retrospectivo: no está confirmado en el holdout.

## 14. Auditoría correctiva: qué cambió y cuánto

{prev_note}

| Hallazgo | Causa | Corrección |
|---|---|---|
| A. Referencia `{PRIMARY_NULL}` | La fórmula `p(minuto) × tasa(año)/tasa(global)` no conserva los totales por minuto y por año a la vez | Ajuste proporcional iterativo que conserva ambos (§6). La fórmula original queda como referencia secundaria de comparación |
| B. Calibración G2 | Criterio "la media cae dentro" + banda que ignora la estimación de umbrales, estados y tasas | Banda obtenida del procedimiento completo sobre series nulas; tasas de falsa detección con intervalo binomial en réplicas reservadas; criterio de aceptación escrito (§12) |
| C. Intervalos | El IC remuestreaba conteos con umbrales y referencia fijos | Bootstrap que re-estima umbrales y referencia para la variante primaria; el resto se rotula "condicional"; se declara lo que ninguno cubre (§6) |
| D. TH06 | `EVENTO_REAL` se leía como verificación | Las categorías no cambian; se explicita que es una clasificación operativa y se añade la sensibilidad a las reglas (§13) |
| H3 (2.ª auditoría). Familias de control | Las dos familias mezclaban los años: no cubrían un mundo sin dependencia con perfil horario distinto cada año | Tercera familia `PERMUTACION_MINUTO_X_ANIO`; banda = la mayor de las tres; criterio de G2 sobre 12 tasas (§12) |
| H1–H2 (2.ª auditoría). Criterio de G2 | El máximo admitido se llamaba "exacto" sin serlo, y "G2 superada" sugería una tasa demostrada | Cota conservadora exacta (estadístico de orden inferior); rótulo "compatible con los controles evaluados" y tabla de potencia (§12) |
| H5 (2.ª auditoría). Bloques | Filtrar los bloques con la población común podría unir tramos separados por una barra excluida | `restrict_block_ids` (sin efecto en los datos reales: las exclusiones son sólo el tramo inicial) |
| TH27 | Un único veredicto mezclaba "persiste con el EWMA" con "fenómeno propio" | Tres afirmaciones separadas; controles de sólo volatilidad (§13) |

**Antes y después, variante primaria:**

{_md_table(prev_tab, list(prev_tab.columns))}

**Efecto de corregir sólo la referencia** (misma muestra, mismos umbrales; IC condicional):

{_md_table(legacy_cmp, ["serie", "referencia", "lado", "observado", "esperado si independientes", "cociente [IC95 condicional]"])}

Se conservaron sin cambios, como base de comparación: nivel {100 * PRIMARY_LEVEL:g} %, definición `{PRIMARY_DEFINITION}`, ventana de {PRIMARY_WINDOW} barras, series `r` y `z`, semivida {PRIMARY_ESTIMATOR[1]:g} min y sus sensibilidades, y las reglas de triage R1–R5. La foto de los resultados previos (con los hash de los artefactos anteriores) está en `{config.tda13_audit_previous_csv_name}`.

## 15. Consecuencias para el resto del roadmap

- **TDA-13 Parte B (EVT) no se ejecuta.** Su criterio de entrada 1 exige que TDA-12 hubiera concluido `ESCALA Y FORMA`; TDA-12 cerró `NO CONCLUYENTE`. Nada de esta etapa cambia ni reinterpreta ese resultado. `STOP-13` sigue no activado; la rama EVT sigue no autorizada.
- **Tamaño de muestra efectivo de la cola.** {('Las excedencias no son observaciones independientes: ' + ', '.join(f"{_pct(float(r_['observed']), 1)} de los extremos de {r_['series']} forma parte de una racha de dos o más barras" for _, r_ in run_frac.loc[run_frac['side'] == SIDE_BOTH].iterrows()) + '. Esto es cierto sea cual sea su origen (volatilidad o no) y afecta a la incertidumbre de cualquier cuantil extremo. Los IC por jornadas de etapas anteriores ya incorporan la dependencia dentro del día, no la que cruza de un día a otro.') if c1 == CLAIM_YES else 'No se ha establecido agrupamiento con la calibración exigida: no se deriva ninguna corrección.'}
- **TDA-14** (estabilidad temporal) es la siguiente etapa; aquí no se anticipa.
- **Decisiones de IRIS:** ninguna.

## 16. Limitaciones

1. Todo es **exploratorio y retrospectivo**: umbrales, referencias y cortes de segmento usan toda la muestra, y nada se ha confirmado en el holdout (G2).
2. La definición de extremo, las reglas de triage, la referencia `{PRIMARY_NULL}`, la ventana de {PRIMARY_WINDOW} barras, las familias de control, el criterio de aceptación de G2 y la regla de veredicto son **decisiones de esta etapa** (las tres últimas, introducidas en la auditoría). Ninguna estaba aprobada previamente. Los resultados se leen condicionados a ellas.
3. La afirmación 3 de TH27 **no queda identificada por este procedimiento** (§13): no se afirma ni se descarta, y no se sostiene que sea imposible de distinguir con otros datos o métodos. Los controles de sólo volatilidad usan tres sendas elegidas en la auditoría; ninguna es "la volatilidad verdadera", y la más rápida es retrospectiva y opera a la misma escala que la ventana de medida.
4. La banda calibrada depende de qué mundos sin dependencia se consideran. Se usan tres familias de permutación (global, por minuto, por minuto y año). La garantía de §12 vale para mundos intercambiables dentro de (minuto, año); un mundo sin dependencia con otra heterogeneidad (por mes, por día de la semana, por tipo de jornada) no está cubierto y podría producir falsas detecciones. La familia por minuto y año destruye además el orden de las jornadas dentro del año, de modo que cualquier persistencia entre días —incluida la de la volatilidad— cuenta como "dependencia": la afirmación 1 no distingue entre ellas. La distancia entre los cocientes reales y la banda indica cuánto margen hay.
5. La calibración mide falsas detecciones con {n_cal_half} réplicas reservadas por familia: detecta desviaciones apreciables del nominal, no de décimas de punto.
6. El IC con re-estimación no cubre la estimación de `sigma_hat`, `z` y el decil, ni los cortes de segmento, ni la dependencia entre jornadas (§6). Los demás IC son condicionales.
7. `EVENTO_REAL` es una clasificación operativa, no una verificación. Sin datos tick a tick ni una segunda fuente no se puede confirmar el origen de un extremo (G0). `ERROR_DATO` sólo detecta lo que TDA-00 detecta.
8. No hay calendario fechado de eventos: ninguna atribución causal. La marca de las 08:30 es una coincidencia de reloj.
9. La referencia pone probabilidad 0 en los minutos sin ningún extremo observado, y estima sus tasas con los mismos datos.
10. El agrupamiento sólo se mide dentro de bloques continuos: no dice nada sobre si una jornada con extremos va seguida de otra, salvo por el índice de dispersión diaria.
11. En las celdas pequeñas y en el nivel {100 * min(SENSITIVITY_LEVELS):g} % hay pocas excedencias; esos resultados son orientativos.
12. Reutilizar la firma de "tick aislado" de TDA-03 y los niveles de cuantil de TDA-07 no significa que su uso aquí estuviera aprobado: los valores son heredados, la aplicación es nueva.

{timing_text}

## 18. Archivos generados

`{config.tda13_extremes_csv_name}` (inventario con triage), `{config.tda13_thresholds_csv_name}`, `{config.tda13_triage_summary_csv_name}`, `{config.tda13_triage_sensitivity_csv_name}`, `{config.tda13_clustering_csv_name}` (observado frente a la referencia: todas las series, definiciones, niveles, lados y referencias), `{config.tda13_gap_distribution_csv_name}`, `{config.tda13_run_distribution_csv_name}`, `{config.tda13_clustering_by_group_csv_name}`, `{config.tda13_hourly_csv_name}`, `{config.tda13_triage_impact_csv_name}`, `{config.tda13_reference_conservation_csv_name}`, `{config.tda13_calibration_csv_name}` (réplicas), `{config.tda13_false_detection_csv_name}` (tasas y bandas), `{config.tda13_full_bootstrap_csv_name}`, `{config.tda13_volatility_control_csv_name}` (15 CSV) + `{config.tda13_clustering_png_name}`, `{config.tda13_gap_distribution_png_name}`, `{config.tda13_hourly_png_name}`, `{config.tda13_triage_png_name}` (4 PNG) + este informe. `{config.tda13_audit_previous_csv_name}` es una foto fija de los resultados previos a la auditoría: se lee, no se regenera.

## 19. Modo sencillo

**¿Qué se hizo?** Se listaron los movimientos de 1 minuto más grandes de MNQ —grandes *para su hora y para lo agitado que estaba el mercado*— y se revisó cada uno con las comprobaciones de calidad de las etapas anteriores.

**¿Son errores?** Ninguna comprobación demuestra errores de datos entre ellos ({_fmt(th06['counts'][CAT_ERROR])} casos), lo que no garantiza que no los haya. Los artefactos conocidos (saltos al cambiar de contrato o al cruzar un cierre) son el {_pct(frac[CAT_ROLL] + frac[CAT_BOUNDARY])} y ya estaban fuera de todos los análisis. El {_pct(frac[CAT_REAL], 0)} no muestra ningún indicio de artefacto y tiene señales favorables (se clasifica como movimiento real, sin poder verificarlo); el {_pct(frac[CAT_UNDETERMINED], 0)} restante no se puede decidir con barras de 1 minuto.

**¿Llegan en grupos?** {plain27}

**¿Qué NO significa?** No es una señal ni una regla de operación, no predice nada y no decide nada del diseño de IRIS. Tampoco se ha modelado la cola (EVT): eso sigue sin autorizar.

## 20. Estado

{status_line(closure)}

| Elemento | Estado |
|---|---|
| TH06 | `{th06['verdict']}` — clasificación operativa; colas reportadas como rango |
| TH27, afirmación 1 (agrupamiento frente a independencia) | `{CLAIM_DISPLAY.get(c1, c1)}` |
| TH27, afirmación 2 (persiste con el EWMA de {PRIMARY_ESTIMATOR[1]:g} min) | `{CLAIM_DISPLAY.get(c2, c2)}` |
| TH27, afirmación 3 (agrupamiento propio, ajeno a la volatilidad) | `NO IDENTIFICADO POR ESTE PROCEDIMIENTO` (ni afirmado ni descartado) |
| TH27, veredicto | `{TH27_DISPLAY[verdict27]}` |
| Calibración G2 | {cal_ok_txt} |
| TDA-13 Parte B (EVT) / TH28 | no ejecutada; no autorizada |
| `STOP-13` | sin cambios respecto de TDA-12 (no activado) |

Todos los resultados son exploratorios (G2) y dependen de las decisiones listadas en §16.
"""


def persist_artifacts(result: TDA13Result, config: SnapshotConfig, t0: float, run_command: str, closure: dict | None = None) -> list[Path]:
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    result.inventory.to_csv(config.tda13_extremes_csv_path, index=False)
    result.thresholds.to_csv(config.tda13_thresholds_csv_path, index=False)
    result.triage_summary.to_csv(config.tda13_triage_summary_csv_path, index=False)
    result.cluster_table.to_csv(config.tda13_clustering_csv_path, index=False)
    result.gap_distribution.to_csv(config.tda13_gap_distribution_csv_path, index=False)
    result.run_distribution.to_csv(config.tda13_run_distribution_csv_path, index=False)
    result.cluster_by_group.to_csv(config.tda13_clustering_by_group_csv_path, index=False)
    result.hourly.to_csv(config.tda13_hourly_csv_path, index=False)
    result.triage_impact.to_csv(config.tda13_triage_impact_csv_path, index=False)
    reps = result.null_replicates
    fam = reps.loc[~reps["kind"].isin([KIND_FULL_BOOTSTRAP, KIND_BERNOULLI_DIAG])]
    fam.to_csv(config.tda13_calibration_csv_path, index=False)
    rates = result.calibration_rates.copy()
    rates.insert(0, "table", "false_detection_rate")
    bands = result.calibration_bands.copy()
    bands.insert(0, "table", "calibrated_band")
    diag = result.bernoulli_diagnostic.copy()
    diag.insert(0, "table", "band_diagnostic")
    pd.concat([rates, bands, diag], ignore_index=True).to_csv(config.tda13_false_detection_csv_path, index=False)
    result.full_bootstrap.to_csv(config.tda13_full_bootstrap_csv_path, index=False)
    result.reference_conservation.to_csv(config.tda13_reference_conservation_csv_path, index=False)
    result.triage_sensitivity.to_csv(config.tda13_triage_sensitivity_csv_path, index=False)
    result.th27["volatility_only_evidence"].to_csv(config.tda13_volatility_control_csv_path, index=False)

    _draw_clustering(result, config.tda13_clustering_png_path)
    _draw_gaps(result, config.tda13_gap_distribution_png_path)
    _draw_hourly(result, config.tda13_hourly_png_path)
    _draw_triage(result, config.tda13_triage_png_path)

    config.tda13_report_path.write_text(render_report(result, config, time.perf_counter() - t0, run_command, closure), encoding="utf-8")
    return [getattr(config, attr) for attr in ARTIFACT_PATH_ATTRS]


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Ejecuta TDA-13 Parte A (inventario, triage y agrupamiento de extremos; TH06, TH27).")
    parser.add_argument("--config", default="configs/mnq_snapshot.yaml")
    parser.add_argument("--closure", default=None, help="archivo de cierre (YAML) con el estado y las incidencias de ejecucion; opcional")
    parser.add_argument("--amend-report", action="store_true",
                        help="NO ejecuta el analisis: reescribe solo las secciones 17 y 20 del informe existente a partir de --closure")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))
    closure = load_closure(args.closure)
    if args.amend_report:
        if closure is None:
            parser.error("--amend-report requiere --closure")
        try:
            changed = amend_report(config.tda13_report_path, closure)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"TDA-13A amend-report: FAIL\nMotivo: {exc}", file=sys.stderr)
            return 1
        print(f"TDA-13A amend-report: {'secciones 17 y 20 actualizadas' if changed else 'sin cambios (ya estaba actualizado)'} en {config.tda13_report_path}")
        return 0
    run_command = "python -m ohlcv_dataroad.ingest.run_tda13 --config " + str(args.config)

    t0 = time.perf_counter()
    try:
        result = run_tda13_analysis(config, verbose=True)
    except (
        FileNotFoundError, HoldoutIsolationError, TimestampAlignmentError, NaiveReturnContradictionError,
        LookaheadLeakageError, CausalDecileLeakageError, MissingEvidenceError, RuntimeError,
    ) as exc:
        print("TDA-13A status: FAIL", file=sys.stderr)
        print(f"Motivo: {exc}", file=sys.stderr)
        return 1

    print(f"[TDA13 {TOTAL_STAGES}/{TOTAL_STAGES}] Persistencia de artefactos (CSV/PNG/MD)...")
    written = persist_artifacts(result, config, t0, run_command, closure)
    total = time.perf_counter() - t0

    print()
    print("TDA-13A status: analisis completado")
    print(f"universo: {result.n_universe:,}; validos: {result.n_valid:,}; poblacion comun: {result.n_eligible:,}; jornadas: {result.n_days:,}")
    print(f"extremos inventariados: {len(result.inventory):,} -> {result.th06['counts']}")
    print(f"TH06 = {result.th06['verdict']}")
    print(f"TH27 = {result.th27['verdict']} (afirmacion 1: {result.th27['claim_1']}; afirmacion 2: {result.th27['claim_2']}; afirmacion 3: {result.th27['claim_3']}; G2: {result.th27.get('g2_status', g2_status(result.th27['g2_ok']))})")
    print(f"Tiempo total: {total:.1f}s (~{total / 60:.1f} min)")
    print()
    print("Artefactos escritos:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
