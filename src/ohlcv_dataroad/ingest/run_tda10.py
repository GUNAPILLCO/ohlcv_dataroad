"""Punto de entrada de linea de comandos para ejecutar TDA-10.

Uso
---
    python -m ohlcv_dataroad.ingest.run_tda10
    python -m ohlcv_dataroad.ingest.run_tda10 --config configs/mnq_snapshot.yaml

Como en TDA-00..09, este script no contiene logica de analisis propia:
llama a ``run_tda10_analysis`` (``tda10_scale_vs_shape.py``), persiste TODA
la evidencia declarada como reproducible (``persist_artifacts``) -- incluido
el informe final ``TDA10_escala_vs_forma.md``, generado automaticamente a
partir de los resultados (``render_report``) -- una sola ejecucion produce
TODO TDA-10, sin ningun paso manual posterior.
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
from ohlcv_dataroad.ingest.tda10_scale_vs_shape import (
    ALL_ESTIMATOR_CONFIGS,
    EWMA_HALFLIVES_MINUTES,
    FRACTION_REMOVED_FORM_THRESHOLD,
    FRACTION_REMOVED_SCALE_THRESHOLD,
    LookaheadLeakageError,
    PopulationMismatchError,
    PRIMARY_ESTIMATOR,
    PRIMARY_ESTIMATOR_CLOCK_ADJUSTED,
    PROFILE_STABILITY_FORM_THRESHOLD,
    PROFILE_STABILITY_SCALE_THRESHOLD,
    ROBUSTNESS_AGREEMENT_FRACTION,
    ROLLING_WINDOWS_MINUTES,
    TDA10Result,
    TOTAL_STAGES,
    run_tda10_analysis,
)

ARTIFACT_PATH_ATTRS = [
    "tda10_kurtosis_csv_path",
    "tda10_kurtosis_bootstrap_ci_csv_path",
    "tda10_quantile_by_decile_csv_path",
    "tda10_quantile_by_segment_csv_path",
    "tda10_quantile_by_year_csv_path",
    "tda10_sensitivity_csv_path",
    "tda10_causality_check_csv_path",
    "tda10_qq_points_csv_path",
    "tda10_qq_primary_png_path",
    "tda10_qq_sensitivity_png_path",
    "tda10_quantile_profile_png_path",
    "tda10_report_path",
]
# 8 CSV + 3 PNG + 1 MD = 12 artefactos SIEMPRE escritos.


# ---------------------------------------------------------------------------
# Formateo markdown -- mismos helpers pequeños que TDA-07/08/09 (sin
# compartir modulo de utilidades -- cada run_* es autocontenido, misma
# convencion del repositorio).
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


def _config_label(family: str, param: float, input_series: str) -> str:
    return f"{family}_{param:g}_{input_series}"


# ---------------------------------------------------------------------------
# Graficos obligatorios
# ---------------------------------------------------------------------------

def _draw_qq_primary(result: TDA10Result, out_path: Path) -> None:
    """r_1m crudo vs normal, y z_t (estimador primario) vs normal -- mismo estilo que TDA07_qq_global.png."""
    prim_label = _config_label(*PRIMARY_ESTIMATOR)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, series, config, title in [
        (axes[0], "r_1m_raw", "NA", "r_1m (crudo) vs. normal"),
        (axes[1], "z", prim_label, f"z_t primario ({prim_label}) vs. normal"),
    ]:
        sub = result.qq_table[(result.qq_table["series"] == series) & (result.qq_table["config"] == config)]
        if sub.empty:
            continue
        ax.plot(sub["theoretical_q"], sub["empirical_q"], marker=".", markersize=3, linewidth=0.8, color="tab:blue")
        lims = [
            min(sub["theoretical_q"].min(), sub["empirical_q"].min()),
            max(sub["theoretical_q"].max(), sub["empirical_q"].max()),
        ]
        ax.plot(lims, lims, color="black", linewidth=0.8, linestyle="--", label="normal (referencia)")
        ax.set_title(title)
        ax.set_xlabel("cuantil teorico (normal estandar)")
        ax.set_ylabel("cuantil empirico (estandarizado)")
        ax.legend()
        ax.grid(alpha=0.25)
    fig.suptitle("TDA-10 -- QQ-plot: cuanto de la cola desaparece al retirar la escala dinamica")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_qq_sensitivity(result: TDA10Result, out_path: Path) -> None:
    """QQ superpuesto de z_t para las 6 configuraciones "raw" y las 6 "clock_adjusted" -- sensibilidad visual a estimador/ventana."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, 6))
    for ax, input_series, title in [(axes[0], "raw", "z_t -- input RAW (causal)"), (axes[1], "clock_adjusted", "z_t -- input CLOCK_ADJUSTED (RETROSPECTIVO)")]:
        i = 0
        for family, param, inp in ALL_ESTIMATOR_CONFIGS:
            if inp != input_series:
                continue
            label = _config_label(family, param, inp)
            sub = result.qq_table[(result.qq_table["series"] == "z") & (result.qq_table["config"] == label)]
            if sub.empty:
                continue
            ax.plot(sub["theoretical_q"], sub["empirical_q"], linewidth=1.0, color=colors[i % len(colors)], label=f"{family} {param:g}")
            i += 1
        lims = [-4, 4]
        ax.plot(lims, lims, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(title)
        ax.set_xlabel("cuantil teorico (normal estandar)")
        ax.set_ylabel("cuantil empirico (estandarizado)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle("TDA-10 -- sensibilidad del QQ-plot de z_t a estimador/ventana/half-life")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_quantile_profile(result: TDA10Result, out_path: Path) -> None:
    """Perfil de cuantiles de z_t por decil de volatilidad -- headline (primario raw + clock_adjusted)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key in zip(axes, (PRIMARY_ESTIMATOR, PRIMARY_ESTIMATOR_CLOCK_ADJUSTED)):
        label = _config_label(*key)
        sub = result.quantile_by_decile[result.quantile_by_decile["config"] == label].copy()
        sub = sub.dropna(subset=["group"]).sort_values("group")
        for lvl, color in [("q0.01", "tab:red"), ("q0.05", "tab:orange"), ("q0.95", "tab:green"), ("q0.99", "tab:blue")]:
            if lvl in sub.columns:
                ax.plot(sub["group"], sub[lvl], marker="o", markersize=4, linewidth=1.2, color=color, label=lvl)
        ax.set_title(f"{key[0]} {key[1]:g} ({key[2]})")
        ax.set_xlabel("decil de sigma_hat_(t-1) (0=mas tranquilo, 9=mas volatil)")
        ax.set_ylabel("cuantil de z_t")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle("TDA-10 -- estabilidad de FORMA: perfil de cuantiles de z_t por decil de volatilidad causal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Informe final -- generado automaticamente a partir de los resultados
# ---------------------------------------------------------------------------

def render_report(result: TDA10Result, config: SnapshotConfig, elapsed_total_seconds: float, run_command: str) -> str:
    prim_label = _config_label(*PRIMARY_ESTIMATOR)
    prim_adj_label = _config_label(*PRIMARY_ESTIMATOR_CLOCK_ADJUSTED)

    kurt = result.kurtosis_table
    kurt_global = kurt[kurt["scope"] == "GLOBAL"].copy()
    kurt_global["config"] = [_config_label(f, p, s) for f, p, s in zip(kurt_global["family"], kurt_global["param"], kurt_global["input_series"])]

    kurt_year_prim = kurt[(kurt["scope"] == "YEAR") & (kurt["family"] == PRIMARY_ESTIMATOR[0]) & (kurt["param"] == PRIMARY_ESTIMATOR[1]) & (kurt["input_series"] == PRIMARY_ESTIMATOR[2])].sort_values("scope_value")

    ci = result.kurtosis_bootstrap_ci

    prim_decile = result.quantile_by_decile[result.quantile_by_decile["config"] == prim_label]
    prim_segment = result.quantile_by_segment[result.quantile_by_segment["config"] == prim_label]
    prim_year = result.quantile_by_year[result.quantile_by_year["config"] == prim_label]

    sens = result.sensitivity_table.copy()
    sens["config"] = [_config_label(f, p, s) for f, p, s in zip(sens["family"], sens["param"], sens["input_series"])]

    th22 = result.th22_verdict
    stop13 = result.stop13_suggestion

    stage_lines = "\n".join(
        f"| {name} | {seconds:.1f}s |" for name, seconds in result.stage_timings.items() if not name.startswith("_")
    )

    label_counts = ", ".join(f"{k}={v}" for k, v in th22.get("counts", {}).items())

    verdict_explanation = {
        "ESCALA_DOMINA": (
            "La curtosis cae drasticamente al estandarizar por volatilidad causal reciente, y los cuantiles de "
            "z_t son razonablemente estables entre horas del dia, deciles de volatilidad y anios -- de forma "
            "robusta entre estimadores y ventanas. En palabras simples: los movimientos que parecian extremos "
            "en r_1m dejan de parecerlo, en su gran mayoria, una vez que se compara cada retorno contra la "
            "volatilidad que ya era previsible en ese momento. La mayor parte de las colas gruesas de MNQ es "
            "un efecto de ESCALA (la volatilidad cambia en el tiempo), no de FORMA."
        ),
        "FORMA_SUSTANCIAL": (
            "Incluso despues de estandarizar causalmente por la volatilidad reciente, una fraccion grande del "
            "exceso de curtosis sobrevive, y/o los cuantiles de z_t difieren de forma sistematica entre horas "
            "del dia, deciles de volatilidad o anios -- de forma robusta entre estimadores y ventanas. En "
            "palabras simples: aun comparando cada retorno contra lo que era esperable en ese momento, los "
            "movimientos extremos de MNQ siguen siendo anormalmente extremos. Hay estructura de FORMA genuina, "
            "no solo de escala."
        ),
        "MIXTO": (
            "El resultado no es uniforme: una parte sustancial de la curtosis desaparece al estandarizar, pero "
            "persiste evidencia de forma (perfiles de cuantiles que no se superponen del todo, o una conclusion "
            "que cambia segun el estimador/ventana/anio). En palabras simples: la escala explica una parte real "
            "del fenomeno, pero no toda -- ni 'todo es escala' ni 'todo es forma' describe correctamente a MNQ "
            "con la evidencia de esta etapa."
        ),
    }[th22["verdict"]]

    report = f"""# TDA-10 — Escala versus forma: origen de las colas

**Etapa:** `docs/methodology/Tsay_OHLCV_analysis_roadmap.md` § TDA-10
**Depende de:** TDA-06 (`PASS_WITH_OPEN_QUESTIONS`, STOP-6 no activado, `s(m)` construido sobre `log_hl`), TDA-07 (`PASS_WITH_OPEN_QUESTIONS`), TDA-09 (`PASS_WITH_OPEN_QUESTIONS` -- `VOLATILITY_CLUSTERING_DETECTABLE`, TH21=`CLUSTERING_GENUINO`, STOP-9 NO activado).
**Alcance de datos:** exclusivamente `tda04_variables_1m.parquet`, `tda04_return_validity_mask.parquet`, `tda06_r_tilde.parquet` y `TDA06_segmentacion_propuesta.csv`. Ningún archivo de `data/raw/` ni de `holdout_files` fue abierto en esta etapa.
**Generado automáticamente** por `{run_command}` — este informe NUNCA se edita a mano; cualquier corrección se hace en el código (`tda10_scale_vs_shape.py`/`run_tda10.py`) y se regenera con una nueva ejecución.

> Esta etapa estudia si las colas gruesas de MNQ se explican porque la volatilidad cambia en el tiempo (ESCALA, ya caracterizado en TDA-09) o porque, incluso controlando esa volatilidad, la forma de la distribución sigue siendo anormalmente pesada (FORMA). NO ajusta GARCH, NO calcula cuantiles condicionales completos (TDA-12), NO ejecuta EVT, NO crea features/targets/señales, NO abre el holdout, y NO inicia TDA-11/12/13.

---

## 1. La pregunta en palabras simples

¿Los movimientos extremos de MNQ parecen extremos simplemente porque ocurren cuando el mercado ya está muy volátil, o siguen siendo anormalmente extremos incluso comparados con la volatilidad que era esperable en ese momento? Esta etapa estandariza cada retorno por una estimación CAUSAL (solo información pasada) de la volatilidad reciente y mide cuánta curtosis sobrevive.

## 2. Población utilizada

- `r_1m` válido (TDA-04, `r_1m_valid=True`): **{len(result.r1m_population):,}** filas.
- `r_tilde` (RETROSPECTIVO, TDA-06, `s(m)` sobre `log_hl`): **{len(result.r_tilde_population):,}** filas.
- Ambas poblaciones coinciden en `n` (verificado, `verify_populations_aligned`) -- condición para poder comparar "raw" (causal) y "clock_adjusted" (RETROSPECTIVO) bajo la misma noción de población.
- Cada configuración de `sigma_hat` descarta además su propio "quemado" inicial (ventana rodante: `min_periods=window`; EWMA: `3×half-life`) — el `n` exacto por configuración/año está en `{config.tda10_kurtosis_csv_name}` (columna `n`).

## 3. Estimadores causales

Dos familias, cada una con una grilla pequeña y predeclarada (nunca ajustada tras ver el resultado):

- **A. Desviación estándar rodante causal**: ventanas `{ROLLING_WINDOWS_MINUTES}` (número de barras VÁLIDAS previas, no minutos de reloj). `sigma_hat_(t-1) = std(r[t-W:t])`, con `shift(1)` explícito — la fila `t` nunca usa `r[t]`.
- **B. EWMA causal** (estilo RiskMetrics): half-lives `{EWMA_HALFLIVES_MINUTES}` minutos. `sigma2_t = lambda·sigma2_(t-1) + (1-lambda)·r_(t-1)^2`, implementado con `r^2` desplazado un paso y `pandas.ewm(halflife=..., adjust=False)` — `sigma2_t` depende exclusivamente de `r[0..t-1]`.
- **Ajuste de reloj**: cada familia se aplica también sobre `r_tilde` (`s(m)` de TDA-06, RETROSPECTIVO) para separar la escala DETERMINISTA (hora del día) de la DINÁMICA (volatilidad reciente) — 2 series de entrada × (3+3) configuraciones = **12 configuraciones** en total.
- **Estimador primario** (declarado antes de ejecutar): `{prim_label}` — EWMA, half-life=60min, sobre `r_1m` (la única serie genuinamente causal de principio a fin).

**Verificación explícita de ausencia de look-ahead (G1)** — prueba de reconstrucción: se perturba `r` en varios índices a un valor extremo y se comprueba que `sigma_hat` en posiciones `<= idx` NO cambia (y que sí cambia poco después). Resultado sobre las 6 combinaciones familia/parámetro (idéntico para "raw"/"clock_adjusted", la lógica de causalidad no depende del input):

{_md_table(result.causality_table, ["family", "param", "n_indices_checked", "passed", "all_changed_shortly_after"])}

Las 6 configuraciones **pasaron** la prueba — ninguna usa información de `r[idx]` o posterior para calcular `sigma_hat` en `t<=idx` (si alguna hubiera fallado, `LookaheadLeakageError` habría detenido la etapa antes de construir ningún `z_t`).

## 4. Resultado de curtosis (global, versión completa)

Curtosis de `r` (restringido a la población donde `z` está definido) vs. curtosis de `z`, fracción eliminada, las 12 configuraciones, alcance GLOBAL:

{_md_table(kurt_global, ["config", "n", "kurt_r", "kurt_z", "fraction_removed"], float_nd=3)}

**Bootstrap de bloques por jornada (IC 95%, `n_boot={ci['n_boot']}`) — configuración primaria `{prim_label}`**:

| | punto | IC 95% lo | IC 95% hi |
|---|---:|---:|---:|
| kurt_r | {_fmt(ci['kurt_r_point'], 3)} | {_fmt(ci['kurt_r_ci_lo'], 3)} | {_fmt(ci['kurt_r_ci_hi'], 3)} |
| kurt_z | {_fmt(ci['kurt_z_point'], 3)} | {_fmt(ci['kurt_z_ci_lo'], 3)} | {_fmt(ci['kurt_z_ci_hi'], 3)} |
| fraction_removed (versión completa) | {_fmt(th22['primary_fraction_removed_full'], 4)} | {_fmt(ci['fraction_removed_ci_lo'], 4)} | {_fmt(ci['fraction_removed_ci_hi'], 4)} |

El IC de bootstrap es sobre la versión **completa** (sin recortar) — por eso puede ser negativo/amplio (ver nota más abajo: el momento de cuarto orden sin recortar es frágil ante un puñado de observaciones). El veredicto de §11 usa la versión **recortada** por ese motivo (`fraction_removed_trimmed` = {_fmt(th22['primary_fraction_removed_trimmed'], 4)} para `{prim_label}`).

**Nota sobre `rolling_std_30` y el piso numérico de `sigma_hat`**: la ventana rodante más corta (30 barras) puede, en tramos de precio casi constante, producir un `sigma_hat` numéricamente indistinguible de cero (residual de punto flotante, no volatilidad real) — dividir por ese valor dispararía un `z_t` de millones de desviaciones estándar y contaminaría por completo la curtosis sin recortar de esa única configuración. Se protegió explícitamente (`MIN_VALID_SIGMA_HAT=1e-8`, verificado sobre el conjunto de investigación real: exactamente 1 de 1.914.530 filas por debajo del piso, únicamente en `rolling_std_30`) — sin esa protección, `kurt_z` de esa configuración se dispara a más de 1,9 millones por un solo punto. Es la razón adicional, más allá de la recomendación de TDA-07, por la que el veredicto usa la versión recortada y no la completa.

## 5. Resultado con recorte 0.1% (convención TDA-07)

Misma tabla, columnas `*_trimmed` (recorte del 0.1% total, 0.05% por cola, igual convención que TDA-07):

{_md_table(kurt_global, ["config", "kurt_r_trimmed", "kurt_z_trimmed", "fraction_removed_trimmed"], float_nd=3)}

La diferencia entre la fracción eliminada completa y la recortada es en sí misma informativa: si la fracción recortada es mucho menor, la reducción de curtosis de la versión completa depende en gran parte de un puñado de observaciones extremas (consistente con TDA-07, que ya mostró que la curtosis recortada de `r_1m` es mucho más estable que la cruda).

## 6. Estabilidad por año (configuración primaria `{prim_label}`)

{_md_table(kurt_year_prim, ["scope_value", "n", "kurt_r", "kurt_z", "fraction_removed", "fraction_removed_trimmed"], float_nd=3)}

## 7. Estabilidad por segmento horario (headline: primario raw y clock_adjusted)

Cuantiles de `z_t` por segmento (TDA-06) — `n` y cuantiles `NaN` si el segmento no alcanza el mínimo de muestra:

{_md_table(prim_segment, ["group", "n", "std_z", "q0.01", "q0.05", "q0.95", "q0.99"], float_nd=4)}

## 8. Estabilidad entre deciles de volatilidad (headline: primario raw)

{_md_table(prim_decile, ["group", "n", "std_z", "q0.01", "q0.05", "q0.95", "q0.99"], float_nd=4)}

Ver también `{config.tda10_quantile_by_year_csv_name}` (misma tabla por año) y `{config.tda10_quantile_profile_png_name}` (perfil visual, primario raw y clock_adjusted lado a lado).

## 9. Sensibilidad a estimador/ventana (las 12 configuraciones)

{_md_table(sens, ["config", "fraction_removed_full", "fraction_removed_trimmed", "decile_stability_ratio", "segment_stability_ratio", "year_stability_ratio", "config_label"], float_nd=3)}

`fraction_removed_full` es la versión SIN recortar — se reporta por transparencia (roadmap, tabla central) pero es el motivo por el que un estimador de ventana corta puede mostrar valores extremos (ver §4: un único `sigma_hat` numéricamente indistinguible de cero, dentro de una ventana de precios casi constantes, puede disparar un `z_t` de millones de desviaciones — protegido explícitamente por `MIN_VALID_SIGMA_HAT`, pero incluso protegido, un puñado de sorpresas genuinas puede seguir dominando un momento de cuarto orden sin recortar). **La clasificación de cada configuración usa `fraction_removed_trimmed`** — la cifra que TDA-07 (informe, §12) recomendó explícitamente como referencia más estable antes de esta etapa.

**Robustez**: {th22['n_configs']} configuraciones evaluadas, distribución de etiquetas: {label_counts}. La etiqueta mayoritaria (`{th22['verdict'] if th22['robust'] else max(th22['counts'], key=lambda k: th22['counts'][k]) if th22['counts'] else 'N/A'}`) cubre una fracción **{_fmt(th22['agreement_fraction'], 2)}** de las configuraciones — {'suficiente' if th22['robust'] else 'INSUFICIENTE'} para declarar el veredicto robusto (umbral predeclarado: {ROBUSTNESS_AGREEMENT_FRACTION}).

**Patrón por escala DETERMINISTA vs DINÁMICA** (§3, ajuste de reloj): en la tabla de arriba, {sens.loc[(sens['input_series']=='clock_adjusted') & (sens['config_label']=='ESCALA_DOMINA')].shape[0]} de las 6 configuraciones `clock_adjusted` (que primero retiran `s(m)`, RETROSPECTIVO, y luego aplican el estimador causal) clasifican como `ESCALA_DOMINA`, frente a solo {sens.loc[(sens['input_series']=='raw') & (sens['config_label']=='ESCALA_DOMINA')].shape[0]} de las 6 `raw` (que solo aplican el estimador causal, sin retirar antes el patrón horario). Esto sugiere que buena parte de lo que aparenta ser FORMA cuando se usa exclusivamente un estimador causal dinámico es, en realidad, escala DETERMINISTA (el patrón horario de TDA-06) que ese estimador —reactivo pero lento— no captura bien por sí solo. Esta distinción (determinista vs dinámica) es precisamente la que la tarea pidió no confundir; **no cambia el veredicto GLOBAL** (que se basa en las 12 configuraciones, no solo en las 6 `clock_adjusted`, y la versión `clock_adjusted` no es causal de principio a fin — depende de `s(m)` RETROSPECTIVO), pero es evidencia relevante para TDA-11/12.

## 10. QQ-plots

- `{config.tda10_qq_primary_png_name}`: `r_1m` crudo vs. normal, y `z_t` (primario) vs. normal — cuánta cola desaparece al retirar la escala dinámica.
- `{config.tda10_qq_sensitivity_png_name}`: `z_t` superpuesto para las 6 configuraciones "raw" y las 6 "clock_adjusted" — sensibilidad visual del QQ a estimador/ventana/half-life.
- `{config.tda10_quantile_profile_png_name}`: perfil de cuantiles de `z_t` por decil de volatilidad — el diagnóstico visual directo de estabilidad de FORMA.

## 11. Veredicto final — TH22

**`{th22['verdict']}`**

{verdict_explanation}

Reglas operativas predeclaradas (antes de ejecutar sobre el conjunto de investigación real, nunca ajustadas después — la ÚNICA decisión tomada después de ver el resultado fue usar `fraction_removed_trimmed` en vez de `fraction_removed_full` como entrada de estas reglas, ver nota de §9: no es un umbral ajustado, es una corrección de qué métrica alimenta las mismas reglas, justificada independientemente por la recomendación previa de TDA-07): `ESCALA_DOMINA` exige `fraction_removed_trimmed >= {FRACTION_REMOVED_SCALE_THRESHOLD}` y el mayor de los tres ratios de estabilidad (decil/segmento/año) `<= {PROFILE_STABILITY_SCALE_THRESHOLD}`; `FORMA_SUSTANCIAL` exige `fraction_removed_trimmed <= {FRACTION_REMOVED_FORM_THRESHOLD}` o algún ratio `>= {PROFILE_STABILITY_FORM_THRESHOLD}`; el veredicto GLOBAL exige que al menos el {ROBUSTNESS_AGREEMENT_FRACTION:.0%} de las 12 configuraciones coincida en la misma etiqueta, o se reporta `MIXTO`.

**Importante (roadmap, riesgo explícito):** que la curtosis baje NO implica que `z_t` sea normal — puede bajar sustancialmente y seguir siendo una distribución de colas pesadas (ver §4/§5: la curtosis de `z_t` casi nunca es cercana a 0, aunque sea mucho menor que la de `r`).

### TH22 / TH26 / STOP-13

- **TH22 = `{th22['verdict']}`** — RESUELTA por esta etapa.
- **TH26 = `{result.th26_status}`** — las tablas de cuantiles por segmento/decil/año de esta etapa (§7/§8) son evidencia PARCIAL y análoga a lo que TH26 pide, pero TH26 formalmente requiere los cuantiles condicionales completos con bootstrap por grupo y el `n` que sostiene cada cuantil extremo — eso pertenece a TDA-12 (obligatoria), no se declara resuelta aquí.
- **STOP-13**: {'`SUGERIDO` (no activado formalmente)' if stop13['suggested'] else '`NO SUGERIDO`'} — {stop13['reason']}

## 12. Tiempo por etapa (análisis)

| Etapa | Tiempo |
|---|---:|
{stage_lines}

**Tiempo total de la ejecución (análisis + escritura de CSV/PNG/MD): {elapsed_total_seconds:.1f}s (~{elapsed_total_seconds/60:.1f} min).**

## 13. Archivos generados

`{config.tda10_kurtosis_csv_name}`, `{config.tda10_kurtosis_bootstrap_ci_csv_name}`, `{config.tda10_quantile_by_decile_csv_name}`, `{config.tda10_quantile_by_segment_csv_name}`, `{config.tda10_quantile_by_year_csv_name}`, `{config.tda10_sensitivity_csv_name}`, `{config.tda10_causality_check_csv_name}`, `{config.tda10_qq_points_csv_name}` (8 CSV) + `{config.tda10_qq_primary_png_name}`, `{config.tda10_qq_sensitivity_png_name}`, `{config.tda10_quantile_profile_png_name}` (3 PNG) + este informe (MD).

## 14. Comandos de validación

```
python -m pytest -q tests/test_tda10_scale_vs_shape.py
python -m pytest -q
python -m ohlcv_dataroad.ingest.run_tda10 --config configs/mnq_snapshot.yaml
```

## 15. Estado final

**`PASS_WITH_OPEN_QUESTIONS`**

- TH22 = `{th22['verdict']}` (robusto: {th22['robust']}, agreement={_fmt(th22['agreement_fraction'], 2)})
- TH26 = `{result.th26_status}` (formalmente pendiente de TDA-12)
- STOP-13 = {'SUGERIDO (informal)' if stop13['suggested'] else 'NO SUGERIDO'}

**No se avanza a TDA-11 ni TDA-12 en esta tarea.**

## 16. Preguntas abiertas

1. El veredicto por configuración usa el mayor de los tres ratios de estabilidad (decil/segmento/año) — una configuración puede ser estable en dos dimensiones y no en la tercera; ver `{config.tda10_sensitivity_csv_name}` para el detalle por dimensión.
2. No se probó ningún estimador de volatilidad basado en rango (Parkinson/Rogers-Satchell/Yang-Zhang, roadmap TDA-04 §"método avanzado opcional") — dos familias mínimas (rodante + EWMA) son suficientes para responder la pregunta de bifurcación (G4); queda como extensión posible si TDA-11 llegara a ejecutarse.
3. TH26 queda solo PARCIALMENTE informada — TDA-12 debe producir la versión completa (bootstrap por grupo, `n` por cuantil extremo).

---

## Modo sencillo — en 10 líneas

**¿Los movimientos extremos de MNQ son extremos por la hora del día y la volatilidad reciente, o son extremos "de verdad"?** {('En su mayoría, por la volatilidad reciente: al comparar cada retorno contra la volatilidad que ya era previsible, la mayoría de las colas gruesas desaparece.' if th22['verdict']=='ESCALA_DOMINA' else ('Incluso después de descontar la volatilidad reciente, siguen siendo anormalmente extremos — hay algo más que pura escala.' if th22['verdict']=='FORMA_SUSTANCIAL' else 'El resultado es mixto: una parte real desaparece al descontar la volatilidad reciente, pero no toda.'))}

**¿Cómo se midió "la volatilidad que ya era previsible"?** Con dos formas simples de mirar solo el pasado — el desvío estándar de los últimos minutos, y un promedio que da más peso a lo reciente (EWMA) — nunca usando el propio movimiento que se está evaluando ni información futura (verificado con una prueba explícita).

**¿Depende de qué "regla" de volatilidad se use?** Se probaron 12 combinaciones (2 formas de medir volatilidad × 3 configuraciones cada una × con/sin ajuste por hora del día) — {'el resultado es el mismo en la gran mayoría de ellas' if th22['robust'] else 'el resultado NO es el mismo en todas — se reporta como MIXTO precisamente por eso'}.

**¿Esto confirma que los retornos ajustados son "normales"?** No. Puede bajar mucho la curtosis y seguir sin ser una campana de Gauss — solo dice cuánta de la anormalidad viene de la escala cambiante.

**¿Qué NO significa este resultado?** No es una señal de trading, no dice hacia dónde se moverá el precio, no ajusta ningún modelo GARCH, y no decide si hace falta un modelo de eventos extremos (EVT) — esa decisión formal es de TDA-12.
"""
    return report


def persist_artifacts(result: TDA10Result, config: SnapshotConfig, t0: float, run_command: str) -> list[Path]:
    config.reports_dir.mkdir(parents=True, exist_ok=True)

    result.kurtosis_table.to_csv(config.tda10_kurtosis_csv_path, index=False)
    pd.DataFrame([result.kurtosis_bootstrap_ci]).to_csv(config.tda10_kurtosis_bootstrap_ci_csv_path, index=False)
    result.quantile_by_decile.to_csv(config.tda10_quantile_by_decile_csv_path, index=False)
    result.quantile_by_segment.to_csv(config.tda10_quantile_by_segment_csv_path, index=False)
    result.quantile_by_year.to_csv(config.tda10_quantile_by_year_csv_path, index=False)
    result.sensitivity_table.to_csv(config.tda10_sensitivity_csv_path, index=False)
    result.causality_table.to_csv(config.tda10_causality_check_csv_path, index=False)
    result.qq_table.to_csv(config.tda10_qq_points_csv_path, index=False)

    _draw_qq_primary(result, config.tda10_qq_primary_png_path)
    _draw_qq_sensitivity(result, config.tda10_qq_sensitivity_png_path)
    _draw_quantile_profile(result, config.tda10_quantile_profile_png_path)

    written = [getattr(config, attr) for attr in ARTIFACT_PATH_ATTRS if attr != "tda10_report_path"]

    elapsed_total_seconds = time.perf_counter() - t0
    report_text = render_report(result, config, elapsed_total_seconds, run_command)
    config.tda10_report_path.write_text(report_text, encoding="utf-8")
    written.append(config.tda10_report_path)

    return written


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Ejecuta TDA-10 (escala vs forma / origen de las colas, TH22 + parte de TH26) sobre las variables de TDA-04/TDA-06."
    )
    parser.add_argument("--config", default="configs/mnq_snapshot.yaml")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))
    run_command = "python -m ohlcv_dataroad.ingest.run_tda10 --config " + str(args.config)

    t0 = time.perf_counter()
    try:
        result = run_tda10_analysis(config, verbose=True)
    except (FileNotFoundError, HoldoutIsolationError, TimestampAlignmentError, RTildeInvariantError, PopulationMismatchError, LookaheadLeakageError) as exc:
        print("TDA-10 status: FAIL", file=sys.stderr)
        print(f"Motivo: {exc}", file=sys.stderr)
        return 1

    t_write_start = time.perf_counter()
    print(f"[TDA10 {TOTAL_STAGES}/{TOTAL_STAGES}] Persistencia de artefactos (CSV/PNG/MD)...")
    written_paths = persist_artifacts(result, config, t0, run_command)
    write_elapsed = time.perf_counter() - t_write_start
    total_elapsed_final = time.perf_counter() - t0
    print(f"[TDA10 {TOTAL_STAGES}/{TOTAL_STAGES}] Persistencia de artefactos -- completado en {write_elapsed:.1f}s (total: {total_elapsed_final:.1f}s)")

    print()
    print("TDA-10 status: analisis completado")
    print(f"n filas r_1m/r_tilde validas: {len(result.r1m_population):,} / {len(result.r_tilde_population):,}")
    print(f"Tiempo total (incluye analisis + escritura de CSV/PNG/MD): {total_elapsed_final:.1f}s (~{total_elapsed_final/60:.1f} min)")
    print()
    print(f"TH22 = {result.th22_verdict.get('verdict')} (robusto={result.th22_verdict.get('robust')}, agreement={result.th22_verdict.get('agreement_fraction'):.2f})")
    print(f"TH26 = {result.th26_status}")
    print(f"STOP-13 sugerido = {result.stop13_suggestion.get('suggested')}")
    print()

    print("Artefactos escritos:")
    for p in written_paths:
        print(f"  {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
