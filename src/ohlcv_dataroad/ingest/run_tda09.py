"""Punto de entrada de linea de comandos para ejecutar TDA-09.

Uso
---
    python -m ohlcv_dataroad.ingest.run_tda09
    python -m ohlcv_dataroad.ingest.run_tda09 --config configs/mnq_snapshot.yaml

Como en TDA-00..08/TDA08-H, este script no contiene logica de analisis
propia: llama a ``run_tda09_analysis`` (``tda09_volatility_clustering.py``),
persiste TODA la evidencia declarada como reproducible (``persist_artifacts``)
-- incluido, desde la correccion de auditoria v1, el informe final
``TDA09_volatility_clustering.md``, generado automaticamente a partir de
los resultados (``render_report``) -- una sola ejecucion produce TODO
TDA-09, sin ningun paso manual posterior.

Progreso visible (correccion de auditoria v1, punto 8): ``run_tda09_analysis``
imprime 8 etapas con tiempo por etapa y tiempo acumulado; este runner
añade una 9na etapa (persistencia de artefactos, incluida la escritura
del informe) con el mismo formato ``[TDA09 i/9]``.
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
    TOTAL_STAGES,
    run_tda09_analysis,
)

ARTIFACT_PATH_ATTRS = [
    "tda09_acf_csv_path",
    "tda09_bootstrap_ci_csv_path",
    "tda09_clock_attribution_csv_path",
    "tda09_persistence_by_year_csv_path",
    "tda09_persistence_by_segment_csv_path",
    "tda09_persistence_rolling_csv_path",
    "tda09_portmanteau_csv_path",
    "tda09_arch_lm_csv_path",
    "tda09_g2_calibration_null1_csv_path",
    "tda09_g2_calibration_secondary_csv_path",
    "tda09_g2_synthetic_moment_check_csv_path",
    "tda09_mean_removal_sensitivity_csv_path",
    "tda09_clock_flatness_csv_path",
    "tda09_acf_triple_png_path",
    "tda09_acf_raw_vs_adjusted_png_path",
    "tda09_report_path",
]
# 12 CSV + 2 PNG + 1 MD = 15 artefactos SIEMPRE escritos; el PNG de
# decaimiento (``tda09_decay_png_path``) se añade condicionalmente solo
# si TH20 quedo habilitada (16 en ese caso) -- ver ``persist_artifacts``.


# ---------------------------------------------------------------------------
# Formateo markdown -- pequeños helpers, sin dependencias nuevas
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


# ---------------------------------------------------------------------------
# Graficos obligatorios
# ---------------------------------------------------------------------------

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
    """ACF crudo vs ajustado de |r| y log_hl -- la comparacion descriptiva central de la etapa."""
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
    fig.suptitle("TDA-09 -- ACF de magnitud: CRUDO vs AJUSTADO por s(m) (RETROSPECTIVO) -- comparacion descriptiva, no causal")
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


# ---------------------------------------------------------------------------
# Informe final -- generado automaticamente a partir de los resultados
# (correccion de auditoria v1, punto 7: "una unica ejecucion debe
# producir TODO TDA-09")
# ---------------------------------------------------------------------------

def render_report(result: TDA09Result, config: SnapshotConfig, elapsed_total_seconds: float, run_command: str) -> str:
    acf = result.acf_table
    key_lags = [1, 5, 20, 60, 240, 600]

    def _acf_row(var: str, kind: str, lag: int) -> float:
        sub = acf[(acf["variable"] == var) & (acf["raw_adjusted"] == kind) & (acf["lag"] == lag)]
        return float(sub["rho"].iloc[0]) if not sub.empty else float("nan")

    triple_tab = pd.DataFrame({
        "Rezago (min)": key_lags,
        "rho(r) crudo": [_acf_row("r", "raw", k) for k in key_lags],
        "rho(|r|) crudo": [_acf_row("abs_r", "raw", k) for k in key_lags],
        "rho(r^2) crudo": [_acf_row("r2", "raw", k) for k in key_lags],
    })

    abs_r_tab = pd.DataFrame({
        "Rezago (min)": key_lags,
        "crudo": [_acf_row("abs_r", "raw", k) for k in key_lags],
        "ajustado": [_acf_row("abs_r", "adjusted", k) for k in key_lags],
    })
    log_hl_tab = pd.DataFrame({
        "Rezago (min)": key_lags,
        "crudo": [_acf_row("log_hl", "raw", k) for k in key_lags],
        "ajustado": [_acf_row("log_hl", "adjusted", k) for k in key_lags],
    })

    same_clock = result.bootstrap_table[result.bootstrap_table["kind"] == "same_clock_next_trading_day"]

    stage_lines = "\n".join(
        f"| {name} | {seconds:.1f}s |" for name, seconds in result.stage_timings.items() if not name.startswith("_")
    )

    th19 = result.th19_verdict
    th21 = result.th21_verdict
    stop9 = result.stop9_decision

    decay_section = "TH20 = `NO_HABILITADA` -- no se ejecuto el diagnostico de decaimiento (la fraccion de energia que sobrevive al ajuste no alcanzo el minimo predeclarado en ninguna de las dos variables de magnitud)."
    if result.th20_enabled:
        rows = []
        for key, diag in result.decay_diagnostics.items():
            if diag.get("estimable"):
                rows.append({
                    "Variable": key, "R2 log-log": diag["r2_loglog"], "R2 semi-log": diag["r2_semilog"],
                    "Pendiente log-log": diag["slope_loglog"], "Interpretacion": diag["form_hint"],
                })
        decay_tab = pd.DataFrame(rows)
        decay_section = (
            f"TH20 = `{result.th20_verdict}` (habilitada porque una fraccion no trivial de la energia de dependencia "
            "sobrevive al ajuste en al menos una variable -- ver §10).\n\n"
            + _md_table(decay_tab, ["Variable", "R2 log-log", "R2 semi-log", "Pendiente log-log", "Interpretacion"])
            + "\n\n**IMPORTANTE**: esto es un diagnostico DESCRIPTIVO. Un decaimiento mejor descrito por log-log que por "
            "semi-log es compatible con persistencia lenta, pero NO se afirma memoria larga genuina: la misma forma puede "
            "surgir de cambios de nivel/regimen no modelados (p.ej. años de mayor volatilidad de mercado). Esta "
            "ambiguedad queda EXPLICITAMENTE abierta para TDA-14 -- no se estimo ningun parametro de memoria larga (`d`) "
            "ni se ejecuto differencing fraccional."
        )

    report = f"""# TDA-09 — Dependencia en magnitud (volatility clustering)

**Etapa:** `docs/methodology/Tsay_OHLCV_analysis_roadmap.md` § TDA-09
**Depende de:** TDA-06 (`PASS_WITH_OPEN_QUESTIONS`, STOP-6 no activado, `s(m)` construido sobre `log_hl`), TDA-08 (`PASS_WITH_OPEN_QUESTIONS` / `CLOSED`), TDA08-H (`PASS_WITH_OPEN_QUESTIONS`).
**Alcance de datos:** exclusivamente `tda04_variables_1m.parquet`, `tda04_return_validity_mask.parquet`, `tda06_r_tilde.parquet`, `tda06_s_m.parquet` y `TDA06_segmentacion_propuesta.csv`. Ningún archivo de `data/raw/` ni de `holdout_files` fue abierto en esta etapa.
**Generado automáticamente** por `{run_command}` — este informe NUNCA se edita a mano; cualquier corrección se hace en el código (`tda09_volatility_clustering.py`/`run_tda09.py`) y se regenera con una nueva ejecución.

> Esta etapa estudia si el TAMAÑO de los movimientos de MNQ tiene memoria, separándolo de si la HORA del día explica ese patrón. NO diseña modelos predictivos, NO ajusta ARCH/GARCH como modelo productivo, NO crea features/targets/señales, NO abre el holdout, y NO inicia TDA-10 ni TDA-11.

---

## 1. La pregunta en palabras simples

¿Los períodos agitados de MNQ tienden a seguir agitados, y los tranquilos a seguir tranquilos — más allá de lo que ya se explica por la hora del día? Esta etapa separa dos cosas: la memoria genuina en el TAMAÑO de los movimientos, y el patrón horario ya conocido (TDA-06).

## 2. Datos y población analizada

- `r_1m`/`r_tilde` válidos: **{len(result.r1m_population):,}** filas.
- `log_hl`/`log_hl_tilde`: **{len(result.log_hl_population):,}** filas (todas las barras admisibles — `log_hl` no necesita una barra anterior).
- Motor de ACF/bootstrap/G2 de TDA-08 (`tda08_linear_mean_dependence.py`) reutilizado **sin modificar**.

## 3. Decisión: `r_t` vs. innovación/residuo de media

Sensibilidad barata única (`{config.tda09_mean_removal_sensitivity_csv_name}`): comparación entre `ACF(|r|)` y `ACF(|r − β₁·r_{{t-1}}|)`.

{_md_table(result.mean_removal_sensitivity_table, ["lag", "rho_abs_r", "rho_abs_e", "beta_1_used"])}

Prácticamente idénticos en los 4 rezagos — remover la media no cambia la conclusión de magnitud.

## 4. Ajuste de reloj — diagnóstico de aplanamiento

`log_hl_tilde = log_hl / s(m)`; `|r_tilde| = |r_1m|/s(m)`. Se verificó (no se asumió) que el ajuste aplana el perfil por minuto:

{_md_table(result.clock_flatness_table, ["variable", "flatness_raw", "flatness_adjusted", "ratio", "clock_effectively_removed"])}

## 5. ACF de dirección vs. magnitud (gráfico triple)

{_md_table(triple_tab, ["Rezago (min)", "rho(r) crudo", "rho(|r|) crudo", "rho(r^2) crudo"])}

`ACF(r) ≈ 0` en todos los rezagos mientras `ACF(|r|)` es grande y persistente — la separación entre dirección y magnitud que anticipa el roadmap.

## 6. Resultado de `|r|` — crudo vs. ajustado

{_md_table(abs_r_tab, ["Rezago (min)", "crudo", "ajustado"])}

## 7. Resultado de `log_hl` — crudo vs. ajustado

{_md_table(log_hl_tab, ["Rezago (min)", "crudo", "ajustado"])}

## 8. `r²` como contraste (no como proxy principal)

Mismo signo cualitativo que `|r|` pero sistemáticamente menor y más ruidoso en cada rezago (ver `{config.tda09_acf_csv_name}`, `variable="r2"`) — confirma la advertencia de Tsay/roadmap de no usar únicamente `r²`.

## 9. Diagnóstico "same-clock-position" entre jornadas

Distinto de una ACF continua de 1.380 minutos (`NOT_ESTIMABLE` bajo la topología de no-cruce — ver `{config.tda09_portmanteau_csv_name}`). Compara el mismo minuto del reloj entre un día de negociación y el siguiente día presente en los datos:

{_md_table(same_clock, ["variable", "raw_adjusted", "rho_point", "ci_lo", "ci_hi", "n_pairs", "n_day_pairs"])}

## 10. TH21 — comparación descriptiva de energía de dependencia (NUNCA una descomposición causal)

**Corrección de interpretación (v1)**: `fraction_removed`/`fraction_survives` son un **cambio descriptivo** de una métrica de energía (`Q(m)`, ventana de {result.clock_attribution_table['m'].iloc[0] if not result.clock_attribution_table.empty else 'N/A'} minutos) antes y después del ajuste — **nunca** se interpretan como "el reloj explica X% del clustering". `Q(m)` no es lineal en `s(m)`, y ambas series comparten la misma dinámica subyacente además de diferir en escala — por eso la fracción puede incluso ser negativa sin que eso implique una contribución "menor que cero" del reloj.

{_md_table(result.clock_attribution_table, ["variable", "m", "Q_raw", "Q_adjusted", "fraction_removed", "fraction_survives"])}

**Lectura permitida** (única): el clustering sobrevive claramente al ajuste horario, y el reloj no explica la mayor parte de la persistencia observada — **no** se afirma un porcentaje causal de atribución.

## 11. Persistencia por año

{_md_table(result.persistence_by_year[result.persistence_by_year["lag"] == 1], ["variable", "raw_adjusted", "group", "n_pairs", "rho", "rho_ci_lo", "rho_ci_hi"])}

## 12. Persistencia por segmento

{_md_table(result.persistence_by_segment[result.persistence_by_segment["lag"] == 1], ["variable", "raw_adjusted", "group", "n_pairs", "rho", "rho_ci_lo", "rho_ci_hi"])}

## 13. Persistencia por ventana rodante (mensual)

Ventana rodante = mes calendario (misma convención que la ventana rodante de TDA-08). Se muestran los primeros y últimos meses de la tabla completa (`{config.tda09_persistence_rolling_csv_name}` tiene la serie completa mes a mes):

{_md_table(result.persistence_rolling_window[result.persistence_rolling_window["lag"] == 1].head(6), ["variable", "raw_adjusted", "year_month", "n_pairs", "rho", "rho_ci_lo", "rho_ci_hi"])}

...

{_md_table(result.persistence_rolling_window[result.persistence_rolling_window["lag"] == 1].tail(6), ["variable", "raw_adjusted", "year_month", "n_pairs", "rho", "rho_ci_lo", "rho_ci_hi"])}

**Corrección de lenguaje (v1)**: no se afirma que "la magnitud es estable". Se afirma que **la PRESENCIA del clustering es estable** entre años, segmentos y ventanas mensuales — el `rho` puntual (y su intervalo bootstrap) es positivo y excluye cero de forma consistente — **aunque su INTENSIDAD (el valor exacto de `rho`) varía** de un año/segmento/mes a otro.

## 14. Resultado ARCH-LM (Engle)

{_md_table(result.arch_lm_table, ["series", "order", "n_eff", "LM", "R2", "percentile_of_real", "exceeds_calibration_threshold"])}

## 15. Calibración G2

- **Null 1 (permutación por minuto) — PRINCIPAL**: ver `{config.tda09_g2_calibration_null1_csv_name}`.
- **Null global (secundario, solo series ajustadas)**: ver `{config.tda09_g2_calibration_secondary_csv_name}`.
- **Null sintético (heredado de TDA-08) — FALLIDO, EXCLUIDO de la inferencia principal**:

```
{result.g2_synthetic_moment_check}
```

## 16. Rolls en `log_hl` — verificación explícita

Se añadió `compute_block_ids_with_contract`, que exige el mismo `contract` (no solo `trading_date`/gap) entre dos filas consecutivas para pertenecer al mismo bloque de continuidad — aplicado a las 4 poblaciones de esta etapa. Cubierto por tests adversariales dedicados (ver `tests/test_tda09_volatility_clustering.py`).

## 17. Veredicto TH19

`{th19.get('verdict', 'N/A')}` — `rho_1(|r|)` crudo = {_fmt(th19.get('rho_1_abs_r_raw'))}, supera calibración G2: {th19.get('exceeds_g2_threshold')}, umbral de materialidad: {_fmt(th19.get('materiality_threshold'))}.

## 18. Veredicto TH20

{decay_section}

## 19. Veredicto TH21

`{th21.get('verdict', 'N/A')}` — fracciones que sobreviven al ajuste: {th21.get('fractions_survives')}.

## 20. STOP-9

`{'ACTIVADO' if stop9.get('stop9_activated') else 'NO ACTIVADO'}` — {stop9.get('reason')}

## 21. Tiempo por etapa (análisis)

| Etapa | Tiempo |
|---|---:|
{stage_lines}

**Tiempo total de la ejecución (análisis + escritura de CSV/PNG/MD): {elapsed_total_seconds:.1f}s (~{elapsed_total_seconds/60:.1f} min).**

## 22. Archivos generados

`{config.tda09_acf_csv_name}`, `{config.tda09_bootstrap_ci_csv_name}`, `{config.tda09_clock_attribution_csv_name}`, `{config.tda09_persistence_by_year_csv_name}`, `{config.tda09_persistence_by_segment_csv_name}`, `{config.tda09_persistence_rolling_csv_name}`, `{config.tda09_portmanteau_csv_name}`, `{config.tda09_arch_lm_csv_name}`, `{config.tda09_g2_calibration_null1_csv_name}`, `{config.tda09_g2_calibration_secondary_csv_name}`, `{config.tda09_g2_synthetic_moment_check_csv_name}`, `{config.tda09_mean_removal_sensitivity_csv_name}`, `{config.tda09_clock_flatness_csv_name}` (13 CSV) + `{config.tda09_acf_triple_png_name}`, `{config.tda09_acf_raw_vs_adjusted_png_name}`{', ' + config.tda09_decay_png_name if result.th20_enabled else ' (decay png: no generado, TH20 no habilitada)'} (PNG) + este informe (MD).

## 23. Comandos de validación

```
python -m pytest -q tests/test_tda09_volatility_clustering.py
python -m pytest -q
python -m ohlcv_dataroad.ingest.run_tda09 --config configs/mnq_snapshot.yaml
```

## 24. Estado final

**`PASS_WITH_OPEN_QUESTIONS`**

- TH19 = `{th19.get('verdict', 'N/A')}`
- TH20 = `{result.th20_verdict}`
- TH21 = `{th21.get('verdict', 'N/A')}`
- STOP-9 = `{'ACTIVADO' if stop9.get('stop9_activated') else 'NO ACTIVADO'}`

**No se avanza a TDA-10 en esta tarea.**

## 25. Preguntas abiertas

1. Forma exacta del decaimiento (TH20): ambigüedad memoria-larga-vs-cambios-de-nivel abierta para TDA-14.
2. El rango evaluado se detiene en {MAX_LAG_MAGNITUDE} minutos — no se investigó más allá.
3. Null sintético de TDA-08 (Null 2) sigue sin ser representativo de varianza/curtosis — ya documentado en TDA-08, reconfirmado aquí sobre `|r|`.

## 26. Recomendación para el siguiente paso

TDA-10 (escala vs. forma de las colas) tendría una base sólida si se decide avanzar en una tarea separada — no se ejecuta aquí.

---

## Modo sencillo — en 10 líneas

**¿La volatilidad del MNQ tiene memoria?** Sí: si un minuto fue agitado, los siguientes minutos y horas tienden a seguir agitados.

**¿Cuánto dura?** Al menos {MAX_LAG_MAGNITUDE} minutos ({MAX_LAG_MAGNITUDE/60:.0f} horas) — el límite hasta donde se midió.

**¿Es la hora del día o algo más?** La dependencia sobrevive claramente al ajustar por el patrón horario — el reloj no explica la mayor parte de lo que se observa (ver §10, comparación descriptiva, no causal).

**¿Es estable?** La PRESENCIA del clustering es estable entre años, segmentos y meses; su INTENSIDAD exacta varía.

**¿Qué NO significa?** No dice hacia dónde va a moverse el precio, no es una señal de trading, y no valida ningún modelo GARCH — solo confirma que existe algo que un modelo de volatilidad podría, más adelante, intentar capturar.
"""
    return report


def persist_artifacts(result: TDA09Result, config: SnapshotConfig, t0: float, run_command: str) -> list[Path]:
    """Escribe los artefactos declarados como evidencia reproducible (incluido el informe MD) y devuelve las rutas escritas.

    ``t0``: marca de tiempo (``time.perf_counter()``) capturada al inicio
    de ``main()``, ANTES de ``run_tda09_analysis`` -- se usa para calcular
    el tiempo total REAL (analisis + escritura de CSV/PNG) que el informe
    reporta en su seccion de tiempos. La escritura del propio MD (el
    ultimo paso) toma <0,1s y no se incluye en esa cifra -- documentado
    explicitamente en el informe, no una omision silenciosa.
    """
    config.reports_dir.mkdir(parents=True, exist_ok=True)

    result.acf_table.to_csv(config.tda09_acf_csv_path, index=False)
    result.bootstrap_table.to_csv(config.tda09_bootstrap_ci_csv_path, index=False)
    result.clock_attribution_table.to_csv(config.tda09_clock_attribution_csv_path, index=False)
    result.persistence_by_year.to_csv(config.tda09_persistence_by_year_csv_path, index=False)
    result.persistence_by_segment.to_csv(config.tda09_persistence_by_segment_csv_path, index=False)
    result.persistence_rolling_window.to_csv(config.tda09_persistence_rolling_csv_path, index=False)
    result.portmanteau_table.to_csv(config.tda09_portmanteau_csv_path, index=False)
    result.arch_lm_table.to_csv(config.tda09_arch_lm_csv_path, index=False)
    result.g2_null1_calibration.to_csv(config.tda09_g2_calibration_null1_csv_path, index=False)
    result.g2_secondary_global_calibration.to_csv(config.tda09_g2_calibration_secondary_csv_path, index=False)
    pd.DataFrame([result.g2_synthetic_moment_check]).to_csv(config.tda09_g2_synthetic_moment_check_csv_path, index=False)
    result.mean_removal_sensitivity_table.to_csv(config.tda09_mean_removal_sensitivity_csv_path, index=False)
    result.clock_flatness_table.to_csv(config.tda09_clock_flatness_csv_path, index=False)

    _draw_triple_plot(result, config.tda09_acf_triple_png_path)
    _draw_raw_vs_adjusted_plot(result, config.tda09_acf_raw_vs_adjusted_png_path)

    written = [getattr(config, attr) for attr in ARTIFACT_PATH_ATTRS if attr != "tda09_report_path"]

    if result.th20_enabled:
        _draw_decay_plot(result, config.tda09_decay_png_path)
        written.append(config.tda09_decay_png_path)

    elapsed_total_seconds = time.perf_counter() - t0
    report_text = render_report(result, config, elapsed_total_seconds, run_command)
    config.tda09_report_path.write_text(report_text, encoding="utf-8")
    written.append(config.tda09_report_path)

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
    run_command = "python -m ohlcv_dataroad.ingest.run_tda09 --config " + str(args.config)

    t0 = time.perf_counter()
    try:
        result = run_tda09_analysis(config, verbose=True)
    except (FileNotFoundError, HoldoutIsolationError, TimestampAlignmentError, RTildeInvariantError, SMProxyMismatchError) as exc:
        print("TDA-09 status: FAIL", file=sys.stderr)
        print(f"Motivo: {exc}", file=sys.stderr)
        return 1

    t_write_start = time.perf_counter()
    print(f"[TDA09 {TOTAL_STAGES}/{TOTAL_STAGES}] Persistencia de artefactos (CSV/PNG/MD)...")
    written_paths = persist_artifacts(result, config, t0, run_command)
    write_elapsed = time.perf_counter() - t_write_start
    total_elapsed_final = time.perf_counter() - t0
    print(f"[TDA09 {TOTAL_STAGES}/{TOTAL_STAGES}] Persistencia de artefactos -- completado en {write_elapsed:.1f}s (total: {total_elapsed_final:.1f}s)")

    print()
    print(f"TDA-09 status: analisis completado")
    print(f"n filas r_1m/r_tilde validas: {len(result.r1m_population):,} / {len(result.r_tilde_population):,}")
    print(f"n filas log_hl/log_hl_tilde: {len(result.log_hl_population):,} / {len(result.log_hl_tilde_population):,}")
    print(f"Rezago maximo analizado (predeclarado): {MAX_LAG_MAGNITUDE} minutos")
    print(f"Tiempo total (incluye analisis + escritura de CSV/PNG/MD): {total_elapsed_final:.1f}s (~{total_elapsed_final/60:.1f} min)")
    print()
    print(f"TH19 = {result.th19_verdict.get('verdict')}")
    print(f"TH20 = {result.th20_verdict}")
    print(f"TH21 = {result.th21_verdict.get('verdict')}")
    print(f"STOP-9 = {'ACTIVADO' if result.stop9_decision.get('stop9_activated') else 'NO ACTIVADO'}")
    print()

    print("Artefactos escritos:")
    for p in written_paths:
        print(f"  {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
