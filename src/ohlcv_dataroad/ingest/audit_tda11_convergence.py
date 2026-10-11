"""Auditoria correctiva de TDA-11 -- evidencia reproducible de C2 (convergencia) y C1 (optimo sin restriccion).

Uso
---
    python -m ohlcv_dataroad.ingest.audit_tda11_convergence
    python -m ohlcv_dataroad.ingest.audit_tda11_convergence --config configs/mnq_snapshot.yaml

Este script NO forma parte del pipeline de TDA-11 (`run_tda11.py`) y NO
produce ninguna de sus conclusiones: conserva, de forma reproducible, las
dos comprobaciones de DIAGNOSTICO que justificaron la auditoria
correctiva (docstring de `tda11_parametric_volatility.py`, puntos 9 y
10), que hasta ahora solo existian como scripts temporales:

**Parte A -- C2: `convergence_flag=0` de `arch` no acredita convergencia.**
Para cada uno de los 3 ajustes globales de TDA-11 (primario
`clock_adjusted`/normal, secundario `raw`/normal, sensibilidad
`clock_adjusted`/Student-t) se repite CADA intento de la grilla
`GARCH_OPTIMIZER_EPS_GRID` con las MISMAS llamadas que
`_arch_multistart`, y se registra por intento: lo que declaro SLSQP
(`convergence_flag`, iteraciones, mensaje), si se movio del punto de
partida, su log-verosimilitud, y el gradiente proyectado KKT
(`kkt_check_garch`) en ese punto. Al lado, el optimo VERIFICADO que
devuelve `fit_garch11`. La tabla permite a un revisor externo comprobar
por si mismo que varios intentos "exitosos" no son puntos estacionarios.

**Parte B -- C1: la persistencia `1.0000` del primario es el tope de la
restriccion.** Se maximiza la MISMA verosimilitud gaussiana (misma
recursion, `garch_sigma2_path`, procedimiento primario `carry`) pero
permitiendo `alpha+beta` hasta `PERSISTENCE_PROBE_MAX` (>1), y se traza
el perfil de verosimilitud sobre una grilla fija de persistencias
(`PERSISTENCE_PROFILE_GRID`). Es una SONDA descriptiva: no sustituye la
estimacion de TDA-11 (que conserva la restriccion de `arch`), no elige
un modelo y no se interpreta como evidencia de IGARCH ni de memoria
infinita -- solo demuestra que la restriccion esta ACTIVA (el maximo
libre cae por encima de 1) y cuanto cuesta, en log-verosimilitud,
imponer una persistencia menor.

Alcance de datos: exactamente los mismos insumos que TDA-11
(`tda04_variables_1m.parquet`, `tda04_return_validity_mask.parquet`,
`tda06_r_tilde.parquet`, `tda06_s_m.parquet`,
`TDA06_segmentacion_propuesta.csv`), con las mismas protecciones de
hold-out. No abre `data/raw/` ni `config.holdout_files`. No modifica
ningun artefacto de TDA-00..TDA-11: escribe UNICAMENTE los tres
archivos `TDA11_auditoria_*` declarados abajo. Determinista (sin
aleatoriedad): dos ejecuciones producen las mismas tablas.

Todas las grillas y cotas son constantes fijas de este modulo; no hay
umbrales de decision (el unico umbral que se cita,
`GARCH_KKT_GRADIENT_TOL`, se importa sin modificar de TDA-11).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize

from ohlcv_dataroad.config import SnapshotConfig, load_config
from ohlcv_dataroad.ingest.holdout_guard import (
    validate_last_timestamps_before_boundary,
    validate_research_holdout_disjoint,
)
from ohlcv_dataroad.ingest.tda07_marginal_distribution import (
    build_r1m_population,
    build_r_tilde_population,
    load_r_tilde,
    load_segmentation_cutoffs,
    load_tda04_inputs,
    verify_r_tilde_invariants,
    verify_timestamp_alignment,
)
from ohlcv_dataroad.ingest.tda11_parametric_volatility import (
    GARCH_KKT_GRADIENT_TOL,
    GARCH_OPTIMIZER_EPS_GRID,
    GARCH_OPTIMIZER_MAXITER,
    GARCH_SCALE_FACTOR,
    fit_garch11,
    garch_sigma2_path,
    kkt_check_garch,
    verify_populations_row_aligned,
)

__all__ = [
    "ATTEMPTS_CSV_NAME",
    "PROFILE_CSV_NAME",
    "REPORT_NAME",
    "PERSISTENCE_PROBE_MAX",
    "PERSISTENCE_PROFILE_GRID",
    "free_persistence_mean_nll",
    "fit_free_persistence",
    "audit_arch_attempts",
    "audit_persistence_profile",
    "load_audit_series",
    "render_audit_report",
    "main",
]

ATTEMPTS_CSV_NAME = "TDA11_auditoria_c2_intentos_optimizador.csv"
PROFILE_CSV_NAME = "TDA11_auditoria_c1_perfil_persistencia.csv"
REPORT_NAME = "TDA11_auditoria_convergencia.md"

# Sonda C1: cota superior de la persistencia cuando se RELAJA la
# restriccion de `arch`. Solo necesita ser holgadamente mayor que el
# maximo libre para que este quede en el interior (se comprueba y se
# reporta `probe_bound_active`).
PERSISTENCE_PROBE_MAX = 1.02
PERSISTENCE_PROFILE_GRID: tuple[float, ...] = (
    0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999, 1.0, 1.0001, 1.0005, 1.001, 1.002, 1.005,
)
_BIG = 1e10


# ---------------------------------------------------------------------------
# Parte B -- verosimilitud con persistencia libre (misma recursion que TDA-11)
# ---------------------------------------------------------------------------

def free_persistence_mean_nll(log10_omega: float, alpha: float, persistence: float, x: np.ndarray) -> float:
    """Log-verosimilitud negativa MEDIA gaussiana de GARCH(1,1) con `beta = persistence - alpha`, SIN la cota `persistence<=1`.

    Para `persistence<=1` coincide exactamente con
    `tda11_parametric_volatility.garch_mean_neg_loglik` (misma recursion
    `garch_sigma2_path`, modo `carry`, mismo valor de pre-muestra); la
    unica diferencia es que aqui `persistence>1` no se rechaza.
    """
    beta = persistence - alpha
    if alpha < 0.0 or beta < 0.0:
        return _BIG
    s2 = garch_sigma2_path(x, 10.0 ** log10_omega, alpha, beta)
    if not np.all(s2 > 0.0) or not np.all(np.isfinite(s2)):
        return _BIG
    value = 0.5 * float(np.mean(np.log(2.0 * np.pi) + np.log(s2) + x / s2))
    return value if np.isfinite(value) else _BIG


def fit_free_persistence(x: np.ndarray, start: tuple[float, float, float], fixed_persistence: float | None = None) -> dict:
    """Maximiza la verosimilitud con persistencia fija (`fixed_persistence`) o libre hasta `PERSISTENCE_PROBE_MAX`, partiendo de `start=(log10_omega, alpha, persistence)`.

    Nelder-Mead (sin derivadas), repetido hasta que deja de mejorar --
    determinista. Devuelve `omega_scaled`, `alpha`, `beta`,
    `persistence`, `loglikelihood` (suma) y `mean_nll`.
    """
    n = x.size

    def objective(t: np.ndarray) -> float:
        p = fixed_persistence if fixed_persistence is not None else float(t[2])
        if p > PERSISTENCE_PROBE_MAX or p <= 0.0:
            return _BIG
        return free_persistence_mean_nll(float(t[0]), float(t[1]), p, x)

    theta = np.array(start[:2] if fixed_persistence is not None else start, dtype=float)
    if fixed_persistence is not None:
        # arranque admisible: alpha no puede exceder la persistencia fijada
        theta[1] = min(theta[1], 0.9 * fixed_persistence)
    best = objective(theta)
    for _ in range(4):
        res = minimize(objective, theta, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 4000, "maxfev": 4000})
        gain = best - float(res.fun)
        if res.fun < best:
            theta, best = np.asarray(res.x, dtype=float), float(res.fun)
        if gain < 1e-12:
            break
    persistence = fixed_persistence if fixed_persistence is not None else float(theta[2])
    return {
        "omega_scaled": 10.0 ** float(theta[0]), "alpha": float(theta[1]), "beta": persistence - float(theta[1]),
        "persistence": persistence, "loglikelihood": -best * n, "mean_nll": best,
    }


def audit_persistence_profile(r: np.ndarray, constrained_fit: dict, scale: float = GARCH_SCALE_FACTOR) -> pd.DataFrame:
    """Perfil de verosimilitud sobre `PERSISTENCE_PROFILE_GRID` + maximo con persistencia libre, relativo al optimo restringido `constrained_fit` (salida de `fit_garch11`)."""
    x = (np.asarray(r, dtype=float) * scale) ** 2
    start = (float(np.log10(constrained_fit["omega"] * scale ** 2)), float(constrained_fit["alpha"]), float(constrained_fit["persistence"]))
    llf_constrained = float(constrained_fit["loglikelihood"])
    rows = []
    for p in PERSISTENCE_PROFILE_GRID:
        f = fit_free_persistence(x, start, fixed_persistence=float(p))
        rows.append({"kind": "profile_fixed_persistence", **f})
    free = fit_free_persistence(x, start)
    rows.append({"kind": "free_persistence_optimum", **free})
    rows.append({
        "kind": "constrained_optimum_tda11", "omega_scaled": constrained_fit["omega"] * scale ** 2, "alpha": constrained_fit["alpha"],
        "beta": constrained_fit["beta"], "persistence": constrained_fit["persistence"], "loglikelihood": llf_constrained,
        "mean_nll": -llf_constrained / x.size,
    })
    out = pd.DataFrame(rows)
    out["llf_minus_constrained_optimum"] = out["loglikelihood"] - llf_constrained
    out["feasible_under_arch_constraint"] = out["persistence"] <= 1.0 + 1e-12
    out["probe_bound_active"] = out["persistence"] >= PERSISTENCE_PROBE_MAX - 1e-9
    out["n"] = x.size
    return out


# ---------------------------------------------------------------------------
# Parte A -- cada intento del optimizador de arch, uno por uno
# ---------------------------------------------------------------------------

def audit_arch_attempts(r: np.ndarray, dist: str, label: str, scale: float = GARCH_SCALE_FACTOR) -> tuple[pd.DataFrame, dict]:
    """Repite cada intento de `GARCH_OPTIMIZER_EPS_GRID` (mismas llamadas que `_arch_multistart`) y lo contrasta con el optimo verificado de `fit_garch11`.

    Devuelve `(tabla, fit)`: una fila por intento + una fila
    `verified_optimum_fit_garch11`. `kkt_max_abs_gradient` se calcula con
    `kkt_check_garch` (el MISMO criterio que usa TDA-11); es `NaN` con
    `kkt_note` explicativo si el punto esta fuera de la region admisible
    (p. ej. un intento que viola `alpha+beta<=1`).
    """
    y = np.asarray(r, dtype=float) * scale
    x = y ** 2
    am = arch_model(y, mean="Zero", vol="GARCH", p=1, o=0, q=1, dist=dist)
    start = np.asarray(am.volatility.starting_values(y), dtype=float)

    def _kkt(omega: float, alpha: float, beta: float, nu: float | None) -> tuple[float, str]:
        persistence = alpha + beta
        if alpha < 0.0 or beta < 0.0 or persistence > 1.0 or omega <= 0.0:
            return float("nan"), "punto fuera de la region admisible (viola alpha+beta<=1 o una cota)"
        theta = [np.log10(omega), alpha, persistence] + ([nu] if dist == "t" else [])
        g = kkt_check_garch(np.asarray(theta, dtype=float), x, None, False, dist)
        return (float(np.max(np.abs(g))), "") if np.isfinite(g).all() else (float("nan"), "gradiente no evaluable")

    fit = fit_garch11(r, dist=dist, scale=scale)
    rows = []
    for eps in GARCH_OPTIMIZER_EPS_GRID:
        options = {"maxiter": GARCH_OPTIMIZER_MAXITER}
        if eps is not None:
            options["eps"] = eps
        res = am.fit(disp="off", cov_type="robust", show_warning=False, options=options)
        pa = res.params
        omega, alpha, beta = float(pa["omega"]), float(pa["alpha[1]"]), float(pa["beta[1]"])
        nu = float(pa["nu"]) if dist == "t" else None
        kkt, note = _kkt(omega, alpha, beta, nu)
        opt = res.optimization_result
        rows.append({
            "fit": label, "dist": dist, "source": "arch_slsqp_attempt", "eps": "scipy_default" if eps is None else f"{eps:g}",
            "convergence_flag": int(res.convergence_flag), "slsqp_message": str(opt.message), "n_iterations": int(opt.nit),
            "n_function_evals": int(opt.nfev), "stayed_at_starting_values": bool(np.allclose(pa.to_numpy()[:3], start, rtol=1e-10, atol=0.0)),
            "omega_scaled": omega, "alpha": alpha, "beta": beta, "nu": nu if nu is not None else float("nan"),
            "persistence": alpha + beta, "loglikelihood": float(res.loglikelihood),
            "kkt_max_abs_gradient": kkt, "kkt_note": note,
        })
    rows.append({
        "fit": label, "dist": dist, "source": "verified_optimum_fit_garch11", "eps": "",
        "convergence_flag": np.nan, "slsqp_message": "", "n_iterations": np.nan, "n_function_evals": np.nan,
        "stayed_at_starting_values": False, "omega_scaled": fit["omega"] * scale ** 2, "alpha": fit["alpha"], "beta": fit["beta"],
        "nu": fit["nu"], "persistence": fit["persistence"], "loglikelihood": fit["loglikelihood"],
        "kkt_max_abs_gradient": fit["kkt_max_abs_gradient"], "kkt_note": "",
    })
    out = pd.DataFrame(rows)
    out["llf_minus_verified_optimum"] = out["loglikelihood"] - float(fit["loglikelihood"])
    out["passes_kkt_tolerance"] = out["kkt_max_abs_gradient"] < GARCH_KKT_GRADIENT_TOL
    out["kkt_tolerance"] = GARCH_KKT_GRADIENT_TOL
    out["n"] = x.size
    return out, fit


# ---------------------------------------------------------------------------
# Carga de series -- mismos insumos y protecciones que run_tda11_analysis
# ---------------------------------------------------------------------------

def load_audit_series(config: SnapshotConfig) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve `(r_tilde, r_1m)` de la poblacion de TDA-11 (misma construccion, mismo orden), tras las validaciones de hold-out."""
    validate_research_holdout_disjoint(config)
    variables, validity = load_tda04_inputs(config)
    validate_last_timestamps_before_boundary(config, variables.groupby("source_file")["timestamp"].max().to_dict())
    verify_timestamp_alignment(variables, validity)
    cutoffs = load_segmentation_cutoffs(config)
    r1m_pop, _ = build_r1m_population(variables, validity, cutoffs)
    r1m_pop = r1m_pop.sort_values("timestamp").reset_index(drop=True)
    r_tilde = load_r_tilde(config)
    verify_r_tilde_invariants(r_tilde, variables)
    r_tilde_pop, _ = build_r_tilde_population(r_tilde, cutoffs)
    r_tilde_pop = r_tilde_pop.sort_values("timestamp").reset_index(drop=True)
    verify_populations_row_aligned(r1m_pop, r_tilde_pop)
    return r_tilde_pop["r_tilde"].to_numpy(dtype=float), r1m_pop["r_1m"].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------

def _md(df: pd.DataFrame, columns: list[str], formats: dict[str, str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---:"] * len(columns)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in columns:
            v = row[c]
            if isinstance(v, (float, np.floating)) and np.isnan(v):
                cells.append("—")
            elif c in formats and not isinstance(v, (str, bool, np.bool_)):
                cells.append(format(float(v), formats[c]))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_audit_report(attempts: pd.DataFrame, profile: pd.DataFrame, elapsed_seconds: float, run_command: str) -> str:
    att = attempts.loc[attempts["source"] == "arch_slsqp_attempt"]
    n_att = len(att)
    n_flag0 = int((att["convergence_flag"] == 0).sum())
    flag0 = att.loc[att["convergence_flag"] == 0]
    n_flag0_fail = int((~flag0["passes_kkt_tolerance"].astype(bool)).sum())
    n_stayed = int(att["stayed_at_starting_values"].sum())
    per_fit_lines = []
    for label, sub in att.groupby("fit", sort=False):
        ok = sub.loc[(sub["convergence_flag"] == 0)]
        best_declared = ok.loc[ok["loglikelihood"].idxmax()] if len(ok) else None
        if best_declared is not None:
            per_fit_lines.append(
                f"- **{label}**: el mejor intento que `arch` declaró exitoso queda a "
                f"{best_declared['llf_minus_verified_optimum']:,.2f} unidades de log-verosimilitud del óptimo verificado "
                f"(gradiente proyectado {best_declared['kkt_max_abs_gradient']:.2e}; "
                f"{'supera' if bool(best_declared['passes_kkt_tolerance']) else 'NO supera'} la verificación)."
            )
    free = profile.loc[profile["kind"] == "free_persistence_optimum"].iloc[0]
    con = profile.loc[profile["kind"] == "constrained_optimum_tda11"].iloc[0]
    prof = profile.loc[profile["kind"] == "profile_fixed_persistence"]
    return f"""# TDA-11 — Auditoría correctiva: evidencia de convergencia (C2) y de restricción activa (C1)

**Generado automáticamente** por `{run_command}` — nunca se edita a mano. No forma parte del pipeline de TDA-11 ni de sus conclusiones: conserva de forma reproducible el diagnóstico que justificó la auditoría correctiva (ver docstring de `audit_tda11_convergence.py`).
**Alcance de datos:** los mismos insumos derivados que TDA-11; ningún archivo de `data/raw/` ni de `holdout_files` fue abierto.

## A. C2 — ¿acredita convergencia el `convergence_flag=0` de `arch`?

Cada fila `arch_slsqp_attempt` es un intento del optimizador interno de `arch` (SLSQP) con un paso de diferencias finitas `eps` de la grilla `{GARCH_OPTIMIZER_EPS_GRID}`; la fila `verified_optimum_fit_garch11` es el resultado que publica TDA-11. `llf_minus_verified_optimum` = cuánta log-verosimilitud le falta al intento. `kkt_max_abs_gradient` = gradiente proyectado de la log-verosimilitud media (condición de primer orden con restricciones, `kkt_check_garch`); un punto se acepta solo si es menor que `GARCH_KKT_GRADIENT_TOL={GARCH_KKT_GRADIENT_TOL}`.

{_md(attempts, ["fit", "source", "eps", "convergence_flag", "n_iterations", "stayed_at_starting_values", "alpha", "beta", "persistence", "loglikelihood", "llf_minus_verified_optimum", "kkt_max_abs_gradient", "passes_kkt_tolerance", "kkt_note"], {"alpha": ".5f", "beta": ".5f", "persistence": ".6f", "loglikelihood": ",.2f", "llf_minus_verified_optimum": ",.2f", "kkt_max_abs_gradient": ".2e", "convergence_flag": ".0f", "n_iterations": ".0f"})}

**Hechos.** De {n_att} intentos, {n_flag0} terminaron con `convergence_flag=0` ("éxito"); de esos, **{n_flag0_fail} NO son puntos estacionarios** (no superan la verificación), y {n_stayed} ni siquiera se movieron del punto de partida.

{chr(10).join(per_fit_lines)}

## B. C1 — ¿es `alpha+beta = 1` una estimación o el tope de la restricción?

Misma verosimilitud gaussiana y misma recursion que el ajuste primario de TDA-11 (`clock_adjusted`), permitiendo `alpha+beta` hasta {PERSISTENCE_PROBE_MAX}. `llf_minus_constrained_optimum` = log-verosimilitud menos la del óptimo restringido que publica TDA-11.

{_md(profile, ["kind", "persistence", "alpha", "beta", "loglikelihood", "llf_minus_constrained_optimum", "feasible_under_arch_constraint"], {"persistence": ".6f", "alpha": ".5f", "beta": ".5f", "loglikelihood": ",.2f", "llf_minus_constrained_optimum": ",.2f"})}

**Hechos.** El máximo con persistencia libre está en `alpha+beta = {free['persistence']:.6f}` (`alpha` = {free['alpha']:.5f}), {'en el interior de la sonda' if not bool(free['probe_bound_active']) else 'EN LA COTA de la sonda (ampliar PERSISTENCE_PROBE_MAX)'}, con {free['llf_minus_constrained_optimum']:,.2f} unidades de log-verosimilitud más que el óptimo restringido (`alpha+beta = {con['persistence']:.6f}`). En el perfil, la verosimilitud {'sigue creciendo al pasar de 1' if float(prof.loc[prof['persistence'] > 1.0, 'loglikelihood'].max()) > float(prof.loc[np.isclose(prof['persistence'], 1.0), 'loglikelihood'].max()) else 'NO crece al pasar de 1'}: la restricción `alpha+beta ≤ 1` de `arch` está **activa**, y el `1.0000` de TDA-11 es ese tope.

**Qué NO significa.** No es una estimación de que la persistencia "verdadera" sea mayor que 1, ni evidencia de IGARCH o de memoria infinita: un máximo libre por encima de 1 en un GARCH(1,1) gaussiano es un síntoma conocido de especificación insuficiente (cambios de nivel de volatilidad, colas pesadas), compatible con las tres explicaciones que TDA-11 deja abiertas. Esta sonda no sustituye la estimación de TDA-11 ni interviene en TH23/TH24/TH25/STOP-11.

---
Tiempo de ejecución: {elapsed_seconds:.1f}s. Archivos: `{ATTEMPTS_CSV_NAME}`, `{PROFILE_CSV_NAME}`, este informe.
"""


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Auditoria correctiva de TDA-11: evidencia reproducible de C2 (convergencia) y C1 (restriccion activa).")
    parser.add_argument("--config", default="configs/mnq_snapshot.yaml")
    args = parser.parse_args(argv)
    config = load_config(Path(args.config))
    run_command = "python -m ohlcv_dataroad.ingest.audit_tda11_convergence --config " + str(args.config)

    t0 = time.perf_counter()
    r_adj, r_raw = load_audit_series(config)
    print(f"[AUDIT-TDA11] poblacion cargada: n={r_adj.size:,}")

    tables = []
    primary_fit = None
    for label, r, dist in (
        ("garch11_primary_global", r_adj, "normal"),
        ("garch11_secondary_raw_global", r_raw, "normal"),
        ("garch11_student_t_sensitivity", r_adj, "t"),
    ):
        print(f"[AUDIT-TDA11] A/C2 -- intentos del optimizador: {label}...")
        tab, fit = audit_arch_attempts(r, dist, label)
        tables.append(tab)
        if label == "garch11_primary_global":
            primary_fit = fit
    attempts = pd.concat(tables, ignore_index=True)

    print("[AUDIT-TDA11] B/C1 -- perfil de persistencia y maximo sin restriccion (primario)...")
    profile = audit_persistence_profile(r_adj, primary_fit)

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    paths = [config.reports_dir / ATTEMPTS_CSV_NAME, config.reports_dir / PROFILE_CSV_NAME, config.reports_dir / REPORT_NAME]
    attempts.to_csv(paths[0], index=False)
    profile.to_csv(paths[1], index=False)
    paths[2].write_text(render_audit_report(attempts, profile, time.perf_counter() - t0, run_command), encoding="utf-8")
    print("Artefactos escritos:")
    for p in paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
