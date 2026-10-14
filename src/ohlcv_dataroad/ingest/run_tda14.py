"""Punto de entrada de linea de comandos de TDA-14 (estabilidad temporal; TH29, TH30, TH31; STOP-14).

Uso
---
    python -m ohlcv_dataroad.ingest.run_tda14 calibrar [--config-tda14 configs/tda14.yaml] [--workers N]
    python -m ohlcv_dataroad.ingest.run_tda14 analizar [--config configs/mnq_snapshot.yaml] [--config-tda14 configs/tda14.yaml]

Como en TDA-00..13, este script no contiene logica estadistica propia: llama a ``tda14_pipeline`` (que compone el
modulo auditado ``tda14_temporal_stability``) y persiste toda la evidencia, incluido el informe generado
automaticamente. Bloqueos: ``calibrar`` exige un protocolo CONGELADO sin parametros pendientes; ``analizar`` exige
``analisis.autorizado_mnq: true``. El hold-out nunca se abre.
"""
from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ohlcv_dataroad.config import SnapshotConfig, load_config
from ohlcv_dataroad.ingest import tda14_pipeline as pl
from ohlcv_dataroad.ingest import tda14_temporal_stability as t14


# ---------------------------------------------------------------------------
# Calibracion: persistencia y recarga con verificacion de huellas
# ---------------------------------------------------------------------------

def persist_calibration(out: pl.CalibrationOutcome, cfg: pl.TDA14Config, out_dir: Path) -> list[Path]:
    a = cfg.artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    p = out_dir / a["umbrales_json"]
    pl.write_json(p, pl.thresholds_to_dict(out.thresholds))
    paths.append(p)
    p = out_dir / a["manifiesto_json"]
    pl.write_json(p, pl.manifest_record(out.manifest))
    paths.append(p)
    pv, ps_ = out_dir / a["validacion_json"], out_dir / a["validacion_csv"]
    pl.save_validation_files(out.validation, pv, ps_)
    paths += [ps_, pv]
    p = out_dir / a["especificaciones_csv"]
    (out.validation.specs_table if out.validation.specs_table is not None else pd.DataFrame()).to_csv(p, index=False)
    paths.append(p)
    p = out_dir / a["criterios_csv"]
    pl.acceptance_criteria_table(out.manifest).to_csv(p, index=False)
    paths.append(p)
    p = out_dir / a["calibracion_json"]
    pl.write_json(p, {"mundos": out.n_worlds, "procesos_efectivos": out.workers, "segundos": out.seconds,
                      "configuracion_tda14_sha256": cfg.sha256, "umbrales_sha256": out.thresholds.sha256,
                      "manifiesto_sha256": out.manifest.sha256(), "estado_global": out.validation.global_validated})
    paths.append(p)
    return paths


def load_calibration(cfg: pl.TDA14Config, base: Path) -> tuple[t14.FrozenThresholds | None, t14.ValidationResult | None]:
    """Umbrales y validacion indicados en la configuracion; ``(None, None)`` si no hay calibracion. Se verifica que los
    umbrales no se alteraron, que la validacion es de esos umbrales y que ambos corresponden al manifiesto actual."""
    a = cfg.raw["analisis"]
    if not a.get("umbrales"):
        return None, None
    thr = pl.thresholds_from_dict(pl.read_json(base / a["umbrales"]))
    manifest_sha = pl.build_manifest(cfg).sha256()
    if dict(thr.calibration_info).get("manifiesto_sha256") != manifest_sha:
        raise t14.ValidationManifestError("los umbrales no corresponden al manifiesto de la configuracion actual")
    if not a.get("validacion"):
        return thr, None
    val = pl.load_validation_files(base / a["validacion"], base / a["tabla_validacion"])
    if val.thresholds_sha256 != thr.sha256 or val.manifest_sha256 != manifest_sha:
        raise t14.ValidationManifestError("la validacion no corresponde a estos umbrales o a este manifiesto")
    return thr, val


# ---------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------

def _fmt(x, nd: int = 4) -> str:
    if x is None:
        return "-"
    if isinstance(x, (bool, np.bool_)):
        return "si" if x else "no"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}".replace(",", ".")
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return "-"
        return f"{x:.{nd}g}" if abs(x) < 1e-3 and x != 0 else f"{x:.{nd}f}"
    return str(x)


def _md_table(df: pd.DataFrame, columns: list[str] | None = None, nd: int = 4, max_rows: int | None = None) -> str:
    if df is None or len(df) == 0:
        return "_(sin filas)_"
    cols = columns or list(df.columns)
    d = df[cols] if max_rows is None else df[cols].head(max_rows)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(_fmt(v, nd) for v in row) + " |" for row in d.itertuples(index=False)]
    if max_rows is not None and len(df) > max_rows:
        lines.append(f"\n_({len(df) - max_rows} filas mas en el CSV)_")
    return "\n".join(lines)


def render_report(res: pl.TDA14Result, cfg: pl.TDA14Config, elapsed: float, command: str) -> str:
    a = cfg.artifacts
    dec = res.decisions
    rec_sum = pl.reconciliation_summary(res.reconciliation) if len(res.reconciliation) else pd.DataFrame()
    disc = res.reconciliation.loc[res.reconciliation["clase"] == pl.REC_DISCREPANCY] if len(res.reconciliation) else pd.DataFrame()
    annual_pt = res.annual.loc[res.annual["bloque_jornadas"] == res.annual["bloque_jornadas"].min()] if len(res.annual) else res.annual
    ne_annual = res.annual.loc[~res.annual["evaluable"]] if len(res.annual) else res.annual
    stop = res.stop14
    out = [
        "# TDA-14 — Estabilidad temporal y cambio estructural (TH29, TH30, TH31)",
        "",
        f"Generado automaticamente: {datetime.datetime.now().isoformat(timespec='seconds')}. Orden: `{command}`. Duracion: {elapsed:.1f} s.",
        "",
        "## 1. Estado y alcance",
        "",
        f"- Estado de las decisiones: **{res.status}**.",
        f"- Protocolo: **{cfg.protocol.get('estado')}**. Configuracion `{cfg.path}` (SHA-256 `{cfg.sha256[:16]}…`).",
        "- Analisis exploratorio y retrospectivo. Un cambio estadistico no identifica un regimen economico ni una causa.",
        "- Sin umbrales calibrados y validados no hay veredictos oficiales ni fechas: las tablas son descriptivas (`NO VALIDADO`).",
        "- El volumen (familia ACTIVIDAD) es descriptivo, no calibrado y sin veredicto oficial.",
        "- Hold-out: no abierto (guardias de TDA-04 antes y despues de leer).",
        "",
        "## 2. Poblacion",
        "",
        f"- Jornadas en el panel: {_fmt(int(res.stats.n_sessions_used + res.stats.n_sessions_dropped))} "
        f"(usadas en ventanas: {_fmt(int(res.stats.n_sessions_used))}; sobrantes al final: {_fmt(int(res.stats.n_sessions_dropped))}).",
        f"- Año de la jornada igual al año calendario NY de todas las barras: {_fmt(res.year_check.get('coinciden'))} "
        f"(barras con año distinto: {_fmt(res.year_check.get('barras_con_anio_distinto'))}).",
        f"- Ventanas: {len(res.stats.window_start)} ({res.stats.window_sessions} jornadas, paso {res.stats.window_sessions // res.stats.offset}); "
        f"con cobertura suficiente: {int(res.stats.evaluable.sum())}.",
        "",
        "## 3. Reconciliacion con TDA-05..TDA-13",
        "",
        "Clases predeclaradas: `REPRODUCIBLE` (misma definicion; igual hasta el redondeo), `COMPARABLE_CON_DIFERENCIA_EXPLICADA` "
        "(diferencia conocida; `_VERIFICADA` si se comprueba exactamente), `NO_COMPARABLE` (sin homologo), `DISCREPANCIA_NO_EXPLICADA` "
        "(deberia reproducirse y no lo hace), `HEREDADO_AUSENTE`.",
        "",
        _md_table(rec_sum),
        "",
        f"Discrepancias no explicadas: **{len(disc)}**." + (" Detalle:\n\n" + _md_table(disc, ["etapa", "estadistico", "grupo", "valor_heredado", "valor_tda14"]) if len(disc) else ""),
        f"Detalle completo: `{a['reconciliacion_csv']}`.",
        "",
        "## 4. Tabla anual (años completos 2020-2024)",
        "",
        "Valor puntual e IC percentil 95 % por bootstrap circular de bloques de jornadas dentro de cada año, para bloques de 1, 5 y 20 "
        f"jornadas (`{a['anual_csv']}`). Valores puntuales:",
        "",
        _md_table(annual_pt.pivot(index="estadistico", columns="anio", values="valor").reset_index() if len(annual_pt) else annual_pt),
        "",
        f"Celdas no evaluables (todas las longitudes de bloque): {len(ne_annual)}. Evaluabilidad de la curtosis "
        "(semiamplitud relativa <= 25 % en al menos 4 de 5 años):",
        "",
        _md_table(res.kurtosis),
        "",
        f"Tabla año x segmento: `{a['anio_segmento_csv']}` ({len(res.year_segment)} filas; no evaluables: "
        f"{int((~res.year_segment['evaluable']).sum()) if len(res.year_segment) else 0}). Ventanas rodantes: `{a['ventanas_csv']}`.",
        "",
        "## 5. TH29 — estabilidad por familia",
        "",
        _md_table(res.th29),
        "",
        f"Detalle por estadistico: `{a['th29_estadisticos_csv']}`.",
        "",
        "## 6. TH31 — fechas candidatas y eventos institucionales predeclarados",
        "",
        (_md_table(res.dates) if len(res.dates) else "_Ninguna fecha candidata publicada (sin calibracion, no validadas o sin detecciones)._"),
        "",
        "Eventos (solo distancia en jornadas; no se atribuye causalidad):",
        "",
        _md_table(res.events, ["evento", "fecha_negociacion", "tipo", "en_muestra", "distancia_jornadas_a_intervalo", "dentro_de_intervalo", "estado_fechas"]),
        "",
        "Cobertura de barras en la ventana de reloj predeclarada (hipotesis comprobable, no causal):",
        "",
        _md_table(res.coverage, ["evento", "ventana_reloj_ny", "anio", "periodo", "jornadas", "jornadas_con_barras", "fraccion_jornadas_con_barras", "barras"]),
        "",
        "## 7. TH30 — log-precio frente a retornos (dentro de cada contrato)",
        "",
        _md_table(res.th30_summary) if len(res.th30_summary) else "_(sin precio de cierre)_",
        "",
        "La conclusion se limita a la estabilidad relativa de los momentos dentro de cada contrato; no demuestra estacionariedad.",
        "",
        "## 8. STOP-14",
        "",
        f"- Alcance: **{stop.get('alcance')}**; definitivo: {_fmt(bool(stop.get('definitivo')))}.",
        f"- Familias persistentes confirmadas: {stop.get('familias_persistentes', [])}; pendientes de comprobaciones: {stop.get('familias_pendientes', [])}; "
        f"contradictorias: {stop.get('familias_contradictorias', [])}; cambios en el borde: {stop.get('cambios_en_borde', [])}; episodios: {stop.get('episodios', [])}.",
        "- Las comprobaciones anual y de sensibilidad a rolls y huecos son entradas EXPLICITAS de la configuracion; un valor ausente "
        "nunca cuenta como superado.",
        "",
        "## 9. Controles G2",
        "",
        "Criterios de aceptacion fijados antes de validar (limite superior de Clopper-Pearson; confianza repartida por Bonferroni):",
        "",
        _md_table(res.g2_criteria),
        "",
        _md_table(res.g2, [c for c in ("control", "decision", "eventos", "n", "tasa", "limite_confianza", "criterio", "estado") if c in res.g2.columns], max_rows=40),
        "",
        "La aceptacion significa compatibilidad con los controles evaluados, no una garantia para cualquier mundo real.",
        "",
        "## 10. Trazabilidad",
        "",
        _md_table(res.traceability.loc[~res.traceability["parametro"].str.startswith(("c_fecha.", "c_veredicto.", "detector.", "materialidad."))]),
        "",
        f"Tabla completa (umbrales por estadistico, detector, materialidades): `{a['trazabilidad_csv']}`.",
        "",
        "## 11. Tiempos por etapa",
        "",
        _md_table(pd.DataFrame([{"etapa": k, "segundos": v} for k, v in res.stage_seconds.items()])),
        "",
        "## 12. Limitaciones",
        "",
        "- Las materialidades y la capacidad minima de tipos (50 %) son convenciones documentadas, no leyes estadisticas.",
        "- Con 100 replicas por tipo o nivel de potencia la incertidumbre es de unos +-8 puntos.",
        "- El error comun por lotes del detector ignora la persistencia; su escala absoluta no se interpreta (lo absorben los umbrales calibrados).",
        "- La regla operativa del analisis anual y de la sensibilidad a rolls y huecos para STOP-14 no esta definida en el protocolo: hasta "
        "predeclararla, STOP-14 no puede ser definitivo.",
        "- Las magnitudes de potencia fuera de la escala no estan fijadas; sin ellas esas familias no pueden declararse ESTABLE.",
        "",
    ]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Graficos
# ---------------------------------------------------------------------------

def _shade_dates(ax, res: pl.TDA14Result) -> None:
    for r in res.dates.itertuples():
        ax.axvspan(pd.Timestamp(r.fecha_inicio), pd.Timestamp(r.fecha_fin), color="tab:red", alpha=0.15, lw=0)
    for r in res.events.itertuples():
        if r.en_muestra:
            ax.axvline(pd.Timestamp(r.fecha_negociacion), color="0.4", ls=":", lw=0.8)


def draw_windows(res: pl.TDA14Result, path: Path) -> None:
    stats = list(t14.DETECTOR_STATISTICS)
    w = res.windows.loc[res.windows["bloque_jornadas"] == res.windows["bloque_jornadas"].min()] if len(res.windows) else res.windows
    fig, axes = plt.subplots(len(stats), 1, figsize=(10, 2.0 * len(stats)), sharex=True)
    for ax, st in zip(np.atleast_1d(axes), stats):
        sub = w.loc[w["estadistico"] == st]
        x = pd.to_datetime(sub["fecha_inicio"])
        ax.plot(x, sub["valor"], marker=".", lw=0.8, color="tab:blue")
        ax.plot(x[~sub["evaluable"]], sub.loc[~sub["evaluable"], "valor"], "x", color="tab:gray")
        _shade_dates(ax, res)
        ax.set_ylabel(st, fontsize=7)
    np.atleast_1d(axes)[0].set_title(f"TDA-14: estadisticos por ventana ({res.stats.window_sessions} jornadas, paso {res.stats.window_sessions // res.stats.offset}); "
                                     "sombreado = fechas candidatas publicadas; punteado = eventos", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def draw_annual(res: pl.TDA14Result, path: Path) -> None:
    stats = list(t14.DETECTOR_STATISTICS)
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))
    for ax, st in zip(axes.ravel(), stats):
        sub = res.annual.loc[res.annual["estadistico"] == st]
        for j, (b, g) in enumerate(sub.groupby("bloque_jornadas")):
            x = g["anio"].to_numpy() + (j - 1) * 0.15
            # IC percentil dibujado como segmento propio: puede no contener al valor puntual (no se recorta ni se corrige).
            line = ax.plot(x, g["valor"], "o", ms=3, label=f"bloque {b}")[0]
            ax.vlines(x, g["ic95_inf"], g["ic95_sup"], color=line.get_color(), lw=0.8)
        ax.set_title(st, fontsize=8)
    axes.ravel()[0].legend(fontsize=6)
    fig.suptitle("TDA-14: tabla anual con IC 95 % (bootstrap de bloques de jornadas)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def draw_counts(res: pl.TDA14Result, thr: t14.FrozenThresholds | None, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    if res.decisions is None:
        ax.text(0.5, 0.5, "Sin umbrales calibrados: no hay recuento de familias por frontera", ha="center", va="center")
        ax.set_axis_off()
    else:
        cnt = res.decisions["recuento_por_frontera"]
        d = pd.to_datetime(res.stats.session_date)[res.stats.window_start[res.stats.offset:]]
        ax.step(d, cnt, where="mid")
        ax.axhline(thr.k_star, color="tab:red", ls="--", lw=0.8, label=f"k* = {thr.k_star}")
        _shade_dates(ax, res)
        ax.set_ylabel("familias que marcan la frontera")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def persist_analysis(res: pl.TDA14Result, cfg: pl.TDA14Config, out_dir: Path, thr: t14.FrozenThresholds | None, elapsed: float, command: str) -> list[Path]:
    a = cfg.artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {"anual_csv": res.annual, "curtosis_csv": res.kurtosis, "anio_segmento_csv": res.year_segment, "ventanas_csv": res.windows,
              "volumen_csv": res.volume, "reconciliacion_csv": res.reconciliation, "th29_csv": res.th29, "th29_estadisticos_csv": res.statistic_verdicts,
              "th31_fechas_csv": res.dates, "eventos_csv": res.events, "cobertura_csv": res.coverage, "th30_csv": res.th30,
              "th30_resumen_csv": res.th30_summary, "stop14_csv": res.stop14_families, "trazabilidad_csv": res.traceability}
    # Las tablas G2 (validacion, especificaciones, criterios) son artefactos de `calibrar`: el analisis no las reescribe.
    paths = []
    for key, df in tables.items():
        p = out_dir / a[key]
        df.to_csv(p, index=False)
        paths.append(p)
    for key, fn in (("fig_ventanas", lambda p: draw_windows(res, p)), ("fig_anual", lambda p: draw_annual(res, p)),
                    ("fig_recuento", lambda p: draw_counts(res, thr, p))):
        p = out_dir / a[key]
        fn(p)
        paths.append(p)
    p = out_dir / a["informe"]
    p.write_text(render_report(res, cfg, elapsed, command), encoding="utf-8")
    paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TDA-14: estabilidad temporal (TH29, TH30, TH31; STOP-14).")
    parser.add_argument("accion", choices=("calibrar", "analizar"))
    parser.add_argument("--config", default="configs/mnq_snapshot.yaml")
    parser.add_argument("--config-tda14", default="configs/tda14.yaml")
    parser.add_argument("--workers", type=int, default=None, help="procesos pedidos (los limita la proteccion de memoria)")
    parser.add_argument("--salida", default=None, help="carpeta de salida (por defecto, la de informes de la configuracion)")
    args = parser.parse_args(argv)
    command = "python -m ohlcv_dataroad.ingest.run_tda14 " + " ".join(argv if argv is not None else sys.argv[1:])
    cfg = pl.load_tda14_config(args.config_tda14)
    t0 = time.perf_counter()
    try:
        if args.accion == "calibrar":
            out_dir = Path(args.salida) if args.salida else load_config(args.config).reports_dir
            outcome = pl.run_calibration(cfg, args.workers)
            for p in persist_calibration(outcome, cfg, out_dir):
                print("escrito:", p)
            print(f"calibracion: {outcome.n_worlds}; procesos {outcome.workers}; global validado: {outcome.validation.global_validated}")
            return 0
        if not cfg.raw["analisis"].get("autorizado_mnq"):
            print("ERROR: el analisis sobre MNQ no esta autorizado (analisis.autorizado_mnq: false). No se lee ningun dato.", file=sys.stderr)
            return 2
        config: SnapshotConfig = load_config(args.config)
        out_dir = Path(args.salida) if args.salida else config.reports_dir
        thr, val = load_calibration(cfg, out_dir)
        inputs = pl.load_tda14_inputs(config)
        res = pl.run_analysis(inputs, cfg, thr, val, inherited_dir=config.reports_dir)
        for p in persist_analysis(res, cfg, out_dir, thr, time.perf_counter() - t0, command):
            print("escrito:", p)
        return 0
    except (pl.TDA14ConfigError, t14.ValidationManifestError, t14.FrozenThresholdsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
