"""TDA-14 -- integracion del pipeline (configuracion, carga, tablas, reconciliacion, calibracion, decisiones).

Este modulo NO añade estadisticos, umbrales, hipotesis ni procedimientos de seleccion: compone las piezas auditadas de
``tda14_temporal_stability`` (detector, clasificacion, calibracion G2, validacion, STOP-14, TH30) con las piezas
heredadas (carga de TDA-04, guardia de hold-out, cortes de segmento de TDA-06, minimos de soporte de TDA-10/TDA-12).

Lo unico "nuevo" son operaciones de contabilidad, sin parametros estadisticos:

- tablas por grupo (año, año x segmento, ventana) con el MISMO bootstrap circular por bloques de jornadas que
  ``annual_table`` (una test comprueba que coinciden), para las tres longitudes de bloque del protocolo (1, 5, 20);
- reconciliacion con TDA-05..TDA-13 segun un REGISTRO predeclarado (:data:`RECONCILIATION_REGISTRY`): cada fila declara
  si la definicion heredada es identica (``REPRODUCIBLE``), distinta con una diferencia conocida
  (``COMPARABLE_CON_DIFERENCIA_EXPLICADA``) o sin homologo (``NO_COMPARABLE``). La unica tolerancia numerica es la de
  redondeo de coma flotante (:data:`ROUNDING_RTOL`), no una tolerancia estadistica; un valor que deberia reproducirse y
  no lo hace se marca ``DISCREPANCIA_NO_EXPLICADA`` (nunca se oculta);
- cobertura de barras en ventanas de reloj predeclaradas junto a un evento institucional (hipotesis documentada, sin
  causalidad);
- serializacion de umbrales, manifiesto y validacion con verificacion de huellas SHA-256.

Bloqueos de seguridad: ``calibrar`` exige ``protocolo.estado == CONGELADO`` y ningun parametro pendiente (``null``);
``analizar`` sobre MNQ exige ``analisis.autorizado_mnq: true``. El hold-out nunca se abre (guardias heredadas).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ohlcv_dataroad.config import SnapshotConfig
from ohlcv_dataroad.ingest import tda14_stop14_rules as stop14_rules
from ohlcv_dataroad.ingest import tda14_temporal_stability as t14
from ohlcv_dataroad.ingest.holdout_guard import validate_last_timestamps_before_boundary, validate_research_holdout_disjoint
from ohlcv_dataroad.ingest.tda06_intraday_calendar_profile import COMPLETE_YEARS, attach_calendar_fields
from ohlcv_dataroad.ingest.tda07_marginal_distribution import (
    assign_segment_label,
    load_segmentation_cutoffs,
    load_tda04_inputs,
    verify_timestamp_alignment,
)
from ohlcv_dataroad.ingest.tda10_scale_vs_shape import MIN_N_FOR_STABILITY_GROUP

PROTOCOL_FROZEN, PROTOCOL_PROVISIONAL = "CONGELADO", "PROVISIONAL"
ROUNDING_RTOL, ROUNDING_ATOL = 1e-9, 1e-12          # tolerancia de redondeo de coma flotante (no estadistica)
STATUS_NO_CALIBRATION = "SIN_CALIBRACION_NO_VALIDADO"
STREAM_ANNUAL, STREAM_WINDOWS, STREAM_SEGMENTS = 99, 98, 97   # 99 = la corriente de ``annual_table`` (identica)

REC_REPRODUCIBLE = "REPRODUCIBLE"
REC_EXPLAINED = "COMPARABLE_CON_DIFERENCIA_EXPLICADA"
REC_NOT_COMPARABLE = "NO_COMPARABLE"
REC_DISCREPANCY = "DISCREPANCIA_NO_EXPLICADA"
REC_ABSENT = "HEREDADO_AUSENTE"
REC_NOT_DIRECT = "POBLACIONES_NO_DIRECTAMENTE_COMPARABLES_PENDIENTE"   # año asignado distinto y diferencia NO demostrada


class TDA14ConfigError(Exception):
    """Configuracion de TDA-14 incompleta, provisional o no autorizada para la accion pedida."""


# ---------------------------------------------------------------------------
# Configuracion reproducible
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TDA14Config:
    raw: dict
    path: str
    sha256: str

    @property
    def protocol(self) -> dict:
        return self.raw["protocolo"]

    @property
    def artifacts(self) -> dict:
        return self.raw["artefactos"]


def load_tda14_config(path: Path | str) -> TDA14Config:
    data = Path(path).read_bytes()
    raw = yaml.safe_load(data)
    for key in ("protocolo", "calibracion", "analisis", "eventos", "stop14", "artefactos"):
        if key not in raw:
            raise TDA14ConfigError(f"falta la seccion {key!r} en {path}")
    return TDA14Config(raw=raw, path=str(path), sha256=hashlib.sha256(data).hexdigest())


def detector_from_config(cfg: TDA14Config) -> t14.DetectorSpec:
    return t14.DetectorSpec(**cfg.protocol["detector"])


def _change(d: dict) -> t14.Change:
    return t14.Change(family=d["familia"], shape=d["forma"], magnitude=float(d["magnitud"]),
                      start=float(d.get("inicio", 0.5)), end=float(d.get("fin", 1.0)))


def pending_parameters(cfg: TDA14Config) -> list[str]:
    """Parametros del protocolo todavia sin valor (``null``): no puede calibrarse con ellos."""
    out = [f"mundo_base.{k}" for k, v in cfg.protocol["mundo_base"].items() if v is None]
    for name, v in cfg.protocol["variantes_negativas"].items():
        out += [f"variantes_negativas.{name}.{k}" for k, val in (v.get("parametros") or {}).items() if val is None]
    for name, v in (cfg.protocol.get("estres") or {}).items():
        out += [f"estres.{name}.{k}" for k, val in (v.get("parametros") or {}).items() if val is None]
    controls = ([cfg.protocol["parcial"]] if cfg.protocol.get("parcial") else []) + list(cfg.protocol.get("tipos") or []) + list(cfg.protocol.get("potencia") or [])
    for c in controls:
        out += [f"{c['etiqueta']}.cambio.{k}" for k, val in c["cambio"].items() if val is None]
    return out


def build_manifest(cfg: TDA14Config) -> t14.ValidationManifest:
    """Manifiesto congelado a partir de la configuracion. Exige que no quede ningun parametro pendiente."""
    pend = pending_parameters(cfg)
    if pend:
        raise TDA14ConfigError(f"parametros del protocolo pendientes (null): {pend}")
    p = cfg.protocol
    base = dict(p["mundo_base"])
    neg = {}
    for name, v in p["variantes_negativas"].items():
        neg[name] = t14.WorldSpec(**{**base, **(v.get("parametros") or {}), "label": v["etiqueta"]})
    stress = {name: t14.WorldSpec(**{**base, **(v.get("parametros") or {}), "label": v["etiqueta"]}) for name, v in (p.get("estres") or {}).items()}

    def on(variant: str, label: str, change: dict) -> t14.WorldSpec:
        src = neg.get(variant) or stress.get(variant)
        if src is None:
            raise TDA14ConfigError(f"variante base desconocida: {variant}")
        return replace(src, label=label, changes=(_change(change),))

    par = p.get("parcial")
    m = p["minimos"]
    return t14.ValidationManifest(
        negative_specs=tuple(neg.items()),
        partial_spec=on(par["variante_base"], par["etiqueta"], par["cambio"]) if par else None,
        type_specs=tuple((d["tipo"], d["familia"], on(d["variante_base"], d["etiqueta"], d["cambio"])) for d in p.get("tipos") or []),
        power_specs=tuple((d["familia"], d["nivel"], on(d["variante_base"], d["etiqueta"], d["cambio"])) for d in p.get("potencia") or []),
        detector_spec=detector_from_config(cfg), stress_specs=tuple(stress.items()),
        min_negative_replicates=int(m["negativos"]), min_partial_replicates=int(m["parcial"]),
        min_type_replicates=int(m["tipo"]), min_power_replicates=int(m["potencia"]))


def manifest_record(manifest: t14.ValidationManifest) -> dict:
    return {"sha256": manifest.sha256(), "huellas_especificaciones": manifest.spec_hashes(),
            "manifiesto": json.loads(t14._canonical(asdict(manifest)))}


# ---------------------------------------------------------------------------
# Serializacion con verificacion de huellas
# ---------------------------------------------------------------------------

def thresholds_to_dict(thr: t14.FrozenThresholds) -> dict:
    t14.verify_frozen(thr, require_complete=False)
    return {"sha256": thr.sha256, **thr.payload()}


def thresholds_from_dict(d: dict) -> t14.FrozenThresholds:
    def tup(x):
        return tuple((k, v) for k, v in x)
    thr = t14.FrozenThresholds(alpha=float(d["alpha"]), k_star=int(d["k_star"]), c_date=tup(d["c_date"]), c_verdict=tup(d["c_verdict"]),
                               type_margin=None if d["type_margin"] is None else float(d["type_margin"]), materiality=tup(d["materiality"]),
                               detector_spec=tup(d["detector_spec"]), calibration_info=tup(d["calibration_info"]), sha256=d["sha256"])
    t14.verify_frozen(thr, require_complete=False)          # rechaza cualquier archivo alterado
    return thr


VALIDATION_FLAGS = ("global_validated", "dates_validated", "families_validated", "stop14_scope_validated", "types_validated",
                    "power_ok", "thresholds_sha256", "manifest_sha256", "missing_controls", "types_validated_by_type")


def validation_to_dict(val: t14.ValidationResult) -> dict:
    d = {k: getattr(val, k) for k in VALIDATION_FLAGS}
    d["missing_controls"] = list(val.missing_controls)
    return d


def validation_from_dict(d: dict, table: pd.DataFrame, specs: pd.DataFrame | None = None) -> t14.ValidationResult:
    """Reconstruye la validacion. La integridad de la tabla se comprueba ANTES, con la huella de los bytes del archivo
    (:func:`load_validation_files`)."""
    return t14.ValidationResult(table=table, global_validated=d["global_validated"], dates_validated=d["dates_validated"],
                                families_validated=d["families_validated"], stop14_scope_validated=d["stop14_scope_validated"],
                                types_validated=d["types_validated"], power_ok=d["power_ok"], thresholds_sha256=d["thresholds_sha256"],
                                manifest_sha256=d["manifest_sha256"], missing_controls=tuple(d["missing_controls"]),
                                specs_table=specs, types_validated_by_type=d.get("types_validated_by_type", {}))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (Path,)):
        return str(o)
    raise TypeError(type(o))


def write_json(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False, default=_json_default), encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_validation_files(val: t14.ValidationResult, summary_path: Path, table_path: Path) -> None:
    """Escribe la tabla y despues el resumen con la huella SHA-256 de los BYTES de la tabla escrita."""
    val.table.to_csv(table_path, index=False)
    write_json(summary_path, {**validation_to_dict(val), "tabla_archivo": Path(table_path).name, "tabla_sha256": file_sha256(table_path)})


def load_validation_files(summary_path: Path, table_path: Path) -> t14.ValidationResult:
    d = read_json(summary_path)
    if file_sha256(table_path) != d.get("tabla_sha256"):
        raise t14.ValidationManifestError("la tabla de validacion no corresponde al resumen guardado (huella distinta)")
    return validation_from_dict(d, pd.read_csv(table_path))


# ---------------------------------------------------------------------------
# Calibracion y validacion G2 (mundos sinteticos; proteccion de memoria heredada)
# ---------------------------------------------------------------------------

@dataclass
class CalibrationOutcome:
    thresholds: t14.FrozenThresholds
    manifest: t14.ValidationManifest
    validation: t14.ValidationResult
    n_worlds: dict
    workers: int
    seconds: float


def run_calibration(cfg: TDA14Config, n_workers: int | None = None) -> CalibrationOutcome:
    """Calibracion (mundos de CALIBRACION) y validacion (mundos de VALIDACION, creados DESPUES de congelar los umbrales).
    Solo con un protocolo ``CONGELADO`` sin parametros pendientes. Numero de procesos: el pedido, limitado por la
    proteccion de memoria y CPU heredada de TDA-13 (``resolve_workers``)."""
    if cfg.protocol.get("estado") != PROTOCOL_FROZEN:
        raise TDA14ConfigError(f"el protocolo esta {cfg.protocol.get('estado')!r}: la calibracion exige {PROTOCOL_FROZEN!r}")
    manifest = build_manifest(cfg)
    c = cfg.raw["calibracion"]
    req = int(n_workers if n_workers is not None else c["n_workers"])
    dspec = manifest.detector_spec
    t0 = time.perf_counter()
    cal_tasks = []
    for _, spec in manifest.negative_specs:
        cal_tasks += t14.make_tasks(t14.ROLE_CALIBRATION, spec, int(c["mundos_negativos_por_variante"]), dspec)
    for _, spec in manifest.stress_specs:
        cal_tasks += t14.make_tasks(t14.ROLE_CALIBRATION, spec, int(c["mundos_estres_por_variante"]), dspec)
    type_tasks = []
    for _, _, spec in manifest.type_specs:
        type_tasks += t14.make_tasks(t14.ROLE_CALIBRATION, spec, int(c["mundos_tipo_por_control"]), dspec)
    workers = t14.resolve_workers(req, len(cal_tasks)) if req > 1 else 1
    cal = t14.run_world_tasks(cal_tasks, req)
    thr = t14.calibrate_thresholds(cal, manifest)
    full = t14.calibrate_type_margin(thr, t14.run_world_tasks(type_tasks, req), manifest)
    # Validacion: tareas creadas SOLO ahora, con los umbrales congelados completos.
    val_tasks = {}
    for name, spec in manifest.negative_specs:
        val_tasks[("NEG", name)] = t14.make_tasks(t14.ROLE_VALIDATION, spec, manifest.min_negative_replicates, dspec, thr=full)
    if manifest.partial_spec is not None:
        val_tasks[("PARCIAL", "")] = t14.make_tasks(t14.ROLE_VALIDATION, manifest.partial_spec, manifest.min_partial_replicates, dspec, thr=full)
    for _, _, spec in manifest.type_specs:
        val_tasks[("TIPO", spec.label)] = t14.make_tasks(t14.ROLE_VALIDATION, spec, manifest.min_type_replicates, dspec, thr=full)
    for fam, lv, spec in manifest.power_specs:
        val_tasks[("POT", (fam, lv))] = t14.make_tasks(t14.ROLE_VALIDATION, spec, manifest.min_power_replicates, dspec, thr=full)
    flat = [tk for tasks in val_tasks.values() for tk in tasks]
    done = {(r.label, r.rep): r for r in t14.run_world_tasks(flat, req)}
    got = {k: [done[(tk[1], tk[2])] for tk in tasks] for k, tasks in val_tasks.items()}
    val = t14.validate_procedure(full, manifest, {k[1]: v for k, v in got.items() if k[0] == "NEG"}, got.get(("PARCIAL", "")),
                                 [r for k, v in got.items() if k[0] == "TIPO" for r in v],
                                 {k[1]: v for k, v in got.items() if k[0] == "POT"})
    n_worlds = {"calibracion": len(cal), "calibracion_tipos": len(type_tasks), "validacion": len(flat)}
    return CalibrationOutcome(full, manifest, val, n_worlds, workers, time.perf_counter() - t0)


def acceptance_criteria_table(manifest: t14.ValidationManifest) -> pd.DataFrame:
    """Criterio de aceptacion de cada control negativo, fijado antes de validar (limite superior de Clopper-Pearson)."""
    conf = 1.0 - (1.0 - t14.G2_CONFIDENCE) / manifest.n_controls
    rows = []
    for name, n in [(nm, manifest.min_negative_replicates) for nm in manifest.negative_variants] + [("NEGATIVO_PARCIAL_SOLO_ESCALA", manifest.min_partial_replicates)]:
        rows.append({"control": name, "n_minimo": n, "confianza_bonferroni": conf, "tasa_a_demostrar": t14.G2_TARGET_RATE,
                     "max_falsas_alarmas_aceptables": t14.max_accepted_false_alarms(n, t14.G2_TARGET_RATE, conf),
                     "prob_aceptar_si_tasa_real_2_5pct": t14.acceptance_probability(n, 0.025, t14.G2_TARGET_RATE, conf),
                     "prob_aceptar_si_tasa_real_5pct": t14.acceptance_probability(n, 0.05, t14.G2_TARGET_RATE, conf)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Carga de datos reales (no se ejecuta en esta etapa)
# ---------------------------------------------------------------------------

@dataclass
class TDA14Inputs:
    panel: t14.SessionPanel
    segment_by_column: np.ndarray            # etiqueta de segmento (TDA-06) de cada minuto de sesion
    segment_labels: tuple[str, ...]
    year_check: dict                          # coincidencia año de jornada / año calendario NY de cada barra
    source: str
    year_ny_by_cell: np.ndarray | None = None  # año calendario NY de cada barra en el panel (-1 sin barra)


def segments_for_columns(cutoffs: list[int], width: int = t14.PANEL_WIDTH) -> tuple[np.ndarray, tuple[str, ...]]:
    clock = (np.arange(width) + t14.SESSION_OPEN_MINUTE_NY) % t14.PANEL_WIDTH
    labels, ordered = assign_segment_label(pd.Series(clock), cutoffs)
    return labels.to_numpy(), tuple(str(x) for x in ordered)


def load_tda14_inputs(config: SnapshotConfig) -> TDA14Inputs:
    """Guardias de hold-out ANTES y DESPUES de leer (las mismas que ``t14.load_population``), mas los cortes de segmento
    de TDA-06 y la comprobacion explicita de que el año de la jornada coincide con el año calendario NY de cada barra."""
    validate_research_holdout_disjoint(config)
    variables, validity = load_tda04_inputs(config)
    validate_last_timestamps_before_boundary(config, variables.groupby("source_file")["timestamp"].max().to_dict())
    verify_timestamp_alignment(variables, validity)
    df = attach_calendar_fields(variables, validity)
    df = df.loc[~pd.to_datetime(df["trading_date"]).dt.year.isin(t14.EXCLUDED_YEARS)]
    td_year = pd.to_datetime(df["trading_date"]).dt.year.to_numpy()
    mismatch = int((td_year != df["year_ny"].to_numpy()).sum())
    panel = t14.build_session_panel(df, config.holdout_boundary_utc)
    seg, labels = segments_for_columns(load_segmentation_cutoffs(config))
    # Año NY por celda, con la MISMA correspondencia fila/columna que ``build_session_panel``: permite DEMOSTRAR (o no)
    # que una diferencia anual con TDA-05/07/08/09/12 se debe solo a la asignacion de año.
    td = pd.to_datetime(df["trading_date"]).dt.normalize()
    row = np.searchsorted(np.sort(td.unique()), td.to_numpy())
    ny = np.full(panel.r.shape, -1, dtype=np.int16)
    ny[row, t14.session_minute(df["minute_of_day"].to_numpy())] = df["year_ny"].to_numpy()
    return TDA14Inputs(panel, seg, labels, {"barras": int(len(df)), "barras_con_anio_distinto": mismatch, "coinciden": mismatch == 0},
                       "TDA-04", ny)


# ---------------------------------------------------------------------------
# Tablas por grupo con bootstrap por bloques de jornadas
# ---------------------------------------------------------------------------

STAT_NAMES: tuple[str, ...] = tuple(t14.DETECTOR_STATISTICS) + tuple(t14.DESCRIPTIVE_STATISTICS)
QUANTILE_STATS = ("log_q01_en_sigma", "log_q99_en_sigma")


def masked_panel(panel: t14.SessionPanel, columns: np.ndarray) -> t14.SessionPanel:
    """Panel con solo las columnas indicadas (las demas sin barra): los pares de la ACF no salen del segmento."""
    keep = np.zeros(panel.r.shape[1], dtype=bool)
    keep[columns] = True
    r = np.where(keep[None, :], panel.r, np.nan)
    present = panel.present & keep[None, :]
    vol = None if panel.volume is None else np.where(present, panel.volume, np.nan)
    return t14.SessionPanel(r=r, present=present, session_date=panel.session_date, contract=panel.contract, volume=vol, close=panel.close)


def group_bootstrap_table(panel: t14.SessionPanel, groups: list[tuple[dict, np.ndarray]], n_boot: int, block_sessions: int, seed: int,
                          spec: t14.DetectorSpec, stream: int, suff: dict | None = None, ref_lp: np.ndarray | None = None) -> pd.DataFrame:
    """Valor e IC percentil 95 % por bootstrap circular de bloques de jornadas DENTRO de cada grupo. Con ``stream=99`` y
    grupos = años es exactamente ``t14.annual_table`` (mismo generador, mismo orden). Evaluabilidad con los minimos
    heredados: ``n >= MIN_N_FOR_STABILITY_GROUP`` (TDA-10) y, para los cuantiles, ``n*0,01 >= min_tail_support`` (TDA-12)."""
    suff = t14.session_sufficient_stats(panel, spec.max_lag, spec.profile_bin_minutes) if suff is None else suff
    ref_lp = t14.reference_log_profile(suff, spec.min_profile_bin_bars) if ref_lp is None else ref_lp
    rng = np.random.default_rng([t14.BASE_SEED, stream, seed, block_sessions])
    rows = []
    for key, idx in groups:
        idx = np.asarray(idx)
        if idx.size == 0:
            continue
        q01, q99, nv = t14._quantiles(panel, idx)
        point = t14.statistics_from_sums(t14._sum_rows(suff, idx), q01, q99, ref_lp, spec.min_profile_bin_bars)
        boot = {k: [] for k in STAT_NAMES}
        nb = int(math.ceil(idx.size / block_sessions))
        for _ in range(n_boot):
            starts = rng.integers(0, idx.size, size=nb)
            pick = idx[(starts[:, None] + np.arange(block_sessions)[None, :]).ravel() % idx.size][: idx.size]
            bq01, bq99, _ = t14._quantiles(panel, pick)
            st = t14.statistics_from_sums(t14._sum_rows(suff, pick), bq01, bq99, ref_lp, spec.min_profile_bin_bars)
            for k in STAT_NAMES:
                boot[k].append(st[k])
        for k in STAT_NAMES:
            b = np.asarray(boot[k], dtype=float)
            b = b[np.isfinite(b)]
            lo, hi = (np.quantile(b, [0.025, 0.975]) if b.size else (np.nan, np.nan))
            fam = t14.DETECTOR_STATISTICS[k][0] if k in t14.DETECTOR_STATISTICS else t14.DESCRIPTIVE_STATISTICS[k]
            reasons = []
            if nv < MIN_N_FOR_STABILITY_GROUP:
                reasons.append(f"n<{MIN_N_FOR_STABILITY_GROUP}")
            if k in QUANTILE_STATS and nv * 0.01 < spec.min_tail_support:
                reasons.append(f"soporte_cola<{spec.min_tail_support}")
            if not np.isfinite(point[k]):
                reasons.append("no_estimable")
            elif not (np.isfinite(lo) and np.isfinite(hi)):
                reasons.append("ic_no_estimable")
            rows.append({**key, "estadistico": k, "familia": fam, "en_detector": k in t14.DETECTOR_STATISTICS,
                         "estado_calibracion": "DETECTOR" if k in t14.DETECTOR_STATISTICS else t14.VOLUME_STATUS if fam == t14.FAM_ACTIVITY else "DESCRIPTIVO",
                         "valor": point[k], "ic95_inf": float(lo), "ic95_sup": float(hi), "bloque_jornadas": block_sessions, "n_boot": n_boot,
                         "n_jornadas": int(idx.size), "n_retornos": int(nv), "evaluable": not reasons, "motivo_no_evaluable": ";".join(reasons)})
    return pd.DataFrame(rows)


def year_groups(panel: t14.SessionPanel, years=COMPLETE_YEARS) -> list[tuple[dict, np.ndarray]]:
    yr = pd.to_datetime(panel.session_date).year.to_numpy()
    return [({"anio": int(y)}, np.flatnonzero(yr == y)) for y in years if (yr == y).any()]


def annual_tables(panel, spec, n_boot, seed, blocks=t14.ANNUAL_BLOCK_SESSIONS) -> pd.DataFrame:
    suff = t14.session_sufficient_stats(panel, spec.max_lag, spec.profile_bin_minutes)
    ref = t14.reference_log_profile(suff, spec.min_profile_bin_bars)
    return pd.concat([group_bootstrap_table(panel, year_groups(panel), n_boot, b, seed, spec, STREAM_ANNUAL, suff, ref) for b in blocks], ignore_index=True)


def year_segment_tables(panel, segment_by_column, segment_labels, spec, n_boot, seed, blocks=t14.ANNUAL_BLOCK_SESSIONS) -> pd.DataFrame:
    full_suff = t14.session_sufficient_stats(panel, spec.max_lag, spec.profile_bin_minutes)
    ref = t14.reference_log_profile(full_suff, spec.min_profile_bin_bars)
    out = []
    for i, seg in enumerate(segment_labels):
        cols = np.flatnonzero(segment_by_column[: panel.r.shape[1]] == seg)
        if cols.size == 0:
            continue
        sp = masked_panel(panel, cols)
        suff = t14.session_sufficient_stats(sp, spec.max_lag, spec.profile_bin_minutes)
        groups = [({**k, "segmento": seg}, idx) for k, idx in year_groups(sp)]
        for b in blocks:
            out.append(group_bootstrap_table(sp, groups, n_boot, b, seed, spec, STREAM_SEGMENTS * 100 + i, suff, ref))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def window_tables(panel, ps: t14.PanelStats, spec, n_boot, seed, blocks=t14.ANNUAL_BLOCK_SESSIONS) -> pd.DataFrame:
    """Ventanas rodantes (63 jornadas, paso 21): valor, error comun por lotes del detector, error por ventana y el IC por
    bootstrap de bloques dentro de la ventana para cada longitud de bloque. Evaluabilidad: cobertura del detector + minimos."""
    suff = t14.session_sufficient_stats(panel, spec.max_lag, spec.profile_bin_minutes)
    ref = t14.reference_log_profile(suff, spec.min_profile_bin_bars)
    dates = pd.to_datetime(panel.session_date)
    groups = []
    for i, s0 in enumerate(ps.window_start):
        idx = np.arange(s0, s0 + ps.window_sessions)
        groups.append(({"ventana": i, "jornada_inicio": int(s0), "fecha_inicio": dates[s0].date().isoformat(),
                        "fecha_fin": dates[idx[-1]].date().isoformat(), "sin_solapamiento": i % ps.offset == 0,
                        "cobertura_suficiente": bool(ps.evaluable[i])}, idx))
    out = pd.concat([group_bootstrap_table(panel, groups, n_boot, b, seed, spec, STREAM_WINDOWS, suff, ref) for b in blocks], ignore_index=True)
    se_common = {(k, i): ps.se[k][i] for k in ps.se for i in range(len(ps.window_start))}
    se_win = {(k, i): ps.se_by_window[k][i] for k in ps.se_by_window for i in range(len(ps.window_start))}
    out["error_comun_detector"] = [se_common.get((k, i), np.nan) for k, i in zip(out["estadistico"], out["ventana"])]
    out["error_por_lotes_ventana"] = [se_win.get((k, i), np.nan) for k, i in zip(out["estadistico"], out["ventana"])]
    out["evaluable"] = out["evaluable"] & out["cobertura_suficiente"]
    out.loc[~out["cobertura_suficiente"], "motivo_no_evaluable"] = (out.loc[~out["cobertura_suficiente"], "motivo_no_evaluable"] + ";cobertura").str.strip(";")
    return out


def kurtosis_evaluability_by_block(annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b, sub in annual.groupby("bloque_jornadas"):
        ev = t14.kurtosis_evaluability(sub)
        rows.append({"bloque_jornadas": b, "evaluable": ev["evaluable"], "anios_que_cumplen": ev["anios_que_cumplen"],
                     "semiamplitud_relativa": json.dumps([None if not np.isfinite(x) else round(float(x), 6) for x in ev["semiamplitud_relativa"]])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reconciliacion con TDA-05..TDA-13
# ---------------------------------------------------------------------------

def _pairwise_acf_by_group(values: np.ndarray, valid: np.ndarray, lags, row_groups: np.ndarray | None, col_groups: np.ndarray | None,
                           cell_groups: np.ndarray | None = None):
    """ACF HEREDADA (TDA-08/TDA-09, ``compute_acf``/``acf_by_group``): Pearson pairwise-complete sobre los pares (t-k, t)
    del mismo bloque de retornos validos, cada lado centrado por SU media sobre esos pares; cada par se etiqueta con el
    grupo de ``t`` (la barra mas reciente). Devuelve {(grupo, lag): (rho, n_pares)}."""
    runs = t14.continuity_runs(valid)
    x = np.where(valid, values, 0.0)
    out = {}
    for lag in lags:
        pm = t14.pair_mask(valid, runs, lag)
        xt, xk = x[:, lag:], x[:, :-lag]
        if cell_groups is not None:
            g = cell_groups[:, lag:]
        elif row_groups is not None:
            g = np.broadcast_to(row_groups[:, None], pm.shape)
        else:
            g = np.broadcast_to(col_groups[None, lag:], pm.shape)
        for lab in pd.unique(g[pm]):
            sel = pm & (g == lab)
            n = int(sel.sum())
            a, b = xt[sel], xk[sel]
            if n < 2:
                continue
            ea, eb = a - a.mean(), b - b.mean()
            den = math.sqrt(float((ea * ea).sum()) * float((eb * eb).sum()))
            out[(str(lab), lag)] = (float((ea * eb).sum() / den) if den > 0 else np.nan, n)
    return out


def _tda07_moments(x: np.ndarray) -> dict:
    """Definiciones de TDA-07 (``compute_moments_quantiles``): std con ddof=1; curtosis en exceso con momentos ddof=0."""
    m = x.mean()
    m2 = ((x - m) ** 2).mean()
    q = np.quantile(x, [0.01, 0.99])
    return {"n": float(x.size), "mean": float(m), "std": float(x.std(ddof=1)), "kurtosis_excess": float(((x - m) ** 4).mean() / m2 ** 2 - 3.0),
            "q0.01": float(q[0]), "q0.99": float(q[1]), "zero_fraction": float((x == 0).mean())}


@dataclass(frozen=True)
class RecEntry:
    stage: str
    file: str
    statistic: str
    expected: str
    explanation: str


# Registro PREDECLARADO: que se compara, con que definicion y por que.
RECONCILIATION_REGISTRY: tuple[RecEntry, ...] = (
    RecEntry("TDA-07", "TDA07_momentos_cuantiles_drift_colas.csv", "n|mean|std|kurtosis_excess|q0.01|q0.99 por año (r_1m)", REC_REPRODUCIBLE,
             "misma poblacion (r_1m valido) y mismas definiciones de TDA-07; año de la jornada = año calendario NY si la carga lo confirma"),
    RecEntry("TDA-12", "TDA12_cuantiles_condicionales.csv", "std y q (0,01/0,99) por año (r_1m, bloque raw)", REC_REPRODUCIBLE,
             "TDA-12 usa std ddof=1 y los cuantiles de TDA-07"),
    RecEntry("TDA-05", "TDA05_resolucion_por_anio.csv", "n y zero_fraction por año", REC_REPRODUCIBLE, "fraccion de r_1m valido igual a 0"),
    RecEntry("TDA-08", "TDA08_acf_por_grupo.csv", "rho lag 1 por año (r_1m)", REC_REPRODUCIBLE, "ACF pairwise-complete heredada, par etiquetado por t"),
    RecEntry("TDA-09", "TDA09_persistencia_por_anio.csv", "rho de |r| (raw) lags 1/5/20/60 por año", REC_REPRODUCIBLE, "ACF pairwise-complete heredada sobre |r_1m|"),
    RecEntry("TDA-09", "TDA09_persistencia_por_segmento.csv", "rho de |r| (raw) lags 1/5/20/60 por segmento", REC_EXPLAINED,
             "TDA-09 incluye las 6 jornadas de 2019, que TDA-14 excluye; diferencia esperada pequeña, no verificable exactamente"),
    RecEntry("TDA-14 vs TDA-07", "TDA07_momentos_cuantiles_drift_colas.csv", "log_sigma, curtosis_total, media_en_sigma", REC_EXPLAINED,
             "TDA-14 usa sigma con ddof=0 y curtosis total (exceso+3): diferencia exacta y verificada con el factor sqrt((n-1)/n) y +3"),
    RecEntry("TDA-14 vs TDA-12", "TDA12_cuantiles_condicionales.csv", "log_q01_en_sigma, log_q99_en_sigma", REC_EXPLAINED,
             "q/sigma de TDA-14 (ddof=0) frente a q_over_std de TDA-12 (ddof=1): diferencia exacta y verificada"),
    RecEntry("TDA-14 vs TDA-05", "TDA05_resolucion_por_anio.csv", "log_fraccion_cero", REC_REPRODUCIBLE, "misma fraccion de ceros, en logaritmo"),
    RecEntry("TDA-14 vs TDA-08", "TDA08_acf_por_grupo.csv", "rho1", REC_EXPLAINED,
             "TDA-14 centra con la media comun del grupo; TDA-08 con la media de cada lado sobre los pares: diferencia no verificable exactamente"),
    RecEntry("TDA-14", "-", "persistencia_acf_abs", REC_NOT_COMPARABLE, "indicador derivado nuevo de TDA-14 (media de la ACF de |r|, lags 1..60) sin homologo heredado"),
    RecEntry("TDA-06", "TDA06_perfil_minuto_por_anio.csv", "perfil de reloj", REC_NOT_COMPARABLE,
             "TDA-06 resume |r| con la MEDIANA por minuto; TDA-14 con la MEDIA por tramo de 30 min normalizada"),
    RecEntry("TDA-06", "TDA06_perfil_minuto_por_anio.csv", "volumen", REC_NOT_COMPARABLE, "poblaciones de barras no documentadas como identicas; volumen solo descriptivo"),
    RecEntry("TDA-10", "TDA10_cuantiles_por_anio.csv", "cuantiles de z (EWMA)", REC_NOT_COMPARABLE, "TDA-14 no estandariza con EWMA"),
    RecEntry("TDA-11", "TDA11_garch_parametros.csv", "parametros GARCH", REC_NOT_COMPARABLE, "TDA-14 no ajusta modelos parametricos"),
    RecEntry("TDA-13", "TDA13_agrupamiento_por_anio_segmento.csv", "agrupamiento de extremos", REC_NOT_COMPARABLE, "TDA-14 no estudia extremos"),
)


def _close(a: float, b: float) -> bool:
    return bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= ROUNDING_ATOL + ROUNDING_RTOL * max(abs(a), abs(b)))


def _row(entry: RecEntry, group: str, stat: str, inherited, tda14, cls: str, note: str = "") -> dict:
    inh = float(inherited) if inherited is not None and inherited == inherited else np.nan
    own = float(tda14) if tda14 is not None and tda14 == tda14 else np.nan
    return {"etapa": entry.stage, "archivo": entry.file, "estadistico": stat, "grupo": group, "valor_heredado": inh, "valor_tda14": own,
            "diferencia": own - inh if np.isfinite(inh) and np.isfinite(own) else np.nan, "clase_declarada": entry.expected,
            "clase": cls, "explicacion": entry.explanation + (f" | {note}" if note else "")}


def _check(entry: RecEntry, group: str, stat: str, inherited, tda14, year_ok: bool = True, tda14_ny=None) -> dict:
    """REPRODUCIBLE declarado: igual hasta el redondeo -> REPRODUCIBLE. Si no coincide:

    - años iguales en todas las barras -> ``DISCREPANCIA_NO_EXPLICADA``;
    - algunas barras con año distinto (correccion de cierre): la diferencia solo es EXPLICADA si se DEMUESTRA, es decir,
      si el mismo calculo con el año NY de cada barra (``tda14_ny``) reproduce el valor heredado (``..._VERIFICADA``).
      Si ese recalculo no lo reproduce -> ``DISCREPANCIA_NO_EXPLICADA`` (una diferencia de poblacion nunca oculta un
      error independiente). Sin recalculo posible -> ``POBLACIONES_NO_DIRECTAMENTE_COMPARABLES_PENDIENTE``."""
    if inherited is None or not np.isfinite(inherited):
        return _row(entry, group, stat, inherited, tda14, REC_ABSENT)
    if _close(float(inherited), float(tda14)):
        return _row(entry, group, stat, inherited, tda14, REC_REPRODUCIBLE)
    if year_ok:
        return _row(entry, group, stat, inherited, tda14, REC_DISCREPANCY)
    if tda14_ny is None or not np.isfinite(tda14_ny):
        return _row(entry, group, stat, inherited, tda14, REC_NOT_DIRECT, "asignacion de año distinta en algunas barras; diferencia no demostrada")
    if _close(float(inherited), float(tda14_ny)):
        return _row(entry, group, stat, inherited, tda14, REC_EXPLAINED + "_VERIFICADA",
                    f"recalculado con el año NY de cada barra coincide ({float(tda14_ny):.12g}): la diferencia es solo de asignacion de año")
    return _row(entry, group, stat, inherited, tda14, REC_DISCREPANCY,
                f"no coincide ni con el año de la jornada ni recalculado con el año NY de cada barra ({float(tda14_ny):.12g})")


def reconcile(panel: t14.SessionPanel, annual: pd.DataFrame, segment_by_column: np.ndarray, inherited_dir: Path | str | None,
              year_check: dict | None = None, year_ny_by_cell: np.ndarray | None = None) -> pd.DataFrame:
    """Reconciliacion explicita segun :data:`RECONCILIATION_REGISTRY`. Nunca ajusta nada: compara y clasifica."""
    reg = {(e.stage, e.statistic): e for e in RECONCILIATION_REGISTRY}
    d = Path(inherited_dir) if inherited_dir is not None else None
    year_ok = bool((year_check or {}).get("coinciden", True))
    yr = pd.to_datetime(panel.session_date).year.to_numpy()
    valid = np.isfinite(panel.r)
    rows = []

    def read(name):
        p = d / name if d is not None else None
        return pd.read_csv(p) if p is not None and p.exists() else None

    by_year = {int(y): _tda07_moments(panel.r[yr == y][valid[yr == y]]) for y in np.unique(yr)}
    # Recalculo con el año NY de cada barra (solo si hay barras con año distinto y se dispone del año por celda).
    ny = year_ny_by_cell if (not year_ok and year_ny_by_cell is not None) else None
    by_ny = {} if ny is None else {int(y): _tda07_moments(panel.r[(ny == y) & valid]) for y in np.unique(ny[valid])}

    def nyv(y, key):
        return by_ny.get(y, {}).get(key) if ny is not None else None
    point = annual.loc[annual["bloque_jornadas"] == annual["bloque_jornadas"].min()] if len(annual) else annual
    own = {(int(r.anio), r.estadistico): r.valor for r in point.itertuples()} if len(point) else {}

    # TDA-07 por año (definiciones heredadas)
    e = reg[("TDA-07", "n|mean|std|kurtosis_excess|q0.01|q0.99 por año (r_1m)")]
    t07 = read(e.file)
    for y, mom in by_year.items():
        sel = None if t07 is None else t07.loc[(t07["series"] == "r_1m") & (t07["scope"] == "YEAR") & (t07["scope_value"].astype(str) == str(y))]
        for s in ("n", "mean", "std", "kurtosis_excess", "q0.01", "q0.99"):
            inh = None if sel is None or sel.empty else sel.iloc[0][s]
            rows.append(_check(e, str(y), s, inh, mom[s], year_ok, nyv(y, s)))
        # TDA-14 frente a TDA-07: diferencias exactas y verificables
        if (y, "log_sigma") in own and sel is not None and not sel.empty:
            ex = reg[("TDA-14 vs TDA-07", "log_sigma, curtosis_total, media_en_sigma")]
            n = mom["n"]
            for stat, inh, val in (("log_sigma -> std(ddof=1)", sel.iloc[0]["std"], math.exp(own[(y, "log_sigma")]) * math.sqrt(n / (n - 1))),
                                   ("curtosis_total -> kurtosis_excess", sel.iloc[0]["kurtosis_excess"], own[(y, "curtosis_total")] - 3.0),
                                   ("media_en_sigma -> mean/std(ddof=1)", sel.iloc[0]["mean"] / sel.iloc[0]["std"], own[(y, "media_en_sigma")] * math.sqrt((n - 1) / n))):
                r = _check(ex, str(y), stat, inh, val, year_ok)
                if r["clase"] == REC_REPRODUCIBLE:
                    r["clase"] = REC_EXPLAINED + "_VERIFICADA"
                rows.append(r)
    if t07 is None:
        rows.append(_row(e, "-", "archivo", None, None, REC_ABSENT))

    # TDA-12 por año
    e = reg[("TDA-12", "std y q (0,01/0,99) por año (r_1m, bloque raw)")]
    t12 = read(e.file)
    for y, mom in by_year.items():
        if t12 is None:
            break
        sel = t12.loc[(t12["series"] == "r_1m") & (t12["block"] == "raw") & (t12["grouping"] == "YEAR") & (t12["group"].astype(str) == str(y))]
        rows.append(_check(e, str(y), "std", None if sel.empty else sel.iloc[0]["std"], mom["std"], year_ok, nyv(y, "std")))
        for lvl, key in ((0.01, "q0.01"), (0.99, "q0.99")):
            s = sel.loc[np.isclose(sel["level"].astype(float), lvl)]
            rows.append(_check(e, str(y), f"q{lvl}", None if s.empty else s.iloc[0]["q"], mom[key], year_ok, nyv(y, key)))
            if (y, "log_q99_en_sigma") in own and not s.empty:
                ex = reg[("TDA-14 vs TDA-12", "log_q01_en_sigma, log_q99_en_sigma")]
                n = mom["n"]
                stat = "log_q01_en_sigma" if lvl == 0.01 else "log_q99_en_sigma"
                sign = -1.0 if lvl == 0.01 else 1.0
                val = sign * math.exp(own[(y, stat)]) * math.sqrt((n - 1) / n)
                r = _check(ex, str(y), f"{stat} -> q_over_std", s.iloc[0]["q_over_std"], val, year_ok)
                if r["clase"] == REC_REPRODUCIBLE:
                    r["clase"] = REC_EXPLAINED + "_VERIFICADA"
                rows.append(r)
    if t12 is None:
        rows.append(_row(e, "-", "archivo", None, None, REC_ABSENT))

    # TDA-05 por año
    e = reg[("TDA-05", "n y zero_fraction por año")]
    t05 = read(e.file)
    for y, mom in by_year.items():
        sel = None if t05 is None else t05.loc[t05["year_ny"].astype(str) == str(y)]
        for s in ("n", "zero_fraction"):
            rows.append(_check(e, str(y), s, None if sel is None or sel.empty else sel.iloc[0][s], mom[s], year_ok, nyv(y, s)))
        if (y, "log_fraccion_cero") in own:
            ex = reg[("TDA-14 vs TDA-05", "log_fraccion_cero")]
            rows.append(_check(ex, str(y), "exp(log_fraccion_cero)", None if sel is None or sel.empty else sel.iloc[0]["zero_fraction"],
                               math.exp(own[(y, "log_fraccion_cero")]), year_ok))
    if t05 is None:
        rows.append(_row(e, "-", "archivo", None, None, REC_ABSENT))

    # TDA-08 rho lag 1 por año
    e = reg[("TDA-08", "rho lag 1 por año (r_1m)")]
    t08 = read(e.file)
    acf_r = _pairwise_acf_by_group(panel.r, valid, (1,), yr.astype(str), None)
    acf_r_ny = {} if ny is None else _pairwise_acf_by_group(panel.r, valid, (1,), None, None, ny.astype(str))
    for (g, lag), (rho, npairs) in sorted(acf_r.items()):
        sel = None if t08 is None else t08.loc[(t08["group_type"] == "year") & (t08["group"].astype(str) == g) & (t08["lag"] == lag)]
        inh = None if sel is None or sel.empty else sel.iloc[0]["rho"]
        rows.append(_check(e, g, "rho_lag1", inh, rho, year_ok, acf_r_ny.get((g, lag), (None,))[0] if ny is not None else None))
        if (int(g), "rho1") in own:
            ex = reg[("TDA-14 vs TDA-08", "rho1")]
            cls = REC_ABSENT if inh is None else (REC_EXPLAINED if year_ok else REC_NOT_DIRECT)
            rows.append(_row(ex, g, "rho1 (centrado comun)", inh, own[(int(g), "rho1")], cls))
    if t08 is None:
        rows.append(_row(e, "-", "archivo", None, None, REC_ABSENT))

    # TDA-09 |r| por año y por segmento
    absr = np.abs(panel.r)
    e = reg[("TDA-09", "rho de |r| (raw) lags 1/5/20/60 por año")]
    t09 = read(e.file)
    acf_a_ny = {} if ny is None else _pairwise_acf_by_group(absr, valid, (1, 5, 20, 60), None, None, ny.astype(str))
    for (g, lag), (rho, _) in sorted(_pairwise_acf_by_group(absr, valid, (1, 5, 20, 60), yr.astype(str), None).items()):
        sel = None if t09 is None else t09.loc[(t09["variable"] == "abs_r") & (t09["raw_adjusted"] == "raw") & (t09["group"].astype(str) == g) & (t09["lag"] == lag)]
        rows.append(_check(e, g, f"rho_abs_lag{lag}", None if sel is None or sel.empty else sel.iloc[0]["rho"], rho, year_ok,
                           acf_a_ny.get((g, lag), (None,))[0] if ny is not None else None))
    e = reg[("TDA-09", "rho de |r| (raw) lags 1/5/20/60 por segmento")]
    t09s = read(e.file)
    seg = segment_by_column[: panel.r.shape[1]]
    for (g, lag), (rho, _) in sorted(_pairwise_acf_by_group(absr, valid, (1, 5, 20, 60), None, seg).items()):
        sel = None if t09s is None else t09s.loc[(t09s["variable"] == "abs_r") & (t09s["raw_adjusted"] == "raw") & (t09s["group"].astype(str) == g) & (t09s["lag"] == lag)]
        inh = None if sel is None or sel.empty else sel.iloc[0]["rho"]
        rows.append(_row(e, g, f"rho_abs_lag{lag}", inh, rho, REC_EXPLAINED if inh is not None else REC_ABSENT))

    for e in RECONCILIATION_REGISTRY:
        if e.expected == REC_NOT_COMPARABLE:
            rows.append(_row(e, "-", e.statistic, None, None, REC_NOT_COMPARABLE))
    return pd.DataFrame(rows)


def reconciliation_summary(rec: pd.DataFrame) -> pd.DataFrame:
    return rec.groupby(["etapa", "clase"]).size().rename("filas").reset_index()


# ---------------------------------------------------------------------------
# Decisiones: TH29, TH31, eventos institucionales, cobertura, STOP-14, TH30
# ---------------------------------------------------------------------------

def th29_table(dec: dict | None) -> pd.DataFrame:
    if dec is None:
        return pd.DataFrame([{"familia": f, "veredicto": STATUS_NO_CALIBRATION, "anotacion": "", "soporte": "", "veredicto_doble_materialidad": "",
                              "validacion": t14.STATUS_NOT_VALIDATED} for f in t14.DETECTOR_FAMILIES]
                            + [{"familia": t14.FAM_ACTIVITY, "veredicto": t14.VOLUME_VERDICT, "anotacion": t14.VOLUME_STATUS, "soporte": "",
                                "veredicto_doble_materialidad": "", "validacion": t14.VOLUME_STATUS}])
    rows = []
    for f in t14.DETECTOR_FAMILIES:
        v, s = dec["veredictos_familia"][f], dec["veredictos_familia_estricta"][f]
        rows.append({"familia": f, "veredicto": v["veredicto"], "anotacion": v.get("anotacion", ""), "soporte": t14.support_of(v),
                     "veredicto_doble_materialidad": s["veredicto"], "validacion": v.get("validacion", t14.STATUS_NOT_VALIDATED)})
    rows.append({"familia": t14.FAM_ACTIVITY, "veredicto": t14.VOLUME_VERDICT, "anotacion": t14.VOLUME_STATUS, "soporte": "",
                 "veredicto_doble_materialidad": "", "validacion": t14.VOLUME_STATUS})
    return pd.DataFrame(rows)


def statistic_verdicts_table(dec: dict | None) -> pd.DataFrame:
    if dec is None:
        return pd.DataFrame(columns=["estadistico", "familia", "veredicto", "anotacion", "soporte", "D", "modelo", "recorrido", "variacion_observada", "n_ventanas"])
    rows = []
    for st, v in dec["veredictos_estadistico"].items():
        rows.append({"estadistico": st, "familia": t14.DETECTOR_STATISTICS[st][0], "veredicto": v["veredicto"], "anotacion": v.get("anotacion", ""),
                     "soporte": t14.support_of(v), "D": v.get("D"), "modelo": v.get("modelo", ""), "recorrido": v.get("recorrido"),
                     "variacion_observada": v.get("variacion_observada"), "n_ventanas": v.get("n_ventanas")})
    return pd.DataFrame(rows)


def dates_table(dec: dict | None, panel: t14.SessionPanel) -> pd.DataFrame:
    cols = ["intervalo", "jornada_inicio", "jornada_fin", "fecha_inicio", "fecha_fin", "familias_max"]
    if not dec or not dec["fechas"]:
        return pd.DataFrame(columns=cols)
    d = pd.to_datetime(panel.session_date)
    return pd.DataFrame([{"intervalo": i, "jornada_inicio": f["jornada_inicio"], "jornada_fin": f["jornada_fin"],
                          "fecha_inicio": d[f["jornada_inicio"]].date().isoformat(), "fecha_fin": d[f["jornada_fin"]].date().isoformat(),
                          "familias_max": f["familias_max"]} for i, f in enumerate(dec["fechas"])], columns=cols)


def events_table(events: list[dict], panel: t14.SessionPanel, dates: pd.DataFrame, dates_published: bool) -> pd.DataFrame:
    """Eventos institucionales PREDECLARADOS frente a los intervalos candidatos: solo distancia en jornadas y si cae dentro.
    Nunca se atribuye causalidad: una coincidencia temporal no identifica un regimen ni un efecto."""
    d = pd.to_datetime(panel.session_date)
    rows = []
    for ev in events:
        day = pd.Timestamp(ev["fecha_negociacion"])
        idx = int(np.searchsorted(d.values, day.to_datetime64()))
        in_sample = 0 < idx < len(d)
        dist, inside = np.nan, False
        if in_sample and dates_published and len(dates):
            gaps = [0 if a <= idx <= b else min(abs(idx - a), abs(idx - b)) for a, b in zip(dates["jornada_inicio"], dates["jornada_fin"])]
            dist, inside = float(min(gaps)), min(gaps) == 0
        rows.append({"evento": ev["id"], "fecha_negociacion": day.date().isoformat(), "tipo": ev["tipo"], "cambio": ev["cambio"],
                     "fuente": ev["fuente"], "mnq_citado": ev.get("mnq_citado", ""), "jornada": idx if in_sample else np.nan,
                     "en_muestra": in_sample, "distancia_jornadas_a_intervalo": dist, "dentro_de_intervalo": inside,
                     "estado_fechas": "PUBLICADAS" if dates_published else "NO_PUBLICADAS (no validadas o sin calibracion)",
                     "interpretacion": "coincidencia temporal descriptiva; no se atribuye causalidad"})
    return pd.DataFrame(rows)


def clock_window_coverage(panel: t14.SessionPanel, events: list[dict]) -> pd.DataFrame:
    """Cobertura de barras en las ventanas de reloj NY predeclaradas por evento (hipotesis comprobable, sin causalidad):
    por año y antes/despues de la fecha del evento, jornadas con alguna barra en la ventana y barras totales."""
    rows = []
    d = pd.to_datetime(panel.session_date)
    for ev in events:
        w = ev.get("ventana_reloj_ny")
        if not w:
            continue
        h0, m0 = map(int, w["inicio"].split(":"))
        h1, m1 = map(int, w["fin"].split(":"))
        clock = (np.arange(panel.present.shape[1]) + t14.SESSION_OPEN_MINUTE_NY) % t14.PANEL_WIDTH
        cols = np.flatnonzero((clock >= h0 * 60 + m0) & (clock < h1 * 60 + m1))
        bars = panel.present[:, cols].sum(1) if cols.size else np.zeros(len(d), dtype=int)
        after = d >= pd.Timestamp(ev["fecha_negociacion"])
        for (yr, aft), idx in pd.Series(np.arange(len(d))).groupby([d.year, after]):
            ii = idx.to_numpy()
            rows.append({"evento": ev["id"], "ventana_reloj_ny": f"{w['inicio']}-{w['fin']}", "anio": int(yr),
                         "periodo": "desde_evento" if aft else "antes_evento", "jornadas": int(ii.size),
                         "jornadas_con_barras": int((bars[ii] > 0).sum()), "fraccion_jornadas_con_barras": float((bars[ii] > 0).mean()),
                         "barras": int(bars[ii].sum()), "hipotesis": w.get("hipotesis", ""), "interpretacion": "descriptivo; no demuestra causalidad"})
    return pd.DataFrame(rows)


def th30_tables(panel: t14.SessionPanel) -> tuple[pd.DataFrame, pd.DataFrame]:
    tabs, summ = [], []
    for L in t14.TH30_SUB_SESSIONS:
        tab = t14.th30_within_contract(panel, L)
        tabs.append(tab)
        summ.append({"jornadas_subventana": L, **t14.th30_summary(tab)})
    return (pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame()), pd.DataFrame(summ)


def stop14_table(dec: dict | None) -> pd.DataFrame:
    if dec is None:
        return pd.DataFrame([{"familia": f, "estado_persistencia": STATUS_NO_CALIBRATION} for f in t14.DETECTOR_FAMILIES])
    st = dec["stop14"]
    return pd.DataFrame([{"familia": f, "estado_persistencia": s} for f, s in st.get("estado_por_familia", {}).items()])


# ---------------------------------------------------------------------------
# Analisis completo
# ---------------------------------------------------------------------------

@dataclass
class TDA14Result:
    stats: t14.PanelStats
    annual: pd.DataFrame
    kurtosis: pd.DataFrame
    year_segment: pd.DataFrame
    windows: pd.DataFrame
    volume: pd.DataFrame
    reconciliation: pd.DataFrame
    decisions: dict | None
    th29: pd.DataFrame
    statistic_verdicts: pd.DataFrame
    dates: pd.DataFrame
    events: pd.DataFrame
    coverage: pd.DataFrame
    th30: pd.DataFrame
    th30_summary: pd.DataFrame
    stop14: dict
    stop14_families: pd.DataFrame
    g2: pd.DataFrame
    g2_specs: pd.DataFrame
    g2_criteria: pd.DataFrame
    traceability: pd.DataFrame
    status: str
    year_check: dict
    stage_seconds: dict = field(default_factory=dict)
    stop14_annual_evidence: pd.DataFrame = field(default_factory=pd.DataFrame)
    stop14_sensitivity_evidence: pd.DataFrame = field(default_factory=pd.DataFrame)
    stop14_checks_status: str = "NO_EJECUTADAS_SIN_UMBRALES"


def reject_manual_stop14_checks(cfg: TDA14Config) -> None:
    """Las comprobaciones de STOP-14 se CALCULAN (``tda14_stop14_rules``): una configuracion con valores manuales de
    confirmacion anual o de sensibilidad se rechaza, para que ninguna comprobacion pueda declararse sin ejecutarla. Si la
    configuracion declara la version de las reglas, debe ser la del codigo."""
    st = cfg.raw["stop14"] or {}
    manual = {k: st.get(k) for k in ("confirmacion_anual", "sensibilidad_rolls_huecos") if st.get(k)}
    if manual:
        raise TDA14ConfigError(f"STOP-14: no se aceptan comprobaciones manuales en la configuracion ({sorted(manual)}); se calculan con las reglas A y B")
    if st.get("reglas_version") not in (None, stop14_rules.RULES_VERSION):
        raise TDA14ConfigError(f"STOP-14: la configuracion declara las reglas {st.get('reglas_version')!r}; el codigo aplica {stop14_rules.RULES_VERSION!r}")


def run_analysis(inputs: TDA14Inputs, cfg: TDA14Config, thresholds: t14.FrozenThresholds | None = None,
                 validation: t14.ValidationResult | None = None, inherited_dir: Path | str | None = None,
                 extra_trace: dict | None = None) -> TDA14Result:
    """Analisis completo sobre un panel YA cargado (real o sintetico). Sin umbrales calibrados no hay veredictos ni fechas:
    todo queda descriptivo y ``NO VALIDADO``. Con umbrales, las consecuencias de la validacion se aplican tal cual
    (``t14.final_decisions``)."""
    a = cfg.raw["analisis"]
    spec = thresholds and t14._spec_from(thresholds) or detector_from_config(cfg)
    if thresholds is not None and t14.detector_spec_sha256(spec) != t14.detector_spec_sha256(detector_from_config(cfg)):
        raise TDA14ConfigError("los umbrales se calibraron con un detector distinto del de la configuracion")
    n_boot, seed = int(a["n_boot"]), int(a["semilla"])
    timer = {}
    t0 = time.perf_counter()
    ps = t14.analyze_panel(inputs.panel, spec)
    timer["detector"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    annual = annual_tables(inputs.panel, spec, n_boot, seed)
    timer["tabla_anual"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    ys = year_segment_tables(inputs.panel, inputs.segment_by_column, inputs.segment_labels, spec, n_boot, seed)
    timer["tabla_anio_segmento"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    win = window_tables(inputs.panel, ps, spec, n_boot, seed)
    timer["tabla_ventanas"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    rec = reconcile(inputs.panel, annual, inputs.segment_by_column, inherited_dir, inputs.year_check, inputs.year_ny_by_cell)
    timer["reconciliacion"] = time.perf_counter() - t0
    reject_manual_stop14_checks(cfg)
    checks = None
    if thresholds is None:
        dec, status = None, STATUS_NO_CALIBRATION
        stop = {"alcance": STATUS_NO_CALIBRATION, "definitivo": False}
    else:
        # Reglas A y B ejecutadas sobre ESTE analisis (mismos umbrales congelados): unica fuente de las comprobaciones.
        t0 = time.perf_counter()
        checks = stop14_rules.compute_stop14_checks(inputs.panel, ps, annual, thresholds)
        timer["stop14_reglas_A_B"] = time.perf_counter() - t0
        dec = t14.final_decisions(ps, thresholds, validation, checks.annual, checks.sensitivity)
        status, stop = dec["estado"], dec["stop14"]
    dates = dates_table(dec, inputs.panel)
    published = bool(dec and dec["fechas"])
    events = events_table(cfg.raw["eventos"], inputs.panel, dates, published)
    th30, th30s = th30_tables(inputs.panel) if inputs.panel.close is not None else (pd.DataFrame(), pd.DataFrame())
    if validation is not None:
        g2, g2s = validation.table, validation.specs_table if validation.specs_table is not None else pd.DataFrame()
    else:
        g2 = pd.DataFrame([{"control": "-", "decision": "-", "estado": "CALIBRACION_NO_EJECUTADA"}])
        g2s = pd.DataFrame()
    try:
        crit = acceptance_criteria_table(build_manifest(cfg))
    except TDA14ConfigError as exc:
        crit = pd.DataFrame([{"control": "-", "estado": f"MANIFIESTO_NO_CONSTRUIBLE: {exc}"}])
    checks_status = f"EJECUTADAS ({stop14_rules.RULES_VERSION})" if checks is not None else "NO_EJECUTADAS_SIN_UMBRALES"
    trace = traceability_table(cfg, spec, thresholds, validation, inputs, {**(extra_trace or {}), "stop14_comprobaciones": checks_status,
                                                                          "stop14_jornadas_excluidas_sensibilidad": checks.excluded if checks else {}})
    return TDA14Result(ps, annual, kurtosis_evaluability_by_block(annual), ys, win, t14.volume_descriptive(ps), rec, dec, th29_table(dec),
                       statistic_verdicts_table(dec), dates, events, clock_window_coverage(inputs.panel, cfg.raw["eventos"]), th30, th30s,
                       stop, stop14_table(dec), g2, g2s, crit, trace, status, inputs.year_check, timer,
                       checks.annual_evidence if checks else pd.DataFrame(), checks.sensitivity_evidence if checks else pd.DataFrame(), checks_status)


def file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def traceability_table(cfg: TDA14Config, spec: t14.DetectorSpec, thr: t14.FrozenThresholds | None, val: t14.ValidationResult | None,
                       inputs: TDA14Inputs, extra: dict | None = None) -> pd.DataFrame:
    rows = [("configuracion_tda14", cfg.path, "archivo"), ("configuracion_tda14_sha256", cfg.sha256, "archivo"),
            ("protocolo_estado", cfg.protocol.get("estado"), "configuracion"),
            ("modulo_tda14_sha256", file_sha256(t14.__file__), "codigo"), ("modulo_pipeline_sha256", file_sha256(__file__), "codigo"),
            ("detector_sha256", t14.detector_spec_sha256(spec), "configuracion"), ("fuente_datos", inputs.source, "carga"),
            ("jornadas_panel", inputs.panel.n_sessions, "carga"), ("anio_jornada_coincide_con_anio_ny", inputs.year_check.get("coinciden"), "carga"),
            ("objetivo_calibracion_g2", t14.G2_CALIBRATION_RATE, "codigo"), ("tasa_a_demostrar_g2", t14.G2_TARGET_RATE, "codigo"),
            ("confianza_g2", t14.G2_CONFIDENCE, "codigo"), ("capacidad_minima_tipo_provisional", t14.TYPE_CAPACITY_MIN, "codigo"),
            ("bloques_bootstrap", list(t14.ANNUAL_BLOCK_SESSIONS), "codigo"), ("n_boot", cfg.raw["analisis"]["n_boot"], "configuracion"),
            ("semilla", cfg.raw["analisis"]["semilla"], "configuracion")]
    rows += [(f"detector.{k}", v, "configuracion") for k, v in asdict(spec).items()]
    rows += [(f"materialidad.{s}", m, "codigo (convencion documentada)") for s, (_, m) in t14.DETECTOR_STATISTICS.items()]
    if thr is not None:
        rows += [("umbrales_sha256", thr.sha256, "calibracion"), ("alpha", thr.alpha, "calibracion"), ("k_star", thr.k_star, "calibracion"),
                 ("margen_tipo", thr.type_margin, "calibracion")]
        rows += [(f"c_fecha.{s}", c, "calibracion") for s, c in thr.c_date] + [(f"c_veredicto.{s}", c, "calibracion") for s, c in thr.c_verdict]
        rows += [(f"calibracion.{k}", v, "calibracion") for k, v in thr.calibration_info]
    else:
        rows.append(("umbrales_sha256", STATUS_NO_CALIBRATION, "calibracion"))
    rows.append(("manifiesto_validacion_sha256", val.manifest_sha256 if val else STATUS_NO_CALIBRATION, "validacion"))
    rows += [(k, v, "ejecucion") for k, v in (extra or {}).items()]
    return pd.DataFrame([{"parametro": k, "valor": json.dumps(v, default=_json_default) if isinstance(v, (list, dict)) else v, "origen": o} for k, v, o in rows])
