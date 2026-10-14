"""TDA-14 -- reglas A y B de STOP-14 (aprobadas en la preparacion del protocolo), ejecutadas por el pipeline.

Las comprobaciones de STOP-14 que ``decide_stop14`` necesita (confirmacion anual; sensibilidad a rolls y huecos) se
CALCULAN aqui a partir del analisis realmente ejecutado; el pipeline ya no acepta valores escritos a mano en la
configuracion. Un valor ``None`` significa "evidencia insuficiente o mixta" y deja STOP-14 PENDIENTE: nunca cuenta como
comprobacion superada.

REGLA A -- confirmacion anual (todas las familias del detector; años completos 2020-2024)
  Para cada estadistico de la familia:
    - intervalo de cada año = UNION de los IC 95 % de las longitudes de bloque 1, 5 y 20 (el mas ancho);
    - evaluable si al menos 4 de los 5 años son evaluables en las tres longitudes de bloque;
    - cambio anual si dos años tienen intervalos DISJUNTOS y |diferencia de valores puntuales| >= materialidad;
    - direccion anual = signo(media de los dos ultimos años evaluables - media de los dos primeros); direccion de
      ventanas = signo(media de las 4 ultimas ventanas sin solapamiento evaluables - media de las 4 primeras).
  Familia: True si algun estadistico muestra cambio anual (en la misma direccion que las ventanas si la familia es
  candidata en ventanas) y NINGUNO lo muestra en direccion contraria; False si todos son evaluables y ninguno lo muestra,
  o si algun cambio anual va en direccion contraria a sus ventanas, aunque otro estadistico confirme (el pipeline lo marca
  CONTRADICTORIA: revision humana); None en otro caso. En una familia candidata, una direccion nula (anual o de ventanas)
  nunca confirma ni contradice: ese cambio queda con direccion indeterminada (None si no hay otra evidencia).

REGLA B -- sensibilidad a rolls y huecos (mismos umbrales congelados)
  S1: sin la jornada anterior y las dos posteriores a cada cambio de contrato;
  S2: sin las jornadas con algun hueco interior de MAS de 15 minutos (la pausa programada de 15 minutos de las 16:15 NY,
      vigente hasta el 28-06-2021, es un horario institucional y no cuenta);
  S3: rejilla alineada con los contratos (el panel empieza en el primer cambio de contrato).
  Familia candidata a persistencia: True (explicada) si en >= 2 de 3 pasa a ESTABLE; False si en las 3 sigue no estable
  con soporte persistente; None en otro caso. Familia no candidata: False (no hay inestabilidad que explicar).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ohlcv_dataroad.ingest import tda14_temporal_stability as t14

RULES_VERSION = "STOP14_REGLAS_A_B_v1"
YEARS = (2020, 2021, 2022, 2023, 2024)
BLOCKS = (1, 5, 20)
MIN_EVALUABLE_YEARS = 4
EDGE_WINDOWS = 4
ROLL_BEFORE, ROLL_AFTER = 1, 2
GAP_LONGER_THAN_MINUTES = 15
SENSITIVITIES = ("S1_rolls", "S2_huecos", "S3_rejilla_alineada")
EXPLAINED_MIN = 2


@dataclass
class Stop14Checks:
    annual: dict
    sensitivity: dict
    annual_evidence: pd.DataFrame
    sensitivity_evidence: pd.DataFrame
    excluded: dict
    version: str = RULES_VERSION
    executed: bool = True
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Regla A
# ---------------------------------------------------------------------------

def window_direction(ps: t14.PanelStats, stat: str) -> float:
    v, _ = t14.nonoverlapping(ps, stat)
    v = v[np.isfinite(v)]
    if v.size < 2 * EDGE_WINDOWS:
        return 0.0
    return float(np.sign(v[-EDGE_WINDOWS:].mean() - v[:EDGE_WINDOWS].mean()))


def annual_statistic_status(annual: pd.DataFrame, stat: str) -> dict:
    sub = annual.loc[(annual["estadistico"] == stat) & (annual["anio"].isin(YEARS))]
    per_year = {}
    for y, g in sub.groupby("anio"):
        if set(g["bloque_jornadas"]) != set(BLOCKS) or not g["evaluable"].all():
            continue
        per_year[int(y)] = (float(g["valor"].iloc[0]), float(g["ic95_inf"].min()), float(g["ic95_sup"].max()))
    m = t14.DETECTOR_STATISTICS[stat][1]
    change = False
    ys = sorted(per_year)
    for i, y1 in enumerate(ys):
        for y2 in ys[i + 1:]:
            v1, lo1, hi1 = per_year[y1]
            v2, lo2, hi2 = per_year[y2]
            if (hi1 < lo2 or hi2 < lo1) and abs(v2 - v1) >= m:
                change = True
    direction = 0.0
    if len(ys) >= MIN_EVALUABLE_YEARS:
        direction = float(np.sign(np.mean([per_year[y][0] for y in ys[-2:]]) - np.mean([per_year[y][0] for y in ys[:2]])))
    return {"evaluable": len(ys) >= MIN_EVALUABLE_YEARS, "cambio_anual": change, "direccion_anual": direction, "anios_evaluables": len(ys)}


AN_CONFIRMS, AN_OPPOSITE, AN_UNDETERMINED, AN_NO_CHANGE = "CONFIRMA", "DIRECCION_CONTRARIA", "DIRECCION_INDETERMINADA", "SIN_CAMBIO"


def _annual_class(status: dict, wdir: float, candidate: bool) -> str:
    if not status["cambio_anual"]:
        return AN_NO_CHANGE
    if not candidate:
        return AN_CONFIRMS
    da = status["direccion_anual"]
    if da == 0 or wdir == 0:                     # una direccion nula nunca prueba que dos cambios vayan en la misma direccion
        return AN_UNDETERMINED
    return AN_CONFIRMS if da == wdir else AN_OPPOSITE


def annual_confirmation(annual: pd.DataFrame, ps: t14.PanelStats, family_verdicts: dict) -> tuple[dict, pd.DataFrame]:
    out, rows = {}, []
    for fam in t14.DETECTOR_FAMILIES:
        stats = [s for s, (f, _) in t14.DETECTOR_STATISTICS.items() if f == fam]
        candidate = family_verdicts.get(fam, {}).get("veredicto") in t14.NONSTABLE_VERDICTS
        st = {s: annual_statistic_status(annual, s) for s in stats}
        wd = {s: window_direction(ps, s) for s in stats}
        cls = {s: _annual_class(st[s], wd[s], candidate) for s in stats}
        consistent = [s for s in stats if cls[s] == AN_CONFIRMS]
        opposite = [s for s in stats if cls[s] == AN_OPPOSITE]
        if consistent and opposite:
            out[fam], why = False, "CONTRADICTORIA_ENTRE_ESTADISTICOS"     # nunca True: revision humana
        elif consistent:
            out[fam], why = True, "CONFIRMADA"
        elif opposite:
            out[fam], why = False, "CONTRADICTORIA_DIRECCION"
        elif any(cls[s] == AN_UNDETERMINED for s in stats):
            out[fam], why = None, "DIRECCION_INDETERMINADA"               # direccion 0: no confirma ni contradice
        elif all(st[s]["evaluable"] for s in stats):
            out[fam], why = False, "SIN_CAMBIO_ANUAL"
        else:
            out[fam], why = None, "INSUFICIENTE"
        for s in stats:
            rows.append({"familia": fam, "estadistico": s, "candidata_en_ventanas": candidate, **st[s], "direccion_ventanas": wd[s],
                         "clasificacion_estadistico": cls[s], "motivo_familia": why, "resultado_familia": out[fam]})
    return out, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Regla B
# ---------------------------------------------------------------------------

def _drop_sessions(panel: t14.SessionPanel, drop: np.ndarray) -> t14.SessionPanel:
    r = panel.r.copy()
    pres = panel.present.copy()
    r[drop] = np.nan
    pres[drop] = False
    vol = None if panel.volume is None else np.where(pres, panel.volume, np.nan)
    return t14.SessionPanel(r=r, present=pres, session_date=panel.session_date, contract=panel.contract, volume=vol, close=panel.close)


def longest_interior_gap(panel: t14.SessionPanel) -> np.ndarray:
    out = np.zeros(panel.n_sessions, dtype=int)
    for i in range(panel.n_sessions):
        cols = np.flatnonzero(panel.present[i])
        run = 0
        for v in (~panel.present[i, cols[0]: cols[-1] + 1]) if cols.size else ():
            run = run + 1 if v else 0
            out[i] = max(out[i], run)
    return out


def sensitivity_panels(panel: t14.SessionPanel) -> tuple[dict, dict]:
    s = panel.n_sessions
    switch = np.flatnonzero(panel.contract[1:] != panel.contract[:-1]) + 1
    s1 = np.zeros(s, dtype=bool)
    for k in switch:
        s1[max(k - ROLL_BEFORE, 0): min(k + ROLL_AFTER + 1, s)] = True
    s2 = longest_interior_gap(panel) > GAP_LONGER_THAN_MINUTES
    start = int(switch[0]) if switch.size else 0
    s3 = t14.SessionPanel(r=panel.r[start:], present=panel.present[start:], session_date=panel.session_date[start:],
                          contract=panel.contract[start:], volume=None if panel.volume is None else panel.volume[start:],
                          close=None if panel.close is None else panel.close[start:])
    pans = {"S1_rolls": _drop_sessions(panel, s1), "S2_huecos": _drop_sessions(panel, s2), "S3_rejilla_alineada": s3}
    return pans, {"S1_rolls": int(s1.sum()), "S2_huecos": int(s2.sum()), "S3_rejilla_alineada": start}


def sensitivity_explanation(base_families: dict, base_strict: dict, sens_families: dict[str, dict]) -> tuple[dict, pd.DataFrame]:
    out, rows = {}, []
    for fam in t14.DETECTOR_FAMILIES:
        cand = t14.persistence_status(base_families[fam], base_strict[fam], True, False) == t14.PS_CONFIRMED
        res = {k: d[fam] for k, d in sens_families.items()}
        if not cand:
            out[fam] = False
        else:
            stable = sum(v["veredicto"] == t14.V_STABLE for v in res.values())
            persistent = sum(v["veredicto"] in t14.NONSTABLE_VERDICTS and t14.support_of(v) in t14.PERSISTENT_SUPPORTS for v in res.values())
            out[fam] = True if stable >= EXPLAINED_MIN else (False if persistent == len(res) else None)
        for k, v in res.items():
            rows.append({"familia": fam, "candidata_en_ventanas": cand, "sensibilidad": k, "veredicto": v["veredicto"],
                         "anotacion": v.get("anotacion", ""), "soporte": t14.support_of(v), "resultado_familia": out[fam]})
    return out, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ejecucion completa
# ---------------------------------------------------------------------------

def compute_stop14_checks(panel: t14.SessionPanel, ps: t14.PanelStats, annual: pd.DataFrame, thr: t14.FrozenThresholds) -> Stop14Checks:
    """Ejecuta las reglas A y B con los umbrales congelados. Solo esta funcion produce las entradas de STOP-14."""
    t14.verify_frozen(thr)
    base = t14.apply_rule(ps, thr)
    annual_res, annual_ev = annual_confirmation(annual, ps, base["veredictos_familia"])
    spec = t14._spec_from(thr)
    pans, excluded = sensitivity_panels(panel)
    sens, notes = {}, []
    for k, p in pans.items():
        try:
            sens[k] = t14.apply_rule(t14.analyze_panel(p, spec), thr)["veredictos_familia"]
        except ValueError as exc:            # panel sin jornadas suficientes: NO_EVALUABLE (nunca "superada")
            sens[k] = {f: {"veredicto": t14.V_NOT_EVALUABLE, "anotacion": t14.A_FEW_WINDOWS} for f in t14.DETECTOR_FAMILIES}
            notes.append(f"{k}: {exc}")
    sens_res, sens_ev = sensitivity_explanation(base["veredictos_familia"], base["veredictos_familia_estricta"], sens)
    return Stop14Checks(annual_res, sens_res, annual_ev, sens_ev, excluded, notes=notes)
