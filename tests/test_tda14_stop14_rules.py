"""Tests de las reglas A y B de STOP-14 (``tda14_stop14_rules``). SOLO datos sinteticos."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ohlcv_dataroad.ingest import tda14_stop14_rules as rules
from ohlcv_dataroad.ingest import tda14_temporal_stability as t14

J = {"veredicto": t14.V_JUMP, "anotacion": "", "soporte": t14.SUP_PERMANENT}
S = {"veredicto": t14.V_STABLE, "anotacion": t14.A_NO_CHANGE}
NE = {"veredicto": t14.V_NOT_EVALUABLE, "anotacion": t14.A_FEW_WINDOWS}


def fams(**over):
    return {f: dict(S) for f in t14.DETECTOR_FAMILIES} | over


class _PS:
    """Ventanas sin solapamiento minimas para la direccion de las ventanas."""
    def __init__(self, **series):
        n = 20
        self.values = {s: np.zeros(n) for s in t14.DETECTOR_STATISTICS} | series
        self.se = {s: np.full(n, 0.01) for s in t14.DETECTOR_STATISTICS}
        self.window_start = np.arange(n)
        self.offset = 1


UP = np.r_[np.zeros(10), np.full(10, 0.5)]


def annual(values: dict, half=0.01, not_evaluable=()):
    rows = []
    for st in t14.DETECTOR_STATISTICS:
        vals = values.get(st, [0.0] * 5)
        for y, v in zip(rules.YEARS, vals):
            for b in rules.BLOCKS:
                rows.append({"anio": y, "estadistico": st, "bloque_jornadas": b, "valor": v, "ic95_inf": v - half * b / 5,
                             "ic95_sup": v + half * b / 5, "evaluable": (st, y) not in not_evaluable})
    return pd.DataFrame(rows)


def test_rule_A_sufficient_insufficient_and_contradictory():
    ps = _PS(log_sigma=UP)
    f = fams(ESCALA=J)
    res, ev = rules.annual_confirmation(annual({"log_sigma": [0, 0, 0.5, 0.5, 0.5]}), ps, f)
    assert res[t14.FAM_SCALE] is True and res[t14.FAM_CLOCK] is False
    assert set(ev.columns) >= {"evaluable", "cambio_anual", "direccion_anual", "direccion_ventanas", "resultado_familia"}
    assert rules.annual_confirmation(annual({"log_sigma": [0.5, 0.5, 0, 0, 0]}), ps, f)[0][t14.FAM_SCALE] is False      # direccion contraria
    assert rules.annual_confirmation(annual({"log_sigma": [0, 0, 0.1, 0.1, 0.1]}), ps, f)[0][t14.FAM_SCALE] is False    # no material
    assert rules.annual_confirmation(annual({"log_sigma": [0, 0, 0.5, 0.5, 0.5]}, half=0.4), ps, f)[0][t14.FAM_SCALE] is False   # solapados
    ne = {("log_sigma", 2023), ("log_sigma", 2024)}
    assert rules.annual_confirmation(annual({}, not_evaluable=ne), ps, f)[0][t14.FAM_SCALE] is None                     # insuficiente
    assert rules.annual_confirmation(annual({"rho1": [0, 0, 0.2, 0.2, 0.2]}), _PS(), fams())[0][t14.FAM_LINEAR] is True


def test_rule_A_mixed_directions_between_statistics_is_contradictory_never_true():
    """FORMA (q01 y q99): uno confirma y el otro cambia en direccion contraria a sus ventanas -> False (CONTRADICTORIA)."""
    ps = _PS(log_q01_en_sigma=UP, log_q99_en_sigma=UP)
    f = fams(FORMA=J)
    up, down = [0, 0, 0.5, 0.5, 0.5], [0.5, 0.5, 0, 0, 0]
    res, ev = rules.annual_confirmation(annual({"log_q01_en_sigma": up, "log_q99_en_sigma": down}), ps, f)
    assert res[t14.FAM_SHAPE] is False
    e = ev.loc[ev["familia"] == t14.FAM_SHAPE].set_index("estadistico")
    assert e.loc["log_q01_en_sigma", "clasificacion_estadistico"] == rules.AN_CONFIRMS
    assert e.loc["log_q99_en_sigma", "clasificacion_estadistico"] == rules.AN_OPPOSITE
    assert set(e["motivo_familia"]) == {"CONTRADICTORIA_ENTRE_ESTADISTICOS"}
    # en STOP-14 la familia candidata queda CONTRADICTORIA (revision humana), nunca confirmada
    assert t14.persistence_status(J, J, res[t14.FAM_SHAPE], False) == t14.PS_CONTRADICTORY
    # control: si el segundo estadistico no cambia, el primero si confirma
    assert rules.annual_confirmation(annual({"log_q01_en_sigma": up}), ps, f)[0][t14.FAM_SHAPE] is True


def test_rule_A_zero_direction_is_never_confirmation():
    f = fams(ESCALA=J)
    jump = [0, 0, 0.5, 0.5, 0.5]
    # direccion de ventanas nula (ventanas planas) frente a cambio anual creciente: no confirma
    res, ev = rules.annual_confirmation(annual({"log_sigma": jump}), _PS(), f)
    assert res[t14.FAM_SCALE] is None
    assert set(ev.loc[ev["estadistico"] == "log_sigma", "clasificacion_estadistico"]) == {rules.AN_UNDETERMINED}
    # direccion anual nula (cambio material entre años, mismo nivel al principio y al final) con ventanas crecientes
    bump = [0, 0.5, 0.5, 0.5, 0]
    st = rules.annual_statistic_status(annual({"log_sigma": bump}), "log_sigma")
    assert st["cambio_anual"] and st["direccion_anual"] == 0
    assert rules.annual_confirmation(annual({"log_sigma": bump}), _PS(log_sigma=UP), f)[0][t14.FAM_SCALE] is None
    # ambas nulas: 0 == 0 no prueba la misma direccion
    assert rules.annual_confirmation(annual({"log_sigma": bump}), _PS(), f)[0][t14.FAM_SCALE] is None
    # una direccion nula no anula la contradiccion de otro estadistico
    ps = _PS(log_q01_en_sigma=UP, log_q99_en_sigma=UP)
    out = rules.annual_confirmation(annual({"log_q01_en_sigma": bump, "log_q99_en_sigma": [0.5, 0.5, 0, 0, 0]}), ps, fams(FORMA=J))[0]
    assert out[t14.FAM_SHAPE] is False


def test_rule_A_missing_block_length_is_not_evaluable():
    a = annual({})
    a = a.loc[~((a["estadistico"] == "log_sigma") & (a["bloque_jornadas"] == 20))]
    assert rules.annual_statistic_status(a, "log_sigma")["evaluable"] is False


def test_rule_B_outcomes():
    f = fams(ESCALA=J)
    stable_all = {k: fams() for k in rules.SENSITIVITIES}
    assert rules.sensitivity_explanation(f, f, stable_all)[0][t14.FAM_SCALE] is True
    two = dict(stable_all) | {"S3_rejilla_alineada": dict(f)}
    assert rules.sensitivity_explanation(f, f, two)[0][t14.FAM_SCALE] is True
    keep = {k: dict(f) for k in rules.SENSITIVITIES}
    assert rules.sensitivity_explanation(f, f, keep)[0][t14.FAM_SCALE] is False
    mixed = {"S1_rolls": fams(), "S2_huecos": dict(f), "S3_rejilla_alineada": fams(ESCALA=NE)}
    out, ev = rules.sensitivity_explanation(f, f, mixed)
    assert out[t14.FAM_SCALE] is None and out[t14.FAM_CLOCK] is False
    assert len(ev) == 3 * len(t14.DETECTOR_FAMILIES)


def _panel_with_gaps():
    spec = t14.WorldSpec(label="REGLAS", n_sessions=180, width=240, active_minutes=240, roll_every=45, gap_probability=0.0,
                         daily_phi=0.8, daily_sd=0.05)
    p = t14.simulate_world(spec, t14.world_rng(t14.ROLE_TEST, "REGLAS", 0))
    p.present[10, 100:115] = False          # 15 minutos (como la pausa programada): NO se excluye
    p.r[10, 100:116] = np.nan
    p.present[20, 50:66] = False            # 16 minutos: se excluye
    p.r[20, 50:67] = np.nan
    return p


def test_rule_B_panels_and_scheduled_15_minute_pause():
    p = _panel_with_gaps()
    gaps = rules.longest_interior_gap(p)
    assert gaps[10] == 15 and gaps[20] == 16
    pans, excl = rules.sensitivity_panels(p)
    assert excl["S2_huecos"] == 1 and not pans["S2_huecos"].present[20].any() and pans["S2_huecos"].present[10].any()
    switch = np.flatnonzero(p.contract[1:] != p.contract[:-1]) + 1
    assert excl["S1_rolls"] == len(switch) * (rules.ROLL_BEFORE + rules.ROLL_AFTER + 1)
    assert pans["S3_rejilla_alineada"].n_sessions == p.n_sessions - int(switch[0]) == p.n_sessions - excl["S3_rejilla_alineada"]


def test_compute_checks_requires_frozen_thresholds_and_marks_unrunnable_sensitivity_not_evaluable():
    import pytest
    from dataclasses import asdict
    p = _panel_with_gaps()
    spec = t14.DetectorSpec(batch_sessions=3, window_batches=3, step_batches=1, min_side_windows=4, min_evaluable_windows=12,
                            min_profile_bin_bars=20, max_lag=10, profile_bin_minutes=30)
    st = tuple((s, 1e9) for s in t14.DETECTOR_STATISTICS)
    thr = t14.freeze_thresholds(t14.FrozenThresholds(alpha=0.01, k_star=2, c_date=st, c_verdict=st, type_margin=2.0,
                                                     materiality=tuple((s, m) for s, (_, m) in t14.DETECTOR_STATISTICS.items()),
                                                     detector_spec=tuple(sorted(asdict(spec).items())), calibration_info=()))
    ps = t14.analyze_panel(p, spec)
    ann = pd.DataFrame(columns=["anio", "estadistico", "bloque_jornadas", "valor", "ic95_inf", "ic95_sup", "evaluable"])
    chk = rules.compute_stop14_checks(p, ps, ann, thr)
    assert chk.version == rules.RULES_VERSION and chk.executed
    assert all(v is None for v in chk.annual.values())                       # sin tabla anual: insuficiente, nunca True
    assert all(v is False for v in chk.sensitivity.values())                 # umbrales enormes: ninguna familia candidata
    with pytest.raises(t14.FrozenThresholdsError):
        rules.compute_stop14_checks(p, ps, ann, t14.FrozenThresholds(**{**thr.payload(), "k_star": 3, "sha256": thr.sha256}))
    short = t14.SessionPanel(r=p.r[:40], present=p.present[:40], session_date=p.session_date[:40], contract=np.array(["C000"] * 40, dtype=object),
                             volume=None, close=None)
    short.contract[38:] = "C001"          # S3 queda con 2 jornadas: menos que una ventana
    chk2 = rules.compute_stop14_checks(short, t14.analyze_panel(short, spec), ann, thr)
    assert chk2.notes and all(v is False for v in chk2.sensitivity.values())
