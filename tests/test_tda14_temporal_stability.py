"""Tests de TDA-14 (estabilidad temporal). SOLO datos sinteticos: ningun test lee data/ ni el hold-out."""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ohlcv_dataroad.ingest import tda14_temporal_stability as t14
from ohlcv_dataroad.ingest.holdout_guard import HoldoutIsolationError

# Mundos y detector reducidos (rapidos): jornadas de 240 minutos, lotes de 3 jornadas, ventanas de 9, paso 3.
SPEC = t14.DetectorSpec(batch_sessions=3, window_batches=3, step_batches=1, min_side_windows=4, min_evaluable_windows=12,
                        min_profile_bin_bars=20, max_lag=10, profile_bin_minutes=30, min_tail_support=20)
# Persistencia diaria reducida respecto del valor por defecto: estos tests prueban la MECANICA; el efecto de la
# persistencia sobre los umbrales se prueba aparte (test_higher_persistence_raises_calibrated_thresholds).
WORLD = t14.WorldSpec(n_sessions=180, width=240, active_minutes=240, roll_every=45, gap_probability=0.05, daily_phi=0.8, daily_sd=0.05)


def world(label="NEGATIVO", rep=0, role=t14.ROLE_TEST, **kw):
    spec = replace(WORLD, label=label, **kw)
    return t14.simulate_world(spec, t14.world_rng(role, label, rep))


def stats_of(panel):
    return t14.analyze_panel(panel, SPEC)


def tiny_panel():
    """2 jornadas x 8 minutos con un hueco: valores conocidos."""
    r = np.array([[np.nan, 0.1, -0.2, 0.3, np.nan, np.nan, 0.1, 0.0],
                  [np.nan, 0.2, 0.0, -0.1, 0.4, -0.3, np.nan, np.nan]])
    present = np.array([[1, 1, 1, 1, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0]], dtype=bool)
    return t14.SessionPanel(r=r, present=present, session_date=np.array(["2021-01-04", "2021-01-05"], dtype="datetime64[D]"),
                            contract=np.array(["A", "A"], dtype=object))


# ===========================================================================
# Etapa 1: estimadores, mascaras, ventanas
# ===========================================================================

def test_session_minute_starts_at_ny_open_and_keeps_time_order():
    assert t14.session_minute([18 * 60, 23 * 60 + 59, 0, 16 * 60 + 59]).tolist() == [0, 359, 360, 1379]


def test_continuity_runs_never_join_across_gap_or_session():
    p = tiny_panel()
    valid = np.isfinite(p.r)
    runs = t14.continuity_runs(valid)
    assert runs[0, 3] != runs[0, 6] and runs[0, 4] == -1 and runs[0, 5] == -1
    assert runs[0, 7] != runs[1, 1]
    pm = t14.pair_mask(valid, runs, 1)
    # pares validos de la jornada 0: (1,2), (2,3), (6,7); el par (5,6) no: el retorno 5 no es valido (primera barra tras el hueco)
    assert np.flatnonzero(pm[0]).tolist() == [1, 2, 6]
    assert np.flatnonzero(pm[1]).tolist() == [1, 2, 3, 4]


def test_no_pair_crosses_an_invalid_return_even_if_the_bar_exists():
    """Regresion de auditoria: barras todas presentes, retorno invalido en el minuto 4. Ningun par de ningun retardo
    puede saltar sobre el (antes, con bloques de barras presentes, el par (3, 5) de retardo 2 se colaba)."""
    r = np.array([[np.nan, 0.1, -0.2, 0.3, np.nan, 0.2, -0.1, 0.4, 0.1]])
    present = np.ones_like(r, dtype=bool)
    valid = np.isfinite(r)
    runs = t14.continuity_runs(valid)
    for lag in range(1, 8):
        pm = t14.pair_mask(valid, runs, lag)
        for a in np.flatnonzero(pm[0]):
            assert not (a < 4 < a + lag), (lag, a)
    assert not t14.pair_mask(valid, runs, 2)[0, 3]
    p = t14.SessionPanel(r=r, present=present, session_date=np.array(["2021-01-04"], dtype="datetime64[D]"), contract=np.array(["A"], dtype=object))
    suff = t14.session_sufficient_stats(p, max_lag=3, bin_minutes=3)
    assert suff["lag_n"][0].tolist() == [5, 3, 1]               # lag 1: (1,2)(2,3)(5,6)(6,7)(7,8); lag 2: (1,3)(5,7)(6,8); lag 3: (5,8)


def test_pairs_match_inherited_tda09_block_definition():
    """Los pares de cada retardo coinciden con los que da ``compute_block_ids_with_contract`` (TDA-09) aplicado a la
    poblacion de retornos validos: mismo bloque y exactamente ``lag`` minutos de distancia."""
    from ohlcv_dataroad.ingest.tda09_volatility_clustering import compute_block_ids_with_contract
    p = world(n_sessions=12)
    p.r[3, 50] = np.nan                                        # retorno invalido con barra presente
    p.r[5, 100:103] = np.nan
    valid = np.isfinite(p.r)
    s_idx, c_idx = np.nonzero(valid)
    ts = pd.Timestamp("2021-01-04") + pd.to_timedelta(s_idx * 2880 + c_idx, unit="min")
    blocks = compute_block_ids_with_contract(pd.Series(ts), pd.Series(s_idx), pd.Series(p.contract[s_idx]), 60.0)
    minutes = s_idx * 2880 + c_idx
    runs = t14.continuity_runs(valid)
    for lag in (1, 2, 5, 10):
        inherited = 0
        for b in np.unique(blocks):
            m = minutes[blocks == b]
            inherited += int(np.isin(m + lag, m).sum())
        assert t14.pair_mask(valid, runs, lag).sum() == inherited, lag


def test_sufficient_stats_reproduce_direct_estimates():
    p = tiny_panel()
    suff = t14.session_sufficient_stats(p, max_lag=2, bin_minutes=4)
    v = p.r[np.isfinite(p.r)]
    sm = t14._sum_rows(suff, slice(None))
    st = t14.statistics_from_sums(sm, -0.25, 0.35, np.zeros(2), 1)
    assert st["log_sigma"] == pytest.approx(math.log(v.std()))
    assert st["media_en_sigma"] == pytest.approx(v.mean() / v.std())
    m = v.mean()
    pairs = [(0, 1, 2), (0, 2, 3), (0, 6, 7), (1, 1, 2), (1, 2, 3), (1, 3, 4), (1, 4, 5)]
    cov = np.mean([(p.r[i, a] - m) * (p.r[i, b] - m) for i, a, b in pairs])
    assert st["rho1"] == pytest.approx(cov / v.var())
    a = np.abs(v)
    am = a.mean()
    lag2 = [(0, 1, 3), (1, 1, 3), (1, 2, 4), (1, 3, 5)]
    c2 = np.mean([(abs(p.r[i, x]) - am) * (abs(p.r[i, y]) - am) for i, x, y in lag2])
    c1 = np.mean([(abs(p.r[i, x]) - am) * (abs(p.r[i, y]) - am) for i, x, y in pairs])
    assert st["persistencia_acf_abs"] == pytest.approx(np.mean([c1, c2]) / a.var())
    assert st["log_fraccion_cero"] == pytest.approx(math.log(2 / v.size))
    assert st["log_q01_en_sigma"] == pytest.approx(math.log(0.25 / v.std()))
    assert st["curtosis_total"] == pytest.approx(np.mean((v - m) ** 4) / v.var() ** 2)


def test_build_session_panel_layout_masks_and_contracts():
    ts = pd.to_datetime(["2021-01-04 23:00", "2021-01-04 23:01", "2021-01-04 23:02", "2021-01-05 23:00"])
    df = pd.DataFrame({"timestamp": ts, "trading_date": ["2021-01-05", "2021-01-05", "2021-01-05", "2021-01-06"],
                       "contract": ["MNQH1"] * 4, "minute_of_day": [18 * 60, 18 * 60 + 1, 18 * 60 + 2, 18 * 60],
                       "r_1m": [np.nan, 0.001, 0.5, np.nan], "r_1m_valid": [False, True, False, False], "volume": [5, 6, 7, 8], "close": [1.0, 2, 3, 4]})
    p = t14.build_session_panel(df, "2025-06-23")
    assert p.r.shape == (2, 1440) and p.present[0, :3].all() and p.present[1, 0]
    assert p.r[0, 1] == 0.001 and np.isnan(p.r[0, 2])          # un retorno invalido de TDA-04 no entra aunque tenga valor
    assert list(p.contract) == ["MNQH1", "MNQH1"] and p.volume[0, 2] == 7


def test_build_session_panel_refuses_any_bar_at_or_after_holdout():
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2025-06-23 00:00"]), "trading_date": ["2025-06-23"], "contract": ["X"],
                       "minute_of_day": [0], "r_1m": [0.0], "r_1m_valid": [True]})
    with pytest.raises(HoldoutIsolationError):
        t14.build_session_panel(df, "2025-06-23")


def test_build_session_panel_refuses_two_contracts_in_one_session():
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2021-01-04 23:00", "2021-01-04 23:01"]), "trading_date": ["2021-01-05"] * 2,
                       "contract": ["A", "B"], "minute_of_day": [1080, 1081], "r_1m": [np.nan, 0.0], "r_1m_valid": [False, True]})
    with pytest.raises(ValueError, match="mas de un contrato"):
        t14.build_session_panel(df, "2025-06-23")


def test_load_population_runs_holdout_guard_before_reading_anything(monkeypatch):
    calls = []

    def guard(_):
        calls.append("guard")
        raise HoldoutIsolationError("solapamiento")

    monkeypatch.setattr(t14, "validate_research_holdout_disjoint", guard)
    monkeypatch.setattr(t14, "load_tda04_inputs", lambda _: calls.append("read"))
    with pytest.raises(HoldoutIsolationError):
        t14.load_population(object())
    assert calls == ["guard"]


def test_window_grid_offsets_and_nonoverlapping_subset():
    ps = stats_of(world())
    assert ps.window_sessions == 9 and ps.offset == 3 and ps.n_sessions_used == 180
    assert ps.window_start.tolist() == list(range(0, 180 - 9 + 1, 3))
    v, _ = t14.nonoverlapping(ps, "log_sigma")
    assert len(v) == 20
    d, z, b = t14.boundary_series(ps, "log_sigma")
    assert len(d) == len(ps.window_start) - 3 and b[0] == 9


def test_window_without_coverage_is_not_evaluable():
    p = world()
    p.r[30:36] = np.nan                    # 6 jornadas sin retornos
    ps = stats_of(p)
    bad = [(s <= 35 and s + 9 > 30) for s in ps.window_start]
    assert not ps.evaluable[bad].any() and ps.evaluable[~np.array(bad)].all()
    assert np.isnan(ps.values["log_sigma"][bad]).all()


def test_simulated_world_is_reproducible_and_roles_use_different_streams():
    a, b = world(rep=3), world(rep=3)
    np.testing.assert_array_equal(a.r, b.r)
    c = world(rep=3, role=t14.ROLE_VALIDATION)
    assert not np.array_equal(np.nan_to_num(a.r), np.nan_to_num(c.r))


def test_simulated_world_structure():
    p = world()
    assert np.isnan(p.r[:, 0]).all()                                   # primera barra de cada jornada: sin retorno
    c = p.close[p.present]
    assert np.allclose(np.round(c / 0.25) * 0.25, c)                   # precios en ticks
    assert (p.r[np.isfinite(p.r)] == 0).mean() > 0                     # discrecion: hay barras sin cambio
    assert (~p.present).any() and len(set(p.contract)) == 4            # huecos y rolls


@pytest.mark.parametrize("family,mag,stat,direction", [
    (t14.FAM_SCALE, 2.0, "log_sigma", 1), (t14.FAM_SHAPE, 2.2, "log_q99_en_sigma", -1), (t14.FAM_LEVEL, 0.3, "media_en_sigma", 1),
    (t14.FAM_LINEAR, 0.4, "rho1", 1), (t14.FAM_MAGNITUDE, 4.0, "persistencia_acf_abs", 1), (t14.FAM_CLOCK, 2.0, "log_amplitud_perfil", 1),
    (t14.FAM_DISCRETE, 2.0, "log_fraccion_cero", -1)])
def test_each_injected_change_moves_its_own_statistic(family, mag, stat, direction):
    ch = t14.Change(family, t14.SHAPE_STEP, mag, start=0.5)
    ps = stats_of(world(label=f"POS_{family}", changes=(ch,)))
    v, _ = t14.nonoverlapping(ps, stat)
    assert direction * (np.nanmean(v[12:]) - np.nanmean(v[:8])) > 0


def test_scale_step_realized_ratio():
    p = world(label="ESCALA", changes=(t14.Change(t14.FAM_SCALE, t14.SHAPE_STEP, 2.0, start=0.5),))
    first, second = p.r[:90], p.r[90:]
    ratio = np.nanstd(second) / np.nanstd(first)
    assert 1.5 < ratio < 2.6


def test_volume_is_descriptive_only_and_outside_detector():
    ps = stats_of(world())
    assert t14.FAM_ACTIVITY not in t14.DETECTOR_FAMILIES
    assert "log_volumen_medio" not in ps.values and "log_volumen_medio" in ps.descriptive
    vd = t14.volume_descriptive(ps)
    assert set(vd["estado"]) == {t14.VOLUME_STATUS} and set(vd["veredicto_oficial"]) == {t14.VOLUME_VERDICT}
    assert "curtosis_total" not in ps.values                          # la curtosis tampoco entra en el detector


# ===========================================================================
# Etapa 2: clasificacion (series controladas)
# ===========================================================================

N = 20
SE = np.full(N, 0.1)


def classify(y, m=0.3, c=2.0, margin=2.0, **kw):
    return t14.classify_series(np.asarray(y, float), kw.pop("se", SE[: len(y)]), m, c, margin, 4, 12)


def noise(seed=0, n=N):
    return np.random.default_rng(seed).normal(0, 0.1, n)


def test_classify_stable_series():
    assert classify(noise())["veredicto"] == t14.V_STABLE


def test_classify_statistical_but_not_material_change_is_stable():
    y = noise() + np.r_[np.zeros(10), np.full(10, 0.4)]
    r = classify(y, m=0.5)
    assert r["veredicto"] == t14.V_STABLE and r["anotacion"] == t14.A_NOT_MATERIAL


def test_classify_permanent_step():
    r = classify(noise(1) + np.r_[np.zeros(10), np.ones(10)])
    assert (r["veredicto"], r["anotacion"]) == (t14.V_JUMP, "")


def test_classify_trend():
    r = classify(noise(2) + np.linspace(0, 1.5, N))
    assert r["veredicto"] == t14.V_TREND


def test_classify_transient_episode_with_levels_and_duration():
    r = classify(noise(3) + np.r_[np.zeros(7), np.ones(6), np.zeros(7)])
    assert (r["veredicto"], r["anotacion"]) == (t14.V_UNSTABLE, t14.A_EPISODE)
    assert r["episodio_ventanas"] == 6 and abs(r["niveles"][0] - r["niveles"][2]) < 0.3


def test_classify_change_at_edge_cannot_be_called_permanent():
    r = classify(noise(4) + np.r_[np.zeros(18), np.ones(2)])
    assert (r["veredicto"], r["anotacion"]) == (t14.V_UNSTABLE, t14.A_EDGE)
    assert r["soporte"] == t14.SUP_EDGE and r["borde"] == "FIN" and r["ventanas_tras_el_cambio"] == 2
    r = classify(noise(4) + np.r_[np.ones(2), np.zeros(18)])
    assert r["anotacion"] == t14.A_EDGE and r["borde"] == "INICIO"


def test_classify_without_pattern():
    y = noise(5) + np.tile([0, 0, 0, 0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0], 2)
    r = classify(y)
    assert (r["veredicto"], r["anotacion"]) == (t14.V_UNSTABLE, t14.A_NO_PATTERN)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("block", [1, 2, 3, 5])
def test_classify_repeated_alternation_is_never_stable(seed, block):
    """Regresion de auditoria: alternancias repetidas entre los niveles 0 y 1 (bloques de 1, 2, 3 o 5 ventanas),
    materialidad 0,3. La curva ajustada puede ser casi plana, pero la variacion observada y el residuo son grandes:
    nunca ESTABLE. Mismo umbral c=2 que el resto de tests (no se sube ningun umbral)."""
    pattern = np.tile(np.r_[np.zeros(block), np.ones(block)], N)[:N]
    r = classify(noise(seed) + pattern, m=0.3)
    assert r["veredicto"] == t14.V_UNSTABLE, r
    assert r["variacion_observada"] >= 0.3


@pytest.mark.parametrize("seed", range(10))
def test_small_alternation_below_materiality_stays_stable(seed):
    """Contrapartida: la misma alternancia con amplitud 0,1 (< materialidad 0,3) sigue siendo ESTABLE: la correccion no
    convierte en inestable cualquier oscilacion."""
    pattern = np.tile([0.0, 0.1], N // 2)
    r = classify(noise(seed) * 0.2 + pattern, m=0.3, se=np.full(N, 0.02))
    assert r["veredicto"] == t14.V_STABLE, r


def test_observed_variation_equals_jump_for_half_half_step():
    y = np.r_[np.zeros(10), np.ones(10)]
    assert t14.observed_variation(y, np.zeros(20)) == pytest.approx(2 * np.sqrt(np.var(y, ddof=1)))
    assert t14.observed_variation(y, np.zeros(20)) == pytest.approx(1.0, abs=0.06)


def test_classify_undistinguishable_type_keeps_uncertainty():
    y = noise(6) + np.r_[np.zeros(10), np.ones(10)]
    r = classify(y, margin=1e9)          # ningun tipo puede separarse de los demas
    assert (r["veredicto"], r["anotacion"]) == (t14.V_UNSTABLE, t14.A_TYPE_UNKNOWN)
    assert r["soporte"] == t14.SUP_UNPROVEN                  # entre las alternativas hay un borde: permanencia no demostrada
    assert r["veredicto"] != t14.V_TREND


def test_classify_not_evaluable_is_separate_state():
    assert classify(noise()[:10])["veredicto"] == t14.V_NOT_EVALUABLE
    r = classify(noise(), m=0.05, se=np.full(N, 0.5))
    assert (r["veredicto"], r["anotacion"]) == (t14.V_NOT_EVALUABLE, t14.A_WIDE_CI)
    assert t14.V_NOT_EVALUABLE not in t14.OFFICIAL_VERDICTS
    assert t14.OFFICIAL_VERDICTS == ("ESTABLE", "TENDENCIA", "CAMBIO_ABRUPTO", "INESTABLE")


def test_combine_family_rules():
    st = {"veredicto": t14.V_STABLE, "anotacion": t14.A_NO_CHANGE}
    jump = {"veredicto": t14.V_JUMP, "anotacion": ""}
    trend = {"veredicto": t14.V_TREND, "anotacion": ""}
    ne = {"veredicto": t14.V_NOT_EVALUABLE, "anotacion": t14.A_FEW_WINDOWS}
    assert t14.combine_family([st, jump])["veredicto"] == t14.V_JUMP
    assert t14.combine_family([jump, trend]) == {"veredicto": t14.V_UNSTABLE, "anotacion": t14.A_TYPE_UNKNOWN, "soporte": t14.SUP_PERMANENT}
    assert t14.combine_family([st, ne])["veredicto"] == t14.V_NOT_EVALUABLE
    assert t14.combine_family([st, st])["veredicto"] == t14.V_STABLE


# ===========================================================================
# STOP-14
# ===========================================================================

S_ = {"veredicto": t14.V_STABLE, "anotacion": t14.A_NO_CHANGE}
J_ = {"veredicto": t14.V_JUMP, "anotacion": ""}
E_ = {"veredicto": t14.V_UNSTABLE, "anotacion": t14.A_EPISODE}


def fams(**over):
    d = {f: dict(S_) for f in t14.DETECTOR_FAMILIES}
    d.update(over)
    return d


ALL_NO = dict.fromkeys(t14.DETECTOR_FAMILIES, False)        # sensibilidad: NO la explican rolls ni huecos (comprobado)
COHERENT = object()


def annual_like(primary):
    """Analisis anual COHERENTE con las ventanas: muestra inestabilidad exactamente donde las ventanas la ven."""
    return {f: v["veredicto"] in t14.NONSTABLE_VERDICTS for f, v in primary.items()}


def stop(primary, strict=None, annual=COHERENT, sens=ALL_NO, **kw):
    annual = annual_like(primary) if annual is COHERENT else annual
    return t14.decide_stop14(primary, primary if strict is None else strict, annual, sens, **kw)


def test_stop14_scopes_with_explicit_checks():
    assert stop(fams())["alcance"] == t14.STOP_NOT_ACTIVE
    sc = fams(ESCALA=J_, DISCRECION=J_)
    assert stop(sc)["alcance"] == t14.STOP_SCALE and stop(sc)["definitivo"]
    one = fams(ESCALA=J_, RELOJ=J_)
    out = stop(one)
    assert out["alcance"] == t14.STOP_SPECIFIC and t14.STOP_CONSEQUENCES["RELOJ"] in out["consecuencias"]
    two = fams(RELOJ=J_, FORMA=J_)
    assert stop(two)["alcance"] == t14.STOP_BROAD


def test_stop14_single_property_does_not_activate_general_restriction():
    out = stop(fams(DEPENDENCIA_LINEAL=J_))
    assert out["alcance"] == t14.STOP_SPECIFIC and out["nivel5"] != "por subperiodo o no se ejecuta"


def test_stop14_episodes_mark_periods_but_do_not_activate():
    out = stop(fams(RELOJ=E_, FORMA=E_))
    assert out["alcance"] == t14.STOP_NOT_ACTIVE and out["episodios"] == ["FORMA", "RELOJ"]


def test_stop14_missing_information_is_never_a_passed_check():
    """Correccion de auditoria: sin analisis anual o sin sensibilidad, NO hay activacion definitiva."""
    two = fams(RELOJ=J_, FORMA=J_)
    yes = annual_like(two)
    for annual, sens in ((None, ALL_NO), (yes, None), (None, None), ({"RELOJ": True}, ALL_NO), (yes, {"FORMA": False})):
        out = t14.decide_stop14(two, two, annual, sens)
        assert out["alcance"] == t14.STOP_PENDING and not out["definitivo"], (annual, sens)
        assert out["alcance_provisional"] == t14.STOP_BROAD and out["familias_pendientes"]
        assert out["familias_persistentes"] != ["FORMA", "RELOJ"]
    one = t14.decide_stop14(fams(ESCALA=J_), fams(ESCALA=J_))
    assert one["alcance"] == t14.STOP_PENDING and one["estado_por_familia"]["ESCALA"] == t14.PS_PENDING


def test_stop14_pending_that_does_not_change_scope_keeps_definitive_decision():
    four = fams(RELOJ=J_, FORMA=J_, ESCALA=J_, DISCRECION=J_)
    out = t14.decide_stop14(four, four, annual_like(four) | {"DISCRECION": None}, ALL_NO)
    assert out["alcance"] == t14.STOP_BROAD and out["definitivo"] and out["familias_pendientes"] == ["DISCRECION"]


def test_stop14_contradictory_results_go_to_human_review():
    two = fams(RELOJ=J_, FORMA=J_)
    out = stop(two, annual=annual_like(two) | {"RELOJ": False})                           # ventanas si, anual no
    assert out["alcance"] == t14.STOP_CONTRADICTORY and not out["definitivo"] and out["familias_contradictorias"] == ["RELOJ"]
    one = fams(RELOJ=J_)
    out = stop(one, annual=annual_like(one) | {"FORMA": True})                            # anual dice inestable donde ventanas dicen ESTABLE
    assert out["estado_por_familia"]["FORMA"] == t14.PS_CONTRADICTORY and out["alcance"] == t14.STOP_CONTRADICTORY
    ne = {"veredicto": t14.V_NOT_EVALUABLE, "anotacion": t14.A_FEW_WINDOWS}
    assert t14.persistence_status(ne, ne, True, False) == t14.PS_NONE                     # sin evidencia en ventanas no hay contradiccion


def test_stop14_explicit_negative_checks_deactivate():
    two = fams(RELOJ=J_, FORMA=J_)
    out = stop(two, annual=None, sens=ALL_NO | {"RELOJ": True, "FORMA": True})            # rolls/huecos la explican (comprobado)
    assert out["alcance"] == t14.STOP_NOT_ACTIVE and out["definitivo"]
    assert out["estado_por_familia"]["RELOJ"] == t14.PS_EXPLAINED
    assert stop(two, strict=fams())["alcance"] == t14.STOP_NOT_ACTIVE                     # desaparece con el doble de materialidad


def test_stop14_not_validated_goes_to_human_review():
    two = fams(RELOJ=J_, FORMA=J_)
    assert stop(two, procedure_validated=False)["alcance"] == t14.STOP_HUMAN_REVIEW
    assert stop(two, scope_validated=False)["alcance"] == t14.STOP_HUMAN_REVIEW
    assert stop(fams(ESCALA=J_), scope_validated=False)["alcance"] == t14.STOP_SCALE


# ===========================================================================
# Aceptacion binomial
# ===========================================================================

def test_clopper_pearson_known_values():
    assert t14.clopper_pearson_upper(0, 59, 0.95) == pytest.approx(1 - 0.05 ** (1 / 59), rel=1e-6)   # "regla de 3"
    assert t14.clopper_pearson_upper(5, 5, 0.95) == 1.0 and t14.clopper_pearson_lower(0, 5, 0.95) == 0.0
    assert t14.clopper_pearson_lower(50, 50, 0.95) == pytest.approx(0.05 ** (1 / 50), rel=1e-6)


def test_tolerance_limit_is_not_a_demonstration():
    # 33/500 era un limite de tolerancia: su limite superior de confianza esta por encima del 5 %.
    assert t14.clopper_pearson_upper(33, 500, 0.95) > 0.05
    kmax = t14.max_accepted_false_alarms(500, 0.05, 0.95)
    assert t14.clopper_pearson_upper(kmax, 500, 0.95) <= 0.05 < t14.clopper_pearson_upper(kmax + 1, 500, 0.95)
    assert kmax < 25                                                  # demostrar <= 5 % exige observar menos del 5 %
    assert t14.acceptance_probability(500, 0.025) > t14.acceptance_probability(500, 0.04) > t14.acceptance_probability(500, 0.05)
    assert t14.max_accepted_false_alarms(20, 0.05, 0.95) == -1        # con 20 mundos no se puede demostrar nada


# ===========================================================================
# Calibracion, congelacion y validacion (mundos pequeños)
# ===========================================================================

N_CAL = 120
STEP2 = t14.Change(t14.FAM_SCALE, t14.SHAPE_STEP, 3.0, start=0.5)
TREND = t14.Change(t14.FAM_SCALE, t14.SHAPE_TREND, 4.0)
EPIS = t14.Change(t14.FAM_SCALE, t14.SHAPE_EPISODE, 3.0, start=0.35, end=0.65)
EDGE = t14.Change(t14.FAM_SCALE, t14.SHAPE_STEP, 3.0, start=0.93)
TRUTH = {"P_SALTO": (t14.FAM_SCALE, t14.TYPE_STEP), "P_TENDENCIA": (t14.FAM_SCALE, t14.TYPE_TREND),
         "P_EPISODIO": (t14.FAM_SCALE, t14.TYPE_EPISODE), "P_BORDE": (t14.FAM_SCALE, t14.TYPE_EDGE)}
CHANGES = {"P_SALTO": STEP2, "P_TENDENCIA": TREND, "P_EPISODIO": EPIS, "P_BORDE": EDGE}


def wspec(label, changes=(), **kw):
    return replace(WORLD, label=label, changes=changes, **kw)


# Manifiesto reducido con las especificaciones de mundo CONGELADAS (las mismas que construye ``results``).
SMALL = t14.ValidationManifest(
    negative_specs=(("NEGATIVO_A", wspec("NEGATIVO")),), partial_spec=wspec("PARCIAL", (STEP2,)),
    type_specs=tuple((TRUTH[lab][1], TRUTH[lab][0], wspec(lab, (ch,))) for lab, ch in CHANGES.items()),
    power_specs=((t14.FAM_SCALE, t14.POWER_LEVEL_DOUBLE, wspec("POT", (STEP2,))),),
    detector_spec=SPEC, min_negative_replicates=30, min_partial_replicates=10, min_type_replicates=5, min_power_replicates=10)


def results(role, label, n, changes=(), first=0, thr=None, **kw):
    wspec = replace(WORLD, label=label, changes=changes, **kw)
    return t14.run_world_tasks(t14.make_tasks(role, wspec, n, SPEC, first_rep=first, thr=thr), n_workers=1)


@pytest.fixture(scope="module")
def calibrated():
    neg = results(t14.ROLE_CALIBRATION, "NEGATIVO", N_CAL)
    thr = t14.calibrate_thresholds(neg, SMALL, target=0.05)
    pos = [r for lab, ch in CHANGES.items() for r in results(t14.ROLE_CALIBRATION, lab, 10, (ch,))]
    full = t14.calibrate_type_margin(thr, pos, SMALL)
    return neg, thr, full


def test_calibration_controls_global_false_alarm_jointly(calibrated):
    neg, thr, full = calibrated
    info = dict(thr.calibration_info)
    assert info["tasa_global_calibracion"] <= 0.05 and thr.k_star >= 2
    rate = np.mean([t14.apply_rule(r.stats, full)["alarma_global"] for r in neg])
    assert rate <= 0.05                                               # incluye familias, fechas y STOP-14
    assert thr.type_margin is None and full.type_margin is not None and full.sha256 != thr.sha256


def test_calibration_refuses_validation_worlds(calibrated):
    neg, thr, _ = calibrated
    val = results(t14.ROLE_VALIDATION, "NEGATIVO", 2, thr=calibrated[2])
    with pytest.raises(t14.CalibrationRoleError):
        t14.calibrate_thresholds(neg + val, SMALL)


def test_validation_requires_frozen_complete_thresholds(calibrated):
    _, thr, full = calibrated
    with pytest.raises(t14.FrozenThresholdsError):
        t14.make_tasks(t14.ROLE_VALIDATION, WORLD, 1, SPEC)                       # sin umbrales
    with pytest.raises(t14.FrozenThresholdsError):
        t14.make_tasks(t14.ROLE_VALIDATION, WORLD, 1, SPEC, thr=thr)              # sin margen de tipo
    tampered = replace(full, k_star=full.k_star + 1)
    with pytest.raises(t14.FrozenThresholdsError):
        t14.verify_frozen(tampered)




@pytest.fixture(scope="module")
def val_worlds(calibrated):
    full = calibrated[2]
    return {"neg": results(t14.ROLE_VALIDATION, "NEGATIVO", 30, thr=full),
            "part": results(t14.ROLE_VALIDATION, "PARCIAL", 10, (STEP2,), thr=full),
            "typed": [r for lab, ch in CHANGES.items() for r in results(t14.ROLE_VALIDATION, lab, 5, (ch,), thr=full)],
            "power": results(t14.ROLE_VALIDATION, "POT", 10, (STEP2,), thr=full)}


def test_validation_without_negative_controls_cannot_be_approved(calibrated, val_worlds):
    full = calibrated[2]
    with pytest.raises(t14.ValidationManifestError, match="sin controles negativos"):
        t14.validate_procedure(full, SMALL, {})
    with pytest.raises(t14.ValidationManifestError, match="distintas de las del manifiesto"):
        t14.validate_procedure(full, SMALL, {"OTRA": val_worlds["neg"]})
    with pytest.raises(t14.ValidationManifestError, match="exige al menos"):
        t14.validate_procedure(full, SMALL, {"NEGATIVO_A": val_worlds["neg"][:29]})
    two = replace(SMALL, negative_specs=SMALL.negative_specs + (("NEGATIVO_B", wspec("NEGATIVO_B")),))
    with pytest.raises(t14.ValidationManifestError):                                          # falta una variante obligatoria
        t14.validate_procedure(full, two, {"NEGATIVO_A": val_worlds["neg"]})


def test_validation_refuses_wrong_roles_foreign_thresholds_and_duplicates(calibrated, val_worlds):
    neg_cal, _, full = calibrated
    with pytest.raises(t14.ValidationManifestError, match="rol"):
        t14.validate_procedure(full, SMALL, {"NEGATIVO_A": neg_cal[:30]})
    unsigned = [replace(r, thresholds_sha256="") for r in val_worlds["neg"]]
    with pytest.raises(t14.ValidationManifestError, match="huella"):
        t14.validate_procedure(full, SMALL, {"NEGATIVO_A": unsigned})
    with pytest.raises(t14.ValidationManifestError, match="repetidos"):
        t14.validate_procedure(full, SMALL, {"NEGATIVO_A": val_worlds["neg"]}, partial_negatives=val_worlds["neg"][:10])
    with pytest.raises(t14.ValidationManifestError, match="no previstos"):
        t14.validate_procedure(full, SMALL, {"NEGATIVO_A": val_worlds["neg"]}, power_worlds={(t14.FAM_CLOCK, t14.POWER_LEVEL_DOUBLE): val_worlds["power"]})
    with pytest.raises(t14.ValidationManifestError, match="sin especificacion congelada"):
        t14.validate_procedure(full, replace(SMALL, type_specs=()), {"NEGATIVO_A": val_worlds["neg"]}, typed_positives=val_worlds["typed"])


def test_missing_partial_types_and_power_are_explicit_and_block(calibrated, val_worlds):
    full = calibrated[2]
    val = t14.validate_procedure(full, SMALL, {"NEGATIVO_A": val_worlds["neg"]})
    assert set(val.missing_controls) == {"NEGATIVO_PARCIAL_SOLO_ESCALA", "POSITIVOS_CON_TIPO", f"POTENCIA_{t14.FAM_SCALE}_{t14.POWER_LEVEL_DOUBLE}"}
    states = set(val.table["estado"])
    assert t14.STATUS_ABSENT in states
    assert not val.stop14_scope_validated and not val.types_validated and not any(val.power_ok.values())
    short = t14.validate_procedure(full, SMALL, {"NEGATIVO_A": val_worlds["neg"]}, partial_negatives=val_worlds["part"][:4],
                                   typed_positives=val_worlds["typed"][:6], power_worlds={(t14.FAM_SCALE, t14.POWER_LEVEL_DOUBLE): val_worlds["power"][:3]})
    assert (short.table["estado"] == t14.STATUS_INSUFFICIENT).sum() == 3
    assert not short.stop14_scope_validated and not short.types_validated and not short.power_ok[t14.FAM_SCALE]


def test_blocked_decisions_follow_missing_validations(calibrated):
    """Aunque el error global estuviera validado: sin tipos validados no se publican tipos, sin potencia ninguna familia es
    ESTABLE y sin control parcial validado STOP-14 nunca es especifico/amplio automatico."""
    full = calibrated[2]
    val = t14.ValidationResult(table=pd.DataFrame(), global_validated=True, dates_validated=True,
                               families_validated=dict.fromkeys(t14.DETECTOR_FAMILIES, True), stop14_scope_validated=False,
                               types_validated=False, power_ok=dict.fromkeys(t14.DETECTOR_FAMILIES, False), thresholds_sha256=full.sha256,
                               manifest_sha256=SMALL.sha256())
    r = results(t14.ROLE_TEST, "P_MULTI", 1, (STEP2, t14.Change(t14.FAM_CLOCK, t14.SHAPE_STEP, 3.0, 0.5)), first=701)[0]
    pre = t14.apply_rule(r.stats, full)["veredictos_familia"]
    dec = t14.final_decisions(r.stats, full, val, annual_confirmed=annual_like(pre), explained_by_sensitivity=ALL_NO)
    fv = dec["veredictos_familia"]
    assert fv[t14.FAM_SCALE]["veredicto"] == t14.V_UNSTABLE and fv[t14.FAM_SCALE]["anotacion"] == t14.A_TYPE_UNKNOWN
    assert all(v["veredicto"] != t14.V_STABLE for v in fv.values())
    assert dec["stop14"]["alcance"] not in (t14.STOP_SPECIFIC, t14.STOP_BROAD)
    no_info = t14.final_decisions(r.stats, full, val)                                          # sin comprobaciones anual/sensibilidad
    assert not no_info["stop14"]["definitivo"]


def test_detection_of_types_on_synthetic_worlds(calibrated):
    full = calibrated[2]
    got = {}
    for lab, ch in CHANGES.items():
        r = results(t14.ROLE_TEST, lab, 1, (ch,), first=500)[0]
        got[lab] = t14.apply_rule(r.stats, full)["veredictos_familia"][t14.FAM_SCALE]
    pair = {k: (v["veredicto"], v["anotacion"]) for k, v in got.items()}
    assert pair["P_SALTO"] == (t14.V_JUMP, "")
    assert pair["P_TENDENCIA"] == (t14.V_TREND, "")
    assert pair["P_EPISODIO"] == (t14.V_UNSTABLE, t14.A_EPISODE)
    assert pair["P_BORDE"] == (t14.V_UNSTABLE, t14.A_EDGE)           # un cambio en el borde nunca se declara permanente


def test_type_recognition_over_several_worlds(calibrated):
    """Sobre 8 mundos por tipo: la mayoria recibe su tipo correcto y casi ninguno un tipo especifico equivocado."""
    full = calibrated[2]
    for lab, ch in CHANGES.items():
        res = results(t14.ROLE_TEST, lab, 8, (ch,), first=1000)
        got = [t14.apply_rule(r.stats, full)["veredictos_familia"][t14.FAM_SCALE] for r in res]
        right = sum((g["veredicto"], g["anotacion"]) == t14.EXPECTED_TYPE[TRUTH[lab][1]] for g in got)
        te = t14.type_errors(res, TRUTH, full, full.type_margin)
        assert right >= 5, (lab, [(g["veredicto"], g["anotacion"]) for g in got])
        assert te["errores"] <= 1, (lab, te)


def test_stable_world_and_date_of_a_step(calibrated):
    full = calibrated[2]
    neg = results(t14.ROLE_TEST, "NEGATIVO", 1, first=700)[0]
    dec = t14.apply_rule(neg.stats, full)
    assert dec["veredictos_familia"][t14.FAM_SCALE]["veredicto"] in (t14.V_STABLE, t14.V_NOT_EVALUABLE)
    pos = results(t14.ROLE_TEST, "P_MULTI", 1, (t14.Change(t14.FAM_SCALE, t14.SHAPE_STEP, 3.0, 0.5), t14.Change(t14.FAM_CLOCK, t14.SHAPE_STEP, 3.0, 0.5)), first=701)[0]
    dec = t14.apply_rule(pos.stats, full)
    assert dec["fechas"], "un cambio simultaneo en dos familias debe dar una fecha candidata"
    assert any(d["jornada_inicio"] - 9 <= 90 <= d["jornada_fin"] + 9 for d in dec["fechas"])


def test_validate_procedure_end_to_end_and_consequences(calibrated, val_worlds):
    _, _, full = calibrated
    sha = full.sha256
    neg = {"NEGATIVO_A": val_worlds["neg"]}
    power = {(t14.FAM_SCALE, t14.POWER_LEVEL_DOUBLE): val_worlds["power"]}
    val = t14.validate_procedure(full, SMALL, neg, val_worlds["part"], val_worlds["typed"], power)
    assert full.sha256 == sha and val.thresholds_sha256 == sha and val.manifest_sha256 == SMALL.sha256()
    assert val.missing_controls == ()
    tab = val.table
    assert {"limite_confianza", "wilson_95_inf", "estado"} <= set(tab.columns)
    g = tab.loc[(tab["control"] == "NEGATIVO_A") & (tab["decision"] == "GLOBAL")].iloc[0]
    assert g["confianza"] == pytest.approx(1 - 0.05 / 2)              # Bonferroni: variante negativa + control parcial
    # con 30 mundos no puede demostrarse <= 5 %: el procedimiento queda NO validado y todo es descriptivo
    assert g["estado"] == t14.STATUS_NOT_VALIDATED and not val.global_validated
    dec = t14.final_decisions(neg["NEGATIVO_A"][0].stats, full, val)
    assert dec["fechas"] == [] and dec["estado"].startswith("SOLO_DESCRIPTIVO")
    assert all(v["validacion"] == t14.STATUS_NOT_VALIDATED for v in dec["veredictos_familia"].values())
    # sin potencia demostrada ninguna familia puede declararse ESTABLE
    assert all(v["veredicto"] != t14.V_STABLE for f, v in dec["veredictos_familia"].items() if not val.power_ok[f])
    assert "garant" not in " ".join(tab["estado"].astype(str)).lower()


def test_power_requirement_turns_stable_into_not_evaluable(calibrated):
    full = calibrated[2]
    r = results(t14.ROLE_TEST, "NEGATIVO", 1, first=900)[0]
    base = t14.apply_rule(r.stats, full)
    low = t14.apply_rule(r.stats, full, power_ok=dict.fromkeys(t14.DETECTOR_FAMILIES, False))
    for f in t14.DETECTOR_FAMILIES:
        if base["veredictos_familia"][f]["veredicto"] == t14.V_STABLE:
            assert low["veredictos_familia"][f] == {"veredicto": t14.V_NOT_EVALUABLE, "anotacion": t14.A_LOW_POWER}


def test_final_decisions_without_validation_publish_no_dates(calibrated):
    full = calibrated[2]
    r = results(t14.ROLE_TEST, "P_MULTI", 1, (STEP2, t14.Change(t14.FAM_CLOCK, t14.SHAPE_STEP, 3.0, 0.5)), first=701)[0]
    dec = t14.final_decisions(r.stats, full, None)
    assert dec["estado"] == "DESCRIPTIVO_SIN_VALIDACION" and dec["fechas"] == []


# ===========================================================================
# Reproducibilidad
# ===========================================================================

def test_results_do_not_depend_on_number_of_processes():
    tasks = t14.make_tasks(t14.ROLE_TEST, replace(WORLD, n_sessions=90), 3, SPEC)
    one = t14.run_world_tasks(tasks, n_workers=1)
    two = t14.run_world_tasks(tasks, n_workers=2)
    for a, b in zip(one, two):
        for k in a.stats.values:
            np.testing.assert_array_equal(a.stats.values[k], b.stats.values[k])


def test_higher_persistence_raises_calibrated_thresholds():
    """Con volatilidad diaria mas persistente la escala varia mas entre ventanas sin que nada cambie: el umbral calibrado
    de la escala sube (y la potencia baja). Por eso el mundo negativo debe usar la persistencia real (extremo alto)."""
    low = results(t14.ROLE_CALIBRATION, "NEG_BAJA", 60, daily_phi=0.5, daily_sd=0.05)
    high = results(t14.ROLE_CALIBRATION, "NEG_ALTA", 60, daily_phi=0.97, daily_sd=0.10)
    d_low = np.nanquantile([t14.constancy_statistic(*t14.nonoverlapping(r.stats, "log_sigma")) for r in low], 0.9)
    d_high = np.nanquantile([t14.constancy_statistic(*t14.nonoverlapping(r.stats, "log_sigma")) for r in high], 0.9)
    assert d_high > 3 * d_low


def test_thresholds_hash_is_deterministic(calibrated):
    _, thr, full = calibrated
    assert t14.freeze_thresholds(full).sha256 == full.sha256
    t14.verify_frozen(full)


def test_worker_count_respects_memory_protection(monkeypatch):
    monkeypatch.setattr(t14, "available_memory_gb", lambda: 5.0)            # (5 - 4) * 0,6 / 1 -> 0 -> 1 proceso
    assert t14.resolve_workers(8, 100) == 1
    monkeypatch.setattr(t14, "available_memory_gb", lambda: None)
    assert t14.resolve_workers(8, 100) == 1


# ===========================================================================
# Tabla anual, curtosis y TH30
# ===========================================================================

def test_annual_table_and_kurtosis_evaluability():
    p = world(n_sessions=150)
    p.session_date = np.array([np.datetime64(f"{2020 + i // 30}-03-01") for i in range(150)], dtype="datetime64[D]")
    tab = t14.annual_table(p, n_boot=20, block_sessions=5, seed=1, spec=SPEC)
    assert set(tab["anio"]) == {2020, 2021, 2022, 2023, 2024}
    row = tab.loc[(tab["anio"] == 2020) & (tab["estadistico"] == "log_sigma")].iloc[0]
    assert row["ic95_inf"] <= row["valor"] <= row["ic95_sup"]
    ev = t14.kurtosis_evaluability(tab)
    assert isinstance(ev["evaluable"], bool) and len(ev["semiamplitud_relativa"]) == 5
    fake = pd.DataFrame({"estadistico": "curtosis_total", "valor": [10.0] * 5, "ic95_inf": [9, 9, 9, 2, 2], "ic95_sup": [11, 11, 11, 30, 30]})
    assert t14.kurtosis_evaluability(fake)["evaluable"] is False


def test_th30_compares_dimensionless_indices_and_never_claims_stationarity():
    p = world(n_sessions=126, roll_every=63)
    tab = t14.th30_within_contract(p, 21)
    assert set(tab["serie"]) == {"LN_C", "R"} and set(tab["contrato"]) == {"C000", "C001"}
    out = t14.th30_summary(tab)
    assert out["desplazamiento_mediano_ln_c"] > out["desplazamiento_mediano_r"]
    assert out["conclusion"] in (t14.TH30_R_MORE_STABLE, t14.TH30_NEITHER)
    assert "no demuestra estacionariedad" in out["nota"]
    assert all("ESTACIONARI" not in c for c in (t14.TH30_R_MORE_STABLE, t14.TH30_NEITHER, t14.TH30_INCONCLUSIVE))


# ===========================================================================
# Tercera auditoria: borde y persistencia, capacidad de tipos, trazabilidad
# ===========================================================================

def _family_from(y, **kw):
    """Veredicto de familia (un solo estadistico) a partir de una serie controlada, tal como lo produce el detector."""
    return t14.combine_family([classify(y, **kw)])


def test_edge_change_never_confirms_persistence_even_with_favourable_checks():
    """Regresion: un salto en las ultimas ventanas (sin historia posterior) no activa STOP-14 aunque el analisis anual lo
    muestre y las sensibilidades no lo expliquen. Antes salia como 'tipo no identificable' y se confirmaba persistente."""
    edge = _family_from(noise(4) + np.r_[np.zeros(18), np.ones(2)])
    assert edge["anotacion"] == t14.A_EDGE and edge["soporte"] == t14.SUP_EDGE
    assert t14.persistence_status(edge, edge, True, False) == t14.PS_EDGE
    fam = fams(RELOJ=edge, FORMA=edge)
    out = t14.decide_stop14(fam, fam, dict.fromkeys(t14.DETECTOR_FAMILIES, False) | {"RELOJ": True, "FORMA": True}, ALL_NO)
    assert out["alcance"] == t14.STOP_NOT_ACTIVE and out["cambios_en_borde"] == ["FORMA", "RELOJ"]
    assert out["familias_persistentes"] == []


def test_three_kinds_of_instability_are_distinguished_for_stop14():
    permanent = _family_from(noise(1) + np.r_[np.zeros(10), np.ones(10)])
    no_pattern = _family_from(noise(5) + np.tile([0, 0, 0, 0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0], 2))
    edge = _family_from(noise(4) + np.r_[np.zeros(18), np.ones(2)])
    assert (permanent["veredicto"], permanent["soporte"]) == (t14.V_JUMP, t14.SUP_PERMANENT)
    assert (no_pattern["anotacion"], no_pattern["soporte"]) == (t14.A_NO_PATTERN, t14.SUP_NO_PATTERN)
    assert (edge["anotacion"], edge["soporte"]) == (t14.A_EDGE, t14.SUP_EDGE)
    assert t14.persistence_status(permanent, permanent, True, False) == t14.PS_CONFIRMED
    assert t14.persistence_status(no_pattern, no_pattern, True, False) == t14.PS_CONFIRMED
    assert t14.persistence_status(edge, edge, True, False) == t14.PS_EDGE


def test_type_unknown_without_proven_support_is_not_persistent():
    # Sin campo de soporte (veredicto construido a mano): conservador, la permanencia no esta demostrada.
    bare = {"veredicto": t14.V_UNSTABLE, "anotacion": t14.A_TYPE_UNKNOWN}
    assert t14.support_of(bare) == t14.SUP_UNPROVEN
    assert t14.persistence_status(bare, bare, True, False) == t14.PS_UNPROVEN
    # Tipo no identificable entre alternativas que incluyen un borde: tampoco.
    unk = _family_from(noise(6) + np.r_[np.zeros(10), np.ones(10)], margin=1e9)
    assert unk["anotacion"] == t14.A_TYPE_UNKNOWN and t14.persistence_status(unk, unk, True, False) == t14.PS_UNPROVEN
    # Tipo no identificable entre alternativas PERMANENTES (tres niveles sin retorno): si es persistente.
    stair = classify(noise(7) + np.r_[np.zeros(7), np.full(6, 1.0), np.full(7, 2.0)], margin=0.0)
    assert (stair["anotacion"], stair["soporte"]) == (t14.A_TYPE_UNKNOWN, t14.SUP_PERMANENT)
    assert t14.persistence_status(stair, stair, True, False) == t14.PS_CONFIRMED


def test_family_support_uses_strongest_demonstrated_support():
    j = {"veredicto": t14.V_JUMP, "anotacion": "", "soporte": t14.SUP_PERMANENT}
    e = {"veredicto": t14.V_UNSTABLE, "anotacion": t14.A_EDGE, "soporte": t14.SUP_EDGE}
    assert t14.combine_family([j, e]) == {"veredicto": t14.V_UNSTABLE, "anotacion": t14.A_TYPE_UNKNOWN, "soporte": t14.SUP_PERMANENT}
    assert t14.combine_family([e, e])["soporte"] == t14.SUP_EDGE


def test_edge_world_does_not_activate_stop14_end_to_end(calibrated):
    full = calibrated[2]
    r = results(t14.ROLE_TEST, "P_BORDE", 1, (EDGE,), first=500)[0]
    dec = t14.apply_rule(r.stats, full)
    fv = dec["veredictos_familia"][t14.FAM_SCALE]
    assert fv["anotacion"] == t14.A_EDGE
    out = t14.decide_stop14(dec["veredictos_familia"], dec["veredictos_familia_estricta"],
                            annual_like(dec["veredictos_familia"]), ALL_NO)
    assert t14.FAM_SCALE not in out["familias_persistentes"] and t14.FAM_SCALE in out["cambios_en_borde"]


# --- capacidad de identificar tipos -----------------------------------------

UNKNOWN_V = {"veredicto": t14.V_UNSTABLE, "anotacion": t14.A_TYPE_UNKNOWN}


def test_100_detected_and_100_unidentified_is_not_classification_capacity():
    """100 cambios detectados, 100 tipos no identificables: 0 errores, pero capacidad 0. No se validan los tipos."""
    tm = t14.type_metrics([(t14.TYPE_STEP, UNKNOWN_V)] * 100)
    assert (tm["detectados"], tm["errores"], tm["correctos"], tm["abstenciones"]) == (100, 0, 0, 100)
    assert tm["tasa_errores"] == 0.0 and tm["capacidad"] == 0.0 and tm["tasa_abstencion"] == 1.0
    ok, rows = t14.type_metric_rows(tm)
    by = {r["decision"]: r for r in rows if r["control"] == "POSITIVOS_CON_TIPO"}
    assert by["TIPOS_ESPECIFICOS_INCORRECTOS"]["estado"] == t14.STATUS_COMPATIBLE         # la tasa de errores, sola, "pasaria"
    assert by["CAPACIDAD_DE_IDENTIFICAR_TIPOS"]["estado"] == t14.STATUS_NOT_VALIDATED
    assert by["ABSTENCIONES"]["tasa"] == 1.0 and not ok


def test_type_metrics_good_classifier_and_error_definitions():
    jump = {"veredicto": t14.V_JUMP, "anotacion": ""}
    edge = {"veredicto": t14.V_UNSTABLE, "anotacion": t14.A_EDGE}
    outcomes = [(t14.TYPE_STEP, jump)] * 90 + [(t14.TYPE_STEP, {"veredicto": t14.V_TREND, "anotacion": ""})] * 3 \
        + [(t14.TYPE_STEP, UNKNOWN_V)] * 7 + [(t14.TYPE_STEP, None)] * 5
    tm = t14.type_metrics(outcomes)
    assert (tm["mundos"], tm["detectados"], tm["correctos"], tm["errores"], tm["abstenciones"]) == (105, 100, 90, 3, 7)
    ok, _ = t14.type_metric_rows(tm)
    assert ok
    assert t14.type_metrics([(t14.TYPE_EDGE, edge)])["correctos"] == 1                  # borde reconocido como borde
    assert t14.type_metrics([(t14.TYPE_EDGE, jump)])["errores"] == 1                    # afirmar permanencia en un borde es error
    assert t14.type_metrics([(t14.TYPE_STEP, edge)])["abstenciones"] == 1               # borde en un salto central: abstencion


def test_validation_reports_both_type_metrics(calibrated, val_worlds):
    full = calibrated[2]
    val = t14.validate_procedure(full, SMALL, {"NEGATIVO_A": val_worlds["neg"]}, val_worlds["part"], val_worlds["typed"],
                                 {(t14.FAM_SCALE, t14.POWER_LEVEL_DOUBLE): val_worlds["power"]})
    dec = set(val.table.loc[val.table["control"] == "POSITIVOS_CON_TIPO", "decision"])
    assert {"TIPOS_ESPECIFICOS_INCORRECTOS", "CAPACIDAD_DE_IDENTIFICAR_TIPOS", "ABSTENCIONES", "DETECTADOS"} <= dec
    assert set(val.type_metrics) >= {"tasa_errores", "capacidad", "tasa_abstencion", "por_tipo"}


# --- trazabilidad de los mundos sinteticos -----------------------------------

def test_executed_worlds_carry_their_spec_and_detector_hashes(calibrated, val_worlds):
    full = calibrated[2]
    r = val_worlds["neg"][0]
    assert r.world_spec_sha256 == t14.world_spec_sha256(wspec("NEGATIVO"))
    assert r.detector_spec_sha256 == t14.detector_spec_sha256(SPEC)
    val = t14.validate_procedure(full, SMALL, {"NEGATIVO_A": val_worlds["neg"]}, val_worlds["part"], val_worlds["typed"],
                                 {(t14.FAM_SCALE, t14.POWER_LEVEL_DOUBLE): val_worlds["power"]})
    hashes = SMALL.spec_hashes()
    st = val.specs_table
    assert st.loc[st["control"] == "NEGATIVO:NEGATIVO_A", "sha256_especificacion"].item() == hashes["NEGATIVO:NEGATIVO_A"]
    assert st.loc[st["control"] == "PARCIAL", "sha256_especificacion"].item() == hashes["PARCIAL"]
    assert st["mundos_verificados"].sum() == 30 + 10 + 20 + 10
    assert "especificaciones_calibracion" in dict(calibrated[1].calibration_info)


def test_spec_hash_covers_every_methodological_parameter():
    base = wspec("X", (STEP2,))
    variants = [replace(base, daily_phi=0.81), replace(base, seasonal_amplitude=0.1), replace(base, t_df=5.0),
                replace(base, intra_phi=0.97), replace(base, gap_probability=0.06), replace(base, tick=0.5),
                replace(base, changes=(replace(STEP2, magnitude=2.9),)), replace(base, changes=(replace(STEP2, start=0.51),)),
                replace(base, n_sessions=181)]
    hashes = {t14.world_spec_sha256(v) for v in variants}
    assert len(hashes) == len(variants) and t14.world_spec_sha256(base) not in hashes


def test_world_executed_with_different_spec_is_rejected(calibrated, val_worlds):
    """Misma etiqueta y misma cantidad, pero persistencia distinta de la congelada: se rechaza."""
    full = calibrated[2]
    other = results(t14.ROLE_VALIDATION, "NEGATIVO", 30, thr=full, first=5000, daily_phi=0.81)
    with pytest.raises(t14.ValidationManifestError, match="especificacion distinta"):
        t14.validate_procedure(full, SMALL, {"NEGATIVO_A": other})
    mixed = val_worlds["neg"][:29] + other[:1]                                      # un solo mundo discrepante basta
    with pytest.raises(t14.ValidationManifestError, match="especificacion distinta"):
        t14.validate_procedure(full, SMALL, {"NEGATIVO_A": mixed})
    forged = [replace(r, world_spec_sha256="0" * 64) for r in val_worlds["neg"]]
    with pytest.raises(t14.ValidationManifestError, match="especificacion distinta"):
        t14.validate_procedure(full, SMALL, {"NEGATIVO_A": forged})
    with pytest.raises(t14.ValidationManifestError, match="especificacion distinta"):
        t14.validate_procedure(full, replace(SMALL, partial_spec=wspec("PARCIAL", (replace(STEP2, magnitude=2.0),))),
                               {"NEGATIVO_A": val_worlds["neg"]}, val_worlds["part"])


def test_world_analyzed_with_other_detector_or_label_is_rejected(calibrated, val_worlds):
    full = calibrated[2]
    other_det = [replace(r, detector_spec_sha256=t14.detector_spec_sha256(replace(SPEC, max_lag=12))) for r in val_worlds["neg"]]
    with pytest.raises(t14.ValidationManifestError, match="detector distinto"):
        t14.validate_procedure(full, SMALL, {"NEGATIVO_A": other_det})
    relabeled = replace(SMALL, negative_specs=(("NEGATIVO_A", wspec("NEGATIVO_OTRO")),))
    with pytest.raises(t14.ValidationManifestError, match="etiqueta"):
        t14.validate_procedure(full, relabeled, {"NEGATIVO_A": val_worlds["neg"]})
    with pytest.raises(t14.ValidationManifestError, match="repite etiquetas"):
        t14.ValidationManifest(negative_specs=(("A", wspec("NEGATIVO")), ("B", wspec("NEGATIVO"))))
    with pytest.raises(t14.ValidationManifestError, match="no congela"):
        t14.validate_procedure(full, t14.ValidationManifest(), {"NEGATIVO_A": val_worlds["neg"]})


def test_manifest_hash_changes_with_any_frozen_spec():
    assert SMALL.sha256() != replace(SMALL, partial_spec=wspec("PARCIAL", (replace(STEP2, magnitude=2.5),))).sha256()
    assert SMALL.sha256() != replace(SMALL, detector_spec=replace(SPEC, max_lag=12)).sha256()
    assert SMALL.sha256() == replace(SMALL).sha256()


def test_task_label_must_match_its_spec():
    task = (t14.ROLE_TEST, "OTRA", 0, wspec("NEGATIVO", n_sessions=90), SPEC, "")
    with pytest.raises(ValueError, match="no coincide"):
        t14.run_world_task(task)


# ===========================================================================
# Cuarta auditoria: capacidad por tipo y coherencia del manifiesto y las fases
# ===========================================================================

JUMP_V = {"veredicto": t14.V_JUMP, "anotacion": ""}
TREND_V = {"veredicto": t14.V_TREND, "anotacion": ""}
EPIS_V = {"veredicto": t14.V_UNSTABLE, "anotacion": t14.A_EPISODE}
EDGE_V = {"veredicto": t14.V_UNSTABLE, "anotacion": t14.A_EDGE}
ALL_TYPES = (t14.TYPE_STEP, t14.TYPE_TREND, t14.TYPE_EPISODE, t14.TYPE_EDGE)


def test_type_never_identified_is_not_validated_even_if_global_capacity_passes():
    """100 casos de cada tipo: salto, tendencia y episodio identificados siempre; borde nunca (100 abstenciones).
    Capacidad global 300/400 = 75 %: pasaria sola. La capacidad del tipo BORDE es 0: los tipos no se validan."""
    outcomes = [(t14.TYPE_STEP, JUMP_V)] * 100 + [(t14.TYPE_TREND, TREND_V)] * 100 + [(t14.TYPE_EPISODE, EPIS_V)] * 100 \
        + [(t14.TYPE_EDGE, UNKNOWN_V)] * 100
    tm = t14.type_metrics(outcomes)
    assert tm["capacidad"] == 0.75 and tm["errores"] == 0
    ok, rows = t14.type_metric_rows(tm, required_types=ALL_TYPES)
    glob = {r["decision"]: r for r in rows if r["control"] == "POSITIVOS_CON_TIPO"}
    per = {r["control"]: r for r in rows if r["control"].startswith("POSITIVOS_TIPO_")}
    assert glob["CAPACIDAD_DE_IDENTIFICAR_TIPOS"]["estado"] == t14.STATUS_COMPATIBLE
    assert per["POSITIVOS_TIPO_BORDE"]["estado"] == t14.STATUS_NOT_VALIDATED and per["POSITIVOS_TIPO_BORDE"]["tasa"] == 0.0
    assert all(per[f"POSITIVOS_TIPO_{t}"]["estado"] == t14.STATUS_COMPATIBLE for t in ALL_TYPES[:3])
    assert not ok


def test_per_type_rows_keep_binomial_uncertainty_with_bonferroni():
    outcomes = [(t, v) for t, v in zip(ALL_TYPES, (JUMP_V, TREND_V, EPIS_V, EDGE_V)) for _ in range(100)]
    ok, rows = t14.type_metric_rows(t14.type_metrics(outcomes), required_types=ALL_TYPES)
    per = [r for r in rows if r["control"].startswith("POSITIVOS_TIPO_")]
    assert ok and len(per) == 4
    for r in per:
        assert r["confianza"] == pytest.approx(1 - 0.05 / 4) and r["tipo_limite"] == "inferior"
        assert r["limite_confianza"] < 1.0 and {"wilson_95_inf", "wilson_95_sup"} <= set(r)
    # 57/100 aciertos: la capacidad puntual supera el 50 %, pero su limite inferior no (incertidumbre explicita).
    weak = [(t14.TYPE_STEP, JUMP_V)] * 57 + [(t14.TYPE_STEP, UNKNOWN_V)] * 43
    ok, rows = t14.type_metric_rows(t14.type_metrics(weak), required_types=(t14.TYPE_STEP,))
    assert not ok and [r for r in rows if r["control"] == "POSITIVOS_TIPO_SALTO"][0]["limite_confianza"] < 0.5


def test_required_type_without_detections_is_not_validated():
    outcomes = [(t, v) for t, v in zip(ALL_TYPES[:3], (JUMP_V, TREND_V, EPIS_V)) for _ in range(100)] + [(t14.TYPE_EDGE, None)] * 100
    ok, rows = t14.type_metric_rows(t14.type_metrics(outcomes), required_types=ALL_TYPES)
    edge = [r for r in rows if r["control"] == "POSITIVOS_TIPO_BORDE"][0]
    assert not ok and edge["n"] == 0 and edge["estado"] == t14.STATUS_NOT_VALIDATED


def test_validation_reports_type_validation_by_type(calibrated, val_worlds):
    full = calibrated[2]
    val = t14.validate_procedure(full, SMALL, {"NEGATIVO_A": val_worlds["neg"]}, val_worlds["part"], val_worlds["typed"],
                                 {(t14.FAM_SCALE, t14.POWER_LEVEL_DOUBLE): val_worlds["power"]})
    assert set(val.types_validated_by_type) == set(ALL_TYPES)
    assert val.types_validated == all(val.types_validated_by_type.values()) or not val.global_validated


# --- coherencia de las especificaciones (contradicciones con hashes validos) ---

def manifest_with(**over):
    return replace(SMALL, **over)


def test_implied_type_of_the_reference_controls():
    for lab, ch in CHANGES.items():
        assert t14.implied_type(ch, wspec(lab, (ch,)), SPEC) == TRUTH[lab][1], lab
    assert t14.implied_type(t14.Change(t14.FAM_SCALE, t14.SHAPE_EPISODE, 3.0, 0.40, 0.42), WORLD, SPEC) == "AMBIGUO"


def test_negative_controls_cannot_contain_injected_changes():
    with pytest.raises(t14.ValidationManifestError, match="control negativo 'NEGATIVO_A' con cambios"):
        manifest_with(negative_specs=(("NEGATIVO_A", wspec("NEGATIVO", (STEP2,))),))
    with pytest.raises(t14.ValidationManifestError, match="control estres"):
        manifest_with(stress_specs=(("ESTRES", wspec("ESTRES", (EPIS,))),))


def test_partial_control_must_contain_only_the_planned_scale_change():
    clock = t14.Change(t14.FAM_CLOCK, t14.SHAPE_STEP, 3.0, 0.5)
    with pytest.raises(t14.ValidationManifestError, match="solo un cambio de ESCALA"):
        manifest_with(partial_spec=wspec("PARCIAL", (clock,)))
    with pytest.raises(t14.ValidationManifestError, match="solo un cambio de ESCALA"):
        manifest_with(partial_spec=wspec("PARCIAL", (STEP2, clock)))
    with pytest.raises(t14.ValidationManifestError, match="mundo de fondo distinto"):
        manifest_with(partial_spec=wspec("PARCIAL", (STEP2,), daily_phi=0.97))


def test_declared_type_and_family_must_match_injected_change():
    def typed(typ, fam, lab, ch):
        return manifest_with(type_specs=((typ, fam, wspec(lab, (ch,))),))
    with pytest.raises(t14.ValidationManifestError, match="tipo declarado SALTO, el cambio inyectado produce TENDENCIA"):
        typed(t14.TYPE_STEP, t14.FAM_SCALE, "X", TREND)
    with pytest.raises(t14.ValidationManifestError, match="familia declarada RELOJ, inyectada ESCALA"):
        typed(t14.TYPE_STEP, t14.FAM_CLOCK, "X", STEP2)
    with pytest.raises(t14.ValidationManifestError, match="tipo declarado SALTO, el cambio inyectado produce BORDE"):
        typed(t14.TYPE_STEP, t14.FAM_SCALE, "X", EDGE)                       # 'salto' en realidad en el borde
    with pytest.raises(t14.ValidationManifestError, match="tipo declarado BORDE, el cambio inyectado produce SALTO"):
        typed(t14.TYPE_EDGE, t14.FAM_SCALE, "X", STEP2)
    with pytest.raises(t14.ValidationManifestError, match="AMBIGUO"):
        typed(t14.TYPE_EPISODE, t14.FAM_SCALE, "X", t14.Change(t14.FAM_SCALE, t14.SHAPE_EPISODE, 3.0, 0.40, 0.42))
    with pytest.raises(t14.ValidationManifestError, match="exactamente uno"):
        manifest_with(type_specs=((t14.TYPE_STEP, t14.FAM_SCALE, wspec("X", (STEP2, t14.Change(t14.FAM_CLOCK, t14.SHAPE_STEP, 3.0, 0.5)))),))


def test_power_controls_must_inject_their_declared_family():
    with pytest.raises(t14.ValidationManifestError, match="unico cambio de RELOJ"):
        manifest_with(power_specs=((t14.FAM_CLOCK, t14.POWER_LEVEL_DOUBLE, wspec("POT", (STEP2,))),))
    with pytest.raises(t14.ValidationManifestError, match="nivel desconocido"):
        manifest_with(power_specs=((t14.FAM_SCALE, "TRIPLE", wspec("POT", (STEP2,))),))


def test_positive_on_undeclared_background_is_rejected_unless_stress_is_predeclared():
    off = wspec("P_SALTO", (STEP2,), t_df=6.0)                                # colas distintas de las del protocolo
    with pytest.raises(t14.ValidationManifestError, match="se aparta del protocolo"):
        manifest_with(type_specs=((t14.TYPE_STEP, t14.FAM_SCALE, off),))
    ok = manifest_with(type_specs=((t14.TYPE_STEP, t14.FAM_SCALE, off),), stress_specs=(("COLAS_6", wspec("ESTRES_COLAS_6", t_df=6.0)),))
    assert "ESTRES:COLAS_6" in ok.spec_hashes()


def test_contradictory_spec_is_rejected_even_when_executed_worlds_match_its_hash():
    """Los mundos se ejecutan exactamente con la especificacion (huella valida), pero esa especificacion contradice lo
    declarado: el manifiesto no puede construirse y, por tanto, no hay validacion."""
    bad = wspec("P_SALTO", (TREND,))
    worlds = results(t14.ROLE_TEST, "P_SALTO", 2, (TREND,), first=9000)
    assert all(r.world_spec_sha256 == t14.world_spec_sha256(bad) for r in worlds)
    with pytest.raises(t14.ValidationManifestError, match="especificacion contradictoria"):
        manifest_with(type_specs=((t14.TYPE_STEP, t14.FAM_SCALE, bad),))


# --- coherencia entre fases ----------------------------------------------------

def test_calibration_only_accepts_predeclared_protocol_worlds(calibrated):
    neg = calibrated[0]
    undeclared = results(t14.ROLE_CALIBRATION, "NEGATIVO_NUEVO", 3, first=20)
    with pytest.raises(t14.ValidationManifestError, match="sin especificacion predeclarada"):
        t14.calibrate_thresholds(neg + undeclared, SMALL)
    drifted = results(t14.ROLE_CALIBRATION, "NEGATIVO", 3, first=7000, daily_phi=0.97)
    with pytest.raises(t14.ValidationManifestError, match="especificacion distinta"):
        t14.calibrate_thresholds(neg + drifted, SMALL)
    two = manifest_with(negative_specs=SMALL.negative_specs + (("NEGATIVO_B", wspec("NEGATIVO_B")),))
    with pytest.raises(t14.ValidationManifestError, match="faltan mundos de las variantes negativas"):
        t14.calibrate_thresholds(neg, two)


def test_predeclared_stress_variant_is_accepted_in_calibration(calibrated):
    neg = calibrated[0]
    stress = results(t14.ROLE_CALIBRATION, "ESTRES_PERSISTENTE", 10, daily_phi=0.9)
    m = manifest_with(stress_specs=(("PERSISTENCIA_ALTA", wspec("ESTRES_PERSISTENTE", daily_phi=0.9)),))
    thr = t14.calibrate_thresholds(neg + stress, m, target=0.05)
    info = dict(thr.calibration_info)
    assert info["manifiesto_sha256"] == m.sha256() and t14.world_spec_sha256(wspec("ESTRES_PERSISTENTE", daily_phi=0.9)) in info["especificaciones_calibracion"]


def test_validation_and_type_calibration_require_the_calibration_manifest(calibrated, val_worlds):
    neg, thr, full = calibrated
    assert dict(full.calibration_info)["manifiesto_sha256"] == SMALL.sha256()
    other = manifest_with(min_power_replicates=11)                                  # otro protocolo (otra huella)
    with pytest.raises(t14.ValidationManifestError, match="otro manifiesto"):
        t14.validate_procedure(full, other, {"NEGATIVO_A": val_worlds["neg"]})
    with pytest.raises(t14.ValidationManifestError, match="otro manifiesto"):
        t14.calibrate_type_margin(thr, [], other)
    det = manifest_with(detector_spec=replace(SPEC, max_lag=12))
    with pytest.raises(t14.ValidationManifestError):
        t14.validate_procedure(full, det, {"NEGATIVO_A": val_worlds["neg"]})


def test_final_decisions_keep_thresholds_and_manifest_traceability(calibrated, val_worlds):
    full = calibrated[2]
    val = t14.validate_procedure(full, SMALL, {"NEGATIVO_A": val_worlds["neg"]})
    r = val_worlds["neg"][0]
    dec = t14.final_decisions(r.stats, full, val)
    tr = dec["trazabilidad"]
    assert tr["umbrales_sha256"] == full.sha256 == tr["validacion_umbrales_sha256"]
    assert tr["manifiesto_calibracion_sha256"] == SMALL.sha256() == tr["manifiesto_validacion_sha256"]
    assert (tr["alpha"], tr["k_star"], tr["margen_tipo"]) == (full.alpha, full.k_star, full.type_margin)
    assert t14.final_decisions(r.stats, full, None)["trazabilidad"]["umbrales_sha256"] == full.sha256
    foreign = replace(val, manifest_sha256="f" * 64)
    with pytest.raises(t14.ValidationManifestError, match="manifiesto distinto"):
        t14.final_decisions(r.stats, full, foreign)
