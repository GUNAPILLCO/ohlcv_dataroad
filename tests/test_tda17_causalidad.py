"""Pruebas SINTETICAS de la auditoria causal de TH37 (E02, D17-1a; src/ohlcv_dataroad/ingest/tda17_causalidad.py).

Sin datos de MNQ, sin holdout. Se comprueba que el motor real no da discrepancias y que fugas INYECTADAS (con
monkeypatch, solo en la prueba) se detectan con igualdad exacta: dependencia del final del array (solo la detecta el
truncamiento), del contenido futuro (ambos modos), de peso minimo (1e-15, invisible para la tolerancia por defecto
de pandas), en D7 (truncamiento global), en las variables de TDA-04, en el origen de los parametros y en el roll.
"""
from __future__ import annotations

import copy
import math
import re

import numpy as np
import pandas as pd
import pytest

from ohlcv_dataroad.ingest import tda17_analysis as an
from ohlcv_dataroad.ingest import tda17_causalidad as ca
from ohlcv_dataroad.ingest import tda17_enmiendas_e02 as m2
from ohlcv_dataroad.ingest import tda17_indicators as ind
from ohlcv_dataroad.ingest import tda17_primitives as pr
from tda17_synthetic import make_frame


@pytest.fixture(scope="module")
def cfg():
    return m2.effective_document_e02(borrador=True)


@pytest.fixture(scope="module")
def data(cfg):
    df = make_frame().drop(columns=[ind.SYNTHETIC_MARKER])
    full = ind._compute(df, cfg)
    return df, full, ca.contract_cut_plan(df, *full, cfg)


def _subplan(plan, cuts_per_contract=4):
    return [{**p, "cortes": dict(list(p["cortes"].items())[:cuts_per_contract]), "no_aplicables": []} for p in plan]


def _plan_at(plan, contrato, cuts):
    p = next(x for x in plan if x["contrato"] == contrato)
    return [{**p, "cortes": {c: ["PRUEBA"] for c in cuts}, "no_aplicables": []}]


# =========================================================================== comparacion exacta
def test_compare_arrays_is_bit_exact():
    assert ca.compare_arrays(np.array([1.0, np.nan]), np.array([1.0, np.nan])) == (0, 0.0)
    assert ca.compare_arrays(np.array([0.0]), np.array([-0.0]))[0] == 1               # bits distintos
    n, mad = ca.compare_arrays(np.array([1.0, 2.0]), np.array([1.0, 2.0 + 1e-15]))
    assert n == 1 and 0 < mad < 1e-14
    assert ca.compare_arrays(np.array([1.0, np.nan]), np.array([1.0, 3.0])) == (1, 0.0) or \
        ca.compare_arrays(np.array([1.0, np.nan]), np.array([1.0, 3.0]))[0] == 1
    assert ca.compare_arrays(np.array([1.0], np.float32), np.array([1.0], np.float32)) == (0, 0.0)
    assert ca.compare_arrays(np.array([1.0], np.float32), np.array([1.0]))[0] == 1         # tipos distintos
    s = pd.Series(["CONS", None, "FJ"], dtype="string")
    assert ca.compare_arrays(s, s.copy())[0] == 0
    assert ca.compare_arrays(s, pd.Series(["CONS", "HI", "FJ"], dtype="string"))[0] == 1
    assert ca.compare_arrays(np.array([True, False]), np.array([True, True]))[0] == 1


def test_audit_section_must_be_the_implemented_one(cfg):
    assert ca.audit_config(cfg)["tabla"]["campos"] == ca.FIELDS
    bad = copy.deepcopy(cfg)
    bad["analisis"]["auditoria_causalidad"]["futuro"]["igualdad"] = "TOLERANCIA_1E-12"
    with pytest.raises(ind.ProtocolNotAuthorizedError, match="^auditoria causal distinta de la implementada$"):
        ca.audit_config(bad)


# =========================================================================== plan de cortes
def test_cut_plan_events_neighbourhood_quantiles_and_random_cuts(data, cfg):
    df, full, plan = data
    assert [p["contrato"] for p in plan] == ["MNQ_A", "MNQ_B"]
    for i, p in enumerate(plan):
        n, cuts = p["barras"], p["cortes"]
        assert all(2 <= c <= n - 1 for c in cuts)
        assert "BARRA_2" in cuts[2] and "BARRA_3" in cuts[3] and "BARRA_3+1" in cuts[4]
        for q in (25, 50, 75):
            c = math.ceil(q / 100 * n)
            assert f"CUANTIL_{q}" in cuts[c] and f"CUANTIL_{q}-1" in cuts[c - 1] and f"CUANTIL_{q}+1" in cuts[c + 1]
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(3717).spawn(2)[i]))
        for k, c in enumerate(rng.integers(2, n, size=10).tolist()):
            assert f"ALEATORIO_{k + 1}" in cuts[c]
        motives = {m for ms in cuts.values() for m in ms}
        assert {"PRIMER_FJL", "ULTIMO_FJL", "PRIMER_HI_CORTO", "ULTIMO_HI_LARGO"} <= motives
        assert any(m.startswith("ELEGIBLE_K1343_") for m in motives)                      # K_max alcanzado
        assert any(m.startswith("ACTIVACION_A5_") for m in motives) and any(m.startswith("ACTIVACION_A8_") for m in motives)
        assert any(m.startswith("PRIMER_CAMBIO_DIR_A11_") for m in motives)
        assert any(m.startswith("VV_PERDIDA_") for m in motives) and any(m.startswith("VV_RECUPERACION_") for m in motives)
        na = {(c, m.split(" ")[0], r) for c, m, r in p["no_aplicables"]}
        assert (1, "BARRA_1", "CONTRATO_CORTO") in na and (n, "ULTIMA_BARRA", "CONTRATO_CORTO") in na
    na_b = {m for c, m, r in plan[1]["no_aplicables"] if r == "EVENTO_INEXISTENTE"}
    assert {"PRIMER_FJ", "ULTIMO_FJ"} <= na_b                                              # MNQ_B no tiene FJ: sin sustituto
    assert ca.contract_cut_plan(df, *full, cfg) == plan                                    # determinista


def test_short_contract_marks_k_max_as_not_applicable(cfg):
    df = make_frame(contracts=(("MNQ_C", 1),), bars_per_day=400, dense_gaps=None).drop(columns=[ind.SYNTHETIC_MARKER])
    full = ind._compute(df, cfg)
    (p,) = ca.contract_cut_plan(df, *full, cfg)
    assert any(m.startswith("ELEGIBLE_K1343_") and r == "CONTRATO_CORTO" and c is None for c, m, r in p["no_aplicables"])
    assert all(not m.startswith("ELEGIBLE_K1343_") for ms in p["cortes"].values() for m in ms)


# =========================================================================== componente 1: futuro
def test_engine_is_causal_at_the_sampled_cuts(data, cfg):
    df, full, plan = data
    sub = _subplan(plan, 6)
    for p, q in zip(sub, plan):
        p["no_aplicables"] = q["no_aplicables"]
    tab = ca.audit_table(ca.audit_future(df, cfg, sub, full))
    assert set(tab["resultado"]) == {"CONFORME", "NO_APLICABLE"}
    assert set(tab["prueba"]) == {"TRUNCAMIENTO", "PERTURBACION_FUTURA"}
    n_cols = full[0].shape[1] + full[1].shape[1] + len(ca.TDA04_VARIABLES) + len(ind.VALIDITY_COLUMNS) + 1
    assert (tab["resultado"] == "CONFORME").sum() == 2 * 6 * 2 * n_cols
    na = tab[tab["resultado"] == "NO_APLICABLE"]
    assert na["discrepancias"].isna().all() and (na["filas_comparadas"] == 0).all()
    assert ca.causal_consequences(tab) == {"series_excluidas": [], "bloqueos": []}


def test_perturbation_keeps_the_skeleton_and_the_tick_grid(data):
    df = data[0]
    alt = ca.perturb_after(df, 100)
    assert alt.iloc[:100].equals(df.iloc[:100])
    keep = [c for c in df.columns if c not in ("open", "high", "low", "close", "volume")]
    assert alt[keep].equals(df[keep])
    ohlc = ["open", "high", "low", "close"]
    assert (alt[ohlc].iloc[100:] != df[ohlc].iloc[100:]).any(axis=1).all()                 # toda barra posterior cambia
    assert ((alt[ohlc] / pr.TICK) % 1 == 0).all().all()                                    # en la rejilla
    for shape in (lambda f: f["high"] - f["low"], lambda f: f["close"] - f["open"], lambda f: f["high"] - f["close"]):
        assert (shape(alt).iloc[100:] != shape(df).iloc[100:]).mean() >= 0.5              # cambia la FORMA, no solo el nivel
    assert (alt["high"] >= alt[["open", "close"]].max(axis=1)).all() and (alt["low"] <= alt[["open", "close"]].min(axis=1)).all()
    assert (alt["volume"].iloc[100:] > df["volume"].iloc[100:]).all()


def _leak(monkeypatch, kind):
    orig = ind.CONTRACT_FUNCTIONS["E5"]

    def leaky(b, p):
        out = dict(orig(b, p))
        v = out["valor"].copy()
        if kind == "siguiente":                       # usa la barra siguiente
            v[:-1] = v[:-1] + 1e-15 * v[1:]
        elif kind == "final":                         # trata de forma especial el final del array
            v[-1] = np.nan
        elif kind == "global":                        # normaliza con la media del contrato completo
            v = v / np.nanmean(v)
        out["valor"] = v
        return out
    monkeypatch.setitem(ind.CONTRACT_FUNCTIONS, "E5", leaky)


@pytest.mark.parametrize("kind,trunc,pert", [("siguiente", True, True), ("final", True, False), ("global", True, True)])
def test_injected_future_leaks_are_detected_exactly(data, cfg, monkeypatch, kind, trunc, pert):
    df, _, plan = data
    _leak(monkeypatch, kind)
    full = ind._compute(df, cfg)
    tab = ca.audit_table(ca.audit_future(df, cfg, _plan_at(plan, "MNQ_A", [500, 1700]), full))
    bad = tab[tab["resultado"] == "DISCREPANCIA"]
    assert set(bad["serie"]) == {"E5_barra__valor"}
    assert set(bad["prueba"]) == {m for m, on in (("TRUNCAMIENTO", trunc), ("PERTURBACION_FUTURA", pert)) if on}
    cons = ca.causal_consequences(tab)
    assert cons == {"series_excluidas": ["E5_barra__valor"], "bloqueos": ["DISCREPANCIA_CAUSAL: E5_barra__valor"]}


def test_tiny_leak_is_invisible_to_the_default_pandas_tolerance_but_not_to_the_audit(data, cfg, monkeypatch):
    df, full0, plan = data
    _leak(monkeypatch, "siguiente")
    full = ind._compute(df, cfg)
    s_t, _ = ind._compute(df.iloc[:800].reset_index(drop=True), cfg)
    pd.testing.assert_series_equal(full[0]["E5_barra__valor"].iloc[:800], s_t["E5_barra__valor"])   # pandas lo acepta
    tab = ca.audit_table(ca.audit_future(df, cfg, _plan_at(plan, "MNQ_A", [800]), full))
    assert (tab.loc[tab["serie"] == "E5_barra__valor", "resultado"] == "DISCREPANCIA").all()


def test_d7_is_audited_with_global_truncation(aligned, cfg, monkeypatch):
    df, full0 = aligned
    plan = ca.contract_cut_plan(df, *full0, cfg)
    orig = ind.relative_volume

    def leaky(d, k, days):
        rv, cross = orig(d, k, days)
        return rv + 1e-12 * len(d), cross                    # depende de la longitud total de la entrada
    monkeypatch.setattr(ind, "relative_volume", leaky)
    full = ind._compute(df, cfg)
    assert np.isfinite(full[0]["D7_k5__rvol_log"].to_numpy()[:1200]).any()
    tab = ca.audit_table(ca.audit_future(df, cfg, _plan_at(plan, "MNQ_B", [300]), full))
    bad = tab[tab["resultado"] == "DISCREPANCIA"]
    d7s, _ = ca._d7_columns(cfg)
    live = {c for c in d7s if np.isfinite(full[0][c].to_numpy()[:ind.contract_segments(df)[1][0] + 300]).any()}
    assert live == {"D7_k5__rvol_log"}                                                     # k20 sin J(t) completo: NaN
    assert set(bad["serie"]) == live and set(bad["prueba"]) == {"TRUNCAMIENTO"}


def test_tda04_variables_are_audited(data, cfg, monkeypatch):
    df, full, plan = data
    orig = ca.build_analysis_variables

    def leaky(canonical, validity):
        out = orig(canonical, validity)
        out["log_co"] = out["log_co"] + 1e-15 * out["log_hl"].shift(-1).fillna(0.0)
        return out
    monkeypatch.setattr(ca, "build_analysis_variables", leaky)
    tab = ca.audit_table(ca.audit_future(df, cfg, _plan_at(plan, "MNQ_A", [1200]), full))
    bad = tab[tab["resultado"] == "DISCREPANCIA"]
    assert set(bad["serie"]) == {"log_co"} and set(bad["prueba"]) == {"TRUNCAMIENTO", "PERTURBACION_FUTURA"}


# =========================================================================== componente 2: origen de parametros
def test_parameter_origin_table_and_input_independence(data, cfg, monkeypatch):
    df, full, _ = data
    tab = ca.audit_table(ca.audit_parameter_origin(df, cfg, full))
    assert tab[["parametro", "origen", "etiqueta"]].values.tolist() == [list(r) for r in ca.PARAMETER_ORIGINS]
    assert (tab["resultado"] == "CONFORME").all() and (tab["prueba"] == "ORIGEN_PARAMETRO").all()
    orig = ind._bars

    def leaky(d, a, z):                                       # una serie lee una variable de TDA-04 (no permitido)
        b = orig(d, a, z)
        b.v = b.v + np.nan_to_num(d["log_co"].to_numpy(float)[a:z]) * 1e-9
        return b
    monkeypatch.setattr(ind, "_bars", leaky)
    full2 = ind._compute(df, cfg)
    tab = ca.audit_table(ca.audit_parameter_origin(df, cfg, full2))
    assert (tab.loc[tab["parametro"].notna(), "resultado"] == "DISCREPANCIA").all()
    assert tab.loc[tab["parametro"].isna(), "serie"].notna().all()                         # series afectadas listadas


# =========================================================================== componente 3: roll
@pytest.fixture(scope="module")
def aligned(cfg):
    df = make_frame(contracts=(("MNQ_A", 6), ("MNQ_B", 6)), bars_per_day=150, aligned=True, dense_gaps=None)
    df = df.drop(columns=[ind.SYNTHETIC_MARKER])
    return df, ind._compute(df, cfg)


def test_roll_price_and_volume_invariance(aligned, cfg):
    df, full = aligned
    tab = ca.audit_table(ca.audit_roll(df, cfg, full))
    assert (tab["resultado"] == "CONFORME").all()
    assert set(tab["prueba"]) == {"ROLL_PRECIO", "ROLL_VOLUMEN"}
    motives = set(tab.loc[tab["prueba"] == "ROLL_PRECIO", "motivo_corte"])
    assert {"PRECIO_MULTIPLICATIVA", "PRECIO_ADITIVA", "PRECIO_ARBITRARIA", "ENTRADA_SIN_AJUSTES"} <= motives
    cross = full[1]["D7_k5__rvol_cruza_roll"].to_numpy(bool)
    assert cross.any()                                                                     # la prueba de volumen es efectiva
    alt = df.copy()
    rows = np.arange(len(df))
    a, z = ind.contract_segments(df)[1]
    other = (rows < a) | (rows >= z)
    alt["volume"] = np.where(other, df["volume"] + rows % 5 + 1, df["volume"])
    s2, _ = ind._compute(alt, cfg)
    changed = full[0]["D7_k5__rvol_log"].to_numpy()[a:z] != s2["D7_k5__rvol_log"].to_numpy()[a:z]
    assert changed.any() and not (changed & ~cross[a:z]).any()                             # solo filas rvol_cruza_roll


def test_roll_adjustment_leak_is_detected(aligned, cfg, monkeypatch):
    df, _ = aligned
    orig = ind._bars

    def adjusted(d, a, z):                                    # apertura enlazada con el cierre del contrato anterior
        b = orig(d, a, z)
        if a > 0:
            b.o = b.o.copy()
            b.o[0] = d["close"].to_numpy(float)[a - 1]
        return b
    monkeypatch.setattr(ind, "_bars", adjusted)
    full = ind._compute(df, cfg)
    tab = ca.audit_table(ca.audit_roll(df, cfg, full))
    bad = tab[tab["resultado"] == "DISCREPANCIA"]
    assert set(bad["prueba"]) == {"ROLL_PRECIO"} and set(bad["contrato"]) == {"MNQ_B"}
    assert "E1_barra__cuerpo" in set(bad["serie"]) or not bad.empty


def test_input_with_adjusted_columns_is_rejected(aligned, cfg):
    df, full = aligned
    with pytest.raises(ind.ProtocolNotAuthorizedError) as info:
        ca.audit_roll(df.assign(close_ajustado=df["close"]), cfg, full)
    assert str(info.value) == "la entrada contiene columnas no declaradas (posibles ajustes): ['close_ajustado']"
    with pytest.raises(ValueError, match="^fila de auditoria con prueba o resultado no declarados$"):
        ca.audit_table([ca._row("OTRA", None, None, None, None, 0, 0, 0.0, "CONFORME")])


# =========================================================================== auditoria completa
@pytest.fixture(scope="module")
def small_audit(cfg):
    df = make_frame(contracts=(("MNQ_A", 2), ("MNQ_B", 2)), bars_per_day=200, dense_gaps=(40, 80))
    df = df.drop(columns=[ind.SYNTHETIC_MARKER])
    return df, ca.run_causal_audit(df, cfg)


def test_full_causal_audit_on_a_small_frame(small_audit):
    _, tab = small_audit
    assert list(tab.columns) == ca.FIELDS
    assert set(tab["resultado"]) == {"CONFORME", "NO_APLICABLE"}
    assert set(tab["prueba"]) == set(ca.TESTS)
    assert (tab.loc[tab["resultado"] == "NO_APLICABLE", "motivo_corte"].str.contains(r"\[(?:CONTRATO_CORTO|EVENTO_INEXISTENTE)\]")).all()
    assert ca.causal_consequences(tab)["bloqueos"] == []


def test_catalog_config_must_be_the_phase1_catalog(cfg):
    ca.check_catalog_config(cfg)
    bad = copy.deepcopy(cfg)
    bad["catalogo"][0]["configuraciones"][0]["params"] = {**bad["catalogo"][0]["configuraciones"][0]["params"], "n": 999}
    with pytest.raises(ind.ProtocolNotAuthorizedError, match=r"^el catalogo de la configuracion no es el de la fase 1 \(E01\)$"):
        ca.check_catalog_config(bad)



# =========================================================================== C2: verificacion de la auditoria (barrera TH37)
def _rejects(exc_type, pattern, fn, *args):
    with pytest.raises(exc_type) as info:
        fn(*args)
    assert type(info.value) is exc_type and re.fullmatch(pattern, str(info.value)), str(info.value)


def test_complete_audit_is_exactly_the_predeclared_plan(small_audit, cfg):
    df, tab = small_audit
    expected = ca.expected_audit_rows(df, cfg)
    assert len(expected) == len(tab) and {ca._key(r) for r in tab.to_dict("records")} == set(expected)
    assert ca.verify_causal_audit(tab, df, cfg) == {"series_excluidas": [], "bloqueos": []}


def test_absent_audit_is_rejected(small_audit, cfg):
    df, tab = small_audit
    msg = re.escape("[TH37_AUSENTE] no hay auditoria causal de la entrada: TH35 no puede publicarse")
    for bad in (None, tab.iloc[0:0], [] ):
        _rejects(ind.ProtocolNotAuthorizedError, msg, ca.verify_causal_audit, bad, df, cfg)


def test_incomplete_audits_are_rejected(small_audit, cfg):
    df, tab = small_audit
    _rejects(ind.ProtocolNotAuthorizedError, r"\[TH37_INCOMPLETA\] faltan 1 filas del plan de la auditoria \(p\. ej\. .*\)",
             ca.verify_causal_audit, tab.drop(index=tab.index[100]), df, cfg)
    no_roll = tab[tab["prueba"] != "ROLL_PRECIO"]
    _rejects(ind.ProtocolNotAuthorizedError, r"\[TH37_INCOMPLETA\] faltan \d+ filas del plan.*",
             ca.verify_causal_audit, no_roll, df, cfg)
    _rejects(ind.ProtocolNotAuthorizedError, re.escape("[TH37_INCOMPLETA] la tabla de auditoria no tiene el esquema de E02"),
             ca.verify_causal_audit, tab.drop(columns=["max_abs_diff"]), df, cfg)
    na = tab.copy()                                                          # comparacion marcada como NO_APLICABLE
    i = na.index[na["resultado"] == "CONFORME"][0]
    na.loc[i, "resultado"] = "NO_APLICABLE"
    _rejects(ind.ProtocolNotAuthorizedError, r"\[TH37_INCOMPLETA\] 1 filas con un resultado incoherente con el plan .*",
             ca.verify_causal_audit, na, df, cfg)
    other = make_frame(contracts=(("MNQ_A", 2), ("MNQ_B", 2)), bars_per_day=180, dense_gaps=(40, 80))
    _rejects(ind.ProtocolNotAuthorizedError, r"\[TH37_INCOMPLETA\] faltan \d+ filas del plan.*",   # auditoria de OTRA entrada
             ca.verify_causal_audit, tab, other.drop(columns=[ind.SYNTHETIC_MARKER]), cfg)


def test_repeated_or_foreign_rows_are_rejected(small_audit, cfg):
    df, tab = small_audit
    dup = pd.concat([tab, tab.iloc[[5]]], ignore_index=True)
    _rejects(ind.ProtocolNotAuthorizedError, re.escape("[TH37_FUERA_DEL_PLAN] filas repetidas o ajenas al plan: 1 repetidas, 0 ajenas"),
             ca.verify_causal_audit, dup, df, cfg)
    extra = pd.concat([tab, pd.DataFrame([ca._row("TRUNCAMIENTO", "MNQ_A", 7, "INVENTADO", "E5_barra__valor", 1, 0, 0.0,
                                                   "CONFORME")])], ignore_index=True)
    _rejects(ind.ProtocolNotAuthorizedError, re.escape("[TH37_FUERA_DEL_PLAN] filas repetidas o ajenas al plan: 0 repetidas, 1 ajenas"),
             ca.verify_causal_audit, extra, df, cfg)


def test_discrepancy_blocks_th35(small_audit, cfg):
    df, tab = small_audit
    bad = tab.copy()
    for serie in ("E5_barra__valor", "D7_k5__rvol_log"):
        i = bad.index[(bad["prueba"] == "PERTURBACION_FUTURA") & (bad["serie"] == serie)][0]
        bad.loc[i, ["resultado", "discrepancias"]] = ["DISCREPANCIA", 3]
    _rejects(an.BlockedUntilExplainedError,
             re.escape("[TH37_DISCREPANCIA] TH35 BLOQUEADA_HASTA_EXPLICAR: series con discrepancia causal "
                       "['D7_k5__rvol_log', 'E5_barra__valor']"), ca.verify_causal_audit, bad, df, cfg)
