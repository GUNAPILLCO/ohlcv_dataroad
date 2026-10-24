"""Pruebas SINTETICAS del motor estadistico de TDA-17, fase 2 (src/ohlcv_dataroad/ingest/tda17_analysis.py).

Sin datos de MNQ, sin holdout, sin data/raw: paneles generados con semillas fijas y marcos del constructor
sintetico de la fase 1 (``tda17_synthetic``). Las referencias independientes (numpy.corrcoef, scipy.stats,
bucles explicitos) se calculan aqui, no en el modulo. La configuracion es la efectiva E01 + E02 en modo BORRADOR
(E02 PENDIENTE_AUDITORIA): sirve para probar la mecanica, nunca para producir resultados.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ohlcv_dataroad.ingest import tda17_analysis as an
from ohlcv_dataroad.ingest import tda17_enmiendas as enm
from ohlcv_dataroad.ingest import tda17_enmiendas_e02 as m2
from ohlcv_dataroad.ingest import tda17_indicators as ind
from ohlcv_dataroad.ingest import tda17_primitives as pr
from tda17_synthetic import make_frame, rebuild_tda04


@pytest.fixture(scope="module")
def cfg():
    return m2.effective_document_e02(borrador=True)


# =========================================================================== paneles y referencias
def test_panel_counts_and_b_definition(cfg):
    assert an.check_panel_counts(cfg) == {"P": 134, "B": 123, "L": 16}
    p, b, out = an.p_columns(cfg), an.b_columns(cfg), an.out_of_b(cfg)
    assert len(out) == len(set(out)) == 11 and set(b) == set(p) - set(out)
    assert b == [c for c in p if c not in out]                              # mismo orden que P
    assert an.reference_columns(cfg) == ["r_1m", "abs_r_1m", "log_hl", "log_co", "log_oc_prev", "ln_volume"]
    assert set(an.reference_columns(cfg)) <= set(p)


def test_l_counterparts_exist_in_p(cfg):
    cp = an.l_counterparts(cfg)
    assert set(cp) == set(an.l_columns(cfg)) and len(cp) == 16
    assert set(cp.values()) <= set(an.p_columns(cfg))
    assert cp["A6_f12_s26_g9__conv_hist"] == "A6_f12_s26_g9__n_hist" and cp["E5_barra__valor"] == "log_hl"
    assert cp["C1_n240__atr_ticks"] == "C2_n240__ln_natr" and cp["B2_n60__mom_ticks"] == "B1_n60__roc"


def test_build_panel_b_requires_the_official_p(cfg):
    p = pd.DataFrame(np.zeros((3, 134)), columns=an.p_columns(cfg))
    assert list(an.build_panel_b(p, cfg).columns) == an.b_columns(cfg)
    with pytest.raises(ValueError):
        an.build_panel_b(p[p.columns[::-1]], cfg)


@pytest.fixture(scope="module")
def engine(cfg):
    """Marco sintetico con huecos internos (cortos, largos, tramo denso), FJ, FJL y un roll."""
    df = make_frame()
    series, marks = ind.compute_catalog_synthetic(df, cfg)
    base = df.drop(columns=[ind.SYNTHETIC_MARKER])
    return base, series, marks, ind._build_panel_p(series, base, cfg)


def test_tda04_references_are_taken_verbatim(cfg, engine):
    df, _, _, panel = engine
    for r in ["r_1m", "abs_r_1m", "log_hl", "log_co", "log_oc_prev"]:
        np.testing.assert_array_equal(panel[r].to_numpy(), df[r].to_numpy())   # sin recalcular
    with np.errstate(divide="ignore"):
        expected = np.where(df["volume"] > 0, np.log(df["volume"].to_numpy()), np.nan)
    np.testing.assert_array_equal(panel["ln_volume"].to_numpy(), expected)


# =========================================================================== entrada, poblacion, calentamiento
def test_expected_inputs_are_checked_before_computing(cfg, engine):
    df = engine[0]
    with pytest.raises(ind.ProtocolNotAuthorizedError, match="no es la esperada"):
        an.check_expected_inputs(df, cfg)
    c2 = copy.deepcopy(cfg)
    c2["entradas"]["esperado"] = {"filas": len(df), "retornos_validos": int(df["r_1m_valid"].sum()),
                                  "jornadas": df["trading_date"].nunique(), "contratos": 2, "rolls": 1}
    assert an.check_expected_inputs(df, c2)["rolls"] == 1


def test_population_is_complete_cases_of_p(engine):
    panel = engine[3]
    ok = an.complete_cases(panel)
    np.testing.assert_array_equal(ok, panel.notna().all(axis=1).to_numpy() & np.isfinite(panel.to_numpy()).all(axis=1))
    rep = an.population_report(panel)
    assert rep.attrs["casos_completos"] == ok.sum() and rep.attrs["perdidas"] == len(panel) - ok.sum()
    row = rep.set_index("columna").loc["A2_n240__dist"]
    assert row["no_finitos"] == (~np.isfinite(panel["A2_n240__dist"])).sum()


def _window_count(cons, n, t):
    return sum(cons[i] for i in range(max(t - n, 0), t))                     # pares que terminan en t-n+1..t (base 1)


def test_warmup_registry_follows_the_three_levels(cfg, engine):
    df, series, _, _ = engine
    tab = an.warmup_table(df, series, cfg)
    assert set(tab["panel"]) == {"P", "L"} and len(tab) == 144 * 2           # 144 series x 2 contratos
    act, el, fin = tab["primera_activacion"], tab["primera_barra_elegible"], tab["primera_salida_finita"]
    sel = el.notna() & fin.notna()
    assert (el[sel] <= act[sel]).all() and (act[sel] <= fin[sel]).all()      # elegibilidad <= activacion <= finita
    exacta = tab[(tab["clase"] == "EXACTA") & (tab["K_teorico"] <= tab["barras_contrato"])]
    assert (exacta["primera_barra_elegible"] == exacta["K_teorico"]).all() and (exacta["retraso"] == 0).all()
    assert tab.loc[tab["clase"] == "JORNADAS", "retraso"].isna().all()       # REGISTRO_D7_RETRASO


def test_warmup_registry_against_independent_loops(cfg, engine):
    df, series, _, _ = engine
    tab = an.warmup_table(df, series, cfg).set_index(["serie", "contrato"])
    (a, z) = ind.contract_segments(df)[0]
    cons = df["r_1m_valid"].to_numpy(bool)[a:z].copy()
    cons[0] = False
    contract = df["contract"].iloc[a]
    u = 0
    first_u = {}
    for t in range(1, z - a + 1):                                           # contador CONS explicito
        u += int(cons[t - 1])
        for U in (77, 125, 153, 335, 502, 669):
            first_u.setdefault(U, t) if u >= U else None
    assert tab.loc[("B3_n14__rsi", contract), "primera_barra_elegible"] == first_u[77]
    assert tab.loc[("A8_n14__di_pos", contract), "primera_barra_elegible"] == first_u[153]
    assert tab.loc[("B9_rsi14_n14_k3_d3__k", contract), "primera_barra_elegible"] == first_u[77] + 17
    ok = [t for t in range(11, z - a + 1) if _window_count(cons, 10, t) >= pr.min_valid(10)]
    assert tab.loc[("B1_n10__roc", contract), "primera_barra_elegible"] == ok[0]
    adx = series["A8_n14__adx"].to_numpy()[a:z]
    assert tab.loc[("A8_n14__adx", contract), "primera_activacion"] == np.flatnonzero(np.isfinite(adx))[0] + 1
    e = int(tab.loc[("A2_n240__dist", contract), "primera_barra_elegible"])
    assert tab.loc[("A2_n240__dist", contract), "barras_no_cons_calentamiento"] == (~cons[1:e]).sum()


# =========================================================================== celdas
def test_cells_years_ny_and_segments_half_open(cfg):
    ts = pd.Series(pd.to_datetime(["2020-01-01 03:00", "2020-01-01 05:00", "2021-06-01 16:00", "2021-06-01 16:01",
                                   "2025-03-03 14:30", "2025-03-03 13:29"]))            # UTC
    cells = an.cell_definitions(ts, cfg)
    assert len(cells) == 1 + 7 + 7 + 49
    assert cells["ANIO_2019"]["estado"] == "NO_EVALUABLE" and cells["ANIO_2019"]["mascara"].tolist()[:2] == [True, False]
    assert cells["ANIO_2025"]["estado"] == "EVALUABLE_PARCIAL"
    # 2021-06-01 16:00 UTC = 12:00 NY (09:30-16:02); 2025-03-03 14:30 UTC = 09:30 NY (inicio incluido); 13:29 -> 08:29
    assert cells["SEG_09:30-16:02"]["mascara"].tolist() == [False, False, True, True, True, False]
    assert cells["SEG_03:00-08:30"]["mascara"].tolist() == [False, False, False, False, False, True]
    assert cells["SEG_00:00-02:00"]["mascara"].tolist() == [False, True, False, False, False, False]   # 00:00 NY
    assert cells["SEG_20:00-24:00"]["mascara"].tolist() == [True, False, False, False, False, False]   # 22:00 NY
    assert all(v["estado"] == "SOLO_DESCRIPTIVO" for k, v in cells.items() if "__SEG_" in k)
    with pytest.raises(ind.ProtocolNotAuthorizedError, match="anios no declarados"):
        an.cell_definitions(pd.Series(pd.to_datetime(["2018-06-01 12:00"])), cfg)


# =========================================================================== estadisticos de celda
def test_cell_stats_match_numpy_and_reject_degenerate_cells():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((5000, 7)) @ rng.standard_normal((7, 7))
    st = an.cell_stats(X, chunk=777)
    np.testing.assert_allclose(st.corr, np.corrcoef(X, rowvar=False), atol=1e-12)
    np.testing.assert_allclose(st.gram, (X - X.mean(0)).T @ (X - X.mean(0)), rtol=1e-10)
    with pytest.raises(ValueError, match="sin varianza"):
        an.cell_stats(np.c_[X[:, :2], np.ones(5000)])
    with pytest.raises(ValueError, match="menos de 2"):
        an.cell_stats(X[:1])
    Y = X.copy()
    Y[3, 2] = np.nan
    with pytest.raises(ValueError, match="no finitos"):
        an.cell_stats(Y)


def test_spearman_is_pearson_of_average_ranks_with_ties():
    rng = np.random.default_rng(2)
    X = np.round(rng.standard_normal((3000, 4)), 1)                         # muchos empates
    X[:, 3] = X[:, 0] ** 2
    ref = stats.spearmanr(X).statistic
    np.testing.assert_allclose(an.spearman(X), ref, atol=1e-12)


def test_s3_winsorization_bounds_and_clip(cfg):
    lo, hi = an.s3_bounds(cfg)
    assert (lo, hi) == (0.001, 0.999)
    rng = np.random.default_rng(3)
    X = rng.standard_t(2, (10000, 2))
    W = an.winsorize(X, lo, hi)
    q = np.quantile(X, [lo, hi], axis=0)
    np.testing.assert_array_equal(W, np.clip(X, q[0], q[1]))


# =========================================================================== R2 y exactitud (prototipo v4a)
def _v4a_panel():
    rng = np.random.default_rng(20261016)
    n = 20_000
    X, Y, W, V, U = rng.standard_normal((5, n))
    A, B = rng.uniform(0, 0.5, (2, n))
    cols = {"X": X, "Y": Y, "Z": X + Y, "A": A, "B": B, "C": 1.0 - A - B,
            "W": W, "N2": W + 1e-3 * rng.standard_normal(n), "U": U, "U2": U + 1e-7 * rng.standard_normal(n), "V": V}
    return pd.DataFrame(cols)


def _v4a_cfg(cfg):
    c = copy.deepcopy(cfg)
    c["identidades"]["lineales"] = [
        {"id": "I1", "panel": "P", "terminos": [[1, "Z"], [-1, "X"], [-1, "Y"]], "constante": 0, "fuera_de_B": "Z"},
        {"id": "I2", "panel": "P", "terminos": [[1, "A"], [1, "B"], [1, "C"]], "constante": 1, "fuera_de_B": "C"}]
    return c


def test_three_r2_methods_agree_without_unregistered_collinearity(cfg):
    P = _v4a_panel().drop(columns=["U", "U2"])
    tab = an.r2_table(an.cell_stats(P.to_numpy()), list(P.columns), None, 1e-10, 1e-8)
    assert tab["publicable"].all() and tab["dif_cruzada"].max() < 1e-8
    for j, c in enumerate(P.columns):                                       # referencia independiente: MCO sobre filas
        Mc = P.to_numpy() - P.to_numpy().mean(0)
        beta, *_ = np.linalg.lstsq(np.delete(Mc, j, 1), Mc[:, j], rcond=None)
        res = Mc[:, j] - np.delete(Mc, j, 1) @ beta
        assert abs(tab.loc[j, "r2"] - (1 - res @ res / (Mc[:, j] @ Mc[:, j]))) < 1e-8, c


def test_unregistered_collinearity_makes_other_r2_unpublishable(cfg):
    P = _v4a_panel()
    tab = an.r2_table(an.cell_stats(P.to_numpy()), list(P.columns), None, 1e-10, 1e-8).set_index("columna")
    assert not tab.loc["V", "publicable"]
    assert tab.loc["V", "etiqueta_publicacion"] == "R2_NO_PUBLICABLE_POR_COLINEALIDAD"
    assert tab.loc[["X", "Y", "Z", "A", "B", "C"], "publicable"].all()


def test_exactness_labels_by_registry_and_residual(cfg):
    P, c = _v4a_panel(), _v4a_cfg(cfg)
    tab = an.r2_table(an.cell_stats(P.to_numpy()), list(P.columns), None, 1e-10, 1e-8)
    res = an.identity_residuals(P, P, c, "P")
    assert res["I1"]["residuo_relativo"] < 1e-15 and res["I2"]["residuo_relativo"] < 1e-15
    lab = an.exactness_labels(dict(zip(tab["columna"], tab["r2"])), res, c).set_index("columna")["etiqueta"]
    assert (lab[["X", "Y", "Z", "A", "B", "C"]] == "EXACTA_ALGEBRAICA").all()
    assert lab["W"] == lab["N2"] == "CASI_EXACTA_COLINEAL"                  # 1 - R2 ~ 1e-6: nunca EXACTA
    assert lab["U"] == lab["U2"] == "ALERTA_DEPENDENCIA_NO_REGISTRADA"
    assert pd.isna(lab["V"])


def test_broken_registered_identity_is_an_error_that_blocks(cfg):
    P, c = _v4a_panel(), _v4a_cfg(cfg)
    P.loc[0, "Z"] += 1e-3
    res = an.identity_residuals(P, P, c, "P")
    lab = an.exactness_labels({k: 1.0 for k in P.columns}, res, c)
    assert set(lab.loc[lab["etiqueta"] == "ERROR_IDENTIDAD_REGISTRADA", "columna"]) == {"X", "Y", "Z"}
    assert any(b.startswith("ERROR_IDENTIDAD_REGISTRADA") for b in an.blocking_findings(lab, []))


def test_null_count_eigenvalue_convention_against_the_registry():
    P = _v4a_panel()
    counts = an.null_counts(an.cell_stats(P.to_numpy()), 1e-10)
    # 2 identidades + U/U2 (1 - R2 ~ 1e-14): nula para los autovalores (lambda ~ 1e-14) pero no para los valores
    # singulares (s ~ 1e-7). E02 (A1-a): solo los autovalores se comparan con el registro -> bloqueo hasta explicarlo
    # en una celda BLOQUEANTE; los singulares (2 = registro) solo se informan
    assert counts == {"autovalores": 3, "singulares": 2}
    chk = an.rank_check(counts, 2, "prototipo", "BLOQUEANTE")
    assert chk["estado"] == "BLOQUEAR_HASTA_EXPLICAR" and chk["singulares_igual_registro"] is True
    assert an.rank_check(counts, 2, "prototipo", "INFORMATIVA")["estado"] == "DIFIERE_INFORMATIVO"
    clean = an.null_counts(an.cell_stats(P.drop(columns=["U2"]).to_numpy()), 1e-10)
    assert an.rank_check(clean, 2, "prototipo", "BLOQUEANTE")["estado"] == "CONFORME"


# =========================================================================== P sintetico con el registro completo
def _registry_panel(cfg, n=2000, seed=17):
    """Columnas de B aleatorias; las 11 fuera de B se construyen con su identidad (como el test del protocolo)."""
    rng = np.random.default_rng(seed)
    cols = an.p_columns(cfg)
    out_b = {i["fuera_de_B"]: i for i in cfg["identidades"]["lineales"] if i["panel"] == "P"}
    data = {c: rng.standard_normal(n) for c in cols if c not in out_b}
    for c, ident in out_b.items():
        coef = dict((col, k) for k, col in ident["terminos"])[c]
        data[c] = (ident["constante"] - sum(k * data[col] for k, col in ident["terminos"] if col != c)) / coef
    panel = pd.DataFrame({c: data[c] for c in cols})
    df = pd.DataFrame({"high": np.full(n, 2.0), "low": np.ones(n), "r_1m_valid": np.ones(n, bool)})
    return panel, df


@pytest.fixture(scope="module")
def registry_cell(cfg):
    panel, df = _registry_panel(cfg)
    return panel, df, an.analyze_cell(panel, df, cfg, celda="GLOBAL")


def test_registered_identities_give_declared_ranks_and_exact_labels(cfg, registry_cell):
    _, _, out = registry_cell
    assert out["bloqueos"] == []
    assert {r["matriz"]: (r["autovalores"], r["singulares"], r["estado"]) for r in out["rango"]} == {
        "P_sin_centrar": (10, 10, "CONFORME"), "P_centrada": (11, 11, "CONFORME"), "B_centrada": (0, 0, "CONFORME")}
    lab = out["exactitud"].set_index("columna")["etiqueta"]
    involved = {col for i in cfg["identidades"]["lineales"] if i["panel"] == "P" for _, col in i["terminos"]}
    assert len(involved) == 32 and (lab[sorted(involved)] == "EXACTA_ALGEBRAICA").all()
    assert lab.drop(sorted(involved)).isna().all()
    r2 = out["r2_todas"].set_index("columna")
    assert (1 - r2.loc[sorted(involved), "r2"] <= 1e-9).all() and r2["publicable"].all()
    assert all(v["residuo_relativo"] <= 1e-12 for v in out["identidades"].values())
    assert out["dimension_B"]["p"] == 123 and out["dimension_P"]["p"] == 134
    assert np.sum(out["autovalores_P"] <= 1e-10 * out["autovalores_P"].max()) == 11


def test_r2_tda04_uses_only_the_six_references(cfg, registry_cell):
    panel, _, out = registry_cell
    tab = out["r2_tda04"].set_index("columna")
    assert tab.loc["log_oc_prev", "r2"] == pytest.approx(1.0, abs=1e-9)    # L11: r_1m - log_co - log_oc_prev = 0
    assert tab.loc["A1_n20__dist", "r2"] < 0.05                              # aleatoria frente a las referencias
    names = list(panel.columns)
    sets = an.tda04_predictors(names, cfg)
    assert len(sets["A1_n20__dist"]) == 6 and len(sets["r_1m"]) == 5 and names.index("r_1m") not in sets["r_1m"]


def test_unregistered_exact_dependency_alerts_and_blocks(cfg):
    panel, df = _registry_panel(cfg)
    panel["C7_n20__ln_sigma"] = 2.0 * panel["C8_n20__ln_sigma"] + 1.0
    out = an.analyze_cell(panel, df, cfg, celda="GLOBAL")
    lab = out["exactitud"].set_index("columna")["etiqueta"]
    assert lab["C7_n20__ln_sigma"] == lab["C8_n20__ln_sigma"] == "ALERTA_DEPENDENCIA_NO_REGISTRADA"
    estados = {r["matriz"]: r["estado"] for r in out["rango"]}
    assert estados["P_centrada"] == estados["B_centrada"] == "BLOQUEAR_HASTA_EXPLICAR"
    assert any("BLOQUEAR_HASTA_EXPLICAR" in b for b in out["bloqueos"])


def test_insufficient_population_is_rank_deficient_and_blocks(cfg):
    """Menos filas que columnas: no hay regla nueva de poblacion minima; el propio registro de rango bloquea."""
    panel, df = _registry_panel(cfg, n=100)
    out = an.analyze_cell(panel, df, cfg, celda="GLOBAL")
    assert {r["matriz"] for r in out["rango"] if r["estado"] != "CONFORME"} == {"P_sin_centrar", "P_centrada", "B_centrada"}
    assert out["bloqueos"]


def test_analyze_cell_rejects_rows_outside_the_population(cfg):
    panel, df = _registry_panel(cfg, n=300)
    panel.iloc[5, 3] = np.nan
    with pytest.raises(ValueError, match="casos completos"):
        an.analyze_cell(panel, df, cfg, celda="GLOBAL")


# =========================================================================== motor de la fase 1 de extremo a extremo
def _jitter_opens(df, shift=14500.0, seed=5):
    """El constructor sintetico abre siempre en el cierre anterior (log_oc_prev = 0 constante, sin varianza) y
    se mueve ~3e-5 por barra en terminos relativos (unas 10 veces menos que MNQ): se baja el nivel ``shift``
    puntos (volatilidad relativa ~1e-3), se desplaza la apertura +-2 ticks y se reconstruyen las variables con
    las MISMAS funciones de TDA-04."""
    rng = np.random.default_rng(seed)
    can = df[["timestamp", "source_file", "contract", "trading_date", "segment_id", "open", "high", "low", "close",
              "volume"]].copy()
    for k in ("open", "high", "low", "close"):
        can[k] = can[k] - shift
    can["open"] = can["open"] + pr.TICK * rng.integers(-2, 3, len(can))
    can["high"] = np.maximum(can["high"], can["open"])
    can["low"] = np.minimum(can["low"], can["open"])
    out = rebuild_tda04(can)
    out[ind.SYNTHETIC_MARKER] = True
    return out


def _engine_cell(cfg, shift):
    df = _jitter_opens(make_frame(contracts=(("MNQ_A", 25),), bars_per_day=300, aligned=True, seed=4, dense_gaps=None),
                       shift=shift)
    series, marks = ind.compute_catalog_synthetic(df, cfg)
    base = df.drop(columns=[ind.SYNTHETIC_MARKER])
    panel = ind._build_panel_p(series, base, cfg)
    ok = an.complete_cases(panel)
    return base, series, marks, panel, ok


@pytest.fixture(scope="module")
def engine_cell(cfg):
    """Un contrato alineado de 25 jornadas: RVOL k20 valido desde la jornada 21 y K = 1343 alcanzado."""
    return _engine_cell(cfg, 14500.0)


def test_engine_panel_registered_identities_are_exact(cfg, engine_cell):
    base, _, _, panel, ok = engine_cell
    assert ok.sum() > 600
    out = an.analyze_cell(panel[ok], base[ok], cfg, celda="GLOBAL")
    lab = out["exactitud"].set_index("columna")["etiqueta"]
    involved = {col for i in cfg["identidades"]["lineales"] if i["panel"] == "P" for _, col in i["terminos"]}
    assert (lab[sorted(involved)] == "EXACTA_ALGEBRAICA").all()
    assert not (lab == "ERROR_IDENTIDAD_REGISTRADA").any()
    assert all(v["residuo_relativo"] <= 1e-9 and v["filas"] > 0 for v in out["identidades"].values())
    rk = {r["matriz"]: r for r in out["rango"]}
    # valores singulares (informativos): exactamente el registro; autovalores (tau 1e-10, convencion de E02):
    # ademas relaciones casi lineales (N7, N18, N19) -> bloqueo hasta explicarlo si difieren del registro
    assert (rk["P_sin_centrar"]["singulares"], rk["P_centrada"]["singulares"], rk["B_centrada"]["singulares"]) == (10, 11, 0)
    assert all(r["autovalores"] >= r["singulares"] for r in out["rango"])
    assert bool(out["bloqueos"]) == any(r["autovalores"] != r["esperado"] for r in out["rango"])


def test_low_relative_volatility_turns_near_linear_relations_into_unregistered_nulls(cfg):
    """Con el nivel original (~15.000 puntos, volatilidad relativa ~3e-5 por barra) los terminos de segundo
    orden de ln(C/MA) frente a (C - MA)/C caen por debajo de tau: aparecen dependencias no registradas en A1, A3,
    A12 (N7) y A2, A6, A7 (N18, N19) y la etapa se bloquea. Nunca se etiquetan como EXACTA."""
    base, _, _, panel, ok = _engine_cell(cfg, 0.0)
    out = an.analyze_cell(panel[ok], base[ok], cfg, celda="GLOBAL")
    lab = out["exactitud"].set_index("columna")["etiqueta"]
    assert {"A1_n20__dist", "A3_n20__dist", "A12_n20__pend_rel"} <= set(lab[lab == "ALERTA_DEPENDENCIA_NO_REGISTRADA"].index)
    assert any("P_centrada" in b for b in out["bloqueos"])
    involved = {col for i in cfg["identidades"]["lineales"] if i["panel"] == "P" for _, col in i["terminos"]}
    assert set(lab[lab == "EXACTA_ALGEBRAICA"].index) == involved


def test_s1_union_rule_and_marks_inside_the_population(cfg, engine_cell):
    _, _, marks, panel, ok = engine_cell
    cols = an.p_columns(cfg)
    excl = an.s1_exclusion(marks, cfg, cols, rows=ok)
    windows = ind.mark_windows(cfg)
    manual = np.zeros(ok.sum(), bool)
    for c in cols:
        if c in windows:
            manual |= ind.series_discontinuity(marks, windows[c])["cruza_hueco_largo"][ok] > 0
    np.testing.assert_array_equal(excl, manual)
    assert excl.any()
    short = an.s1_exclusion(marks, cfg, ["A1_n20__dist"], rows=ok)
    assert short.any() and not short.all()                                  # una ventana corta solo cruza algunas FJL
    refs_only = an.s1_exclusion(marks, cfg, an.reference_columns(cfg) + ["D7_k5__rvol_log"], rows=ok)
    assert not refs_only.any()                                              # TDA-04 y RVOL: sin marcas de S1


def test_s1_effective_population_control_with_the_union_rule(cfg, engine_cell):
    """Jornadas de 300 barras separadas por FJL: con ventanas de hasta K = 1343 barras, la union de las 134
    columnas excluye TODA la poblacion; el control lo hace visible (no se cambia ningun umbral)."""
    _, _, marks, _, ok = engine_cell
    rep = an.s1_population_report(marks, cfg, an.p_columns(cfg), ok)
    union = rep.set_index("serie").loc["__UNION__", "excluye"]
    assert rep.attrs["poblacion"] == ok.sum() and union == ok.sum() - rep.attrs["poblacion_efectiva"]
    assert rep.attrs["poblacion_efectiva"] == 0 and not rep.attrs["calculable"]
    per = rep.set_index("serie").drop("__UNION__")["excluye"]
    assert per["A1_n20__dist"] < per["A2_n240__dist"] <= ok.sum()          # la exclusion crece con la ventana
    small = an.s1_population_report(marks, cfg, ["A1_n20__dist", "B1_n10__roc"], ok)
    assert small.attrs["calculable"] and 0 < small.attrs["fraccion_retenida"] < 1


def test_s1_rejects_nan_marks_inside_the_rows(cfg, engine_cell):
    _, _, marks, _, ok = engine_cell
    early = np.zeros(len(marks), bool)
    early[:5] = True                                                        # antes de K: marcas NaN
    with pytest.raises(ind.ProtocolNotAuthorizedError, match="contradice E01"):
        an.s1_exclusion(marks, cfg, ["A1_n240__dist"], rows=early)


def test_s1_report_and_exclusion_reject_the_same_nan_marks(cfg, engine_cell):
    """C1: una marca NaN en las filas seleccionadas nunca cuenta como 'sin hueco': el informe de poblacion y la
    exclusion rechazan exactamente la misma entrada invalida, con el mismo motivo."""
    _, _, marks, _, ok = engine_cell
    early = ok.copy()
    early[:5] = True                                                        # filas anteriores a K: marcas NaN
    cols = ["A1_n240__dist", "B1_n10__roc"]
    errs = []
    for fn in (an.s1_exclusion, an.s1_population_report):
        with pytest.raises(ind.ProtocolNotAuthorizedError, match="contradice E01") as exc:
            fn(marks, cfg, cols, early)
        errs.append(str(exc.value))
    assert errs[0] == errs[1] == "marca NaN de A1_n240__dist dentro de la poblacion: contradice E01"
    rep = an.s1_population_report(marks, cfg, cols, ok)                     # en la poblacion valida, coinciden
    union = an.s1_exclusion(marks, cfg, cols, ok)
    assert rep.set_index("serie").loc["__UNION__", "excluye"] == union.sum()


# --------------------------------------------------------------------------- C4: poblacion efectiva de S1
# Dos niveles de prueba:
#  - SUBCONJUNTOS (5 a 8 jornadas, ``_days_frame``): menos jornadas de las que exige RVOL k20, de modo que la
#    poblacion es la de las columnas de P SIN D7. Comprueban la MECANICA de la union de marcas, no la poblacion
#    oficial.
#  - POBLACION OFICIAL (28 jornadas, ``test_s1_on_the_official_population_...``): las 134 columnas, D7 k5 y k20
#    incluidas, y la poblacion CASOS_COMPLETOS_DE_P de ``complete_cases`` sin excluir ninguna columna.
def _days_inputs(cfg, lengths, bars_per_day=1380):
    """Un contrato con jornadas alineadas en la apertura del calendario; la jornada d conserva sus primeras
    ``lengths[d]`` barras. Jornada completa (1380 barras) -> FJ de 61 min con la siguiente; jornada truncada o fin
    de semana -> FJL. Sin HI. Variables recalculadas con las funciones de TDA-04. Devuelve (base, marcas, P)."""
    raw = make_frame(contracts=(("MNQ_A", len(lengths)),), bars_per_day=bars_per_day, aligned=True, hi_every=0,
                     long_hi_every=0, dense_gaps=None, seed=9)
    can = raw[["timestamp", "source_file", "contract", "trading_date", "segment_id", "open", "high", "low", "close",
               "volume"]]
    pos = can.groupby("trading_date", sort=False).cumcount().to_numpy()
    day = pd.factorize(can["trading_date"])[0]
    df = rebuild_tda04(can[pos < np.asarray(lengths)[day]].reset_index(drop=True))
    df[ind.SYNTHETIC_MARKER] = True
    series, marks = ind.compute_catalog_synthetic(df, cfg)
    base = df.drop(columns=[ind.SYNTHETIC_MARKER])
    return base, marks, ind._build_panel_p(series, base, cfg)


def _days_frame(cfg, lengths, bars_per_day=1380):
    """PRUEBA DE SUBCONJUNTO: poblacion = filas con las columnas de P SIN D7 finitas (RVOL k20 no tiene J(t)
    completo con 5 a 8 jornadas). No es la poblacion oficial CASOS_COMPLETOS_DE_P."""
    base, marks, panel = _days_inputs(cfg, lengths, bars_per_day)
    cols = [c for c in an.p_columns(cfg) if not c.startswith("D7_")]
    rows = np.isfinite(panel[cols].to_numpy()).all(axis=1)
    return base, marks, cols, rows


def _brute_union(base, cfg, cols, rows):
    """Referencia independiente de la union: como todas las ventanas terminan en t, la fila se excluye si hay un
    par largo (FJL o HI > 90 min) entre los W_max pares que terminan en t (sin cruzar el reinicio)."""
    windows = ind.mark_windows(cfg)
    w_max = max(max((s["niveles"] or 1) - 1, s["pares"] or 0) for c, s in windows.items() if c in cols)
    tipo = ind.pair_types(base)
    long = (tipo == ind.PAIR_FJL) | ((tipo == ind.PAIR_HI) & (base["delta_minutes"].to_numpy() > 90))
    out = np.zeros(len(base), bool)
    last = -10 ** 9
    for i in range(len(base)):
        if tipo[i] == ind.PAIR_REINICIO:
            last = -10 ** 9
        elif long[i]:
            last = i
        out[i] = i - last < w_max
    return out[rows], w_max


@pytest.fixture(scope="module")
def variable_days(cfg):
    return _days_frame(cfg, [1380, 900, 1380, 1380, 1380, 1380, 400, 1380])  # truncadas: 2.a y 7.a; fin de semana


def test_s1_subset_union_with_full_and_truncated_days_matches_brute_force(cfg, variable_days):
    base, marks, cols, rows = variable_days
    tipo = ind.pair_types(base)
    assert (tipo == ind.PAIR_FJL).sum() == 3 and (tipo == ind.PAIR_FJ).sum() == 4 and (tipo == ind.PAIR_HI).sum() == 0
    rep = an.s1_population_report(marks, cfg, cols, rows)
    ref, w_max = _brute_union(base, cfg, cols, rows)
    assert w_max == 1343                                                    # memoria efectiva de C1/C2 n240
    np.testing.assert_array_equal(an.s1_exclusion(marks, cfg, cols, rows), ref)
    assert rep.attrs["poblacion_efectiva"] == (~ref).sum()
    assert 0 < rep.attrs["poblacion_efectiva"] < rep.attrs["poblacion"] and rep.attrs["calculable"]


def test_s1_subset_long_memory_columns_drive_the_union_loss(cfg, variable_days):
    _, marks, cols, rows = variable_days
    per = an.s1_population_report(marks, cfg, cols, rows).set_index("serie")["excluye"]
    for fam in (["A1_n20__dist", "A1_n60__dist", "A1_n240__dist"], ["C2_n14__ln_natr", "C2_n60__ln_natr", "C2_n240__ln_natr"],
                ["A2_n20__dist", "A2_n60__dist", "A2_n240__dist"]):
        assert list(per[fam]) == sorted(per[fam]), fam                      # mas memoria, mas filas excluidas
    short = [c for c in cols if c in ind.mark_windows(cfg) and c.startswith(("E", "A1_n20", "B1_n10", "C7_n20"))]
    full = an.s1_population_report(marks, cfg, cols, rows).attrs["poblacion_efectiva"]
    part = an.s1_population_report(marks, cfg, short, rows).attrs["poblacion_efectiva"]
    assert full < part <= rows.sum()                                        # sin memorias largas se pierde menos
    assert per.max() == per["C2_n240__ln_natr"]                             # la mayor memoria domina la union


def test_s1_subset_without_long_gaps_keeps_the_whole_population(cfg):
    base, marks, cols, rows = _days_frame(cfg, [1380] * 5)                 # una semana: solo FJ de 61 min
    assert (ind.pair_types(base) == ind.PAIR_FJL).sum() == 0
    rep = an.s1_population_report(marks, cfg, cols, rows)
    assert rep.attrs["poblacion_efectiva"] == rep.attrs["poblacion"] == rows.sum() > 0


def test_s1_subset_total_loss_of_population_is_reported_not_hidden(cfg):
    base, marks, cols, rows = _days_frame(cfg, [600] * 6, bars_per_day=600)   # FJL tras cada jornada de 600 barras
    rep = an.s1_population_report(marks, cfg, cols, rows)
    ref, _ = _brute_union(base, cfg, cols, rows)
    assert rows.sum() > 0 and ref.all()
    assert rep.attrs["poblacion_efectiva"] == 0 and rep.attrs["fraccion_retenida"] == 0 and not rep.attrs["calculable"]


OFFICIAL_LENGTHS = ([1380, 1380, 900, 1380, 1380] + [1380, 400, 1380, 1380, 1380] + [1380, 1380, 1380, 700, 1380] +
                    [1380] * 13)                                            # 28 jornadas; tramo final sin truncar


@pytest.fixture(scope="module")
def official(cfg):
    base, marks, panel = _days_inputs(cfg, OFFICIAL_LENGTHS)
    return base, marks, panel, an.complete_cases(panel)


def test_s1_on_the_official_population_of_all_134_columns(cfg, official):
    """Poblacion OFICIAL: casos completos de las 134 columnas de P (D7 k5 y k20 incluidas), sin excluir ninguna.
    S1 con la regla de union frente a una referencia independiente; D7 sin marcas generales (E01)."""
    base, marks, panel, ok = official
    cols = an.p_columns(cfg)
    assert list(panel.columns) == cols and len(cols) == 134
    assert {"D7_k5__rvol_log", "D7_k20__rvol_log"} <= set(cols)
    tipo = ind.pair_types(base)
    assert base["trading_date"].nunique() == 28 and (tipo == ind.PAIR_FJL).sum() >= 6 and (tipo == ind.PAIR_FJ).sum() >= 10
    assert (tipo == ind.PAIR_HI).sum() == 0 and pd.Series(OFFICIAL_LENGTHS[-13:]).eq(1380).all()
    assert ok.sum() > 0                                                     # poblacion oficial no vacia
    for k in ("D7_k5__rvol_log", "D7_k20__rvol_log"):
        assert np.isfinite(panel.loc[ok, k]).all() and int(np.isfinite(panel[k]).sum()) >= ok.sum()
    windows = ind.mark_windows(cfg)
    assert not any(c.startswith("D7_") for c in windows)                    # RVOL: sin marcas generales
    excl = an.s1_exclusion(marks, cfg, cols, ok)
    ref, w_max = _brute_union(base, cfg, cols, ok)
    np.testing.assert_array_equal(excl, ref)
    no_d7 = [c for c in cols if not c.startswith("D7_")]
    np.testing.assert_array_equal(an.s1_exclusion(marks, cfg, no_d7, ok), excl)   # D7 no aporta exclusiones
    rep = an.s1_population_report(marks, cfg, cols, ok)
    union = int(rep.set_index("serie").loc["__UNION__", "excluye"])
    assert "D7_k20__rvol_log" not in set(rep["serie"])
    registro = {"poblacion_inicial": int(ok.sum()), "excluida": union, "retenida": rep.attrs["poblacion_efectiva"],
                "fraccion_retenida": rep.attrs["fraccion_retenida"], "calculable": rep.attrs["calculable"], "W_max": w_max}
    print("REGISTRO_S1_POBLACION_OFICIAL", registro)                        # visible con pytest -s
    assert registro["poblacion_inicial"] == rep.attrs["poblacion"] == len(panel[ok])
    assert registro["excluida"] == int(excl.sum()) and registro["retenida"] == int((~excl).sum())
    assert registro["excluida"] + registro["retenida"] == registro["poblacion_inicial"]
    assert 0 < registro["retenida"] < registro["poblacion_inicial"] and registro["calculable"]
    assert registro["fraccion_retenida"] == pytest.approx(registro["retenida"] / registro["poblacion_inicial"])


def test_s2_is_blocked_without_scale_assignment(cfg, engine_cell):
    marks = engine_cell[2]
    with pytest.raises(ind.ProtocolNotAuthorizedError, match="S2 no ejecutable"):
        an.s2_exclusion(marks, cfg, an.p_columns(cfg))


def test_s5_removes_the_minute_by_year_mean(engine_cell):
    base, _, _, panel, ok = engine_cell
    sub = panel[ok].iloc[:, :5]
    d = an.s5_demean(sub, base.loc[ok, "timestamp"])
    minute = ind.minute_of_day_ny(base.loc[ok, "timestamp"])
    g = pd.DataFrame(d.to_numpy()).groupby(minute).mean()
    assert np.abs(g.to_numpy()).max() < 1e-12


def test_s6_closed_list(cfg):
    td = pd.Series(pd.to_datetime(["2020-03-09", "2020-03-10", "2020-10-22", "2020-10-23"]).date)
    np.testing.assert_array_equal(an.s6_exclusion(td, cfg), [True, False, True, False])


def test_panel_l_diagnostics_by_contract(cfg, engine_cell):
    df, series, _, panel, ok = engine_cell
    tab = an.panel_l_diagnostics(series, panel, df, cfg, ok)
    assert len(tab) == 16 and set(tab["contrato"]) == {"MNQ_A"}
    assert (tab["n"] == ok.sum()).all()                                      # L finita en toda la poblacion de P
    assert np.isfinite(tab[["pearson_contraparte", "spearman_contraparte", "pearson_C", "spearman_C"]].to_numpy()).all()


# =========================================================================== PCA, agrupamiento
def test_dimension_metrics_on_known_spectra():
    eye = an.dimension_metrics(np.ones(10))
    assert (eye["k80"], eye["k90"], eye["k95"]) == (8, 9, 10)
    assert eye["ratio_participacion"] == pytest.approx(10) and eye["rango_efectivo_entropia"] == pytest.approx(10)
    assert eye["M_eff_Li_Ji"] == pytest.approx(10)
    one = an.dimension_metrics(np.r_[10.0, np.zeros(9), -1e-15])
    assert (one["k80"], one["ratio_participacion"], one["rango_efectivo_entropia"], one["M_eff_Li_Ji"]) == (1, 1.0, 1.0, 1.0)
    mix = an.dimension_metrics(np.array([2.5, 1.0, 0.5]))
    assert mix["M_eff_Li_Ji"] == pytest.approx((1 + 0.5) + (1 + 0) + (0 + 0.5))
    assert mix["k80"] == 2 and mix["k95"] == 3


def test_hierarchical_clusters_cut_at_the_declared_rho():
    R = np.eye(6)
    for i, j, r in [(0, 1, 0.95), (2, 3, 0.80), (4, 5, 0.60)]:
        R[i, j] = R[j, i] = r
    tab = an.hierarchical_clusters(R, list("abcdef"), [0.90, 0.70]).set_index("columna")
    g90, g70 = tab["grupo_abs_rho_0.90"], tab["grupo_abs_rho_0.70"]
    assert g90["a"] == g90["b"] and g90.nunique() == 5
    assert g70["c"] == g70["d"] and g70["a"] == g70["b"] and g70["e"] != g70["f"] and g70.nunique() == 4


# =========================================================================== bootstrap y clasificacion
def test_circular_blocks_are_contiguous_and_reproducible():
    def gen(seed):
        return np.random.Generator(np.random.PCG64(seed))
    draws = an.bootstrap_day_draws(23, 5, 50, gen(1717))
    assert draws.shape == (50, 23) and draws.min() >= 0 and draws.max() < 23
    np.testing.assert_array_equal(draws, an.bootstrap_day_draws(23, 5, 50, gen(1717)))
    for row in draws:
        for b in range(0, 20, 5):                                           # bloques de 5 consecutivos modulo D
            assert all((row[b + i + 1] - row[b + i]) % 23 == 1 for i in range(4))
    assert not np.array_equal(draws, an.bootstrap_day_draws(23, 5, 50, gen(1718)))


def test_replica_correlation_equals_recomputation_on_resampled_rows():
    rng = np.random.default_rng(8)
    days = np.repeat(np.arange(30), rng.integers(5, 15, 30))
    X = rng.standard_normal((days.size, 4)) + 3.0
    X[:, 1] += 0.5 * X[:, 0]
    m = an.daily_moments(X, days)
    for draw in an.bootstrap_day_draws(30, 5, 5, np.random.Generator(np.random.PCG64(1717))):
        rows = np.concatenate([np.flatnonzero(days == d) for d in draw])
        np.testing.assert_allclose(an.replica_correlation(m, draw), np.corrcoef(X[rows], rowvar=False), atol=1e-12)


def test_interval_quantiles_come_from_e02_and_never_from_e01(cfg):
    lo, hi = an.percentile_interval(np.arange(101.0), (0.05, 0.95))
    assert (lo, hi) == (pytest.approx(5.0), pytest.approx(95.0))
    assert an.interval_quantiles(cfg) == (0.05, 0.95)
    with pytest.raises(ind.ProtocolNotAuthorizedError, match=r"no incluye la enmienda E02 \(analisis\.bootstrap\.intervalo\)"):
        an.interval_quantiles(enm.load_effective_protocol())


def _cells(r2, lo, hi, n_years_ok=6, n_segs_ok=7, publicable=True):
    rows = []
    for k, y in enumerate([2020, 2021, 2022, 2023, 2024, 2025]):
        ok = k < n_years_ok
        rows.append({"celda": f"ANIO_{y}", "tipo": "anio", "estado": "EVALUABLE_PARCIAL" if y == 2025 else "EVALUABLE",
                     "r2": r2 if ok else 0.5, "ic_inf": lo if ok else 0.4, "ic_sup": hi if ok else 0.95,
                     "publicable": publicable})
    rows.append({"celda": "ANIO_2019", "tipo": "anio", "estado": "NO_EVALUABLE", "r2": r2, "ic_inf": lo, "ic_sup": hi,
                 "publicable": True})
    for k in range(7):
        ok = k < n_segs_ok
        rows.append({"celda": f"SEG_{k}", "tipo": "segmento", "estado": "EVALUABLE", "r2": r2 if ok else 0.5,
                     "ic_inf": lo if ok else 0.4, "ic_sup": hi if ok else 0.95, "publicable": publicable})
    out = pd.DataFrame(rows)
    out["ic_estado"] = "CALCULADO"
    out["estado_matriz"] = "DEFINIDA"
    return out


def test_classification_rule(cfg):
    assert an.classify_column(_cells(0.95, 0.92, 0.97), cfg) == "RECONSTRUIBLE"
    assert an.classify_column(_cells(0.95, 0.92, 0.97, n_years_ok=5, n_segs_ok=6), cfg) == "RECONSTRUIBLE"
    assert an.classify_column(_cells(0.95, 0.92, 0.97, n_years_ok=4), cfg) == "INESTABLE"
    assert an.classify_column(_cells(0.95, 0.92, 0.97, n_segs_ok=5), cfg) == "INESTABLE"
    assert an.classify_column(_cells(0.95, 0.89, 0.97), cfg) == "INESTABLE"     # IC inferior < 0,90
    assert an.classify_column(_cells(0.40, 0.30, 0.60), cfg) == "NUEVA_REPRESENTACION"
    assert an.classify_column(_cells(0.85, 0.80, 0.91), cfg) == "INESTABLE"     # IC superior >= 0,90
    assert an.classify_column(_cells(0.95, 0.92, 0.97, publicable=False), cfg) == "INESTABLE"
    assert "NUEVA_INFORMACION" not in {an.classify_column(_cells(r, r - .05, r + .05), cfg) for r in np.linspace(0, 1, 21)}


def test_subnivel_and_transfer_rules(cfg):
    assert an.subnivel(True, 0.3, cfg) == "EXACTA"
    assert an.subnivel(False, 0.50, cfg) == "PARCIAL" and an.subnivel(False, 0.4999, cfg) == "LINEALMENTE_DISTINTA"
    assert an.subnivel(False, 0.90, cfg) == "SIN_SUBNIVEL"
    assert an.transfer_status(0.89, 0.95, cfg) == "NO_APLICA"
    assert an.transfer_status(0.95, 0.91, cfg) == "TRANSFERIBLE"
    assert an.transfer_status(0.97, 0.91, cfg) == "NO_TRANSFERIBLE"            # delta 0,06 > 0,05
    assert an.transfer_status(0.95, 0.89, cfg) == "NO_TRANSFERIBLE"


def test_min_norm_fit_predicts_held_out_rows():
    rng = np.random.default_rng(11)
    X = rng.standard_normal((4000, 3))
    y = 2 * X[:, 0] - X[:, 1] + 5
    M = np.c_[X, y, X[:, 0] + X[:, 1]]                                      # columna redundante: minima norma
    st = an.cell_stats(M[:3000])
    beta, b0 = an.min_norm_fit(st, 3, [0, 1, 2, 4])
    np.testing.assert_allclose(M[3000:, [0, 1, 2, 4]] @ beta + b0, y[3000:], atol=1e-9)


# =========================================================================== definiciones pendientes
@pytest.mark.parametrize("call,key", [
    (lambda c: an.other_indicators_predictors([], c), "R2_OTROS_INDICADORES"),
    (lambda c: an.spectral_gap_status(), "HUECO_ESPECTRAL"),
    (lambda c: an.publish_exact_by_cell(), "HUECO_ESPECTRAL"),
    (lambda c: an.s7_s8_exclusion(), "S7_S8_ALCANCE"),
])
def test_undefined_points_are_never_resolved_in_code(cfg, call, key):
    with pytest.raises(an.PendingDefinitionError, match=key):
        call(cfg)
    assert issubclass(an.PendingDefinitionError, ind.ProtocolNotAuthorizedError)


def test_production_blockers_list_every_pending_point():
    keys = {b.split(":")[0] for b in an.production_blockers()}
    assert keys == set(an.PENDIENTES) and {"T3_V2", "K_PA_MAX_T_V3", "S2_ESCALAS", "HUECO_ESPECTRAL"} <= keys
    # resueltos por E02 (registro de decisiones): ya no son puntos pendientes
    assert not keys & {"IC_NIVEL", "BOOTSTRAP_CELDAS", "SPEARMAN_VALIDACION", "CONVENCION_NULOS", "SUBNIVEL_R2"}


def test_analysis_module_reads_no_files():
    src = (enm.ROOT / "src" / "ohlcv_dataroad" / "ingest" / "tda17_analysis.py").read_text(encoding="utf-8")
    for token in ("read_parquet", "read_csv", "open(", "data/raw", "to_parquet", "to_csv", "read_text", "read_bytes"):
        assert token not in src, token
