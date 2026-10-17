"""Pruebas SINTETICAS del motor de calculo de TDA-17 (src/ohlcv_dataroad/ingest/tda17_indicators.py).

Verifican, sin datos de MNQ ni holdout:
- catalogo completo: 146 series (128 P, 16 L, 2 Aux) y 134 columnas oficiales de P;
- primera barra elegible = K declarado en el protocolo para las 102 configuraciones;
- formulas frente a referencias INDEPENDIENTES (bucles directos escritos aqui y prototipos de las
  pruebas del protocolo congelado), con huecos, fronteras de jornada y rolls;
- las 14 identidades lineales registradas y las relaciones exactas no lineales del registro;
- politica P1 (incrementos solo CONS, TR/BP locales, VWAP, OBV, ADL), reinicio por roll, RVOL que cruza
  rolls con marca, marcas de Choppiness;
- ausencia de filtracion futura en las 146 series y en las marcas.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import test_tda17_calentamiento_huecos as cal
import test_tda17_supertrend as stt
from ohlcv_dataroad.ingest import tda17_enmiendas as enm
from ohlcv_dataroad.ingest import tda17_indicators as ind
from ohlcv_dataroad.ingest import tda17_primitives as pr
from ohlcv_dataroad.ingest.session_calendar import build_session_schedule
from tda17_synthetic import TICK, approved_amendment_root, clean_bars, make_frame, rebuild_tda04

ROOT = Path(__file__).resolve().parents[1]
RTOL = 1e-10


@pytest.fixture(scope="module", autouse=True)
def approved_root(tmp_path_factory):
    """La via de produccion exige E01 APROBADA; en el repositorio esta PENDIENTE_AUDITORIA. Las pruebas usan
    una copia temporal del protocolo con la aprobacion simulada (el repositorio no se toca)."""
    root = approved_amendment_root(tmp_path_factory.mktemp("enmienda_aprobada"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(enm, "ROOT", root)
        yield root


@pytest.fixture(scope="module")
def cfg(approved_root):
    return enm.load_effective_protocol()


@pytest.fixture(scope="module")
def frame():
    return make_frame()


def prod(df):
    """Via de PRODUCCION (protocolo congelado verificado) sobre un marco sintetico sin su marca."""
    return ind.compute_catalog(df.drop(columns=[ind.SYNTHETIC_MARKER], errors="ignore"))


@pytest.fixture(scope="module")
def computed(frame, approved_root):
    return prod(frame)


def _conf(cfg, el, cid):
    e = next(e for e in cfg["catalogo"] if e["id"] == el)
    return next(c for c in e["configuraciones"] if c["id"] == cid)


def _segments(frame):
    return ind.contract_segments(frame)


def _close(a, b, rtol=RTOL, atol=1e-12):
    a, b = np.asarray(a, float), np.asarray(b, float)
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    m = ~np.isnan(a)
    assert m.sum() > 0, "comparacion trivial: ningun valor finito"
    np.testing.assert_allclose(a[m], b[m], rtol=rtol, atol=atol)


# --------------------------------------------------------------------------- 1. catalogo y paneles
def test_series_names_counts_and_panels(cfg, computed, frame):
    series, marks = computed
    cat = ind.catalog_series(cfg)
    assert list(series.columns) == [s["id"] for s in cat]
    assert {p: sum(s["panel"] == p for s in cat) for p in ("P", "L", "Aux")} == {"P": 128, "L": 16, "Aux": 2}
    assert len(cat) == cfg["recuentos"]["series_nuevas"] == 146
    P = ind.build_panel_p(series, frame)
    assert P.shape[1] == cfg["recuentos"]["columnas_P"] == 134
    assert list(P.columns) == ind.p_columns(cfg)
    assert list(P.columns[-6:]) == ["r_1m", "abs_r_1m", "log_hl", "log_co", "log_oc_prev", "ln_volume"]
    assert set(ind.CONTRACT_FUNCTIONS) | ind.CROSS_CONTRACT == {e["id"] for e in cfg["catalogo"]}


def test_engine_refuses_a_protocol_that_is_not_frozen(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text(ind.PROTOCOL_YAML.read_text(encoding="utf-8").replace("  estado: CONGELADO", "  estado: PENDIENTE_AUDITORIA_FINAL"),
                 encoding="utf-8")
    with pytest.raises(ind.ProtocolNotAuthorizedError):
        ind.load_protocol(p)


# --------------------------------------------------------------------------- 2. primera barra elegible = K
def _clean_bars():
    o, h, lo, c, v, cons, nd = clean_bars()
    return ind.Bars(o=o, h=h, l=lo, c=c, v=v, cons=cons, new_day=nd)


CONFIG_CASES = [(e["id"], c["id"]) for e in ind.load_protocol()["catalogo"] for c in e["configuraciones"] if e["id"] != "D7"]


@pytest.mark.parametrize("el,cid", CONFIG_CASES)
def test_first_eligible_bar_equals_declared_K(cfg, el, cid):
    c = _conf(cfg, el, cid)
    K = c.get("K", c.get("K_min"))
    res = ind.CONTRACT_FUNCTIONS[el](_clean_bars(), c["params"])
    for sid, arr in res.items():
        fin = np.flatnonzero(np.isfinite(arr))
        assert fin.size and fin[0] + 1 == K, (el, cid, sid, fin[0] + 1 if fin.size else None, K)


def test_rvol_requires_k_previous_trading_days(cfg):
    df = make_frame(contracts=(("MNQ_A", 12), ("MNQ_B", 14)), bars_per_day=60, hi_every=0, long_hi_every=0, fjl_every=5,
                    aligned=True)
    series, _ = prod(df)
    day_idx = pd.Index(calendar_days(df)).get_indexer(df["trading_date"])
    for cid, k in (("k5", 5), ("k20", 20)):
        x = series[f"D7_{cid}__rvol_log"].to_numpy()
        assert np.isnan(x[day_idx < k]).all() and np.isfinite(x[day_idx >= k]).any()


# --------------------------------------------------------------------------- 3. referencias independientes (bucles)
def ref_ema(x, n, start=0, alpha=None):
    alpha = alpha if alpha is not None else 2.0 / (n + 1)
    out = np.full(len(x), np.nan)
    acc, y = [], None
    for t in range(start, len(x)):
        if y is None:
            acc.append(x[t])
            if len(acc) == n:
                y = sum(acc) / n
                out[t] = y
        else:
            y = y + alpha * (x[t] - y)
            out[t] = y
    return out, start + n - 1 + pr.k_weight(alpha)


def mask(x, first):
    x = np.array(x, float)
    x[:first] = np.nan
    return x


def loop_window(x, n, f):
    out = np.full(len(x), np.nan)
    for t in range(n - 1, len(x)):
        w = x[t - n + 1:t + 1]
        out[t] = np.nan if np.isnan(w).any() else f(w)
    return out


def div(a, b):
    return a / b if (b != 0 and np.isfinite(b) and np.isfinite(a)) else np.nan


@pytest.fixture(scope="module")
def seg_bars(frame):
    """Contrato B del marco sintetico (con huecos HI, FJ, FJL) como Bars."""
    a, z = _segments(frame)[1]
    return ind._bars(frame, a, z), slice(a, z)


def _sl(computed, sid, s):
    return computed[0][sid].to_numpy()[s]


@pytest.mark.parametrize("n", [20, 60, 240])
def test_trend_levels_against_loops(computed, seg_bars, n):
    b, s = seg_bars
    e, first = ref_ema(b.c, n)
    _close(_sl(computed, f"A2_n{n}__dist", s), mask(np.log(b.c / e), first))
    w = np.arange(1, n + 1)
    wma = loop_window(b.c, n, lambda v: (v * w).sum() / w.sum())
    _close(_sl(computed, f"A3_n{n}__dist", s), np.log(b.c / wma))
    sma = loop_window(b.c, n, np.mean)
    _close(_sl(computed, f"A1_n{n}__dist", s), np.log(b.c / sma))
    slope = loop_window(b.c, n, lambda v: np.polyfit(np.arange(1, n + 1), v, 1)[0])
    _close(_sl(computed, f"A12_n{n}__pend_rel", s), slope / b.c, rtol=1e-7)


def test_hma_against_loop(computed, seg_bars, cfg):
    b, s = seg_bars
    for c in _conf_list(cfg, "A4"):
        p = c["params"]
        def wma(x, n):
            w = np.arange(1, n + 1)
            return loop_window(x, n, lambda v: (v * w).sum() / w.sum())
        hma = wma(2 * wma(b.c, p["n_mitad"]) - wma(b.c, p["n"]), p["n_raiz"])
        _close(_sl(computed, f"A4_{c['id']}__dist", s), np.log(b.c / hma))


def _conf_list(cfg, el):
    return next(e for e in cfg["catalogo"] if e["id"] == el)["configuraciones"]


def test_macd_ppo_trix_against_loops(computed, seg_bars, cfg):
    b, s = seg_bars
    for c in _conf_list(cfg, "A6"):
        f, sl, g = c["params"]["f"], c["params"]["s"], c["params"]["g"]
        ef, kf = ref_ema(b.c, f)
        es, ks = ref_ema(b.c, sl)
        k1 = max(kf, ks)
        line = ef - es
        sig, first = ref_ema(line, g, start=k1)
        # MACD convencional = diferencia de dos EMA de ~1,5e4: la cancelacion deja ~1e-11 de ruido de redondeo
        _close(_sl(computed, f"A6_{c['id']}__conv_linea", s), mask(line, first), rtol=1e-8, atol=1e-9)
        _close(_sl(computed, f"A6_{c['id']}__conv_senal", s), mask(sig, first), rtol=1e-8, atol=1e-9)
        ppo = 100 * (ef - es) / es
        psig, pfirst = ref_ema(ppo, g, start=k1)
        _close(_sl(computed, f"A7_{c['id']}__senal", s), mask(psig, pfirst), rtol=1e-8, atol=1e-11)
        _close(_sl(computed, f"A7_{c['id']}__hist", s), mask(ppo - psig, pfirst), rtol=1e-8, atol=1e-11)
    for c in _conf_list(cfg, "A10"):
        n = c["params"]["n"]
        e1, k1 = ref_ema(b.c, n)
        e2, k2 = ref_ema(e1, n, start=k1)
        e3, k3 = ref_ema(e2, n, start=k2)
        trix = np.full(b.n, np.nan)
        trix[1:] = 100 * (e3[1:] / e3[:-1] - 1)
        _close(_sl(computed, f"A10_{c['id']}__trix", s), mask(trix, k3 + 1), rtol=1e-8, atol=1e-13)


def test_aroon_against_loop(computed, seg_bars, cfg):
    b, s = seg_bars
    for c in _conf_list(cfg, "A9"):
        n = c["params"]["n"]
        up = np.full(b.n, np.nan)
        dn = np.full(b.n, np.nan)
        for t in range(n, b.n):
            wh, wl = b.h[t - n:t + 1], b.l[t - n:t + 1]
            since_h = n - max(i for i in range(n + 1) if wh[i] == wh.max())      # aparicion mas reciente
            since_l = n - max(i for i in range(n + 1) if wl[i] == wl.min())
            up[t], dn[t] = 100 * (n - since_h) / n, 100 * (n - since_l) / n
        _close(_sl(computed, f"A9_{c['id']}__up", s), up)
        _close(_sl(computed, f"A9_{c['id']}__osc", s), up - dn)


def test_momentum_family_against_prototypes_and_loops(computed, seg_bars, cfg):
    b, s = seg_bars
    for n in (14, 60):
        _close(_sl(computed, f"B3_n{n}__rsi", s), cal.rsi(b.c, b.cons, n))
    for c in _conf_list(cfg, "B8"):
        _close(_sl(computed, f"B8_{c['id']}__tsi", s), cal.tsi(b.c, b.cons, c["params"]["r"], c["params"]["s"]))
    out, _ = cal.stoch_rsi(b.c, b.cons, 14, 14, 3, 3)
    _close(_sl(computed, "B9_rsi14_n14_k3_d3__d", s), out, rtol=1e-8)
    for n in (10, 60, 240):
        _close(_sl(computed, f"B1_n{n}__roc", s), cal.roc_window(b.c, b.cons, n), rtol=1e-9, atol=1e-14)
    for n in (14, 60):
        hh, ll = loop_window(b.h, n, np.max), loop_window(b.l, n, np.min)
        _close(_sl(computed, f"B5_n{n}__wr", s), -100 * (hh - b.c) / (hh - ll))
        raw = 100 * (b.c - ll) / (hh - ll)
        k = loop_window(raw, 3, np.mean)
        _close(_sl(computed, f"B4_n{n}_k3_d3__d", s), loop_window(k, 3, np.mean))
        cmo = np.full(b.n, np.nan)
        for t in range(n, b.n):
            d = b.dC[t - n + 1:t + 1]
            d = d[~np.isnan(d)]
            if d.size >= math.ceil(0.8 * n):
                sp, sm = d[d > 0].sum(), -d[d < 0].sum()
                cmo[t] = div(100 * (sp - sm), sp + sm)
        _close(_sl(computed, f"B7_n{n}__cmo", s), cmo)
    for n in (20, 60):
        _close(_sl(computed, f"B6_n{n}__cci", s), cal.cci(b.h, b.l, b.c, n))
    for c in _conf_list(cfg, "B10"):
        p = c["params"]
        def avg(x):
            return loop_window(b.bp, x, np.sum) / loop_window(b.tr, x, np.sum)
        _close(_sl(computed, f"B10_{c['id']}__uo", s), 100 * (4 * avg(p["a"]) + 2 * avg(p["b"]) + avg(p["c"])) / 7)
    for n in (10, 60, 240):
        mom = np.full(b.n, np.nan)
        for t in range(n, b.n):
            d = b.dC[t - n + 1:t + 1]
            if np.sum(~np.isnan(d)) >= math.ceil(0.8 * n):
                mom[t] = np.nansum(d) / TICK
        _close(_sl(computed, f"B2_n{n}__mom_ticks", s), mom)


def test_adx_kama_supertrend_against_prototypes(computed, seg_bars, cfg):
    b, s = seg_bars
    for n in (14, 60):
        ref = cal.adx(b.h, b.l, b.c, b.cons, n)
        _close(_sl(computed, f"A8_n{n}__di_pos", s), ref[:, 0])
        _close(_sl(computed, f"A8_n{n}__di_neg", s), ref[:, 1])
        _close(_sl(computed, f"A8_n{n}__adx", s), ref[:, 2], rtol=1e-9)
    for c in _conf_list(cfg, "A5"):
        out, _, _ = cal.kama(b.c, b.cons, c["params"]["er"])
        _close(_sl(computed, f"A5_{c['id']}__dist", s), out)
    for c in _conf_list(cfg, "A11"):
        n, m = c["params"]["n"], c["params"]["m"]
        r = stt.supertrend_iris(b.h, b.l, b.c, b.cons, n, m)
        ref = stt.pine_reference(b.h, b.l, b.c, b.cons, n, m)
        _close(_sl(computed, f"A11_{c['id']}__dist", s), r["dist"])
        _close(_sl(computed, f"A11_{c['id']}__dir", s), r["dir"])
        K = c["K"]
        np.testing.assert_array_equal(_sl(computed, f"A11_{c['id']}__dir", s)[K - 1:], ref["dir_iris"][K - 1:])


def test_volatility_family_against_loops(computed, seg_bars, cfg):
    b, s = seg_bars
    for n in (14, 60, 240):
        _close(_sl(computed, f"C1_n{n}__atr_ticks", s), cal.atr(b.h, b.l, b.c, b.cons, n) / TICK)
        atr = cal.atr(b.h, b.l, b.c, b.cons, n)
        _close(_sl(computed, f"C2_n{n}__ln_natr", s), np.log(100 * atr / b.c))
    for n in (20, 60, 240):
        _close(_sl(computed, f"C3_n{n}_k2__pct_b", s), cal.bollinger_pct_b(b.c, n))
        sd = loop_window(b.c, n, np.std)
        _close(_sl(computed, f"C3_n{n}_k2__ln_bw", s), np.log(4 * sd / loop_window(b.c, n, np.mean)))
        _close(_sl(computed, f"C5_n{n}__pos", s), cal.donchian_pos(b.h, b.l, b.c, n))
        _close(_sl(computed, f"C5_n{n}__ln_ancho", s), np.log((loop_window(b.h, n, np.max) - loop_window(b.l, n, np.min)) / b.c))
        hl = np.log(b.h / b.l)
        _close(_sl(computed, f"C7_n{n}__ln_sigma", s), 0.5 * np.log(loop_window(hl ** 2, n, np.mean) / (4 * math.log(2))))
        gk = 0.5 * hl ** 2 - (2 * math.log(2) - 1) * np.log(b.c / b.o) ** 2
        _close(_sl(computed, f"C8_n{n}__ln_sigma", s), 0.5 * np.log(loop_window(gk, n, np.mean)))
        rs = np.log(b.h / b.c) * np.log(b.h / b.o) + np.log(b.l / b.c) * np.log(b.l / b.o)
        _close(_sl(computed, f"C9_n{n}__ln_sigma", s), 0.5 * np.log(loop_window(rs, n, np.mean)), rtol=1e-9)
        sig = np.full(b.n, np.nan)
        for t in range(n - 1, b.n):
            w = b.r[t - n + 1:t + 1]
            w = w[~np.isnan(w)]
            if w.size >= math.ceil(0.8 * n):
                sd_r = np.std(w, ddof=1)
                sig[t] = math.log(sd_r) if sd_r >= 1e-8 else np.nan
        _close(_sl(computed, f"C6_n{n}__ln_sigma", s), sig, rtol=1e-8)
    for n in (14, 60):
        _close(_sl(computed, f"C10_n{n}__chop", s), cal.chop(b.h, b.l, b.c, b.cons, n))
    for c in _conf_list(cfg, "C4"):
        p = c["params"]
        e, ke = ref_ema(b.c, p["n"])
        a, ka = ref_ema(b.tr, p["atr"], alpha=1 / p["atr"])
        up, lo = e + p["m"] * a, e - p["m"] * a
        first = max(ke, ka)
        _close(_sl(computed, f"C4_{c['id']}__pos", s), mask((b.c - lo) / (up - lo), first), rtol=1e-8, atol=1e-9)
        _close(_sl(computed, f"C4_{c['id']}__ln_ancho", s), mask(np.log((up - lo) / e), first))


def test_volume_and_candle_families_against_loops(computed, seg_bars, cfg, frame):
    b, s = seg_bars
    for n in (20, 60, 240):
        _close(_sl(computed, f"D3_n{n}__cmf", s), cal.cmf(b.h, b.l, b.c, b.v, n))
        flujo = np.full(b.n, np.nan)
        for t in range(n, b.n):
            idx = [i for i in range(t - n + 1, t + 1) if b.cons[i]]
            if len(idx) >= math.ceil(0.8 * n):
                flujo[t] = div(sum(np.sign(b.c[i] - b.c[i - 1]) * b.v[i] for i in idx), sum(b.v[i] for i in idx))
        _close(_sl(computed, f"D1_n{n}__flujo", s), flujo)
    for n in (14, 60):
        _close(_sl(computed, f"D6_n{n}__vroc_log", s), cal.vroc(b.v, n))
        mfi = np.full(b.n, np.nan)
        for t in range(n, b.n):
            idx = [i for i in range(t - n + 1, t + 1) if b.cons[i]]
            if len(idx) >= math.ceil(0.8 * n):
                pos = sum(b.tp[i] * b.v[i] for i in idx if b.tp[i] > b.tp[i - 1])
                neg = sum(b.tp[i] * b.v[i] for i in idx if b.tp[i] < b.tp[i - 1])
                mfi[t] = div(100 * pos, pos + neg)
        _close(_sl(computed, f"D2_n{n}__mfi", s), mfi)
    rng_ = np.where(b.h > b.l, b.h - b.l, 1.0)
    clv = np.where(b.h > b.l, ((b.c - b.l) - (b.h - b.c)) / rng_, 0.0)
    adl = np.cumsum(clv * b.v)
    _close(_sl(computed, "D4_acumulado__adl_acumulado", s), adl)
    for c in _conf_list(cfg, "D5"):
        ef, kf = ref_ema(adl, c["params"]["f"])
        es, ks = ref_ema(adl, c["params"]["s"])
        ev, _ = ref_ema(b.v, c["params"]["s"])
        _close(_sl(computed, f"D5_{c['id']}__chaikin_norm", s), mask((ef - es) / ev, max(kf, ks)), rtol=1e-8, atol=1e-12)
    vwap = np.full(b.n, np.nan)
    pv = vv = 0.0
    for t in range(b.n):
        if b.new_day[t]:
            pv = vv = 0.0                                                    # FJ, FJL y roll: reinicio
        pv += b.tp[t] * b.v[t]
        vv += b.v[t]
        vwap[t] = pv / vv
    _close(_sl(computed, "D8_sesion__dist", s), np.log(b.c / vwap))
    rng = b.h - b.l
    _close(_sl(computed, "E1_barra__valor", s), np.where(rng > 0, np.abs(b.c - b.o) / np.where(rng > 0, rng, 1), np.nan))
    _close(_sl(computed, "E4_barra__valor", s), np.where(rng > 0, ((b.c - b.l) - (b.h - b.c)) / np.where(rng > 0, rng, 1), np.nan))
    _close(_sl(computed, "E5_barra__valor", s), rng / TICK)


def calendar_days(df):
    """Jornadas del calendario CME_Equity del proyecto entre la primera y la ultima trading_date (independiente del motor)."""
    first, last = df["trading_date"].iloc[0], df["trading_date"].iloc[-1]
    return [d for d in build_session_schedule(pd.Timestamp(first), pd.Timestamp(last)).table.index if first <= d <= last]


def ref_rvol(df, k, days=None):
    """J(t) = las k jornadas del CALENDARIO anteriores (por defecto) o, con ``days``, otra lista de jornadas."""
    days = calendar_days(df) if days is None else days
    pos = {d: i for i, d in enumerate(days)}
    minute = ind.minute_of_day_ny(df["timestamp"])
    cell = {(pos[d], m): (v, c) for d, m, v, c in zip(df["trading_date"], minute, df["volume"], df["contract"])}
    out, cross = np.full(len(df), np.nan), np.zeros(len(df), bool)
    for r, (d, m, v, c) in enumerate(zip(df["trading_date"], minute, df["volume"], df["contract"])):
        i = pos[d]
        if i < k:
            continue
        obs = [cell[(j, m)] for j in range(i - k, i) if (j, m) in cell]
        if len(obs) >= math.ceil(0.8 * k):
            mean = sum(x for x, _ in obs) / len(obs)
            out[r] = math.log(v / mean) if v > 0 and mean > 0 else np.nan
            cross[r] = any(cc != c for _, cc in obs)
    return out, cross


def _recompute_validity(df):
    """Tras borrar filas (minutos ausentes), rehace la mascara y las variables con las funciones de TDA-04."""
    return rebuild_tda04(df[["timestamp", "source_file", "contract", "trading_date", "segment_id", "open", "high",
                             "low", "close", "volume"]])


def test_rvol_against_loop_and_roll_mark(cfg):
    df = make_frame(contracts=(("MNQ_A", 12), ("MNQ_B", 14)), bars_per_day=60, hi_every=7, long_hi_every=0, fjl_every=5,
                    aligned=True)
    df = df.drop(index=df.index[(df.index % 17 == 5) & ~df["is_roll_boundary"] & ~df["is_first_observation"]]).reset_index(drop=True)
    df = _recompute_validity(df)
    series, marks = prod(df)
    for cid, k in (("k5", 5), ("k20", 20)):
        ref, cross = ref_rvol(df, k)
        _close(series[f"D7_{cid}__rvol_log"].to_numpy(), ref)
        np.testing.assert_array_equal(marks[f"D7_{cid}__rvol_cruza_roll"].to_numpy(), cross)
        assert cross.any()                                                  # J(t) cruza el roll


# --------------------------------------------------------------------------- 4. identidades
def _ident_residual(series, frame, ident):
    def col(name):
        return series[name].to_numpy() if name in series.columns else frame[name].to_numpy(float)
    terms = np.column_stack([coef * col(name) for coef, name in ident["terminos"]])
    ok = np.isfinite(terms).all(axis=1)
    if ident["id"] == "L9":
        ok &= (frame["high"] > frame["low"]).to_numpy()
    if ident["id"] == "L11":
        ok &= frame["r_1m_valid"].to_numpy(bool)
    res = np.abs(terms[ok].sum(axis=1) - ident["constante"])
    scale = np.maximum(np.abs(terms[ok]).max(axis=1), 1.0 if ident["constante"] else 1e-300)
    return ok.sum(), (res / scale).max() if ok.any() else 0.0


def test_registered_linear_identities_L1_to_L14(cfg, computed, frame):
    series, _ = computed
    for ident in cfg["identidades"]["lineales"]:
        n_ok, rel = _ident_residual(series, frame, ident)
        assert n_ok > 100, ident["id"]
        assert rel <= cfg["analisis"]["exactitud"]["residuo_max"], (ident["id"], rel)


def test_exact_nonlinear_relations(computed, frame):
    s, _ = computed
    S = lambda k: s[k].to_numpy()                                           # noqa: E731
    C, H, L, O, V = (frame[k].to_numpy(float) for k in ("close", "high", "low", "open", "volume"))
    def check(a, b, tol=1e-9):
        ok = np.isfinite(a) & np.isfinite(b)
        assert ok.sum() > 100
        np.testing.assert_allclose(a[ok], b[ok], rtol=tol, atol=1e-12)
    for n in (20, 60, 240):                                                 # N7: beta = 6 (WMA - SMA)/(n - 1)
        sma, wma = C * np.exp(-S(f"A1_n{n}__dist")), C * np.exp(-S(f"A3_n{n}__dist"))
        check(S(f"A12_n{n}__pend_rel") * C, 6 * (wma - sma) / (n - 1), tol=1e-6)
    sma20 = C * np.exp(-S("A1_n20__dist"))                                  # N8 (Bollinger 20)
    check(S("C3_n20_k2__pct_b") - 0.5, (C / sma20 - 1) / np.exp(S("C3_n20_k2__ln_bw")))
    ema20 = C * np.exp(-S("A2_n20__dist"))                                  # N9 (Keltner 20)
    check(S("C4_n20_atr10_m2__pos") - 0.5, (C / ema20 - 1) / np.exp(S("C4_n20_atr10_m2__ln_ancho")))
    hl = H > L                                                              # N11
    clv = np.sign(C - O) * S("E1_barra__valor") + S("E3_barra__valor") - S("E2_barra__valor")
    check(np.where(hl, S("E4_barra__valor"), np.nan), np.where(hl, clv, np.nan))
    for n in (14, 60, 240):                                                 # N14: NATR = 25 ATR_ticks / C
        check(np.exp(S(f"C2_n{n}__ln_natr")), 25 * S(f"C1_n{n}__atr_ticks") / C)
    check(S("E5_barra__valor"), L * (np.exp(frame["log_hl"].to_numpy()) - 1) / TICK)          # N16
    for n in (20, 60, 240):                                                 # N17 (Parkinson con log_hl)
        lh = frame["log_hl"].to_numpy()
        park = pd.Series(lh ** 2).groupby(frame["contract"]).transform(lambda x: x.rolling(n).mean()).to_numpy()
        check(S(f"C7_n{n}__ln_sigma"), 0.5 * np.log(park / (4 * math.log(2))))
    for cid in ("f12_s26_g9", "f28_s60_g21", "f111_s240_g83"):             # N18
        for x in ("linea", "senal", "hist"):
            check(S(f"A6_{cid}__n_{x}"), S(f"A6_{cid}__conv_{x}") / C)
    for cid, s_ in (("f28_s60_g21", 60), ("f111_s240_g83", 240)):           # N19 con columnas de P
        ema_s = C * np.exp(-S(f"A2_n{s_}__dist"))
        check(S(f"A7_{cid}__linea"), 100 * S(f"A6_{cid}__n_linea") * C / ema_s)
    assert np.all(np.abs(frame["log_co"]) <= frame["log_hl"] + 1e-15)      # N21


def test_window_relations_N12_N13_N15_N20(computed, frame):
    s, _ = computed
    g = frame.groupby("contract", sort=False)
    C = frame["close"].to_numpy(float)
    cons = frame["r_1m_valid"].to_numpy(bool)
    for n in (20, 60, 240):
        adl = s["D4_acumulado__adl_acumulado"]
        d_adl = (adl - adl.groupby(frame["contract"]).shift(n)).to_numpy()
        sum_v = g["volume"].transform(lambda x: x.rolling(n).sum()).to_numpy()
        a, b = s[f"D3_n{n}__cmf"].to_numpy(), d_adl / sum_v
        ok = np.isfinite(a) & np.isfinite(b)
        np.testing.assert_allclose(a[ok], b[ok], rtol=1e-8, atol=1e-10)    # N12
        obv = s["D1__obv_acumulado"]
        d_obv = (obv - obv.groupby(frame["contract"]).shift(n)).to_numpy()
        v_cons = pd.Series(np.where(cons, frame["volume"], 0.0)).groupby(frame["contract"]).transform(
            lambda x: x.rolling(n).sum()).to_numpy()
        f = s[f"D1_n{n}__flujo"].to_numpy()
        ok = np.isfinite(f)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = d_obv / v_cons
        np.testing.assert_allclose(f[ok], ratio[ok], rtol=1e-9)              # N13
    for n in (10, 60, 240):
        full = pd.Series(cons.astype(float)).groupby(frame["contract"]).transform(lambda x: x.rolling(n).sum()).to_numpy() == n
        c_lag = frame.groupby("contract")["close"].shift(n).to_numpy()
        roc = s[f"B1_n{n}__roc"].to_numpy()
        np.testing.assert_allclose(roc[full], np.log(C / c_lag)[full], rtol=1e-9, atol=1e-14)            # N20
        mom = s[f"B2_n{n}__mom_ticks"].to_numpy()
        np.testing.assert_allclose(mom[full], (C[full] * (1 - np.exp(-roc[full]))) / TICK, rtol=1e-7, atol=1e-6)  # N15
        if n <= 60:
            assert full.sum() > 50 and (~full & np.isfinite(roc)).sum() > 0 # tambien hay ventanas con discontinuidad


# --------------------------------------------------------------------------- 5. P1, roll y marcas
def test_p1_accumulators_and_increments(computed, frame):
    s, marks = computed
    tipo = marks["tipo_par"].to_numpy()
    assert set(tipo) == {"REINICIO", "CONS", "HI", "FJ", "FJL"}
    obv = s["D1__obv_acumulado"].to_numpy()
    d_obv = np.diff(obv, prepend=0.0)
    reset = tipo == "REINICIO"
    assert np.all(d_obv[(tipo != "CONS") & ~reset] == 0)                    # OBV retiene en HI, FJ y FJL
    assert np.all(obv[reset] == 0)
    adl = s["D4_acumulado__adl_acumulado"].to_numpy()
    H, L, C, V = (frame[k].to_numpy(float) for k in ("high", "low", "close", "volume"))
    clv = np.where(H > L, ((C - L) - (H - C)) / np.where(H > L, H - L, 1), 0.0)
    d_adl = np.where(reset, adl, np.diff(adl, prepend=0.0))
    np.testing.assert_allclose(d_adl, clv * V, atol=1e-9)                   # ADL suma CLV*V en todas las barras


def test_roll_restarts_every_contract_series(cfg, computed, frame):
    s, _ = computed
    a, z = _segments(frame)[1]
    sub = frame.iloc[a:z].copy()
    sub.loc[sub.index[0], ["is_first_observation", "is_roll_boundary", "r_1m_valid"]] = [True, False, False]
    alone, _ = prod(sub.reset_index(drop=True))
    for col in s.columns:
        if col.startswith("D7_"):
            continue                                                        # RVOL cruza rolls por diseno
        _close(s[col].to_numpy()[a:z], alone[col].to_numpy(), rtol=1e-12, atol=1e-12)


def test_choppiness_marks(computed, frame):
    s, marks = computed
    for n in (14, 60):
        nd = marks[f"C10_n{n}__chop_n_disc"].to_numpy()
        non_cons = (marks["tipo_par"] != "CONS").astype(float)
        ref = non_cons.groupby(frame["contract"]).transform(lambda x: x.rolling(n).sum()).to_numpy()
        _close(nd, ref)
        chop = s[f"C10_n{n}__chop"].to_numpy()
        np.testing.assert_array_equal(marks[f"C10_n{n}__chop_fuera_rango"].to_numpy(),
                                      np.isfinite(chop) & ((chop < 0) | (chop > 100)))
        assert (nd > 0).any()


# --------------------------------------------------------------------------- 6. causalidad
def test_no_future_leakage_in_any_series_or_mark(cfg):
    df = make_frame(contracts=(("MNQ_A", 9), ("MNQ_B", 14)), bars_per_day=120, hi_every=11, long_hi_every=53, fjl_every=4)
    base_s, base_m = prod(df)
    t0 = int(len(df) * 0.7)
    alt = df.copy()
    rng = np.random.default_rng(5)
    bump = TICK * rng.integers(-40, 41, len(df) - t0 - 1)
    for col in ("open", "high", "low", "close"):
        alt.loc[t0 + 1:, col] = alt.loc[t0 + 1:, col].to_numpy() + bump + 100.0
    alt.loc[t0 + 1:, "high"] = alt.loc[t0 + 1:, ["open", "high", "close"]].max(axis=1) + 1.0
    alt.loc[t0 + 1:, "volume"] = rng.integers(0, 500, len(df) - t0 - 1).astype(float)
    alt_s, alt_m = prod(alt)
    pd.testing.assert_frame_equal(base_s.iloc[:t0 + 1], alt_s.iloc[:t0 + 1])
    pd.testing.assert_frame_equal(base_m.iloc[:t0 + 1], alt_m.iloc[:t0 + 1])
    assert not base_s.iloc[t0 + 1:].equals(alt_s.iloc[t0 + 1:])


# --------------------------------------------------------------------------- 7. validacion de entrada y codigo
def test_input_validation(cfg, frame):
    bad = frame.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError):
        prod(bad)
    bad2 = frame.copy()
    bad2.loc[0, "r_1m_valid"] = True
    with pytest.raises(ValueError):
        prod(bad2)
    with pytest.raises(ValueError):
        prod(frame.drop(columns=["delta_minutes"]))


def test_engine_code_has_no_forbidden_operations_or_data_access():
    for mod in ("tda17_indicators.py", "tda17_primitives.py"):
        text = (ROOT / "src" / "ohlcv_dataroad" / "ingest" / mod).read_text(encoding="utf-8")
        for token in ("center=True", "shift(-", "adjust=True", "adj_ratio", "adj_diff", "r_tilde",
                      "read_parquet", "read_csv", "to_parquet", "data/raw", "holdout", "bfill", "fillna("):
            assert token not in text, (mod, token)
        assert not re.search(r"\bs_m\b", text), mod                         # s_m retrospectivo de TDA-06


# =========================================================================== correcciones de la auditoria (C1-C4)
import inspect
import json
import shutil


# --------------------------------------------------------------------------- C1: protocolo congelado y autorizado
def test_c1_production_api_accepts_no_external_configuration():
    assert list(inspect.signature(ind.compute_catalog).parameters) == ["df"]
    assert list(inspect.signature(ind.build_panel_p).parameters) == ["series", "df"]
    with pytest.raises(TypeError):
        ind.compute_catalog(make_frame(), {"protocolo": {"estado": "PENDIENTE_AUDITORIA_FINAL"}})


def test_c1_frozen_protocol_hash_matches_commit_and_manifest():
    cfg = ind.load_protocol()
    assert cfg["protocolo"]["estado"] == "CONGELADO"
    m = json.loads(ind.MANIFEST_JSON.read_text(encoding="utf-8"))
    assert m["sha256_protocolo"]["configs/tda17.yaml"] == ind.FROZEN_YAML_SHA256
    assert m["congelamiento"]["git_head"] == "6245b5498e9380110df2cd6c51217d54d4f2cdd0"
    assert ind.FROZEN_COMMIT == "9cdea2f4657de1c54e3fccf86cf380b2eac8704e"


def _variant(tmp_path, transform, name="v.yaml"):
    text = ind.PROTOCOL_YAML.read_text(encoding="utf-8")
    p = tmp_path / name
    p.write_text(transform(text), encoding="utf-8")
    return p


@pytest.mark.parametrize("case", ["pendiente", "parametro_manipulado", "umbral_manipulado", "elemento_eliminado",
                                  "vacio", "comentario_anadido"])
def test_c1_load_protocol_rejects_non_authorized_yaml(tmp_path, case):
    tf = {
        "pendiente": lambda s: s.replace("  estado: CONGELADO", "  estado: PENDIENTE_AUDITORIA_FINAL"),
        "parametro_manipulado": lambda s: s.replace("{id: n14, params: {n: 14}, K: 78, U: 77}", "{id: n14, params: {n: 15}, K: 78, U: 77}"),
        "umbral_manipulado": lambda s: s.replace("umbral_r2: 0.90", "umbral_r2: 0.80"),
        "elemento_eliminado": lambda s: s.replace("  - {id: E5,", "  - {id: E5_retirado,"),
        "vacio": lambda s: "protocolo: {estado: CONGELADO}\n",
        "comentario_anadido": lambda s: s + "\n# cambio inocuo en apariencia\n",
    }[case]
    p = _variant(tmp_path, tf)
    assert p.read_text(encoding="utf-8") != ind.PROTOCOL_YAML.read_text(encoding="utf-8")
    with pytest.raises(ind.ProtocolNotAuthorizedError):
        ind.load_protocol(p)


def test_c1_load_protocol_rejects_manifest_that_does_not_certify_it(tmp_path):
    m = json.loads(ind.MANIFEST_JSON.read_text(encoding="utf-8"))
    m["estado"] = "PENDIENTE_AUDITORIA_FINAL"
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ind.ProtocolNotAuthorizedError):
        ind.load_protocol(ind.PROTOCOL_YAML, mp)


def test_c1_synthetic_path_is_separate_and_requires_marked_frames(cfg):
    frame = make_frame()
    with pytest.raises(ValueError):
        ind.compute_catalog(frame)                                          # produccion rechaza el marco sintetico
    unmarked = frame.drop(columns=[ind.SYNTHETIC_MARKER])
    pending = json.loads(json.dumps(cfg, default=str))
    pending["protocolo"]["estado"] = "PENDIENTE_AUDITORIA_FINAL"
    with pytest.raises(ValueError):
        ind.compute_catalog_synthetic(unmarked, pending)                     # sin marca: nunca con otra config
    partial = frame.copy()
    partial.loc[partial.index[-1], ind.SYNTHETIC_MARKER] = False
    with pytest.raises(ValueError):
        ind.compute_catalog_synthetic(partial, pending)
    incomplete = json.loads(json.dumps(cfg, default=str))
    incomplete["catalogo"] = incomplete["catalogo"][:-1]
    with pytest.raises(ind.ProtocolNotAuthorizedError):
        ind.compute_catalog_synthetic(frame, incomplete)
    s_syn, _ = ind.compute_catalog_synthetic(frame, cfg)                    # mecanismo de prueba explicito
    s_prod, _ = ind.compute_catalog(unmarked)
    pd.testing.assert_frame_equal(s_syn, s_prod)


# --------------------------------------------------------------------------- C2: inicializacion de ADX/DMI
def _flat_then_moving(n_flat, n_bars, seed=3, gaps=()):
    o, h, lo, c, v, cons, nd = clean_bars(n_bars, seed=seed)
    c[:n_flat] = 15000.0
    o[:n_flat] = 15000.0
    h[:n_flat] = 15000.5                                                     # H - L > 0 pero sin movimiento direccional
    lo[:n_flat] = 14999.5
    cons = cons.copy()
    cons[list(gaps)] = False
    return ind.Bars(o=o, h=h, l=lo, c=c, v=v, cons=cons, new_day=nd)


def _dx_stage(b, n):
    """DX por actualizacion CONS, calculado de forma independiente (bucle) para las pruebas."""
    idx = np.flatnonzero(b.cons)
    alpha = 1 / n
    w = {"tr": None, "p": None, "m": None}
    seeds = {"tr": [], "p": [], "m": []}
    dx = []
    for j in idx:
        up, dn = b.h[j] - b.h[j - 1], b.l[j - 1] - b.l[j]
        x = {"tr": max(b.h[j], b.c[j - 1]) - min(b.l[j], b.c[j - 1]),
             "p": up if (up > dn and up > 0) else 0.0, "m": dn if (dn > up and dn > 0) else 0.0}
        for key in w:
            if w[key] is None:
                seeds[key].append(x[key])
                if len(seeds[key]) == n:
                    w[key] = sum(seeds[key]) / n
            else:
                w[key] += alpha * (x[key] - w[key])
        if w["tr"] is None:
            dx.append(np.nan)
            continue
        pdi = 100 * w["p"] / w["tr"] if w["tr"] else np.nan
        mdi = 100 * w["m"] / w["tr"] if w["tr"] else np.nan
        s = pdi + mdi
        dx.append(100 * abs(pdi - mdi) / s if (np.isfinite(s) and s != 0) else np.nan)
    return idx, np.array(dx)


def _wilder_on_finite_reference(stage, n):
    """E01-A, bucle independiente: semilla con las n primeras DX finitas, actualizacion solo con DX finitas,
    estado retenido con DX no definido, publicado con n + k(1/n) DX finitas."""
    out, seed, state, cnt = np.full(stage.size, np.nan), [], None, 0
    for i, x in enumerate(stage):
        if np.isfinite(x):
            cnt += 1
            if state is None:
                seed.append(x)
                state = sum(seed) / n if len(seed) == n else None
            else:
                state = state + (x - state) / n
        if state is not None and cnt >= n + pr.k_weight(1 / n):
            out[i] = state
    return out


def _expected_adx(b, n, stage_values):
    u1 = n + pr.k_weight(1 / n)
    idx = np.flatnonzero(b.cons)
    adx_upd = np.full(idx.size, np.nan)
    adx_upd[u1 - 1:] = stage_values
    u = np.cumsum(b.cons)
    return np.where(u >= 2 * u1 - 1, np.where(u > 0, adx_upd[np.maximum(u - 1, 0)], np.nan), np.nan)


@pytest.mark.parametrize("n", [14, 60])
def test_e01a_constant_prices_during_warmup_seed_with_finite_dx_only(cfg, n):
    K = _conf(cfg, "A8", f"n{n}")["K"]
    u1 = n + pr.k_weight(1 / n)
    b = _flat_then_moving(u1 + 5, K + 400)                                   # DX indefinido en parte de la semilla
    idx, dx = _dx_stage(b, n)
    stage = dx[u1 - 1:]
    assert np.isnan(stage[:n]).any() and np.isfinite(stage[:n]).any()
    r = ind.a8_adx(b, {"n": n})
    _close(r["adx"], _expected_adx(b, n, _wilder_on_finite_reference(stage, n)), rtol=1e-11)
    first = int(np.flatnonzero(np.isfinite(r["adx"]))[0]) + 1
    fin = np.flatnonzero(np.isfinite(stage))
    assert first == max(K, idx[u1 - 1 + fin[n + pr.k_weight(1 / n) - 1]] + 1) > K   # activacion con n + k finitas


@pytest.mark.parametrize("n", [14, 60])
def test_e01a_dx_undefined_for_whole_seed_reseeds_with_later_finite_dx(cfg, n):
    K = _conf(cfg, "A8", f"n{n}")["K"]
    u1 = n + pr.k_weight(1 / n)
    b = _flat_then_moving(u1 + n + 30, K + 600)
    _, dx = _dx_stage(b, n)
    stage = dx[u1 - 1:]
    assert np.isnan(stage[:n]).all()
    r = ind.a8_adx(b, {"n": n})
    _close(r["adx"], _expected_adx(b, n, _wilder_on_finite_reference(stage, n)), rtol=1e-11)
    assert np.isfinite(r["adx"]).any()                                       # B siembra con las DX finitas posteriores


@pytest.mark.parametrize("n", [14, 60])
def test_e01a_complete_finite_seed_matches_conventional_wilder(cfg, n):
    K = _conf(cfg, "A8", f"n{n}")["K"]
    u1 = n + pr.k_weight(1 / n)
    b = _flat_then_moving(u1 - 20, K + 400, gaps=(u1 + 3, u1 + 7, K + 10))   # semilla completa; huecos entre CONS
    idx, dx = _dx_stage(b, n)
    stage = dx[u1 - 1:]
    assert np.isfinite(stage).all()
    ref = np.full(stage.size, np.nan)
    ref[n - 1] = stage[:n].mean()
    for i in range(n, stage.size):
        ref[i] = ref[i - 1] + (stage[i] - ref[i - 1]) / n                     # Wilder convencional
    expected = _expected_adx(b, n, ref)
    _close(ind.a8_adx(b, {"n": n})["adx"], expected, rtol=1e-11)
    assert np.isfinite(expected).any()


def test_e01a_undefined_entry_after_activation_retains_state_without_decay():
    n, k = 14, pr.k_weight(1 / 14)
    x = np.array([10.0] * (n + k) + [20.0, np.nan, np.nan, 30.0])
    y = ind.wilder_finite_inputs(x, n, 1 / n)
    assert np.isnan(y[:n + k - 1]).all() and y[n + k - 1] == 10.0           # activacion con n + k entradas finitas
    assert math.isclose(y[n + k], 10 + 10 / n)
    assert y[n + k + 1] == y[n + k + 2] == y[n + k]                          # retenido, sin decaimiento ni NaN
    assert math.isclose(y[n + k + 3], y[n + k] + (30 - y[n + k]) / n)


def test_e01a_never_enough_finite_dx_gives_no_finite_output(cfg):
    n, k = 14, pr.k_weight(1 / 14)
    x = np.r_[np.full(500, np.nan), np.full(n + k - 1, 5.0), np.full(100, np.nan)]
    assert np.isnan(ind.wilder_finite_inputs(x, n, 1 / n)).all()            # n + k - 1 finitas: nunca se activa
    b = _flat_then_moving(3000, 3000)                                        # contrato entero con precios constantes
    r = ind.a8_adx(b, {"n": n})
    K = _conf(cfg, "A8", "n14")["K"]
    assert np.isnan(r["adx"]).all()                                          # elegible desde K, pero nunca activado
    assert (r["di_pos"][K - 1:] == 0).all() and (r["di_neg"][K - 1:] == 0).all()   # +DI/-DI definidos (= 0)


def test_e01a_roll_resets_adx_state(cfg):
    n = 14
    K = _conf(cfg, "A8", "n14")["K"]
    u1 = n + pr.k_weight(1 / n)
    a = _flat_then_moving(u1 + 5, K + 200, seed=4)
    b = _flat_then_moving(0, K + 200, seed=5)
    rb = ind.a8_adx(b, {"n": n})
    assert np.isfinite(rb["adx"][K - 1:]).all() and np.isnan(rb["adx"][:K - 1]).all()
    df = make_frame(contracts=(("MNQ_A", 2), ("MNQ_B", 2)), bars_per_day=900, hi_every=0, long_hi_every=0,
                    dense_gaps=None, fjl_every=0)
    s, _ = prod(df)
    roll = int(np.flatnonzero(df["is_roll_boundary"].to_numpy())[0])
    x = s["A8_n14__adx"].to_numpy()
    assert np.isnan(x[roll:roll + K - 1]).all() and np.isfinite(x[roll + K - 1])   # el estado no cruza el roll


# --------------------------------------------------------------------------- C3: RVOL con el calendario CME_Equity
def _rvol_frame(drop_days=(), drop_minutes_every=0, start=None):
    kw = dict(contracts=(("MNQ_A", 14), ("MNQ_B", 16)), bars_per_day=40, hi_every=0, long_hi_every=0, fjl_every=0,
              aligned=True, dense_gaps=None, drop_days=drop_days)
    if start is not None:
        kw["start"] = start
    df = make_frame(**kw)
    if drop_minutes_every:
        keep = ~((df.index % drop_minutes_every == 3) & ~df["is_roll_boundary"] & ~df["is_first_observation"])
        df = rebuild_tda04(df.loc[keep, ["timestamp", "source_file", "contract", "trading_date", "segment_id", "open",
                                         "high", "low", "close", "volume"]])
    return df


@pytest.mark.parametrize("k", [5, 20])
def test_c3_day_present_with_missing_minutes(k):
    df = _rvol_frame(drop_minutes_every=7)
    s, m = prod(df)
    ref, cross = ref_rvol(df, k)
    _close(s[f"D7_k{k}__rvol_log"].to_numpy(), ref)
    np.testing.assert_array_equal(m[f"D7_k{k}__rvol_cruza_roll"].to_numpy(), cross)


@pytest.mark.parametrize("k,n_missing", [(5, 1), (5, 2), (20, 4), (20, 5)])
def test_c3_fully_missing_trading_day_is_not_replaced_by_an_older_one(k, n_missing):
    missing = tuple(range(22, 22 + n_missing))                               # jornadas del calendario sin ninguna barra
    df = _rvol_frame(drop_days=missing)
    cal_days = calendar_days(df)
    present = list(pd.unique(df["trading_date"]))
    assert len(cal_days) == len(present) + n_missing
    s, _ = prod(df)
    got = s[f"D7_k{k}__rvol_log"].to_numpy()
    ref_cal, _ = ref_rvol(df, k)                                             # J(t) con el calendario
    ref_data, _ = ref_rvol(df, k, days=present)                              # comportamiento ANTERIOR (sustituye)
    _close(got, ref_cal)
    first_after = cal_days[missing[-1] + 1]
    rows = (df["trading_date"] == first_after).to_numpy()
    if n_missing > k - math.ceil(0.8 * k):
        assert np.isnan(got[rows]).all() and np.isfinite(ref_data[rows]).all()   # no evaluable; antes se sustituia
    else:
        assert np.isfinite(got[rows]).all() and not np.allclose(got[rows], ref_data[rows])   # media de menos jornadas


def test_c3_weekends_and_holidays_are_not_trading_days(cfg):
    import datetime as dt
    df = _rvol_frame(start=dt.date(2020, 12, 1))
    days = calendar_days(df)
    for holiday in (dt.date(2020, 12, 25), dt.date(2021, 1, 1)):            # dias laborables cerrados en CME_Equity
        assert holiday.weekday() < 5 and holiday not in days
    assert all(d.weekday() < 5 for d in days)
    s, _ = prod(df)
    for after in (dt.date(2020, 12, 28), dt.date(2021, 1, 4)):
        rows = (df["trading_date"] == after).to_numpy()
        assert np.isfinite(s["D7_k5__rvol_log"].to_numpy()[rows]).all()     # J(t) salta el feriado sin perder jornadas
    ref, _ = ref_rvol(df, 5)
    _close(s["D7_k5__rvol_log"].to_numpy(), ref)


@pytest.mark.parametrize("k", [5, 20])
def test_c3_contract_crossing_mark(k):
    df = _rvol_frame()
    s, m = prod(df)
    mark = m[f"D7_k{k}__rvol_cruza_roll"].to_numpy()
    day_pos = pd.Index(calendar_days(df)).get_indexer(df["trading_date"])
    roll_day = day_pos[df["is_roll_boundary"].to_numpy()][0]
    in_b = (df["contract"] == "MNQ_B").to_numpy()
    crossing = in_b & (day_pos < roll_day + k) & np.isfinite(s[f"D7_k{k}__rvol_log"].to_numpy())
    assert crossing.any() and mark[crossing].all()
    assert not mark[in_b & (day_pos >= roll_day + k)].any()


def test_c3_trading_date_outside_calendar_is_rejected():
    import datetime as dt
    df = _rvol_frame().drop(columns=[ind.SYNTHETIC_MARKER])
    df.loc[df.index[-40:], "trading_date"] = dt.date(2021, 5, 1)            # sabado
    with pytest.raises(ValueError):
        ind.compute_catalog(df)


# --------------------------------------------------------------------------- C4: marcas generales de discontinuidad
# Dos conceptos: NIVELES (pares internos de las L barras de la ventana) y PARES (los P pares de los que dependen
# las entradas TR, BP, incrementos o r_1m, incluido el que termina en la primera barra de la ventana).
def _ref_marks(df, tipo, n_pairs, n_bars):
    """Bucle independiente: pares que terminan en t-n_pairs+1..t dentro del contrato; NaN si no caben n_bars barras."""
    N = len(df)
    contract = df["contract"].to_numpy()
    start = np.r_[0, np.flatnonzero(contract[1:] != contract[:-1]) + 1]
    seg_start = np.repeat(start, np.diff(np.r_[start, N]))
    delta = df["delta_minutes"].to_numpy(float)
    out = {k: np.full(N, np.nan) for k in ind.MARK_KEYS}
    for t in range(N):
        if t - seg_start[t] < n_bars - 1:
            continue
        pairs = range(max(t - n_pairs + 1, seg_start[t] + 1), t + 1)
        out["n_disc_HI"][t] = sum(tipo[p] == "HI" for p in pairs)
        out["n_disc_FJ"][t] = sum(tipo[p] == "FJ" for p in pairs)
        out["n_disc_FJL"][t] = sum(tipo[p] == "FJL" for p in pairs)
        out["cruza_hueco_largo"][t] = float(any(tipo[p] == "FJL" or (tipo[p] == "HI" and delta[p] > 90) for p in pairs))
    return out


def test_c4_mark_specification_per_element(cfg, computed):
    _, marks = computed
    spec = ind.mark_windows(cfg)
    assert len(spec) == 146 - 2 - 2                                           # sin Aux ni las 2 series de RVOL
    k = lambda el, cid: _conf(cfg, el, cid).get("K", _conf(cfg, el, cid).get("K_min"))    # noqa: E731
    assert spec["C10_n14__chop"] == {"niveles": 14, "pares": 14, "sesion": False}                # HH/LL y TR de las 14 barras
    assert spec["B10_a7_b14_c28__uo"] == {"niveles": None, "pares": 28, "sesion": False}
    assert spec["C6_n20__ln_sigma"] == {"niveles": None, "pares": 20, "sesion": False}           # r_1m de las 20 barras
    assert spec["B1_n10__roc"] == {"niveles": None, "pares": 10, "sesion": False}                # n pares sobre n + 1 barras
    assert spec["A1_n20__dist"] == {"niveles": 20, "pares": None, "sesion": False}
    assert spec["B3_n14__rsi"] == {"niveles": None, "pares": k("B3", "n14") - 1, "sesion": False}
    assert spec["C1_n240__atr_ticks"] == {"niveles": None, "pares": k("C1", "n240"), "sesion": False}
    assert spec["A11_n10_m3__dist"] == {"niveles": k("A11", "n10_m3"), "pares": k("A11", "n10_m3"), "sesion": False}
    assert spec["C4_n20_atr10_m2__pos"] == {"niveles": 67, "pares": 54, "sesion": False}
    assert spec["E1_barra__valor"] == {"niveles": None, "pares": None, "sesion": False}
    for s in spec.values():
        for fam in ("niveles", "pares"):
            if s[fam]:
                for key in ind.MARK_KEYS:
                    assert ind.mark_column(fam, s[fam], key) in marks.columns
    assert not any(c.startswith("w") and "__n_disc" in c for c in marks.columns)   # sin la definicion anterior
    assert spec["D8_sesion__dist"] == {"niveles": None, "pares": None, "sesion": True}   # E01-B: VWAP por sesion
    assert all(ind.mark_column("sesion", None, key) in marks.columns for key in ind.MARK_KEYS)
    for sid, s in spec.items():                                                # marcas NaN solo antes de K
        el, cid = sid.split("__")[0].split("_", 1)
        assert max(s["niveles"] or 0, s["pares"] or 0) <= k(el, cid), sid


@pytest.mark.parametrize("family,w", [("niveles", 14), ("niveles", 20), ("niveles", 61), ("niveles", 240),
                                      ("pares", 14), ("pares", 20), ("pares", 60), ("pares", 77), ("pares", 1343)])
def test_c4_marks_against_independent_loop(computed, frame, family, w):
    _, marks = computed
    n_pairs = w - 1 if family == "niveles" else w
    ref = _ref_marks(frame, marks["tipo_par"].to_numpy(), n_pairs, w)
    for key, arr in ref.items():
        got = marks[ind.mark_column(family, w, key)].to_numpy().astype(float)
        np.testing.assert_array_equal(np.isnan(got), np.isnan(arr))
        np.testing.assert_array_equal(got[~np.isnan(got)], arr[~np.isnan(arr)])


def _single(kind, pos, N=60, delta=3.0, roll_at=None):
    """Esqueleto sintetico de un contrato (o dos con roll) con UNA discontinuidad en el par que termina en ``pos``."""
    tipo = np.array(["CONS"] * N, dtype=object)
    tipo[0] = "REINICIO"
    long_hi = np.zeros(N, bool)
    if kind is not None:
        tipo[pos] = kind
        long_hi[pos] = kind == "HI" and delta > 90
    segs = [(0, N)]
    if roll_at is not None:
        tipo[roll_at] = "REINICIO"
        segs = [(0, roll_at), (roll_at, N)]
    return tipo, long_hi, segs


@pytest.mark.parametrize("kind,delta,key,cruza", [("HI", 3.0, "n_disc_HI", 0), ("HI", 121.0, "n_disc_HI", 1),
                                                   ("FJ", 61.0, "n_disc_FJ", 0), ("FJL", 2941.0, "n_disc_FJL", 1)])
@pytest.mark.parametrize("where", ["antes", "primera_barra", "segunda_barra", "interior", "ultima_barra"])
def test_c4_start_interior_end_of_window_for_each_discontinuity(kind, delta, key, cruza, where):
    W, t = 14, 40                                                            # ventana de 14 barras que termina en t
    pos = {"antes": t - W, "primera_barra": t - W + 1, "segunda_barra": t - W + 2, "interior": t - 6, "ultima_barra": t}[where]
    tipo, long_hi, segs = _single(kind, pos, delta=delta)
    niv = ind.discontinuity_marks(tipo, long_hi, segs, W - 1, W)               # niveles: pares internos
    par = ind.discontinuity_marks(tipo, long_hi, segs, W, W)                   # pares: entradas de las 14 barras
    in_niv = where in ("segunda_barra", "interior", "ultima_barra")
    in_par = where != "antes"
    assert niv[key][t] == in_niv and par[key][t] == in_par
    assert niv["cruza_hueco_largo"][t] == (cruza and in_niv) and par["cruza_hueco_largo"][t] == (cruza and in_par)
    other = [k for k in ("n_disc_HI", "n_disc_FJ", "n_disc_FJL") if k != key]
    assert all(niv[k][t] == 0 and par[k][t] == 0 for k in other)


def test_c4_roll_restarts_both_families():
    W, roll = 14, 30
    tipo, long_hi, segs = _single(None, 0, N=70, roll_at=roll)
    niv = ind.discontinuity_marks(tipo, long_hi, segs, W - 1, W)
    par = ind.discontinuity_marks(tipo, long_hi, segs, W, W)
    for m in (niv, par):
        assert np.isnan(m["n_disc_HI"][roll:roll + W - 1]).all()               # la ventana no cabe tras el roll
        assert m["n_disc_HI"][roll + W - 1] == 0                               # el reinicio no es una discontinuidad
        assert m["n_disc_HI"][roll - 1] == 0 and np.isfinite(m["n_disc_HI"][roll - 1])


def test_c4_chop_counterexample_and_no_false_positive_for_level_series(cfg):
    """Contraejemplo de la auditoria: HI en el par que termina en la PRIMERA barra de la ventana de CHOP(14)."""
    df = make_frame(contracts=(("MNQ_A", 2),), bars_per_day=300, hi_every=0, long_hi_every=0, dense_gaps=None, fjl_every=0)
    ts = df["timestamp"].copy()
    ts.iloc[100:] = ts.iloc[100:] + pd.Timedelta(minutes=2)                    # HI en el par (99, 100)
    df = rebuild_tda04(df.assign(timestamp=ts)[["timestamp", "source_file", "contract", "trading_date", "segment_id",
                                               "open", "high", "low", "close", "volume"]])
    s, m = prod(df)
    t = 100 + 13                                                               # CHOP(14) usa el TR LOCAL de la barra 100
    assert m["tipo_par"].iloc[100] == "HI" and m["C10_n14__chop_n_disc"].iloc[t] == 1
    spec = ind.mark_windows(cfg)
    chop = ind.series_discontinuity(m, spec["C10_n14__chop"])
    sma14 = ind.series_discontinuity(m, spec["B5_n14__wr"])                    # %R(14): solo niveles H/L/C de 14 barras
    assert m["niv14__n_disc_HI"].iloc[t] == 0 and m["par14__n_disc_HI"].iloc[t] == 1
    assert chop["n_disc_HI"][t] == 1                                          # antes: la marca general daba 0
    assert sma14["n_disc_HI"][t] == 0                                         # sin falso positivo para una serie de niveles
    assert chop["n_disc_HI"][t + 1] == 0 and m["C10_n14__chop_n_disc"].iloc[t + 1] == 0


def test_c4_series_marks_match_chop_specific_mark(cfg, computed):
    _, m = computed
    spec = ind.mark_windows(cfg)
    for n in (14, 60):
        gen = ind.series_discontinuity(m, spec[f"C10_n{n}__chop"])
        total = gen["n_disc_HI"] + gen["n_disc_FJ"] + gen["n_disc_FJL"]
        own = m[f"C10_n{n}__chop_n_disc"].to_numpy()                          # cuenta los TR no CONS de la ventana
        ok = np.isfinite(own) & np.isfinite(total) & (m["tipo_par"] != "REINICIO").to_numpy()
        reset_in_window = pd.Series((m["tipo_par"] == "REINICIO").astype(float)).rolling(n).sum().to_numpy() > 0
        ok &= ~reset_in_window
        np.testing.assert_array_equal(total[ok], own[ok])


# --------------------------------------------------------------------------- C1: uso exclusivo de compute_catalog en produccion
def test_c1_only_tests_use_the_synthetic_path():
    src = ROOT / "src" / "ohlcv_dataroad"
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if path.name == "tda17_indicators.py":
            continue
        assert "compute_catalog_synthetic" not in text and "._compute(" not in text, path
    engine = (src / "ingest" / "tda17_indicators.py").read_text(encoding="utf-8")
    assert "prueba criptografica de procedencia" in engine                   # SYNTHETIC_MARKER documentado
