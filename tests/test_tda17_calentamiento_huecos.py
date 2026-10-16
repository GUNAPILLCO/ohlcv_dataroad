"""Evidencia sintetica del calentamiento de TDA-17 con discontinuidades (seccion `calentamiento` del protocolo).

Distingue el K TEORICO MINIMO declarado de la PRIMERA BARRA ELEGIBLE cuando las actualizaciones de los
suavizados recursivos se detienen fuera de los pares CONS (RSI, TSI, ADX/DMI, Stoch RSI), y comprueba
que la clase EXACTA (por ejemplo, ATR con TR local) no se retrasa. Distingue ademas la primera barra
elegible de la primera salida FINITA (ventanas planas, volumen cero).

Las implementaciones de aqui son prototipos de referencia minimos sobre precios sinteticos; no son la
implementacion de TDA-17. Solo se lee `configs/tda17.yaml`; ningun dato de MNQ, ningun holdout.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "tda17.yaml"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))


def _conf(cfg, el, cid):
    e = next(e for e in cfg["catalogo"] if e["id"] == el)
    return next(c for c in e["configuraciones"] if c["id"] == cid)


def k_weight(alpha: float) -> int:
    return math.ceil(math.log(0.01) / math.log(1 - alpha))


# --------------------------------------------------------------------------- prototipos de referencia
class Smoother:
    """EMA/Wilder: semilla = media de las n primeras ENTRADAS; despues, recursion. Emite tras n + k entradas."""

    def __init__(self, n: int, alpha: float):
        self.n, self.alpha, self.seed, self.value, self.updates = n, alpha, [], None, 0
        self.required = n + k_weight(alpha)

    def feed(self, x: float) -> None:
        self.updates += 1
        if self.value is None:
            self.seed.append(x)
            if len(self.seed) == self.n:
                self.value = float(np.mean(self.seed))
        else:
            self.value += self.alpha * (x - self.value)

    @property
    def emitted(self) -> bool:
        return self.updates >= self.required


def synthetic_bars(n_bars: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    c = 20000 + np.cumsum(rng.choice([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0], n_bars))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + rng.choice([0.0, 0.25, 0.5], n_bars)
    lo = np.minimum(o, c) - rng.choice([0.0, 0.25, 0.5], n_bars)
    return h, lo, c


def cons_mask(n_bars: int, non_cons=()) -> np.ndarray:
    """Par CONS por barra (indice 0-based). La primera barra tras el reinicio nunca es CONS."""
    m = np.ones(n_bars, bool)
    m[0] = False
    m[list(non_cons)] = False
    return m


def rsi(c, cons, n):
    up, dn = Smoother(n, 1 / n), Smoother(n, 1 / n)
    out = np.full(c.size, np.nan)
    for t in range(c.size):
        if cons[t]:
            d = c[t] - c[t - 1]
            up.feed(max(d, 0.0))
            dn.feed(max(-d, 0.0))
        if up.emitted:
            s = up.value + dn.value
            out[t] = 100 * up.value / s if s > 0 else np.nan
    return out


def tsi(c, cons, r, s):
    m1, a1 = Smoother(r, 2 / (r + 1)), Smoother(r, 2 / (r + 1))
    m2, a2 = Smoother(s, 2 / (s + 1)), Smoother(s, 2 / (s + 1))
    out = np.full(c.size, np.nan)
    for t in range(c.size):
        if cons[t]:
            d = c[t] - c[t - 1]
            m1.feed(d)
            a1.feed(abs(d))
            if m1.emitted:                       # la etapa 2 consume desde la primera barra elegible de la etapa 1
                m2.feed(m1.value)
                a2.feed(a1.value)
        if m2.emitted and a2.value > 0:
            out[t] = 100 * m2.value / a2.value
    return out


def adx(h, lo, c, cons, n):
    tr, pdm, ndm, sdx = (Smoother(n, 1 / n) for _ in range(4))
    out = np.full((c.size, 3), np.nan)
    for t in range(c.size):
        if cons[t]:                              # terna retenida fuera de CONS
            up, down = h[t] - h[t - 1], lo[t - 1] - lo[t]
            tr.feed(max(h[t], c[t - 1]) - min(lo[t], c[t - 1]))
            pdm.feed(up if (up > down and up > 0) else 0.0)
            ndm.feed(down if (down > up and down > 0) else 0.0)
            if tr.emitted:
                dip, din = 100 * pdm.value / tr.value, 100 * ndm.value / tr.value
                sdx.feed(100 * abs(dip - din) / (dip + din) if dip + din > 0 else 0.0)
        if sdx.emitted:                          # elegibilidad por configuracion: la salida mas exigente (ADX)
            out[t] = (100 * pdm.value / tr.value, 100 * ndm.value / tr.value, sdx.value)
    return out


def stoch_rsi(c, cons, n_rsi, n, k, d):
    r = rsi(c, cons, n_rsi)
    out = np.full(c.size, np.nan)
    e = int(np.flatnonzero(~np.isnan(r))[0])
    raw = np.full(c.size, np.nan)
    for t in range(e + n - 1, c.size):
        w = r[t - n + 1:t + 1]
        raw[t] = (r[t] - w.min()) / (w.max() - w.min()) if w.max() > w.min() else np.nan
    kk = np.full(c.size, np.nan)
    for t in range(e + n - 1 + k - 1, c.size):
        kk[t] = 100 * np.mean(raw[t - k + 1:t + 1])
    for t in range(e + n - 1 + k - 1 + d - 1, c.size):
        out[t] = np.mean(kk[t - d + 1:t + 1])
    return out, e


def atr(h, lo, c, cons, n):
    s = Smoother(n, 1 / n)
    out = np.full(c.size, np.nan)
    for t in range(c.size):
        trv = max(h[t], c[t - 1]) - min(lo[t], c[t - 1]) if cons[t] else h[t] - lo[t]   # TR segun P1
        s.feed(trv)
        if s.emitted:
            out[t] = s.value
    return out


def roc_window(c, cons, n):
    """Ventana finita de n pares (barras t-n+1..t); emite si t+1 >= n+1 y hay >= ceil(0.8 n) pares CONS."""
    out = np.full(c.size, np.nan)
    r = np.where(cons, np.log(c / np.r_[c[0], c[:-1]]), np.nan)
    for t in range(n, c.size):
        w = r[t - n + 1:t + 1]
        if np.sum(~np.isnan(w)) >= math.ceil(0.8 * n):
            out[t] = np.nansum(w)
    return out


def first(x) -> int:
    """Primera salida finita (indice base 1). Con los precios sinteticos no planos de estos tests coincide con
    la primera barra elegible; los tests de ventanas planas y volumen cero muestran que en general no."""
    valid = ~np.isnan(x) if x.ndim == 1 else ~np.isnan(x).any(axis=1)
    return int(np.flatnonzero(valid)[0]) + 1


def u_counter(cons) -> np.ndarray:
    return np.cumsum(cons)


# --------------------------------------------------------------------------- declaracion del protocolo
def test_warmup_classes_partition_the_catalog(cfg):
    cls = cfg["calentamiento"]["clases"]
    members = [e for c in cls.values() for e in c["elementos"]]
    assert len(members) == len(set(members)) == 45
    assert set(members) == {e["id"] for e in cfg["catalogo"]}
    assert set(cls["CONTADOR_CONS"]["elementos"]) == {"A8", "B3", "B8", "B9"}
    assert set(cls["DEPENDIENTE_DE_DATOS"]["elementos"]) == {"A5"}
    assert "A11" in cls["EXACTA"]["elementos"] and len(cls["EXACTA"]["elementos"]) == 33


def test_declared_update_requirements(cfg):
    for cid in ("n14", "n60"):
        n = _conf(cfg, "B3", cid)["params"]["n"]
        c = _conf(cfg, "B3", cid)
        assert c["U"] == n + k_weight(1 / n) and c["K"] == c["U"] + 1
        a = _conf(cfg, "A8", cid)
        u1 = n + k_weight(1 / n)
        assert a["U"] == u1 + u1 - 1 and a["K"] == a["U"] + 1
    for cid in ("r25_s13", "r100_s52"):
        c = _conf(cfg, "B8", cid)
        r, s = c["params"]["r"], c["params"]["s"]
        assert c["U"] == (r + k_weight(2 / (r + 1))) + (s + k_weight(2 / (s + 1))) - 1 and c["K"] == c["U"] + 1
    b9 = _conf(cfg, "B9", "rsi14_n14_k3_d3")
    p = b9["params"]
    assert b9["U"] == _conf(cfg, "B3", "n14")["U"]
    assert b9["barras_tras_E_RSI"] == (p["n"] - 1) + (p["k"] - 1) + (p["d"] - 1)
    assert b9["K"] == b9["U"] + 1 + b9["barras_tras_E_RSI"]
    for el in cfg["calentamiento"]["clases"]["CONTADOR_CONS"]["elementos"]:
        assert all("U" in c for c in next(e for e in cfg["catalogo"] if e["id"] == el)["configuraciones"])


# --------------------------------------------------------------------------- sin discontinuidades: E = K
CASES = [("B3", "n14"), ("B3", "n60"), ("B8", "r25_s13"), ("B8", "r100_s52"), ("A8", "n14"), ("A8", "n60")]


def _run(el, c, h, lo, close, cons):
    p = c["params"]
    if el == "B3":
        return rsi(close, cons, p["n"])
    if el == "B8":
        return tsi(close, cons, p["r"], p["s"])
    return adx(h, lo, close, cons, p["n"])


@pytest.mark.parametrize("el,cid", CASES)
def test_without_gaps_first_emission_equals_declared_K(cfg, el, cid):
    c = _conf(cfg, el, cid)
    h, lo, close = synthetic_bars(c["K"] + 50)
    assert first(_run(el, c, h, lo, close, cons_mask(close.size))) == c["K"]


# --------------------------------------------------------------------------- huecos durante el calentamiento
@pytest.mark.parametrize("el,cid", CASES)
@pytest.mark.parametrize("where", ["semilla", "tras_semilla", "justo_antes_de_K", "rafaga"])
def test_gaps_during_warmup_delay_first_emission_by_non_cons_count(cfg, el, cid, where):
    c = _conf(cfg, el, cid)
    K, U = c["K"], c["U"]
    n0 = c["params"].get("n", c["params"].get("r"))
    gaps = {"semilla": [3, 4, 5], "tras_semilla": [n0 + 5, n0 + 9], "justo_antes_de_K": [K - 2],
            "rafaga": list(range(K // 2, K // 2 + 25))}[where]
    h, lo, close = synthetic_bars(K + 200, seed=1)
    cons = cons_mask(close.size, gaps)
    e = first(_run(el, c, h, lo, close, cons))
    assert e == K + len(gaps)                                       # E - K = barras no CONS en 2..E
    u = u_counter(cons)
    assert e == int(np.flatnonzero(u >= U)[0]) + 1                  # E = min{t : u_t >= U}
    assert e > K


def test_rsi_seed_uses_first_n_cons_increments_only():
    n = 14
    h, lo, close = synthetic_bars(200, seed=2)
    gaps = [2, 6, 7]
    cons = cons_mask(close.size, gaps)
    up = Smoother(n, 1 / n)
    for t in range(close.size):
        if cons[t]:
            up.feed(max(close[t] - close[t - 1], 0.0))
        if len(up.seed) == n:
            break
    idx = [t for t in range(1, close.size) if cons[t]][:n]
    assert up.seed == [max(close[t] - close[t - 1], 0.0) for t in idx]
    assert max(idx) == n + len(gaps)                                # la semilla abarca mas de n barras


def test_retained_state_after_emission_and_no_return_to_nan():
    n = 14
    h, lo, close = synthetic_bars(400, seed=3)
    gaps = [150, 151, 152, 300]
    out = rsi(close, cons_mask(close.size, gaps), n)
    for g in gaps:
        assert out[g] == out[g - 1]                                 # valor retenido, sin decaimiento
    assert not np.isnan(out[first(out) - 1:]).any()


def test_counter_is_causal():
    n = 14
    h, lo, close = synthetic_bars(300, seed=4)
    cons = cons_mask(close.size, [10, 11])
    base = rsi(close, cons, n)
    t0 = 120
    close2 = close.copy()
    close2[t0 + 1:] += 7.0
    cons2 = cons.copy()
    cons2[t0 + 1:t0 + 40] = False
    alt = rsi(close2, cons2, n)
    np.testing.assert_array_equal(base[:t0 + 1], alt[:t0 + 1])      # nada posterior a t altera la salida en t


def test_roll_resets_the_counter():
    n = 14
    K = 1 + n + k_weight(1 / n)
    h, lo, close = synthetic_bars(400, seed=5)
    a, b = close[:200], close[200:]
    first_contract = rsi(a, cons_mask(a.size, [20]), n)
    second_contract = rsi(b, cons_mask(b.size), n)                  # reinicio: nada se arrastra del contrato anterior
    assert first(first_contract) == K + 1 and first(second_contract) == K


# --------------------------------------------------------------------------- Stoch RSI: RSI por contador, resto en barras
def test_stoch_rsi_first_emission(cfg):
    c = _conf(cfg, "B9", "rsi14_n14_k3_d3")
    p = c["params"]
    h, lo, close = synthetic_bars(400, seed=6)
    out, e = stoch_rsi(close, cons_mask(close.size), p["rsi"], p["n"], p["k"], p["d"])
    assert e + 1 == c["U"] + 1 and first(out) == c["K"]
    gaps_warm = [5, 40]
    out2, e2 = stoch_rsi(close, cons_mask(close.size, gaps_warm), p["rsi"], p["n"], p["k"], p["d"])
    assert e2 + 1 == c["U"] + 1 + len(gaps_warm) and first(out2) == c["K"] + len(gaps_warm)
    gaps_after = [c["U"] + 3, c["U"] + 4]                           # tras E_RSI: las etapas en barras no se retrasan
    out3, e3 = stoch_rsi(close, cons_mask(close.size, gaps_after), p["rsi"], p["n"], p["k"], p["d"])
    assert e3 == e and first(out3) == c["K"]


# --------------------------------------------------------------------------- clase EXACTA y VENTANA_VALIDEZ
@pytest.mark.parametrize("cid", ["n14", "n60", "n240"])
def test_exact_class_tr_based_is_not_delayed_by_gaps(cfg, cid):
    c = _conf(cfg, "C1", cid)
    assert "C1" in cfg["calentamiento"]["clases"]["EXACTA"]["elementos"]
    h, lo, close = synthetic_bars(c["K"] + 50, seed=7)
    gaps = [2, 3, c["K"] // 2, c["K"] - 1]
    assert first(atr(h, lo, close, cons_mask(close.size), c["params"]["n"])) == c["K"]
    assert first(atr(h, lo, close, cons_mask(close.size, gaps), c["params"]["n"])) == c["K"]


def test_window_validity_class_K_is_a_lower_bound(cfg):
    c = _conf(cfg, "B1", "n10")
    assert "B1" in cfg["calentamiento"]["clases"]["VENTANA_VALIDEZ"]["elementos"]
    n, K = c["params"]["n"], c["K"]
    h, lo, close = synthetic_bars(60, seed=8)
    assert first(roc_window(close, cons_mask(close.size), n)) == K
    assert first(roc_window(close, cons_mask(close.size, [3, 5]), n)) == K          # 8 >= ceil(0.8*10) pares
    late = roc_window(close, cons_mask(close.size, [3, 5, 7]), n)                     # 7 < 8 en t = K
    assert np.isnan(late[K - 1]) and first(late) > K


# --------------------------------------------------------------------------- elegibilidad frente a salida finita (F4)
def eligible_from(n_bars: int, K: int) -> np.ndarray:
    """Clase EXACTA: elegible desde K (base 1), solo por el esqueleto, nunca por los valores."""
    return np.arange(1, n_bars + 1) >= K


def rolling(x, n, fn):
    out = np.full(x.size, np.nan)
    for t in range(n - 1, x.size):
        out[t] = fn(x[t - n + 1:t + 1])
    return out


def nan_if_bad(num, den):
    """reglas_frontera.divisiones: denominador 0 o no finito -> NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        v = num / den
    return np.where(np.isfinite(den) & (den != 0) & np.isfinite(v), v, np.nan)


def donchian_pos(h, lo, c, n):
    hh, ll = rolling(h, n, np.max), rolling(lo, n, np.min)
    return nan_if_bad(c - ll, hh - ll)


def bollinger_pct_b(c, n):
    sma, sd = rolling(c, n, np.mean), rolling(c, n, np.std)
    sd = np.where(sd < 1e-8, np.nan, sd)                                    # sigma < 1e-8 -> NaN
    return nan_if_bad(c - sma + 2 * sd, 4 * sd)


def cci(h, lo, c, n):
    tp = (h + lo + c) / 3
    sma = rolling(tp, n, np.mean)
    md = rolling(tp, n, lambda w: np.mean(np.abs(w - w.mean())))
    return nan_if_bad(tp - sma, 0.015 * md)


def chop(h, lo, c, cons, n):
    tr = np.where(cons, np.maximum(h, np.r_[c[0], c[:-1]]) - np.minimum(lo, np.r_[c[0], c[:-1]]), h - lo)
    ratio = nan_if_bad(rolling(tr, n, np.sum), rolling(h, n, np.max) - rolling(lo, n, np.min))
    with np.errstate(divide="ignore", invalid="ignore"):
        v = 100 * np.log10(ratio) / np.log10(n)
    return np.where(np.isfinite(v), v, np.nan)


def cmf(h, lo, c, v, n):
    with np.errstate(divide="ignore", invalid="ignore"):
        clv = np.where(h > lo, ((c - lo) - (h - c)) / (h - lo), 0.0)       # CLV := 0 si H = L
    return nan_if_bad(rolling(clv * v, n, np.sum), rolling(v, n, np.sum))


def vroc(v, n):
    out = np.full(v.size, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[n:] = np.log(nan_if_bad(v[n:], v[:-n]))
    return np.where(np.isfinite(out), out, np.nan)


def flat_then_moving(n_flat: int, n_bars: int, seed: int = 9):
    h, lo, c = synthetic_bars(n_bars, seed)
    c[:n_flat] = 20000.0
    h[:n_flat] = 20000.0
    lo[:n_flat] = 20000.0
    return h, lo, c


@pytest.mark.parametrize("el,cid", [("C5", "n20"), ("C3", "n20_k2"), ("B6", "n20"), ("C10", "n14")])
def test_flat_window_is_eligible_at_K_but_not_finite(cfg, el, cid):
    c_ = _conf(cfg, el, cid)
    K, n = c_["K"], c_["params"]["n"]
    assert el in cfg["calentamiento"]["clases"]["EXACTA"]["elementos"]
    n_flat = K + 10                                                         # ventana plana mas alla de K
    h, lo, c = flat_then_moving(n_flat, K + 60)
    cons = cons_mask(c.size)
    out = {"C5": lambda: donchian_pos(h, lo, c, n), "C3": lambda: bollinger_pct_b(c, n),
           "B6": lambda: cci(h, lo, c, n), "C10": lambda: chop(h, lo, c, cons, n)}[el]()
    elig = eligible_from(c.size, K)
    assert int(np.flatnonzero(elig)[0]) + 1 == K                            # primera barra elegible = K
    assert np.all(np.isnan(out[~elig]))                                     # nada antes de K
    assert np.isnan(out[K - 1])                                             # elegible pero NaN (denominador 0)
    assert first(out) > K                                                   # primera salida finita posterior
    assert first(out) <= n_flat + 1
    assert np.all(elig[~np.isnan(out)])                                     # finita => elegible


@pytest.mark.parametrize("cid", ["n20", "n60"])
def test_zero_volume_window_cmf_is_eligible_but_nan(cfg, cid):
    c_ = _conf(cfg, "D3", cid)
    K, n = c_["K"], c_["params"]["n"]
    h, lo, c = synthetic_bars(K + 40, seed=10)
    v = np.ones(c.size)
    v[:K + 5] = 0.0
    out = cmf(h, lo, c, v, n)
    assert np.isnan(out[K - 1]) and eligible_from(c.size, K)[K - 1]
    assert first(out) == K + 6                                              # primera ventana con volumen > 0
    h2, lo2, c2 = flat_then_moving(K + 40, K + 40)                          # H = L: CLV := 0, no NaN
    out2 = cmf(h2, lo2, c2, np.ones(c2.size), n)
    assert first(out2) == K and out2[K - 1] == 0.0


def test_zero_volume_in_volume_roc_is_eligible_but_nan(cfg):
    c_ = _conf(cfg, "D6", "n14")
    K, n = c_["K"], c_["params"]["n"]
    v = np.arange(1.0, 61.0)
    v[0] = 0.0                                                              # denominador 0 en t = K
    v[K + 2] = 0.0                                                          # ln(0) no finito en t = K + 3
    out = vroc(v, n)
    assert np.isnan(out[K - 1]) and np.isfinite(out[K]) and np.isnan(out[K + 2])
    assert first(out) == K + 1


def test_flat_prices_rsi_is_eligible_at_K_but_nan(cfg):
    c_ = _conf(cfg, "B3", "n14")
    K, n = c_["K"], c_["params"]["n"]
    h, lo, c = flat_then_moving(K + 20, K + 80, seed=11)
    up = Smoother(n, 1 / n)
    cons = cons_mask(c.size)
    for t in range(K):                                                      # elegibilidad: solo el contador
        if cons[t]:
            up.feed(0.0)
    assert up.emitted and up.updates == c_["U"]
    out = rsi(c, cons, n)
    assert np.isnan(out[K - 1])                                             # W(dC+) + W(dC-) = 0 -> NaN
    assert first(out) > K


# --------------------------------------------------------------------------- B: elegibilidad, activacion y salida finita (KAMA)
def kama(c, cons, n, fast=2, slow=30):
    """KAMA segun el protocolo: elegibilidad estructural (esqueleto), activacion con ER finito, estado retenido."""
    sc_f, sc_s = 2 / (fast + 1), 2 / (slow + 1)
    d = np.where(cons, c - np.r_[c[0], c[:-1]], np.nan)
    elig = np.zeros(c.size, bool)
    out = np.full(c.size, np.nan)
    state, activation = None, None
    for t in range(c.size):
        er = np.nan
        if t >= n:                                                          # 1-based t + 1 >= n + 1
            w = d[t - n + 1:t + 1]
            elig[t] = np.sum(~np.isnan(w)) >= math.ceil(0.8 * n)            # solo esqueleto
            den = np.nansum(np.abs(w))
            if elig[t] and den > 0:
                er = abs(np.nansum(w)) / den
        if np.isfinite(er):
            if state is None:
                state, activation = c[t], t + 1                              # activacion: KAMA := C
            else:
                state += (er * (sc_f - sc_s) + sc_s) ** 2 * (c[t] - state)
            out[t] = math.log(c[t] / state)
    return out, elig, activation


@pytest.fixture(scope="module")
def kama_cfg(cfg):
    c_ = _conf(cfg, "A5", "er10")
    assert "A5" in cfg["calentamiento"]["clases"]["DEPENDIENTE_DE_DATOS"]["elementos"]
    return c_["K_min"], c_["params"]["er"]


def test_kama_variable_prices_activate_at_structural_eligibility(kama_cfg):
    K_min, n = kama_cfg
    h, lo, c = synthetic_bars(200, seed=21)
    out, elig, act = kama(c, cons_mask(c.size), n)
    assert int(np.flatnonzero(elig)[0]) + 1 == K_min                        # elegibilidad estructural
    assert act == K_min and first(out) == K_min and out[K_min - 1] == 0.0    # activacion = primera finita


def test_kama_constant_prices_are_eligible_but_not_activated(kama_cfg):
    K_min, n = kama_cfg
    c_var = synthetic_bars(200, seed=21)[2]
    c_flat = np.full(200, 20000.0)
    cons = cons_mask(200)
    out_var, elig_var, _ = kama(c_var, cons, n)
    out_flat, elig_flat, act_flat = kama(c_flat, cons, n)
    np.testing.assert_array_equal(elig_flat, elig_var)                      # la elegibilidad no depende de los valores
    assert act_flat is None and np.isnan(out_flat).all()                     # ER = 0/0: nunca se activa
    c_mix = c_flat.copy()
    c_mix[40:] = c_var[40:] - c_var[40] + 20000.25                          # primer incremento no nulo en la barra 41
    out_mix, elig_mix, act_mix = kama(c_mix, cons, n)
    np.testing.assert_array_equal(elig_mix, elig_var)
    assert act_mix == 41 > K_min and first(out_mix) == 41
    assert np.isnan(out_mix[:40]).all()


def test_kama_flat_stretch_after_activation_retains_state_and_outputs_nan(kama_cfg):
    K_min, n = kama_cfg
    c = synthetic_bars(200, seed=22)[2]
    c[80:80 + n + 5] = c[80]                                                # tramo plano mas largo que la ventana
    out, elig, act = kama(c, cons_mask(c.size), n)
    flat_window_ends = range(80 + n, 80 + n + 5)                            # ventanas con todos los dC = 0
    assert act == K_min and all(elig[t] and np.isnan(out[t]) for t in flat_window_ends)
    assert np.isfinite(out[80 + n + 5:]).all()                              # se reanuda con el estado retenido


def test_kama_gaps_delay_structural_eligibility(kama_cfg):
    K_min, n = kama_cfg
    c = synthetic_bars(80, seed=23)[2]
    out, elig, act = kama(c, cons_mask(c.size, [3, 5, 7]), n)               # 7 < ceil(0.8 * 10) pares en K_min
    assert not elig[K_min - 1] and act > K_min and act == int(np.flatnonzero(elig)[0]) + 1
