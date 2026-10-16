"""Evidencia sintetica de Supertrend A11 (convencion clasica de TradingView, publicacion desde K).

Separa el ESTADO INTERNO del indicador, que empieza con el primer ATR disponible (barra n, semilla SMA de
los n primeros TR), de la PUBLICACION de sus salidas, que empieza en K = n + k(1/n) (54 y 335).

Dos implementaciones independientes:
- `supertrend_iris`: maquina de estados de IRIS (signo +1 alcista / -1 bajista);
- `pine_reference`: traduccion literal, serie a serie, de la formulacion documentada de TradingView
  (`ta.rma`, `nz(band[1])`, rama segun `superTrend[1] == upperBand[1]`, signo de Pine: 1 bajista,
  -1 alcista), con su propio calculo de TR. No comparten codigo.

Solo se lee `configs/tda17.yaml`. Ningun dato de MNQ, ningun holdout.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "tda17.yaml"
TICK = 0.25


@pytest.fixture(scope="module")
def a11() -> dict:
    cfg = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    return next(e for e in cfg["catalogo"] if e["id"] == "A11")


def k_weight(alpha: float) -> int:
    return math.ceil(math.log(0.01) / math.log(1 - alpha))


def K_of(n: int) -> int:
    return n + k_weight(1 / n)


# --------------------------------------------------------------------------- implementacion IRIS (estado)
def supertrend_iris(h, lo, c, cons, n, m):
    """Estado interno desde el primer ATR (barra n); salidas publicadas desde K. Un solo contrato."""
    N = c.size
    atr = np.full(N, np.nan)
    fu, fl, st, d = (np.full(N, np.nan) for _ in range(4))
    seed, val = [], None
    for t in range(N):
        if cons[t]:                                                     # P1: TR convencional solo en CONS
            tr = max(h[t], c[t - 1]) - min(lo[t], c[t - 1])
        else:                                                           # primera barra, HI, FJ, FJL: TR local
            tr = h[t] - lo[t]
        if val is None:
            seed.append(tr)
            if len(seed) == n:
                val = sum(seed) / n
        else:
            val = val + (tr - val) / n
        if val is None:
            continue
        atr[t] = val
        mid = (h[t] + lo[t]) / 2
        bu, bl = mid + m * val, mid - m * val
        if t == 0 or np.isnan(atr[t - 1]):                              # primer ATR: bandas finales = basicas
            fu[t], fl[t], d[t] = bu, bl, -1                             # inicializacion bajista
        else:
            fu[t] = bu if (bu < fu[t - 1] or c[t - 1] > fu[t - 1]) else fu[t - 1]
            fl[t] = bl if (bl > fl[t - 1] or c[t - 1] < fl[t - 1]) else fl[t - 1]
            if st[t - 1] == fu[t - 1]:                                  # rama de TradingView por VALOR de ST
                d[t] = 1 if c[t] > fu[t] else -1
            else:
                d[t] = -1 if c[t] < fl[t] else 1
        st[t] = fl[t] if d[t] == 1 else fu[t]
    K = K_of(n)
    pub = np.arange(1, N + 1) >= K
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.where(pub & (st > 0), np.log(c / st), np.nan)
    dist = np.where(np.isfinite(dist), dist, np.nan)
    return {"atr": atr, "fu": fu, "fl": fl, "st": st, "dir_interna": d,
            "dist": dist, "dir": np.where(pub, d, np.nan)}


def supertrend_by_contract(h, lo, c, cons, contract, n, m):
    out = {k: np.full(c.size, np.nan) for k in ("atr", "fu", "fl", "st", "dir_interna", "dist", "dir")}
    for k in np.unique(contract):
        idx = np.flatnonzero(contract == k)
        cc = cons[idx].copy()
        cc[0] = False                                                   # reinicio por roll
        r = supertrend_iris(h[idx], lo[idx], c[idx], cc, n, m)
        for key in out:
            out[key][idx] = r[key]
    return out


# --------------------------------------------------------------------------- referencia independiente (Pine literal)
def pine_reference(h, lo, c, cons, n, factor):
    """Traduccion serie a serie de la formulacion documentada de ta.supertrend (signo de Pine)."""
    N = c.size
    nan = float("nan")
    close_prev = np.r_[nan, c[:-1]]
    # ta.tr(true) con P1: fuera de CONS no hay close[1] utilizable -> H - L
    tr = np.where(cons, np.maximum(h - lo, np.maximum(np.abs(h - close_prev), np.abs(lo - close_prev))), h - lo)
    rma = [nan] * N                                                     # ta.rma: SMA de semilla y luego (prev*(n-1)+x)/n
    for i in range(N):
        if i == n - 1:
            rma[i] = float(np.mean(tr[:n]))
        elif i >= n:
            rma[i] = (rma[i - 1] * (n - 1) + tr[i]) / n
    def nz(x):
        return 0.0 if math.isnan(x) else x
    upper, lower, st, direction = [nan] * N, [nan] * N, [nan] * N, [nan] * N
    for i in range(N):
        atr = rma[i]
        src = (h[i] + lo[i]) / 2
        ub, lb = src + factor * atr, src - factor * atr
        prev_lb = nz(lower[i - 1]) if i > 0 else 0.0
        prev_ub = nz(upper[i - 1]) if i > 0 else 0.0
        c1 = close_prev[i]
        lower[i] = lb if (lb > prev_lb or (not math.isnan(c1) and c1 < prev_lb)) else prev_lb
        upper[i] = ub if (ub < prev_ub or (not math.isnan(c1) and c1 > prev_ub)) else prev_ub
        if math.isnan(atr):
            upper[i] = lower[i] = nan
            continue
        prev_st = st[i - 1] if i > 0 else nan
        if i == 0 or math.isnan(rma[i - 1]):
            direction[i] = 1
        elif prev_st == upper[i - 1]:
            direction[i] = -1 if c[i] > upper[i] else 1
        else:
            direction[i] = 1 if c[i] < lower[i] else -1
        st[i] = lower[i] if direction[i] == -1 else upper[i]
    pine_dir = np.array(direction)
    return {"st": np.array(st), "dir_iris": -pine_dir, "fu": np.array(upper), "fl": np.array(lower)}


# --------------------------------------------------------------------------- datos sinteticos
def walk(N, seed, drift=0.0, scale=1.0):
    rng = np.random.default_rng(seed)
    c = 20000 + TICK * np.round((np.cumsum(rng.standard_t(3, N)) * scale + drift * np.arange(N)) / TICK)
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + TICK * rng.integers(0, 3, N)
    lo = np.minimum(o, c) - TICK * rng.integers(0, 3, N)
    return h, lo, c


def cons_mask(N, gaps=()):
    m = np.ones(N, bool)
    m[0] = False
    m[list(gaps)] = False
    return m


CONFIGS = [(10, 3), (60, 3)]


# --------------------------------------------------------------------------- protocolo
def test_protocol_declares_a11_publication_K_and_exact_class(a11):
    cfg = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    ks = {c["id"]: c["K"] for c in a11["configuraciones"]}
    assert ks == {"n10_m3": K_of(10), "n60_m3": K_of(60)} == {"n10_m3": 54, "n60_m3": 335}
    assert not any(c.get("K_data_dependiente") for c in a11["configuraciones"])
    assert "A11" in cfg["calentamiento"]["clases"]["EXACTA"]["elementos"]
    assert "A11" not in cfg["calentamiento"]["clases"]["DEPENDIENTE_DE_DATOS"]["elementos"]
    conv = a11["convencion"]
    for key in ("referencia", "estado_interno", "inicializacion", "bandas_finales", "direccion", "publicacion",
                "signo", "reinicio", "discontinuidades", "relacion_N10"):
        assert key in conv


# --------------------------------------------------------------------------- 1. ATR frente a publicacion; estado en calentamiento
@pytest.mark.parametrize("n,m", CONFIGS)
def test_first_atr_vs_first_publication_and_internal_state(n, m):
    h, lo, c = walk(K_of(n) + 200, 1)
    r = supertrend_iris(h, lo, c, cons_mask(c.size), n, m)
    first_atr = int(np.flatnonzero(~np.isnan(r["atr"]))[0]) + 1
    assert first_atr == n                                               # semilla SMA de n TR
    assert int(np.flatnonzero(~np.isnan(r["dir_interna"]))[0]) + 1 == n  # estado interno desde el primer ATR
    assert r["dir_interna"][n - 1] == -1 and r["st"][n - 1] == r["fu"][n - 1]
    K = K_of(n)
    assert np.isnan(r["dist"][:K - 1]).all() and np.isnan(r["dir"][:K - 1]).all()   # nada publicado antes de K
    assert not np.isnan(r["st"][n - 1:K - 1]).any()                      # ...pero el estado ya existe
    assert np.isfinite(r["dist"][K - 1]) and r["dir"][K - 1] == r["dir_interna"][K - 1]


@pytest.mark.parametrize("n,m", CONFIGS)
def test_bands_and_direction_evolve_during_warmup(n, m):
    K = K_of(n)
    h, lo, c = walk(K + 50, 2, drift=0.0)
    c[n:] += 1.5 * np.arange(c.size - n)                                 # tendencia alcista desde el primer ATR
    h[n:] += 1.5 * np.arange(c.size - n)
    lo[n:] += 1.5 * np.arange(c.size - n)
    r = supertrend_iris(h, lo, c, cons_mask(c.size), n, m)
    seg = slice(n - 1, K - 1)
    assert np.unique(r["fl"][seg]).size > 1                               # bandas evolucionan en el calentamiento
    flips = np.flatnonzero(np.diff(r["dir_interna"][seg]) != 0)
    assert flips.size >= 1                                               # la direccion cambia antes de K
    assert r["dir"][K - 1] == 1


# --------------------------------------------------------------------------- 2. referencia independiente
@pytest.mark.parametrize("n,m", CONFIGS)
@pytest.mark.parametrize("seed", range(12))
def test_matches_independent_pine_reference(n, m, seed):
    rng = np.random.default_rng(100 + seed)
    h, lo, c = walk(2500, 100 + seed, drift=rng.choice([-0.5, 0.0, 0.5]))
    gaps = rng.choice(np.arange(1, c.size), 40, replace=False)
    cons = cons_mask(c.size, gaps)
    r = supertrend_iris(h, lo, c, cons, n, m)
    ref = pine_reference(h, lo, c, cons, n, m)
    ok = ~np.isnan(ref["st"])
    np.testing.assert_array_equal(np.isnan(r["st"]), ~ok)
    np.testing.assert_array_equal(r["dir_interna"][ok], ref["dir_iris"][ok])
    np.testing.assert_allclose(r["st"][ok], ref["st"][ok], rtol=1e-12, atol=0)
    K = K_of(n)
    np.testing.assert_array_equal(r["dir"][K - 1:], ref["dir_iris"][K - 1:])


# --------------------------------------------------------------------------- 3. escenarios
@pytest.mark.parametrize("n,m", CONFIGS)
def test_uptrend_downtrend_lateral_and_changes(n, m):
    K = K_of(n)
    N = K + 600
    base = np.full(N, 20000.0)
    up = base + 2.0 * np.arange(N)
    down = base - 2.0 * np.arange(N)
    for c, expected in ((up, 1), (down, -1)):
        r = supertrend_iris(c + 0.5, c - 0.5, c, cons_mask(N), n, m)
        assert (r["dir"][K - 1:] == expected).all()
    lateral = base.copy()                                                # cierre constante, H - L > 0, ATR > 0
    r = supertrend_iris(lateral + 1.0, lateral - 1.0, lateral, cons_mask(N), n, m)
    assert (r["atr"][n - 1:] > 0).all()
    assert (r["dir"][K - 1:] == -1).all() and np.isfinite(r["dist"][K - 1:]).all()   # nunca NaN permanente
    assert (r["dist"][K - 1:] < 0).all()
    zig = np.r_[up[:K + 200], up[K + 199] - 3.0 * np.arange(1, 201), up[K + 199] - 600 + 3.0 * np.arange(1, 201)]
    r = supertrend_iris(zig + 0.5, zig - 0.5, zig, cons_mask(zig.size), n, m)
    ref = pine_reference(zig + 0.5, zig - 0.5, zig, cons_mask(zig.size), n, m)
    pub = r["dir"][K - 1:]
    assert ((pub[1:] == -1) & (pub[:-1] == 1)).any() and ((pub[1:] == 1) & (pub[:-1] == -1)).any()
    np.testing.assert_array_equal(r["dir"][K - 1:], ref["dir_iris"][K - 1:])


# --------------------------------------------------------------------------- 4. contrato, huecos, tick
@pytest.mark.parametrize("n,m", CONFIGS)
def test_contract_restart(n, m):
    h1, l1, c1 = walk(900, 3)
    h2, l2, c2 = walk(900, 4)
    h, lo, c = np.r_[h1, h2 + 500], np.r_[l1, l2 + 500], np.r_[c1, c2 + 500]
    contract = np.r_[np.zeros(900, int), np.ones(900, int)]
    out = supertrend_by_contract(h, lo, c, cons_mask(c.size), contract, n, m)
    alone = supertrend_iris(h2 + 500, l2 + 500, c2 + 500, cons_mask(900), n, m)
    for key in ("atr", "st", "dir_interna", "dist", "dir"):
        np.testing.assert_array_equal(out[key][900:], alone[key])
    assert np.isnan(out["atr"][900:900 + n - 1]).all() and out["dir_interna"][900 + n - 1] == -1


@pytest.mark.parametrize("n,m", CONFIGS)
def test_gaps_use_local_true_range(n, m):
    h, lo, c = walk(800, 5)
    gaps = [n + 3, n + 20, 400]
    r = supertrend_iris(h, lo, c, cons_mask(c.size, gaps), n, m)
    r_conv = supertrend_iris(h, lo, c, cons_mask(c.size), n, m)
    g = n + 3                                                            # en el hueco el TR es H - L
    assert math.isclose(r["atr"][g], r["atr"][g - 1] + (h[g] - lo[g] - r["atr"][g - 1]) / n, rel_tol=1e-15)
    assert not np.array_equal(r["atr"][g:], r_conv["atr"][g:]) or h[g] - lo[g] == max(h[g], c[g - 1]) - min(lo[g], c[g - 1])
    ref = pine_reference(h, lo, c, cons_mask(c.size, gaps), n, m)
    ok = ~np.isnan(ref["st"])
    np.testing.assert_array_equal(r["dir_interna"][ok], ref["dir_iris"][ok])


@pytest.mark.parametrize("n,m", CONFIGS)
def test_tick_grid_prices(n, m):
    h, lo, c = walk(1500, 6)
    for x in (h, lo, c):
        assert np.all(x / TICK == np.round(x / TICK))
    r = supertrend_iris(h, lo, c, cons_mask(c.size), n, m)
    K = K_of(n)
    assert np.isfinite(r["dist"][K - 1:]).all() and set(np.unique(r["dir"][K - 1:])) <= {-1.0, 1.0}


# --------------------------------------------------------------------------- 5. N10 (diagnostico, no identidad), empates y no definidos
@pytest.mark.parametrize("n,m,st_expected", [(10, 3, 20002.65), (60, 3, 20007.525)])
def test_N10_is_not_universal_regression(n, m, st_expected):
    """Contraejemplo: barras estables (TR = 0,5) y una barra H=20020, L=20000, C=20002. El cierre supera la banda
    superior previa (giro alcista) pero la banda inferior nueva queda por ENCIMA del cierre: dir = +1, ln(C/ST) < 0."""
    K = K_of(n)
    N = K + 6
    c = np.full(N, 20000.0)
    h, lo = c + 0.25, c - 0.25
    h[-1], lo[-1], c[-1] = 20020.0, 20000.0, 20002.0
    r = supertrend_iris(h, lo, c, cons_mask(N), n, m)
    ref = pine_reference(h, lo, c, cons_mask(N), n, m)
    assert r["atr"][-2] == 0.5 and r["dir"][-2] == -1                    # antes: lateral bajista
    assert math.isclose(r["st"][-1], st_expected, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(ref["st"][-1], st_expected, rel_tol=0, abs_tol=1e-9)
    assert r["dir"][-1] == ref["dir_iris"][-1] == 1                       # IRIS = referencia Pine
    assert np.sign(r["dist"][-1]) == -1                                   # signos discordantes: N10 no es identidad
    cfg = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    n10 = next(x for x in cfg["identidades"]["no_lineales_o_fuera_de_P"] if x["id"] == "N10")
    assert n10["tipo"] == "DIAGNOSTICO_NO_UNIVERSAL"


@pytest.mark.parametrize("n,m", CONFIGS)
def test_N10_diagnostic_ties_and_undefined(n, m):
    h, lo, c = walk(2000, 7)
    r = supertrend_iris(h, lo, c, cons_mask(c.size), n, m)
    both = ~np.isnan(r["dist"]) & ~np.isnan(r["dir"])
    assert np.array_equal(both, ~np.isnan(r["dir"]))                     # definidos a la vez (precios > 0)
    K = K_of(n)
    assert not both[:K - 1].any()                                         # antes de K: no definidos
    agree = np.mean(np.sign(r["dist"][both & (r["dist"] != 0)]) == r["dir"][both & (r["dist"] != 0)])
    assert 0.0 <= agree <= 1.0                                            # solo se describe; no se exige
    # empate explicito: barras planas -> ATR = 0 exacto, ST = (H+L)/2 = C, dist = 0, dir bajista
    N = K + 20
    flat = np.full(N, 20000.0)
    rf = supertrend_iris(flat, flat, flat, cons_mask(N), n, m)
    assert (rf["atr"][n - 1:] == 0).all()
    assert (rf["dist"][K - 1:] == 0).all() and (rf["dir"][K - 1:] == -1).all()


@pytest.mark.parametrize("n,m", [(10, 3)])
def test_tie_rule_by_value_matches_pine_where_a_pure_state_machine_differs(n, m):
    """Tras una tendencia alcista, un tramo plano largo hace que las bandas finales coincidan en coma flotante.
    TradingView elige la rama por VALOR (ST[1] == upper[1]); una maquina de estados pura no."""
    up = 20000.0 + 2.0 * np.arange(150)
    c = np.r_[up, np.full(400, up[-1])]
    h, lo = c.copy(), c.copy()
    h[:150] += 0.5
    lo[:150] -= 0.5
    r = supertrend_iris(h, lo, c, cons_mask(c.size), n, m)
    ref = pine_reference(h, lo, c, cons_mask(c.size), n, m)
    np.testing.assert_array_equal(r["dir_interna"][n - 1:], ref["dir_iris"][n - 1:])
    tie = np.flatnonzero(r["fu"] == r["fl"])
    assert tie.size > 0 and r["dir_interna"][149] == 1
    naive = r["dir_interna"].copy()
    for t in range(tie[0] + 1, c.size):                                 # maquina de estados sin la regla por valor
        naive[t] = (-1 if c[t] < r["fl"][t] else 1) if naive[t - 1] == 1 else (1 if c[t] > r["fu"][t] else -1)
    assert (r["dir_interna"][tie[0] + 1:] == -1).all() and (naive[tie[0] + 1:] == 1).all()


# --------------------------------------------------------------------------- 6. causalidad
@pytest.mark.parametrize("n,m", CONFIGS)
def test_no_future_data(n, m):
    h, lo, c = walk(1200, 8)
    base = supertrend_iris(h, lo, c, cons_mask(c.size), n, m)
    t0 = 700
    h2, lo2, c2 = h.copy(), lo.copy(), c.copy()
    c2[t0 + 1:] -= 300.0
    h2[t0 + 1:] += 50.0
    lo2[t0 + 1:] -= 350.0
    alt = supertrend_iris(h2, lo2, c2, cons_mask(c.size, range(t0 + 5, t0 + 30)), n, m)
    for key in base:
        np.testing.assert_array_equal(base[key][:t0 + 1], alt[key][:t0 + 1])


# --------------------------------------------------------------------------- 7. inicializar en el primer ATR frente a hacerlo en K
def supertrend_init_at_K_wrong(h, lo, c, cons, n, m):
    """Variante INCORRECTA: descarta el estado previo e inicializa bandas y direccion en K."""
    r = supertrend_iris(h, lo, c, cons, n, m)
    K = K_of(n)
    d = np.full(c.size, np.nan)
    fu, fl, st = r["fu"].copy(), r["fl"].copy(), np.full(c.size, np.nan)
    mid = (h + lo) / 2
    fu[K - 1], fl[K - 1] = mid[K - 1] + m * r["atr"][K - 1], mid[K - 1] - m * r["atr"][K - 1]
    d[K - 1] = -1
    st[K - 1] = fu[K - 1]
    for t in range(K, c.size):
        bu, bl = mid[t] + m * r["atr"][t], mid[t] - m * r["atr"][t]
        fu[t] = bu if (bu < fu[t - 1] or c[t - 1] > fu[t - 1]) else fu[t - 1]
        fl[t] = bl if (bl > fl[t - 1] or c[t - 1] < fl[t - 1]) else fl[t - 1]
        d[t] = (1 if c[t] > fu[t] else -1) if st[t - 1] == fu[t - 1] else (-1 if c[t] < fl[t] else 1)
        st[t] = fl[t] if d[t] == 1 else fu[t]
    return d


@pytest.mark.parametrize("n,m", CONFIGS)
def test_initialising_at_K_gives_different_results(n, m):
    K = K_of(n)
    N = K + 300
    c = 20000.0 + 2.0 * np.arange(N)                                     # tendencia desde el primer ATR
    h, lo = c + 0.5, c - 0.5
    good = supertrend_iris(h, lo, c, cons_mask(N), n, m)["dir"]
    wrong = supertrend_init_at_K_wrong(h, lo, c, cons_mask(N), n, m)
    ref = pine_reference(h, lo, c, cons_mask(N), n, m)["dir_iris"]
    assert good[K - 1] == ref[K - 1] == 1 and wrong[K - 1] == -1         # discrepan en la primera publicacion
    assert np.sum(good[K - 1:] != wrong[K - 1:]) >= 1
    np.testing.assert_array_equal(good[K - 1:], ref[K - 1:])
