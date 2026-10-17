"""TDA-17 -- motor de calculo del catalogo (45 elementos, 102 configuraciones, 146 series).

Implementa ``configs/tda17.yaml`` (protocolo CONGELADO en el commit 9cdea2f) con la enmienda E01
(``configs/tda17_enmienda_01.yaml``: ADX con DX no definido y ventanas de las marcas), aplicada por
``tda17_enmiendas`` y solo si esta APROBADA. Fase 1 de la etapa: SOLO el calculo de las series. Las fases posteriores (correlaciones, R2, PCA, agrupamiento,
controles G2, sensibilidades) no estan aqui.

Este modulo no lee ni escribe ficheros: recibe un ``DataFrame`` ya construido (variables y mascara de
validez de TDA-04, unidas por fila) y devuelve las series. No abre ningun fichero de datos: ni los
brutos ni el periodo reservado.

Esquema de entrada (una fila por barra, orden canonico por ``timestamp``):
``timestamp`` (UTC sin zona, cierre de barra), ``contract``, ``trading_date``, ``open``, ``high``,
``low``, ``close``, ``volume``, ``r_1m_valid``, ``is_first_observation``, ``is_roll_boundary``,
``is_trading_date_boundary``, ``is_non_consecutive_minute``, ``delta_minutes`` y, para las 6
referencias de P, ``r_1m``, ``abs_r_1m``, ``log_hl``, ``log_co``, ``log_oc_prev``.

Politica P1: reinicio solo en roll; incrementos solo en pares CONS (``r_1m_valid``); niveles encadenados
dentro del contrato; TR y BP locales fuera de CONS; RVOL cruza rolls (con marca); VWAP reinicia en cada
trading_date.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ohlcv_dataroad.ingest import tda17_primitives as pr
from ohlcv_dataroad.ingest.session_calendar import build_session_schedule

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_YAML = ROOT / "configs" / "tda17.yaml"
MANIFEST_JSON = ROOT / "reports" / "mnq" / "TDA17_manifiesto_protocolo.json"
# Protocolo autorizado: commit de congelamiento y SHA-256 (LF) del YAML CONGELADO en ese commit.
FROZEN_COMMIT = "9cdea2f4657de1c54e3fccf86cf380b2eac8704e"
FROZEN_YAML_SHA256 = "920cda43e59365e9a72f13ba6e8d5adb182640b188bf23a25059a3b4e9438938"
# Columna que identifica un marco SINTETICO del ENTORNO DE PRUEBAS (nunca existe en las entradas de TDA-04).
# Es una identificacion de entorno, NO una prueba criptografica de procedencia: cualquiera podria anadirla
# a un DataFrame. La garantia de produccion no descansa en ella, sino en que el pipeline de produccion solo
# llama a ``compute_catalog()`` (protocolo congelado verificado por hash); ``compute_catalog_synthetic`` solo
# se usa en tests (comprobado estaticamente en tests/test_tda17_indicators.py).
SYNTHETIC_MARKER = "__tda17_sintetico__"
NY_TZ = "America/New_York"
LONG_GAP_MIN = 90.0

REQUIRED_COLUMNS = ["timestamp", "contract", "trading_date", "open", "high", "low", "close", "volume",
                    "r_1m_valid", "is_first_observation", "is_roll_boundary", "is_trading_date_boundary",
                    "is_non_consecutive_minute", "delta_minutes"]
REFERENCE_COLUMNS = ["r_1m", "abs_r_1m", "log_hl", "log_co", "log_oc_prev"]

# Tipos de par por barra (reglas_frontera.tipos_par). REINICIO = primera barra del contrato (sin par).
PAIR_REINICIO, PAIR_CONS, PAIR_HI, PAIR_FJ, PAIR_FJL = "REINICIO", "CONS", "HI", "FJ", "FJL"


class ProtocolNotAuthorizedError(RuntimeError):
    """El protocolo leido no es el protocolo CONGELADO y autorizado."""


def load_protocol(path: Path = PROTOCOL_YAML, manifest_path: Path = MANIFEST_JSON) -> dict:
    """Unica via de produccion para obtener el protocolo: exige el YAML CONGELADO cuyo SHA-256
    (normalizado a LF) coincide con el del commit de congelamiento y con el manifiesto."""
    raw = Path(path).read_bytes()
    cfg = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(cfg, dict) or cfg.get("protocolo", {}).get("estado") != "CONGELADO":
        raise ProtocolNotAuthorizedError("el motor de TDA-17 solo se ejecuta contra el protocolo CONGELADO")
    digest = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    if digest != FROZEN_YAML_SHA256:
        raise ProtocolNotAuthorizedError(f"configs/tda17.yaml no es el congelado en {FROZEN_COMMIT[:7]} (SHA-256 {digest})")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("estado") != "CONGELADO" or manifest.get("sha256_protocolo", {}).get("configs/tda17.yaml") != FROZEN_YAML_SHA256:
        raise ProtocolNotAuthorizedError("el manifiesto no acredita el protocolo congelado")
    _validate_cfg_structure(cfg)
    return cfg


def _validate_cfg_structure(cfg: dict) -> None:
    """Estructura minima: 45 elementos con funcion de calculo, 102 configuraciones, 146 series."""
    try:
        ids = [e["id"] for e in cfg["catalogo"]]
        n_conf = sum(len(e["configuraciones"]) for e in cfg["catalogo"])
        n_series = len(catalog_series(cfg))
        refs = [r["id"] for r in cfg["referencias_tda04"]["en_panel_P"]]
    except (KeyError, TypeError) as exc:
        raise ProtocolNotAuthorizedError(f"configuracion incompleta: {exc!r}") from exc
    if set(ids) != set(CONTRACT_FUNCTIONS) | CROSS_CONTRACT or len(ids) != 45:
        raise ProtocolNotAuthorizedError("el catalogo no contiene exactamente los 45 elementos implementados")
    if n_conf != 102 or n_series != 146 or len(refs) != 6:
        raise ProtocolNotAuthorizedError(f"recuentos incompatibles: {n_conf} configuraciones, {n_series} series, {len(refs)} referencias")


def catalog_series(cfg: dict) -> list[dict]:
    """Las 146 series del catalogo, en el orden del YAML (misma enumeracion que los tests del protocolo)."""
    out = []
    for el in cfg["catalogo"]:
        for c in el["configuraciones"]:
            for s in el["salidas"]:
                out.append({"id": f"{el['id']}_{c['id']}__{s['id']}", "elemento": el["id"], "config": c["id"],
                            "salida": s["id"], "panel": s["panel"], "transformacion": s.get("transformacion")})
        for s in el.get("salidas_auxiliares_elemento", []):
            out.append({"id": f"{el['id']}__{s['id']}", "elemento": el["id"], "config": None,
                        "salida": s["id"], "panel": s["panel"], "transformacion": None})
    return out


def p_columns(cfg: dict) -> list[str]:
    """Las 134 columnas oficiales de P: 128 series del catalogo + 6 referencias de TDA-04."""
    cols = [s["id"] for s in catalog_series(cfg) if s["panel"] == "P"]
    return cols + [r["id"] for r in cfg["referencias_tda04"]["en_panel_P"]]


# --------------------------------------------------------------------------- un contrato
@dataclass
class Bars:
    """Barras de UN contrato (entre dos reinicios), en orden temporal."""
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    cons: np.ndarray            # par (t-1, t) CONS; False en la primera barra
    new_day: np.ndarray         # la barra abre una trading_date nueva respecto de la anterior (FJ, FJL)

    def __post_init__(self):
        self.n = self.c.size
        self.c_prev = np.r_[np.nan, self.c[:-1]]
        self.h_prev = np.r_[np.nan, self.h[:-1]]
        self.l_prev = np.r_[np.nan, self.l[:-1]]
        with np.errstate(invalid="ignore"):
            self.dC = np.where(self.cons, self.c - self.c_prev, np.nan)
            self.r = np.where(self.cons, np.log(self.c / self.c_prev), np.nan)
            conv_tr = np.maximum(self.h, self.c_prev) - np.minimum(self.l, self.c_prev)
            self.tr = np.where(self.cons, conv_tr, self.h - self.l)             # TR segun P1
            self.bp = np.where(self.cons, self.c - np.minimum(self.l, self.c_prev), self.c - self.l)
        self.tp = (self.h + self.l + self.c) / 3.0
        self.clv0 = np.where(self.h > self.l, pr.safe_div((self.c - self.l) - (self.h - self.c), self.h - self.l), 0.0)


def _ema_level(x, n, start=0):
    vals, first = pr.smoother_on_bars(x, n, pr.ema_alpha(n), start)
    return vals, first


# ------------------------------------------------------------------ tendencia
def a1_sma(b, p):
    return {"dist": pr.safe_log(b.c / pr.rolling_mean(b.c, p["n"]))}


def a2_ema(b, p):
    e, first = _ema_level(b.c, p["n"])
    return {"dist": pr.mask_before(pr.safe_log(pr.safe_div(b.c, e)), first)}


def a3_wma(b, p):
    return {"dist": pr.safe_log(pr.safe_div(b.c, pr.rolling_wma(b.c, p["n"])))}


def a4_hma(b, p):
    raw = 2.0 * pr.rolling_wma(b.c, p["n_mitad"]) - pr.rolling_wma(b.c, p["n"])
    hma = pr.rolling_wma(raw, p["n_raiz"])
    return {"dist": pr.safe_log(pr.safe_div(b.c, hma))}


def a5_kama(b, p):
    n = p["er"]
    fast, slow = 2.0 / (p["rapida"] + 1), 2.0 / (p["lenta"] + 1)
    s, _, elig = pr.increment_window(b.dC, n)
    a, _, _ = pr.increment_window(np.abs(b.dC), n)
    er = np.where(elig & (a > 0), pr.safe_div(np.abs(s), a), np.nan)       # activacion: ER finito
    out = np.full(b.n, np.nan)
    state = None
    for t in range(b.n):
        e = er[t]
        if not np.isfinite(e):
            continue                                                        # estado retenido, salida NaN
        if state is None:
            state = b.c[t]                                                  # KAMA := C en la activacion
        else:
            sc = (e * (fast - slow) + slow) ** 2
            state = state + sc * (b.c[t] - state)
        out[t] = math.log(b.c[t] / state) if state > 0 else np.nan
    return {"dist": out}


def _macd_parts(b, p):
    ef, kf = _ema_level(b.c, p["f"])
    es, ks = _ema_level(b.c, p["s"])
    k1 = max(kf, ks)
    return ef, es, k1


def a6_macd(b, p):
    ef, es, k1 = _macd_parts(b, p)
    line = ef - es
    sig, first = _ema_level(line, p["g"], start=k1)                         # la etapa 2 consume desde K1
    hist = line - sig
    out = {"conv_linea": line, "conv_senal": sig, "conv_hist": hist,
           "n_linea": line / b.c, "n_senal": sig / b.c, "n_hist": hist / b.c}
    return {k: pr.mask_before(v, first) for k, v in out.items()}


def a7_ppo(b, p):
    ef, es, k1 = _macd_parts(b, p)
    ppo = 100.0 * pr.safe_div(ef - es, es)
    sig, first = _ema_level(np.where(np.arange(b.n) >= k1, ppo, np.nan), p["g"], start=k1)
    out = {"linea": ppo, "senal": sig, "hist": ppo - sig}
    return {k: pr.mask_before(v, first) for k, v in out.items()}


def a8_adx(b, p):
    """Terna Wilder (TR, +DM, -DM) actualizada solo en pares CONS; ADX = Wilder(DX) encadenado.

    Enmienda E01-A (catalogo.A8.dx_no_definido): DX no definido (+DI + -DI = 0 o W(TR) = 0 -> NaN por
    reglas_frontera.divisiones) sigue siendo NaN y NO es entrada de ADX: semilla = media de las n primeras
    DX FINITAS desde el comienzo de la etapa 2; despues solo las DX finitas actualizan y con DX no definido
    el estado se retiene sin decaimiento. adx se ACTIVA con n + k(1/n) DX finitas; si nunca se acumulan,
    adx es NaN en todo el contrato. +DI, -DI, DX y la elegibilidad estructural (u_t >= U) no cambian."""
    n = p["n"]
    alpha = pr.wilder_alpha(n)
    idx = np.flatnonzero(b.cons)
    up = b.h[idx] - b.h_prev[idx]
    dn = b.l_prev[idx] - b.l[idx]
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    w_tr = pr.seeded_smoother(b.tr[idx], n, alpha)
    w_p = pr.seeded_smoother(pdm, n, alpha)
    w_m = pr.seeded_smoother(mdm, n, alpha)
    u1 = n + pr.k_weight(alpha)
    pdi = 100.0 * pr.safe_div(w_p, w_tr)
    mdi = 100.0 * pr.safe_div(w_m, w_tr)
    dx = 100.0 * pr.safe_div(np.abs(pdi - mdi), pdi + mdi)
    adx = np.full(idx.size, np.nan)
    if idx.size >= u1:
        adx[u1 - 1:] = wilder_finite_inputs(dx[u1 - 1:], n, alpha)            # la etapa 2 empieza en U1
    U = 2 * u1 - 1
    elig = pr.cons_counter(b.cons) >= U
    res = {"di_pos": values_at_bars_masked(pdi, b.cons, elig), "di_neg": values_at_bars_masked(mdi, b.cons, elig),
           "adx": values_at_bars_masked(adx, b.cons, elig)}
    return res


def wilder_finite_inputs(x: np.ndarray, n: int, alpha: float) -> np.ndarray:
    """E01-A: Wilder sobre las entradas FINITAS de ``x`` (una NaN no es entrada). ``y[j]`` = estado tras la
    ultima entrada finita <= j (retenido sin decaimiento), publicado solo con n + k(alpha) entradas finitas
    acumuladas; NaN antes (y en todo el tramo si nunca se acumulan)."""
    finite = np.isfinite(x)
    count = np.cumsum(finite)
    states = pr.seeded_smoother(x[finite], n, alpha)                        # estado tras cada entrada finita
    y = np.full(x.size, np.nan)
    active = count >= n + pr.k_weight(alpha)
    y[active] = states[count[active] - 1]
    return y


def values_at_bars_masked(y_upd, cons, elig):
    return np.where(elig, pr.values_at_bars(y_upd, cons), np.nan)


def a9_aroon(b, p):
    n = p["n"]
    up = 100.0 * (n - pr.bars_since_extreme(b.h, n + 1, "max")) / n
    dn = 100.0 * (n - pr.bars_since_extreme(b.l, n + 1, "min")) / n
    return {"up": up, "down": dn, "osc": up - dn}


def a10_trix(b, p):
    n = p["n"]
    e1, k1 = _ema_level(b.c, n)
    e2, k2 = _ema_level(e1, n, start=k1)
    e3, k3 = _ema_level(e2, n, start=k2)
    e3_prev = np.r_[np.nan, e3[:-1]]
    trix = 100.0 * (pr.safe_div(e3, e3_prev) - 1.0)
    return {"trix": pr.mask_before(trix, k3 + 1)}


def a11_supertrend(b, p):
    """Convencion clasica de TradingView: estado desde el primer ATR (barra n), publicacion desde K."""
    n, m = p["n"], p["m"]
    atr = pr.seeded_smoother(b.tr, n, pr.wilder_alpha(n))
    fu, fl, st, d = (np.full(b.n, np.nan) for _ in range(4))
    for t in range(b.n):
        a = atr[t]
        if not np.isfinite(a):
            continue
        mid = (b.h[t] + b.l[t]) / 2.0
        bu, bl = mid + m * a, mid - m * a
        if t == 0 or not np.isfinite(atr[t - 1]):
            fu[t], fl[t], d[t] = bu, bl, -1.0                               # inicializacion bajista
        else:
            fu[t] = bu if (bu < fu[t - 1] or b.c[t - 1] > fu[t - 1]) else fu[t - 1]
            fl[t] = bl if (bl > fl[t - 1] or b.c[t - 1] < fl[t - 1]) else fl[t - 1]
            if st[t - 1] == fu[t - 1]:                                     # rama por VALOR (TradingView)
                d[t] = 1.0 if b.c[t] > fu[t] else -1.0
            else:
                d[t] = -1.0 if b.c[t] < fl[t] else 1.0
        st[t] = fl[t] if d[t] == 1.0 else fu[t]
    first = n - 1 + pr.k_weight(pr.wilder_alpha(n))
    return {"dist": pr.mask_before(pr.safe_log(pr.safe_div(b.c, st)), first), "dir": pr.mask_before(d, first)}


def a12_slope(b, p):
    return {"pend_rel": pr.safe_div(pr.rolling_ols_slope(b.c, p["n"]), b.c)}


# ------------------------------------------------------------------ momentum
def _inc_ratio(num_inc, n):
    total, _, valid = pr.increment_window(num_inc, n)
    return np.where(valid, total, np.nan)


def b1_roc(b, p):
    return {"roc": _inc_ratio(b.r, p["n"])}


def b2_momentum(b, p):
    return {"mom_ticks": _inc_ratio(b.dC / pr.TICK, p["n"])}


def _rsi_bar(b, n):
    alpha = pr.wilder_alpha(n)
    idx = np.flatnonzero(b.cons)
    d = b.dC[idx]
    wp = pr.seeded_smoother(np.maximum(d, 0.0), n, alpha)
    wm = pr.seeded_smoother(np.maximum(-d, 0.0), n, alpha)
    rsi_upd = 100.0 * pr.safe_div(wp, wp + wm)
    U = n + pr.k_weight(alpha)
    elig = pr.cons_counter(b.cons) >= U
    return values_at_bars_masked(rsi_upd, b.cons, elig), elig


def b3_rsi(b, p):
    return {"rsi": _rsi_bar(b, p["n"])[0]}


def b4_stoch(b, p):
    n = p["n"]
    hh, ll = pr.rolling_max(b.h, n), pr.rolling_min(b.l, n)
    raw = 100.0 * pr.safe_div(b.c - ll, hh - ll)
    k = pr.rolling_mean(raw, p["k"])
    first = n - 1 + (p["k"] - 1) + (p["d"] - 1)                             # elegibilidad por configuracion (%D)
    return {"k": pr.mask_before(k, first), "d": pr.rolling_mean(k, p["d"])}


def b5_williams(b, p):
    n = p["n"]
    hh, ll = pr.rolling_max(b.h, n), pr.rolling_min(b.l, n)
    return {"wr": -100.0 * pr.safe_div(hh - b.c, hh - ll)}


def b6_cci(b, p):
    n = p["n"]
    return {"cci": pr.safe_div(b.tp - pr.rolling_mean(b.tp, n), 0.015 * pr.rolling_mean_abs_dev(b.tp, n))}


def b7_cmo(b, p):
    n = p["n"]
    sp, _, valid = pr.increment_window(np.maximum(b.dC, 0.0), n)
    sm, _, _ = pr.increment_window(np.maximum(-b.dC, 0.0), n)
    return {"cmo": np.where(valid, 100.0 * pr.safe_div(sp - sm, sp + sm), np.nan)}


def b8_tsi(b, p):
    r, s = p["r"], p["s"]
    ar, as_ = pr.ema_alpha(r), pr.ema_alpha(s)
    idx = np.flatnonzero(b.cons)
    d = b.dC[idx]
    m1 = pr.seeded_smoother(d, r, ar)
    a1 = pr.seeded_smoother(np.abs(d), r, ar)
    u1 = r + pr.k_weight(ar)
    m2 = np.full(idx.size, np.nan)
    a2 = np.full(idx.size, np.nan)
    if idx.size >= u1:
        m2[u1 - 1:] = pr.seeded_smoother(m1[u1 - 1:], s, as_)               # etapa 2 desde la elegibilidad de la 1
        a2[u1 - 1:] = pr.seeded_smoother(a1[u1 - 1:], s, as_)
    tsi_upd = 100.0 * pr.safe_div(m2, a2)
    U = u1 + (s + pr.k_weight(as_)) - 1
    elig = pr.cons_counter(b.cons) >= U
    return {"tsi": values_at_bars_masked(tsi_upd, b.cons, elig)}


def b9_stochrsi(b, p):
    rsi, elig = _rsi_bar(b, p["rsi"])
    n = p["n"]
    raw = pr.safe_div(rsi - pr.rolling_min(rsi, n), pr.rolling_max(rsi, n) - pr.rolling_min(rsi, n))
    k = 100.0 * pr.rolling_mean(raw, p["k"])
    e_rsi = int(np.argmax(elig)) if elig.any() else b.n                    # primera barra elegible del RSI
    first = e_rsi + (n - 1) + (p["k"] - 1) + (p["d"] - 1)                   # elegibilidad por configuracion (%D)
    return {"k": pr.mask_before(k, first), "d": pr.mask_before(pr.rolling_mean(k, p["d"]), first)}


def b10_uo(b, p):
    def avg(x):
        return pr.safe_div(pr.rolling_sum(b.bp, x), pr.rolling_sum(b.tr, x))
    return {"uo": 100.0 * (4 * avg(p["a"]) + 2 * avg(p["b"]) + avg(p["c"])) / 7.0}


# ------------------------------------------------------------------ volatilidad
def _atr_bars(b, n):
    vals, first = pr.smoother_on_bars(b.tr, n, pr.wilder_alpha(n))
    return pr.mask_before(vals, first)


def c1_atr(b, p):
    return {"atr_ticks": _atr_bars(b, p["n"]) / pr.TICK}


def c2_natr(b, p):
    return {"ln_natr": pr.safe_log(100.0 * pr.safe_div(_atr_bars(b, p["n"]), b.c))}


def c3_bollinger(b, p):
    n, k = p["n"], p["k"]
    sma = pr.rolling_mean(b.c, n)
    sd = pr.rolling_pstd(b.c, n)
    sd = np.where(sd < pr.SIGMA_MIN, np.nan, sd)
    return {"pct_b": pr.safe_div(b.c - sma + k * sd, 2 * k * sd), "ln_bw": pr.safe_log(pr.safe_div(2 * k * sd, sma))}


def c4_keltner(b, p):
    ema, ke = _ema_level(b.c, p["n"])
    atr, ka = pr.smoother_on_bars(b.tr, p["atr"], pr.wilder_alpha(p["atr"]))
    first = max(ke, ka)
    up, lo = ema + p["m"] * atr, ema - p["m"] * atr
    pos = pr.safe_div(b.c - lo, up - lo)
    ancho = pr.safe_log(pr.safe_div(up - lo, ema))
    return {"pos": pr.mask_before(pos, first), "ln_ancho": pr.mask_before(ancho, first)}


def c5_donchian(b, p):
    n = p["n"]
    hh, ll = pr.rolling_max(b.h, n), pr.rolling_min(b.l, n)
    return {"pos": pr.safe_div(b.c - ll, hh - ll), "ln_ancho": pr.safe_log(pr.safe_div(hh - ll, b.c))}


def c6_sd_returns(b, p):
    n = p["n"]
    present = ~np.isnan(b.r)
    m = pr.rolling(present.astype(float), n, lambda w: w.sum(axis=-1))
    r0 = np.where(present, b.r, 0.0)
    s1 = pr.rolling_sum(r0, n)
    s2 = pr.rolling(r0, n, lambda w: (w ** 2).sum(axis=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 ** 2 / m) / (m - 1)
    ok = np.nan_to_num(m) >= pr.min_valid(n)
    sd = np.sqrt(np.where(ok & (var > 0), var, np.nan))
    sd = np.where(sd < pr.SIGMA_MIN, np.nan, sd)
    return {"ln_sigma": pr.safe_log(sd)}


def _ln_sigma_from_var(var):
    return 0.5 * pr.safe_log(var)


def c7_parkinson(b, p):
    hl = np.log(b.h / b.l)
    return {"ln_sigma": _ln_sigma_from_var(pr.rolling_mean(hl ** 2, p["n"]) / (4 * math.log(2)))}


def c8_garman_klass(b, p):
    term = 0.5 * np.log(b.h / b.l) ** 2 - (2 * math.log(2) - 1) * np.log(b.c / b.o) ** 2
    return {"ln_sigma": _ln_sigma_from_var(pr.rolling_mean(term, p["n"]))}


def c9_rogers_satchell(b, p):
    term = np.log(b.h / b.c) * np.log(b.h / b.o) + np.log(b.l / b.c) * np.log(b.l / b.o)
    return {"ln_sigma": _ln_sigma_from_var(pr.rolling_mean(term, p["n"]))}


def c10_chop(b, p):
    n = p["n"]
    ratio = pr.safe_div(pr.rolling_sum(b.tr, n), pr.rolling_max(b.h, n) - pr.rolling_min(b.l, n))
    return {"chop": 100.0 * pr.safe_log(ratio) / math.log(10) / math.log10(n)}


# ------------------------------------------------------------------ volumen
def d1_obv(b, p):
    n = p["n"]
    sgn_v = np.where(b.cons, np.sign(b.dC) * b.v, np.nan)
    v_cons = np.where(b.cons, b.v, np.nan)
    num, _, valid = pr.increment_window(sgn_v, n)
    den, _, _ = pr.increment_window(v_cons, n)
    return {"flujo": np.where(valid, pr.safe_div(num, den), np.nan)}


def obv_accumulated(b):
    return np.cumsum(np.where(b.cons, np.sign(np.nan_to_num(b.dC)) * b.v, 0.0))


def d2_mfi(b, p):
    n = p["n"]
    tp_prev = np.r_[np.nan, b.tp[:-1]]
    dtp = np.where(b.cons, b.tp - tp_prev, np.nan)
    rmf = b.tp * b.v
    pos = np.where(b.cons, np.where(dtp > 0, rmf, 0.0), np.nan)
    neg = np.where(b.cons, np.where(dtp < 0, rmf, 0.0), np.nan)
    sp, _, valid = pr.increment_window(pos, n)
    sn, _, _ = pr.increment_window(neg, n)
    return {"mfi": np.where(valid, 100.0 * pr.safe_div(sp, sp + sn), np.nan)}


def d3_cmf(b, p):
    n = p["n"]
    return {"cmf": pr.safe_div(pr.rolling_sum(b.clv0 * b.v, n), pr.rolling_sum(b.v, n))}


def adl_accumulated(b):
    return np.cumsum(b.clv0 * b.v)


def d4_adl(b, p):
    return {"adl_acumulado": adl_accumulated(b)}


def d5_chaikin(b, p):
    adl = adl_accumulated(b)
    ef, kf = _ema_level(adl, p["f"])
    es, ks = _ema_level(adl, p["s"])
    ev, _ = _ema_level(b.v, p["s"])
    return {"chaikin_norm": pr.mask_before(pr.safe_div(ef - es, ev), max(kf, ks))}


def d6_vroc(b, p):
    n = p["n"]
    out = np.full(b.n, np.nan)
    if b.n > n:
        out[n:] = pr.safe_log(pr.safe_div(b.v[n:], b.v[:-n]))
    return {"vroc_log": out}


def d8_vwap(b, p):
    day = np.cumsum(np.r_[True, b.new_day[1:]])                           # reinicia en FJ/FJL (y en roll)
    pv = pd.Series(b.tp * b.v).groupby(day).cumsum().to_numpy()
    vv = pd.Series(b.v).groupby(day).cumsum().to_numpy()
    return {"dist": pr.safe_log(pr.safe_div(b.c, pr.safe_div(pv, vv)))}


# ------------------------------------------------------------------ velas
def e1_body(b, p):
    return {"valor": pr.safe_div(np.abs(b.c - b.o), b.h - b.l)}


def e2_upper(b, p):
    return {"valor": pr.safe_div(b.h - np.maximum(b.o, b.c), b.h - b.l)}


def e3_lower(b, p):
    return {"valor": pr.safe_div(np.minimum(b.o, b.c) - b.l, b.h - b.l)}


def e4_clv(b, p):
    return {"valor": pr.safe_div((b.c - b.l) - (b.h - b.c), b.h - b.l)}


def e5_range(b, p):
    return {"valor": (b.h - b.l) / pr.TICK}


CONTRACT_FUNCTIONS = {
    "A1": a1_sma, "A2": a2_ema, "A3": a3_wma, "A4": a4_hma, "A5": a5_kama, "A6": a6_macd, "A7": a7_ppo,
    "A8": a8_adx, "A9": a9_aroon, "A10": a10_trix, "A11": a11_supertrend, "A12": a12_slope,
    "B1": b1_roc, "B2": b2_momentum, "B3": b3_rsi, "B4": b4_stoch, "B5": b5_williams, "B6": b6_cci,
    "B7": b7_cmo, "B8": b8_tsi, "B9": b9_stochrsi, "B10": b10_uo,
    "C1": c1_atr, "C2": c2_natr, "C3": c3_bollinger, "C4": c4_keltner, "C5": c5_donchian,
    "C6": c6_sd_returns, "C7": c7_parkinson, "C8": c8_garman_klass, "C9": c9_rogers_satchell, "C10": c10_chop,
    "D1": d1_obv, "D2": d2_mfi, "D3": d3_cmf, "D4": d4_adl, "D5": d5_chaikin, "D6": d6_vroc, "D8": d8_vwap,
    "E1": e1_body, "E2": e2_upper, "E3": e3_lower, "E4": e4_clv, "E5": e5_range,
}
CROSS_CONTRACT = {"D7"}                                                     # RVOL cruza rolls


# --------------------------------------------------------------------------- esqueleto y RVOL
def pair_types(df: pd.DataFrame) -> np.ndarray:
    """REINICIO / CONS / HI / FJ / FJL por barra (la barra t con su par (t-1, t))."""
    reset = df["is_first_observation"].to_numpy(bool) | df["is_roll_boundary"].to_numpy(bool)
    cons = df["r_1m_valid"].to_numpy(bool)
    newday = df["is_trading_date_boundary"].to_numpy(bool)
    long_gap = df["delta_minutes"].to_numpy(float) > LONG_GAP_MIN
    out = np.where(newday & long_gap, PAIR_FJL, np.where(newday, PAIR_FJ, PAIR_HI)).astype(object)
    out[cons] = PAIR_CONS
    out[reset] = PAIR_REINICIO
    return out


def _validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"faltan columnas de entrada: {missing}")
    ts = df["timestamp"].to_numpy()
    if not (ts[1:] > ts[:-1]).all():
        raise ValueError("timestamps no estrictamente crecientes (orden canonico requerido)")
    reset = df["is_first_observation"].to_numpy(bool) | df["is_roll_boundary"].to_numpy(bool)
    if df["r_1m_valid"].to_numpy(bool)[reset].any():
        raise ValueError("una barra de reinicio no puede tener par CONS")
    contract = df["contract"].to_numpy()
    changed = np.r_[True, contract[1:] != contract[:-1]]
    if not np.array_equal(changed, reset):
        raise ValueError("los reinicios no coinciden con los cambios de contrato")


def contract_segments(df: pd.DataFrame) -> list[tuple[int, int]]:
    reset = df["is_first_observation"].to_numpy(bool) | df["is_roll_boundary"].to_numpy(bool)
    starts = np.flatnonzero(reset)
    ends = np.r_[starts[1:], len(df)]
    return list(zip(starts.tolist(), ends.tolist()))


def minute_of_day_ny(ts: pd.Series) -> np.ndarray:
    t = pd.to_datetime(ts)
    t = t.dt.tz_localize("UTC") if t.dt.tz is None else t.dt.tz_convert("UTC")
    ny = t.dt.tz_convert(NY_TZ)
    return (ny.dt.hour * 60 + ny.dt.minute).to_numpy()


def calendar_trading_days(df: pd.DataFrame) -> list:
    """Jornadas de negociacion del calendario CME_Equity del proyecto (``session_calendar``) entre la primera
    y la ultima trading_date de la entrada. Todas las trading_date de la entrada deben pertenecer a el."""
    first, last = df["trading_date"].iloc[0], df["trading_date"].iloc[-1]
    table = build_session_schedule(pd.Timestamp(first), pd.Timestamp(last)).table
    days = [d for d in table.index if first <= d <= last]
    missing = set(pd.unique(df["trading_date"])) - set(days)
    if missing:
        raise ValueError(f"trading_date fuera del calendario CME_Equity: {sorted(missing)[:5]}")
    return days


def relative_volume(df: pd.DataFrame, k: int, trading_days: list) -> tuple[np.ndarray, np.ndarray]:
    """D7: ln(V_t / media{V(d, m(t)) : d en J(t), barra observada}).

    J(t) = las k jornadas del CALENDARIO inmediatamente anteriores a trading_date(t), fijas y sin
    sustitucion: una jornada del calendario sin ninguna barra ocupa su lugar en J(t) y solo aporta
    observaciones ausentes. Fines de semana y feriados no son jornadas del calendario. Valida con
    >= ceil(0.8 k) observaciones; antes de la jornada k (contada desde la primera del calendario dentro
    de la entrada) J(t) no esta completo y el valor es NaN (K_jornadas = k).
    Marca rvol_cruza_roll: alguna observacion usada pertenece a otro contrato."""
    pos = {d: i for i, d in enumerate(trading_days)}
    di = np.array([pos[d] for d in df["trading_date"]])
    if (np.diff(di) < 0).any():
        raise ValueError("trading_date no es monotona en el orden canonico")
    minute = minute_of_day_ny(df["timestamp"])
    if pd.Series(di * 1440 + minute).duplicated().any():
        raise ValueError("celdas (trading_date, minuto NY) duplicadas: RVOL no esta definido")
    D = len(trading_days)
    vol = np.full((D, 1440), np.nan)
    cid = np.full((D, 1440), np.nan)
    codes = pd.factorize(df["contract"])[0].astype(float)
    vol[di, minute] = df["volume"].to_numpy(float)
    cid[di, minute] = codes
    obs = ~np.isnan(vol)
    cs_v = np.vstack([np.zeros((1, 1440)), np.cumsum(np.where(obs, vol, 0.0), axis=0)])
    cs_n = np.vstack([np.zeros((1, 1440)), np.cumsum(obs, axis=0)])
    rows = np.arange(D)
    lo = np.clip(rows - k, 0, None)
    sum_v = cs_v[rows] - cs_v[lo]                                           # jornadas rows-k .. rows-1
    cnt = cs_n[rows] - cs_n[lo]
    complete = (rows >= k)[:, None]
    mean = np.where(complete & (cnt >= pr.min_valid(k)), sum_v / np.where(cnt > 0, cnt, 1), np.nan)
    cmin = np.full((D, 1440), np.nan)
    cmax = np.full((D, 1440), np.nan)
    for i in range(k, D):
        win = cid[i - k:i]
        cmin[i] = np.where(np.isnan(win), np.inf, win).min(axis=0)
        cmax[i] = np.where(np.isnan(win), -np.inf, win).max(axis=0)
    m_row = mean[di, minute]
    rv = pr.safe_log(pr.safe_div(df["volume"].to_numpy(float), m_row))
    cmn, cmx = cmin[di, minute], cmax[di, minute]
    cross = np.isfinite(m_row) & ((cmn != codes) | (cmx != codes))
    return rv, cross


# --------------------------------------------------------------------------- marcas generales por ventana
# Ventanas de las marcas generales (enmienda E01-B, reglas_frontera.ventanas_marcas de la configuracion efectiva).
# Tres conceptos distintos, que no se mezclan:
#  - NIVELES: la serie usa niveles (C, H, L, TP, V, valores locales) de las ultimas L barras observadas; una
#    discontinuidad es un par INTERNO de esa ventana (pares que terminan en t-L+2..t: L-1 pares).
#  - PARES: la serie usa P entradas que dependen cada una del par que termina en su barra (TR, BP, incrementos,
#    r_1m); cuentan los P pares que terminan en t-P+1..t, INCLUIDO el que termina en la primera barra.
#  - SESION (VWAP): pares internos de la trading_date en curso; se reinicia con VWAP y no cuenta el par de apertura.
# En los recursivos, K es una ventana de MEMORIA EFECTIVA (peso residual < 0,01), no toda su dependencia historica.
# Expresiones admitidas en la asignacion (cualquier otra se rechaza).
_WINDOW_EXPRESSIONS = {
    "n": lambda p, K: p["n"],
    "n+1": lambda p, K: p["n"] + 1,
    "K": lambda p, K: K,
    "K-1": lambda p, K: K - 1,
    "er": lambda p, K: p["er"],
    "er+1": lambda p, K: p["er"] + 1,
    "c": lambda p, K: p["c"],
    "n+k_ema(n)": lambda p, K: p["n"] + pr.k_weight(pr.ema_alpha(p["n"])),
    "atr+k_wilder(atr)": lambda p, K: p["atr"] + pr.k_weight(pr.wilder_alpha(p["atr"])),
}
MARK_FAMILIES = ("niveles", "pares", "sesion")


def _mark_assignment(cfg: dict) -> dict:
    """Asignacion por elemento de E01-B; sin ella (protocolo sin E01) no hay definicion de marcas."""
    try:
        table = cfg["reglas_frontera"]["ventanas_marcas"]["asignacion"]
    except (KeyError, TypeError) as exc:
        raise ProtocolNotAuthorizedError("la configuracion no incluye la enmienda E01-B (ventanas de marcas)") from exc
    if set(table) != set(CONTRACT_FUNCTIONS):
        raise ProtocolNotAuthorizedError("la asignacion de ventanas no cubre exactamente los 44 elementos por contrato")
    return table


def _mark_spec(rule: dict, p: dict, c: dict) -> dict:
    """Ventanas {"niveles": L o None, "pares": P o None, "sesion": bool} de una configuracion."""
    if not set(rule) <= set(MARK_FAMILIES) or rule.get("sesion", "trading_date") != "trading_date":
        raise ProtocolNotAuthorizedError(f"familia de ventana no admitida: {rule}")
    K = int(c.get("K", c.get("K_min", 0)))
    spec = {"niveles": None, "pares": None, "sesion": "sesion" in rule}
    for family in ("niveles", "pares"):
        if family in rule:
            expr = str(rule[family])
            if expr not in _WINDOW_EXPRESSIONS:
                raise ProtocolNotAuthorizedError(f"expresion de ventana no admitida: {expr}")
            spec[family] = int(_WINDOW_EXPRESSIONS[expr](p, K))
    return spec


def mark_windows(cfg: dict) -> dict[str, dict]:
    """Ventanas de marcas de cada serie del catalogo (sin Aux ni RVOL, que tiene su propia marca)."""
    table = _mark_assignment(cfg)
    out = {}
    for s in catalog_series(cfg):
        if s["panel"] == "Aux" or s["elemento"] == "D7":
            continue
        el = next(e for e in cfg["catalogo"] if e["id"] == s["elemento"])
        c = next(c for c in el["configuraciones"] if c["id"] == s["config"])
        out[s["id"]] = _mark_spec(table[s["elemento"]] or {}, c["params"], c)
    return out


def mark_column(family: str, window: int | None, key: str) -> str:
    if family == "sesion":
        return f"ses__{key}"
    return f"{'niv' if family == 'niveles' else 'par'}{window}__{key}"


MARK_KEYS = ("n_disc_HI", "n_disc_FJ", "n_disc_FJL", "cruza_hueco_largo")


def discontinuity_marks(tipo_par: np.ndarray, long_hi: np.ndarray, segs, n_pairs: int, n_bars: int) -> dict[str, np.ndarray]:
    """Cuenta los pares (t'-1, t') con t' en t-n_pairs+1..t dentro del contrato (la barra de reinicio no tiene
    par). NaN mientras la ventana de ``n_bars`` barras no cabe en el contrato.
    NIVELES de L barras: n_pairs = L - 1, n_bars = L. PARES de P entradas: n_pairs = P, n_bars = P.
    cruza_hueco_largo = algun FJL o HI largo (> 90 min) entre esos pares."""
    N = tipo_par.size
    out = {k: np.full(N, np.nan, dtype=np.float32) for k in MARK_KEYS}
    flags = {"n_disc_HI": tipo_par == PAIR_HI, "n_disc_FJ": tipo_par == PAIR_FJ, "n_disc_FJL": tipo_par == PAIR_FJL,
             "cruza_hueco_largo": (tipo_par == PAIR_FJL) | ((tipo_par == PAIR_HI) & long_hi)}
    for a, z in segs:
        n = z - a
        if n < n_bars:
            continue
        t = np.arange(n_bars - 1, n)
        lo = np.maximum(t - n_pairs + 1, 0)
        for key, flag in flags.items():
            f = flag[a:z].astype(np.int64)
            f[0] = 0                                                    # reinicio: sin par
            cs = np.r_[0, np.cumsum(f)]
            cnt = cs[t + 1] - cs[lo]
            out[key][a + n_bars - 1:z] = (cnt > 0) if key == "cruza_hueco_largo" else cnt
    return out


def session_marks(tipo_par: np.ndarray, long_hi: np.ndarray) -> dict[str, np.ndarray]:
    """E01-B, VWAP: pares internos de la trading_date en curso hasta t (acumulativa). Se reinicia donde VWAP
    (FJ, FJL, roll) y el par de apertura no se cuenta; dentro de una sesion solo puede haber HI."""
    opens = np.isin(tipo_par, [PAIR_REINICIO, PAIR_FJ, PAIR_FJL])
    session = np.cumsum(opens)
    hi = (tipo_par == PAIR_HI).astype(np.int64)
    n_hi = pd.Series(hi).groupby(session).cumsum().to_numpy()
    n_long = pd.Series(hi * long_hi).groupby(session).cumsum().to_numpy()
    zero = np.zeros(tipo_par.size, dtype=np.float32)
    return {"n_disc_HI": n_hi.astype(np.float32), "n_disc_FJ": zero, "n_disc_FJL": zero.copy(),
            "cruza_hueco_largo": (n_long > 0).astype(np.float32)}


def series_discontinuity(marks: pd.DataFrame, spec: dict) -> dict[str, np.ndarray]:
    """Marcas de UNA serie para S1/S2 (E01-B, sensibilidades.marcas_por_serie): union de sus ventanas. Como
    todas terminan en t, la union es la ventana con mas pares; no se cuenta ningun par fuera de las entradas o
    niveles realmente usados (sin falsos positivos). Sin ventana: marcas 0."""
    if spec.get("sesion"):
        if spec["niveles"] or spec["pares"]:
            raise ProtocolNotAuthorizedError("la ventana de sesion no se combina con otras familias")
        return {k: marks[mark_column("sesion", None, k)].to_numpy() for k in MARK_KEYS}
    cands = [(w - 1, mark_column("niveles", w, "")) for w in [spec["niveles"]] if w] + \
            [(w, mark_column("pares", w, "")) for w in [spec["pares"]] if w]
    if not cands:
        return {k: np.zeros(len(marks), dtype=np.float32) for k in MARK_KEYS}
    _, prefix = max(cands)
    return {k: marks[prefix + k].to_numpy() for k in MARK_KEYS}


# --------------------------------------------------------------------------- catalogo completo
VALIDITY_COLUMNS = ["r_1m_valid", "is_first_observation", "is_roll_boundary", "is_trading_date_boundary",
                    "is_non_consecutive_minute", "delta_minutes"]


def join_inputs(variables: pd.DataFrame, validity: pd.DataFrame) -> pd.DataFrame:
    """Une las variables y la mascara de validez de TDA-04 fila a fila (mismos timestamps, mismo orden)."""
    v = variables.sort_values("timestamp").reset_index(drop=True)
    m = validity.sort_values("timestamp").reset_index(drop=True)
    if len(v) != len(m) or not (v["timestamp"].to_numpy() == m["timestamp"].to_numpy()).all():
        raise ValueError("variables y mascara de validez no comparten timestamps en el mismo orden")
    out = v.copy()
    for col in VALIDITY_COLUMNS:
        out[col] = m[col].to_numpy()
    return out


def compute_catalog(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PRODUCCION: calcula las 146 series y las marcas con la configuracion EFECTIVA verificada: protocolo
    CONGELADO + enmiendas requeridas APROBADAS (``tda17_enmiendas.load_effective_protocol``).

    No admite ninguna configuracion externa. Si una enmienda requerida falta, no esta aprobada o esta alterada,
    falla (``ProtocolNotAuthorizedError``): nunca calcula con el comportamiento anterior a la enmienda."""
    if SYNTHETIC_MARKER in df.columns:
        raise ValueError("la via de produccion no acepta marcos marcados como sinteticos de prueba")
    from ohlcv_dataroad.ingest.tda17_enmiendas import load_effective_protocol
    return _compute(df, load_effective_protocol())


def compute_catalog_synthetic(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SOLO PRUEBAS: permite una configuracion arbitraria (p. ej. sintetica o no congelada), pero exclusivamente
    sobre marcos sinteticos que llevan la columna ``SYNTHETIC_MARKER`` en todas sus filas. Las entradas de
    TDA-04 nunca la tienen; la via de produccion la rechaza."""
    if SYNTHETIC_MARKER not in df.columns or not df[SYNTHETIC_MARKER].astype(bool).all():
        raise ValueError("compute_catalog_synthetic solo acepta marcos sinteticos marcados")
    _validate_cfg_structure(cfg)
    return _compute(df.drop(columns=[SYNTHETIC_MARKER]), cfg)


def _compute(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (series, marcas): ``series`` con una columna por serie (ids del YAML, mismo indice que
    ``df``); ``marcas`` con el tipo de par, las marcas generales por ventana (``niv{L}__...``,
    ``par{P}__...``, ``ses__...``), las de Choppiness y las de RVOL. ``mark_windows(cfg)`` da las ventanas de
    cada serie y ``series_discontinuity`` sus marcas. Exige la enmienda E01 en ``cfg``."""
    require_e01(cfg)
    _validate(df)
    df = df.reset_index(drop=True)
    cols: dict[str, np.ndarray] = {}
    N = len(df)
    tipo = pair_types(df)
    marks = {"tipo_par": tipo}
    segs = contract_segments(df)
    trading_days = calendar_trading_days(df)
    for el in cfg["catalogo"]:
        eid = el["id"]
        if eid in CROSS_CONTRACT:
            for c in el["configuraciones"]:
                rv, cross = relative_volume(df, c["params"]["k_jornadas"], trading_days)
                for s in el["salidas"]:
                    cols[f"{eid}_{c['id']}__{s['id']}"] = rv
                marks[f"{eid}_{c['id']}__rvol_cruza_roll"] = cross
            continue
        fn = CONTRACT_FUNCTIONS[eid]
        for c in el["configuraciones"]:
            outs = {s["id"]: np.full(N, np.nan) for s in el["salidas"]}
            for a, z in segs:
                b = _bars(df, a, z)
                res = fn(b, c["params"])
                for sid in outs:
                    outs[sid][a:z] = res[sid]
            for sid, arr in outs.items():
                cols[f"{eid}_{c['id']}__{sid}"] = arr
        for s in el.get("salidas_auxiliares_elemento", []):
            arr = np.full(N, np.nan)
            for a, z in segs:
                arr[a:z] = obv_accumulated(_bars(df, a, z))
            cols[f"{eid}__{s['id']}"] = arr
    long_hi = df["delta_minutes"].to_numpy(float) > LONG_GAP_MIN
    specs = mark_windows(cfg).values()
    for family in ("niveles", "pares"):
        for w in sorted({s[family] for s in specs if s[family]}):
            n_pairs = w - 1 if family == "niveles" else w
            for key, arr in discontinuity_marks(tipo, long_hi, segs, n_pairs, w).items():
                marks[mark_column(family, w, key)] = arr
    if any(s["sesion"] for s in specs):
        for key, arr in session_marks(tipo, long_hi).items():
            marks[mark_column("sesion", None, key)] = arr
    for c in next(e for e in cfg["catalogo"] if e["id"] == "C10")["configuraciones"]:
        n = c["params"]["n"]
        ndisc = np.full(N, np.nan)
        for a, z in segs:
            b = _bars(df, a, z)
            ndisc[a:z] = pr.rolling((~b.cons).astype(float), n, lambda w: w.sum(axis=-1))
        chop = cols[f"C10_{c['id']}__chop"]
        marks[f"C10_{c['id']}__chop_n_disc"] = ndisc
        marks[f"C10_{c['id']}__chop_fuera_rango"] = np.isfinite(chop) & ((chop < 0) | (chop > 100))
    expected = [s["id"] for s in catalog_series(cfg)]
    series = pd.DataFrame({k: cols[k] for k in expected}, index=df.index)
    return series, pd.DataFrame(marks, index=df.index)


def _bars(df: pd.DataFrame, a: int, z: int) -> Bars:
    sl = slice(a, z)
    cons = df["r_1m_valid"].to_numpy(bool)[sl].copy()
    cons[0] = False
    new_day = df["is_trading_date_boundary"].to_numpy(bool)[sl].copy()
    new_day[0] = True
    f = lambda col: df[col].to_numpy(float)[sl]                              # noqa: E731
    return Bars(o=f("open"), h=f("high"), l=f("low"), c=f("close"), v=f("volume"), cons=cons, new_day=new_day)


def require_e01(cfg: dict) -> None:
    """El motor implementa E01 (ADX E01-A y marcas E01-B): rechaza una configuracion que no la contiene."""
    a8 = next((e for e in cfg.get("catalogo", []) if e.get("id") == "A8"), {})
    if (a8.get("dx_no_definido") or {}).get("regla") != "OMITIR_EN_ETAPA_2":
        raise ProtocolNotAuthorizedError("la configuracion no incluye la enmienda E01-A (ADX con DX no definido)")
    _mark_assignment(cfg)


def build_panel_p(series: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """PRODUCCION: las 134 columnas oficiales (128 series P + 6 referencias de TDA-04, ln_volume = ln V)."""
    from ohlcv_dataroad.ingest.tda17_enmiendas import load_effective_protocol
    return _build_panel_p(series, df, load_effective_protocol())


def _build_panel_p(series: pd.DataFrame, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    refs = {}
    for r in cfg["referencias_tda04"]["en_panel_P"]:
        if r["id"] == "ln_volume":
            refs["ln_volume"] = pr.safe_log(df["volume"].to_numpy(float))
        else:
            refs[r["id"]] = df[r["id"]].to_numpy(float)
    cols = p_columns(cfg)
    data = {c: (series[c].to_numpy() if c in series.columns else refs[c]) for c in cols}
    return pd.DataFrame(data, index=series.index)
