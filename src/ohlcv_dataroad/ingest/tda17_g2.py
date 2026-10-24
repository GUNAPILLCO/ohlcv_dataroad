"""TDA-17 -- fase 2: infraestructura de los controles G2 (g2 del protocolo efectivo; D11, D24).

Generativos (criterio V1): G2_1a y G2_1b (permutacion de vectores de barra en ticks) y G2_2 (t de Student con
puente browniano, especificado al detalle en g2.generativos.G2_2.generador). Desplazamientos (criterio V1'):
G2_3a (desfases relativos entre pares sobre la rejilla jornada x minuto) y G2_3b (desplazamientos circulares de
las columnas de B con garantia por pares y certificado previo de imposibilidad, salvaguarda v4b).

Este modulo NO ejecuta ninguna simulacion: recibe esqueletos y matrices. No lee ficheros, ni datos brutos ni
el holdout. Lo que el protocolo no define operativamente (T3/V2, banda max-T, k_PA y V3) lanza
``PendingDefinitionError`` (ver ``tda17_analysis.PENDIENTES``). El control de nulos sobre replicas G2 es INFORMATIVO
(E02, A4-a). Cualquier uso de produccion de G2 obtiene la configuracion solo con ``official_config(P2)``.

Convenciones de implementacion (no metodologicas) que el protocolo no fija y que quedan a la vista para la
auditoria: G2_1a/G2_1b y las replicas de G2_3b usan ``numpy.random.default_rng(semilla)`` (PCG64), un unico
generador por semilla consumido en orden determinista (grupos de G2_1 por orden de aparicion; replicas de
G2_3b en orden 0..199).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ohlcv_dataroad.ingest import tda17_indicators as ind
from ohlcv_dataroad.ingest import tda17_primitives as pr
from ohlcv_dataroad.ingest.tda07_marginal_distribution import compute_moments_quantiles
from ohlcv_dataroad.ingest.tda17_analysis import official_config, pending, rank_check

SUBSTEPS = 30


def q_tick(p):
    """g2.G2_2.generador.tick_redondeo: q(p) = 0,25 * numpy.round(p / 0,25) (mitad al par)."""
    return pr.TICK * np.round(np.asarray(p, np.float64) / pr.TICK)


# =========================================================================== V1: invariantes
def v1_ohlc_check(o, h, l, c) -> list[str]:
    """V1 (invariantes OHLC y grilla de tick): precios en la rejilla, L > 0 y L <= min(O, C) <= max(O, C) <= H."""
    out = []
    for name, x in (("open", o), ("high", h), ("low", l), ("close", c)):
        x = np.asarray(x, np.float64)
        if not np.isfinite(x).all() or not np.array_equal(x / pr.TICK, np.round(x / pr.TICK)):
            out.append(f"{name} fuera de la rejilla de tick o no finito")
    o, h, l, c = (np.asarray(x, np.float64) for x in (o, h, l, c))
    if not (l > 0).all():
        out.append("precio minimo <= 0")
    if not ((l <= np.minimum(o, c)) & (np.maximum(o, c) <= h)).all():
        out.append("invariante L <= min(O, C) <= max(O, C) <= H violado")
    return out


def v1_identity_check(residuals: dict[str, dict], cfg: dict) -> list[str]:
    """V1: identidades registradas con residuo <= residuo_max (1e-9). N10 (DIAGNOSTICO_NO_UNIVERSAL) no se exige."""
    tol = cfg["analisis"]["exactitud"]["residuo_max"]
    return [f"identidad {k}: residuo {v['residuo_relativo']:.3e} > {tol}" for k, v in residuals.items()
            if not (v["residuo_relativo"] <= tol)]


CANONICAL_COLUMNS = ["timestamp", "source_file", "contract", "trading_date", "segment_id",
                     "open", "high", "low", "close", "volume"]
SKELETON_COLUMNS = ["timestamp", "contract", "trading_date", "r_1m_valid", "is_first_observation", "is_roll_boundary",
                    "is_trading_date_boundary", "is_non_consecutive_minute", "delta_minutes"]


def rebuild_inputs(replica: pd.DataFrame) -> pd.DataFrame:
    """Entrada del motor para una replica G2: variables y mascara de validez recalculadas con las MISMAS
    funciones de TDA-04 sobre los precios generados. El esqueleto (timestamps, contratos, CONS, rolls) debe
    salir identico al de la replica; si no, se detiene."""
    from ohlcv_dataroad.ingest.tda04_analysis_variables import build_analysis_variables, build_return_validity_mask
    canonical = replica[CANONICAL_COLUMNS].reset_index(drop=True)
    roll = canonical[["timestamp"]].copy()
    roll["is_roll_boundary"] = replica["is_roll_boundary"].to_numpy(bool)
    validity = build_return_validity_mask(canonical, roll)
    out = ind.join_inputs(build_analysis_variables(canonical, validity), validity)
    ref = replica.reset_index(drop=True)
    for col in SKELETON_COLUMNS:
        if not pd.Series(out[col].to_numpy()).equals(pd.Series(ref[col].to_numpy())):   # NaN == NaN
            raise ValueError(f"la replica cambia el esqueleto: {col}")
    return out


# =========================================================================== G2_1a / G2_1b
def g2_1_generate(df: pd.DataFrame, seed: int, by_minute: bool) -> pd.DataFrame:
    """G2_1a (``by_minute`` False) y G2_1b (True): permutacion, dentro de cada contrato (y, en G2_1b, de cada
    minuto del dia NY), de los vectores en ticks (O - C_prev, H - O, L - O, C - O, V) de las barras CONS; las
    barras no CONS (reinicio, HI, FJ, FJL) conservan su vector en su sitio ("fronteras en su sitio").
    Reconstruccion por contrato: la primera barra conserva su OHLC; despues O_t = C_(t-1) + g1, H = O + g2,
    L = O + g3, C = O + g4, V = g5. El esqueleto (timestamps, contratos, trading_date, CONS) no cambia."""
    ind._validate(df)
    df = df.reset_index(drop=True)
    tk = {k: np.round(df[k].to_numpy(np.float64) / pr.TICK) for k in ("open", "high", "low", "close")}
    for k, v in tk.items():
        if not np.array_equal(v * pr.TICK, df[k].to_numpy(np.float64)):
            raise ValueError(f"{k} fuera de la rejilla de tick: G2_1 opera en ticks")
    o, h, l, c = (tk[k].astype(np.int64) for k in ("open", "high", "low", "close"))
    vol = df["volume"].to_numpy(np.float64)
    segs = ind.contract_segments(df)
    first = np.zeros(len(df), bool)
    first[[a for a, _ in segs]] = True
    c_prev = np.r_[0, c[:-1]]
    vec = np.column_stack([o - c_prev, h - o, l - o, c - o]).astype(np.int64)
    cons = df["r_1m_valid"].to_numpy(bool) & ~first
    key = df["contract"].astype(str).to_numpy()
    if by_minute:
        key = np.char.add(np.char.add(key.astype(str), "|"), ind.minute_of_day_ny(df["timestamp"]).astype(str))
    rng = np.random.default_rng(seed)
    new_vec, new_vol = vec.copy(), vol.copy()
    codes, _ = pd.factorize(key)                                            # orden de primera aparicion
    for g in range(codes.max() + 1):
        idx = np.flatnonzero((codes == g) & cons)
        if idx.size > 1:
            src = idx[rng.permutation(idx.size)]
            new_vec[idx], new_vol[idx] = vec[src], vol[src]
    O, H, L, C = (np.empty(len(df), np.int64) for _ in range(4))
    for a, z in segs:
        O[a], H[a], L[a], C[a] = o[a], h[a], l[a], c[a]
        g = new_vec[a + 1:z]
        C[a + 1:z] = c[a] + np.cumsum(g[:, 0] + g[:, 3])
        O[a + 1:z] = C[a:z - 1] + g[:, 0]
        H[a + 1:z] = O[a + 1:z] + g[:, 1]
        L[a + 1:z] = O[a + 1:z] + g[:, 2]
    out = df.copy()
    for k, arr in (("open", O), ("high", H), ("low", L), ("close", C)):
        out[k] = arr.astype(np.float64) * pr.TICK
    out["volume"] = new_vol
    return out


# =========================================================================== G2_2
def g2_2_scale(cfg: dict) -> float:
    g = cfg["g2"]["generativos"]["G2_2"]
    nu = g["t_student"]["df"]
    if not nu > 2:
        raise ValueError("G2_2 NO_EVALUABLE: df <= 2")
    return g["sigma_r_1m"] * math.sqrt((nu - 2) / nu)


def g2_2_innovations(seed: int, n_rows: int, cfg: dict) -> np.ndarray:
    """Flujo 0 de SeedSequence(semilla).spawn(3): standard_t(df, size=n_filas) en UNA llamada, escalado."""
    s0 = np.random.SeedSequence(seed).spawn(3)[0]
    nu = cfg["g2"]["generativos"]["G2_2"]["t_student"]["df"]
    return g2_2_scale(cfg) * np.random.Generator(np.random.PCG64(s0)).standard_t(nu, size=n_rows)


def g2_2_verify(seed: int, cfg: dict, n_rows: int | None = None) -> dict:
    """g2.G2_2.verificacion sobre el vector completo de innovaciones escaladas (antes del redondeo): curtosis en
    exceso recortada (definicion de TDA-07, compute_moments_quantiles) y RIC relativo; con n_filas del protocolo
    se comparan ademas con los valores esperados por semilla. Fallo -> error de implementacion (bloqueo)."""
    g = cfg["g2"]["generativos"]["G2_2"]
    v = g["verificacion"]
    n = cfg["entradas"]["esperado"]["filas"] if n_rows is None else n_rows
    r = g2_2_innovations(seed, n, cfg)
    from scipy import stats
    nu = g["t_student"]["df"]
    iqr_th = g2_2_scale(cfg) * (stats.t.ppf(0.75, nu) - stats.t.ppf(0.25, nu))
    kt = compute_moments_quantiles(r)["kurtosis_excess_trimmed"]
    ric = float((np.quantile(r, 0.75) - np.quantile(r, 0.25)) / iqr_th)
    ok = abs(kt - v["curtosis_recortada"]["objetivo"]) <= v["curtosis_recortada"]["tolerancia_abs"] and \
        abs(ric - v["ric_relativo"]["objetivo"]) <= v["ric_relativo"]["tolerancia_abs"]
    out = {"semilla": seed, "n": n, "curtosis_recortada": kt, "ric_relativo": ric, "dentro_de_tolerancia": ok}
    exp = v["valores_esperados_semillas"].get(str(seed))
    if exp is not None and n == cfg["entradas"]["esperado"]["filas"]:
        out["coincide_valores_esperados"] = math.isclose(kt, exp["curtosis_recortada"], rel_tol=1e-9) and \
            math.isclose(ric, exp["ric_relativo"], rel_tol=1e-9)
    return out


def canonical_order(skel: pd.DataFrame) -> np.ndarray:
    """g2.G2_2.generador.orden_filas: ordenacion estable por (contract como cadena, timestamp)."""
    tmp = pd.DataFrame({"c": skel["contract"].astype(str).to_numpy(), "t": skel["timestamp"].to_numpy()})
    return tmp.sort_values(["c", "t"], kind="stable").index.to_numpy()


def g2_2_generate(skel: pd.DataFrame, seed: int, cfg: dict, *, check_rows: bool = True,
                  chunk: int = 250_000) -> pd.DataFrame:
    """Generador G2_2 (g2.generativos.G2_2.generador) sobre un esqueleto con contract, timestamp, open (para el
    ancla A_c) y volume. Devuelve las filas en el orden canonico con O, H, L, C, V, la innovacion r, el camino
    latente x e I (indice de volumen). ``check_rows`` exige n_filas = entradas.esperado.filas (produccion)."""
    g = cfg["g2"]["generativos"]["G2_2"]
    nu, sigma = g["t_student"]["df"], g["sigma_r_1m"]
    order = canonical_order(skel)
    sk = skel.iloc[order].reset_index(drop=True)
    n = len(sk)
    if check_rows and n != cfg["entradas"]["esperado"]["filas"]:
        raise ValueError(f"n_filas = {n} distinto de entradas.esperado.filas")
    s0, s1, s2 = np.random.SeedSequence(seed).spawn(3)
    r = g2_2_scale(cfg) * np.random.Generator(np.random.PCG64(s0)).standard_t(nu, size=n)
    z = np.random.Generator(np.random.PCG64(s1)).standard_normal(size=(n, SUBSTEPS))
    idx = np.random.Generator(np.random.PCG64(s2)).integers(0, n, size=n)
    contract = sk["contract"].astype(str).to_numpy()
    first = np.r_[True, contract[1:] != contract[:-1]]
    if pd.Series(contract[first]).duplicated().any():
        raise ValueError("un contrato aparece en tramos no contiguos del orden canonico")
    opens = sk["open"].to_numpy(np.float64)
    starts = np.flatnonzero(first)
    ends = np.r_[starts[1:], n]
    x = np.empty(n)
    x_prev = np.empty(n)
    anchor = np.empty(n)
    for a, b in zip(starts, ends):
        la = math.log(opens[a])
        x[a:b] = np.cumsum(np.r_[la, r[a:b]])[1:]
        x_prev[a] = la
        x_prev[a + 1:b] = x[a:b - 1]
        anchor[a:b] = opens[a]
    k = np.arange(SUBSTEPS + 1) / SUBSTEPS
    bmax, bmin = np.empty(n), np.empty(n)
    for a in range(0, n, chunk):
        zz = z[a:a + chunk]
        S = np.concatenate([np.zeros((zz.shape[0], 1)), np.cumsum(zz, axis=1)], axis=1)
        b = x_prev[a:a + chunk, None] + k[None, :] * r[a:a + chunk, None] + \
            sigma * (S - k[None, :] * S[:, -1:]) / math.sqrt(SUBSTEPS)
        bmax[a:a + chunk], bmin[a:a + chunk] = b.max(axis=1), b.min(axis=1)
    C = q_tick(np.exp(x))
    O = np.where(first, anchor, np.r_[np.nan, C[:-1]])
    H = np.maximum(np.maximum(O, C), q_tick(np.exp(bmax)))
    L = np.minimum(np.minimum(O, C), q_tick(np.exp(bmin)))
    out = sk[["timestamp", "contract"]].copy()
    out["open"], out["high"], out["low"], out["close"] = O, H, L, C
    out["volume"] = sk["volume"].to_numpy(np.float64)[idx]
    out["r"], out["x"], out["I"] = r, x, idx
    out["fila_esqueleto"] = order                       # trazabilidad: posicion de la fila g en el esqueleto recibido
    return out


def frontier_counts(df: pd.DataFrame) -> dict[str, int]:
    """Fronteras del esqueleto: pares CONS, HI, FJ, FJL, reinicios y rolls (is_roll_boundary)."""
    tipo = ind.pair_types(df)
    out = {t: int((tipo == t).sum()) for t in (ind.PAIR_CONS, ind.PAIR_HI, ind.PAIR_FJ, ind.PAIR_FJL, ind.PAIR_REINICIO)}
    out["ROLL"] = int(df["is_roll_boundary"].astype(bool).sum())
    return out


def g2_2_restore(gen: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve la replica G2_2 en el ORDEN TEMPORAL de ``df`` (la entrada del motor): cada fila generada vuelve a
    la fila del esqueleto de la que procede (``fila_esqueleto``), con sus metadatos originales (timestamp,
    source_file, contract, trading_date, segment_id y mascara de validez) y el OHLCV generado. Rechaza filas
    duplicadas, perdidas o desalineadas (timestamp o contrato distintos de los del esqueleto)."""
    ref = df.reset_index(drop=True)
    n = len(ref)
    pos = gen["fila_esqueleto"].to_numpy()
    if len(gen) != n:
        raise ValueError(f"replica G2_2 con {len(gen)} filas para un esqueleto de {n}: filas perdidas o sobrantes")
    if pd.Series(pos).duplicated().any():
        raise ValueError("replica G2_2 con filas del esqueleto duplicadas")
    if not np.array_equal(np.sort(pos), np.arange(n)):
        raise ValueError("replica G2_2 con filas del esqueleto perdidas o fuera de rango")
    back = gen.iloc[np.argsort(pos, kind="stable")].reset_index(drop=True)
    for col in ("timestamp", "contract"):
        if not np.array_equal(back[col].astype(str).to_numpy(), ref[col].astype(str).to_numpy()):
            raise ValueError(f"replica G2_2 desalineada con el esqueleto: {col}")
    replica = ref[list(dict.fromkeys(CANONICAL_COLUMNS + SKELETON_COLUMNS))].copy()
    for col in ("open", "high", "low", "close", "volume"):
        replica[col] = back[col].to_numpy(np.float64)
    return replica


def g2_2_replica(df: pd.DataFrame, seed: int, cfg: dict, *, check_rows: bool = True) -> pd.DataFrame:
    """Replica G2_2 integrada con TDA-04 a partir de la entrada del motor ``df`` (orden temporal, variables y
    mascara de TDA-04): (1) generacion en el orden canonico congelado (contract como cadena, timestamp), sin
    cambios; (2) vuelta al orden temporal fila a fila con sus metadatos (``g2_2_restore``); (3) V1 de OHLC y
    rejilla; (4) variables y mascara de validez recalculadas con las funciones de TDA-04, comprobando que el
    esqueleto y las fronteras CONS, HI, FJ, FJL y ROLL son identicos a los de ``df``."""
    ind._validate(df)
    ref = df.reset_index(drop=True)
    gen = g2_2_generate(ref[["contract", "timestamp", "open", "volume"]], seed, cfg, check_rows=check_rows)
    replica = g2_2_restore(gen, ref)
    errs = v1_ohlc_check(replica["open"], replica["high"], replica["low"], replica["close"])
    if errs:
        raise ValueError(f"V1 de G2_2: {errs}")
    inputs = rebuild_inputs(replica)
    if not np.array_equal(ind.pair_types(inputs), ind.pair_types(ref)) or frontier_counts(inputs) != frontier_counts(ref):
        raise ValueError("la replica G2_2 cambia las fronteras CONS/HI/FJ/FJL/ROLL")
    return inputs


# =========================================================================== G2_3a
def g2_3a_grid(day_codes: np.ndarray, minute: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rejilla jornada x minuto NY de una celda: (present (D, M), valores (D, M, p), minutos). Sin imputacion:
    la mascara son las filas recibidas (poblacion). Claves (jornada, minuto) duplicadas detienen (V1')."""
    days, di = np.unique(day_codes, return_inverse=True)
    mins, mi = np.unique(minute, return_inverse=True)
    if pd.Series(di * mins.size + mi).duplicated().any():
        raise ValueError("V1': claves (jornada, minuto) duplicadas en la rejilla")
    values = np.asarray(values, np.float64)
    if values.ndim == 1:
        values = values[:, None]
    present = np.zeros((days.size, mins.size), bool)
    grid = np.full((days.size, mins.size, values.shape[1]), np.nan)
    present[di, mi] = True
    grid[di, mi] = values
    return present, grid, mins


def omega_sizes(present: np.ndarray) -> np.ndarray:
    """|Omega_k| = numero de celdas (d, m) presentes en d y en d + k (circular), k = 0..D-1."""
    return np.array([int((present & np.roll(present, -k, axis=0)).sum()) for k in range(present.shape[0])])


def g2_3a_offsets(present: np.ndarray, cfg: dict) -> dict:
    """Desfases elegibles k en {k_min, ..., floor(D/2)} con |Omega_k| >= omega_min_fraccion * N; menos de
    desfases_min -> NO_EVALUABLE; mas de desfases_max -> muestreo sin reemplazo con la semilla del protocolo."""
    a = cfg["g2"]["desplazamientos"]["G2_3a"]
    D, N = present.shape[0], int(present.sum())
    om = omega_sizes(present)
    elig = [k for k in range(a["k_min"], D // 2 + 1) if om[k] >= a["omega_min_fraccion"] * N]
    base = {"D": D, "N": N, "elegibles": len(elig), "omega": {k: int(om[k]) for k in elig}}
    if len(elig) < a["desfases_min"]:
        return {**base, "estado": "NO_EVALUABLE", "desfases": []}
    if len(elig) <= a["desfases_max"]:
        return {**base, "estado": "EVALUABLE", "desfases": elig}
    chosen = np.random.default_rng(a["semilla"]).choice(elig, a["desfases_max"], replace=False)
    return {**base, "estado": "EVALUABLE", "desfases": sorted(int(x) for x in chosen)}


def g2_3a_pair_corr(xi: np.ndarray, xj: np.ndarray, present: np.ndarray, k: int) -> float:
    """Correlacion simetrizada 0,5 (C(i, j, k) + C(j, i, k)) con C(a, b, k) = corr(a(d, m), b(d + k, m)) sobre
    las celdas presentes en ambas jornadas (Omega_k); la alineacion de minuto es exacta por construccion."""
    mk = present & np.roll(present, -k, axis=0)

    def C(a, b):
        return float(np.corrcoef(a[mk], np.roll(b, -k, axis=0)[mk])[0, 1])
    return 0.5 * (C(xi, xj) + C(xj, xi))


def v1_prime_3a(present: np.ndarray, grid: np.ndarray, k: int) -> list[str]:
    """V1' de G2_3a: el desplazamiento de jornadas es una biyeccion de la rejilla (mismos valores) y la
    coincidencia de minuto es del 100 % (columnas fijas); informa |Omega_k| en g2_3a_offsets."""
    out = []
    rolled_p, rolled = np.roll(present, -k, axis=0), np.roll(grid, -k, axis=0)
    if int(rolled_p.sum()) != int(present.sum()) or \
            not np.array_equal(np.sort(rolled[rolled_p], axis=0), np.sort(grid[present], axis=0)):
        out.append("el desplazamiento no es una biyeccion de la rejilla")
    return out


# =========================================================================== G2_3b
def alignment_table(minute: np.ndarray) -> np.ndarray:
    """a(delta) exacta por FFT: fraccion de filas u con minuto(u) = minuto(u + delta) (circular), delta = 0..N-1."""
    minute = np.asarray(minute)
    N = minute.size
    acc = np.zeros(N)
    for m in np.unique(minute):
        f = np.fft.rfft((minute == m).astype(float))
        acc += np.fft.irfft(f * np.conj(f), n=N)
    return np.rint(acc) / N


def bad_mask(a: np.ndarray, L_min: int, a_max: float) -> np.ndarray:
    """Bad = {delta != 0 : a(delta) > a_max o distancia circular(delta) < L_min} (simetrico)."""
    N = a.size
    d = np.arange(N)
    bad = (a > a_max) | (np.minimum(d, N - d) < L_min)
    bad[0] = False
    return bad


def _clique_block(bad: np.ndarray) -> int:
    run = 1
    while run < bad.size and bad[run]:
        run += 1
    return run


def _clique_progression(a: np.ndarray, bad: np.ndarray, a_max: float, top: int) -> int:
    N = a.size
    half = np.arange(1, N // 2 + 1)
    cand = half[bad[half] & (a[half] > a_max)]
    best = 1
    for s in cand[np.argsort(-a[cand], kind="stable")][:top]:
        k = 1
        while True:
            dk = (k * int(s)) % N
            if dk == 0 or not bad[dk]:
                break
            k += 1
        best = max(best, k)
    return best


def g2_3b_viability(minute: np.ndarray, n_days: int, cfg: dict) -> dict:
    """Diagnostico previo de G2_3b (solo timestamps de la poblacion de la celda, en orden temporal):
    separacion |D| / (2p) >= separacion_min_jornadas; certificado de imposibilidad floor(N / omega_lb) < p con
    omega acotado por el bloque {0..L_min-1} y por progresiones de los 50 pasos de mayor alineacion."""
    b = cfg["g2"]["desplazamientos"]["G2_3b"]
    p = b["p"]
    N = int(np.asarray(minute).size)
    base = {"N": N, "D": n_days, "p": p, "L_min": N // (2 * p)}
    if n_days / (2 * p) < b["separacion_min_jornadas"]:
        return {**base, "estado": "NO_EVALUABLE", "motivo": "SEPARACION_INSUFICIENTE", "intentos": 0}
    a = alignment_table(minute)
    bad = bad_mask(a, base["L_min"], b["a_max"])
    omega = max(_clique_block(bad), _clique_progression(a, bad, b["a_max"], 50))
    out = {**base, "omega_lb": omega, "alpha_ub": N // omega, "alineacion": a, "bad": bad}
    if N // omega < p:
        return {**out, "estado": "NO_EVALUABLE", "motivo": "CERTIFICADO_IMPOSIBILIDAD", "intentos": 0}
    return {**out, "estado": "VIABLE"}


def draw_offsets(bad: np.ndarray, p: int, rng: np.random.Generator, max_tries: int) -> tuple[list[int] | None, int]:
    """Un juego de p desfases con todas las diferencias fuera de Bad (garantia POR PARES)."""
    N = bad.size
    s: list[int] = []
    tries = 0
    for _ in range(p):
        for _t in range(max_tries):
            tries += 1
            c = int(rng.integers(0, N))
            if all((c - x) % N != 0 and not bad[(c - x) % N] for x in s):
                s.append(c)
                break
        else:
            return None, tries
    return s, tries


def g2_3b_replicas(viability: dict, cfg: dict) -> dict:
    """Desfases de las replicas de G2_3b (si la celda es VIABLE). Mas de fraccion_max_replicas_no_generadas
    -> NO_EVALUABLE."""
    b = cfg["g2"]["desplazamientos"]["G2_3b"]
    if viability["estado"] != "VIABLE":
        return {"estado": viability["estado"], "motivo": viability.get("motivo"), "replicas": []}
    rng = np.random.default_rng(b["semilla"])
    reps, failed, tries = [], 0, 0
    for _ in range(b["replicas"]):
        s, t = draw_offsets(viability["bad"], b["p"], rng, b["intentos_max_por_columna"])
        tries += t
        if s is None:
            failed += 1
        else:
            reps.append(s)
    frac = failed / b["replicas"]
    estado = "NO_EVALUABLE" if frac > b["fraccion_max_replicas_no_generadas"] else "GENERADO"
    return {"estado": estado, "motivo": "REPLICAS_NO_GENERADAS" if estado == "NO_EVALUABLE" else None,
            "replicas": reps, "no_generadas": failed, "intentos": tries}


def shift_columns(B: np.ndarray, offsets: list[int]) -> np.ndarray:
    """Columna j desplazada circularmente s_j filas (permutacion ciclica exacta)."""
    B = np.asarray(B)
    return np.column_stack([np.roll(B[:, j], -s) for j, s in enumerate(offsets)])


def v1_prime_3b(offsets: list[int], bad: np.ndarray, B: np.ndarray, shifted: np.ndarray) -> list[str]:
    """V1' de G2_3b: cada columna es una permutacion ciclica exacta y todas las parejas cumplen la garantia."""
    out = []
    N = bad.size
    for j, s in enumerate(offsets):
        if not np.array_equal(shifted[:, j], np.roll(B[:, j], -s)):
            out.append(f"columna {j}: no es una permutacion ciclica exacta")
    for i in range(len(offsets)):
        for j in range(i + 1, len(offsets)):
            d = (offsets[j] - offsets[i]) % N
            if d == 0 or bad[d]:
                out.append(f"pareja ({i}, {j}): desfase relativo {d} en Bad")
    return out


def parallel_analysis(*_args, **_kwargs):
    pending("K_PA_MAX_T_V3")


def t3_v2(*_args, **_kwargs):
    pending("T3_V2")


def g2_replica_rank_check(counts: dict[str, int], expected: int, what: str) -> dict:
    """Control de nulos sobre una replica G2 (E02, A4-a: REPLICAS_G2 es ambito INFORMATIVO, nunca bloquea)."""
    return rank_check(counts, expected, what, "INFORMATIVA")


def official_g2_controls(publicacion: str, root=None, *_args, **_kwargs):
    """PRODUCCION: controles G2 sobre replicas con la configuracion de la ejecucion oficial. Su uso (k_PA, banda
    max-T, V3, T3/V2) no esta definido operativamente: bloqueado tras verificar la configuracion oficial."""
    official_config(publicacion, root)
    pending("K_PA_MAX_T_V3")
