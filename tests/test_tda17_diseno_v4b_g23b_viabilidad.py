"""Evidencia sintética versionada del diseño de TDA-17: viabilidad de G2-3b con p = 123 (salvaguarda v4b).

Sin datos de MNQ, sin holdout, sin lectura de ficheros. Origen: test_v4b_g23b_viabilidad.py del
diseño auditado (aserciones sin cambios).

Bad = {delta != 0 : a(delta) > AMAX  o  distancia circular(delta) < L_min}  (simétrico).
Buscamos S (|S| = p) con todas las diferencias fuera de Bad: un conjunto independiente
del grafo de Cayley Cay(Z_N, Bad). Para grafos de Cayley (vértice-transitivos)
alpha * omega <= N. Si encontramos un clique de tamaño K, alpha <= floor(N/K); si eso es < p,
la generación es IMPOSIBLE y no se intenta (certificado basado sólo en timestamps).
"""
import itertools

import numpy as np
import pytest

AMAX = 0.05
MAX_TRIES_PER_COLUMN = 10_000
TOP_STEPS = 50


def alignment_table(minute):
    N = minute.size
    acc = np.zeros(N)
    for m in np.unique(minute):
        f = np.fft.rfft((minute == m).astype(float))
        acc += np.fft.irfft(f * np.conj(f), n=N)
    return np.rint(acc) / N


def bad_mask(a, L_min):
    N = a.size
    d = np.arange(N)
    circ = np.minimum(d, N - d)
    bad = (a > AMAX) | (circ < L_min)
    bad[0] = False
    return bad


def clique_lower_bound(bad):
    """Clique del bloque contiguo {0, ..., run-1}: todas sus diferencias son < run y están en Bad
    (en particular, las de distancia circular < L_min). Las progresiones se acotan en best_progression."""
    N = bad.size
    best = 1
    # bloque contiguo: todas las diferencias < L_min
    run = 1
    while run < N and bad[run]:
        run += 1
    best = max(best, run)
    return best


def best_progression(a, bad):
    N = a.size
    half = np.arange(1, N // 2 + 1)
    cand = half[bad[half] & (a[half] > AMAX)]
    order = cand[np.argsort(-a[cand], kind="stable")][:TOP_STEPS]
    best = 1
    for s in order:
        k = 1
        while True:
            dk = (k * int(s)) % N
            if dk == 0 or not bad[dk]:
                break
            k += 1
        best = max(best, k)          # {0, s, ..., (k-1)s}: diferencias m*s, m=1..k-1, todas en Bad
    return best


def certify_infeasible(minute, p):
    """Diagnóstico previo (sólo timestamps): devuelve (imposible, omega_lb, alpha_ub, L_min)."""
    N = minute.size
    L_min = N // (2 * p)
    a = alignment_table(minute)
    bad = bad_mask(a, L_min)
    omega = max(clique_lower_bound(bad), best_progression(a, bad))
    alpha_ub = N // omega
    return alpha_ub < p, omega, alpha_ub, L_min, a, bad


def generate(bad, p, seed=11):
    N = bad.size
    rng = np.random.default_rng(seed)
    s, tries = [], 0
    for _ in range(p):
        for _t in range(MAX_TRIES_PER_COLUMN):
            tries += 1
            c = int(rng.integers(0, N))
            if all(not bad[(c - x) % N] and (c - x) % N != 0 for x in s):
                s.append(c)
                break
        else:
            return None, tries
    return s, tries


def diagnose(minute, p, seed=11):
    impossible, omega, alpha_ub, L_min, a, bad = certify_infeasible(minute, p)
    if impossible:
        return {"estado": "NO_EVALUABLE", "motivo": "CERTIFICADO_IMPOSIBILIDAD",
                "omega_lb": omega, "alpha_ub": alpha_ub, "intentos": 0}
    s, tries = generate(bad, p, seed)
    if s is None:
        return {"estado": "NO_EVALUABLE", "motivo": "GENERACION_AGOTADA", "intentos": tries,
                "omega_lb": omega, "alpha_ub": alpha_ub}
    return {"estado": "EVALUABLE", "offsets": s, "intentos": tries, "L_min": L_min,
            "omega_lb": omega, "alpha_ub": alpha_ub}


def regular(W, days):
    return np.tile(np.arange(W), days)


# --------------------------------------------------------------------------- tests
@pytest.mark.parametrize("W", [60, 120])
def test_regular_short_segments_are_certified_impossible_without_attempts(W):
    minute = regular(W, 300)
    out = diagnose(minute, 123)
    assert out["estado"] == "NO_EVALUABLE"
    assert out["motivo"] == "CERTIFICADO_IMPOSIBILIDAD"
    assert out["intentos"] == 0
    assert out["alpha_ub"] == W < 123          # cota de fases: como máximo W desfases compatibles


def test_regular_wide_segment_is_feasible_and_all_pairs_verified():
    W, p = 150, 123
    minute = regular(W, 300)
    out = diagnose(minute, p)
    assert out["estado"] == "EVALUABLE"
    s, N, L = out["offsets"], minute.size, out["L_min"]
    for i, j in itertools.combinations(range(p), 2):
        d = (s[j] - s[i]) % N
        assert min(d, N - d) >= L
        assert np.mean(minute == np.roll(minute, -d)) <= AMAX
    assert len({x % W for x in s}) == p        # fases horarias distintas, como exige la cota


def test_broad_irregular_case_is_feasible():
    rng = np.random.default_rng(2026)
    minute = np.concatenate([np.arange(L) for L in rng.integers(300, 401, 100)])
    out = diagnose(minute, 123)
    assert out["estado"] == "EVALUABLE"
    s, N = out["offsets"], minute.size
    a = alignment_table(minute)
    for i, j in itertools.combinations(range(123), 2):
        d = (s[j] - s[i]) % N
        assert a[d] <= AMAX and min(d, N - d) >= out["L_min"]


def test_certificate_is_sound_by_brute_force():
    """Instancia pequeña: el certificado dice alpha <= 5; la búsqueda exhaustiva lo confirma."""
    W, days, p = 5, 4, 6
    minute = regular(W, days)                   # N = 20
    impossible, omega, alpha_ub, L_min, a, bad = certify_infeasible(minute, p)
    assert impossible and alpha_ub == 5
    N = minute.size
    def independent(S):
        return all(not bad[(y - x) % N] for x, y in itertools.permutations(S, 2))
    assert not any(independent(S) for S in itertools.combinations(range(N), p))
    assert any(independent(S) for S in itertools.combinations(range(N), alpha_ub))


def test_bad_set_is_symmetric_as_required_by_alpha_omega_bound():
    """La cota alpha*omega <= N (mapa inyectivo (i, c) -> i + c) exige Bad simétrico."""
    minute = regular(6, 5)
    _, omega, _, L_min, a, bad = certify_infeasible(minute, 4)
    assert np.array_equal(bad, bad[(-np.arange(bad.size)) % bad.size])   # simetría de Bad
