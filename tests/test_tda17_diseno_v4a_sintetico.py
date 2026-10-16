"""Evidencia sintética versionada del diseño de TDA-17: adenda v4a (R² completo, etiquetas de
exactitud, rejilla y desfases de G2-3a, garantía por pares de G2-3b).

Sin datos de MNQ, sin holdout, sin lectura de ficheros: todo se genera con semillas fijas.
Las funciones de aquí son prototipos mínimos para verificar el diseño, no la implementación de TDA-17.
Origen: test_v4a_sintetico.py del diseño auditado (aserciones sin cambios).
"""
import numpy as np
import pytest

TAU = 1e-10            # tolerancia relativa de rango
EXACT_TOL = 1e-9       # residuo de identidad / 1-R2 "indistinguible de 1"
NEAR_TOL = 1e-4        # 1-R2 <= 1e-4 -> CASI_EXACTA_COLINEAL (descriptivo)


# ----------------------------------------------------------------- utilidades R2
def corr(M):
    return np.corrcoef(M, rowvar=False)


def r2_lstsq(M, j):
    Mc = M - M.mean(0)
    y, Xo = Mc[:, j], np.delete(Mc, j, axis=1)
    beta, *_ = np.linalg.lstsq(Xo, y, rcond=None)
    res = y - Xo @ beta
    return 1 - res @ res / (y @ y)


def r2_eig(R, j, tol=TAU):
    idx = [k for k in range(R.shape[0]) if k != j]
    lam, U = np.linalg.eigh(R[np.ix_(idx, idx)])
    keep = lam > tol * lam.max()
    coef = U[:, keep].T @ R[idx, j]
    return float(np.sum(coef**2 / lam[keep]))


def pivoted_cholesky_basis(R, tol=TAU):
    p = R.shape[0]
    d = np.diag(R).astype(float).copy()
    L = np.zeros_like(R, dtype=float)
    chosen = []
    for k in range(p):
        cand = np.where(np.isin(np.arange(p), chosen), -np.inf, d)
        i = int(np.argmax(cand))
        if d[i] <= tol * np.max(np.diag(R)):
            break
        chosen.append(i)
        L[i, k] = np.sqrt(d[i])
        for m in range(p):
            if m not in chosen:
                L[m, k] = (R[m, i] - L[m, :k] @ L[i, :k]) / L[i, k]
                d[m] -= L[m, k] ** 2
    return chosen


def r2_basis(R, j, tol=TAU):
    idx = [k for k in range(R.shape[0]) if k != j]
    basis = [idx[k] for k in pivoted_cholesky_basis(R[np.ix_(idx, idx)], tol)]
    r = R[basis, j]
    return float(r @ np.linalg.solve(R[np.ix_(basis, basis)], r))


def classify(name, r2, registry, residuals):
    """Clasificación de exactitud de la adenda (§3)."""
    reg = name in registry
    if reg:
        if residuals[registry[name]] > EXACT_TOL:
            return "ERROR_IDENTIDAD_REGISTRADA"
        return "EXACTA_ALGEBRAICA"
    if 1 - r2 <= EXACT_TOL:
        return "ALERTA_DEPENDENCIA_NO_REGISTRADA"
    if 1 - r2 <= NEAR_TOL:
        return "CASI_EXACTA_COLINEAL"
    return "SIN_ETIQUETA_DE_EXACTITUD"


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(20261016)
    n = 20_000
    X, Y, W, V, U = rng.standard_normal((5, n))
    A, B = rng.uniform(0, 0.5, (2, n))
    cols = {
        "X": X, "Y": Y, "Z": X + Y,                     # identidad homogénea registrada
        "A": A, "B": B, "C": 1.0 - A - B,               # identidad afín registrada
        "W": W, "N2": W + 1e-3 * rng.standard_normal(n),  # casi colineal (1-R2 ~ 1e-6)
        "U": U, "U2": U + 1e-7 * rng.standard_normal(n),  # colinealidad indistinguible (1-R2 ~ 1e-14)
        "V": V,
    }
    names = list(cols)
    return names, np.column_stack([cols[k] for k in names])


REGISTRY = {"X": "I1", "Y": "I1", "Z": "I1", "A": "I2", "B": "I2", "C": "I2"}


def identity_residuals(names, P):
    c = dict(zip(names, P.T))
    scale1 = np.max(np.abs(np.column_stack([c["X"], c["Y"], c["Z"]])))
    return {
        "I1": np.max(np.abs(c["Z"] - c["X"] - c["Y"])) / scale1,
        "I2": np.max(np.abs(c["A"] + c["B"] + c["C"] - 1.0)),
    }


# ----------------------------------------------------------------- tests R2 / D23
def clean(panel):
    names, P = panel
    keep = [k for k, nm in enumerate(names) if nm not in ("U", "U2")]
    return [names[k] for k in keep], P[:, keep]


def test_r2_full_panel_three_methods_agree_without_unregistered_collinearity(panel):
    names, P = clean(panel)
    R = corr(P)
    for j in range(len(names)):
        a, e, b = r2_lstsq(P, j), r2_eig(R, j), r2_basis(R, j)
        assert abs(a - e) < 1e-8 and abs(a - b) < 1e-8, names[j]


def test_unregistered_collinearity_triggers_cross_method_discrepancy(panel):
    """Con U2 = U + 1e-7·ruido, la tolerancia tau trunca esa dirección en la matriz de
    correlación pero no en MCO sobre filas: el R2 de OTRAS columnas (V) cambia ~1e-5.
    La verificación cruzada (>1e-8) debe detectarlo y el R2 afectado no se publica."""
    names, P = panel
    R = corr(P)
    disc = {nm: abs(r2_lstsq(P, j) - r2_basis(R, j)) for j, nm in enumerate(names)}
    flagged = {nm for nm, d in disc.items() if d > 1e-8}
    assert "V" in flagged
    assert not flagged & {"X", "Y", "Z", "A", "B", "C"}


def test_all_members_of_identities_have_r2_one(panel):
    names, P = panel
    R = corr(P)
    for nm in ("X", "Y", "Z", "A", "B", "C"):
        assert 1 - r2_basis(R, names.index(nm)) <= EXACT_TOL, nm


def test_v3_reduced_basis_method_is_wrong(panel):
    """Retirar Z y C (y asignarles 1) infraestima el R2 de X, Y, A, B."""
    names, P = panel
    keep = [k for k, nm in enumerate(names) if nm not in ("Z", "C")]
    RB = corr(P[:, keep])
    inv = np.linalg.inv(RB)
    for nm in ("X", "Y", "A", "B"):
        jb = keep.index(names.index(nm))
        assert 1 - 1 / inv[jb, jb] < 0.01, nm


def test_full_r2_equals_reduced_basis_for_columns_outside_identities(panel):
    names, P = panel
    R = corr(P)
    keep = [k for k, nm in enumerate(names) if nm not in ("Z", "C")]
    RB = corr(P[:, keep])
    for nm in ("W", "N2", "V"):
        jb = keep.index(names.index(nm))
        rb = RB.copy()
        idx = [k for k in range(len(keep)) if k != jb]
        lam, Uv = np.linalg.eigh(rb[np.ix_(idx, idx)])
        ok = lam > TAU * lam.max()
        coef = Uv[:, ok].T @ rb[idx, jb]
        r2_b = np.sum(coef**2 / lam[ok])
        assert abs(r2_basis(R, names.index(nm)) - r2_b) < 1e-8, nm


def test_exactness_classification(panel):
    names, P = panel
    R = corr(P)
    res = identity_residuals(names, P)
    lab = {nm: classify(nm, r2_basis(R, j), REGISTRY, res) for j, nm in enumerate(names)}
    for nm in ("X", "Y", "Z", "A", "B", "C"):
        assert lab[nm] == "EXACTA_ALGEBRAICA"
    # Casi colinealidad: R2 ~ 1 - 1e-6 nunca es EXACTA
    assert lab["W"] == lab["N2"] == "CASI_EXACTA_COLINEAL"
    # Colinealidad numéricamente indistinguible de 1, sin identidad registrada: alerta, no EXACTA
    assert lab["U"] == lab["U2"] == "ALERTA_DEPENDENCIA_NO_REGISTRADA"
    assert lab["V"] == "SIN_ETIQUETA_DE_EXACTITUD"


def test_registered_identity_with_broken_residual_is_an_error(panel):
    names, P = panel
    P2 = P.copy()
    P2[0, names.index("Z")] += 1e-3                     # corrompe la identidad en una fila
    res = identity_residuals(names, P2)
    assert classify("Z", 1.0, REGISTRY, res) == "ERROR_IDENTIDAD_REGISTRADA"


def test_null_eigenvalue_count_matches_registry_and_alerts_extra(panel):
    names, P = panel
    lam = np.linalg.eigvalsh(corr(P))
    n_null = int(np.sum(lam <= TAU * lam.max()))
    # 2 identidades registradas + 1 colinealidad no registrada (U/U2) -> 3: debe disparar alerta
    assert n_null == 3
    assert n_null != len(set(REGISTRY.values()))


def test_pca_full_vs_reduced(panel):
    names, P = panel
    drop = {"Z", "C", "U2"}
    lamP = np.linalg.eigvalsh(corr(P))
    lamB = np.linalg.eigvalsh(corr(P[:, [k for k, nm in enumerate(names) if nm not in drop]]))
    assert int(np.sum(lamP <= TAU * lamP.max())) == 3
    assert int(np.sum(lamB <= TAU * lamB.max())) == 0
    def pr(l):
        l = np.clip(l, 0, None)
        return l.sum() ** 2 / np.sum(l**2)
    assert pr(lamP) < pr(lamB)     # las copias exactas concentran varianza


# ----------------------------------------------------------------- rejilla G2-3a
def make_grid(D, M, seed=7):
    rng = np.random.default_rng(seed)
    present = np.ones((D, M), bool)
    present[0, :] = False
    present[2, M - 2:] = False
    present[5, 1] = False
    present[6, 3] = False
    return present, rng.standard_normal((D, M))


def test_row_shift_is_permutation_but_misaligns_minutes():
    present, val = make_grid(8, 6)
    _, minute = np.nonzero(present)
    for k in (5, 6, 11):
        assert np.mean(np.roll(minute, -k) == minute) < 1.0


def test_day_shift_on_grid_is_bijection_and_fully_aligned():
    present, val = make_grid(8, 6)
    for kd in (1, 3):
        vs, ps = np.roll(val, -kd, 0), np.roll(present, -kd, 0)
        assert sorted(vs[ps]) == sorted(val[present])
        # alineación: misma columna de minuto por construcción; pares sólo en la intersección
        assert (present & ps).sum() < present.sum()


def test_symmetrized_cross_corr_k_equals_D_minus_k():
    rng = np.random.default_rng(3)
    D, M = 30, 5
    present = rng.random((D, M)) > 0.1
    Xi, Xj = rng.standard_normal((2, D, M))

    def C(a, b, k):
        mk = present & np.roll(present, -k, 0)
        return np.corrcoef(a[mk], np.roll(b, -k, 0)[mk])[0, 1]

    for k in (3, 7, 12):
        s1 = 0.5 * (C(Xi, Xj, k) + C(Xj, Xi, k))
        s2 = 0.5 * (C(Xi, Xj, D - k) + C(Xj, Xi, D - k))
        assert abs(s1 - s2) < 1e-12


def offsets_3a(D, omega_sizes, n_cell, seed=1, kmin=20, kmax_count=100, min_eff=40, omega_frac=0.5):
    """Regla predeclarada de desfases G2-3a (adenda §1)."""
    elig = [k for k in range(kmin, D // 2 + 1) if omega_sizes[k] >= omega_frac * n_cell]
    if len(elig) < min_eff:
        return "NO_EVALUABLE", len(elig), []
    if len(elig) <= kmax_count:
        return "EVALUABLE", len(elig), elig
    chosen = np.random.default_rng(seed).choice(elig, kmax_count, replace=False)
    return "EVALUABLE", kmax_count, sorted(int(x) for x in chosen)


@pytest.mark.parametrize("D,expected_status,expected_n", [
    (254, "EVALUABLE", 100),   # año completo: 108 elegibles -> 100 sin reemplazo
    (128, "EVALUABLE", 45),    # año parcial: 64-19 = 45 elegibles
    (70, "NO_EVALUABLE", 16),  # demasiado corto: 35-19 = 16
])
def test_offsets_3a_rule(D, expected_status, expected_n):
    omega = {k: 1000 for k in range(D)}
    status, n_eff, ks = offsets_3a(D, omega, 1000)
    assert status == expected_status and n_eff == expected_n
    assert len(set(ks)) == len(ks)                       # sin reemplazo
    assert all(20 <= k <= D // 2 for k in ks)


def test_offsets_3a_omega_filter():
    D = 128
    omega = {k: (1000 if k % 2 else 400) for k in range(D)}   # desfases pares con Omega pequeño
    status, n_eff, ks = offsets_3a(D, omega, 1000)
    assert all(k % 2 for k in ks) and status == "NO_EVALUABLE" and n_eff == 22


def test_intersection_survival_motivates_pairwise_design():
    assert 0.985 ** 134 < 0.15 and 0.985 ** 2 > 0.97


# ----------------------------------------------------------------- G2-3b
def alignment_table(minute):
    """a(delta) = fracción de filas u con minute(u) == minute(u+delta), para todo delta (FFT)."""
    N = minute.size
    acc = np.zeros(N)
    for m in np.unique(minute):
        f = np.fft.rfft((minute == m).astype(float))
        acc += np.fft.irfft(f * np.conj(f), n=N)
    return np.rint(acc) / N


def alignment_brute(minute, delta):
    return float(np.mean(minute == np.roll(minute, -delta)))


def test_alignment_fft_matches_brute_force():
    rng = np.random.default_rng(5)
    lengths = rng.integers(8, 12, 40)
    minute = np.concatenate([np.arange(L) for L in lengths])
    a = alignment_table(minute)
    for d in (0, 1, 7, 9, 10, 11, 57, 200):
        assert abs(a[d] - alignment_brute(minute, d)) < 1e-12


def draw_offsets_3b(minute, p, seed=11, amax=0.05, max_tries=10_000):
    """Desfases G2-3b con garantía POR PARES: separación circular >= L_min y a(s_j - s_i) <= amax."""
    N = minute.size
    a = alignment_table(minute)
    L_min = N // (2 * p)
    rng = np.random.default_rng(seed)
    s = []
    for _ in range(p):
        for _try in range(max_tries):
            c = int(rng.integers(0, N))
            ok = all(min((c - x) % N, (x - c) % N) >= L_min and a[(c - x) % N] <= amax for x in s)
            if ok:
                s.append(c)
                break
        else:
            return None, L_min
    return s, L_min


def test_reference_only_check_misses_mutually_aligned_pair():
    minute = np.tile(np.arange(10), 200)                 # jornadas regulares de 10 filas
    s1, s2 = 3, 13                                       # cada uno desalineado frente a la referencia
    assert alignment_brute(minute, s1) == 0.0 and alignment_brute(minute, s2) == 0.0
    assert alignment_brute(minute, s2 - s1) == 1.0       # ... pero alineados entre sí
    a = alignment_table(minute)
    assert a[(s2 - s1) % minute.size] > 0.05             # la comprobación por pares lo detecta


def test_pairwise_procedure_guarantees_all_pairs():
    rng = np.random.default_rng(9)
    minute = np.concatenate([np.arange(L) for L in rng.integers(50, 61, 120)])
    p = 12
    s, L_min = draw_offsets_3b(minute, p)
    assert s is not None and len(s) == p
    N = minute.size
    for i in range(p):
        for j in range(i + 1, p):
            d = (s[j] - s[i]) % N
            assert min(d, N - d) >= L_min
            assert alignment_brute(minute, d) <= 0.05


def test_lmin_packing_bound():
    """Con L_min = N//(2p) caben p puntos; con L_min > N/p sería imposible."""
    N, p = 10_000, 123
    assert (N // (2 * p)) * p <= N
    assert (N // p + 1) * p > N
