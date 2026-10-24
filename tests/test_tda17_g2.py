"""Pruebas SINTETICAS de la infraestructura G2 de TDA-17, fase 2 (src/ohlcv_dataroad/ingest/tda17_g2.py).

Sin datos de MNQ, sin holdout, sin data/raw y sin ejecutar ninguna simulacion sobre datos reales: esqueletos
sinteticos y semillas fijas. Las referencias (bucle fila a fila de G2_2, fuerza bruta de alineaciones y del
certificado de G2_3b) se escriben aqui de forma independiente.
"""
from __future__ import annotations

import copy
import itertools
import math

import numpy as np
import pandas as pd
import pytest

from ohlcv_dataroad.ingest import tda17_analysis as an
from ohlcv_dataroad.ingest import tda17_enmiendas as enm
from ohlcv_dataroad.ingest import tda17_enmiendas_e02 as m2
from ohlcv_dataroad.ingest import tda17_g2 as g2
from ohlcv_dataroad.ingest import tda17_indicators as ind
from tda17_synthetic import make_frame


@pytest.fixture(scope="module")
def cfg():
    return m2.effective_document_e02(borrador=True)


# =========================================================================== G2_2
def _reference_g2_2(contract, anchors, volume, seed, nu, sigma):
    """Bucle fila a fila del generador declarado (independiente de la version vectorizada)."""
    n = contract.size
    s0, s1, s2 = (np.random.Generator(np.random.PCG64(c)) for c in np.random.SeedSequence(seed).spawn(3))
    r = sigma * math.sqrt((nu - 2) / nu) * s0.standard_t(nu, size=n)
    z = s1.standard_normal(size=(n, 30))
    idx = s2.integers(0, n, size=n)
    x, O, H, L, C = (np.empty(n) for _ in range(5))
    k = np.arange(31) / 30
    for g in range(n):
        first = g == 0 or contract[g] != contract[g - 1]
        x_prev = math.log(anchors[contract[g]]) if first else x[g - 1]
        x[g] = x_prev + r[g]
        S = np.r_[0.0, np.cumsum(z[g])]
        b = x_prev + k * r[g] + sigma * (S - k * S[-1]) / math.sqrt(30)
        C[g] = 0.25 * np.round(np.exp(x[g]) / 0.25)
        O[g] = anchors[contract[g]] if first else C[g - 1]
        H[g] = max(O[g], C[g], 0.25 * np.round(np.exp(b.max()) / 0.25))
        L[g] = min(O[g], C[g], 0.25 * np.round(np.exp(b.min()) / 0.25))
    return {"r": r, "x": x, "O": O, "H": H, "L": L, "C": C, "V": volume[idx], "I": idx}


@pytest.fixture(scope="module")
def skeleton():
    rng = np.random.default_rng(123)
    contract = np.repeat(np.array(["c0", "c1", "c2"]), [400, 1, 300])        # incluye un contrato de una fila
    anchors = {"c0": 20000.25, "c1": 21000.0, "c2": 19850.75}
    ts = np.concatenate([pd.date_range("2021-01-04", periods=m, freq="min").to_numpy() for m in (400, 1, 300)])
    volume = rng.integers(0, 50, contract.size).astype(float)
    opens = np.array([anchors[c] for c in contract]) + 0.25 * rng.integers(0, 3, contract.size)
    firsts = np.r_[True, contract[1:] != contract[:-1]]
    opens[firsts] = [anchors[c] for c in contract[firsts]]
    skel = pd.DataFrame({"contract": contract, "timestamp": ts, "open": opens, "volume": volume})
    return skel, anchors


def test_g2_2_vectorized_generator_equals_row_by_row_reference(cfg, skeleton):
    skel, anchors = skeleton
    shuffled = skel.sample(frac=1.0, random_state=7)                        # el orden canonico se impone
    out = g2.g2_2_generate(shuffled, 301, cfg, check_rows=False, chunk=97)
    t = cfg["g2"]["generativos"]["G2_2"]
    ref = _reference_g2_2(skel["contract"].to_numpy(), anchors, skel["volume"].to_numpy(), 301,
                          t["t_student"]["df"], t["sigma_r_1m"])
    np.testing.assert_array_equal(out["contract"].to_numpy(), skel["contract"].to_numpy())
    for k_out, k_ref in (("r", "r"), ("x", "x"), ("I", "I"), ("volume", "V")):
        np.testing.assert_array_equal(out[k_out].to_numpy(), ref[k_ref], err_msg=k_out)
    for k_out, k_ref in (("open", "O"), ("high", "H"), ("low", "L"), ("close", "C")):
        np.testing.assert_array_equal(out[k_out].to_numpy(), ref[k_ref], err_msg=k_out)
    assert g2.v1_ohlc_check(out["open"], out["high"], out["low"], out["close"]) == []
    first = np.flatnonzero(np.r_[True, skel["contract"].to_numpy()[1:] != skel["contract"].to_numpy()[:-1]])
    assert [out["open"].iloc[i] for i in first] == [20000.25, 21000.0, 19850.75]   # ancla y primera innovacion
    with pytest.raises(ValueError, match="n_filas"):
        g2.g2_2_generate(skel, 301, cfg)                                    # produccion exige n_filas del protocolo


@pytest.mark.parametrize("seed", [301, 302])
def test_g2_2_declared_seeds_pass_verification_with_expected_values(cfg, seed):
    out = g2.g2_2_verify(seed, cfg)
    assert out["n"] == cfg["entradas"]["esperado"]["filas"]
    assert out["dentro_de_tolerancia"] and out["coincide_valores_esperados"]


def test_v1_identity_check_never_accepts_an_identity_without_evaluable_rows(cfg):
    res = {"L1": {"residuo_relativo": 1e-16, "filas": 10}, "L3": {"residuo_relativo": np.nan, "filas": 0},
           "L4": {"residuo_relativo": 2e-9, "filas": 10}}
    errs = g2.v1_identity_check(res, cfg)
    assert len(errs) == 2 and any("L3" in e for e in errs) and any("L4" in e for e in errs)


def test_v1_ohlc_check_detects_violations():
    o, h, l, c = (np.array([100.0, 100.25]) for _ in range(4))
    assert g2.v1_ohlc_check(o, h + 0.25, l - 0.25, c) == []
    assert g2.v1_ohlc_check(o + 0.1, h + 0.25, l - 0.25, c)                 # fuera de la rejilla
    assert g2.v1_ohlc_check(o, h - 0.5, l, c)                               # H < max(O, C)


# --------------------------------------------------------------------------- C2: integracion G2_2 con TDA-04
@pytest.fixture(scope="module")
def unordered():
    """Tres contratos en orden cronologico Z -> A -> M (lexicograficamente A < M < Z): el orden canonico
    (contract, timestamp) NO es el temporal. Con HI cortos y largos, tramo denso, FJ, FJL y dos rolls."""
    df = make_frame(contracts=(("MNQ_Z", 2), ("MNQ_A", 2), ("MNQ_M", 2)), bars_per_day=600)   # 1200 > K MACD 1067
    return df.drop(columns=[ind.SYNTHETIC_MARKER])


def test_canonical_order_differs_from_time_order(unordered):
    order = g2.canonical_order(unordered)
    assert not np.array_equal(order, np.arange(len(unordered)))
    assert list(pd.unique(unordered["contract"].to_numpy()[order])) == ["MNQ_A", "MNQ_M", "MNQ_Z"]
    f = g2.frontier_counts(unordered)
    assert f["ROLL"] == 2 and min(f["HI"], f["FJ"], f["FJL"], f["CONS"]) > 0


def test_g2_2_replica_restores_time_order_metadata_and_frontiers(cfg, unordered):
    rep_ = g2.g2_2_replica(unordered, 301, cfg, check_rows=False)
    ref = unordered.reset_index(drop=True)
    for col in g2.SKELETON_COLUMNS + ["source_file", "segment_id"]:
        assert rep_[col].astype(str).tolist() == ref[col].astype(str).tolist(), col
    np.testing.assert_array_equal(ind.pair_types(rep_), ind.pair_types(ref))
    assert g2.frontier_counts(rep_) == g2.frontier_counts(ref)
    assert g2.v1_ohlc_check(rep_["open"], rep_["high"], rep_["low"], rep_["close"]) == []
    cons = rep_["r_1m_valid"].to_numpy()                                    # TDA-04 recalculada sobre los precios G2_2
    c = rep_["close"].to_numpy()
    np.testing.assert_allclose(rep_["r_1m"].to_numpy()[cons], np.log(c[cons] / np.r_[np.nan, c[:-1]][cons]), rtol=1e-12)
    assert rep_["r_1m"].isna().to_numpy()[~cons].all()


def test_g2_2_replica_row_correspondence_against_the_row_by_row_reference(cfg, unordered):
    """Cada fila temporal recibe exactamente la fila canonica generada a partir de ella: se compara con el bucle
    de referencia aplicado al esqueleto ordenado canonicamente y devuelto con la permutacion inversa."""
    rep_ = g2.g2_2_replica(unordered, 302, cfg, check_rows=False)
    ref = unordered.reset_index(drop=True)
    order = g2.canonical_order(ref)
    sk = ref.iloc[order]
    contract = sk["contract"].astype(str).to_numpy()
    first = np.r_[True, contract[1:] != contract[:-1]]
    anchors = dict(zip(contract[first], sk["open"].to_numpy()[first]))
    t = cfg["g2"]["generativos"]["G2_2"]
    exp = _reference_g2_2(contract, anchors, sk["volume"].to_numpy(), 302, t["t_student"]["df"], t["sigma_r_1m"])
    for col, k in (("open", "O"), ("high", "H"), ("low", "L"), ("close", "C"), ("volume", "V")):
        back = np.empty(len(ref))
        back[order] = exp[k]
        np.testing.assert_array_equal(rep_[col].to_numpy(), back, err_msg=col)
    for c in ("MNQ_Z", "MNQ_A", "MNQ_M"):                                   # ancla: apertura de su primera fila temporal
        i = int(np.flatnonzero(ref["contract"].to_numpy() == c)[0])
        assert rep_["open"].iloc[i] == ref["open"].iloc[i]
    gen = g2.g2_2_generate(ref[["contract", "timestamp", "open", "volume"]], 302, cfg, check_rows=False)
    np.testing.assert_array_equal(gen["r"].to_numpy(), g2.g2_2_innovations(302, len(ref), cfg))   # innovaciones intactas


def test_g2_2_replica_satisfies_v1_identities(cfg, unordered):
    inputs = g2.g2_2_replica(unordered, 301, cfg, check_rows=False)
    inputs[ind.SYNTHETIC_MARKER] = True
    series, _ = ind.compute_catalog_synthetic(inputs, cfg)
    base = inputs.drop(columns=[ind.SYNTHETIC_MARKER])
    res = {**an.identity_residuals(ind._build_panel_p(series, base, cfg), base, cfg, "P"),
           **an.identity_residuals(an.build_panel_l(series, cfg), base, cfg, "L")}
    assert len(res) == 14 and g2.v1_identity_check(res, cfg) == []


@pytest.mark.parametrize("case", ["duplicada", "perdida", "sobrante", "contrato_desalineado", "timestamp_desalineado"])
def test_g2_2_restore_rejects_duplicated_lost_or_misaligned_rows(cfg, unordered, case):
    ref = unordered.reset_index(drop=True)
    gen = g2.g2_2_generate(ref[["contract", "timestamp", "open", "volume"]], 301, cfg, check_rows=False)
    g2.g2_2_restore(gen, ref)                                               # control: la replica integra
    bad = gen.copy()
    pos = bad["fila_esqueleto"].to_numpy().copy()
    if case == "duplicada":
        pos[1] = pos[0]
    elif case == "perdida":
        bad = bad.iloc[1:]
        pos = pos[1:]
    elif case == "sobrante":
        bad = pd.concat([bad, bad.iloc[:1]])
        pos = np.r_[pos, pos[0]]
    elif case == "contrato_desalineado":                                    # intercambio entre contratos distintos
        i, j = 0, len(pos) - 1
        pos[i], pos[j] = pos[j], pos[i]
    elif case == "timestamp_desalineado":                                   # intercambio dentro del mismo contrato
        pos[0], pos[1] = pos[1], pos[0]
    bad = bad.assign(fila_esqueleto=pos)
    match = {"duplicada": "duplicadas", "perdida": "perdidas", "sobrante": "perdidas o sobrantes",
             "contrato_desalineado": "desalineada", "timestamp_desalineado": "desalineada"}[case]
    with pytest.raises(ValueError, match=match):
        g2.g2_2_restore(bad, ref)


# =========================================================================== G2_1a / G2_1b
@pytest.fixture(scope="module")
def frame():
    return make_frame()


def _vectors(df):
    o, h, l, c = (np.round(df[k].to_numpy() / 0.25).astype(np.int64) for k in ("open", "high", "low", "close"))
    return np.column_stack([o - np.r_[0, c[:-1]], h - o, l - o, c - o, df["volume"].to_numpy()])


@pytest.mark.parametrize("by_minute", [False, True])
def test_g2_1_permutes_cons_vectors_within_contract_and_keeps_frontiers(frame, by_minute):
    gen = g2.g2_1_generate(frame, 101 if not by_minute else 201, by_minute)
    for col in g2.SKELETON_COLUMNS:
        assert gen[col].equals(frame[col]) or np.array_equal(gen[col].to_numpy(), frame[col].to_numpy(), equal_nan=True)
    assert g2.v1_ohlc_check(gen["open"], gen["high"], gen["low"], gen["close"]) == []
    first = frame["is_first_observation"].to_numpy() | frame["is_roll_boundary"].to_numpy()
    cons = frame["r_1m_valid"].to_numpy() & ~first
    v0, v1 = _vectors(frame), _vectors(gen)
    np.testing.assert_array_equal(v1[~cons & ~first], v0[~cons & ~first])  # fronteras en su sitio
    for k in ("open", "high", "low", "close", "volume"):
        np.testing.assert_array_equal(gen[k].to_numpy()[first], frame[k].to_numpy()[first])
    key = frame["contract"].astype(str).to_numpy()
    if by_minute:
        key = key + "|" + ind.minute_of_day_ny(frame["timestamp"]).astype(str)
    for g in np.unique(key[cons]):
        sel = cons & (key == g)
        a, b = v0[sel], v1[sel]
        assert np.array_equal(a[np.lexsort(a.T)], b[np.lexsort(b.T)]), g   # mismo multiconjunto por grupo
    assert not np.array_equal(v0[cons], v1[cons])                           # realmente permuta


def test_g2_1_is_reproducible_by_seed(frame):
    a, b, c = (g2.g2_1_generate(frame, s, False) for s in (101, 101, 102))
    assert a["close"].equals(b["close"]) and not a["close"].equals(c["close"])


def test_g2_1_replica_satisfies_v1_identities(cfg, frame):
    """V1 sobre una replica: variables de TDA-04 recalculadas con sus funciones y las 14 identidades lineales
    registradas (11 de P y 3 de L) con residuo <= 1e-9."""
    gen = g2.g2_1_generate(frame, 101, False)
    inputs = g2.rebuild_inputs(gen)
    inputs[ind.SYNTHETIC_MARKER] = True
    series, _ = ind.compute_catalog_synthetic(inputs, cfg)
    base = inputs.drop(columns=[ind.SYNTHETIC_MARKER])
    res_p = an.identity_residuals(ind._build_panel_p(series, base, cfg), base, cfg, "P")
    res_l = an.identity_residuals(an.build_panel_l(series, cfg), base, cfg, "L")
    assert len(res_p) == 11 and len(res_l) == 3
    assert g2.v1_identity_check({**res_p, **res_l}, cfg) == []
    assert all(v["filas"] > 0 for v in {**res_p, **res_l}.values())


def test_rebuild_inputs_rejects_a_changed_skeleton(frame):
    from ohlcv_dataroad.ingest.tda04_analysis_variables import RollConsistencyError
    gen = g2.g2_1_generate(frame, 101, False)
    bad_roll = gen.copy()
    bad_roll.loc[5, "is_roll_boundary"] = True
    with pytest.raises(RollConsistencyError):                               # lo detecta la propia TDA-04
        g2.rebuild_inputs(bad_roll)
    bad_delta = gen.copy()
    bad_delta.loc[5, "delta_minutes"] = 2.0                                 # TDA-04 lo recalcula: lo detecta G2
    with pytest.raises(ValueError, match="cambia el esqueleto: delta_minutes"):
        g2.rebuild_inputs(bad_delta)


# =========================================================================== G2_3a
@pytest.mark.parametrize("D,estado,n", [(254, "EVALUABLE", 100), (128, "EVALUABLE", 45), (70, "NO_EVALUABLE", 16)])
def test_g2_3a_offset_rule(cfg, D, estado, n):
    out = g2.g2_3a_offsets(np.ones((D, 5), bool), cfg)
    assert out["estado"] == estado and (len(out["desfases"]) if estado == "EVALUABLE" else out["elegibles"]) == n
    assert len(set(out["desfases"])) == len(out["desfases"]) and all(20 <= k <= D // 2 for k in out["desfases"])
    if D == 254:
        assert out["desfases"] == g2.g2_3a_offsets(np.ones((D, 5), bool), cfg)["desfases"]   # semilla 401


def test_g2_3a_omega_filter(cfg):
    present = np.zeros((128, 5), bool)
    present[::2] = True                                                     # desfases impares: Omega vacio
    out = g2.g2_3a_offsets(present, cfg)
    assert out["estado"] == "NO_EVALUABLE" and out["elegibles"] == 23 and all(k % 2 == 0 for k in out["omega"])


def test_g2_3a_grid_symmetry_and_bijection():
    rng = np.random.default_rng(3)
    D, M = 30, 5
    day = np.repeat(np.arange(D), M)
    minute = np.tile(np.arange(M) * 7, D)
    keep = rng.random(D * M) > 0.1
    present, grid, mins = g2.g2_3a_grid(day[keep], minute[keep], rng.standard_normal((keep.sum(), 2)))
    xi, xj = grid[..., 0], grid[..., 1]
    for k in (3, 7, 12):
        assert abs(g2.g2_3a_pair_corr(xi, xj, present, k) - g2.g2_3a_pair_corr(xi, xj, present, D - k)) < 1e-12
        assert g2.v1_prime_3a(present, grid, k) == []
    with pytest.raises(ValueError, match="duplicadas"):
        g2.g2_3a_grid(np.r_[0, 0], np.r_[5, 5], np.zeros(2))


# =========================================================================== G2_3b
def _brute_alignment(minute, delta):
    return float(np.mean(minute == np.roll(minute, -delta)))


def test_alignment_fft_matches_brute_force():
    rng = np.random.default_rng(5)
    minute = np.concatenate([np.arange(L) for L in rng.integers(8, 12, 40)])
    a = g2.alignment_table(minute)
    for d in (0, 1, 7, 9, 10, 11, 57, 200):
        assert abs(a[d] - _brute_alignment(minute, d)) < 1e-12


@pytest.mark.parametrize("W", [60, 120])
def test_short_regular_segments_are_certified_impossible_without_attempts(cfg, W):
    out = g2.g2_3b_viability(np.tile(np.arange(W), 300), 300, cfg)
    assert (out["estado"], out["motivo"], out["intentos"]) == ("NO_EVALUABLE", "CERTIFICADO_IMPOSIBILIDAD", 0)
    assert out["alpha_ub"] == W < 123
    assert g2.g2_3b_replicas(out, cfg)["replicas"] == []                    # sin ningun intento de generacion


def test_separation_rule_precedes_everything(cfg):
    out = g2.g2_3b_viability(np.tile(np.arange(300), 200), 200, cfg)        # 200 / 246 < 1 jornada
    assert (out["estado"], out["motivo"], out["intentos"]) == ("NO_EVALUABLE", "SEPARACION_INSUFICIENTE", 0)


def test_wide_segment_is_viable_and_replicas_satisfy_the_pairwise_guarantee(cfg):
    minute = np.tile(np.arange(150), 300)
    via = g2.g2_3b_viability(minute, 300, cfg)
    assert via["estado"] == "VIABLE"
    small = copy.deepcopy(cfg)
    small["g2"]["desplazamientos"]["G2_3b"]["replicas"] = 3                  # mecanica, no la ejecucion oficial
    reps = g2.g2_3b_replicas(via, small)
    assert reps["estado"] == "GENERADO" and len(reps["replicas"]) == 3 and reps["no_generadas"] == 0
    B = np.random.default_rng(0).standard_normal((minute.size, 123))
    for s in reps["replicas"]:
        assert len({x % 150 for x in s}) == 123                             # fases horarias distintas
        assert g2.v1_prime_3b(s, via["bad"], B, g2.shift_columns(B, s)) == []
        for i, j in itertools.combinations(range(0, 123, 11), 2):           # muestra de parejas por fuerza bruta
            d = (s[j] - s[i]) % minute.size
            assert _brute_alignment(minute, d) <= 0.05 and min(d, minute.size - d) >= via["L_min"]
    assert reps["replicas"][0] == g2.g2_3b_replicas(via, small)["replicas"][0]   # semilla 501


def test_replica_failure_fraction_makes_k_pa_not_evaluable(cfg):
    small = copy.deepcopy(cfg)
    small["g2"]["desplazamientos"]["G2_3b"].update(replicas=5, intentos_max_por_columna=50)
    bad = np.ones(1000, bool)
    bad[0] = False                                                          # ningun segundo desfase es compatible
    out = g2.g2_3b_replicas({"estado": "VIABLE", "bad": bad}, small)
    assert (out["estado"], out["motivo"], out["no_generadas"]) == ("NO_EVALUABLE", "REPLICAS_NO_GENERADAS", 5)


def test_certificate_is_sound_by_exhaustive_search():
    minute = np.tile(np.arange(5), 4)                                       # N = 20, p = 6
    N, p = minute.size, 6
    L_min = N // (2 * p)
    a = g2.alignment_table(minute)
    bad = g2.bad_mask(a, L_min, 0.05)
    omega = max(g2._clique_block(bad), g2._clique_progression(a, bad, 0.05, 50))
    assert N // omega == 5 < p
    independent = lambda S: all(not bad[(y - x) % N] for x, y in itertools.permutations(S, 2))   # noqa: E731
    assert not any(independent(S) for S in itertools.combinations(range(N), p))
    assert any(independent(S) for S in itertools.combinations(range(N), N // omega))
    assert np.array_equal(bad, bad[(-np.arange(N)) % N])                    # Bad simetrico


def test_v1_prime_3b_detects_a_non_cyclic_column_and_a_bad_pair():
    bad = np.zeros(10, bool)
    bad[[3, 7]] = True
    B = np.arange(30.0).reshape(10, 3)
    S = g2.shift_columns(B, [0, 3, 5])
    errs = g2.v1_prime_3b([0, 3, 5], bad, B, S)
    assert any("pareja (0, 1)" in e for e in errs)
    S2 = S.copy()
    S2[0, 2] += 1
    assert any("columna 2" in e for e in g2.v1_prime_3b([0, 1, 5], np.zeros(10, bool), B, S2))


# =========================================================================== pendientes
def test_undefined_g2_points_are_blocked():
    with pytest.raises(an.PendingDefinitionError, match="K_PA_MAX_T_V3"):
        g2.parallel_analysis()
    with pytest.raises(an.PendingDefinitionError, match="T3_V2"):
        g2.t3_v2()


def test_g2_module_reads_no_files():
    src = (enm.ROOT / "src" / "ohlcv_dataroad" / "ingest" / "tda17_g2.py").read_text(encoding="utf-8")
    for token in ("read_parquet", "read_csv", "open(", "data/raw", "to_parquet", "to_csv", "read_text", "read_bytes"):
        assert token not in src, token
