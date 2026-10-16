"""Evidencia sintetica de la convencion de primera barra del generador G2-2 (g2.generativos.G2_2.generador).

Prototipo de referencia minimo sobre un esqueleto SINTETICO (contratos, anclas y volumenes inventados);
no es la implementacion de TDA-17. Solo se lee `configs/tda17.yaml` (df, sigma, semillas). Ningun dato
de MNQ, ningun holdout. Comprueba: primera barra, segunda barra, cambio de contrato, redondeo a tick,
invariantes OHLC y que ninguna innovacion se descarta ni se duplica.
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
SUBSTEPS = 30


@pytest.fixture(scope="module")
def g22() -> dict:
    return yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))["g2"]["generativos"]["G2_2"]


def q(p):
    return TICK * np.round(np.asarray(p) / TICK)


def generate(contract, anchors, volume, seed, df, sigma):
    """Filas ya en orden canonico (contract, timestamp). Devuelve OHLCV y trazas internas."""
    n = contract.size
    s0, s1, s2 = (np.random.Generator(np.random.PCG64(c)) for c in np.random.SeedSequence(seed).spawn(3))
    r = sigma * math.sqrt((df - 2) / df) * s0.standard_t(df, size=n)
    z = s1.standard_normal(size=(n, SUBSTEPS))
    idx = s2.integers(0, n, size=n)
    x = np.empty(n)
    x_prev_used = np.empty(n)
    O, H, L, C = (np.empty(n) for _ in range(4))
    k = np.arange(SUBSTEPS + 1) / SUBSTEPS
    for g in range(n):
        first = g == 0 or contract[g] != contract[g - 1]
        x_prev = math.log(anchors[contract[g]]) if first else x[g - 1]
        x_prev_used[g] = x_prev
        x[g] = x_prev + r[g]
        S = np.r_[0.0, np.cumsum(z[g])]
        b = x_prev + k * r[g] + sigma * (S - k * S[-1]) / math.sqrt(SUBSTEPS)
        C[g] = q(math.exp(x[g]))
        O[g] = anchors[contract[g]] if first else C[g - 1]
        H[g] = max(O[g], C[g], q(math.exp(b.max())))
        L[g] = min(O[g], C[g], q(math.exp(b.min())))
        if g in (0, 1):
            assert b[0] == x_prev and abs(b[-1] - x[g]) <= 1e-12 * abs(x[g])
    return {"O": O, "H": H, "L": L, "C": C, "V": volume[idx], "r": r, "x": x, "x_prev": x_prev_used, "idx": idx}


@pytest.fixture(scope="module")
def sample(g22):
    rng = np.random.default_rng(123)
    contract = np.repeat(np.array(["c0", "c1", "c2"]), [400, 1, 300])      # incluye un contrato de una sola fila
    anchors = {"c0": 20000.25, "c1": 21000.0, "c2": 19850.75}
    volume = rng.integers(0, 50, contract.size).astype(float)
    t = g22["t_student"]
    return contract, anchors, volume, generate(contract, anchors, volume, g22["semillas"][0], t["df"], g22["sigma_r_1m"])


def _firsts(contract):
    return np.r_[True, contract[1:] != contract[:-1]]


def test_yaml_declares_the_first_bar_convention(g22):
    gen = g22["generador"]
    for key in ("orden_filas", "innovaciones", "contabilidad", "tick_redondeo", "ancla", "camino", "precios", "puente",
                "maximo_minimo", "volumen"):
        assert key in gen
    assert "USA su innovacion" in gen["contabilidad"]
    assert "O_g = A_c en la primera fila" in gen["precios"]


def test_first_bar_of_each_contract(sample):
    contract, anchors, _, out = sample
    for g in np.flatnonzero(_firsts(contract)):
        a = anchors[contract[g]]
        assert out["O"][g] == a                                            # apertura = ancla
        assert out["x_prev"][g] == math.log(a)                             # el puente parte del ancla
        assert out["x"][g] == math.log(a) + out["r"][g]                    # la primera innovacion SI se usa
        assert out["C"][g] == q(math.exp(math.log(a) + out["r"][g]))


def test_second_bar_uses_latent_path_and_previous_close(sample):
    contract, _, _, out = sample
    g = 1
    assert contract[g] == contract[g - 1]
    assert out["O"][g] == out["C"][g - 1]                                  # apertura = cierre redondeado anterior
    assert out["x_prev"][g] == out["x"][g - 1]                             # camino latente, no ln(C redondeado)
    assert out["x"][g] == out["x"][g - 1] + out["r"][g]


def test_contract_change_restarts_at_the_new_anchor(sample):
    contract, anchors, _, out = sample
    starts = np.flatnonzero(_firsts(contract))
    assert list(starts) == [0, 400, 401]
    for g in starts[1:]:
        assert out["O"][g] == anchors[contract[g]] != out["C"][g - 1]
        assert out["x_prev"][g] != out["x"][g - 1]


def test_every_innovation_used_exactly_once(sample, g22):
    contract, anchors, volume, out = sample
    n = contract.size
    np.testing.assert_allclose(out["x"] - out["x_prev"], out["r"], rtol=0, atol=1e-12)   # fila g <-> r_g
    t = g22["t_student"]
    s0 = np.random.Generator(np.random.PCG64(np.random.SeedSequence(g22["semillas"][0]).spawn(3)[0]))
    ref = g22["sigma_r_1m"] * math.sqrt((t["df"] - 2) / t["df"]) * s0.standard_t(t["df"], size=n + 1)
    np.testing.assert_array_equal(out["r"], ref[:n])                       # ni salto inicial ni extraccion de mas
    assert out["idx"].size == n and out["V"].size == n


def test_prices_on_tick_grid_and_ohlc_invariants(sample):
    _, _, _, out = sample
    for k in ("O", "H", "L", "C"):
        v = out[k] / TICK
        assert np.all(v == np.round(v)), k
    assert np.all(out["L"] > 0)
    assert np.all(out["L"] <= np.minimum(out["O"], out["C"]))
    assert np.all(np.maximum(out["O"], out["C"]) <= out["H"])


def test_tick_rounding_convention():
    assert q(100.124) == 100.0 and q(100.126) == 100.25
    assert q(100.125) == 100.0 and q(100.375) == 100.5                     # mitad al par (numpy.round)
    assert q(19999.99) == 20000.0


def test_volume_is_resampled_by_index(sample):
    _, _, volume, out = sample
    np.testing.assert_array_equal(out["V"], volume[out["idx"]])
    assert out["idx"].min() >= 0 and out["idx"].max() < volume.size
