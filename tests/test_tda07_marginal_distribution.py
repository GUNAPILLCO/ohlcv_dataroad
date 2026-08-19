"""Tests de TDA-07 -- distribucion marginal y por segmento.

Igual que TDA-04/05/06: datos sinteticos minimos para aislar cada
propiedad (invariantes bloqueantes de alineacion de timestamps y de
coincidencia r_naive/r_1m, construccion del contrafactual de TH08,
aislamiento por causa, conservacion de etiquetas RETROSPECTIVO/
CONTRAFACTUAL, segmentacion leida sin alterar TDA-06, proteccion del
hold-out, momentos/cuantiles con resultado conocido a mano,
reproducibilidad del bootstrap/HAC). No se reproducen aqui los numeros
exactos del conjunto de investigacion real (esos viven en
``reports/mnq/TDA07_distribucion_marginal.md``).
"""
from __future__ import annotations

import datetime
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ohlcv_dataroad.config import load_config
from ohlcv_dataroad.ingest.holdout_guard import HoldoutIsolationError
from ohlcv_dataroad.ingest.tda07_marginal_distribution import (
    CONTRAFACTUAL_LABEL,
    NAIVE_MATCH_TOLERANCE,
    NaiveReturnContradictionError,
    RTildeInvariantError,
    TimestampAlignmentError,
    analyze_group,
    assign_segment_label,
    build_contrafactual_table,
    build_naive_return,
    build_r1m_population,
    build_r_tilde_population,
    compute_hac_block_ids,
    compute_moments_quantiles,
    compute_tick_return_repr,
    day_block_bootstrap,
    hac_bandwidth,
    hac_mean_se,
    load_segmentation_cutoffs,
    qq_points,
    run_tda07_analysis,
    th08_global_comparison,
    th08_secondary_by_cause,
    verify_naive_matches_valid,
    verify_r_tilde_invariants,
    verify_timestamp_alignment,
)

TICK = 0.25


# ---------------------------------------------------------------------------
# Fixtures sinteticas
# ---------------------------------------------------------------------------

def _variables_and_validity(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    variables = pd.DataFrame(
        [
            {
                "timestamp": r["timestamp"], "source_file": r.get("source_file", "f1"),
                "contract": r.get("contract", "H24"), "trading_date": r["trading_date"],
                "segment_id": r.get("segment_id", 1),
                "open": r.get("open", r["close"]), "high": r.get("high", r["close"] + 0.5),
                "low": r.get("low", r["close"] - 0.5), "close": r["close"], "volume": r.get("volume", 10),
                "r_1m": r.get("r_1m", np.nan),
            }
            for r in rows
        ]
    )
    validity = pd.DataFrame(
        [{"timestamp": r["timestamp"], "r_1m_valid": r["r_1m_valid"], "invalid_reason": r["invalid_reason"]} for r in rows]
    )
    return variables, validity


def _synthetic_valid_sequence(closes: list[float], start_ts: pd.Timestamp, trading_date: datetime.date) -> list[dict]:
    """Secuencia de barras CONSECUTIVAS (1 minuto), todas VALID salvo la primera (FIRST_OBSERVATION)."""
    ts = pd.date_range(start_ts, periods=len(closes), freq="1min")
    rows = []
    for i, c in enumerate(closes):
        if i == 0:
            rows.append({"timestamp": ts[i], "close": c, "trading_date": trading_date, "r_1m_valid": False, "invalid_reason": "FIRST_OBSERVATION"})
        else:
            r_1m = float(np.log(closes[i] / closes[i - 1]))
            rows.append({"timestamp": ts[i], "close": c, "trading_date": trading_date, "r_1m_valid": True, "invalid_reason": "VALID", "r_1m": r_1m})
    return rows


def _synthetic_multi_day_rows(n_days: int, start: datetime.date, bars_per_day: int = 6, seed: int = 0) -> list[dict]:
    """Varios dias habiles, cada uno con ``bars_per_day`` barras consecutivas (todas VALID salvo la 1a global)."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    d = start
    day_i = 0
    added = 0
    price = 15000.0
    first_row = True
    while added < n_days:
        date = start + datetime.timedelta(days=day_i)
        day_i += 1
        if date.weekday() >= 5:
            continue
        base_ts = pd.Timestamp(date) + pd.Timedelta(hours=9, minutes=30)
        for b in range(bars_per_day):
            ts = base_ts + pd.Timedelta(minutes=b * 90)  # separa barras -> minutos distintos, dias distintos
            price = price * (1.0 + rng.normal(0.0, 0.0005))
            price = round(price / TICK) * TICK
            if first_row:
                rows.append({"timestamp": ts, "close": price, "trading_date": date, "r_1m_valid": False, "invalid_reason": "FIRST_OBSERVATION"})
                first_row = False
            else:
                prev_close = rows[-1]["close"]
                r_1m = float(np.log(price / prev_close))
                rows.append({"timestamp": ts, "close": price, "trading_date": date, "r_1m_valid": True, "invalid_reason": "VALID", "r_1m": r_1m})
        added += 1
    return rows


# ---------------------------------------------------------------------------
# Invariante #1 -- alineacion de timestamps
# ---------------------------------------------------------------------------

def test_verify_timestamp_alignment_passes_when_identical():
    rows = _synthetic_valid_sequence([100.0, 100.25, 100.0], pd.Timestamp("2024-01-08 10:00:00"), datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    verify_timestamp_alignment(variables, validity)  # no debe lanzar


def test_verify_timestamp_alignment_raises_on_different_length():
    rows = _synthetic_valid_sequence([100.0, 100.25, 100.0], pd.Timestamp("2024-01-08 10:00:00"), datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    validity_short = validity.iloc[:-1]
    with pytest.raises(TimestampAlignmentError, match="filas"):
        verify_timestamp_alignment(variables, validity_short)


def test_verify_timestamp_alignment_raises_on_mismatched_timestamps():
    rows = _synthetic_valid_sequence([100.0, 100.25, 100.0], pd.Timestamp("2024-01-08 10:00:00"), datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    validity = validity.copy()
    validity.loc[1, "timestamp"] = validity.loc[1, "timestamp"] + pd.Timedelta(seconds=1)
    with pytest.raises(TimestampAlignmentError, match="no coinciden"):
        verify_timestamp_alignment(variables, validity)


# ---------------------------------------------------------------------------
# Construccion del contrafactual: r_naive_1m
# ---------------------------------------------------------------------------

def test_build_naive_return_matches_formula_exactly():
    rows = _synthetic_valid_sequence([100.0, 100.25, 100.75, 100.50], pd.Timestamp("2024-01-08 10:00:00"), datetime.date(2024, 1, 8))
    variables, _ = _variables_and_validity(rows)
    naive = build_naive_return(variables)
    expected = [np.nan, np.log(100.25 / 100.0), np.log(100.75 / 100.25), np.log(100.50 / 100.75)]
    np.testing.assert_allclose(naive["r_naive_1m"].to_numpy(), expected, equal_nan=True)


def test_build_naive_return_is_nan_only_at_first_row_of_full_series():
    """r_naive_1m NO debe ser NaN en una frontera de trading_date/roll -- solo en la PRIMERA fila global."""
    rows = _synthetic_valid_sequence([100.0, 100.25], pd.Timestamp("2024-01-08 10:00:00"), datetime.date(2024, 1, 8))
    # Segunda "jornada" -- fila inmediatamente siguiente en la tabla, pero
    # trading_date distinta (r_1m de TDA-04 seria NaN aqui; r_naive_1m NO).
    rows.append({"timestamp": pd.Timestamp("2024-01-09 10:00:00"), "close": 101.00, "trading_date": datetime.date(2024, 1, 9), "r_1m_valid": False, "invalid_reason": "TRADING_DATE_BOUNDARY"})
    variables, _ = _variables_and_validity(rows)
    naive = build_naive_return(variables)
    assert pd.isna(naive["r_naive_1m"].iloc[0])
    assert not pd.isna(naive["r_naive_1m"].iloc[1])
    assert not pd.isna(naive["r_naive_1m"].iloc[2])  # cruza la frontera de trading_date -- eso es EXACTAMENTE lo contrafactual
    assert naive["r_naive_1m"].iloc[2] == pytest.approx(np.log(101.00 / 100.25))


def test_build_naive_return_label_is_contrafactual():
    rows = _synthetic_valid_sequence([100.0, 100.25], pd.Timestamp("2024-01-08 10:00:00"), datetime.date(2024, 1, 8))
    variables, _ = _variables_and_validity(rows)
    naive = build_naive_return(variables)
    assert (naive["label"] == CONTRAFACTUAL_LABEL).all()
    assert CONTRAFACTUAL_LABEL == "CONTRAFACTUAL_VIOLA_NO_CRUCE"


# ---------------------------------------------------------------------------
# Invariante #2 -- r_naive_1m == r_1m en filas VALID
# ---------------------------------------------------------------------------

def test_verify_naive_matches_valid_passes_on_consistent_data():
    rows = _synthetic_valid_sequence([100.0, 100.25, 100.75], pd.Timestamp("2024-01-08 10:00:00"), datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    naive = build_naive_return(variables)
    verify_naive_matches_valid(
        naive["r_naive_1m"].to_numpy(), variables["r_1m"].to_numpy(), validity["invalid_reason"].to_numpy()
    )  # no debe lanzar


def test_verify_naive_matches_valid_raises_on_contradiction():
    rows = _synthetic_valid_sequence([100.0, 100.25, 100.75], pd.Timestamp("2024-01-08 10:00:00"), datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    naive = build_naive_return(variables)
    corrupted_r1m = variables["r_1m"].to_numpy().copy()
    corrupted_r1m[2] = corrupted_r1m[2] + 1.0  # corrompe una fila VALID
    with pytest.raises(NaiveReturnContradictionError, match="invalid_reason=VALID"):
        verify_naive_matches_valid(naive["r_naive_1m"].to_numpy(), corrupted_r1m, validity["invalid_reason"].to_numpy())


def test_verify_naive_matches_valid_respects_tolerance():
    rows = _synthetic_valid_sequence([100.0, 100.25, 100.75], pd.Timestamp("2024-01-08 10:00:00"), datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    naive = build_naive_return(variables)
    tiny_noise = variables["r_1m"].to_numpy().copy()
    tiny_noise[2] = tiny_noise[2] + NAIVE_MATCH_TOLERANCE / 10.0
    verify_naive_matches_valid(naive["r_naive_1m"].to_numpy(), tiny_noise, validity["invalid_reason"].to_numpy())  # no debe lanzar


# ---------------------------------------------------------------------------
# build_contrafactual_table -- ensamblaje completo + TH08
# ---------------------------------------------------------------------------

def test_build_contrafactual_table_runs_invariants_and_assembles_columns():
    rows = _synthetic_multi_day_rows(n_days=5, start=datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    out = build_contrafactual_table(variables, validity)
    expected_cols = {"timestamp", "trading_date", "r_1m", "r_naive_1m", "invalid_reason", "r_1m_valid", "label"}
    assert expected_cols.issubset(out.columns)
    assert (out["label"] == CONTRAFACTUAL_LABEL).all()


def test_build_contrafactual_table_propagates_contradiction():
    rows = _synthetic_valid_sequence([100.0, 100.25, 100.75], pd.Timestamp("2024-01-08 10:00:00"), datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    variables = variables.copy()
    variables.loc[2, "r_1m"] = variables.loc[2, "r_1m"] + 1.0
    with pytest.raises(NaiveReturnContradictionError):
        build_contrafactual_table(variables, validity)


def test_th08_global_comparison_first_observation_excluded_from_b():
    rows = _synthetic_valid_sequence([100.0, 100.25, 100.75], pd.Timestamp("2024-01-08 10:00:00"), datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    cf = build_contrafactual_table(variables, validity)
    comparison = th08_global_comparison(cf)
    a_row = comparison.set_index("population").loc["A_r_1m_valid"]
    b_row = comparison.set_index("population").loc["B_r_naive_global"]
    assert int(a_row["n"]) == 2  # 2 filas VALID (3 barras - 1 FIRST_OBSERVATION)
    assert int(b_row["n"]) == 2  # r_naive_1m NaN solo en la primera fila -- misma n aqui porque no hay otras fronteras


def test_th08_global_comparison_b_includes_boundary_crossings_a_excludes():
    """Con una frontera TRADING_DATE_BOUNDARY, B debe tener MAS observaciones que A."""
    rows = _synthetic_valid_sequence([100.0, 100.25], pd.Timestamp("2024-01-08 10:00:00"), datetime.date(2024, 1, 8))
    rows.append({"timestamp": pd.Timestamp("2024-01-09 10:00:00"), "close": 101.00, "trading_date": datetime.date(2024, 1, 9), "r_1m_valid": False, "invalid_reason": "TRADING_DATE_BOUNDARY"})
    variables, validity = _variables_and_validity(rows)
    cf = build_contrafactual_table(variables, validity)
    comparison = th08_global_comparison(cf).set_index("population")
    assert int(comparison.loc["B_r_naive_global", "n"]) == 2  # cruza la frontera: SI cuenta en B
    assert int(comparison.loc["A_r_1m_valid", "n"]) == 1  # NO cuenta en A (regla de no-cruce de TDA-04)


def test_th08_secondary_by_cause_isolates_each_reason():
    rows = _synthetic_valid_sequence([100.0, 100.25], pd.Timestamp("2024-01-08 10:00:00"), datetime.date(2024, 1, 8))
    rows.append({"timestamp": pd.Timestamp("2024-01-08 10:02:00"), "close": 105.00, "trading_date": datetime.date(2024, 1, 8), "r_1m_valid": False, "invalid_reason": "ROLL_BOUNDARY"})
    rows.append({"timestamp": pd.Timestamp("2024-01-09 10:00:00"), "close": 101.00, "trading_date": datetime.date(2024, 1, 9), "r_1m_valid": False, "invalid_reason": "TRADING_DATE_BOUNDARY"})
    rows.append({"timestamp": pd.Timestamp("2024-01-09 10:05:00"), "close": 101.50, "trading_date": datetime.date(2024, 1, 9), "r_1m_valid": False, "invalid_reason": "NON_CONSECUTIVE_MINUTE"})
    variables, validity = _variables_and_validity(rows)
    cf = build_contrafactual_table(variables, validity)
    by_cause = th08_secondary_by_cause(cf).set_index("invalid_reason")
    assert set(by_cause.index) == {"ROLL_BOUNDARY", "TRADING_DATE_BOUNDARY", "NON_CONSECUTIVE_MINUTE"}
    assert int(by_cause.loc["ROLL_BOUNDARY", "n"]) == 1
    assert int(by_cause.loc["TRADING_DATE_BOUNDARY", "n"]) == 1
    assert int(by_cause.loc["NON_CONSECUTIVE_MINUTE", "n"]) == 1
    expected_roll = float(np.log(105.00 / 100.25))
    assert by_cause.loc["ROLL_BOUNDARY", "mean"] == pytest.approx(expected_roll)


# ---------------------------------------------------------------------------
# Momentos y cuantiles -- resultado conocido a mano
# ---------------------------------------------------------------------------

def test_moments_quantiles_symmetric_distribution_known_values():
    x = np.tile(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), 500)
    out = compute_moments_quantiles(x)
    assert out["n"] == 2500
    assert out["mean"] == pytest.approx(0.0, abs=1e-10)
    assert out["skewness"] == pytest.approx(0.0, abs=1e-10)
    # m2 = mean(e^2) = (4+1+0+1+4)/5 = 2 ; m4 = mean(e^4) = (16+1+0+1+16)/5 = 6.8
    # kurtosis_excess = m4/m2^2 - 3 = 6.8/4 - 3 = -1.3
    assert out["kurtosis_excess"] == pytest.approx(-1.3, abs=1e-10)
    assert out["q0.5"] == pytest.approx(0.0)


def test_moments_quantiles_trimming_removes_dominant_outlier_effect():
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 1.0, size=5000)
    x = base.copy()
    # 1 outlier por cada cola (0.02% cada uno) -- bien por debajo del 0.05%
    # recortado por lado, para que el recorte los excluya con certeza (un
    # outlier justo EN el borde del percentil de corte no se removeria,
    # ver nota en compute_moments_quantiles: el corte es inclusivo).
    x[0] = 500.0
    x[1] = -500.0
    out = compute_moments_quantiles(x)
    assert out["kurtosis_excess"] > 50  # dominado por los outliers
    assert abs(out["kurtosis_excess_trimmed"]) < 5  # el recorte del 0.1% los retira -- curtosis vuelve a un rango normal


def test_moments_quantiles_empty_array_returns_nan_not_exception():
    out = compute_moments_quantiles(np.array([]))
    assert out["n"] == 0
    assert np.isnan(out["mean"])


def test_moments_quantiles_small_sample_kurtosis_trimmed_is_nan():
    out = compute_moments_quantiles(np.array([1.0, 2.0, 3.0]))
    assert np.isnan(out["kurtosis_excess_trimmed"])


# ---------------------------------------------------------------------------
# HAC (Newey-West)
# ---------------------------------------------------------------------------

def test_hac_bandwidth_matches_closed_formula():
    for T in (10, 100, 1000, 10000, 1_900_000):
        expected = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
        assert hac_bandwidth(T) == expected
    assert hac_bandwidth(100) == 4  # ejemplo citado en docs/tsay/Tsay_Cap2 para T=100


def test_hac_mean_se_is_deterministic_and_positive_for_iid_noise():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=2000)
    block_ids = np.zeros(2000, dtype=int)  # una unica jornada continua -- sin fronteras
    se1, l1 = hac_mean_se(x, block_ids)
    se2, l2 = hac_mean_se(x, block_ids)
    assert se1 == se2 and l1 == l2
    assert se1 > 0
    assert l1 == hac_bandwidth(2000)


def test_hac_mean_se_larger_than_naive_se_under_positive_autocorrelation():
    """Con dependencia POSITIVA fuerte DENTRO de un unico bloque continuo, el e.e. HAC debe superar al e.e. iid ingenuo."""
    rng = np.random.default_rng(0)
    n = 3000
    e = rng.normal(0.0, 1.0, size=n)
    x = np.convolve(e, np.ones(20) / 20, mode="same")  # fuerte dependencia de corto plazo
    block_ids = np.zeros(n, dtype=int)
    hac_se, _ = hac_mean_se(x, block_ids)
    naive_se = x.std(ddof=1) / np.sqrt(n)
    assert hac_se > naive_se


def test_hac_mean_se_matches_naive_formula_when_series_is_fully_continuous():
    """Sin ninguna frontera (un solo bloque cubriendo toda la serie), el HAC corregido debe coincidir EXACTAMENTE con el sandwich de Newey-West normalizado por T (Tsay Ec. 2.50, reimplementado aqui literalmente para el contraste) -- y quedar MUY CERCA (no exacto) de la convencion alternativa que normaliza por T-j, la diferencia esperada entre ambas normalizaciones para T,l de este orden."""
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, size=1500)
    block_ids = np.zeros(len(x), dtype=int)
    se_new, l_new = hac_mean_se(x, block_ids)

    T = x.size
    l_old = hac_bandwidth(T)
    xbar = x.mean()
    e = x - xbar
    gamma0 = np.mean(e * e)

    # Referencia CORRECTA (normalizada por T -- la que reproduce el
    # sandwich de Tsay Ec. 2.50 para un regresor constante, ver docstring
    # de hac_mean_se).
    s_T = gamma0
    # Referencia ALTERNATIVA (normalizada por T-j -- la convencion que
    # usaba la implementacion pre-correccion de continuidad, valida SOLO
    # quando no hay bloques, y solo aproximadamente equivalente a la de T
    # cuando l << T).
    s_Tminusj = gamma0
    for j in range(1, l_old + 1):
        sum_prod = np.sum(e[j:] * e[:-j])
        w_j = 1.0 - j / (l_old + 1.0)
        s_T += 2.0 * w_j * (sum_prod / T)
        s_Tminusj += 2.0 * w_j * (sum_prod / (T - j))
    se_T = np.sqrt(max(s_T, 0.0) / T)
    se_Tminusj = np.sqrt(max(s_Tminusj, 0.0) / T)

    assert l_new == l_old
    assert se_new == pytest.approx(se_T, rel=1e-10)  # coincide EXACTAMENTE con la normalizacion por T
    assert se_new == pytest.approx(se_Tminusj, rel=1e-2)  # y queda MUY cerca de la normalizacion por T-j (l << T)


# ---------------------------------------------------------------------------
# compute_hac_block_ids -- CORRECCION PUNTUAL: HAC no debe cruzar discontinuidades
# ---------------------------------------------------------------------------

def test_compute_hac_block_ids_single_continuous_run_is_one_block():
    ts = pd.date_range("2024-01-08 09:30:00", periods=10, freq="1min")
    trading_date = np.full(10, datetime.date(2024, 1, 8), dtype=object)
    block_ids = compute_hac_block_ids(pd.Series(ts), pd.Series(trading_date))
    assert (block_ids == 0).all()


def test_compute_hac_block_ids_breaks_on_a_gap_left_by_an_invalidated_row():
    """Dos bloques validos separados por un hueco (una fila NON_CONSECUTIVE_MINUTE, ya excluida de la poblacion) nunca deben contribuir a una autocovarianza cruzada."""
    ts = pd.to_datetime([
        "2024-01-08 09:30:00", "2024-01-08 09:31:00", "2024-01-08 09:32:00",  # bloque 1
        "2024-01-08 11:00:00", "2024-01-08 11:01:00", "2024-01-08 11:02:00",  # bloque 2 (hueco de ~1h28m -- la fila intermedia real, invalida, no esta en la poblacion)
    ])
    trading_date = np.full(6, datetime.date(2024, 1, 8), dtype=object)
    block_ids = compute_hac_block_ids(pd.Series(ts), pd.Series(trading_date))
    np.testing.assert_array_equal(block_ids, [0, 0, 0, 1, 1, 1])


def test_compute_hac_block_ids_breaks_on_trading_date_change_even_when_delta_is_exactly_60s():
    """Dos jornadas distintas NUNCA se consideran consecutivas -- ni siquiera si el delta fuera exactamente 60s."""
    ts = pd.to_datetime(["2024-01-08 23:59:00", "2024-01-09 00:00:00"])  # exactamente 60s de diferencia
    trading_date = np.array([datetime.date(2024, 1, 8), datetime.date(2024, 1, 9)], dtype=object)
    block_ids = compute_hac_block_ids(pd.Series(ts), pd.Series(trading_date))
    np.testing.assert_array_equal(block_ids, [0, 1])


def test_compute_hac_block_ids_breaks_across_what_would_be_a_roll_boundary():
    """Una frontera de ROLL_BOUNDARY (la fila del roll, invalida, ya excluida de la poblacion) nunca se cruza.

    TDA-04 (informe, S5) certifico que TODO roll coincide con un cambio de
    trading_date -- por eso basta con la condicion de trading_date para
    excluirlo tambien aqui, sin necesitar una bandera de roll separada.
    """
    ts = pd.to_datetime(["2024-03-14 16:00:00", "2024-03-15 18:01:00"])
    trading_date = np.array([datetime.date(2024, 3, 14), datetime.date(2024, 3, 15)], dtype=object)
    block_ids = compute_hac_block_ids(pd.Series(ts), pd.Series(trading_date))
    np.testing.assert_array_equal(block_ids, [0, 1])


def test_compute_hac_block_ids_segment_population_does_not_link_same_time_of_day_across_days():
    """Analisis POR SEGMENTO: el mismo tramo horario de dos dias distintos NO debe volverse vecino artificial."""
    ts = pd.to_datetime([
        "2024-01-08 10:00:00", "2024-01-08 10:01:00",  # dia 1, mismo segmento
        "2024-01-09 10:00:00", "2024-01-09 10:01:00",  # dia 2, mismo segmento -- ~24h despues
    ])
    trading_date = np.array([datetime.date(2024, 1, 8)] * 2 + [datetime.date(2024, 1, 9)] * 2, dtype=object)
    block_ids = compute_hac_block_ids(pd.Series(ts), pd.Series(trading_date))
    np.testing.assert_array_equal(block_ids, [0, 0, 1, 1])


def test_hac_mean_se_excludes_cross_block_pairs_reproducing_the_original_bug():
    """Test que FALLA con la implementacion original (que trataba el array compactado como un unico bloque) y PASA con la corregida."""
    x = np.array([1.0, 2.0, 3.0, 100.0, 101.0, 102.0])  # dos bloques con niveles MUY distintos
    block_ids_correct = np.array([0, 0, 0, 1, 1, 1])
    se_correct, _ = hac_mean_se(x, block_ids_correct)

    # Comportamiento del BUG original: compactar y tratar todo como un solo bloque continuo.
    naive_block_ids = np.zeros(len(x), dtype=int)
    se_bugged, _ = hac_mean_se(x, naive_block_ids)

    assert se_correct != pytest.approx(se_bugged)


def test_hac_mean_se_normalizes_by_full_sample_not_by_surviving_pair_count():
    """CORRECCION MATEMATICA (normalizacion): gamma_j debe normalizarse por T (la muestra COMPLETA), no por el numero de pares que sobreviven al filtro de bloque -- los pares cross-block deben aportar CERO a la suma, no hacer que los pares restantes reciban MAS peso.

    Tres bloques CORTOS con niveles muy distintos (1-3, 10-12, 100-102):
    la suma de covarianzas permitidas (solo pares dentro del mismo
    bloque) se conoce de forma independiente en este test; se comprueba
    tanto que los pares cross-block se excluyen de la suma como que la
    normalizacion final es por T, no por el numero de pares sobrevivientes.
    """
    x = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 100.0, 101.0, 102.0])
    block_ids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    T = x.size
    l = hac_bandwidth(T)
    assert l == 2  # verificado para T=9 -- fija el numero de rezagos que siguen en este test

    xbar = x.mean()
    e = x - xbar
    gamma0 = np.mean(e * e)

    # Referencia INDEPENDIENTE, correcta: SUMA (no promedio) de los
    # productos de los pares SAME-BLOCK -- los pares cross-block quedan
    # fuera de esta suma por construccion (aportan literalmente cero) --
    # dividida por T (la muestra COMPLETA).
    s_correct = gamma0
    # Referencia INCORRECTA (el bug de normalizacion que este test debe
    # detectar): la MISMA suma de pares same-block, mal dividida por el
    # numero de pares que sobrevivieron (same_block.sum()) en vez de T.
    s_wrong = gamma0
    for j in range(1, l + 1):
        same_block = block_ids[j:] == block_ids[:-j]
        sum_prod = np.sum((e[j:] * e[:-j])[same_block])
        n_surviving = same_block.sum()
        w_j = 1.0 - j / (l + 1.0)
        s_correct += 2.0 * w_j * (sum_prod / T)
        s_wrong += 2.0 * w_j * (sum_prod / n_surviving)
    se_correct_expected = np.sqrt(max(s_correct, 0.0) / T)
    se_wrong_expected = np.sqrt(max(s_wrong, 0.0) / T)

    se_actual, l_actual = hac_mean_se(x, block_ids)

    assert l_actual == l
    assert se_actual == pytest.approx(se_correct_expected, rel=1e-10)  # normalizacion CORRECTA (por T)
    assert se_actual != pytest.approx(se_wrong_expected)  # NO coincide con normalizar por el conteo de pares sobrevivientes
    assert se_actual < se_wrong_expected  # normalizar por menos pares sobrepondera -- infla el e.e. respecto del correcto, en este ejemplo


# ---------------------------------------------------------------------------
# Bootstrap de bloques por jornada
# ---------------------------------------------------------------------------

def test_day_block_bootstrap_is_reproducible_with_same_seed():
    rng = np.random.default_rng(0)
    values = rng.normal(size=300)
    trading_date = np.repeat(np.arange(30), 10)
    boot1 = day_block_bootstrap(values, trading_date, lambda v: v.mean(), n_boot=50, seed=7)
    boot2 = day_block_bootstrap(values, trading_date, lambda v: v.mean(), n_boot=50, seed=7)
    np.testing.assert_array_equal(boot1, boot2)


def test_day_block_bootstrap_only_draws_from_observed_days():
    values = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    trading_date = np.array(["d1", "d1", "d2", "d2", "d3", "d3"])
    boot = day_block_bootstrap(values, trading_date, lambda v: v.mean(), n_boot=200, seed=0)
    assert boot.min() >= 1.0 and boot.max() <= 3.0


def test_day_block_bootstrap_vector_stat_returns_matrix():
    rng = np.random.default_rng(0)
    values = rng.normal(size=100)
    trading_date = np.repeat(np.arange(10), 10)
    boot = day_block_bootstrap(values, trading_date, lambda v: np.array([v.mean(), v.std()]), n_boot=20, seed=0)
    assert boot.shape == (20, 2)


# ---------------------------------------------------------------------------
# analyze_group (TH11 + TH12 + TH13 combinados)
# ---------------------------------------------------------------------------

def _group_df(
    values: np.ndarray, trading_dates: np.ndarray, timestamps: pd.DatetimeIndex | None = None,
    close: np.ndarray | None = None,
) -> pd.DataFrame:
    if timestamps is None:
        timestamps = pd.date_range("2024-01-08 09:30:00", periods=len(values), freq="1min")
    df = pd.DataFrame({"timestamp": timestamps, "r_1m": values, "trading_date": trading_dates})
    if close is not None:
        df["close"] = close
    return df


def test_analyze_group_handles_n_less_than_2_gracefully():
    df = _group_df(np.array([0.001]), np.array(["d1"]))
    out = analyze_group(df, "r_1m", n_boot=5, seed=0, tick_return_repr=1e-5)
    assert out["n"] == 1
    assert np.isnan(out["hac_se"])


def test_analyze_group_is_reproducible_with_same_seed():
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 0.001, size=500)
    trading_date = np.repeat(np.arange(50), 10).astype(str)
    df = _group_df(values, trading_date)
    out1 = analyze_group(df, "r_1m", n_boot=30, seed=3, tick_return_repr=None)
    out2 = analyze_group(df, "r_1m", n_boot=30, seed=3, tick_return_repr=None)
    assert out1["boot_mean_ci_lo"] == out2["boot_mean_ci_lo"]
    assert out1["freq_diff_ci_hi"] == out2["freq_diff_ci_hi"]


def test_analyze_group_translates_to_ticks_only_when_tick_return_repr_given():
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 0.001, size=300)
    trading_date = np.repeat(np.arange(30), 10).astype(str)
    df = _group_df(values, trading_date)
    with_ticks = analyze_group(df, "r_1m", n_boot=10, seed=0, tick_return_repr=1e-5)
    without_ticks = analyze_group(df, "r_1m", n_boot=10, seed=0, tick_return_repr=None)
    assert not np.isnan(with_ticks["mean_ticks"])
    assert np.isnan(without_ticks["mean_ticks"])


def test_analyze_group_hac_within_a_truly_continuous_block_behaves_as_expected():
    """Dentro de un bloque VERDADERAMENTE continuo (un dia, minuto a minuto), HAC debe seguir detectando dependencia positiva fuerte."""
    rng = np.random.default_rng(0)
    n = 2000
    e = rng.normal(0.0, 1.0, size=n)
    values = np.convolve(e, np.ones(20) / 20, mode="same")
    timestamps = pd.date_range("2024-01-08 09:30:00", periods=n, freq="1min")
    trading_date = np.full(n, datetime.date(2024, 1, 8), dtype=object)
    df = _group_df(values, trading_date, timestamps=timestamps)
    out = analyze_group(df, "r_1m", n_boot=5, seed=0, tick_return_repr=None)
    naive_se = values.std(ddof=1) / np.sqrt(n)
    assert out["hac_se"] > naive_se


def test_analyze_group_segment_population_does_not_fabricate_cross_day_dependence():
    """Poblacion tipo SEGMENTO (mismo minuto del dia, muchos dias): sin ninguna pareja dentro del mismo bloque, HAC debe reducirse exactamente al termino gamma_0/T (equivalente a e.e. iid con ddof=0)."""
    rng = np.random.default_rng(0)
    n_days = 40
    values = rng.normal(0.0, 0.001, size=n_days)
    trading_date = np.array([datetime.date(2024, 1, 8) + datetime.timedelta(days=d) for d in range(n_days)])
    timestamps = pd.to_datetime([pd.Timestamp(d) + pd.Timedelta(hours=10) for d in trading_date])
    df = _group_df(values, trading_date, timestamps=timestamps)
    out = analyze_group(df, "r_1m", n_boot=5, seed=0, tick_return_repr=None)

    e = values - values.mean()
    expected_se = np.sqrt(np.mean(e**2) / n_days)  # gamma_0/T -- ningun termino de rezago j>=1 puede sumar nada
    assert out["hac_l"] >= 1  # el bandwidth SI intenta sumar rezagos...
    assert out["hac_se"] == pytest.approx(expected_se)  # ...pero ninguno encuentra una pareja dentro del mismo bloque


def test_compute_tick_return_repr_matches_manual_formula():
    close = np.array([100.0, 100.25, 100.75, 100.50])
    out = compute_tick_return_repr(close, TICK)
    close_repr = float(np.median(close))
    expected = float(np.log((close_repr + TICK) / close_repr))
    assert out == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Segmentacion de TDA-06 -- particion, nunca modificada
# ---------------------------------------------------------------------------

def _write_segmentation_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_segmentation_cutoffs_filters_only_stable_rows(tmp_path):
    path = tmp_path / "TDA06_segmentacion_propuesta.csv"
    _write_segmentation_csv(path, [
        {"minute_of_day": 120, "n_years_supporting": 3, "years_supporting": "2020,2021,2022", "stable": True},
        {"minute_of_day": 570, "n_years_supporting": 5, "years_supporting": "2020,2021,2022,2023,2024", "stable": True},
        {"minute_of_day": 700, "n_years_supporting": 1, "years_supporting": "2021", "stable": False},
    ])

    class _Cfg:
        tda06_segmentation_csv_path = path

    cutoffs = load_segmentation_cutoffs(_Cfg())
    assert cutoffs == [120, 570]


def test_load_segmentation_cutoffs_does_not_modify_the_source_file(tmp_path):
    path = tmp_path / "TDA06_segmentacion_propuesta.csv"
    _write_segmentation_csv(path, [{"minute_of_day": 570, "n_years_supporting": 5, "years_supporting": "x", "stable": True}])
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    class _Cfg:
        tda06_segmentation_csv_path = path

    load_segmentation_cutoffs(_Cfg())
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == after


def test_assign_segment_label_produces_contiguous_bins():
    minute = pd.Series([0, 119, 120, 569, 570, 1439])
    labels, all_labels = assign_segment_label(minute, cutoffs=[120, 570])
    assert list(labels) == ["00:00-02:00", "00:00-02:00", "02:00-09:30", "02:00-09:30", "09:30-24:00", "09:30-24:00"]
    assert all_labels == ["00:00-02:00", "02:00-09:30", "09:30-24:00"]


# ---------------------------------------------------------------------------
# QQ-plot (sin scipy)
# ---------------------------------------------------------------------------

def test_qq_points_shapes_and_identity_for_standard_normal_like_sample():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=200_000)
    theo, emp = qq_points(x, n_points=200)
    assert theo.shape == (200,) and emp.shape == (200,)
    # Para una muestra realmente normal, cuantil empirico estandarizado ~ cuantil teorico
    np.testing.assert_allclose(theo, emp, atol=0.05)


# ---------------------------------------------------------------------------
# Poblaciones r_1m / r_tilde -- año y segmento adjuntos
# ---------------------------------------------------------------------------

def test_build_r1m_population_only_keeps_valid_rows_and_adds_segment_label():
    rows = _synthetic_multi_day_rows(n_days=3, start=datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    pop, labels = build_r1m_population(variables, validity, cutoffs=[600, 900])
    assert pop["r_1m_valid"].all()
    assert "segment_label" in pop.columns
    assert set(pop["segment_label"].unique()).issubset(set(labels))


def test_build_r_tilde_population_drops_nan_and_preserves_retrospectivo_label():
    r_tilde = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-08 10:00:00", periods=4, freq="1min"),
        "source_file": "f1", "contract": "H24",
        "trading_date": [datetime.date(2024, 1, 8)] * 4,
        "minute_of_day": [600, 601, 602, 603],
        "weekday": [0, 0, 0, 0],
        "r_1m": [np.nan, 0.001, np.nan, -0.001],
        "r_1m_valid": [False, True, False, True],
        "s_m": [np.nan, 1.0, np.nan, 1.0],
        "r_tilde": [np.nan, 0.001, np.nan, -0.001],
        "label": "RETROSPECTIVO",
    })
    pop, labels = build_r_tilde_population(r_tilde, cutoffs=[540, 960])
    assert len(pop) == 2
    assert (pop["label"] == "RETROSPECTIVO").all()
    assert "year_ny" in pop.columns and "segment_label" in pop.columns


# ---------------------------------------------------------------------------
# Invariante #3 -- r_tilde alineado con TDA-04 y etiquetado RETROSPECTIVO
# ---------------------------------------------------------------------------

def _synthetic_r_tilde(variables: pd.DataFrame, label: str = "RETROSPECTIVO") -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": variables["timestamp"].to_numpy(),
        "source_file": variables["source_file"].to_numpy(),
        "contract": variables["contract"].to_numpy(),
        "trading_date": variables["trading_date"].to_numpy(),
        "minute_of_day": 0,
        "weekday": 0,
        "r_1m": variables["r_1m"].to_numpy(),
        "r_1m_valid": True,
        "s_m": 1.0,
        "r_tilde": variables["r_1m"].to_numpy(),
        "label": label,
    })


def test_verify_r_tilde_invariants_passes_when_aligned_and_labeled():
    rows = _synthetic_valid_sequence([100.0, 100.25, 100.75], pd.Timestamp("2024-01-08 10:00:00"), datetime.date(2024, 1, 8))
    variables, _ = _variables_and_validity(rows)
    r_tilde = _synthetic_r_tilde(variables)
    verify_r_tilde_invariants(r_tilde, variables)  # no debe lanzar


def test_verify_r_tilde_invariants_raises_on_row_count_mismatch():
    rows = _synthetic_valid_sequence([100.0, 100.25, 100.75], pd.Timestamp("2024-01-08 10:00:00"), datetime.date(2024, 1, 8))
    variables, _ = _variables_and_validity(rows)
    r_tilde = _synthetic_r_tilde(variables).iloc[:-1]
    with pytest.raises(RTildeInvariantError, match="filas"):
        verify_r_tilde_invariants(r_tilde, variables)


def test_verify_r_tilde_invariants_raises_on_timestamp_mismatch():
    rows = _synthetic_valid_sequence([100.0, 100.25, 100.75], pd.Timestamp("2024-01-08 10:00:00"), datetime.date(2024, 1, 8))
    variables, _ = _variables_and_validity(rows)
    r_tilde = _synthetic_r_tilde(variables).copy()
    r_tilde.loc[1, "timestamp"] = r_tilde.loc[1, "timestamp"] + pd.Timedelta(seconds=1)
    with pytest.raises(RTildeInvariantError, match="no coinciden"):
        verify_r_tilde_invariants(r_tilde, variables)


def test_verify_r_tilde_invariants_raises_when_label_is_not_retrospectivo():
    rows = _synthetic_valid_sequence([100.0, 100.25, 100.75], pd.Timestamp("2024-01-08 10:00:00"), datetime.date(2024, 1, 8))
    variables, _ = _variables_and_validity(rows)
    r_tilde = _synthetic_r_tilde(variables).copy()
    r_tilde.loc[1, "label"] = "CAUSAL"
    with pytest.raises(RTildeInvariantError, match="RETROSPECTIVO"):
        verify_r_tilde_invariants(r_tilde, variables)


# ---------------------------------------------------------------------------
# Orquestador end-to-end (config sintetica, sin abrir raw/holdout)
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path, research_files, holdout_files=None, boundary_utc="2099-01-01 00:00:00"):
    config_dir = tmp_path / "configs"
    config_dir.mkdir(exist_ok=True)
    config_yaml = {
        "instrument": "MNQ",
        "raw_data": {
            "dir": "data/raw/mnq", "file_pattern": "*.Last.txt", "separator": ";",
            "columns": ["timestamp", "open", "high", "low", "close", "volume"],
            "timestamp_format": "%Y%m%d %H%M%S", "timestamp_raw_timezone": "UTC",
        },
        "instrument_spec": {"tick_size": TICK, "tick_size_source": "test"},
        "holdout": {
            "boundary_utc": boundary_utc, "boundary_source": "test",
            "research_files": research_files, "holdout_files": holdout_files or [],
        },
        "tda00": {
            "output_dir_reports": "reports/mnq", "output_dir_interim": "data/interim/mnq",
            "inventory_report": "TDA00_inventario.md", "violations_report": "TDA00_violaciones.csv",
            "per_file_summary": "TDA00_resumen_por_archivo.csv", "bad_data_mask": "tda00_bad_data_mask.parquet",
        },
        "tda04": {
            "output_dir_reports": "reports/mnq", "output_dir_interim": "data/interim/mnq",
            "report_name": "TDA04_variables_analisis.md", "variables_parquet_name": "tda04_variables_1m.parquet",
            "validity_mask_parquet_name": "tda04_return_validity_mask.parquet",
            "losses_by_cause_csv": "TDA04_perdidas_por_causa.csv", "th07_r_vs_R_csv": "TDA04_th07_r_vs_R.csv",
        },
        "tda06": {
            "output_dir_reports": "reports/mnq", "output_dir_interim": "data/interim/mnq",
            "n_boot": 5, "extreme_quantile": 0.99, "smoothing_window_minutes": 15,
            "max_breakpoints": 6, "min_spacing_minutes": 60, "year_stability_tolerance_minutes": 15,
            "year_stability_min_years": 3,
            "report_name": "TDA06_perfil_intradia_calendario.md",
            "minute_global_csv": "TDA06_perfil_minuto_global.csv",
            "minute_by_year_csv": "TDA06_perfil_minuto_por_anio.csv",
            "weekday_csv": "TDA06_perfil_dia_semana.csv",
            "segmentation_csv": "TDA06_segmentacion_propuesta.csv",
            "calibration_csv": "TDA06_calibracion_null.csv",
            "profile_png": "TDA06_perfil_intradia.png",
            "s_m_parquet_name": "tda06_s_m.parquet",
            "r_tilde_parquet_name": "tda06_r_tilde.parquet",
        },
        "tda07": {
            "output_dir_reports": "reports/mnq",
            "n_boot": 5,
            "report_name": "TDA07_distribucion_marginal.md",
            "th08_global_csv": "TDA07_th08_contrafactual_global.csv",
            "th08_by_cause_csv": "TDA07_th08_contrafactual_por_causa.csv",
            "distribution_tables_csv": "TDA07_momentos_cuantiles_drift_colas.csv",
            "qq_global_png": "TDA07_qq_global.png",
            "qq_by_segment_png": "TDA07_qq_por_segmento.png",
            "qq_th08_png": "TDA07_qq_th08_contrafactual.png",
        },
    }
    config_path = config_dir / "mnq_snapshot.yaml"
    config_path.write_text(yaml.safe_dump(config_yaml), encoding="utf-8")
    return load_config(config_path)


def _write_tda04_artifacts(config, variables: pd.DataFrame, validity: pd.DataFrame) -> None:
    config.interim_dir.mkdir(parents=True, exist_ok=True)
    variables.to_parquet(config.tda04_variables_parquet_path, index=False)
    validity.to_parquet(config.tda04_validity_mask_parquet_path, index=False)


def _write_tda06_artifacts(config, r1m_population: pd.DataFrame, cutoffs: list[int]) -> None:
    """Escribe r_tilde/segmentacion sinteticos de TDA-06 DIRECTAMENTE (no ejecuta TDA-06)."""
    config.interim_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)

    r_tilde = pd.DataFrame({
        "timestamp": r1m_population["timestamp"], "source_file": r1m_population["source_file"],
        "contract": r1m_population["contract"], "trading_date": r1m_population["trading_date"],
        "minute_of_day": r1m_population["minute_of_day"], "weekday": r1m_population["weekday"],
        "r_1m": r1m_population["r_1m"], "r_1m_valid": r1m_population["r_1m_valid"],
        "s_m": 1.0, "r_tilde": r1m_population["r_1m"], "label": "RETROSPECTIVO",
    })
    r_tilde.to_parquet(config.tda06_r_tilde_parquet_path, index=False)

    seg_rows = [{"minute_of_day": c, "n_years_supporting": 5, "years_supporting": "2020,2021,2022,2023,2024", "stable": True} for c in cutoffs]
    pd.DataFrame(seg_rows).to_csv(config.tda06_segmentation_csv_path, index=False)


def _prepare_full_fixture(config, n_days=8, start=datetime.date(2024, 1, 8), cutoffs=(600, 900)):
    rows = _synthetic_multi_day_rows(n_days=n_days, start=start)
    variables, validity = _variables_and_validity(rows)
    _write_tda04_artifacts(config, variables, validity)
    from ohlcv_dataroad.ingest.tda06_intraday_calendar_profile import attach_calendar_fields
    # r_tilde de TDA-06 tiene una fila por cada fila de la serie canonica
    # (validas o no, r_tilde=NaN donde r_1m es invalido) -- NUNCA se
    # filtra a r_1m_valid=True antes de escribirla, o la invariante #3
    # (alineacion exacta con tda04_variables_1m.parquet, verify_r_tilde_invariants)
    # no se cumpliria ni siquiera en este fixture sintetico.
    calendar_df = attach_calendar_fields(variables, validity)
    _write_tda06_artifacts(config, calendar_df, list(cutoffs))
    return variables, validity


def test_run_tda07_raises_when_research_and_holdout_overlap(tmp_path):
    config = _make_config(
        tmp_path, research_files=["00_mnq_03_24.Last.txt"],
        holdout_files=["00_mnq_03_24.Last.txt", "no_existe.Last.txt"],
    )
    with pytest.raises(HoldoutIsolationError, match="disjuntos"):
        run_tda07_analysis(config)


def test_run_tda07_raises_when_row_reaches_holdout_boundary(tmp_path):
    config = _make_config(
        tmp_path, research_files=["00_mnq_03_24.Last.txt"],
        holdout_files=["no_existe.Last.txt"],
        boundary_utc="2024-01-08 15:01:00",
    )
    rows = _synthetic_valid_sequence([100.0, 100.25, 100.50], pd.Timestamp("2024-01-08 15:00:00"), datetime.date(2024, 1, 8))
    for r in rows:
        r["source_file"] = "00_mnq_03_24.Last.txt"
    variables, validity = _variables_and_validity(rows)
    _write_tda04_artifacts(config, variables, validity)
    with pytest.raises(HoldoutIsolationError, match="frontera de hold-out"):
        run_tda07_analysis(config)
    assert not (config.raw_dir / "no_existe.Last.txt").exists()


def test_run_tda07_never_opens_any_raw_or_holdout_file(tmp_path):
    config = _make_config(
        tmp_path, research_files=["00_mnq_03_24.Last.txt"],
        holdout_files=["archivo_que_no_existe.Last.txt"],
    )
    _prepare_full_fixture(config)
    assert not (config.raw_dir / "00_mnq_03_24.Last.txt").exists()
    result = run_tda07_analysis(config)
    assert len(result.contrafactual) > 0


def test_run_tda07_raises_on_misaligned_timestamps(tmp_path):
    config = _make_config(tmp_path, research_files=["00_mnq_03_24.Last.txt"])
    rows = _synthetic_valid_sequence([100.0, 100.25, 100.50], pd.Timestamp("2024-01-08 10:00:00"), datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    validity = validity.drop(index=1).reset_index(drop=True)  # rompe la alineacion deliberadamente
    _write_tda04_artifacts(config, variables, validity)
    with pytest.raises(TimestampAlignmentError):
        run_tda07_analysis(config)


def test_run_tda07_raises_on_naive_contradiction(tmp_path):
    config = _make_config(tmp_path, research_files=["00_mnq_03_24.Last.txt"])
    rows = _synthetic_valid_sequence([100.0, 100.25, 100.50], pd.Timestamp("2024-01-08 10:00:00"), datetime.date(2024, 1, 8))
    variables, validity = _variables_and_validity(rows)
    variables = variables.copy()
    variables.loc[2, "r_1m"] = variables.loc[2, "r_1m"] + 1.0  # corrompe una fila VALID directamente en el parquet
    _write_tda04_artifacts(config, variables, validity)
    with pytest.raises(NaiveReturnContradictionError):
        run_tda07_analysis(config)


def test_run_tda07_end_to_end_labels_conserved(tmp_path):
    config = _make_config(tmp_path, research_files=["00_mnq_03_24.Last.txt"])
    _prepare_full_fixture(config)
    result = run_tda07_analysis(config)
    assert (result.contrafactual["label"] == CONTRAFACTUAL_LABEL).all()
    assert (result.r_tilde_population["label"] == "RETROSPECTIVO").all()


def test_run_tda07_end_to_end_segmentation_matches_tda06_file(tmp_path):
    config = _make_config(tmp_path, research_files=["00_mnq_03_24.Last.txt"])
    _prepare_full_fixture(config, cutoffs=(600, 900))
    result = run_tda07_analysis(config)
    assert result.segment_cutoffs == [600, 900]
    assert len(result.segment_labels) == 3


def test_run_tda07_end_to_end_distribution_tables_cover_global_year_segment(tmp_path):
    config = _make_config(tmp_path, research_files=["00_mnq_03_24.Last.txt"])
    _prepare_full_fixture(config, n_days=8)
    result = run_tda07_analysis(config)
    scopes = set(result.distribution_tables["scope"].unique())
    assert scopes == {"GLOBAL", "YEAR", "SEGMENT"}
    assert set(result.distribution_tables["series"].unique()) == {"r_1m", "r_tilde"}
