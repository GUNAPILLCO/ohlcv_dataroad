"""Pruebas SINTETICAS de las reglas de E02 en el motor de la fase 2 (tda17_analysis, tda17_g2, tda17_salidas).

Sin datos de MNQ, sin holdout, sin data/raw. La configuracion es la efectiva E01 + E02 en modo BORRADOR (E02
PENDIENTE_AUDITORIA): prueba la mecanica de las decisiones autorizadas; las rutas de produccion y de ejecucion
oficial deben RECHAZAR mientras E02 no este aprobada y publicada. Las pruebas sobre el repositorio del proyecto
admiten su registro PENDIENTE o APROBADO (aprobacion preparatoria, v11): la ejecucion oficial sigue rechazada. Negativas: tipo de excepcion EXACTO y mensaje
completo (``pytest.raises(..., match=re.escape(...) + "$")`` con ``type(...) is``).
"""
from __future__ import annotations

import copy
import inspect
import math
import re

import numpy as np
import pandas as pd
import pytest

from ohlcv_dataroad.ingest import tda17_analysis as an
from ohlcv_dataroad.ingest import tda17_causalidad as ca
from ohlcv_dataroad.ingest import tda17_enmiendas as enm
from ohlcv_dataroad.ingest import tda17_enmiendas_e02 as m2
from ohlcv_dataroad.ingest import tda17_g2 as g2
from ohlcv_dataroad.ingest import tda17_indicators as ind
from ohlcv_dataroad.ingest import tda17_salidas as sal
from test_tda17_enmienda_02 import NO_OFICIAL_MSG, project_e02_approved, project_rejection

P2_FAKE = "f" * 40


def raises_exactly(exc_type, message, fn, *args, **kwargs):
    """Negativa valida solo con el tipo EXACTO y el mensaje COMPLETO."""
    with pytest.raises(exc_type) as info:
        fn(*args, **kwargs)
    assert type(info.value) is exc_type, type(info.value)
    assert str(info.value) == message, str(info.value)


@pytest.fixture(scope="module")
def cfg():
    return m2.effective_document_e02(borrador=True)


@pytest.fixture(scope="module")
def cfg_e01():
    return enm.load_effective_protocol()


# =========================================================================== configuracion y rutas de produccion
def test_e02_keys_are_present_and_e01_config_is_rejected(cfg, cfg_e01):
    assert an.e02_value(cfg, "analisis.bootstrap.intervalo")["cuantiles"] == [0.05, 0.95]
    assert an.e02_value(cfg, "analisis.auditoria_causalidad")["futuro"]["cortes_aleatorios"]["semilla"] == 3717
    raises_exactly(ind.ProtocolNotAuthorizedError,
                   "la configuracion no incluye la enmienda E02 (analisis.bootstrap.intervalo)",
                   an.e02_value, cfg_e01, "analisis.bootstrap.intervalo")
    for fn in (an.interval_quantiles, an.bootstrap_stream_cells, an.check_null_convention, ca.audit_config):
        with pytest.raises(ind.ProtocolNotAuthorizedError, match="no incluye la enmienda E02"):
            fn(cfg_e01)
    tr = cfg["trazabilidad_protocolo"]
    assert (tr["modo"], tr["activable"], tr["ejecucion_oficial"], tr["publicacion_E02"]) == ("BORRADOR", False, False, None)


def test_decision_11_postponed_and_no_frozen_value_changed(cfg, cfg_e01):
    assert "hueco_espectral_definicion" not in cfg["analisis"]["exactitud"]
    assert cfg["analisis"]["r2"]["hueco_espectral_minimo"] == cfg_e01["analisis"]["r2"]["hueco_espectral_minimo"]
    changed = enm.diff_paths({k: v for k, v in cfg_e01.items() if k != "trazabilidad_protocolo"},
                             {k: v for k, v in cfg.items() if k != "trazabilidad_protocolo"})
    assert changed == {r for _, r in m2.WHITELIST_E02}
    for path in ("analisis.bootstrap", "analisis.clasificacion", "analisis.spearman", "identidades.verificacion_numerica"):
        a, b = cfg_e01, cfg
        for seg in path.split("."):
            a, b = a[seg], b[seg]
        assert {k: v for k, v in b.items() if k in a} == a                  # claves congeladas intactas


@pytest.mark.parametrize("call, oficial", [
    (lambda: an.production_config(), False),
    (lambda: an.official_config(P2_FAKE), True),
    (lambda: an.official_th35(P2_FAKE), True),
    (lambda: ca.official_causal_audit(P2_FAKE), True),
    (lambda: g2.official_g2_controls(P2_FAKE), True),
    (lambda: sal.official_publish_th35(P2_FAKE), True),
    (lambda: sal.check_official_publication(P2_FAKE), True),
    (lambda: sal.reproduce_official_publication(P2_FAKE), True),
])
def test_production_routes_are_blocked_while_e02_is_pending(call, oficial):
    """Proyecto PENDIENTE: toda ruta rechaza (E02_PENDIENTE). Proyecto APROBADO (aprobacion preparatoria): la
    configuracion de produccion es activable pero nunca oficial (E02_NO_OFICIAL) y toda ruta oficial con una P2
    inexistente rechaza (E02_P2_EXISTE)."""
    if oficial or not project_e02_approved():
        raises_exactly(ind.ProtocolNotAuthorizedError, project_rejection(P2_FAKE), call)
        return
    prep = call()
    tr = prep["trazabilidad_protocolo"]
    assert (tr["modo"], tr["activable"], tr["ejecucion_oficial"], tr["publicacion_E02"]) == ("PRODUCCION", True, False, None)
    raises_exactly(ind.ProtocolNotAuthorizedError, NO_OFICIAL_MSG, m2.require_e02_oficial, prep)


def test_official_config_requires_the_official_execution(cfg, cfg_e01, tmp_path):
    """Toda configuracion que no sea la oficial derivada de M2 se rechaza (C1). La publicacion no recibe ninguna
    configuracion ni tabla: solo P2 (``official_publish_th35``)."""
    msg_req = "[E02_REQUERIDA] la configuracion no contiene E01 y E02 APROBADAS y activables"
    raises_exactly(ind.ProtocolNotAuthorizedError, msg_req, m2.require_e02_oficial, cfg, tmp_path)
    raises_exactly(ind.ProtocolNotAuthorizedError, msg_req, m2.require_e02_oficial, cfg_e01, tmp_path)
    prep = copy.deepcopy(cfg)                                              # forma de una aprobacion PREPARATORIA
    prep["trazabilidad_protocolo"].update(modo="PRODUCCION", activable=True)
    prep["trazabilidad_protocolo"]["enmiendas"][-1]["estado"] = "APROBADA"
    raises_exactly(ind.ProtocolNotAuthorizedError,
                   "[E02_NO_OFICIAL] la configuracion no procede de la ejecucion oficial vinculada a P2",
                   m2.require_e02_oficial, prep, tmp_path)
    ofi = copy.deepcopy(prep)                                               # C1: indicadores FABRICADOS
    ofi["trazabilidad_protocolo"].update(ejecucion_oficial=True, publicacion_E02=P2_FAKE)
    raises_exactly(ind.ProtocolNotAuthorizedError,
                   f"[E02_OFICIAL_NO_VERIFICABLE] la configuracion oficial no se puede derivar de M2: "
                   f"{project_rejection(P2_FAKE)}", m2.require_e02_oficial, ofi)
    raises_exactly(ind.ProtocolNotAuthorizedError,
                   f"[E02_OFICIAL_NO_VERIFICABLE] la configuracion oficial no se puede derivar de M2: "
                   f"[GIT] {tmp_path} no es la raiz de un repositorio Git",
                   m2.require_e02_oficial, ofi, tmp_path)
    assert not hasattr(sal, "write_official_table")
    for fn in (sal.official_publish_th35, sal.check_official_publication, sal.reproduce_official_publication):
        assert list(inspect.signature(fn).parameters) == ["publicacion", "root"], fn.__name__
    raises_exactly(an.PendingDefinitionError, f"SALIDAS_OTRAS_TABLAS: definicion pendiente -- {an.PENDIENTES['SALIDAS_OTRAS_TABLAS']}",
                   sal.other_official_table, "TDA17_catalogo.csv")
    assert not (tmp_path / "reports").exists() and not (enm.ROOT / "reports" / "mnq" / "TDA17_catalogo.csv").exists()


# =========================================================================== A1-a + A4-a + F5-a + A2-a
def _registry_panel(cfg, n=1500, seed=23):
    rng = np.random.default_rng(seed)
    cols = an.p_columns(cfg)
    out_b = {i["fuera_de_B"]: i for i in cfg["identidades"]["lineales"] if i["panel"] == "P"}
    data = {c: rng.standard_normal(n) for c in cols if c not in out_b}
    for c, ident in out_b.items():
        coef = dict((col, k) for k, col in ident["terminos"])[c]
        data[c] = (ident["constante"] - sum(k * data[col] for k, col in ident["terminos"] if col != c)) / coef
    panel = pd.DataFrame({c: data[c] for c in cols})
    df = pd.DataFrame({"high": np.full(n, 2.0), "low": np.ones(n), "r_1m_valid": np.ones(n, bool)})
    return panel, df


def test_rank_scope_follows_a4(cfg):
    assert an.rank_scope("GLOBAL", cfg) == "BLOQUEANTE"
    assert [an.rank_scope(f"ANIO_{y}", cfg) for y in (2019, 2020, 2025)] == ["INFORMATIVA", "BLOQUEANTE", "BLOQUEANTE"]
    assert an.rank_scope("SEG_09:30-16:02", cfg) == "BLOQUEANTE"
    assert an.rank_scope("ANIO_2021__SEG_09:30-16:02", cfg) == "INFORMATIVA"
    raises_exactly(ind.ProtocolNotAuthorizedError, "celda no declarada: ANIO_2018", an.rank_scope, "ANIO_2018", cfg)
    raises_exactly(ind.ProtocolNotAuthorizedError, "celda no declarada: SEG_10:00-11:00", an.rank_scope, "SEG_10:00-11:00", cfg)
    bad = copy.deepcopy(cfg)
    bad["identidades"]["verificacion_numerica"]["ambito"]["bloqueantes"] = ["GLOBAL"]
    with pytest.raises(ind.ProtocolNotAuthorizedError, match="ambito del control distinto del implementado"):
        an.rank_scope("GLOBAL", bad)


def test_unregistered_dependency_blocks_only_in_blocking_cells_and_emits_the_dossier(cfg):
    panel, df = _registry_panel(cfg)
    panel["C7_n20__ln_sigma"] = 2.0 * panel["C8_n20__ln_sigma"] + 1.0
    out = an.analyze_cell(panel, df, cfg, celda="SEG_09:30-16:02")
    assert out["estado_conclusiones"] == "BLOQUEADO_HASTA_EXPLICAR" and out["alcance_bloqueo"] == "TH35"
    dz = out["diagnostico_bloqueo"]
    assert dz["estado"] == "BLOQUEADO_HASTA_EXPLICAR" and dz["levantamiento"] == "ENMIENDA_FORMAL_POSTERIOR_AUDITADA_CICLO_M2"
    rk = {r["matriz"]: r for r in out["rango"]}
    affected = {k for k, r in rk.items() if r["estado"] == "BLOQUEAR_HASTA_EXPLICAR"}
    assert {"P_centrada", "B_centrada"} <= affected and set(dz["espectros"]) == set(dz["direcciones_nulas"]) == affected
    for what, d in dz["direcciones_nulas"].items():
        assert len(d) == rk[what]["autovalores"]                            # una fila por direccion nula
        assert (d["autovalor"] <= 1e-10 * dz["espectros"][what]["autovalores"].max()).all()
    loads = dz["direcciones_nulas"]["B_centrada"].drop(columns="autovalor")
    assert loads.shape[1] == 123 and (loads[["C7_n20__ln_sigma", "C8_n20__ln_sigma"]].abs() > 0.1).all().all()
    assert (dz["r2_diagnostico"]["R2_TODAS"]["marca"] == "DIAGNOSTICO_BLOQUEADO").all()
    rel = {r["id"]: r for r in dz["relaciones_N"]}
    assert list(rel) == [f"N{i}" for i in range(1, 22)] and all(r["valor_numerico"] is None for r in rel.values())
    assert {k for k, r in rel.items() if r["tipo"] in ("LA-P", "NL-P")} == {"N2", "N4", "N7", "N8", "N9", "N11", "N17", "N19"}
    c17 = rel["N17"]["cargas"]["B_centrada"]                                # cargas observadas de C7/C8 en la direccion nula
    assert len(c17) == rk["B_centrada"]["autovalores"] and (c17[["C7_n20__ln_sigma", "C8_n20__ln_sigma"]].abs() > 0.1).all().all()
    assert set(rel["N4"]["cargas"]) == affected and "cargas" not in rel["N3"] and rel["N3"]["diagnostico"] == "SIN_DIRECCION_EN_P"
    viol = int((panel["log_co"].abs() > panel["log_hl"]).sum())             # panel aleatorio: viola la desigualdad
    assert viol > 0 and (rel["N21"]["violaciones"], rel["N21"]["filas"]) == (viol, len(panel)) and rel["N21"]["estado"] is None
    assert rel["N10"]["diagnostico"] == "NO_IDENTIDAD" and rel["N1"]["estado"] == "PENDIENTE_EXPEDIENTE"
    assert dz["expediente"] == ["celdas", "recuentos", "direcciones", "relaciones_candidatas", "atribucion", "tratamiento_propuesto"]
    info = an.analyze_cell(panel, df, cfg, celda="ANIO_2019")              # misma matriz, celda INFORMATIVA
    assert info["bloqueos"] == [] and info["estado_conclusiones"] == "SIN_BLOQUEO" and "diagnostico_bloqueo" not in info
    assert {r["estado"] for r in info["rango"]} == {"DIFIERE_INFORMATIVO"}                 # las 3 matrices difieren
    sens = an.analyze_cell(panel, df, cfg, celda="GLOBAL", sensibilidad="S3")
    assert sens["alcance_bloqueo"] == "SENSIBILIDAD_S3" and sens["diagnostico_bloqueo"]["ambito_afectado"]["alcance"] == "SENSIBILIDAD_S3"


def test_undefined_matrix_f5a(cfg):
    panel, df = _registry_panel(cfg, n=600)
    panel["B3_n14__rsi"] = 50.0                                             # varianza cero en la poblacion
    out = an.analyze_cell(panel, df, cfg, celda="GLOBAL")
    assert out["estado_matriz"] == "NO_DEFINIDA" and out["varianza_cero"] == ["B3_n14__rsi"] and out["norma_cero"] == []
    assert out["bloqueos"] == ["MATRIZ_NO_DEFINIDA: celda GLOBAL con columnas sin varianza ['B3_n14__rsi']"]
    assert out["estado_conclusiones"] == "BLOQUEADO_HASTA_EXPLICAR" and "rango" not in out and "r2_todas" not in out
    year = an.analyze_cell(panel, df, cfg, celda="ANIO_2020")
    assert year["bloqueos"] == [] and year["estado_conclusiones"] == "INCUMPLIMIENTO_TODAS_LAS_COLUMNAS"
    panel["B3_n14__rsi"] = 0.0
    zero = an.analyze_cell(panel, df, cfg, celda="SEG_00:00-02:00")
    assert zero["norma_cero"] == ["B3_n14__rsi"] and zero["varianza_cero"] == ["B3_n14__rsi"]
    assert an.analyze_cell(panel, df, cfg, celda="GLOBAL", sensibilidad="S1")["alcance_bloqueo"] == "SENSIBILIDAD_S1"
    panel["B3_n14__rsi"] = 50.0 + 1e-9 * np.arange(len(panel))             # varianza diminuta pero positiva: DEFINIDA
    assert an.analyze_cell(panel, df, cfg, celda="GLOBAL")["estado_matriz"] == "DEFINIDA"


def test_null_convention_check_rejects_another_convention(cfg):
    assert an.check_null_convention(cfg)["cantidad"] == "AUTOVALORES"
    bad = copy.deepcopy(cfg)
    bad["identidades"]["verificacion_numerica"]["convencion"]["cantidad"] = "VALORES_SINGULARES"
    with pytest.raises(ind.ProtocolNotAuthorizedError, match="convencion de nulos distinta de la implementada"):
        an.check_null_convention(bad)


def test_negative_rounding_eigenvalues_count_as_null_without_clipping():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(4000)
    X = np.c_[x, 2 * x + 1, rng.standard_normal(4000)]
    st = an.cell_stats(X)
    lam = np.linalg.eigvalsh(an.normalized_matrix(st))
    assert an.null_counts(st, 1e-10)["autovalores"] == int(np.sum(lam <= 1e-10 * lam.max())) == 1


# =========================================================================== B6 + B2-a: flujos por celda
def test_stream_order_and_generators(cfg):
    cells = an.bootstrap_stream_cells(cfg)
    assert cells[:8] == ["GLOBAL"] + [f"ANIO_{y}" for y in range(2019, 2026)]
    assert cells[8:] == [f"SEG_{lab}" for lab in cfg["analisis"]["celdas"]["segmentos"]["etiquetas"]] and len(cells) == 15
    for i, c in enumerate(cells):
        ref = np.random.Generator(np.random.PCG64(np.random.SeedSequence(1717).spawn(15)[i]))
        np.testing.assert_array_equal(an.cell_generator(cfg, c).integers(0, 10**9, 5), ref.integers(0, 10**9, 5))
    raises_exactly(ind.ProtocolNotAuthorizedError, "celda sin flujo bootstrap declarado: ANIO_2020__SEG_09:30-16:02",
                   an.cell_generator, cfg, "ANIO_2020__SEG_09:30-16:02")


def test_cell_draws_consume_one_integers_call_per_replica_and_s4_restarts(cfg):
    D = 37
    draws = an.cell_draws(cfg, "ANIO_2022", D, bloque=5)
    assert draws.shape == (300, D)
    ref = an.cell_generator(cfg, "ANIO_2022")
    for r in range(300):
        starts = ref.integers(0, D, size=math.ceil(D / 5))
        np.testing.assert_array_equal(draws[r], ((starts[:, None] + np.arange(5)) % D).ravel()[:D])
    np.testing.assert_array_equal(an.cell_draws(cfg, "ANIO_2022", D, bloque=5, replicas=30), draws[:30])
    s4 = an.cell_draws(cfg, "ANIO_2022", D, bloque=10)
    first = an.cell_generator(cfg, "ANIO_2022").integers(0, D, size=math.ceil(D / 10))
    np.testing.assert_array_equal(s4[0], ((first[:, None] + np.arange(10)) % D).ravel()[:D])
    assert an.cell_draws(cfg, "ANIO_2022", D, bloque=1).shape == (300, D)
    raises_exactly(ind.ProtocolNotAuthorizedError, "longitud de bloque no declarada: 7", an.cell_draws, cfg, "GLOBAL", D, bloque=7)
    raises_exactly(ValueError, "numero de replicas fuera de 1..300: 301", an.cell_draws, cfg, "GLOBAL", D, bloque=5, replicas=301)


# =========================================================================== B1-a + B3-a + F2-a: IC de R2
def _days_panel(n_days=40, per_day=30, p=4, seed=5):
    rng = np.random.default_rng(seed)
    days = np.repeat(np.arange(n_days), per_day)
    X = rng.standard_normal((days.size, p))
    X[:, 1] += 0.8 * X[:, 0]
    X[:, 3] = 0.5 * X[:, 0] - 0.4 * X[:, 2] + 0.3 * rng.standard_normal(days.size)
    return X, days


def test_bootstrap_r2_matches_recomputation_on_resampled_rows(cfg):
    X, days = _days_panel()
    names = ["u", "v", "w", "z"]
    m = an.daily_moments(X, days)
    draws = an.cell_draws(cfg, "GLOBAL", m.n.size, bloque=5, replicas=40)
    point = an.r2_table(an.cell_stats(X), names, None, 1e-10, 1e-8)
    out = an.bootstrap_r2(m, draws, names, point, cfg).set_index("columna")
    for j, nm in enumerate(names):
        ref = []
        for d in draws:
            rows = np.concatenate([np.flatnonzero(days == k) for k in d])
            C = np.corrcoef(X[rows], rowvar=False)
            ref.append(an.r2_basis(C, j, [k for k in range(4) if k != j], 1e-10))
        lo, hi = np.quantile(ref, [0.05, 0.95])                             # tipo 7 = interpolacion lineal (numpy)
        assert out.loc[nm, "ic_estado"] == "CALCULADO" and out.loc[nm, "replicas_invalidas"] == 0
        assert out.loc[nm, "ic_inf"] == pytest.approx(lo, abs=1e-10) and out.loc[nm, "ic_sup"] == pytest.approx(hi, abs=1e-10)
    assert out.attrs["tamano_base"].shape == (40, 4) and (out.attrs["tamano_base"] == 3).all()


def test_invalid_replicas_make_the_interval_not_computable_and_are_never_discarded(cfg):
    X, days = _days_panel(n_days=12)
    X[days != 0, 2] = 1.0                                                   # w solo varia en la jornada 0
    names = ["u", "v", "w", "z"]
    m = an.daily_moments(X, days)
    draws = an.cell_draws(cfg, "SEG_02:00-03:00", m.n.size, bloque=5, replicas=60)
    point = an.r2_table(an.cell_stats(X), names, None, 1e-10, 1e-8)
    out = an.bootstrap_r2(m, draws, names, point, cfg).set_index("columna")
    without_day0 = [r for r, d in enumerate(draws) if 0 not in d]
    assert without_day0                                                     # el flujo produce replicas sin la jornada 0
    for nm in names:                                                        # en R2_TODAS intervienen todas las columnas
        assert out.loc[nm, "ic_estado"] == "IC_NO_CALCULABLE" and np.isnan(out.loc[nm, "ic_inf"])
        assert out.loc[nm, "indices_invalidas"] == without_day0
        assert out.loc[nm, "invalidas_por_causa"] == {"VAR_CERO": len(without_day0)} and out.loc[nm, "replicas"] == 60


def test_r2_official_checked_causes():
    C = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])    # no semidefinida: R2 > 1
    v, cause, nb = an.r2_official_checked(C, 0, [1, 2], 1e-10)
    assert cause == "FUERA_DE_RANGO" and v > 1 and nb == 2
    assert an.r2_official_checked(C, 0, [], 1e-10)[1:] == ("BASE_DEGENERADA", 0)
    Cn = C.copy()
    Cn[1, 2] = Cn[2, 1] = np.nan
    assert an.r2_official_checked(Cn, 0, [1, 2], 1e-10)[1] == "NO_FINITO"
    ok = np.array([[1.0, 0.5], [0.5, 1.0]])
    assert an.r2_official_checked(ok, 0, [1], 1e-10) == (pytest.approx(0.25), None, 1)


def test_registered_and_non_publishable_columns(cfg):
    X, days = _days_panel()
    reg = sorted(an.registered_columns(cfg))[0]
    names = ["u", reg, "w", "z"]
    m = an.daily_moments(X, days)
    draws = an.cell_draws(cfg, "GLOBAL", m.n.size, bloque=5, replicas=20)
    point = an.r2_table(an.cell_stats(X), names, None, 1e-10, 1e-8)
    point.loc[point["columna"] == "w", "publicable"] = False
    out = an.bootstrap_r2(m, draws, names, point, cfg).set_index("columna")
    assert (out.loc[reg, "ic_estado"], out.loc[reg, "ic_inf"], out.loc[reg, "ic_sup"]) == ("DEGENERADO_REGISTRO", 1.0, 1.0)
    assert out.loc["w", "ic_estado"] == "R2_NO_PUBLICABLE" and np.isnan(out.loc["w", "ic_inf"])
    assert out.loc["u", "ic_estado"] == "CALCULADO"


def test_interval_and_replica_rules_must_be_the_implemented_ones(cfg):
    X, days = _days_panel(n_days=10)
    m = an.daily_moments(X, days)
    point = an.r2_table(an.cell_stats(X), list("abcd"), None, 1e-10, 1e-8)
    bad = copy.deepcopy(cfg)
    bad["analisis"]["bootstrap"]["estadistico_replica"]["replica_invalida"]["tratamiento"] = "DESCARTAR"
    with pytest.raises(ind.ProtocolNotAuthorizedError, match="estadistico de replica distinto del implementado"):
        an.bootstrap_r2(m, np.zeros((2, 10), int), list("abcd"), point, bad)
    bad = copy.deepcopy(cfg)
    bad["analisis"]["bootstrap"]["intervalo"]["metodo_cuantil"] = "BCA"
    with pytest.raises(ind.ProtocolNotAuthorizedError, match="intervalo distinto del implementado"):
        an.interval_quantiles(bad)


# =========================================================================== Spearman: rangos fijos y validacion (F3-I + F3-I-b)
def test_spearman_fixed_ranks_interval_matches_resampled_fixed_ranks(cfg):
    X, days = _days_panel()
    ranks = an.average_ranks(X)
    m_rank = an.daily_moments(ranks, days)
    draws = an.cell_draws(cfg, "ANIO_2024", m_rank.n.size, bloque=5, replicas=30)
    rho = an.cell_stats(ranks).corr
    tab = an.bootstrap_spearman_fixed(m_rank, draws, list("abcd"), rho, cfg)
    assert len(tab) == 6 and (tab["ic_estado"] == "CALCULADO").all()
    rec = tab[(tab["columna_i"] == "a") & (tab["columna_j"] == "d")].iloc[0]
    ref = [np.corrcoef(ranks[np.concatenate([np.flatnonzero(days == k) for k in d])], rowvar=False)[0, 3] for d in draws]
    assert rec["ic_inf"] == pytest.approx(np.quantile(ref, 0.05), abs=1e-12)
    assert rec["ic_sup"] == pytest.approx(np.quantile(ref, 0.95), abs=1e-12)
    assert rec["rho"] == pytest.approx(rho[0, 3])


def test_spearman_constant_ranks_in_a_replica_invalidate_the_pair_only(cfg):
    X, days = _days_panel(n_days=12)
    X[days != 3, 1] = 7.0
    ranks = an.average_ranks(X)
    m_rank = an.daily_moments(ranks, days)
    draws = an.cell_draws(cfg, "GLOBAL", m_rank.n.size, bloque=5, replicas=40)
    tab = an.bootstrap_spearman_fixed(m_rank, draws, list("abcd"), an.cell_stats(ranks).corr, cfg)
    with_b = tab[(tab["columna_i"] == "b") | (tab["columna_j"] == "b")]
    assert (with_b["ic_estado"] == "IC_NO_CALCULABLE").all()
    assert with_b["invalidas_por_causa"].map(lambda d: set(d) == {"VAR_CERO"}).all()
    assert (tab.drop(with_b.index)["ic_estado"] == "CALCULADO").all()


def _validation_cells(seed=9):
    out = {}
    for k, c in enumerate(["GLOBAL", "ANIO_2024"]):
        X, days = _days_panel(n_days=25, per_day=20, seed=seed + k)
        out[c] = (X, days)
    return out


def test_spearman_validation_passes_with_30_and_uses_the_stream_prefix(cfg):
    cells = _validation_cells()
    res = an.spearman_validation(cells, cfg)
    assert res["resultado"] == "VALIDADA" and res["publicar_ic_spearman"] is True
    assert [r["replicas"] for r in res["rondas"]] == [30]
    X, days = cells["GLOBAL"]
    ranks = an.average_ranks(X)
    m_rank = an.daily_moments(ranks, days)
    rbd = an.day_rows(days)
    draws = an.cell_draws(cfg, "GLOBAL", 25, bloque=5)[:30]
    f = np.array([an.replica_correlation(m_rank, d)[np.triu_indices(4, 1)] for d in draws])
    e = np.array([an.exact_rank_replica_correlation(X, rbd, d)[np.triu_indices(4, 1)] for d in draws])
    lf, hf = np.quantile(f, [0.05, 0.95], axis=0)
    le, he = np.quantile(e, [0.05, 0.95], axis=0)
    expected = float(np.median(np.r_[np.abs(lf - le), np.abs(hf - he)]))
    assert res["rondas"][0]["celdas"][0]["mediana_dif_ic"] == pytest.approx(expected, abs=1e-15)


def test_spearman_validation_extended_then_not_published(cfg):
    cells = _validation_cells()
    strict = copy.deepcopy(cfg)
    strict["analisis"]["spearman"]["validacion"]["tolerancia_mediana_dif_ic"] = 0.0
    res = an.spearman_validation(cells, strict)
    assert res["resultado"] == "NO_VALIDADA" and res["publicar_ic_spearman"] is False
    assert [r["replicas"] for r in res["rondas"]] == [30, 100] and not any(r["pasa"] for r in res["rondas"])
    table = pd.DataFrame({"columna_i": ["a"], "columna_j": ["b"], "rho": [0.5], "ic_inf": [0.4], "ic_sup": [0.6],
                          "ic_estado": ["CALCULADO"]})
    pub = an.spearman_publication(table, res)
    assert pub["rho"].tolist() == [0.5] and pub[["ic_inf", "ic_sup"]].isna().all().all()
    assert pub["ic_estado"].tolist() == ["IC_NO_PUBLICADO_VALIDACION_FALLIDA"]
    assert an.spearman_publication(table, an.spearman_validation(cells, cfg)).equals(table)


def test_spearman_validation_with_invalid_replica_fails_the_cell_in_that_round_s_a(cfg):
    """Decision 14 (S-a): una replica INVALIDA en cualquiera de los dos metodos hace fallar la validacion de ESA celda
    en esa ronda (sin imputar ni excluir el par); despues, F3-I-b (ronda ampliada y, si vuelve a fallar, IC no
    publicados). La otra celda conserva su mediana."""
    cells = _validation_cells()
    X, days = cells["ANIO_2024"]
    X = X.copy()
    X[days != 0, 2] = 3.0                                                   # rangos de la columna 2 constantes sin la jornada 0
    cells["ANIO_2024"] = (X, days)
    res = an.spearman_validation(cells, cfg)
    assert res["resultado"] == "NO_VALIDADA" and res["publicar_ic_spearman"] is False
    assert [r["replicas"] for r in res["rondas"]] == [30, 100] and not any(r["pasa"] for r in res["rondas"])
    for rnd in res["rondas"]:
        g, y = rnd["celdas"]
        assert g["celda"] == "GLOBAL" and g["replicas_invalidas"] == [] and g["mediana_dif_ic"] is not None
        draws = an.cell_draws(cfg, "ANIO_2024", 25, bloque=5)[:rnd["replicas"]]
        expected = [r for r, d in enumerate(draws) if 0 not in d]
        assert expected and y["replicas_invalidas"] == expected
        assert (y["mediana_dif_ic"], y["motivo"]) == (None, "REPLICA_INVALIDA")
    loose = copy.deepcopy(cfg)                                              # ni con tolerancia infinita pasa: no hay imputacion
    loose["analisis"]["spearman"]["validacion"]["tolerancia_mediana_dif_ic"] = float("inf")
    assert an.spearman_validation(cells, loose)["resultado"] == "NO_VALIDADA"
    clean = _validation_cells()
    assert an.spearman_validation(clean, loose)["resultado"] == "VALIDADA"
    for key, val in (("regla", "IMPUTAR_INFINITO"), ("imputacion", "PEOR_CASO"), ("exclusion_del_par", "PERMITIDA"),
                     ("metodos", ["RANGOS_FIJOS"])):
        bad = copy.deepcopy(cfg)
        bad["analisis"]["spearman"]["validacion_replica_invalida"][key] = val
        with pytest.raises(ind.ProtocolNotAuthorizedError, match="regla de replica invalida en la validacion distinta"):
            an.spearman_validation(clean, bad)
    raises_exactly(ind.ProtocolNotAuthorizedError, "la validacion exige exactamente las celdas ['GLOBAL', 'ANIO_2024']",
                   an.spearman_validation, {"GLOBAL": cells["GLOBAL"], "ANIO_2023": cells["GLOBAL"]}, cfg)


# =========================================================================== B13-a, B9, bloqueo de conclusiones
def _per_cell(r2=0.95, lo=0.92, hi=0.97, bad=None):
    rows = []
    for y in [2020, 2021, 2022, 2023, 2024, 2025]:
        rows.append({"celda": f"ANIO_{y}", "tipo": "anio", "estado": "EVALUABLE", "r2": r2, "ic_inf": lo, "ic_sup": hi,
                     "publicable": True, "ic_estado": "CALCULADO", "estado_matriz": "DEFINIDA"})
    for k in range(7):
        rows.append({"celda": f"SEG_{k}", "tipo": "segmento", "estado": "EVALUABLE", "r2": r2, "ic_inf": lo, "ic_sup": hi,
                     "publicable": True, "ic_estado": "CALCULADO", "estado_matriz": "DEFINIDA"})
    out = pd.DataFrame(rows)
    for cell, col, val in (bad or []):
        out.loc[out["celda"] == cell, col] = val
    return out


def test_non_publishable_cells_count_as_failure_of_both_conditions(cfg):
    assert an.classify_column(_per_cell(), cfg) == "RECONSTRUIBLE"
    one = [("ANIO_2020", "ic_estado", "IC_NO_CALCULABLE")]
    assert an.classify_column(_per_cell(bad=one), cfg) == "RECONSTRUIBLE"                      # 5 de 6 anios
    two = one + [("ANIO_2021", "estado_matriz", "NO_DEFINIDA")]
    assert an.classify_column(_per_cell(bad=two), cfg) == "INESTABLE"                         # 4 de 6: no se descuenta
    low = [("ANIO_2020", "publicable", False), ("ANIO_2021", "publicable", False)]
    assert an.classify_column(_per_cell(0.3, 0.2, 0.4, bad=low), cfg) == "INESTABLE"           # tampoco NUEVA
    assert an.classify_column(_per_cell(0.3, 0.2, 0.4, bad=low[:1]), cfg) == "NUEVA_REPRESENTACION"
    counts = an.non_publishable_counts(_per_cell(bad=two + [("SEG_3", "publicable", False)]), cfg)
    assert counts == {"MATRIZ_NO_DEFINIDA": 1, "R2_NO_PUBLICABLE_POR_COLINEALIDAD": 1, "IC_NO_CALCULABLE": 1,
                      "total_no_publicables": 3}
    reg = _per_cell(bad=[(f"ANIO_{y}", "ic_estado", "DEGENERADO_REGISTRO") for y in range(2020, 2026)])
    assert an.classify_column(reg, cfg) == "RECONSTRUIBLE"


def test_classification_and_subniveles_are_blocked_until_explained(cfg):
    pc = pd.concat([_per_cell().assign(columna="x"), _per_cell(0.3, 0.2, 0.4).assign(columna="y")])
    blk = ["BLOQUEAR_HASTA_EXPLICAR: nulos de P_centrada = 12"]
    raises_exactly(an.BlockedUntilExplainedError, f"CLASIFICACION_TH35 BLOQUEADA_HASTA_EXPLICAR: {blk}",
                   an.classify_columns, pc, cfg, blk)
    r2g = pd.DataFrame({"columna": ["x"], "r2": [0.7], "publicable": [True]})
    raises_exactly(an.BlockedUntilExplainedError, f"SUBNIVELES BLOQUEADA_HASTA_EXPLICAR: {blk}", an.subniveles, r2g, cfg, blk)
    assert issubclass(an.BlockedUntilExplainedError, ind.ProtocolNotAuthorizedError)
    tab = an.classify_columns(pc, cfg, []).set_index("columna")
    assert tab["clasificacion"].to_dict() == {"x": "RECONSTRUIBLE", "y": "NUEVA_REPRESENTACION"}
    assert (tab["total_no_publicables"] == 0).all()


def test_subniveles_use_the_global_point_r2(cfg):
    reg = sorted(an.registered_columns(cfg))[0]
    r2g = pd.DataFrame({"columna": [reg, "a", "b", "c", "d"], "r2": [1.0, 0.7, 0.2, 0.95, 0.6],
                        "publicable": [True, True, True, True, False]})
    out = an.subniveles(r2g, cfg, []).set_index("columna")
    assert out["subnivel"].drop("d").to_dict() == {reg: "EXACTA", "a": "PARCIAL", "b": "LINEALMENTE_DISTINTA", "c": "SIN_SUBNIVEL"}
    assert (out.loc["d", "subnivel"], out.loc["d", "motivo"]) == ("SIN_SUBNIVEL_R2_NO_PUBLICABLE",
                                                                  "R2_NO_PUBLICABLE_POR_COLINEALIDAD")   # N-a
    assert out.loc["d", "r2_global"] == 0.6                                 # se publica el R2, sin sustituirlo
    na = copy.deepcopy(cfg)
    na["analisis"]["clasificacion"]["subniveles_r2_global_no_publicable"]["sustitucion"] = "R2_ESPECTRAL"
    with pytest.raises(ind.ProtocolNotAuthorizedError, match="subnivel con R2 global no publicable distinto del implementado"):
        an.subniveles(r2g, na, [])
    bad = copy.deepcopy(cfg)
    bad["analisis"]["clasificacion"]["subniveles_r2"]["fuente"] = "MEDIANA_DE_CELDAS"
    with pytest.raises(ind.ProtocolNotAuthorizedError, match="fuente de los subniveles distinta de la implementada"):
        an.subniveles(r2g, bad, [])


# =========================================================================== R-a corregida: anexo N -> columnas de P
def test_relations_annex_types_and_columns(cfg):
    rels = {r["id"]: r for r in an.relations_n(cfg)}
    assert list(rels) == [r["id"] for r in cfg["identidades"]["no_lineales_o_fuera_de_P"]]
    types = {}
    for k, r in rels.items():
        types.setdefault(r["tipo"], []).append(k)
    assert types == {"LA-P": ["N4"], "LA-F": ["N3", "N5", "N12", "N13", "N20"],
                     "NL-P": ["N2", "N7", "N8", "N9", "N11", "N17", "N19"], "NL-F": ["N1", "N6", "N14", "N15", "N16", "N18"],
                     "DES": ["N21"], "DIAG": ["N10"]}
    l10 = next(i for i in cfg["identidades"]["lineales"] if i["id"] == "L10")
    assert rels["N4"]["coincide_con"] == "L10" and set(rels["N4"]["columnas_P"]) == {c for _, c in l10["terminos"]}
    rn = an.e02_value(cfg, "identidades.verificacion_numerica.relaciones_N")
    assert rn["nulos_esperados_adicionales"] == 0                           # ninguna N anade nulos esperados
    assert cfg["identidades"]["verificacion_numerica"]["nulos_esperados"] == {"P_sin_centrar": 10, "P_centrada": 11,
                                                                              "B_centrada": 0, "L": 3}


@pytest.mark.parametrize("mutate", [
    lambda rn: rn.update(nulos_esperados_adicionales=1),
    lambda rn: rn["relaciones"].pop(),
    lambda rn: rn["relaciones"][0].update(tipo="LINEAL"),
    lambda rn: rn["relaciones"][0].update(columnas_P=["no_existe"]),
    lambda rn: rn["relaciones"][3].update(coincide_con="L99"),
    lambda rn: rn["relaciones"][20]["desigualdad"].update(menor_o_igual_que="r_1m"),
    lambda rn: rn["tipos"]["NL-P"].update(diagnostico="SIN_DIRECCION_EN_P"),
    lambda rn: rn.update(valor_numerico="EVALUAR_FORMULA"),
])
def test_relations_annex_is_rejected_if_different(cfg, mutate):
    bad = copy.deepcopy(cfg)
    mutate(bad["identidades"]["verificacion_numerica"]["relaciones_N"])
    with pytest.raises(ind.ProtocolNotAuthorizedError, match="^anexo de relaciones N distinto del implementado"):
        an.relations_n(bad)


def test_relation_diagnostics_by_type(cfg):
    rels = an.relations_n(cfg)
    names = an.p_columns(cfg)
    d = pd.DataFrame([[0.0] + [0.5 if c in ("C5_n60__pos", "B5_n60__wr") else 0.0 for c in names]],
                     columns=["autovalor"] + names)
    panel = pd.DataFrame({"log_co": [0.1, -0.3, 0.2, 0.0], "log_hl": [0.2, 0.3, 0.1, 0.0]})   # fila 2: |0.2| > 0.1
    out = {r["id"]: r for r in an.relation_diagnostics(rels, {"P_centrada": d}, panel)}
    assert out["N4"]["cargas"]["P_centrada"].iloc[0].to_dict() == {"autovalor": 0.0, "C5_n60__pos": 0.5, "B5_n60__wr": 0.5}
    assert (out["N21"]["violaciones"], out["N21"]["filas"]) == (1, 4)
    assert all("cargas" not in out[k] for k in ("N1", "N3", "N10", "N21"))
    assert {k: out[k]["diagnostico"] for k in ("N1", "N3", "N10", "N21")} == {
        "N1": "SIN_DIRECCION_EN_P", "N3": "SIN_DIRECCION_EN_P", "N10": "NO_IDENTIDAD", "N21": "RECUENTO_DE_VIOLACIONES"}


def test_exact_by_cell_stays_blocked_by_decision_11():
    raises_exactly(an.PendingDefinitionError, f"HUECO_ESPECTRAL: definicion pendiente -- {an.PENDIENTES['HUECO_ESPECTRAL']}",
                   an.publish_exact_by_cell)
    assert "APLAZADA" in an.PENDIENTES["HUECO_ESPECTRAL"]


# =========================================================================== extremo a extremo: celda + bootstrap (mecanica)
def test_cell_bootstrap_end_to_end_on_the_registry_panel(cfg):
    small = copy.deepcopy(cfg)
    small["analisis"]["bootstrap"]["replicas"] = 12                       # solo mecanica (copia de prueba)
    panel, df = _registry_panel(cfg, n=1200)
    days = np.repeat(np.arange(40), 30)
    out = an.analyze_cell(panel, df, small, celda="GLOBAL")
    assert out["bloqueos"] == [] and out["estado_matriz"] == "DEFINIDA"
    bs = an.cell_bootstrap(panel, days, out, small, "GLOBAL")
    r2 = bs["r2_todas"].set_index("columna")
    reg = an.registered_columns(cfg)
    assert (r2.loc[sorted(reg), "ic_estado"] == "DEGENERADO_REGISTRO").all()
    free = r2.drop(sorted(reg))
    assert (free["ic_estado"] == "CALCULADO").all() and (free["ic_inf"] <= free["ic_sup"]).all()
    assert ((free["ic_inf"] >= 0) & (free["ic_sup"] <= 1)).all()
    assert len(bs["spearman"]) == 134 * 133 // 2 and bs["jornadas"] == 40
    nd = an.analyze_cell(panel.assign(B3_n14__rsi=1.0), df, small, celda="ANIO_2020")
    raises_exactly(ValueError, "la celda no tiene matriz definida: sin bootstrap (F5-a)", an.cell_bootstrap,
                   panel, days, nd, small, "ANIO_2020")


def test_panel_l_rank_checks_are_informative(cfg):
    rng = np.random.default_rng(1)
    cols = an.l_columns(cfg)
    n = 800
    series = pd.DataFrame(rng.standard_normal((n, len(cols))), columns=cols)
    series[cols[1]] = series[cols[0]] * 2.0                                  # dependencia exacta adicional
    df = pd.DataFrame({"contract": ["A"] * 400 + ["B"] * 400})
    out = an.panel_l_rank_checks(series, df, cfg, np.ones(n, bool))
    assert [r["matriz"] for r in out] == ["L_A", "L_B"]
    assert all(r["ambito"] == "INFORMATIVA" and r["estado"] == "DIFIERE_INFORMATIVO" for r in out)
    series.loc[:399, cols[2]] = 1.0
    out = an.panel_l_rank_checks(series, df, cfg, np.ones(n, bool))
    assert out[0]["estado"] == "NO_DEFINIDA" and out[1]["estado"] == "DIFIERE_INFORMATIVO"
    assert g2.g2_replica_rank_check({"autovalores": 1, "singulares": 0}, 0, "G2_2_r0")["estado"] == "DIFIERE_INFORMATIVO"


# =========================================================================== garantias estaticas (checklist M2)
PHASE2_MODULES = ["tda17_analysis.py", "tda17_g2.py", "tda17_causalidad.py", "tda17_salidas.py", "tda17_enmiendas_e02.py",
                  "tda17_entradas.py"]


def _src(name):
    return (enm.ROOT / "src" / "ohlcv_dataroad" / "ingest" / name).read_text(encoding="utf-8")


def test_no_omission_mechanisms_in_the_real_implementation():
    pattern = re.compile(r"_DISABLED|os\.environ|getenv|omitir_controles|skip_checks|desactivar|bypass", re.I)
    for name in PHASE2_MODULES:
        assert not pattern.search(_src(name)), name
    for fn in (m2.official_config_e02, m2.load_effective_protocol_e02, an.official_config, an.production_config,
               an.official_th35, ca.official_causal_audit, g2.official_g2_controls, sal.official_publish_th35,
               sal.check_official_publication, sal.reproduce_official_publication):
        params = set(inspect.signature(fn).parameters) - {"root", "publicacion", "_args", "_kwargs"}
        assert params == set(), (fn.__name__, params)


def test_phase2_never_uses_the_e01_loader_to_produce_results():
    for name in ["tda17_analysis.py", "tda17_g2.py", "tda17_causalidad.py", "tda17_salidas.py", "tda17_entradas.py"]:
        src = _src(name)
        if name == "tda17_causalidad.py":                                   # unico uso: comparar el catalogo con E01
            guard = inspect.getsource(ca.check_catalog_config)
            assert guard.count("load_effective_protocol()") == 1 and "raise ProtocolNotAuthorizedError" in guard
            src = src.replace(guard, "")
        assert not re.search(r"\bload_effective_protocol\(", src), name
        assert "._compute(" not in src and "compute_catalog_synthetic" not in src, name
        assert "effective_document(" not in _src(name) and "effective_document_e02(" not in _src(name), name


def test_only_the_single_publication_route_writes_files():
    """Decision 18: la escritura solo ocurre dentro de ``official_publish_th35`` (que recibe solo P2) y despues de
    obtener la configuracion oficial y ejecutar ``official_th35`` en el mismo proceso; ninguna otra funcion de la fase
    2 escribe ficheros ni recibe una tabla para publicarla."""
    for name in ["tda17_analysis.py", "tda17_g2.py", "tda17_causalidad.py"]:
        for token in ("read_parquet", "read_csv", "open(", "data/raw", "to_parquet", "to_csv", "write_text", "write_bytes"):
            assert token not in _src(name), (name, token)
    mod = _src("tda17_salidas.py")
    publish = inspect.getsource(sal.official_publish_th35)
    run = inspect.getsource(sal._official_run)
    assert "os.replace(" not in mod and "shutil.move(" not in mod and "os.rename(" not in mod   # sin sobrescritura
    promote = inspect.getsource(sal._promote)                     # enlace duro; el preparado ancla la propiedad
    assert "os.link(src, dst)" in promote and "unlink" not in promote
    assert "write_bytes(" not in mod and "write_text" not in mod and mod.count("open(") == mod.count("os.open(") == 2
    assert mod.count("to_csv(") == inspect.getsource(sal.csv_bytes).count("to_csv(") == 1        # serializa, no escribe
    assert run.index("an.official_config(publicacion, root)") < run.index("an.official_th35(publicacion, root)") < \
        run.index("official_tables(result, cfg)")
    # publicacion transaccional: las primitivas de escritura solo estan en la ruta y en sus auxiliares privados, que
    # solo llama la ruta (o se llaman entre si), despues del calculo oficial
    writers = {"_exclusive", "_write_durable", "_promote", "_recover", "_rollback", "_stage_and_promote",
               "official_publish_th35"}
    funcs = {n: inspect.getsource(f) for n, f in vars(sal).items() if inspect.isfunction(f) and f.__module__ == sal.__name__}
    funcs["_exclusive"] = inspect.getsource(sal._exclusive.__wrapped__)
    for name, src in funcs.items():
        if any(t in src for t in ("os.open(", "os.write(", "os.rename(", "os.link(", "os.unlink(", ".unlink(", "rmtree(",
                                  ".mkdir(")):
            assert name in writers, name
    for helper in writers - {"official_publish_th35"}:
        callers = {n for n, src in funcs.items() if n != helper and f"{helper}(" in src}
        assert callers and callers <= writers, (helper, callers)
    assert publish.index("_official_run(publicacion, root)") < publish.index(".mkdir(") < \
        publish.index("with _exclusive(out_dir):") < publish.index("_recover(out_dir)") < \
        publish.index("SALIDA_EXISTENTE") < publish.index("_stage_and_promote(out_dir, files)")
    public = {n for n, f in vars(sal).items() if inspect.isfunction(f) and f.__module__ == sal.__name__ and not n.startswith("_")}
    assert public == {"schema", "diagnostic_table", "official_tables", "csv_bytes", "official_publish_th35",
                      "check_official_publication", "reproduce_official_publication", "other_official_table"}


def test_every_official_route_obtains_the_official_config():
    routes = [an.official_th35, ca.official_causal_audit, g2.official_g2_controls]
    for fn in routes:
        assert "official_config(publicacion, root)" in inspect.getsource(fn), fn.__name__
    for fn in (an.official_th35, ca.official_causal_audit):              # datos: solo las entradas verificadas
        assert "ent.load_official_inputs(cfg, root)" in inspect.getsource(fn), fn.__name__
    found = {f"{mod.__name__}.{n}" for mod in (an, ca, g2) for n, f in vars(mod).items()
             if n.startswith("official_") and inspect.isfunction(f) and f.__module__ == mod.__name__}
    assert found == {"ohlcv_dataroad.ingest.tda17_analysis.official_config",
                     "ohlcv_dataroad.ingest.tda17_analysis.official_th35",
                     "ohlcv_dataroad.ingest.tda17_causalidad.official_causal_audit",
                     "ohlcv_dataroad.ingest.tda17_g2.official_g2_controls"}


def test_th35_pipeline_derives_everything_and_depends_on_th37():
    """Procedencia y C2: la ruta oficial de TH35 solo recibe P2; el orquestador ejecuta y verifica TH37 antes de
    calcular nada y deriva celdas, IC, per_cell y R2 global; no existe ningun parametro para aportarlos."""
    assert list(inspect.signature(an.official_th35).parameters) == ["publicacion", "root"]
    assert list(inspect.signature(an.th35_pipeline).parameters) == ["df", "cfg"]
    src = inspect.getsource(an.th35_pipeline)
    order = [src.index(t) for t in ("ca.run_causal_audit(df, cfg)", "ca.verify_causal_audit(table, df, cfg)",
                                    "ind.compute_catalog(df)", "ind.build_panel_p(series, df)", "analyze_cell(sub, sub_df, cfg",
                                    "cell_bootstrap(sub", "_cell_rows(cell", "classify_columns(per_cell, cfg, blocks)",
                                    "subniveles(r2_global, cfg, blocks)")]
    assert order == sorted(order)
    assert not hasattr(an, "official_classification") and not hasattr(an, "official_cell_results")
