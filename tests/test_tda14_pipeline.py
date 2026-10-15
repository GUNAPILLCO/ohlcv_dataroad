"""Tests de integracion de TDA-14 (pipeline, configuracion, reconciliacion, informe). SOLO datos sinteticos: ningun
test lee data/, los informes reales de reports/ ni el hold-out."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from ohlcv_dataroad.ingest import run_tda14
from ohlcv_dataroad.ingest import tda14_pipeline as pl
from ohlcv_dataroad.ingest import tda14_temporal_stability as t14
from ohlcv_dataroad.ingest.holdout_guard import HoldoutIsolationError

REAL_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "tda14.yaml"

BASE = dict(n_sessions=180, width=240, active_minutes=240, base_sigma=3e-4, daily_phi=0.8, daily_sd=0.05, seasonal_amplitude=0.0,
            seasonal_period=252, t_df=4.0, intra_phi=0.98, intra_sd=0.05, profile_amplitude=1.0, ma_theta=0.0, drift_sigma=0.0,
            base_price=15000.0, tick=0.25, roll_every=45, gap_probability=0.05)
DETECTOR = dict(batch_sessions=3, window_batches=3, step_batches=1, min_side_windows=4, min_evaluable_windows=12, min_session_bar_fraction=0.9,
                min_window_complete_fraction=57 / 63, min_window_bar_fraction=0.9, min_tail_support=20, min_profile_bin_bars=20, max_lag=10,
                profile_bin_minutes=30, min_batches_for_se=3)
CUTOFFS = [19 * 60, 20 * 60 + 30]          # tres segmentos dentro de 18:00-22:00 NY (el panel sintetico cubre 240 minutos)


def synthetic_config_dict() -> dict:
    raw = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
    raw["protocolo"]["estado"] = pl.PROTOCOL_FROZEN
    raw["protocolo"]["detector"] = dict(DETECTOR)
    raw["protocolo"]["mundo_base"] = dict(BASE)
    raw["protocolo"]["variantes_negativas"]["NEGATIVO_ESTACIONAL"]["parametros"] = {"seasonal_amplitude": 0.05}
    raw["protocolo"]["potencia"] = [p for p in raw["protocolo"]["potencia"] if p["familia"] == t14.FAM_SCALE]
    raw["protocolo"]["minimos"] = {"negativos": 30, "parcial": 10, "tipo": 5, "potencia": 5}
    raw["calibracion"] = {"mundos_negativos_por_variante": 100, "mundos_estres_por_variante": 0, "mundos_tipo_por_control": 6, "n_workers": 1}
    raw["analisis"]["n_boot"] = 5
    raw["eventos"].append({"id": "EVENTO_SINTETICO", "fecha_negociacion": "2022-06-01", "tipo": "prueba", "cambio": "prueba",
                           "fuente": "test", "ventana_reloj_ny": {"inicio": "19:00", "fin": "19:15", "hipotesis": "prueba"}})
    return raw


def make_cfg(tmp_path: Path, raw: dict, name: str = "tda14_test.yaml") -> pl.TDA14Config:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return pl.load_tda14_config(p)


def spread_dates(n: int) -> np.ndarray:
    """Jornadas repartidas en 2020-2024 (36 por año) para ejercitar las tablas anuales."""
    days = np.concatenate([np.datetime64(f"{y}-01-06") + np.arange(n // 5) * 9 for y in range(2020, 2025)])
    return days.astype("datetime64[D]")


def synthetic_inputs(changes=(), rep=0, **over) -> pl.TDA14Inputs:
    spec = t14.WorldSpec(**{**BASE, **over}, label="ANALISIS", changes=changes)
    panel = t14.simulate_world(spec, t14.world_rng(t14.ROLE_TEST, "ANALISIS", rep))
    panel.session_date = spread_dates(panel.n_sessions)
    seg, labels = pl.segments_for_columns(CUTOFFS, width=panel.r.shape[1])
    return pl.TDA14Inputs(panel, seg, labels, {"barras": int(panel.present.sum()), "barras_con_anio_distinto": 0, "coinciden": True}, "SINTETICO")


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    return make_cfg(tmp_path_factory.mktemp("cfg"), synthetic_config_dict())


@pytest.fixture(scope="module")
def calibration(cfg):
    return pl.run_calibration(cfg)


# ===========================================================================
# Configuracion real y bloqueos
# ===========================================================================

FROZEN_MANIFEST_SHA256 = "350b4eac372c3532df81305e110cd245f43c99b7b5d9d11e6b48f51808ca9522"


def test_real_config_is_frozen_complete_and_not_authorized_for_mnq():
    """Protocolo congelado: sin pendientes, manifiesto con la identidad aprobada, analisis MNQ no autorizado."""
    from ohlcv_dataroad.ingest import tda14_stop14_rules as rules
    real = pl.load_tda14_config(REAL_CONFIG)
    assert real.protocol["estado"] == pl.PROTOCOL_FROZEN and real.raw["analisis"]["autorizado_mnq"] is False
    assert pl.pending_parameters(real) == []
    man = pl.build_manifest(real)
    assert man.sha256() == FROZEN_MANIFEST_SHA256
    assert all(f != t14.FAM_MAGNITUDE for f, _ in man.required_power)          # sin potencia validada: nunca ESTABLE
    assert real.raw["stop14"]["reglas_version"] == rules.RULES_VERSION
    pl.reject_manual_stop14_checks(real)                                     # sin comprobaciones manuales
    assert pl.detector_from_config(real) == t14.DetectorSpec()          # el detector de la configuracion es el auditado


def test_calibration_guards_still_refuse_provisional_or_incomplete_protocols():
    real = pl.load_tda14_config(REAL_CONFIG)
    prov = replace(real, raw={**real.raw, "protocolo": {**real.protocol, "estado": pl.PROTOCOL_PROVISIONAL}})
    with pytest.raises(pl.TDA14ConfigError, match="exige 'CONGELADO'"):
        pl.run_calibration(prov)
    incomplete = copy.deepcopy(real.raw)
    incomplete["protocolo"]["mundo_base"]["daily_phi"] = None
    with pytest.raises(pl.TDA14ConfigError, match="pendientes"):
        pl.run_calibration(replace(real, raw=incomplete))


def test_real_config_events_are_the_four_predeclared_ones():
    ev = pl.load_tda14_config(REAL_CONFIG).raw["eventos"]
    assert [e["fecha_negociacion"] for e in ev] == ["2020-10-12", "2020-10-26", "2021-06-28", "2021-11-01"]
    assert all("Submission" in e["fuente"] for e in ev) and ev[0]["mnq_citado"] == "anexos A y B"
    assert ev[2]["ventana_reloj_ny"] == {"inicio": "16:15", "fin": "16:30", "hipotesis": ev[2]["ventana_reloj_ny"]["hipotesis"]}


def test_cli_refuses_analysis_without_authorization_and_reads_nothing(monkeypatch):
    called = []
    monkeypatch.setattr(pl, "load_tda14_inputs", lambda *_: called.append(1))
    assert run_tda14.main(["analizar", "--config-tda14", str(REAL_CONFIG)]) == 2
    assert called == []


# ===========================================================================
# Carga: guardias heredadas
# ===========================================================================

def _fake_tda04(boundary_ok: bool = True):
    ts = pd.to_datetime(["2021-01-04 23:00", "2021-01-04 23:01", "2021-01-04 23:02", "2021-01-05 23:00", "2021-01-05 23:01"])
    if not boundary_ok:
        ts = ts + pd.Timedelta(days=2000)
    variables = pd.DataFrame({"timestamp": ts, "trading_date": ["2021-01-05"] * 3 + ["2021-01-06"] * 2, "contract": ["MNQH1"] * 5,
                              "r_1m": [np.nan, 0.001, 0.0, np.nan, -0.002], "close": [1.0, 2, 2, 3, 3], "volume": [1, 2, 3, 4, 5],
                              "source_file": ["a"] * 5})
    validity = pd.DataFrame({"timestamp": ts, "r_1m_valid": [False, True, True, False, True], "trading_date": variables["trading_date"]})
    return variables, validity


def _fake_config():
    return SimpleNamespace(holdout_boundary_utc="2025-06-23", research_files=["a"], holdout_files=["b"])


def test_loader_applies_holdout_guards_in_order(monkeypatch):
    calls = []
    monkeypatch.setattr(pl, "validate_research_holdout_disjoint", lambda c: calls.append("disjuntos"))
    monkeypatch.setattr(pl, "load_tda04_inputs", lambda c: (calls.append("lectura"), _fake_tda04())[1])
    monkeypatch.setattr(pl, "validate_last_timestamps_before_boundary", lambda c, d: calls.append("frontera"))
    monkeypatch.setattr(pl, "load_segmentation_cutoffs", lambda c: [120, 180, 510, 570, 962, 1200])
    inp = pl.load_tda14_inputs(_fake_config())
    assert calls == ["disjuntos", "lectura", "frontera"]
    assert inp.panel.n_sessions == 2 and inp.year_check["coinciden"] and len(inp.segment_labels) == 7
    assert inp.panel.r[0, 1] == 0.001 and np.isnan(inp.panel.r[0, 0])
    assert inp.year_ny_by_cell.shape == inp.panel.r.shape and inp.year_ny_by_cell[0, 0] == 2021 and inp.year_ny_by_cell[0, 5] == -1


def test_loader_stops_before_reading_when_holdout_lists_overlap(monkeypatch):
    monkeypatch.setattr(pl, "load_tda04_inputs", lambda c: pytest.fail("no debe leerse nada"))
    with pytest.raises(HoldoutIsolationError):
        pl.load_tda14_inputs(SimpleNamespace(holdout_boundary_utc="2025-06-23", research_files=["a"], holdout_files=["a"]))


def test_loader_refuses_rows_beyond_holdout_boundary(monkeypatch):
    monkeypatch.setattr(pl, "load_tda04_inputs", lambda c: _fake_tda04(boundary_ok=False))
    monkeypatch.setattr(pl, "load_segmentation_cutoffs", lambda c: [120])
    with pytest.raises(HoldoutIsolationError):
        pl.load_tda14_inputs(_fake_config())


# ===========================================================================
# Tablas
# ===========================================================================

def test_group_bootstrap_equals_audited_annual_table():
    inp = synthetic_inputs()
    spec = t14.DetectorSpec(**DETECTOR)
    ours = pl.annual_tables(inp.panel, spec, n_boot=4, seed=3, blocks=(5,))
    ref = t14.annual_table(inp.panel, n_boot=4, block_sessions=5, seed=3, spec=spec)
    m = ours.merge(ref, on=["anio", "estadistico"], suffixes=("", "_ref"))
    assert len(m) == len(ref) > 0
    for c in ("valor", "ic95_inf", "ic95_sup"):
        np.testing.assert_allclose(m[c], m[f"{c}_ref"], rtol=0, atol=0, equal_nan=True)


def test_annual_year_segment_and_window_tables_have_intervals_and_evaluability():
    inp = synthetic_inputs()
    spec = t14.DetectorSpec(**DETECTOR)
    ann = pl.annual_tables(inp.panel, spec, 3, 0)
    assert set(ann["anio"]) == {2020, 2021, 2022, 2023, 2024} and set(ann["bloque_jornadas"]) == {1, 5, 20}
    assert {"ic95_inf", "ic95_sup", "evaluable", "motivo_no_evaluable", "estado_calibracion"} <= set(ann.columns)
    vol = ann.loc[ann["estadistico"] == "log_volumen_medio"]
    assert set(vol["estado_calibracion"]) == {t14.VOLUME_STATUS} and not vol["en_detector"].any()
    ys = pl.year_segment_tables(inp.panel, inp.segment_by_column, inp.segment_labels, spec, 2, 0, blocks=(5,))
    assert set(ys["segmento"]) == set(inp.segment_labels) and (~ys["evaluable"]).any()      # segmentos de pocos minutos: no todo es evaluable
    ps = t14.analyze_panel(inp.panel, spec)
    win = pl.window_tables(inp.panel, ps, spec, 2, 0, blocks=(1,))
    assert win["ventana"].nunique() == len(ps.window_start)
    assert {"error_comun_detector", "error_por_lotes_ventana", "sin_solapamiento", "cobertura_suficiente"} <= set(win.columns)


def test_segment_tables_never_pair_across_segments():
    inp = synthetic_inputs()
    seg0 = np.flatnonzero(inp.segment_by_column == inp.segment_labels[0])
    sp = pl.masked_panel(inp.panel, seg0)
    assert np.isnan(sp.r[:, seg0.max() + 1:]).all()
    runs = t14.continuity_runs(np.isfinite(sp.r))
    pm = t14.pair_mask(np.isfinite(sp.r), runs, 5)
    assert not pm[:, seg0.max() - 4:].any()


# ===========================================================================
# Reconciliacion
# ===========================================================================

def _independent_acf(values, valid, cell_years, lags):
    """Implementacion INDEPENDIENTE de la ACF heredada: recorre cada tramo de retornos validos consecutivos; cada par se
    etiqueta con el año de la barra mas reciente."""
    out = {}
    for lag in lags:
        acc = {}
        for i in range(values.shape[0]):
            j = 0
            row_ok = valid[i]
            while j < values.shape[1]:
                if not row_ok[j]:
                    j += 1
                    continue
                k = j
                while k < values.shape[1] and row_ok[k]:
                    k += 1
                for t in range(j + lag, k):
                    acc.setdefault(str(cell_years[i, t]), []).append((values[i, t], values[i, t - lag]))
                j = k
        for g, pairs in acc.items():
            p = np.asarray(pairs)
            out[(g, lag)] = float(np.corrcoef(p[:, 0], p[:, 1])[0, 1])
    return out


def write_inherited(panel, d: Path, cell_years: np.ndarray | None = None) -> None:
    """Tablas 'heredadas' sinteticas calculadas con las definiciones de TDA-05/07/08/09/12, agrupando por el año de CADA
    BARRA (``cell_years``; por defecto, el año de su jornada)."""
    yr = pd.to_datetime(panel.session_date).year.to_numpy()
    cy = np.broadcast_to(yr[:, None], panel.r.shape) if cell_years is None else cell_years
    valid = np.isfinite(panel.r)
    r07, r05, r12 = [], [], []
    for y in np.unique(cy[valid]):
        x = panel.r[(cy == y) & valid]
        m, m2 = x.mean(), ((x - x.mean()) ** 2).mean()
        q01, q99 = np.quantile(x, [0.01, 0.99])
        r07.append({"series": "r_1m", "scope": "YEAR", "scope_value": str(y), "n": x.size, "mean": m, "std": x.std(ddof=1),
                    "kurtosis_excess": ((x - m) ** 4).mean() / m2 ** 2 - 3, "q0.01": q01, "q0.99": q99})
        r05.append({"year_ny": y, "n": x.size, "zero_fraction": (x == 0).mean()})
        for lvl, q in ((0.01, q01), (0.99, q99)):
            r12.append({"series": "r_1m", "block": "raw", "grouping": "YEAR", "group": str(y), "level": lvl, "q": q, "std": x.std(ddof=1),
                        "q_over_std": q / x.std(ddof=1)})
    pd.DataFrame(r07).to_csv(d / "TDA07_momentos_cuantiles_drift_colas.csv", index=False)
    pd.DataFrame(r05).to_csv(d / "TDA05_resolucion_por_anio.csv", index=False)
    pd.DataFrame(r12).to_csv(d / "TDA12_cuantiles_condicionales.csv", index=False)
    acf = _independent_acf(panel.r, valid, cy, (1,))
    pd.DataFrame([{"group_type": "year", "group": g, "lag": lag, "rho": v} for (g, lag), v in acf.items()]).to_csv(d / "TDA08_acf_por_grupo.csv", index=False)
    acf = _independent_acf(np.abs(panel.r), valid, cy, (1, 5, 20, 60))
    pd.DataFrame([{"variable": "abs_r", "raw_adjusted": "raw", "group": g, "lag": lag, "rho": v} for (g, lag), v in acf.items()]).to_csv(
        d / "TDA09_persistencia_por_anio.csv", index=False)


def test_reconciliation_reproduces_inherited_definitions_and_flags_discrepancies(tmp_path):
    inp = synthetic_inputs()
    spec = t14.DetectorSpec(**DETECTOR)
    ann = pl.annual_tables(inp.panel, spec, 2, 0, blocks=(1,))
    write_inherited(inp.panel, tmp_path)
    rec = pl.reconcile(inp.panel, ann, inp.segment_by_column, tmp_path, inp.year_check)
    declared_repro = rec.loc[(rec["clase_declarada"] == pl.REC_REPRODUCIBLE) & (rec["valor_heredado"].notna())]
    assert len(declared_repro) > 50 and set(declared_repro["clase"]) == {pl.REC_REPRODUCIBLE}
    verified = rec.loc[rec["clase"] == pl.REC_EXPLAINED + "_VERIFICADA"]
    assert {"log_sigma -> std(ddof=1)", "curtosis_total -> kurtosis_excess", "log_q99_en_sigma -> q_over_std"} <= set(verified["estadistico"])
    assert set(rec.loc[rec["clase_declarada"] == pl.REC_NOT_COMPARABLE, "clase"]) == {pl.REC_NOT_COMPARABLE}
    assert (rec["clase"] == pl.REC_DISCREPANCY).sum() == 0
    # Un valor heredado alterado se marca como discrepancia (nunca se oculta).
    t07 = pd.read_csv(tmp_path / "TDA07_momentos_cuantiles_drift_colas.csv")
    t07.loc[0, "std"] *= 1.001
    t07.to_csv(tmp_path / "TDA07_momentos_cuantiles_drift_colas.csv", index=False)
    rec2 = pl.reconcile(inp.panel, ann, inp.segment_by_column, tmp_path, inp.year_check)
    flagged = rec2.loc[rec2["clase"] == pl.REC_DISCREPANCY]
    # la std alterada y las dos comprobaciones verificadas que dependen de ella (sigma y media en sigmas), solo de ese año
    assert set(flagged["estadistico"]) == {"std", "log_sigma -> std(ddof=1)", "media_en_sigma -> mean/std(ddof=1)"}
    assert flagged["grupo"].nunique() == 1 and set(flagged["etapa"]) == {"TDA-07", "TDA-14 vs TDA-07"}
    # Archivos ausentes: HEREDADO_AUSENTE explicito.
    rec3 = pl.reconcile(inp.panel, ann, inp.segment_by_column, tmp_path / "no_existe", inp.year_check)
    assert pl.REC_ABSENT in set(rec3["clase"]) and pl.REC_REPRODUCIBLE not in set(rec3["clase"])


def _shift_year(inp):
    """Población con asignación de año distinta: las barras de la primera jornada de 2021 tienen año NY 2020."""
    yr = pd.to_datetime(inp.panel.session_date).year.to_numpy()
    ny = np.broadcast_to(yr[:, None], inp.panel.r.shape).astype(np.int16).copy()
    first_2021 = int(np.flatnonzero(yr == 2021)[0])
    ny[first_2021, :] = 2020
    return ny, {"coinciden": False, "barras_con_anio_distinto": int(inp.panel.present[first_2021].sum())}


def test_year_assignment_difference_is_never_explained_without_demonstration(tmp_path):
    """Corrección de cierre: que algunas barras tengan otro año NO basta para declarar 'explicada' una diferencia anual.
    Sin el año NY por barra no se puede demostrar: la fila queda como poblaciones no directamente comparables."""
    inp = synthetic_inputs()
    ann = pl.annual_tables(inp.panel, t14.DetectorSpec(**DETECTOR), 2, 0, blocks=(1,))
    ny, check = _shift_year(inp)
    write_inherited(inp.panel, tmp_path, ny)
    rec = pl.reconcile(inp.panel, ann, inp.segment_by_column, tmp_path, check)          # sin año por barra
    diff = rec.loc[rec["etapa"].isin(["TDA-07", "TDA-05", "TDA-12", "TDA-08", "TDA-09"]) & rec["grupo"].isin(["2020", "2021"])
                   & (rec["clase_declarada"] == pl.REC_REPRODUCIBLE) & (rec["clase"] != pl.REC_REPRODUCIBLE)]
    assert len(diff) > 0 and set(diff["clase"]) == {pl.REC_NOT_DIRECT}
    declared = rec.loc[rec["clase_declarada"] == pl.REC_REPRODUCIBLE, "clase"].astype(str)
    assert not declared.str.startswith(pl.REC_EXPLAINED).any()                  # nada declarado reproducible se da por explicado
    # años no afectados: siguen siendo reproducibles
    assert set(rec.loc[(rec["etapa"] == "TDA-07") & (rec["grupo"] == "2023"), "clase"]) == {pl.REC_REPRODUCIBLE}


def test_population_difference_does_not_hide_an_independent_numeric_error(tmp_path):
    """Con el año NY por barra, la diferencia de población se DEMUESTRA (recálculo idéntico -> _VERIFICADA). Un error
    numérico independiente en otro valor del mismo año sigue apareciendo como DISCREPANCIA_NO_EXPLICADA."""
    inp = synthetic_inputs()
    ann = pl.annual_tables(inp.panel, t14.DetectorSpec(**DETECTOR), 2, 0, blocks=(1,))
    ny, check = _shift_year(inp)
    write_inherited(inp.panel, tmp_path, ny)
    rec = pl.reconcile(inp.panel, ann, inp.segment_by_column, tmp_path, check, ny)
    t07 = rec.loc[(rec["etapa"] == "TDA-07") & rec["grupo"].isin(["2020", "2021"])]
    assert set(t07["clase"]) <= {pl.REC_REPRODUCIBLE, pl.REC_EXPLAINED + "_VERIFICADA"}
    assert (t07["clase"] == pl.REC_EXPLAINED + "_VERIFICADA").sum() >= 5 and (rec["clase"] == pl.REC_DISCREPANCY).sum() == 0
    # error independiente: q0.99 de 2021 alterado en el archivo heredado
    p = tmp_path / "TDA07_momentos_cuantiles_drift_colas.csv"
    t = pd.read_csv(p)
    t.loc[t["scope_value"].astype(str) == "2021", "q0.99"] *= 1.0005
    t.to_csv(p, index=False)
    rec2 = pl.reconcile(inp.panel, ann, inp.segment_by_column, tmp_path, check, ny)
    bad = rec2.loc[rec2["clase"] == pl.REC_DISCREPANCY]
    assert list(zip(bad["etapa"], bad["grupo"], bad["estadistico"])) == [("TDA-07", "2021", "q0.99")]
    assert "año NY de cada barra" in bad.iloc[0]["explicacion"]
    # el resto de 2021 sigue demostrado como diferencia de asignación de año
    ok = rec2.loc[(rec2["etapa"] == "TDA-07") & (rec2["grupo"] == "2021") & (rec2["estadistico"] != "q0.99")]
    assert set(ok["clase"]) <= {pl.REC_REPRODUCIBLE, pl.REC_EXPLAINED + "_VERIFICADA"}


# ===========================================================================
# Calibracion de extremo a extremo (pequeña), persistencia y trazabilidad
# ===========================================================================

def test_calibration_end_to_end_freezes_and_validates_under_one_manifest(calibration, cfg):
    out = calibration
    man = pl.build_manifest(cfg)
    assert out.manifest.sha256() == man.sha256()
    assert dict(out.thresholds.calibration_info)["manifiesto_sha256"] == man.sha256() == out.validation.manifest_sha256
    assert out.validation.thresholds_sha256 == out.thresholds.sha256 and out.thresholds.type_margin is not None
    n_neg = len(man.negative_specs)          # variantes negativas heredadas de la configuracion real (4 en el protocolo congelado)
    assert n_neg == 4
    assert out.n_worlds == {"calibracion": 100 * n_neg, "calibracion_tipos": 24, "validacion": 30 * n_neg + 10 + 5 * 4 + 5 * 2}
    st = out.validation.specs_table
    assert st["mundos_verificados"].sum() == out.n_worlds["validacion"]
    # Con 30 mundos por control no puede demostrarse <= 5 %: NO VALIDADO, sin forzar conclusiones.
    assert not out.validation.global_validated


def test_calibration_respects_memory_protection(cfg, monkeypatch):
    small = copy.deepcopy(cfg.raw)
    small["calibracion"].update(mundos_negativos_por_variante=1, mundos_tipo_por_control=1)
    monkeypatch.setattr(t14, "available_memory_gb", lambda: 5.0)          # (5 - 4) * 0,6 / 1 GB -> 0 -> 1 proceso
    c = replace(cfg, raw=small)
    assert t14.resolve_workers(4, 10) == 1
    with pytest.raises((t14.CalibrationFailure, t14.ValidationManifestError)):
        pl.run_calibration(c, n_workers=4)          # 2 mundos negativos no bastan para calibrar: falla de forma explicita


def test_persisted_calibration_roundtrip_and_tamper_detection(calibration, cfg, tmp_path):
    paths = run_tda14.persist_calibration(calibration, cfg, tmp_path)
    assert all(p.exists() for p in paths)
    a = cfg.artifacts
    raw = copy.deepcopy(cfg.raw)
    raw["analisis"].update(umbrales=a["umbrales_json"], validacion=a["validacion_json"], tabla_validacion=a["validacion_csv"])
    c2 = replace(cfg, raw=raw)
    thr, val = run_tda14.load_calibration(c2, tmp_path)
    assert thr.sha256 == calibration.thresholds.sha256 and val.global_validated == calibration.validation.global_validated
    assert val.power_ok == calibration.validation.power_ok
    d = json.loads((tmp_path / a["umbrales_json"]).read_text(encoding="utf-8"))
    d["k_star"] += 1
    (tmp_path / a["umbrales_json"]).write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(t14.FrozenThresholdsError):
        run_tda14.load_calibration(c2, tmp_path)
    run_tda14.persist_calibration(calibration, cfg, tmp_path)
    with open(tmp_path / a["validacion_csv"], "a", encoding="utf-8") as fh:
        fh.write("x\n")
    with pytest.raises(t14.ValidationManifestError, match="huella"):
        run_tda14.load_calibration(c2, tmp_path)


def test_calibration_from_other_protocol_is_rejected_at_load(calibration, cfg, tmp_path):
    run_tda14.persist_calibration(calibration, cfg, tmp_path)
    raw = copy.deepcopy(cfg.raw)
    a = cfg.artifacts
    raw["analisis"].update(umbrales=a["umbrales_json"], validacion=a["validacion_json"], tabla_validacion=a["validacion_csv"])
    raw["protocolo"]["minimos"]["potencia"] = 6                                   # otro protocolo
    with pytest.raises(t14.ValidationManifestError, match="manifiesto"):
        run_tda14.load_calibration(replace(cfg, raw=raw), tmp_path)


# ===========================================================================
# Analisis de extremo a extremo e informe
# ===========================================================================

STEP_MULTI = (t14.Change(t14.FAM_SCALE, t14.SHAPE_STEP, 3.0, 0.5), t14.Change(t14.FAM_CLOCK, t14.SHAPE_STEP, 3.0, 0.5))


def test_analysis_without_calibration_is_descriptive_and_not_validated(cfg, tmp_path):
    inp = synthetic_inputs(STEP_MULTI)
    res = pl.run_analysis(inp, cfg, inherited_dir=tmp_path)
    assert res.status == pl.STATUS_NO_CALIBRATION and res.decisions is None and len(res.dates) == 0
    assert set(res.th29.loc[res.th29["familia"] != t14.FAM_ACTIVITY, "veredicto"]) == {pl.STATUS_NO_CALIBRATION}
    assert res.th29.loc[res.th29["familia"] == t14.FAM_ACTIVITY, "veredicto"].item() == t14.VOLUME_VERDICT
    assert res.stop14["alcance"] == pl.STATUS_NO_CALIBRATION and not res.stop14["definitivo"]
    assert set(res.events["estado_fechas"]) == {"NO_PUBLICADAS (no validadas o sin calibracion)"}


def test_analysis_with_calibration_applies_validation_consequences(calibration, cfg, tmp_path):
    inp = synthetic_inputs(STEP_MULTI)
    res = pl.run_analysis(inp, cfg, calibration.thresholds, calibration.validation, inherited_dir=tmp_path)
    assert res.status.startswith("SOLO_DESCRIPTIVO") and len(res.dates) == 0          # validacion no aprobada: sin fechas
    assert set(res.th29.loc[res.th29["familia"] != t14.FAM_ACTIVITY, "validacion"]) == {t14.STATUS_NOT_VALIDATED}
    assert not res.stop14["definitivo"]
    tr = dict(zip(res.traceability["parametro"], res.traceability["valor"]))
    assert tr["umbrales_sha256"] == calibration.thresholds.sha256 and tr["manifiesto_validacion_sha256"] == calibration.validation.manifest_sha256


def test_published_dates_and_events_when_validation_passes(calibration, cfg, tmp_path):
    """Con una validacion aprobada (construida aqui solo para ejercitar el camino de publicacion), las fechas se publican y
    los eventos se describen por distancia, sin atribuir causalidad."""
    val = t14.ValidationResult(table=pd.DataFrame(), global_validated=True, dates_validated=True,
                               families_validated=dict.fromkeys(t14.DETECTOR_FAMILIES, True), stop14_scope_validated=True,
                               types_validated=True, power_ok=dict.fromkeys(t14.DETECTOR_FAMILIES, True),
                               thresholds_sha256=calibration.thresholds.sha256, manifest_sha256=calibration.validation.manifest_sha256)
    inp = synthetic_inputs(STEP_MULTI)
    res = pl.run_analysis(inp, cfg, calibration.thresholds, val, inherited_dir=tmp_path)
    assert res.status == t14.STATUS_COMPATIBLE and len(res.dates) >= 1
    ev = res.events.loc[res.events["en_muestra"]]
    assert len(ev) and ev["distancia_jornadas_a_intervalo"].notna().all()
    assert set(res.events["interpretacion"]) == {"coincidencia temporal descriptiva; no se atribuye causalidad"}
    # STOP-14 sin comprobaciones anual/sensibilidad explicitas: nunca definitivo si hay candidatas.
    if res.stop14.get("familias_pendientes"):
        assert res.stop14["alcance"] == t14.STOP_PENDING and not res.stop14["definitivo"]


def _passing_validation(calibration):
    return t14.ValidationResult(table=pd.DataFrame(), global_validated=True, dates_validated=True,
                                families_validated=dict.fromkeys(t14.DETECTOR_FAMILIES, True), stop14_scope_validated=True,
                                types_validated=True, power_ok=dict.fromkeys(t14.DETECTOR_FAMILIES, True),
                                thresholds_sha256=calibration.thresholds.sha256, manifest_sha256=calibration.validation.manifest_sha256)


def test_stop14_checks_are_computed_by_rules_A_and_B(calibration, cfg, tmp_path):
    """Las comprobaciones de STOP-14 salen de ejecutar las reglas A y B sobre este analisis (no de la configuracion)."""
    from ohlcv_dataroad.ingest import tda14_stop14_rules as rules
    inp = synthetic_inputs(STEP_MULTI)
    res = pl.run_analysis(inp, cfg, calibration.thresholds, _passing_validation(calibration), inherited_dir=tmp_path)
    assert res.stop14_checks_status == f"EJECUTADAS ({rules.RULES_VERSION})"
    ev_a, ev_b = res.stop14_annual_evidence, res.stop14_sensitivity_evidence
    assert set(ev_a["familia"]) == set(t14.DETECTOR_FAMILIES) and len(ev_a) == len(t14.DETECTOR_STATISTICS)
    assert set(ev_b["sensibilidad"]) == set(rules.SENSITIVITIES) and len(ev_b) == 3 * len(t14.DETECTOR_FAMILIES)
    # las decisiones finales usan exactamente los resultados de las reglas (reejecucion determinista)
    chk = rules.compute_stop14_checks(inp.panel, res.stats, res.annual, calibration.thresholds)
    expected = {f: t14.persistence_status(res.decisions["veredictos_familia"][f], res.decisions["veredictos_familia_estricta"][f],
                                          chk.annual[f], chk.sensitivity[f]) for f in t14.DETECTOR_FAMILIES}
    assert res.stop14["estado_por_familia"] == expected
    assert all(v in (True, False, None) for v in list(chk.annual.values()) + list(chk.sensitivity.values()))
    tr = dict(zip(res.traceability["parametro"], res.traceability["valor"]))
    assert tr["stop14_comprobaciones"].startswith("EJECUTADAS")


def test_manual_stop14_checks_in_configuration_are_rejected(calibration, cfg, tmp_path):
    """Ninguna comprobacion puede declararse superada sin ejecutarla: valores manuales -> error, sin analisis parcial."""
    inp = synthetic_inputs(STEP_MULTI)
    for key, value in (("confirmacion_anual", {"ESCALA": True}), ("sensibilidad_rolls_huecos", {"RELOJ": False})):
        raw = copy.deepcopy(cfg.raw)
        raw["stop14"][key] = value
        with pytest.raises(pl.TDA14ConfigError, match="no se aceptan comprobaciones manuales"):
            pl.run_analysis(inp, replace(cfg, raw=raw), calibration.thresholds, _passing_validation(calibration), inherited_dir=tmp_path)
    raw = copy.deepcopy(cfg.raw)
    raw["stop14"]["reglas_version"] = "OTRA"
    with pytest.raises(pl.TDA14ConfigError, match="reglas"):
        pl.run_analysis(inp, replace(cfg, raw=raw), calibration.thresholds, _passing_validation(calibration), inherited_dir=tmp_path)


def test_without_thresholds_stop14_checks_are_not_executed(cfg, tmp_path):
    res = pl.run_analysis(synthetic_inputs(), cfg, inherited_dir=tmp_path)
    assert res.stop14_checks_status == "NO_EJECUTADAS_SIN_UMBRALES" and res.stop14_annual_evidence.empty


def test_family_without_power_controls_can_never_be_stable(calibration, cfg, tmp_path):
    """DEPENDENCIA_MAGNITUD sin controles de potencia en el manifiesto: sigue en el detector, pero nunca ESTABLE."""
    man = calibration.manifest
    assert all(f != t14.FAM_MAGNITUDE for f, _ in man.required_power)
    assert calibration.validation.power_ok[t14.FAM_MAGNITUDE] is False
    raw = copy.deepcopy(cfg.raw)
    assert all(p["familia"] != t14.FAM_MAGNITUDE for p in raw["protocolo"]["potencia"])
    val = replace(_passing_validation(calibration), power_ok={**dict.fromkeys(t14.DETECTOR_FAMILIES, True), t14.FAM_MAGNITUDE: False})
    for rep in range(3):
        res = pl.run_analysis(synthetic_inputs(rep=rep), cfg, calibration.thresholds, val, inherited_dir=tmp_path)
        v = res.decisions["veredictos_familia"][t14.FAM_MAGNITUDE]
        assert v["veredicto"] != t14.V_STABLE
        assert t14.FAM_MAGNITUDE in t14.DETECTOR_FAMILIES and "persistencia_acf_abs" in res.stats.values


def test_clock_window_coverage_before_and_after_event(cfg):
    inp = synthetic_inputs()
    clock = (np.arange(inp.panel.present.shape[1]) + t14.SESSION_OPEN_MINUTE_NY) % 1440
    cols = np.flatnonzero((clock >= 19 * 60) & (clock < 19 * 60 + 15))
    before = pd.to_datetime(inp.panel.session_date) < pd.Timestamp("2022-06-01")
    inp.panel.present[np.ix_(before, cols)] = False
    inp.panel.r[np.ix_(before, cols)] = np.nan
    cov = pl.clock_window_coverage(inp.panel, cfg.raw["eventos"])
    cov = cov.loc[cov["evento"] == "EVENTO_SINTETICO"]
    assert (cov.loc[cov["periodo"] == "antes_evento", "jornadas_con_barras"] == 0).all()
    assert (cov.loc[cov["periodo"] == "desde_evento", "fraccion_jornadas_con_barras"] > 0.9).all()
    assert set(cov["interpretacion"]) == {"descriptivo; no demuestra causalidad"}


def test_th30_in_pipeline_never_claims_stationarity(cfg, tmp_path):
    res = pl.run_analysis(synthetic_inputs(), cfg, inherited_dir=tmp_path)
    assert set(res.th30_summary["jornadas_subventana"]) == set(t14.TH30_SUB_SESSIONS)
    assert set(res.th30["serie"]) == {"LN_C", "R"}
    assert all("no demuestra estacionariedad" in n for n in res.th30_summary["nota"])


def test_report_and_artifacts_are_written(calibration, cfg, tmp_path):
    inp = synthetic_inputs(STEP_MULTI)
    write_inherited(inp.panel, tmp_path)
    res = pl.run_analysis(inp, cfg, calibration.thresholds, calibration.validation, inherited_dir=tmp_path)
    paths = run_tda14.persist_analysis(res, cfg, tmp_path / "salida", calibration.thresholds, 1.0, "test")
    names = {p.name for p in paths}
    a = cfg.artifacts
    for key in ("informe", "anual_csv", "anio_segmento_csv", "ventanas_csv", "reconciliacion_csv", "th29_csv", "th31_fechas_csv",
                "eventos_csv", "cobertura_csv", "th30_resumen_csv", "stop14_csv", "trazabilidad_csv", "fig_ventanas", "fig_anual", "fig_recuento"):
        assert a[key] in names, key
    assert a["validacion_csv"] not in names                       # el analisis no reescribe artefactos de la calibracion
    text = (tmp_path / "salida" / a["informe"]).read_text(encoding="utf-8")
    ws, step = res.stats.window_sessions, res.stats.window_sessions // res.stats.offset
    assert (ws, step) == (DETECTOR["batch_sessions"] * DETECTOR["window_batches"], DETECTOR["batch_sessions"] * DETECTOR["step_batches"])
    assert f"({ws} jornadas, paso {step})" in text and "63 jornadas, paso 21" not in text
    for needle in ("## 3. Reconciliacion", "## 5. TH29", "## 6. TH31", "## 7. TH30", "## 8. STOP-14", "## 9. Controles G2",
                   "## 10. Trazabilidad", "no una garantia", "no demuestra estacionariedad", "no se atribuye causalidad", res.status):
        assert needle in text, needle
    assert "estacionariedad demostrada" not in text.lower() and "garantiza" not in text.lower()


def test_analysis_is_reproducible(cfg, tmp_path):
    inp = synthetic_inputs(STEP_MULTI)
    a = pl.run_analysis(inp, cfg, inherited_dir=tmp_path)
    b = pl.run_analysis(inp, cfg, inherited_dir=tmp_path)
    for x, y in ((a.annual, b.annual), (a.year_segment, b.year_segment), (a.windows, b.windows), (a.reconciliation, b.reconciliation)):
        pd.testing.assert_frame_equal(x, y)


def test_analysis_rejects_thresholds_from_another_detector(calibration, cfg, tmp_path):
    raw = copy.deepcopy(cfg.raw)
    raw["protocolo"]["detector"]["max_lag"] = 12
    with pytest.raises(pl.TDA14ConfigError, match="detector distinto"):
        pl.run_analysis(synthetic_inputs(), replace(cfg, raw=raw), calibration.thresholds, calibration.validation, inherited_dir=tmp_path)


# ===========================================================================
# Cierre: lineas del generador dependientes de la calibracion y enmienda del informe sin recalcular
# ===========================================================================

def _closure(tmp_path: Path, original: str = "0" * 64, closed: str = "1" * 64) -> dict:
    p = tmp_path / "cierre.yaml"
    p.write_text(yaml.safe_dump({"informe": {"sha256_original": original, "sha256_cerrado": closed}, "estado": "CERRADA",
                                 "detalle": "prueba", "auditoria": "auditoria sintetica", "conclusiones_validas": ["conclusion A"],
                                 "restricciones_heredadas": ["restriccion B"]}, allow_unicode=True), encoding="utf-8")
    return run_tda14.load_closure(p)


def test_report_lines_follow_calibration_and_protocol(calibration, cfg, tmp_path):
    no_thr = run_tda14.render_report(pl.run_analysis(synthetic_inputs(), cfg, inherited_dir=tmp_path), cfg, 1.0, "test")
    assert run_tda14.LEGACY_SCOPE_LINE in no_thr and run_tda14.LEGACY_POWER_LINE not in no_thr
    res = pl.run_analysis(synthetic_inputs(), cfg, calibration.thresholds, calibration.validation, inherited_dir=tmp_path)
    text = run_tda14.render_report(res, cfg, 1.0, "test")
    assert run_tda14.LEGACY_SCOPE_LINE not in text and run_tda14.thresholds_scope_line(True) in text
    without = [f for f in t14.DETECTOR_FAMILIES if f != t14.FAM_SCALE]       # la configuracion sintetica solo tiene potencia de ESCALA
    assert f"Familias sin controles de potencia en el protocolo: {', '.join(without)}." in text
    real = pl.load_tda14_config(REAL_CONFIG)
    assert "Familias sin controles de potencia en el protocolo: DEPENDENCIA_MAGNITUD." in run_tda14.power_limitation_line(real)
    assert run_tda14.CLOSURE_HEADING not in text                             # sin archivo de cierre, nada de cierre


def test_amend_report_changes_only_the_two_lines_and_adds_closure(calibration, cfg, tmp_path):
    res = pl.run_analysis(synthetic_inputs(), cfg, calibration.thresholds, calibration.validation, inherited_dir=tmp_path)
    current = run_tda14.render_report(res, cfg, 1.0, "test")
    legacy = current.replace(run_tda14.thresholds_scope_line(True), run_tda14.LEGACY_SCOPE_LINE).replace(
        run_tda14.power_limitation_line(cfg), run_tda14.LEGACY_POWER_LINE)              # lo que escribia el generador anterior
    probe = _closure(tmp_path)
    expected = current + run_tda14.closure_section(probe)
    assert run_tda14.amended_report_text(legacy, cfg, probe) == expected
    assert run_tda14.render_report(res, cfg, 1.0, "test", probe).endswith(run_tda14.closure_section(probe))
    report = tmp_path / "informe.md"
    report.write_text(legacy, encoding="utf-8")                              # CRLF en Windows: el formato se conserva
    raw = report.read_bytes()
    nl = "\r\n" if b"\r\n" in raw else "\n"
    closed = expected.replace("\n", nl).encode("utf-8")
    sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
    with pytest.raises(RuntimeError, match="no coincide"):                   # cerrado mal declarado: no escribe
        run_tda14.amend_report(report, cfg, {**probe, "informe": {"sha256_original": sha(raw), "sha256_cerrado": "f" * 64}})
    assert report.read_bytes() == raw
    closure = {**probe, "informe": {"sha256_original": sha(raw), "sha256_cerrado": sha(closed)}}
    assert run_tda14.amend_report(report, cfg, closure) is True and report.read_bytes() == closed
    assert run_tda14.amend_report(report, cfg, closure) is False and report.read_bytes() == closed   # idempotente
    report.write_bytes(raw + b" ")
    with pytest.raises(RuntimeError, match="ni el original ni el cerrado"):
        run_tda14.amend_report(report, cfg, closure)
    assert report.read_bytes() == raw + b" "
    with pytest.raises(ValueError, match="exactamente una vez"):
        run_tda14.amended_report_text(current, cfg, probe)                  # sin lineas antiguas: no hay nada que enmendar


def test_amend_cli_reads_no_data_and_requires_closure(calibration, cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "load_tda14_inputs", lambda *_: (_ for _ in ()).throw(AssertionError("no debe leer datos")))
    monkeypatch.setattr(pl, "run_analysis", lambda *_, **__: (_ for _ in ()).throw(AssertionError("no debe analizar")))
    with pytest.raises(SystemExit):
        run_tda14.main(["enmendar-informe", "--salida", str(tmp_path)])
    p = tmp_path / "cierre.yaml"
    p.write_text(yaml.safe_dump({"estado": "CERRADA", "informe": {"sha256_original": "0" * 64, "sha256_cerrado": "1" * 64}}), encoding="utf-8")
    (tmp_path / cfg.artifacts["informe"]).write_text("otro informe", encoding="utf-8")
    assert run_tda14.main(["enmendar-informe", "--closure", str(p), "--salida", str(tmp_path)]) == 2
    assert (tmp_path / cfg.artifacts["informe"]).read_text(encoding="utf-8") == "otro informe"


def test_real_closure_file_is_documental_and_complete():
    closure = run_tda14.load_closure(REAL_CONFIG.parent / "tda14_cierre.yaml")
    assert closure["estado"] == "CERRADA TRAS AUDITORÍA"
    h = closure["informe"]
    assert all(len(h[k]) == 64 and h[k] != "PENDIENTE" for k in ("sha256_original", "sha256_cerrado")) and h["sha256_original"] != h["sha256_cerrado"]
    for key, _ in run_tda14.CLOSURE_SECTIONS:
        assert closure.get(key), key
    text = " ".join(str(x) for k, _ in run_tda14.CLOSURE_SECTIONS for x in closure[k])
    for needle in ("PARCIAL", "RELOJ", "S1", "ruptura estructural", "tipos", "DEPENDENCIA_MAGNITUD", "holdout",
                   "350b4eac372c3532df81305e110cd245f43c99b7b5d9d11e6b48f51808ca9522", "9e7a891127fecb6e166f32b2169fd8b4075f1e87b93b2634300510b906a8fa5e"):
        assert needle in text, needle
    assert not ({"protocolo", "analisis", "calibracion", "umbrales"} & set(closure))          # no lleva parametros estadisticos
    assert pl.load_tda14_config(REAL_CONFIG).raw["analisis"]["autorizado_mnq"] is False
