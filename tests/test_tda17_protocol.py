"""Tests del protocolo de TDA-17 (configuracion, recuentos, coherencia e integridad).

No calculan ningun indicador ni ejecutan analisis sobre MNQ. Leen `configs/tda17.yaml`,
`configs/mnq_snapshot.yaml`, el manifiesto, el informe del protocolo y artefactos historicos
publicados de TDA-02, TDA-06 y TDA-07 (CSV/MD). Los tests de integridad calculan SHA-256 de ficheros
(bytes) para comprobar que nada cambio; nunca abren data/raw/ ni el hold-out.

Modos de integridad (variable de entorno TDA17_INTEGRIDAD):
- `portable` (por defecto): los ficheros ausentes se omiten con SKIP "MODO PORTABLE"; NO vale para
  congelar ni para autorizar la ejecucion.
- `estricta`: cualquier fichero ausente o distinto es un FALLO. Es el modo de la verificacion oficial
  (`python -m ohlcv_dataroad.ingest.tda17_protocol_freeze verificar`, siempre estricta).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import integrate, optimize, stats

from ohlcv_dataroad.ingest import tda17_protocol_freeze as frz

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "tda17.yaml"
SNAPSHOT = ROOT / "configs" / "mnq_snapshot.yaml"
MANIFEST = ROOT / "reports" / "mnq" / "TDA17_manifiesto_protocolo.json"
REPORT = ROOT / "reports" / "mnq" / "TDA17_protocolo_congelamiento.md"
SEGMENTATION = ROOT / "reports" / "mnq" / "TDA06_segmentacion_propuesta.csv"

INTEGRITY_MODE = os.environ.get("TDA17_INTEGRIDAD", "portable").strip().lower()
if INTEGRITY_MODE not in {"portable", "estricta"}:
    raise RuntimeError(f"TDA17_INTEGRIDAD debe ser 'portable' o 'estricta', no {INTEGRITY_MODE!r}")
STRICT = INTEGRITY_MODE == "estricta"
STATES = {"PENDIENTE_AUDITORIA_FINAL", "CONGELADO"}

PANELS = {"P", "L", "Aux"}
TRANSFORMS = {"ninguna", "log"}


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _series(cfg: dict) -> list[dict]:
    """Enumera las series del catalogo: una por (configuracion, salida) mas las auxiliares de elemento."""
    out = []
    for el in cfg["catalogo"]:
        for c in el["configuraciones"]:
            for s in el["salidas"]:
                out.append({"id": f"{el['id']}_{c['id']}__{s['id']}", "elemento": el["id"], "panel": s["panel"],
                            "transformacion": s.get("transformacion")})
        for s in el.get("salidas_auxiliares_elemento", []):
            out.append({"id": f"{el['id']}__{s['id']}", "elemento": el["id"], "panel": s["panel"], "transformacion": None})
    return out


def _p_columns(cfg: dict) -> list[str]:
    cols = [s["id"] for s in _series(cfg) if s["panel"] == "P"]
    return cols + [r["id"] for r in cfg["referencias_tda04"]["en_panel_P"]]


TEXT_SUFFIXES = {".md", ".csv", ".yaml", ".yml", ".json", ".py", ".txt"}


def _sha256(path: Path) -> str:
    """SHA-256 del contenido. En ficheros de texto se normaliza CRLF -> LF para que el hash
    no dependa de core.autocrlf (el repositorio usa `* text=auto`); los binarios se hashean tal cual."""
    if path.suffix.lower() in TEXT_SUFFIXES:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_hash_is_line_ending_independent_for_text_files(tmp_path):
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_bytes(b"x: 1\ny: 2\n")
    b.write_bytes(b"x: 1\r\ny: 2\r\n")
    assert _sha256(a) == _sha256(b)
    assert frz.sha256_file(a) == frz.sha256_file(b) == _sha256(a)      # dos implementaciones independientes


# --------------------------------------------------------------------------- estado y gobierno
def test_protocol_state_and_governance(cfg, manifest):
    p = cfg["protocolo"]
    assert p["estado"] in STATES
    assert manifest["estado"] == p["estado"]
    assert frz.report_state(REPORT.read_bytes()) == p["estado"]
    if p["estado"] == "PENDIENTE_AUDITORIA_FINAL":
        assert "congelamiento" not in manifest
    else:                                         # CONGELADO solo con registro y reversibilidad exacta
        assert frz.verify_freeze_record(ROOT, manifest) == []
    assert p["estatus_g2"] == "EXPLORATORIO_PREDECLARADO"
    assert p["decision_iris"] == "NINGUNA"
    assert p["holdout"]["usar_en_tda17"] is False
    assert p["holdout"]["confirmacion_futura_programada"] is False
    assert p["dependencias_estado"] == {"TDA-15": "CERRADA_STOP_15A", "TDA-16": "NO_EJECUTADA"}


def test_holdout_boundary_matches_snapshot_and_no_holdout_file_is_referenced(cfg):
    snap = yaml.safe_load(SNAPSHOT.read_text(encoding="utf-8"))
    assert cfg["protocolo"]["holdout"]["boundary_utc"] == snap["holdout"]["boundary_utc"]
    text = PROTOCOL.read_text(encoding="utf-8")
    for f in snap["holdout"]["holdout_files"]:
        assert f not in text
    assert cfg["entradas"]["abrir_data_raw"] is False
    for key in ("variables", "validez"):
        assert cfg["entradas"][key].startswith("data/interim/mnq/tda04_")
    assert "data/raw" not in text


# --------------------------------------------------------------------------- recuentos
def test_element_and_configuration_counts(cfg):
    cat = cfg["catalogo"]
    rec = cfg["recuentos"]
    assert len(cat) == rec["elementos"] == 45
    assert len({e["id"] for e in cat}) == 45
    fam = pd.Series([e["familia"] for e in cat]).value_counts().to_dict()
    assert fam == rec["por_familia"] == {"tendencia": 12, "momentum": 10, "volatilidad": 10, "volumen": 8, "velas": 5}
    assert sum(len(e["configuraciones"]) for e in cat) == rec["configuraciones"] == 102


def test_series_counts_and_panels(cfg):
    ser = _series(cfg)
    rec = cfg["recuentos"]
    ids = [s["id"] for s in ser]
    assert len(ids) == len(set(ids))
    assert {s["panel"] for s in ser} <= PANELS
    n = {pnl: sum(s["panel"] == pnl for s in ser) for pnl in PANELS}
    assert n["P"] == rec["series_P"] == 128
    assert n["L"] == rec["series_L"] == 16
    assert n["Aux"] == rec["series_Aux"] == 2
    assert n["P"] + n["L"] == rec["salidas_calculadas"] == 144
    assert len(ser) == rec["series_nuevas"] == 146
    assert len(cfg["referencias_tda04"]["en_panel_P"]) == rec["referencias_tda04_en_P"] == 6
    assert len(_p_columns(cfg)) == rec["columnas_P"] == 134
    assert {s["transformacion"] for s in ser if s["panel"] != "Aux"} <= TRANSFORMS


def test_element_panel_status_41_3_1(cfg):
    ser = _series(cfg)
    by_el: dict[str, set] = {}
    for s in ser:
        by_el.setdefault(s["elemento"], set()).add(s["panel"])
    with_p = sorted(e for e, p in by_el.items() if "P" in p)
    only_l = sorted(e for e, p in by_el.items() if p == {"L"})
    only_aux = sorted(e for e, p in by_el.items() if p == {"Aux"})
    rec = cfg["recuentos"]
    assert (len(with_p), len(only_l), len(only_aux)) == (rec["elementos_con_P"], rec["elementos_solo_L"], rec["elementos_solo_Aux"]) == (41, 3, 1)
    assert only_l == sorted(cfg["panel_L"]["elementos_solo_L"]) == ["B2", "C1", "E5"]
    assert only_aux == sorted(cfg["panel_L"]["elementos_solo_Aux"]) == ["D4"]
    assert cfg["panel_L"]["en_salidas_obligatorias"] is False


# --------------------------------------------------------------------------- identidades y rango
def _identities(cfg, panel):
    return [i for i in cfg["identidades"]["lineales"] if i["panel"] == panel]


def test_linear_identities_reference_existing_series_in_their_panel(cfg):
    ser = {s["id"]: s["panel"] for s in _series(cfg)}
    refs = {r["id"] for r in cfg["referencias_tda04"]["en_panel_P"]}
    for ident in cfg["identidades"]["lineales"]:
        for _, col in ident["terminos"]:
            panel = "P" if col in refs else ser[col]
            assert panel == ident["panel"], (ident["id"], col)


def test_identity_counts_types_and_involved_columns(cfg):
    p_ids, l_ids = _identities(cfg, "P"), _identities(cfg, "L")
    r = cfg["identidades"]["rango"]
    hom = [i for i in p_ids if i["constante"] == 0]
    aff = [i for i in p_ids if i["constante"] != 0]
    assert (len(p_ids), len(l_ids)) == (cfg["recuentos"]["identidades_lineales_P"], cfg["recuentos"]["identidades_lineales_L"]) == (11, 3)
    assert (len(hom), len(aff)) == (r["P_dependencias_homogeneas"], r["P_dependencias_afines"]) == (9, 2)
    involved = {col for i in p_ids for _, col in i["terminos"]}
    assert len(involved) == r["columnas_implicadas_P"] == 32
    out_b = [i["fuera_de_B"] for i in p_ids]
    assert len(set(out_b)) == 11
    for i in p_ids:   # cada identidad tiene exactamente una columna fuera de B
        assert sum(col in out_b for _, col in i["terminos"]) == 1
    assert len(_p_columns(cfg)) - len(out_b) == r["B_columnas"] == cfg["recuentos"]["columnas_B"] == 123


def test_declared_rank_bounds_match_algebra(cfg):
    r = cfg["identidades"]["rango"]
    n_hom, n_aff, p = r["P_dependencias_homogeneas"], r["P_dependencias_afines"], r["P_columnas"]
    uncentered_deps = n_hom + max(n_aff - 1, 0)
    assert p - uncentered_deps == r["P_sin_centrar_max"] == 124
    assert p + 1 - (n_hom + n_aff) == r["P_con_intercepto_max"] == 124
    assert p - (n_hom + n_aff) == r["P_centrada_max"] == 123
    assert r["L_columnas"] - len(_identities(cfg, "L")) == r["L_centrada_max"] == 13
    nul = cfg["identidades"]["verificacion_numerica"]["nulos_esperados"]
    assert nul == {"P_sin_centrar": p - r["P_sin_centrar_max"], "P_centrada": p - r["P_centrada_max"], "B_centrada": 0, "L": 3}


def test_identity_registry_produces_declared_numeric_ranks_on_synthetic_columns(cfg):
    """Columnas de B aleatorias (sinteticas, no MNQ); las 11 retiradas se construyen con su identidad."""
    rng = np.random.default_rng(17)
    cols = _p_columns(cfg)
    p_ids = _identities(cfg, "P")
    out_b = {i["fuera_de_B"]: i for i in p_ids}
    n = 600
    data = {c: rng.standard_normal(n) for c in cols if c not in out_b}
    for c, ident in out_b.items():
        coef_c = dict((col, k) for k, col in ident["terminos"])[c]
        rest = sum(k * data[col] for k, col in ident["terminos"] if col != c)
        data[c] = (ident["constante"] - rest) / coef_c
    X = np.column_stack([data[c] for c in cols])
    tol = 1e-10
    def rank(M):
        s = np.linalg.svd(M, compute_uv=False)
        return int(np.sum(s > tol * s[0]))
    r = cfg["identidades"]["rango"]
    assert rank(X) == r["P_sin_centrar_max"]
    assert rank(np.column_stack([np.ones(n), X])) == r["P_con_intercepto_max"]
    assert rank(X - X.mean(0)) == r["P_centrada_max"]
    B = np.column_stack([data[c] for c in cols if c not in out_b])
    assert rank(B - B.mean(0)) == r["B_columnas"]


def test_nonlinear_identity_registry_present(cfg):
    ids = [i["id"] for i in cfg["identidades"]["no_lineales_o_fuera_de_P"]]
    assert ids == [f"N{k}" for k in range(1, 22)]


def test_N10_is_a_non_universal_diagnostic_never_required(cfg):
    n10 = next(i for i in cfg["identidades"]["no_lineales_o_fuera_de_P"] if i["id"] == "N10")
    assert n10["tipo"] == "DIAGNOSTICO_NO_UNIVERSAL" and "NO identidad" in n10["relacion"]
    lineales = {col for i in cfg["identidades"]["lineales"] for _, col in i["terminos"]}
    assert not {c for c in lineales if c.startswith("A11_")}             # Supertrend fuera de las 11 + 3 lineales
    v1 = cfg["g2"]["criterios"]["V1"]
    assert "N10" in v1["no_exige"] and "N10" not in v1["exige"]
    a11 = next(e for e in cfg["catalogo"] if e["id"] == "A11")
    assert "NO es una identidad" in a11["convencion"]["relacion_N10"]


# --------------------------------------------------------------------------- calentamiento
def _k(alpha: float) -> int:
    return math.ceil(math.log(0.01) / math.log(1 - alpha))


def _ema(n, start=1):
    return start + n - 1 + _k(2 / (n + 1))


def _wilder(n, start=1):
    return start + n - 1 + _k(1 / n)


def _expected_K(el: str, p: dict) -> int | None:
    if el in {"A1", "A3", "A12", "C3", "C5", "C6", "C7", "C8", "C9", "D3", "B5", "B6", "C10"}:
        return p["n"]
    if el == "A2":
        return _ema(p["n"])
    if el == "A4":
        return p["n"] + p["n_raiz"] - 1
    if el in {"A6", "A7"}:
        k1 = max(_ema(p["f"]), _ema(p["s"]))
        return _ema(p["g"], start=k1)
    if el == "A8":
        k1 = _wilder(p["n"], start=2)
        return _wilder(p["n"], start=k1)
    if el == "A9":
        return p["n"] + 1
    if el == "A10":
        k = _ema(p["n"])
        k = _ema(p["n"], start=k)
        k = _ema(p["n"], start=k)
        return k + 1
    if el in {"B1", "B2", "B7", "D1", "D2", "D6"}:
        return p["n"] + 1
    if el == "B3":
        return _wilder(p["n"], start=2)
    if el == "B4":
        return p["n"] + (p["k"] - 1) + (p["d"] - 1)
    if el == "B8":
        return _ema(p["s"], start=_ema(p["r"], start=2))
    if el == "B9":
        return _wilder(p["rsi"], start=2) + (p["n"] - 1) + (p["k"] - 1) + (p["d"] - 1)
    if el == "B10":
        return p["c"]
    if el in {"C1", "C2", "A11"}:                # A11: K = primer ATR calentado (frontera de publicacion)
        return _wilder(p["n"])
    if el == "C4":
        return max(_ema(p["n"]), _wilder(p["atr"]))
    if el == "D5":
        return max(_ema(p["f"]), _ema(p["s"]))
    return None


def test_declared_warmups_follow_the_convention(cfg):
    checked = 0
    for el in cfg["catalogo"]:
        for c in el["configuraciones"]:
            exp = _expected_K(el["id"], c["params"])
            if exp is not None:
                assert c["K"] == exp, (el["id"], c["id"], c["K"], exp)
                checked += 1
    assert checked >= 85
    ks = [c["K"] for e in cfg["catalogo"] for c in e["configuraciones"] if "K" in c]
    assert max(ks) == cfg["calentamiento"]["K_maximo"] == 1343


def test_data_dependent_warmups_are_declared(cfg):
    dd = {(e["id"], c["id"]) for e in cfg["catalogo"] for c in e["configuraciones"] if c.get("K_data_dependiente")}
    assert dd == {("A5", "er10"), ("A5", "er60")}
    rv = {c["id"]: c["K_jornadas"] for e in cfg["catalogo"] if e["id"] == "D7" for c in e["configuraciones"]}
    assert rv == {"k5": 5, "k20": 20}


# --------------------------------------------------------------------------- decisiones y reglas
def test_decisions_d1_to_d24(cfg):
    d = cfg["decisiones"]
    assert list(d) == [f"D{k}" for k in range(1, 25)]
    explicit = {k for k, v in d.items() if v["estado"] == "APROBADA_EXPLICITAMENTE"}
    assert explicit == {"D14", "D22", "D23", "D24"}
    assert all(v["estado"] in {"RESUELTA", "APROBADA_EXPLICITAMENTE"} for v in d.values())
    assert "EXPLORATORIO_PREDECLARADO" in d["D14"]["contenido"]


def test_choppiness_is_c2_without_variants(cfg):
    chop = next(e for e in cfg["catalogo"] if e["id"] == "C10")
    assert chop["nombre"] == "Choppiness_C2"
    assert len(chop["salidas"]) == 1 and len(chop["configuraciones"]) == 2
    assert "HH_n - LL_n" in chop["formula"] and "sin variantes" in chop["formula"]
    assert "chop_fuera_rango" in cfg["reglas_frontera"]["marcas_por_fila"]


def test_rvol_definition_and_policy(cfg):
    rv = next(e for e in cfg["catalogo"] if e["id"] == "D7")
    assert "inmediatamente anteriores" in rv["formula"] and "sin sustitucion" in rv["formula"]
    assert cfg["relative_volume_politica"]["chequeo_comparabilidad"]["consecuencia_metodologica"] == "NINGUNA"
    assert cfg["relative_volume_politica"]["sensibilidad"] == "S7"


def test_segments_match_tda06_stable_cuts(cfg):
    seg = pd.read_csv(SEGMENTATION)
    cuts = sorted(int(m) for m in seg.loc[seg["stable"], "minute_of_day"])
    s = cfg["analisis"]["celdas"]["segmentos"]
    assert cuts == s["cortes_min"]
    edges = [0] + cuts + [1440]
    labels = [f"{a // 60:02d}:{a % 60:02d}-{b // 60:02d}:{b % 60:02d}" for a, b in zip(edges[:-1], edges[1:])]
    assert labels == s["etiquetas"]


def test_g2_controls_and_criteria(cfg):
    g = cfg["g2"]
    assert set(g["generativos"]) - {"criterio"} == {"G2_1a", "G2_1b", "G2_2"}
    assert g["criterios"]["V1"]["aplica_a"] == ["G2_1a", "G2_1b", "G2_2"]
    assert g["criterios"]["V1_prima"]["aplica_a"] == ["G2_3a", "G2_3b"]
    a = g["desplazamientos"]["G2_3a"]
    assert (a["k_min"], a["desfases_max"], a["desfases_min"], a["omega_min_fraccion"], a["muestreo"]) == (20, 100, 40, 0.5, "SIN_REEMPLAZO")
    b = g["desplazamientos"]["G2_3b"]
    assert b["p"] == cfg["recuentos"]["columnas_B"] and b["garantia"] == "POR_PARES" and b["matriz"] == "B"
    assert "imposible_si" in b["viabilidad_previa"] and "sin intentos" in b["viabilidad_previa"]["accion"]
    v3 = g["criterios"]["V3"]
    assert v3["replicas"] == b["replicas"] == 200
    assert v3["tolerancia"] == round(v3["nominal"] + 2 * math.sqrt(v3["nominal"] * (1 - v3["nominal"]) / v3["replicas"]), 3)
    g22 = g["generativos"]["G2_2"]
    tda07 = pd.read_csv(ROOT / "reports" / "mnq" / "TDA07_momentos_cuantiles_drift_colas.csv")
    row = tda07[(tda07["series"] == "r_1m") & (tda07["scope"] == "GLOBAL")].iloc[0]
    assert math.isclose(g22["sigma_r_1m"], row["std"], rel_tol=1e-12)
    assert math.isclose(g22["curtosis_exceso_recortada_objetivo"], row["kurtosis_excess_trimmed"], rel_tol=1e-12)


def _trimmed_kurtosis_t(nu: float, q: float) -> float:
    c = stats.t.ppf(1 - q, nu)
    m = [2 * integrate.quad(lambda x, j=j: x**j * stats.t.pdf(x, nu), 0, c, epsabs=0, epsrel=1e-13, limit=500)[0] / (1 - 2 * q)
         for j in (2, 4)]
    return m[1] / m[0] ** 2 - 3


def _trimmed_kurtosis_sample(x: np.ndarray, q: float) -> float:
    """Misma definicion que TDA-07 (compute_moments_quantiles)."""
    lo, hi = np.quantile(x, [q, 1 - q])
    e = x[(x >= lo) & (x <= hi)]
    e = e - e.mean()
    return float(np.mean(e**4) / np.mean(e**2) ** 2 - 3)


def test_g2_2_student_t_df_is_fixed_by_the_declared_procedure(cfg):
    g = cfg["g2"]["generativos"]["G2_2"]
    t = g["t_student"]
    target, q = g["curtosis_exceso_recortada_objetivo"], g["recorte_por_cola"]
    nu, res = optimize.brentq(lambda v: _trimmed_kurtosis_t(v, q) - target, 1.0, 1000.0,
                              xtol=1e-12, rtol=1e-14, maxiter=200, full_output=True)
    assert res.converged and nu > 2
    assert abs(nu - t["df"]) <= t["calibracion"]["tolerancia_recalculo"]
    assert g["semillas"] == [301, 302]


def test_g2_2_declared_seeds_pass_verification_on_synthetic_draws(cfg):
    """Solo extracciones t sinteticas con las semillas del protocolo (ni datos de MNQ ni resultados de TDA-17)."""
    g = cfg["g2"]["generativos"]["G2_2"]
    nu, sigma, q = g["t_student"]["df"], g["sigma_r_1m"], g["recorte_por_cola"]
    v = g["verificacion"]
    n = cfg["entradas"]["esperado"]["filas"]
    scale = sigma * math.sqrt((nu - 2) / nu)
    iqr_th = scale * (stats.t.ppf(0.75, nu) - stats.t.ppf(0.25, nu))
    for seed in g["semillas"]:
        stream = np.random.SeedSequence(seed).spawn(3)[0]
        r = scale * np.random.Generator(np.random.PCG64(stream)).standard_t(nu, n)
        kt = _trimmed_kurtosis_sample(r, q)
        iqr = (np.quantile(r, 0.75) - np.quantile(r, 0.25)) / iqr_th
        assert abs(kt - v["curtosis_recortada"]["objetivo"]) <= v["curtosis_recortada"]["tolerancia_abs"]
        assert abs(iqr - v["ric_relativo"]["objetivo"]) <= v["ric_relativo"]["tolerancia_abs"]
        exp = v["valores_esperados_semillas"][str(seed)]
        assert math.isclose(kt, exp["curtosis_recortada"], rel_tol=1e-9)
        assert math.isclose(iqr, exp["ric_relativo"], rel_tol=1e-9)


def test_s6_dates_are_the_closed_list_of_tda02_section_4_1(cfg):
    s6 = cfg["sensibilidades"]["S6"]
    assert s6["lista_cerrada"] is True
    for f in s6["fuentes"]:
        assert f.startswith("reports/mnq/TDA02_")
    text = (ROOT / "reports" / "mnq" / "TDA02_cobertura.md").read_text(encoding="utf-8")
    assert "concentrados en 2020-03-09, 16, 17 y 18" in text
    assert "Semana anómala del 18 al 22 de octubre de 2020" in text
    days = pd.read_csv(ROOT / "reports" / "mnq" / "TDA02_dias_incompletos.csv", dtype={"trading_date": str})
    march = ["2020-03-09", "2020-03-16", "2020-03-17", "2020-03-18"]
    october = days[(days["trading_date"] >= "2020-10-18") & (days["trading_date"] <= "2020-10-22")]
    assert list(october["trading_date"]) == ["2020-10-19", "2020-10-20", "2020-10-21", "2020-10-22"]
    assert (october["present"] == 1292).all() and (october["missing"] == 73).all()
    assert s6["fechas"] == march + list(october["trading_date"])
    sel = days[days["trading_date"].isin(s6["fechas"])]
    assert len(sel) == 8 and (sel["coverage_pct"] < 100).all() and not sel["is_early_close"].any()
    assert set(s6["no_incluidas"]) == {"2020-10-23"} and not set(s6["no_incluidas"]) & set(s6["fechas"])
    boundary = cfg["protocolo"]["holdout"]["boundary_utc"][:10]
    assert all(d < boundary for d in s6["fechas"])


def test_sensitivities_and_no_evaluable_rules(cfg):
    assert list(cfg["sensibilidades"]) == [f"S{k}" for k in range(1, 9)]
    ambitos = {r["ambito"] for r in cfg["reglas_no_evaluable"]}
    assert {"celda_anio_2019", "G2_3a", "k_PA", "R2", "EXACTA", "T3", "transferencia", "RVOL", "general"} <= ambitos
    assert sum(r["ambito"] == "k_PA" for r in cfg["reglas_no_evaluable"]) == 4


def test_analysis_settings_consistent(cfg):
    a = cfg["analisis"]
    assert a["r2"]["oficial"] == "CADA_COLUMNA_DE_P_FRENTE_A_LAS_OTRAS_133"
    assert a["pca"]["oficial"] == "P_COMPLETO" and a["pca"]["complementario"] == "B"
    assert a["pca"]["analisis_paralelo"]["matriz"] == "B"
    assert a["spearman"]["definicion"].startswith("Pearson entre rangos convencionales con rango medio")
    assert a["exactitud"]["residuo_max"] < a["exactitud"]["casi_exacta_max"]
    assert a["clasificacion"]["nunca"] == "NUEVA_INFORMACION"
    assert 2019 in a["celdas"]["anios"]["no_evaluables"]
    assert a["bootstrap"]["replicas"] == 300 and a["bootstrap"]["longitud_bloque_jornadas"] == 5


def test_report_mentions_state_and_counts():
    text = REPORT.read_text(encoding="utf-8")
    for token in ("PENDIENTE_AUDITORIA_FINAL", "CONGELADO", "EXPLORATORIO_PREDECLARADO", "DECISIÓN IRIS: NINGUNA", "1.343", "41 con salida en P",
                  "C2", "NO_EVALUABLE"):
        assert token in text


# --------------------------------------------------------------------------- integridad (manifiesto)
def test_manifest_state_and_counts_match_protocol(cfg, manifest):
    assert manifest["protocolo_id"] == cfg["protocolo"]["id"]
    assert manifest["estado"] == cfg["protocolo"]["estado"]
    assert manifest["base_git"] == cfg["protocolo"]["base_git"]
    assert manifest["recuentos"] == cfg["recuentos"]


def test_manifest_hashes_of_protocol_files(manifest):
    assert list(manifest["sha256_protocolo"]) == frz.PROTOCOL_FILES
    for rel, digest in manifest["sha256_protocolo"].items():
        assert _sha256(ROOT / rel) == digest, rel


def test_manifest_declares_exact_input_and_artifact_counts(cfg, manifest):
    assert len(manifest["sha256_protocolo"]) == len(frz.PROTOCOL_FILES) == cfg["integridad"]["ficheros_protocolo"] == 9
    assert len(manifest["sha256_entradas"]) == manifest["n_entradas"] == cfg["integridad"]["n_entradas"] == 5
    assert len(manifest["sha256_artefactos_previos"]) == manifest["n_artefactos_previos"] \
        == cfg["integridad"]["n_artefactos_previos"] == 188
    assert set(manifest["sha256_entradas"]) >= {cfg["entradas"]["variables"], cfg["entradas"]["validez"], cfg["entradas"]["segmentacion"]}


def _check_group(manifest, group):
    missing = []
    for rel, digest in manifest[group].items():
        path = ROOT / rel
        if not path.is_file():
            missing.append(rel)
            continue
        assert _sha256(path) == digest, rel
    if missing:
        if STRICT:
            pytest.fail(f"MODO ESTRICTO: faltan {len(missing)} ficheros de {group}: {missing[:10]}")
        pytest.skip(f"MODO PORTABLE (no valido para congelar): faltan {len(missing)} ficheros de {group}")


def test_manifest_hashes_of_inputs(manifest):
    _check_group(manifest, "sha256_entradas")


def test_previous_artifacts_unchanged(manifest):
    _check_group(manifest, "sha256_artefactos_previos")


def test_portable_and_strict_modes_differ_on_missing_files(monkeypatch, tmp_path):
    fake = {"sha256_entradas": {"no/existe.csv": "0" * 64}}
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    monkeypatch.setattr(sys.modules[__name__], "STRICT", False)
    with pytest.raises(pytest.skip.Exception, match="MODO PORTABLE"):
        _check_group(fake, "sha256_entradas")
    monkeypatch.setattr(sys.modules[__name__], "STRICT", True)
    with pytest.raises(pytest.fail.Exception, match="MODO ESTRICTO"):
        _check_group(fake, "sha256_entradas")


def test_official_strict_verification():
    if not STRICT:
        pytest.skip("MODO PORTABLE: la verificacion oficial solo se ejecuta con TDA17_INTEGRIDAD=estricta")
    assert frz.verify_strict(ROOT) == []


def test_synthetic_evidence_reads_no_data(cfg):
    """La evidencia sintetica versionada no lee ficheros de datos, MNQ ni holdout."""
    snap = yaml.safe_load(SNAPSHOT.read_text(encoding="utf-8"))
    for rel in cfg["integridad"]["evidencia_sintetica"]:
        text = (ROOT / rel).read_text(encoding="utf-8")
        for token in ("data/", "parquet", "read_csv", "read_parquet", "open(", "mnq_snapshot"):
            assert token not in text, (rel, token)
        for f in snap["holdout"]["holdout_files"]:
            assert f not in text
        assert rel in frz.PROTOCOL_FILES


def test_manifest_lists_no_holdout_or_raw_file(manifest):
    snap = yaml.safe_load(SNAPSHOT.read_text(encoding="utf-8"))
    names = " ".join(list(manifest["sha256_entradas"]) + list(manifest["sha256_artefactos_previos"]))
    assert "data/raw" not in names
    for f in snap["holdout"]["holdout_files"]:
        assert f not in names


# --------------------------------------------------------------------------- procedimiento de congelamiento (copia temporal)
@pytest.fixture()
def mini_repo(tmp_path, manifest):
    """Copia del protocolo en un directorio temporal, en PENDIENTE, con entradas y artefactos SINTETICOS.

    Reproduce la estructura real (5 entradas, 188 artefactos) con contenido falso de pocos bytes, para
    probar el procedimiento sin tocar el repositorio ni leer datos de MNQ."""
    for rel in frz.PROTOCOL_FILES + [frz.MANIFEST_JSON, frz.SNAPSHOT_YAML]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, tmp_path / rel)
    m = json.loads((tmp_path / frz.MANIFEST_JSON).read_text(encoding="utf-8"))
    if m["estado"] == "CONGELADO":            # reconstruye la version auditada deshaciendo las transiciones
        for rel, old, new in frz.TRANSICIONES:
            (tmp_path / rel).write_bytes(frz.replace_line((tmp_path / rel).read_bytes(), new, old))
            m["sha256_protocolo"][rel] = frz.sha256_file(tmp_path / rel)
        m["estado"] = "PENDIENTE_AUDITORIA_FINAL"
        m.pop("congelamiento")
    for group in ("sha256_entradas", "sha256_artefactos_previos"):
        for i, rel in enumerate(list(m[group])):
            if rel == frz.SNAPSHOT_YAML:      # copia real: la verificacion lee la lista de ficheros del holdout
                continue
            path = tmp_path / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"sintetico {group} {i}\n".encode())
            m[group][rel] = frz.sha256_file(path)
    (tmp_path / frz.MANIFEST_JSON).write_bytes(frz.manifest_bytes(m))
    assert frz.verify_strict(tmp_path) == []
    return tmp_path


def _snapshot(root):
    return {rel: (root / rel).read_bytes() for rel in (frz.PROTOCOL_YAML, frz.REPORT_MD, frz.MANIFEST_JSON)}


def test_freeze_end_to_end_changes_only_state_lines(mini_repo):
    before = _snapshot(mini_repo)
    m = frz.freeze(mini_repo, confirmar=True, comprobar_git=False, fecha_utc="2000-01-01T00:00:00Z")
    after = _snapshot(mini_repo)
    assert m["estado"] == "CONGELADO"
    assert yaml.safe_load(after[frz.PROTOCOL_YAML])["protocolo"]["estado"] == "CONGELADO"
    assert frz.report_state(after[frz.REPORT_MD]) == "CONGELADO"
    for rel, old, new in frz.TRANSICIONES:
        a, b = before[rel].split(b"\n"), after[rel].split(b"\n")
        diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        assert len(a) == len(b) and len(diff) == 1
        assert a[diff[0]].rstrip(b"\r") == old.encode() and b[diff[0]].rstrip(b"\r") == new.encode()
    assert frz.verify_strict(mini_repo) == []
    rec = m["congelamiento"]
    old_m = json.loads(before[frz.MANIFEST_JSON])
    assert list(rec["sha256_auditado"]) == frz.PROTOCOL_FILES                 # huellas auditadas de todos
    assert rec["sha256_auditado"] == old_m["sha256_protocolo"]
    state_files = {rel for rel, _, _ in frz.TRANSICIONES}
    unchanged = [r for r in frz.PROTOCOL_FILES if r not in state_files]
    assert len(unchanged) == len(frz.PROTOCOL_FILES) - 2 == 7
    assert all(m["sha256_protocolo"][r] == old_m["sha256_protocolo"][r] for r in unchanged)
    assert m["sha256_entradas"] == old_m["sha256_entradas"]
    assert m["sha256_artefactos_previos"] == old_m["sha256_artefactos_previos"]
    with pytest.raises(frz.ProtocolError):                     # no se puede congelar dos veces
        frz.freeze(mini_repo, confirmar=True, comprobar_git=False)
    with pytest.raises(frz.ProtocolError):                     # ni regenerar el manifiesto tras congelar
        frz.refresh_protocol_hashes(mini_repo)


def test_freeze_requires_explicit_confirmation(mini_repo):
    before = _snapshot(mini_repo)
    with pytest.raises(frz.ProtocolError):
        frz.freeze(mini_repo, confirmar=False, comprobar_git=False)
    assert _snapshot(mini_repo) == before


@pytest.mark.parametrize("group", ["sha256_entradas", "sha256_artefactos_previos"])
def test_strict_verification_and_freeze_fail_if_any_file_is_missing(mini_repo, group):
    m = frz.load_manifest(mini_repo)
    victim = list(m[group])[-1]
    (mini_repo / victim).unlink()
    errors = frz.verify_strict(mini_repo)
    assert any("FALTA" in e and victim in e for e in errors)
    before = _snapshot(mini_repo)
    with pytest.raises(frz.ProtocolError):
        frz.freeze(mini_repo, confirmar=True, comprobar_git=False)
    assert _snapshot(mini_repo) == before


def test_strict_verification_detects_modified_artifact(mini_repo):
    victim = list(frz.load_manifest(mini_repo)["sha256_artefactos_previos"])[0]
    (mini_repo / victim).write_bytes(b"modificado\n")
    assert any("HASH DISTINTO" in e and victim in e for e in frz.verify_strict(mini_repo))


def test_freeze_rolls_back_all_files_on_failure_after_writing(mini_repo):
    before = _snapshot(mini_repo)
    def boom():
        raise RuntimeError("fallo simulado tras la escritura")
    with pytest.raises(RuntimeError):
        frz.freeze(mini_repo, confirmar=True, comprobar_git=False, _hook_tras_escritura=boom)
    assert _snapshot(mini_repo) == before
    assert frz.verify_strict(mini_repo) == []
    assert not list(mini_repo.rglob("*.tda17tmp"))


def test_frozen_protocol_edited_outside_state_line_is_detected(mini_repo):
    frz.freeze(mini_repo, confirmar=True, comprobar_git=False)
    y = mini_repo / frz.PROTOCOL_YAML
    y.write_bytes(y.read_bytes().replace(b"tolerancia_abs: 0.85", b"tolerancia_abs: 0.95"))
    m = frz.load_manifest(mini_repo)                           # incluso si alguien rehace el hash del manifiesto
    m["sha256_protocolo"][frz.PROTOCOL_YAML] = frz.sha256_file(y)
    (mini_repo / frz.MANIFEST_JSON).write_bytes(frz.manifest_bytes(m))
    errors = frz.verify_strict(mini_repo)
    assert any("difiere de la version auditada" in e for e in errors)


@pytest.mark.parametrize("rel", frz.PROTOCOL_FILES)
def test_any_frozen_protocol_file_modified_and_rehashed_is_not_conforming(mini_repo, rel):
    """Regresion F1: tras congelar, modificar CUALQUIERA de los ficheros del protocolo (tests y herramienta incluidos)
    y recalcular su hash ordinario en el manifiesto debe dar NO CONFORME."""
    frz.freeze(mini_repo, confirmar=True, comprobar_git=False)
    assert frz.verify_strict(mini_repo) == []
    path = mini_repo / rel
    marker = b"\n<!-- alterado -->\n" if path.suffix == ".md" else b"\n# alterado tras el congelamiento\n"
    path.write_bytes(path.read_bytes() + marker)
    m = frz.load_manifest(mini_repo)
    m["sha256_protocolo"][rel] = frz.sha256_file(path)
    (mini_repo / frz.MANIFEST_JSON).write_bytes(frz.manifest_bytes(m))
    errors = frz.verify_strict(mini_repo)
    assert errors and any(rel in e and "auditada" in e for e in errors)
    assert not any("HASH DISTINTO" in e for e in errors)                   # el hash ordinario si cuadra


def _git(root, *args):
    import subprocess
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.autocrlf=false", *args],
                          cwd=root, capture_output=True, text=True, check=True).stdout.strip()


def test_git_commit_is_the_external_reference_for_a_full_manifest_rewrite(mini_repo):
    """Una reescritura COMPLETA del manifiesto (incluido el registro) solo la detecta la referencia externa."""
    frz.freeze(mini_repo, confirmar=True, comprobar_git=False)
    _git(mini_repo, "init", "-q")
    _git(mini_repo, "add", "-A")
    _git(mini_repo, "commit", "-q", "-m", "congelamiento sintetico")
    commit = _git(mini_repo, "rev-parse", "HEAD")
    assert frz.verify_against_git(mini_repo, commit) == []
    rel = "tests/test_tda17_calentamiento_huecos.py"
    path = mini_repo / rel
    path.write_bytes(path.read_bytes() + b"\n# alterado\n")
    m = frz.load_manifest(mini_repo)
    m["sha256_protocolo"][rel] = m["congelamiento"]["sha256_auditado"][rel] = frz.sha256_file(path)
    (mini_repo / frz.MANIFEST_JSON).write_bytes(frz.manifest_bytes(m))
    assert frz.verify_strict(mini_repo) == []                              # coherente internamente...
    errors = frz.verify_against_git(mini_repo, commit)                     # ...pero no frente al commit
    assert any(rel in e for e in errors) and any(frz.MANIFEST_JSON in e for e in errors)


def test_manifest_refresh_failing_before_writing_leaves_manifest_byte_identical(mini_repo):
    """Regresion F2: un artefacto historico alterado hace fallar la regeneracion sin tocar el manifiesto."""
    before = (mini_repo / frz.MANIFEST_JSON).read_bytes()
    victim = list(frz.load_manifest(mini_repo)["sha256_artefactos_previos"])[5]
    (mini_repo / victim).write_bytes(b"alterado\n")
    t = mini_repo / "tests/test_tda17_protocol.py"                         # hay algo que regenerar
    t.write_bytes(t.read_bytes() + b"\n")
    with pytest.raises(frz.ProtocolError, match="no se escribe nada"):
        frz.refresh_protocol_hashes(mini_repo)
    assert (mini_repo / frz.MANIFEST_JSON).read_bytes() == before
    assert not list(mini_repo.rglob("*.tda17tmp"))


def test_manifest_refresh_failing_after_writing_restores_manifest(mini_repo):
    before = (mini_repo / frz.MANIFEST_JSON).read_bytes()
    t = mini_repo / "tests/test_tda17_protocol.py"
    t.write_bytes(t.read_bytes() + b"\n")
    def boom():
        raise RuntimeError("fallo simulado tras la escritura")
    with pytest.raises(RuntimeError):
        frz.refresh_protocol_hashes(mini_repo, _hook_tras_escritura=boom)
    assert (mini_repo / frz.MANIFEST_JSON).read_bytes() == before
    frz.refresh_protocol_hashes(mini_repo)                                 # sin fallo, si se regenera
    assert frz.verify_strict(mini_repo) == []


# --------------------------------------------------------------------------- A: rutas validadas antes de abrir nada
HOLDOUT_FAKE = "22_mnq_09_25.Last.txt"          # nombre de la lista holdout_files; contenido FICTICIO en tmp


class _Spy:
    """Registra toda lectura de fichero (Path.read_bytes/read_text/open) y toda llamada a sha256_file."""

    def __init__(self, monkeypatch):
        self.opened: list[Path] = []
        self.hashed: list[Path] = []
        spy = self
        for name in ("read_bytes", "read_text", "open"):
            orig = getattr(Path, name)
            def wrapper(self_, *a, _orig=orig, **k):
                spy.opened.append(Path(os.path.abspath(self_)))
                return _orig(self_, *a, **k)
            monkeypatch.setattr(Path, name, wrapper)
        orig_hash = frz.sha256_file
        def hash_wrapper(path):
            spy.hashed.append(Path(os.path.abspath(path)))
            return orig_hash(path)
        monkeypatch.setattr(frz, "sha256_file", hash_wrapper)

    def touched(self, target: Path) -> bool:
        t = Path(os.path.abspath(target))
        tr = t.resolve()
        return any(p == t or p.resolve() == tr for p in self.opened + self.hashed)


def _inject(root, rel, target):
    """Sustituye la primera entrada de artefactos por ``rel`` (se mantienen los 188) y escribe el manifiesto."""
    m = frz.load_manifest(root)
    arts = m["sha256_artefactos_previos"]
    first = next(iter(arts))
    m["sha256_artefactos_previos"] = {(rel if k == first else k): v for k, v in arts.items()}
    m["sha256_artefactos_previos"][rel] = hashlib.sha256(b"ficticio\n").hexdigest()
    (root / frz.MANIFEST_JSON).write_bytes(frz.manifest_bytes(m))
    assert len(m["sha256_artefactos_previos"]) == 188
    return target


def _make(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"ficticio\n")
    return path


@pytest.mark.parametrize("case", ["holdout_en_data_raw", "holdout_fuera_de_data_raw", "data_raw_investigacion",
                                  "padre", "absoluta", "barra_invertida", "enlace_a_holdout", "enlace_fuera"])
def test_forbidden_paths_are_rejected_before_any_file_is_opened(mini_repo, monkeypatch, case):
    root = mini_repo
    outside = root.parent / f"fuera_{case}"
    if case == "holdout_en_data_raw":
        target = _inject(root, f"data/raw/mnq/{HOLDOUT_FAKE}", _make(root / "data/raw/mnq" / HOLDOUT_FAKE))
    elif case == "holdout_fuera_de_data_raw":
        target = _inject(root, f"reports/mnq/{HOLDOUT_FAKE}", _make(root / "reports/mnq" / HOLDOUT_FAKE))
    elif case == "data_raw_investigacion":
        target = _inject(root, "data/raw/mnq/00_mnq_03_20.Last.txt", _make(root / "data/raw/mnq/00_mnq_03_20.Last.txt"))
    elif case == "padre":
        target = _inject(root, f"../{outside.name}/x.csv", _make(outside / "x.csv"))
    elif case == "absoluta":
        target = _make(outside / "abs.csv")
        _inject(root, target.as_posix(), target)
    elif case == "barra_invertida":
        target = _inject(root, "reports\\mnq\\x.csv", _make(root / "reports/mnq/x.csv"))
    else:
        dest = _make(root / "data/raw/mnq" / HOLDOUT_FAKE) if case == "enlace_a_holdout" else _make(outside / "y.csv")
        link = root / "reports/mnq/enlace.csv"
        try:
            os.symlink(dest, link)
        except (OSError, NotImplementedError):
            pytest.skip("este sistema no permite crear enlaces simbolicos")
        target = _inject(root, "reports/mnq/enlace.csv", dest)
    before = (root / frz.MANIFEST_JSON).read_bytes()
    spy = _Spy(monkeypatch)
    errors = frz.verify_strict(root)
    assert errors and all("RUTA PROHIBIDA" in e for e in errors)
    assert spy.hashed == []                                                # ningun hash antes de validar
    assert not spy.touched(target)                                         # el fichero prohibido nunca se abre
    with pytest.raises(frz.ProtocolError):
        frz.refresh_protocol_hashes(root)
    with pytest.raises(frz.ProtocolError):
        frz.freeze(root, confirmar=True, comprobar_git=False)
    assert spy.hashed == [] and not spy.touched(target)
    monkeypatch.undo()
    assert (root / frz.MANIFEST_JSON).read_bytes() == before


@pytest.mark.parametrize("dest_kind", ["holdout", "fuera"])
def test_path_resolving_to_forbidden_location_is_rejected_without_opening(mini_repo, monkeypatch, dest_kind):
    """Enlace simbolico SIMULADO (no requiere privilegios): reports/mnq/enlace.csv resuelve al holdout o fuera."""
    root = mini_repo
    dest = _make(root / "data/raw/mnq" / HOLDOUT_FAKE) if dest_kind == "holdout" else _make(root.parent / "fuera_sim" / "z.csv")
    link = _make(root / "reports/mnq/enlace.csv")
    target = _inject(root, "reports/mnq/enlace.csv", dest)
    orig_resolve = Path.resolve
    def fake_resolve(self, *a, **k):
        r = orig_resolve(self, *a, **k)
        return orig_resolve(dest) if r == orig_resolve(link) else r
    monkeypatch.setattr(Path, "resolve", fake_resolve)
    spy = _Spy(monkeypatch)
    errors = frz.verify_strict(root)
    assert errors and all("RUTA PROHIBIDA" in e and "resuelve" in e for e in errors)
    assert spy.hashed == [] and not spy.touched(target) and not spy.touched(link)


def test_normal_strict_verification_still_hashes_every_declared_file(mini_repo, monkeypatch):
    spy = _Spy(monkeypatch)
    assert frz.verify_strict(mini_repo) == []
    assert len(spy.hashed) == len(frz.PROTOCOL_FILES) + 5 + 188
    assert frz.validate_manifest_paths(ROOT, json.loads(MANIFEST.read_text(encoding="utf-8")),
                                       yaml.safe_load(SNAPSHOT.read_text(encoding="utf-8"))["holdout"]["holdout_files"]) == []
