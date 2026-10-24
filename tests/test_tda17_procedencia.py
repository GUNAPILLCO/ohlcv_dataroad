"""Pruebas SINTETICAS de la procedencia de la ejecucion oficial (tda17_entradas) y del orquestador TH35
(tda17_analysis.th35_pipeline / official_th35, tda17_causalidad.official_causal_audit).

La ejecucion oficial no acepta datos del llamante: lee las entradas declaradas en el manifiesto y comprueba su
huella sobre los mismos bytes que interpreta; TH37, las celdas, los IC, ``per_cell`` y el R2 global se derivan
dentro. Sin MNQ ni holdout: en los repositorios temporales las rutas de entrada se REDIRIGEN (monkeypatch) a
ficheros parquet SINTETICOS antes de cualquier lectura, y una guarda comprueba que nunca apuntan a data/interim.
Las entradas reales solo se mencionan por su ruta y su huella del manifiesto (nunca se leen ni se hashean aqui).
"""
from __future__ import annotations

import copy
import re

import numpy as np
import pandas as pd
import pytest

from ohlcv_dataroad.ingest import tda17_analysis as an
from ohlcv_dataroad.ingest import tda17_causalidad as ca
from ohlcv_dataroad.ingest import tda17_enmiendas_e02 as m2
from ohlcv_dataroad.ingest import tda17_entradas as ent
from ohlcv_dataroad.ingest import tda17_indicators as ind
from ohlcv_dataroad.ingest import tda17_primitives as pr
from ohlcv_dataroad.ingest import tda17_protocol_freeze as frz
from tda17_synthetic import make_frame, rebuild_tda04
from test_tda17_enmienda_02 import NO_OFICIAL_MSG, built, project_e02_approved, project_rejection, repo  # noqa: F401 (fixtures)

Err = ind.ProtocolNotAuthorizedError
SYN_VAR, SYN_VAL = "data/sintetico/variables.parquet", "data/sintetico/validez.parquet"


def raises_exactly(exc_type, message, fn, *args, **kwargs):
    with pytest.raises(exc_type) as info:
        fn(*args, **kwargs)
    assert type(info.value) is exc_type and str(info.value) == message, f"{type(info.value).__name__}: {info.value}"


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entrada sintetica -> (variables de TDA-04, mascara de validez), como los dos parquet de la entrada real."""
    base = df.drop(columns=[ind.SYNTHETIC_MARKER], errors="ignore")
    validity = base[["timestamp", *ind.VALIDITY_COLUMNS]].copy()
    variables = base.drop(columns=ind.VALIDITY_COLUMNS)
    return variables, validity


def _small_frame():
    return make_frame(contracts=(("MNQ_A", 2), ("MNQ_B", 2)), bars_per_day=150, dense_gaps=(30, 60))


@pytest.fixture
def synthetic_inputs(repo, monkeypatch):
    """Escribe las entradas SINTETICAS en el repositorio temporal y redirige a ellas las rutas oficiales; las huellas
    esperadas siguen siendo las del manifiesto salvo que la prueba las fije."""
    root = repo["root"]
    variables, validity = _split(_small_frame())
    (root / "data" / "sintetico").mkdir(parents=True, exist_ok=True)
    variables.to_parquet(root / SYN_VAR, index=False)
    validity.to_parquet(root / SYN_VAL, index=False)
    monkeypatch.setattr(ent, "VARIABLES", SYN_VAR)
    monkeypatch.setattr(ent, "VALIDITY", SYN_VAL)
    monkeypatch.setattr(ent, "INPUTS", (SYN_VAR, SYN_VAL))
    assert not any(r.startswith("data/interim") for r in ent.INPUTS)        # guarda: nunca la entrada real
    hashes = {SYN_VAR: frz.sha256_file(root / SYN_VAR), SYN_VAL: frz.sha256_file(root / SYN_VAL)}
    return {**repo, "variables": variables, "validity": validity, "hashes": hashes}


def _trust_synthetic(monkeypatch, ctx):
    monkeypatch.setattr(ent, "expected_input_hashes", lambda root: dict(ctx["hashes"]))
    monkeypatch.setattr(an, "check_expected_inputs", lambda df, cfg: {"filas": len(df)})   # recuentos de MNQ


# =========================================================================== entradas: procedencia
def test_official_inputs_are_the_manifest_entries():
    m = frz.load_manifest(m2.ROOT)
    assert ent.INPUTS == ("data/interim/mnq/tda04_variables_1m.parquet", "data/interim/mnq/tda04_return_validity_mask.parquet")
    assert ent.expected_input_hashes(m2.ROOT) == {r: m["sha256_entradas"][r] for r in ent.INPUTS}


def test_official_inputs_are_the_manifest_entries_or_rejected(monkeypatch):
    real = frz.load_manifest(m2.ROOT)
    for rel, bad in ((ent.VARIABLES, None), (ent.VALIDITY, "corta")):
        m = copy.deepcopy(real)
        if bad is None:
            del m["sha256_entradas"][rel]
        else:
            m["sha256_entradas"][rel] = bad
        monkeypatch.setattr(frz, "load_manifest", lambda root, m=m: m)
        raises_exactly(Err, f"[ENTRADA_NO_DECLARADA] {rel} no figura en sha256_entradas del manifiesto",
                       ent.expected_input_hashes, m2.ROOT)


def test_inputs_read_verified_checks_the_same_bytes(tmp_path):
    (tmp_path / "x.parquet").write_bytes(b"datos sinteticos")
    good = frz.sha256_bytes(b"datos sinteticos", ".parquet")
    assert ent.read_verified(tmp_path, "x.parquet", good) == b"datos sinteticos"
    raises_exactly(Err, f"[ENTRADA_PROCEDENCIA] x.parquet no es la entrada del manifiesto (SHA-256 {good})",
                   ent.read_verified, tmp_path, "x.parquet", "0" * 64)
    raises_exactly(Err, "[ENTRADA_PROCEDENCIA] falta la entrada y.parquet", ent.read_verified, tmp_path, "y.parquet", good)
    for bad in ("data/raw/mnq/00_mnq_03_20.Last.txt", "data/holdout/x.parquet"):
        raises_exactly(Err, f"[ENTRADA_PROHIBIDA] ruta de entrada prohibida: {bad}", ent.read_verified, tmp_path, bad, good)


def test_inputs_are_never_read_without_the_verified_official_config(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("se leyo una entrada sin configuracion oficial verificada")
    monkeypatch.setattr(ent, "read_verified", forbidden)
    cfg = m2.effective_document_e02(borrador=True)
    raises_exactly(Err, "[E02_REQUERIDA] la configuracion no contiene E01 y E02 APROBADAS y activables",
                   ent.load_official_inputs, cfg)
    forged = copy.deepcopy(cfg)
    forged["trazabilidad_protocolo"].update(modo="PRODUCCION", activable=True, ejecucion_oficial=True, publicacion_E02="f" * 40)
    forged["trazabilidad_protocolo"]["enmiendas"][-1]["estado"] = "APROBADA"
    rejection = project_rejection("f" * 40)                                 # proyecto PENDIENTE o APROBADO (v11)
    raises_exactly(Err, f"[E02_OFICIAL_NO_VERIFICABLE] la configuracion oficial no se puede derivar de M2: {rejection}",
                   ent.load_official_inputs, forged)
    raises_exactly(Err, rejection, an.official_th35, "f" * 40)
    raises_exactly(Err, rejection, ca.official_causal_audit, "f" * 40)
    if project_e02_approved():                                              # aprobacion preparatoria: nunca oficial
        raises_exactly(Err, NO_OFICIAL_MSG, ent.load_official_inputs, an.production_config())


def test_official_inputs_positive_with_provenance(synthetic_inputs, monkeypatch):
    ctx = synthetic_inputs
    _trust_synthetic(monkeypatch, ctx)
    cfg = m2.official_config_e02(ctx["root"], ctx["P2"])
    df, prov = ent.load_official_inputs(cfg, ctx["root"])
    pd.testing.assert_frame_equal(df, ind.join_inputs(ctx["variables"], ctx["validity"]), check_exact=True)
    assert prov == {"entradas": ctx["hashes"], "huella_marco": ent.frame_fingerprint(df), "filas": len(df),
                    "publicacion_E02": ctx["P2"], "sha256_efectiva": cfg["trazabilidad_protocolo"]["sha256_efectiva"]}


def test_official_inputs_reject_unverified_or_unexpected_data(synthetic_inputs, monkeypatch):
    ctx = synthetic_inputs
    root, cfg = ctx["root"], m2.official_config_e02(ctx["root"], ctx["P2"])
    monkeypatch.setattr(ent, "expected_input_hashes", lambda r: {SYN_VAR: "a" * 64, SYN_VAL: ctx["hashes"][SYN_VAL]})
    raises_exactly(Err, f"[ENTRADA_PROCEDENCIA] {SYN_VAR} no es la entrada del manifiesto (SHA-256 {ctx['hashes'][SYN_VAR]})",
                   ent.load_official_inputs, cfg, root)
    monkeypatch.setattr(ent, "expected_input_hashes", lambda r: dict(ctx["hashes"]))
    with pytest.raises(Err, match=r"^la entrada no es la esperada \(obtenido, esperado\): "):   # recuentos de MNQ
        ent.load_official_inputs(cfg, root)
    tampered = ctx["validity"].copy()
    tampered.loc[5, "r_1m_valid"] = not tampered.loc[5, "r_1m_valid"]
    tampered.to_parquet(root / SYN_VAL, index=False)                           # alterada tras fijar la huella
    digest = frz.sha256_file(root / SYN_VAL)
    raises_exactly(Err, f"[ENTRADA_PROCEDENCIA] {SYN_VAL} no es la entrada del manifiesto (SHA-256 {digest})",
                   ent.load_official_inputs, cfg, root)


def test_official_th35_uses_only_verified_inputs(synthetic_inputs, monkeypatch):
    ctx = synthetic_inputs
    _trust_synthetic(monkeypatch, ctx)
    seen = {}

    def capture(df, cfg):
        seen.update(df=df, cfg=cfg)
        return {"estado": "CAPTURADO"}
    monkeypatch.setattr(an, "th35_pipeline", capture)
    out = an.official_th35(ctx["P2"], ctx["root"])
    assert out["estado"] == "CAPTURADO" and out["procedencia"]["huella_marco"] == ent.frame_fingerprint(seen["df"])
    assert out["procedencia"]["entradas"] == ctx["hashes"]
    tr = seen["cfg"]["trazabilidad_protocolo"]
    assert (tr["ejecucion_oficial"], tr["publicacion_E02"]) == (True, ctx["P2"])
    ctx["variables"].iloc[:-1].to_parquet(ctx["root"] / SYN_VAR, index=False)  # otra entrada: no llega al orquestador
    seen.clear()
    with pytest.raises(Err, match=r"^\[ENTRADA_PROCEDENCIA\] "):
        an.official_th35(ctx["P2"], ctx["root"])
    assert seen == {}


def test_official_causal_audit_on_verified_inputs(synthetic_inputs, monkeypatch, small_audit):
    ctx = synthetic_inputs
    _trust_synthetic(monkeypatch, ctx)
    df_small, table = small_audit
    calls = []

    def run(df, cfg):
        calls.append(ent.frame_fingerprint(df))
        return table
    monkeypatch.setattr(ca, "run_causal_audit", run)
    out, prov = ca.official_causal_audit(ctx["P2"], ctx["root"])
    assert out is table and calls == [prov["huella_marco"]]
    pd.testing.assert_frame_equal(ind.join_inputs(ctx["variables"], ctx["validity"]), df_small, check_exact=True)


# =========================================================================== orquestador TH35 (sin datos del llamante)
@pytest.fixture(scope="module")
def small_audit():
    """Entrada sintetica pequena (la de ``synthetic_inputs``) y su auditoria causal completa, calculada una vez."""
    variables, validity = _split(_small_frame())
    df = ind.join_inputs(variables, validity)
    return df, ca.run_causal_audit(df, m2.effective_document_e02(borrador=True))


def _engine_input(bars_per_day=300, **kw):
    """Un contrato alineado de 25 jornadas con nivel y aperturas desplazados (como el panel del motor). Con 110 barras
    por jornada y sin huecos internos (18:01-19:50 NY) toda la poblacion cae en un anio y un segmento."""
    df = make_frame(contracts=(("MNQ_A", 25),), bars_per_day=bars_per_day, aligned=True, seed=4, dense_gaps=None, **kw)
    rng = np.random.default_rng(5)
    can = df[["timestamp", "source_file", "contract", "trading_date", "segment_id", "open", "high", "low", "close",
              "volume"]].copy()
    for k in ("open", "high", "low", "close"):
        can[k] = can[k] - 14500.0
    can["open"] = can["open"] + pr.TICK * rng.integers(-2, 3, len(can))
    can["high"] = np.maximum(can["high"], can["open"])
    can["low"] = np.minimum(can["low"], can["open"])
    return rebuild_tda04(can)


@pytest.fixture(scope="module")
def mech_cfg():
    cfg = copy.deepcopy(m2.effective_document_e02(borrador=True))
    cfg["analisis"]["bootstrap"]["replicas"] = 3                              # solo mecanica (copia de prueba)
    return cfg


@pytest.fixture(scope="module")
def blocked_run(mech_cfg):
    df = _engine_input()
    return df, an.th35_pipeline(df, mech_cfg)


def test_th35_pipeline_derives_cells_and_blocks_until_explained(blocked_run, mech_cfg):
    df, out = blocked_run
    assert out["estado"] == "BLOQUEADO_HASTA_EXPLICAR" and "clasificacion" not in out and "subniveles" not in out
    assert all(re.match(r"^(GLOBAL|ANIO_\d{4}|SEG_\S+): ", b) for b in out["bloqueos"])
    assert any(b.startswith("GLOBAL: BLOQUEAR_HASTA_EXPLICAR: nulos de P_centrada") for b in out["bloqueos"])
    pc = out["per_cell"]
    cells = [c for c in an.bootstrap_stream_cells(mech_cfg) if c != "GLOBAL"]
    assert sorted(pc["celda"].unique()) == sorted(cells) and len(pc) == len(cells) * 134
    nd = pc[pc["estado_matriz"] == "NO_DEFINIDA"]
    assert (nd["ic_estado"] == "MATRIZ_NO_DEFINIDA").all() and not nd["publicable"].any()
    d21 = pc[pc["celda"] == "ANIO_2021"].set_index("columna")
    a21 = out["analisis_celdas"]["ANIO_2021"]["r2_todas"].set_index("columna")
    np.testing.assert_array_equal(d21["r2"].to_numpy(), a21.loc[d21.index, "r2"].to_numpy())   # derivado, no aportado
    assert out["validacion_spearman"]["resultado"] == "NO_EJECUTABLE"
    assert all(t["ic_inf"].isna().all() for t in out["spearman"].values())  # sin validacion: IC no publicados
    assert ca.verify_causal_audit(out["auditoria_causalidad"], df, mech_cfg)["bloqueos"] == []


def test_th35_pipeline_classifies_without_blocks(mech_cfg, monkeypatch):
    """Rama sin bloqueos. Toda la poblacion cae en GLOBAL = ANIO_2021 = SEG_16:02-20:00 (misma matriz); el registro
    de la COPIA de prueba se fija a los recuentos sinteticos observados en esa matriz."""
    df = _engine_input(bars_per_day=110, hi_every=0, long_hi_every=0)
    first = an.th35_pipeline(df, mech_cfg)
    defined = {c for c, a in first["analisis_celdas"].items() if a["estado_matriz"] == "DEFINIDA"}
    assert defined == {"GLOBAL", "ANIO_2021", "SEG_16:02-20:00"} and first["estado"] == "BLOQUEADO_HASTA_EXPLICAR"
    rk = {r["matriz"]: r["autovalores"] for r in first["analisis_celdas"]["GLOBAL"]["rango"]}
    cfg = copy.deepcopy(mech_cfg)
    cfg["identidades"]["verificacion_numerica"]["nulos_esperados"].update(rk)
    monkeypatch.setattr(ca, "run_causal_audit", lambda d, c: first["auditoria_causalidad"])  # TH37 ya calculada
    res = an.th35_pipeline(df, cfg)
    assert res["bloqueos"] == [] and res["estado"] == "SIN_BLOQUEO"
    cl = res["clasificacion"]
    assert len(cl) == 134 and set(cl["clasificacion"]) <= {"RECONSTRUIBLE", "NUEVA_REPRESENTACION", "INESTABLE"}
    assert (cl["total_no_publicables"] >= 11).all() and (cl["MATRIZ_NO_DEFINIDA"] == 11).all()   # 5 anios + 6 segmentos
    assert set(cl["clasificacion"]) == {"INESTABLE"}                         # 1 de 6 anios: nunca hay evidencia suficiente
    assert len(res["subniveles"]) == 134


@pytest.mark.parametrize("case", ["ausente", "incompleta", "discrepancia"])
def test_th35_route_rejects_absent_incomplete_or_discrepant_th37(small_audit, mech_cfg, monkeypatch, case):
    df, table = small_audit
    bad = {"ausente": None, "incompleta": table.iloc[:-5], "discrepancia": table.copy()}[case]
    if case == "discrepancia":
        i = bad.index[(bad["prueba"] == "TRUNCAMIENTO") & (bad["serie"] == "E5_barra__valor")][0]
        bad.loc[i, ["resultado", "discrepancias"]] = ["DISCREPANCIA", 1]
    monkeypatch.setattr(ca, "run_causal_audit", lambda d, c: bad)

    def no_catalog(*a, **k):
        raise AssertionError("TH35 calculo el catalogo sin TH37 verificada")
    monkeypatch.setattr(ind, "build_panel_p", no_catalog)
    with pytest.raises(Err) as info:
        an.th35_pipeline(df, mech_cfg)
    msg = str(info.value)
    if case == "ausente":
        assert type(info.value) is Err and msg == "[TH37_AUSENTE] no hay auditoria causal de la entrada: TH35 no puede publicarse"
    elif case == "incompleta":
        assert type(info.value) is Err and re.fullmatch(r"\[TH37_INCOMPLETA\] faltan 5 filas del plan de la auditoria \(p\. ej\. .*\)", msg)
    else:
        assert type(info.value) is an.BlockedUntilExplainedError
        assert msg == "[TH37_DISCREPANCIA] TH35 BLOQUEADA_HASTA_EXPLICAR: series con discrepancia causal ['E5_barra__valor']"


def test_th35_pipeline_rejects_the_synthetic_marker(mech_cfg):
    raises_exactly(ValueError, "th35_pipeline recibe la entrada sin la columna de marco sintetico",
                   an.th35_pipeline, _small_frame(), mech_cfg)
