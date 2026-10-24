"""Pruebas SINTETICAS de la publicacion oficial de TH35 (tda17_salidas; E02, decisiones 17-19).

Ruta unica ``official_publish_th35(P2)``: configuracion oficial derivada de M2, ``official_th35`` en el mismo proceso
sobre entradas verificadas, tablas del esquema minimo construidas a partir de ese resultado, registro de procedencia
y verificacion posterior (huellas y reproduccion). Sin MNQ ni holdout: las publicaciones se hacen en REPOSITORIOS
GIT TEMPORALES con aprobacion simulada y entradas SINTETICAS (``synthetic_inputs`` redirige las rutas antes de
cualquier lectura). Negativas: tipo EXACTO y mensaje COMPLETO.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import subprocess
import sys
import threading

import pandas as pd
import pytest
import yaml

from ohlcv_dataroad.ingest import tda17_analysis as an
from ohlcv_dataroad.ingest import tda17_causalidad as ca
from ohlcv_dataroad.ingest import tda17_enmiendas as e01
from ohlcv_dataroad.ingest import tda17_enmiendas_e02 as m2
from ohlcv_dataroad.ingest import tda17_entradas as ent
from ohlcv_dataroad.ingest import tda17_protocol_freeze as frz
from ohlcv_dataroad.ingest import tda17_salidas as sal
from test_tda17_enmienda_02 import built, repo  # noqa: F401 (fixtures)
from test_tda17_procedencia import (_engine_input, _trust_synthetic, blocked_run, mech_cfg,  # noqa: F401 (fixtures)
                                    synthetic_inputs)

Err = m2.ProtocolNotAuthorizedError
SRC = m2.ROOT


def raises_exactly(exc_type, message, fn, *args, **kwargs):
    with pytest.raises(exc_type) as info:
        fn(*args, **kwargs)
    assert type(info.value) is exc_type and str(info.value) == message, f"{type(info.value).__name__}: {info.value}"


@pytest.fixture(scope="module")
def cfg():
    return m2.effective_document_e02(borrador=True)


@pytest.fixture(scope="module")
def unblocked_run(mech_cfg):
    """Resultado SIN_BLOQUEO (registro de la COPIA de prueba fijado a los recuentos sinteticos, como en
    test_tda17_procedencia.test_th35_pipeline_classifies_without_blocks)."""
    df = _engine_input(bars_per_day=110, hi_every=0, long_hi_every=0)
    first = an.th35_pipeline(df, mech_cfg)
    rk = {r["matriz"]: r["autovalores"] for r in first["analisis_celdas"]["GLOBAL"]["rango"]}
    c = copy.deepcopy(mech_cfg)
    c["identidades"]["verificacion_numerica"]["nulos_esperados"].update(rk)
    res = an.th35_pipeline(df, c)
    assert res["estado"] == "SIN_BLOQUEO"
    return res


# =========================================================================== autorizacion y esquema de E02
def test_authorizations_cite_the_approved_documents():
    doc = yaml.safe_load((SRC / m2.E02_YAML).read_text(encoding="utf-8"))["enmienda"]
    for key in ("autorizacion_metodologica", "autorizacion_complementaria"):
        a = doc[key]
        assert frz.sha256_file(SRC / a["documento"]) == a["sha256_documento"], key
    comp = doc["autorizacion_complementaria"]
    assert comp["documento"] == "reports/mnq/TDA17_E02_propuesta_ambiguedades_y_salidas.md" and comp["fecha"] == "2026-10-19"
    assert sorted(comp["decisiones"], key=int) == [str(i) for i in range(14, 20)]
    assert (SRC / comp["anexo"]).is_file()
    assert list(doc["autorizacion_metodologica"]["decisiones"]) == [str(i) for i in range(1, 14)]   # trazabilidad anterior


def test_schema_is_the_approved_minimal_schema(cfg):
    sch = sal.schema(cfg)
    assert sch["directorio"] == "reports/mnq"
    assert sch["columnas"] == {
        sal.CAUSAL: ["prueba", "contrato", "punto_corte", "motivo_corte", "serie", "filas_comparadas", "discrepancias",
                     "max_abs_diff", "resultado", "parametro", "origen", "etiqueta"],
        sal.STABILITY: ["columna", "celda", "tipo", "estado", "estado_matriz", "r2", "ic_inf", "ic_sup", "publicable", "ic_estado"],
        sal.R2: ["celda", "columna", "r2", "r2_espectral", "r2_minima_norma", "dif_cruzada", "publicable",
                 "etiqueta_publicacion", "etiqueta_exactitud"],
        sal.SPEARMAN: ["celda", "columna_i", "columna_j", "rho", "ic_inf", "ic_sup", "ic_estado", "replicas_invalidas"],
        sal.CLASSIFICATION: ["columna", "clasificacion", "subnivel", "motivo_subnivel", "r2_global", "MATRIZ_NO_DEFINIDA",
                             "R2_NO_PUBLICABLE_POR_COLINEALIDAD", "IC_NO_CALCULABLE", "total_no_publicables"],
        sal.DIAGNOSTIC: ["celda", "alcance", "matriz", "seccion", "elemento", "columna", "metodo", "valor", "etiqueta", "marca"]}
    assert set(sal.UNBLOCKED) <= set(cfg["salidas"]["tablas"]) and sal.DIAGNOSTIC not in cfg["salidas"]["tablas"]


@pytest.mark.parametrize("mutate", [
    lambda s, d: s.update(recibe_tablas=True),
    lambda s, d: s.update(ruta_publicacion="write_official_table"),
    lambda s, d: s["con_bloqueo"].append(sal.STABILITY),
    lambda s, d: s["sin_bloqueo"].pop(),
    lambda s, d: s["tablas"].pop(sal.SPEARMAN),
    lambda s, d: s["procedencia"]["campos"].pop(),
    lambda s, d: s["formato"].update(fin_de_linea="CRLF"),
    lambda s, d: d.update(levanta_el_bloqueo=True),
    lambda s, d: d["secciones"].pop(),
    lambda s, d: d.update(solo_con_estado="SIEMPRE"),
])
def test_schema_is_rejected_if_different(cfg, mutate):
    bad = copy.deepcopy(cfg)
    mutate(bad["salidas"]["esquema_minimo"], bad["salidas"]["diagnostico_bloqueo"])
    raises_exactly(Err, "[SALIDA_ESQUEMA] el esquema de salidas de E02 no es el implementado", sal.schema, bad)


# =========================================================================== tablas (funciones puras)
def test_blocked_result_gives_only_the_causal_audit_and_the_diagnostic(blocked_run, cfg):
    _df, res = blocked_run
    assert res["estado"] == "BLOQUEADO_HASTA_EXPLICAR"
    tabs = sal.official_tables(res, cfg)
    assert list(tabs) == [sal.CAUSAL, sal.DIAGNOSTIC]                      # nunca clasificacion ni estabilidad
    cols = sal.schema(cfg)["columnas"]
    assert all(list(t.columns) == cols[n] for n, t in tabs.items())
    pd.testing.assert_frame_equal(tabs[sal.CAUSAL], res["auditoria_causalidad"][cols[sal.CAUSAL]].reset_index(drop=True))
    dg = tabs[sal.DIAGNOSTIC]
    assert (dg["marca"] == "DIAGNOSTICO_BLOQUEADO").all() and set(dg["seccion"]) == set(sal.DIAGNOSTIC_SECTIONS)
    amb = dg[(dg["seccion"] == "AMBITO") & (dg["elemento"] == "bloqueo")]
    assert [f"{c}: {t}" for c, t in zip(amb["celda"], amb["etiqueta"])] == res["bloqueos"]
    dz = res["analisis_celdas"]["GLOBAL"]["diagnostico_bloqueo"]
    g = dg[dg["celda"] == "GLOBAL"]
    for what, sp in dz["espectros"].items():                               # espectro completo, por las dos convenciones
        e = g[(g["seccion"] == "ESPECTRO") & (g["matriz"] == what)]
        assert e[e["metodo"] == "AUTOVALOR_A1a"]["valor"].tolist() == [float(v) for v in sp["autovalores"]]
        assert e[e["metodo"] == "SINGULAR"]["valor"].tolist() == [float(v) for v in sp["singulares"]]
    for what, d in dz["direcciones_nulas"].items():                        # todas las cargas, sin umbral
        dn = g[(g["seccion"] == "DIRECCION_NULA") & (g["matriz"] == what) & (g["metodo"] == "CARGA")]
        assert len(dn) == d.shape[0] * (d.shape[1] - 1)
    rel = g[g["seccion"] == "RELACION_N"]
    assert set(rel["elemento"]) == {f"N{i}" for i in range(1, 22)}
    assert set(rel.loc[rel["elemento"] == "N10", "etiqueta"]) == {"NO_IDENTIDAD"}
    assert set(rel.loc[rel["elemento"] == "N3", "etiqueta"]) == {"SIN_DIRECCION_EN_P", "PENDIENTE_EXPEDIENTE"}
    assert rel.loc[(rel["elemento"] == "N21") & (rel["etiqueta"] == "VIOLACIONES"), "valor"].tolist() == [
        float(next(r for r in dz["relaciones_N"] if r["id"] == "N21")["violaciones"])]
    n4 = rel[(rel["elemento"] == "N4") & rel["etiqueta"].str.startswith("DIRECCION_")]
    assert set(n4["columna"]) == {"C5_n60__pos", "B5_n60__wr"} and set(n4["metodo"]) == {"LA-P"}
    assert not dg["etiqueta"].str.contains("RECONSTRUIBLE|NUEVA_REPRESENTACION|INESTABLE").any()


def test_unblocked_result_gives_the_five_tables(unblocked_run, cfg):
    res = unblocked_run
    tabs = sal.official_tables(res, cfg)
    assert list(tabs) == sal.UNBLOCKED
    cols = sal.schema(cfg)["columnas"]
    assert all(list(t.columns) == cols[n] for n, t in tabs.items())
    pd.testing.assert_frame_equal(tabs[sal.STABILITY], res["per_cell"][cols[sal.STABILITY]].reset_index(drop=True))
    defined = [c for c, a in res["analisis_celdas"].items() if a["estado_matriz"] == "DEFINIDA"]
    r2 = tabs[sal.R2]
    assert len(r2) == 134 * len(defined) and list(dict.fromkeys(r2["celda"])) == defined
    g = r2[r2["celda"] == "GLOBAL"].reset_index(drop=True)
    pd.testing.assert_series_equal(g["r2"], res["analisis_celdas"]["GLOBAL"]["r2_todas"]["r2"], check_names=False)
    assert len(tabs[sal.SPEARMAN]) == 134 * 133 // 2 * len(res["spearman"])
    cl = tabs[sal.CLASSIFICATION]
    assert len(cl) == 134 and cl["columna"].tolist() == res["clasificacion"]["columna"].tolist()
    assert cl["subnivel"].notna().all() and set(cl["clasificacion"]) <= {"RECONSTRUIBLE", "NUEVA_REPRESENTACION", "INESTABLE"}
    assert sal.csv_bytes(cl).count(b"\r") == 0 and sal.csv_bytes(cl) == sal.csv_bytes(cl.copy())


def test_official_tables_reject_inconsistent_results(blocked_run, unblocked_run, cfg):
    _df, blk = blocked_run
    raises_exactly(Err, "[SALIDA_ESTADO] estado del resultado no publicable: None", sal.official_tables,
                   {k: v for k, v in blk.items() if k != "estado"}, cfg)
    raises_exactly(Err, "[SALIDA_ESTADO] el estado del resultado no corresponde a sus bloqueos", sal.official_tables,
                   {**blk, "estado": "SIN_BLOQUEO"}, cfg)
    raises_exactly(Err, "[SALIDA_ESTADO] el estado del resultado no corresponde a sus bloqueos", sal.official_tables,
                   {**unblocked_run, "estado": "BLOQUEADO_HASTA_EXPLICAR"}, cfg)
    raises_exactly(Err, "[SALIDA_ESTADO] un resultado bloqueado no puede contener clasificacion ni subniveles",
                   sal.official_tables, {**blk, "clasificacion": unblocked_run["clasificacion"]}, cfg)
    bad = {**unblocked_run, "per_cell": unblocked_run["per_cell"].drop(columns="ic_estado")}
    raises_exactly(Err, f"[SALIDA_COLUMNAS] {sal.STABILITY}: el resultado oficial no contiene las columnas ['ic_estado']",
                   sal.official_tables, bad, cfg)


# =========================================================================== ruta unica en un repositorio temporal
@pytest.fixture
def publish_ctx(synthetic_inputs, monkeypatch, blocked_run):
    """Repositorio temporal en P2 con entradas sinteticas de confianza. El orquestador se sustituye por un resultado
    precalculado (mecanica de la publicacion; la ruta oficial lee y verifica las entradas igualmente) y se registra
    con que marco se llama."""
    ctx = synthetic_inputs
    _trust_synthetic(monkeypatch, ctx)
    ctx["calls"] = []
    ctx["result"] = blocked_run[1]

    def pipeline(df, cfg):
        ctx["calls"].append((ent.frame_fingerprint(df), cfg["trazabilidad_protocolo"]["publicacion_E02"]))
        return copy.deepcopy(ctx["result"])
    monkeypatch.setattr(an, "th35_pipeline", pipeline)
    ctx["out"] = ctx["root"] / "reports" / "mnq"
    return ctx


def _record(ctx):
    return json.loads((ctx["out"] / sal.PROVENANCE).read_text(encoding="utf-8"))


def _write_record(ctx, rec):
    (ctx["out"] / sal.PROVENANCE).write_bytes((json.dumps(rec, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def test_publish_blocked_writes_only_the_causal_audit_the_diagnostic_and_the_provenance(publish_ctx):
    ctx = publish_ctx
    root, p2 = ctx["root"], ctx["P2"]
    rec = sal.official_publish_th35(p2, root)
    cfg = m2.official_config_e02(root, p2)
    assert len(ctx["calls"]) == 1 and ctx["calls"][0][1] == p2               # calculo en este proceso, con la oficial
    assert list(rec) == sal.PROVENANCE_FIELDS and rec == _record(ctx)
    assert (rec["publicacion_E02"], rec["estado"], rec["bloqueos"]) == (p2, "BLOQUEADO_HASTA_EXPLICAR", ctx["result"]["bloqueos"])
    assert rec["sha256_efectiva"] == cfg["trazabilidad_protocolo"]["sha256_efectiva"]
    assert rec["remoto_E02"] == {"repositorio": str(ctx["bare"]), "ref": "refs/heads/main", "sha": p2}
    assert rec["entradas"] == ctx["hashes"] and rec["huella_marco"] == ctx["calls"][0][0]
    assert sorted(rec["tablas"]) == sorted(sal.BLOCKED)
    expected = sal.official_tables(ctx["result"], cfg)
    for n, t in expected.items():
        data = (ctx["out"] / n).read_bytes()
        assert data == sal.csv_bytes(t) and hashlib.sha256(data).hexdigest() == rec["tablas"][n]
    for n in (sal.STABILITY, sal.R2, sal.SPEARMAN, sal.CLASSIFICATION):
        assert not (ctx["out"] / n).exists(), n
    assert sal.check_official_publication(p2, root) == rec
    assert sal.reproduce_official_publication(p2, root) == rec and len(ctx["calls"]) == 2
    errors, info = m2.verify(root, p2)
    assert errors == [] and info["salidas_oficiales"] == "BLOQUEADO_HASTA_EXPLICAR"
    raises_exactly(Err, f"[SALIDA_EXISTENTE] ya existen salidas oficiales (no se sobrescriben): "
                        f"{[sal.CAUSAL, sal.DIAGNOSTIC, sal.PROVENANCE]}", sal.official_publish_th35, p2, root)


def test_publish_unblocked_writes_the_five_tables(publish_ctx, unblocked_run):
    ctx = publish_ctx
    ctx["result"] = unblocked_run
    root, p2 = ctx["root"], ctx["P2"]
    rec = sal.official_publish_th35(p2, root)
    assert rec["estado"] == "SIN_BLOQUEO" and rec["bloqueos"] == [] and sorted(rec["tablas"]) == sorted(sal.UNBLOCKED)
    assert not (ctx["out"] / sal.DIAGNOSTIC).exists()
    cl = pd.read_csv(ctx["out"] / sal.CLASSIFICATION, dtype=str)
    assert len(cl) == 134 and list(cl.columns) == sal.schema(m2.official_config_e02(root, p2))["columnas"][sal.CLASSIFICATION]
    assert sal.reproduce_official_publication(p2, root) == rec
    (ctx["out"] / sal.DIAGNOSTIC).write_bytes(b"celda\n")                   # salida oficial no registrada para el estado
    raises_exactly(Err, f"[SALIDA_CONJUNTO] salidas oficiales no registradas para el estado SIN_BLOQUEO: {[sal.DIAGNOSTIC]}",
                   sal.check_official_publication, p2, root)
    errors, _ = m2.verify(root, p2)
    assert errors == [f"[SALIDA_CONJUNTO] salidas oficiales no registradas para el estado SIN_BLOQUEO: {[sal.DIAGNOSTIC]}"]


def test_publication_tampering_is_detected(publish_ctx):
    ctx = publish_ctx
    root, p2, out = ctx["root"], ctx["P2"], ctx["out"]
    rec = sal.official_publish_th35(p2, root)
    causal = out / sal.CAUSAL
    original = causal.read_bytes()
    causal.write_bytes(original.replace(b"CONFORME", b"DISCREPANCIA", 1))   # tabla alterada: huella
    raises_exactly(Err, f"[SALIDA_HUELLA] {sal.CAUSAL} falta o no tiene la huella registrada",
                   sal.check_official_publication, p2, root)
    forged = copy.deepcopy(rec)                                              # alteracion coherente con el registro:
    forged["tablas"][sal.CAUSAL] = frz.sha256_file(causal)                    # solo la reproduccion la detecta
    _write_record(ctx, forged)
    assert sal.check_official_publication(p2, root) == forged
    raises_exactly(Err, f"[SALIDA_NO_REPRODUCIBLE] tablas publicadas distintas del recalculo oficial: {[sal.CAUSAL]}",
                   sal.reproduce_official_publication, p2, root)
    causal.write_bytes(original.replace(b"prueba,", b"ensayo,", 1))         # cabecera ajena al esquema, huella coherente
    forged["tablas"][sal.CAUSAL] = frz.sha256_file(causal)
    _write_record(ctx, forged)
    raises_exactly(Err, f"[SALIDA_ESQUEMA] {sal.CAUSAL} no tiene las columnas del esquema",
                   sal.check_official_publication, p2, root)
    causal.write_bytes(original)
    _write_record(ctx, rec)
    for field, value, message in (
            ("publicacion_E02", ctx["A2"], f"[SALIDA_P2] el registro cita la publicacion {ctx['A2']}, no {p2}"),
            ("sha256_efectiva", "0" * 64, "[SALIDA_EFECTIVA] el registro no corresponde a la configuracion oficial vigente "
                                          "de esta publicacion"),
            ("remoto_E02", {"repositorio": "otro", "ref": "refs/heads/main", "sha": p2},
             "[SALIDA_EFECTIVA] el registro no corresponde a la configuracion oficial vigente de esta publicacion"),
            ("entradas", {"data/otra.parquet": "0" * 64}, "[SALIDA_ENTRADAS] el registro no cita las entradas del manifiesto"),
            ("estado", "SIN_BLOQUEO", "[SALIDA_CONJUNTO] tablas registradas no corresponden al estado 'SIN_BLOQUEO'"),
            ("bloqueos", [], "[SALIDA_CONJUNTO] tablas registradas no corresponden al estado 'BLOQUEADO_HASTA_EXPLICAR'")):
        _write_record(ctx, {**rec, field: value})
        raises_exactly(Err, message, sal.check_official_publication, p2, root)
    bad = dict(rec)
    bad.pop("filas")
    _write_record(ctx, bad)
    raises_exactly(Err, f"[SALIDA_REGISTRO] el registro {sal.PROVENANCE} no tiene exactamente los campos {sal.PROVENANCE_FIELDS}",
                   sal.check_official_publication, p2, root)
    (out / sal.PROVENANCE).write_bytes(b"{no es json")
    raises_exactly(Err, f"[SALIDA_REGISTRO] falta o no es JSON valido el registro {sal.PROVENANCE}",
                   sal.check_official_publication, p2, root)
    (out / sal.PROVENANCE).unlink()
    raises_exactly(Err, f"[SALIDA_REGISTRO] falta o no es JSON valido el registro {sal.PROVENANCE}",
                   sal.check_official_publication, p2, root)
    _write_record(ctx, rec)
    (out / sal.DIAGNOSTIC).unlink()
    raises_exactly(Err, f"[SALIDA_HUELLA] {sal.DIAGNOSTIC} falta o no tiene la huella registrada",
                   sal.check_official_publication, p2, root)


def test_reproduction_detects_a_different_state(publish_ctx, unblocked_run):
    ctx = publish_ctx
    root, p2 = ctx["root"], ctx["P2"]
    sal.official_publish_th35(p2, root)
    ctx["result"] = unblocked_run                                        # el recalculo ya no da el mismo estado
    raises_exactly(Err, "[SALIDA_NO_REPRODUCIBLE] el recalculo no produce el mismo conjunto de tablas ni el mismo estado",
                   sal.reproduce_official_publication, p2, root)


def test_publish_rejects_a_result_from_another_configuration(publish_ctx, monkeypatch):
    ctx = publish_ctx
    real = an.official_th35

    def other(publicacion, root=None):
        out = real(publicacion, root)
        out["procedencia"]["sha256_efectiva"] = "0" * 64
        return out
    monkeypatch.setattr(an, "official_th35", other)
    raises_exactly(Err, "[SALIDA_PROCEDENCIA] el resultado de official_th35 no procede de la configuracion oficial de esta "
                        "publicacion", sal.official_publish_th35, ctx["P2"], ctx["root"])
    assert not ctx["out"].joinpath(sal.PROVENANCE).exists() and not ctx["out"].joinpath(sal.CAUSAL).exists()


def test_publish_never_writes_when_the_official_route_fails(publish_ctx):
    """Entradas alteradas tras fijar su huella o E02 no aprobada: la ruta rechaza antes de escribir nada."""
    ctx = publish_ctx
    root, p2 = ctx["root"], ctx["P2"]
    ctx["variables"].iloc[:-1].to_parquet(root / "data" / "sintetico" / "variables.parquet", index=False)
    with pytest.raises(Err, match=r"^\[ENTRADA_PROCEDENCIA\] "):
        sal.official_publish_th35(p2, root)
    assert ctx["calls"] == [] and not ctx["out"].joinpath(sal.PROVENANCE).exists()
    ctx["variables"].to_parquet(root / "data" / "sintetico" / "variables.parquet", index=False)
    reg = json.loads((root / m2.REGISTRY_E02).read_text(encoding="utf-8"))
    reg["enmiendas"][0].update(estado=e01.ESTADO_PENDIENTE, aprobacion=None)
    (root / m2.REGISTRY_E02).write_bytes((json.dumps(reg, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    raises_exactly(Err, "[E02_PENDIENTE] E02 no aprobada (estado PENDIENTE_AUDITORIA): la configuracion no es activable",
                   sal.official_publish_th35, p2, root)
    assert ctx["calls"] == [] and not ctx["out"].joinpath(sal.CAUSAL).exists()


def test_real_pipeline_end_to_end_through_the_single_route(synthetic_inputs, monkeypatch):
    """Sin sustituir el orquestador: TH37 real, catalogo, celdas y publicacion sobre la entrada sintetica pequena (copia
    de la configuracion con 3 replicas SOLO para acotar el tiempo de la mecanica)."""
    ctx = synthetic_inputs
    _trust_synthetic(monkeypatch, ctx)
    real = an.th35_pipeline

    def small(df, cfg):
        c = copy.deepcopy(cfg)
        c["analisis"]["bootstrap"]["replicas"] = 3
        return real(df, c)
    monkeypatch.setattr(an, "th35_pipeline", small)
    rec = sal.official_publish_th35(ctx["P2"], ctx["root"])
    assert rec["estado"] in sal.STATES and sorted(rec["tablas"]) == sorted(sal.BLOCKED if rec["bloqueos"] else sal.UNBLOCKED)
    causal = pd.read_csv(ctx["root"] / "reports" / "mnq" / sal.CAUSAL, dtype=str)
    assert len(causal) > 0 and set(causal["resultado"]) <= {"CONFORME", "NO_APLICABLE"}
    assert sal.reproduce_official_publication(ctx["P2"], ctx["root"]) == rec


# =========================================================================== publicacion transaccional (auditoria v5)
class _Crash(BaseException):
    """Muerte simulada del proceso: no la captura ningun ``except Exception``."""


_REAL = {n: getattr(sal, n) for n in ("_promote", "_rollback", "_write_durable")}


def _restore(monkeypatch):
    """Nuevo proceso tras la muerte simulada: auxiliares reales (el resto de parches de la fixture se mantiene)."""
    for name, fn in _REAL.items():
        monkeypatch.setattr(sal, name, fn)


def _crash_at(monkeypatch, k):
    """El proceso 'muere' en la promocion numero ``k`` (0 = ninguna promovida) sin deshacer nada."""
    real, count = sal._promote, [0]

    def promote(src, dst):
        if count[0] == k:
            raise _Crash()
        count[0] += 1
        real(src, dst)
    monkeypatch.setattr(sal, "_promote", promote)
    monkeypatch.setattr(sal, "_rollback", lambda out_dir, hashes: None)


def _assert_valid_after_retry(ctx, monkeypatch):
    root, p2, out = ctx["root"], ctx["P2"], ctx["out"]
    _restore(monkeypatch)
    rec = sal.official_publish_th35(p2, root)
    assert sal.check_official_publication(p2, root) == rec == _record(ctx)
    assert sal.reproduce_official_publication(p2, root) == rec
    assert not (out / sal.STAGING).exists()
    assert sorted(n for n in sal.OFFICIAL if (out / n).exists()) == sorted(sal.BLOCKED + [sal.PROVENANCE])


def test_publish_failure_in_process_rolls_back_and_the_retry_publishes(publish_ctx, monkeypatch):
    """Fallo ordinario a mitad de la promocion: se deshace todo, no queda publicacion parcial y se puede reintentar."""
    ctx = publish_ctx
    root, p2, out = ctx["root"], ctx["P2"], ctx["out"]
    real, count = sal._promote, [0]

    def promote(src, dst):
        if count[0] == 1:
            raise OSError("disco lleno (simulado)")
        count[0] += 1
        real(src, dst)
    monkeypatch.setattr(sal, "_promote", promote)
    with pytest.raises(OSError, match=r"^disco lleno \(simulado\)$"):
        sal.official_publish_th35(p2, root)
    assert [n for n in sal.OFFICIAL if (out / n).exists()] == [] and not (out / sal.STAGING).exists()
    _assert_valid_after_retry(ctx, monkeypatch)


@pytest.mark.parametrize("punto", ["preparacion", "promocion_0", "promocion_1", "promocion_2"])
def test_interrupted_publication_is_not_valid_and_is_recovered(publish_ctx, monkeypatch, punto):
    """Proceso muerto durante la preparacion o tras promover k de los 3 ficheros (causal, diagnostico, registro): sin
    registro confirmado no hay publicacion valida; el reintento retira los restos del diario y publica."""
    ctx = publish_ctx
    root, p2, out = ctx["root"], ctx["P2"], ctx["out"]
    if punto == "preparacion":
        real, count = sal._write_durable, [0]

        def write(path, data):
            if count[0] == 1:
                raise _Crash()
            count[0] += 1
            real(path, data)
        monkeypatch.setattr(sal, "_write_durable", write)
        monkeypatch.setattr(sal, "_rollback", lambda out_dir, hashes: None)
        k = 0
    else:
        k = int(punto[-1])
        _crash_at(monkeypatch, k)
    with pytest.raises(_Crash):
        sal.official_publish_th35(p2, root)
    present = [n for n in sal.OFFICIAL if (out / n).exists()]
    assert present == [sal.CAUSAL, sal.DIAGNOSTIC][:k] and (out / sal.STAGING).is_dir()
    assert (out / sal.STAGING / sal.JOURNAL).exists() == (punto != "preparacion")
    msg = "[SALIDA_INCOMPLETA] publicacion oficial interrumpida o en curso (sin registro de procedencia confirmado)"
    raises_exactly(Err, msg, sal.check_official_publication, p2, root)
    errors, info = m2.verify(root, p2)
    assert errors == [msg] and "salidas_oficiales" not in info
    _assert_valid_after_retry(ctx, monkeypatch)


def test_leftover_staging_after_the_confirmation_is_cleaned_and_never_overwrites(publish_ctx):
    """Muerte despues de promover el registro y antes de retirar la preparacion: la publicacion es valida; el
    reintento retira el resto y rechaza con SALIDA_EXISTENTE sin tocar la publicacion."""
    ctx = publish_ctx
    root, p2, out = ctx["root"], ctx["P2"], ctx["out"]
    rec = sal.official_publish_th35(p2, root)
    before = {n: (out / n).read_bytes() for n in sal.BLOCKED + [sal.PROVENANCE]}
    (out / sal.STAGING).mkdir()
    (out / sal.STAGING / sal.JOURNAL).write_bytes(
        json.dumps({n: hashlib.sha256(b).hexdigest() for n, b in before.items()}).encode("utf-8"))
    assert sal.check_official_publication(p2, root) == rec
    raises_exactly(Err, f"[SALIDA_EXISTENTE] ya existen salidas oficiales (no se sobrescriben): "
                        f"{[sal.CAUSAL, sal.DIAGNOSTIC, sal.PROVENANCE]}", sal.official_publish_th35, p2, root)
    assert not (out / sal.STAGING).exists() and {n: (out / n).read_bytes() for n in before} == before
    assert sal.check_official_publication(p2, root) == rec


_INCOMPLETA = "[SALIDA_INCOMPLETA] publicacion oficial interrumpida o en curso (sin registro de procedencia confirmado)"


def _blocked(names):
    return (f"[SALIDA_RECUPERACION] no puede demostrarse que {names} pertenezcan a la publicacion interrumpida: "
            f"se conservan y la recuperacion queda bloqueada")


def _identical_replacement(path):
    """Otro proceso sustituye ``path`` por un fichero NUEVO con los mismos bytes (otra identidad)."""
    tmp = path.with_name(path.name + ".ajeno")
    tmp.write_bytes(path.read_bytes())
    os.replace(tmp, path)


@pytest.mark.parametrize("alteracion", ["bytes_distintos", "bytes_identicos"])
def test_recovery_never_deletes_files_it_cannot_prove_its_own(publish_ctx, monkeypatch, alteracion):
    """Proceso muerto tras promover la tabla causal; despues otro proceso la altera (bytes distintos) o la SUSTITUYE
    por un fichero con bytes IDENTICOS. Ninguno puede demostrarse de la publicacion interrumpida: la recuperacion no
    borra nada, conserva la preparacion y queda bloqueada (SALIDA_RECUPERACION), tambien en reintentos; la comprobacion
    sigue viendo una publicacion incompleta."""
    ctx = publish_ctx
    root, p2, out = ctx["root"], ctx["P2"], ctx["out"]
    _crash_at(monkeypatch, 1)
    with pytest.raises(_Crash):
        sal.official_publish_th35(p2, root)
    _restore(monkeypatch)
    causal = out / sal.CAUSAL
    if alteracion == "bytes_distintos":
        causal.write_bytes(b"ajeno\n")
    else:
        _identical_replacement(causal)
    before, ident = causal.read_bytes(), (os.stat(causal).st_dev, os.stat(causal).st_ino)
    for _ in range(2):
        raises_exactly(Err, _blocked([sal.CAUSAL]), sal.official_publish_th35, p2, root)
        assert causal.read_bytes() == before and (os.stat(causal).st_dev, os.stat(causal).st_ino) == ident
        assert (out / sal.STAGING / sal.JOURNAL).exists() and not (out / sal.PROVENANCE).exists()
    raises_exactly(Err, _INCOMPLETA, sal.check_official_publication, p2, root)
    errors, _ = m2.verify(root, p2)
    assert errors == [_INCOMPLETA]


@pytest.mark.parametrize("modo", ["fallo_en_el_proceso", "muerte_del_proceso"])
def test_failure_right_after_the_link_never_leaves_a_partial_publication(publish_ctx, monkeypatch, modo):
    """C1: el enlace de la tabla causal se crea y JUSTO DESPUES falla la promocion (en v7, ``os.unlink(src)`` tras
    ``os.link``), antes de que la publicacion pueda anotarlo. Dentro del proceso, el deshacer retira la tabla (su
    propiedad se demuestra por identidad, no por una lista de promovidos); si el proceso muere, la recuperacion la
    retira. En ambos casos el reintento publica una publicacion valida."""
    ctx = publish_ctx
    root, p2, out = ctx["root"], ctx["P2"], ctx["out"]
    real = _REAL["_promote"]

    def promote(src, dst):
        real(src, dst)
        if dst.name == sal.CAUSAL:
            assert dst.exists() and os.path.samefile(src, dst)
            if modo == "muerte_del_proceso":
                raise _Crash()
            raise OSError("fallo tras enlazar (simulado)")
    monkeypatch.setattr(sal, "_promote", promote)
    if modo == "muerte_del_proceso":
        monkeypatch.setattr(sal, "_rollback", lambda out_dir, hashes: None)
        with pytest.raises(_Crash):
            sal.official_publish_th35(p2, root)
        assert [n for n in sal.OFFICIAL if (out / n).exists()] == [sal.CAUSAL]
        raises_exactly(Err, _INCOMPLETA, sal.check_official_publication, p2, root)
    else:
        with pytest.raises(OSError, match=r"^fallo tras enlazar \(simulado\)$"):
            sal.official_publish_th35(p2, root)
        assert [n for n in sal.OFFICIAL if (out / n).exists()] == [] and not (out / sal.STAGING).exists()
    _assert_valid_after_retry(ctx, monkeypatch)


def test_promotion_is_atomic_and_never_overwrites(tmp_path):
    """``_promote`` (os.link en Windows y en POSIX): si el destino no existe lo enlaza y el preparado SIGUE enlazado
    (misma identidad: ancla la propiedad); si existe, FileExistsError y ninguno de los dos ficheros cambia."""
    src, dst = tmp_path / "preparado.csv", tmp_path / "oficial.csv"
    src.write_bytes(b"nuevo\n")
    sal._promote(src, dst)
    assert src.exists() and os.path.samefile(src, dst) and dst.read_bytes() == b"nuevo\n"
    assert sal._owned(src, dst, hashlib.sha256(b"nuevo\n").hexdigest())
    other = tmp_path / "otro.csv"
    other.write_bytes(b"otro\n")
    with pytest.raises(FileExistsError):
        sal._promote(other, dst)
    assert other.read_bytes() == b"otro\n" and dst.read_bytes() == b"nuevo\n" and os.path.samefile(src, dst)


def test_ownership_is_identity_and_bytes(tmp_path):
    """C2: un fichero es propio solo con la MISMA identidad que su preparado Y los bytes del diario."""
    src, dst = tmp_path / "preparado.csv", tmp_path / "oficial.csv"
    src.write_bytes(b"tabla\n")
    digest = hashlib.sha256(b"tabla\n").hexdigest()
    sal._promote(src, dst)
    assert sal._owned(src, dst, digest)
    assert not sal._owned(src, dst, None) and not sal._owned(src, dst, "0" * 64)
    assert not sal._owned(tmp_path / "no_existe", dst, digest) and not sal._owned(src, tmp_path / "no_existe", digest)
    _identical_replacement(dst)                                   # mismos bytes, otro fichero
    assert dst.read_bytes() == b"tabla\n" and not sal._owned(src, dst, digest)


# --------------------------------------------------------------------------- enlaces simbolicos (auditoria v8)
def _symlinks_allowed() -> bool:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        try:
            os.symlink(os.path.join(d, "destino"), os.path.join(d, "enlace"))
        except (OSError, NotImplementedError):
            return False
    return True


_SYMLINKS = _symlinks_allowed()
_MODOS = [pytest.param("real", marks=pytest.mark.skipif(not _SYMLINKS, reason="el sistema no permite crear enlaces "
                                                                               "simbolicos (Windows sin privilegio)")),
          "emulado"]


def _symlink_substitute(path, target, modo, monkeypatch):
    """Otro proceso sustituye ``path`` (atomicamente) por un enlace simbolico hacia ``target``. Devuelve una funcion
    que comprueba que ese sustituto ajeno sigue intacto.
    - real: ``os.symlink`` de verdad.
    - emulado (sistemas sin privilegio de enlaces): un fichero ajeno con los bytes de ``target`` y el sistema de
      ficheros emulado para esa ruta como un enlace: ``os.stat`` sigue al destino (como el sistema operativo) y
      ``os.lstat`` / ``follow_symlinks=False`` lo ven como enlace simbolico (igual ``os.path.exists`` / ``lexists``)."""
    tmp = path.with_name(path.name + ".enlace")
    if modo == "real":
        os.symlink(target, tmp)
        os.replace(tmp, path)
        return lambda: os.path.islink(path) and os.readlink(path) == str(target)
    tmp.write_bytes(target.read_bytes() if target.exists() else b"")
    os.replace(tmp, path)
    real_stat, real_lstat = os.stat, os.lstat
    key = os.path.normcase(os.path.abspath(path))
    mine = lambda p: isinstance(p, (str, os.PathLike)) and os.path.normcase(os.path.abspath(p)) == key  # noqa: E731

    def as_link(p):
        return os.stat_result((stat.S_IFLNK | 0o777,) + tuple(real_lstat(p))[1:10])

    def fake_stat(p, *a, dir_fd=None, follow_symlinks=True):
        if mine(p):
            return real_stat(target) if follow_symlinks else as_link(p)
        return real_stat(p, *a, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    def fake_lstat(p, *a, **k):
        return as_link(p) if mine(p) else real_lstat(p, *a, **k)
    def answers(fn):
        def probe(p):
            try:
                fn(p)
            except (OSError, ValueError):
                return False
            return True
        return probe
    monkeypatch.setattr(os, "stat", fake_stat)
    monkeypatch.setattr(os, "lstat", fake_lstat)
    monkeypatch.setattr(os.path, "exists", answers(fake_stat))       # en Windows son atajos en C que no usan os.stat
    monkeypatch.setattr(os.path, "lexists", answers(fake_lstat))
    before, ident = path.read_bytes(), (real_lstat(path).st_dev, real_lstat(path).st_ino)
    return lambda: path.read_bytes() == before and (real_lstat(path).st_dev, real_lstat(path).st_ino) == ident


@pytest.mark.parametrize("modo", _MODOS)
def test_ownership_rejects_a_symbolic_link_to_the_staged_file(tmp_path, monkeypatch, modo):
    """Auditoria v8: un enlace simbolico hacia el preparado tiene, SIGUIENDO el enlace, su identidad y sus bytes; no es
    el preparado y nunca es propio (la identidad se lee sin seguir enlaces)."""
    src, dst = tmp_path / "preparado.csv", tmp_path / "oficial.csv"
    src.write_bytes(b"tabla\n")
    digest = hashlib.sha256(b"tabla\n").hexdigest()
    sal._promote(src, dst)
    assert sal._owned(src, dst, digest)
    intact = _symlink_substitute(dst, src, modo, monkeypatch)
    assert os.stat(dst).st_ino == os.stat(src).st_ino and dst.read_bytes() == b"tabla\n"   # siguiendo: "el mismo"
    assert not sal._owned(src, dst, digest) and intact()


@pytest.mark.skipif(sys.platform != "win32", reason="union de directorios (junction) de NTFS")
def test_ownership_rejects_other_reparse_points(tmp_path):
    """Otros sustitutos sin privilegio en Windows: una union (punto de reanalisis) en lugar de la tabla no es propia."""
    import _winapi
    src, dst, folder = tmp_path / "preparado.csv", tmp_path / "oficial.csv", tmp_path / "carpeta"
    src.write_bytes(b"tabla\n")
    folder.mkdir()
    _winapi.CreateJunction(str(folder), str(dst))
    assert not sal._plain(os.lstat(dst)) and sal._plain(os.lstat(src))
    assert not sal._owned(src, dst, hashlib.sha256(b"tabla\n").hexdigest()) and os.path.lexists(dst)


@pytest.mark.parametrize("modo", _MODOS)
def test_rollback_never_deletes_a_symbolic_link_to_its_staged_file(publish_ctx, monkeypatch, modo):
    """Auditoria v8: tras promover la tabla causal, otro proceso la sustituye por un enlace simbolico hacia su
    preparado y despues la publicacion falla. El deshacer no borra el enlace ajeno; el reintento no lo sobrescribe."""
    ctx = publish_ctx
    root, p2, out = ctx["root"], ctx["P2"], ctx["out"]
    real, intact = _REAL["_promote"], []

    def promote(src, dst):
        if dst.name == sal.DIAGNOSTIC:
            intact.append(_symlink_substitute(out / sal.CAUSAL, out / sal.STAGING / sal.CAUSAL, modo, monkeypatch))
            raise OSError("fallo tras la sustitucion (simulado)")
        real(src, dst)
    monkeypatch.setattr(sal, "_promote", promote)
    with pytest.raises(OSError, match=r"^fallo tras la sustitucion \(simulado\)$"):
        sal.official_publish_th35(p2, root)
    assert intact[0]() and [n for n in sal.OFFICIAL if os.path.lexists(out / n)] == [sal.CAUSAL]
    _restore(monkeypatch)
    raises_exactly(Err, f"[SALIDA_EXISTENTE] ya existen salidas oficiales (no se sobrescriben): {[sal.CAUSAL]}",
                   sal.official_publish_th35, p2, root)
    assert intact[0]()


@pytest.mark.parametrize("destino", ["preparado", "roto"])
@pytest.mark.parametrize("modo", _MODOS)
def test_recovery_never_deletes_a_symbolic_link(publish_ctx, monkeypatch, modo, destino):
    """Auditoria v8: proceso muerto tras promover la tabla causal; otro proceso la sustituye por un enlace simbolico
    hacia su preparado (o un enlace roto). La recuperacion no puede demostrar que sea suyo: no borra nada, conserva la
    preparacion y queda bloqueada (SALIDA_RECUPERACION), tambien en reintentos."""
    ctx = publish_ctx
    root, p2, out = ctx["root"], ctx["P2"], ctx["out"]
    _crash_at(monkeypatch, 1)
    with pytest.raises(_Crash):
        sal.official_publish_th35(p2, root)
    _restore(monkeypatch)
    target = out / sal.STAGING / sal.CAUSAL if destino == "preparado" else out / "no_existe.csv"
    intact = _symlink_substitute(out / sal.CAUSAL, target, modo, monkeypatch)
    for _ in range(2):
        raises_exactly(Err, _blocked([sal.CAUSAL]), sal.official_publish_th35, p2, root)
        assert intact() and (out / sal.STAGING / sal.JOURNAL).exists() and not (out / sal.PROVENANCE).exists()
    raises_exactly(Err, _INCOMPLETA, sal.check_official_publication, p2, root)
    errors, _ = m2.verify(root, p2)
    assert errors == [_INCOMPLETA] and intact()


# --------------------------------------------------------------------------- verificador y enlaces (auditoria v9)
def _not_regular(n):
    if n == sal.PROVENANCE:
        return f"[SALIDA_REGISTRO] el registro {sal.PROVENANCE} no es un fichero regular (enlace o punto de reanalisis)"
    return f"[SALIDA_HUELLA] {n} no es un fichero regular (enlace o punto de reanalisis)"


@pytest.mark.parametrize("objetivo", ["causal", "diagnostico", "registro"])
@pytest.mark.parametrize("modo", _MODOS)
def test_check_rejects_an_official_file_replaced_by_a_valid_symbolic_link(publish_ctx, monkeypatch, modo, objetivo):
    """Auditoria v9: una tabla oficial (o el registro) sustituida por un enlace simbolico hacia una copia con contenido
    VALIDO (mismos bytes y huella) no es la publicacion: la comprobacion, la reproduccion y ``verify`` la rechazan."""
    ctx = publish_ctx
    root, p2, out = ctx["root"], ctx["P2"], ctx["out"]
    sal.official_publish_th35(p2, root)
    n = {"causal": sal.CAUSAL, "diagnostico": sal.DIAGNOSTIC, "registro": sal.PROVENANCE}[objetivo]
    copy_ = root / "copia_ajena.bin"
    copy_.write_bytes((out / n).read_bytes())
    intact = _symlink_substitute(out / n, copy_, modo, monkeypatch)
    assert (out / n).is_file() and (out / n).read_bytes() == copy_.read_bytes()      # siguiendo el enlace: "valida"
    raises_exactly(Err, _not_regular(n), sal.check_official_publication, p2, root)
    raises_exactly(Err, _not_regular(n), sal.reproduce_official_publication, p2, root)
    errors, _ = m2.verify(root, p2)
    assert errors == [_not_regular(n)] and intact()


@pytest.mark.parametrize("objetivo", ["causal", "registro", "no_registrada"])
@pytest.mark.parametrize("modo", _MODOS)
def test_check_detects_broken_symbolic_links_as_present_outputs(publish_ctx, monkeypatch, modo, objetivo):
    """Auditoria v9: un enlace roto es una salida oficial PRESENTE: en lugar de una tabla registrada o del registro no
    es un fichero regular, y con el nombre de una tabla no registrada para el estado es una salida no autorizada."""
    ctx = publish_ctx
    root, p2, out = ctx["root"], ctx["P2"], ctx["out"]
    sal.official_publish_th35(p2, root)
    extra = [t for t in sal.UNBLOCKED if t not in sal.BLOCKED][0]
    n = {"causal": sal.CAUSAL, "registro": sal.PROVENANCE, "no_registrada": extra}[objetivo]
    intact = _symlink_substitute(out / n, root / "no_existe.csv", modo, monkeypatch)
    assert not (out / n).exists() and os.path.lexists(out / n)
    message = (f"[SALIDA_CONJUNTO] salidas oficiales no registradas para el estado BLOQUEADO_HASTA_EXPLICAR: {[extra]}"
               if objetivo == "no_registrada" else _not_regular(n))
    raises_exactly(Err, message, sal.check_official_publication, p2, root)
    errors, _ = m2.verify(root, p2)
    assert errors == [message] and intact()


@pytest.mark.parametrize("modo", _MODOS)
def test_verify_checks_a_directory_with_only_a_broken_symbolic_link(publish_ctx, monkeypatch, modo):
    """Auditoria v9: sin publicacion, un enlace roto con el nombre de una salida oficial obliga a comprobar (y falla)."""
    ctx = publish_ctx
    root, p2, out = ctx["root"], ctx["P2"], ctx["out"]
    out.mkdir(parents=True, exist_ok=True)
    assert m2.verify(root, p2)[0] == []
    intact = _symlink_substitute(out / sal.CAUSAL, root / "no_existe.csv", modo, monkeypatch)
    errors, _ = m2.verify(root, p2)
    assert errors == [f"[SALIDA_REGISTRO] falta o no es JSON valido el registro {sal.PROVENANCE}"] and intact()


@pytest.mark.parametrize("caso", ["causal", "diagnostico", "registro", "diagnostico_y_reemplazo",
                                  "diagnostico_y_reemplazo_identico"])
def test_destination_created_after_the_check_is_never_overwritten(publish_ctx, monkeypatch, caso):
    """Otro proceso, sin el cerrojo, crea un destino DESPUES de SALIDA_EXISTENTE y antes de su promocion (y, en el
    ultimo caso, ademas sustituye la tabla ya promovida). La publicacion falla con SALIDA_EXISTENTE, los ficheros ajenos
    quedan intactos, lo promovido por esta publicacion se retira y no queda ninguna publicacion valida."""
    ctx = publish_ctx
    root, p2, out = ctx["root"], ctx["P2"], ctx["out"]
    target = {"causal": sal.CAUSAL, "diagnostico": sal.DIAGNOSTIC, "registro": sal.PROVENANCE,
              "diagnostico_y_reemplazo": sal.DIAGNOSTIC, "diagnostico_y_reemplazo_identico": sal.DIAGNOSTIC}[caso]
    foreign = {target: b"ajeno\n"}
    if caso == "diagnostico_y_reemplazo":
        foreign[sal.CAUSAL] = b"ajeno, sustituye a la tabla promovida\n"
    real = _REAL["_promote"]

    def promote(src, dst):
        if dst.name == target and not dst.exists():
            assert sal.CAUSAL == target or (out / sal.CAUSAL).exists()       # ya paso la comprobacion inicial
            if caso == "diagnostico_y_reemplazo_identico":                   # C2: mismos bytes, fichero ajeno
                foreign[sal.CAUSAL] = (out / sal.CAUSAL).read_bytes()
                _identical_replacement(out / sal.CAUSAL)
            for n, b in foreign.items():
                if not (n == sal.CAUSAL and caso == "diagnostico_y_reemplazo_identico"):
                    (out / n).write_bytes(b)
        real(src, dst)
    monkeypatch.setattr(sal, "_promote", promote)
    raises_exactly(Err, f"[SALIDA_EXISTENTE] {target} aparecio durante la publicacion (no se sobrescribe)",
                   sal.official_publish_th35, p2, root)
    assert {n: (out / n).read_bytes() for n in sal.OFFICIAL if (out / n).exists()} == foreign
    assert not (out / sal.STAGING).exists()
    raises_exactly(Err, f"[SALIDA_REGISTRO] falta o no es JSON valido el registro {sal.PROVENANCE}",
                   sal.check_official_publication, p2, root)
    _restore(monkeypatch)
    raises_exactly(Err, f"[SALIDA_EXISTENTE] ya existen salidas oficiales (no se sobrescriben): "
                        f"{[n for n in sal.OFFICIAL if n in foreign]}", sal.official_publish_th35, p2, root)
    assert {n: (out / n).read_bytes() for n in sal.OFFICIAL if (out / n).exists()} == foreign


def test_two_simultaneous_publications_in_one_process(publish_ctx, monkeypatch):
    """La primera publicacion se detiene en la preparacion con el cerrojo tomado; la segunda rechaza con
    SALIDA_CONCURRENTE sin escribir; la primera termina y un reintento posterior rechaza con SALIDA_EXISTENTE."""
    ctx = publish_ctx
    root, p2, out = ctx["root"], ctx["P2"], ctx["out"]
    inside, release, outcome = threading.Event(), threading.Event(), {}
    real = sal._write_durable

    def write(path, data):
        if not inside.is_set():
            inside.set()
            assert release.wait(600)
        real(path, data)
    monkeypatch.setattr(sal, "_write_durable", write)

    def first():
        try:
            outcome["rec"] = sal.official_publish_th35(p2, root)
        except BaseException as exc:                     # noqa: BLE001 (se comprueba abajo)
            outcome["exc"] = exc
    t = threading.Thread(target=first)
    t.start()
    try:
        assert inside.wait(600)
        raises_exactly(Err, "[SALIDA_CONCURRENTE] otra publicacion oficial esta en curso",
                       sal.official_publish_th35, p2, root)
        assert [n for n in sal.OFFICIAL if (out / n).exists()] == []
    finally:
        release.set()
        t.join(600)
    assert "exc" not in outcome and sal.check_official_publication(p2, root) == outcome["rec"]
    raises_exactly(Err, f"[SALIDA_EXISTENTE] ya existen salidas oficiales (no se sobrescriben): "
                        f"{[sal.CAUSAL, sal.DIAGNOSTIC, sal.PROVENANCE]}", sal.official_publish_th35, p2, root)


def test_a_publication_in_another_process_excludes_this_one_and_its_death_releases_the_lock(publish_ctx):
    """Otro proceso con el cerrojo tomado: SALIDA_CONCURRENTE. Al matarlo, el sistema libera el cerrojo (no queda
    huerfano) y la publicacion procede."""
    ctx = publish_ctx
    root, p2, out = ctx["root"], ctx["P2"], ctx["out"]
    out.mkdir(parents=True, exist_ok=True)
    code = ("import sys; sys.path[:0] = %r\n"
            "from pathlib import Path\n"
            "from ohlcv_dataroad.ingest import tda17_salidas as sal\n"
            "with sal._exclusive(Path(%r)):\n"
            "    print('CERROJO', flush=True)\n"
            "    sys.stdin.read()\n") % ([str(SRC / "src")], str(out))
    child = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "CERROJO"
        raises_exactly(Err, "[SALIDA_CONCURRENTE] otra publicacion oficial esta en curso",
                       sal.official_publish_th35, p2, root)
        assert [n for n in sal.OFFICIAL if (out / n).exists()] == []
    finally:
        child.kill()
        child.wait(60)
    rec = sal.official_publish_th35(p2, root)
    assert sal.check_official_publication(p2, root) == rec


def test_line_ending_changes_are_detected_strictly(publish_ctx):
    """Tabla convertida a CRLF: la huella (bytes exactos) la detecta; con un registro coherente con los bytes CRLF, la
    reproduccion la rechaza porque compara byte a byte sin normalizar."""
    ctx = publish_ctx
    root, p2, out = ctx["root"], ctx["P2"], ctx["out"]
    rec = sal.official_publish_th35(p2, root)
    causal = out / sal.CAUSAL
    lf = causal.read_bytes()
    assert b"\r" not in lf
    crlf = lf.replace(b"\n", b"\r\n")
    causal.write_bytes(crlf)
    raises_exactly(Err, f"[SALIDA_HUELLA] {sal.CAUSAL} falta o no tiene la huella registrada",
                   sal.check_official_publication, p2, root)
    forged = copy.deepcopy(rec)
    forged["tablas"][sal.CAUSAL] = hashlib.sha256(crlf).hexdigest()
    _write_record(ctx, forged)
    assert sal.check_official_publication(p2, root) == forged
    raises_exactly(Err, f"[SALIDA_NO_REPRODUCIBLE] tablas publicadas distintas del recalculo oficial: {[sal.CAUSAL]}",
                   sal.reproduce_official_publication, p2, root)
    causal.write_bytes(lf)
    _write_record(ctx, rec)
    assert sal.reproduce_official_publication(p2, root) == rec


def test_other_tables_stay_blocked_and_the_route_is_unique():
    for n in sal.UNBLOCKED + [sal.DIAGNOSTIC]:
        raises_exactly(Err, f"[SALIDA_RUTA_UNICA] {n} solo se publica con official_publish_th35", sal.other_official_table, n)
    raises_exactly(an.PendingDefinitionError,
                   f"SALIDAS_OTRAS_TABLAS: definicion pendiente -- {an.PENDIENTES['SALIDAS_OTRAS_TABLAS']}",
                   sal.other_official_table, "TDA17_sensibilidades.csv")
    assert ca.verify_causal_audit is not None and not hasattr(sal, "write_official_table")
