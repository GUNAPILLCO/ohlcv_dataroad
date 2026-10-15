"""Tests documentales del cierre de TDA-15 por STOP-15a (sin ejecucion estadistica).

Comprueban la trazabilidad (cada evidencia de `configs/tda15_cierre.yaml` aparece literalmente en su fuente),
la concordancia entre el YAML y `reports/mnq/TDA15_no_linealidad_residual.md`, y que el cierre no fabrica
resultados. Solo leen documentos versionados: ningun test lee data/ ni el hold-out.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CLOSURE = ROOT / "configs" / "tda15_cierre.yaml"
REPORT = ROOT / "reports" / "mnq" / "TDA15_no_linealidad_residual.md"
VERDICT = ("TDA-15: CERRADA METODOLÓGICAMENTE — STOP-15a ACTIVADO — ANÁLISIS NO EJECUTADO.", "TH32: NO EVALUADA.",
           "TH33: NO EVALUADA.", "DECISIONES DE DISEÑO IRIS: NINGUNA.")
# Categorias que el roadmap reserva para pruebas EJECUTADAS (no aplicables a un cierre sin ejecucion).
EXECUTED_TEST_VERDICTS = ("SIN EVIDENCIA", "EVIDENCIA INESTABLE", "EVIDENCIA ESTABLE, FORMA NO IDENTIFICADA")


@pytest.fixture(scope="module")
def closure() -> dict:
    return yaml.safe_load(CLOSURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def report() -> str:
    return REPORT.read_text(encoding="utf-8")


def _all_evidence(c: dict) -> list[dict]:
    return list(c["evidencias"]) + list(c["contexto_no_normativo"])


def test_every_evidence_is_a_literal_quote_of_a_versioned_source(closure):
    seen = set()
    for ev in _all_evidence(closure):
        assert ev["id"] not in seen, ev["id"]
        seen.add(ev["id"])
        src = ROOT / ev["fuente"]
        assert src.is_file(), ev["fuente"]
        assert not ev["fuente"].startswith("data/"), ev["fuente"]
        assert ev["cita"] in src.read_text(encoding="utf-8"), f"{ev['id']}: cita no encontrada en {ev['fuente']}"


def test_conditions_reference_existing_evidence_and_A_blocks(closure):
    ids = {ev["id"]: ev for ev in closure["evidencias"]}
    cond = closure["condiciones_entrada"]
    assert cond["A_residuo_limpio"]["estado"] == "NO CUMPLIDA"
    assert cond["B_estabilidad_temporal"]["estado"] == "CUMPLIDA FORMALMENTE, CON LIMITACIONES"
    for key in ("C_pregunta_cientifica", "D_multiplicidad_y_calibracion"):
        assert cond[key]["estado"] == "PENDIENTE — NO DESARROLLADA" and "evidencias" not in cond[key]
    for key, letter in (("A_residuo_limpio", "A"), ("B_estabilidad_temporal", "B")):
        refs = cond[key]["evidencias"]
        assert refs and all(ids[r]["condicion"] == letter for r in refs)
    assert {r for k in ("A_residuo_limpio", "B_estabilidad_temporal") for r in cond[k]["evidencias"]} == set(ids)


def test_hypotheses_are_not_evaluated_and_verdict_is_registered(closure):
    assert closure["decision"] == "STOP-15a: NO EJECUTAR" and closure["decisiones_diseno_iris"] == "NINGUNA"
    assert closure["hipotesis_estado"]["TH32"]["estado"] == "NO EVALUADA POR STOP-15a"
    assert closure["hipotesis_estado"]["TH33"]["estado"] == "NO EVALUADA POR DEPENDENCIA DE TH32"
    assert tuple(closure["veredicto"]) == VERDICT
    assert closure["estado"] + "." == VERDICT[0].removeprefix("TDA-15: ")


def test_report_agrees_with_closure_file(closure, report):
    for line in VERDICT:
        assert line in report, line
    for c in closure["condiciones_entrada"].values():
        assert f"**{c['estado']}**" in report, c["estado"]
    for h in closure["hipotesis_estado"].values():
        assert h["estado"] in report and h["detalle"].strip() in " ".join(report.split())
    for item in closure["no_realizado"] + closure["restricciones_heredadas"]:
        assert f"- {item}" in report, item
    cited = set(re.findall(r"\b([EX]\d{2})\b", report))
    assert cited == {ev["id"] for ev in _all_evidence(closure)}            # el informe cita exactamente las evidencias del YAML


def test_closure_fabricates_no_results(closure, report):
    for term in EXECUTED_TEST_VERDICTS:
        assert term not in report and term not in CLOSURE.read_text(encoding="utf-8"), term
    assert not [p.name for p in (ROOT / "reports" / "mnq").glob("TDA15_*") if p.suffix != ".md"]   # sin CSV, graficos ni tablas
    assert "0,02" not in closure["condiciones_entrada"]["A_residuo_limpio"]["resumen"]             # el umbral de TDA-09 no es regla de TDA-15
    assert "NO fue predeclarado como criterio de aceptación de TDA-15" in closure["contexto_no_normativo"][0]["lectura"]
    assert not ({"protocolo", "calibracion", "umbrales", "analisis"} & set(closure))               # sin parametros estadisticos
