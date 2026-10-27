"""Pruebas SINTETICAS del motor paralelo E03 (``ohlcv_dataroad.tda17_e03.motor``).

Criterio: el motor E03 produce EXACTAMENTE el resultado del motor E02 (``an.th35_pipeline``) en el mismo entorno
numerico: tablas oficiales identicas byte a byte y resultado completo identico (sin tolerancias). Ademas: datos de
TH37 compartidos de solo lectura e identicos bit a bit, sorteos en el proceso principal, orden de agregacion de E02,
invariancia al numero de workers, al reparto y al orden de finalizacion, control de memoria (privada + compartida
contada una vez), hilos BLAS fijados y verificados, progreso, estimacion del tiempo restante, puntos de control y
reanudacion SINTETICA, reanudacion OFICIAL deshabilitada y propagacion de errores. Sin MNQ ni holdout.
"""
from __future__ import annotations

import copy
import inspect
import json
import os
import pickle
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))             # tests/ (ayudantes sinteticos de E01/E02)

from ohlcv_dataroad.ingest import tda17_analysis as an  # noqa: E402
from ohlcv_dataroad.ingest import tda17_causalidad as ca  # noqa: E402
from ohlcv_dataroad.ingest import tda17_enmiendas_e02 as m2  # noqa: E402
from ohlcv_dataroad.ingest import tda17_indicators as ind  # noqa: E402
from ohlcv_dataroad.ingest import tda17_salidas as sal  # noqa: E402
from ohlcv_dataroad.tda17_e03 import motor as mo  # noqa: E402
from e03_equivalencia import deep_equal, pickle_sha  # noqa: E402
from tda17_synthetic import make_frame  # noqa: E402
from test_tda17_procedencia import _engine_input  # noqa: E402

SRC = m2.ROOT
HELPER = SRC / "tests" / "tda17_e03" / "e03_equivalencia.py"


def raises_exactly(exc_type, message, fn, *args, **kwargs):
    with pytest.raises(exc_type) as info:
        fn(*args, **kwargs)
    assert type(info.value) is exc_type and str(info.value) == message, f"{type(info.value).__name__}: {info.value}"


@pytest.fixture(scope="module")
def cfg():
    return copy.deepcopy(m2.effective_document_e02(borrador=True))


def _motor(path: Path, procesos: int = 2, **kw) -> mo._Motor:
    ej = mo.Ejecucion(path, {"prueba": path.name})
    return mo._Motor(mo.Recursos(procesos=procesos, intervalo_estado_s=0.2, **kw), ej)


def _tables(res, c):
    return {n: sal.csv_bytes(t) for n, t in sal.official_tables(res, c).items()}


def _events(d: Path) -> list[dict]:
    return [json.loads(x) for x in (d / "progreso.jsonl").read_text(encoding="utf-8").splitlines()]


def _subprocess(args: list[str], env_extra: dict | None, tmp_path: Path, drop_blas: bool = False):
    env = {k: v for k, v in os.environ.items() if not (drop_blas and k in mo.BLAS_VARIABLES)}
    env.update({"PYTHONPATH": os.pathsep.join(sys.path), **(env_extra or {})})
    proc = subprocess.run([sys.executable, str(HELPER), *args], cwd=SRC, env=env, capture_output=True, text=True,
                          timeout=5400)
    assert proc.returncode == 0, proc.stderr[-4000:]


# =========================================================================== equivalencia extremo a extremo
def test_end_to_end_equivalence_in_a_fixed_numeric_environment(tmp_path):
    """Subproceso con OPENBLAS_NUM_THREADS=1 (principal y workers con los mismos hilos): casos bloqueado y sin bloqueo;
    E03 con 0, 1, 2 y 4 procesos, bloques de cortes 1/3/7/1000, bloques de replicas 1/2/3 y orden de despacho
    barajado; datos de TH37 compartidos; interrupcion simulada y reanudacion verificada (sintetica). Tablas y resultado
    completo identicos a E02."""
    out = tmp_path / "informe.json"
    _subprocess([str(out), str(tmp_path)], {k: "1" for k in mo.BLAS_VARIABLES}, tmp_path)
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert set(rep["hilos_blas"].values()) == {1}
    assert {n: c["estado_E02"] for n, c in rep["casos"].items()} == {"bloqueado": "BLOQUEADO_HASTA_EXPLICAR",
                                                                     "sin_bloqueo": "SIN_BLOQUEO"}
    for nombre, caso in rep["casos"].items():
        assert len(caso["E03"]) == 4
        for run in caso["E03"]:
            assert run["identicas"] and run["tablas"] == caso["tablas_E02"], (nombre, run)
            assert run["diferencias_profundas"] == [] and run["completa"], (nombre, run)
        assert max(r["max_workers_activos"] for r in caso["E03"]) >= 2
        re_ = caso["reanudacion"]
        assert re_["interrupcion"].startswith("[E03_INTERRUPCION_SIMULADA]") and re_["sin_COMPLETA_tras_interrupcion"]
        assert re_["reutilizadas"] == 6 and re_["identicas"] and re_["diferencias_profundas"] == []
        assert re_["completa"] and re_["estado_final"] == "COMPLETA"


def test_end_to_end_equivalence_in_the_inherited_environment(tmp_path, cfg):
    """Mismo entorno numerico que el proceso de pruebas (hilos BLAS heredados por los workers y verificados)."""
    c = copy.deepcopy(cfg)
    c["analisis"]["bootstrap"]["replicas"] = 3
    df = _engine_input()
    ref = an.th35_pipeline(df, c)
    res, resumen = mo.run_th35_e03(df, c, tmp_path / "ej", mo.Recursos(procesos=2, replicas_por_bloque=3))
    assert _tables(res, c) == _tables(ref, c) and deep_equal(ref, res) == []
    assert resumen["hilos_blas"] == mo.hilos_blas_efectivos() and resumen["max_workers_activos"] == 2
    assert resumen["datos_compartidos"] == ["th37"] and resumen["medida_memoria"] == "PRIVADA"


def test_end_to_end_equivalence_with_one_blas_thread_fixed_at_runtime(tmp_path, cfg):
    """Ruta oficial: 1 hilo fijado en tiempo de ejecucion en el proceso principal (``hilos_blas_fijados``); los
    workers arrancan con 1 hilo y se verifican. Mismo resultado que el motor E02 con el mismo hilo fijado."""
    c = copy.deepcopy(cfg)
    c["analisis"]["bootstrap"]["replicas"] = 3
    df = _engine_input()
    antes = mo.hilos_blas_efectivos()
    with mo.hilos_blas_fijados(1) as efectivos:
        assert set(efectivos.values()) == {1} and sorted(efectivos) == sorted(antes)
        ref = an.th35_pipeline(df, c)
        res, resumen = mo.run_th35_e03(df, c, tmp_path / "ej", mo.Recursos(procesos=2, replicas_por_bloque=2))
    assert mo.hilos_blas_efectivos() == antes                                    # restaurados
    assert _tables(res, c) == _tables(ref, c) and deep_equal(ref, res) == []
    assert resumen["hilos_blas"] == efectivos
    done = [e for e in _events(tmp_path / "ej") if e["evento"] == "tarea_completa"]
    assert done and all(e["hilos_blas"] == efectivos for e in done)


def test_blas_threads_fixed_at_runtime_equal_threads_fixed_at_startup(tmp_path):
    """El motor E02 da las MISMAS tablas con OPENBLAS_NUM_THREADS=1 al arrancar que con los hilos por defecto y 1 hilo
    fijado en tiempo de ejecucion (la ruta oficial de E03 y su reproduccion con el motor E02 usan lo segundo)."""
    a, b = tmp_path / "arranque.json", tmp_path / "fijado.json"
    _subprocess(["--tablas-e02", str(a)], {k: "1" for k in mo.BLAS_VARIABLES}, tmp_path)
    _subprocess(["--tablas-e02", str(b), "1"], None, tmp_path, drop_blas=True)
    ra, rb = json.loads(a.read_text(encoding="utf-8")), json.loads(b.read_text(encoding="utf-8"))
    assert set(ra["hilos_blas"].values()) == {1} and set(rb["hilos_blas"].values()) == {1} and rb["fijados"] == 1
    assert ra["casos"] == rb["casos"] and set(ra["casos"]) == {"bloqueado", "sin_bloqueo"}


def test_blas_threads_cannot_be_fixed_without_a_verifiable_openblas(monkeypatch):
    monkeypatch.setattr(mo, "_openblas", lambda: [])
    with pytest.raises(mo.MotorE03Error) as info:
        with mo.hilos_blas_fijados(1):
            pass
    assert str(info.value) == "[E03_BLAS] no se pueden fijar ni verificar los hilos de OpenBLAS de este proceso: []"


def test_workers_use_the_effective_blas_threads_of_the_main_process_not_the_environment(tmp_path, cfg, monkeypatch):
    """Una variable de entorno cambiada despues de importar numpy (p. ej. por otra prueba) no altera los hilos de los
    workers: arrancan con los EFECTIVOS del principal y lo verifican."""
    actual = max(mo.hilos_blas_efectivos().values())
    distinto = str(actual - 1) if actual > 1 else "2"
    for k in mo.BLAS_VARIABLES:
        monkeypatch.setenv(k, distinto)
    motor = _motor(tmp_path / "ej", 2)
    try:
        motor.ejecutar("B", _tiny_tasks(motor, cfg))
    finally:
        motor.cerrar()
    assert all(e["hilos_blas"] == mo.hilos_blas_efectivos() for e in _events(tmp_path / "ej")
               if e["evento"] == "tarea_completa")


def test_forced_blas_threads_are_recorded_in_the_identity(cfg):
    """``hilos_blas`` forzado (solo benchmark) cambia el entorno numerico de los workers: queda en la identidad de la
    ejecucion, de modo que sus puntos de control nunca se mezclan con los de una ejecucion con hilos heredados."""
    rec = mo.Recursos(procesos=1, hilos_blas=1)
    ident = mo.identidad_ejecucion(_engine_input(bars_per_day=40), cfg, rec)
    assert ident["hilos_blas_forzados"] == 1
    assert mo.identidad_ejecucion(_engine_input(bars_per_day=40), cfg, mo.Recursos())["hilos_blas_forzados"] is None


# =========================================================================== TH37 y datos compartidos
@pytest.fixture(scope="module")
def th37_frame():
    return make_frame(contracts=(("MNQ_A", 2), ("MNQ_B", 2), ("MNQ_C", 1)), bars_per_day=200,
                      dense_gaps=(40, 80)).drop(columns=[ind.SYNTHETIC_MARKER]).reset_index(drop=True)


@pytest.mark.parametrize("procesos,cortes", [(0, 1000), (2, 1), (3, 4)])
def test_th37_decomposition_equals_run_causal_audit(tmp_path, cfg, th37_frame, procesos, cortes):
    ref = ca.run_causal_audit(th37_frame, cfg)
    motor = _motor(tmp_path / "ej", procesos, cortes_por_bloque=cortes, barajar=procesos or None)
    try:
        got = mo.causal_audit_e03(th37_frame, cfg, motor)
    finally:
        motor.cerrar()
    pd.testing.assert_frame_equal(got, ref, check_exact=True)
    assert sal.csv_bytes(got) == sal.csv_bytes(ref)
    assert ca.verify_causal_audit(got, th37_frame, cfg) == ca.verify_causal_audit(ref, th37_frame, cfg)
    n_roll = len(ind.contract_segments(th37_frame))
    assert motor.etapas[0].total == sum(len(mo._bloques(list(p["cortes"].items()), cortes))
                                        for p in ca.contract_cut_plan(th37_frame, *ca._catalog(th37_frame), cfg)) + 1 + n_roll
    assert "th37" in motor.compartido_cargado


def test_shared_frames_are_bit_identical_read_only_and_not_copied(tmp_path, th37_frame):
    """Marco de entrada y catalogo completo: misma estructura de bloques, tipos, columnas, indice y bits; cada bloque
    numerico es una vista de solo lectura de la proyeccion del fichero (sin copia) y escribir en el falla."""
    full = ca._catalog(th37_frame)
    ej = mo.Ejecucion(tmp_path / "ej", {"prueba": 1})
    ref, mb = mo.compartir_marcos(ej, "th37", (th37_frame, *full))
    assert mb == pytest.approx(sum(b.values.nbytes for f in (th37_frame, *full) for b in f._mgr.blocks
                                   if mo._compartible(b.values)) / mo.MB)
    got = mo._abrir_marcos(ref)
    for a, b in zip((th37_frame, *full), got):
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        assert mo._estructura(a) == mo._estructura(b) and list(a.dtypes) == list(b.dtypes)
        shared = [blk.values for blk in b._mgr.blocks if type(blk.values) is np.ndarray and not blk.values.flags.writeable]
        assert len(shared) == sum(1 for blk in a._mgr.blocks if mo._compartible(blk.values)) > 0
        assert all(mo._respaldado_por_fichero(v) for v in shared)
        with pytest.raises(ValueError, match="read-only"):
            shared[0][0, 0] = shared[0][0, 0]
    proyectados = sum(1 for f in got for blk in f._mgr.blocks if type(blk.values) is np.ndarray and not blk.values.flags.writeable)
    assert proyectados == sum(1 for f in (th37_frame, *full) for blk in f._mgr.blocks if mo._compartible(blk.values))
    assert sorted(p.suffix for p in (tmp_path / "ej" / "datos").iterdir()) == \
        [".npy"] * proyectados + [".pkl"]


def test_shared_frames_give_the_same_th37_rows_as_private_frames(tmp_path, cfg, th37_frame):
    """``audit_future`` y ``audit_parameter_origin`` sobre los marcos proyectados = sobre los originales."""
    full = ca._catalog(th37_frame)
    ej = mo.Ejecucion(tmp_path / "ej", {"prueba": 1})
    ref, _ = mo.compartir_marcos(ej, "th37", (th37_frame, *full))
    df_s, s_s, m_s = mo._abrir_marcos(ref)
    plan = [{**p, "cortes": dict(list(p["cortes"].items())[:3])} for p in ca.contract_cut_plan(th37_frame, *full, cfg)]
    assert pickle_sha(ca.audit_future(df_s, cfg, plan, (s_s, m_s))) == pickle_sha(ca.audit_future(th37_frame, cfg, plan, full))
    assert pickle_sha(ca.audit_parameter_origin(df_s, cfg, (s_s, m_s))) == pickle_sha(ca.audit_parameter_origin(th37_frame, cfg, full))


def test_a_modified_shared_block_is_rejected_in_the_main_process_and_in_the_workers(tmp_path, cfg, th37_frame):
    full = ca._catalog(th37_frame)
    motor = _motor(tmp_path / "ej", 1)
    ref, _ = mo.compartir_marcos(motor.ej, "th37", (th37_frame, *full))
    victim = sorted((tmp_path / "ej" / "datos").glob("th37_1_*.npy"))[0]
    data = bytearray(victim.read_bytes())
    data[-1] ^= 1
    victim.write_bytes(bytes(data))
    raises_exactly(mo.MotorE03Error, f"[E03_DATOS] fichero de datos del worker alterado: {victim}", mo._abrir_marcos, ref)
    with pytest.raises(mo.MotorE03Error) as info:
        motor.ejecutar("TH37", [mo.Tarea("TH37O", "_t_th37_origin", (ref, cfg), "TH37O")])
    assert str(info.value) == f"[E03_DATOS] fichero de datos del worker alterado: {victim}" and motor.workers == []


def test_a_reconstruction_that_differs_from_the_original_is_rejected(tmp_path, th37_frame):
    full = ca._catalog(th37_frame)
    otro = full[0].copy()
    otro.iloc[-1, 0] = np.nextafter(otro.iloc[-1, 0], np.inf)                 # un bit distinto (valor finito)
    assert np.isfinite(full[0].iloc[-1, 0])
    ref, _ = mo.compartir_marcos(mo.Ejecucion(tmp_path / "ej", {"p": 1}), "x", (otro,))
    raises_exactly(mo.MotorE03Error, "[E03_COMPARTIDO] marco 0, bloque 0: contenido distinto del original",
                   mo._verificar_reconstruccion, (full[0],), mo._abrir_marcos(ref))
    raises_exactly(mo.MotorE03Error, "[E03_COMPARTIDO] marco 0, bloque 0: no es una proyeccion de solo lectura",
                   mo._verificar_reconstruccion, (full[0],), (full[0].copy(),))
    raises_exactly(mo.MotorE03Error, "[E03_COMPARTIDO] marco 0: estructura reconstruida distinta de la original",
                   mo._verificar_reconstruccion, (th37_frame,), (th37_frame.astype({"volume": "float32"}),))


# =========================================================================== TH35: bootstrap y validacion
def _moments_case(cfg, n_days=40, rows=30, p=6, seed=3, constant_col=True):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_days * rows, p))
    codes = np.repeat(np.arange(n_days), rows)
    if constant_col:
        X[codes != 0, 0] = 1.5                                                 # VAR_CERO si la replica no tiene el dia 0
    reg = sorted(an.registered_columns(cfg))
    names = [reg[0]] + [f"c{j}" for j in range(1, p)]
    point = pd.DataFrame({"columna": names, "r2": np.linspace(0.1, 0.6, p), "publicable": [True] * (p - 1) + [False]})
    return X, codes, names, point


@pytest.mark.parametrize("procesos,por_bloque", [(0, 7), (2, 1), (3, 4)])
def test_bootstrap_r2_blocks_equal_bootstrap_r2(tmp_path, cfg, procesos, por_bloque):
    X, codes, names, point = _moments_case(cfg)
    m = an.daily_moments(X, codes)
    draws = an.cell_draws(cfg, "GLOBAL", m.n.size, bloque=5, replicas=None)[:23]
    ref = an.bootstrap_r2(m, draws, names, point, cfg)
    assert (ref["ic_estado"] == "IC_NO_CALCULABLE").any() and (ref["ic_estado"] == "DEGENERADO_REGISTRO").any()
    motor = _motor(tmp_path / "ej", procesos, barajar=5)
    try:
        tau, q, reg = mo._r2_config(cfg)
        todo = [j for j, nm in enumerate(names) if nm not in reg]
        mref = motor.ej.dato("m.pkl", m)
        rb = motor.ejecutar("B", mo.r2_block_tasks("GLOBAL", mref, draws, len(names), todo, tau, por_bloque, m.n.size))
    finally:
        motor.cerrar()
    vals, cause, base = mo.r2_from_blocks("GLOBAL", rb, len(draws), por_bloque)
    got = mo._r2_aggregate(vals, cause, base, names, point, q, reg, len(draws))
    pd.testing.assert_frame_equal(got, ref, check_exact=True)
    assert np.array_equal(got.attrs["tamano_base"], ref.attrs["tamano_base"]) and got.attrs["cuantiles"] == ref.attrs["cuantiles"]


def _validation_inputs(n_days=40, rows=25, p=5, seed=8, invalid=False):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B").date
    codes = np.repeat(np.array(dates, dtype=object), rows)
    X = rng.normal(size=(n_days * rows, p)) + np.repeat(rng.normal(size=(n_days, p)), rows, axis=0)
    if invalid:
        X[codes != dates[0], 1] = 2.0
    panel = pd.DataFrame(X, columns=[f"c{j}" for j in range(p)])
    df = pd.DataFrame({"trading_date": codes})
    masks = {"GLOBAL": np.ones(len(df), bool), "ANIO_2024": np.isin(codes, dates[::2])}
    return panel, df, masks


@pytest.mark.parametrize("variant", ["pasa", "ampliada_y_falla", "replica_invalida"])
def test_spearman_validation_equals_e02(tmp_path, cfg, variant):
    c = copy.deepcopy(cfg)
    panel, df, masks = _validation_inputs(invalid=variant == "replica_invalida")
    if variant == "ampliada_y_falla":
        c["analisis"]["spearman"]["validacion"]["tolerancia_mediana_dif_ic"] = 0.0
    vcells = an.e02_value(c, "analisis.spearman.validacion_detalle")["celdas"]
    masks = {k: masks[k] for k in vcells}
    data = {k: (panel[mk].to_numpy(np.float64), df.loc[mk, "trading_date"].to_numpy()) for k, mk in masks.items()}
    ref = an.spearman_validation(data, c)
    motor = _motor(tmp_path / "ej", 3, barajar=9)
    try:
        got = mo.spearman_validation_e03(panel, df, masks, c, motor)
    finally:
        motor.cerrar()
    assert pickle_sha(got) == pickle_sha(ref) and repr(got) == repr(ref)
    k1, k2 = c["analisis"]["spearman"]["validacion"]["replicas_exactas"], \
        c["analisis"]["spearman"]["validacion"]["alternativa_si_falla"]["replicas_exactas"]
    totals = {e.nombre: e.total for e in motor.etapas}
    if variant == "pasa":
        assert ref["resultado"] == "VALIDADA" and totals[f"TH35_VALIDACION_{k1}"] == 2 * k1 and f"TH35_VALIDACION_{k2}" not in totals
    else:
        assert ref["resultado"] == "NO_VALIDADA" and totals[f"TH35_VALIDACION_{k2}"] == 2 * (k2 - k1)   # prefijo reutilizado
    if variant == "replica_invalida":
        assert all(r["replicas_invalidas"] for r in ref["rondas"][0]["celdas"])


def test_draws_are_generated_only_in_the_main_process():
    for name, fn in mo.TASKS.items():
        src = inspect.getsource(fn)
        for token in ("cell_draws", "SeedSequence", "default_rng", "Generator", "PCG64", "random"):
            assert token not in src, (name, token)
    assert "an.cell_draws(" in inspect.getsource(mo._cell_draws)
    assert "_cell_draws(cfg, cell, n_days)" in inspect.getsource(mo.th35_pipeline_e03)
    assert "_cell_draws(cfg, c, n_days, replicas=k)" in inspect.getsource(mo.spearman_validation_e03)


def test_th35_pipeline_e03_keeps_the_e02_order_and_barrier():
    src = inspect.getsource(mo.th35_pipeline_e03)
    order = [src.index(t) for t in ("causal_audit_e03(df, cfg, motor)", "ca.verify_causal_audit(table, df, cfg)",
                                    "ind.compute_catalog(df)", "ind.build_panel_p(series, df)", "an.cell_definitions(",
                                    'motor.ejecutar("TH35_CELDAS"', 'motor.ejecutar("TH35_BOOTSTRAP_R2"',
                                    "an._cell_rows(cell", "spearman_validation_e03(", "an.classify_columns(per_cell, cfg, blocks)",
                                    "an.subniveles(r2_global, cfg, blocks)")]
    assert order == sorted(order)
    assert "for cell in stream:" in src and "an.bootstrap_stream_cells(cfg)" in src


# =========================================================================== robustez
def _tiny_tasks(motor, cfg, n=6):
    X, codes, names, _ = _moments_case(cfg, n_days=12, rows=10, p=4, constant_col=False)
    m = an.daily_moments(X, codes)
    draws = an.cell_draws(cfg, "GLOBAL", m.n.size, bloque=5)[:n]
    ref = motor.ej.dato("m.pkl", m)
    return mo.r2_block_tasks("GLOBAL", ref, draws, len(names), list(range(len(names))), 1e-10, 1, m.n.size)


def _th37_ref(motor, frame):
    return mo.compartir_marcos(motor.ej, "th37", (frame, *ca._catalog(frame)))[0]


def test_worker_error_propagates_with_the_same_type_and_message(tmp_path, cfg, th37_frame):
    bad = copy.deepcopy(cfg)
    bad["analisis"]["auditoria_causalidad"]["futuro"]["unidad"] = "OTRA"
    motor = _motor(tmp_path / "ej", 2)
    datos = _th37_ref(motor, th37_frame)
    with pytest.raises(ca.ProtocolNotAuthorizedError) as info:
        motor.ejecutar("TH37", [mo.Tarea("TH37O", "_t_th37_origin", (datos, bad), "TH37O")])
    assert str(info.value) == "auditoria causal distinta de la implementada" and motor.workers == []
    assert any(e["evento"] == "tarea_error" and e["id"] == "TH37O" for e in _events(tmp_path / "ej"))


# --------------------------------------------------------------------------- workers muertos o bloqueados (R4)
def _bounded(fn, plazo: float = 120.0):
    """Ejecuta ``fn`` con un vigilante: si el motor no termina en ``plazo`` la prueba FALLA (bloqueo indefinido) en
    lugar de colgar la suite. Devuelve la excepcion lanzada (None si no lanza)."""
    out = {}

    def run():
        try:
            fn()
        except BaseException as exc:                                          # noqa: BLE001
            out["exc"] = exc
    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(plazo)
    if th.is_alive():
        for c in mo.psutil.Process().children(recursive=True):
            c.kill()
        pytest.fail(f"el motor sigue esperando tras {plazo:g} s: bloqueo indefinido")
    return out.get("exc")


def _spy_workers(motor, on_new=None, before_send=None) -> list:
    """Procesos de los workers creados (para comprobar la limpieza) y acciones al crearlos o antes de cada envio."""
    procs, real_new, real_send = [], motor._nuevo_worker, motor._enviar

    def new():
        w = real_new()
        procs.append(w.proc)
        if on_new is not None:
            on_new(w)
        return w

    def send(w):
        if before_send is not None:
            before_send(w)
        return real_send(w)
    motor._nuevo_worker, motor._enviar = new, send
    return procs


def _assert_failed_cleanly(exc, motor, procs, ej: Path, fragment: str) -> None:
    """Rechazo EXPLICITO E03_TRABAJADOR, ningun worker vivo, evento de fallo, ninguna tarea completada ni resultado."""
    assert type(exc) is mo.MotorE03Error and str(exc).startswith("[E03_TRABAJADOR] el worker "), repr(exc)
    assert fragment in str(exc), str(exc)
    assert motor.workers == [] and procs and not [p for p in procs if p.is_alive()]
    ev = _events(ej)
    assert [e for e in ev if e["evento"] == "worker_fallo"] and not [e for e in ev if e["evento"] == "tarea_completa"]
    assert not (ej / "COMPLETA.json").exists()


def _large_cfg(cfg) -> dict:
    big = copy.deepcopy(cfg)
    big["_relleno_prueba_r4"] = "x" * (4 * 1024 * 1024)                    # tarea GRANDE: mucho mayor que el bufer
    return big


def _th37_tasks(datos, c, n: int = 1) -> list:
    return [mo.Tarea(f"TH37O/{k}", "_t_th37_origin", (datos, c), "TH37O") for k in range(n)]


def test_a_dead_worker_fails_the_whole_run(tmp_path, cfg, th37_frame):
    """La prueba original (asesino 1 s despues de asignar la tarea), reforzada en R4: rechazo explicito, limpieza de
    procesos, evento de fallo y nunca un resultado; un bloqueo indefinido (el defecto de R3) hace FALLAR la prueba."""
    motor = _motor(tmp_path / "ej", 1)
    datos = _th37_ref(motor, th37_frame)
    procs = _spy_workers(motor)

    def killer():
        for _ in range(600):
            busy = [w for w in motor.workers if w.tarea is not None]
            if busy:
                time.sleep(1.0)
                busy[0].proc.terminate()
                return
            time.sleep(0.05)
    th = threading.Thread(target=killer, daemon=True)
    th.start()
    exc = _bounded(lambda: motor.ejecutar("TH37", _th37_tasks(datos, cfg, 3)))
    th.join()
    _assert_failed_cleanly(exc, motor, procs, tmp_path / "ej", "")


def test_a_worker_that_dies_before_receiving_a_large_task_fails_explicitly(tmp_path, cfg, th37_frame):
    """Reproduccion determinista del bloqueo de R3: el worker muere ANTES de tomar su extremo del pipe (sigue duplicado
    en el principal) y la tarea no cabe en el bufer; antes el envio esperaba para siempre."""
    motor = _motor(tmp_path / "ej", 1)
    datos = _th37_ref(motor, th37_frame)
    big = _large_cfg(cfg)
    assert len(pickle.dumps(("TH37O/0", "_t_th37_origin", (datos, big)))) > 4 * 1024 * 1024

    def kill_now(w):
        w.proc.terminate()
        w.proc.join(30)
    procs = _spy_workers(motor, on_new=kill_now)
    exc = _bounded(lambda: motor.ejecutar("TH37", _th37_tasks(datos, big)))
    _assert_failed_cleanly(exc, motor, procs, tmp_path / "ej", "termino (codigo")


def test_a_worker_that_dies_during_the_send_of_a_large_task_fails_explicitly(tmp_path, cfg, th37_frame):
    """El worker ya confirmo su arranque, se suspende (no lee) y muere mientras el principal le envia la tarea grande."""
    assert mo.psutil is not None
    motor = _motor(tmp_path / "ej", 1)
    datos = _th37_ref(motor, th37_frame)

    def suspend_then_kill(w):
        mo.psutil.Process(w.pid).suspend()
        threading.Timer(1.0, w.proc.terminate).start()
    procs = _spy_workers(motor, before_send=suspend_then_kill)
    exc = _bounded(lambda: motor.ejecutar("TH37", _th37_tasks(datos, _large_cfg(cfg))))
    _assert_failed_cleanly(exc, motor, procs, tmp_path / "ej", "durante el envio")


def test_a_blocked_worker_that_never_reads_its_task_fails_explicitly(tmp_path, cfg, th37_frame):
    """Worker vivo pero bloqueado (suspendido): el envio tiene plazo; se termina el worker y se rechaza."""
    assert mo.psutil is not None
    motor = _motor(tmp_path / "ej", 1, espera_envio_s=2.0)
    datos = _th37_ref(motor, th37_frame)
    procs = _spy_workers(motor, before_send=lambda w: mo.psutil.Process(w.pid).suspend())
    exc = _bounded(lambda: motor.ejecutar("TH37", _th37_tasks(datos, _large_cfg(cfg))))
    _assert_failed_cleanly(exc, motor, procs, tmp_path / "ej", "no completo el envio en 2 s")


def test_a_worker_that_never_confirms_its_start_fails_explicitly(tmp_path, cfg, th37_frame):
    """Worker suspendido nada mas crearse: nunca confirma su arranque y nunca se le envia la tarea."""
    assert mo.psutil is not None
    motor = _motor(tmp_path / "ej", 1, espera_arranque_s=2.0)
    datos = _th37_ref(motor, th37_frame)
    sends = []
    procs = _spy_workers(motor, on_new=lambda w: mo.psutil.Process(w.pid).suspend(), before_send=sends.append)
    exc = _bounded(lambda: motor.ejecutar("TH37", _th37_tasks(datos, cfg)))
    _assert_failed_cleanly(exc, motor, procs, tmp_path / "ej", "no confirmo su arranque en 2 s")
    assert sends == []


def test_a_worker_that_dies_while_idle_fails_explicitly(tmp_path, cfg, th37_frame):
    """Un worker libre (entre etapas) que muere se detecta al despachar: rechazo explicito, nunca se le asigna nada."""
    motor = _motor(tmp_path / "ej", 1)
    datos = _th37_ref(motor, th37_frame)
    procs = _spy_workers(motor)
    assert _bounded(lambda: motor.ejecutar("TH37A", _th37_tasks(datos, cfg))) is None
    (w,) = motor.workers
    assert w.tarea is None and w.listo
    w.proc.terminate()
    w.proc.join(30)
    exc = _bounded(lambda: motor.ejecutar("TH37B", [mo.Tarea("TH37O/b", "_t_th37_origin", (datos, cfg), "TH37O")]))
    assert type(exc) is mo.MotorE03Error and str(exc).startswith(f"[E03_TRABAJADOR] el worker {w.pid} termino (codigo ")
    assert str(exc).endswith(") sin tarea asignada") and motor.workers == [] and not [p for p in procs if p.is_alive()]
    assert [e for e in _events(tmp_path / "ej") if e["evento"] == "worker_fallo"]


def test_the_pipe_transfers_are_bounded_and_watch_the_worker(tmp_path):
    """``_transferir`` (envio y recepcion): plazo, muerte del worker, pipe roto y transferencia normal."""
    motor = _motor(tmp_path / "ej", 1, espera_envio_s=0.3)

    class _Proc:
        def __init__(self, alive):
            self.alive, self.exitcode, self.terminated = alive, None if alive else 9, False

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True
    never = threading.Event()

    def broken():
        raise EOFError

    def worker(pid, alive=True):
        return mo._Worker(proc=_Proc(alive), conn=None, pid=pid, listo=True)
    def rejects(message, w, fn, que):
        exc = _bounded(lambda: motor._transferir(w, fn, que), 30.0)
        assert type(exc) is mo.MotorE03Error and str(exc) == message, repr(exc)
    w1, w2, w3 = worker(4241), worker(4242, alive=False), worker(4243)
    rejects("[E03_TRABAJADOR] el worker 4241 no completo la recepcion en 0.3 s", w1, never.wait, "la recepcion")
    rejects("[E03_TRABAJADOR] el worker 4242 termino (codigo 9) durante el envio", w2, never.wait, "el envio")
    rejects("[E03_TRABAJADOR] el worker 4243 termino durante la recepcion (EOFError)", w3, broken, "la recepcion")
    assert w1.proc.terminated and w2.proc.terminated and w3.proc.terminated
    assert motor._transferir(worker(4244), lambda: ("ok", "T", 1, {}), "la recepcion") == ("ok", "T", 1, {})
    fallos = [e for e in _events(tmp_path / "ej") if e["evento"] == "worker_fallo"]
    assert [e["pid"] for e in fallos] == [4241, 4242, 4243]
    never.set()


def test_low_memory_runs_one_task_at_a_time(tmp_path, cfg, monkeypatch):
    monkeypatch.setattr(mo, "memoria_disponible_mb", lambda: 100.0)
    motor = _motor(tmp_path / "ej", 4)
    try:
        res = motor.ejecutar("B", _tiny_tasks(motor, cfg))
    finally:
        motor.cerrar()
    assert len(res) == 6 and motor.max_activos == 1 and motor.secuencial_por_memoria == 6
    assert sum(e["evento"] == "memoria_baja_secuencial" for e in _events(tmp_path / "ej")) == 6


def test_enough_memory_runs_tasks_in_parallel(tmp_path, cfg, monkeypatch):
    monkeypatch.setattr(mo, "memoria_disponible_mb", lambda: 1e9)
    motor = _motor(tmp_path / "ej", 3, max_sin_medir=3)
    try:
        motor.ejecutar("B", _tiny_tasks(motor, cfg))
    finally:
        motor.cerrar()
    assert motor.max_activos == 3 and motor.secuencial_por_memoria == 0


def test_memory_gate_counts_the_possible_private_growth_of_running_tasks(tmp_path, monkeypatch):
    motor = _motor(tmp_path / "ej", 4, reserva_mb=1000.0)
    monkeypatch.setattr(mo, "memoria_disponible_mb", lambda: 5000.0)
    monkeypatch.setattr(mo, "privada_mb", lambda pid=None: 500.0)
    monkeypatch.setattr(mo, "rss_mb", lambda pid=None: 4000.0)                # el RSS (con compartidas) no decide
    t = mo.Tarea("x", "_t_th35_r2_block", (), "T", pico_inicial_mb=2000.0)
    motor.pico["T"] = 2000.0
    busy = mo._Worker(proc=None, conn=None, pid=1, tarea=t)
    motor.workers = [busy]
    assert motor._cabe(t, None)                      # 5000 - (2000-500) - 2000 = 1500 >= 1000
    motor.workers = [busy, mo._Worker(proc=None, conn=None, pid=2, tarea=t)]
    assert not motor._cabe(t, None)                  # 5000 - 2*1500 - 2000 = 0 < 1000
    motor.workers = []


def _no_workers_started(d: Path) -> bool:
    return not any(e["evento"] == "worker_inicio" for e in _events(d))


def test_strict_memory_never_starts_a_worker_for_a_task_that_does_not_fit(tmp_path, cfg, monkeypatch):
    """Ruta oficial (``memoria_estricta``): sin ninguna tarea activa, una tarea cuya memoria prevista mas la reserva
    excede la disponible NO arranca ningun worker y la ejecucion se rechaza de forma explicita (la reserva de 4 GB no
    se elimina). Sin ``memoria_estricta`` (pruebas y benchmarks) se lanzaria sola."""
    monkeypatch.setattr(mo, "memoria_disponible_mb", lambda: 4000.0)              # < reserva por defecto (4096)
    monkeypatch.setattr(mo, "privada_mb", lambda pid=None: 0.0)
    motor = _motor(tmp_path / "ej", 4, memoria_estricta=True, espera_memoria_s=0.3)
    assert motor.r.reserva_mb == 4096.0
    tareas = _tiny_tasks(motor, cfg)
    try:
        with pytest.raises(mo.MotorE03Error) as info:
            motor.ejecutar("B", tareas)
    finally:
        motor.cerrar()
    msg = str(info.value)
    assert msg.startswith("[E03_MEMORIA_INSUFICIENTE] la tarea ") and "< reserva 4096 MB; no se lanza" in msg, msg
    assert motor.max_activos == 0 and motor.secuencial_por_memoria == 0 and _no_workers_started(tmp_path / "ej")
    (ev,) = [e for e in _events(tmp_path / "ej") if e["evento"] == "memoria_insuficiente"]
    assert ev["disponible_mb"] == 4000 and ev["reserva_mb"] == 4096 and ev["prevista_mb"] > 0


def test_strict_memory_counts_the_task_the_pending_shared_data_and_the_reserve(tmp_path, monkeypatch):
    motor = _motor(tmp_path / "ej", 4, reserva_mb=1000.0, memoria_estricta=True, espera_memoria_s=0.0)
    monkeypatch.setattr(mo, "memoria_disponible_mb", lambda: 5000.0)
    monkeypatch.setattr(mo, "privada_mb", lambda pid=None: 0.0)
    t = mo.Tarea("x", "_t_th37_future", (), "TH37F", pico_inicial_mb=2000.0, compartido=("th37", 2500.0))
    b = motor._balance_memoria(t, None)
    assert (b["compartida_pendiente_mb"], b["prevista_mb"], b["cabe"]) == (2500.0, 2000.0, False)   # 5000-2500-2000 < 1000
    assert not motor._cabe(t, None)                                               # tambien sin ninguna activa
    raises_exactly(mo.MotorE03Error, "[E03_MEMORIA_INSUFICIENTE] la tarea x no cabe: disponibles 5000 MB - compartidos "
                   "pendientes 2500 MB - prevista 2000 MB < reserva 1000 MB; no se lanza y la ejecucion se rechaza",
                   motor._despachar, [t], {})
    assert motor.workers == [] and _no_workers_started(tmp_path / "ej")
    motor.compartido_cargado.add("th37")                                          # ya residentes: 5000-2000 >= 1000
    assert motor._cabe(t, None)
    t2 = mo.Tarea("y", "_t_th37_future", (), "TH37F", pico_inicial_mb=4500.0)       # 5000-4500 < 1000 sin compartidos
    raises_exactly(mo.MotorE03Error, "[E03_MEMORIA_INSUFICIENTE] la tarea y no cabe: disponibles 5000 MB - compartidos "
                   "pendientes 0 MB - prevista 4500 MB < reserva 1000 MB; no se lanza y la ejecucion se rechaza",
                   motor._despachar, [t2], {})
    assert motor.workers == []


def test_strict_memory_waits_while_tasks_are_running(tmp_path, monkeypatch):
    """Con tareas activas, una tarea que no cabe espera (no se lanza ni se rechaza)."""
    motor = _motor(tmp_path / "ej", 4, reserva_mb=1000.0, memoria_estricta=True)
    monkeypatch.setattr(mo, "memoria_disponible_mb", lambda: 2500.0)
    monkeypatch.setattr(mo, "privada_mb", lambda pid=None: 0.0)
    t = mo.Tarea("x", "_t_th35_r2_block", (), "T", pico_inicial_mb=1000.0)
    motor.pico["T"] = 1000.0
    motor.workers = [mo._Worker(proc=None, conn=None, pid=1, tarea=t)]
    cola = [mo.Tarea("y", "_t_th35_r2_block", (), "T", pico_inicial_mb=1000.0)]
    motor._despachar(cola, {})                                   # 2500 - 1000 - 1000 = 500 < 1000: espera
    assert [x.id for x in cola] == ["y"] and len(motor.workers) == 1
    motor.workers = []


def test_strict_memory_closes_idle_workers_and_launches_only_when_it_fits(tmp_path, cfg, monkeypatch):
    """Sin activas y sin memoria: se cierran los workers libres (sus caches) y la tarea solo arranca si entonces cabe;
    el resultado no cambia."""
    motor = _motor(tmp_path / "ej", 3, max_sin_medir=3, memoria_estricta=True, espera_memoria_s=5.0)
    try:
        ref = motor.ejecutar("A", _tiny_tasks(motor, cfg))                         # memoria real: workers libres
        assert motor.workers
        monkeypatch.setattr(mo, "memoria_disponible_mb", lambda: 1e9 if not motor.workers else 100.0)
        otra = [mo.Tarea("B" + t.id, t.fn, t.args, t.tipo, t.rasgos, t.pico_inicial_mb) for t in _tiny_tasks(motor, cfg)]
        res = motor.ejecutar("B", otra)
    finally:
        motor.cerrar()
    assert {k[1:]: v for k, v in res.items()}.keys() == ref.keys()
    assert all(pickle_sha(res["B" + k]) == pickle_sha(v) for k, v in ref.items())
    ev = [e["evento"] for e in _events(tmp_path / "ej")]
    assert "worker_cerrado_por_memoria" in ev and "memoria_recuperada" in ev and "memoria_insuficiente" not in ev
    assert motor.secuencial_por_memoria == 0


def test_an_official_run_needs_strict_memory_control(tmp_path, cfg):
    df = _engine_input(bars_per_day=40)
    for r in (mo.Recursos(procesos=2), mo.Recursos(procesos=0, memoria_estricta=True),
              mo.Recursos(procesos=2, memoria_estricta=True, reserva_mb=0.0)):
        with pytest.raises(mo.MotorE03Error, match=r"^\[E03_RECURSOS_OFICIALES\] la ejecucion oficial exige control "
                                                    r"de memoria estricto"):
            mo.run_th35_e03(df, cfg, tmp_path / "ej", r, oficial=True)
        assert not (tmp_path / "ej").exists()


def test_shared_memory_is_counted_once_until_it_is_resident(tmp_path, monkeypatch):
    """Los ficheros proyectados se cuentan UNA vez (no por worker) mientras ninguna tarea que los usa ha terminado;
    despues ya estan en la memoria medida como no disponible y no se vuelven a contar."""
    motor = _motor(tmp_path / "ej", 8, reserva_mb=1000.0)
    monkeypatch.setattr(mo, "memoria_disponible_mb", lambda: 10000.0)
    monkeypatch.setattr(mo, "privada_mb", lambda pid=None: 0.0)
    t = mo.Tarea("x", "_t_th37_future", (), "TH37F", pico_inicial_mb=1000.0, compartido=("th37", 3000.0))
    motor.pico["TH37F"] = 1000.0
    busy = [mo._Worker(proc=None, conn=None, pid=k, tarea=t) for k in range(6)]
    motor.workers = busy[:5]
    assert motor._compartido_pendiente([w.tarea for w in motor.workers] + [t]) == 3000.0   # una vez, no 6 x 3000
    assert motor._cabe(t, None)                      # 10000 - 3000 - 5*1000 - 1000 = 1000 >= 1000
    motor.workers = busy
    assert not motor._cabe(t, None)                  # 10000 - 3000 - 6*1000 - 1000 = 0 < 1000
    motor.compartido_cargado.add("th37")             # ya residentes: la memoria disponible medida los incluye
    assert motor._compartido_pendiente([w.tarea for w in motor.workers] + [t]) == 0.0
    assert motor._cabe(t, None)                      # 10000 - 6*1000 - 1000 = 3000 >= 1000
    motor.workers = []


def test_performance_cap_only_when_more_workers_are_slower(tmp_path):
    motor = _motor(tmp_path / "ej")
    motor.medidas["T"] = [((1.0,), 1.0, 1)] * 3 + [((1.0,), 4.0, 4)] * 3
    motor._actualizar_rendimiento("T")
    assert "T" not in motor.tope_rendimiento                    # misma productividad: sin tope
    motor.medidas["T"] = [((1.0,), 1.0, 1)] * 3 + [((1.0,), 5.0, 4)] * 3
    motor._actualizar_rendimiento("T")
    assert motor.tope_rendimiento["T"] == 1                     # 4/5 < 0.95 * 1/1


def test_time_remaining_only_from_measured_tasks(tmp_path):
    motor = _motor(tmp_path / "ej", minimo_medidas_eta=3)
    pend = [mo.Tarea(f"t{i}", "f", (), "T", (2.0,)) for i in range(4)]
    assert motor._eta(pend)["estado"] == "NO_ESTIMABLE"
    motor.medidas["T"] = [((1.0,), 3.0, 1)] * 3
    motor.medidas["T"] += [((2.0,), 6.0, 1)] * 2
    eta = motor._eta(pend)
    assert eta["estado"] == "ESTIMADO" and eta["segundos"] == pytest.approx(24.0) and eta["medidas"] == {"T": 5}
    otra = mo.Tarea("u", "f", (), "U", (1.0,))                                 # tipo sin medidas suficientes
    eta = motor._eta(pend + [otra])
    assert eta["estado"] == "COTA_INFERIOR" and eta["segundos"] == pytest.approx(24.0) and eta["sin_estimar"] == {"U": 1}


def test_progress_is_verifiable(tmp_path, cfg):
    motor = _motor(tmp_path / "ej", 2)
    try:
        motor.ejecutar("B", _tiny_tasks(motor, cfg))
    finally:
        motor.cerrar()
    est = json.loads((tmp_path / "ej" / "estado.json").read_text(encoding="utf-8"))
    assert est["estado"] == "ETAPA_COMPLETA" and est["etapas"] == [
        {"etapa": "B", "hechas": 6, "total": 6, "reutilizadas": 0, "progreso": "6 de 6", "duracion_s": est["etapas"][0]["duracion_s"]}]
    done = [e for e in _events(tmp_path / "ej") if e["evento"] == "tarea_completa"]
    assert len(done) == 6 and all(e["duracion_s"] >= 0 and len(e["sha256"]) == 64 for e in done)
    assert all("privada_pico_mb" in e and "rss_pico_mb" in e for e in done)
    if mo.psutil is not None:                                                  # con psutil: memoria medida
        assert all(e["privada_pico_mb"] > 0 and e["rss_pico_mb"] > 0 for e in done)
    assert [e["evento"] for e in _events(tmp_path / "ej") if e["evento"].startswith("etapa")] == ["etapa_inicio", "etapa_fin"]


# =========================================================================== puntos de control
def test_checkpoints_resume_only_verified_tasks(tmp_path, cfg):
    """Reanudacion SINTETICA (fuera de la ruta oficial)."""
    d = tmp_path / "ej"
    motor = _motor(d, 2, fallo_tras=2)
    with pytest.raises(mo.MotorE03Error, match=r"^\[E03_INTERRUPCION_SIMULADA\]"):
        motor.ejecutar("B", _tiny_tasks(motor, cfg))
    motor.cerrar()
    ident = json.loads((d / "identidad.json").read_text(encoding="utf-8"))
    raises_exactly(mo.MotorE03Error, f"[E03_EJECUCION] el directorio de la ejecucion no esta vacio: {d}",
                   mo.Ejecucion, d, ident)
    raises_exactly(mo.MotorE03Error, "[E03_CHECKPOINT] la identidad de la ejecucion (configuracion, entradas, motor o "
                                     "entorno) no coincide", mo.Ejecucion, d, {**ident, "otra": 1}, reanudar=True)
    ej = mo.Ejecucion(d, ident, reanudar=True)
    motor = mo._Motor(mo.Recursos(procesos=2), ej)
    try:
        res = motor.ejecutar("B", _tiny_tasks(motor, cfg))
    finally:
        motor.cerrar()
    assert len(ej.reutilizadas) == 2 and len(res) == 6
    fresh = _motor(tmp_path / "fresca", 0)
    ref = fresh.ejecutar("B", _tiny_tasks(fresh, cfg))
    assert all(pickle_sha(res[k]) == pickle_sha(ref[k]) for k in ref)
    victim = d / "tareas" / f"{mo._file_id(ej.reutilizadas[0])}.pkl"
    victim.write_bytes(victim.read_bytes() + b"x")
    ej2 = mo.Ejecucion(d, ident, reanudar=True)
    with pytest.raises(mo.MotorE03Error, match=r"^\[E03_CHECKPOINT\] punto de control alterado o ausente"):
        mo._Motor(mo.Recursos(procesos=0), ej2).ejecutar("B", _tiny_tasks(motor, cfg))


def test_a_complete_run_is_never_resumed_and_an_incomplete_one_has_no_result(tmp_path, cfg):
    df = _engine_input(bars_per_day=40)
    c = copy.deepcopy(cfg)
    c["analisis"]["bootstrap"]["replicas"] = 2
    with pytest.raises(mo.MotorE03Error, match=r"^\[E03_INTERRUPCION_SIMULADA\]"):
        mo.run_th35_e03(df, c, tmp_path / "ej", mo.Recursos(procesos=0, fallo_tras=3))
    assert not (tmp_path / "ej" / "COMPLETA.json").exists()
    assert json.loads((tmp_path / "ej" / "estado.json").read_text(encoding="utf-8"))["estado"] == "FALLO"
    res, resumen = mo.run_th35_e03(df, c, tmp_path / "ej", mo.Recursos(procesos=0), reanudar=True)
    assert resumen["reanudada"] and len(resumen["reutilizadas"]) == 3 and (tmp_path / "ej" / "COMPLETA.json").is_file()
    assert _tables(res, c) == _tables(an.th35_pipeline(df, c), c)
    raises_exactly(mo.MotorE03Error, "[E03_CHECKPOINT] la ejecucion ya esta completa: no se reanuda",
                   mo.run_th35_e03, df, c, tmp_path / "ej", mo.Recursos(procesos=0), reanudar=True)


def test_official_runs_never_resume_and_never_read_checkpoints(tmp_path, cfg):
    """Ruta oficial: sin reanudacion y en un directorio NUEVO. Un directorio preparado con puntos de control (p. ej.
    resultados falsificados con huellas coherentes) se rechaza antes de leer nada."""
    df = _engine_input(bars_per_day=40)
    c = copy.deepcopy(cfg)
    c["analisis"]["bootstrap"]["replicas"] = 2
    d = tmp_path / "ej"
    with pytest.raises(mo.MotorE03Error, match=r"^\[E03_INTERRUPCION_SIMULADA\]"):
        mo.run_th35_e03(df, c, d, mo.Recursos(procesos=0, fallo_tras=3))       # puntos de control de 3 tareas
    ident = json.loads((d / "identidad.json").read_text(encoding="utf-8"))
    raises_exactly(mo.MotorE03Error, "[E03_REANUDACION_OFICIAL] la reanudacion de una ejecucion oficial esta deshabilitada",
                   mo.Ejecucion, d, ident, reanudar=True, oficial=True)
    raises_exactly(mo.MotorE03Error, f"[E03_EJECUCION] el directorio de una ejecucion oficial no puede existir: {d}",
                   mo.Ejecucion, d, ident, oficial=True)
    estricta = mo.Recursos(procesos=2, memoria_estricta=True)
    raises_exactly(mo.MotorE03Error, f"[E03_EJECUCION] el directorio de una ejecucion oficial no puede existir: {d}",
                   mo.run_th35_e03, df, c, d, estricta, oficial=True)
    (tmp_path / "vacio").mkdir()
    raises_exactly(mo.MotorE03Error, f"[E03_EJECUCION] el directorio de una ejecucion oficial no puede existir: "
                                     f"{tmp_path / 'vacio'}", mo.Ejecucion, tmp_path / "vacio", ident, oficial=True)
    ej = mo.Ejecucion(tmp_path / "nueva", ident, oficial=True)
    ej.registro = {"x": {"sha256": "0" * 64}}                                  # aunque hubiera registro, no se lee
    assert ej.guardado("x") == (False, None)
    res, resumen = mo.run_th35_e03(df, c, tmp_path / "oficial", estricta, oficial=True)
    assert resumen["reutilizadas"] == [] and not resumen["reanudada"] and resumen["lanzadas_sin_memoria_suficiente"] == 0
    assert _tables(res, c) == _tables(an.th35_pipeline(df, c), c)
