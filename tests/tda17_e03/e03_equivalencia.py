"""Equivalencia E02/E03 extremo a extremo con datos SINTETICOS (no es un test; lo lanza ``test_e03_motor`` en un
subproceso cuyo entorno numerico fija el lanzador, p. ej. OPENBLAS_NUM_THREADS=1, para que el proceso principal y
los workers tengan los mismos hilos BLAS). Ningun dato de MNQ, ningun holdout.

Uso: ``python tests/tda17_e03/e03_equivalencia.py <salida.json> <directorio_trabajo>``. Escribe un informe JSON con, por
caso, la huella de cada tabla oficial de E02 y de cada configuracion del motor E03 y las comparaciones profundas.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))          # tests/ (ayudantes sinteticos de E01/E02)

from ohlcv_dataroad.ingest import tda17_analysis as an  # noqa: E402
from ohlcv_dataroad.ingest import tda17_enmiendas_e02 as m2  # noqa: E402
from ohlcv_dataroad.ingest import tda17_salidas as sal  # noqa: E402
from ohlcv_dataroad.tda17_e03 import motor as mo  # noqa: E402
from test_tda17_procedencia import _engine_input  # noqa: E402

CONFIGURACIONES = [  # (procesos, cortes_por_bloque, replicas_por_bloque, barajar)
    (0, 1000, 3, None),
    (1, 7, 1, None),
    (2, 1, 2, 11),
    (4, 3, 1, 12),
]


def tables(res: dict, cfg: dict) -> dict[str, bytes]:
    return {n: sal.csv_bytes(t) for n, t in sal.official_tables(res, cfg).items()}


def deep_equal(a: dict, b: dict) -> list[str]:
    """Comparaciones del resultado completo mas alla de las tablas (claves, per_cell, bootstrap con attrs, validacion,
    auditoria, analisis por celda)."""
    bad = []
    if list(a) != list(b):
        bad.append(f"claves {list(a)} != {list(b)}")
    for k in ("estado", "bloqueos", "poblacion", "validacion_spearman"):
        if repr(a.get(k)) != repr(b.get(k)):
            bad.append(k)
    for k in ("auditoria_causalidad", "per_cell"):
        try:
            pd.testing.assert_frame_equal(a[k], b[k], check_exact=True)
        except AssertionError:
            bad.append(k)
    if list(a["bootstrap"]) != list(b["bootstrap"]):
        bad.append("bootstrap: celdas")
    for c in a["bootstrap"]:
        ra, rb = a["bootstrap"][c], b["bootstrap"][c]
        if ra["jornadas"] != rb["jornadas"]:
            bad.append(f"bootstrap {c}: jornadas")
        for k in ("r2_todas", "spearman"):
            try:
                pd.testing.assert_frame_equal(ra[k], rb[k], check_exact=True)
            except AssertionError:
                bad.append(f"bootstrap {c}: {k}")
        ta, tb = ra["r2_todas"].attrs, rb["r2_todas"].attrs
        if not (np.array_equal(ta["tamano_base"], tb["tamano_base"]) and ta["cuantiles"] == tb["cuantiles"]):
            bad.append(f"bootstrap {c}: attrs")
    if list(a["analisis_celdas"]) != list(b["analisis_celdas"]):
        bad.append("analisis_celdas: celdas")
    for c in a["analisis_celdas"]:
        if repr(sorted(a["analisis_celdas"][c])) != repr(sorted(b["analisis_celdas"][c])):
            bad.append(f"analisis {c}: claves")
        for k, v in a["analisis_celdas"][c].items():
            w = b["analisis_celdas"][c][k]
            if isinstance(v, pd.DataFrame):
                try:
                    pd.testing.assert_frame_equal(v, w, check_exact=True)
                except AssertionError:
                    bad.append(f"analisis {c}: {k}")
            elif isinstance(v, np.ndarray):
                if not (v.shape == w.shape and v.tobytes() == w.tobytes()):
                    bad.append(f"analisis {c}: {k}")
            elif pickle_sha(v) != pickle_sha(w):
                bad.append(f"analisis {c}: {k}")
    for k in ("clasificacion", "subniveles"):
        if (k in a) != (k in b):
            bad.append(k)
        elif k in a:
            try:
                pd.testing.assert_frame_equal(a[k], b[k], check_exact=True)
            except AssertionError:
                bad.append(k)
    return bad


def pickle_sha(obj) -> str:
    """Huella estable de objetos anidados (dict/list/DataFrame/ndarray) por su representacion completa."""
    def norm(o):
        if isinstance(o, pd.DataFrame):
            return ("DF", list(o.columns), [str(t) for t in o.dtypes], o.to_csv(lineterminator="\n"))
        if isinstance(o, np.ndarray):                  # objetos: por valor (los bytes serian punteros)
            data = repr(o.tolist()) if o.dtype == object else o.tobytes().hex()
            return ("ND", o.dtype.str, o.shape, data)
        if isinstance(o, dict):
            return ("D", [(k, norm(v)) for k, v in o.items()])
        if isinstance(o, (list, tuple)):
            return ("L", [norm(v) for v in o])
        if isinstance(o, float):
            return ("F", o.hex() if o == o else "nan")
        return ("O", repr(o))
    return hashlib.sha256(repr(norm(obj)).encode("utf-8")).hexdigest()


def run(out: Path, work: Path) -> dict:
    cfg = copy.deepcopy(m2.effective_document_e02(borrador=True))
    cfg["analisis"]["bootstrap"]["replicas"] = 3
    df_u = _engine_input(bars_per_day=110, hi_every=0, long_hi_every=0)
    first = an.th35_pipeline(df_u, cfg)
    rk = {r["matriz"]: r["autovalores"] for r in first["analisis_celdas"]["GLOBAL"]["rango"]}
    cfg_u = copy.deepcopy(cfg)
    cfg_u["identidades"]["verificacion_numerica"]["nulos_esperados"].update(rk)
    casos = {"bloqueado": (_engine_input(), cfg), "sin_bloqueo": (df_u, cfg_u)}
    report = {"hilos_blas": mo.hilos_blas_efectivos(), "casos": {}}
    for nombre, (df, c) in casos.items():
        t = time.time()
        ref = an.th35_pipeline(df, c)
        rb = tables(ref, c)
        caso = {"estado_E02": ref["estado"], "segundos_E02": round(time.time() - t, 2),
                "tablas_E02": {n: hashlib.sha256(b).hexdigest() for n, b in rb.items()}, "E03": []}
        for k, (procs, cpb, rpb, barajar) in enumerate(CONFIGURACIONES):
            t = time.time()
            rec = mo.Recursos(procesos=procs, cortes_por_bloque=cpb, replicas_por_bloque=rpb, barajar=barajar,
                              intervalo_estado_s=0.5, minimo_medidas_eta=2)
            res, resumen = mo.run_th35_e03(df, c, work / f"{nombre}_{k}", rec)
            b = tables(res, c)
            caso["E03"].append({"procesos": procs, "cortes_por_bloque": cpb, "replicas_por_bloque": rpb,
                                "barajar": barajar, "segundos": round(time.time() - t, 2),
                                "tablas": {n: hashlib.sha256(x).hexdigest() for n, x in b.items()},
                                "identicas": b == rb, "diferencias_profundas": deep_equal(ref, res),
                                "max_workers_activos": resumen["max_workers_activos"],
                                "completa": (work / f"{nombre}_{k}" / "COMPLETA.json").is_file()})
        # interrupcion simulada + reanudacion verificada (procesos 2)
        d = work / f"{nombre}_reanudacion"
        rec = mo.Recursos(procesos=2, cortes_por_bloque=5, replicas_por_bloque=1, fallo_tras=6)
        try:
            mo.run_th35_e03(df, c, d, rec)
            interrumpida = "NO"
        except mo.MotorE03Error as exc:
            interrumpida = str(exc)
        incompleta = not (d / "COMPLETA.json").exists()
        rec = mo.Recursos(procesos=2, cortes_por_bloque=5, replicas_por_bloque=1)
        res, resumen = mo.run_th35_e03(df, c, d, rec, reanudar=True)
        caso["reanudacion"] = {"interrupcion": interrumpida, "sin_COMPLETA_tras_interrupcion": incompleta,
                               "reutilizadas": len(resumen["reutilizadas"]), "identicas": tables(res, c) == rb,
                               "diferencias_profundas": deep_equal(ref, res),
                               "completa": (d / "COMPLETA.json").is_file()}
        estado = json.loads((d / "estado.json").read_text(encoding="utf-8"))
        caso["reanudacion"]["estado_final"] = estado["estado"]
        report["casos"][nombre] = caso
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return report


def tablas_e02(out: Path, fijar: int | None) -> dict:
    """Huellas de las tablas del motor E02 en los casos BLOQUEADO y SIN_BLOQUEO, con los hilos de OpenBLAS del
    arranque o fijados en tiempo de ejecucion (``mo.hilos_blas_fijados``)."""
    cfg = copy.deepcopy(m2.effective_document_e02(borrador=True))
    cfg["analisis"]["bootstrap"]["replicas"] = 3
    df_u = _engine_input(bars_per_day=110, hi_every=0, long_hi_every=0)
    ctx = mo.hilos_blas_fijados(fijar) if fijar is not None else contextlib.nullcontext(mo.hilos_blas_efectivos())
    with ctx as hilos:
        first = an.th35_pipeline(df_u, cfg)
        rk = {r["matriz"]: r["autovalores"] for r in first["analisis_celdas"]["GLOBAL"]["rango"]}
        cfg_u = copy.deepcopy(cfg)
        cfg_u["identidades"]["verificacion_numerica"]["nulos_esperados"].update(rk)
        rep = {"hilos_blas": hilos, "fijados": fijar, "casos": {}}
        for nombre, (df, c) in {"bloqueado": (_engine_input(), cfg), "sin_bloqueo": (df_u, cfg_u)}.items():
            rep["casos"][nombre] = {n: hashlib.sha256(b).hexdigest() for n, b in tables(an.th35_pipeline(df, c), c).items()}
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return rep


if __name__ == "__main__":
    if sys.argv[1] == "--tablas-e02":
        tablas_e02(Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else None)
    else:
        run(Path(sys.argv[1]), Path(sys.argv[2]))
