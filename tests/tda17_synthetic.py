"""Constructor de datos SINTETICOS para las pruebas del motor de TDA-17 (no es un test).

Genera barras OHLCV en la rejilla de tick con contratos, trading_date, huecos internos (HI, cortos y
largos), fronteras de jornada (FJ <= 90 min, FJL > 90 min) y rolls, y aplica las MISMAS funciones de
TDA-04 (mascara de validez y variables) que la serie real. Ningun dato de MNQ, ningun holdout.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ohlcv_dataroad.ingest.tda04_analysis_variables import build_analysis_variables, build_return_validity_mask
from ohlcv_dataroad.ingest.session_calendar import build_session_schedule
from ohlcv_dataroad.ingest.tda17_indicators import SYNTHETIC_MARKER, join_inputs

TICK = 0.25


def calendar_sessions(start, n_days):
    """Primeras ``n_days`` jornadas del calendario CME_Equity del proyecto desde ``start`` (fecha, apertura UTC)."""
    table = build_session_schedule(pd.Timestamp(start), pd.Timestamp(start) + pd.Timedelta(days=2 * n_days + 30)).table
    rows = [(d, r["market_open"].tz_convert("UTC").tz_localize(None)) for d, r in table.iterrows() if d >= start]
    return rows[:n_days]


def make_frame(contracts=(("MNQ_A", 4), ("MNQ_B", 3)), bars_per_day=700, seed=0, hi_every=97,
               long_hi_every=331, fjl_every=3, zero_volume_every=0, flat_run=None, aligned=False,
               dense_gaps=(150, 260), gap_jump=6, start=dt.date(2021, 3, 22), drop_days=()):
    """Devuelve el DataFrame de entrada del motor (variables + validez de TDA-04), marcado como sintetico.

    Las trading_date son jornadas REALES del calendario CME_Equity del proyecto (sin fines de semana ni
    feriados), consecutivas desde ``start``.
    contracts: (nombre, numero de jornadas) en orden; el roll ocurre al empezar cada contrato nuevo.
    aligned: cada jornada empieza en la apertura de su sesion del calendario (18:00 NY), de modo que los
    minutos del dia son comparables entre jornadas (necesario para RVOL); si no, la serie es continua con
    fronteras FJ de 61 min (FJL tras fin de semana/feriado o cada ``fjl_every`` jornadas).
    dense_gaps: tramo (inicio, fin) de barras de la SEGUNDA jornada de cada contrato separadas por 2
    minutos (todos sus pares son HI), para que las reglas de validez 0,8n se activen.
    gap_jump: salto de precio (en ticks, aleatorio en +-gap_jump) en la apertura tras cualquier
    discontinuidad, para que el TR convencional y el local difieran.
    drop_days: indices (desde 0) de jornadas del calendario que faltan POR COMPLETO en la entrada."""
    rng = np.random.default_rng(seed)
    total = sum(n for _, n in contracts)
    sessions = calendar_sessions(start, total)
    rows = []
    price = 15000.0
    seg = 0
    k = 0
    t = sessions[0][1]
    for name, n_days in contracts:
        seg += 1
        for _d in range(n_days):
            day, open_utc = sessions[k]
            k += 1
            if aligned:
                t = open_utc
            for i in range(bars_per_day):
                step = 1
                if dense_gaps and _d == 1 and dense_gaps[0] <= i < dense_gaps[1]:
                    step = 2                                                # tramo denso de huecos internos
                if i > 0 and hi_every and i % hi_every == 0:
                    step = 3                                                # hueco interno corto
                if i > 0 and long_hi_every and i % long_hi_every == 0:
                    step = 121                                              # hueco interno largo (> 90 min)
                t = t + pd.Timedelta(minutes=step)
                if (i == 0 or step > 1) and gap_jump:
                    price = max(TICK, price + TICK * rng.integers(-gap_jump, gap_jump + 1))   # salto en la discontinuidad
                o = price
                c = max(TICK, o + TICK * rng.choice([-3, -2, -1, 0, 0, 1, 2, 3]))
                h = max(o, c) + TICK * rng.integers(0, 3)
                lo = min(o, c) - TICK * rng.integers(0, 3)
                v = float(rng.integers(1, 60))
                if zero_volume_every and i % zero_volume_every == 0:
                    v = 0.0
                if (k - 1) not in drop_days:
                    rows.append((t, f"{name}.txt", name, day, seg, o, h, lo, c, v))
                price = c
            if not aligned:
                nxt = sessions[k][0] if k < len(sessions) else day
                long_break = (nxt - day).days > 1 or (fjl_every and k % fjl_every == 0)
                t = t + pd.Timedelta(minutes=60 + (48 * 60 if long_break else 0))
        price += 250.0                                                      # salto de nivel en el roll
    canonical = pd.DataFrame(rows, columns=["timestamp", "source_file", "contract", "trading_date", "segment_id",
                                            "open", "high", "low", "close", "volume"])
    if flat_run is not None:
        a, z = flat_run
        canonical.loc[a:z, ["open", "high", "low", "close"]] = canonical.loc[a, "open"]
    out = rebuild_tda04(canonical)
    out[SYNTHETIC_MARKER] = True
    return out


def rebuild_tda04(canonical):
    """Aplica las MISMAS funciones de TDA-04 (mascara de validez y variables) a una serie canonica sintetica."""
    canonical = canonical.reset_index(drop=True)
    roll = canonical[["timestamp"]].copy()
    roll["is_roll_boundary"] = canonical["contract"].ne(canonical["contract"].shift()) & canonical.index.to_series().gt(0)
    validity = build_return_validity_mask(canonical, roll)
    variables = build_analysis_variables(canonical, validity)
    return join_inputs(variables, validity)


def clean_bars(n_bars=2600, seed=1):
    """Un contrato sin discontinuidades (todos los pares CONS salvo la primera barra)."""
    rng = np.random.default_rng(seed)
    c = 15000 + TICK * np.cumsum(rng.choice([-3, -2, -1, 0, 1, 2, 3], n_bars))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + TICK * rng.integers(1, 3, n_bars)
    lo = np.minimum(o, c) - TICK * rng.integers(1, 3, n_bars)
    v = rng.integers(1, 60, n_bars).astype(float)
    cons = np.ones(n_bars, bool)
    cons[0] = False
    new_day = np.zeros(n_bars, bool)
    new_day[0] = True
    return o, h, lo, c, v, cons, new_day


def git(root, *args, check=True):
    """git en un repositorio TEMPORAL de pruebas (identidad de prueba, sin conversion de finales de linea)."""
    import subprocess
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.autocrlf=false", *args],
                          cwd=root, capture_output=True, text=True, check=check)


def commit_tree(root, message="A: version auditada (sintetica)"):
    """Inicializa (si hace falta) un repositorio Git en ``root``, confirma su arbol y devuelve el commit."""
    from pathlib import Path
    if not (Path(root) / ".git").exists():
        git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", message)
    return git(root, "rev-parse", "HEAD").stdout.strip()


def approve_registry(root, commit_auditado, fecha="2026-10-17"):
    """Escribe en el registro de ``root`` una aprobacion SIMULADA completa que cita ``commit_auditado`` (un commit
    real del repositorio temporal) y las huellas de los ficheros auditados tal como estan en ``root``. Identidad y
    documento son marcadores de prueba, no una aprobacion real."""
    import json
    from pathlib import Path

    from ohlcv_dataroad.ingest import tda17_enmiendas as enm

    root = Path(root)
    reg = json.loads((root / enm.REGISTRY_JSON).read_text(encoding="utf-8"))
    for e in reg["enmiendas"]:
        e["estado"] = enm.ESTADO_APROBADA
        e["aprobacion"] = {
            "fecha": fecha, "aprobada_por": "PRUEBA_SINTETICA",
            "referencia_auditoria": {"documento": "PRUEBA_SINTETICA", "commit_auditado": commit_auditado},
            "sha256_auditados": {rel: enm.sha256_text_file(root / rel) for rel in enm.AUDITED_FILES}}
    (root / enm.REGISTRY_JSON).write_text(json.dumps(reg, indent=2), encoding="utf-8")
    return reg


def approved_amendment_root(dest, approve=True):
    """Raiz TEMPORAL de pruebas con el protocolo base, las enmiendas, el registro y los ficheros de implementacion
    copiados del repositorio y (si ``approve``) un repositorio Git temporal con el commit auditado A (registro
    PENDIENTE) y una aprobacion simulada que lo cita. El repositorio del proyecto sigue PENDIENTE_AUDITORIA."""
    import shutil
    from pathlib import Path

    from ohlcv_dataroad.ingest import tda17_enmiendas as enm
    from ohlcv_dataroad.ingest import tda17_protocol_freeze as frz

    src, dest = Path(enm.ROOT), Path(dest)
    files = set(frz.PROTOCOL_FILES) | {frz.MANIFEST_JSON} | set(enm.AMENDMENT_TRACKED_FILES)
    for rel in files:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / rel, dest / rel)
    if approve:
        approve_registry(dest, commit_tree(dest))
    return dest
