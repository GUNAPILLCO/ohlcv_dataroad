"""TDA-17 -- fase 2: auditoria causal de TH37 (E02, ``analisis.auditoria_causalidad``, decision D17-1a).

Componentes separados, con filas propias en ``TDA17_auditoria_causalidad.csv``:
1. Informacion futura (TRUNCAMIENTO y PERTURBACION_FUTURA), por contrato. Para cada punto de corte c (indice base 1
   de la ultima barra conservada del contrato): (i) catalogo calculado con las barras 1..c del contrato; (ii)
   catalogo del contrato completo con todas las barras posteriores a c perturbadas en la rejilla de tick. En ambos
   casos las filas <= c deben ser IDENTICAS BIT A BIT a las del calculo completo (series, NaN, marcas y mascaras).
   D7 (RVOL, cruza rolls) se audita con truncamiento y perturbacion GLOBALES hasta la fila del corte; las variables
   de TDA-04 de P, con las funciones de TDA-04 sobre la serie canonica truncada o perturbada.
   Cortes deterministas en c - 1, c y c + 1 de cada evento (FJ, FJL, HI corto y largo, barras 1-3 y ultima,
   primera barra elegible por K y clase, activacion de KAMA y adx, cambios de direccion de Supertrend, perdida y
   recuperacion de VENTANA_VALIDEZ, cuantiles 25/50/75 %) y 10 cortes aleatorios por contrato (semilla 3717,
   ``SeedSequence(3717).spawn(n_contratos)``). Un evento inexistente o un corte fuera de 2..n-1 es NO_APLICABLE
   sin sustitucion. Es una auditoria por MUESTREO, no exhaustiva.
2. Origen de los parametros (ORIGEN_PARAMETRO): tabla declarada y prueba dinamica de que las series solo dependen
   del OHLCV y de la mascara de validez de TDA-04.
3. Ajustes de roll (3a): precios OHLC de los DEMAS contratos transformados (multiplicativa, aditiva, arbitraria):
   las filas del contrato deben ser identicas bit a bit (ROLL_PRECIO). Volumen de los demas contratos perturbado:
   solo puede cambiar D7 y solo en las filas con ``rvol_cruza_roll`` (ROLL_VOLUMEN). La entrada no contiene
   columnas ajustadas.
Consecuencia: una serie con DISCREPANCIA queda fuera de las salidas oficiales y bloquea TH35 hasta corregirla.

Convenciones de implementacion (no normativas; a la vista de la auditoria): perturbacion de la barra a distancia
k >= 1 del corte = desplazamiento s = (k mod 5 + 1) ticks de toda la barra, ademas H + (k mod 3 + 1) ticks,
L - ((k + 1) mod 3 + 1) ticks y C + (k mod 2) ticks (cambian nivel y FORMA: rango, cuerpo y mechas), y volumen
+ (k mod 4 + 1); transformaciones de roll = factor 2,
+1000 ticks y + ((fila mod 7) + 1) ticks; cuantil q = barra ceil(q n); cortes aleatorios =
``Generator.integers(2, n, size=10)`` (barras 2..n-1, con reemplazo; los repetidos se usan una vez).

Este modulo no lee ni escribe ficheros ni datos brutos ni el holdout.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ohlcv_dataroad.ingest import tda17_analysis as an
from ohlcv_dataroad.ingest import tda17_indicators as ind
from ohlcv_dataroad.ingest import tda17_primitives as pr
from ohlcv_dataroad.ingest.tda04_analysis_variables import build_analysis_variables, build_return_validity_mask

ProtocolNotAuthorizedError = ind.ProtocolNotAuthorizedError
FIELDS = ["prueba", "contrato", "punto_corte", "motivo_corte", "serie", "filas_comparadas", "discrepancias",
          "max_abs_diff", "resultado", "parametro", "origen", "etiqueta"]
TESTS = ["TRUNCAMIENTO", "PERTURBACION_FUTURA", "ROLL_PRECIO", "ROLL_VOLUMEN", "ORIGEN_PARAMETRO"]
RESULTS = ["CONFORME", "DISCREPANCIA", "NO_APLICABLE"]
CANONICAL_COLUMNS = ["timestamp", "source_file", "contract", "trading_date", "segment_id",
                     "open", "high", "low", "close", "volume"]
TDA04_VARIABLES = ["r_1m", "R_1m", "abs_r_1m", "r2_1m", "zero_1m", "log_hl", "log_co", "log_oc_prev"]
KNOWN_INPUT_COLUMNS = set(CANONICAL_COLUMNS) | set(TDA04_VARIABLES) | set(ind.VALIDITY_COLUMNS) | set(ind.REQUIRED_COLUMNS)
SERIES_INPUTS = set(ind.REQUIRED_COLUMNS)                                   # OHLCV, esqueleto y mascara de validez
PRICE_COLUMNS = ["open", "high", "low", "close"]
QUANTILES = (0.25, 0.5, 0.75)
# Componente 2 (propuesta de E02, seccion 4.3): (parametro, origen, etiqueta).
PARAMETER_ORIGINS = [
    ("PARAMETROS_CATALOGO", "fijados a priori (D2)", "SIN_ESTIMACION"),
    ("UMBRALES_TAU_TOLERANCIAS_SEMILLAS", "protocolo congelado, antes de ver datos", "SIN_ESTIMACION"),
    ("G2_2_SIGMA_Y_CURTOSIS_RECORTADA", "TDA-07, muestra completa anterior al holdout (solo nulo G2-2)", "RETROSPECTIVO"),
    ("CORTES_DE_SEGMENTO", "TDA-06, muestra completa (definen celdas)", "RETROSPECTIVO"),
    ("MEDIA_MINUTO_X_ANIO_S5", "poblacion de analisis (solo S5)", "RETROSPECTIVO"),
    ("R2_PCA_BOOTSTRAP_CLASIFICACION", "poblacion de analisis", "RETROSPECTIVO"),
]


def audit_config(cfg: dict) -> dict:
    """Seccion de E02; si no es la implementada, el motor no aplica otra en silencio."""
    ac = an.e02_value(cfg, "analisis.auditoria_causalidad")
    fut = ac["futuro"]
    if ac["tabla"]["campos"] != FIELDS or ac["tabla"]["pruebas"] != TESTS or ac["tabla"]["resultados"] != RESULTS or \
            fut["modos"] != ["TRUNCAMIENTO", "PERTURBACION_FUTURA"] or fut["unidad"] != "CONTRATO" or \
            fut["vecindad_del_evento"] != [-1, 0, 1] or fut["d7_rvol"] != "TRUNCAMIENTO_GLOBAL" or \
            fut["igualdad"] != "EXACTA_BIT_A_BIT_INCLUIDOS_NAN_MARCAS_Y_MASCARAS" or \
            [float(q) for q in fut["cortes_deterministas"]["cuantiles"]] != list(QUANTILES) or \
            fut["no_aplicable"]["sustitucion"] != "NINGUNA":
        raise ProtocolNotAuthorizedError("auditoria causal distinta de la implementada")
    return ac


def _catalog(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Catalogo por la UNICA via de produccion de la fase 1 (``compute_catalog``: protocolo congelado + E01
    verificadas). E02 no cambia el catalogo (``check_catalog_config``), de modo que es el mismo calculo que el de las
    salidas oficiales."""
    return ind.compute_catalog(frame.reset_index(drop=True))


def check_catalog_config(cfg: dict) -> None:
    """La configuracion recibida tiene exactamente el catalogo, las reglas de frontera y el calentamiento de la
    configuracion efectiva de E01 con la que calcula ``compute_catalog``."""
    from ohlcv_dataroad.ingest.tda17_enmiendas import load_effective_protocol
    base = load_effective_protocol()
    if any(cfg.get(k) != base.get(k) for k in ("catalogo", "reglas_frontera", "calentamiento")):
        raise ProtocolNotAuthorizedError("el catalogo de la configuracion no es el de la fase 1 (E01)")


# --------------------------------------------------------------------------- comparacion exacta
def compare_arrays(a, b) -> tuple[int, float]:
    """(discrepancias, max |a - b| entre las finitas que difieren) con igualdad BIT A BIT: mismos bits en los
    flotantes (NaN en la misma posicion; -0.0 != 0.0), mismo valor en los demas tipos (ausentes en la misma
    posicion). Tipos distintos: todas las filas discrepan."""
    sa, sb = pd.Series(a), pd.Series(b)
    if len(sa) != len(sb):
        raise ValueError("comparacion de longitudes distintas")
    if sa.dtype != sb.dtype:
        return len(sa), math.nan
    if sa.dtype.kind == "f":
        va, vb = sa.to_numpy(), sb.to_numpy()
        ui = {2: np.uint16, 4: np.uint32, 8: np.uint64}[va.dtype.itemsize]
        diff_mask = va.view(ui) != vb.view(ui)
        d = np.abs(va.astype(np.float64) - vb.astype(np.float64))[diff_mask]
        d = d[np.isfinite(d)]
        return int(diff_mask.sum()), float(d.max()) if d.size else (0.0 if not diff_mask.any() else math.nan)
    same = (sa.astype(object) == sb.astype(object)).fillna(False).to_numpy(bool) | (sa.isna() & sb.isna()).to_numpy(bool)
    return int((~same).sum()), 0.0 if same.all() else math.nan


def _row(prueba, contrato, corte, motivo, serie, n, disc, mad, resultado, parametro=None, origen=None, etiqueta=None):
    return dict(zip(FIELDS, [prueba, contrato, corte, motivo, serie, n, disc, mad, resultado, parametro, origen, etiqueta]))


def _compare_frame(prueba, contrato, corte, motivo, full: pd.DataFrame, part: pd.DataFrame, columns: list[str]) -> list[dict]:
    out = []
    for col in columns:
        disc, mad = compare_arrays(full[col].reset_index(drop=True), part[col].reset_index(drop=True))
        out.append(_row(prueba, contrato, corte, motivo, col, len(full), disc, mad, "CONFORME" if disc == 0 else "DISCREPANCIA"))
    return out


# --------------------------------------------------------------------------- entrada
def check_unadjusted_input(df: pd.DataFrame) -> None:
    """La entrada solo contiene las columnas de TDA-03/TDA-04 (serie canonica sin ajustes de roll, variables y
    mascara de validez): una columna desconocida (p. ej. precios ajustados) detiene la auditoria."""
    extra = sorted(set(df.columns) - KNOWN_INPUT_COLUMNS)
    if extra:
        raise ProtocolNotAuthorizedError(f"la entrada contiene columnas no declaradas (posibles ajustes): {extra}")
    missing = sorted(set(CANONICAL_COLUMNS) - set(df.columns))
    if missing:
        raise ProtocolNotAuthorizedError(f"faltan columnas de la serie canonica: {missing}")


def _d7_columns(cfg: dict) -> tuple[list[str], list[str]]:
    el = next(e for e in cfg["catalogo"] if e["id"] == "D7")
    series = [f"D7_{c['id']}__{s['id']}" for c in el["configuraciones"] for s in el["salidas"]]
    marks = [f"D7_{c['id']}__rvol_cruza_roll" for c in el["configuraciones"]]
    return series, marks


def _rvol(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """D7 de toda la entrada (series y marcas), con el calendario de la propia entrada."""
    days = ind.calendar_trading_days(df)
    el = next(e for e in cfg["catalogo"] if e["id"] == "D7")
    out = {}
    for c in el["configuraciones"]:
        rv, cross = ind.relative_volume(df, c["params"]["k_jornadas"], days)
        for s in el["salidas"]:
            out[f"D7_{c['id']}__{s['id']}"] = rv
        out[f"D7_{c['id']}__rvol_cruza_roll"] = cross
    return pd.DataFrame(out, index=df.index)


def _tda04(df: pd.DataFrame) -> pd.DataFrame:
    """Variables de TDA-04 de P (y la mascara de validez) recalculadas con las funciones de TDA-04 sobre la serie
    canonica recibida; ``ln_volume`` = ln V como en el panel P."""
    canon = df[CANONICAL_COLUMNS].reset_index(drop=True)
    roll = pd.DataFrame({"timestamp": canon["timestamp"], "is_roll_boundary": df["is_roll_boundary"].to_numpy(bool)})
    validity = build_return_validity_mask(canon, roll)
    var = build_analysis_variables(canon, validity)
    out = var[TDA04_VARIABLES].copy()
    for c in ind.VALIDITY_COLUMNS:
        out[c] = validity[c].to_numpy()
    out["ln_volume"] = pr.safe_log(canon["volume"].to_numpy(float))
    return out


def perturb_after(frame: pd.DataFrame, start: int, *, prices: bool = True, volume: bool = True) -> pd.DataFrame:
    """Copia con las filas de posicion >= ``start`` perturbadas en la rejilla (k >= 1 = distancia al corte): toda la
    barra + s = (k mod 5 + 1) ticks; H + (k mod 3 + 1), L - ((k + 1) mod 3 + 1) y C + (k mod 2) ticks adicionales
    (cambian rango, cuerpo y mechas, no solo el nivel); volumen + (k mod 4 + 1). Conserva el esqueleto y las
    invariantes OHLC (H >= max(O, C), L <= min(O, C))."""
    out = frame.copy()
    k = np.arange(len(out)) - start + 1
    sel = k >= 1
    if prices:
        s = pr.TICK * ((k % 5) + 1)
        extra = {"open": 0.0, "high": pr.TICK * ((k % 3) + 1), "low": -pr.TICK * (((k + 1) % 3) + 1),
                 "close": pr.TICK * (k % 2)}
        for col in PRICE_COLUMNS:
            v = out[col].to_numpy(float)
            out[col] = np.where(sel, v + s + extra[col], v)
    if volume:
        out["volume"] = np.where(sel, out["volume"].to_numpy(float) + (k % 4) + 1, out["volume"].to_numpy(float))
    return out


# --------------------------------------------------------------------------- puntos de corte (predeclarados)
def contract_cut_plan(df: pd.DataFrame, series: pd.DataFrame, marks: pd.DataFrame, cfg: dict) -> list[dict]:
    """Plan de cortes por contrato (solo esqueleto y calculo completo, nunca resultados del analisis):
    ``[{contrato, a, z, barras, cortes: {c: [motivos]}, no_aplicables: [(c o None, motivo, razon)]}]``."""
    ac = audit_config(cfg)
    fut = ac["futuro"]
    df = df.reset_index(drop=True)
    segs = ind.contract_segments(df)
    warm = an.warmup_table(df, series, cfg)
    classes = an._element_classes(cfg)
    kmax = cfg["calentamiento"]["K_maximo"]
    if int(np.nanmax(warm.loc[warm["clase"] != "JORNADAS", "K_teorico"].astype(float))) != kmax:
        raise ProtocolNotAuthorizedError("el registro de calentamiento no contiene K_max")
    rnd = fut["cortes_aleatorios"]
    streams = np.random.SeedSequence(rnd["semilla"]).spawn(len(segs))
    long_gap = df["delta_minutes"].to_numpy(float) > ind.LONG_GAP_MIN
    tipo = marks["tipo_par"].astype(object).to_numpy()
    plan = []
    for i, (a, z) in enumerate(segs):
        n = z - a
        name = df["contract"].iloc[a]
        events: list[tuple[str, float | None, str]] = []          # (motivo, barra del evento o None, razon si falta)

        def first_last(mask, label):
            idx = np.flatnonzero(mask) + 1
            events.append((f"PRIMER_{label}", float(idx[0]) if idx.size else None, "EVENTO_INEXISTENTE"))
            events.append((f"ULTIMO_{label}", float(idx[-1]) if idx.size else None, "EVENTO_INEXISTENTE"))

        t, lg = tipo[a:z], long_gap[a:z]
        first_last(t == ind.PAIR_FJ, "FJ")
        first_last(t == ind.PAIR_FJL, "FJL")
        first_last((t == ind.PAIR_HI) & ~lg, "HI_CORTO")
        first_last((t == ind.PAIR_HI) & lg, "HI_LARGO")
        events += [("BARRA_1", 1.0, ""), ("BARRA_2", 2.0, ""), ("BARRA_3", 3.0, ""), ("ULTIMA_BARRA", float(n), "")]
        w = warm[warm["contrato"] == name]
        n_days = df["trading_date"].iloc[a:z].nunique()
        for (k, cls), g in w.groupby(["K_teorico", "clase"], sort=True):
            vals = sorted(set(g["primera_barra_elegible"].dropna().astype(float)))
            if vals:
                events += [(f"ELEGIBLE_K{int(k)}_{cls}", v, "") for v in vals]
            else:
                short = (n_days < k) if cls == "JORNADAS" else (n < k)
                events.append((f"ELEGIBLE_K{int(k)}_{cls}", None, "CONTRATO_CORTO" if short else "EVENTO_INEXISTENTE"))
        for rec in w.itertuples():
            el = rec.serie.split("_", 1)[0]
            if el == "A5" or (el == "A8" and rec.serie.endswith("__adx")):
                act = rec.primera_activacion
                reason = "CONTRATO_CORTO" if not np.isfinite(rec.primera_barra_elegible) and n < rec.K_teorico else "EVENTO_INEXISTENTE"
                events.append((f"ACTIVACION_{rec.serie}", float(act) if np.isfinite(act) else None, reason))
        for sid in [c for c in series.columns if c.startswith("A11_") and c.endswith("__dir")]:
            d = series[sid].to_numpy(float)[a:z]
            ch = np.flatnonzero(np.isfinite(d[1:]) & np.isfinite(d[:-1]) & (d[1:] != d[:-1])) + 2
            events.append((f"PRIMER_CAMBIO_DIR_{sid}", float(ch[0]) if ch.size else None, "EVENTO_INEXISTENTE"))
            events.append((f"ULTIMO_CAMBIO_DIR_{sid}", float(ch[-1]) if ch.size else None, "EVENTO_INEXISTENTE"))
        cons = df["r_1m_valid"].to_numpy(bool)[a:z].copy()
        cons[0] = False
        for el in cfg["catalogo"]:
            if classes.get(el["id"]) != "VENTANA_VALIDEZ":
                continue
            for conf in el["configuraciones"]:
                e = an.structural_eligibility("VENTANA_VALIDEZ", el, conf, cons)
                f0 = np.flatnonzero(e)
                loss = np.flatnonzero(~e[f0[0]:]) + f0[0] if f0.size else np.array([], int)
                rec_ = np.flatnonzero(e[loss[0]:]) + loss[0] if loss.size else np.array([], int)
                tag = f"{el['id']}_{conf['id']}"
                events.append((f"VV_PERDIDA_{tag}", float(loss[0] + 1) if loss.size else None, "EVENTO_INEXISTENTE"))
                events.append((f"VV_RECUPERACION_{tag}", float(rec_[0] + 1) if rec_.size else None, "EVENTO_INEXISTENTE"))
        events += [(f"CUANTIL_{int(q * 100)}", float(math.ceil(q * n)), "") for q in QUANTILES]
        cuts: dict[int, list[str]] = {}
        na: list[tuple[int | None, str, str]] = []
        for motive, bar, reason in events:
            if bar is None:
                na.append((None, motive, reason))
                continue
            for off in fut["vecindad_del_evento"]:
                c = int(bar) + off
                lab = f"{motive}{off:+d}" if off else motive
                if 2 <= c <= n - 1:
                    cuts.setdefault(c, []).append(lab)
                else:
                    na.append((c, f"{lab} (corte fuera de 2..{n - 1})", "CONTRATO_CORTO"))
        if n >= 3:
            rng = np.random.Generator(np.random.PCG64(streams[i]))
            for k, c in enumerate(rng.integers(2, n, size=rnd["por_contrato"]).tolist()):
                cuts.setdefault(int(c), []).append(f"ALEATORIO_{k + 1}")
        else:
            na.append((None, "ALEATORIOS", "CONTRATO_CORTO"))
        plan.append({"contrato": name, "a": a, "z": z, "barras": n, "cortes": dict(sorted(cuts.items())),
                     "no_aplicables": na})
    return plan


# --------------------------------------------------------------------------- componente 1: informacion futura
def audit_future(df: pd.DataFrame, cfg: dict, plan: list[dict], full: tuple[pd.DataFrame, pd.DataFrame]) -> list[dict]:
    """TRUNCAMIENTO y PERTURBACION_FUTURA para los cortes de ``plan`` (``contract_cut_plan`` o un subconjunto),
    comparando con ``full`` = ``(series, marcas)`` del calculo completo."""
    audit_config(cfg)
    df = df.reset_index(drop=True)
    series_f, marks_f = full
    d7s, d7m = _d7_columns(cfg)
    ser_cols = [c for c in series_f.columns if c not in d7s]
    mark_cols = [c for c in marks_f.columns if c not in d7m]
    rvol_f = _rvol(df, cfg)
    tda04_f = df[TDA04_VARIABLES + ind.VALIDITY_COLUMNS].copy()
    tda04_f["ln_volume"] = pr.safe_log(df["volume"].to_numpy(float))
    tda04_cols = list(tda04_f.columns)
    rows: list[dict] = []
    for p in plan:
        a, z, name = p["a"], p["z"], p["contrato"]
        for c, motive, reason in p["no_aplicables"]:
            for mode in ("TRUNCAMIENTO", "PERTURBACION_FUTURA"):
                rows.append(_row(mode, name, c, f"{motive} [{reason}]", None, 0, None, None, "NO_APLICABLE"))
        contract = df.iloc[a:z].reset_index(drop=True)
        for c, motives in p["cortes"].items():
            motive = ";".join(motives)
            g = a + c                                                   # filas globales 0..g-1 conservadas
            fs, fm = series_f.iloc[a:g], marks_f.iloc[a:g]
            st, mt = _catalog(contract.iloc[:c])
            rows += _compare_frame("TRUNCAMIENTO", name, c, motive, fs, st, ser_cols)
            rows += _compare_frame("TRUNCAMIENTO", name, c, motive, fm, mt, mark_cols)
            sp, mp = _catalog(perturb_after(contract, c))
            rows += _compare_frame("PERTURBACION_FUTURA", name, c, motive, fs, sp.iloc[:c], ser_cols)
            rows += _compare_frame("PERTURBACION_FUTURA", name, c, motive, fm, mp.iloc[:c], mark_cols)
            rows += _compare_frame("TRUNCAMIENTO", name, c, motive, rvol_f.iloc[:g], _rvol(df.iloc[:g].reset_index(drop=True), cfg), d7s + d7m)
            rows += _compare_frame("PERTURBACION_FUTURA", name, c, motive, rvol_f.iloc[:g],
                                   _rvol(perturb_after(df, g, prices=False), cfg).iloc[:g], d7s + d7m)
            rows += _compare_frame("TRUNCAMIENTO", name, c, motive, tda04_f.iloc[:g], _tda04(df.iloc[:g]), tda04_cols)
            rows += _compare_frame("PERTURBACION_FUTURA", name, c, motive, tda04_f.iloc[:g],
                                   _tda04(perturb_after(df, g)).iloc[:g], tda04_cols)
    return rows


# --------------------------------------------------------------------------- componente 2: origen de parametros
def audit_parameter_origin(df: pd.DataFrame, cfg: dict, full: tuple[pd.DataFrame, pd.DataFrame]) -> list[dict]:
    """Tabla declarada de origen de parametros y prueba dinamica: alterar todas las columnas de la entrada que no
    son OHLCV, esqueleto ni mascara de validez (variables de TDA-04, fichero, segmento) no cambia ninguna serie ni
    marca. El resultado de la prueba se asigna a cada fila declarada."""
    audit_config(cfg)
    df = df.reset_index(drop=True)
    alt = df.copy()
    for col in sorted(set(df.columns) - SERIES_INPUTS):
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            alt[col] = df[col].to_numpy(float) * -3.0 + 7.0
        else:
            alt[col] = df[col].astype(str) + "_X"
    s2, m2 = _catalog(alt)
    checks = _compare_frame("ORIGEN_PARAMETRO", None, None, "SOLO_OHLCV_Y_MASCARA", full[0], s2, list(full[0].columns)) + \
        _compare_frame("ORIGEN_PARAMETRO", None, None, "SOLO_OHLCV_Y_MASCARA", full[1], m2, list(full[1].columns))
    ok = all(r["resultado"] == "CONFORME" for r in checks)
    bad = sum(r["discrepancias"] for r in checks)
    return [_row("ORIGEN_PARAMETRO", None, None, "SOLO_OHLCV_Y_MASCARA", "TODAS", len(df), bad, None,
                 "CONFORME" if ok else "DISCREPANCIA", par, org, lab) for par, org, lab in PARAMETER_ORIGINS] + \
        [r for r in checks if r["resultado"] != "CONFORME"]


# --------------------------------------------------------------------------- componente 3: roll (3a)
ROLL_TRANSFORMS = {
    "MULTIPLICATIVA": lambda v, rows: v * 2.0,
    "ADITIVA": lambda v, rows: v + 1000 * pr.TICK,
    "ARBITRARIA": lambda v, rows: v + pr.TICK * ((rows % 7) + 1),
}


def audit_roll(df: pd.DataFrame, cfg: dict, full: tuple[pd.DataFrame, pd.DataFrame]) -> list[dict]:
    """ROLL_PRECIO: para cada contrato, los precios de los DEMAS transformados de tres formas (en la rejilla y con
    las invariantes OHLC): series, marcas y variables de TDA-04 del contrato identicas bit a bit. ROLL_VOLUMEN:
    volumen de los demas perturbado: solo D7 puede cambiar y solo en las filas con rvol_cruza_roll."""
    ac = audit_config(cfg)
    if ac["roll"]["precio"]["transformaciones"] != list(ROLL_TRANSFORMS) or ac["roll"]["volumen"]["solo_puede_cambiar"] != "D7":
        raise ProtocolNotAuthorizedError("auditoria de roll distinta de la implementada")
    df = df.reset_index(drop=True)
    check_unadjusted_input(df)
    series_f, marks_f = full
    d7s, d7m = _d7_columns(cfg)
    tda_f = _tda04(df)
    rows_idx = np.arange(len(df))
    out = [_row("ROLL_PRECIO", None, None, "ENTRADA_SIN_AJUSTES", "COLUMNAS_DE_ENTRADA", len(df), 0, 0.0, "CONFORME")]
    for a, z in ind.contract_segments(df):
        name = df["contract"].iloc[a]
        other = (rows_idx < a) | (rows_idx >= z)
        for tname, fn in ROLL_TRANSFORMS.items():
            alt = df.copy()
            for col in PRICE_COLUMNS:
                v = alt[col].to_numpy(float)
                alt[col] = np.where(other, fn(v, rows_idx), v)
            s2, m2 = _catalog(alt)
            motive = f"PRECIO_{tname}"
            out += _compare_frame("ROLL_PRECIO", name, None, motive, series_f.iloc[a:z], s2.iloc[a:z], list(series_f.columns))
            out += _compare_frame("ROLL_PRECIO", name, None, motive, marks_f.iloc[a:z], m2.iloc[a:z], list(marks_f.columns))
            out += _compare_frame("ROLL_PRECIO", name, None, motive, tda_f.iloc[a:z], _tda04(alt).iloc[a:z], list(tda_f.columns))
        alt = df.copy()
        v = alt["volume"].to_numpy(float)
        alt["volume"] = np.where(other, v + (rows_idx % 5) + 1, v)
        s2, m2 = _catalog(alt)
        non_d7 = [c for c in series_f.columns if c not in d7s]
        out += _compare_frame("ROLL_VOLUMEN", name, None, "VOLUMEN_DEMAS_CONTRATOS", series_f.iloc[a:z], s2.iloc[a:z], non_d7)
        out += _compare_frame("ROLL_VOLUMEN", name, None, "VOLUMEN_DEMAS_CONTRATOS", marks_f.iloc[a:z], m2.iloc[a:z], list(marks_f.columns))
        for col in d7s:
            cid = col.split("__")[0]
            cross = marks_f[f"{cid}__rvol_cruza_roll"].to_numpy(bool)[a:z]
            fa, fb = series_f[col].iloc[a:z].to_numpy(), s2[col].iloc[a:z].to_numpy()
            disc, mad = compare_arrays(fa[~cross], fb[~cross])
            out.append(_row("ROLL_VOLUMEN", name, None, "VOLUMEN_DEMAS_CONTRATOS (filas sin rvol_cruza_roll)", col,
                            int((~cross).sum()), disc, mad, "CONFORME" if disc == 0 else "DISCREPANCIA"))
    return out


# --------------------------------------------------------------------------- tabla y consecuencias
def audit_table(rows: list[dict]) -> pd.DataFrame:
    t = pd.DataFrame(rows, columns=FIELDS)
    if not t["prueba"].isin(TESTS).all() or not t["resultado"].isin(RESULTS).all():
        raise ValueError("fila de auditoria con prueba o resultado no declarados")
    return t


def causal_consequences(table: pd.DataFrame) -> dict:
    """Series con alguna DISCREPANCIA: fuera de las salidas oficiales; TH35 bloqueada hasta corregir y repetir la
    auditoria (sin tolerancia ni excepcion automatica)."""
    bad = table[table["resultado"] == "DISCREPANCIA"]
    series = sorted(set(bad["serie"].dropna()))
    return {"series_excluidas": series, "bloqueos": [f"DISCREPANCIA_CAUSAL: {s}" for s in series]}


def _reject(cid: str, msg: str, exc=ProtocolNotAuthorizedError):
    raise exc(f"[{cid}] {msg}")


def _key(row: dict) -> tuple:
    """Identidad de una fila de la tabla: (prueba, contrato, punto_corte, motivo_corte, serie, parametro)."""
    def norm(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return int(v) if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) else v
    return tuple(norm(row[f]) for f in ("prueba", "contrato", "punto_corte", "motivo_corte", "serie", "parametro"))


def expected_audit_rows(df: pd.DataFrame, cfg: dict) -> dict[tuple, str]:
    """Filas que DEBE contener la auditoria completa de ``df`` segun el plan predeclarado (componentes 1, 2 y 3):
    {clave: 'NO_APLICABLE' o 'COMPARACION'}. Se deriva del esqueleto y del catalogo, nunca de la tabla auditada."""
    audit_config(cfg)
    df = df.reset_index(drop=True)
    series_f, marks_f = _catalog(df)
    plan = contract_cut_plan(df, series_f, marks_f, cfg)
    d7s, d7m = _d7_columns(cfg)
    tda04_cols = TDA04_VARIABLES + ind.VALIDITY_COLUMNS + ["ln_volume"]
    future_cols = [c for c in series_f.columns if c not in d7s] + [c for c in marks_f.columns if c not in d7m] + \
        d7s + d7m + tda04_cols
    out: dict[tuple, str] = {}
    for p in plan:
        for c, motive, reason in p["no_aplicables"]:
            for mode in ("TRUNCAMIENTO", "PERTURBACION_FUTURA"):
                out[(mode, p["contrato"], c, f"{motive} [{reason}]", None, None)] = "NO_APLICABLE"
        for c, motives in p["cortes"].items():
            for mode in ("TRUNCAMIENTO", "PERTURBACION_FUTURA"):
                for col in future_cols:
                    out[(mode, p["contrato"], c, ";".join(motives), col, None)] = "COMPARACION"
    for par, _, _ in PARAMETER_ORIGINS:
        out[("ORIGEN_PARAMETRO", None, None, "SOLO_OHLCV_Y_MASCARA", "TODAS", par)] = "COMPARACION"
    out[("ROLL_PRECIO", None, None, "ENTRADA_SIN_AJUSTES", "COLUMNAS_DE_ENTRADA", None)] = "COMPARACION"
    for a, _ in ind.contract_segments(df):
        name = df["contract"].iloc[a]
        for tname in ROLL_TRANSFORMS:
            for col in list(series_f.columns) + list(marks_f.columns) + tda04_cols:
                out[("ROLL_PRECIO", name, None, f"PRECIO_{tname}", col, None)] = "COMPARACION"
        for col in [c for c in series_f.columns if c not in d7s] + list(marks_f.columns):
            out[("ROLL_VOLUMEN", name, None, "VOLUMEN_DEMAS_CONTRATOS", col, None)] = "COMPARACION"
        for col in d7s:
            out[("ROLL_VOLUMEN", name, None, "VOLUMEN_DEMAS_CONTRATOS (filas sin rvol_cruza_roll)", col, None)] = "COMPARACION"
    return out


def verify_causal_audit(table: pd.DataFrame | None, df: pd.DataFrame, cfg: dict) -> dict:
    """Barrera TH37 de las conclusiones de TH35: la auditoria de ``df`` debe existir (TH37_AUSENTE), tener el esquema
    de E02 y EXACTAMENTE las filas del plan predeclarado, sin repetidas ni ajenas (TH37_INCOMPLETA,
    TH37_FUERA_DEL_PLAN; solo se admiten las filas de detalle de una DISCREPANCIA de ORIGEN_PARAMETRO), con el
    resultado coherente (NO_APLICABLE solo donde el plan lo preve) y sin ninguna DISCREPANCIA (TH37_DISCREPANCIA:
    BlockedUntilExplainedError). Devuelve las consecuencias (sin bloqueos)."""
    if table is None or not isinstance(table, pd.DataFrame) or table.empty:
        _reject("TH37_AUSENTE", "no hay auditoria causal de la entrada: TH35 no puede publicarse")
    if list(table.columns) != FIELDS:
        _reject("TH37_INCOMPLETA", "la tabla de auditoria no tiene el esquema de E02")
    expected = expected_audit_rows(df, cfg)
    rows = table.to_dict("records")
    seen: dict[tuple, int] = {}
    for r in rows:
        k = _key(r)
        seen[k] = seen.get(k, 0) + 1
    missing = [k for k in expected if k not in seen]
    if missing:
        _reject("TH37_INCOMPLETA", f"faltan {len(missing)} filas del plan de la auditoria (p. ej. {missing[0]})")
    repeated = [k for k, n in seen.items() if n > 1]
    detail = {_key(r) for r in rows if r["prueba"] == "ORIGEN_PARAMETRO" and r["resultado"] == "DISCREPANCIA"
              and _key(r)[5] is None}
    foreign = [k for k in seen if k not in expected and k not in detail]
    if repeated or foreign:
        _reject("TH37_FUERA_DEL_PLAN", f"filas repetidas o ajenas al plan: {len(repeated)} repetidas, {len(foreign)} ajenas")
    wrong = [_key(r) for r in rows if _key(r) in expected and
             (r["resultado"] == "NO_APLICABLE") != (expected[_key(r)] == "NO_APLICABLE")]
    if wrong:
        _reject("TH37_INCOMPLETA", f"{len(wrong)} filas con un resultado incoherente con el plan (p. ej. {wrong[0]})")
    cons = causal_consequences(table)
    if cons["series_excluidas"]:
        _reject("TH37_DISCREPANCIA", f"TH35 BLOQUEADA_HASTA_EXPLICAR: series con discrepancia causal "
                f"{cons['series_excluidas']}", an.BlockedUntilExplainedError)
    return cons


def run_causal_audit(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Auditoria completa (componentes 1, 2 y 3) con el plan de cortes completo, sobre la entrada de TDA-04."""
    audit_config(cfg)
    check_catalog_config(cfg)
    check_unadjusted_input(df)
    df = df.reset_index(drop=True)
    full = _catalog(df)
    plan = contract_cut_plan(df, *full, cfg)
    return audit_table(audit_future(df, cfg, plan, full) + audit_parameter_origin(df, cfg, full) + audit_roll(df, cfg, full))


def official_causal_audit(publicacion: str, root=None) -> tuple[pd.DataFrame, dict]:
    """PRODUCCION: auditoria causal completa de las entradas OFICIALES (leidas y verificadas contra el manifiesto por
    ``tda17_entradas``; el llamante no aporta datos), con la configuracion de la ejecucion oficial (P2), verificada
    contra el plan predeclarado. Devuelve (tabla, procedencia)."""
    from ohlcv_dataroad.ingest import tda17_entradas as ent
    cfg = an.official_config(publicacion, root)
    df, procedencia = ent.load_official_inputs(cfg, root)
    table = run_causal_audit(df, cfg)
    verify_causal_audit(table, df, cfg)
    return table, procedencia
