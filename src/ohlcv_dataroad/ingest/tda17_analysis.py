"""TDA-17 -- fase 2: motor estadistico de redundancia informacional (TH35, TH37).

Implementa las secciones ``analisis``, ``identidades`` y ``sensibilidades`` de la configuracion EFECTIVA
(protocolo CONGELADO 9cdea2f + enmienda E01 APROBADA + enmienda E02, cargador escalonado M2
``tda17_enmiendas_e02``) sobre los paneles que calcula la fase 1 (``tda17_indicators``). Este modulo no lee ni
escribe ficheros, no abre datos brutos ni el holdout y no ejecuta ningun analisis por si mismo: cada funcion recibe
matrices o DataFrames. Las salidas oficiales se escriben solo con ``tda17_salidas`` (ejecucion oficial, P2).

Contenido (en el orden del protocolo):
1. Paneles oficiales P (134), B (123 = P sin las 11 columnas ``fuera_de_B``) y L (16), con las 6 referencias de
   TDA-04 tal como vienen de la entrada (sin recalcularlas; ``ln_volume`` = ln V, ``origen: volume``).
2. Controles de entrada (``entradas.esperado``), poblacion (casos completos de P, perdidas por columna) y
   registro de calentamiento (``calentamiento.registro`` de E01).
3. Celdas: global, anios (anio NY del timestamp; 2019 NO_EVALUABLE, 2025 parcial), 7 segmentos de TDA-07 y
   anio x segmento solo descriptivo.
4. Estadisticos de celda: una QR por bloques de la matriz centrada (factor R = estadistico suficiente de MCO),
   Pearson, Spearman (rango medio dentro de la celda) y Pearson winsorizado (S3). Matriz NO_DEFINIDA (E02, F5-a)
   si alguna columna tiene varianza cero (o norma cero) en la poblacion de la celda.
5. R2 de cada columna frente a las demas: base especifica por objetivo (Cholesky con pivoteo, tau relativa),
   verificada con la pseudoinversa espectral y con MCO de minima norma sobre el factor R; etiquetas de
   exactitud por registro y residuo. Recuento de nulos con la convencion de E02 (A1-a: autovalores de la
   correlacion o de los cosenos; valores singulares solo informativos), ambito del control (A4-a) y, si bloquea,
   diagnosticos obligatorios de BLOQUEADO_HASTA_EXPLICAR (A2-a).
6. PCA (P oficial, B complementario): k80, k90, k95, ratio de participacion, rango efectivo por entropia y
   M_eff de Li y Ji. Agrupamiento jerarquico de enlace medio con distancia 1 - |rho_spearman|.
7. Bootstrap de bloques moviles circulares de jornadas por celda (E02, B2-a) con un flujo PCG64 por celda
   (B6), por estadisticos suficientes diarios; IC percentil 0,05/0,95 (B1-a) de R2_TODAS (metodo oficial en
   cada replica, replicas invalidas -> IC NO_CALCULABLE, B3-a + F2-a) y de Spearman con rangos fijos, validados
   contra el recalculo exacto (F3-I + F3-I-b; una replica INVALIDA hace fallar la validacion de la celda en esa
   ronda, S-a). Clasificacion RECONSTRUIBLE / NUEVA_REPRESENTACION / INESTABLE con las celdas no publicables como
   incumplimiento (B13-a) y subniveles con el R2 global (B9; SIN_SUBNIVEL_R2_NO_PUBLICABLE si no es publicable, N-a).
   Diagnostico de bloqueo con las relaciones N1-N21 clasificadas por tipo segun el anexo de E02 (R-a corregida).
8. Mascaras de sensibilidad S1 (marcas de E01-B, regla de union), S3, S5, S6; S2 bloqueada sin asignacion de
   escalas; diagnostico auxiliar del panel L por contrato.

La auditoria causal de TH37 (E02, D17-1a) esta en ``tda17_causalidad``.

Lo que la configuracion efectiva no define con precision suficiente NO se resuelve aqui: cada punto tiene una clave
en ``PENDIENTES`` y la via afectada lanza ``PendingDefinitionError`` (subclase de ``ProtocolNotAuthorizedError``).
Las funciones de bajo nivel reciben esos parametros de forma EXPLICITA y sin valor por defecto, para poder probar
su mecanica sin elegir un criterio. Las rutas de produccion obtienen la configuracion SOLO con
``production_config()`` (aprobacion preparatoria) u ``official_config(P2)`` (ejecucion oficial), nunca con el
cargador de E01.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

from ohlcv_dataroad.ingest import tda17_indicators as ind
from ohlcv_dataroad.ingest import tda17_primitives as pr

ProtocolNotAuthorizedError = ind.ProtocolNotAuthorizedError


class PendingDefinitionError(ProtocolNotAuthorizedError):
    """La configuracion efectiva no define este punto con precision suficiente: no se ejecuta hasta que una
    enmienda aprobada lo defina (nunca se elige un criterio en el codigo)."""


class BlockedUntilExplainedError(ProtocolNotAuthorizedError):
    """Conclusion bloqueada (E02, procedimiento_bloqueo): hay un hallazgo bloqueante en su alcance. Solo una
    enmienda formal posterior auditada la levanta; no hay desbloqueo automatico."""


# Puntos sin definicion normativa suficiente en la configuracion efectiva (ninguno se resuelve en el codigo).
# E02 resolvio IC_NIVEL, BOOTSTRAP_CELDAS, SPEARMAN_VALIDACION, CONVENCION_NULOS y SUBNIVEL_R2; su autorizacion
# complementaria (decisiones 14-19) resolvio SPEARMAN_VALIDACION_REPLICA_INVALIDA, SUBNIVEL_R2_GLOBAL_NO_PUBLICABLE,
# RELACIONES_N_VALOR y SALIDAS_ESQUEMA_VINCULACION (esta ultima solo para las cinco tablas del esquema minimo).
PENDIENTES = {
    "R2_OTROS_INDICADORES": "variante R2_OTROS_INDICADORES: no se fija si excluye solo las referencias de TDA-04 o "
                            "tambien las demas salidas/configuraciones del mismo elemento",
    "R2_TRANSFERENCIA": "transferencia: no se fija sobre que muestra se mide R2_in ni respecto de que media se calcula "
                        "la suma de cuadrados total de la celda retenida",
    "HUECO_ESPECTRAL": "decision 11 de E02 APLAZADA: el 'hueco espectral' no esta definido (que cociente de "
                       "autovalores, de que matriz); la publicacion de EXACTA por celda (SEPARACION_NUMERICA_"
                       "INSUFICIENTE) queda BLOQUEADA",
    "T3_V2": "T3 (materialidad 0,10; falsas <= 5 %) y el criterio V2 no estan definidos operativamente",
    "K_PA_MAX_T_V3": "analisis paralelo: la banda global max-T, la 'secuencia fija' y V3 'dejando una replica fuera' "
                     "no estan definidos operativamente (estandarizacion, cuantil, regla de parada)",
    "S7_S8_ALCANCE": "S7 y S8: no se fija si la exclusion es global o solo en los calculos en que intervienen D7 / "
                     "C10, ni con que configuracion (k5/k20, n14/n60)",
    "REGISTRO_D7_RETRASO": "registro de calentamiento de D7 (K en jornadas): el 'retraso' en barras no esta definido",
    "S2_ESCALAS": "S2: falta la asignacion de cada configuracion a las escalas S, M o L "
                  "(tda17_enmiendas.require_s2_scale_assignment)",
    "SALIDAS_OTRAS_TABLAS": "salidas.tablas: E02 (salidas.esquema_minimo) solo fija el esquema y la vinculacion de "
                            "cinco tablas y del diagnostico de bloqueo; las demas (catalogo, ventanas, identidades, "
                            "perdidas, Pearson, transferencia, dimension, agrupamiento, sensibilidades, G2, panel L, "
                            "rvol) siguen sin esquema y no se publican",
}


def pending(key: str):
    """Lanza ``PendingDefinitionError`` para un punto de ``PENDIENTES``."""
    raise PendingDefinitionError(f"{key}: definicion pendiente -- {PENDIENTES[key]}")


def e02_value(cfg: dict, ruta: str):
    """Valor de una clave anadida por E02. Una configuracion sin E02 (p. ej. la de E01) se rechaza: el motor nunca
    vuelve al comportamiento anterior a la enmienda."""
    node = cfg
    for seg in ruta.split("."):
        if not isinstance(node, dict) or seg not in node:
            raise ProtocolNotAuthorizedError(f"la configuracion no incluye la enmienda E02 ({ruta})")
        node = node[seg]
    return node


def production_config(root=None) -> dict:
    """APROBACION PREPARATORIA (SPEC-M2 6.1): E01 + E02 APROBADAS y verificadas. No basta para salidas oficiales."""
    from ohlcv_dataroad.ingest import tda17_enmiendas_e02 as m2
    cfg = m2.load_effective_protocol_e02(root)
    m2.require_e02(cfg)
    return cfg


def official_config(publicacion: str, root=None) -> dict:
    """EJECUCION OFICIAL: unica configuracion para producir resultados del informe (vinculada a la publicacion P2)."""
    from ohlcv_dataroad.ingest import tda17_enmiendas_e02 as m2
    cfg = m2.official_config_e02(root, publicacion)
    m2.require_e02_oficial(cfg, root)
    return cfg


# =========================================================================== 1. paneles
def p_columns(cfg: dict) -> list[str]:
    return ind.p_columns(cfg)


def out_of_b(cfg: dict) -> list[str]:
    """Columnas de P retiradas de B: la ``fuera_de_B`` de cada identidad lineal de P (11)."""
    return [i["fuera_de_B"] for i in cfg["identidades"]["lineales"] if i["panel"] == "P"]


def b_columns(cfg: dict) -> list[str]:
    drop = set(out_of_b(cfg))
    return [c for c in p_columns(cfg) if c not in drop]


def l_columns(cfg: dict) -> list[str]:
    return [s["id"] for s in ind.catalog_series(cfg) if s["panel"] == "L"]


def reference_columns(cfg: dict) -> list[str]:
    return [r["id"] for r in cfg["referencias_tda04"]["en_panel_P"]]


def check_panel_counts(cfg: dict) -> dict:
    """Recuentos declarados (identidades.rango y recuentos): P 134, B 123, L 16; si no, el protocolo no es el
    implementado."""
    r = cfg["identidades"]["rango"]
    got = {"P": len(p_columns(cfg)), "B": len(b_columns(cfg)), "L": len(l_columns(cfg))}
    exp = {"P": r["P_columnas"], "B": r["B_columnas"], "L": r["L_columnas"]}
    if got != exp or len(set(out_of_b(cfg))) != exp["P"] - exp["B"]:
        raise ProtocolNotAuthorizedError(f"recuentos de paneles incompatibles: {got} frente a {exp}")
    return got


def build_panel_b(panel_p: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """B = P sin las columnas ``fuera_de_B`` (mismas filas, mismo orden de columnas que P)."""
    if list(panel_p.columns) != p_columns(cfg):
        raise ValueError("el panel P no tiene exactamente las 134 columnas oficiales en su orden")
    return panel_p[b_columns(cfg)]


def build_panel_l(series: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Panel L (diagnostico auxiliar): las 16 salidas con panel L, sin transformar."""
    return series[l_columns(cfg)]


def l_counterparts(cfg: dict) -> dict[str, str]:
    """Contraparte adimensional de cada serie L (panel_L.analisis): MACD convencional -> MACD_N (N18),
    Momentum -> ROC (B2 ~ B1), ATR -> NATR (N14, columna ln_natr de P), rango en ticks -> log_hl (N16)."""
    out = {}
    for s in ind.catalog_series(cfg):
        if s["panel"] != "L":
            continue
        e, c, o = s["elemento"], s["config"], s["salida"]
        if e == "A6":
            out[s["id"]] = f"A6_{c}__n_{o.removeprefix('conv_')}"
        elif e == "B2":
            out[s["id"]] = f"B1_{c}__roc"
        elif e == "C1":
            out[s["id"]] = f"C2_{c}__ln_natr"
        elif e == "E5":
            out[s["id"]] = "log_hl"
        else:
            raise ProtocolNotAuthorizedError(f"serie L sin contraparte declarada: {s['id']}")
    return out


# =========================================================================== 2. entrada, poblacion, calentamiento
def check_expected_inputs(df: pd.DataFrame, cfg: dict) -> dict:
    """entradas.esperado (TDA-03/TDA-04), verificado ANTES de calcular: filas, retornos validos, jornadas,
    contratos y rolls. Cualquier diferencia detiene la ejecucion."""
    got = {"filas": int(len(df)), "retornos_validos": int(df["r_1m_valid"].astype(bool).sum()),
           "jornadas": int(df["trading_date"].nunique()), "contratos": int(df["contract"].nunique()),
           "rolls": int(df["is_roll_boundary"].astype(bool).sum())}
    exp = {k: int(v) for k, v in cfg["entradas"]["esperado"].items()}
    if got != exp:
        diff = {k: (got[k], exp[k]) for k in exp if got.get(k) != exp[k]}
        raise ProtocolNotAuthorizedError(f"la entrada no es la esperada (obtenido, esperado): {diff}")
    return got


def complete_cases(panel: pd.DataFrame) -> np.ndarray:
    """Poblacion oficial (analisis.poblacion = CASOS_COMPLETOS_DE_P): todas las columnas finitas."""
    return np.isfinite(panel.to_numpy(np.float64)).all(axis=1)


def population_report(panel: pd.DataFrame) -> pd.DataFrame:
    """Perdidas de poblacion por columna: valores no finitos, filas en que la columna es la UNICA no finita, y
    una fila resumen ``__TOTAL__`` (filas, casos completos, filas perdidas)."""
    ok = np.isfinite(panel.to_numpy(np.float64))
    bad = ~ok
    only = bad & (bad.sum(axis=1) == 1)[:, None]
    rep = pd.DataFrame({"columna": panel.columns, "no_finitos": bad.sum(axis=0), "unica_no_finita": only.sum(axis=0)})
    n, full = len(panel), int(ok.all(axis=1).sum())
    total = pd.DataFrame({"columna": ["__TOTAL__"], "no_finitos": [n - full], "unica_no_finita": [np.nan]})
    rep = pd.concat([rep, total], ignore_index=True)
    rep.attrs.update(filas=n, casos_completos=full, perdidas=n - full)
    return rep


def _element_classes(cfg: dict) -> dict[str, str]:
    out = {}
    for name, cl in cfg["calentamiento"]["clases"].items():
        for e in cl.get("elementos", []):
            out[e] = name
    return out


def _first(mask: np.ndarray) -> float:
    idx = np.flatnonzero(mask)
    return float(idx[0] + 1) if idx.size else np.nan                      # indice base 1 dentro del contrato


def structural_eligibility(cls: str, el: dict, conf: dict, cons: np.ndarray) -> np.ndarray:
    """Elegibilidad ESTRUCTURAL por barra de un contrato (calentamiento.clases); solo depende del esqueleto
    (reinicio y pares CONS; ``cons[0]`` = False). No cubre JORNADAS (D7), que se calcula sobre el calendario."""
    n_b = cons.size
    t = np.arange(1, n_b + 1)
    u = np.cumsum(cons)                                                     # u_t: barras CONS en 1..t
    p = conf["params"]
    if cls == "EXACTA":
        return t >= conf["K"]
    if cls == "CONTADOR_CONS":
        if el["id"] == "B9":
            e_rsi = _first(u >= conf["U"])
            return t >= e_rsi + conf["barras_tras_E_RSI"] if np.isfinite(e_rsi) else np.zeros(n_b, bool)
        return u >= conf["U"]
    if cls in ("VENTANA_VALIDEZ", "DEPENDIENTE_DE_DATOS"):
        n = p["er"] if cls == "DEPENDIENTE_DE_DATOS" else p["n"]
        k = conf["K_min"] if cls == "DEPENDIENTE_DE_DATOS" else conf["K"]
        cs = np.r_[0, u]
        cnt = cs[t] - cs[np.maximum(t - n, 0)]                              # pares CONS que terminan en t-n+1..t
        return (t >= k) & (cnt >= pr.min_valid(n))
    raise ValueError(f"clase sin elegibilidad por contrato: {cls}")


def _activation_by_values(element: str, output: str) -> bool:
    """Activacion algoritmica dependiente de los valores: KAMA (A5) y la salida adx de A8 (E01-A). En ambas la
    salida es finita en la barra de activacion (KAMA := C -> 0; ADX = media de DX finitas), de modo que la
    primera activacion es la primera salida finita entre las barras elegibles."""
    return element == "A5" or (element == "A8" and output == "adx")


def warmup_table(df: pd.DataFrame, series: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Registro de calentamiento (E01, calentamiento.registro) por serie (P y L) y contrato: clase, K_teorico, U,
    primera_barra_elegible, retraso, barras no CONS durante el calentamiento, primera_activacion,
    primera_salida_finita y barras elegibles con NaN por regla de division (indices base 1 en el contrato).
    D7 (JORNADAS) se registra con K_jornadas; su 'retraso' en barras queda NaN (REGISTRO_D7_RETRASO)."""
    ind._validate(df)
    df = df.reset_index(drop=True)
    classes = _element_classes(cfg)
    segs = ind.contract_segments(df)
    contracts = df["contract"].to_numpy()
    cons_all = df["r_1m_valid"].to_numpy(bool)
    d7 = _rvol_eligibility(df, cfg) if any(e["id"] == "D7" for e in cfg["catalogo"]) else {}
    rows = []
    for s in ind.catalog_series(cfg):
        if s["panel"] == "Aux":
            continue
        el = next(e for e in cfg["catalogo"] if e["id"] == s["elemento"])
        conf = next(c for c in el["configuraciones"] if c["id"] == s["config"])
        cls = classes[el["id"]]
        vals = series[s["id"]].to_numpy(np.float64)
        for a, z in segs:
            cons = cons_all[a:z].copy()
            cons[0] = False
            if cls == "JORNADAS":
                elig = d7[conf["params"]["k_jornadas"]][a:z]
                k_teo, u_dec = conf["K_jornadas"], np.nan
            else:
                elig = structural_eligibility(cls, el, conf, cons)
                k_teo, u_dec = conf.get("K", conf.get("K_min")), conf.get("U", np.nan)
            v = vals[a:z]
            first_elig = _first(elig)
            if _activation_by_values(el["id"], s["salida"]):
                act = _first(elig & np.isfinite(v))
            else:
                act = first_elig
            t = np.arange(1, z - a + 1)
            active = elig & (t >= act) if np.isfinite(act) else np.zeros(z - a, bool)
            end = int(first_elig) if np.isfinite(first_elig) else z - a
            rows.append({
                "serie": s["id"], "panel": s["panel"], "contrato": contracts[a], "clase": cls,
                "K_teorico": k_teo, "U": u_dec, "barras_contrato": z - a,
                "primera_barra_elegible": first_elig,
                "retraso": (first_elig - k_teo) if cls != "JORNADAS" else np.nan,
                "barras_no_cons_calentamiento": int((~cons[1:end]).sum()),
                "primera_activacion": act,
                "primera_salida_finita": _first(active & np.isfinite(v)),
                "elegibles_nan_division": int((active & ~np.isfinite(v)).sum()),
            })
    return pd.DataFrame(rows)


def _rvol_eligibility(df: pd.DataFrame, cfg: dict) -> dict[int, np.ndarray]:
    """Elegibilidad estructural de D7 por barra (clase JORNADAS): J(t) completo (k jornadas del calendario
    anteriores) y >= ceil(0,8 k) barras observadas en el mismo minuto NY. Solo esqueleto, sin valores."""
    days = ind.calendar_trading_days(df)
    pos = {d: i for i, d in enumerate(days)}
    di = np.array([pos[d] for d in df["trading_date"]])
    minute = ind.minute_of_day_ny(df["timestamp"])
    obs = np.zeros((len(days), 1440), np.int64)
    obs[di, minute] = 1
    cs = np.vstack([np.zeros((1, 1440), np.int64), np.cumsum(obs, axis=0)])
    out = {}
    for el in cfg["catalogo"]:
        if el["id"] != "D7":
            continue
        for c in el["configuraciones"]:
            k = c["params"]["k_jornadas"]
            lo = np.clip(di - k, 0, None)
            cnt = cs[di, minute] - cs[lo, minute]
            out[k] = (di >= k) & (cnt >= pr.min_valid(k))
    return out


# =========================================================================== 3. celdas
def cell_definitions(timestamps: pd.Series, cfg: dict) -> dict[str, dict]:
    """Celdas de analisis.celdas: {nombre: {"mascara", "tipo", "estado"}}.

    Anio = anio NY del timestamp (criterio year_ny_del_timestamp); segmento = intervalo [inicio, fin) del minuto
    del dia NY con los cortes estables de TDA-06. Un anio presente que no figure en el protocolo detiene."""
    c = cfg["analisis"]["celdas"]
    t = pd.to_datetime(timestamps)
    t = t.dt.tz_localize("UTC") if t.dt.tz is None else t.dt.tz_convert("UTC")
    ny = t.dt.tz_convert(ind.NY_TZ)
    year = ny.dt.year.to_numpy()
    minute = (ny.dt.hour * 60 + ny.dt.minute).to_numpy()
    declared = set(c["anios"]["evaluables"]) | set(c["anios"]["no_evaluables"])
    extra = sorted(set(np.unique(year).tolist()) - declared)
    if extra:
        raise ProtocolNotAuthorizedError(f"anios no declarados en el protocolo: {extra}")
    cuts, labels = c["segmentos"]["cortes_min"], c["segmentos"]["etiquetas"]
    seg = np.searchsorted(np.asarray(cuts), minute, side="right")
    out = {"GLOBAL": {"mascara": np.ones(year.size, bool), "tipo": "global", "estado": "EVALUABLE"}}
    for y in sorted(declared):
        estado = "NO_EVALUABLE" if y in c["anios"]["no_evaluables"] else \
                 "EVALUABLE_PARCIAL" if y in c["anios"]["parciales"] else "EVALUABLE"
        out[f"ANIO_{y}"] = {"mascara": year == y, "tipo": "anio", "estado": estado, "anio": y}
    for k, lab in enumerate(labels):
        out[f"SEG_{lab}"] = {"mascara": seg == k, "tipo": "segmento", "estado": "EVALUABLE", "segmento": lab}
    for y in sorted(declared):
        for k, lab in enumerate(labels):
            out[f"ANIO_{y}__SEG_{lab}"] = {"mascara": (year == y) & (seg == k), "tipo": "anio_x_segmento",
                                           "estado": "SOLO_DESCRIPTIVO"}
    return out


# =========================================================================== 4. estadisticos de celda
@dataclass
class CellStats:
    """Estadisticos suficientes de una celda: n, medias y factor R (p x p) de la QR de la matriz (centrada si
    ``centered``). ``R.T @ R`` es la matriz de productos cruzados; ``corr`` la de correlacion."""
    n: int
    mean: np.ndarray
    R: np.ndarray
    centered: bool

    @property
    def gram(self) -> np.ndarray:
        return self.R.T @ self.R

    @property
    def corr(self) -> np.ndarray:
        g = self.gram
        d = np.sqrt(np.diag(g))
        return g / np.outer(d, d)


def cell_stats(X: np.ndarray, *, centered: bool = True, chunk: int = 200_000) -> CellStats:
    """QR por bloques (TSQR) de X (n x p) en float64. Exige valores finitos, n >= 2 y varianza > 0 en todas las
    columnas (una columna constante no tiene correlacion: se detiene, no se descarta en silencio)."""
    X = np.asarray(X, np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("celda con menos de 2 filas")
    if not np.isfinite(X).all():
        raise ValueError("la celda contiene valores no finitos (debe ser poblacion de casos completos)")
    n, p = X.shape
    mean = X.mean(axis=0) if centered else np.zeros(p)
    R = np.zeros((0, p))
    for a in range(0, n, chunk):
        R = np.linalg.qr(np.vstack([R, X[a:a + chunk] - mean]), mode="r")
    if R.shape[0] < p:
        R = np.vstack([R, np.zeros((p - R.shape[0], p))])
    st = CellStats(n=n, mean=mean, R=R, centered=centered)
    if centered and (np.diag(st.gram) <= 0).any():
        bad = np.flatnonzero(np.diag(st.gram) <= 0).tolist()
        raise ValueError(f"columnas sin varianza en la celda (indices {bad}): correlacion no definida")
    return st


def average_ranks(X: np.ndarray) -> np.ndarray:
    """Rangos convencionales con rango medio en empates, por columna, dentro de la celda (spearman.definicion)."""
    return rankdata(np.asarray(X, np.float64), method="average", axis=0)


def winsorize(X: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """S3: recorte de cada columna en sus cuantiles ``lower`` y ``upper`` de la celda (numpy, interpolacion lineal)."""
    X = np.asarray(X, np.float64)
    lo, hi = np.quantile(X, [lower, upper], axis=0)
    return np.clip(X, lo, hi)


def pearson(X: np.ndarray) -> np.ndarray:
    return cell_stats(X).corr


def spearman(X: np.ndarray) -> np.ndarray:
    return cell_stats(average_ranks(X)).corr


# =========================================================================== 5. R2 y exactitud
def pivoted_cholesky_basis(C: np.ndarray, tau: float) -> list[int]:
    """Base por Cholesky con pivoteo diagonal completo: elige en cada paso la columna de mayor varianza residual
    y para cuando esta es <= tau * max(diag C). Devuelve los indices elegidos (en el orden de eleccion)."""
    p = C.shape[0]
    d = np.diag(C).astype(np.float64).copy()
    L = np.zeros((p, p))
    chosen: list[int] = []
    free = np.ones(p, bool)
    thr = tau * np.max(np.diag(C))
    for k in range(p):
        cand = np.where(free, d, -np.inf)
        i = int(np.argmax(cand))
        if not free[i] or d[i] <= thr:
            break
        chosen.append(i)
        free[i] = False
        L[i, k] = math.sqrt(d[i])
        m = np.flatnonzero(free)
        L[m, k] = (C[m, i] - L[m, :k] @ L[i, :k]) / L[i, k]
        d[m] -= L[m, k] ** 2
    return chosen


def r2_basis(C: np.ndarray, j: int, predictors: list[int], tau: float) -> float:
    """Metodo oficial (BASE_ESPECIFICA_POR_OBJETIVO_CHOLESKY_PIVOTEO): base de los predictores de ``j`` por
    Cholesky con pivoteo y R2 = r' C_BB^-1 r sobre esa base."""
    sub = C[np.ix_(predictors, predictors)]
    basis = [predictors[k] for k in pivoted_cholesky_basis(sub, tau)]
    if not basis:
        return 0.0
    r = C[basis, j]
    return float(r @ np.linalg.solve(C[np.ix_(basis, basis)], r))


def r2_spectral(C: np.ndarray, j: int, predictors: list[int], tau: float) -> float:
    """Verificacion cruzada PSEUDOINVERSA_ESPECTRAL: autovalores <= tau * maximo se descartan."""
    lam, U = np.linalg.eigh(C[np.ix_(predictors, predictors)])
    keep = lam > tau * lam.max()
    coef = U[:, keep].T @ C[predictors, j]
    return float(np.sum(coef ** 2 / lam[keep]))


def r2_min_norm(st: CellStats, j: int, predictors: list[int]) -> float:
    """Verificacion cruzada MCO_MINIMA_NORMA_ESTADISTICOS_SUFICIENTES: minima norma (lstsq, rcond por defecto)
    sobre el factor R de la QR de las filas centradas; equivale a MCO sobre las filas (||X b - y|| = ||R b - r||)."""
    if not st.centered:
        raise ValueError("el R2 se define sobre la matriz centrada")
    y, Xo = st.R[:, j], st.R[:, predictors]
    beta, *_ = np.linalg.lstsq(Xo, y, rcond=None)
    res = y - Xo @ beta
    return float(1.0 - (res @ res) / (y @ y))


def r2_table(st: CellStats, names: list[str], predictor_sets: dict[str, list[int]] | None, tau: float,
             tol_cross: float) -> pd.DataFrame:
    """R2 de cada columna (oficial y dos verificaciones). ``predictor_sets`` = {nombre: indices}; None = todas las
    demas (R2_TODAS). R2_NO_PUBLICABLE_POR_COLINEALIDAD si alguna verificacion difiere mas de ``tol_cross``."""
    C = st.corr
    p = len(names)
    rows = []
    for j in range(p):
        preds = [k for k in (range(p) if predictor_sets is None else predictor_sets[names[j]]) if k != j]
        rb, re_, rm = r2_basis(C, j, preds, tau), r2_spectral(C, j, preds, tau), r2_min_norm(st, j, preds)
        dif = max(abs(rb - re_), abs(rb - rm))
        rows.append({"columna": names[j], "r2": rb, "r2_espectral": re_, "r2_minima_norma": rm, "dif_cruzada": dif,
                     "publicable": dif <= tol_cross,
                     "etiqueta_publicacion": None if dif <= tol_cross else "R2_NO_PUBLICABLE_POR_COLINEALIDAD"})
    return pd.DataFrame(rows)


def tda04_predictors(names: list[str], cfg: dict) -> dict[str, list[int]]:
    """Variante R2_TDA04: cada columna frente a las 6 referencias de TDA-04 (sin ella misma)."""
    refs = [names.index(r) for r in reference_columns(cfg)]
    return {nm: [k for k in refs if k != j] for j, nm in enumerate(names)}


def other_indicators_predictors(names: list[str], cfg: dict):
    pending("R2_OTROS_INDICADORES")


IDENTITY_CONDITIONS = {"H > L": lambda df: df["high"].to_numpy(float) > df["low"].to_numpy(float),
                       "r_1m_valid": lambda df: df["r_1m_valid"].to_numpy(bool)}


def identity_residuals(panel: pd.DataFrame, df: pd.DataFrame, cfg: dict, panel_id: str) -> dict[str, dict]:
    """Residuo directo RELATIVO de cada identidad lineal del panel: max_filas |sum c x - constante| / escala, con
    escala = max(|constante|, max_filas max_i |c_i x_i|), sobre las filas de ``panel`` que cumplen la condicion
    de la identidad (``df`` alineado por indice con ``panel``)."""
    out = {}
    for ident in cfg["identidades"]["lineales"]:
        if ident["panel"] != panel_id:
            continue
        mask = np.ones(len(panel), bool)
        if "condicion" in ident:
            if ident["condicion"] not in IDENTITY_CONDITIONS:
                raise ProtocolNotAuthorizedError(f"condicion de identidad no implementada: {ident['condicion']}")
            mask &= IDENTITY_CONDITIONS[ident["condicion"]](df.loc[panel.index])
        terms = np.column_stack([k * panel[col].to_numpy(np.float64) for k, col in ident["terminos"]])
        mask &= np.isfinite(terms).all(axis=1)
        t = terms[mask]
        if not len(t):
            out[ident["id"]] = {"residuo_relativo": np.nan, "filas": 0}
            continue
        res = np.abs(t.sum(axis=1) - ident["constante"])
        scale = max(abs(ident["constante"]), float(np.abs(t).max()))
        out[ident["id"]] = {"residuo_relativo": float(res.max() / scale), "filas": int(mask.sum())}
    return out


def normalized_matrix(st: CellStats) -> np.ndarray:
    """Matriz de A1-a (E02, convencion.matrices): correlacion de Pearson si la celda esta centrada, cosenos si no."""
    g = st.gram
    d = np.sqrt(np.diag(g))
    return g / np.outer(d, d)


def null_counts(st: CellStats, tau: float) -> dict[str, int]:
    """Recuento de nulos (E02, A1-a): autovalores de la matriz normalizada (correlacion o cosenos) <= tau * maximo,
    sin recortar los negativos por redondeo (cuentan como nulos). Los valores singulares de las filas (factor R)
    <= tau * maximo se publican como diagnostico NO bloqueante."""
    lam = np.linalg.eigvalsh(normalized_matrix(st))
    s = np.linalg.svd(st.R, compute_uv=False)
    return {"autovalores": int(np.sum(lam <= tau * lam.max())), "singulares": int(np.sum(s <= tau * s.max()))}


def check_null_convention(cfg: dict) -> dict:
    """La convencion de E02 es la implementada (si no, el motor no aplica otra en silencio)."""
    conv = e02_value(cfg, "identidades.verificacion_numerica.convencion")
    expected = {"cantidad": "AUTOVALORES", "negativos_por_redondeo": "CUENTAN_COMO_NULOS_SIN_RECORTE",
                "valores_singulares": "DIAGNOSTICO_PUBLICADO_NO_BLOQUEANTE"}
    if any(conv.get(k) != v for k, v in expected.items()) or \
            conv.get("matrices", {}).get("P_sin_centrar") != "COSENOS_SIN_CENTRAR":
        raise ProtocolNotAuthorizedError(f"convencion de nulos distinta de la implementada: {conv}")
    return conv


RANK_SCOPES = ("BLOQUEANTE", "INFORMATIVA")


def rank_scope(celda: str, cfg: dict) -> str:
    """Ambito del control de nulos de una celda (E02, A4-a): BLOQUEANTE en GLOBAL, anios evaluables y segmentos;
    INFORMATIVA en los anios no evaluables (2019) y en anio x segmento."""
    amb = e02_value(cfg, "identidades.verificacion_numerica.ambito")
    if amb.get("bloqueantes") != ["GLOBAL", "ANIOS_EVALUABLES", "SEGMENTOS"] or \
            amb.get("informativas") != ["ANIO_2019", "ANIO_X_SEGMENTO", "PANEL_L_POR_CONTRATO", "REPLICAS_G2"]:
        raise ProtocolNotAuthorizedError(f"ambito del control distinto del implementado: {amb}")
    c = cfg["analisis"]["celdas"]
    if celda == "GLOBAL":
        return "BLOQUEANTE"
    if "__SEG_" in celda and celda.startswith("ANIO_"):
        return "INFORMATIVA"
    if celda.startswith("SEG_") and celda[4:] in c["segmentos"]["etiquetas"]:
        return "BLOQUEANTE"
    if celda.startswith("ANIO_") and celda[5:].isdigit():
        y = int(celda[5:])
        if y in c["anios"]["evaluables"]:
            return "BLOQUEANTE"
        if y in c["anios"]["no_evaluables"]:
            return "INFORMATIVA"
    raise ProtocolNotAuthorizedError(f"celda no declarada: {celda}")


def rank_check(counts: dict[str, int], expected: int, what: str, ambito: str) -> dict:
    """Control de nulos (E02, A1-a + A4-a): solo los AUTOVALORES se comparan con el registro. En una celda
    BLOQUEANTE una diferencia es BLOQUEAR_HASTA_EXPLICAR; en una INFORMATIVA solo se informa (DIFIERE_INFORMATIVO).
    Los valores singulares se informan siempre, sin efecto."""
    if ambito not in RANK_SCOPES:
        raise ValueError(f"ambito desconocido: {ambito}")
    ok = counts["autovalores"] == expected
    estado = "CONFORME" if ok else "BLOQUEAR_HASTA_EXPLICAR" if ambito == "BLOQUEANTE" else "DIFIERE_INFORMATIVO"
    return {"matriz": what, **counts, "esperado": expected, "ambito": ambito, "estado": estado,
            "singulares_igual_registro": counts["singulares"] == expected}


def undefined_columns(X: np.ndarray, names: list[str]) -> dict[str, list[str]]:
    """Matriz NO_DEFINIDA (E02, F5-a), sin umbral: columnas con varianza cero (todos sus valores iguales) en la
    poblacion de la celda, que anulan la correlacion, y columnas con norma cero (todos 0), que anulan los cosenos.
    Con menos de 2 filas ninguna columna tiene varianza: la matriz no esta definida."""
    X = np.asarray(X, np.float64)
    if X.shape[0] < 2:
        return {"varianza_cero": list(names), "norma_cero": [n for j, n in enumerate(names) if (X[:, j] == 0).all()]}
    return {"varianza_cero": [names[j] for j in np.flatnonzero(np.ptp(X, axis=0) == 0)],
            "norma_cero": [names[j] for j in np.flatnonzero((X == 0).all(axis=0))]}


def exactness_labels(r2: dict[str, float], residuals: dict[str, dict], cfg: dict) -> pd.DataFrame:
    """Etiquetas de exactitud (analisis.exactitud, D23), por REGISTRO y RESIDUO, nunca solo por R2:
    - columna en una identidad registrada de P: EXACTA_ALGEBRAICA si todos sus residuos <= residuo_max; si no,
      ERROR_IDENTIDAD_REGISTRADA (bloquea la etapa);
    - sin registro: ALERTA_DEPENDENCIA_NO_REGISTRADA si 1 - R2 <= residuo_max; CASI_EXACTA_COLINEAL si
      <= casi_exacta_max; si no, sin etiqueta."""
    ex = cfg["analisis"]["exactitud"]
    member: dict[str, list[str]] = {}
    for ident in cfg["identidades"]["lineales"]:
        if ident["panel"] == "P":
            for _, col in ident["terminos"]:
                member.setdefault(col, []).append(ident["id"])
    rows = []
    for col, v in r2.items():
        ids = member.get(col, [])
        if ids:
            worst = max(residuals[i]["residuo_relativo"] for i in ids)
            lab = "EXACTA_ALGEBRAICA" if worst <= ex["residuo_max"] else "ERROR_IDENTIDAD_REGISTRADA"
        else:
            worst = np.nan
            gap = 1.0 - v
            lab = "ALERTA_DEPENDENCIA_NO_REGISTRADA" if gap <= ex["residuo_max"] else \
                  "CASI_EXACTA_COLINEAL" if gap <= ex["casi_exacta_max"] else None
        rows.append({"columna": col, "identidades": ",".join(ids), "residuo_max": worst, "r2": v, "etiqueta": lab})
    return pd.DataFrame(rows)


def blocking_findings(labels: pd.DataFrame, rank_checks: list[dict]) -> list[str]:
    """Hallazgos que bloquean: identidad registrada con residuo > 1e-9 (bloquea la etapa) y recuentos de nulos por
    autovalores distintos del registro en una celda BLOQUEANTE (BLOQUEAR_HASTA_EXPLICAR, E02 A1-a + A4-a)."""
    out = [f"ERROR_IDENTIDAD_REGISTRADA: {c}" for c in labels.loc[labels["etiqueta"] == "ERROR_IDENTIDAD_REGISTRADA", "columna"]]
    out += [f"BLOQUEAR_HASTA_EXPLICAR: nulos de {r['matriz']} = {r['autovalores']} (autovalores; singulares "
            f"{r['singulares']}), registrados {r['esperado']}"
            for r in rank_checks if r["estado"] == "BLOQUEAR_HASTA_EXPLICAR"]
    return out


def null_directions(st: CellStats, names: list[str], tau: float) -> pd.DataFrame:
    """Direcciones nulas (E02, A2-a): autovectores de la matriz de A1-a con lambda <= tau * lambda_max y TODAS sus
    cargas por columna, sin umbral de seleccion."""
    lam, U = np.linalg.eigh(normalized_matrix(st))
    keep = np.flatnonzero(lam <= tau * lam.max())
    out = pd.DataFrame(U[:, keep].T, columns=names)
    out.insert(0, "autovalor", lam[keep])
    return out


# Tipos de relacion N (E02, decision 16) y lo que el diagnostico publica de cada uno.
N_TYPES = {"LA-P": "CARGAS_EN_DIRECCIONES_NULAS", "LA-F": "SIN_DIRECCION_EN_P", "NL-P": "CARGAS_EN_DIRECCIONES_NULAS",
           "NL-F": "SIN_DIRECCION_EN_P", "DES": "RECUENTO_DE_VIOLACIONES", "DIAG": "NO_IDENTIDAD"}


def relations_n(cfg: dict) -> list[dict]:
    """Anexo de E02 (``relaciones_N``): tipo y columnas de P de cada relacion N1-N21. Se exige que cubra exactamente
    las relaciones congeladas, en su orden, con columnas de P existentes, sin nulos esperados nuevos; N4 (LA-P) debe
    coincidir con una identidad lineal registrada y N21 (DES) declarar sus dos columnas. No evalua ninguna formula."""
    rn = e02_value(cfg, "identidades.verificacion_numerica.relaciones_N")
    rels = rn.get("relaciones") if isinstance(rn.get("relaciones"), list) else []
    cols = set(p_columns(cfg))
    linear = {i["id"] for i in cfg["identidades"]["lineales"]}
    frozen = [r["id"] for r in cfg["identidades"]["no_lineales_o_fuera_de_P"]]
    ok = (rn.get("nulos_esperados_adicionales") == 0 and rn.get("valor_numerico") == "EXPEDIENTE_DE_LA_ENMIENDA_DE_EXPLICACION"
          and {t: v.get("diagnostico") for t, v in rn.get("tipos", {}).items()} == N_TYPES
          and [r.get("id") for r in rels] == frozen)
    for r in rels if ok else []:
        ok = ok and r.get("tipo") in N_TYPES and bool(r.get("columnas_P")) and set(r["columnas_P"]) <= cols
        if ok and r["tipo"] == "LA-P":
            ok = r.get("coincide_con") in linear
        if ok and r["tipo"] == "DES":
            d = r.get("desigualdad") or {}
            ok = sorted([d.get("valor_absoluto"), d.get("menor_o_igual_que")]) == sorted(r["columnas_P"])
    if not ok:
        raise ProtocolNotAuthorizedError(f"anexo de relaciones N distinto del implementado: {rn}")
    return rels


def relation_diagnostics(rels: list[dict], directions: dict[str, pd.DataFrame], panel: pd.DataFrame) -> list[dict]:
    """Relaciones N en el diagnostico de bloqueo (E02, decision 16): LA-P y NL-P, cargas de sus columnas de P en cada
    direccion nula observada; LA-F y NL-F, SIN_DIRECCION_EN_P; DES, recuento de filas de la poblacion de la celda que
    violan la desigualdad (comparacion exacta); DIAG, NO_IDENTIDAD. Ninguna relacion anade nulos esperados y su valor
    numerico queda para el expediente de la enmienda de explicacion."""
    out = []
    for r in rels:
        rec = {"id": r["id"], "tipo": r["tipo"], "columnas_P": list(r["columnas_P"]), "diagnostico": N_TYPES[r["tipo"]],
               "valor_numerico": None,
               "estado": "PENDIENTE_EXPEDIENTE" if r["tipo"] in ("LA-P", "LA-F", "NL-P", "NL-F") else None}
        if r["tipo"] in ("LA-P", "NL-P"):
            rec["cargas"] = {what: d[["autovalor"] + [c for c in r["columnas_P"] if c in d.columns]]
                             for what, d in directions.items()}
        elif r["tipo"] == "DES":
            d = r["desigualdad"]
            a = np.abs(panel[d["valor_absoluto"]].to_numpy(np.float64))
            b = panel[d["menor_o_igual_que"]].to_numpy(np.float64)
            rec.update(violaciones=int(np.sum(a > b)), filas=int(a.size))
        out.append(rec)
    return out


def blocking_dossier(mats: dict[str, tuple[CellStats, list[str]]], checks: list[dict], r2_tables: dict[str, pd.DataFrame],
                     labels: pd.DataFrame, residuals: dict[str, dict], cfg: dict, celda: str, alcance: str,
                     panel: pd.DataFrame) -> dict:
    """Diagnosticos obligatorios de BLOQUEADO_HASTA_EXPLICAR (E02, procedimiento_bloqueo): emitirlos NO levanta el
    bloqueo. (1) espectro completo de cada matriz afectada por las dos convenciones; (2) direcciones nulas con sus
    cargas; (3) R2 por los tres metodos y etiquetas, marcados como diagnostico bloqueado; (4) residuos de las
    identidades registradas y relaciones N1-N21 segun su tipo (``relation_diagnostics``; ``panel`` = poblacion de la
    celda, para el recuento de la desigualdad); (5) celda, matrices y alcance afectados. ``expediente`` lista los
    campos minimos del expediente de la enmienda posterior."""
    pb = e02_value(cfg, "identidades.verificacion_numerica.procedimiento_bloqueo")
    rels = relations_n(cfg)
    tau = cfg["identidades"]["verificacion_numerica"]["tau_relativa"]
    affected = [r["matriz"] for r in checks if r["estado"] == "BLOQUEAR_HASTA_EXPLICAR"]
    spectra, directions = {}, {}
    for what in affected:
        st, names = mats[what]
        spectra[what] = {"autovalores": np.sort(np.linalg.eigvalsh(normalized_matrix(st)))[::-1],
                         "singulares": np.linalg.svd(st.R, compute_uv=False)}
        directions[what] = null_directions(st, names, tau)
    r2_marked = {k: t.assign(marca="DIAGNOSTICO_BLOQUEADO") for k, t in r2_tables.items()}
    relations = relation_diagnostics(rels, directions, panel)
    return {"estado": pb["estado"], "levantamiento": pb["levantamiento"], "espectros": spectra,
            "direcciones_nulas": directions, "r2_diagnostico": r2_marked,
            "exactitud_diagnostico": labels.assign(marca="DIAGNOSTICO_BLOQUEADO"),
            "residuos_identidades": residuals, "relaciones_N": relations,
            "ambito_afectado": {"celda": celda, "matrices": affected, "alcance": alcance,
                                "recuentos": {r["matriz"]: (r["autovalores"], r["esperado"]) for r in checks}},
            "expediente": list(pb["expediente"])}


def spectral_gap_status(*_args, **_kwargs):
    pending("HUECO_ESPECTRAL")


def publish_exact_by_cell(*_args, **_kwargs):
    """Publicacion de la etiqueta EXACTA en una celda: exige la comprobacion del hueco espectral (decision 11 de
    E02, APLAZADA). Bloqueada."""
    pending("HUECO_ESPECTRAL")


# =========================================================================== 6. dimension efectiva y agrupamiento
def dimension_metrics(eigenvalues: np.ndarray) -> dict:
    """Metricas de analisis.pca.metricas sobre los autovalores de una matriz de correlacion (los negativos por
    redondeo se recortan a 0): k80/k90/k95 = menor k con varianza acumulada >= 80/90/95 %; ratio de
    participacion (sum l)^2 / sum l^2; rango efectivo = exp(entropia de l / sum l); M_eff de Li y Ji (2005)
    = sum_i [1(l_i >= 1) + (l_i - floor(l_i))]."""
    lam = np.sort(np.clip(np.asarray(eigenvalues, np.float64), 0.0, None))[::-1]
    total = lam.sum()
    cum = np.cumsum(lam) / total
    k = {f"k{int(q * 100)}": int(np.searchsorted(cum, q - 1e-12) + 1) for q in (0.80, 0.90, 0.95)}
    w = lam[lam > 0] / total
    return {**k, "ratio_participacion": float(total ** 2 / np.sum(lam ** 2)),
            "rango_efectivo_entropia": float(np.exp(-np.sum(w * np.log(w)))),
            "M_eff_Li_Ji": float(np.sum((lam >= 1).astype(float) + (lam - np.floor(lam)))),
            "p": int(lam.size)}


def correlation_eigenvalues(C: np.ndarray) -> np.ndarray:
    return np.sort(np.linalg.eigvalsh(C))[::-1]


def hierarchical_clusters(rho_spearman: np.ndarray, names: list[str], cuts_abs_rho: list[float]) -> pd.DataFrame:
    """analisis.agrupamiento: enlace medio sobre d = 1 - |rho_spearman|; corte en |rho| = c <=> d = 1 - c
    (fcluster por distancia cofenetica <= 1 - c). Sin representantes."""
    D = 1.0 - np.abs(np.asarray(rho_spearman, np.float64))
    D = np.clip((D + D.T) / 2.0, 0.0, None)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    out = pd.DataFrame({"columna": names})
    for c in cuts_abs_rho:
        out[f"grupo_abs_rho_{c:.2f}"] = fcluster(Z, t=1.0 - c, criterion="distance")
    out.attrs["linkage"] = Z
    return out


# =========================================================================== 7. bootstrap y clasificacion
def circular_block_days(n_days: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Una replica de bloques moviles circulares: ceil(D / b) inicios uniformes en 0..D-1, bloques de b jornadas
    consecutivas (modulo D), concatenados y truncados a D jornadas."""
    if n_days < 1 or block < 1:
        raise ValueError("bootstrap sin jornadas o con bloque < 1")
    starts = rng.integers(0, n_days, size=math.ceil(n_days / block))
    return ((starts[:, None] + np.arange(block)[None, :]) % n_days).ravel()[:n_days]


def bootstrap_day_draws(n_days: int, block: int, replicas: int, rng: np.random.Generator) -> np.ndarray:
    """(replicas, D) jornadas remuestreadas consumiendo ``rng`` en orden: una llamada
    ``integers(0, D, size=ceil(D / block))`` por replica (E02, B6)."""
    return np.vstack([circular_block_days(n_days, block, rng) for _ in range(replicas)])


def bootstrap_stream_cells(cfg: dict) -> list[str]:
    """Orden declarado de las celdas con flujo propio (E02, B6): GLOBAL, anios ascendentes (todos los declarados)
    y segmentos en el orden de ``etiquetas``. Anio x segmento (solo descriptivo) no tiene flujo."""
    rep = e02_value(cfg, "analisis.bootstrap.reproducibilidad")
    if rep.get("bitgen") != "PCG64" or e02_value(cfg, "analisis.bootstrap.ambito").get("por_celda") is not True:
        raise ProtocolNotAuthorizedError(f"reproducibilidad o ambito del bootstrap distintos de los implementados: {rep}")
    c = cfg["analisis"]["celdas"]
    years = sorted(set(c["anios"]["evaluables"]) | set(c["anios"]["no_evaluables"]))
    return ["GLOBAL"] + [f"ANIO_{y}" for y in years] + [f"SEG_{lab}" for lab in c["segmentos"]["etiquetas"]]


def cell_generator(cfg: dict, celda: str) -> np.random.Generator:
    """Generador PCG64 de la celda: hijo ``i`` de ``SeedSequence(analisis.bootstrap.semilla).spawn(n_celdas)``,
    con ``i`` la posicion de la celda en el orden declarado. Cada llamada devuelve el flujo REINICIADO (S4 usa los
    mismos flujos)."""
    cells = bootstrap_stream_cells(cfg)
    if celda not in cells:
        raise ProtocolNotAuthorizedError(f"celda sin flujo bootstrap declarado: {celda}")
    seq = np.random.SeedSequence(cfg["analisis"]["bootstrap"]["semilla"]).spawn(len(cells))[cells.index(celda)]
    return np.random.Generator(np.random.PCG64(seq))


def cell_draws(cfg: dict, celda: str, n_days: int, *, bloque: int, replicas: int | None = None) -> np.ndarray:
    """Replicas de la celda (B2-a + B6). ``bloque`` es la longitud oficial o una de S4 (``sensibilidad_bloques``),
    con el mismo flujo reiniciado; ``replicas`` por defecto = ``analisis.bootstrap.replicas`` (un prefijo del
    flujo si es menor, como en la validacion de Spearman)."""
    b = cfg["analisis"]["bootstrap"]
    if bloque not in [b["longitud_bloque_jornadas"], *b["sensibilidad_bloques"]]:
        raise ProtocolNotAuthorizedError(f"longitud de bloque no declarada: {bloque}")
    r = b["replicas"] if replicas is None else replicas
    if not 0 < r <= b["replicas"]:
        raise ValueError(f"numero de replicas fuera de 1..{b['replicas']}: {r}")
    return bootstrap_day_draws(n_days, bloque, r, cell_generator(cfg, celda))


@dataclass
class DailyMoments:
    """Estadisticos suficientes por jornada de una celda, CENTRADOS en la media de la celda (reduce la
    cancelacion): filas, sumas y productos cruzados de cada jornada, y minimo y maximo por columna (detectan sin
    umbral una columna constante en una replica: VAR_CERO). ``codes``: jornadas en orden cronologico."""
    n: np.ndarray
    s: np.ndarray
    g: np.ndarray
    dmin: np.ndarray
    dmax: np.ndarray
    codes: np.ndarray


def daily_moments(X: np.ndarray, day_codes: np.ndarray) -> DailyMoments:
    X = np.asarray(X, np.float64)
    codes, inv = np.unique(day_codes, return_inverse=True)
    Xc = X - X.mean(axis=0)
    D, p = codes.size, X.shape[1]
    n = np.bincount(inv, minlength=D).astype(np.float64)
    s = np.zeros((D, p))
    np.add.at(s, inv, Xc)
    g = np.zeros((D, p, p))
    dmin, dmax = np.zeros((D, p)), np.zeros((D, p))
    order = np.argsort(inv, kind="stable")
    bounds = np.r_[0, np.cumsum(n.astype(np.int64))]
    for d in range(D):
        idx = order[bounds[d]:bounds[d + 1]]
        rows = Xc[idx]
        g[d] = rows.T @ rows
        dmin[d], dmax[d] = X[idx].min(axis=0), X[idx].max(axis=0)
    return DailyMoments(n=n, s=s, g=g, dmin=dmin, dmax=dmax, codes=codes)


def replica_zero_variance(m: DailyMoments, days: np.ndarray) -> np.ndarray:
    """Columnas con varianza cero en la replica (todos los valores de las jornadas elegidas iguales): exacto."""
    sel = np.unique(days)
    return m.dmax[sel].max(axis=0) == m.dmin[sel].min(axis=0)


def replica_correlation(m: DailyMoments, days: np.ndarray) -> np.ndarray:
    """Correlacion de la replica formada por las jornadas ``days`` (con repeticion), exacta a partir de los
    estadisticos diarios (igual que recalcular sobre las filas remuestreadas)."""
    w = np.bincount(days, minlength=m.n.size).astype(np.float64)
    N = w @ m.n
    S = w @ m.s
    G = np.tensordot(w, m.g, axes=1) - np.outer(S, S) / N
    d = np.sqrt(np.diag(G))
    return G / np.outer(d, d)


def percentile_interval(samples: np.ndarray, quantiles: tuple[float, float]) -> tuple[float, float]:
    """Intervalo percentil con cuantiles EXPLICITOS, empiricos de tipo 7 (interpolacion lineal, ``method='linear'``
    de numpy)."""
    lo, hi = np.quantile(np.asarray(samples, np.float64), list(quantiles), method="linear")
    return float(lo), float(hi)


def interval_quantiles(cfg: dict) -> tuple[float, float]:
    """Cuantiles del IC (E02, B1-a): percentil, tipo 7, para R2_TODAS y SPEARMAN."""
    iv = e02_value(cfg, "analisis.bootstrap.intervalo")
    if iv.get("tipo") != "PERCENTIL" or iv.get("metodo_cuantil") != "EMPIRICO_INTERPOLACION_LINEAL_TIPO_7" or \
            iv.get("aplica_a") != ["R2_TODAS", "SPEARMAN"]:
        raise ProtocolNotAuthorizedError(f"intervalo distinto del implementado: {iv}")
    lo, hi = (float(q) for q in iv["cuantiles"])
    if not 0.0 < lo < hi < 1.0:
        raise ProtocolNotAuthorizedError(f"cuantiles del IC no validos: {iv['cuantiles']}")
    return lo, hi


def registered_columns(cfg: dict) -> set[str]:
    """Columnas de P que intervienen en una identidad lineal registrada (subnivel EXACTA por registro)."""
    return {col for i in cfg["identidades"]["lineales"] if i["panel"] == "P" for _, col in i["terminos"]}


INVALID_CAUSES = ("VAR_CERO", "BASE_DEGENERADA", "NO_FINITO", "FUERA_DE_RANGO")


def r2_official_checked(C: np.ndarray, j: int, predictors: list[int], tau: float) -> tuple[float, str | None, int]:
    """R2 por el metodo oficial (base de Cholesky con pivoteo) sobre la correlacion ``C`` de una replica, con la
    causa de invalidez de E02 (F2): BASE_DEGENERADA (base vacia o sistema sin solucion finita), NO_FINITO (matriz o
    R2 no finitos) o FUERA_DE_RANGO (R2 fuera de [0, 1], sin margen). Devuelve (R2, causa o None, tamano de base)."""
    if not predictors:
        return np.nan, "BASE_DEGENERADA", 0
    idx = [j, *predictors]
    if not np.isfinite(C[np.ix_(idx, idx)]).all():
        return np.nan, "NO_FINITO", 0
    sub = C[np.ix_(predictors, predictors)]
    basis = [predictors[k] for k in pivoted_cholesky_basis(sub, tau)]
    if not basis:
        return np.nan, "BASE_DEGENERADA", 0
    r = C[basis, j]
    try:
        coef = np.linalg.solve(C[np.ix_(basis, basis)], r)
    except np.linalg.LinAlgError:
        return np.nan, "BASE_DEGENERADA", len(basis)
    if not np.isfinite(coef).all():
        return np.nan, "BASE_DEGENERADA", len(basis)
    v = float(r @ coef)
    if not math.isfinite(v):
        return v, "NO_FINITO", len(basis)
    if v < 0.0 or v > 1.0:
        return v, "FUERA_DE_RANGO", len(basis)
    return v, None, len(basis)


def bootstrap_r2(m: DailyMoments, draws: np.ndarray, names: list[str], point: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """IC bootstrap de R2_TODAS de una celda (E02, B3-a + F2-a + B1-a). En cada replica, R2 por el metodo oficial
    sobre la correlacion de la replica; se usan SIEMPRE todas las replicas, sin descartar, regenerar, sustituir ni
    recortar ninguna. Si alguna es INVALIDA para la columna (VAR_CERO de cualquier columna que interviene,
    BASE_DEGENERADA, NO_FINITO, FUERA_DE_RANGO), su IC es NO_CALCULABLE, con el recuento por causa y los indices.
    Columnas registradas: IC degenerado [1, 1] por registro. Si la estimacion puntual (``point``, de ``r2_table``)
    no es publicable, el IC no se publica (R2_NO_PUBLICABLE). ``attrs['tamano_base']``: (replicas, p)."""
    er = e02_value(cfg, "analisis.bootstrap.estadistico_replica")
    if er.get("r2") != "METODO_OFICIAL_SOBRE_LA_CORRELACION_DE_LA_REPLICA" or \
            er.get("replica_invalida", {}).get("tratamiento") != "IC_NO_CALCULABLE_SI_ALGUNA_INVALIDA" or \
            list(er.get("replica_invalida", {}).get("causas", [])) != list(INVALID_CAUSES):
        raise ProtocolNotAuthorizedError(f"estadistico de replica distinto del implementado: {er}")
    tau = cfg["analisis"]["r2"]["tau_relativa"]
    q = interval_quantiles(cfg)
    reg = registered_columns(cfg)
    p, R = len(names), draws.shape[0]
    vals = np.full((R, p), np.nan)
    cause = np.full((R, p), None, dtype=object)
    base = np.zeros((R, p), np.int64)
    todo = [j for j, nm in enumerate(names) if nm not in reg]
    for r, days in enumerate(draws):
        zero = replica_zero_variance(m, days)
        if zero.any():
            cause[r, todo] = "VAR_CERO"                                     # en R2_TODAS intervienen todas las columnas
            continue
        C = replica_correlation(m, days)
        for j in todo:
            vals[r, j], cause[r, j], base[r, j] = r2_official_checked(C, j, [k for k in range(p) if k != j], tau)
    pt = point.set_index("columna")
    rows = []
    for j, nm in enumerate(names):
        rec = {"columna": nm, "r2": float(pt.loc[nm, "r2"]), "replicas": R, "ic_inf": np.nan, "ic_sup": np.nan,
               "replicas_invalidas": 0, "invalidas_por_causa": {}, "indices_invalidas": []}
        if nm in reg:
            rec.update(ic_estado="DEGENERADO_REGISTRO", ic_inf=1.0, ic_sup=1.0)
        else:
            bad = np.flatnonzero(cause[:, j] != None)                          # noqa: E711 (array de objetos)
            rec.update(replicas_invalidas=int(bad.size), indices_invalidas=bad.tolist(),
                       invalidas_por_causa={c: int(np.sum(cause[:, j] == c)) for c in INVALID_CAUSES if np.any(cause[:, j] == c)})
            if not bool(pt.loc[nm, "publicable"]):
                rec["ic_estado"] = "R2_NO_PUBLICABLE"
            elif bad.size:
                rec["ic_estado"] = "IC_NO_CALCULABLE"
            else:
                rec["ic_estado"] = "CALCULADO"
                rec["ic_inf"], rec["ic_sup"] = percentile_interval(vals[:, j], q)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.attrs.update(tamano_base=base, cuantiles=q)
    return out


def bootstrap_spearman_fixed(m_rank: DailyMoments, draws: np.ndarray, names: list[str], rho: np.ndarray,
                             cfg: dict) -> pd.DataFrame:
    """IC de Spearman con RANGOS FIJOS de la celda (spearman.ic_bootstrap): correlacion de replica de los rangos
    calculados una vez en la celda. Una replica con los rangos de i o j constantes (VAR_CERO) o con la correlacion
    no finita (NO_FINITO) es INVALIDA para el par e IC NO_CALCULABLE (E02, F2-a). Pares i < j."""
    q = interval_quantiles(cfg)
    p, R = len(names), draws.shape[0]
    samples = np.full((R, p, p), np.nan)
    zero = np.zeros((R, p), bool)
    for r, days in enumerate(draws):
        zero[r] = replica_zero_variance(m_rank, days)
        if not zero[r].all():
            with np.errstate(invalid="ignore", divide="ignore"):
                samples[r] = replica_correlation(m_rank, days)
    rows = []
    for i in range(p):
        for j in range(i + 1, p):
            vz = zero[:, i] | zero[:, j]
            nf = ~vz & ~np.isfinite(samples[:, i, j])
            bad = np.flatnonzero(vz | nf)
            rec = {"columna_i": names[i], "columna_j": names[j], "rho": float(rho[i, j]), "replicas": R,
                   "ic_inf": np.nan, "ic_sup": np.nan, "replicas_invalidas": int(bad.size),
                   "invalidas_por_causa": {k: int(v.sum()) for k, v in (("VAR_CERO", vz), ("NO_FINITO", nf)) if v.any()},
                   "indices_invalidas": bad.tolist()}
            if bad.size:
                rec["ic_estado"] = "IC_NO_CALCULABLE"
            else:
                rec["ic_estado"] = "CALCULADO"
                rec["ic_inf"], rec["ic_sup"] = percentile_interval(samples[:, i, j], q)
            rows.append(rec)
    return pd.DataFrame(rows)


def day_rows(day_codes: np.ndarray) -> list[np.ndarray]:
    """Filas de cada jornada de la celda, en el orden cronologico de ``daily_moments`` (``np.unique``)."""
    _, inv = np.unique(day_codes, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(inv))]
    return [order[bounds[d]:bounds[d + 1]] for d in range(bounds.size - 1)]


def exact_rank_replica_correlation(X: np.ndarray, rows_by_day: list[np.ndarray], days: np.ndarray) -> np.ndarray:
    """Spearman con RECALCULO EXACTO en la replica: filas de las jornadas elegidas (con repeticion), rangos medios
    recalculados sobre esas filas y Pearson de los rangos. Una columna de rangos constante da NaN."""
    rows = np.concatenate([rows_by_day[d] for d in days])
    Rk = average_ranks(np.asarray(X, np.float64)[rows])
    Rc = Rk - Rk.mean(axis=0)
    G = Rc.T @ Rc
    d = np.sqrt(np.diag(G))
    with np.errstate(invalid="ignore", divide="ignore"):
        return G / np.outer(d, d)


def spearman_invalid_rule(cfg: dict) -> dict:
    """Replica INVALIDA en la validacion de Spearman (E02, decision 14, S-a): en cualquiera de los dos metodos, la
    validacion de ESA CELDA falla en esa ronda; despues se aplica F3-I-b. Sin imputacion ni exclusion del par."""
    ri = e02_value(cfg, "analisis.spearman.validacion_replica_invalida")
    if ri.get("regla") != "FALLA_LA_VALIDACION_DE_LA_CELDA_EN_LA_RONDA" or ri.get("imputacion") != "NINGUNA" or \
            ri.get("metodos") != ["RANGOS_FIJOS", "RECALCULO_EXACTO"] or ri.get("causas") != ["VAR_CERO", "NO_FINITO"] or \
            ri.get("exclusion_del_par") != "PROHIBIDA":
        raise ProtocolNotAuthorizedError(f"regla de replica invalida en la validacion distinta de la implementada: {ri}")
    return ri


def _validation_cell(X: np.ndarray, day_codes: np.ndarray, cfg: dict, celda: str, k: int) -> dict:
    """Mediana, sobre todos los pares i < j y los dos extremos, de |IC_rangos_fijos - IC_exacto| con las primeras k
    replicas del flujo de la celda (las mismas para los dos metodos) y los cuantiles de B1-a. Si alguna replica es
    INVALIDA en cualquiera de los dos metodos (rangos constantes o correlacion no finita), la celda no tiene mediana
    y su validacion falla en esta ronda (S-a): se registran los indices de esas replicas."""
    q = interval_quantiles(cfg)
    X = np.asarray(X, np.float64)
    ranks = average_ranks(X)
    m_rank = daily_moments(ranks, day_codes)
    rbd = day_rows(day_codes)
    draws = cell_draws(cfg, celda, m_rank.n.size, bloque=cfg["analisis"]["bootstrap"]["longitud_bloque_jornadas"], replicas=k)
    p = X.shape[1]
    iu = np.triu_indices(p, 1)
    fixed, exact = np.empty((k, iu[0].size)), np.empty((k, iu[0].size))
    invalid = []
    for r, days in enumerate(draws):
        zf = replica_zero_variance(m_rank, days)
        with np.errstate(invalid="ignore", divide="ignore"):
            cf = np.full((p, p), np.nan) if zf.all() else replica_correlation(m_rank, days)
        ce = exact_rank_replica_correlation(X, rbd, days)
        fixed[r], exact[r] = cf[iu], ce[iu]
        if zf.any() or not np.isfinite(fixed[r]).all() or not np.isfinite(exact[r]).all():
            invalid.append(r)
    out = {"celda": celda, "replicas": k, "pares": int(iu[0].size), "replicas_invalidas": invalid}
    if invalid:
        return {**out, "mediana_dif_ic": None, "motivo": "REPLICA_INVALIDA"}
    lo_f, hi_f = np.quantile(fixed, list(q), axis=0, method="linear")
    lo_e, hi_e = np.quantile(exact, list(q), axis=0, method="linear")
    diffs = np.r_[np.abs(lo_f - lo_e), np.abs(hi_f - hi_e)]
    return {**out, "mediana_dif_ic": float(np.median(diffs))}


def spearman_validation(cells: dict[str, tuple[np.ndarray, np.ndarray]], cfg: dict) -> dict:
    """Validacion del IC de Spearman con rangos fijos (E02, F3-I + F3-I-b). ``cells`` = {celda: (X de la poblacion
    de la celda, trading_date de cada fila)} para exactamente las celdas de ``validacion_detalle``. Primero con
    ``replicas_exactas`` (30); si falla, VALIDACION AMPLIADA con ``alternativa_si_falla.replicas_exactas`` (100),
    mismo estadistico y tolerancia. Pasa si la mediana <= tolerancia en ambas celdas; una celda con una replica
    INVALIDA falla en esa ronda (S-a). Si ambas rondas fallan: IC_SPEARMAN_NO_PUBLICADOS (los coeficientes si se
    publican)."""
    vd = e02_value(cfg, "analisis.spearman.validacion_detalle")
    spearman_invalid_rule(cfg)
    val = cfg["analisis"]["spearman"]["validacion"]
    if vd.get("si_falla_la_ampliada") != "IC_SPEARMAN_NO_PUBLICADOS_COEFICIENTES_SI" or \
            cfg["analisis"]["spearman"]["ic_bootstrap"] != "RANGOS_DE_LA_CELDA_FIJOS":
        raise ProtocolNotAuthorizedError(f"validacion de Spearman distinta de la implementada: {vd}")
    if sorted(cells) != sorted(vd["celdas"]):
        raise ProtocolNotAuthorizedError(f"la validacion exige exactamente las celdas {vd['celdas']}")
    tol = val["tolerancia_mediana_dif_ic"]
    rounds = []
    for k in (val["replicas_exactas"], val["alternativa_si_falla"]["replicas_exactas"]):
        res = [_validation_cell(X, codes, cfg, c, k) for c, (X, codes) in cells.items()]
        ok = all(not r["replicas_invalidas"] and r["mediana_dif_ic"] <= tol for r in res)
        rounds.append({"replicas": k, "celdas": res, "pasa": ok})
        if ok:
            return {"resultado": "VALIDADA", "rondas": rounds, "publicar_ic_spearman": True}
    return {"resultado": "NO_VALIDADA", "rondas": rounds, "publicar_ic_spearman": False,
            "publicacion": vd["si_falla_la_ampliada"]}


def spearman_publication(table: pd.DataFrame, validation: dict) -> pd.DataFrame:
    """Tabla de Spearman para publicar: coeficientes siempre; IC solo si la validacion paso (F3-I-b)."""
    if validation.get("resultado") == "VALIDADA" and validation.get("publicar_ic_spearman") is True:
        return table.copy()
    out = table.copy()
    out[["ic_inf", "ic_sup"]] = np.nan
    out["ic_estado"] = "IC_NO_PUBLICADO_VALIDACION_FALLIDA"
    return out


def cell_bootstrap(panel_p: pd.DataFrame, day_codes: np.ndarray, analysis: dict, cfg: dict, celda: str) -> dict:
    """IC de una celda con matriz definida: replicas del flujo de la celda (las mismas para R2_TODAS y Spearman),
    estadisticos suficientes diarios de los valores y de los rangos fijos de la celda."""
    if analysis.get("estado_matriz") != "DEFINIDA":
        raise ValueError("la celda no tiene matriz definida: sin bootstrap (F5-a)")
    X = panel_p.to_numpy(np.float64)
    names = list(panel_p.columns)
    m = daily_moments(X, day_codes)
    m_rank = daily_moments(average_ranks(X), day_codes)
    draws = cell_draws(cfg, celda, m.n.size, bloque=cfg["analisis"]["bootstrap"]["longitud_bloque_jornadas"])
    return {"jornadas": int(m.n.size), "r2_todas": bootstrap_r2(m, draws, names, analysis["r2_todas"], cfg),
            "spearman": bootstrap_spearman_fixed(m_rank, draws, names, analysis["spearman"].to_numpy(), cfg)}


def classification_usable(per_cell: pd.DataFrame, cfg: dict) -> pd.Series:
    """Celdas que aportan evidencia a la clasificacion (E02, B13-a): R2 puntual publicable, IC calculado (o
    degenerado por registro) y matriz definida. Las demas (R2_NO_PUBLICABLE_POR_COLINEALIDAD, IC_NO_CALCULABLE,
    MATRIZ_NO_DEFINIDA) cuentan como incumplimiento de AMBAS condiciones."""
    cnp = e02_value(cfg, "analisis.clasificacion.celdas_no_publicables")
    if cnp.get("tratamiento") != "CUENTA_COMO_INCUMPLIMIENTO_DE_AMBAS_CONDICIONES" or \
            cnp.get("incluye") != ["R2_NO_PUBLICABLE_POR_COLINEALIDAD", "IC_NO_CALCULABLE", "MATRIZ_NO_DEFINIDA"]:
        raise ProtocolNotAuthorizedError(f"celdas no publicables distintas de las implementadas: {cnp}")
    return (per_cell["publicable"].astype(bool) & per_cell["ic_estado"].isin(["CALCULADO", "DEGENERADO_REGISTRO"])
            & (per_cell["estado_matriz"] == "DEFINIDA"))


def non_publishable_counts(per_cell: pd.DataFrame, cfg: dict) -> dict[str, int]:
    """Recuento de celdas evaluables (anios y segmentos) no publicables por motivo (informe de B13-a)."""
    ev = per_cell[per_cell["estado"].isin(["EVALUABLE", "EVALUABLE_PARCIAL"]) & per_cell["tipo"].isin(["anio", "segmento"])]
    nodef = ev["estado_matriz"] != "DEFINIDA"
    nopub = ~nodef & ~ev["publicable"].astype(bool)
    noic = ~nodef & ev["publicable"].astype(bool) & ~ev["ic_estado"].isin(["CALCULADO", "DEGENERADO_REGISTRO"])
    usable = classification_usable(ev, cfg)
    return {"MATRIZ_NO_DEFINIDA": int(nodef.sum()), "R2_NO_PUBLICABLE_POR_COLINEALIDAD": int(nopub.sum()),
            "IC_NO_CALCULABLE": int(noic.sum()), "total_no_publicables": int((~usable).sum())}


def classify_column(per_cell: pd.DataFrame, cfg: dict) -> str:
    """analisis.clasificacion: ``per_cell`` con columnas celda, tipo (anio/segmento), estado, r2, ic_inf, ic_sup,
    publicable, ic_estado y estado_matriz. Cuentan los anios EVALUABLE/EVALUABLE_PARCIAL y los segmentos; una celda
    no publicable (E02, B13-a) no satisface ninguna condicion. Nunca devuelve NUEVA_INFORMACION."""
    cl = cfg["analisis"]["clasificacion"]
    thr = cl["umbral_r2"]
    ev = per_cell[per_cell["estado"].isin(["EVALUABLE", "EVALUABLE_PARCIAL"]) & classification_usable(per_cell, cfg)]
    years, segs = ev[ev["tipo"] == "anio"], ev[ev["tipo"] == "segmento"]
    n_years = len(cfg["analisis"]["celdas"]["anios"]["evaluables"])
    n_segs = len(cfg["analisis"]["celdas"]["segmentos"]["etiquetas"])
    if (n_years, n_segs) != (6, 7):
        raise ProtocolNotAuthorizedError("la regla 5/6 anios y 6/7 segmentos exige 6 anios y 7 segmentos")

    def count(d, cond):
        return int(cond(d).sum())
    rec = (count(years, lambda d: (d["r2"] >= thr) & (d["ic_inf"] >= thr)) >= cl["anios_min"] and
           count(segs, lambda d: (d["r2"] >= thr) & (d["ic_inf"] >= thr)) >= cl["segmentos_min"])
    new = (count(years, lambda d: (d["r2"] < thr) & (d["ic_sup"] < thr)) >= cl["anios_min"] and
           count(segs, lambda d: (d["r2"] < thr) & (d["ic_sup"] < thr)) >= cl["segmentos_min"])
    if rec and new:
        raise AssertionError("una columna no puede cumplir a la vez RECONSTRUIBLE y NUEVA_REPRESENTACION")
    return "RECONSTRUIBLE" if rec else "NUEVA_REPRESENTACION" if new else "INESTABLE"


def subnivel(registered: bool, r2_value: float, cfg: dict) -> str:
    """Subnivel con un R2 dado EXPLICITAMENTE (en produccion, el de ``subniveles``)."""
    cl = cfg["analisis"]["clasificacion"]
    if registered:
        return "EXACTA"
    return "PARCIAL" if cl["umbral_parcial"] <= r2_value < cl["umbral_r2"] else \
           "LINEALMENTE_DISTINTA" if r2_value < cl["umbral_parcial"] else "SIN_SUBNIVEL"


def require_unblocked(bloqueos: list[str], conclusion: str) -> None:
    """Conclusiones de TH35 bloqueadas mientras haya un hallazgo bloqueante en su alcance (E02, procedimiento_bloqueo:
    prohibido consultarlas o calcularlas antes del levantamiento; sin desbloqueo automatico)."""
    if bloqueos:
        raise BlockedUntilExplainedError(f"{conclusion} BLOQUEADA_HASTA_EXPLICAR: {list(bloqueos)}")


def subniveles(r2_global: pd.DataFrame, cfg: dict, bloqueos: list[str]) -> pd.DataFrame:
    """Subniveles (E02, B9): EXACTA por registro; PARCIAL / LINEALMENTE_DISTINTA con el R2_TODAS PUNTUAL de la celda
    GLOBAL (``r2_global`` de ``r2_table``). Descriptivos, fuera de la regla de estabilidad. Una columna no
    registrada con R2 global R2_NO_PUBLICABLE_POR_COLINEALIDAD recibe SIN_SUBNIVEL_R2_NO_PUBLICABLE con ese motivo
    (E02, decision 15, N-a): no se sustituye el R2 por ningun otro."""
    require_unblocked(bloqueos, "SUBNIVELES")
    src = e02_value(cfg, "analisis.clasificacion.subniveles_r2")
    if src.get("fuente") != "R2_TODAS_PUNTUAL_CELDA_GLOBAL":
        raise ProtocolNotAuthorizedError(f"fuente de los subniveles distinta de la implementada: {src}")
    na = e02_value(cfg, "analisis.clasificacion.subniveles_r2_global_no_publicable")
    if na.get("subnivel") != "SIN_SUBNIVEL_R2_NO_PUBLICABLE" or na.get("motivo") != "R2_NO_PUBLICABLE_POR_COLINEALIDAD" or \
            na.get("sustitucion") != "NINGUNA":
        raise ProtocolNotAuthorizedError(f"subnivel con R2 global no publicable distinto del implementado: {na}")
    reg = registered_columns(cfg)
    rows = []
    for rec in r2_global.to_dict("records"):
        col = rec["columna"]
        if col in reg:
            rows.append({"columna": col, "subnivel": "EXACTA", "r2_global": rec["r2"], "motivo": "IDENTIDAD_REGISTRADA"})
        elif rec["publicable"]:
            rows.append({"columna": col, "subnivel": subnivel(False, rec["r2"], cfg), "r2_global": rec["r2"], "motivo": None})
        else:
            rows.append({"columna": col, "subnivel": na["subnivel"], "r2_global": rec["r2"], "motivo": na["motivo"]})
    return pd.DataFrame(rows)


def classify_columns(per_cell: pd.DataFrame, cfg: dict, bloqueos: list[str]) -> pd.DataFrame:
    """Clasificacion TH35 de todas las columnas (``per_cell`` largo con la columna ``columna``), con el recuento de
    celdas no publicables por columna. Exige que no haya ningun hallazgo bloqueante (A2-a, A4-a, F5-a, D17-1a)."""
    require_unblocked(bloqueos, "CLASIFICACION_TH35")
    rows = []
    for col, g in per_cell.groupby("columna", sort=False):
        rows.append({"columna": col, "clasificacion": classify_column(g, cfg), **non_publishable_counts(g, cfg)})
    return pd.DataFrame(rows)


def transfer_status(r2_in: float, r2_transf: float, cfg: dict) -> str:
    """Regla de transferencia con R2_in y R2_transf dados (como se miden esta pendiente: R2_TRANSFERENCIA)."""
    t = cfg["analisis"]["r2"]["transferencia"]["transferible_si"]
    if r2_in < cfg["analisis"]["clasificacion"]["umbral_r2"]:
        return "NO_APLICA"
    return "TRANSFERIBLE" if r2_transf >= t["r2_transf_min"] and (r2_in - r2_transf) <= t["delta_max"] else "NO_TRANSFERIBLE"


def min_norm_fit(st_train: CellStats, j: int, predictors: list[int]) -> tuple[np.ndarray, float]:
    """Solucion de minima norma sin regularizacion (transferencia.solucion) en la muestra de entrenamiento:
    devuelve (coeficientes sobre columnas centradas, intercepto) para predecir la columna j."""
    beta, *_ = np.linalg.lstsq(st_train.R[:, predictors], st_train.R[:, j], rcond=None)
    return beta, float(st_train.mean[j] - st_train.mean[predictors] @ beta)


# =========================================================================== 8. sensibilidades y panel L
def series_marks(marks: pd.DataFrame, cfg: dict) -> dict[str, dict[str, np.ndarray]]:
    """Marcas de cada serie con ventana (E01-B, sensibilidades.marcas_por_serie.marca_de_serie)."""
    return {sid: ind.series_discontinuity(marks, spec) for sid, spec in ind.mark_windows(cfg).items()}


def _marked(columns: list[str], cfg: dict) -> list[str]:
    windows = ind.mark_windows(cfg)
    return [c for c in columns if c in windows]                              # referencias TDA-04 y D7: sin marcas


def _s1_marks(marks: pd.DataFrame, cfg: dict, columns: list[str], rows: np.ndarray | None) -> tuple[list[str], np.ndarray]:
    """Marcas cruza_hueco_largo de las series marcadas de ``columns`` en las filas ``rows`` (todas si None), como
    matriz booleana filas x series. Una marca NaN en esas filas contradice E01 (solo antes de K) y detiene: nunca
    se interpreta como 'sin hueco'. Unica lectura de marcas de S1 (exclusion e informe de poblacion)."""
    windows = ind.mark_windows(cfg)
    cols = _marked(columns, cfg)
    n = len(marks) if rows is None else int(np.asarray(rows, bool).sum())
    out = np.zeros((n, len(cols)), bool)
    for k, c in enumerate(cols):
        m = ind.series_discontinuity(marks, windows[c])["cruza_hueco_largo"]
        sel = m if rows is None else m[rows]
        if np.isnan(sel).any():
            raise ProtocolNotAuthorizedError(f"marca NaN de {c} dentro de la poblacion: contradice E01")
        out[:, k] = sel > 0
    return cols, out


def s1_exclusion(marks: pd.DataFrame, cfg: dict, columns: list[str], rows: np.ndarray | None = None) -> np.ndarray:
    """S1 (E01): excluye la fila si cruza_hueco_largo = 1 para CUALQUIERA de las series del calculo (regla
    varias_series). Una marca NaN en las filas consideradas contradice E01 (solo antes de K) y detiene."""
    return _s1_marks(marks, cfg, columns, rows)[1].any(axis=1)


def s1_population_report(marks: pd.DataFrame, cfg: dict, columns: list[str], rows: np.ndarray) -> pd.DataFrame:
    """Control de POBLACION EFECTIVA y evaluabilidad de S1 con la regla de union (requisito de la auditoria de
    E01; sin umbrales nuevos): por serie marcada, filas de ``rows`` que excluye su propia marca y filas que SOLO
    ella excluye; fila ``__UNION__`` con las excluidas por la union, la poblacion efectiva restante y si el
    calculo es posible (al menos 2 filas: condicion matematica, no un umbral)."""
    cols, M = _s1_marks(marks, cfg, columns, rows)
    union = M.any(axis=1)
    only = M & (M.sum(axis=1) == 1)[:, None]
    rep = pd.DataFrame({"serie": cols, "excluye": M.sum(axis=0), "excluye_solo_ella": only.sum(axis=0)})
    n, kept = int(rows.sum()), int((~union).sum())
    rep = pd.concat([rep, pd.DataFrame({"serie": ["__UNION__"], "excluye": [int(union.sum())],
                                        "excluye_solo_ella": [np.nan]})], ignore_index=True)
    rep.attrs.update(poblacion=n, poblacion_efectiva=kept, fraccion_retenida=kept / n if n else np.nan,
                     calculable=kept >= 2)
    return rep


def s2_exclusion(marks: pd.DataFrame, cfg: dict, columns: list[str]):
    """S2 solo para escalas S y M: bloqueada mientras la asignacion de escalas no figure en una enmienda aprobada."""
    from ohlcv_dataroad.ingest.tda17_enmiendas import require_s2_scale_assignment
    require_s2_scale_assignment(cfg)
    pending("S2_ESCALAS")


def s3_bounds(cfg: dict) -> tuple[float, float]:
    """S3: 'Pearson winsorizado 0.1 / 99.9 % por celda' -> cuantiles 0.001 y 0.999."""
    txt = cfg["sensibilidades"]["S3"]
    if "0.1 / 99.9 %" not in txt:
        raise ProtocolNotAuthorizedError(f"S3 no es la declarada: {txt}")
    return 0.001, 0.999


def s5_demean(panel: pd.DataFrame, timestamps: pd.Series) -> pd.DataFrame:
    """S5 (RETROSPECTIVO): resta a cada serie su media por minuto del dia NY x anio NY, calculada sobre las
    filas recibidas (la poblacion de la celda)."""
    t = pd.to_datetime(timestamps)
    t = t.dt.tz_localize("UTC") if t.dt.tz is None else t.dt.tz_convert("UTC")
    ny = t.dt.tz_convert(ind.NY_TZ)
    key = pd.MultiIndex.from_arrays([ny.dt.year.to_numpy(), (ny.dt.hour * 60 + ny.dt.minute).to_numpy()])
    vals = panel.reset_index(drop=True)
    means = vals.groupby(key).transform("mean")
    return pd.DataFrame(vals.to_numpy() - means.to_numpy(), index=panel.index, columns=panel.columns)


def s6_exclusion(trading_date: pd.Series, cfg: dict) -> np.ndarray:
    """S6: excluye las filas cuya trading_date esta en la lista cerrada."""
    s6 = cfg["sensibilidades"]["S6"]
    if not s6.get("lista_cerrada"):
        raise ProtocolNotAuthorizedError("S6 exige una lista cerrada de fechas")
    dates = {pd.Timestamp(d).date() for d in s6["fechas"]}
    return pd.to_datetime(trading_date).dt.date.isin(dates).to_numpy()


def s7_s8_exclusion(*_args, **_kwargs):
    pending("S7_S8_ALCANCE")


def panel_l_diagnostics(series: pd.DataFrame, panel_p: pd.DataFrame, df: pd.DataFrame, cfg: dict,
                        population: np.ndarray) -> pd.DataFrame:
    """panel_L.analisis (por contrato): Pearson y Spearman de cada serie L con su contraparte adimensional y con
    C_t, en las filas de la poblacion (casos completos de P) con la serie L finita."""
    cp = l_counterparts(cfg)
    contract = df["contract"].to_numpy()
    close = df["close"].to_numpy(np.float64)
    rows = []
    for lcol, pcol in cp.items():
        x_all = series[lcol].to_numpy(np.float64)
        y_all = panel_p[pcol].to_numpy(np.float64)
        for c in pd.unique(contract):
            m = population & (contract == c) & np.isfinite(x_all)
            rec = {"contrato": c, "serie_L": lcol, "contraparte": pcol, "n": int(m.sum())}
            for tag, y in (("contraparte", y_all), ("C", close)):
                rec[f"pearson_{tag}"], rec[f"spearman_{tag}"] = _pair_corr(x_all[m], y[m])
            rows.append(rec)
    return pd.DataFrame(rows)


def panel_l_rank_checks(series: pd.DataFrame, df: pd.DataFrame, cfg: dict, population: np.ndarray) -> list[dict]:
    """Control de nulos del panel L por contrato (E02: correlacion de Pearson por contrato, ambito INFORMATIVO):
    filas de la poblacion con las 16 series L finitas; matriz NO_DEFINIDA si alguna tiene varianza cero."""
    check_null_convention(cfg)
    cols = l_columns(cfg)
    vn = cfg["identidades"]["verificacion_numerica"]
    L = series[cols].to_numpy(np.float64)
    contract = df["contract"].to_numpy()
    out = []
    for c in pd.unique(contract):
        m = population & (contract == c) & np.isfinite(L).all(axis=1)
        und = undefined_columns(L[m], cols)
        if und["varianza_cero"]:
            out.append({"matriz": f"L_{c}", "estado": "NO_DEFINIDA", "ambito": "INFORMATIVA", "n": int(m.sum()), **und})
            continue
        rec = rank_check(null_counts(cell_stats(L[m]), vn["tau_relativa"]), vn["nulos_esperados"]["L"], f"L_{c}", "INFORMATIVA")
        out.append({**rec, "n": int(m.sum())})
    return out


def _pair_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan
    pe = float(np.corrcoef(x, y)[0, 1])
    sp = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    return pe, sp


# =========================================================================== orquestacion por celda
def settings(cfg: dict) -> dict:
    """Parametros numericos de la configuracion efectiva que usa el motor (todos leidos del protocolo)."""
    a = cfg["analisis"]
    return {"tau": a["r2"]["tau_relativa"], "tol_cruzada": a["r2"]["tolerancia_cruzada"],
            "tau_rango": cfg["identidades"]["verificacion_numerica"]["tau_relativa"],
            "nulos": cfg["identidades"]["verificacion_numerica"]["nulos_esperados"],
            "cortes": a["agrupamiento"]["cortes_abs_rho"]}


def analyze_cell(panel_p: pd.DataFrame, df: pd.DataFrame, cfg: dict, *, celda: str,
                 sensibilidad: str | None = None) -> dict:
    """Analisis de redundancia de UNA celda sobre su poblacion (casos completos de P): Pearson, Spearman,
    R2_TODAS (oficial, sobre rangos como variante secundaria) y R2_TDA04 con verificaciones cruzadas, residuos
    de identidades, etiquetas de exactitud, recuentos de nulos (P sin centrar, P centrada, B) con la convencion y el
    ambito de E02, PCA de P y B y agrupamiento. ``celda`` (nombre de ``cell_definitions``) fija el ambito del
    control; con ``sensibilidad`` el bloqueo alcanza solo a esa sensibilidad (E02, A4-a).

    Matriz NO_DEFINIDA (F5-a): si alguna columna tiene varianza cero en la poblacion, no se calcula ningun analisis
    matricial; la celda cuenta como incumplimiento para todas las columnas y, si es GLOBAL, bloquea (TH35 o la
    sensibilidad). Si hay bloqueos, se emiten los diagnosticos obligatorios (``diagnostico_bloqueo``) y las
    conclusiones quedan BLOQUEADO_HASTA_EXPLICAR. No calcula IC (``cell_bootstrap``), transferencia ni k_PA."""
    check_panel_counts(cfg)
    check_null_convention(cfg)
    if list(panel_p.columns) != p_columns(cfg):
        raise ValueError("el panel no tiene las 134 columnas oficiales de P en su orden")
    if not complete_cases(panel_p).all():
        raise ValueError("analyze_cell recibe solo la poblacion de casos completos de P")
    ambito = rank_scope(celda, cfg)
    alcance = f"SENSIBILIDAD_{sensibilidad}" if sensibilidad else "TH35"
    s = settings(cfg)
    names = list(panel_p.columns)
    X = panel_p.to_numpy(np.float64)
    und = undefined_columns(X, names)
    if und["varianza_cero"] or und["norma_cero"]:
        blocks = [f"MATRIZ_NO_DEFINIDA: celda GLOBAL con columnas sin varianza {und['varianza_cero']}"] \
            if celda == "GLOBAL" else []
        return {"celda": celda, "n": int(X.shape[0]), "estado_matriz": "NO_DEFINIDA", "ambito": ambito, **und,
                "bloqueos": blocks, "alcance_bloqueo": alcance if blocks else None,
                "estado_conclusiones": "BLOQUEADO_HASTA_EXPLICAR" if blocks else "INCUMPLIMIENTO_TODAS_LAS_COLUMNAS"}
    st = cell_stats(X)
    st_rank = cell_stats(average_ranks(X))
    st_raw = cell_stats(X, centered=False)
    bcols = b_columns(cfg)
    st_b = cell_stats(panel_p[bcols].to_numpy(np.float64))
    r2_all = r2_table(st, names, None, s["tau"], s["tol_cruzada"])
    r2_rank = r2_table(st_rank, names, None, s["tau"], s["tol_cruzada"])
    r2_ref = r2_table(st, names, tda04_predictors(names, cfg), s["tau"], s["tol_cruzada"])
    residuals = identity_residuals(panel_p, df, cfg, "P")
    labels = exactness_labels(dict(zip(r2_all["columna"], r2_all["r2"])), residuals, cfg)
    checks = [rank_check(null_counts(st_raw, s["tau_rango"]), s["nulos"]["P_sin_centrar"], "P_sin_centrar", ambito),
              rank_check(null_counts(st, s["tau_rango"]), s["nulos"]["P_centrada"], "P_centrada", ambito),
              rank_check(null_counts(st_b, s["tau_rango"]), s["nulos"]["B_centrada"], "B_centrada", ambito)]
    lam_p, lam_b = correlation_eigenvalues(st.corr), correlation_eigenvalues(st_b.corr)
    blocks = blocking_findings(labels, checks)
    out = {"celda": celda, "n": st.n, "estado_matriz": "DEFINIDA", "ambito": ambito,
           "pearson": pd.DataFrame(st.corr, index=names, columns=names),
           "spearman": pd.DataFrame(st_rank.corr, index=names, columns=names),
           "r2_todas": r2_all, "r2_rangos": r2_rank, "r2_tda04": r2_ref,
           "identidades": residuals, "exactitud": labels, "rango": checks,
           "bloqueos": blocks, "alcance_bloqueo": alcance if blocks else None,
           "estado_conclusiones": "BLOQUEADO_HASTA_EXPLICAR" if blocks else "SIN_BLOQUEO",
           "dimension_P": dimension_metrics(lam_p), "dimension_B": dimension_metrics(lam_b),
           "autovalores_P": lam_p, "autovalores_B": lam_b,
           "agrupamiento": hierarchical_clusters(st_rank.corr, names, s["cortes"])}
    if any(r["estado"] == "BLOQUEAR_HASTA_EXPLICAR" for r in checks):
        mats = {"P_sin_centrar": (st_raw, names), "P_centrada": (st, names), "B_centrada": (st_b, bcols)}
        out["diagnostico_bloqueo"] = blocking_dossier(mats, checks, {"R2_TODAS": r2_all, "R2_TDA04": r2_ref}, labels,
                                                      residuals, cfg, celda, alcance, panel_p)
    return out


def production_blockers() -> list[str]:
    """Puntos que impiden hoy una ejecucion COMPLETA de TDA-17 sobre datos reales (definiciones pendientes)."""
    return [f"{k}: {v}" for k, v in PENDIENTES.items()]


# =========================================================================== orquestacion TH35 y rutas de produccion
PIPELINE_CELL_TYPES = {"anio": "anio", "segmento": "segmento"}


def _cell_rows(cell: str, analysis: dict, boot: dict | None, cdef: dict, names: list[str]) -> list[dict]:
    """Filas de ``per_cell`` (una por columna) de una celda de anio o segmento, DERIVADAS del analisis y del
    bootstrap de la propia celda (nunca aportadas por el llamante)."""
    base = {"celda": cell, "tipo": PIPELINE_CELL_TYPES[cdef["tipo"]], "estado": cdef["estado"]}
    if analysis["estado_matriz"] != "DEFINIDA":
        return [{**base, "columna": c, "r2": np.nan, "ic_inf": np.nan, "ic_sup": np.nan, "publicable": False,
                 "ic_estado": "MATRIZ_NO_DEFINIDA", "estado_matriz": "NO_DEFINIDA"} for c in names]
    r2 = analysis["r2_todas"].set_index("columna")
    ic = boot["r2_todas"].set_index("columna")
    return [{**base, "columna": c, "r2": float(r2.loc[c, "r2"]), "ic_inf": float(ic.loc[c, "ic_inf"]),
             "ic_sup": float(ic.loc[c, "ic_sup"]), "publicable": bool(r2.loc[c, "publicable"]),
             "ic_estado": ic.loc[c, "ic_estado"], "estado_matriz": "DEFINIDA"} for c in names]


def th35_pipeline(df: pd.DataFrame, cfg: dict) -> dict:
    """TH35 COMPLETO a partir de la entrada de TDA-04 ``df`` (sin columna sintetica), sin ningun resultado intermedio
    aportado desde fuera:

    1. TH37 (obligatoria): auditoria causal completa de ``df`` y verificacion contra el plan predeclarado
       (ausente, incompleta, fuera del plan o con discrepancias -> rechazo; TH35 nunca se calcula sin ella).
    2. Catalogo y panel P por la via de produccion de la fase 1; poblacion = casos completos de P.
    3. Por cada celda con flujo (GLOBAL, anios, segmentos): ``analyze_cell`` y, si la matriz esta definida,
       ``cell_bootstrap`` (R2_TODAS y Spearman con rangos fijos, mismas replicas).
    4. ``per_cell`` y el R2 global DERIVADOS de 3; bloqueos de todas las celdas (rango, matriz GLOBAL no definida,
       identidad registrada).
    5. Validacion de Spearman (GLOBAL y ANIO_2024): si alguna de las dos celdas no tiene matriz definida, la
       validacion NO es ejecutable y los IC de Spearman no se publican (nunca se dan por validados).
    6. Con bloqueos: BLOQUEADO_HASTA_EXPLICAR, sin clasificacion ni subniveles. Sin bloqueos: clasificacion TH35,
       recuento de celdas no publicables y subniveles."""
    from ohlcv_dataroad.ingest import tda17_causalidad as ca
    if ind.SYNTHETIC_MARKER in df.columns:
        raise ValueError("th35_pipeline recibe la entrada sin la columna de marco sintetico")
    df = df.reset_index(drop=True)
    table = ca.run_causal_audit(df, cfg)
    ca.verify_causal_audit(table, df, cfg)
    series, _marks = ind.compute_catalog(df)
    panel = ind.build_panel_p(series, df)
    ok = complete_cases(panel)
    cells = cell_definitions(df["timestamp"], cfg)
    names = list(panel.columns)
    analyses, boots, rows, blocks = {}, {}, [], []
    for cell in bootstrap_stream_cells(cfg):
        sel = cells[cell]["mascara"] & ok
        sub, sub_df = panel[sel], df[sel]
        a = analyze_cell(sub, sub_df, cfg, celda=cell)
        analyses[cell] = a
        blocks += [f"{cell}: {b}" for b in a["bloqueos"]]
        if a["estado_matriz"] == "DEFINIDA":
            boots[cell] = cell_bootstrap(sub, sub_df["trading_date"].to_numpy(), a, cfg, cell)
        if cells[cell]["tipo"] in PIPELINE_CELL_TYPES:
            rows += _cell_rows(cell, a, boots.get(cell), cells[cell], names)
    per_cell = pd.DataFrame(rows)
    vcells = e02_value(cfg, "analisis.spearman.validacion_detalle")["celdas"]
    if all(analyses.get(c, {}).get("estado_matriz") == "DEFINIDA" for c in vcells):
        data = {c: (panel[cells[c]["mascara"] & ok].to_numpy(np.float64),
                    df.loc[cells[c]["mascara"] & ok, "trading_date"].to_numpy()) for c in vcells}
        validation = spearman_validation(data, cfg)
    else:
        validation = {"resultado": "NO_EJECUTABLE", "publicar_ic_spearman": False,
                      "motivo": f"celdas de validacion sin matriz definida: {vcells}"}
    spearman = {c: spearman_publication(b["spearman"], validation) for c, b in boots.items()}
    out = {"auditoria_causalidad": table, "analisis_celdas": analyses, "bootstrap": boots, "per_cell": per_cell,
           "validacion_spearman": validation, "spearman": spearman, "bloqueos": blocks,
           "poblacion": int(ok.sum())}
    if blocks:
        out["estado"] = "BLOQUEADO_HASTA_EXPLICAR"
        return out
    r2_global = analyses["GLOBAL"]["r2_todas"]
    out.update(estado="SIN_BLOQUEO", clasificacion=classify_columns(per_cell, cfg, blocks),
               subniveles=subniveles(r2_global, cfg, blocks))
    return out


def official_th35(publicacion: str, root=None) -> dict:
    """PRODUCCION: TH35 de la ejecucion oficial. Solo recibe la publicacion P2: la configuracion se deriva de M2, las
    entradas se leen y se verifican contra el manifiesto (``tda17_entradas``) y todo lo demas (TH37, celdas, IC,
    clasificacion) lo calcula ``th35_pipeline``. Devuelve tambien la procedencia."""
    from ohlcv_dataroad.ingest import tda17_entradas as ent
    cfg = official_config(publicacion, root)
    df, procedencia = ent.load_official_inputs(cfg, root)
    out = th35_pipeline(df, cfg)
    out["procedencia"] = procedencia
    return out
