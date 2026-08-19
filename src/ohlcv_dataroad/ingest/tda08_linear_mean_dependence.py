"""TDA-08 -- Dependencia lineal en la media.

Implementa la etapa TDA-08 del roadmap
(``docs/methodology/Tsay_OHLCV_analysis_roadmap.md``, seccion "TDA-08"):
mide la dependencia lineal de ``r_t`` con su pasado (ACF/PACF/portmanteau),
la traduce a una magnitud interpretable en ticks, y la somete
OBLIGATORIAMENTE a un protocolo de refutacion microestructural antes de
llamarla "dependencia".

Resuelve TH16, TH17 y TH18 (ver informe,
``reports/mnq/TDA08_dependencia_lineal_media.md``, para el estado final
de cada una).

CORRECCION METODOLOGICA -- 1a revision (posterior al cierre inicial de
esta etapa; ver seccion "Correccion metodologica" del informe): una
revision externa detecto que la primera version de ``compute_acf``
normalizaba ``gamma_k`` por ``T`` (la poblacion COMPLETA), heredado
conceptualmente del HAC de TDA-07 -- pero HAC y ACF no son el mismo
objeto. Esa normalizacion ATENUABA ``rho_k`` mecanicamente en funcion de
cuantos pares fueran observables en cada rezago. Se corrigio a una
correlacion pairwise sobre exactamente los pares validos de cada rezago.

CORRECCION METODOLOGICA -- 2a revision ("centrado del estimador"): una
segunda revision detecto que, aun siendo "pairwise" en que pares se
incluian, el CENTRADO (``e = x - media``) seguia usando la media GLOBAL
de toda la poblacion en vez de la media propia de cada conjunto de pares
-- y, de forma mas grave, ``acf_by_group`` (año/segmento/decil de
volumen/mes) centraba TODOS los grupos por esa misma media global, lo
que puede fabricar dependencia serial artificial a partir de una simple
diferencia de medias ENTRE grupos. Se corrigio ambos casos a un centrado
LOCAL (por conjunto de pares en ``compute_acf``/``bootstrap_rho``, por
grupo en ``acf_by_group``) -- ver docstrings de cada funcion para el
detalle y la justificacion tecnica. Se añadio ademas ``g2_synthetic_null_moment_check``,
que verifico -- en vez de solo afirmar -- si el null sintetico de G2
preserva volatilidad/curtosis reales.

CORRECCION METODOLOGICA -- 3a revision ("cierre final, correccion
puntual de G2"): el diagnostico de momentos de la 2a revision, ejecutado
sobre el conjunto de investigacion real, demostro que Null 2 (empirico
reescalado por ``s(m)``) NO preserva varianza ni curtosis reales
(varianza ~37% mayor, curtosis en exceso ~3,6 veces mayor) -- solo
preserva el perfil de escala por minuto. Por eso la inferencia PRINCIPAL
de G2 pasa a usar EXCLUSIVAMENTE Null 1 (``g2_null1_calibration_summary``,
``g2_null1_portmanteau_summary``); la calibracion combinada de ambos
nulls se conserva SOLO como sensibilidad historica/secundaria
(``g2_combined_calibration_summary``, ``g2_combined_portmanteau_summary``,
renombradas para dejar claro su rol). Null 2 no se rediseño -- se
documenta su fallo y se excluye de la inferencia principal, tal como
prefirio la revision.

CIERRE DEFINITIVO -- 4a revision (metadatos, documentacion, consistencia,
sin cambios de metodologia estadistica): la 3a revision dejo un problema
de metadatos sin resolver -- ``annotate_portmanteau_calibration`` se
aplicaba de forma generica a ambas series (`r_1m` y `r_tilde`), por lo
que `r_tilde` podia aparecer marcada ``G2_CALIBRATED`` para `m<=60` SIN
tener ninguna calibracion Null-1 propia (Null 1 se construye
exclusivamente sobre `r_1m`). Se corrigio exigiendo que cada llamador
declare EXPLICITAMENTE, via el argumento obligatorio
``has_null1_calibration``, si la serie tiene calibracion G2 propia --
`r_1m` (``True``) recibe ``G2_CALIBRATED``/``DESCRIPTIVE_UNCALIBRATED``
segun ``m``; `r_tilde` (``False``) recibe siempre ``NOT_G2_CALIBRATED``
para los rezagos estimables. No se implemento una calibracion G2 Null-1
propia para `r_tilde` -- decision explicita de cierre (`r_tilde` es un
diagnostico retrospectivo complementario en esta etapa, no la base de
TH16; ampliar el sistema de nulls solo para calibrarla habria expandido
el alcance sin necesidad metodologica). Ademas: limpieza de referencias
documentales obsoletas (conteos de artefactos/tests de revisiones
anteriores sin etiquetar como historicos, terminologia residual
"Ljung-Box" para el estadistico vigente, menciones de nombres de
funciones ya renombradas sin indicarlo).

TOPOLOGIA TEMPORAL (bloqueante): ningun calculo de ACF/PACF/portmanteau
compacta primero los retornos validos y los trata despues como una
serie continua. Cada rezago k solo empareja dos observaciones si existe
una cadena temporal REALMENTE continua de k minutos entre ellas -- misma
logica de bloques que TDA-07 (HAC) y TH10 (r[h]), generalizada aqui a un
delta esperado configurable (60s para r_1m nativo; 5/10 minutos para el
analisis 1->5->10 no solapado de TH17).

Que NO hace este modulo (deliberadamente, fuera de alcance de TDA-08):

- No estudia volatility clustering en |r|/r^2 (TDA-09).
- No ajusta ARCH/GARCH (TDA-09/TDA-11).
- No ejecuta EVT ni tests de no linealidad (BDS) -- etapas posteriores.
- No crea features, targets, señales operativas ni hace backtesting.
- No abre `data/raw/` ni `config.holdout_files`.
- No modifica ningun artefacto de TDA-00..TDA-07 ni del complemento TH10.
- No ajusta AR(p) por defecto (G4).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ohlcv_dataroad.config import SnapshotConfig
from ohlcv_dataroad.ingest.holdout_guard import (
    validate_last_timestamps_before_boundary,
    validate_research_holdout_disjoint,
)
from ohlcv_dataroad.ingest.session_calendar import NY_TZ
from ohlcv_dataroad.ingest.tda07_marginal_distribution import (
    build_r1m_population,
    build_r_tilde_population,
    compute_tick_return_repr,
    load_r_tilde,
    load_segmentation_cutoffs,
    load_tda04_inputs,
    verify_r_tilde_invariants,
    verify_timestamp_alignment,
)
from ohlcv_dataroad.ingest.th10_horizon_scaling import (
    build_horizon_returns,
    non_overlap_mask,
)

# ---------------------------------------------------------------------------
# Convenciones PREDECLARADAS
# ---------------------------------------------------------------------------

SESSION_MINUTES = 1380  # TDA-02: "1.380 minutos esperados por sesion, sin excepcion"
MAX_LAG_REQUESTED = 2 * SESSION_MINUTES  # 2.760 -- el objetivo ASPIRACIONAL del roadmap ("2 jornadas")

SHORT_MAX_LAG = 20
DECILE_MAX_LAG = 5
BOOTSTRAP_LAGS = (1, 2, 5, 10, 20)
LJUNG_BOX_M = (10, 20, 40, 60, SESSION_MINUTES, MAX_LAG_REQUESTED)
PACF_MAX_LAG = 40

DEFAULT_N_BOOT = 300
DEFAULT_BOOTSTRAP_SEED = 0

DEFAULT_N_PERM = 200
PERMUTATION_SEED = 1
SYNTHETIC_NULL_SEED = 2
NULL_PERCENTILE = 97.5
G2_NULL_MAX_LAG = 60

N_VOLUME_DECILES = 10
MIN_N_FOR_SUBGROUP_ACF = 200

WINDOW_OPEN = (571, 575)   # 09:31-09:35 NY inclusive
WINDOW_CLOSE = (952, 962)  # 15:52-16:02 NY inclusive

COMPLETE_YEARS = (2020, 2021, 2022, 2023, 2024)
PARTIAL_YEARS = (2019, 2025)


# ---------------------------------------------------------------------------
# Bloques de continuidad temporal -- generalizacion parametrizada de la
# MISMA logica de TDA-07 (``compute_hac_block_ids``) y TH10
# ---------------------------------------------------------------------------

def compute_block_ids(timestamp: pd.Series, trading_date: pd.Series, expected_delta_seconds: float) -> np.ndarray:
    """Bloques de continuidad GENUINA: dos filas consecutivas (por timestamp) quedan en el mismo bloque sii delta==expected_delta_seconds Y mismo trading_date."""
    order = np.argsort(pd.to_datetime(timestamp).to_numpy())
    ts_sorted = pd.to_datetime(timestamp).to_numpy()[order]
    td_sorted = np.asarray(trading_date)[order]

    n = len(ts_sorted)
    if n == 0:
        return np.array([], dtype=int)

    delta_seconds = np.diff(ts_sorted).astype("timedelta64[ns]").astype(np.int64) / 1e9
    same_date = td_sorted[1:] == td_sorted[:-1]
    continues = (np.abs(delta_seconds - expected_delta_seconds) < 1e-6) & same_date
    new_block = np.concatenate([[True], ~continues])
    block_ids_sorted = np.cumsum(new_block) - 1

    block_ids = np.empty(n, dtype=int)
    block_ids[order] = block_ids_sorted
    return block_ids


def _day_block_index_map(trading_date: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    unique_dates, inverse = np.unique(trading_date, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    boundaries = np.searchsorted(sorted_inverse, np.arange(len(unique_dates) + 1))
    date_to_rows = [order[boundaries[i]:boundaries[i + 1]] for i in range(len(unique_dates))]
    return unique_dates, date_to_rows


# ---------------------------------------------------------------------------
# ACF -- correlacion PAIRWISE sobre pares validos (CORREGIDA)
# ---------------------------------------------------------------------------

def compute_acf(values: np.ndarray, block_ids: np.ndarray, max_lag: int) -> pd.DataFrame:
    """``rho_k`` = correlacion de Pearson pairwise-complete LITERAL sobre los pares (t,t-k) del mismo bloque.

    CORRECCION METODOLOGICA (2a revision -- "centrado del estimador"):
    la version anterior calculaba ``e = x - mean(x)`` con la media
    GLOBAL de TODA la poblacion, una sola vez, y despues seleccionaba
    los pares validos de cada rezago sobre ese ``e`` ya centrado. Eso
    NO es Pearson pairwise-complete literal (que es el nombre que el
    docstring ya usaba) -- una correlacion pairwise-complete genuina
    (el mismo comportamiento que ``cor(x, y, use="pairwise.complete.obs")``
    en R) centra ``x`` (el lado ``t``) y ``y`` (el lado ``t-k``) CADA UNO
    por SU PROPIA media, calculada EXCLUSIVAMENTE sobre el conjunto de
    pares ``P_k`` de ese rezago -- nunca por la media de la poblacion
    completa.

    Se decidio explicitamente la opcion (A) frente a la alternativa (B)
    de mantener una media comun global bajo un supuesto de media
    estacionaria unica: (A) es la definicion que corresponde al nombre
    "pairwise-complete" ya declarado, y es la unica que evita que una
    diferencia de medias ENTRE GRUPOS (año/segmento/decil de volumen)
    se traduzca en dependencia serial artificial al agrupar (ver
    ``acf_by_group`` mas abajo, y el test dedicado
    ``test_acf_by_group_does_not_inject_spurious_dependence_from_between_group_mean_differences``).
    Para la ACF GLOBAL (una sola poblacion, sin agrupar) el efecto
    numerico de este cambio es pequeño -- el conjunto de pares ``P_k``
    excluye solo los pares que cruzan una frontera de bloque, una
    fraccion tipicamente pequeña de la poblacion total salvo en la cola
    de rezagos largos -- pero la definicion correcta se aplica siempre,
    sin excepciones, para no tener dos convenciones distintas segun el
    caso de uso.

    Definicion:

        x_t  = valores en la posicion ``t``  para ``(t,t-k) en P_k``
        x_tk = valores en la posicion ``t-k`` para ``(t,t-k) en P_k``
        e_t  = x_t  - mean(x_t)    (media SOLO sobre ``P_k``)
        e_tk = x_tk - mean(x_tk)   (media SOLO sobre ``P_k``)

        rho_k = sum(e_t * e_{t-k}) / sqrt(sum(e_t^2) * sum(e_{t-k}^2))

    -- garantizado en ``[-1,1]`` por Cauchy-Schwarz sobre ``P_k``, sin
    depender mecanicamente de cuantos pares existan fuera de el
    (verificado en ``test_acf_multiblock_does_not_attenuate_known_ar1_dependence``).

    Ademas de ``rho`` se calcula ``beta``: la pendiente OLS **con
    intercepto** de regresar ``x_t`` sobre ``x_tk`` sobre exactamente
    ``P_k`` (``beta_k = sum(e_t*e_{t-k})/sum(e_{t-k}^2)``, con las
    mismas medias locales que ``rho`` -- esto ES, literalmente, la
    pendiente OLS con intercepto sobre la muestra ``P_k``, porque el
    intercepto de una regresion OLS es exactamente el que hace que los
    residuos centrados por la media de cada variable tengan media cero;
    restar la media de ``x_t`` y de ``x_tk`` por separado sobre la MISMA
    muestra ``P_k`` es equivalente a estimar la regresion con
    intercepto). ``rho`` y ``beta`` son cantidades DISTINTAS que solo
    coinciden exactamente cuando ``sum(e_t^2)==sum(e_{t-k}^2)``; nunca se
    presupone su igualdad (ver ``translate_dependence_to_ticks``).

    Rendimiento (seccion 13 de la 1a revision): ``n_pairs_k`` es, por
    construccion, NO CRECIENTE en ``k`` -- en cuanto llega a 0 en algun
    rezago, sera 0 para TODO rezago mayor, y se rellenan esos rezagos
    como ``NOT ESTIMABLE`` sin volver a recorrer el array.

    Salida: ``DataFrame`` con ``lag``, ``rho``, ``beta``, ``n_pairs``,
    ``estimable`` (``False`` cuando ``n_pairs<2``).
    """
    x_all = np.asarray(values, dtype=float)
    T = x_all.size

    rows: list[dict] = []
    k = 1
    while k <= max_lag:
        if k >= T:
            rows.append({"lag": k, "rho": np.nan, "beta": np.nan, "n_pairs": 0, "estimable": False})
            k += 1
            continue
        same_block = block_ids[k:] == block_ids[:-k]
        n_pairs = int(same_block.sum())
        if n_pairs == 0:
            # No creciente en k (bloques de longitud fija): sera 0 para
            # siempre a partir de aqui -- se rellena sin recalcular.
            for kk in range(k, max_lag + 1):
                rows.append({"lag": kk, "rho": np.nan, "beta": np.nan, "n_pairs": 0, "estimable": False})
            break
        if n_pairs < 2:
            rows.append({"lag": k, "rho": np.nan, "beta": np.nan, "n_pairs": n_pairs, "estimable": False})
            k += 1
            continue

        x_t = x_all[k:][same_block]
        x_tk = x_all[:-k][same_block]
        mean_t = x_t.mean()
        mean_tk = x_tk.mean()
        e_t = x_t - mean_t
        e_tk = x_tk - mean_tk
        sum_prod = float(np.sum(e_t * e_tk))
        sum_t2 = float(np.sum(e_t * e_t))
        sum_tk2 = float(np.sum(e_tk * e_tk))
        denom = np.sqrt(sum_t2 * sum_tk2)
        rho = (sum_prod / denom) if denom > 0 else np.nan
        beta = (sum_prod / sum_tk2) if sum_tk2 > 0 else np.nan
        rows.append({"lag": k, "rho": rho, "beta": beta, "n_pairs": n_pairs, "estimable": True})
        k += 1

    return pd.DataFrame(rows)


def bartlett_se(rho: np.ndarray, T: int) -> np.ndarray:
    """Error estandar de Bartlett -- referencia CLASICA APROXIMADA, NUNCA el intervalo principal (seccion 5 de la 2a revision).

    ``SE(rho_k) = sqrt((1 + 2*sum_{j=1}^{k-1} rho_j^2) / T)``. Formula de
    texto (Box-Jenkins/Brockwell-Davis) derivada para una ACF de
    denominador fijo ``T`` sobre una UNICA serie continua sin bloques --
    no fue derivada para el estimador pairwise-complete de esta etapa,
    que se calcula sobre un conjunto de pares ``P_k`` variable por
    rezago y con topologia fragmentada por bloques de continuidad. Se
    reporta EXCLUSIVAMENTE como diagnostico visual clasico aproximado
    (banda "Bartlett 95%" en el grafico) -- la inferencia PRINCIPAL de
    esta etapa es siempre el bootstrap de bloques de jornada
    (``bootstrap_rho``), que si respeta la dependencia intradia y la
    topologia fragmentada. Nunca se presenta la banda de Bartlett como
    un intervalo de confianza valido para la topologia bloqueada.
    ``rho`` puede contener ``NaN`` en rezagos no estimables -- una vez
    que aparece un ``NaN`` en la secuencia, todo ``SE`` posterior tambien
    es (correctamente) indefinido, porque la formula necesita TODOS los
    rezagos anteriores.
    """
    sq = np.asarray(rho, dtype=float) ** 2
    excl_cumsum = np.concatenate([[0.0], np.cumsum(sq)[:-1]])
    var = (1.0 + 2.0 * excl_cumsum) / T
    return np.sqrt(np.maximum(var, 0.0))


def durbin_levinson_pacf(rho: np.ndarray, max_lag: int) -> np.ndarray:
    """PACF (``phi_kk``) a partir de la ACF, via Durbin-Levinson -- diagnostico CLASICO aproximado solo en rezagos bajos.

    Como ``bartlett_se``, la recursion de Durbin-Levinson asume una ACF
    de una unica serie continua con denominador fijo -- se aplica aqui
    como referencia de forma (¿hay corte abrupto? ¿decaimiento
    gradual?), no como una descomposicion exacta de la topologia
    bloqueada de esta etapa. Diagnostico, no inferencia principal.
    """
    rho = np.asarray(rho, dtype=float)
    phi_prev = np.zeros(max_lag + 1)
    pacf = np.zeros(max_lag)
    phi_prev[1] = rho[0]
    pacf[0] = rho[0]
    for k in range(2, max_lag + 1):
        num = rho[k - 1] - float(np.sum(phi_prev[1:k] * rho[k - 2::-1]))
        den = 1.0 - float(np.sum(phi_prev[1:k] * rho[0:k - 1]))
        phi_kk = num / den if den != 0 else 0.0
        phi_curr = phi_prev.copy()
        phi_curr[k] = phi_kk
        for j in range(1, k):
            phi_curr[j] = phi_prev[j] - phi_kk * phi_prev[k - j]
        pacf[k - 1] = phi_kk
        phi_prev = phi_curr
    return pacf


def compute_portmanteau_q(acf_tab: pd.DataFrame, m_values: tuple[int, ...]) -> pd.DataFrame:
    """Estadistico portmanteau adaptado a la correlacion PAIRWISE: ``Q(m) = sum_{k=1}^{m} n_pairs_k * rho_k^2``.

    NO es literalmente Ljung-Box clasico (que asume el estimador de ACF
    de denominador fijo ``T``, con el factor de correccion
    ``T(T+2)/(T-k)``) -- con la correlacion PAIRWISE de esta etapa, la
    varianza aproximada de ``rho_k`` bajo la hipotesis nula (sin
    dependencia) es ``~1/n_pairs_k`` (el resultado estandar para una
    correlacion de Pearson estimada de ``n_pairs_k`` pares
    aproximadamente independientes) -- por eso cada termino se pondera
    por su PROPIO ``n_pairs_k``, no por un ``T`` fijo compartido. Es un
    estadistico diagnostico (G5), nunca interpretado contra una tabla
    chi-cuadrado clasica: para las series con calibracion G2 propia
    (``r_1m`` en esta etapa), se interpreta contra la distribucion
    EMPIRICA de este mismo estadistico bajo Null 1
    (``g2_null1_portmanteau_summary``, la inferencia PRINCIPAL desde la
    3a revision -- nunca ``g2_combined_portmanteau_summary``, que es solo
    sensibilidad secundaria). Para series SIN calibracion G2 propia (p.
    ej. ``r_tilde`` en esta etapa) o para ``m`` mas alla de lo que Null 1
    calibra, `Q(m)` sigue siendo un numero valido pero se marca
    ``NOT_G2_CALIBRATED``/``DESCRIPTIVE_UNCALIBRATED`` por
    ``annotate_portmanteau_calibration`` -- ver esa funcion para el
    detalle de cuando cada estado aplica.

    Para ``m`` que exige rezagos SIN NINGUN par valido
    (``estimable=False`` en ``acf_tab``), el resultado es
    ``NOT_ESTIMABLE`` -- nunca se reporta un numero que solo cubrio una
    fraccion de ``m`` como si fuera "Q(m) completo" (seccion 2 de la
    tarea).

    Salida: ``DataFrame`` con ``m``, ``Q`` (``NaN`` si no estimable),
    ``estimable``, ``note``. Se añade ademas una fila con
    ``m=max_estimable_lag`` (el mayor rezago realmente calculable bajo
    las reglas de no-cruce de este conjunto de datos).
    """
    rho = acf_tab["rho"].to_numpy()
    n_pairs = acf_tab["n_pairs"].to_numpy()
    estimable = acf_tab["estimable"].to_numpy()
    lags = acf_tab["lag"].to_numpy()
    max_estimable_lag = int(lags[estimable].max()) if estimable.any() else 0

    def _q_up_to(m: int) -> float:
        terms = np.where(estimable[:m], n_pairs[:m].astype(float) * rho[:m] ** 2, 0.0)
        return float(np.sum(terms))

    rows = []
    for m in m_values:
        if m > len(rho) or m > max_estimable_lag:
            rows.append({
                "m": m, "Q": np.nan, "estimable": False,
                "note": f"NOT_ESTIMABLE -- excede el maximo rezago con pares validos bajo no-cruce (={max_estimable_lag})",
            })
        else:
            rows.append({"m": m, "Q": _q_up_to(m), "estimable": True, "note": ""})

    if max_estimable_lag not in m_values:
        rows.append({
            "m": max_estimable_lag, "Q": _q_up_to(max_estimable_lag), "estimable": True,
            "note": "maximo rezago REALMENTE estimable bajo las reglas de no-cruce (no un m predeclarado)",
        })
    return pd.DataFrame(rows)


def annotate_portmanteau_calibration(
    portmanteau_tab: pd.DataFrame, has_null1_calibration: bool, g2_calibrated_max_m: int = G2_NULL_MAX_LAG,
) -> pd.DataFrame:
    """Distingue, para cada fila de ``Q(m)``, si esta calibrada empiricamente por Null 1 (G2 PRINCIPAL) o no (seccion 7 de la 2a revision; cierre final -- seccion 2/3).

    ``has_null1_calibration`` es OBLIGATORIO (sin valor por defecto): cada
    llamador debe declarar EXPLICITAMENTE si la serie que esta anotando
    tiene una calibracion G2 Null-1 propia calculada -- no puede quedar
    implicito. Esto hace estructuralmente imposible que una serie SIN
    null propio termine marcada ``G2_CALIBRATED`` por accidente (p. ej.
    solo porque ``m<=60``, sin verificar que existe una calibracion real
    para esa serie).

    En esta etapa: ``r_1m`` tiene ``has_null1_calibration=True`` (Null 1,
    permutacion dentro de ``minute_of_day``, se construye sobre
    ``values_raw`` -- ver ``g2_null1_calibration_summary``/
    ``g2_null1_portmanteau_summary``, PRINCIPAL, 200 replicas). ``r_tilde``
    tiene ``has_null1_calibration=False`` -- no existe una calibracion G2
    Null-1 especifica para `r_tilde` en esta etapa (decision de cierre
    final: `r_tilde` es un diagnostico retrospectivo complementario;
    construir un segundo sistema de nulls solo para calibrarla ampliaria
    el alcance sin una razon metodologica que lo exija para TH16).

    G2 solo calibra empiricamente hasta ``G2_NULL_MAX_LAG`` (=60) --
    calibrar mas alla exigiria recalcular ``compute_acf`` sobre cada una
    de las 200 replicas de Null 1 hasta ese ``m`` mayor, muy por encima
    del presupuesto ya benchmarkeado (seccion 6 del informe). Por eso
    ``Q(1.378)`` (el maximo rezago realmente estimable de `r_1m`) se
    persiste como estadistico DESCRIPTIVO -- informa la magnitud
    observada, pero NUNCA se interpreta contra la distribucion empirica
    de Null 1 (que no lo cubre).

    Se distinguen cuatro estados: ``G2_CALIBRATED`` (``has_null1_calibration``
    y ``m<=g2_calibrated_max_m``, estimable), ``DESCRIPTIVE_UNCALIBRATED``
    (``has_null1_calibration`` pero ``m`` excede lo que Null 1 calibra, o
    estimable pero sin evaluar contra ningun null), ``NOT_G2_CALIBRATED``
    (``not has_null1_calibration``, estimable -- la serie simplemente no
    tiene calibracion G2 propia), ``NOT_ESTIMABLE`` (heredado de
    ``compute_portmanteau_q``).
    """
    out = portmanteau_tab.copy()

    def _status(row: pd.Series) -> str:
        if not bool(row["estimable"]):
            return "NOT_ESTIMABLE"
        if not has_null1_calibration:
            return "NOT_G2_CALIBRATED"
        if row["m"] <= g2_calibrated_max_m:
            return "G2_CALIBRATED"
        return "DESCRIPTIVE_UNCALIBRATED"

    out["calibration_status"] = out.apply(_status, axis=1)
    return out


# ---------------------------------------------------------------------------
# Bootstrap de bloques por jornada -- incertidumbre de rho_k/beta_k (G5)
# ---------------------------------------------------------------------------

def bootstrap_rho(
    values: np.ndarray, block_ids: np.ndarray, trading_date: np.ndarray,
    lags: tuple[int, ...] = BOOTSTRAP_LAGS, n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap de bloques de jornada para ``rho_k``/``beta_k`` -- clave COMPUESTA (block_id, ranura de remuestreo).

    Un dia remuestreado con reemplazo puede salir sorteado dos veces en
    la MISMA repeticion. Sin una clave compuesta, el final de una copia y
    el inicio de la otra (mismo ``block_id`` real) se contarian como un
    par valido -- un vecino fabricado. La clave compuesta
    (``block_id * (n_dates+1) + ranura``) impide que dos copias
    DISTINTAS del mismo dia se fusionen entre si, incluso si son
    identicas en contenido.

    CORRECCION METODOLOGICA (2a revision): igual que ``compute_acf``,
    cada rezago ``k`` de cada replica centra el lado ``t`` y el lado
    ``t-k`` por SU PROPIA media sobre los pares validos de esa replica y
    ese rezago -- nunca por la media global de la replica completa.

    Salida: ``(rho_boot, beta_boot)``, cada uno ``(n_boot, len(lags))``.
    """
    unique_dates, date_to_rows = _day_block_index_map(trading_date)
    n_dates = len(unique_dates)
    rng = np.random.default_rng(seed)
    rho_out = np.full((n_boot, len(lags)), np.nan)
    beta_out = np.full((n_boot, len(lags)), np.nan)

    for b in range(n_boot):
        sampled_days = rng.integers(0, n_dates, size=n_dates)
        idx_parts = []
        slot_parts = []
        for slot, d in enumerate(sampled_days):
            rows = date_to_rows[d]
            idx_parts.append(rows)
            slot_parts.append(np.full(len(rows), slot, dtype=np.int64))
        idx = np.concatenate(idx_parts)
        slots = np.concatenate(slot_parts)

        vals = values[idx]
        composite_block = block_ids[idx].astype(np.int64) * (n_dates + 1) + slots
        T = vals.size

        for li, k in enumerate(lags):
            if k >= T:
                continue
            same = composite_block[k:] == composite_block[:-k]
            if same.sum() < 2:
                continue
            x_t = vals[k:][same]
            x_tk = vals[:-k][same]
            mean_t = x_t.mean()
            mean_tk = x_tk.mean()
            e_t = x_t - mean_t
            e_tk = x_tk - mean_tk
            sum_prod = float(np.sum(e_t * e_tk))
            sum_t2 = float(np.sum(e_t * e_t))
            sum_tk2 = float(np.sum(e_tk * e_tk))
            denom = np.sqrt(sum_t2 * sum_tk2)
            if denom > 0:
                rho_out[b, li] = sum_prod / denom
            if sum_tk2 > 0:
                beta_out[b, li] = sum_prod / sum_tk2

    return rho_out, beta_out


# ---------------------------------------------------------------------------
# G2 -- calibracion del detector: dos nulls, pero SOLO UNO es la inferencia
# PRINCIPAL (3a revision -- "cierre final, correccion puntual de G2")
# ---------------------------------------------------------------------------
#
# NULL 1 (permutacion "por minuto") -- INFERENCIA PRINCIPAL. TDA-06
# certifico heterogeneidad deterministA fuerte por minute_of_day
# (magnitud/actividad). Una permutacion GLOBAL de los valores destruye
# tambien esa estructura -- un null "demasiado facil". En su lugar, se
# permutan los valores DENTRO de cada minute_of_day (a traves de los
# distintos dias que comparten ese mismo minuto) -- al ser una PERMUTACION
# de los valores REALES (no un resorteo sintetico), reproduce EXACTAMENTE
# la distribucion marginal de cada minuto (escala, discrecion, colas,
# curtosis -- sin necesidad de verificarlo, es la misma poblacion de
# valores, solo reordenada), y destruye la relacion serial entre un minuto
# y el siguiente (el valor de cada minuto ahora proviene de un dia
# aleatorio, tipicamente distinto del dia del minuto vecino). Por no
# depender de ningun supuesto de momentos que deba verificarse, es el null
# usado para TODA conclusion de calibracion de G2 (``g2_null1_calibration_summary``,
# ``g2_null1_portmanteau_summary``).
#
# NULL 2 (empirico reescalado por s(m)) -- SOLO SENSIBILIDAD FALLIDA /
# DIAGNOSTICO, EXCLUIDO de la inferencia principal. Se diseño para
# resamplear CON REEMPLAZO de la distribucion EMPIRICA de r_tilde
# (RETROSPECTIVO, TDA-06) reescalada por ``s(m)`` en la posicion de
# destino, con la INTENCION de preservar el perfil de escala por minuto Y
# la volatilidad/curtosis real. El diagnostico de momentos
# (``g2_synthetic_null_moment_check``, verificado sobre el conjunto de
# investigacion real) demostro que esa intencion NO se cumple del todo:
# el perfil de escala por minuto SI se preserva (correlacion 0,997), pero
# la varianza sale ~37% mayor y la curtosis en exceso sale ~3,6 veces
# mayor que la real -- el resorteo INDEPENDIENTE de un valor del pool y
# una posicion con su ``s(m)`` puede combinar una cola extrema del pool
# con un minuto de escala alta con mas frecuencia de la que esas dos cosas
# coocurren genuinamente en los datos reales. NO se afirma que esta mayor
# varianza/curtosis vuelva a Null 2 automaticamente "mas conservador" para
# la correlacion -- esa relacion no esta demostrada aqui, y afirmarla sin
# demostracion seria el mismo tipo de error que la afirmacion original
# refutada. Por eso Null 2 se excluye de la inferencia PRINCIPAL y se
# conserva unicamente como sensibilidad historica/secundaria, combinado
# con Null 1 (``g2_combined_calibration_summary``,
# ``g2_combined_portmanteau_summary`` -- renombradas en esta revision,
# desde ``g2_calibration_summary``/``g2_portmanteau_null_summary``, para
# que el nombre ya declare su rol secundario, nunca principal).

def g2_permutation_null_by_minute(
    values: np.ndarray, minute_of_day: np.ndarray, block_ids: np.ndarray, lags: tuple[int, ...],
    n_perm: int = DEFAULT_N_PERM, seed: int = PERMUTATION_SEED,
) -> np.ndarray:
    """Permuta los valores DENTRO de cada minute_of_day (preserva el perfil de reloj; destruye la dependencia serial)."""
    minute_of_day = np.asarray(minute_of_day)
    order = np.argsort(minute_of_day, kind="stable")
    sorted_minute = minute_of_day[order]
    boundaries = np.searchsorted(sorted_minute, np.arange(1441))
    groups = [order[boundaries[i]:boundaries[i + 1]] for i in range(1440) if boundaries[i + 1] > boundaries[i]]

    rng = np.random.default_rng(seed)
    max_lag = max(lags)
    out = np.full((n_perm, len(lags)), np.nan)
    for p in range(n_perm):
        permuted = values.copy()
        for g in groups:
            if len(g) > 1:
                permuted[g] = rng.permutation(values[g])
        acf_tab = compute_acf(permuted, block_ids, max_lag).set_index("lag")
        for li, k in enumerate(lags):
            if k in acf_tab.index:
                out[p, li] = acf_tab.loc[k, "rho"]
    return out


def g2_synthetic_empirical_null(
    r_tilde_pool: np.ndarray, s_m_values: np.ndarray, block_ids: np.ndarray, lags: tuple[int, ...],
    n_reps: int = DEFAULT_N_PERM, seed: int = SYNTHETIC_NULL_SEED,
) -> np.ndarray:
    """Ruido i.i.d. resampleado de la distribucion empirica de ``r_tilde``, reescalado por ``s(m)``.

    SOLO SENSIBILIDAD FALLIDA / DIAGNOSTICO -- EXCLUIDO de la inferencia
    PRINCIPAL de G2 (ver ``g2_null1_calibration_summary``). Se diseño con
    la intencion de preservar volatilidad/curtosis reales ademas del
    perfil de escala por minuto, pero ``g2_synthetic_null_moment_check``
    demostro sobre el conjunto de investigacion real que NO preserva
    volatilidad ni curtosis (varianza ~37% mayor, curtosis en exceso ~3,6
    veces mayor que la real) -- solo preserva el perfil de escala por
    minuto (correlacion 0,997). No se afirma que esto preserve
    volatilidad/curtosis; se afirma unicamente lo que esta funcion hace
    mecanicamente (resamplear y reescalar), sin presuponer que el
    resultado tenga las propiedades deseadas -- esas se VERIFICAN por
    separado, nunca se dan por hechas.
    """
    rng = np.random.default_rng(seed)
    n = len(s_m_values)
    max_lag = max(lags)
    out = np.full((n_reps, len(lags)), np.nan)
    for r in range(n_reps):
        draws = rng.choice(r_tilde_pool, size=n, replace=True)
        synthetic = draws * s_m_values
        acf_tab = compute_acf(synthetic, block_ids, max_lag).set_index("lag")
        for li, k in enumerate(lags):
            if k in acf_tab.index:
                out[r, li] = acf_tab.loc[k, "rho"]
    return out


def draw_synthetic_empirical_sample(r_tilde_pool: np.ndarray, s_m_values: np.ndarray, seed: int) -> np.ndarray:
    """Un UNICO sorteo del generador del null 2 (misma mecanica que ``g2_synthetic_empirical_null``, una sola replica) -- para el diagnostico de momentos, no para el null combinado de 200 replicas."""
    rng = np.random.default_rng(seed)
    n = len(s_m_values)
    draws = rng.choice(r_tilde_pool, size=n, replace=True)
    return draws * s_m_values


def g2_synthetic_null_moment_check(
    real_values: np.ndarray, synthetic_sample: np.ndarray, minute_of_day: np.ndarray, s_m_values: np.ndarray,
    variance_tolerance: float = 0.25, kurtosis_relative_tolerance: float = 0.5, scale_profile_min_correlation: float = 0.9,
) -> dict:
    """Verifica -- con criterios PREDECLARADOS, pequeños y reproducibles -- la afirmacion de que el null sintetico preserva volatilidad/curtosis/perfil de escala (seccion 6 de la 2a revision).

    La version anterior AFIRMABA esto sin comprobarlo con un test
    explicito. Se implementa la opcion (A) (verificacion real, preferida
    por la revision) en vez de solo suavizar el lenguaje: se compara UNA
    realizacion del sorteo sintetico contra la serie real en tres
    criterios independientes, cada uno con su propia tolerancia
    predeclarada (nunca ajustada mirando el resultado real):

    - varianza: razon ``var(synthetic)/var(real)`` dentro de
      ``1 ± variance_tolerance`` (25% por defecto -- una banda ancha
      porque la varianza de un retorno de 1 minuto es ya de por si
      ruidosa entre realizaciones distintas del mismo generador).
    - curtosis en exceso (Fisher, ``m4/m2^2 - 3``): diferencia relativa
      respecto de la curtosis real dentro de ``kurtosis_relative_tolerance``
      (50% por defecto -- la curtosis muestral de datos de cola pesada es
      un estadistico de alta varianza; un umbral estrecho generaria
      falsos rechazos constantes sin que el generador este mal
      especificado).
    - perfil de escala por minuto: correlacion entre una escala ROBUSTA
      (MAD reescalada, ``mediana(|x-mediana(x)|) x 1,4826`` -- consistente
      con la desviacion estandar bajo normalidad, pero mucho menos
      sensible a un puñado de extremos) del sintetico agrupado por
      ``minute_of_day`` y el ``s_m_values`` objetivo en esa misma
      agrupacion, debe superar ``scale_profile_min_correlation`` (0,9 por
      defecto). Se usa MAD y no la desviacion estandar cruda porque, con
      datos de cola tan pesada como los de esta etapa (curtosis en exceso
      de decenas, ver TDA-07), la varianza MUESTRAL por minuto es
      dominada por un numero pequeño de observaciones extremas y es un
      estimador de escala demasiado ruidoso con las pocas centenas de
      observaciones por minuto disponibles -- verificado empiricamente:
      con desviacion estandar cruda, la correlacion del perfil de escala
      cae a ~0,76 incluso para el generador CORRECTO por puro ruido
      muestral, mientras que con MAD sube a ~0,91 para el generador
      correcto y baja a ~-0,11 para el gaussiano ingenuo (sin perfil de
      escala real que recuperar).

    Un null mal especificado (p. ej. el gaussiano i.i.d. con sigma global
    de la version original de TDA-08, sin perfil de escala ni cola
    pesada) falla el criterio de curtosis y el de perfil de escala --
    verificado en
    ``test_g2_synthetic_null_moment_check_passes_for_correct_construction_and_fails_for_naive_gaussian``.
    """
    real = np.asarray(real_values, dtype=float)
    real = real[~np.isnan(real)]
    synth = np.asarray(synthetic_sample, dtype=float)
    synth = synth[~np.isnan(synth)]

    def _excess_kurtosis(a: np.ndarray) -> float:
        c = a - a.mean()
        m2 = float(np.mean(c ** 2))
        m4 = float(np.mean(c ** 4))
        return (m4 / (m2 ** 2) - 3.0) if m2 > 0 else float("nan")

    var_real = float(np.var(real))
    var_synth = float(np.var(synth))
    var_ratio = (var_synth / var_real) if var_real > 0 else float("nan")
    var_ok = bool(abs(var_ratio - 1.0) <= variance_tolerance) if not np.isnan(var_ratio) else False

    kurt_real = _excess_kurtosis(real)
    kurt_synth = _excess_kurtosis(synth)
    kurt_rel_diff = (abs(kurt_synth - kurt_real) / abs(kurt_real)) if kurt_real not in (0.0, None) and not np.isnan(kurt_real) else float("nan")
    kurt_ok = bool(kurt_rel_diff <= kurtosis_relative_tolerance) if not np.isnan(kurt_rel_diff) else False

    def _mad_scale(a: pd.Series) -> float:
        med = np.median(a)
        return float(np.median(np.abs(a - med)) * 1.4826)

    minute_of_day = np.asarray(minute_of_day)
    profile_df = pd.DataFrame({"minute_of_day": minute_of_day, "synth": synthetic_sample, "target_s_m": s_m_values})
    synth_profile = profile_df.groupby("minute_of_day")["synth"].apply(_mad_scale)
    target_profile = profile_df.groupby("minute_of_day")["target_s_m"].mean()
    common = synth_profile.index.intersection(target_profile.index)
    if len(common) > 2 and synth_profile.loc[common].std() > 0 and target_profile.loc[common].std() > 0:
        profile_corr = float(np.corrcoef(synth_profile.loc[common], target_profile.loc[common])[0, 1])
    else:
        profile_corr = float("nan")
    profile_ok = bool(profile_corr >= scale_profile_min_correlation) if not np.isnan(profile_corr) else False

    return {
        "var_real": var_real, "var_synth": var_synth, "var_ratio": var_ratio, "var_within_tolerance": var_ok,
        "kurt_real": kurt_real, "kurt_synth": kurt_synth, "kurt_rel_diff": kurt_rel_diff, "kurt_within_tolerance": kurt_ok,
        "scale_profile_correlation": profile_corr, "scale_profile_within_tolerance": profile_ok,
    }


def g2_null1_calibration_summary(real_rho: dict[int, float], perm_null: np.ndarray, lags: tuple[int, ...]) -> pd.DataFrame:
    """Calibracion G2 PRINCIPAL -- percentil empirico de ``|rho_k|`` real dentro de Null 1 EXCLUSIVAMENTE (3a revision).

    Null 1 (permutacion dentro de ``minute_of_day``) es la unica base de
    la inferencia G2: al ser una permutacion de los valores REALES,
    reproduce la distribucion marginal real de cada minuto sin depender
    de ningun supuesto de momentos que deba verificarse -- a diferencia
    de Null 2, que demostradamente NO preserva varianza/curtosis reales
    (``g2_synthetic_null_moment_check``). Ver ``g2_combined_calibration_summary``
    para la sensibilidad historica/secundaria que combina ambos nulls.
    """
    col_abs = np.abs(perm_null)
    rows = []
    for li, k in enumerate(lags):
        real = abs(real_rho.get(k, np.nan))
        col = col_abs[:, li]
        col = col[~np.isnan(col)]
        pct = float((col < real).mean() * 100.0) if col.size and not np.isnan(real) else float("nan")
        rows.append({
            "lag": k, "rho_real_abs": real,
            "n_null1_replicas": int(col.size),
            "null1_p50": float(np.nanmedian(col)) if col.size else np.nan,
            "null1_p975": float(np.nanpercentile(col, NULL_PERCENTILE)) if col.size else np.nan,
            "percentile_of_real_in_null1": pct,
            "exceeds_calibration_threshold": bool(pct >= NULL_PERCENTILE) if not np.isnan(pct) else False,
        })
    return pd.DataFrame(rows)


def g2_null1_portmanteau_summary(
    perm_null: np.ndarray, real_n_pairs: np.ndarray, m_values: tuple[int, ...], real_q: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Portmanteau G2 PRINCIPAL -- ``Q(m)`` real vs distribucion empirica de ``Q(m)`` bajo Null 1 EXCLUSIVAMENTE (3a revision).

    Reutiliza ``n_pairs`` REAL (no depende de los valores permutados). Si
    se pasa ``real_q`` (``{m: Q_real}``), añade tambien el percentil del
    ``Q(m)`` real dentro de la distribucion nula y si supera P97,5 --
    mismo criterio que ``g2_null1_calibration_summary`` para `rho_k`.
    """
    rows = []
    for m in m_values:
        if m > perm_null.shape[1]:
            continue
        terms = perm_null[:, :m] ** 2 * real_n_pairs[:m]
        q_dist = np.nansum(terms, axis=1)
        row = {
            "m": m, "n_null1_replicas": int(perm_null.shape[0]),
            "Q_null1_p50": float(np.median(q_dist)), "Q_null1_p975": float(np.percentile(q_dist, NULL_PERCENTILE)),
        }
        if real_q is not None and m in real_q:
            real_val = real_q[m]
            pct = float((q_dist < real_val).mean() * 100.0)
            row["Q_real"] = real_val
            row["percentile_of_real_in_null1"] = pct
            row["exceeds_calibration_threshold"] = bool(pct >= NULL_PERCENTILE)
        rows.append(row)
    return pd.DataFrame(rows)


def g2_combined_calibration_summary(real_rho: dict[int, float], perm_null: np.ndarray, synth_null: np.ndarray, lags: tuple[int, ...]) -> pd.DataFrame:
    """SENSIBILIDAD HISTORICA/SECUNDARIA -- percentil empirico de ``|rho_k|`` real dentro de AMBOS nulls COMBINADOS.

    Renombrada en la 3a revision desde ``g2_calibration_summary`` (el
    nombre anterior no distinguia rol principal de secundario). NUNCA es
    la inferencia principal desde que ``g2_synthetic_null_moment_check``
    demostro que Null 2 no preserva varianza/curtosis reales -- se
    conserva unicamente como sensibilidad historica, claramente separada
    de ``g2_null1_calibration_summary`` (la inferencia principal) tanto en
    el nombre de la funcion como en el artefacto persistido.
    ``n_perm`` cada uno -- ``2*n_perm`` réplicas totales combinadas, nunca
    ``n_perm**2``.
    """
    combined = np.concatenate([np.abs(perm_null), np.abs(synth_null)], axis=0)
    rows = []
    for li, k in enumerate(lags):
        real = abs(real_rho.get(k, np.nan))
        col = combined[:, li]
        col = col[~np.isnan(col)]
        pct = float((col < real).mean() * 100.0) if col.size and not np.isnan(real) else float("nan")
        rows.append({
            "lag": k, "rho_real_abs": real,
            "n_null_replicas_combinadas": int(col.size),
            "null_p50": float(np.nanmedian(col)) if col.size else np.nan,
            "null_p975": float(np.nanpercentile(col, NULL_PERCENTILE)) if col.size else np.nan,
            "percentile_of_real_in_null": pct,
            "exceeds_calibration_threshold": bool(pct >= NULL_PERCENTILE) if not np.isnan(pct) else False,
        })
    return pd.DataFrame(rows)


def g2_combined_portmanteau_summary(
    perm_null: np.ndarray, synth_null: np.ndarray, real_n_pairs: np.ndarray, m_values: tuple[int, ...],
) -> pd.DataFrame:
    """SENSIBILIDAD HISTORICA/SECUNDARIA -- ``Q(m)`` real vs distribucion empirica combinada de ``Q(m)`` bajo ambos nulls.

    Renombrada en la 3a revision desde ``g2_portmanteau_null_summary``.
    NUNCA es la inferencia principal -- ver ``g2_null1_portmanteau_summary``.
    """
    combined = np.concatenate([perm_null, synth_null], axis=0)
    rows = []
    for m in m_values:
        if m > combined.shape[1]:
            continue
        terms = combined[:, :m] ** 2 * real_n_pairs[:m]
        q_dist = np.nansum(terms, axis=1)
        rows.append({
            "m": m, "n_null_replicas_combinadas": int(combined.shape[0]),
            "Q_null_p50": float(np.median(q_dist)), "Q_null_p975": float(np.percentile(q_dist, NULL_PERCENTILE)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Traduccion a ticks -- rho_1 (correlacion) Y beta_1 (pendiente) SEPARADOS
# ---------------------------------------------------------------------------

def translate_dependence_to_ticks(rho_1: float, beta_1: float, sigma_r: float, close_values: np.ndarray, tick_size: float) -> dict:
    """Traduccion a ticks -- usa ``beta_1`` (la pendiente OLS real), no ``rho_1``, para el movimiento lineal implicito.

    ``rho_1`` (correlacion, adimensional) y ``beta_1`` (pendiente de
    regresion) son cantidades DISTINTAS -- NUNCA se presupone que
    coinciden. El "movimiento lineal implicito" (la respuesta esperada
    de ``r_t`` ante un movimiento previo de 1 sigma en ``r_{t-1}``) es,
    por definicion de regresion lineal, ``beta_1 * sigma_r`` -- exacto,
    no una aproximacion bajo estacionariedad. Se reporta tambien
    ``rho_1`` y la razon ``beta_1/rho_1`` (que muestra CUANTO se
    parecen, en vez de presuponerlo).
    """
    tick_return_repr = compute_tick_return_repr(close_values, tick_size)
    close_repr = float(np.median(close_values))
    implied_move_return = beta_1 * sigma_r
    return {
        "rho_1": rho_1,
        "beta_1": beta_1,
        "beta_over_rho_ratio": (beta_1 / rho_1) if rho_1 not in (0, None) and not np.isnan(rho_1) else float("nan"),
        "escala_usada_sigma_r": sigma_r,
        "movimiento_previo_referencia": sigma_r,
        "movimiento_lineal_implicito_return": implied_move_return,
        "close_representativo": close_repr,
        "conversion_puntos": implied_move_return * close_repr,
        "tick_return_representativo": tick_return_repr,
        "conversion_ticks": implied_move_return / tick_return_repr if tick_return_repr != 0 else float("nan"),
        "supuestos": (
            "movimiento lineal implicito = beta_1 * sigma_r, EXACTO por definicion de pendiente OLS "
            "(no presupone Var(r_t)=Var(r_t-1); rho_1 se reporta aparte, no se usa para esta conversion); "
            "aproximacion lineal r~R de primer orden para la conversion a puntos "
            "(TH07/TDA-04: despreciable a escala de 1 minuto); "
            "close_representativo = mediana del grupo analizado (misma convencion que TDA-05/07)."
        ),
    }


# ---------------------------------------------------------------------------
# Deciles de volumen (TH17) -- convencion GLOBAL predeclarada
# ---------------------------------------------------------------------------

def assign_volume_decile(volume: pd.Series, n_deciles: int = N_VOLUME_DECILES) -> pd.Series:
    """Decil GLOBAL (0=menor actividad .. n_deciles-1=mayor actividad) del volumen de la barra ``t``.

    Se usa "volumen"/"actividad" deliberadamente, nunca "liquidez": con
    OHLCV Last solo se observa volumen/actividad negociada, no spread ni
    profundidad -- "liquidez" seria un proxy no declarado.
    """
    return pd.qcut(volume, n_deciles, labels=False, duplicates="drop")


# ---------------------------------------------------------------------------
# ACF agrupada -- motor CORREGIDO (pares construidos sobre la TOPOLOGIA
# COMPLETA, agrupados despues por la etiqueta de la fila t)
# ---------------------------------------------------------------------------

def acf_by_group(
    values: np.ndarray, block_ids: np.ndarray, group_labels: np.ndarray, lags: tuple[int, ...],
    min_n_pairs: int = MIN_N_FOR_SUBGROUP_ACF,
) -> pd.DataFrame:
    """``rho_k``/``beta_k`` por grupo (año, segmento, decil de volumen, mes...) -- CORREGIDO dos veces.

    CORRECCION 1 (seccion 3 de la 1a revision, conservada): los pares
    validos del rezago ``k`` se construyen UNA VEZ sobre la topologia
    COMPLETA; cada par ``(t, t-k)`` se etiqueta con ``group_labels[t]``
    (la fila MAS RECIENTE); se agrupan los PARES por esa etiqueta. Nunca
    se exige que ``t-k`` pertenezca al mismo grupo que ``t``.

    CORRECCION 2 (seccion 3 de la 2a revision -- "centrado del
    estimador"): la version intermedia calculaba ``e = x - mean(x)`` con
    la media GLOBAL de TODA la poblacion antes de separar los pares por
    grupo. Eso es exactamente el mismo error, en otra forma, que el
    corregido en ``compute_acf``: si dos grupos tienen medias
    verdaderamente distintas (p. ej. un año alcista vs. uno bajista, o un
    decil de volumen con sesgo horario), centrar por una media COMUN
    fabrica covarianza artificial ``e_t*e_tk`` proporcional al PRODUCTO
    de los desvios de cada grupo respecto de esa media comun, incluso si
    dentro de cada grupo ``x`` e ``y`` son independientes. Se corrige
    calculando, DENTRO DE CADA GRUPO, sus propias
    ``mean_t_g``/``mean_tk_g`` (sobre los pares de ESE grupo unicamente)
    antes de construir ``e_t``/``e_tk`` -- exactamente la pregunta
    ``Corr(r_t, r_{t-1} | group_t=g)`` con las medias/varianzas/
    covarianza estimadas sobre la muestra condicionada, nunca sobre la
    poblacion completa. Verificado con un test adversarial dedicado
    (``test_acf_by_group_does_not_inject_spurious_dependence_from_between_group_mean_differences``):
    dos grupos con medias muy distintas, independencia genuina DENTRO de
    cada grupo -- el estimador corregido da ``rho≈0`` en ambos; la
    version centrada por media global no.

    Salida: ``DataFrame`` LARGO con ``group``, ``lag``, ``n_pairs``,
    ``rho``, ``beta``, ``insufficient_sample``.
    """
    x_all = np.asarray(values, dtype=float)
    T = x_all.size
    group_labels = np.asarray(group_labels)

    rows = []
    for k in lags:
        if k >= T:
            continue
        same_block = block_ids[k:] == block_ids[:-k]
        idx_t = np.arange(k, T)[same_block]
        if idx_t.size == 0:
            continue
        x_t = x_all[idx_t]
        x_tk = x_all[idx_t - k]
        grp = group_labels[idx_t]

        pairs = pd.DataFrame({"group": grp, "x_t": x_t, "x_tk": x_tk})
        grouped = pairs.groupby("group", observed=True)
        mean_t_g = grouped["x_t"].transform("mean").to_numpy()   # media de x_t DENTRO de cada grupo
        mean_tk_g = grouped["x_tk"].transform("mean").to_numpy()  # media de x_tk DENTRO de cada grupo
        e_t = pairs["x_t"].to_numpy() - mean_t_g
        e_tk = pairs["x_tk"].to_numpy() - mean_tk_g

        agg_input = pd.DataFrame({"group": grp, "prod": e_t * e_tk, "et2": e_t * e_t, "etk2": e_tk * e_tk})
        agg = agg_input.groupby("group", observed=True).agg(
            n_pairs=("prod", "size"), sum_prod=("prod", "sum"), sum_et2=("et2", "sum"), sum_etk2=("etk2", "sum"),
        )
        for g, row in agg.iterrows():
            n = int(row["n_pairs"])
            if n < min_n_pairs:
                rows.append({"group": g, "lag": k, "n_pairs": n, "rho": np.nan, "beta": np.nan, "insufficient_sample": True})
                continue
            denom = np.sqrt(row["sum_et2"] * row["sum_etk2"])
            rho = (row["sum_prod"] / denom) if denom > 0 else np.nan
            beta = (row["sum_prod"] / row["sum_etk2"]) if row["sum_etk2"] > 0 else np.nan
            rows.append({"group": g, "lag": k, "n_pairs": n, "rho": rho, "beta": beta, "insufficient_sample": False})
    return pd.DataFrame(rows)


def acf_by_group_strict_same_block(
    df: pd.DataFrame, value_col: str, group_col: str, lags: tuple[int, ...], min_n_pairs: int = MIN_N_FOR_SUBGROUP_ACF,
) -> pd.DataFrame:
    """SENSIBILIDAD SECUNDARIA (nunca el analisis principal): exige ADEMAS que ``t-1`` este en el MISMO grupo que ``t``.

    Filtra las filas del grupo y recalcula bloques SOLO sobre ese
    subconjunto -- la convencion ORIGINAL (mas restrictiva: equivale a
    ``group_t == group_{t-1}``). Se conserva unicamente para el decil de
    volumen, como sensibilidad, tal como autoriza explicitamente la
    correccion de esta etapa -- nunca reemplaza a ``acf_by_group``.
    """
    rows = []
    for g, sub in df.groupby(group_col, observed=True):
        sub = sub.sort_values("timestamp")
        n = len(sub)
        if n < min_n_pairs:
            for k in lags:
                rows.append({"group": g, "lag": k, "n_pairs": 0, "rho": np.nan, "beta": np.nan, "insufficient_sample": True})
            continue
        block_ids = compute_block_ids(sub["timestamp"], sub["trading_date"], 60.0)
        values = sub[value_col].to_numpy(dtype=float)
        acf_tab = compute_acf(values, block_ids, max(lags)).set_index("lag")
        for k in lags:
            if k in acf_tab.index:
                r = acf_tab.loc[k]
                rows.append({
                    "group": g, "lag": k, "n_pairs": int(r["n_pairs"]),
                    "rho": float(r["rho"]) if r["estimable"] else np.nan,
                    "beta": float(r["beta"]) if r["estimable"] else np.nan,
                    "insufficient_sample": not bool(r["estimable"]) or int(r["n_pairs"]) < min_n_pairs,
                })
    return pd.DataFrame(rows)


def rolling_rho1_by_month(r1m_pop: pd.DataFrame) -> pd.DataFrame:
    """``rho_1`` por mes calendario de ``trading_date`` -- ventana rodante mas simple y transparente (TH18), motor CORREGIDO."""
    td = pd.to_datetime(r1m_pop["trading_date"])
    year_month = td.dt.year.astype(str) + "-" + td.dt.month.astype(str).str.zfill(2)
    block_ids = compute_block_ids(r1m_pop["timestamp"], r1m_pop["trading_date"], 60.0)
    values = r1m_pop["r_1m"].to_numpy(dtype=float)
    table = acf_by_group(values, block_ids, year_month.to_numpy(), lags=(1,))
    return table.rename(columns={"group": "year_month"}).sort_values("year_month").reset_index(drop=True)


# ---------------------------------------------------------------------------
# TH17 -- 1 -> 5 -> 10 minutos, SIEMPRE con ventanas NO SOLAPADAS (TH10)
# ---------------------------------------------------------------------------

def compute_multi_frequency_rho1(
    variables: pd.DataFrame, validity: pd.DataFrame, n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_BOOTSTRAP_SEED,
    horizons: tuple[int, ...] = (1, 5, 10),
) -> pd.DataFrame:
    """``rho_1``/``beta_1`` a 1/5/10 minutos -- refutacion TH17, punto (i). Reutiliza TH10 (``build_horizon_returns``/``non_overlap_mask``) sin modificarlo."""
    hdf = build_horizon_returns(variables, validity, horizons=horizons)
    rows = []
    for h in horizons:
        col = f"r_{h}"
        if h == 1:
            sub = hdf.dropna(subset=[col]).copy()
            expected_delta = 60.0
        else:
            mask = non_overlap_mask(hdf["run_length"].to_numpy(), h)
            sub = hdf.loc[mask].dropna(subset=[col]).copy()
            expected_delta = float(h * 60)
        sub = sub.sort_values("timestamp")
        n = len(sub)
        if n < MIN_N_FOR_SUBGROUP_ACF:
            rows.append({"h_minutes": h, "n": n, "n_pairs_lag1": 0, "rho_1": np.nan, "beta_1": np.nan, "rho_ci_lo": np.nan, "rho_ci_hi": np.nan})
            continue
        block_ids = compute_block_ids(sub["timestamp"], sub["trading_date"], expected_delta)
        values = sub[col].to_numpy(dtype=float)
        acf1 = compute_acf(values, block_ids, max_lag=1)
        rho1 = float(acf1.loc[0, "rho"])
        beta1 = float(acf1.loc[0, "beta"])
        n_pairs = int(acf1.loc[0, "n_pairs"])
        rho_boot, _ = bootstrap_rho(values, block_ids, sub["trading_date"].to_numpy(), lags=(1,), n_boot=n_boot, seed=seed)
        finite = rho_boot[:, 0][~np.isnan(rho_boot[:, 0])]
        ci_lo, ci_hi = (np.percentile(finite, [2.5, 97.5]) if finite.size else (np.nan, np.nan))
        rows.append({"h_minutes": h, "n": n, "n_pairs_lag1": n_pairs, "rho_1": rho1, "beta_1": beta1, "rho_ci_lo": float(ci_lo), "rho_ci_hi": float(ci_hi)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ventanas heredadas de TDA-06 -- solo estas dos
# ---------------------------------------------------------------------------

def _day_block_bootstrap_mean(values: np.ndarray, trading_date: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    unique_dates, date_to_rows = _day_block_index_map(trading_date)
    n_dates = len(unique_dates)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.integers(0, n_dates, size=n_dates)
        idx = np.concatenate([date_to_rows[d] for d in sampled])
        out[b] = values[idx].mean()
    return out


def analyze_window(
    df: pd.DataFrame, window: tuple[int, int], tick_size: float, n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict:
    """Contraste de la ventana ``window`` -- signo/magnitud POR BARRA, estabilidad por año, minutos adyacentes, lag-1, ticks."""
    lo, hi = window
    sub = df.loc[(df["minute_of_day"] >= lo) & (df["minute_of_day"] <= hi)].sort_values("timestamp")
    values = sub["r_1m"].to_numpy(dtype=float)
    n = len(sub)
    n_minutes = hi - lo + 1
    mean = float(values.mean()) if n else float("nan")
    boot_mean = _day_block_bootstrap_mean(values, sub["trading_date"].to_numpy(), n_boot, seed) if n >= MIN_N_FOR_SUBGROUP_ACF else np.array([np.nan])
    mean_ci = (float(np.nanpercentile(boot_mean, 2.5)), float(np.nanpercentile(boot_mean, 97.5)))

    tick_return_repr = compute_tick_return_repr(sub["close"].to_numpy(dtype=float), tick_size) if n else float("nan")
    mean_ticks = (mean / tick_return_repr) if n and tick_return_repr else float("nan")
    mean_ci_ticks = (
        (mean_ci[0] / tick_return_repr, mean_ci[1] / tick_return_repr) if n and tick_return_repr else (float("nan"), float("nan"))
    )

    if n >= MIN_N_FOR_SUBGROUP_ACF:
        block_ids = compute_block_ids(sub["timestamp"], sub["trading_date"], 60.0)
        acf1 = compute_acf(values, block_ids, max_lag=1)
        rho_1 = float(acf1.loc[0, "rho"]) if bool(acf1.loc[0, "estimable"]) else float("nan")
        n_pairs_1 = int(acf1.loc[0, "n_pairs"])
    else:
        rho_1, n_pairs_1 = float("nan"), 0

    by_year = sub.groupby("year_ny")["r_1m"].agg(["mean", "count"]) if n else pd.DataFrame()
    adjacent = df.loc[(df["minute_of_day"] >= lo - 2) & (df["minute_of_day"] <= hi + 2)].groupby("minute_of_day")["r_1m"].agg(["mean", "count"])

    return {
        "window": window, "n": n, "n_minutes": n_minutes,
        "mean_per_bar": mean, "mean_per_bar_ci_lo": mean_ci[0], "mean_per_bar_ci_hi": mean_ci[1],
        "mean_per_bar_ticks": mean_ticks, "mean_per_bar_ci_lo_ticks": mean_ci_ticks[0], "mean_per_bar_ci_hi_ticks": mean_ci_ticks[1],
        "rho_1": rho_1, "n_pairs_rho_1": n_pairs_1,
        "by_year": by_year, "adjacent_minutes": adjacent,
    }


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------

@dataclass
class TDA08Result:
    r1m_population: pd.DataFrame
    r_tilde_population: pd.DataFrame
    segment_labels: list[str]
    acf_raw: pd.DataFrame
    acf_adjusted: pd.DataFrame
    bartlett_se_raw: np.ndarray
    bartlett_se_adjusted: np.ndarray
    pacf_raw: np.ndarray
    pacf_adjusted: np.ndarray
    portmanteau_raw: pd.DataFrame
    portmanteau_adjusted: pd.DataFrame
    bootstrap_table_raw: pd.DataFrame
    bootstrap_table_adjusted: pd.DataFrame
    g2_null1_calibration_raw: pd.DataFrame
    g2_null1_portmanteau_raw: pd.DataFrame
    g2_calibration_combined_secondary: pd.DataFrame
    g2_portmanteau_combined_secondary: pd.DataFrame
    g2_synthetic_null_moment_check: dict
    ticks_translation: dict
    multi_frequency: pd.DataFrame
    acf_by_group_long: pd.DataFrame
    volume_decile_sensitivity: pd.DataFrame
    window_open: dict
    window_close: dict
    max_estimable_lag_raw: int


def run_tda08_analysis(config: SnapshotConfig) -> TDA08Result:
    """Orquesta TDA-08 completo (version corregida): ACF pairwise centrada localmente, G2 con Null 1 como inferencia PRINCIPAL (Null 2 solo sensibilidad secundaria, no supero el diagnostico de momentos), decil/segmento/año condicionados y centrados correctamente por grupo, rezagos largos NOT_ESTIMABLE explicito."""
    validate_research_holdout_disjoint(config)

    variables, validity = load_tda04_inputs(config)
    last_timestamps = variables.groupby("source_file")["timestamp"].max().to_dict()
    validate_last_timestamps_before_boundary(config, last_timestamps)
    verify_timestamp_alignment(variables, validity)

    n_boot = config.tda08_n_boot
    n_perm = config.tda08_n_perm
    tick_size = config.tick_size

    cutoffs = load_segmentation_cutoffs(config)
    r1m_pop, segment_labels = build_r1m_population(variables, validity, cutoffs)
    r1m_pop = r1m_pop.sort_values("timestamp").reset_index(drop=True)
    r1m_pop["volume_decile"] = assign_volume_decile(r1m_pop["volume"])

    r_tilde = load_r_tilde(config)
    verify_r_tilde_invariants(r_tilde, variables)
    r_tilde_pop, _ = build_r_tilde_population(r_tilde, cutoffs)
    r_tilde_pop = r_tilde_pop.sort_values("timestamp").reset_index(drop=True)

    # --- TH16: ACF/PACF/Bartlett/portmanteau, crudo y ajustado -----------
    block_ids_raw = compute_block_ids(r1m_pop["timestamp"], r1m_pop["trading_date"], 60.0)
    block_ids_adj = compute_block_ids(r_tilde_pop["timestamp"], r_tilde_pop["trading_date"], 60.0)

    values_raw = r1m_pop["r_1m"].to_numpy(dtype=float)
    values_adj = r_tilde_pop["r_tilde"].to_numpy(dtype=float)

    acf_raw = compute_acf(values_raw, block_ids_raw, MAX_LAG_REQUESTED)
    acf_adjusted = compute_acf(values_adj, block_ids_adj, MAX_LAG_REQUESTED)
    max_estimable_lag_raw = int(acf_raw.loc[acf_raw["estimable"], "lag"].max())

    T_raw = values_raw.size
    T_adj = values_adj.size
    bartlett_se_raw = bartlett_se(acf_raw["rho"].to_numpy(), T_raw)
    bartlett_se_adjusted = bartlett_se(acf_adjusted["rho"].to_numpy(), T_adj)

    pacf_raw = durbin_levinson_pacf(acf_raw["rho"].to_numpy()[:PACF_MAX_LAG], PACF_MAX_LAG)
    pacf_adjusted = durbin_levinson_pacf(acf_adjusted["rho"].to_numpy()[:PACF_MAX_LAG], PACF_MAX_LAG)

    # r_1m: Null 1 se construye sobre esta serie -- SI tiene calibracion G2 propia.
    portmanteau_raw = annotate_portmanteau_calibration(compute_portmanteau_q(acf_raw, LJUNG_BOX_M), has_null1_calibration=True)
    # r_tilde: NO existe una calibracion G2 Null-1 propia en esta etapa (decision de cierre final, ver informe seccion 2) -- nunca G2_CALIBRATED.
    portmanteau_adjusted = annotate_portmanteau_calibration(compute_portmanteau_q(acf_adjusted, LJUNG_BOX_M), has_null1_calibration=False)

    # --- Bootstrap de incertidumbre (rezagos clave) -----------------------
    rho_boot_raw, beta_boot_raw = bootstrap_rho(values_raw, block_ids_raw, r1m_pop["trading_date"].to_numpy(), BOOTSTRAP_LAGS, n_boot, DEFAULT_BOOTSTRAP_SEED)
    rho_boot_adj, beta_boot_adj = bootstrap_rho(values_adj, block_ids_adj, r_tilde_pop["trading_date"].to_numpy(), BOOTSTRAP_LAGS, n_boot, DEFAULT_BOOTSTRAP_SEED)

    def _bootstrap_table(acf_tab, rho_boot, beta_boot):
        acf_idx = acf_tab.set_index("lag")
        rows = []
        for i, k in enumerate(BOOTSTRAP_LAGS):
            rho_fin = rho_boot[:, i][~np.isnan(rho_boot[:, i])]
            beta_fin = beta_boot[:, i][~np.isnan(beta_boot[:, i])]
            rows.append({
                "lag": k,
                "rho_point": float(acf_idx.loc[k, "rho"]) if k in acf_idx.index else np.nan,
                "beta_point": float(acf_idx.loc[k, "beta"]) if k in acf_idx.index else np.nan,
                "rho_ci_lo": float(np.percentile(rho_fin, 2.5)) if rho_fin.size else np.nan,
                "rho_ci_hi": float(np.percentile(rho_fin, 97.5)) if rho_fin.size else np.nan,
                "beta_ci_lo": float(np.percentile(beta_fin, 2.5)) if beta_fin.size else np.nan,
                "beta_ci_hi": float(np.percentile(beta_fin, 97.5)) if beta_fin.size else np.nan,
            })
        return pd.DataFrame(rows)

    bootstrap_table_raw = _bootstrap_table(acf_raw, rho_boot_raw, beta_boot_raw)
    bootstrap_table_adjusted = _bootstrap_table(acf_adjusted, rho_boot_adj, beta_boot_adj)

    # --- G2: dos nulls rediseñados, sobre r_1m crudo -----------------------
    null_lags = tuple(range(1, G2_NULL_MAX_LAG + 1))
    minute_of_day_raw = r1m_pop["minute_of_day"].to_numpy()
    perm_null = g2_permutation_null_by_minute(values_raw, minute_of_day_raw, block_ids_raw, null_lags, n_perm, PERMUTATION_SEED)

    s_m_values = r_tilde_pop["s_m"].to_numpy(dtype=float)
    r_tilde_pool = values_adj[~np.isnan(values_adj)]
    # s_m_values esta indexado por la poblacion de r_tilde (misma poblacion que r1m_pop, r_1m_valid=True);
    # para el null 2 se reescala por s_m EN LA MISMA POSICION que r1m_pop (topologia identica).
    synth_null = g2_synthetic_empirical_null(r_tilde_pool, s_m_values, block_ids_raw, null_lags, n_perm, SYNTHETIC_NULL_SEED)

    real_rho_by_lag = {k: float(acf_raw.loc[acf_raw["lag"] == k, "rho"].iloc[0]) for k in BOOTSTRAP_LAGS}
    bootstrap_lag_idx = [null_lags.index(k) for k in BOOTSTRAP_LAGS]

    # --- G2 PRINCIPAL: SOLO Null 1 (3a revision -- Null 2 no supero el diagnostico de momentos, seccion 11.1 del informe) ---
    g2_null1_calibration_raw = g2_null1_calibration_summary(real_rho_by_lag, perm_null[:, bootstrap_lag_idx], BOOTSTRAP_LAGS)

    short_m = tuple(m for m in LJUNG_BOX_M if m <= G2_NULL_MAX_LAG)
    real_q_by_m = {
        m: float(portmanteau_raw.loc[portmanteau_raw["m"] == m, "Q"].iloc[0])
        for m in short_m if (portmanteau_raw["m"] == m).any()
    }
    g2_null1_portmanteau_raw = g2_null1_portmanteau_summary(perm_null, acf_raw["n_pairs"].to_numpy(), short_m, real_q=real_q_by_m)

    # --- G2 SECUNDARIO: sensibilidad historica combinando Null 1 + Null 2 (nunca la inferencia principal) ---
    g2_calibration_combined_secondary = g2_combined_calibration_summary(
        real_rho_by_lag, perm_null[:, bootstrap_lag_idx], synth_null[:, bootstrap_lag_idx], BOOTSTRAP_LAGS
    )
    g2_portmanteau_combined_secondary = g2_combined_portmanteau_summary(perm_null, synth_null, acf_raw["n_pairs"].to_numpy(), short_m)

    # --- Diagnostico de momentos del null sintetico (seccion 6) --------------
    synthetic_sample_for_check = draw_synthetic_empirical_sample(r_tilde_pool, s_m_values, SYNTHETIC_NULL_SEED)
    moment_check_result = g2_synthetic_null_moment_check(values_raw, synthetic_sample_for_check, minute_of_day_raw, s_m_values)

    # --- Traduccion a ticks (rho_1 y beta_1 separados) ----------------------
    rho_1_raw = float(acf_raw.loc[acf_raw["lag"] == 1, "rho"].iloc[0])
    beta_1_raw = float(acf_raw.loc[acf_raw["lag"] == 1, "beta"].iloc[0])
    sigma_r_raw = float(values_raw.std(ddof=1))
    ticks_translation = translate_dependence_to_ticks(rho_1_raw, beta_1_raw, sigma_r_raw, r1m_pop["close"].to_numpy(dtype=float), tick_size)

    # --- TH17: 1->5->10 minutos, decil de volumen (CORREGIDO + sensibilidad) ---
    multi_frequency = compute_multi_frequency_rho1(variables, validity, n_boot=n_boot, seed=DEFAULT_BOOTSTRAP_SEED)

    decile_labels = r1m_pop["volume_decile"].to_numpy()
    decile_primary = acf_by_group(values_raw, block_ids_raw, decile_labels, lags=tuple(range(1, DECILE_MAX_LAG + 1)))
    decile_primary.insert(0, "group_type", "volume_decile")
    volume_decile_sensitivity = acf_by_group_strict_same_block(r1m_pop, "r_1m", "volume_decile", lags=tuple(range(1, DECILE_MAX_LAG + 1)))

    # --- TH18: año, segmento, mes -- todo con el motor CORREGIDO -----------
    year_table = acf_by_group(values_raw, block_ids_raw, r1m_pop["year_ny"].to_numpy(), lags=tuple(range(1, SHORT_MAX_LAG + 1)))
    year_table.insert(0, "group_type", "year")
    segment_table = acf_by_group(values_raw, block_ids_raw, r1m_pop["segment_label"].to_numpy(), lags=tuple(range(1, SHORT_MAX_LAG + 1)))
    segment_table.insert(0, "group_type", "segment")
    rolling_table = rolling_rho1_by_month(r1m_pop)
    rolling_table.insert(0, "group_type", "year_month")
    rolling_table = rolling_table.rename(columns={"year_month": "group"})

    acf_by_group_long = pd.concat([year_table, segment_table, decile_primary, rolling_table], ignore_index=True)

    # --- Ventanas heredadas de TDA-06 --------------------------------------
    window_open = analyze_window(r1m_pop, WINDOW_OPEN, tick_size, n_boot, DEFAULT_BOOTSTRAP_SEED)
    window_close = analyze_window(r1m_pop, WINDOW_CLOSE, tick_size, n_boot, DEFAULT_BOOTSTRAP_SEED)

    return TDA08Result(
        r1m_population=r1m_pop, r_tilde_population=r_tilde_pop, segment_labels=segment_labels,
        acf_raw=acf_raw, acf_adjusted=acf_adjusted,
        bartlett_se_raw=bartlett_se_raw, bartlett_se_adjusted=bartlett_se_adjusted,
        pacf_raw=pacf_raw, pacf_adjusted=pacf_adjusted,
        portmanteau_raw=portmanteau_raw, portmanteau_adjusted=portmanteau_adjusted,
        bootstrap_table_raw=bootstrap_table_raw, bootstrap_table_adjusted=bootstrap_table_adjusted,
        g2_null1_calibration_raw=g2_null1_calibration_raw, g2_null1_portmanteau_raw=g2_null1_portmanteau_raw,
        g2_calibration_combined_secondary=g2_calibration_combined_secondary,
        g2_portmanteau_combined_secondary=g2_portmanteau_combined_secondary,
        g2_synthetic_null_moment_check=moment_check_result,
        ticks_translation=ticks_translation, multi_frequency=multi_frequency,
        acf_by_group_long=acf_by_group_long, volume_decile_sensitivity=volume_decile_sensitivity,
        window_open=window_open, window_close=window_close,
        max_estimable_lag_raw=max_estimable_lag_raw,
    )
