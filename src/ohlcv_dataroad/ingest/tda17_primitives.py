"""TDA-17 -- primitivas causales para el calculo de indicadores (protocolo congelado ``configs/tda17.yaml``).

Todas las funciones trabajan sobre UN contrato (entre dos reinicios) y solo usan barras ``<= t`` para la
salida en ``t``: ventanas que terminan en ``t`` (nunca centradas), recursiones hacia delante, ningun
desplazamiento negativo y ningun parametro estimado con la muestra completa.

Convenciones (seccion ``calentamiento`` del protocolo):
- EMA y Wilder: semilla = media simple de las n primeras ENTRADAS recibidas; despues, ``y += alpha (x - y)``.
- Peso residual de la inicializacion < 0,01: ``k = ceil(ln 0,01 / ln(1 - alpha))`` ACTUALIZACIONES.
- Ventanas en barras observadas; ``NaN`` si cualquier valor de la ventana es ``NaN`` (sin relleno).
- Ventanas de incrementos: n pares que terminan en las barras t-n+1..t; validas solo con t >= n
  (indice base 0) y >= ceil(0,8 n) pares CONS.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter

TICK = 0.25
VALIDEZ_MINIMA = 0.8
SIGMA_MIN = 1e-8


def k_weight(alpha: float) -> int:
    """Actualizaciones adicionales para que el peso de la inicializacion sea < 0,01."""
    return math.ceil(math.log(0.01) / math.log(1.0 - alpha))


def ema_alpha(n: int) -> float:
    return 2.0 / (n + 1)


def wilder_alpha(n: int) -> float:
    return 1.0 / n


def min_valid(n: int) -> int:
    return math.ceil(VALIDEZ_MINIMA * n)


# --------------------------------------------------------------------------- divisiones y logaritmos
def safe_div(num, den):
    """reglas_frontera.divisiones: denominador 0 o no finito -> NaN; resultado no finito -> NaN."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return np.where(np.isfinite(den) & (den != 0) & np.isfinite(out), out, np.nan)


def safe_log(x):
    """ln solo de valores finitos y > 0; en otro caso NaN."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(x)
    return np.where(np.isfinite(x) & (x > 0) & np.isfinite(out), out, np.nan)


# --------------------------------------------------------------------------- suavizados con semilla
def seeded_smoother(x: np.ndarray, n: int, alpha: float) -> np.ndarray:
    """Suavizado sobre una secuencia de ENTRADAS (ya en orden de actualizacion, todas finitas).

    ``y[i]`` es el valor tras i+1 entradas: ``NaN`` para i < n-1; ``y[n-1]`` = media de las n primeras;
    despues ``y[i] = y[i-1] + alpha (x[i] - y[i-1])`` (recursion exacta mediante ``lfilter``)."""
    x = np.asarray(x, dtype=float)
    y = np.full(x.size, np.nan)
    if x.size < n:
        return y
    seed = float(np.mean(x[:n]))
    y[n - 1] = seed
    if x.size > n:
        y[n:], _ = lfilter([alpha], [1.0, alpha - 1.0], x[n:], zi=[(1.0 - alpha) * seed])
    return y


def smoother_on_bars(x: np.ndarray, n: int, alpha: float, start: int = 0) -> tuple[np.ndarray, int]:
    """Suavizado que se actualiza en TODAS las barras desde ``start`` (base 0).

    Devuelve los valores internos (desde la semilla) y el indice base 0 de la primera barra ELEGIBLE:
    ``start + n - 1 + k`` (convencion K = b + n - 1 + k con b = start + 1)."""
    out = np.full(x.size, np.nan)
    if start < x.size:
        out[start:] = seeded_smoother(x[start:], n, alpha)
    return out, start + n - 1 + k_weight(alpha)


def mask_before(x: np.ndarray, first_idx: int) -> np.ndarray:
    """Salida publicada: ``NaN`` antes de la primera barra elegible (base 0)."""
    out = np.array(x, dtype=float, copy=True)
    out[: max(first_idx, 0)] = np.nan
    return out


# --------------------------------------------------------------------------- contador de actualizaciones CONS
def cons_counter(cons: np.ndarray) -> np.ndarray:
    """u_t = numero de barras <= t (del contrato) cuyo par es CONS; causal."""
    return np.cumsum(np.asarray(cons, dtype=bool))


def values_at_bars(y_updates: np.ndarray, cons: np.ndarray) -> np.ndarray:
    """Valor del estado en cada barra = valor tras la ultima actualizacion CONS (estado retenido)."""
    u = cons_counter(cons)
    out = np.full(cons.size, np.nan)
    has = u > 0
    out[has] = y_updates[u[has] - 1]
    return out


# --------------------------------------------------------------------------- ventanas
def rolling(x: np.ndarray, n: int, fn) -> np.ndarray:
    """Ventana de n barras observadas que termina en t; ``NaN`` en las n-1 primeras barras.
    ``fn`` recibe la vista (filas = ventanas) y opera sobre ``axis=-1``. Un ``NaN`` en la ventana da ``NaN``."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.size, np.nan)
    if x.size >= n:
        out[n - 1:] = fn(sliding_window_view(x, n))
    return out


def rolling_mean(x, n):
    return rolling(x, n, lambda w: w.mean(axis=-1))


def rolling_sum(x, n):
    return rolling(x, n, lambda w: w.sum(axis=-1))


def rolling_max(x, n):
    return rolling(x, n, lambda w: w.max(axis=-1))


def rolling_min(x, n):
    return rolling(x, n, lambda w: w.min(axis=-1))


def rolling_pstd(x, n):
    """Desviacion tipica POBLACIONAL (ddof = 0) de la ventana."""
    return rolling(x, n, lambda w: w.std(axis=-1))


def rolling_wma(x, n):
    """Media ponderada lineal: peso 1 a la barra mas antigua y n a la barra t."""
    w = np.arange(1, n + 1, dtype=float)
    return rolling(x, n, lambda v: (v @ w) / w.sum())


def rolling_mean_abs_dev(x, n):
    return rolling(x, n, lambda w: np.abs(w - w.mean(axis=-1, keepdims=True)).mean(axis=-1))


def rolling_ols_slope(x, n):
    """Pendiente MCO de x sobre 1..n en la ventana que termina en t."""
    t = np.arange(1, n + 1, dtype=float)
    tc = t - t.mean()
    return rolling(x, n, lambda w: (w @ tc) / (tc @ tc))


def bars_since_extreme(x, window, kind):
    """Barras transcurridas desde el maximo (o minimo) de la ventana de ``window`` barras que termina en t.
    Empate: se toma la aparicion MAS RECIENTE."""
    def fn(w):
        rev = w[:, ::-1]
        idx = np.argmax(rev, axis=-1) if kind == "max" else np.argmin(rev, axis=-1)
        bad = np.isnan(w).any(axis=-1)
        return np.where(bad, np.nan, idx.astype(float))
    return rolling(x, window, fn)


def increment_window(inc: np.ndarray, n: int):
    """Ventana de n pares (barras t-n+1..t); ``inc`` es ``NaN`` fuera de CONS.

    Devuelve (suma de incrementos CONS, numero de pares CONS, mascara de validez estructural)."""
    inc = np.asarray(inc, dtype=float)
    present = ~np.isnan(inc)
    count = rolling(present.astype(float), n, lambda w: w.sum(axis=-1))
    total = rolling(np.where(present, inc, 0.0), n, lambda w: w.sum(axis=-1))
    idx = np.arange(inc.size)
    valid = (idx >= n) & (np.nan_to_num(count) >= min_valid(n))
    return total, count, valid
