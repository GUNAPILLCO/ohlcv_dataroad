"""Diagnostico SINTETICO de las dos convenciones de recuento de nulos y de la convencion que fija E02 (A1-a).

Documenta con aserciones que consecuencias tiene aplicar tau = identidades.verificacion_numerica.tau_relativa
(1e-10) a AUTOVALORES de la correlacion (prototipo v4a; convencion NORMATIVA de E02) o a VALORES SINGULARES de las
filas centradas (test de rango del protocolo; diagnostico publicado NO bloqueante en E02), y comprueba que una
diferencia de los AUTOVALORES con el registro bloquea (BLOQUEAR_HASTA_EXPLICAR) en una celda bloqueante. Sin datos
de MNQ.

Panel: u, u2 = u + eps * z, v, w (n = 5000, normales independientes). La relacion u2 ~ u es una dependencia NO
registrada de intensidad eps: 1 - R2(u2) ~ eps^2, autovalor minimo relativo ~ eps^2 / 4, valor singular minimo
relativo ~ eps / 2.
"""
from __future__ import annotations

import numpy as np
import pytest

from ohlcv_dataroad.ingest import tda17_analysis as an

TAU = 1e-10


def _panel(eps, n=5000, seed=0, scale_v=1.0):
    rng = np.random.default_rng(seed)
    u, z, v, w = rng.standard_normal((4, n))
    return np.column_stack([u, u + eps * z, scale_v * v, w])


def _spectra(X):
    st = an.cell_stats(X)
    lam = np.linalg.eigvalsh(st.corr)
    s = np.linalg.svd(st.R, compute_uv=False)
    Xn = (X - X.mean(0)) / np.linalg.norm(X - X.mean(0), axis=0)
    s_norm = np.linalg.svd(Xn, compute_uv=False)
    return st, lam, s, s_norm


@pytest.mark.parametrize("eps", [1e-1, 1e-3, 1e-6, 1e-8])
def test_tau_on_eigenvalues_equals_sqrt_tau_on_normalized_singular_values(eps):
    """Sobre columnas normalizadas, autovalores de la correlacion = valores singulares al cuadrado: aplicar tau a
    autovalores equivale a aplicar sqrt(tau) = 1e-5 a valores singulares (las convenciones difieren por un
    cuadrado, no por un detalle numerico)."""
    _, lam, _, s_norm = _spectra(_panel(eps))
    np.testing.assert_allclose(np.sort(lam)[::-1][:3], np.sort(s_norm ** 2)[::-1][:3], rtol=1e-9)
    eig_null = int(np.sum(lam <= TAU * lam.max()))
    sing_sqrt = int(np.sum(s_norm <= np.sqrt(TAU) * s_norm.max()))
    assert eig_null == sing_sqrt


@pytest.mark.parametrize("eps,expected", [
    (1e-2, (0, 0)), (1e-4, (0, 0)), (3e-5, (0, 0)),         # dependencia debil: ninguna la anula
    (1e-5, (1, 0)), (1e-7, (1, 0)), (1e-9, (1, 0)),         # franja de discrepancia: solo los autovalores
    (1e-11, (1, 1)), (0.0, (1, 1)),                         # dependencia exacta a precision de maquina: ambas
])
def test_disagreement_band_between_conventions(eps, expected):
    st = an.cell_stats(_panel(eps))
    counts = an.null_counts(st, TAU)
    assert (counts["autovalores"], counts["singulares"]) == expected
    status = an.rank_check(counts, 0, "diagnostico", "BLOQUEANTE")["estado"]   # el registro no tiene esta dependencia
    assert status == ("CONFORME" if expected[0] == 0 else "BLOQUEAR_HASTA_EXPLICAR")


def test_eigenvalue_nulls_fall_inside_the_alert_label_and_singular_nulls_at_machine_precision():
    """Relacion con las etiquetas de exactitud: un nulo por autovalores aparece con 1 - R2 ~ 1e-10, dentro de
    ALERTA_DEPENDENCIA_NO_REGISTRADA (<= 1e-9); un nulo por valores singulares solo cuando la dependencia es exacta
    a precision de maquina (1 - R2 ~ 1e-20, irrepresentable). Por encima de 1e-9 ninguna convencion anula y la
    columna es CASI_EXACTA_COLINEAL o queda sin etiqueta."""
    for eps in (1e-1, 1e-2, 1e-3, 1e-4, 3e-5, 1e-5, 1e-6, 1e-7, 1e-9, 1e-11):
        st = an.cell_stats(_panel(eps))
        counts = an.null_counts(st, TAU)
        r2 = an.r2_table(st, ["u", "u2", "v", "w"], None, TAU, 1e-8).set_index("columna")["r2"]
        gap = 1.0 - r2["u2"]
        if counts["autovalores"]:
            assert gap <= 1e-9, eps                                         # nulo por autovalores => zona de ALERTA
        if counts["singulares"]:
            assert eps <= 1e-10 and gap <= 1e-15, eps
        if gap > 1e-9:
            assert counts == {"autovalores": 0, "singulares": 0}, eps


def test_eigenvalue_resolution_floor():
    """Los autovalores de la correlacion en doble precision tienen error absoluto ~1e-16: por debajo de esa escala
    (eps <~ 1e-8) el autovalor minimo es ruido (incluso negativo) y ya no mide la intensidad de la dependencia;
    los valores singulares conservan la escala eps / 2 hasta ~1e-16."""
    for eps in (1e-8, 1e-10, 1e-12):
        _, lam, s, _ = _spectra(_panel(eps))
        assert abs(lam.min()) / lam.max() < 1e-15
        assert s.min() / s.max() == pytest.approx(eps / 2, rel=0.05)


def test_cross_verification_of_r2_fails_in_the_same_band():
    """En la franja de discrepancia, la verificacion cruzada del R2 de las demas columnas tambien falla
    (R2_NO_PUBLICABLE_POR_COLINEALIDAD); con una copia exacta (eps = 0) los tres metodos coinciden."""
    for eps, publishable in ((1e-3, True), (1e-5, False), (1e-9, False), (1e-12, False), (0.0, True)):
        tab = an.r2_table(an.cell_stats(_panel(eps)), ["u", "u2", "v", "w"], None, TAU, 1e-8)
        assert bool(tab["publicable"].all()) is publishable, eps


@pytest.mark.parametrize("scale,expected", [(1e-4, (0, 0)), (1e-8, (0, 0)), (1e-10, (0, 1)), (1e-12, (0, 1))])
def test_singular_convention_on_unnormalized_rows_depends_on_column_scale(scale, expected):
    """Valores singulares de las filas SIN normalizar: una columna independiente con escala relativa <= tau cuenta
    como nula aunque no haya dependencia (los autovalores de la correlacion no dependen de la escala). En P las
    escalas relativas de columnas van de ~1e-4 (r_1m) a ~1e2 (CCI): lejos de 1e-10, pero la convencion lo
    permite. Con E02 los valores singulares son solo informativos: no bloquean."""
    counts = an.null_counts(an.cell_stats(_panel(1e-1, scale_v=scale)), TAU)
    assert (counts["autovalores"], counts["singulares"]) == expected
    chk = an.rank_check(counts, 0, "escala", "BLOQUEANTE")
    assert chk["estado"] == "CONFORME" and chk["singulares_igual_registro"] is (expected[1] == 0)


def test_e02_convention_eigenvalues_normative_and_singular_values_informative():
    assert "CONVENCION_NULOS" not in an.PENDIENTES
    for counts, expected, bloq, info in (({"autovalores": 11, "singulares": 11}, 11, "CONFORME", "CONFORME"),
                                         ({"autovalores": 12, "singulares": 11}, 11, "BLOQUEAR_HASTA_EXPLICAR", "DIFIERE_INFORMATIVO"),
                                         ({"autovalores": 11, "singulares": 12}, 11, "CONFORME", "CONFORME"),
                                         ({"autovalores": 12, "singulares": 12}, 11, "BLOQUEAR_HASTA_EXPLICAR", "DIFIERE_INFORMATIVO")):
        assert an.rank_check(counts, expected, "P_centrada", "BLOQUEANTE")["estado"] == bloq
        assert an.rank_check(counts, expected, "P_centrada", "INFORMATIVA")["estado"] == info
    with pytest.raises(ValueError, match="ambito desconocido"):
        an.rank_check({"autovalores": 11, "singulares": 11}, 11, "P_centrada", "OTRO")
