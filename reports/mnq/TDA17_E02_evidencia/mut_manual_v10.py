"""Mutantes manuales (correccion v10), elegidos con TMUT_MANUAL:
- SIGUE_ENLACES: el verificador vuelve a leer siguiendo enlaces (``is_file`` + ``read_bytes``), como en v9.
- SIN_TIPO: ``_plain`` acepta cualquier cosa (enlaces y puntos de reanalisis).
- EXISTS_SALIDAS: ``tda17_salidas`` mira la presencia con ``exists`` (ignora enlaces rotos).
- EXISTS_VERIFY: ``verify`` (E02) decide si comprobar con ``exists`` (ignora enlaces rotos)."""
import os


def _following_os():
    class _Path:
        def __getattr__(self, name):
            return getattr(os.path, name)

        def lexists(self, p):
            return os.path.exists(p)

    class _Os:
        path = _Path()

        def __getattr__(self, name):
            return getattr(os, name)
    return _Os()


def pytest_configure(config):
    from ohlcv_dataroad.ingest import tda17_enmiendas_e02 as m2
    from ohlcv_dataroad.ingest import tda17_salidas as sal
    kind = os.environ["TMUT_MANUAL"]
    if kind == "SIGUE_ENLACES":
        sal._regular_bytes = lambda p: p.read_bytes() if p.is_file() else None
    elif kind == "SIN_TIPO":
        sal._plain = lambda st: True
    elif kind == "EXISTS_SALIDAS":
        sal.os = _following_os()
    elif kind == "EXISTS_VERIFY":
        m2.os = _following_os()
    else:
        raise ValueError(kind)
