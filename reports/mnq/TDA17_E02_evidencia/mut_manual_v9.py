"""Mutantes manuales (correccion v9), elegidos con TMUT_MANUAL:
- V8_STAT: la propiedad vuelve a ``os.stat`` (sigue enlaces simbolicos), como en v8.
- SIN_TIPO: ``os.lstat`` pero sin exigir fichero regular (acepta enlaces y puntos de reanalisis).
- EXISTS: la recuperacion y la comprobacion previa vuelven a ``Path.exists`` (ignoran enlaces rotos)."""
import os


def pytest_configure(config):
    from ohlcv_dataroad.ingest import tda17_salidas as sal
    kind = os.environ["TMUT_MANUAL"]
    if kind == "V8_STAT":
        def owned(staged, path, digest):
            try:
                a, b = os.stat(staged), os.stat(path)
            except OSError:
                return False
            return digest is not None and a.st_ino != 0 and (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino) and \
                sal._sha(path.read_bytes()) == digest
        sal._owned = owned
    elif kind == "SIN_TIPO":
        sal._plain = lambda st: True
    elif kind == "EXISTS":                                    # solo en el modulo: lexists sigue los enlaces
        class _Path:
            def __getattr__(self, name):
                return getattr(os.path, name)

            def lexists(self, p):
                return os.path.exists(p)

        class _Os:
            path = _Path()

            def __getattr__(self, name):
                return getattr(os, name)
        sal.os = _Os()
    else:
        raise ValueError(kind)
