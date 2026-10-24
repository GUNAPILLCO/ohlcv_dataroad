"""Mutantes manuales (T-MUT v8), elegidos con TMUT_MANUAL:
- C1_UNLINK: la promocion vuelve a v7 POSIX (enlace y despues borrar el preparado).
- C2_SOLO_BYTES: la propiedad se decide solo por los bytes (sin identidad), como en v7.
- REPLACE: la promocion vuelve a os.replace (sobrescribe)."""
import hashlib
import os


def pytest_configure(config):
    from ohlcv_dataroad.ingest import tda17_salidas as sal
    kind = os.environ["TMUT_MANUAL"]
    if kind == "C1_UNLINK":
        def promote(src, dst):
            os.link(src, dst)
            os.unlink(src)
        sal._promote = promote
    elif kind == "C2_SOLO_BYTES":
        def owned(staged, path, digest):
            try:
                return digest is not None and hashlib.sha256(path.read_bytes()).hexdigest() == digest
            except OSError:
                return False
        sal._owned = owned
    elif kind == "REPLACE":
        sal._promote = lambda src, dst: os.replace(src, dst)
    else:
        raise ValueError(kind)
