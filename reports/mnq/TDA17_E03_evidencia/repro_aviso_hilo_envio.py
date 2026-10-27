"""Reproduccion ligera (un pipe, sin motor ni pytest) del aviso ``BufferError`` de los logs R4.

Imita ``_Motor._transferir``: el envio corre en un hilo daemon que guarda la excepcion en ``out``. Si el otro extremo
muere durante el envio, ``out["error"]`` (BrokenPipeError) -> traceback -> frame de ``_send_bytes`` -> memoryview de
``ForkingPickler.dumps`` (``buf.getbuffer()``) y frame de ``run`` -> celda ``out`` forman un ciclo. El GC ciclico puede
destruir el ``BytesIO`` antes que su memoryview: CPython lo informa como excepcion no capturable
(``Exception ignored in: <_io.BytesIO>`` / ``BufferError: Existing exports of data``), que pytest convierte en
``PytestUnraisableExceptionWarning`` en la prueba que este corriendo cuando pasa el GC. Caso de control: envio correcto.
"""
import gc
import json
import multiprocessing as mp
import sys
import threading
import time

AVISOS: list[str] = []


def _hook(u):
    AVISOS.append(f"{type(u.exc_value).__name__}: {u.exc_value} | objeto {type(u.object).__name__}")


def _hijo_muere(conn):
    conn.recv_bytes(16)  # lee un poco y muere a mitad de la transferencia


def _hijo_lee(conn):
    conn.recv()


def transferir(conn, msg, limite_s=10.0):
    out: dict = {}

    def run():
        try:
            out["valor"] = conn.send(msg)
        except BaseException as exc:  # noqa: BLE001
            out["error"] = exc
    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(limite_s)
    return ("vivo" if th.is_alive() else type(out.get("error")).__name__ if "error" in out else "ok"), out


def caso(destino, n_bytes):
    gc.collect()
    AVISOS.clear()
    a, b = mp.Pipe()
    p = mp.get_context("spawn").Process(target=destino, args=(b,), daemon=True)
    p.start()
    b.close()
    time.sleep(1.5 if destino is _hijo_muere else 0.0)
    gc.disable()
    estado, out = transferir(a, b"x" * n_bytes)
    p.join(10)
    del out
    gc.enable()
    n = gc.collect()
    a.close()
    return {"destino": destino.__name__, "bytes": n_bytes, "envio": estado, "gc_recogidos": n, "avisos": list(AVISOS)}


if __name__ == "__main__":
    sys.unraisablehook = _hook
    res = {"python": sys.version, "casos": [caso(_hijo_lee, 200_000), caso(_hijo_muere, 2_000_000),
                                           caso(_hijo_muere, 2_000_000), caso(_hijo_lee, 2_000_000)]}
    print(json.dumps(res, indent=2, ensure_ascii=False))
