"""Controles de la aprobacion DEFINITIVA de E03 R4 (APROBADA sin commit, HEAD = A4, P3 inexistente). Fuera del
repositorio. Comprueba que la aprobacion cita exactamente A4 con sus huellas y que la ruta oficial sigue bloqueada sin
P3 (y sin crear el directorio de ejecucion)."""
import json
import subprocess
import tempfile
from pathlib import Path

import psutil
from ohlcv_dataroad.tda17_e03 import enmienda as m3

R = Path.cwd().resolve()
A3, A4 = "fbd7ef64bb9bd038c0b49c36c65eab169eab9036", "7fc78bcbc28f5e94c02bb545f46c7e89201408a2"
assert Path(m3.__file__).resolve().is_relative_to(R) and Path(m3.ROOT).resolve() == R, (m3.__file__, m3.ROOT, R)
print("# Controles de la aprobacion E03 R4 (APROBADA sin commit, HEAD = A4, P3 inexistente); psutil", psutil.__version__)
head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=R, capture_output=True, text=True).stdout.strip()
print("HEAD:", head, "| == A4:", head == A4)
info = m3.load_e03_aprobada(R)
a = info["aprobacion"]
print("load_e03_aprobada: estado", info["estado"], "| modo", info["modo"], "| ejecucion_oficial", info["ejecucion_oficial"],
      "| publicacion_E03", info["publicacion_E03"])
c = a["referencia_auditoria"]["commit_auditado"]
print("commit_auditado citado:", c, "| == A4:", c == A4, "| == A3 (obsoleto):", c == A3)
print("documento:", a["referencia_auditoria"]["documento"], "| existe:", (R / a["referencia_auditoria"]["documento"]).is_file())
print("fecha:", a["fecha"], "| aprobada_por:", a["aprobada_por"])
print("sha256_auditados = AUDITED_E03:", sorted(a["sha256_auditados"]) == sorted(m3.AUDITED_E03))
print("huellas aprobadas = blobs de A4:", all(m3.m2._hash_at(R, A4, r) == h for r, h in a["sha256_auditados"].items()))
reg = json.loads((R / m3.REGISTRY_E03).read_text(encoding="utf-8"))
pend = json.loads(subprocess.run(["git", "show", f"{A4}:{m3.REGISTRY_E03}"], cwd=R, capture_output=True).stdout)
(e,) = reg["enmiendas"]
e = dict(e, estado=pend["enmiendas"][0]["estado"], aprobacion=None)
print("registro = el de A4 salvo estado y aprobacion:", dict(reg, enmiendas=[e]) == pend)
for p3 in ("0" * 40, A4, A3):
    try:
        m3.official_e03(R, p3)
        print("official_e03", p3[:12], "-> ACTIVABLE (NO ESPERADO)")
    except m3.ProtocolNotAuthorizedError as exc:
        print("official_e03", p3[:12], "-> RECHAZADA:", exc)
for p3 in ("0" * 40, A4):
    with tempfile.TemporaryDirectory(dir=r"C:\tm19\r4ap") as t:
        ej = Path(t) / "ej"
        try:
            m3.official_publish_th35_e03(p3, R, ejecucion=ej)
            print("official_publish_th35_e03 -> PUBLICADA (NO ESPERADO)")
        except m3.ProtocolNotAuthorizedError as exc:
            print(f"official_publish_th35_e03({p3[:12]}) -> RECHAZADA:", exc, "| directorio de ejecucion creado:", ej.exists())
errors, info = m3.verify(R)
print("verify(): errores", errors, "| estado", info["estado"], "| ruta_oficial_e03:", info["ruta_oficial_e03"])
