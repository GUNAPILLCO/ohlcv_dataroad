"""Huella de BYTES, tamano, mtime e inodo de los 9 ficheros data/ del manifiesto (solo lectura).
Uso: python datos_integridad.py <salida.txt>   (desde la raiz del repositorio)"""
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))
from ohlcv_dataroad.ingest import tda17_protocol_freeze as frz  # noqa: E402

m = frz.load_manifest(Path.cwd())
lines = []
for g in ("sha256_entradas", "sha256_artefactos_previos"):
    for rel in m[g]:
        if rel.startswith("data/"):
            st = os.stat(rel)
            h = hashlib.sha256(Path(rel).read_bytes()).hexdigest()
            ok = "MANIFIESTO_OK" if h == m[g][rel] else "DISTINTO_DEL_MANIFIESTO"
            lines.append(f"{h}  {st.st_size}  mtime_ns={st.st_mtime_ns}  ino={st.st_ino}  {ok}  {rel}")
Path(sys.argv[1]).write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
