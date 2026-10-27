"""Verificacion del commit A4 de E03 (fuera del repositorio; solo lectura salvo git fetch).
Uso: python verificar_a4.py antes|despues   (cwd = repositorio del proyecto, PYTHONPATH con src)
antes:   HEAD = A3, indice vacio, arbol limpio (o solo los 4 ficheros de R4 si se pasa 'copiados').
despues: HEAD = A4 (unico padre A3, solo los 4 ficheros), 8 auditados = huellas R4, registro PENDIENTE."""
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

R = Path.cwd().resolve()
ZIP = Path(r"C:\Users\heguu\Desktop\TDA17_E03_motor_paralelo\PAQUETE_AUDITORIA_R4_E03.zip")
ZIP_SHA = "73db0a649cdd934ea05d3e0805faab2cd22c5435f3582635169af0c94f4e2d27"
PQ = "PAQUETE_AUDITORIA_R4_E03"
P1, A2, P2 = "24f9f1d2bb831a25e4dec6a651c0a5129f0bbca1", "2b6ad7abdf6609e6970f1fe6ee1e2569edc698b4", \
    "ff224dc7e30f7631dc46eea17db5a60dd44b07cc"
A3 = "fbd7ef64bb9bd038c0b49c36c65eab169eab9036"
FROZEN = "9cdea2f4657de1c54e3fccf86cf380b2eac8704e"
CAMBIADOS = {"src/ohlcv_dataroad/tda17_e03/motor.py": ("8c2f9fce9825700e700635c5dc11e7f893112b3c209047859248c78f9539e752", 89, 9),
             "tests/tda17_e03/test_e03_motor.py": ("89d2c92d611fad3db4b0fbc162c7cdfc69cb83a6df1730ffea52b62965c3d875", 168, 3),
             "tests/tda17_e03/test_e03_gobernanza.py": ("51338e941e21773f175060f4e6018af87f1653e547ddc5241b8e3c86dcaf8d17", 180, 11),
             "tests/tda17_e03/test_e03_cobertura.py": ("3743ad9e422d5110191fc2dd050be53b1017d431e024100f762abea96e4819fc", 9, 0)}
FASE = sys.argv[1]

from ohlcv_dataroad.ingest import tda17_protocol_freeze as frz  # noqa: E402
from ohlcv_dataroad.ingest import tda17_enmiendas as e01  # noqa: E402
from ohlcv_dataroad.tda17_e03 import enmienda as m3  # noqa: E402

assert Path(m3.__file__).resolve().is_relative_to(R) and Path(m3.ROOT).resolve() == R, (m3.__file__, m3.ROOT, R)
out, fallos = [], []


def check(ok, msg):
    out.append(f"  {'OK   ' if ok else 'FALLO'} {msg}")
    if not ok:
        fallos.append(msg)


def git(*a):
    return subprocess.run(["git", *a], cwd=R, capture_output=True, text=True).stdout.strip()


def blob(commit, rel):
    return subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=R, capture_output=True).stdout


def at(commit, rel):
    return frz.sha256_bytes(blob(commit, rel), Path(rel).suffix)


def h(b):
    return hashlib.sha256(b).hexdigest()


git("fetch", "-q", "origin")
head = git("rev-parse", "HEAD")
out.append(f"# {FASE}; {dt.datetime.now().astimezone().isoformat(timespec='seconds')}; HEAD {head}")
out.append("## Git")
check(git("rev-parse", "origin/main") == P2, f"origin/main = P2 {P2}")
check(git("branch", "--show-current") == "main" and git("rev-parse", "main") == head, "rama main = HEAD")
st = git("status", "--porcelain", "--untracked-files=all")
out.append("  cambios del arbol: " + (", ".join(l.strip() for l in st.splitlines()) or "(ninguno)"))
if FASE == "antes":
    check(head == A3, f"HEAD = A3 {A3}")
    check(git("rev-parse", f"{A3}^@") == P2, "A3 tiene un unico padre, P2")
    check(git("diff", "--cached", "--name-only") == "", "indice vacio")
    check(st == "", "arbol limpio")
else:
    check(head != A3 and git("rev-parse", f"{head}^@") == A3, "A4 tiene un unico padre, A3")
    check(sorted(git("diff-tree", "--no-commit-id", "--name-only", "-r", head).splitlines()) == sorted(CAMBIADOS),
          "A4 contiene exactamente los 4 ficheros de R4")
    ns = {l.split("\t")[2]: (int(l.split("\t")[0]), int(l.split("\t")[1])) for l in git("diff", "--numstat", A3, head).splitlines()}
    check(ns == {f: v[1:] for f, v in CAMBIADOS.items()}, f"diff A3->A4 = 4 ficheros, +{sum(v[0] for v in ns.values())}/-{sum(v[1] for v in ns.values())} (auditado +446/-23)")
    pat = zipfile.ZipFile(ZIP).read(f"{PQ}/diffs/DIFF_R4_A3_a_A4prima.patch")
    mio = subprocess.run(["git", "-c", "core.autocrlf=false", "diff", "--binary", A3, head], cwd=R, capture_output=True).stdout
    check(mio == pat, "diff A3->A4 identico byte a byte a DIFF_R4_A3_a_A4prima.patch del paquete auditado")
    check(git("diff", "--cached", "--name-only") == "" and st == "", "indice vacio y arbol limpio")

out.append("## Paquete R4 auditado")
check(h(ZIP.read_bytes()) == ZIP_SHA, f"zip R4 sha256 {ZIP_SHA}")
z = zipfile.ZipFile(ZIP)
huellas = json.loads(z.read(f"{PQ}/HUELLAS_E03_A3_vs_A4prima.json"))
for f, (hh, _, _) in CAMBIADOS.items():
    check(h(z.read(f"{PQ}/codigo_A4prima/{f}")) == hh == huellas[f]["sha256_LF_A4prima"], f"codigo_A4prima/{f} = {hh[:12]} (procedimiento)")

out.append("## 8 ficheros auditados de E03 (" + ("A3" if FASE == "antes" else "A4") + "; arbol = commit = huella esperada)")
ref = "sha256_LF_A3" if FASE == "antes" else "sha256_LF_A4prima"
check(sorted(huellas) == sorted(m3.AUDITED_E03), "HUELLAS cubre exactamente AUDITED_E03")
for f in m3.AUDITED_E03:
    w = h((R / f).read_bytes().replace(b"\r\n", b"\n"))
    c = h(blob("HEAD", f))
    check(w == c == huellas[f][ref], f"{f} = {huellas[f][ref][:12]}")
check(m3.E03_SHA256 == frz.sha256_file(R / m3.E03_YAML) == "85cbf62b6c1ac9b5e0fca7e869d6502a0f5784bcde5ed26be809d3cc8042c747",
      "E03_SHA256 = huella del YAML (85cbf62b...)")
reg = json.loads((R / m3.REGISTRY_E03).read_text(encoding="utf-8"))
(entry,) = reg["enmiendas"]
check(entry["estado"] == e01.ESTADO_PENDIENTE and entry["aprobacion"] is None, "registro E03: PENDIENTE_AUDITORIA, sin aprobacion")
check(blob("HEAD", m3.REGISTRY_E03) == blob(A3, m3.REGISTRY_E03), "registro E03 identico al de A3")

out.append("## E02 (15 huellas: arbol = registro = A2 = HEAD)")
e2 = json.loads((R / "reports/mnq/TDA17_enmiendas_E02.json").read_text(encoding="utf-8"))["enmiendas"][0]
hs = e2["aprobacion"]["sha256_auditados"]
check(len(hs) == 15 and e2["estado"] == "APROBADA", "registro E02 APROBADA con 15 huellas")
for rel, hh in hs.items():
    check(frz.sha256_file(R / rel) == hh == at(A2, rel) == at("HEAD", rel), f"{rel} = {hh[:12]}")
check(blob(P2, "reports/mnq/TDA17_enmiendas_E02.json") == blob("HEAD", "reports/mnq/TDA17_enmiendas_E02.json"),
      "registro E02 = el publicado en P2")

out.append("## E01 (huellas aprobadas)")
e1 = json.loads((R / "reports/mnq/TDA17_enmiendas.json").read_text(encoding="utf-8"))["enmiendas"][0]
for rel, hh in e1["aprobacion"]["sha256_auditados"].items():
    check(frz.sha256_file(R / rel) == hh, f"{rel} = {hh[:12]}")

out.append("## Fuera de E03 nada cambia respecto de P2")
fuera = [l for l in git("diff", "--name-only", P2, "HEAD").splitlines()
         if l not in m3.AUDITED_E03 and l != m3.REGISTRY_E03]
check(fuera == [], f"diff P2->HEAD fuera de E03: {fuera or 'ninguno'}")

out.append("## Verificadores oficiales (TDA17_INTEGRIDAD=estricta)")
env = {**os.environ, "TDA17_INTEGRIDAD": "estricta"}
for mod, ref_ in (("ohlcv_dataroad.ingest.tda17_protocol_freeze", FROZEN), ("ohlcv_dataroad.ingest.tda17_enmiendas", P1),
                  ("ohlcv_dataroad.ingest.tda17_enmiendas_e02", P2), ("ohlcv_dataroad.tda17_e03.enmienda", None)):
    cmd = [sys.executable, "-m", mod, "verificar"] + (["--referencia-git", ref_] if ref_ else [])
    r = subprocess.run(cmd, cwd=R, capture_output=True, text=True, env=env)
    out.append(f"  $ python -m {mod} verificar{' --referencia-git ' + ref_ if ref_ else ''}  (rc={r.returncode})")
    out += ["    " + l for l in (r.stdout + r.stderr).strip().splitlines()]
    check(r.returncode == 0, f"{mod.split('.')[-1]} CONFORME")
    if mod.endswith("tda17_e03.enmienda"):
        check("BLOQUEADA: [E03_PENDIENTE]" in r.stdout, "ruta oficial E03 BLOQUEADA por E03_PENDIENTE")

out.append(f"# RESULTADO: {'TODO CONFORME' if not fallos else 'FALLOS: ' + '; '.join(fallos)}")
print("\n".join(out))
sys.exit(1 if fallos else 0)
