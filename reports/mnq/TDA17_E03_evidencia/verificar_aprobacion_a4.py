"""Verificacion de la aprobacion DEFINITIVA de E03 R4 (registro APROBADA citando A4, sin commit; fuera del repositorio;
solo lectura salvo git fetch). Uso (cwd = repositorio, PYTHONPATH con src): python verificar_aprobacion_a4.py"""
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

R = Path.cwd().resolve()
D = Path(r"C:\Users\heguu\Desktop\TDA17_E03_motor_paralelo")
ZIP3 = ("PAQUETE_AUDITORIA_R3_E03.zip", "4cbf5ecf914d6024c9a6425b9b6d3b5a3eaf0550fd7e7adec5d4e949dfb290f9")
ZIP4 = ("PAQUETE_AUDITORIA_R4_E03.zip", "73db0a649cdd934ea05d3e0805faab2cd22c5435f3582635169af0c94f4e2d27")
P1, A2, P2 = "24f9f1d2bb831a25e4dec6a651c0a5129f0bbca1", "2b6ad7abdf6609e6970f1fe6ee1e2569edc698b4", \
    "ff224dc7e30f7631dc46eea17db5a60dd44b07cc"
A3, A4 = "fbd7ef64bb9bd038c0b49c36c65eab169eab9036", "7fc78bcbc28f5e94c02bb545f46c7e89201408a2"
FROZEN = "9cdea2f4657de1c54e3fccf86cf380b2eac8704e"
ACTA = "reports/mnq/TDA17_E03_acta_aprobacion.md"
EVD = "reports/mnq/TDA17_E03_evidencia"
FECHA = "2026-10-26"
POR = "Responsable del proyecto IRIS (declaración de aprobación; identidad no autenticada)"

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


def h(b):
    return hashlib.sha256(b).hexdigest()


git("fetch", "-q", "origin")
out.append(f"# aprobacion E03 R4 (sin commit); {dt.datetime.now().astimezone().isoformat(timespec='seconds')}")
out.append("## Git")
check(git("rev-parse", "HEAD") == A4, f"HEAD = A4 {A4}")
check(git("rev-parse", f"{A4}^@") == A3 and git("rev-parse", f"{A3}^@") == P2, "A4 -> unico padre A3 -> unico padre P2")
check(git("rev-parse", "origin/main") == P2, f"origin/main = P2 {P2} (sin push)")
check(git("branch", "--show-current") == "main", "rama main")
check(git("diff", "--cached", "--name-only") == "", "indice vacio (sin commit)")
st = sorted(subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=R, capture_output=True,
                           text=True).stdout.splitlines())          # sin strip: conserva la columna de estado " M"
evid = sorted(p.relative_to(R).as_posix() for p in (R / EVD).rglob("*") if p.is_file())
esperado = sorted([f" M {m3.REGISTRY_E03}", f"?? {ACTA}"] + [f"?? {e}" for e in evid])
check(st == esperado, f"cambios = registro (M) + acta + {len(evid)} evidencias (nuevos); nada mas")
check(git("diff", "--name-only") == m3.REGISTRY_E03, "unico fichero seguido modificado: el registro de E03")

out.append("## 8 ficheros auditados (arbol = A4 = aprobacion = paquete R4)")
reg = json.loads((R / m3.REGISTRY_E03).read_text(encoding="utf-8"))
(entry,) = reg["enmiendas"]
a = entry["aprobacion"] or {}
z4 = zipfile.ZipFile(D / ZIP4[0])
huellas = json.loads(z4.read("PAQUETE_AUDITORIA_R4_E03/HUELLAS_E03_A3_vs_A4prima.json"))
check(sorted(a.get("sha256_auditados", {})) == sorted(m3.AUDITED_E03) == sorted(huellas), "la aprobacion cubre exactamente AUDITED_E03")
for f in m3.AUDITED_E03:
    w = h((R / f).read_bytes().replace(b"\r\n", b"\n"))
    check(w == h(blob(A4, f)) == a["sha256_auditados"][f] == huellas[f]["sha256_LF_A4prima"], f"{f} = {w[:12]}")
check(m3.E03_SHA256 == frz.sha256_file(R / m3.E03_YAML) == entry["sha256"], "E03_SHA256 = YAML = registro (85cbf62b...)")

out.append("## Registro E03")
check(entry["estado"] == e01.ESTADO_APROBADA, "estado APROBADA")
check(list(a) == list(e01.APPROVAL_FIELDS), f"aprobacion con exactamente {list(e01.APPROVAL_FIELDS)}")
check(a["fecha"] == FECHA and a["aprobada_por"] == POR, f"fecha {FECHA}; aprobada_por = la declaracion autorizada")
check(a["referencia_auditoria"] == {"documento": ACTA, "commit_auditado": A4}, "referencia_auditoria = {acta, A4} (cita exclusivamente A4)")
pend = json.loads(blob(A4, m3.REGISTRY_E03))
check(dict(reg, enmiendas=[dict(entry, estado=e01.ESTADO_PENDIENTE, aprobacion=None)]) == pend,
      "registro = el de A4 (PENDIENTE) salvo estado y aprobacion")
check((R / m3.REGISTRY_E03).read_bytes().replace(b"\r\n", b"\n") == (json.dumps(reg, indent=2, ensure_ascii=False) + "\n").encode(),
      "formato del registro (json indent 2, LF)")

out.append("## Acta")
acta = (R / ACTA).read_text(encoding="utf-8")
check((R / ACTA).is_file() and not (R / ACTA).is_symlink(), "acta existente (fichero regular)")
check(f"**Commit auditado (A4):** `{A4}`" in acta, "el acta declara A4 como commit auditado")
check(f"**Fecha:** {FECHA}" in acta and f"**Aprobada por:** {POR}" in acta, "fecha y responsable = registro")
check(all(f"| `{f}` | `{a['sha256_auditados'][f]}` |" in acta for f in m3.AUDITED_E03), "las 8 huellas del acta = registro")
check(ZIP4[1] in acta and ZIP3[1] in acta, "el acta cita los SHA-256 de los paquetes R3 y R4 auditados")
check(h(blob(A4, m3.REGISTRY_E03)) in acta and m3.E03_SHA256 in acta, "huellas del registro PENDIENTE (A4) y del YAML")
check(re.search(r"commit auditado \(A3\)", acta, re.I) is None, "el acta no cita A3 como commit auditado")
check("BufferError" in acta and "AVISOS_HILO_ENVIO.json" in acta, "observacion BufferError documentada")

out.append("## Evidencias")
for nombre, sha in (ZIP3, ZIP4):
    check(h((D / nombre).read_bytes()) == sha, f"{nombre} = {sha[:12]}")
ev = R / EVD
readme = (ev / "README.md").read_text(encoding="utf-8")
filas = dict(re.findall(r"^\| `([^`]+)` \| `([0-9a-f]{64})` \|", readme, re.M))
presentes = {p.relative_to(ev).as_posix() for p in ev.rglob("*") if p.is_file()} - {"README.md", "verificadores_despues.txt"}
check(set(filas) == presentes, f"README lista exactamente los {len(presentes)} ficheros de evidencia")
check(all(h((ev / f).read_bytes()) == s for f, s in filas.items() if (ev / f).exists()), "cada evidencia = su SHA-256 del README")
check((ev / ".gitattributes").read_bytes() == b"* -text\n", "evidencias byte a byte (* -text)")
man4 = z4.read("PAQUETE_AUDITORIA_R4_E03/MANIFIESTO_SHA256.txt").decode()
check((ev / "R4_MANIFIESTO_SHA256.txt").read_text(encoding="utf-8") == man4, "manifiesto R4 = el del zip auditado")
diff = subprocess.run(["git", "-c", "core.autocrlf=false", "diff", "--binary", A3, A4], cwd=R, capture_output=True).stdout
check((ev / "R4_DIFF_R4_A3_a_A4prima.patch").read_bytes() == diff, "diff auditado = git diff A3 A4 (byte a byte)")
apr = json.loads((ev / "registro_aprobacion_salida.json").read_text(encoding="utf-8"))
check(apr == a, "registro_aprobacion_salida.json = aprobacion registrada")
ctl = (ev / "controles_aprobada.txt").read_text(encoding="utf-8")
check("== A4: True" in ctl and "NO ESPERADO" not in ctl and "creado: True" not in ctl, "controles: cita A4, nada activable ni creado")
check("1 passed" in (ev / "test_estado_proyecto_aprobada.log").read_text(encoding="utf-8"), "prueba de estado del proyecto en APROBADA: 1 passed")
check("9 passed" in (ev / "cobertura_aprobada.log").read_text(encoding="utf-8"), "cobertura en APROBADA: 9 passed")
for n, txt in (("R4_suite_PENDIENTE.log", "1917 passed, 13 skipped"), ("R4_suite_APROBADA.log", "1917 passed, 13 skipped"),
               ("R4_e03_PENDIENTE.log", "121 passed"), ("R4_e03_APROBADA.log", "121 passed")):
    t = (ev / n).read_text(encoding="utf-8").rstrip().splitlines()
    check(txt in "\n".join(t[-4:]) and t[-2] == "rc=0", f"{n}: {txt}, rc=0")

out.append("## E02 (15 huellas: arbol = registro = A2 = HEAD) y E01")
e2 = json.loads((R / "reports/mnq/TDA17_enmiendas_E02.json").read_text(encoding="utf-8"))["enmiendas"][0]
hs = e2["aprobacion"]["sha256_auditados"]
check(len(hs) == 15 and e2["estado"] == "APROBADA", "registro E02 APROBADA con 15 huellas")
for rel, hh in hs.items():
    check(frz.sha256_file(R / rel) == hh == frz.sha256_bytes(blob(A2, rel), Path(rel).suffix) ==
          frz.sha256_bytes(blob("HEAD", rel), Path(rel).suffix), f"{rel} = {hh[:12]}")
check(blob(P2, "reports/mnq/TDA17_enmiendas_E02.json") == blob("HEAD", "reports/mnq/TDA17_enmiendas_E02.json"), "registro E02 = P2")
e1 = json.loads((R / "reports/mnq/TDA17_enmiendas.json").read_text(encoding="utf-8"))["enmiendas"][0]
for rel, hh in e1["aprobacion"]["sha256_auditados"].items():
    check(frz.sha256_file(R / rel) == hh, f"E01 {rel} = {hh[:12]}")
fuera = [l for l in git("diff", "--name-only", P2, "HEAD").splitlines() if l not in m3.TRACKED_E03]
check(fuera == [], f"diff P2->A4 fuera de E03: {fuera or 'ninguno'}")

out.append("## Verificadores oficiales (TDA17_INTEGRIDAD=estricta)")
env = {**os.environ, "TDA17_INTEGRIDAD": "estricta"}
for mod, ref in (("ohlcv_dataroad.ingest.tda17_protocol_freeze", FROZEN), ("ohlcv_dataroad.ingest.tda17_enmiendas", P1),
                 ("ohlcv_dataroad.ingest.tda17_enmiendas_e02", P2), ("ohlcv_dataroad.tda17_e03.enmienda", None)):
    cmd = [sys.executable, "-m", mod, "verificar"] + (["--referencia-git", ref] if ref else [])
    r = subprocess.run(cmd, cwd=R, capture_output=True, text=True, env=env)
    out.append(f"  $ python -m {mod} verificar{' --referencia-git ' + ref if ref else ''}  (rc={r.returncode})")
    out += ["    " + l for l in (r.stdout + r.stderr).strip().splitlines()]
    check(r.returncode == 0, f"{mod.split('.')[-1]} CONFORME")
    if mod.endswith("tda17_e03.enmienda"):
        check("estado APROBADA" in r.stdout and "BLOQUEADA: [E03_P3_EXISTE]" in r.stdout,
              "E03 APROBADA y ruta oficial BLOQUEADA sin P3 (E03_P3_EXISTE)")

out.append("## Ruta oficial sin P3")
info = m3.load_e03_aprobada(R)
check((info["estado"], info["ejecucion_oficial"], info["publicacion_E03"]) == (e01.ESTADO_APROBADA, False, None),
      "load_e03_aprobada: APROBADA, ejecucion_oficial False, sin publicacion")
for p3, cod in (("0" * 40, "[E03_P3_EXISTE]"), (A4, "[E03_P3_A3]"), (A3, "[E03_P3_A3]")):
    try:
        m3.official_e03(R, p3)
        check(False, f"official_e03({p3[:12]}) rechazada")
    except m3.ProtocolNotAuthorizedError as exc:
        check(str(exc).startswith(cod), f"official_e03({p3[:12]}) rechazada {cod}")

out.append(f"# RESULTADO: {'TODO CONFORME' if not fallos else 'FALLOS: ' + '; '.join(fallos)}")
print("\n".join(out))
sys.exit(1 if fallos else 0)
