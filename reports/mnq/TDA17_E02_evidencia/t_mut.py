"""T-MUT (SPEC-M2 seccion 11): mutaciones de CODIGO FUENTE del cargador real de E02, cada una en un clon Git.

Para cada identificador de control [ID] de tda17_enmiendas_e02.py, el mutante sustituye TODAS sus llamadas
``_fail("ID", ...)`` por una funcion que no lanza (elimina el rechazo). En el clon (historia real + ficheros de E02 del
arbol + COPIAS independientes de los datos del manifiesto, nunca enlaces a los originales) se ejecutan las pruebas seleccionadas de
tests/test_tda17_enmienda_02.py y tests/test_tda17_e02_motor.py con PYTHONPATH = clon/src: cada escenario de
prueba clona A SU VEZ el clon mutante y recalcula las aprobaciones con las huellas del mutante. Un mutante esta
DETECTADO si las pruebas terminan con codigo != 0. Control: el clon sin mutar debe dar codigo 0.

Uso: python t_mut.py <directorio_de_trabajo_corto> [ID1,ID2,...]   (opcional: solo esos identificadores)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(r"C:\Users\heguu\Desktop\ohlcv_dataroad")
LOADER = "src/ohlcv_dataroad/ingest/tda17_enmiendas_e02.py"
TESTS = ["tests/test_tda17_enmienda_02.py", "tests/test_tda17_e02_motor.py", "tests/test_tda17_causalidad.py",
         "tests/test_tda17_procedencia.py", "tests/test_tda17_salidas.py"]
REJECT_MODULES = ["src/ohlcv_dataroad/ingest/tda17_causalidad.py", "src/ohlcv_dataroad/ingest/tda17_entradas.py",
                  "src/ohlcv_dataroad/ingest/tda17_salidas.py"]
SELECT = {
    "GIT": "GIT or without_repository",
    "E02_P2_FORMATO": "p2_format_and_existence", "E02_P2_EXISTE": "p2_format_and_existence",
    "E02_P2_A2": "p2_format_and_existence", "E02_P2_HEAD": "branch_from_a2 or preparatory_approval",
    "E02_P2_ARBOL": "tree_differs_from_p2", "E02_P2_APROBADA": "publication_without_the_approval",
    "PROTOCOLO_ESTRICTO": "strict_protocol", "PROTOCOLO_GIT": "freeze_commit",
    "E02_REMOTO": "remote", "E02_REMOTO_HISTORIA": "remote or official_execution",
    "E02_REMOTO_NO_AUTORIZADO": "remote",
    "E02_RUTAS": "paths_barrier", "E02_PENDIENTE_CON_DATOS": "pending_with_approval_data",
    "E02_LISTA_BLANCA": "whitelist_in_borrador", "E02_ANADIR": "whitelist_in_borrador",
    "E02_REQUERIDA": "official_config_requires or production_load or production_routes",
    "E02_NO_OFICIAL": "official_config_requires or production_load",
    "E02_PENDIENTE": "E02_PENDIENTE or pending or production_routes",
    "E02_OFICIAL_NO_VERIFICABLE": "forged or official_config_requires",
    "E02_OFICIAL_DISTINTA": "forged",
    "TH37_AUSENTE": "th35_route or absent_audit",
    "TH37_INCOMPLETA": "th35_route or incomplete_audits",
    "TH37_FUERA_DEL_PLAN": "repeated_or_foreign",
    "TH37_DISCREPANCIA": "th35_route or discrepancy_blocks",
    "ENTRADA_PROCEDENCIA": "inputs or official_th35",
    "ENTRADA_PROHIBIDA": "inputs_read_verified",
    "ENTRADA_NO_DECLARADA": "official_inputs_are_the_manifest_entries",
    "SALIDA_ESQUEMA": "schema or tampering",
    "SALIDA_COLUMNAS": "inconsistent",
    "SALIDA_ESTADO": "inconsistent",
    "SALIDA_PROCEDENCIA": "another_configuration",
    "SALIDA_EXISTENTE": "publish_blocked or created_after_the_check or rollback_never_deletes_a_symbolic",
    "SALIDA_REGISTRO": "tampering or valid_symbolic_link or broken_symbolic or only_a_broken", "SALIDA_P2": "tampering", "SALIDA_EFECTIVA": "tampering",
    "SALIDA_ENTRADAS": "tampering", "SALIDA_HUELLA": "tampering or valid_symbolic_link or broken_symbolic",
    "SALIDA_CONJUNTO": "tampering or publish_unblocked or broken_symbolic",
    "SALIDA_NO_REPRODUCIBLE": "tampering or different_state",
    "SALIDA_RUTA_UNICA": "route_is_unique or official_execution",
    "SALIDA_CONCURRENTE": "simultaneous or another_process",
    "SALIDA_INCOMPLETA": "interrupted or right_after_the_link",
    "SALIDA_RECUPERACION": "cannot_prove_its_own or symbolic_link",
}


def run(cmd, cwd, env=None, timeout=7200):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired as exc:                # infraestructura: nunca cuenta como deteccion
        return None, f"TIMEOUT tras {timeout} s: {exc}"
    return r.returncode, r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")


def classify(code, out: str) -> tuple[str, list[str], list[str]]:
    """Cierre: DETECTADO solo si pytest termina con codigo 1 (pruebas fallidas), con al menos un FAILED y sin
    ERROR de recogida o de preparacion. Codigo 0 = SUPERVIVIENTE. Cualquier otra cosa (timeout, codigos 2-5,
    errores de importacion o de fixture, ninguna prueba seleccionada) = INFRAESTRUCTURA, que no cuenta como
    detectado y obliga a revisar."""
    failed = re.findall(r"^FAILED (\S+)", out, re.M)
    errors = re.findall(r"^ERROR (\S+)", out, re.M)
    if code == 0:
        return "SUPERVIVIENTE", failed, errors
    if code == 1 and failed and not errors:
        return "DETECTADO", failed, errors
    return "INFRAESTRUCTURA", failed, errors


def tracked_e02() -> list[str]:
    sys.path.insert(0, str(SRC / "src"))
    from ohlcv_dataroad.ingest import tda17_enmiendas_e02 as m2
    from ohlcv_dataroad.ingest import tda17_protocol_freeze as frz
    m = frz.load_manifest(SRC)
    data = [r for g in ("sha256_entradas", "sha256_artefactos_previos") for r in m[g] if r.startswith("data/")]
    docs = ["reports/mnq/TDA17_E02_propuesta_decisiones_normativas.md",       # documentos autorizados (sus huellas
            "reports/mnq/TDA17_E02_propuesta_ambiguedades_y_salidas.md",       # las comprueba test_tda17_salidas)
            "reports/mnq/TDA17_E02_anexo_relaciones_N.md"]
    return m2.TRACKED_E02 + docs, data


def make_clone(dest: Path, files: list[str], data: list[str]) -> None:
    if dest.exists():
        shutil.rmtree(dest, onerror=lambda f, p, e: (os.chmod(p, 0o700), f(p)))
    run(["git", "clone", "-q", str(SRC), str(dest)], dest.parent)
    run(["git", "remote", "remove", "origin"], dest)
    # origin con la URL autorizada (solo configuracion; las pruebas no usan la red) y push IMPOSIBLE
    run(["git", "remote", "add", "origin", "https://github.com/GUNAPILLCO/ohlcv_dataroad.git"], dest)
    run(["git", "remote", "set-url", "--push", "origin", "no-push://deshabilitado"], dest)
    for rel in files:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SRC / rel, dest / rel)
    for rel in data:                                       # copia: otro inodo, ninguna escritura llega al original
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SRC / rel, dest / rel)
        assert not os.path.samefile(SRC / rel, dest / rel), rel


def selector(cid: str) -> str:
    return SELECT.get(cid, cid)


def main(work: Path) -> int:
    work.mkdir(parents=True, exist_ok=True)
    files, data = tracked_e02()
    text = (SRC / LOADER).read_text(encoding="utf-8")
    ids = sorted(set(re.findall(r'_fail\("([A-Z0-9_]+)"', text)))
    clone = work / "c"
    log = {"ids": ids, "resultados": {}}
    env = {**os.environ, "PYTHONPATH": str(clone / "src"), "TDA17_INTEGRIDAD": "estricta"}
    make_clone(clone, files, data)
    code, out = run([sys.executable, "-c", "import ohlcv_dataroad.ingest.tda17_enmiendas_e02 as m; print(m.__file__)"], clone, env)
    origin = out.strip()
    assert origin.lower().startswith(str(clone).lower()), f"el modulo no se importa del clon: {origin}"
    log["modulo_importado"] = origin
    t0 = time.time()
    code, out = run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", *TESTS], clone, env)
    log["control"] = {"codigo": code, "resumen": out.strip().splitlines()[-1], "segundos": round(time.time() - t0)}
    print("CONTROL", log["control"], flush=True)
    if code != 0 or " passed" not in log["control"]["resumen"] or "failed" in log["control"]["resumen"] or             "error" in log["control"]["resumen"]:
        print(out[-4000:])
        return 1
    owners = {cid: (LOADER, text, "_fail") for cid in ids}
    for mod in REJECT_MODULES:
        mtext = (SRC / mod).read_text(encoding="utf-8")
        for cid in sorted(set(re.findall(r'_reject\("([A-Z0-9_]+)"', mtext))):
            owners[cid] = (mod, mtext, "_reject")
    if len(sys.argv) > 2:                                  # v6: solo los identificadores indicados
        owners = {c: o for c, o in owners.items() if c in sys.argv[2].split(",")}
        assert sorted(owners) == sorted(sys.argv[2].split(",")), owners
    log["ids"] = list(owners)
    for cid, (target, base, fn) in owners.items():
        mutant = base.replace(f'{fn}("{cid}",', f'_no{fn[1:]}("{cid}",')
        mutant = mutant.replace(f"\ndef {fn}(", f"\ndef _no{fn[1:]}(*_a, **_k):\n    return None\n\n\ndef {fn}(", 1)
        assert mutant != base
        (clone / target).write_text(mutant, encoding="utf-8")
        t0 = time.time()
        code, out = run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", *TESTS, "-k", selector(cid)],
                        clone, env)
        tail = out.strip().splitlines()[-1] if out.strip() else ""
        estado, failed, errors = classify(code, out)
        log["resultados"][cid] = {"estado": estado, "detectado": estado == "DETECTADO", "codigo": code,
                                  "resumen": tail, "pruebas_que_fallan": failed[:12], "errores": errors[:12],
                                  "selector": selector(cid), "modulo": target, "segundos": round(time.time() - t0)}
        if estado == "INFRAESTRUCTURA":
            log["resultados"][cid]["salida"] = out[-3000:]
        print(cid, estado, tail, flush=True)
        (clone / target).write_text(base, encoding="utf-8")
    (work / "t_mut_resultados.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    survivors = [c for c, r in log["resultados"].items() if r["estado"] == "SUPERVIVIENTE"]
    infra = [c for c, r in log["resultados"].items() if r["estado"] == "INFRAESTRUCTURA"]
    print("SUPERVIVIENTES:", survivors)
    print("INFRAESTRUCTURA:", infra)
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
