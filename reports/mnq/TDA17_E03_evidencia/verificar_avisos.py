"""Clasifica los avisos de los logs pytest de R4 (seccion ``warnings summary``) y comprueba que los del hilo de envio
son solo el ``BufferError`` de la memoryview abandonada (ver ``repro_aviso_hilo_envio.py``) y que no hay excepciones
de hilos (``PytestUnhandledThreadExceptionWarning``) ni de ``e03-pipe``.
Uso: python verificar_avisos.py LOG... > avisos.json ; rc=0 si todos los logs cumplen."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ESPERADO = ("BufferError: Existing exports of data: object cannot be re-sized", "_io.BytesIO")


def entradas(texto: str):
    m = re.search(r"^=+ warnings summary =+$(.*?)^-- Docs:", texto, re.S | re.M)
    if not m:
        return []
    out, tests, cuerpo = [], [], []
    for linea in m.group(1).splitlines():
        if not linea.strip():
            cuerpo.append("")
            continue
        if not linea.startswith(" "):
            if cuerpo and any(c.strip() for c in cuerpo):
                out.append((tests, "\n".join(cuerpo)))
                tests, cuerpo = [], []
            tests.append(linea.strip())
        else:
            cuerpo.append(linea)
    if tests:
        out.append((tests, "\n".join(cuerpo)))
    return out


def analizar(log: Path) -> dict:
    texto = log.read_text(encoding="utf-8", errors="replace")
    fin = [l for l in texto.splitlines() if l.startswith("rc=")]
    resumen = [l for l in texto.splitlines() if re.search(r"\d+ (passed|failed)", l) and " in " in l]
    hilo_envio, otros, problemas = [], Counter(), []
    for tests, cuerpo in entradas(texto):
        if "PytestUnhandledThreadExceptionWarning" in cuerpo or "e03-pipe" in cuerpo:
            problemas.append({"tests": tests, "texto": cuerpo[:600]})
        elif "PytestUnraisableExceptionWarning" in cuerpo:
            ok = all(e in cuerpo for e in ESPERADO)
            for t in tests:
                hilo_envio.append({"test": t, "buffererror_bytesio": ok})
            if not ok:
                problemas.append({"tests": tests, "texto": cuerpo[:600]})
        else:
            cat = re.search(r"(\w+Warning)", cuerpo)
            otros[cat.group(1) if cat else "?"] += len(tests)
    return {"log": log.name, "rc": fin[-1] if fin else None, "resumen": resumen[-1].strip("= ") if resumen else None,
            "avisos_hilo_envio": len(hilo_envio), "tests_con_aviso_hilo_envio": sorted(Counter(
                h["test"] for h in hilo_envio).items()), "todos_buffererror_bytesio": all(
                h["buffererror_bytesio"] for h in hilo_envio), "otros_avisos_por_categoria": dict(otros),
            "problemas": problemas, "cumple": not problemas and bool(fin) and fin[-1] == "rc=0"}


if __name__ == "__main__":
    res = [analizar(Path(p)) for p in sys.argv[1:]]
    print(json.dumps(res, indent=2, ensure_ascii=False))
    sys.exit(0 if all(r["cumple"] for r in res) else 1)
