"""Mutaciones del motor de TDA-17 en una copia aislada del paquete (nunca toca el repositorio)."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(sys.argv[1])
BASE = Path(sys.argv[2]) / "mut_engine"
IND = "ohlcv_dataroad/ingest/tda17_indicators.py"
PRI = "ohlcv_dataroad/ingest/tda17_primitives.py"
MUT = [
    ("M01 VWAP sin reinicio en FJ/FJL", IND, "day = np.cumsum(np.r_[True, b.new_day[1:]])", "day = np.ones(b.n)"),
    ("M02 TR convencional en huecos", IND, "self.tr = np.where(self.cons, conv_tr, self.h - self.l)",
     "self.tr = np.where(np.isfinite(conv_tr), conv_tr, self.h - self.l)"),
    ("M03 ventanas de incrementos sin regla 0,8n", PRI, "valid = (idx >= n) & (np.nan_to_num(count) >= min_valid(n))", "valid = idx >= n"),
    ("M04 Aroon: empate con la aparicion mas antigua", PRI,
     'idx = np.argmax(rev, axis=-1) if kind == "max" else np.argmin(rev, axis=-1)',
     'idx = (w.shape[1] - 1) - (np.argmax(w, axis=-1) if kind == "max" else np.argmin(w, axis=-1))'),
    ("M05 Bollinger con ddof=1", PRI, 'return rolling(x, n, lambda w: w.std(axis=-1))', 'return rolling(x, n, lambda w: w.std(axis=-1, ddof=1))'),
    ("M06 ADX actualiza en todas las barras", IND, "    idx = np.flatnonzero(b.cons)\n    up = b.h[idx]", "    idx = np.arange(1, b.n)\n    up = b.h[idx]"),
    ("M07 Supertrend publica desde el primer ATR", IND, "    first = n - 1 + pr.k_weight(pr.wilder_alpha(n))\n    return {\"dist\"",
     "    first = n - 1\n    return {\"dist\""),
    ("M08 KAMA publica el estado retenido", IND, "            continue                                                        # estado retenido, salida NaN",
     "            out[t] = math.log(b.c[t] / state) if state is not None else np.nan\n            continue"),
    ("M09 RVOL sin exigir k jornadas previas", IND, "complete = (rows >= k)[:, None]", "complete = (rows >= 1)[:, None]"),
    ("M10 MACD: senal desde la barra 1", IND, 'sig, first = _ema_level(line, p["g"], start=k1)', 'sig, first = _ema_level(np.nan_to_num(line), p["g"], start=0)'),
    ("M11 RSI sin el criterio de peso residual", IND, "    U = n + pr.k_weight(alpha)\n    elig = pr.cons_counter(b.cons) >= U\n    return values_at_bars_masked(rsi_upd",
     "    U = n\n    elig = pr.cons_counter(b.cons) >= U\n    return values_at_bars_masked(rsi_upd"),
    ("M12 OBV acumula fuera de CONS", IND, "return np.cumsum(np.where(b.cons, np.sign(np.nan_to_num(b.dC)) * b.v, 0.0))",
     "return np.cumsum(np.sign(np.nan_to_num(b.c - b.c_prev)) * b.v)"),
    ("M13 CLV NaN cuando H = L", IND, "self.clv0 = np.where(self.h > self.l, pr.safe_div((self.c - self.l) - (self.h - self.c), self.h - self.l), 0.0)",
     "self.clv0 = pr.safe_div((self.c - self.l) - (self.h - self.c), self.h - self.l)"),
    ("M14 sigma de retornos con ddof=0", IND, "var = (s2 - s1 ** 2 / m) / (m - 1)", "var = (s2 - s1 ** 2 / m) / m"),
    ("M15 marca rvol_cruza_roll siempre False", IND, "cross = np.isfinite(m_row) & ((cmn != codes) | (cmx != codes))", "cross = np.zeros(len(df), bool)"),
    ("M16 Chaikin normaliza con EMA_f(V)", IND, 'ev, _ = _ema_level(b.v, p["s"])', 'ev, _ = _ema_level(b.v, p["f"])'),
    ("M17 Estocastico publica %K antes que %D", IND, 'return {"k": pr.mask_before(k, first), "d": pr.rolling_mean(k, p["d"])}',
     'return {"k": k, "d": pr.rolling_mean(k, p["d"])}'),
    ("M18 HMA con media simple final", IND, 'hma = pr.rolling_wma(raw, p["n_raiz"])', 'hma = pr.rolling_mean(raw, p["n_raiz"])'),
    ("M19 TRIX publica una barra antes", IND, "pr.mask_before(trix, k3 + 1)", "pr.mask_before(trix, k3)"),
    ("M20 UO con pesos intercambiados", IND, '100.0 * (4 * avg(p["a"]) + 2 * avg(p["b"]) + avg(p["c"])) / 7.0',
     '100.0 * (2 * avg(p["a"]) + 4 * avg(p["b"]) + avg(p["c"])) / 7.0'),
    ("M21 C1 sin comprobar el hash del protocolo", IND, "    if digest != FROZEN_YAML_SHA256:", "    if False:"),
    ("M22 C1 produccion acepta marcos sinteticos", IND, "    if SYNTHETIC_MARKER in df.columns:\n        raise", "    if False:\n        raise"),
    ("M23 E01-A activacion con n DX finitas (sin k)", IND, "    active = count >= n + pr.k_weight(alpha)", "    active = count >= n"),
    ("M24 C3 J(t) con las fechas presentes en los datos", IND, "    trading_days = calendar_trading_days(df)\n", "    trading_days = list(pd.unique(df['trading_date']))\n"),
    ("M25 C4 omite el par de la primera barra (contraejemplo CHOP)", IND, "lo = np.maximum(t - n_pairs + 1, 0)", "lo = np.maximum(t - n_pairs + 2, 0)"),
    ("M26 C4 cruza_hueco_largo ignora los HI largos", IND, '"cruza_hueco_largo": (tipo_par == PAIR_FJL) | ((tipo_par == PAIR_HI) & long_hi)}',
     '"cruza_hueco_largo": (tipo_par == PAIR_FJL)}'),
    ("M27 C4 marcas sin reinicio por roll", IND, "    for a, z in segs:\n        n = z - a\n        if n < n_bars:",
     "    for a, z in [(0, N)]:\n        n = z - a\n        if n < n_bars:"),
    ("M28 C4 ventana K-1 evaluada como K", IND, '"K-1": lambda p, K: K - 1,', '"K-1": lambda p, K: K,'),
    ("M29 C4 marca por serie con la ventana menor", IND, "    _, prefix = max(cands)", "    _, prefix = min(cands)"),
    ("M30 E01-A DX no definido := 0 (opcion C)", IND, "wilder_finite_inputs(dx[u1 - 1:], n, alpha)", "wilder_finite_inputs(np.nan_to_num(dx[u1 - 1:]), n, alpha)"),
    ("M31 E01-A opcion A literal en el motor", IND, "    finite = np.isfinite(x)\n", "    finite = np.ones(x.size, bool)\n"),
    ("M32 E01-B sesion de VWAP sin reinicio por jornada", IND, "opens = np.isin(tipo_par, [PAIR_REINICIO, PAIR_FJ, PAIR_FJL])", "opens = np.isin(tipo_par, [PAIR_REINICIO])"),
    ("M33 E01-B sesion ignora HI largos", IND, "n_long = pd.Series(hi * long_hi)", "n_long = pd.Series(hi * 0)"),
    ("M34 E01-B VWAP sin marca de sesion por serie", IND, '    if spec.get("sesion"):\n', '    if False:\n'),
    ("M35 E01-B sesion cuenta el par de apertura", IND, 'hi = (tipo_par == PAIR_HI).astype(np.int64)', 'hi = np.isin(tipo_par, [PAIR_HI, PAIR_FJ, PAIR_FJL]).astype(np.int64)'),
    ("M36 integridad: sin exigir aprobacion", "ohlcv_dataroad/ingest/tda17_enmiendas.py", '    if require_approved and entry.get("estado") != ESTADO_APROBADA:', "    if False:"),
    ("M37 integridad: sin huella autorizada", "ohlcv_dataroad/ingest/tda17_enmiendas.py", '    if digest != auth["sha256"]:', '    if False:'),
    ("M38 integridad: sin comprobar los 9 ficheros", "ohlcv_dataroad/ingest/tda17_enmiendas.py", '    if errors:\n        raise ProtocolNotAuthorizedError("ficheros protegidos', '    if False:\n        raise ProtocolNotAuthorizedError("ficheros protegidos'),
    ("M39 integridad: sin huella del manifiesto", "ohlcv_dataroad/ingest/tda17_enmiendas.py", "    if digest != FROZEN_MANIFEST_SHA256:", "    if False:"),
    ("M40 integridad: sin lista blanca", "ohlcv_dataroad/ingest/tda17_enmiendas.py", "    if outside:", "    if False:"),
    ("M41 integridad: sin control de rutas cambiadas", "ohlcv_dataroad/ingest/tda17_enmiendas.py", "    if changed != allowed_paths:", "    if False:"),
    ("M42 integridad: acepta enmiendas no autorizadas", "ohlcv_dataroad/ingest/tda17_enmiendas.py", "    if unknown:", "    if False:"),
    ("M43 integridad: git ignora enmiendas y registro", "ohlcv_dataroad/ingest/tda17_enmiendas.py", "    for rel in AMENDMENT_TRACKED_FILES:", "    for rel in []:"),
    ("M44 integridad: registro sin la huella autorizada", "ohlcv_dataroad/ingest/tda17_enmiendas.py", '    if entry.get("sha256") != auth["sha256"]:', '    if False:'),
    ("M45 motor sin exigir E01", IND, '    if (a8.get("dx_no_definido") or {}).get("regla") != "OMITIR_EN_ETAPA_2":', '    if False:'),
    ("M46 integridad: sustituir sin comprobar cambio", "ohlcv_dataroad/ingest/tda17_enmiendas.py", "(key not in parent or parent[key] == o[\"valor\"])", "(key not in parent)"),
]
MUT += [
    ("M47 F1 git/aprobacion sin el motor de indicadores", "ohlcv_dataroad/ingest/tda17_enmiendas.py", '    "src/ohlcv_dataroad/ingest/tda17_indicators.py",\n', ""),
    ("M48 F2 fecha sin control de rango", "ohlcv_dataroad/ingest/tda17_enmiendas.py", "    if not prep <= fecha <= (today or dt.date.today()):", "    if False:"),
    ("M49 F2 commit auditado sin control de formato", "ohlcv_dataroad/ingest/tda17_enmiendas.py",
     '    if not isinstance(ref["commit_auditado"], str) or not _HEX40.fullmatch(ref["commit_auditado"]):', "    if False:"),
    ("M50 F2 sin comparar huellas auditadas con el arbol", "ohlcv_dataroad/ingest/tda17_enmiendas.py", "    if changed:\n        fail(", "    if False:\n        fail("),
    ("M51 F2 commit auditado sin comparar huellas", "ohlcv_dataroad/ingest/tda17_enmiendas.py",
     '        elif frz.sha256_bytes(shown.stdout, Path(rel).suffix) != a["sha256_auditados"][rel]:', "        elif False:"),
    ("M52 S2 ejecutable sin asignacion S/M/L", "ohlcv_dataroad/ingest/tda17_enmiendas.py", "    if not scales:\n        raise", "    if False:\n        raise"),
    ("M53 F2 aprobacion sin validar", "ohlcv_dataroad/ingest/tda17_enmiendas.py", "        check_approval(root, entry, amendment_id)\n", "        pass\n"),
    ("M54 F2 responsable sin control", "ohlcv_dataroad/ingest/tda17_enmiendas.py", '    if not isinstance(a["aprobada_por"], str) or not a["aprobada_por"].strip():', "    if False:"),
]
MUT += [
    ("M55 F2 produccion sin barrera Git del commit auditado", "ohlcv_dataroad/ingest/tda17_enmiendas.py",
     "        require_audited_commit(root, entry, amendment_id)\n", "        pass\n"),
    ("M56 F2 Git sin comprobar la raiz del repositorio", "ohlcv_dataroad/ingest/tda17_enmiendas.py", "    if not same:\n        return", "    if False:\n        return"),
    ("M57 F2 borrador activable", "ohlcv_dataroad/ingest/tda17_enmiendas.py", '        "activable": require_approved,', '        "activable": True,'),
]
ONLY = sys.argv[3] if len(sys.argv) > 3 else None                      # filtro opcional por prefijo (p. ej. M46)
if ONLY:
    MUT = [m for m in MUT if m[0].split()[0] in ONLY.split("+")]
    assert MUT, ONLY
tests = ["tests/test_tda17_indicators.py", "tests/test_tda17_enmienda_01.py"]
def run(name, rel=None, old=None, new=None):
    if BASE.exists():
        shutil.rmtree(BASE)
    shutil.copytree(REPO / "src", BASE / "src")
    shutil.copytree(REPO / "configs", BASE / "configs")                 # el motor resuelve ROOT desde su ruta
    (BASE / "reports" / "mnq").mkdir(parents=True)
    for f in ["reports/mnq/TDA17_manifiesto_protocolo.json", "reports/mnq/TDA17_enmiendas.json",
                "reports/mnq/TDA17_protocolo_congelamiento.md", "tests/test_tda17_protocol.py",
                "tests/test_tda17_calentamiento_huecos.py", "tests/test_tda17_diseno_v4a_sintetico.py",
                "tests/test_tda17_diseno_v4b_g23b_viabilidad.py", "tests/test_tda17_g2_2_generador.py",
                "tests/test_tda17_supertrend.py", "tests/test_tda17_indicators.py",
                "tests/test_tda17_enmienda_01.py", "tests/tda17_synthetic.py"]:
        (BASE / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / f, BASE / f)
    if rel:
        p = BASE / "src" / rel
        t = p.read_text(encoding="utf-8")
        assert t.count(old) == 1, name
        p.write_text(t.replace(old, new), encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(BASE / "src"))
    r = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *tests, "--tb=no", "-q", "-rf"],
                       cwd=REPO, capture_output=True, text=True, env=env)
    failed = [l.split("::")[1].split(" ")[0] for l in r.stdout.splitlines() if l.startswith("FAILED")]
    errors = [l for l in r.stdout.splitlines() if l.startswith("ERROR")]
    return r, failed, errors


import datetime as _dt
import hashlib as _hl
import json as _json
EVID = Path(sys.argv[2]) / (f"mut_evidencia_{ONLY}.jsonl" if ONLY else "mut_evidencia.jsonl")
_ev = EVID.open("w", encoding="utf-8")
def _sha(path):
    return _hl.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
_ev.write(_json.dumps({"comando": [sys.executable, *sys.argv], "inicio_utc": _dt.datetime.utcnow().isoformat(timespec="seconds"),
                       "sha256_arnes": _sha(__file__),
                       "sha256_repo": {f: _sha(REPO / f) for f in ["src/ohlcv_dataroad/ingest/tda17_indicators.py",
                           "src/ohlcv_dataroad/ingest/tda17_primitives.py", "src/ohlcv_dataroad/ingest/tda17_enmiendas.py",
                           "tests/test_tda17_indicators.py", "tests/test_tda17_enmienda_01.py", "tests/tda17_synthetic.py",
                           "configs/tda17_enmienda_01.yaml", "reports/mnq/TDA17_enmiendas.json"]},
                       "pytest": [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *tests, "--tb=no", "-q", "-rf"]},
                      ensure_ascii=False) + "\n")
r, failed, errors = run("CONTROL")
probe = subprocess.run([sys.executable, "-c", "import ohlcv_dataroad.ingest.tda17_indicators as m; print(m.__file__)"],
                       capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(BASE / "src")))
assert str(BASE) in probe.stdout, "la copia aislada no se esta importando"
print("CONTROL sin mutacion:", r.stdout.strip().splitlines()[-1], "| importa:", probe.stdout.strip())
_ev.write(_json.dumps({"mutacion": "CONTROL", "resultado": r.stdout.strip().splitlines()[-1], "fallan": failed}) + "\n")
assert r.returncode == 0 and not failed and not errors, "el control debe pasar"
detected = 0
for name, rel, old, new in MUT:
    r, failed, errors = run(name, rel, old, new)
    assert not errors, (name, errors[:3])                               # un error de coleccion NO cuenta como deteccion
    ok = len(failed) > 0
    detected += ok
    print(f"{'DETECTADA   ' if ok else 'NO DETECTADA'} {name}: {len(failed)} pruebas fallan; p. ej. {failed[:2]}")
    _ev.write(_json.dumps({"mutacion": name, "fichero": "src/" + rel, "original": old, "mutado": new,
                           "resultado_pytest": r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "",
                           "fallan": failed, "detectada": ok}, ensure_ascii=False) + "\n")
shutil.rmtree(BASE)
print(f"{detected}/{len(MUT)} mutaciones detectadas")
_ev.write(_json.dumps({"resumen": f"{detected}/{len(MUT)} mutaciones detectadas",
                       "fin_utc": _dt.datetime.utcnow().isoformat(timespec="seconds")}) + "\n")
_ev.close()
