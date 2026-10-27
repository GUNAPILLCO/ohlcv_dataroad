"""Guarda de COBERTURA EXHAUSTIVA de E03 y garantias estaticas.

La guarda estatica auditada de E02 (``tests/test_tda17_enmienda_02.py::test_static_production_guards``) exige que todo
``src/ohlcv_dataroad/ingest/tda17_*.py`` y todo ``tests/test_tda17_*.py`` este auditado por E01/E02/protocolo. E03 vive
en su espacio de nombres propio (``src/ohlcv_dataroad/tda17_e03``, ``tests/tda17_e03``) para no modificar ni debilitar
esa guarda, y esta guarda propia es MAS AMPLIA: recorre TODO ``src/``, ``tests/`` y ``configs/`` (cualquier
profundidad) y exige que cada fichero relacionado con TDA-17 -- por su ruta o porque su contenido nombra un modulo de
TDA-17 -- este auditado por exactamente una de E01, E02, el protocolo congelado o E03. Ademas: el espacio de nombres de
E03 contiene exactamente sus ficheros auditados (ningun modulo oculto, compilado o enlace), E01/E02 no importan E03, los
modulos importados son los ficheros auditados del repositorio y la guarda de E02 sigue intacta y en verde.
"""
from __future__ import annotations

import importlib
import inspect
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))             # tests/ (guarda de E02)

from ohlcv_dataroad.ingest import tda17_enmiendas as e01  # noqa: E402
from ohlcv_dataroad.ingest import tda17_enmiendas_e02 as m2  # noqa: E402
from ohlcv_dataroad.ingest import tda17_indicators as ind  # noqa: E402
from ohlcv_dataroad.ingest import tda17_protocol_freeze as frz  # noqa: E402
from ohlcv_dataroad.tda17_e03 import enmienda as m3  # noqa: E402
from ohlcv_dataroad.tda17_e03 import motor as mo  # noqa: E402

SRC = Path(m2.ROOT)
P2_REAL = "ff224dc7e30f7631dc46eea17db5a60dd44b07cc"
_TDA17 = re.compile(r"tda17", re.I)
_EXCLUDED_DIRS = {"__pycache__", ".pytest_cache"}


def _files(base: str) -> list[str]:
    root = SRC / base
    return sorted(p.relative_to(SRC).as_posix() for p in root.rglob("*")
                  if p.is_file() and not _EXCLUDED_DIRS & set(p.relative_to(SRC).parts))


def _related(rel: str) -> bool:
    """Relacionado con TDA-17: la ruta lo nombra o, si es codigo o configuracion, su contenido nombra TDA-17."""
    if _TDA17.search(rel):
        return True
    if Path(rel).suffix in (".py", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".pth"):
        return bool(_TDA17.search((SRC / rel).read_text(encoding="utf-8", errors="replace")))
    return False


def _covered() -> dict[str, str]:
    cov = {}
    for owner, rels in (("E01", m2.E01_FILES_SHA256), ("E02", m2.TRACKED_E02), ("PROTOCOLO", frz.PROTOCOL_FILES),
                        ("E03", m3.TRACKED_E03)):
        for r in rels:
            assert r not in cov, f"{r} auditado por {cov.get(r)} y por {owner}"
            cov[r] = owner
    return cov


def test_every_tda17_file_in_the_repository_is_audited_exactly_once():
    cov = _covered()
    universe = [r for base in ("src", "tests", "configs") for r in _files(base) if _related(r)]
    universe += [r for r in _files("reports/mnq") if re.fullmatch(r"reports/mnq/TDA17_enmiendas[^/]*\.json", r)]
    missing = sorted(set(universe) - set(cov))
    assert missing == [], f"ficheros de TDA-17 sin auditar: {missing}"
    assert all((SRC / r).is_file() for r in cov), [r for r in cov if not (SRC / r).is_file()]


def test_the_e03_namespace_contains_exactly_its_audited_files():
    """Ningun modulo oculto, compilado, de datos ni enlace en el espacio de nombres de E03."""
    for base in (m3.NAMESPACE, m3.TESTS):
        found = _files(base)
        assert found == sorted(r for r in m3.AUDITED_E03 if r.startswith(base + "/")), found
        assert not [p for p in (SRC / base).rglob("*") if p.is_symlink()]
    assert set(m3.AUDITED_E03) - set(_files(m3.NAMESPACE) + _files(m3.TESTS)) == {m3.E03_YAML}
    assert m3.TRACKED_E03 == [m3.REGISTRY_E03] + m3.AUDITED_E03 and m3.REGISTRY_E03 not in m3.AUDITED_E03


def test_no_compiled_extension_or_path_hook_can_shadow_the_package():
    pkg = SRC / "src" / "ohlcv_dataroad"
    odd = [p for p in pkg.rglob("*") if p.is_file() and not _EXCLUDED_DIRS & set(p.parts) and
           p.suffix in (".pyc", ".pyd", ".so", ".pth", ".dll", ".zip", ".egg")]
    assert odd == []
    for name in ("ohlcv_dataroad.tda17_e03", "ohlcv_dataroad.tda17_e03.motor", "ohlcv_dataroad.tda17_e03.enmienda"):
        mod = importlib.import_module(name)
        rel = Path(mod.__file__).resolve().relative_to(SRC.resolve()).as_posix()
        assert rel in m3.AUDITED_E03, (name, rel)


def test_e01_e02_and_the_protocol_never_import_e03():
    for rel in list(m2.E01_FILES_SHA256) + m2.TRACKED_E02 + frz.PROTOCOL_FILES:
        assert "tda17_e03" not in (SRC / rel).read_text(encoding="utf-8", errors="replace"), rel
    others = [r for r in _files("src") if r.endswith(".py") and not r.startswith(m3.NAMESPACE + "/")]
    assert [r for r in others if "tda17_e03" in (SRC / r).read_text(encoding="utf-8", errors="replace")] == []


def test_the_audited_e02_static_guard_is_unchanged_and_passes():
    reg = json.loads((SRC / m2.REGISTRY_E02).read_text(encoding="utf-8"))
    rel = "tests/test_tda17_enmienda_02.py"
    assert e01.sha256_text_file(SRC / rel) == reg["enmiendas"][0]["aprobacion"]["sha256_auditados"][rel]
    import test_tda17_enmienda_02 as t02
    t02.test_static_production_guards()


def test_e02_e01_and_the_protocol_files_are_untouched():
    reg = json.loads((SRC / m2.REGISTRY_E02).read_text(encoding="utf-8"))
    for rel, h in reg["enmiendas"][0]["aprobacion"]["sha256_auditados"].items():
        assert e01.sha256_text_file(SRC / rel) == h, rel
    for rel in m2.TRACKED_E02:
        assert m2._hash_at(SRC, P2_REAL, rel) == e01.sha256_text_file(SRC / rel), rel
    for rel, h in m2.E01_FILES_SHA256.items():
        assert e01.sha256_text_file(SRC / rel) == h, rel
    assert frz.verify_against_git(SRC, ind.FROZEN_COMMIT) == []


def test_e03_never_reads_raw_data_or_the_holdout():
    for rel in m3.AUDITED_E03:
        if rel.startswith(m3.NAMESPACE):
            src = (SRC / rel).read_text(encoding="utf-8")
            for token in ("data/raw", "holdout", "read_parquet", "read_csv", "data/interim"):
                assert token not in src, (rel, token)


def test_static_governance_guards_of_e03():
    src = (SRC / m3.NAMESPACE / "enmienda.py").read_text(encoding="utf-8")
    assert "_DISABLED" not in src and "environ" not in src and "getenv" not in src
    ids = set(re.findall(r'_fail\("([A-Z0-9_]+)"', src))
    assert ids == {"GIT", "E03_REGISTRO", "E03_REG_ESTRUCTURA", "E03_BASE", "E03_REQUIERE", "E03_ENTRADA", "E03_DOC",
                   "E03_CAMPOS", "E03_FECHA", "E03_RESPONSABLE", "E03_REFERENCIA", "E03_COMMIT_FORMATO",
                   "E03_COBERTURA", "E03_ARBOL", "E03_COMMIT", "E03_ASCENDENCIA", "E03_A3_HEAD", "E03_HUELLA",
                   "E03_ESTADO", "E03_PENDIENTE", "E03_PENDIENTE_CON_DATOS", "E03_REMOTO_NO_AUTORIZADO", "E03_REMOTO",
                   "E03_REMOTO_HISTORIA", "E03_P3_FORMATO", "E03_P3_EXISTE", "E03_P3_A3", "E03_P3_HEAD",
                   "E03_P3_ARBOL", "E03_P3_APROBADA", "E03_EJECUCION_EN_REPOSITORIO", "E03_EJECUCION_EXISTENTE",
                   "E03_ENTORNO", "E03_SALIDA_RECUPERACION", "E03_COMPROBANTE", "E03_COMPROBANTE_VINCULO"}, ids
    official = inspect.getsource(m3.official_e03)
    order = [official.index(t) for t in ("E03_P3_FORMATO", "load_e03_aprobada(root)", "E03_P3_EXISTE", "E03_P3_A3",
                                         "E03_P3_HEAD", "E03_P3_ARBOL", "E03_P3_APROBADA",
                                         "verify_remote_publication_e03(", "an.official_config(E02_P2, root)")]
    assert order == sorted(order)
    run = inspect.getsource(m3._official_run_e03)
    order = [run.index(t) for t in ("_check_run_directory(", "E03_ENTORNO", "official_e03(root, publicacion)",
                                    "antes_del_calculo(cfg)", "mo.hilos_blas_fijados(HILOS_BLAS_OFICIALES)",
                                    "ent.load_official_inputs(", "oficial=True")]
    assert order == sorted(order)
    # publicacion: comprobante de E03 en la MISMA transaccion de E02, despues de las tablas y antes del registro
    pub = inspect.getsource(m3._publish_files_e03)
    assert pub.index("**blobs") < pub.index("COMPROBANTE:") < pub.index("sal.PROVENANCE:") < pub.index("sal._exclusive(")
    assert pub.index("_destino_libre(out_dir)") < pub.index("sal._stage_and_promote(out_dir, files)")
    rec = inspect.getsource(m3._recover_e03)
    assert rec.index("sal._owned(") < rec.index("sal._recover(out_dir)") < rec.index("comp.unlink()")
    assert "write_bytes" not in inspect.getsource(m3.official_publish_th35_e03)    # nada fuera de la transaccion
    chk = inspect.getsource(m3.check_official_publication_e03)
    assert chk.index("official_e03(root, publicacion)") < chk.index("sal.check_official_publication(E02_P2, root)") \
        < chk.index("_check_comprobante(")
    for f in (m3.reproduce_official_publication_e03, m3.reproduce_with_e02_engine_e03):
        assert "check_official_publication_e03(publicacion, root)" in inspect.getsource(f)
    assert "memoria_estricta=True" in inspect.getsource(m3.official_resources)
    assert "reanudar" not in run and "reanudar" not in inspect.getsource(m3.official_publish_th35_e03)
    assert m3.HILOS_BLAS_OFICIALES == 1 and m3.REANUDACION_OFICIAL == "DESHABILITADA"
    assert not set(m3.AUDITED_E03) & (set(m2.TRACKED_E02) | set(m2.E01_FILES_SHA256) | set(frz.PROTOCOL_FILES))


def test_static_guards_of_the_engine():
    src = (SRC / m3.NAMESPACE / "motor.py").read_text(encoding="utf-8")
    assert "compute_catalog_synthetic" not in src and "._compute(" not in src
    assert 'mmap_mode="r"' in src and 'mmap_mode="c"' not in src and 'mmap_mode="r+"' not in src
    assert inspect.getsource(mo.Ejecucion.guardado).count("self.oficial") == 1
    # memoria: el balance se evalua SIEMPRE (tambien sin activas); en modo estricto nunca se lanza sin caber
    cabe = inspect.getsource(mo._Motor._cabe)
    assert "return True" not in cabe and cabe.rstrip().endswith('return self._balance_memoria(t, w)["cabe"]')
    sin = inspect.getsource(mo._Motor._sin_memoria_y_sin_activas)
    assert sin.index("if not self.r.memoria_estricta:") < sin.index("_fail(\"E03_MEMORIA_INSUFICIENTE\"")
    assert "E03_RECURSOS_OFICIALES" in inspect.getsource(mo.run_th35_e03)
    # R4: ninguna tarea antes de que el worker confirme su arranque; envio y recepcion siempre con plazo
    trab = inspect.getsource(mo._trabajador)
    assert trab.index('conn.send(("listo", os.getpid(), None, None))') < trab.index("conn.recv()")
    desp = inspect.getsource(mo._Motor._despachar)
    assert "conn.send" not in desp and desp.index("if w.listo:") < desp.index("self._enviar(w)")
    assert "self._transferir(w, lambda: w.conn.send(msg)" in inspect.getsource(mo._Motor._enviar)
    proc = inspect.getsource(mo._Motor._ejecutar_procesos)
    assert "conn.recv()" not in proc and "self._transferir(w, w.conn.recv," in proc and "conn.poll()" not in proc
    assert src.count("conn.recv()") == 1                                  # solo el bucle del worker
