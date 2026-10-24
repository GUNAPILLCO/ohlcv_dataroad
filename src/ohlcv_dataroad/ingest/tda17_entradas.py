"""TDA-17 -- fase 2: UNICA via de lectura de las entradas de la ejecucion oficial (procedencia verificable).

La ejecucion oficial no acepta datos del llamante: lee ella misma las dos entradas de TDA-04 declaradas en el
manifiesto congelado (``sha256_entradas``), calcula la huella SHA-256 de los MISMOS bytes que interpreta (sin ventana
entre verificar y leer) y exige que coincida con la del manifiesto; despues une variables y mascara
(``tda17_indicators.join_inputs``) y comprueba ``entradas.esperado`` (``tda17_analysis.check_expected_inputs``). La
procedencia devuelta (huellas de los ficheros, huella del marco unido, P2 y configuracion efectiva) acompana a todos
los resultados oficiales.

Solo se usa con la configuracion de la ejecucion oficial (``require_e02_oficial``, que vuelve a derivarla de M2). No
lee data/raw ni el holdout (las entradas de TDA-04 se construyeron solo sobre el conjunto de investigacion).
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pandas as pd

from ohlcv_dataroad.ingest import tda17_analysis as an
from ohlcv_dataroad.ingest import tda17_enmiendas_e02 as m2
from ohlcv_dataroad.ingest import tda17_indicators as ind
from ohlcv_dataroad.ingest import tda17_protocol_freeze as frz

ProtocolNotAuthorizedError = ind.ProtocolNotAuthorizedError
VARIABLES = "data/interim/mnq/tda04_variables_1m.parquet"
VALIDITY = "data/interim/mnq/tda04_return_validity_mask.parquet"
INPUTS = (VARIABLES, VALIDITY)


def _reject(cid: str, msg: str):
    raise ProtocolNotAuthorizedError(f"[{cid}] {msg}")


def expected_input_hashes(root: Path) -> dict[str, str]:
    """Huellas de las entradas en el manifiesto congelado (cuya huella fija verifica el cargador de E01)."""
    m = frz.load_manifest(root)
    out = {}
    for rel in INPUTS:
        h = m.get("sha256_entradas", {}).get(rel)
        if not isinstance(h, str) or len(h) != 64:
            _reject("ENTRADA_NO_DECLARADA", f"{rel} no figura en sha256_entradas del manifiesto")
        out[rel] = h
    return out


def read_verified(root: Path, rel: str, expected: str) -> bytes:
    """Bytes de ``rel`` cuya huella (calculada sobre esos mismos bytes) es ``expected``; si no, rechazo."""
    if rel.startswith("data/raw") or "holdout" in rel.lower():
        _reject("ENTRADA_PROHIBIDA", f"ruta de entrada prohibida: {rel}")
    path = Path(root) / rel
    if not path.is_file():
        _reject("ENTRADA_PROCEDENCIA", f"falta la entrada {rel}")
    data = path.read_bytes()
    digest = frz.sha256_bytes(data, path.suffix)
    if digest != expected:
        _reject("ENTRADA_PROCEDENCIA", f"{rel} no es la entrada del manifiesto (SHA-256 {digest})")
    return data


def frame_fingerprint(df: pd.DataFrame) -> str:
    """Huella del marco unido (columnas, tipos y valores; independiente del indice)."""
    h = hashlib.sha256()
    h.update(repr([(c, str(t)) for c, t in df.dtypes.items()]).encode("utf-8"))
    h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return h.hexdigest()


def load_official_inputs(cfg: dict, root: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Entradas de la ejecucion oficial: configuracion oficial verificada, bytes con la huella del manifiesto, union
    variables + mascara y ``entradas.esperado``. Devuelve (marco, procedencia)."""
    root = Path(m2.ROOT if root is None else root)
    m2.require_e02_oficial(cfg, root)
    expected = expected_input_hashes(root)
    frames = {rel: pd.read_parquet(io.BytesIO(read_verified(root, rel, h))) for rel, h in expected.items()}
    df = ind.join_inputs(frames[VARIABLES], frames[VALIDITY])
    an.check_expected_inputs(df, cfg)
    tr = cfg["trazabilidad_protocolo"]
    return df, {"entradas": expected, "huella_marco": frame_fingerprint(df), "filas": int(len(df)),
                "publicacion_E02": tr["publicacion_E02"], "sha256_efectiva": tr["sha256_efectiva"]}
