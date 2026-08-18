"""Punto de entrada de linea de comandos para ejecutar TDA-00.

Uso
---
    python -m ohlcv_dataroad.ingest.run_tda00
    python -m ohlcv_dataroad.ingest.run_tda00 --config configs/mnq_snapshot.yaml

Este script deliberadamente NO contiene logica de negocio: su unico
trabajo es (1) cargar la configuracion, (2) invocar ``run_tda00`` de
``tda00_integrity.py``, y (3) traducir un fallo de pipeline (archivo
faltante, fallo de conservacion, violacion de aislamiento del hold-out) en
un mensaje de salida claro con codigo de salida distinto de cero, en vez
de dejar que una traza de Python cruda sea el unico resultado visible al
ejecutarlo desde una terminal.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ohlcv_dataroad.config import load_config
from ohlcv_dataroad.ingest.tda00_integrity import HoldoutIsolationError, run_tda00


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ejecuta TDA-00 (integridad intra-barra e invariantes fisicos) "
        "sobre el conjunto de investigacion de MNQ."
    )
    parser.add_argument(
        "--config",
        default="configs/mnq_snapshot.yaml",
        help="Ruta al YAML de configuracion del snapshot (por defecto: configs/mnq_snapshot.yaml).",
    )
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))

    try:
        stats = run_tda00(config)
    except (FileNotFoundError, AssertionError, HoldoutIsolationError) as exc:
        print("TDA-00 status: FAIL", file=sys.stderr)
        print(f"Motivo: {exc}", file=sys.stderr)
        return 1

    print(f"TDA-00 status: {stats['status']}")
    print(f"Archivos procesados (conjunto de investigacion): {stats['n_files']}")
    print(f"Lineas totales leidas: {stats['total_lines']:,}")
    print(f"Filas bien formadas: {stats['parsed_rows']:,}")
    print(f"Errores de parseo: {stats['parse_error_rows']:,}")
    print(f"Filas con violacion de invariante duro: {stats['hard_violation_rows']:,} "
          f"({stats['fraction_hard_violation_rows']*100:.6f} %)")
    print(f"Barras con volumen = 0 (observacional, no es violacion): {stats['zero_volume_rows']:,}")
    print(f"Rango temporal: {stats['first_timestamp']} -> {stats['last_timestamp']}")
    print("")
    print(f"Informe de inventario : {config.inventory_report_path}")
    print(f"Violaciones (csv)      : {config.violations_report_path}")
    print(f"Resumen por archivo    : {config.per_file_summary_path}")
    print(f"Mascara bad_data       : {config.bad_data_mask_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
