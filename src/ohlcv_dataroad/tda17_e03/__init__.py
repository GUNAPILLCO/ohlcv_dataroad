"""TDA-17 -- espacio de nombres PROPIO de la enmienda E03 (motor paralelo; solo implementacion).

Modulos (todos auditados por E03, ``enmienda.AUDITED_E03``):
- ``motor``: motor paralelo de TH37 -> TH35 (mismo resultado que ``tda17_analysis.th35_pipeline``, byte a byte).
- ``enmienda``: registro, aprobacion (commit auditado A3, publicacion P3) y ruta oficial con el motor E03.

E03 vive fuera de ``ohlcv_dataroad/ingest`` para no alterar ni debilitar la guarda estatica auditada de E02
(``tests/test_tda17_enmienda_02.py::test_static_production_guards``, que cubre los modulos ``ingest/tda17_*.py`` de
E01/E02). La cobertura de TODO el codigo de TDA-17 (E01, E02, protocolo y E03, en cualquier directorio) la exige la
guarda propia de E03 (``tests/tda17_e03/test_e03_cobertura.py``). Ningun modulo de E01/E02 importa este paquete.
"""
