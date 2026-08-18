"""Protección común del hold-out, reutilizada por TDA-00, TDA-01 y cualquier
etapa futura del roadmap que procese exclusivamente ``config.research_files``.

Este módulo existe porque la protección del hold-out no es un detalle de
una etapa concreta: es una regla de gobernanza transversal
(`docs/instruments/mnq/MNQ_DATA_PRIOR_KNOWLEDGE.md`, sección 11 —
HOLDOUT GOVERNANCE). Antes de esta corrección, TDA-00 tenía su propia
implementación de estas dos comprobaciones y TDA-01 no tenía ninguna —
duplicar la lógica en cada etapa nueva habría sido, tarde o temprano, una
fuente de inconsistencia (una etapa con la protección ligeramente distinta
de otra). Centralizarla aquí garantiza que **todas** las etapas que la usan
comparten exactamente la misma regla.

Las dos comprobaciones, en el orden en que deben invocarse:

1. :func:`validate_research_holdout_disjoint` -- se llama ANTES de abrir
   cualquier archivo. Es aritmética pura sobre las listas de nombres de la
   configuración.
2. :func:`validate_last_timestamps_before_boundary` -- se llama DESPUÉS de
   parsear el conjunto de investigación, usando exclusivamente los
   timestamps máximos ya calculados por cada etapa a partir de sus propios
   archivos de investigación. Nunca abre un archivo de
   ``config.holdout_files``: la frontera es un valor de configuración, no
   algo observado en el hold-out.
"""
from __future__ import annotations

import pandas as pd

from ohlcv_dataroad.config import SnapshotConfig


class HoldoutIsolationError(Exception):
    """Se dispara cuando una validación de aislamiento del hold-out falla.

    Cubre dos comprobaciones distintas, ambas defensivas frente a un error
    de configuración (no frente a un problema de los datos):

    1. ``research_files`` y ``holdout_files`` no son conjuntos disjuntos
       (un archivo declarado en ambas listas a la vez), o alguna de las dos
       listas contiene nombres repetidos dentro de sí misma.
    2. Una fila procesada como conjunto de investigación tiene un timestamp
       que llega o supera la frontera del hold-out (``holdout_boundary_utc``)
       -- lo que indicaría que ``research_files`` está mal declarado y se
       estaría tratando como "investigación" contenido que en realidad
       pertenece al período protegido.

    En ningún caso se dispara abriendo un archivo de ``holdout_files``:
    ambas comprobaciones se resuelven enteramente con la configuración y con
    los archivos de ``research_files`` ya parseados por la etapa que llama.
    """


def validate_research_holdout_disjoint(config: SnapshotConfig) -> None:
    """``research_files`` y ``holdout_files`` deben ser conjuntos disjuntos.

    Entrada: la configuración cargada (dos listas de nombres de archivo).
    Transformación: se convierten ambas listas a ``set`` y se calcula la
    intersección. Esta comprobación es pura aritmética sobre texto -- NO
    abre ningún archivo, ni de investigación ni de hold-out -- por lo que
    debe ejecutarse antes que cualquier otra cosa en la etapa que la llama.

    También se comprueba que ninguna de las dos listas tenga nombres
    repetidos dentro de sí misma: un archivo listado dos veces en
    ``research_files`` se procesaría (y contaría) dos veces, lo cual rompe
    silenciosamente cualquier aritmética de conservación o de agregación
    aguas abajo, aunque en sentido estricto no sea una violación del
    aislamiento del hold-out -- es el mismo tipo de error de configuración
    que esta función existe para atrapar antes de procesar nada.

    Salida: ninguna si todo está bien; ``HoldoutIsolationError`` si no.
    """
    research_list = config.research_files
    holdout_list = config.holdout_files
    research_set = set(research_list)
    holdout_set = set(holdout_list)

    overlap = research_set & holdout_set
    if overlap:
        raise HoldoutIsolationError(
            "research_files y holdout_files no son disjuntos: "
            f"{sorted(overlap)} aparece(n) declarado(s) en ambas listas de "
            "configs/mnq_snapshot.yaml. La etapa aborta sin procesar ningun archivo."
        )

    if len(research_list) != len(research_set):
        dupes = sorted({f for f in research_list if research_list.count(f) > 1})
        raise HoldoutIsolationError(
            f"research_files contiene archivo(s) duplicado(s): {dupes}. "
            "La etapa aborta sin procesar ningun archivo."
        )

    if len(holdout_list) != len(holdout_set):
        dupes = sorted({f for f in holdout_list if holdout_list.count(f) > 1})
        raise HoldoutIsolationError(
            f"holdout_files contiene archivo(s) duplicado(s): {dupes}. "
            "La etapa aborta sin procesar ningun archivo."
        )


def validate_last_timestamps_before_boundary(
    config: SnapshotConfig, last_timestamps: dict[str, "pd.Timestamp | None"]
) -> None:
    """Ninguna fila de investigación puede alcanzar o superar el hold-out.

    Entrada: la configuración (para leer ``holdout_boundary_utc``) y un
    diccionario ``{nombre_de_archivo: ultimo_timestamp_o_None}`` -- el
    máximo timestamp entre las filas bien formadas de cada archivo de
    investigación, que la etapa que llama ya tiene que haber calculado al
    parsear esos archivos. Esta función NO parsea nada: sólo compara
    valores que ya existen.

    Transformación: para cada archivo, se compara su último timestamp
    contra la frontera declarada. Basta con mirar el máximo por archivo
    -- no hace falta reescanear fila a fila -- porque si el máximo de un
    archivo es estrictamente anterior a la frontera, automáticamente
    TODAS sus filas lo son también.

    Ni esta función ni la etapa que la llama abren, en ningún momento, un
    archivo de ``config.holdout_files``: la frontera se conoce por
    configuración, no observando el hold-out.

    Salida: ninguna si todo timestamp de investigación es anterior a la
    frontera; ``HoldoutIsolationError`` si algún archivo la alcanza o la
    supera.
    """
    boundary = pd.Timestamp(config.holdout_boundary_utc)
    offending = [
        (name, ts) for name, ts in last_timestamps.items() if ts is not None and ts >= boundary
    ]
    if offending:
        detail = "; ".join(f"{name}: last_timestamp={ts}" for name, ts in offending)
        raise HoldoutIsolationError(
            f"Violacion de la frontera de hold-out (holdout_boundary_utc={boundary}): "
            f"{detail}. Archivo(s) declarado(s) como research_files contienen filas "
            "que alcanzan o superan la frontera del hold-out -- la configuracion de "
            "research_files esta mal declarada. La etapa aborta sin escribir artefactos."
        )
