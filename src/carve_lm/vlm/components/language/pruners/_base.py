from __future__ import annotations

from carve_lm._pruning.pruner_base import _BaseBlockPruner as _SharedBaseBlockPruner
from carve_lm._pruning.pruner_base import _BaseLayerPruner as _SharedBaseLayerPruner
from carve_lm._pruning.pruner_base import _BasePruner as _SharedBasePruner

from ..adapters import resolve_model_adapter


class _AdapterBoundPruner:
    """Bind the shared pruner bases to the VLM language adapter registry."""

    _resolve_model_adapter = staticmethod(resolve_model_adapter)


class _BasePruner(_AdapterBoundPruner, _SharedBasePruner):
    pass


class _BaseLayerPruner(_AdapterBoundPruner, _SharedBaseLayerPruner):
    pass


class _BaseBlockPruner(_AdapterBoundPruner, _SharedBaseBlockPruner):
    pass


__all__ = [
    "_BaseBlockPruner",
    "_BaseLayerPruner",
    "_BasePruner",
]
