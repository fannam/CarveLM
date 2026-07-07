from __future__ import annotations

from carve_lm._pruning.facade import DepthLayerPruner as _SharedDepthLayerPruner
from carve_lm._pruning.facade import WidthChannelPruner as _SharedWidthChannelPruner
from carve_lm._pruning.facade import WidthGroupPruner as _SharedWidthGroupPruner

from ...adapters import resolve_model_adapter
from .estimation import estimate_scores
from .manifest import MANIFEST_FILENAME


class _LLMEngineHooks:
    """Bind the shared engine to the LLM adapter/estimator registries."""

    _resolve_model_adapter = staticmethod(resolve_model_adapter)
    _estimate_scores = staticmethod(estimate_scores)
    _manifest_filename = MANIFEST_FILENAME


class WidthGroupPruner(_LLMEngineHooks, _SharedWidthGroupPruner):
    pass


class WidthChannelPruner(_LLMEngineHooks, _SharedWidthChannelPruner):
    pass


class DepthLayerPruner(_LLMEngineHooks, _SharedDepthLayerPruner):
    pass


__all__ = [
    "DepthLayerPruner",
    "WidthChannelPruner",
    "WidthGroupPruner",
]
