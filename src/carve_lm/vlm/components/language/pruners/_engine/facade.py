from __future__ import annotations

from carve_lm._pruning.facade import DepthLayerPruner as _SharedDepthLayerPruner
from carve_lm._pruning.facade import WidthChannelPruner as _SharedWidthChannelPruner
from carve_lm._pruning.facade import WidthGroupPruner as _SharedWidthGroupPruner

from ...adapters import resolve_model_adapter
from .estimation import estimate_scores
from .manifest import MANIFEST_FILENAME


class _VLMLanguageEngineHooks:
    """Bind the shared engine to the VLM language adapter/estimator registries."""

    _resolve_model_adapter = staticmethod(resolve_model_adapter)
    _estimate_scores = staticmethod(estimate_scores)
    _manifest_filename = MANIFEST_FILENAME


class WidthGroupPruner(_VLMLanguageEngineHooks, _SharedWidthGroupPruner):
    pass


class WidthChannelPruner(_VLMLanguageEngineHooks, _SharedWidthChannelPruner):
    pass


class DepthLayerPruner(_VLMLanguageEngineHooks, _SharedDepthLayerPruner):
    pass


__all__ = [
    "DepthLayerPruner",
    "WidthChannelPruner",
    "WidthGroupPruner",
]
