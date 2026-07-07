from __future__ import annotations

from ..core import ESTIMATOR_REGISTRY
from ._base import _BaseVisionRandomEstimator


@ESTIMATOR_REGISTRY.register("random.element")
class RandomEstimator(_BaseVisionRandomEstimator):
    """Random baseline estimator for Qwen-style vision transformer blocks."""


__all__ = ["RandomEstimator"]
