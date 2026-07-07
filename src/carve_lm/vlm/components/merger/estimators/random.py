from __future__ import annotations

from ..core import ESTIMATOR_REGISTRY
from ._base import _BaseMergerRandomEstimator


@ESTIMATOR_REGISTRY.register("random.element")
class RandomEstimator(_BaseMergerRandomEstimator):
    """Random baseline estimator for Qwen2.5-VL patch merger channels."""


__all__ = ["RandomEstimator"]
