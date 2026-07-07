"""
Level-1 (element-level) width pruners.

Prunes individual neurons, attention heads, GQA groups, or embedding
channels within each decoder layer without removing entire layers or blocks.
Also exposes the group-level and channel-level structured width pruners that
operate on discovered coupling groups of parameters.

The pruner and strategy implementations are shared with the VLM stack via
``carve_lm._pruning.width``; this module registers them into the LLM registries
and binds the LLM adapter resolver, strategy registry, and embedding-scoring
helper.
"""
from __future__ import annotations

import warnings

from carve_lm._pruning.width import (
    AttentionGroupPruningStrategy as _SharedAttentionGroupPruningStrategy,
)
from carve_lm._pruning.width import (
    AttentionQueryPruningStrategy as _SharedAttentionQueryPruningStrategy,
)
from carve_lm._pruning.width import BaseElementPruningStrategy
from carve_lm._pruning.width import (
    EmbeddingChannelPruningStrategy as _SharedEmbeddingChannelPruningStrategy,
)
from carve_lm._pruning.width import MLPPruningStrategy as _SharedMLPPruningStrategy
from carve_lm._pruning.width import WidthPruner as _SharedWidthPruner

from ...adapters import resolve_model_adapter
from ...core import (
    PRUNER_REGISTRY,
    PRUNING_STRATEGY_REGISTRY,
    calculate_embedding_channels_global_score,
)
from .._compat import warn_pruner_alias
from .._engine.facade import WidthChannelPruner as _EngineWidthChannelPruner
from .._engine.facade import WidthGroupPruner as _EngineWidthGroupPruner


@PRUNER_REGISTRY.register("width", aliases=("element",))
class WidthPruner(_SharedWidthPruner):
    _resolve_model_adapter = staticmethod(resolve_model_adapter)
    _strategy_registry = PRUNING_STRATEGY_REGISTRY
    _embedding_global_score = staticmethod(calculate_embedding_channels_global_score)


AttentionQueryPruningStrategy = PRUNING_STRATEGY_REGISTRY.register("element.attention_query")(
    _SharedAttentionQueryPruningStrategy
)
AttentionGroupPruningStrategy = PRUNING_STRATEGY_REGISTRY.register("element.attention_group")(
    _SharedAttentionGroupPruningStrategy
)
MLPPruningStrategy = PRUNING_STRATEGY_REGISTRY.register("element.mlp")(_SharedMLPPruningStrategy)
EmbeddingChannelPruningStrategy = PRUNING_STRATEGY_REGISTRY.register("element.embedding_channels")(
    _SharedEmbeddingChannelPruningStrategy
)


@PRUNER_REGISTRY.register("width.group")
class WidthGroupPruner(_EngineWidthGroupPruner):
    """
    Structured width pruner that discovers coupled parameter groups
    (e.g. attention head bundles + tied MLP rows) and prunes by
    group-level importance scores.
    """


@PRUNER_REGISTRY.register("width.channel")
class WidthChannelPruner(_EngineWidthChannelPruner):
    """
    Structured width pruner that discovers channel bundles and prunes
    along the hidden (embedding) dimension.
    """


def available_element_pruning_strategies() -> tuple[str, ...]:
    warnings.warn(
        "available_element_pruning_strategies() is deprecated; use WidthPruner.available_strategies().",
        DeprecationWarning,
        stacklevel=2,
    )
    return tuple(
        name
        for name in PRUNING_STRATEGY_REGISTRY.names()
        if name.startswith("element.")
    )


class ElementPruner(WidthPruner):
    """Backward-compatible alias for legacy code."""

    def __init__(self, *args, **kwargs):
        warn_pruner_alias("ElementPruner", "WidthPruner", stacklevel=3)
        super().__init__(*args, **kwargs)


class Llama3ElementPruner(ElementPruner):
    """Backward-compatible alias for legacy code."""


class Qwen2ElementPruner(ElementPruner):
    """Backward-compatible alias for legacy code."""


class MistralElementPruner(ElementPruner):
    """Backward-compatible alias for legacy code."""


__all__ = [
    "BaseElementPruningStrategy",
    "WidthPruner",
    "WidthGroupPruner",
    "WidthChannelPruner",
    "AttentionQueryPruningStrategy",
    "AttentionGroupPruningStrategy",
    "MLPPruningStrategy",
    "EmbeddingChannelPruningStrategy",
    "ElementPruner",
    "Llama3ElementPruner",
    "Qwen2ElementPruner",
    "MistralElementPruner",
    "available_element_pruning_strategies",
]
