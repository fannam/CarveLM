from __future__ import annotations

import torch
from tests.vlm.fixtures.synthetic_vlm import (
    ensure_registered_synthetic_vlm_adapter,
    make_synthetic_vlm,
)

from carve_lm.vlm.auto_model import apply_component_pruning_from_config
from carve_lm.vlm.components.language.adapters import get_model_adapter
from carve_lm.vlm.components.language.core import AttentionPasser, FeedForwardPasser
from carve_lm.vlm.components.language.pruners import ComponentPruner


def test_apply_component_pruning_from_config_replays_language_component_pruning():
    ensure_registered_synthetic_vlm_adapter()
    torch.manual_seed(0)

    adapter = get_model_adapter("synthetic_vlm")
    model = make_synthetic_vlm()
    num_layers = len(adapter.get_layers(model))
    assert num_layers >= 2
    last = num_layers - 1

    pruned = ComponentPruner(model, device="cpu", model_adapter="synthetic_vlm").prune(
        importance_scores={
            "attention": [0.0] + [1.0] * (num_layers - 1),
            "mlp": [1.0] * (num_layers - 1) + [0.0],
        },
        prune_counts={"attention": 1, "mlp": 1},
    )
    assert pruned.config.attention_layer_to_prune == [0]
    assert pruned.config.mlp_layer_to_prune == [last]

    # A fresh model carrying only the recorded prune lists is restored to the
    # same identity-module layout by the auto-model replay helper.
    fresh = make_synthetic_vlm()
    fresh.config.attention_layer_to_prune = list(pruned.config.attention_layer_to_prune)
    fresh.config.mlp_layer_to_prune = list(pruned.config.mlp_layer_to_prune)

    restored = apply_component_pruning_from_config(fresh, model_adapter="synthetic_vlm")
    layers = adapter.get_layers(restored)

    assert isinstance(adapter.get_attention_module(layers[0]), AttentionPasser)
    assert isinstance(adapter.get_mlp_module(layers[last]), FeedForwardPasser)
    assert not isinstance(adapter.get_attention_module(layers[last]), AttentionPasser)
    assert not isinstance(adapter.get_mlp_module(layers[0]), FeedForwardPasser)


def test_apply_component_pruning_from_config_is_noop_without_recorded_layers():
    ensure_registered_synthetic_vlm_adapter()
    model = make_synthetic_vlm()
    adapter = get_model_adapter("synthetic_vlm")

    restored = apply_component_pruning_from_config(model, model_adapter="synthetic_vlm")
    layers = adapter.get_layers(restored)
    assert not isinstance(adapter.get_attention_module(layers[0]), AttentionPasser)
