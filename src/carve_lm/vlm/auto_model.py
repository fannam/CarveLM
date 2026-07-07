from __future__ import annotations

from collections.abc import Iterable

from transformers import AutoModelForImageTextToText

from .components.language.adapters import BaseModelAdapter, resolve_model_adapter


def _normalize_pruned_indices(
    indices: Iterable[int] | None,
    *,
    num_layers: int,
    field_name: str,
) -> list[int]:
    normalized = sorted({int(index) for index in (indices or ())})
    for index in normalized:
        if not 0 <= index < num_layers:
            raise ValueError(
                "{} contains invalid layer index {} for a model with {} language layers.".format(
                    field_name,
                    index,
                    num_layers,
                )
            )
    return normalized


def apply_component_pruning_from_config(
    model,
    model_adapter: BaseModelAdapter | str | None = None,
):
    """
    Replay decoder component pruning encoded on a multimodal model's config.

    CarveLM VLM component pruning keeps the original config class and records the
    removed attention/MLP layer indices on ``model.config`` (top level, same as
    the LLM stack). This helper reconstructs the matching identity-module layout
    on the language component after a regular HF load.
    """

    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Model does not expose a config, so component pruning cannot be replayed.")

    attention_layers = getattr(config, "attention_layer_to_prune", None)
    mlp_layers = getattr(config, "mlp_layer_to_prune", None)
    if not attention_layers and not mlp_layers:
        return model

    adapter = resolve_model_adapter(model, model_adapter)
    layers = adapter.get_layers(model)
    num_layers = len(layers)

    normalized_attention_layers = _normalize_pruned_indices(
        attention_layers,
        num_layers=num_layers,
        field_name="attention_layer_to_prune",
    )
    normalized_mlp_layers = _normalize_pruned_indices(
        mlp_layers,
        num_layers=num_layers,
        field_name="mlp_layer_to_prune",
    )

    for layer_idx in normalized_attention_layers:
        adapter.set_attention_module(layers[layer_idx], adapter.make_identity_attention())
    for layer_idx in normalized_mlp_layers:
        adapter.set_mlp_module(layers[layer_idx], adapter.make_identity_mlp())

    config.attention_layer_to_prune = normalized_attention_layers
    config.mlp_layer_to_prune = normalized_mlp_layers
    return model


class PrunedVLMAutoModel(AutoModelForImageTextToText):
    """
    Auto-model loader that restores CarveLM component-pruned VLM decoders.

    Save the pruned checkpoint with the normal model ``save_pretrained`` flow,
    then reload it with this class so any attention/MLP layers recorded on the
    config are rebuilt as identity pass-through modules on the language stack.
    """

    @classmethod
    def from_config(cls, config, **kwargs):
        model = super().from_config(config, **kwargs)
        return apply_component_pruning_from_config(model)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return apply_component_pruning_from_config(model)


__all__ = [
    "PrunedVLMAutoModel",
    "apply_component_pruning_from_config",
]
