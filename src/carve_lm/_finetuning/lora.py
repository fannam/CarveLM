"""LoRA fine-tuning support backed by PEFT."""

from __future__ import annotations

from .supervised import SupervisedFineTuner


def _apply_lora(model, lora_config):
    try:
        from peft import get_peft_model
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LoRAFineTuner requires PEFT. Install the `train` extra: pip install -e '.[train]'."
        ) from exc

    return get_peft_model(model, lora_config)


class LoRAFineTuner(SupervisedFineTuner):
    """Supervised fine-tuning trainer that adapts a model with a PEFT LoRA config."""

    def __init__(self, model, lora_config, train_loader, val_loader, **kwargs):
        self.base_model = model
        self.lora_config = lora_config
        super().__init__(
            model=_apply_lora(model, lora_config),
            train_loader=train_loader,
            val_loader=val_loader,
            **kwargs,
        )


__all__ = ["LoRAFineTuner"]
