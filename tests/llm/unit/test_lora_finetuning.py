from __future__ import annotations

import sys
from types import ModuleType


def test_lora_finetuner_applies_peft_before_initializing_supervised_training(monkeypatch):
    import carve_lm._finetuning.lora as lora_module
    from carve_lm.llm.finetuning import LoRAFineTuner
    from carve_lm.vlm.finetuning import LoRAFineTuner as VLMLoRAFineTuner

    peft = ModuleType("peft")
    calls = []
    adapted_model = object()

    def get_peft_model(model, lora_config):
        calls.append((model, lora_config))
        return adapted_model

    def fake_supervised_init(self, **kwargs):
        self.supervised_init_kwargs = kwargs

    peft.get_peft_model = get_peft_model
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setattr(lora_module.SupervisedFineTuner, "__init__", fake_supervised_init)

    model = object()
    lora_config = object()
    trainer = LoRAFineTuner(
        model=model,
        lora_config=lora_config,
        train_loader="train",
        val_loader="eval",
        tokenizer="tokenizer",
    )

    assert VLMLoRAFineTuner is LoRAFineTuner
    assert calls == [(model, lora_config)]
    assert trainer.base_model is model
    assert trainer.lora_config is lora_config
    assert trainer.supervised_init_kwargs == {
        "model": adapted_model,
        "train_loader": "train",
        "val_loader": "eval",
        "tokenizer": "tokenizer",
    }
