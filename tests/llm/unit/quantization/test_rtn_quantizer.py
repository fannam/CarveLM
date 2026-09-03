from __future__ import annotations

import torch
from tests.fixtures.synthetic_models import SyntheticCausalLM, SyntheticConfig

from carve_lm._quantization.config import WeightQuantConfig
from carve_lm._quantization.metrics import get_model_size_mb
from carve_lm.llm.quantization import quantize_llm


def test_rtn_quantize_synthetic_model():
    torch.manual_seed(42)
    config = SyntheticConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=24,
        vocab_size=32,
        head_dim=4,
    )
    model = SyntheticCausalLM(config)
    orig_size = get_model_size_mb(model)

    cfg = WeightQuantConfig(
        bits=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    quantized_model, result = quantize_llm(model, config=cfg)

    assert result.quantized_layers_count > 0
    assert result.quantized_size_mb <= orig_size

    # Verify model forward pass works
    input_ids = torch.randint(0, 32, (2, 8))
    labels = torch.randint(0, 32, (2, 8))
    outputs = quantized_model(input_ids=input_ids, labels=labels)
    assert "logits" in outputs
    assert "loss" in outputs
    assert outputs["logits"].shape == (2, 8, 32)
    assert not torch.isnan(outputs["loss"])


def test_rtn_quantize_int4_packed():
    torch.manual_seed(42)
    config = SyntheticConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=48,
        vocab_size=64,
        head_dim=8,
    )
    model = SyntheticCausalLM(config)
    orig_size = get_model_size_mb(model)

    cfg = WeightQuantConfig(bits=4, pack_weights=True)
    quantized_model, result = quantize_llm(model, config=cfg)

    assert result.quantized_size_mb < orig_size
    assert result.compression_ratio > 1.0

    input_ids = torch.randint(0, 64, (2, 4))
    outputs = quantized_model(input_ids=input_ids)
    assert outputs["logits"].shape == (2, 4, 64)
