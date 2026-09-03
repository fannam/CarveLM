from __future__ import annotations

import torch
from tests.fixtures.synthetic_models import SyntheticCausalLM, SyntheticConfig

from carve_lm._quantization.config import GPTQConfig
from carve_lm._quantization.quantizer import AutoQuantizer


def test_gptq_layer_and_model_quantization():
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

    calibration_data = [{"input_ids": torch.randint(0, 32, (2, 8))} for _ in range(4)]

    gptq_config = GPTQConfig(
        bits=4,
        block_size=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    quantizer = AutoQuantizer(gptq_config)
    quantized_model, result = quantizer.quantize(
        model,
        dataloader=calibration_data,
        num_calibration_batches=4,
    )

    assert result.quantized_layers_count > 0

    input_ids = torch.randint(0, 32, (2, 8))
    outputs = quantized_model(input_ids=input_ids)
    assert outputs["logits"].shape == (2, 8, 32)
