from __future__ import annotations

import tempfile

import torch
from tests.fixtures.synthetic_models import SyntheticCausalLM, SyntheticConfig

from carve_lm._quantization.config import WeightQuantConfig
from carve_lm._quantization.manifest import load_quantized, save_quantized
from carve_lm.llm.quantization import quantize_llm


def test_quantization_save_and_load_roundtrip():
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
    cfg = WeightQuantConfig(bits=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    quantized_model, result = quantize_llm(model, config=cfg)

    input_ids = torch.randint(0, 32, (2, 8))
    orig_quant_logits = quantized_model(input_ids=input_ids)["logits"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_quantized(quantized_model, tmp_dir, config=cfg, result=result)

        # Instantiate fresh unquantized model and reload
        fresh_model = SyntheticCausalLM(config)
        reloaded_model = load_quantized(tmp_dir, base_model=fresh_model, device="cpu")

        reloaded_logits = reloaded_model(input_ids=input_ids)["logits"]
        assert torch.allclose(orig_quant_logits, reloaded_logits, atol=1e-5)
