"""
Example script demonstrating model quantization with CarveLM.
Supports RTN (Round-to-Nearest), SmoothQuant (W8A8), and GPTQ with INT8/INT4 precision.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM

from carve_lm.llm.quantization import (
    WeightQuantConfig,
    get_model_size_mb,
    load_quantized,
    quantize_llm,
    save_quantized,
)


def main():
    model_id = "meta-llama/Llama-3.2-1B"
    print(f"Loading base model {model_id}...")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    orig_size = get_model_size_mb(model)
    print(f"Original model footprint: {orig_size:.2f} MB")

    # 1. Weight-only 4-bit per-group RTN quantization
    print("\n--- 1. Applying Weight-only INT4 Quantization (RTN) ---")
    int4_config = WeightQuantConfig(
        bits=4,
        granularity="per_group",
        group_size=128,
        pack_weights=True,
    )
    quantized_model, result = quantize_llm(model, config=int4_config)
    print(
        f"Quantized footprint: {result.quantized_size_mb:.2f} MB "
        f"({result.compression_ratio:.2f}x compression, {result.quantized_layers_count} layers quantized)"
    )

    # Save quantized model
    save_dir = "artifacts/llama_quantized_int4"
    save_quantized(quantized_model, save_dir, config=int4_config, result=result)
    print(f"Saved quantized model to {save_dir}")

    # Reload quantized model
    reloaded_base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    reloaded_model = load_quantized(save_dir, base_model=reloaded_base)
    print(f"Reloaded quantized model successfully: {type(reloaded_model).__name__}")


if __name__ == "__main__":
    main()
