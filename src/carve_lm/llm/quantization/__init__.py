from __future__ import annotations

from typing import Any, Iterator, Sequence

import torch.nn as nn

from ..._quantization import (
    AutoQuantizer,
    AWQConfig,
    AWQQuantizer,
    GPTQConfig,
    GPTQQuantizer,
    QuantConfig,
    QuantGranularity,
    QuantizationResult,
    QuantizedLinear,
    QuantMethod,
    QuantPrecision,
    QuantScheme,
    RTNQuantizer,
    SmoothQuantConfig,
    SmoothQuantQuantizer,
    WeightQuantConfig,
    get_model_size_mb,
    get_quantization_summary,
    load_quantized,
    save_quantized,
)

LLM_DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

LLM_DEFAULT_EXCLUDE_MODULES = (
    "lm_head",
    "embed_tokens",
)


def quantize_llm(
    model: nn.Module,
    config: QuantConfig | None = None,
    dataloader: Iterator[Any] | Sequence[Any] | None = None,
    num_calibration_batches: int = 16,
    inplace: bool = True,
) -> tuple[nn.Module, QuantizationResult]:
    """
    Convenience function to quantize an LLM with model-appropriate module defaults.
    """
    cfg = config or WeightQuantConfig()
    if cfg.target_modules is None:
        cfg.target_modules = LLM_DEFAULT_TARGET_MODULES
    if cfg.exclude_modules is None:
        cfg.exclude_modules = LLM_DEFAULT_EXCLUDE_MODULES

    quantizer = AutoQuantizer(cfg)
    return quantizer.quantize(
        model,
        dataloader=dataloader,
        config=cfg,
        num_calibration_batches=num_calibration_batches,
        inplace=inplace,
    )


__all__ = [
    "AWQConfig",
    "AWQQuantizer",
    "AutoQuantizer",
    "GPTQConfig",
    "GPTQQuantizer",
    "LLM_DEFAULT_EXCLUDE_MODULES",
    "LLM_DEFAULT_TARGET_MODULES",
    "QuantConfig",
    "QuantGranularity",
    "QuantMethod",
    "QuantPrecision",
    "QuantScheme",
    "QuantizationResult",
    "QuantizedLinear",
    "RTNQuantizer",
    "SmoothQuantConfig",
    "SmoothQuantQuantizer",
    "WeightQuantConfig",
    "get_model_size_mb",
    "get_quantization_summary",
    "load_quantized",
    "quantize_llm",
    "save_quantized",
]
