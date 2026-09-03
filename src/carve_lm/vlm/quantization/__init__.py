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

VLM_DEFAULT_LANGUAGE_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

VLM_DEFAULT_VISION_TARGETS = (
    "qkv",
    "proj",
    "fc1",
    "fc2",
)

VLM_DEFAULT_MERGER_TARGETS = (
    "linear",
    "mlp",
)


def quantize_vlm(
    model: nn.Module,
    config: QuantConfig | None = None,
    dataloader: Iterator[Any] | Sequence[Any] | None = None,
    components: Sequence[str] = ("language", "vision", "merger"),
    num_calibration_batches: int = 16,
    inplace: bool = True,
) -> tuple[nn.Module, QuantizationResult]:
    """
    Convenience function to quantize a Vision-Language Model (VLM).
    Allows targeting specific multimodal components ('language', 'vision', 'merger').
    """
    cfg = config or WeightQuantConfig()
    targets: list[str] = []
    if "language" in components:
        targets.extend(VLM_DEFAULT_LANGUAGE_TARGETS)
    if "vision" in components:
        targets.extend(VLM_DEFAULT_VISION_TARGETS)
    if "merger" in components:
        targets.extend(VLM_DEFAULT_MERGER_TARGETS)

    if cfg.target_modules is None:
        cfg.target_modules = tuple(targets)

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
    "VLM_DEFAULT_LANGUAGE_TARGETS",
    "VLM_DEFAULT_MERGER_TARGETS",
    "VLM_DEFAULT_VISION_TARGETS",
    "WeightQuantConfig",
    "get_model_size_mb",
    "get_quantization_summary",
    "load_quantized",
    "quantize_vlm",
    "save_quantized",
]
