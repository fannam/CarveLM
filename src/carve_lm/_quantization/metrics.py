from __future__ import annotations

from typing import Any

import torch.nn as nn

from .modules import QuantizedLinear


def get_model_size_mb(model: nn.Module) -> float:
    """
    Calculate the memory footprint of the model parameters and buffers in megabytes (MB).
    Correctly accounts for packed 4-bit / 8-bit quantized weights.
    """
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.numel() * param.element_size()
    for buffer in model.buffers():
        total_bytes += buffer.numel() * buffer.element_size()
    return total_bytes / (1024.0 * 1024.0)


def get_quantization_summary(model: nn.Module) -> dict[str, Any]:
    """
    Summarize quantized vs non-quantized layers in the model.
    """
    total_linear = 0
    quantized_linear = 0
    total_params = 0
    quantized_params = 0
    layers_info = []

    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            total_linear += 1
            quantized_linear += 1
            param_count = module.in_features * module.out_features
            quantized_params += param_count
            total_params += param_count
            layers_info.append({
                "name": name,
                "type": "QuantizedLinear",
                "in_features": module.in_features,
                "out_features": module.out_features,
                "bits": module.bits,
                "scheme": module.scheme.value,
                "granularity": module.granularity.value,
            })
        elif isinstance(module, nn.Linear):
            total_linear += 1
            param_count = module.in_features * module.out_features
            total_params += param_count
            layers_info.append({
                "name": name,
                "type": "nn.Linear (unquantized)",
                "in_features": module.in_features,
                "out_features": module.out_features,
            })

    return {
        "total_linear_layers": total_linear,
        "quantized_linear_layers": quantized_linear,
        "quantization_ratio": (quantized_linear / total_linear) if total_linear > 0 else 0.0,
        "layers": layers_info,
        "model_size_mb": get_model_size_mb(model),
    }
