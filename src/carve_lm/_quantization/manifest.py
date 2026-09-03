from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn

from .config import QuantConfig
from .modules import QuantizedLinear
from .quantizer import QuantizationResult

logger = logging.getLogger(__name__)

QUANTIZATION_MANIFEST_NAME = "quantization_manifest.json"
QUANTIZATION_WEIGHTS_NAME = "quantized_model.pt"


def save_quantized(
    model: nn.Module,
    save_directory: str | Path,
    config: QuantConfig | None = None,
    result: QuantizationResult | None = None,
) -> Path:
    """
    Save quantized model weights, manifest, and configuration.
    """
    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)

    quantized_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            quantized_modules[name] = {
                "in_features": module.in_features,
                "out_features": module.out_features,
                "bias": module.bias is not None,
                "bits": module.bits,
                "scheme": module.scheme.value,
                "granularity": module.granularity.value,
                "group_size": module.group_size,
                "act_bits": module.act_bits,
                "pack_weights": module.pack_weights,
            }

    manifest = {
        "format": "carvelm-quantization-v1",
        "quantization_config": config.to_dict() if config else (result.config.to_dict() if result else {}),
        "quantized_modules": quantized_modules,
    }

    manifest_path = save_dir / QUANTIZATION_MANIFEST_NAME
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    weights_path = save_dir / QUANTIZATION_WEIGHTS_NAME
    torch.save(model.state_dict(), weights_path)

    # If model has config and save_pretrained, also save HF config
    if hasattr(model, "config") and hasattr(model.config, "to_dict"):
        cfg_dict = model.config.to_dict() if callable(model.config.to_dict) else dict(model.config.__dict__)
        with open(save_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, indent=2)

    logger.info(f"Quantized model saved to {save_dir}")
    return save_dir


def load_quantized(
    save_directory: str | Path,
    base_model: nn.Module,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """
    Load a quantized model into a base model instance according to quantization_manifest.json.
    """
    save_dir = Path(save_directory)
    manifest_path = save_dir / QUANTIZATION_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    quantized_modules = manifest.get("quantized_modules", {})

    # Replace modules in base_model with QuantizedLinear placeholders
    for name, meta in quantized_modules.items():
        q_layer = QuantizedLinear(
            in_features=meta["in_features"],
            out_features=meta["out_features"],
            bias=meta["bias"],
            bits=meta["bits"],
            scheme=meta["scheme"],
            granularity=meta["granularity"],
            group_size=meta.get("group_size"),
            act_bits=meta.get("act_bits"),
            pack_weights=meta.get("pack_weights", True),
            device=device,
        )
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = base_model.get_submodule(parent_name) if parent_name else base_model
        setattr(parent, child_name, q_layer)

    # Load weights
    weights_path = save_dir / QUANTIZATION_WEIGHTS_NAME
    state_dict = torch.load(weights_path, map_location=device)
    base_model.load_state_dict(state_dict)
    base_model.to(device)

    logger.info(f"Quantized model successfully loaded from {save_dir}")
    return base_model
