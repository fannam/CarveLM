from __future__ import annotations

import torch
from tests.vlm.fixtures.synthetic_vlm import SyntheticQwen25VLConfig, SyntheticQwen25VLModel

from carve_lm._quantization.config import WeightQuantConfig
from carve_lm.vlm.quantization import quantize_vlm


def test_vlm_quantization_language_only():
    torch.manual_seed(42)
    config = SyntheticQwen25VLConfig()
    model = SyntheticQwen25VLModel(config)

    cfg = WeightQuantConfig(bits=8)
    quantized_model, result = quantize_vlm(model, config=cfg, components=("language",))

    assert result.quantized_layers_count > 0


def test_vlm_quantization_all_components():
    torch.manual_seed(42)
    config = SyntheticQwen25VLConfig()
    model = SyntheticQwen25VLModel(config)

    cfg = WeightQuantConfig(bits=8)
    quantized_model, result = quantize_vlm(model, config=cfg, components=("language", "vision", "merger"))

    assert result.quantized_layers_count > 0
