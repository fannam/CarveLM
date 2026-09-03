from __future__ import annotations

import torch
import torch.nn as nn

from carve_lm._quantization.config import QuantConfig, QuantGranularity, QuantScheme, WeightQuantConfig
from carve_lm._quantization.modules import QuantizedLinear


def test_quantized_linear_from_float_int8():
    torch.manual_seed(42)
    linear = nn.Linear(32, 16, bias=True)
    cfg = WeightQuantConfig(bits=8, scheme=QuantScheme.SYMMETRIC, granularity=QuantGranularity.PER_CHANNEL)
    
    q_linear = QuantizedLinear.from_float(linear, cfg)
    assert q_linear.in_features == 32
    assert q_linear.out_features == 16
    assert q_linear.bits == 8
    assert q_linear.qweight.shape == (16, 32)
    assert q_linear.scales.shape == (16, 1)
    assert q_linear.bias is not None

    x = torch.randn(2, 4, 32)
    out_orig = linear(x)
    out_quant = q_linear(x)

    assert out_quant.shape == (2, 4, 16)
    cos_sim = torch.cosine_similarity(out_orig.flatten(), out_quant.flatten(), dim=0)
    assert cos_sim > 0.99


def test_quantized_linear_from_float_int4_packed():
    torch.manual_seed(42)
    linear = nn.Linear(64, 32, bias=False)
    cfg = WeightQuantConfig(
        bits=4,
        scheme=QuantScheme.SYMMETRIC,
        granularity=QuantGranularity.PER_GROUP,
        group_size=32,
        pack_weights=True,
    )

    q_linear = QuantizedLinear.from_float(linear, cfg)
    assert q_linear.qweight.shape == (32, 32)  # packed from 64 to 32 bytes
    assert q_linear.scales.shape == (32, 2)    # 64 / 32 = 2 groups

    x = torch.randn(2, 64)
    out_orig = linear(x)
    out_quant = q_linear(x)

    assert out_quant.shape == (2, 32)
    cos_sim = torch.cosine_similarity(out_orig.flatten(), out_quant.flatten(), dim=0)
    assert cos_sim > 0.95


def test_quantized_linear_to_float():
    torch.manual_seed(42)
    linear = nn.Linear(20, 10, bias=True)
    cfg = WeightQuantConfig(bits=8, scheme=QuantScheme.SYMMETRIC)
    q_linear = QuantizedLinear.from_float(linear, cfg)

    reconstructed_linear = q_linear.to_float()
    assert isinstance(reconstructed_linear, nn.Linear)
    assert reconstructed_linear.in_features == 20
    assert reconstructed_linear.out_features == 10
    assert reconstructed_linear.bias is not None


def test_quantized_linear_w8a8():
    torch.manual_seed(42)
    linear = nn.Linear(32, 16, bias=False)
    cfg = QuantConfig(bits=8, act_bits=8, scheme=QuantScheme.SYMMETRIC)
    q_linear = QuantizedLinear.from_float(linear, cfg)

    x = torch.randn(3, 32)
    out = q_linear(x)
    assert out.shape == (3, 16)
