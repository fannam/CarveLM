from __future__ import annotations

import torch

from carve_lm._quantization.config import QuantGranularity, QuantScheme
from carve_lm._quantization.functional import (
    dequantize_asymmetric,
    dequantize_symmetric,
    dynamic_quantize_activation,
    fake_quantize,
    pack_int4,
    quantize_asymmetric,
    quantize_symmetric,
    unpack_int4,
)


def test_quantize_symmetric_per_channel():
    torch.manual_seed(42)
    x = torch.randn(16, 32)
    q, scale = quantize_symmetric(x, bits=8, granularity=QuantGranularity.PER_CHANNEL, dim=0)
    assert q.shape == (16, 32)
    assert scale.shape == (16, 1)
    assert q.dtype == torch.int8

    deq = dequantize_symmetric(q, scale, granularity=QuantGranularity.PER_CHANNEL, dim=0)
    assert deq.shape == (16, 32)
    cos_sim = torch.cosine_similarity(x.flatten(), deq.flatten(), dim=0)
    assert cos_sim > 0.99


def test_quantize_symmetric_per_group():
    torch.manual_seed(42)
    x = torch.randn(8, 64)
    q, scale = quantize_symmetric(x, bits=8, granularity=QuantGranularity.PER_GROUP, group_size=16)
    assert q.shape == (8, 64)
    assert scale.shape == (8, 4)

    deq = dequantize_symmetric(q, scale, granularity=QuantGranularity.PER_GROUP, group_size=16)
    assert deq.shape == (8, 64)
    cos_sim = torch.cosine_similarity(x.flatten(), deq.flatten(), dim=0)
    assert cos_sim > 0.99


def test_quantize_asymmetric_per_channel():
    torch.manual_seed(42)
    x = torch.randn(12, 24) + 2.0
    q, scale, zp = quantize_asymmetric(x, bits=8, granularity=QuantGranularity.PER_CHANNEL, dim=0)
    assert q.shape == (12, 24)
    assert scale.shape == (12, 1)
    assert zp.shape == (12, 1)
    assert q.dtype == torch.uint8

    deq = dequantize_asymmetric(q, scale, zp, granularity=QuantGranularity.PER_CHANNEL, dim=0)
    assert deq.shape == (12, 24)
    cos_sim = torch.cosine_similarity(x.flatten(), deq.flatten(), dim=0)
    assert cos_sim > 0.99


def test_pack_unpack_int4_roundtrip():
    torch.manual_seed(42)
    # Signed int8 values in [-8, 7]
    orig = torch.randint(-8, 8, (4, 16), dtype=torch.int8)
    packed = pack_int4(orig)
    assert packed.shape == (4, 8)
    assert packed.dtype == torch.uint8

    unpacked = unpack_int4(packed, orig.shape, signed=True)
    assert torch.equal(orig, unpacked)


def test_dynamic_quantize_activation():
    torch.manual_seed(42)
    act = torch.randn(2, 8, 32)
    q, scale = dynamic_quantize_activation(act, bits=8, per_token=True, symmetric=True)
    assert q.shape == (2, 8, 32)
    assert scale.shape == (2, 8, 1)
    deq = q.float() * scale
    cos_sim = torch.cosine_similarity(act.flatten(), deq.flatten(), dim=0)
    assert cos_sim > 0.99


def test_fake_quantize():
    torch.manual_seed(42)
    x = torch.randn(10, 20)
    fake_q = fake_quantize(x, bits=8, scheme=QuantScheme.SYMMETRIC, granularity=QuantGranularity.PER_CHANNEL)
    assert fake_q.shape == x.shape
    assert fake_q.dtype == x.dtype
    assert torch.allclose(x, fake_q, atol=0.1)
