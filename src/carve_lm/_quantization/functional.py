from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import QuantGranularity, QuantScheme


def _get_qmin_qmax(bits: int, symmetric: bool) -> tuple[int, int]:
    if symmetric:
        qmax = (1 << (bits - 1)) - 1
        qmin = -qmax
    else:
        qmin = 0
        qmax = (1 << bits) - 1
    return qmin, qmax


def quantize_symmetric(
    x: torch.Tensor,
    bits: int = 8,
    granularity: str | QuantGranularity = QuantGranularity.PER_CHANNEL,
    dim: int = 0,
    group_size: int | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric quantization of tensor x.

    Returns:
        (q_tensor, scales)
    """
    granularity = QuantGranularity(granularity)
    qmin, qmax = _get_qmin_qmax(bits, symmetric=True)
    orig_shape = x.shape

    if granularity == QuantGranularity.PER_TENSOR:
        max_val = torch.amax(torch.abs(x)).clamp(min=eps)
        scale = max_val / float(qmax)
        q = torch.clamp(torch.round(x / scale), qmin, qmax)
        return q.to(torch.int8), scale

    elif granularity == QuantGranularity.PER_CHANNEL:
        # Per-channel along specified dim (default 0 for out_features of Linear weight)
        keep_dims = [1] * x.ndim
        keep_dims[dim] = orig_shape[dim]
        max_val = torch.amax(torch.abs(x), dim=[i for i in range(x.ndim) if i != dim], keepdim=True).clamp(min=eps)
        scale = max_val / float(qmax)
        q = torch.clamp(torch.round(x / scale), qmin, qmax)
        return q.to(torch.int8), scale

    elif granularity == QuantGranularity.PER_GROUP:
        if group_size is None or group_size <= 0:
            raise ValueError(f"group_size must be a positive integer, got {group_size}")
        
        # We quantize along the second dimension (in_features for 2D weight)
        if x.ndim != 2:
            raise ValueError(f"per_group quantization currently expects 2D tensors, got shape {orig_shape}")
        
        out_features, in_features = orig_shape
        pad_len = (group_size - (in_features % group_size)) % group_size
        if pad_len > 0:
            x_padded = F.pad(x, (0, pad_len))
        else:
            x_padded = x

        num_groups = x_padded.shape[1] // group_size
        x_reshaped = x_padded.view(out_features, num_groups, group_size)
        max_val = torch.amax(torch.abs(x_reshaped), dim=-1, keepdim=True).clamp(min=eps)
        scale = max_val / float(qmax)
        q_reshaped = torch.clamp(torch.round(x_reshaped / scale), qmin, qmax)
        q = q_reshaped.view(out_features, -1)[:, :in_features]
        return q.to(torch.int8), scale.squeeze(-1)

    raise ValueError(f"Unsupported granularity: {granularity}")


def dequantize_symmetric(
    q: torch.Tensor,
    scales: torch.Tensor,
    granularity: str | QuantGranularity = QuantGranularity.PER_CHANNEL,
    dim: int = 0,
    group_size: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Dequantize symmetrically quantized integer tensor.
    """
    granularity = QuantGranularity(granularity)
    q_float = q.to(dtype)

    if granularity == QuantGranularity.PER_TENSOR:
        return q_float * scales.to(dtype)

    elif granularity == QuantGranularity.PER_CHANNEL:
        if scales.ndim != q.ndim:
            shape = [1] * q.ndim
            shape[dim] = q.shape[dim]
            scales_reshaped = scales.view(*shape).to(dtype)
        else:
            scales_reshaped = scales.to(dtype)
        return q_float * scales_reshaped

    elif granularity == QuantGranularity.PER_GROUP:
        if group_size is None or group_size <= 0:
            raise ValueError("group_size must be specified for per_group dequantization")
        out_features, in_features = q.shape
        pad_len = (group_size - (in_features % group_size)) % group_size
        if pad_len > 0:
            q_padded = F.pad(q_float, (0, pad_len))
        else:
            q_padded = q_float
        num_groups = q_padded.shape[1] // group_size
        q_reshaped = q_padded.view(out_features, num_groups, group_size)
        scales_reshaped = scales.view(out_features, num_groups, 1).to(dtype)
        deq_reshaped = q_reshaped * scales_reshaped
        return deq_reshaped.view(out_features, -1)[:, :in_features]

    raise ValueError(f"Unsupported granularity: {granularity}")


def quantize_asymmetric(
    x: torch.Tensor,
    bits: int = 8,
    granularity: str | QuantGranularity = QuantGranularity.PER_CHANNEL,
    dim: int = 0,
    group_size: int | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Asymmetric quantization of tensor x.

    Returns:
        (q_tensor, scales, zero_points)
    """
    granularity = QuantGranularity(granularity)
    qmin, qmax = _get_qmin_qmax(bits, symmetric=False)
    orig_shape = x.shape

    if granularity == QuantGranularity.PER_TENSOR:
        min_val = torch.amin(x)
        max_val = torch.amax(x)
        scale = ((max_val - min_val).clamp(min=eps)) / float(qmax - qmin)
        zero_point = torch.clamp(torch.round(-min_val / scale), qmin, qmax)
        q = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
        return q.to(torch.uint8), scale, zero_point.to(torch.uint8)

    elif granularity == QuantGranularity.PER_CHANNEL:
        reduce_dims = [i for i in range(x.ndim) if i != dim]
        min_val = torch.amin(x, dim=reduce_dims, keepdim=True)
        max_val = torch.amax(x, dim=reduce_dims, keepdim=True)
        scale = ((max_val - min_val).clamp(min=eps)) / float(qmax - qmin)
        zero_point = torch.clamp(torch.round(-min_val / scale), qmin, qmax)
        q = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
        return q.to(torch.uint8), scale, zero_point.to(torch.uint8)

    elif granularity == QuantGranularity.PER_GROUP:
        if group_size is None or group_size <= 0:
            raise ValueError(f"group_size must be a positive integer, got {group_size}")
        if x.ndim != 2:
            raise ValueError(f"per_group quantization expects 2D tensors, got shape {orig_shape}")
        out_features, in_features = orig_shape
        pad_len = (group_size - (in_features % group_size)) % group_size
        if pad_len > 0:
            x_padded = F.pad(x, (0, pad_len))
        else:
            x_padded = x

        num_groups = x_padded.shape[1] // group_size
        x_reshaped = x_padded.view(out_features, num_groups, group_size)
        min_val = torch.amin(x_reshaped, dim=-1, keepdim=True)
        max_val = torch.amax(x_reshaped, dim=-1, keepdim=True)
        scale = ((max_val - min_val).clamp(min=eps)) / float(qmax - qmin)
        zero_point = torch.clamp(torch.round(-min_val / scale), qmin, qmax)
        q_reshaped = torch.clamp(torch.round(x_reshaped / scale) + zero_point, qmin, qmax)
        q = q_reshaped.view(out_features, -1)[:, :in_features]
        return q.to(torch.uint8), scale.squeeze(-1), zero_point.squeeze(-1).to(torch.uint8)

    raise ValueError(f"Unsupported granularity: {granularity}")


def dequantize_asymmetric(
    q: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    granularity: str | QuantGranularity = QuantGranularity.PER_CHANNEL,
    dim: int = 0,
    group_size: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Dequantize asymmetrically quantized integer tensor.
    """
    granularity = QuantGranularity(granularity)
    q_float = q.to(dtype)
    zp_float = zero_points.to(dtype)

    if granularity == QuantGranularity.PER_TENSOR:
        return (q_float - zp_float) * scales.to(dtype)

    elif granularity == QuantGranularity.PER_CHANNEL:
        if scales.ndim != q.ndim:
            shape = [1] * q.ndim
            shape[dim] = q.shape[dim]
            scales_reshaped = scales.view(*shape).to(dtype)
            zp_reshaped = zp_float.view(*shape)
        else:
            scales_reshaped = scales.to(dtype)
            zp_reshaped = zp_float
        return (q_float - zp_reshaped) * scales_reshaped

    elif granularity == QuantGranularity.PER_GROUP:
        if group_size is None or group_size <= 0:
            raise ValueError("group_size must be specified for per_group dequantization")
        out_features, in_features = q.shape
        pad_len = (group_size - (in_features % group_size)) % group_size
        if pad_len > 0:
            q_padded = F.pad(q_float, (0, pad_len))
        else:
            q_padded = q_float
        num_groups = q_padded.shape[1] // group_size
        q_reshaped = q_padded.view(out_features, num_groups, group_size)
        scales_reshaped = scales.view(out_features, num_groups, 1).to(dtype)
        zp_reshaped = zp_float.view(out_features, num_groups, 1)
        deq_reshaped = (q_reshaped - zp_reshaped) * scales_reshaped
        return deq_reshaped.view(out_features, -1)[:, :in_features]

    raise ValueError(f"Unsupported granularity: {granularity}")


def pack_int4(tensor: torch.Tensor) -> torch.Tensor:
    """
    Pack 4-bit integer values (stored in int8 / uint8 with values in [-8, 7] or [0, 15])
    along the last dimension into uint8 tensor (2 elements per byte).
    """
    orig_shape = tensor.shape
    last_dim = orig_shape[-1]
    pad_needed = last_dim % 2
    if pad_needed != 0:
        tensor = F.pad(tensor, (0, 1))

    # Map signed [-8, 7] or unsigned [0, 15] to unsigned nibble [0, 15]
    u4 = (tensor.to(torch.int32) & 0x0F).to(torch.uint8)
    even = u4[..., 0::2]
    odd = u4[..., 1::2]
    packed = even | (odd << 4)
    return packed


def unpack_int4(
    packed: torch.Tensor,
    original_shape: tuple[int, ...] | torch.Size,
    signed: bool = True,
) -> torch.Tensor:
    """
    Unpack 4-bit integer values from packed uint8 tensor back to original shape.
    """
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F

    stacked = torch.stack([low, high], dim=-1)
    unpacked_flat = stacked.view(*packed.shape[:-1], packed.shape[-1] * 2)

    last_dim = original_shape[-1]
    unpacked = unpacked_flat[..., :last_dim]

    if signed:
        # Convert unsigned nibble 0..15 back to signed int8 in [-8, 7]
        # values >= 8 are negative (subtract 16)
        signed_vals = unpacked.to(torch.int8)
        signed_vals = torch.where(signed_vals >= 8, signed_vals - 16, signed_vals)
        return signed_vals.view(original_shape)
    else:
        return unpacked.to(torch.uint8).view(original_shape)


def dynamic_quantize_activation(
    x: torch.Tensor,
    bits: int = 8,
    per_token: bool = True,
    symmetric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dynamic quantization of activations during forward pass.
    """
    qmin, qmax = _get_qmin_qmax(bits, symmetric=symmetric)
    if per_token:
        # Quantize per token along the last dimension (hidden dimension)
        max_val = torch.amax(torch.abs(x), dim=-1, keepdim=True).clamp(min=1e-8)
    else:
        max_val = torch.amax(torch.abs(x)).clamp(min=1e-8)

    scale = max_val / float(qmax)
    q = torch.clamp(torch.round(x / scale), qmin, qmax)
    return q, scale


def fake_quantize(
    x: torch.Tensor,
    bits: int = 8,
    scheme: str | QuantScheme = QuantScheme.SYMMETRIC,
    granularity: str | QuantGranularity = QuantGranularity.PER_CHANNEL,
    dim: int = 0,
    group_size: int | None = None,
) -> torch.Tensor:
    """
    Simulated (fake) quantization for QAT / validation.
    """
    scheme = QuantScheme(scheme)
    if scheme == QuantScheme.SYMMETRIC:
        q, scales = quantize_symmetric(x, bits=bits, granularity=granularity, dim=dim, group_size=group_size)
        return dequantize_symmetric(q, scales, granularity=granularity, dim=dim, group_size=group_size, dtype=x.dtype)
    else:
        q, scales, zp = quantize_asymmetric(x, bits=bits, granularity=granularity, dim=dim, group_size=group_size)
        return dequantize_asymmetric(
            q, scales, zp, granularity=granularity, dim=dim, group_size=group_size, dtype=x.dtype
        )
