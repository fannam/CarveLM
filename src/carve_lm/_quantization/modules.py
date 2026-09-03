from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import QuantConfig, QuantGranularity, QuantScheme
from .functional import (
    dequantize_asymmetric,
    dequantize_symmetric,
    dynamic_quantize_activation,
    pack_int4,
    quantize_asymmetric,
    quantize_symmetric,
    unpack_int4,
)


class QuantizedLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear holding quantized weights (INT8 / INT4 packed).
    Supports symmetric/asymmetric quantization across per-channel, per-group, and per-tensor granularities.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        bits: int = 8,
        scheme: str | QuantScheme = QuantScheme.SYMMETRIC,
        granularity: str | QuantGranularity = QuantGranularity.PER_CHANNEL,
        group_size: int | None = 128,
        act_bits: int | None = None,
        pack_weights: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.scheme = QuantScheme(scheme)
        self.granularity = QuantGranularity(granularity)
        self.group_size = group_size
        self.act_bits = act_bits
        self.pack_weights = pack_weights and (bits == 4)
        self.weight_shape = (out_features, in_features)

        # Allocate buffers for quantized weights and scales
        if self.pack_weights:
            packed_in_features = (in_features + 1) // 2
            qweight_tensor = torch.zeros((out_features, packed_in_features), dtype=torch.uint8, device=device)
        else:
            qweight_tensor = torch.zeros(
                (out_features, in_features),
                dtype=torch.int8 if self.scheme == QuantScheme.SYMMETRIC else torch.uint8,
                device=device,
            )
        self.register_buffer("qweight", qweight_tensor)

        # Scale buffer shape
        if self.granularity == QuantGranularity.PER_TENSOR:
            scales_shape = (1,)
        elif self.granularity == QuantGranularity.PER_CHANNEL:
            scales_shape = (out_features, 1)
        elif self.granularity == QuantGranularity.PER_GROUP:
            num_groups = (in_features + group_size - 1) // group_size
            scales_shape = (out_features, num_groups)
        else:
            raise ValueError(f"Unknown granularity: {self.granularity}")

        self.register_buffer("scales", torch.ones(scales_shape, dtype=dtype, device=device))

        if self.scheme == QuantScheme.ASYMMETRIC:
            self.register_buffer(
                "zero_points",
                torch.zeros(
                    scales_shape,
                    dtype=torch.uint8,
                    device=device,
                ),
            )
        else:
            self.zero_points = None

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        config: QuantConfig,
    ) -> "QuantizedLinear":
        """
        Construct a QuantizedLinear layer from a float nn.Linear layer and QuantConfig.
        """
        has_bias = linear.bias is not None
        device = linear.weight.device
        dtype = linear.weight.dtype

        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=has_bias,
            bits=config.bits,
            scheme=config.scheme,
            granularity=config.granularity,
            group_size=config.group_size,
            act_bits=config.act_bits,
            pack_weights=config.pack_weights,
            device=device,
            dtype=dtype,
        )

        with torch.no_grad():
            w = linear.weight.data
            if layer.scheme == QuantScheme.SYMMETRIC:
                q, scales = quantize_symmetric(
                    w,
                    bits=layer.bits,
                    granularity=layer.granularity,
                    dim=0,
                    group_size=layer.group_size,
                )
                if layer.pack_weights:
                    layer.qweight.copy_(pack_int4(q))
                else:
                    layer.qweight.copy_(q.to(layer.qweight.dtype))
                layer.scales.copy_(scales.view(layer.scales.shape).to(dtype))
            else:
                q, scales, zp = quantize_asymmetric(
                    w,
                    bits=layer.bits,
                    granularity=layer.granularity,
                    dim=0,
                    group_size=layer.group_size,
                )
                if layer.pack_weights:
                    layer.qweight.copy_(pack_int4(q))
                else:
                    layer.qweight.copy_(q.to(layer.qweight.dtype))
                layer.scales.copy_(scales.view(layer.scales.shape).to(dtype))
                if layer.zero_points is not None:
                    layer.zero_points.copy_(zp.view(layer.zero_points.shape))

            if has_bias:
                layer.bias.copy_(linear.bias.data)

        return layer

    def dequantize_weight(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """
        Dequantize stored integer weights to float tensor of shape (out_features, in_features).
        """
        target_dtype = dtype or self.scales.dtype

        if self.pack_weights:
            signed = self.scheme == QuantScheme.SYMMETRIC
            q = unpack_int4(self.qweight, self.weight_shape, signed=signed)
        else:
            q = self.qweight

        if self.scheme == QuantScheme.SYMMETRIC:
            return dequantize_symmetric(
                q,
                self.scales,
                granularity=self.granularity,
                dim=0,
                group_size=self.group_size,
                dtype=target_dtype,
            )
        else:
            return dequantize_asymmetric(
                q,
                self.scales,
                self.zero_points,
                granularity=self.granularity,
                dim=0,
                group_size=self.group_size,
                dtype=target_dtype,
            )

    def to_float(self) -> nn.Linear:
        """
        Convert back to standard float nn.Linear layer.
        """
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.qweight.device,
            dtype=self.scales.dtype,
        )
        with torch.no_grad():
            linear.weight.copy_(self.dequantize_weight())
            if self.bias is not None:
                linear.bias.copy_(self.bias)
        return linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_dequant = self.dequantize_weight(dtype=x.dtype)

        if self.act_bits is not None:
            # Dynamic activation quantization (e.g. W8A8)
            x_q, x_scale = dynamic_quantize_activation(
                x,
                bits=self.act_bits,
                per_token=True,
                symmetric=True,
            )
            # Reconstruct simulated dynamic activations
            x = x_q.to(x.dtype) * x_scale.to(x.dtype)

        return F.linear(x, w_dequant, self.bias)

    def extra_repr(self) -> str:
        s = f"in_features={self.in_features}, out_features={self.out_features}, bits={self.bits}"
        s += f", scheme={self.scheme.value}, granularity={self.granularity.value}"
        if self.granularity == QuantGranularity.PER_GROUP:
            s += f", group_size={self.group_size}"
        if self.act_bits:
            s += f", act_bits={self.act_bits}"
        if self.pack_weights:
            s += ", packed=True"
        if self.bias is None:
            s += ", bias=False"
        return s
