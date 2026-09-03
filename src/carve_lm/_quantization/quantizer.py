from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

import torch.nn as nn

from .config import AWQConfig, GPTQConfig, QuantConfig, QuantMethod, SmoothQuantConfig, WeightQuantConfig
from .methods.awq import AWQQuantizer
from .methods.gptq import GPTQQuantizer
from .methods.rtn import RTNQuantizer
from .methods.smoothquant import SmoothQuantQuantizer
from .metrics import get_model_size_mb, get_quantization_summary

logger = logging.getLogger(__name__)


@dataclass
class QuantizationResult:
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    quantized_layers_count: int
    total_linear_layers: int
    config: QuantConfig


class AutoQuantizer:
    """
    High-level quantizer supporting RTN, SmoothQuant, and GPTQ methods.
    """

    def __init__(self, config: QuantConfig | None = None):
        self.config = config or WeightQuantConfig()

    def quantize(
        self,
        model: nn.Module,
        dataloader: Iterator[Any] | Sequence[Any] | None = None,
        config: QuantConfig | None = None,
        num_calibration_batches: int = 16,
        forward_fn: Callable[[nn.Module, Any], Any] | None = None,
        inplace: bool = True,
    ) -> tuple[nn.Module, QuantizationResult]:
        """
        Quantize model according to configuration.
        """
        cfg = config or self.config
        method = QuantMethod(cfg.method) if isinstance(cfg.method, str) else cfg.method

        orig_size = get_model_size_mb(model)

        if method == QuantMethod.RTN:
            quantizer = RTNQuantizer(cfg)
            quantized_model = quantizer.quantize_model(model, inplace=inplace)

        elif method == QuantMethod.SMOOTHQUANT:
            sq_cfg = (
                cfg
                if isinstance(cfg, SmoothQuantConfig)
                else SmoothQuantConfig(
                    bits=cfg.bits,
                    act_bits=cfg.act_bits or 8,
                    scheme=cfg.scheme,
                    granularity=cfg.granularity,
                    group_size=cfg.group_size,
                    target_modules=cfg.target_modules,
                    exclude_modules=cfg.exclude_modules,
                )
            )
            quantizer = SmoothQuantQuantizer(sq_cfg)
            quantized_model = quantizer.quantize_model(
                model,
                dataloader=dataloader,
                num_batches=num_calibration_batches,
                forward_fn=forward_fn,
                inplace=inplace,
            )

        elif method == QuantMethod.GPTQ:
            if dataloader is None:
                raise ValueError("GPTQ quantization requires a calibration dataloader")
            gptq_cfg = (
                cfg
                if isinstance(cfg, GPTQConfig)
                else GPTQConfig(
                    bits=cfg.bits,
                    scheme=cfg.scheme,
                    granularity=cfg.granularity,
                    group_size=cfg.group_size,
                    target_modules=cfg.target_modules,
                    exclude_modules=cfg.exclude_modules,
                )
            )
            quantizer = GPTQQuantizer(gptq_cfg)
            quantized_model = quantizer.quantize_model(
                model,
                dataloader=dataloader,
                num_batches=num_calibration_batches,
                forward_fn=forward_fn,
                inplace=inplace,
            )

        elif method == QuantMethod.AWQ:
            if dataloader is None:
                raise ValueError("AWQ quantization requires a calibration dataloader")
            awq_cfg = (
                cfg
                if isinstance(cfg, AWQConfig)
                else AWQConfig(
                    bits=cfg.bits,
                    scheme=cfg.scheme,
                    granularity=cfg.granularity,
                    group_size=cfg.group_size,
                    target_modules=cfg.target_modules,
                    exclude_modules=cfg.exclude_modules,
                )
            )
            quantizer = AWQQuantizer(awq_cfg)
            quantized_model = quantizer.quantize_model(
                model,
                dataloader=dataloader,
                num_batches=num_calibration_batches,
                forward_fn=forward_fn,
                inplace=inplace,
            )

        else:
            raise ValueError(f"Unsupported quantization method: {method}")

        quant_size = get_model_size_mb(quantized_model)
        summary = get_quantization_summary(quantized_model)
        comp_ratio = (orig_size / quant_size) if quant_size > 0 else 1.0

        result = QuantizationResult(
            original_size_mb=orig_size,
            quantized_size_mb=quant_size,
            compression_ratio=comp_ratio,
            quantized_layers_count=summary["quantized_linear_layers"],
            total_linear_layers=summary["total_linear_layers"],
            config=cfg,
        )

        logger.info(
            f"Model quantized successfully: {orig_size:.2f} MB -> {quant_size:.2f} MB "
            f"({comp_ratio:.2f}x compression, "
            f"{result.quantized_layers_count}/{result.total_linear_layers} linear layers quantized)."
        )

        return quantized_model, result
