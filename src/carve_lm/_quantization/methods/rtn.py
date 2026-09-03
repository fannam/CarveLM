from __future__ import annotations

import logging
from typing import Sequence

import torch.nn as nn

from ..config import QuantConfig, WeightQuantConfig
from ..modules import QuantizedLinear

logger = logging.getLogger(__name__)


class RTNQuantizer:
    """
    Round-To-Nearest (RTN) Quantizer for weight-only post-training quantization.
    Data-free, fast, and supports INT8 and INT4 with per-channel or per-group scaling.
    """

    def __init__(self, config: QuantConfig | WeightQuantConfig | None = None):
        self.config = config or WeightQuantConfig()

    def quantize_linear(self, linear: nn.Linear, config: QuantConfig | None = None) -> QuantizedLinear:
        cfg = config or self.config
        return QuantizedLinear.from_float(linear, cfg)

    def quantize_model(
        self,
        model: nn.Module,
        target_modules: Sequence[str] | None = None,
        exclude_modules: Sequence[str] | None = None,
        config: QuantConfig | None = None,
        inplace: bool = True,
    ) -> nn.Module:
        """
        Quantize all matching linear layers in the model using RTN.
        """
        cfg = config or self.config
        targets = tuple(target_modules if target_modules is not None else (cfg.target_modules or ()))
        excludes = tuple(exclude_modules if exclude_modules is not None else (cfg.exclude_modules or ()))

        target_model = model if inplace else copy_model(model)
        replaced_count = 0

        for name, module in list(target_model.named_modules()):
            if isinstance(module, nn.Linear):
                # Check exclusion
                if any(ex in name for ex in excludes):
                    logger.debug(f"Excluding {name} from quantization")
                    continue

                # Check inclusion if targets are specified
                if targets and not any(t in name for t in targets):
                    logger.debug(f"Skipping {name} (not in target_modules)")
                    continue

                # Replace with QuantizedLinear
                parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
                parent = target_model.get_submodule(parent_name) if parent_name else target_model
                quantized_layer = self.quantize_linear(module, cfg)
                setattr(parent, child_name, quantized_layer)
                replaced_count += 1

        logger.info(f"RTN Quantization complete: replaced {replaced_count} linear layers.")
        return target_model


def copy_model(model: nn.Module) -> nn.Module:
    import copy

    return copy.deepcopy(model)
