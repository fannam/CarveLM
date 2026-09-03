from __future__ import annotations

import logging
from typing import Any, Callable, Iterator, Sequence

import torch
import torch.nn as nn

from ..config import SmoothQuantConfig
from ..modules import QuantizedLinear

logger = logging.getLogger(__name__)


class SmoothQuantQuantizer:
    """
    SmoothQuant (Xiao et al.) Post-Training Quantization for W8A8.
    Calibrates activation scales across channels, smooths activations & weights,
    and converts linear layers to QuantizedLinear with activation quantization.
    """

    def __init__(self, config: SmoothQuantConfig | None = None):
        self.config = config or SmoothQuantConfig()

    @torch.no_grad()
    def calibrate_activation_scales(
        self,
        model: nn.Module,
        dataloader: Iterator[Any] | Sequence[Any],
        num_batches: int = 32,
        forward_fn: Callable[[nn.Module, Any], Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Record maximum absolute activation values across channels for each linear layer.
        """
        model.eval()
        act_scales: dict[str, torch.Tensor] = {}
        hooks = []

        def get_hook(name: str):
            def hook(module: nn.Module, input: tuple[torch.Tensor, ...], output: Any):
                inp = input[0].detach()
                # shape: [batch, seq_len, in_features] or [..., in_features]
                inp_flat = inp.view(-1, inp.shape[-1])
                batch_max = torch.amax(torch.abs(inp_flat), dim=0)
                if name not in act_scales:
                    act_scales[name] = batch_max
                else:
                    act_scales[name] = torch.maximum(act_scales[name], batch_max)

            return hook

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(get_hook(name)))

        batches_processed = 0
        with torch.no_grad():
            for batch in dataloader:
                if forward_fn is not None:
                    forward_fn(model, batch)
                elif isinstance(batch, dict):
                    model(**batch)
                elif isinstance(batch, (list, tuple)):
                    model(*batch)
                else:
                    model(batch)

                batches_processed += 1
                if batches_processed >= num_batches:
                    break

        for hook in hooks:
            hook.remove()

        logger.info(f"SmoothQuant calibration complete on {batches_processed} batches for {len(act_scales)} layers.")
        return act_scales

    @torch.no_grad()
    def apply_smoothing(
        self,
        prev_norm: nn.Module | None,
        linear_layers: list[nn.Linear],
        act_scale: torch.Tensor,
        alpha: float = 0.5,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """
        Compute smoothing scale s = act_scale^alpha / weight_scale^(1-alpha),
        smooth linear weights W = W * diag(s), and update preceding norm / module.
        """
        # Aggregate weight scales across all branches sharing this input (e.g. q, k, v)
        weight_maxes = []
        for linear in linear_layers:
            # linear.weight: [out_features, in_features]
            w_max = torch.amax(torch.abs(linear.weight), dim=0)
            weight_maxes.append(w_max)

        weight_scale = torch.stack(weight_maxes, dim=0).amax(dim=0).clamp(min=eps)
        act_scale = act_scale.to(device=weight_scale.device, dtype=weight_scale.dtype).clamp(min=eps)

        scale = (act_scale.pow(alpha) / weight_scale.pow(1.0 - alpha)).clamp(min=eps)

        # Scale weights: W = W * scale (broadcast along columns)
        for linear in linear_layers:
            linear.weight.data.mul_(scale.view(1, -1))

        # Adjust preceding norm layer if provided
        if prev_norm is not None:
            if hasattr(prev_norm, "weight") and prev_norm.weight is not None:
                prev_norm.weight.data.div_(scale)
            if hasattr(prev_norm, "bias") and prev_norm.bias is not None:
                prev_norm.bias.data.div_(scale)

        return scale

    def quantize_model(
        self,
        model: nn.Module,
        dataloader: Iterator[Any] | Sequence[Any] | None = None,
        act_scales: dict[str, torch.Tensor] | None = None,
        num_batches: int = 32,
        forward_fn: Callable[[nn.Module, Any], Any] | None = None,
        config: SmoothQuantConfig | None = None,
        inplace: bool = True,
    ) -> nn.Module:
        """
        Calibrate and quantize model using SmoothQuant.
        """
        cfg = config or self.config
        target_model = model if inplace else copy_model(model)

        if act_scales is None:
            if dataloader is None:
                raise ValueError("Either dataloader or pre-computed act_scales must be provided for SmoothQuant")
            act_scales = self.calibrate_activation_scales(
                target_model, dataloader, num_batches=num_batches, forward_fn=forward_fn
            )

        # Smooth and replace layers
        targets = tuple(cfg.target_modules or ())
        excludes = tuple(cfg.exclude_modules or ())

        for name, module in list(target_model.named_modules()):
            if isinstance(module, nn.Linear):
                if any(ex in name for ex in excludes):
                    continue
                if targets and not any(t in name for t in targets):
                    continue

                # Apply smoothing if activation scale exists
                if name in act_scales:
                    w_max = torch.amax(torch.abs(module.weight), dim=0).clamp(min=1e-5)
                    a_scale = act_scales[name].to(
                        device=module.weight.device, dtype=module.weight.dtype
                    ).clamp(min=1e-5)
                    scale = (a_scale.pow(cfg.alpha) / w_max.pow(1.0 - cfg.alpha)).clamp(min=1e-5)
                    module.weight.data.mul_(scale.view(1, -1))

                # Replace with QuantizedLinear
                parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
                parent = target_model.get_submodule(parent_name) if parent_name else target_model
                quantized_layer = QuantizedLinear.from_float(module, cfg)
                setattr(parent, child_name, quantized_layer)

        logger.info("SmoothQuant quantization applied successfully.")
        return target_model


def copy_model(model: nn.Module) -> nn.Module:
    import copy

    return copy.deepcopy(model)
