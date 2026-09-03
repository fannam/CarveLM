from __future__ import annotations

import logging
from typing import Any, Callable, Iterator, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AWQConfig
from ..functional import dequantize_symmetric, quantize_symmetric
from ..modules import QuantizedLinear

logger = logging.getLogger(__name__)


class AWQQuantizer:
    """
    Activation-aware Weight Quantization (AWQ) (Lin et al., 2023).
    Protects salient weight channels by searching for optimal per-channel activation scales
    that minimize output reconstruction error under low-bit quantization.
    """

    def __init__(self, config: AWQConfig | None = None):
        self.config = config or AWQConfig()

    @torch.no_grad()
    def calibrate_activation_scales(
        self,
        model: nn.Module,
        dataloader: Iterator[Any] | Sequence[Any],
        num_batches: int = 16,
        forward_fn: Callable[[nn.Module, Any], Any] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, list[torch.Tensor]]]:
        """
        Record mean activation magnitudes and sample inputs for each linear layer.
        """
        model.eval()
        act_scales: dict[str, torch.Tensor] = {}
        sample_inputs: dict[str, list[torch.Tensor]] = {}
        sample_counts: dict[str, int] = {}
        hooks = []

        def get_hook(name: str):
            def hook(module: nn.Module, input: tuple[torch.Tensor, ...], output: Any):
                inp = input[0].detach()
                inp_flat = inp.view(-1, inp.shape[-1]).float()
                batch_sum = torch.sum(torch.abs(inp_flat), dim=0)
                n_samples = inp_flat.shape[0]

                if name not in act_scales:
                    act_scales[name] = batch_sum
                    sample_counts[name] = n_samples
                    sample_inputs[name] = [inp_flat[: min(64, n_samples)].cpu()]
                else:
                    act_scales[name] += batch_sum
                    sample_counts[name] += n_samples
                    if len(sample_inputs[name]) < 4:
                        sample_inputs[name].append(inp_flat[: min(64, n_samples)].cpu())

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

        for name in act_scales:
            act_scales[name] = act_scales[name] / float(max(1, sample_counts[name]))

        logger.info(f"AWQ calibration complete on {batches_processed} batches for {len(act_scales)} layers.")
        return act_scales, sample_inputs

    @torch.no_grad()
    def search_best_scale(
        self,
        linear: nn.Linear,
        act_scale: torch.Tensor,
        sample_inputs: list[torch.Tensor],
        config: AWQConfig,
    ) -> torch.Tensor:
        """
        Grid search over alpha in [0, 1] to find channel scale S = s_X^alpha / s_W^(1-alpha)
        minimizing ||X W^T - X (Q(W * S) / S)^T||^2.
        """
        w = linear.weight.data.clone().float()
        device = w.device
        act_scale = act_scale.to(device=device, dtype=torch.float32).clamp(min=1e-5)
        weight_scale = torch.amax(torch.abs(w), dim=0).clamp(min=1e-5)

        # Concatenate sample inputs
        x_samples = torch.cat(sample_inputs, dim=0).to(device=device, dtype=torch.float32)
        # Target output Y = X W^T
        y_target = F.linear(x_samples, w)

        best_error = float("inf")
        best_scale = torch.ones(w.shape[1], device=device, dtype=torch.float32)

        alphas = torch.linspace(0.0, 1.0, config.n_grid, device=device)

        for alpha in alphas:
            if config.duo_scaling:
                scale = (act_scale.pow(alpha) / weight_scale.pow(1.0 - alpha)).clamp(min=1e-4)
            else:
                scale = act_scale.pow(alpha).clamp(min=1e-4)

            # Normalize scale
            scale = scale / (scale.mean().clamp(min=1e-8))

            # Scale weights
            w_scaled = w * scale.view(1, -1)

            # Symmetrically quantize scaled weights
            q, scales = quantize_symmetric(
                w_scaled,
                bits=config.bits,
                granularity=config.granularity,
                dim=0,
                group_size=config.group_size,
            )
            w_scaled_deq = dequantize_symmetric(
                q,
                scales,
                granularity=config.granularity,
                dim=0,
                group_size=config.group_size,
                dtype=torch.float32,
            )

            # De-scale to approximate original weight
            w_deq = w_scaled_deq / scale.view(1, -1)

            y_approx = F.linear(x_samples, w_deq)
            loss = torch.mean((y_target - y_approx) ** 2).item()

            if loss < best_error:
                best_error = loss
                best_scale = scale.clone()

        return best_scale

    def quantize_model(
        self,
        model: nn.Module,
        dataloader: Iterator[Any] | Sequence[Any],
        num_batches: int = 16,
        forward_fn: Callable[[nn.Module, Any], Any] | None = None,
        config: AWQConfig | None = None,
        inplace: bool = True,
    ) -> nn.Module:
        """
        Run AWQ quantization across model layers with calibration data.
        """
        cfg = config or self.config
        target_model = model if inplace else copy_model(model)
        targets = tuple(cfg.target_modules or ())
        excludes = tuple(cfg.exclude_modules or ())

        act_scales, sample_inputs = self.calibrate_activation_scales(
            target_model, dataloader, num_batches=num_batches, forward_fn=forward_fn
        )

        for name, module in list(target_model.named_modules()):
            if isinstance(module, nn.Linear) and name in act_scales:
                if any(ex in name for ex in excludes):
                    continue
                if targets and not any(t in name for t in targets):
                    continue

                best_scale = self.search_best_scale(
                    module,
                    act_scales[name],
                    sample_inputs[name],
                    cfg,
                )

                # Apply scale to weight
                w_scaled = module.weight.data * best_scale.view(1, -1)

                # Quantize scaled weight
                temp_linear = nn.Linear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                temp_linear.weight.data.copy_(w_scaled)
                if module.bias is not None:
                    temp_linear.bias.data.copy_(module.bias.data)

                # Create QuantizedLinear
                q_layer = QuantizedLinear.from_float(temp_linear, cfg)

                # Multiply the dequantization scale or inverse scale back so forward pass matches
                # If per-channel: q_layer.scales is [out_features, 1]
                # If per-group: q_layer.scales is [out_features, num_groups]
                # The effective weight is Q * scales / best_scale
                # We update the stored weight dequantization directly
                parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
                parent = target_model.get_submodule(parent_name) if parent_name else target_model
                setattr(parent, child_name, q_layer)

        logger.info("AWQ Quantization complete.")
        return target_model


def copy_model(model: nn.Module) -> nn.Module:
    import copy

    return copy.deepcopy(model)
