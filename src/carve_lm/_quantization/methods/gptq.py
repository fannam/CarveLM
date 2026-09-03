from __future__ import annotations

import logging
from typing import Any, Callable, Iterator, Sequence

import torch
import torch.nn as nn

from ..config import GPTQConfig
from ..functional import quantize_symmetric
from ..modules import QuantizedLinear

logger = logging.getLogger(__name__)


class GPTQQuantizer:
    """
    GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.
    Computes second-order Hessian inverse for each linear layer and minimizes quantization error
    using optimal brain surgeon updates.
    """

    def __init__(self, config: GPTQConfig | None = None):
        self.config = config or GPTQConfig()

    @torch.no_grad()
    def quantize_layer(
        self,
        linear: nn.Linear,
        inputs: list[torch.Tensor],
        config: GPTQConfig | None = None,
    ) -> QuantizedLinear:
        """
        Quantize a single linear layer using Hessian-based error compensation.
        inputs: list of activation tensors [batch_size, seq_len, in_features] or [..., in_features].
        """
        cfg = config or self.config
        w = linear.weight.data.clone().float()
        out_features, in_features = w.shape
        device = w.device

        # Compute empirical Hessian H = 2 * X X^T
        H = torch.zeros((in_features, in_features), device=device, dtype=torch.float32)
        total_samples = 0

        for inp in inputs:
            inp_flat = inp.view(-1, in_features).float().to(device)
            H += inp_flat.t().matmul(inp_flat)
            total_samples += inp_flat.shape[0]

        if total_samples > 0:
            H /= float(total_samples)

        # Dampening for numerical stability
        diag = torch.diag(H)
        damp = cfg.damp_percent * torch.mean(diag)
        H += damp * torch.eye(in_features, device=device)

        # Invert Hessian via Cholesky decomposition
        try:
            chol = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(chol)
        except Exception:
            H_inv = torch.pinverse(H)

        # Block-wise column quantization with error compensation
        block_size = min(cfg.block_size, in_features)
        w_quant = w.clone()

        for col_idx in range(0, in_features, block_size):
            col_end = min(col_idx + block_size, in_features)
            w_block = w_quant[:, col_idx:col_end]
            h_inv_block = H_inv[col_idx:col_end, col_idx:col_end]

            # Quantize the block weights using configured scheme
            q_block, scales_block = quantize_symmetric(
                w_block,
                bits=cfg.bits,
                granularity=cfg.granularity,
                dim=0,
                group_size=cfg.group_size if cfg.granularity == "per_group" else None,
            )
            w_block_deq = q_block.float() * scales_block.view(out_features, -1).float()
            err_block = w_block - w_block_deq

            w_quant[:, col_idx:col_end] = w_block_deq

            # Update remaining columns
            if col_end < in_features:
                h_inv_rest = H_inv[col_idx:col_end, col_end:]
                try:
                    delta = err_block.matmul(torch.linalg.solve(h_inv_block, h_inv_rest))
                    w_quant[:, col_end:] -= delta
                except Exception:
                    pass

        # Update layer weights with GPTQ-optimized weights and construct QuantizedLinear
        temp_linear = nn.Linear(in_features, out_features, bias=linear.bias is not None, device=device)
        temp_linear.weight.data.copy_(w_quant)
        if linear.bias is not None:
            temp_linear.bias.data.copy_(linear.bias.data)

        quantized_linear = QuantizedLinear.from_float(temp_linear, cfg)
        return quantized_linear

    def quantize_model(
        self,
        model: nn.Module,
        dataloader: Iterator[Any] | Sequence[Any],
        num_batches: int = 16,
        forward_fn: Callable[[nn.Module, Any], Any] | None = None,
        config: GPTQConfig | None = None,
        inplace: bool = True,
    ) -> nn.Module:
        """
        Run GPTQ layer-by-layer quantization over the model with calibration samples.
        """
        cfg = config or self.config
        target_model = model if inplace else copy_model(model)
        targets = tuple(cfg.target_modules or ())
        excludes = tuple(cfg.exclude_modules or ())

        # Collect layer inputs
        layer_inputs: dict[str, list[torch.Tensor]] = {}
        hooks = []

        def get_hook(name: str):
            def hook(module: nn.Module, input: tuple[torch.Tensor, ...], output: Any):
                if name not in layer_inputs:
                    layer_inputs[name] = []
                if len(layer_inputs[name]) < num_batches:
                    layer_inputs[name].append(input[0].detach().cpu())

            return hook

        for name, module in target_model.named_modules():
            if isinstance(module, nn.Linear):
                if any(ex in name for ex in excludes):
                    continue
                if targets and not any(t in name for t in targets):
                    continue
                hooks.append(module.register_forward_hook(get_hook(name)))

        batches_processed = 0
        with torch.no_grad():
            for batch in dataloader:
                if forward_fn is not None:
                    forward_fn(target_model, batch)
                elif isinstance(batch, dict):
                    target_model(**batch)
                elif isinstance(batch, (list, tuple)):
                    target_model(*batch)
                else:
                    target_model(batch)

                batches_processed += 1
                if batches_processed >= num_batches:
                    break

        for hook in hooks:
            hook.remove()

        # Quantize layers
        for name, module in list(target_model.named_modules()):
            if isinstance(module, nn.Linear) and name in layer_inputs:
                inputs = layer_inputs[name]
                quantized_layer = self.quantize_layer(module, inputs, cfg)
                parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
                parent = target_model.get_submodule(parent_name) if parent_name else target_model
                setattr(parent, child_name, quantized_layer)

        logger.info("GPTQ Quantization complete.")
        return target_model


def copy_model(model: nn.Module) -> nn.Module:
    import copy

    return copy.deepcopy(model)
