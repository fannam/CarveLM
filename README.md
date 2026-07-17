# CarveLM

Tri-level structured pruning library for LLM and VLM transformer models, published as `carve-lm`.

![Framework Overview](docs/assets/tri-level-framework.png)

## Install

```bash
git clone https://github.com/fannam/CarveLM.git
cd CarveLM
pip install -e .
```

Optional extras:

```bash
pip install -e ".[train]"      # accelerate, datasets, PEFT, wandb — recovery workflows
pip install -e ".[dev]"        # pytest, ruff, build, twine
pip install -e ".[notebooks]"  # jupyter
```

## Package Map

| Module | Description |
|--------|-------------|
| `carve_lm.llm.adapters` | LLM adapter contracts and registry. Concrete adapters: `LlamaModelAdapter`, `Qwen2ModelAdapter`, `Qwen3ModelAdapter` (when available), `MistralModelAdapter`, `GenericDecoderModelAdapter`. |
| `carve_lm.llm.core` | LLM registries (`ESTIMATOR_REGISTRY`, `PRUNER_REGISTRY`), identity pass-through layers, and scoring helpers. |
| `carve_lm.llm.estimators` | **Tri-level** LLM importance estimators. Factory: `create_estimator`. |
| `carve_lm.llm.pruners` | **Tri-level** LLM structured pruners. Config types and `create_pruner` factory. |
| `carve_lm.vlm.components.language.adapters` | Decoder-language adapter contracts and registry for multimodal models. Qwen2.5-VL and Qwen3-VL adapters are registered when the local `transformers` build exposes them. |
| `carve_lm.vlm.components.language.estimators` | Tri-level VLM estimators for decoder-side pruning on the language component. |
| `carve_lm.vlm.components.language.pruners` | Tri-level VLM pruners for decoder-side pruning on the language component. |
| `carve_lm.vlm.components.vision.*` | Vision-component adapters, estimators, and pruners for Qwen-style visual transformers. |
| `carve_lm.vlm.components.merger.*` | Patch-merger adapters, estimators, and pruners for Qwen-style multimodal bridge modules. |
| `carve_lm.llm.distillation` | LLM knowledge-distillation helpers: `LogitsDistiller`, `HybridDistiller`, and `HybridOTDistiller`. |
| `carve_lm.llm.finetuning` | LLM training helpers: `SupervisedFineTuner` and `LoRAFineTuner` (requires `[train]`). |
| `carve_lm.llm.evaluation` | Text-generation latency and throughput measurement via `LLMMeasurer`. |
| `carve_lm.llm.auto_model` | Reload component-pruned LLMs. `PrunedAutoModelForCausalLM` replays the identity-passthrough attention/MLP sublayers recorded on the config after a normal HF load. |
| `carve_lm.vlm.distillation` | VLM recovery helpers with multimodal batch forwarding for decoder-side distillation. |
| `carve_lm.vlm.finetuning` | VLM training helpers: `SupervisedFineTuner` and `LoRAFineTuner` (requires `[train]`). |
| `carve_lm.vlm.evaluation` | Multimodal generation latency and throughput measurement via `VLMMeasurer`. |
| `carve_lm.vlm.auto_model` | Reload component-pruned VLMs. `PrunedVLMAutoModel` / `apply_component_pruning_from_config` restore the pruned language-decoder layout after a normal HF load. |

## Architecture

The LLM and VLM stacks are thin, model-family-specific layers over two shared,
model-agnostic cores:

| Shared core | What it holds | How a domain plugs in |
|-------------|---------------|-----------------------|
| `carve_lm._pruning` | The whole structured-pruning engine — discovery, importance selection, executors, manifest/persistence, the width pruner, and the element pruning strategies. | Each domain re-exports the engine and injects its own **adapter resolver**, **estimator factory**, **strategy registry**, and **manifest filename** through class-level hooks. |
| `carve_lm._distillation` | The knowledge-distillation core — logits, hybrid feature, hybrid-OT distillers, and batch/loss helpers. | Each domain re-exports the distillers; only the multimodal data collator (`data.py`) stays domain-specific. |
| `carve_lm._finetuning` | Training core — supervised fine-tuning and PEFT LoRA adapter training. | LLM and VLM namespaces re-export the same trainers. |

The upshot: pruning, knowledge-distillation, and supervised fine-tuning logic
live in exactly one place, so a fix or feature reaches LLM and VLM at once. A model family is supported by writing
an **adapter** (see below) — never by copying the engine. The batch helpers are
written for the general (multimodal) case and degrade gracefully to plain
text-only models, so the same code path serves both.

`SupervisedFineTuner` optimizes the model's loss from dataset `labels`; it does
not use a teacher model. Teacher/student loss matching belongs to the
`*.distillation` namespaces.

## LoRA Fine-Tuning

`LoRAFineTuner` applies a PEFT `LoraConfig` before using the same supervised
training loop as `SupervisedFineTuner`. Only LoRA adapters are trainable and
`save_model` writes adapter weights plus their PEFT config.

```python
from peft import LoraConfig, TaskType

from carve_lm.llm.finetuning import LoRAFineTuner

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
trainer = LoRAFineTuner(
    model=model,
    lora_config=lora_config,
    train_loader=train_loader,
    val_loader=eval_loader,
    tokenizer=tokenizer,
    config={"num_epochs": 3, "learning_rate": 2e-4},
)
trainer.train()
trainer.save_model("artifacts/lora")
```

Choose `target_modules` for model architecture. Same API is available from
`carve_lm.vlm.finetuning` for PEFT-supported VLMs.

## Tri-level Framework

Pruning operates at three independent levels:

| Level | Target | Pruners | Estimators |
|-------|--------|---------|------------|
| **Element** (L1) | Attention heads, GQA groups, MLP neurons, embedding channels | `WidthGroupPruner`, `WidthChannelPruner` | `activation.*`, `magnitude.*`, `taylor.*`, `random.*` |
| **Layer** (L2) | Attention or MLP sublayer within a decoder block | `ComponentPruner` | `similarity.layer` |
| **Block** (L3) | Contiguous decoder blocks | `DepthBlockPruner`, `DepthLayerPruner` | `similarity.block`, `perplexity.block` |

### Registered estimators and pruners by component

Names are the registry keys accepted by `create_estimator` / `create_pruner`
(and the `EstimatorSpec` name). The decoder-side stacks (LLM and VLM language)
are fully symmetric.

| Component | Estimators | Pruners |
|-----------|------------|---------|
| `carve_lm.llm` (decoder) | `activation.element`, `magnitude.{element,group,channel}`, `taylor.group`, `random.group`, `similarity.{layer,block}`, `perplexity.{layer,block}` | `width`, `width.group`, `width.channel`, `component`, `depth.block`, `depth.layer` |
| `carve_lm.vlm.components.language` (decoder) | same as LLM | same as LLM |
| `carve_lm.vlm.components.vision` | `activation.element`, `magnitude.element`, `random.element`, `similarity.{layer,block}` | `width`, `width.channel`, `depth.layer` |
| `carve_lm.vlm.components.merger` | `activation.element`, `magnitude.element`, `random.element` | `width`, `width.bridge` |

Notes on estimator meaning:

- **`activation.*`** — data-driven; runs the model over a calibration
  `dataloader` and scores channels/heads by activation magnitude.
- **`magnitude.*`** — data-free; scores from weight norms (`l1`/`l2`).
- **`taylor.group`** — first/second-order salience from causal-LM loss gradients
  (decoder-side only, where a task loss exists).
- **`random.*`** — reproducible random baseline for measuring how much a real
  estimator actually buys you.
- **`similarity.layer` / `similarity.block`** — cosine distance between a
  sublayer/block input and output; low change ⇒ safe to drop.
- **`perplexity.layer` / `perplexity.block`** — perplexity increase when a
  sublayer/block is replaced by identity; higher ⇒ more important.

## Supported Models

Natively registered adapters:

- `carve_lm.llm.adapters`
- **Llama** (`LlamaForCausalLM`) — Llama 2 / 3 family
- **Qwen2** (`Qwen2ForCausalLM`) — Qwen 2 / 2.5 family
- **Qwen3** (`Qwen3ForCausalLM`) — Qwen 3 family
- **Mistral** (`MistralForCausalLM`) — Mistral family

- `carve_lm.vlm.components.language.adapters`
- **Qwen2.5-VL** (`Qwen2_5_VLForConditionalGeneration`) — decoder-side, vision-component, and merger-component pruning when supported by the installed `transformers`.
- **Qwen3-VL** (`Qwen3VLForConditionalGeneration`) — decoder-side, vision-component, and merger-component pruning when supported by the installed `transformers`.

Any LLM that follows the standard HuggingFace decoder layout (`model.model.layers[*].{self_attn, mlp, input_layernorm, post_attention_layernorm}`) is automatically picked up by `GenericDecoderModelAdapter`.

Custom adapters can be registered at runtime:

```python
from carve_lm.llm.adapters import register_model_adapter, DecoderModelAdapter
from transformers import MyModelForCausalLM

class MyModelAdapter(DecoderModelAdapter):
    def __init__(self):
        super().__init__(name="my_model", model_cls=MyModelForCausalLM)

register_model_adapter(MyModelAdapter())
```

## Quick Start

Classic estimator + pruner flow:

```python
from transformers import AutoModelForCausalLM

from carve_lm.llm.estimators import create_estimator
from carve_lm.llm.pruners import create_pruner

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

estimator = create_estimator("magnitude.element", model, device="cpu")
pruner = create_pruner("width", model, device="cpu")

head_scores = estimator.estimate_attention_heads(agg="l1")
pruned_model = pruner.prune_attention_query(
    head_importance=head_scores,
    target_num_attention_heads=model.config.num_attention_heads // 2,
)
```

Structured pruning flow (GQA-aware):

```python
from carve_lm.llm.pruners import (
    EstimatorSpec,
    WidthGroupConfig,
    WidthGroupPruner,
)

pruner = WidthGroupPruner(
    model,
    WidthGroupConfig(
        pruning_ratio=0.2,
        estimator=EstimatorSpec("magnitude.group", {"norm": "l1"}),
    ),
    device="cpu",
)

context = pruner.discover()
scores = pruner.estimate(dataloader=None)
plan = pruner.select(scores)
result = pruner.apply(plan)

pruner.save_pruned("artifacts/block", result)
reloaded = WidthGroupPruner.load_pruned("artifacts/block", device="cpu")
```

## Canonical Pruning API

```python
from carve_lm.llm.pruners import (
    DepthLayerConfig,
    EstimatorSpec,
    PruningResult,
    WidthChannelConfig,
    WidthChannelPruner,
    WidthGroupConfig,
    WidthGroupPruner,
)
```

Structured block-wise attention groups are GQA-aware:

- **MHA**: one head is one atomic attention group
- **GQA/MQA**: one atomic group = one KV group + all attached query heads + the matching `o_proj` slice

Structured MLP groups are always coupled:

- one `gate_proj` row + one `up_proj` row + one `down_proj` column

## Reloading Pruned Models

There are two persistence paths, matching the two ways a model can be pruned:

**Width / block pruning changes tensor shapes.** Use the pruner's own
`save_pruned` / `load_pruned`, which write a manifest plus the resized weights
and rebuild the pruned architecture on load (see the Quick Start above).

**Component (layer) pruning keeps shapes** — it swaps whole attention/MLP
sublayers for identity pass-throughs and records their indices on the config.
Such models save with the normal HuggingFace `save_pretrained`, and reload with
a CarveLM auto-model that replays the identity layout:

```python
from carve_lm.llm.pruners import ComponentPruner
from carve_lm.llm.auto_model import PrunedAutoModelForCausalLM

pruned = ComponentPruner(model, device="cpu").prune(
    importance_scores={"attention": attn_scores, "mlp": mlp_scores},
    prune_counts={"attention": 2, "mlp": 2},
)
pruned.save_pretrained("artifacts/component")

# Reload: attention/MLP layers recorded on the config become identity modules again.
reloaded = PrunedAutoModelForCausalLM.from_pretrained("artifacts/component")
```

For multimodal models the analogous loader restores the pruned **language**
decoder of a VLM:

```python
from carve_lm.vlm.auto_model import PrunedVLMAutoModel, apply_component_pruning_from_config

reloaded = PrunedVLMAutoModel.from_pretrained("artifacts/vlm_component")
# or, to replay onto an already-loaded model in place:
apply_component_pruning_from_config(model, model_adapter="qwen2_5_vl")
```

## Examples

- [examples/pruning/basic_usage.py](examples/pruning/basic_usage.py)
- [examples/structured/llm_pruner_usage.py](examples/structured/llm_pruner_usage.py)
- [examples/evaluation/measure_latency.py](examples/evaluation/measure_latency.py)

## Recovery Scripts

Post-pruning supervised fine-tuning and knowledge-distillation scripts live under `scripts/recovery/`:

- `finetune_llama.py` — SFT fine-tuning for Llama models
- `finetune_qwen.py` — SFT fine-tuning for Qwen models
- `supervised_finetune_accelerate.py` — supervised fine-tuning with Accelerate

## Development

```bash
pip install -e ".[dev,train]"
ruff check .
pytest
```
