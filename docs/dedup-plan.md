<!-- last_updated: 2026-07-07 -->

# Deduplication & Module-Completion Plan

## Motivation

Concrete measurements (2026-07-07) show three parallel code trees that are
near-duplicates, where the VLM fork is consistently a *generalization* of the
LLM fork (handles nested `text_config`, dict-shaped multimodal batches, and
adapter-driven `set_layers`). Features drift between forks: hidden-stream
channel pruning exists only in VLM, `layer/perplexity` only in LLM.

Guiding thesis: **adopt the generalized (VLM) implementation as the shared
base; the LLM path becomes a special case.**

### Duplication measured

LLM `llm/pruners/_engine/*` vs VLM `vlm/components/language/pruners/_engine/*`:

| File | LLM lines | Diff lines |
|------|-----------|------------|
| config, estimation, facade, types, utils | 199+53+240+185+105 | 0 (byte-identical) |
| block/depth, layer/component, structured/importance | — | 0 |
| manifest | 96 | 2 (filename only) |
| element/width | 743 | 28 |
| executors | 503 | 115 (VLM-only hidden-stream feature) |
| discovery | 369 | 157 (VLM-only hidden-stream feature) |

Distillation `llm/distillation/*` vs `vlm/distillation/*`:

| File | LLM lines | Diff lines | Note |
|------|-----------|------------|------|
| teacher_correction | 224 | 14 | share |
| logits | 238 | 38 | share (batch-dict generalization) |
| hybrid | 444 | 51 | share |
| _common | 191 | 55 | share |
| data | 193 | 207 | keep domain-specific (multimodal collator) |

Estimated removal: ~2,500 (engine) + ~1,000 (distillation) ≈ **3,500 lines**.

## Principles

- **No public API change.** `carve_lm.llm.pruners.*` and
  `carve_lm.vlm.components.language.pruners.*` keep their import paths; only the
  internals re-export from the shared package.
- **Registries stay domain-local.** `llm/core` and `vlm/.../core` are separate
  registry instances, so key collisions (`"width"`, etc.) cannot happen. The
  shared engine never self-registers; each domain registers its own facade.
- Each phase ends green: `ruff check` + `pytest`. Phases are independent.

## Phase 0 — Prerequisite (blocks Phase 1)

The generalized engine calls `adapter.set_layers(...)` and
`adapter.uses_hidden_stream_channel_pruning(...)`, which LLM `BaseModelAdapter`
lacks.

1. Add `set_layers(model, layers)` to `llm/adapters/base.py`
   (default: `model.model.layers = nn.ModuleList(list(layers))`).
2. Add `uses_hidden_stream_channel_pruning(model=None) -> bool` (returns `False`).
3. Confirm concrete adapters (llama/qwen2/qwen3/mistral) need no override.
4. Run LLM depth-pruning tests.

Risk: low.

## Phase 1 — Merge pruning engine into `carve_lm/_pruning/`

Create `src/carve_lm/_pruning/` holding the generalized (VLM) engine. The only
behavioral parameterization needed:

- `manifest.py`: `MANIFEST_FILENAME` becomes a parameter
  (LLM `llm_pruner_manifest.json`, VLM `vlm_pruner_manifest.json`).
- Hidden-stream branch: already gated by
  `adapter.uses_hidden_stream_channel_pruning()`; LLM returns `False` and never
  enters it. Backport is free and does not change LLM behavior.

Order: build shared engine → point VLM at it → run VLM tests → point LLM at it →
run LLM tests (isolates any LLM regression to the final step).

Sensitive spot: `element/width.py` reads dims from `adapter.get_attention_handles`
in the VLM version (vs `config` in LLM) — verify LLM adapters return correct
num_heads/head_dim.

Tests: `test_structured_pruning`, `test_layer_perplexity`, `test_pruned_auto_model`
(LLM); `test_vlm_qwen_synthetic_e2e`, `test_vlm_persistence_roundtrip` (VLM).

## Phase 2 — Finish distillation dedup

`_distillation/` already holds OT + wrappers. Lift the rest, using the
VLM-generalized (batch-dict) versions:

1. `_common.py` (`move_batch_to_device`, `filter_model_inputs`,
   `prepare_causal_lm_batch` returning a batch dict) → `_distillation/_common.py`.
2. `logits.py`, `hybrid.py`, `teacher_correction.py` → shared; llm/vlm re-export.
3. `data.py` stays domain-specific (multimodal collator genuinely differs).

Risk: low–medium. `filter_model_inputs` filters kwargs by model signature, so
feeding an LLM a dict batch is safe.

Tests: `test_distillation`, `test_distillation_shared`, `test_vlm_distillation`.

## Phase 3 — Fill missing modules (symmetry)

Independent, lowest risk:

1. `vlm/.../language/estimators/layer/perplexity.py` — LLM has it, VLM lacks it.
2. Vision + Merger estimators: add `random.py` and `taylor.py` to match LLM's
   `activation/magnitude/random/taylor` set.
3. `vlm/auto_model.py` → `PrunedVLMAutoModel` — analog of
   `PrunedAutoModelForCausalLM`, replaying component pruning recorded on
   `config.text_config`. Also unblocks the P0 real-model validation milestone.

Risk: low (additive only).

## Recommended order

Phase 0 → 1 → 3 → 2. Engine merge first (largest dedup + makes new Phase-3
modules single-source). Distillation is fully independent, done last. Phase 3
estimator additions are near-zero-risk and can be pulled forward for momentum.

## Progress log

### 2026-07-07 — Phase 0 done

- Added `set_layers` (concrete default: `model.model.layers = nn.ModuleList(...)`)
  and `uses_hidden_stream_channel_pruning` (returns `False`) to
  `llm/adapters/base.py` `BaseModelAdapter`.
- Full suite green: 52 passed / 3 skipped (LLM), 43 passed (VLM+integration).

### 2026-07-07 — Phase 1 (engine core) done

- Created `carve_lm/_pruning/` holding the generalized engine
  (`types, config, utils, discovery, executors, manifest, facade`) plus a
  `_protocol.ModelAdapter` for annotations. Zero domain imports.
- Domain hooks injected via facade class attrs `_resolve_model_adapter`,
  `_estimate_scores`, `_manifest_filename` (set by per-domain shim mixins).
  `manifest.save/load` take a `filename` kwarg.
- Both domains' `_engine/*.py` are now thin re-export shims; `estimation.py`
  stays domain-local (lazy `create_estimator` factory binding).
- Result: engine is single-source. Domain `_engine` shims now differ by only
  the manifest filename (2 lines) and the hook-class name (10 lines).
  **src −1321 lines** (19,828 → 18,507). Full suite green: 95 passed / 3
  skipped. `ruff check .` clean.

### 2026-07-07 — Phase 3 done

- **3a**: ported `_BaseLayerPerplexityEstimator` into VLM
  `estimators/_base.py`; added `layer/perplexity.py` (`perplexity.layer`).
- **3b**: added `random.element` estimators to vision + merger (data-free,
  reproducible, shapes mirror magnitude). Updated `test_namespace_smoke`.
- **3c**: deferred with rationale (see Phase 3 plan) — not fabricated.
- **3d**: added `vlm/auto_model.py` (`PrunedVLMAutoModel`,
  `apply_component_pruning_from_config`) + `tests/vlm/unit/test_pruned_vlm_auto_model.py`.
- Full suite: 97 passed / 3 skipped. `ruff check .` clean.

### 2026-07-07 — Phase 2 done

- Lifted the generalized (VLM) `_common.py`, `logits.py`, `hybrid.py`,
  `teacher_correction.py` into `carve_lm/_distillation/`. Domain copies are now
  5-line re-export shims; `data.py` stays domain-local.
- Updated `test_distillation` wandb monkeypatch target to
  `carve_lm._distillation.hybrid`.
- Net so far from the original baseline: **19,828 → 17,836 lines** (−1,992,
  while Phase 3 *added* ~380 lines of new features). Full suite: 97 passed / 3
  skipped. `ruff check .` clean.

### Remaining (Phase 1b)

Registered pruner wrappers still duplicated because they register strategies to
domain registries via import-time decorators:

| File | LLM lines | Diff |
|------|-----------|------|
| element/width.py | 743 | 28 |
| block/depth.py | 71 | 0 |
| layer/component.py | 56 | 0 |

Sharing these needs registry injection (decorators bind to a specific registry
at import).

### 2026-07-07 — Phase 1b done

- Moved `_BasePruner`/`_BaseLayerPruner`/`_BaseBlockPruner` into
  `carve_lm/_pruning/pruner_base.py` (adapter resolver via `_resolve_model_adapter`
  hook); domain `_base.py` are 32-line binding shims.
- Moved `WidthPruner` + the four element strategies into
  `carve_lm/_pruning/width.py` (generalized/VLM variant). `WidthPruner` takes
  `_strategy_registry` and `_embedding_global_score` hooks; each domain
  `element/width.py` (743 → 125 lines) registers the shared classes into its own
  registry and binds the hooks. `block/depth.py` and `layer/component.py` stay
  domain-local (thin registration + legacy aliases only).
- Domain `element/width.py` now differ by 8 lines (docstring/adapter wording).
  Registries resolve identically across domains.
- Final: **19,828 → 17,147 lines (−2,681 net)**, while Phase 3 added ~380 lines
  of new features. Full suite: 97 passed / 3 skipped. `ruff check .` clean.

## Full execution plan for Phases 3, 2, 1b (2026-07-07)

Order: **3 → 2 → 1b** (additive/safe first, invasive last). Test gate after each
sub-phase: `pytest` (95/3 baseline) + `ruff check .`.

### Phase 3 — fill missing modules (additive, low risk)

Investigation results:
- VLM `estimators/_base.py` has `_BaseBlockPerplexityEstimator` but **not**
  `_BaseLayerPerplexityEstimator` (LLM `_base.py` differs by ~199 lines, that
  class being the main gap).
- Vision estimators use a per-layer interface
  (`estimate_attention_heads/mlp_neurons/hidden_channels -> Dict`), NOT the
  structured-group `estimate(context)` interface — so LLM `random.py`/`taylor.py`
  cannot be copied verbatim; they must match the vision/merger interface.
- Merger interface: `estimate_input/intermediate/output_channels`.
- Vision activation estimator uses forward hooks under `no_grad`; taylor adds a
  backward pass (activation × gradient).
- VLM component pruning records layers on `config.text_config`; language layers
  live at `model.model.language_model.layers`.

Steps:
1. **3a** — Port `_BaseLayerPerplexityEstimator` into VLM
   `estimators/_base.py`; add `estimators/layer/perplexity.py`
   (`perplexity.layer`) + register in `layer/__init__.py`. Verify text-only
   forward works on the synthetic VLM.
2. **3b** — Add `random.py` to vision + merger estimators: implement the
   domain's `estimate_*` methods returning random tensors of the same shapes the
   magnitude estimator produces. Register `random.element` (vision) /
   `random.*` (merger).
3. **3c — DEFERRED (design item, not a gap).** LLM `taylor.group` scores
   discovered groups from **causal-LM loss gradients**; the VLM *language*
   element estimators already ship `taylor.py` (same interface). Vision and
   merger use a different per-block interface with no local task loss to
   backprop, so a meaningful taylor there is a new gradient-path design, not a
   copy of the LLM estimator. Not fabricated. Their symmetric set is
   `activation + magnitude + random` (+ `similarity` for vision).
4. **3d** — Add `vlm/auto_model.py` `PrunedVLMAutoModel` replaying component
   pruning from `config.text_config` on `model.model.language_model.layers`.

### Phase 2 — finish distillation dedup

Investigation: VLM `_common.py` is the generalized version — adds
`text_config`, `text_hidden_size`, `text_num_hidden_layers`,
`filter_model_inputs`, and a `prepare_causal_lm_batch` that returns
`(batch_dict, shifted_labels, loss_mask)` (vs the LLM tuple of
`input_ids, attention_mask, ...`). The `text_*` helpers fall back to a flat
config, so they are safe for LLM.

Steps:
1. Lift the VLM versions of `_common.py`, `logits.py`, `hybrid.py`,
   `teacher_correction.py` into `carve_lm/_distillation/` (which already holds
   `wrappers.py`, `optimal_transport.py`, `hybrid_ot.py`).
2. Replace `llm/distillation/*` and `vlm/distillation/*` copies with re-export
   shims. Keep `data.py` domain-local (multimodal collator, 207-line diff).
3. Verify LLM distillation still runs with the batch-dict API.

### Phase 1b — share registered pruner wrappers

Investigation: `_base.py` and `core/registry.py` are byte-identical across
domains (coupled only by relative adapter/registry imports). `element/width.py`
differs by the same 28-line generalization (reads dims from adapter handles;
guards `set_head_dim` behind the hidden-stream flag) — adopt the VLM version.

Steps:
1. Move the pruner + strategy classes (`element/width.py`, `block/depth.py`,
   `layer/component.py`) into `carve_lm/_pruning/` **undecorated**, plus a shared
   `_BasePruner` (the `_base.py` body) parameterized by an adapter resolver hook.
2. Each domain's wrapper module imports the shared classes and registers them
   into its own `PRUNER_REGISTRY` / `PRUNING_STRATEGY_REGISTRY`, keeping the
   legacy alias subclasses.
3. Verify registries resolve the same names and all pruning tests pass.

Expected additional dedup: ~840 lines.
