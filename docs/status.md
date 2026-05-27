<!-- last_updated: 2026-05-27 -->

# Status

## Repo Snapshot

| Field | Value |
|-------|-------|
| Package | `carve-lm` |
| Version | `0.1.0` |
| Development stage | Alpha (`Development Status :: 3 - Alpha`) |
| Python | `>=3.10` |
| License | MIT |
| Default branch | `master` |
| Maintained docs | `docs/status.md`, `docs/roadmap.md` |

## Current Status

- LLM pruning stack is in place for Llama, Qwen2, Qwen3, and Mistral through `carve_lm.llm.*`.
- VLM pruning stack supports Qwen2.5-VL and Qwen3-VL across language, vision, and merger components under `carve_lm.vlm.*`.
- Distillation and evaluation helpers exist for both LLM and VLM flows.
- CI runs lint and tests on `master`; automated coverage is strongest for unit tests and synthetic VLM integration flows.
- Real-model multimodal validation still depends on a manually gated script and suitable GPU environment.

## Recent Changes

Recent history below is bounded to latest meaningful repo changes before this docs consolidation.

| Date | Files | Change |
|------|-------|--------|
| 2026-05-13 | `tests/vlm/unit/test_qwen2_5_vl_support.py` | Fixed Qwen2.5-VL test behavior and tightened unit coverage. |
| 2026-05-13 | `src/carve_lm/vlm/components/language/adapters/models/qwen2_5_vl.py`, `docs/assets/qwen2.5-vl-architecture.md`, `docs/assets/qwen3-vl-architecture.md` | Added model-architecture reference material and updated Qwen2.5-VL adapter support. |
| 2026-05-13 | `setup.py` | Restored packaging entrypoint for editable/install flows. |
| 2026-05-12 | `scripts/validation/validate_real_qwen_vlm.py`, `pyproject.toml`, `uv.lock` | Hardened real-model validation setup and added validation extra dependencies. |
| 2026-05-12 | `.github/workflows/ci.yml`, `tests/integration/*`, `src/carve_lm/*/distillation/*`, `src/carve_lm/*/evaluation/*`, `README.md` | Cleaned repo layout, aligned CI with `master`, expanded integration coverage, and split shared distillation/evaluation internals more clearly. |
| 2026-05-04 | `src/carve_lm/vlm/components/**`, `tests/vlm/unit/test_qwen3_vl_support.py` | Added Qwen3-VL support across language, vision, and merger pruning paths. |
| 2026-05-04 | `src/carve_lm/llm/distillation/data.py`, `src/carve_lm/vlm/distillation/data.py`, distillation tests | Added dataloader utilities for recovery and distillation flows. |
| 2026-05-04 | `src/carve_lm/llm/adapters/**`, `src/carve_lm/vlm/components/*/adapters/**` | Refactored adapter layout into clearer per-model registries and modules. |
| 2026-05-04 | `src/carve_lm/vlm/components/vision/**`, `src/carve_lm/vlm/components/merger/**`, `tests/vlm/unit/test_qwen2_5_vl_support.py` | Added Qwen2.5-VL vision and merger pruning support with synthetic test coverage. |
| 2026-04-13 | `src/carve_lm/llm/auto_model.py`, layer estimators, `tests/llm/unit/test_layer_perplexity.py`, `tests/llm/unit/test_pruned_auto_model.py` | Added layer-pruning override handling and stronger LLM loading / evaluation coverage. |

## Current Issues, Pending Problems, Weaknesses

| Area | Status | Details |
|------|--------|---------|
| Real-model VLM validation | Pending | Qwen2.5-VL and Qwen3-VL validation is still manual-only through `scripts/validation/validate_real_qwen_vlm.py`. CI covers synthetic paths, not full downloaded models. |
| Release process | Weak | Packaging works, but there is no release automation or repeatable published release record yet. |
| API stability | Weak | Project is still Alpha. Public paths may still change, especially around multimodal adapters and component pruners. |
| Recovery workflows | Partial | `scripts/recovery/` and examples have smoke/import coverage, but not full end-to-end training validation in CI. |
| Variant coverage | Partial | Vision and merger pruners are mainly exercised against Qwen-style layouts; broader real-model variation is still unproven. |

## Very Next Step

### Priority

Record real-model validation results for Qwen2.5-VL and Qwen3-VL, then use those results to decide whether repo is ready for first release / Beta push or needs more pruning fixes first.

### Step-by-step plan

1. Prepare validation environment:
   `uv sync --locked --extra dev --extra train --extra validation`
2. Run Qwen2.5-VL validation on GPU:
   `CARVE_LM_RUN_REAL_VLM_VALIDATION=1 uv run python scripts/validation/validate_real_qwen_vlm.py --models qwen2_5_vl --device cuda --dtype float16 --keep-artifacts`
3. Review generated artifacts and record pass/fail for:
   language pruning, vision pruning, merger pruning, and save/load round-trip behavior.
4. Run Qwen3-VL validation with same flow:
   `CARVE_LM_RUN_REAL_VLM_VALIDATION=1 uv run python scripts/validation/validate_real_qwen_vlm.py --models qwen3_vl --device cuda --dtype float16 --keep-artifacts`
5. If any step fails, capture exact failing component, stack trace, and affected module path in `status.md` under issues.
6. If both runs pass, update `status.md` recent changes and issues, then move roadmap focus to release readiness and broader variant hardening.
7. After results are recorded, decide one of two paths:
   release-prep path if both models pass,
   fix-and-repeat path if either model fails.

### Acceptance signal

- Both real-model validation runs have recorded outcomes.
- Repo has an explicit pass/fail note for each VLM component family.
- Next engineering priority is based on recorded evidence, not synthetic-only confidence.

For milestone context, see [roadmap.md](roadmap.md).
