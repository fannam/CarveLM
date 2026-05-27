<!-- last_updated: 2026-05-27 -->

# Roadmap

## Repo Goal

Build `carve-lm` into a dependable structured pruning toolkit for LLM and VLM transformer models, with clear adapter boundaries, reproducible pruning workflows, usable recovery/evaluation tools, and enough validation to move from Alpha toward a first public release.

## Active Milestones

| Priority | Milestone | Target outcome |
|----------|-----------|----------------|
| P0 | Record real-model Qwen2.5-VL and Qwen3-VL validation | Verified language, vision, merger, and persistence behavior on real downloaded models, not only synthetic fixtures. |
| P1 | Harden multimodal pruners against more real variants | Fewer architecture-specific assumptions in `carve_lm.vlm.components.vision.*` and `carve_lm.vlm.components.merger.*`. |
| P1 | Prepare first release path | Repeatable release checklist, successful package build, and explicit go/no-go criteria for leaving Alpha. |
| P1 | Stabilize public API surface | Fewer breaking renames around adapters, pruners, and multimodal component boundaries. |
| P2 | Expand operational coverage | Better end-to-end confidence for examples, recovery scripts, and longer-running workflows outside pure smoke tests. |
| P2 | Broaden evaluation story | Add quality-oriented evaluation signals alongside current latency / throughput helpers. |

## Completed Milestones

| Milestone | Result |
|-----------|--------|
| Core LLM pruning framework | Tri-level LLM estimators, pruners, adapters, and registries ship under `carve_lm.llm.*`. |
| Qwen2.5-VL support | Language, vision, and merger component support landed with synthetic test coverage. |
| Qwen3-VL support | Language, vision, and merger component support landed with synthetic test coverage. |
| Adapter cleanup | LLM and VLM adapters were reorganized into clearer per-model modules and registries. |
| Distillation / evaluation split | Domain-local LLM and VLM distillation/evaluation helpers were separated and covered by tests. |
| CI baseline | `master` branch CI runs lint plus tests through locked `uv` environment. |
| Packaging baseline | `setup.py` restored and package metadata kept installable. |

## Later Milestones

| Horizon | Milestone |
|---------|-----------|
| Later | Support more model families through current adapter extension points, especially beyond Llama / Qwen / Mistral. |
| Later | Improve persistence and pruning robustness across more multimodal architectures, not only Qwen-style layouts. |
| Later | Decide long-term compatibility policy for public API churn before Beta. |
| Stretch | Add stronger documentation/examples for common prune -> recover -> measure workflows. |

## Notes

- `status.md` is operational truth; update it after each meaningful maintenance batch.
- `roadmap.md` should change only when milestone priorities or repo goals change.
- Beta readiness should require recorded real-model multimodal validation, not synthetic-only test success.
