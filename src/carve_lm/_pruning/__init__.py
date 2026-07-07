"""Domain-agnostic structured-pruning engine shared by the LLM and VLM stacks.

Both ``carve_lm.llm.pruners`` and
``carve_lm.vlm.components.language.pruners`` re-export this engine through thin
per-domain shims. The engine never imports a concrete adapter, estimator
registry, or manifest filename directly; each domain injects those through the
facade hooks (``_resolve_model_adapter``, ``_estimate_scores``,
``_manifest_filename``).
"""
from __future__ import annotations
