from __future__ import annotations

from carve_lm._pruning.manifest import STATE_FILENAME, build_manifest
from carve_lm._pruning.manifest import load_pruned_result as _load_pruned_result
from carve_lm._pruning.manifest import save_pruned_result as _save_pruned_result

MANIFEST_FILENAME = "llm_pruner_manifest.json"


def save_pruned_result(output_dir, result, *, filename: str = MANIFEST_FILENAME):
    return _save_pruned_result(output_dir, result, filename=filename)


def load_pruned_result(pruner_cls, output_dir, device=None, dtype=None, *, filename: str = MANIFEST_FILENAME):
    return _load_pruned_result(pruner_cls, output_dir, device=device, dtype=dtype, filename=filename)


__all__ = [
    "MANIFEST_FILENAME",
    "STATE_FILENAME",
    "build_manifest",
    "load_pruned_result",
    "save_pruned_result",
]
