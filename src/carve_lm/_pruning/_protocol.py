from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelAdapter(Protocol):
    """Structural contract the shared pruning engine relies on.

    Concrete adapters live in the LLM and VLM adapter packages; the engine only
    needs the structural surface below. Kept intentionally minimal — it exists
    so engine modules can annotate ``adapter`` parameters without importing a
    domain-specific base class.
    """

    name: str
