from __future__ import annotations

from importlib import import_module

__all__ = [
    "__package_name__",
    "__version__",
    "llm",
    "quantization",
    "vlm",
]

__version__ = "0.1.0"
__package_name__ = "carve-lm"


def __getattr__(name: str):
    if name in {"llm", "quantization", "vlm"}:
        return import_module(".{}".format(name), __name__)
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
