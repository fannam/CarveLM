from __future__ import annotations

__all__ = ["LoRAFineTuner", "SupervisedFineTuner"]


def __getattr__(name: str):
    if name == "LoRAFineTuner":
        try:
            from .lora import LoRAFineTuner
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "LoRAFineTuner requires optional training dependencies. "
                "Install the `train` extra to use it."
            ) from exc

        return LoRAFineTuner
    if name == "SupervisedFineTuner":
        try:
            from .supervised import SupervisedFineTuner
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SupervisedFineTuner requires optional training dependencies. "
                "Install the `train` extra to use it."
            ) from exc

        return SupervisedFineTuner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
