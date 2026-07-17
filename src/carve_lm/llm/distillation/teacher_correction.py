"""Deprecated compatibility module for supervised fine-tuning."""

from __future__ import annotations

import warnings

from carve_lm.llm.finetuning import SupervisedFineTuner

warnings.warn(
    "carve_lm.llm.distillation.teacher_correction is now "
    "carve_lm.llm.finetuning. Import SupervisedFineTuner instead.",
    DeprecationWarning,
    stacklevel=2,
)

TeacherCorrection = SupervisedFineTuner

__all__ = ["TeacherCorrection"]
