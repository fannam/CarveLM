"""Deprecated compatibility module for supervised fine-tuning."""

from __future__ import annotations

import warnings

from carve_lm.vlm.finetuning import SupervisedFineTuner

warnings.warn(
    "carve_lm.vlm.distillation.teacher_correction is now "
    "carve_lm.vlm.finetuning. Import SupervisedFineTuner instead.",
    DeprecationWarning,
    stacklevel=2,
)

TeacherCorrection = SupervisedFineTuner

__all__ = ["TeacherCorrection"]
