"""Deprecated compatibility module for :mod:`carve_lm._finetuning`."""

from __future__ import annotations

import warnings

from carve_lm._finetuning import SupervisedFineTuner

warnings.warn(
    "TeacherCorrection has been renamed to SupervisedFineTuner and moved to "
    "carve_lm._finetuning. Update imports before the next major release.",
    DeprecationWarning,
    stacklevel=2,
)

TeacherCorrection = SupervisedFineTuner

__all__ = ["TeacherCorrection"]
