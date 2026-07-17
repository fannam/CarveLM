"""Deprecated entrypoint for :mod:`supervised_finetune_accelerate`."""

from __future__ import annotations

import warnings

from scripts.recovery.supervised_finetune_accelerate import main

warnings.warn(
    "teacher_correction_accelerate is now supervised_finetune_accelerate. "
    "Update script references before the next major release.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    main()
