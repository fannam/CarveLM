from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Sequence


class QuantPrecision(str, Enum):
    INT8 = "int8"
    INT4 = "int4"
    FP8 = "fp8"

    @property
    def num_bits(self) -> int:
        if self in {QuantPrecision.INT8, QuantPrecision.FP8}:
            return 8
        if self == QuantPrecision.INT4:
            return 4
        raise ValueError(f"Unknown precision: {self}")


class QuantScheme(str, Enum):
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"


class QuantGranularity(str, Enum):
    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    PER_GROUP = "per_group"


class QuantMethod(str, Enum):
    RTN = "rtn"
    SMOOTHQUANT = "smoothquant"
    GPTQ = "gptq"
    AWQ = "awq"


@dataclass
class QuantConfig:
    """
    Base configuration for model quantization.
    """

    method: str | QuantMethod = QuantMethod.RTN
    bits: int = 8
    scheme: str | QuantScheme = QuantScheme.SYMMETRIC
    granularity: str | QuantGranularity = QuantGranularity.PER_CHANNEL
    group_size: int | None = 128
    act_bits: int | None = None
    target_modules: Sequence[str] | None = None
    exclude_modules: Sequence[str] | None = field(
        default_factory=lambda: ("lm_head", "embed_tokens")
    )
    pack_weights: bool = True

    def __post_init__(self):
        if isinstance(self.method, str):
            self.method = QuantMethod(self.method.lower())
        if isinstance(self.scheme, str):
            self.scheme = QuantScheme(self.scheme.lower())
        if isinstance(self.granularity, str):
            self.granularity = QuantGranularity(self.granularity.lower())
        if self.exclude_modules is not None:
            self.exclude_modules = tuple(self.exclude_modules)
        if self.target_modules is not None:
            self.target_modules = tuple(self.target_modules)
        if self.bits not in {4, 8}:
            raise ValueError(f"Supported bits are 4 and 8, got {self.bits}")
        if self.granularity == QuantGranularity.PER_GROUP:
            if not self.group_size or self.group_size <= 0:
                raise ValueError(
                    f"group_size must be a positive integer when granularity is PER_GROUP, got {self.group_size}"
                )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["method"] = self.method.value if isinstance(self.method, QuantMethod) else str(self.method)
        payload["scheme"] = self.scheme.value if isinstance(self.scheme, QuantScheme) else str(self.scheme)
        payload["granularity"] = (
            self.granularity.value if isinstance(self.granularity, QuantGranularity) else str(self.granularity)
        )
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QuantConfig":
        return cls(**payload)


@dataclass
class WeightQuantConfig(QuantConfig):
    """
    Weight-only quantization configuration (RTN).
    """

    method: str | QuantMethod = QuantMethod.RTN
    act_bits: int | None = None


@dataclass
class SmoothQuantConfig(QuantConfig):
    """
    Configuration for SmoothQuant (W8A8 / W4A8 activation-weight smoothing).
    """

    method: str | QuantMethod = QuantMethod.SMOOTHQUANT
    bits: int = 8
    act_bits: int = 8
    alpha: float = 0.5
    granularity: str | QuantGranularity = QuantGranularity.PER_CHANNEL

    def __post_init__(self):
        super().__post_init__()
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"SmoothQuant alpha must be between 0.0 and 1.0, got {self.alpha}")


@dataclass
class GPTQConfig(QuantConfig):
    """
    Configuration for GPTQ data-aware second-order error compensation.
    """

    method: str | QuantMethod = QuantMethod.GPTQ
    bits: int = 4
    damp_percent: float = 0.01
    block_size: int = 128
    perchannel: bool = True
    act_order: bool = False

    def __post_init__(self):
        super().__post_init__()
        if not (0.0 < self.damp_percent < 1.0):
            raise ValueError(f"damp_percent must be in (0, 1), got {self.damp_percent}")


@dataclass
class AWQConfig(QuantConfig):
    """
    Configuration for Activation-aware Weight Quantization (AWQ).
    Protects salient weight channels by searching for optimal per-channel activation scaling.
    """

    method: str | QuantMethod = QuantMethod.AWQ
    bits: int = 4
    granularity: str | QuantGranularity = QuantGranularity.PER_GROUP
    group_size: int | None = 128
    n_grid: int = 20
    duo_scaling: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.n_grid <= 0:
            raise ValueError(f"n_grid must be a positive integer, got {self.n_grid}")
