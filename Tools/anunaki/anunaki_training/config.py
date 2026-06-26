from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    dataset: Path
    val_fraction: float = 0.08
    seed: int = 5819
    max_train_samples: int | None = None
    max_val_samples: int | None = None


@dataclass(frozen=True)
class ModelConfig:
    width: int = 64
    depth: int = 8


@dataclass(frozen=True)
class OptimConfig:
    epochs: int = 80
    batch_size: int = 2
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    grad_accum_steps: int = 1


@dataclass(frozen=True)
class RunConfig:
    out_dir: Path
    device: str = "auto"
    sample_every_epochs: int = 5
    checkpoint_every_epochs: int = 5
    num_workers: int = 0
    deterministic: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig
    run: RunConfig

    @staticmethod
    def load(path: str | Path) -> "TrainingConfig":
        raw = yaml.safe_load(Path(path).read_text())
        return TrainingConfig.from_dict(raw)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "TrainingConfig":
        return TrainingConfig(
            data=DataConfig(**_coerce_paths(raw["data"], ["dataset"])),
            model=ModelConfig(**raw.get("model", {})),
            optim=OptimConfig(**raw.get("optim", {})),
            run=RunConfig(**_coerce_paths(raw["run"], ["out_dir"])),
        )

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "data": _jsonable(self.data),
            "model": _jsonable(self.model),
            "optim": _jsonable(self.optim),
            "run": _jsonable(self.run),
        }


def _coerce_paths(raw: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    out = dict(raw)
    for key in keys:
        if key in out:
            out[key] = Path(out[key])
    return out


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, Path):
        return str(value)
    return value

