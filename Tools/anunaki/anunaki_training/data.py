from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray


class AnunakiTerrainDataset(Dataset):
    def __init__(self, dataset_path: Path, indices: np.ndarray):
        data = np.load(dataset_path)
        self.terrain = torch.from_numpy(data["terrain"]).float()
        self.conditioning = torch.from_numpy(data["conditioning"]).float()
        self.indices = torch.from_numpy(indices.astype(np.int64))

        if self.terrain.ndim != 4 or self.terrain.shape[1] != 1:
            raise ValueError("dataset terrain must be [N,1,H,W]")
        if self.conditioning.ndim != 4 or self.conditioning.shape[1] != 5:
            raise ValueError("dataset conditioning must be [N,5,H,W]")
        if self.terrain.shape[0] != self.conditioning.shape[0]:
            raise ValueError("terrain and conditioning sample counts differ")

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        terrain = self.terrain[index]
        conditioning = self.conditioning[index]
        return build_training_targets(terrain, conditioning)


def split_indices(sample_count: int, val_fraction: float, seed: int) -> SplitIndices:
    rng = np.random.default_rng(seed)
    indices = np.arange(sample_count)
    rng.shuffle(indices)
    val_count = max(1, int(round(sample_count * val_fraction))) if sample_count > 1 else 0
    return SplitIndices(train=indices[val_count:], val=indices[:val_count])


def dataset_sample_count(dataset_path: Path) -> int:
    data = np.load(dataset_path)
    return int(data["terrain"].shape[0])


def limit_indices(indices: np.ndarray, limit: int | None) -> np.ndarray:
    return indices if limit is None else indices[:limit]


def build_training_targets(terrain: torch.Tensor, conditioning: torch.Tensor) -> dict[str, torch.Tensor]:
    terrain_b = terrain.unsqueeze(0)
    conditioning_b = conditioning.unsqueeze(0)
    terrain64 = F.interpolate(terrain_b, size=(64, 64), mode="bilinear", align_corners=False).squeeze(0)
    conditioning64 = F.interpolate(conditioning_b, size=(64, 64), mode="bilinear", align_corners=False).squeeze(0)

    slope = conditioning64[1:2]
    curvature = conditioning64[2:3]
    basin = conditioning64[3:4]
    style = conditioning64[4:5]
    coarse_target = torch.cat([terrain64, slope, curvature, basin, style, terrain64 * style], dim=0)
    base_target = torch.cat([terrain64, slope, curvature, basin, style], dim=0)
    decoder_target = terrain

    return {
        "conditioning64": conditioning64,
        "coarse_target": coarse_target,
        "base_target": base_target,
        "decoder_target": decoder_target,
    }

