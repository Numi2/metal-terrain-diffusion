from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class DeployableConvStack(nn.Module):
    """Conv/SiLU-only network that maps exactly onto MetalGraphDenoiser ops."""

    def __init__(self, in_channels: int, out_channels: int, width: int, depth: int):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2")

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=False),
            nn.SiLU(),
        ]
        for _ in range(depth - 2):
            layers.extend(
                [
                    nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
                    nn.SiLU(),
                ]
            )
        layers.append(nn.Conv2d(width, out_channels, kernel_size=3, padding=1, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class StageSpec:
    name: str
    input_channels: int
    output_channels: int
    tile_size: int
    conditioning_name: str | None


STAGE_SPECS = {
    "coarse": StageSpec("coarse", input_channels=6, output_channels=6, tile_size=64, conditioning_name="conditioning"),
    "base": StageSpec("base", input_channels=5, output_channels=5, tile_size=64, conditioning_name="coarse"),
    "decoder": StageSpec("decoder", input_channels=6, output_channels=1, tile_size=512, conditioning_name=None),
}


def make_stage_model(stage: str, width: int, depth: int) -> DeployableConvStack:
    spec = STAGE_SPECS[stage]
    effective_input_channels = {
        "coarse": 11,
        "base": 11,
        "decoder": 6,
    }[stage]
    return DeployableConvStack(
        in_channels=effective_input_channels,
        out_channels=spec.output_channels,
        width=width,
        depth=depth,
    )

