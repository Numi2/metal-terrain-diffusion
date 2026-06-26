from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .models import STAGE_SPECS


def export_all(checkpoint_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stage in ["coarse", "base", "decoder"]:
        export_stage(checkpoint_dir / f"{stage}_model.pt", out_dir / f"{stage}_model.metalgraph", stage)


def export_stage(checkpoint_path: Path, root: Path, stage: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["model"]
    spec = STAGE_SPECS[stage]
    root.mkdir(parents=True, exist_ok=True)

    weights: list[dict[str, Any]] = []
    ops: list[dict[str, Any]] = []
    previous = "input"
    if spec.conditioning_name:
        previous = "model_input"
        ops.append(
            {
                "kind": "concat_channels",
                "name": "concat_conditioning",
                "inputs": ["input", f"cond.{spec.conditioning_name}"],
                "outputs": [previous],
                "attrs": {},
            }
        )

    conv_weights = [(name, value) for name, value in state.items() if name.endswith(".weight") and value.ndim == 4]
    for index, (weight_name, weight) in enumerate(conv_weights):
        weights.append(write_tensor(root, weight_name, weight))
        cout, _cin, kernel_h, kernel_w = list(weight.shape)
        output = f"x{index}"
        ops.append(
            {
                "kind": "conv2d",
                "name": f"conv{index}",
                "inputs": [previous, weight_name],
                "outputs": [output],
                "attrs": {
                    "cout": str(cout),
                    "kh": str(kernel_h),
                    "kw": str(kernel_w),
                    "pad_y": str(kernel_h // 2),
                    "pad_x": str(kernel_w // 2),
                },
            }
        )
        if index == len(conv_weights) - 1:
            previous = output
        else:
            activated = f"{output}_silu"
            ops.append({"kind": "mp_silu", "name": f"silu{index}", "inputs": [output], "outputs": [activated], "attrs": {}})
            previous = activated

    ops.append({"kind": "identity", "name": "output", "inputs": [previous], "outputs": ["output"], "attrs": {}})
    manifest = {
        "name": spec.name,
        "inputChannels": spec.input_channels,
        "outputChannels": spec.output_channels,
        "tileHeight": spec.tile_size,
        "tileWidth": spec.tile_size,
        "legalBatchSizes": [1],
        "weights": weights,
        "ops": ops,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))


def write_tensor(root: Path, name: str, tensor: torch.Tensor) -> dict[str, Any]:
    root.joinpath("weights").mkdir(parents=True, exist_ok=True)
    value = tensor.detach().float()
    array = value.contiguous().cpu().numpy().astype(np.float32)
    file_name = f"weights/{name}.bin".replace("/", "__")
    array.tofile(root / file_name)
    return {"name": name, "shape": list(array.shape), "file": file_name}
