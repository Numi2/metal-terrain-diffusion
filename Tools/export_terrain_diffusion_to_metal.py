#!/usr/bin/env python3
"""
Export terrain-diffusion PyTorch checkpoints into the MetalGraphDenoiser archive format.

This exporter writes one directory per model:
  coarse_model.metalgraph/manifest.json + weights/*.bin
  base_model.metalgraph/manifest.json + weights/*.bin
  decoder_model.metalgraph/manifest.json + weights/*.bin

The runtime intentionally consumes a lowered op graph, not Python modules. For maximum fidelity,
export from an eval-mode checkpoint after calling norm_weights() on every MP layer.
"""
from __future__ import annotations
import argparse, json, math, os, pathlib, re, struct
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import torch


def mp_normalize(w: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    w32 = w.detach().float()
    norm = torch.linalg.vector_norm(w32)
    norm = norm + eps
    return w32 / norm


def write_tensor(root: pathlib.Path, name: str, tensor: torch.Tensor, normalize: bool = False, gain: float = 1.0) -> dict[str, Any]:
    root.joinpath("weights").mkdir(parents=True, exist_ok=True)
    t = mp_normalize(tensor) if normalize else tensor.detach().float()
    if t.ndim == 4:
        scale = gain / math.sqrt(float(t[0].numel()))
        t = t * scale
    elif t.ndim == 2:
        scale = gain / math.sqrt(float(t.shape[1]))
        t = t * scale
    t = t.contiguous().cpu().numpy().astype(np.float32)
    fn = f"weights/{name}.bin".replace("/", "__")
    t.tofile(root / fn)
    shape = list(t.shape)
    if len(shape) == 2:
        # Swift archive stores 2D as shape [out,in]; runtime can still identify it.
        pass
    return {"name": name, "shape": shape, "file": fn}


def export_flat_conv_stack(model: torch.nn.Module, root: pathlib.Path, name: str, input_channels: int, output_channels: int, tile_size: int, legal_batch_sizes=(1,2,4,8,16)) -> None:
    """
    Conservative lowering path: exports Conv-like MPConv modules in module traversal order.
    This is useful for deployment checkpoints that have already been fused/traced into a conv stack.
    For unfused EDMUnet2D, use a tracing/lowering pass and emit the same manifest schema.
    """
    root.mkdir(parents=True, exist_ok=True)
    weights, ops = [], []
    prev = "input"
    i = 0
    for module_name, module in model.named_modules():
        if hasattr(module, "weight") and getattr(module, "weight").ndim == 4:
            wname = f"{module_name}.weight" if module_name else f"conv{i}.weight"
            weights.append(write_tensor(root, wname, module.weight, normalize=True))
            cout, cin, kh, kw = list(module.weight.shape)
            out = f"x{i}"
            ops.append({"kind":"conv2d", "name":f"conv{i}", "inputs":[prev, wname], "outputs":[out], "attrs":{"cout":str(cout), "kh":str(kh), "kw":str(kw), "pad_y":str(kh//2), "pad_x":str(kw//2)}})
            if i != len(list(model.named_modules())) - 1:
                act = f"x{i}_silu"
                ops.append({"kind":"mp_silu", "name":f"silu{i}", "inputs":[out], "outputs":[act], "attrs":{}})
                prev = act
            else:
                prev = out
            i += 1
    if not ops:
        ops.append({"kind":"identity", "name":"identity", "inputs":["input"], "outputs":["output"], "attrs":{}})
    else:
        ops.append({"kind":"identity", "name":"output", "inputs":[prev], "outputs":["output"], "attrs":{}})
    manifest = {"name": name, "inputChannels": input_channels, "outputChannels": output_channels, "tileHeight": tile_size, "tileWidth": tile_size, "legalBatchSizes": list(legal_batch_sizes), "weights": weights, "ops": ops}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))


def load_model(path: str):
    from terrain_diffusion.models.edm_unet import EDMUnet2D
    m = EDMUnet2D.from_pretrained(path)
    m.eval()
    if hasattr(m, "norm_weights"):
        m.norm_weights()
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse", required=True, help="coarse_model checkpoint directory")
    ap.add_argument("--base", required=True, help="base_model checkpoint directory")
    ap.add_argument("--decoder", required=True, help="decoder_model checkpoint directory")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    coarse = load_model(args.coarse)
    base = load_model(args.base)
    decoder = load_model(args.decoder)
    export_flat_conv_stack(coarse, out / "coarse_model.metalgraph", "coarse", 11, 6, 64, (1,))
    export_flat_conv_stack(base, out / "base_model.metalgraph", "base", 5, 5, 64, (1,2,4,8,16))
    export_flat_conv_stack(decoder, out / "decoder_model.metalgraph", "decoder", 6, 1, 512, (1,))
    print(f"Wrote Metal archives to {out}")


if __name__ == "__main__":
    main()
