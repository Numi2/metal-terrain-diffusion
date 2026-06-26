#!/usr/bin/env python3
"""Create a tiny deterministic Anunaki NPZ dataset for CI/smoke training."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from prepare_anunaki_dataset import alien_augment, conditioning_for, normalize_height


def synthetic_dem(size: int, sample_index: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    phase = sample_index * 0.37
    base = (
        np.sin(xx * 0.021 + phase)
        + np.cos(yy * 0.018 - phase * 0.6)
        + np.sin((xx + yy) * 0.009 + phase * 1.7)
    )
    ridge = np.sin(xx * 0.055 + yy * 0.017 + rng.uniform(0, math.tau)) * 0.35
    return (base + ridge).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=5819)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    terrain, conditioning = [], []
    for index in range(args.samples):
        base = normalize_height(synthetic_dem(args.size, index, rng))
        alien, style = alien_augment(base, rng)
        terrain.append(alien[None, :, :])
        conditioning.append(conditioning_for(alien, style))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, terrain=np.stack(terrain), conditioning=np.stack(conditioning))
    print(f"Wrote smoke dataset with {args.samples} samples to {out}")


if __name__ == "__main__":
    main()
