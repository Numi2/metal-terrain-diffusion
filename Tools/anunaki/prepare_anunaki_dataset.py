#!/usr/bin/env python3
"""Prepare Anunaki terrain training crops from DEM GeoTIFFs.

The output is a compressed NPZ with:
  terrain: [N,1,H,W] normalized alien-island elevation targets
  conditioning: [N,5,H,W] low elevation, slope, ridge, basin, style mask

Requires: numpy, rasterio.
"""
from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import numpy as np


def require_rasterio():
    try:
        import rasterio  # type: ignore
    except ImportError as exc:
        raise SystemExit("Install rasterio first: python -m pip install rasterio") from exc
    return rasterio


def normalize_height(patch: np.ndarray) -> np.ndarray:
    patch = patch.astype(np.float32)
    finite = np.isfinite(patch)
    if not finite.any():
        return np.zeros_like(patch, dtype=np.float32)
    median = np.nanmedian(patch)
    patch = np.where(finite, patch, median)
    patch = patch - np.percentile(patch, 5)
    scale = max(np.percentile(patch, 98) - np.percentile(patch, 2), 1.0)
    return np.clip((patch / scale) * 2 - 1, -1.5, 1.5).astype(np.float32)


def box_low_frequency(patch: np.ndarray, factor: int = 16) -> np.ndarray:
    h, w = patch.shape
    small = patch[: h - h % factor, : w - w % factor].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
    return np.repeat(np.repeat(small, factor, axis=0), factor, axis=1)[:h, :w].astype(np.float32)


def alien_augment(patch: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    h, w = patch.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    cx = w * rng.uniform(0.42, 0.58)
    cy = h * rng.uniform(0.42, 0.58)
    radius = min(h, w) * rng.uniform(0.38, 0.54)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius
    island = np.clip(1 - dist ** 2.15, 0, 1)
    edge_falloff = island ** rng.uniform(0.75, 1.45)

    terrain = patch * rng.uniform(1.2, 2.4) * edge_falloff
    for _ in range(rng.integers(2, 6)):
        crater_x = w * rng.uniform(0.2, 0.8)
        crater_y = h * rng.uniform(0.2, 0.8)
        crater_r = min(h, w) * rng.uniform(0.045, 0.14)
        crater_d = np.sqrt((xx - crater_x) ** 2 + (yy - crater_y) ** 2) / crater_r
        bowl = np.exp(-(crater_d ** 2) * 1.7)
        rim = np.exp(-((crater_d - 1.0) ** 2) * 16.0)
        terrain += rim * rng.uniform(0.15, 0.42) - bowl * rng.uniform(0.18, 0.55)

    ridge = np.sin(xx * rng.uniform(0.025, 0.055) + yy * rng.uniform(0.012, 0.04) + rng.uniform(0, math.tau))
    terrain += ridge.astype(np.float32) * edge_falloff * rng.uniform(0.05, 0.18)
    terrain = np.clip(terrain, -1.5, 1.8).astype(np.float32)
    return terrain, edge_falloff.astype(np.float32)


def conditioning_for(terrain: np.ndarray, style: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(terrain)
    slope = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    slope = slope / max(float(np.percentile(slope, 98)), 1e-4)
    gyy, _ = np.gradient(gy)
    _, gxx = np.gradient(gx)
    curvature = (gxx + gyy).astype(np.float32)
    curvature = np.clip(curvature / max(float(np.percentile(np.abs(curvature), 98)), 1e-4), -1, 1)
    basin = (terrain < np.percentile(terrain, 35)).astype(np.float32)
    low = box_low_frequency(terrain)
    return np.stack([low, slope, curvature, basin, style], axis=0).astype(np.float32)


def iter_crops(path: Path, crop_size: int, stride: int):
    rasterio = require_rasterio()
    with rasterio.open(path) as src:
        band = src.read(1).astype(np.float32)
    h, w = band.shape
    for y in range(0, max(1, h - crop_size + 1), stride):
        for x in range(0, max(1, w - crop_size + 1), stride):
            crop = band[y : y + crop_size, x : x + crop_size]
            if crop.shape == (crop_size, crop_size):
                yield crop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", required=True, help="DEM GeoTIFF glob, e.g. '/data/dem/**/*.tif'")
    parser.add_argument("--out", required=True)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--max-crops", type=int, default=512)
    parser.add_argument("--seed", type=int, default=5819)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    terrain, conditioning = [], []
    for dem in sorted(glob.glob(args.input_glob, recursive=True)):
        for crop in iter_crops(Path(dem), args.crop_size, args.stride):
            base = normalize_height(crop)
            alien, style = alien_augment(base, rng)
            terrain.append(alien[None, :, :])
            conditioning.append(conditioning_for(alien, style))
            if len(terrain) >= args.max_crops:
                break
        if len(terrain) >= args.max_crops:
            break

    if not terrain:
        raise SystemExit("No valid DEM crops found.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, terrain=np.stack(terrain), conditioning=np.stack(conditioning))
    print(f"Wrote {len(terrain)} Anunaki crops to {out}")


if __name__ == "__main__":
    main()
