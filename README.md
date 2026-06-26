# MetalTerrainDiffusionPro

Apple Metal implementation of the InfiniteDiffusion terrain pipeline from arXiv:2512.08309v4.

This is not the prior runtime-only scaffold. It includes a Metal-first implementation of the lazy InfiniteDiffusion tensor runtime, deterministic coordinate-addressed Gaussian fields, packed `C+1` accumulation, LRU and persistent tile stores, a GPU graph denoiser archive runtime, terrain hierarchy wiring, signed square-root transforms, and CLI/export tooling.

The trained model weights are external. Export them with `Tools/export_terrain_diffusion_to_metal.py`, then load the resulting `*.metalgraph` archives from Swift. The package does not bundle copyrighted or third-party checkpoint files.

## Paper/runtime mapping

The implementation follows the paper and reference pipeline structure:

- Infinite tensors are queried lazily by overlapping sliding windows.
- Each produced window is packed as value channels plus one blend-weight channel.
- Query output is the accumulated numerator divided by accumulated weight.
- Tile coordinates support negative world coordinates.
- Noise is deterministic by world seed and integer coordinate, not by call order.
- The coarse stage uses 64×64 windows at stride 48.
- The latent stage uses 64×64 windows at stride 32 and supports one- or two-step consistency updates.
- The decoder stage uses configurable high-resolution tiles, default 512×512 at stride 384.
- Elevation output applies residual scaling and inverse signed square-root.

## Build

```bash
cd MetalTerrainDiffusionPro
swift build -c release
```

Requires macOS 14+ or iOS 17+ with Apple Metal.

## Export models

```bash
python Tools/export_terrain_diffusion_to_metal.py \
  --coarse /path/to/coarse_model \
  --base /path/to/base_model \
  --decoder /path/to/decoder_model \
  --out /path/to/metal_archives
```

The exporter writes:

```text
/path/to/metal_archives/
  coarse_model.metalgraph/
  base_model.metalgraph/
  decoder_model.metalgraph/
```

The provided exporter emits the archive schema consumed by `MetalGraphDenoiser`. For production checkpoints with attention or unfused residual blocks, lower the PyTorch model into the same op schema before export. The Swift runtime intentionally rejects unsupported graph ops instead of silently changing the network.

## Run

```bash
swift run terrain-diffusion-metal \
  --models /path/to/metal_archives \
  --cache /tmp/terrain-metal-cache \
  --x 0 --y 0 --width 1024 --height 1024 --seed 1234
```

For a GPU/runtime smoke test without model archives:

```bash
swift run terrain-diffusion-metal --debug-identity --width 512 --height 512
```

## Main files

`InfiniteDiffusionEngine.swift` implements dependency-planned lazy window generation, recursive parent queries, C+1 packed accumulation, batching, and cache insertion.

`TerrainDiffusionPipeline.swift` wires the paper's coarse, latent, and decoder stages.

`TerrainDiffusionKernels.metal` contains kernels for Gaussian generation, accumulation, normalization, channel concatenation, convolution, activations, signed square-root, downsample, and upsample.

`GraphRuntime.swift` loads `.metalgraph` archives and executes lowered denoiser graphs on Metal buffers.

`TileStore.swift` provides an in-memory LRU store and a persistent disk-backed tile cache.
