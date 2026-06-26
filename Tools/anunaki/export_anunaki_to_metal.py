#!/usr/bin/env python3
"""Export Anunaki production checkpoints to MetalGraphDenoiser archives."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anunaki_training.exporter import export_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True, help="Directory containing *_model.pt checkpoints")
    parser.add_argument("--out", required=True, help="Output directory for *.metalgraph archives")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--min-variance", type=float, default=0.0)
    args = parser.parse_args()

    export_all(Path(args.checkpoints), Path(args.out))
    print(f"Wrote Anunaki Metal archives to {args.out}")

    if args.validate:
        subprocess.run(
            [
                "swift",
                "run",
                "terrain-diffusion-metal",
                "--models",
                args.out,
                "--width",
                "512",
                "--height",
                "512",
                "--validate-archives",
                "--require-finite",
                "--min-variance",
                str(args.min_variance),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
