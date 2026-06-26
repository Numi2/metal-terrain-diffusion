#!/usr/bin/env python3
"""Run a production Anunaki terrain training job from a YAML config."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anunaki_training.config import TrainingConfig
from anunaki_training.trainer import AnunakiTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML training config")
    parser.add_argument("--resume", action="store_true", help="Resume from latest per-stage checkpoints")
    args = parser.parse_args()

    config = TrainingConfig.load(args.config)
    trainer = AnunakiTrainer(config, resume=args.resume)
    trainer.run()


if __name__ == "__main__":
    main()
