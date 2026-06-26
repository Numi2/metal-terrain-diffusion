from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .data import AnunakiTerrainDataset, dataset_sample_count, limit_indices, split_indices
from .models import STAGE_SPECS, make_stage_model


class AnunakiTrainer:
    def __init__(self, config: TrainingConfig, resume: bool = False):
        self.config = config
        self.device = self._device_for(config.run.device)
        self.out_dir = config.run.out_dir
        self.resume = resume
        self.metrics_path = self.out_dir / "metrics.jsonl"
        self.checkpoint_dir = self.out_dir / "checkpoints"
        self.sample_dir = self.out_dir / "samples"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)

        self._seed_everything(config.data.seed)
        (self.out_dir / "training_config.json").write_text(json.dumps(config.to_jsonable(), indent=2))

    def run(self) -> None:
        train_loader, val_loader = self._loaders()
        for stage in ["coarse", "base", "decoder"]:
            self._train_stage(stage, train_loader, val_loader)
        self._write_run_manifest()

    def _loaders(self) -> tuple[DataLoader, DataLoader]:
        sample_count = dataset_sample_count(self.config.data.dataset)
        splits = split_indices(sample_count, self.config.data.val_fraction, self.config.data.seed)
        train_indices = limit_indices(splits.train, self.config.data.max_train_samples)
        val_indices = limit_indices(splits.val, self.config.data.max_val_samples)
        if len(train_indices) == 0:
            raise ValueError("training split is empty")
        if len(val_indices) == 0:
            val_indices = train_indices[:1]

        train_dataset = AnunakiTerrainDataset(self.config.data.dataset, train_indices)
        val_dataset = AnunakiTerrainDataset(self.config.data.dataset, val_indices)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.optim.batch_size,
            shuffle=True,
            num_workers=self.config.run.num_workers,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.optim.batch_size,
            shuffle=False,
            num_workers=self.config.run.num_workers,
            drop_last=False,
        )
        return train_loader, val_loader

    def _train_stage(self, stage: str, train_loader: DataLoader, val_loader: DataLoader) -> None:
        model = make_stage_model(stage, self.config.model.width, self.config.model.depth).to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.weight_decay,
        )
        start_epoch = self._load_latest_checkpoint(stage, model, optimizer) if self.resume else 0

        for epoch in range(start_epoch, self.config.optim.epochs):
            started = time.time()
            train_loss = self._run_epoch(stage, model, train_loader, optimizer)
            val_loss = self._validate(stage, model, val_loader)
            duration = time.time() - started
            self._record_metric(stage, epoch + 1, train_loss, val_loss, duration)

            if (epoch + 1) % self.config.run.checkpoint_every_epochs == 0 or epoch + 1 == self.config.optim.epochs:
                self._save_checkpoint(stage, epoch + 1, model, optimizer, train_loss, val_loss)
            if (epoch + 1) % self.config.run.sample_every_epochs == 0 or epoch + 1 == self.config.optim.epochs:
                self._write_sample(stage, epoch + 1, model, val_loader)

    def _run_epoch(self, stage: str, model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        batches = 0

        for step, batch in enumerate(loader, start=1):
            inputs, targets = self._stage_batch(stage, batch)
            predictions = model(inputs)
            loss = F.l1_loss(predictions, targets) / self.config.optim.grad_accum_steps
            loss.backward()

            if step % self.config.optim.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.optim.grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.detach().cpu()) * self.config.optim.grad_accum_steps
            batches += 1

        if batches % self.config.optim.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.optim.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        return total_loss / max(batches, 1)

    @torch.no_grad()
    def _validate(self, stage: str, model: nn.Module, loader: DataLoader) -> float:
        model.eval()
        total_loss = 0.0
        batches = 0
        for batch in loader:
            inputs, targets = self._stage_batch(stage, batch)
            total_loss += float(F.l1_loss(model(inputs), targets).detach().cpu())
            batches += 1
        return total_loss / max(batches, 1)

    def _stage_batch(self, stage: str, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        conditioning64 = batch["conditioning64"].to(self.device)
        coarse_target = batch["coarse_target"].to(self.device)
        base_target = batch["base_target"].to(self.device)
        decoder_target = batch["decoder_target"].to(self.device)

        if stage == "coarse":
            noise = torch.randn_like(coarse_target) * 0.35
            return torch.cat([noise, conditioning64], dim=1), coarse_target
        if stage == "base":
            noise = torch.randn_like(base_target) * 0.35
            coarse_context = F.interpolate(coarse_target, size=(4, 4), mode="bilinear", align_corners=False)
            coarse_context = F.interpolate(coarse_context, size=base_target.shape[-2:], mode="nearest")
            return torch.cat([noise, coarse_context], dim=1), base_target
        if stage == "decoder":
            noise = torch.randn_like(decoder_target) * 0.35
            latent = F.interpolate(base_target, size=decoder_target.shape[-2:], mode="bilinear", align_corners=False)
            return torch.cat([noise, latent], dim=1), decoder_target
        raise ValueError(f"unknown stage {stage}")

    def _save_checkpoint(
        self,
        stage: str,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loss: float,
        val_loss: float,
    ) -> None:
        spec = STAGE_SPECS[stage]
        payload = {
            "stage": stage,
            "epoch": epoch,
            "model": model.cpu().state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "stage_spec": asdict(spec),
            "model_config": asdict(self.config.model),
        }
        torch.save(payload, self.checkpoint_dir / f"{stage}_epoch_{epoch:04d}.pt")
        torch.save(payload, self.checkpoint_dir / f"{stage}_model.pt")
        model.to(self.device)

    def _load_latest_checkpoint(self, stage: str, model: nn.Module, optimizer: torch.optim.Optimizer) -> int:
        checkpoint_path = self.checkpoint_dir / f"{stage}_model.pt"
        if not checkpoint_path.exists():
            return 0
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        return int(checkpoint["epoch"])

    def _record_metric(self, stage: str, epoch: int, train_loss: float, val_loss: float, duration: float) -> None:
        with self.metrics_path.open("a") as fp:
            fp.write(
                json.dumps(
                    {
                        "stage": stage,
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "duration_seconds": duration,
                    }
                )
                + "\n"
            )
        print(f"{stage} epoch={epoch} train={train_loss:.5f} val={val_loss:.5f} seconds={duration:.1f}")

    @torch.no_grad()
    def _write_sample(self, stage: str, epoch: int, model: nn.Module, loader: DataLoader) -> None:
        model.eval()
        batch = next(iter(loader))
        inputs, targets = self._stage_batch(stage, batch)
        predictions = model(inputs)
        image = self._sample_image(predictions[0, :1], targets[0, :1])
        image.save(self.sample_dir / f"{stage}_epoch_{epoch:04d}.png")

    def _sample_image(self, prediction: torch.Tensor, target: torch.Tensor) -> Image.Image:
        pred = _to_uint8(prediction.detach().cpu()[0])
        tgt = _to_uint8(target.detach().cpu()[0])
        combined = np.concatenate([pred, tgt, np.abs(pred.astype(np.int16) - tgt.astype(np.int16)).astype(np.uint8)], axis=1)
        return Image.fromarray(combined, mode="L")

    def _write_run_manifest(self) -> None:
        manifest = {
            "status": "complete",
            "device": str(self.device),
            "checkpoint_dir": str(self.checkpoint_dir),
            "metrics": str(self.metrics_path),
            "stages": list(STAGE_SPECS),
        }
        (self.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.config.run.deterministic:
            torch.use_deterministic_algorithms(False)

    @staticmethod
    def _device_for(name: str) -> torch.device:
        if name == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(name)


def _to_uint8(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.numpy()
    lo = float(np.percentile(array, 1))
    hi = float(np.percentile(array, 99))
    if hi <= lo:
        hi = lo + 1e-4
    return np.clip((array - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
