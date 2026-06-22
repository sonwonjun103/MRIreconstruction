"""Train the unconditional diffusion prior p(x) on fully-sampled SENSE images.

This is the *prior-learning* stage of zero-shot diffusion reconstruction. The
trained checkpoint is later used by ``eval --method diffusion`` (or
``recon_diffusion``) to reconstruct undersampled scans via data-consistency
-guided posterior sampling -- no task-specific training required.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.loaders import list_slice_files
from ..data.datasets import DiffusionDataset
from ..models.diffusion import DiffusionUNet, GaussianDiffusion
from ..core.common import (set_seed, get_device, acc_dir, save_checkpoint,
                     save_json, save_curves)


class DiffusionTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = get_device(cfg.device)

    def _build(self):
        cfg = self.cfg
        tr = list_slice_files(cfg.data_root, cfg.tissue, "train", cfg.max_slices, cfg.modality, cfg.full_subject)
        va = list_slice_files(cfg.data_root, cfg.tissue, "val", cfg.max_slices, cfg.modality, cfg.full_subject)
        self.train_ds = DiffusionDataset(cfg, tr, train=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=cfg.batch_size,
                                   shuffle=True, num_workers=cfg.num_workers)
        self.val_dl = DataLoader(DiffusionDataset(cfg, va, train=False),
                                 batch_size=cfg.batch_size, shuffle=False,
                                 num_workers=cfg.num_workers)

        self.model = DiffusionUNet(in_ch=2, base=cfg.diff_dim).to(self.device)
        self.diffusion = GaussianDiffusion(self.model, timesteps=cfg.diff_timesteps,
                                           schedule=cfg.diff_schedule, device=self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

    @torch.no_grad()
    def _val_loss(self):
        """Mean noise-MSE on the val set. Uses a fixed seed (and restores the
        training RNG) so the value is comparable across epochs."""
        self.model.eval()
        cpu_state = torch.get_rng_state()
        cuda_state = (torch.cuda.get_rng_state_all()
                      if self.device == "cuda" and torch.cuda.is_available() else None)
        torch.manual_seed(self.cfg.seed)
        losses = []
        for batch in self.val_dl:
            losses.append(self.diffusion.p_losses(batch["x0"].to(self.device)).item())
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        return float(sum(losses) / max(len(losses), 1))

    def train(self):
        set_seed(self.cfg.seed)
        self._build()
        rdir = acc_dir(self.cfg, use_acc=False)   # prior is acc-independent
        save_json(self.cfg.to_dict(), os.path.join(rdir, "config.json"))
        self.tag = (f"diffusion prior dim={self.cfg.diff_dim} T={self.cfg.diff_timesteps} "
                    f"{self.cfg.diff_schedule}  (acc_rate only used at recon)")
        print(f"[train] {self.tag} | tissue={self.cfg.tissue} "
              f"modality={self.cfg.modality or 'all'} | {len(self.train_ds)} slices")

        history, n = [], len(self.train_dl)
        train_t0 = time.time()
        for epoch in range(self.cfg.epochs):
            self.model.train()
            t0, ep_loss = time.time(), 0.0
            for i, batch in enumerate(self.train_dl):
                x0 = batch["x0"].to(self.device)
                loss = self.diffusion.p_losses(x0)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                ep_loss += loss.item()
                if i % 50 == 0:
                    print(f"  [ep {epoch+1}] {i}/{n} loss={loss.item():.5f}")

            ep_loss /= max(n, 1)
            val_loss = self._val_loss()
            self.model.train()
            print(f"epoch {epoch+1}/{self.cfg.epochs} loss={ep_loss:.5f} "
                  f"val_loss={val_loss:.5f} ({time.time()-t0:.1f}s)")
            history.append({"epoch": epoch + 1, "loss": ep_loss, "val_loss": val_loss})
            save_checkpoint(self.model, self.cfg, os.path.join(rdir, "last.pt"),
                            extra={"epoch": epoch + 1})
            save_json(history, os.path.join(rdir, "history.json"))
            save_curves(history, os.path.join(rdir, "curves.png"),
                        title=f"{self.cfg.run_name} | {self.tag}")

        train_seconds = time.time() - train_t0
        save_json({"phase": "train", "method": "diffusion", "tag": self.tag,
                   "epochs": len(history), "train_seconds": round(train_seconds, 2),
                   "sec_per_epoch": round(train_seconds / max(len(history), 1), 2)},
                  os.path.join(rdir, "timing.json"))
        print(f"done. diffusion prior saved  (train {train_seconds:.1f}s)")
        print(f"  last.pt : {os.path.abspath(os.path.join(rdir, 'last.pt'))}")
        return history
