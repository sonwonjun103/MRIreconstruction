"""Supervised U-Net training: zero-filled SENSE image -> fully-sampled image."""

from __future__ import annotations

import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.loaders import list_slice_files
from ..data.datasets import SupervisedDataset
from ..models import build_supervised
from ..losses import SupervisedLoss
from ..metrics import all_metrics
from .common import (save_curves, set_seed, get_device, acc_dir, save_checkpoint, save_json,
                     center_crop)
from ..data.transforms import r2c_np


class SupervisedTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = get_device(cfg.device)

    def _build(self):
        cfg = self.cfg
        tr = list_slice_files(cfg.data_root, cfg.tissue, "train", cfg.max_slices, cfg.modality, cfg.full_subject)
        va = list_slice_files(cfg.data_root, cfg.tissue, "val", cfg.max_slices, cfg.modality, cfg.full_subject)

        self.train_ds = SupervisedDataset(cfg, tr, train=True)
        self.val_ds = SupervisedDataset(cfg, va, train=False)
        self.train_dl = DataLoader(self.train_ds, batch_size=cfg.batch_size,
                                   shuffle=True, num_workers=cfg.num_workers)
        self.val_dl = DataLoader(self.val_ds, batch_size=cfg.batch_size,
                                 shuffle=False, num_workers=cfg.num_workers)

        self.model = build_supervised(cfg).to(self.device)
        self.tag = (f"supervised arch={cfg.arch} loss={cfg.loss} | acc={cfg.acc_rate} "
                    f"acs={cfg.acs_lines} mask={cfg.mask_type} data={'full' if cfg.full_subject else 'central'}")
        print(f"[train] {self.tag} | tissue={cfg.tissue} modality={cfg.modality or 'all'} "
              f"| train {len(tr)} / val {len(va)} slices")
        self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.loss_fn = SupervisedLoss(kind=cfg.loss, ssim_weight=cfg.ssim_weight).to(self.device)

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        mets = {"ssim": [], "psnr": [], "nmse": [], "nmae": []}
        losses = []
        for batch in self.val_dl:
            x = batch["x_in"].to(self.device)
            tgt = batch["target"].to(self.device)
            out_t = self.model(x)
            losses.append(self.loss_fn(out_t, tgt).item())     # validation loss (same loss fn)
            out = out_t.cpu().numpy()
            tgt_np = tgt.cpu().numpy()
            for b in range(out.shape[0]):
                recon = np.abs(r2c_np(out[b], axis=0))
                ref = np.abs(r2c_np(tgt_np[b], axis=0))
                m = all_metrics(center_crop(ref), center_crop(recon))
                for k in mets:
                    mets[k].append(m[k])
        result = {k: float(np.nanmean(v)) for k, v in mets.items()}
        result["val_loss"] = float(np.mean(losses))
        return result

    def train(self):
        set_seed(self.cfg.seed)
        self._build()
        rdir = acc_dir(self.cfg)
        save_json(self.cfg.to_dict(), os.path.join(rdir, "config.json"))

        history, best = [], -1.0
        n = len(self.train_dl)
        train_t0 = time.time()
        for epoch in range(self.cfg.epochs):
            self.model.train()
            t0, ep_loss = time.time(), 0.0
            for i, batch in enumerate(self.train_dl):
                x = batch["x_in"].to(self.device)
                tgt = batch["target"].to(self.device)
                out = self.model(x)
                loss = self.loss_fn(out, tgt)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                ep_loss += loss.item()
                if i % 50 == 0:
                    print(f"  [ep {epoch+1}] {i}/{n} loss={loss.item():.5f}")

            val = self._validate()
            ep_loss /= max(n, 1)
            print(f"epoch {epoch+1}/{self.cfg.epochs} "
                  f"loss={ep_loss:.5f} val_loss={val['val_loss']:.5f} "
                  f"ssim={val['ssim']:.4f} psnr={val['psnr']:.3f} "
                  f"nmse={val['nmse']:.5f} nmae={val['nmae']:.5f} ({time.time()-t0:.1f}s)")
            history.append({"epoch": epoch + 1, "loss": ep_loss, **val})

            last_path = os.path.join(rdir, "last.pt")
            save_checkpoint(self.model, self.cfg, last_path, extra={"epoch": epoch + 1})
            if val["ssim"] > best:
                best = val["ssim"]
                best_path = os.path.join(rdir, "best.pt")
                save_checkpoint(self.model, self.cfg, best_path,
                                extra={"epoch": epoch + 1, "val": val})
                print(f"  -> best.pt updated (epoch {epoch+1}, ssim {best:.4f}): "
                      f"{os.path.abspath(best_path)}")
            save_json(history, os.path.join(rdir, "history.json"))
            save_curves(history, os.path.join(rdir, "curves.png"),
                        title=f"{self.cfg.run_name} | {self.tag}")

        train_seconds = time.time() - train_t0
        save_json({"phase": "train", "method": "supervised", "tag": self.tag,
                   "epochs": len(history), "train_seconds": round(train_seconds, 2),
                   "sec_per_epoch": round(train_seconds / max(len(history), 1), 2)},
                  os.path.join(rdir, "timing.json"))
        print(f"done. best val SSIM={best:.4f}  [{self.tag}]")
        print(f"  train time : {train_seconds:.1f}s ({len(history)} epochs)")
        print(f"  best.pt : {os.path.abspath(os.path.join(rdir, 'best.pt'))}")
        print(f"  last.pt : {os.path.abspath(os.path.join(rdir, 'last.pt'))}")
        return history
