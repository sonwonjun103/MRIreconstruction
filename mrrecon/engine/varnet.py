"""Supervised E2E-VarNet training: multi-coil k-space -> RSS image.

VarNet outputs in the RSS domain, the same domain as the fastMRI ground truth,
so the RSS metrics have no SENSE-vs-RSS ceiling (a perfect recon reaches SSIM
1.0) and are directly leaderboard-comparable. The refinement CNN is the toolkit
U-Net (``--cnn unet``) or the hierarchical Mamba backbone (``--cnn mamba``);
the official E2E-VarNet (learned SME) is selected by the ``varnet`` method.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.loaders import list_slice_files
from ..data.datasets import VarNetDataset
from ..models.varnet import build_recon
from ..losses import SupervisedLoss
from ..metrics import all_metrics
from .common import (save_curves, save_mask_preview, set_seed, get_device, acc_dir,
                     save_checkpoint, save_json, center_crop)


class VarNetTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = get_device(cfg.device)

    def _build(self):
        cfg = self.cfg
        tr = list_slice_files(cfg.data_root, cfg.tissue, "train", cfg.max_slices, cfg.modality, cfg.full_subject)
        va = list_slice_files(cfg.data_root, cfg.tissue, "val", cfg.max_slices, cfg.modality, cfg.full_subject)
        self.train_dl = DataLoader(VarNetDataset(cfg, tr, train=True),
                                   batch_size=cfg.batch_size, shuffle=True,
                                   num_workers=cfg.num_workers)
        self.val_dl = DataLoader(VarNetDataset(cfg, va, train=False),
                                 batch_size=1, shuffle=False, num_workers=cfg.num_workers)
        self.model = build_recon(cfg).to(self.device)
        if getattr(cfg, "varnet_official", False):
            self.tag = (f"varnet(official-SME) cascades={cfg.varnet_cascades} loss={cfg.loss}")
        else:
            self.tag = (f"dccnn cnn={cfg.cnn} cascades={cfg.dc_cascades} loss={cfg.loss}")
        self.tag += (f" | acc={cfg.acc_rate} acs={cfg.acs_lines} "
                     f"mask={cfg.mask_type} data={'full' if cfg.full_subject else 'central'}")
        print(f"[train] {self.tag} | tissue={cfg.tissue} modality={cfg.modality or 'all'} "
              f"| train {len(tr)} / val {len(va)} slices")
        self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.loss_fn = SupervisedLoss(kind=cfg.loss, ssim_weight=cfg.ssim_weight).to(self.device)

    def _recon(self, batch):
        mk = batch["masked_kspace"].to(self.device)
        sens = batch["sens"].to(self.device)
        mask = batch["mask"].to(self.device)
        rss = self.model.reconstruct(mk, sens, mask).unsqueeze(1)      # (B,1,H,W)
        return rss

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        mets = {"ssim": [], "psnr": [], "nmse": [], "nmae": []}
        losses = []
        for batch in self.val_dl:
            tgt = batch["target"].to(self.device)
            out = self._recon(batch)
            losses.append(self.loss_fn(out, tgt).item())
            o = out.cpu().numpy(); t = tgt.cpu().numpy()
            for b in range(o.shape[0]):
                m = all_metrics(center_crop(t[b, 0]), center_crop(o[b, 0]))  # RSS vs RSS
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
        from ..data.loaders import peek_shape
        _, H, W = peek_shape(self.cfg.data_root, self.cfg.tissue, "train", self.cfg.full_subject)
        if self.cfg.crop_size > 0:
            H = W = self.cfg.crop_size
        save_mask_preview(rdir, self.cfg, (H, W))

        history, best = [], -1.0
        n = len(self.train_dl)
        train_t0 = time.time()
        for epoch in range(self.cfg.epochs):
            self.model.train()
            t0, ep_loss = time.time(), 0.0
            for i, batch in enumerate(self.train_dl):
                tgt = batch["target"].to(self.device)
                out = self._recon(batch)
                loss = self.loss_fn(out, tgt)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                ep_loss += loss.item()
                if i % 25 == 0:
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
        method_name = "varnet" if getattr(self.cfg, "varnet_official", False) else "dccnn"
        save_json({"phase": "train", "method": method_name, "tag": self.tag,
                   "epochs": len(history), "train_seconds": round(train_seconds, 2),
                   "sec_per_epoch": round(train_seconds / max(len(history), 1), 2)},
                  os.path.join(rdir, "timing.json"))
        print(f"done. best val SSIM={best:.4f}  [{self.tag}]")
        print(f"  train time : {train_seconds:.1f}s ({len(history)} epochs)")
        print(f"  best.pt : {os.path.abspath(os.path.join(rdir, 'best.pt'))}")
        print(f"  last.pt : {os.path.abspath(os.path.join(rdir, 'last.pt'))}")
        return history
