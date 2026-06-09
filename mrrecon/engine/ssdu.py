"""SSDU self-supervised training across the dataset.

The network never sees fully-sampled data. Each slice's acquisition mask is
split into a data-consistency set (Theta) and a loss set (Lambda); the loss is
the normalised L1+L2 k-space error at the Lambda locations. Validation still
reports image-domain SSIM/PSNR/NMSE against the fully-sampled SENSE image (for
monitoring only -- these labels are not used for training).
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.loaders import list_slice_files, read_slice
from ..data.datasets import SSDUDataset
from ..models import build_unrolled
from ..losses import MixL1L2Loss
from ..metrics import all_metrics
from ..data.masks import undersampling_mask
from .common import (save_curves, save_mask_preview, set_seed, get_device, acc_dir,
                     save_checkpoint, save_json, center_crop)
from .inference import recon_unrolled


class SSDUTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = get_device(cfg.device)

    def _build(self):
        cfg = self.cfg
        tr = list_slice_files(cfg.data_root, cfg.tissue, "train", cfg.max_slices, cfg.modality, cfg.full_subject)
        self.val_files = list_slice_files(cfg.data_root, cfg.tissue, "val", cfg.max_slices, cfg.modality, cfg.full_subject)
        self.train_ds = SSDUDataset(cfg, tr, train=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=cfg.batch_size,
                                   shuffle=True, num_workers=cfg.num_workers)
        # deterministic SSDU val samples for the k-space validation loss
        self.val_loss_dl = DataLoader(SSDUDataset(cfg, self.val_files, train=False),
                                      batch_size=1, shuffle=False, num_workers=0)

        self.model = build_unrolled(cfg).to(self.device)
        self.tag = (f"ssdu model={cfg.model} | acc={cfg.acc_rate} acs={cfg.acs_lines} "
                    f"mask={cfg.mask_type} rho={cfg.rho} data={'full' if cfg.full_subject else 'central'}")
        print(f"[train] {self.tag} | tissue={cfg.tissue} modality={cfg.modality or 'all'} "
              f"| train {len(tr)} / val {len(self.val_files)} slices")
        self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.loss_fn = MixL1L2Loss().to(self.device)

    @torch.no_grad()
    def _validate(self):
        cfg = self.cfg
        from ..metrics import rss_metrics
        mets = {"ssim": [], "psnr": [], "nmse": [], "nmae": []}
        for i, fpath in enumerate(self.val_files):
            kspace, sens, rss = read_slice(fpath, crop_size=cfg.crop_size)
            H, W = kspace.shape[1:]
            rng = np.random.default_rng(i + 1)
            omega = undersampling_mask((H, W), cfg.acc_rate, cfg.acs_lines,
                                       cfg.mask_type, rng=rng, vds_power=cfg.vds_power)
            ref, _, recon = recon_unrolled(self.model, kspace, sens, omega, self.device)
            # model selection on the RSS ground truth (fall back to SENSE if absent)
            if rss is not None:
                m = rss_metrics(rss, recon, crop_fn=center_crop)
            else:
                m = all_metrics(center_crop(ref), center_crop(recon))
            for k in mets:
                mets[k].append(m[k])
        result = {k: float(np.nanmean(v)) for k, v in mets.items()}

        # validation k-space loss (same MixL1L2 as training, deterministic split)
        losses = []
        for b in self.val_loss_dl:
            x = b["x_in"].to(self.device); sens = b["sens_maps"].to(self.device)
            ref_k = b["ref_kspace"].to(self.device)
            trn = b["trn_mask"].to(self.device); loss_m = b["loss_mask"].to(self.device)
            _, _, nw_k = self.model(x, sens, trn, loss_m)
            losses.append(self.loss_fn(nw_k, ref_k).item())
        result["val_loss"] = float(np.mean(losses))
        return result

    def train(self):
        set_seed(self.cfg.seed)
        self._build()
        rdir = acc_dir(self.cfg)
        save_json(self.cfg.to_dict(), os.path.join(rdir, "config.json"))
        k0, _, _ = read_slice(self.train_ds.files[0], crop_size=self.cfg.crop_size)
        save_mask_preview(rdir, self.cfg, k0.shape[1:])

        history, best = [], -1.0
        n = len(self.train_dl)
        train_t0 = time.time()
        for epoch in range(self.cfg.epochs):
            self.model.train()
            t0, ep_loss = time.time(), 0.0
            for i, batch in enumerate(self.train_dl):
                x = batch["x_in"].to(self.device)
                sens = batch["sens_maps"].to(self.device)
                ref_k = batch["ref_kspace"].to(self.device)
                trn = batch["trn_mask"].to(self.device)
                loss_m = batch["loss_mask"].to(self.device)

                _, _, nw_k = self.model(x, sens, trn, loss_m)
                loss = self.loss_fn(nw_k, ref_k)
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
        save_json({"phase": "train", "method": "ssdu", "tag": self.tag,
                   "epochs": len(history), "train_seconds": round(train_seconds, 2),
                   "sec_per_epoch": round(train_seconds / max(len(history), 1), 2)},
                  os.path.join(rdir, "timing.json"))
        print(f"done. best val SSIM={best:.4f}  [{self.tag}]")
        print(f"  train time : {train_seconds:.1f}s ({len(history)} epochs)")
        print(f"  best.pt : {os.path.abspath(os.path.join(rdir, 'best.pt'))}")
        print(f"  last.pt : {os.path.abspath(os.path.join(rdir, 'last.pt'))}")
        return history
