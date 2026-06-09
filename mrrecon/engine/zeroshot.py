"""Zero-shot self-supervised reconstruction (ZS-SSL) on a single scan.

A single slice's acquisition mask Omega is partitioned into a held-out
validation set Gamma (for early stopping) and a training portion Omega\\Gamma.
Each training step re-splits the training portion into (Theta, Lambda) and
minimises the SSDU k-space loss. The validation k-space loss on Gamma drives
early stopping; final inference uses the full Omega for data consistency.

No fully-sampled data is used at any point -- image-domain SSIM/PSNR/NMSE are
reported only for monitoring against the (separately available) SENSE image.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.loaders import list_slice_files, read_slice
from ..data.datasets import ZeroShotDataset
from ..data.masks import undersampling_mask
from ..models import build_unrolled
from ..losses import MixL1L2Loss
from ..metrics import all_metrics
from .common import (save_curves, save_mask_preview, set_seed, get_device, acc_dir,
                     save_checkpoint, save_json, center_crop, load_checkpoint)
from .inference import recon_unrolled


class ZeroShotTrainer:
    def __init__(self, cfg, split="test"):
        self.cfg = cfg
        self.split = split
        self.device = get_device(cfg.device)

    def _build(self):
        cfg = self.cfg
        files = list_slice_files(cfg.data_root, cfg.tissue, self.split, modality=cfg.modality, full=cfg.full_subject)
        idx = (len(files) // 2) if cfg.zs_slice < 0 else cfg.zs_slice
        idx = int(np.clip(idx, 0, len(files) - 1))
        self.slice_idx = idx
        self.slice_file = files[idx]
        self.kspace_slice, self.sens_slice, _ = read_slice(self.slice_file, crop_size=cfg.crop_size)

        H, W = self.kspace_slice.shape[1:]
        rng = np.random.default_rng(cfg.seed)
        self.omega = undersampling_mask((H, W), cfg.acc_rate, cfg.acs_lines,
                                        cfg.mask_type, rng=rng, vds_power=cfg.vds_power)

        self.ds = ZeroShotDataset(cfg, self.kspace_slice, self.sens_slice, self.omega)
        self.dl = DataLoader(self.ds, batch_size=1, shuffle=False, num_workers=0)

        self.model = build_unrolled(cfg).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.loss_fn = MixL1L2Loss().to(self.device)

    def _batch_to_device(self, b):
        return (b["x_in"].to(self.device), b["sens_maps"].to(self.device),
                b["ref_kspace"].to(self.device), b["trn_mask"].to(self.device),
                b["loss_mask"].to(self.device))

    @torch.no_grad()
    def _val_loss(self):
        self.model.eval()
        b = {k: v.unsqueeze(0) if torch.is_tensor(v) else v
             for k, v in self.ds.val_sample().items()}
        x, sens, ref_k, trn, loss_m = self._batch_to_device(b)
        _, _, nw_k = self.model(x, sens, trn, loss_m)
        return self.loss_fn(nw_k, ref_k).item()

    @torch.no_grad()
    def _image_metrics(self):
        ref, _, recon = recon_unrolled(self.model, self.kspace_slice,
                                       self.sens_slice, self.omega, self.device)
        return all_metrics(center_crop(ref), center_crop(recon)), recon, ref

    def train(self):
        set_seed(self.cfg.seed)
        self._build()
        rdir = acc_dir(self.cfg)
        save_json(self.cfg.to_dict(), os.path.join(rdir, "config.json"))
        self.tag = f"zeroshot model={self.cfg.model} | acc={self.cfg.acc_rate} mask={self.cfg.mask_type}"
        print(f"[train] {self.tag} | fitting {self.cfg.tissue} "
              f"{self.cfg.modality or ''} {self.split} slice {self.slice_idx}")
        save_mask_preview(rdir, self.cfg, self.kspace_slice.shape[1:])

        history = []
        best_val, best_epoch, since_improve = float("inf"), 0, 0
        best_path = os.path.join(rdir, "best.pt")
        last_path = os.path.join(rdir, "last.pt")

        train_t0 = time.time()
        for epoch in range(self.cfg.epochs):
            self.model.train()
            t0, ep_loss, steps = time.time(), 0.0, 0
            for b in self.dl:
                x, sens, ref_k, trn, loss_m = self._batch_to_device(b)
                _, _, nw_k = self.model(x, sens, trn, loss_m)
                loss = self.loss_fn(nw_k, ref_k)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                ep_loss += loss.item()
                steps += 1

            ep_loss /= max(steps, 1)
            val_loss = self._val_loss()
            img, _, _ = self._image_metrics()
            print(f"epoch {epoch+1}/{self.cfg.epochs} "
                  f"train={ep_loss:.5f} val={val_loss:.5f} "
                  f"ssim={img['ssim']:.4f} psnr={img['psnr']:.3f} "
                  f"nmse={img['nmse']:.5f} nmae={img['nmae']:.5f} ({time.time()-t0:.1f}s)")
            history.append({"epoch": epoch + 1, "train_loss": ep_loss,
                            "val_loss": val_loss, **img})

            if val_loss < best_val:
                best_val, best_epoch, since_improve = val_loss, epoch + 1, 0
                save_checkpoint(self.model, self.cfg, best_path,
                                extra={"epoch": epoch + 1, "val_loss": val_loss})
                print(f"  -> best.pt updated (epoch {epoch+1}, val {val_loss:.5f}): "
                      f"{os.path.abspath(best_path)}")
            else:
                since_improve += 1
            save_checkpoint(self.model, self.cfg, last_path, extra={"epoch": epoch + 1})
            save_json(history, os.path.join(rdir, "history.json"))
            save_curves(history, os.path.join(rdir, "curves.png"),
                        title=f"{self.cfg.run_name} | {self.tag}")

            if since_improve >= self.cfg.zs_patience:
                print(f"early stopping at epoch {epoch+1} "
                      f"(best val={best_val:.5f} @ epoch {best_epoch})")
                break

        train_seconds = time.time() - train_t0

        # final inference with the best checkpoint (timed separately)
        ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        inf_t0 = time.time()
        img, recon, ref = self._image_metrics()
        inference_seconds = time.time() - inf_t0
        print(f"final (best @ epoch {best_epoch}): "
              f"ssim={img['ssim']:.4f} psnr={img['psnr']:.3f} "
              f"nmse={img['nmse']:.5f} nmae={img['nmae']:.5f}")
        timing = {"phase": "train+inference", "method": f"zeroshot/{self.cfg.model}",
                  "epochs": len(history),
                  "train_seconds": round(train_seconds, 2),
                  "inference_seconds": round(inference_seconds, 3),
                  "total_seconds": round(train_seconds + inference_seconds, 2)}
        save_json(timing, os.path.join(rdir, "timing.json"))
        self._save_outputs(rdir, recon, ref, img,
                           extra={"best_epoch": best_epoch, "timing": timing})
        print(f"  train time     : {train_seconds:.1f}s ({len(history)} epochs)")
        print(f"  inference time : {inference_seconds:.3f}s")
        print(f"  best.pt : {os.path.abspath(best_path)}")
        print(f"  last.pt : {os.path.abspath(last_path)}")
        print(f"  recon   : {os.path.abspath(os.path.join(rdir, 'recon.npy'))}")
        return img

    def _save_outputs(self, rdir, recon, ref, img, extra=None):
        np.save(os.path.join(rdir, "recon.npy"), recon)
        np.save(os.path.join(rdir, "reference.npy"), ref)
        payload = {"slice": self.slice_idx, "metrics": img}
        if extra:
            payload.update(extra)
        save_json(payload, os.path.join(rdir, "result.json"))

    @torch.no_grad()
    def infer(self, ckpt_path):
        """Reconstruct the fitted scan from a saved checkpoint -- no training.

        Rebuilds the exact same single slice and acquisition mask Omega (both are
        determined by the config + ``cfg.seed``), loads the checkpoint, and runs
        the final full-Omega reconstruction. Use this to re-run inference (e.g.
        regenerate ``recon.npy`` / figures) without re-fitting.
        """
        set_seed(self.cfg.seed)
        self._build()
        rdir = acc_dir(self.cfg)
        load_checkpoint(self.model, ckpt_path, self.device)
        inf_t0 = time.time()
        img, recon, ref = self._image_metrics()
        inference_seconds = time.time() - inf_t0
        print(f"zero-shot inference (slice {self.slice_idx}): "
              f"ssim={img['ssim']:.4f} psnr={img['psnr']:.3f} "
              f"nmse={img['nmse']:.5f} nmae={img['nmae']:.5f} "
              f"({inference_seconds:.3f}s)")
        self._save_outputs(rdir, recon, ref, img,
                           extra={"ckpt": ckpt_path,
                                  "inference_seconds": round(inference_seconds, 3)})
        return img
