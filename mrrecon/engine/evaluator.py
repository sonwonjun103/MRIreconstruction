"""Evaluate a trained model over a dataset split: SSIM / PSNR / NMSE / NMAE.

Handles all three methods. ``supervised`` builds a U-Net; ``ssdu`` and
``zeroshot`` build the unrolled network (identical architecture). Metrics are
computed on center-cropped magnitude images; per-slice and summary results are
written to JSON, with optional comparison PNGs.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from ..data.masks import undersampling_mask
from ..data.loaders import list_slice_files, read_slice
from ..models import build_supervised, build_unrolled
from ..models.diffusion import DiffusionUNet, GaussianDiffusion
from ..metrics import all_metrics
from ..metrics import match_scale as _match_scale
from .common import get_device, acc_dir, save_json, center_crop, load_checkpoint
from .inference import (recon_supervised, recon_unrolled, recon_sense,
                        recon_diffusion, recon_varnet)


class Evaluator:
    def __init__(self, cfg, method, ckpt, split="test", save_figs=False):
        self.cfg = cfg
        self.method = method
        self.ckpt = ckpt
        self.split = split
        self.save_figs = save_figs
        self.device = get_device(cfg.device)

    def _build_model(self):
        # classical SENSE needs no trained model
        if self.method == "sense":
            cfg = self.cfg
            self.recon_fn = (lambda _m, k, s, o, d:
                             recon_sense(k, s, o, d, cfg.sense_lam, cfg.sense_cg_iter))
            return None

        if self.method in ("dccnn", "varnet") or (self.method == "supervised"
                                                  and getattr(self.cfg, "use_dc", False)):
            from ..models.varnet import build_recon
            self.cfg.varnet_official = (self.method == "varnet")   # 'varnet' = official SME
            if self.method == "supervised":
                self.cfg.cnn = self.cfg.arch            # --arch is the DCCNN backbone
            model = build_recon(self.cfg)
            self.recon_fn = recon_varnet
        elif self.method == "supervised":
            model = build_supervised(self.cfg)
            mode = getattr(self.cfg, "sup_target", "rss")
            self.recon_fn = (lambda m, k, s, o, d:
                             recon_supervised(m, k, s, o, d, target_mode=mode))
        elif self.method == "diffusion":
            cfg = self.cfg
            model = DiffusionUNet(in_ch=2, base=cfg.diff_dim)
            if not self.ckpt:
                raise ValueError("--ckpt (trained diffusion prior) required for 'diffusion'")
            model = model.to(self.device)
            load_checkpoint(model, self.ckpt, self.device)
            model.eval()
            diffusion = GaussianDiffusion(model, timesteps=cfg.diff_timesteps,
                                          schedule=cfg.diff_schedule, device=self.device)
            self.recon_fn = (lambda _m, k, s, o, d:
                             recon_diffusion(diffusion, k, s, o, d,
                                             steps=cfg.diff_sampling_steps,
                                             dc_lam=cfg.diff_dc_lam,
                                             dc_iter=cfg.diff_dc_iter))
            return model
        else:  # ssdu / zeroshot -> unrolled net selected by cfg.model
            model = build_unrolled(self.cfg)
            self.recon_fn = recon_unrolled

        if not self.ckpt:
            raise ValueError(f"--ckpt is required for method '{self.method}'")
        model = model.to(self.device)
        load_checkpoint(model, self.ckpt, self.device)
        model.eval()
        return model

    def _save_fig(self, rdir, idx, ref, zf, recon, m, mask=None, mzf=None,
                  rss=None, m_rss=None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        vmax = 0.6 * ref.max()
        zf_title = "zero-filled"
        if mzf is not None:
            zf_title += f"  ssim={mzf['ssim']:.3f} psnr={mzf['psnr']:.2f}"
        rec_title = f"recon (vs SENSE)  ssim={m['ssim']:.3f} psnr={m['psnr']:.2f}"
        if m_rss is not None:
            rec_title += f"\n(vs RSS)  ssim={m_rss['ssim']:.3f} psnr={m_rss['psnr']:.2f}"
        panels = [(ref, "fully-sampled reference", vmax)]
        if rss is not None:
            panels.append((rss, "RSS ground truth", 0.6 * rss.max()))
        panels += [(zf, zf_title, vmax), (recon, rec_title, vmax)]
        if mask is not None:
            panels.append((mask, "mask (Omega)", 1.0))
        fig, ax = plt.subplots(1, len(panels), figsize=(5 * len(panels), 6))
        for a, (im, t, vm) in zip(ax, panels):
            a.imshow(im, cmap="gray", vmax=vm)
            a.set_title(t, fontsize=10)
            a.axis("off")
        fdir = os.path.join(rdir, "figs")
        os.makedirs(fdir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(fdir, f"slice_{idx:03d}.png"), dpi=120)
        plt.close(fig)

    def _save_metric_curves(self, rdir, per_slice, has_rss):
        """Per-slice SSIM & PSNR plotted for the SENSE reference and (if
        available) the RSS ground truth, so both reference choices are visible."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        idx = [p["slice"] for p in per_slice]
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        for col, key in enumerate(["ssim", "psnr"]):
            ax[col].plot(idx, [p["recon"][key] for p in per_slice],
                         marker=".", label="recon vs SENSE")
            if has_rss:
                ax[col].plot(idx, [p["recon_rss"][key] for p in per_slice],
                             marker=".", label="recon vs RSS")
                ax[col].plot(idx, [p["zero_filled_rss"][key] for p in per_slice],
                             ls="--", alpha=0.6, label="zero-filled vs RSS")
            ax[col].set_title(key.upper()); ax[col].set_xlabel("slice")
            ax[col].grid(alpha=0.3); ax[col].legend(fontsize=8)
        fig.suptitle(f"{self.cfg.run_name} | {self.method} acc={self.cfg.acc_rate} "
                     f"({self.split})", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(rdir, "metrics_per_slice.png"), dpi=120)
        plt.close(fig)

    def _save_mask_overview(self, rdir, masks_arr, cfg):
        """masks.png: per-slice column patterns + mean sampling density."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].imshow(masks_arr, aspect="auto", cmap="gray", interpolation="nearest")
        ax[0].set_title(f"masks per slice  (acc={cfg.acc_rate} acs={cfg.acs_lines} "
                        f"{cfg.mask_type})")
        ax[0].set_xlabel("phase-encode column"); ax[0].set_ylabel("slice")
        ax[1].plot(masks_arr.mean(0))
        ax[1].set_title(f"mean sampling prob per column "
                        f"(frac={masks_arr.mean():.3f})")
        ax[1].set_xlabel("phase-encode column"); ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(rdir, "masks.png"), dpi=120)
        plt.close(fig)

    def evaluate(self):
        cfg = self.cfg
        model = self._build_model()
        files = list_slice_files(cfg.data_root, cfg.tissue, self.split, cfg.max_slices, cfg.modality, cfg.full_subject)
        # results go under runs/<run_name>/acc{acc_rate}/ so different
        # accelerations of the same run are kept separately
        rdir = acc_dir(cfg)
        print(f"eval results -> {os.path.abspath(rdir)}")

        per_slice = []
        agg = {"ssim": [], "psnr": [], "nmse": [], "nmae": []}
        zf_agg = {"ssim": [], "psnr": [], "nmse": [], "nmae": []}
        agg_rss = {"ssim": [], "psnr": [], "nmse": [], "nmae": []}
        zf_rss_agg = {"ssim": [], "psnr": [], "nmse": [], "nmae": []}
        has_rss = False
        masks_1d = []
        eval_t0 = time.time()
        for i, fpath in enumerate(files):
            kspace_i, sens_i, rss_i = read_slice(fpath, crop_size=cfg.crop_size)
            H, W = kspace_i.shape[1:]
            rng = np.random.default_rng(i + 1)
            omega = undersampling_mask((H, W), cfg.acc_rate, cfg.acs_lines,
                                       cfg.mask_type, rng=rng, vds_power=cfg.vds_power)
            masks_1d.append(omega[0].astype(np.uint8))     # column pattern (W,)
            ref, zf, recon = self.recon_fn(model, kspace_i, sens_i, omega, self.device)
            rc, zc, rec = center_crop(ref), center_crop(zf), center_crop(recon)
            m = all_metrics(rc, rec)                        # vs SENSE reference
            mzf = all_metrics(rc, zc)
            for k in agg:
                agg[k].append(m[k])
                zf_agg[k].append(mzf[k])
            row = {"slice": i, "recon": m, "zero_filled": mzf,
                   "sampling_fraction": float(omega.mean())}

            # vs the official fastMRI RSS ground truth (if stored in the slice).
            # Our recon lives in the normalised-k-space SENSE domain, so it is
            # least-squares scale-matched to the RSS before scoring (a single
            # global scalar -- the intensity scale is otherwise arbitrary).
            m_rss = None
            if rss_i is not None:
                has_rss = True
                rss_ref = center_crop(np.abs(rss_i), rec.shape[-1])
                m_rss = all_metrics(rss_ref, _match_scale(rss_ref, rec))
                mzf_rss = all_metrics(rss_ref, _match_scale(rss_ref, zc))
                row["recon_rss"] = m_rss
                row["zero_filled_rss"] = mzf_rss
                for k in agg_rss:
                    agg_rss[k].append(m_rss[k])
                    zf_rss_agg[k].append(mzf_rss[k])

            per_slice.append(row)
            if self.save_figs:
                rss_disp = None if rss_i is None else center_crop(np.abs(rss_i), rec.shape[-1])
                self._save_fig(rdir, i, ref, zf, recon, m, mask=omega, mzf=mzf,
                               rss=rss_disp, m_rss=m_rss)
            if i % 20 == 0:
                extra = "" if m_rss is None else (f" | vs RSS ssim={m_rss['ssim']:.4f} "
                                                  f"psnr={m_rss['psnr']:.3f}")
                print(f"  [{i}/{len(files)}] vs SENSE ssim={m['ssim']:.4f} "
                      f"psnr={m['psnr']:.3f} nmse={m['nmse']:.5f} nmae={m['nmae']:.5f}{extra}")

        # save the masks used (1-D column patterns, exact) + an overview PNG
        masks_arr = np.stack(masks_1d)                     # (N, W) uint8
        np.save(os.path.join(rdir, f"masks_{self.method}_{self.split}.npy"), masks_arr)
        self._save_mask_overview(rdir, masks_arr, cfg)
        self._save_metric_curves(rdir, per_slice, has_rss)

        eval_seconds = time.time() - eval_t0
        summary = {
            "method": self.method, "split": self.split, "tissue": cfg.tissue,
            "n_slices": len(files),
            "acc_rate": cfg.acc_rate, "acs_lines": cfg.acs_lines,
            "mask_type": cfg.mask_type,
            "sampling_fraction": float(masks_arr.mean()),
            "eval_seconds": round(eval_seconds, 2),
            "sec_per_slice": round(eval_seconds / max(len(files), 1), 3),
            "reference_note": "recon/zero_filled = vs SENSE; *_rss = vs fastMRI "
                              "reconstruction_rss (recon least-squares scale-matched)",
            "recon": {k: float(np.nanmean(v)) for k, v in agg.items()},
            "zero_filled": {k: float(np.nanmean(v)) for k, v in zf_agg.items()},
        }
        if has_rss:
            summary["recon_rss"] = {k: float(np.nanmean(v)) for k, v in agg_rss.items()}
            summary["zero_filled_rss"] = {k: float(np.nanmean(v)) for k, v in zf_rss_agg.items()}
        save_json({"summary": summary, "per_slice": per_slice},
                  os.path.join(rdir, f"eval_{self.method}_{self.split}.json"))
        # also a standalone timing file alongside training's timing.json
        save_json({"phase": "eval", "method": self.method, "split": self.split,
                   "n_slices": len(files), "eval_seconds": round(eval_seconds, 2),
                   "sec_per_slice": round(eval_seconds / max(len(files), 1), 3)},
                  os.path.join(rdir, f"timing_eval_{self.method}_{self.split}.json"))
        print(f"\neval time : {eval_seconds:.1f}s ({len(files)} slices, "
              f"{eval_seconds/max(len(files),1):.3f}s/slice)")
        def _line(tag, d):
            print(f"{tag} : SSIM={d['ssim']:.4f} PSNR={d['psnr']:.3f} "
                  f"NMSE={d['nmse']:.5f} NMAE={d['nmae']:.5f}")

        if has_rss:                                        # RSS is the primary reference
            print("=== summary (vs fastMRI RSS ground truth, scale-matched) [PRIMARY] ===")
            _line("zero-filled", summary["zero_filled_rss"])
            _line("recon      ", summary["recon_rss"])
        print("=== summary (vs SENSE reference) [secondary] ===")
        _line("zero-filled", summary["zero_filled"])
        _line("recon      ", summary["recon"])
        return summary
