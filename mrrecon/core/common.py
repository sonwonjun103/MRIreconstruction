"""Shared helpers for the training engines: seeding, device, checkpoints, crop."""

from __future__ import annotations

import os
import sys
import json
import random
from datetime import datetime

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(requested: str) -> str:
    return requested if (requested == "cuda" and torch.cuda.is_available()) else "cpu"


def run_dir(cfg) -> str:
    d = os.path.join(cfg.out_dir, cfg.run_name)
    os.makedirs(d, exist_ok=True)
    return d


def acc_dir(cfg, use_acc: bool = True) -> str:
    """Output dir for a run. With ``use_acc`` (default) results are separated by
    acceleration: ``runs/<run_name>/acc{acc_rate}/``. Diffusion passes
    ``use_acc=False`` (its prior training does not undersample)."""
    d = run_dir(cfg)
    if use_acc:
        d = os.path.join(d, f"acc{cfg.acc_rate}")
        os.makedirs(d, exist_ok=True)
    return d


def save_checkpoint(model, cfg, path: str, extra: dict | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"state_dict": model.state_dict(), "config": cfg.to_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model, path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    return ckpt


class _Tee:
    """Write to several streams at once (console + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def start_file_logging(cfg, fname: str = "log.txt"):
    """Mirror stdout+stderr into ``runs/<run_name>/<fname>`` (append).

    Non-invasive: engines keep using ``print``; this just tees the streams.
    Returns the log path. Safe to call once per process.
    """
    path = os.path.join(run_dir(cfg), fname)
    f = open(path, "a", buffering=1)
    f.write(f"\n{'='*70}\n[{datetime.now().isoformat(timespec='seconds')}] "
            f"cmd: {' '.join(sys.argv)}\n{'='*70}\n")
    sys.stdout = _Tee(sys.__stdout__, f)
    sys.stderr = _Tee(sys.__stderr__, f)
    return path


def save_curves(history, path: str, title: str = None):
    """Plot per-epoch curves (loss / ssim / psnr / nmse / nmae) -> PNG.

    Works with any history list of dicts; only the keys present are plotted
    (supervised/ssdu: loss,ssim,psnr,nmse,nmae; zeroshot: train_loss,val_loss,...;
    diffusion: loss). ``title`` (e.g. model + acc_rate) is shown as a suptitle.
    Safe to call every epoch (overwrites)."""
    if not history:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["loss", "train_loss", "val_loss", "ssim", "psnr", "nmse", "nmae"]
    keys = [k for k in order if any(k in h for h in history)]
    if not keys:
        return
    ncols = min(3, len(keys))
    nrows = int(np.ceil(len(keys) / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.4 * nrows), squeeze=False)
    ax = ax.ravel()
    for i, k in enumerate(keys):
        xs = [h.get("epoch", j + 1) for j, h in enumerate(history) if k in h]
        ys = [h[k] for h in history if k in h]
        ax[i].plot(xs, ys, marker="o", ms=3)
        ax[i].set_title(k)
        ax[i].set_xlabel("epoch")
        ax[i].grid(alpha=0.3)
    for j in range(len(keys), len(ax)):
        ax[j].axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_mask_preview(rdir, cfg, shape_hw, n: int = 4, fname: str = "train_masks.png"):
    """Save a few example acquisition masks (Omega) used during training -> PNG.

    Shows n random masks at the training resolution plus the mean per-column
    sampling probability, so you can see the pattern (random / gaussian1d / vds)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..data.masks import undersampling_mask

    H, W = shape_hw
    masks = [undersampling_mask((H, W), cfg.acc_rate, cfg.acs_lines, cfg.mask_type,
                                rng=np.random.default_rng(i + 1), vds_power=cfg.vds_power)
             for i in range(n)]
    fig, ax = plt.subplots(1, n + 1, figsize=(3 * (n + 1), 3.4))
    for i, a in enumerate(ax[:n]):
        a.imshow(masks[i], cmap="gray", aspect="auto")
        a.set_title(f"Omega #{i + 1}  ({masks[i].mean():.2f})", fontsize=9)
        a.axis("off")
    ax[n].plot(np.mean([m[0] for m in masks], axis=0))
    ax[n].set_title("mean col prob"); ax[n].set_xlabel("PE column"); ax[n].grid(alpha=0.3)
    extra = f" vds_power={cfg.vds_power}" if cfg.mask_type == "vds" else ""
    fig.suptitle(f"train masks — acc={cfg.acc_rate} acs={cfg.acs_lines} "
                 f"{cfg.mask_type}{extra}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(rdir, fname), dpi=110)
    plt.close(fig)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def center_crop(img: np.ndarray, size: int = 320) -> np.ndarray:
    """Center-crop a 2-D array to (size, size); no-op if already smaller."""
    h, w = img.shape[-2:]
    size_h, size_w = min(size, h), min(size, w)
    top = (h - size_h) // 2
    left = (w - size_w) // 2
    return img[..., top:top + size_h, left:left + size_w]
