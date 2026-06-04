"""Profiling: per-model parameter count, FLOPs, and training-time estimates.

For each method this reports:
    * #parameters (total / trainable)
    * forward FLOPs at the data's H x W (via torch's built-in FlopCounterMode)
    * measured wall-clock per training step (forward+backward) and a per-epoch
      estimate (step time x steps-per-epoch)

Notes / caveats
---------------
* FLOPs are counted by ``torch.utils.flop_counter`` for the registered ops
  (conv, matmul, linear, ...). FFTs and complex elementwise ops inside the CG
  data-consistency block are NOT counted, so the unrolled/diffusion numbers are
  a lower bound dominated by the (correctly counted) denoiser, multiplied by the
  number of unrolled iterations.
* ``sense`` is classical (no learnable parameters); its FLOPs/params are n/a.
* Timing uses dummy tensors of the correct shape (no multi-GB data load) on the
  available device, after a short warm-up, so it reflects compute only.
"""

from __future__ import annotations

import time
import copy

import h5py
import numpy as np
import torch

from .models import UNet, build_unrolled
from .models.diffusion import DiffusionUNet
from .losses import MixL1L2Loss


# model spec -> "unrolled" methods share build_unrolled via cfg.model
UNROLLED = {"ssdu", "mymodel", "mamba"}
LEARNED = {"supervised", "diffusion"} | UNROLLED
ALL_METHODS = ["sense", "supervised", "ssdu", "mymodel", "mamba", "diffusion"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def peek_hw(cfg):
    """Read (H, W) and the train slice count from the per-slice dataset."""
    from .data.loaders import peek_shape, list_slice_files
    _, h, w = peek_shape(cfg.data_root, cfg.tissue, "train")
    n = len(list_slice_files(cfg.data_root, cfg.tissue, "train"))
    return int(h), int(w), int(n)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def build_model(method, cfg, device):
    if method == "supervised":
        m = UNet(in_ch=2, out_ch=2, chans=cfg.unet_chans,
                 num_pools=cfg.unet_pools, drop=cfg.unet_drop)
    elif method == "diffusion":
        m = DiffusionUNet(in_ch=2, base=cfg.diff_dim)
    elif method in UNROLLED:
        c = copy.copy(cfg)
        c.model = method
        m = build_unrolled(c)
    else:
        raise ValueError(f"no learnable model for method '{method}'")
    return m.to(device)


def make_inputs(method, cfg, H, W, device):
    """Dummy forward inputs of the right shape for one slice (batch 1)."""
    x = torch.randn(1, 2, H, W, device=device)
    if method == "supervised":
        return (x,)
    if method == "diffusion":
        t = torch.randint(0, cfg.diff_timesteps, (1,), device=device)
        return (x, t)
    # unrolled: x_in, sens (complex), trn_mask, loss_mask
    sens = torch.randn(1, cfg.n_coils, H, W, device=device, dtype=torch.complex64)
    mask = torch.ones(1, 1, H, W, device=device)
    return (x, sens, mask, mask)


# --------------------------------------------------------------------------- #
# FLOPs
# --------------------------------------------------------------------------- #
@torch.no_grad()
def model_flops(model, inputs):
    from torch.utils.flop_counter import FlopCounterMode
    fc = FlopCounterMode(display=False)
    model.eval()
    try:
        with fc:
            model(*inputs)
        return int(fc.get_total_flops())
    except Exception:
        return -1  # some op unsupported by the counter


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #
def _forward_loss(method, model, inputs):
    out = model(*inputs)
    if method == "supervised":
        return (out ** 2).mean()
    if method == "diffusion":
        return (out ** 2).mean()
    # unrolled returns (image, mu, nw_kspace); train loss is on nw_kspace
    nw = out[2]
    return (nw ** 2).mean() if nw is not None else (out[0] ** 2).mean()


def time_train_step(method, model, inputs, device, iters=5, warmup=2):
    """Mean seconds for one forward+backward training step."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    cuda = device == "cuda"
    t0 = None
    for i in range(warmup + iters):
        if i == warmup:
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        opt.zero_grad()
        loss = _forward_loss(method, model, inputs)
        loss.backward()
        opt.step()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


# --------------------------------------------------------------------------- #
# top-level
# --------------------------------------------------------------------------- #
def profile_methods(cfg, methods=None, time_iters=5):
    methods = methods or ALL_METHODS
    device = "cuda" if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"
    H, W, n_train = peek_hw(cfg)  # train slice count is the train folder size
    rows = []
    print(f"profiling on {device} | tissue={cfg.tissue} H={H} W={W} "
          f"train_slices~{n_train} batch={cfg.batch_size}\n")

    for method in methods:
        if method == "sense":
            rows.append({"method": "sense", "params": None, "gflops": None,
                         "step_s": None, "epoch_s": None,
                         "note": "classical (no training)"})
            continue
        model = build_model(method, cfg, device)
        inputs = make_inputs(method, cfg, H, W, device)
        params = count_params(model)
        flops = model_flops(model, inputs)
        try:
            step_s = time_train_step(method, model, inputs, device, iters=time_iters)
        except Exception as e:  # pragma: no cover
            step_s = float("nan")
            print(f"  [{method}] timing failed: {e}")

        # dataset-training epoch estimate (zero-shot epoch is noted separately)
        spe = int(np.ceil(n_train / max(cfg.batch_size, 1)))
        epoch_s = step_s * spe if step_s == step_s else float("nan")

        rows.append({
            "method": method,
            "params": params["total"],
            "gflops": (flops / 1e9) if flops and flops > 0 else flops,
            "step_s": step_s,
            "epoch_s": epoch_s,
            "note": "" if (flops and flops > 0) else "flops uncounted",
        })
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    _print_table(rows, cfg)
    return rows


def _fmt(n):
    if n is None:
        return "   n/a"
    if n >= 1e6:
        return f"{n/1e6:6.2f}M"
    if n >= 1e3:
        return f"{n/1e3:6.2f}K"
    return f"{n:6.0f}"


def _print_table(rows, cfg):
    print(f"{'method':<12}{'params':>10}{'GFLOPs/fwd':>13}"
          f"{'step (s)':>11}{'epoch (s)':>12}  note")
    print("-" * 72)
    for r in rows:
        g = "   n/a" if r["gflops"] is None else (
            f"{r['gflops']:9.2f}" if r["gflops"] >= 0 else " uncounted")
        step = "   n/a" if r["step_s"] is None else f"{r['step_s']:9.4f}"
        ep = "   n/a" if r["epoch_s"] is None else f"{r['epoch_s']:10.2f}"
        print(f"{r['method']:<12}{_fmt(r['params']):>10}{g:>13}"
              f"{step:>11}{ep:>12}  {r['note']}")
    print(f"\nepoch (s) = step time x ceil(train_slices/batch)  [dataset training].")
    print(f"zero-shot (single scan): epoch (s) = step time x --zs_num_splits "
          f"(={cfg.zs_num_splits}); total run also bounded by --epochs / early stopping.")
    print("FLOPs exclude FFT / CG ops (lower bound); see profiling.py docstring.")
