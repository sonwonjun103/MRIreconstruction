"""Single-slice reconstruction helpers shared by validation and evaluation.

Given one multi-coil k-space slice (C,H,W), its sensitivity maps and an
acquisition mask Omega, each function returns three magnitude images:
``(reference, zero_filled, reconstruction)``.
"""

from __future__ import annotations

import numpy as np
import torch

from ..data.transforms import sense_combine_np, c2r_np, r2c_np, rss_np
from ..models.sense import cg_sense_recon


def _to_complex_tensor(arr, device):
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device)


def _prep_inputs(kspace_slice, sens_slice, omega, device):
    scale = np.max(np.abs(kspace_slice))
    kspace = kspace_slice / scale if scale else kspace_slice
    ref = sense_combine_np(kspace, sens_slice)                 # (H,W) complex
    zf = sense_combine_np(kspace * omega[None], sens_slice)    # (H,W) complex
    return kspace, ref, zf


@torch.no_grad()
def recon_supervised(model, kspace_slice, sens_slice, omega, device, target_mode="rss"):
    model.eval()
    if target_mode == "rss":
        scale = np.max(np.abs(kspace_slice))
        kspace = kspace_slice / scale if scale else kspace_slice
        ref = rss_np(kspace)                               # fully-sampled RSS
        zf = rss_np(kspace * omega[None])                  # zero-filled RSS
        x = torch.from_numpy(zf.astype(np.float32))[None, None].to(device)
        recon = model(x).cpu().numpy()[0, 0]               # (H,W) magnitude
        return ref, zf, np.abs(recon)
    # sense (2-channel complex) target
    _, ref, zf = _prep_inputs(kspace_slice, sens_slice, omega, device)
    x = torch.from_numpy(c2r_np(zf).astype(np.float32))[None].to(device)
    out = model(x).cpu().numpy()[0]
    recon = r2c_np(out, axis=0)
    return np.abs(ref), np.abs(zf), np.abs(recon)


@torch.no_grad()
def recon_sense(kspace_slice, sens_slice, omega, device, lam=1e-2, cg_iter=30):
    """Classical CG-SENSE; no model. Signature mirrors the learned recon fns
    (the leading ``model`` arg is accepted and ignored for a uniform call site)."""
    _, ref, zf = _prep_inputs(kspace_slice, sens_slice, omega, device)
    recon = cg_sense_recon(kspace_slice, sens_slice, omega, lam=lam,
                           cg_iter=cg_iter, device=device).cpu().numpy()
    return np.abs(ref), np.abs(zf), np.abs(recon)


@torch.no_grad()
def recon_unrolled(model, kspace_slice, sens_slice, omega, device):
    model.eval()
    _, ref, zf = _prep_inputs(kspace_slice, sens_slice, omega, device)
    x = torch.from_numpy(c2r_np(zf).astype(np.float32))[None].to(device)
    sens = _to_complex_tensor(sens_slice, device)[None]                 # (1,C,H,W)
    mask = torch.from_numpy(omega.astype(np.float32))[None, None].to(device)  # (1,1,H,W)
    out, _, _ = model(x, sens, mask, None)
    recon = r2c_np(out.cpu().numpy()[0], axis=0)
    return np.abs(ref), np.abs(zf), np.abs(recon)


@torch.no_grad()
def recon_diffusion(diffusion, kspace_slice, sens_slice, omega, device,
                    steps=100, dc_lam=10.0, dc_iter=5):
    """Zero-shot diffusion reconstruction: DC-guided posterior sampling.

    ``diffusion`` is a GaussianDiffusion wrapping the trained prior. The scan is
    rescaled so the zero-filled image has unit max magnitude (the prior's data
    range); metrics are scale-invariant so this rescaling is harmless.
    """
    _, ref, zf = _prep_inputs(kspace_slice, sens_slice, omega, device)
    scale = np.max(np.abs(zf))
    scale = scale if scale > 0 else 1.0

    AHy = torch.from_numpy(c2r_np(zf / scale).astype(np.float32))[None].to(device)
    sens = _to_complex_tensor(sens_slice, device)[None]
    mask = torch.from_numpy(omega.astype(np.float32))[None, None].to(device)
    out = diffusion.reconstruct(AHy, sens, mask, steps=steps,
                                dc_lam=dc_lam, dc_iter=dc_iter)
    # undo the per-scan normalisation so the recon lives in the same domain as
    # ref/zf (NMSE is not scale-invariant; SSIM/PSNR are, but this keeps all three consistent)
    recon = r2c_np(out.cpu().numpy()[0], axis=0) * scale
    return np.abs(ref), np.abs(zf), np.abs(recon)
