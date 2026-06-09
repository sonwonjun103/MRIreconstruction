"""Training losses."""

from __future__ import annotations

import torch
import torch.nn as nn


class MixL1L2Loss(nn.Module):
    """Normalised L1+L2 loss used by SSDU (Yaman et al.).

    ``0.5 * ||yhat - y||_2 / ||y||_2 + 0.5 * ||yhat - y||_1 / ||y||_1``.
    Operates on the real-valued k-space tensors (B,C,H,W,2).
    """

    def __init__(self, scaler: float = 0.5, eps: float = 1e-8):
        super().__init__()
        self.scaler = scaler
        self.eps = eps

    def forward(self, yhat, y):
        l2 = torch.norm(yhat - y) / (torch.norm(y) + self.eps)
        l1 = torch.norm(yhat - y, p=1) / (torch.norm(y, p=1) + self.eps)
        return self.scaler * l2 + self.scaler * l1


def _magnitude(x):
    """-> (B,1,H,W) magnitude. (B,2,H,W) real/imag -> sqrt; (B,1,H,W) -> |x|."""
    if x.shape[1] == 1:
        return x.abs()
    return torch.sqrt(x[:, 0:1] ** 2 + x[:, 1:2] ** 2 + 1e-12)


class SupervisedLoss(nn.Module):
    """Image-domain supervised loss for reconstruction.

    kind:
      'l1'      -> L1 on the complex 2-channel image (default)
      'l2'      -> MSE on the complex 2-channel image
      'ssim'    -> 1 - SSIM on the magnitude image (fastMRI/VarNet style)
      'l1ssim'  -> L1 + ssim_weight * (1 - SSIM) 
    SSIM is computed on per-sample max-normalised magnitudes (data_range=1).
    """

    def __init__(self, kind="l1", ssim_weight=1.0):
        super().__init__()
        self.kind = kind
        self.ssim_weight = ssim_weight
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        if kind in ("ssim", "l1ssim"):
            from monai.losses import SSIMLoss
            self.ssim = SSIMLoss(spatial_dims=2, data_range=1.0)

    def _ssim_term(self, out, tgt):
        mo, mt = _magnitude(out), _magnitude(tgt)
        scale = mt.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
        return self.ssim(mo / scale, mt / scale)        # already 1 - SSIM

    def forward(self, out, tgt):
        if self.kind == "l1":
            return self.l1(out, tgt)
        if self.kind == "l2":
            return self.l2(out, tgt)
        if self.kind == "ssim":
            return self._ssim_term(out, tgt)
        if self.kind == "l1ssim":
            return self.l1(out, tgt) + self.ssim_weight * self._ssim_term(out, tgt)
        raise ValueError(f"unknown supervised loss: {self.kind}")
