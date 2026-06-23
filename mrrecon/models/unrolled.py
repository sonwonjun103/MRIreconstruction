"""Unrolled SSDU network: alternating denoiser + CG data-consistency.

For ``nb_unroll_blocks`` iterations::

    x  <- Denoiser(x)
    x  <- argmin_x  || M_Theta (E x - y) ||^2 + mu || x - (x_in + mu*z) ||^2
          (solved by conjugate gradient in ``dc_block``)

The training forward also returns the network output mapped to k-space at the
loss (Lambda) locations, which is what the SSDU loss is computed against.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .resnet import ResNetDenoiser
from .data_consistency import dc_block, to_loss_kspace 


class UnrolledSSDU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.denoiser = ResNetDenoiser(in_ch=2, features=64,
                                       num_blocks=cfg.res_blocks)
        self.lam = nn.Parameter(torch.tensor(cfg.mu, dtype=torch.float32))

    def forward(self, x_in, sens_maps, trn_mask, loss_mask=None):
        """x_in: (B,2,H,W). Returns (image (B,2,H,W), lam, loss_kspace or None)."""
        x = x_in
        for _ in range(self.cfg.nb_unroll_blocks):
            z = self.denoiser(x.float())
            rhs = x_in + self.lam * z
            x = dc_block(rhs, sens_maps, trn_mask, self.lam, self.cfg.cg_iter)

        nw_kspace = None
        if loss_mask is not None:
            nw_kspace = to_loss_kspace(x, sens_maps, loss_mask)
        return x, self.lam, nw_kspace
