"""mymodel: the recommended reconstruction network.

A physics-guided **unrolled** network -- like SSDU / MoDL -- but with a
multi-scale **U-Net regulariser** in place of the shallow ResNet denoiser used
by SSDU and ZS-SSL. It is a drop-in replacement for ``UnrolledSSDU``: identical
``forward(input_x, sens_maps, trn_mask, loss_mask)`` signature, so it works
unchanged in the supervised-free SSDU and zero-shot engines and in the evaluator.

Why this should beat the baselines
-----------------------------------
* SSDU / ZS-SSL regularise each unrolled step with a ResNet whose receptive
  field is small (stacked 3x3 convs). Cartesian undersampling produces
  *coherent, large-scale* aliasing along the phase-encode direction, which a
  small receptive field struggles to remove. A U-Net sees the whole image
  through its encoder/decoder, so it suppresses global aliasing far better --
  this is exactly why U-Net regularisers (VarNet, MoDL-UNet) top the fastMRI
  leaderboard.
* Weights are **shared across unrolled iterations** (recurrent), so the
  parameter count stays modest. That matters for the zero-shot setting where we
  fit a *single* scan: fewer parameters + early stopping = less over-fitting.
* The data-consistency block is unchanged (conjugate-gradient SENSE), so the
  measured k-space is still enforced exactly and the SSDU / ZS-SSL
  self-supervised k-space loss applies verbatim.

Architecture (per unrolled iteration k = 1..K)::

    z_k = UNet(x_{k-1})                                  # multi-scale denoise / de-alias
    x_k = argmin_x ||M_Theta (E x - y)||^2 + mu ||x - (x_in + mu z_k)||^2
        = CG-solve of  (E^H M_Theta E + mu I) x = x_in + mu z_k

with a single learnable scalar ``mu`` (>0 via softplus) and one shared U-Net.
For the SSDU / ZS-SSL loss, the final image x_K is mapped to k-space at the
loss (Lambda) locations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import UNet
from .data_consistency import dc_block, to_loss_kspace


class UnrolledUNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # shared U-Net regulariser; residual so it learns a correction term
        self.regularizer = UNet(in_ch=2, out_ch=2, chans=cfg.mymodel_chans,
                                num_pools=cfg.mymodel_pools, residual=True)
        # parametrise mu through softplus so it stays strictly positive
        self._mu_raw = nn.Parameter(torch.tensor(float(cfg.mu)))

    @property
    def mu(self) -> torch.Tensor:
        return F.softplus(self._mu_raw)

    def forward(self, input_x, sens_maps, trn_mask, loss_mask=None):
        """input_x: (B,2,H,W). Returns (image (B,2,H,W), mu, loss_kspace|None)."""
        mu = self.mu
        x = input_x
        for _ in range(self.cfg.nb_unroll_blocks):
            z = self.regularizer(x.float())
            rhs = input_x + mu * z
            x = dc_block(rhs, sens_maps, trn_mask, mu, self.cfg.cg_iter)

        nw_kspace = None
        if loss_mask is not None:
            nw_kspace = to_loss_kspace(x, sens_maps, loss_mask)
        return x, mu, nw_kspace
