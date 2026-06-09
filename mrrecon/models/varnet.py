"""End-to-end VarNet: multi-coil k-space refinement with an RSS output.

Unlike the SENSE-domain unrolled nets (``ssdu`` / ``mymodel``), VarNet keeps the
**full multi-coil k-space** as its state and only root-sum-of-squares (RSS)
combines at the very end. The reconstructed RSS therefore lives in the *same
domain as the fastMRI ground truth* -- so scoring against the RSS GT has no
SENSE-vs-RSS "ceiling": a perfect recon reaches SSIM 1.0. This makes every
method directly comparable to the RSS leaderboard.

Each cascade (E2E-VarNet, Sriram et al. 2020)::

    k_{t+1} = k_t - eta_t * M (k_t - k0)               # k-space data consistency
                  - Expand( CNN( Reduce(k_t) ) )       # sensitivity-modelled refinement

where ``Reduce`` is the SENSE coil-combination (E^H), ``Expand`` is its adjoint
(E), and ``CNN`` is a shared-per-cascade image denoiser. The denoiser is reused
from the rest of the toolkit (the fastMRI U-Net or the hierarchical Mamba
backbone), selected by ``--varnet_cnn``. Sensitivity maps are the pre-computed
ESPIRiT maps passed in with each slice.

The same backbone serves all three training regimes:
  * supervised      -> RSS(output) vs RSS ground truth (SSIM/L1),
  * self-supervised -> SSDU k-space loss on the held-out (Lambda) lines of the
    output multi-coil k-space (VarNet is natively a k-space model),
  * zero-shot       -> the self-supervised loss fit on a single scan.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.transforms import fft2c, ifft2c
from .unet import UNet


# --------------------------------------------------------------------------- #
# SENSE reduce / expand (batched, multi-coil)
# --------------------------------------------------------------------------- #
def sens_reduce(kspace: torch.Tensor, sens: torch.Tensor) -> torch.Tensor:
    """E^H : multi-coil k-space (B,C,H,W) complex -> image (B,2,H,W) real."""
    coil = ifft2c(kspace)                                  # (B,C,H,W) complex
    comb = (coil * torch.conj(sens)).sum(dim=1)            # (B,H,W) complex
    return torch.stack([comb.real, comb.imag], dim=1)      # (B,2,H,W)


def sens_expand(image: torch.Tensor, sens: torch.Tensor) -> torch.Tensor:
    """E : image (B,2,H,W) real -> multi-coil k-space (B,C,H,W) complex."""
    comb = torch.complex(image[:, 0], image[:, 1]).unsqueeze(1)   # (B,1,H,W)
    return fft2c(sens * comb)                              # (B,C,H,W) complex


def kspace_to_rss(kspace: torch.Tensor) -> torch.Tensor:
    """Multi-coil k-space (B,C,H,W) -> RSS magnitude image (B,H,W)."""
    coil = ifft2c(kspace)
    return torch.sqrt((coil.abs() ** 2).sum(dim=1) + 1e-12)


# --------------------------------------------------------------------------- #
# denoiser factory (reuse the toolkit's CNNs)
# --------------------------------------------------------------------------- #
def _build_cnn(cfg):
    # The cascade itself provides the residual (k - soft_dc - model_term), so the
    # refinement CNN must be NON-residual (output a pure correction), matching the
    # E2E-VarNet NormUnet. A residual CNN would inject a spurious E E^H k term.
    kind = getattr(cfg, "varnet_cnn", "unet")
    if kind == "unet":
        return UNet(in_ch=2, out_ch=2, chans=cfg.unet_chans,
                    num_pools=cfg.unet_pools, residual=False)
    if kind == "mamba":
        from .mymodel import MambaUNetDenoiser
        return MambaUNetDenoiser(
            in_ch=2, chans=cfg.mymodel_chans, pools=cfg.mymodel_pools,
            ssm_blocks=getattr(cfg, "mymodel_ssm_blocks", 2),
            mamba_levels=getattr(cfg, "mymodel_mamba_levels", 1),
            d_state=getattr(cfg, "mymodel_dstate", 16),
            expand=getattr(cfg, "mymodel_expand", 1), residual=False)
    raise ValueError(f"unknown varnet_cnn: {kind}")


# --------------------------------------------------------------------------- #
# cascade + full network
# --------------------------------------------------------------------------- #
class VarNetBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = _build_cnn(cfg)
        self.dc_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, k, k0, mask, sens):
        soft_dc = mask * (k - k0) * self.dc_weight         # (B,C,H,W), mask broadcasts over C
        model_term = sens_expand(self.model(sens_reduce(k, sens)), sens)
        return k - soft_dc - model_term


class VarNet(nn.Module):
    """Stack of VarNet cascades. Returns the refined multi-coil k-space.

    ``forward`` signature mirrors the other models loosely but operates on
    *multi-coil k-space* rather than a coil-combined image:
        masked_kspace (B,C,H,W) complex, sens (B,C,H,W) complex, mask (B,1,H,W).
    Use :func:`kspace_to_rss` (or :meth:`reconstruct`) for the RSS image.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        n = getattr(cfg, "varnet_cascades", 8)
        self.cascades = nn.ModuleList([VarNetBlock(cfg) for _ in range(n)])

    def forward(self, masked_kspace, sens, mask):
        k = masked_kspace
        for cascade in self.cascades:
            k = cascade(k, masked_kspace, mask, sens)
        return k                                           # (B,C,H,W) complex

    def reconstruct(self, masked_kspace, sens, mask):
        """-> RSS magnitude image (B,H,W)."""
        return kspace_to_rss(self.forward(masked_kspace, sens, mask))
