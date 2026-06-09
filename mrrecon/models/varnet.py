"""Multi-coil k-space reconstruction with an RSS output. Two models:

  * ``DCCNN`` -- our **Deep Cascade**: soft-DC cascades with a pluggable backbone
    (``--cnn`` unet/swin/mamba) and FIXED precomputed ESPIRiT sensitivities. NOT
    the official E2E-VarNet (no learned sensitivity-map estimation).
  * ``VarNet`` -- the **official** facebookresearch/fastMRI E2E-VarNet wrapper
    (learned SME). Selected by the ``varnet`` method.

Unlike the SENSE-domain unrolled nets (``ssdu`` / ``mymodel``), both keep the
**full multi-coil k-space** as state and only root-sum-of-squares (RSS) combine
at the very end. The RSS therefore lives in the *same domain as the fastMRI
ground truth* -- scoring against the RSS GT has no SENSE-vs-RSS "ceiling" (a
perfect recon reaches SSIM 1.0), directly comparable to the RSS leaderboard.

Each DCCNN cascade (cf. E2E-VarNet, Sriram et al. 2020)::

    k_{t+1} = k_t - eta_t * M (k_t - k0)               # k-space data consistency
                  - Expand( CNN( Reduce(k_t) ) )       # sensitivity-modelled refinement

where ``Reduce`` is the SENSE coil-combination (E^H), ``Expand`` is its adjoint
(E), and ``CNN`` is a shared-per-cascade image denoiser. The denoiser is reused
from the rest of the toolkit (MONAI U-Net/SwinUNETR or the hierarchical Mamba
backbone), selected by ``--cnn``. Sensitivity maps are the pre-computed
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
    # unet/swin reuse the SAME MONAI backbones as the no-DC supervised path
    # (in_out_ch=2 for the complex coil-combined image, residual=False because the
    # cascade supplies the residual) -> a clean "same backbone +/- DC" ablation.
    kind = getattr(cfg, "cnn", "unet")
    if kind == "unet":
        from .monai_nets import build_monai_unet
        return build_monai_unet(cfg, in_out_ch=2, residual=False)
    if kind == "swin":
        from .monai_nets import build_monai_swinunetr
        return build_monai_swinunetr(cfg, in_out_ch=2, residual=False)
    if kind == "mamba":
        from .mymodel import MambaUNetDenoiser
        return MambaUNetDenoiser(
            in_ch=2, chans=cfg.mymodel_chans, pools=cfg.mymodel_pools,
            ssm_blocks=getattr(cfg, "mymodel_ssm_blocks", 2),
            mamba_levels=getattr(cfg, "mymodel_mamba_levels", 1),
            d_state=getattr(cfg, "mymodel_dstate", 16),
            expand=getattr(cfg, "mymodel_expand", 1), residual=False)
    raise ValueError(f"unknown cnn backbone: {kind}")


# --------------------------------------------------------------------------- #
# cascade + full network
# --------------------------------------------------------------------------- #
class DCBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = _build_cnn(cfg)
        self.dc_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, k, k0, mask, sens):
        soft_dc = mask * (k - k0) * self.dc_weight         # (B,C,H,W), mask broadcasts over C
        model_term = sens_expand(self.model(sens_reduce(k, sens)), sens)
        return k - soft_dc - model_term


class DCCNN(nn.Module):
    """Deep Cascade: stack of soft-DC cascades with a pluggable backbone (--cnn:
    unet/swin/mamba) and FIXED (precomputed ESPIRiT) sensitivity maps. Returns the
    refined multi-coil k-space; operates on
        masked_kspace (B,C,H,W) complex, sens (B,C,H,W) complex, mask (B,1,H,W).
    Use :func:`kspace_to_rss` (or :meth:`reconstruct`) for the RSS image.

    NOTE: not the official E2E-VarNet -- there is no learned sensitivity-map
    estimation here. The official VarNet (learned SME) is the ``VarNet`` class.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        n = getattr(cfg, "dc_cascades", 8)
        self.cascades = nn.ModuleList([DCBlock(cfg) for _ in range(n)])

    def forward(self, masked_kspace, sens, mask):
        k = masked_kspace
        for cascade in self.cascades:
            k = cascade(k, masked_kspace, mask, sens)
        return k                                           # (B,C,H,W) complex

    def reconstruct(self, masked_kspace, sens, mask):
        """-> RSS magnitude image (B,H,W)."""
        return kspace_to_rss(self.forward(masked_kspace, sens, mask))


class VarNet(nn.Module):
    """Thin wrapper around the official facebookresearch/fastMRI ``VarNet``
    (verbatim ``fastmri.models.VarNet``, including the learned Sensitivity-Map
    Estimation module). Adapts our calling convention -- complex k-space
    (B,C,H,W), our (B,1,H,W) mask -- to the official one (real-last (B,C,H,W,2)
    k-space, bool (B,1,1,W,1) mask) and returns the RSS image (B,H,W).

    The official net estimates sensitivity maps internally from the ACS lines, so
    the precomputed ESPIRiT ``sens`` argument is ignored. Install with
    ``pip install fastmri --no-deps`` (we already provide torch/numpy/h5py)."""

    def __init__(self, cfg):
        super().__init__()
        from fastmri.models import VarNet as _FastmriVarNet
        self.acs = cfg.acs_lines
        self.net = _FastmriVarNet(
            num_cascades=cfg.varnet_cascades,
            sens_chans=getattr(cfg, "varnet_sens_chans", 8),
            sens_pools=getattr(cfg, "varnet_sens_pools", 4),
            chans=getattr(cfg, "varnet_unet_chans", 18),
            pools=getattr(cfg, "varnet_unet_pools", 4))

    def reconstruct(self, masked_kspace, sens, mask):
        import torch as _t
        B, _, _, W = masked_kspace.shape
        ks = _t.view_as_real(masked_kspace.contiguous())        # (B,C,H,W,2)
        prof = (mask[:, :, 0, :] > 0).reshape(B, 1, 1, W, 1)    # (B,1,1,W,1) bool
        return self.net(ks, prof, num_low_frequencies=self.acs)  # (B,H,W) RSS

    def forward(self, masked_kspace, sens, mask):
        return self.reconstruct(masked_kspace, sens, mask)


def build_recon(cfg):
    """Official E2E-VarNet (learned SME) if ``cfg.varnet_official`` else our DCCNN
    (Deep Cascade with a pluggable backbone and fixed ESPIRiT sensitivities)."""
    return VarNet(cfg) if getattr(cfg, "varnet_official", False) else DCCNN(cfg)
