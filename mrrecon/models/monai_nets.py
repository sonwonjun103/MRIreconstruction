"""Supervised reconstruction backbones from MONAI (UNet, SwinUNETR).

Thin wrapper that adapts MONAI's segmentation-oriented nets to our SENSE
image-to-image setup on 2-channel (real, imag) inputs (B,2,H,W):

  * per-instance standardisation (mean/std) for scale stability,
  * zero-padding H/W up to the network's required multiple, cropped back,
  * a global residual connection (learn the de-aliasing correction).

`build_monai_unet` -> monai UNet (arch='unet'); `build_monai_swinunetr` ->
monai SwinUNETR (arch='swin'). Requires `pip install monai einops`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MonaiSupervised(nn.Module):
    """Wrap a MONAI net for 2-channel image->image with norm/pad/residual."""

    def __init__(self, net: nn.Module, divisor: int, residual: bool = True):
        super().__init__()
        self.net = net
        self.divisor = divisor
        self.residual = residual

    @staticmethod
    def _norm(x):
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True) + 1e-12
        return (x - mean) / std, mean, std

    def _pad(self, x):
        _, _, h, w = x.shape
        ph = (self.divisor - h % self.divisor) % self.divisor
        pw = (self.divisor - w % self.divisor) % self.divisor
        return F.pad(x, (0, pw, 0, ph)), (h, w)

    def forward(self, x):
        xn, mean, std = self._norm(x)
        xp, (h, w) = self._pad(xn)
        out = self.net(xp)[..., :h, :w]
        if self.residual:
            out = xn[..., :h, :w] + out
        return out * std + mean


def _io_channels(cfg):
    """1 channel for the RSS-magnitude target, 2 for the complex SENSE target."""
    return 1 if getattr(cfg, "sup_target", "rss") == "rss" else 2


def build_monai_unet(cfg):
    """MONAI 2-D UNet. Depth/width from --unet_pools / --unet_chans."""
    from monai.networks.nets import UNet
    ch = _io_channels(cfg)
    pools = max(2, cfg.unet_pools)
    channels = tuple(cfg.unet_chans * (2 ** i) for i in range(pools))
    strides = (2,) * (pools - 1)
    net = UNet(spatial_dims=2, in_channels=ch, out_channels=ch,
               channels=channels, strides=strides, num_res_units=2,
               dropout=cfg.unet_drop)
    return MonaiSupervised(net, divisor=2 ** (pools - 1))


def build_monai_swinunetr(cfg):
    """MONAI 2-D SwinUNETR. feature_size from --swin_dim (multiple of 12)."""
    from monai.networks.nets import SwinUNETR
    ch = _io_channels(cfg)
    feat = cfg.swin_dim if cfg.swin_dim % 12 == 0 else 48
    net = SwinUNETR(in_channels=ch, out_channels=ch, spatial_dims=2,
                    feature_size=feat, drop_rate=cfg.unet_drop)
    return MonaiSupervised(net, divisor=32)   # SwinUNETR needs H/W divisible by 32
