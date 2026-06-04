"""A standard U-Net (fastMRI-style), used for the supervised baseline.

Operates on 2-channel (real, imag) SENSE images: input (B,2,H,W) -> (B,2,H,W).
Architecture mirrors the fastMRI U-Net: double-conv blocks with InstanceNorm +
LeakyReLU, max-pool downsampling, transpose-conv upsampling, skip connections.
Input is zero-padded to a multiple of ``2**num_pools`` and cropped back so
arbitrary H/W (e.g. 640x368) are handled.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(drop),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(drop),
        )

    def forward(self, x):
        return self.layers(x)


class _TransposeConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=2, chans=32, num_pools=4, drop=0.0,
                 residual=True):
        super().__init__()
        self.num_pools = num_pools
        self.residual = residual

        self.down = nn.ModuleList([_ConvBlock(in_ch, chans, drop)])
        ch = chans
        for _ in range(num_pools - 1):
            self.down.append(_ConvBlock(ch, ch * 2, drop))
            ch *= 2
        self.bottleneck = _ConvBlock(ch, ch * 2, drop)

        self.up_conv = nn.ModuleList()
        self.up_block = nn.ModuleList()
        for _ in range(num_pools):
            self.up_conv.append(_TransposeConv(ch * 2, ch))
            self.up_block.append(_ConvBlock(ch * 2, ch, drop))
            ch //= 2

        self.final = nn.Conv2d(ch * 2, out_ch, 1)

    @staticmethod
    def _pad_to(x, mult):
        _, _, h, w = x.shape
        ph = (mult - h % mult) % mult
        pw = (mult - w % mult) % mult
        x = F.pad(x, (0, pw, 0, ph))
        return x, (h, w)

    @staticmethod
    def _norm(x):
        """Per-sample standardisation (fastMRI convention) for scale stability."""
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True) + 1e-12
        return (x - mean) / std, mean, std

    def _backbone(self, x):
        x, (h, w) = self._pad_to(x, 2 ** self.num_pools)
        stack = []
        for layer in self.down:
            x = layer(x)
            stack.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)

        for tconv, block in zip(self.up_conv, self.up_block):
            x = tconv(x)
            skip = stack.pop()
            # pad in case of odd sizes from pooling
            x = F.pad(x, (0, skip.shape[-1] - x.shape[-1],
                          0, skip.shape[-2] - x.shape[-2]))
            x = block(torch.cat([x, skip], dim=1))

        x = self.final(x)
        return x[..., :h, :w]

    def forward(self, x):
        xn, mean, std = self._norm(x)
        out = self._backbone(xn)
        if self.residual:
            out = xn + out
        return out * std + mean
