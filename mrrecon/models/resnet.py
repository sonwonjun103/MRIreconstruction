"""ResNet denoiser used as the regulariser inside the unrolled SSDU network.

Ported from ``Paper/SSDU/models/model.py``: an input conv, a stack of residual
blocks with a fixed 0.1 scaling, a long skip connection, and an output conv.
Operates on 2-channel (real, imag) images: (B,2,H,W) -> (B,2,H,W).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv(in_ch, out_ch, activation=True):
    layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)]
    if activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class _ResBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.body = nn.Sequential(_conv(features, features, activation=True),
                                  _conv(features, features, activation=False))

    def forward(self, x):
        return self.body(x)


class ResNetDenoiser(nn.Module):
    def __init__(self, in_ch=2, features=64, num_blocks=15, scale=0.1):
        super().__init__()
        self.scale = scale
        self.head = _conv(in_ch, features, activation=False)
        self.blocks = nn.ModuleList([_ResBlock(features) for _ in range(num_blocks)])
        self.mid = _conv(features, features, activation=False)
        self.tail = _conv(features, in_ch, activation=False)

    def forward(self, x):
        h = self.head(x)
        out = h
        for block in self.blocks:
            out = out + self.scale * block(out)
        out = self.mid(out) + h
        return self.tail(out)
