"""SwinIR-style Transformer for supervised reconstruction (image -> image).

A compact SwinIR: shallow conv -> stack of Residual Swin Transformer Blocks
(window / shifted-window multi-head self-attention with relative position bias)
-> reconstruction conv, with a global residual. Operates on 2-channel (real,
imag) SENSE images (B,2,H,W). Per-instance standardisation + zero-initialised
output conv make it start near identity (stable training), matching the U-Net
baseline's conventions. Arbitrary H/W are zero-padded to a multiple of the
window size and cropped back.

This is the "Transformer 계열" supervised model (select with ``--arch swin``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, ws):
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, C)


def window_reverse(win, ws, H, W):
    B = int(win.shape[0] / (H * W / ws / ws))
    x = win.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, ws, heads):
        super().__init__()
        self.ws, self.heads = ws, heads
        self.scale = (dim // heads) ** -0.5
        self.rel_bias = nn.Parameter(torch.zeros((2 * ws - 1) ** 2, heads))
        coords = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws),
                                            indexing="ij"))
        coords = torch.flatten(coords, 1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= 2 * ws - 1
        self.register_buffer("rel_idx", rel.sum(-1))
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.rel_bias, std=0.02)

    def forward(self, x, mask=None):
        Bw, N, C = x.shape
        qkv = self.qkv(x).reshape(Bw, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.rel_bias[self.rel_idx.view(-1)].view(N, N, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(Bw // nW, nW, self.heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.heads, N, N)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(Bw, N, C)
        return self.proj(x)


def _shift_mask(H, W, ws, shift, device):
    img = torch.zeros((1, H, W, 1), device=device)
    cnt = 0
    for h in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
        for w in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
            img[:, h, w, :] = cnt
            cnt += 1
    mw = window_partition(img, ws).view(-1, ws * ws)
    attn_mask = mw.unsqueeze(1) - mw.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, ws, shift, mlp_ratio=2.0):
        super().__init__()
        self.ws, self.shift = ws, shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, ws, heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift > 0:
            x = torch.roll(x, (-self.shift, -self.shift), dims=(1, 2))
            mask = _shift_mask(H, W, self.ws, self.shift, x.device)
        else:
            mask = None
        xw = window_partition(x, self.ws)
        xw = self.attn(xw, mask)
        x = window_reverse(xw, self.ws, H, W)
        if self.shift > 0:
            x = torch.roll(x, (self.shift, self.shift), dims=(1, 2))
        x = shortcut + x.view(B, L, C)
        return x + self.mlp(self.norm2(x))


class RSTB(nn.Module):
    """Residual Swin Transformer Block: several SwinBlocks + conv, with residual."""

    def __init__(self, dim, heads, ws, n_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(dim, heads, ws, shift=0 if (i % 2 == 0) else ws // 2)
            for i in range(n_blocks)])
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x, H, W):
        res = x
        for blk in self.blocks:
            x = blk(x, H, W)
        B, L, C = x.shape
        x = self.conv(x.transpose(1, 2).view(B, C, H, W)).flatten(2).transpose(1, 2)
        return x + res


class SwinIR(nn.Module):
    def __init__(self, in_ch=2, dim=48, depths=4, blocks_per_stage=4, heads=6,
                 window=8, residual=True):
        super().__init__()
        self.window = window
        self.residual = residual
        self.conv_first = nn.Conv2d(in_ch, dim, 3, padding=1)
        self.stages = nn.ModuleList([
            RSTB(dim, heads, window, blocks_per_stage) for _ in range(depths)])
        self.norm = nn.LayerNorm(dim)
        self.conv_body = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv_last = nn.Conv2d(dim, in_ch, 3, padding=1)
        nn.init.zeros_(self.conv_last.weight)
        nn.init.zeros_(self.conv_last.bias)

    @staticmethod
    def _norm(x):
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True) + 1e-12
        return (x - mean) / std, mean, std

    def _pad(self, x):
        _, _, h, w = x.shape
        ph = (self.window - h % self.window) % self.window
        pw = (self.window - w % self.window) % self.window
        return F.pad(x, (0, pw, 0, ph)), (h, w)

    def forward(self, x):
        xn, mean, std = self._norm(x)
        xp, (h0, w0) = self._pad(xn)
        B, C, H, W = xp.shape

        f = self.conv_first(xp)
        res = f
        tok = f.flatten(2).transpose(1, 2)               # (B, H*W, dim)
        for stage in self.stages:
            tok = stage(tok, H, W)
        tok = self.norm(tok)
        body = tok.transpose(1, 2).view(B, -1, H, W)
        f = self.conv_body(body) + res
        out = self.conv_last(f)[..., :h0, :w0]

        out = xn[..., :h0, :w0] + out if self.residual else out
        return out * std + mean
