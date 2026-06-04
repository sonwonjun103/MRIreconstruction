"""Diffusion model for zero-shot MRI reconstruction (score-MRI / DPS style).

Two pieces:

* ``DiffusionUNet`` -- a compact time-conditioned U-Net that predicts the noise
  ``eps`` added to a 2-channel (real, imag) SENSE image.
* ``GaussianDiffusion`` -- the DDPM process: training loss (noise MSE) and a
  **data-consistency-guided DDIM sampler** for reconstruction.

Why this is "zero-shot"
-----------------------
The U-Net is trained as an *unconditional* image prior ``p(x)`` on fully-sampled
SENSE images -- it never sees undersampling masks or paired under/fully-sampled
data. Reconstruction of any accelerated scan is then performed at inference time
by **posterior sampling**: at every reverse-diffusion step the prior's estimate
``x0`` is pulled toward the measured k-space by a SENSE data-consistency step
(the same conjugate-gradient ``dc_block`` the unrolled networks use). No
retraining is needed for new scans, masks or acceleration rates -- hence
zero-shot. This is the score-MRI (Chung & Ye) / DPS recipe with multi-coil
SENSE data consistency.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_consistency import dc_block


# --------------------------------------------------------------------------- #
# time-conditioned U-Net (noise predictor)
# --------------------------------------------------------------------------- #
def timestep_embedding(t, dim):
    """Sinusoidal embedding of integer timesteps t (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class _ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(t)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class DiffusionUNet(nn.Module):
    """eps-predictor on (B,2,H,W). Symmetric U-Net with ``levels`` down/up
    stages; input is padded to a multiple of ``2**levels`` and cropped back."""

    def __init__(self, in_ch=2, base=64, ch_mults=(1, 2, 2), t_dim=256):
        super().__init__()
        self.levels = len(ch_mults)
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(),
                                   nn.Linear(t_dim, t_dim))
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)

        chans = [base * m for m in ch_mults]
        self.down = nn.ModuleList()
        self.downsample = nn.ModuleList()
        prev = base
        for c in chans:
            self.down.append(_ResBlock(prev, c, t_dim))
            self.downsample.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
            prev = c
        self.mid = _ResBlock(prev, prev, t_dim)

        self.up = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for c in reversed(chans):
            self.upsample.append(nn.ConvTranspose2d(prev, c, 2, stride=2))
            self.up.append(_ResBlock(c * 2, c, t_dim))  # concat skip
            prev = c
        self.out_norm = nn.GroupNorm(min(8, prev), prev)
        self.out_conv = nn.Conv2d(prev, in_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _pad(self, x):
        m = 2 ** self.levels
        _, _, h, w = x.shape
        ph, pw = (m - h % m) % m, (m - w % m) % m
        return F.pad(x, (0, pw, 0, ph)), (h, w)

    def forward(self, x, t):
        temb = self.t_mlp(timestep_embedding(t, self.t_dim))
        x, (h0, w0) = self._pad(x)
        h = self.in_conv(x)

        skips = []
        for block, ds in zip(self.down, self.downsample):
            h = block(h, temb)
            skips.append(h)
            h = ds(h)
        h = self.mid(h, temb)

        for up, block, skip in zip(self.upsample, self.up, reversed(skips)):
            h = up(h)
            h = F.pad(h, (0, skip.shape[-1] - h.shape[-1],
                          0, skip.shape[-2] - h.shape[-2]))
            h = block(torch.cat([h, skip], dim=1), temb)

        h = self.out_conv(F.silu(self.out_norm(h)))
        return h[..., :h0, :w0]


# --------------------------------------------------------------------------- #
# Gaussian diffusion process (DDPM training + DC-guided DDIM sampling)
# --------------------------------------------------------------------------- #
def make_beta_schedule(timesteps, kind="cosine"):
    if kind == "linear":
        return torch.linspace(1e-4, 0.02, timesteps)
    # cosine schedule (Nichol & Dhariwal)
    s = 0.008
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    ac = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return betas.clamp(1e-4, 0.999)


class GaussianDiffusion:
    def __init__(self, model, timesteps=1000, schedule="cosine", device="cpu"):
        self.model = model
        self.T = timesteps
        self.device = device
        betas = make_beta_schedule(timesteps, schedule).to(device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.acum = torch.cumprod(self.alphas, dim=0)            # \bar alpha_t

    # ---- training ---- #
    def q_sample(self, x0, t, noise):
        a = self.acum[t][:, None, None, None]
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def p_losses(self, x0):
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = self.model(x_t, t)
        return F.mse_loss(pred, noise)

    def _predict_x0(self, x_t, t, eps, clip=True):
        a = self.acum[t][:, None, None, None]
        x0 = (x_t - (1 - a).sqrt() * eps) / a.sqrt().clamp(min=1e-8)
        # the prior is trained on images normalised to max-magnitude 1, so the
        # real/imag channels live in [-1, 1]; clamping prevents the 1/sqrt(a_t)
        # blow-up at large t and is standard practice (static thresholding).
        return x0.clamp(-1.0, 1.0) if clip else x0

    # ---- DC-guided DDIM reconstruction ---- #
    @torch.no_grad()
    def reconstruct(self, AHy, sens, mask, steps=100, dc_lam=10.0, dc_iter=5,
                    eta=0.0):
        """Zero-shot posterior sampling for one slice.

        AHy  : (1,2,H,W) zero-filled SENSE recon  (E^H y, network input)
        sens : (1,C,H,W) complex sensitivity maps
        mask : (1,1,H,W) acquisition mask Omega
        Returns the reconstructed image (1,2,H,W).
        """
        self.model.eval()
        device = self.device
        lam = torch.tensor(float(dc_lam), device=device)

        # DDIM timestep subsequence
        ts = torch.linspace(self.T - 1, 0, steps, device=device).round().long()
        x = torch.randn_like(AHy)
        for i in range(steps):
            t = ts[i]
            t_b = t.expand(AHy.shape[0])
            eps = self.model(x, t_b)
            x0 = self._predict_x0(x, t_b, eps)

            # data consistency: x0 <- argmin ||M F S x - y||^2 + lam||x - x0||^2
            #                   = CG-solve (E^H M E + lam I) x = E^H y + lam x0
            rhs = AHy + lam * x0
            x0 = dc_block(rhs, sens, mask, lam, dc_iter).clamp(-1.0, 1.0)

            if i == steps - 1:
                x = x0
                break
            # DDIM step toward the next (smaller) timestep
            a_t = self.acum[t]
            a_prev = self.acum[ts[i + 1]]
            eps = (x - a_t.sqrt() * x0) / (1 - a_t).sqrt().clamp(min=1e-8)
            sigma = eta * ((1 - a_prev) / (1 - a_t)).sqrt() * (1 - a_t / a_prev).sqrt()
            x = a_prev.sqrt() * x0 + (1 - a_prev - sigma ** 2).clamp(min=0).sqrt() * eps
            if sigma > 0:
                x = x + sigma * torch.randn_like(x)
        return x
