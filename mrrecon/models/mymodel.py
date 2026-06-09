"""mymodel: the **proposed** zero-shot reconstruction network -- ZS-MambaRecon.

A physics-guided **unrolled** network (MoDL/SSDU-style: a learned regulariser
alternating with conjugate-gradient SENSE data consistency) whose regulariser is
a purpose-built **hierarchical 2-D-selective-state-space (Mamba) restoration
backbone**. Drop-in replacement for ``UnrolledSSDU`` -- identical
``forward(input_x, sens_maps, trn_mask, loss_mask)`` signature -- so it runs
unchanged in the SSDU / zero-shot engines and the evaluator (``--model mymodel``).

Design rationale (what recent Mamba-for-MRI work gets wrong, and the fix)
------------------------------------------------------------------------
1. **No ViT patchify.** Vision-Mamba / the old ``mamba.py`` patch-embed with a
   16x16 strided conv, i.e. 16x downsampling before the SSM. For *classification*
   that is fine; for *reconstruction* it discards exactly the high-frequency
   detail we must recover. We keep the feature map at **full resolution** with a
   shallow conv stem (MambaIR's lesson for low-level vision).

2. **2-D selective scan (SS2D), not a 1-D raster scan.** A single flattened
   raster order puts vertically-adjacent pixels W tokens apart and destroys 2-D
   locality. ``SS2D`` scans the token grid in **four directions** (forward/back
   along readout and along phase-encode) and sums them, giving a near-isotropic
   global receptive field. The phase-encode-aligned scans are physically
   motivated: Cartesian undersampling makes aliasing *coherent* along the PE
   image axis, so a sequence model scanning along that axis sees the aliasing
   replicas consecutively and can learn to cancel them -- an inductive bias a
   small-receptive-field conv cannot express.

3. **Local + channel mixing inside every block.** A selective scan mixes tokens
   globally but (a) "forgets" fine local structure and (b) does not mix channels.
   Each Residual State-Space Block (``RSSB``) therefore runs the SS2D in parallel
   with a **depth-wise 3x3 convolution** (local-detail enhancement) and follows it
   with **squeeze-excite channel attention** + a channel MLP. This is the
   combination MambaIR found necessary for restoration-quality output.

4. **Hierarchical (U-shaped) placement of Mamba.** Coherent aliasing is a
   *large-scale* structure (needs a wide receptive field) while edges/texture are
   *local*. We use convolutions at fine scales and place the (quadratic-in-
   sequence-length-free but still costly) Mamba blocks only at the **coarse
   scales**, where the receptive field matters most and the token sequence is
   shortest. ``mymodel_mamba_levels`` controls how many of the coarsest scales
   use Mamba (default: bottleneck only).

5. **Exact data consistency.** The CG-SENSE ``dc_block`` is kept verbatim, so the
   measured k-space is enforced exactly every iteration and the self-supervised
   SSDU / ZS-SSL k-space loss is unchanged -- essential when there is **no
   ground truth** (the measurements are the only supervision).

6. **Per-iteration FiLM on shared weights.** Unrolled weights are *shared* across
   the K iterations (few parameters -> less over-fitting of the single zero-shot
   scan), but early iterations should de-alias aggressively while late ones
   refine. We recover that flexibility for ~0 extra parameters by conditioning
   the shared backbone on the (normalised) iteration index via **FiLM**
   (feature-wise scale/shift) applied to the stem features.

Speed note
----------
The selective scan uses the official CUDA kernel (``mamba_ssm`` /
``selective_scan_fn``) when importable, otherwise a correct pure-PyTorch
reference scan (a sequential loop over the token sequence -- slower). Because
Mamba runs only at coarse scales the sequence length is bounded, so the
reference path is still usable; install ``mamba_ssm`` + ``causal_conv1d`` for
fast runs, or lower ``--mymodel_pools`` / keep ``--mymodel_mamba_levels 1``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import UNet
from .data_consistency import dc_block, to_loss_kspace

# Fast selective-scan kernel if present; else fall back to the reference scan.
try:  # pragma: no cover - depends on environment
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _ssm_fn
    _HAS_SSM_FN = True
except Exception:  # pragma: no cover
    _HAS_SSM_FN = False


# --------------------------------------------------------------------------- #
# selective scan core
# --------------------------------------------------------------------------- #
def _selective_scan_ref(u, delta, A, B, C, D):
    """Canonical (correct, slow) selective scan.

    Shapes: ``u, delta`` (b, d, l); ``A`` (d, n); ``B, C`` (b, n, l); ``D`` (d,).
    Returns ``y`` (b, d, l). This is the reference implementation from the Mamba
    paper -- a sequential recurrence over the L token axis.
    """
    b, d, l = u.shape
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))          # (b,d,l,n)
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)           # (b,d,l,n)
    h = u.new_zeros((b, d, A.shape[1]))
    ys = []
    for i in range(l):
        h = deltaA[:, :, i] * h + deltaB_u[:, :, i]
        ys.append(torch.einsum("bdn,bn->bd", h, C[:, :, i]))
    y = torch.stack(ys, dim=2)                                          # (b,d,l)
    return y + u * D.unsqueeze(-1)


def _selective_scan(u, delta, A, B, C, D):
    """Dispatch to the CUDA kernel when available, else the reference scan."""
    if _HAS_SSM_FN and u.is_cuda:
        # selective_scan_fn expects B,C as (b, g=1, n, l)
        return _ssm_fn(u, delta, A, B.unsqueeze(1), C.unsqueeze(1), D,
                       delta_softplus=False)
    return _selective_scan_ref(u, delta, A, B, C, D)


# --------------------------------------------------------------------------- #
# small building blocks
# --------------------------------------------------------------------------- #
class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for (B, C, H, W) tensors."""

    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        v = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(v + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation channel attention (restores cross-channel mixing)."""

    def __init__(self, c, reduction=8):
        super().__init__()
        h = max(c // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, h, 1), nn.SiLU(inplace=True),
            nn.Conv2d(h, c, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x)


class ConvResBlock(nn.Module):
    """Local conv residual block used at the fine (high-resolution) scales."""

    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(
            LayerNorm2d(c), nn.Conv2d(c, c, 3, padding=1), nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1))
        self.ca = ChannelAttention(c)

    def forward(self, x):
        return x + self.ca(self.body(x))


# --------------------------------------------------------------------------- #
# SS2D: four-directional 2-D selective scan
# --------------------------------------------------------------------------- #
class SS2D(nn.Module):
    """Four-directional selective state-space mixer for a (B, C, H, W) feature map.

    Forward/backward scans along width (readout) and along height (phase-encode)
    are summed, giving a global, near-isotropic receptive field in linear time.
    Each direction has its own (dt, B, C) projection and A/D parameters; a
    depth-wise conv + SiLU precedes the scan (the standard Mamba short conv) and a
    SiLU gate follows it.
    """

    def __init__(self, dim, d_state=16, expand=1, dt_rank=None, n_dirs=4):
        super().__init__()
        self.dim = dim
        self.d_inner = int(expand * dim)
        self.d_state = d_state
        self.dt_rank = dt_rank or max(math.ceil(dim / 16), 1)
        self.n_dirs = n_dirs

        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)     # -> (x, gate z)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, 3, padding=1,
                                groups=self.d_inner, bias=True)         # local short conv
        self.act = nn.SiLU()                                            # out-of-place (acts on views)

        self.x_proj = nn.ModuleList([
            nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
            for _ in range(n_dirs)])
        self.dt_proj = nn.ModuleList([
            nn.Linear(self.dt_rank, self.d_inner, bias=True) for _ in range(n_dirs)])

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_logs = nn.Parameter(torch.log(A).unsqueeze(0).repeat(n_dirs, 1, 1))
        self.Ds = nn.Parameter(torch.ones(n_dirs, self.d_inner))

        self.out_norm = LayerNorm2d(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

    def _scan_one(self, seq, d):
        """seq: (B, d_inner, L) for direction d -> (B, d_inner, L)."""
        xdbl = self.x_proj[d](seq.transpose(1, 2))                     # (B,L,dt_rank+2N)
        dt, Bp, Cp = torch.split(xdbl, [self.dt_rank, self.d_state, self.d_state], -1)
        dt = F.softplus(self.dt_proj[d](dt)).transpose(1, 2)           # (B,d_inner,L)
        Bp = Bp.transpose(1, 2).contiguous()                           # (B,N,L)
        Cp = Cp.transpose(1, 2).contiguous()
        A = -torch.exp(self.A_logs[d])                                 # (d_inner,N)
        return _selective_scan(seq.contiguous(), dt, A, Bp, Cp, self.Ds[d])

    def forward(self, x):                                              # (B,dim,H,W)
        B, _, H, W = x.shape
        xz = self.in_proj(x.permute(0, 2, 3, 1))                       # (B,H,W,2*d_inner)
        xs, z = xz.chunk(2, dim=-1)
        xs = self.act(self.conv2d(xs.permute(0, 3, 1, 2)))             # (B,d_inner,H,W)

        # four token orderings
        s_h = xs.flatten(2)                                            # rows, W-fast (readout)
        s_v = xs.transpose(2, 3).flatten(2)                            # cols, H-fast (phase-encode)
        seqs = [s_h, s_h.flip(-1), s_v, s_v.flip(-1)][: self.n_dirs]

        y = 0
        for d, seq in enumerate(seqs):
            out = self._scan_one(seq, d)                               # (B,d_inner,L)
            if d == 0:                                                 # h-forward
                out = out.reshape(B, self.d_inner, H, W)
            elif d == 1:                                               # h-backward
                out = out.flip(-1).reshape(B, self.d_inner, H, W)
            elif d == 2:                                               # v-forward
                out = out.reshape(B, self.d_inner, W, H).transpose(2, 3)
            else:                                                      # v-backward
                out = out.flip(-1).reshape(B, self.d_inner, W, H).transpose(2, 3)
            y = y + out

        y = self.out_norm(y) * self.act(z.permute(0, 3, 1, 2))         # gated
        return self.out_proj(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class RSSB(nn.Module):
    """Residual State-Space Block: global SS2D + local conv, then channel MLP/SE.

    ``out = x + SS2D(norm(x)) + DWConv(norm(x))`` followed by
    ``out = out + SE(MLP(norm(out)))`` -- global token mixing, local-detail
    recovery, and channel mixing in one block.
    """

    def __init__(self, c, d_state=16, expand=1):
        super().__init__()
        self.norm1 = LayerNorm2d(c)
        self.ss2d = SS2D(c, d_state=d_state, expand=expand)
        self.local = nn.Conv2d(c, c, 3, padding=1, groups=c)          # depth-wise local
        self.norm2 = LayerNorm2d(c)
        self.mlp = nn.Sequential(nn.Conv2d(c, 2 * c, 1), nn.SiLU(inplace=True),
                                 nn.Conv2d(2 * c, c, 1))
        self.ca = ChannelAttention(c)

    def forward(self, x):
        n = self.norm1(x)
        x = x + self.ss2d(n) + self.local(n)
        x = x + self.ca(self.mlp(self.norm2(x)))
        return x


# --------------------------------------------------------------------------- #
# hierarchical Mamba denoiser (the unrolled regulariser)
# --------------------------------------------------------------------------- #
class MambaUNetDenoiser(nn.Module):
    """U-shaped backbone: conv stages at fine scales, RSSB (Mamba) at coarse scales.

    Returns a residual-denoised image of the same shape (B, 2, H, W). Per-sample
    standardisation (fastMRI convention) is applied for scale stability and the
    output projection is zero-initialised, so at initialisation the denoiser is
    the identity (data consistency dominates early -> well-conditioned training).
    A FiLM modulation from the unrolled-iteration index ``t in [0,1]`` lets the
    shared backbone specialise per iteration.
    """

    def __init__(self, in_ch=2, chans=32, pools=3, ssm_blocks=2, mamba_levels=1,
                 d_state=16, expand=1):
        super().__init__()
        self.pools = pools
        self.in_conv = nn.Conv2d(in_ch, chans, 3, padding=1)
        self.iter_mlp = nn.Sequential(nn.Linear(1, chans), nn.SiLU(inplace=True),
                                      nn.Linear(chans, 2 * chans))

        # coarsest `mamba_levels` of the (pools + 1) scales use Mamba
        def use_mamba(depth):
            return depth >= (pools + 1) - mamba_levels

        def stage(ch, depth):
            if use_mamba(depth):
                return nn.Sequential(*[RSSB(ch, d_state, expand)
                                       for _ in range(ssm_blocks)])
            return ConvResBlock(ch)

        self.enc, self.down = nn.ModuleList(), nn.ModuleList()
        ch = chans
        for d in range(pools):
            self.enc.append(stage(ch, d))
            self.down.append(nn.Conv2d(ch, ch * 2, 2, stride=2))
            ch *= 2
        self.bottleneck = stage(ch, pools)

        self.up, self.fuse, self.dec = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for d in range(pools - 1, -1, -1):
            self.up.append(nn.ConvTranspose2d(ch, ch // 2, 2, stride=2))
            ch //= 2
            self.fuse.append(nn.Conv2d(ch * 2, ch, 1))
            self.dec.append(stage(ch, d))

        self.out_conv = nn.Conv2d(chans, in_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, t):
        mean = x.mean((1, 2, 3), keepdim=True)
        std = x.std((1, 2, 3), keepdim=True) + 1e-12
        xn = (x - mean) / std

        mult = 2 ** self.pools
        _, _, h, w = xn.shape
        ph, pw = (mult - h % mult) % mult, (mult - w % mult) % mult
        f = self.in_conv(F.pad(xn, (0, pw, 0, ph)))

        gamma, beta = self.iter_mlp(t.reshape(1, 1)).chunk(2, dim=-1)   # (1,chans) each
        f = f * (1 + gamma[..., None, None]) + beta[..., None, None]

        skips = []
        for stage, down in zip(self.enc, self.down):
            f = stage(f)
            skips.append(f)
            f = down(f)
        f = self.bottleneck(f)
        for up, fuse, dec in zip(self.up, self.fuse, self.dec):
            f = up(f)
            s = skips.pop()
            f = F.pad(f, (0, s.shape[-1] - f.shape[-1], 0, s.shape[-2] - f.shape[-2]))
            f = dec(fuse(torch.cat([f, s], dim=1)))

        out = self.out_conv(f)[..., :h, :w]
        return (xn + out) * std + mean                                 # residual, de-standardised


# --------------------------------------------------------------------------- #
# proposed unrolled network
# --------------------------------------------------------------------------- #
class ZSMambaNet(nn.Module):
    """Unrolled MoDL/SSDU network with the hierarchical Mamba regulariser."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.regularizer = MambaUNetDenoiser(
            in_ch=2, chans=cfg.mymodel_chans, pools=cfg.mymodel_pools,
            ssm_blocks=getattr(cfg, "mymodel_ssm_blocks", 2),
            mamba_levels=getattr(cfg, "mymodel_mamba_levels", 1),
            d_state=getattr(cfg, "mymodel_dstate", 16),
            expand=getattr(cfg, "mymodel_expand", 1))
        self._mu_raw = nn.Parameter(torch.tensor(float(cfg.mu)))

    @property
    def mu(self) -> torch.Tensor:
        return F.softplus(self._mu_raw)

    def forward(self, input_x, sens_maps, trn_mask, loss_mask=None):
        """input_x: (B,2,H,W). Returns (image (B,2,H,W), mu, loss_kspace|None)."""
        mu = self.mu
        x = input_x
        K = self.cfg.nb_unroll_blocks
        for k in range(K):
            t = x.new_tensor(k / max(K - 1, 1))                        # iteration in [0,1]
            z = self.regularizer(x.float(), t)
            rhs = input_x + mu * z
            x = dc_block(rhs, sens_maps, trn_mask, mu, self.cfg.cg_iter)

        nw_kspace = None
        if loss_mask is not None:
            nw_kspace = to_loss_kspace(x, sens_maps, loss_mask)
        return x, mu, nw_kspace


# --------------------------------------------------------------------------- #
# legacy U-Net-regularised unrolled net (kept for reference / comparison)
# --------------------------------------------------------------------------- #
class UnrolledUNet(nn.Module):
    """Previous ``mymodel``: a multi-scale U-Net regulariser unrolled with CG-SENSE
    data consistency. Retained as a baseline (the Mamba ``ZSMambaNet`` above is the
    current proposed model)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.regularizer = UNet(in_ch=2, out_ch=2, chans=cfg.mymodel_chans,
                                num_pools=cfg.mymodel_pools, residual=True)
        self._mu_raw = nn.Parameter(torch.tensor(float(cfg.mu)))

    @property
    def mu(self) -> torch.Tensor:
        return F.softplus(self._mu_raw)

    def forward(self, input_x, sens_maps, trn_mask, loss_mask=None):
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
