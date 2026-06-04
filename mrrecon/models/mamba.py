"""mamba: a Mamba-ViT-regularised unrolled network (zero-shot capable).

Same physics-guided unrolled scheme as ``mymodel`` / ``UnrolledSSDU`` (denoiser
alternating with conjugate-gradient SENSE data consistency, SSDU/ZS-SSL k-space
loss), but the regulariser is a **Mamba-ViT**: the image is patch-embedded into
a token sequence, processed by bidirectional selective state-space (Mamba)
blocks, and un-embedded with a residual connection.

Why Mamba for MRI reconstruction
---------------------------------
Selective state-space models mix information along the whole token sequence with
linear-time complexity, giving a global receptive field like a Transformer but
without quadratic attention cost. That global context helps remove the coherent,
long-range aliasing of Cartesian undersampling -- the same motivation as the
U-Net regulariser, with a sequence-model inductive bias instead of multi-scale
convolutions.

Drop-in: identical ``forward(input_x, sens_maps, trn_mask, loss_mask)`` API to
``UnrolledSSDU``, so it runs unchanged in the SSDU and zero-shot engines and the
evaluator (select with ``--model mamba``).

Performance note
----------------
If the official CUDA kernels (``mamba_ssm`` + ``causal_conv1d``) are installed
they are used automatically and are fast. Otherwise a **pure-PyTorch fallback**
selective scan runs -- correct but slow (a Python loop over the token sequence).
For real zero-shot runs without the CUDA kernels, keep ``--mamba_patch`` large
(shorter sequence) and ``--mamba_depth`` small, or install ``mamba_ssm``.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_consistency import dc_block, to_loss_kspace

try:
    from mamba_ssm import Mamba as _MambaSSM
    _HAS_MAMBA_SSM = True
except Exception:  # pragma: no cover - depends on environment
    _HAS_MAMBA_SSM = False


# --------------------------------------------------------------------------- #
# selective SSM: official kernel if available, else a pure-PyTorch fallback
# --------------------------------------------------------------------------- #
class _FallbackSelectiveSSM(nn.Module):
    """Correctness-only pure-PyTorch selective scan (slow; sequential over L)."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):                                  # (B, L, D)
        B, L, _ = x.shape
        x_, z = self.in_proj(x).chunk(2, dim=-1)
        x_ = self.act(self.conv1d(x_.transpose(1, 2))[..., :L].transpose(1, 2))

        dt, B_, C_ = torch.split(self.x_proj(x_),
                                 [self.d_inner, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)

        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            dt_t = dt[:, t]
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            B_bar = dt_t.unsqueeze(-1) * B_[:, t].unsqueeze(1)
            h = A_bar * h + B_bar * x_[:, t].unsqueeze(-1)
            ys.append((h * C_[:, t].unsqueeze(1)).sum(-1) + self.D * x_[:, t])
        y = torch.stack(ys, dim=1) * self.act(z)
        return self.out_proj(y)


def _make_mamba(d_model, d_state, d_conv, expand):
    if _HAS_MAMBA_SSM:
        return _MambaSSM(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    return _FallbackSelectiveSSM(d_model, d_state, d_conv, expand)


class _BidirectionalMamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.fwd = _make_mamba(d_model, d_state, d_conv, expand)
        self.bwd = _make_mamba(d_model, d_state, d_conv, expand)

    def forward(self, x):                                  # (B, L, D)
        return 0.5 * (self.fwd(x) + self.bwd(x.flip(1)).flip(1))


class _MambaViTBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = _BidirectionalMamba(dim, d_state, d_conv, expand)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MambaViTDenoiser(nn.Module):
    """Image (B,2,H,W) -> image (B,2,H,W), residual. Handles arbitrary H, W.

    Inputs are zero-padded to a multiple of ``patch`` before patch embedding and
    cropped back afterwards. A 2-D learnable positional embedding is bilinearly
    interpolated to the current patch grid, so non-square sizes (640x368, etc.)
    are supported without reconfiguration.
    """

    def __init__(self, in_ch=2, dim=128, depth=4, patch=16, d_state=16,
                 d_conv=4, expand=2, base_grid=(40, 24)):
        super().__init__()
        self.patch = patch
        self.dim = dim
        self.embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, dim, *base_grid))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([
            _MambaViTBlock(dim, 2.0, d_state, d_conv, expand) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.unembed = nn.ConvTranspose2d(dim, in_ch, kernel_size=patch, stride=patch)
        # zero-init the output projection -> denoiser starts as identity (x + 0),
        # so data consistency dominates early and training is well-conditioned.
        nn.init.zeros_(self.unembed.weight)
        nn.init.zeros_(self.unembed.bias)

    def forward(self, x):
        B, _, H, W = x.shape
        ph = (self.patch - H % self.patch) % self.patch
        pw = (self.patch - W % self.patch) % self.patch
        xp = F.pad(x, (0, pw, 0, ph))

        tok = self.embed(xp)                               # (B, dim, gh, gw)
        gh, gw = tok.shape[-2:]
        pos = F.interpolate(self.pos, size=(gh, gw), mode="bilinear",
                            align_corners=False)
        tok = (tok + pos).flatten(2).transpose(1, 2)       # (B, gh*gw, dim)
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.norm(tok).transpose(1, 2).reshape(B, self.dim, gh, gw)

        out = self.unembed(tok)[..., :H, :W]
        return x + out                                     # residual


# --------------------------------------------------------------------------- #
# unrolled network with the Mamba-ViT regulariser (shared across iterations)
# --------------------------------------------------------------------------- #
class UnrolledMamba(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if not _HAS_MAMBA_SSM:
            warnings.warn(
                "mamba_ssm not found -> using the slow pure-PyTorch selective-scan "
                "fallback. Install 'mamba_ssm' and 'causal_conv1d' for fast runs, "
                "or keep --mamba_patch large / --mamba_depth small.")
        self.regularizer = MambaViTDenoiser(
            in_ch=2, dim=cfg.mamba_dim, depth=cfg.mamba_depth, patch=cfg.mamba_patch,
            d_state=cfg.mamba_dstate, expand=cfg.mamba_expand)
        self._mu_raw = nn.Parameter(torch.tensor(float(cfg.mu)))

    @property
    def mu(self):
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
