"""SENSE data-consistency for the unrolled network (complex, conjugate gradient).

Implements the operators from the SSDU paper:
    * ``EhE_Op``  : (E^H M E + mu I) x   -- the regularised normal operator
    * ``conjgrad``: solves (E^H M E + mu I) x = rhs by conjugate gradient
    * ``to_kspace``: maps the network image to (masked) multi-coil k-space

All FFTs use the centered orthonormal convention from ``data.transforms`` so
the operator matches the numpy pre-processing in the datasets.
"""

from __future__ import annotations

import torch

from ..data.transforms import fft2c, ifft2c, c2r_chw, r2c_chw, c2r_last


def _zdot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Real part of the complex inner product <a, b>, summed over all dims."""
    return (torch.conj(a) * b).sum().real


class SenseEncoder:
    """E and E^H for one slice: sens_maps (C,H,W) complex, mask (H,W) real/complex."""

    def __init__(self, sens_maps: torch.Tensor, mask: torch.Tensor):
        self.sens = sens_maps
        self.mask = mask

    def EhE(self, img: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        coil_imgs = self.sens * img                          # (C,H,W)
        kspace = fft2c(coil_imgs) * self.mask
        comb = (ifft2c(kspace) * torch.conj(self.sens)).sum(dim=0)
        return comb + mu * img

    def to_kspace(self, img: torch.Tensor, apply_mask: bool = True) -> torch.Tensor:
        kspace = fft2c(self.sens * img)                      # (C,H,W)
        return kspace * self.mask if apply_mask else kspace


def conjgrad(rhs_real: torch.Tensor, sens_maps, mask, mu, cg_iter: int) -> torch.Tensor:
    """Solve (E^H M E + mu I) x = rhs for one slice. rhs_real: (2,H,W) -> (2,H,W)."""
    enc = SenseEncoder(sens_maps, mask)
    rhs = r2c_chw(rhs_real)
    mu = mu.type(torch.complex64)

    x = torch.zeros_like(rhs)
    r = rhs
    p = rhs
    rsold = _zdot(r, r)

    for _ in range(cg_iter):
        Ap = enc.EhE(p, mu)
        alpha = rsold / _zdot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = _zdot(r, r)
        beta = rsnew / rsold
        rsold = rsnew
        p = beta * p + r

    return c2r_chw(x)


def dc_block(rhs: torch.Tensor, sens_maps, mask, mu, cg_iter: int) -> torch.Tensor:
    """Batched CG data-consistency. rhs: (B,2,H,W) -> (B,2,H,W)."""
    out = [conjgrad(rhs[i], sens_maps[i], mask[i], mu, cg_iter).unsqueeze(0)
           for i in range(rhs.shape[0])]
    return torch.cat(out, 0)


def to_loss_kspace(nw_output: torch.Tensor, sens_maps, mask) -> torch.Tensor:
    """Network image (B,2,H,W) -> masked k-space (B,C,H,W,2) for the SSDU loss."""
    out = []
    for i in range(nw_output.shape[0]):
        enc = SenseEncoder(sens_maps[i], mask[i])
        kspace = enc.to_kspace(r2c_chw(nw_output[i]), apply_mask=True)
        out.append(c2r_last(kspace).unsqueeze(0))
    return torch.cat(out, 0)
