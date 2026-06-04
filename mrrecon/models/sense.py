"""Classical CG-SENSE reconstruction (parallel imaging, no learning).

The non-learned baseline. Given multi-coil k-space ``y`` undersampled by mask
``M`` and coil sensitivities ``S``, SENSE solves the regularised least squares

    min_x  || M F S x - y ||^2 + lam || x ||^2

whose normal equations  ``(E^H M E + lam I) x = E^H y``  are solved by the same
conjugate-gradient routine the unrolled networks use for data consistency
(``E x = F S x``). With ``lam`` small this is the standard CG-SENSE recon; a
nonzero ``lam`` (Tikhonov) stabilises ill-conditioned coil geometries.
"""

from __future__ import annotations

import numpy as np
import torch

from ..data.transforms import ifft2c, c2r_chw, r2c_chw
from .data_consistency import conjgrad


@torch.no_grad()
def cg_sense_recon(kspace_slice, sens_slice, omega, lam=1e-2, cg_iter=30,
                   device="cpu", normalize=True):
    """Reconstruct one slice. Returns a complex (H,W) image tensor on ``device``.

    kspace_slice : (C,H,W) complex   -- fully-sampled k-space (will be masked)
    sens_slice   : (C,H,W) complex   -- sensitivity maps
    omega        : (H,W) real        -- acquisition mask
    """
    if normalize:
        scale = np.max(np.abs(kspace_slice))
        kspace_slice = kspace_slice / scale if scale else kspace_slice

    sens = torch.from_numpy(np.ascontiguousarray(sens_slice)).to(device)        # (C,H,W)
    mask = torch.from_numpy(omega.astype(np.float32)).to(device)                # (H,W)
    masked_k = torch.from_numpy(
        np.ascontiguousarray(kspace_slice * omega[None])).to(device)            # (C,H,W) complex

    # rhs = E^H y = sum_c conj(S_c) * ifft2c(M y_c)
    rhs_complex = (ifft2c(masked_k) * torch.conj(sens)).sum(dim=0)              # (H,W)
    rhs_real = c2r_chw(rhs_complex)                                             # (2,H,W)

    mu = torch.tensor(float(lam), device=device)
    x_real = conjgrad(rhs_real, sens, mask, mu, cg_iter)                        # (2,H,W)
    return r2c_chw(x_real)
