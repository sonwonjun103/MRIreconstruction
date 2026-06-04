"""Undersampling masks and SSDU train/loss mask splitting.

Two distinct things live here:

1. The *acquisition* mask ``Omega`` -- which k-space columns were sampled
   (1-D Cartesian undersampling with a fully-sampled ACS region).

2. SSDU splitting -- partitioning the sampled points of ``Omega`` into a
   data-consistency set ``Theta`` (network input) and a loss set ``Lambda``
   (where the k-space loss is evaluated). Ported from the reference
   ``Paper/SSDU/utils/utils.py`` with a seedable RNG.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# acquisition mask Omega
# --------------------------------------------------------------------------- #
def undersampling_mask(shape_hw, acc_rate=4, acs_lines=24, mask_type="random",
                       rng: np.random.Generator | None = None,
                       vds_power: float = 3.0) -> np.ndarray:
    """1-D Cartesian undersampling mask, returned as a 2-D (H, W) float array.

    Columns (the W / phase-encode axis) are subsampled; every row keeps the same
    column pattern. A central block of ``acs_lines`` columns is always fully
    sampled (the autocalibration region). The remaining columns are picked by a
    probability density set by ``mask_type``:

      * ``random``    : uniform  (every outer column equally likely)
      * ``gaussian1d``: variable density, prob ∝ exp(-4|k|)        (center-weighted)
      * ``vds``       : variable density, prob ∝ (1-|k|)**vds_power (polynomial;
                        larger power -> more concentrated near the centre)
    """
    H, W = shape_hw
    if rng is None:
        rng = np.random.default_rng()

    c0 = W // 2 - acs_lines // 2
    c1 = c0 + acs_lines
    mask_1d = np.zeros(W, dtype=np.float32)
    mask_1d[c0:c1] = 1.0

    n_keep = max(int(round(W / acc_rate)), acs_lines)
    n_rand = max(n_keep - acs_lines, 0)

    outer = np.setdiff1d(np.arange(W), np.arange(c0, c1))
    if n_rand > 0:
        grid = np.linspace(-1, 1, W)
        if mask_type == "gaussian1d":
            pdf = np.exp(-4.0 * np.abs(grid))                # center-weighted (Laplacian)
        elif mask_type == "vds":
            pdf = np.clip(1.0 - np.abs(grid), 0, 1) ** vds_power   # polynomial VD
        else:                                                # "random" = uniform
            pdf = np.ones(W, dtype=np.float64)
        pdf[c0:c1] = 0.0
        pdf = pdf[outer]
        pdf /= pdf.sum()
        picks = rng.choice(outer, size=n_rand, replace=False, p=pdf)
        mask_1d[picks] = 1.0

    return np.broadcast_to(mask_1d[None, :], (H, W)).astype(np.float32).copy()


# --------------------------------------------------------------------------- #
# SSDU split: Omega -> (Theta train, Lambda loss)
# --------------------------------------------------------------------------- #
class SSDUMaskSplitter:
    """Split a sampling mask into disjoint train / loss masks.

    Parameters
    ----------
    rho : fraction of the sampled points placed in the loss mask ``Lambda``.
    small_acs_block : a tiny central block kept out of the loss set so the
        DC operator always sees the very centre of k-space.
    """

    def __init__(self, rho=0.4, small_acs_block=(4, 4)):
        self.rho = rho
        self.small_acs_block = small_acs_block

    # -- helpers ---------------------------------------------------------- #
    @staticmethod
    def _norm(tensor, axes=(0, 1, 2)):
        for axis in axes:
            tensor = np.linalg.norm(tensor, axis=axis, keepdims=True)
        return tensor

    def _find_center_ind(self, kspace, axes=(1, 2, 3)):
        center_locs = self._norm(kspace, axes=axes).squeeze()
        return int(np.argsort(center_locs)[-1])

    @staticmethod
    def _flatten2d(ind, shape):
        arr = np.zeros(int(np.prod(shape)))
        arr[ind] = 1
        ind_nd = np.nonzero(np.reshape(arr, shape))
        return [list(i) for i in ind_nd]

    # -- splitters -------------------------------------------------------- #
    def gaussian(self, input_data, input_mask, std_scale=4.0,
                 rng: np.random.Generator | None = None):
        """Gaussian-weighted selection of loss points (data laid out H,W,C)."""
        if rng is None:
            rng = np.random.default_rng()
        nrow, ncol = input_data.shape[0], input_data.shape[1]
        cx = self._find_center_ind(input_data, axes=(1, 2))
        cy = self._find_center_ind(input_data, axes=(0, 2))

        temp = np.copy(input_mask)
        temp[cx - self.small_acs_block[0] // 2: cx + self.small_acs_block[0] // 2,
             cy - self.small_acs_block[1] // 2: cy + self.small_acs_block[1] // 2] = 0

        loss_mask = np.zeros_like(input_mask)
        count, target = 0, int(np.ceil(np.sum(input_mask) * self.rho))
        while count < target:
            ix = int(np.round(rng.normal(cx, (nrow - 1) / std_scale)))
            iy = int(np.round(rng.normal(cy, (ncol - 1) / std_scale)))
            if 0 <= ix < nrow and 0 <= iy < ncol and temp[ix, iy] == 1 and loss_mask[ix, iy] != 1:
                loss_mask[ix, iy] = 1
                count += 1

        trn_mask = input_mask - loss_mask
        return trn_mask.astype(np.float32), loss_mask.astype(np.float32)

    def uniform(self, input_data, input_mask, rng: np.random.Generator | None = None):
        """Uniformly random selection of loss points (data laid out H,W,C)."""
        if rng is None:
            rng = np.random.default_rng()
        nrow, ncol = input_data.shape[0], input_data.shape[1]
        cx = self._find_center_ind(input_data, axes=(1, 2))
        cy = self._find_center_ind(input_data, axes=(0, 2))

        temp = np.copy(input_mask)
        temp[cx - self.small_acs_block[0] // 2: cx + self.small_acs_block[0] // 2,
             cy - self.small_acs_block[1] // 2: cy + self.small_acs_block[1] // 2] = 0

        pr = temp.flatten()
        ind = rng.choice(np.arange(nrow * ncol),
                         size=int(np.count_nonzero(pr) * self.rho),
                         replace=False, p=pr / pr.sum())
        ix, iy = self._flatten2d(ind, (nrow, ncol))

        loss_mask = np.zeros_like(input_mask)
        loss_mask[ix, iy] = 1
        trn_mask = input_mask - loss_mask
        return trn_mask.astype(np.float32), loss_mask.astype(np.float32)

    def split(self, input_data, input_mask, method="Gaussian_selection",
              rng: np.random.Generator | None = None):
        if method == "uniform_selection":
            return self.uniform(input_data, input_mask, rng=rng)
        return self.gaussian(input_data, input_mask, rng=rng)
