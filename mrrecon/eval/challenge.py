"""Inference on the official fastMRI *test* (challenge) split.

The challenge set is **prospectively undersampled** (its k-space already has only
~1/acc of the phase lines acquired) and ships with **no** ``reconstruction_rss``
ground truth, so the usual RSS metrics cannot be computed.

What this module does, per slice:
  * read the stored (already undersampled) multi-coil k-space, ESPIRiT maps and
    the REAL acquisition mask;
  * reconstruct with the chosen method/model (parameters loaded from --ckpt/--run),
    using the stored mask for data consistency (NO extra retrospective masking);
  * build a SENSE-based reference and score the reconstruction against it.

About the reference (important):
  Plain SENSE coil-combination of undersampled k-space is just the *aliased
  zero-filled* image -- it is the network input, not a ground truth. The only
  de-aliased SENSE-based reference obtainable from undersampled data is the
  classical **CG-SENSE** parallel-imaging reconstruction. We therefore score
  ``recon`` against the CG-SENSE image and label the metrics accordingly
  ("vs CG-SENSE reference"), NOT as absolute ground-truth quality.

Reuses the model-loading and figure helpers of :class:`Evaluator`.
"""

from __future__ import annotations

import os
import time

import h5py as h5
import numpy as np

from ..data.loaders import list_slice_files
from ..metrics import all_metrics, match_scale as _match_scale
from ..core.common import center_crop, save_json
from ..core.inference import recon_sense
from .evaluator import Evaluator


class ChallengeInference(Evaluator):
    """Inference + CG-SENSE-referenced metrics on the prospectively-undersampled
    fastMRI test (challenge) split. ``method`` / ``ckpt`` are the same as for the
    normal evaluator, so any trained model is loaded by its parameters."""

    def __init__(self, cfg, method, ckpt, split="test_challenge", save_figs=False,
                 reference="cgsense"):
        super().__init__(cfg, method, ckpt, split=split, save_figs=save_figs)
        self.reference = reference            # "cgsense" (de-aliased) | "zerofilled"

    def _read_challenge(self, path):
        """Native-size read (NO image crop -- the phase axis is undersampled, so
        cropping it would alias). Returns kspace (C,H,W), sens (C,H,W), mask (H,W)."""
        with h5.File(path, "r") as f:
            kspace = np.ascontiguousarray(f["kspace"][:])
            sens = np.ascontiguousarray(f["sens_map"][:])
            mask_1d = f["mask"][:].astype(np.float32) if "mask" in f else None
        H, W = kspace.shape[1:]
        if mask_1d is None:                   # fall back: infer from acquired columns
            mask_1d = (np.abs(kspace).sum((0, 1)) > 0).astype(np.float32)
        omega = np.broadcast_to(mask_1d[None, :], (H, W)).astype(np.float32).copy()
        return kspace, sens, omega

    def run(self):
        cfg = self.cfg
        model = self._build_model()
        files = list_slice_files(cfg.data_root, cfg.tissue, self.split,
                                 cfg.max_slices, cfg.modality, cfg.full_subject,
                                 drop_edge=cfg.drop_edge_slices)
        rdir = os.path.join(cfg.out_dir, cfg.run_name, f"acc_challenge")
        os.makedirs(rdir, exist_ok=True)
        print(f"challenge inference -> {os.path.abspath(rdir)}")
        print(f"  reference = {self.reference}  ({'CG-SENSE de-aliased' if self.reference=='cgsense' else 'zero-filled SENSE (aliased)'}); "
              f"metrics are vs this reference, NOT ground truth")

        per_slice = []
        agg = {"ssim": [], "psnr": [], "nmse": [], "nmae": []}
        zf_agg = {"ssim": [], "psnr": [], "nmse": [], "nmae": []}
        figbase = os.path.join(rdir, "result", self._method_tag())
        t0 = time.time()
        for i, fpath in enumerate(files):
            kspace, sens, omega = self._read_challenge(fpath)

            # reconstruction with the chosen method, using the REAL stored mask
            _, zf, recon = self.recon_fn(model, kspace, sens, omega, self.device)

            # SENSE-based reference from the same undersampled data
            if self.reference == "cgsense":
                _, _, ref = recon_sense(kspace, sens, omega, self.device,
                                        cfg.sense_lam, cfg.sense_cg_iter)
            else:                              # zero-filled SENSE coil-combination
                ref = zf

            # score on the central 320x320 (fastMRI convention), scale-matched
            ref_c = center_crop(np.abs(ref), 320)
            rec_c = center_crop(np.abs(recon), 320)
            zf_c = center_crop(np.abs(zf), 320)
            m = all_metrics(ref_c, _match_scale(ref_c, rec_c))
            mzf = all_metrics(ref_c, _match_scale(ref_c, zf_c))
            for k in agg:
                agg[k].append(m[k]); zf_agg[k].append(mzf[k])
            per_slice.append({"slice": i, "recon": m, "zero_filled": mzf,
                              "file": os.path.basename(fpath)})

            if self.save_figs:
                refname = "CG-SENSE" if self.reference == "cgsense" else "SENSE(zf)"
                self._save_comparison(os.path.join(figbase, "vs_" + self.reference), i,
                                      ref_c, zf_c, _match_scale(ref_c, rec_c),
                                      m, mzf, refname,
                                      mask=center_crop(omega, 320))
            if i % 20 == 0:
                print(f"  [{i}/{len(files)}] vs {self.reference} "
                      f"ssim={m['ssim']:.4f} psnr={m['psnr']:.3f}")

        secs = time.time() - t0
        summary = {k: float(np.nanmean(v)) for k, v in agg.items()}
        zf_summary = {k: float(np.nanmean(v)) for k, v in zf_agg.items()}
        print(f"\n=== test_challenge (prospective undersample, NO GT) | "
              f"recon vs {self.reference} reference ===")
        print(f"zero-filled : SSIM={zf_summary['ssim']:.4f} PSNR={zf_summary['psnr']:.3f} "
              f"NMSE={zf_summary['nmse']:.5f} NMAE={zf_summary['nmae']:.5f}")
        print(f"recon       : SSIM={summary['ssim']:.4f} PSNR={summary['psnr']:.3f} "
              f"NMSE={summary['nmse']:.5f} NMAE={summary['nmae']:.5f}")
        print(f"({len(files)} slices, {secs:.1f}s)")
        out = {"split": "test_challenge", "method": self.method,
               "model_tag": self._method_tag(), "reference": self.reference,
               "n_slices": len(files), "recon": summary, "zero_filled": zf_summary,
               "note": "metrics are vs the SENSE-based reference, NOT ground truth "
                       "(the challenge set is prospectively undersampled with no GT)",
               "per_slice": per_slice}
        save_json(out, os.path.join(rdir, f"challenge_{self._method_tag()}_{self.reference}.json"))
        return out
