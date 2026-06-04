"""SSCU self-supervised training -- STUB (not implemented yet).

Placeholder for the SSCU method (self-supervised, alongside SSDU). The exact
algorithm/definition is still to be decided, so this file only sketches the
interface. Fill in ``_build`` and ``train`` when the method is finalised.

It can reuse the existing machinery:
  * data         : mrrecon.data.datasets (e.g. a variant of SSDUDataset)
  * model        : mrrecon.models.build_unrolled (ssdu/mymodel/mamba)
  * loss / DC    : mrrecon.losses.MixL1L2Loss, mrrecon.models.data_consistency
  * validation   : mrrecon.engine.inference.recon_unrolled
"""

from __future__ import annotations

from .common import get_device


class SSCUTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = get_device(cfg.device)

    def _build(self):
        raise NotImplementedError("SSCU: define the dataset/split scheme here.")

    def train(self):
        raise NotImplementedError(
            "SSCU is not implemented yet. This is a stub — fill in "
            "mrrecon/engine/sscu.py (SSCUTrainer) with the SSCU training loop.")
