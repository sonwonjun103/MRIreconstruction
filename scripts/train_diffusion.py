#!/usr/bin/env python
"""Train the unconditional diffusion prior (for zero-shot diffusion recon).

After training, reconstruct undersampled scans with:
    python scripts/evaluate.py --method diffusion --tissue knee --split test \
        --ckpt runs/<run>/last.pt --run_name <run>_eval

Example:
    python scripts/train_diffusion.py --tissue knee --epochs 100 \
        --diff_dim 64 --run_name diff_knee
"""
import sys
import os
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from mrrecon.config import build_parser, config_from_args
from mrrecon.engine.diffusion import DiffusionTrainer


def main():
    args = build_parser("diffusion").parse_args()
    cfg = config_from_args(args)
    DiffusionTrainer(cfg).train()


if __name__ == "__main__":
    main()
