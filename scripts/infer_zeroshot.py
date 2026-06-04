#!/usr/bin/env python
"""Reconstruct a zero-shot-fitted scan from a saved checkpoint (no training).

Rebuilds the same slice + acquisition mask (config + --seed are deterministic),
loads the checkpoint, and writes recon.npy / reference.npy / result.json.
Pass the SAME flags used during `train_zeroshot.py` (tissue, split, zs_slice,
seed, model, acc_rate, ...) plus --ckpt.

Example:
    python scripts/infer_zeroshot.py --tissue knee --split test --model mymodel \
        --ckpt runs/zs_knee/best.pt --run_name zs_knee_infer
"""
import sys
import os
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from mrrecon.config import build_parser, config_from_args
from mrrecon.engine.zeroshot import ZeroShotTrainer


def main():
    args = build_parser("zeroshot-infer").parse_args()
    cfg = config_from_args(args)
    ZeroShotTrainer(cfg, split=args.split).infer(args.ckpt)


if __name__ == "__main__":
    main()
