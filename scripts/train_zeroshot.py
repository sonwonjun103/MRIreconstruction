#!/usr/bin/env python
"""Zero-shot self-supervised reconstruction (ZS-SSL) on a single scan.

Fits one slice of the chosen split with early stopping on a held-out k-space
validation set.

Example:
    python scripts/train_zeroshot.py --tissue knee --split test --zs_slice -1 \
        --run_name zs_knee --epochs 300 --zs_patience 25 --zs_num_splits 1
"""
import sys
import os
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from mrrecon.config import build_parser, config_from_args
from mrrecon.zero_shot.zsssl import ZeroShotTrainer


def main():
    args = build_parser("zeroshot").parse_args()
    cfg = config_from_args(args)
    ZeroShotTrainer(cfg, split=args.split).train()


if __name__ == "__main__":
    main()
