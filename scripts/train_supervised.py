#!/usr/bin/env python
"""Train the supervised U-Net baseline.

Example:
    python scripts/train_supervised.py --tissue knee --epochs 50 \
        --run_name unet_knee --acc_rate 4
"""
import sys
import os
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from mrrecon.config import build_parser, config_from_args
from mrrecon.supervised.trainer import SupervisedTrainer


def main():
    args = build_parser("supervised").parse_args()
    cfg = config_from_args(args)
    SupervisedTrainer(cfg).train()


if __name__ == "__main__":
    main()
