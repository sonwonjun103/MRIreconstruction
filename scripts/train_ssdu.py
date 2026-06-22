#!/usr/bin/env python
"""Train the self-supervised SSDU unrolled network across the dataset.

Example:
    python scripts/train_ssdu.py --tissue knee --epochs 50 \
        --run_name ssdu_knee --nb_unroll_blocks 10 --cg_iter 10
"""
import sys
import os
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from mrrecon.config import build_parser, config_from_args
from mrrecon.self_supervised.ssdu import SSDUTrainer


def main():
    args = build_parser("ssdu").parse_args()
    cfg = config_from_args(args)
    SSDUTrainer(cfg).train()


if __name__ == "__main__":
    main()
