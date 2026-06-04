#!/usr/bin/env python
"""Unified entry point for the mrrecon toolkit.

One command runs every method via subcommands:

    python main.py sense       --tissue knee --split test --run_name sense_knee
    python main.py supervised  --tissue knee --epochs 50  --run_name unet_knee
    python main.py ssdu        --tissue knee --epochs 50  --run_name ssdu_knee [--model mymodel]
    python main.py zeroshot    --tissue knee --split test --model mymodel \
                               --epochs 300 --zs_patience 25 --lr 5e-4 --run_name zs_knee
    python main.py diffusion   --tissue knee --epochs 100 --run_name diff_knee   # train prior
    python main.py eval        --method diffusion --tissue knee --split test \
                               --ckpt runs/diff_knee/last.pt --run_name diff_eval # zero-shot recon
    python main.py eval        --method ssdu --tissue knee --split test \
                               --ckpt runs/ssdu_knee/best.pt --run_name ssdu_eval --save_figs

Run ``python main.py <subcommand> -h`` to see the flags for that subcommand, or
``python main.py -h`` for the list. Every subcommand shares the same flag names
as the standalone scripts in ``scripts/`` (this just dispatches to them).
"""

from __future__ import annotations

import os 
import sys
import argparse
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from mrrecon.config import configure_parser, config_from_args


# command -> (configure key, runner). Runners import lazily so a single
# subcommand doesn't pay the import cost of all the others.
def _run_sense(cfg, args):
    from mrrecon.engine.evaluator import Evaluator
    Evaluator(cfg, method="sense", ckpt=None,
              split=args.split, save_figs=args.save_figs).evaluate()


def _run_supervised(cfg, args):
    from mrrecon.engine.supervised import SupervisedTrainer
    SupervisedTrainer(cfg).train()


def _run_ssdu(cfg, args):
    from mrrecon.engine.ssdu import SSDUTrainer
    SSDUTrainer(cfg).train()


def _run_zeroshot(cfg, args):
    from mrrecon.engine.zeroshot import ZeroShotTrainer
    ZeroShotTrainer(cfg, split=args.split).train() 


def _run_zeroshot_infer(cfg, args):
    from mrrecon.engine.zeroshot import ZeroShotTrainer
    ZeroShotTrainer(cfg, split=args.split).infer(args.ckpt)


def _run_sscu(cfg, args):
    from mrrecon.engine.sscu import SSCUTrainer
    SSCUTrainer(cfg).train()


def _run_diffusion(cfg, args):
    from mrrecon.engine.diffusion import DiffusionTrainer
    DiffusionTrainer(cfg).train()


def _run_eval(cfg, args):
    from mrrecon.engine.evaluator import Evaluator
    Evaluator(cfg, method=args.method, ckpt=args.ckpt,
              split=args.split, save_figs=args.save_figs).evaluate()


def _run_profile(cfg, args):
    from mrrecon.profiling import profile_methods
    profile_methods(cfg, methods=args.profile_methods, time_iters=args.profile_iters)


COMMANDS = {
    "sense": (_run_sense, "classical CG-SENSE baseline (no training)"),
    "supervised": (_run_supervised, "train the supervised U-Net baseline"),
    "ssdu": (_run_ssdu, "train SSDU self-supervised across the dataset"),
    "sscu": (_run_sscu, "train SSCU self-supervised (STUB -- not implemented)"),
    "zeroshot": (_run_zeroshot, "zero-shot (ZS-SSL) self-supervised on one scan"),
    "zeroshot-infer": (_run_zeroshot_infer, "reconstruct a fitted scan from a zero-shot checkpoint (no training)"),
    "diffusion": (_run_diffusion, "train the diffusion prior (for zero-shot diffusion recon)"),
    "eval": (_run_eval, "evaluate a checkpoint (SSIM / PSNR / NMSE / NMAE)"),
    "profile": (_run_profile, "report params / FLOPs / training time per model"),
}


def main():
    parser = argparse.ArgumentParser(
        prog="main.py", description="mrrecon: MRI reconstruction toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    for name, (_, help_text) in COMMANDS.items():
        sp = sub.add_parser(name, help=help_text,
                            description=f"mrrecon {name}: {help_text}")
        configure_parser(sp, name)

    args = parser.parse_args()
    cfg = config_from_args(args)
    runner = COMMANDS[args.command][0]

    from mrrecon.engine.common import start_file_logging
    log_path = start_file_logging(cfg)      # tee stdout/stderr -> runs/<run_name>/log.txt
    mod = f" modality={cfg.modality}" if cfg.modality else ""
    print(f">>> mrrecon {args.command} | tissue={cfg.tissue}{mod} acc={cfg.acc_rate} "
          f"run={cfg.run_name} | log -> {log_path}")
    try:
        runner(cfg, args)
    except Exception:
        import traceback
        traceback.print_exc()              # also captured in the log file
        raise


if __name__ == "__main__":
    main()
