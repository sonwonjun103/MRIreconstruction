#!/usr/bin/env python
"""Legacy unified entry point for the mrrecon toolkit (kept for back-compat).

PREFER the per-method folder entry points -- each shows ONLY its own flags:

    python -m mrrecon.supervised      --net unet   --tissue knee --run_name unet_knee
    python -m mrrecon.supervised      --net dccnn  --tissue knee --backbone unet
    python -m mrrecon.supervised      --net varnet --tissue knee
    python -m mrrecon.self_supervised --algo ssdu  --tissue knee [--model mymodel]
    python -m mrrecon.zero_shot       --algo zsssl --tissue knee --split test
    python -m mrrecon.zero_shot       --algo diffusion --tissue knee   # train prior
    python -m mrrecon.eval --method ssdu --run runs/ssdu_knee --tissue knee --split test

This file still works (same subcommands as before) and dispatches to the same
trainers, now living under mrrecon/{supervised,self_supervised,zero_shot,eval}/.
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
    from mrrecon.eval.evaluator import Evaluator
    Evaluator(cfg, method="sense", ckpt=None,
              split=args.split, save_figs=args.save_figs).evaluate()


def _run_supervised(cfg, args):
    if getattr(cfg, "use_dc", False):       # wrap --arch in a DCCNN (data consistency)
        cfg.cnn = cfg.arch
        cfg.varnet_official = False
        from mrrecon.supervised.dc import VarNetTrainer
        VarNetTrainer(cfg).train()
    else:
        from mrrecon.supervised.trainer import SupervisedTrainer
        SupervisedTrainer(cfg).train()


def _run_dccnn(cfg, args):
    cfg.varnet_official = False             # our Deep Cascade (pluggable backbone)
    from mrrecon.supervised.dc import VarNetTrainer
    VarNetTrainer(cfg).train()


def _run_varnet(cfg, args):
    cfg.varnet_official = True              # official fastMRI E2E-VarNet (learned SME)
    from mrrecon.supervised.dc import VarNetTrainer
    VarNetTrainer(cfg).train()


def _run_ssdu(cfg, args):
    from mrrecon.self_supervised.ssdu import SSDUTrainer
    SSDUTrainer(cfg).train()


def _run_zeroshot(cfg, args):
    from mrrecon.zero_shot.zsssl import ZeroShotTrainer
    ZeroShotTrainer(cfg, split=args.split).train() 


def _run_zeroshot_infer(cfg, args):
    from mrrecon.zero_shot.zsssl import ZeroShotTrainer
    ZeroShotTrainer(cfg, split=args.split).infer(args.ckpt)


def _run_sscu(cfg, args):
    from mrrecon.self_supervised.sscu import SSCUTrainer
    SSCUTrainer(cfg).train()


def _run_diffusion(cfg, args):
    from mrrecon.zero_shot.diffusion import DiffusionTrainer
    DiffusionTrainer(cfg).train()


def _run_eval(cfg, args):
    from mrrecon.eval.evaluator import Evaluator
    Evaluator(cfg, method=args.method, ckpt=args.ckpt,
              split=args.split, save_figs=args.save_figs).evaluate()


def _run_profile(cfg, args):
    from mrrecon.profiling import profile_methods
    profile_methods(cfg, methods=args.profile_methods, time_iters=args.profile_iters)


COMMANDS = {
    "sense": (_run_sense, "classical CG-SENSE baseline (no training)"),
    "supervised": (_run_supervised, "train the supervised U-Net baseline"),
    "dccnn": (_run_dccnn, "train Deep Cascade (DC + unet/swin/mamba backbone, RSS output)"),
    "varnet": (_run_varnet, "train official E2E-VarNet (learned SME, multi-coil, RSS output)"),
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

    from mrrecon.core.common import start_file_logging
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
