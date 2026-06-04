# mrrecon — zero-shot / supervised / self-supervised MRI reconstruction (fastMRI)

A small, reusable toolkit for accelerated multi-coil MRI reconstruction on the
fastMRI dataset. Three methods share one SENSE-based data pipeline and a common
metrics module, so they are directly comparable:

| method        | supervision        | model                       | trains on        |
|---------------|--------------------|-----------------------------|------------------|
| `sense`       | none (classical)   | CG-SENSE (no learning)      | nothing          |
| `supervised`  | fully-sampled label| U-Net                       | whole dataset    |
| `ssdu`        | self-supervised    | unrolled + CG DC            | whole dataset    |
| `zeroshot`    | self-supervised    | unrolled + CG DC            | a **single** scan|
| `diffusion`   | unsupervised prior | DDPM U-Net + DC sampling    | prior: dataset; **recon: zero-shot** |

Each learned unrolled method can pick its regulariser via `--model`:
`ssdu` (shallow ResNet, the original), **`mymodel`** (multi-scale U-Net,
recommended — see *Recommended model*), or **`mamba`** (Mamba-ViT
state-space regulariser — see *Mamba model*). All three are drop-in: same
unrolled scheme, CG data consistency, and self-supervised loss.

All methods reconstruct in the **multi-coil SENSE image domain** and report
**SSIM, PSNR, NMSE, NMAE**.

## Data

Built by `MakeDataset.py` (in this `code/` folder) from raw fastMRI files:
**one HDF5 file per slice** under `--data_root` (default `/mnt/d/research/MRRecon/data`):

```
{data_root}/{tissue[_full]}/{train,val,test}/{subject}_{slice:03d}.h5
    kspace   (C, H, W) complex64   -- multi-coil k-space
    sens_map (C, H, W) complex64   -- ESPIRiT sensitivity maps (BART ecalib)
    rss      (h, w)    float32     -- reference / preview (optional)
```

The **split is the folder** (`train` / `val` / `test`); the **zero-shot set is
the test split**. Slices are loaded lazily (one file at a time), so the multi-GB
volumes never sit in RAM.

Two dataset variants (subject-level split, modality-stratified for brain):
- **central** (`--mode central`): the 3 central slices per subject → `{tissue}/`
- **full** (`--mode full`): ALL slices per subject → `{tissue}_full/`

Training/eval select the variant with `--full_subject` (full) or omit it (central).

```bash
python MakeDataset.py --tissue both --mode both --save_png   # build both variants
```

`--save_png` also writes a magnitude PNG preview per slice under
`{split}/preview/` (data itself is always .h5 — PNG cannot store complex
multi-coil k-space). A `manifest.json` records the subject membership of each
split for reproducibility/auditing.

### Sensitivity maps (ESPIRiT)

`MakeDataset.py` computes the maps with BART once and stores them in each slice
file. To compute maps from raw k-space yourself (e.g. a new scan):

```python
from mrrecon.data.sensitivity import estimate_sensitivity
maps = estimate_sensitivity(kspace_slice, method="bart")   # (C,H,W) -> (C,H,W)
```

Backends (`mrrecon/data/sensitivity.py`): `bart` (ESPIRiT via BART `ecalib`,
**recommended** — reproduces the stored maps exactly), `sigpy`
(`sigpy.mri.app.EspiritCalib`, if installed), `numpy` (dependency-free but
**approximate**). `sens_maps_volume(vol)` runs a whole `(S,C,H,W)` volume.

`--tissue {knee,brain}` is **required** (no default). Coil layout is `(C,H,W)`;
masks broadcast over coils. FFTs are centered & orthonormal everywhere, so the
numpy pre-processing and the torch data-consistency operator match exactly.

## Layout

```
mrrecon/
  config.py              Config dataclass + argparse builders
  data/
    transforms.py        centered orthonormal FFT, SENSE E / E^H, c2r/r2c
    masks.py             acquisition mask Omega + SSDU split (Theta/Lambda)
    loaders.py           list/read per-slice .h5 from {tissue}/{split}/
    datasets.py          SupervisedDataset / SSDUDataset / ZeroShotDataset
  models/
    unet.py              fastMRI-style U-Net (2-channel complex)
    resnet.py            ResNet denoiser (unrolled regulariser)
    data_consistency.py  SENSE encoder + conjugate-gradient DC block
    unrolled.py          UnrolledSSDU (ResNet denoiser <-> CG, K iterations)
    mymodel.py           UnrolledUNet -- recommended: U-Net regulariser <-> CG
    mamba.py             UnrolledMamba -- Mamba-ViT regulariser <-> CG
    diffusion.py         DiffusionUNet + GaussianDiffusion (DC-guided DDIM)
    sense.py             classical CG-SENSE recon (no learning)
    __init__.py          build_unrolled(cfg) factory (ssdu | mymodel | mamba)
  losses.py              normalised L1+L2 k-space loss (SSDU)
  metrics.py             ssim / psnr / nmse / nmae
  engine/
    supervised.py  ssdu.py  zeroshot.py   trainers
    inference.py   evaluator.py            single-slice recon + evaluation
scripts/
  train_supervised.py  train_ssdu.py  train_zeroshot.py  evaluate.py
```

## Quickstart

```bash
pip install -r requirements.txt   # torch already present

# 0) classical CG-SENSE baseline (no training)
python scripts/recon_sense.py --tissue knee --split test --run_name sense_knee

# 1) supervised U-Net baseline
python scripts/train_supervised.py --tissue knee --epochs 50 --run_name unet_knee

# 2) SSDU self-supervised (across dataset)
python scripts/train_ssdu.py --tissue knee --epochs 50 --run_name ssdu_knee
#    ...or with the recommended U-Net regulariser:
python scripts/train_ssdu.py --tissue knee --epochs 50 --model mymodel --run_name my_knee

# 3) zero-shot single-scan (ZS-SSL, early stopping)
python scripts/train_zeroshot.py --tissue knee --split test --zs_slice -1 \
    --epochs 300 --zs_patience 25 --lr 5e-4 --run_name zs_knee
#    zero-shot with the recommended model:
python scripts/train_zeroshot.py --tissue knee --split test --model mymodel \
    --epochs 300 --zs_patience 25 --lr 5e-4 --run_name zs_my_knee

# evaluate any checkpoint (pass the SAME --model it was trained with)
python scripts/evaluate.py --method ssdu --tissue knee --split test \
    --ckpt runs/ssdu_knee/best.pt --run_name ssdu_knee_eval --save_figs
```

Outputs land in `runs/<run_name>/`: `best.pt`, `last.pt`, `config.json`,
`history.json`, eval JSON, and (zero-shot) `recon.npy` / `reference.npy`.

### Smoke test

Add `--max_slices 4 --epochs 1` to any trainer to exercise the full pipeline on
a tiny subset in seconds before launching a real run.

## Key flags

- **acceleration**: `--acc_rate 4 --acs_lines 24 --mask_type {random,gaussian1d}`
- **unrolled**: `--nb_unroll_blocks 10 --cg_iter 10 --res_blocks 15 --mu 0.05`
- **SSDU split**: `--divide_method {Gaussian_selection,uniform_selection} --rho 0.4`
- **U-Net**: `--unet_chans 32 --unet_pools 4`
- **zero-shot**: `--zs_val_rho 0.2 --zs_num_splits 1 --zs_patience 25 --zs_slice -1`

## Method notes

- **SENSE** (classical parallel imaging): solves
  `min_x ||M F S x - y||^2 + lam||x||^2` by conjugate gradient — no learning,
  no training data. Strongest with *equispaced* sampling; under `random` masks
  it amplifies noise (g-factor), so raise `--sense_lam` or use `--mask_type`
  patterns closer to regular for a fair classical baseline.
- **Supervised**: zero-filled SENSE image → U-Net → image, L1 loss vs the
  fully-sampled SENSE image.
- **SSDU** (Yaman et al., MRM 2020): split Omega into Theta (data consistency)
  and Lambda (loss); the unrolled net alternates a ResNet denoiser with a CG
  SENSE data-consistency block; loss is normalised L1+L2 in k-space at Lambda.
- **Zero-shot / ZS-SSL** (Yaman et al., *ICLR 2022*, "Zero-Shot Self-Supervised
  Learning for MRI Reconstruction"): the same unrolled net fit to **one** scan.
  Omega is partitioned into three disjoint sets — Theta (DC) and Lambda (loss)
  for self-supervision, and a fixed validation set Gamma for **early stopping**.
  Each epoch re-draws `--zs_num_splits` (paper: 25) fresh (Theta, Lambda) splits
  of `Omega \ Gamma`; validation uses `Omega \ Gamma` in DC and measures the
  k-space loss on Gamma; final inference uses the **full Omega** in DC. Faithful
  to the reference in `../Paper/ZeroShot`.

## Recommended model (`mymodel`)

`mymodel.UnrolledUNet` keeps the SSDU/ZS-SSL physics (CG SENSE data consistency,
the same self-supervised k-space loss) but replaces the shallow **ResNet**
regulariser with a multi-scale **U-Net**, shared across the K unrolled
iterations:

```
for k in 1..K:                              # K = --nb_unroll_blocks
    z_k = UNet(x_{k-1})                      # multi-scale de-aliasing
    x_k = CG-solve (E^H M_Theta E + mu I) x = x_in + mu z_k   # data consistency
```

Why it should beat the baselines: Cartesian undersampling causes *coherent,
large-scale* aliasing along the phase-encode axis. A ResNet's small receptive
field cannot "see" that global structure; a U-Net encoder/decoder can, which is
why U-Net regularisers (E2E-VarNet, MoDL-UNet) lead the fastMRI benchmark.
Weights are **shared across iterations**, so the parameter count stays small —
important for zero-shot, where we fit a single scan and rely on early stopping.
It is a drop-in replacement: select with `--model mymodel` in `train_ssdu.py`,
`train_zeroshot.py` and `evaluate.py`. Tune capacity with `--mymodel_chans`
(default 32) and `--mymodel_pools` (default 3).

## Mamba model (`--model mamba`)

`mamba.UnrolledMamba` is the same unrolled scheme with a **Mamba-ViT**
regulariser: the image is patch-embedded into tokens, mixed by **bidirectional
selective state-space (Mamba)** blocks, and un-embedded with a residual
connection (final layer zero-initialised, so the denoiser starts as identity and
data consistency dominates early). Selective SSMs give a global receptive field
in **linear** time — Transformer-like long-range context without quadratic
attention — which suits the coherent long-range aliasing of Cartesian
undersampling. Size-robust: arbitrary H×W (640×368, 640×320) via padding and an
interpolated positional grid.

It is fully zero-shot capable — train and infer on one scan:

```bash
python main.py zeroshot --tissue knee --split test --model mamba \
    --mamba_patch 16 --mamba_dim 128 --mamba_depth 4 \
    --epochs 300 --zs_patience 25 --lr 5e-4 --run_name zs_mamba_knee
```

Knobs: `--mamba_dim 128 --mamba_depth 4 --mamba_patch 16 --mamba_dstate 16
--mamba_expand 2`.

> **Speed:** if the CUDA kernels `mamba_ssm` + `causal_conv1d` are installed they
> are used automatically (fast). Otherwise a pure-PyTorch selective-scan
> *fallback* runs — correct but slow (a Python loop over tokens). Without the
> kernels, keep `--mamba_patch` large (shorter sequence) and `--mamba_depth`
> small, or install: `pip install mamba-ssm causal-conv1d`.

## Zero-shot diffusion (`diffusion`)

Two stages (score-MRI / DPS recipe with multi-coil SENSE data consistency):

1. **Learn an unconditional prior** `p(x)` — a DDPM noise-predictor U-Net
   ([models/diffusion.py](mrrecon/models/diffusion.py)) trained on fully-sampled
   SENSE images. It never sees undersampling masks or paired data, so it is just
   an image prior (unsupervised density model).
2. **Zero-shot reconstruction** — for any accelerated scan, run DC-guided DDIM
   sampling: at each reverse step the prior's `x0` estimate is pulled toward the
   measured k-space by the same conjugate-gradient SENSE `dc_block` the unrolled
   nets use. No retraining for new scans/masks/acceleration → zero-shot.

```bash
# stage 1: train the prior (across dataset)
python main.py diffusion --tissue knee --epochs 200 --diff_dim 64 --run_name diff_knee

# stage 2: zero-shot reconstruction of undersampled scans
python main.py eval --method diffusion --tissue knee --split test \
    --ckpt runs/diff_knee/last.pt --diff_sampling_steps 100 --diff_dc_lam 1.0 \
    --run_name diff_eval --save_figs
```

Knobs: `--diff_dim 64 --diff_timesteps 1000 --diff_schedule cosine`
(training); `--diff_sampling_steps 100 --diff_dc_lam 1.0 --diff_dc_iter 5`
(reconstruction). Lower `--diff_dc_lam` trusts the measurements more; higher
trusts the prior. Stability: `x0` is clamped to [-1,1] each step (the prior's
data range) — required, or the `1/√ᾱ_t` term blows up at large t.

> The diffusion prior needs **real training** (e.g. 100–200 epochs on the full
> dataset) to reconstruct well — unlike the unrolled methods, it cannot lean on
> data consistency *during* training, so a lightly-trained prior gives poor
> recon. The sampling/DC machinery itself is verified end-to-end.

## Supervised vs self-supervised vs zero-shot — what actually differs

The **forward model and inference are identical** for all learned methods:
zero-filled SENSE image in → network → reconstructed image, with metrics on the
magnitude. What differs is *the training signal and the data each is fit on*:

| aspect            | supervised            | SSDU (self-sup.)            | zero-shot (ZS-SSL)                |
|-------------------|-----------------------|-----------------------------|-----------------------------------|
| labels            | fully-sampled image   | **none**                    | **none**                          |
| loss domain       | image (L1)            | k-space at Lambda (L1+L2)   | k-space at Lambda (L1+L2)         |
| mask use          | one Omega per slice   | split Omega → Theta/Lambda  | split Omega → Theta/Lambda/Gamma  |
| training set      | many slices           | many slices                 | **one** slice                     |
| stopping          | fixed epochs / val SSIM | fixed epochs / val SSIM   | **early stop** on Gamma k-space loss |
| inference DC mask | Omega                 | Omega                       | Omega (full)                      |

So in code: the **models** (`UNet`, `UnrolledSSDU`, `UnrolledUNet`) and the
**single-slice inference** (`engine/inference.py`) are shared; only the
**datasets** (`SupervisedDataset` / `SSDUDataset` / `ZeroShotDataset`) and the
**trainer loops** (`engine/supervised.py` / `ssdu.py` / `zeroshot.py`) differ.
Zero-shot is essentially SSDU restricted to one slice plus the Gamma-based
early-stopping loop — both train and infer on the *same* scan.

> **Zero-shot inference is built into training.** `main.py zeroshot` fits the
> scan and then *immediately* reconstructs it (saving `recon.npy`,
> `reference.npy`, `result.json`) — there is no separate inference step in the
> normal flow. To re-reconstruct from a saved checkpoint *without* re-fitting
> (e.g. regenerate outputs/figures), use `zeroshot-infer`, which rebuilds the
> identical slice + mask (deterministic from the config + `--seed`) and runs the
> full-Omega reconstruction:
>
> ```bash
> python main.py zeroshot-infer --tissue knee --split test --model mymodel \
>     --seed 1234 --ckpt runs/zs_knee/best.pt --run_name zs_knee_infer
> ```
> Pass the **same flags + `--seed`** used at training so the scan/mask match.

## Profiling: parameters / FLOPs / training time

```bash
python main.py profile --tissue knee \
    --profile_methods supervised ssdu mymodel mamba diffusion
# pass the model flags you'll train with so numbers match (e.g. --nb_unroll_blocks,
# --cg_iter, --mamba_depth, --diff_dim). Standalone: scripts/profile_models.py
```

Prints a table of #params, GFLOPs/forward (torch `FlopCounterMode`), measured
seconds/training-step, and a per-epoch estimate. Example (knee 640×368, one GPU,
`--nb_unroll_blocks 5 --cg_iter 6`):

| method     | params | GFLOPs/fwd | step (s) | epoch (s) |
|------------|--------|-----------:|---------:|----------:|
| sense      | n/a    | n/a        | n/a      | n/a       |
| supervised | 7.76M  |      86.6  |   0.028  |     6.6   |
| ssdu       | 1.15M  |    2696.9  |   0.884  |   207.8   |
| mymodel    | 1.92M  |     330.7  |   0.487  |   114.5   |
| diffusion  | 2.80M  |     215.6  |   0.086  |    20.2   |

Caveats: FLOPs exclude FFT / CG-data-consistency ops (lower bound, dominated by
the counted denoiser × unrolled iterations); `epoch (s)` = step × steps/epoch
(dataset: `ceil(train_slices/batch)`; zero-shot: `--zs_num_splits`). `mamba` is
slow without the CUDA kernels (pure-PyTorch fallback) — see *Mamba model*.

## Key flags (additional)

- **model choice** (ssdu/zeroshot/eval): `--model {ssdu,mymodel,mamba}`
- **mymodel**: `--mymodel_chans 32 --mymodel_pools 3`
- **mamba**: `--mamba_dim 128 --mamba_depth 4 --mamba_patch 16`
- **SENSE**: `--sense_lam 1e-2 --sense_cg_iter 30`
- **ZS-SSL**: `--zs_num_splits 25 --zs_val_rho 0.2 --zs_patience 25 --lr 5e-4`
```
