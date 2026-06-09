"""Render a paper-style architecture figure for ZS-MambaRecon (mymodel).

Three panels:
  (a) the unrolled MoDL/SSDU pipeline (input k-space -> reconstructed image),
  (b) the hierarchical Mamba U-Net regulariser (the per-iteration denoiser),
  (c) the Residual State-Space Block (RSSB) and its SS2D four-directional scan.

Output: docs/architecture.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---- palette ---------------------------------------------------------------
C_DATA  = "#cfe8ff"   # data / tensors
C_PHYS  = "#ffe0b3"   # physics ops (E, CG-DC)
C_NET   = "#d8f0d0"   # learned conv stages
C_MAMBA = "#e7d4f7"   # Mamba / SS2D
C_ATTN  = "#ffd6e0"   # attention / mlp
C_LOSS  = "#f6f6a8"   # loss
EC = "#333333"


def box(ax, x, y, w, h, text, fc, fs=10, ec=EC, lw=1.3, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.012,rounding_size=0.05",
                 fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(ax, p1, p2, color=EC, lw=1.6, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14,
                 lw=lw, color=color, ls=ls,
                 connectionstyle=f"arc3,rad={rad}"))


fig = plt.figure(figsize=(15, 17))
gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.15, 1.25], hspace=0.16)

# =========================================================================== #
# (a) unrolled pipeline
# =========================================================================== #
ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 16); ax.set_ylim(0, 5.2); ax.axis("off")
ax.text(0.1, 4.9, "(a)  ZS-MambaRecon — physics-guided unrolled pipeline",
        fontsize=14, fontweight="bold")

box(ax, 0.2, 2.3, 1.9, 1.1,
    "Undersampled\nmulti-coil\nk-space  $y$\n(mask $\\Omega/\\Theta$, $S$)", C_DATA, 9)
box(ax, 2.6, 2.5, 1.3, 0.7, "$E^{H}$\n(SENSE)", C_PHYS, 9)
box(ax, 4.3, 2.3, 1.8, 1.1, "zero-filled\n$x_{in}=E^{H}y$\n(B,2,H,W)", C_DATA, 9)

# the recurrent unrolled cell
cx = 6.7
box(ax, cx, 3.2, 3.0, 1.1,
    "Denoiser  $D_\\theta(x,\\,t_k)$\nMamba U-Net (panel b)\nshared weights, FiLM($t_k$)", C_NET, 9)
box(ax, cx + 0.45, 1.95, 2.1, 0.75, "$rhs = x_{in} + \\mu\\, z$", C_DATA, 9)
box(ax, cx, 0.7, 3.0, 0.95,
    "CG data consistency\n$(E^{H}M_\\Theta E+\\mu I)x=rhs$", C_PHYS, 9)

# dashed box around the cell + xK loop label
ax.add_patch(FancyBboxPatch((cx - 0.45, 0.45), 4.0, 4.15,
             boxstyle="round,pad=0.02", fc="none", ec="#9b59b6", lw=1.6, ls="--"))
ax.text(cx + 1.55, 4.78, "unrolled cell  ×K", fontsize=10, color="#9b59b6",
        ha="center", fontweight="bold")

box(ax, 11.4, 2.3, 1.9, 1.1, "recon image\n$x_K$\n$\\to |x_K|$", C_DATA, 9)
box(ax, 13.7, 3.0, 2.1, 1.5,
    "to k-space @ $\\Lambda$\n$\\|M_\\Lambda(E x_K - y)\\|$\nSSDU / ZS-SSL\nself-sup. loss", C_LOSS, 8.5)
box(ax, 13.7, 1.2, 2.1, 1.2, "SSIM / PSNR\nNMSE / NMAE\n(monitor only)", C_DATA, 8.5)

# arrows
arrow(ax, (2.1, 2.85), (2.6, 2.85))
arrow(ax, (3.9, 2.85), (4.3, 2.85))
arrow(ax, (6.1, 2.95), (cx + 1.0, 3.2))               # x_in -> denoiser (and as x_0)
arrow(ax, (cx + 1.5, 3.2), (cx + 1.5, 2.7))           # denoiser z -> rhs
arrow(ax, (cx + 1.5, 1.95), (cx + 1.5, 1.65))         # rhs -> CG
# recurrence: CG output back up to denoiser
arrow(ax, (cx + 3.0, 1.15), (cx + 3.35, 1.15), rad=0)
ax.add_patch(FancyArrowPatch((cx + 3.35, 1.15), (cx + 3.35, 3.75),
             arrowstyle="-", lw=1.6, color="#9b59b6", ls="--"))
arrow(ax, (cx + 3.35, 3.75), (cx + 3.0, 3.75), color="#9b59b6", ls="--")
ax.text(cx + 3.55, 2.4, "$x_k$", fontsize=10, color="#9b59b6")
# x_in skip into rhs
arrow(ax, (5.6, 2.3), (cx + 0.55, 2.32), rad=-0.25, color="#1f6feb", ls="--")
ax.text(6.0, 1.7, "$x_{in}$ skip", fontsize=8.5, color="#1f6feb")
# final
arrow(ax, (cx + 3.0, 1.15), (11.4, 2.3), rad=-0.2)
arrow(ax, (13.3, 3.0), (13.7, 3.4))
arrow(ax, (13.3, 2.5), (13.7, 1.9))

# =========================================================================== #
# (b) Mamba U-Net denoiser
# =========================================================================== #
ax = fig.add_subplot(gs[1]); ax.set_xlim(0, 16); ax.set_ylim(0, 5.6); ax.axis("off")
ax.text(0.1, 5.25, "(b)  Denoiser $D_\\theta$ — hierarchical Mamba U-Net "
        "(conv at fine scales, Mamba at coarse scales)",
        fontsize=14, fontweight="bold")

box(ax, 0.2, 2.6, 1.5, 1.0, "$x$\n(B,2,H,W)", C_DATA, 9)
box(ax, 1.9, 2.6, 1.3, 1.0, "standardize\n+ conv stem\n$2\\!\\to\\!C$", C_NET, 8.5)
box(ax, 3.4, 4.2, 1.5, 0.8, "FiLM($t_k$)\nscale/shift", C_ATTN, 8.5)
arrow(ax, (4.15, 4.2), (4.15, 3.6))

# encoder (descending), decoder (ascending) — U shape
enc = [("ConvResBlock\nC, H", C_NET, 2.9),
       ("ConvResBlock\n2C, H/2", C_NET, 1.9),
       ("ConvResBlock\n4C, H/4", C_NET, 0.9)]
xs = 3.4
for i, (t, c, y) in enumerate(enc):
    box(ax, xs + i * 1.55, y, 1.45, 0.8, t, c, 8.2)
# bottleneck (Mamba)
box(ax, xs + 3 * 1.55, 0.0, 1.7, 0.85, "RSSB ×N\n(Mamba)\n8C, H/8", C_MAMBA, 8.2, bold=True)
dec = [("ConvResBlock\n4C, H/4", C_NET, 0.9),
       ("ConvResBlock\n2C, H/2", C_NET, 1.9),
       ("ConvResBlock\nC, H", C_NET, 2.9)]
xd = xs + 3 * 1.55 + 1.95
for i, (t, c, y) in enumerate(dec):
    box(ax, xd + i * 1.55, y, 1.45, 0.8, t, c, 8.2)

box(ax, xd + 3 * 1.55, 2.6, 1.5, 1.0,
    "out conv\n$C\\!\\to\\!2$\n(zero-init)", C_NET, 8.2)
box(ax, xd + 3 * 1.55 + 1.7, 2.6, 1.4, 1.0, "$z$\n(B,2,H,W)\nresidual", C_DATA, 8.5)

# flow arrows down the encoder, across bottleneck, up the decoder
arrow(ax, (1.7, 3.1), (1.9, 3.1))
arrow(ax, (3.2, 3.1), (3.4, 3.3))
for i in range(2):
    arrow(ax, (xs + i * 1.55 + 0.72, enc[i][2]), (xs + (i + 1) * 1.55 + 0.72, enc[i + 1][2] + 0.8), rad=0)
arrow(ax, (xs + 2 * 1.55 + 0.72, 0.9), (xs + 3 * 1.55 + 0.85, 0.85))   # -> bottleneck
arrow(ax, (xs + 3 * 1.55 + 1.7, 0.42), (xd + 0.72, 0.9))              # bottleneck -> dec
for i in range(2):
    arrow(ax, (xd + i * 1.55 + 0.72, dec[i][2] + 0.8), (xd + (i + 1) * 1.55 + 0.72, dec[i + 1][2]), rad=0)
arrow(ax, (xd + 2 * 1.55 + 1.45, 3.3), (xd + 3 * 1.55, 3.1))
arrow(ax, (xd + 3 * 1.55 + 1.5, 3.1), (xd + 3 * 1.55 + 1.7, 3.1))

# skip connections (dashed) encoder<->decoder at matching scales
for i in range(3):
    y = enc[i][2]
    arrow(ax, (xs + i * 1.55 + 1.45, y + 0.4), (xd + (2 - i) * 1.55, y + 0.4),
          color="#1f6feb", ls="--", rad=-0.18 - 0.06 * i, lw=1.2)
ax.text(8.0, 3.55, "skip (concat + fuse)", fontsize=8.5, color="#1f6feb")

# =========================================================================== #
# (c) RSSB + SS2D
# =========================================================================== #
ax = fig.add_subplot(gs[2]); ax.set_xlim(0, 16); ax.set_ylim(0, 6.0); ax.axis("off")
ax.text(0.1, 5.7, "(c)  Residual State-Space Block (RSSB)  &  SS2D four-directional scan",
        fontsize=14, fontweight="bold")

# ---- RSSB (left) ----
ax.text(0.2, 5.15, "Residual State-Space Block", fontsize=11, fontweight="bold")
box(ax, 0.2, 4.0, 1.1, 0.7, "in\n(B,C,H,W)", C_DATA, 8.2)
box(ax, 1.7, 4.05, 1.0, 0.6, "LayerNorm", C_NET, 8.2)
box(ax, 3.1, 4.55, 1.5, 0.7, "SS2D\n(global, panel→)", C_MAMBA, 8.2, bold=True)
box(ax, 3.1, 3.55, 1.5, 0.7, "DWConv 3×3\n(local detail)", C_NET, 8.2)
box(ax, 5.0, 4.05, 0.7, 0.6, "$\\oplus$", C_DATA, 11)
box(ax, 6.0, 4.05, 1.0, 0.6, "LayerNorm", C_NET, 8.2)
box(ax, 7.3, 4.55, 1.3, 0.7, "channel MLP\n1×1, 2C", C_ATTN, 8.2)
box(ax, 7.3, 3.55, 1.3, 0.7, "SE channel\nattention", C_ATTN, 8.2)
box(ax, 8.9, 4.05, 0.7, 0.6, "$\\oplus$", C_DATA, 11)
box(ax, 9.9, 4.05, 1.0, 0.7, "out\n(B,C,H,W)", C_DATA, 8.2)

arrow(ax, (1.3, 4.35), (1.7, 4.35))
arrow(ax, (2.7, 4.35), (3.1, 4.7))
arrow(ax, (2.7, 4.35), (3.1, 3.9))
arrow(ax, (4.6, 4.9), (5.0, 4.5))
arrow(ax, (4.6, 3.9), (5.0, 4.2))
arrow(ax, (0.75, 4.0), (5.35, 4.0), rad=-0.32, color="#1f6feb", ls="--", lw=1.1)  # residual
arrow(ax, (5.7, 4.35), (6.0, 4.35))
arrow(ax, (7.0, 4.35), (7.3, 4.7))
arrow(ax, (8.6, 4.9), (7.3, 3.9), rad=0)        # mlp -> SE (sequential)
arrow(ax, (8.6, 3.9), (8.9, 4.2))
arrow(ax, (5.85, 4.05), (8.95, 4.05), rad=-0.3, color="#1f6feb", ls="--", lw=1.1)  # residual2
arrow(ax, (9.6, 4.35), (9.9, 4.35))

# ---- SS2D (bottom, full width) ----
ax.text(0.2, 2.85, "SS2D — selective state-space, four scan directions", fontsize=11, fontweight="bold")
box(ax, 0.2, 1.7, 1.1, 0.8, "feat\n(B,C,H,W)", C_DATA, 8.2)
box(ax, 1.6, 1.75, 1.2, 0.7, "in_proj\n→ x, gate z", C_NET, 8.2)
box(ax, 3.1, 1.75, 1.3, 0.7, "DWConv\n+ SiLU", C_NET, 8.2)

# four directions
dirs = ["→ readout\n(fwd)", "← readout\n(bwd)", "↓ phase-enc\n(fwd)", "↑ phase-enc\n(bwd)"]
for i, d in enumerate(dirs):
    box(ax, 4.8, 3.3 - i * 0.78, 1.6, 0.62, d, C_MAMBA, 7.8)
box(ax, 6.7, 1.0, 2.2, 2.9,
    "selective scan\nper direction:\n$\\Delta,B,C=f(x)$,\n$\\bar A=e^{\\Delta A}$\n"
    "$h_t=\\bar A h_{t-1}+\\bar B x_t$\n$y_t=C h_t+Dx_t$", C_MAMBA, 8.0)
box(ax, 9.2, 1.7, 1.1, 0.8, "merge\n(sum 4)", C_DATA, 8.2)
box(ax, 10.6, 1.75, 1.2, 0.7, "LayerNorm\n× SiLU(z)", C_ATTN, 8.2)
box(ax, 12.1, 1.75, 1.2, 0.7, "out_proj", C_NET, 8.2)
box(ax, 13.6, 1.7, 1.1, 0.8, "out\n(B,C,H,W)", C_DATA, 8.2)

arrow(ax, (1.3, 2.1), (1.6, 2.1))
arrow(ax, (2.8, 2.1), (3.1, 2.1))
for i in range(4):
    arrow(ax, (4.4, 2.1), (4.8, 3.6 - i * 0.78), rad=0.05)
    arrow(ax, (6.4, 3.6 - i * 0.78), (6.7, 2.45 - i * 0.05), rad=0.0)
arrow(ax, (8.9, 2.4), (9.2, 2.1))
arrow(ax, (10.3, 2.1), (10.6, 2.1))
arrow(ax, (11.8, 2.1), (12.1, 2.1))
arrow(ax, (13.3, 2.1), (13.6, 2.1))
ax.text(4.9, 0.55,
        "phase-encode scans are physics-aligned: Cartesian aliasing is coherent "
        "along the PE axis →\nthe sequence model sees aliasing replicas consecutively "
        "and learns to cancel them.",
        fontsize=8.6, color="#555555")

os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
