"""
Per-neuron class selectivity heatmap for OCT-MNIST checkpoints.

For each checkpoint: computes mean activation per neuron per class (256 × 4),
sorts neurons by preferred class, plots as heatmap.
Dead neurons (always zero) are grouped at the bottom in grey.

Default: 2×2 grid — SimCLR A0 | SimCLR A6 / NCL A0 | NCL A6

Usage:
    python eval_heatmap_octmnist.py \
        --ckpts  trained_models/simclr/c1gu74hk/simclr-resnet18-octmnist-c1gu74hk-ep=99.ckpt \
                 trained_models/simclr/8wilumpu/simclr-resnet18-octmnist-8wilumpu-ep=99.ckpt \
                 trained_models/simclr/icuj06ea/ncl-A0-icuj06ea-ep=99.ckpt \
                 trained_models/simclr/99tku1fz/simclr-resnet18-octmnist-ncl-99tku1fz-ep=99.ckpt \
        --panel_labels "SimCLR A0" "SimCLR A6" "NCL A0" "NCL A6" \
        --nrows 2 --ncols 2 \
        --output heatmap_a0_vs_a6.png
"""
import argparse, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

sys.path.insert(0, ".")
from MedMNIST.octmnist_dataset import OCTMnistDataset, OCTMNIST_MEAN, OCTMNIST_STD

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ckpts", nargs="+", required=True)
parser.add_argument("--panel_labels", nargs="+", default=None)
parser.add_argument("--nrows", type=int, default=2)
parser.add_argument("--ncols", type=int, default=2)
parser.add_argument("--split",       default="test")
parser.add_argument("--proj_hidden", type=int, default=2048)
parser.add_argument("--proj_output", type=int, default=256)
parser.add_argument("--batch_size",  type=int, default=512)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--output",      default="heatmap_octmnist.png")
parser.add_argument("--dpi",         type=int, default=180)
args = parser.parse_args()

assert len(args.ckpts) == args.nrows * args.ncols
panel_labels = args.panel_labels or [Path(c).stem for c in args.ckpts]
assert len(panel_labels) == len(args.ckpts)

CLASS_NAMES  = ["CNV", "DME", "DRUSEN", "NORMAL"]
CLASS_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
N_CLASSES    = 4

# ── Data ──────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=OCTMNIST_MEAN, std=OCTMNIST_STD),
])
loader = DataLoader(
    OCTMnistDataset(split=args.split, transform=transform),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}  |  Split: {args.split}")

# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(ckpt_path):
    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Identity()
    projector = nn.Sequential(
        nn.Linear(512, args.proj_hidden),
        nn.ReLU(),
        nn.Linear(args.proj_hidden, args.proj_output),
    )
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    backbone.load_state_dict(
        {k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")}, strict=True)
    projector.load_state_dict(
        {k.replace("projector.", ""): v for k, v in sd.items() if k.startswith("projector.")}, strict=True)
    backbone.to(device).eval()
    projector.to(device).eval()
    for p in list(backbone.parameters()) + list(projector.parameters()):
        p.requires_grad = False
    is_ncl = "ncl" in Path(ckpt_path).stem.lower()
    return backbone, projector, is_ncl


def extract(backbone, projector, is_ncl):
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            h = backbone(x.to(device))
            z = projector(h)
            if is_ncl:
                z = F.relu(z)
            feats.append(z.cpu().float())
            labels.append(y)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def class_mean_matrix(feats, labels):
    """Returns (n_neurons, n_classes) matrix of mean activations per class."""
    mat = np.zeros((feats.shape[1], N_CLASSES))
    for c in range(N_CLASSES):
        mat[:, c] = feats[labels == c].mean(axis=0)
    return mat


def sort_neurons(mat):
    """
    Sort neurons: first by preferred class (argmax), then by max activation.
    Dead neurons (all zeros) go to the bottom.
    Returns sorted matrix and a per-neuron preferred-class label array.
    """
    row_max   = mat.max(axis=1)
    dead_mask = row_max == 0
    pref_cls  = mat.argmax(axis=1)
    pref_cls[dead_mask] = N_CLASSES  # sentinel — placed last

    # sort: primary = preferred class, secondary = descending max activation
    order = np.lexsort((-row_max, pref_cls))
    return mat[order], pref_cls[order], dead_mask[order]


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(args.nrows, args.ncols,
                         figsize=(4.5 * args.ncols, 6 * args.nrows))
axes = np.array(axes).reshape(args.nrows, args.ncols)

for idx, (ckpt_path, label) in enumerate(zip(args.ckpts, panel_labels)):
    row, col = divmod(idx, args.ncols)
    ax = axes[row, col]

    print(f"\n[{idx+1}/{len(args.ckpts)}] {label}")
    backbone, projector, is_ncl = load_model(ckpt_path)
    feats, labels_arr = extract(backbone, projector, is_ncl)

    mat = class_mean_matrix(feats, labels_arr)
    sorted_mat, pref_cls, dead_mask = sort_neurons(mat)

    n_dead  = int(dead_mask.sum())
    n_alive = args.proj_output - n_dead
    print(f"  dead: {n_dead}/{args.proj_output}  alive: {n_alive}")

    # row-normalise so each neuron's max = 1 (makes colour comparable across neurons)
    row_max = sorted_mat.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1  # avoid divide-by-zero for dead neurons
    norm_mat = sorted_mat / row_max

    # build a custom colormap: dead neurons show in light grey
    cmap = plt.cm.YlOrRd
    display = norm_mat.copy()

    im = ax.imshow(display, aspect="auto", cmap=cmap,
                   vmin=0, vmax=1, interpolation="nearest")

    # grey overlay for dead neurons
    if n_dead > 0:
        dead_overlay = np.ones((n_dead, N_CLASSES, 4))
        dead_overlay[:, :, :3] = 0.82  # light grey RGB
        dead_overlay[:, :, 3]  = 1.0
        ax.imshow(dead_overlay, aspect="auto",
                  extent=[-0.5, N_CLASSES - 0.5, n_alive - 0.5, args.proj_output - 0.5],
                  interpolation="nearest", zorder=2)

    # divider line between alive and dead
    if 0 < n_dead < args.proj_output:
        ax.axhline(n_alive - 0.5, color="white", linewidth=1.5, linestyle="--", zorder=3)

    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, fontsize=9, fontweight="bold")
    ax.set_ylabel("Neuron index (sorted by preferred class)", fontsize=8)
    ax.set_yticks([])
    ax.set_title(f"{label}\n({n_dead}/{args.proj_output} dead neurons)", fontsize=11, fontweight="bold")

    # colour bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Norm. activation", fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    # class colour bands on top x-axis
    for c, color in enumerate(CLASS_COLORS):
        ax.axvline(c - 0.5, color="white", linewidth=0.5)
        ax.get_xticklabels()[c].set_color(color)

    # dead neuron legend
    if n_dead > 0:
        ax.text(N_CLASSES / 2 - 0.5, n_alive + n_dead / 2,
                f"Dead\n({n_dead})", ha="center", va="center",
                fontsize=8, color="#555", zorder=4)

fig.suptitle("OCT-MNIST Projector — Per-Neuron Class Selectivity\n(row-normalised mean activation, sorted by preferred class)",
             fontsize=12, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
print(f"\nSaved: {args.output}")