"""
Top-activating image visualisation for OCT-MNIST projector neurons.

For each model, finds the most class-selective neuron for each OCT class
(CNV, DME, DRUSEN, NORMAL) and displays the top-K images that activate it most.

Layout: NCL A6 vs SimCLR A6 side-by-side, 4 rows (one per class).
Each cell = 3×3 mosaic of top-9 images; border colour = true class of each image.

Usage:
    python eval_top_neurons_octmnist.py \
        --ckpt_ncl  trained_models/simclr/99tku1fz/simclr-resnet18-octmnist-ncl-99tku1fz-ep=99.ckpt \
        --ckpt_sim  trained_models/simclr/8wilumpu/simclr-resnet18-octmnist-8wilumpu-ep=99.ckpt \
        --label_ncl "NCL A6" --dead_ncl 20 \
        --label_sim "SimCLR A6" --dead_sim 0 \
        --split train --top_k 9 \
        --output top_neurons_ncl_vs_simclr.png
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
import matplotlib.gridspec as gridspec

sys.path.insert(0, ".")
from MedMNIST.octmnist_dataset import OCTMnistDataset, OCTMNIST_MEAN, OCTMNIST_STD

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_ncl",   required=True)
parser.add_argument("--ckpt_sim",   required=True)
parser.add_argument("--label_ncl",  default="NCL A6")
parser.add_argument("--label_sim",  default="SimCLR A6")
parser.add_argument("--dead_ncl",   type=int, default=20,  help="Known dead neuron count for NCL (from eval_sc)")
parser.add_argument("--dead_sim",   type=int, default=0,   help="Known dead neuron count for SimCLR")
parser.add_argument("--split",      default="train", choices=["train", "val", "test"])
parser.add_argument("--top_k",      type=int, default=9)
parser.add_argument("--proj_hidden",type=int, default=2048)
parser.add_argument("--proj_output",type=int, default=256)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--num_workers",type=int, default=0)
parser.add_argument("--output",     default="top_neurons_ncl_vs_simclr.png")
parser.add_argument("--dpi",        type=int, default=150)
args = parser.parse_args()

mosaic_n = int(args.top_k ** 0.5)
assert mosaic_n ** 2 == args.top_k, "--top_k must be a perfect square"

CLASS_NAMES  = ["CNV", "DME", "DRUSEN", "NORMAL"]
CLASS_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
N_CLASSES    = 4
CELL         = 64
BORDER       = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}  |  Split: {args.split}")

# ── Dataset ───────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=OCTMNIST_MEAN, std=OCTMNIST_STD),
])
ds         = OCTMnistDataset(split=args.split, transform=transform)
raw_images = ds.images   # (N, 64, 64) uint8
all_labels = ds.labels
loader     = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

# ── Model ─────────────────────────────────────────────────────────────────────
def load_model(ckpt_path):
    backbone  = models.resnet18(weights=None)
    backbone.fc = nn.Identity()
    projector = nn.Sequential(
        nn.Linear(512, args.proj_hidden), nn.ReLU(),
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
    feats = []
    with torch.no_grad():
        for x, _ in loader:
            h = backbone(x.to(device))
            z = projector(h)
            if is_ncl:
                z = F.relu(z)
            feats.append(z.cpu().float())
    return torch.cat(feats).numpy()


def best_neuron_per_class(feats):
    """For each class, return the neuron with highest mean activation for that class
    relative to its overall mean. Dead neurons (max activation == 0) are excluded."""
    class_means = np.zeros((args.proj_output, N_CLASSES))
    for c in range(N_CLASSES):
        class_means[:, c] = feats[all_labels == c].mean(axis=0)

    # exclude dead neurons
    live_mask = class_means.max(axis=1) > 0
    class_means[~live_mask] = -1

    # selectivity = class_mean / mean across all classes (floor at 1e-8)
    global_mean = np.maximum(class_means.mean(axis=1, keepdims=True), 1e-8)
    selectivity  = class_means / global_mean
    selectivity[~live_mask] = -1

    return selectivity.argmax(axis=0)   # (N_CLASSES,)


def top_k_indices(feats, neuron_idx):
    return np.argsort(feats[:, neuron_idx])[::-1][:args.top_k]


def make_mosaic(img_indices):
    """Returns a numpy canvas of mosaic_n × mosaic_n images with class-coloured borders."""
    canvas_px = mosaic_n * (CELL + 2 * BORDER)
    canvas    = np.ones((canvas_px, canvas_px, 3), dtype=np.uint8) * 240

    for pos, img_idx in enumerate(img_indices):
        row, col = divmod(pos, mosaic_n)
        y = row * (CELL + 2 * BORDER)
        x = col * (CELL + 2 * BORDER)

        # border colour = true class of this image
        rgb = [int(v * 255) for v in plt.matplotlib.colors.to_rgb(CLASS_COLORS[all_labels[img_idx]])]
        canvas[y:y + CELL + 2*BORDER, x:x + CELL + 2*BORDER] = rgb

        patch = raw_images[img_idx]
        canvas[y+BORDER:y+BORDER+CELL, x+BORDER:x+BORDER+CELL] = np.stack([patch]*3, axis=-1)

    return canvas


# ── Extract features ──────────────────────────────────────────────────────────
print(f"\nLoading {args.label_ncl} ...")
bb_n, proj_n, is_ncl_n = load_model(args.ckpt_ncl)
feats_ncl  = extract(bb_n, proj_n, is_ncl_n)
neurons_ncl = best_neuron_per_class(feats_ncl)
print(f"  class neurons: { {c: int(neurons_ncl[i]) for i, c in enumerate(CLASS_NAMES)} }")

print(f"\nLoading {args.label_sim} ...")
bb_s, proj_s, is_ncl_s = load_model(args.ckpt_sim)
feats_sim  = extract(bb_s, proj_s, is_ncl_s)
neurons_sim = best_neuron_per_class(feats_sim)
print(f"  class neurons: { {c: int(neurons_sim[i]) for i, c in enumerate(CLASS_NAMES)} }")

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(8, 2.6 * N_CLASSES + 1.2))

# header row + 4 data rows
gs = gridspec.GridSpec(N_CLASSES + 1, 2,
                       height_ratios=[0.18] + [1] * N_CLASSES,
                       hspace=0.08, wspace=0.06,
                       figure=fig)

# ── Column headers ────────────────────────────────────────────────────────────
for col, (label, n_dead) in enumerate([
        (args.label_ncl, args.dead_ncl),
        (args.label_sim, args.dead_sim)]):
    ax_hdr = fig.add_subplot(gs[0, col])
    ax_hdr.set_xlim(0, 1); ax_hdr.set_ylim(0, 1)
    ax_hdr.axis("off")
    ax_hdr.text(0.5, 0.5,
                f"{label}   ({n_dead}/{args.proj_output} dead neurons)",
                ha="center", va="center", fontsize=12, fontweight="bold",
                transform=ax_hdr.transAxes)

# ── Mosaic panels ─────────────────────────────────────────────────────────────
for cls_idx, (cls_name, cls_color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
    for col, (feats, neurons, label) in enumerate([
            (feats_ncl, neurons_ncl, args.label_ncl),
            (feats_sim, neurons_sim, args.label_sim)]):

        ax = fig.add_subplot(gs[cls_idx + 1, col])

        neuron_idx = neurons[cls_idx]
        top_idx    = top_k_indices(feats, neuron_idx)
        top_labels = all_labels[top_idx]
        purity     = (top_labels == cls_idx).mean() * 100

        mosaic = make_mosaic(top_idx)
        ax.imshow(mosaic, interpolation="nearest")
        ax.axis("off")

        # thin coloured border around the whole panel = target class colour
        for spine in ax.spines.values():
            spine.set_edgecolor(cls_color)
            spine.set_linewidth(3)
            spine.set_visible(True)

        ax.set_title(f"neuron {neuron_idx}  →  {cls_name}   |   purity {purity:.0f}%",
                     fontsize=8.5, pad=3)

    # class label between the two columns — use fig.text
    fig.text(0.505, 1 - (cls_idx + 1.45) / (N_CLASSES + 1),
             cls_name, ha="center", va="center",
             fontsize=10, fontweight="bold", color=cls_color)

fig.suptitle("Top-Activating Images per Class-Selective Projector Neuron\n"
             "Border colour = true class label of each image",
             fontsize=12, fontweight="bold", y=1.0)

fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
print(f"\nSaved: {args.output}")
