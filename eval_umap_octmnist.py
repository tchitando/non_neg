"""
UMAP visualisation of projector features for OCT-MNIST checkpoints.

Produces a grid of UMAP plots coloured by class label, one panel per checkpoint.
Default layout: 2 rows × 2 cols  →  [SimCLR A0 | SimCLR A6]
                                     [NCL A0    | NCL A6   ]

Usage (defaults to A0 vs A6 comparison):
    python eval_umap_octmnist.py \
        --ckpts  trained_models/simclr/c1gu74hk/simclr-A0-c1gu74hk-ep=99.ckpt \
                 trained_models/simclr/8wilumpu/simclr-A6-8wilumpu-ep=99.ckpt \
                 trained_models/simclr/icuj06ea/ncl-A0-icuj06ea-ep=99.ckpt \
                 trained_models/simclr/99tku1fz/ncl-A6-99tku1fz-ep=99.ckpt \
        --panel_labels "SimCLR A0" "SimCLR A6" "NCL A0" "NCL A6" \
        --nrows 2 --ncols 2 \
        --output umap_a0_vs_a6.png
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
import matplotlib.patches as mpatches

sys.path.insert(0, ".")
from MedMNIST.octmnist_dataset import OCTMnistDataset, OCTMNIST_MEAN, OCTMNIST_STD

try:
    import umap
except ImportError:
    sys.exit("umap-learn not installed — run: pip install umap-learn")

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ckpts", nargs="+", required=True,
                    help="Checkpoint paths, one per panel (in panel order)")
parser.add_argument("--panel_labels", nargs="+", default=None,
                    help="Label for each panel (same length as --ckpts)")
parser.add_argument("--nrows", type=int, default=2)
parser.add_argument("--ncols", type=int, default=2)
parser.add_argument("--split", default="test", choices=["train", "val", "test"])
parser.add_argument("--proj_hidden", type=int, default=2048)
parser.add_argument("--proj_output", type=int, default=256)
parser.add_argument("--batch_size",  type=int, default=512)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--n_neighbors", type=int, default=15,  help="UMAP n_neighbors")
parser.add_argument("--min_dist",    type=float, default=0.1, help="UMAP min_dist")
parser.add_argument("--seed",        type=int, default=42)
parser.add_argument("--output",      default="umap_octmnist.png")
parser.add_argument("--dpi",         type=int, default=150)
args = parser.parse_args()

assert len(args.ckpts) == args.nrows * args.ncols, \
    f"Expected {args.nrows}×{args.ncols}={args.nrows*args.ncols} checkpoints, got {len(args.ckpts)}"

panel_labels = args.panel_labels or [Path(c).stem for c in args.ckpts]
assert len(panel_labels) == len(args.ckpts), "--panel_labels must match --ckpts length"

CLASS_NAMES  = ["CNV", "DME", "DRUSEN", "NORMAL"]
CLASS_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]  # red, blue, green, orange

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

# ── Feature extraction ────────────────────────────────────────────────────────
def load_model(ckpt_path):
    backbone_name = "resnet50" if "resnet50" in str(ckpt_path).lower() else "resnet18"
    if backbone_name == "resnet50":
        backbone = models.resnet50(weights=None)
    else:
        backbone = models.resnet18(weights=None)
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()

    projector = nn.Sequential(
        nn.Linear(feat_dim, args.proj_hidden),
        nn.ReLU(),
        nn.Linear(args.proj_hidden, args.proj_output),
    )

    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    backbone.load_state_dict(
        {k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")},
        strict=True)
    projector.load_state_dict(
        {k.replace("projector.", ""): v for k, v in sd.items() if k.startswith("projector.")},
        strict=True)

    backbone.to(device).eval()
    projector.to(device).eval()
    for p in list(backbone.parameters()) + list(projector.parameters()):
        p.requires_grad = False

    # apply ReLU to projector output for NCL checkpoints (rep_relu forward pass)
    is_ncl = "ncl" in Path(ckpt_path).stem.lower()
    return backbone, projector, is_ncl


def extract_features(backbone, projector, is_ncl):
    all_feats, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            h = backbone(x)
            z = projector(h)
            if is_ncl:
                z = F.relu(z)
            all_feats.append(z.cpu().float())
            all_labels.append(y)
    feats  = torch.cat(all_feats).numpy()
    labels = torch.cat(all_labels).numpy()
    return feats, labels


# ── UMAP + Plot ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(args.nrows, args.ncols,
                         figsize=(5 * args.ncols, 4.5 * args.nrows))
axes = np.array(axes).reshape(args.nrows, args.ncols)

reducer = umap.UMAP(
    n_neighbors=args.n_neighbors,
    min_dist=args.min_dist,
    n_components=2,
    metric="euclidean",
    random_state=args.seed,
)

for idx, (ckpt_path, label) in enumerate(zip(args.ckpts, panel_labels)):
    row, col = divmod(idx, args.ncols)
    ax = axes[row, col]

    print(f"\n[{idx+1}/{len(args.ckpts)}] Loading {label} ...")
    backbone, projector, is_ncl = load_model(ckpt_path)
    feats, labels = extract_features(backbone, projector, is_ncl)

    n_dead = int((feats.sum(axis=0) == 0).sum())
    print(f"  features: {feats.shape}  |  dead neurons: {n_dead}/{args.proj_output}  |  NCL: {is_ncl}")

    print(f"  Running UMAP ...")
    embedding = reducer.fit_transform(feats)

    for cls_idx, (cls_name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        mask = labels == cls_idx
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=color, s=4, alpha=0.6, rasterized=True, label=cls_name)

    ax.set_title(f"{label}\n({n_dead}/{args.proj_output} dead neurons)", fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal", "datalim")

# shared legend on last panel
legend_patches = [mpatches.Patch(color=c, label=n) for c, n in zip(CLASS_COLORS, CLASS_NAMES)]
axes[-1, -1].legend(handles=legend_patches, fontsize=8, loc="lower right",
                    framealpha=0.8, title="Class")

fig.suptitle("OCT-MNIST Projector Features — UMAP by Augmentation Config",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
print(f"\nSaved: {args.output}")
