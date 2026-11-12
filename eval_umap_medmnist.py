"""
UMAP visualisation of projector features for MedMNIST checkpoints.
Supports octmnist, pathmnist, organamnist.

Usage:
    python eval_umap_medmnist.py --dataset octmnist \
        --ckpts <ckpt1> <ckpt2> <ckpt3> <ckpt4> \
        --panel_labels "SimCLR A0" "SimCLR A6" "NCL A0" "NCL A6" \
        --output umap_octmnist.png
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

try:
    import umap
except ImportError:
    sys.exit("umap-learn not installed — run: pip install umap-learn")

# ── Dataset registry ──────────────────────────────────────────────────────────
DATASETS = {
    "octmnist": {
        "mean": (0.1894, 0.1894, 0.1894),
        "std":  (0.2071, 0.2071, 0.2071),
        "class_names": ["CNV", "DME", "DRUSEN", "NORMAL"],
        "colors": ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"],
        "title": "OCT-MNIST",
    },
    "pathmnist": {
        "mean": (0.7406, 0.5331, 0.7059),
        "std":  (0.1292, 0.1765, 0.1381),
        "class_names": ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"],
        "colors": ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
                   "#1abc9c","#e67e22","#34495e","#e91e63"],
        "title": "PathMNIST",
    },
    "organamnist": {
        "mean": (0.4677, 0.4677, 0.4677),
        "std":  (0.2479, 0.2479, 0.2479),
        "class_names": ["bladder","femur-L","femur-R","heart","kidney-L",
                        "kidney-R","liver","lung","pancreas","spleen","vessel"],
        "colors": ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
                   "#1abc9c","#e67e22","#34495e","#e91e63","#00bcd4","#8bc34a"],
        "title": "OrganAMNIST",
    },
}

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
parser.add_argument("--ckpts", nargs="+", required=True)
parser.add_argument("--panel_labels", nargs="+", default=None)
parser.add_argument("--nrows", type=int, default=2)
parser.add_argument("--ncols", type=int, default=2)
parser.add_argument("--split", default="test", choices=["train", "val", "test"])
parser.add_argument("--proj_hidden", type=int, default=2048)
parser.add_argument("--proj_output", type=int, default=256)
parser.add_argument("--batch_size",  type=int, default=512)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--n_neighbors", type=int, default=15)
parser.add_argument("--min_dist",    type=float, default=0.1)
parser.add_argument("--seed",        type=int, default=42)
parser.add_argument("--output",      default=None)
parser.add_argument("--dpi",         type=int, default=150)
args = parser.parse_args()

assert len(args.ckpts) == args.nrows * args.ncols, \
    f"Expected {args.nrows*args.ncols} checkpoints, got {len(args.ckpts)}"

panel_labels = args.panel_labels or [Path(c).stem for c in args.ckpts]
assert len(panel_labels) == len(args.ckpts)

cfg = DATASETS[args.dataset]
output = args.output or f"umap_{args.dataset}.png"

# ── Data ──────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=cfg["mean"], std=cfg["std"]),
])

if args.dataset == "octmnist":
    from MedMNIST.octmnist_dataset import OCTMnistDataset
    dataset = OCTMnistDataset(split=args.split, transform=transform)
elif args.dataset == "pathmnist":
    from MedMNIST.pathmnist.dataset import PathMNISTDataset
    dataset = PathMNISTDataset(split=args.split, transform=transform)
elif args.dataset == "organamnist":
    from MedMNIST.organamnist.dataset import OrganAMNISTDataset
    dataset = OrganAMNISTDataset(split=args.split, transform=transform)

loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dataset: {args.dataset}  |  Device: {device}  |  Split: {args.split}")

# ── Model ─────────────────────────────────────────────────────────────────────
def load_model(ckpt_path):
    backbone_name = "resnet50" if "resnet50" in str(ckpt_path).lower() else "resnet18"
    backbone = models.resnet50(weights=None) if backbone_name == "resnet50" \
               else models.resnet18(weights=None)
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    projector = nn.Sequential(
        nn.Linear(feat_dim, args.proj_hidden), nn.ReLU(),
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
    is_ncl = "ncl" in Path(ckpt_path).stem.lower()
    return backbone, projector, is_ncl


def extract_features(backbone, projector, is_ncl):
    all_feats, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1]
            z = projector(backbone(x))
            if is_ncl:
                z = F.relu(z)
            all_feats.append(z.cpu().float())
            lbl = y.squeeze() if y.ndim > 1 else y
            all_labels.append(lbl)
    return torch.cat(all_feats).numpy(), torch.cat(all_labels).numpy().astype(int)


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(args.nrows, args.ncols,
                         figsize=(5 * args.ncols, 4.5 * args.nrows))
axes = np.array(axes).reshape(args.nrows, args.ncols)

reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                    n_components=2, metric="euclidean", random_state=args.seed)

class_names = cfg["class_names"]
colors = cfg["colors"]

for idx, (ckpt_path, label) in enumerate(zip(args.ckpts, panel_labels)):
    row, col = divmod(idx, args.ncols)
    ax = axes[row, col]

    print(f"\n[{idx+1}/{len(args.ckpts)}] {label} ...")
    backbone, projector, is_ncl = load_model(ckpt_path)
    feats, labels = extract_features(backbone, projector, is_ncl)

    n_dead = int((feats.sum(axis=0) == 0).sum())
    print(f"  shape={feats.shape}  dead={n_dead}/{args.proj_output}  NCL={is_ncl}")

    embedding = reducer.fit_transform(feats)

    for cls_idx, (cls_name, color) in enumerate(zip(class_names, colors)):
        mask = labels == cls_idx
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=color, s=4, alpha=0.6, rasterized=True)

    ax.set_title(f"{label}  ({n_dead}/{args.proj_output} dead)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal", "datalim")

legend_patches = [mpatches.Patch(color=c, label=n)
                  for c, n in zip(colors, class_names)]
axes[-1, -1].legend(handles=legend_patches, fontsize=7, loc="lower right",
                    framealpha=0.8, title="Class", ncol=2 if len(class_names) > 6 else 1)

fig.suptitle(f"{cfg['title']} — Projector Features UMAP",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
print(f"\nSaved: {output}")
