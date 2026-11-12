"""
Mechanistic interpretability metrics for OCT-MNIST projector representations.

Computes four internal measures for each checkpoint:
  1. Effective rank (participation ratio) — how many dimensions are used
  2. Uniformity (Wang & Isola 2020) — how spread representations are
  3. Alignment (Wang & Isola 2020) — how well positive pairs are pulled together
  4. Per-neuron entropy — distribution of class selectivity across all 256 neurons

Output: two figures
  - eval_mechanistic_bars.png   — global metrics bar chart (4 models × 4 metrics)
  - eval_mechanistic_entropy.png — per-neuron entropy histograms (4 models)

Usage:
    python eval_mechanistic_octmnist.py \
        --ckpts \
            trained_models/simclr/c1gu74hk/simclr-resnet18-octmnist-c1gu74hk-ep=99.ckpt \
            trained_models/simclr/8wilumpu/simclr-resnet18-octmnist-8wilumpu-ep=99.ckpt \
            trained_models/simclr/icuj06ea/ncl-A0-icuj06ea-ep=99.ckpt \
            trained_models/simclr/99tku1fz/simclr-resnet18-octmnist-ncl-99tku1fz-ep=99.ckpt \
        --labels "SimCLR A0" "SimCLR A6" "NCL A0" "NCL A6" \
        --dead 0 0 172 20
"""
import argparse, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, ".")
from MedMNIST.octmnist_dataset import OCTMnistDataset, OCTMNIST_MEAN, OCTMNIST_STD

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ckpts",      nargs="+", required=True)
parser.add_argument("--labels",     nargs="+", required=True)
parser.add_argument("--dead",       nargs="+", type=int, default=None,
                    help="Known dead neuron counts (from eval_sc) per checkpoint")
parser.add_argument("--split",      default="test")
parser.add_argument("--proj_hidden",type=int, default=2048)
parser.add_argument("--proj_output",type=int, default=256)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--num_workers",type=int, default=0)
parser.add_argument("--n_aug_pairs",type=int, default=500,
                    help="Number of test images to use for alignment (augmented twice each)")
parser.add_argument("--output_bars",   default="eval_mechanistic_bars.png")
parser.add_argument("--output_entropy",default="eval_mechanistic_entropy.png")
parser.add_argument("--dpi", type=int, default=150)
args = parser.parse_args()

assert len(args.ckpts) == len(args.labels)
dead_counts = args.dead or [None] * len(args.ckpts)
assert len(dead_counts) == len(args.ckpts)

CLASS_NAMES  = ["CNV", "DME", "DRUSEN", "NORMAL"]
N_CLASSES    = 4
MODEL_COLORS = ["#3498db", "#2980b9", "#e74c3c", "#c0392b"]  # SimCLR light/dark, NCL light/dark

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}  |  Split: {args.split}")

# ── Data ──────────────────────────────────────────────────────────────────────
norm_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(OCTMNIST_MEAN, OCTMNIST_STD),
])

# augmentation transform for alignment pairs (standard SSL view)
aug_tf = transforms.Compose([
    transforms.RandomResizedCrop(64, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.8),
    transforms.ToTensor(),
    transforms.Normalize(OCTMNIST_MEAN, OCTMNIST_STD),
])

ds      = OCTMnistDataset(split=args.split, transform=norm_tf)
loader  = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                     num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
labels_all = torch.tensor(ds.labels)

# Dataset that returns two augmented views of each image (for alignment)
class AugPairDataset(Dataset):
    def __init__(self, base_ds, n):
        self.ds = base_ds
        self.n  = min(n, len(base_ds))
    def __len__(self): return self.n
    def __getitem__(self, idx):
        from PIL import Image
        img_arr = self.ds.images[idx]
        img_pil = Image.fromarray(img_arr).convert("RGB")
        return aug_tf(img_pil), aug_tf(img_pil)

aug_ds     = AugPairDataset(ds, args.n_aug_pairs)
aug_loader = DataLoader(aug_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

# ── Model ─────────────────────────────────────────────────────────────────────
def load_model(ckpt_path):
    bb  = models.resnet18(weights=None); bb.fc = nn.Identity()
    proj = nn.Sequential(nn.Linear(512, args.proj_hidden), nn.ReLU(),
                         nn.Linear(args.proj_hidden, args.proj_output))
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    bb.load_state_dict(  {k.replace("backbone.",  ""): v for k, v in sd.items() if k.startswith("backbone.")},  strict=True)
    proj.load_state_dict({k.replace("projector.", ""): v for k, v in sd.items() if k.startswith("projector.")}, strict=True)
    bb.to(device).eval(); proj.to(device).eval()
    for p in list(bb.parameters()) + list(proj.parameters()): p.requires_grad = False
    is_ncl = "ncl" in Path(ckpt_path).stem.lower()
    return bb, proj, is_ncl


def extract(bb, proj, is_ncl, ldr=None):
    ldr = ldr or loader
    feats = []
    with torch.no_grad():
        for batch in ldr:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if isinstance(x, (list, tuple)): x = x[0]
            h = bb(x.to(device))
            z = proj(h)
            if is_ncl: z = F.relu(z)
            feats.append(z.cpu().float())
    return torch.cat(feats)


# ── Mechanistic metrics ────────────────────────────────────────────────────────
def effective_rank(feats):
    """Participation ratio via SVD: (Σσ²)² / Σσ⁴  — stable even with zero dimensions."""
    f = feats - feats.mean(0)
    sv = torch.linalg.svdvals(f).numpy()   # singular values, descending
    lam = sv ** 2 / len(f)                 # proportional to eigenvalues
    lam = lam[lam > 1e-10]                 # drop numerical zeros
    if len(lam) == 0: return 0.0
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def uniformity(feats):
    """log E[exp(-2||z_i - z_j||²)]  (Wang & Isola 2020, lower = more uniform)."""
    z = F.normalize(feats, dim=1)
    sq = torch.cdist(z, z).pow(2)
    return float(torch.log(torch.exp(-2 * sq).mean()))


def alignment(bb, proj, is_ncl):
    """Mean -||z1 - z2||²  for augmented positive pairs (higher = better aligned)."""
    vals = []
    with torch.no_grad():
        for v1, v2 in aug_loader:
            z1 = proj(bb(v1.to(device))); z2 = proj(bb(v2.to(device)))
            if is_ncl: z1 = F.relu(z1); z2 = F.relu(z2)
            z1 = F.normalize(z1, dim=1); z2 = F.normalize(z2, dim=1)
            vals.append(-(z1 - z2).pow(2).sum(dim=1).cpu())
    return float(torch.cat(vals).mean())


def neuron_entropy(feats, labels):
    """Per-neuron Shannon entropy over class-conditional mean activations.
    Returns array of shape (proj_output,). Low = selective, high = broad."""
    class_means = np.zeros((args.proj_output, N_CLASSES))
    for c in range(N_CLASSES):
        class_means[:, c] = feats[labels == c].mean(axis=0)
    # softmax-normalise across classes per neuron
    class_means = np.maximum(class_means, 0)
    row_sum = class_means.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    p = class_means / row_sum
    p = np.clip(p, 1e-9, 1)
    return -(p * np.log(p)).sum(axis=1)   # (256,)


# ── Run ────────────────────────────────────────────────────────────────────────
results = []
for ckpt, label, n_dead in zip(args.ckpts, args.labels, dead_counts):
    print(f"\n── {label} ──")
    bb, proj, is_ncl = load_model(ckpt)

    feats = extract(bb, proj, is_ncl)
    labels_np = labels_all.numpy()

    er   = effective_rank(feats)
    unif = uniformity(feats)
    aln  = alignment(bb, proj, is_ncl)
    ent  = neuron_entropy(feats.numpy(), labels_np)

    print(f"  Effective rank : {er:.1f}")
    print(f"  Uniformity     : {unif:.4f}")
    print(f"  Alignment      : {aln:.4f}")
    print(f"  Mean entropy   : {ent.mean():.4f}  (max possible = {np.log(N_CLASSES):.4f})")
    if n_dead is not None:
        print(f"  Dead neurons   : {n_dead}/{args.proj_output}")

    results.append(dict(label=label, er=er, unif=unif, aln=aln, ent=ent, dead=n_dead))


# ── Figure 1: Bar charts ──────────────────────────────────────────────────────
metrics = [
    ("Effective rank\n(max=256)", "er",   True,  "Higher = more dimensions used"),
    ("Uniformity\n(↑ better)",    "unif", True,  "Wang & Isola 2020 (normalised)"),
    ("Alignment\n(↑ better)",     "aln",  True,  "Wang & Isola 2020 (normalised pairs)"),
    ("Mean neuron entropy\n(↓ = more selective)", "aln", False, ""),
]

fig1, axes1 = plt.subplots(1, 3, figsize=(12, 4.5))

bar_metrics = [
    ("Effective rank\n(out of 256)", "er"),
    ("Uniformity",                   "unif"),
    ("Alignment",                    "aln"),
]

x = np.arange(len(results))
for ax, (title, key) in zip(axes1, bar_metrics):
    vals   = [r[key] for r in results]
    colors = MODEL_COLORS[:len(results)]
    bars   = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.8, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in results], fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8)

fig1.suptitle("Mechanistic Metrics — OCT-MNIST Projector (256-d)",
              fontsize=13, fontweight="bold")
fig1.tight_layout()
fig1.savefig(args.output_bars, dpi=args.dpi, bbox_inches="tight")
print(f"\nSaved: {args.output_bars}")


# ── Figure 2: Per-neuron entropy histograms ───────────────────────────────────
fig2, axes2 = plt.subplots(1, len(results), figsize=(3.5 * len(results), 4),
                           sharey=False)
max_ent = np.log(N_CLASSES)

for ax, r, color in zip(axes2, results, MODEL_COLORS):
    ent = r["ent"]
    dead = r["dead"]

    # separate dead neurons (entropy undefined — all zeros) from alive
    alive_ent = ent[ent > 0] if dead is None else ent[np.argsort(ent)[dead:]]

    ax.hist(alive_ent, bins=20, range=(0, max_ent), color=color,
            alpha=0.85, edgecolor="white", linewidth=0.5)

    if dead:
        ax.axvline(0, color="grey", linestyle="--", linewidth=1.2, label=f"{dead} dead (not shown)")
        ax.legend(fontsize=7)

    ax.axvline(max_ent, color="black", linestyle=":", linewidth=1,
               label=f"Max entropy ({max_ent:.2f})")
    ax.set_xlabel("Entropy (nats)", fontsize=9)
    ax.set_ylabel("Neuron count", fontsize=9)
    ax.set_title(f"{r['label']}\nmean={alive_ent.mean():.3f}", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max_ent + 0.05)

fig2.suptitle("Per-Neuron Class Entropy Distribution\n"
              "Low entropy = class-selective (monosemantic)   |   High entropy = broadly tuned",
              fontsize=12, fontweight="bold")
fig2.tight_layout()
fig2.savefig(args.output_entropy, dpi=args.dpi, bbox_inches="tight")
print(f"Saved: {args.output_entropy}")
