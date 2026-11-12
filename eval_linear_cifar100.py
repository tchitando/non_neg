"""
Linear evaluation for SimCLR / NCL checkpoints on CIFAR-100.

Matches the paper's pipeline: backbone is frozen, linear classifier is trained
on augmented images (RandomCrop + RandomHorizontalFlip) each epoch.

Usage:
    python eval_linear_cifar100.py --ckpt trained_models/simclr/10/simclr-resnet18-cifar100-10-ep=199.ckpt
    python eval_linear_cifar100.py --ckpt trained_models/simclr/11/simclr-resnet18-cifar100-ncl-11-ep=199.ckpt --ncl true
"""
import argparse, sys, math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets

sys.path.insert(0, ".")
from solo.utils.lars import LARS

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",        required=True)
parser.add_argument("--ncl",         default="auto", help="true|false|auto")
parser.add_argument("--data_path",   default="./data")
parser.add_argument("--epochs",       default=50,   type=int)
parser.add_argument("--lr",           default=0.1,  type=float)
parser.add_argument("--weight_decay", default=1e-4, type=float)
parser.add_argument("--warmup_epochs",default=10,   type=int)
parser.add_argument("--batch_size",   default=256,  type=int)
parser.add_argument("--num_workers", default=0,   type=int)
parser.add_argument("--noise",       default=0.0, type=float, help="Symmetric label noise rate")
parser.add_argument("--proj_hidden", default=2048, type=int, help="Projector hidden dim")
parser.add_argument("--proj_output", default=256,  type=int, help="Projector output dim")
parser.add_argument("--seed",        default=42,   type=int, help="Random seed for reproducibility")
args = parser.parse_args()

import random
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

apply_relu = ("ncl" in args.ckpt.lower()) if args.ncl.lower() == "auto" else (args.ncl.lower() == "true")
print(f"NCL mode (apply ReLU to backbone features): {apply_relu}")

# ── Backbone + Projector ──────────────────────────────────────────────────────
backbone = models.resnet18(weights=None)
backbone.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
backbone.maxpool = nn.Identity()
feat_dim = backbone.fc.in_features
backbone.fc = nn.Identity()

ckpt = torch.load(args.ckpt, map_location="cpu")
sd = ckpt["state_dict"]
backbone_sd = {k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")}
backbone.load_state_dict(backbone_sd, strict=True)
backbone = backbone.cuda().eval()
for p in backbone.parameters():
    p.requires_grad = False

# Load projector for post-projector evaluation (BYI paper convention, Section C.4)
projector = nn.Sequential(
    nn.Linear(feat_dim, args.proj_hidden),
    nn.ReLU(),
    nn.Linear(args.proj_hidden, args.proj_output),
)
projector_sd = {k.replace("projector.", ""): v for k, v in sd.items() if k.startswith("projector.")}
projector.load_state_dict(projector_sd, strict=True)
projector = projector.cuda().eval()
for p in projector.parameters():
    p.requires_grad = False

print(f"Loaded backbone from: {args.ckpt}")

# ── Data ──────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

train_loader = DataLoader(
    datasets.CIFAR100(args.data_path, train=True,  download=True, transform=transform),
    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,
)
val_loader = DataLoader(
    datasets.CIFAR100(args.data_path, train=False, download=True, transform=transform),
    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,
)

# ── Feature extraction ────────────────────────────────────────────────────────
def extract(loader):
    bb_feats, proj_feats, labels = [], [], []
    with torch.no_grad():
        for x, y in loader:
            h = backbone(x.cuda())
            z = projector(h)
            if apply_relu:
                h = F.relu(h)
                z = F.relu(z)  # matching BYI eval: relu only, no normalization
            bb_feats.append(h.cpu())
            proj_feats.append(z.cpu())
            labels.append(y)
    return torch.cat(bb_feats), torch.cat(proj_feats), torch.cat(labels)

print("Extracting features (one-time)...")
train_bb, train_proj, train_labels = extract(train_loader)
val_bb,   val_proj,   val_labels   = extract(val_loader)
print(f"  backbone  — train: {train_bb.shape}, val: {val_bb.shape}")
print(f"  projector — train: {train_proj.shape}, val: {val_proj.shape}")

# ── Label noise ────────────────────────────────────────────────────────────────
import numpy as np
num_classes = int(train_labels.max().item()) + 1
print(f"  num_classes: {num_classes}")

noisy_labels = train_labels.clone()
if args.noise > 0:
    n = len(noisy_labels)
    n_corrupt = int(n * args.noise)
    idx = torch.randperm(n)[:n_corrupt]
    for i in idx.tolist():
        orig = noisy_labels[i].item()
        choices = [c for c in range(num_classes) if c != orig]
        noisy_labels[i] = int(np.random.choice(choices))
    print(f"  Applied {args.noise*100:.0f}% symmetric label noise ({n_corrupt} labels corrupted)")

# ── Linear probe helper ────────────────────────────────────────────────────────
def run_linear_probe(train_feats, val_feats, train_labels, val_labels, label="backbone"):
    in_dim = train_feats.shape[1]
    clf = nn.Linear(in_dim, num_classes).cuda()
    opt = LARS(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay,
               clip_lr=True, eta=0.02, exclude_bias_n_norm=True)

    steps_per_epoch = math.ceil(len(train_feats) / args.batch_size)
    total_steps  = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def cosine_lr(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=cosine_lr)

    def accuracy(feats, labels):
        clf.eval()
        with torch.no_grad():
            logits = clf(feats.cuda())
            preds  = logits.argmax(dim=1)
            return (preds == labels.cuda()).float().mean().item() * 100

    train_gpu  = train_feats.cuda()
    noisy_gpu  = noisy_labels.cuda()

    print(f"\nTraining linear classifier [{label}] for {args.epochs} epochs (LARS + warmup cosine)...")
    for epoch in range(1, args.epochs + 1):
        clf.train()
        perm = torch.randperm(len(train_gpu))
        for i in range(0, len(perm), args.batch_size):
            idx    = perm[i:i + args.batch_size]
            logits = clf(train_gpu[idx])
            loss   = F.cross_entropy(logits, noisy_gpu[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if epoch % 10 == 0 or epoch == args.epochs:
            val_acc = accuracy(val_feats, val_labels)
            print(f"  Epoch {epoch:3d}/{args.epochs} | val acc: {val_acc:.2f}%")

    train_acc = accuracy(train_feats, train_labels)
    val_acc   = accuracy(val_feats,   val_labels)

    print(f"\n{'='*50}")
    print(f"Linear Evaluation [{label}] (CIFAR-100, noise={args.noise})")
    print(f"  Checkpoint : {args.ckpt}")
    print(f"  Train acc  : {train_acc:.2f}%")
    print(f"  Val acc    : {val_acc:.2f}%")
    print(f"{'='*50}")

# Run on backbone features (NCL paper convention)
run_linear_probe(train_bb, val_bb, train_labels, val_labels,
                 label="backbone (pre-projector, NCL convention)")

# Run on projector features (BYI paper convention, Section C.4)
run_linear_probe(train_proj, val_proj, train_labels, val_labels,
                 label="projector (post-projector, BYI convention)")
