"""
Linear evaluation for SimCLR / NCL checkpoints on OCT-MNIST.

Freezes the pretrained backbone, trains a linear classifier on top,
and reports train / val / test accuracy.

Usage:
    python eval_linear_octmnist.py --ckpt trained_models/simclr/25/simclr-resnet18-octmnist-25-ep=199.ckpt
    python eval_linear_octmnist.py --ckpt trained_models/simclr/25/simclr-resnet18-octmnist-25-ep=199.ckpt --ncl true
"""
import argparse, json, sys, math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models

sys.path.insert(0, ".")
from solo.utils.lars import LARS

sys.path.insert(0, ".")
from MedMNIST.octmnist_dataset import OCTMnistDataset, OCTMNIST_MEAN, OCTMNIST_STD

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",        required=True)
parser.add_argument("--ncl",         default="auto", help="true|false|auto")
parser.add_argument("--epochs",       default=50,   type=int)
parser.add_argument("--lr",           default=0.1,  type=float)
parser.add_argument("--weight_decay", default=1e-4, type=float)
parser.add_argument("--warmup_epochs",default=10,   type=int)
parser.add_argument("--batch_size",   default=256,  type=int)
parser.add_argument("--num_workers", default=0,   type=int)
parser.add_argument("--noise",        default=0.0, type=float, help="symmetric label noise rate [0,1)")
parser.add_argument("--knn_k",        default=[5, 20, 100], nargs="+", type=int)
parser.add_argument("--no_knn",       action="store_true",  help="Skip k-NN evaluation")
parser.add_argument("--proj_only",    action="store_true",  help="Only run linear probe on projector features (BYI convention)")
parser.add_argument("--proj_hidden",  default=2048, type=int, help="Projector hidden dim")
parser.add_argument("--proj_output",  default=256,  type=int, help="Projector output dim")
parser.add_argument("--seed",         default=42,   type=int, help="Random seed for reproducibility")
parser.add_argument("--backbone",     default="auto", help="resnet18|resnet50|auto (auto detects from ckpt name)")
args = parser.parse_args()

import random, numpy as np
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

apply_relu = ("ncl" in args.ckpt.lower()) if args.ncl.lower() == "auto" else (args.ncl.lower() == "true")
print(f"NCL mode (apply ReLU to backbone features): {apply_relu}")

# ── Backbone + Projector ──────────────────────────────────────────────────────
backbone_name = args.backbone
if backbone_name == "auto":
    backbone_name = "resnet50" if "resnet50" in args.ckpt.lower() else "resnet18"
print(f"Backbone: {backbone_name}")

if backbone_name == "resnet50":
    backbone = models.resnet50(weights=None)
else:
    backbone = models.resnet18(weights=None)
feat_dim = backbone.fc.in_features
backbone.fc = nn.Identity()

ckpt = torch.load(args.ckpt, map_location="cpu")
sd = ckpt["state_dict"]
backbone_sd = {k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")}
backbone.load_state_dict(backbone_sd, strict=True)
backbone = backbone.cuda().eval()
for p in backbone.parameters():
    p.requires_grad = False

# Load projector for post-projector evaluation (following BYI paper, Section C.4)
proj_hidden = args.proj_hidden
proj_output = args.proj_output
projector = nn.Sequential(
    nn.Linear(feat_dim, proj_hidden),
    nn.ReLU(),
    nn.Linear(proj_hidden, proj_output),
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
    transforms.Normalize(mean=OCTMNIST_MEAN, std=OCTMNIST_STD),
])

def make_loader(split, shuffle=False):
    ds = OCTMnistDataset(split=split, transform=transform)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, pin_memory=True)

train_loader = make_loader("train", shuffle=True)
val_loader   = make_loader("val")
test_loader  = make_loader("test")

# ── Feature extraction ────────────────────────────────────────────────────────
def extract(loader):
    backbone_feats, proj_feats, labels = [], [], []
    with torch.no_grad():
        for x, y in loader:
            h = backbone(x.cuda())
            z = projector(h)
            # apply non-neg activation to projector output only (matching training)
            if apply_relu:
                z = F.relu(z)
            backbone_feats.append(h.cpu())
            proj_feats.append(z.cpu())
            labels.append(y)
    return torch.cat(backbone_feats), torch.cat(proj_feats), torch.cat(labels)

print("Extracting features...")
train_bb, train_proj, train_labels = extract(train_loader)
val_bb,   val_proj,   val_labels   = extract(val_loader)
test_bb,  test_proj,  test_labels  = extract(test_loader)
print(f"  backbone — train: {train_bb.shape}, val: {val_bb.shape}, test: {test_bb.shape}")
print(f"  projector — train: {train_proj.shape}, val: {val_proj.shape}, test: {test_proj.shape}")

# ── Label noise ───────────────────────────────────────────────────────────────
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
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
    print(f"Label noise: corrupted {n_corrupt}/{n} labels (rate={args.noise})")

# ── Linear probe helper ────────────────────────────────────────────────────────
def run_linear_probe(train_feats, val_feats, test_feats, train_labels, val_labels, test_labels, label="backbone"):
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

    def evaluate(feats, labels):
        clf.eval()
        with torch.no_grad():
            logits = clf(feats.cuda())
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
        labels_np = labels.numpy()
        acc     = (preds == labels_np).mean() * 100
        bal_acc = balanced_accuracy_score(labels_np, preds) * 100
        auc     = roc_auc_score(labels_np, probs, multi_class="ovr", average="macro") * 100
        return acc, bal_acc, auc

    train_gpu = train_feats.cuda()
    noisy_gpu = noisy_labels.cuda()

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
            _, val_bal, _ = evaluate(val_feats, val_labels)
            print(f"  Epoch {epoch:3d}/{args.epochs} | val bal acc: {val_bal:.2f}%")

    tr = evaluate(train_feats, train_labels)
    va = evaluate(val_feats,   val_labels)
    te = evaluate(test_feats,  test_labels)

    print(f"\n{'='*60}")
    print(f"Linear Evaluation [{label}] (OCT-MNIST, noise={args.noise})")
    print(f"{'Split':<8} {'Acc':>8} {'Bal Acc':>10} {'AUC':>8}")
    print(f"{'-'*36}")
    print(f"{'Train':<8} {tr[0]:>7.2f}% {tr[1]:>9.2f}% {tr[2]:>7.2f}%")
    print(f"{'Val':<8} {va[0]:>7.2f}% {va[1]:>9.2f}% {va[2]:>7.2f}%")
    print(f"{'Test':<8} {te[0]:>7.2f}% {te[1]:>9.2f}% {te[2]:>7.2f}%")
    print(f"{'='*60}")
    return clf, evaluate

# Run on backbone features (NCL paper convention) — skipped if --proj_only
if not args.proj_only:
    classifier, evaluate = run_linear_probe(
        train_bb, val_bb, test_bb,
        train_labels, val_labels, test_labels,
        label="backbone (pre-projector)")

# Run on projector features (BYI paper convention, Section C.4)
run_linear_probe(
    train_proj, val_proj, test_proj,
    train_labels, val_labels, test_labels,
    label="projector (post-projector, BYI convention)")

# ── k-NN evaluation (backbone features) ───────────────────────────────────────
if args.no_knn:
    print(f"\nSkipping k-NN evaluation (--no_knn).")
else:
    print(f"\nRunning k-NN evaluation...")
    train_norm = F.normalize(train_bb, dim=1).cuda()
    val_norm   = F.normalize(val_bb,   dim=1).cuda()
    test_norm  = F.normalize(test_bb,  dim=1).cuda()
    train_knn_labels = train_labels.cuda()

    def knn_predict(query, bank, bank_labels, k, num_classes):
        sim = query @ bank.T
        topk_sim, topk_idx = sim.topk(k, dim=1)
        topk_labels = bank_labels[topk_idx]
        scores = torch.zeros(query.size(0), num_classes, device=query.device)
        scores.scatter_add_(1, topk_labels, topk_sim)
        return scores.argmax(dim=1)

    def knn_evaluate(query, bank, bank_labels, query_labels, k):
        preds = knn_predict(query, bank, bank_labels, k, num_classes).cpu().numpy()
        labels_np = query_labels.numpy()
        acc     = (preds == labels_np).mean() * 100
        bal_acc = balanced_accuracy_score(labels_np, preds) * 100
        return acc, bal_acc

    for k in args.knn_k:
        val_acc_knn,  val_bal_knn  = knn_evaluate(val_norm,  train_norm, train_knn_labels, val_labels,  k)
        test_acc_knn, test_bal_knn = knn_evaluate(test_norm, train_norm, train_knn_labels, test_labels, k)
        print(f"  k={k:>3} | val acc: {val_acc_knn:.2f}% bal: {val_bal_knn:.2f}% | test acc: {test_acc_knn:.2f}% bal: {test_bal_knn:.2f}%")
