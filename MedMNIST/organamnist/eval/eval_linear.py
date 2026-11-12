"""
Linear evaluation for SimCLR / NCL checkpoints on OrganAMNIST.

Freezes the pretrained backbone (ResNet-50), trains a linear classifier,
and reports train / val / test balanced accuracy and AUC.

Usage:
    python MedMNIST/organamnist/eval/eval_linear.py \
        --ckpt trained_models/organamnist/simclr/offline-3fc0no59/organamnist-simclr-resnet50-A0-offline-3fc0no59-ep=99.ckpt

    python MedMNIST/organamnist/eval/eval_linear.py \
        --ckpt trained_models/organamnist/simclr/offline-248gfko5/organamnist-ncl-resnet50-A0-offline-248gfko5-ep=99.ckpt \
        --ncl true
"""
import argparse, math, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # organamnist/

from solo.utils.lars import LARS
from dataset import OrganAMNISTDataset, ORGANAMNIST_MEAN, ORGANAMNIST_STD

N_CLASSES = 11

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",          required=True)
parser.add_argument("--ncl",           default="auto", help="true|false|auto")
parser.add_argument("--epochs",        default=50,   type=int)
parser.add_argument("--lr",            default=0.1,  type=float)
parser.add_argument("--weight_decay",  default=1e-4, type=float)
parser.add_argument("--warmup_epochs", default=10,   type=int)
parser.add_argument("--batch_size",    default=256,  type=int)
parser.add_argument("--num_workers",   default=4,    type=int)
parser.add_argument("--proj_hidden",   default=2048, type=int)
parser.add_argument("--proj_output",   default=256,  type=int)
parser.add_argument("--knn_k",         default=[5, 20, 100], nargs="+", type=int)
parser.add_argument("--no_knn",        action="store_true")
parser.add_argument("--seed",          default=42,   type=int)
args = parser.parse_args()

torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed); random.seed(args.seed)

apply_relu = ("ncl" in Path(args.ckpt).stem.lower()) if args.ncl == "auto" else (args.ncl.lower() == "true")
print(f"NCL mode (apply ReLU to projector output): {apply_relu}")

# ── Backbone + Projector ──────────────────────────────────────────────────────
backbone = models.resnet50(weights=None)
feat_dim = backbone.fc.in_features   # 2048 for ResNet-50
backbone.fc = nn.Identity()

sd = torch.load(args.ckpt, map_location="cpu")["state_dict"]
backbone.load_state_dict({k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")}, strict=True)
backbone = backbone.cuda().eval()
for p in backbone.parameters(): p.requires_grad = False

projector = nn.Sequential(
    nn.Linear(feat_dim, args.proj_hidden), nn.ReLU(),
    nn.Linear(args.proj_hidden, args.proj_output),
)
projector.load_state_dict({k.replace("projector.", ""): v for k, v in sd.items() if k.startswith("projector.")}, strict=True)
projector = projector.cuda().eval()
for p in projector.parameters(): p.requires_grad = False

print(f"Loaded: {args.ckpt}")

# ── Data ──────────────────────────────────────────────────────────────────────
norm_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(ORGANAMNIST_MEAN, ORGANAMNIST_STD),
])

def make_loader(split, shuffle=False):
    ds = OrganAMNISTDataset(split=split, transform=norm_tf)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, pin_memory=True)

train_loader = make_loader("train", shuffle=True)
val_loader   = make_loader("val")
test_loader  = make_loader("test")

# ── Feature extraction ─────────────────────────────────────────────────────────
def extract(loader):
    bb_feats, proj_feats, lbls = [], [], []
    with torch.no_grad():
        for x, y in loader:
            h = backbone(x.cuda())
            z = projector(h)
            if apply_relu: z = F.relu(z)
            bb_feats.append(h.cpu()); proj_feats.append(z.cpu()); lbls.append(y)
    return torch.cat(bb_feats), torch.cat(proj_feats), torch.cat(lbls)

print("Extracting features...")
train_bb, train_proj, train_labels = extract(train_loader)
val_bb,   val_proj,   val_labels   = extract(val_loader)
test_bb,  test_proj,  test_labels  = extract(test_loader)
print(f"  backbone  — train: {train_bb.shape}, val: {val_bb.shape}, test: {test_bb.shape}")
print(f"  projector — train: {train_proj.shape}, val: {val_proj.shape}, test: {test_proj.shape}")

# ── Linear probe ──────────────────────────────────────────────────────────────
def evaluate(clf, feats, labels):
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


def run_linear_probe(train_feats, val_feats, test_feats, label="backbone"):
    in_dim = train_feats.shape[1]
    clf = nn.Linear(in_dim, N_CLASSES).cuda()
    opt = LARS(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay,
               clip_lr=True, eta=0.02, exclude_bias_n_norm=True)

    steps_per_epoch = math.ceil(len(train_feats) / args.batch_size)
    total_steps  = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_fn(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_fn)
    train_gpu = train_feats.cuda()
    labels_gpu = train_labels.cuda()

    print(f"\nTraining linear classifier [{label}] for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        clf.train()
        perm = torch.randperm(len(train_gpu))
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            loss = F.cross_entropy(clf(train_gpu[idx]), labels_gpu[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if epoch % 10 == 0 or epoch == args.epochs:
            _, vb, _ = evaluate(clf, val_feats, val_labels)
            print(f"  Epoch {epoch:3d}/{args.epochs} | val bal acc: {vb:.2f}%")

    tr = evaluate(clf, train_feats, train_labels)
    va = evaluate(clf, val_feats,   val_labels)
    te = evaluate(clf, test_feats,  test_labels)

    print(f"\n{'='*60}")
    print(f"Linear Evaluation [{label}] (OrganAMNIST)")
    print(f"{'Split':<8} {'Acc':>8} {'Bal Acc':>10} {'AUC':>8}")
    print(f"{'-'*38}")
    print(f"{'Train':<8} {tr[0]:>7.2f}% {tr[1]:>9.2f}% {tr[2]:>7.2f}%")
    print(f"{'Val':<8} {va[0]:>7.2f}% {va[1]:>9.2f}% {va[2]:>7.2f}%")
    print(f"{'Test':<8} {te[0]:>7.2f}% {te[1]:>9.2f}% {te[2]:>7.2f}%")
    print(f"{'='*60}")


run_linear_probe(train_bb,   val_bb,   test_bb,   label="backbone (pre-projector)")
run_linear_probe(train_proj, val_proj, test_proj, label="projector (post-projector)")

# ── k-NN evaluation ────────────────────────────────────────────────────────────
if not args.no_knn:
    print("\nRunning k-NN evaluation...")
    train_norm = F.normalize(train_bb, dim=1).cuda()
    val_norm   = F.normalize(val_bb,   dim=1).cuda()
    test_norm  = F.normalize(test_bb,  dim=1).cuda()
    train_knn_labels = train_labels.cuda()

    def knn_predict(query, bank, bank_labels, k):
        sim = query @ bank.T
        topk_idx = sim.topk(k, dim=1).indices
        topk_labels = bank_labels[topk_idx]
        scores = torch.zeros(query.size(0), N_CLASSES, device=query.device)
        scores.scatter_add_(1, topk_labels, sim.topk(k, dim=1).values)
        return scores.argmax(dim=1)

    for k in args.knn_k:
        vp  = knn_predict(val_norm,  train_norm, train_knn_labels, k).cpu().numpy()
        tp  = knn_predict(test_norm, train_norm, train_knn_labels, k).cpu().numpy()
        vba = balanced_accuracy_score(val_labels.numpy(),  vp) * 100
        tba = balanced_accuracy_score(test_labels.numpy(), tp) * 100
        print(f"  k={k:>3} | val bal: {vba:.2f}% | test bal: {tba:.2f}%")
