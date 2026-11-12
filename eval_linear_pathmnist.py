"""
Linear evaluation for SimCLR / NCL checkpoints on PathMNIST (9 classes).
Backbone frozen. Reports backbone and projector linear probe accuracy + AUC.

Usage:
    python eval_linear_pathmnist.py --ckpt trained_models/pathmnist/simclr/offline-xxx/pathmnist-ncl-resnet18-A0-offline-xxx-ep=99.ckpt --ncl true
    python eval_linear_pathmnist.py --ckpt trained_models/pathmnist/simclr/offline-xxx/pathmnist-simclr-resnet18-A0-offline-xxx-ep=99.ckpt --ncl false
"""
import argparse, sys, math, random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models

sys.path.insert(0, ".")
from solo.utils.lars import LARS

PATH_MEAN = (0.7406, 0.5331, 0.7059)
PATH_STD  = (0.1292, 0.1765, 0.1381)
N_CLASSES  = 9

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",         required=True)
parser.add_argument("--ncl",          default="auto", help="true | false | auto")
parser.add_argument("--epochs",       default=50,   type=int)
parser.add_argument("--lr",           default=0.1,  type=float)
parser.add_argument("--weight_decay", default=1e-4, type=float)
parser.add_argument("--warmup_epochs",default=10,   type=int)
parser.add_argument("--batch_size",   default=256,  type=int)
parser.add_argument("--num_workers",  default=0,    type=int)
parser.add_argument("--proj_hidden",  default=2048, type=int)
parser.add_argument("--proj_output",  default=256,  type=int)
parser.add_argument("--seed",         default=42,   type=int)
parser.add_argument("--proj_only",    action="store_true")
parser.add_argument("--data_root",    default="./MedMNIST")
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

apply_relu = ("ncl" in args.ckpt.lower()) if args.ncl.lower() == "auto" else (args.ncl.lower() == "true")
print(f"NCL mode (apply ReLU to projector): {apply_relu}")

# ── Dataset ───────────────────────────────────────────────────────────────────
class PathDataset(torch.utils.data.Dataset):
    def __init__(self, split, transform):
        try:
            sys.path.insert(0, "MedMNIST/pathmnist")
            from dataset import PathMNISTDataset
            self._ds = PathMNISTDataset(split=split, transform=transform)
        except Exception:
            from medmnist import PathMNIST
            self._ds = PathMNIST(split=split, transform=transform,
                                 download=False, as_rgb=True, root=args.data_root,
                                 size=64)

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        img, label = self._ds[idx]
        label = int(label[0]) if hasattr(label, '__len__') else int(label)
        return img, label

# ── Backbone + Projector — ResNet-50, feat_dim=2048 ───────────────────────────
backbone = models.resnet50(weights=None)
feat_dim = backbone.fc.in_features  # 2048
backbone.fc = nn.Identity()

ckpt = torch.load(args.ckpt, map_location="cpu")
sd   = ckpt["state_dict"]
backbone.load_state_dict(
    {k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")},
    strict=True)
backbone = backbone.cuda().eval()
for p in backbone.parameters():
    p.requires_grad = False

projector = nn.Sequential(
    nn.Linear(feat_dim, args.proj_hidden),
    nn.ReLU(),
    nn.Linear(args.proj_hidden, args.proj_output),
)
projector.load_state_dict(
    {k.replace("projector.", ""): v for k, v in sd.items() if k.startswith("projector.")},
    strict=True)
projector = projector.cuda().eval()
for p in projector.parameters():
    p.requires_grad = False

print(f"Loaded: {args.ckpt}")

# ── Data ──────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=PATH_MEAN, std=PATH_STD),
])

def make_loader(split, shuffle=False):
    ds = PathDataset(split=split, transform=transform)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, pin_memory=True)

train_loader = make_loader("train", shuffle=True)
val_loader   = make_loader("val")
test_loader  = make_loader("test")

# ── Feature extraction ────────────────────────────────────────────────────────
def extract(loader):
    bb_feats, proj_feats, labels = [], [], []
    with torch.no_grad():
        for x, y in loader:
            h = backbone(x.cuda())
            z = projector(h)
            if apply_relu:
                z = F.relu(z)
            bb_feats.append(h.cpu())
            proj_feats.append(z.cpu())
            labels.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
    return torch.cat(bb_feats), torch.cat(proj_feats), torch.cat(labels)

print("Extracting features...")
train_bb, train_proj, train_labels = extract(train_loader)
val_bb,   val_proj,   val_labels   = extract(val_loader)
test_bb,  test_proj,  test_labels  = extract(test_loader)
print(f"  backbone  — train: {train_bb.shape}, val: {val_bb.shape}, test: {test_bb.shape}")
print(f"  projector — train: {train_proj.shape}, val: {val_proj.shape}, test: {test_proj.shape}")

# ── Linear probe ──────────────────────────────────────────────────────────────
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

def run_linear_probe(train_feats, val_feats, test_feats,
                     train_lbl, val_lbl, test_lbl, label="backbone"):
    in_dim = train_feats.shape[1]
    clf = nn.Linear(in_dim, N_CLASSES).cuda()
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

    def evaluate(feats, lbl):
        clf.eval()
        with torch.no_grad():
            logits = clf(feats.cuda())
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
        lbl_np  = lbl.numpy()
        acc     = (preds == lbl_np).mean() * 100
        bal_acc = balanced_accuracy_score(lbl_np, preds) * 100
        auc     = roc_auc_score(lbl_np, probs, multi_class="ovr", average="macro") * 100
        return acc, bal_acc, auc

    train_gpu = train_feats.cuda()
    lbl_gpu   = train_lbl.cuda()

    print(f"\nTraining linear classifier [{label}] for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        clf.train()
        perm = torch.randperm(len(train_gpu))
        for i in range(0, len(perm), args.batch_size):
            idx    = perm[i:i + args.batch_size]
            logits = clf(train_gpu[idx])
            loss   = F.cross_entropy(logits, lbl_gpu[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if epoch % 10 == 0 or epoch == args.epochs:
            _, val_bal, _ = evaluate(val_feats, val_lbl)
            print(f"  Epoch {epoch:3d}/{args.epochs} | val bal acc: {val_bal:.2f}%")

    tr = evaluate(train_feats, train_lbl)
    va = evaluate(val_feats,   val_lbl)
    te = evaluate(test_feats,  test_lbl)

    print(f"\n{'='*60}")
    print(f"Linear Evaluation [{label}] (PathMNIST, 9 classes)")
    print(f"{'Split':<8} {'Acc':>8} {'Bal Acc':>10} {'AUC':>8}")
    print(f"{'-'*36}")
    print(f"{'Train':<8} {tr[0]:>7.2f}% {tr[1]:>9.2f}% {tr[2]:>7.2f}%")
    print(f"{'Val':<8}   {va[0]:>7.2f}% {va[1]:>9.2f}% {va[2]:>7.2f}%")
    print(f"{'Test':<8}  {te[0]:>7.2f}% {te[1]:>9.2f}% {te[2]:>7.2f}%")
    print(f"{'='*60}")

if not args.proj_only:
    run_linear_probe(train_bb, val_bb, test_bb,
                     train_labels, val_labels, test_labels,
                     label="backbone (pre-projector)")

run_linear_probe(train_proj, val_proj, test_proj,
                 train_labels, val_labels, test_labels,
                 label="projector (post-projector)")
