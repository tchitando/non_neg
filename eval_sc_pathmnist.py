"""
CCR (Semantic Consistency) evaluation for SimCLR / NCL checkpoints on PathMNIST.

Usage:
    python eval_sc_pathmnist.py --ckpt trained_models/pathmnist/simclr/offline-xxx/pathmnist-ncl-resnet18-A0-offline-xxx-ep=99.ckpt --non_neg rep_relu
    python eval_sc_pathmnist.py --ckpt trained_models/pathmnist/simclr/offline-xxx/pathmnist-simclr-resnet18-A0-offline-xxx-ep=99.ckpt --non_neg none
"""
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import DataLoader

PATH_MEAN = (0.7406, 0.5331, 0.7059)
PATH_STD  = (0.1292, 0.1765, 0.1381)
N_CLASSES  = 9
CHANCE     = 100.0 / N_CLASSES  # 11.11%

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",        required=True,  help="Path to .ckpt file")
parser.add_argument("--non_neg",     default="none", help="none | rep_relu")
parser.add_argument("--split",       default="test", help="train | val | test")
parser.add_argument("--n_images",    default=None,   type=int)
parser.add_argument("--batch_size",  default=256,    type=int)
parser.add_argument("--num_workers", default=0,      type=int)
parser.add_argument("--proj_hidden", default=2048,   type=int)
parser.add_argument("--proj_output", default=256,    type=int)
parser.add_argument("--data_root",   default="./MedMNIST")
args = parser.parse_args()

print(f"non_neg: {args.non_neg}")

# ── Dataset ───────────────────────────────────────────────────────────────────
class PathDataset(torch.utils.data.Dataset):
    def __init__(self, split, transform):
        try:
            sys.path.insert(0, "MedMNIST/pathmnist")
            from dataset import PathMNISTDataset
            self._ds = PathMNISTDataset(split=split, transform=transform)
            self._custom = True
        except Exception:
            from medmnist import PathMNIST
            self._ds = PathMNIST(split=split, transform=transform,
                                 download=False, as_rgb=True, root=args.data_root,
                                 size=64)
            self._custom = False

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        img, label = self._ds[idx]
        label = int(label[0]) if hasattr(label, '__len__') else int(label)
        return img, label

# ── Model — ResNet-50 backbone + 2-layer projector ────────────────────────────
class SimCLRModel(nn.Module):
    def __init__(self, proj_hidden=2048, proj_output=256):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        feat_dim = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, proj_hidden),
            nn.ReLU(),
            nn.Linear(proj_hidden, proj_output),
        )

    def forward(self, x):
        return self.projector(self.backbone(x))

model = SimCLRModel(args.proj_hidden, args.proj_output)

ckpt = torch.load(args.ckpt, map_location="cpu")
sd   = ckpt["state_dict"]
model.backbone.load_state_dict(
    {k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")},
    strict=True)
model.projector.load_state_dict(
    {k.replace("projector.", ""): v for k, v in sd.items() if k.startswith("projector.")},
    strict=True)
model = model.cuda().eval()
print(f"Loaded: {args.ckpt}")

# ── Data ──────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=PATH_MEAN, std=PATH_STD),
])

def make_loader(split, n_images=None):
    ds = PathDataset(split=split, transform=transform)
    if n_images is not None and n_images < len(ds):
        ds = torch.utils.data.Subset(ds, list(range(n_images)))
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.num_workers, pin_memory=True)

# ── Extraction ─────────────────────────────────────────────────────────────────
def extract(loader):
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            z = model(x.cuda())
            if args.non_neg in ("relu", "rep_relu"):
                z = F.relu(z)
            feats.append(z.cpu())
            labels.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
    return torch.cat(feats), torch.cat(labels)

# ── CCR (Semantic Consistency) ─────────────────────────────────────────────────
def ccr(features, labels, eps=1e-5):
    total_dims = features.shape[1]
    # For NCL (non-negative), sum > 0 correctly identifies active neurons.
    # For SimCLR (signed), use abs().sum() to avoid falsely marking neurons dead.
    if args.non_neg in ("relu", "rep_relu"):
        alive_mask = features.sum(0) > 0
    else:
        alive_mask = features.abs().sum(0) > 0
    features    = features[:, alive_mask]
    active_dims = features.shape[1]
    features = F.normalize(features, dim=1)
    scores = []
    for d in range(active_dims):
        mask = features.abs()[:, d] > eps
        lbl  = labels[mask]
        try:
            dist = lbl.bincount(minlength=N_CLASSES)
            dist = dist / dist.sum()
            scores.append(dist.max().item())
        except Exception:
            pass
    scores = torch.tensor(scores) * 100
    return len(scores), active_dims, total_dims, scores.mean().item(), scores.std().item()

# ── Run ────────────────────────────────────────────────────────────────────────
print(f"\nExtracting {args.split} features (n_images={args.n_images or 'all'})...")
feats, labels = extract(make_loader(args.split, args.n_images))
active, active_dims, total_dims, mean_sc, std_sc = ccr(feats, labels)
print(f"  CCR / Semantic Consistency: {mean_sc:.2f}% ± {std_sc:.2f}%")
print(f"  Active neurons: {active_dims} / {total_dims}  (dead: {total_dims - active_dims})")
print(f"  Random baseline: {CHANCE:.2f}%  (1/{N_CLASSES} classes)")
