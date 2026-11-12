"""
CCR (Semantic Consistency) evaluation for SimCLR / NCL checkpoints on PneumoniaMNIST.

Usage:
    python eval_sc_pneumoniamnist.py --ckpt trained_models/simclr/offline-n37mk7o6/simclr-resnet18-pneumoniamnist-200ep-offline-n37mk7o6-ep=199.ckpt --non_neg none
    python eval_sc_pneumoniamnist.py --ckpt trained_models/simclr/offline-w7z99kq0/ncl-resnet18-pneumoniamnist-200ep-offline-w7z99kq0-ep=199.ckpt --non_neg relu
"""
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from MedMNIST.pneumoniamnist_dataset import PneumoniaMnistDataset, PNEUMONIAMNIST_MEAN, PNEUMONIAMNIST_STD

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",        required=True,  help="Path to .ckpt file")
parser.add_argument("--non_neg",     default="none", help="Activation on projector output: none, relu, rep_relu, softplus")
parser.add_argument("--batch_size",  default=256, type=int)
parser.add_argument("--num_workers", default=0,   type=int)
parser.add_argument("--proj_hidden", default=2048, type=int, help="Projector hidden dim")
parser.add_argument("--proj_output", default=256,  type=int, help="Projector output dim")
args = parser.parse_args()

assert args.non_neg in ("none", "relu", "rep_relu", "softplus"), \
    f"--non_neg must be one of: none, relu, rep_relu, softplus"
print(f"non_neg activation: {args.non_neg}")

# ── Model ─────────────────────────────────────────────────────────────────────
class SimCLRModel(nn.Module):
    def __init__(self, proj_hidden=2048, proj_output=256):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(feat_dim, proj_hidden),
            nn.ReLU(),
            nn.Linear(proj_hidden, proj_output),
        )

    def forward(self, x):
        feats = self.backbone(x)
        z     = self.projector(feats)
        return z

model = SimCLRModel(args.proj_hidden, args.proj_output)

ckpt = torch.load(args.ckpt, map_location="cpu")
sd   = ckpt["state_dict"]
backbone_sd  = {k.replace("backbone.",  ""): v for k, v in sd.items() if k.startswith("backbone.")}
projector_sd = {k.replace("projector.", ""): v for k, v in sd.items() if k.startswith("projector.")}

model.backbone.load_state_dict(backbone_sd,  strict=True)
model.projector.load_state_dict(projector_sd, strict=True)
model = model.cuda()
print(f"Loaded: {args.ckpt}")

# ── Data ──────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=PNEUMONIAMNIST_MEAN, std=PNEUMONIAMNIST_STD),
])

def make_loader(split):
    ds = PneumoniaMnistDataset(split=split, transform=transform)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.num_workers, pin_memory=True)

# ── Extraction ─────────────────────────────────────────────────────────────────
def extract(loader):
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            z = model(x.cuda())
            if args.non_neg != "none":
                z = F.normalize(z, dim=-1)
            if args.non_neg == "relu" or args.non_neg == "rep_relu":
                z = F.relu(z)
            elif args.non_neg == "softplus":
                z = F.softplus(z)
            feats.append(z.cpu())
            labels.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
    return torch.cat(feats), torch.cat(labels)

# ── Metrics ───────────────────────────────────────────────────────────────────
def ccr(features, labels, eps=1e-5):
    total_dims = features.shape[1]
    features = features[:, F.relu(features).sum(0) > 0]  # remove globally dead dims
    active_dims = features.shape[1]
    features = F.normalize(features, dim=1)
    scores = []
    for d in range(active_dims):
        mask = features.abs()[:, d] > eps
        labels_selected = labels[mask]
        try:
            dist = labels_selected.bincount()
            dist = dist / dist.sum()
            scores.append(dist.max().item())
        except:
            pass
    scores = torch.tensor(scores) * 100
    return len(scores), active_dims, total_dims, scores.mean().item(), scores.std().item()

# ── Run ────────────────────────────────────────────────────────────────────────
print(f"\nExtracting test features...")
feats, labels = extract(make_loader("test"))
active, active_dims, total_dims, mean, std = ccr(feats, labels)
print(f"  CCR / Semantic Consistency: {mean:.2f}% ± {std:.2f}%")
print(f"  Active neurons: {active_dims} / {total_dims}  (dead: {total_dims - active_dims})")
maj = labels.bincount().float() / len(labels) * 100
print(f"  Class balance: {maj.tolist()}  (majority class baseline: {maj.max():.2f}%)")

# Balanced subset (equal samples per class)
n_min = labels.bincount().min().item()
idx = torch.cat([torch.where(labels == c)[0][:n_min] for c in range(labels.bincount().shape[0])])
b_feats, b_labels = feats[idx], labels[idx]
_, _, _, b_mean, b_std = ccr(b_feats, b_labels)
print(f"\n  [Balanced {n_min}+{n_min}] CCR: {b_mean:.2f}% ± {b_std:.2f}%  (baseline: 50.00%)")
