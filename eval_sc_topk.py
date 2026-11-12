"""
Top-K neuron CCR test — checks if CCR increase with batch size is a dead neuron
selection artifact or genuine monosemanticity improvement.

For each checkpoint, computes CCR using only the top-K most active neurons
(by total activation magnitude across the dataset), fixing K across batch sizes.

Usage:
    python eval_sc_topk.py --ckpt /path/to/ckpt.ckpt --non_neg relu --top_k 721
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",       required=True)
parser.add_argument("--non_neg",    default="none", choices=["none", "relu", "rep_relu", "softplus"])
parser.add_argument("--top_k",      required=True, type=int, help="Number of top active neurons to use")
parser.add_argument("--data_path",  default="./data")
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--num_workers",default=4,   type=int)
parser.add_argument("--proj_hidden",default=16384, type=int)
parser.add_argument("--proj_output",default=2048,  type=int)
parser.add_argument("--eps",        default=1e-5, type=float)
args = parser.parse_args()


class SimCLRModel(nn.Module):
    def __init__(self, proj_hidden=16384, proj_output=2048):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        self.backbone.maxpool = nn.Identity()
        feat_dim = self.backbone.fc.in_features
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
sd = ckpt["state_dict"]
model.backbone.load_state_dict({k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")}, strict=True)
model.projector.load_state_dict({k.replace("projector.", ""): v for k, v in sd.items() if k.startswith("projector.")}, strict=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
])

loader = DataLoader(
    datasets.CIFAR100(args.data_path, train=False, download=True, transform=transform),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=True,
)


def apply_activation(z):
    if args.non_neg == "none":
        return F.normalize(z, dim=1)
    z = F.normalize(z, dim=1)
    if args.non_neg in ("relu", "rep_relu"):
        z = F.relu(z)
    elif args.non_neg == "softplus":
        z = F.softplus(z)
    return F.normalize(z, dim=1)


feats, labels = [], []
with torch.no_grad():
    for x, y in loader:
        z = model(x.to(device))
        z = apply_activation(z)
        feats.append(z.cpu())
        labels.append(y)
feats  = torch.cat(feats)
labels = torch.cat(labels)

# All active neurons
active_mask = feats.abs().sum(0) > 0
all_active = int(active_mask.sum().item())
total_dead  = feats.shape[1] - all_active

# Select top-K by total activation magnitude
total_activity = feats.abs().sum(0)
top_k_idx = torch.topk(total_activity, min(args.top_k, all_active)).indices
feats_topk = feats[:, top_k_idx]

print(f"Checkpoint : {args.ckpt}")
print(f"non_neg    : {args.non_neg}")
print(f"Total dims : {feats.shape[1]}  |  All active: {all_active}  |  Dead: {total_dead}")
print(f"Top-K      : {args.top_k} (using {feats_topk.shape[1]} neurons)")

scores = []
for d in range(feats_topk.shape[1]):
    mask = feats_topk[:, d].abs() > args.eps
    if mask.sum() == 0:
        continue
    dist = labels[mask].bincount()
    dist = dist / dist.sum().float()
    scores.append(dist.max().item())

scores = torch.tensor(scores) * 100
print(f"\nCCR (top-{args.top_k}): {scores.mean():.2f}% ± {scores.std():.2f}%")
print(f"Active neurons in top-K: {len(scores)}")
