"""
Standalone Semantic Consistency / CCR evaluation for SimCLR / NCL checkpoints.

Evaluates SC on PROJECTOR features, not backbone features.

Paper-aligned projector:
    512 -> 16384 -> 2048

Usage examples:
    python eval_sc_TC.py --ckpt /path/to/simclr.ckpt --non_neg none
    python eval_sc_TC.py --ckpt /path/to/ncl.ckpt --non_neg relu
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True, help="Path to .ckpt file")
parser.add_argument(
    "--non_neg",
    default="none",
    choices=["none", "relu", "rep_relu", "softplus"],
    help="Activation applied to projector output before SC computation",
)
parser.add_argument("--data_path", default="./data")
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--proj_hidden", default=16384, type=int, help="Projector hidden dim")
parser.add_argument("--proj_output", default=2048, type=int, help="Projector output dim")
parser.add_argument("--eps", default=1e-5, type=float, help="Activation threshold for SC computation")
parser.add_argument("--top_k", default=None, type=int, help="Use only top-K most active neurons (by total activation magnitude)")
args = parser.parse_args()

print(f"non_neg activation: {args.non_neg}")


class SimCLRModel(nn.Module):
    def __init__(self, proj_hidden=16384, proj_output=2048):
        super().__init__()

        # CIFAR ResNet18 stem
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
        feats = self.backbone(x)
        z = self.projector(feats)
        return z


model = SimCLRModel(args.proj_hidden, args.proj_output)

# Load checkpoint
ckpt = torch.load(args.ckpt, map_location="cpu")
sd = ckpt["state_dict"]

backbone_sd = {
    k.replace("backbone.", ""): v
    for k, v in sd.items()
    if k.startswith("backbone.")
}
projector_sd = {
    k.replace("projector.", ""): v
    for k, v in sd.items()
    if k.startswith("projector.")
}

msg_backbone = model.backbone.load_state_dict(backbone_sd, strict=True)
msg_projector = model.projector.load_state_dict(projector_sd, strict=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

print(f"Loaded checkpoint: {args.ckpt}")
print(f"Using device: {device}")
print(f"Projector dims: 512 -> {args.proj_hidden} -> {args.proj_output}")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
])


def make_loader(train: bool):
    ds = datasets.CIFAR100(
        args.data_path,
        train=train,
        download=True,
        transform=transform,
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def apply_activation(z: torch.Tensor) -> torch.Tensor:
    """Match training sequence: normalize → activation → normalize."""
    if args.non_neg == "none":
        return F.normalize(z, dim=1)
    # Step 1: normalize (matching training forward pass)
    z = F.normalize(z, dim=1)
    # Step 2: activation
    if args.non_neg in ("relu", "rep_relu"):
        z = F.relu(z)
    elif args.non_neg == "softplus":
        z = F.softplus(z)
    # Step 3: re-normalize (matching training_step)
    z = F.normalize(z, dim=1)
    return z


def extract(loader):
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            z = model(x)
            z = apply_activation(z)
            feats.append(z.cpu())
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def sparsity(features: torch.Tensor, eps: float = 1e-2) -> float:
    return (features.abs() < eps).float().mean().item() * 100.0


def semantic_consistency(features: torch.Tensor, labels: torch.Tensor, eps: float = 1e-5, top_k: int = None):
    original_dims = features.shape[1]

    # Keep any dimension with nonzero absolute activity
    active_dim_mask = features.abs().sum(0) > 0
    features = features[:, active_dim_mask]
    active_dims = int(active_dim_mask.sum().item())

    # Optionally restrict to top-K most active neurons
    if top_k is not None and top_k < active_dims:
        total_activity = features.abs().sum(0)
        top_k_idx = torch.topk(total_activity, top_k).indices
        features = features[:, top_k_idx]
        active_dims = top_k

    scores = []
    for d in range(features.shape[1]):
        active_sample_mask = features.abs()[:, d] > eps
        labels_selected = labels[active_sample_mask]

        if len(labels_selected) == 0:
            continue

        dist = labels_selected.bincount()
        dist = dist / dist.sum()
        scores.append(dist.max().item())

    if len(scores) == 0:
        return original_dims, active_dims, float("nan"), float("nan")

    scores = torch.tensor(scores) * 100.0
    return original_dims, active_dims, scores.mean().item(), scores.std().item()


for split, train in [("test", False)]:
    print(f"\nExtracting {split} features...")
    feats, labels = extract(make_loader(train))

    total_dims, active_dims, mean_sc, std_sc = semantic_consistency(feats, labels, eps=args.eps)
    sp = sparsity(feats)

    print(f"  Feature tensor shape: {tuple(feats.shape)}")
    print(f"  Activation threshold (eps): {args.eps}")
    print(f"  CCR / Semantic Consistency: {mean_sc:.2f}% ± {std_sc:.2f}%")
    print(f"  Active neurons: {active_dims} / {total_dims}  (dead: {total_dims - active_dims})")
    print(f"  Sparsity (<1e-2): {sp:.2f}%")