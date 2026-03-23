"""
Standalone CCR (Semantic Consistency) evaluation for SimCLR / NCL checkpoints.
Requirements: torch torchvision
Usage:
    python eval_sc_standalone.py --ckpt path/to/checkpoint.ckpt --ncl true|false
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",        required=True,  help="Path to .ckpt file")
parser.add_argument("--ncl",         required=True,  help="true = apply ReLU (NCL), false = SimCLR")
parser.add_argument("--data_path",   default="./data")
parser.add_argument("--batch_size",  default=256, type=int)
parser.add_argument("--num_workers", default=4,   type=int)
parser.add_argument("--proj_hidden", default=2048, type=int, help="Projector hidden dim")
parser.add_argument("--proj_output", default=256,  type=int, help="Projector output dim")
args = parser.parse_args()

apply_relu = args.ncl.lower() == "true"
print(f"NCL mode (apply ReLU): {apply_relu}")

# ── Model ─────────────────────────────────────────────────────────────────────
class SimCLRModel(nn.Module):
    def __init__(self, proj_hidden=2048, proj_output=256):
        super().__init__()
        # CIFAR ResNet18: replace first conv + remove maxpool
        self.backbone = models.resnet18(weights=None)
        self.backbone.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
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
        z     = self.projector(feats)
        return z

model = SimCLRModel(args.proj_hidden, args.proj_output)

# Load checkpoint — strip the "backbone." / "projector." prefixes from solo's state dict
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
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
])

def make_loader(train):
    ds = datasets.CIFAR100(args.data_path, train=train, download=True, transform=transform)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.num_workers, pin_memory=True)

# ── Extraction ────────────────────────────────────────────────────────────────
def extract(loader):
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            z = model(x.cuda())
            if apply_relu:
                z = F.relu(z)
            feats.append(z.cpu())
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)

# ── Metrics ───────────────────────────────────────────────────────────────────
def sparsity(f, eps=1e-2):
    return (f.abs() < eps).float().mean().item() * 100

def ccr(features, labels, eps=1e-5):
    features = features[:, F.relu(features).sum(0) > 0]  # filter inactive dims before normalizing
    features = F.normalize(features, dim=1)
    D = features.shape[1]
    scores = []
    for d in range(D):
        mask = features.abs()[:, d] > eps
        labels_selected = labels[mask]
        try:
            dist = labels_selected.bincount()
            dist = dist / dist.sum()
            scores.append(dist.max().item())
        except:
            pass
    scores = torch.tensor(scores) * 100
    return len(scores), D, scores.mean().item(), scores.std().item()

# ── Run ───────────────────────────────────────────────────────────────────────
for split, train in [("test", False)]:
    print(f"\nExtracting {split} features...")
    feats, labels = extract(make_loader(train))
    active, total, mean, std = ccr(feats, labels)
    print(f"  CCR / Semantic Consistency: {mean:.2f}%")
