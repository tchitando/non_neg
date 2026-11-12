"""
Balanced SC (Semantic Consistency / CCR) for all 3 datasets × 4 methods.

Balances the test split to equal class counts before SC computation,
removing majority-class floor inflation from class-imbalanced datasets.

Usage (run from /data/tchitando/Non_neg):
    python eval_sc_balanced_all.py
    python eval_sc_balanced_all.py --seed 42 --batch_size 256
"""
import sys, glob, argparse, importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--seed",        default=0,    type=int)
parser.add_argument("--batch_size",  default=256,  type=int)
parser.add_argument("--num_workers", default=0,    type=int)
parser.add_argument("--data_root",   default="./MedMNIST")
args = parser.parse_args()

BASE = "trained_models"

# (dataset_tag, method, aug, ckpt_dir, non_neg, n_classes, mean, std)
RUNS = [
    ("OCT",   "SimCLR", "A0", f"{BASE}/simclr/offline-843saz1u",            "none",     4,  (0.1894,0.1894,0.1894), (0.2071,0.2071,0.2071)),
    ("OCT",   "SimCLR", "A6", f"{BASE}/simclr/offline-72sp8q3m",            "none",     4,  (0.1894,0.1894,0.1894), (0.2071,0.2071,0.2071)),
    ("OCT",   "NCL",    "A0", f"{BASE}/simclr/offline-8bpjs318",            "rep_relu", 4,  (0.1894,0.1894,0.1894), (0.2071,0.2071,0.2071)),
    ("OCT",   "NCL",    "A6",    f"{BASE}/simclr/offline-k54r73ll",            "rep_relu", 4,  (0.1894,0.1894,0.1894), (0.2071,0.2071,0.2071)),
    ("OCT",   "NCL",    "E1_r0",  f"{BASE}/e1/simclr/offline-b6481hbh",          "rep_relu", 4,  (0.1894,0.1894,0.1894), (0.2071,0.2071,0.2071)),
    ("OCT",   "NCL",    "E1_r05", f"{BASE}/e1/simclr/offline-95tni1w1",         "rep_relu", 4,  (0.1894,0.1894,0.1894), (0.2071,0.2071,0.2071)),
    ("OCT",   "NCL",    "E1_r1",  f"{BASE}/e1/simclr/offline-na1017bs",         "rep_relu", 4,  (0.1894,0.1894,0.1894), (0.2071,0.2071,0.2071)),
    ("Path",  "SimCLR", "A0", f"{BASE}/pathmnist/simclr/offline-82pf75fm",  "none",     9,  (0.7406,0.5331,0.7059), (0.1292,0.1765,0.1381)),
    ("Path",  "SimCLR", "A6", f"{BASE}/pathmnist/simclr/offline-1m8f30pm",  "none",     9,  (0.7406,0.5331,0.7059), (0.1292,0.1765,0.1381)),
    ("Path",  "NCL",    "A0", f"{BASE}/pathmnist/simclr/offline-02f6itz6",  "rep_relu", 9,  (0.7406,0.5331,0.7059), (0.1292,0.1765,0.1381)),
    ("Path",  "NCL",    "A6", f"{BASE}/pathmnist/simclr/offline-du729it6",  "rep_relu", 9,  (0.7406,0.5331,0.7059), (0.1292,0.1765,0.1381)),
    ("Organ", "SimCLR", "A0", f"{BASE}/organamnist/simclr/offline-3fc0no59","none",     11, (0.4677,0.4677,0.4677), (0.2479,0.2479,0.2479)),
    ("Organ", "SimCLR", "A6", f"{BASE}/organamnist/simclr/offline-9zo60wa7","none",     11, (0.4677,0.4677,0.4677), (0.2479,0.2479,0.2479)),
    ("Organ", "NCL",    "A0", f"{BASE}/organamnist/simclr/offline-248gfko5","rep_relu", 11, (0.4677,0.4677,0.4677), (0.2479,0.2479,0.2479)),
    ("Organ", "NCL",    "A6", f"{BASE}/organamnist/simclr/offline-1aqq196y","rep_relu", 11, (0.4677,0.4677,0.4677), (0.2479,0.2479,0.2479)),
]

# ── Model ─────────────────────────────────────────────────────────────────────
class SimCLRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 256)
        )

    def forward(self, x):
        return self.projector(self.backbone(x))

def load_model(ckpt_path):
    model = SimCLRModel()
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.backbone.load_state_dict(
        {k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")},
        strict=True)
    model.projector.load_state_dict(
        {k.replace("projector.", ""): v for k, v in sd.items() if k.startswith("projector.")},
        strict=True)
    return model.cuda().eval()

def find_ckpt(directory):
    candidates = glob.glob(f"{directory}/*.ckpt") + glob.glob(f"{directory}/**/*.ckpt", recursive=True)
    ep99 = [c for c in candidates if "ep=99" in c]
    return ep99[0] if ep99 else candidates[0]

# ── Dataset ───────────────────────────────────────────────────────────────────
def _load_module(file_path):
    spec = importlib.util.spec_from_file_location("_ds_module", file_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def get_dataset(tag, split, transform):
    if tag == "OCT":
        sys.path.insert(0, ".")
        from MedMNIST.octmnist_dataset import OCTMnistDataset
        ds = OCTMnistDataset(split=split, transform=transform)
        return _wrap(ds)
    if tag == "Path":
        mod = _load_module("MedMNIST/pathmnist/dataset.py")
        ds  = mod.PathMNISTDataset(split=split, transform=transform)
        return _wrap(ds)
    if tag == "Organ":
        mod = _load_module("MedMNIST/organamnist/dataset.py")
        ds  = mod.OrganAMNISTDataset(split=split, transform=transform)
        return _wrap(ds)
    raise ValueError(tag)

class _wrap(torch.utils.data.Dataset):
    def __init__(self, ds): self._ds = ds
    def __len__(self): return len(self._ds)
    def __getitem__(self, i):
        img, lbl = self._ds[i]
        lbl = int(lbl[0]) if hasattr(lbl, "__len__") else int(lbl)
        return img, lbl

# ── Extraction ────────────────────────────────────────────────────────────────
def extract(model, tag, mean, std, non_neg):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    ds = get_dataset(tag, "test", tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            z = model(x.cuda())
            if non_neg in ("relu", "rep_relu"):
                z = F.relu(z)
            feats.append(z.cpu())
            labels.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
    return torch.cat(feats), torch.cat(labels)

# ── Balance ───────────────────────────────────────────────────────────────────
def balance(feats, labels, n_classes):
    g = torch.Generator()
    g.manual_seed(args.seed)
    min_count = min((labels == c).sum().item() for c in range(n_classes))
    idx = []
    for c in range(n_classes):
        cidx = (labels == c).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(cidx), generator=g)[:min_count]
        idx.append(cidx[perm])
    idx = torch.cat(idx)
    return feats[idx], labels[idx], min_count

# ── CCR ───────────────────────────────────────────────────────────────────────
def ccr(features, labels, n_classes, non_neg, eps=1e-5):
    total_dims = features.shape[1]
    if non_neg in ("relu", "rep_relu"):
        alive_mask = features.sum(0) > 0
    else:
        alive_mask = features.abs().sum(0) > 0
    features    = features[:, alive_mask]
    active_dims = features.shape[1]
    features    = F.normalize(features, dim=1)
    scores = []
    for d in range(active_dims):
        mask = features.abs()[:, d] > eps
        lbl  = labels[mask]
        try:
            dist = lbl.bincount(minlength=n_classes).float()
            dist = dist / dist.sum()
            scores.append(dist.max().item())
        except Exception:
            pass
    scores = torch.tensor(scores) * 100
    return active_dims, total_dims, scores.mean().item(), scores.std().item()

# ── Run all ───────────────────────────────────────────────────────────────────
print(f"\n{'Dataset':<8} {'Method':<8} {'Aug':<4} {'SC (balanced)':<20} {'Dead':<12} {'n/class'}")
print("-" * 72)

results = []
for (tag, method, aug, ckpt_dir, non_neg, n_cls, mean, std) in RUNS:
    ckpt_path = find_ckpt(ckpt_dir)
    model = load_model(ckpt_path)
    feats, labels = extract(model, tag, mean, std, non_neg)
    b_feats, b_labels, min_count = balance(feats, labels, n_cls)
    active, total, sc_mean, sc_std = ccr(b_feats, b_labels, n_cls, non_neg)
    dead = total - active
    floor = 100.0 / n_cls
    print(f"{tag:<8} {method:<8} {aug:<4} {sc_mean:6.2f}% ±{sc_std:5.2f}%       {dead}/{total:<8} {min_count}  (floor {floor:.2f}%)")
    results.append((tag, method, aug, sc_mean, sc_std, dead, total, min_count))
    del model

print("\nDone.")
