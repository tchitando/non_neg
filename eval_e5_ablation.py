"""
E5 Feature Ablation — projects units from ranked sets, measures accuracy drop ratio R(k).

R(k) = ΔAcc(top-k set) / E[ΔAcc(random-k)]

Sets ranked by balanced SC (class-selective) and MS; random matched on firing frequency.
Runs the projector probe (primary) and a refit probe after ablation (secondary).

Usage (from /data/tchitando/Non_neg):
    python eval_e5_ablation.py --dataset oct --method ncl \
        --ckpt trained_models/simclr/offline-n371k5uk/octmnist-ncl-resnet50-A6-offline-n371k5uk-ep=99.ckpt \
        --non_neg rep_relu

    python eval_e5_ablation.py --dataset path --method simclr \
        --ckpt trained_models/pathmnist/simclr/offline-1m8f30pm/pathmnist-simclr-resnet50-A6-offline-1m8f30pm-ep=99.ckpt \
        --non_neg none
"""
import sys, glob, argparse, importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",     required=True, choices=["oct","path","organ"])
parser.add_argument("--ckpt",        required=True)
parser.add_argument("--non_neg",     default="none", choices=["none","rep_relu"])
parser.add_argument("--ks",          default="4,8,16,32")
parser.add_argument("--n_rand_sets", default=20,  type=int, help="Random sets to average over")
parser.add_argument("--batch_size",  default=256, type=int)
parser.add_argument("--num_workers", default=0,   type=int)
parser.add_argument("--seed",        default=0,   type=int)
parser.add_argument("--split_train", default="train")
parser.add_argument("--split_test",  default="test")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

KS = [int(k) for k in args.ks.split(",")]

DATASET_CFG = {
    "oct":   dict(n_classes=4,  mean=(0.1894,0.1894,0.1894), std=(0.2071,0.2071,0.2071)),
    "path":  dict(n_classes=9,  mean=(0.7406,0.5331,0.7059), std=(0.1292,0.1765,0.1381)),
    "organ": dict(n_classes=11, mean=(0.4677,0.4677,0.4677), std=(0.2479,0.2479,0.2479)),
}
cfg = DATASET_CFG[args.dataset]
N_CLASSES = cfg["n_classes"]

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

model = SimCLRModel()
sd = torch.load(args.ckpt, map_location="cpu")["state_dict"]
model.backbone.load_state_dict(
    {k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")}, strict=True)
model.projector.load_state_dict(
    {k.replace("projector.", ""): v for k, v in sd.items() if k.startswith("projector.")}, strict=True)
model = model.cuda().eval()
print(f"Loaded: {args.ckpt}")

# ── Dataset ───────────────────────────────────────────────────────────────────
def _load_module(path):
    spec = importlib.util.spec_from_file_location("_ds", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class _wrap(torch.utils.data.Dataset):
    def __init__(self, ds): self._ds = ds
    def __len__(self): return len(self._ds)
    def __getitem__(self, i):
        img, lbl = self._ds[i]
        return img, int(lbl[0]) if hasattr(lbl, "__len__") else int(lbl)

def get_dataset(split, transform):
    if args.dataset == "oct":
        sys.path.insert(0, ".")
        from MedMNIST.octmnist_dataset import OCTMnistDataset
        return _wrap(OCTMnistDataset(split=split, transform=transform))
    if args.dataset == "path":
        mod = _load_module("MedMNIST/pathmnist/dataset.py")
        return _wrap(mod.PathMNISTDataset(split=split, transform=transform))
    if args.dataset == "organ":
        mod = _load_module("MedMNIST/organamnist/dataset.py")
        return _wrap(mod.OrganAMNISTDataset(split=split, transform=transform))

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cfg["mean"], cfg["std"]),
])

# ── Extraction ────────────────────────────────────────────────────────────────
def extract(split):
    loader = DataLoader(get_dataset(split, tf), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, pin_memory=True)
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            z = model(x.cuda())
            if args.non_neg == "rep_relu":
                z = F.relu(z)
            feats.append(z.cpu())
            labels.append(torch.tensor(y) if not isinstance(y, torch.Tensor) else y)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()

print("Extracting train features...")
Z_tr, y_tr = extract(args.split_train)
print("Extracting test features...")
Z_te, y_te = extract(args.split_test)

# ── Projector probe (full) ────────────────────────────────────────────────────
scaler = StandardScaler().fit(Z_tr)
Z_tr_s = scaler.transform(Z_tr)
Z_te_s = scaler.transform(Z_te)

probe = LogisticRegression(max_iter=1000, C=1.0, random_state=args.seed)
probe.fit(Z_tr_s, y_tr)
acc_full = probe.score(Z_te_s, y_te)
print(f"\nProjector probe (full): {acc_full*100:.2f}%")

# ── Per-unit SC (balanced) ────────────────────────────────────────────────────
def unit_sc(Z, y, eps=1e-5):
    Z_t = torch.tensor(Z)
    y_t = torch.tensor(y, dtype=torch.long)
    # balance
    min_n = min((y_t == c).sum().item() for c in range(N_CLASSES))
    idx = []
    for c in range(N_CLASSES):
        cidx = (y_t == c).nonzero(as_tuple=True)[0]
        idx.append(cidx[torch.randperm(len(cidx))[:min_n]])
    idx = torch.cat(idx)
    Z_b, y_b = Z_t[idx], y_t[idx]
    Z_n = F.normalize(Z_b, dim=1)
    scores = []
    for d in range(Z_n.shape[1]):
        mask = Z_n.abs()[:, d] > eps
        if mask.sum() == 0:
            scores.append(0.0)
            continue
        dist = y_b[mask].bincount(minlength=N_CLASSES).float()
        dist /= dist.sum()
        scores.append(dist.max().item())
    return np.array(scores)

print("Computing per-unit SC...")
sc_scores = unit_sc(Z_tr, y_tr)

# alive mask
if args.non_neg == "rep_relu":
    alive = torch.tensor(Z_tr).sum(0) > 0
else:
    alive = torch.tensor(Z_tr).abs().sum(0) > 0
alive = alive.numpy()
n_alive = alive.sum()
print(f"Alive units: {n_alive}/256")

# SC* (chance-normalised)
chance = 1.0 / N_CLASSES
sc_star = (sc_scores[alive].mean() - chance) / (1.0 - chance)
print(f"SC (balanced, alive units): {sc_scores[alive].mean()*100:.2f}%")
print(f"SC* (chance-normalised):    {sc_star*100:.2f}%")

# ── Firing frequency per unit ─────────────────────────────────────────────────
firing_freq = (torch.tensor(Z_tr) > 1e-6).float().mean(0).numpy()

# ── Ablation ─────────────────────────────────────────────────────────────────
alive_idx = np.where(alive)[0]
sc_alive   = sc_scores[alive_idx]

# rank sets: high SC (class-selective), low SC
rank_order   = np.argsort(-sc_alive)        # descending SC
high_sc_idx  = alive_idx[rank_order]        # global unit indices, high SC first
low_sc_idx   = alive_idx[rank_order[::-1]]  # low SC first

def ablate_and_score(unit_indices, k, Z_tr_s, Z_te_s, y_tr, y_te, refit=False):
    mask = np.ones(Z_tr_s.shape[1], dtype=bool)
    mask[unit_indices[:k]] = False
    Z_tr_a = Z_tr_s.copy(); Z_tr_a[:, ~mask] = 0.0
    Z_te_a = Z_te_s.copy(); Z_te_a[:, ~mask] = 0.0
    if refit:
        p = LogisticRegression(max_iter=1000, C=1.0, random_state=args.seed)
        p.fit(Z_tr_a, y_tr)
        return p.score(Z_te_a, y_te)
    else:
        return probe.score(Z_te_a, y_te)

def random_matched_sets(reference_idx, k, n_sets):
    ref_freq = firing_freq[reference_idx[:k]].mean()
    # pool alive units not in reference top-k, matched on firing frequency
    pool = np.setdiff1d(alive_idx, reference_idx[:k])
    pool_freq = firing_freq[pool]
    accs = []
    rng = np.random.RandomState(args.seed)
    for _ in range(n_sets):
        weights = np.exp(-np.abs(pool_freq - ref_freq))
        weights /= weights.sum()
        chosen = rng.choice(pool, size=k, replace=False, p=weights)
        accs.append(ablate_and_score(chosen, k, Z_tr_s, Z_te_s, y_tr, y_te, refit=False))
    return np.array(accs)

print(f"\n{'k':>4}  {'ΔAcc(high-SC)':>14}  {'ΔAcc(low-SC)':>13}  {'E[ΔAcc(rand)]':>14}  {'R(k) high':>10}  {'R(k) low':>9}")
print("-" * 75)

results = {}
for k in KS:
    if k > n_alive:
        print(f"{k:>4}  skipped (k > alive count {n_alive})")
        continue

    acc_high = ablate_and_score(high_sc_idx, k, Z_tr_s, Z_te_s, y_tr, y_te)
    acc_low  = ablate_and_score(low_sc_idx,  k, Z_tr_s, Z_te_s, y_tr, y_te)
    rand_accs = random_matched_sets(high_sc_idx, k, args.n_rand_sets)
    acc_rand  = rand_accs.mean()

    delta_high = acc_full - acc_high
    delta_low  = acc_full - acc_low
    delta_rand = acc_full - acc_rand
    R_high = delta_high / delta_rand if delta_rand > 1e-6 else float("nan")
    R_low  = delta_low  / delta_rand if delta_rand > 1e-6 else float("nan")

    print(f"{k:>4}  {delta_high*100:>+13.2f}%  {delta_low*100:>+12.2f}%  {delta_rand*100:>+13.2f}%  {R_high:>10.3f}  {R_low:>9.3f}")
    results[k] = dict(R_high=R_high, R_low=R_low, delta_high=delta_high,
                      delta_low=delta_low, delta_rand=delta_rand)

print(f"\nR(k) > 1 means high-SC units matter more than matched random units.")
print(f"Dataset: {args.dataset}  Method: {args.non_neg}  Full probe acc: {acc_full*100:.2f}%")
