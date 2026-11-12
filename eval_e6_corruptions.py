import ctypes as _ctypes, os as _os
_ml_lib = '/data/tchitando/.conda/envs/ml/lib'
for _lib in ['libMagickCore-7.Q16HDRI.so.10', 'libMagickWand-7.Q16HDRI.so.10']:
    _path = _os.path.join(_ml_lib, _lib)
    if _os.path.exists(_path):
        _ctypes.CDLL(_path, mode=_ctypes.RTLD_GLOBAL)

"""
E6 Held-out Corruptions — MedMNIST-C difference-in-differences.

DiD_c = [Rel(NCL,D) - Rel(NCL,S)] - [Rel(SimCLR,D) - Rel(SimCLR,S)]
Rel_c = Acc_corrupted / Acc_clean

Groups:
  simulated : corruptions Policy D explicitly uses (speckle, contrast)
  unseen    : all other dataset corruptions

H5 holds if DiD > 0 on the unseen group.

Usage (from /data/tchitando/Non_neg):
    python eval_e6_corruptions.py --dataset oct
    python eval_e6_corruptions.py --dataset path
    python eval_e6_corruptions.py --dataset organ
"""
import sys, argparse, importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, choices=["oct", "path", "organ"])
parser.add_argument("--severity", default=-1,  type=int, help="-1=default, 0-4=index into severity_params")
parser.add_argument("--batch_size",  default=256, type=int)
parser.add_argument("--num_workers", default=0,   type=int)
parser.add_argument("--seed",        default=0,   type=int)
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# ── Per-dataset config ────────────────────────────────────────────────────────
CFGS = {
    "oct": dict(
        n_classes = 4,
        mean = (0.1894, 0.1894, 0.1894),
        std  = (0.2071, 0.2071, 0.2071),
        ds_name = "octmnist",
        ckpts = {
            "simclr_S": "trained_models/simclr/offline-843saz1u",
            "simclr_D": "trained_models/simclr/offline-72sp8q3m",
            "ncl_S":    "trained_models/simclr/offline-8bpjs318",
            "ncl_D":    "trained_models/simclr/offline-k54r73ll",
        },
        simulated = {"speckle_noise", "contrast_down"},
    ),
    "path": dict(
        n_classes = 9,
        mean = (0.7406, 0.5331, 0.7059),
        std  = (0.1292, 0.1765, 0.1381),
        ds_name = "pathmnist",
        ckpts = {
            "simclr_S": "trained_models/pathmnist/simclr/offline-82pf75fm",
            "simclr_D": "trained_models/pathmnist/simclr/offline-1m8f30pm",
            "ncl_S":    "trained_models/pathmnist/simclr/offline-02f6itz6",
            "ncl_D":    "trained_models/pathmnist/simclr/offline-du729it6",
        },
        simulated = {"contrast_up", "contrast_down"},
    ),
    "organ": dict(
        n_classes = 11,
        mean = (0.4677, 0.4677, 0.4677),
        std  = (0.2479, 0.2479, 0.2479),
        ds_name = "organamnist",
        ckpts = {
            "simclr_S": "trained_models/organamnist/simclr/offline-3fc0no59",
            "simclr_D": "trained_models/organamnist/simclr/offline-9zo60wa7",
            "ncl_S":    "trained_models/organamnist/simclr/offline-248gfko5",
            "ncl_D":    "trained_models/organamnist/simclr/offline-1aqq196y",
        },
        simulated = {"speckle_noise", "contrast_up", "contrast_down"},
    ),
}

cfg     = CFGS[args.dataset]
N_CLS   = cfg["n_classes"]
DS_NAME = cfg["ds_name"]

# ── Corruptions ───────────────────────────────────────────────────────────────
from medmnistc.corruptions.registry import CORRUPTIONS_DS
corruptions = CORRUPTIONS_DS[DS_NAME]
simulated   = {k: v for k, v in corruptions.items() if k in cfg["simulated"]}
unseen      = {k: v for k, v in corruptions.items() if k not in cfg["simulated"]}
print(f"Dataset: {args.dataset} | simulated: {list(simulated)} | unseen: {list(unseen)}")

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

import glob
def find_ckpt(d):
    hits = glob.glob(f"{d}/*.ckpt") + glob.glob(f"{d}/**/*.ckpt", recursive=True)
    return next(c for c in hits if "ep=99" in c)

def load_model(ckpt_dir, non_neg):
    m = SimCLRModel()
    sd = torch.load(find_ckpt(ckpt_dir), map_location="cpu")["state_dict"]
    m.backbone.load_state_dict({k.replace("backbone.",""): v for k,v in sd.items() if k.startswith("backbone.")}, strict=True)
    m.projector.load_state_dict({k.replace("projector.",""): v for k,v in sd.items() if k.startswith("projector.")}, strict=True)
    m._non_neg = non_neg
    return m.cuda().eval()

models_dict = {
    "simclr_S": load_model(cfg["ckpts"]["simclr_S"], "none"),
    "simclr_D": load_model(cfg["ckpts"]["simclr_D"], "none"),
    "ncl_S":    load_model(cfg["ckpts"]["ncl_S"],    "rep_relu"),
    "ncl_D":    load_model(cfg["ckpts"]["ncl_D"],    "rep_relu"),
}
print("All checkpoints loaded.")

# ── Dataset ───────────────────────────────────────────────────────────────────
def _load_module(path):
    spec = importlib.util.spec_from_file_location("_ds", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class _wrap(torch.utils.data.Dataset):
    def __init__(self, ds, transform, corruption=None, severity=-1):
        self._ds = ds
        self._tf = transform
        self._corruption = corruption
        self._severity   = severity
    def __len__(self): return len(self._ds)
    def __getitem__(self, i):
        img, lbl = self._ds[i]
        if self._corruption is not None:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img)).convert('RGB')
            else:
                img = img.convert('RGB')
            result = self._corruption.apply(img, severity=self._severity)
            if isinstance(result, Image.Image):
                img = result.convert('RGB')
            else:
                arr = np.array(result)
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                elif arr.shape[-1] != 3:
                    arr = arr[:, :, :3]
                img = Image.fromarray(arr.astype(np.uint8)).convert('RGB')
        img = self._tf(img)
        lbl = int(lbl[0]) if hasattr(lbl, "__len__") else int(lbl)
        return img, lbl

def get_base_ds(split):
    if args.dataset == "oct":
        sys.path.insert(0, ".")
        from MedMNIST.octmnist_dataset import OCTMnistDataset
        return OCTMnistDataset(split=split, transform=None)
    if args.dataset == "path":
        mod = _load_module("MedMNIST/pathmnist/dataset.py")
        return mod.PathMNISTDataset(split=split, transform=None)
    if args.dataset == "organ":
        mod = _load_module("MedMNIST/organamnist/dataset.py")
        return mod.OrganAMNISTDataset(split=split, transform=None)

norm_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cfg["mean"], cfg["std"])])

# ── Extraction ────────────────────────────────────────────────────────────────
def extract(model, base_ds, corruption=None):
    ds = _wrap(base_ds, norm_tf, corruption=corruption, severity=args.severity)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            z = model(x.cuda())
            if model._non_neg == "rep_relu":
                z = F.relu(z)
            feats.append(z.cpu())
            labels.append(torch.tensor(y) if not isinstance(y, torch.Tensor) else y)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()

# ── Probe fit (clean features) ────────────────────────────────────────────────
base_train = get_base_ds("train")
base_test  = get_base_ds("test")

probes, scalers, clean_accs = {}, {}, {}
print("\nFitting probes on clean features...")
for name, m in models_dict.items():
    Z_tr, y_tr = extract(m, base_train)
    Z_te, y_te = extract(m, base_test)
    sc = StandardScaler().fit(Z_tr)
    pr = LogisticRegression(max_iter=1000, C=1.0, random_state=args.seed)
    pr.fit(sc.transform(Z_tr), y_tr)
    acc = pr.score(sc.transform(Z_te), y_te)
    probes[name], scalers[name], clean_accs[name] = pr, sc, acc
    print(f"  {name}: clean acc = {acc*100:.2f}%")

# ── Corruption evaluation ─────────────────────────────────────────────────────
def eval_corruption(corruption_name, corruption_obj):
    rel = {}
    for name, m in models_dict.items():
        try:
            Z_te, y_te = extract(m, base_test, corruption=corruption_obj)
            acc = probes[name].score(scalers[name].transform(Z_te), y_te)
            rel[name] = acc / clean_accs[name]
        except Exception as e:
            print(f"  [skip {corruption_name}/{name}: {e}]")
            return None
    return rel

print(f"\n{'Corruption':<20} {'Group':<12} {'Rel SimCLR-S':>13} {'Rel SimCLR-D':>13} {'Rel NCL-S':>10} {'Rel NCL-D':>10} {'DiD':>8}")
print("-" * 95)

all_results = {}
for group_name, group in [("simulated", simulated), ("unseen", unseen)]:
    group_dids = []
    for cname, cobj in group.items():
        rel = eval_corruption(cname, cobj)
        if rel is None:
            print(f"{cname:<20} {group_name:<12} {'SKIPPED (medmnistc bug)':>60}")
            continue
        did = (rel["ncl_D"] - rel["ncl_S"]) - (rel["simclr_D"] - rel["simclr_S"])
        group_dids.append(did)
        print(f"{cname:<20} {group_name:<12} {rel['simclr_S']:>13.3f} {rel['simclr_D']:>13.3f} {rel['ncl_S']:>10.3f} {rel['ncl_D']:>10.3f} {did:>+8.3f}")
        all_results[cname] = dict(rel=rel, did=did, group=group_name)
    if group_dids:
        mean_did = np.mean(group_dids)
        print(f"{'  MEAN ' + group_name:<20} {'':12} {'':13} {'':13} {'':10} {'':10} {mean_did:>+8.3f}")
    print()

print(f"\nH5: DiD > 0 on unseen group → NCL-D generalises beyond what policy D trained on.")
unseen_dids = [v["did"] for v in all_results.values() if v["group"] == "unseen"]
if unseen_dids:
    print(f"Mean DiD (unseen): {np.mean(unseen_dids):+.3f}  ({'HOLDS' if np.mean(unseen_dids) > 0 else 'FAILS'})")
