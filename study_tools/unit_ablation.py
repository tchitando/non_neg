#!/usr/bin/env python3
"""Which units does a model rely on, and does the augmentation change that?  (OCT, ResNet-50, existing checkpoints)

    python study_tools/unit_ablation.py --dry-run                 # list the runs and checkpoints it would use
    CUDA_VISIBLE_DEVICES=1 python study_tools/unit_ablation.py    # 4 policies x NCL, SimCLR x seeds 42-44 = 24 runs
    python study_tools/unit_ablation.py --policies croponly croponly_bright --objectives ncl --seeds 42

Two stages, on validation images only, with disjoint halves so that units are never chosen and tested on the same images.

1. IDENTIFY (half A, clean images). For every unit of the projector read-out z (NCL: ReLU(z); SimCLR: z) and of the
   backbone h, on the l2-normalised features:
     mean_act, sparsity      mean activation; share of images with |u| > 1e-5
     pref_class, auroc       the class with the highest mean activation, and how well the unit alone separates it
                             from the other classes (one-vs-rest AUROC; 0.5 = no separation)
     rho_mean, rho_std       Spearman correlation with the image's mean intensity and with its standard deviation
     inv_<policy>            correlation of the unit's activation between the two views of an image made with that
                             policy (half B); 1 = the unit ignores what the policy changes
   Unit groups, each k = --frac of the units that are ever active:
     intensity               the k units with the largest |rho_mean|
     class                   the k units with the largest auroc
     random                  k units drawn at random (--draws times)
     matched_intensity       for each intensity unit, a random other unit from the same mean-activation decile
     matched_class           the same for the class units   (controls for "these units are simply larger")
   The overlap between the intensity and class groups is recorded; a unit can be both.

2. TEST (half B). Each group is switched off (its activations set to 0, before normalisation), then:
     view_<policy>           view matching: two views of each image are made with that policy (the same views for
                             every model); within groups of --pool images, the share of views whose nearest neighbour
                             (cosine) is their own partner, both directions; chance = 1/pool. The own training policy
                             is the task the loss trained; the others are the cross-test (e.g. a crop-only model
                             matching views that differ in brightness).
     class_bal_acc           class decoding: a linear softmax probe trained on --train-per-class training images per
                             class (same units off), balanced accuracy on half B.
   'none' = nothing switched off (the reference).

Read-outs: SimCLR's projector has no privileged basis (its loss is unchanged by rotating z), so its z-unit results
describe arbitrary directions; compare the methods on h, where both have a ReLU.

Outputs (appended; runs already present are skipped):
  study/derived/unit_stats.csv       one row per run x layer x unit
  study/derived/unit_ablation.csv    one row per run x layer x group x draw x test
"""
import argparse, csv, os, sys
from pathlib import Path

sys.path.insert(0, ".")
import ctypes as _ctypes                                   # ImageMagick before torch, as in main_pretrain.py
for _lib in ("libMagickCore-7.Q16HDRI.so.10", "libMagickWand-7.Q16HDRI.so.10"):
    _p = os.path.join(sys.prefix, "lib", _lib)
    if os.path.exists(_p):
        try:
            _ctypes.CDLL(_p, mode=_ctypes.RTLD_GLOBAL)
        except OSError:
            pass

import numpy as np
from scipy.stats import rankdata

AUG_DIR = Path("MedMNIST/scripts/pretrain/octmnist/augmentations")
POLICIES = ["croponly", "croponly_bright", "croponly_contrast_f030", "stem"]
EPS = 1e-5


# ------------------------------------------------------------------ runs and data
def find_runs(policies, objectives, seeds, master):
    import pandas as pd
    m = pd.read_csv(master).drop_duplicates("run_id", keep="last").set_index("run_id")
    runs = []
    for pol in policies:
        for obj in objectives:
            for s in seeds:
                rid = f"{obj}-resnet50-octmnist-{pol}" + ("" if s == 42 else f"-s{s}")
                if rid not in m.index:
                    print(f"  skip {rid}: not in {master}"); continue
                runs.append(dict(run_id=rid, objective=obj, policy_id=pol, seed=s, ckpt=m.loc[rid, "ckpt_path"]))
    return runs


def halves(labels, seed=0):
    """Two disjoint class-balanced halves of the validation set (each n_min // 2 per class)."""
    rng = np.random.default_rng(seed)
    n = min(np.bincount(labels)) // 2
    A, B = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        A += list(idx[:n]); B += list(idx[n:2 * n])
    return np.array(sorted(A)), np.array(sorted(B))


def train_subset(labels, per_class, seed=1):
    rng = np.random.default_rng(seed)
    return np.array(sorted(i for c in np.unique(labels)
                           for i in rng.choice(np.where(labels == c)[0], min(per_class, (labels == c).sum()), replace=False)))


def make_views(ds, idx, policy, seed=0):
    """Two training views per image with the policy's own pipeline, identical for every model (fixed seed)."""
    import random, torch, yaml
    from omegaconf import OmegaConf
    from MedMNIST.pretrain_dataloader import build_transform_pipeline, NCropAugmentation
    cfg = OmegaConf.create(yaml.safe_load((AUG_DIR / f"p_{policy}.yaml").read_text())[0])
    tf = NCropAugmentation(build_transform_pipeline("octmnist", cfg), 2, sharing=0.0)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    v1, v2 = [], []
    for i in idx:
        a, b = tf(ds[int(i)][0])
        v1.append(a.half()); v2.append(b.half())
    return torch.stack(v1), torch.stack(v2)


# ------------------------------------------------------------------ model
def load_model(ckpt, device):
    import torch, torch.nn as nn
    from torchvision import models
    bb = models.resnet50(weights=None); bb.fc = nn.Identity()
    pj = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 256))
    sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    bb.load_state_dict({k[9:]: v for k, v in sd.items() if k.startswith("backbone.")}, strict=True)
    pj.load_state_dict({k[10:]: v for k, v in sd.items() if k.startswith("projector.")}, strict=True)
    return bb.to(device).eval(), pj.to(device).eval()


def features(bb, pj, X, relu, device, bs=512):
    """h and the projector read-out (NCL: ReLU(z), SimCLR: z), unnormalised, for a stack of model-ready images."""
    import torch, torch.nn.functional as F
    hs, zs = [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            h = bb(X[i:i + bs].float().to(device)); z = pj(h)
            hs.append(h.cpu()); zs.append((F.relu(z) if relu else z).cpu())
    return {"h": torch.cat(hs).numpy(), "z": torch.cat(zs).numpy()}


def clean_stack(ds, idx):
    import torch
    from torchvision import transforms
    from MedMNIST.octmnist_dataset import OCTMNIST_MEAN, OCTMNIST_STD
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(OCTMNIST_MEAN, OCTMNIST_STD)])
    return torch.stack([tf(ds[int(i)][0]) for i in idx])


# ------------------------------------------------------------------ unit measures
def l2n(F_):
    return F_ / np.maximum(np.linalg.norm(F_, axis=1, keepdims=True), 1e-12)


def spearman_cols(U, x):
    r = rankdata(U, axis=0); rx = rankdata(x)
    r = r - r.mean(0); rx = rx - rx.mean()
    den = np.sqrt((r ** 2).sum(0) * (rx ** 2).sum())
    return np.where(den > 0, (r * rx[:, None]).sum(0) / np.maximum(den, 1e-12), 0.0)


def auroc_cols(U, pos):
    r = rankdata(U, axis=0); n1 = pos.sum(); n0 = len(pos) - n1
    return (r[pos].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def corr_cols(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    den = np.sqrt((A ** 2).sum(0) * (B ** 2).sum(0))
    return np.where(den > 0, (A * B).sum(0) / np.maximum(den, 1e-12), np.nan)


def unit_table(Fa, ya, img_mean, img_std, views_B):
    U = l2n(Fa)
    alive = (np.abs(U) > EPS).any(0)
    cls_means = np.stack([U[ya == c].mean(0) for c in np.unique(ya)])
    pref = cls_means.argmax(0)
    auroc = np.array([auroc_cols(U[:, [k]], ya == pref[k])[0] for k in range(U.shape[1])])
    t = dict(alive=alive, mean_act=U.mean(0), sparsity=(np.abs(U) > EPS).mean(0), pref_class=pref, auroc=auroc,
             rho_mean=spearman_cols(U, img_mean), rho_std=spearman_cols(U, img_std))
    for pol, (V1, V2) in views_B.items():
        t[f"inv_{pol}"] = corr_cols(l2n(V1), l2n(V2))
    return t


def groups(t, frac, draws, rng):
    alive = np.where(t["alive"])[0]
    k = max(1, int(round(frac * len(alive))))
    by = lambda v: alive[np.argsort(-v[alive])[:k]]
    g = {"none": [np.array([], int)], "intensity": [by(np.abs(t["rho_mean"]))], "class": [by(t["auroc"])],
         "random": [rng.choice(alive, k, replace=False) for _ in range(draws)]}
    dec = np.digitize(t["mean_act"], np.quantile(t["mean_act"][alive], np.linspace(0, 1, 11)[1:-1]))

    def matched(target):
        """For each target unit, a different unit from the same mean-activation decile (outside the target group)."""
        tset = set(target.tolist())
        by_dec = {d: [v for v in alive if dec[v] == d and v not in tset] for d in np.unique(dec[alive])}
        rest = [v for v in alive if v not in tset]
        out = []
        for _ in range(draws):
            pick, used = [], set()
            for u in target:
                cand = [v for v in by_dec.get(dec[u], []) if v not in used] or [v for v in rest if v not in used]
                v = int(rng.choice(cand)); pick.append(v); used.add(v)
            out.append(np.array(pick))
        return out
    g["matched_intensity"] = matched(g["intensity"][0]); g["matched_class"] = matched(g["class"][0])
    overlap = len(set(g["intensity"][0]) & set(g["class"][0]))
    return g, k, overlap


# ------------------------------------------------------------------ tests
def view_match(V1, V2, off, pool, seed=0):
    V1 = V1.copy(); V2 = V2.copy(); V1[:, off] = 0; V2[:, off] = 0
    V1, V2 = l2n(V1), l2n(V2)
    perm = np.random.default_rng(seed).permutation(len(V1)); hits = n = 0
    for s in range(0, len(perm) - pool + 1, pool):
        i = perm[s:s + pool]; S = V1[i] @ V2[i].T
        hits += (S.argmax(1) == np.arange(pool)).sum() + (S.argmax(0) == np.arange(pool)).sum(); n += 2 * pool
    return 100 * hits / n


def probe_bal_acc(Ftr, ytr, Fte, yte, off, device, steps=400, wd=1e-4):
    import torch
    Ftr = l2n(Ftr.copy()); Fte = l2n(Fte.copy()); Ftr[:, off] = 0; Fte[:, off] = 0
    mu, sd = Ftr.mean(0), Ftr.std(0); sd[sd < 1e-8] = 1
    Xtr = torch.tensor((Ftr - mu) / sd, dtype=torch.float32, device=device)
    Xte = torch.tensor((Fte - mu) / sd, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, device=device); C = int(ytr.max()) + 1
    w = torch.tensor(len(ytr) / (C * np.bincount(ytr, minlength=C)), dtype=torch.float32, device=device)
    torch.manual_seed(0)
    lin = torch.nn.Linear(Xtr.shape[1], C).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad(); torch.nn.functional.cross_entropy(lin(Xtr), ytr_t, weight=w).backward(); opt.step()
    pred = lin(Xte).argmax(1).cpu().numpy()
    return 100 * np.mean([np.mean(pred[yte == c] == c) for c in np.unique(yte)])


# ------------------------------------------------------------------ main
def append(path, rows):
    if not rows:
        return
    new = not Path(path).exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", default=POLICIES)
    ap.add_argument("--objectives", nargs="+", default=["ncl", "simclr"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--test-policies", nargs="+", default=POLICIES, help="policies used to make views for view matching")
    ap.add_argument("--master", default="study/results_master.csv")
    ap.add_argument("--out-dir", default="study/derived")
    ap.add_argument("--frac", type=float, default=0.10, help="group size as a share of the units that are ever active")
    ap.add_argument("--draws", type=int, default=20, help="draws for the random and matched groups")
    ap.add_argument("--pool", type=int, default=256, help="group size for view matching (the contrastive batch)")
    ap.add_argument("--train-per-class", type=int, default=2000)
    ap.add_argument("--root", default=None, help="folder with octmnist_64.npz (default: the dataset module's)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    runs = find_runs(a.policies, a.objectives, a.seeds, a.master)
    for r in runs:
        print(f"{r['run_id']:52s} {r['ckpt']}")
    print(f"\n{len(runs)} runs; test policies: {a.test_policies}")
    if a.dry_run:
        return
    import torch
    from MedMNIST.octmnist_dataset import OCTMnistDataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_stats, out_abl = Path(a.out_dir) / "unit_stats.csv", Path(a.out_dir) / "unit_ablation.csv"
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    done = set()
    if out_abl.exists():
        done = {(r["run_id"], r["layer"]) for r in csv.DictReader(open(out_abl))}

    val, tr = OCTMnistDataset(split="val", root=a.root), OCTMnistDataset(split="train", root=a.root)
    A, B = halves(val.labels)
    T = train_subset(tr.labels, a.train_per_class)
    ya, yb, yt = val.labels[A], val.labels[B], tr.labels[T]
    img = val.images.astype(np.float32)
    img_mean, img_std = img[A].reshape(len(A), -1).mean(1), img[A].reshape(len(A), -1).std(1)
    print(f"validation halves: A {len(A)}, B {len(B)} images (balanced); probe training images: {len(T)}; device {device}")
    XA, XB, XT = clean_stack(val, A), clean_stack(val, B), clean_stack(tr, T)
    views = {}
    for p in a.test_policies:
        print(f"making views: {p}", flush=True)
        views[p] = make_views(val, B, p)

    for r in runs:
        if all((r["run_id"], L) in done for L in ("z", "h")):
            print(f"{r['run_id']}: already done"); continue
        if not Path(r["ckpt"]).exists():
            print(f"{r['run_id']}: checkpoint not found ({r['ckpt']})"); continue
        print(f"\n=== {r['run_id']}", flush=True)
        relu = r["objective"] == "ncl"
        bb, pj = load_model(r["ckpt"], device)
        fA, fB, fT = features(bb, pj, XA, relu, device), features(bb, pj, XB, relu, device), features(bb, pj, XT, relu, device)
        fV = {p: (features(bb, pj, v1, relu, device), features(bb, pj, v2, relu, device)) for p, (v1, v2) in views.items()}
        for L in ("z", "h"):
            if (r["run_id"], L) in done:
                continue
            t = unit_table(fA[L], ya, img_mean, img_std, {p: (v1[L], v2[L]) for p, (v1, v2) in fV.items()})
            meta = dict(run_id=r["run_id"], objective=r["objective"], policy_id=r["policy_id"], seed=r["seed"], layer=L)
            append(out_stats, [dict(meta, unit=k, **{c: (float(v[k]) if v.dtype.kind == "f" else int(v[k]))
                                                    for c, v in t.items()}) for k in range(len(t["alive"]))])
            g, k, overlap = groups(t, a.frac, a.draws, np.random.default_rng(0))
            rows, summary = [], {}
            for name, sets in g.items():
                for d, off in enumerate(sets):
                    res = {f"view_{p}": view_match(v1[L], v2[L], off, a.pool) for p, (v1, v2) in fV.items()}
                    res["class_bal_acc"] = probe_bal_acc(fT[L], yt, fB[L], yb, off, device)
                    for test, val_ in res.items():
                        rows.append(dict(meta, group=name, draw=d, k=len(off), overlap=overlap, test=test, value=round(val_, 3)))
                        summary.setdefault((name, test), []).append(val_)
            append(out_abl, rows)
            tests = ["class_bal_acc"] + [f"view_{p}" for p in fV]
            print(f"  layer {L}: k = {k} of {int(t['alive'].sum())} active units, intensity/class overlap {overlap}")
            print("  " + f"{'group':20s}" + "".join(f"{x[:22]:>24s}" for x in tests))
            for name in g:
                print("  " + f"{name:20s}" + "".join(f"{np.mean(summary[(name, x)]):24.2f}" for x in tests))
        del bb, pj
        if device == "cuda":
            torch.cuda.empty_cache()
    print(f"\nwrote {out_stats} and {out_abl}")


if __name__ == "__main__":
    main()
