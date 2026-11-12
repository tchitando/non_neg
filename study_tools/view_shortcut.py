#!/usr/bin/env python3
"""How easily can the two training views of a scan be matched by intensity statistics alone (the view shortcut)?

    python study_tools/view_shortcut.py                  # OCT training images, the crop-only policies and the old stem
    python study_tools/view_shortcut.py --n 4000 --pool 256 --policies croponly croponly_bright croponly_contrast_f050

Hypothesis: two crops of one scan share its overall intensity (gain, brightness), so the contrastive task can be partly
solved by matching those statistics instead of the content. Brightness jitter changes a view's mean intensity and
should break the cue; our contrast operation keeps each view's mean (it scales values towards the image mean), so it
should leave the mean cue intact and only weaken cues that depend on spread.

For each policy the script builds the training augmentation exactly as main_pretrain.py does (the policy's yaml through
MedMNIST.pretrain_dataloader.build_transform_pipeline, two views per image with NCropAugmentation), makes two views of
--n training images, and describes every view by a few intensity statistics:
    mean        the view's mean intensity
    mean+sd     mean and standard deviation
    quantiles   the 10th, 50th, 90th and 99th percentiles
    histogram   a 16-bin intensity histogram (square-rooted)
Within random groups of --pool images (the contrastive batch: one positive and pool-1 negatives), each first view
looks for its partner among the group's second views by nearest neighbour on the standardised statistics, in both
directions. The score is the share of views whose partner is found (top-1; chance = 1/pool). No network is involved:
the score measures how much of the contrastive task the intensity cue alone can solve under that augmentation.

If study/derived/croponly_summary.csv exists (croponly_table.py), the script also prints, across policies, the
correlation of the shortcut score with NCL's SC above shuffle, projector purity and the backbone probe.
Output: study/derived/view_shortcut.csv (policy, statistic, pool, top-1 %, chance %).
"""
import argparse, csv, os, sys
from pathlib import Path

sys.path.insert(0, ".")
# ImageMagick (medmnistc's blur operations) must be loaded before torch, as in main_pretrain.py
import ctypes as _ctypes
for _lib in ("libMagickCore-7.Q16HDRI.so.10", "libMagickWand-7.Q16HDRI.so.10"):
    _p = os.path.join(sys.prefix, "lib", _lib)
    if os.path.exists(_p):
        try:
            _ctypes.CDLL(_p, mode=_ctypes.RTLD_GLOBAL)
        except OSError:
            pass
try:
    import MedMNIST.medmnistc_transforms  # noqa: F401  (loads wand before torch)
except Exception:
    pass

import numpy as np

AUG_DIR = Path("MedMNIST/scripts/pretrain/octmnist/augmentations")
DEFAULT = ["croponly", "croponly_flip", "croponly_bright_b020", "croponly_bright_b040", "croponly_bright",
           "croponly_bright_b095", "croponly_contrast", "croponly_contrast_f050", "croponly_contrast_f030",
           "croponly_contrast_f010", "croponly_blur_sig060", "croponly_blur", "croponly_blur_sig200",
           "croponly_crop020", "croponly_crop050", "stem", "contrast"]
FEATS = ["mean", "mean+sd", "quantiles", "histogram"]


def view_stats(x):
    """x: (n, H, W) float array of views -> dict of feature matrices."""
    f = x.reshape(len(x), -1)
    q = np.percentile(f, [10, 50, 90, 99], axis=1).T
    lo, hi = f.min(), f.max()
    hist = np.stack([np.histogram(v, bins=16, range=(lo, hi))[0] for v in f]).astype(np.float64)
    hist = np.sqrt(hist / hist.sum(1, keepdims=True))
    m, s = f.mean(1, keepdims=True), f.std(1, keepdims=True)
    return {"mean": m, "mean+sd": np.hstack([m, s]), "quantiles": q, "histogram": hist}


def match_rate(A, B, pool, rng):
    """Top-1 partner retrieval within random groups of `pool` images, both directions, standardised features."""
    Z = np.vstack([A, B])
    mu, sd = Z.mean(0), Z.std(0) + 1e-9
    A, B = (A - mu) / sd, (B - mu) / sd
    idx = rng.permutation(len(A))
    hits = tot = 0
    for g in range(len(A) // pool):
        sel = idx[g * pool:(g + 1) * pool]
        a, b = A[sel], B[sel]
        d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        hits += (d.argmin(1) == np.arange(pool)).sum() + (d.argmin(0) == np.arange(pool)).sum()
        tot += 2 * pool
    return 100.0 * hits / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="*", default=DEFAULT)
    ap.add_argument("--n", type=int, default=4096, help="training images (two views each)")
    ap.add_argument("--pool", type=int, default=256, help="group size: one positive among pool-1 negatives")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default=None, help="folder with octmnist_64.npz (default: the dataset module's)")
    ap.add_argument("--out", default="study/derived/view_shortcut.csv")
    ap.add_argument("--summary", default="study/derived/croponly_summary.csv")
    a = ap.parse_args()
    import random, torch, yaml
    from omegaconf import OmegaConf
    from MedMNIST.octmnist_dataset import OCTMnistDataset
    from MedMNIST.pretrain_dataloader import build_transform_pipeline, NCropAugmentation

    ds = OCTMnistDataset(split="train", transform=None, root=a.root)
    rng = np.random.default_rng(a.seed)
    pick = rng.choice(len(ds), size=min(a.n, len(ds)), replace=False)
    imgs = [ds[int(i)][0] for i in pick]
    rows = []
    print(f"{len(imgs)} training images, groups of {a.pool} (chance {100 / a.pool:.2f}%)\n")
    print(f"{'policy':28s}" + "".join(f"{f:>12s}" for f in FEATS))
    for pol in a.policies:
        y = AUG_DIR / f"p_{pol}.yaml"
        if not y.exists():
            print(f"{pol:28s}  (no {y.name}; skipped)"); continue
        try:
            cfg = OmegaConf.create(yaml.safe_load(y.read_text())[0])
            tf = NCropAugmentation(build_transform_pipeline("octmnist", cfg), 2, sharing=0.0)
            random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
            v1, v2 = [], []
            for im in imgs:
                x1, x2 = tf(im)
                v1.append(x1[0].numpy()); v2.append(x2[0].numpy())
        except Exception as e:  # one policy failing (e.g. a missing library) does not stop the others
            print(f"{pol:28s}  failed: {type(e).__name__}: {str(e).splitlines()[0][:80]}"); continue
        s1, s2 = view_stats(np.stack(v1)), view_stats(np.stack(v2))
        res = {f: match_rate(s1[f], s2[f], a.pool, np.random.default_rng(a.seed)) for f in FEATS}
        print(f"{pol:28s}" + "".join(f"{res[f]:12.2f}" for f in FEATS), flush=True)
        for f in FEATS:
            rows.append(dict(policy_id=pol, statistic=f, pool=a.pool, n_images=len(imgs),
                             top1_pct=round(res[f], 3), chance_pct=round(100 / a.pool, 3)))
    if not rows:
        sys.exit("no policy found")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out}")

    if Path(a.summary).exists():
        import pandas as pd
        sc = pd.read_csv(a.summary)
        sh = pd.DataFrame(rows).pivot(index="policy_id", columns="statistic", values="top1_pct")
        print("\nAcross policies: correlation of the shortcut score with specialisation and accuracy (Pearson r)")
        for obj, cols in (("ncl", ["sc_excess_mean", "purity_z_mean", "purity_h_mean", "probe_bb_mean"]),
                          ("simclr", ["purity_z_mean", "purity_h_mean", "probe_bb_mean"])):
            m = sc[sc.objective == obj].set_index("policy_id")[cols].join(sh, how="inner").dropna()
            if len(m) < 4:
                print(f"  {obj}: fewer than 4 policies with both scores; skipped"); continue
            for f in FEATS:
                print(f"  {obj:6s} {f:10s} " + "  ".join(f"{c.replace('_mean', '')} {np.corrcoef(m[f], m[c])[0, 1]:+.2f}"
                                                        for c in cols) + f"   ({len(m)} policies)")


if __name__ == "__main__":
    main()
