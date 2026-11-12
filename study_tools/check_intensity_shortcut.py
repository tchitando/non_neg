#!/usr/bin/env python3
"""Can the OCT class be read from image intensity statistics alone, and which training augmentation removes that?

    python study_tools/check_intensity_shortcut.py                        # OCTMNIST 64 px
    python study_tools/check_intensity_shortcut.py --npz MedMNIST/octmnist_64.npz --out study/derived

Three numbers per image (pixels scaled to 0-1):
    mean   mean intensity (overall brightness)
    std    standard deviation of the pixels (contrast)
    cv     std / mean (contrast relative to brightness). Brightness jitter multiplies every pixel by one factor,
           which scales mean and std together and leaves cv unchanged; contrast decrease scales std and leaves mean
           unchanged, so it changes cv. cv is the statistic the stem cannot remove and the contrast operation can.
Part 1: per split and class, the average of each statistic.
Part 2: how much class information each statistic still carries once the training augmentations are applied.
        A logistic regression (balanced class weights) is fitted on training-set statistics and scored on val and
        test (balanced accuracy, chance 25%), with the same augmentation applied to all three splits:
        clean; the stem's brightness jitter (factor U[0.2, 1.8], p 0.8); brightness plus the contrast operation
        (factor 0.8, p 0.5); brightness plus a stronger contrast operation (factor 0.6, p 0.5). Each is applied
        exactly as in the pretraining pipeline. If cv still predicts the class after brightness jitter and falls
        towards chance as contrast decrease is added, the contrast operation removes a cue the stem leaves in place.
No GPU, no model; reads the npz once.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NAMES = {0: "CNV", 1: "DME", 2: "Drusen", 3: "Normal"}


def stats(X, rng=None, bright=False, contrast=None, chunk=8192):
    """Per-image mean, std and cv after the chosen augmentations (applied as the pretraining pipeline does)."""
    out = []
    for s in range(0, len(X), chunk):
        x = X[s:s + chunk].astype(np.float32) / 255.0
        if x.ndim == 4:                                   # (n, h, w, c) -> grey
            x = x.mean(-1)
        n = len(x)
        if bright:                                        # ColorJitter(brightness=0.8) with p 0.8: x * U[0.2, 1.8], clipped
            b = np.where(rng.random(n) < 0.8, rng.uniform(0.2, 1.8, n), 1.0).astype(np.float32)
            x = np.clip(x * b[:, None, None], 0, 1)
        if contrast:                                      # ContrastDecrease(factor) with p 0.5: mean + factor (x - mean), clipped
            f = np.where(rng.random(n) < 0.5, contrast, 1.0).astype(np.float32)
            m = x.mean((1, 2), keepdims=True)
            x = np.clip(m + f[:, None, None] * (x - m), 0, 1)
        mu = x.mean((1, 2)); sd = x.std((1, 2))
        out.append(np.stack([mu, sd, sd / np.maximum(mu, 1e-6)], 1))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="MedMNIST/octmnist_64.npz")
    ap.add_argument("--out", default="study/derived")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = np.load(a.npz)
    X = {s: d[f"{s}_images"] for s in ("train", "val", "test")}
    y = {s: d[f"{s}_labels"].reshape(-1).astype(int) for s in ("train", "val", "test")}
    S = {s: stats(X[s]) for s in X}

    rows = []
    for s in ("train", "val", "test"):
        for c in sorted(NAMES):
            k = y[s] == c
            rows.append(dict(split=s, cls=NAMES[c], n=int(k.sum()), share=round(100 * k.mean(), 1),
                             mean=S[s][k, 0].mean(), std=S[s][k, 1].mean(), cv=S[s][k, 2].mean(),
                             cv_sd=S[s][k, 2].std()))
    t = pd.DataFrame(rows)
    print("Part 1: average per image (pixels 0-1)")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    for s in ("train", "test"):
        nrm = t[(t.split == s) & (t.cls == "Normal")].iloc[0]
        r = {c: t[(t.split == s) & (t.cls == c)].iloc[0].cv / nrm.cv for c in NAMES.values()}
        print(f"  {s}: cv relative to Normal  " + "  ".join(f"{c} {v:.3f}" for c, v in r.items()))

    rng = np.random.default_rng(a.seed)
    conds = {"clean": dict(), "brightness": dict(bright=True), "bright + contrast 0.8": dict(bright=True, contrast=0.8),
             "bright + contrast 0.6": dict(bright=True, contrast=0.6)}
    feats = {"mean": [0], "std": [1], "cv": [2], "mean+std": [0, 1]}
    acc = []
    for c_name, kw in conds.items():
        Sc = S if not kw else {s: stats(X[s], rng, **kw) for s in X}
        for f_name, cols in feats.items():
            clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
            clf.fit(np.log(Sc["train"][:, cols] + 1e-6), y["train"])
            r = dict(augmentation=c_name, features=f_name)
            for s in ("val", "test"):
                r[f"bal_acc_{s}"] = 100 * balanced_accuracy_score(y[s], clf.predict(np.log(Sc[s][:, cols] + 1e-6)))
            acc.append(r)
    acc = pd.DataFrame(acc)
    print("\nPart 2: class predicted from intensity statistics alone (balanced accuracy %, chance 25)")
    for s in ("val", "test"):
        print(f"  {s}")
        print(acc.pivot(index="features", columns="augmentation", values=f"bal_acc_{s}")
              .reindex(index=list(feats), columns=list(conds)).round(1).to_string())

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "intensity_by_class.csv", index=False)
    acc.to_csv(out / "intensity_shortcut_acc.csv", index=False)
    print(f"\nwrote {out / 'intensity_by_class.csv'} and {out / 'intensity_shortcut_acc.csv'}")


if __name__ == "__main__":
    main()
