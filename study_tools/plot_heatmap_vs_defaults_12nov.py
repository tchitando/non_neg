#!/usr/bin/env python3
"""The seed-42 composition heatmap (crop-only base, OCT, ResNet-50) shown against the SimCLR default augmentations
(policy 'cifar': crop 0.08, colour jitter 0.8/0.8/0.8/0.2 p 0.8, grayscale p 0.2, blur p 0.5, flip p 0.5).
Each cell is (cell - SimCLR defaults) in backbone probe h, both at seed 42, per method. Blue = above the defaults,
red = below, grey = level; one symmetric colour scale shared by the two panels.

    python study_tools/plot_heatmap_vs_defaults_12nov.py --master study/results_master.csv --out study/summary/figs_story
    python study_tools/plot_heatmap_vs_defaults_12nov.py --test study/test_results_octmnist.csv --out study/summary/figs_story
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

plt.rcParams.update({"font.family": "serif", "font.serif": ["Garamond", "EB Garamond", "Times New Roman", "DejaVu Serif"],
                     "font.size": 11, "axes.titlesize": 12})
INK, MUTED, EMPTY = "#222222", "#666666", "#ECEBE8"
DIV = LinearSegmentedColormap.from_list("div", ["#b22a2a", "#e34948", "#f0efec", "#3987e5", "#1c5cab"])
OPS = [("bright", "Brightness 0.8"), ("flip", "Flip"), ("blur", "Blur σ≤0.6"), ("contrast", "Contrast 0.3"),
       ("speckle", "Speckle 0.1"), ("jpeg", "JPEG 70")]
SINGLE = {"bright": "croponly_bright", "flip": "croponly_flip", "blur": "croponly_blur_sig060",
          "contrast": "croponly_contrast_f030", "speckle": "croponly_speckle", "jpeg": "croponly_jpeg"}
PART = {"bright": "bright", "contrast": "contrast_f030", "blur": "blur_sig060", "flip": "flip",
        "speckle": "speckle_s010", "jpeg": "jpeg_q70"}
ORDER = ["bright", "contrast", "blur", "flip", "speckle", "jpeg"]
NAME = {"ncl": "NCL", "simclr": "SimCLR"}


def pair_pid(a, b):
    if {a, b} == {"bright", "flip"}:
        return "stem"
    x, y = sorted([a, b], key=ORDER.index)
    return f"croponly_{PART[x]}_{PART[y]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default="study/results_master.csv")
    ap.add_argument("--test", default=None, help="test results csv; draws the test split")
    ap.add_argument("--out", default="study/summary/figs_story")
    ap.add_argument("--tag", default="12nov")
    a = ap.parse_args()
    split = "test" if a.test else "validation"
    m = pd.read_csv(a.test or a.master).drop_duplicates("run_id", keep="last")
    m = m[m.run_id.str.contains("resnet50-octmnist") & (m.seed == 42)]
    h = {(r.policy_id, r.objective): r.probe_bb_bal_acc for r in m.itertuples()}
    k = len(OPS)
    ref = {o: h[("cifar", o)] for o in NAME}
    D = {}
    for o in NAME:
        D[o] = np.full((k, k), np.nan)
        for i in range(k):
            for j in range(i + 1):
                pid = SINGLE[OPS[i][0]] if i == j else pair_pid(OPS[i][0], OPS[j][0])
                D[o][i, j] = h.get((pid, o), np.nan) - ref[o]
    lim = np.ceil(np.nanmax(np.abs(np.concatenate([D[o][np.isfinite(D[o])] for o in NAME]))))
    norm = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.4))
    labels = [n for _, n in OPS]
    tri = np.tril(np.ones((k, k), bool))
    for ax, o in zip(axes, NAME):
        A = D[o]
        ax.imshow(np.where(tri & ~np.isfinite(A), 1, np.nan), cmap=LinearSegmentedColormap.from_list("e", [EMPTY, EMPTY]))
        im = ax.imshow(np.where(tri, A, np.nan), cmap=DIV, norm=norm)
        for i in range(k):
            for j in range(i + 1):
                v = A[i, j]
                if np.isfinite(v):
                    rgb = np.array(im.cmap(im.norm(v))[:3])
                    dark = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] < 0.5
                    ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=10.5, color="white" if dark else INK)
                else:
                    ax.text(j, i, "pending", ha="center", va="center", fontsize=8.5, color=MUTED, style="italic")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(range(k), labels, rotation=40, ha="right"); ax.set_yticks(range(k), labels)
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-.5, k, 1), minor=True); ax.set_yticks(np.arange(-.5, k, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="minor", length=0)
        ax.set_title(f"{NAME[o]}: probe h minus SimCLR defaults ({ref[o]:.1f})", fontsize=12)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.outline.set_visible(False)
        cb.set_label("points vs SimCLR defaults", fontsize=10)
    fig.text(0.5, -0.03, f"OCT, ResNet-50, {split}, seed 42 for every cell and for the reference. Cells: crop only (0.08) plus the operations named; "
             "brightness + flip uses the stem runs.\nReference: SimCLR default augmentations (crop 0.08, colour jitter 0.8/0.8/0.8/0.2 p 0.8, "
             "grayscale p 0.2, Gaussian blur p 0.5, flip p 0.5), same training settings.\nBlue: above the defaults; red: below. "
             "One seed per cell: the 3-seed SD of probe h is about 0.1-0.7 on validation and 0.6-3.5 on test.",
             ha="center", va="top", fontsize=9.5, color="#555555")
    fig.tight_layout()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stem = f"fig6b_heatmap_vs_defaults_{'test' if a.test else 'val'}_seed42_{a.tag}"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", out / f"{stem}.png")


if __name__ == "__main__":
    main()
