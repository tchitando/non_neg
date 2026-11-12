#!/usr/bin/env python3
"""Composition heatmap on the crop-only base (OCT, ResNet-50), seed 42, in the layout of Chen et al. (2020), Fig. 5:
one operation on the diagonal, the pair in the lower triangle (both views get every operation, so the matrix is
symmetric and the upper triangle is left empty). Three panels: NCL, SimCLR (one shared colour scale) and NCL - SimCLR.
Cells without a scored seed-42 run are greyed and say why.

    python study_tools/plot_heatmap_11nov.py --master study/results_master.csv --out study/summary/figs_story --split val
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
SEQ = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#b22a2a", "#e34948", "#f0efec", "#3987e5", "#1c5cab"])

OPS = [("bright", "Brightness 0.8"), ("flip", "Flip"), ("blur", "Blur σ≤0.6"), ("contrast", "Contrast 0.3"),
       ("speckle", "Speckle 0.1"), ("jpeg", "JPEG 70")]
SINGLE = {"bright": "croponly_bright", "flip": "croponly_flip", "blur": "croponly_blur_sig060",
          "contrast": "croponly_contrast_f030", "speckle": "croponly_speckle", "jpeg": "croponly_jpeg"}
PART = {"bright": "bright", "contrast": "contrast_f030", "blur": "blur_sig060", "flip": "flip",
        "speckle": "speckle_s010", "jpeg": "jpeg_q70"}
ORDER = ["bright", "contrast", "blur", "flip", "speckle", "jpeg"]       # order of the parts in a pair's policy id


def pair_pid(a, b):
    if {a, b} == {"bright", "flip"}:
        return "stem"                                                   # crop + brightness 0.8 + flip = the stem
    x, y = sorted([a, b], key=ORDER.index)
    return f"croponly_{PART[x]}_{PART[y]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default="study/results_master.csv")
    ap.add_argument("--out", default="study/summary/figs_story")
    ap.add_argument("--tag", default="11nov")
    a = ap.parse_args()
    m = pd.read_csv(a.master).drop_duplicates("run_id", keep="last")
    m = m[m.run_id.str.contains("resnet50-octmnist") & (m.seed == 42)]
    val = {(r.policy_id, r.objective): r.probe_bb_bal_acc for r in m.itertuples()}
    status = {"croponly_bright_blur_sig060": "training", "croponly_blur_sig060_flip": "training"}

    k = len(OPS)
    M = {o: np.full((k, k), np.nan) for o in ("ncl", "simclr")}
    note = np.full((k, k), "", dtype=object)
    for i in range(k):
        for j in range(i + 1):
            pid = SINGLE[OPS[i][0]] if i == j else pair_pid(OPS[i][0], OPS[j][0])
            for o in ("ncl", "simclr"):
                M[o][i, j] = val.get((pid, o), np.nan)
            note[i, j] = status.get(pid, "to make")
    G = M["ncl"] - M["simclr"]
    both = np.concatenate([M["ncl"][np.isfinite(M["ncl"])], M["simclr"][np.isfinite(M["simclr"])]])
    lo, hi = np.floor(both.min()), np.ceil(both.max())
    g = np.nanmax(np.abs(G))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4))
    panels = [("NCL", M["ncl"], SEQ, dict(vmin=lo, vmax=hi)), ("SimCLR", M["simclr"], SEQ, dict(vmin=lo, vmax=hi)),
              ("NCL − SimCLR", G, DIV, dict(norm=TwoSlopeNorm(0, -g, g)))]
    labels = [n for _, n in OPS]
    for ax, (title, A, cmap, kw) in zip(axes, panels):
        tri = np.tril(np.ones((k, k), bool))
        ax.imshow(np.where(tri & ~np.isfinite(A), 1, np.nan), cmap=LinearSegmentedColormap.from_list("e", [EMPTY, EMPTY]))
        im = ax.imshow(np.where(tri, A, np.nan), cmap=cmap, **kw)
        for i in range(k):
            for j in range(i + 1):
                v = A[i, j]
                if np.isfinite(v):
                    rgb = np.array(im.cmap(im.norm(v))[:3])
                    dark = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] < 0.5
                    ax.text(j, i, f"{v:+.2f}" if title.startswith("NCL −") else f"{v:.1f}", ha="center", va="center",
                            fontsize=10.5, color="white" if dark else INK)
                else:
                    ax.text(j, i, note[i, j], ha="center", va="center", fontsize=8.5, color=MUTED, style="italic")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(range(k), labels, rotation=40, ha="right")
        ax.set_yticks(range(k), labels)
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-.5, k, 1), minor=True); ax.set_yticks(np.arange(-.5, k, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="minor", length=0)
        ax.set_title(title + ("" if title.startswith("NCL −") else ": backbone probe h (%)"), fontsize=12)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.outline.set_visible(False)
        if title.startswith("NCL −"):
            cb.set_label("points (blue: NCL ahead, red: SimCLR ahead)", color=MUTED)
    base = {o: val.get(("croponly", o), np.nan) for o in ("ncl", "simclr")}
    fig.text(0.5, -0.04, "OCT, ResNet-50, validation, seed 42. Every cell is crop only (crop min area 0.08) plus the operations named: "
             "one on the diagonal, two in the lower triangle; both views get every operation. Brightness + flip uses the stem runs.\n"
             f"Crop only at seed 42: NCL {base['ncl']:.1f}, SimCLR {base['simclr']:.1f}. Levels from the selection rule "
             "(best mean of NCL and SimCLR over seeds 42-44 on validation); speckle 0.1 and JPEG 70 are the rule's picks of 11 Nov. "
             "Grey cells: not trained or not scored yet.", ha="center", va="top", fontsize=9.5, color="#555555")
    fig.tight_layout()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig6_heatmap_val_seed42_{a.tag}.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", out / f"fig6_heatmap_val_seed42_{a.tag}.png")


if __name__ == "__main__":
    main()
