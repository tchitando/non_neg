#!/usr/bin/env python3
"""Version b (12 Nov): the SimCLR default augmentations (policy 'cifar', same seed) are drawn as a reference cell in the
empty top-right corner of each panel, on the same colour scale, and every heatmap cell above the defaults is outlined.

Composition heatmap on the crop-only base (OCT, ResNet-50), seed 42, in the layout of Chen et al. (2020), Fig. 5:
one operation on the diagonal, the pair in the lower triangle (both views get every operation, so the matrix is
symmetric and the upper triangle is left empty). Three panels, magma: backbone probe h for NCL and for SimCLR (one shared
colour scale) and NCL's SC % (its own scale; SimCLR's SC sits at chance by construction). Cells without a scored
seed-42 run are grey and marked pending.

    python study_tools/plot_heatmap_12nov.py --master study/results_master.csv --out study/summary/figs_story
    python study_tools/plot_heatmap_12nov.py --test study/test_results_octmnist.csv --out study/summary/figs_story
With --test the cells are read from the test results (eval_test.py, runs listed in study/test_plan_heatmap.csv):
probe h on the test split, and NCL's raw SC % = 100/C + (1 - 1/C) SC* (test stores SC* in %).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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
    ap.add_argument("--tag", default="12nov_b")
    ap.add_argument("--test", default=None, help="test results csv; draws the test-split heatmap")
    a = ap.parse_args()
    split = "test" if a.test else "validation"
    m = pd.read_csv(a.test or a.master).drop_duplicates("run_id", keep="last")
    m = m[m.run_id.str.contains("resnet50-octmnist") & (m.seed == 42)]
    val = {(r.policy_id, r.objective): r.probe_bb_bal_acc for r in m.itertuples()}
    if a.test:
        C = 4
        scv = {(r.policy_id, r.objective): 100 / C + (1 - 1 / C) * r.sc_star for r in m.itertuples()}
    else:
        scv = {(r.policy_id, r.objective): r.sc for r in m.itertuples()}

    k = len(OPS)
    M = {o: np.full((k, k), np.nan) for o in ("ncl", "simclr", "ncl_sc")}
    for i in range(k):
        for j in range(i + 1):
            pid = SINGLE[OPS[i][0]] if i == j else pair_pid(OPS[i][0], OPS[j][0])
            for o in ("ncl", "simclr"):
                M[o][i, j] = val.get((pid, o), np.nan)
            M["ncl_sc"][i, j] = scv.get((pid, "ncl"), np.nan)
    REF = {"ncl": val.get(("cifar", "ncl"), np.nan), "simclr": val.get(("cifar", "simclr"), np.nan),
           "ncl_sc": scv.get(("cifar", "ncl"), np.nan)}
    both = np.concatenate([M["ncl"][np.isfinite(M["ncl"])], M["simclr"][np.isfinite(M["simclr"])],
                           [REF["ncl"], REF["simclr"]]])
    lo, hi = np.floor(both.min()), np.ceil(both.max())
    sc = np.concatenate([M["ncl_sc"][np.isfinite(M["ncl_sc"])], [REF["ncl_sc"]]])
    slo, shi = np.floor(sc.min()), np.ceil(sc.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4))
    panels = [("NCL: backbone probe h (%)", M["ncl"], "magma", dict(vmin=lo, vmax=hi), REF["ncl"]),
              ("SimCLR: backbone probe h (%)", M["simclr"], "magma", dict(vmin=lo, vmax=hi), REF["simclr"]),
              ("NCL: SC (%)", M["ncl_sc"], "magma", dict(vmin=slo, vmax=shi), REF["ncl_sc"])]
    labels = [n for _, n in OPS]
    from matplotlib.patches import Rectangle
    for ax, (title, A, cmap, kw, ref) in zip(axes, panels):
        tri = np.tril(np.ones((k, k), bool))
        ax.imshow(np.where(tri & ~np.isfinite(A), 1, np.nan), cmap=LinearSegmentedColormap.from_list("e", [EMPTY, EMPTY]))
        im = ax.imshow(np.where(tri, A, np.nan), cmap=cmap, **kw)
        for i in range(k):
            for j in range(i + 1):
                v = A[i, j]
                if np.isfinite(v):
                    rgb = np.array(im.cmap(im.norm(v))[:3])
                    dark = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] < 0.5
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            fontsize=10.5, color="white" if dark else INK)
                    if np.isfinite(ref) and v > ref:          # above the SimCLR defaults: outlined
                        ax.add_patch(Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False, ec="#1c5cab", lw=2.2))
                else:
                    ax.text(j, i, "pending", ha="center", va="center", fontsize=8.5, color=MUTED, style="italic")
        if np.isfinite(ref):                                # reference cell in the empty top-right corner
            rc, rr = k - 1, 0
            rgb = np.array(im.cmap(im.norm(ref))[:3])
            dark = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] < 0.5
            ax.add_patch(Rectangle((rc - 0.5, rr - 0.5), 1, 1, fc=im.cmap(im.norm(ref)), ec=INK, lw=1.2, zorder=3))
            ax.text(rc, rr, f"{ref:.1f}", ha="center", va="center", fontsize=10.5, zorder=4,
                    color="white" if dark else INK)
            ax.text(rc - 0.6, rr, "SimCLR\ndefaults", ha="right", va="center", fontsize=9.5, zorder=4, color=INK,
                    linespacing=1.1)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(range(k), labels, rotation=40, ha="right")
        ax.set_yticks(range(k), labels)
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-.5, k, 1), minor=True); ax.set_yticks(np.arange(-.5, k, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="minor", length=0)
        ax.set_title(title, fontsize=12)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.outline.set_visible(False)
    base = {o: val.get(("croponly", o), np.nan) for o in ("ncl", "simclr")}
    base_sc = scv.get(("croponly", "ncl"), np.nan)
    fig.text(0.5, -0.04, f"OCT, ResNet-50, {split}, seed 42. Every cell is crop only (crop min area 0.08) plus the operations named: "
             "one on the diagonal, two in the lower triangle; both views get every operation. Brightness + flip uses the stem runs.\n"
             f"Crop only at seed 42: probe h NCL {base['ncl']:.1f}, SimCLR {base['simclr']:.1f}; NCL SC {base_sc:.1f}. SimCLR's SC is 25 (chance) in every cell.\nTop-right cell: SimCLR default augmentations (crop 0.08, colour jitter 0.8/0.8/0.8/0.2 p 0.8, grayscale p 0.2, blur p 0.5, flip p 0.5), same seed; blue outline: cell above the defaults.\nLevels from the selection rule "
             "(best mean of NCL and SimCLR over seeds 42-44 on validation); speckle 0.1 and JPEG 70 are the rule's picks of 11 Nov. "
             + ("\nTest scored only for runs in study/test_plan_heatmap.csv, fixed 12 Nov before scoring. " if a.test else "") + "Grey cells: pending (not trained or not scored yet).", ha="center", va="top", fontsize=9.5, color="#555555")
    fig.tight_layout()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig6_heatmap_{'test' if a.test else 'val'}_seed42_{a.tag}.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", out / f"fig6_heatmap_{'test' if a.test else 'val'}_seed42_{a.tag}.png")


if __name__ == "__main__":
    main()
