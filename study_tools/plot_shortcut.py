#!/usr/bin/env python3
"""Mechanism figure: the view shortcut against specialisation and accuracy, one point per policy.

    python study_tools/plot_shortcut.py                     # after view_shortcut.py and croponly_table.py
    python study_tools/plot_shortcut.py --stat mean+sd

x: how often a view's partner is found from its intensity statistics alone in a batch of 256 (view_shortcut.py,
log scale; chance 0.39%). y: NCL's SC above shuffle, then the backbone probe for NCL and for SimCLR (croponly_table.py
means over seeds; bars = seed SD). Points are the crop-only policies and the old stem, coloured and shaped by family and
labelled. Crop area 0.5 is off the scale (shortcut 51%, SimCLR probe 66.7) and left out of the plot and of r.
Output: study/summary/fig_shortcut.png / .pdf
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FAMILY = {  # policy -> (family, short label)
    "croponly": ("crop only", "crop only"), "croponly_flip": ("flip", "flip"),
    "croponly_bright_b020": ("brightness", "b0.2"), "croponly_bright_b040": ("brightness", "b0.4"),
    "croponly_bright": ("brightness", "b0.8"), "croponly_bright_b095": ("brightness", "b0.95"),
    "croponly_contrast": ("contrast", "c0.8"), "croponly_contrast_f050": ("contrast", "c0.5"),
    "croponly_contrast_f030": ("contrast", "c0.3"), "croponly_contrast_f010": ("contrast", "c0.1"),
    "croponly_blur_sig060": ("blur", "σ0.6"), "croponly_blur": ("blur", "σ1.2"), "croponly_blur_sig200": ("blur", "σ2.0"),
    "croponly_crop020": ("weaker crop", "area 0.2"), "stem": ("old stem", "old stem"),
    "contrast": ("old stem", "old stem + c0.8"),
}
STYLE = {  # family -> (colour, marker); colour plus marker plus label, so identity never rests on colour alone
    "crop only": ("#2C2C2A", "o"), "brightness": ("#2a78d6", "o"), "contrast": ("#eb6834", "s"),
    "blur": ("#1baf7a", "^"), "flip": ("#e87ba4", "D"), "weaker crop": ("#eda100", "v"), "old stem": ("#6250d6", "P"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shortcut", default="study/derived/view_shortcut.csv")
    ap.add_argument("--summary", default="study/derived/croponly_summary.csv")
    ap.add_argument("--stat", default="histogram", choices=["mean", "mean+sd", "quantiles", "histogram"])
    ap.add_argument("--out", default="study/summary/fig_shortcut")
    a = ap.parse_args()
    sh = pd.read_csv(a.shortcut)
    sh = sh[sh.statistic == a.stat].set_index("policy_id")["top1_pct"]
    sm = pd.read_csv(a.summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    GREY, INK, MUTED = "#8A8984", "#2C2C2A", "#5F5E5A"
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    panels = [("ncl", "sc_excess", "NCL: SC above shuffle (% points)"),
              ("ncl", "probe_bb", "NCL: backbone probe (bal. acc., %)"),
              ("simclr", "probe_bb", "SimCLR: backbone probe (bal. acc., %)")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    labels_per_ax = []
    for ax, (obj, col, title) in zip(axes, panels):
        placed = []
        d = sm[(sm.objective == obj) & sm.policy_id.isin(FAMILY)].set_index("policy_id")
        d = d.join(sh.rename("x"), how="inner").dropna(subset=[f"{col}_mean", "x"])
        ax.grid(color="#E4E2DA", lw=0.5, zorder=0)
        for pid, r in d.iterrows():
            fam, lab = FAMILY[pid]
            c, m = STYLE[fam]
            sd = r.get(f"{col}_sd")
            if sd == sd and sd is not None:
                ax.errorbar(r.x, r[f"{col}_mean"], yerr=sd, fmt="none", ecolor=c, elinewidth=1, alpha=0.6, zorder=2)
            ax.scatter(r.x, r[f"{col}_mean"], color=c, marker=m, s=40, edgecolor="white", linewidth=1, zorder=3)
            placed.append((lab, r.x, r[f"{col}_mean"]))
        rr = np.corrcoef(np.log(d.x), d[f"{col}_mean"])[0, 1] if len(d) > 2 else float("nan")
        ax.set_xscale("log")
        if len(d):
            ax.set_xlim(d.x.min() / 1.25, d.x.max() * 1.6)   # room for the labels of the right-most points
        ax.xaxis.set_major_locator(FixedLocator([2, 3, 5, 10, 20, 30, 50]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
        labels_per_ax.append((ax, placed))
        ax.set_title(f"{title}\nr = {rr:+.2f} (log shortcut, {len(d)} policies)", fontsize=8.5, color=INK)
        ax.set_xlabel(f"view shortcut: partner found from the intensity {a.stat} (%)")
    handles = [plt.Line2D([], [], color=c, marker=m, ls="", markersize=6, label=f) for f, (c, m) in STYLE.items()]
    fig.legend(handles=handles, loc="upper center", ncol=len(STYLE), frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=7.5)
    fig.text(0.5, 0.005, "Validation, OCT, ResNet-50; means over seeds (bars: seed SD). Shortcut: share of views whose partner "
             "is the nearest neighbour among 256 by intensity statistics alone (chance 0.39%). Crop area 0.5 is off the scale.",
             ha="center", fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    # place each point's label at the first offset that overlaps neither earlier labels nor other points
    tries = [(5, 3, "left"), (5, -10, "left"), (-5, 3, "right"), (-5, -10, "right"), (0, 9, "center"), (0, -15, "center"),
             (8, 12, "left"), (8, -18, "left"), (-8, 12, "right"), (-8, -18, "right")]
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for ax, placed in labels_per_ax:
        pts = [ax.transData.transform((x, y)) for _l, x, y in placed]
        boxes = []
        for (lab, x, y), (px, py) in sorted(zip(placed, pts), key=lambda t: -t[1][1]):
            best = None
            for dx, dy, ha in tries:
                t = ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7, color=INK, ha=ha)
                bb = t.get_window_extent(rend).expanded(1.05, 1.15)
                hit = any(bb.overlaps(b) for b in boxes) or any(bb.contains(qx, qy) for qx, qy in pts)
                if not hit:
                    best = t; break
                t.remove()
            if best is None:
                best = ax.annotate(lab, (x, y), xytext=(5, 3), textcoords="offset points", fontsize=7, color=INK)
            boxes.append(best.get_window_extent(rend))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200)
    print(f"wrote {out.with_suffix('.png')} (and .pdf)")


if __name__ == "__main__":
    main()
