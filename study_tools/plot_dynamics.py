#!/usr/bin/env python3
"""Training-dynamics figure: accuracy (top) and SC (bottom) against epoch, one line per run.

    python study_tools/plot_dynamics.py                       # projector probe; writes study/dyn/dynamics.png and .pdf
    python study_tools/plot_dynamics.py --layer bb            # backbone probe instead

Colour = policy, line style = objective (NCL solid, SimCLR dashed). SC is shown for NCL only (SimCLR sits at
chance by construction); the dotted line marks chance (25%). Two panels share the epoch axis; no second y-axis.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLOURS = {"stem": "#5F5E5A", "contrast": "#0072B2", "cifar": "#D55E00", "motion": "#009E73"}   # Okabe-Ito + grey
MARKERS = {"stem": "o", "contrast": "s", "cifar": "^", "motion": "D"}
NAMES = {"stem": "Stem", "contrast": "Contrast", "cifar": "Natural-image set", "motion": "Motion"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="study/dyn/dyn_results.csv")
    ap.add_argument("--layer", choices=["proj", "bb"], default="proj")
    ap.add_argument("--out", default="study/dyn/dynamics")
    a = ap.parse_args()
    d = pd.read_csv(a.csv).sort_values("epoch")
    d["epoch_1"] = d["epoch"] + 1                      # epoch index 0..99 -> epochs completed 1..100
    acc = "probe_proj_bal_acc" if a.layer == "proj" else "probe_bb_bal_acc"
    acc_name = "Projector probe" if a.layer == "proj" else "Backbone probe"

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#888780", "axes.labelcolor": "#2C2C2A",
                         "xtick.color": "#5F5E5A", "ytick.color": "#5F5E5A"})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.6, 5.2), sharex=True, gridspec_kw={"hspace": 0.12})
    pols = [p for p in COLOURS if p in set(d.policy_id)]
    for p in pols:
        for obj, ls in (("ncl", "-"), ("simclr", "--")):
            s = d[(d.policy_id == p) & (d.objective == obj)]
            if s.empty:
                continue
            kw = dict(color=COLOURS[p], ls=ls, lw=1.6, marker=MARKERS[p], ms=4, mec="white", mew=0.6)
            ax1.plot(s.epoch_1, s[acc], label=f"{NAMES[p]}, {'NCL' if obj == 'ncl' else 'SimCLR'}", **kw)
            if obj == "ncl":
                ax2.plot(s.epoch_1, s.sc_pct, label=f"{NAMES[p]}, NCL", **kw)
    ax2.axhline(25, color="#888780", ls=":", lw=1)
    ax2.text(100, 25.6, "chance", ha="right", va="bottom", fontsize=8, color="#5F5E5A")
    ax1.set_ylabel(f"{acc_name},\nbalanced accuracy (val, %)")
    ax2.set_ylabel("SC (val, %)")
    ax2.set_xlabel("Epoch")
    ax2.set_xlim(0, 101)
    for ax in (ax1, ax2):
        ax.grid(axis="y", color="#D3D1C7", lw=0.5)
    ax1.legend(frameon=False, fontsize=7.5, ncol=1, loc="lower right")
    fig.savefig(a.out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(a.out + ".pdf", bbox_inches="tight")
    print("wrote", a.out + ".png", a.out + ".pdf")


if __name__ == "__main__":
    main()
