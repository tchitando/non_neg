#!/usr/bin/env python3
"""One figure linking augmentation, monosemanticity and performance (OCT, ResNet-50, NCL, validation).

    python study_tools/plot_aug_story.py                  # projector probe; writes study/summary/fig_aug_story.png and .pdf
    python study_tools/plot_aug_story.py --layer bb       # backbone probe instead

(a) What each augmentation does: factorial main effect of each operation on SC (x) and on accuracy (y), with 95% CIs.
    Main effects come from one regression over the factorial policies (stem, singles, pairs, five-op sets, all six),
    so each point is that operation's change averaged over every combination it appeared in.
(b) What turning an augmentation up does: contrast factor 1.0 (stem) -> 0.9 -> 0.8 -> 0.6, and crop minimum area
    0.08 (stem) -> 0.2 -> 0.5, each as policy minus stem, in the same axes as (a).
(c) Specialisation against accuracy across every policy; the fitted line and r use the policies with the default crop.
SC here is SC % minus its shuffled-label value (SC % = 100/C + (1 - 1/C) SC*), so the small-sample floor is removed.
Collapsed runs (backbone probe < 60) are left out, as in the workbook. Prints the mediation check: the effect of
contrast on accuracy with and without SC in the regression.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OPS = ["speckle", "contrast", "blur", "motion", "pixelate", "jpeg"]
NAMES = {"speckle": "speckle", "contrast": "contrast", "blur": "blur", "motion": "motion blur", "pixelate": "pixelation",
         "jpeg": "JPEG"}
BLUE, ORANGE, GREEN, GREY, INK, MUTED = "#0072B2", "#D55E00", "#009E73", "#8A8984", "#2C2C2A", "#5F5E5A"
C = 4


def load(master, spec):
    v = pd.read_csv(master)
    s = pd.read_csv(spec).drop_duplicates("run_id", keep="last")
    m = v.drop(columns=[c for c in ("sc_star", "ms") if c in v]).merge(s[["run_id", "sc_star_excess"]], on="run_id")
    m = m[~(m.probe_bb_bal_acc < 60)]
    m["scx"] = (1 - 1 / C) * m.sc_star_excess
    return m


def ols(d, cols, y):
    X = np.column_stack([np.ones(len(d))] + [d[c].values for c in cols])
    Y = d[y].values
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    r = Y - X @ b
    dof = len(Y) - X.shape[1]
    se = np.sqrt(np.diag((r @ r / dof) * np.linalg.inv(X.T @ X)))
    h = stats.t.ppf(0.975, dof) * se
    return {c: (b[i + 1], h[i + 1]) for i, c in enumerate(cols)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default="study/results_master.csv")
    ap.add_argument("--spec", default="study/results_spec.csv")
    ap.add_argument("--layer", choices=["proj", "bb"], default="proj")
    ap.add_argument("--out", default="study/summary/fig_aug_story")
    a = ap.parse_args()
    acc = "probe_proj_bal_acc" if a.layer == "proj" else "probe_bb_bal_acc"
    acc_name = "projector probe" if a.layer == "proj" else "backbone probe"

    m = load(a.master, a.spec)
    ncl = m[m.objective == "ncl"]
    fac = ncl[ncl.policy_id.apply(lambda p: p in ("stem", "all6") or all(x in OPS for x in p.split("_")))].copy()
    for o in OPS:
        fac[o] = fac.policy_id.apply(lambda p: 1 if p == "all6" else int(o in p.split("_")))
    eff_sc, eff_acc = ols(fac, OPS, "scx"), ols(fac, OPS, acc)
    fac_sc = fac.assign(sc=fac.scx)
    with_sc = ols(fac_sc, OPS + ["sc"], acc)
    pol = ncl.groupby("policy_id")[["scx", acc]].mean()
    d = lambda p: (pol.loc[p, "scx"] - pol.loc["stem", "scx"], pol.loc[p, acc] - pol.loc["stem", acc])

    print(f"{len(fac)} NCL runs over {fac.policy_id.nunique()} factorial policies")
    print(f"{'operation':12s} {'dSC':>12s} {'dAcc':>12s}")
    for o in OPS:
        print(f"{o:12s} {eff_sc[o][0]:+6.2f}±{eff_sc[o][1]:4.2f} {eff_acc[o][0]:+6.2f}±{eff_acc[o][1]:4.2f}")
    print(f"mediation: contrast on {acc_name} {eff_acc['contrast'][0]:+.2f}±{eff_acc['contrast'][1]:.2f} alone, "
          f"{with_sc['contrast'][0]:+.2f}±{with_sc['contrast'][1]:.2f} with SC in the model; "
          f"SC slope {with_sc['sc'][0]:+.2f}±{with_sc['sc'][1]:.2f} accuracy points per SC point")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left"})
    fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.3), gridspec_kw={"wspace": 0.38})
    xl, yl = "$\\Delta$ SC vs stem (points, val)", f"$\\Delta$ {acc_name} vs stem (points, val)"

    ax = axs[0]
    for o in OPS:
        (x, ex), (y, ey) = eff_sc[o], eff_acc[o]
        col = BLUE if o == "contrast" else GREY
        ax.errorbar(x, y, xerr=ex, yerr=ey, fmt="o", color=col, ms=6, mec="white", mew=0.8, elinewidth=1, capsize=0, zorder=3)
        off = {"contrast": (8, -12), "speckle": (22, 30), "jpeg": (34, 8), "blur": (-8, 34), "motion": (30, -30),
               "pixelate": (10, -14)}[o]
        far = o in ("speckle", "jpeg", "blur", "motion")
        ax.annotate(NAMES[o], (x, y), xytext=off, textcoords="offset points", fontsize=7.5, color=INK,
                    ha="right" if off[0] < 0 else "left", va="center",
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6, shrinkA=1, shrinkB=4) if far else None)
    ax.set_title("(a) Effect of each augmentation")

    ax = axs[1]
    paths = [("contrast factor", BLUE, "o", [("stem", "1.0"), ("contrast_f090", "0.9"), ("contrast", "0.8"), ("contrast_f060", "0.6")]),
             ("crop min. area", ORANGE, "s", [("stem", "0.08"), ("stem_crop020", "0.2"), ("stem_crop050", "0.5")])]
    for name, col, mk, pts in paths:
        pts = [(p, lab) for p, lab in pts if p in pol.index]
        xy = np.array([d(p) for p, _ in pts])
        ax.plot(xy[:, 0], xy[:, 1], color=col, lw=1.6, zorder=2)
        ax.scatter(xy[:, 0], xy[:, 1], color=col, marker=mk, s=34, edgecolor="white", linewidth=0.8, zorder=3)
        for (p, lab), (x, y) in zip(pts, xy):
            if p != "stem":
                o2 = {"0.9": (-4, 9), "0.8": (0, 9), "0.6": (7, -3), "0.2": (7, 0), "0.5": (7, 0)}.get(lab, (6, 2))
                ax.annotate(lab, (x, y), xytext=o2, textcoords="offset points", fontsize=7.5, color=INK,
                            ha="left" if o2[0] > 0 else "center", va="center")
        lab_xy = {"contrast factor": (1.6, 0.8), "crop min. area": (2.6, -10.5)}[name]
        ax.text(*lab_xy, name, fontsize=7.5, color=col, fontweight="bold", ha="left", va="center")
    if "cifar" in pol.index:
        x, y = d("cifar")
        ax.scatter([x], [y], color=GREEN, marker="D", s=30, edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate("natural-\nimage set", (x, y), xytext=(0, 7), textcoords="offset points", fontsize=7.5, color=INK,
                    ha="center", va="bottom")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, max(hi, y + 0.17 * (hi - lo)))
        ax.set_xlim(-0.4, max(8.3, x + 1.5))
    ax.annotate("stem", (0, 0), xytext=(6, -8), textcoords="offset points", fontsize=7.5, color=MUTED, ha="left")
    ax.set_title("(b) Effect of strength")

    for ax in axs[:2]:
        ax.axhline(0, color=GREY, lw=0.6, zorder=1)
        ax.axvline(0, color=GREY, lw=0.6, zorder=1)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)

    ax = axs[2]
    crop = pol.index.str.contains("crop")
    hasc = pol.index.to_series().apply(lambda p: p == "cifar" or "contrast" in p.split("_") or p == "all6").values
    for mask, col, lab, kw in ((~crop & hasc, BLUE, "contains contrast", {}), (~crop & ~hasc, GREY, "no contrast", {}),
                               (crop, ORANGE, "crop changed", {"facecolors": "none"})):
        sub = pol[mask]
        ax.scatter(sub.scx, sub[acc], s=26, marker="s" if lab == "crop changed" else "o",
                   edgecolors=col if lab == "crop changed" else "white", linewidth=1 if lab == "crop changed" else 0.6,
                   color=col if lab != "crop changed" else None, label=lab, zorder=3, **kw)
    base = pol[~crop]
    r, p = stats.pearsonr(base.scx, base[acc])
    k, c0 = np.polyfit(base.scx, base[acc], 1)
    xs = np.linspace(base.scx.min(), base.scx.max(), 10)
    ax.plot(xs, k * xs + c0, color=INK, lw=1, ls="--", zorder=2)
    ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.text(0.03, 0.30, f"default crop (n = {len(base)})\nr = {r:.2f}, {ptxt}", transform=ax.transAxes,
            fontsize=7.5, color=INK, va="top")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.12 * (hi - lo))
    ax.set_xlabel("SC (points above shuffled labels, val)")
    ax.set_ylabel(f"{acc_name.capitalize()} bal. acc. (%, val)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("(c) SC and accuracy, all policies")

    for ax in axs:
        ax.grid(color="#E4E2DA", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(a.out + ".pdf", bbox_inches="tight")
    print("wrote", a.out + ".png", a.out + ".pdf")


if __name__ == "__main__":
    main()
