#!/usr/bin/env python3
"""Three figures for the augmentation story, each showing one relationship (OCT, ResNet-50).

Run from the Non_neg folder:
    python study_tools/plot_story.py --workbook study/summary/TC_results_5nov_e.xlsx --out study/summary/figs_story

  fig1_probe_vs_strength  augmentation strength -> backbone probe h, NCL and SimCLR (brightness and contrast sweeps, seeds 42-44)
  fig2_sc_vs_accuracy    NCL specialisation (raw SC %) -> probe h, one point per policy, same split on both axes,
                         coloured by design group; fig2_policies.csv lists every plotted policy
  fig3_backbone_vs_projector   change in backbone probe h -> change in projector probe z on test, crop-only design,
                               each policy vs crop only on matched seeds (above the diagonal: the projector gains more)
  fig4_sc_vs_strength    augmentation strength -> NCL raw SC % (brightness and contrast sweeps, seeds 42-44, validation and test)
  fig5_sweep_curves_val / _test   sweep curves: SC %, MS, backbone and projector probe against strength for brightness,
                         contrast, blur and crop area, NCL and SimCLR, every seed plus the mean (one figure per split)
  fig6_composition_val / _test    composition heatmap after the SimCLR paper (Fig. 5): stem + one or two of the six pool
                         augmentations, absolute values on seed 42 with a row-average column (probe h for NCL and SimCLR, NCL SC)
  fig7_family_accuracy   story sheet 3: probe h per augmentation added to crop only, NCL and SimCLR, validation and test
  fig8_backbone_projector_bars   story sheet 4: backbone and projector changes on test per comparison
  fig9_validation_vs_test        story sheet 6: the same augmentation's change on validation and on test
  (story sheet 2 = fig1, sheet 5 = fig2)
  fig10_family_trends    NCL: change in accuracy, SC and MS against strength, one line per augmentation, coloured by family
  fig11_view_shortcut    the view shortcut (intensity cue, view_shortcut.py) against NCL raw SC and the backbone probe
  fig12_method_accuracy  backbone probe h per augmentation, NCL and SimCLR mean ± SD, validation and test, with the NCL − SimCLR difference
  fig13_backbone_vs_projector    NCL test h and z changes vs matched reference, OCT and PathMNIST (dumbbell)
  fig14_strength_accuracy_sc     fig1 and fig4 combined: brightness/contrast strength -> probe h (val, test) and NCL SC
  fig15_sc_change_vs_accuracy_change    NCL change in SC vs change in h and z, three-seed policies vs matched reference
Reads the '8 Runs (val)' and '8 Runs (test)' sheets of the results workbook. Saves PNG and PDF.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

NCL, SIM = "#1F5FAD", "#B5651D"
SEEDS = [42, 43, 44]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Garamond", "EB Garamond", "Times New Roman", "DejaVu Serif"],
                     "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#444444",
                     "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.6, "legend.frameon": False,
                     "xtick.color": "#444444", "ytick.color": "#444444"})


def load(path, test_ms=None):
    v = pd.read_excel(path, sheet_name="8 Runs (val)")
    t = pd.read_excel(path, sheet_name="8 Runs (test)")
    v = v[(v.Dataset == "OCT") & (v.Backbone == "ResNet-50") & (v.Collapsed == "no")]
    t = t[(t.Dataset == "OCT") & (t.Backbone == "ResNet-50") & (t["Probe backbone"] >= 60)]
    keys = ["Policy id", "Objective", "Seed"]
    v = v[keys + ["Probe backbone", "Probe projector", "SC %", "MS"]].rename(
        columns={"Probe backbone": "val_h", "Probe projector": "val_z", "SC %": "val_sc", "MS": "val_ms"})
    t = t[keys + ["Probe backbone", "Probe projector", "SC*"]].rename(columns={"Probe backbone": "test_h", "Probe projector": "test_z"})
    t["test_sc"] = 25 + 0.75 * t.pop("SC*")                       # raw SC % from SC* (4 classes): SC = 1/C + (1 - 1/C) SC*
    d = v.merge(t, on=keys, how="outer")
    if test_ms and Path(test_ms).exists():                       # test MS per run (eval_test_ms.py output)
        m = pd.read_csv(test_ms)[["policy_id", "objective", "seed", "ms"]].rename(
            columns={"policy_id": "Policy id", "objective": "Objective", "seed": "Seed", "ms": "test_ms"})
        d = d.merge(m, on=keys, how="left")
    else:
        d["test_ms"] = np.nan
        print("test MS file not found: the test grid leaves its MS row empty")
    return d


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / f"{name}.png")


def fig1(d, out):
    """Backbone probe h for NCL and SimCLR against brightness and contrast strength, validation and test, seeds 42-44
    (line = mean, dots = seeds). The gap between the two lines is NCL's accuracy gap to SimCLR."""
    sweeps = [("Brightness strength", [("crop\nonly", "croponly"), ("0.2", "croponly_bright_b020"), ("0.4", "croponly_bright_b040"),
                                       ("0.8", "croponly_bright"), ("0.95", "croponly_bright_b095")]),
              ("Contrast factor (lower = stronger)", [("crop\nonly", "croponly"), ("0.8", "croponly_contrast"), ("0.5", "croponly_contrast_f050"),
                                                      ("0.3", "croponly_contrast_f030"), ("0.1", "croponly_contrast_f010")])]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharey="row")
    for i, (split, col) in enumerate((("validation", "val_h"), ("test", "test_h"))):
        for j, (name, levels) in enumerate(sweeps):
            ax = axes[i, j]
            for obj, colr, mk, lab, off in (("ncl", NCL, "o", "NCL", -0.06), ("simclr", SIM, "s", "SimCLR", 0.06)):
                means = []
                for k, (_, pid) in enumerate(levels):
                    v = d[(d["Policy id"] == pid) & (d.Objective == obj) & d.Seed.isin(SEEDS)][col].dropna()
                    ax.scatter(np.full(len(v), k) + off, v, s=12, color=colr, alpha=0.4, marker=mk, zorder=2)
                    means.append(v.mean())
                ax.plot(range(len(levels)), means, "-", marker=mk, color=colr, ms=7, lw=2, label=lab, zorder=3)
            ax.set_xticks(range(len(levels)), [l for l, _ in levels])
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(name.split(" (")[0])
            if i == 1:
                ax.set_xlabel(name)
            if j == 0:
                ax.set_ylabel(f"backbone probe h, {split} (%)")
    axes[0, 0].legend(loc="lower right")
    fig.text(0.5, -0.01, "OCT, ResNet-50, seeds 42-44 (line = mean, dots = seeds). The distance between the lines is NCL's gap to SimCLR.",
             ha="center", fontsize=9, color="#666666")
    fig.tight_layout()
    save(fig, out, "fig1_probe_vs_strength")


GROUPS = [("croponly", "Crop only, and crop only + one change", "#0072B2"),
          ("stem1", "Stem, and stem + one augmentation", "#009E73"),
          ("stem2", "Stem + two augmentations", "#CC79A7"),
          ("stem56", "Stem + five or six augmentations", "#E69F00"),
          ("crop", "Stem with crop area 0.2 or 0.5", "#D55E00"),
          ("cifar", "Natural-image set (colour jitter, grayscale)", "#222222")]


def policy_group(pid):
    """Which design a policy comes from (used to colour fig2 and in fig2_policies.csv)."""
    if pid.startswith("croponly"):
        return "croponly"
    if pid == "cifar":
        return "cifar"
    if "crop0" in pid:
        return "crop"
    if pid == "stem" or pid.startswith("contrast_f") or "_" not in pid and pid != "all6":
        return "stem1"
    return "stem2" if pid.count("_") == 1 else "stem56"


def fig2(d, out):
    """NCL policy means: SC % against backbone probe h, on the same split (validation SC with validation probe, test SC
    with test probe), coloured by design group, large markers = 3+ seeds. r over all policies in the panel title; the
    caption gives r without the two crop-0.5 policies. Also writes fig2_policies.csv listing every plotted policy."""
    g = d[d.Objective == "ncl"].groupby("Policy id")
    n = g[["val_sc", "val_h", "test_sc", "test_h"]].mean().join(g.Seed.nunique().rename("seeds"))
    n = n.dropna(subset=["val_sc", "val_h"])
    n["group"] = [policy_group(p_) for p_ in n.index]
    gname = {k: lab for k, lab, _ in GROUPS}; gcol = {k: c for k, _, c in GROUPS}
    mark = {"croponly": "crop only", "stem": "stem", "croponly_bright_b095": "crop only + brightness 0.95",
            "croponly_crop020": "crop only, crop area 0.2", "stem_crop050": "stem, crop area 0.5",
            "contrast_crop050": "stem + contrast, crop area 0.5"}
    where = {"val_sc": {"croponly": (50.3, 78.6, "left"), "stem": (54.4, 87.6, "left"), "croponly_bright_b095": (60.6, 83.9, "left"),
                        "croponly_crop020": (53.3, 76.3, "left"), "stem_crop050": (54.3, 71.3, "left"),
                        "contrast_crop050": (58.3, 73.9, "left")},
             "test_sc": {"croponly": (55.0, 68.2, "left"), "stem": (66.3, 71.4, "left"), "croponly_bright_b095": (56.0, 79.6, "left"),
                         "croponly_crop020": (69.6, 67.4, "left"), "stem_crop050": (69.0, 70.0, "right"),
                         "contrast_crop050": (79.4, 70.2, "right")}}
    crop050 = ["stem_crop050", "contrast_crop050"]
    def rci(m, xc, yc):
        r, _ = pearsonr(m[xc], m[yc]); z, se = np.arctanh(r), 1 / np.sqrt(len(m) - 3)
        return r, np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se), len(m)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.78, bottom=0.30, wspace=0.18)
    notes = []
    for ax, (xc, yc, split) in zip(axes, (("val_sc", "val_h", "validation"), ("test_sc", "test_h", "test"))):
        m = n[[xc, yc, "group", "seeds"]].dropna()
        r, lo, hi, k = rci(m, xc, yc)
        r2, lo2, hi2, k2 = rci(m.drop([p_ for p_ in crop050 if p_ in m.index]), xc, yc)
        b = np.polyfit(m[xc], m[yc], 1); xs = np.linspace(m[xc].min(), m[xc].max(), 50)
        ax.plot(xs, np.polyval(b, xs), color="#999999", lw=1, ls="--", zorder=1)
        for key, lab, c in GROUPS:
            q = m[m.group == key]
            ax.scatter(q[xc], q[yc], s=np.where(q.seeds >= 3, 46, 18), color=c, alpha=0.85, zorder=3,
                       edgecolors="white", linewidths=0.5)
        for pid, lab in mark.items():
            if pid in m.index:
                x0, y0 = m.loc[pid, xc], m.loc[pid, yc]
                ax.scatter([x0], [y0], s=95, facecolors="none", edgecolors="black", linewidths=0.9, zorder=4)
                tx, ty, ha = where[xc][pid]
                ax.annotate(lab, (x0, y0), xytext=(tx, ty), textcoords="data", fontsize=8.5, ha=ha, va="center",
                            color="#222222", arrowprops=dict(arrowstyle="-", color="#888888", lw=0.7, shrinkA=2, shrinkB=5))
        within = []
        for gl, keys in (("crop-only", ["croponly"]), ("stem-based", ["stem1", "stem2", "stem56"])):
            q = m[m.group.isin(keys)]
            within.append(f"within {gl}: r = {pearsonr(q[xc], q[yc])[0]:+.2f} ({len(q)})")
        ax.set_title(f"{split.capitalize()}: r = {r:+.2f} [{lo:+.2f}, {hi:+.2f}], {k} policies\n" + "; ".join(within).replace("-0", "−0"),
                     fontsize=10.5)
        ax.set_xlabel(f"NCL raw SC %, {split}")
        ax.set_ylabel(f"NCL backbone probe h, {split} (%)")
        notes.append(f"{split} r = {r2:+.2f} [{lo2:+.2f}, {hi2:+.2f}]")
    cnt = n.group.value_counts()
    h = [Line2D([0], [0], marker="o", color=c, ls="", ms=8, label=f"{lab} ({cnt.get(k_, 0)})") for k_, lab, c in GROUPS]
    h += [Line2D([0], [0], marker="o", color="#777777", ls="", ms=8, label="3 or more seeds"),
          Line2D([0], [0], marker="o", color="#777777", ls="", ms=5, label="1 or 2 seeds")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=4, fontsize=9.5, columnspacing=1.4)
    fig.suptitle("NCL's specialisation and accuracy across policies (one point per policy, policy mean over its seeds)", y=0.97)
    fig.text(0.5, 0.005, "OCT, ResNet-50. Dashed line = fit over all policies. Without the two crop-0.5 policies (one seed each, "
             "far below the line): " + "; ".join(notes) + ".\nPolicies are not independent (many share a base and augmentations), "
             "so the intervals are indicative. The full list is in fig2_policies.csv.",
             ha="center", va="bottom", fontsize=9, color="#666666")
    save(fig, out, "fig2_sc_vs_accuracy")
    t = n.assign(group=n.group.map(gname)).reset_index().rename(columns={"Policy id": "policy"})
    t = t[["group", "policy", "seeds", "val_sc", "val_h", "test_sc", "test_h"]].round(2)
    t["order"] = t.group.map({lab: i for i, (_, lab, _) in enumerate(GROUPS)})
    t.sort_values(["order", "policy"]).drop(columns="order").to_csv(Path(out) / "fig2_policies.csv", index=False)


def fig3(d, out):
    """Crop-only design: per policy, change in test h (x) and test z (y) vs crop only on matched seeds."""
    pids = sorted(p for p in d["Policy id"].dropna().unique() if str(p).startswith("croponly_"))
    base = d[d["Policy id"] == "croponly"].set_index(["Objective", "Seed"])
    names = {"croponly_bright": "brightness 0.8", "croponly_contrast": "contrast 0.8", "croponly_blur": "blur",
             "croponly_flip": "flip", "croponly_crop020": "crop 0.2"}
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for obj, colr in (("ncl", NCL), ("simclr", SIM)):
        pts = []
        for pid in pids:
            x = d[(d["Policy id"] == pid) & (d.Objective == obj)].set_index("Seed")
            sh = [s for s in x.index if (obj, s) in base.index]
            if not sh:
                continue
            dh = np.nanmean([x.loc[s, "test_h"] - base.loc[(obj, s), "test_h"] for s in sh])
            dz = np.nanmean([x.loc[s, "test_z"] - base.loc[(obj, s), "test_z"] for s in sh])
            if np.isfinite(dh) and np.isfinite(dz):
                pts.append((pid, dh, dz, len(sh)))
        p = pd.DataFrame(pts, columns=["pid", "dh", "dz", "n"])
        ax.scatter(p.dh, p.dz, s=np.where(p.n >= 3, 36, 14), color=colr, alpha=0.75,
                   label=f"{'NCL' if obj == 'ncl' else 'SimCLR'} (large = 3+ seeds)")
        if obj == "ncl":
            for _, q in p[p.pid.isin(names)].iterrows():
                ax.annotate(names[q.pid], (q.dh, q.dz), xytext=(4, 3), textcoords="offset points", fontsize=8)
    lim = ax.get_xlim() + ax.get_ylim()
    lo, hi = min(lim), max(lim)
    ax.plot([lo, hi], [lo, hi], color="grey", lw=0.8, ls=":")
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("Change in backbone probe h, test (pp)")
    ax.set_ylabel("Change in projector probe z, test (pp)")
    ax.set_title("Crop-only design, vs crop only on matched seeds\nabove the dotted line: the projector gains more")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, out, "fig3_backbone_vs_projector")


def fig4(d, out):
    """NCL raw SC % against augmentation strength, per seed and mean, validation and test (brightness and contrast sweeps)."""
    sweeps = [("Brightness strength", [("crop only", "croponly"), ("0.2", "croponly_bright_b020"), ("0.4", "croponly_bright_b040"),
                                       ("0.8", "croponly_bright"), ("0.95", "croponly_bright_b095")]),
              ("Contrast factor (lower = stronger)", [("crop only", "croponly"), ("0.8", "croponly_contrast"), ("0.5", "croponly_contrast_f050"),
                                                      ("0.3", "croponly_contrast_f030"), ("0.1", "croponly_contrast_f010")])]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (name, levels) in zip(axes, sweeps):
        for split, col, ls, mk in (("validation", "val_sc", "-", "o"), ("test", "test_sc", "--", "s")):
            means = []
            for i, (lab, pid) in enumerate(levels):
                x = d[(d["Policy id"] == pid) & (d.Objective == "ncl") & d.Seed.isin(SEEDS)][col].dropna()
                ax.scatter(np.full(len(x), i) + (-0.08 if split == "validation" else 0.08), x, s=16, marker=mk,
                           facecolors="none", edgecolors=NCL, alpha=0.55, linewidths=0.9, zorder=2)
                means.append(x.mean())
            ax.plot(range(len(levels)), means, ls, marker=mk, color=NCL, label=split, zorder=3)
        ax.set_xticks(range(len(levels)), [l for l, _ in levels])
        ax.set_xlabel(name)
        ax.set_title(name.split(" (")[0])
    axes[0].set_ylabel("NCL SC (%)")
    axes[0].legend(handles=[Line2D([0], [0], ls="-", marker="o", color=NCL, label="validation, mean"),
                            Line2D([0], [0], ls="--", marker="s", color=NCL, label="test, mean"),
                            Line2D([0], [0], ls="", marker="o", mfc="none", mec=NCL, alpha=0.55, label="one seed (42, 43, 44)")],
                   frameon=False, loc="lower right", fontsize=9)
    fig.suptitle("NCL's class selectivity against augmentation strength", y=1.06)
    fig.text(0.5, 0.985, "Every point is NCL. SimCLR's raw SC is 25% at every level, so it is not drawn.", ha="center",
             fontsize=10, color="#555555")
    save(fig, out, "fig4_sc_vs_strength")


def fig5(d, out, split):
    """Sweep curves: rows = SC %, backbone probe h, projector probe z (MS left out: it barely moves within a sweep); columns = brightness, contrast, blur, crop area.
    Large markers: mean over valid seeds (hollow = one seed); small: single seeds. Crop only is the base of every sweep.
    The crop column has its own y-axes (SimCLR's crop-0.5 run would otherwise squash the other columns) and evenly
    spaced levels; NCL's crop-0.5 run diverged and is marked."""
    import matplotlib.transforms as mtrans
    p = "val" if split == "validation" else "test"
    cols = [("Brightness", "brightness strength", [(0, "croponly"), (0.2, "croponly_bright_b020"), (0.4, "croponly_bright_b040"),
                                                   (0.8, "croponly_bright"), (0.95, "croponly_bright_b095")], False, None),
            ("Contrast", "contrast factor (lower = stronger)", [(1.0, "croponly"), (0.8, "croponly_contrast"), (0.5, "croponly_contrast_f050"),
                                                                (0.3, "croponly_contrast_f030"), (0.1, "croponly_contrast_f010")], True, None),
            ("Blur", "blur sigma max", [(0, "croponly"), (0.6, "croponly_blur_sig060"), (1.2, "croponly_blur"), (2.0, "croponly_blur_sig200")],
             False, None),
            ("Crop strength", "crop min area (higher = weaker crop)", [(0, "croponly"), (1, "croponly_crop020"), (2, "croponly_crop050")],
             False, ["0.08\n(crop only)", "0.2", "0.5\n(NCL diverged)"])]
    rows = [(f"{p}_sc", "SC (%)"), (f"{p}_h", "backbone probe h\n(bal. acc., %)"), (f"{p}_z", "projector probe z\n(bal. acc., %)")]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(13, 8.6), sharex="col")
    for i, (col, ylab) in enumerate(rows):
        for j, (name, xlab, levels, invert, ticklabels) in enumerate(cols):
            ax = axes[i, j]
            for obj, colr, mk, lab in (("ncl", NCL, "o", "NCL"), ("simclr", SIM, "s", "SimCLR")):
                xs, ms, ns = [], [], []
                for x, pid in levels:
                    v = d[(d["Policy id"] == pid) & (d.Objective == obj)][col].dropna()
                    if len(v):
                        ax.scatter(np.full(len(v), x), v, s=10, color=colr, alpha=0.35, marker=mk, zorder=2)
                        xs.append(x); ms.append(v.mean()); ns.append(len(v))
                if xs:
                    ax.plot(xs, ms, "-", color=colr, lw=1.6, zorder=3, label=lab)
                    for x, m_, n_ in zip(xs, ms, ns):
                        ax.scatter([x], [m_], s=40, marker=mk, zorder=4, edgecolors=colr, linewidths=1.5,
                                   facecolors="white" if n_ == 1 else colr)
            if ticklabels:                                         # crop column: NCL at 0.5 diverged (said in the tick label)
                ax.set_xticks(range(len(ticklabels)), ticklabels)
                ax.set_xlim(-0.4, 2.4)
            ax.grid(alpha=0.3)
            if invert and i == 0:
                ax.invert_xaxis()
            if i == 0:
                ax.set_title(name, fontsize=11)
            if j == 0:
                ax.set_ylabel(ylab)
            if i == len(rows) - 1:
                ax.set_xlabel(xlab)
        # share y across the first three columns of the row; the crop column keeps its own range
        lo = min(axes[i, j].get_ylim()[0] for j in range(3)); hi = max(axes[i, j].get_ylim()[1] for j in range(3))
        for j in range(3):
            axes[i, j].set_ylim(lo, hi)
            if j > 0:
                axes[i, j].tick_params(labelleft=False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.text(0.5, -0.01, f"OCT, ResNet-50, {split}. Crop only (crop min area 0.08, nothing else) is the first point of each curve "
             "(the contrast axis runs from weak to strong). Large markers: mean over valid seeds, hollow = one seed; small: single seeds. "
             "The crop column has its own y-axis. NCL at crop area 0.5 diverged (NaN features), so it has no NCL point.", ha="center", fontsize=9, color="#555555", wrap=True)
    fig.tight_layout()
    save(fig, out, f"fig5_sweep_curves_{p}")


def fig6(d, out, split):
    """Composition heatmap in the style of Chen et al. (2020), Fig. 5: stem + one pool augmentation on the diagonal,
    stem + the pair below it. The matrix is symmetric (both augmentations are applied to both views), so only the lower
    triangle is drawn. Absolute values on seed 42 (every cell has seed 42): a single-seed screen."""
    p = "val" if split == "validation" else "test"
    pool = [("speckle", "Speckle"), ("contrast", "Contrast"), ("blur", "Blur"), ("motion", "Motion"), ("pixelate", "Pixelation"), ("jpeg", "JPEG")]
    ids = set(d["Policy id"].dropna())
    def pid_of(a, b):
        if a == b:
            return a
        for x in (f"{a}_{b}", f"{b}_{a}"):
            if x in ids:
                return x
    def val(pid, obj, col):
        x = d[(d["Policy id"] == pid) & (d.Objective == obj) & (d.Seed == 42)][col].dropna()
        return float(x.iloc[0]) if len(x) else np.nan
    k = len(pool)
    def matrix(obj, col):
        M = np.full((k, k), np.nan)
        for i in range(k):
            for j in range(i + 1):                                 # lower triangle and diagonal only
                pid = pid_of(pool[i][0], pool[j][0])
                if pid:
                    M[i, j] = val(pid, obj, col)
        return M
    panels = [("NCL", matrix("ncl", f"{p}_h"), "backbone probe h (%)", "acc"),
              ("SimCLR", matrix("simclr", f"{p}_h"), "backbone probe h (%)", "acc"),
              ("NCL", matrix("ncl", f"{p}_sc"), "SC (%)", "sc")]
    acc = np.concatenate([panels[0][1].ravel(), panels[1][1].ravel()])
    lims = {"acc": (np.nanmin(acc), np.nanmax(acc)), "sc": (np.nanmin(panels[2][1]), np.nanmax(panels[2][1]))}
    stem = {k_: (val("stem", "ncl", f"{p}_h"), val("stem", "simclr", f"{p}_h"), val("stem", "ncl", f"{p}_sc")) for k_ in [0]}[0]
    labels = [n for _, n in pool]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    for ax, (meth, M, what, key) in zip(axes, panels):
        lo, hi = lims[key]
        im = ax.imshow(M, cmap="magma", vmin=lo, vmax=hi, aspect="equal")
        ax.grid(False)                                             # no grid lines across the cells
        ax.tick_params(length=0)
        for i in range(k):
            for j in range(k):
                if np.isfinite(M[i, j]):
                    frac = (M[i, j] - lo) / (hi - lo) if hi > lo else 0.5
                    ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=10,
                            color="black" if frac > 0.6 else "white")
        ax.set_xticks(range(k), labels, rotation=40, ha="right")
        ax.set_yticks(range(k), labels)
        ax.set_title(f"{meth}: {what}", fontsize=12)
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    fig.suptitle(f"Stem + one (diagonal) or two (off-diagonal) pool augmentations, OCT ResNet-50, {split}, seed 42", y=0.99, fontsize=13)
    fig.text(0.5, -0.07, f"Stem = crop 0.08 + brightness 0.8 + flip; stem alone: NCL {stem[0]:.1f}, SimCLR {stem[1]:.1f} probe h, NCL SC {stem[2]:.1f}%. "
             "\nDiagonal = stem + that augmentation; below it = stem + both. Both are applied to both views, so the matrix is symmetric "
             "and only one triangle is shown.\nOne seed per cell (seed 42): a screen for which combinations to follow up, "
             "not replicated evidence. Layout after Chen et al. (2020), Fig. 5.", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, out, f"fig6_composition_{p}")


FAMILIES = [  # crop-only design: (label, family, colour, levels in order of increasing strength)
    ("brightness", "photometric", "#08306B", ["croponly_bright_b020", "croponly_bright_b040", "croponly_bright", "croponly_bright_b095"]),
    ("contrast", "photometric", "#4A90D9", ["croponly_contrast", "croponly_contrast_f050", "croponly_contrast_f030", "croponly_contrast_f010"]),
    ("blur", "degradation", "#E08214", ["croponly_blur_sig060", "croponly_blur", "croponly_blur_sig200"]),
    ("motion", "degradation", "#FDB863", ["croponly_motion_r3", "croponly_motion", "croponly_motion_r8"]),
    ("pixelation", "degradation", "#B35806", ["croponly_pixelate_f090", "croponly_pixelate", "croponly_pixelate_f060"]),
    ("JPEG", "degradation", "#7F3B08", ["croponly_jpeg_q90", "croponly_jpeg", "croponly_jpeg_q40"]),
    ("speckle", "imaging noise", "#1B7837", ["croponly_speckle_s005", "croponly_speckle", "croponly_speckle_s020"]),
    ("crop area", "geometric", "#C51B7D", ["croponly_crop020", "croponly_crop050"]),
    ("flip", "geometric", "#DE77AE", ["croponly_flip"]),
]


def fig10(d, out):
    """NCL small multiples: one column per augmentation (grouped by family), change vs crop only on matched seeds against
    the actual levels; top row backbone probe h, bottom row SC; solid = test, dashed = validation; hollow = one seed."""
    base = d[(d["Policy id"] == "croponly") & (d.Objective == "ncl")].set_index("Seed")
    def delta(pid, col):
        x = d[(d["Policy id"] == pid) & (d.Objective == "ncl")].set_index("Seed")
        sh = [s_ for s_ in x.index if s_ in base.index and pd.notna(x.loc[s_, col]) and pd.notna(base.loc[s_, col])]
        return (float(np.mean([x.loc[s_, col] - base.loc[s_, col] for s_ in sh])), len(sh)) if sh else (np.nan, 0)
    ticks = {"brightness": ["0.2", "0.4", "0.8", "0.95"], "contrast": ["0.8", "0.5", "0.3", "0.1"], "blur": ["0.6", "1.2", "2.0"],
             "motion": ["r3", "r5", "r8"], "pixelation": ["0.9", "0.8", "0.6"], "JPEG": ["q90", "q70", "q40"],
             "speckle": ["0.05", "0.1", "0.2"], "crop area": ["0.2", "0.5"], "flip": ["on"]}
    fams = [f_ for f_ in FAMILIES if any(np.isfinite(delta(p_, "val_h")[0]) for p_ in f_[3])]
    rows = [("h", "Δ backbone probe h (pp)"), ("sc", "Δ NCL SC (pp)")]
    fig, axes = plt.subplots(2, len(fams), figsize=(2.15 * len(fams) + 1, 6.4), sharey="row")
    for j, (name, fam, colr, pids) in enumerate(fams):
        for i, (key, ylab) in enumerate(rows):
            ax = axes[i, j]
            ax.axhline(0, color="#888888", lw=0.8, zorder=1)
            for p, ls, alpha, lab in (("val", "--", 0.55, "validation"), ("test", "-", 1.0, "test")):
                xs, ys, ns = [0], [0.0], [99]
                for k, p_ in enumerate(pids, start=1):
                    v, n = delta(p_, f"{p}_{key}")
                    if np.isfinite(v):
                        xs.append(k); ys.append(v); ns.append(n)
                ax.plot(xs, ys, ls, color=colr, lw=2, alpha=alpha, zorder=3, label=lab)
                for x_, y_, n_ in zip(xs[1:], ys[1:], ns[1:]):
                    ax.scatter([x_], [y_], s=26, zorder=4, edgecolors=colr, linewidths=1.4, alpha=alpha,
                               facecolors="white" if n_ == 1 else colr)
            ax.set_xticks(range(len(pids) + 1), ["crop\nonly"] + ticks[name], fontsize=9)
            ax.set_xlim(-0.4, len(pids) + 0.4)
            if i == 0:
                ax.set_title(f"{name}\n", color=colr, fontsize=12, fontweight="bold")
                ax.text(0.5, 1.02, fam, transform=ax.transAxes, ha="center", fontsize=9, color="#666666")
            if j == 0:
                ax.set_ylabel(ylab)
    axes[0, 0].legend(loc="upper left", fontsize=9, handlelength=2.2)
    fig.text(0.5, -0.02, "NCL, OCT ResNet-50, crop-only design: change vs crop only on matched seeds. Solid = test, dashed = validation, "
             "hollow = one seed. NCL crop 0.5 diverged.", ha="center", fontsize=9, color="#666666")
    fig.tight_layout()
    save(fig, out, "fig10_family_trends")


def fig11(d, out, shortcut_csv, stat="histogram"):
    """The view shortcut (view_shortcut.py: how often a view's partner is found from intensity statistics alone, chance
    0.39%) against NCL's raw SC and the backbone probe for both methods, validation, one point per policy (seed means)."""
    if not Path(shortcut_csv).exists():
        print(f"{shortcut_csv} not found: fig11 skipped (run study_tools/view_shortcut.py on the server and copy it)")
        return
    sh = pd.read_csv(shortcut_csv)
    sh = sh[sh.statistic == stat].set_index("policy_id")["top1_pct"]
    fam = {"croponly": ("crop only", "#222222", "o"), "croponly_flip": ("flip", "#DE77AE", "D"), "croponly_crop020": ("crop area", "#C51B7D", "v"),
           "stem": ("old stem", "#6A3D9A", "P"), "contrast": ("old stem", "#6A3D9A", "P")}
    lab = {"croponly": "crop only", "croponly_flip": "flip", "croponly_crop020": "area 0.2", "stem": "stem", "contrast": "stem + c0.8"}
    for pid in sh.index:
        if pid.startswith("croponly_bright"):
            fam[pid] = ("brightness", "#08306B", "o"); lab[pid] = "b" + {"croponly_bright": "0.8"}.get(pid, pid.split("_b0")[-1].join(["0.", ""]) if "_b0" in pid else pid)
        elif pid.startswith("croponly_contrast"):
            fam[pid] = ("contrast", "#4A90D9", "s"); lab[pid] = "c" + {"croponly_contrast": "0.8"}.get(pid, "0." + pid[-2:-1] if pid.endswith("0") else pid)
        elif pid.startswith("croponly_blur"):
            fam[pid] = ("blur", "#E08214", "^"); lab[pid] = {"croponly_blur": "σ1.2", "croponly_blur_sig060": "σ0.6", "croponly_blur_sig200": "σ2.0"}[pid]
    pols = [p_ for p_ in sh.index if p_ in fam and p_ != "croponly_crop050"]
    panels = [("ncl", "val_sc", "NCL SC (%)"), ("ncl", "val_h", "NCL backbone probe h (%)"), ("simclr", "val_h", "SimCLR backbone probe h (%)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (obj, col, ylab) in zip(axes, panels):
        xs, ys = [], []
        for p_ in pols:
            v = d[(d["Policy id"] == p_) & (d.Objective == obj)][col].dropna()
            if not len(v):
                continue
            name, colr, mk = fam[p_]
            ax.errorbar(sh[p_], v.mean(), yerr=v.std(ddof=1) if len(v) > 1 else 0, fmt=mk, color=colr, ms=7, capsize=0, lw=1)
            ax.annotate(lab[p_], (sh[p_], v.mean()), xytext=(5, 3), textcoords="offset points", fontsize=8)
            xs.append(np.log10(sh[p_])); ys.append(v.mean())
        r = np.corrcoef(xs, ys)[0, 1]
        ax.set_xscale("log")
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
        ax.xaxis.set_major_locator(FixedLocator([2, 3, 5, 10, 20, 30]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("view shortcut: partner found from the intensity histogram (%)")
        ax.set_ylabel(ylab)
        ax.set_title(f"{ylab.split(' (')[0]}\nr = {r:+.2f} (log shortcut, {len(xs)} policies)", fontsize=11)
    fig.text(0.5, -0.03, "Validation, OCT ResNet-50; means over seeds (bars: seed SD). Shortcut: share of views whose partner is the "
             "nearest neighbour among 256 by intensity statistics alone (chance 0.39%), measured on the augmentation pipeline "
             "before any training. Crop area 0.5 is off the scale.", ha="center", fontsize=9, color="#666666", wrap=True)
    fig.tight_layout()
    save(fig, out, "fig11_view_shortcut")


def load_all(path):
    """Both datasets (ResNet-50), for the figures that include PathMNIST."""
    v = pd.read_excel(path, sheet_name="8 Runs (val)")
    t = pd.read_excel(path, sheet_name="8 Runs (test)")
    v = v[(v.Backbone == "ResNet-50") & (v.Collapsed == "no")]
    t = t[(t.Backbone == "ResNet-50") & (t["Probe backbone"] >= 60)]
    keys = ["Dataset", "Policy id", "Objective", "Seed"]
    v = v[keys + ["Probe backbone", "Probe projector"]].rename(columns={"Probe backbone": "val_h", "Probe projector": "val_z"})
    t = t[keys + ["Probe backbone", "Probe projector"]].rename(columns={"Probe backbone": "test_h", "Probe projector": "test_z"})
    return v.merge(t, on=keys, how="outer")


def matched(D, ds, base, pid, obj, col, seeds=SEEDS):
    """Per-seed change pid − base on the seeds both have (matched seeds)."""
    a = D[(D.Dataset == ds) & (D["Policy id"] == base) & (D.Objective == obj)].set_index("Seed")[col]
    b = D[(D.Dataset == ds) & (D["Policy id"] == pid) & (D.Objective == obj)].set_index("Seed")[col]
    sh = [x for x in seeds if x in a.index and x in b.index and pd.notna(a[x]) and pd.notna(b[x])]
    return (b[sh] - a[sh]).values


def bars(ax, groups, series, ylab, zero=False):
    """Grouped bars (mean) with every seed as a dot. series: list of (label, colour, hatch, values per group)."""
    w = 0.8 / len(series)
    for k_, (lab, colr, hatch, vals) in enumerate(series):
        xs = np.arange(len(groups)) + (k_ - (len(series) - 1) / 2) * w
        means = [np.mean(v) if len(v) else np.nan for v in vals]
        ax.bar(xs, means, w * 0.92, color=colr if not hatch else "white", edgecolor=colr, hatch=hatch, lw=1.2, label=lab, zorder=2)
        for x, v in zip(xs, vals):
            ax.scatter(np.full(len(v), x), v, s=12, color="black", zorder=3)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylabel(ylab)
    if zero:
        ax.axhline(0, color="grey", lw=0.8)
    ax.grid(axis="y", alpha=0.3)


def fig7(D, out):
    """Story sheet 3: backbone probe h with one augmentation added to crop only, NCL and SimCLR, validation and test,
    seeds 42-44. Dot plot (not bars, because the axis does not start at zero): small dots = seeds, in the method's
    colour; the larger marker with a line = the mean."""
    pols = [("crop only", "croponly"), ("brightness 0.8", "croponly_bright"), ("contrast 0.8", "croponly_contrast"),
            ("blur", "croponly_blur"), ("flip", "croponly_flip"), ("crop 0.2", "croponly_crop020")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (col, split) in zip(axes, (("val_h", "validation"), ("test_h", "test"))):
        for obj, lab, colr, mk, off in (("ncl", "NCL", NCL, "o", -0.15), ("simclr", "SimCLR", SIM, "s", 0.15)):
            for k, (_, pid) in enumerate(pols):
                v = D[(D.Dataset == "OCT") & (D["Policy id"] == pid) & (D.Objective == obj) & D.Seed.isin(SEEDS)][col].dropna().values
                ax.scatter(np.full(len(v), k + off), v, s=16, color=colr, alpha=0.45, marker=mk, zorder=2)
                ax.plot([k + off - 0.1, k + off + 0.1], [v.mean()] * 2, color=colr, lw=2.5, zorder=3)
                ax.scatter([k + off], [v.mean()], s=60, color=colr, marker=mk, zorder=4, label=lab if k == 0 else None)
        ax.axvline(0.5, color="#BBBBBB", lw=0.8, ls=":")              # crop only (reference) | one augmentation added
        ax.set_xticks(range(len(pols)), [g for g, _ in pols], rotation=20)
        ax.set_ylabel(f"backbone probe h, {split} (%)")
        ax.set_title(split)
    axes[0].legend(loc="lower left")
    fig.suptitle("Accuracy with one augmentation added to crop only (OCT, ResNet-50, seeds 42-44)", y=1.02)
    fig.text(0.5, -0.04, "Small dots = individual seeds; large marker and line = mean over the three seeds. "
             "Crop only, left of the dotted line, is the reference.", ha="center", fontsize=9, color="#666666")
    fig.tight_layout()
    save(fig, out, "fig7_family_accuracy")


def fig8(D, out):
    """Story sheet 4: change in backbone (h) and projector (z) probe on test, per comparison, NCL and SimCLR."""
    comps = [("brightness\n0.8", "OCT", "croponly", "croponly_bright"), ("contrast\n0.8", "OCT", "croponly", "croponly_contrast"),
             ("blur", "OCT", "croponly", "croponly_blur"), ("stem +\ncontrast", "OCT", "stem", "contrast"),
             ("PathMNIST\nstain", "PathMNIST", "stem", "stain")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, (obj, name, colr) in zip(axes, (("ncl", "NCL", NCL), ("simclr", "SimCLR", SIM))):
        series = [("backbone h", colr, None, [matched(D, ds, b, p_, obj, "test_h") for _, ds, b, p_ in comps]),
                  ("projector z", colr, "///", [matched(D, ds, b, p_, obj, "test_z") for _, ds, b, p_ in comps])]
        bars(ax, [g for g, *_ in comps], series, "change on test (pp)", zero=True)
        ax.set_title(name)
        ax.legend(frameon=False)
    fig.suptitle("Backbone and projector gains on test (matched seeds; bars = mean, dots = seeds). References: crop only for the first three, "
                 "the stem for the last two; PathMNIST NCL has 2 seeds", y=1.03, fontsize=10)
    fig.tight_layout()
    save(fig, out, "fig8_backbone_projector_bars")


def fig9(D, out):
    """Story sheet 6: the same augmentation's change on validation and on test, NCL and SimCLR."""
    comps = [("PathMNIST: stain\nvs stem", "PathMNIST", "stem", "stain"), ("OCT: brightness 0.8\nvs crop only", "OCT", "croponly", "croponly_bright"),
             ("OCT: contrast 0.8\nvs crop only", "OCT", "croponly", "croponly_contrast")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (obj, name, colr) in zip(axes, (("ncl", "NCL", NCL), ("simclr", "SimCLR", SIM))):
        series = [("validation", colr, None, [matched(D, ds, b, p_, obj, "val_h") for _, ds, b, p_ in comps]),
                  ("test", colr, "///", [matched(D, ds, b, p_, obj, "test_h") for _, ds, b, p_ in comps])]
        bars(ax, [g for g, *_ in comps], series, "change in backbone probe h (pp)", zero=True)
        ax.set_title(name)
        ax.legend(frameon=False)
    fig.suptitle("The same augmentation measured on validation and on test (matched seeds; bars = mean, dots = seeds)", y=1.02)
    fig.tight_layout()
    save(fig, out, "fig9_validation_vs_test")


# ---------------------------------------------------------------- method gap and layer figures (dumbbell style)
_TEAL, _PURPLE, _CORAL, _GOLD, _BLUE = "#007F88", "#7756AF", "#DC6957", "#C99639", "#4781C4"
_NAVY, _INK, _GRAY, _LINE = "#263D59", "#273241", "#687889", "#DBE2E9"
_STYLE = {"font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 11.5, "axes.labelsize": 9.5, "text.color": _INK,
          "axes.labelcolor": _INK, "xtick.color": _GRAY, "ytick.color": _INK, "axes.spines.top": False, "axes.spines.right": False,
          "axes.spines.left": False, "axes.edgecolor": _LINE, "axes.grid": False, "grid.color": "#E8EDF1"}
LAYERS = [("Brightness 0.95", "OCT", "croponly_bright_b095", "croponly"), ("Crop minimum 0.20", "OCT", "croponly_crop020", "croponly"),
          ("Gaussian blur 1.2", "OCT", "croponly_blur", "croponly"), ("Stem + contrast", "OCT", "contrast", "stem"),
          ("Stem + motion", "OCT", "motion", "stem"), ("Stain · PathMNIST", "PathMNIST", "stain", "stem")]


def _gap(D, pid, col, ds="OCT"):
    """NCL − SimCLR on the same policy and seed, seeds 42-44 where both runs are valid."""
    x = D[(D.Dataset == ds) & (D["Policy id"] == pid) & D.Seed.isin(SEEDS)].pivot_table(index="Seed", columns="Objective", values=col)
    x = x.dropna() if {"ncl", "simclr"} <= set(x.columns) else x.iloc[0:0]
    return (x["ncl"] - x["simclr"]).values if len(x) else np.array([])


def _headline(fig, title, subtitle, top=0.972):
    fig.text(0.065, top, title, fontweight="bold", fontsize=15, color=_NAVY, ha="left", va="top")
    fig.text(0.065, top - 0.045, subtitle, fontsize=9, color=_GRAY, ha="left", va="top")


def _pair(D, pid, col, ds="OCT"):
    """Per-seed NCL and SimCLR values for one policy, seeds 42-44 where both runs are valid (columns ncl, simclr)."""
    x = D[(D.Dataset == ds) & (D["Policy id"] == pid) & D.Seed.isin(SEEDS)].pivot_table(index="Seed", columns="Objective", values=col)
    return x[["ncl", "simclr"]].dropna() if {"ncl", "simclr"} <= set(x.columns) else None


def fig12(D, out):
    """Backbone probe h per augmentation, NCL and SimCLR, validation and test panels (mean ± SD over seeds 42-44),
    with the mean per-seed difference NCL − SimCLR written in a column at the right of each panel."""
    rows = [("Crop only: crop area ≥ 0.08", "croponly", True), ("+ brightness 0.8", "croponly_bright", False),
            ("+ brightness 0.95", "croponly_bright_b095", False), ("+ contrast 0.8", "croponly_contrast", False),
            ("+ contrast 0.3", "croponly_contrast_f030", False), ("+ blur σ ≤ 1.2", "croponly_blur", False),
            ("+ flip", "croponly_flip", False), ("crop area ≥ 0.20 instead", "croponly_crop020", False),
            (None, None, None),
            ("Stem: crop, brightness 0.8, flip", "stem", True), ("+ contrast 0.8", "contrast", False)]
    ys = np.arange(len(rows))[::-1].astype(float); off = 0.16
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 7.0), sharey=True)
    fig.subplots_adjust(left=0.20, right=0.985, top=0.83, bottom=0.20, wspace=0.08)
    for ax, col, title in zip(axes, ("val_h", "test_h"), ("Validation", "Test")):
        vals = []
        for (label, pid, ref), y in zip(rows, ys):
            if pid is None:
                ax.axhline(y, color="#BBBBBB", lw=0.8, ls=(0, (4, 3)))
                continue
            t = _pair(D, pid, col)
            if t is None or not len(t):
                continue
            for obj, c, mk, dy in (("ncl", NCL, "o", off), ("simclr", SIM, "s", -off)):
                m, sd = t[obj].mean(), (t[obj].std(ddof=1) if len(t) > 1 else 0.0)
                ax.errorbar(m, y + dy, xerr=sd, fmt=mk, color=c, ms=7, elinewidth=1.4, capsize=3, zorder=3,
                            markeredgecolor="white", markeredgewidth=0.6)
                vals += [m - sd, m + sd]
            dd = (t["ncl"] - t["simclr"]); g = dd.mean(); gs = dd.std(ddof=1) if len(dd) > 1 else 0.0
            ax.text(0.89, y + 0.13, f"{g:+.1f} ± {gs:.1f}".replace("-", "−"), transform=ax.get_yaxis_transform(), ha="center",
                    va="center", fontsize=10, color=NCL if g > 0 else SIM, fontweight="bold")
            ax.text(0.89, y - 0.25, f"NCL ahead {int((dd > 0).sum())}/{len(dd)}", transform=ax.get_yaxis_transform(), ha="center",
                    va="center", fontsize=7.5, color="#666666")
        lo = np.floor(min(vals)) - 0.5; dmax = max(vals) + 0.4; hi = lo + (dmax - lo) / 0.77   # data stays left of 0.77
        ax.set_xlim(lo, hi); ax.set_ylim(-0.6, len(rows) - 0.1)
        ax.add_patch(plt.Rectangle((0.78, 0), 0.22, 1, transform=ax.transAxes, color="#F2F2F2", zorder=0, lw=0))
        ax.text(0.89, len(rows) - 0.55, "NCL − SimCLR\nmean ± SD", transform=ax.get_yaxis_transform(), ha="center", va="bottom",
                fontsize=8.5, color="#444444")
        tk = np.arange(np.ceil(lo / 2) * 2, dmax + 0.01, 2)
        ax.set_xticks(tk, [f"{t:g}" for t in tk])
        ax.grid(axis="y", visible=False); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
        ax.set_title(title, fontsize=12.5, pad=8)
        ax.set_xlabel("Backbone probe, balanced accuracy (%)")
    axes[0].set_yticks([y for (l, p, r), y in zip(rows, ys) if p], [l for l, p, r in rows if p])
    for t, (l, p, r) in zip(axes[0].get_yticklabels(), [x for x in rows if x[1]]):
        if r:
            t.set_fontweight("bold")
    axes[1].legend(handles=[Line2D([0], [0], marker="o", color=NCL, ls="", ms=7, label="NCL"),
                            Line2D([0], [0], marker="s", color=SIM, ls="", ms=7, label="SimCLR")],
                   loc="lower center", bbox_to_anchor=(-0.05, 1.07), ncol=2, fontsize=11)
    fig.suptitle("Backbone probe accuracy per augmentation, NCL and SimCLR (OCT, ResNet-50)", fontsize=13.5, y=0.975)
    fig.text(0.5, 0.02, "Marker = mean over seeds 42, 43, 44; bar = ±1 SD across those seeds (not a confidence interval). Right column = mean ± SD of "
             "the per-seed differences NCL − SimCLR\n(same augmentation, same seed; blue when NCL is more accurate) and the number of seeds on "
             "which NCL is more accurate. Rows under a bold reference add one augmentation to it.\nEvery run uses its final (epoch-99) "
             "checkpoint. Policies and settings are chosen on validation; test is scored only for the runs in a test plan written in advance. "
             "Runs where a method collapsed are left out.", ha="center", va="bottom", fontsize=9, color="#444444")
    save(fig, out, "fig12_method_accuracy")


def fig13(D, out):
    """NCL: change in backbone (h) and projector (z) test probe vs the matched reference (crop only or stem)."""
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(11.6, 6.1)); fig.subplots_adjust(left=0.265, right=0.96, top=0.80, bottom=0.18)
        ypos = np.array([6, 5, 4, 3, 2, 0.45])
        for j, (label, ds, pid, ref) in enumerate(LAYERS):
            dh = matched(D, ds, ref, pid, "ncl", "test_h"); dz = matched(D, ds, ref, pid, "ncl", "test_z")
            if not len(dh) or not len(dz):
                continue
            y = ypos[j]; h, z = dh.mean(), dz.mean()
            ax.plot([h, z], [y, y], lw=2.7, color="#D1DBE2", zorder=1)
            ax.scatter(h, y, color=_BLUE, s=79, marker="o", edgecolors="white", zorder=3)
            ax.scatter(z, y, color=_CORAL, s=82, marker="s", edgecolors="white", zorder=3)
            if len(dh) > 1:
                ax.errorbar(h, y, xerr=dh.std(ddof=1), fmt="none", ecolor=_BLUE, lw=1.25, alpha=0.58, capsize=2)
                ax.errorbar(z, y, xerr=dz.std(ddof=1), fmt="none", ecolor=_CORAL, lw=1.25, alpha=0.58, capsize=2)
            ax.text(15.3, y, f"n={len(dh)}", color=_GRAY, fontsize=8, va="center")
        ax.axhline(1.25, color=_LINE, lw=0.95, ls="--")
        ax.text(-8.5, 1.04, "Different imaging modality ↓", color=_GRAY, fontsize=8.5, va="top")
        ax.axvline(0, color="#91A0AB", lw=1.2)
        ax.set_yticks(ypos, [x[0] for x in LAYERS]); ax.set_ylim(-0.1, 6.7); ax.set_xlim(-9, 18)
        ax.grid(axis="x", linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(length=0, pad=7)
        ax.set_xlabel("Change in test probe balanced accuracy relative to matched reference (pp)")
        ax.legend(handles=[Line2D([0], [0], marker="o", color=_BLUE, ls="", label="Backbone h", markersize=9),
                           Line2D([0], [0], marker="s", color=_CORAL, ls="", label="Projector z", markersize=9)],
                  loc="upper right", frameon=False, ncol=2)
        _headline(fig, "The same augmentation can affect h and z differently",
                  "NCL · changes versus matched crop-only or stem reference; last condition is PathMNIST rather than OCT")
        fig.text(0.5, 0.035, "Points = mean across matching successful seeds (42–44); whiskers = ±1 seed SD; "
                 "PathMNIST stain has n=2 because the NCL stem's seed 42 collapsed.", fontsize=8, color=_GRAY, ha="center", va="bottom")
        save(fig, out, "fig13_backbone_vs_projector")


def fig14(d, out):
    """Augmentation strength (crop-only base): backbone probe h for NCL and SimCLR on validation (A-B) and on test (C-D),
    and NCL's raw SC (E-F; validation solid, test dashed), for the brightness and contrast sweeps. Seeds 42-44."""
    d = d.drop_duplicates(["Policy id", "Objective", "Seed"])
    sweeps = [("Brightness strength", [("crop\nonly", "croponly"), ("0.2", "croponly_bright_b020"), ("0.4", "croponly_bright_b040"),
                                       ("0.8", "croponly_bright"), ("0.95", "croponly_bright_b095")]),
              ("Contrast factor (lower = stronger)", [("crop\nonly", "croponly"), ("0.8", "croponly_contrast"), ("0.5", "croponly_contrast_f050"),
                                                      ("0.3", "croponly_contrast_f030"), ("0.1", "croponly_contrast_f010")])]
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 11), sharex="col")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.91, bottom=0.12, hspace=0.22, wspace=0.08)
    sel = lambda pid, obj: d[(d["Policy id"] == pid) & (d.Objective == obj) & d.Seed.isin(SEEDS)]
    for j, (name, levels) in enumerate(sweeps):
        xs = np.arange(len(levels))
        for i, col in enumerate(("val_h", "test_h")):
            ax = axes[i, j]
            for obj, colr, mk, off in (("ncl", NCL, "o", -0.06), ("simclr", SIM, "s", 0.06)):
                means = []
                for k, (_, pid) in enumerate(levels):
                    v = sel(pid, obj)[col].dropna()
                    ax.scatter(np.full(len(v), k) + off, v, s=12, color=colr, alpha=0.4, marker=mk, zorder=2, linewidths=0)
                    means.append(v.mean())
                ax.plot(xs, means, "-", marker=mk, color=colr, ms=7, lw=2, zorder=3)
        ax = axes[2, j]
        for col, ls, mk, off in (("val_sc", "-", "o", -0.08), ("test_sc", "--", "s", 0.08)):
            means = []
            for k, (_, pid) in enumerate(levels):
                v = sel(pid, "ncl")[col].dropna()
                ax.scatter(np.full(len(v), k) + off, v, s=16, marker=mk, facecolors="none", edgecolors=NCL, alpha=0.55,
                           linewidths=0.9, zorder=2)
                means.append(v.mean())
            ax.plot(xs, means, ls, marker=mk, color=NCL, ms=7, lw=2, zorder=3)
        ax.set_xticks(xs, [l for l, _ in levels]); ax.set_xlabel(name)
        axes[0, j].set_title(name.split(" (")[0], fontsize=13)
    for i in range(3):
        lo = min(a.get_ylim()[0] for a in axes[i]); hi = max(a.get_ylim()[1] for a in axes[i])
        for a in axes[i]:
            a.set_ylim(lo, hi); a.grid(alpha=0.3)
        axes[i, 1].tick_params(labelleft=False)
    axes[0, 0].set_ylabel("Backbone probe h,\nvalidation (%)"); axes[1, 0].set_ylabel("Backbone probe h,\ntest (%)")
    axes[2, 0].set_ylabel("NCL SC (%)")
    for k, ax in enumerate(axes.flat):
        ax.text(-0.01, 1.03, "ABCDEF"[k], transform=ax.transAxes, fontsize=13, fontweight="bold", ha="right", va="bottom")
    axes[0, 1].legend(handles=[Line2D([0], [0], marker="o", color=NCL, lw=2, label="NCL"),
                               Line2D([0], [0], marker="s", color=SIM, lw=2, label="SimCLR")], loc="lower right", fontsize=10)
    axes[2, 0].legend(handles=[Line2D([0], [0], ls="-", marker="o", color=NCL, label="validation"),
                               Line2D([0], [0], ls="--", marker="s", color=NCL, label="test"),
                               Line2D([0], [0], ls="", marker="o", mfc="none", mec=NCL, alpha=0.55, label="one seed")],
                      loc="upper left", fontsize=9.5, ncol=3, columnspacing=1.0, handlelength=2.2)
    fig.suptitle("Augmentation strength changes accuracy and NCL's specialisation (OCT, ResNet-50, crop-only base)", fontsize=13.5, y=0.975)
    fig.text(0.5, 0.015, "Line = mean over seeds 42, 43, 44; dots = the seeds. The first point of each panel is crop only. Each row has its own "
             "y-axis: test accuracies are 8-10 pp below validation,\nso compare within a row. A-D: backbone probe accuracy, NCL and SimCLR. "
             "E-F: NCL's raw SC; SimCLR's raw SC is 25% at every level and is not drawn.", ha="center", va="bottom", fontsize=9, color="#555555")
    save(fig, out, "fig14_strength_accuracy_sc")


def _place_labels(fig, ax, items, bold=(), extra=()):
    """Greedy label placement: for each (text, (x, y)), try offsets around the point and keep the one that overlaps
    the fewest points and earlier labels while staying inside the axes."""
    from matplotlib.transforms import Bbox
    fig.canvas.draw(); r = fig.canvas.get_renderer(); axbb = ax.get_window_extent(r)
    taken, heavy = [], []
    for x, y in [xy for _, xy in items]:
        px, py = ax.transData.transform((x, y)); taken.append(Bbox.from_bounds(px - 7, py - 7, 14, 14))
    for x, y in extra:                                             # e.g. the reference marker: never cover it
        px, py = ax.transData.transform((x, y)); heavy.append(Bbox.from_bounds(px - 9, py - 9, 18, 18))
    cands = [(7, 4, "left", "bottom"), (7, -4, "left", "top"), (-7, 4, "right", "bottom"), (-7, -4, "right", "top"),
             (0, 9, "center", "bottom"), (0, -9, "center", "top"), (18, 16, "left", "bottom"), (-18, -16, "right", "top"),
             (18, -16, "left", "top"), (-18, 16, "right", "bottom"), (30, 0, "left", "center"), (-30, 0, "right", "center"),
             (0, 24, "center", "bottom"), (0, -24, "center", "top"), (36, 22, "left", "bottom"), (-36, -22, "right", "top"),
             (36, -22, "left", "top"), (-36, 22, "right", "bottom")]
    cx, cy = np.mean([xy[0] for _, xy in items]), np.mean([xy[1] for _, xy in items])
    order = sorted(items, key=lambda it: ax.transData.transform(it[1]).tolist() and
                   np.hypot(*(ax.transData.transform(it[1]) - ax.transData.transform((cx, cy)))))
    for name, (x, y) in order:                                     # crowded points (near the centre) are placed first
        best, best_ov = None, 1e9
        for dx, dy, ha, va in cands:
            far = abs(dx) > 10 or abs(dy) > 10
            t = ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=8.5,
                            fontweight="bold" if name in bold else "normal", color="#222222", zorder=6,
                            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6, shrinkA=0, shrinkB=4) if far else None)
            bb = t.get_window_extent(r)
            inside = bb.x0 >= axbb.x0 and bb.x1 <= axbb.x1 and bb.y0 >= axbb.y0 and bb.y1 <= axbb.y1
            ov = sum(bb.overlaps(o) for o in taken) + 5 * sum(bb.overlaps(o) for o in heavy) + (0 if inside else 10) + (0.5 if far else 0)
            if ov < best_ov:
                if best is not None:
                    best.remove()
                best, best_ov = t, ov
            else:
                t.remove()
            if ov == 0:
                break
        taken.append(best.get_window_extent(r))


def fig15(d, out):
    """NCL: change in raw SC against change in backbone (h) and projector (z) probe accuracy, each policy minus its
    matched reference (crop only or stem) on seeds 42-44; only policies with all three seeds. Bars = ±1 SD of the
    per-seed changes. Rows = validation, test; columns = h, z; all panels share the y-axis."""
    d = d.drop_duplicates(["Policy id", "Objective", "Seed"]); n = d[d.Objective == "ncl"]
    def delta(pid, ref, col):
        a = n[(n["Policy id"] == ref) & n.Seed.isin(SEEDS)].set_index("Seed")[col]
        b = n[(n["Policy id"] == pid) & n.Seed.isin(SEEDS)].set_index("Seed")[col]
        sh = [x for x in SEEDS if x in a.index and x in b.index and pd.notna(a[x]) and pd.notna(b[x])]
        return (b[sh] - a[sh]).values
    singles = [("brightness 0.8", "croponly_bright", "croponly"), ("contrast 0.8", "croponly_contrast", "croponly"),
               ("blur σ ≤ 1.2", "croponly_blur", "croponly"), ("flip", "croponly_flip", "croponly"),
               ("crop area 0.2", "croponly_crop020", "croponly"),
               ("stem + contrast 0.8", "contrast", "stem"), ("stem + motion", "motion", "stem")]
    paths = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5), sharey=True)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.87, bottom=0.13, hspace=0.27, wspace=0.06)
    ys_all = []
    for i, (split, sname) in enumerate((("val", "validation"), ("test", "test"))):
        xs_all = []
        for j, (layer, lname) in enumerate((("h", "Backbone h"), ("z", "Projector z"))):
            ax = axes[i, j]; items = []
            ax.axhline(0, color="#777777", lw=0.9, zorder=1); ax.axvline(0, color="#777777", lw=0.9, zorder=1)
            def one(pid, ref, mk):
                dx, dy = delta(pid, ref, f"{split}_sc"), delta(pid, ref, f"{split}_{layer}")
                if len(dx) < 3 or len(dy) < 3:
                    return None
                x, y = dx.mean(), dy.mean(); xs_all.extend([x - dx.std(ddof=1), x + dx.std(ddof=1)])
                ax.errorbar(x, y, xerr=dx.std(ddof=1), yerr=dy.std(ddof=1), fmt="none", ecolor=NCL, alpha=0.35, lw=1.0, zorder=2)
                ax.scatter([x], [y], marker=mk, s=60, color=NCL, edgecolors="white", linewidths=0.6, zorder=4)
                ys_all.extend([y - dy.std(ddof=1), y + dy.std(ddof=1)])
                return (x, y)
            for name, ids in paths:
                xy = [(0.0, 0.0)] + [p for p in (one(q, "croponly", "o") for q in ids) if p is not None]
                ax.plot(*zip(*xy), "-", color=NCL, lw=1.0, alpha=0.55, zorder=3)
                items.append((name, xy[-1]))
            for name, pid, ref in singles:
                xy = one(pid, ref, "o" if ref == "croponly" else "s")
                if xy is not None:
                    items.append((name, xy))
            ax.scatter([0], [0], marker="P", s=70, color="black", zorder=5)
            ax.grid(alpha=0.25)
            for (tx, ty, ha, va, lab) in ((0.98, 0.98, "right", "top", "SC ↑, accuracy ↑"), (0.98, 0.02, "right", "bottom", "SC ↑, accuracy ↓"),
                                          (0.02, 0.98, "left", "top", "SC ↓, accuracy ↑"), (0.02, 0.02, "left", "bottom", "SC ↓, accuracy ↓")):
                ax.text(tx, ty, lab, transform=ax.transAxes, ha=ha, va=va, fontsize=8.5, color="#888888", style="italic")
            if i == 0:
                ax.set_title(lname, fontsize=13)
            ax.set_xlabel(f"Change in NCL SC, {sname} (pp)")
            ax._items = items
        lo, hi = min(min(xs_all), 0) - 1.2, max(xs_all) + 1.2
        for ax in axes[i]:
            ax.set_xlim(lo, hi)
        axes[i, 0].set_ylabel(f"Change in probe accuracy,\n{sname} (pp)")
    for a in axes.flat:
        a.set_ylim(min(ys_all) - 1.5, max(ys_all) + 2.0)
    for ax in axes.flat:
        _place_labels(fig, ax, ax._items, bold=("brightness 0.8", "crop area 0.2"), extra=[(0.0, 0.0)])
    for k, ax in enumerate(axes.flat):
        ax.text(-0.01, 1.03, "ABCD"[k], transform=ax.transAxes, fontsize=13, fontweight="bold", ha="right", va="bottom")
    fig.legend(handles=[Line2D([0], [0], marker="o", color=NCL, ls="", ms=8, label="change vs crop only"),
                        Line2D([0], [0], marker="s", color=NCL, ls="", ms=8, label="change vs stem"),
                        Line2D([0], [0], marker="P", color="black", ls="", ms=9, label="reference (no change)")],
               loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=3, fontsize=10, frameon=False)
    fig.suptitle("Changes in NCL's specialisation against changes in accuracy (OCT, ResNet-50)", fontsize=13.5, y=0.975)
    fig.text(0.5, 0.012, "Each point: NCL with one augmentation added to its reference (crop only for circles, stem for squares), minus the "
             "reference,\non the same seeds (42, 43, 44); bars = ±1 SD of the per-seed changes. The other brightness strengths (0.2, 0.4, 0.95) "
             "sit with brightness 0.8 (fig 14).\nAll panels share the y-axis, so backbone and projector changes can be compared directly.",
             ha="center", va="bottom", fontsize=9, color="#555555")
    save(fig, out, "fig15_sc_change_vs_accuracy_change")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workbook", default="study/summary/TC_results_5nov_e.xlsx")
    ap.add_argument("--out", default="study/summary/figs_story")
    ap.add_argument("--test-ms", default="study/summary/test_ms_octmnist_5nov.csv", help="test MS per run (eval_test_ms.py)")
    ap.add_argument("--shortcut", default="study/derived/view_shortcut.csv", help="view_shortcut.py output")
    a = ap.parse_args()
    d = load(a.workbook, a.test_ms)
    out = Path(a.out)
    fig1(d, out); fig2(d, out); fig3(d, out); fig4(d, out)
    fig14(d, out); fig15(d, out)
    fig5(d, out, "validation"); fig5(d, out, "test")
    fig6(d, out, "validation"); fig6(d, out, "test")
    fig10(d, out)
    fig11(d, out, a.shortcut)
    D = load_all(a.workbook)
    fig7(D, out); fig8(D, out); fig9(D, out)
    fig12(D, out); fig13(D, out)


if __name__ == "__main__":
    main()
