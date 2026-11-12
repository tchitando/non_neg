#!/usr/bin/env python3
"""Crop-only base (Part 7) and strength scan (Part 8): one table per objective, the checks planned on 29 Oct, and the
strength-curve figure.

    python study_tools/croponly_table.py

Run the two scorers first (validation split only); runs that are not scored yet are shown as missing:
    CUDA_VISIBLE_DEVICES=0 python study_tools/eval_run.py --only 'croponly'
    CUDA_VISIBLE_DEVICES=0 python study_tools/eval_spec.py --only 'croponly'

Reads the latest row of every OCT ResNet-50 run of the policies below from study/results_master.csv (eval_run.py) and
study/results_spec.csv (eval_spec.py, version s2). Nothing is trained or scored here.

    probe h, probe z   linear probe on the backbone feature h / the projector output z, balanced accuracy (%)
    kNN-20             k-NN (k = 20) on h, balanced accuracy (%)
    SC > shuffle       NCL's semantic consistency above its shuffled-label value, in SC % points
                       (0.75 x (SC* - SC*_shuffled)). Not shown for SimCLR: its units fire on every image, so SC is 25%
                       by construction; projector purity is the SimCLR measure.
    purity z, purity h top-100 class purity of the projector units / backbone units, chance-corrected (0 = chance)
Collapsed runs (backbone probe below 60) are listed and left out of every mean.

Noise: for each objective and measure, the pooled seed SD over the policies that have 2 or more seeds. A difference
from the crop-only base is marked * when it is larger than 2 standard errors (SE from the pooled SD and the seed counts
of the two policies). For one-seed levels this says whether a level is worth more seeds, not that the effect is confirmed.

Outputs: study/derived/croponly_runs.csv (one row per run: every seed), study/derived/croponly_summary.csv (one row per
policy and objective: seed count, per-seed values, mean, SD, difference from the base, SE), and
study/summary/fig_croponly_curves.png / .pdf (strength curves).
"""
import argparse, math, re, sys
from pathlib import Path

import numpy as np
import pandas as pd

# (policy id, family, level, label)
POLICIES = [
    ("stem", "old stem", None, "old stem (crop + brightness + flip)"),
    ("contrast", "old stem", None, "old stem + contrast 0.8"),
    ("croponly", "base", None, "crop only (base)"),
    ("croponly_bright", "brightness", 0.8, "+ brightness jitter 0.8 (stem's)"),
    ("croponly_flip", "stem part", None, "+ flip (stem's)"),
    ("croponly_bright_b020", "brightness", 0.2, "+ brightness jitter 0.2"),
    ("croponly_bright_b040", "brightness", 0.4, "+ brightness jitter 0.4"),
    ("croponly_bright_b095", "brightness", 0.95, "+ brightness jitter 0.95"),
    ("croponly_contrast", "contrast", 0.8, "+ contrast 0.8"),
    ("croponly_contrast_f050", "contrast", 0.5, "+ contrast 0.5"),
    ("croponly_contrast_f030", "contrast", 0.3, "+ contrast 0.3"),
    ("croponly_contrast_f010", "contrast", 0.1, "+ contrast 0.1"),
    ("croponly_blur_sig060", "blur", 0.6, "+ blur sigma <= 0.6"),
    ("croponly_blur", "blur", 1.2, "+ blur sigma <= 1.2"),
    ("croponly_blur_sig200", "blur", 2.0, "+ blur sigma <= 2.0"),
    ("croponly_crop020", "crop", 0.2, "crop area >= 0.2"),
    ("croponly_crop050", "crop", 0.5, "crop area >= 0.5"),
]
BASE = "croponly"
BASE_LEVEL = {"contrast": 1.0, "brightness": 0.0, "blur": 0.0, "crop": 0.08}
XLABEL = {"contrast": "contrast factor\n(lower = stronger)", "brightness": "brightness jitter s\n(factor in [1-s, 1+s])",
          "blur": "blur sigma max", "crop": "crop min area\n(higher = weaker crop)"}
LABEL = {p: lab for p, _, _, lab in POLICIES}
METRICS = [  # (column, label)
    ("probe_bb", "probe h"), ("probe_proj", "probe z"), ("knn20", "kNN-20"),
    ("sc_excess", "SC > shuffle"), ("purity_z", "purity z"), ("purity_h", "purity h"),
]
OBJ = {"ncl": "NCL", "simclr": "SimCLR"}
COLLAPSE = 60.0


def latest(path, version_col=None, version=None):
    if not Path(path).exists():
        return pd.DataFrame()
    d = pd.read_csv(path)
    if version_col and version_col in d:
        d = d[d[version_col].astype(str) == version]
    return d.drop_duplicates("run_id", keep="last").set_index("run_id")


def collect(master, spec):
    rows = []
    for pid, fam, lev, _ in POLICIES:
        pat = re.compile(rf"^(ncl|simclr)-resnet50-octmnist-{re.escape(pid)}(-s(\d+))?$")
        for rid in master.index:
            m = pat.match(rid)
            if not m:
                continue
            r, s = master.loc[rid], spec.loc[rid] if rid in spec.index else None
            seed = int(m.group(3)) if m.group(3) else 42
            g = lambda col, src=r: float(src[col]) if src is not None and col in src and pd.notna(src[col]) else np.nan
            obj = m.group(1)
            rows.append(dict(
                run_id=rid, objective=obj, policy_id=pid, family=fam, level=lev, seed=seed,
                probe_bb=g("probe_bb_bal_acc"), probe_proj=g("probe_proj_bal_acc"), knn20=g("knn20_bal_acc"),
                sc_raw=g("sc"), effdim_z=g("effdim_proj"), alive_z=g("alive_proj"),
                sc_excess=(0.75 * g("sc_star_excess", s)) if obj == "ncl" else np.nan,
                purity_z=g("purity_proj_star", s), purity_h=g("purity_bb_star", s), ms_star=g("ms_star", s),
                spec_scored=s is not None))
    runs = pd.DataFrame(rows)
    if len(runs):
        runs["collapsed"] = runs.probe_bb < COLLAPSE
    return runs


def pooled_sd(runs):
    out = {}
    ok = runs[~runs.collapsed]
    for obj in OBJ:
        for col, _ in METRICS:
            num = den = 0.0
            for _, g in ok[ok.objective == obj].groupby("policy_id"):
                v = g[col].dropna()
                if len(v) >= 2:
                    num += (len(v) - 1) * v.var(ddof=1); den += len(v) - 1
            out[(obj, col)] = math.sqrt(num / den) if den else np.nan
    return out


def summarise(runs, sdp, seeds):
    rows = []
    ok = runs[~runs.collapsed]
    for obj in OBJ:
        base = ok[(ok.objective == obj) & (ok.policy_id == BASE)]
        for pid, fam, lev, lab in POLICIES:
            g = ok[(ok.objective == obj) & (ok.policy_id == pid)]
            row = dict(objective=obj, policy_id=pid, family=fam, level=lev, label=lab,
                       n_collapsed=int(runs[(runs.objective == obj) & (runs.policy_id == pid)].collapsed.sum()))
            for col, _ in METRICS:
                v = g.set_index("seed")[col].dropna()
                b = base[col].dropna()
                row[f"{col}_n"] = len(v)
                for sd in seeds:
                    row[f"{col}_s{sd}"] = v.get(sd, np.nan)
                row[f"{col}_mean"] = v.mean() if len(v) else np.nan
                row[f"{col}_sd"] = v.std(ddof=1) if len(v) >= 2 else np.nan
                if pid != BASE and len(v) and len(b):
                    d = v.mean() - b.mean()
                    se = sdp[(obj, col)] * math.sqrt(1 / len(v) + 1 / len(b))
                    row[f"{col}_diff"], row[f"{col}_se"] = d, se
                else:
                    row[f"{col}_diff"] = row[f"{col}_se"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def cell(r, col, prec=2, diff=True):
    n, m, sd = r[f"{col}_n"], r[f"{col}_mean"], r[f"{col}_sd"]
    if not n:
        return f"{'-':>13s}" + (f"{'':>9s}" if diff else "")
    s = f"{m:.{prec}f}" + (f" ±{sd:.{prec}f}" if n >= 2 else "      " if prec == 2 else "       ")
    out = f"{s:>13s}"
    if diff:
        d, se = r[f"{col}_diff"], r[f"{col}_se"]
        out += f"{'':>9s}" if d != d else f"{d:+8.{prec}f}{'*' if se == se and abs(d) > 2 * se else ' '}"
    return out


def print_tables(summ, sdp, seeds):
    for obj, name in OBJ.items():
        s = summ[summ.objective == obj]
        print(f"\n{name}, validation. Mean ± seed SD; the column after each value is the difference from crop only "
              f"(* = more than 2 SE of seed noise).")
        print(f"  {'policy':36s}{'n':>2s}{'probe h':>13s}{'':9s}{'probe z':>13s}{'':9s}{'kNN-20':>13s}{'':9s}")
        for _, r in s.iterrows():
            print(f"  {r.label:36s}{r.probe_bb_n:>2d}{cell(r, 'probe_bb')}{cell(r, 'probe_proj')}{cell(r, 'knn20')}"
                  + (f"   ({r.n_collapsed} collapsed, left out)" if r.n_collapsed else ""))
        cols = ([("sc_excess", 2)] if obj == "ncl" else []) + [("purity_z", 3), ("purity_h", 3)]
        head = "".join(f"{dict(METRICS)[c]:>13s}{'':9s}" for c, _ in cols)
        print(f"  {'':36s}{'n':>2s}{head}")
        for _, r in s.iterrows():
            print(f"  {r.label:36s}{r.purity_z_n:>2d}" + "".join(cell(r, c, p) for c, p in cols))
        print("  pooled seed SD: " + ", ".join(f"{dict(METRICS)[c]} {sdp[(obj, c)]:.3g}" for c, _ in METRICS
                                                 if sdp[(obj, c)] == sdp[(obj, c)]))
    print("\nEvery seed (validation balanced accuracy, %): backbone probe | projector probe")
    sh = "".join(f"{'s' + str(sd):>7s}" for sd in seeds)
    print(f"  {'policy':36s}{'obj':>7s}{sh}{'mean':>8s}{'SD':>6s}   |{sh}{'mean':>8s}{'SD':>6s}")
    f = lambda x, w=7: f"{x:{w}.2f}" if x == x else f"{'-':>{w}s}"
    for _, r in summ.iterrows():
        if not r.probe_bb_n:
            continue
        print(f"  {r.label:36s}{OBJ[r.objective]:>7s}" +
              "".join(f(r[f"probe_bb_s{sd}"]) for sd in seeds) + f(r.probe_bb_mean, 8) + f(r.probe_bb_sd, 6) + "   |" +
              "".join(f(r[f"probe_proj_s{sd}"]) for sd in seeds) + f(r.probe_proj_mean, 8) + f(r.probe_proj_sd, 6))


def verdict(r, col, sign):
    d, se = r[f"{col}_diff"], r[f"{col}_se"]
    if d != d:
        return "not scored yet"
    beyond = se == se and abs(d) > 2 * se
    nn = f"{int(r[f'{col}_n'])} seed{'s' if r[f'{col}_n'] > 1 else ''}"
    if beyond:
        return f"{'holds' if np.sign(d) == sign else 'does not hold: opposite direction'} ({d:+.2f}, beyond noise, {nn})"
    return f"not shown: {'same' if np.sign(d) == sign else 'opposite'} direction but within noise ({d:+.2f}, {nn})"


def checks(summ):
    get = lambda obj, pid: summ[(summ.objective == obj) & (summ.policy_id == pid)].iloc[0]
    print("\nChecks planned on 29 Oct, before these results (differences are from crop only):")
    for obj in OBJ:
        b, st = get(obj, BASE), get(obj, "stem")
        if b.probe_bb_n:
            d = st.probe_bb_mean - b.probe_bb_mean
            print(f"  1. {OBJ[obj]:6s} crop only trains normally: probe h {b.probe_bb_mean:.2f} (n {b.probe_bb_n}, "
                  f"{b.n_collapsed} collapsed); the old stem is {d:+.2f} above it")
    c8 = get("ncl", "croponly_contrast")
    print(f"  2a. NCL  crop only specialises less than the old stem (old stem minus crop only): "
          f"{verdict(get('ncl', 'stem'), 'sc_excess', +1)}")
    print(f"  2b. NCL  contrast 0.8 raises SC on crop only: {verdict(c8, 'sc_excess', +1)}")
    print(f"  2c. SimCLR contrast 0.8 raises projector purity on crop only: {verdict(get('simclr', 'croponly_contrast'), 'purity_z', +1)}")
    for fam, what in (("contrast", "contrast"), ("brightness", "brightness"), ("blur", "blur"), ("crop", "crop")):
        for obj in OBJ:
            s = summ[(summ.objective == obj) & ((summ.family == fam) | (summ.policy_id == BASE))].copy()
            s["x"] = [BASE_LEVEL[fam] if p == BASE else lv for p, lv in zip(s.policy_id, s.level)]
            s = s[s.probe_bb_n > 0].sort_values("x")
            if len(s) < 2:
                continue
            spec = "sc_excess" if obj == "ncl" else "purity_z"
            parts = []
            for col in ("probe_bb", spec):
                v = s.dropna(subset=[f"{col}_mean"])
                if len(v) < 2:
                    continue
                best = v.loc[v[f"{col}_mean"].idxmax()]
                inner = 0 < list(v.x).index(best.x) < len(v) - 1
                d, se = best[f"{col}_diff"], best[f"{col}_se"]
                vs = "the base itself" if best.policy_id == BASE else \
                    f"{d:+.2f}{'*' if se == se and abs(d) > 2 * se else ''} vs crop only"
                parts.append(f"best {dict(METRICS)[col]} at {best.x:g} ({best[f'{col}_mean']:.2f}, {vs}, "
                             f"{'bracketed by worse levels' if inner else 'at the end of the tested range'})")
            print(f"  3. {OBJ[obj]:6s} {what}: " + "; ".join(parts))
    print(f"  4a. NCL  brightness jitter raises SC on crop only: {verdict(get('ncl', 'croponly_bright'), 'sc_excess', +1)}")
    print(f"  4b. NCL  flip raises SC on crop only (expected: no): {verdict(get('ncl', 'croponly_flip'), 'sc_excess', +1)}")
    for obj in OBJ:
        print(f"  4c. {OBJ[obj]:6s} backbone probe, brightness: {verdict(get(obj, 'croponly_bright'), 'probe_bb', +1)}; "
              f"flip: {verdict(get(obj, 'croponly_flip'), 'probe_bb', +1)}")
    print("  Expected: contrast raises SC/purity with a best level inside the range; blur leaves SC unchanged and costs\n"
          "  accuracy; a weaker crop (larger area) raises SC and costs accuracy.")


def figure(summ, runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, GREY, INK, MUTED = "#0072B2", "#D55E00", "#8A8984", "#2C2C2A", "#5F5E5A"
    col = {"ncl": BLUE, "simclr": ORANGE}
    mk = {"ncl": "o", "simclr": "s"}
    rows = [("sc_excess", "NCL SC above\nshuffle (% points)", ["ncl"]),
            ("purity_z", "projector purity\n(chance-corrected)", ["ncl", "simclr"]),
            ("probe_bb", "backbone probe\n(bal. acc., %)", ["ncl", "simclr"]),
            ("probe_proj", "projector probe\n(bal. acc., %)", ["ncl", "simclr"])]
    fams = ["contrast", "brightness", "blur", "crop"]
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(len(rows), len(fams), figsize=(9.2, 8.6), sharex="col", sharey="row")
    ok = runs[~runs.collapsed]
    for i, (metric, ylab, objs) in enumerate(rows):
        for j, fam in enumerate(fams):
            ax = axes[i, j]
            ax.grid(color="#E4E2DA", lw=0.5, zorder=0)
            for obj in objs:
                s = summ[(summ.objective == obj) & ((summ.family == fam) | (summ.policy_id == BASE))].copy()
                s["x"] = [BASE_LEVEL[fam] if p == BASE else lv for p, lv in zip(s.policy_id, s.level)]
                s = s.dropna(subset=[f"{metric}_mean"]).sort_values("x")
                if not len(s):
                    continue
                ax.plot(s.x, s[f"{metric}_mean"], color=col[obj], lw=2, zorder=2)
                ax.scatter(s.x, s[f"{metric}_mean"], color=col[obj], marker=mk[obj], s=36, edgecolor="white",
                           linewidth=1, zorder=4, label=OBJ[obj])
                r = ok[(ok.objective == obj) & (ok.policy_id.isin(s.policy_id))]
                xs = [BASE_LEVEL[fam] if p == BASE else dict(zip(s.policy_id, s.x))[p] for p in r.policy_id]
                ax.scatter(xs, r[metric], color=col[obj], marker=mk[obj], s=10, alpha=0.45, linewidth=0, zorder=3)
            if fam == "contrast":
                ax.set_xlim(1.05, 0.05)
            if i == len(rows) - 1:
                ax.set_xlabel(XLABEL[fam])
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 0:
                ax.set_title({"contrast": "Contrast", "brightness": "Brightness", "blur": "Blur", "crop": "Crop strength"}[fam],
                             fontsize=9, color=INK)
    h, l = axes[1, 0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.005, "Crop-only base (crop min area 0.08, no other augmentation) at the left end of each curve. "
             "Large markers: mean over seeds; small: single seeds.", ha="center", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200)
    print(f"\nfigure: {out.with_suffix('.png')} (and .pdf)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default="study/results_master.csv")
    ap.add_argument("--spec", default="study/results_spec.csv")
    ap.add_argument("--out-dir", default="study/derived")
    ap.add_argument("--fig", default="study/summary/fig_croponly_curves")
    a = ap.parse_args()
    master, spec = latest(a.master), latest(a.spec, "spec_version", "s2")
    if master.empty:
        sys.exit(f"{a.master} not found or empty")
    runs = collect(master, spec)
    if runs.empty:
        sys.exit("no crop-only or stem runs in the results table yet")
    want = {f"{o}-resnet50-octmnist-{p}" for o in OBJ for p, *_ in POLICIES if p.startswith("croponly")}
    missing = sorted(w for w in want if w not in set(runs.run_id))
    if missing:
        print("not scored yet (seed 42): " + ", ".join(m.replace("-resnet50-octmnist-", " ") for m in missing))
    nospec = runs[~runs.spec_scored].run_id.tolist()
    if nospec:
        print(f"{len(nospec)} run(s) scored by eval_run.py but not yet by eval_spec.py (no SC or purity): "
              + ", ".join(nospec[:6]) + (", ..." if len(nospec) > 6 else ""))
    col = runs[runs.collapsed]
    if len(col):
        print("collapsed (backbone probe < 60), left out of means: " + ", ".join(f"{r} ({p:.1f})" for r, p in zip(col.run_id, col.probe_bb)))
    sdp = pooled_sd(runs)
    seeds = sorted(int(x) for x in runs.seed.unique())
    summ = summarise(runs, sdp, seeds)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    runs.sort_values(["objective", "policy_id", "seed"]).to_csv(out / "croponly_runs.csv", index=False)
    summ.to_csv(out / "croponly_summary.csv", index=False)
    print_tables(summ, sdp, seeds)
    checks(summ)
    print(f"\nwrote {out / 'croponly_runs.csv'} and {out / 'croponly_summary.csv'}")
    try:
        figure(summ, runs, Path(a.fig))
    except Exception as e:
        print(f"figure not drawn: {e!r}")


if __name__ == "__main__":
    main()
