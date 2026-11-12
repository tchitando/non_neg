#!/usr/bin/env python3
"""Build every results table (LaTeX), figure (PDF) and quoted number (JSON) for the OCTMNIST study.

    python study_tools/make_paper_results.py --master study/results_master.csv --spec study/results_spec.csv --out paper

Inputs: the append-only results files (validation split only; the latest row per run_id is used; the audit
metrics come from spec version s2, the training-matched readout).
Outputs: <out>/tables/*.tex, <out>/figs/*.pdf, <out>/results_numbers.json. Nothing here is typed by hand.

Statistics
  * A policy's value is the mean over its seeds; effects are differences from the stem mean (3 seeds).
  * Seed noise: the pooled within-policy standard deviation s of the policies that have three seeds
    (stem, contrast, motion, natural-image reference), per objective and metric, with 8 degrees of freedom.
  * The standard error of a difference from the stem is s*sqrt(1/n + 1/3) (n seeds of the policy). An effect is
    marked when it exceeds t(0.975, 8) standard errors: for three-seed policies this is the 95% confidence
    interval excluding zero, for one-seed policies it is the same rule with the pooled noise.
  * Synergy of a pair: D(AB) - D(A) - D(B), standard error s*sqrt(1/n_AB + 1/n_A + 1/n_B + 1/3).
    Contribution of an operation to the full stack: D(all six) - D(all six without X), standard error s*sqrt(2).
"""
import argparse, json, math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

POOL = ["speckle", "contrast", "blur", "motion", "pixelate", "jpeg"]
LABEL = {"speckle": "Speckle", "contrast": "Contrast", "blur": "Gaussian blur", "motion": "Motion blur",
         "pixelate": "Pixelation", "jpeg": "JPEG", "cifar": "Natural-image ref.", "all6": "All six", "stem": "Stem"}
SHORT = {"speckle": "speckle", "contrast": "contrast", "blur": "blur", "motion": "motion", "pixelate": "pixelate", "jpeg": "JPEG"}
OBJS = [("ncl", "NCL"), ("simclr", "SimCLR")]
# metric key -> (results column, table header, decimals)
MET = {
    "probe": ("probe_bb_bal_acc", "Probe", 2),
    "probe_z": ("probe_proj_bal_acc", "Probe ($z$)", 2),
    "knn": ("knn20_bal_acc", "$k$-NN", 2),
    "sc": ("sc_star", "SC$^*$", 1),
    "scsh": ("sc_star_shuf", "SC$^*_{\\text{shuf}}$", 1),
    "pz": ("purity_proj_star", "Pur.$^*$ ($z$)", 3),
    "ph": ("purity_bb_star", "Pur.$^*$ ($h$)", 3),
    "ms": ("ms_star", "MS$^*$", 3),
    "edh": ("effdim_bb", "ED ($h$)", 1),
    "edz": ("effdim_proj", "ED ($z$)", 1),
}
STRENGTH = ["contrast_f060", "contrast_f090", "stem_crop020", "stem_crop050", "contrast_crop020", "contrast_crop050",
            "speckle_contrast_crop020"]   # the last is the ResNet-50 equivalent of the ResNet-18 "A6" policy
C_NCL, C_SIM = "#2a78d6", "#e8a33d"

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "pdf.fonttype": 42})


def five(x):
    return "_".join(p for p in POOL if p != x)


def pairs():
    return [(POOL[i], POOL[j]) for i in range(6) for j in range(i + 1, 6)]


def tex(x, d, sign=True, bold=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    if float(s) == 0:
        s = s.replace("-", "+") if sign else s.lstrip("-")
    if s.startswith("-"):
        s = "$-$" + s[1:]
    return f"\\textbf{{{s}}}" if bold else s


def load(master, spec):
    m = pd.read_csv(master).drop_duplicates("run_id", keep="last")
    s = pd.read_csv(spec)
    s = s[s.spec_version == "s2"].drop_duplicates("run_id", keep="last")
    keep = ["run_id", "sc_star", "sc_star_shuf", "ms", "ms_floor", "ms_star", "purity_proj_star", "purity_bb_star"]
    m = m.rename(columns={"sc": "sc_raw_master"})
    df = m.drop(columns=[c for c in keep[1:] if c in m.columns]).merge(s[keep], on="run_id", how="left")
    df = df.rename(columns={"ms": "ms_raw", "sc": "sc_raw"}).rename(columns={v[0]: k for k, v in MET.items()})
    df["seed"] = df["seed"].astype(int)
    ctrl = s[s.policy_id == "random-init"].rename(columns={"ms": "ms_raw"}).rename(columns={v[0]: k for k, v in MET.items()})
    return df, ctrl, s


class Stats:
    def __init__(self, df):
        self.df = df
        self.pol = df.groupby(["objective", "policy_id"])
        self.mean = self.pol[list(MET)].mean()
        self.n = self.pol.size()
        self.sd = {}
        self.dof = {}
        for o, _ in OBJS:
            sub = df[df.objective == o]
            multi = sub.groupby("policy_id").filter(lambda g: len(g) >= 3)
            g = multi.groupby("policy_id")
            for k in MET:
                ss = sum(((x[k] - x[k].mean()) ** 2).sum() for _, x in g)
                dof = sum(len(x) - 1 for _, x in g)
                self.sd[(o, k)] = math.sqrt(ss / dof) if dof else float("nan")
                self.dof[o] = dof
            self.multi_policies = sorted(multi.policy_id.unique())
        self.tcrit = {o: stats.t.ppf(0.975, self.dof[o]) for o, _ in OBJS}

    def has(self, o, p):
        return (o, p) in self.mean.index

    def v(self, o, p, k):
        return float(self.mean.loc[(o, p), k]) if self.has(o, p) else float("nan")

    def nn(self, o, p):
        return int(self.n.loc[(o, p)]) if self.has(o, p) else 0

    def d(self, o, p, k, ref="stem"):
        return self.v(o, p, k) - self.v(o, ref, k)

    def se(self, o, k, *ns):
        return self.sd[(o, k)] * math.sqrt(sum(1.0 / n for n in ns))

    def hw(self, o, p, k, ref="stem"):
        """95% half-width of the difference p - ref."""
        return self.tcrit[o] * self.se(o, k, self.nn(o, p), self.nn(o, ref))

    def flag(self, o, p, k, ref="stem"):
        dv = self.d(o, p, k, ref)
        return (not math.isnan(dv)) and abs(dv) > self.hw(o, p, k, ref)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    return float(stats.spearmanr(x[ok], y[ok]).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default="study/results_master.csv")
    ap.add_argument("--spec", default="study/results_spec.csv")
    ap.add_argument("--out", default="paper")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    df, ctrl, spec = load(a.master, a.spec)
    S = Stats(df)
    N = {"n_runs": int(len(df)), "n_runs_by_objective": df.objective.value_counts().to_dict(),
         "pending_audit": sorted(df[df.ms.isna()].run_id.tolist()), "three_seed_policies": S.multi_policies,
         "dof": S.dof, "tcrit": {o: round(S.tcrit[o], 3) for o, _ in OBJS},
         "ms_floor": round(float(spec.ms_floor.dropna().iloc[0]), 4)}
    screen_pols = ["stem"] + POOL + ["cifar"] + [f"{x}_{y}" for x, y in pairs()] + [five(x) for x in POOL] + ["all6"]
    N["n_screen_policies"] = len(screen_pols)

    # ------------------------------------------------------------------ reference values
    cm = {o: ctrl[ctrl.objective == o][list(k for k in ("sc", "scsh", "pz", "ph", "ms"))].mean() for o, _ in OBJS}
    cols = ["probe", "knn", "sc", "scsh", "pz", "ph", "ms"]
    L = ["\\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}", "\\toprule",
         " & " + " & ".join(MET[k][1] for k in cols) + " \\\\", "\\midrule"]
    for o, nm in OBJS:
        L.append(f"{nm} stem (3 seeds) & " + " & ".join(tex(S.v(o, 'stem', k), MET[k][2], sign=False) for k in cols) + " \\\\")
        L.append(f"\\quad seed s.d.\\ (pooled) & " + " & ".join(tex(S.sd[(o, k)], MET[k][2], sign=False) for k in cols) + " \\\\")
        L.append(f"\\quad untrained (3 inits) & -- & -- & " + " & ".join(tex(float(cm[o][k]), MET[k][2], sign=False) for k in ("sc", "scsh", "pz", "ph", "ms")) + " \\\\")
        if o == "ncl":
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    (out / "tables" / "tab_reference.tex").write_text("\n".join(L) + "\n")
    for o, _ in OBJS:
        N[f"stem_{o}"] = {k: round(S.v(o, "stem", k), 4) for k in MET}
        N[f"stem_{o}_seed_sd"] = {k: round(float(S.df[(S.df.objective == o) & (S.df.policy_id == 'stem')][k].std()), 4) for k in MET}
        N[f"pooled_sd_{o}"] = {k: round(S.sd[(o, k)], 4) for k in MET}
        N[f"untrained_{o}"] = {k: round(float(v), 4) for k, v in cm[o].items()}

    # ------------------------------------------------------------------ confirmed policies (3 seeds)
    conf = [p for p in ("contrast", "motion", "cifar") if all(S.nn(o, p) >= 3 for o, _ in OBJS)]
    ncl_c, sim_c = ["probe", "knn", "sc", "pz", "ph", "ms"], ["probe", "knn", "pz", "ph"]
    L = ["\\begin{tabular}{@{}l " + "c" * len(ncl_c) + " " + "c" * len(sim_c) + "@{}}", "\\toprule",
         f"& \\multicolumn{{{len(ncl_c)}}}{{c}}{{NCL}} & \\multicolumn{{{len(sim_c)}}}{{c}}{{SimCLR}} \\\\",
         f"\\cmidrule(lr){{2-{1 + len(ncl_c)}}}\\cmidrule(l){{{2 + len(ncl_c)}-{1 + len(ncl_c) + len(sim_c)}}}",
         "Policy & " + " & ".join(MET[k][1] for k in ncl_c) + " & " + " & ".join(MET[k][1] for k in sim_c) + " \\\\", "\\midrule"]
    N["confirmed"] = {}
    for p in conf:
        cells = []
        for o, ks in (("ncl", ncl_c), ("simclr", sim_c)):
            for k in ks:
                dv, h = S.d(o, p, k), S.hw(o, p, k)
                cells.append(f"{tex(dv, MET[k][2], bold=abs(dv) > h)}\\,{{\\scriptsize$\\pm${h:.{MET[k][2]}f}}}")
                N["confirmed"][f"{o}_{p}_{k}"] = {"delta": round(dv, 4), "hw95": round(h, 4), "lo": round(dv - h, 4),
                                                  "hi": round(dv + h, 4), "value": round(S.v(o, p, k), 4),
                                                  "seeds": [round(float(x), 4) for x in S.df[(S.df.objective == o) & (S.df.policy_id == p)].sort_values("seed")[k]]}
        L.append(f"{LABEL[p]} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (out / "tables" / "tab_confirm.tex").write_text("\n".join(L) + "\n")

    # ------------------------------------------------------------------ screen tables: singles, stacks
    ncl_s, sim_s = ["probe", "knn", "sc", "pz", "ms"], ["probe", "knn", "pz"]

    def head():
        return ["\\begin{tabular}{@{}l " + "c" * len(ncl_s) + " " + "c" * len(sim_s) + "@{}}", "\\toprule",
                f"& \\multicolumn{{{len(ncl_s)}}}{{c}}{{NCL}} & \\multicolumn{{{len(sim_s)}}}{{c}}{{SimCLR}} \\\\",
                f"\\cmidrule(lr){{2-{1 + len(ncl_s)}}}\\cmidrule(l){{{2 + len(ncl_s)}-{1 + len(ncl_s) + len(sim_s)}}}",
                "Policy & " + " & ".join(MET[k][1] for k in ncl_s) + " & " + " & ".join(MET[k][1] for k in sim_s) + " \\\\", "\\midrule"]

    def row(p, label):
        c = [tex(S.d(o, p, k), MET[k][2], bold=S.flag(o, p, k)) for o, ks in (("ncl", ncl_s), ("simclr", sim_s)) for k in ks]
        dag = "$^\\dagger$" if S.nn("ncl", p) >= 3 else ""
        return f"{label}{dag} & " + " & ".join(c) + " \\\\"

    L = head() + [row(p, LABEL[p]) for p in POOL] + ["\\midrule", row("cifar", LABEL["cifar"]), "\\bottomrule", "\\end{tabular}"]
    (out / "tables" / "tab_singles.tex").write_text("\n".join(L) + "\n")
    L = head() + [row(five(x), f"All six $-$ {SHORT[x]}") for x in POOL] + [row("all6", "All six"), "\\midrule",
         f"\\multicolumn{{{1 + len(ncl_s) + len(sim_s)}}}{{@{{}}l}}{{\\emph{{Contribution to the full stack}}}} \\\\"]
    contrib = {}
    for x in POOL:
        c = []
        for o, ks in (("ncl", ncl_s), ("simclr", sim_s)):
            for k in ks:
                v = S.v(o, "all6", k) - S.v(o, five(x), k)
                h = S.tcrit[o] * S.se(o, k, S.nn(o, "all6"), S.nn(o, five(x)))
                c.append(tex(v, MET[k][2], bold=abs(v) > h))
                contrib[f"{o}_{x}_{k}"] = round(v, 4)
        L.append(f"{LABEL[x]} & " + " & ".join(c) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (out / "tables" / "tab_stacks.tex").write_text("\n".join(L) + "\n")
    N["contribution"] = contrib
    N["screen"] = {f"{o}_{p}": {k: {"delta": round(S.d(o, p, k), 4), "flag": bool(S.flag(o, p, k)), "n": S.nn(o, p)}
                                for k in MET} for o, _ in OBJS for p in screen_pols[1:] + STRENGTH if S.has(o, p)}

    # counts of flagged screen effects (all 29 non-stem screen policies)
    for o, _ in OBJS:
        P = screen_pols[1:]
        for k in ("probe", "knn", "sc", "pz", "ph", "ms"):
            up = sum(1 for p in P if S.flag(o, p, k) and S.d(o, p, k) > 0)
            dn = sum(1 for p in P if S.flag(o, p, k) and S.d(o, p, k) < 0)
            ds = [S.d(o, p, k) for p in P]
            N[f"count_{o}_{k}"] = {"up": up, "down": dn, "of": len(P), "min": round(min(ds), 4),
                                   "median": round(float(np.median(ds)), 4), "max": round(max(ds), 4)}
        hasc = [p for p in P if p == "cifar" or "contrast" in p.split("_") or p == "all6"]
        noc = [p for p in P if p not in hasc]
        for k in ("sc", "pz", "probe"):
            N[f"{o}_{k}_with_contrast"] = {"n": len(hasc), "flag_up": int(sum(S.flag(o, p, k) and S.d(o, p, k) > 0 for p in hasc)),
                                           "range": [round(min(S.d(o, p, k) for p in hasc), 3), round(max(S.d(o, p, k) for p in hasc), 3)],
                                           "median": round(float(np.median([S.d(o, p, k) for p in hasc])), 3)}
            N[f"{o}_{k}_without_contrast"] = {"n": len(noc), "flag_up": int(sum(S.flag(o, p, k) and S.d(o, p, k) > 0 for p in noc)),
                                              "flag_down": int(sum(S.flag(o, p, k) and S.d(o, p, k) < 0 for p in noc)),
                                              "range": [round(min(S.d(o, p, k) for p in noc), 3), round(max(S.d(o, p, k) for p in noc), 3)],
                                              "median": round(float(np.median([S.d(o, p, k) for p in noc])), 3)}
        N[f"probe_range_{o}"] = [round(min(S.v(o, p, "probe") for p in screen_pols), 3), round(max(S.v(o, p, "probe") for p in screen_pols), 3)]

    # ------------------------------------------------------------------ pairs: accuracy and synergy
    acc, syn, synf = {}, {}, {}
    for o, _ in OBJS:
        A = np.full((6, 6), np.nan); Sy = np.full((6, 6), np.nan); F = np.zeros((6, 6), bool)
        for i, x in enumerate(POOL):
            A[i, i] = S.v(o, x, "probe")
        for x, y in pairs():
            i, j = POOL.index(x), POOL.index(y); p = f"{x}_{y}"
            A[i, j] = A[j, i] = S.v(o, p, "probe")
            sv = S.d(o, p, "probe") - S.d(o, x, "probe") - S.d(o, y, "probe")
            se = S.se(o, "probe", S.nn(o, p), S.nn(o, x), S.nn(o, y), S.nn(o, "stem"))
            Sy[i, j] = Sy[j, i] = sv
            F[i, j] = F[j, i] = abs(sv) > S.tcrit[o] * se
        acc[o], syn[o], synf[o] = A, Sy, F
        iu = np.triu_indices(6, 1); sv = Sy[iu]
        N[f"synergy_{o}"] = {f"{POOL[i]}+{POOL[j]}": round(float(Sy[i, j]), 3) for i, j in zip(*iu)}
        N[f"synergy_{o}_summary"] = {"positive": int((sv > 0).sum()), "negative": int((sv < 0).sum()),
                                     "flag_pos": int(((sv > 0) & F[iu]).sum()), "flag_neg": int(((sv < 0) & F[iu]).sum()),
                                     "sign_test_p": round(float(stats.binomtest(int((sv > 0).sum()), 15).pvalue), 4),
                                     "median": round(float(np.median(sv)), 3)}
        best = max(pairs(), key=lambda q: S.v(o, f"{q[0]}_{q[1]}", "probe"))
        N[f"best_pair_{o}"] = {"pair": "+".join(best), "probe": round(S.v(o, f"{best[0]}_{best[1]}", "probe"), 3)}
        N[f"pairs_above_stem_{o}"] = int(sum(S.d(o, f"{x}_{y}", "probe") > 0 for x, y in pairs()))

    fig, axes = plt.subplots(2, 2, figsize=(6.85, 6.4), constrained_layout=True)
    vlo = min(np.nanmin(acc[o]) for o, _ in OBJS); vhi = max(np.nanmax(acc[o]) for o, _ in OBJS)
    lim = max(np.nanmax(np.abs(syn[o])) for o, _ in OBJS)
    ticks = [SHORT[p] for p in POOL]
    for c, (o, nm) in enumerate(OBJS):
        ax = axes[0, c]
        im = ax.imshow(acc[o], cmap="inferno", vmin=vlo, vmax=vhi)
        for i in range(6):
            for j in range(6):
                v = acc[o][i, j]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                        color="black" if (v - vlo) / (vhi - vlo) > 0.6 else "white", fontweight="bold" if i == j else "normal")
        ax.set_title(f"{nm}: probe balanced accuracy (stem {S.v(o, 'stem', 'probe'):.1f})")
        ax.set_xticks(range(6)); ax.set_xticklabels(ticks, rotation=45, ha="right"); ax.set_yticks(range(6)); ax.set_yticklabels(ticks)
        if c == 1:
            fig.colorbar(im, ax=axes[0, :], shrink=0.8, label="balanced accuracy (%)")
        ax = axes[1, c]
        im2 = ax.imshow(syn[o], cmap="RdBu", norm=TwoSlopeNorm(0, -lim, lim))
        for i in range(6):
            for j in range(6):
                if i != j:
                    v = syn[o][i, j]
                    ax.text(j, i, f"{v:+.1f}" + ("*" if synf[o][i, j] else ""), ha="center", va="center", fontsize=7.5,
                            color="black" if abs(v) < 0.6 * lim else "white", fontweight="bold" if synf[o][i, j] else "normal")
        ax.set_title(f"{nm}: synergy (pair $-$ sum of singles)")
        ax.set_xticks(range(6)); ax.set_xticklabels(ticks, rotation=45, ha="right"); ax.set_yticks(range(6)); ax.set_yticklabels(ticks)
        if c == 1:
            fig.colorbar(im2, ax=axes[1, :], shrink=0.8, label="synergy (points)")
    fig.savefig(out / "figs" / "fig_pairs.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ strength table and figure
    blocks = [("Contrast factor (stem + contrast)", [("contrast_f060", "0.6"), ("contrast", "0.8"), ("contrast_f090", "0.9"), ("stem", "1.0 (stem)")], "contrast"),
              ("Crop minimum area (stem)", [("stem", "0.08"), ("stem_crop020", "0.2"), ("stem_crop050", "0.5")], "stem"),
              ("Crop minimum area (stem + contrast)", [("contrast", "0.08"), ("contrast_crop020", "0.2"), ("contrast_crop050", "0.5")], "contrast"),
              ("Crop minimum area (stem + speckle + contrast)", [("speckle_contrast", "0.08"), ("speckle_contrast_crop020", "0.2 (A6)")], "speckle_contrast")]
    blocks = [(t, [(p, l) for p, l in rows if S.has("ncl", p)], ref) for t, rows, ref in blocks]
    blocks = [b for b in blocks if len(b[1]) > 1]
    st_n, st_s = ["probe", "knn", "sc", "pz", "ms"], ["probe", "knn", "pz"]
    L = ["\\begin{tabular}{@{}l c " + "c" * len(st_n) + " " + "c" * len(st_s) + "@{}}", "\\toprule",
         f"& & \\multicolumn{{{len(st_n)}}}{{c}}{{NCL}} & \\multicolumn{{{len(st_s)}}}{{c}}{{SimCLR}} \\\\",
         f"\\cmidrule(lr){{3-{2 + len(st_n)}}}\\cmidrule(l){{{3 + len(st_n)}-{2 + len(st_n) + len(st_s)}}}",
         "Setting & $n$ & " + " & ".join(MET[k][1] for k in st_n) + " & " + " & ".join(MET[k][1] for k in st_s) + " \\\\"]
    N["strength"] = {}
    for title, rows, ref in blocks:
        L += ["\\midrule", f"\\multicolumn{{{2 + len(st_n) + len(st_s)}}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\"]
        for p, lab in rows:
            cells = []
            for o, ks in (("ncl", st_n), ("simclr", st_s)):
                for k in ks:
                    v = S.v(o, p, k)
                    b = p != ref and S.flag(o, p, k, ref=ref)
                    cells.append(tex(v, MET[k][2], sign=False, bold=b))
                    N["strength"][f"{o}_{p}_{k}"] = {"value": round(v, 4) if not math.isnan(v) else None,
                                                     "delta_vs_ref": round(S.d(o, p, k, ref), 4) if not math.isnan(v) else None,
                                                     "ref": ref, "flag": bool(b)}
            n = S.nn("ncl", p)
            L.append(f"\\quad {lab} & {n} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (out / "tables" / "tab_strength.tex").write_text("\n".join(L) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.3), constrained_layout=True, sharex="col")
    crop_x = [0.08, 0.2, 0.5]
    lines = [("ncl", "stem", ["stem", "stem_crop020", "stem_crop050"], "NCL stem", C_NCL, "o", "-"),
             ("ncl", "contrast", ["contrast", "contrast_crop020", "contrast_crop050"], "NCL + contrast", C_NCL, "s", "--"),
             ("simclr", "stem", ["stem", "stem_crop020", "stem_crop050"], "SimCLR stem", C_SIM, "o", "-"),
             ("simclr", "contrast", ["contrast", "contrast_crop020", "contrast_crop050"], "SimCLR + contrast", C_SIM, "s", "--")]
    for o, _, ps, lab, col, mk, ls in lines:
        axes[0, 0].plot(crop_x, [S.v(o, p, "probe") for p in ps], marker=mk, ls=ls, color=col, label=lab, ms=4, lw=1.1, mfc="white" if ls == "--" else col)
        if o == "ncl":
            axes[1, 0].plot(crop_x, [S.v(o, p, "sc") for p in ps], marker=mk, ls=ls, color=col, ms=4, lw=1.1, mfc="white" if ls == "--" else col)
    fx = [0.6, 0.8, 0.9, 1.0]; fp = ["contrast_f060", "contrast", "contrast_f090", "stem"]
    for o, col, mk in (("ncl", C_NCL, "o"), ("simclr", C_SIM, "^")):
        axes[0, 1].plot(fx, [S.v(o, p, "probe") for p in fp], marker=mk, color=col, ms=4, lw=1.1)
        if o == "ncl":
            axes[1, 1].plot(fx, [S.v(o, p, "sc") for p in fp], marker=mk, color=col, ms=4, lw=1.1)
    axes[0, 0].set_ylabel("probe bal. acc. (%)"); axes[1, 0].set_ylabel("NCL SC$^*$")
    axes[1, 0].set_xlabel("crop minimum area"); axes[1, 1].set_xlabel("contrast factor (1.0 = none)")
    axes[1, 0].set_xticks(crop_x); axes[1, 1].set_xticks(fx)
    axes[0, 1].set_ylim(86.4, 88.8)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handlelength=1.8, columnspacing=0.8)
    fig.savefig(out / "figs" / "fig_strength.pdf", bbox_inches="tight"); plt.close(fig)

    # ------------------------------------------------------------------ NCL SC* vs probe
    P = screen_pols[1:]
    hasc = lambda p: p == "cifar" or "contrast" in p.split("_") or p == "all6"
    fig, ax = plt.subplots(figsize=(3.3, 2.8), constrained_layout=True)
    hx, hy = S.hw("ncl", "speckle", "probe"), S.hw("ncl", "speckle", "sc")
    ax.axvspan(-hx, hx, color="0.92", zorder=0); ax.axhspan(-hy, hy, color="0.92", zorder=0)
    for k, col, mk, lab in ((True, C_NCL, "o", "contains contrast"), (False, "#e34948", "s", "no contrast")):
        ax.scatter([S.d("ncl", p, "probe") for p in P if hasc(p) == k], [S.d("ncl", p, "sc") for p in P if hasc(p) == k],
                   s=16, color=col, marker=mk, label=lab, zorder=3)
    cp = [p for p in ("stem_crop020", "contrast_crop020") if S.has("ncl", p)]
    ax.scatter([S.d("ncl", p, "probe") for p in cp], [S.d("ncl", p, "sc") for p in cp], s=22, marker="^",
               facecolors="none", edgecolors="black", label="crop area 0.2", zorder=3)
    for p, t, off in (("contrast", "contrast", (3, -9)), ("cifar", "natural ref.", (-24, 5)), ("all6", "all six", (-30, 0)),
                      (five("contrast"), "all six $-$ contrast", (3, 3)), ("stem_crop020", "stem, crop 0.2", (3, -9)),
                      ("contrast_crop020", "contrast, crop 0.2", (4, 0))):
        if S.has("ncl", p):
            ax.annotate(t, (S.d("ncl", p, "probe"), S.d("ncl", p, "sc")), xytext=off, textcoords="offset points", fontsize=7)
    ax.axhline(0, color="0.5", lw=0.5); ax.axvline(0, color="0.5", lw=0.5)
    ax.set_xlabel("$\\Delta$ probe balanced accuracy (points)"); ax.set_ylabel("$\\Delta$ SC$^*$ (points)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, handletextpad=0.3, columnspacing=0.8)
    fig.savefig(out / "figs" / "fig_sc_vs_probe.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ purity bars with 95% intervals
    pl = POOL + ["cifar", "all6"]
    fig, ax = plt.subplots(figsize=(3.3, 2.3), constrained_layout=True)
    w = 0.38; xx = np.arange(len(pl))
    for k, (o, nm) in enumerate(OBJS):
        v = [S.d(o, p, "pz") for p in pl]; e = [S.hw(o, p, "pz") for p in pl]
        ax.bar(xx + (k - 0.5) * w, v, w, yerr=e, label=nm, color=[C_NCL, C_SIM][k], hatch=["", "////"][k],
               edgecolor="white" if k == 0 else "#7a4a00", linewidth=0.4, error_kw=dict(lw=0.6, capsize=1.5, ecolor="0.25"))
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(xx); ax.set_xticklabels([SHORT.get(p, {"cifar": "natural ref.", "all6": "all six"}.get(p)) for p in pl], rotation=40, ha="right")
    ax.set_ylabel("$\\Delta$ projector purity$^*$")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.savefig(out / "figs" / "fig_purity.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ agreement between measures across policies
    am = ["probe", "knn", "sc", "pz", "ph", "ms", "edh"]
    R = {}
    for o, _ in OBJS:
        R[o] = np.array([[spearman([S.v(o, p, a_) for p in screen_pols], [S.v(o, p, b_) for p in screen_pols]) for b_ in am] for a_ in am])
        N[f"rho_{o}"] = {f"{am[i]}~{am[j]}": round(float(R[o][i, j]), 3) for i in range(len(am)) for j in range(i + 1, len(am))}
    L = ["\\begin{tabular}{@{}l" + "c" * len(am) + "@{}}", "\\toprule", " & " + " & ".join(MET[k][1] for k in am) + " \\\\", "\\midrule"]
    for i, a_ in enumerate(am):
        c = []
        for j in range(len(am)):
            if i == j:
                c.append("--")
            else:
                o = "ncl" if i > j else "simclr"
                c.append(tex(R[o][i, j], 2))
        L.append(f"{MET[a_][1]} & " + " & ".join(c) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (out / "tables" / "tab_agreement.tex").write_text("\n".join(L) + "\n")
    xs = [p for p in screen_pols[1:]]
    N["rho_ncl_vs_simclr_delta_probe"] = round(spearman([S.d("ncl", p, "probe") for p in xs], [S.d("simclr", p, "probe") for p in xs]), 3)
    N["rho_ncl_vs_simclr_delta_knn"] = round(spearman([S.d("ncl", p, "knn") for p in xs], [S.d("simclr", p, "knn") for p in xs]), 3)
    N["rho_ncl_vs_simclr_delta_pz"] = round(spearman([S.d("ncl", p, "pz") for p in xs], [S.d("simclr", p, "pz") for p in xs]), 3)

    # ------------------------------------------------------------------ full per-policy tables (supplementary)
    order = (["stem"] + POOL + ["cifar"] + [f"{x}_{y}" for x, y in pairs()] + [five(x) for x in POOL] + ["all6"] + STRENGTH)
    names = dict(LABEL)
    names.update({f"{x}_{y}": f"{SHORT[x]} + {SHORT[y]}" for x, y in pairs()})
    names.update({five(x): f"all six $-$ {SHORT[x]}" for x in POOL})
    names.update({"contrast_f060": "contrast, factor 0.6", "contrast_f090": "contrast, factor 0.9", "stem_crop020": "stem, crop 0.2",
                  "stem_crop050": "stem, crop 0.5", "contrast_crop020": "contrast, crop 0.2", "contrast_crop050": "contrast, crop 0.5",
                  "speckle_contrast_crop020": "speckle + contrast, crop 0.2 (A6)"})
    fk = ["probe", "probe_z", "knn", "sc", "pz", "ph", "ms", "edh", "edz"]
    for o, nm in OBJS:
        L = ["\\begin{tabular}{@{}lc" + "c" * len(fk) + "@{}}", "\\toprule", "Policy & $n$ & " + " & ".join(MET[k][1] for k in fk) + " \\\\", "\\midrule"]
        for p in order:
            if not S.has(o, p):
                continue
            if p in ("speckle", "cifar", f"{POOL[0]}_{POOL[1]}", five(POOL[0]), STRENGTH[0]):
                L.append("\\midrule")
            L.append(f"{names[p]} & {S.nn(o, p)} & " + " & ".join(tex(S.v(o, p, k), MET[k][2], sign=False) for k in fk) + " \\\\")
        L += ["\\bottomrule", "\\end{tabular}"]
        (out / "tables" / f"tab_full_{o}.tex").write_text("\n".join(L) + "\n")

    # ------------------------------------------------------------------ readout check: s1 against s2
    s_all = pd.read_csv(a.spec)
    s1 = s_all[s_all.spec_version == "s1"].drop_duplicates("run_id", keep="last").set_index("run_id")
    s2 = s_all[s_all.spec_version == "s2"].drop_duplicates("run_id", keep="last").set_index("run_id")
    com = [r for r in s1.index.intersection(s2.index) if not r.startswith("random-init")]
    for o, _ in OBJS:
        ids = [r for r in com if r.startswith(o + "-")]
        for c in ("purity_proj_star", "ms_star", "sc_star"):
            N[f"s1_vs_s2_{o}_{c}"] = {"rho": round(spearman(s1.loc[ids, c], s2.loc[ids, c]), 3),
                                      "s1_mean": round(float(s1.loc[ids, c].mean()), 4), "s2_mean": round(float(s2.loc[ids, c].mean()), 4)}
        st_ids = [r for r in ids if r.split("-")[3] == "stem"]
        N[f"s1_stem_{o}_purity_proj_star"] = round(float(s1.loc[st_ids, "purity_proj_star"].mean()), 4)

    (out / "results_numbers.json").write_text(json.dumps(N, indent=1, default=str))
    print(f"wrote tables, figures and {out / 'results_numbers.json'}; runs {len(df)}; pending audit: {N['pending_audit']}")


if __name__ == "__main__":
    main()
