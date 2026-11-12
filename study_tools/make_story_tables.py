#!/usr/bin/env python3
"""Story tables for the 6 Nov 2026 supervisor meeting: the evidence behind each finding, extracted from the results.

    python3 make_story_tables.py --workbook TC_results_5nov_e.xlsx --test-ms <snapshot>/study/test_ms_octmnist.csv \
        --test-ms-path <snapshot>/study/test_ms_pathmnist.csv --out TC_story_tables_5nov.xlsx

Matched seeds: a change "vs crop only" is computed seed by seed (seed 42 minus seed 42, ...) on seeds 42-44, then averaged.
Values are plain numbers (no formulas). Garamond. Green = NCL ahead of SimCLR (gaps), or NCL's change larger than SimCLR's.
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

FONT = "Garamond"
HEAD = PatternFill("solid", fgColor="E7E6E6")
BLOCK = PatternFill("solid", fgColor="F2F2F2")
GREEN = PatternFill("solid", fgColor="C6EFCE")
THIN = Side(style="thin", color="999999")
SEEDS = [42, 43, 44]
CLASSES = {"OCT": 4, "PathMNIST": 9}


def cell(ws, r, c, v=None, bold=False, italic=False, fill=None, fmt=None, wrap=False, size=10, color=None):
    x = ws.cell(row=r, column=c)
    if v is not None:
        x.value = (round(float(v), 4) if isinstance(v, (float, np.floating)) and np.isfinite(v) else
                   (None if isinstance(v, (float, np.floating)) else v))
    x.font = Font(name=FONT, size=size, bold=bold, italic=italic, color=color)
    if fill:
        x.fill = fill
    if fmt:
        x.number_format = fmt
    x.alignment = Alignment(wrap_text=wrap, vertical="top")
    return x


def header(ws, r, heads, widths=None):
    for j, h in enumerate(heads, start=1):
        cell(ws, r, j, h, bold=True, fill=HEAD, wrap=True)
        ws.cell(row=r, column=j).border = Border(bottom=THIN)
    ws.row_dimensions[r].height = 32
    if widths:
        for j, w in enumerate(widths, start=1):
            ws.column_dimensions[get_column_letter(j)].width = w


def title(ws, text, note=None):
    cell(ws, 1, 1, text, bold=True, size=13)
    if note:
        cell(ws, 2, 1, note, italic=True, size=9, color="595959")


def load(a):
    v = pd.read_excel(a.workbook, sheet_name="8 Runs (val)")
    t = pd.read_excel(a.workbook, sheet_name="8 Runs (test)")
    v = v[v.Collapsed == "no"].copy()
    t = t[t["Probe backbone"] >= 60].copy()
    t["test_sc"] = t.apply(lambda r: 100 / CLASSES[r.Dataset] + (1 - 1 / CLASSES[r.Dataset]) * r["SC*"], axis=1)
    ms = pd.concat([pd.read_csv(p)[["run_id", "ms"]] for p in (a.test_ms, a.test_ms_path)])
    t = t.merge(ms.rename(columns={"run_id": "Run id", "ms": "test_ms"}), on="Run id", how="left")
    v = v.rename(columns={"Probe backbone": "val_h", "Probe projector": "val_z", "SC %": "val_sc", "MS": "val_ms", "Purity z": "val_pu"})
    t = t.rename(columns={"Probe backbone": "test_h", "Probe projector": "test_z"})
    keys = ["Dataset", "Backbone", "Policy id", "Objective", "Seed"]
    d = v[keys + ["val_h", "val_z", "val_sc", "val_ms", "val_pu"]].merge(
        t[keys + ["test_h", "test_z", "test_sc", "test_ms"]], on=keys, how="outer")
    return d


def runs(d, ds, pid, obj, bb="ResNet-50"):
    return d[(d.Dataset == ds) & (d.Backbone == bb) & (d["Policy id"] == pid) & (d.Objective == obj)].set_index("Seed")


def deltas(d, ds, base, pid, obj, cols, seeds=SEEDS):
    a, b = runs(d, ds, base, obj), runs(d, ds, pid, obj)
    sh = [s for s in seeds if s in a.index and s in b.index]
    return (b.loc[sh, cols] - a.loc[sh, cols]), sh


def fisher(x, y):
    m = x.notna() & y.notna()
    r, _ = pearsonr(x[m], y[m])
    k = int(m.sum()); z = np.arctanh(r); se = 1 / np.sqrt(k - 3)
    return r, np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se), k


def seed_block(ws, r, label, df, cols, fmts, green_pairs=()):
    """Rows: one per seed, then mean and SD. df: index = seed, columns = cols."""
    first = r
    for s in df.index:
        cell(ws, r, 1, label); cell(ws, r, 2, int(s))
        for j, (c, fm) in enumerate(zip(cols, fmts), start=3):
            cell(ws, r, j, df.loc[s, c], fmt=fm)
        r += 1
    means, sds = df[cols].mean(), df[cols].std(ddof=1)
    for lab, ser, it in (("mean", means, False), ("SD", sds, True)):
        cell(ws, r, 1, label, bold=not it); cell(ws, r, 2, lab, bold=not it, italic=it)
        for j, (c, fm) in enumerate(zip(cols, fmts), start=3):
            cell(ws, r, j, ser[c], fmt=fm, bold=not it, italic=it, color="595959" if it else None)
        if lab == "mean":
            for ncol, scol in green_pairs:
                if pd.notna(means[ncol]) and pd.notna(means[scol]) and means[ncol] > means[scol]:
                    ws.cell(row=r, column=3 + cols.index(ncol)).fill = GREEN
        r += 1
    for c in range(1, len(cols) + 3):
        ws.cell(row=r - 1, column=c).border = Border(bottom=THIN)
    return r, means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workbook", required=True)
    ap.add_argument("--test-ms", required=True)
    ap.add_argument("--test-ms-path", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = load(a)
    oct50 = d[(d.Dataset == "OCT") & (d.Backbone == "ResNet-50")]
    pol = oct50.groupby(["Policy id", "Objective"]).agg(
        n=("Seed", "size"), val_h=("val_h", "mean"), test_h=("test_h", "mean"), val_z=("val_z", "mean"), test_z=("test_z", "mean"),
        val_sc=("val_sc", "mean"), val_ms=("val_ms", "mean"), val_pu=("val_pu", "mean"), test_sc=("test_sc", "mean")).reset_index()
    N = pol[pol.Objective == "ncl"].set_index("Policy id"); S = pol[pol.Objective == "simclr"].set_index("Policy id")
    B = N.join(S, lsuffix="_n", rsuffix="_s", how="inner")
    gap_v, gap_t = B.val_h_n - B.val_h_s, B.test_h_n - B.test_h_s
    r_v = fisher(N.val_sc, N.val_h); r_t = fisher(N.test_sc, N.test_h)          # same split on both axes

    wb = Workbook()
    # ------------------------------------------------------------------ 1 Story
    ws = wb.active; ws.title = "1 Story"
    title(ws, "The story and its evidence (OCT, ResNet-50 unless marked)",
          "Changes vs crop only use matched seeds 42-44. Probe h = backbone, probe z = projector (balanced accuracy, %). "
          "Gap = NCL − SimCLR. Details in sheets 2-6.")
    cell(ws, 3, 1, "Central claim", bold=True)
    cell(ws, 3, 2, "In medical-image contrastive learning, augmentation policy shapes how specialised the learned units are, how accurate "
                   "the representation is, and how the two relate. NCL consistently has higher measured unit specialisation than SimCLR, "
                   "and the augmentation sets how large its accuracy gap to SimCLR is. Photometric augmentations give the clearest "
                   "replicated gains for NCL on OCT; geometric and degradation augmentations have mixed, layer-dependent effects. "
                   "Domain relevance and specialisation alone do not predict whether an augmentation helps.", wrap=True)
    ws.merge_cells(start_row=3, start_column=2, end_row=3, end_column=7)
    ws.row_dimensions[3].height = 75
    heads = ["Finding", "Claim", "Comparison", "Validation", "Test", "Seeds", "Status", "Sheet"]
    header(ws, 5, heads, [16, 42, 34, 22, 22, 14, 18, 8])
    fam = {}
    for pid in ["croponly_bright", "croponly_contrast", "croponly_blur", "croponly_flip", "croponly_crop020"]:
        dn, _ = deltas(d, "OCT", "croponly", pid, "ncl", ["val_h", "test_h", "test_z", "val_sc", "test_sc"])
        ds_, _ = deltas(d, "OCT", "croponly", pid, "simclr", ["val_h", "test_h"])
        fam[pid] = (dn.mean(), ds_.mean())
    g = lambda pid, k, s="n": fam[pid][0 if s == "n" else 1][k]
    gp = {}
    for pid in ["croponly", "croponly_bright", "croponly_bright_b095", "croponly_contrast"]:
        x = runs(d, "OCT", pid, "ncl").loc[SEEDS]; y = runs(d, "OCT", pid, "simclr").loc[SEEDS]
        gp[pid] = ((x.val_h - y.val_h).mean(), (x.test_h - y.test_h).mean())
    st_n, _ = deltas(d, "PathMNIST", "stem", "stain", "ncl", ["val_h", "test_h", "test_z"])
    st_s, _ = deltas(d, "PathMNIST", "stem", "stain", "simclr", ["val_h", "test_h", "test_z"])
    sc_n, _ = deltas(d, "OCT", "stem", "contrast", "ncl", ["test_h", "test_z"])
    f = lambda x: f"{x:+.1f}"
    rows = [
        ("1 Specialisation", "NCL has higher MS than SimCLR in every OCT policy", "NCL vs SimCLR, all 56 policies",
         f"MS {N.val_ms.min():.3f}-{N.val_ms.max():.3f} vs {S.val_ms.min():.3f}; higher in {(B.val_ms_n > B.val_ms_s).sum()}/{len(B)}",
         "", "1-5 per policy", "Supported", "5"),
        ("1 Specialisation", "NCL's SC is high; SimCLR's units are never zero, so SC reads 25% for SimCLR",
         "NCL vs SimCLR, all 56 policies", f"SC {N.val_sc.min():.0f}-{N.val_sc.max():.0f}% vs {S.val_sc.mean():.0f}%",
         "", "1-5 per policy", "Supported (SC cannot compare the methods)", "5"),
        ("1 Accuracy gap", "SimCLR usually keeps a probe h advantage", "NCL − SimCLR, all 56 policies",
         f"{gap_v.mean():+.1f}; NCL ahead in {(gap_v > 0).sum()}/{gap_v.notna().sum()}",
         f"{gap_t.mean():+.1f}; NCL ahead in {(gap_t > 0).sum()}/{gap_t.notna().sum()}", "1-5 per policy", "Supported", "5"),
        ("1 Accuracy gap", "Augmentation sets the size of the gap", "Gap at crop only → brightness 0.95",
         f"{f(gp['croponly'][0])} → {f(gp['croponly_bright_b095'][0])}", f"{f(gp['croponly'][1])} → {f(gp['croponly_bright_b095'][1])}",
         "42-44", "Supported", "2"),
        ("2 Families", "Photometric: brightness gives the largest NCL gain", "Brightness 0.8 vs crop only, NCL (SimCLR)",
         f"{f(g('croponly_bright','val_h'))} ({f(g('croponly_bright','val_h','s'))})",
         f"{f(g('croponly_bright','test_h'))} ({f(g('croponly_bright','test_h','s'))})", "42-44", "Supported", "3"),
        ("2 Families", "Photometric: contrast helps NCL", "Contrast 0.8 vs crop only, NCL (SimCLR)",
         f"{f(g('croponly_contrast','val_h'))} ({f(g('croponly_contrast','val_h','s'))})",
         f"{f(g('croponly_contrast','test_h'))} ({f(g('croponly_contrast','test_h','s'))})", "42-44", "Supported", "3"),
        ("2 Families", "Brightness helps at every strength; levels 0.2-0.95 are within noise of each other",
         "Brightness sweep, NCL", "within noise (TC_results sheet 5)", "", "42-44", "Supported", "—"),
        ("2 Families", "Geometric: raises NCL SC without raising accuracy", "Crop area 0.2 vs crop only, NCL",
         f"h {f(g('croponly_crop020','val_h'))}, SC {f(g('croponly_crop020','val_sc'))}",
         f"h {f(g('croponly_crop020','test_h'))}, SC {f(g('croponly_crop020','test_sc'))}", "42-44", "Supported", "3"),
        ("2 Families", "Geometric: flip costs SimCLR far more than NCL on test", "Flip vs crop only, NCL (SimCLR)",
         f"{f(g('croponly_flip','val_h'))} ({f(g('croponly_flip','val_h','s'))})",
         f"{f(g('croponly_flip','test_h'))} ({f(g('croponly_flip','test_h','s'))})", "42-44", "Supported", "3"),
        ("2 Families", "Degradation: blur leaves the backbone flat and raises the projector", "Blur vs crop only, NCL: probe h / probe z",
         "", f"h {f(g('croponly_blur','test_h'))} / z {f(g('croponly_blur','test_z'))}", "42-44", "Supported", "3, 4"),
        ("2 Families", "Speckle and motion: no accuracy effect beyond noise", "Crop-only levels", "", "", "42 only",
         "Screen (1 seed)", "—"),
        ("2 Families", "Being in the domain pool does not predict benefit", "Contrast (helps) vs speckle (no effect)", "", "", "",
         "Supported for contrast; speckle is a screen", "3"),
        ("2 Families", "Best policy on validation: stem + contrast 0.8, both methods", "Validation probe h, 3+ seeds", "", "", "42-44",
         "Validation choice", "—"),
        ("3 Layer", "Projector gains are often larger than backbone gains", "NCL test z / test h, vs reference", "",
         f"brightness {f(g('croponly_bright','test_z'))} / {f(g('croponly_bright','test_h'))}; blur {f(g('croponly_blur','test_z'))} / "
         f"{f(g('croponly_blur','test_h'))}; stem+contrast {f(sc_n.test_z.mean())} / {f(sc_n.test_h.mean())}", "42-44", "Supported", "4"),
        ("3 Layer", "Same on PathMNIST (stain vs stem)", "NCL test z / test h", "", f"{f(st_n.test_z.mean())} / {f(st_n.test_h.mean())}",
         "43-44 (NCL stem s42 collapsed)", "Supported", "4"),
        ("4 SC and accuracy", "NCL's SC and probe h are associated across policies", f"Correlation over {r_v[3]} NCL policies",
         f"r = {r_v[0]:+.2f} [{r_v[1]:+.2f}, {r_v[2]:+.2f}]", f"r = {r_t[0]:+.2f} [{r_t[1]:+.2f}, {r_t[2]:+.2f}]", "policy means",
         "Supported (association)", "5"),
        ("4 SC and accuracy", "Higher SC alone does not guarantee accuracy", "Crop area 0.2 vs crop only, NCL",
         f"SC {f(g('croponly_crop020','val_sc'))}, h {f(g('croponly_crop020','val_h'))}",
         f"SC {f(g('croponly_crop020','test_sc'))}, h {f(g('croponly_crop020','test_h'))}", "42-44", "Supported (counterexample)", "3"),
        ("Observed", "Augmentation gains are larger on test than on validation", "PathMNIST stain vs stem, NCL (SimCLR)",
         f"{f(st_n.val_h.mean())} ({f(st_s.val_h.mean())})", f"{f(st_n.test_h.mean())} ({f(st_s.test_h.mean())})", "NCL 43-44, SimCLR 42-44",
         "Observed; explanation open", "6"),
        ("Hypothesis", "Crop-only views share the scan's intensity; brightness and contrast jitter remove that cue", "", "", "", "",
         "Hypothesis, not tested", "—"),
        ("Not claimed", "Domain-specific augmentations are universally better; specialisation causes the accuracy changes; higher SC/MS "
         "means clinically meaningful features; test gains show robustness to scanner or hospital shift", "", "", "", "", "Not claimed", "—"),
    ]
    r = 6
    for row in rows:
        for j, v in enumerate(row, start=1):
            cell(ws, r, j, v, wrap=True, bold=(j == 1))
        r += 1
    ws.freeze_panes = "A6"

    # ------------------------------------------------------------------ 2 Gap by policy
    ws = wb.create_sheet("2 Gap by policy")
    title(ws, "NCL − SimCLR probe h gap, per seed, for crop only, brightness and contrast strengths and the stem",
          "OCT, ResNet-50, seeds 42-44. Green = NCL ahead.")
    cols = ["val_h_ncl", "val_h_simclr", "gap_val", "test_h_ncl", "test_h_simclr", "gap_test"]
    header(ws, 4, ["Policy", "Seed", "Val h NCL", "Val h SimCLR", "Gap val", "Test h NCL", "Test h SimCLR", "Gap test"],
           [30, 8] + [12] * 6)
    r = 5
    for lab, pid in [("Crop only", "croponly"), ("Brightness 0.2", "croponly_bright_b020"), ("Brightness 0.4", "croponly_bright_b040"),
                     ("Brightness 0.8", "croponly_bright"), ("Brightness 0.95", "croponly_bright_b095"),
                     ("Contrast 0.8", "croponly_contrast"), ("Contrast 0.5", "croponly_contrast_f050"), ("Contrast 0.3", "croponly_contrast_f030"),
                     ("Contrast 0.1", "croponly_contrast_f010"), ("Stem (crop + brightness + flip)", "stem"), ("Stem + contrast 0.8", "contrast")]:
        x = runs(d, "OCT", pid, "ncl"); y = runs(d, "OCT", pid, "simclr")
        sh = [s for s in SEEDS if s in x.index and s in y.index]
        df = pd.DataFrame({"val_h_ncl": x.loc[sh, "val_h"], "val_h_simclr": y.loc[sh, "val_h"], "test_h_ncl": x.loc[sh, "test_h"],
                           "test_h_simclr": y.loc[sh, "test_h"]})
        df["gap_val"] = df.val_h_ncl - df.val_h_simclr; df["gap_test"] = df.test_h_ncl - df.test_h_simclr
        start = r
        r, _ = seed_block(ws, r, lab, df, cols, ["0.00"] * 6)
        for rr in range(start, r):
            for c in (5, 8):
                v_ = ws.cell(row=rr, column=c).value
                if isinstance(v_, (int, float)) and v_ > 0 and ws.cell(row=rr, column=2).value != "SD":
                    ws.cell(row=rr, column=c).fill = GREEN
    ws.freeze_panes = "C5"

    # ------------------------------------------------------------------ 3 Family effects
    ws = wb.create_sheet("3 Family effects")
    title(ws, "Accuracy, SC and MS per seed (seeds 42-44), NCL and SimCLR, crop only and one augmentation added",
          "Absolute values per seed, then mean and SD. Last row of each block: change vs crop only, matched seed by seed. "
          "Green: NCL above SimCLR (mean row), or NCL's change larger than SimCLR's (change row).")
    cols = ["ncl_val_h", "ncl_test_h", "sim_val_h", "sim_test_h", "ncl_val_z", "ncl_test_z", "sim_test_z", "ncl_val_sc", "ncl_test_sc",
            "ncl_val_ms", "ncl_test_ms"]
    heads = ["Augmentation", "Seed", "NCL val h", "NCL test h", "SimCLR val h", "SimCLR test h", "NCL val z", "NCL test z", "SimCLR test z",
             "NCL val SC %", "NCL test SC %", "NCL val MS", "NCL test MS"]
    header(ws, 4, heads, [26, 8] + [11] * 11)
    fm = ["0.00"] * 7 + ["0.0", "0.0", "0.000", "0.000"]
    r = 5
    def frame(base, pid, absolute=False):
        x = runs(d, "OCT", pid, "ncl"); y = runs(d, "OCT", pid, "simclr")
        if absolute:
            bx, by = 0, 0
        else:
            bx, by = runs(d, "OCT", base, "ncl"), runs(d, "OCT", base, "simclr")
        sh = [s for s in SEEDS if s in x.index and s in y.index]
        sub = lambda a_, b_, c_: a_.loc[sh, c_] - (0 if absolute else b_.loc[sh, c_])
        return pd.DataFrame({"ncl_val_h": sub(x, bx, "val_h"), "ncl_test_h": sub(x, bx, "test_h"), "sim_val_h": sub(y, by, "val_h"),
                             "sim_test_h": sub(y, by, "test_h"), "ncl_val_z": sub(x, bx, "val_z"), "ncl_test_z": sub(x, bx, "test_z"),
                             "sim_test_z": sub(y, by, "test_z"), "ncl_val_sc": sub(x, bx, "val_sc"), "ncl_test_sc": sub(x, bx, "test_sc"),
                             "ncl_val_ms": sub(x, bx, "val_ms"), "ncl_test_ms": sub(x, bx, "test_ms")})
    r, _ = seed_block(ws, r, "Crop only (reference)", frame(None, "croponly", True), cols, fm,
                      green_pairs=[("ncl_val_h", "sim_val_h"), ("ncl_test_h", "sim_test_h"), ("ncl_test_z", "sim_test_z")])
    for lab, pid in [("Brightness 0.8 (photometric)", "croponly_bright"), ("Contrast 0.8 (photometric)", "croponly_contrast"),
                     ("Blur σ ≤ 1.2 (degradation)", "croponly_blur"), ("Flip (geometric)", "croponly_flip"),
                     ("Crop area 0.2 (geometric)", "croponly_crop020")]:
        pairs = [("ncl_val_h", "sim_val_h"), ("ncl_test_h", "sim_test_h"), ("ncl_test_z", "sim_test_z")]
        r, _ = seed_block(ws, r, lab, frame(None, pid, True), cols, fm, green_pairs=pairs)
        ch = frame("croponly", pid).mean()
        cell(ws, r, 1, lab, italic=True); cell(ws, r, 2, "change vs crop only", italic=True)
        for j, (c_, fm_) in enumerate(zip(cols, fm), start=3):
            cell(ws, r, j, ch[c_], fmt=fm_.replace("0.", "+0.", 1) + ";-" + fm_ if fm_ else None, italic=True)
        for nc, sc_ in pairs:
            if pd.notna(ch[nc]) and pd.notna(ch[sc_]) and ch[nc] > ch[sc_]:
                ws.cell(row=r, column=3 + cols.index(nc)).fill = GREEN
        for c_ in range(1, len(cols) + 3):
            ws.cell(row=r, column=c_).border = Border(bottom=THIN)
        r += 1
    ws.freeze_panes = "C5"

    # ------------------------------------------------------------------ 4 Layer
    ws = wb.create_sheet("4 Backbone vs projector")
    title(ws, "Backbone (h) and projector (z) changes on test, per seed",
          "Change vs the reference on the same seed. 'z − h' > 0: the projector gains more than the backbone. "
          "PathMNIST NCL: the stem's seed 42 collapsed, so 2 matched seeds.")
    cols = ["ncl_h", "ncl_z", "ncl_zh", "sim_h", "sim_z", "sim_zh"]
    header(ws, 4, ["Comparison", "Seed", "NCL test h", "NCL test z", "NCL z − h", "SimCLR test h", "SimCLR test z", "SimCLR z − h"],
           [40, 8] + [12] * 6)
    r = 5
    for lab, ds, base, pid in [("OCT: brightness 0.8 vs crop only", "OCT", "croponly", "croponly_bright"),
                               ("OCT: contrast 0.8 vs crop only", "OCT", "croponly", "croponly_contrast"),
                               ("OCT: blur vs crop only", "OCT", "croponly", "croponly_blur"),
                               ("OCT: stem + contrast 0.8 vs stem", "OCT", "stem", "contrast"),
                               ("PathMNIST: stem + stain vs stem", "PathMNIST", "stem", "stain")]:
        dn, shn = deltas(d, ds, base, pid, "ncl", ["test_h", "test_z"])
        dsm, shs = deltas(d, ds, base, pid, "simclr", ["test_h", "test_z"])
        idx = sorted(set(shn) | set(shs))
        df = pd.DataFrame(index=idx)
        df["ncl_h"], df["ncl_z"] = dn.test_h, dn.test_z
        df["sim_h"], df["sim_z"] = dsm.test_h, dsm.test_z
        df["ncl_zh"], df["sim_zh"] = df.ncl_z - df.ncl_h, df.sim_z - df.sim_h
        r, _ = seed_block(ws, r, lab, df, cols, ["0.00"] * 6)
    ws.freeze_panes = "C5"

    # ------------------------------------------------------------------ 5 Per policy
    ws = wb.create_sheet("5 Per policy")
    title(ws, "Every OCT ResNet-50 policy: specialisation and accuracy, NCL and SimCLR (policy means over valid seeds)",
          f"NCL SC vs probe h over {r_v[3]} NCL policies, same split on both axes: validation r = {r_v[0]:+.2f} [{r_v[1]:+.2f}, {r_v[2]:+.2f}], "
          f"test r = {r_t[0]:+.2f} [{r_t[1]:+.2f}, {r_t[2]:+.2f}] (95% CI, Fisher z). Green = NCL ahead.")
    heads = ["Policy id", "Seeds NCL", "Seeds SimCLR", "Val h NCL", "Val h SimCLR", "Gap val", "Test h NCL", "Test h SimCLR", "Gap test",
             "Val SC % NCL", "Val SC % SimCLR", "Val MS NCL", "Val MS SimCLR", "Val purity z NCL", "Val purity z SimCLR"]
    header(ws, 4, heads, [30] + [10] * 14)
    T = B.assign(gapv=gap_v, gapt=gap_t).sort_values("val_h_n", ascending=False)
    r = 5
    for pid, x in T.iterrows():
        vals = [pid, int(x.n_n), int(x.n_s), x.val_h_n, x.val_h_s, x["gapv"], x.test_h_n, x.test_h_s, x["gapt"], x.val_sc_n, x.val_sc_s,
                x.val_ms_n, x.val_ms_s, x.val_pu_n, x.val_pu_s]
        fms = [None, None, None] + ["0.00"] * 6 + ["0.0", "0.0", "0.000", "0.000", "0.000", "0.000"]
        for j, (v_, fm_) in enumerate(zip(vals, fms), start=1):
            cell(ws, r, j, v_, fmt=fm_)
        for c, gval in ((6, x["gapv"]), (9, x["gapt"])):
            if pd.notna(gval) and gval > 0:
                ws.cell(row=r, column=c).fill = GREEN
        r += 1
    ws.freeze_panes = "B5"

    # ------------------------------------------------------------------ 6 Validation vs test
    ws = wb.create_sheet("6 Validation vs test")
    title(ws, "Augmentation gains on validation and on test, per seed",
          "PathMNIST: stem + stain vs stem (matched seeds; NCL stem seed 42 collapsed, so NCL has 2). "
          "OCT rows repeat the brightness and contrast changes from sheet 3.")
    cols = ["n_v", "n_t", "s_v", "s_t"]
    header(ws, 4, ["Comparison", "Seed", "NCL val h", "NCL test h", "SimCLR val h", "SimCLR test h"], [40, 8] + [12] * 4)
    r = 5
    for lab, ds, base, pid in [("PathMNIST: stem + stain vs stem", "PathMNIST", "stem", "stain"),
                               ("OCT: brightness 0.8 vs crop only", "OCT", "croponly", "croponly_bright"),
                               ("OCT: contrast 0.8 vs crop only", "OCT", "croponly", "croponly_contrast")]:
        dn, shn = deltas(d, ds, base, pid, "ncl", ["val_h", "test_h"])
        dsm, shs = deltas(d, ds, base, pid, "simclr", ["val_h", "test_h"])
        df = pd.DataFrame(index=sorted(set(shn) | set(shs)))
        df["n_v"], df["n_t"], df["s_v"], df["s_t"] = dn.val_h, dn.test_h, dsm.val_h, dsm.test_h
        r, _ = seed_block(ws, r, lab, df, cols, ["0.00"] * 4)
    ws.freeze_panes = "C5"

    wb.save(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
