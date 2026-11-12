#!/usr/bin/env python3
"""Comprehensive core results workbook: a compact summary layer (one row per method and policy, mean and SD, test-split
change against the reference on the same seeds, role of each row) over seed-level sheets (every seed, then mean and SD).

Measures: linear probe on the backbone h and the projector z (validation, test), k-NN-20 on h (validation, test),
SC % (validation, test), SC above shuffle (validation, test) and raw MS (validation).

From the repository root:
    python study_tools/make_core_results.py --out study/summary/TC_core_results.xlsx
Reads study/summary/results_workbook.xlsx ('8 Runs (val)', '8 Failed runs'), study/test_results_octmnist.csv,
study/test_results_pathmnist.csv, and the augmentation descriptions from study_tools/make_results_workbook.py.
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

FONT = "Garamond"
G_HEAD = PatternFill("solid", start_color="D9D9D9")      # header rows
G_SUB = PatternFill("solid", start_color="EDEDED")       # sub-header, block titles
G_REF = PatternFill("solid", start_color="F2F2F2")       # reference rows
G_MAIN = PatternFill("solid", start_color="E4E4E4")      # main-result rows
G_MEAN = PatternFill("solid", start_color="F7F7F7")      # mean / SD rows in seed sheets
AHEAD = PatternFill("solid", start_color="E2EFDA")       # NCL mean above SimCLR mean (accuracy measures only)
THIN = Side(style="thin", color="BFBFBF")
MUTED = "595959"
NCLASS = {"OCT": 4, "PathMNIST": 9}
NM = {"ncl": "NCL", "simclr": "SimCLR"}

MEASURES = [  # (group header, column, number format, accuracy measure?)
    ("Val probe h", "val_h", "0.00", True), ("Test probe h", "test_h", "0.00", True),
    ("Val probe z", "val_z", "0.00", True), ("Test probe z", "test_z", "0.00", True),
    ("Val k-NN-20", "val_knn", "0.00", True), ("Test k-NN-20", "test_knn", "0.00", True),
    ("Val SC %", "val_sc", "0.0", False), ("Test SC %", "test_sc", "0.0", False),
    ("Val SC above shuffle", "val_scab", "0.0", False), ("Test SC above shuffle", "test_scab", "0.0", False),
    ("Val MS (raw)", "val_ms", "0.0000", False), ("Test MS (raw)", "test_ms", "0.0000", False),
]
DELTAS = [("Δ test h", "test_h"), ("Δ test z", "test_z"), ("Δ test SC above shuffle", "test_scab")]

# ---------------------------------------------------------------- what goes where
R50, R18 = ("OCT", "ResNet-50"), ("OCT", "ResNet-18")
PATH = ("PathMNIST", "ResNet-50")
MAIN = [  # (role, label, key, reference key)
    ("Reference", "Crop only", (*R50, "croponly"), None),
    ("Main result", "+ brightness 0.8", (*R50, "croponly_bright"), (*R50, "croponly")),
    ("Intensity", "+ contrast 0.8", (*R50, "croponly_contrast"), (*R50, "croponly")),
    ("Control", "+ blur σ max 1.2", (*R50, "croponly_blur"), (*R50, "croponly")),
    ("Control", "+ flip", (*R50, "croponly_flip"), (*R50, "croponly")),
    ("Departure", "Crop area 0.2", (*R50, "croponly_crop020"), (*R50, "croponly")),
    ("Combination", "Stem = + brightness 0.8 + flip", (*R50, "stem"), (*R50, "croponly")),
    ("Departure", "Stem + contrast 0.8", (*R50, "contrast"), (*R50, "stem")),
]
SWEEPS = [  # (family, [(level, policy id)], reference policy id); first level = base
    ("Brightness strength (p 0.8), crop-only base",
     [("off", "croponly"), ("0.2", "croponly_bright_b020"), ("0.4", "croponly_bright_b040"), ("0.8", "croponly_bright"),
      ("0.95", "croponly_bright_b095")]),
    ("Contrast factor (p 0.5), crop-only base",
     [("off", "croponly"), ("0.8", "croponly_contrast"), ("0.5", "croponly_contrast_f050"), ("0.3", "croponly_contrast_f030"),
      ("0.1", "croponly_contrast_f010")]),
    ("Blur σ max (p 0.5), crop-only base",
     [("off", "croponly"), ("0.6", "croponly_blur_sig060"), ("1.2", "croponly_blur"), ("2.0", "croponly_blur_sig200")]),
    ("Minimum crop area, crop-only base", [("0.08", "croponly"), ("0.2", "croponly_crop020"), ("0.5", "croponly_crop050")]),
    ("Speckle std (p 0.5), crop-only base", [("off", "croponly"), ("0.05", "croponly_speckle_s005"), ("0.1", "croponly_speckle"),
                                             ("0.2", "croponly_speckle_s020")]),
    ("Motion radius, σ (p 0.5), crop-only base", [("off", "croponly"), ("3, 1", "croponly_motion_r3"), ("5, 2", "croponly_motion"),
                                                  ("8, 3", "croponly_motion_r8")]),
    ("Pixelation factor (p 0.5), crop-only base", [("off", "croponly"), ("0.9", "croponly_pixelate_f090"), ("0.8", "croponly_pixelate"),
                                                   ("0.6", "croponly_pixelate_f060")]),
    ("JPEG quality (p 0.5), crop-only base", [("off", "croponly"), ("90", "croponly_jpeg_q90"), ("70", "croponly_jpeg"),
                                              ("40", "croponly_jpeg_q40")]),
    ("Contrast factor (p 0.5), old stem", [("off", "stem"), ("0.9", "contrast_f090"), ("0.8", "contrast"), ("0.6", "contrast_f060")]),
    ("Minimum crop area, old stem", [("0.08", "stem"), ("0.2", "stem_crop020"), ("0.5", "stem_crop050")]),
    ("Minimum crop area, old stem + contrast 0.8", [("0.08", "contrast"), ("0.2", "contrast_crop020"), ("0.5", "contrast_crop050")]),
]
TRAINING = [  # pool families on the crop-only base, 1 seed per level, training on 3 Nov 2026
    ("Speckle std (p 0.5), crop-only base", "0.05 / 0.1 / 0.2"),
    ("Motion radius, σ (p 0.5), crop-only base", "3, 1 / 5, 2 / 8, 3"),
    ("Pixelation factor (p 0.5), crop-only base", "0.9 / 0.8 / 0.6"),
    ("JPEG quality (p 0.5), crop-only base", "90 / 70 / 40"),
]
SUPPORT = [  # (block title, [(role, label, key, reference key)])
    ("OCT · ResNet-50 · other replicated policies", [
        ("Reference", "Stem", (*R50, "stem"), None),
        ("Check", "Natural-image set (SimCLR default recipe)", (*R50, "cifar"), (*R50, "stem")),
        ("Check", "Stem + motion", (*R50, "motion"), (*R50, "stem")),
        ("Architecture", "Crop only, projector output 2048", (*R50, "croponly_proj2048"), (*R50, "croponly")),
    ]),
    ("OCT · ResNet-18 · architecture check", [
        ("Reference", "Stem", (*R18, "stem"), None),
        ("Check", "A6: stem + speckle + contrast, crop 0.2", (*R18, "speckle_contrast_crop020"), (*R18, "stem")),
    ]),
    ("PathMNIST · ResNet-50 · dataset check", [
        ("Reference", "Stem", (*PATH, "stem"), None),
        ("Check", "Stem + contrast", (*PATH, "contrast"), (*PATH, "stem")),
        ("Check", "Stem + rotation + flip", (*PATH, "rotflip"), (*PATH, "stem")),
        ("Check", "Stem + stain", (*PATH, "stain"), (*PATH, "stem")),
        ("Check", "Stem + defocus", (*PATH, "defocus"), (*PATH, "stem")),
        ("Check", "Stem + pixelation", (*PATH, "pixelate"), (*PATH, "stem")),
        ("Check", "Stem + JPEG", (*PATH, "jpeg"), (*PATH, "stem")),
        ("Check", "Natural-image set (SimCLR default recipe)", (*PATH, "cifar"), (*PATH, "stem")),
        ("Screen (1 seed)", "Stem + all six PathMNIST augmentations", (*PATH, "all6"), (*PATH, "stem")),
        ("Screen (1 seed)", "Stem, crop area 0.2", (*PATH, "stem_crop020"), (*PATH, "stem")),
    ]),
]
PENDING_AUGS = {"croponly_bright_b020": "crop 0.08 · brightness 0.2 (p 0.8)", "croponly_bright_b040": "crop 0.08 · brightness 0.4 (p 0.8)",
                "croponly_bright_b095": "crop 0.08 · brightness 0.95 (p 0.8)", "croponly_proj2048": "crop 0.08 · projector output 2048",
                "croponly_flip": "crop 0.08 · flip (p 0.5)", "croponly_crop050": "crop 0.5",
                "croponly_speckle_s005": "crop 0.08 · speckle std 0.05 (p 0.5)", "croponly_speckle": "crop 0.08 · speckle std 0.1 (p 0.5)",
                "croponly_speckle_s020": "crop 0.08 · speckle std 0.2 (p 0.5)",
                "croponly_motion_r3": "crop 0.08 · motion radius 3, σ 1, ±15° (p 0.5)", "croponly_motion": "crop 0.08 · motion radius 5, σ 2, ±15° (p 0.5)",
                "croponly_motion_r8": "crop 0.08 · motion radius 8, σ 3, ±15° (p 0.5)",
                "croponly_pixelate_f090": "crop 0.08 · pixelation 0.9 (p 0.5)", "croponly_pixelate": "crop 0.08 · pixelation 0.8 (p 0.5)",
                "croponly_pixelate_f060": "crop 0.08 · pixelation 0.6 (p 0.5)",
                "croponly_jpeg_q90": "crop 0.08 · JPEG quality 90 (p 0.5)", "croponly_jpeg": "crop 0.08 · JPEG quality 70 (p 0.5)",
                "croponly_jpeg_q40": "crop 0.08 · JPEG quality 40 (p 0.5)"}


# ---------------------------------------------------------------- data
def load(workbook, tests):
    v = pd.read_excel(workbook, sheet_name="8 Runs (val)")
    v = v.rename(columns={"Dataset": "ds", "Backbone": "bb", "Policy id": "pid", "Objective": "method", "Seed": "seed",
                          "Run id": "run_id", "Collapsed": "collapsed", "Probe backbone": "val_h", "Probe projector": "val_z",
                          "k-NN-20": "val_knn", "SC %": "val_sc", "SC above shuffle": "val_scab", "MS": "val_ms",
                          "MS floor": "ms_floor", "MS*": "val_mss"})
    floors = v.groupby("ds").ms_floor.first().to_dict()
    coll = v[v.collapsed.astype(str).str.lower() == "yes"][["ds", "bb", "pid", "method", "seed", "val_h"]]
    v = v[v.collapsed.astype(str).str.lower() != "yes"]
    ts = []
    for f in tests:
        t = pd.read_csv(f)
        t["ds"] = t.dataset.map({"octmnist": "OCT", "pathmnist": "PathMNIST"})
        t["bb"] = t.backbone.astype(str).map({"resnet50": "ResNet-50", "resnet18": "ResNet-18"})
        C = t.ds.map(NCLASS)
        t["test_sc"] = 100 * (1 / C + (t.sc_star / 100) * (1 - 1 / C))
        t["test_scab"] = (1 - 1 / C) * (t.sc_star - t.sc_star_shuf)
        msf = Path(str(f).replace("test_results_", "test_ms_"))          # test MS from eval_test_ms.py, when scored
        if msf.exists():
            ms = pd.read_csv(msf)[["run_id", "ms", "ms_star", "ms_floor"]].rename(
                columns={"ms": "test_ms", "ms_star": "test_mss", "ms_floor": "test_ms_floor"})
            t = t.merge(ms, on="run_id", how="left")
            for dsn, fl in t.dropna(subset=["test_ms_floor"]).groupby("ds").test_ms_floor.first().items():
                floors[f"test_{dsn}"] = fl
        ts.append(t.rename(columns={"policy_id": "pid", "objective": "method", "probe_bb_bal_acc": "test_h",
                                    "probe_proj_bal_acc": "test_z", "knn20_bal_acc": "test_knn"}))
    t = pd.concat(ts)
    keys = ["ds", "bb", "pid", "method", "seed"]
    t = t.merge(coll[keys].assign(_c=1), on=keys, how="left")
    t = t[t._c.isna()]                                                   # collapsed runs are left out of test too
    for c in ("test_ms", "test_mss"):
        if c not in t.columns:
            t[c] = np.nan
    d = v[keys + ["run_id", "val_h", "val_z", "val_knn", "val_sc", "val_scab", "val_ms", "val_mss"]].merge(
        t[keys + ["test_h", "test_z", "test_knn", "test_sc", "test_scab", "test_ms", "test_mss"]], on=keys, how="outer")
    f = pd.read_excel(workbook, sheet_name="8 Failed runs", header=2).dropna(subset=["Run id"])
    f = f.rename(columns={"Dataset": "ds", "Backbone": "bb", "Policy id": "pid", "Method": "method", "Seed": "seed",
                          "Failure": "failure"})
    f["method"] = f.method.str.lower()
    f["kind"] = np.where(f.failure.str.startswith("collapsed"), "collapsed", "diverged")
    return d, f, floors


def runs(d, key, method):
    ds, bb, pid = key
    return d[(d.ds == ds) & (d.bb == bb) & (d.pid == pid) & (d.method == method)].sort_values("seed")


def fails(f, key, method):
    ds, bb, pid = key
    return f[(f.ds == ds) & (f.bb == bb) & (f.pid == pid) & (f.method == method)].sort_values("seed")


def delta(d, key, ref, method, col):
    """Mean over shared seeds of (policy - reference), on runs that are valid in both."""
    if ref is None:
        return None
    a, b = runs(d, key, method).set_index("seed")[col].dropna(), runs(d, ref, method).set_index("seed")[col].dropna()
    s = a.index.intersection(b.index)
    return float((a[s] - b[s]).mean()) if len(s) else None


# ---------------------------------------------------------------- writing helpers
def cell(ws, r, c, v=None, bold=False, italic=False, fmt=None, fill=None, color=None, size=10, wrap=False, center=False):
    x = ws.cell(row=r, column=c, value=v)
    x.font = Font(name=FONT, bold=bold, italic=italic, size=size, color=color)
    if fmt:
        x.number_format = fmt
    if fill:
        x.fill = fill
    if wrap or center:
        x.alignment = Alignment(wrap_text=wrap, vertical="center", horizontal="center" if center else None)
    return x


def fnum(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def legend(ws, r, lines, width):
    for txt in lines:
        cell(ws, r, 1, txt, italic=True, color=MUTED, size=9)
        r += 1
    return r


# ---------------------------------------------------------------- compact (summary) sheets
COMPACT_ID = ["Role", "Policy", "Method", "n valid", "Failed"]
COMPACT_TAIL = ["Reference"] + [h for h, _ in DELTAS] + ["Augmentations applied", "Policy ID", "Seeds"]


def compact_header(ws, r):
    col = 1
    for h in COMPACT_ID:
        cell(ws, r, col, None, fill=G_HEAD); cell(ws, r + 1, col, h, bold=True, fill=G_SUB, wrap=True)
        col += 1
    for h, _, _, _ in MEASURES:
        cell(ws, r, col, h, bold=True, fill=G_HEAD, center=True, wrap=True)
        ws.merge_cells(start_row=r, start_column=col, end_row=r, end_column=col + 1)
        cell(ws, r + 1, col, "mean", bold=True, fill=G_SUB, center=True)
        cell(ws, r + 1, col + 1, "SD", bold=True, fill=G_SUB, center=True)
        col += 2
    cell(ws, r, col, "Change on test vs reference, same seeds", bold=True, fill=G_HEAD, center=True)
    ws.merge_cells(start_row=r, start_column=col, end_row=r, end_column=col + len(DELTAS))
    for h in COMPACT_TAIL[: len(DELTAS) + 1]:
        cell(ws, r + 1, col, h, bold=True, fill=G_SUB, wrap=True, center=True)
        col += 1
    cell(ws, r, col, "Traceability", bold=True, fill=G_HEAD, center=True)
    ws.merge_cells(start_row=r, start_column=col, end_row=r, end_column=col + 2)
    for h in COMPACT_TAIL[len(DELTAS) + 1:]:
        cell(ws, r + 1, col, h, bold=True, fill=G_SUB, wrap=True)
        col += 1
    ws.row_dimensions[r].height = 30
    ws.row_dimensions[r + 1].height = 30
    for c in range(1, col):
        ws.cell(row=r + 1, column=c).border = Border(bottom=THIN)
    return r + 2, col - 1


def compact_rows(ws, r, d, f, augs, role, label, key, ref, ref_label, ncols, pending_text=None, best=None):
    """Two rows (NCL, SimCLR) for one policy. Returns the next row."""
    means = {}
    first = r
    for meth in ("ncl", "simclr"):
        x, fx = runs(d, key, meth), fails(f, key, meth)
        # a method with no runs and no failures at a screened or replicated level is still pending
        mrole = "Pending" if x.empty and fx.empty and role in ("Screen (1 seed)", "Level") else role
        italic = mrole.startswith("Screen")
        rowfill = G_MAIN if mrole == "Main result" else G_REF if mrole in ("Reference", "Base") else None
        bold_row = mrole == "Main result"
        rl = mrole + (f" · best on val h" if best and best.get(meth) else "")
        cell(ws, r, 1, rl, fill=rowfill, bold=bold_row, italic=italic, color=MUTED if italic else None)
        cell(ws, r, 2, label, fill=rowfill, bold=bold_row, italic=italic)
        cell(ws, r, 3, NM[meth], fill=rowfill, bold=bold_row, italic=italic)
        cell(ws, r, 4, int(len(x)), fill=rowfill, italic=italic)
        ftxt = ", ".join(f"s{int(s)} {k}" for s, k in zip(fx.seed, fx.kind)) if len(fx) else 0
        cell(ws, r, 5, ftxt, fill=rowfill, italic=italic, color="9C0006" if len(fx) else None)
        col = 6
        if x.empty:
            why = "all seeds failed" if len(fx) else (pending_text or "no results")
            cell(ws, r, col, why, italic=True, color=MUTED, fill=rowfill)
            for c in range(col + 1, ncols + 1):
                cell(ws, r, c, fill=rowfill)
            cell(ws, r, ncols - 1, key[2], fill=rowfill, color=MUTED, size=9)
            cell(ws, r, ncols - 2, augs.get(key, PENDING_AUGS.get(key[2], "")), fill=rowfill, color=MUTED, size=9)
            r += 1
            continue
        m = {}
        for _, c_, fm, _ in MEASURES:
            vals = x[c_].dropna()
            if meth == "simclr" and c_ in ("val_scab", "test_scab"):
                vals = pd.Series(dtype=float)                          # SimCLR's SC sits at chance by construction
            mu = float(vals.mean()) if len(vals) else None
            sd = float(vals.std(ddof=1)) if len(vals) > 1 else None
            m[c_] = mu
            cell(ws, r, col, mu, fmt=fm, fill=rowfill, bold=bold_row, italic=italic)
            cell(ws, r, col + 1, sd, fmt=fm, fill=rowfill, italic=True, color=MUTED, size=9)
            col += 2
        means[meth] = (r, m)
        cell(ws, r, col, ref_label if ref else "—", fill=rowfill, color=MUTED, size=9)
        col += 1
        for _, c_ in DELTAS:
            dv = None if (meth == "simclr" and c_ == "test_scab") else delta(d, key, ref, meth, c_)
            fm = "+0.0;-0.0;0.0" if c_ == "test_scab" else "+0.00;-0.00;0.00"
            cell(ws, r, col, dv, fmt=fm, fill=rowfill, bold=bold_row, italic=italic)
            col += 1
        cell(ws, r, col, augs.get(key, PENDING_AUGS.get(key[2], "")), fill=rowfill, color=MUTED, size=9)
        cell(ws, r, col + 1, key[2], fill=rowfill, color=MUTED, size=9)
        cell(ws, r, col + 2, ", ".join(str(int(s)) for s in x.seed), fill=rowfill, color=MUTED, size=9)
        r += 1
    if "ncl" in means and "simclr" in means:
        rn, mn = means["ncl"]; _, msim = means["simclr"]
        col = 6
        for _, c_, _, acc in MEASURES:
            if acc and mn.get(c_) is not None and msim.get(c_) is not None and mn[c_] > msim[c_]:
                ws.cell(row=rn, column=col).fill = AHEAD
            col += 2
    for c in range(1, ncols + 1):
        ws.cell(row=r - 1, column=c).border = Border(bottom=THIN)
    return r


def block_title(ws, r, text, ncols):
    for c in range(1, ncols + 1):
        cell(ws, r, c, fill=G_SUB)
    cell(ws, r, 1, text, bold=True, fill=G_SUB)
    return r + 1


def finish(ws, ncols, freeze_row):
    widths = [18, 34, 9, 7, 14]
    for _, c_, _, _ in MEASURES:
        widths += [9, 8] if c_ in ("val_ms", "test_ms") else [8, 7]
    widths += [12, 9, 9, 11, 46, 22, 16]
    for j, w in enumerate(widths[:ncols], start=1):
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.freeze_panes = ws.cell(row=freeze_row, column=4)
    ws.page_setup.orientation = "landscape"
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0


LEGEND = [
    "Each policy has two rows, NCL then SimCLR: means and SDs over valid seeds (SD blank = one seed). Balanced accuracy in %.",
    "Change on test = mean over seeds shared with the reference of (policy − reference), per method. SimCLR's SC above shuffle "
    "and its change are blank: its units are never zero, so its SC sits at chance by construction.",
    "Shading: dark grey row = main result; light grey row = reference; green cell = NCL mean above SimCLR mean (accuracy "
    "columns only); italic = one seed, a screen. Roles and chance levels: see the 'Key' sheet.",
]


def summary_sheet(wb, d, f, augs, floors):
    ws = wb.create_sheet("1 Main results")
    cell(ws, 1, 1, "Main results: OCTMNIST, ResNet-50, crop-only design (one augmentation added to crop 0.08) and the stem",
         bold=True, size=13)
    r = legend(ws, 2, LEGEND + [f"Chance: SC % 25 (OCT); MS floor {floors['OCT']:.4f} on validation, "
                                f"{floors.get('test_OCT', float('nan')):.4f} on test (OCT)."], 30)
    r, ncols = compact_header(ws, r + 1)
    first = r
    r = block_title(ws, r, "OCT · ResNet-50", ncols)
    for role, label, key, ref in MAIN:
        rl = {"croponly": "crop only", "stem": "stem"}.get(ref[2], ref[2]) if ref else None
        r = compact_rows(ws, r, d, f, augs, role, label, key, ref, rl, ncols)
    finish(ws, ncols, first)


def best_levels(d, levels):
    """Per method: the level with the highest validation probe h mean among levels with 2+ valid seeds."""
    out = {}
    for meth in ("ncl", "simclr"):
        cand = []
        for lab, pid in levels:
            x = runs(d, (*R50, pid), meth)
            if len(x) >= 2:
                cand.append((x.val_h.mean(), pid))
        out[meth] = max(cand)[1] if len(cand) >= 2 else None
    return out


def sweeps_sheet(wb, d, f, augs, floors):
    ws = wb.create_sheet("2 Sweeps")
    cell(ws, 1, 1, "Severity sweeps: OCTMNIST, ResNet-50, one augmentation varied at a time", bold=True, size=13)
    r = legend(ws, 2, LEGEND + ["'best on val h' marks, per method, the level with the highest validation probe h mean among "
                                "levels with 2+ seeds (needs 2 such levels). Change on test is against the sweep's base level."], 30)
    r, ncols = compact_header(ws, r + 1)
    first = r
    for fam, levels in SWEEPS:
        r = block_title(ws, r, fam, ncols)
        base = levels[0][1]
        best = best_levels(d, levels)
        for i, (lab, pid) in enumerate(levels):
            key = (*R50, pid)
            n_ncl, n_sim = len(runs(d, key, "ncl")), len(runs(d, key, "simclr"))
            if i == 0:
                role = "Base"
            elif n_ncl == 0 and n_sim == 0 and fails(f, key, "ncl").empty:
                role = "Pending"
            elif max(n_ncl, n_sim) <= 1:
                role = "Screen (1 seed)"
            else:
                role = "Level"
            bmark = {m: best[m] == pid for m in best}
            r = compact_rows(ws, r, d, f, augs, role, lab, key, None if i == 0 else (*R50, base), levels[0][0] + " (base)",
                             ncols, pending_text="not scored yet", best=bmark)
    finish(ws, ncols, first)


def support_sheet(wb, d, f, augs, floors):
    ws = wb.create_sheet("3 Supporting experiments")
    cell(ws, 1, 1, "Supporting experiments: other replicated OCT policies, ResNet-18, PathMNIST", bold=True, size=13)
    r = legend(ws, 2, LEGEND + [f"Chance: SC % 25 (OCT), 11.1 (PathMNIST); MS floor on validation {floors['OCT']:.4f} (OCT), "
                                f"{floors['PathMNIST']:.4f} (PathMNIST), on test {floors.get('test_OCT', float('nan')):.4f} (OCT), "
                                f"{floors.get('test_PathMNIST', float('nan')):.4f} (PathMNIST). Failed = collapsed (probe below 60, left out of "
                                f"the means) or diverged (non-finite features)."], 30)
    r, ncols = compact_header(ws, r + 1)
    first = r
    for title, rows in SUPPORT:
        r = block_title(ws, r, title, ncols)
        for role, label, key, ref in rows:
            r = compact_rows(ws, r, d, f, augs, role, label, key, ref, "stem" if ref and ref[2] == "stem" else
                             ("crop only" if ref else None), ncols, pending_text="trained, not scored yet")
    finish(ws, ncols, first)


def screen_sheet(wb, d, f, augs, floors, used):
    import make_results_workbook as G
    ws = wb.create_sheet("4 Old-stem screen")
    cell(ws, 1, 1, "Old-stem screen: OCTMNIST, ResNet-50, stem plus one to five pool augmentations (mostly one seed)",
         bold=True, size=13)
    r = legend(ws, 2, LEGEND + ["One-seed rows are a screen: they point to policies worth more seeds, they do not establish "
                                "an effect."], 30)
    r, ncols = compact_header(ws, r + 1)
    first = r
    r = block_title(ws, r, "OCT · ResNet-50 · old stem + pool augmentations", ncols)
    r = compact_rows(ws, r, d, f, augs, "Reference", "Stem", (*R50, "stem"), None, None, ncols)
    pids = sorted({p for p in d[(d.ds == "OCT") & (d.bb == "ResNet-50")].pid} - used,
                  key=lambda p: G.policy_info(p, "octmnist")[2])
    for pid in pids:
        lab = G.policy_info(pid, "octmnist")[0]
        if pid == "speckle_contrast_crop020":
            lab = "A6: stem + speckle + contrast, crop 0.2"
        elif not lab.startswith("Stem"):
            lab = "Stem + " + (lab[0].lower() + lab[1:] if lab[1:2].islower() else lab)
        lab = lab.replace("jpeg", "JPEG").replace("jPEG", "JPEG")
        n = max(len(runs(d, (*R50, pid), m)) for m in ("ncl", "simclr"))
        r = compact_rows(ws, r, d, f, augs, "Screen (1 seed)" if n == 1 else "Level", lab, (*R50, pid), (*R50, "stem"),
                         "stem", ncols)
    finish(ws, ncols, first)


# ---------------------------------------------------------------- seed-level sheets
SEED_MEAS = MEASURES


def seed_sheet(wb, name, title, blocks, d, f, augs, floors):
    """blocks: [(block title, [(label, key)])]. Every seed, failed seeds listed, then mean and SD rows."""
    ws = wb.create_sheet(name)
    cell(ws, 1, 1, title, bold=True, size=13)
    cell(ws, 2, 1, "Every seed, then the mean and SD over valid seeds. Failed seeds are listed and left out of the means. "
                   "SimCLR's SC above shuffle is blank (its SC sits at chance by construction). Green: NCL mean above "
                   "SimCLR mean (accuracy columns only).", italic=True, color=MUTED, size=9)
    heads = ["Policy", "Augmentations applied", "Method", "Seed"] + [h for h, _, _, _ in SEED_MEAS]
    r = 4
    for j, h in enumerate(heads, start=1):
        cell(ws, r, j, h, bold=True, fill=G_HEAD, wrap=True).border = Border(bottom=THIN)
    ws.row_dimensions[r].height = 30
    r += 1
    for btitle, items in blocks:
        for j in range(1, len(heads) + 1):
            cell(ws, r, j, fill=G_SUB)
        cell(ws, r, 1, btitle, bold=True, fill=G_SUB)
        r += 1
        for label, key in items:
            mrow = {}
            for meth in ("ncl", "simclr"):
                x, fx = runs(d, key, meth), fails(f, key, meth)
                rows = [(int(s), "run", row) for s, (_, row) in zip(x.seed, x.iterrows())] + \
                       [(int(s), k, None) for s, k in zip(fx.seed, fx.kind)]
                if not rows:
                    cell(ws, r, 1, label); cell(ws, r, 2, augs.get(key, PENDING_AUGS.get(key[2], "")), color=MUTED, size=9)
                    cell(ws, r, 3, NM[meth]); cell(ws, r, 5, "trained, not scored yet", italic=True, color=MUTED)
                    r += 1
                    continue
                for s, kind, row in sorted(rows, key=lambda t: t[0]):
                    cell(ws, r, 1, label); cell(ws, r, 2, augs.get(key, PENDING_AUGS.get(key[2], "")), color=MUTED, size=9)
                    cell(ws, r, 3, NM[meth]); cell(ws, r, 4, s)
                    if row is None:
                        cell(ws, r, 5, f"{kind}: left out of the means", italic=True, color="9C0006")
                    else:
                        for j, (_, c_, fm, _) in enumerate(SEED_MEAS, start=5):
                            v = None if (meth == "simclr" and c_.endswith("scab")) else fnum(row[c_])
                            cell(ws, r, j, v, fmt=fm)
                    r += 1
                if x.empty:
                    continue
                for tag in ("mean", "SD"):
                    cell(ws, r, 1, label, fill=G_MEAN); cell(ws, r, 2, None, fill=G_MEAN)
                    cell(ws, r, 3, NM[meth], fill=G_MEAN, bold=tag == "mean"); cell(ws, r, 4, tag, fill=G_MEAN, bold=tag == "mean")
                    vals = {}
                    for j, (_, c_, fm, _) in enumerate(SEED_MEAS, start=5):
                        s_ = x[c_].dropna()
                        if meth == "simclr" and c_.endswith("scab"):
                            s_ = pd.Series(dtype=float)
                        v = (float(s_.mean()) if len(s_) else None) if tag == "mean" else (float(s_.std(ddof=1)) if len(s_) > 1 else None)
                        vals[c_] = v
                        cell(ws, r, j, v, fmt=fm, fill=G_MEAN, bold=tag == "mean", italic=tag == "SD")
                    if tag == "mean":
                        mrow[meth] = (r, vals)
                    r += 1
            if "ncl" in mrow and "simclr" in mrow:
                rn, a = mrow["ncl"]; _, b = mrow["simclr"]
                for j, (_, c_, _, acc) in enumerate(SEED_MEAS, start=5):
                    if acc and a.get(c_) is not None and b.get(c_) is not None and a[c_] > b[c_]:
                        ws.cell(row=rn, column=j).fill = AHEAD
            for j in range(1, len(heads) + 1):
                ws.cell(row=r - 1, column=j).border = Border(bottom=THIN)
        r += 1
    for j, w in enumerate([30, 44, 9, 7] + [10] * len(SEED_MEAS), start=1):
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.freeze_panes = ws.cell(row=5, column=5)
    ws.page_setup.orientation = "landscape"
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0


def key_sheet(wb, floors):
    ws = wb.active
    ws.title = "Key"
    cell(ws, 1, 1, "Key: how to read this workbook", bold=True, size=13)
    rows = [
        ("Sheets", None),
        ("1 Main results", "The story set, OCT ResNet-50: crop only and one augmentation at a time, plus the stem and stem + contrast. One row per method; means and SDs."),
        ("2 Sweeps", "Each augmentation at several strengths, with the best level on validation marked. Pending levels listed."),
        ("3 Supporting experiments", "Other replicated OCT policies, ResNet-18 and PathMNIST: does the pattern hold elsewhere?"),
        ("4 Old-stem screen", "The remaining OCT ResNet-50 policies, mostly one seed: leads, not results."),
        ("5–7 Seeds - …", "Every seed behind sheets 1–3, then the mean and SD; failed seeds listed."),
        ("Roles (column 'Role')", None),
        ("Reference", "The policy each change is measured against (crop only, the stem, or a sweep's base level)."),
        ("Main result", "Brightness 0.8: the largest change in accuracy and in specialisation for NCL, on both splits."),
        ("Intensity", "Contrast 0.8: a second intensity augmentation (its full strength sweep is on sheet 2)."),
        ("Control", "Blur and flip: augmentations that do not raise accuracy; specialisation stays roughly flat."),
        ("Combination", "The stem = crop 0.08 + brightness 0.8 + flip; also the reference for stem + contrast."),
        ("Departure", "Accuracy and specialisation move apart: crop area 0.2 (backbone accuracy down, test SC above shuffle "
                      "up); stem + contrast (projector accuracy and SC above shuffle up, backbone accuracy flat)."),
        ("Base / Level", "Sweep base level, and a level or policy with 2+ seeds."),
        ("Check", "Supporting policy outside the crop-only design."),
        ("Screen (1 seed)", "One seed: a lead for more seeds, not an established effect (shown in italics)."),
        ("Architecture", "Projector output size 2048 (default 256) on crop only: a separate factor, not an augmentation."),
        ("Pending", "Planned or training, not scored yet."),
        ("Columns", None),
        ("Probe h / z", "Linear probe balanced accuracy (%) on the backbone feature h and the projector output z."),
        ("k-NN-20", "k-nearest-neighbour balanced accuracy (%), k = 20, on h."),
        ("SC %", "Per unit, the share of the most frequent class among images where the unit is non-zero, averaged over "
                 "units, on a class-balanced subset. NCL only in practice: SimCLR's units are never zero, so its SC sits at chance."),
        ("SC above shuffle", "SC minus its value with shuffled labels, on the 0–(1 − 1/classes) scale: removes the chance "
                             "selectivity that sparse units have. Use this column to compare specialisation between policies "
                             "and between splits (raw test SC % runs higher partly because the test subset is smaller)."),
        ("MS (raw)", "Monosemanticity score (DINOv2 activation-weighted similarity) on a fixed class-balanced subset: "
                     "validation 500 images per class; test 250 per class (OCT, all test images) or 339 per class "
                     "(PathMNIST). Each split has its own floor, so compare raw MS within a split."),
        ("Which measure for which comparison", "Within one split (policy against policy, NCL against SimCLR): raw SC % "
                "and raw MS, each with that split's chance level. Between splits, raw values are not directly comparable "
                "because the chance levels differ: use SC above shuffle for SC, and read each MS against its own split's floor."),
        ("Change on test", "Mean over the seeds shared with the reference of (policy − reference), per method, on the test "
                           "split. Blank when the policy and its reference share no valid seed."),
        ("n valid / Failed", "Valid seeds in the means; failed seeds: collapsed (backbone probe below 60) or diverged "
                             "(non-finite features), left out of the means."),
        ("Chance levels", None),
        ("SC %", "25 on OCT (4 classes), 11.1 on PathMNIST (9 classes)."),
        ("MS floor", f"Validation {floors['OCT']:.4f} (OCT), {floors['PathMNIST']:.4f} (PathMNIST); test "
                     f"{floors.get('test_OCT', float('nan')):.4f} (OCT), {floors.get('test_PathMNIST', float('nan')):.4f} (PathMNIST) "
                     "(mean pairwise DINOv2 similarity "
                     "of the subset: a unit with no preference scores this)."),
        ("Shading", None),
        ("Dark grey row", "Main result."),
        ("Light grey row", "Reference or sweep base."),
        ("Green cell", "NCL mean above the SimCLR mean, accuracy columns only (higher, not necessarily beyond noise)."),
        ("Red text", "Failed seeds."),
    ]
    r = 3
    for a, b in rows:
        if b is None:
            r += 1
            cell(ws, r, 1, a, bold=True, fill=G_SUB); cell(ws, r, 2, None, fill=G_SUB)
        else:
            cell(ws, r, 1, a, bold=True); cell(ws, r, 2, b, wrap=True)
        r += 1
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 120


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workbook", default="study/summary/results_workbook.xlsx")
    ap.add_argument("--tests", nargs="+", default=["study/test_results_octmnist.csv", "study/test_results_pathmnist.csv"])
    ap.add_argument("--generator", default="study_tools", help="folder with make_results_workbook.py (augmentation text)")
    ap.add_argument("--out", default="study/summary/TC_core_results.xlsx")
    a = ap.parse_args()
    sys.path.insert(0, a.generator)
    import make_results_workbook as G
    augs = G.read_policy_augs()
    augs.update({(*R50, k): v for k, v in PENDING_AUGS.items() if (*R50, k) not in augs})
    d, f, floors = load(a.workbook, a.tests)
    # runs that stopped before epoch 99 (study/failed_part8.txt): shown as failed seeds, left out of the means
    extra = pd.DataFrame([dict(ds="OCT", bb="ResNet-50", pid=p, method=m, seed=42, failure="did not finish", kind="did not finish")
                          for p, m in [("croponly_motion", "simclr"), ("croponly_motion_r8", "ncl"), ("croponly_motion_r8", "simclr")]])
    f = pd.concat([f, extra], ignore_index=True)

    wb = Workbook()
    key_sheet(wb, floors)
    summary_sheet(wb, d, f, augs, floors)
    sweeps_sheet(wb, d, f, augs, floors)
    support_sheet(wb, d, f, augs, floors)
    used = {k[2] for _, _, k, _ in MAIN} | {p for _, lv in SWEEPS for _, p in lv} | \
           {k[2] for _, rows in SUPPORT for _, _, k, _ in rows if k[:2] == R50}
    screen_sheet(wb, d, f, augs, floors, used)
    seed_sheet(wb, "5 Seeds - main", "Seed-level results behind '1 Main results'",
               [("OCT · ResNet-50", [(lab, k) for _, lab, k, _ in MAIN])], d, f, augs, floors)
    seed_sheet(wb, "6 Seeds - sweeps", "Seed-level results behind '2 Sweeps' (base levels are on '5 Seeds - main')",
               [(fam, [(lab, (*R50, p)) for lab, p in lv[1:]]) for fam, lv in SWEEPS], d, f, augs, floors)
    seed_sheet(wb, "7 Seeds - supporting", "Seed-level results behind '3 Supporting experiments'",
               [(t, [(lab, k) for role, lab, k, _ in rows if not role.startswith("Screen")]) for t, rows in SUPPORT],
               d, f, augs, floors)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    wb.save(a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
