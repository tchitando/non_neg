#!/usr/bin/env python3
"""Core results table (OCT, ResNet-50): validation and test probes, k-NN-20, SC % and MS, every seed, then mean and SD.

Reads study/summary/results_workbook.xlsx (sheet '8 Runs (val)') and study/test_results_octmnist.csv, from the
repository root:   python study_tools/make_core_table.py [--out study/summary/TC_core_results.xlsx]
"""
import argparse
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

FONT = "Garamond"
HEAD = PatternFill("solid", start_color="DCEBEC")
BLOCK = PatternFill("solid", start_color="F2F6F7")
MEANROW = PatternFill("solid", start_color="FAFAF7")
AHEAD = PatternFill("solid", start_color="D9EAD3")       # NCL mean above SimCLR mean
THIN = Side(style="thin", color="C9D3D6")
MUTED = "555F66"
C = 4                                                    # OCT classes
SC_CHANCE = 100 / C

GROUPS = [
    ("Crop-only design: crop 0.08, one augmentation added at a time", [
        ("Crop only", "croponly"),
        ("+ brightness 0.8", "croponly_bright"),
        ("+ contrast 0.8", "croponly_contrast"),
        ("+ blur σ max 1.2", "croponly_blur"),
        ("+ flip", "croponly_flip"),
        ("Crop area 0.2", "croponly_crop020"),
    ]),
    ("Old stem: crop 0.08 + brightness 0.8 + flip", [
        ("Stem", "stem"),
        ("Stem + contrast 0.8", "contrast"),
    ]),
]
AUGS = {
    "croponly": "crop 0.08", "croponly_bright": "crop 0.08 · brightness 0.8 (p 0.8)",
    "croponly_contrast": "crop 0.08 · contrast 0.8 (p 0.5)", "croponly_contrast_f050": "crop 0.08 · contrast 0.5 (p 0.5)",
    "croponly_contrast_f030": "crop 0.08 · contrast 0.3 (p 0.5)", "croponly_contrast_f010": "crop 0.08 · contrast 0.1 (p 0.5)",
    "croponly_blur_sig060": "crop 0.08 · blur σ 0.1–0.6 (p 0.5)", "croponly_blur": "crop 0.08 · blur σ 0.1–1.2 (p 0.5)",
    "croponly_blur_sig200": "crop 0.08 · blur σ 0.1–2.0 (p 0.5)", "croponly_flip": "crop 0.08 · flip (p 0.5)",
    "croponly_crop020": "crop 0.2", "croponly_crop050": "crop 0.5",
    "stem": "crop 0.08 · brightness 0.8 (p 0.8) · flip (p 0.5)",
    "contrast_f090": "stem · contrast 0.9 (p 0.5)", "contrast": "stem · contrast 0.8 (p 0.5)",
    "contrast_f060": "stem · contrast 0.6 (p 0.5)",
    "stem_crop020": "crop 0.2 · brightness 0.8 (p 0.8) · flip (p 0.5)",
    "stem_crop050": "crop 0.5 · brightness 0.8 (p 0.8) · flip (p 0.5)",
    "contrast_crop020": "crop 0.2 · brightness 0.8 (p 0.8) · flip (p 0.5) · contrast 0.8 (p 0.5)",
    "contrast_crop050": "crop 0.5 · brightness 0.8 (p 0.8) · flip (p 0.5) · contrast 0.8 (p 0.5)",
}
METRICS = [  # (header, column, number format)
    ("Val probe h", "val_h", "0.00"), ("Val probe z", "val_z", "0.00"),
    ("Test probe h", "test_h", "0.00"), ("Test probe z", "test_z", "0.00"),
    ("Val k-NN-20", "val_knn", "0.00"), ("Test k-NN-20", "test_knn", "0.00"),
    ("Val SC %", "val_sc", "0.0"), ("Test SC %", "test_sc", "0.0"),
    ("Val MS", "val_ms", "0.0000"),
]


def load(workbook, test_csv):
    v = pd.read_excel(workbook, sheet_name="8 Runs (val)")
    v = v[(v.Dataset == "OCT") & (v.Backbone == "ResNet-50")]
    v = v.rename(columns={"Policy id": "policy", "Objective": "method", "Seed": "seed", "Probe backbone": "val_h",
                          "Probe projector": "val_z", "k-NN-20": "val_knn", "SC %": "val_sc", "MS": "val_ms",
                          "MS floor": "ms_floor", "Collapsed": "collapsed"})
    t = pd.read_csv(test_csv)
    t = t[t.backbone.astype(str).str.contains("50")].copy()
    t["test_sc"] = 100 * (1 / C + (t.sc_star / 100) * (1 - 1 / C))      # SC % from SC*
    t = t.rename(columns={"policy_id": "policy", "objective": "method", "probe_bb_bal_acc": "test_h",
                          "probe_proj_bal_acc": "test_z", "knn20_bal_acc": "test_knn"})
    d = v.merge(t[["policy", "method", "seed", "test_h", "test_z", "test_knn", "test_sc"]],
                on=["policy", "method", "seed"], how="outer")
    return d, float(v.ms_floor.dropna().iloc[0])


def cell(ws, r, c, v=None, bold=False, italic=False, fmt=None, fill=None, color=None, size=10, wrap=False):
    x = ws.cell(row=r, column=c, value=v)
    x.font = Font(name=FONT, bold=bold, italic=italic, size=size, color=color)
    if fmt:
        x.number_format = fmt
    if fill:
        x.fill = fill
    if wrap:
        x.alignment = Alignment(wrap_text=True, vertical="top")
    return x


SWEEPS = [  # (family, [(level label, policy id)]); the first entry is the family's base level
    ("Brightness strength (p 0.8), crop-only base", [("off (crop only)", "croponly"), ("0.2", "croponly_bright_b020"),
        ("0.4", "croponly_bright_b040"), ("0.8", "croponly_bright"), ("0.95", "croponly_bright_b095")]),
    ("Contrast factor (p 0.5), crop-only base", [("off (crop only)", "croponly"), ("0.8", "croponly_contrast"),
        ("0.5", "croponly_contrast_f050"), ("0.3", "croponly_contrast_f030"), ("0.1", "croponly_contrast_f010")]),
    ("Blur σ max (p 0.5), crop-only base", [("off (crop only)", "croponly"), ("0.6", "croponly_blur_sig060"),
        ("1.2", "croponly_blur"), ("2.0", "croponly_blur_sig200")]),
    ("Minimum crop area, crop-only base", [("0.08 (crop only)", "croponly"), ("0.2", "croponly_crop020"), ("0.5", "croponly_crop050")]),
    ("Contrast factor (p 0.5), old stem", [("off (stem)", "stem"), ("0.9", "contrast_f090"), ("0.8", "contrast"), ("0.6", "contrast_f060")]),
    ("Minimum crop area, old stem", [("0.08 (stem)", "stem"), ("0.2", "stem_crop020"), ("0.5", "stem_crop050")]),
    ("Minimum crop area, old stem + contrast 0.8", [("0.08 (stem + contrast)", "contrast"), ("0.2", "contrast_crop020"),
        ("0.5", "contrast_crop050")]),
]
AUGS.update({"croponly_bright_b020": "crop 0.08 · brightness 0.2 (p 0.8)", "croponly_bright_b040": "crop 0.08 · brightness 0.4 (p 0.8)",
             "croponly_bright_b095": "crop 0.08 · brightness 0.95 (p 0.8)"})
NM = {"ncl": "NCL", "simclr": "SimCLR"}


def write_policy(ws, r, lab, pid, d, n_cols, means_only=False):
    """Every seed (unless means_only), then the mean and SD rows, for NCL and SimCLR. Returns the next row."""
    means = {}
    for meth in ("ncl", "simclr"):
        x = d[(d.policy == pid) & (d.method == meth)].sort_values("seed")
        if x.empty:
            why = "diverged (non-finite features), no results" if pid == "croponly_crop050" and meth == "ncl" \
                else "trained, not scored yet"
            for j, val in enumerate([lab, AUGS.get(pid, pid), NM[meth], None, why], 1):
                cell(ws, r, j, val, italic=(j == 5), color=MUTED if j == 5 else None)
            r += 1
            continue
        if not means_only:
            for _, row in x.iterrows():
                cell(ws, r, 1, lab); cell(ws, r, 2, AUGS.get(pid, pid), color=MUTED, size=9); cell(ws, r, 3, NM[meth])
                cell(ws, r, 4, int(row.seed))
                for j, (_, col, fmt) in enumerate(METRICS, start=5):
                    val = row[col]
                    cell(ws, r, j, None if pd.isna(val) else float(val), fmt=fmt)
                r += 1
        m, s = x[[c for _, c, _ in METRICS]].mean(), x[[c for _, c, _ in METRICS]].std(ddof=1)
        means[meth] = m
        for tag, ser in (("mean", m), ("SD", s)):
            note = f"{tag} ({len(x)} seeds)" if means_only else tag
            cell(ws, r, 1, lab, fill=MEANROW)
            cell(ws, r, 2, "base level; every seed on the 'Core results' sheet" if (means_only and tag == "mean") else None,
                 fill=MEANROW, color=MUTED, size=9, italic=True)
            cell(ws, r, 3, NM[meth], fill=MEANROW, bold=tag == "mean")
            cell(ws, r, 4, note, fill=MEANROW, bold=tag == "mean", italic=tag == "SD")
            for j, (_, col, fmt) in enumerate(METRICS, start=5):
                val = ser[col]
                cell(ws, r, j, None if pd.isna(val) else float(val), fmt=fmt, fill=MEANROW, bold=tag == "mean", italic=tag == "SD")
            if tag == "mean":
                means[(meth, "row")] = r
            r += 1
    if "ncl" in means and "simclr" in means:
        for j, (_, col, _) in enumerate(METRICS, start=5):
            a_, b_ = means["ncl"][col], means["simclr"][col]
            if pd.notna(a_) and pd.notna(b_) and a_ > b_:
                ws.cell(row=means[("ncl", "row")], column=j).fill = AHEAD
    for j in range(1, n_cols + 1):
        ws.cell(row=r - 1, column=j).border = Border(bottom=THIN)
    return r


def write_sheet(ws, title, heading, groups, d, floor, base_short):
    ws.title = title
    cell(ws, 1, 1, heading, bold=True, size=13)
    extra = (" In each sweep the base level (first row) shows only its mean and SD; its seeds are on the 'Core results' "
             "sheet.") if base_short else ""
    cell(ws, 2, 1, f"Every seed, then the mean and SD over seeds. Balanced accuracy (%) for the linear probes on the backbone "
                   f"(h) and projector (z) and for k-NN (k = 20, on h). SC % chance = {SC_CHANCE:.0f}% (SimCLR's units are "
                   f"never zero, so its SC sits at chance by construction). MS on validation only; chance floor "
                   f"{floor:.4f}. Green: NCL mean above SimCLR mean.{extra}", italic=True, color=MUTED, size=9)
    heads = ["Policy" if not base_short else "Level", "Augmentations", "Method", "Seed"] + [h for h, _, _ in METRICS]
    r = 4
    for j, h in enumerate(heads, start=1):
        cell(ws, r, j, h, bold=True, fill=HEAD, wrap=True, size=10).border = Border(bottom=THIN)
    ws.row_dimensions[r].height = 30
    r += 1
    for gname, pols in groups:
        for j in range(1, len(heads) + 1):
            cell(ws, r, j, fill=BLOCK)
        cell(ws, r, 1, gname, bold=True, fill=BLOCK)
        r += 1
        for i, (lab, pid) in enumerate(pols):
            r = write_policy(ws, r, lab, pid, d, len(heads), means_only=(base_short and i == 0))
        r += 1
    for j, w in enumerate([30, 44, 12, 9] + [10] * len(METRICS), start=1):
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.freeze_panes = ws.cell(row=5, column=5)
    ws.page_setup.orientation = "landscape"
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workbook", default="study/summary/results_workbook.xlsx")
    ap.add_argument("--test", default="study/test_results_octmnist.csv")
    ap.add_argument("--out", default="study/summary/TC_core_results.xlsx")
    a = ap.parse_args()
    d, floor = load(a.workbook, a.test)

    wb = Workbook()
    write_sheet(wb.active, "Core results", "Core results: OCTMNIST, ResNet-50, NCL and SimCLR", GROUPS, d, floor, base_short=False)
    write_sheet(wb.create_sheet(), "Sweeps", "Severity sweeps: OCTMNIST, ResNet-50, NCL and SimCLR, one augmentation varied at a time",
                SWEEPS, d, floor, base_short=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    wb.save(a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
