#!/usr/bin/env python3
"""Write the sweep table as a plain workbook: seed values are data, every mean, SD, score and pick is a formula."""
import sys
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils import get_column_letter as L

t = pd.read_csv(sys.argv[1]); out = sys.argv[2]
cols = ["Operation", "Parameter", "Level", "Policy ID", "Method", "n seeds",
        "Val h s42", "Val h s43", "Val h s44", "Val h mean", "Val h SD",
        "Val SC % s42", "Val SC % s43", "Val SC % s44", "Val SC % mean", "Val SC % SD",
        "NCL h mean above SimCLR", "Score 1/2 (NCL + SimCLR h mean)", "Picked for heatmap", "Note"]
C = {c: L(i + 1) for i, c in enumerate(cols)}
wb = Workbook(); ws = wb.active; ws.title = "Sweeps"
ws.append(cols)
TEXT_PICK = {"Flip": "yes (on)", "Pixelation": "not in heatmap: no level with seeds 42-44 for both methods",
             "Motion blur": "not in heatmap: no level with seeds 42-44 for both methods",
             "Crop min area": "not in heatmap: crop is the base operation", "Base (crop only)": ""}
# first and last row of each operation block (for the MAX over its scores)
t["row"] = range(2, len(t) + 2)
blk = t.groupby("Operation", sort=False).row.agg(["min", "max"])
for _, r in t.iterrows():
    i = r.row
    hs = [C[f"Val h s{s}"] + str(i) for s in (42, 43, 44)]
    ss = [C[f"Val SC % s{s}"] + str(i) for s in (42, 43, 44)]
    hr, sr = f"{hs[0]}:{hs[2]}", f"{ss[0]}:{ss[2]}"
    row = {c: r[c] for c in ("Operation", "Parameter", "Level", "Policy ID", "Method")}
    row["Note"] = r["Note"] if isinstance(r["Note"], str) else None
    for s in (42, 43, 44):
        for k in (f"Val h s{s}", f"Val SC % s{s}"):
            row[k] = None if pd.isna(r[k]) else round(float(r[k]), 2)
    row["n seeds"] = f"=COUNT({hr})"
    row["Val h mean"] = f'=IF(COUNT({hr})=0,"",AVERAGE({hr}))'
    row["Val h SD"] = f'=IF(COUNT({hr})<2,"",STDEV({hr}))'
    row["Val SC % mean"] = f'=IF(COUNT({sr})=0,"",AVERAGE({sr}))'
    row["Val SC % SD"] = f'=IF(COUNT({sr})<2,"",STDEV({sr}))'
    ncl = i if r.Method == "NCL" else i - 1                     # rows come in pairs: NCL, then SimCLR
    sim = ncl + 1
    if r.Method == "NCL":
        row["NCL h mean above SimCLR"] = (f'=IF(AND(ISNUMBER({C["Val h mean"]}{ncl}),ISNUMBER({C["Val h mean"]}{sim})),'
                                          f'IF({C["Val h mean"]}{ncl}>{C["Val h mean"]}{sim},"yes","no"),"")')
    op = r.Operation
    if op != "Base (crop only)":
        row["Score 1/2 (NCL + SimCLR h mean)"] = (
            f'=IF(AND(COUNT({C["Val h s42"]}{ncl}:{C["Val h s44"]}{ncl})=3,COUNT({C["Val h s42"]}{sim}:{C["Val h s44"]}{sim})=3),'
            f'({C["Val h mean"]}{ncl}+{C["Val h mean"]}{sim})/2,"")')
    if op in TEXT_PICK:
        row["Picked for heatmap"] = TEXT_PICK[op] or None
    else:
        sc = C["Score 1/2 (NCL + SimCLR h mean)"]; a, b = blk.loc[op, "min"], blk.loc[op, "max"]
        row["Picked for heatmap"] = (f'=IF({sc}{i}="","not eligible: needs seeds 42-44 for both methods",'
                                     f'IF({sc}{i}=MAX({sc}{a}:{sc}{b}),"yes","no"))')
    ws.append([row.get(c) for c in cols])
for c in ("Val h mean", "Val h SD", "Val SC % mean", "Val SC % SD"):
    for cell in ws[C[c]][1:]:
        cell.number_format = "0.00"
for cell in ws[C["Score 1/2 (NCL + SimCLR h mean)"]][1:]:
    cell.number_format = "0.000"

r = wb.create_sheet("Rule")
for line in [
    "Severity sweeps on the crop-only base (crop min area 0.08): OCTMNIST, ResNet-50, 100 epochs, one operation varied at a time.",
    "Val h = linear probe on the backbone features h, balanced accuracy (%), validation split. Val SC % = raw SC of the projector units on class-balanced validation subsets (5 draws); SimCLR sits at 25 by construction.",
    "Seeds 42-44 only. Base and contrast 0.5 also have seeds 45-46; they are left out here.",
    "Selection rule for the heatmap: among levels with seeds 42-44 for both methods, pick the level with the highest 1/2 (NCL mean + SimCLR mean) of Val h. One shared level for both methods; no tie-break.",
    "Mean, SD (sample SD), n seeds, the score and the pick are formulas over the seed cells.",
    "Source: study/results_master.csv as of 11 Nov 2026 (eval_run.py, validation split). Test numbers are not in this table.",
]:
    r.append([line])
wb.save(out)
print("wrote", out)
